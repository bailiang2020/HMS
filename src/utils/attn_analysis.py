from __future__ import annotations
import os
import sys
import importlib
from collections import defaultdict

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor

import matplotlib

matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from data.utils.vlm_datasets import VLMDataset
from src.utils.collate import CollateFn
from src.utils.utils import *
import csv
import numpy as np
from peft import PeftModel

# --- LMDB dependencies for streaming and loading ---
import lmdb
import pickle
import io

MIN_PIXELS = 0 * 32 * 32
MAX_PIXELS = 256 * 32 * 32


# ---- picklable collate to replace lambda (required for num_workers>0 with spawn) ----
def first_item_collate(batch):
    # DataLoader(dataset, batch_size=1) + our Dataset returns objects directly; we just unwrap the single item
    return batch[0]


def parse_topk_heatmap_file(topk_file: str):
    """解析由 _print_topk_from_heatmap 导出的 topk 坐标文件。

    支持行格式：
      layer=10, head=3, value=0.123456

    返回：
      items: list[dict]，每项含 {layer, head, value, rank}
      coord_to_item: dict[(layer, head)] -> item
    """
    items = []
    with open(topk_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            if not line.startswith("layer="):
                continue
            # e.g. layer=1, head=2, value=0.1234
            parts = [p.strip() for p in line.split(",")]
            kv = {}
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip()
            if "layer" in kv and "head" in kv and "value" in kv:
                item = {
                    "layer": int(kv["layer"]),
                    "head": int(kv["head"]),
                    "value": float(kv["value"]),
                }
                items.append(item)

    # 默认文件内已经是按 value 降序；这里再稳妥排序一次
    items.sort(key=lambda x: x["value"], reverse=True)
    for i, it in enumerate(items, start=1):
        it["rank"] = i

    coord_to_item = {(it["layer"], it["head"]): it for it in items}
    return items, coord_to_item


def plot_topk_value_distribution(
    topk_file: str,
    out_path: str | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """绘制单个 topk 文件的值分布图（论文友好风格）。

    横轴：按 value 降序后的头索引（间隔为 1）
    纵轴：score/value
    """
    items, _ = parse_topk_heatmap_file(topk_file)
    if len(items) == 0:
        raise ValueError(f"empty topk file: {topk_file}")

    xs = np.arange(len(items))
    ys = np.array([it["value"] for it in items], dtype=np.float64)

    if out_path is None:
        out_path = os.path.splitext(topk_file)[0] + "_value_distribution.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    # 柱状+折线，接近你给的示例风格
    ax.bar(xs, ys, width=0.9, color="#f6c453", alpha=0.35, edgecolor="none")
    ax.plot(xs, ys, color="#e69f00", linewidth=1.4)

    ax.set_xlabel("Ranked Layer-Head Id")
    ax.set_ylabel("Attention Share")
    ax.set_title(title or "Attention Heads Ranked by Attention Share")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(-1, len(items))

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_topk_value_distribution_compare(
    topk_file_a: str,
    topk_file_b: str,
    out_path: str | None = None,
    label_a: str = "A",
    label_b: str = "B",
    title: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """绘制两个 topk 文件的值分布对比图。

    横轴：按各自 value 降序后的头索引（间隔为 1）
    纵轴：score/value
    """
    items_a, _ = parse_topk_heatmap_file(topk_file_a)
    items_b, _ = parse_topk_heatmap_file(topk_file_b)
    if len(items_a) == 0 or len(items_b) == 0:
        raise ValueError("empty topk file in compare")

    if out_path is None:
        base_dir = os.path.dirname(topk_file_b) or "."
        out_path = os.path.join(base_dir, "topk_value_distribution_compare.png")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    xa = np.arange(1, len(items_a) + 1)
    ya = np.array([it["value"] for it in items_a], dtype=np.float64)
    xb = np.arange(1, len(items_b) + 1)
    yb = np.array([it["value"] for it in items_b], dtype=np.float64)

    plt.figure(figsize=(8, 4))
    plt.plot(xa, ya, linewidth=2.0, label=label_a)
    plt.plot(xb, yb, linewidth=2.0, label=label_b)
    plt.xlabel("Head index (sorted by value)")
    plt.ylabel("Score")
    plt.title(title or "TopK Value Distribution Compare")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_topk_overlap_profile_three(
    topk_file_base: str,
    topk_file_multi: str,
    topk_file_unimodal: str,
    out_path: str | None = None,
    label_base: str = "Base",
    label_multi: str = "Multi-SFT",
    label_unimodal: str = "Unimodal-SFT",
    title: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    # 可选：在基图上增加一个对比曲线（例如新的 text/img 模型）
    topk_file_extra: str | None = None,
    label_extra: str = "Extra",
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 12,
    save_dpi: int = 400,
):
    """三模型分布重叠图（论文友好），支持额外对比曲线。

    - 横轴：排序后的头序号（等间隔 1,2,...,k）
    - 纵轴：attention share
    - 曲线使用浅色 fill_between 填充，便于显示包含/重叠关系
    """
    items_b, _ = parse_topk_heatmap_file(topk_file_base)
    items_m, _ = parse_topk_heatmap_file(topk_file_multi)
    items_u, _ = parse_topk_heatmap_file(topk_file_unimodal)

    items_e = None
    if topk_file_extra is not None:
        items_e, _ = parse_topk_heatmap_file(topk_file_extra)

    lens = [len(items_b), len(items_m), len(items_u)]
    if items_e is not None:
        lens.append(len(items_e))
    k = min(lens)
    if k <= 0:
        raise ValueError("empty topk file in compare")

    x = np.arange(1, k + 1)
    y_base = np.array([it["value"] for it in items_b[:k]], dtype=np.float64)
    y_multi = np.array([it["value"] for it in items_m[:k]], dtype=np.float64)
    y_uni = np.array([it["value"] for it in items_u[:k]], dtype=np.float64)
    y_extra = np.array([it["value"] for it in items_e[:k]], dtype=np.float64) if items_e is not None else None

    # 仍返回集合重叠统计，便于在文中同时报告数字
    set_b = set((it["layer"], it["head"]) for it in items_b[:k])
    set_m = set((it["layer"], it["head"]) for it in items_m[:k])
    set_u = set((it["layer"], it["head"]) for it in items_u[:k])
    overlap_bm = len(set_b & set_m) / k
    overlap_bu = len(set_b & set_u) / k
    overlap_mu = len(set_m & set_u) / k
    overlap_all = len(set_b & set_m & set_u) / k

    overlap_be = None
    overlap_me = None
    overlap_ue = None
    if items_e is not None:
        set_e = set((it["layer"], it["head"]) for it in items_e[:k])
        overlap_be = len(set_b & set_e) / k
        overlap_me = len(set_m & set_e) / k
        overlap_ue = len(set_u & set_e) / k

    if out_path is None:
        base_dir = os.path.dirname(topk_file_unimodal) or "."
        out_path = os.path.join(base_dir, "topk_overlap_profile_three.png")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4.6))

    # 浅色实心 + 清晰边线
    ax.fill_between(x, 0, y_base, color="#4C78A8", alpha=0.16, linewidth=0)
    ax.fill_between(x, 0, y_multi, color="#F58518", alpha=0.16, linewidth=0)
    ax.fill_between(x, 0, y_uni, color="#54A24B", alpha=0.16, linewidth=0)
    if y_extra is not None:
        ax.fill_between(x, 0, y_extra, color="#B279A2", alpha=0.14, linewidth=0)

    ax.plot(x, y_base, color="#4C78A8", linewidth=1.8, label=label_base)
    ax.plot(x, y_multi, color="#F58518", linewidth=1.8, label=label_multi)
    ax.plot(x, y_uni, color="#54A24B", linewidth=1.8, label=label_unimodal)
    if y_extra is not None:
        ax.plot(x, y_extra, color="#B279A2", linewidth=1.8, label=label_extra)

    ax.set_xlabel("Ranked Head Id", fontsize=label_fontsize)
    ax.set_ylabel("Attention Share", fontsize=label_fontsize)
    ax.set_title(title or "Top-k Attention Share Distribution (Overlay)", fontsize=title_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(1, k)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(
        frameon=False,
        ncol=4 if y_extra is not None else 3,
        loc="upper right",
        fontsize=legend_fontsize,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=save_dpi)
    plt.close(fig)
    return {
        "out_path": out_path,
        "k": k,
        "overlap_base_multi": overlap_bm,
        "overlap_base_unimodal": overlap_bu,
        "overlap_multi_unimodal": overlap_mu,
        "overlap_all_three": overlap_all,
        "overlap_base_extra": overlap_be,
        "overlap_multi_extra": overlap_me,
        "overlap_unimodal_extra": overlap_ue,
    }


def compare_topk_heatmap_files(topk_file_a: str, topk_file_b: str):
    """比较两个 topk heatmap 文件的分布差异（训练前 vs 训练后）。

    输出指标包括：
    1) 重合度：overlap、jaccard、overlap@k
    2) 重合头值变化：绝对变化与相对变化
    3) 重合头在两侧 topk 排名中的位置变化（rank shift）
    """
    items_a, map_a = parse_topk_heatmap_file(topk_file_a)
    items_b, map_b = parse_topk_heatmap_file(topk_file_b)

    set_a = set(map_a.keys())
    set_b = set(map_b.keys())
    inter = set_a & set_b
    union = set_a | set_b

    k_a = len(items_a)
    k_b = len(items_b)
    k_ref = min(k_a, k_b) if min(k_a, k_b) > 0 else 1

    overlap_cnt = len(inter)
    overlap_rate_a = overlap_cnt / max(1, k_a)
    overlap_rate_b = overlap_cnt / max(1, k_b)
    jaccard = overlap_cnt / max(1, len(union))
    overlap_at_k = overlap_cnt / k_ref

    value_deltas = []
    rel_deltas = []
    rank_shifts = []
    overlap_rows = []

    for coord in sorted(inter):
        a = map_a[coord]
        b = map_b[coord]
        dv = b["value"] - a["value"]
        rv = dv / max(abs(a["value"]), 1e-12)
        dr = b["rank"] - a["rank"]

        value_deltas.append(dv)
        rel_deltas.append(rv)
        rank_shifts.append(dr)
        overlap_rows.append({
            "layer": coord[0],
            "head": coord[1],
            "value_a": a["value"],
            "value_b": b["value"],
            "delta": dv,
            "rel_delta": rv,
            "rank_a": a["rank"],
            "rank_b": b["rank"],
            "rank_shift": dr,
        })

    def _stats(xs):
        if len(xs) == 0:
            return {"mean": None, "median": None, "std": None, "min": None, "max": None}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "file_a": topk_file_a,
        "file_b": topk_file_b,
        "k_a": k_a,
        "k_b": k_b,
        "overlap_count": overlap_cnt,
        "overlap_rate_a": overlap_rate_a,
        "overlap_rate_b": overlap_rate_b,
        "overlap_at_k": overlap_at_k,
        "jaccard": jaccard,
        "value_delta_stats": _stats(value_deltas),
        "rel_delta_stats": _stats(rel_deltas),
        "rank_shift_stats": _stats(rank_shifts),
        "overlap_rows": overlap_rows,
    }

    return summary


def print_topk_compare_summary(summary: dict):
    """打印 compare_topk_heatmap_files 的摘要结果。"""
    print("=" * 88)
    print("TopK Heatmap 对比摘要")
    print(f"A: {summary['file_a']}")
    print(f"B: {summary['file_b']}")
    print(f"k(A)={summary['k_a']}  k(B)={summary['k_b']}")
    print(f"overlap={summary['overlap_count']} | overlap@k={summary['overlap_at_k']:.4f} | jaccard={summary['jaccard']:.4f}")
    print(f"overlap_rate_a={summary['overlap_rate_a']:.4f} | overlap_rate_b={summary['overlap_rate_b']:.4f}")

    vd = summary["value_delta_stats"]
    rd = summary["rel_delta_stats"]
    rk = summary["rank_shift_stats"]

    print("- value delta (B-A):", vd)
    print("- relative delta:", rd)
    print("- rank shift (rank_B-rank_A):", rk)

    rows = summary.get("overlap_rows", [])
    rows_sorted = sorted(rows, key=lambda x: abs(x["delta"]), reverse=True)
    print("- 重合头中 |delta| 最大的前10个:")
    for r in rows_sorted[:10]:
        print(
            f"  (L{r['layer']:02d},H{r['head']:02d}) "
            f"vA={r['value_a']:.6f} vB={r['value_b']:.6f} "
            f"d={r['delta']:+.6f} rel={r['rel_delta']:+.4f} "
            f"rank {r['rank_a']}->{r['rank_b']}"
        )
    print("=" * 88)


def sweep_k_overlap_from_topk_files(
    topk_file_a: str,
    topk_file_b: str,
    k_values: list[int] | None = None,
):
    """基于两个 topK 文件（如 top200）扫描不同 k，并给出推荐 k。

    评分标准（按你的要求）：
    score(k) = sum_{h in (A_k ∩ B_k)} value_B(h) / k

    其中：
    - A_k/B_k 是两侧前 k 个头的集合
    - value_B(h) 是头 h 在文件 B 中的分数
    """
    items_a, map_a = parse_topk_heatmap_file(topk_file_a)
    items_b, map_b = parse_topk_heatmap_file(topk_file_b)

    coords_a = [(x["layer"], x["head"]) for x in items_a]
    coords_b = [(x["layer"], x["head"]) for x in items_b]
    max_k = min(len(coords_a), len(coords_b))

    if max_k <= 0:
        raise ValueError("empty topk files")

    if k_values is None:
        # 默认扫 5 到 max_k，步长 5
        k_values = list(range(5, max_k + 1, 5))
        if max_k not in k_values:
            k_values.append(max_k)
    else:
        # 清洗并裁剪到合法范围
        k_values = sorted(set(int(k) for k in k_values if 1 <= int(k) <= max_k))
        if len(k_values) == 0:
            raise ValueError(f"No valid k in range [1, {max_k}]")

    rows = []
    for k in k_values:
        A_k = set(coords_a[:k])
        B_k = set(coords_b[:k])
        inter_coords = A_k & B_k
        inter = len(inter_coords)
        union = len(A_k | B_k)

        overlap_at_k = inter / k
        jaccard_at_k = inter / max(1, union)

        # 按你的定义：score = sum(value_B on overlap) / k
        sum_value_b_on_overlap = float(sum(map_b[c]["value"] for c in inter_coords))
        score = sum_value_b_on_overlap / k

        rows.append({
            "k": k,
            "overlap_count": inter,
            "overlap_at_k": overlap_at_k,
            "jaccard_at_k": jaccard_at_k,
            "sum_value_b_on_overlap": sum_value_b_on_overlap,
            "score": score,
        })

    best_by_score = max(rows, key=lambda r: r["score"])
    best_by_overlap = max(rows, key=lambda r: r["overlap_at_k"])
    best_by_jaccard = max(rows, key=lambda r: r["jaccard_at_k"])

    return {
        "file_a": topk_file_a,
        "file_b": topk_file_b,
        "max_k": max_k,
        "rows": rows,
        "best_by_score": best_by_score,
        "best_by_overlap": best_by_overlap,
        "best_by_jaccard": best_by_jaccard,
    }


def print_k_sweep_report(report: dict, topn: int = 10):
    """打印 sweep_k_overlap_from_topk_files 的结果。"""
    print("=" * 88)
    print("TopK K-Sweep 报告（score = sum(value_B on overlap)/k）")
    print(f"A: {report['file_a']}")
    print(f"B: {report['file_b']}")
    print(f"max_k={report['max_k']}")

    print("- best_by_score:", report["best_by_score"])
    print("- best_by_overlap:", report["best_by_overlap"])
    print("- best_by_jaccard:", report["best_by_jaccard"])

    rows = sorted(report["rows"], key=lambda r: r["score"], reverse=True)
    print(f"\nTop {topn} ks by score:")
    for r in rows[:topn]:
        print(
            f"  k={r['k']:>3d} | overlap@k={r['overlap_at_k']:.4f} | "
            f"jaccard={r['jaccard_at_k']:.4f} | "
            f"sumB(overlap)={r['sum_value_b_on_overlap']:.4f} | score={r['score']:.4f}"
        )
    print("=" * 88)


def _resolve_qwen_vl_model_class(model_type: str):
    """Return the class object to load based on model_type.
    Supports the base Qwen2_5_VL and any number of custom versions
    following the naming scheme: Qwen2_5_VL_Custom_<suffix> (e.g., v1, v2, v3...).
    """
    if model_type == "Qwen3_VL":
        from transformers import Qwen3VLForConditionalGeneration as _Cls
        return _Cls

    prefix = "Qwen3_VL_Custom_"
    if model_type.startswith(prefix):
        suffix = model_type[len(prefix):]  # e.g., "v1", "v2", ...
        module_name = f"src.models.qwen3_vl_custom.modeling_qwen3_vl_{suffix}"
        cls_name = "Qwen3VLForConditionalGeneration_Custom"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            raise ImportError(f"Failed to import module '{module_name}' for model_type='{model_type}': {e}")
        try:
            return getattr(mod, cls_name)
        except AttributeError:
            raise ImportError(
                f"Class '{cls_name}' not found in module '{module_name}'. "
                f"Ensure your custom implementation defines this class."
            )

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_gen_dict(test_dataloader, total_batch_num, args, lmdb_path: str | None = None, write_to_lmdb: bool = False):
    ModelCls = _resolve_qwen_vl_model_class(args.model_type)
    model = ModelCls.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if args.lora and args.lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            is_trainable=False
        )

    model.cuda()
    model.eval()

    # 如果需要，打开 LMDB 持久化（实时写入）
    env = None
    if write_to_lmdb and lmdb_path is not None:
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        env = lmdb.open(
            lmdb_path,
            map_size=250 * 1024 * 1024 * 1024,
            subdir=False,
            lock=True,
            readahead=True,
            writemap=False,
            map_async=True,  # 异步刷盘，提交更快，最终可用 env.sync() 刷到磁盘
            metasync=False,  # 放宽 meta 区同步（更快）
            sync=False  # 事务提交不强制 fsync（更快；进程结束或手动 env.sync() 刷盘）
        )
        # 记录一些元信息
        with env.begin(write=True) as txn:
            meta = {"model_type": args.model_type, "model_path": args.model_path}
            txn.put(b"__meta__", pickle.dumps(meta, protocol=4))
        # ------ 本地批量缓冲：合并写入，显著减少 txn 次数 ------
        FLUSH_EVERY = 512  # 每累积 256 条写一次（可按需调大 512/1024）
        FLUSH_BYTES = 20 * 1024 ** 3  # 或累计到 20GB 左右再写（上限保护，避免内存暴涨）
        _batch_pairs: list[tuple[bytes, bytes]] = []
        _batch_bytes = 0

        def _flush_batch():
            nonlocal _batch_pairs, _batch_bytes
            if not _batch_pairs:
                return
            # 单个事务批量写入；允许覆盖同键，去掉 append 约束
            with env.begin(write=True, buffers=True) as txn:
                for k, v in tqdm(
                        _batch_pairs, total=len(_batch_pairs), desc="LMDB flush", dynamic_ncols=True,
                        leave=False
                        ):
                    txn.put(k, v, overwrite=True)
            _batch_pairs = []
            _batch_bytes = 0

    with torch.inference_mode():
        pbar = tqdm(
            test_dataloader,
            total=min(total_batch_num, len(test_dataloader)),
            desc=f"Evaluating",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.2,
            leave=True,
            file=sys.stdout,
        )
        for bidx, (inputs, *_) in enumerate(pbar):
            if bidx >= total_batch_num:
                break
            inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            # print(inputs.keys())
            # 兼容两种输入结构：
            # 1) 旧结构：{"encodings": {...}}
            # 2) 当前结构：直接是 encodings dict（来自 CollateFn）
            if isinstance(inputs, dict) and isinstance(inputs.get("encodings", None), dict):
                inputs["encodings"].pop("ppb_image_mask", None)
                inputs["encodings"].pop("ppb_news_text_mask", None)
            else:
                inputs.pop("ppb_image_mask", None)
                inputs.pop("ppb_news_text_mask", None)

            generated_dict = model.generate(
                **inputs, use_cache=True, output_attentions=True,
                return_dict_in_generate=True, max_new_tokens=1
                )
            # 1. 获取第一步生成的所有层 Attention (原始数据，显存占用大)
            # generated_dict["attentions"] 结构通常是 (step_0_layers, step_1_layers, ...)
            # 我们只取 step_0 (即 Prompt 处理阶段)
            raw_layers_attn = generated_dict["attentions"][0]

            # 2. 创建一个新列表来存“瘦身”后的数据
            processed_layers = []

            for layer_attn in raw_layers_attn:
                # layer_attn shape: (Batch=1, Heads, Q_len, K_len)

                # 取出第一个 Batch -> (Heads, Q_len, K_len)
                A = layer_attn[0]

                # 提取 Last Query (最后一行)
                # 逻辑：无论 Q 是多少 (Prompt阶段 Q=SeqLen, Decode阶段 Q=1)，-1 总是指向最新的那个 Token
                last_q = A[:, -1, :]  # Shape: (Heads, K_len)

                # 转换为 float64 (为了后续累加精度，同时切断计算图节省显存)
                # .detach() 也是个好习惯，确保不带梯度
                last_q = last_q.to(dtype=torch.float64).detach()

                # 存入新列表
                processed_layers.append(last_q)

            # 3. 回写到 generated_dict
            # 为了保持结构一致性（让后续代码不用大改），我们把它包回一个 tuple
            # 现在的结构变成了：((Layer0_LastQ, Layer1_LastQ, ...), )
            generated_dict["attentions"] = (tuple(processed_layers),)

            # 为了可序列化，统一搬到 CPU（不改变数据结构）
            cpu_generated = {}
            for k, v in generated_dict.items():
                if torch.is_tensor(v):
                    cpu_generated[k] = v.to("cpu")
                elif isinstance(v, (list, tuple)):
                    # 递归把张量挪到CPU
                    def _to_cpu(x):
                        if torch.is_tensor(x):
                            return x.to("cpu")
                        elif isinstance(x, (list, tuple)):
                            return type(x)(_to_cpu(xx) for xx in x)
                        elif isinstance(x, dict):
                            return {kk: _to_cpu(vv) for kk, vv in x.items()}
                        else:
                            return x

                    cpu_generated[k] = _to_cpu(v)
                elif isinstance(v, dict):
                    cpu_generated[k] = {kk: (vv.to("cpu") if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                else:
                    cpu_generated[k] = v

            # 实时写入 LMDB（批量事务，允许覆盖同键 bidx）；bs=1
            if env is not None:
                key = f"{bidx:08d}".encode("ascii")
                # 使用 protocol=5（更快/更小）；复用 BytesIO 以减少重复分配
                # blob = pickle.dumps(cpu_generated, protocol=pickle.HIGHEST_PROTOCOL)
                buffer = io.BytesIO()
                torch.save(cpu_generated, buffer)  # 使用 torch.save，这样读取时 map_location='cpu' 才会生效
                blob = buffer.getvalue()
                _batch_pairs.append((key, memoryview(blob)))
                _batch_bytes += len(blob)
                if (len(_batch_pairs) >= FLUSH_EVERY) or (_batch_bytes >= FLUSH_BYTES):
                    _flush_batch()

    if env is not None:
        # 别忘了冲刷最后一批
        _flush_batch()
        # 手动刷盘（因为 sync=False / map_async=True）
        env.sync()
        env.close()
    return None


# ---- LMDB-backed Dataset: 按条读取生成字典，支持 DataLoader 多线程/预取 ----
class GenDictLMDBDataset(torch.utils.data.Dataset):
    """
    只读 LMDB，每次 __getitem__ 读取一条生成字典（与原先每步写入的 pickle 完全一致）。
    支持 DataLoader(num_workers>0, prefetch_factor, persistent_workers)。注意不要在 __init__ 里持有已打开的
    lmdb.Environment（不可被 pickle），而是每个 worker 在首次访问时各自打开（lazy per-worker）。
    """

    def __init__(self, lmdb_path: str, limit: int | None = None):
        assert os.path.exists(lmdb_path), f"LMDB path not found: {lmdb_path}"
        self.lmdb_path = lmdb_path
        self._env = None  # 每个进程/线程独立 lazy 打开

        # 用一个临时 env 统计长度，然后立刻关闭，避免把 env 保存在实例里被 DataLoader 拷贝
        with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, subdir=False, max_readers=4096) as env:
            with env.begin(write=False) as txn:
                stat = txn.stat()
                n_entries = stat.get('entries', 0)
                has_meta = txn.get(b"__meta__") is not None
                n = n_entries - (1 if has_meta else 0)
        if limit is not None:
            n = min(n, limit)
        self._length = max(n, 0)

    def __len__(self):
        return self._length

    def _get_env(self):
        # 在每个 worker 进程内首次调用时打开一个只读 env
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=True, subdir=False, max_readers=4096
            )
        return self._env

    def __getitem__(self, idx: int):
        env = self._get_env()
        key = f"{idx:08d}".encode("ascii")
        with env.begin(write=False) as txn:
            blob = txn.get(key)
            if blob is None:
                raise IndexError(f"LMDB key not found: {key!r}")
            # 优先用 torch.load(map_location='cpu') 强制映射到 CPU，避免反序列化为 CUDA 张量
            try:
                item = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
            except Exception:
                # 回退到普通 pickle，再递归搬到 CPU（兼容历史旧数据）
                item = pickle.loads(blob)

                def _to_cpu(x):
                    if torch.is_tensor(x):
                        return x.to('cpu')
                    elif isinstance(x, (list, tuple)):
                        return type(x)(_to_cpu(xx) for xx in x)
                    elif isinstance(x, dict):
                        return {kk: _to_cpu(vv) for kk, vv in x.items()}
                    return x

                item = _to_cpu(item)
        return item

    def __del__(self):
        try:
            if getattr(self, "_env", None) is not None:
                self._env.close()
        except Exception:
            pass


def test_inter_modal_attn_ratio():
    pass


def debug_start(gen_loader, processor, test_dataloader, save_csv_path: str | None = None):
    """
    统计：在生成第一个 token (t=0) 时，各层对【图片 / 正文文本 / 指令+BOS】三类键的注意力强度，
    并在所有样本上做平均；此外收集“逐样本×逐层”的图/文注意力指标，并与标签做相关与FDR校正。
    """
    tok = getattr(processor, "tokenizer", processor)

    # 特殊标记 id（用于图像区间）
    vid_s = tok.convert_tokens_to_ids("<|vision_start|>")
    vid_e = tok.convert_tokens_to_ids("<|vision_end|>")

    # 多种锚点候选，增强匹配鲁棒性
    left_candidates = ["Text: ", "Text:", "\nText: ", "\nText:", "Text:\n", "\nText:\n","文字: ", "文字:", "\n文字: ", "\n文字:", "文字:\n", "\n文字:\n"]
    right_candidates = [
        "\n\nPlease reply with:",
        "\n\nPlease reply with: ",
        "\nPlease reply with:",
        "\nPlease reply with: ",
        "Please reply with: ",
        "Please reply with:",
        "\n\n请回复：",
        "\n\n请回复： ",
        "\n请回复：",
        "\n请回复： ",
        "请回复： ",
        "请回复："
    ]

    def _tok_ids(s: str):
        return tok(s, add_special_tokens=False, return_attention_mask=False, return_tensors=None)["input_ids"]

    left_id_seqs = [_tok_ids(s) for s in left_candidates]
    right_id_seqs = [_tok_ids(s) for s in right_candidates]

    def find_first(hay: torch.Tensor, needles: list[list[int]]):
        h = hay.tolist()
        Lh = len(h)
        for ids in needles:
            Ln = len(ids)
            if Ln == 0 or Lh < Ln:
                continue
            for i in range(Lh - Ln + 1):
                if h[i:i + Ln] == ids:
                    return i, Ln
        return None, None

    # 从 gen_loader 里先拉一条样本来确定层数（不会一次性加载全部）
    _first_item = next(iter(gen_loader))
    # DataLoader(batch_size=1, collate_fn=lambda b: b[0]) -> 直接是 dict
    first_step_layers = _first_item["attentions"][0]
    num_layers = len(first_step_layers)

    # 仅保留 ALL 统计（不再按标签分组）
    total_samples_overall = 0  # 总样本数

    # Head×Layer 的“正常版占比”（每个 head 的 img/txt/ins share），用于热力图
    # 采用 sum / count 的形式，样本结束后做平均；H 在第一条样本第0层处确定
    hl_sum_img = None  # np.ndarray[num_layers, H]
    hl_sum_txt = None
    hl_sum_ins = None
    hl_cnt = None      # np.ndarray[num_layers]

    # -------- pooled accumulators (sum-then-divide across heads & samples) --------
    def _new_pooled(num_layers: int):
        return {
            # normal（未除token数）：累计“各头求和”的总量与头数
            "norm_img_headsum": np.zeros((num_layers,), dtype=np.float64),
            "norm_txt_headsum": np.zeros((num_layers,), dtype=np.float64),
            "norm_ins_headsum": np.zeros((num_layers,), dtype=np.float64),
            "norm_H_total"    : np.zeros((num_layers,), dtype=np.float64),
            "norm_count"      : np.zeros((num_layers,), dtype=np.int64),
            # per-token：累计 “(各头求和)/n_token”的总量与头数
            "pt_img_headsum"  : np.zeros((num_layers,), dtype=np.float64),
            "pt_txt_headsum"  : np.zeros((num_layers,), dtype=np.float64),
            "pt_ins_headsum"  : np.zeros((num_layers,), dtype=np.float64),
            "pt_H_total"      : np.zeros((num_layers,), dtype=np.float64),
            "pt_count"        : np.zeros((num_layers,), dtype=np.int64),
        }

    # 仅 ALL 的 pooled 统计
    pooled_all = _new_pooled(num_layers)

    # 重新构建 gen_loader 的迭代器（上面 next(iter(...)) 已消耗一条，这里重新获取一次）
    gen_iter = iter(gen_loader)
    test_iter = iter(test_dataloader)
    for bidx, ((inputs, _, _), gen_item) in enumerate(
            tqdm(
                    zip(test_iter, gen_iter),
                    desc="Testing", total=len(gen_loader)
                    )
            ):
        # bs=1
        assert inputs["input_ids"].shape[0] == 1, "本函数假设 batch_size=1。"

        if bidx == 0:
            print("input_ids:", inputs["input_ids"].shape)
            print("attention_mask:", inputs["attention_mask"].shape)

        # 取该样本 t=0 的 attention（步 0）
        per_step_list = gen_item["attentions"]
        if not per_step_list:
            print("<UNK>\n")
            continue
        per_layer_t0 = per_step_list[0]  # list/tuple[len = num_layers]

        input_ids = inputs["input_ids"][0]  # (L,)
        attn_mask = inputs.get("attention_mask", None)
        key_valid = attn_mask[0].bool() if attn_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # 1) 图像区间：<|vision_start|>...<|vision_end|>
        pos_vs = (input_ids == vid_s).nonzero(as_tuple=False)
        pos_ve = (input_ids == vid_e).nonzero(as_tuple=False)
        if len(pos_vs) > 0 and len(pos_ve) > 0:
            vs = int(pos_vs[0].item())
            ve = int(pos_ve[-1].item())
            pos = torch.arange(input_ids.size(0), device=input_ids.device)
            in_vision = (pos >= vs) & (pos <= ve)
            # img_mask = in_vision & (input_ids == vid_pad) & key_valid
            # if not img_mask.any():
            #     img_mask = in_vision & key_valid
            img_mask = in_vision & key_valid
        else:
            img_mask = torch.zeros_like(key_valid)

        # 2) 正文区间：在 token 序列中寻找 "Text:" 左锚与 "Please reply with:" 右锚
        L, Llen = find_first(input_ids, left_id_seqs)
        R, _ = find_first(input_ids, right_id_seqs)
        if (L is None) or (R is None) or (R <= L + (Llen or 0)):
            txt_mask = (~img_mask) & key_valid
            inst_mask = torch.zeros_like(key_valid)
        else:
            txt_start = L + Llen
            txt_end = R
            pos = torch.arange(input_ids.size(0), device=input_ids.device)
            txt_mask = (pos >= txt_start) & (pos < txt_end) & key_valid & (~img_mask)
            inst_mask = key_valid & (~img_mask) & (~txt_mask)

        if not (key_valid.any() and (img_mask.any() or txt_mask.any() or inst_mask.any())):
            continue
        if bidx < 3:  # 抽查前 3 条
            print(f"\n[DEBUG Audit {bidx}]")
            # 解码被 mask 选中的部分，看看是不是真的是 Image 和 Text
            ids = input_ids.cpu()
            print(f"  Visual Tokens Found:{img_mask.sum().item()}")
            print(f"  Text Content:{processor.tokenizer.decode(ids[txt_mask], skip_special_tokens=False)}")
            print(f"  Inst Content:{processor.tokenizer.decode(ids[inst_mask], skip_special_tokens=False)}")
        # continue

        n_img = int(img_mask.sum().item())
        n_txt = int(txt_mask.sum().item())
        total_samples_overall += 1

        for layer_idx in range(num_layers):
            attn = per_layer_t0[layer_idx]
            if attn is None:
                continue

            # # (B=1, H, Q, K) -> 取 Q 的最后一位（或解码Q=1时第0位）
            # A = attn[0]  # (H, Q, K)
            # H, Q, K = A.shape
            # last_q = A[:, 0, :] if Q == 1 else A[:, -1, :]  # (H, K)
            # # 提高数值精度，后续求和/比值用 float64
            # last_q = last_q.to(torch.float64)
            last_q = attn
            H, _ = attn.shape

            # 确保是 float64 (如果在外面转过了，这里可以省)
            if last_q.dtype != torch.float64:
                last_q = last_q.to(torch.float64)

            # 1) 各头对三类键的注意力总和（head 维度保留）
            img_h = last_q[:, img_mask].sum(dim=-1)  # (H,)
            txt_h = last_q[:, txt_mask].sum(dim=-1)  # (H,)
            ins_h = last_q[:, inst_mask].sum(dim=-1)  # (H,)
            eps = float(torch.finfo(last_q.dtype).eps) if last_q.is_floating_point() else 1e-18
            if eps < 1e-18:
                eps = 1e-18

            # --- 计算 head 级别的“正常版占比”（每个 head 的 share）以便画热力图 ---
            tot_h = img_h + txt_h + ins_h + eps
            share_img_heads = (img_h / tot_h).detach().float().cpu().numpy()  # (H,)
            share_txt_heads = (txt_h / tot_h).detach().float().cpu().numpy()  # (H,)
            share_ins_heads = (ins_h / tot_h).detach().float().cpu().numpy()  # (H,)

            # 懒初始化：在第一次看到样本时，确定 H 并建表
            if hl_sum_img is None:
                H_local = share_img_heads.shape[0]
                hl_sum_img = np.zeros((num_layers, H_local), dtype=np.float64)
                hl_sum_txt = np.zeros((num_layers, H_local), dtype=np.float64)
                hl_sum_ins = np.zeros((num_layers, H_local), dtype=np.float64)
                hl_cnt = np.zeros((num_layers,), dtype=np.int64)

            hl_sum_img[layer_idx, :share_img_heads.shape[0]] += share_img_heads
            hl_sum_txt[layer_idx, :share_txt_heads.shape[0]] += share_txt_heads
            hl_sum_ins[layer_idx, :share_ins_heads.shape[0]] += share_ins_heads
            hl_cnt[layer_idx] += 1
            # -------- accumulate pooled sums (normal) --------
            sum_img = float(img_h.sum().item())
            sum_txt = float(txt_h.sum().item())
            sum_ins = float(ins_h.sum().item())

            P = pooled_all
            P["norm_img_headsum"][layer_idx] += sum_img
            P["norm_txt_headsum"][layer_idx] += sum_txt
            P["norm_ins_headsum"][layer_idx] += sum_ins
            P["norm_H_total"][layer_idx] += float(H)
            P["norm_count"][layer_idx] += 1

            # -------- accumulate pooled sums (per-token) --------
            n_img_safe = max(n_img, 1)
            n_txt_safe = max(n_txt, 1)
            n_ins = int(inst_mask.sum().item())
            n_ins_safe = max(n_ins, 1)

            img_headsum_pt = float((img_h / (n_img_safe + eps)).sum().item())
            txt_headsum_pt = float((txt_h / (n_txt_safe + eps)).sum().item())
            ins_headsum_pt = float((ins_h / (n_ins_safe + eps)).sum().item())

            P = pooled_all
            P["pt_img_headsum"][layer_idx] += img_headsum_pt
            P["pt_txt_headsum"][layer_idx] += txt_headsum_pt
            P["pt_ins_headsum"][layer_idx] += ins_headsum_pt
            P["pt_H_total"][layer_idx] += float(H)
            P["pt_count"][layer_idx] += 1

    def _safe_div(a, b, eps=1e-18):
        return a / (b + eps)

    print(f"=== 样本总数: {total_samples_overall} 条 ===")
    print("=== 生成第一个 token 时，各层注意力【图/文/指令】统计（样本汇总：先加后除，head与样本均为sum-then-divide） ===")
    PALL = pooled_all
    for l in range(num_layers):
        # normal
        avg_img = _safe_div(PALL["norm_img_headsum"][l], PALL["norm_H_total"][l])
        avg_txt = _safe_div(PALL["norm_txt_headsum"][l], PALL["norm_H_total"][l])
        avg_ins = _safe_div(PALL["norm_ins_headsum"][l], PALL["norm_H_total"][l])
        avg_tot = avg_img + avg_txt + avg_ins
        r_it = _safe_div(avg_img, avg_txt)
        r_ii = _safe_div(avg_img, avg_ins)
        r_ti = _safe_div(avg_txt, avg_ins)
        s_i = _safe_div(avg_img, avg_tot)
        s_t = _safe_div(avg_txt, avg_tot)
        s_s = _safe_div(avg_ins, avg_tot)

        # per-token
        avg_img_pt = _safe_div(PALL["pt_img_headsum"][l], PALL["pt_H_total"][l])
        avg_txt_pt = _safe_div(PALL["pt_txt_headsum"][l], PALL["pt_H_total"][l])
        avg_ins_pt = _safe_div(PALL["pt_ins_headsum"][l], PALL["pt_H_total"][l])
        avg_tot_pt = avg_img_pt + avg_txt_pt + avg_ins_pt
        pr_it = _safe_div(avg_img_pt, avg_txt_pt)
        pr_ii = _safe_div(avg_img_pt, avg_ins_pt)
        pr_ti = _safe_div(avg_txt_pt, avg_ins_pt)
        ps_i = _safe_div(avg_img_pt, avg_tot_pt)
        ps_t = _safe_div(avg_txt_pt, avg_tot_pt)
        ps_s = _safe_div(avg_ins_pt, avg_tot_pt)

        sep = "─" * 80
        print(sep)
        print(f"Layer {l:02d}")
        print(f"  [normal]   ratio(img/txt, img/ins, txt/ins) = ({r_it:.4f}, {r_ii:.4f}, {r_ti:.4f})")
        print(f"  [normal]   share(img, txt, ins)             = ({s_i:.4f}, {s_t:.4f}, {s_s:.4f})")
        print(f"  [per-token]ratio(img/txt, img/ins, txt/ins) = ({pr_it:.4f}, {pr_ii:.4f}, {pr_ti:.4f})")
        print(f"  [per-token]share(img, txt, ins)             = ({ps_i:.4f}, {ps_t:.4f}, {ps_s:.4f})")

    # 不再保存 per_sample_rows

    # ---------- 画图并保存 ----------
    def _plot_lines(
        series_dict: dict[str, list[float]], x_vals: list[int], out_path: str, ylabel: str, title: str,
        colors: dict[str, str] | None = None, hlines: list[float] | None = None,
        ylim: tuple[float, float] | None = None
        ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure()
        for label, ys in series_dict.items():
            if ys is None or len(ys) == 0:
                continue
            color = colors.get(label) if colors else None
            plt.plot(x_vals, ys, label=label, linewidth=2.0, alpha=0.95, color=color)
        if hlines:
            for y in hlines:
                plt.axhline(y, linestyle="--", linewidth=1.0, alpha=0.8, color="#666666")
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend(loc="best", frameon=False)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _emit_heatmap(mat: np.ndarray, out_path: str, title: str, ylabel: str = "Layer", xlabel: str = "Head"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Keep the original figure shape while using larger typography.
        fig, ax = plt.subplots()
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        # 双保险：强制锁死色轴范围
        im.set_clim(0.0, 1.0)

        cbar = fig.colorbar(im, ax=ax)
        cbar.mappable.set_clim(0.0, 1.0)
        ticks = np.linspace(0.0, 1.0, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.set_label("Share", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=400)
        plt.close(fig)

    def _print_topk_from_heatmap(mat: np.ndarray, k: int, name: str, out_dir: str | None = None):
        """从给定热力图矩阵中提取 top-k 的 (layer, head) 坐标，并打印/保存。

        mat: 2D numpy array, shape (num_layers, num_heads)
        k: 要输出的前 k 大值（用于控制打印与最大截断）；
           若 out_dir 不为 None，会额外基于同一排序结果导出多个 top-k 版本
           （例如 top5/10/15/20/25/50/75/100/125/150/175/200/225/250）。
        name: 热力图名称前缀，用于日志和文件名
        out_dir: 若不为 None，则将结果同时写入 out_dir 下的若干 txt 文件
        """
        if mat is None:
            return
        flat = mat.ravel()
        if flat.size == 0:
            return
        k = min(k, flat.size)
        # 找到 top-k 的扁平索引
        idx_topk = np.argpartition(flat, -k)[-k:]
        # 按值从大到小排序，得到全局 top-k 顺序
        idx_topk = idx_topk[np.argsort(flat[idx_topk])[::-1]]
        coords = []
        for idx in idx_topk:
            layer, head = np.unravel_index(int(idx), mat.shape)
            value = float(flat[idx])
            coords.append((layer, head, value))

        # 排序后的坐标（按 layer, head 升序），便于直接复制使用
        sorted_coords = sorted(coords, key=lambda x: (x[0], x[1]))

        # 控制台只打印当前 k 的完整列表（按数值降序）
        print(f"[Top-{k}] {name} heatmap (layer, head, value):")
        for layer, head, value in coords:
            print(f"  (layer={layer}, head={head}) = {value:.6f}")

        print(f"[Top-{k}] {name} heatmap (copy-paste coordinates):")
        for layer, head, _ in sorted_coords:
            print(f"    ({layer}, {head}),")

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            # 预定义需要导出的多个 top-k 级别；只保留不超过当前 k 和总元素数的那些
            k_levels = [5,10,15,20,25,50,75,100,125,150,175,200,225]
            max_n = len(coords)
            for kk in k_levels:
                kk_eff = min(kk, k, max_n)
                if kk_eff <= 0:
                    continue
                # 按数值降序截断前 kk_eff 个
                coords_kk = coords[:kk_eff]
                # 对应的 (layer, head) 再按 layer/head 升序给一个 copy-paste 版本
                sorted_coords_kk = sorted(coords_kk, key=lambda x: (x[0], x[1]))

                txt_path = os.path.join(out_dir, f"top{kk_eff}_{name}_coords.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"# Top-{kk_eff} coordinates for {name} heatmap (layer, head, value)\n")
                    for layer, head, value in coords_kk:
                        f.write(f"layer={layer}, head={head}, value={value:.6f}\n")
                    f.write("\n# Copy-paste style coordinates (layer, head)\n")
                    for layer, head, _ in sorted_coords_kk:
                        f.write(f"({layer}, {head}),\n")
                print(f"[Top-{kk_eff}] {name} heatmap coords saved to: {txt_path}")

    def _layer_means_all() -> dict:
        P = pooled_all
        res = {k: [0.0] * num_layers for k in [
            "norm_ratio_img_txt", "norm_ratio_img_ins", "norm_ratio_txt_ins",
            "norm_share_img", "norm_share_txt", "norm_share_ins",
            "pt_ratio_img_txt", "pt_ratio_img_ins", "pt_ratio_txt_ins",
            "pt_share_img", "pt_share_txt", "pt_share_ins",
        ]}
        for l in range(num_layers):
            avg_img = _safe_div(P["norm_img_headsum"][l], P["norm_H_total"][l])
            avg_txt = _safe_div(P["norm_txt_headsum"][l], P["norm_H_total"][l])
            avg_ins = _safe_div(P["norm_ins_headsum"][l], P["norm_H_total"][l])
            avg_tot = avg_img + avg_txt + avg_ins
            res["norm_ratio_img_txt"][l] = _safe_div(avg_img, avg_txt)
            res["norm_ratio_img_ins"][l] = _safe_div(avg_img, avg_ins)
            res["norm_ratio_txt_ins"][l] = _safe_div(avg_txt, avg_ins)
            res["norm_share_img"][l] = _safe_div(avg_img, avg_tot)
            res["norm_share_txt"][l] = _safe_div(avg_txt, avg_tot)
            res["norm_share_ins"][l] = _safe_div(avg_ins, avg_tot)

            avg_img_pt = _safe_div(P["pt_img_headsum"][l], P["pt_H_total"][l])
            avg_txt_pt = _safe_div(P["pt_txt_headsum"][l], P["pt_H_total"][l])
            avg_ins_pt = _safe_div(P["pt_ins_headsum"][l], P["pt_H_total"][l])
            avg_tot_pt = avg_img_pt + avg_txt_pt + avg_ins_pt
            res["pt_ratio_img_txt"][l] = _safe_div(avg_img_pt, avg_txt_pt)
            res["pt_ratio_img_ins"][l] = _safe_div(avg_img_pt, avg_ins_pt)
            res["pt_ratio_txt_ins"][l] = _safe_div(avg_txt_pt, avg_ins_pt)
            res["pt_share_img"][l] = _safe_div(avg_img_pt, avg_tot_pt)
            res["pt_share_txt"][l] = _safe_div(avg_txt_pt, avg_tot_pt)
            res["pt_share_ins"][l] = _safe_div(avg_ins_pt, avg_tot_pt)
        return res

    def _emit_text_report(out_dir_base: str, sample_count: int = 0):
        report_dir = os.path.join(out_dir_base, "reports", "ALL")
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, "summary.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 分析报告 - ALL\n\n")
            f.write(f"样本数量：**{sample_count}**\n\n")
            if sample_count == 0:
                f.write("（没有样本，未生成图像，仅占位报告。）\n")
                return
            lm = _layer_means_all()
            for L in range(num_layers):
                r_it = lm['norm_ratio_img_txt'][L]
                r_ii = lm['norm_ratio_img_ins'][L]
                r_ti = lm['norm_ratio_txt_ins'][L]
                s_i = lm['norm_share_img'][L]
                s_t = lm['norm_share_txt'][L]
                s_s = lm['norm_share_ins'][L]
                pr_it = lm['pt_ratio_img_txt'][L]
                pr_ii = lm['pt_ratio_img_ins'][L]
                pr_ti = lm['pt_ratio_txt_ins'][L]
                ps_i = lm['pt_share_img'][L]
                ps_t = lm['pt_share_txt'][L]
                ps_s = lm['pt_share_ins'][L]
                f.write(
                    f"Layer {L:02d} | [normal] ratio(img/txt, img/ins, txt/ins)=({r_it:.4f},{r_ii:.4f},{r_ti:.4f}) "
                    f"[normal] share(img,txt,ins)=({s_i:.4f},{s_t:.4f},{s_s:.4f}) | "
                    f"[per-token] ratio=({pr_it:.4f},{pr_ii:.4f},{pr_ti:.4f}) "
                    f"[per-token] share(img,txt,ins)=({ps_i:.4f},{ps_t:.4f},{ps_s:.4f})\n"
                )
        print(f"[Report] Saved: {path}")

    def _emit_all_plots(out_dir_base: str):
        lm = _layer_means_all()
        x_layers = list(range(num_layers))
        base_dir = os.path.join(out_dir_base, "plots", "ALL")
        os.makedirs(base_dir, exist_ok=True)

        _plot_lines(
            {"img": lm["norm_share_img"], "txt": lm["norm_share_txt"], "ins": lm["norm_share_ins"]},
            x_layers, os.path.join(base_dir, "norm_share.png"), ylabel="Share",
            title="[ALL] Normal Share per Layer (img/txt/ins)", colors=None, hlines=None,
            ylim=(0.0, 1.0)
        )
        _plot_lines(
            {"img": lm["pt_share_img"], "txt": lm["pt_share_txt"], "ins": lm["pt_share_ins"]},
            x_layers, os.path.join(base_dir, "pt_share.png"), ylabel="Per-token Share",
            title="[ALL] Per-token Share per Layer (img/txt/ins)", colors=None, hlines=None,
            ylim=(0.0, 1.0)
        )
        _plot_lines(
            {"img/txt": lm["norm_ratio_img_txt"], "img/ins": lm["norm_ratio_img_ins"], "txt/ins": lm["norm_ratio_txt_ins"]},
            x_layers, os.path.join(base_dir, "norm_ratio.png"), ylabel="Ratio",
            title="[ALL] Normal Ratios per Layer", colors=None, hlines=[1.0],
            ylim=(0.0, 3.0)
        )
        _plot_lines(
            {"img/txt": lm["pt_ratio_img_txt"], "img/ins": lm["pt_ratio_img_ins"], "txt/ins": lm["pt_ratio_txt_ins"]},
            x_layers, os.path.join(base_dir, "pt_ratio.png"), ylabel="Per-token Ratio",
            title="[ALL] Per-token Ratios per Layer", colors=None, hlines=[1.0],
            ylim=(0.0, 3.0)
        )

        img_mat, txt_mat = None, None
        if hl_sum_img is not None and hl_cnt is not None:
            cnt = hl_cnt.clip(min=1)
            img_mat = hl_sum_img / cnt[:, None]
            txt_mat = hl_sum_txt / cnt[:, None]
            _emit_heatmap(img_mat, os.path.join(base_dir, "heatmap_norm_share_img.png"),
                          title="[ALL] Head×Layer Heatmap (IMG share)")
            _emit_heatmap(txt_mat, os.path.join(base_dir, "heatmap_norm_share_txt.png"),
                          title="[ALL] Head×Layer Heatmap (TXT share)")
            _print_topk_from_heatmap(txt_mat, k=200, name="heatmap_norm_share_text", out_dir=base_dir)
            _print_topk_from_heatmap(img_mat, k=200, name="heatmap_norm_share_img", out_dir=base_dir)
        return img_mat, txt_mat

    out_root = os.path.dirname(save_csv_path) if save_csv_path else "."
    _emit_all_plots(out_root)
    _emit_text_report(out_root, sample_count=total_samples_overall)

    print(f"[Outputs] Saved ALL charts and report under: {out_root}")

    layer_means_all = _layer_means_all()
    return {
        "layer_means": layer_means_all
    }


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = get_parser().parse_args()
    set_random_seed(args.seed)
    # ******************************************************************************
    args.data_parallel_size = 1
    args.train_micro_batch_size_per_gpu = 1
    args.model_type = "Qwen3_VL"
    args.img_dir = "./data/raw/"
    args.model_path = "./models/Qwen/Qwen3-VL-2B-Instruct"
    args.data_path = "./data/processed/DGM4/raw/test.pkl"
    args.prompt_version = "DGM4_sft"
    args.lora = True if 1 == 1 else False
    args.lora_path = None if not args.lora else "results/DGM4/baseline/sdpa/qwen3_vl_2B/lora/mix_ratio_0.0025/0-256/bs24_lr5e-05_wd1e-06_ep5_dp1_gacc2_clip1.0_loraTrue_r256_a512_d0.05/prompt_DGM4_sft/model_best"
    total_batch_num = 500
    outputs_dir = "results/DGM4/attn_analysis/multi/Qwen3_VL_2B/lora/mix_0.0025/0-256"
    gen_dict_dir = os.path.join(outputs_dir, "gen_dict.pt")
    collect_dict_flag = True if 1 == 1 else False
    is_cn = False
    # ******************************************************************************

    from transformers import AutoProcessor as ModelProcessor

    processor = ModelProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
        use_fast=True,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    test_dataset = VLMDataset(args)
    collate_fn = CollateFn(args, processor, cn=is_cn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.train_micro_batch_size_per_gpu,
        num_workers=2, pin_memory=True,
        collate_fn=collate_fn, prefetch_factor=4,
        persistent_workers=False, shuffle=False,
    )
    # 确保 batch size 为 1，以保证 LMDB 写入顺序与 DataLoader 顺序一致
    assert args.train_micro_batch_size_per_gpu == 1, "本脚本假定每次输入批次为 1，以保证 LMDB 写入顺序与 DataLoader 顺序一致。"

    # 使用 LMDB 文件存储 gen_dict
    gen_dict_dir = os.path.join(outputs_dir, "gen_dict.lmdb")
    if collect_dict_flag:
        os.makedirs(outputs_dir, exist_ok=True)
        _ = get_gen_dict(test_dataloader, total_batch_num, args, lmdb_path=gen_dict_dir, write_to_lmdb=True)
        print(f"gen_dict streamed to LMDB at {gen_dict_dir}")

    # 使用多线程/预加载的 DataLoader 从 LMDB 拉取生成结果
    gen_dataset = GenDictLMDBDataset(gen_dict_dir, limit=total_batch_num)
    # NOTE: collate_fn must be picklable for DataLoader with num_workers>0 (spawn); lambdas are not picklable!
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=max(1, os.cpu_count() // 4),
        num_workers=2,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=first_item_collate,
    )
    print(f"gen_dict will be streamed from LMDB with DataLoader (len={len(gen_dataset)})")

    # Analyze attention weights
    save_csv_path = os.path.join(outputs_dir, "attn_analysis.csv")
    debug_start(gen_loader, processor, test_dataloader, save_csv_path=save_csv_path)

# CUDA_VISIBLE_DEVICES=1 python -m src.utils.attn_analysis
