import os
import json
import pickle
import argparse
from typing import Iterable, Tuple, List, Dict, Any
import random
from pprint import pprint
from collections import Counter

"""
准备 DGM4 数据（按新需求）：
1) 遍历 --data_dir 下的所有 .json 文件，记录其相对路径 sub_path（例如："bbc/train.json"）。
2) 对于每个 json，输出目录 = --output_dir / sub_path 的父目录，并将数据存为同名 .pkl（如 train.pkl）。
3) 输出字段（每个字段为等长 list）：
   - id         <- json['id']（或备选 json['post_id']）
   - image_path <- json['image']（或备选 json['image_name']）
   - text       <- json['text']（若为 list 则拼接成字符串）
   - label      <- fake_cls 映射：'orig'->0，其余->1（若缺失则回退到 json['label'] 等并按同规则映射）
   - img_label  <- fake_cls 映射：包含 'face' -> 1，否则 0
   - text_label <- fake_cls 映射：包含 'text' -> 1，否则 0
   - 其它可选键原样保留（如 'fake_image_box', 'fake_text_pos', 'mtcnn_boxes', 'fake_cls'）。
4) 兼容性：若顶层是字典，尝试从 data/posts 等常见键读取；若是列表则直接处理。
"""


def _iter_json_files(data_dir: str) -> Iterable[Tuple[str, str]]:
    """遍历 data_dir，返回 (sub_path, full_path)。"""
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith('.json'):
                full_path = os.path.join(root, fn)
                sub_path = os.path.relpath(full_path, data_dir).replace('\\', '/')
                yield sub_path, full_path


def _load_items(json_path: str) -> List[Dict[str, Any]]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except json.JSONDecodeError:
        items = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    # 顶层可能是 list 或 dict
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        # 常见容器字段兜底
        for key in ('data', 'posts', 'items', 'examples'):
            v = content.get(key)
            if isinstance(v, list):
                return v
        # 若 dict 本身就是一条记录，则包一层
        if content:
            # 如果 value 中存在 list，则选其中一个 list
            for v in content.values():
                if isinstance(v, list):
                    return v
            return [content]
    return []


optional_passthrough_keys = ['fake_image_box', 'fake_text_pos', 'mtcnn_boxes', 'fake_cls']


def _normalize_text(val: Any) -> str:
    if val is None:
        return ''
    if isinstance(val, list):
        return ' '.join(str(x) for x in val)
    return str(val)


def _to_str_lower(val: Any) -> str:
    if val is None:
        return ''
    return str(val).strip().lower()


def _labels_from_fake_cls(fake_cls: Any, fallback_label: Any = None) -> Tuple[int, int, int]:
    """Return (label, img_label, text_label)."""
    fc = _to_str_lower(fake_cls)
    if not fc and fallback_label is not None:
        fc = _to_str_lower(fallback_label)
    # label: orig -> 0, others -> 1 (default to 1 when unknown)
    label = 0 if fc == 'orig' else 1
    img_label = 1 if 'face' in fc else 0
    text_label = 1 if 'text' in fc else 0
    return label, img_label, text_label


def build_pkl_for_json(json_path: str, sub_path: str, out_root: str) -> Tuple[str, int]:
    items = _load_items(json_path)
    data: Dict[str, List[Any]] = {
        'id': [],
        'image_path': [],
        'text': [],
        'label': [],
        'img_label': [],
        'text_label': [],
    }
    # 动态补充可选键，若某条没有该键则用 None 占位，确保对齐
    present_optional: Dict[str, bool] = {k: False for k in optional_passthrough_keys}

    for obj in items:
        if not isinstance(obj, dict):
            continue
        rec_id = obj.get('id', obj.get('post_id'))
        image_path = obj.get('image', obj.get('image_name'))
        text_val = _normalize_text(obj.get('text', ''))

        fake_cls = obj.get('fake_cls')
        fallback = obj.get('label', obj.get('gt_label', obj.get('target')))
        label_val, img_label_val, text_label_val = _labels_from_fake_cls(fake_cls, fallback)

        data['id'].append(rec_id)
        data['image_path'].append(image_path)
        data['text'].append(text_val)
        data['label'].append(label_val)
        data['img_label'].append(img_label_val)
        data['text_label'].append(text_label_val)

        for k in optional_passthrough_keys:
            if k in obj:
                present_optional[k] = True
                data.setdefault(k, [])
                data[k].append(obj[k])
            else:
                if k in data:
                    data[k].append(None)

    # 若某个可选键在所有样本中都不存在，则不写入 data
    for k in list(data.keys()):
        if k in present_optional and not present_optional[k]:
            data.pop(k, None)

    # 附加 sub_path 方便回溯
    data['sub_path'] = [sub_path] * len(data['id'])

    out_dir = os.path.join(out_root, os.path.dirname(sub_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(sub_path))[0]  # train / val / test
    out_path = os.path.join(out_dir, f"{base}.pkl")

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

    return out_path, len(data['id'])


def _peek_pkl(pkl_path: str, k: int, check: bool = False, img_root: str = "") -> None:
    """预览 pkl 内容：随机抽样打印、可选一致性统计、可选图片存在性抽查。"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("[peek] 空文件或无法解析：", pkl_path)
        return

    # 基本信息
    n = len(next(iter(data.values())))
    print("[peek] path:", pkl_path)
    print("[peek] keys:", list(data.keys()))
    print("[peek] num_samples:", n, "\n")

    # 抽样打印
    k = max(1, min(k, n))
    idxs = random.sample(range(n), k=k)
    for i in idxs:
        item = {key: data[key][i] for key in data.keys()}
        pprint(item, width=120)
        print('-' * 80)

    # 一致性统计
    if check and 'label' in data:
        print("\n[peek] === 统计 ===")
        print("label 分布 (0=orig, 1=非orig):", Counter(data['label']))
        if 'img_label' in data:
            print("img_label 分布 (1=含face):", Counter(data['img_label']))
        if 'text_label' in data:
            print("text_label 分布 (1=含text):", Counter(data['text_label']))
        if 'fake_cls' in data:
            pairs = Counter(zip(data['fake_cls'], data['label']))
            print("按 fake_cls × label 交叉统计（前20）:")
            for (cls, lab), cnt in pairs.most_common(20):
                print(f"{str(cls):30s} -> {lab}: {cnt}")

    # 图片存在性抽查
    if img_root:
        print("\n[peek] === 图片存在性抽查 ===")
        chk_n = min(5000, n)
        ok = 0
        for i in random.sample(range(n), k=chk_n):
            p = os.path.join(img_root, data['image_path'][i])
            if os.path.exists(p):
                ok += 1
        print(f"随机抽查 {chk_n} 张，存在 {ok} 张")
        if n > 0:
            sample_path = os.path.join(img_root, data['image_path'][0])
            print("样例路径：", sample_path)


def _save_truncated_pkl(pkl_path: str, out_path: str, max_n: int = 10000) -> Tuple[str, int]:
    """Keep the first `max_n` samples (preserving order) and save a new pkl."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not data:
        raise ValueError(f"Cannot truncate: empty pkl {pkl_path}")

    n = len(next(iter(data.values())))
    take = min(max_n, n)
    truncated = {k: v[:take] for k, v in data.items()}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(truncated, f)

    return out_path, take


def main(args: argparse.Namespace) -> None:
    # 如果指定了 --peek，则只做抽样/统计预览并退出
    if getattr(args, 'peek', ''):
        _peek_pkl(args.peek, args.k, args.check, args.img_root)
        return

    summary = []
    for sub_path, full_path in _iter_json_files(args.data_dir):
        out_path, n = build_pkl_for_json(full_path, sub_path, args.output_dir)
        print(f"Saved {n} posts → {out_path}")

        if getattr(args, 'truncate_train', False):
            base = os.path.splitext(os.path.basename(out_path))[0]
            if base == 'train':
                trunc_path = os.path.join(os.path.dirname(out_path), f"{base}_first{args.truncate_n}.pkl")
                try:
                    trunc_path, n_trunc = _save_truncated_pkl(out_path, trunc_path, max_n=args.truncate_n)
                    print(f"  Truncated: first {n_trunc} → {trunc_path}")
                except Exception as e:
                    print(f"  [truncate skipped] {e}")

        summary.append((sub_path, out_path, n))

    # 保存一次参数
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w', encoding='utf-8') as f:
        f.write(str(vars(args)))

    print(f"\nDone. {len(summary)} files processed.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DGM4 数据")
    parser.add_argument("--data_dir", type=str, default="data/raw/DGM4/metadata/", help="DGM4 元数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed/DGM4/raw/", help="输出目录（将镜像子路径）")
    parser.add_argument("--peek", type=str, default="", help="预览指定的 pkl 文件；若提供则仅执行预览并退出")
    parser.add_argument("--k", type=int, default=5, help="--peek 时随机抽样条数")
    parser.add_argument("--check", action="store_true", help="--peek 时输出 label/fake_cls 统计")
    parser.add_argument("--img_root", type=str, default="", help="--peek 时用于抽查图片是否存在的根目录")
    parser.add_argument("--truncate_train", action="store_true", help="为 train.pkl 额外生成一个按原顺序截断前 N 条的 train_firstN.pkl")
    parser.add_argument("--truncate_n", type=int, default=10000, help="--truncate_train 时截断的样本数 N")
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    main(args)

# python data/prepare/DGM4.py --truncate_train --truncate_n 40000
# 预览：python data/prepare/DGM4.py --peek data/processed/DGM4/raw/guardian/train.pkl --k 5 --check --img_root data/raw/