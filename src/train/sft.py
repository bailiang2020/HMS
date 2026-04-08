import argparse
import os
import torch
import pickle
import lmdb
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.optim as optim
from data.utils.vlm_datasets import VLMDataset
from src.utils.utils import *
from src.utils.collate import CollateFn
from transformers import AutoProcessor, get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, DistributedSampler, Subset, Sampler
import json
import sys
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import timedelta
from typing import Optional, Any, List, Dict, Tuple
import deepspeed
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel

import math
import time
import platform
import importlib
import torch.multiprocessing as mp
import gc
from collections import defaultdict

try:
    import psutil  # optional
except Exception:
    psutil = None
import logging
from logging.handlers import RotatingFileHandler
import re
import copy
from copy import deepcopy

# args.min_pixels = 256 * 28 * 28
# args.max_pixels = 512 * 28 * 28
# args.min_pixels = 4 * 28 * 28
# args.max_pixels = 256 * 28 * 28
LOGGER: logging.Logger | None = None


# Logger setup utilities
class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def _normalize_attn_impl(val: str) -> str:
    """Map user arg to HF expected strings. Default to 'sdpa' for reproducibility."""
    v = (val or "sdpa").lower()
    if v in ("fa2", "flash_attn_2", "flash-attn-2", "flash_attention_2", "flashattention2"):
        return "flash_attention_2"
    return "sdpa"


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
        suffix = model_type[len(prefix):]  # e.g., "v1", "v2", "probing_v1", "purification_v1"

        # [PPB] 检查是否为ppb系列（probing或purification）
        if suffix.startswith("probing_"):
            version = suffix  # e.g., "probing_v1"
            module_name = f"src.models.qwen3_vl_custom.ppb.probing.modeling_qwen3_vl_{version}"
        elif suffix.startswith("purification_"):
            version = suffix  # e.g., "purification_v1"
            module_name = f"src.models.qwen3_vl_custom.ppb.purification.modeling_qwen3_vl_{version}"
        else:
            # 原有逻辑，如MHA_v3
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


def setup_logger(output_dir: str, rank: int) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"train.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid duplicate handlers if called twice
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s [rank %(rank)d] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fpath = os.path.join(output_dir, f"train_rank{rank}.log")
        fh = RotatingFileHandler(fpath, maxBytes=20 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.addFilter(_RankFilter(rank))
        logger.addHandler(fh)
        # Only rank0 outputs to console to avoid clutter
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh.addFilter(_RankFilter(rank))
            logger.addHandler(sh)
    return logger


def logi(msg: str):
    global LOGGER
    if LOGGER is not None:
        LOGGER.info(msg)


# ---- Loss dict extraction helper ----
def extract_loss_items(outputs):
    """Return a dict of scalar loss components from a HF-style ModelOutput.
    Priority:
      1) outputs.loss (main loss)
      2) outputs.loss_dict (if the model provides a dict of losses)
      3) any scalar attributes whose name contains the substring 'loss'
    Values are kept as tensors (scalars) for consistency and detached to float
    only at logging time.
    """
    items = {}
    try:
        # main loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            items["loss"] = outputs.loss
        # optional explicit dict from the model
        loss_dict = getattr(outputs, "loss_dict", None)
        if isinstance(loss_dict, dict):
            for k, v in loss_dict.items():
                items[str(k)] = v
        # fallback: scan attributes with 'loss' in name
        for name in dir(outputs):
            if name.startswith("_"):
                continue
            if "loss" in name and name not in items:
                try:
                    val = getattr(outputs, name)
                except Exception:
                    continue
                if val is None:
                    continue
                if torch.is_tensor(val):
                    if val.numel() == 1:
                        items[name] = val
                elif isinstance(val, (float, int)):
                    items[name] = torch.as_tensor(val, dtype=torch.float32)
    except Exception:
        pass
    return items


# ====== PEFT / LoRA summary helper ======
def log_peft_summary(model, peft_cfg: Optional[LoraConfig] | None = None, header: str = "[PEFT]"):
    """Log which modules were wrapped by LoRA and which params are trainable.
    - Detects LoRA-wrapped modules by checking for attributes `lora_A`/`lora_B` on modules.
    - Optionally cross-checks with `peft_cfg.target_modules` to show intended vs. actual hits.
    """
    try:
        # 1) Find LoRA-wrapped modules
        lora_wrapped = []
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                lora_wrapped.append(name)

        # 2) Collect trainable params (usually LoRA params)
        trainable = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable.append(n)

        # 3) Optional: planned target modules
        planned = []
        if peft_cfg is not None:
            try:
                planned = list(peft_cfg.target_modules) if peft_cfg.target_modules is not None else []
            except Exception:
                planned = []

        # 4) Derive which planned keywords actually matched something
        matched_by_keyword = {}
        for kw in planned:
            hit = [m for m in lora_wrapped if kw in m]
            if hit:
                matched_by_keyword[kw] = hit

        # 5) Emit logs (rank-aware via logi)
        logi("==== PEFT / LoRA Summary ====")
        if planned:
            logi(f"Planned target modules (keywords): {planned}")
        logi(f"LoRA-wrapped module count: {len(lora_wrapped)}")
        if lora_wrapped:
            # show up to first 20 for brevity
            preview = lora_wrapped[:20]
            more = "" if len(lora_wrapped) <= 20 else f" ... (+{len(lora_wrapped) - 20} more)"
            logi(f"Wrapped modules (preview): {preview}{more}")
        if matched_by_keyword:
            # compact view: only list first N hits per keyword
            lines = []
            for kw, hits in matched_by_keyword.items():
                pv = hits[:10]
                more = "" if len(hits) <= 10 else f" ... (+{len(hits) - 10})"
                lines.append(f"  - '{kw}' -> {len(hits)} hits: {pv}{more}")
            logi("Keyword matches:\n" + "\n".join(lines))
        trainable_param_elems = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
        logi(f"Trainable parameters (elements): {trainable_param_elems:,}")
        total_param_elems = sum(p.numel() for p in model.parameters())
        ratio = 100.0 * trainable_param_elems / max(1, total_param_elems)
        logi(f"Trainable ratio: {ratio:.2f}% of total {total_param_elems:,}")
        if trainable:
            pv = trainable[:30]
            more = "" if len(trainable) <= 30 else f" ... (+{len(trainable) - 30} more)"
            logi(f"Trainable parameter names (preview): {pv}{more}")
        logi("==============================")
    except Exception as e:
        logi(f"[PEFT] Logging summary failed: {e}")


# ====== Rich logging helpers ======

def _env(key: str, default: str = "-") -> str:
    return os.environ.get(key, default)


def log_system_snapshot(args, ds_config):
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        rank, world = 0, 1
    # CPU / OS
    logical = os.cpu_count() or -1
    physical = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    load = None
    if psutil is not None:
        try:
            load = psutil.getloadavg()[0]
        except Exception:
            load = None
    # CUDA / GPU
    cuda_count = torch.cuda.device_count()
    cur_dev = torch.cuda.current_device() if cuda_count > 0 else -1
    try:
        gpu_name = torch.cuda.get_device_name(cur_dev) if cur_dev >= 0 else "cpu"
        total_mem = torch.cuda.get_device_properties(cur_dev).total_memory if cur_dev >= 0 else 0
    except Exception:
        gpu_name, total_mem = "unknown", 0
    # Threads
    intra = torch.get_num_threads()
    inter = torch.get_num_interop_threads()
    # NCCL / envs
    envs = {
        "CUDA_VISIBLE_DEVICES": _env("CUDA_VISIBLE_DEVICES"),
        "LOCAL_RANK": _env("LOCAL_RANK", "0"),
        "RANK": _env("RANK", "0"),
        "WORLD_SIZE": _env("WORLD_SIZE", str(world)),
        "NCCL_IB_DISABLE": _env("NCCL_IB_DISABLE", "-"),
        "NCCL_DEBUG": _env("NCCL_DEBUG", "-"),
        "OMP_NUM_THREADS": _env("OMP_NUM_THREADS", "-"),
        "MKL_NUM_THREADS": _env("MKL_NUM_THREADS", "-"),
        "NUMEXPR_NUM_THREADS": _env("NUMEXPR_NUM_THREADS", "-"),
        "TOKENIZERS_PARALLELISM": _env("TOKENIZERS_PARALLELISM", "-"),
    }
    # DS / train sizes
    mb_per_gpu = ds_config.get("train_micro_batch_size_per_gpu",
                               getattr(args, 'train_micro_batch_size_per_gpu', '-')) if isinstance(ds_config,
                                                                                                   dict) else getattr(
        args, 'train_micro_batch_size_per_gpu', '-')
    gacc = getattr(args, 'gradient_accumulation_steps', '-')
    dp = getattr(args, 'data_parallel_size', '-')
    try:
        global_micro = (mb_per_gpu or 0) * (dp or 1)
    except Exception:
        global_micro = "-"
    logi(
        "\n".join([
            "===== SYSTEM SNAPSHOT =====",
            f"rank/world: {rank}/{world}",
            f"OS: {os_info}",
            f"CPU logical/physical: {logical}/{physical}",
            f"Load(1m): {load}",
            f"Threads (intra/inter): {intra}/{inter}",
            f"GPU[{cur_dev}] {gpu_name} total_mem={total_mem / 1024 / 1024 / 1024:.2f} GB (cuda_count={cuda_count})",
            f"Attention backend: {_normalize_attn_impl(getattr(args, 'attn_impl', 'sdpa'))}",
            f"Env: {envs}",
            f"Train micro batch / GPU: {mb_per_gpu} | DP={dp} GACC={gacc} | global_micro(b/sample-step)={global_micro}",
            "==========================="
        ])
    )


def log_loader_config(name: str, loader, sampler, dp_rank: int, dp_world: int):
    if loader is None:
        logi(f"[Loader:{name}] None on rank={dp_rank}")
        return
    try:
        bs = getattr(loader, 'batch_size', None)
        nw = getattr(loader, 'num_workers', None)
        pf = getattr(loader, 'prefetch_factor', None)
        pw = getattr(loader, 'persistent_workers', None)
        try:
            length = len(loader)
        except Exception:
            length = None
        sname = type(sampler).__name__ if sampler is not None else None
        try:
            slen = len(sampler) if sampler is not None else None
        except Exception:
            slen = None
        logi(
            f"[Loader:{name}] bs={bs} num_workers={nw} prefetch={pf} persistent={pw} len(loader)={length} sampler={sname} len(sampler)={slen} rank={dp_rank}/{dp_world}")
    except Exception as e:
        logi(f"[Loader:{name}] logging error: {e}")


def bytes_gb(x: int) -> float:
    return x / (1024 ** 3)


def get_model(args):
    # Use the resolver to get the class and load via from_pretrained
    ModelCls = _resolve_qwen_vl_model_class(args.model_type)
    attn_impl = _normalize_attn_impl(getattr(args, "attn_impl", "sdpa"))
    logi(f"[Model] Using attention implementation: {attn_impl}")
    model = ModelCls.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.fake_token_id = args.fake_id
    model.config.real_token_id = args.real_id

    # ---------------------------------------------------------------------
    # [Purification] Load Head Roles from Probing Stage
    # ---------------------------------------------------------------------
    if hasattr(model, 'load_head_roles') and args.head_roles_path:
        logi(f"[Purification] Loading head roles from {args.head_roles_path}")
        try:
            model.load_head_roles(
                args.head_roles_path,
            )
            logi("[Purification] Head roles loaded successfully.")
        except Exception as e:
            logi(f"[Warning] Failed to load head roles: {e}")
            logi("[Warning] Training will proceed without purification constraint.")
    elif hasattr(model, 'load_head_roles') and not args.head_roles_path:
        logi("[Purification] Warning: Model supports purification but --head_roles_path not provided.")
        logi("[Purification] Training will proceed without purification constraint.")
    # ---------------------------------------------------------------------

    if args.lora:
        target_modules_to_save = [
            
        ]

        logi(f"[LoRA] modules_to_save: {target_modules_to_save}")

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            modules_to_save=target_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
        # Log what PEFT actually wrapped
        log_peft_summary(model, peft_config)

        print("\n=== All trainable parameters besides LoRA ===")
        trainable_count = 0
        for name, param in model.named_parameters():
            if "lora" in name:
                continue
            if param.requires_grad:
                print(f"Trainable: {name}")
                trainable_count += 1
        print(f"Total trainable parameters besides LoRA: {trainable_count}")

        logi("Attempting to register Purification gradient hooks...")
        logi(f"[Purification] protect_scale={args.purification_protect_scale}")

        # 兼容：不同封装层级下找到真正实现 register_purification_hooks 的对象
        def _try_register(module):
            if hasattr(module, "register_purification_hooks"):
                module.register_purification_hooks(protect_scale=args.purification_protect_scale)
                return True
            return False

        registered = False
        # 1) 直接在当前对象上
        registered = registered or _try_register(model)
        # 2) PeftModel -> base_model
        if (not registered) and hasattr(model, "base_model"):
            registered = registered or _try_register(model.base_model)
        # 3) PeftModel -> base_model.model (常见)
        if (not registered) and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            registered = registered or _try_register(model.base_model.model)
        # 4) 最暴力：遍历所有子模块
        if not registered:
            for module in model.modules():
                if _try_register(module):
                    registered = True
                    break

        if not registered:
            logi("Could not find 'register_purification_hooks' method (this is OK for some model versions)")
        else:
            logi("Found 'register_purification_hooks' method")


    return model


def _is_cot_prompt_version(prompt_version: Optional[str]) -> bool:
    return (prompt_version or "").strip() in {"weibo", "DGM4"}


def _resolve_cot_prompt_version(prompt_version: Optional[str], en: bool = False) -> str:
    mapping = {
        "weibo": "weibo",
        "weibo_sft": "weibo",
        "DGM4": "DGM4",
        "DGM4_sft": "DGM4",
    }
    key = (prompt_version or "").strip()
    return mapping.get(key, "DGM4" if en else "weibo")


def _resolve_non_cot_prompt_version(prompt_version: Optional[str], en: bool = False) -> str:
    mapping = {
        "weibo": "weibo_sft",
        "weibo_sft": "weibo_sft",
        "DGM4": "DGM4_sft",
        "DGM4_sft": "DGM4_sft",
    }
    key = (prompt_version or "").strip()
    return mapping.get(key, "DGM4_sft" if en else "weibo_sft")


def _resolve_eval_mode(eval_branch: Optional[str]) -> str:
    branch = (eval_branch or "current").strip().lower()
    return "generate" if branch == "cot" else "token"


def _sanitize_inputs_for_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    clean_inputs = dict(inputs)
    if isinstance(clean_inputs.get("encodings"), dict):
        encodings = dict(clean_inputs["encodings"])
        encodings.pop("ppb_image_mask", None)
        encodings.pop("ppb_news_text_mask", None)
        clean_inputs["encodings"] = encodings
    else:
        clean_inputs.pop("ppb_image_mask", None)
        clean_inputs.pop("ppb_news_text_mask", None)
    return clean_inputs


def _cot_letter_to_pred(letter: str, default: str = "A") -> int:
    final_letter = (letter or default).strip().upper()
    if final_letter not in {"A", "B"}:
        final_letter = default.upper()
    return 1 if final_letter == "A" else 0


def _flatten_text_chunks(text_chunks) -> List[str]:
    flat_text: List[str] = []
    for batch in text_chunks or []:
        if isinstance(batch, (list, tuple)):
            flat_text.extend(str(x) for x in batch)
        else:
            flat_text.append(str(batch))
    return flat_text


def _evaluate_model_token(model, test_dataloader, split_name: str = "VAL", args=None):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        try:
            logi(
                f"[Eval:{split_name}][token] loader_len={len(test_dataloader)} batch_size={getattr(test_dataloader, 'batch_size', None)}")
        except Exception:
            pass
    model.eval()
    true_id = test_dataloader.collate_fn.true_id
    processor = test_dataloader.collate_fn.processor
    preds_chunks = []
    labels_chunks = []
    gen_text_chunks = []
    input_text_chunks = []

    with torch.inference_mode():
        pbar = tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc=f"Evaluating[{split_name}]",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.2,
            leave=True,
            file=sys.stdout,
            disable=(dist.is_initialized() and dist.get_rank() != 0)
        )
        for bidx, (inputs, labels, *_) in enumerate(pbar):
            input_text = processor.batch_decode(inputs['input_ids'], )
            inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            # 统一在同一次forward中请求attn收集
            fwd_kwargs = {}
            outputs = model(**inputs, use_cache=False, return_dict=True, **fwd_kwargs)
            # 从同一次前向的 logits 里取“下一 token”的预测（等价于 generate(max_new_tokens=1) 的贪心一步）
            logits = outputs.logits  # (bsz, seq, vocab)
            # 固定左填充场景使用
            next_token_logits = logits[:, -1, :]  # (bsz, vocab)
            next_tokens = next_token_logits.argmax(dim=-1)  # (bsz,)
            gen_text = processor.batch_decode(next_tokens)
            preds_tensor = (next_tokens != true_id)
            labels_tensor = torch.tensor(labels, device=logits.device)

            # 立刻搬到CPU，避免显存累积
            preds_chunks.append(preds_tensor.to("cpu", dtype=torch.int8))
            labels_chunks.append(labels_tensor.to("cpu", dtype=torch.int8))
            gen_text_chunks.append(gen_text)
            input_text_chunks.append(input_text)

    preds = torch.cat(preds_chunks)
    tol_labels = torch.cat(labels_chunks)
    # gen_text_chunks [[bsz:str], ...]
    return preds, tol_labels, gen_text_chunks, input_text_chunks


def _evaluate_model_generate(model, test_dataloader, split_name: str = "VAL", args=None):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        try:
            logi(
                f"[Eval:{split_name}][generate] loader_len={len(test_dataloader)} batch_size={getattr(test_dataloader, 'batch_size', None)}")
        except Exception:
            pass
    model.eval()
    processor = test_dataloader.collate_fn.processor
    cot_max_new_tokens = max(1, int(getattr(args, "cot_max_new_tokens", 256)))

    preds_chunks = []
    labels_chunks = []
    gen_text_chunks = []
    input_text_chunks = []

    with torch.inference_mode():
        pbar = tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc=f"Evaluating[{split_name}]",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.2,
            leave=True,
            file=sys.stdout,
            disable=(dist.is_initialized() and dist.get_rank() != 0)
        )
        for _, (inputs, labels, *_) in enumerate(pbar):
            input_text = processor.batch_decode(inputs["input_ids"])
            inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            gen_inputs = _sanitize_inputs_for_generation(inputs)
            generated_ids = model.generate(
                **gen_inputs,
                use_cache=True,
                do_sample=False,
                max_new_tokens=cot_max_new_tokens,
            )
            prompt_len = gen_inputs["input_ids"].shape[1]
            new_token_ids = generated_ids[:, prompt_len:]
            gen_text = processor.batch_decode(new_token_ids, skip_special_tokens=True)
            pred_ids = [_cot_letter_to_pred(extract_raw_lm_answer(text)) for text in gen_text]

            preds_tensor = torch.tensor(pred_ids, device=generated_ids.device, dtype=torch.int8)
            labels_tensor = torch.tensor(labels, device=generated_ids.device)

            preds_chunks.append(preds_tensor.to("cpu", dtype=torch.int8))
            labels_chunks.append(labels_tensor.to("cpu", dtype=torch.int8))
            gen_text_chunks.append(gen_text)
            input_text_chunks.append(input_text)

    preds = torch.cat(preds_chunks)
    tol_labels = torch.cat(labels_chunks)
    return preds, tol_labels, gen_text_chunks, input_text_chunks


def evaluate_model(model, test_dataloader, split_name: str = "VAL", args=None, eval_mode: Optional[str] = None):
    mode = (eval_mode or "token").lower()
    if mode == "generate":
        return _evaluate_model_generate(model, test_dataloader, split_name=split_name, args=args)
    if mode != "token":
        raise ValueError(f"Unsupported eval_mode: {mode}")
    return _evaluate_model_token(model, test_dataloader, split_name=split_name, args=args)


def parse_args():
    parser = get_parser()
    parser.add_argument("--train_data_path", type=str, help="Path to the train file")
    parser.add_argument("--test_data_path", type=str, help="Path to the test file")
    parser.add_argument("--val_data_path", type=str, help="Path to the val file")
    parser.add_argument("--en", action="store_true", help="english mode, use 'Real' and 'Fake' as labels")
    parser.add_argument("--cot_max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate during CoT evaluation")
    parser.add_argument("--eval_branch", type=str, default="current",
                        choices=["current", "base", "cot"],
                        help="Evaluation branch: current=usual evaluation, base=plain base model, cot=plain base model with CoT prompt")
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attn_2", "flash_attention_2", "fa2", "eager"],
        help="Attention backend: 'sdpa' for fully reproducible runs; "
             "'flash_attention_2' (aka 'fa2') for speed with small numerical differences."
    )
    parser.add_argument("--min_pixels", type=int, default=0 * 32 * 32,
                        help="Min pixels")
    parser.add_argument("--max_pixels", type=int, default=256 * 32 * 32, help="Max pixels")
    parser.add_argument("--fake_id", type=int, default=1, help="Fake label ID")
    parser.add_argument("--real_id", type=int, default=1, help="Real label ID")
    parser.add_argument("--unimodal_ratio", type=float, default=0, help="Unimodal label ratio")
    parser.add_argument("--train_type", type=str, default="mix",
                        choices=["multi", "img_only", "text_only", "mix"],
                        help="训练模式: multi=仅多模态, img_only=仅图片, text_only=仅文本, mix=混合")

    # 单模态人工标注子集 JSON
    parser.add_argument("--train_unimodal_json", type=str, default=None,
                        help="Path to human-annotated unimodal subset json for training (optional)")
    parser.add_argument("--test_unimodal_json", type=str, default=None,
                        help="Path to human-annotated unimodal subset json for testing (optional)")

    # [Purification] 新增参数
    parser.add_argument("--head_roles_path", type=str, default=None,
                        help="Path to head_roles.json from probing stage")
    parser.add_argument("--purification_threshold", type=float, default=0.1,
                        help="Purification loss threshold for attention ratio (default: 0.1)")
    parser.add_argument("--purification_protect_scale", type=float, default=0.8,
                        help="Gradient scaling factor for protected heads in purification hooks (default: 0.8)")

    args = parser.parse_args()
    return args


def setup_deepspeed_config(args):
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    args.deepspeed_config = None
    return ds_config


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _uses_mix_schedule(train_type: str) -> bool:
    return (train_type or "").lower() == "mix"


# ====== Unimodal annotated subset helpers ======

def _load_unimodal_json(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unimodal json must be a list, got: {type(data)}")
    return data


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _build_dataset_lookup(dataset) -> Dict[Tuple[Any, ...], int]:
    lookup: Dict[Tuple[Any, ...], int] = {}
    ids = None
    try:
        ids = dataset.data.get("id", dataset.data.get("ids", None))
    except Exception:
        ids = None
    has_ids = isinstance(ids, (list, tuple)) and len(ids) == len(dataset)

    for i in range(len(dataset)):
        if has_ids:
            lookup[("id", _safe_str(ids[i]))] = i
        try:
            text = dataset.text[i]
        except Exception:
            text = None
        try:
            img = dataset.image_path[i]
        except Exception:
            img = None
        lookup[("ti", _safe_str(text), _safe_str(img))] = i
    return lookup


def _ensure_label_list(dataset, key: str) -> List[Any]:
    try:
        v = dataset.data.get(key, None)
    except Exception:
        v = None
    if isinstance(v, list) and len(v) == len(dataset):
        return v
    if isinstance(v, tuple) and len(v) == len(dataset):
        return list(v)
    # fallback to base labels
    base = None
    try:
        if isinstance(dataset.labels, (list, tuple)) and len(dataset.labels) == len(dataset):
            base = list(dataset.labels)
    except Exception:
        base = None
    if base is None:
        base = [0 for _ in range(len(dataset))]
    return base


def _apply_unimodal_subset(dataset, json_path: str, modality: str):
    subset = _load_unimodal_json(json_path)
    lookup = _build_dataset_lookup(dataset)
    label_key = "text_label" if modality == "text_only" else "img_label"

    # 按要求：最终要替换原数据集中的 label 键
    label_list = _ensure_label_list(dataset, "label")

    indices: List[int] = []
    seen = set()
    missed = 0

    for item in subset:
        if not isinstance(item, dict):
            missed += 1
            continue
        idx = None
        if "id" in item:
            idx = lookup.get(("id", _safe_str(item.get("id"))))
        if idx is None:
            text = item.get("text", None)
            img = item.get("image_path", item.get("image", item.get("img", None)))
            idx = lookup.get(("ti", _safe_str(text), _safe_str(img)))
        if idx is None:
            missed += 1
            continue

        # 不允许 JSON 中重复样本，避免错位
        if idx in seen:
            raise AssertionError(f"Duplicate sample in unimodal json at id/index={idx}")
        seen.add(idx)

        # 严格对齐：若 JSON 提供了字段，则必须与数据集对应项一致
        if "id" in item:
            try:
                ids = dataset.data.get("id", dataset.data.get("ids", None))
            except Exception:
                ids = None
            if isinstance(ids, (list, tuple)) and len(ids) == len(dataset):
                assert _safe_str(ids[idx]) == _safe_str(item.get("id")), (
                    f"Unimodal id mismatch at idx={idx}: dataset={ids[idx]} json={item.get('id')}"
                )
        if "text" in item:
            assert _safe_str(dataset.text[idx]) == _safe_str(item.get("text")), (
                f"Unimodal text mismatch at idx={idx}"
            )
        if ("image_path" in item) or ("image" in item) or ("img" in item):
            item_img = item.get("image_path", item.get("image", item.get("img", None)))
            assert _safe_str(dataset.image_path[idx]) == _safe_str(item_img), (
                f"Unimodal image_path mismatch at idx={idx}: dataset={dataset.image_path[idx]} json={item_img}"
            )

        if modality == "text_only":
            new_label = item.get("text_label", item.get("label", None))
        else:
            new_label = item.get("img_label", item.get("label", None))

        if new_label is not None:
            try:
                label_list[idx] = int(new_label)
            except Exception:
                label_list[idx] = new_label

        # 保持子集顺序与 JSON 顺序一致
        indices.append(idx)

    # 回写 label（核心需求）
    try:
        dataset.data["label"] = label_list
    except Exception:
        pass
    try:
        dataset.labels = label_list
    except Exception:
        pass

    # 兼容现有 VLMDataset 在 text_only/img_only 下优先读取 *_label 的逻辑：同步一份
    try:
        dataset.data[label_key] = label_list
    except Exception:
        pass

    stats = {
        "total": len(subset),
        "matched": len(indices),
        "missed": missed,
    }
    assert stats["matched"] == stats["total"], (
        f"Unimodal subset mismatch: matched {stats['matched']}/{stats['total']} (missed {stats['missed']})"
    )
    return Subset(dataset, indices), stats


def _create_eval_dataloader(dataset, args, ds_config, collate_fn, mp_ctx, seed_worker, g,
                            dp_rank, dp_world_size, name="eval"):

    loader = DataLoader(
        dataset,
        sampler=None,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        multiprocessing_context=mp_ctx,
        prefetch_factor=None if args.num_workers == 0 else 4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    log_loader_config(name, loader, None, dp_rank, dp_world_size)
    return loader


def _build_eval_args(args, prompt_version_override: Optional[str] = None):
    train_type_eval = getattr(args, 'train_type', 'mix').lower()
    if train_type_eval == 'img_only':
        args_eval = deepcopy(args)
        args_eval.img_only = True
        args_eval.text_only = False
    elif train_type_eval == 'text_only':
        args_eval = deepcopy(args)
        args_eval.text_only = True
        args_eval.img_only = False
    else:
        args_eval = deepcopy(args)
    if prompt_version_override is not None:
        args_eval.prompt_version = prompt_version_override
    return args_eval


def _build_eval_dataset(args, split: str = "test", prompt_version_override: Optional[str] = None):
    args_eval = _build_eval_args(args, prompt_version_override=prompt_version_override)
    data_attr = f"{split}_data_path"
    if not hasattr(args, data_attr):
        raise ValueError(f"Unknown split '{split}', missing args.{data_attr}")
    args_eval.data_path = getattr(args, data_attr)
    dataset = VLMDataset(args_eval)

    if split == "test" and getattr(args, "test_only", False) and getattr(args, "test_unimodal_json", None):
        if getattr(args_eval, "text_only", False):
            dataset, stats_test_text = _apply_unimodal_subset(
                dataset, args.test_unimodal_json, "text_only"
            )
            logi(
                f"[Unimodal][Test][text_only] json={args.test_unimodal_json} | "
                f"matched {stats_test_text['matched']}/{stats_test_text['total']} "
                f"(missed {stats_test_text['missed']})"
            )
        elif getattr(args_eval, "img_only", False):
            dataset, stats_test_img = _apply_unimodal_subset(
                dataset, args.test_unimodal_json, "img_only"
            )
            logi(
                f"[Unimodal][Test][img_only] json={args.test_unimodal_json} | "
                f"matched {stats_test_img['matched']}/{stats_test_img['total']} "
                f"(missed {stats_test_img['missed']})"
            )
    return args_eval, dataset


def create_eval_loader_for_split(args, ds_config, split: str = "test",
                                 prompt_version_override: Optional[str] = None,
                                 name: Optional[str] = None):
    g = torch.Generator()
    g.manual_seed(args.seed)
    mp_ctx = mp.get_context("spawn") if args.num_workers > 0 else None
    args_eval, dataset = _build_eval_dataset(args, split=split, prompt_version_override=prompt_version_override)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    collate_fn_eval = CollateFn(args_eval, processor, cn=False if args.en else True)
    dp_world_size = max(1, getattr(args, "data_parallel_size", 1))
    dp_rank = dist.get_rank() % dp_world_size if dist.is_initialized() else 0
    loader_name = name or f"{split}_prompt_{args_eval.prompt_version}"
    loader = _create_eval_dataloader(
        dataset, args_eval, ds_config, collate_fn_eval, mp_ctx, seed_worker, g,
        dp_rank, dp_world_size, name=loader_name
    )
    return loader, args_eval


def get_test_loader_for_prompt(args, ds_config, main_test_loader, prompt_version: str):
    args_eval = _build_eval_args(args, prompt_version_override=prompt_version)
    if prompt_version == getattr(args, "prompt_version", None):
        return main_test_loader, args_eval
    return create_eval_loader_for_split(
        args, ds_config, split="test", prompt_version_override=prompt_version,
        name=f"test_prompt_{prompt_version}"
    )


def prepare_dataloaders(args, ds_config):
    g = torch.Generator()
    g.manual_seed(args.seed)
    mp_ctx = mp.get_context("spawn") if args.num_workers > 0 else None

    # 1. 加载全量训练集
    args.data_path = args.train_data_path
    train_dataset = VLMDataset(args)

    # 2. 准备 Text-Only 和 Img-Only 的参数 (不修改原 args)
    args_text_only = deepcopy(args)
    args_text_only.text_only = True
    args_text_only.img_only = False

    args_img_only = deepcopy(args)
    args_img_only.img_only = True
    args_img_only.text_only = False


    _full_text_ds = VLMDataset(args_text_only)
    _full_img_ds = VLMDataset(args_img_only)

    # 3. 根据提供的人工标注子集 JSON 替换 text_only / img_only 训练子集
    if getattr(args, "train_unimodal_json", None):
        train_text_only_dataset, stats_text = _apply_unimodal_subset(
            _full_text_ds, args.train_unimodal_json, "text_only"
        )
        train_img_only_dataset, stats_img = _apply_unimodal_subset(
            _full_img_ds, args.train_unimodal_json, "img_only"
        )
        logi(
            f"[Unimodal][Train] json={args.train_unimodal_json} | "
            f"text_only matched {stats_text['matched']}/{stats_text['total']} (missed {stats_text['missed']}) | "
            f"img_only matched {stats_img['matched']}/{stats_img['total']} (missed {stats_img['missed']})"
        )
    else:
        total_len = len(train_dataset)
        if hasattr(args, 'unimodal_ratio') and args.unimodal_ratio > 0:
            subset_size = int(total_len * args.unimodal_ratio)
        else:
            subset_size = 0

        logi(f"Using fixed subset: Top {subset_size} samples for unimodal training.")
        subset_indices = list(range(subset_size))

        # 创建子集
        train_text_only_dataset = Subset(_full_text_ds, subset_indices)
        train_img_only_dataset = Subset(_full_img_ds, subset_indices)

    # 根据 train_type 动态计算总迭代次数
    train_type_calc = getattr(args, 'train_type', 'mix').lower()
    if _uses_mix_schedule(train_type_calc):
        args.total_iterations = len(train_dataset) + len(train_img_only_dataset) + len(train_text_only_dataset)
    elif train_type_calc == 'img_only':
        args.total_iterations = len(train_img_only_dataset)
    elif train_type_calc == 'text_only':
        args.total_iterations = len(train_text_only_dataset)
    elif train_type_calc == 'multi':
        args.total_iterations = len(train_dataset)
    else:
        raise ValueError(f"Unknown train type: {train_type_calc}")
    args.total_iterations = args.total_iterations // args.batch_size * args.epochs

    # 根据 train_type 决定验证/测试时的数据模态
    args_eval, test_dataset = _build_eval_dataset(args, split="test")
    _, val_dataset = _build_eval_dataset(args, split="val")

    # ---- debug: 打印替换后子集中的若干样本（仅元信息，不触发图片加载） ----
    def _debug_log_subset_samples(ds_or_subset, name: str, k: int = 3):
        try:
            if isinstance(ds_or_subset, Subset):
                base_ds = ds_or_subset.dataset
                idxs = list(ds_or_subset.indices[:k])
            else:
                base_ds = ds_or_subset
                idxs = list(range(min(k, len(base_ds))))

            ids = base_ds.data.get("id", base_ds.data.get("ids", None)) if hasattr(base_ds, "data") else None
            labels = base_ds.data.get("label", None) if hasattr(base_ds, "data") else None
            text_labels = base_ds.data.get("text_label", None) if hasattr(base_ds, "data") else None
            img_labels = base_ds.data.get("img_label", None) if hasattr(base_ds, "data") else None

            logi(f"[DEBUG][{name}] sample_count_to_show={len(idxs)}")
            for ridx, i in enumerate(idxs):
                item_id = ids[i] if isinstance(ids, (list, tuple)) and i < len(ids) else None
                text = base_ds.text[i] if hasattr(base_ds, "text") and i < len(base_ds.text) else None
                img = base_ds.image_path[i] if hasattr(base_ds, "image_path") and i < len(base_ds.image_path) else None
                lab = labels[i] if isinstance(labels, (list, tuple)) and i < len(labels) else None
                tlab = text_labels[i] if isinstance(text_labels, (list, tuple)) and i < len(text_labels) else None
                ilab = img_labels[i] if isinstance(img_labels, (list, tuple)) and i < len(img_labels) else None
                txt_preview = str(text)[:120] if text is not None else None
                logi(
                    f"[DEBUG][{name}] #{ridx} base_idx={i} id={item_id} image_path={img} "
                    f"label={lab} text_label={tlab} img_label={ilab} text[:120]={txt_preview}"
                )
        except Exception as e:
            logi(f"[DEBUG][{name}] failed to dump samples: {e}")

    _debug_log_subset_samples(train_text_only_dataset, "train_text_only")
    _debug_log_subset_samples(train_img_only_dataset, "train_img_only")
    if isinstance(test_dataset, Subset):
        _debug_log_subset_samples(test_dataset, "test_subset")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code,
                                              padding_side="left", use_fast=True, min_pixels=args.min_pixels,
                                              max_pixels=args.max_pixels)
    args.pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    collate_fn = CollateFn(args, processor, cn=False if args.en else True)
    collate_fn_text_only = CollateFn(args_text_only, processor, cn=False if args.en else True)
    collate_fn_img_only = CollateFn(args_img_only, processor, cn=False if args.en else True)
    collate_fn_eval = CollateFn(args_eval, processor, cn=False if args.en else True)

    dp_world_size = args.data_parallel_size
    dp_rank = dist.get_rank() % dp_world_size

    # 主 Loader (有标签的 multi 模态数据)
    train_loader = DataLoader(
        train_dataset, sampler=None, shuffle=True,batch_size=ds_config["train_micro_batch_size_per_gpu"],
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False, prefetch_factor=None if args.num_workers == 0 else 4,
        multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
    )

    if len(train_text_only_dataset) == 0:
        train_text_only_loader = DataLoader([])
    else:
        train_text_only_loader = DataLoader(
            train_text_only_dataset, sampler=None, shuffle=True,batch_size=ds_config["train_micro_batch_size_per_gpu"],
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_text_only,
            persistent_workers=True, prefetch_factor=None if args.num_workers == 0 else 4,
            multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
        )

    if len(train_img_only_dataset) == 0:
        train_img_only_loader = DataLoader([])
    else:
        train_img_only_loader = DataLoader(
            train_img_only_dataset, sampler=None, shuffle=True,batch_size=ds_config["train_micro_batch_size_per_gpu"],
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_img_only,
            persistent_workers=True, prefetch_factor=None if args.num_workers == 0 else 4,
            multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
        )


    true_label_id = collate_fn.true_id
    try:
        logi(
            f"Datasets — train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} | dp_rank={dp_rank}/{dp_world_size}")
    except Exception:
        pass

    assert collate_fn_text_only.args.text_only == True

    # 创建测试和验证 DataLoader（使用与 train_type 对齐的 eval collate）
    test_loader = _create_eval_dataloader(
        test_dataset, args_eval, ds_config, collate_fn_eval, mp_ctx, seed_worker, g,
        dp_rank, dp_world_size, name="test"
    )
    val_loader = _create_eval_dataloader(
        val_dataset, args_eval, ds_config, collate_fn_eval, mp_ctx, seed_worker, g,
        dp_rank, dp_world_size, name="val"
    )
    return (train_loader, train_text_only_loader, train_img_only_loader,
            test_loader, val_loader, true_label_id)


def build_training_components(args, ds_config):
    device = torch.cuda.current_device()
    model = get_model(args).to(device)
    if args.lora:
        model.print_trainable_parameters()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    lr_scheduler = get_scheduler(
        name="cosine",  
        optimizer=optimizer,
        num_warmup_steps=args.total_iterations // 10,  
        num_training_steps=args.total_iterations
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        config_params=ds_config
    )

    try:
        total_params = sum(p.numel() for p in model.parameters())
    except Exception:
        total_params = -1
    logi(f"Model initialized with parameters: {total_params / 1e6:.2f}M | Optimizer: {type(optimizer).__name__}")
    return model, optimizer, lr_scheduler


def _get_batch_for_phase(phase_name, iter_full, iter_text, iter_img,
                         train_loader, train_text_only_loader, train_img_only_loader):
    if phase_name == 'full':
        try:
            batch = next(iter_full)
        except StopIteration:
            iter_full = iter(train_loader)
            batch = next(iter_full)
    elif phase_name == 'text_only':
        try:
            batch = next(iter_text)
        except StopIteration:
            iter_text = iter(train_text_only_loader)
            batch = next(iter_text)
    elif phase_name == 'img_only':
        try:
            batch = next(iter_img)
        except StopIteration:
            iter_img = iter(train_img_only_loader)
            batch = next(iter_img)
        
    else:
        raise ValueError(f"Unknown phase: {phase_name}")
    
    inputs, _, label_ids = batch

    return inputs, label_ids, iter_full, iter_text, iter_img



def _run_evaluation_and_save(model, val_loader, args, epoch, best_acc, device, global_avg):
    """运行评估，更新最佳模型，并保存结果。

    返回: best_acc
    """
    eval_preds, labels, _, _ = evaluate_model(model, val_loader, split_name="VAL", args=args)
    eval_preds = eval_preds.float().to(device)
    labels = labels.float().to(device)

    labels_lst = labels.tolist()
    eval_preds_lst = eval_preds.tolist()
    report = gen_report(labels_lst, eval_preds_lst)
    logi(f"Evaluation Report for Epoch {epoch}:\n{report}")

    current_acc = accuracy_score(labels_lst, eval_preds_lst)

    is_best = current_acc > best_acc
    if is_best:
        best_acc = current_acc
        save_path = os.path.join(args.output_path, "model_best")
        model.save_pretrained(save_path)
        logi(f"Model saved at epoch {epoch} with Acc {current_acc:.4f}")

    logi(f"Current Acc: {current_acc:.4f}")
    logi(f"Best Acc so far: {best_acc:.4f}")

    report_save_path = os.path.join(args.output_path, f"report_val_epoch_{epoch}.txt")
    with open(report_save_path, "w") as f:
        f.write(f"Epoch {epoch} Evaluation Report:\n{report}\n")
        f.write(f"Best Acc so far: {best_acc:.4f}\n")
        f.write(f"平均 Loss: {global_avg:.4f}\n")

    return best_acc


def train_and_evaluate(args, model, train_loader, train_text_only_loader, train_img_only_loader, val_loader,
                       lr_scheduler, true_label_id):
    best_acc = 0.0

    # 保存args
    args_save_path = os.path.join(args.output_path, "args.json")
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logi(f"Arguments saved to {args_save_path}")
    device = torch.cuda.current_device()

    # 构建传递给模型的额外 kwargs（用于 purification 等功能）
    model_kwargs = {}
    if hasattr(args, 'purification_threshold'):
        model_kwargs['purification_threshold'] = args.purification_threshold
        logi(f"[Model Kwargs] purification_threshold={args.purification_threshold}")

    for epoch in range(args.epochs):
        # Set epoch for all three samplers
        # 放到一个列表里遍历
        all_loaders = [train_loader, train_text_only_loader, train_img_only_loader]

        for loader in all_loaders:
            g = torch.Generator()
            g.manual_seed(args.seed + epoch)
            loader.sampler.generator = g

        epoch_start = time.time()

        # 根据 train_type 决定执行哪些阶段（需要先定义，因为后面会用到）
        train_type = getattr(args, 'train_type', 'mix').lower()
        if train_type == 'multi':
            active_phases = ['full']
            phase_names = ["full"]
        elif train_type == 'img_only':
            active_phases = ['img_only']
            phase_names = ["img_only"]
        elif train_type == 'text_only':
            active_phases = ['text_only']
            phase_names = ["text_only"]
        else:  # mix 或默认
            active_phases = ['full', 'text_only', 'img_only']
            phase_names = ["full", "text_only", "img_only"]

        logi(f"[Train] train_type={train_type}, active_phases={active_phases}")

        # per‑phase time accumulators (只初始化活跃阶段)
        phase_time_sums = {p: {"data": 0.0, "step": 0.0, "infer": 0.0} for p in active_phases}
        # per‑phase window accumulators for arbitrary loss components
        loss_sums_window = {p: defaultdict(float) for p in active_phases}
        loss_counts_window = {p: defaultdict(int) for p in active_phases}
        # 追踪窗口内实际步数（避免第一次 log 时计算错误）
        phase_steps_in_window = {p: 0 for p in active_phases}
        last_time = time.time()
        model.train()
        epoch_loss = 0.0

        # 根据活跃阶段计算每个epoch的微步数
        # 在mix模式下，依次遍历img、text、multi三个loader，所以是三个loader长度之和
        # 在单一模式下，只遍历对应的loader一次
        if _uses_mix_schedule(train_type):
            # mix 模式族：依次遍历 img_only、text_only、full 三个 loader
            micro_steps_per_epoch = (
                    len(train_img_only_loader) +
                    len(train_text_only_loader) +
                    len(train_loader)
            )
        elif train_type == 'multi':
            micro_steps_per_epoch = len(train_loader)
        elif train_type == 'text_only':
            micro_steps_per_epoch = len(train_text_only_loader)
        elif train_type == 'img_only':
            micro_steps_per_epoch = len(train_img_only_loader)
        else:
            micro_steps_per_epoch = len(train_loader) * len(active_phases)

        updates_per_epoch = math.ceil(micro_steps_per_epoch / max(1, args.gradient_accumulation_steps))
        updates_done = 0

        pbar = tqdm(
            total=updates_per_epoch,
            desc=f"Epoch {epoch + 1} [updates] gacc={args.gradient_accumulation_steps} train_type={train_type}",
            dynamic_ncols=True,
        )

        micro_step_idx = 0  # count micro-steps across phases
        gacc_loss_sum = 0.0
        gacc_loss_count = 0

        # mix 模式族：依次遍历 img、text、multi 三个 loader
        if _uses_mix_schedule(train_type):
            # 定义遍历顺序：img_only -> text_only -> full
            phase_loader_pairs = [
                ('img_only', train_img_only_loader),
                ('text_only', train_text_only_loader),
                ('full', train_loader)
            ]

            for phase_name, current_loader in phase_loader_pairs:
                logi(f"[Mix Mode] Starting phase: {phase_name}, loader_length={len(current_loader)}")

                # 为当前phase创建迭代器
                
                iter_current = iter(current_loader)

                for step in range(len(current_loader)):
                    
                    # 获取当前阶段的batch
                    try:
                        batch = next(iter_current)
                    except StopIteration:
                        iter_current = iter(current_loader)
                        batch = next(iter_current)

                    # 解析batch
                    inputs, _, label_ids = batch

                    inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                              for k, v in inputs.items()}
                    label_ids = label_ids.cuda()

                    now = time.time()
                    data_time = now - last_time

                    t_infer_start = time.time()
                    outputs = model(**inputs, labels=label_ids, **model_kwargs)
                    loss_items = extract_loss_items(outputs)
                    for k, v in loss_items.items():
                        try:
                            val = float(v.detach().to(torch.float32).item()) if torch.is_tensor(v) else float(v)
                            loss_sums_window[phase_name][k] += val
                            loss_counts_window[phase_name][k] += 1
                        except Exception:
                            pass
                    loss = loss_items.get("loss", outputs.loss)

                    now = time.time()
                    t_infer_time = now - t_infer_start

                    t_step_start = now
                    model.backward(loss)

                    model.step()

                    micro_step_idx += 1
                    try:
                        boundary = bool(getattr(model, "is_gradient_accumulation_boundary", lambda: False)())
                    except Exception:
                        boundary = ((micro_step_idx) % max(1, args.gradient_accumulation_steps) == 0)
                    gacc_loss_sum += loss.item()
                    gacc_loss_count += 1
                    if boundary or (micro_step_idx == micro_steps_per_epoch):
                        updates_done += 1
                        pbar.update(1)
                    if gacc_loss_count == args.gradient_accumulation_steps:
                        epoch_loss += (gacc_loss_sum / gacc_loss_count)
                        gacc_loss_sum = 0.0
                        gacc_loss_count = 0

                    t_step_end = time.time()
                    step_time = t_step_end - t_step_start
                    phase_time_sums[phase_name]["data"] += data_time
                    phase_time_sums[phase_name]["step"] += step_time
                    phase_time_sums[phase_name]["infer"] += t_infer_time
                    phase_steps_in_window[phase_name] += 1
                    last_time = t_step_end

                    # ===== Logging for current phase =====
                    if step % args.log_interval == 0:
                        mb_per_gpu = getattr(args, 'train_micro_batch_size_per_gpu', None)
                        per_gpu_samples = mb_per_gpu if mb_per_gpu is not None else getattr(current_loader,
                                                                                            'batch_size', 0)
                        gacc = max(1, args.gradient_accumulation_steps)
                        dp_world = max(1, args.data_parallel_size)
                        global_micro_bsz = (per_gpu_samples or 0) * dp_world
                        window_steps = max(1, args.log_interval)

                        # 当前phase的统计
                        dt_sum = phase_time_sums[phase_name]["data"]
                        st_sum = phase_time_sums[phase_name]["step"]
                        inf_sum = phase_time_sums[phase_name]["infer"]

                        # 使用实际步数而非固定 window_steps，避免第一次 log 时计算错误
                        actual_steps = phase_steps_in_window[phase_name]
                        actual_steps = max(1, actual_steps)  # 防止除零

                        avg_step = st_sum / actual_steps
                        avg_data = dt_sum / actual_steps
                        avg_infer = inf_sum / actual_steps
                        avg_step = max(1e-12, avg_step)

                        # loss组成
                        loss_comp_str = ""
                        phase_loss_sums = loss_sums_window[phase_name]
                        phase_loss_counts = loss_counts_window[phase_name]
                        if len(phase_loss_sums) > 0:
                            parts = []
                            for k in sorted(phase_loss_sums.keys()):
                                cnt = max(1, phase_loss_counts.get(k, window_steps))
                                parts.append(f"{k}={phase_loss_sums[k] / cnt:.4f}")
                            loss_comp_str = " " + " ".join(parts)

                        micro_s = 1.0 / avg_step if avg_step > 0 else 0.0
                        samples_per_s_global = global_micro_bsz * micro_s
                        updates_per_s = micro_s / gacc if gacc > 0 else 0.0
                        time_per_update = 1.0 / max(1e-12, updates_per_s) if updates_per_s > 0 else 0.0
                        eff_bsz_per_update = global_micro_bsz * gacc

                        try:
                            cur_lr = optimizer.param_groups[0]['lr']
                        except Exception:
                            try:
                                cur_lr = model.optimizer.param_groups[0]['lr']
                            except Exception:
                                cur_lr = -1

                        epoch_avg_loss = epoch_loss / (updates_done if updates_done > 0 else 1)
                        eff_bsz_desc = f"{eff_bsz_per_update} (= {per_gpu_samples}×{dp_world}×gacc{gacc})"

                        logi(
                            f"[Rank0] Ep {epoch + 1} | phase={phase_name} | step={step}/{len(current_loader)} | "
                            f"upd {updates_done}/{updates_per_epoch} (micro_step {micro_step_idx}/{micro_steps_per_epoch}) | "
                            f"loss(epoch_avg)={epoch_avg_loss:.4f} | "
                            f"{phase_name}: data/step={avg_data:.3f}s step={avg_step:.3f}s infer={avg_infer:.3f}s{loss_comp_str} | "
                            f"micro/s={micro_s:.2f} | upd/s={updates_per_s:.2f} | s/upd={time_per_update:.2f} | "
                            f"samples/s(global)={samples_per_s_global:.1f} | eff_bsz/upd={eff_bsz_desc} | lr={cur_lr:.6g}"
                        )

                        # 清空当前phase的窗口统计量
                        phase_time_sums[phase_name]["data"] = 0.0
                        phase_time_sums[phase_name]["step"] = 0.0
                        phase_time_sums[phase_name]["infer"] = 0.0
                        phase_steps_in_window[phase_name] = 0
                        loss_sums_window[phase_name].clear()
                        loss_counts_window[phase_name].clear()

                logi(f"[Mix Mode] Completed phase: {phase_name}")
        else:
            # 非mix模式：保持原有逻辑
            # 创建迭代器
            iter_full = iter(train_loader) if 'full' in active_phases else None
            iter_text = iter(train_text_only_loader) if 'text_only' in active_phases else None
            iter_img = iter(train_img_only_loader) if 'img_only' in active_phases else None

            if train_type == 'multi':
                max_steps = len(train_loader)
            elif train_type == 'text_only':
                max_steps = len(train_text_only_loader)
            elif train_type == 'img_only':
                max_steps = len(train_img_only_loader)
            else:
                max_steps = len(train_loader)

            for step in range(max_steps):
                # 根据 active_phases 动态执行阶段
                for phase_name in active_phases:
                    # 获取对应阶段的batch
                    inputs, label_ids, iter_full, iter_text, iter_img = _get_batch_for_phase(
                        phase_name, iter_full, iter_text, iter_img,
                        train_loader, train_text_only_loader, train_img_only_loader
                    )

                    inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                              for k, v in inputs.items()}
                    label_ids = label_ids.cuda()

                    now = time.time()
                    data_time = now - last_time

                    t_infer_start = time.time()
                    outputs = model(**inputs, labels=label_ids, **model_kwargs)
                    loss_items = extract_loss_items(outputs)
                    for k, v in loss_items.items():
                        try:
                            val = float(v.detach().to(torch.float32).item()) if torch.is_tensor(v) else float(v)
                            loss_sums_window[phase_name][k] += val
                            loss_counts_window[phase_name][k] += 1
                        except Exception:
                            pass
                    loss = loss_items.get("loss", outputs.loss)

                    now = time.time()
                    t_infer_time = now - t_infer_start

                    t_step_start = now
                    model.backward(loss)

                    model.step()

                    micro_step_idx += 1
                    try:
                        boundary = bool(getattr(model, "is_gradient_accumulation_boundary", lambda: False)())
                    except Exception:
                        boundary = ((micro_step_idx) % max(1, args.gradient_accumulation_steps) == 0)
                    gacc_loss_sum += loss.item()
                    gacc_loss_count += 1
                    if boundary or (micro_step_idx == micro_steps_per_epoch):
                        updates_done += 1
                        pbar.update(1)
                    if gacc_loss_count == args.gradient_accumulation_steps:
                        epoch_loss += (gacc_loss_sum / gacc_loss_count)
                        gacc_loss_sum = 0.0
                        gacc_loss_count = 0

                    t_step_end = time.time()
                    step_time = t_step_end - t_step_start
                    phase_time_sums[phase_name]["data"] += data_time
                    phase_time_sums[phase_name]["step"] += step_time
                    phase_time_sums[phase_name]["infer"] += t_infer_time
                    phase_steps_in_window[phase_name] += 1
                    last_time = t_step_end

                # ===== Combined logging for this window over all three phases =====
                if step % args.log_interval == 0:
                    mb_per_gpu = getattr(args, 'train_micro_batch_size_per_gpu', None)
                    per_gpu_samples = mb_per_gpu if mb_per_gpu is not None else getattr(train_loader, 'batch_size', 0)
                    gacc = max(1, args.gradient_accumulation_steps)
                    dp_world = max(1, args.data_parallel_size)
                    global_micro_bsz = (per_gpu_samples or 0) * dp_world
                    window_steps = max(1, args.log_interval)

                    phase_logs = []
                    total_step_time_sum = 0.0
                    total_micro_steps_in_window = 0

                    for pname in phase_names:
                        dt_sum = phase_time_sums[pname]["data"]
                        st_sum = phase_time_sums[pname]["step"]
                        inf_sum = phase_time_sums[pname]["infer"]

                        # 使用实际步数而非固定 window_steps
                        actual_steps = phase_steps_in_window[pname]
                        actual_steps = max(1, actual_steps)

                        # 该 phase 的平均时间（按实际步数做平均）
                        avg_step = st_sum / actual_steps
                        avg_data = dt_sum / actual_steps
                        avg_infer = inf_sum / actual_steps

                        avg_step = max(1e-12, avg_step)

                        # 该 phase 的 loss 组成（按累计次数平均）
                        loss_comp_str = ""
                        phase_loss_sums = loss_sums_window[pname]
                        phase_loss_counts = loss_counts_window[pname]
                        if len(phase_loss_sums) > 0:
                            parts = []
                            for k in sorted(phase_loss_sums.keys()):
                                cnt = max(1, phase_loss_counts.get(k, window_steps))
                                parts.append(f"{k}={phase_loss_sums[k] / cnt:.4f}")
                            loss_comp_str = " " + " ".join(parts)

                        phase_logs.append(
                            f"{pname}: data/step={avg_data:.3f}s step={avg_step:.3f}s infer={avg_infer:.3f}s{loss_comp_str}"
                        )

                        total_step_time_sum += st_sum
                        total_micro_steps_in_window += actual_steps

                        # 清空该 phase 的窗口统计量
                        phase_time_sums[pname]["data"] = 0.0
                        phase_time_sums[pname]["step"] = 0.0
                        phase_time_sums[pname]["infer"] = 0.0
                        phase_steps_in_window[pname] = 0
                        loss_sums_window[pname].clear()
                        loss_counts_window[pname].clear()

                    # 计算整体的 micro/s 和 samples/s（把三种 phase 都视作 micro‑step）
                    if total_micro_steps_in_window > 0 and total_step_time_sum > 0:
                        avg_step_overall = total_step_time_sum / total_micro_steps_in_window
                        avg_step_overall = max(1e-12, avg_step_overall)
                        micro_s = 1.0 / avg_step_overall
                    else:
                        avg_step_overall = 0.0
                        micro_s = 0.0

                    samples_per_s_global = global_micro_bsz * micro_s
                    updates_per_s = micro_s / gacc if gacc > 0 else 0.0
                    time_per_update = 1.0 / max(1e-12, updates_per_s) if updates_per_s > 0 else 0.0
                    eff_bsz_per_update = global_micro_bsz * gacc

                    try:
                        cur_lr = optimizer.param_groups[0]['lr']
                    except Exception:
                        try:
                            cur_lr = model.optimizer.param_groups[0]['lr']
                        except Exception:
                            cur_lr = -1

                    epoch_avg_loss = epoch_loss / (updates_done if updates_done > 0 else 1)

                    eff_bsz_desc = f"{eff_bsz_per_update} (= {per_gpu_samples}×{dp_world}×gacc{gacc})"
                    phase_logs_str = " | ".join(phase_logs)
                    logi(
                        f"[Rank0] Ep {epoch + 1} | step={step} upd {updates_done}/{updates_per_epoch} "
                        f"(micro_step {micro_step_idx}/{micro_steps_per_epoch}) | "
                        f"loss(epoch_avg)={epoch_avg_loss:.4f} | {phase_logs_str} | "
                        f"micro/s(overall)={micro_s:.2f} | upd/s={updates_per_s:.2f} | s/upd={time_per_update:.2f} | "
                        f"samples/s(global)={samples_per_s_global:.1f} | eff_bsz/upd={eff_bsz_desc} | lr={cur_lr:.6g}"
                    )

        pbar.close()
        epoch_dur = time.time() - epoch_start
        try:
            max_alloc = bytes_gb(torch.cuda.max_memory_allocated())
            max_reserved = bytes_gb(torch.cuda.max_memory_reserved())
        except Exception:
            max_alloc = max_reserved = 0.0
        mb_per_gpu = getattr(args, 'train_micro_batch_size_per_gpu', None)
        # 根据活跃阶段计算步数
        steps_per_epoch = (
                (len(train_loader) if 'full' in active_phases else 0) +
                (len(train_text_only_loader) if 'text_only' in active_phases else 0) +
                (len(train_img_only_loader) if 'img_only' in active_phases else 0)
        )
        logi(
            f"[Rank0] Epoch {epoch + 1} done in {epoch_dur:.1f}s | steps={steps_per_epoch} | "
            f"max_mem_alloc={max_alloc:.2f}GB reserved={max_reserved:.2f}GB | mb_per_gpu={mb_per_gpu}"
        )
        torch.cuda.reset_peak_memory_stats()
        total_loss_tensor = torch.tensor(epoch_loss, device='cuda')
        total_steps = (
                (len(train_loader) if 'full' in active_phases else 0) +
                (len(train_text_only_loader) if 'text_only' in active_phases else 0) +
                (len(train_img_only_loader) if 'img_only' in active_phases else 0)
        )
        global_avg = total_loss_tensor.item() / (dist.get_world_size() * total_steps)
        if dist.get_rank() == 0:
            logi(f"[Rank 0] Epoch {epoch + 1} 全局平均 Loss: {global_avg:.4f}")
        # 运行评估并保存最佳模型
        best_acc = _run_evaluation_and_save(model, val_loader, args, epoch + 1, best_acc, device, global_avg)


def log_numeric_fingerprint(args):
    import torch, os, platform
    try:
        import flash_attn
        fa_ver = flash_attn.__version__
    except Exception:
        fa_ver = None
    attn_impl_sel = _normalize_attn_impl(getattr(args, "attn_impl", "sdpa"))
    fp = {
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "attn_impl": attn_impl_sel,
        "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdp_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdp_math": torch.backends.cuda.math_sdp_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "PYTORCH_FORCE_DISABLE_FUSED_SDPA": os.environ.get("PYTORCH_FORCE_DISABLE_FUSED_SDPA", None),
        "flash_attn_version": fa_ver,
        "os": f"{platform.system()} {platform.release()}",
    }
    for k, v in fp.items():
        logi(f"[numeric] {k}={v}")


def update_args_from_var(args):
    args.data_parallel_size = args.n_gpus

    if args.data_parallel_size > 1:
        raise NotImplementedError(
            f"Multi-GPU training is not yet implemented!\n"
        )

    args.train_micro_batch_size_per_gpu = args.batch_size // args.data_parallel_size // args.gradient_accumulation_steps
    assert args.batch_size % args.train_micro_batch_size_per_gpu == 0
    args.output_path = os.path.join(
        args.output_path,
        f"bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.epochs}_dp{args.data_parallel_size}_gacc{args.gradient_accumulation_steps}_clip{args.clip_grad}"
        + (
            f"_loraTrue_r{args.lora_r}_a{args.lora_alpha}_d{args.lora_dropout}" if args.lora else "_loraFalse")
    )
    # In test-only mode, keep/use the directory the user intends (typically LoRA path without trailing 'model_best').
    if getattr(args, "test_only", False):
        # If lora_path is provided, strip trailing 'model_best' and use its parent as the output/log directory.
        lp = getattr(args, "lora_path", None)
        if lp:
            lp = lp.rstrip("/")
            base_dir = os.path.dirname(lp) if os.path.basename(lp) == "model_best" else lp
            args.output_path = base_dir
        args.output_path = os.path.join(args.output_path, "test_only")
        if getattr(args, "test_set", False):
            args.output_path = os.path.join(args.output_path, f"test_for_set_{args.test_set}")
            args.test_data_path = args.train_data_path if args.test_set == "train" else (
                args.val_data_path if args.test_set == "val" else args.test_data_path)
    if getattr(args, "prompt_version", False):
        args.output_path = os.path.join(args.output_path, f"prompt_{args.prompt_version}")
    if getattr(args, "text_only", False):
        args.output_path = os.path.join(args.output_path, "text_only")
    elif getattr(args, "img_only", False):
        args.output_path = os.path.join(args.output_path, "img_only")

    os.makedirs(args.output_path, exist_ok=True)


def load_model_for_eval(args, use_lora: Optional[bool] = None):
    """Load the evaluation model.

    - use_lora=True: load base model + LoRA adapter from model_best / lora_path
    - use_lora=False: load the plain base checkpoint
    """
    ModelCls = _resolve_qwen_vl_model_class(args.model_type)
    attn_impl = _normalize_attn_impl(getattr(args, "attn_impl", "sdpa"))
    logi(f"[Model(EVAL)] Using attention implementation: {attn_impl}")
    apply_lora = args.lora if use_lora is None else bool(use_lora)
    model_source_path = args.model_path
    if not apply_lora and not getattr(args, "test_only", False):
        full_ckpt_path = os.path.join(args.output_path, "model_best")
        if os.path.isdir(full_ckpt_path):
            model_source_path = full_ckpt_path
    logi(f"[Model(EVAL)] Loading base checkpoint from: {model_source_path}")
    base_model = ModelCls.from_pretrained(
        model_source_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    base_model.config.fake_token_id = args.fake_id
    base_model.config.real_token_id = args.real_id

    if apply_lora:
        lora_path = args.lora_path or os.path.join(args.output_path, "model_best")
        logi(f"[Model(EVAL)] Loading LoRA adapter from: {lora_path}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=False
        )
        log_peft_summary(lora_model, None)
        lora_model.eval()
        return lora_model

    base_model.eval()
    return base_model


def get_best_model_with_lora(args):
    return load_model_for_eval(args, use_lora=bool(args.lora))


def _get_eval_artifact_paths(output_dir: str, tag: str) -> Dict[str, str]:
    safe_tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", tag)
    if safe_tag == "final":
        return {
            "report": os.path.join(output_dir, "report_test_final.txt"),
            "labels": os.path.join(output_dir, "labels_test_final.json"),
            "preds": os.path.join(output_dir, "preds_test_final.json"),
            "pairs": os.path.join(output_dir, "test_final_pairs.jsonl"),
            "debug": os.path.join(output_dir, "debug_text_test_final.txt"),
        }
    return {
        "report": os.path.join(output_dir, f"report_test_{safe_tag}.txt"),
        "labels": os.path.join(output_dir, f"labels_test_{safe_tag}.json"),
        "preds": os.path.join(output_dir, f"preds_test_{safe_tag}.json"),
        "pairs": os.path.join(output_dir, f"test_{safe_tag}_pairs.jsonl"),
        "debug": os.path.join(output_dir, f"debug_text_test_{safe_tag}.txt"),
    }


def _save_eval_outputs(output_dir: str, tag: str, report: str,
                       labels_lst: List[float], eval_preds_lst: List[float],
                       gen_text_all: Optional[List[str]] = None,
                       input_text_all: Optional[List[str]] = None):
    paths = _get_eval_artifact_paths(output_dir, tag)
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(f"Final Evaluation Report:\n{report}\n")

    labels_lst_int = [int(x) for x in labels_lst]
    preds_lst_int = [int(x) for x in eval_preds_lst]

    with open(paths["labels"], "w", encoding="utf-8") as f:
        json.dump(labels_lst_int, f, indent=4)
    with open(paths["preds"], "w", encoding="utf-8") as f:
        json.dump(preds_lst_int, f, indent=4)
    with open(paths["pairs"], "w", encoding="utf-8") as f:
        for lab, pr in zip(labels_lst_int, preds_lst_int):
            f.write(json.dumps({"label": lab, "pred": pr}) + "\n")

    logi(
        f"[Eval:{tag}] Saved labels to {paths['labels']}, preds to {paths['preds']}, "
        f"pairs to {paths['pairs']}"
    )

    if gen_text_all is not None and input_text_all is not None and len(gen_text_all) == len(labels_lst):
        with open(paths["debug"], "w", encoding="utf-8") as f:
            for idx in range(len(gen_text_all)):
                f.write("[sample " + str(idx + 1) + "]\n")
                f.write("-" * 10 + "\n")
                f.write("input text: " + str(input_text_all[idx]) + "\n")
                f.write("-" * 10 + "\n")
                f.write("generated text: " + str(gen_text_all[idx]) + "\n")
                f.write("-" * 10 + "\n")
                f.write("label: " + str(labels_lst[idx]) + "\n")
                f.write("-" * 10 + "\n")
                f.write("pred: " + str(eval_preds_lst[idx]) + "\n")
                f.write("-" * 10 + "\n")
                f.write("*" * 50 + "\n")
        logi(f"[Eval:{tag}] Saved generated text (for debugging) to {paths['debug']}")


def run_final_evaluation(model, test_loader, args, tag: str = "final", eval_mode: Optional[str] = None):
    device = torch.cuda.current_device()
    mode = (eval_mode or "token").lower()
    eval_preds, labels, gen_text, input_text = evaluate_model(
        model, test_loader, split_name=f"TEST:{tag}", args=args, eval_mode=mode
    )
    eval_preds = eval_preds.float().to(device)
    labels = labels.float().to(device)

    labels_lst = labels.tolist()
    eval_preds_lst = eval_preds.tolist()
    gen_text_all = _flatten_text_chunks(gen_text)
    input_text_all = _flatten_text_chunks(input_text)

    logi(
        f"[Eval:{tag}] mode={mode} prompt={getattr(args, 'prompt_version', None)} "
        f"| labels={len(labels_lst)} preds={len(eval_preds_lst)}"
    )
    report = gen_report(labels_lst, eval_preds_lst)
    acc = accuracy_score(labels_lst, eval_preds_lst)
    logi(f"[Eval:{tag}] Final Evaluation Report:\n{report}")
    logi(f"[Eval:{tag}] Accuracy: {acc:.4f}")

    _save_eval_outputs(
        args.output_path,
        tag,
        report,
        labels_lst,
        eval_preds_lst,
        gen_text_all=gen_text_all,
        input_text_all=input_text_all,
    )
    return {
        "tag": tag,
        "accuracy": float(acc),
        "eval_mode": mode,
        "prompt_version": getattr(args, "prompt_version", None),
    }


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    initialize(args)
    update_args_from_var(args)
    rank = dist.get_rank() if dist.is_initialized() else 0

    LOGGER = setup_logger(args.output_path, rank)
    logi(f"Logger initialized. Logs will be written to {os.path.join(args.output_path, f'train_rank{rank}.log')}")

    try:
        mp.set_start_method("spawn", force=True)
        logi("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logi(f"Unable to set start method to 'spawn', continuing with current method.")

    ds_config = setup_deepspeed_config(args)
    log_system_snapshot(args, ds_config)
    log_numeric_fingerprint(args)
    try:
        logi(
            f"DeepSpeed cfg: micro_bsz/gpu={ds_config.get('train_micro_batch_size_per_gpu')} | gacc={args.gradient_accumulation_steps} | clip={args.clip_grad} | zero={ds_config.get('zero_optimization')}")
    except Exception:
        pass

    train_loader, train_text_only_loader, train_img_only_loader, test_loader, val_loader, true_label_id = prepare_dataloaders(
        args, ds_config)

    args.fake_id = train_loader.collate_fn.false_id
    args.real_id = train_loader.collate_fn.true_id

    if not args.test_only:
        model, optimizer, lr_scheduler = build_training_components(args, ds_config)
        train_and_evaluate(args, model, train_loader, train_text_only_loader, train_img_only_loader, val_loader,
                           lr_scheduler, true_label_id)
        del model
        del optimizer
        del lr_scheduler
        torch.cuda.empty_cache()

    eval_branch = getattr(args, "eval_branch", "current")
    device = torch.cuda.current_device()

    if eval_branch == "current":
        eval_model = get_best_model_with_lora(args)
        eval_loader = test_loader
        eval_args = args
        eval_tag = "final"
    elif eval_branch == "base":
        eval_model = load_model_for_eval(args, use_lora=False)
        base_prompt_version = _resolve_non_cot_prompt_version(args.prompt_version, en=args.en)
        eval_loader, eval_args = get_test_loader_for_prompt(
            args, ds_config, test_loader, base_prompt_version
        )
        eval_tag = "base"
    elif eval_branch == "cot":
        eval_model = load_model_for_eval(args, use_lora=False)
        cot_prompt_version = _resolve_cot_prompt_version(args.prompt_version, en=args.en)
        eval_loader, eval_args = get_test_loader_for_prompt(
            args, ds_config, test_loader, cot_prompt_version
        )
        eval_tag = "cot"
    else:
        raise ValueError(f"Unsupported eval_branch: {eval_branch}")

    eval_model.to(device)
    run_final_evaluation(
        eval_model,
        eval_loader,
        eval_args,
        tag=eval_tag,
        eval_mode=_resolve_eval_mode(eval_branch),
    )

    del eval_model
    torch.cuda.empty_cache()


    if dist.is_initialized():
        dist.destroy_process_group()
