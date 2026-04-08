

import argparse
import os
import torch
import numpy as np
import traceback

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from data.utils.vlm_datasets import VLMDataset
from src.utils.utils import *
from src.utils.collate import CollateFn
from transformers import AutoProcessor
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, DistributedSampler, Subset, Sampler
import json
import sys
import torch.distributed as dist
from typing import Optional
import deepspeed
from peft import LoraConfig, get_peft_model
import math
import platform
import importlib
import torch.multiprocessing as mp
from copy import deepcopy

try:
    import psutil  # optional
except Exception:
    psutil = None
import logging
from logging.handlers import RotatingFileHandler

LOGGER: logging.Logger | None = None


# ============================================================================
# Logger setup
# ============================================================================
class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def setup_logger(output_dir: str, rank: int) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"train.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s [rank %(rank)d] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fpath = os.path.join(output_dir, f"probing_rank{rank}.log")
        fh = RotatingFileHandler(fpath, maxBytes=20 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.addFilter(_RankFilter(rank))
        logger.addHandler(fh)
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


# ============================================================================
# Model and attention implementation utilities
# ============================================================================
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


# ============================================================================
# PEFT / LoRA logging
# ============================================================================
def log_peft_summary(model, peft_cfg: Optional[LoraConfig] | None = None, header: str = "[PEFT]"):
    """Log which modules were wrapped by LoRA and which params are trainable."""
    try:
        lora_wrapped = []
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                lora_wrapped.append(name)

        trainable = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable.append(n)

        planned = []
        if peft_cfg is not None:
            try:
                planned = list(peft_cfg.target_modules) if peft_cfg.target_modules is not None else []
            except Exception:
                planned = []

        matched_by_keyword = {}
        for kw in planned:
            hit = [m for m in lora_wrapped if kw in m]
            if hit:
                matched_by_keyword[kw] = hit

        logi("==== PEFT / LoRA Summary ====")
        if planned:
            logi(f"Planned target modules (keywords): {planned}")
        logi(f"LoRA-wrapped module count: {len(lora_wrapped)}")
        if lora_wrapped:
            preview = lora_wrapped[:20]
            more = "" if len(lora_wrapped) <= 20 else f" ... (+{len(lora_wrapped) - 20} more)"
            logi(f"Wrapped modules (preview): {preview}{more}")
        if matched_by_keyword:
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


# ============================================================================
# System logging helpers
# ============================================================================
def _env(key: str, default: str = "-") -> str:
    return os.environ.get(key, default)


def bytes_gb(x: int) -> float:
    return x / (1024 ** 3)


def log_system_snapshot(args, ds_config):
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        rank, world = 0, 1
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
    cuda_count = torch.cuda.device_count()
    cur_dev = torch.cuda.current_device() if cuda_count > 0 else -1
    try:
        gpu_name = torch.cuda.get_device_name(cur_dev) if cur_dev >= 0 else "cpu"
        total_mem = torch.cuda.get_device_properties(cur_dev).total_memory if cur_dev >= 0 else 0
    except Exception:
        gpu_name, total_mem = "unknown", 0
    intra = torch.get_num_threads()
    inter = torch.get_num_interop_threads()
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


# ============================================================================
# Custom Sampler for uneven data splits
# ============================================================================
class CustomSampler(Sampler):
    def __init__(self, data_source, rank, num_replicas):
        self.data_source = data_source
        self.rank = rank
        self.num_replicas = num_replicas
        self.range_lst = []

    def __iter__(self):
        total = len(self.data_source)
        per_rank = math.ceil(total / self.num_replicas)
        start = self.rank * per_rank
        end = min(start + per_rank, total)
        range_lst = list(range(start, end))
        self.range_lst = range_lst
        return iter(range_lst)

    def __len__(self):
        total = len(self.data_source)
        per_rank = math.ceil(total / self.num_replicas)
        start = self.rank * per_rank
        end = min(start + per_rank, total)
        return max(0, end - start)


# ============================================================================
# Model loading
# ============================================================================
def get_model(args):
    """Load model and optionally apply LoRA."""
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

    # [PPB] Mask 由 CollateFn 在 CPU 侧预生成，模型无需调用 set_anchor_ids

    return model


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    parser = get_parser()
    parser.add_argument("--train_data_path", type=str, help="Path to the training file")

    # [Probing] Additional parameters
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="[Probing] Threshold for head role assignment")
    parser.add_argument("--feature_weight", type=float, default=0.3,
                        help="[Probing] Feature weight in combined score (0-1)")
    parser.add_argument("--en", action="store_true", help="english mode, use 'Real' and 'Fake' as labels")
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attn_2", "flash_attention_2", "fa2", "eager"],
        help="Attention backend: 'sdpa' for fully reproducible runs; "
             "'flash_attention_2' (aka 'fa2') for speed with small numerical differences."
    )
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                        help="Min pixels (e.g. 256*28*28 for Qwen2.5, 256*32*32 for Qwen3)")
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28, help="Max pixels")
    parser.add_argument("--fake_id", type=int, default=1, help="Fake label ID")
    parser.add_argument("--real_id", type=int, default=1, help="Real label ID")
    parser.add_argument("--unimodal_ratio", type=float, default=0.1, help="Unimodal label ratio")
    parser.add_argument("--use_multimodal", action="store_true", 
                        help="[Probing] Also collect from multimodal data")
    args = parser.parse_args()
    return args


# ============================================================================
# DeepSpeed config
# ============================================================================
def setup_deepspeed_config(args):
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    
    # [Probing Mode] Disable ZeRO optimization since we don't need optimizer
    # Only use DeepSpeed for model wrapping and large model loading
    if "zero_optimization" in ds_config:
        ds_config["zero_optimization"]["stage"] = 0
        logi("⚠️  [Probing] Disabled ZeRO optimization (stage -> 0) for inference-only mode")
    
    args.deepspeed_config = None
    return ds_config


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def prepare_dataloaders(args, ds_config):
    """Prepare dataloaders for probing: img_only, text_only, and optionally multi."""
    g = torch.Generator()
    g.manual_seed(args.seed)
    mp_ctx = mp.get_context("spawn") if args.num_workers > 0 else None

    # 1. Load full training dataset
    args.data_path = args.train_data_path
    train_dataset = VLMDataset(args)

    # 2. Prepare Text-Only and Img-Only args
    args_text_only = deepcopy(args)
    args_text_only.text_only = True
    args_text_only.img_only = False

    args_img_only = deepcopy(args)
    args_img_only.img_only = True
    args_img_only.text_only = False

    # 3. Create mode-specific datasets
    _full_text_ds = VLMDataset(args_text_only)
    _full_img_ds = VLMDataset(args_img_only)

    # 4. Select subset based on unimodal_ratio
    total_len = len(train_dataset)
    if hasattr(args, 'unimodal_ratio') and args.unimodal_ratio > 0:
        subset_size = int(total_len * args.unimodal_ratio)
    else:
        subset_size = 0

    logi(f"Using fixed subset: Top {subset_size} samples for unimodal probing.")
    subset_indices = list(range(subset_size))

    train_text_only_dataset = Subset(_full_text_ds, subset_indices)
    train_img_only_dataset = Subset(_full_img_ds, subset_indices)

    # 6. Processor and CollateFn
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code,
                                              padding_side="left", use_fast=True, min_pixels=args.min_pixels,
                                              max_pixels=args.max_pixels)
    collate_fn = CollateFn(args, processor, cn=False if args.en else True)
    collate_fn_text_only = CollateFn(args_text_only, processor, cn=False if args.en else True)

    dp_world_size = args.data_parallel_size
    dp_rank = dist.get_rank() % dp_world_size

    # 7. Create samplers
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True, drop_last=False, rank=dp_rank, num_replicas=dp_world_size, seed=args.seed
    )
    text_sampler = DistributedSampler(
        train_text_only_dataset, shuffle=True, drop_last=False, rank=dp_rank, num_replicas=dp_world_size, seed=args.seed
    )
    img_sampler = DistributedSampler(
        train_img_only_dataset, shuffle=True, drop_last=False, rank=dp_rank, num_replicas=dp_world_size, seed=args.seed
    )

    # 8. Create dataloaders
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=ds_config["train_micro_batch_size_per_gpu"],
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False, prefetch_factor=None if args.num_workers == 0 else 4,
        multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
    )

    train_text_only_loader = DataLoader(
        train_text_only_dataset, sampler=text_sampler, batch_size=ds_config["train_micro_batch_size_per_gpu"],
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_text_only,
        persistent_workers=False, prefetch_factor=None if args.num_workers == 0 else 4,
        multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
    )

    train_img_only_loader = DataLoader(
        train_img_only_dataset, sampler=img_sampler, batch_size=ds_config["train_micro_batch_size_per_gpu"],
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False, prefetch_factor=None if args.num_workers == 0 else 4,
        multiprocessing_context=mp_ctx, worker_init_fn=seed_worker, generator=g
    )

    true_label_id = collate_fn.true_id
    
    # Log dataset sizes
    try:
        logi(
            f"Datasets — train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} | dp_rank={dp_rank}/{dp_world_size}")
    except Exception:
        pass


    return train_loader, train_text_only_loader, train_img_only_loader, None , None , true_label_id


# ============================================================================
# Args update
# ============================================================================
def update_args_from_var(args):
    """Update args for probing mode (simplified from training version)."""
    args.data_parallel_size = args.n_gpus
    
    # [PPB Probing] Multi-GPU probing not yet implemented
    if args.data_parallel_size > 1:
        raise NotImplementedError(
            f"[PPB Probing] Multi-GPU probing is not yet implemented!\n"
            f"Reason: The model uses extensive register_buffer for attention collection\n"
            f"        which require manual reduction across GPUs.\n"
            f"Current setup: {args.data_parallel_size} GPUs detected\n"
            f"Workaround: Please use single GPU (--n_gpus 1) for now."
        )
    
    # Calculate micro batch size per GPU
    args.train_micro_batch_size_per_gpu = args.batch_size // args.data_parallel_size // args.gradient_accumulation_steps
    assert args.batch_size % args.train_micro_batch_size_per_gpu == 0

    os.makedirs(args.output_path, exist_ok=True)
    logi(f"[Probing] Output path: {args.output_path}")


# ============================================================================
# [Probing] Collection function
# ============================================================================
def probing_collect_from_loader(model, dataloader, device, desc="Collecting"):
    """
    [Probing] Collect attention statistics from dataloader.
    Model's forward automatically handles collection internally.
    
    Args:
        model: Can be DeepSpeed wrapped model or raw model
        dataloader: DataLoader for the data
        device: Target device
        desc: Description for progress bar
    """
    model.eval()
    
    # Penetrate DeepSpeed wrapper to access actual model for stats checking
    actual_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            # if batch_idx > 10:
            #     break
            inputs, labels, label_ids = batch

            # if batch_idx < 3:  # 抽查前 3 条
            #     from transformers import AutoProcessor as ModelProcessor
            #
            #     processor = ModelProcessor.from_pretrained(
            #         args.model_path,
            #         trust_remote_code=args.trust_remote_code,
            #         padding_side="left",
            #         use_fast=True,
            #         min_pixels=args.min_pixels, max_pixels=args.max_pixels
            #     )
            #     print(f"\n[DEBUG Audit {batch_idx}]")
            #     input_ids = inputs["input_ids"][0].cpu()  # (L,)
            #
            #     # 1. 强制将所有 mask 转换为布尔类型，防止整型按位运算出错
            #     txt_mask = inputs["ppb_news_text_mask"][0].bool().cpu()
            #     img_mask = inputs["ppb_image_mask"][0].bool().cpu()
            #     attn_mask = inputs.get("attention_mask", None)[0].bool().cpu()
            #
            #     # 2. 正确的逻辑：是有效 Token (attn_mask) 且 不是文本 且 不是图片
            #     inst_mask = attn_mask & ~txt_mask & ~img_mask
            #
            #     print(f"  Visual Tokens Found: {img_mask.sum().item()}")
            #
            #     # 解码验证
            #     print(f"  Text Content: {processor.tokenizer.decode(input_ids[txt_mask], skip_special_tokens=False)}")
            #     print(f"  Inst Content: {processor.tokenizer.decode(input_ids[inst_mask], skip_special_tokens=False)}")
            
            # Move inputs to GPU
            inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}

            try:
                # Forward pass (automatic collection)
                # Use the wrapper model for forward pass (DeepSpeed handles this)
                outputs = model(**inputs)

                # Update progress bar (check actual_model for stats)
                if hasattr(actual_model, '_attention_stats'):
                    total_layers = len(actual_model._attention_stats)
                    pbar.set_postfix({'layers': total_layers})

            except Exception as e:
                # 获取详细的堆栈信息
                detailed_traceback = traceback.format_exc()

                # 将简短错误和详细堆栈一起打印
                logi(f"\n⚠️  Error processing batch {batch_idx}: {e}\n【详细堆栈信息】:\n{detailed_traceback}")
                continue


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    initialize(args)
    update_args_from_var(args)
    rank = dist.get_rank() if dist.is_initialized() else 0

    LOGGER = setup_logger(args.output_path, rank)
    logi(f"Logger initialized. Logs will be written to {os.path.join(args.output_path, f'probing_rank{rank}.log')}")

    try:
        mp.set_start_method("spawn", force=True)
        logi("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logi(f"Unable to set start method to 'spawn', continuing with current method.")

    ds_config = setup_deepspeed_config(args)
    log_system_snapshot(args, ds_config)
    log_numeric_fingerprint(args)

    # ========================================================================
    # [Probing Mode] Inference-only attention collection
    # ========================================================================
    logi("=" * 80)
    logi("🔍 [Probing v1] Inference-time Modal Head Localization")
    logi("=" * 80)
    logi(f"📊 Unimodal ratio: {args.unimodal_ratio * 100}%")
    logi("=" * 80)
    
    # Prepare dataloaders
    train_loader, train_text_only_loader, train_img_only_loader, test_loader, val_loader, true_label_id = prepare_dataloaders(
        args, ds_config)
    
    # Load model with DeepSpeed (important for large model loading and consistency)
    logi("\n[1/4] Loading model with DeepSpeed...")
    device = torch.cuda.current_device()
    base_model = get_model(args).to(device)
    
    # Initialize with DeepSpeed (no optimizer, only model wrapping)
    # This is important for: 1) consistent variable control, 2) large model loading
    model_engine, _, _, _ = deepspeed.initialize(
        model=base_model,
        config_params=ds_config
    )
    model_engine.eval()
    logi(f"Model initialized with DeepSpeed on device {device}")
    
    # [Probing] Penetrate DeepSpeed wrapper to access the actual model
    # DeepSpeed wraps the model, need to access the underlying model for custom methods
    actual_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    logi(f"Model type after penetration: {type(actual_model).__name__}")
    
    # Enable attention collection
    logi("\n[2/4] Enabling attention collection...")
    if hasattr(actual_model, 'enable_attention_collection'):
        actual_model.enable_attention_collection()
        logi("Attention collection enabled")
    else:
        logi("⚠️  WARNING: Model does not support attention collection")
    
    # Collect statistics
    logi("\n[3/4] Collecting attention statistics...")
    logi("  [3.1] Collecting from img_only data...")
    if len(train_img_only_loader) > 0:
        probing_collect_from_loader(model_engine, train_img_only_loader, device, desc="Img-only")
    
    logi("  [3.2] Collecting from text_only data...")
    if len(train_text_only_loader) > 0:
        probing_collect_from_loader(model_engine, train_text_only_loader, device, desc="Text-only")
    
    if getattr(args, 'use_multimodal', False):
        logi("  [3.3] Collecting from multimodal data...")
        probing_collect_from_loader(model_engine, train_loader, device, desc="Multi-modal")
    
    # Disable collection (access through actual_model)
    if hasattr(actual_model, 'disable_attention_collection'):
        actual_model.disable_attention_collection()
    
    # Save results (access through actual_model)
    logi("\n[4/4] Saving results...")
    if hasattr(actual_model, 'save_head_roles') and hasattr(actual_model, 'get_accumulated_stats'):
        accumulated_stats = actual_model.get_accumulated_stats()
        
        if not accumulated_stats:
            logi("\n⚠️  WARNING: No attention statistics collected!")
            logi("   Check model implementation and data loading")
        else:
            saved_path = actual_model.save_head_roles(
                accumulated_stats=accumulated_stats,
                output_path=args.output_path
            )
            
            logi(f"\n✅ Head roles saved to: {saved_path}")
            
            num_layers = len(accumulated_stats)
            total_heads = sum(len(heads) for heads in accumulated_stats.values())
            logi(f"   - Layers processed: {num_layers}")
            logi(f"   - Total heads: {total_heads}")
    else:
        logi("\n❌ Model does not support required methods")
        logi(f"   - save_head_roles: {hasattr(actual_model, 'save_head_roles')}")
        logi(f"   - get_accumulated_stats: {hasattr(actual_model, 'get_accumulated_stats')}")
    
    logi("\n" + "=" * 80)
    logi("✅ [Probing v1] Inference completed successfully!")
    logi("=" * 80)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
