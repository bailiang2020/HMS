import numpy as np
import os
import random
import torch
import re
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
import argparse
from transformers import AutoTokenizer, AutoModel
from src.utils.prompt_templates import TEMPLATES
from vllm import LLM, SamplingParams
import pickle
from data.utils.vlm_datasets import VLMDataset
from tqdm import tqdm
from datetime import timedelta
import deepspeed
import torch.distributed as dist
from torch.distributed import group
from typing import Optional, Any



def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        # 设置 Python、NumPy 和 PyTorch 的随机数种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # # 确保 CUDNN 的行为是确定性的
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        torch.use_deterministic_algorithms(True)


def extract_lm_answer(generated_text: str, default: str = "A") -> int:
    """
    从生成文本中提取最终答案（A->1, B->0）。如果匹配不到“ANSWER: $LETTER”或指定关键词，则返回默认值。
    """
    # 清除一些常见格式符号（如加粗、强调）
    generated_text = re.sub(r'\*\*|__|~~', '', generated_text)

    # 匹配形式：ANSWER: A、Final answer: B、Answer is A 等
    answer_pattern = r'(?:ANSWER|Answer|answer|Final answer|final answer)[^\w]{0,3}"?([A-Za-z])"?'

    matches = re.findall(answer_pattern, generated_text)
    if matches:
        letter = matches[-1].upper()
        return 1 if letter == "A" else 0
    print(f"未能从生成文本中提取答案，使用默认值 {default}，文本：{generated_text}")
    return 1 if default.upper() == "A" else 0


def extract_raw_lm_answer(generated_text: str, default: str = "A") -> str:
    """
    从生成文本中提取最终答案。如果匹配不到“ANSWER: $LETTER”或指定关键词，则返回默认值。
    """
    # 清除一些常见格式符号（如加粗、强调）
    generated_text = re.sub(r'\*\*|__|~~', '', generated_text)

    # 匹配形式：ANSWER: A、Final answer: B、Answer is A 等
    answer_pattern = r'(?:ANSWER|Answer|answer|Final answer|final answer)[^\w]{0,3}"?([A-Za-z])"?'

    matches = re.findall(answer_pattern, generated_text)
    if matches:
        letter = matches[-1].upper()
        return letter
    print(f"未能从生成文本中提取答案，使用默认值 {default}，文本：{generated_text}")
    return default.upper()


def switch_question_version(text: str, version: str) -> str:
    """
    根据版本号切换问题的格式。
    """
    template = TEMPLATES.get(version)
    if template is None:
        raise ValueError(f"不支持的 prompt_version: {version}")
    return template.format(text=text)


# def vlm_prompt(text: str, model_type: str, version: str) -> str:
#     question = switch_question_version(text, version)
#     if model_type == "InternVL":
#         placeholder = "<image>"
#     elif model_type == "Qwen3_VL":
#         placeholder = "<|image_pad|>"
#     else:
#         raise ValueError(f"不支持的 model_type: {model_type}")
#     return question.format(image=placeholder,text=text)


def vlm_stop_token_ids(args) -> list:
    """
    返回 VLM 模型的停止标记 ID 列表。
    """
    if args.model_type == "InternVL":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
        return stop_token_ids
    else:
        return []


def gen_report(labels, preds):
    # 转换标签类型为字符串
    str_true = [str(label) for label in labels]
    str_pred = [str(p) for p in preds]

    # 生成并打印分类报告
    target_names = ['real 0', 'fake 1']
    report = classification_report(
        str_true, str_pred,
        target_names=target_names,
        digits=5
    )
    print("\nClassification Report:")
    print(report)
    return report


def get_parser():
    parser = argparse.ArgumentParser(description="Command line arguments for VLM inference")

    # ===== 模型路径与推理设置 =====
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--model_type", type=str, help="Type of model")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save outputs")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--enable_prefix_caching", action='store_true', help="Enable prefix caching")

    # ===== GPU与分布式设置 =====
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process in distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes in distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank of the process in distributed training")
    parser.add_argument("--data_parallel_size", type=int, default=1,)
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--deepspeed_config", type=str, default='config/deepspeed/ds_config_zero0_bf16.json', help="Path to the DeepSpeed configuration file")

    # ===== 数据与缓存设置 =====
    parser.add_argument("--data_path", type=str, help="Path to preprocessed data pickle file")
    parser.add_argument("--img_dir", type=str, help="Directory containing images")
    parser.add_argument("--use_cache", action='store_true', help="Use cached data")
    parser.add_argument("--img_only", action='store_true', help="Use image-only data")
    parser.add_argument("--text_only", action='store_true', help="Use text-only data")

    # ===== LMDB 与 meta 返回控制 =====
    parser.add_argument(
        "--meta_in_item", type=str, default="none", choices=["none", "index", "full"],
        help="是否把 meta 随样本一并返回：none=不返回；index=仅返回当前索引的子集；full=返回整份 meta（小数据可用）。"
    )
    parser.add_argument(
        "--meta_keys_for_item", type=str, nargs="+", default=None,
        help="当 meta_in_item='index' 时，指定需要返回的 meta 键集合；不指定则返回所有键的当前索引值。"
    )

    # ===== Prompt设置 =====
    parser.add_argument("--prompt_version", type=str, default="weibo", help="Version of the prompt to use (e.g., 'v1', 'v2')")

    # ===== LoRA相关参数 =====
    parser.add_argument("--lora", action='store_true', help="Use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=256, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=512, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="Target modules for LoRA")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights file")

    # ===== 训练参数设置 =====
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=8, help="Micro batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for model loading")
    parser.add_argument("--total_iterations", type=int, default=-1, help="Total number of iterations for training, auto-detected if -1")
    parser.add_argument("--test_only", action='store_true', help="Only run inference without training")
    parser.add_argument("--test_with_single_gpu", action='store_true', help="Run inference with single GPU even in distributed mode")

    return parser


def get_vllm_model(args) -> LLM:
    # os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    model = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        seed=args.seed,
        data_parallel_size=args.data_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    return model


def inference_vllm(
        args, vllm_model,
):
    """
    使用 vLLM 接口对多模态数据进行推理。仅支持 LLMTorch 方式的 generate。
    """
    stop_token_ids = vlm_stop_token_ids(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_model_len,
        stop_token_ids=stop_token_ids,
    )
    dataset = VLMDataset(args)
    inputs, labels = dataset.get_all_data()
    outputs = vllm_model.generate(inputs, sampling_params=sampling_params)
    return outputs,labels


def handle_output(
        args, outputs):
    gen = []
    for i, out in tqdm(enumerate(outputs), total=len(outputs), desc="Processing outputs"):
        txt = out.outputs[0].text
        gen.append(txt)

    os.makedirs(args.output_path, exist_ok=True)
    return gen


def get_report(
        args, gen, labels):
    """
    处理答案提取
    """
    preds = []
    for i, text in enumerate(gen):
        ans = extract_lm_answer(text)
        preds.append(ans)
    # 生成分类报告
    report = gen_report(labels, preds)

    return report,preds


def save_vars(args, data, filename):
    """ 保存数据到指定路径的文件
    """
    os.makedirs(args.output_path, exist_ok=True)
    file_path = os.path.join(args.output_path, filename
                             )
    with open(file_path, "w") as f:
        f.write(str(vars(data)))
    f.close()


def save_pkl(args, data, filename):
    """
    保存数据到 pickle 文件
    """
    os.makedirs(args.output_path, exist_ok=True)
    pkl_path = os.path.join(args.output_path, filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    f.close()
    print(f"Data saved to {pkl_path}")


def save_txt(args, data, filename):
    """
    保存文本数据到文件
    """
    os.makedirs(args.output_path, exist_ok=True)
    txt_path = os.path.join(args.output_path, filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")
    f.close()
    print(f"Text data saved to {txt_path}")


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")
    torch.cuda.set_device(f"cuda:{args.local_rank}")

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def initialize(args):
    init_distributed_ds(args)


def all_gather(
    t: torch.Tensor, dim: int = 0, world_size: Optional[int] = None,
    group: Optional[group] = None, op: str = "cat"
) -> torch.Tensor:
    if world_size is None:
        world_size = dist.get_world_size()
    # all_t = [torch.zeros_like(t) for _ in range(world_size)]
    # dist.all_gather(all_t, t, group=group)
    all_t = [None for _ in range(world_size)]
    dist.all_gather_object(all_t, t, group=group)
    device = t.device
    all_t = [x.to(device) for x in all_t]  # 保证拼接前在同一设备
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t

def all_gather_list(
    data: Any, world_size: Optional[int] = None,
    group: Optional[group] = None
) -> list:
    if world_size is None:
        world_size = dist.get_world_size()
    all_data = [None for _ in range(world_size)]
    dist.all_gather_object(all_data, data, group=group)
    return all_data
