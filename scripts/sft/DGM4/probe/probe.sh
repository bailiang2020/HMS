#!/bin/bash

TRAIN_MODULE="src.train.probing_inference"
UNIMODAL_RATIO=0

# 像素/Token 配置 (Qwen3 = 32)
BASE_SIDE=32
MIN_TOKENS=0
MAX_TOKENS=256

# 自动构建路径和模型名称
OUTPUT_PATH="results/DGM4/probe/sdpa/qwen3_vl_2B/0-256"
MODEL_TYPE="Qwen3_VL_Custom_probing"

# 自动计算像素值
MIN_PIXELS=$((MIN_TOKENS * BASE_SIDE * BASE_SIDE))
MAX_PIXELS=$((MAX_TOKENS * BASE_SIDE * BASE_SIDE))
# =================================================

export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 打印配置
echo "================================================="
echo "PPB Probing Phase - Modal Head Localization"
echo "================================================="
echo "Experiment:         ${EXP_NAME}"
echo "Training Module:    ${TRAIN_MODULE}"
echo "Model Type:         ${MODEL_TYPE}"
echo "Unimodal Ratio:     ${UNIMODAL_RATIO}"
echo "Configured for Qwen3-VL (Base Side: ${BASE_SIDE})"
echo "Token Range:        ${MIN_TOKENS} - ${MAX_TOKENS}"
echo "Pixel Range:        ${MIN_PIXELS} - ${MAX_PIXELS}"
echo "Output Path:        ${OUTPUT_PATH}"
echo "================================================="

# [Probing] 调用推理模块（无训练）
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port 19101 --module "${TRAIN_MODULE}" \
    --model_path models/Qwen/Qwen3-VL-2B-Instruct \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --output_path "${OUTPUT_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --train_data_path data/processed/DGM4/raw/train_first40000.pkl \
    --img_dir data/raw/ \
    --num_workers 1 \
    --n_gpus 1 \
    --batch_size 24 \
    --gradient_accumulation_steps 1 \
    --deepspeed_config config/deepspeed/ds_config_zero2_bf16.json \
    --en \
    --attn_impl sdpa \
    --prompt_version DGM4_sft \
    --unimodal_ratio ${UNIMODAL_RATIO} \
    --log_interval 25 \
    --use_multimodal

# 使用方法:
# bash scripts/sft/DGM4/probe/probe.sh
