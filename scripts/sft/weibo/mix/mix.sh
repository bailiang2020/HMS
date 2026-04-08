#!/bin/bash

# ================= Configuration =================
# 模型: Qwen3VL_2B_Instruct
# 数据集: weibo
# 训练方法: mix (混合训练)
EXP_NAME="mix"

TRAIN_MODULE="src.train.sft"

# 像素/Token 配置 (Qwen3 = 32, Qwen2.5 = 28)
BASE_SIDE=32
MIN_TOKENS=0
MAX_TOKENS=256

# 自动构建路径和模型名称
OUTPUT_PATH="results/weibo/baseline/sdpa/qwen3_vl_2B/lora/${EXP_NAME}/0-256"
MODEL_TYPE="Qwen3_VL"

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
echo "Running Experiment: ${EXP_NAME}"
echo "Training Module:    ${TRAIN_MODULE}"
echo "Model Type:         ${MODEL_TYPE}"
echo "Configured for Qwen3-VL (Base Side: ${BASE_SIDE})"
echo "Token Range:        ${MIN_TOKENS} - ${MAX_TOKENS}"
echo "Pixel Range:        ${MIN_PIXELS} - ${MAX_PIXELS}"
echo "Output Path:        ${OUTPUT_PATH}"
echo "================================================="

# 调用训练模块
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port 29601 --module "${TRAIN_MODULE}" \
    --model_path models/Qwen/Qwen3-VL-2B-Instruct \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --output_path "${OUTPUT_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --train_data_path data/processed/weibo/raw/train_data.pkl \
    --test_data_path data/processed/weibo/raw/test_data.pkl \
    --val_data_path data/processed/weibo/raw/test_data.pkl \
    --img_dir data/raw/weibo/images \
    --num_workers 1 \
    --n_gpus 1 \
    --batch_size 24 \
    --gradient_accumulation_steps 2 \
    --lr 1e-04 \
    --epochs 5 \
    --deepspeed_config config/deepspeed/ds_config_zero2_bf16.json \
    --lora \
    --lora_target_modules \
      q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --attn_impl sdpa \
    --train_unimodal_json data/processed/weibo/annotation/train_annotation.json \
    --test_unimodal_json data/processed/weibo/annotation/test_annotation.json \
    --prompt_version weibo_sft \
    --log_interval 25 \
    --train_type mix \
#    --test_only \
#    --img_only \
#    --lora_path "${OUTPUT_PATH}/bs24_lr0.0001_wd1e-06_ep5_dp1_gacc2_clip1.0_loraTrue_r256_a512_d0.05/prompt_weibo_sft/model_best"


# bash scripts/sft/weibo/mix/mix.sh