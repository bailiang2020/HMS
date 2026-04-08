#!/bin/bash

# ================= Configuration =================
# 模型: Qwen3VL_2B_Instruct
# 数据集: washington_post
# 训练方法: mix (混合训练)
UNIMODAL_RATIO=0.0025
PURIFICATION_PROTECT_SCALE=0.4
EXP_NAME="ablation_ukr_mix_ratio_0.0025/th_0.4"

TRAIN_MODULE="src.train.sft"

# 像素/Token 配置 (Qwen3 = 32, Qwen2.5 = 28)
BASE_SIDE=32
MIN_TOKENS=0
MAX_TOKENS=256

# 自动构建路径和模型名称
OUTPUT_PATH="results/DGM4/ours/sdpa/qwen3_vl_2B/lora/${EXP_NAME}/0-256"
MODEL_TYPE="Qwen3_VL_Custom_ablation_ukr"
HEAD_ROLES_FILE="results/DGM4/probe/sdpa/qwen3_vl_2B/0-256/head_roles_top50_20260327_125042.json"

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
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port 29302 --module "${TRAIN_MODULE}" \
    --model_path models/Qwen/Qwen3-VL-2B-Instruct \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --output_path "${OUTPUT_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --train_data_path data/processed/DGM4/raw/train_first40000.pkl \
    --test_data_path data/processed/DGM4/raw/test.pkl \
    --val_data_path data/processed/DGM4/raw/val.pkl \
    --img_dir data/raw/ \
    --num_workers 1 \
    --n_gpus 1 \
    --batch_size 24 \
    --gradient_accumulation_steps 2 \
    --lr 5e-05 \
    --epochs 5 \
    --deepspeed_config config/deepspeed/ds_config_zero2_bf16.json \
    --en \
    --lora \
    --lora_target_modules \
      q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --attn_impl sdpa \
    --unimodal_ratio ${UNIMODAL_RATIO} \
    --prompt_version DGM4_sft \
    --log_interval 25 \
    --train_type mix \
    --head_roles_path "${HEAD_ROLES_FILE}" \
    --purification_protect_scale ${PURIFICATION_PROTECT_SCALE} \
#    --test_only \
#    --img_only \
#    --lora_path "${OUTPUT_PATH}/bs24_lr5e-05_wd1e-06_ep5_dp1_gacc2_clip1.0_loraTrue_r256_a512_d0.05/prompt_DGM4_sft/model_best"


# bash scripts/sft/DGM4/ablation/ukr/th_0.4/mix_0.0025.sh