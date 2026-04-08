#!/bin/bash

# ================= Configuration =================
# 模型: Qwen3VL_2B_Instruct
# 数据集: DGM4
# 消融: random critical heads
UNIMODAL_RATIO=0.01
PURIFICATION_THRESHOLD=0.2
PURIFICATION_PROTECT_SCALE=0.2
EXP_NAME="ablation_random_heads_mix_ratio_${UNIMODAL_RATIO}/hms_${PURIFICATION_THRESHOLD}_ukr_${PURIFICATION_PROTECT_SCALE}"

TRAIN_MODULE="src.train.sft"

# 像素/Token 配置
BASE_SIDE=32
MIN_TOKENS=0
MAX_TOKENS=256

OUTPUT_PATH="results/DGM4/ours/sdpa/qwen3_vl_2B/lora/${EXP_NAME}/0-256"
MODEL_TYPE="Qwen3_VL_Custom_ours"
HEAD_ROLES_FILE="results/DGM4/probe/sdpa/qwen3_vl_2B/0-256/head_roles_top50_random_control_seed42_search50000.json"

MIN_PIXELS=$((MIN_TOKENS * BASE_SIDE * BASE_SIDE))
MAX_PIXELS=$((MAX_TOKENS * BASE_SIDE * BASE_SIDE))
# =================================================

export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "================================================="
echo "Running Experiment: ${EXP_NAME}"
echo "Training Module:    ${TRAIN_MODULE}"
echo "Model Type:         ${MODEL_TYPE}"
echo "Token Range:        ${MIN_TOKENS} - ${MAX_TOKENS}"
echo "Pixel Range:        ${MIN_PIXELS} - ${MAX_PIXELS}"
echo "Output Path:        ${OUTPUT_PATH}"
echo "Head Roles File:    ${HEAD_ROLES_FILE}"
echo "================================================="

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port 29343 --module "${TRAIN_MODULE}" \
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
    --purification_threshold ${PURIFICATION_THRESHOLD} \
    --purification_protect_scale ${PURIFICATION_PROTECT_SCALE} \
#    --test_only \
#    --img_only \
#    --lora_path "${OUTPUT_PATH}/bs24_lr5e-05_wd1e-06_ep5_dp1_gacc2_clip1.0_loraTrue_r256_a512_d0.05/prompt_DGM4_sft/model_best"



# bash scripts/sft/DGM4/ablation/random_heads/mix_0.01.sh
