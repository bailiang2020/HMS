#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# NOTE:
# Update COORDS_FILE if your actual attention analysis directory uses a slightly different naming scheme.
EVAL_SCRIPT="scripts/sft/DGM4/mix/mix_0.0025.sh"
COORDS_FILE="results/DGM4/attn_analysis/multi/Qwen3_VL_2B/lora/mix_0.0025/0-256/plots/ALL/top200_heatmap_norm_share_img_coords.txt"
TEST_DATA_PATH_OVERRIDE="data/processed/DGM4/raw/test_subset_5000.pkl"
MASK_COUNTS=(10 20 30 40 50 75 100)
LABEL="mix_0.0025_img"
RUN_NAME="mix_0.0025_img_mask"
CUDA_DEVICES="1"
MASTER_PORT_BASE="29800"

cd "${ROOT_DIR}"

python3 scripts/utils/run_mask_sweep.py \
  --eval_script "${EVAL_SCRIPT}" \
  --coords_file "${COORDS_FILE}" \
  --test_data_path_override "${TEST_DATA_PATH_OVERRIDE}" \
  --mask_counts "${MASK_COUNTS[@]}" \
  --label "${LABEL}" \
  --run_name "${RUN_NAME}" \
  --cuda_devices "${CUDA_DEVICES}" \
  --master_port_base "${MASTER_PORT_BASE}"

# bash scripts/utils/run_mask_sweep_mix_0.0025_img.sh