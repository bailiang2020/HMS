#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

EVAL_SCRIPT="scripts/sft/DGM4/mix/mix_0.0025.sh"
TEST_DATA_PATH_OVERRIDE="data/processed/DGM4/raw/test_subset_5000.pkl"
MASK_COUNTS=(10 20 30 40 50 75 100)
RANDOM_SEED="42"
LABEL="mix_0.0025_img_random"
RUN_NAME="mix_0.0025_img_random_mask"
CUDA_DEVICES="2"
MASTER_PORT_BASE="29900"

cd "${ROOT_DIR}"

python3 scripts/utils/run_mask_sweep.py \
  --eval_script "${EVAL_SCRIPT}" \
  --mask_source random \
  --test_data_path_override "${TEST_DATA_PATH_OVERRIDE}" \
  --mask_counts "${MASK_COUNTS[@]}" \
  --random_seed "${RANDOM_SEED}" \
  --label "${LABEL}" \
  --run_name "${RUN_NAME}" \
  --cuda_devices "${CUDA_DEVICES}" \
  --master_port_base "${MASTER_PORT_BASE}"

# bash scripts/utils/run_random_mask_sweep_mix_0.0025_img.sh