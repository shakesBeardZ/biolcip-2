#!/bin/bash
set -euo pipefail

# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

CONDA_BASE=${CONDA_BASE:-"/home/felembaa/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

CSV_PATH=${CSV_PATH:-"/home/felembaa/reefnet/data/reefnet_with_07_filteration_v02_hard_corals_no_exif_with_taxonomy.csv"}
CORAL_PATH_REPLACE_FROM=${CORAL_PATH_REPLACE_FROM:-"/ibex/project/c2253/CoralNet_Images/patch_folders/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}
CORAL_PATH_REPLACE_TO=${CORAL_PATH_REPLACE_TO:-"/raid/felembaa/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}

BASE_LOG_DIR=${BASE_LOG_DIR:-"./logs_finetune/sequential_runs"}
STAMP=${STAMP:-"$(date +%Y%m%d_%H%M%S)"}

BIO_RUN_NAME=${BIO_RUN_NAME:-"coral_genus_ft_bioclip_seq_${STAMP}"}
CLIP_RUN_NAME=${CLIP_RUN_NAME:-"coral_genus_ft_clip_seq_${STAMP}"}

BIO_LAUNCH_LOG=${BIO_LAUNCH_LOG:-"${BASE_LOG_DIR}/${BIO_RUN_NAME}.launch.log"}
CLIP_LAUNCH_LOG=${CLIP_LAUNCH_LOG:-"${BASE_LOG_DIR}/${CLIP_RUN_NAME}.launch.log"}

# Optional per-run overrides (falls back to each script defaults if omitted)
BIO_BATCH_SIZE=${BIO_BATCH_SIZE:-256}
BIO_EPOCHS=${BIO_EPOCHS:-20}
BIO_LR=${BIO_LR:-3e-4}
BIO_WORKERS=${BIO_WORKERS:-16}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-256}
CLIP_EPOCHS=${CLIP_EPOCHS:-20}
CLIP_LR=${CLIP_LR:-3e-4}
CLIP_WORKERS=${CLIP_WORKERS:-16}

mkdir -p "${BASE_LOG_DIR}"

echo "========================================="
echo "Sequential launch started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "CSV_PATH=${CSV_PATH}"
echo "========================================="

run_bioclip() {
  echo "[1/2] Starting BioCLIP..."
  RUN_NAME="${BIO_RUN_NAME}" \
  LOG_DIR="${BASE_LOG_DIR}/bioclip" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  CONDA_BASE="${CONDA_BASE}" \
  CONDA_ENV="${CONDA_ENV}" \
  CSV_PATH="${CSV_PATH}" \
  CORAL_PATH_REPLACE_FROM="${CORAL_PATH_REPLACE_FROM}" \
  CORAL_PATH_REPLACE_TO="${CORAL_PATH_REPLACE_TO}" \
  BATCH_SIZE="${BIO_BATCH_SIZE}" \
  EPOCHS="${BIO_EPOCHS}" \
  LR="${BIO_LR}" \
  WORKERS="${BIO_WORKERS}" \
  bash eccv_2026_runs/finetune_coral_genus_bioclip_local.sh \
  > "${BIO_LAUNCH_LOG}" 2>&1
  echo "[1/2] BioCLIP finished successfully."
}

run_clip() {
  echo "[2/2] Starting CLIP..."
  RUN_NAME="${CLIP_RUN_NAME}" \
  LOG_DIR="${BASE_LOG_DIR}/clip" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  CONDA_BASE="${CONDA_BASE}" \
  CONDA_ENV="${CONDA_ENV}" \
  CSV_PATH="${CSV_PATH}" \
  CORAL_PATH_REPLACE_FROM="${CORAL_PATH_REPLACE_FROM}" \
  CORAL_PATH_REPLACE_TO="${CORAL_PATH_REPLACE_TO}" \
  BATCH_SIZE="${CLIP_BATCH_SIZE}" \
  EPOCHS="${CLIP_EPOCHS}" \
  LR="${CLIP_LR}" \
  WORKERS="${CLIP_WORKERS}" \
  bash eccv_2026_runs/finetune_coral_genus_clip_local.sh \
  > "${CLIP_LAUNCH_LOG}" 2>&1
  echo "[2/2] CLIP finished successfully."
}

run_bioclip
run_clip

echo "========================================="
echo "Sequential launch completed: $(date)"
echo "BioCLIP log: ${BIO_LAUNCH_LOG}"
echo "CLIP log:    ${CLIP_LAUNCH_LOG}"
echo "========================================="
