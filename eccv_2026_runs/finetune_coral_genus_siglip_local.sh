#!/bin/bash
set -euo pipefail

# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}

CONDA_ENV=${CONDA_ENV:-"pytoenv"}
CONDA_BASE=${CONDA_BASE:-""}
CSV_PATH=${CSV_PATH:-"/home/felembaa/reefnet/data/reefnet_with_07_filteration_v02_hard_corals_no_exif_with_taxonomy.csv"}
VAL_DATA_PATH=${VAL_DATA_PATH:-""}
CORAL_PATH_REPLACE_FROM=${CORAL_PATH_REPLACE_FROM:-"/ibex/project/c2253/CoralNet_Images/patch_folders/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}
CORAL_PATH_REPLACE_TO=${CORAL_PATH_REPLACE_TO:-"/raid/felembaa/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}
CORAL_SPLIT_COLUMN=${CORAL_SPLIT_COLUMN:-split}
CORAL_TRAIN_SPLIT=${CORAL_TRAIN_SPLIT:-train}
CORAL_VAL_SPLIT=${CORAL_VAL_SPLIT:-val}

RUN_NAME=${RUN_NAME:-"coral_genus_ft_siglip_local_$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"./logs_finetune/siglip_vitb16_local"}

BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-20}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-500}
WD=${WD:-0.01}
PRECISION=${PRECISION:-amp}
WORKERS=${WORKERS:-16}

# SigLIP defaults; override via env if needed.
SIGLIP_MODEL=${SIGLIP_MODEL:-"ViT-B-16-SigLIP"}
SIGLIP_PRETRAINED=${SIGLIP_PRETRAINED:-"webli"}

EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-""}

# Optional environment activation:
#   CONDA_BASE="/home/$USER/anaconda3" CONDA_ENV="pytoenv" bash eccv_2026_runs/finetune_coral_genus_siglip_local.sh
if [ -n "${CONDA_ENV}" ]; then
  if [ -z "${CONDA_BASE}" ]; then
    if [ -d "/home/${USER}/anaconda3" ]; then
      CONDA_BASE="/home/${USER}/anaconda3"
    elif [ -d "/home/${USER}/miniconda3" ]; then
      CONDA_BASE="/home/${USER}/miniconda3"
    elif command -v conda >/dev/null 2>&1; then
      CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
  fi

  if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "Unable to locate conda.sh. Set CONDA_BASE (e.g. /home/${USER}/anaconda3)."
    exit 1
  fi

  # Some conda activation hooks reference unset vars and break under `set -u`.
  set +u
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi

export OMP_NUM_THREADS
if [ ! -f "${CSV_PATH}" ]; then
  echo "CSV file not found: ${CSV_PATH}"
  exit 1
fi

echo "Running on $(hostname) with ${GPUS_PER_NODE} GPU(s)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CSV_PATH=${CSV_PATH}"
echo "CORAL_PATH_REPLACE_FROM=${CORAL_PATH_REPLACE_FROM}"
echo "CORAL_PATH_REPLACE_TO=${CORAL_PATH_REPLACE_TO}"
echo "SIGLIP_MODEL=${SIGLIP_MODEL}"
echo "SIGLIP_PRETRAINED=${SIGLIP_PRETRAINED}"
nvidia-smi

mkdir -p "${LOG_DIR}"

TRAIN_ARGS=(
  --model "${SIGLIP_MODEL}"
  --pretrained "${SIGLIP_PRETRAINED}"
  --siglip
  --train-data "${CSV_PATH}"
  --dataset-type coral_split
  --coral-split-column "${CORAL_SPLIT_COLUMN}"
  --coral-train-split "${CORAL_TRAIN_SPLIT}"
  --coral-val-split "${CORAL_VAL_SPLIT}"
  --coral-target-level genus
  --coral-caption-mode chain
  --coral-path-key patch_path
  --coral-path-replace-from "${CORAL_PATH_REPLACE_FROM}"
  --coral-path-replace-to "${CORAL_PATH_REPLACE_TO}"
  --coral-no-infer-genus-from-species
  --coral-keep-missing-targets
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --warmup "${WARMUP}"
  --wd "${WD}"
  --precision "${PRECISION}"
  --workers "${WORKERS}"
  --logs "${LOG_DIR}"
  --name "${RUN_NAME}"
)

if [ -n "${VAL_DATA_PATH}" ]; then
  TRAIN_ARGS+=(--val-data "${VAL_DATA_PATH}")
fi

if [ -n "${EXTRA_TRAIN_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
  TRAIN_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPUS_PER_NODE}" \
  -m src.training.main \
  -- \
  "${TRAIN_ARGS[@]}"

echo "========================================="
echo "Local SigLIP job finished at : $(date)"
echo "========================================="
