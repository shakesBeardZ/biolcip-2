#!/bin/bash
set -euo pipefail

# Runs few-shot finetuning matrix sequentially:
#   models: bioclip, clip, openclip, siglip
#   k-shot: 1, 5, 10, 20
#
# Prereq:
#   - Prepare CSVs first via tools/prepare_coral_fewshot_splits.py
#   - Resulting files expected at FEWSHOT_DIR/fewshot_train_k{K}.csv

FEWSHOT_DIR=${FEWSHOT_DIR:-"data/fewshot_manual_pool/splits"}
K_LIST=${K_LIST:-"10"}
MODELS=${MODELS:-"clip openclip"}

CONDA_BASE=${CONDA_BASE:-"/home/${USER}/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-8}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-0}
WD=${WD:-0.01}
PRECISION=${PRECISION:-amp}

CORAL_PATH_REPLACE_FROM=${CORAL_PATH_REPLACE_FROM:-"/ibex/project/c2253/CoralNet_Images/patch_folders/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}
CORAL_PATH_REPLACE_TO=${CORAL_PATH_REPLACE_TO:-"/raid/felembaa/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}

BASE_LOG_DIR=${BASE_LOG_DIR:-"/raid/felembaa/reefnet/logs_finetune/fewshot_genus_clean"}

# mkdir -p "${BASE_LOG_DIR}"

for k in ${K_LIST}; do
  csv_path="${FEWSHOT_DIR}/fewshot_train_k${k}.csv"
  if [ ! -f "${csv_path}" ]; then
    echo "Missing few-shot CSV: ${csv_path}"
    exit 1
  fi

  for model in ${MODELS}; do
    case "${model}" in
      bioclip)
        train_script="eccv_2026_runs/finetune_coral_genus_bioclip_local.sh"
        ;;
      clip)
        train_script="eccv_2026_runs/finetune_coral_genus_clip_local.sh"
        ;;
      openclip)
        train_script="eccv_2026_runs/finetune_coral_genus_openclip_local.sh"
        ;;
      siglip)
        train_script="eccv_2026_runs/finetune_coral_genus_siglip_local.sh"
        ;;
      *)
        echo "Unsupported model: ${model}"
        exit 1
        ;;
    esac

    run_name="coral_genus_fewshot_${model}_k${k}_$(date +%Y%m%d_%H%M%S)"
    log_dir="${BASE_LOG_DIR}/${model}/k${k}"
    mkdir -p "${log_dir}"
    launch_log="${log_dir}/${run_name}.launch.log"

    echo "========================================="
    echo "Launching ${model} k=${k}"
    echo "CSV=${csv_path}"
    echo "RUN_NAME=${run_name}"
    echo "LOG_DIR=${log_dir}"
    echo "========================================="

    CONDA_BASE="${CONDA_BASE}" \
    CONDA_ENV="${CONDA_ENV}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    CSV_PATH="${csv_path}" \
    CORAL_SPLIT_COLUMN="split" \
    CORAL_TRAIN_SPLIT="train" \
    CORAL_VAL_SPLIT="train" \
    CORAL_PATH_REPLACE_FROM="${CORAL_PATH_REPLACE_FROM}" \
    CORAL_PATH_REPLACE_TO="${CORAL_PATH_REPLACE_TO}" \
    EPOCHS="${EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    WORKERS="${WORKERS}" \
    LR="${LR}" \
    WARMUP="${WARMUP}" \
    WD="${WD}" \
    PRECISION="${PRECISION}" \
    RUN_NAME="${run_name}" \
    LOG_DIR="${log_dir}" \
    bash "${train_script}" 2>&1 | tee "${launch_log}"
  done
done

echo "Few-shot matrix finished at: $(date)"
