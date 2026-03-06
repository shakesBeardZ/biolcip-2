#!/bin/bash
set -euo pipefail

# Evaluate all epoch checkpoints with closed-domain eval and dump:
# - per-sample predictions CSV (true/pred labels + top-k columns)
# - metrics JSON per checkpoint
# - one aggregate summary CSV across the matrix
#
# Matrix dimensions:
# - setup: full, k1, k5, k10, k20
# - model: bioclip, clip, openclip
#
# By default:
# - full: picks latest run per model from known full-FT paths
# - few-shot: picks latest run per (model, k) from FEWSHOT_ROOT
#
# You can override checkpoint directories explicitly via env vars:
# FULL_CKPT_DIR_BIOCLIP, FULL_CKPT_DIR_CLIP, FULL_CKPT_DIR_OPENCLIP
#
# Also configurable:
# RUN_PICK=latest|all  (latest = one run dir per setup/model, all = every found run dir)
# SETUPS="full k1 k5 k10 k20"
# MODELS="bioclip clip openclip"

CONDA_BASE=${CONDA_BASE:-"/home/${USER}/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

EVAL_CSV=${EVAL_CSV:-"data/fewshot_manual_pool/splits/test_fixed_catalogue.csv"}
OUTPUT_BASE=${OUTPUT_BASE:-"/raid/felembaa/reefnet/logs_eval/closed_domain_checkpoint_matrix_clean"}

SETUPS=${SETUPS:-"k10 k20"}
MODELS=${MODELS:-"bioclip"} #  clip openclip
RUN_PICK=${RUN_PICK:-latest}  # latest|all
ALLOW_CONCURRENT=${ALLOW_CONCURRENT:-0}

BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-16}
PRECISION=${PRECISION:-fp32}
TEMPLATE_STYLE=${TEMPLATE_STYLE:-plain}
RANK=${RANK:-genus}
NORMALIZE_ANTHOZOA=${NORMALIZE_ANTHOZOA:-1}
DUMP_TOPK=${DUMP_TOPK:-5}

FEWSHOT_ROOT=${FEWSHOT_ROOT:-"/raid/felembaa/reefnet/logs_finetune/fewshot_genus_clean"}

# Optional explicit checkpoint dirs for full FT
FULL_CKPT_DIR_BIOCLIP=${FULL_CKPT_DIR_BIOCLIP:-"/raid/felembaa/reefnet/sequential_runs/bioclip/coral_genus_ft_bioclip_seq_20260303_173219/checkpoints"}
FULL_CKPT_DIR_CLIP=${FULL_CKPT_DIR_CLIP:-"/raid/felembaa/reefnet/sequential_runs/clip/coral_genus_ft_clip_seq_20260303_173219/checkpoints"}
FULL_CKPT_DIR_OPENCLIP=${FULL_CKPT_DIR_OPENCLIP:-"/home/felembaa/reefnet/biolcip-2/logs_finetune/openclip_vith14_local/coral_genus_ft_openclip_local_20260303_223929/checkpoints"}

if [ ! -f "${EVAL_CSV}" ]; then
  echo "EVAL_CSV not found: ${EVAL_CSV}"
  exit 1
fi

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  echo "Unable to locate conda.sh at ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${OUTPUT_BASE}"
SUMMARY_CSV=${SUMMARY_CSV:-"${OUTPUT_BASE}/summary.csv"}

if [ "${ALLOW_CONCURRENT}" != "1" ]; then
  LOCK_DIR="${OUTPUT_BASE}/.checkpoint_matrix_lock"
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "$$" > "${LOCK_DIR}/pid"
  else
    holder_pid=""
    if [ -f "${LOCK_DIR}/pid" ]; then
      holder_pid=$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)
    fi
    if [ -n "${holder_pid}" ] && kill -0 "${holder_pid}" 2>/dev/null; then
      echo "Another checkpoint-matrix eval runner is already active: ${holder_pid}"
      echo "Stop it first, or set ALLOW_CONCURRENT=1 to override."
      exit 1
    fi
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
    echo "$$" > "${LOCK_DIR}/pid"
  fi
  trap 'rm -rf "${LOCK_DIR}"' EXIT
fi

echo "Run config:"
echo "  SETUPS=${SETUPS}"
echo "  MODELS=${MODELS}"
echo "  RUN_PICK=${RUN_PICK}"
echo "  EVAL_CSV=${EVAL_CSV}"
echo "  OUTPUT_BASE=${OUTPUT_BASE}"
echo "  SUMMARY_CSV=${SUMMARY_CSV}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  WORKERS=${WORKERS}"
echo "  PRECISION=${PRECISION}"
echo "  TEMPLATE_STYLE=${TEMPLATE_STYLE}"
echo "  RANK=${RANK}"
echo "  NORMALIZE_ANTHOZOA=${NORMALIZE_ANTHOZOA}"

echo "setup,model,run_name,checkpoint_name,checkpoint_path,top1,top3,top5,n_samples,num_classes,predictions_csv,metrics_json" > "${SUMMARY_CSV}"

discover_checkpoint_dirs() {
  local setup="$1"
  local model="$2"
  local -a dirs=()

  if [ "${setup}" = "full" ]; then
    case "${model}" in
      bioclip)
        if [ -n "${FULL_CKPT_DIR_BIOCLIP}" ]; then
          dirs+=("${FULL_CKPT_DIR_BIOCLIP}")
        else
          while IFS= read -r d; do dirs+=("${d}"); done < <(find /raid/felembaa/reefnet/sequential_runs/bioclip -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
          while IFS= read -r d; do dirs+=("${d}"); done < <(find logs_finetune/bioclip2_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
          while IFS= read -r d; do dirs+=("${d}"); done < <(find /raid/felembaa/reefnet/logs_finetune/bioclip2_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
        fi
        ;;
      clip)
        if [ -n "${FULL_CKPT_DIR_CLIP}" ]; then
          dirs+=("${FULL_CKPT_DIR_CLIP}")
        else
          while IFS= read -r d; do dirs+=("${d}"); done < <(find /raid/felembaa/reefnet/sequential_runs/clip -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
          while IFS= read -r d; do dirs+=("${d}"); done < <(find logs_finetune/clip_vitl14_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
          while IFS= read -r d; do dirs+=("${d}"); done < <(find /raid/felembaa/reefnet/logs_finetune/clip_vitl14_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
        fi
        ;;
      openclip)
        if [ -n "${FULL_CKPT_DIR_OPENCLIP}" ]; then
          dirs+=("${FULL_CKPT_DIR_OPENCLIP}")
        else
          while IFS= read -r d; do dirs+=("${d}"); done < <(find logs_finetune/openclip_vith14_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
          while IFS= read -r d; do dirs+=("${d}"); done < <(find /raid/felembaa/reefnet/logs_finetune/openclip_vith14_local -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
        fi
        ;;
    esac
  else
    local k="${setup#k}"
    local base="${FEWSHOT_ROOT}/${model}/k${k}"
    while IFS= read -r d; do dirs+=("${d}"); done < <(find "${base}" -maxdepth 3 -type d -name checkpoints 2>/dev/null | sort)
  fi

  if [ "${#dirs[@]}" -eq 0 ]; then
    return 0
  fi

  # Deduplicate.
  mapfile -t dirs < <(printf "%s\n" "${dirs[@]}" | awk '!seen[$0]++')

  if [ "${RUN_PICK}" = "latest" ]; then
    local latest
    latest=$(ls -1dt "${dirs[@]}" 2>/dev/null | head -n1 || true)
    if [ -n "${latest}" ]; then
      printf "%s\n" "${latest}"
    fi
  else
    printf "%s\n" "${dirs[@]}"
  fi
}

for setup in ${SETUPS}; do
  for model in ${MODELS}; do
    case "${model}" in
      bioclip)
        model_name="hf-hub:imageomics/bioclip-2"
        pretrained_tag=""
        ;;
      clip)
        model_name="ViT-L-14"
        pretrained_tag="openai"
        ;;
      openclip)
        model_name="ViT-H-14"
        pretrained_tag="laion2b_s32b_b79k"
        ;;
      *)
        echo "Unsupported model: ${model}"
        exit 1
        ;;
    esac

    mapfile -t ckpt_dirs < <(discover_checkpoint_dirs "${setup}" "${model}")
    if [ "${#ckpt_dirs[@]}" -eq 0 ]; then
      echo "[WARN] No checkpoint dirs found for setup=${setup}, model=${model}"
      continue
    fi

    for ckpt_dir in "${ckpt_dirs[@]}"; do
      run_name=$(basename "$(dirname "${ckpt_dir}")")
      mapfile -t ckpts < <(ls -1 "${ckpt_dir}"/epoch_*.pt 2>/dev/null | sort -V || true)
      if [ "${#ckpts[@]}" -eq 0 ]; then
        echo "[WARN] No epoch_*.pt checkpoints in ${ckpt_dir}"
        continue
      fi

      for ckpt in "${ckpts[@]}"; do
        ckpt_file=$(basename "${ckpt}")
        ckpt_stem="${ckpt_file%.pt}"
        out_dir="${OUTPUT_BASE}/${setup}/${model}/${run_name}/${ckpt_stem}"
        mkdir -p "${out_dir}"

        eval_log="${out_dir}/eval.log"
        pred_csv="${out_dir}/predictions.csv"
        mis_csv="${out_dir}/misclassified.csv"
        metrics_json="${out_dir}/metrics.json"

        cmd=(
          python -m src.evaluation.closed_domain_eval
          --model "${model_name}"
          --csv "${EVAL_CSV}"
          --rank "${RANK}"
          --template_style "${TEMPLATE_STYLE}"
          --batch-size "${BATCH_SIZE}"
          --workers "${WORKERS}"
          --precision "${PRECISION}"
          --checkpoint "${ckpt}"
          --dump_topk "${DUMP_TOPK}"
          --dump_predictions_csv "${pred_csv}"
          --dump_misclassified_csv "${mis_csv}"
          --dump_metrics_json "${metrics_json}"
        )

        if [ -n "${pretrained_tag}" ]; then
          cmd+=(--pretrained "${pretrained_tag}")
        fi
        if [ "${NORMALIZE_ANTHOZOA}" = "1" ]; then
          cmd+=(--normalize_anthozoa)
        fi

        echo "========================================="
        echo "Eval setup=${setup} model=${model} run=${run_name} ckpt=${ckpt_file}"
        echo "Output: ${out_dir}"
        echo "========================================="
        "${cmd[@]}" 2>&1 | tee "${eval_log}"

        metrics_csv_line=$(python - <<PY
import json
p = "${metrics_json}"
d = json.load(open(p))
print(f"{d.get('top1','')},{d.get('top3','')},{d.get('top5','')},{d.get('n_samples','')},{d.get('num_classes','')}")
PY
)
        echo "${setup},${model},${run_name},${ckpt_file},${ckpt},${metrics_csv_line},${pred_csv},${metrics_json}" >> "${SUMMARY_CSV}"
      done
    done
  done
done

echo "Done. Summary: ${SUMMARY_CSV}"
