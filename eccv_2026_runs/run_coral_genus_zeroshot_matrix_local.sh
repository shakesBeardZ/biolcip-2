#!/bin/bash
set -euo pipefail

# Runs closed-domain zero-shot evaluation matrix for:
#   bioclip, clip, openclip
#
# Outputs are aligned with checkpoint-matrix eval artifacts:
#   - predictions.csv
#   - misclassified.csv
#   - metrics.json
#   - eval.log
#   - summary.csv (across models)
#
# By default this evaluates baseline pretrained models (no checkpoint).
# Optionally set CHECKPOINT_BIOCLIP / CHECKPOINT_CLIP / CHECKPOINT_OPENCLIP.

CONDA_BASE=${CONDA_BASE:-"/home/${USER}/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

EVAL_CSV=${EVAL_CSV:-"data/fewshot_manual_pool/splits/test_fixed_catalogue.csv"}
OUTPUT_BASE=${OUTPUT_BASE:-"/raid/felembaa/reefnet/logs_eval/closed_domain_checkpoint_matrix/zero_shot"}
MODELS=${MODELS:-"bioclip clip openclip"}
RUN_TAG=${RUN_TAG:-"zero_shot_$(date +%Y%m%d_%H%M%S)"}

BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-8}
PRECISION=${PRECISION:-fp32}
TEMPLATE_STYLE=${TEMPLATE_STYLE:-plain}
RANK=${RANK:-genus}
NORMALIZE_ANTHOZOA=${NORMALIZE_ANTHOZOA:-1}
DUMP_TOPK=${DUMP_TOPK:-5}

CHECKPOINT_BIOCLIP=${CHECKPOINT_BIOCLIP:-""}
CHECKPOINT_CLIP=${CHECKPOINT_CLIP:-""}
CHECKPOINT_OPENCLIP=${CHECKPOINT_OPENCLIP:-""}

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
SUMMARY_CSV="${OUTPUT_BASE}/summary.csv"
echo "setup,model,run_name,checkpoint_name,checkpoint_path,top1,top3,top5,n_samples,num_classes,predictions_csv,metrics_json" > "${SUMMARY_CSV}"

for model in ${MODELS}; do
  case "${model}" in
    bioclip)
      model_name="hf-hub:imageomics/bioclip-2"
      pretrained_tag=""
      checkpoint_path="${CHECKPOINT_BIOCLIP}"
      ;;
    clip)
      model_name="ViT-L-14"
      pretrained_tag="openai"
      checkpoint_path="${CHECKPOINT_CLIP}"
      ;;
    openclip)
      model_name="ViT-H-14"
      pretrained_tag="laion2b_s32b_b79k"
      checkpoint_path="${CHECKPOINT_OPENCLIP}"
      ;;
    *)
      echo "Unsupported model: ${model}"
      exit 1
      ;;
  esac

  run_dir="${OUTPUT_BASE}/${model}/${RUN_TAG}"
  mkdir -p "${run_dir}"
  run_log="${run_dir}/eval.log"
  pred_csv="${run_dir}/predictions.csv"
  mis_csv="${run_dir}/misclassified.csv"
  metrics_json="${run_dir}/metrics.json"
  pred_json="${run_dir}/predictions_topk.json"

  cmd=(
    python -m src.evaluation.closed_domain_eval
    --model "${model_name}"
    --csv "${EVAL_CSV}"
    --rank "${RANK}"
    --template_style "${TEMPLATE_STYLE}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --precision "${PRECISION}"
    --dump_predictions_csv "${pred_csv}"
    --dump_metrics_json "${metrics_json}"
    --dump_predictions "${pred_json}"
    --dump_topk "${DUMP_TOPK}"
    --dump_misclassified_csv "${mis_csv}"
  )

  if [ -n "${pretrained_tag}" ]; then
    cmd+=(--pretrained "${pretrained_tag}")
  fi
  if [ -n "${checkpoint_path}" ]; then
    cmd+=(--checkpoint "${checkpoint_path}")
  fi
  if [ "${NORMALIZE_ANTHOZOA}" = "1" ]; then
    cmd+=(--normalize_anthozoa)
  fi

  echo "========================================="
  echo "Zero-shot eval: ${model}"
  echo "Model: ${model_name}"
  echo "Pretrained: ${pretrained_tag:-none}"
  echo "Checkpoint: ${checkpoint_path:-none}"
  echo "CSV: ${EVAL_CSV}"
  echo "Output: ${run_dir}"
  echo "========================================="
  "${cmd[@]}" 2>&1 | tee "${run_log}"

  if [ -n "${checkpoint_path}" ]; then
    checkpoint_name="$(basename "${checkpoint_path}")"
    checkpoint_for_summary="${checkpoint_path}"
  else
    checkpoint_name="pretrained"
    checkpoint_for_summary=""
  fi

  metrics_csv_line=$(python - <<PY
import json
p = "${metrics_json}"
d = json.load(open(p))
print(f"{d.get('top1','')},{d.get('top3','')},{d.get('top5','')},{d.get('n_samples','')},{d.get('num_classes','')}")
PY
)
  echo "zero_shot,${model},${RUN_TAG},${checkpoint_name},${checkpoint_for_summary},${metrics_csv_line},${pred_csv},${metrics_json}" >> "${SUMMARY_CSV}"
done

echo "Zero-shot matrix finished at: $(date)"
echo "Summary: ${SUMMARY_CSV}"
