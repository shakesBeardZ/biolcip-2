#!/bin/bash
set -euo pipefail

# Evaluate best checkpoints listed in BEST_CSV against a target CSV dataset.
#
# Expected BEST_CSV columns (from aggregate script):
#   setup,model,run_name,checkpoint_name,checkpoint_path
#
# Default target use-case:
#   - Input: best_by_setup_model_macro_recall.csv from prior runs
#   - Eval CSV: external RSG dataset converted to catalogue-style CSV
#   - Output: one folder per setup/model/run/checkpoint with
#       predictions.csv, misclassified.csv, metrics.json, eval.log

CONDA_BASE=${CONDA_BASE:-"/home/${USER}/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

BEST_CSV=${BEST_CSV:-"/raid/felembaa/reefnet/logs_eval/closed_domain_checkpoint_matrix/best_by_setup_model_macro_recall.csv"}
EVAL_CSV=${EVAL_CSV:-"data/rsg_eval/rsg_test_catalogue.csv"}

OUTPUT_BASE=${OUTPUT_BASE:-"/raid/felembaa/reefnet/logs_eval/rsg_best_checkpoint_matrix"}
RUN_TAG=${RUN_TAG:-"rsg_best_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR="${OUTPUT_BASE}/${RUN_TAG}"
SUMMARY_CSV=${SUMMARY_CSV:-"${RUN_DIR}/summary.csv"}

# Optional filters; empty means all rows from BEST_CSV.
SETUPS=${SETUPS:-""}            # e.g. "full k1 k5 k10 k20 zero_shot"
MODELS=${MODELS:-""}            # e.g. "bioclip clip openclip siglip"

BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-16}
PRECISION=${PRECISION:-fp32}
TEMPLATE_STYLE=${TEMPLATE_STYLE:-plain}
RANK=${RANK:-genus}
RANK_ONLY=${RANK_ONLY:-0}
NORMALIZE_ANTHOZOA=${NORMALIZE_ANTHOZOA:-1}
DUMP_TOPK=${DUMP_TOPK:-5}

# closed_domain_eval dataset loader mode:
#   catalogue -> FranCatalogueLoader
#   rsg       -> RSGPatchDataset
DATASET_LOADER=${DATASET_LOADER:-catalogue}
RSG_TAXONOMY_CSV=${RSG_TAXONOMY_CSV:-""}
RSG_CORALS_ONLY=${RSG_CORALS_ONLY:-0}
RSG_PATH_KEY=${RSG_PATH_KEY:-""}
RSG_LABEL_KEY=${RSG_LABEL_KEY:-""}

# Optional extra eval filters/constraints.
CLASS_LIST=${CLASS_LIST:-""}
FILTER_RANK=${FILTER_RANK:-""}
FILTER_VALUE=${FILTER_VALUE:-""}

ALLOW_CONCURRENT=${ALLOW_CONCURRENT:-0}
STRICT=${STRICT:-0}

if [ ! -f "${BEST_CSV}" ]; then
  echo "BEST_CSV not found: ${BEST_CSV}"
  exit 1
fi
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

mkdir -p "${RUN_DIR}"

if [ "${ALLOW_CONCURRENT}" != "1" ]; then
  LOCK_DIR="${OUTPUT_BASE}/.best_checkpoint_eval_lock"
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "$$" > "${LOCK_DIR}/pid"
  else
    holder_pid=""
    if [ -f "${LOCK_DIR}/pid" ]; then
      holder_pid=$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)
    fi
    if [ -n "${holder_pid}" ] && kill -0 "${holder_pid}" 2>/dev/null; then
      echo "Another best-checkpoint eval runner is already active: ${holder_pid}"
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
echo "  BEST_CSV=${BEST_CSV}"
echo "  EVAL_CSV=${EVAL_CSV}"
echo "  OUTPUT_BASE=${OUTPUT_BASE}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_DIR=${RUN_DIR}"
echo "  SUMMARY_CSV=${SUMMARY_CSV}"
echo "  SETUPS=${SETUPS:-<all>}"
echo "  MODELS=${MODELS:-<all>}"
echo "  DATASET_LOADER=${DATASET_LOADER}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  WORKERS=${WORKERS}"
echo "  PRECISION=${PRECISION}"
echo "  TEMPLATE_STYLE=${TEMPLATE_STYLE}"
echo "  RANK=${RANK}"
echo "  RANK_ONLY=${RANK_ONLY}"
echo "  NORMALIZE_ANTHOZOA=${NORMALIZE_ANTHOZOA}"

echo "setup,model,run_name,checkpoint_name,checkpoint_path,top1,top3,top5,n_samples,num_classes,predictions_csv,metrics_json" > "${SUMMARY_CSV}"

mapfile -t BEST_ROWS < <(
  BEST_CSV="${BEST_CSV}" SETUPS="${SETUPS}" MODELS="${MODELS}" python - <<'PY'
import csv
import os

best_csv = os.environ["BEST_CSV"]
setups_raw = os.environ.get("SETUPS", "")
models_raw = os.environ.get("MODELS", "")

setups = {s.strip() for s in setups_raw.split() if s.strip()}
models = {m.strip().lower() for m in models_raw.split() if m.strip()}

with open(best_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    required = ["setup", "model", "run_name", "checkpoint_name", "checkpoint_path"]
    missing = [c for c in required if c not in (reader.fieldnames or [])]
    if missing:
        raise SystemExit(f"BEST_CSV missing required columns: {missing}")

    for row in reader:
        setup = (row.get("setup") or "").strip()
        model = (row.get("model") or "").strip().lower()
        run_name = (row.get("run_name") or "").strip()
        ckpt_name = (row.get("checkpoint_name") or "").strip()
        ckpt_path = (row.get("checkpoint_path") or "").strip()

        if not setup or not model:
            continue
        if setups and setup not in setups:
            continue
        if models and model not in models:
            continue

        print("\t".join([setup, model, run_name, ckpt_name, ckpt_path]))
PY
)

if [ "${#BEST_ROWS[@]}" -eq 0 ]; then
  echo "No rows selected from BEST_CSV with current SETUPS/MODELS filters."
  exit 1
fi

for row in "${BEST_ROWS[@]}"; do
  IFS=$'\t' read -r setup model run_name checkpoint_name checkpoint_path <<< "${row}"

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
    siglip)
      model_name="ViT-B-16-SigLIP"
      pretrained_tag="webli"
      ;;
    *)
      echo "[WARN] Unsupported model '${model}' in BEST_CSV. Skipping."
      continue
      ;;
  esac

  if [ -n "${checkpoint_path}" ] && [ ! -f "${checkpoint_path}" ]; then
    msg="[WARN] Missing checkpoint file: ${checkpoint_path} (setup=${setup}, model=${model}, run=${run_name})"
    if [ "${STRICT}" = "1" ]; then
      echo "${msg}"
      exit 1
    fi
    echo "${msg}"
    continue
  fi

  if [ -z "${run_name}" ]; then
    run_name="unknown_run"
  fi

  ckpt_stem="${checkpoint_name}"
  if [ -z "${ckpt_stem}" ] && [ -n "${checkpoint_path}" ]; then
    ckpt_stem="$(basename "${checkpoint_path}")"
  fi
  ckpt_stem="${ckpt_stem%.pt}"
  if [ -z "${ckpt_stem}" ]; then
    ckpt_stem="pretrained"
  fi

  out_dir="${RUN_DIR}/${setup}/${model}/${run_name}/${ckpt_stem}"
  mkdir -p "${out_dir}"

  eval_log="${out_dir}/eval.log"
  pred_csv="${out_dir}/predictions.csv"
  mis_csv="${out_dir}/misclassified.csv"
  metrics_json="${out_dir}/metrics.json"

  cmd=(
    python -m src.evaluation.closed_domain_eval
    --model "${model_name}"
    --dataset_loader "${DATASET_LOADER}"
    --csv "${EVAL_CSV}"
    --rank "${RANK}"
    --template_style "${TEMPLATE_STYLE}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --precision "${PRECISION}"
    --dump_topk "${DUMP_TOPK}"
    --dump_predictions_csv "${pred_csv}"
    --dump_misclassified_csv "${mis_csv}"
    --dump_metrics_json "${metrics_json}"
  )

  if [ -n "${pretrained_tag}" ]; then
    cmd+=(--pretrained "${pretrained_tag}")
  fi
  if [ -n "${checkpoint_path}" ]; then
    cmd+=(--checkpoint "${checkpoint_path}")
  fi
  if [ "${RANK_ONLY}" = "1" ]; then
    cmd+=(--rank_only)
  fi
  if [ "${NORMALIZE_ANTHOZOA}" = "1" ]; then
    cmd+=(--normalize_anthozoa)
  fi
  if [ -n "${CLASS_LIST}" ]; then
    cmd+=(--class_list "${CLASS_LIST}")
  fi
  if [ -n "${FILTER_RANK}" ] && [ -n "${FILTER_VALUE}" ]; then
    cmd+=(--filter_rank "${FILTER_RANK}" --filter_value "${FILTER_VALUE}")
  fi

  if [ "${DATASET_LOADER}" = "rsg" ]; then
    if [ -n "${RSG_TAXONOMY_CSV}" ]; then
      cmd+=(--rsg_taxonomy_csv "${RSG_TAXONOMY_CSV}")
    fi
    if [ "${RSG_CORALS_ONLY}" = "1" ]; then
      cmd+=(--rsg_corals_only)
    fi
    if [ -n "${RSG_PATH_KEY}" ]; then
      cmd+=(--rsg_path_key "${RSG_PATH_KEY}")
    fi
    if [ -n "${RSG_LABEL_KEY}" ]; then
      cmd+=(--rsg_label_key "${RSG_LABEL_KEY}")
    fi
  fi

  echo "========================================="
  echo "Eval setup=${setup} model=${model} run=${run_name} ckpt=${ckpt_stem}"
  echo "Checkpoint: ${checkpoint_path:-<pretrained>}"
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
  echo "${setup},${model},${run_name},${ckpt_stem},${checkpoint_path},${metrics_csv_line},${pred_csv},${metrics_json}" >> "${SUMMARY_CSV}"
done

echo "Done. Summary: ${SUMMARY_CSV}"
