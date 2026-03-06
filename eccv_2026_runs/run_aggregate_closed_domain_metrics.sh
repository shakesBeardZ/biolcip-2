#!/bin/bash
set -euo pipefail

# Configurable wrapper for tools/aggregate_closed_domain_metrics.py

CONDA_BASE=${CONDA_BASE:-"/home/${USER}/anaconda3"}
CONDA_ENV=${CONDA_ENV:-"pytoenv"}

BASE_DIR=${BASE_DIR:-"/raid/felembaa/reefnet/logs_eval/closed_domain_checkpoint_matrix"}
OUTPUT_CSV=${OUTPUT_CSV:-"${BASE_DIR}/metrics_table.csv"}
BEST_ACC1_CSV=${BEST_ACC1_CSV:-"${BASE_DIR}/best_by_setup_model_acc1.csv"}
BEST_MACRO_RECALL_CSV=${BEST_MACRO_RECALL_CSV:-"${BASE_DIR}/best_by_setup_model_macro_recall.csv"}

PREDICTIONS_NAME=${PREDICTIONS_NAME:-"predictions.csv"}
WRITE_SUMMARY_JSON=${WRITE_SUMMARY_JSON:-0}
SUMMARY_JSON_NAME=${SUMMARY_JSON_NAME:-"new_summary.json"}
STRICT=${STRICT:-0}

# Optional filters; leave empty for all.
SETUPS=${SETUPS:-"full k20 k10 k1 k5 zero_shot"}  # e.g. "full k1 k5 k10 k20 zero_shot"
MODELS=${MODELS:-"bioclip clip openclip"}  # e.g. "bioclip clip openclip"

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  echo "Unable to locate conda.sh at ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

cmd=(
  python tools/aggregate_closed_domain_metrics.py
  --base-dir "${BASE_DIR}"
  --predictions-name "${PREDICTIONS_NAME}"
  --output-csv "${OUTPUT_CSV}"
  --best-acc1-csv "${BEST_ACC1_CSV}"
  --best-macro-recall-csv "${BEST_MACRO_RECALL_CSV}"
  --summary-json-name "${SUMMARY_JSON_NAME}"
)

if [ "${WRITE_SUMMARY_JSON}" = "1" ]; then
  cmd+=(--write-summary-json)
fi
if [ "${STRICT}" = "1" ]; then
  cmd+=(--strict)
fi

if [ -n "${SETUPS}" ]; then
  # shellcheck disable=SC2206
  setups_array=(${SETUPS})
  cmd+=(--setups "${setups_array[@]}")
fi
if [ -n "${MODELS}" ]; then
  # shellcheck disable=SC2206
  models_array=(${MODELS})
  cmd+=(--models "${models_array[@]}")
fi

echo "Running aggregation with:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  OUTPUT_CSV=${OUTPUT_CSV}"
echo "  BEST_ACC1_CSV=${BEST_ACC1_CSV}"
echo "  BEST_MACRO_RECALL_CSV=${BEST_MACRO_RECALL_CSV}"
"${cmd[@]}"
