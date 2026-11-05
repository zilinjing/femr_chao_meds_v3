#!/usr/bin/env bash
# Value-aware tabular (MEDS-Tab) feature extraction for the XGBoost / DeepHit baselines.
#
# Usually started by script/<dataset>/baseline.sh extract, which sets the paths below from the
# dataset's env file (see evaluation/README.md). Every task folder of each cohort is extracted.
#
# Feature spec: MEDS-Tab aggregation set
#   code/count, value/count, value/sum, value/sum_sqd, value/min, value/max
# over nested look-back windows [7d, full]. 12 columns per numeric code.
#
# Usage:  bash extract.sh              # all cohorts
#         bash extract.sh tte          # phenotype TTE only
#         bash extract.sh binary       # phenotype binary only
#         bash extract.sh outcome      # patient outcomes only
#
# Required env: MEDS_READER, OUTCOME_COHORT, PHENOTYPE_BINARY_COHORT, PHENOTYPE_TTE_COHORT,
#               MEDS_TAB_DIR (output)
# Env overrides: PYTHON, NUM_THREADS, MIN_SUPPORT, WINDOWS

set -uo pipefail

# Feature extraction needs only meds_reader + this femr (the `tte` conda env).
PYTHON="${PYTHON:-python}"
"${PYTHON}" -c "import femr, meds_reader" || {
  echo "ERROR: ${PYTHON} cannot import femr/meds_reader" >&2; exit 1; }

MODULE="femr.evaluation.baseline.meds_tab.generate_tabular_features"
WINDOWS="${WINDOWS:-7,full}"
MIN_SUPPORT="${MIN_SUPPORT:-10}"
# Lower this while other I/O-heavy jobs are running: the preprocess pass is bound
# by reads from the MEDS database, not by CPU, so extra threads mostly add
# contention.  NUM_THREADS=8 bash extract.sh
NUM_THREADS="${NUM_THREADS:-32}"

OUT="${MEDS_TAB_DIR}"

# task folders of a cohort directory
tasks_in () { find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort; }

extract () {
  local task_dir="$1" out_dir="$2"
  echo "=============================================================="
  echo ">>> ${out_dir}"
  "${PYTHON}" -u -m "${MODULE}" \
    --pretraining_data "${out_dir}" \
    --meds_reader "${MEDS_READER}" \
    --cohort_dir "${task_dir}" \
    --windows "${WINDOWS}" \
    --min_support "${MIN_SUPPORT}" \
    --num_threads "${NUM_THREADS}"
}

WHICH="${1:-all}"

# ----------------------------------------------------------------------------
# Phenotypes -- time-to-event  (DeepHit)
# ----------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "tte" ]]; then
  for name in $(tasks_in "${PHENOTYPE_TTE_COHORT}"); do
    extract "${PHENOTYPE_TTE_COHORT}/${name}/" "${OUT}/${name}/"
  done
fi

# ----------------------------------------------------------------------------
# Phenotypes -- binary  (XGBoost)
# ----------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "binary" ]]; then
  for name in $(tasks_in "${PHENOTYPE_BINARY_COHORT}"); do
    extract "${PHENOTYPE_BINARY_COHORT}/${name}/" "${OUT}/${name}_binary/"
  done
fi

# ----------------------------------------------------------------------------
# Patient outcomes -- binary  (XGBoost)
# ----------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "outcome" ]]; then
  for name in $(tasks_in "${OUTCOME_COHORT}"); do
    extract "${OUTCOME_COHORT}/${name}/" "${OUT}/${name}_binary/"
  done
fi

echo "=============================================================="
echo "Done. Extraction stats: ${OUT}/*/features/*_extraction_stats.json"
echo "Next:  bash <script/.../baseline.sh> outcome | phenotype"
