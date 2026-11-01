#!/usr/bin/env bash
# Value-aware tabular feature extraction for the strengthened MIMIC-IV baseline.
#
# Same cohorts as evaluation/baseline/extract.sh, repointed at the current
# cluster layout and writing to a new output root so the submitted feature
# pickles are preserved and both baseline rows can be reported.
#
# Feature spec: MEDS-Tab aggregation set
#   code/count, value/count, value/sum, value/sum_sqd, value/min, value/max
# over nested look-back windows [7d, full]. 12 columns per numeric code.
#
# Usage:  bash extract.sh              # all 9 cohorts
#         bash extract.sh tte          # phenotype TTE only
#         bash extract.sh binary       # phenotype binary only
#         bash extract.sh outcome      # patient outcomes only
#
# Env overrides: PYTHON, NUM_THREADS, MIN_SUPPORT, WINDOWS

set -uo pipefail

# The `femr_ehr` conda env referenced by the original extract.sh does not exist on
# this cluster. `envs/tte` has meds_reader + an editable install of this femr, which
# is all feature extraction needs. Override with:  PYTHON=/path/to/python bash extract.sh
PYTHON="${PYTHON:-/data/zilinjing/ethos_fm/envs/tte/bin/python}"
"${PYTHON}" -c "import femr, meds_reader" || {
  echo "ERROR: ${PYTHON} cannot import femr/meds_reader" >&2; exit 1; }

MODULE="femr.evaluation.baseline_value.generate_tabular_features"
WINDOWS="${WINDOWS:-7,full}"
MIN_SUPPORT="${MIN_SUPPORT:-10}"
# Lower this while other I/O-heavy jobs are running: the preprocess pass is bound
# by reads from the MEDS database, not by CPU, so extra threads mostly add
# contention.  NUM_THREADS=8 bash extract.sh
NUM_THREADS="${NUM_THREADS:-32}"

ROOT=/data/zilinjing/ethos_fm/storage/ora
MEDS_READER="${ROOT}/meds_v0.6_reader"
COHORT="${ROOT}/cohort"
OUT="${ROOT}/meds_tab"

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
  for name in ami stroke masld; do
    extract "${COHORT}/phenotype/tte/${name}/" "${OUT}/${name}/"
  done
fi

# ----------------------------------------------------------------------------
# Phenotypes -- binary  (XGBoost)
# ----------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "binary" ]]; then
  for name in ami stroke masld; do
    extract "${COHORT}/phenotype/binary/${name}/" "${OUT}/${name}_binary/"
  done
fi

# ----------------------------------------------------------------------------
# Patient outcomes -- binary  (XGBoost)
# ----------------------------------------------------------------------------
if [[ "${WHICH}" == "all" || "${WHICH}" == "outcome" ]]; then
  for name in in_hospital_mortality long_los readmission; do
    extract "${COHORT}/outcome/cohort/${name}/" "${OUT}/${name}_binary/"
  done
fi

echo "=============================================================="
echo "Done. Extraction stats: ${OUT}/*/features/*_extraction_stats.json"
echo "Next:  cd $(dirname "$0") && python run_all.py --n_jobs 24"
