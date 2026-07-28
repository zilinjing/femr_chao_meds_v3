#!/usr/bin/env bash
# Linear-probe evaluation of the ORA-Factorized model on all MIMIC-IV tasks, one GPU per stream.
#
# GPU layout (8 streams, ~8 GPUs):
#   0  in_hospital_mortality   (outcome,   AUROC)
#   1  long_los                (outcome,   AUROC)
#   2  readmission             (outcome,   AUROC)
#   3  ami                     (phenotype, AUROC + C-index)
#   4  masld                   (phenotype, AUROC + C-index)
#   5  stroke                  (phenotype, AUROC + C-index)
#   6  creatinine, glucose     (regression, R^2 - run back to back)
#   7  pao2, platelets         (regression, R^2 - run back to back)
#
# Note there is no separate time-to-event stream: finetune_motor.py computes the C-index inside
# the phenotype branch (`if args.cohort_type == "phenotype"`), reusing the very same features as
# the binary probe. So TTE costs no extra GPU work, which frees a GPU for regression.
#
# Each stream runs under CUDA_VISIBLE_DEVICES=<n> with --device cuda:0, so every process sees its
# own GPU as device 0. That sidesteps any cuda:0 assumption in the code and keeps one stream from
# allocating on another's GPU.
#
# Usage:
#   bash run_eval_mtpp_factorized.sh                # everything
#   bash run_eval_mtpp_factorized.sh outcome        # only the outcome classification tasks
#   bash run_eval_mtpp_factorized.sh phenotype      # only phenotype (AUROC + C-index)
#   bash run_eval_mtpp_factorized.sh regression     # only regression
set -uo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-/data/zilinjing/ethos_fm/envs/tte}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# ---------------------------------------------------------------- configuration
STORAGE="${STORAGE:-/data/zilinjing/ethos_fm/storage/ora}"
MODEL_NAME="${MODEL_NAME:-mtpp_factorized}"
MODEL_PATH="${MODEL_PATH:-$STORAGE/mtpp_factorized/transformer/best_model}"
MEDS_READER="${MEDS_READER:-$STORAGE/meds_v0.6_reader}"
ONTOLOGY_PATH="${ONTOLOGY_PATH:-$STORAGE/metadata/ontology.pkl}"
MAIN_SPLIT_PATH="${MAIN_SPLIT_PATH:-$STORAGE/mtpp_factorized/main_split.csv}"
COHORT_ROOT="${COHORT_ROOT:-$STORAGE/cohort}"
OUT_ROOT="${OUT_ROOT:-$STORAGE/results/$MODEL_NAME}"
CACHE_ROOT="${CACHE_ROOT:-$STORAGE/hf_cache_eval}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

# ORA-Factorized uses the hierarchical tokenizer, so use_ntp stays "no".
USE_NTP="${USE_NTP:-no}"
TOKENS_PER_BATCH="${TOKENS_PER_BATCH:-65536}"
MIN_SUBJECTS_PER_BATCH="${MIN_SUBJECTS_PER_BATCH:-8}"
# 8 streams share this node's 112 cores; 12 each keeps feature generation from thrashing.
NUM_PROC="${NUM_PROC:-12}"
# Phenotype settings from the previous runs; outcome tasks use no observation window.
PHENO_TRUNCATION="${PHENO_TRUNCATION:-300}"
PHENO_OBS_WINDOW="${PHENO_OBS_WINDOW:-730}"

WHICH="${1:-all}"

# ---------------------------------------------------------------- preflight
for p in "$MODEL_PATH" "$MEDS_READER" "$ONTOLOGY_PATH" "$MAIN_SPLIT_PATH" "$COHORT_ROOT"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p" >&2; exit 1; }
done
for f in model.safetensors dictionary.msgpack config.json; do
  [[ -e "$MODEL_PATH/$f" ]] || { echo "ERROR: $MODEL_PATH is missing $f" >&2; exit 1; }
done
mkdir -p "$OUT_ROOT" "$CACHE_ROOT" "$LOG_DIR"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [[ "$busy" -gt 0 ]]; then
  echo "WARNING: $busy process(es) already using the GPUs (pretraining still running?)."
  echo "         Evaluation will contend for memory. Ctrl-C now if that is not intended."
  sleep 10
fi

echo "=============================================================="
echo " Linear-probe evaluation | $MODEL_NAME"
echo "   model    : $MODEL_PATH"
echo "   cohorts  : $COHORT_ROOT"
echo "   results  : $OUT_ROOT"
echo "   running  : $WHICH"
echo "=============================================================="

# ---------------------------------------------------------------- streams
# $1 gpu, $2 task name
run_outcome() {
  local gpu="$1" task="$2"
  CUDA_VISIBLE_DEVICES="$gpu" bash run.sh "$task" \
    --meds_reader "$MEDS_READER" \
    --num_proc "$NUM_PROC" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --tokens_per_batch "$TOKENS_PER_BATCH" \
    --device cuda:0 \
    --use_ntp "$USE_NTP" \
    --cohort_type outcome \
    --cache_path "$CACHE_ROOT/$task" \
    --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
    --ontology_path "$ONTOLOGY_PATH" \
    --main_split_path "$MAIN_SPLIT_PATH" \
    --output_dir "$OUT_ROOT/outcome" \
    --classification_cohort "$COHORT_ROOT/outcome/cohort/" \
    > "$LOG_DIR/${task}.log" 2>&1
}

# $1 gpu, $2 task name. Produces AUROC *and* C-index in one pass.
run_phenotype() {
  local gpu="$1" task="$2"
  CUDA_VISIBLE_DEVICES="$gpu" bash run.sh "$task" \
    --meds_reader "$MEDS_READER" \
    --num_proc "$NUM_PROC" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --tokens_per_batch "$TOKENS_PER_BATCH" \
    --device cuda:0 \
    --use_ntp "$USE_NTP" \
    --cohort_type phenotype \
    --cache_path "$CACHE_ROOT/$task" \
    --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
    --truncation "$PHENO_TRUNCATION" \
    --observation_window "$PHENO_OBS_WINDOW" \
    --ontology_path "$ONTOLOGY_PATH" \
    --main_split_path "$MAIN_SPLIT_PATH" \
    --output_dir "$OUT_ROOT/phenotype" \
    --classification_cohort "$COHORT_ROOT/phenotype/binary/" \
    --tte_cohort "$COHORT_ROOT/phenotype/tte/" \
    > "$LOG_DIR/${task}.log" 2>&1
}

# $1 gpu, $2 comma-separated task list (run sequentially inside the stream)
run_regression() {
  local gpu="$1" tasks="$2"
  CUDA_VISIBLE_DEVICES="$gpu" bash regression/run_regression.sh \
    --meds_reader "$MEDS_READER" \
    --num_proc "$NUM_PROC" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --tokens_per_batch "$TOKENS_PER_BATCH" \
    --device cuda:0 \
    --use_ntp "$USE_NTP" \
    --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
    --ontology_path "$ONTOLOGY_PATH" \
    --main_split_path "$MAIN_SPLIT_PATH" \
    --cohort_dir "$COHORT_ROOT/regression/icu_stay_4h_100k/" \
    --output_dir "$OUT_ROOT/regression" \
    --cache_path "$CACHE_ROOT/regression_gpu${gpu}" \
    --tasks "$tasks" \
    --regression \
    > "$LOG_DIR/regression_gpu${gpu}.log" 2>&1
}

declare -a PIDS=() NAMES=()
launch() { "$@" & PIDS+=($!); NAMES+=("$2${3:+ $3}"); }

if [[ "$WHICH" == "all" || "$WHICH" == "outcome" ]]; then
  launch run_outcome 0 in_hospital_mortality
  launch run_outcome 1 long_los
  launch run_outcome 2 readmission
fi
if [[ "$WHICH" == "all" || "$WHICH" == "phenotype" ]]; then
  launch run_phenotype 3 ami
  launch run_phenotype 4 masld
  launch run_phenotype 5 stroke
fi
if [[ "$WHICH" == "all" || "$WHICH" == "regression" ]]; then
  launch run_regression 6 "creatinine,glucose"
  launch run_regression 7 "pao2,platelets"
fi

echo "launched ${#PIDS[@]} streams; logs in $LOG_DIR"
printf '  %s\n' "${NAMES[@]}"

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done] ${NAMES[$i]}"
  else
    echo "[FAILED] ${NAMES[$i]} -- see $LOG_DIR"
    FAILED=$((FAILED + 1))
  fi
done

echo "=============================================================="
echo "finished: $(( ${#PIDS[@]} - FAILED ))/${#PIDS[@]} streams succeeded"
echo "results under $OUT_ROOT"
echo "collect metrics with:  bash collect_eval_results.sh $OUT_ROOT"
exit $(( FAILED > 0 ? 1 : 0 ))
