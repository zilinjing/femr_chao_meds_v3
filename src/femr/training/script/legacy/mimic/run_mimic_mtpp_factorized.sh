#!/usr/bin/env bash
# Pretrain the ORA-Factorized ablation (loss_type=mtpp_factorized) on MIMIC-IV.
#
# ORA-Factorized answers Reviewer 1 Q2: it is ORA with the time-value table forced to be a
# multiplication table (one distribution for *when*, one for *what*), i.e. the "two separate
# losses" design.  It reuses ORA's task definition (mtpp_task.pkl) and batches unchanged, so
# the only difference between the two arms is the head.
#
# Usage:   bash script/mimic/run_mimic_mtpp_factorized.sh
# Resume:  bash script/mimic/run_mimic_mtpp_factorized.sh   (prepare skips finished artifacts; training
#          picks up with CHECKPOINT_DIR=<path> if you pass one)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."  # prepare.py/pretrain.py live two levels up

# ---------------------------------------------------------------- configuration
# Activate the `tte` environment unless one is already active.
VENV="${VENV:-/data/zilinjing/ethos_fm/envs/tte}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

STORAGE="${STORAGE:-/data/zilinjing/ethos_fm/storage/ora}"

# Inputs (already on this node).
MEDS_READER="${MEDS_READER:-$STORAGE/meds_v0.6_reader}"
ONTOLOGY_PATH="${ONTOLOGY_PATH:-$STORAGE/metadata/ontology.pkl}"

# Outputs: tokenizer, mtpp_task.pkl, train_batches, val_batches, main_split.csv, and the model.
#
# FAIR COMPARISON: this directory holds everything the loss does not touch.  To pretrain ORA
# (mtpp_shared) against a bit-for-bit identical setup, point its run at this same
# PRETRAINING_DATA and only change --loss_type and --output_dir (see the note at the bottom).
PRETRAINING_DATA="${PRETRAINING_DATA:-$STORAGE/mtpp_factorized}"
OUTPUT_DIR="${OUTPUT_DIR:-$PRETRAINING_DATA/transformer}"
HF_CACHE_PATH="${HF_CACHE_PATH:-$STORAGE/hf_cache}"

# Discretisation and vocabulary: identical to the ORA runs in the paper (T=8, V=5).
TIME_BINS="${TIME_BINS:-8}"
VALUE_BINS="${VALUE_BINS:-5}"
NUM_FINAL_TASKS="${NUM_FINAL_TASKS:-6100}"
NUM_THREADS="${NUM_THREADS:-32}"

# Training: all 8 GPUs, one process each.  Note this sets the effective batch size to
# 8 x PER_DEVICE_TRAIN_BATCH_SIZE, so the companion ORA run must use the same 8 processes (and
# therefore run after this one, not alongside it) for the comparison to stay controlled.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
# n_epochs, learning rate and early stopping are left at pretrain.py's defaults so they match
# the ORA runs (20 epochs, lr 1e-5 for the transformer, early stopping patience 3).
#
# The lowest-eval_loss model is mirrored to $OUTPUT_DIR/best_model (weights + config + tokenizer +
# best_model_info.json) every time validation loss improves, and rewritten once more when training
# ends -- whether early stopping fired or all 20 epochs ran.  Point evaluation --model_path there.

# Weights & Biases.  NOTE: pretrain.py currently sets report_to="none", so nothing is logged
# and RUN_NAME is only recorded in the HF trainer state.  To actually log, change
# `report_to="none"` to `report_to="wandb"` in pretrain.py (~line 359); these exports are then
# picked up automatically by the HF Trainer integration.
RUN_NAME="${RUN_NAME:-mimic_transformer_mtpp_factorized_${NUM_FINAL_TASKS}_T${TIME_BINS}_V${VALUE_BINS}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
export WANDB_NAME="$RUN_NAME"

LOG_DIR="${LOG_DIR:-$PRETRAINING_DATA/logs}"

# ---------------------------------------------------------------- preflight
for path in "$MEDS_READER" "$ONTOLOGY_PATH" "$MEDS_READER/metadata/subject_splits.parquet"; do
  [[ -e "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 1; }
done
mkdir -p "$PRETRAINING_DATA" "$HF_CACHE_PATH" "$LOG_DIR"

# pretrain.py reads the ontology from $PRETRAINING_DATA/ontology.pkl rather than from a flag, so
# link the shared one into place instead of copying 871 MB.
[[ -e "$PRETRAINING_DATA/ontology.pkl" ]] || ln -s "$ONTOLOGY_PATH" "$PRETRAINING_DATA/ontology.pkl"

echo "=============================================================="
echo " ORA-Factorized (mtpp_factorized) | MIMIC-IV | Transformer"
echo "   meds_reader      : $MEDS_READER"
echo "   ontology         : $ONTOLOGY_PATH"
echo "   pretraining_data : $PRETRAINING_DATA"
echo "   output_dir       : $OUTPUT_DIR"
echo "   T x V            : $TIME_BINS x $VALUE_BINS   (tasks: $NUM_FINAL_TASKS)"
echo "   GPUs             : $GPUS  ($NUM_PROCESSES processes)"
echo "   run name         : $RUN_NAME"
echo "=============================================================="

# ---------------------------------------------------------------- 1. data preparation
# --loss_type mtpp is intentional: it writes mtpp_task.pkl, which is the task file both ORA
# (mtpp_shared) and ORA-Factorized load.  prepare.py skips any artifact that already exists,
# so re-running this script is safe.
echo "[$(date)] step 1/2: preparing tokenizer, task and batches"
python -u prepare.py \
  --pretraining_data "$PRETRAINING_DATA" \
  --ontology_path "$ONTOLOGY_PATH" \
  --meds_reader "$MEDS_READER" \
  --time_bins "$TIME_BINS" \
  --value_bins "$VALUE_BINS" \
  --loss_type mtpp \
  --num_final_tasks "$NUM_FINAL_TASKS" \
  --num_threads "$NUM_THREADS" \
  --hf_cache_path "$HF_CACHE_PATH" \
  --model transformer \
  2>&1 | tee "$LOG_DIR/prepare_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------- 2. pretraining
# Set CHECKPOINT_DIR=<path to a checkpoint-* dir> to resume instead of starting over.
RESUME_ARGS=()
[[ -n "${CHECKPOINT_DIR:-}" ]] && RESUME_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR")

echo "[$(date)] step 2/2: pretraining on GPUs $GPUS ${CHECKPOINT_DIR:+(resuming from $CHECKPOINT_DIR)}"
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  pretrain.py \
  --pretraining_data "$PRETRAINING_DATA" \
  --meds_reader "$MEDS_READER" \
  --output_dir "$OUTPUT_DIR" \
  --model transformer \
  --loss_type mtpp_factorized \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --hf_run_name "$RUN_NAME" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] done. model -> $OUTPUT_DIR"

# ---------------------------------------------------------------- companion ORA run
# To pretrain the ORA arm on exactly the same tokenizer, task, bins and batches (the fair
# comparison for Reviewer 1 Q2), run this *after* the factorized run finishes, with the same 8
# processes so the effective batch size matches, changing only the loss and the output dir:
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
#     --num_processes 8 \
#     pretrain.py \
#     --pretraining_data /data/zilinjing/ethos_fm/storage/ora/mtpp_factorized \
#     --meds_reader     /data/zilinjing/ethos_fm/storage/ora/meds_v0.6_reader \
#     --output_dir      /data/zilinjing/ethos_fm/storage/ora/mtpp_shared/transformer \
#     --model transformer \
#     --loss_type mtpp_shared \
#     --per_device_train_batch_size 1 \
#     --hf_run_name mimic_transformer_mtpp_shared_6100_T8_V5_$(date +%Y%m%d_%H%M%S)
#
# For the appendix, grep the training log for:
#   "Model param count:"  -> head= should exceed ORA's head parameter count
#   the loss curve / eval_loss -> comparable to ORA's, since both average per (point, code) pair
