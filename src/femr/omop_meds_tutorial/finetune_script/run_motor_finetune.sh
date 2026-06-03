#!/bin/bash
set -e

# ============================================================
# MOTOR fine-tuning launcher (flag-based)
#
# Usage:
#   bash run_motor_finetune.sh --task <stroke|masld|readmission> [flags...]

# bash run_motor_finetune.sh \
#     --task stroke --device cuda:0 \
#     --use_wandb --wandb_project ehr_benchmark_finetune_mimic
#
# All inputs are passed as --flags so the paths can be edited per cluster.
# Every flag has a default pointing at the CURRENT cluster; override any of
# them when running elsewhere.
#
# Flags:
#   --task             stroke | masld | readmission   (required; selects cohort subdir + obs window)
#   --device           torch device                    (default: cuda:0)
#   --model_path       pretrained checkpoint dir (also the tokenizer dir unless --tokenizer_path is set)
#   --tokenizer_path   dir holding dictionary.msgpack  (default: --model_path)
#   --meds_reader      meds_reader database dir
#   --cohort_base      dir containing <task>/{train,tuning,held_out}.parquet
#   --ontology_path    ontology pickle
#   --main_split_path  main_split.csv (unused for presplit cohorts, but passed through)
#   --output_dir       results dir                     (default: <cluster>/finetune_model/<task>)
#   --cache_dir        local-disk HF batch cache, ~17GB/task (default: $HOME/femr_ft_cache/<task>)
#   --num_proc         worker processes                (default: 16)
#   <anything else>    forwarded to finetuning.py (e.g. --debug, --lr_ratios 0.05,0.5)
#
# Examples:
# bash run_motor_finetune.sh \
#     --task stroke \
#     --device cuda:0 \
#     --model_path /data/models/zj2398/femr/mimic_motor_birth/transformer/best_134160 \
#     --tokenizer_path  /data/models/zj2398/femr/mimic_motor_birth/transformer/best_134160 \
#     --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#     --cohort_base   /data2/processed_datasets/mimic/fomoh_subsample_100k \
#     --ontology_path   /data/models/zj2398/femr/mimic_motor_birth/ontology.pkl \
#     --main_split_path /data/models/zj2398/femr/mimic_motor_birth/main_split.csv \
#     --output_dir /data/models/zj2398/femr/mimic_motor_birth/finetune_model/stroke \
#     --cache_dir /data2/huggingface/zj2398 \
#     --num_proc 32 \
#     --use_wandb \
#     --wandb_project ehr_benchmark_finetune_mimic
# ============================================================

# ---------- Defaults (current cluster) ----------
TASK=""
DEVICE="cuda:0"
MODEL_PATH="/mnt/hanoverdev/scratch/t-clb-zilin/motor/motor_birth/best_134160"
TOKENIZER_PATH="/mnt/hanoverdev/scratch/t-clb-zilin/motor/motor_birth/best_134160"
MEDS_READER="/mnt/hanoverdev/scratch/t-clb-zilin/motor/MEDS_cohort-reader"
COHORT_BASE="/mnt/hanoverdev/scratch/t-clb-zilin/motor/fomoh_subsample_100k"
ONTOLOGY_PATH="/mnt/hanoverdev/scratch/t-clb-zilin/motor/motor_pre/ontology.pkl"
MAIN_SPLIT_PATH="/mnt/hanoverdev/scratch/t-clb-zilin/motor/motor_pre/main_split.csv"
OUTPUT_DIR=""                                                        # default set per-task after parsing (see below)
CACHE_DIR=""                                                         # default set per-task after parsing (see below)
NUM_PROC=24
EXTRA_ARGS=()



# ---------- Parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --model_path)      MODEL_PATH="$2"; shift 2 ;;
        --tokenizer_path)  TOKENIZER_PATH="$2"; shift 2 ;;
        --meds_reader)     MEDS_READER="$2"; shift 2 ;;
        --cohort_base)     COHORT_BASE="$2"; shift 2 ;;
        --ontology_path)   ONTOLOGY_PATH="$2"; shift 2 ;;
        --main_split_path) MAIN_SPLIT_PATH="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --cache_dir)       CACHE_DIR="$2"; shift 2 ;;
        --num_proc)        NUM_PROC="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;   # forward unknown flags to finetuning.py
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: bash $0 --task <stroke|masld|readmission> [--device cuda:0] [--model_path ...] [...]"
    exit 1
fi

# ---------- Derived / less-frequently-changed ----------
# TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"                       # checkpoint dir holds dictionary.msgpack
COHORT_DIR="$COHORT_BASE/$TASK"
# OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hanoverdev/scratch/t-clb-zilin/motor/finetune_model/$TASK}"
# CACHE_DIR="${CACHE_DIR:-$HOME/femr_ft_cache/$TASK}"                  # local disk, not the blobfuse2 scratch

# ---------- Observation window: phenotype=730, outcome=full history ----------
case "$TASK" in
    stroke|masld|scz|schizophrenia)
        OBS_WINDOW=730 ;;
    *)
        OBS_WINDOW=-1 ;;   # <=0 => full history (None), e.g. readmission
esac

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

echo "========================================"
echo "Task:          $TASK"
echo "Device:        $DEVICE"
echo "Model:         $MODEL_PATH"
echo "Tokenizer:     $TOKENIZER_PATH"
echo "MEDS reader:   $MEDS_READER"
echo "Cohort:        $COHORT_DIR"
echo "Ontology:      $ONTOLOGY_PATH"
echo "Main split:    $MAIN_SPLIT_PATH"
echo "Output:        $OUTPUT_DIR"
echo "Cache:         $CACHE_DIR"
echo "Obs window:    $OBS_WINDOW"
echo "Extra args:    ${EXTRA_ARGS[*]}"
echo "========================================"

# ---------- Verify required paths exist ----------
for p in "$COHORT_DIR" "$MODEL_PATH" "$MEDS_READER" "$TOKENIZER_PATH"; do
    if [ ! -d "$p" ]; then echo "ERROR: directory does not exist: $p"; exit 1; fi
done
if [ ! -f "$ONTOLOGY_PATH" ]; then echo "ERROR: ontology file does not exist: $ONTOLOGY_PATH"; exit 1; fi

python -u -m femr.omop_meds_tutorial.finetune_script.finetuning \
    --meds_reader "${MEDS_READER}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --ontology_path "${ONTOLOGY_PATH}" \
    --main_split_path "${MAIN_SPLIT_PATH}" \
    --cohort_dir "${COHORT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --task_name "${TASK}" \
    --device "${DEVICE}" \
    --cache_dir "${CACHE_DIR}" \
    --observation_window "${OBS_WINDOW}" \
    --num_proc "${NUM_PROC}" \
    --tokens_per_batch 65536 \
    --min_subjects_per_batch 8 \
    --num_epochs 10 \
    --weight_decay 0.1 \
    --pretraining_lr 0.00001 \
    "${EXTRA_ARGS[@]}"
