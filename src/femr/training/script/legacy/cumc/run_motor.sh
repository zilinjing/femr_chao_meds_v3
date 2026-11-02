#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."  # prepare.py/pretrain.py live two levels up

# ---- config (edit if you like) ----
# LOG_DIR="${LOG_DIR:-./logs}"
# PREP_LOG="$LOG_DIR/prepare_motor_$(date +%Y%m%d_%H%M%S).log"
# TRAIN_LOG="$LOG_DIR/train_mamba_$(date +%Y%m%d_%H%M%S).log"
# mkdir -p "$LOG_DIR"

# echo "[$(date)] starting both jobs..."
# echo "Logs:"
# echo "  prepare -> $PREP_LOG"
# echo "  train   -> $TRAIN_LOG"
# echo

python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/motor \
  --ontology_path  /data2/processed_datasets/zj2398/femr/cumc/motor/ontology.pkl \
  --loss_type motor \
  --tokens_per_batch 8192 \
  --model transformer \
  --num_threads 64 \
  --hf_cache_path /data2/huggingface/zj2398 \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader


  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=0,1


accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/motor \
  --meds_reader  /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/cumc/motor/transformer \
  --hf_run_name cumc_motor \
  --model transformer \
  --loss_type motor
