#!/usr/bin/env bash
set -euo pipefail

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

TIME_BINS=$1
VALUE_BINS=$2
PRETRAINING_DATA=$3 
ONTOLOGY_PATH=$4

# ---- job 1: data prep (background) ----
# python prepare.py \
#   --pretraining_data $PRETRAINING_DATA \
#   --ontology_path  $ONTOLOGY_PATH \
#   --time_bins $TIME_BINS \
#   --value_bins $VALUE_BINS \
#   --loss_type mtpp \
#   --num_final_tasks 6100 \
#   --num_threads 32 \
#   --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader

python prepare.py \
  --pretraining_data $PRETRAINING_DATA \
  --ontology_path  $ONTOLOGY_PATH \
  --loss_type motor \
  --num_final_tasks 6100 \
  --num_threads 32 \
  --hf_cache_path /data2/huggingface/zj2398 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader


# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=1

# accelerate launch \
#   --num_processes 4 \
#   pretrain.py \
#   --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /user/zj2398/cache/mtpp_all_shared_6100_bin_5/mamba \
#   --model mamba \
#   --hf_run_name mimic_mamba_mtpp_shared_6100_bin_5 \
#   --loss_type mtpp_shared

# accelerate launch \
#   --num_processes 3 \

python pretrain.py \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/mimic/motor/transformer \
  --model transformer \
  --n_epochs 2 \
  --hf_run_name mimic_transformer_motor_6100_test \
  --loss_type motor


#bash run_mimic_motor.sh 8 5 /data2/processed_datasets/zj2398/femr/mimic/motor /data2/processed_datasets/zj2398/femr/mimic/motor/ontology.pkl

# python pretrain.py \
#   --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/motor  \
#   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data2/processed_datasets/zj2398/femr/mimic/motor/transformer \
#   --model transformer \
#   --n_epochs 2 \
#   --hf_run_name mimic_transformer_motor_6100_test \
#   --loss_type motor