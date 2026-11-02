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

TIME_BINS=$1
VALUE_BINS=$2

# ---- job 1: data prep (background) ----
python prepare.py \
  --pretraining_data /user/zj2398/cache/mtpp_8k_T_8_V_10 \
  --ontology_path  /user/zj2398/cache/mtpp_8k_T_8_V_10/ontology.pkl \
  --time_bins $TIME_BINS \
  --value_bins $VALUE_BINS \
  --loss_type mtpp \
  --num_final_tasks 6100 \
  --num_threads 32 \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader


# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=4,5,6,7

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

accelerate launch \
  --num_processes 5 \
  pretrain.py \
  --pretraining_data /user/zj2398/cache/mtpp_8k_T_8_V_10 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/mtpp_8k_T_8_V_10/transformer \
  --model transformer \
  --hf_run_name mimic_transformer_mtpp_6100_T_8_V_10 \
  --loss_type mtpp_shared

#bash script/mimic/run_mimic.sh 16 5