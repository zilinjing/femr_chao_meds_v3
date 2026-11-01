#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."  # prepare.py/pretrain.py live one level up

# ---- config (edit if you like) ----
# LOG_DIR="${LOG_DIR:-./logs}"
# PREP_LOG="$LOG_DIR/prepare_motor_$(date +%Y%m%d_%H%M%S).log"
# TRAIN_LOG="$LOG_DIR/train_mamba_$(date +%Y%m%d_%H%M%S).log"
# mkdir -p "$LOG_DIR"

echo "[$(date)] starting both jobs..."
# echo "Logs:"
# echo "  prepare -> $PREP_LOG"
# echo "  train   -> $TRAIN_LOG"
# echo

# kuvira
python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/ntp \
  --ontology_path  /data2/processed_datasets/zj2398/femr/mimic/ntp/ontology.pkl \
  --loss_type ntp \
  --num_threads 32 \
  --model mamba \
  --hf_cache_path /data/huggingface/zj2398 \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader

# python pretrain.py \
#   --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/ntp \
#   --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/transformer \
#   --model transformer \
#   --hf_run_name mimic_transformer_ntp \
#   --loss_type ntp


# gsb
# python prepare.py \
#   --pretraining_data /user/zj2398/cache/ntp \
#   --ontology_path  /user/zj2398/cache/ntp/ontology.pkl \
#   --loss_type ntp \
#   --num_threads 32 \
#   --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader


# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=1,2
accelerate launch \
  --num_processes 2 \
torchrun \
  --nproc_per_node=2 \
  pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/ntp \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/mamba \
  --checkpoint_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/mamba/checkpoint-1848322 \
  --model mamba \
  --learning_rate 2e-4 \
  --max_grad_norm 5.0 \
  --n_layers 28 \
  --hf_run_name cumc_mamba_ntp_2e-4_5.0_continue \
  --loss_type ntp

  python pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/ntp \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/mamba \
  --checkpoint_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/mamba/checkpoint-1848322 \
  --model mamba \
  --learning_rate 2e-4 \
  --max_grad_norm 5.0 \
  --n_layers 28 \
  --hf_run_name cumc_mamba_ntp_2e-4_5.0_continue \
  --loss_type ntp



# accelerate launch \
#   --num_processes 2 \
#   pretrain.py \
#   --pretraining_data /user/zj2398/cache/ntp \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /shared/share_mala/zj2398/models/ntp_8k/transformer \
#   --model transformer \
#   # --n_layers 28 \
#   --hf_run_name mimic_transformer_ntp_shared_6100 \
#   --loss_type ntp



