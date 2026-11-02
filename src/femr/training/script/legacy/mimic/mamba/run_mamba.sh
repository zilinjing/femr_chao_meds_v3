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
  --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/mtpp_bin_10 \
  --ontology_path /data2/processed_datasets/zj2398/femr/mimic/mtpp_bin_10/ontology.pkl \
  --time_bins 8 \
  --num_threads 80 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 8192 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /data/huggingface/zj2398 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader

  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=0,1,3,5

# accelerate launch \
#   --num_processes 4 \
#   pretrain.py \
#     --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
#     --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#     --per_device_train_batch_size 1 \
#     --learning_rate 5e-6 \
#     --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba  \
#     --hf_run_name mimic_mamba_revised_mtpp_6100_grad_clip \
#     --model mamba \
#     --n_layers 28 \
#     --loss_type mtpp_shared

  # accelerate launch \
  #   --num_processes 6 \
  #   pretrain.py \
  #   --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  #   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  #   --per_device_train_batch_size 1 \
  #   --learning_rate 1e-5 \
  #   --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba  \
  #   --hf_run_name mimic_mamba_revised_mtpp_6100_1e-5 \
  #   --model mamba \
  #   --n_layers 28 \
  #   --loss_type mtpp_shared

# export CUDA_VISIBLE_DEVICES=0,1,2

  # accelerate launch \
  #   --num_processes 3 \
  #   pretrain.py \
  #   --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  #   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  #   --per_device_train_batch_size 1 \
  #   --learning_rate 8e-6 \
  #   --max_grad_norm 8.0 \
  #   --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba_8e-6_5.0_cosine  \
  #   --hf_run_name mimic_mamba_revised_mtpp_6100_8e-6_5.0_cosine \
  #   --checkpoint_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba_8e-6_5.0_cosine/checkpoint-42728 \
  #   --model mamba \
  #   --n_layers 28 \
  #   --loss_type mtpp_shared

# export CUDA_VISIBLE_DEVICES=4,5
  accelerate launch \
    --num_processes 4 \
    pretrain.py \
    --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
    --meds_reader xxx \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --max_grad_norm 5.0 \
    --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba_2e-4_5.0_cosine  \
    --hf_run_name mimic_mamba_revised_mtpp_6100_2e-4_5.0_cosine \
    --model mamba \
    --n_layers 28 \
    --loss_type mtpp_shared

  accelerate launch \
    --num_processes 3 \
    pretrain.py \
    --pretraining_data /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 5.0 \
    --per_device_train_batch_size 1 \
    --output_dir /user/zj2398/cache/mtpp_8k_T_8_V_5_all_nonnume/mamba  \
    --model mamba \
    --loss_type mtpp_shared \
    --n_epochs 20 \
    --n_layers 28 \
    --hf_run_name mimic_mtpp_all_nonnumerical_mamba_2e-4_5.0 

# accelerate launch \
#     --num_processes 3 \
    python pretrain.py \
    --pretraining_data /user/zj2398/cache/icarefm \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --learning_rate 2e-4 \
    --max_grad_norm 5.0 \
    --per_device_train_batch_size 1 \
    --output_dir /user/zj2398/cache/icarefm/mamba \
    --model mamba \
    --loss_type icarefm \
    --n_epochs 20 \
    --n_layers 28 \
    --hf_run_name mimic_icarefm_mamba_2e-4_5.0_value_1000

  # python pretrain.py  \
  # --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5   \
  # --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader  \
  # --per_device_train_batch_size 1   \
  # --output_dir /user/zj2398/cache/mtpp_all_shared_6100_bin_5/mamba \
  # --model mamba  \
  # --n_layers 28  \
  # --hf_run_name mimic_mamba_mtpp_shared_6100_bin_5  \
  # --loss_type mtpp_shared