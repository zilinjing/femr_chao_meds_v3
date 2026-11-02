#!/usr/bin/env bash
set -euo pipefail

# prepare.py / pretrain.py below are invoked by bare name, so pin the cwd to this script's
# directory -- otherwise whichever copy sits in the caller's cwd runs instead.
cd "$(dirname "$(readlink -f "$0")")/../.."  # prepare.py/pretrain.py live two levels up

# SurviveHR pretraining on MIMIC-IV, transformer backbone.
#
# Identical to run_icarefm.sh except for the three variables marked "differs" below and the
# one objective-specific flag noted at the pretrain step, so the two models differ only in
# their pretraining objective.
#
# --num_final_tasks does NOT apply here. prepare.py forwards it only to the mtpp/mtpp_shared/
# mtpp_value tasks; SurvivEHRTask and ICareFMTask take --value_count_threshold instead, which
# is what produced the 1081 value-normalised codes in both task pickles.
#
# --n_layers is intentionally omitted: the Hydra default (conf/model/transformer.yaml, 11
# layers / 768 hidden / 12 heads / attention_width 496) is what both existing transformer
# checkpoints were trained with.
#
# Re-running prepare.py reuses tokenizer/ and survivehr_task.pkl when they already exist and
# only rebuilds train_batches/val_batches. Delete them first for a genuinely fresh fit.

# MODEL_DIR=/user/zj2398/cache/survivehr             # differs
# LOSS_TYPE=survivehr                                # differs
# RUN_NAME=mimic_survivehr_transformer_value_1000    # differs

# MEDS_READER=/shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
# HF_CACHE=/shared/share_mala/zj2398/huggingface/datasets

MODEL_DIR=/data2/processed_datasets/zj2398/femr/mimic/survivehr         # differs
LOSS_TYPE=survivehr                                # differs
RUN_NAME=mimic_survivehr_transformer_value_1000_test    # differs

MEDS_READER=/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader
HF_CACHE=/data2/huggingface/zj2398

python prepare.py \
  --pretraining_data "$MODEL_DIR" \
  --ontology_path "$MODEL_DIR/ontology.pkl" \
  --meds_reader "$MEDS_READER" \
  --num_threads 64 \
  --tokens_per_batch 16384 \
  --value_count_threshold 1000 \
  --model transformer \
  --loss_type "$LOSS_TYPE" \
  --hf_cache_path "$HF_CACHE"

# --survivehr_time_scale_days is the one flag with no ICareFM counterpart. It is passed
# explicitly because survivehr_task.pkl was pickled with the old 365.25 (years) default and
# pretrain.py overrides that attribute from the CLI; 1.0 keeps the ODE head in days.
# accelerate launch \
#   --num_processes 3 \
#   pretrain.py \
#   --pretraining_data "$MODEL_DIR" \
#   --meds_reader "$MEDS_READER" \
#   --output_dir "$MODEL_DIR/transformer" \
#   --model transformer \
#   --loss_type "$LOSS_TYPE" \
#   --learning_rate 1e-5 \
#   --max_grad_norm 1.0 \
#   --n_epochs 20 \
#   --survivehr_time_scale_days 1.0 \
#   --hf_run_name "$RUN_NAME"

# --- Pretraining FLOPs / time -------------------------------------------------
# One short run measures both: steps 0-499 under the profiler give FLOPs/token, steps
# 500-999 with the profiler off give seconds/token. Both rates are extrapolated to one
# epoch in <output_dir>/training_flops_time.json; multiply by the epochs the real run
# actually reached (early stopping, not --n_epochs).
# Keep every other flag identical to the real run above, or the numbers won't describe it.
# --n_epochs is ignored here because --max_steps takes precedence.
#
# CUDA_VISIBLE_DEVICES is required, not optional. Plain `python pretrain.py` on a multi-GPU
# box makes HF Trainer wrap the model in nn.DataParallel across every visible GPU while
# world_size stays 1, so the measured rates would not describe a single GPU.
export CUDA_VISIBLE_DEVICES=0

python pretrain.py \
  --pretraining_data "$MODEL_DIR" \
  --meds_reader "$MEDS_READER" \
  --output_dir "$MODEL_DIR/transformer" \
  --model transformer \
  --loss_type survivehr \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --n_epochs 20 \
  --survivehr_time_scale_days 1.0 \
  --hf_run_name mimic_survivehr_transformer_profile \
  --profile_steps 1000 \
  --time_steps 1000 \
  --max_steps 2000

# --- Mamba variant ------------------------------------------------------------
# Recipe behind /user/zj2398/cache/survivehr/mamba_2e-4. Different learning rate and
# grad-norm clip, so not a like-for-like counterpart of the transformer run above.
#
# python prepare.py \
#   --pretraining_data /user/zj2398/cache/survivehr \
#   --ontology_path /user/zj2398/cache/survivehr/ontology.pkl \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --num_threads 32 \
#   --tokens_per_batch 16384 \
#   --value_count_threshold 1000 \
#   --model mamba \
#   --loss_type survivehr \
#   --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets
#
# accelerate launch \
#   --num_processes 3 \
#   pretrain.py \
#   --pretraining_data /user/zj2398/cache/survivehr \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --output_dir /user/zj2398/cache/survivehr/mamba_2e-4 \
#   --model mamba \
#   --loss_type survivehr \
#   --learning_rate 2e-4 \
#   --max_grad_norm 5.0 \
#   --n_epochs 20 \
#   --n_layers 28 \
#   --survivehr_time_scale_days 1.0 \
#   --hf_run_name mimic_survivehr_mamba_2e-4_5.0_value_1000
