#!/usr/bin/env bash
set -euo pipefail

python prepare.py \
  --pretraining_data /user/zj2398/cache/icarefm \
  --ontology_path /user/zj2398/cache/icarefm/ontology.pkl \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_threads 24 \
  --tokens_per_batch 16384 \
  --value_count_threshold 1000 \
  --model mamba \
  --loss_type icarefm \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets

accelerate launch \
    --num_processes 3 \
    python pretrain.py \
    --pretraining_data /user/zj2398/cache/icarefm \
    --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --output_dir /user/zj2398/cache/icarefm/mamba \
    --model mamba \
    --loss_type icarefm \
    --learning_rate 2e-5 \
    --n_epochs 20 \
    --hf_run_name mimic_icarefm_mamba_value_1000