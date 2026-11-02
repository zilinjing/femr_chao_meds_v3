#!/usr/bin/env bash
# MIMIC-IV on GSB, mamba backbone.
# Usage: bash mamba.sh tasks                                  (tokenizer + all task pickles)
#        bash mamba.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> [prepare|pretrain|all|train] [extra args]
source "$(dirname "$(readlink -f "$0")")/../../lib.sh" mimic_gsb

ARCH=mamba
ARCH_ARGS=(--n_layers 28 --learning_rate 2e-4 --max_grad_norm 5.0 --n_epochs "${N_EPOCHS:-20}")

main "$@"
