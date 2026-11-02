#!/usr/bin/env bash
# CUMC OMOP on the cumc cluster, mamba backbone.
# Usage: bash mamba.sh tasks                                  (tokenizer + all task pickles)
#        bash mamba.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> [prepare|pretrain|all|train] [extra args]
source "$(dirname "$(readlink -f "$0")")/../lib.sh" cumc_cumc

ARCH=mamba
ARCH_ARGS=(--n_layers 28 --learning_rate 2e-4 --max_grad_norm 5.0 --n_epochs "${N_EPOCHS:-20}")

main "$@"
