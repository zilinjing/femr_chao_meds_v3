#!/usr/bin/env bash
# MIMIC-IV on the kuvira cluster, transformer backbone.
# Usage: bash transformer.sh tasks                                  (tokenizer + all task pickles)
#        bash transformer.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> [prepare|pretrain|all|train] [extra args]
source "$(dirname "$(readlink -f "$0")")/../../lib.sh" mimic_cumc

ARCH=transformer
ARCH_ARGS=(--n_layers 11 --learning_rate 1e-5 --max_grad_norm 1.0 --n_epochs "${N_EPOCHS:-20}")

main "$@"
