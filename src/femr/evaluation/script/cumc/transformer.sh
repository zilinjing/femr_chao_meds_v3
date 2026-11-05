#!/usr/bin/env bash
# CUMC on the cumc cluster: evaluate a transformer model. Guide: src/femr/evaluation/README.md
# Usage: bash transformer.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> <outcome|phenotype|regression|all> [task1,task2]
#        bash transformer.sh collect
source "$(dirname "$(readlink -f "$0")")/../lib.sh" cumc_cumc

ARCH=transformer
main "$@"
