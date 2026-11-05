#!/usr/bin/env bash
# CUMC on the cumc cluster: build the evaluation batches before any model is evaluated (CPU, no model
# needed). Guide: src/femr/evaluation/README.md
# Usage: bash batches.sh <outcome|phenotype|regression|all> [task1,task2]
source "$(dirname "$(readlink -f "$0")")/../lib.sh" cumc_cumc

ARCH=transformer
batches_main "$@"
