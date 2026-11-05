#!/usr/bin/env bash
# CUMC on the cumc cluster: baselines (XGBoost, DeepHit, most recent value). Guide: src/femr/evaluation/README.md
# Usage: bash baseline.sh <extract|outcome|phenotype|regression> [task1,task2]
source "$(dirname "$(readlink -f "$0")")/../lib.sh" cumc_cumc

baseline_main "$@"
