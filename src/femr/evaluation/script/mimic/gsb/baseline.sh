#!/usr/bin/env bash
# MIMIC-IV on GSB: baselines (XGBoost, DeepHit, most recent value). Guide: src/femr/evaluation/README.md
# Usage: bash baseline.sh <extract|outcome|phenotype|regression> [task1,task2]
source "$(dirname "$(readlink -f "$0")")/../../lib.sh" mimic_gsb

baseline_main "$@"
