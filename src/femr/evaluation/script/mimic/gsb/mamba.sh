#!/usr/bin/env bash
# MIMIC-IV on GSB: evaluate a mamba model. Guide: src/femr/evaluation/README.md
# Usage: bash mamba.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> <outcome|phenotype|regression|all> [task1,task2]
#        bash mamba.sh collect
source "$(dirname "$(readlink -f "$0")")/../../lib.sh" mimic_gsb

ARCH=mamba
main "$@"
