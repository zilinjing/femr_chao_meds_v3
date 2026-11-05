#!/usr/bin/env bash
# MIMIC-IV on GSB: evaluate a transformer model. Guide: src/femr/evaluation/README.md
# Usage: bash transformer.sh <ntp|motor|tpp|mtpp|icarefm|survivehr> <outcome|phenotype|regression|all> [task1,task2]
#        bash transformer.sh collect
source "$(dirname "$(readlink -f "$0")")/../../lib.sh" mimic_gsb

ARCH=transformer
main "$@"
