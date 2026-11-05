#!/usr/bin/env bash
# CUMC on the cumc cluster: read-only checks of the labels, tokenizer, packages and machine. Run it
# before batches.sh, and again after baseline.sh extract. Guide: src/femr/evaluation/README.md
# Usage: bash preflight.sh
source "$(dirname "$(readlink -f "$0")")/../lib.sh" cumc_cumc

python -m femr.evaluation.common.preflight \
    --meta_dir "$META_DIR" \
    --models_root "$MODELS_ROOT" \
    --result_root "$RESULT_ROOT" \
    --hf_cache "$HF_CACHE" \
    --meds_tab_dir "$MEDS_TAB_DIR" \
    --outcome_cohort "$OUTCOME_COHORT" \
    --binary_cohort "$PHENOTYPE_BINARY_COHORT" \
    --tte_cohort "$PHENOTYPE_TTE_COHORT" \
    --regression_cohort "$REGRESSION_COHORT" \
    --regression_tasks "$REGRESSION_TASKS" \
    --tte_col "$TTE_COL"
