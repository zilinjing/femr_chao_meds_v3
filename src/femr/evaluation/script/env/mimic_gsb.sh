# MIMIC-IV on the GSB cluster. Any variable can be overridden from the environment.
STORAGE=/shared/share_mala/zilin/storage/ora
DATASET=mimic
MEDS_READER=${MEDS_READER:-$STORAGE/meds_v0.6_reader}
META_DIR=${META_DIR:-$STORAGE/metadata}            # ontology.pkl + main_split.csv
MODELS_ROOT=${MODELS_ROOT:-$STORAGE/models}        # pretrained models: <model>/<arch>/best_model
RESULT_ROOT=${RESULT_ROOT:-$STORAGE/result}        # results: <family>/<task>/<model>/
HF_CACHE=${HF_CACHE:-$STORAGE/hf_cache}            # temporary batch files
MEDS_TAB_DIR=${MEDS_TAB_DIR:-$STORAGE/meds_tab}    # baseline (MEDS-Tab) features

# Labels: one sub-folder per task.
OUTCOME_COHORT=${OUTCOME_COHORT:-$STORAGE/cohort/outcome/cohort_100k}   # 100k train rows, full test set
PHENOTYPE_BINARY_COHORT=${PHENOTYPE_BINARY_COHORT:-$STORAGE/cohort/phenotype/binary}
PHENOTYPE_TTE_COHORT=${PHENOTYPE_TTE_COHORT:-$STORAGE/cohort/phenotype/tte}
REGRESSION_COHORT=${REGRESSION_COHORT:-$STORAGE/cohort/regression/icu_stay_4h_100k}
REGRESSION_TASKS=${REGRESSION_TASKS:-}             # empty = every task folder

TRUNCATION=${TRUNCATION:-300}                      # C-index horizon (days) of the phenotype TTE tasks
NUM_PROC=${NUM_PROC:-24}                           # CPU workers
