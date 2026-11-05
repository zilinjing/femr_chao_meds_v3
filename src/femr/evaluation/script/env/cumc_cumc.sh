# CUMC OMOP on the cumc cluster. Any variable can be overridden from the environment.
CUMC_MEDS=/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped
DATASET=cumc
MEDS_READER=${MEDS_READER:-$CUMC_MEDS/post_transform_meds_reader}
META_DIR=${META_DIR:-/data/models/zj2398/femr/metadata}          # ontology.pkl + main_split.csv
MODELS_ROOT=${MODELS_ROOT:-/data/models/zj2398/femr/models}      # pretrained models: <model>/<arch>/best_model
RESULT_ROOT=${RESULT_ROOT:-/data/models/zj2398/femr/results}     # results: <family>/<task>/<model>/
HF_CACHE=${HF_CACHE:-/data2/huggingface/zj2398}                  # temporary batch files
MEDS_TAB_DIR=${MEDS_TAB_DIR:-/data/models/zj2398/femr/meds_tab}  # baseline (MEDS-Tab) features

# Labels: one sub-folder per task.
OUTCOME_COHORT=${OUTCOME_COHORT:-$CUMC_MEDS/task_labels/patient_outcomes_clean}
PHENOTYPE_BINARY_COHORT=${PHENOTYPE_BINARY_COHORT:-$CUMC_MEDS/task_labels/phenotype_tte/binary}
PHENOTYPE_TTE_COHORT=${PHENOTYPE_TTE_COHORT:-$CUMC_MEDS/task_labels/phenotype_tte/tte}
REGRESSION_COHORT=${REGRESSION_COHORT:-/data2/processed_datasets/zj2398/cumc_data/regression_100k}
REGRESSION_TASKS=${REGRESSION_TASKS:-platelets,creatinine,spo2,glucose}   # bilirubin is not used

TRUNCATION=${TRUNCATION:-365}                      # C-index horizon (days) of the phenotype TTE tasks
NUM_PROC=${NUM_PROC:-24}                           # CPU workers
BATCH_WORKERS=${BATCH_WORKERS:-28}                 # batches.sh: workers per task family

# Baselines. DeepHit searches 72 configs: lr x batch x dropout (2 values each) x hidden width
# {0, 64, 128} (the "full" grid) x these 3 weight decays.
DEEPHIT_GRID=${DEEPHIT_GRID:-full}
DEEPHIT_WD=${DEEPHIT_WD:-0,1e-4,1e-2}
TTE_COL=${TTE_COL:-time_to_event}                  # time column of the TTE labels (MIMIC: time_to_event_days)
