# CUMC OMOP on the cumc cluster. Any variable can be overridden from the environment.
MEDS_READER=${MEDS_READER:-/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader}
META_DIR=${META_DIR:-/data/models/zj2398/femr/metadata}   # ontology.pkl + shared tokenizer/
OUT_ROOT=${OUT_ROOT:-/data/models/zj2398/femr/models}     # one directory per model
HF_CACHE=${HF_CACHE:-/data2/huggingface/zj2398}
TIME_BINS=${TIME_BINS:-8}
VALUE_BINS=${VALUE_BINS:-5}
NUM_THREADS=${NUM_THREADS:-24}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}   # GPU ids for pretraining (1, 2 or 4 ids), e.g. GPUS=0,1
export WANDB_PROJECT=${WANDB_PROJECT:-ora_cumc}   # wandb project of the pretraining runs
# Columns that get no tokens. None for CUMC: its only ID column, visit_id, is always skipped by the tokenizer.
BANNED_PROPERTIES=(${BANNED_PROPERTIES-})
