# MIMIC-IV on the cumc cluster. Any variable can be overridden from the environment.
# Copy META_DIR from GSB (ontology.pkl, tokenizer/) so both clusters use identical files.
MEDS_READER=${MEDS_READER:-/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader}
META_DIR=${META_DIR:-/data2/processed_datasets/zj2398/femr/mimic/metadata}   # ontology.pkl + shared tokenizer/
OUT_ROOT=${OUT_ROOT:-/data2/processed_datasets/zj2398/femr/mimic/models}     # one directory per model
HF_CACHE=${HF_CACHE:-/data2/huggingface/zj2398}
TIME_BINS=${TIME_BINS:-8}
VALUE_BINS=${VALUE_BINS:-5}
NUM_THREADS=${NUM_THREADS:-32}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}   # GPU ids for pretraining (1, 2 or 4 ids), e.g. GPUS=0,1
export WANDB_PROJECT=${WANDB_PROJECT:-ora_mimic}   # wandb project of the pretraining runs
# Columns that get no tokens (ID columns took most of the numeric-bin budget). Override: BANNED_PROPERTIES="a b"
BANNED_PROPERTIES=(${BANNED_PROPERTIES-hadm_id icustay_id})
