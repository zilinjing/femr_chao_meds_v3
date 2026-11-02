# MIMIC-IV on the GSB cluster. Any variable can be overridden from the environment.
MEDS_READER=${MEDS_READER:-/shared/share_mala/zilin/storage/ora/meds_v0.6_reader}
META_DIR=${META_DIR:-/shared/share_mala/zilin/storage/ora/metadata}   # ontology.pkl + shared tokenizer/
OUT_ROOT=${OUT_ROOT:-/shared/share_mala/zilin/storage/ora/models}     # one directory per model
HF_CACHE=${HF_CACHE:-/shared/share_mala/zilin/storage/ora/hf_cache}
TIME_BINS=${TIME_BINS:-8}
VALUE_BINS=${VALUE_BINS:-5}
NUM_THREADS=${NUM_THREADS:-32}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}   # GPU ids for pretraining (1, 2 or 4 ids), e.g. GPUS=0,1
export WANDB_PROJECT=${WANDB_PROJECT:-ora_mimic}   # wandb project of the pretraining runs
# Columns that get no tokens (ID columns took most of the numeric-bin budget). Override: BANNED_PROPERTIES="a b"
BANNED_PROPERTIES=(${BANNED_PROPERTIES-hadm_id icustay_id})
