#!/bin/bash
set -e

DATASET="$1"
TASK="$2"
DEVICE="${3:-cuda:0}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

# ============== CUMC PATHS ==============
if [ "$DATASET" == "CUMC" ]; then
    MEDS_READER="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader"
    MODEL_PATH="/data/models/zj2398/femr/cumc_full/tpp/transformer/checkpoint-547696"
    TOKENIZER_PATH="/data/models/zj2398/femr/cumc_full/tpp/tokenizer"
    ONTOLOGY_PATH="/data/models/zj2398/femr/cumc_full/tpp/ontology.pkl"
    MAIN_SPLIT_PATH="/data/models/zj2398/femr/cumc_full/tpp/main_split.csv"

    if [ "$TASK" == "stroke" ]; then
        COHORT_DIR="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_sample/Ischemic_Stroke"
    elif [ "$TASK" == "scz" ]; then
        COHORT_DIR="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_sample/Schizophrenia"
    elif [ "$TASK" == "long_los" ]; then
        COHORT_DIR="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/patient_outcomes_sample/long_los"
    else
        echo "Unknown CUMC task: $TASK"
        echo "Available: stroke, scz, long_los"
        exit 1
    fi

# ============== MIMIC PATHS ==============
elif [ "$DATASET" == "MIMIC" ]; then
    MEDS_READER="/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader"
    MODEL_PATH="/data/models/zj2398/femr/mimic_motor_8k/output/best_100620"
    TOKENIZER_PATH="/data/models/zj2398/femr/mimic_motor_8k/tokenizer"
    ONTOLOGY_PATH="/data/models/zj2398/femr/mimic_motor_8k/ontology.pkl"
    MAIN_SPLIT_PATH="/data/models/zj2398/femr/mimic_motor_8k/main_split.csv"

    if [ "$TASK" == "stroke" ]; then
        COHORT_DIR="/data2/processed_datasets/mimic/fomoh_subsample_100k/stroke"
    elif [ "$TASK" == "masld" ]; then
        COHORT_DIR="/data2/processed_datasets/mimic/fomoh_subsample_100k/masld"
    elif [ "$TASK" == "readmission" ]; then
        COHORT_DIR="/data2/processed_datasets/mimic/fomoh_subsample_100k/readmission"
    else
        echo "Unknown MIMIC task: $TASK"
        echo "Available: stroke, masld, readmission"
        exit 1
    fi

else
    echo "Usage: $0 <CUMC|MIMIC> <task> [device] [extra_args]"
    echo ""
    echo "CUMC tasks: stroke, scz, long_los"
    echo "MIMIC tasks: stroke, masld, readmission"
    echo ""
    echo "Examples:"
    echo "  $0 CUMC stroke cuda:0"
    echo "  $0 CUMC stroke cuda:0 --debug"
    exit 1
fi

# Map task names to match other models' output directories
if [ "$DATASET" == "CUMC" ]; then
    if [ "$TASK" == "stroke" ]; then
        OUTPUT_DIR="/data/results/foundation_benchmarking/${DATASET}/fine_tune/Ischemic_Stroke"
    elif [ "$TASK" == "scz" ]; then
        OUTPUT_DIR="/data/results/foundation_benchmarking/${DATASET}/fine_tune/Schizophrenia"
    else
        OUTPUT_DIR="/data/results/foundation_benchmarking/${DATASET}/fine_tune/${TASK}"
    fi
else
    OUTPUT_DIR="/data/results/foundation_benchmarking/${DATASET}/fine_tune/${TASK}"
fi

echo "========================================"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo "Device: $DEVICE"
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Cohort: $COHORT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Extra args: $EXTRA_ARGS"
echo "========================================"

# Verify paths exist
if [ ! -d "$COHORT_DIR" ]; then
    echo "ERROR: Cohort directory does not exist: $COHORT_DIR"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer path does not exist: $TOKENIZER_PATH"
    exit 1
fi

python -u -m femr.omop_meds_tutorial.evaluation.finetune_motor_e2e \
    --meds_reader "${MEDS_READER}" \
    --model_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --ontology_path "${ONTOLOGY_PATH}" \
    --main_split_path "${MAIN_SPLIT_PATH}" \
    --cohort_dir "${COHORT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --task_name "${TASK}" \
    --device "${DEVICE}" \
    --num_proc ${NUM_PROC:-16} \
    --tokens_per_batch 65536 \
    --min_subjects_per_batch 8 \
    --num_epochs 10 \
    --weight_decay 0.1 \
    --pretraining_lr 0.00001 \
    $EXTRA_ARGS