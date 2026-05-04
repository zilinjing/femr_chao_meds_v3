set -e

# LOGFILE="$(dirname "$0")/run_motor_output.log"
# exec > >(tee -a "$LOGFILE") 2>&1

SCRIPT_NAME=$(basename "$0")
NUM_PROC=10
TOKENS_PER_BATCH=8192
OBSERVATION_WINDOW=""
USE_LINEAR_INTERPOLATION=false
REGRESSION=false
DEVICE="cuda"

show_help() {
    echo "Usage: $SCRIPT_NAME COHORT_BASE_DIR [OPTIONS]"
    echo
    echo "Options:"
    echo "  --meds_reader            Override OMOP_MEDS_READER env var"
    echo "  --model_path             Path to MOTOR encoder checkpoint"
    echo "  --num_proc               Number of processors (default: 10)"
    echo "  --device                 Device string (default: cuda)"
    echo "  --tokens_per_batch       Tokens per batch (default: 8192)"
    echo "  --min_subjects_per_batch Minimum subjects per batch"
    echo "  --ontology_path          Path to ontology.pkl"
    echo "  --main_split_path        Path to CSV with subject_id,split"
    echo "  --observation_window     Observation window in days"
    echo "  --linear_interpolation   Enable linear interpolation"
    echo "  --regression             Use regression evaluator (RMSE/MAE/R2)"
    echo "  --tasks                  Comma/space-separated list of task names to run"
    echo "                           e.g. --tasks \"Readmission,Mortality\" or --tasks \"(Readmission, Mortality)\""
}

# PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""
COHORT_BASE_DIR=""
MIN_SUBJECTS_PER_BATCH="1"

# NEW: storage for task filters
FILTER_TASKS=()

# Helper: push parsed names into FILTER_TASKS

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        # --pretraining_data) PRETRAINING_DATA_ARG="$2"; shift 2 ;;
        --meds_reader) OMOP_MEDS_READER_ARG="$2"; shift 2 ;;
        --model_type) MEDS_TYPE="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --num_proc) NUM_PROC="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --tokens_per_batch) TOKENS_PER_BATCH="$2"; shift 2 ;;
        --observation_window) OBSERVATION_WINDOW="$2"; shift 2 ;;
        --min_subjects_per_batch) MIN_SUBJECTS_PER_BATCH="$2"; shift 2 ;;
        --ontology_path) ONTOLOGY_PATH="$2"; shift 2 ;;
        --cache_path) CACHE_PATH="$2"; shift 2 ;;
        --main_split_path) MAIN_SPLIT_PATH="$2"; shift 2 ;;
        --use_ntp) USE_NTP="$2"; shift 2 ;;
        --cohort_dir) COHORT_BASE_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks) TASK_LIST="$2"; shift 2 ;;   # <-- NEW
        --regression) REGRESSION=true; shift ;;
        -*) echo "Error: Unknown option: $1" >&2; exit 1 ;;
        *)
            # ignored positional tail in your current script
            shift
            ;;
    esac
done

[ -z "$COHORT_BASE_DIR" ] && { echo "Error: Missing required argument COHORT_BASE_DIR" >&2; exit 1; }

# [ -n "$PRETRAINING_DATA_ARG" ] && PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
[ -n "$OMOP_MEDS_READER_ARG" ] && OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"

if [ -z "$OMOP_MEDS_READER" ]; then
    echo "Error: OMOP_MEDS_READER are not set." >&2
    exit 1
fi

[ ! -d "$COHORT_BASE_DIR" ] && { echo "Error: Cohort base directory does not exist: $COHORT_BASE_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# RUN_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

GEN_FEATURES_PY="$REPO_ROOT/femr/omop_meds_tutorial/evaluation/generate_features.py"
FINETUNE_PY="$REPO_ROOT/femr/omop_meds_tutorial/evaluation/regression/finetune_regression.py"

[ -f "$GEN_FEATURES_PY" ] || { echo "Missing: $GEN_FEATURES_PY"; exit 1; }
[ -f "$FINETUNE_PY" ] || { echo "Missing: $FINETUNE_PY"; exit 1; }

echo "Using configuration:"
echo "  COHORT_BASE_DIR:          $COHORT_BASE_DIR"
# echo "  PRETRAINING_DATA:         $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER:         $OMOP_MEDS_READER"
echo "  NUM_PROC:                 $NUM_PROC"
echo "  TOKENS_PER_BATCH:         $TOKENS_PER_BATCH"
echo "  OBSERVATION_WINDOW:       $([ -z "$OBSERVATION_WINDOW" ] && echo "Not specified" || echo "$OBSERVATION_WINDOW")"
echo "  REGRESSION:               $REGRESSION"
echo "  PYTHONPATH head:          $REPO_ROOT"
# echo "  RUN_ROOT (outputs):       $RUN_ROOT"
if [ ${#FILTER_TASKS[@]} -gt 0 ]; then
  echo "  TASK FILTER:              ${FILTER_TASKS[*]}"
fi
echo

# Helper: membership test
task_selected() {
    local t="$1"
    # if no filter provided, everything is selected
    [ ${#FILTER_TASKS[@]} -eq 0 ] && return 0
    for x in "${FILTER_TASKS[@]}"; do
        if [ "$x" = "$t" ]; then
            return 0
        fi
    done
    return 1
}

TASK_COUNT=0

for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")
    TASK_COUNT=$((TASK_COUNT + 1))

    echo "[$TASK_COUNT] Found task: $TASK_NAME"
done
if [ "$TASK_COUNT" -eq 0 ]; then
    echo "No prediction tasks found in $COHORT_BASE_DIR"
    exit 0
fi

CURRENT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")

    # Check if task list is specified and if current task is in the list
    if [ -n "$TASK_LIST" ]; then
        # Convert comma-separated list to array and check if task is in it
        if ! echo ",$TASK_LIST," | grep -q ",$TASK_NAME,"; then
            SKIPPED=$((SKIPPED + 1))
            echo "[$SKIPPED skipped] Skipping task: $TASK_NAME (not in specified task list)"
            continue
        fi
    fi

    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TASK_COUNT] Processing task: $TASK_NAME"
    echo "cohort_base_dir: $COHORT_BASE_DIR"
    echo "Task directory: $TASK_DIR"
    motor_features_name="${TASK_NAME}_${MODEL_NAME}"

    FEATURES_DIR="${OUTPUT_DIR}/${MODEL_NAME}/features"
    FEATURE_FILE="${FEATURES_DIR}/${motor_features_name}.pkl"

    if [[ -f "$FEATURE_FILE" ]]; then
        echo "Found existing features, reusing: $FEATURE_FILE"
    else
        GENERATE_CMD="python -u \"$GEN_FEATURES_PY\" \
            --model_path \"$MODEL_PATH\" \
            --model_name \"$MODEL_NAME\" \
            --meds_reader \"$OMOP_MEDS_READER\" \
            --num_proc \"$NUM_PROC\" \
            --tokens_per_batch \"$TOKENS_PER_BATCH\" \
            --device \"$DEVICE\" \
            --min_subjects_per_batch \"$MIN_SUBJECTS_PER_BATCH\" \
            --cohort_dir \"$TASK_DIR\" \
            --ontology_path \"$ONTOLOGY_PATH\" \
            --output_root \"$OUTPUT_DIR\" \
            --use_ntp \"$USE_NTP\" \
            --cache_path \"$CACHE_PATH\" \
            --loss_type "labeled_subjects" \
            --task_type "regression" " 
        [ -n "$OBSERVATION_WINDOW" ] && GENERATE_CMD="$GENERATE_CMD --observation_window \"$OBSERVATION_WINDOW\""

        echo "Executing: $GENERATE_CMD"
        eval $GENERATE_CMD || { echo "Error: feature generation failed for $TASK_NAME"; echo "----------------------------------------"; continue; }
    fi
    FINETUNE_CMD="python -u \"$FINETUNE_PY\" \
        --cohort_label \"$TASK_NAME\" \
        --model_name \"$MODEL_NAME\" \
        --main_split_path \"$MAIN_SPLIT_PATH\" \
        --meds_reader \"$OMOP_MEDS_READER\" \
        --model_path \"$MODEL_PATH\" \
        --output_root \"$OUTPUT_DIR\""
    [ -n "$OBSERVATION_WINDOW" ] && FINETUNE_CMD="$FINETUNE_CMD --observation_window \"$OBSERVATION_WINDOW\""

    echo "Executing: $FINETUNE_CMD"
    eval $FINETUNE_CMD || { echo "Error: finetune failed for $TASK_NAME"; echo "----------------------------------------"; continue; }

    echo "Completed processing of task: $TASK_NAME"
    echo "----------------------------------------"
done

echo "All selected tasks processed."

#export CUDA_VISIBLE_DEVICES=6
# glucose,creatinine,pao2,platelets

# bash run_regression.sh \
#   --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --num_proc           24 \
#   --model_path         /user/zj2398/cache/survivehr/mamba_2e-4/checkpoint-29220 \
#   --model_name         survivehr_mamba \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --use_ntp yes \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /user/zj2398/cache/survivehr/ontology.pkl \
#   --main_split_path     /user/zj2398/cache/survivehr/main_split.csv  \
#   --cohort_dir   /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
#   --output_dir   /shared/share_mala/zj2398/mimic/task/regression/ \
#   --cache_path /shared/share_mala/zj2398/huggingface/datasets \
#   --tasks "glucose,creatinine,pao2,platelets" \
#   --regression 


# glucose,creatinine
# pao2,platelets


# bash run_regression.sh \
#   --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --num_proc           24 \
#   --model_path         /user/zj2398/cache/mtpp_8k_T_16_V_5/transformer/best_112686 \
#   --model_name         transformer_mtpp_T_16_V_5 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /user/zj2398/cache/mtpp_8k_T_8_V_10/ontology.pkl \
#   --main_split_path     /user/zj2398/cache/mtpp_8k_T_8_V_10/main_split.csv \
#   --cohort_dir   /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
#   --output_dir   /shared/share_mala/zj2398/mimic/task/regression \
#   --use_ntp no \
#   --cache_path /shared/share_mala/zj2398/huggingface/datasets \
#   --tasks "pao2,platelets,creatinine" \
#   --regression 



##kuvira
# mimic

# bash run_regression.sh \
#   --meds_reader        /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#   --num_proc           32 \
#   --model_path         /data2/processed_datasets/zj2398/femr/mimic/mtpp_T_16_V_10/transformer/best-147565 \
#   --model_name         transformer_mtpp_T_16_V_10 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /data2/processed_datasets/zj2398/femr/mimic/ntp/ontology.pkl \
#   --main_split_path     /data2/processed_datasets/zj2398/femr/mimic/ntp/main_split.csv \
#   --cohort_dir   /data2/processed_datasets/mimic/regression/icu_stay_4h_100k/ \
#   --output_dir   /data2/processed_datasets/zj2398/femr/mimic/results/regression \
#   --tasks "pao2,platelets,creatinine" \
#   --regression 

# kuvira cumc
# bash run_regression.sh \
#   --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
#   --num_proc           24 \
#   --model_path         /data2/processed_datasets/zj2398/femr/cumc/icarefm/transformer/best_922142 \
#   --model_name         transformer_icarefm \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
#   --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
#   --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
#   --output_dir   /data/results/zj2398/femr/cumc/regression \
#   --use_ntp yes \
#   --cache_path  /data2/huggingface/zj2398 \
#   --tasks "platelets,creatinine,spo2,glucose" \
#   --regression 


# bash run_regression.sh \
#   --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
#   --num_proc           24 \
#   --model_path         /data2/processed_datasets/zj2398/femr/cumc/survivehr/transformer/best_256914 \
#   --model_name         transformer_survivehr \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path      /data2/processed_datasets/zj2398/femr/ontology.pkl \
#   --main_split_path    /data2/processed_datasets/zj2398/femr/cumc/motor/main_split.csv \
#   --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
#   --output_dir   /data/results/zj2398/femr/cumc/regression \
#   --use_ntp no \
#   --cache_path  /data2/huggingface/zj2398 \
#   --tasks "platelets,bilirubin,creatinine,spo2" \
#   --regression 





