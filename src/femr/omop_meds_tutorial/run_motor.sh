#!/bin/sh
LOGFILE="$(dirname "$0")/run_motor_output2.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Default values
SCRIPT_NAME=$(basename "$0")
NUM_PROC=10
TOKENS_PER_BATCH=8192
# Use empty value to indicate no observation window specified
OBSERVATION_WINDOW=""



# Function to display help
show_help() {
    echo "Usage: $SCRIPT_NAME COHORT_BASE_DIR [OPTIONS]"
    echo
    echo "Run MOTOR generation and fine-tuning for all cohorts in the specified directory."
    echo
    echo "Arguments:"
    echo "  COHORT_BASE_DIR          Base directory containing prediction task subdirectories"
    echo
    echo "Options:"
    echo "  -h, --help               Display this help message and exit"
    echo "  --pretraining_data       Override PRETRAINING_DATA environment variable, load ontology"
    echo "  --meds_reader            Override OMOP_MEDS_READER environment variable"
    echo "  --num_proc               Number of processors to use (default: 10)"
    echo "  --tokens_per_batch       Tokens per batch (default: 231072)"
    echo "  --observation_window     Observation window in days (optional integer value)"
    echo
    echo "Environment Variables:"
    echo "  PRETRAINING_DATA         Path to pretraining data (required if not set with --pretraining_data)"
    echo "  OMOP_MEDS_READER         Path to OMOP MEDS reader (required if not set with --meds_reader)"
    echo
    echo "Example:"
    echo "  $SCRIPT_NAME /path/to/cohorts"
    echo "  $SCRIPT_NAME /path/to/cohorts --pretraining_data /path/to/pretraining --meds_reader /path/to/reader --num_proc 8"
    echo "  $SCRIPT_NAME /path/to/cohorts --observation_window 30"
}

# Parse command line options
PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""
COHORT_BASE_DIR=""
DEVICE="cuda"

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --pretraining_data)
            PRETRAINING_DATA_ARG="$2"
            shift 2
            ;;
        --meds_reader)
            OMOP_MEDS_READER_ARG="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_proc)
            NUM_PROC="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tokens_per_batch)
            TOKENS_PER_BATCH="$2"
            shift 2
            ;;
        --observation_window)
            OBSERVATION_WINDOW="$2"
            shift 2
            ;;
        --min_subjects_per_batch)
            MIN_SUBJECTS_PER_BATCH="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Try '$SCRIPT_NAME --help' for more information." >&2
            exit 1
            ;;
        *)
            # First non-option argument is the cohort base directory
            if [ -z "$COHORT_BASE_DIR" ]; then
                COHORT_BASE_DIR="$1"
            else
                echo "Error: Unexpected argument: $1" >&2
                echo "Try '$SCRIPT_NAME --help' for more information." >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if cohort base directory was provided
if [ -z "$COHORT_BASE_DIR" ]; then
    echo "Error: Missing required argument COHORT_BASE_DIR" >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

# Use command line arguments if provided, otherwise use environment variables
if [ -n "$PRETRAINING_DATA_ARG" ]; then
    PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
fi

if [ -n "$OMOP_MEDS_READER_ARG" ]; then
    OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"
fi

# Check if the required variables are set
if [ -z "$PRETRAINING_DATA" ] || [ -z "$OMOP_MEDS_READER" ]; then
    echo "Error: PRETRAINING_DATA or OMOP_MEDS_READER are not set." >&2
    echo "Set them as environment variables or use --pretraining_data and --meds_reader options." >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

# Check if the cohort base directory exists
if [ ! -d "$COHORT_BASE_DIR" ]; then
    echo "Error: Cohort base directory does not exist: $COHORT_BASE_DIR" >&2
    exit 1
fi

echo "Using configuration:"
echo "  COHORT_BASE_DIR: $COHORT_BASE_DIR"
echo "  PRETRAINING_DATA: $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER: $OMOP_MEDS_READER"
echo "  NUM_PROC: $NUM_PROC"
echo "  TOKENS_PER_BATCH: $TOKENS_PER_BATCH"
    echo "  OBSERVATION_WINDOW: $([ -z "$OBSERVATION_WINDOW" ] && echo "Not specified" || echo "$OBSERVATION_WINDOW")"
echo

# Iterate over all task directories in the cohort folder
echo "Discovering prediction tasks..."
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

echo "Found $TASK_COUNT prediction tasks."
# echo

# Process tasks
CURRENT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")
    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TASK_COUNT] Processing task: $TASK_NAME"
    echo "cohort_base_dir: $COHORT_BASE_DIR"
    echo "Task directory: $TASK_DIR"

    # Run the first command: generate MOTOR features
    echo "Running MOTOR feature generation for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    GENERATE_CMD="python -u -m femr.omop_meds_tutorial.generate_motor_features \
      --pretraining_data \"$PRETRAINING_DATA\" \
      --model_path \"$MODEL_PATH\" \
      --meds_reader \"$OMOP_MEDS_READER\" \
      --num_proc \"$NUM_PROC\" \
      --tokens_per_batch \"$TOKENS_PER_BATCH\" \
      --device \"$DEVICE\" \
      --min_subjects_per_batch \"$MIN_SUBJECTS_PER_BATCH\" \
      --cohort_dir \"$TASK_DIR\""

    # Add observation_window parameter if specified
    if [ -n "$OBSERVATION_WINDOW" ]; then
        GENERATE_CMD="$GENERATE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    # Print the command
    echo "Executing command: $GENERATE_CMD"

    # Execute the command
    eval $GENERATE_CMD

    # Check if the first command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: MOTOR feature generation failed for task $TASK_NAME"
        continue
    fi

    # Run the second command: fine-tune MOTOR
    echo "Running MOTOR fine-tuning for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    FINETUNE_CMD="python -u -m femr.omop_meds_tutorial.finetune_motor \
      --pretraining_data \"$PRETRAINING_DATA\" \
      --meds_reader \"$OMOP_MEDS_READER\" \
      --cohort_label \"$TASK_NAME\""

    # Add observation_window parameter if specified
    if [ -n "$OBSERVATION_WINDOW" ]; then
        FINETUNE_CMD="$FINETUNE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    # Print the command
    echo "Executing command: $FINETUNE_CMD"

    # Execute the command
    eval $FINETUNE_CMD

    # Check if the second command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: MOTOR fine-tuning failed for task $TASK_NAME"
        continue
    fi

    # Determine the MOTOR prediction folder path based on observation window
    if [ -n "$OBSERVATION_WINDOW" ]; then
        MOTOR_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/motor_$OBSERVATION_WINDOW/test_predictions"
        MOTOR_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/motor_$OBSERVATION_WINDOW/"
    else
        MOTOR_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/motor/test_predictions"
        MOTOR_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/motor/"
    fi

    # Build the evaluation command
    EVAL_CMD="meds-evaluation-cli predictions_path=\"$MOTOR_PREDICTION_FOLDER\" \
      output_dir=\"$MOTOR_OUTPUT_DIR\""

    # Run the third command to compute the metrics
    echo "Running meds-evaluation for $TASK_NAME..."
    echo "Executing command: $EVAL_CMD"

    # Execute the command
    eval $EVAL_CMD

    # Check if the third command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Running meds-evaluation failed for task $TASK_NAME"
    fi

    echo "Completed processing of task: $TASK_NAME"
    echo "----------------------------------------"
done

echo "All tasks processed."

# export CUDA_VISIBLE_DEVICES=2
# bash run_motor.sh \
#   --pretraining_data   /user/zj2398/cache/motor_mimic \
#   --meds_reader        /user/zj2398/cache/mimic/meds_v0.6_reader \
#   --num_proc           100 \
#   --model_path         /user/zj2398/cache/motor_mimic/output/motor_8192 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   /user/zj2398/cache/mimic/mimic-3.1-meds/patient_outcome_tasks/task/



