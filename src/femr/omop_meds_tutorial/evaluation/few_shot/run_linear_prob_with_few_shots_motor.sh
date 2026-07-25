#!/bin/bash

# Display help information
show_help() {
  echo "Usage: src/ehr_foundation_model_benchmark/tools/linear_prob/run_linear_prob_with_few_shots.sh [options]"
  echo
  echo "This script runs the EHR foundation model benchmark across multiple cohorts."
  echo
  echo "Options:"
  echo "  --base_dir DIR       Base directory containing cohort folders"
  echo "  --output_dir DIR     Output directory for results"
  echo "  --meds_dir DIR       Medical Event Data Standard data directory"
  echo "  --model_name NAME    Model name to use"
  echo "  -h, --help           Display this help message and exit"
  echo
  echo "All options are required."
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
  show_help
  exit 1
fi

# Process command line arguments
while [ $# -gt 0 ]; do
  case $1 in
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --meds_dir)
      MEDS_DIR="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --task_list)
      TASK_LIST="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$BASE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MEDS_DIR" ] || [ -z "$MODEL_NAME" ]; then
  echo "Error: Missing required arguments."
  show_help
  exit 1
fi

# Verify directories exist
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base directory $BASE_DIR does not exist."
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Error: Output directory $OUTPUT_DIR does not exist."
  exit 1
fi

if [ ! -d "$MEDS_DIR" ]; then
  echo "Error: Medical Event Data Standard directory $MEDS_DIR does not exist."
  exit 1
fi

# Iterate through each cohort folder
# for cohort_dir in "$BASE_DIR"*/; do

for cohort_dir in "$BASE_DIR"/*/; do
  # Extract task name from folder name
  task_name=$(basename "$cohort_dir")

  # Skip if not a directory
  if [ ! -d "$cohort_dir" ]; then
    continue
  fi

  # Or if TASK_LIST is comma-separated like "task1,task2,task3"
  if [ -n "$TASK_LIST" ] && ! echo "$TASK_LIST" | grep -q "\<$task_name\>"; then
    echo "Skip $task_name, not in the list"
    continue
  fi

  echo "Processing cohort: $task_name"

  # if [task_name.lower() in ["ami", "celiac", "cll", "htn", "ischemic_stroke","stroke","masld", "osteoporosis", "pancreatic_Cancer", "sle", "t2dm"]]; then
  #   $MODEL_NAME_FEATURE = $MODEL_NAME"_{birth_730}"
  # else:
  #   $MODEL_NAME_FEATURE = $MODEL_NAME"_{birth}"
  # fi

  task_name_lower=$(echo "$task_name" | tr '[:upper:]' '[:lower:]')

  if [[ "$task_name_lower" =~ ^(ami|celiac|cll|htn|ischemic_stroke|stroke|masld|osteoporosis|pancreatic_cancer|sle|t2dm)$ ]]; then
    MODEL_NAME_FEATURE="${MODEL_NAME}_birth_730"
  else
    MODEL_NAME_FEATURE="${MODEL_NAME}_birth"
  fi

  echo "Processing model feature: $MODEL_NAME_FEATURE"

  # cohort_dir="$cohort_dir/$MODEL_NAME_FEATURE"
  cohort_model_dir="${cohort_dir%/}/$MODEL_NAME_FEATURE"
  # Find features_with_label directory
  features_dir=$(find "$cohort_dir" -type d -name "features_with_label" | head -n 1)

  if [ -z "$features_dir" ]; then
    echo "Warning: No features_with_label directory found for $task_name. Skipping..."
    continue
  fi

  echo "Found features directory: $features_dir"

# for pkl_file in "$BASE_DIR"*.pkl; do
#   # Skip if the glob didn't match anything
#   [ -e "$pkl_file" ] || continue

#   task_name=$(basename "$pkl_file" .pkl)

#   echo "Processing file: $task_name"
#   echo "Full path: $pkl_file"

#   # >>> your processing code here <<<

  # Run the fine-tuning command
  echo "Running benchmark for $task_name..."
  python few_shot_motor.py \
    --features_label_input_dir "$features_dir" \
    --model_name "$MODEL_NAME" \
    --task_name "$task_name" \
    --output_dir "$OUTPUT_DIR" \
    --meds_dir "$MEDS_DIR"

  # Check if finetune command succeeded
  if [ $? -ne 0 ]; then
    echo "Warning: Benchmark failed for $task_name"
    echo "---------------------------------"
    continue
  fi

  # Find all model directories created by the finetune process
  for model_dir in "$OUTPUT_DIR/$task_name/$MODEL_NAME"_*/; do
    # Skip if not a directory
    if [ ! -d "$model_dir" ]; then
      continue
    fi

    model_folder=$(basename "$model_dir")
    echo "Found model folder: $model_folder"

    # Check if results.json already exists
    if [ -f "$model_dir/results.json" ]; then
      echo "Skipping evaluation for $task_name/$model_folder - results.json already exists"
    # Check if test_predictions directory exists
    elif [ -d "$model_dir/test_predictions" ]; then
      echo "Running meds-evaluation-cli for $task_name/$model_folder"

      # Run the meds-evaluation-cli command
      meds-evaluation-cli predictions_path="$model_dir/test_predictions" output_dir="$model_dir"

      # Check if command succeeded
      if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully"
      else
        echo "Warning: Evaluation failed for $task_name/$model_folder"
      fi
    else
      echo "Warning: No test_predictions directory found in $model_dir. Skipping..."
    fi
  done

  echo "Completed $task_name"
  echo "---------------------------------"
done

echo "All tasks completed!"

# '''
# bash run_linear_prob_with_few_shots_motor.sh \
# --base_dir /shared/share_mala/zj2398/mimic/task/patient_outcome_task/results/ \
# --output_dir /shared/share_mala/zj2398/mimic/task/patient_outcome_task/few_shot_results/ \
# --meds_dir /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
# --task_list "in_hospital_mortality,long_los,readmission" \
# --model_name motor

# bash run_linear_prob_with_few_shots_motor.sh \
# --base_dir /shared/share_mala/zj2398/mimic/task/phenotype_task/results/ \
# --output_dir /shared/share_mala/zj2398/mimic/task/phenotype_task/few_shot_results/ \
# --meds_dir /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
# --task_list "ami,masld,stroke" \
# --model_name motor_730

# bash run_linear_prob_with_few_shots_motor.sh \
# --base_dir /data/results/zj2398/fomoh/cumc/linear_probing/motor/results \
# --output_dir /data/results/foundation_benchmarking/CUMC/log_prob \
# --meds_dir /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
# --model_name motor

# bash run_linear_prob_with_few_shots_motor.sh \
# --base_dir /data/results/zj2398/fomoh/mimic/linear_probing/results \
# --output_dir //data/results/foundation_benchmarking/MIMIC/log_prob \
# --meds_dir /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
# --model_name motor



# --task_list "in_hospital_mortality,long_los,readmission" \

# python few_shot_motor.py \
#   --features_label_input_dir /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor_large_no_visit_id/results/Schizophrenia/motor/features_with_label \
#   --model_name motor \
#   --task_name Schizophrenia \
#   --output_dir /data/results/zj2398/fomoh/cumc/linear_probing/results \
#   --meds_dir /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform 
# '''