# Define output file with timestamp
# OUTPUT_FILE="motor_execution_$(date +%Y%m%d_%H%M%S).txt"
LOGFILE="$(dirname "$0")/prepare_motor_output.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Print start message
# echo "Starting MOTOR preparation script at $(date)" | tee "$OUTPUT_FILE"
# echo "Output will be logged to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
# echo "=================================" | tee -a "$OUTPUT_FILE"

echo "Starting MOTOR preparation script at $(date)"
echo "Output will be logged to: $LOGFILE"
echo "================================="

python prepare_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/cumc/motor_cumc_bin_8 \
  --athena_path " " \
  --num_bins 8 \
  --num_threads 32 \
  --meds_reader /data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader

# Execute the Python script with all arguments and capture output in real time
# python prepare_motor.py \
#   --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
#   --athena_path " " \
#   --num_bins 8 \
#   --num_threads 1 \
#   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader 
#   2>&1 | tee -a "$OUTPUT_FILE"
