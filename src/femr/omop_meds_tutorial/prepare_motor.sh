# Define output file with timestamp
# OUTPUT_FILE="motor_execution_$(date +%Y%m%d_%H%M%S).txt"

# Print start message
# echo "Starting MOTOR preparation script at $(date)" | tee "$OUTPUT_FILE"
# echo "Output will be logged to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
# echo "=================================" | tee -a "$OUTPUT_FILE"

# Execute the Python script with all arguments and capture output in real time
python prepare_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
  --athena_path " " \
  --num_bins 8 \
  --num_threads 1 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader 
#   2>&1 | tee -a "$OUTPUT_FILE"
