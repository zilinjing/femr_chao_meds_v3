bash run.sh \
  --meds_reader        /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           32 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        outcome \
  --cache_path         /data2/huggingface/zj2398 \
  --min_subjects_per_batch 8 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --output_dir   /data/results/zj2398/femr/cumc/outcome/  \
  --classification_cohort /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/patient_outcomes_clean/ \
  death

bash regression/run_regression.sh \
  --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
  --output_dir   /data/results/zj2398/femr/cumc/regression \
  --use_ntp yes \
  --cache_path  /data2/huggingface/zj2398 \
  --tasks "platelets,creatinine,spo2,glucose" \
  --regression 

bash run.sh \
  --meds_reader        /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        phenotype \
  --cache_path        /data2/huggingface/zj2398/ \
  --min_subjects_per_batch 8 \
  --truncation  365 \
  --observation_window 730 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --output_dir   /data/results/zj2398/femr/cumc/phenotype  \
  --classification_cohort  /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte/binary/ \
  --tte_cohort  /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_tte/tte/ 