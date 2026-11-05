
# kuvira cumc
bash run_regression.sh \
  --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/tpp/mamba/best_832692 \
  --model_name         mamba_tpp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
  --output_dir   /data/results/zj2398/femr/cumc/regression \
  --use_ntp no \
  --cache_path  /data2/huggingface/zj2398 \
  --tasks "glucose" \
  --regression 


bash run_regression.sh \
  --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/ntp/transformer/checkpoint-best \
  --model_name         transformer_ntp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
  --output_dir   /data/results/zj2398/femr/cumc/regression \
  --use_ntp yes \
  --cache_path  /data2/huggingface/zj2398 \
  --tasks "glucose" \
  --regression 


bash run_regression.sh \
  --meds_reader         /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /data2/processed_datasets/zj2398/femr/cumc/mtpp/mamba/best_902083 \
  --model_name         mamba_mtpp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path       /data2/processed_datasets/zj2398/femr/cumc/survivehr/ontology.pkl \
  --main_split_path     /data2/processed_datasets/zj2398/femr/cumc/survivehr/main_split.csv \
  --cohort_dir   /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
  --output_dir   /data/results/zj2398/femr/cumc/regression \
  --use_ntp no \
  --cache_path  /data2/huggingface/zj2398 \
  --tasks "glucose" \
  --regression 



