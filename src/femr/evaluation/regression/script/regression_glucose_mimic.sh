# mamba_mtpp_2e-4,mamba_tpp_2e-4,transformer_ntp, transformer_tpp,

# kuvira cumc
bash run_regression.sh \
  --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_proc           24 \
  --model_path         /shared/share_mala/zj2398/models/tpp_8k/transformer/best_221573 \
  --model_name         transformer_tpp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp no \
  --min_subjects_per_batch 8 \
  --ontology_path       /user/zj2398/cache/survivehr/ontology.pkl \
  --main_split_path     /user/zj2398/cache/survivehr/main_split.csv  \
  --cohort_dir   /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
  --output_dir   /shared/share_mala/zj2398/mimic/task/regression/ \
  --cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --tasks "glucose" \
  --regression 


bash run_regression.sh \
  --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_proc           24 \
  --model_path         /shared/share_mala/zj2398/models/ntp_8k/transformer/best-269840 \
  --model_name         transformer_ntp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp yes \
  --min_subjects_per_batch 8 \
  --ontology_path       /user/zj2398/cache/survivehr/ontology.pkl \
  --main_split_path     /user/zj2398/cache/survivehr/main_split.csv  \
  --cohort_dir   /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
  --output_dir   /shared/share_mala/zj2398/mimic/task/regression/ \
  --cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --tasks "glucose" \
  --regression 
