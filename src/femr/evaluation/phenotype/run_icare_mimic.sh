EVAL="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # femr/evaluation

bash "$EVAL/regression/script/run_regression.sh" \
  --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --num_proc           24 \
  --model_path         /user/zj2398/cache/icarefm/mamba/best_45080 \
  --model_name         icare_mamba \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp yes \
  --min_subjects_per_batch 8 \
  --ontology_path       /user/zj2398/cache/survivehr/ontology.pkl \
  --main_split_path     /user/zj2398/cache/survivehr/main_split.csv  \
  --cohort_dir   /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
  --output_dir   /shared/share_mala/zj2398/mimic/task/regression/ \
  --cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --tasks "glucose,creatinine,pao2,platelets" \
  --regression 

bash "$EVAL/script/run.sh" \
  --meds_reader        /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader  \
  --num_proc           24 \
  --model_path         /user/zj2398/cache/icarefm/mamba/best_45080 \
  --model_name         icare_mamba \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        phenotype \
  --cache_path         /shared/share_mala/zj2398/huggingface/datasets \
  --min_subjects_per_batch 8 \
  --ontology_path       /user/zj2398/cache/survivehr/ontology.pkl \
  --main_split_path     /user/zj2398/cache/survivehr/main_split.csv  \
  --output_dir   /shared/share_mala/zj2398/mimic/task/phenotype_task  \
  --truncation   300 \
  --observation_window 730 \
  --classification_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/binary/ \
  --tte_cohort /shared/share_mala/zj2398/mimic/task/phenotype_task/cohort/tte/ 