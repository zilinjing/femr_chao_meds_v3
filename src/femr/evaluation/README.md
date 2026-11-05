# How to evaluate the pretrained models

This guide explains how to measure how good each pretrained model (**ntp, motor, tpp, mtpp,
icarefm, survivehr**) is on the downstream tasks, on MIMIC-IV (GSB cluster) and on CUMC (cumc
cluster). You don't need to know the code to follow it: every step is one command.

## 1. What evaluation does

A pretrained model turns a patient's history into a vector of numbers (the *features*). We freeze
the model and check how much those features know, with a simple model on top (a *linear probe*):

```
patient history ──> frozen pretrained model ──> features ──> simple probe ──> predictions ──> metrics
                    (GPU, minutes to hours)                  (CPU, minutes)                (1000 resamples)
```

There are three task families:

| Family | Question | Probe | Metric(s) | MIMIC tasks |
|---|---|---|---|---|
| `outcome` | Will this happen during/after this hospital stay? (yes/no) | logistic regression | AUROC, AUPRC | `in_hospital_mortality`, `long_los`, `readmission` |
| `phenotype` | Will the patient develop this disease? (yes/no) and **when**? | logistic regression + DeepHit (time-to-event) | AUROC, AUPRC, C-index | `ami`, `masld`, `stroke` |
| `regression` | What will this lab value be? (a number) | linear regression | R² | `creatinine`, `glucose`, `pao2`, `platelets` |

CUMC uses every task folder of its cohorts; for regression: `platelets`, `creatinine`, `spo2`, `glucose`.

- **AUROC / AUPRC**: how well the yes/no predictions rank patients (higher is better; AUROC 0.5 =
  random; AUPRC is more sensitive for rare outcomes).
- **C-index**: how well the predicted risk orders patients by *when* the disease appears, counting
  only the first 300 days (MIMIC) / 365 days (CUMC) after the prediction time (higher is better).
- **R²**: share of the variation in the lab value that the prediction explains (1 is perfect, 0 is
  no better than always predicting the average).

**How numbers are reported.** Every number is written as `mean (std)`, exactly like
[meds-evaluation](https://github.com/kamilest/meds-evaluation) does it: the test set is resampled
1000 times (rows drawn with replacement, resample *b* uses random seed *b*), the metric is computed
on each resample, and we report the mean and the standard deviation of those 1000 values. The std
shows how much the number would move with a different test set of the same size. Every model and
baseline uses the same 1000 resamples. (C-index only: a resample that happens to contain no disease
event before the horizon has no C-index and is left out; `metrics.json` records how many were used.
This only happens for very rare diseases or tiny test sets.)

## 2. One-time setup

```bash
cd /user/zj2398/femr_chao_meds_v3                       # repository root (cumc: your clone)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tte
```

All commands below are run from the repository root. Pick the launcher for your dataset:

| Dataset / cluster | Models (transformer) | Models (mamba) | Baselines |
|---|---|---|---|
| MIMIC-IV on GSB | `src/femr/evaluation/script/mimic/gsb/transformer.sh` | `.../mimic/gsb/mamba.sh` | `.../mimic/gsb/baseline.sh` |
| CUMC on the cumc cluster | `src/femr/evaluation/script/cumc/transformer.sh` | `.../cumc/mamba.sh` | `.../cumc/baseline.sh` |

The launchers are three lines each; they load the shared code `script/lib.sh` (settings that must be
identical for every model) and an env file with the dataset's paths (`script/env/mimic_gsb.sh`,
`script/env/cumc_cumc.sh`). This mirrors the pretraining launchers in `src/femr/training/script/`.

## 3. Evaluate one model

```
bash <launcher> <model> <family> [task1,task2]
```

- `<model>`: `ntp`, `motor`, `tpp`, `mtpp`, `icarefm` or `survivehr`. The model is read from
  `<MODELS_ROOT>/<model>/transformer/best_model` (the best checkpoint of pretraining).
- `<family>`: `outcome`, `phenotype`, `regression`, or `all` (the three, one after the other).
- `[task1,task2]` (optional): only these tasks. Without it, every task of the family is run.

Put settings in front of the command: `GPU=` picks the GPU (default 0), `NUM_PROC=` the CPU workers.

**MIMIC on GSB**, motor, one family per GPU:

```bash
L=src/femr/evaluation/script/mimic/gsb/transformer.sh
GPU=0 nohup bash $L motor outcome    > eval_motor_outcome.out 2>&1 &
GPU=1 nohup bash $L motor phenotype  > eval_motor_phenotype.out 2>&1 &
GPU=2 nohup bash $L motor regression > eval_motor_regression.out 2>&1 &
```

The same for every model: change only the model name (and the GPU).

```bash
GPU=3 nohup bash $L mtpp all > eval_mtpp.out 2>&1 &                 # all three families
GPU=4 nohup bash $L ntp outcome readmission > eval_ntp.out 2>&1 &   # one task
```

All six models, one GPU each, all families:

```bash
gpu=0
for m in ntp motor tpp mtpp icarefm survivehr; do
  GPU=$gpu nohup bash $L $m all > eval_$m.out 2>&1 &
  gpu=$((gpu + 1))
done
```

**CUMC**: identical, with the CUMC launcher.

```bash
L=src/femr/evaluation/script/cumc/transformer.sh
GPU=0 nohup bash $L survivehr all > eval_survivehr.out 2>&1 &
```

**Mamba models**: use `mamba.sh` instead of `transformer.sh`. The model is read from
`<MODELS_ROOT>/<model>/mamba/best_model` and its row in the tables is `<model>_mamba`.

**Another checkpoint** (not the best one): give its path and a row name for the tables.

```bash
MODEL_PATH=/shared/share_mala/zilin/storage/ora/models/motor/transformer/checkpoint-47630 \
TAG=motor_ckpt47630 GPU=0 bash $L motor outcome
```

Follow the progress in the `.out` file. The same output is also kept in
`<RESULT_ROOT>/logs/<model>_<family>_<date>.log`, together with the git commit and exact command.
At the end, each task says `ok` or `FAILED`; a failed task does not stop the other tasks.

**Saved batches.** Most of the time goes into turning patients into batches, and batches do not depend
on the model weights. The first model of a task saves them in `<HF_CACHE>/eval_batches/`; later models
with the same tokenizer load them (`preprocess_time` near 0), so run one model first and the others
after it. Delete that folder to rebuild the batches.

**CUMC: build the batches first.** On CUMC, build the batches of every task before evaluating any
model, then run the models as usual; each one finds its batches already there. No GPU or trained model
is needed: only the shared tokenizer in `META_DIR/tokenizer/`.

```bash
bash src/femr/evaluation/script/cumc/preflight.sh | tee preflight.txt   # read-only checks; fix every PROBLEM line
nohup bash src/femr/evaluation/script/cumc/batches.sh all > batches.out 2>&1 &
```

`all` runs outcome, phenotype and regression at the same time, each with `BATCH_WORKERS` (default 28)
workers and its own log, `<RESULT_ROOT>/logs/batches_<family>_<date>.log`. To run one family:
`bash .../batches.sh regression`. A task that is already built is skipped. Batches are built for the
transformer models; a mamba model builds its own on its first run.

Memory depends on the number of workers, not on the cohort: every worker keeps its own copy of the
tokenizer and ontology (5.7 GB per reader worker on MIMIC, so about 450 GB for the 72 reader workers
of `all`). `preflight.sh` prints the estimate for this ontology. If it does not fit, lower
`BATCH_WORKERS` or run the families one after the other. Each task also has a fixed start-up cost of
several minutes, however small its cohort.

## 4. Results

| | MIMIC on GSB | CUMC |
|---|---|---|
| `RESULT_ROOT` | `/shared/share_mala/zilin/storage/ora/result` | `/data/models/zj2398/femr/results` |

**The summary tables** are rebuilt automatically at the end of every run (one row per model or
baseline, one column per task, cells `mean (std)`):

| File | Content |
|---|---|
| `<RESULT_ROOT>/outcome/outcome_auroc.csv` | AUROC of the outcome tasks |
| `<RESULT_ROOT>/outcome/outcome_auprc.csv` | AUPRC of the outcome tasks |
| `<RESULT_ROOT>/phenotype/phenotype_auroc.csv` | AUROC of the phenotype tasks |
| `<RESULT_ROOT>/phenotype/phenotype_auprc.csv` | AUPRC of the phenotype tasks |
| `<RESULT_ROOT>/phenotype/phenotype_cindex.csv` | C-index of the phenotype tasks |
| `<RESULT_ROOT>/regression/regression_r2.csv` | R² of the regression tasks |

Example (`outcome_auroc.csv`):

```
model,in_hospital_mortality,long_los,readmission
motor,0.962 (0.003),...,...
mtpp,...
xgboost,...
```

To rebuild the tables by hand (e.g. after deleting a result folder): `bash <launcher> collect`.

**Per model and task**, in `<RESULT_ROOT>/<family>/<task>/<model>/`:

| File | Content |
|---|---|
| `metrics.json` | the numbers of the tables, e.g. `{"auroc": {"mean": 0.962, "std": 0.003, "n_boot": 1000}}` |
| `predictions.parquet` | one row per test prediction: `subject_id`, `prediction_time`, the true label and the prediction (binary: `boolean_value`, `predicted_boolean_probability`; regression: `y_true`, `y_pred`) |
| `tte_predictions.parquet` | phenotype only: `time_to_event`, `tte_label` and the predicted `risk` |

The predictions are kept so that models can later be compared with statistical tests without
rerunning anything.

**Features** are stored in `<RESULT_ROOT>/<family>/_features/<model>/` and reused: running the same
model and task again only reruns the probe (minutes). Delete the `.pkl` file to recompute them,
e.g. after the model was retrained.

## 5. Baselines

Each task has one baseline that does not use a pretrained model:

| Family | Baseline | Row in the tables |
|---|---|---|
| outcome | XGBoost on MEDS-Tab features (counts and value statistics of every code, over the last 7 days and the whole history) | `xgboost` |
| phenotype, AUROC/AUPRC | XGBoost on MEDS-Tab features | `xgboost` |
| phenotype, C-index | DeepHit on MEDS-Tab features | `deephit` |
| regression | the most recent value of the same lab before the prediction time (training average if there is none) | `most_recent` |

Commands (same on both clusters; use the dataset's `baseline.sh`):

```bash
B=src/femr/evaluation/script/mimic/gsb/baseline.sh       # CUMC: src/femr/evaluation/script/cumc/baseline.sh

# Regression: no extra step (CPU).
nohup bash $B regression > baseline_regression.out 2>&1 &

# Outcome and phenotype: 1) build the MEDS-Tab features once (CPU, slow), 2) train.
nohup bash $B extract > baseline_extract.out 2>&1 &          # all cohorts; or: extract outcome|binary|tte
nohup bash $B outcome > baseline_outcome.out 2>&1 &          # XGBoost (CPU)
GPU=0 nohup bash $B phenotype > baseline_phenotype.out 2>&1 &   # XGBoost + DeepHit (DeepHit uses the GPU)
```

- `XGB_DEVICE=cuda` in front of `outcome`/`phenotype` trains XGBoost on the GPU (much faster).
- Like the models, `[task1,task2]` limits a command to some tasks, e.g. `bash $B phenotype ami`.
- The baseline results land in the same tables as the models. The MEDS-Tab features are kept in
  `MEDS_TAB_DIR` (`/shared/share_mala/zilin/storage/ora/meds_tab` on GSB,
  `/data/models/zj2398/femr/meds_tab` on CUMC); training logs and extra files in `MEDS_TAB_DIR/results`.
- A baseline task that already has its result in the table is skipped when you rerun a command.
  To retrain it, delete `<RESULT_ROOT>/<family>/<task>/<xgboost|deephit>/` and the matching
  `MEDS_TAB_DIR/results/<task>_<xgboost|deephit>*` files.
- The MEDS-Tab baseline is expensive: DeepHit runs a hyperparameter search of 32 configurations per
  task (over an hour per MIMIC task), and the feature matrix of CUMC is much larger than MIMIC's.
  Details of the features and the search: `baseline/meds_tab/README.md`.
- On CUMC, `DEEPHIT_GRID` and `DEEPHIT_WD` in the env file select 72 configurations: learning rate
  {1e-3, 1e-4}, batch size {1024, 2048}, dropout {0, 0.1}, hidden width {0, 64, 128} and weight decay
  {0, 1e-4, 1e-2}.
- DeepHit keeps a dense copy of the features in memory. After `extract`, rerun `preflight.sh`: the
  `dense_float64_gb` of a task is about the peak RAM of its DeepHit run (within 3% on MIMIC).
- A baseline task that crashes prints `[ERROR]` and the command still ends normally, so check with
  `grep -n '\[ERROR\]' <RESULT_ROOT>/logs/baseline_*.log`.
- XGBoost needs the `xgboost` package: `pip install xgboost` in the `tte` env (already installed on GSB).

## 6. Settings

Everything that must be the same for every model is fixed in `script/lib.sh`:

| Setting | Value | Meaning |
|---|---|---|
| `TOKENS_PER_BATCH`, `MIN_SUBJECTS_PER_BATCH` | 65536, 8 | at most 8192 events per patient, as in pretraining |
| `PHENOTYPE_OBS_WINDOW` | 730 | phenotype features only see the last 730 days before the prediction time |
| `N_BOOT` | 1000 | test-set resamples behind every `mean (std)` |

Dataset paths are in the env files (`script/env/*.sh`). Any of them can be changed for one command
by putting it in front:

| Variable | MIMIC on GSB | CUMC |
|---|---|---|
| `MEDS_READER` (patient data) | `/shared/share_mala/zilin/storage/ora/meds_v0.6_reader` | `.../ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader` |
| `META_DIR` (`ontology.pkl`, `main_split.csv`) | `/shared/share_mala/zilin/storage/ora/metadata` | `/data/models/zj2398/femr/metadata` |
| `MODELS_ROOT` (pretrained models) | `/shared/share_mala/zilin/storage/ora/models` | `/data/models/zj2398/femr/models` |
| `RESULT_ROOT` (results) | `/shared/share_mala/zilin/storage/ora/result` | `/data/models/zj2398/femr/results` |
| `OUTCOME_COHORT` | `.../ora/cohort/outcome/cohort_100k` | `.../task_labels/patient_outcomes_clean` |
| `PHENOTYPE_BINARY_COHORT`, `PHENOTYPE_TTE_COHORT` | `.../ora/cohort/phenotype/{binary,tte}` | `.../task_labels/phenotype_tte/{binary,tte}` |
| `REGRESSION_COHORT` | `.../ora/cohort/regression/icu_stay_4h_100k` | `/data2/processed_datasets/zj2398/cumc_data/regression_100k` |
| `REGRESSION_TASKS` | all task folders | `platelets,creatinine,spo2,glucose` |
| `TRUNCATION` (C-index horizon, days) | 300 | 365 |
| `NUM_PROC` (CPU workers) | 24 | 24 |
| `BATCH_WORKERS` (`batches.sh`, workers per family) | not used | 28 |
| `DEEPHIT_GRID`, `DEEPHIT_WD` (DeepHit search) | `run_all.py` defaults | `full`, `0,1e-4,1e-2` (72 configs) |
| `TTE_COL` (time column of the TTE labels) | `time_to_event_days` | `time_to_event` |
| `GPU` | 0 | 0 |

Example: write results to a test folder instead: `RESULT_ROOT=/tmp/eval_test bash $L motor outcome`.

The train/test split is `META_DIR/main_split.csv`, the same file used in pretraining: the probes are
trained on the train patients and scored on the test patients.

## 7. Troubleshooting

| Message | Meaning and fix |
|---|---|
| `no trained model at .../best_model` | This model is not pretrained yet (or the backbone is wrong). Train it first, or set `MODEL_PATH=`. |
| `no cohort folder ...` | The task name is misspelled, or the cohort path in the env file is wrong. `ls` the cohort folder to see the task names. |
| `<task>: FAILED` | Look above it in the log for the Python error. The other tasks still ran. |
| CUDA out of memory | The GPU is shared with another job. Pick another `GPU=`. |
| A number did not change after retraining a model | The old features were reused. Delete `<RESULT_ROOT>/<family>/_features/<model>/` and rerun. |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` in the `tte` env. |
| `[ERROR] ... survival: ValueError: all samples are censored` (baseline) | The test set has almost no disease events before the horizon (`TRUNCATION`); the DeepHit row cannot be scored for that task. |

## 8. Where the code is

```
evaluation/
├── README.md                     this guide
├── script/                       launchers (start here)
│   ├── lib.sh                    shared code and settings
│   ├── env/                      dataset paths: mimic_gsb.sh, cumc_cumc.sh
│   ├── mimic/gsb/                transformer.sh, mamba.sh, baseline.sh
│   └── cumc/                     transformer.sh, mamba.sh, baseline.sh, batches.sh, preflight.sh
├── common/
│   ├── generate_features.py      features of the frozen model (GPU)
│   ├── preflight.py              read-only checks before a run (labels, tokenizer, machine)
│   ├── metrics.py                AUROC/AUPRC (meds-evaluation), C-index, R², 1000 resamples
│   └── collect.py                builds the summary CSVs
├── outcome/probe.py              logistic regression probe
├── phenotype/probe.py            logistic regression + DeepHit probe
├── phenotype/tte/                older TTE analysis scripts (stratification, paired tests)
├── regression/probe.py           linear regression probe
├── regression/baseline/          most-recent-value baseline
├── regression/cohort/            how the regression cohorts were built
├── baseline/meds_tab/            MEDS-Tab features + XGBoost/DeepHit baselines
├── baseline/count/               older count-feature baseline (not in the tables)
├── analysis/                     stratified analyses and older significance tests
├── model/few_shot/               few-shot experiments
└── archive/                      old scripts, kept for reference only
```

Each Python file can also be run on its own; `python -m femr.evaluation.<family>.probe --help`
lists its inputs.
