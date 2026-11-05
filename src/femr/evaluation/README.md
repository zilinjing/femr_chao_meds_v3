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

| Dataset / cluster | Shared batches | Models (transformer) | Models (mamba) | Baselines |
|---|---|---|---|---|
| MIMIC-IV on GSB | built by the first model | `src/femr/evaluation/script/mimic/gsb/transformer.sh` | `.../mimic/gsb/mamba.sh` | `.../mimic/gsb/baseline.sh` |
| CUMC on the cumc cluster | `src/femr/evaluation/script/cumc/batches.sh` | `src/femr/evaluation/script/cumc/transformer.sh` | `.../cumc/mamba.sh` | `.../cumc/baseline.sh` |

The launchers are three lines each; they load the shared code `script/lib.sh` (settings that must be
identical for every model) and an env file with the dataset's paths (`script/env/mimic_gsb.sh`,
`script/env/cumc_cumc.sh`). This mirrors the pretraining launchers in `src/femr/training/script/`.

## 3. Evaluate the models

Evaluation has two steps, always in this order:

```
step 1: shared batches (CPU, once per task)  ──>  step 2: run the models (GPU, once per model)
```

Turning patients into batches is the slow part, and batches do not depend on the model weights. So
they are made once per task, and every model with the same tokenizer reuses them.

### Step 1: prepare the shared batches

**CUMC.** Do this before running any model. It needs no GPU and no trained model, only the shared
tokenizer in `META_DIR/tokenizer/`.

First check the labels, tokenizer, packages and machine. This only reads files. Fix every `PROBLEM`
line before going on.

```bash
S=src/femr/evaluation/script/cumc
bash $S/preflight.sh | tee preflight.txt
```

Then build the batches. There is one group per task family:

| Group | Tasks on CUMC |
|---|---|
| `outcome` | `death`, `long_los`, `readmission` |
| `phenotype` | `AMI`, `Celiac`, `Ischemic_Stroke`, `MASLD` |
| `regression` | `platelets`, `creatinine`, `spo2`, `glucose` |

All three groups at the same time, with one command:

```bash
nohup bash $S/batches.sh all > batches.out 2>&1 &
```

Or start each group with its own command. A command only builds the group it names. The three lines
below also run at the same time, because each ends with `&`, so this is the same as `all` with one
`.out` file per group:

```bash
nohup bash $S/batches.sh outcome    > batches_outcome.out 2>&1 &
nohup bash $S/batches.sh phenotype  > batches_phenotype.out 2>&1 &
nohup bash $S/batches.sh regression > batches_regression.out 2>&1 &
```

If memory is short, start a group only after the previous one has finished.

Only some tasks of a group:

```bash
nohup bash $S/batches.sh regression glucose,platelets > batches_regression.out 2>&1 &
```

A group runs its tasks one after the other with `BATCH_WORKERS` workers (default 28). To change it,
put it in front: `BATCH_WORKERS=32 nohup bash $S/batches.sh all > batches.out 2>&1 &`.

Each group has its own log. With `all`, `batches.out` only shows the start and the end, so follow the
logs:

```bash
R=/data/models/zj2398/femr/results
tail -f $R/logs/batches_phenotype_*.log
```

Step 1 is finished when all 11 tasks say `ok` or `already built`, and none says `FAILED`:

```bash
grep -hE 'batches [a-z]+/.*: (ok|FAILED)|already built' $R/logs/batches_*.log
```

**Wait for step 1 to finish before starting step 2.** A model started too early finds no batches and
builds the same ones itself; both jobs then use the same temporary folder, and one of them can crash.

Good to know:

- Rerunning a command is safe: tasks that are already built are skipped, so after a `FAILED` task just
  run the same command again.
- Batches are stored in `<HF_CACHE>/eval_batches/<family>_<task>_<key>/`. To rebuild a task, delete its
  folder and rerun. The key changes with the tokenizer, ontology, labels and batch settings, so
  changed labels get a new folder by themselves; delete the old one to free the disk.
- Every task has a fixed start-up cost of several minutes, however small its cohort.
- Memory depends on the number of workers, not on the size of the cohort. Each group has 24 reader
  workers plus its `BATCH_WORKERS`, and a reader worker held 5.7 GB on MIMIC. Plan for up to about
  900 GB for `all` with the default 28 (batch workers were not measured, so this assumes they cost the
  same). The figure printed by `preflight.sh` assumes memory follows the size of the ontology, which
  was not tested. If memory runs out, lower `BATCH_WORKERS` or run the groups one after the other.
- `batches.sh` builds batches for the transformer models. A mamba model builds its own on its first
  run, so run one mamba model first, as on MIMIC below.

**MIMIC on GSB.** There is no separate step: the first model of a task builds its batches and saves
them in `<HF_CACHE>/eval_batches/`, and later models load them. So run one model first, wait for it to
finish, then run the others.

### Step 2: run the models

For each task, a model computes its features from the saved batches (GPU), then the probe is trained
and scored (CPU).

```
bash <launcher> <model> <family> [task1,task2]
```

- `<model>`: `ntp`, `motor`, `tpp`, `mtpp`, `icarefm` or `survivehr`. The model is read from
  `<MODELS_ROOT>/<model>/transformer/best_model` (the best checkpoint of pretraining).
- `<family>`: `outcome`, `phenotype`, `regression`, or `all` (the three, one after the other).
- `[task1,task2]` (optional): only these tasks. Without it, every task of the family is run.

Put settings in front of the command: `GPU=` picks the GPU (default 0), `NUM_PROC=` the CPU workers
of this job (default 24). GPUs are shared, so check `nvidia-smi` before picking one. The examples use
the CUMC launcher; for MIMIC only `L` changes:

```bash
L=src/femr/evaluation/script/cumc/transformer.sh     # MIMIC: src/femr/evaluation/script/mimic/gsb/transformer.sh
```

**One model:**

```bash
GPU=0 nohup bash $L motor all > eval_motor.out 2>&1 &                          # all three families
GPU=0 nohup bash $L motor outcome > eval_motor_outcome.out 2>&1 &              # one family
GPU=0 nohup bash $L motor regression glucose,spo2 > eval_motor_lab.out 2>&1 &  # some tasks
```

**One model, faster:** one family per GPU.

```bash
GPU=0 nohup bash $L motor outcome    > eval_motor_outcome.out 2>&1 &
GPU=1 nohup bash $L motor phenotype  > eval_motor_phenotype.out 2>&1 &
GPU=2 nohup bash $L motor regression > eval_motor_regression.out 2>&1 &
```

**Several models at once:** one model per GPU. List as many models as you have free GPUs, and leave
out any model that is not trained yet (`ls <MODELS_ROOT>/*/transformer/best_model/config.json` shows
the trained ones).

```bash
gpu=0
for m in ntp motor tpp icarefm; do
  GPU=$gpu nohup bash $L $m all > eval_$m.out 2>&1 &
  gpu=$((gpu + 1))
done
```

**More models than GPUs:** give each GPU a queue. Every line is one GPU; its models run one after the
other, and a model that fails does not stop the next one.

```bash
export L
nohup bash -c 'for m in ntp survivehr; do GPU=0 bash $L $m all > eval_$m.out 2>&1; done' > /dev/null 2>&1 &
nohup bash -c 'for m in motor mtpp;    do GPU=1 bash $L $m all > eval_$m.out 2>&1; done' > /dev/null 2>&1 &
nohup bash -c 'for m in tpp;           do GPU=2 bash $L $m all > eval_$m.out 2>&1; done' > /dev/null 2>&1 &
nohup bash -c 'for m in icarefm;       do GPU=3 bash $L $m all > eval_$m.out 2>&1; done' > /dev/null 2>&1 &
```

Models running at the same time do not disturb each other: each writes to its own folders, and the
summary tables are rebuilt under a lock.

**Follow the progress** in the `.out` files. The same output is also kept in
`<RESULT_ROOT>/logs/<model>_<family>_<date>.log`, together with the git commit and exact command.
Each task ends with `ok` or `FAILED`, and a failed task does not stop the other tasks:

```bash
grep -E ': (ok|FAILED)$' eval_*.out
```

**Check that the models used the shared batches.** Each task that computed features should print
`Loading saved batches`, and `Got batches` should not appear, because it means the model built its own:

```bash
grep -c 'Loading saved batches' eval_motor.out    # one per task
grep -l 'Got batches' eval_*.out                  # should print nothing
```

**Mamba models**: use `mamba.sh` instead of `transformer.sh`. The model is read from
`<MODELS_ROOT>/<model>/mamba/best_model` and its row in the tables is `<model>_mamba`.

**Another checkpoint** (not the best one): give its path and a row name for the tables. It uses the
same batches if it has the same tokenizer.

```bash
MODEL_PATH=/shared/share_mala/zilin/storage/ora/models/motor/transformer/checkpoint-47630 \
TAG=motor_ckpt47630 GPU=0 bash $L motor outcome
```

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
| `batches <family>/<task>: FAILED` | Look above it in `<RESULT_ROOT>/logs/batches_<family>_*.log` for the Python error, then rerun the same command: finished tasks are skipped. |
| `no tokenizer at .../tokenizer` | The shared tokenizer is missing from `META_DIR`. Run the `tasks` step of the training launcher first. |
| The machine runs out of memory during step 1 | Lower `BATCH_WORKERS`, or run the groups one after the other. |
| A model log shows `Got batches` instead of `Loading saved batches` | The model did not find shared batches and built its own: step 1 was not finished for that task, its tokenizer differs from the shared one (`preflight.sh` shows this), or it is a mamba model. Results are still valid; it is only slower. |
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
