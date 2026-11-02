# How to prepare and pretrain all models (MIMIC on GSB, CUMC on the cumc cluster)

This guide shows the exact commands to go from a MEDS reader database to six pretrained models:
**ntp, motor, tpp, mtpp, icarefm, survivehr**. Transformer backbone; mamba works the same way with
`mamba.sh` instead of `transformer.sh`.

## 1. The big picture

There are only **three commands** to know. All are run from the repository root.

| Step | Command | What it does | When |
|---|---|---|---|
| 1 | `bash <launcher> tasks` | Trains the **shared tokenizer** once, then fits the **task** (prediction targets) of all six models. No batches, no GPUs. | once per dataset |
| 2 | `python src/femr/training/script/check_tokenization.py ...` | Read-only checks of the tokenizer and the task codes of all six models. | after step 1 finishes |
| 3 | `bash <launcher> <model> train` | For one model: builds its training **batches** (CPU) and then **pretrains** it (GPU). | once per model, as soon as that model's task exists |

`<launcher>` is the script for your dataset and cluster:

| Dataset / cluster | Launcher (transformer) |
|---|---|
| MIMIC-IV on GSB | `src/femr/training/script/mimic/gsb/transformer.sh` |
| CUMC on the cumc cluster | `src/femr/training/script/cumc/transformer.sh` |
| MIMIC-IV on the cumc cluster (kuvira) | `src/femr/training/script/mimic/kuvira/transformer.sh` |

**Do not run `lib.sh` directly.** Each launcher loads `lib.sh` (the shared code: settings, logging,
the `tasks`/`train` logic) and an env file with the dataset's paths:

```
mimic/gsb/transformer.sh  ──loads──>  lib.sh  +  env/mimic_gsb.sh    (paths for MIMIC on GSB)
cumc/transformer.sh       ──loads──>  lib.sh  +  env/cumc_cumc.sh    (paths for CUMC)
```

Why two steps? All six models must use **the same tokenizer**. Step 1 trains it once (inside the
motor job) and only then starts the other five, so nobody trains a second tokenizer. Step 3 never
trains a tokenizer or fits a task; it refuses to start if they are missing.

## 2. Paths (set in the env files)

| Variable | MIMIC on GSB (`env/mimic_gsb.sh`) | CUMC (`env/cumc_cumc.sh`) |
|---|---|---|
| `MEDS_READER` (input data) | `/shared/share_mala/zilin/storage/ora/meds_v0.6_reader` | `/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader` |
| `META_DIR` (`ontology.pkl`; the shared `tokenizer/` is written here) | `/shared/share_mala/zilin/storage/ora/metadata` | `/data/models/zj2398/femr/metadata` |
| `OUT_ROOT` (one folder per model) | `/shared/share_mala/zilin/storage/ora/models` | `/data/models/zj2398/femr/models` |
| `HF_CACHE` (temporary batch files) | `/shared/share_mala/zilin/storage/ora/hf_cache` | `/data2/huggingface/zj2398` |
| `TIME_BINS` | 8 | 8 |
| `NUM_THREADS` (CPU workers per job) | 32 | 24 |
| `GPUS` (GPU ids for pretraining; 1, 2 or 4 ids) | `0,1,2,3` | `0,1,2,3` |
| `WANDB_PROJECT` (wandb project of the pretraining runs) | `ora_mimic` | `ora_cumc` |
| `BANNED_PROPERTIES` (columns that get no tokens) | `hadm_id icustay_id` | none |

Any of these can be changed for one command by putting it in front, e.g.
`NUM_THREADS=16 bash <launcher> tasks`. More settings you can change the same way:

| Variable | Default | Used in | Meaning |
|---|---|---|---|
| `TIME_BINS` | 8 | step 1 | time bins of MOTOR, TPP, MTPP |
| `VALUE_BINS` | 5 | step 1 | value bins of MTPP |
| `N_EPOCHS` | 20 | step 3 | training epochs |
| `RUN_NAME` | `<model>_transformer`, e.g. `motor_transformer` | step 3 | wandb run name |

`TIME_BINS` and `VALUE_BINS` are stored in each model's task when step 1 fits it, so set them for
step 1. A model folder holds one task: to try other bins, use another `OUT_ROOT` (the shared tokenizer
in `META_DIR` can stay), e.g.
`OUT_ROOT=/shared/share_mala/zilin/storage/ora/models_T16 TIME_BINS=16 bash <launcher> tasks`.
If the bins you give do not match an existing task, prepare stops with an error instead of silently
using the old bins.

## 3. One-time setup

```bash
cd /user/zj2398/femr_chao_meds_v3                       # repository root (cumc: your clone)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tte
```

Check the inputs:

```bash
ls <META_DIR>/ontology.pkl     # must exist
ls <OUT_ROOT>                  # should be empty (or not exist) for a fresh run
```

**Fresh run = empty model folders.** `prepare.py` silently **reuses** an existing
`<OUT_ROOT>/<model>/<model>_task.pkl` and an existing `<META_DIR>/tokenizer/`. To redo everything,
move these away first (do not just delete; keep old results until you have new ones):

```bash
mv <META_DIR>/tokenizer <META_DIR>/tokenizer_old
mv <OUT_ROOT> <OUT_ROOT>_old
```

## 4. Step 1: shared tokenizer + all tasks

MIMIC on GSB:
```bash
nohup bash src/femr/training/script/mimic/gsb/transformer.sh tasks > tasks_mimic.out 2>&1 &
```

CUMC:
```bash
nohup bash src/femr/training/script/cumc/transformer.sh tasks > tasks_cumc.out 2>&1 &
```

What happens inside (you do not run these yourself):

1. **motor** starts. Its first step trains the shared tokenizer into `<META_DIR>/tokenizer/` and copies
   it to `<OUT_ROOT>/motor/tokenizer/`; then it fits the MOTOR task.
2. As soon as that copy exists, **ntp, tpp, mtpp, icarefm, survivehr** start **in parallel**. They load
   the same tokenizer (and copy it into their own folder) and fit their tasks.
3. Each model writes `<OUT_ROOT>/<model>/<model>_task.pkl` and stops (`--tasks_only`: no batches).

Follow the progress:

```bash
cat tasks_mimic.out                         # start / ok / FAILED per model
ls <OUT_ROOT>/*/*_task.pkl                  # finished tasks
tail -f <OUT_ROOT>/tasks_logs/mtpp.out      # console of one model
```

Every run also writes a full log with date, host, git commit, the exact command, end time and exit
code to `<OUT_ROOT>/<model>/logs/prepare_<model>_transformer_<timestamp>.log`.

Typical times on MIMIC (GSB, 32 threads per job): tokenizer ~5 min; ntp, icarefm, survivehr 1–4 min;
motor ~20 min; tpp ~30 min; mtpp ~40 min. CUMC is much larger and will take longer (not measured yet;
the end time in each log tells you). Memory: ~110 GB per MIMIC job, mostly the ontology loaded in
every worker; with less memory, use fewer threads: `NUM_THREADS=16 bash <launcher> tasks`.

Running `tasks` again is safe: models whose task already exists just load it and stop.

## 5. Step 2: check the tokenizer and the tasks

When all six `*_task.pkl` exist:

MIMIC on GSB:
```bash
python src/femr/training/script/check_tokenization.py \
  --out_root /shared/share_mala/zilin/storage/ora/models \
  --meta_dir /shared/share_mala/zilin/storage/ora/metadata \
  --meds_reader /shared/share_mala/zilin/storage/ora/meds_v0.6_reader | tee check_mimic.txt
```

CUMC:
```bash
python src/femr/training/script/check_tokenization.py \
  --out_root /data/models/zj2398/femr/models \
  --meta_dir /data/models/zj2398/femr/metadata \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  | tee check_cumc.txt
```

It reads the data once to count events per code (~10 min on MIMIC, longer on CUMC). Add
`--no_event_counts` for a quick run without these counts.

What it prints and what to expect:

| Section | Expected |
|---|---|
| A. `identical to shared` | `True` for all six models |
| A. numeric bins per column | number of bins and where the bin edges are for each numeric column |
| H1: MOTOR == TPP | `True` |
| H2: MTPP vs TPP | MTPP adds numeric codes (codes measured with values); drops a few codes with < 1000 values |
| H3: NTP == SurvivEHR event codes | `True` |
| H4: ICareFM ⊇ MTPP numeric | `True` or very close (the few exceptions are listed with the reason) |

Every differing code is listed with its event counts (no value / numeric / text / numeric+text) and the
filter message from the model's log. With an ontology (CUMC), codes tagged `[ancestor-only]` are
ontology parents: MOTOR/TPP/MTPP may predict them, NTP/SurvivEHR do not.

## 6. Step 3: batches + pretraining, one command per model

You can start a model **as soon as its `<model>_task.pkl` exists**, even while step 1 is still fitting
the other models. Each command first builds that model's batches on the CPU (`NUM_THREADS` workers),
then pretrains it on the GPUs.

Settings per job:

- `GPUS` = the GPU ids to use, e.g. `GPUS=0,1`. Give **1, 2 or 4** ids. Gradient accumulation is set
  from their number, so every model sees 65,536 tokens per optimizer step. Several jobs can run on
  one machine at the same time on different GPUs.
- `NUM_THREADS` = CPU workers for building batches.
- `N_EPOCHS` = training epochs (default 20).
- `RUN_NAME` = wandb run name (default `<model>_transformer`). Runs go to the wandb project
  `ora_mimic` (MIMIC) or `ora_cumc` (CUMC). If you train the same model twice, give the second run its
  own name, e.g. `RUN_NAME=motor_transformer_e5`.

MIMIC on GSB, example with two models on two GPUs each:
```bash
L=src/femr/training/script/mimic/gsb/transformer.sh
GPUS=0,1 NUM_THREADS=32 nohup bash $L motor train > train_motor.out 2>&1 &
GPUS=2,3 NUM_THREADS=32 nohup bash $L tpp   train > train_tpp.out   2>&1 &
```

The same for every model; change only the model name and the GPUs:
```bash
GPUS=0 NUM_THREADS=32 nohup bash $L ntp       train > train_ntp.out       2>&1 &
GPUS=1 NUM_THREADS=32 nohup bash $L mtpp      train > train_mtpp.out      2>&1 &
GPUS=2 NUM_THREADS=32 nohup bash $L icarefm   train > train_icarefm.out   2>&1 &
GPUS=3 NUM_THREADS=32 nohup bash $L survivehr train > train_survivehr.out 2>&1 &
```

CUMC: identical, with the CUMC launcher:
```bash
L=src/femr/training/script/cumc/transformer.sh
GPUS=0,1,2,3 NUM_THREADS=24 N_EPOCHS=20 nohup bash $L motor train > train_motor.out 2>&1 &
```

Anything after `train` is passed to `pretrain.py`. For a quick test that the whole path works
(batches are built in full, training stops after 3 steps):
```bash
GPUS=0 bash $L motor train --max_steps 3
```
With `--max_steps` the run is a test only: no evaluation, no checkpoints. The batches it built are kept
and reused by the real run.

Outputs per model in `<OUT_ROOT>/<model>/`:

| Path | Content |
|---|---|
| `tokenizer/` | copy of the shared tokenizer |
| `<model>_task.pkl` | the task (step 1) |
| `train_batches/`, `val_batches/` | training and validation batches (step 3) |
| `transformer/` | checkpoints and the trained model (step 3) |
| `logs/` | one log per prepare / pretrain run |

## 7. Hyperparameter search (example: MTPP)

Every configuration is one command. Which command depends on what you change.

### 7.1 Bins (`TIME_BINS`, `VALUE_BINS`): new task and new batches

Bins are stored in the task when it is fit, and the batch labels are built from the task. So every bin
setting needs **its own output folder** (`OUT_ROOT`). The `all` stage does everything for one model in
one command: fit its task (only this model), build its batches, then pretrain. It reuses the shared
tokenizer in `META_DIR`, so the inputs stay identical to the other models.

```bash
L=src/femr/training/script/mimic/gsb/transformer.sh
OUT_ROOT=/shared/share_mala/zilin/storage/ora/models_T8_V10 TIME_BINS=8 VALUE_BINS=10 \
GPUS=0,1 RUN_NAME=mtpp_T8_V10 nohup bash $L mtpp all > mtpp_T8_V10.out 2>&1 &
```

A small grid, one setting after the other on the same GPUs:
```bash
for vb in 3 5 10; do
  OUT_ROOT=/shared/share_mala/zilin/storage/ora/models_T8_V$vb VALUE_BINS=$vb GPUS=0,1 \
  RUN_NAME=mtpp_T8_V$vb bash $L mtpp all > mtpp_T8_V$vb.out 2>&1
done
```
To run settings in parallel instead, start each command with `&` on different `GPUS`.

Cost per setting on MIMIC: MTPP task fit ~40–55 min, then batch building, then training; and a full
copy of the MTPP batches on disk.

### 7.2 Training settings (learning rate, layers, epochs): reuse existing batches

Use the `pretrain` stage on a model folder that already has its batches (after step 3 or 7.1).
Anything after `pretrain` goes to `pretrain.py`, and a later value of the same option wins over the
default, so the options you add override the launcher's settings (`--n_layers`, `--learning_rate`,
`--max_grad_norm`, `--n_epochs`). Epochs can also be set with `N_EPOCHS`.

```bash
L=src/femr/training/script/mimic/gsb/transformer.sh
D=/shared/share_mala/zilin/storage/ora/models/mtpp
GPUS=2,3 RUN_NAME=mtpp_lr5e-5 nohup bash $L mtpp pretrain \
  --learning_rate 5e-5 --output_dir $D/transformer_lr5e-5 > mtpp_lr5e-5.out 2>&1 &
```

**Always give each training run its own `--output_dir` and its own `RUN_NAME`.** Without
`--output_dir`, every run writes its checkpoints to the same `<OUT_ROOT>/<model>/transformer/` and
overwrites the previous one; without `RUN_NAME`, all runs appear under the same name in wandb.
The best model is saved in `<output_dir>/best_model`, so it is kept apart as well.

A grid over learning rates in parallel, two GPUs each:
```bash
gpus=(0,1 2,3)
i=0
for lr in 5e-5 1e-4; do
  GPUS=${gpus[$i]} RUN_NAME=mtpp_lr$lr nohup bash $L mtpp pretrain \
    --learning_rate $lr --output_dir $D/transformer_lr$lr > mtpp_lr$lr.out 2>&1 &
  i=$((i + 1))
done
```

### 7.3 Things to keep in mind

- **Fair comparison.** If only MTPP is tuned, part of its advantage over the other models comes from
  the extra tuning. Give the other models the same budget (e.g. bins for MOTOR/TPP, learning rate for
  all), or report the tuning protocol.
- **Validation loss is not comparable across bin settings.** More bins change MTPP's loss itself
  (more classes per prediction). Compare bin settings on downstream tasks, not on validation loss.
  Validation losses are comparable across training settings that share the same task and batches
  (7.2).
- **CUMC** works the same way with `L=src/femr/training/script/cumc/transformer.sh` and CUMC paths
  (`/data/models/zj2398/femr/models_...` for 7.1).

## 8. Troubleshooting

| Message | Meaning and fix |
|---|---|
| `shared tokenizer or .../<model>_task.pkl missing; run '... tasks' first` | Step 3 was started before step 1 produced this model's task. Wait or run step 1. |
| `... differs from the shared tokenizer ...` | The model folder has a tokenizer from an older run. Move `<OUT_ROOT>/<model>/tokenizer` away and rerun. |
| `GPUS must list 1, 2 or 4 GPU ids` | Give 1, 2 or 4 ids, e.g. `GPUS=0,1`. |
| `... was fit with time_bins=8, but --time_bins 16 was given` | The model folder already has a task with other bins. Use another `OUT_ROOT`, or move `<model>_task.pkl` away to refit. |
| `motor exited before the shared tokenizer was ready` | Tokenizer training failed; read `<OUT_ROOT>/tasks_logs/motor.out`. |
| A model was not refit after a code change | Its old `<model>_task.pkl` was reused. Move it away and rerun step 1. |

## 9. Reference: all launcher commands

```
bash <launcher> tasks [extra prepare.py args]       step 1 for all models
bash <launcher> <model> train [pretrain.py args]    step 3 for one model
bash <launcher> <model> prepare [prepare.py args]   prepare only (tokenizer if missing, task if missing, batches)
bash <launcher> <model> pretrain [pretrain.py args] pretrain only (batches must exist)
bash <launcher> <model> all [pretrain.py args]      prepare + pretrain without the checks of "train"
```
`<model>` is one of `ntp motor tpp mtpp icarefm survivehr`. Shared settings that must be identical
for all models (vocabulary size, tokens per batch, window length, value bins, thresholds) are in
`lib.sh`; backbone settings (layers, learning rate, epochs) are in the launcher.
