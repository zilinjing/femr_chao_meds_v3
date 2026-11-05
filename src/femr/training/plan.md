# Plan and change log: shared tokenizer, uncapped targets, unified launch scripts

Status as of 2026-11-01 (late evening). Branch `original`; changes are uncommitted.
Companion document explaining the design: `src/femr/training/tokenizer_and_targets.md`.

## Goal

Fair comparison of pretraining objectives (NTP, MOTOR, TPP, MTPP, SurvivEHR, ICareFM) on MIMIC-IV
and CUMC, with both transformer and mamba backbones:

1. Every model uses **one shared tokenizer**, trained once per dataset, so inputs are identical.
2. MOTOR/TPP/MTPP start from **all vocabulary code tokens** with **no task cap**; each keeps only
   the filters its loss needs. NTP keeps predicting all observed code tokens.
3. One launcher with **identical batching** (tokens per step) for every model, and reproducible logs.

## Decisions made with the user

| Topic | Decision |
|---|---|
| Tokenizer | One class `Tokenizer` (former `CLMBRTokenizer` logic). Delete `HierarchicalTokenizer`/`FlatTokenizer` files; keep both old names as aliases. Old dictionaries without `original_codes` must still load. |
| `original_codes` | Computed **after** the `vocab_size` cut. |
| Determinism | Tokenizer training must be reproducible (same data and `num_threads`). |
| ID columns (`hadm_id`, `order_id`, …) | Keep as numeric features (unchanged). |
| NTP / SurvivEHR event targets | All `original_codes`, no filtering. |
| MOTOR / TPP | Upstream rule unchanged (only value-less events count). Candidates = all vocab code tokens, no cap. |
| MTPP | Candidates = all vocab code tokens, no cap. Keep `rate == 0` and `frac_events < 1/1000`. Text-valued events stay excluded. Codes with any numeric value but < 1000 **raw** numeric-only values stay **dropped** (user's choice; raw count since Phase 4, same rule as ICareFM/SurvivEHR). |
| SurvivEHR / ICareFM value codes | Keep their own rule (≥ 1000 raw numeric-only events). |
| Ontology (CUMC) | Native targets: MOTOR/TPP/MTPP may keep ancestor tasks. |
| MTPP variant | Prepare as `mtpp`, pretrain as `mtpp_shared`. No `mtpp_factorized` / `mtpp_value` scripts. |
| Batching | `tokens_per_batch=16384`, `min_subjects_per_batch=2` (transformer windows 8192), mamba windows 8192, global tokens per optimizer step 65,536 → grad accumulation 4/2/1 for 1/2/4 GPUs. |
| ID columns (Phase 4) | Excluded from the tokenizer per dataset (`BANNED_PROPERTIES` in `env/*.sh`): MIMIC `hadm_id icustay_id` (they took ~75% of the 1,000-bin numeric budget); CUMC none (its only ID column, `visit_id`, is always skipped by the tokenizer). Reverses the earlier "keep ID columns" decision for these two. |
| Reservoir sampler (Phase 4) | Fix `femr.stat_utils.ReservoirSampler` (bug inherited from upstream FEMR); retrain tokenizer and refit all tasks. |
| MTPP value-less occurrences (Phase 4) | Option (a): a value-less next occurrence of a numeric task is a time-only event; loss = `−log Σ_v p(t*, v)`. |
| Run flow | `tasks` (shared tokenizer + all task pickles) once per dataset, then `<model> train` (batches + pretrain) per model with its own GPUs/threads. Transformer only for now. CUMC task fitting runs unchanged and is timed from the logs. |
| Hyperparameters | Transformer: 11 layers, lr 1e-5, clip 1.0, 20 epochs. Mamba: 28 layers, lr 2e-4, clip 5.0, 20 epochs. `time_bins` 8 (MIMIC) / 10 (CUMC), `value_bins` 5, `value_count_threshold` 1000, SurvivEHR `--survivehr_time_scale_days 1.0`. |
| Scripts | `lib.sh` + `env/*.sh` + one script per dataset/cluster/arch taking the model as argument; old scripts moved to `legacy/`. |
| Paths | GSB MIMIC: reader `/shared/share_mala/zilin/storage/ora/meds_v0.6_reader`, metadata `.../ora/metadata`, models `.../ora/models`, HF cache `.../ora/hf_cache`. cumc-cluster MIMIC: metadata copied from GSB to `/data2/processed_datasets/zj2398/femr/mimic/metadata`, models `.../mimic/models`. CUMC: metadata `/data/models/zj2398/femr/metadata` (`ontology.pkl`, `main_split.csv`, shared `tokenizer/`), models `/data/models/zj2398/femr/models/<model>`; reader and HF cache unchanged under `/data2`. |

## Changes made

### Phase 1: one shared tokenizer

| File | Change |
|---|---|
| `src/femr/models/tokenizer/clmbr_tokenizer.py` → `tokenizer.py` | `git mv`; class renamed `Tokenizer`. `original_codes` merged as a **set** (was a list with O(n²) membership checks). Each shard seeds `random` with its first subject id and returns `first_subject_id`; `train()` sorts shard results by it and seeds `random.seed(0)` before merging (reservoir merge is random). Vocabulary sorted with `vocab_sort_key` (score, then type/property/name/value) instead of score only, so ties do not depend on hash order. `original_codes` computed after `vocab[:vocab_size]`. `__init__` accepts dictionaries without `original_codes`; `get_event_code_token` then raises a clear error. |
| `src/femr/models/tokenizer/hierarchical_tokenizer.py`, `flat_tokenizer.py` | Deleted (`git rm`). |
| `src/femr/models/tokenizer/__init__.py` | Exports `Tokenizer`; `HierarchicalTokenizer = CLMBRTokenizer = Tokenizer`. |
| `src/femr/models/processor.py` | Removed the four `FlatTokenizer` branches and the now-unused `per_subject_tokens`; the hierarchical code path is unchanged apart from indentation. |
| `src/femr/models/architecture/embedding.py` | Tokenizer loading uses `Tokenizer.from_pretrained` (the `use_ntp` argument is kept but no longer changes loading). |
| `src/femr/training/pretrain.py` | One `Tokenizer.from_pretrained` call instead of the loss-type branch. |
| `src/femr/training/prepare.py` | New `--tokenizer_path` (shared tokenizer; trained there if missing, then copied into `<pretraining_data>/tokenizer`; an existing copy that differs byte-wise raises an error). New `--min_subjects_per_batch` (default 1) passed to `convert_dataset`. One train/load path. |
| `tests/models/test_tokenizer.py` | New: ancestor P is an input token but not in `original_codes`; `original_codes` ⊆ vocab after a binding `vocab_size`; two trainings with `num_threads=2` give identical msgpack bytes; old dictionary without `original_codes` loads; aliases. Builds a toy `meds_reader` DB with `meds_reader_convert`. |

Not changed on purpose (reported only): `is_hierarchical=False` code paths in `embedding.py`/`config.py`
are now unused; type hints in task files and `finetune_script/finetuning.py` still name the aliases
(they work). (`banned_properties` was fixed in Phase 4.)

### Phase 2: target rules

| File | Change |
|---|---|
| `src/femr/training/prepare.py` | `num_tasks` for motor/tpp/mtpp/mtpp_shared/mtpp_value = number of `code` entries in the tokenizer vocabulary (was `8 * 1024`). `--num_final_tasks` default `None` → no cap (falls back to the number of code tokens). |

Task-fitting code (`motor.py`, `tpp.py`, `mtpp.py`, `survivehr_task.py`, `icarefm_task.py`, `ntp.py`) is unchanged.

### Phase 3: launch scripts (`src/femr/training/script/`)

| File | Content |
|---|---|
| `lib.sh` | Shared settings, `prepare`, `pretrain`, `run` (logs to `<model dir>/logs/<stage>_<model>_<arch>_<timestamp>.log` with date, host, `git describe --dirty`, python path, `CUDA_VISIBLE_DEVICES`, the exact quoted command, and (Phase 5) end time and exit code), `main` (validates model/stage, creates the model dir, symlinks the shared `ontology.pkl` into it because `pretrain.py` reads it from there). `PYTHONUNBUFFERED=1`. Phase 5: `tasks` and `train` (see below); extra args of `prepare` go to `prepare.py`. |
| `env/mimic_gsb.sh`, `env/mimic_cumc.sh`, `env/cumc_cumc.sh` | Paths, `TIME_BINS`, `NUM_THREADS`, `NUM_GPUS`, `BANNED_PROPERTIES` (Phase 4/5); every variable can be overridden from the environment (`BANNED_PROPERTIES="a b"`). |
| `mimic/gsb/{transformer,mamba}.sh`, `mimic/kuvira/{transformer,mamba}.sh` (env `mimic_cumc`), `cumc/{transformer,mamba}.sh` | `ARCH` and `ARCH_ARGS`; usage `bash <script> tasks` or `bash <script> <ntp|motor|tpp|mtpp|icarefm|survivehr> [prepare|pretrain|all|train] [extra args]`. |
| `legacy/` | All previous scripts moved unchanged (their relative `cd ../..` no longer points at `training/`; kept for reference only). |

### Phase 4: fixes found while preparing MIMIC (Stage A)

| File | Change |
|---|---|
| `src/femr/stat_utils.py` | `ReservoirSampler.add`: once full, a sample enters with probability `size·w/W` (was `w/W`, the size-1 rule, so the reservoir kept ~only the first `size` samples: 1M values, k=10k → 100% from the first 10k, ~3 replacements). `combine`: two full reservoirs merge slot-wise (each slot from `other` with probability `W_other/(W_self+W_other)`); a partial one is replayed into the full one. Same bug in upstream FEMR. Affected: tokenizer numeric bin edges, MOTOR global time bins, TPP/MTPP per-task time bins, MTPP value bins, SurvivEHR 1–99% value ranges. ICareFM (mean/std) and NTP were not affected. |
| `src/femr/models/tokenizer/tokenizer.py` | `banned_properties` now works (filters the properties passed to `map_statistics`) and no longer mutates `db.properties`. |
| `src/femr/training/prepare.py` | `--banned_properties` (columns without tokens) and `--tasks_only` (stop after the tokenizer and the task pickle; no batches). |
| `src/femr/models/tasks/mtpp.py` (Q1) | `_prefit_motor_map` also counts raw numeric-only finite values per task (ICareFM rule); the numeric-task threshold uses that count instead of `len(value samples)`, which counted a value once per label position before it (ratio to raw 0.01–80, median 3.1). Value bins still come from the label-position samples. On v1, all 92 MTPP numeric codes missing from ICareFM had 28–957 raw values. |
| `src/femr/models/tasks/mtpp.py` (Q2) | `add_event`: every next occurrence of a numeric task is recorded; a value-less one gets value NaN (before: nothing recorded → labelled censored at end of record, and later valued occurrences hidden). `_create_time_value_bins`: assert relaxed to "value ⇒ time". Affected 176 MIMIC numeric codes with value-less events (6.0M occurrences, 5.1% of their events; 38 codes > 50% value-less). |
| `src/femr/models/architecture/embedding.py` (Q2) | `MTPPSharedTaskHead`: third loss case for events without a value bin, `−log Σ_v p(t*, v)` (`L_num_time_only`). Other MTPP heads (`MTPPTaskHead`, factorized, value) are unchanged and would still drop these rows; the scripts do not use them. |
| `tests/test_stat_utils.py`, `tests/models/test_mtpp.py`, `tests/models/test_tokenizer.py` | New: sampler is uniform and respects weights through `combine`; banned column gets no tokens and the DB is untouched; MTPP uses the raw count; a value-less next occurrence is an uncensored time-only event and its loss equals `−log` of the time bin summed over value bins. |

### Phase 5: two-step run flow (`src/femr/training/script/`)

| File | Change |
|---|---|
| `lib.sh` | `bash <arch>.sh tasks [prepare args]`: motor `--tasks_only` first (trains the shared tokenizer), then, once `OUT_ROOT/motor/tokenizer` exists, the other five in parallel; console logs in `OUT_ROOT/tasks_logs/`. `bash <arch>.sh <model> train [pretrain args]`: refuses to start unless the shared tokenizer and `<model>_task.pkl` exist, then builds batches (reusing both) and pretrains. `run` logs end time and exit code. |
| `env/*.sh` | `BANNED_PROPERTIES` per dataset (moved out of `lib.sh`). `cumc_cumc.sh`: new metadata/model paths. |
| `check_tokenization.py` (moved from `mimic/gsb/`) | Read-only check for any dataset: tokenizer copies byte-identical; vocabulary by type, numeric bins per column (count and edge quantiles), text entries per column; task code sets (MOTOR = TPP, MTPP vs TPP, NTP = SurvivEHR, ICareFM vs MTPP numeric); each differing code with event counts by kind, `[ancestor-only]` tag under an ontology, and the filter message of each model's log. `--no_event_counts` skips the slow pass. |
| `mimic/gsb/prepare_all.sh` | Removed (replaced by `tasks`). |
| `env/*.sh`, `lib.sh`, launchers (later the same day) | `GPUS` = GPU ids (1, 2 or 4; defaults to `CUDA_VISIBLE_DEVICES`, else `0,1,2,3`) replaces `NUM_GPUS`; their number sets processes and gradient accumulation; multi-GPU jobs get a free `--main_process_port` so several can share a node. `WANDB_PROJECT` per dataset (`ora_mimic`, `ora_cumc`). Run name `RUN_NAME`, default `<model>_<arch>`. `N_EPOCHS` (default 20) and `VALUE_BINS` (default 5) overridable like `TIME_BINS`. |
| `prepare.py` | An existing MOTOR/TPP/MTPP task whose time/value bins differ from `--time_bins`/`--value_bins` raises an error (it was reused silently). |
| `script/README.md` | Step-by-step guide for MIMIC on GSB and CUMC (`tasks` → `check_tokenization.py` → `<model> train`). |

### Documentation

`src/femr/training/tokenizer_and_targets.md`: tokenizer design, entropy filter, event tokenization,
target rules for every model, code-set relationships, MIMIC statistics.

## Verification

### Done

- Unit tests: `python -m pytest -q -p no:torchtyping tests --ignore=tests/test_survivehr_causal.py`
  → **21 passed** (11 existing + 10 new). `tests/test_survivehr_causal.py` already failed before these
  changes (imports the non-existent `femr.models.tasks.survivehr_causal`). `-p no:torchtyping` is
  needed because the installed torchtyping plugin crashes with this torch version.
- `bash -n` on all new scripts; invalid model name prints the usage line.

### MIMIC on GSB: Stage A (tokenizer + task pickles, no batches)

v1 (18:27–19:11, before the Phase 4 fixes) passed all checks except the bugs fixed in Phase 4; its
outputs are kept for reference in `ora/models/_stageA_v1/` (tokenizer, task pickles, logs, check output).
v1 task sets: MOTOR = TPP = 6,270; MTPP 7,223 (1,171 numeric); NTP = SurvivEHR events = 8,390;
ICareFM = SurvivEHR values = 1,081. Peak memory ~430 GB with five 32-thread jobs (~108 GB per job,
mostly the ontology in each worker).

v2 (current, with the Phase 4 fixes) in `/shared/share_mala/zilin/storage/ora/`:

| Path | Status |
|---|---|
| `metadata/tokenizer` | v2: 13,098 entries = 8,390 code + 4,026 text + 682 numeric (`numeric_value` 468 bins, was 169; `order_id`/`link_order_id`/`emar_seq` 68 each) |
| `models/<model>/tokenizer` | byte-identical copies, all six models |
| `models/{ntp,icarefm,survivehr,motor}/*_task.pkl` | v2, done 22:30–22:48 |
| `models/{tpp,mtpp}/*_task.pkl` | v2, done 23:32 (TPP) and 01:11 (MTPP, after the merge fix below) |
| batches | none yet |

Check results v2 (`ora/models/check_tokenization_v2.txt`): all tokenizer copies identical; MOTOR = TPP =
6,270; MTPP 7,129 (1,079 numeric + 6,050 non-numeric), MTPP non-numeric ⊆ TPP, MTPP adds 917 numeric codes
and drops 58 TPP codes (codes with some numeric values but < 1000 raw values; v1 dropped 42: 14 of the 16
new ones were v1 numeric tasks passing only through the label-position count, 2, e.g. `LAB//50920//UNK`
with 1 value, were v1 non-numeric because their value never showed up at a label position; kept as the
user's rule); NTP = SurvivEHR events = 8,390; ICareFM = SurvivEHR values = 1,081 ⊇ MTPP numeric (all
1,079; the other 2, `INFUSION_START/END//225151`, fail MTPP's `frac_events < 1/1000`); 162 MTPP numeric
codes also have value-less events (time-only events, Q2).

Bugs found during the v2 run (fixed, with tests):
- `mtpp._prefit_motor_agg` (Q1 change) assigned into a tuple (`first[3] += ...`) → `TypeError` in the
  merge. meds_reader then hangs instead of failing (its shutdown joins workers still sending results), so
  the run sat at 0% CPU without a traceback. Fixed with `np.add(..., out=...)`; test merges shard results.
- `prepare.py`: `fail_fast(database)` wraps the meds_reader block: a dead worker or an exception in the
  parent now prints the error, kills the workers and exits 1 (tested; without it a dead worker hangs).
- Batch path smoke test (300 random MIMIC train subjects, real v2 tokenizer and tasks,
  `convert_dataset(tokens_per_batch=16384, min_subjects_per_batch=2)`, then `collate` + task `cleanup` on
  5 batches each): all six models passed (15–16 batches each, windows ≤ 8,192, batches ≤ 16,384 tokens).
  ~10 min per model, almost all in the single-threaded HF fingerprinting of the batch generator.

### Next steps

1. MIMIC (GSB): `python script/check_tokenization.py --out_root ... --meta_dir ... --meds_reader ...`;
   then per model `CUDA_VISIBLE_DEVICES=.. NUM_GPUS=.. NUM_THREADS=.. bash script/mimic/gsb/transformer.sh <model> train`.
2. CUMC (cumc cluster): `/data/models/zj2398/femr/metadata` holds `ontology.pkl` and `main_split.csv`;
   `models/` is empty. Run `bash script/cumc/transformer.sh tasks`, then `<model> train` per model as
   its task pickle appears. Note: `prepare.py` silently reuses an existing `<model>_task.pkl`, and stops
   if a model's `tokenizer/` copy differs from the shared one.
3. Mamba batches later with the same commands via `mamba.sh`.

## How to review

```bash
git status --short && git diff -M HEAD --stat
git diff -M HEAD -- src/femr/models src/femr/training/prepare.py src/femr/training/pretrain.py
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate tte
python -m pytest -q -p no:torchtyping tests --ignore=tests/test_survivehr_causal.py
```

Unrelated working-tree changes that were present before this work and are not part of it:
deleted `src/femr/icarefm_new.pdf`, `src/femr/media-1.pdf`, and `script/mimic/run_mamba.sh` having
been moved by the user to `script/mimic/mamba/` (now `script/legacy/mimic/mamba/`).

## Appendix: building the smoke-test subset

```bash
python -I make_subset.py <dir>/meds && meds_reader_convert <dir>/meds <dir>/reader --num_threads 8
# then: export MEDS_READER=<dir>/reader META_DIR=<dir>/meta OUT_ROOT=<dir>/models HF_CACHE=<dir>/hf NUM_GPUS=1
# with <dir>/meta/ontology.pkl -> /shared/share_mala/zilin/storage/ora/metadata/ontology.pkl
```

`make_subset.py`:

```python
"""Export ~2000 MIMIC subjects from the GSB meds_reader database into a small MEDS dataset."""
import json, shutil, sys, pathlib
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, meds_reader

src = pathlib.Path("/shared/share_mala/zilin/storage/ora/meds_v0.6_reader")
out = pathlib.Path(sys.argv[1])
(out / "data").mkdir(parents=True, exist_ok=True); (out / "metadata").mkdir(exist_ok=True)

splits = pd.read_parquet(src / "metadata/subject_splits.parquet")
rng = np.random.default_rng(0)
train = splits[splits.split != "held_out"].subject_id.to_numpy()
held = splits[splits.split == "held_out"].subject_id.to_numpy()
chosen = np.concatenate([rng.choice(train, 1800, replace=False), rng.choice(held, 200, replace=False)])

db = meds_reader.SubjectDatabase(str(src))
props = dict(db.properties)
cols = {name: [] for name in ["subject_id", *props]}
for sid in chosen:
    for e in db[int(sid)].events:
        cols["subject_id"].append(int(sid))
        for name in props:
            cols[name].append(getattr(e, name, None))
schema = pa.schema([("subject_id", pa.int64())] + [(n, t) for n, t in props.items()])
pq.write_table(pa.table(cols, schema=schema), out / "data/0.parquet")
splits[splits.subject_id.isin(chosen)].to_parquet(out / "metadata/subject_splits.parquet", index=False)
shutil.copy(src / "metadata/codes.parquet", out / "metadata/codes.parquet")
shutil.copy(src / "metadata/dataset.json", out / "metadata/dataset.json")
print("events", len(cols["subject_id"]), "properties", list(props))
```
