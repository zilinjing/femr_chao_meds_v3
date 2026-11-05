# Value-aware tabular baseline (MEDS-Tab aggregation set)

Answers Reviewer 3's objection that the submitted XGBoost / DeepHit baselines discard laboratory
values. Full analysis: [`rebuttal/reviewer3/Q1.md`](../../../../../../../rebuttal/reviewer3/Q1.md).

## Why this exists

The submitted baseline builds features with `femr.featurizers.CountFeaturizer` under default flags
(`numeric_value_decile=False`). Under those defaults, **every event carrying a numeric value is dropped
from the feature matrix entirely** — not only the value, but the occurrence count — because
`add_preprocess_data` routes numeric events into an `elif` branch that does nothing when the flag is
off, so the code never enters `observed_codes` and no column is allocated. XGBoost and DeepHit never
learn that a creatinine was even ordered.

This package restores that information by reproducing the **MEDS-Tab** (arXiv:2411.00200) aggregation
set exactly:

```
code/count, value/count, value/sum, value/sum_sqd, value/min, value/max
```

over nested look-back windows `[7d, full]`. 12 columns per numeric code. No `last`, no recency, no
derived mean/std, no per-code standardization — the feature specification is the published one, so the
strengthened baseline is not ours to defend.

**MIMIC-IV only.** Paths point at the current cluster:

| | |
|---|---|
| MEDS reader | `/data/zilinjing/ethos_fm/storage/ora/meds_v0.6_reader` |
| Cohorts | `storage/ora/cohort/{outcome/cohort_100k,phenotype/binary,phenotype/tte}/<task>/` |
| Features out | `storage/ora/meds_tab/<task>[_binary]/` |
| Results out | `storage/ora/result/meds_tab/` |
| Splits | `storage/ora/metadata/main_split.csv` |

## Files

| File | Purpose |
|---|---|
| `meds_tab_featurizer.py` | `MedsTabNumericFeaturizer` — the FEMR `Featurizer` |
| `generate_tabular_features.py` | Feature extraction entry point (adds the new featurizer to the list) |
| `extract.sh` | All 9 MIMIC-IV cohorts (3 TTE + 3 phenotype-binary + 3 outcome) |
| `validate_featurizer.py` | Brute-force correctness checks — **run this first** |
| `train_utils.py` | Copy of `baseline/train_utils.py` + widened grids + sparse handling |
| `run_all.py` | Driver; sparse for XGBoost, dense+scaled for DeepHit |
| `collect_compute.py` | Merges per-task runtime/memory JSON into an appendix table |

## Run order

**Environments.** Extraction needs only `meds_reader` + `femr`, both present in
`/data/zilinjing/ethos_fm/envs/tte` (which `extract.sh` uses by default; override with `PYTHON=...`).
The `femr_ehr` env the original `baseline/extract.sh` activates does not exist on this cluster.
**`run_all.py` additionally needs `xgboost` and `scikit-survival`, which are NOT installed in `tte`** —
`pip install xgboost scikit-survival` before step 3.

```bash
cd .../evaluation/baseline/meds_tab

# 1. Validate against a brute-force reference on real data (minutes). Do not skip.
python -m femr.evaluation.baseline.meds_tab.validate_featurizer --num_subjects 200

# 2. Extract features for all 9 cohorts (3 TTE + 3 phenotype-binary + 3 outcome)
bash extract.sh

# 3. Train: DeepHit for phenotype TTE, XGBoost for all binary tasks
python run_all.py --n_jobs 24

# 4. Compute table for the appendix
python collect_compute.py --csv compute.csv
```

`run_all.py` uses `from train_utils import ...`, so run it as a script from inside this directory
(same as the original), not with `-m`.

## What differs from `evaluation/baseline/count/`

Nothing about the cohorts, splits, metrics or bootstrap protocol. The changes are:

1. **Features.** `MedsTabNumericFeaturizer` added to the featurizer list. `CountFeaturizer` is
   unchanged, so non-numeric codes are featurized exactly as before.
2. **Output paths.** Repointed at the current cluster (`storage/ora/meds_tab/`,
   `storage/ora/result/meds_tab/`). The submitted artifacts are untouched, so both baseline rows can
   be reported in the revision. Train/tuning/hyper resampling uses the same seeds as the submitted
   baseline, so the two rows are directly comparable.
3. **Sparse loading.** `load_features_labels` aligns rows through the MultiIndex and row-slices the CSR
   matrix instead of calling `.todense()`. The value-aware matrix is far wider and the original path
   would risk OOM. XGBoost consumes CSR directly (trees are invariant to the dropped
   `StandardScaler`); DeepHit still gets dense, scaled input with all-zero training columns removed.
4. **Hyperparameter search.** The submitted tabular baseline was tuned far more weakly than the
   foundation-model probes it is compared against. Both arms now match or exceed them:

   | | FM probe | Submitted baseline | Here |
   |---|---|---|---|
   | DeepHit architecture | `[]` (linear, hard-coded) | `[]`, `[50]`, `[50,50]` | `[]`, `[50]`, `[50,50]` |
   | DeepHit lr | {1e-3, 1e-4} | **{1e-4}** | {1e-3, 1e-4} |
   | DeepHit batch | {1024, 2048} | **{1024}** | {1024, 2048} |
   | DeepHit epochs | 1000 | **100** | 1000 |
   | DeepHit configs | 8 | 6 | 24 |
   | Binary probe | `LogisticRegressionCV` (10 C values, 5-fold) | XGBoost, **2 configs** | XGBoost, 24 configs + early stopping |

   The FM probes are deliberately linear — that is what linear probing means, and it should stay.
   The handicap was on the tabular side: a single learning rate and 10x fewer epochs than the probe
   it is measured against. Sources: `archive/tte_evaluation/tte_grid_search_filter_new_phenotype.py:79-97`
   and `finetune_motor.py:115`.
5. **Compute accounting.** Extraction writes `<task>_extraction_stats.json` (wall-clock, peak RSS,
   column budget, density); training writes `<task>_<model>_compute.json` (load time, train+tune time,
   peak RSS, `n_jobs`). These answer the reviewer's question about baseline cost.
6. **Bugs fixed.**
   - The `continue` that skipped the binary arm of the phenotype loop is removed, so every phenotype
     yields both a TTE and a binary result.
   - `evaluation/baseline/count/generate_tabular_features.py` imported
     `.motor_evaluation.generate_labels`, a module path that no longer exists — it raised
     `ModuleNotFoundError` on any run. Fixed in place to `.evaluation.generate_labels`; this package
     imports the shared parser from there rather than redeclaring one.
   - Cohort loading prefers `train/tuning/held_out.parquet` over a recursive glob. The TTE cohort
     directories also contain a whole-cohort parquet that is the union of the three splits
     (`cohort/phenotype/tte/ami/ami.parquet` = 118,627 = 94,672 + 11,546 + 12,409), so a plain glob
     would have read every TTE label twice.

## Verification status

All green as of the last run.

| Test | Result |
|---|---|
| Unit test vs brute-force reference (stubbed deps), 4 window configs, edge cases | pass — untimed events, empty subjects, tail loop, labels before all events, event at exactly the prediction time |
| `validate_featurizer.py` on real MIMIC (50 subjects, 249 labels) | pass — **66,180 (code, window) cells** matched brute force; column budget and nested-window monotonicity OK |
| End-to-end extraction on a 360-subject AMI cohort | pass — 1,809 labels, 24,213 columns, density 6.7%, 191 s, peak RSS 0.19 GB |
| Sparse loader: MultiIndex alignment, split partition, densify path | pass — 1,809/1,809 labels matched, splits partition exactly, dense float32 all finite |
| Cohort splits vs `main_split.csv` | consistent — cohort `train`+`tuning` ⊂ main-split `train`, cohort `held_out` = main-split `test` |

**Column budget.** On that cohort the numeric featurizer roughly doubles the matrix:
`AgeFeaturizer=1`, `CountFeaturizer=11,468`, `MedsTabNumericFeaturizer=12,744` (1,062 numeric codes
× 6 aggs × 2 windows). Preprocessing dominates runtime (188 s vs 3 s to featurize), so budget the
extraction sweep accordingly and run it detached.

## Event composition — the number to quote in the rebuttal

Measured on MIMIC-IV via `validate_featurizer.py`:

| Event kind | Share | Seen by submitted baseline | Seen after this fix |
|---|---|---|---|
| numeric value | **~47%** | no | **yes** |
| text value | ~34% | no | no |
| neither ("plain" codes) | ~19% | yes | yes |

So the submitted tabular baseline was built from roughly **one fifth of the record**. Re-run with a
larger `--num_subjects` before quoting an exact figure in the reply.

## Why text-valued events stay excluded

`string_value_combination` remains `False`. This is now an evidence-backed choice, not just parity:

* **MEDS-Tab has no text aggregation.** Its agg set is numeric-only, so excluding text *is* the
  published specification.
* **The high-volume text codes are numbers stored as strings.** `Blood Pressure` = `"100/48"` (1,569
  distinct values in 300 subjects), `Weight (Lbs)` = `"100.5"` (907 distinct), `BMI (kg/m2)` (295).
  One-hot encoding these is meaningless and would add **4,402 columns from 300 subjects alone**,
  growing without bound as the cohort grows.
* **The one regression-target overlap is a red herring.** `LAB//50931//mg/dL` (glucose) appears with a
  text value, but that value is `'___'` — the de-identification placeholder, i.e. missing data.

**The honest caveat**, which belongs in the reply: a minority of text codes *are* genuinely categorical
and clinically meaningful — heart rhythm (`LAB//220048`: "AF (Atrial Fibrillation)", "A Paced", …),
ectopy type, pain level. These remain excluded from the tabular baseline. One sentence:
*"Following the MEDS-Tab specification, the tabular baseline aggregates numeric measurements only;
categorical result codes remain excluded."*
