"""
Bootstrap R^2 hypothesis tests for regression models.

This script compares ORA (transformer_mtpp) against every baseline listed in
BASELINES below. Adding a model is a one-line change there.


Why the alignment code is more explicit than before:
- In these regression cohorts, some test rows share the same
  (subject_id, prediction_time) but have different y_true values.
- Because of those duplicate keys, simply sorting by subject_id and
  prediction_time is not enough to guarantee row-level alignment.
- We therefore use the cohort test labels as the reference and normalize both
  labels and prediction files with the same stable sort:
      subject_id, prediction_time, y_true

That normalization makes two things possible:
1. We can verify that ntp, motor, mtpp, and imputation all correspond to the
   same cohort test rows.
2. We can run the bootstrap comparison on correctly aligned y_true / y_pred
   pairs even when duplicate keys are present.

example: you can directly run python regression_test.py,

or   python /user/zj2398/femr_chao_meds_v3/src/femr/evaluation/regression/baseline/baseline.py \
    --cohort_dir /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k \
    --meds_path /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --split_path /user/zj2398/cache/mtpp_8k_T_8_V_10/main_split.csv \
    --dataset mimic \
    --output_dir /shared/share_mala/zj2398/mimic/task/regression
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

# ── Config ──
DEFAULT_TASK_LIST = ["platelets", "pao2", "glucose", "creatinine"]
DEFAULT_REGRESSION_BASE = Path("/shared/share_mala/zj2398/mimic/task/regression")
DEFAULT_COHORT_DIR = DEFAULT_REGRESSION_BASE / "cohort" / "icu_stay_4h_100k"
DEFAULT_SPLIT_PATH = Path("/user/zj2398/cache/mtpp_8k_T_8_V_10/main_split.csv")

# Directory name of the model under test, and the baselines it is compared to.
# All entries are the MIMIC transformer runs; `icarefm` is the transformer run
# despite the name (the mamba one is `icare_mamba`).
ORA_DIR = "transformer_mtpp"
BASELINES = [
    ("imputation", "MOST RECENT"),
    ("transformer_ntp", "NTP"),
    ("motor", "MOTOR"),
    ("icarefm", "ICARE-FM"),
    ("transformer_survivehr", "SurviveHR"),
]

N_BOOTSTRAP = 10000
SEED = 42
N_JOBS = 32  # number of parallel workers


def parse_args():
    """Expose the main paths and bootstrap settings as CLI arguments."""
    parser = argparse.ArgumentParser(description="Bootstrap R^2 tests for regression models.")
    parser.add_argument("--regression_base", type=Path, default=DEFAULT_REGRESSION_BASE)
    parser.add_argument("--cohort_dir", type=Path, default=DEFAULT_COHORT_DIR)
    parser.add_argument("--split_path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASK_LIST)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_jobs", type=int, default=N_JOBS)
    return parser.parse_args()


def load_test_labels(task: str, cohort_dir: Path, split_path: Path) -> pd.DataFrame:
    """
    Load the cohort test labels for one task.

    This is the reference table that every model prediction file must match.
    We intentionally keep all rows, including duplicate key rows.
    """
    split_df = pd.read_csv(split_path).rename(columns={"split_name": "split"})
    labels = pd.read_parquet(cohort_dir / task / f"{task}.parquet")
    labels = labels.merge(split_df[["subject_id", "split"]], on="subject_id", how="inner")
    labels = labels[labels["split"] == "test"][["subject_id", "prediction_time", "numerical_value"]].copy()
    labels = labels.rename(columns={"numerical_value": "y_true"})
    return normalize_rows(labels[["subject_id", "prediction_time", "y_true"]])


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a label/prediction DataFrame into a comparable row order.

    The stable sort is important. Some tasks contain duplicate
    (subject_id, prediction_time) pairs, but within those duplicates the label
    values are distinct. Sorting by y_true as a third key gives us a consistent
    row order for both the cohort labels and the prediction files.
    """
    normalized = df.copy()
    normalized["subject_id"] = normalized["subject_id"].astype("int64")
    normalized["prediction_time"] = pd.to_datetime(normalized["prediction_time"])
    normalized["y_true"] = normalized["y_true"].astype(float)
    return normalized.sort_values(["subject_id", "prediction_time", "y_true"], kind="mergesort").reset_index(drop=True)


def load_predictions(path: Path) -> pd.DataFrame:
    """
    Load one model's saved regression predictions.

    Expected schema is the shared regression output format:
    subject_id, prediction_time, y_true, y_pred, residual(optional)
    """
    df = pd.read_parquet(path)
    expected_cols = {"subject_id", "prediction_time", "y_true", "y_pred"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    keep_cols = ["subject_id", "prediction_time", "y_true", "y_pred"]
    if "residual" in df.columns:
        keep_cols.append("residual")

    return normalize_rows(df[keep_cols])


def resolve_prediction_path(regression_base: Path, model_name: str, task: str) -> Path:
    """
    Resolve the on-disk prediction parquet for one model/task pair.

    The learned models store files under:
        <root>/<model>/results/<task>/<run_name>/test_predictions_50k.parquet

    The imputation baseline stores files under:
        <root>/imputation/results/<task>/test_predictions_50k.parquet

    Later runs (icarefm, transformer_survivehr) dropped the `_50k` suffix and
    write plain test_predictions.parquet, so we fall back to that name. The
    `_50k` file is preferred where both exist, because for the earlier models
    the two differ by a row or two and only the `_50k` row set matches the
    cohort labels. check_labels() verifies whichever file we end up loading.
    """
    if model_name == "imputation":
        path = regression_base / "imputation" / "results" / task / "test_predictions_50k.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {model_name}/{task}: {path}")
        return path

    task_dir = regression_base / model_name / "results" / task
    for filename in ("test_predictions_50k.parquet", "test_predictions.parquet"):
        matches = sorted(task_dir.glob(f"*/{filename}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"Expected one {filename} under {task_dir}, found {len(matches)}")
    raise FileNotFoundError(f"No test prediction parquet found under {task_dir}")


def check_labels(reference_labels, predictions_by_model):
    """
    Assert every prediction file matches the cohort reference rows exactly.

    We check three things for each model:
    1. Same number of rows as the cohort test labels.
    2. Same ordered subject_id / prediction_time pairs after normalization.
    3. Same ordered y_true values as the cohort labels.

    If any mismatch appears, we print a small preview to make debugging easier.
    """
    for col in ["subject_id", "prediction_time", "y_true"]:
        assert col in reference_labels.columns, f"Reference labels missing column '{col}'"
        assert all(col in df.columns for df in predictions_by_model.values()), \
            f"Column '{col}' missing in one of the prediction DataFrames"

    reference_subject_ids = reference_labels["subject_id"].to_numpy()
    reference_prediction_times = reference_labels["prediction_time"].to_numpy(dtype="datetime64[ns]")
    reference_y_true = reference_labels["y_true"].to_numpy()

    for name, df in predictions_by_model.items():
        assert len(df) == len(reference_labels), \
            f"Length mismatch for {name}: reference={len(reference_labels)}, {name}={len(df)}"
        assert np.array_equal(df["subject_id"].to_numpy(), reference_subject_ids), \
            f"subject_id mismatch between cohort labels and {name}"
        assert np.array_equal(df["prediction_time"].to_numpy(dtype="datetime64[ns]"), reference_prediction_times), \
            f"prediction_time mismatch between cohort labels and {name}"

        diff = np.abs(df["y_true"].to_numpy() - reference_y_true)
        max_diff = np.max(diff)
        print(f"  max |y_true| diff — cohort vs {name}: {max_diff:.2e}")
        mask = diff > 1e-3
        if mask.any():
            mismatch = pd.DataFrame({
                "subject_id": reference_subject_ids[mask],
                "prediction_time": reference_prediction_times[mask],
                "y_true_cohort": reference_y_true[mask],
                f"y_true_{name}": df["y_true"].to_numpy()[mask],
                "abs_diff": diff[mask],
            })
            print(f"  ⚠ {len(mismatch)} rows differ (atol>1e-3) between cohort and {name}:")
            print(mismatch.head(20).to_string(index=False))
            assert False, f"y_true mismatch between cohort and {name} (max diff={max_diff:.2e}, n_mismatch={len(mismatch)})"
    print(f"  ✓ Labels consistent across cohort and all models (n={len(reference_labels)})")



def _one_bootstrap(seed, labels, preds_a, preds_b):
    """Single bootstrap iteration — designed for joblib parallelism."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    idx = rng.randint(0, n, size=n)
    y = labels[idx]
    if np.var(y) == 0:
        return np.nan
    return r2_score(y, preds_a[idx]) - r2_score(y, preds_b[idx])


def bootstrap_r2_test(labels, preds_a, preds_b, n_bootstrap=N_BOOTSTRAP, seed=SEED, n_jobs=N_JOBS):
    """
    Bootstrap test: H0: R2_a <= R2_b, H1: R2_a > R2_b.
    Returns: observed_diff, ci_lower, ci_upper, p_value
    """
    observed_diff = r2_score(labels, preds_a) - r2_score(labels, preds_b)

    seeds = np.random.RandomState(seed).randint(0, 2**31, size=n_bootstrap)
    diffs = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap)(s, labels, preds_a, preds_b) for s in seeds
    ))

    diffs = diffs[~np.isnan(diffs)]
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    p_value = np.mean(diffs <= 0)
    return observed_diff, ci_lower, ci_upper, p_value


def main():
    args = parse_args()
    results = []

    # We treat ORA as the "model A" system and compare it against each
    # baseline using the same bootstrap test implementation.
    model_dirs = {"ora": ORA_DIR}
    model_dirs.update({name: directory for directory, name in BASELINES})

    for task in args.tasks:
        print(f"\nTask: {task}")

        # Load the cohort labels first. They define the row set that every
        # prediction file must match before any metric or hypothesis test is run.
        reference_labels = load_test_labels(task, args.cohort_dir, args.split_path)
        predictions = {
            model_name: load_predictions(resolve_prediction_path(args.regression_base, directory_name, task))
            for model_name, directory_name in model_dirs.items()
        }

        check_labels(reference_labels, predictions)

        labels = reference_labels["y_true"].to_numpy()
        ora_preds = predictions["ora"]["y_pred"].to_numpy()

        # Report single-split point estimates for context before the bootstrap
        # comparisons.
        r2_scores = {
            model_name: r2_score(labels, df["y_pred"].to_numpy())
            for model_name, df in predictions.items()
        }
        print(
            "  R² — "
            + ", ".join(f"{model_name}: {score:.4f}" for model_name, score in r2_scores.items())
        )

        for _, baseline_name in BASELINES:
            # Hypothesis test:
            #   H0: R2(ora) <= R2(baseline)
            #   H1: R2(ora) >  R2(baseline)
            diff, ci_lo, ci_hi, p = bootstrap_r2_test(
                labels,
                ora_preds,
                predictions[baseline_name]["y_pred"].to_numpy(),
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(
                f"  ORA vs {baseline_name}: Δ={diff:+.4f}, "
                f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}"
            )
            results.append(
                dict(
                    task=task,
                    baseline=baseline_name,
                    diff=diff,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    p=p,
                )
            )

    df_results = pd.DataFrame(results)

    # ── LaTeX table ──
    task_display = {
        "creatinine": "Creatinine",
        "platelets": "Platelets",
        "pao2": "PaO2",
        "glucose": "Glucose",
    }

    cols = " & ".join(task_display.get(task, task) for task in args.tasks)

    print("\n% ── LaTeX table (ORA minus baseline, 95% CI on ΔR²) ──")
    print("\\begin{tabular}{l" + "c" * len(args.tasks) + "}")
    print("\\toprule")
    print(f"Method & {cols} \\\\")
    print("\\midrule")
    for _, baseline_name in BASELINES:
        cells = []
        for task in args.tasks:
            row = df_results[(df_results.task == task) & (df_results.baseline == baseline_name)].iloc[0]
            sig = "*" if row.p < 0.05 else ""
            cells.append(f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]{sig}")
        print(f"{baseline_name} & {' & '.join(cells)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
