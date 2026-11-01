"""
Bootstrap AUROC hypothesis tests for the binary classification tasks.

ORA (transformer_mtpp) is compared against every baseline in BASELINES using the
same bootstrap resamples, and the result is printed as a LaTeX table of
95% CIs on the AUROC difference (ORA minus baseline).

Run with: python classification_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ── Config ──
OUTCOME_TASKS = ["in_hospital_mortality", "long_los", "readmission"]
PHENOTYPE_TASKS = ["ami", "masld", "stroke"]
TASK_LIST = OUTCOME_TASKS + PHENOTYPE_TASKS

OUTCOME_BASE = Path("/shared/share_mala/zj2398/mimic/task/patient_outcome_task/results")
PHENOTYPE_BASE = Path("/shared/share_mala/zj2398/mimic/task/phenotype_task/results")

# Directory name of the model under test, and the baselines it is compared to.
# Adding a model is a single line here: (directory name, display name).
#
# All entries are the MIMIC transformer runs. Note `icarefm` lacks a
# `transformer_` prefix but is the transformer one (cache/icarefm/transformer);
# the mamba run lives in `icare_mamba`, and `transformer_icarefm` is CUMC.
# Likewise use `transformer_survivehr`, not `survivehr_mamba`.
ORA_DIR = "transformer_mtpp"
BASELINES = [
    ("xgboost", "XGBOOST"),
    ("transformer_ntp", "NTP"),
    ("motor", "MOTOR"),
    ("icarefm", "ICARE-FM"),
    ("transformer_survivehr", "SurviveHR"),
]

N_BOOTSTRAP = 10000
SEED = 42
N_JOBS = 16  # number of parallel workers


def resolve_prediction_path(task: str, model_dir: str) -> Path:
    """
    Resolve the prediction parquet for one model/task pair.

    Two naming conventions have to be handled:
    - Phenotype runs of the learned models carry a `_730` suffix on the run
      directory (e.g. motor_730); the outcome runs do not. XGBoost is the
      exception — its directory is plain `xgboost` in both cohorts.
    - The learned models write test_predictions/predictions.parquet, while
      XGBoost writes <task>_xgboost_predictions.parquet directly.
    """
    is_phenotype = task in PHENOTYPE_TASKS
    base = PHENOTYPE_BASE if is_phenotype else OUTCOME_BASE

    if model_dir == "xgboost":
        path = base / task / "xgboost" / f"{task}_xgboost_predictions.parquet"
    else:
        run_dir = f"{model_dir}_730" if is_phenotype else model_dir
        path = base / task / run_dir / "test_predictions" / "predictions.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_dir}/{task}: {path}")
    return path


def load_predictions(path: Path) -> pd.DataFrame:
    """Load one model's predictions in a deterministic row order."""
    return (
        pd.read_parquet(path)
        .sort_values(["subject_id", "prediction_time"])
        .reset_index(drop=True)
    )


def check_labels(predictions_by_model):
    """
    Assert every model's prediction file covers the same test rows in the same
    order, so the bootstrap resamples are paired across models.

    The first model in the dict is used as the reference.
    """
    reference_name, reference = next(iter(predictions_by_model.items()))
    for name, df in predictions_by_model.items():
        for col in ["subject_id", "prediction_time", "boolean_value"]:
            assert col in df.columns, f"Column '{col}' missing in {name}"
        assert len(df) == len(reference), \
            f"Length mismatch: {reference_name}={len(reference)}, {name}={len(df)}"
        for col in ["subject_id", "prediction_time", "boolean_value"]:
            assert (df[col].values == reference[col].values).all(), \
                f"{col} mismatch between {reference_name} and {name}"
    print(f"  ✓ Labels consistent across all {len(predictions_by_model)} models (n={len(reference)})")


def _one_bootstrap(seed, labels, preds_a, preds_b):
    """Single bootstrap iteration — designed for joblib parallelism."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    idx = rng.randint(0, n, size=n)
    y = labels[idx]
    if y.sum() == 0 or y.sum() == n:
        return np.nan
    return roc_auc_score(y, preds_a[idx]) - roc_auc_score(y, preds_b[idx])


def bootstrap_auroc_test(labels, preds_a, preds_b, n_bootstrap=N_BOOTSTRAP, seed=SEED, n_jobs=N_JOBS):
    """
    Bootstrap test: H0: AUROC_a <= AUROC_b, H1: AUROC_a > AUROC_b.
    Returns: observed_diff, ci_lower, ci_upper, p_value
    """
    observed_diff = roc_auc_score(labels, preds_a) - roc_auc_score(labels, preds_b)

    seeds = np.random.RandomState(seed).randint(0, 2**31, size=n_bootstrap)
    diffs = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap)(s, labels, preds_a, preds_b) for s in seeds
    ))

    diffs = diffs[~np.isnan(diffs)]
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    p_value = np.mean(diffs <= 0)
    return observed_diff, ci_lower, ci_upper, p_value


def main():
    results = []

    for task in TASK_LIST:
        print(f"\n{'='*60}\nTask: {task}\n{'='*60}")

        # ORA first so it is the reference row set in check_labels.
        model_dirs = [("ora", ORA_DIR)] + [(name, directory) for directory, name in BASELINES]
        predictions = {
            name: load_predictions(resolve_prediction_path(task, directory))
            for name, directory in model_dirs
        }
        check_labels(predictions)

        labels = predictions["ora"]["boolean_value"].values.astype(int)
        preds = {
            name: df["predicted_boolean_probability"].values
            for name, df in predictions.items()
        }

        aurocs = {name: roc_auc_score(labels, p) for name, p in preds.items()}
        print("  AUROC — " + ", ".join(f"{name}: {auc:.4f}" for name, auc in aurocs.items()))

        for _, baseline_name in BASELINES:
            diff, ci_lo, ci_hi, p = bootstrap_auroc_test(labels, preds["ora"], preds[baseline_name])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  ORA vs {baseline_name}: Δ={diff:+.4f}, "
                  f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
            results.append(dict(task=task, baseline=baseline_name, diff=diff,
                                ci_lo=ci_lo, ci_hi=ci_hi, p=p))

    df_results = pd.DataFrame(results)

    # ── LaTeX table ──
    task_display = {
        "in_hospital_mortality": "Mortality",
        "long_los": "Long LOS",
        "readmission": "Readmission",
        "ami": "AMI",
        "masld": "MASLD",
        "stroke": "Stroke",
    }

    print("\n% ── LaTeX table (ORA minus baseline, 95% CI on ΔAUROC) ──")
    print("\\begin{tabular}{l" + "c" * len(TASK_LIST) + "}")
    print("\\toprule")
    print("Method & " + " & ".join(task_display[t] for t in TASK_LIST) + " \\\\")
    print("\\midrule")
    for _, baseline_name in BASELINES:
        cells = []
        for task in TASK_LIST:
            row = df_results[(df_results.task == task) & (df_results.baseline == baseline_name)].iloc[0]
            sig = "*" if row.p < 0.05 else ""
            cells.append(f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]{sig}")
        print(f"{baseline_name} & {' & '.join(cells)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
