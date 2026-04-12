import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ── Config ──
task_list = ["in_hospital_mortality", "long_los", "readmission", "ami", "masld", "stroke"]
outcome_base = Path("/shared/share_mala/zj2398/mimic/task/patient_outcome_task/results")
phenotype_base = Path("/shared/share_mala/zj2398/mimic/task/phenotype_task/results")
N_BOOTSTRAP = 10000
SEED = 42
N_JOBS = 16  # number of parallel workers


def check_labels(mtpp, ntp, motor,xgboost):
    """Assert all three DataFrames have identical subject_id, prediction_time, boolean_value."""
    for col in ["subject_id", "prediction_time", "boolean_value"]:
        assert col in mtpp.columns and col in ntp.columns and col in motor.columns, \
            f"Column '{col}' missing in one of the DataFrames"
    assert len(mtpp) == len(ntp) == len(motor), \
        f"Length mismatch: mtpp={len(mtpp)}, ntp={len(ntp)}, motor={len(motor)}"
    assert (mtpp["subject_id"].values == ntp["subject_id"].values).all(), \
        "subject_id mismatch between mtpp and ntp"
    assert (mtpp["subject_id"].values == motor["subject_id"].values).all(), \
        "subject_id mismatch between mtpp and motor"
    assert (mtpp["subject_id"].values == xgboost["subject_id"].values).all(), \
        "subject_id mismatch between mtpp and motor"
      
    assert (mtpp["prediction_time"].values == ntp["prediction_time"].values).all(), \
        "prediction_time mismatch between mtpp and ntp"
    assert (mtpp["prediction_time"].values == motor["prediction_time"].values).all(), \
        "prediction_time mismatch between mtpp and motor"
    assert (mtpp["prediction_time"].values == xgboost["prediction_time"].values).all(), \
        "prediction_time mismatch between mtpp and motor"
    
    assert (mtpp["boolean_value"].values == ntp["boolean_value"].values).all(), \
        "boolean_value mismatch between mtpp and ntp"
    assert (mtpp["boolean_value"].values == motor["boolean_value"].values).all(), \
        "boolean_value mismatch between mtpp and motor"
    assert (mtpp["boolean_value"].values == xgboost["boolean_value"].values).all(), \
        "boolean_value mismatch between mtpp and motor"
    print(f"  ✓ Labels consistent across all 4 models (n={len(mtpp)})")


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


# ── Main ──
results = []

for task in task_list:
    print(f"\n{'='*60}\nTask: {task}\n{'='*60}")
    if task in ["in_hospital_mortality", "long_los","readmission"]:
        ntp = pd.read_parquet(outcome_base  / task / "transformer_ntp/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        motor = pd.read_parquet(outcome_base  / task / "motor/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        mtpp = pd.read_parquet(outcome_base  / task / "transformer_mtpp/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        xgboost = pd.read_parquet(outcome_base  / task / f"xgboost/{task}_xgboost_predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    else:
        ntp = pd.read_parquet(phenotype_base  / task / "transformer_ntp_730/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        motor = pd.read_parquet(phenotype_base / task / "motor_730/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        mtpp = pd.read_parquet(phenotype_base  / task / "transformer_mtpp_730/test_predictions/predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        xgboost = pd.read_parquet(phenotype_base  / task / f"xgboost/{task}_xgboost_predictions.parquet") \
            .sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)

    check_labels(mtpp, ntp, motor,xgboost)

    labels = mtpp["boolean_value"].values.astype(int)
    mtpp_preds = mtpp["predicted_boolean_probability"].values  # adjust column name if needed
    ntp_preds = ntp["predicted_boolean_probability"].values
    motor_preds = motor["predicted_boolean_probability"].values
    xgboost_preds = xgboost["predicted_boolean_probability"].values

    # Point AUROCs
    auc_mtpp = roc_auc_score(labels, mtpp_preds)
    auc_ntp = roc_auc_score(labels, ntp_preds)
    auc_motor = roc_auc_score(labels, motor_preds)
    auc_xgboost = roc_auc_score(labels, xgboost_preds)
    print(f"  AUROC — mtpp: {auc_mtpp:.4f}, ntp: {auc_ntp:.4f}, motor: {auc_motor:.4f}")

    # # MTPP vs ntp
    # diff, ci_lo, ci_hi, p = bootstrap_auroc_test(labels, mtpp_preds, ntp_preds)
    # sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    # print(f"  MTPP vs ntp: Δ={diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
    # results.append(dict(task=task, comparison="mtpp_vs_ntp", diff=diff, ci_lo=ci_lo, ci_hi=ci_hi, p=p))

    # # MTPP vs MOTOR
    # diff, ci_lo, ci_hi, p = bootstrap_auroc_test(labels, mtpp_preds, motor_preds)
    # sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    # print(f"  MTPP vs MOTOR: Δ={diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
    # results.append(dict(task=task, comparison="mtpp_vs_motor", diff=diff, ci_lo=ci_lo, ci_hi=ci_hi, p=p))

    # # MTPP vs xgboost
    diff, ci_lo, ci_hi, p = bootstrap_auroc_test(labels, mtpp_preds, xgboost_preds)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  MTPP vs xgboost Δ={diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
    results.append(dict(task=task, comparison="mtpp_vs_xgboost", diff=diff, ci_lo=ci_lo, ci_hi=ci_hi, p=p))

# ── Summary table ──
df_results = pd.DataFrame(results)

# ── LaTeX table ──
task_display = {
    "in_hospital_mortality": "In-Hosp. Mortality",
    "long_los": "Long LOS",
    "readmission": "Readmission",
    "ami": "AMI",
    "masld": "MASLD",
    "stroke": "Stroke",
}
# baseline_display = {"mtpp_vs_ntp": "ntp", "mtpp_vs_motor": "MOTOR"}
baseline_display = {"mtpp_vs_xgboost":"xgboost"}

cols = " & ".join(task_display[t] for t in task_list)
header = f"Method & {cols} \\\\"

rows_latex = []
for comp, label in baseline_display.items():
    cells = []
    for task in task_list:
        row = df_results[(df_results.task == task) & (df_results.comparison == comp)].iloc[0]
        sig = "*" if row.p < 0.05 else ""
        cells.append(f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]{sig}")
    rows_latex.append(f"{label} & {' & '.join(cells)} \\\\")

print("\n% ── LaTeX table (MTPP minus baseline, 95% CI) ──")
print("\\begin{tabular}{l" + "c" * len(task_list) + "}")
print("\\toprule")
print(header)
print("\\midrule")
for r in rows_latex:
    print(r)
print("\\bottomrule")
print("\\end{tabular}")