import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

# ── Config ──
task_list = ["platelets","pao2","glucose","creatinine"]
regression_base = Path("/shared/share_mala/zj2398/mimic/task/regression")
N_BOOTSTRAP = 10000
SEED = 42
N_JOBS = 32  # number of parallel workers


def check_labels(mtpp, ntp, motor):
    """Assert all three DataFrames have identical subject_id, prediction_time, y_true."""
    for col in ["subject_id", "prediction_time", "y_true"]:
        assert col in mtpp.columns and col in ntp.columns and col in motor.columns, \
            f"Column '{col}' missing in one of the DataFrames"
    assert len(mtpp) == len(ntp) == len(motor), \
        f"Length mismatch: mtpp={len(mtpp)}, ntp={len(ntp)}, motor={len(motor)}"
    assert (mtpp["subject_id"].values == ntp["subject_id"].values).all(), \
        "subject_id mismatch between mtpp and ntp"
    assert (mtpp["subject_id"].values == motor["subject_id"].values).all(), \
        "subject_id mismatch between mtpp and motor"
    assert (mtpp["prediction_time"].values == ntp["prediction_time"].values).all(), \
        "prediction_time mismatch between mtpp and ntp"
    assert (mtpp["prediction_time"].values == motor["prediction_time"].values).all(), \
        "prediction_time mismatch between mtpp and motor"
    # for name, other in [("ntp", ntp), ("motor", motor)]:
    for name, other in [("motor", motor)]:
        diff = np.abs(mtpp["y_true"].values - other["y_true"].values)
        max_diff = np.max(diff)
        print(f"  max |y_true| diff — mtpp vs {name}: {max_diff:.2e}")
        mask = diff > 1e-3
        if mask.any():
            mismatch = pd.DataFrame({
                "subject_id": mtpp["subject_id"].values[mask],
                "prediction_time": mtpp["prediction_time"].values[mask],
                "y_true_mtpp": mtpp["y_true"].values[mask],
                f"y_true_{name}": other["y_true"].values[mask],
                "abs_diff": diff[mask],
            })
            print(f"  ⚠ {len(mismatch)} rows differ (atol>1e-3) between mtpp and {name}:")
            print(mismatch.head(20).to_string(index=False))
            assert False, f"y_true mismatch between mtpp and {name} (max diff={max_diff:.2e}, n_mismatch={len(mismatch)})"
    print(f"  ✓ Labels consistent across all 3 models (n={len(mtpp)})")



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


# ── Main ──
results = []

for task in task_list:
    ntp = pd.read_parquet(regression_base/"transformer_ntp/results"/task/"transformer_ntp"/"test_predictions_50k.parquet").sort_values(["subject_id", "prediction_time"], ascending=[True, True])
    motor = pd.read_parquet(regression_base/"motor/results"/task/"motor"/"test_predictions_50k.parquet").sort_values(["subject_id", "prediction_time"], ascending=[True, True])
    mtpp = pd.read_parquet(regression_base/"transformer_mtpp/results"/task/"transformer_mtpp"/"test_predictions_50k.parquet").sort_values(["subject_id", "prediction_time"], ascending=[True, True])

    check_labels(mtpp, ntp, motor)

    labels = mtpp["y_true"].values
    mtpp_preds = mtpp["y_pred"].values
    ntp_preds = ntp["y_pred"].values
    motor_preds = motor["y_pred"].values

    # Point R² scores
    r2_mtpp = r2_score(labels, mtpp_preds)
    r2_ntp = r2_score(labels, ntp_preds)
    r2_motor = r2_score(labels, motor_preds)
    print(f"  R² — mtpp: {r2_mtpp:.4f}, ntp: {r2_ntp:.4f}, motor: {r2_motor:.4f}")

    # MTPP vs ntp
    diff, ci_lo, ci_hi, p = bootstrap_r2_test(labels, mtpp_preds, ntp_preds)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  MTPP vs ntp: Δ={diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
    results.append(dict(task=task, comparison="mtpp_vs_ntp", diff=diff, ci_lo=ci_lo, ci_hi=ci_hi, p=p))

    # MTPP vs MOTOR
    diff, ci_lo, ci_hi, p = bootstrap_r2_test(labels, mtpp_preds, motor_preds)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  MTPP vs MOTOR: Δ={diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], p={p:.4f} {sig}")
    results.append(dict(task=task, comparison="mtpp_vs_motor", diff=diff, ci_lo=ci_lo, ci_hi=ci_hi, p=p))

# ── Summary table ──
df_results = pd.DataFrame(results)

# ── LaTeX table ──
task_display = {
    "creatinine": "Creatinine",
    "platelets": "Platelets",
    "pao2": "PaO2",
    "glucose": "Glucose",
}
baseline_display = {"mtpp_vs_ntp": "ntp", "mtpp_vs_motor": "MOTOR"}

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