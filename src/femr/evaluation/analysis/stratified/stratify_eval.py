"""AUROC and R^2 by history-length regime, for every model and task.

Re-scores existing held-out predictions inside low/mid/high history strata, then reports
paired ORA-minus-baseline gains per stratum and tests whether those gains differ between
the high- and low-history strata.

    python stratify_eval.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from femr.evaluation.analysis.stratified.common import (
    N_BOOT,
    REGIMES,
    append_csv,
    assign_regime,
    bootstrap_indices,
    heterogeneity,
    paired_delta,
    save_samples,
    summarize,
    tertile_thresholds,
    write_wide_table,
)
from femr.evaluation.analysis.stratified.paths import (
    BASELINES,
    MODELS,
    ORA,
    OUTCOME_TASKS,
    PHENOTYPE_TASKS,
    REGRESSION_TASKS,
    counts_path,
    outcome_predictions,
    phenotype_predictions,
    regression_predictions,
)


def load_counts(family, task):
    counts = pd.read_parquet(counts_path(family, task))
    counts["prediction_time"] = counts["prediction_time"].astype("datetime64[us]")
    return counts


def load_predictions(resolver, task, model, value_cols):
    df = pd.read_parquet(resolver(task, model))
    df["prediction_time"] = df["prediction_time"].astype("datetime64[us]")
    return df[["subject_id", "prediction_time"] + value_cols]


def attach_regimes(df, counts, c33, c67):
    merged = df.merge(counts, on=["subject_id", "prediction_time"], how="left")
    assert len(merged) == len(df), f"join changed row count {len(df)} -> {len(merged)}"
    assert merged["n_events"].notna().all(), "some prediction points have no event count"
    merged["regime"] = assign_regime(merged["n_events"].to_numpy(), c33, c67)
    return merged


def evaluate_task(family, task, resolver, metric, value_cols, threshold_rows,
                  metric_rows, delta_rows, hetero_rows, table_rows):
    counts = load_counts(family, task)
    c33, c67 = tertile_thresholds(counts["n_events"])

    frames = {}
    keys = None
    for model in MODELS:
        df = attach_regimes(load_predictions(resolver, task, model, value_cols), counts, c33, c67)
        df = df.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
        k = list(zip(df["subject_id"], df["prediction_time"]))
        if keys is None:
            keys = k
        else:
            assert k == keys, f"{family}/{task}: {model} evaluates a different key set"
        frames[model] = df

    ref = frames[MODELS[0]]
    sizes = {r: int((ref["regime"] == r).sum()) for r in REGIMES}
    assert sum(sizes.values()) == len(ref)
    threshold_rows.append({
        "family": family, "task": task, "metric": metric, "n": len(ref),
        "c33": c33, "c67": c67, "median_n_events": float(ref["n_events"].median()),
        **{f"n_{r}": sizes[r] for r in REGIMES},
    })

    deltas_by_regime = {}
    for regime in REGIMES + ["all"]:
        mask = np.ones(len(ref), dtype=bool) if regime == "all" else (ref["regime"] == regime).to_numpy()
        idx = np.flatnonzero(mask)
        boot = bootstrap_indices(len(idx))

        stratum = ref.loc[idx]
        if metric == "auroc":
            y = stratum["boolean_value"].to_numpy().astype(int)
            extra = {"n_positive": int(y.sum()), "prevalence": float(y.mean())}
        else:
            extra = {"n_positive": None,
                     "prevalence": float(stratum["y_true"].mean())}

        samples = {}
        for model in MODELS:
            sub = frames[model].loc[idx]
            if metric == "auroc":
                y = sub["boolean_value"].to_numpy().astype(int)
                p = sub["predicted_boolean_probability"].to_numpy()
                point = roc_auc_score(y, p)
                aupr = average_precision_score(y, p)
                vals = np.array([roc_auc_score(y[b], p[b]) for b in boot])
            else:
                yt = sub["y_true"].to_numpy()
                yp = sub["y_pred"].to_numpy()
                point = r2_score(yt, yp)
                aupr = None
                vals = np.array([r2_score(yt[b], yp[b]) for b in boot])
            samples[model] = vals
            stats = summarize(vals, point=point)
            metric_rows.append({
                "family": family, "task": task, "metric": metric, "regime": regime,
                "model": model, "n": len(idx), **extra, "auprc": aupr,
                "n_boot": N_BOOT, **stats,
            })
            table_rows.append({"task": f"{family}:{task}", "regime": regime,
                               "model": model, "point": point, "std": stats["std"]})

        save_samples(family, task, regime, metric, samples)
        for base in BASELINES:
            d, raw = paired_delta(samples[ORA], samples[base])
            delta_rows.append({"family": family, "task": task, "metric": metric,
                               "regime": regime, "baseline": base, "n": len(idx), **d})
            deltas_by_regime.setdefault(base, {})[regime] = raw

    for base in BASELINES:
        h = heterogeneity(deltas_by_regime[base]["high"], deltas_by_regime[base]["low"])
        hetero_rows.append({
            "family": family, "task": task, "metric": metric, "baseline": base,
            **{f"delta_{r}": float(np.mean(deltas_by_regime[base][r])) for r in REGIMES},
            **h,
        })

    print(f"{family}/{task}: c33={c33:.0f} c67={c67:.0f} sizes={sizes}")


def main():
    threshold_rows, metric_rows, delta_rows, hetero_rows = [], [], [], []
    auroc_table, r2_table = [], []

    for task in OUTCOME_TASKS:
        evaluate_task("outcome", task, outcome_predictions, "auroc",
                      ["boolean_value", "predicted_boolean_probability"],
                      threshold_rows, metric_rows, delta_rows, hetero_rows, auroc_table)
    for task in PHENOTYPE_TASKS:
        evaluate_task("phenotype_binary", task, phenotype_predictions, "auroc",
                      ["boolean_value", "predicted_boolean_probability"],
                      threshold_rows, metric_rows, delta_rows, hetero_rows, auroc_table)
    for task in REGRESSION_TASKS:
        evaluate_task("regression", task, regression_predictions, "r2",
                      ["y_true", "y_pred"],
                      threshold_rows, metric_rows, delta_rows, hetero_rows, r2_table)

    write_wide_table(auroc_table, "auroc")
    write_wide_table(r2_table, "r2")
    append_csv(threshold_rows, "thresholds_binary_regression.csv")
    append_csv(metric_rows, "stratified_metrics_binary_regression.csv")
    append_csv(delta_rows, "stratified_deltas_binary_regression.csv")
    append_csv(hetero_rows, "heterogeneity_binary_regression.csv")


if __name__ == "__main__":
    main()
