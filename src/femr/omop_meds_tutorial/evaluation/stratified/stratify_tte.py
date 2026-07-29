"""IPCW C-index by history-length regime for the phenotype survival task.

The saved tte_results pickle only stores bootstrap C-indices, so per-sample risk scores
are recovered by reloading the saved DeepHit network and replaying the deterministic
train/val/hyper split from evaluation/p_test/tte/tte_test.py. Each (task, model) pair is
gated on reproducing the published full-test C-index before any stratified number is kept.

    python stratify_tte.py [--n_jobs 16]
"""

import argparse
import glob
import json
import pickle

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pycox.models import DeepHitSingle
from sksurv.metrics import concordance_index_ipcw

from femr.omop_meds_tutorial.evaluation.p_test.tte.utils import make_structured_array
from femr.omop_meds_tutorial.evaluation.stratified.common import (
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
from femr.omop_meds_tutorial.evaluation.stratified.paths import (
    BASELINES,
    MODELS,
    ORA,
    PHENOTYPE_ROOT,
    PHENOTYPE_TASKS,
    TTE_TRUNCATION,
    counts_path,
    phenotype_tte_dir,
)

REPRO_TOL = 0.005


def _target(df):
    return df["time_to_event"].values, df["tte_label"].values


def load_split(task, model):
    """Rebuild the exact train/test frames used when the saved model was fit."""
    cohort_files = sorted(glob.glob(f"{PHENOTYPE_ROOT}/cohort/tte/{task}/*.parquet"))
    cohort = pd.concat([pd.read_parquet(f) for f in cohort_files], ignore_index=True)
    cohort = cohort.rename(columns={"time_to_event_days": "time_to_event"}, errors="ignore")
    keys = cohort[["subject_id", "prediction_time", "time_to_event", "tte_label"]].drop_duplicates()

    d = phenotype_tte_dir(task, model)
    out = []
    for name in ("train", "test"):
        df = pd.read_parquet(f"{d}/features_with_label/{name}.parquet")
        df = df.merge(keys, on=["subject_id", "prediction_time"], how="inner")
        out.append(df.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True))
    return out


def load_model(task, model, train):
    """Reload the saved net; labtrans.cuts is re-derived from the same train subset."""
    d = phenotype_tte_dir(task, model)
    val = train.sample(frac=0.2, replace=False, random_state=42)
    fit_train = train.drop(val.index)

    labtrans = DeepHitSingle.label_transform(100)
    labtrans.fit_transform(*_target(fit_train))

    net = torch.load(f"{d}/tte_model_{TTE_TRUNCATION}.pt", map_location="cpu",
                     weights_only=False)
    return DeepHitSingle(net, duration_index=labtrans.cuts), fit_train


def risk_scores(task, model):
    """Per-sample risk at the truncation horizon, in saved test order."""
    d = phenotype_tte_dir(task, model)
    train, test = load_split(task, model)
    saved = pickle.load(open(f"{d}/tte_results_{TTE_TRUNCATION}.pkl", "rb"))

    order = list(zip(test["subject_id"].tolist(), test["prediction_time"].astype(str).tolist()))
    assert order == saved["test_order"], f"{task}/{model}: test order does not match saved run"

    net, fit_train = load_model(task, model, train)
    x_test = np.stack(test.features.values).astype(np.float32)
    surv = net.predict_surv_df(x_test)
    times = surv.index.to_numpy()
    risk = 1.0 - surv.iloc[int(np.argmin(np.abs(times - TTE_TRUNCATION)))].values

    et_train = make_structured_array(*_target(fit_train))
    et_test = make_structured_array(*_target(test))

    published = json.load(open(f"{d}/tte_metrics_{TTE_TRUNCATION}.json"))["mean"]
    recomputed = concordance_index_ipcw(et_train, et_test, risk, tau=TTE_TRUNCATION)[0]
    assert abs(recomputed - published) < REPRO_TOL, (
        f"{task}/{model}: recomputed C-index {recomputed:.5f} != published {published:.5f}"
    )
    print(f"  {model}: reproduced C-index {recomputed:.5f} (published {published:.5f})")
    return test, et_train, et_test, risk, recomputed


def usable_bootstrap(et_test, boot):
    """Keep replicates with at least one event inside the truncation horizon.

    concordance_index_ipcw restricts the test set to t < tau and then needs a
    non-censored sample there. Stroke has only 5-6 such events per stratum, so a few
    replicates are degenerate. Filtering once per stratum keeps every model on the same
    replicate set, so the paired deltas stay valid.
    """
    in_window = et_test["e"] & (et_test["t"] < TTE_TRUNCATION)
    keep = np.array([in_window[b].any() for b in boot])
    return boot[keep], int((~keep).sum())


def _cindex_chunk(chunk, et_train, et_test, risk, tau):
    return [concordance_index_ipcw(et_train, et_test[b], risk[b], tau=tau)[0] for b in chunk]


def bootstrap_cindex(et_train, et_test, risk, boot, n_jobs):
    chunks = np.array_split(boot, n_jobs)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_cindex_chunk)(c, et_train, et_test, risk, TTE_TRUNCATION) for c in chunks
    )
    return np.array([v for part in parts for v in part])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()

    threshold_rows, metric_rows, delta_rows, hetero_rows, table_rows = [], [], [], [], []

    for task in PHENOTYPE_TASKS:
        print(f"=== {task} ===")
        counts = pd.read_parquet(counts_path("phenotype_tte", task))
        counts["prediction_time"] = counts["prediction_time"].astype("datetime64[us]")
        c33, c67 = tertile_thresholds(counts["n_events"])

        per_model, regimes, n_rows = {}, None, None
        for model in MODELS:
            test, et_train, et_test, risk, point = risk_scores(task, model)
            merged = test[["subject_id", "prediction_time"]].copy()
            merged["prediction_time"] = merged["prediction_time"].astype("datetime64[us]")
            merged = merged.merge(counts, on=["subject_id", "prediction_time"], how="left")
            assert len(merged) == len(test) and merged["n_events"].notna().all()
            r = assign_regime(merged["n_events"].to_numpy(), c33, c67)
            if regimes is None:
                regimes, n_rows = r, len(test)
            else:
                assert np.array_equal(r, regimes), f"{task}/{model}: regime assignment differs"
                assert np.array_equal(et_test, per_model[MODELS[0]][1]), \
                    f"{task}/{model}: test labels differ from {MODELS[0]}"
            per_model[model] = (et_train, et_test, risk, point)

        sizes = {reg: int((regimes == reg).sum()) for reg in REGIMES}
        threshold_rows.append({
            "family": "phenotype_tte", "task": task, "metric": "cindex", "n": n_rows,
            "c33": c33, "c67": c67, "median_n_events": float(counts["n_events"].median()),
            **{f"n_{reg}": sizes[reg] for reg in REGIMES},
        })

        deltas_by_regime = {}
        for regime in REGIMES + ["all"]:
            mask = np.ones(n_rows, dtype=bool) if regime == "all" else (regimes == regime)
            idx = np.flatnonzero(mask)
            boot, n_dropped = usable_bootstrap(per_model[MODELS[0]][1][idx],
                                               bootstrap_indices(len(idx)))
            if n_dropped:
                print(f"  regime={regime}: dropped {n_dropped}/{N_BOOT} replicates "
                      f"with no event before tau")

            samples = {}
            for model in MODELS:
                et_train, et_test, risk, full_point = per_model[model]
                ets, rs = et_test[idx], risk[idx]
                point = (full_point if regime == "all"
                         else concordance_index_ipcw(et_train, ets, rs, tau=TTE_TRUNCATION)[0])
                vals = bootstrap_cindex(et_train, ets, rs, boot, args.n_jobs)
                samples[model] = vals
                stats = summarize(vals, point=point)
                metric_rows.append({
                    "family": "phenotype_tte", "task": task, "metric": "cindex",
                    "regime": regime, "model": model, "n": len(idx),
                    "n_event": int(ets["e"].sum()), "event_rate": float(ets["e"].mean()),
                    "n_event_before_tau": int((ets["e"] & (ets["t"] < TTE_TRUNCATION)).sum()),
                    "n_boot": len(boot), "n_boot_dropped": n_dropped, **stats,
                })
                table_rows.append({"task": f"phenotype_tte:{task}", "regime": regime,
                                   "model": model, "point": point, "std": stats["std"]})

            save_samples("phenotype_tte", task, regime, "cindex", samples)
            for base in BASELINES:
                d, raw = paired_delta(samples[ORA], samples[base])
                delta_rows.append({"family": "phenotype_tte", "task": task,
                                   "metric": "cindex", "regime": regime,
                                   "baseline": base, "n": len(idx), **d})
                deltas_by_regime.setdefault(base, {})[regime] = raw
            print(f"  regime={regime} n={len(idx)} done")

        for base in BASELINES:
            hetero_rows.append({
                "family": "phenotype_tte", "task": task, "metric": "cindex", "baseline": base,
                **{f"delta_{r}": float(np.mean(deltas_by_regime[base][r])) for r in REGIMES},
                **heterogeneity(deltas_by_regime[base]["high"], deltas_by_regime[base]["low"]),
            })

    write_wide_table(table_rows, "cindex")
    append_csv(threshold_rows, "thresholds_tte.csv")
    append_csv(metric_rows, "stratified_metrics_tte.csv")
    append_csv(delta_rows, "stratified_deltas_tte.csv")
    append_csv(hetero_rows, "heterogeneity_tte.csv")


if __name__ == "__main__":
    main()
