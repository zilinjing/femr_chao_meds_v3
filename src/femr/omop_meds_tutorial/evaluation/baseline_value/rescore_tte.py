"""Re-score the already-trained DeepHit TTE baselines at a different truncation.

Why this exists
---------------
`train_survival` never passes `truncation` to `evaluate_survival`, so the sweep ran
at the signature default `5 * 365 = 1825` days (train_utils.py:162, :394). The
foundation-model arm (`finetune_motor.py`, run with `--truncation 300`) reports the
C-index at 300 days, so the two arms were not comparable.

Truncation enters ONLY at evaluation time -- the `tau` handed to
`concordance_index_ipcw`, the row of `surv_df` the risk score is read from, and the
IBS time grid. Nothing in the training path reads it: model selection is by
validation NLL (train_utils.py:352) and `LabTransDiscreteTime(100)` is fit on raw,
uncapped durations (train_utils.py:274). `finetune_motor.py` is the same -- raw
`fit_transform` at line 218, truncation used only from line 272. So the trained
model is truncation-agnostic and there is no reason to repeat the 48-config grid
search (~4000-5200s per task). Reloading the saved net and re-running the forward
pass reproduces the exact same `surv_df`.

What it reloads
---------------
`<task>_deephit_bootstrap/tte_model_1825.pt` is written by `Model.save_net`, which
is `torch.save(self.net, path)` -- the whole `MLPVanilla` module, not a state_dict.
So it comes back directly, with no need to rebuild the architecture from
`best_params`. Everything else needed to turn raw features back into model input
(`keep_columns`, `scaler_mean`, `scaler_scale`, `labtrans_cuts`) is in
`reproducibility.pkl`, written by train_survival.

This only densifies the *test* rows. `densify_for_mlp` densifies all four splits to
fit the scaler, which peaked around 55 GB; here the scaler is already fitted, so the
train/val/hyper matrices are never materialised. Train durations still come from the
label table (IPCW needs the training censoring distribution), but those are one
column, not a matrix.

Two verification gates run before anything is written:

1. The reconstructed test row order is asserted equal to the `test_order` stored in
   `tte_results_1825.pkl`. That is what proves the split reconstruction here matches
   the one the model was actually trained and scored under -- and it is also the
   list `p_test.tte.utils.paired_bootstrap_test` asserts on.
2. The first `--verify_draws` bootstrap resamples are re-scored at 1825 and compared
   against the stored per-draw C-index values in `<task>_deephit.pickle`. The RNG is
   `default_rng(42)` in both, so these must agree to floating-point noise. If they
   do, the reloaded model is bit-for-bit the model that produced the old numbers.

Usage::

    python rescore_tte.py --truncation 300 --n_jobs 16
    python rescore_tte.py --tasks ami --truncation 300 --dry_run
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw

from pycox.models import DeepHitSingle

from train_utils import (
    evaluate_survival,
    load_metrics,
    make_structured_array,
    save_metrics,
    summarize,
)

ROOT = "/data/zilinjing/ethos_fm/storage/ora"
DATA_PATH = os.path.join(ROOT, "meds_tab")
RESULTS_PATH = os.path.join(ROOT, "result", "meds_tab")
SPLIT_PATH = os.path.join(ROOT, "metadata", "main_split.csv")
TTE_COL = "time_to_event_days"

PHENOTYPES = ["ami", "stroke", "masld"]
# Truncation of the artifacts this script reloads. The original sweep wrote 1825;
# once those are deleted, pass --from_truncation 300 to re-score from the 300 files.
DEFAULT_FROM_TRUNCATION = 5 * 365


def build_labels(label_path, split_path):
    """Reproduce run_all.load_features_labels' label handling, without the features.

    run_all.py cannot be imported (module-level argparse + the sweep loop run on
    import), so the split derivation is duplicated here. It must stay byte-identical
    to run_all.py:119-138 -- the assertion on `test_order` in `rescore_one` is what
    catches it if it ever drifts.
    """
    labels = pd.read_parquet(label_path).set_index(["subject_id", "prediction_time"])
    labels = labels[~labels.index.duplicated(keep="last")]

    split = pd.read_csv(split_path, index_col="subject_id")
    labels = labels.join(split)

    val = labels[labels.split_name == "train"].index.to_series().sample(
        frac=0.2, replace=False, random_state=42
    )
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    labels.loc[val, "split_name"] = "tuning"
    labels.loc[hyper, "split_name"] = "hyper"
    return labels


def align_features(feature_path, labels):
    """Row-align the sparse matrix to `labels`, mirroring run_all.py:108-128.

    Returns the CSR matrix restricted to label rows, still sparse.
    """
    data = pickle.load(open(feature_path, "rb"))
    features = data["features"].tocsr()

    idx = pd.MultiIndex.from_arrays(
        [data["subject_ids"], data["feature_times"]],
        names=["subject_id", "prediction_time"],
    )
    keep = ~idx.duplicated(keep="last")
    features = features[np.flatnonzero(keep)]
    idx = idx[keep]

    pos = idx.get_indexer(labels.index)
    missing = int((pos < 0).sum())
    if missing:
        raise RuntimeError(
            f"{missing}/{len(labels)} labels have no feature row. run_all.py drops "
            "these with a warning; if that happened during training the split here "
            "will not match and the test_order assertion below would fail anyway."
        )
    return features[pos]


def restore_scaler(bundle):
    """Rebuild the fitted StandardScaler from the saved statistics.

    Going through a real StandardScaler rather than `(x - mean) / scale` by hand
    keeps the transform path identical to densify_for_mlp (train_utils.py:129).
    """
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(bundle["scaler_mean"])
    scaler.scale_ = np.asarray(bundle["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1
    return scaler


def verify_against_stored(surv_df, et_train, et_test, stored_cindex, n_draws, from_trunc):
    """Re-run the first `n_draws` reported resamples at `from_trunc` and compare.

    `evaluate_survival` seeds `np.random.default_rng(42)` and draws
    `rng.choice(n, n)` per iteration, so replaying the same generator here
    reproduces the same resamples. Any mismatch means the reloaded model, the
    feature reconstruction, or the split is not what produced the stored numbers.
    """
    times = surv_df.index.to_numpy()
    tau_idx = int(np.argmin(np.abs(times - from_trunc)))

    rng = np.random.default_rng(42)
    fresh = []
    for _ in range(n_draws):
        sample = rng.choice(len(et_test), len(et_test))
        risk = 1.0 - surv_df.iloc[tau_idx, sample].values
        fresh.append(
            concordance_index_ipcw(et_train, et_test[sample], risk, tau=from_trunc)[0]
        )

    fresh = np.asarray(fresh)
    ref = np.asarray(stored_cindex[:n_draws])
    delta = np.abs(fresh - ref).max()
    print(f"  verify @{from_trunc}: max|new - stored| = {delta:.2e} over {n_draws} draws")
    for i, (f, r) in enumerate(zip(fresh, ref)):
        print(f"    draw {i}: {f:.10f} vs stored {r:.10f}")
    if delta > 1e-6:
        raise RuntimeError(
            f"reloaded model does not reproduce the stored {from_trunc} C-index "
            f"(max delta {delta:.2e}). Refusing to write new results."
        )
    return delta


def rescore_one(task, truncation, n_jobs, verify_draws, dry_run, from_trunc,
                metrics_out="overwrite"):
    print("=" * 72)
    print(f"{task}: re-scoring at truncation={truncation} (from {from_trunc} artifacts)")

    boot_dir = os.path.join(RESULTS_PATH, f"{task}_deephit_bootstrap")
    bundle_path = os.path.join(boot_dir, "reproducibility.pkl")
    net_path = os.path.join(boot_dir, f"tte_model_{from_trunc}.pt")
    old_results_path = os.path.join(boot_dir, f"tte_results_{from_trunc}.pkl")
    # Gate 2 always reads the canonical metrics file; where results are *written*
    # depends on --metrics_out, so a re-score can leave the reference intact.
    metrics_path = os.path.join(RESULTS_PATH, f"{task}_deephit.pickle")
    out_metrics_path = (metrics_path if metrics_out == "overwrite"
                        else os.path.join(RESULTS_PATH, f"{task}_deephit_{truncation}.pickle"))

    for p in (bundle_path, net_path, old_results_path, metrics_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} -- if the {from_trunc} artifacts were deleted, point "
                "--from_truncation at a truncation that still has them."
            )

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    labels = build_labels(bundle["label_path"], bundle["split_path"])
    split = labels["split_name"].values
    tte = labels[bundle["tte_col"]].values
    event = labels["tte_label"].values.astype(bool)

    train_mask = split == "train"
    test_mask = split == "test"

    test_idx = labels.index[test_mask]
    test_order = [
        (int(s_), pd.Timestamp(t_).isoformat())
        for s_, t_ in zip(
            test_idx.get_level_values("subject_id"),
            test_idx.get_level_values("prediction_time"),
        )
    ]

    # Gate 1: the split reconstruction must match what the model was scored under.
    with open(old_results_path, "rb") as f:
        old_results = pickle.load(f)
    if list(old_results["test_order"]) != [list(t) for t in test_order] and \
       [tuple(t) for t in old_results["test_order"]] != test_order:
        raise RuntimeError(
            f"{task}: reconstructed test_order does not match the one stored in "
            f"{old_results_path}. The split derivation in build_labels has drifted "
            "from run_all.load_features_labels."
        )
    print(f"  test_order matches stored ({len(test_order)} rows)")

    y_train = (tte[train_mask], event[train_mask])
    y_test = (tte[test_mask], event[test_mask])
    print(f"  n_train={train_mask.sum()}  n_test={test_mask.sum()}")

    # Only the test rows are densified; the scaler is already fitted.
    X = align_features(bundle["feature_path"], labels)
    keep = bundle["keep_columns"]
    if X.shape[1] != bundle["n_columns_in"]:
        raise RuntimeError(
            f"{task}: feature matrix has {X.shape[1]} columns, bundle expects "
            f"{bundle['n_columns_in']}"
        )
    scaler = restore_scaler(bundle)
    x_test = scaler.transform(
        np.asarray(X[np.flatnonzero(test_mask)][:, keep].todense(), dtype=np.float32)
    ).astype("float32")
    del X
    print(f"  x_test {x_test.shape} ({x_test.nbytes / 1024**3:.1f} GB)")
    if x_test.shape[1] != bundle["in_features"]:
        raise RuntimeError(
            f"{task}: rebuilt {x_test.shape[1]} features, net expects "
            f"{bundle['in_features']}"
        )

    net = torch.load(net_path, map_location="cpu", weights_only=False)
    model = DeepHitSingle(net, duration_index=bundle["labtrans_cuts"])
    print(f"  loaded net from {os.path.basename(net_path)} on {model.device}")

    surv_df = model.predict_surv_df(x_test)

    et_train = make_structured_array(*y_train)
    et_test = make_structured_array(*y_test)

    # Gate 2: reproduce the stored numbers before writing anything new. The stored
    # metrics pickle carries no truncation in its filename, so `_truncation` (absent
    # on the original 1825 sweep) decides whether it is a valid reference here.
    stored = load_metrics(metrics_path)
    stored_trunc = stored.get("_truncation", float(DEFAULT_FROM_TRUNCATION))
    if float(stored_trunc) == float(from_trunc):
        verify_against_stored(
            surv_df, et_train, et_test, stored["C-Index"], verify_draws, from_trunc
        )
    else:
        print(f"  WARNING: {metrics_path} holds tau={stored_trunc} metrics, not "
              f"{from_trunc} -- skipping the reproduction gate. The test_order "
              "check above still applies.")

    times = surv_df.index.to_numpy()
    cut = float(times[int(np.argmin(np.abs(times - truncation)))])
    print(f"  tau={truncation}; nearest discrete-time cut = {cut:.2f} days")

    if dry_run:
        print("  --dry_run: verification passed, nothing written")
        return None

    # `model=` makes save_survival_results write tte_model_<tau>.pt as well, so the
    # trained net survives the deletion of the 1825 artifacts.
    metrics = evaluate_survival(
        surv_df, y_train, y_test,
        truncation=truncation,
        test_order=test_order,
        save_path=boot_dir,
        model_name=bundle["model_name"],
        model=model,
        best_params=bundle["best_params"],
        n_jobs=n_jobs,
    )
    metrics["_best_params"] = stored.get("_best_params")
    metrics["_truncation"] = float(truncation)
    metrics["_nearest_cut"] = cut
    metrics["_rescored_from"] = os.path.basename(net_path)

    save_metrics(metrics, out_metrics_path)
    print(f"  wrote {out_metrics_path}")

    if metrics_out == "overwrite":
        # Keep the bundle's net_file pointer honest -- train_utils.py:382 hardcodes
        # 1825. Skipped in 'suffixed' mode so the source sweep stays byte-identical.
        bundle["net_file"] = f"tte_model_{truncation}.pt"
        bundle["truncation"] = float(truncation)
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=PHENOTYPES)
    ap.add_argument("--truncation", type=float, default=300.0)
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--verify_draws", type=int, default=5,
                    help="Reported resamples replayed at --from_truncation to prove "
                         "the reload is exact.")
    ap.add_argument("--from_truncation", type=float, default=float(DEFAULT_FROM_TRUNCATION),
                    help="Truncation whose tte_model_<tau>.pt / tte_results_<tau>.pkl "
                         "this reloads. Use 300 once the 1825 artifacts are gone.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Run both verification gates, write nothing.")
    ap.add_argument("--results_dir", default=None,
                    help="Sweep directory to re-score (default: the main meds_tab "
                         "results tree). Use this to score an alternative sweep such "
                         "as meds_tab_wd1e-4.")
    ap.add_argument("--metrics_out", default="overwrite", choices=["overwrite", "suffixed"],
                    help="'overwrite' replaces <task>_deephit.pickle; 'suffixed' writes "
                         "<task>_deephit_<tau>.pickle and leaves the source sweep's "
                         "records untouched.")
    a = ap.parse_args()

    if a.results_dir:
        global RESULTS_PATH
        RESULTS_PATH = a.results_dir
    print(f"results dir: {RESULTS_PATH}")

    as_int = lambda x: int(x) if float(x).is_integer() else x
    trunc = as_int(a.truncation)
    from_trunc = as_int(a.from_truncation)

    summary = {}
    for task in a.tasks:
        m = rescore_one(task, trunc, a.n_jobs, a.verify_draws, a.dry_run, from_trunc,
                        a.metrics_out)
        if m is not None:
            summary[task] = m
            print(f"--- {task} @ {trunc} ---")
            summarize(m)

    if summary:
        print("=" * 72)
        print(f"{'task':<10} {'C-index (100)':<22} {'C-index (10k saved)':<24} {'Int-Brier'}")
        for task, m in summary.items():
            c = np.asarray(m["C-Index"])
            b = np.asarray(m["Int-Brier"])
            s = m["_saved_bootstrap"]
            print(f"{task:<10} {c.mean():.4f} ({c.std():.4f})       "
                  f"{s['mean']:.4f} ({s['std']:.4f})         {b.mean():.4f} ({b.std():.4f})")


if __name__ == "__main__":
    main()
