"""Export the best-per-task DeepHit TTE model at tau=300 into a self-contained dir.

Selection provenance (IMPORTANT): the winning sweep is chosen per task by *test*
C-index at tau=300. That is test-set selection -- it biases the reported number
upward and is not a clean model-selection procedure. The choice and both
candidates' scores are recorded in MANIFEST.json so this is never invisible.
Use these artifacts as predictions to inspect, not as a defensible baseline row
without stating how the sweep was picked.

For each task this writes: the checkpoint, the reproducibility bundle needed to
re-apply it, both bootstrap result files, and -- new -- the actual test-set
predictions, which no TTE run has ever persisted (only the XGBoost binary arm
wrote a predictions parquet). Predictions are regenerated here from the
checkpoint and gated against the stored bootstrap before being written.
"""

import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_ipcw

from pycox.models import DeepHitSingle

import rescore_tte as R
from train_utils import make_structured_array

ROOT = "/data/zilinjing/ethos_fm/storage/ora/result"
OUT = os.path.join(ROOT, "meds_tab_best_300")
TAU = 300

# task -> (source sweep dir, top-level metrics pickle name in that sweep)
BEST = {
    "ami":    ("meds_tab_wd1e-4", "ami_deephit_300.pickle"),
    "stroke": ("meds_tab",        "stroke_deephit.pickle"),
    "masld":  ("meds_tab_wd1e-4", "masld_deephit_300.pickle"),
}


def export(task, sweep, metrics_name, manifest):
    src_dir = os.path.join(ROOT, sweep)
    boot = os.path.join(src_dir, f"{task}_deephit_bootstrap")
    dst = os.path.join(OUT, task)
    os.makedirs(dst, exist_ok=True)
    print(f"=== {task}: from {sweep}")

    bundle = pickle.load(open(os.path.join(boot, "reproducibility.pkl"), "rb"))

    labels = R.build_labels(bundle["label_path"], bundle["split_path"])
    sp = labels["split_name"].values
    t = labels[bundle["tte_col"]].values
    e = labels["tte_label"].values.astype(bool)
    te = sp == "test"
    tr = sp == "train"

    X = R.align_features(bundle["feature_path"], labels)
    scaler = R.restore_scaler(bundle)
    x_test = scaler.transform(
        np.asarray(X[np.flatnonzero(te)][:, bundle["keep_columns"]].todense(),
                   dtype=np.float32)
    ).astype("float32")
    del X

    net = torch.load(os.path.join(boot, f"tte_model_{TAU}.pt"),
                     map_location="cpu", weights_only=False)
    model = DeepHitSingle(net, duration_index=bundle["labtrans_cuts"])
    surv = model.predict_surv_df(x_test)          # index = times, columns = test rows
    times = surv.index.to_numpy()
    ti = int(np.argmin(np.abs(times - TAU)))
    risk = 1.0 - surv.iloc[ti].values

    # ---- gate: reproduce the stored reported C-index before writing -----------
    stored = pickle.load(open(os.path.join(src_dir, metrics_name), "rb"))
    et_train = make_structured_array(t[tr], e[tr])
    et_test = make_structured_array(t[te], e[te])
    rng = np.random.default_rng(42)
    fresh = []
    for _ in range(5):
        s = rng.choice(len(et_test), len(et_test))
        fresh.append(concordance_index_ipcw(
            et_train, et_test[s], 1.0 - surv.iloc[ti, s].values, tau=float(TAU))[0])
    delta = np.abs(np.asarray(fresh) - np.asarray(stored["C-Index"][:5])).max()
    print(f"  gate: max|new - stored| = {delta:.2e}")
    if delta > 1e-6:
        raise RuntimeError(f"{task}: predictions do not reproduce stored results")

    # ---- predictions ----------------------------------------------------------
    idx = labels.index[te]
    pred = pd.DataFrame({
        "subject_id": idx.get_level_values("subject_id").astype("int64"),
        "prediction_time": pd.to_datetime(idx.get_level_values("prediction_time")),
        "tte_label": e[te],
        "time_to_event_days": t[te],
        f"risk_at_{TAU}": risk,
        f"surv_at_{TAU}": surv.iloc[ti].values,
        "cut_used_days": float(times[ti]),
    })
    pred.to_parquet(os.path.join(dst, f"{task}_test_predictions.parquet"), index=False)

    curves = pd.DataFrame(surv.values.T, columns=[f"t_{x:.4f}" for x in times])
    curves.insert(0, "prediction_time", pred["prediction_time"].values)
    curves.insert(0, "subject_id", pred["subject_id"].values)
    curves.to_parquet(os.path.join(dst, f"{task}_test_surv_curves.parquet"), index=False)

    # ---- checkpoint + provenance ---------------------------------------------
    for fn in [f"tte_model_{TAU}.pt", f"tte_metrics_{TAU}.json", f"tte_results_{TAU}.pkl"]:
        shutil.copy2(os.path.join(boot, fn), os.path.join(dst, fn))
    with open(os.path.join(dst, "reproducibility.pkl"), "wb") as f:
        b = dict(bundle)
        b["net_file"] = f"tte_model_{TAU}.pt"
        b["truncation"] = float(TAU)
        b["source_sweep"] = sweep
        pickle.dump(b, f)
    shutil.copy2(os.path.join(src_dir, metrics_name),
                 os.path.join(dst, f"{task}_metrics_{TAU}.pickle"))

    c10k = json.load(open(os.path.join(boot, f"tte_metrics_{TAU}.json")))
    c100 = np.asarray(stored["C-Index"])
    manifest[task] = {
        "source_sweep": sweep,
        "checkpoint": f"{task}/tte_model_{TAU}.pt",
        "predictions": f"{task}/{task}_test_predictions.parquet",
        "best_params": bundle["best_params"],
        "truncation": TAU,
        "nearest_cut_days": float(times[ti]),
        "n_test_rows": int(te.sum()),
        "n_test_events_within_tau": int((e[te] & (t[te] <= TAU)).sum()),
        "cindex_100draw_mean": float(c100.mean()),
        "cindex_100draw_std": float(c100.std()),
        "cindex_10k_mean": c10k["mean"],
        "cindex_10k_std": c10k["std"],
        "selection_note": "sweep chosen by TEST c-index at tau=300 (test-set selection)",
    }
    print(f"  wrote {dst}")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for task, (sweep, mname) in BEST.items():
        export(task, sweep, mname, manifest)

    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    rows = [{"task": k, **{kk: vv for kk, vv in v.items()
                           if kk in ("source_sweep", "cindex_100draw_mean",
                                     "cindex_100draw_std", "cindex_10k_mean",
                                     "cindex_10k_std", "n_test_rows",
                                     "n_test_events_within_tau")}}
            for k, v in manifest.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary_300.csv"), index=False)
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
