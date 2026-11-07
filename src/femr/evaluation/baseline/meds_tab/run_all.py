"""Train the strengthened tabular baselines (XGBoost + DeepHit) on value-aware features.

Usually started by script/<dataset>/baseline.sh (see evaluation/README.md), which passes the
dataset's paths. With --eval_root, every task also gets
<eval_root>/<outcome|phenotype>/<task>/<xgboost|deephit>/{predictions.parquet or
tte_predictions.parquet, metrics.json}, scored like the foundation-model probes, so the baselines
appear as rows of the summary tables.

Mirrors evaluation/baseline/run_all.py; differences, all deliberate:

1. Reads from storage/ora/meds_tab/ instead of the submitted feature root, so the
   original results are preserved and both baseline rows can be reported.
2. Keeps the feature matrix **sparse** for XGBoost instead of calling `.todense()`;
   the value-aware matrix is much wider and the dense path risks OOM.
   DeepHit still gets dense, scaled input (see train_utils.densify_for_mlp).
3. The `continue` that skipped the binary arm of the phenotype loop is removed --
   both binary and TTE results are produced for every phenotype.
4. Wall-clock and peak RSS are recorded per task and written next to the metrics,
   to answer the reviewer's question about baseline compute cost.

Run as a script from inside this directory (it does `from train_utils import ...`)::

    python run_all.py --n_jobs 24                      # everything, sequential
    python run_all.py --phenotypes ami --skip survival  # one task, one family

The DeepHit arm is the expensive one and is GPU-bound, so run the three TTE tasks
concurrently on separate idle GPUs -- this is a free 3x reduction in wall-clock:

    CUDA_VISIBLE_DEVICES=2 python run_all.py --phenotypes ami    --skip binary --n_jobs 8 &
    CUDA_VISIBLE_DEVICES=3 python run_all.py --phenotypes stroke --skip binary --n_jobs 8 &
    CUDA_VISIBLE_DEVICES=4 python run_all.py --phenotypes masld  --skip binary --n_jobs 8 &

The XGBoost arm is CPU-bound and can run alongside them:

    python run_all.py --skip survival --n_jobs 24
"""

import argparse
import json
import os
import pickle
import resource
import time

import numpy as np
import pandas as pd

from femr.evaluation.common.metrics import binary_metrics, cindex_metrics, write_metrics
from train_utils import (
    load_metrics,
    save_metrics,
    split_binary,
    split_survival,
    summarize,
    train_binary,
    train_survival,
)

ROOT = "/data/zilinjing/ethos_fm/storage/ora"
DATA_PATH = os.path.join(ROOT, "meds_tab")
RESULTS_PATH = os.path.join(ROOT, "result", "meds_tab")
SPLIT_PATH = os.path.join(ROOT, "metadata", "main_split.csv")
TTE_COL = "time_to_event_days"

PHENOTYPES = ["ami", "stroke", "masld"]
BINARY_ONLY = ["in_hospital_mortality", "readmission", "long_los"]

parser = argparse.ArgumentParser()
parser.add_argument("--phenotypes", nargs="*", default=None,
                    help=f"Subset of {PHENOTYPES}")
parser.add_argument("--binary_only", nargs="*", default=None,
                    help=f"Subset of {BINARY_ONLY}")
parser.add_argument("--skip", nargs="*", default=[], choices=["survival", "binary"],
                    help="Task families to skip")
parser.add_argument("--n_jobs", type=int, default=16,
                    help="CPU cores for XGBoost (-1 = all). Report this alongside timings.")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--results_dir", default=None,
                    help="Override the results directory (use a fresh one to avoid "
                         "overwriting a previous sweep).")
parser.add_argument("--wd", default=None,
                    help="Comma-separated weight-decay values, overriding the default "
                         "[0.0, 1e-2]. e.g. --wd 1e-4 gives a single value (24 configs).")
parser.add_argument("--deephit_grid", default="fast", choices=["fast", "full"],
                    help="'fast' = 48 configs (drops the [50,50] architecture); 'full' = 72.")
parser.add_argument("--device", default="cpu",
                    help="XGBoost device: 'cpu' or 'cuda'. cuda is ~9x faster here.")
parser.add_argument("--patience", type=int, default=5,
                    help="EarlyStopping patience for DeepHit (torchtuples default is 10).")
parser.add_argument("--data_path", default=DATA_PATH, help="MEDS-Tab features written by extract.sh")
parser.add_argument("--split_path", default=SPLIT_PATH, help="main_split.csv")
parser.add_argument("--truncation", type=float, default=5 * 365,
                    help="C-index horizon in days (300 for MIMIC, 365 for CUMC)")
parser.add_argument("--tte_col", default=TTE_COL,
                    help="Time column of the TTE labels (time_to_event_days for MIMIC, "
                         "time_to_event for CUMC)")
parser.add_argument("--eval_root", default=None,
                    help="Also write predictions + metrics.json in the evaluation results layout")
args = parser.parse_args()

if args.results_dir:
    RESULTS_PATH = args.results_dir
DATA_PATH = args.data_path
SPLIT_PATH = args.split_path
TTE_COL = args.tte_col
WD_VALUES = [float(x) for x in args.wd.split(",")] if args.wd else None


def peak_rss_gb() -> float:
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_kb, child_kb) / 1024**2


def load_features_labels(feature_path, label_path, split_path):
    """Align the sparse feature matrix to the label table without densifying.

    The original implementation built a dense DataFrame via `.todense()` and used
    `.loc`. With the value-aware feature set that matrix is far larger, so rows
    are aligned through the MultiIndex and the CSR matrix is row-sliced instead.

    Duplicate (subject_id, prediction_time) keys are resolved keep="last" on both
    sides, matching the original `remove_duplicates` behaviour. The train/tuning/
    hyper resampling is byte-identical to the submitted baseline (same seeds), so
    the two rows are directly comparable.
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

    labels = pd.read_parquet(label_path).set_index(["subject_id", "prediction_time"])
    labels = labels[~labels.index.duplicated(keep="last")]

    pos = idx.get_indexer(labels.index)
    missing = int((pos < 0).sum())
    if missing:
        print(f"  WARNING: {missing}/{len(labels)} labels have no feature row; dropping them")
        labels = labels[pos >= 0]
        pos = pos[pos >= 0]
    X = features[pos]
    # CountFeaturizer stores columns in first-seen order. xgboost 3.2 (the last version for Python
    # 3.11) needs them sorted on the GPU: it trains bad models without any error otherwise.
    X.sort_indices()

    split = pd.read_csv(split_path, index_col="subject_id")
    labels = labels.join(split)

    val = labels[labels.split_name == "train"].index.to_series().sample(
        frac=0.2, replace=False, random_state=42
    )
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    labels.loc[val, "split_name"] = "tuning"
    labels.loc[hyper, "split_name"] = "hyper"

    print(f"  features {X.shape}, nnz={X.nnz}, labels {len(labels)}")
    return X, labels


def write_eval_outputs(family, name, model, y, pred_df=None, surv_df=None):
    """Write <eval_root>/<family>/<task>/<model>/ in the same format as the foundation-model probes."""
    out_dir = os.path.join(args.eval_root, family, name, model)
    os.makedirs(out_dir, exist_ok=True)
    if surv_df is None:  # binary
        pred_df.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
        write_metrics(out_dir, binary_metrics(pred_df))
        return
    test = y[y["split_name"] == "test"]
    train = y[y["split_name"].isin(["train", "tuning", "hyper"])]
    row = int(np.argmin(np.abs(surv_df.index.to_numpy() - args.truncation)))
    pred = pd.DataFrame({
        "subject_id": test.index.get_level_values("subject_id"),
        "prediction_time": test.index.get_level_values("prediction_time"),
        "time_to_event": test[TTE_COL].to_numpy(),
        "tte_label": test["tte_label"].to_numpy(),
        "risk": 1.0 - surv_df.iloc[row].to_numpy(),
    })
    pred.to_parquet(os.path.join(out_dir, "tte_predictions.parquet"), index=False)
    write_metrics(out_dir, cindex_metrics(pred, train[TTE_COL], train["tte_label"], args.truncation,
                                          n_jobs=args.n_jobs))


def run_one(name: str, suffix: str, model: str, outcome: str, family: str) -> None:
    out_file = os.path.join(RESULTS_PATH, f"{name}_{model}.pickle")
    print("-" * 42)
    print(f"mimic {name} [{outcome}]")

    # Reuse a finished run, unless its evaluation row (--eval_root) has not been written yet.
    eval_done = not args.eval_root or os.path.exists(
        os.path.join(args.eval_root, family, name, model, "metrics.json"))
    if os.path.exists(out_file) and eval_done and not args.overwrite:
        summarize(load_metrics(out_file))
        return

    try:
        base_dir = os.path.join(DATA_PATH, f"{name}{suffix}")
        t0 = time.perf_counter()
        X, y = load_features_labels(
            os.path.join(base_dir, "features", f"{name}.pkl"),
            os.path.join(base_dir, "labels", f"{name}.parquet"),
            SPLIT_PATH,
        )
        t_load = time.perf_counter() - t0

        # Per-task directory for the 10,000-resample artifacts used by
        # p_test.tte.utils.paired_bootstrap_test, mirroring the FM arms' layout.
        boot_dir = os.path.join(RESULTS_PATH, f"{name}_{model}_bootstrap")

        t1 = time.perf_counter()
        if outcome == "survival":
            # Collects the column subset + scaler needed to re-apply the saved net.
            artifacts = {"feature_path": os.path.join(base_dir, "features", f"{name}.pkl"),
                         "label_path": os.path.join(base_dir, "labels", f"{name}.parquet"),
                         "split_path": SPLIT_PATH, "tte_col": TTE_COL}
            splits = split_survival(X, y, TTE_COL, artifacts=artifacts)
            # test_order must match the FM arms exactly: (subject_id, prediction_time)
            # in test-row order. paired_bootstrap_test asserts equality of this list.
            test_idx = y.index[y["split_name"].values == "test"]
            test_order = [
                (int(s_), pd.Timestamp(t_).isoformat())
                for s_, t_ in zip(
                    test_idx.get_level_values("subject_id"),
                    test_idx.get_level_values("prediction_time"),
                )
            ]
            metrics, surv_df = train_survival(
                *splits, model_type="deephit", truncation=args.truncation,
                test_order=test_order, save_path=boot_dir,
                model_name=f"meds_tab_deephit_{name}",
                grid_mode=args.deephit_grid, patience=args.patience,
                n_jobs=args.n_jobs, artifacts=artifacts, wd_values=WD_VALUES,
            )
            if args.eval_root:
                write_eval_outputs(family, name, model, y, surv_df=surv_df)
        else:
            splits = split_binary(X, y)
            metrics, pred_df = train_binary(
                *splits, n_jobs=args.n_jobs, save_path=boot_dir,
                model_name=f"meds_tab_xgboost_{name}", device=args.device,
            )
            pred_df.to_parquet(out_file.replace(".pickle", "_predictions.parquet"), index=False)
            if args.eval_root:
                write_eval_outputs(family, name, model, y, pred_df=pred_df)
        t_train = time.perf_counter() - t1

        save_metrics(metrics, out_file)
        with open(out_file.replace(".pickle", "_compute.json"), "w") as f:
            json.dump(
                {
                    "task": name,
                    "outcome": outcome,
                    "model": model,
                    "n_rows": int(X.shape[0]),
                    "n_columns": int(X.shape[1]),
                    "nnz": int(X.nnz),
                    "load_seconds": round(t_load, 1),
                    "train_and_tune_seconds": round(t_train, 1),
                    "peak_rss_gb": round(peak_rss_gb(), 2),
                    "n_jobs": args.n_jobs,
                    "device": args.device if outcome == "binary" else "cuda",
                },
                f,
                indent=2,
            )
        print(f"  load {t_load:.1f}s, train+tune {t_train:.1f}s, peak RSS {peak_rss_gb():.1f} GB")

    except Exception as e:  # noqa: BLE001 -- keep the sweep going
        print(f"[ERROR] mimic | {name} | {outcome}: {type(e).__name__}: {e}")


os.makedirs(RESULTS_PATH, exist_ok=True)
phenotypes = args.phenotypes if args.phenotypes is not None else PHENOTYPES
binary_only = args.binary_only if args.binary_only is not None else BINARY_ONLY

# Phenotypes: both survival and binary.
for name in phenotypes:
    for outcome, suffix, model in [("survival", "", "deephit"), ("binary", "_binary", "xgboost")]:
        if outcome in args.skip:
            continue
        run_one(name, suffix, model, outcome, "phenotype")

# Patient outcomes: binary only.
if "binary" not in args.skip:
    for name in binary_only:
        run_one(name, "_binary", "xgboost", "binary", "outcome")
