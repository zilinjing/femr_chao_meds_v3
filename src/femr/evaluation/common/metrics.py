"""Metrics shared by every probe and baseline, so all rows of a results table are scored the same way.

Every metric is reported like meds-evaluation reports it: draw N_BOOT resamples of the whole test
set (rows drawn with replacement, resample b uses seed b), compute the metric on each, and report
the mean and the std (np.std, ddof=0) of the N_BOOT values.

- AUROC / AUPRC: meds_evaluation.evaluate_bootstrapped_binary_classification itself.
- C-index and R^2: the same resampling recipe (polars sample with seed b), so resample b picks the
  same test rows for every metric and every model.

Each run writes metrics.json: {"auroc": {"mean": .., "std": .., "n_boot": ..}, ...}.
common/collect.py turns all metrics.json files into the summary CSVs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from meds_evaluation.evaluate import evaluate_bootstrapped_binary_classification
from sklearn.metrics import r2_score
from sksurv.metrics import concordance_index_ipcw

N_BOOT = 1000


def resample_indices(n: int, b: int) -> np.ndarray:
    """Row indices of resample b. Identical to the rows meds-evaluation draws with seed b."""
    rows = pl.DataFrame({"i": np.arange(n)})
    return rows.sample(n=n, with_replacement=True, seed=b)["i"].to_numpy()


def summarize(values, n_boot: int) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_boot": n_boot}


def binary_metrics(pred: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """pred columns: subject_id, prediction_time, boolean_value, predicted_boolean_probability."""
    df = pl.DataFrame({
        "subject_id": pred["subject_id"].astype("int64").to_numpy(),
        "prediction_time": pd.to_datetime(pred["prediction_time"]).to_numpy(),
        "boolean_value": pred["boolean_value"].astype(bool).to_numpy(),
        "predicted_boolean_probability": pred["predicted_boolean_probability"].astype("float64").to_numpy(),
    }).with_columns(pl.lit(None, dtype=pl.Boolean).alias("predicted_boolean_value"))
    res = evaluate_bootstrapped_binary_classification(df, bootstrapping=n_boot)
    return {
        "auroc": {"mean": res["mean_roc_auc_score"], "std": res["std_roc_auc_score"], "n_boot": n_boot},
        "auprc": {"mean": res["mean_average_precision_score"], "std": res["std_average_precision_score"],
                  "n_boot": n_boot},
    }


def r2_metrics(y_true, y_pred, n_boot: int = N_BOOT) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    values = []
    for b in range(n_boot):
        idx = resample_indices(len(y_true), b)
        values.append(r2_score(y_true[idx], y_pred[idx]))
    return {"r2": summarize(values, n_boot)}


def survival_array(times, events) -> np.ndarray:
    return np.array(list(zip(np.asarray(events, bool), np.asarray(times, float))),
                    dtype=[("e", bool), ("t", float)])


def _cindex_chunk(bs, train, test, risk, tau):
    out = []
    for b in bs:
        idx = resample_indices(len(test), b)
        try:
            out.append(concordance_index_ipcw(train, test[idx], risk[idx], tau=tau)[0])
        except ValueError:  # no event (or no comparable pair) in this resample: C-index undefined
            pass
    return out


def cindex_metrics(tte_pred: pd.DataFrame, train_times, train_events, tau: float,
                   n_boot: int = N_BOOT, n_jobs: int = 16) -> dict:
    """IPCW C-index up to tau (days).

    tte_pred columns: time_to_event, tte_label, risk (higher = earlier event).
    train_times / train_events: the TTE cohort's training rows; they estimate the censoring
    distribution, and every model uses the same ones (see train_tte_labels).
    """
    train = survival_array(train_times, train_events)
    test = survival_array(tte_pred["time_to_event"], tte_pred["tte_label"])
    risk = tte_pred["risk"].to_numpy(float)
    chunks = np.array_split(np.arange(n_boot), n_jobs)
    values = Parallel(n_jobs=n_jobs)(delayed(_cindex_chunk)(c, train, test, risk, tau) for c in chunks)
    values = [v for chunk in values for v in chunk]
    # n_boot = resamples that had an event; it is below the requested number only for very rare tasks.
    return {"cindex": summarize(values, len(values))}


def read_tte_cohort(tte_task_dir) -> pd.DataFrame:
    """TTE labels of one task: subject_id, prediction_time, time_to_event, tte_label (one row per key).

    Uses <task>.parquet when it exists (some folders also hold it next to the split files, so
    reading everything would count every label twice), otherwise all parquet files of the folder.
    """
    tte_task_dir = Path(tte_task_dir)
    whole = tte_task_dir / f"{tte_task_dir.name}.parquet"
    files = [whole] if whole.exists() else sorted(tte_task_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.rename(columns={"time_to_event_days": "time_to_event"})
    df["prediction_time"] = pd.to_datetime(df["prediction_time"]).astype("datetime64[ns]")
    keys = ["subject_id", "prediction_time", "time_to_event", "tte_label"]
    return df[keys].drop_duplicates(subset=["subject_id", "prediction_time"]).reset_index(drop=True)


def train_tte_labels(tte_cohort: pd.DataFrame, split_path):
    """(times, events) of the cohort rows whose subject is in the train split of main_split.csv."""
    split = pd.read_csv(split_path)
    train_ids = split.loc[split["split_name"] == "train", "subject_id"]
    train = tte_cohort[tte_cohort["subject_id"].isin(train_ids)]
    return train["time_to_event"].to_numpy(float), train["tte_label"].to_numpy(bool)


def write_metrics(out_dir, metrics: dict) -> None:
    """Add metrics to <out_dir>/metrics.json (keeps metrics already in the file)."""
    path = Path(out_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(metrics)
    path.write_text(json.dumps(old, indent=2))
    for name, m in metrics.items():
        print(f"{name}: {m['mean']:.4f} ({m['std']:.4f})  [{m['n_boot']} resamples]  -> {path}")
