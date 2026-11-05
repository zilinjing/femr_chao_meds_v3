"""Probes for a phenotype task: binary (logistic regression) + time-to-event (linear DeepHit head).

Both use the same frozen features. Writes to --out_dir:
    predictions.parquet       binary test predictions (see outcome/probe.py)
    tte_predictions.parquet   subject_id, prediction_time, time_to_event, tte_label, risk
                              (risk = 1 - predicted survival at the truncation day)
    metrics.json              auroc, auprc, cindex as mean/std over 1000 test-set resamples

The C-index is the IPCW C-index up to --truncation days (300 for MIMIC, 365 for CUMC).

Normally started by script/lib.sh; see evaluation/README.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from sklearn.model_selection import ParameterGrid

from femr.evaluation.common.metrics import cindex_metrics, read_tte_cohort, train_tte_labels, write_metrics
from femr.evaluation.outcome.probe import arg_parser, run_binary


def deephit_probe(train, test, truncation):
    """Linear DeepHit head; hyperparameters chosen on a held-out part of train. Returns test risk."""
    np.random.seed(42)
    torch.manual_seed(42)

    # 80% fit, 10% early stopping, 10% hyperparameter selection
    val = train.sample(frac=0.2, replace=False, random_state=42)
    fit = train.drop(val.index)
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    val = val.drop(hyper.index)

    labtrans = DeepHitSingle.label_transform(100)  # 100 discrete time bins
    x = lambda df: np.stack(df["features"]).astype(np.float32)
    y = lambda df: (df["time_to_event"].to_numpy(), df["tte_label"].to_numpy())
    y_fit = labtrans.fit_transform(*y(fit))
    val_data = (x(val), labtrans.transform(*y(val)))
    x_hyper, y_hyper = x(hyper), labtrans.transform(*y(hyper))

    best_nll, best_model, best_params = np.inf, None, None
    for params in ParameterGrid({"dropout": [0, 0.1], "lr": [1e-3, 1e-4], "batch": [1024, 2048]}):
        net = tt.practical.MLPVanilla(x(fit).shape[1], [], labtrans.out_features, True, params["dropout"])
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
        model.optimizer.set_lr(params["lr"])
        model.fit(x(fit), y_fit, params["batch"], 1000, [tt.callbacks.EarlyStopping()], val_data=val_data)
        nll = model.score_in_batches(x_hyper, y_hyper)["loss"]
        if nll < best_nll:
            best_nll, best_model, best_params = nll, model, params
    print(f"DeepHit best params: {best_params}, NLL {best_nll:.4f}")

    surv = best_model.predict_surv_df(x(test))  # rows: time bins, columns: test rows
    row = int(np.argmin(np.abs(surv.index.to_numpy() - truncation)))
    print(f"risk read at the time bin closest to {truncation} days: {surv.index[row]:.1f} days")
    return 1.0 - surv.iloc[row].to_numpy()


def main():
    p = arg_parser()
    p.add_argument("--tte_dir", required=True, help="TTE cohort folder of this task")
    p.add_argument("--truncation", type=float, required=True, help="C-index horizon in days")
    p.add_argument("--n_jobs", type=int, default=16, help="CPU workers for the C-index resamples")
    args = p.parse_args()

    train, test = run_binary(args)

    # Same features, TTE labels: keep the rows that are also in the TTE cohort.
    cohort = read_tte_cohort(args.tte_dir)
    keys = ["subject_id", "prediction_time"]
    train = train.merge(cohort, on=keys).sort_values(keys).reset_index(drop=True)
    test = test.merge(cohort, on=keys).sort_values(keys).reset_index(drop=True)
    print(f"TTE train rows: {len(train)}, test rows: {len(test)}")

    pred = test[keys + ["time_to_event", "tte_label"]].copy()
    pred["risk"] = deephit_probe(train, test, args.truncation)
    pred.to_parquet(Path(args.out_dir) / "tte_predictions.parquet", index=False)

    train_times, train_events = train_tte_labels(cohort, args.split)
    write_metrics(args.out_dir, cindex_metrics(pred, train_times, train_events, args.truncation,
                                               args.n_boot, args.n_jobs))


if __name__ == "__main__":
    main()
