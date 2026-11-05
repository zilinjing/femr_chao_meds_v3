"""Linear probe for a regression task (predict the next lab value).

Fits a linear regression (features standardized) on the train split, predicts the test split,
and writes to --out_dir:
    predictions.parquet   subject_id, prediction_time, y_true, y_pred
    metrics.json          r2 as mean/std over 1000 test-set resamples

Normally started by script/lib.sh; see evaluation/README.md.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import femr.featurizers
import femr.splits
from femr.evaluation.common.metrics import N_BOOT, r2_metrics, write_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="task name, e.g. glucose")
    p.add_argument("--features", required=True, help="features .pkl written by common/generate_features.py")
    p.add_argument("--labels", required=True, help="labels .parquet written by common/generate_features.py")
    p.add_argument("--split", required=True, help="main_split.csv (subject_id, split_name)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    args = p.parse_args()

    with open(args.features, "rb") as f:
        features = pickle.load(f)
    labels = pd.read_parquet(args.labels)
    n_labels = len(labels)
    labels = labels[labels.subject_id.isin(features["subject_ids"])]
    if len(labels) < n_labels:
        print(f"{n_labels - len(labels)} labels have no features and are dropped")
    labels = labels.sort_values(["subject_id", "prediction_time"])
    joined = femr.featurizers.join_labels_numerical(features, labels)

    split = femr.splits.SubjectSplit.load_from_csv(args.split)
    train = np.isin(joined["subject_ids"], split.train_subject_ids)
    test = np.isin(joined["subject_ids"], split.test_subject_ids)
    X, y = joined["features"].astype(np.float32), np.asarray(joined["numerical_value"], float)
    print(f"train rows: {train.sum()}, test rows: {test.sum()}")

    model = Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())])
    model.fit(X[train], y[train])

    pred = pd.DataFrame({
        "subject_id": joined["subject_ids"][test],
        "prediction_time": joined["prediction_times"][test],
        "y_true": y[test],
        "y_pred": model.predict(X[test]),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    write_metrics(out_dir, r2_metrics(pred["y_true"], pred["y_pred"], args.n_boot))


if __name__ == "__main__":
    main()
