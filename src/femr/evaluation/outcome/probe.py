"""Linear probe for a binary task (outcome tasks, and the binary half of phenotype tasks).

Fits a logistic regression on the frozen model's features of the train split, predicts the test
split, and writes to --out_dir:
    predictions.parquet   subject_id, prediction_time, boolean_value, predicted_boolean_probability
    metrics.json          auroc, auprc as mean/std over 1000 test-set resamples (meds-evaluation)

Normally started by script/lib.sh; see evaluation/README.md.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

import femr.featurizers
import femr.splits
from femr.evaluation.common.metrics import N_BOOT, binary_metrics, write_metrics


def load_train_test(features_path, labels_path, split_path, task):
    """Join each label to the model's features and split by main_split.csv.

    Returns (train, test) DataFrames: subject_id, prediction_time, boolean_value, features.
    """
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    labels = pd.read_parquet(labels_path)
    n_labels = len(labels)
    labels = labels[labels.subject_id.isin(features["subject_ids"])]
    if len(labels) < n_labels:
        print(f"{n_labels - len(labels)} labels have no features and are dropped")
    labels = labels.sort_values(["subject_id", "prediction_time"])
    joined = femr.featurizers.join_labels(features, labels, task)

    split = femr.splits.SubjectSplit.load_from_csv(split_path)
    frames = []
    for subject_ids in (split.train_subject_ids, split.test_subject_ids):
        mask = np.isin(joined["subject_ids"], subject_ids)
        frames.append(pd.DataFrame({
            "subject_id": joined["subject_ids"][mask],
            "prediction_time": pd.to_datetime(joined["prediction_times"][mask]).astype("datetime64[ns]"),
            "boolean_value": joined["boolean_values"][mask].astype(bool),
            "features": list(joined["features"][mask]),
        }))
    train, test = frames
    print(f"train rows: {len(train)}, test rows: {len(test)}")
    return train, test


def logistic_probe(train, test) -> pd.DataFrame:
    """Logistic regression (C chosen by 5-fold CV on AUROC); returns the test predictions."""
    train = train.sample(frac=1.0, random_state=42)  # shuffle before the CV folds
    model = LogisticRegressionCV(scoring="roc_auc")
    model.fit(np.stack(train["features"]), train["boolean_value"])
    pred = test[["subject_id", "prediction_time", "boolean_value"]].copy()
    pred["predicted_boolean_probability"] = model.predict_proba(np.stack(test["features"]))[:, 1]
    return pred


def run_binary(args):
    train, test = load_train_test(args.features, args.labels, args.split, args.task)
    pred = logistic_probe(train, test)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    write_metrics(out_dir, binary_metrics(pred, args.n_boot))
    return train, test


def arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="task name, e.g. readmission")
    p.add_argument("--features", required=True, help="features .pkl written by common/generate_features.py")
    p.add_argument("--labels", required=True, help="labels .parquet written by common/generate_features.py")
    p.add_argument("--split", required=True, help="main_split.csv (subject_id, split_name)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    return p


if __name__ == "__main__":
    run_binary(arg_parser().parse_args())
