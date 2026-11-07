"""Saved parquet embeddings must reach outcome/probe.py row for row."""
import datetime

import numpy as np
import pandas as pd
import pytest

from femr.evaluation.outcome.import_parquet_features import import_features
from femr.evaluation.outcome.probe import load_train_test

TIME = datetime.datetime(2100, 1, 1, 12, 30)
DAY = datetime.timedelta(days=1)
# subject, time, label, embedding. Subject 1 has two rows; the last row of each file is not in the cohort.
TRAIN = [(1, TIME, 1, [1.0, 2.0]), (1, TIME + DAY, 0, [3.0, 4.0]), (2, TIME, 0, [5.0, 6.0]), (3, TIME, 1, [7.0, 8.0])]
TEST = [(8, TIME, 1, [9.0, 10.0]), (9, TIME, 0, [11.0, 12.0])]


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["subject_id", "prediction_time", "boolean_value", "features"])


def write_inputs(tmp_path, train=TRAIN, test=TEST):
    """Embeddings as the old runs saved them (ns, int labels), a cohort as saved on disk (us, bool labels)."""
    saved, cohort = tmp_path / "saved", tmp_path / "cohort"
    saved.mkdir()
    cohort.mkdir()
    for name, rows in (("train", train), ("test", test)):
        df = frame(rows)
        df["prediction_time"] = df["prediction_time"].astype("datetime64[ns]")
        df.to_parquet(saved / f"{name}.parquet")
    for name, rows in (("train", TRAIN[:2]), ("tuning", TRAIN[2:3]), ("held_out", TEST[:1])):
        df = frame(rows).drop(columns="features")
        df["prediction_time"] = df["prediction_time"].astype("datetime64[us]")
        df["boolean_value"] = df["boolean_value"].astype(bool)
        df.to_parquet(cohort / f"{name}.parquet")
    return saved, cohort


def test_cohort_rows_reach_the_probe_with_their_own_embedding(tmp_path):
    saved, cohort = write_inputs(tmp_path)
    root = tmp_path / "features"
    import_features(saved, cohort, "task", "tag", root)
    split = tmp_path / "split.csv"
    pd.DataFrame({"subject_id": [1, 2, 3, 8, 9], "split_name": ["train"] * 3 + ["test"] * 2}).to_csv(split, index=False)

    train, test = load_train_test(
        root / "tag" / "features" / "task_tag.pkl", root / "tag" / "labels" / "task.parquet", split, "task"
    )

    assert train["subject_id"].tolist() == [1, 1, 2]  # train + tuning; subject 3 is not in the cohort
    assert train["prediction_time"].tolist() == [TIME, TIME + DAY, TIME]
    assert train["boolean_value"].tolist() == [True, False, False]
    assert np.stack(train["features"]).tolist() == [[1, 2], [3, 4], [5, 6]]
    assert np.stack(train["features"]).dtype == np.float32  # as the other models' features
    assert test["subject_id"].tolist() == [8]
    assert test["features"].iloc[0].tolist() == [9, 10]


def test_cohort_row_without_embedding_stops(tmp_path):
    saved, cohort = write_inputs(tmp_path, train=TRAIN[1:])
    with pytest.raises(ValueError, match="1 of 4 cohort rows have no embedding"):
        import_features(saved, cohort, "task", "tag", tmp_path / "features")


def test_label_mismatch_stops(tmp_path):
    saved, cohort = write_inputs(tmp_path, train=[(1, TIME, 0, [1.0, 2.0])] + TRAIN[1:])
    with pytest.raises(ValueError, match="1 labels differ"):
        import_features(saved, cohort, "task", "tag", tmp_path / "features")


def test_value_that_float32_would_change_stops(tmp_path):
    saved, cohort = write_inputs(tmp_path, train=[(1, TIME, 1, [0.1, 2.0])] + TRAIN[1:])
    with pytest.raises(ValueError, match="not exactly representable"):
        import_features(saved, cohort, "task", "tag", tmp_path / "features")
