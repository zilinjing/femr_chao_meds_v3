"""Turn saved embeddings of a model without a local checkpoint into the input of outcome/probe.py.

Reads <parquet_dir>/{train,test}.parquet (subject_id, prediction_time, boolean_value, features) and
keeps the rows of the cohort. Writes, in the layout of common/generate_features.py:
    <output_root>/<tag>/features/<task>_<tag>.pkl    subject_ids, feature_times, features (float32)
    <output_root>/<tag>/labels/<task>.parquet        the cohort's labels

Stops if a cohort row has no embedding, a label differs, or float32 would change a value.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ["subject_id", "prediction_time"]


def read_keyed(paths) -> pl.DataFrame:
    """Concatenate parquet files, with prediction_time and boolean_value in one common type."""
    frames = [pl.read_parquet(p) for p in paths]
    return pl.concat([
        f.select(*KEYS, "boolean_value", *(["features"] if "features" in f.columns else []))
        .with_columns(pl.col("prediction_time").cast(pl.Datetime("ns")), pl.col("boolean_value").cast(pl.Boolean))
        for f in frames
    ])


def import_features(parquet_dir, cohort_dir, task, tag, output_root) -> None:
    labels = read_keyed(sorted(Path(cohort_dir).glob("*.parquet")))
    saved = read_keyed([Path(parquet_dir) / "train.parquet", Path(parquet_dir) / "test.parquet"])
    joined = labels.join(saved, on=KEYS, how="left", suffix="_saved", validate="1:1")

    missing = joined["boolean_value_saved"].null_count()
    if missing:
        raise ValueError(f"{missing} of {labels.height} cohort rows have no embedding")
    different = joined.filter(pl.col("boolean_value") != pl.col("boolean_value_saved")).height
    if different:
        raise ValueError(f"{different} labels differ between the cohort and the embeddings")

    features = np.stack(joined["features"].to_numpy())
    as_float32 = features.astype(np.float32)
    if not np.array_equal(as_float32, features):
        raise ValueError("embeddings are not exactly representable in float32")

    out = Path(output_root) / tag
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    with open(out / "features" / f"{task}_{tag}.pkl", "wb") as f:
        pickle.dump({
            "subject_ids": joined["subject_id"].to_numpy(),
            "feature_times": joined["prediction_time"].to_numpy(),
            "features": as_float32,
        }, f)
    labels.write_parquet(out / "labels" / f"{task}.parquet")
    print(f"{task}/{tag}: {labels.height} rows of {saved.height}, {features.shape[1]} features")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet_dir", required=True, help="folder with train.parquet and test.parquet")
    p.add_argument("--cohort_dir", required=True, help="folder with the task's cohort parquet files")
    p.add_argument("--task", required=True)
    p.add_argument("--tag", required=True, help="row name in the results tables")
    p.add_argument("--output_root", required=True, help="<RESULT_ROOT>/outcome/_features")
    args = p.parse_args()
    import_features(args.parquet_dir, args.cohort_dir, args.task, args.tag, args.output_root)
