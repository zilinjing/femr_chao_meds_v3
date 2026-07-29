"""Count the events preceding every held-out prediction point.

Uses the meds_reader database the models themselves read. Events for a subject are
already in ascending time order, so a bisect gives the number of events strictly before
the prediction time. Phenotype cohorts only count events inside a 730-day window.

    python history_counts.py [--num_threads 16] [--check]
"""

import argparse
import bisect
import datetime
import os
from functools import partial

import meds_reader
import pandas as pd

from femr.omop_meds_tutorial.evaluation.stratified.paths import (
    COUNT_DIR,
    FAMILIES,
    MEDS_READER,
    cohort_keys_path,
    counts_path,
)


def _count_for_subject(times, prediction_time, window_days):
    """Events in [pt - window, pt), or [start of record, pt) when window is None."""
    upper = bisect.bisect_left(times, prediction_time)
    if window_days is None:
        return upper
    lower_time = prediction_time - datetime.timedelta(days=window_days)
    return upper - bisect.bisect_left(times, lower_time)


def count_events(subject_rows, window_days=None):
    """map_func for SubjectDatabase.map_with_data.

    Receives (subject, rows) pairs where rows is a list of namedtuples carrying
    subject_id and prediction_time.
    """
    out = []
    for subject, rows in subject_rows:
        times = [e.time for e in subject.events if e.time is not None]
        for row in rows:
            pt = row.prediction_time
            if hasattr(pt, "to_pydatetime"):
                pt = pt.to_pydatetime()
            out.append((subject.subject_id, row.prediction_time,
                        _count_for_subject(times, pt, window_days)))
    return out


def load_keys(family: str, task: str) -> pd.DataFrame:
    path = cohort_keys_path(family, task)
    keys = pd.read_parquet(path, columns=["subject_id", "prediction_time"])
    keys["prediction_time"] = keys["prediction_time"].astype("datetime64[us]")
    keys = keys.drop_duplicates()
    return keys.sort_values("subject_id").reset_index(drop=True)


def compute(db, family: str, task: str, window_days) -> pd.DataFrame:
    keys = load_keys(family, task)
    fn = partial(count_events, window_days=window_days)
    rows = [r for chunk in db.map_with_data(fn, keys, assume_sorted=True) for r in chunk]
    counts = pd.DataFrame(rows, columns=["subject_id", "prediction_time", "n_events"])
    counts["prediction_time"] = counts["prediction_time"].astype("datetime64[us]")
    assert len(counts) == len(keys), f"{family}/{task}: {len(counts)} counts for {len(keys)} keys"
    return counts.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)


def load_event_index():
    """Ranked event times straight off the MEDS parquet shards (cross-check only)."""
    import glob

    import polars as pl

    from femr.omop_meds_tutorial.evaluation.stratified.paths import MEDS_PARQUET

    shards = sorted(glob.glob(f"{MEDS_PARQUET}/*.parquet"))
    return (
        pl.concat([pl.read_parquet(f, columns=["subject_id", "time"]) for f in shards])
        .filter(pl.col("time").is_not_null())
        .sort(["subject_id", "time"])
        .with_columns(pl.int_range(0, pl.len()).over("subject_id").alias("idx"))
        .sort("time")
    )


def cross_check(counts: pd.DataFrame, window_days, ev, n_sample=3000):
    """Independently recompute a sample of counts via a polars asof join."""
    import polars as pl

    sample = counts.head(n_sample)

    def count_before(frame, col):
        # asof on (t - 1us) gives events strictly before t at microsecond resolution.
        probe = frame.with_columns(
            (pl.col(col).cast(pl.Datetime("us")) - pl.duration(microseconds=1)).alias("_k")
        ).sort("_k")
        joined = probe.join_asof(ev, left_on="_k", right_on="time", by="subject_id",
                                strategy="backward")
        return joined.with_columns(
            pl.when(pl.col("idx").is_null()).then(0).otherwise(pl.col("idx") + 1).alias("_c")
        )

    frame = pl.from_pandas(sample[["subject_id", "prediction_time", "n_events"]])
    upper = count_before(frame, "prediction_time").select(
        ["subject_id", "prediction_time", "n_events", pl.col("_c").alias("hi")]
    )
    if window_days is None:
        ref = upper.with_columns(pl.col("hi").alias("ref"))
    else:
        upper = upper.with_columns(
            (pl.col("prediction_time") - pl.duration(days=window_days)).alias("lb")
        )
        lower = count_before(upper, "lb")
        ref = lower.with_columns((pl.col("hi") - pl.col("_c")).alias("ref"))
    bad = ref.filter(pl.col("ref") != pl.col("n_events")).height
    assert bad == 0, f"cross-check failed on {bad}/{ref.height} rows"
    return ref.height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--check", action="store_true",
                        help="cross-check a sample against the MEDS parquet shards")
    parser.add_argument("--check_only", action="store_true",
                        help="skip counting and only cross-check the existing outputs")
    args = parser.parse_args()

    db = None if args.check_only else meds_reader.SubjectDatabase(MEDS_READER,
                                                                  num_threads=args.num_threads)
    ev = load_event_index() if (args.check or args.check_only) else None

    for family, (tasks, _, window_days) in FAMILIES.items():
        os.makedirs(os.path.join(COUNT_DIR, family), exist_ok=True)
        for task in tasks:
            out = counts_path(family, task)
            if args.check_only:
                counts = pd.read_parquet(out)
            else:
                counts = compute(db, family, task, window_days)
                counts.to_parquet(out, index=False)
            desc = counts["n_events"].describe()
            print(f"{family}/{task}: n={len(counts)} median={desc['50%']:.0f} "
                  f"min={desc['min']:.0f} max={desc['max']:.0f} -> {out}")
            if ev is not None:
                n = cross_check(counts, window_days, ev)
                print(f"  cross-check ok on {n} rows")


if __name__ == "__main__":
    main()
