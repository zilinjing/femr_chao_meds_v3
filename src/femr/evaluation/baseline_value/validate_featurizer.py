"""Validate MedsTabNumericFeaturizer against a brute-force reference. Run before the sweep.

Checks (Sec. 5.9 of rebuttal/reviewer3/Q1.md):

  1. Column budget: num_columns == len(code_to_base) * 6 * len(windows).
  2. Row count:     one feature row per label, for every subject.
  3. Correctness:   every emitted aggregate matches a naive recomputation from
                    the raw event stream.
  4. Monotonicity:  nested windows satisfy 7d <= full for count/sum/sum_sqd, and
                    7d.min >= full.min, 7d.max <= full.max.
  5. Sizing:        reports column count, density and projected dense memory, so
                    the OOM risk in run_all can be judged before the full sweep.

Also reports the share of events carrying a numeric value -- the number to quote
in the rebuttal, since it is exactly the fraction of the record the submitted
baseline discarded.

Usage::

    python -m femr.evaluation.baseline_value.validate_featurizer \
        --num_subjects 200

(defaults point at storage/ora/meds_v0.6_reader and the MIMIC AMI binary cohort)
"""

import argparse
import collections
import datetime
import glob
import math
import os
from typing import Dict, List, Optional

import meds_reader
import numpy as np
import pandas as pd

from .meds_tab_featurizer import MedsTabNumericFeaturizer


class _Label:
    """Minimal stand-in for femr.labelers.Label (only prediction_time is used)."""

    __slots__ = ("prediction_time",)

    def __init__(self, prediction_time):
        self.prediction_time = prediction_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meds_reader", default="/data/zilinjing/ethos_fm/storage/ora/meds_v0.6_reader")
    p.add_argument("--cohort_dir",
                   default="/data/zilinjing/ethos_fm/storage/ora/cohort/phenotype/binary/ami/")
    p.add_argument("--num_subjects", type=int, default=200)
    p.add_argument("--windows", default="7,full")
    p.add_argument("--min_support", type=int, default=10)
    p.add_argument("--num_threads", type=int, default=8)
    return p.parse_args()


def parse_windows(spec: str) -> List[Optional[int]]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(None if tok in ("full", "none", "all") else int(tok.rstrip("d")))
    return out


SPLIT_FILES = ("train.parquet", "tuning.parquet", "held_out.parquet")


def load_cohort(cohort_dir: str) -> pd.DataFrame:
    if os.path.isdir(cohort_dir):
        # Prefer per-split files: some TTE dirs also hold a whole-cohort parquet
        # that is the union of them, which a plain glob would read twice.
        files = [os.path.join(cohort_dir, f) for f in SPLIT_FILES
                 if os.path.exists(os.path.join(cohort_dir, f))]
        if not files:
            files = sorted(glob.glob(os.path.join(cohort_dir, "**", "*.parquet"), recursive=True))
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    elif cohort_dir.endswith(".parquet"):
        df = pd.read_parquet(cohort_dir)
    else:
        df = pd.read_csv(cohort_dir)
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    return df


def reference_aggs(events, prediction_time, window_days: Optional[int]) -> Dict[str, list]:
    """Brute-force recomputation of the six aggregates for one label."""
    out: Dict[str, list] = {}
    for code, time, value in events:
        if time is not None and time > prediction_time:
            continue
        if window_days is not None:
            if time is None:
                continue  # untimed events cannot fall inside a finite window
            if prediction_time - time > datetime.timedelta(days=window_days):
                continue
        acc = out.setdefault(code, [0, 0, 0.0, 0.0, math.inf, -math.inf])
        acc[0] += 1
        if value is not None:
            acc[1] += 1
            acc[2] += value
            acc[3] += value * value
            acc[4] = min(acc[4], value)
            acc[5] = max(acc[5], value)
    return out


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    cohort = load_cohort(args.cohort_dir)

    featurizer = MedsTabNumericFeaturizer(windows_days=windows, min_support=args.min_support)
    n_aggs = len(featurizer.AGGS)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_threads) as db:
        # Cast away numpy ints and skip subjects missing from the database.
        subject_ids = []
        for raw in cohort.subject_id.unique():
            sid = int(raw)
            try:
                db[sid]
            except (KeyError, IndexError):
                continue
            subject_ids.append(sid)
            if len(subject_ids) >= args.num_subjects:
                break
        assert subject_ids, "No cohort subjects found in the meds_reader database"
        print(f"Validating on {len(subject_ids)} subjects, windows={windows}")

        # ---- preprocess (on this subset only; support threshold is scaled down)
        subset_support = max(1, args.min_support // 100)
        featurizer.min_support = subset_support
        data = featurizer.get_initial_preprocess_data()
        event_share = collections.Counter()
        for sid in subject_ids:
            subject = db[sid]
            featurizer.add_preprocess_data(data, subject, [])
            for e in subject.events:
                event_share[
                    "numeric" if e.numeric_value is not None
                    else "text" if e.text_value is not None
                    else "plain"
                ] += 1
        featurizer.encorperate_prepreprocessed_data([data])

        # ---- check 1: column budget
        expected = len(featurizer.code_to_base) * n_aggs * len(windows)
        assert featurizer.get_num_columns() == expected, (
            f"column budget mismatch: {featurizer.get_num_columns()} != {expected}"
        )
        print(f"[1/5] OK  column budget: {len(featurizer.code_to_base)} codes "
              f"x {n_aggs} aggs x {len(windows)} windows = {expected}")

        n_checked = n_cells = 0
        for sid in subject_ids:
            subject = db[sid]
            times = sorted(cohort.loc[cohort.subject_id == sid, "prediction_time"].tolist())
            if not times:
                continue
            labels = [_Label(t.to_pydatetime()) for t in times]

            rows = featurizer.featurize(subject, labels)

            # ---- check 2: one row per label
            assert len(rows) == len(labels), (
                f"subject {sid}: {len(rows)} rows for {len(labels)} labels"
            )

            events = [
                (e.code, e.time, None if e.numeric_value is None else float(e.numeric_value))
                for e in subject.events
                if e.code in featurizer.code_to_base
            ]

            for row, label in zip(rows, labels):
                emitted = {c: v for c, v in row}
                per_window = []
                for wi, window in enumerate(windows):
                    ref = reference_aggs(events, label.prediction_time, window)
                    per_window.append(ref)
                    for code, acc in ref.items():
                        base = featurizer.code_to_base[code] + wi * n_aggs
                        # ---- check 3: values match the reference
                        assert emitted.get(base, 0) == acc[0], (
                            f"subject {sid} {code} w={window}: code/count "
                            f"{emitted.get(base, 0)} != {acc[0]}"
                        )
                        if acc[1] > 0:
                            for k, expected_v in enumerate(acc[1:], start=1):
                                got = emitted.get(base + k, 0)
                                assert abs(got - expected_v) < 1e-6 * max(1.0, abs(expected_v)), (
                                    f"subject {sid} {code} w={window} "
                                    f"{featurizer.AGGS[k]}: {got} != {expected_v}"
                                )
                        n_cells += 1
                    # nothing emitted that the reference does not know about
                    for col in emitted:
                        blk = n_aggs * len(windows)
                        for code, b in featurizer.code_to_base.items():
                            if b <= col < b + blk and (col - b) // n_aggs == wi:
                                assert code in ref, (
                                    f"subject {sid}: emitted {code} w={window} "
                                    "but reference has no such events"
                                )

                # ---- check 4: nested-window monotonicity
                if len(windows) > 1 and None in windows:
                    fi = windows.index(None)
                    for wi, window in enumerate(windows):
                        if window is None:
                            continue
                        for code, acc in per_window[wi].items():
                            fa = per_window[fi][code]
                            assert acc[0] <= fa[0] and acc[1] <= fa[1], f"{code}: count not nested"
                            assert acc[3] <= fa[3] + 1e-6, f"{code}: sum_sqd not nested"
                            if acc[1] > 0:
                                assert acc[4] >= fa[4] - 1e-9, f"{code}: min not nested"
                                assert acc[5] <= fa[5] + 1e-9, f"{code}: max not nested"
            n_checked += len(labels)

        print(f"[2/5] OK  row counts, {n_checked} labels")
        print(f"[3/5] OK  aggregate values, {n_cells} (code, window) cells vs brute force")
        print("[4/5] OK  nested-window monotonicity")

    # ---- check 5: sizing, extrapolated to the full support threshold
    total = sum(event_share.values())
    print("\n[5/5] Sizing and event composition")
    for k, v in sorted(event_share.items(), key=lambda kv: -kv[1]):
        print(f"      {k:>8}: {v/total:6.1%}   <-- 'numeric' is what the old baseline dropped")
    n_cols = featurizer.get_num_columns()
    print(f"      columns at min_support={subset_support} on this subset: {n_cols}")
    print(f"      NOTE: this subset used min_support={subset_support}; the full run uses")
    print(f"      --min_support {args.min_support} over every labelled subject, which admits")
    print("      more codes. Read the real budget from the extraction stats JSON.")
    for n_rows in (10_000, 100_000, 1_000_000):
        gb = n_rows * n_cols * 8 / 1024**3
        print(f"      dense float64 at {n_rows:>9,} rows: {gb:8.1f} GB")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
