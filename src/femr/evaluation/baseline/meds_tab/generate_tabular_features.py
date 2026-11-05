"""Generate value-aware tabular features for the strengthened XGBoost / DeepHit baseline.

Mirrors ``femr.evaluation.baseline.count.generate_tabular_features`` but adds
``MedsTabNumericFeaturizer`` to the featurizer list, so numeric events
(laboratory results, vital signs) contribute to the feature matrix instead of
being silently dropped.

Featurizer list::

    AgeFeaturizer(is_normalize=True)          # unchanged
    CountFeaturizer(...)                      # unchanged -- non-numeric codes
    MedsTabNumericFeaturizer(...)             # NEW -- MEDS-Tab aggregations

``CountFeaturizer`` keeps its default flags, so text-valued events remain
excluded. That is a deliberate choice (rebuttal/reviewer3/Q1.md Sec. 5.7) and
must be disclosed in the reply.

Runtime and peak memory are reported at the end of every run; the reviewer asked
for the compute cost of each baseline, and feature extraction is the dominant
CPU cost of the tabular pipeline.

Usage::

    python -u -m femr.evaluation.baseline.meds_tab.generate_tabular_features \
        --pretraining_data "$OUT_DIR" \
        --meds_reader "$MEDS_READER" \
        --cohort_dir "$TASK_DIR"
"""

import datetime
import glob
import json
import os
import pathlib
import pickle
import resource
import time
from typing import List, Optional, Tuple

import meds_reader
import pandas as pd

import femr.featurizers

from femr.evaluation.common.generate_labels import create_omop_meds_tutorial_arg_parser
from .meds_tab_featurizer import MedsTabNumericFeaturizer

# Cohort directories on this cluster hold one parquet per split. Some also hold a
# whole-cohort parquet next to them (e.g. cohort/phenotype/tte/ami/ami.parquet is
# the union of train/tuning/held_out), so globbing "**/*.parquet" would read every
# label twice. Prefer the split files whenever they are present.
SPLIT_FILES = ("train.parquet", "tuning.parquet", "held_out.parquet")


def create_arg_parser():
    parser = create_omop_meds_tutorial_arg_parser()
    parser.add_argument(
        "--cohort_dir",
        dest="cohort_dir",
        required=True,
        help="Directory or file holding the cohort (subject_id, prediction_time, labels).",
    )
    parser.add_argument(
        "--observation_window",
        dest="observation_window",
        type=int,
        default=None,
        help="Optional global truncation of history, in days, passed to CountFeaturizer.",
    )
    parser.add_argument(
        "--windows",
        dest="windows",
        default="7,full",
        help=(
            "Comma-separated nested look-back windows for the numeric featurizer. "
            "Use 'full' for the whole history. Default '7,full'."
        ),
    )
    parser.add_argument(
        "--min_support",
        dest="min_support",
        type=int,
        default=10,
        help=(
            "Minimum number of numeric observations, across the whole labelled cohort, "
            "for a code to receive a column block. 0 keeps every numeric code. "
            "Matches MEDS-Tab's tabularization.min_code_inclusion_count."
        ),
    )
    parser.add_argument("--num_threads", dest="num_threads", type=int, default=32)
    return parser


def parse_windows(spec: str) -> List[Optional[int]]:
    windows: List[Optional[int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        windows.append(None if token in ("full", "none", "all") else int(token.rstrip("d")))
    assert windows, f"Could not parse any window from {spec!r}"
    return windows


def load_cohort(cohort_dir: str) -> Tuple[str, pd.DataFrame, List[str]]:
    """Return (label_name, cohort, files_read)."""
    if not os.path.isdir(cohort_dir):
        label_name = os.path.basename(os.path.splitext(cohort_dir)[0])
        ext = os.path.splitext(cohort_dir)[1].lower()
        if ext == ".parquet":
            return label_name, pd.read_parquet(cohort_dir), [cohort_dir]
        if ext == ".csv":
            return label_name, pd.read_csv(cohort_dir), [cohort_dir]
        raise RuntimeError(f"Unknown file extension: {ext}")

    label_name = os.path.basename(os.path.normpath(cohort_dir))
    split_paths = [
        os.path.join(cohort_dir, f)
        for f in SPLIT_FILES
        if os.path.exists(os.path.join(cohort_dir, f))
    ]
    if split_paths:
        files = split_paths
    else:
        files = sorted(glob.glob(os.path.join(cohort_dir, "**", "*.parquet"), recursive=True))
    assert files, f"No parquet files found under {cohort_dir}"

    cohort = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return label_name, cohort, files


def get_features_name(label_name: str, observation_window: Optional[int] = None) -> str:
    return f"{label_name}_{observation_window}.pkl" if observation_window else f"{label_name}.pkl"


def get_featurizer_name(label_name: str, observation_window: Optional[int] = None) -> str:
    if observation_window:
        return f"{label_name}_{observation_window}_featurizer.pkl"
    return f"{label_name}_featurizer.pkl"


def sort_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Labels in the order the featurizers need: by subject, then by prediction time.

    CountFeaturizer and MedsTabNumericFeaturizer walk a subject's events once while moving through
    its labels, so a label placed after a later one would silently count the events in between,
    which are in its future. The index is reset because map_with_data looks rows up by index label.
    """
    return labels.sort_values(["subject_id", "prediction_time"], kind="stable", ignore_index=True)


def peak_rss_gb() -> float:
    """Peak resident set size of this process and its children, in GiB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_kb, child_kb) / 1024**2


def main() -> None:
    args = create_arg_parser().parse_args()
    windows = parse_windows(args.windows)

    out_root = pathlib.Path(args.pretraining_data)
    features_path = out_root / "features"
    label_path = out_root / "labels"
    features_path.mkdir(exist_ok=True, parents=True)
    label_path.mkdir(exist_ok=True, parents=True)

    label_name, cohort, files = load_cohort(args.cohort_dir)
    if len(cohort) > 0 and isinstance(cohort.prediction_time.iloc[0], datetime.date):
        cohort["prediction_time"] = pd.to_datetime(cohort["prediction_time"])
    cohort.to_parquet(label_path / f"{label_name}.parquet")

    print(f"[{label_name}] {len(cohort)} labels from {len(files)} file(s): "
          + ", ".join(os.path.basename(f) for f in files))
    print(f"[{label_name}] windows={windows} min_support={args.min_support}")

    feature_output_path = features_path / get_features_name(label_name, args.observation_window)
    if feature_output_path.exists():
        print(f"Features already exist at {feature_output_path}; skipping.")
        return

    labels = sort_labels(pd.read_parquet(label_path / f"{label_name}.parquet"))

    t0 = time.perf_counter()
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_threads) as database:
        numeric_featurizer = MedsTabNumericFeaturizer(
            windows_days=windows,
            min_support=args.min_support,
        )
        featurizer = femr.featurizers.FeaturizerList(
            [
                femr.featurizers.AgeFeaturizer(is_normalize=True),
                # Unchanged from the submitted baseline: covers non-numeric codes.
                femr.featurizers.CountFeaturizer(observation_window=args.observation_window),
                numeric_featurizer,
            ]
        )

        print("Preprocessing")
        featurizer.preprocess_featurizers(database, labels)
        t_preprocess = time.perf_counter() - t0
        print(f"Done preprocessing in {t_preprocess:.1f}s. {numeric_featurizer!r}")

        with open(features_path / get_featurizer_name(label_name, args.observation_window), "wb") as f:
            pickle.dump(featurizer, f)

        print("Featurizing")
        t1 = time.perf_counter()
        features = featurizer.featurize(database, labels)
        t_featurize = time.perf_counter() - t1

    matrix = features["features"]
    total_s = time.perf_counter() - t0

    with open(feature_output_path, "wb") as f:
        pickle.dump(features, f)

    sizes = {type(x).__name__: x.get_num_columns() for x in featurizer.featurizers}
    stats = {
        "task": label_name,
        "n_labels": int(matrix.shape[0]),
        "n_columns": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz) / max(1, matrix.shape[0] * matrix.shape[1]),
        "columns_by_featurizer": sizes,
        "numeric_codes": len(numeric_featurizer.code_to_base),
        "windows": [w if w is not None else "full" for w in windows],
        "min_support": args.min_support,
        "num_threads": args.num_threads,
        "preprocess_seconds": round(t_preprocess, 1),
        "featurize_seconds": round(t_featurize, 1),
        "total_seconds": round(total_s, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "dense_float64_gb": round(matrix.shape[0] * matrix.shape[1] * 8 / 1024**3, 2),
    }
    with open(features_path / f"{label_name}_extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("-" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
