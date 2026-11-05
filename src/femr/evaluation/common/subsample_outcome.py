"""Make a smaller copy of the MIMIC outcome cohort: about 100k training rows per task, full test set.

Whole patients are sampled, because feature generation reads each patient's timeline once however
many prediction times they have. The training patients are shuffled once (same order for every
task); each task keeps patients in that order while its train + tuning rows stay within --train_rows.
held_out.parquet is copied unchanged. Class balance is not touched.

    python -m femr.evaluation.common.subsample_outcome \
        --cohort /shared/share_mala/zilin/storage/ora/cohort/outcome/cohort \
        --split /shared/share_mala/zilin/storage/ora/metadata/main_split.csv \
        --out /shared/share_mala/zilin/storage/ora/cohort/outcome/cohort_100k
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import polars as pl


def shuffled_train_subjects(split_path, seed) -> pl.DataFrame:
    """Training patients in random order: subject_id, rank (0 = first to be kept)."""
    split = pl.read_csv(split_path)
    subjects = np.sort(split.filter(pl.col("split_name") == "train")["subject_id"].unique().to_numpy())
    order = np.random.default_rng(seed).permutation(subjects)
    return pl.DataFrame({"subject_id": order, "rank": np.arange(len(order))})


def kept_subjects(train_rows: pl.DataFrame, order: pl.DataFrame, n_rows: int) -> pl.DataFrame:
    """The first patients in `order` whose rows add up to at most n_rows."""
    counts = train_rows.group_by("subject_id").len().join(order, on="subject_id").sort("rank")
    return counts.filter(pl.col("len").cum_sum() <= n_rows).select("subject_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True, help="folder with <task>/{train,tuning,held_out}.parquet")
    parser.add_argument("--split", required=True, help="main_split.csv")
    parser.add_argument("--out", type=Path, required=True, help="new cohort folder (must not exist)")
    parser.add_argument("--train_rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True)  # refuses to overwrite an existing cohort
    order = shuffled_train_subjects(args.split, args.seed)
    for task_dir in sorted(p for p in args.cohort.iterdir() if p.is_dir()):
        train = pl.read_parquet(task_dir / "train.parquet")
        tuning = pl.read_parquet(task_dir / "tuning.parquet")
        both = pl.concat([train.select("subject_id"), tuning.select("subject_id")])
        keep = kept_subjects(both, order, args.train_rows)

        out_dir = args.out / task_dir.name
        out_dir.mkdir()
        train.join(keep, on="subject_id", how="semi").write_parquet(out_dir / "train.parquet")
        tuning.join(keep, on="subject_id", how="semi").write_parquet(out_dir / "tuning.parquet")
        shutil.copy2(task_dir / "held_out.parquet", out_dir / "held_out.parquet")
        print(f"{task_dir.name}: kept {keep.height:,} of {both['subject_id'].n_unique():,} training patients")


if __name__ == "__main__":
    main()
