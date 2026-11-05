"""Read-only checks to run on a cluster before evaluating: labels, tokenizer, packages, machine.

Prints one report and exits with 1 if any line starts with PROBLEM. Started by
script/cumc/preflight.sh; see evaluation/README.md. Run it again after the MEDS-Tab extraction to
see the size of every feature matrix.
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from femr.evaluation.baseline.meds_tab.generate_tabular_features import load_cohort

PACKAGES = ["femr", "meds_reader", "xgboost", "sksurv", "pycox", "torchtuples", "meds_evaluation", "polars"]
SPLIT_FILES = {"train.parquet", "tuning.parquet", "held_out.parquet"}
MB_PER_SUBJECT = 0.5  # size of the saved batches on MIMIC (0.47-0.54 over 4 tasks)
# Every reader worker of generate_features held 5.7 GB of private memory on MIMIC, whose ontology.pkl
# is 0.81 GB, even on a tiny cohort. Scaling this with the ontology size is an assumption.
WORKER_GB_PER_ONTOLOGY_GB = 7
READER_WORKERS = 24  # fixed in generate_features.py, one set per family

problems = []


def problem(message: str) -> None:
    problems.append(message)
    print(f"PROBLEM: {message}")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def check_packages() -> None:
    print("== packages")
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'ok')}")
        except ImportError as e:
            problem(f"cannot import {name}: {e}")
    print(f"femr is loaded from {importlib.import_module('femr').__file__}: this must be your new clone")


def check_metadata(meta_dir: Path, models_root: Path) -> pd.DataFrame:
    print("== metadata")
    for name in ("ontology.pkl", "main_split.csv", "tokenizer/dictionary.msgpack"):
        if not (meta_dir / name).exists():
            problem(f"missing {meta_dir / name}")
    if (meta_dir / "ontology.pkl").exists():
        size = (meta_dir / "ontology.pkl").stat().st_size / 1024**3
        print(f"ontology.pkl: {size:.2f} GB; batches.sh all needs about "
              f"{3 * READER_WORKERS * WORKER_GB_PER_ONTOLOGY_GB * size:.0f} GB of RAM for its reader workers, "
              f"plus its batch workers (not measured, assume the same per worker)")
    split = pd.read_csv(meta_dir / "main_split.csv")
    counts = split["split_name"].value_counts().to_dict()
    print(f"split: {counts}")
    if set(counts) != {"train", "test"}:
        problem(f"split_name must be train/test, found {sorted(counts)}: the baselines would get no test rows")

    tokenizer = meta_dir / "tokenizer/dictionary.msgpack"
    if tokenizer.exists():
        shared = md5(tokenizer)
        print(f"shared tokenizer: {shared}")
        for path in sorted(models_root.glob("*/transformer/best_model/dictionary.msgpack")):
            same = md5(path) == shared
            print(f"{path.parents[2].name}: {'same tokenizer' if same else md5(path)}")
            if not same:
                problem(f"{path} differs from the shared tokenizer: this model cannot use the shared batches")
    return split


def check_task(kind: str, task_dir: Path, need: list, split: pd.DataFrame, tte_col: str) -> int:
    """Check one task folder; returns its number of subjects."""
    files = sorted(p.name for p in task_dir.glob("*.parquet"))
    nested = [p for p in task_dir.glob("*/**/*.parquet")]
    if not files or nested:
        problem(f"{kind}/{task_dir.name}: parquet files must sit directly in the folder (found {files}, {nested[:2]})")
        return 0
    # generate_features reads every parquet file, so a whole-task file next to the split files
    # would double the labels. The TTE cohort is read by code that handles this.
    if kind != "tte" and f"{task_dir.name}.parquet" in files and len(files) > 1:
        problem(f"{kind}/{task_dir.name}: whole-task file next to other files, labels would be read twice: {files}")
    if kind == "regression" and f"{task_dir.name}.parquet" not in files:
        problem(f"regression/{task_dir.name}: the most_recent baseline reads {task_dir.name}.parquet, found {files}")

    _, df, _ = load_cohort(str(task_dir))
    for column in need:
        if column not in df.columns:
            problem(f"{kind}/{task_dir.name}: no column {column}; columns are {list(df.columns)}")
    if kind != "regression" and "split_name" in df.columns:  # regression does not go through run_all.py
        problem(f"{kind}/{task_dir.name}: a split_name column breaks labels.join(split) in run_all.py")

    subjects = df["subject_id"].nunique()
    in_test = df["subject_id"].isin(split.loc[split["split_name"] == "test", "subject_id"]).sum()
    unknown = (~df["subject_id"].isin(split["subject_id"])).sum()
    line = f"{kind}/{task_dir.name}: {len(df)} rows, {subjects} subjects, {in_test} test rows, files {files}"
    if "boolean_value" in need and "boolean_value" in df.columns:
        line += f", positive {df['boolean_value'].mean():.4f}"
    if kind == "tte" and {tte_col, "tte_label"} <= set(df.columns):
        times = pd.to_numeric(df[tte_col], errors="coerce")
        line += (f", {tte_col} {df[tte_col].dtype} min/median/max {times.min():.1f}/{times.median():.1f}/"
                 f"{times.max():.1f}, events {df['tte_label'].mean():.4f}")
    print(line)
    if unknown:
        print(f"  note: {unknown} rows have a subject outside main_split.csv and are not used")
    if in_test == 0:
        problem(f"{kind}/{task_dir.name}: no test rows")
    return subjects


def check_cohorts(args, split: pd.DataFrame) -> None:
    print("== cohorts")
    binary, label = ["subject_id", "prediction_time", "boolean_value"], ["subject_id", "prediction_time"]
    groups = {
        "outcome": (args.outcome_cohort, binary),
        "binary": (args.binary_cohort, binary),
        "tte": (args.tte_cohort, label + ["tte_label", args.tte_col]),
    }
    names, subjects = {}, 0
    for kind, (root, need) in groups.items():
        names[kind] = sorted(p.name for p in root.iterdir() if p.is_dir())
        for name in names[kind]:
            count = check_task(kind, root / name, need, split, args.tte_col)
            subjects += 0 if kind == "tte" else count
    for name in args.regression_tasks.split(","):
        subjects += check_task("regression", args.regression_cohort / name, label + ["numerical_value"],
                               split, args.tte_col)

    if names["binary"] != names["tte"]:
        problem(f"phenotype folders differ: binary {names['binary']}, tte {names['tte']}")
    if set(names["outcome"]) & set(names["binary"]):
        problem(f"same task name in outcome and phenotype: {set(names['outcome']) & set(names['binary'])}")
    print(f"saved batches, estimate: {subjects * MB_PER_SUBJECT / 1024:.0f} GB (one backbone)")


def check_existing(args) -> None:
    print("== existing outputs (these are reused, not rerun; move them away to rerun)")
    found = list(args.meds_tab_dir.glob("*/features/*.pkl")) + list(args.meds_tab_dir.glob("results/*.pickle"))
    for row in ("xgboost", "deephit", "most_recent"):
        found += list(args.result_root.glob(f"*/*/{row}/metrics.json"))
    print("\n".join(str(p) for p in sorted(found)) or "none")

    stats = sorted(args.meds_tab_dir.glob("*/features/*_extraction_stats.json"))
    if stats:
        print("== MEDS-Tab features (DeepHit peak RAM per task is about dense_float64_gb)")
    for path in stats:
        s = json.loads(path.read_text())
        print(f"{path.parents[1].name}: {s['n_labels']} rows, {s['n_columns']} columns, "
              f"nnz {s['nnz']}, dense_float64_gb {s['dense_float64_gb']}, took {s['total_seconds']:.0f} s")


def check_machine(args) -> None:
    print("== machine")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) / 1024**2
    print(f"cpus {os.cpu_count()}, memory available {available:.0f} GB")
    for path in (args.hf_cache, args.meds_tab_dir, args.result_root):
        existing = next(p for p in [path, *path.parents] if p.exists())
        print(f"disk free for {path}: {shutil.disk_usage(existing).free / 1024**3:.0f} GB")
    if shutil.which("nvidia-smi"):
        query = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"]
        print(subprocess.run(query, capture_output=True, text=True).stdout.strip())
    else:
        print("no GPU found")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("meta_dir", "models_root", "result_root", "hf_cache", "meds_tab_dir", "outcome_cohort",
                 "binary_cohort", "tte_cohort", "regression_cohort"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--regression_tasks", required=True, help="comma-separated task folders")
    p.add_argument("--tte_col", required=True, help="time column of the TTE labels")
    args = p.parse_args()

    check_packages()
    split = check_metadata(args.meta_dir, args.models_root)
    check_cohorts(args, split)
    check_existing(args)
    check_machine(args)
    print(f"== {len(problems)} problem(s)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
