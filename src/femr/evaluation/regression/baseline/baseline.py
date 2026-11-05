"""
Most Recent Imputation Baseline for Regression Tasks

Logic:
1. For each test sample, find the most recent value of the target code BEFORE prediction_time
2. Exclude zero values from consideration
3. If no historical value exists, use per-code mean from training data as fallback

Usage:
python baseline.py \
    --cohort_dir /shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/ \
    --meds_path /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
    --dataset mimic \
    --split_path /user/zj2398/cache/mtpp_8k_T_8_V_10/main_split.csv  \
    --output_dir /shared/share_mala/zj2398/mimic/task/regression

python baseline.py \
    --cohort_dir /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
    --meds_path /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
    --dataset cumc \
    --split_path /data2/processed_datasets/zj2398/femr/cumc/motor/main_split.csv \
    --output_dir /data/results/zj2398/femr/cumc/regression

This script now does two things in one pass:
1. It computes the most-recent-value baseline predictions.
2. It writes those predictions in the same regression format used by the
   learned models so downstream hypothesis tests can read them directly.

Important alignment detail:
- We build the saved prediction parquet directly from the cohort test rows.
- This preserves the exact row multiplicity from the cohort, including cases
  where the same (subject_id, prediction_time) appears more than once with
  different labels.
- That is why the saved file can be checked against the cohort labels without
  losing rows or collapsing duplicates.
"""

import argparse
import functools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import meds_reader
import numpy as np
import pandas as pd

from femr.evaluation.common.metrics import r2_metrics, write_metrics

# ============================================================================
# Constants
# ============================================================================
MODEL_NAME = "most_recent"  # row name in regression_r2.csv
DEFAULT_OUTPUT_DIR = Path("/shared/share_mala/zj2398/mimic/task/regression")

MIMIC_CODE_DICT = {
    "platelets": "LAB//51265//K/uL",
    "creatinine": "LAB//50912//mg/dL",
    "bilirubin": "LAB//50885//mg/dL",
    "pao2":"LAB//50821//mm Hg",
    "glucose":"LAB//50931//mg/dL",
    "lactate":"LAB//50813//mmol/L"
}

# code_dict = {
#    "LAB//51265//K/uL": "Platelets",
#    "LAB//50912//mg/dL": "Creatinine",
#    "LAB//50821//mm Hg": "Spo2",
#    "LAB//50931//mg/dL":"Glucose",
#    "LAB//50813//mmol/L":"Lactate"
# }

# MIMIC_CODE_DICT = {
#     "MAP": "LAB//220052//mmHg",
#     "lactate":"LAB//50813//mmol/L",
#     "urine":"SUBJECT_FLUID_OUTPUT//226559//ml",
# }


CUMC_CODE_DICT = {
    "platelets": "LOINC/26515-7//UCUM/10*9/L",
    "creatinine": "LOINC/2160-0//UCUM/mg/dL",
    # "bilirubin": "LOINC/1968-7//UCUM/mg/dL",
    "spo2":"LOINC/2708-6//UCUM/%",
    "glucose":"LOINC/2339-0//UCUM/mg/dL"
}

# ============================================================================
# Data Loading
# ============================================================================
def load_data(cohort_dir: Path, split_path: Path, task: str,CODE_DICT: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a single task cohort and split it into train/test rows.

    The cohort parquet is the label source of truth. We keep its test rows
    intact so the final prediction parquet has the same row structure as the
    cohort labels used by the other regression pipelines.
    """
    cohort_df = pd.read_parquet(cohort_dir / task / f"{task}.parquet")
    split_df = pd.read_csv(split_path).rename(columns={"split_name": "split"})
    merged = cohort_df.merge(split_df, on="subject_id", how="inner")
    
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)
    train_df = merged[merged["split"] != "test"]
    train_df["code"] = CODE_DICT[task]
    return train_df, test_df


def compute_code_mean(train_df: pd.DataFrame, target_code: str) -> float:
    """
    Compute the fallback value used when no valid history exists.

    We estimate one mean per target code from the non-test rows, excluding
    explicit zeros and missing values, matching the baseline's original intent.
    """
    mask = (
        (train_df["code"] == target_code) & 
        (train_df["numerical_value"] != 0) & 
        (train_df["numerical_value"].notna())
    )
    code_data = train_df.loc[mask, "numerical_value"]
    
    if len(code_data) <= 0:
        raise ValueError("no valid data")
    return code_data.mean()




def build_test_lookup(test_df: pd.DataFrame) -> Dict[int, List[Tuple[int, pd.Timestamp]]]:
    """
    Build subject_id -> [(row_idx, prediction_time), ...] mapping.

    We store the row index from the cohort test DataFrame rather than only the
    key columns. This lets us write predictions back to the exact original test
    rows, including duplicated (subject_id, prediction_time) pairs.
    """
    lookup = defaultdict(list)
    for idx, row in test_df.iterrows():
        lookup[row["subject_id"]].append((idx, row["prediction_time"]))
    return dict(lookup)


# ============================================================================
# Core Prediction Logic
# ============================================================================
def process_subjects(
    subject_iter,
    target_code: str,
    test_lookup: Dict[int, List[Tuple]],
    code_mean: float,
) -> Dict[int, float]:
    """
    Process subjects and generate predictions.
    
    For each test sample:
    - Find most recent value of target_code before prediction_time (excluding zeros)
    - Use code_mean as fallback if no historical value exists

    The returned dictionary is keyed by the original test row index. That row
    index is what ultimately keeps the saved prediction parquet aligned with the
    cohort labels.
    """
    results = {}
    
    for subject in subject_iter:
        if subject.subject_id not in test_lookup:
            continue
        
        # Build time-sorted history for target code (excluding zeros)
        history = [
            (event.time, event.numeric_value)
            for event in subject.events
            if event.code == target_code
            and event.time is not None
            and event.numeric_value is not None
            # and event.numeric_value != 0
        ]

        # history = []
        # for event in subject.events:
        #     if event.code == target_code and event.time is not None:
        #         if event.numeric_value is not None:
        #             history.append((event.time, event.numeric_value))
        #         else:
        #             history.append((event.time, 0))

            # and event.numeric_value != 0
        
        # Generate prediction for each test sample
        for row_idx, pred_time in test_lookup[subject.subject_id]:
            # Find most recent value before prediction_time
            last_value = None
            for event_time, event_value in history:
                if event_time <= pred_time:
                    last_value = event_value
                else:
                    break  # Events are time-sorted
            
            results[row_idx] = last_value if last_value is not None else code_mean
    
    return results


# ============================================================================
# Evaluation
# ============================================================================
def build_prediction_df(
    test_df: pd.DataFrame,
    predictions: Dict[int, float],
    fallback_value: float,
) -> pd.DataFrame:
    """
    Create the saved regression prediction DataFrame.

    Why this helper exists:
    - The p-value script expects the learned-model regression schema:
      subject_id, prediction_time, y_true, y_pred, residual.
    - We therefore convert the baseline outputs into that same schema.
    - We also build the DataFrame from the cohort test rows directly instead of
      from a merged key table. This preserves duplicate key rows exactly.

    Any row that did not receive a subject-level prediction is filled with the
    task-specific fallback mean so that every cohort test row is represented in
    the output parquet.
    """
    y_pred = (
        test_df.index.to_series()
        .map(predictions)
        .fillna(fallback_value)
        .astype(float)
        .to_numpy()
    )

    pred_df = test_df[["subject_id", "prediction_time"]].copy()
    pred_df["y_true"] = test_df["numerical_value"].to_numpy()
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    return pred_df


def evaluate_and_save(
    pred_df: pd.DataFrame,
    task: str,
    output_dir: Path,
):
    """
    Write <output_dir>/<task>/most_recent/{predictions.parquet, metrics.json}, the same layout
    and metric (R^2, mean/std over 1000 test resamples) as the foundation-model probes.
    """
    out_dir = output_dir / task / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    write_metrics(out_dir, r2_metrics(pred_df["y_true"], pred_df["y_pred"]))


# ============================================================================
# Main
# ============================================================================
def run_baseline(
    cohort_dir: Path,
    meds_path: str,
    split_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset: str = "mimic",
    num_threads: int = 40,
    tasks: List[str] = None,
):
    """
    Run the most-recent-value baseline for every task in the dataset-specific map.

    The MEDS pass computes predictions, and then each task writes
    <output_dir>/<task>/most_recent/{predictions.parquet, metrics.json}.
    """
    if dataset == "mimic":
        CODE_DICT = MIMIC_CODE_DICT 
    else:
        CODE_DICT = CUMC_CODE_DICT
    tasks = tasks or list(CODE_DICT.keys())
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing MEDS database: {meds_path}")
    # db = meds_reader.SubjectDatabase(meds_path, num_threads=num_threads)

    with meds_reader.SubjectDatabase(meds_path, num_threads=num_threads) as db:
        # for task in tasks:
        #     ...

        for task in tasks:
            target_code = CODE_DICT[task]
            print(f"\n{'='*60}\nTask: {task} | Code: {target_code}\n{'='*60}")
            
            # Load data
            train_df, test_df = load_data(cohort_dir, split_path, task,CODE_DICT)
            print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
            
            # Compute fallback mean
            code_mean = compute_code_mean(train_df, target_code)
            print(f"Code mean (fallback): {code_mean:.4f}")
            
            # Build test lookup
            test_lookup = build_test_lookup(test_df)
            print(f"Unique test subjects: {len(test_lookup):,}")
            
            # Run parallel inference
            worker = functools.partial(
                process_subjects,
                target_code=target_code,
                test_lookup=test_lookup,
                code_mean=code_mean,
            )
            predictions = functools.reduce(lambda acc, x: {**acc, **x}, db.map(worker), {})
            print(f"Generated {len(predictions):,} predictions")

            pred_df = build_prediction_df(test_df, predictions, code_mean)

            # Evaluate
            evaluate_and_save(pred_df, task, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Most Recent Imputation Baseline")
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument("--meds_path", type=str, required=True)
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="Deprecated alias for --output_dir.",
    )
    parser.add_argument("--dataset", type=str,default="mimic")
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks to run (default: every task of the dataset's code map).")
    args = parser.parse_args()

    # Keep backward compatibility with the older CLI flag name while making the
    # new output layout explicit through --output_dir.
    output_dir = args.result_dir if args.result_dir is not None else args.output_dir

    run_baseline(
        cohort_dir=args.cohort_dir,
        meds_path=args.meds_path,
        split_path=args.split_path,
        output_dir=output_dir,
        dataset=args.dataset,
        num_threads=args.num_threads,
        tasks=args.tasks.split(",") if args.tasks else None,
    )


if __name__ == "__main__":
    main()
