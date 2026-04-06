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
    --result_dir /shared/share_mala/zj2398/mimic/task/regression/baseline_results/result

python baseline.py \
    --cohort_dir /data2/processed_datasets/zj2398/cumc_data/regression_100k/ \
    --meds_path /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
    --dataset cumc \
    --split_path /data2/processed_datasets/zj2398/femr/cumc/motor/main_split.csv \
    --result_dir /data/results/zj2398/femr/cumc/regression/summary_tables
"""

import argparse
import functools
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import meds_reader
import numpy as np
import pandas as pd

from femr.omop_meds_tutorial.evaluation.eval_utils import (
    _bootstrap_regression_metrics,
    _metric_mean_std,
    _update_metric_table,
)

# ============================================================================
# Constants
# ============================================================================
N_BOOT = 100
MODEL_COL = "Most Recent Imputation"

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
    "bilirubin": "LOINC/1968-7//UCUM/mg/dL",
    "spo2":"LOINC/2708-6//UCUM/%",
}

# ============================================================================
# Data Loading
# ============================================================================
def load_data(cohort_dir: Path, split_path: Path, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cohort and split into train/test."""
    cohort_df = pd.read_parquet(cohort_dir / task / f"{task}.parquet")
    split_df = pd.read_csv(split_path).rename(columns={"split_name": "split"})
    merged = cohort_df.merge(split_df, on="subject_id", how="inner")
    
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)
    train_df = merged[merged["split"] != "test"]
    return train_df, test_df


def compute_code_mean(train_df: pd.DataFrame, target_code: str) -> float:
    """Compute mean for target code from training data, excluding zeros."""
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
    """Build subject_id -> [(row_idx, prediction_time), ...] mapping."""
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
def evaluate_and_save(
    test_df: pd.DataFrame,
    predictions: Dict[int, float],
    task: str,
    result_dir: Path,
):
    """Compute bootstrap metrics and save to result tables."""
    y_pred = test_df.index.map(predictions).to_numpy()
    y_true = test_df["numerical_value"].to_numpy()
    
    metrics = _bootstrap_regression_metrics(y_true, y_pred, n_boot=N_BOOT, seed=0)
    
    rmse_mean, rmse_std = _metric_mean_std(metrics, "rmse")
    mae_mean, mae_std = _metric_mean_std(metrics, "mae")
    r2_mean, r2_std = _metric_mean_std(metrics, "r2")
    
    print(f"  RMSE: {rmse_mean:.4f} ({rmse_std:.4f})")
    print(f"  MAE:  {mae_mean:.4f} ({mae_std:.4f})")
    print(f"  R2:   {r2_mean:.4f} ({r2_std:.4f})")
    
    # _update_metric_table(result_dir, "rmse", task, MODEL_COL, rmse_mean, rmse_std)
    # _update_metric_table(result_dir, "mae", task, MODEL_COL, mae_mean, mae_std)
    # _update_metric_table(result_dir, "r2", task, MODEL_COL, r2_mean, r2_std)


# ============================================================================
# Main
# ============================================================================
def run_baseline(
    cohort_dir: Path,
    meds_path: str,
    split_path: Path,
    result_dir: Path,
    dataset: str = "mimic",
    num_threads: int = 40,
):
    """Run most recent imputation baseline for all tasks."""
    if dataset == "mimic":
        CODE_DICT = MIMIC_CODE_DICT 
    else:
        CODE_DICT = CUMC_CODE_DICT
    tasks = list(CODE_DICT.keys())
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing MEDS database: {meds_path}")
    # db = meds_reader.SubjectDatabase(meds_path, num_threads=num_threads)

    with meds_reader.SubjectDatabase(meds_path, num_threads=num_threads) as db:
        # for task in tasks:
        #     ...

        for task in tasks:
            target_code = CODE_DICT[task]
            print(f"\n{'='*60}\nTask: {task} | Code: {target_code}\n{'='*60}")
            
            # Load data
            train_df, test_df = load_data(cohort_dir, split_path, task)
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
            
            # Evaluate
            evaluate_and_save(test_df, predictions, task, result_dir)


def main():
    parser = argparse.ArgumentParser(description="Most Recent Imputation Baseline")
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument("--meds_path", type=str, required=True)
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str,default="mimic")
    parser.add_argument("--num_threads", type=int, default=32)
    args = parser.parse_args()
    
    run_baseline(
        cohort_dir=args.cohort_dir,
        meds_path=args.meds_path,
        split_path=args.split_path,
        result_dir=args.result_dir,
        dataset=args.dataset,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()