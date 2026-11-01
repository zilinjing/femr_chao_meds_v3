"""
Regression Cohort Generation for Lab Values Prediction

This script generates cohorts for regression tasks predicting future lab values.
For each outpatient visit end (prediction time), it finds the first qualifying 
lab event occurring 2-5 months later that also falls within a visit window.

Optimizations over the original code:
1. Uses numpy arrays with timestamps instead of datetime objects for faster comparisons
2. Binary search for finding events in the prediction window
3. Vectorized interval checking using numpy broadcasting
4. Eliminated slow iterrows() calls
5. Pre-sorted visits by end time for efficient iteration
"""

import os
import functools
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import meds_reader

# Lab codes to extract
CODE_DICT = {
    "LOINC/26515-7//UCUM/10*9/L": "platelets",
    "LOINC/2160-0//UCUM/mg/dL": "creatinine",
    "LOINC/1968-7//UCUM/mg/dL": "bilirubin",
}

# Prediction window parameters (in seconds)
WINDOW_START_DAYS = 60   # ~2 months
WINDOW_END_DAYS = 150    # ~5 months
# WINDOW_START_DAYS = 180   # ~2 months
# WINDOW_END_DAYS = 270    # ~5 months
WINDOW_START_SECS = WINDOW_START_DAYS * 86400
WINDOW_END_SECS = WINDOW_END_DAYS * 86400


def load_and_prepare_visits_optimized(visits_parquet_path: str) -> Dict[int, np.ndarray]:
    """
    Load visits data and create efficient numpy-based lookup structure.
    
    Returns:
        Dictionary mapping person_id -> numpy array of shape (N, 2) where
        each row is [start_timestamp, end_timestamp] in seconds.
        Array is sorted by end_timestamp (prediction times).
    
    Optimizations:
    - Converts datetimes to int64 timestamps once (avoid repeated conversions)
    - Uses numpy arrays instead of lists of tuples (faster iteration)
    - Vectorized groupby operations instead of iterrows()
    - Pre-sorted by end time for efficient prediction time iteration
    """
    print(f"Loading visits from {visits_parquet_path}...")
    visits_df = pd.read_parquet(visits_parquet_path)
    
    # Convert to datetime
    print("Converting datetime columns...")
    visits_df['visit_start_datetime'] = pd.to_datetime(visits_df['visit_start_datetime'])
    visits_df['visit_end_datetime'] = pd.to_datetime(visits_df['visit_end_datetime'])
    
    # Remove timezone if present
    if visits_df['visit_start_datetime'].dt.tz is not None:
        visits_df['visit_start_datetime'] = visits_df['visit_start_datetime'].dt.tz_localize(None)
    if visits_df['visit_end_datetime'].dt.tz is not None:
        visits_df['visit_end_datetime'] = visits_df['visit_end_datetime'].dt.tz_localize(None)
    
    # Convert to Unix timestamps (int64 nanoseconds -> seconds as float)
    # This is MUCH faster for comparisons than datetime objects
    print("Converting to timestamps...")
    # print(f"visits_df['visit_start_datetime'] is {visits_df['visit_start_datetime']}")
    visits_df['start_ts'] = visits_df['visit_start_datetime'].astype('int64') // 10**9
    # print(f"visits_df['start_ts'] is {visits_df['start_ts']}")
    visits_df['end_ts'] = visits_df['visit_end_datetime'].astype('int64') // 10**9
    
    print(f"Loaded {len(visits_df):,} visits for {visits_df['person_id'].nunique():,} unique persons")
    
    # Build dictionary with numpy arrays - OPTIMIZED version
    # Using groupby + apply with numpy conversion is much faster than iterrows
    print("Building visits dictionary...")
    
    visits_dict = {}
    
    # Group and convert to numpy arrays efficiently
    grouped = visits_df.groupby('person_id')[['start_ts', 'end_ts']].apply(
        lambda x: x.values.astype(np.int64)
    )
    
    for person_id, visit_array in grouped.items():
        # Sort by end_ts (prediction times) for efficient iteration
        sorted_indices = np.argsort(visit_array[:, 1])
        visits_dict[person_id] = visit_array[sorted_indices]
    
    # Statistics
    visits_per_person = [len(v) for v in visits_dict.values()]
    print(f"Visits per person - mean: {np.mean(visits_per_person):.1f}, "
          f"median: {np.median(visits_per_person):.0f}, "
          f"max: {np.max(visits_per_person)}")
    
    return visits_dict


def is_event_in_any_visit_vectorized(event_ts: int, visit_array: np.ndarray) -> bool:
    """
    Check if event timestamp falls within any visit window using vectorized operations.
    
    Args:
        event_ts: Event timestamp in seconds
        visit_array: numpy array of shape (N, 2) with [start_ts, end_ts] rows
    
    Returns:
        True if event falls within any visit window
    
    This is O(1) memory and O(n) time, but with numpy's optimized C operations
    it's much faster than Python loops for typical visit counts (10-100).
    """
    # Vectorized check: start <= event_ts <= end
    in_window = (visit_array[:, 0] <= event_ts) & (event_ts <= visit_array[:, 1])
    return np.any(in_window)


def extract_subject_events(subject, target_codes: set) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract all qualifying events for a subject as numpy arrays.
    
    Returns:
        Tuple of (timestamps_array, values_array) or None if no qualifying events.
        Both arrays are aligned by index.
    """
    timestamps = []
    values = []
    
    for event in subject.events:
        if event.code in target_codes and event.time is not None:
            # Check if numerical_value is valid
            try:
                val = event.numeric_value
                if val is not None:
                    float_val = float(val)
                    # Convert datetime to timestamp
                    ts = int(event.time.timestamp())
                    timestamps.append(ts)
                    values.append(float_val)
            except (TypeError, ValueError):
                continue
    
    if not timestamps:
        return None
    
    return np.array(timestamps, dtype=np.int64), np.array(values, dtype=np.float64)


def build_cohort_for_code(
    subject_iter,
    target_code: str,
    lab_name: str,
    visits_dict: Dict[int, np.ndarray]
) -> List[Dict]:
    """
    Build regression cohort for a specific lab code.
    
    For each subject:
        For each visit end (prediction_time):
            Find first event in [pred_time + 2mo, pred_time + 5mo] that's within any visit
            Record (subject_id, prediction_time, event_time, numerical_value)
    
    Args:
        subject_iter: Iterator of subjects from MEDS database
        target_code: The lab code to process
        lab_name: Human-readable name for the lab
        visits_dict: Dictionary mapping person_id -> numpy array of visit windows
    
    Returns:
        List of dicts with keys: subject_id, prediction_time, event_time, numerical_value
    """
    results = []
    target_codes = {target_code}
    
    # Statistics
    total_subjects = 0
    subjects_with_visits = 0
    subjects_with_events = 0
    total_predictions = 0
    valid_predictions = 0
    
    for subject in subject_iter:
        total_subjects += 1
        subject_id = subject.subject_id
        
        # Skip if no visits for this subject
        if subject_id not in visits_dict:
            continue
        
        subjects_with_visits += 1
        visit_array = visits_dict[subject_id]  # Shape: (N, 2), sorted by end_ts

        
        # Extract all events for this lab code
        event_data = extract_subject_events(subject, target_codes)
        if event_data is None:
            continue
        
        subjects_with_events += 1
        event_timestamps, event_values = event_data
        
        # Get unique prediction times (visit end times)
        prediction_times = np.unique(visit_array[:, 1])
        
        for pred_ts in prediction_times:
            total_predictions += 1
            
            # Calculate window bounds
            window_start = pred_ts + WINDOW_START_SECS
            window_end = pred_ts + WINDOW_END_SECS
            
            # Binary search to find events in the window
            # Events are already sorted by time (from MEDS)
            left_idx = np.searchsorted(event_timestamps, window_start, side='left')
            right_idx = np.searchsorted(event_timestamps, window_end, side='right')
            
            # Check candidates in order (take first valid one)
            for idx in range(left_idx, right_idx):
                event_ts = event_timestamps[idx]
                
                # Check if this event falls within any visit window
                if is_event_in_any_visit_vectorized(event_ts, visit_array):
                    valid_predictions += 1
                    
                    # Convert timestamps back to datetime for output
                    pred_datetime = datetime.fromtimestamp(pred_ts)
                    event_datetime = datetime.fromtimestamp(event_ts)
                    
                    results.append({
                        'subject_id': subject_id,
                        'prediction_time': pred_datetime,
                        'event_time': event_datetime,
                        'numerical_value': event_values[idx]
                    })
                    break  # Take first qualifying event only
    
    # Print statistics
    print(f"\n  [{lab_name}] Processing statistics:")
    print(f"    Total subjects: {total_subjects:,}")
    print(f"    Subjects with visits: {subjects_with_visits:,} ({100*subjects_with_visits/max(total_subjects,1):.1f}%)")
    print(f"    Subjects with events: {subjects_with_events:,} ({100*subjects_with_events/max(subjects_with_visits,1):.1f}%)")
    print(f"    Total prediction times: {total_predictions:,}")
    print(f"    Valid predictions (with qualifying events): {valid_predictions:,} ({100*valid_predictions/max(total_predictions,1):.1f}%)")
    
    return results


def aggregate_results(init: Dict[str, List], chunk: Dict[str, List]) -> Dict[str, List]:
    """Aggregate results from multiple worker chunks."""
    for k in init:
        init[k].extend(chunk[k])
    return init


def build_all_cohorts(
    subject_iter,
    code_dict: Dict[str, str],
    visits_dict: Dict[int, np.ndarray]
) -> Dict[str, List[Dict]]:
    """
    Build cohorts for all lab codes in a single pass through the data.
    
    This is more efficient than multiple passes because:
    1. Subject iteration is expensive (disk I/O)
    2. Visit lookup is O(1) per subject
    3. We can extract events for all codes in one pass
    """
    # Initialize result containers
    results = {lab_name: [] for lab_name in code_dict.values()}
    
    # Create reverse mapping: lab_name -> code
    name_to_code = {v: k for k, v in code_dict.items()}
    target_codes = set(code_dict.keys())
    
    # Statistics per lab
    stats = {lab: {'subjects_with_events': 0, 'total_predictions': 0, 'valid_predictions': 0} 
             for lab in code_dict.values()}
    
    total_subjects = 0
    subjects_with_visits = 0
    
    for subject in subject_iter:
        total_subjects += 1
        subject_id = subject.subject_id
        
        if subject_id not in visits_dict:
            continue
        
        subjects_with_visits += 1
        visit_array = visits_dict[subject_id]
        
        # Collect events per lab code
        events_by_lab = {lab: ([], []) for lab in code_dict.values()}
        
        for event in subject.events:
            if event.code in target_codes and event.time is not None:
                try:
                    val = event.numeric_value
                    if val is not None:
                        float_val = float(val)
                        ts = int(event.time.timestamp())
                        lab_name = code_dict[event.code]
                        events_by_lab[lab_name][0].append(ts)
                        events_by_lab[lab_name][1].append(float_val)
                except (TypeError, ValueError):
                    continue
        
        # Process each lab code
        prediction_times = np.unique(visit_array[:, 1])
        
        for lab_name in code_dict.values():
            timestamps_list, values_list = events_by_lab[lab_name]
            
            if not timestamps_list:
                continue
            
            stats[lab_name]['subjects_with_events'] += 1
            
            event_timestamps = np.array(timestamps_list, dtype=np.int64)
            event_values = np.array(values_list, dtype=np.float64)
            
            for pred_ts in prediction_times:
                stats[lab_name]['total_predictions'] += 1
                
                window_start = pred_ts + WINDOW_START_SECS
                window_end = pred_ts + WINDOW_END_SECS
                
                left_idx = np.searchsorted(event_timestamps, window_start, side='left')
                right_idx = np.searchsorted(event_timestamps, window_end, side='right')
                
                for idx in range(left_idx, right_idx):
                    event_ts = event_timestamps[idx]
                    
                    if is_event_in_any_visit_vectorized(event_ts, visit_array):
                        stats[lab_name]['valid_predictions'] += 1
                        
                        pred_datetime = datetime.fromtimestamp(pred_ts)
                        event_datetime = datetime.fromtimestamp(event_ts)
                        
                        results[lab_name].append({
                            'subject_id': subject_id,
                            'prediction_time': pred_datetime,
                            'event_time': event_datetime,
                            'numerical_value': event_values[idx]
                        })
                        break
    
    # Print statistics
    print(f"\nProcessing statistics:")
    print(f"  Total subjects: {total_subjects:,}")
    print(f"  Subjects with visits: {subjects_with_visits:,} ({100*subjects_with_visits/max(total_subjects,1):.1f}%)")
    
    for lab_name, s in stats.items():
        print(f"\n  [{lab_name}]")
        print(f"    Subjects with events: {s['subjects_with_events']:,}")
        print(f"    Total prediction times: {s['total_predictions']:,}")
        print(f"    Valid predictions: {s['valid_predictions']:,} "
              f"({100*s['valid_predictions']/max(s['total_predictions'],1):.1f}%)")
    
    return results


def main():
    """
    Main execution function.
    
    Steps:
    1. Load and prepare visits data (optimized with numpy arrays)
    2. Initialize MEDS database reader
    3. Process subjects in parallel to build cohorts for all lab codes
    4. Save each cohort to a separate parquet file
    """
    # Configuration - UPDATE THESE PATHS
    # Configuration - UPDATE THESE PATHS
    meds_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader/"  # Path to your MEDS data
    visits_parquet_path = "/data2/processed_datasets/zj2398/cumc_data/outpatient_visits_new.parquet"  # Output from Task 1
    # meds_path = "xxx"  # Path to your MEDS data
    # visits_parquet_path = "xxx"  # Path to visits parquet from Task 1
    output_dir = "/data2/processed_datasets/zj2398/cumc_data/regression_2-5"  # Output directory for cohort files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load visits with optimized structure
    visits_dict = load_and_prepare_visits_optimized(visits_parquet_path)
    
    # Step 2: Initialize MEDS database
    print(f"\nInitializing MEDS database from {meds_path}...")
    db = meds_reader.SubjectDatabase(meds_path, num_threads=40)
    
    # Step 3: Process subjects and build all cohorts in parallel
    print(f"\nProcessing subjects to build regression cohorts...")
    print(f"Prediction window: {WINDOW_START_DAYS}-{WINDOW_END_DAYS} days after visit end")
    
    # Initialize empty result containers
    init_result = {lab: [] for lab in CODE_DICT.values()}
    
    # Create worker function with visits_dict bound
    worker = functools.partial(
        build_all_cohorts,
        code_dict=CODE_DICT,
        visits_dict=visits_dict
    )
    
    # Map worker across all subjects and reduce results
    all_results = functools.reduce(aggregate_results, db.map(worker), init_result)
    
    # Step 4: Save cohorts to parquet files
    print(f"\n{'='*70}")
    print("Saving cohorts:")
    print(f"{'='*70}")
    
    for lab_name, records in all_results.items():
        if records:
            df = pd.DataFrame(records)
            
            # Ensure proper dtypes
            df['subject_id'] = df['subject_id'].astype('int64')
            df['prediction_time'] = pd.to_datetime(df['prediction_time'])
            df['event_time'] = pd.to_datetime(df['event_time'])
            df['numerical_value'] = df['numerical_value'].astype('float64')
            
            # Sort by subject_id and prediction_time
            df = df.sort_values(['subject_id', 'prediction_time']).reset_index(drop=True)
            df = df[df['numerical_value'] != 0]
            # Save to parquet
            output_path = os.path.join(output_dir,lab_name)
            os.makedirs(output_path,exist_ok=True)
            output_path = os.path.join(output_path, f"{lab_name}.parquet")
            df.to_parquet(output_path, index=False)
            
            print(f"  {lab_name}: {len(df):,} samples -> {output_path}")
            
            # Print sample statistics
            print(f"    Subject IDs: {df['subject_id'].nunique():,} unique")
            print(f"    Regression value - mean: {df['numerical_value'].mean():.2f}, "
                  f"std: {df['numerical_value'].std():.2f}")
        else:
            print(f"  {lab_name}: No samples found")
    
    # Clean up
    db.terminate()
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

