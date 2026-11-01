"""
Task 2: Extract inter-event intervals for lab codes, 
filtering to only events that occur within a visit window

Key changes from original:
1. Load visits data from Task 1
2. Create efficient lookup structure (dict) mapping person_id -> visit windows
3. For each event, check if event.time falls within any visit for that subject
4. Handle datetime conversion from string to datetime for comparison
5. Only track intervals for events that occur within visits
"""

import os
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import meds_reader  # Assuming this is your MEDS library


# Lab codes to extract
code_dict = {
    "LOINC/26515-7//UCUM/10*9/L": "Platelets",
    "LOINC/2160-0//UCUM/mg/dL": "Creatinine",
    "LOINC/1968-7//UCUM/mg/dL": "Bilirubin",
    "LOINC/2708-6//UCUM/%": "Spo2"
}

LABELS = list(code_dict.values())


def load_and_prepare_visits(visits_parquet_path: str) -> Dict[int, List[Tuple[datetime, datetime]]]:
    """
    Load visits data and create efficient lookup structure.
    
    Args:
        visits_parquet_path: Path to combined visits parquet from Task 1
    
    Returns:
        Dictionary mapping person_id -> list of (start_datetime, end_datetime) tuples
    
    Efficiency notes:
    - Pre-processing visits into a dict avoids repeated DataFrame filtering
    - Converting to datetime once upfront (not per event check)
    - Storing as tuples in list for fast iteration
    - This structure allows O(k) lookup where k = number of visits per person
    """
    print(f"Loading visits from {visits_parquet_path}...")
    visits_df = pd.read_parquet(visits_parquet_path)
    
    # Convert string datetime columns to datetime objects
    # Handle common datetime formats automatically
    print("Converting datetime columns...")
    visits_df['visit_start_datetime'] = pd.to_datetime(visits_df['visit_start_datetime'])
    visits_df['visit_end_datetime'] = pd.to_datetime(visits_df['visit_end_datetime'])
    
    # Remove timezone info if present (to match event.time which is naive datetime)
    if visits_df['visit_start_datetime'].dt.tz is not None:
        visits_df['visit_start_datetime'] = visits_df['visit_start_datetime'].dt.tz_localize(None)
    if visits_df['visit_end_datetime'].dt.tz is not None:
        visits_df['visit_end_datetime'] = visits_df['visit_end_datetime'].dt.tz_localize(None)
    
    print(f"Loaded {len(visits_df):,} visits for {visits_df['person_id'].nunique():,} unique persons")
    
    # Create dictionary: person_id -> list of (start, end) tuples
    # This is much faster than filtering DataFrame for each subject
    visits_dict = {}
    for person_id, group in visits_df.groupby('person_id'):
        # Convert each row to (start, end) tuple
        # Using .values accesses underlying numpy array (faster than iterrows)
        visit_windows = [
            (row['visit_start_datetime'], row['visit_end_datetime'])
            for _, row in group[['visit_start_datetime', 'visit_end_datetime']].iterrows()
        ]
        visits_dict[person_id] = visit_windows
    
    # Statistics
    visits_per_person = [len(v) for v in visits_dict.values()]
    print(f"Visits per person - mean: {np.mean(visits_per_person):.1f}, "
          f"median: {np.median(visits_per_person):.0f}, "
          f"max: {np.max(visits_per_person)}")
    
    return visits_dict


def is_event_in_visit(event_time: datetime, 
                       visit_windows: List[Tuple[datetime, datetime]]) -> bool:
    """
    Check if an event falls within any visit window for this subject.
    
    Args:
        event_time: Datetime of the event
        visit_windows: List of (start, end) datetime tuples for this subject's visits
    
    Returns:
        True if event falls within any visit window, False otherwise
    
    Efficiency notes:
    - Early termination when match found (no unnecessary checks)
    - Simple comparison operators (datetime objects handle this efficiently)
    - For subjects with many visits, could optimize with interval tree,
      but for typical cases (dozens of visits) linear search is fine
    """
    for start, end in visit_windows:
        # Check if event_time is within this visit window (inclusive bounds)
        if start <= event_time <= end:
            return True
    return False


def get_interval(subject_iter, code_dict: Dict[str, str], 
                 visits_dict: Dict[int, List[Tuple[datetime, datetime]]]):
    """
    Extract inter-event intervals for lab codes, filtering to events within visits.
    
    Args:
        subject_iter: Iterator of subjects from MEDS database
        code_dict: Mapping of lab codes to lab names
        visits_dict: Dictionary mapping person_id to visit windows
    
    Returns:
        Dictionary mapping lab names to list of inter-event intervals (in days)
    
    Logic changes from original:
    1. Added visits_dict parameter containing pre-processed visit windows
    2. For each subject, look up their visit windows from visits_dict
    3. For each event, check if it falls within any visit window
    4. Only track intervals for events that occur within visits
    5. If subject has no visits in cohort, skip all their events
    
    Efficiency considerations:
    - visits_dict lookup is O(1) per subject
    - Visit window checking is O(k) where k = visits per subject
    - Pre-converted datetimes avoid repeated parsing
    - Early filtering reduces downstream processing
    """
    interval_time_list = {lab: [] for lab in code_dict.values()}
    
    # Track statistics for debugging
    total_subjects = 0
    subjects_with_visits = 0
    total_events_checked = 0
    events_in_visits = 0
    
    for subject in subject_iter:
        total_subjects += 1
        
        # Get this subject's visit windows
        # Assuming subject.subject_id corresponds to person_id in visits data
        subject_id = subject.subject_id
        
        # If subject has no visits in our cohort (visit_concept_id=9202), skip them
        if subject_id not in visits_dict:
            continue
        
        subjects_with_visits += 1
        visit_windows = visits_dict[subject_id]
        
        # Track last event time for each lab code (for this subject)
        # Reset for each subject to avoid cross-subject intervals
        last_time = {code: None for code in code_dict.keys()}
        
        for event in subject.events:
            # Check if this event is one of our target lab codes and has a time
            if event.code in code_dict and event.time is not None:
                total_events_checked += 1
                
                # NEW: Check if event falls within any visit window
                if is_event_in_visit(event.time, visit_windows):
                    events_in_visits += 1
                    
                    # Original logic: calculate interval from previous event
                    prev_time = last_time[event.code]
                    if prev_time is not None:
                        # Calculate interval in days
                        interval_days = (event.time - prev_time).total_seconds() / 86400.0
                        interval_time_list[code_dict[event.code]].append(interval_days)
                    
                    # Update last seen time for this lab code
                    last_time[event.code] = event.time
                # else: Event not in any visit window -> ignore it
    
    # Print statistics
    print(f"\nProcessing statistics:")
    print(f"  Total subjects processed: {total_subjects:,}")
    print(f"  Subjects with visits in cohort: {subjects_with_visits:,} "
          f"({100*subjects_with_visits/max(total_subjects,1):.1f}%)")
    print(f"  Total lab events checked: {total_events_checked:,}")
    print(f"  Events within visit windows: {events_in_visits:,} "
          f"({100*events_in_visits/max(total_events_checked,1):.1f}%)")
    
    return interval_time_list


def aggre(init, chunk):
    """
    Aggregate interval lists from multiple worker chunks.
    
    This is used with functools.reduce to combine results from parallel processing.
    """
    for k in init:
        init[k].extend(chunk[k])
    return init


def plot_distributions_days(all_intervals_days, out_dir="interval_plots_outpatient", 
                            bins=80, clip_q=0.99):
    """
    Plot histogram of inter-event intervals for each lab test.
    
    Args:
        all_intervals_days: Dict mapping lab names to list of intervals in days
        out_dir: Directory to save plots
        bins: Number of histogram bins
        clip_q: Quantile to clip x-axis (to avoid extreme outliers)
    """
    os.makedirs(out_dir, exist_ok=True)

    for lab, vals in all_intervals_days.items():
        x = np.asarray(vals, dtype=float)
        x = x[np.isfinite(x)]  # Remove any NaN or inf values
        x = x[x >= 0]  # Remove negative intervals (shouldn't happen but be safe)

        if x.size == 0:
            print(f"[warn] No intervals for {lab}")
            continue

        # Clip extreme outliers for better visualization
        xmax = np.quantile(x, clip_q)
        x_plot = x[x <= xmax]

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.hist(x_plot, bins=bins, density=True, alpha=0.7, edgecolor='black')
        plt.title(f"Inter-event time: {lab}\n(n={len(x):,} intervals, "
                 f"showing {100*clip_q:.0f}th percentile)")
        plt.xlabel("Days")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        fname = os.path.join(out_dir, f"inter_event_{lab.replace('/', '_')}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved: {fname}")


def main():
    """
    Main execution function.
    
    Steps:
    1. Load and prepare visits data from Task 1
    2. Initialize MEDS database reader
    3. Process subjects in parallel to extract intervals
    4. Aggregate results and plot distributions
    """
    # Configuration - UPDATE THESE PATHS
    meds_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader/"  # Path to your MEDS data
    visits_parquet_path = "/data2/processed_datasets/zj2398/cumc_data/inpatient_visits.parquet"  # Output from Task 1
    
    
    # Step 1: Load visits and create lookup structure
    visits_dict = load_and_prepare_visits(visits_parquet_path)
    
    # Step 2: Initialize MEDS database
    print(f"\nInitializing MEDS database from {meds_path}...")
    db = meds_reader.SubjectDatabase(meds_path, num_threads=40)
    
    # Step 3: Process subjects with parallel workers
    print(f"\nProcessing subjects to extract intervals...")
    init_result = {lab: [] for lab in LABELS}
    
    # Create worker function with visits_dict bound to it
    worker = functools.partial(get_interval, 
                              code_dict=code_dict, 
                              visits_dict=visits_dict)
    
    # Map worker across all subjects and reduce results
    all_interval_days = functools.reduce(aggre, db.map(worker), init_result)
    
    # Step 4: Print summary statistics
    print(f"\n{'='*70}")
    print("Final Results:")
    print(f"{'='*70}")
    for lab, intervals in all_interval_days.items():
        if len(intervals) > 0:
            print(f"{lab:15s}: {len(intervals):7,} intervals | "
                  f"mean={np.mean(intervals):6.2f} days | "
                  f"median={np.median(intervals):6.2f} days")
        else:
            print(f"{lab:15s}: No intervals found")
    
    # Step 5: Plot distributions
    print(f"\nGenerating plots...")
    plot_distributions_days(all_interval_days,"interval_plots_inpatient")
    
    # Clean up
    db.terminate()
    print(f"\nDone!")


if __name__ == "__main__":
    main()