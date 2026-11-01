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
   "LAB//51265//K/uL": "Platelets",
   "LAB//50912//mg/dL": "Creatinine",
   "LAB//50821//mm Hg": "Spo2",
   "LAB//50931//mg/dL":"Glucose",
   "LAB//50813//mmol/L":"Lactate"
}

# TARGETS = {
#     "platelets":  ["LAB//51265//K/uL"],
#     "bilirubin":  ["LAB//50885//mg/dL"],
#     "creatinine": ["LAB//50912//mg/dL"],
#     "pao2":       ["LAB//50821//mm Hg"], 
# }

# TARGETS = {
#     "glucose": ["LAB//50931//mg/dL"],    
#     "lactate": ["LAB//50813//mmol/L"],  
# }


LABELS = list(code_dict.values())

def get_interval(subject_iter, code_dict: Dict[str, str]):
    """
    Extract inter-event intervals and numeric values for lab codes.

    Returns:
        Dict with 'intervals' and 'values', each mapping lab names to lists.
    """
    interval_time_list = {lab: [] for lab in code_dict.values()}
    value_list = {lab: [] for lab in code_dict.values()}
    value_diff_list = {lab: [] for lab in code_dict.values()}

    total_subjects = 0
    total_events_checked = 0

    for subject in subject_iter:
        total_subjects += 1
        last_time = {code: None for code in code_dict.keys()}
        last_value = {code: None for code in code_dict.keys()}

        for event in subject.events:
            if event.code in code_dict and event.time is not None:
                total_events_checked += 1
                lab_name = code_dict[event.code]
                cur_value = getattr(event, 'numeric_value', None)

                # Collect numeric value
                if cur_value is not None:
                    value_list[lab_name].append(cur_value)

                # Calculate interval and value difference from previous event
                prev_time = last_time[event.code]
                if prev_time is not None:
                    interval_days = (event.time - prev_time).total_seconds() / 86400.0
                    interval_time_list[lab_name].append(interval_days)

                prev_val = last_value[event.code]
                if prev_val is not None and cur_value is not None:
                    value_diff_list[lab_name].append(cur_value - prev_val)

                last_time[event.code] = event.time
                if cur_value is not None:
                    last_value[event.code] = cur_value

    print(f"\nProcessing statistics:")
    print(f"  Total subjects processed: {total_subjects:,}")
    print(f"  Total lab events checked: {total_events_checked:,}")

    return {'intervals': interval_time_list, 'values': value_list, 'diffs': value_diff_list}


def aggre(init, chunk):
    """
    Aggregate interval and value lists from multiple worker chunks.
    """
    for k in init['intervals']:
        init['intervals'][k].extend(chunk['intervals'][k])
        init['values'][k].extend(chunk['values'][k])
        init['diffs'][k].extend(chunk['diffs'][k])
    return init


def plot_distributions(all_intervals_days, all_values, all_diffs,
                       out_dir="interval_plots_icu", bins=100, clip_q=0.99):
    """
    For each lab, plot a figure with three subplots:
      Left:   histogram of inter-event intervals (days)
      Middle: histogram of inter-event value differences
      Right:  histogram of numeric values
    """
    os.makedirs(out_dir, exist_ok=True)

    for lab in all_intervals_days:
        intervals = np.asarray(all_intervals_days[lab], dtype=float)
        intervals = intervals[np.isfinite(intervals) & (intervals >= 0)]

        values = np.asarray(all_values[lab], dtype=float)
        values = values[np.isfinite(values)]

        diffs = np.asarray(all_diffs[lab], dtype=float)
        diffs = diffs[np.isfinite(diffs)]

        if intervals.size == 0 and values.size == 0 and diffs.size == 0:
            print(f"[warn] No data for {lab}")
            continue

        fig, (ax_int, ax_diff, ax_val) = plt.subplots(1, 3, figsize=(22, 6))

        # Left subplot: inter-event interval
        if intervals.size > 0:
            xmax = np.quantile(intervals, clip_q)
            x_plot = intervals[intervals <= xmax]
            ax_int.hist(x_plot, bins=bins, density=True, alpha=0.7, edgecolor='black')
            ax_int.set_title(f"Inter-event time: {lab}\n(n={len(intervals):,}, "
                             f"≤{100*clip_q:.0f}th pctl)")
            ax_int.set_xlabel("Days")
            ax_int.set_ylabel("Density")
        else:
            ax_int.set_title(f"Inter-event time: {lab}\n(no data)")
        ax_int.grid(True, alpha=0.3)

        # Middle subplot: inter-event value difference
        if diffs.size > 0:
            dmax = np.quantile(diffs, clip_q)
            dmin = np.quantile(diffs, 1 - clip_q)
            d_plot = diffs[(diffs >= dmin) & (diffs <= dmax)]
            ax_diff.hist(d_plot, bins=bins, density=True, alpha=0.7, edgecolor='black',
                         color='tab:green')
            ax_diff.set_title(f"Value difference: {lab}\n(n={len(diffs):,}, "
                              f"{100*(1-clip_q):.0f}–{100*clip_q:.0f}th pctl)")
            ax_diff.set_xlabel("Δ Value (current − previous)")
            ax_diff.set_ylabel("Density")
        else:
            ax_diff.set_title(f"Value difference: {lab}\n(no data)")
        ax_diff.grid(True, alpha=0.3)

        # Right subplot: numeric value distribution
        if values.size > 0:
            vmax = np.quantile(values, clip_q)
            vmin = np.quantile(values, 1 - clip_q)
            v_plot = values[(values >= vmin) & (values <= vmax)]
            ax_val.hist(v_plot, bins=bins, density=True, alpha=0.7, edgecolor='black',
                        color='tab:orange')
            ax_val.set_title(f"Value distribution: {lab}\n(n={len(values):,}, "
                             f"{100*(1-clip_q):.0f}–{100*clip_q:.0f}th pctl)")
            ax_val.set_xlabel("Value")
            ax_val.set_ylabel("Density")
        else:
            ax_val.set_title(f"Value distribution: {lab}\n(no data)")
        ax_val.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = os.path.join(out_dir, f"{lab.replace('/', '_')}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)
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
    meds_path = "/shared/share_mala/zj2398/mimic/data/meds_v0.6_reader"  # Path to your MEDS data    
    
    # Step 1: Load visits and create lookup structure
    # visits_dict = load_and_prepare_visits(visits_parquet_path)
    
    # Step 2: Initialize MEDS database
    print(f"\nInitializing MEDS database from {meds_path}...")
    db = meds_reader.SubjectDatabase(meds_path, num_threads=40)
    
    # Step 3: Process subjects with parallel workers
    print(f"\nProcessing subjects to extract intervals and values...")
    init_result = {'intervals': {lab: [] for lab in LABELS},
                   'values': {lab: [] for lab in LABELS},
                   'diffs': {lab: [] for lab in LABELS}}

    worker = functools.partial(get_interval, code_dict=code_dict)
    results = functools.reduce(aggre, db.map(worker), init_result)

    all_interval_days = results['intervals']
    all_values = results['values']
    all_diffs = results['diffs']

    # Step 4: Print summary statistics
    print(f"\n{'='*70}")
    print("Final Results:")
    print(f"{'='*70}")
    for lab in LABELS:
        intervals = all_interval_days[lab]
        values = all_values[lab]
        diffs = all_diffs[lab]
        parts = []
        if len(intervals) > 0:
            parts.append(f"{len(intervals):7,} intervals | "
                         f"mean={np.mean(intervals):6.2f} days | "
                         f"median={np.median(intervals):6.2f} days")
        else:
            parts.append("No intervals found")
        if len(diffs) > 0:
            parts.append(f"{len(diffs):7,} diffs | "
                         f"mean={np.mean(diffs):8.2f} | "
                         f"median={np.median(diffs):8.2f}")
        else:
            parts.append("No diffs found")
        if len(values) > 0:
            parts.append(f"{len(values):7,} values | "
                         f"mean={np.mean(values):8.2f} | "
                         f"median={np.median(values):8.2f}")
        else:
            parts.append("No values found")
        print(f"{lab:15s}: {' ; '.join(parts)}")

    # Step 5: Plot distributions
    print(f"\nGenerating plots...")
    plot_distributions(all_interval_days, all_values, all_diffs, "interval_plots_inpatient")
    
    # Clean up
    db.terminate()
    print(f"\nDone!")


if __name__ == "__main__":
    main()