from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, Any
from datetime import datetime,timedelta
import pandas as pd
from functools import partial
import itertools
import meds_reader

# TARGETS = {
#     "platelets":  ["LAB//51265//K/uL"],
#     "bilirubin":  ["LAB//50885//mg/dL"],
#     "creatinine": ["LAB//50912//mg/dL"],
#     "pao2":       ["LAB//50821//mm Hg"], 
# }

TARGETS = {
    "glucose": ["LAB//50931//mg/dL"],    
    "lactate": ["LAB//50813//mmol/L"],  
}

'''
pao2 500k
"glucose": ["50931"],      # optionally review 50809 and 52027 too
"lactate": ["50813"],  
'''

def _coerce_dt(s: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def process_subject_batch(
    subject_data_iter: Iterator[Tuple[meds_reader.Subject, Any]],
    codes: list[str],
) -> list[dict]:
    """
    Process a batch of subjects and return matching lab events.
    
    Args:
        subject_data_iter: Iterator of (Subject, ICU stay data) tuples
        codes: List of lab codes to filter for
    
    Returns:
        List of dictionaries containing lab event data
    """
    rows = []
    
    for subj, stay_data in subject_data_iter:
        pid = subj.subject_id
        # print(pid)

        # print(subj, stay_data)
        
        # stay_data will be the ICU stay rows for this subject_id
        if stay_data is None or len(stay_data) == 0:
            continue
        
        # Convert to DataFrame if needed
        if not isinstance(stay_data, pd.DataFrame):
            stay_df = pd.DataFrame(stay_data)
        else:
            stay_df = stay_data
            
        if stay_df.empty:
            continue
        
        for ev in subj.events:
            code = getattr(ev, "code", None)
            t = getattr(ev, "time", None)
            if code not in codes or t is None:
                continue
            # if code is None or t is None:
            #     continue
            
            # Check if t lies in ANY [intime, outtime] window for this patient
            intime = stay_df["intime"]
            outtime = stay_df["outtime"]
            in_any_window = bool(((intime + timedelta(hours=4) <= t) & (t <= outtime)).any())
            if in_any_window and ev.numeric_value is not None:
                # print(pid,ev.numeric_value)
                rows.append({
                    "subject_id": pid,
                    # "event_time": t,
                    "prediction_time": t-timedelta(hours=4),
                    "code": code,
                    # "numeric_value": getattr(ev, "numeric_value", None),
                    "numerical_value": ev.numeric_value,
                })
    
    return rows

def write_lab_file(
    db_path: str,
    lab_name: str,
    icu_stay_path: str,
    codes: list[str],
    out_dir: str = ".",
    num_threads: int = 8,
) -> None:
    """
    Extract lab events from MEDS database in parallel.
    
    Args:
        db_path: Path to MEDS reader database
        lab_name: Name of the lab (for output filename)
        icu_stay_path: Path to ICU stays CSV file
        codes: List of lab codes to extract
        out_dir: Output directory
        num_threads: Number of threads for parallel processing
    """
    # Load and prepare ICU stays data
    icu_stay = pd.read_csv(icu_stay_path, dtype={"subject_id": "int64"})
    icu_stay = icu_stay[["subject_id", "intime", "outtime"]].dropna()
    icu_stay["intime"] = _coerce_dt(icu_stay["intime"])
    icu_stay["outtime"] = _coerce_dt(icu_stay["outtime"])
    icu_stay = icu_stay.dropna(subset=["intime", "outtime"]).sort_values(
        ["subject_id", "intime"]
    ).reset_index(drop=True)
    
    # Keep subject_id as a column (not index) for map_with_data
    # Ensure subject_id is integer type
    icu_stay = icu_stay.astype({"subject_id": "int64"})
    
    print(f"Processing {lab_name} with {num_threads} threads...")
    
    # Process subjects in parallel
    with meds_reader.SubjectDatabase(db_path, num_threads=num_threads) as db:
        # Create partial function with codes bound
        map_func = partial(process_subject_batch, codes=codes)
        
        # Use map_with_data to process subjects in parallel with their ICU stay data
        # Each worker gets an iterator of (Subject, matching_icu_rows) tuples
        parts_iter = db.map_with_data(map_func, icu_stay)
        
        # Flatten the results: Iterator[List[dict]] -> List[dict]
        all_rows = list(itertools.chain.from_iterable(parts_iter))
    
    # Convert to DataFrame and save
    df = pd.DataFrame(
        all_rows,
        columns=["subject_id", "prediction_time", "code", "numerical_value"]
    )

    # print(f"Initially we have {len(df)} rows ")
    # df = df.dropna(subset=["numeric_value"])
    # print(f"drop null numerical value result {len(df)} rows ")
    
    out_path = Path(out_dir) / lab_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path/f"{lab_name}.parquet", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    db_path = "/shared/share_mala/zj2398/mimic/data/meds_v0.6_reader"
    out_dir = "/shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h"
    icu_stay_path = "/shared/share_mala/zj2398/mimic/data/raw_3.1/icu/icustays.csv"
    
    for lab_name, codes in TARGETS.items():
        write_lab_file(
            db_path, 
            lab_name, 
            icu_stay_path, 
            codes, 
            out_dir=out_dir, 
            num_threads=100
        )