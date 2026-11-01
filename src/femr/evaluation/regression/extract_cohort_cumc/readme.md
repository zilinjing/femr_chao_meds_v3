# Regression Cohort Generation for Lab Values Prediction

## Overview

This pipeline generates cohorts for regression tasks predicting future lab values from Electronic Health Records (EHR) data. For each outpatient visit, it identifies whether a qualifying lab event occurs 2-5 months after the visit ends, enabling prediction of future lab measurements.

## Task Definition

**Prediction Time**: End of each outpatient visit (`visit_end_datetime`)

**Prediction Target**: First qualifying lab event occurring 2-5 months after the prediction time

**Qualifying Events**: Lab events that satisfy ALL of the following:
- Event code matches one of the target lab codes (Platelets, Creatinine, or Bilirubin)
- Event has a valid timestamp (`event.time is not None`)
- Event has a valid numerical value (`event.numerical_value` is a float)
- Event occurs within 2-5 months after the prediction time
- Event falls within any outpatient visit window (start ≤ event_time ≤ end)

## Output Schema

Each cohort is saved as a parquet file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int64 | Unique patient identifier |
| `prediction_time` | datetime64[ns] | End time of the outpatient visit (prediction time point) |
| `event_time` | datetime64[ns] | Time of the first qualifying lab event |
| `regression_value` | float64 | Numerical value of the lab measurement |

## Target Lab Codes

| LOINC Code | Lab Name | Output File |
|------------|----------|-------------|
| `LOINC/26515-7//UCUM/10*9/L` | Platelets | `cohort_Platelets.parquet` |
| `LOINC/2160-0//UCUM/mg/dL` | Creatinine | `cohort_Creatinine.parquet` |
| `LOINC/1968-7//UCUM/mg/dL` | Bilirubin | `cohort_Bilirubin.parquet` |

## Algorithm

```
For each patient:
    For each outpatient visit:
        prediction_time = visit_end_datetime
        window_start = prediction_time + 2 months
        window_end = prediction_time + 5 months
        
        For each lab event in [window_start, window_end]:
            If event is within any visit window AND has valid numerical value:
                Record (subject_id, prediction_time, event_time, value)
                Break (take first qualifying event only)
```

## Performance Optimizations

The implementation includes several optimizations for processing large-scale data:

1. **Timestamp-based comparisons**: All datetime objects are converted to Unix timestamps (int64) for faster numerical comparisons instead of datetime object comparisons.

2. **NumPy arrays for visits**: Visit windows are stored as NumPy arrays `[start_ts, end_ts]` sorted by end time, enabling efficient binary search.

3. **Binary search for event windows**: Events are pre-indexed by timestamp, allowing O(log n) lookup for the 2-5 month window instead of linear scanning.

4. **Vectorized interval checking**: Uses NumPy broadcasting to check if events fall within any visit window.

5. **Early termination**: Stops searching after finding the first qualifying event for each prediction time.

6. **Memory-efficient processing**: Processes subjects in parallel chunks, aggregating results incrementally.

## Usage

```bash
python generate_regression_cohort.py
```

### Configuration

Edit the following paths in `main()`:

```python
meds_path = "/path/to/meds/database"
visits_parquet_path = "/path/to/visits.parquet"  
output_dir = "/path/to/output/cohorts"
```

### Parameters

- `num_threads`: Number of parallel workers for processing (default: 40)
- `WINDOW_START_DAYS`: Start of prediction window in days (default: 60, ~2 months)
- `WINDOW_END_DAYS`: End of prediction window in days (default: 150, ~5 months)

## Input Requirements

### MEDS Database
- Standard MEDS format with subjects containing events
- Events must have: `code`, `time`, `numerical_value` attributes
- Events should be sorted by time within each subject

### Visits Parquet
Required columns:
- `person_id`: Patient identifier (must match MEDS `subject_id`)
- `visit_start_datetime`: Visit start time
- `visit_end_datetime`: Visit end time
- `visit_concept_id`: Should be 9202 for outpatient visits

## Output

Cohorts are saved to the specified output directory:
```
output_dir/
├── cohort_Platelets.parquet
├── cohort_Creatinine.parquet
└── cohort_Bilirubin.parquet
```

## Expected Statistics

The pipeline prints processing statistics including:
- Number of visits and unique patients loaded
- Per-worker processing statistics
- Final cohort sizes for each lab code
- Total samples generated

## Dependencies

- numpy
- pandas
- meds_reader
- pyarrow (for parquet I/O)

## Notes

- If a prediction time has no qualifying events in the 2-5 month window, it is excluded from the cohort
- If multiple events qualify within the window, only the first (earliest) event is included
- Events must occur within a visit window to be considered valid (ensuring clinical context)
- The pipeline handles timezone-aware datetimes by converting to naive UTC