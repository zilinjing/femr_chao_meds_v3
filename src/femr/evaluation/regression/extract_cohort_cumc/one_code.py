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
            Record (subject_id, prediction_time, event_time, regression_value)
    
    Args:
        subject_iter: Iterator of subjects from MEDS database
        target_code: The lab code to process
        lab_name: Human-readable name for the lab
        visits_dict: Dictionary mapping person_id -> numpy array of visit windows
    
    Returns:
        List of dicts with keys: subject_id, prediction_time, event_time, regression_value
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
                        'regression_value': event_values[idx]
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
