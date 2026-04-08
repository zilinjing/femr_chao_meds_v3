# SurvivEHR causal debug notes

Date: 2026-04-07

## Failure reproduced

The batch-preparation pipeline failed inside `SurvivEHRCausalTask.add_event(...)` with:

- `AssertionError: Task/event alignment mismatch: expected next_date=..., received ...`

The mismatch appeared during `processor.convert_subject(..., actually_add=False)` while the MEDS reader was mapping subjects in parallel.

## Root cause

`BatchCreator.add_subject(...)` walks every kept event and calls the task with:

- `current_date = previous kept event timestamp`
- `next_date = current kept event timestamp`

However, the processor deliberately emits **no label** when two consecutive kept events share the exact same timestamp:

```python
if next_date is None or next_date == current_date:
    return 0
```

`SurvivEHRCausalTask` was returning early on that condition **before** advancing its internal `_next_event_cursor`. That meant:

1. same-timestamp events were still consumed by the processor and became the new `last_time`
2. the task cursor did not consume them
3. on the next later event, the task still expected the earlier same-timestamp event and raised an alignment assertion

## Code changes

### 1. `src/femr/models/tasks/survivehr_causal.py`

- Recorded `subject_id` during `start_subject(...)` so any future alignment failure includes the subject in the exception.
- Changed `add_event(...)` to:
  - return early only when `next_date is None`
  - always consume and validate the current kept event against `next_date`
  - emit no label when `next_date == current_date`, but still advance the cursor

This keeps task state aligned with the processor’s traversal semantics.

- Added `__getstate__(...)` to strip transient per-subject/per-batch fields before serialization.

This is required because `datasets.Dataset.from_generator(...)` fingerprints the generator inputs by pickling them. Without dropping `_kept_events`, the task can retain raw `meds_reader.Event` objects from the last preliminary subject and fail with `_pickle.PicklingError: Can't pickle 'Event' object`.

### 2. `src/femr/models/processor.py`

- Moved `self.tokenizer.start_subject()` ahead of `task.start_subject(...)`.

`SurvivEHRCausalTask.start_subject(...)` uses the tokenizer to precompute kept events. The tokenizer is currently effectively stateless here, but initializing it first makes the processor/task contract correct and avoids future drift if tokenizer subject state becomes meaningful.

### 3. `tests/test_survivehr_causal.py`

- Added a regression test covering repeated identical timestamps followed by a later event.
- The test verifies that:
  - the same-timestamp event is consumed without creating a label
  - the cursor stays synchronized
  - the later event produces the expected label and time delta instead of raising
- Added a serialization regression test proving transient subject state is removed before pickling.

## Verification plan

1. Run the new unit regression test.
2. Rerun the original `prepare_survivehr_causal` command.
3. If new failures appear, iterate on the next concrete stack trace.
