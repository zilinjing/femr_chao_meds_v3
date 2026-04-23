# SurvivEHR Revision Notes

This note records the decisions made while revising the SurvivEHR-style loss on top of the current FEMR/MTPP setup, plus the remaining uncertainties.

## Implemented Decisions

### 1. Label Semantics

- The target is the next future kept event inside the selected MTPP task-code universe.
- Intermediate kept events outside the modeled code set are skipped.
- Censoring is used only when no later modeled event exists before end of record.

Why:
- This keeps the comparison inside the same modeled code universe as MTPP.
- It avoids treating out-of-set immediate events as censored outcomes.

### 2. Numerical vs Non-Numerical Codes

- Numerical codes use direct raw-code matching first, then nearest modeled ancestor fallback.
- Non-numerical codes also use direct raw-code matching first, then nearest modeled ancestor fallback.
- The task still returns one modeled code or none for each event.

Why:
- The goal is to compare loss functions while keeping the tokenizer/backbone/task-code universe fixed.
- This preserves coverage for task codes that only appear through ontology mappings while keeping the label target single-valued.

### 3. Label Coverage

- `SurvivEHRCausalTask.needs_exact()` now returns `True`.
- The causal task no longer inherits the non-exact MTPP sampling regime.

Why:
- MTPP samples context times because each context predicts many tasks and is expensive.
- The SurvivEHR next-event objective is cheaper per context and should keep all eligible labels.

### 4. Time Units

- The code keeps the existing head-side normalization by dividing `event_time` by `time_scale_days`.
- Default `time_scale_days=365.25` means training is effectively done in years.

Why:
- This is functionally consistent with the intended year-scale training target.
- Keeping the normalization in the head avoided unnecessary churn in the task/batch format.

### 5. Gaussian Value Loss

- The current value loss keeps one Gaussian head per numerical code.
- Loss reduction is patient-wise per token, then averaged across patients, then summed across token types.
- Missing values for the correct numerical token keep the default `Normal(0,1)` contribution.

Why:
- This matches the original SurvivEHR reduction logic closely enough without adding complexity.

## Remaining Uncertainties

### 1. Ancestor Tie Breaking

The revised task chooses the nearest modeled ancestor, with task-order tie breaking if multiple modeled ancestors exist at the same ontology distance.

Risk:
- This is deterministic and simple, but it is still an implementation choice rather than something defined by the original SurvivEHR codebase.

### 2. Censoring Horizon

The revised task uses the subject’s final observed event time as the censoring horizon.

Risk:
- This is reasonable and aligns with an end-of-record interpretation, but it is still an adaptation of the original full-vocabulary next-token SurvivEHR setting.

### 3. Time Normalization Placement

Time normalization is still performed in the head instead of the task.

Risk:
- This is mathematically simple and explicit, but it differs from a design where the task would emit already-normalized year deltas.

## Files Changed

- `src/femr/models/tasks/survivehr_causal.py`
- `README_survivehr_causal.md`
- `plans.md`
- `survivehr_revision_notes.md`
