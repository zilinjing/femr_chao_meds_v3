# SurvivEHR-on-FEMR Revision Plan

## Goal

Implement a SurvivEHR-style next-event pretraining loss on top of the existing FEMR tokenizer, processor, and backbone while keeping the comparison with MTPP as fair as possible.

## Constraints

- Keep the FEMR tokenizer, backbone, and downstream `compute_features()` / `LabeledSubjectTask` flow unchanged.
- Keep existing baseline code untouched where possible.
- Prefer minimal edits to the new SurvivEHR-specific files only.
- Match original SurvivEHR behavior when it does not conflict with a fair restricted-code comparison against MTPP.

## Clarified Decisions

### 1. MTPP Sampling vs SurvivEHR Coverage

- In MTPP, `desired_task_fraction` is a real keep probability for label positions.
- It is computed in the non-exact batching path and applied later inside `BatchCreator.add_subject(...)`.
- It subsamples context times, not task codes.
- At each retained context time, MTPP still predicts all selected task codes.
- This is useful for MTPP because each context is expensive and highly redundant.

Decision for SurvivEHR-on-FEMR:
- Do not inherit this non-exact sampling behavior.
- Keep all eligible next-event labels.
- Keep the `needs_exact()` interface and set it to `True`.

Reason:
- SurvivEHR next-event prediction is much cheaper per context than MTPP.
- Original SurvivEHR is conceptually trained on all adjacent valid transitions.
- With `needs_exact=False`, long subjects lose earlier valid labels even if no Bernoulli dropping occurs inside the final window.

### 2. Target Definition

- Numerical events:
  - use the raw event code directly if it is in the selected numerical task-code set
  - store the raw scalar value for Gaussian prediction
- Non-numerical events:
  - do not use ontology-parent collapsing
  - use the raw event code directly if it is in the selected non-numerical task-code set
- Out-of-set events:
  - skip them when searching for the target
  - define the target as the next future kept event that is inside the selected task-code universe
  - extend `delta_t` to that next valid modeled event
- Censoring:
  - use censoring only when there is no later valid modeled event before end of record

Reason:
- This is the cleanest restricted-code analogue of SurvivEHR for comparison against MTPP.
- It avoids ontology-dependent label rewriting.
- It avoids treating out-of-set immediate events as censoring.

### 3. Time Handling

- Use explicit year units for the training target:
  - `delta_years = delta_days / 365.25`
- Keep this as a documented training-time unit choice.
- Do not treat original `t_eval = linspace(0, 1, 1000)` as the normalization step.

Reason:
- Original SurvivEHR uses `t_eval` as a generation / evaluation grid, not as the training-time scaling operation inside the head.
- The repo suggests original ages were already scaled upstream before entering the survival head.
- Using years is a reasonable explicit approximation of the original convention while keeping the FEMR pipeline fixed.

### 4. Gaussian Value Loss

- Keep one Gaussian head per numerical code.
- Preserve patient-wise per-token aggregation, matching original SurvivEHR as closely as possible:
  - average over occurrences of a token within each patient
  - then average across patients in the batch
  - then sum across numerical token types
- Preserve the original missing-value behavior as closely as practical:
  - positions of the correct token with missing value keep the default `Normal(0, 1)` contribution
  - positions of other token types do not contribute after masking

Reason:
- This matches the clarified reading of `GaussianRegressionLayer`.
- The bigger remaining mismatches are target definition and batching, not the Gaussian reduction.

## Planned Revisions

### 1. Task Semantics in `SurvivEHRCausalTask`

- Remove ontology-parent collapsing for non-numerical events.
- Search forward to the next modeled event instead of always using the immediate next kept event.
- Emit censoring only when no later modeled event exists.
- Keep numerical events on direct raw-code matching with raw scalar values.

### 2. Batching / Label Coverage

- Change `SurvivEHRCausalTask.needs_exact()` to `True`.
- Keep `get_sampled_labels()` trivial or unused once exact coverage is enabled.
- Ensure every eligible next-event label is covered across windows for long subjects.

### 3. Time Units

- Normalize next-event deltas to years.
- Make that choice explicit in the task / head documentation.
- Keep the distinction clear between:
  - training-time target units
  - inference-time evaluation grid

### 4. Gaussian Head / Loss

- Keep the current subject-wise per-token aggregation unless a remaining mismatch is found.
- Verify that the implementation still reflects:
  - code-specific Gaussian heads
  - patient-wise averaging per token
  - summation across token types
  - default `Normal(0,1)` treatment for missing values of the correct token

### 5. Documentation

- Update `README_survivehr_causal.md` after code changes.
- Document the final choice for:
  - modeled event definition,
  - censoring definition,
  - exact vs non-exact batching,
  - time scaling convention,
  - Gaussian loss aggregation.

## Validation

- Run syntax checks on modified files.
- Run a small preprocessing pass and verify:
  - all eligible next-event labels are covered for long subjects,
  - non-numerical targets use raw task codes instead of ontology parents,
  - out-of-set intermediate events are skipped correctly,
  - censoring occurs only when no later modeled event exists,
  - time deltas are stored in years.
- Run one small train batch and inspect:
  - `event_label`,
  - `event_time`,
  - value-mask behavior for numerical tasks.

## Notes

- The fairness-critical choice is that “next event” means the next future kept event inside the selected MTPP code universe, not necessarily the immediate next kept event overall.
- The batching-critical choice is that SurvivEHR should not inherit MTPP’s non-exact sampled-label regime.
