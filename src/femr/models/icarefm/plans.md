# ICareFM Execution Plan

## Scope

Implement an ICareFM-style dynamic-threshold discrete survival baseline inside the existing FEMR `prepare.py` and `pretrain.py` flow. This is a plan only; no baseline code should be implemented until the plan is reviewed.

Primary constraints:

- Reuse the existing tokenizer, backbone, `FEMRBatchProcessor`, HuggingFace dataset batching, and `Trainer` flow.
- Do not convert stays to a dense hourly grid for this baseline.
- Do not persist sampled thresholds or directions in `train_batches` or `val_batches`.
- Match MTPP's prediction-anchor policy as closely as possible: labels are attached to unique token timestamps and are subsampled by `processor.py`, not generated at every hourly time point.

## Paper Facts To Preserve

From `icarefm_new.pdf` and Supplementary Section B in `media-1.pdf`, the relevant model idea is:

- ICareFM encodes patient history with a time-step encoder plus causal sequence encoder, then uses time-to-event prediction heads.
- The original implementation operates on an imputed hourly grid with a measurement-presence mask.
- For each target clinical variable, a hazard head predicts a discrete survival distribution over a future horizon, commonly 48 hours.
- The event query is threshold conditioned: variable `k`, threshold `tau`, direction `delta`, and horizon `h`.
- Thresholds are randomly varied during training so the model learns a range of abnormal value regions instead of one fixed endpoint.
- Inference converts hazards to cumulative risk:

```text
F_k(h | H_t, tau, delta) = 1 - product_{r=1..h}(1 - hazard_r)
```

The main deliberate deviation for fair comparison with MTPP is anchor construction: use FEMR token timestamps instead of ICareFM's dense hourly time grid.

## Fair Comparison Position

The fair comparison target should isolate the objective, not preprocessing density.

Use the same:

- train/validation/test subject splits,
- tokenizer family and vocabulary used by the compared MTPP run,
- backbone type and size,
- `tokens_per_batch`, sequence truncation, and mamba/transformer packing policy,
- optimizer and `TrainingArguments`,
- prediction-anchor subsampling policy in `processor.py`,
- downstream labels and evaluation script.

Do not use the original ICareFM hourly grid for the main MTPP comparison. Hourly anchors would give ICareFM many more prediction points than MTPP and would train/evaluate it on a different task distribution. Instead:

- A prediction anchor is a token position/timestamp selected by `BatchCreator.add_subject`.
- `ICareFMTask.needs_exact()` should return `False`.
- `ICareFMTask.get_sampled_labels(length)` should initially match MTPP's rule, currently `max(5, length // 10)`, unless an experiment explicitly varies anchor budget.
- `map_preliminary_batch_stats` should continue to compute `desired_task_fraction` from `label_indices`, so MTPP and ICareFM use the same subject-window and label-subsampling machinery.

Keep the ICareFM objective mostly intact by preserving:

- numerical target variables only,
- dynamic threshold and direction conditioning,
- first-threshold-crossing survival labels,
- discrete hourly-offset hazard bins up to `T_surv=48`,
- hazard-to-cumulative-risk inference.

This makes the implemented baseline "ICareFM-style on FEMR/MTPP anchors", not an exact reproduction of the original hourly-grid ICareFM.

## Dynamic Threshold Policy

Thresholds must be sampled after loading prebuilt batches, not inside `prepare.py`.

Persist in `train_batches` and `val_batches`:

- anchor label positions,
- future observed numerical measurements needed to construct labels,
- current observed values at the anchor timestamp if available,
- target-code metadata and normalization statistics in the pickled task.

Do not persist:

- sampled `tau`,
- sampled direction `delta`,
- final threshold-crossing labels `y`,
- final masks `w`.

Recommended sampling location:

- `ICareFMTask.cleanup(batch)` constructs query-level tensors dynamically when `processor.collate` runs.
- The task head receives sampled queries and computes masked BCE.

Training behavior:

- Sample fresh `tau` and `delta` on every collation call.
- Because saved batches are reused across epochs, this automatically varies thresholds across epochs.

Validation behavior:

- Also generate thresholds on the fly, so validation batches do not contain fixed thresholds.
- For stable model selection, use either a deterministic validation seed based on `(subject_id, timestamp, code_idx, eval_seed)` or average validation loss across a small fixed number of dynamic threshold draws.
- Report which policy is used. A single fully random validation draw is noisy and can make early stopping unstable.

Initial threshold distribution:

```text
delta ~ Bernoulli(0.5)
tau_mag ~ Uniform(0, tau_max)
if delta == high: tau = +tau_mag, event is value_scaled >= tau
if delta == low:  tau = -tau_mag, event is value_scaled < tau
```

Use `tau_max=4.0` in standardized units for the first runnable slice, matching `icarefm_plan.md`. Keep this configurable because the paper states thresholds are varied over the value range but does not fully determine one distribution for this MEDS setting.

## Data Representation

Add a new task, likely `femr/models/tasks/icarefm_task.py`.

Task fitting should:

- scan the training database for observed numerical values,
- select target codes with `count >= value_count_threshold`,
- store `target_codes`, `code_to_target_idx`, and value normalization stats,
- exclude non-numerical events and treatment-like targets unless explicitly requested,
- use the same tokenizer and ontology choices as the comparison baseline.

For MTPP comparison, prefer the same tokenizer family as MTPP. If using `CLMBRTokenizer` for a SurvivEHR-style comparison, report that this changes the input representation and is not an objective-only comparison to MTPP.

Per subject, precompute observed target measurements:

```text
measurement = (time, target_idx, value_scaled)
```

At each label anchor `t`, store sparse future evidence rather than dense labels:

```text
future_sparse:
  indptr: [num_anchors + 1]
  target_idx: [num_future_measurements]
  bin_idx: [num_future_measurements]      # ceil((event_time - t) / 1 hour), 1..T_surv
  value_scaled: [num_future_measurements]

current_sparse:
  indptr: [num_anchors + 1]
  target_idx: [num_current_measurements]
  value_scaled: [num_current_measurements]
```

`add_event(current_date, next_date, ...)` should return one label anchor only when at least one selected numerical target has a future observed measurement after `current_date`. The returned anchor is still attached to the previous token, exactly as in the existing processor.

## Query-Level Batch Construction

In `ICareFMTask.cleanup`, convert sparse future evidence into query-level training tensors.

Use query-level rows rather than a mandatory dense `[num_anchors, K, T_surv]` tensor:

```text
query_anchor_idx: [Q]       # row into features already selected by label_indices
query_target_idx: [Q]
query_tau: [Q]
query_direction: [Q]        # 1 high, 0 low
query_y: [Q, T_surv]
query_weight: [Q, T_surv]
```

Initial query sampling:

- Fair but more expensive option: generate one query per `(anchor, target_code)` with future evidence in the horizon.
- Cheaper option: sample `queries_per_anchor` target codes from eligible future-evidence codes.

For the first runnable baseline, use `queries_per_anchor=1` or `2` only if full code expansion is too expensive. If code sampling is used, record the average query count and compare against MTPP under a matched compute or matched anchor budget. The cleanest objective-only comparison is full expansion over the same numerical target set, if memory allows.

Label logic per query:

- If the current observed value at anchor time crosses the sampled threshold, set all weights to zero.
- If the first future crossing occurs in bin `r_star`, set `y[r_star - 1] = 1`, weights through `r_star` to one, and later weights to zero.
- If future measurements exist in the horizon but none cross, set all `y=0` and all weights to one.
- If no future measurement for this target exists in the horizon, set all weights to zero.

This uses observed measurements only. Absence of a measurement is not a normal-value label.

## Model Head

Add `ICareFMTaskHead` in the architecture layer or a small dedicated file.

Minimal shared-head design:

```text
input = concat(H_t, code_embedding[k], tau_embedding(k, tau), direction_embedding[delta])
logits = MLP(input) -> [T_surv]
hazard = sigmoid(logits)
```

This is easier than one MLP per target code and fits the existing `embedding.py` pattern better. If the target-code count is small, code-specific heads can be added later as a fidelity experiment.

Loss:

```text
bce = binary_cross_entropy_with_logits(logits, query_y, reduction="none")
loss = sum(query_weight * bce) / (sum(query_weight) + eps)
```

Track metrics:

- number of anchors,
- number of sampled queries,
- fraction of zero-weight queries,
- event query fraction,
- mean valid bins per query,
- loss by target code or at least top frequent target codes.

## Integration Points

Minimal files to touch when implementing:

- `femr/models/tasks/icarefm_task.py`: new task, target fitting, sparse future evidence, dynamic cleanup.
- `femr/omop_meds_tutorial/prepare.py`: add `loss_type == "icarefm"` and task serialization.
- `femr/omop_meds_tutorial/pretrain.py`: add CLI choice and load task normally.
- `femr/models/architecture/embedding.py`: add `ICareFMTaskHead` branch for `loss_type == "icarefm"`.
- `femr/models/config.py`: no change expected unless the task config needs stricter structured fields.
- `tests/models/`: add small deterministic tests for labels, dynamic sampling, and one forward pass.

Avoid broad refactors of `processor.py`. The existing `needs_exact=False` path and `subsample_task_fraction` already provide the MTPP-like anchor policy.

## Minimal Runnable Slice

1. Implement pure label-building utilities for one anchor and one code.
   Verify with unit tests for first event, no event with future measurement, no future measurement, and current already abnormal.

2. Implement `ICareFMTask.fit_pretraining_task_info` and one synthetic-subject `convert_subject`.
   Verify that saved batch data contains future/current measurements but no sampled thresholds.

3. Implement `cleanup` dynamic sampling.
   Verify that calling cleanup twice on the same saved batch can produce different `tau` or `direction`, while tensor shapes are stable.

4. Implement `ICareFMTaskHead` and run one synthetic forward/backward pass.
   Verify finite loss and nonzero gradients when valid weights exist.

5. Wire `prepare.py` and `pretrain.py`.
   Verify on a tiny MEDS subset before running full pretraining.

## Fairness Checks Before Reporting Results

Before comparing to MTPP, record:

- tokenizer type and vocabulary path for both runs,
- target numerical codes and count,
- total train and validation anchors before and after subsampling,
- effective query count per epoch,
- `T_surv`, threshold distribution, and `tau_max`,
- random seeds for subject split, batch generation, and dynamic threshold sampling,
- model parameter count,
- wall-clock budget and number of optimizer steps.

A comparison is not fair if ICareFM receives dense hourly anchors while MTPP receives only unique timestamp anchors. A comparison is also not objective-only if ICareFM uses a different tokenizer/backbone or substantially more prediction queries without reporting that budget difference.

## Open Decisions

- Whether to use full `(anchor, target_code)` expansion or sampled target codes per anchor.
- Whether validation should use deterministic dynamic thresholds or multiple random threshold draws.
- Whether target-code selection should exactly match MTPP numerical tasks or use ICareFM's original-style curated clinical target set. For a fair MTPP comparison, match MTPP numerical targets where possible.
- Whether to normalize loss globally over all query-bin weights or per target code. Per-code normalization is less dominated by dense labs/vitals, but it is a change from naive query averaging and should be reported.

