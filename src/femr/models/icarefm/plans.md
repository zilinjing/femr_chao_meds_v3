# ICareFM Execution Plan

## Scope

Implement an ICareFM-style dynamic-threshold discrete survival baseline inside the existing FEMR `prepare.py` and `pretrain.py` flow. This is a plan only; no baseline code should be implemented until the plan is reviewed.

Primary constraints:

- Reuse the existing `CLMBRTokenizer`, backbone, `FEMRBatchProcessor`, HuggingFace dataset batching, and `Trainer` flow.
- Do not convert stays to a dense hourly grid for this baseline.
- Do not persist sampled thresholds or directions in `train_batches` or `val_batches`.
- Match MTPP's prediction-anchor policy as closely as possible: labels are attached to unique token timestamps and are subsampled by `processor.py`, not generated at every hourly time point.

Locked implementation decisions:

- Use all selected numerical target codes for each selected prediction anchor.
- Keep `ICareFMTask.needs_exact() == False`.
- Set `ICareFMTask.get_sampled_labels(length)` to MTPP's default `max(5, length // 10)`.
- Let `map_preliminary_batch_stats` and `subsample_task_fraction` control which prediction anchors are included, as in MTPP.
- Select target codes by `value_sample_count >= value_count_threshold`, where `value_sample_count` counts events with `event.numeric_value is not None` and `event.text_value is None`.
- Normalize the implementation loss globally over all valid query-bin weights.
- It is acceptable for a selected anchor to have no observed target measurement inside the 48-hour horizon. The original ICareFM objective uses a measurement-presence mask on the hourly grid; in this FEMR implementation, such anchors/queries should become zero-weight rather than being treated as negative labels.

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
- backbone type and size (transformer or mamba in `FEMRModel`),
- `tokens_per_batch`, sequence truncation, and mamba/transformer packing policy,
- optimizer and `TrainingArguments`,
- prediction-anchor subsampling policy in `processor.py`,
- downstream labels and evaluation script.

Do not use the original ICareFM hourly grid for the main MTPP comparison. Hourly anchors would give ICareFM many more prediction points than MTPP and would train/evaluate it on a different task distribution. Instead:

- A prediction anchor is a token position/timestamp selected by `BatchCreator.add_subject`.
- `ICareFMTask.needs_exact()` should return `False`.
- `ICareFMTask.get_sampled_labels(length)` should match MTPP's rule, currently `max(5, length // 10)`.
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
- Use separate train and eval collators. The train collator samples fresh thresholds/directions on every call. The eval collator samples from the same distribution but with a deterministic seed, initially `eval_seed=42`, so the validation dataset is repeatable across epochs/runs.
- The paper explicitly states that thresholds are randomly varied during training, and describes early stopping on validation loss, but does not specify a separate fixed validation-threshold protocol.
- To keep validation reproducible without persisting thresholds, seed validation draws from stable query identifiers plus `eval_seed` where available, or from deterministic query order within the eval collator as the first runnable slice.
- If validation loss is too noisy, average a small fixed number of deterministic threshold draws and report that choice.

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

- use `CLMBRTokenizer`; all predicted codes should be selected from original codes,
- scan the training database for observed numerical values,
- select target codes with `value_sample_count >= value_count_threshold`,
- define `value_sample_count` as the number of events where `event.numeric_value is not None` and `event.text_value is None`,
- store `target_codes`, `code_to_target_idx`, and value normalization stats,
- do not apply additional clinical curation or treatment exclusion unless explicitly requested,

Per subject, precompute observed target measurements:

```text
measurement = (time, target_idx, value_scaled)
```

Use an MTPP-style cursor rather than binary search for each anchor. Because
`BatchCreator.add_subject` visits candidate anchors in chronological order, the
task can maintain a cursor into the sorted selected numerical measurements and
advance it while `event_time <= current_date`. Measurements at exactly the
prediction timestamp are treated as context/past, not future labels. This avoids
repeated `O(log N)` searches and makes it cheap to scan/check the measurements
within `(current_date, current_date + T_surv]`.

At each label anchor `t`, store sparse future evidence rather than dense labels:

```text
future_sparse:
  indptr: [num_anchors + 1]
  target_idx: [num_future_measurements]
  bin_idx: [num_future_measurements]      # ceil((event_time - t) / 1 hour), 1..T_surv
  value_scaled: [num_future_measurements]
```

Do not store `current_sparse` or any `batch_current` fields in the next
implementation pass. Same-time selected numerical measurements are ignored for
label construction because they are likely already represented in the context
window at the prediction timestamp. The label target is therefore strict future
observed measurements only: `event_time > current_date`. This matches MTPP's
cursor behavior and avoids input-label leakage from same-time measurements.

`add_event(current_date, next_date, ...)` should return one label anchor when at least one selected numerical target has a future observed measurement after `current_date`, even if the next measurement is beyond `T_surv`. The returned anchor is still attached to the previous token, exactly as in the existing processor. If no selected target has an observed measurement inside the survival horizon, the later query construction should produce zero valid weights rather than a false negative.

This is analogous to MTPP's sparse task storage, but the contents differ. MTPP stores the next future time and value per code because its label is a next-event or next-value-bin target. ICareFM stores observed strict-future measurements because the event definition depends on a threshold sampled later during collation.

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

Query definition:

- A query is one `(anchor, target_code k, tau, direction)` tuple.
- If a batch has `A` selected prediction anchors and `K` selected numerical target codes, the conceptual query set has `Q = A * K` queries.
- Threshold sampling granularity is configurable:
  - `per_query`: sample independent `tau` and `direction` for every `(anchor, target_code)` query. This is closest to the current reference implementation and maximizes dynamic-threshold diversity.
  - `batch_shared`: sample one `tau` and one `direction` per collated batch and apply them to all queries in the batch. This is less diverse but simpler to vectorize and is acceptable as an efficiency trade-off if it is recorded in experiment metadata.
  - Optional middle ground: `batch_target_shared`, one `tau`/`direction` per target code per batch.
- For efficiency, zero-weight queries may be dropped after label construction because they contribute exactly zero to the global masked loss. This does not change the objective.

Label logic per query:

- Ignore measurements at exactly the anchor time. They are not future labels and should not trigger current-value masking.
- If the first future crossing occurs in bin `r_star`, set `y[r_star - 1] = 1`, weights through `r_star` to one, and later weights to zero.
- If future measurements exist in the horizon but none cross, set all `y=0` and all weights to one.
- If no future measurement for this target exists in the horizon, set all weights to zero.

This uses observed measurements only. Absence of a measurement is not a normal-value label.

## Model Head

Add `ICareFMTaskHead` in the architecture layer or a small dedicated file.

Paper-faithful head design:

```text
Each variable k has a learnable embedding u_k with dimension R^{d/2}.
The direction embedding has dimension R^{d/2}.
The hidden dimension of the history is d.

condition_k = concat(tau * u_k, direction_embedding[delta])  # dimension d
input = concat(H_t, condition_k)                              # dimension 2d
logits = MLP_k(input) -> R^{T_surv}  #  MLP_k is a variable-specific two-layer network with SiLU activations. Should be different weights for different codes
lambda = sigmoid(logits)
```

`MLP_k` is variable-specific, matching Supplement B.3. If the support-threshold target set becomes too large for one MLP per variable, a shared-head variant can be treated as an explicit scalability ablation rather than the main ICareFM baseline.

Loss:

```text
lambda = sigmoid(logits)

L = - sum_{query q=(t,k,tau,delta)} sum_{r=1..T_surv}
        w_{q,r} * [
          y_{q,r} * log(lambda_{q,r})
          + (1 - y_{q,r}) * log(1 - lambda_{q,r})
        ]
    / (sum_{q,r} w_{q,r} + eps)
```

This is the implementation form of Supplement Eq. (4), with patient index suppressed and with global normalization over valid observed query-bin weights. The paper writes the objective as a masked sum per target head with a final average over heads; the global normalization is the chosen engineering convention for this baseline.

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

4. Replace per-anchor binary-search lookup with an MTPP-style per-subject cursor.
   Verify that same-time measurements are ignored for labels, no separate `current_sparse` arrays are emitted, and future horizon scanning remains chronological over `event_time > current_date`.

5. Implement separate train/eval collators for threshold sampling.
   Verify that train cleanup changes across calls, while eval cleanup with `eval_seed=42` is deterministic.

6. Refactor `cleanup` toward vectorized query construction.
   First keep the existing loop implementation as reference. Then test vectorized versions that expand sparse measurements into `(anchor, target, bin, value)` rows, compute crossing masks in bulk, and use per-query reduction to find the first crossing bin. Prefer `per_query` threshold sampling by default for within-batch diversity;

7. Implement `ICareFMTaskHead` and run one synthetic forward/backward pass.
   Verify finite loss and nonzero gradients when valid weights exist.

8. Wire `prepare.py` and `pretrain.py`.
   Verify on a tiny MEDS subset before running full pretraining.

## Resolved Decisions or future plans

- Use full `(anchor, target_code)` expansion for selected anchors.
- Use dynamic thresholds in validation. Make validation deterministic with a separate eval collator and `eval_seed=42`.
- Select target codes by numerical-value support only: `value_sample_count >= value_count_threshold`.
- Normalize the loss globally over all valid query-bin weights.
- Allow anchors with no within-48-hour observed target measurement to produce zero-weight queries, matching the paper's masked observed-measurement supervision rather than forcing negatives.
- `_bisect_right_time` has been replaced with an MTPP-style cursor in `ICareFMTask.add_event`.
- Remove `current_sparse`/`batch_current`. Same-time selected numerical measurements should be treated as context/past and ignored for labels, not as first-interval future events.
- Vectorized `cleanup` is implemented with default `per_query` threshold sampling: different `tau` and `direction` for each materialized `(anchor, target)` query in a batch. `batch_shared` or `batch_target_shared` should be used only as explicit efficiency ablations.
