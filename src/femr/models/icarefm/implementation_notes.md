# ICareFM Implementation Notes

## Changed Files

- `models/tasks/icarefm_task.py`
  - Added `ICareFMTask`.
  - Fits numerical target codes using `value_sample_count >= value_count_threshold`, where counts only include events with `numeric_value is not None` and `text_value is None`.
  - Stores global mean/std normalization for selected target codes.
  - Uses `needs_exact() == False` and `get_sampled_labels(length) == max(5, length // 10)` so `processor.py` keeps the same MTPP-style anchor subsampling path.
  - Saves sparse strict-future observed numerical measurements in prepared batches, without persisting sampled thresholds, directions, or final labels.
  - Ignores same-time selected numerical measurements for labels because they are context/past relative to the prediction point.
  - Removed `current_sparse` / `batch_current`; there is no current-value masking in label construction.
  - Replaced per-anchor binary search with a per-subject chronological cursor that advances past measurements with `event_time <= current_date`.
  - Vectorized `cleanup()` over sparse `(anchor, target, bin, value)` rows and samples independent `(tau, direction)` for every materialized `(anchor, target)` query.
  - Added optional deterministic cleanup seeding through `threshold_seed`; seeded cleanup hashes sparse batch contents plus the seed to keep validation-style sampling repeatable without storing thresholds in batch files.

- `models/architecture/embedding.py`
  - Added `ICareFMTaskHead`.
  - Uses a per-variable threshold embedding `u_k`, direction embedding, variable-specific two-layer MLP with SiLU, and sigmoid-hazard BCE through `binary_cross_entropy_with_logits`.
  - Added the `loss_type == "icarefm"` task-head branch.

- `training/prepare.py`
  - Added `icarefm` as a supported loss type.
  - Uses `CLMBRTokenizer` for ICareFM, consistent with the plan and the existing SurviveHR/CLMBR path.
  - Serializes `icarefm_task.pkl`.

- `training/pretrain.py`
  - Added `icarefm` as a supported loss type.
  - Loads `CLMBRTokenizer` for ICareFM.
  - Enables `ddp_find_unused_parameters` for ICareFM because the per-variable heads are sparsely used by each batch.
  - Reuses the existing metrics callback to log `ICareFMTaskHead.last_metrics`.
  - Added `--icarefm_eval_seed` with default `42`.
  - Uses a separate eval collator for ICareFM. Training collation keeps fresh unseeded thresholds, while eval collation uses a deep-copied task with deterministic seeded cleanup.

- `tests/test_icarefm.py`
  - Added synthetic tests for first-event label construction, no-crossing labels, strict-future sparse evidence, same-time measurement exclusion, beyond-horizon zero-query rows, dynamic cleanup, and one ICareFM head forward pass.

## Current Behavior

- Prepared train/val batches contain sparse strict-future observed measurements only:

```text
future_indptr: [num_anchors + 1]
future_target_idx: [num_future_measurements]
future_bin_idx: [num_future_measurements]
future_value_scaled: [num_future_measurements]
```

- Same-time measurements are treated as already available context and are not labels.
- Thresholds and directions are sampled during collation through `ICareFMTask.cleanup()`, so reusing the same saved batch across epochs can produce different query thresholds.
- Current sampling granularity is `per_query`: each materialized `(anchor, target)` query gets its own `tau` and `direction`.
- `cleanup()` materializes query rows only for `(anchor, target)` pairs with at least one observed strict-future measurement inside `T_surv`; missing target codes are implicit zero-weight queries and are not sent to the head.
- The loss is globally normalized over valid query-bin weights:

```text
L = - sum_q sum_r w_qr * BCE(lambda_qr, y_qr) / (sum_q sum_r w_qr + eps)
```

- Anchors whose next selected numerical measurement is after `T_surv` can be selected by `add_event()` but produce no materialized queries during `cleanup()`.

## Uncertainties

- Validation determinism depends on the sparse batch contents and `--icarefm_eval_seed`. If validation dataset order or batch construction changes, deterministic samples remain reproducible for the new sparse batches but are not numerically comparable to old batch files.
- `add_event()` declares an anchor valid when any selected numerical target has a future observed measurement after the anchor time. Only measurements within `T_surv` are stored. This follows `plans.md`, but it means anchors whose next selected measurement is beyond the horizon can later become zero-query batches.
- The main head uses one MLP per selected target code, matching Supplement B.3. If the selected numerical target set is very large, memory/compute may be high; a shared-head variant should be treated as a separate scalability ablation.
- `t_surv=48` and `tau_max=4.0` are task defaults and are not exposed as CLI flags yet.
- The tests cover synthetic task/head behavior only. I did not run full `prepare.py` or `pretrain.py` on a MEDS subset.

## Verification

- Passed:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
  /user/zj2398/.conda/envs/tte/bin/python -s -m pytest -q \
  /user/zj2398/femr_chao_meds_v3/tests/test_icarefm.py -p no:cacheprovider
```

- Passed:

```text
PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
  /user/zj2398/.conda/envs/tte/bin/python -s -m py_compile \
  models/tasks/icarefm_task.py models/architecture/embedding.py \
  training/prepare.py training/pretrain.py
```

- Existing adjacent test did not collect:

```text
tests/test_survivehr_causal.py imports femr.models.tasks.survivehr_causal,
but that module is not present in this checkout.
```
