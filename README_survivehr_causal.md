# SurvivEHR Causal Pretraining Implementation

This note documents the current SurvivEHR-style causal objective in `femr_chao_meds_v3`.

The active implementation is now integrated into the standard FEMR pretraining flow:

- `src/femr/omop_meds_tutorial/prepare.py`
- `src/femr/omop_meds_tutorial/pretrain.py`
- `src/femr/models/processor.py`
- `src/femr/models/architecture/embedding.py`

The active command-line loss name is:

```text
survivehr
```

## Active Files

### `src/femr/models/tasks/survivehr_task.py`

This is the active task implementation.

It replaces the earlier MTPP-derived `SurvivEHRCausalTask` design with a simpler CLMBR-style next-token task:

- Uses `CLMBRTokenizer`.
- Uses `tokenizer.original_code_to_id` as the competing-risk label vocabulary.
- Predicts the immediate next observed raw event code.
- Does not use ontology parent mapping for labels.
- Does not emit censor labels during pretraining.
- Keeps label `0` reserved for censoring because `survivehr_head.py` still supports that case.
- Emits observed event labels as `original_code_id + 1`.
- Emits `event_time` as the current-to-next event delta in days.
- Emits `event_value` and `value_mask` for optional Gaussian value regression.

Numerical value-code fitting:

- Runs on `main_database`, meaning all non-`held_out` subjects.
- Counts events where `event.code == code`, `event.numeric_value is not None`, and `event.text_value is None`.
- Keeps a code for value regression if it has at least `value_count_threshold` valid numeric observations.
- Uses sampled per-code 1st and 99th percentiles as normalization bounds.
- Normalizes valid values as `(value - lower) / (upper - lower) - 0.5`.
- Treats missing, `NaN`, non-float, text-valued, or out-of-bound values as missing.

Missing values for selected numerical codes:

- The task still emits the next-code label.
- It emits `event_value=NaN` and `value_mask=False`.
- The head still counts that selected numerical-code row in the value loss through the default `Normal(0, 1).log_prob(0)` contribution.

### `src/femr/models/architecture/survivehr_head.py`

This is the active loss implementation.

The head has two parts:

- `ODESurvMultiple`: continuous-time competing-risk next-event likelihood.
- `GaussianValueHeads`: optional per-code Gaussian value regression.

Important label convention:

- `event_label == 0` means censoring.
- `event_label in 1..K` means an observed original-code event.
- `SurvivEHRTask` currently emits only labels in `1..K`.

Important value-loss behavior:

- `GaussianValueHeads.loss()` loops over selected numerical risk IDs.
- For every selected numerical-code label, it initializes log probability to `Normal(0, 1).log_prob(0)`.
- If `value_mask=True`, it replaces that default with the learned Gaussian log probability for the normalized value.
- If `value_mask=False`, it leaves the default contribution in place.
- It averages per patient and then across patients, matching the original SurvivEHR reduction more closely than a plain per-row average.

Recent head fix:

- Same-time event noise is now applied before clamping event times to `1e-6`.
- This matters because same-time transitions should get the small exponential noise rather than all becoming exactly the clamp value.

### `src/femr/models/processor.py`

The standard FEMR batch processor is now aware of `SurvivEHRTask`.

For normal hierarchical tasks it still passes hierarchical feature IDs.

For `CLMBRTask` and `SurvivEHRTask`, it instead uses:

```python
tokenizer.get_event_code_token(event)
```

For `SurvivEHRTask`, it also passes:

- `event.numeric_value`
- `event.text_value`

This lets the backbone input remain the usual CLMBR hierarchical embedding-bag representation, while the task label remains the raw original event-code ID.

### `src/femr/omop_meds_tutorial/prepare.py`

`prepare.py` now accepts:

```text
--loss_type survivehr
```

For `survivehr`, it:

- trains or loads `CLMBRTokenizer`, same as `ntp`;
- fits `SurvivEHRTask` on `main_database`;
- saves `survivehr_task.pkl`;
- writes the usual `train_batches` / `val_batches` for transformer, or `train_batches_mamba` / `val_batches_mamba` for Mamba.

### `src/femr/omop_meds_tutorial/pretrain.py`

`pretrain.py` now accepts:

```text
--loss_type survivehr
```

For `survivehr`, it:

- loads `CLMBRTokenizer`;
- loads `survivehr_task.pkl`;
- builds the standard transformer or Mamba config;
- instantiates `femr.models.architecture.embedding.FEMRModel` with `loss_type="survivehr"`.

### `src/femr/models/architecture/embedding.py`

The standard unified FEMR model now handles `loss_type="survivehr"`.

Changes:

- Imports `SurvivEHRCausalTaskHead`.
- `create_task_head()` returns `SurvivEHRCausalTaskHead` for `loss_type == "survivehr"`.
- `forward()` adds `label_subject_ids` to the task batch for `survivehr`, because the value loss averages by patient.

This works for both transformer and Mamba because the head is attached after the backbone hidden states are produced.

## Legacy Files

### `src/femr/models/survivehr/survivehr_causal.py`

This is the older standalone task implementation.

It is not used by the current `prepare.py --loss_type survivehr` or `pretrain.py --loss_type survivehr` path.

It was an earlier MTPP-derived design:

- Built a `SurvivEHRCausalTask` from an existing MTPP task.
- Used MTPP-selected task codes rather than all `CLMBRTokenizer.original_code_to_id` codes.
- Had ontology-parent target matching.
- Had separate logic for kept-event indexing.

Keep it only as legacy reference unless intentionally running the old standalone pipeline under `src/femr/models/survivehr/`.

### `src/femr/models/architecture/embedding_survivehr.py`

This is also legacy.

It subclasses the standard `embedding.FEMRModel` and swaps in `SurvivEHRCausalTaskHead` only for:

```text
loss_type == "survivehr_causal"
```

The current active path does not use it. Its useful behavior, adding `label_subject_ids` before calling the head, has been folded into standard `embedding.py` for `loss_type == "survivehr"`.

### `src/femr/models/survivehr/`

The whole `src/femr/models/survivehr/` directory is legacy for the latest implementation.

Files in that directory include:

- `survivehr_causal.py`
- `processor_survivehr.py`
- `prepare_survivehr_causal.py`
- `pretrain_survivehr_causal.py`
- `README.md`

Those files reference each other and can still describe or run the older standalone experiment style, but they are not imported by the current standard `survivehr` path.

### Removed Or Ignored Older Entrypoints

The old separate tutorial entrypoints are no longer the intended path:

- `src/femr/omop_meds_tutorial/prepare_survivehr_causal.py`
- `src/femr/omop_meds_tutorial/pretrain_survivehr_causal.py`
- older task file `src/femr/models/tasks/survivehr_causal.py`

Use standard `prepare.py` and `pretrain.py` instead.

## Tokenization And Labels

There are two separate ID spaces:

- Backbone input token IDs.
- SurvivEHR competing-risk label IDs.

Backbone input token IDs come from `CLMBRTokenizer.get_feature_codes(event)`.

Those IDs index `tokenizer.dictionary["vocab"]`, which contains ontology parent code tokens plus numeric/text property tokens. These are the IDs consumed by the embedding bag.

SurvivEHR label IDs come from `CLMBRTokenizer.get_event_code_token(event)`.

That method looks up `event.code` in `tokenizer.original_code_to_id`, a separate mapping over `tokenizer.dictionary["original_codes"]`. These IDs are raw event-code labels for next-code prediction, not embedding-vocabulary IDs.

Therefore `original_codes` are not simply the first part of the embedding vocabulary. The embedding vocabulary can include ontology ancestor codes and numeric/text tokens, while `original_code_to_id` is a separate raw-code label vocabulary.

## `SurvivEHRTask.add_event()` Behavior

`add_event(current_date, next_date, next_features, ...)` is called while processing event `t+1`, but it creates the label for the previous context event `t`.

It returns:

- `0` if no label should be attached to the previous context.
- `1` if one next-event label should be attached.

Main cases:

- `next_date is None` means the processor reached the last event, so no observed next event is available.
- `next_features is None` means `tokenizer.get_event_code_token(event)` could not map the next raw event code into `original_code_to_id`.
- `len(next_features) != 1` is invalid because SurvivEHR predicts one next raw event code.
- `next_feature < 0 or next_feature >= len(self.original_codes)` is a defensive guard against invalid label IDs.

In the normal active path, `next_feature` comes from:

```python
tokenizer.get_event_code_token(event)
```

So it should always be in:

```text
0 <= next_feature < len(self.original_codes)
```

If it is outside that range, the task cannot safely index `self.original_codes[next_feature]` and cannot create a valid shifted risk label `next_feature + 1`. Returning `0` skips that invalid label rather than crashing or training on a corrupted risk ID.

## Draft Prepare Commands

Transformer batches:

```bash
PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
python -m femr.omop_meds_tutorial.prepare \
  --pretraining_data /path/to/pretraining_data_survivehr \
  --ontology_path /path/to/pretraining_data_survivehr/ontology.pkl \
  --meds_reader /path/to/meds_reader \
  --athena_path "" \
  --num_threads 32 \
  --tokens_per_batch 8192 \
  --vocab_size 16384 \
  --value_count_threshold 1000 \
  --model transformer \
  --loss_type survivehr \
  --hf_cache_path /path/to/hf_cache
```

Mamba batches:

```bash
PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
python -m femr.omop_meds_tutorial.prepare \
  --pretraining_data /path/to/pretraining_data_survivehr \
  --ontology_path /path/to/pretraining_data_survivehr/ontology.pkl \
  --meds_reader /path/to/meds_reader \
  --athena_path "" \
  --num_threads 32 \
  --tokens_per_batch 8192 \
  --mamba_per_sample_max_tokens 8192 \
  --vocab_size 16384 \
  --value_count_threshold 1000 \
  --model mamba \
  --loss_type survivehr \
  --hf_cache_path /path/to/hf_cache
```

## Draft Pretrain Commands

Transformer:

```bash
PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
python -m femr.omop_meds_tutorial.pretrain \
  --pretraining_data /path/to/pretraining_data_survivehr \
  --output_dir /path/to/output_survivehr_transformer \
  --model transformer \
  --loss_type survivehr \
  --learning_rate 1e-5 \
  --n_epochs 20 \
  --n_layers 11 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --hf_run_name survivehr_transformer
```

Mamba:

```bash
PYTHONPATH=/user/zj2398/femr_chao_meds_v3/src \
python -m femr.omop_meds_tutorial.pretrain \
  --pretraining_data /path/to/pretraining_data_survivehr \
  --output_dir /path/to/output_survivehr_mamba \
  --model mamba \
  --loss_type survivehr \
  --learning_rate 1e-5 \
  --n_epochs 20 \
  --n_layers 11 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --hf_run_name survivehr_mamba
```
