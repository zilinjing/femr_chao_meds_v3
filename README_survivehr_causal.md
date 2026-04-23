# SurvivEHR-Style Causal Pretraining on FEMR Backbones

This note documents the additional code added to `femr_chao_meds_v3` to compare a SurvivEHR-style loss against the existing FEMR pretraining objectives while keeping the tokenizer, database interface, and backbone architecture as close as possible to the current FEMR stack.

## Goal

The implementation keeps:

- the same MEDS/FEMR database interface,
- the same hierarchical tokenizer and `EmbeddingBag` input representation,
- the same transformer or mamba backbone wrappers,
- the same downstream feature extraction path for linear probing.

It changes only the pretraining task/head so that pretraining uses:

- immediate next-event prediction,
- a continuous-time competing-risk loss,
- a Gaussian value loss for numerical next events.

The new code lives in:

- `src/femr/models/tasks/survivehr_causal.py`
- `src/femr/models/architecture/survivehr_head.py`
- `src/femr/models/architecture/embedding_survivehr.py`
- `src/femr/omop_meds_tutorial/prepare_survivehr_causal.py`
- `src/femr/omop_meds_tutorial/pretrain_survivehr_causal.py`

## Preprocessing Overview

The preprocessing step assumes you already have:

- `ontology.pkl`
- `tokenizer/`
- an MTPP task pickle such as `mtpp_task.pkl`

The MTPP task is used only to define the modeled code set. In particular:

- `pretraining_task_info` defines the global task/code order,
- `numerical_task_to_index_map.keys()` defines which modeled codes receive Gaussian value heads,
- `task_to_index_map.keys()` defines the modeled non-numerical codes.

The preprocessing script:

- loads the existing FEMR tokenizer,
- loads the existing MTPP task,
- creates a `SurvivEHRCausalTask` from it,
- builds train and validation batches with the standard `FEMRBatchProcessor`,
- keeps exact label coverage for the causal task rather than MTPP-style sampled label times,
- saves new batch datasets under:
  - `train_batches_survivehr_causal`
  - `val_batches_survivehr_causal`
  - or the `_mamba` variants for mamba packing.

Run:

```bash
PYTHONPATH=/path/to/femr_chao_meds_v3/src python -m femr.omop_meds_tutorial.prepare_survivehr_causal \
  --pretraining_data /path/to/pretraining_data \
  --meds_reader /path/to/meds_reader \
  --mtpp_task_path /path/to/mtpp_task.pkl \
  --model transformer \
  --tokens_per_batch 8192 \
  --num_threads 16
```

## SurvivEHRCausalTask

`SurvivEHRCausalTask` converts the existing FEMR event stream into SurvivEHR-style next-event labels.

### What counts as one input step

The input step is still the standard FEMR time step used by `BatchCreator`:

- one chronological event,
- represented by a bag of ontology parent codes and optional discretized property/value tokens,
- deduplicated within a day in the same way as the existing FEMR processor.

So the backbone input is unchanged.

### What counts as one target

For each kept context event:

- input/context is the hidden state at event `j`,
- target time is the interval to the next future kept event that is inside the selected modeled code set,
- target label is one modeled code for that next valid modeled event,
- target value is the raw numeric value if that target event is a modeled numerical code.

The task emits:

- `event_time`
- `event_label`
- `event_value`
- `value_mask`

### Mapping the next event to a single competing-risk label

The implementation uses direct raw-code matching first, with an ontology fallback against the selected MTPP task-code universe:

- direct raw code for numerical events, if the raw code is in the MTPP numerical code set,
- otherwise the nearest numerical ancestor that is in the selected numerical code set,
- direct raw code for non-numerical events, if the raw code is in the selected non-numerical task-code set,
- otherwise the nearest non-numerical ancestor that is in the selected non-numerical task-code set.

If multiple modeled ancestors exist at the same ontology distance, the task uses task-order tie breaking. If no modeled raw code or modeled ancestor exists, that event is skipped when searching for the next target.

### Important consequence

This differs from both the original full-vocabulary SurvivEHR setup and the original MTPP objective:

- the loss is still next-event style, not many-task first-future-occurrence style,
- but the modeled label universe is restricted to the MTPP task codes,
- so the target is the next future kept event inside that restricted code universe.

Censoring means:

- no later modeled event exists before end of record.

## Model Architecture

The new model wrapper in `embedding_survivehr.py` reuses the current FEMR backbones unchanged:

- `FEMRTransformer` for transformer mode,
- `FEMRMamba` for mamba mode.

This means the following remain the same as the current FEMR implementation:

- tokenizer and `EmbeddingBag` logic,
- rotary age embedding / FEMR time features,
- transformer or mamba internals,
- hidden-state extraction for downstream linear probing.

Only the pretraining head changes.

## SurvivEHR-Style Head

The head in `survivehr_head.py` has two parts:

- `ODESurvMultiple`: competing-risk continuous-time head,
- `GaussianValueHeads`: one Gaussian head per numerical modeled code.

### Competing-risk part

For hidden state `h`, time `t`, and modeled next-event label `k`:

- `pi(h)` is a softmax over all modeled codes,
- the ODE network defines a per-risk cumulative term
  - `u_r(t|h) = 1 - exp(-∫ f_r(s,h) ds)`
- the per-risk cumulative incidence is
  - `CIF_r(t|h) = pi_r(h) * u_r(t|h)`

The event log-likelihood for observed next event `k` is:

- `log pi_k(h) + log f_k(t,h) + log(1 - u_k(t|h))`

The censoring term for label `0` is:

- `log(1 - sum_r CIF_r(t|h))`

This mirrors the original SurvivEHR DeSurv-style competing-risk objective.

Time is passed from the FEMR task in days and normalized inside the head by dividing by `time_scale_days` with default `365.25`, so training is effectively done in years.

### Gaussian value part

For numerical next-event labels only, the head predicts:

- `value | h, k ~ Normal(mu_k(h), sigma_k(h))`

and adds the Gaussian negative log-likelihood on the raw numeric value.

The value loss follows the original SurvivEHR reduction closely:

- one Gaussian head per numerical code,
- average over occurrences of that code within each patient,
- average across patients in the batch,
- sum across numerical code types.

If the correct next token is numerical but the numeric value is missing, the implementation keeps the default `Normal(0,1)` contribution for that token position, matching the original baseline logic as closely as practical.

### Combined loss

The final loss is:

`L = lambda_surv * L_surv + lambda_val * L_val`

with `lambda_surv` and `lambda_val` normalized from the provided weights.

## Pretraining Script

`pretrain_survivehr_causal.py`:

- loads the existing tokenizer and ontology,
- loads the prepared `SurvivEHRCausalTask`,
- loads the prepared batch datasets,
- builds a standard FEMR model config from the existing Hydra model config,
- swaps in the SurvivEHR-style head via `embedding_survivehr.FEMRModel`,
- trains with Hugging Face `Trainer`,
- saves the final model and tokenizer under `output_dir/final_model`.

Run:

```bash
PYTHONPATH=/path/to/femr_chao_meds_v3/src python -m femr.omop_meds_tutorial.pretrain_survivehr_causal \
  --pretraining_data /path/to/pretraining_data \
  --output_dir /path/to/output_dir \
  --model transformer \
  --learning_rate 1e-5 \
  --n_epochs 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 16 \
  --hf_run_name survivehr_causal_transformer \
  --bf16
```

For mamba, change `--model mamba`.

## Downstream Linear Probing

The backbone remains compatible with the existing downstream extraction flow.

Use the existing `femr.models.architecture.embedding.compute_features(...)` and point `model_path` to:

- `output_dir/final_model`

When extracting features:

- use `loss_type="labeled_subjects"`,
- the loader drops the pretraining head weights and loads only the backbone weights,
- downstream linear probing therefore uses the same feature path as the current FEMR code.

## Comparison Against MTPP

This implementation differs from MTPP in the objective:

- MTPP predicts the first future occurrence for every selected code in parallel.
- SurvivEHR causal predicts one next future modeled event in the selected code universe.
- MTPP uses discretized time bins and discretized value bins.
- SurvivEHR causal uses continuous time and Gaussian raw-value prediction.
- MTPP uses sampled label positions in its non-exact batching path.
- SurvivEHR causal uses exact label coverage and does not inherit that sampling regime.

The backbone and tokenizer are intentionally shared so the loss/objective is the primary changed factor.

## Consistency With Original SurvivEHR

The implementation is consistent in the core survival/value factorization:

- competing-risk next-event prediction,
- continuous-time DeSurv-style likelihood,
- Gaussian value loss for numerical next events,
- weighted combination of survival and value losses.

It is not identical to the original repository because:

- the input representation is FEMR hierarchical `EmbeddingBag`, not the original SurvivEHR event embedding stack,
- the modeled risk set is the selected MTPP code set, not the full token vocabulary,
- the target is the next future modeled event inside the selected code set rather than the next token in the full vocabulary,
- time scaling is explicit in the head via `time_scale_days`,
- batching and preprocessing follow FEMR’s processor/tokenizer stack rather than the original SurvivEHR datamodule.

So this should be viewed as a controlled FEMR-backbone implementation of the SurvivEHR-style objective, not a byte-for-byte port of the original codebase.
