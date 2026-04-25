# SurvivEHR on FEMR

A SurvivEHR-style next-event pretraining baseline that plugs into the existing FEMR
tokenizer, backbone, and batching pipeline, designed for a fair comparison against
the MTPP baseline defined in `src/femr/models/tasks/mtpp.py`.

## File map

| File | Role |
|---|---|
| `survivehr_causal.py` | The `Task` implementation (`SurvivEHRCausalTask`). Produces one label per valid context event. |
| `processor_survivehr.py` | `SurvivEHRBatchProcessor` / `SurvivEHRBatchCreator`. Same public interface as the vanilla FEMR processor, but precomputes per-subject indices once and hands them to the task. |
| `prepare_survivehr_causal.py` | CLI entry point that builds and caches training / validation batches. |
| `pretrain_survivehr_causal.py` | CLI entry point that trains the transformer or Mamba backbone with the SurvivEHR head. |

## Label semantics

For every kept context event the task emits a single label:

- `event_label == 0` — censored. No later modeled event exists before the end of
  the record. The delta is the distance from the current event to the subject's
  last observed event time.
- `event_label in [1, K]` — the next future modeled code, expressed as a 1-based
  index into the K selected MTPP task codes. Slot 0 is reserved for the censoring
  label.
- `event_value` — the raw numeric value for modeled numerical codes, or NaN
  otherwise. `value_mask` is simply `np.isfinite(event_value)`.

The clarified design decisions behind this scheme live in the repo root:
`plans.md` ("Clarified Decisions") and `survivehr_revision_notes.md`. The short
version:

- Non-numerical codes do not use ontology-parent collapsing as their primary path.
  Raw-code matching is tried first; an ontology-ancestor fallback is used only
  when the raw code is not in the task universe.
- Numerical codes use direct raw-code matching with the raw scalar value, with
  the same ancestor fallback.
- Out-of-set intermediate events are skipped when searching for the target; the
  target is the next kept event inside the selected task universe.
- Censoring fires only when no later modeled event exists.
- `needs_exact()` is `True`. SurvivEHR does not inherit MTPP's sampled-label
  regime.

## Why the task has those four helpers

`SurvivEHRCausalTask` contains four helpers that, at a glance, appear to duplicate
work already done by `BatchCreator.add_subject` in `src/femr/models/processor.py`.
They are there on purpose:

| Helper | Why it exists | Why MTPP does not need it |
|---|---|---|
| `_collect_kept_events` | Replays the processor's per-day dedup filter so the task's position-indexed target stream lines up with the processor's event stream. The alignment is enforced by the assert in `add_event`. | MTPP's `MTPP_SurvivalCalculator` is time-indexed — it looks up futures by wall-clock time, so no positional alignment is needed. |
| `_event_to_modeled_target` | Maps one event to `(label_index, numeric_value)`. SurvivEHR emits a single target per context position, so this is the per-event decision point. | MTPP populates multi-code sparse structures, so it does not collapse to a single label. |
| `_match_modeled_code` | BFS up `ontology.parents_map` to recover coverage for codes that only appear in the task universe via ancestors. Cached per `(raw_code, is_numerical)`. | MTPP bakes ancestor expansion into `MTPP_SurvivalCalculator` at construction time. |
| `_build_next_modeled_event_index` | Right-to-left scan producing, for each kept position, the next future position whose target is a modeled event. Lets `add_event` answer "next modeled event?" in O(1). | MTPP emits per-code survival labels for every context time, so there is no "single next modeled event" concept. |

The first helper in particular looks redundant with the processor because it
re-applies the same dedup rules. That duplication is the point: the task needs to
know the kept positions **before** the processor starts iterating so the cursor
can advance in lockstep.

## Two ways to drive the task

### Default path — vanilla processor

```python
from femr.models.processor import FEMRBatchProcessor
from femr.models.survivehr.survivehr_causal import SurvivEHRCausalTask

task = SurvivEHRCausalTask.from_mtpp_task(mtpp_task).bind_tokenizer(tokenizer)
processor = FEMRBatchProcessor(tokenizer, task)
batches = processor.convert_dataset(db, tokens_per_batch=8192, num_proc=32)
```

In this path `task.start_subject` rebuilds the kept-event list every subject. It
works, but it walks `subject.events` and calls `tokenizer.get_feature_codes` once
in the task and once in the processor.

### Recommended path — SurvivEHR processor

```python
from femr.models.survivehr.processor_survivehr import SurvivEHRBatchProcessor
from femr.models.survivehr.survivehr_causal import SurvivEHRCausalTask

task = SurvivEHRCausalTask.from_mtpp_task(mtpp_task).bind_tokenizer(tokenizer)
processor = SurvivEHRBatchProcessor(tokenizer, task)
batches = processor.convert_dataset(db, tokens_per_batch=8192, num_proc=32)
```

Under this path the processor computes the kept-event list, per-event targets,
and the "next modeled event" pointer array **once** and stashes them on the task
via `preload_subject_index`. The subsequent `task.start_subject` consumes the
cache and skips the recomputation; every `task.add_event` is then an O(1) lookup.
Output is bitwise identical to the default path.

`src/femr/models/processor.py` is untouched.

## Known follow-ups

From `survivehr_revision_notes.md`:

- **Ancestor tie breaking.** The current implementation picks the nearest allowed
  ancestor and breaks ties by `code_to_index` position. This is deterministic and
  simple, but it is an implementation choice, not something the original
  SurvivEHR codebase specifies.
- **Censoring horizon.** The task uses the subject's last observed event time as
  the censoring horizon. This is an end-of-record adaptation of the original
  full-vocabulary next-token setting.
- **Time normalization placement.** Training-time year conversion currently lives
  in the head (`time_scale_days=365.25`) instead of in the task's emitted targets.
