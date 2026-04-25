"""SurvivEHR-aware variant of `FEMRBatchProcessor`.

This module is a thin layer on top of `femr.models.processor`. It exists so the
SurvivEHR causal task can plug into the MTPP batching pipeline without forcing
changes to the shared `processor.py`.

What it does differently from the vanilla processor
---------------------------------------------------
`SurvivEHRCausalTask` needs to align its per-subject target stream with the exact
order of events the `BatchCreator` keeps after per-day deduplication. In the default
path that alignment is achieved by having the task re-run the dedup filter inside
`SurvivEHRCausalTask._collect_kept_events` — that works, but it walks
`subject.events` and calls `tokenizer.get_feature_codes` twice per event.

`SurvivEHRBatchProcessor` replaces that with a single precomputation:

1. The processor iterates the subject's events once, applies the exact same dedup
   rules `BatchCreator.add_subject` uses, and builds:
   - `kept_events`         — the filtered event list
   - `kept_targets`        — `(label_index, numeric_value)` per kept event
   - `next_modeled_index`  — right-to-left pointer to the next modeled event
   - `final_time`          — censoring horizon
2. It hands those arrays to the task via `preload_subject_index(...)`.
3. It calls `BatchCreator.add_subject(...)` unchanged. Inside that call,
   `task.start_subject` sees the preloaded cache and skips the re-filter — every
   later `task.add_event` is then a plain O(1) lookup.

The vanilla `FEMRBatchProcessor` still works with `SurvivEHRCausalTask` unchanged,
because `start_subject` falls back to the task-local helpers when no cache is set.

Staying in sync with `BatchCreator`
-----------------------------------
The event-keeping loop here must match `BatchCreator.add_subject` in
`src/femr/models/processor.py`. If those rules change, update `iter_kept_events`
below in lockstep. The alignment assert inside `SurvivEHRCausalTask.add_event`
will catch any drift on the first affected subject.
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Tuple

import meds_reader

import femr.models.processor
import femr.models.tokenizer
import femr.ontology
import femr.pat_utils


def iter_kept_events(
    subject: meds_reader.Subject,
    tokenizer: femr.models.tokenizer.FEMRTokenizer,
) -> List[meds_reader.Event]:
    """Return the list of events `BatchCreator.add_subject` would keep for this subject.

    Must stay behaviorally identical to the event-keeping logic in
    `femr.models.processor.BatchCreator.add_subject` — same birth/person-table rule,
    same `get_feature_codes` emptiness check, same per-day "all features already seen"
    rule. See the module docstring for the alignment contract.
    """
    try:
        birth = femr.pat_utils.get_subject_birthdate(subject)
    except Exception:
        birth = subject.events[0].time if len(subject.events) > 0 else None

    kept: List[meds_reader.Event] = []
    current_date = None
    codes_seen_today: set = set()

    for event in subject.events:
        if event.time is None:
            continue
        if (
            birth is not None
            and event.time.date() <= birth.date()
            and getattr(event, "table", "person") == "person"
        ):
            continue

        if event.time.date() != current_date:
            current_date = event.time.date()
            codes_seen_today = set()

        features, _ = tokenizer.get_feature_codes(event)
        if len(features) == 0:
            continue
        if all(feature in codes_seen_today for feature in features):
            continue

        codes_seen_today |= set(features)
        kept.append(event)

    return kept


def _get_final_observation_time(subject: meds_reader.Subject) -> Optional[datetime.datetime]:
    # `meds_reader.SubjectEvents` is not reversible; iterate forward and keep the
    # last timed event. MEDS events are stored in chronological order.
    final_time: Optional[datetime.datetime] = None
    for event in subject.events:
        if event.time is not None:
            final_time = event.time
    return final_time


def build_survivehr_subject_index(
    subject: meds_reader.Subject,
    tokenizer: femr.models.tokenizer.FEMRTokenizer,
    task,  # SurvivEHRCausalTask; typed loosely to avoid circular imports.
    ontology: femr.ontology.Ontology,
) -> Tuple[
    List[meds_reader.Event],
    List[Tuple[int, Optional[float]]],
    List[Optional[int]],
    Optional[datetime.datetime],
]:
    """Run the per-subject precomputation the SurvivEHR task needs.

    Returns `(kept_events, kept_targets, next_modeled_index, final_time)`.
    """
    kept_events = iter_kept_events(subject, tokenizer)
    kept_targets = [task._event_to_modeled_target(event, ontology) for event in kept_events]
    next_modeled_index = task._build_next_modeled_event_index(kept_targets)
    final_time = _get_final_observation_time(subject)
    return kept_events, kept_targets, next_modeled_index, final_time


class SurvivEHRBatchCreator(femr.models.processor.BatchCreator):
    """`BatchCreator` variant that precomputes SurvivEHR per-subject indices.

    Overrides `add_subject` to build the kept-event / target / next-modeled-index
    arrays once, stash them on the task via `preload_subject_index`, and then
    delegate to the parent implementation unchanged.
    """

    def add_subject(
        self,
        subject: meds_reader.Subject,
        offset: int = 0,
        max_length: Optional[int] = None,
        subsample_task_fraction: float = 1,
        actually_add: bool = True,
    ):
        if self.task is not None and self.tokenizer.ontology is not None:
            kept_events, kept_targets, next_modeled_index, final_time = (
                build_survivehr_subject_index(subject, self.tokenizer, self.task, self.tokenizer.ontology)
            )
            # The task consumes this cache inside its own `start_subject`, which runs
            # later inside the parent `add_subject` call.
            self.task.preload_subject_index(
                subject,
                kept_events,
                kept_targets,
                next_modeled_index,
                final_time,
            )

        return super().add_subject(
            subject,
            offset=offset,
            max_length=max_length,
            subsample_task_fraction=subsample_task_fraction,
            actually_add=actually_add,
        )


class SurvivEHRBatchProcessor(femr.models.processor.FEMRBatchProcessor):
    """`FEMRBatchProcessor` that uses `SurvivEHRBatchCreator` under the hood.

    The public interface matches `FEMRBatchProcessor` one-to-one — all of
    `convert_subject`, `convert_dataset`, `convert_dataset_mamba`, and `collate`
    route through the inherited implementations.
    """

    def __init__(self, tokenizer, task=None, cache_path=None):
        self.creator = SurvivEHRBatchCreator(tokenizer, task)
        self.cache_path = cache_path
