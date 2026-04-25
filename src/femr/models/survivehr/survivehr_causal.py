from __future__ import annotations

import collections
import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import meds_reader
import numpy as np

import femr.models.config
import femr.ontology
from femr.models.tasks.motor import Task


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SurvivEHRCausalTask(Task):
    """Build SurvivEHR-style next-event labels on top of the FEMR tokenizer pipeline.

    The backbone input stays identical to the existing hierarchical FEMR setup.
    The task produces one label per valid context event:
    - `event_label == 0`: censored because no later modeled code occurs before end of record.
    - `event_label in [1, K]`: the next future modeled code among the K selected MTPP codes.
    - `event_value`: numeric value for modeled numerical codes, otherwise NaN.

    Alignment contract with the processor
    -------------------------------------
    The task exposes a position-indexed target stream: `add_event(current_date, next_date, ...)`
    is called once per event the processor keeps, and the i-th call must return the label for
    the processor's i-th kept event. To honor that contract, the task precomputes the exact
    same per-day-deduplicated event stream that `BatchCreator.add_subject` emits. The helpers
    below implement that precomputation:

    - `_collect_kept_events`      mirrors the processor's dedup filter (kept events list).
    - `_event_to_modeled_target`  maps each kept event to (label_index, numeric_value).
    - `_match_modeled_code`       ontology BFS for the nearest modeled ancestor, cached.
    - `_build_next_modeled_event_index`  per-position pointer to the next kept event whose
                                         target is a modeled code (used for "next future
                                         modeled event" lookup in `add_event`).

    When the task is driven by `SurvivEHRBatchProcessor` (see `processor_survivehr.py`), the
    four helpers are called once by the processor and their outputs are handed in via
    `preload_subject_index(...)` — in that path `start_subject` skips the recomputation.
    """

    @classmethod
    def from_mtpp_task(
        cls,
        mtpp_task,
        *,
        surv_weight: float = 1.0,
        value_weight: float = 0.1,
        survival_hidden_dim: int = 32,
        value_hidden_dim: int = 32,
        quadrature_n: int = 15,
        time_scale_days: float = 365.25,
        concurrent_strategy: Optional[str] = None,
    ) -> "SurvivEHRCausalTask":
        return cls(
            pretraining_task_info=list(mtpp_task.pretraining_task_info),
            numerical_task_codes=list(mtpp_task.numerical_task_to_index_map.keys()),
            surv_weight=surv_weight,
            value_weight=value_weight,
            survival_hidden_dim=survival_hidden_dim,
            value_hidden_dim=value_hidden_dim,
            quadrature_n=quadrature_n,
            time_scale_days=time_scale_days,
            concurrent_strategy=concurrent_strategy,
        )

    def __init__(
        self,
        *,
        pretraining_task_info: List[Tuple[str, float]],
        numerical_task_codes: List[str],
        surv_weight: float = 1.0,
        value_weight: float = 0.1,
        survival_hidden_dim: int = 32,
        value_hidden_dim: int = 32,
        quadrature_n: int = 15,
        time_scale_days: float = 365.25,
        concurrent_strategy: Optional[str] = None,
    ):
        super().__init__()

        self.pretraining_task_info = list(pretraining_task_info)
        self.task_codes = [item[0] if isinstance(item, (tuple, list)) else item for item in self.pretraining_task_info]
        self.numerical_task_codes = list(numerical_task_codes)
        self.numerical_task_code_set: Set[str] = set(self.numerical_task_codes)
        self.non_numerical_task_code_set: Set[str] = set(self.task_codes) - self.numerical_task_code_set
        self.code_to_index: Dict[str, int] = {code: idx for idx, code in enumerate(self.task_codes)}

        self.surv_weight = float(surv_weight)
        self.value_weight = float(value_weight)
        self.survival_hidden_dim = int(survival_hidden_dim)
        self.value_hidden_dim = int(value_hidden_dim)
        self.quadrature_n = int(quadrature_n)
        self.time_scale_days = float(time_scale_days)
        self.concurrent_strategy = concurrent_strategy

        self.tokenizer = None
        self._ancestor_match_cache: Dict[Tuple[str, bool], Optional[str]] = {}
        # Optional cache filled by SurvivEHRBatchProcessor right before start_subject runs.
        # Tuple shape: (subject_id, kept_events, kept_targets, next_modeled_index, final_time).
        self._precomputed_subject_index: Optional[Tuple] = None

    def bind_tokenizer(self, tokenizer) -> "SurvivEHRCausalTask":
        self.tokenizer = tokenizer
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        for transient_key in (
            "_current_subject_id",
            "_current_event_cursor",
            "_kept_events",
            "_kept_event_targets",
            "_next_modeled_event_index",
            "_final_observation_time",
            "_precomputed_subject_index",
            "per_subject_event_time",
            "per_subject_event_label",
            "per_subject_event_value",
            "batch_event_time",
            "batch_event_label",
            "batch_event_value",
        ):
            state.pop(transient_key, None)
        return state

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="survivehr_causal",
            task_kwargs=dict(
                pretraining_task_info=self.pretraining_task_info,
                numerical_task_codes=self.numerical_task_codes,
                surv_weight=self.surv_weight,
                value_weight=self.value_weight,
                survival_hidden_dim=self.survival_hidden_dim,
                value_hidden_dim=self.value_hidden_dim,
                quadrature_n=self.quadrature_n,
                time_scale_days=self.time_scale_days,
                concurrent_strategy=self.concurrent_strategy,
            ),
        )

    def start_subject(self, subject: meds_reader.Subject, ontology: Optional[femr.ontology.Ontology]) -> None:
        assert self.tokenizer is not None, "Tokenizer must be bound before building batches."
        assert ontology is not None, "Ontology must be available before building batches."

        self._current_subject_id = subject.subject_id
        self._current_event_cursor = 0

        # Fast path: SurvivEHRBatchProcessor has already computed the per-subject indices.
        # Consume them and skip the re-filtering below.
        precomputed = self._precomputed_subject_index
        if precomputed is not None and precomputed[0] == subject.subject_id:
            _, kept_events, kept_targets, next_modeled_index, final_time = precomputed
            self._kept_events = kept_events
            self._kept_event_targets = kept_targets
            self._next_modeled_event_index = next_modeled_index
            self._final_observation_time = final_time
            self._precomputed_subject_index = None
        else:
            # Fallback path: vanilla FEMRBatchProcessor. Rebuild the per-subject indices here.
            self._kept_events = self._collect_kept_events(subject)
            self._kept_event_targets = [self._event_to_modeled_target(event, ontology) for event in self._kept_events]
            self._next_modeled_event_index = self._build_next_modeled_event_index(self._kept_event_targets)
            self._final_observation_time = self._get_final_observation_time(subject)

        self.per_subject_event_time: List[float] = []
        self.per_subject_event_label: List[int] = []
        self.per_subject_event_value: List[float] = []

    def preload_subject_index(
        self,
        subject: meds_reader.Subject,
        kept_events: List[meds_reader.Event],
        kept_targets: List[Tuple[int, Optional[float]]],
        next_modeled_index: List[Optional[int]],
        final_time: Optional[datetime.datetime],
    ) -> None:
        """Stash per-subject indices so the next `start_subject` call can skip re-filtering.

        Used by `SurvivEHRBatchProcessor` to hand pre-computed arrays to the task without
        touching `BatchCreator` or `FEMRBatchProcessor`. The cache is consumed exactly once,
        on the very next `start_subject(subject, ...)` call that matches the subject id.
        """
        self._precomputed_subject_index = (
            subject.subject_id,
            kept_events,
            kept_targets,
            next_modeled_index,
            final_time,
        )

    def needs_exact(self) -> bool:
        # SurvivEHR next-event prediction is cheap per context, so every eligible label
        # must be covered. This opts out of MTPP's sampled-label regime. See plans.md
        # ("Clarified Decisions" §1) and survivehr_revision_notes.md (§3).
        return True

    def get_sampled_labels(self, length: int) -> int:
        # With needs_exact() == True this is effectively unused by the processor, but
        # the Task interface still requires it. Returning `length` means "keep them all".
        return length

    def start_batch(self) -> None:
        self.batch_event_time: List[float] = []
        self.batch_event_label: List[int] = []
        self.batch_event_value: List[float] = []

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        self.batch_event_time.extend([self.per_subject_event_time[i] for i in subject_label_offsets])
        self.batch_event_label.extend([self.per_subject_event_label[i] for i in subject_label_offsets])
        self.batch_event_value.extend([self.per_subject_event_value[i] for i in subject_label_offsets])

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
        actually_add: bool = True,
    ) -> int:
        # Return early if the processor is still emitting events past our kept range
        # (e.g. the final-step call from BatchCreator after all kept events are consumed).
        if self._current_event_cursor >= len(self._kept_events):
            return 0

        current_index = self._current_event_cursor
        current_event = self._kept_events[current_index]
        self._current_event_cursor += 1

        # Alignment check: the processor and the task must agree on "event i". Any mismatch
        # means the dedup filter has diverged between the two — treat it as a hard bug, not
        # a recoverable warning, because every later label would be misaligned.
        if current_event.time != current_date:
            raise AssertionError(
                "Task/event alignment mismatch for "
                f"subject_id={self._current_subject_id}, cursor={current_index}: "
                f"expected current_date={current_event.time}, received {current_date}"
            )

        if next_date is not None and current_index + 1 < len(self._kept_events):
            immediate_next_event = self._kept_events[current_index + 1]
            if immediate_next_event.time != next_date:
                raise AssertionError(
                    "Task/event alignment mismatch for "
                    f"subject_id={self._current_subject_id}, cursor={current_index}: "
                    f"expected immediate next_date={immediate_next_event.time}, received {next_date}"
                )

        # Look up the next future kept event whose target is a modeled code.
        target_index = self._next_modeled_event_index[current_index]
        if target_index is not None:
            target_event = self._kept_events[target_index]
            target_label, target_value = self._kept_event_targets[target_index]
            delta_days = (target_event.time - current_date).total_seconds() / 86400.0
        else:
            # Censoring branch: no later modeled event exists. Use the subject's last observed
            # event time as the censoring horizon (see survivehr_revision_notes.md §2).
            # A non-positive horizon means we are already at/past the end — no label emitted.
            if self._final_observation_time is None or self._final_observation_time <= current_date:
                return 0
            target_label = 0
            target_value = None
            delta_days = (self._final_observation_time - current_date).total_seconds() / 86400.0

        if not actually_add:
            return 1

        self.per_subject_event_time.append(delta_days)
        self.per_subject_event_label.append(int(target_label))
        self.per_subject_event_value.append(float(target_value) if target_value is not None else float("nan"))
        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        event_value = np.array(self.batch_event_value, dtype=np.float32)
        return {
            "event_time": np.array(self.batch_event_time, dtype=np.float32),
            "event_label": np.array(self.batch_event_label, dtype=np.int32),
            "event_value": event_value,
            "value_mask": np.isfinite(event_value),
        }

    def cleanup(self, batch):
        return batch

    def _collect_kept_events(self, subject: meds_reader.Subject):
        """Replay the processor's per-day dedup filter and return the list of kept events.

        This must stay behaviorally identical to the event-keeping logic inside
        `femr.models.processor.BatchCreator.add_subject` — same birth/person-table handling,
        same `get_feature_codes` emptiness check, same "all features already seen today" rule.
        If the two ever diverge, `add_event` will raise an alignment AssertionError on the
        first affected subject.

        For the SurvivEHR processor path this function is not called — the processor computes
        the kept-event list once via `iter_kept_events` in `processor_survivehr.py` and hands
        it to the task via `preload_subject_index`. This method stays here as the fallback
        used when the task is driven by the vanilla `FEMRBatchProcessor`.
        """
        birth = None
        try:
            import femr.pat_utils

            birth = femr.pat_utils.get_subject_birthdate(subject)
        except Exception:
            birth = subject.events[0].time if len(subject.events) > 0 else None

        kept_events = []
        current_date = None
        codes_seen_today: Set[int] = set()

        for event in subject.events:
            if event.time is None:
                continue
            # Drop birth-day events that come from the "person" table (mirrors BatchCreator).
            if birth is not None and event.time.date() <= birth.date() and getattr(event, "table", "person") == "person":
                continue

            # Reset the "already seen" set on every new calendar day.
            if event.time.date() != current_date:
                current_date = event.time.date()
                codes_seen_today = set()

            features, _ = self.tokenizer.get_feature_codes(event)
            if len(features) == 0:
                continue
            # Drop events whose every feature already fired earlier today.
            if all(feature in codes_seen_today for feature in features):
                continue

            codes_seen_today |= set(features)
            kept_events.append(event)

        return kept_events

    def _event_to_modeled_target(
        self,
        event: meds_reader.Event,
        ontology: femr.ontology.Ontology,
    ) -> Tuple[int, Optional[float]]:
        """Map one kept event to `(label_index, numeric_value)`.

        Return values
        -------------
        - `label_index == 0`: event does not land on any modeled code. It will not serve as a
          target; the caller advances to the next future kept event instead.
        - `label_index in [1, K]`: 1-based index into the modeled task codes (the `+ 1` leaves
          slot 0 free for the censoring label).
        - `numeric_value`: the raw scalar for numerical codes, or `None` for non-numerical
          codes. `None` is rendered as NaN in the batch tensor (see `get_batch_data`).

        Numerical and non-numerical paths are kept strictly separate: an event with a numeric
        value only competes against the numerical task-code set, and an event with no values
        only competes against the non-numerical set. Events that are text-valued but not
        numeric are treated as out-of-set and return `(0, None)`.
        """
        is_numerical = event.text_value is None and event.numeric_value is not None

        if is_numerical:
            numeric_value = _safe_float(event.numeric_value)
            matched_code = self._match_modeled_code(event.code, self.numerical_task_code_set, ontology, is_numerical=True)
            if matched_code is not None:
                return self.code_to_index[matched_code] + 1, numeric_value

        if event.numeric_value is None and event.text_value is None:
            matched_code = self._match_modeled_code(event.code, self.non_numerical_task_code_set, ontology, is_numerical=False)
            if matched_code is not None:
                return self.code_to_index[matched_code] + 1, None

        return 0, None

    def _match_modeled_code(
        self,
        raw_code: str,
        allowed_codes: Set[str],
        ontology: femr.ontology.Ontology,
        *,
        is_numerical: bool,
    ) -> Optional[str]:
        """Resolve a raw event code to an allowed modeled code via ontology ancestors.

        Direct hits return immediately. Otherwise the function runs a breadth-first walk up
        `ontology.parents_map` and picks the nearest ancestor that is in `allowed_codes`.
        If several ancestors sit at the same distance, the one with the smaller position in
        `code_to_index` wins — this gives deterministic, stable tie breaking across runs.

        Results are cached per `(raw_code, is_numerical)`. The `is_numerical` flag matters
        because the numerical and non-numerical universes differ.

        Returns `None` if no allowed ancestor is reachable.
        """
        if raw_code in allowed_codes:
            return raw_code

        cache_key = (raw_code, is_numerical)
        cached = self._ancestor_match_cache.get(cache_key)
        if cache_key in self._ancestor_match_cache:
            return cached

        queue = collections.deque([(raw_code, 0)])
        visited = {raw_code}
        best_match: Optional[str] = None
        best_distance: Optional[int] = None

        while queue:
            code, distance = queue.popleft()
            next_distance = distance + 1
            for parent in ontology.parents_map.get(code, set()):
                if parent in visited:
                    continue
                visited.add(parent)
                if parent in allowed_codes:
                    # Keep the nearest ancestor; break ties by task-order position.
                    if best_match is None or next_distance < best_distance or (
                        next_distance == best_distance and self.code_to_index[parent] < self.code_to_index[best_match]
                    ):
                        best_match = parent
                        best_distance = next_distance
                queue.append((parent, next_distance))

        self._ancestor_match_cache[cache_key] = best_match
        return best_match

    def _build_next_modeled_event_index(
        self, targets: Sequence[Tuple[int, Optional[float]]]
    ) -> List[Optional[int]]:
        """Right-to-left scan over `targets` producing a "next modeled event" pointer array.

        For each kept-event position `i`, `next_modeled_index[i]` is the smallest index
        `j > i` whose target has `label > 0` (i.e. the event maps to a modeled code), or
        `None` if no such later event exists. `add_event` uses this array to answer
        "where is the next future modeled event?" in O(1), including the case where the
        immediate next kept event is itself out-of-set.
        """
        next_modeled_index: List[Optional[int]] = [None] * len(targets)
        future_index: Optional[int] = None
        for index in range(len(targets) - 1, -1, -1):
            next_modeled_index[index] = future_index
            if targets[index][0] > 0:
                future_index = index
        return next_modeled_index

    def _get_final_observation_time(self, subject: meds_reader.Subject) -> Optional[datetime.datetime]:
        """Return the timestamp of the subject's last timed event; `None` if none exist.

        Used as the censoring horizon in `add_event` when no later modeled event is
        available. See survivehr_revision_notes.md §2 ("Censoring Horizon").
        """
        # `meds_reader.SubjectEvents` is not reversible; iterate forward and keep
        # the last timed event. MEDS events are stored in chronological order.
        final_time: Optional[datetime.datetime] = None
        for event in subject.events:
            if event.time is not None:
                final_time = event.time
        return final_time
