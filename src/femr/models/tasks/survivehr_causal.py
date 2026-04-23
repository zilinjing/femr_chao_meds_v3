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
        self._kept_events = self._collect_kept_events(subject)
        self._kept_event_targets = [self._event_to_modeled_target(event, ontology) for event in self._kept_events]
        self._next_modeled_event_index = self._build_next_modeled_event_index(self._kept_event_targets)
        self._final_observation_time = self._get_final_observation_time(subject)

        self.per_subject_event_time: List[float] = []
        self.per_subject_event_label: List[int] = []
        self.per_subject_event_value: List[float] = []

    def needs_exact(self) -> bool:
        return True

    def get_sampled_labels(self, length: int) -> int:
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
        if self._current_event_cursor >= len(self._kept_events):
            return 0

        current_index = self._current_event_cursor
        current_event = self._kept_events[current_index]
        self._current_event_cursor += 1

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

        target_index = self._next_modeled_event_index[current_index]
        if target_index is not None:
            target_event = self._kept_events[target_index]
            target_label, target_value = self._kept_event_targets[target_index]
            delta_days = (target_event.time - current_date).total_seconds() / 86400.0
        else:
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
            if birth is not None and event.time.date() <= birth.date() and getattr(event, "table", "person") == "person":
                continue

            if event.time.date() != current_date:
                current_date = event.time.date()
                codes_seen_today = set()

            features, _ = self.tokenizer.get_feature_codes(event)
            if len(features) == 0:
                continue
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
        next_modeled_index: List[Optional[int]] = [None] * len(targets)
        future_index: Optional[int] = None
        for index in range(len(targets) - 1, -1, -1):
            next_modeled_index[index] = future_index
            if targets[index][0] > 0:
                future_index = index
        return next_modeled_index

    def _get_final_observation_time(self, subject: meds_reader.Subject) -> Optional[datetime.datetime]:
        for event in reversed(subject.events):
            if event.time is not None:
                return event.time
        return None
