from __future__ import annotations

import datetime
import functools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import meds_reader
import numpy as np

import femr.models.config
import femr.models.tokenizer
import femr.ontology
import femr.stat_utils
from femr.models.tasks.ntp import Task


def _try_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _value_stats_map(
    subjects: Iterable[meds_reader.Subject], *, original_codes: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    original_code_set = set(original_codes)
    stats: Dict[str, Dict[str, Any]] = {}

    for subject in subjects:
        for event in subject.events:
            if event.code not in original_code_set:
                continue
            if event.numeric_value is None or event.text_value is not None:
                continue

            value = _try_float(event.numeric_value)
            if value is None:
                continue

            if event.code not in stats:
                stats[event.code] = {
                    "count": 0,
                    "samples": femr.stat_utils.ReservoirSampler(100_000),
                }
            stats[event.code]["count"] += 1
            stats[event.code]["samples"].add(value, 1)

    return stats


def _value_stats_agg(first: Dict[str, Dict[str, Any]], second: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    for code, second_stats in second.items():
        if code not in first:
            first[code] = {
                "count": 0,
                "samples": femr.stat_utils.ReservoirSampler(100_000),
            }

        first[code]["count"] += second_stats["count"]
        if len(second_stats["samples"].samples) > 0:
            first[code]["samples"].combine(second_stats["samples"])

    return first


class SurvivEHRTask(Task):
    @classmethod
    def fit_pretraining_task_info(
        cls,
        db: meds_reader.SubjectDatabase,
        tokenizer: femr.models.tokenizer.CLMBRTokenizer,
        value_count_threshold: int = 1000,
        time_scale_days: float = 365.25,
        concurrent_strategy: str = "add_noise",
    ) -> "SurvivEHRTask":
        original_codes = [
            code for code, _ in sorted(tokenizer.original_code_to_id.items(), key=lambda item: item[1])
        ]

        value_stats = functools.reduce(
            _value_stats_agg,
            db.map(functools.partial(_value_stats_map, original_codes=original_codes)),
            {},
        )

        numerical_task_codes: List[str] = []
        value_normalization: Dict[str, Tuple[float, float]] = {}
        for code in original_codes:
            stats = value_stats.get(code)
            if stats is None or stats["count"] < value_count_threshold:
                continue

            samples = np.asarray(stats["samples"].samples, dtype=np.float64)
            if samples.shape[0] == 0:
                continue

            lower, upper = np.percentile(samples, [1, 99])
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            if upper <= lower:
                upper = lower + 1.0

            numerical_task_codes.append(code)
            value_normalization[code] = (float(lower), float(upper))

        print(
            "SurvivEHR task fitted with "
            f"{len(original_codes)} competing-risk codes and {len(numerical_task_codes)} value codes"
        )

        return cls(
            original_codes=original_codes,
            numerical_task_codes=numerical_task_codes,
            value_normalization=value_normalization,
            time_scale_days=time_scale_days,
            concurrent_strategy=concurrent_strategy,
        )

    def __init__(
        self,
        original_codes: Sequence[str],
        numerical_task_codes: Sequence[str],
        value_normalization: Mapping[str, Tuple[float, float]],
        time_scale_days: float = 365.25,
        concurrent_strategy: str = "add_noise",
    ):
        self.original_codes = list(original_codes)
        self.original_code_to_id = {code: i for i, code in enumerate(self.original_codes)}
        self.numerical_task_codes = list(numerical_task_codes)
        self.numerical_task_code_set = set(self.numerical_task_codes)
        self.value_normalization = {
            code: (float(bounds[0]), float(bounds[1])) for code, bounds in value_normalization.items()
        }
        self.time_scale_days = time_scale_days
        self.concurrent_strategy = concurrent_strategy

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="survivehr",
            task_kwargs=dict(
                pretraining_task_info=[(code, 1.0) for code in self.original_codes],
                numerical_task_codes=self.numerical_task_codes,
                time_scale_days=self.time_scale_days,
                concurrent_strategy=self.concurrent_strategy,
            ),
        )

    def start_subject(self, _subject: meds_reader.Subject, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.per_subject_event_time: List[float] = []
        self.per_subject_event_label: List[int] = []
        self.per_subject_event_value: List[float] = []
        self.per_subject_value_mask: List[bool] = []

    def needs_exact(self) -> bool:
        return False

    def get_sampled_labels(self, length: int) -> int:
        return length

    def start_batch(self) -> None:
        self.batch_event_time: List[float] = []
        self.batch_event_label: List[int] = []
        self.batch_event_value: List[float] = []
        self.batch_value_mask: List[bool] = []

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        self.batch_event_time.extend([self.per_subject_event_time[i] for i in subject_label_offsets])
        self.batch_event_label.extend([self.per_subject_event_label[i] for i in subject_label_offsets])
        self.batch_event_value.extend([self.per_subject_event_value[i] for i in subject_label_offsets])
        self.batch_value_mask.extend([self.per_subject_value_mask[i] for i in subject_label_offsets])

    def _normalize_value(
        self,
        code: str,
        numeric_value: Optional[Any],
        text_value: Optional[Any],
    ) -> Tuple[float, bool]:
        if code not in self.numerical_task_code_set:
            return float("nan"), False
        if numeric_value is None or text_value is not None:
            return float("nan"), False

        value = _try_float(numeric_value)
        if value is None:
            return float("nan"), False

        lower, upper = self.value_normalization[code]
        if value < lower or value > upper:
            return float("nan"), False

        return ((value - lower) / (upper - lower)) - 0.5, True

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
        actually_add: bool = True,
        next_numeric_value: Optional[Any] = None,
        next_text_value: Optional[Any] = None,
    ) -> int:
        if next_date is None or next_features is None:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("SurvivEHRTask expects one raw next-code token")

        next_feature = next_features[0]
        if next_feature < 0 or next_feature >= len(self.original_codes):
            return 0

        delta_days = (next_date - current_date) / datetime.timedelta(days=1)
        if delta_days < 0:
            raise ValueError(f"Encountered negative next-event time delta: {delta_days}")

        if not actually_add:
            return 1

        code = self.original_codes[next_feature]
        event_value, value_mask = self._normalize_value(code, next_numeric_value, next_text_value)

        self.per_subject_event_time.append(float(delta_days))
        self.per_subject_event_label.append(int(next_feature) + 1)
        self.per_subject_event_value.append(float(event_value))
        self.per_subject_value_mask.append(bool(value_mask))

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {
            "event_time": np.array(self.batch_event_time, dtype=np.float32),
            "event_label": np.array(self.batch_event_label, dtype=np.int32),
            "event_value": np.array(self.batch_event_value, dtype=np.float32),
            "value_mask": np.array(self.batch_value_mask, dtype=np.bool_),
        }
