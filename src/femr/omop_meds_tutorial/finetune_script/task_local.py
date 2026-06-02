import collections
import datetime
from typing import Any, List, Mapping, Optional, Sequence

import meds
import meds_reader
import numpy as np

import femr.models.config
import femr.ontology
from femr.models.tasks.motor import Task


class LabeledSubjectTaskWithLabels(Task):
    """LabeledSubjectTask that returns labels in batch data for fine-tuning."""

    def __init__(self, labels: Sequence[meds.Label], observation_window: Optional[int] = None):
        super().__init__()
        self.label_map: Mapping[int, Any] = collections.defaultdict(list)
        for label in labels:
            self.label_map[label["subject_id"]].append(label)

        for k, v in self.label_map.items():
            v.sort(key=lambda a: a["prediction_time"])

        if observation_window is not None:
            assert observation_window > 0
        self.observation_window = observation_window

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(task_type="labeled_subjects")

    def start_subject(self, subject: meds_reader.Subject, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.current_labels = self.label_map[subject.subject_id]
        self.current_label_index = 0
        self.per_subject_labels: List[int] = []
        self.per_subject_subject_ids: List[int] = []
        self.per_subject_prediction_timestamps: List[int] = []
        self.per_subject_event_timestamps: List[int] = []

    def needs_exact(self) -> bool:
        return True

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []
        self.batch_subject_ids: List[int] = []
        self.batch_prediction_timestamps: List[int] = []
        self.batch_event_timestamps: List[int] = []

    def add_subject_labels(self, subject_label_offsets: List[int]) -> None:
        self.batch_labels.extend([self.per_subject_labels[i] for i in subject_label_offsets])
        self.batch_subject_ids.extend([self.per_subject_subject_ids[i] for i in subject_label_offsets])
        self.batch_prediction_timestamps.extend([self.per_subject_prediction_timestamps[i] for i in subject_label_offsets])
        self.batch_event_timestamps.extend([self.per_subject_event_timestamps[i] for i in subject_label_offsets])

    def add_event(
            self,
            current_date: datetime.datetime,
            next_date: Optional[datetime.datetime],
            next_features: Optional[Sequence[int]] = None,
            actually_add: Optional[bool] = True,
    ) -> int:
        has_label = False
        label_value = None
        matched_label = None

        while True:
            if self.current_label_index == len(self.current_labels):
                break

            current_label = self.current_labels[self.current_label_index]

            if self.observation_window is not None:
                observation_start_time = (
                        current_label["prediction_time"] - datetime.timedelta(days=self.observation_window)
                )
                is_valid = observation_start_time <= current_date <= current_label["prediction_time"]
                next_valid = (
                        next_date is not None and
                        observation_start_time <= next_date <= current_label["prediction_time"]
                )
            else:
                is_valid = current_date <= current_label["prediction_time"]
                next_valid = next_date is not None and next_date <= current_label["prediction_time"]

            if next_valid:
                break

            if is_valid:
                has_label = True
                label_value = current_label["boolean_value"]
                matched_label = current_label
                self.current_label_index += 1
                break
            else:
                break

        if has_label:
            self.per_subject_labels.append(int(label_value))
            self.per_subject_subject_ids.append(matched_label["subject_id"])
            epoch = datetime.datetime(1970, 1, 1)
            pt = matched_label["prediction_time"]
            self.per_subject_prediction_timestamps.append(int((pt - epoch).total_seconds() * 1_000_000_000))
            self.per_subject_event_timestamps.append(int((current_date - epoch).total_seconds() * 1_000_000_000))
            return 1
        else:
            return 0

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {
            "labels": np.array(self.batch_labels, dtype=np.int32),
            "subject_ids": np.array(self.batch_subject_ids, dtype=np.int64),
            "prediction_timestamps": np.array(self.batch_prediction_timestamps, dtype=np.int64),
            "event_timestamps": np.array(self.batch_event_timestamps, dtype=np.int64),
        }

    def get_sampled_labels(self, length: int) -> int:
        return length