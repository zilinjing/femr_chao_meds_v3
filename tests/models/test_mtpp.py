"""MTPP: raw value counts for the numeric-task threshold, and value-less occurrences of numeric tasks."""

import dataclasses
import datetime
import functools
import random
from typing import Any, List, Optional

import meds
import numpy as np
import torch

from femr.models.architecture.embedding import MTPPSharedTaskHead
from femr.models.tasks.mtpp import MTPP_Task, _prefit_motor_agg, _prefit_motor_map

DAY = datetime.timedelta(days=1)
START = datetime.datetime(2020, 1, 1)


@dataclasses.dataclass
class Event:
    code: str
    time: Optional[datetime.datetime]
    numeric_value: Optional[float] = None
    text_value: Optional[str] = None


@dataclasses.dataclass
class Subject:
    subject_id: int
    events: List[Any]


class FlatOntology:
    def get_all_parents(self, code):
        return {code}


def _subject(x_events):
    """Birth, then a daily code A on days 1..20 (label positions), plus the given X events."""
    events = [Event(meds.birth_code, datetime.datetime(1980, 1, 1))]
    events += [Event("A", START + d * DAY) for d in range(1, 21)]
    events += x_events
    events.sort(key=lambda e: e.time)
    return Subject(1, events)


def test_value_count_is_the_raw_count_not_the_label_position_count():
    random.seed(0)
    # One numeric X on day 10: it is the next occurrence at the label positions on days 1..9.
    subject = _subject([Event("X", START + 10 * DAY + datetime.timedelta(hours=1), numeric_value=5.0)])
    _, _, value_samples, raw_counts = _prefit_motor_map([subject], tasks=["X", "A"], ontology=FlatOntology())

    assert raw_counts[0] == 1
    assert len(value_samples[0].samples) > 1  # counted once per label position before it


def test_count_label_positions_keeps_codes_the_raw_count_filters_out():
    class Db:
        def map(self, func):
            return [func([_subject([Event("X", START + 10 * DAY + datetime.timedelta(hours=1), numeric_value=5.0)])])]

    class Tokenizer:
        ontology = FlatOntology()
        dictionary = {"vocab": [{"type": "code", "code_string": code} for code in ("A", "X")]}

    def fit(count_label_positions):
        random.seed(0)
        return MTPP_Task.fit_pretraining_task_info(
            Db(), Tokenizer(), num_tasks=2, num_final_tasks=10, num_bins=2, final_layer_size=4,
            value_count_threshold=3, num_value_bins=2, count_label_positions=count_label_positions,
        )

    assert "X" not in fit(False).numerical_task_to_index_map  # 1 raw value < 3
    assert "X" in fit(True).numerical_task_to_index_map  # counted once per label position before it


def test_shard_results_merge_like_db_map():
    # db.map returns one tuple per shard; MTPP_Task.fit_pretraining_task_info reduces them with
    # _prefit_motor_agg. A failure here made the real run hang inside meds_reader.
    random.seed(0)
    subject = _subject([Event("X", START + 10 * DAY + datetime.timedelta(hours=1), numeric_value=5.0)])
    shards = [_prefit_motor_map([subject], tasks=["X", "A"], ontology=FlatOntology()) for _ in range(3)]
    merged = functools.reduce(_prefit_motor_agg, shards)

    assert merged[3].tolist() == [3, 0]
    assert merged[1][0][1] == 3 * shards[1][1][0][1]  # event counts add up


def _task():
    time_bins = np.array([[0.0, 2 * 86400.0, np.inf]])
    return MTPP_Task(
        pretraining_task_info=[("X", 1.0, 0, 1, 1.0), ("A", 1.0, 0, 1, 1.0)],
        non_numerical_time_bins=time_bins,
        numerical_time_bins=time_bins,
        final_layer_size=4,
        value_bins=np.array([[-np.inf, 3.0, np.inf]]),
        value_valid_mask=np.array([[1, 1, 1]]),
        num_value_bins=2,
        numerical_task_list=["X"],
    )


def test_value_less_occurrence_of_numeric_task_is_a_time_only_event():
    # Day 1 label position: the next X (day 3) has no value; a valued X follows on day 5.
    subject = _subject([
        Event("X", START + 3 * DAY + datetime.timedelta(hours=1)),
        Event("X", START + 5 * DAY + datetime.timedelta(hours=1), numeric_value=7.2),
    ])
    task = _task()
    task.start_subject(subject, FlatOntology())
    task.start_batch()
    assert task.add_event(START + 1 * DAY, START + 2 * DAY) == 1  # value-less next X
    assert task.add_event(START + 4 * DAY, START + 5 * DAY) == 1  # valued next X
    task.add_subject_labels([0, 1])

    data = task.get_batch_data()
    batch = {
        "censor_time": torch.from_numpy(data["censor_time"]),
        "time_sparse": {k: torch.from_numpy(v) for k, v in data["time_sparse"].items()},
        "value_sparse": {k: torch.from_numpy(v) for k, v in data["value_sparse"].items()},
    }
    labels = task.cleanup(batch)

    # Both positions are events for X (not censored); only the second has a value bin.
    assert not labels["numerical_is_censored"].any()
    assert labels["numerical_time_event_bin"].sum(dim=1).tolist() == [[1], [1]]
    assert labels["numerical_value_event_bin"][0].sum() == 0
    assert labels["numerical_value_event_bin"][1, :, 0].tolist() == [False, True]  # 7.2 >= 3.0

    # The head scores the value-less event by its time bin summed over value bins.
    torch.manual_seed(0)
    config = task.get_task_config().task_kwargs
    head = MTPPSharedTaskHead(hidden_size=8, **config)
    labels["value_valid_mask"] = np.array([[1, 1, 1]])
    loss, result = head(torch.randn(2, 8), labels, return_logits=True)
    parts = result["loss_components"]
    assert torch.isfinite(loss)
    assert parts["L_num_time_only"] > 0 and parts["L_num_event"] > 0 and parts["L_num_censor"] == 0

    probs = result["value_dependent_logits"]  # [points, time_bins, value_bins, tasks]
    time_bin = labels["numerical_time_event_bin"][0, :, 0].nonzero().item()
    expected = -torch.log(probs[0, time_bin, :, 0].sum())
    assert torch.allclose(parts["L_num_time_only"], expected, atol=1e-5)
