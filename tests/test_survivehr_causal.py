import datetime
import threading
from types import SimpleNamespace

import dill

from femr.models.tasks.survivehr_causal import SurvivEHRCausalTask


def test_add_event_consumes_same_timestamp_events_without_desynchronizing_cursor():
    task = SurvivEHRCausalTask(pretraining_task_info=[], numerical_task_codes=[])
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    later = datetime.datetime(2024, 1, 1, 12, 10, 0)

    task._current_subject_id = 123
    task._next_event_cursor = 1
    task._kept_events = [
        SimpleNamespace(time=start),
        SimpleNamespace(time=start),
        SimpleNamespace(time=later),
    ]
    task._kept_event_targets = [
        (0, None),
        (1, None),
        (2, 3.5),
    ]
    task.per_subject_event_time = []
    task.per_subject_event_label = []
    task.per_subject_event_value = []

    assert task.add_event(start, start, actually_add=True) == 0
    assert task._next_event_cursor == 2

    assert task.add_event(start, later, actually_add=True) == 1
    assert task._next_event_cursor == 3
    assert task.per_subject_event_label == [2]
    assert task.per_subject_event_value == [3.5]
    assert task.per_subject_event_time == [12.166666666666666 / 24.0]


def test_task_drops_transient_subject_state_during_pickling():
    task = SurvivEHRCausalTask(pretraining_task_info=[], numerical_task_codes=[])
    task._current_subject_id = 123
    task._kept_events = [threading.Lock()]
    task._kept_event_targets = [(1, None)]
    task._next_event_cursor = 1

    restored = dill.loads(dill.dumps(task))

    assert not hasattr(restored, "_kept_events")
    assert not hasattr(restored, "_kept_event_targets")
    assert not hasattr(restored, "_next_event_cursor")
