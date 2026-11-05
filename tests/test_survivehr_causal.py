import datetime
import math

import dill
import pytest

from femr.models.tasks.survivehr_task import SurvivEHRTask

START = datetime.datetime(2024, 1, 1, 0, 0, 0)
LATER = datetime.datetime(2024, 1, 1, 12, 10, 0)


def make_task():
    task = SurvivEHRTask(
        original_codes=["A", "B", "LAB"], numerical_task_codes=["LAB"], value_normalization={"LAB": (0.0, 10.0)}
    )
    task.start_subject(None, None)
    return task


def test_add_event_records_same_timestamp_events_with_time_zero():
    # They are kept: the head's add_noise strategy separates zero times, the task does not drop them.
    task = make_task()

    assert task.add_event(START, START, [0]) == 1
    assert task.add_event(START, LATER, [2], next_numeric_value=7.5) == 1

    assert task.per_subject_event_label == [1, 3]  # code index + 1
    assert task.per_subject_event_time == pytest.approx([0.0, 730 / 1440])  # days
    assert math.isnan(task.per_subject_event_value[0])
    assert task.per_subject_event_value[1] == pytest.approx(0.25)  # 7.5 in (0, 10), centered on 0
    assert task.per_subject_value_mask == [False, True]


def test_task_can_be_pickled_after_processing_a_subject():
    task = make_task()
    task.add_event(START, LATER, [2], next_numeric_value=5.0)

    restored = dill.loads(dill.dumps(task))

    assert restored.value_normalization == {"LAB": (0.0, 10.0)}
    assert restored.per_subject_event_label == [3]
    assert restored.per_subject_event_value == [0.0]
