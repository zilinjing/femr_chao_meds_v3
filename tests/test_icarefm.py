import datetime
from types import SimpleNamespace

import numpy as np
import torch

from femr.models.architecture.embedding import ICareFMTaskHead
from femr.models.tasks.icarefm_task import ICareFMTask, build_first_event_label


def test_build_first_event_label_marks_first_future_crossing():
    y, weight = build_first_event_label(
        future_bins=[1, 3],
        future_values=[-0.5, 1.2],
        tau=1.0,
        direction=1,
        t_surv=4,
    )

    assert y.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert weight.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_build_first_event_label_uses_full_horizon_when_no_crossing():
    y, weight = build_first_event_label(
        future_bins=[1, 3],
        future_values=[-0.5, 0.5],
        tau=1.0,
        direction=1,
        t_surv=4,
    )

    assert y.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert weight.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_icarefm_task_keeps_sparse_future_evidence_and_samples_queries_at_cleanup():
    task = ICareFMTask(
        target_codes=["A", "B"],
        value_normalization={"A": (10.0, 2.0), "B": (0.0, 1.0)},
        t_surv=4,
        tau_max=1.0,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    subject = SimpleNamespace(
        events=[
            SimpleNamespace(time=start, code="A", numeric_value=10.0, text_value=None),
            SimpleNamespace(time=start + datetime.timedelta(hours=1, minutes=30), code="A", numeric_value=14.0, text_value=None),
            SimpleNamespace(time=start + datetime.timedelta(hours=3), code="B", numeric_value=2.0, text_value=None),
        ]
    )

    task.start_subject(subject, None)
    task.start_batch()

    assert task.add_event(start, start + datetime.timedelta(hours=1), actually_add=False) == 1
    assert task.add_event(start, start + datetime.timedelta(hours=1), actually_add=True) == 1
    task.add_subject_labels([0])

    batch = task.get_batch_data()
    assert batch["future_indptr"].tolist() == [0, 2]
    assert batch["future_target_idx"].tolist() == [0, 1]
    assert batch["future_bin_idx"].tolist() == [2, 3]
    assert batch["future_value_scaled"].tolist() == [2.0, 2.0]
    assert "current_indptr" not in batch

    np.random.seed(0)
    clean = task.cleanup(batch)
    assert clean["query_anchor_idx"].tolist() == [0, 0]
    assert clean["query_target_idx"].tolist() == [0, 1]
    assert not torch.equal(clean["query_tau"][0:1], clean["query_tau"][1:2])
    assert clean["query_y"].shape == (2, 4)
    assert clean["query_weight"].shape == (2, 4)
    assert torch.all(clean["query_tau"].abs() <= 1.0)

    np.random.seed(1)
    first = task.cleanup(batch)
    np.random.seed(2)
    second = task.cleanup(batch)
    assert not torch.equal(first["query_tau"], second["query_tau"])


def test_icarefm_task_returns_empty_sparse_row_for_future_measurement_beyond_horizon():
    task = ICareFMTask(
        target_codes=["A"],
        value_normalization={"A": (0.0, 1.0)},
        t_surv=4,
        tau_max=1.0,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    subject = SimpleNamespace(
        events=[
            SimpleNamespace(time=start, code="A", numeric_value=1.0, text_value=None),
            SimpleNamespace(time=start + datetime.timedelta(hours=8), code="A", numeric_value=2.0, text_value=None),
        ]
    )

    task.start_subject(subject, None)
    task.start_batch()

    assert task.add_event(start, start + datetime.timedelta(hours=1), actually_add=True) == 1
    task.add_subject_labels([0])

    batch = task.get_batch_data()
    assert batch["future_indptr"].tolist() == [0, 0]
    clean = task.cleanup(batch)
    assert clean["query_anchor_idx"].numel() == 0


def test_icarefm_task_ignores_same_time_measurements_for_labels():
    task = ICareFMTask(
        target_codes=["A"],
        value_normalization={"A": (0.0, 1.0)},
        t_surv=4,
        tau_max=1.0,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    subject = SimpleNamespace(
        events=[
            SimpleNamespace(time=start, code="A", numeric_value=1.0, text_value=None),
            SimpleNamespace(time=start + datetime.timedelta(hours=1), code="A", numeric_value=2.0, text_value=None),
        ]
    )

    task.start_subject(subject, None)
    task.start_batch()
    assert task.add_event(start, start + datetime.timedelta(minutes=30), actually_add=True) == 1
    task.add_subject_labels([0])

    batch = task.get_batch_data()
    assert batch["future_indptr"].tolist() == [0, 1]
    assert batch["future_bin_idx"].tolist() == [1]
    assert batch["future_value_scaled"].tolist() == [2.0]


def test_icarefm_head_computes_masked_bce_loss():
    head = ICareFMTaskHead(hidden_size=8, target_codes=["A", "B"], t_surv=4)
    features = torch.randn(2, 8)
    batch = {
        "query_anchor_idx": torch.tensor([0, 1], dtype=torch.long),
        "query_target_idx": torch.tensor([0, 1], dtype=torch.long),
        "query_tau": torch.tensor([0.5, -0.25], dtype=torch.float32),
        "query_direction": torch.tensor([1, 0], dtype=torch.long),
        "query_y": torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "query_weight": torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }

    loss, result = head(features, batch, return_logits=True)

    assert torch.isfinite(loss)
    assert result["icarefm_logits"].shape == (2, 4)
    assert head.last_metrics["icarefm/valid_weights"].item() == 6.0
