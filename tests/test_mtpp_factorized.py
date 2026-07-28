"""Tests for the ORA-Factorized ablation head (`MTPPFactorizedTaskHead`).

The claim behind the ablation is that "two separate losses, one for event timing and one
for values" is exactly the ORA loss with the time-value table built as a multiplication
table, `cell[k, l] = when[k] * what[l]`.  `test_factorized_head_matches_ora_on_a_product_table`
checks that numerically: it hands `MTPPSharedTaskHead` (ORA) the table implied by the
factorized head's own two marginals and asserts the two losses agree.
"""

import numpy as np
import torch

from femr.models.architecture.embedding import MTPPFactorizedTaskHead, MTPPSharedTaskHead

T = 4  # time bins
V = 3  # value bins
K_NON = 5  # non-numerical codes
K_NUM = 2  # numerical codes
D = 8  # backbone hidden size
D2 = 6  # final layer size
P = 7  # prediction points


def _head_kwargs():
    # Bin edges only fix the output shapes; the loss reads pre-binned indicators.
    return dict(
        hidden_size=D,
        pretraining_task_info=[(f"code_{i}", 1.0) for i in range(K_NON + K_NUM)],
        non_numerical_task=[f"non_{i}" for i in range(K_NON)],
        numerical_task=[f"num_{i}" for i in range(K_NUM)],
        value_bins=np.tile(np.linspace(-1.0, 1.0, V + 1), (K_NUM, 1)),
        non_numerical_task_time_bins=np.tile(np.linspace(0.0, 1.0, T + 1), (K_NON, 1)),
        numerical_task_time_bins=np.tile(np.linspace(0.0, 1.0, T + 1), (K_NUM, 1)),
        final_layer_size=D2,
    )


def _one_hot(num_bins, num_tasks, indices):
    """One-hot along the bin axis: [P, num_bins, num_tasks]."""
    out = torch.zeros(P, num_bins, num_tasks, dtype=torch.bool)
    out.scatter_(1, (indices % num_bins).unsqueeze(1), True)
    return out


def _make_batch(seed, all_value_bins_valid=True):
    g = torch.Generator().manual_seed(seed)

    # Non-numerical branch: exactly one marked bin per (prediction point, code).
    non_is_event = _one_hot(T, K_NON, torch.randint(0, T, (P, K_NON), generator=g))
    non_is_censored = torch.rand(P, K_NON, generator=g) < 0.4

    # Numerical branch: each (prediction point, code) is either observed or censored.
    is_censored = torch.rand(P, K_NUM, generator=g) < 0.5
    time_bin = _one_hot(T, K_NUM, torch.randint(0, T, (P, K_NUM), generator=g))
    value_bin = _one_hot(V, K_NUM, torch.randint(0, V, (P, K_NUM), generator=g))

    time_event = time_bin & (~is_censored).unsqueeze(1)
    time_censor = time_bin & is_censored.unsqueeze(1)
    value_event = value_bin & (~is_censored).unsqueeze(1)

    value_valid_mask = np.ones((K_NUM, V), dtype=bool)
    if not all_value_bins_valid:
        # Mimic a code whose value quantiles collapsed: its last bin is unusable, so any
        # event that landed there moves to bin 0.
        value_valid_mask[0, -1] = False
        invalid = torch.from_numpy(~value_valid_mask).T.unsqueeze(0)
        value_event = value_event & ~invalid
        value_event[:, 0, :] |= (~value_event.any(dim=1)) & (~is_censored)

    return {
        "non_numerical_is_event": non_is_event,
        "non_numerical_is_censored": non_is_censored,
        "numerical_time_event_bin": time_event,
        "numerical_time_censor_bin": time_censor,
        "numerical_value_event_bin": value_event,
        "value_valid_mask": value_valid_mask,
    }


class _FixedLogits(torch.nn.Module):
    """Stand-in for ORA's `numerical_task_layer` that ignores its input and returns `logits`.

    ORA computes `numerical_task_layer(norm(feats)).reshape(P, T * V, K)`, so the tensor is
    shaped `[P, T, V * K]` to survive that reshape.
    """

    def __init__(self, logits):  # logits: [P, T, V, K]
        super().__init__()
        self.register_buffer("logits", logits.reshape(P, T, V * K_NUM))

    def forward(self, x):
        return self.logits


def _losses(seed, all_value_bins_valid=True):
    """Return (factorized loss, ORA loss on the same multiplication table)."""
    torch.manual_seed(seed)
    batch = _make_batch(seed, all_value_bins_valid)
    features = torch.randn(P, D)

    factorized = MTPPFactorizedTaskHead(**_head_kwargs())
    ora = MTPPSharedTaskHead(**_head_kwargs())

    # Give ORA the same non-numerical branch, so any difference comes from the numerical one.
    ora.final_layer.load_state_dict(factorized.final_layer.state_dict())
    ora.non_numerical_task_layer.load_state_dict(factorized.non_numerical_task_layer.state_dict())
    ora.norm.load_state_dict(factorized.norm.state_dict())

    with torch.no_grad():
        loss_factorized, result = factorized(features, batch, return_logits=True)
        # softmax(log p) == p for a normalised p, so this hands ORA the multiplication table.
        ora.numerical_task_layer = _FixedLogits(torch.log(result["value_dependent_logits"]))
        loss_ora, _ = ora(features, batch)

    return loss_factorized, loss_ora, result


def test_factorized_head_matches_ora_on_a_product_table():
    for seed in (0, 1, 2):
        loss_factorized, loss_ora, _ = _losses(seed)
        assert torch.allclose(loss_factorized, loss_ora, atol=1e-5), \
            f"seed {seed}: {loss_factorized.item()} vs {loss_ora.item()}"


def test_factorized_head_matches_ora_with_collapsed_value_bins():
    loss_factorized, loss_ora, _ = _losses(3, all_value_bins_valid=False)
    assert torch.allclose(loss_factorized, loss_ora, atol=1e-5), \
        f"{loss_factorized.item()} vs {loss_ora.item()}"


def test_marginals_are_normalised_and_respect_the_value_mask():
    _, _, result = _losses(4, all_value_bins_valid=False)
    when, what = result["time_marginal_probs"], result["value_marginal_probs"]
    assert torch.allclose(when.sum(dim=1), torch.ones(P, K_NUM), atol=1e-4)
    assert torch.allclose(what.sum(dim=1), torch.ones(P, K_NUM), atol=1e-4)
    # Code 0's last value bin is masked out, so it must carry no probability mass.
    assert torch.all(what[:, -1, 0] < 1e-5)


def test_head_is_not_smaller_than_ora():
    """The ablation must not be handicapped by having fewer head parameters."""
    n_factorized = sum(p.numel() for p in MTPPFactorizedTaskHead(**_head_kwargs()).parameters())
    n_ora = sum(p.numel() for p in MTPPSharedTaskHead(**_head_kwargs()).parameters())
    assert n_factorized >= n_ora, f"factorized head is smaller than ORA: {n_factorized} < {n_ora}"


def test_gradients_are_finite():
    torch.manual_seed(0)
    features = torch.randn(P, D, requires_grad=True)
    head = MTPPFactorizedTaskHead(**_head_kwargs())
    loss, _ = head(features, _make_batch(5, all_value_bins_valid=False))
    loss.backward()
    assert torch.isfinite(features.grad).all()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"
