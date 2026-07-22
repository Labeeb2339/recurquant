from __future__ import annotations

import pytest
import torch

from recurquant.metrics import (
    paired_logit_summary,
    spearman_correlation,
    tail_mean,
    token_kl_divergence,
)


def test_identical_logits_have_zero_divergence_and_full_agreement() -> None:
    logits = torch.tensor([[[1.0, 2.0, -1.0], [0.2, -0.5, 0.8]]])

    summary = paired_logit_summary(logits, logits.clone())

    assert summary["mean_kl"] == pytest.approx(0.0, abs=1e-7)
    assert summary["cvar95_kl"] == pytest.approx(0.0, abs=1e-7)
    assert summary["top1_agreement"] == 1.0


def test_kl_is_nonnegative_up_to_floating_point_noise() -> None:
    reference = torch.tensor([[1.0, 0.0, -1.0]])
    candidate = torch.tensor([[0.0, 1.0, -1.0]])

    kl = token_kl_divergence(reference, candidate)

    assert kl.item() > 0


def test_tail_mean_selects_largest_values() -> None:
    values = torch.arange(1, 21, dtype=torch.float32)
    assert tail_mean(values, fraction=0.10).item() == pytest.approx(19.5)


def test_tail_mean_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        tail_mean(torch.ones(2), fraction=0)


def test_spearman_correlation_handles_order_and_ties() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 2], [4, 4, 9]) == pytest.approx(1.0)


def test_spearman_correlation_rejects_constant_input() -> None:
    with pytest.raises(ValueError, match="constant"):
        spearman_correlation([1, 1], [2, 3])
