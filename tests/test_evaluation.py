from __future__ import annotations

import pytest
import torch

from recurquant.evaluation import (
    fidelity_summary,
    paired_bootstrap_mean_improvement,
    token_fidelity,
)


def test_token_fidelity_scores_ground_truth_targets() -> None:
    reference = torch.tensor([[[4.0, 1.0, -1.0], [0.0, 2.0, 1.0]]])
    candidate = torch.tensor([[[3.0, 2.0, -1.0], [0.0, 1.0, 2.0]]])
    targets = torch.tensor([[0, 2]])

    result = token_fidelity(reference, candidate, targets)
    summary = fidelity_summary(result)

    assert result.kl.shape == targets.shape
    assert summary["token_count"] == 2
    assert summary["mean_kl"] > 0
    assert summary["candidate_nll"] > 0
    assert summary["top1_agreement"] == pytest.approx(0.5)


def test_token_fidelity_rejects_mismatched_targets() -> None:
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="target_ids"):
        token_fidelity(logits, logits, torch.zeros(1, 1, dtype=torch.long))


def test_paired_bootstrap_is_seeded_and_improvement_is_positive() -> None:
    baseline = [2.0, 3.0, 4.0, 5.0]
    candidate = [1.0, 2.0, 3.0, 4.0]

    first = paired_bootstrap_mean_improvement(baseline, candidate, samples=200, seed=17)
    second = paired_bootstrap_mean_improvement(baseline, candidate, samples=200, seed=17)

    assert first == second
    assert first["mean_improvement"] == pytest.approx(1.0)
    assert first["confidence_interval"] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([1.0], [], "equal length"),
        ([], [], "at least one"),
        ([float("nan")], [1.0], "finite"),
    ],
)
def test_paired_bootstrap_rejects_invalid_values(
    baseline: list[float], candidate: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_mean_improvement(baseline, candidate)
