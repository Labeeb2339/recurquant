"""Output-distribution metrics for paired reference and quantized runs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def token_kl_divergence(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(reference || candidate) over the final vocabulary dimension."""

    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have identical shapes")
    reference_log_probs = torch.log_softmax(reference_logits.to(torch.float32), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.to(torch.float32), dim=-1)
    reference_probs = reference_log_probs.exp()
    return (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1)


def top1_agreement(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have identical shapes")
    return reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)


def tail_mean(values: torch.Tensor, *, fraction: float = 0.05) -> torch.Tensor:
    """Mean of the largest ``fraction`` of finite values (CVaR-style tail mean)."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    flattened = values.detach().to(torch.float32).flatten()
    flattened = flattened[torch.isfinite(flattened)]
    if flattened.numel() == 0:
        raise ValueError("values must contain at least one finite element")
    count = max(1, math.ceil(flattened.numel() * fraction))
    return torch.topk(flattened, k=count, largest=True, sorted=False).values.mean()


def paired_logit_summary(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, float]:
    kl = token_kl_divergence(reference_logits, candidate_logits).flatten()
    agreement = top1_agreement(reference_logits, candidate_logits).to(torch.float32)
    return {
        "mean_kl": float(kl.mean().item()),
        "cvar95_kl": float(tail_mean(kl, fraction=0.05).item()),
        "max_kl": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for exact ties."""

    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    if len(left) < 2:
        raise ValueError("at least two paired values are required")
    left_ranks = torch.tensor(_average_ranks(left), dtype=torch.float64)
    right_ranks = torch.tensor(_average_ranks(right), dtype=torch.float64)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant input")
    return float(torch.dot(left_centered, right_centered).div(denominator).item())
