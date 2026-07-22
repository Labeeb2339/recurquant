"""Teacher-forced fidelity metrics and paired uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .metrics import tail_mean, token_kl_divergence, top1_agreement


@dataclass(frozen=True, slots=True)
class TokenFidelity:
    """Per-token comparison between reference and candidate distributions."""

    kl: torch.Tensor
    reference_nll: torch.Tensor
    candidate_nll: torch.Tensor
    top1_agreement: torch.Tensor

    def to_cpu(self) -> TokenFidelity:
        return TokenFidelity(
            kl=self.kl.detach().to(torch.float32).cpu(),
            reference_nll=self.reference_nll.detach().to(torch.float32).cpu(),
            candidate_nll=self.candidate_nll.detach().to(torch.float32).cpu(),
            top1_agreement=self.top1_agreement.detach().cpu(),
        )


def token_fidelity(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> TokenFidelity:
    """Compare logits while scoring the supplied ground-truth next tokens."""

    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have identical shapes")
    if target_ids.shape != reference_logits.shape[:-1]:
        raise ValueError("target_ids must match every logits dimension except vocabulary")
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_ids must use an integer dtype")

    reference_log_probs = torch.log_softmax(reference_logits.to(torch.float32), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.to(torch.float32), dim=-1)
    targets = target_ids.to(torch.int64).unsqueeze(-1)
    return TokenFidelity(
        kl=token_kl_divergence(reference_logits, candidate_logits),
        reference_nll=-reference_log_probs.gather(-1, targets).squeeze(-1),
        candidate_nll=-candidate_log_probs.gather(-1, targets).squeeze(-1),
        top1_agreement=top1_agreement(reference_logits, candidate_logits),
    )


def fidelity_summary(fidelity: TokenFidelity) -> dict[str, float | int]:
    """Aggregate token metrics without retaining vocabulary-sized logits."""

    kl = fidelity.kl.detach().to(torch.float32).flatten()
    reference_nll = fidelity.reference_nll.detach().to(torch.float32).flatten()
    candidate_nll = fidelity.candidate_nll.detach().to(torch.float32).flatten()
    agreement = fidelity.top1_agreement.detach().to(torch.float32).flatten()
    if not kl.numel():
        raise ValueError("fidelity must contain at least one token")
    tensors = (kl, reference_nll, candidate_nll, agreement)
    if any(not torch.isfinite(values).all() for values in tensors):
        raise ValueError("fidelity metrics must be finite")

    mean_reference_nll = reference_nll.mean()
    mean_candidate_nll = candidate_nll.mean()
    return {
        "token_count": int(kl.numel()),
        "mean_kl": float(kl.mean().item()),
        "cvar95_kl": float(tail_mean(kl, fraction=0.05).item()),
        "max_kl": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
        "reference_nll": float(mean_reference_nll.item()),
        "candidate_nll": float(mean_candidate_nll.item()),
        "delta_nll": float((mean_candidate_nll - mean_reference_nll).item()),
    }


def paired_bootstrap_mean_improvement(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2339,
) -> dict[str, Any]:
    """Bootstrap paired mean improvement, defined as baseline minus candidate."""

    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline_values and candidate_values must have equal length")
    if not baseline_values:
        raise ValueError("at least one paired example is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    baseline = np.asarray(baseline_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired values must be finite")
    differences = baseline - candidate
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    sampled_means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled_means, [alpha, 1.0 - alpha])
    return {
        "paired_examples": int(differences.size),
        "mean_improvement": float(differences.mean()),
        "confidence": confidence,
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_samples": samples,
        "seed": seed,
    }
