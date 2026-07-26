"""Label-free model-Fisher risk for recurrent-state row blocks.

For a model-sampled pseudo-label ``y`` and a recurrent state ``S``, define the
score gradient ``h = grad_S log p(y | context)``.  The local model-Fisher risk
of an endpoint error ``e`` is ``0.5 * E_y[(h dot e)^2]``.  This module compares
the aligned errors of two quantized endpoints row by row:

``0.5 * mean[(h dot e4)^2 - (h dot e8)^2]``.

The dot product is taken over the complete value dimension of each state row
before it is squared.  Consequently, interactions between coordinates within
a row are retained.  Cross-row interactions are deliberately excluded because
the result is a row-allocation score, not the full-state quadratic form.

The scoring function is pure and assumes its gradients came from pseudo-labels
sampled from the model distribution.  :func:`sample_model_pseudo_labels`
provides reproducible samples when passed an explicitly seeded generator.
Computing the corresponding per-example gradients is left to the model adapter
so this primitive does not mutate or retain a live Transformers cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RowBlockModelFisherRisk:
    """Model-Fisher endpoint risks and signed INT4-to-INT8 risk reduction.

    Each tensor has shape ``[heads, key_dim]``.  Positive
    ``risk_difference`` values favor promoting a row from INT4 to INT8;
    negative values are valid and are intentionally not clamped.
    """

    int4_risk: torch.Tensor
    int8_risk: torch.Tensor
    risk_difference: torch.Tensor
    probes: int
    batch_size: int

    @property
    def shape(self) -> torch.Size:
        """Return the shared row-score shape."""

        return self.risk_difference.shape


def _generator_device(generator: torch.Generator) -> torch.device:
    return torch.device(generator.device)


def sample_model_pseudo_labels(
    logits: torch.Tensor,
    *,
    probes: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample categorical pseudo-labels from the unmodified model distribution.

    ``logits`` must have shape ``[batch, vocabulary]``.  The returned labels
    have shape ``[probes, batch]``.  No temperature, top-k, or top-p transform
    is applied because those would define a different Fisher distribution.

    Reproducibility is explicit: callers should construct a generator on the
    same device, seed it, and record both its initial seed and the sampled label
    tensor.  Reusing a generator advances its state; two independently seeded
    generators with the same seed produce the same samples on the same PyTorch
    backend and version.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must contain at least one batch item and vocabulary entry")
    if isinstance(probes, bool) or not isinstance(probes, int) or probes <= 0:
        raise ValueError("probes must be a positive integer")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicitly seeded torch.Generator")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if logits.device.type == "meta":
        raise ValueError("logits must be materialized")
    if _generator_device(generator) != logits.device:
        raise ValueError("generator and logits must be on the same device")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values")

    probability_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probabilities = torch.softmax(logits.detach().to(probability_dtype), dim=-1)
    labels = torch.multinomial(
        probabilities,
        num_samples=probes,
        replacement=True,
        generator=generator,
    )
    return labels.transpose(0, 1).contiguous().detach()


def _validate_row_risk_inputs(
    pseudo_label_score_gradients: torch.Tensor,
    int4_errors: torch.Tensor,
    int8_errors: torch.Tensor,
) -> None:
    if pseudo_label_score_gradients.ndim != 5:
        raise ValueError(
            "pseudo_label_score_gradients must have shape "
            "[probes, batch, heads, key_dim, value_dim]"
        )
    if int4_errors.ndim != 4 or int8_errors.ndim != 4:
        raise ValueError(
            "int4_errors and int8_errors must have shape [batch, heads, key_dim, value_dim]"
        )
    if pseudo_label_score_gradients.shape[1:] != int4_errors.shape:
        raise ValueError("gradient dimensions after probes must match the aligned INT4 error shape")
    if int8_errors.shape != int4_errors.shape:
        raise ValueError("INT4 and INT8 errors must have the same shape")
    if any(size == 0 for size in pseudo_label_score_gradients.shape):
        raise ValueError("probe, batch, head, key, and value dimensions must be non-empty")

    tensors = (
        pseudo_label_score_gradients,
        int4_errors,
        int8_errors,
    )
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("gradients and endpoint errors must use floating-point dtypes")
    if any(tensor.dtype not in (torch.float32, torch.float64) for tensor in tensors):
        raise TypeError("gradients and endpoint errors must use torch.float32 or torch.float64")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise TypeError("gradients and endpoint errors must use the same dtype")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("gradients and endpoint errors must be on the same device")
    if pseudo_label_score_gradients.device.type == "meta":
        raise ValueError("gradients and endpoint errors must be materialized")
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("gradients and endpoint errors must contain only finite values")


def row_block_model_fisher_risk(
    pseudo_label_score_gradients: torch.Tensor,
    int4_errors: torch.Tensor,
    int8_errors: torch.Tensor,
) -> RowBlockModelFisherRisk:
    """Compute signed label-free model-Fisher risk reduction per state row.

    ``pseudo_label_score_gradients`` has shape
    ``[probes, batch, heads, key_dim, value_dim]``.  Probe ``p`` must be the
    score gradient for a label independently sampled from the model predictive
    distribution, not a dataset target.  The sign of the score gradient is
    immaterial after squaring, so gradients of pseudo-label NLL are equivalent.

    ``int4_errors`` and ``int8_errors`` have shape
    ``[batch, heads, key_dim, value_dim]`` and must be aligned errors relative
    to the same full-precision endpoint, for example ``Q4(S) - S`` and
    ``Q8(S) - S``.  Errors are shared across probes because every pseudo-label
    is drawn for the same endpoint.

    The result averages equally over probes and batch items.  It does not use a
    gold target, square ``h dot (e4 - e8)``, replace the row-block quadratic by
    a coordinate-diagonal approximation, or clamp negative differences.
    """

    _validate_row_risk_inputs(
        pseudo_label_score_gradients,
        int4_errors,
        int8_errors,
    )
    gradients = pseudo_label_score_gradients.detach()
    error4 = int4_errors.detach().unsqueeze(0)
    error8 = int8_errors.detach().unsqueeze(0)

    projection4 = (gradients * error4).sum(dim=-1)
    projection8 = (gradients * error8).sum(dim=-1)
    per_probe_int4_risk = 0.5 * projection4.square()
    per_probe_int8_risk = 0.5 * projection8.square()
    per_probe_difference = 0.5 * (projection4.square() - projection8.square())

    int4_risk = per_probe_int4_risk.mean(dim=(0, 1))
    int8_risk = per_probe_int8_risk.mean(dim=(0, 1))
    risk_difference = per_probe_difference.mean(dim=(0, 1))
    outputs = (int4_risk, int8_risk, risk_difference)
    if not all(torch.isfinite(tensor).all().item() for tensor in outputs):
        raise RuntimeError("model-Fisher row risk became non-finite")

    return RowBlockModelFisherRisk(
        int4_risk=int4_risk.detach(),
        int8_risk=int8_risk.detach(),
        risk_difference=risk_difference.detach(),
        probes=pseudo_label_score_gradients.shape[0],
        batch_size=pseudo_label_score_gradients.shape[1],
    )
