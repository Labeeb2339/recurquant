"""Deterministic quantize/dequantize emulation for recurrent matrix states.

The functions in this module measure numerical effects and estimated storage.
They do not claim real memory or latency savings: tensors are dequantized back
to their original dtype so an unmodified PyTorch model can consume them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch

RoundingMode = Literal["nearest", "stochastic"]


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Configuration for symmetric signed group quantization."""

    bits: int
    group_size: int = 128
    scale_bits: int = 16
    flatten_last_dims: int = 2
    rounding: RoundingMode = "nearest"
    seed: int = 2339
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError("bits must be between 2 and 16")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.scale_bits <= 0:
            raise ValueError("scale_bits must be positive")
        if self.flatten_last_dims <= 0:
            raise ValueError("flatten_last_dims must be positive")
        if self.rounding not in ("nearest", "stochastic"):
            raise ValueError("rounding must be 'nearest' or 'stochastic'")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True, slots=True)
class QuantizationResult:
    """A dequantized tensor plus numerical and modeled-storage evidence."""

    tensor: torch.Tensor
    spec: QuantizationSpec
    elements: int
    groups: int
    payload_bits: int
    scale_bits: int
    baseline_bytes: int
    estimated_bytes: int
    mean_squared_error: float
    relative_l2_error: float
    max_absolute_error: float
    quantizer_boundary_fraction: float
    scale_min: float
    scale_max: float

    @property
    def compression_ratio(self) -> float:
        return self.baseline_bytes / self.estimated_bytes

    def evidence_dict(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "elements": self.elements,
            "groups": self.groups,
            "payload_bits": self.payload_bits,
            "scale_bits": self.scale_bits,
            "baseline_bytes": self.baseline_bytes,
            "estimated_bytes": self.estimated_bytes,
            "mean_squared_error": self.mean_squared_error,
            "relative_l2_error": self.relative_l2_error,
            "max_absolute_error": self.max_absolute_error,
            "quantizer_boundary_fraction": self.quantizer_boundary_fraction,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "compression_ratio": self.compression_ratio,
        }


def _stochastic_round(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    lower = torch.floor(values)
    probability_up = values - lower
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)
    draw = torch.rand(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )
    return lower + (draw < probability_up).to(values.dtype)


def quantize_dequantize(tensor: torch.Tensor, spec: QuantizationSpec) -> QuantizationResult:
    """Round-trip ``tensor`` through grouped symmetric integer quantization.

    The last ``spec.flatten_last_dims`` dimensions are flattened into one row.
    All earlier dimensions remain independent rows. For a Gated DeltaNet state
    shaped ``[batch, heads, key_dim, value_dim]``, the default therefore uses
    independent scales within each batch item and head.
    """

    if not tensor.is_floating_point():
        raise TypeError("tensor must use a floating-point dtype")
    if tensor.ndim < spec.flatten_last_dims:
        raise ValueError(
            f"tensor has {tensor.ndim} dimensions, fewer than flatten_last_dims="
            f"{spec.flatten_last_dims}"
        )
    if tensor.numel() == 0:
        raise ValueError("tensor must not be empty")

    original_dtype = tensor.dtype
    working = tensor.detach().to(torch.float32)
    flattened_size = math.prod(working.shape[-spec.flatten_last_dims :])
    rows = working.reshape(-1, flattened_size)

    groups_per_row = math.ceil(flattened_size / spec.group_size)
    padded_size = groups_per_row * spec.group_size
    if padded_size != flattened_size:
        rows = torch.nn.functional.pad(rows, (0, padded_size - flattened_size))
    grouped = rows.reshape(rows.shape[0], groups_per_row, spec.group_size)

    qmax = (1 << (spec.bits - 1)) - 1
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(
        absmax > spec.epsilon,
        absmax / qmax,
        torch.ones_like(absmax),
    )
    normalized = grouped / scales
    if spec.rounding == "nearest":
        quantized = torch.round(normalized)
    else:
        quantized = _stochastic_round(normalized, seed=spec.seed)
    quantized = quantized.clamp(-qmax, qmax)
    dequantized = quantized * scales

    restored = dequantized.reshape(rows.shape[0], padded_size)[:, :flattened_size]
    restored = restored.reshape(working.shape).to(original_dtype)

    error = restored.to(torch.float32) - working
    error_l2 = torch.linalg.vector_norm(error)
    source_l2 = torch.linalg.vector_norm(working)
    relative_l2 = error_l2 / source_l2.clamp_min(spec.epsilon)
    boundary_fraction = (quantized.abs() == qmax).to(torch.float32).mean()

    number_of_groups = grouped.shape[0] * grouped.shape[1]
    payload_bits = tensor.numel() * spec.bits
    scale_bits = number_of_groups * spec.scale_bits
    estimated_bytes = math.ceil((payload_bits + scale_bits) / 8)

    return QuantizationResult(
        tensor=restored,
        spec=spec,
        elements=tensor.numel(),
        groups=number_of_groups,
        payload_bits=payload_bits,
        scale_bits=scale_bits,
        baseline_bytes=tensor.numel() * tensor.element_size(),
        estimated_bytes=estimated_bytes,
        mean_squared_error=float(error.square().mean().item()),
        relative_l2_error=float(relative_l2.item()),
        max_absolute_error=float(error.abs().max().item()),
        quantizer_boundary_fraction=float(boundary_fraction.item()),
        scale_min=float(scales.min().item()),
        scale_max=float(scales.max().item()),
    )
