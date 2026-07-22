"""Grouped quantization and physical INT4/INT8 packing for recurrent states.

``quantize_dequantize`` emulates numerical effects in floating-point storage;
``quantize_pack`` creates integer payload and scale tensors with exact resident
bytes. Neither function alone establishes end-to-end memory or latency gains.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
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
        if self.scale_bits not in (16, 32):
            raise ValueError("scale_bits must be 16 or 32")
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


@dataclass(frozen=True, slots=True)
class PackedQuantizedTensor:
    """Physically packed grouped INT4 or INT8 tensor plus stored scales."""

    payload: torch.Tensor
    scales: torch.Tensor
    spec: QuantizationSpec
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    flattened_size: int
    padded_size: int
    rows: int
    groups_per_row: int

    @property
    def elements(self) -> int:
        return math.prod(self.original_shape)

    @property
    def storage_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size() + (
            self.scales.numel() * self.scales.element_size()
        )

    def dequantize(self) -> torch.Tensor:
        """Materialize the packed tensor in its original dtype and shape."""

        packed_elements = self.rows * self.padded_size
        if self.spec.bits == 8:
            codes = self.payload.to(torch.int8).reshape(-1).to(torch.float32)
        elif self.spec.bits == 4:
            payload = self.payload.reshape(-1)
            low = torch.bitwise_and(payload, 0x0F).to(torch.int16)
            high = torch.bitwise_right_shift(payload, 4).to(torch.int16)
            low = torch.where(low >= 8, low - 16, low)
            high = torch.where(high >= 8, high - 16, high)
            interleaved = torch.empty(
                payload.numel() * 2,
                dtype=torch.int16,
                device=payload.device,
            )
            interleaved[0::2] = low
            interleaved[1::2] = high
            codes = interleaved[:packed_elements].to(torch.float32)
        else:
            raise RuntimeError(f"Packed dequantization does not support INT{self.spec.bits}")

        grouped = codes.reshape(self.rows, self.groups_per_row, self.spec.group_size)
        restored = grouped * self.scales.to(torch.float32).unsqueeze(-1)
        restored = restored.reshape(self.rows, self.padded_size)[:, : self.flattened_size]
        return restored.reshape(self.original_shape).to(self.original_dtype)

    def to(self, device: torch.device | str) -> PackedQuantizedTensor:
        """Move packed payload and scales without materializing the tensor."""

        return PackedQuantizedTensor(
            payload=self.payload.to(device),
            scales=self.scales.to(device),
            spec=self.spec,
            original_shape=self.original_shape,
            original_dtype=self.original_dtype,
            flattened_size=self.flattened_size,
            padded_size=self.padded_size,
            rows=self.rows,
            groups_per_row=self.groups_per_row,
        )


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


def _scale_dtype(scale_bits: int) -> torch.dtype:
    if scale_bits == 16:
        return torch.float16
    if scale_bits == 32:
        return torch.float32
    raise ValueError("scale_bits must be 16 or 32")


def scheduled_quantization_spec(
    spec: QuantizationSpec,
    *,
    layer_index: int,
    layer_update_index: int,
) -> QuantizationSpec:
    """Return the reproducible per-layer stream used for stochastic rounding."""

    if layer_index < 0 or layer_update_index < 0:
        raise ValueError("layer and update indices must be non-negative")
    if spec.rounding == "nearest":
        return spec
    return replace(
        spec,
        seed=spec.seed + layer_index * 1_000_000 + layer_update_index,
    )


def _group_tensor(
    tensor: torch.Tensor,
    spec: QuantizationSpec,
) -> tuple[torch.Tensor, torch.dtype, int, int, int, torch.Tensor]:
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
    if not torch.isfinite(working).all().item():
        raise ValueError("tensor must contain only finite values")
    flattened_size = math.prod(working.shape[-spec.flatten_last_dims :])
    rows = working.reshape(-1, flattened_size)
    groups_per_row = math.ceil(flattened_size / spec.group_size)
    padded_size = groups_per_row * spec.group_size
    if padded_size != flattened_size:
        rows = torch.nn.functional.pad(rows, (0, padded_size - flattened_size))
    grouped = rows.reshape(rows.shape[0], groups_per_row, spec.group_size)
    return working, original_dtype, flattened_size, padded_size, groups_per_row, grouped


def _quantize_groups(
    grouped: torch.Tensor,
    spec: QuantizationSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = (1 << (spec.bits - 1)) - 1
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    ideal_scales = torch.where(
        absmax > spec.epsilon,
        absmax / qmax,
        torch.ones_like(absmax),
    )
    scale_dtype = _scale_dtype(spec.scale_bits)
    if scale_dtype == torch.float16:
        # FP16 scale storage cannot represent arbitrarily small non-zero scales.
        # Clamp to its smallest positive subnormal so division never produces
        # infinities while still emulating the bytes that are actually stored.
        ideal_scales = ideal_scales.clamp(
            min=2.0**-24,
            max=torch.finfo(torch.float16).max,
        )
    stored_scales = ideal_scales.to(scale_dtype)
    working_scales = stored_scales.to(torch.float32)
    normalized = grouped / working_scales
    if spec.rounding == "nearest":
        quantized = torch.round(normalized)
    else:
        quantized = _stochastic_round(normalized, seed=spec.seed)
    return quantized.clamp(-qmax, qmax), stored_scales


def quantize_dequantize(tensor: torch.Tensor, spec: QuantizationSpec) -> QuantizationResult:
    """Round-trip ``tensor`` through grouped symmetric integer quantization.

    The last ``spec.flatten_last_dims`` dimensions are flattened into one row.
    All earlier dimensions remain independent rows. For a Gated DeltaNet state
    shaped ``[batch, heads, key_dim, value_dim]``, the default therefore uses
    independent scales within each batch item and head.
    """

    working, original_dtype, flattened_size, padded_size, groups_per_row, grouped = (
        _group_tensor(tensor, spec)
    )
    quantized, scales = _quantize_groups(grouped, spec)
    dequantized = quantized * scales.to(torch.float32)

    restored = dequantized.reshape(grouped.shape[0], padded_size)[:, :flattened_size]
    restored = restored.reshape(working.shape).to(original_dtype)

    error = restored.to(torch.float32) - working
    error_l2 = torch.linalg.vector_norm(error)
    source_l2 = torch.linalg.vector_norm(working)
    relative_l2 = error_l2 / source_l2.clamp_min(spec.epsilon)
    qmax = (1 << (spec.bits - 1)) - 1
    valid_quantized = quantized.reshape(grouped.shape[0], padded_size)[:, :flattened_size]
    boundary_fraction = (valid_quantized.abs() == qmax).to(torch.float32).mean()

    number_of_groups = grouped.shape[0] * grouped.shape[1]
    payload_bits = number_of_groups * spec.group_size * spec.bits
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


def quantize_pack(tensor: torch.Tensor, spec: QuantizationSpec) -> PackedQuantizedTensor:
    """Quantize and physically pack a tensor using INT4 or INT8 payloads."""

    if spec.bits not in (4, 8):
        raise ValueError("physical packing currently supports only INT4 and INT8")
    working, original_dtype, flattened_size, padded_size, groups_per_row, grouped = (
        _group_tensor(tensor, spec)
    )
    quantized, scales = _quantize_groups(grouped, spec)
    flat_codes = quantized.to(torch.int16).reshape(-1)
    if spec.bits == 8:
        payload = flat_codes.to(torch.int8)
    else:
        nibbles = torch.bitwise_and(flat_codes, 0x0F).to(torch.uint8)
        if nibbles.numel() % 2:
            nibbles = torch.nn.functional.pad(nibbles, (0, 1))
        payload = torch.bitwise_or(
            nibbles[0::2],
            torch.bitwise_left_shift(nibbles[1::2], 4),
        )

    return PackedQuantizedTensor(
        payload=payload.contiguous(),
        scales=scales.squeeze(-1).contiguous(),
        spec=spec,
        original_shape=tuple(working.shape),
        original_dtype=original_dtype,
        flattened_size=flattened_size,
        padded_size=padded_size,
        rows=grouped.shape[0],
        groups_per_row=groups_per_row,
    )
