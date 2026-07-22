"""Physical mixed-group INT4/INT8 packing for recurrent states.

The precision decision is stored as one packed bit per quantization group. Each
group owns one stored scale and either an INT4 or INT8 payload. ``storage_bytes``
counts the resident tensor bytes exactly; Python metadata is not device-resident
model state and is intentionally excluded, matching ``PackedQuantizedTensor``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .quantization import (
    QuantizationSpec,
    _group_tensor,
    _scale_dtype,
    _stochastic_round,
)


def _validate_mixed_specs(
    low_spec: QuantizationSpec,
    high_spec: QuantizationSpec,
) -> None:
    if low_spec.bits != 4 or high_spec.bits != 8:
        raise ValueError("mixed physical packing requires low INT4 and high INT8 specs")

    shared_fields = (
        "group_size",
        "scale_bits",
        "flatten_last_dims",
        "rounding",
        "seed",
        "epsilon",
    )
    mismatched = [
        field for field in shared_fields if getattr(low_spec, field) != getattr(high_spec, field)
    ]
    if mismatched:
        raise ValueError(
            "low_spec and high_spec must differ only in bits; mismatched fields: "
            + ", ".join(mismatched)
        )


def _pack_precision_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    chunks = flat.reshape(-1, 8).to(torch.int16)
    shifts = torch.arange(8, dtype=torch.int16, device=flat.device)
    weights = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    return (chunks * weights).sum(dim=1).to(torch.uint8).contiguous()


def _unpack_precision_mask(packed: torch.Tensor, total_groups: int) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.int16, device=packed.device)
    expanded = torch.bitwise_right_shift(packed.to(torch.int16).unsqueeze(1), shifts)
    return torch.bitwise_and(expanded, 1).reshape(-1)[:total_groups].to(torch.bool)


def _pack_int4_groups(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 codes while preserving a byte boundary per group."""

    nibbles = torch.bitwise_and(codes.to(torch.int16), 0x0F).to(torch.uint8)
    if nibbles.shape[1] % 2:
        nibbles = torch.nn.functional.pad(nibbles, (0, 1))
    return torch.bitwise_or(
        nibbles[:, 0::2],
        torch.bitwise_left_shift(nibbles[:, 1::2], 4),
    ).contiguous()


def _unpack_int4_groups(payload: torch.Tensor, group_size: int) -> torch.Tensor:
    low = torch.bitwise_and(payload, 0x0F).to(torch.int16)
    high = torch.bitwise_right_shift(payload, 4).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    codes = torch.empty(
        (payload.shape[0], payload.shape[1] * 2),
        dtype=torch.int16,
        device=payload.device,
    )
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    return codes[:, :group_size]


def _quantize_mixed_groups(
    grouped: torch.Tensor,
    high_precision_mask: torch.Tensor,
    low_spec: QuantizationSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = torch.where(
        high_precision_mask.unsqueeze(1),
        torch.tensor(127.0, device=grouped.device),
        torch.tensor(7.0, device=grouped.device),
    )
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    ideal_scales = torch.where(
        absmax > low_spec.epsilon,
        absmax / qmax,
        torch.ones_like(absmax),
    )
    scale_dtype = _scale_dtype(low_spec.scale_bits)
    if scale_dtype == torch.float16:
        ideal_scales = ideal_scales.clamp(
            min=2.0**-24,
            max=torch.finfo(torch.float16).max,
        )
    stored_scales = ideal_scales.to(scale_dtype)
    normalized = grouped / stored_scales.to(torch.float32)
    if low_spec.rounding == "nearest":
        quantized = torch.round(normalized)
    else:
        quantized = _stochastic_round(normalized, seed=low_spec.seed)
    quantized = torch.minimum(torch.maximum(quantized, -qmax), qmax)
    return quantized.to(torch.int16), stored_scales.squeeze(1).contiguous()


def _validate_mask_shape(
    mask: torch.Tensor,
    *,
    rows: int,
    groups_per_row: int,
) -> None:
    if mask.dtype != torch.bool:
        raise TypeError("high_precision_mask must use torch.bool")
    expected_flat = (rows * groups_per_row,)
    expected_grouped = (rows, groups_per_row)
    if tuple(mask.shape) not in (expected_flat, expected_grouped):
        raise ValueError(
            "high_precision_mask must have shape "
            f"{expected_flat} or {expected_grouped}, got {tuple(mask.shape)}"
        )


@dataclass(frozen=True, slots=True)
class PackedMixedQuantizedTensor:
    """A tensor with physically separate INT4 and INT8 group payloads."""

    low_payload: torch.Tensor
    high_payload: torch.Tensor
    scales: torch.Tensor
    precision_mask: torch.Tensor
    low_spec: QuantizationSpec
    high_spec: QuantizationSpec
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
    def total_groups(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def low_precision_groups(self) -> int:
        return self.low_payload.shape[0]

    @property
    def high_precision_groups(self) -> int:
        return self.high_payload.shape[0]

    @property
    def payload_bytes(self) -> int:
        return self.low_payload.numel() * self.low_payload.element_size() + (
            self.high_payload.numel() * self.high_payload.element_size()
        )

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    @property
    def mask_bytes(self) -> int:
        return self.precision_mask.numel() * self.precision_mask.element_size()

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes + self.mask_bytes

    def high_precision_mask(self) -> torch.Tensor:
        """Return the unpacked mask in ``[rows, groups_per_row]`` order."""

        return _unpack_precision_mask(self.precision_mask, self.total_groups).reshape(
            self.rows,
            self.groups_per_row,
        )

    def _integer_groups(self) -> torch.Tensor:
        mask = self.high_precision_mask().reshape(-1)
        codes = torch.empty(
            (self.total_groups, self.low_spec.group_size),
            dtype=torch.int16,
            device=self.scales.device,
        )
        codes[~mask] = _unpack_int4_groups(
            self.low_payload,
            self.low_spec.group_size,
        )
        codes[mask] = self.high_payload.to(torch.int16)
        return codes

    def dequantize(self) -> torch.Tensor:
        """Materialize the mixed packed tensor in its original dtype and shape."""

        restored_groups = self._integer_groups().to(torch.float32) * self.scales.to(
            torch.float32
        ).unsqueeze(1)
        restored = restored_groups.reshape(self.rows, self.padded_size)[:, : self.flattened_size]
        return restored.reshape(self.original_shape).to(self.original_dtype)

    def to(self, device: torch.device | str) -> PackedMixedQuantizedTensor:
        """Move all resident tensors without materializing or requantizing."""

        return PackedMixedQuantizedTensor(
            low_payload=self.low_payload.to(device),
            high_payload=self.high_payload.to(device),
            scales=self.scales.to(device),
            precision_mask=self.precision_mask.to(device),
            low_spec=self.low_spec,
            high_spec=self.high_spec,
            original_shape=self.original_shape,
            original_dtype=self.original_dtype,
            flattened_size=self.flattened_size,
            padded_size=self.padded_size,
            rows=self.rows,
            groups_per_row=self.groups_per_row,
        )

    def reorder_batch(self, beam_idx: torch.LongTensor) -> PackedMixedQuantizedTensor:
        """Select packed batch entries without another quantization round trip."""

        if len(self.original_shape) <= self.low_spec.flatten_last_dims:
            raise RuntimeError(
                "Cannot reorder mixed groups when quantization includes the batch dimension"
            )
        if beam_idx.ndim != 1 or beam_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("beam_idx must be a one-dimensional int32 or int64 tensor")

        batch_size = self.original_shape[0]
        if self.rows % batch_size:
            raise RuntimeError("mixed row metadata is inconsistent with its batch dimension")
        if beam_idx.numel() and (beam_idx.min().item() < 0 or beam_idx.max().item() >= batch_size):
            raise IndexError("beam_idx contains an out-of-range batch index")

        groups_per_batch = self.total_groups // batch_size
        selected = beam_idx.to(self.scales.device)
        selected_batch_size = selected.numel()
        codes = self._integer_groups().reshape(
            batch_size,
            groups_per_batch,
            self.low_spec.group_size,
        )
        scales = self.scales.reshape(batch_size, groups_per_batch)
        mask = self.high_precision_mask().reshape(batch_size, groups_per_batch)

        reordered_codes = codes.index_select(0, selected).reshape(
            selected_batch_size * groups_per_batch,
            self.low_spec.group_size,
        )
        reordered_scales = scales.index_select(0, selected).reshape(-1).contiguous()
        reordered_mask = mask.index_select(0, selected).reshape(-1)
        return _from_integer_groups(
            reordered_codes,
            reordered_scales,
            reordered_mask,
            low_spec=self.low_spec,
            high_spec=self.high_spec,
            original_shape=(selected_batch_size, *self.original_shape[1:]),
            original_dtype=self.original_dtype,
            flattened_size=self.flattened_size,
            padded_size=self.padded_size,
            rows=selected_batch_size * (self.rows // batch_size),
            groups_per_row=self.groups_per_row,
        )


def _from_integer_groups(
    codes: torch.Tensor,
    scales: torch.Tensor,
    high_precision_mask: torch.Tensor,
    *,
    low_spec: QuantizationSpec,
    high_spec: QuantizationSpec,
    original_shape: tuple[int, ...],
    original_dtype: torch.dtype,
    flattened_size: int,
    padded_size: int,
    rows: int,
    groups_per_row: int,
) -> PackedMixedQuantizedTensor:
    flat_mask = high_precision_mask.reshape(-1)
    return PackedMixedQuantizedTensor(
        low_payload=_pack_int4_groups(codes[~flat_mask]),
        high_payload=codes[flat_mask].to(torch.int8).contiguous(),
        scales=scales.contiguous(),
        precision_mask=_pack_precision_mask(flat_mask),
        low_spec=low_spec,
        high_spec=high_spec,
        original_shape=original_shape,
        original_dtype=original_dtype,
        flattened_size=flattened_size,
        padded_size=padded_size,
        rows=rows,
        groups_per_row=groups_per_row,
    )


def quantize_pack_mixed(
    tensor: torch.Tensor,
    high_precision_mask: torch.Tensor,
    *,
    low_spec: QuantizationSpec,
    high_spec: QuantizationSpec,
) -> PackedMixedQuantizedTensor:
    """Quantize groups into physically separate INT4 and INT8 payload pools.

    ``high_precision_mask`` may be flat or shaped ``[rows, groups_per_row]`` in
    the exact group order produced by the shared grouped quantizer.
    """

    _validate_mixed_specs(low_spec, high_spec)
    working, original_dtype, flattened_size, padded_size, groups_per_row, grouped = _group_tensor(
        tensor, low_spec
    )
    rows = grouped.shape[0]
    _validate_mask_shape(
        high_precision_mask,
        rows=rows,
        groups_per_row=groups_per_row,
    )
    flat_mask = high_precision_mask.detach().to(grouped.device).reshape(-1)
    flat_groups = grouped.reshape(-1, low_spec.group_size)
    codes, scales = _quantize_mixed_groups(flat_groups, flat_mask, low_spec)
    return _from_integer_groups(
        codes,
        scales,
        flat_mask,
        low_spec=low_spec,
        high_spec=high_spec,
        original_shape=tuple(working.shape),
        original_dtype=original_dtype,
        flattened_size=flattened_size,
        padded_size=padded_size,
        rows=rows,
        groups_per_row=groups_per_row,
    )
