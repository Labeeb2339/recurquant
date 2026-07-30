"""Equal-resident-byte, no-replay baselines for StateLease-H5.

This module implements the three frozen Experiment 010 comparators that spend
StateLease's complete resident allocation on the current recurrent state:

* expanded right-RHT Q4/Q8;
* right-RHT Q4/Q6/Q8; and
* right-RHT base-Q4 plus a selected residual-Q4 code.

The default layout is the pinned Qwen3.5-0.8B recurrent geometry.  Every
returned checkpoint owns physical integer payloads, FP16 scales, packed
precision metadata, the FP32 causal query-energy EMA, and an explicit padding
tensor.  It never retains an FP32 recurrent-state mirror or replay record.

The implementation is a correctness-first reference.  Selection and
materialization use transient FP32 workspaces.  The complete Q4/Q6/Q8 policy
uses the exact ``O(N log N)`` CPU structural allocator, but physical packing
and dequantization remain unfused and make no latency or peak-memory claim.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Literal, TypeAlias

import torch

from .mixed_quantization import PackedMixedQuantizedTensor, quantize_pack_mixed
from .multibit_policy import allocate_exact_multibit_codes_fast
from .multibit_quantization import PackedMultiBitQuantizedTensor, quantize_pack_multibit
from .quantization import (
    PackedQuantizedTensor,
    QuantizationSpec,
    quantize_dequantize,
    quantize_pack,
)
from .rht import RHT_SEED, right_rht_decode, right_rht_encode

EXPANDED_RHT_Q4_Q8 = "expanded_rht_q4_q8"
RHT_Q4_Q6_Q8 = "rht_q4_q6_q8"
RHT_RESIDUAL_Q4 = "rht_residual_q4"
EqualByteCodecName: TypeAlias = Literal[
    "expanded_rht_q4_q8",
    "rht_q4_q6_q8",
    "rht_residual_q4",
]

FROZEN_STATELEASE_RESIDENT_BYTES = 3_454_664
FROZEN_EXPANDED_Q8_PROMOTIONS = 13_587
FROZEN_MULTIBIT_MARGINAL_STEPS = 27_030
FROZEN_RESIDUAL_Q4_ROWS = 13_175
FROZEN_EXPANDED_PADDING_BYTES = 8
FROZEN_MULTIBIT_PADDING_BYTES = 8
FROZEN_RESIDUAL_PADDING_BYTES = 26
FROZEN_RECURRENT_LAYER_INDICES = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
)

QUERY_EMA_DECAY = 2.0 ** (-1.0 / 32.0)
QUERY_L2NORM_EPS = 1e-6
_SUPPORTED_CODECS = frozenset((EXPANDED_RHT_Q4_Q8, RHT_Q4_Q6_Q8, RHT_RESIDUAL_Q4))


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class EqualByteLayout:
    """Geometry and exact physical budgets for all three comparator formats."""

    layer_indices: tuple[int, ...]
    heads: int
    key_rows: int
    value_width: int
    expanded_q8_promotions: int
    multibit_marginal_steps: int
    residual_q4_rows: int
    expanded_padding_bytes: int
    multibit_padding_bytes: int
    residual_padding_bytes: int
    expected_resident_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.layer_indices, tuple) or not self.layer_indices:
            raise ValueError("layer_indices must be a non-empty tuple")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.layer_indices
        ):
            raise ValueError("layer_indices must contain non-negative integers")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be unique")
        _positive_int(self.heads, name="heads")
        _positive_int(self.key_rows, name="key_rows")
        width = _positive_int(self.value_width, name="value_width")
        if width & (width - 1):
            raise ValueError("value_width must be a power of two for the frozen RHT")
        if width % 4:
            raise ValueError("value_width must make Q4 and Q6 payloads byte aligned")

        rows = self.total_rows
        promotions = _nonnegative_int(
            self.expanded_q8_promotions,
            name="expanded_q8_promotions",
        )
        steps = _nonnegative_int(
            self.multibit_marginal_steps,
            name="multibit_marginal_steps",
        )
        residual_rows = _nonnegative_int(
            self.residual_q4_rows,
            name="residual_q4_rows",
        )
        if promotions > rows:
            raise ValueError("expanded_q8_promotions exceeds the number of state rows")
        if steps > 2 * rows:
            raise ValueError("multibit_marginal_steps exceeds two steps per state row")
        if residual_rows > rows:
            raise ValueError("residual_q4_rows exceeds the number of state rows")
        for name in (
            "expanded_padding_bytes",
            "multibit_padding_bytes",
            "residual_padding_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        expected = _positive_int(
            self.expected_resident_bytes,
            name="expected_resident_bytes",
        )
        totals = {
            EXPANDED_RHT_Q4_Q8: self.expected_bytes(EXPANDED_RHT_Q4_Q8),
            RHT_Q4_Q6_Q8: self.expected_bytes(RHT_Q4_Q6_Q8),
            RHT_RESIDUAL_Q4: self.expected_bytes(RHT_RESIDUAL_Q4),
        }
        mismatched = {name: value for name, value in totals.items() if value != expected}
        if mismatched:
            rendered = ", ".join(f"{name}={value}" for name, value in mismatched.items())
            raise ValueError(
                f"all comparator formats must equal expected_resident_bytes={expected}; "
                f"mismatched: {rendered}"
            )

    @property
    def layers(self) -> int:
        return len(self.layer_indices)

    @property
    def rows_per_layer(self) -> int:
        return self.heads * self.key_rows

    @property
    def total_rows(self) -> int:
        return self.layers * self.rows_per_layer

    @property
    def state_elements(self) -> int:
        return self.total_rows * self.value_width

    @property
    def fp32_state_bytes(self) -> int:
        return self.state_elements * 4

    @property
    def q4_payload_bytes(self) -> int:
        return self.total_rows * self.value_width * 4 // 8

    @property
    def scale_bytes(self) -> int:
        return self.total_rows * 2

    @property
    def query_ema_bytes(self) -> int:
        return self.total_rows * 4

    @property
    def one_bit_code_bytes(self) -> int:
        return self.layers * math.ceil(self.rows_per_layer / 8)

    @property
    def two_bit_code_bytes(self) -> int:
        return self.layers * math.ceil(2 * self.rows_per_layer / 8)

    def expected_bytes(self, codec: EqualByteCodecName) -> int:
        if codec == EXPANDED_RHT_Q4_Q8:
            return (
                self.q4_payload_bytes
                + self.scale_bytes
                + self.one_bit_code_bytes
                + self.query_ema_bytes
                + self.expanded_q8_promotions * self.value_width * 4 // 8
                + self.expanded_padding_bytes
            )
        if codec == RHT_Q4_Q6_Q8:
            return (
                self.q4_payload_bytes
                + self.scale_bytes
                + self.two_bit_code_bytes
                + self.query_ema_bytes
                + self.multibit_marginal_steps * self.value_width * 2 // 8
                + self.multibit_padding_bytes
            )
        if codec == RHT_RESIDUAL_Q4:
            return (
                self.q4_payload_bytes
                + self.scale_bytes
                + self.one_bit_code_bytes
                + self.query_ema_bytes
                + self.residual_q4_rows * (self.value_width * 4 // 8 + 2)
                + self.residual_padding_bytes
            )
        raise ValueError(f"unsupported equal-byte codec: {codec!r}")

    def component_bytes(self, codec: EqualByteCodecName) -> dict[str, int]:
        """Return the frozen arithmetic without allocating a checkpoint."""

        if codec == EXPANDED_RHT_Q4_Q8:
            payload = (
                self.q4_payload_bytes + self.expanded_q8_promotions * self.value_width * 4 // 8
            )
            precision = self.one_bit_code_bytes
            padding = self.expanded_padding_bytes
        elif codec == RHT_Q4_Q6_Q8:
            payload = (
                self.q4_payload_bytes + self.multibit_marginal_steps * self.value_width * 2 // 8
            )
            precision = self.two_bit_code_bytes
            padding = self.multibit_padding_bytes
        elif codec == RHT_RESIDUAL_Q4:
            payload = self.q4_payload_bytes + self.residual_q4_rows * self.value_width * 4 // 8
            precision = self.one_bit_code_bytes
            padding = self.residual_padding_bytes
        else:
            raise ValueError(f"unsupported equal-byte codec: {codec!r}")
        scale = self.scale_bytes + (self.residual_q4_rows * 2 if codec == RHT_RESIDUAL_Q4 else 0)
        components = {
            "payload_bytes": payload,
            "scale_bytes": scale,
            "precision_bytes": precision,
            "query_ema_bytes": self.query_ema_bytes,
            "padding_bytes": padding,
        }
        components["resident_bytes"] = sum(components.values())
        return components


FROZEN_QWEN35_EQUAL_BYTE_LAYOUT = EqualByteLayout(
    layer_indices=FROZEN_RECURRENT_LAYER_INDICES,
    heads=16,
    key_rows=128,
    value_width=128,
    expanded_q8_promotions=FROZEN_EXPANDED_Q8_PROMOTIONS,
    multibit_marginal_steps=FROZEN_MULTIBIT_MARGINAL_STEPS,
    residual_q4_rows=FROZEN_RESIDUAL_Q4_ROWS,
    expanded_padding_bytes=FROZEN_EXPANDED_PADDING_BYTES,
    multibit_padding_bytes=FROZEN_MULTIBIT_PADDING_BYTES,
    residual_padding_bytes=FROZEN_RESIDUAL_PADDING_BYTES,
    expected_resident_bytes=FROZEN_STATELEASE_RESIDENT_BYTES,
)


def frozen_equal_byte_accounting() -> dict[str, dict[str, int]]:
    """Return independently checkable component arithmetic for the frozen layout."""

    layout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT
    return {
        codec: layout.component_bytes(codec)
        for codec in (EXPANDED_RHT_Q4_Q8, RHT_Q4_Q6_Q8, RHT_RESIDUAL_Q4)
    }


def _storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _validate_owned_tensors(tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    owners: dict[tuple[str, int], str] = {}
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        logical = tensor.numel() * tensor.element_size()
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.storage_offset() != 0:
            raise ValueError(f"{name} must have zero storage offset")
        if _storage_bytes(tensor) != logical:
            raise ValueError(
                f"{name} owns {_storage_bytes(tensor)} bytes but exposes {logical} bytes"
            )
        if tensor.numel() == 0:
            continue
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        previous = owners.get(identity)
        if previous is not None:
            raise ValueError(f"{name} aliases persistent tensor {previous}")
        owners[identity] = name


def _validate_padding(padding: torch.Tensor, *, expected_bytes: int) -> None:
    if padding.dtype != torch.uint8 or tuple(padding.shape) != (expected_bytes,):
        raise TypeError(
            f"padding must be a one-dimensional torch.uint8 tensor of length {expected_bytes}"
        )
    if padding.numel() and padding.any().item():
        raise ValueError("reserved padding bytes must be zero")


def _validate_query_ema(query_ema: torch.Tensor, layout: EqualByteLayout) -> None:
    expected = (layout.layers, layout.heads, layout.key_rows)
    if query_ema.dtype != torch.float32 or tuple(query_ema.shape) != expected:
        raise TypeError(f"query_energy_ema must have shape {expected} and dtype torch.float32")
    if not torch.isfinite(query_ema).all().item():
        raise ValueError("query_energy_ema must contain only finite values")
    if (query_ema < 0).any().item():
        raise ValueError("query_energy_ema must be non-negative")
    if (query_ema.sum(dim=-1) <= 0).any().item():
        raise ValueError("query_energy_ema must have positive mass for every head")


def _validate_packed_bit_padding(
    packed: torch.Tensor,
    *,
    total_codes: int,
    bits_per_code: int,
) -> None:
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise TypeError("packed precision metadata must be one-dimensional torch.uint8")
    expected = math.ceil(total_codes * bits_per_code / 8)
    if packed.numel() != expected:
        raise ValueError(
            f"packed precision metadata must contain {expected} bytes, got {packed.numel()}"
        )
    used_bits = (total_codes * bits_per_code) % 8
    if used_bits and packed.numel():
        unused_mask = 0xFF ^ ((1 << used_bits) - 1)
        if int(packed[-1].item()) & unused_mask:
            raise ValueError("unused precision padding bits must be zero")


def _validate_common_specs(
    specs: tuple[QuantizationSpec, ...],
    *,
    expected_bits: tuple[int, ...],
    layout: EqualByteLayout,
) -> None:
    if len(specs) != len(expected_bits):
        raise RuntimeError("quantization spec validation received inconsistent widths")
    for bit_width, spec in zip(expected_bits, specs, strict=True):
        if spec.bits != bit_width:
            raise ValueError(f"packed INT{bit_width} metadata has the wrong bit width")
        if (
            spec.group_size != layout.value_width
            or spec.scale_bits != 16
            or spec.flatten_last_dims != 2
            or spec.rounding != "nearest"
            or spec.seed != RHT_SEED
        ):
            raise ValueError("packed quantization metadata does not match the frozen codec")


def _validate_q4_payload(payload: torch.Tensor, *, expected_groups: int, width: int) -> None:
    expected_shape = (expected_groups, width * 4 // 8)
    if payload.dtype != torch.uint8 or tuple(payload.shape) != expected_shape:
        raise TypeError(f"Q4 payload must have shape {expected_shape} and dtype torch.uint8")
    if payload.numel():
        low = torch.bitwise_and(payload, 0x0F)
        high = torch.bitwise_right_shift(payload, 4)
        if ((low == 8) | (high == 8)).any().item():
            raise ValueError("Q4 payload contains the reserved symmetric code -8")


@dataclass(frozen=True, slots=True)
class ExpandedRhtQ4Q8Layer:
    layer_index: int
    packed: PackedMixedQuantizedTensor

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("q4_payload", self.packed.low_payload),
            ("q8_payload", self.packed.high_payload),
            ("scales", self.packed.scales),
            ("precision_mask", self.packed.precision_mask),
        )

    def materialize(self) -> torch.Tensor:
        return self.packed.dequantize()

    def to(self, device: torch.device | str) -> ExpandedRhtQ4Q8Layer:
        return ExpandedRhtQ4Q8Layer(self.layer_index, self.packed.to(device))


@dataclass(frozen=True, slots=True)
class RhtQ4Q6Q8Layer:
    layer_index: int
    packed: PackedMultiBitQuantizedTensor

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("q4_payload", self.packed.int4_payload),
            ("q6_payload", self.packed.int6_payload),
            ("q8_payload", self.packed.int8_payload),
            ("scales", self.packed.scales),
            ("precision_codes", self.packed.packed_precision_codes),
        )

    def materialize(self, *, heads: int) -> torch.Tensor:
        encoded = self.packed.dequantize()
        return right_rht_decode(
            encoded,
            layer_index=self.layer_index,
            expected_heads=heads,
            output_dtype=torch.float32,
        )

    def to(self, device: torch.device | str) -> RhtQ4Q6Q8Layer:
        return RhtQ4Q6Q8Layer(self.layer_index, self.packed.to(device))


@dataclass(frozen=True, slots=True)
class RhtResidualQ4Layer:
    layer_index: int
    base: PackedQuantizedTensor
    residual: PackedQuantizedTensor
    lease_mask: torch.Tensor

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("base_q4_payload", self.base.payload),
            ("base_scales", self.base.scales),
            ("residual_q4_payload", self.residual.payload),
            ("residual_scales", self.residual.scales),
            ("lease_mask", self.lease_mask),
        )

    def materialize(self, *, heads: int, key_rows: int, value_width: int) -> torch.Tensor:
        encoded = self.base.dequantize().to(torch.float32)
        mask = _unpack_bool_mask(self.lease_mask, heads * key_rows)
        if mask.any().item():
            flat = encoded.reshape(heads * key_rows, value_width)
            flat[mask] += self.residual.dequantize().to(torch.float32)
        return right_rht_decode(
            encoded,
            layer_index=self.layer_index,
            expected_heads=heads,
            output_dtype=torch.float32,
        )

    def to(self, device: torch.device | str) -> RhtResidualQ4Layer:
        return RhtResidualQ4Layer(
            self.layer_index,
            self.base.to(device),
            self.residual.to(device),
            self.lease_mask.to(device),
        )


EqualByteLayer: TypeAlias = ExpandedRhtQ4Q8Layer | RhtQ4Q6Q8Layer | RhtResidualQ4Layer


@dataclass(frozen=True, slots=True)
class EqualByteCodecEvidence:
    """Scalar and hash evidence for one successfully built checkpoint."""

    codec: str
    state_elements: int
    fp32_state_bytes: int
    payload_bytes: int
    scale_bytes: int
    precision_bytes: int
    query_ema_bytes: int
    padding_bytes: int
    resident_bytes: int
    selected_units: int
    expected_selected_units: int
    selection_sha256: str
    mean_squared_error: float
    relative_l2_error: float
    max_absolute_error: float

    @property
    def compression_ratio(self) -> float:
        return self.fp32_state_bytes / self.resident_bytes

    def evidence_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["compression_ratio"] = self.compression_ratio
        return result


def _pack_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool or mask.ndim != 1:
        raise TypeError("mask must be a one-dimensional torch.bool tensor")
    flat = mask.to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    chunks = flat.reshape(-1, 8).to(torch.int16)
    shifts = torch.arange(8, dtype=torch.int16, device=flat.device)
    return (
        (chunks * torch.bitwise_left_shift(torch.ones_like(shifts), shifts))
        .sum(dim=1)
        .to(torch.uint8)
        .contiguous()
    )


def _unpack_bool_mask(packed: torch.Tensor, total: int) -> torch.Tensor:
    _validate_packed_bit_padding(packed, total_codes=total, bits_per_code=1)
    shifts = torch.arange(8, dtype=torch.int16, device=packed.device)
    expanded = torch.bitwise_right_shift(packed.to(torch.int16).unsqueeze(1), shifts)
    return torch.bitwise_and(expanded, 1).reshape(-1)[:total].to(torch.bool)


def _validate_expanded_layer(
    layer: ExpandedRhtQ4Q8Layer,
    *,
    expected_index: int,
    layout: EqualByteLayout,
) -> None:
    if layer.layer_index != expected_index:
        raise ValueError("expanded Q4/Q8 layer order does not match the layout")
    packed = layer.packed
    _validate_common_specs(
        (packed.low_spec, packed.high_spec),
        expected_bits=(4, 8),
        layout=layout,
    )
    expected_shape = (1, layout.heads, layout.key_rows, layout.value_width)
    if (
        packed.original_shape != expected_shape
        or packed.original_dtype != torch.float32
        or packed.rows != layout.heads
        or packed.groups_per_row != layout.key_rows
        or packed.flattened_size != layout.key_rows * layout.value_width
        or packed.padded_size != packed.flattened_size
        or packed.right_rht_layer_index != expected_index
        or packed.right_rht_expected_heads != layout.heads
    ):
        raise ValueError("expanded Q4/Q8 packed metadata does not match the layout")
    total = layout.rows_per_layer
    mask = packed.high_precision_mask().reshape(-1)
    _validate_packed_bit_padding(
        packed.precision_mask,
        total_codes=total,
        bits_per_code=1,
    )
    high = int(mask.sum().item())
    _validate_q4_payload(
        packed.low_payload,
        expected_groups=total - high,
        width=layout.value_width,
    )
    expected_high_shape = (high, layout.value_width)
    if packed.high_payload.dtype != torch.int8 or tuple(packed.high_payload.shape) != (
        expected_high_shape
    ):
        raise TypeError(f"Q8 payload must have shape {expected_high_shape} and dtype torch.int8")
    if packed.high_payload.numel() and (packed.high_payload == -128).any().item():
        raise ValueError("Q8 payload contains the reserved symmetric code -128")
    if packed.scales.dtype != torch.float16 or packed.scales.numel() != total:
        raise TypeError("expanded Q4/Q8 scales must contain one FP16 value per row")
    if not torch.isfinite(packed.scales).all().item() or (packed.scales <= 0).any().item():
        raise ValueError("expanded Q4/Q8 scales must be finite and positive")


def _validate_multibit_layer(
    layer: RhtQ4Q6Q8Layer,
    *,
    expected_index: int,
    layout: EqualByteLayout,
) -> None:
    if layer.layer_index != expected_index:
        raise ValueError("Q4/Q6/Q8 layer order does not match the layout")
    packed = layer.packed
    _validate_common_specs(
        (packed.int4_spec, packed.int6_spec, packed.int8_spec),
        expected_bits=(4, 6, 8),
        layout=layout,
    )
    expected_shape = (1, layout.heads, layout.key_rows, layout.value_width)
    if (
        packed.original_shape != expected_shape
        or packed.original_dtype != torch.float32
        or packed.rows != layout.heads
        or packed.groups_per_row != layout.key_rows
        or packed.flattened_size != layout.key_rows * layout.value_width
        or packed.padded_size != packed.flattened_size
    ):
        raise ValueError("Q4/Q6/Q8 packed metadata does not match the layout")
    packed.precision_codes()
    packed._integer_groups()
    _validate_q4_payload(
        packed.int4_payload,
        expected_groups=packed.int4_groups,
        width=layout.value_width,
    )
    if not torch.isfinite(packed.scales).all().item() or (packed.scales <= 0).any().item():
        raise ValueError("Q4/Q6/Q8 scales must be finite and positive")
    if packed.int8_payload.numel() and (packed.int8_payload == -128).any().item():
        raise ValueError("Q8 payload contains the reserved symmetric code -128")


def _validate_residual_packed(
    layer: RhtResidualQ4Layer,
    *,
    expected_index: int,
    layout: EqualByteLayout,
) -> None:
    if layer.layer_index != expected_index:
        raise ValueError("residual-Q4 layer order does not match the layout")
    expected_shape = (1, layout.heads, layout.key_rows, layout.value_width)
    base = layer.base
    _validate_common_specs((base.spec,), expected_bits=(4,), layout=layout)
    if (
        base.original_shape != expected_shape
        or base.original_dtype != torch.float32
        or base.rows != layout.heads
        or base.groups_per_row != layout.key_rows
        or base.flattened_size != layout.key_rows * layout.value_width
        or base.padded_size != base.flattened_size
    ):
        raise ValueError("residual-Q4 base metadata does not match the layout")
    total = layout.rows_per_layer
    expected_base_payload = total * layout.value_width * 4 // 8
    if base.payload.dtype != torch.uint8 or tuple(base.payload.shape) != (expected_base_payload,):
        raise TypeError("residual-Q4 base payload has the wrong dtype or size")
    if base.scales.dtype != torch.float16 or tuple(base.scales.shape) != (
        layout.heads,
        layout.key_rows,
    ):
        raise TypeError("residual-Q4 base scales must contain one FP16 value per row")
    _validate_packed_bit_padding(layer.lease_mask, total_codes=total, bits_per_code=1)
    mask = _unpack_bool_mask(layer.lease_mask, total)
    selected = int(mask.sum().item())

    residual = layer.residual
    expected_spec = replace(base.spec, flatten_last_dims=1)
    if residual.spec != expected_spec:
        raise ValueError("residual-Q4 second-code spec does not match the base spec")
    if (
        residual.original_shape != (selected, layout.value_width)
        or residual.original_dtype != torch.float32
        or residual.rows != selected
        or residual.groups_per_row != 1
        or residual.flattened_size != layout.value_width
        or residual.padded_size != layout.value_width
    ):
        raise ValueError("residual-Q4 second-code metadata does not match its lease mask")
    expected_residual_payload = selected * layout.value_width * 4 // 8
    if residual.payload.dtype != torch.uint8 or tuple(residual.payload.shape) != (
        expected_residual_payload,
    ):
        raise TypeError("residual-Q4 second payload has the wrong dtype or size")
    if residual.scales.dtype != torch.float16 or tuple(residual.scales.shape) != (
        selected,
        1,
    ):
        raise TypeError("residual-Q4 second scales must contain one FP16 value per leased row")
    for name, tensor in (
        ("base scales", base.scales),
        ("residual scales", residual.scales),
    ):
        if tensor.numel() and (
            not torch.isfinite(tensor).all().item() or (tensor <= 0).any().item()
        ):
            raise ValueError(f"{name} must be finite and positive")
    if base.payload.numel():
        flat_base = base.payload.reshape(total, layout.value_width * 4 // 8)
        _validate_q4_payload(flat_base, expected_groups=total, width=layout.value_width)
    if residual.payload.numel():
        flat_residual = residual.payload.reshape(
            selected,
            layout.value_width * 4 // 8,
        )
        _validate_q4_payload(
            flat_residual,
            expected_groups=selected,
            width=layout.value_width,
        )


def _layer_selection_bytes(layer: EqualByteLayer) -> torch.Tensor:
    if isinstance(layer, ExpandedRhtQ4Q8Layer):
        return layer.packed.precision_mask
    if isinstance(layer, RhtQ4Q6Q8Layer):
        return layer.packed.packed_precision_codes
    return layer.lease_mask


def _selection_sha256(layers: tuple[EqualByteLayer, ...]) -> str:
    digest = hashlib.sha256(b"recurquant.statelease.equal-byte.selection.v1\0")
    for layer in layers:
        digest.update(layer.layer_index.to_bytes(4, "little", signed=False))
        digest.update(bytes(_layer_selection_bytes(layer).detach().cpu().tolist()))
    return digest.hexdigest()


def _selected_units(codec: EqualByteCodecName, layers: tuple[EqualByteLayer, ...]) -> int:
    if codec == EXPANDED_RHT_Q4_Q8:
        return sum(
            layer.packed.high_precision_groups
            for layer in layers
            if isinstance(layer, ExpandedRhtQ4Q8Layer)
        )
    if codec == RHT_Q4_Q6_Q8:
        return sum(
            int(layer.packed.precision_codes().to(torch.int64).sum().item())
            for layer in layers
            if isinstance(layer, RhtQ4Q6Q8Layer)
        )
    return sum(
        int(
            _unpack_bool_mask(
                layer.lease_mask,
                layer.base.rows * layer.base.groups_per_row,
            )
            .sum()
            .item()
        )
        for layer in layers
        if isinstance(layer, RhtResidualQ4Layer)
    )


def _component_bytes(
    codec: EqualByteCodecName,
    layers: tuple[EqualByteLayer, ...],
) -> tuple[int, int, int]:
    payload = scale = precision = 0
    for layer in layers:
        if codec == EXPANDED_RHT_Q4_Q8:
            assert isinstance(layer, ExpandedRhtQ4Q8Layer)
            payload += layer.packed.payload_bytes
            scale += layer.packed.scale_bytes
            precision += layer.packed.mask_bytes
        elif codec == RHT_Q4_Q6_Q8:
            assert isinstance(layer, RhtQ4Q6Q8Layer)
            payload += layer.packed.payload_bytes
            scale += layer.packed.scale_bytes
            precision += layer.packed.precision_code_bytes
        else:
            assert isinstance(layer, RhtResidualQ4Layer)
            payload += (
                layer.base.payload.numel() * layer.base.payload.element_size()
                + layer.residual.payload.numel() * layer.residual.payload.element_size()
            )
            scale += (
                layer.base.scales.numel() * layer.base.scales.element_size()
                + layer.residual.scales.numel() * layer.residual.scales.element_size()
            )
            precision += layer.lease_mask.numel() * layer.lease_mask.element_size()
    return payload, scale, precision


@dataclass(frozen=True, slots=True)
class EqualByteCheckpoint:
    """One physically packed complete-state checkpoint with no replay storage."""

    codec: EqualByteCodecName
    layers: tuple[EqualByteLayer, ...]
    query_energy_ema: torch.Tensor
    padding: torch.Tensor
    layout: EqualByteLayout
    evidence: EqualByteCodecEvidence

    def __post_init__(self) -> None:
        self.validate()

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        tensors: list[tuple[str, torch.Tensor]] = []
        for position, layer in enumerate(self.layers):
            tensors.extend(
                (f"layer_{position}.{name}", tensor) for name, tensor in layer.persistent_tensors()
            )
        tensors.extend(
            (
                ("query_energy_ema", self.query_energy_ema),
                ("reserved_padding", self.padding),
            )
        )
        return tuple(tensors)

    @property
    def resident_bytes(self) -> int:
        return sum(_storage_bytes(tensor) for _, tensor in self.persistent_tensors())

    def validate(self) -> None:
        if self.codec not in _SUPPORTED_CODECS:
            raise ValueError(f"unsupported equal-byte codec: {self.codec!r}")
        if not isinstance(self.layers, tuple) or len(self.layers) != self.layout.layers:
            raise ValueError("checkpoint must contain exactly one packed object per layer")
        expected_type: type[EqualByteLayer]
        padding_bytes: int
        expected_selected: int
        if self.codec == EXPANDED_RHT_Q4_Q8:
            expected_type = ExpandedRhtQ4Q8Layer
            padding_bytes = self.layout.expanded_padding_bytes
            expected_selected = self.layout.expanded_q8_promotions
        elif self.codec == RHT_Q4_Q6_Q8:
            expected_type = RhtQ4Q6Q8Layer
            padding_bytes = self.layout.multibit_padding_bytes
            expected_selected = self.layout.multibit_marginal_steps
        else:
            expected_type = RhtResidualQ4Layer
            padding_bytes = self.layout.residual_padding_bytes
            expected_selected = self.layout.residual_q4_rows
        if any(type(layer) is not expected_type for layer in self.layers):
            raise TypeError(f"{self.codec} checkpoint contains a wrong packed-layer type")

        for expected_index, layer in zip(
            self.layout.layer_indices,
            self.layers,
            strict=True,
        ):
            if isinstance(layer, ExpandedRhtQ4Q8Layer):
                _validate_expanded_layer(
                    layer,
                    expected_index=expected_index,
                    layout=self.layout,
                )
            elif isinstance(layer, RhtQ4Q6Q8Layer):
                _validate_multibit_layer(
                    layer,
                    expected_index=expected_index,
                    layout=self.layout,
                )
            else:
                _validate_residual_packed(
                    layer,
                    expected_index=expected_index,
                    layout=self.layout,
                )

        _validate_query_ema(self.query_energy_ema, self.layout)
        _validate_padding(self.padding, expected_bytes=padding_bytes)
        tensors = self.persistent_tensors()
        _validate_owned_tensors(tensors)
        devices = {tensor.device for _, tensor in tensors}
        if len(devices) != 1:
            raise ValueError("all persistent comparator tensors must share one device")
        illegal_fp32 = [
            name
            for name, tensor in tensors
            if tensor.dtype == torch.float32 and name != "query_energy_ema"
        ]
        if illegal_fp32:
            raise ValueError(
                "no-replay comparator retains an unexpected FP32 tensor: " + ", ".join(illegal_fp32)
            )

        payload, scale, precision = _component_bytes(self.codec, self.layers)
        query_bytes = _storage_bytes(self.query_energy_ema)
        actual_padding = _storage_bytes(self.padding)
        resident = payload + scale + precision + query_bytes + actual_padding
        selected = _selected_units(self.codec, self.layers)
        selection_hash = _selection_sha256(self.layers)
        expected_components = self.layout.component_bytes(self.codec)
        actual_components = {
            "payload_bytes": payload,
            "scale_bytes": scale,
            "precision_bytes": precision,
            "query_ema_bytes": query_bytes,
            "padding_bytes": actual_padding,
            "resident_bytes": resident,
        }
        if actual_components != expected_components:
            raise ValueError(
                f"{self.codec} physical byte accounting differs from the layout: "
                f"{actual_components} != {expected_components}"
            )
        if selected != expected_selected:
            raise ValueError(
                f"{self.codec} selected {selected} units, expected {expected_selected}"
            )
        if resident != self.layout.expected_resident_bytes:
            raise ValueError(
                f"{self.codec} owns {resident} resident bytes, expected "
                f"{self.layout.expected_resident_bytes}"
            )

        evidence = self.evidence
        expected_evidence = {
            "codec": self.codec,
            "state_elements": self.layout.state_elements,
            "fp32_state_bytes": self.layout.fp32_state_bytes,
            "payload_bytes": payload,
            "scale_bytes": scale,
            "precision_bytes": precision,
            "query_ema_bytes": query_bytes,
            "padding_bytes": actual_padding,
            "resident_bytes": resident,
            "selected_units": selected,
            "expected_selected_units": expected_selected,
            "selection_sha256": selection_hash,
        }
        for field, expected_value in expected_evidence.items():
            if getattr(evidence, field) != expected_value:
                raise ValueError(f"evidence field {field} does not match physical checkpoint state")
        for field in ("mean_squared_error", "relative_l2_error", "max_absolute_error"):
            value = getattr(evidence, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"evidence field {field} must be finite and non-negative")

    def materialize(self) -> dict[int, torch.Tensor]:
        """Materialize current states without retaining them in the checkpoint."""

        return _materialize_layers(self.layers, layout=self.layout)

    def to(self, device: torch.device | str) -> EqualByteCheckpoint:
        return EqualByteCheckpoint(
            codec=self.codec,
            layers=tuple(layer.to(device) for layer in self.layers),
            query_energy_ema=self.query_energy_ema.to(device),
            padding=self.padding.to(device),
            layout=self.layout,
            evidence=self.evidence,
        )


def _quantization_specs(
    layout: EqualByteLayout,
) -> tuple[QuantizationSpec, QuantizationSpec, QuantizationSpec]:
    common = {
        "group_size": layout.value_width,
        "scale_bits": 16,
        "flatten_last_dims": 2,
        "rounding": "nearest",
        "seed": RHT_SEED,
    }
    return (
        QuantizationSpec(bits=4, **common),
        QuantizationSpec(bits=6, **common),
        QuantizationSpec(bits=8, **common),
    )


def _validate_states(
    states: Mapping[int, torch.Tensor],
    *,
    layout: EqualByteLayout,
) -> torch.device:
    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping from model-layer index to tensor")
    if set(states) != set(layout.layer_indices):
        raise ValueError("states must contain exactly the frozen recurrent-layer indices")
    expected_shape = (1, layout.heads, layout.key_rows, layout.value_width)
    devices: set[torch.device] = set()
    for layer_index in layout.layer_indices:
        state = states[layer_index]
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state for layer {layer_index} must be a torch.Tensor")
        if state.dtype != torch.float32 or tuple(state.shape) != expected_shape:
            raise TypeError(
                f"state for layer {layer_index} must have shape {expected_shape} "
                "and dtype torch.float32"
            )
        if state.device.type == "meta":
            raise ValueError("state tensors must be materialized")
        if not torch.isfinite(state).all().item():
            raise ValueError(f"state for layer {layer_index} contains non-finite values")
        if torch.is_grad_enabled() and state.requires_grad:
            raise RuntimeError("equal-byte recurrent checkpoints are inference-only")
        devices.add(state.device)
    if len(devices) != 1:
        raise ValueError("all state tensors must share one device")
    return next(iter(devices))


def _prepare_query_ema(
    query_energy_ema: torch.Tensor,
    *,
    layout: EqualByteLayout,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(query_energy_ema, torch.Tensor):
        raise TypeError("query_energy_ema must be a torch.Tensor")
    if query_energy_ema.device != device:
        raise ValueError("query_energy_ema and states must share one device")
    _validate_query_ema(query_energy_ema, layout)
    return query_energy_ema.detach().clone(memory_format=torch.contiguous_format)


def _stable_global_top_mask(scores: torch.Tensor, quota: int) -> torch.Tensor:
    if scores.ndim != 1 or not scores.is_floating_point():
        raise TypeError("global scores must be a one-dimensional floating-point tensor")
    if not torch.isfinite(scores).all().item():
        raise ValueError("global scores must contain only finite values")
    if not 0 <= quota <= scores.numel():
        raise ValueError("global selection quota is outside the row count")
    mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    if quota:
        ranked = torch.argsort(scores, descending=True, stable=True)
        mask[ranked[:quota]] = True
    return mask


def _encode_and_distortions(
    states: Mapping[int, torch.Tensor],
    query_ema: torch.Tensor,
    *,
    layout: EqualByteLayout,
    bits: tuple[int, ...],
) -> tuple[dict[int, torch.Tensor], dict[int, tuple[torch.Tensor, ...]]]:
    specs = {spec.bits: spec for spec in _quantization_specs(layout)}
    encoded: dict[int, torch.Tensor] = {}
    distortions: dict[int, tuple[torch.Tensor, ...]] = {}
    for position, layer_index in enumerate(layout.layer_indices):
        transformed = right_rht_encode(
            states[layer_index],
            layer_index=layer_index,
            expected_heads=layout.heads,
            output_dtype=torch.float32,
        )
        encoded[layer_index] = transformed
        per_bit: list[torch.Tensor] = []
        for bit_width in bits:
            restored = quantize_dequantize(transformed, specs[bit_width]).tensor.to(torch.float32)
            row_mse = (
                (restored - transformed)
                .square()
                .mean(dim=-1)
                .reshape(
                    layout.heads,
                    layout.key_rows,
                )
            )
            weighted = query_ema[position] * row_mse
            if not torch.isfinite(weighted).all().item() or (weighted < 0).any().item():
                raise RuntimeError("physical row distortion is invalid")
            per_bit.append(weighted)
        distortions[layer_index] = tuple(per_bit)
    return encoded, distortions


def _split_global_rows(
    values: torch.Tensor,
    *,
    layout: EqualByteLayout,
) -> dict[int, torch.Tensor]:
    if values.numel() != layout.total_rows:
        raise ValueError("global row assignment has the wrong number of entries")
    flat = values.reshape(-1)
    return {
        layer_index: flat[
            position * layout.rows_per_layer : (position + 1) * layout.rows_per_layer
        ].reshape(layout.heads, layout.key_rows)
        for position, layer_index in enumerate(layout.layer_indices)
    }


def _error_metrics(
    source: Mapping[int, torch.Tensor],
    materialized: Mapping[int, torch.Tensor],
    *,
    layout: EqualByteLayout,
) -> tuple[float, float, float]:
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_source = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    for layer_index in layout.layer_indices:
        error = materialized[layer_index].detach().to("cpu", torch.float64) - source[
            layer_index
        ].detach().to("cpu", torch.float64)
        squared_error += error.square().sum()
        squared_source += source[layer_index].detach().to("cpu", torch.float64).square().sum()
        maximum = max(maximum, float(error.abs().max().item()))
    mse = float((squared_error / layout.state_elements).item())
    relative = float((squared_error.sqrt() / squared_source.sqrt().clamp_min(1e-12)).item())
    return mse, relative, maximum


def _materialize_layers(
    layers: tuple[EqualByteLayer, ...],
    *,
    layout: EqualByteLayout,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for layer in layers:
        if isinstance(layer, ExpandedRhtQ4Q8Layer):
            materialized = layer.materialize()
        elif isinstance(layer, RhtQ4Q6Q8Layer):
            materialized = layer.materialize(heads=layout.heads)
        else:
            materialized = layer.materialize(
                heads=layout.heads,
                key_rows=layout.key_rows,
                value_width=layout.value_width,
            )
        result[layer.layer_index] = materialized
    return result


def _make_checkpoint(
    *,
    codec: EqualByteCodecName,
    layers: tuple[EqualByteLayer, ...],
    query_ema: torch.Tensor,
    padding: torch.Tensor,
    source: Mapping[int, torch.Tensor],
    layout: EqualByteLayout,
) -> EqualByteCheckpoint:
    payload, scale, precision = _component_bytes(codec, layers)
    selected = _selected_units(codec, layers)
    if codec == EXPANDED_RHT_Q4_Q8:
        expected_selected = layout.expanded_q8_promotions
    elif codec == RHT_Q4_Q6_Q8:
        expected_selected = layout.multibit_marginal_steps
    else:
        expected_selected = layout.residual_q4_rows

    materialized = _materialize_layers(layers, layout=layout)
    mse, relative, maximum = _error_metrics(source, materialized, layout=layout)
    evidence = EqualByteCodecEvidence(
        codec=codec,
        state_elements=layout.state_elements,
        fp32_state_bytes=layout.fp32_state_bytes,
        payload_bytes=payload,
        scale_bytes=scale,
        precision_bytes=precision,
        query_ema_bytes=_storage_bytes(query_ema),
        padding_bytes=_storage_bytes(padding),
        resident_bytes=payload
        + scale
        + precision
        + _storage_bytes(query_ema)
        + _storage_bytes(padding),
        selected_units=selected,
        expected_selected_units=expected_selected,
        selection_sha256=_selection_sha256(layers),
        mean_squared_error=mse,
        relative_l2_error=relative,
        max_absolute_error=maximum,
    )
    return EqualByteCheckpoint(
        codec=codec,
        layers=layers,
        query_energy_ema=query_ema,
        padding=padding,
        layout=layout,
        evidence=evidence,
    )


def pack_expanded_rht_q4_q8(
    states: Mapping[int, torch.Tensor],
    query_energy_ema: torch.Tensor,
    *,
    layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
) -> EqualByteCheckpoint:
    """Pack the global top-benefit rows at Q8 and all remaining rows at Q4."""

    device = _validate_states(states, layout=layout)
    ema = _prepare_query_ema(query_energy_ema, layout=layout, device=device)
    _, distortions = _encode_and_distortions(
        states,
        ema,
        layout=layout,
        bits=(4, 8),
    )
    benefits = torch.cat(
        [
            (distortions[index][0] - distortions[index][1]).reshape(-1)
            for index in layout.layer_indices
        ]
    )
    global_mask = _stable_global_top_mask(
        benefits,
        layout.expanded_q8_promotions,
    )
    masks = _split_global_rows(global_mask, layout=layout)
    q4_spec, _, q8_spec = _quantization_specs(layout)
    layers: list[EqualByteLayer] = []
    for layer_index in layout.layer_indices:
        packed = quantize_pack_mixed(
            states[layer_index],
            masks[layer_index],
            low_spec=q4_spec,
            high_spec=q8_spec,
            right_rht_layer_index=layer_index,
            right_rht_expected_heads=layout.heads,
        )
        layers.append(ExpandedRhtQ4Q8Layer(layer_index, packed))
    padding = torch.zeros(
        layout.expanded_padding_bytes,
        dtype=torch.uint8,
        device=device,
    )
    return _make_checkpoint(
        codec=EXPANDED_RHT_Q4_Q8,
        layers=tuple(layers),
        query_ema=ema,
        padding=padding,
        source=states,
        layout=layout,
    )


def pack_rht_q4_q6_q8(
    states: Mapping[int, torch.Tensor],
    query_energy_ema: torch.Tensor,
    *,
    layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
) -> EqualByteCheckpoint:
    """Pack the exact complete-state Q4/Q6/Q8 rate-distortion optimum."""

    device = _validate_states(states, layout=layout)
    ema = _prepare_query_ema(query_energy_ema, layout=layout, device=device)
    encoded, distortions = _encode_and_distortions(
        states,
        ema,
        layout=layout,
        bits=(4, 6, 8),
    )
    flattened = [
        torch.cat(
            [distortions[index][position].reshape(-1) for index in layout.layer_indices]
        ).reshape(1, -1)
        for position in range(3)
    ]
    global_codes = allocate_exact_multibit_codes_fast(
        flattened[0],
        flattened[1],
        flattened[2],
        marginal_steps=layout.multibit_marginal_steps,
    ).reshape(-1)
    codes = _split_global_rows(global_codes, layout=layout)
    q4_spec, q6_spec, q8_spec = _quantization_specs(layout)
    layers: list[EqualByteLayer] = []
    for layer_index in layout.layer_indices:
        packed = quantize_pack_multibit(
            encoded[layer_index],
            codes[layer_index].to(device),
            int4_spec=q4_spec,
            int6_spec=q6_spec,
            int8_spec=q8_spec,
        )
        layers.append(RhtQ4Q6Q8Layer(layer_index, packed))
    padding = torch.zeros(
        layout.multibit_padding_bytes,
        dtype=torch.uint8,
        device=device,
    )
    return _make_checkpoint(
        codec=RHT_Q4_Q6_Q8,
        layers=tuple(layers),
        query_ema=ema,
        padding=padding,
        source=states,
        layout=layout,
    )


def _empty_residual_pack(
    *,
    layout: EqualByteLayout,
    device: torch.device,
    spec: QuantizationSpec,
) -> PackedQuantizedTensor:
    return PackedQuantizedTensor(
        payload=torch.empty(0, dtype=torch.uint8, device=device),
        scales=torch.empty((0, 1), dtype=torch.float16, device=device),
        spec=replace(spec, flatten_last_dims=1),
        original_shape=(0, layout.value_width),
        original_dtype=torch.float32,
        flattened_size=layout.value_width,
        padded_size=layout.value_width,
        rows=0,
        groups_per_row=1,
    )


def pack_rht_residual_q4(
    states: Mapping[int, torch.Tensor],
    query_energy_ema: torch.Tensor,
    *,
    layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
) -> EqualByteCheckpoint:
    """Pack base RHT-Q4 plus exact top-benefit residual RHT-Q4 rows."""

    device = _validate_states(states, layout=layout)
    ema = _prepare_query_ema(query_energy_ema, layout=layout, device=device)
    q4_spec, _, _ = _quantization_specs(layout)
    residual_spec = replace(q4_spec, flatten_last_dims=1)
    encoded: dict[int, torch.Tensor] = {}
    base_packed: dict[int, PackedQuantizedTensor] = {}
    residual_sources: dict[int, torch.Tensor] = {}
    benefits: list[torch.Tensor] = []
    for position, layer_index in enumerate(layout.layer_indices):
        transformed = right_rht_encode(
            states[layer_index],
            layer_index=layer_index,
            expected_heads=layout.heads,
            output_dtype=torch.float32,
        )
        encoded[layer_index] = transformed
        base = quantize_pack(transformed, q4_spec)
        base_packed[layer_index] = base
        base_hat = base.dequantize().to(torch.float32)
        residual = (transformed - base_hat).reshape(
            layout.rows_per_layer,
            layout.value_width,
        )
        residual_sources[layer_index] = residual
        residual_hat = quantize_dequantize(residual, residual_spec).tensor.to(torch.float32)
        base_error = (base_hat - transformed).reshape(
            layout.rows_per_layer,
            layout.value_width,
        )
        corrected_error = base_error + residual_hat
        physical_benefit = base_error.square().mean(dim=-1) - corrected_error.square().mean(dim=-1)
        weighted = ema[position].reshape(-1) * physical_benefit
        if not torch.isfinite(weighted).all().item():
            raise RuntimeError("residual-Q4 physical benefit is non-finite")
        benefits.append(weighted)
    global_mask = _stable_global_top_mask(
        torch.cat(benefits),
        layout.residual_q4_rows,
    )
    masks = _split_global_rows(global_mask, layout=layout)
    layers: list[EqualByteLayer] = []
    for layer_index in layout.layer_indices:
        flat_mask = masks[layer_index].reshape(-1)
        selected_source = residual_sources[layer_index][flat_mask]
        if selected_source.shape[0]:
            residual = quantize_pack(selected_source, residual_spec)
        else:
            residual = _empty_residual_pack(
                layout=layout,
                device=device,
                spec=q4_spec,
            )
        layers.append(
            RhtResidualQ4Layer(
                layer_index=layer_index,
                base=base_packed[layer_index],
                residual=residual,
                lease_mask=_pack_bool_mask(flat_mask),
            )
        )
    padding = torch.zeros(
        layout.residual_padding_bytes,
        dtype=torch.uint8,
        device=device,
    )
    return _make_checkpoint(
        codec=RHT_RESIDUAL_Q4,
        layers=tuple(layers),
        query_ema=ema,
        padding=padding,
        source=states,
        layout=layout,
    )


def update_causal_query_ema(
    previous: torch.Tensor | None,
    queries: Mapping[int, torch.Tensor],
    *,
    layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
) -> torch.Tensor:
    """Apply the frozen causal normalized-query-energy EMA for every layer."""

    if not isinstance(queries, Mapping) or set(queries) != set(layout.layer_indices):
        raise ValueError("queries must contain exactly the frozen recurrent-layer indices")
    if previous is not None:
        _validate_query_ema(previous, layout)
    candidates: list[torch.Tensor] = []
    token_count: int | None = None
    common_device: torch.device | None = None
    for position, layer_index in enumerate(layout.layer_indices):
        query = queries[layer_index]
        expected_tail = (layout.heads, layout.key_rows)
        if (
            not isinstance(query, torch.Tensor)
            or query.ndim != 4
            or query.shape[0] != 1
            or tuple(query.shape[2:]) != expected_tail
            or not query.is_floating_point()
        ):
            raise TypeError(
                f"query for layer {layer_index} must have shape "
                f"[1, tokens, {layout.heads}, {layout.key_rows}] and floating dtype"
            )
        if query.shape[1] <= 0:
            raise ValueError("query token count must be positive")
        if not torch.isfinite(query).all().item():
            raise ValueError(f"query for layer {layer_index} contains non-finite values")
        if token_count is None:
            token_count = query.shape[1]
            common_device = query.device
        elif query.shape[1] != token_count:
            raise ValueError("all recurrent layers must observe the same token count")
        if query.device != common_device:
            raise ValueError("all query tensors must share one device")
        if previous is not None and previous.device != query.device:
            raise ValueError("previous query EMA and current queries must share one device")

        source = query.detach().to(torch.float32)
        squared = source.square()
        energy = squared / (squared.sum(dim=-1, keepdim=True) + QUERY_L2NORM_EPS)
        energy = energy.squeeze(0)
        prior = (
            torch.full(
                expected_tail,
                1.0 / layout.key_rows,
                dtype=torch.float32,
                device=query.device,
            )
            if previous is None
            else previous[position]
        )
        assert token_count is not None
        exponents = torch.arange(
            token_count - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=query.device,
        )
        weights = torch.pow(
            torch.tensor(QUERY_EMA_DECAY, dtype=torch.float32, device=query.device),
            exponents,
        )
        candidate = (QUERY_EMA_DECAY**token_count) * prior + (1.0 - QUERY_EMA_DECAY) * (
            energy * weights[:, None, None]
        ).sum(dim=0)
        candidates.append(candidate)
    result = torch.stack(candidates).contiguous()
    _validate_query_ema(result, layout)
    return result


class EqualByteNoReplayCache:
    """Atomic stateful wrapper around one frozen no-replay comparator.

    A commit computes the causal EMA and the complete packed candidate before
    mutating the cache.  Any validation, allocation, quantization, or evidence
    failure leaves the previous checkpoint and counters unchanged.
    """

    def __init__(
        self,
        codec: EqualByteCodecName,
        *,
        layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    ) -> None:
        if codec not in _SUPPORTED_CODECS:
            raise ValueError(f"unsupported equal-byte codec: {codec!r}")
        self.codec = codec
        self.layout = layout
        self.checkpoint: EqualByteCheckpoint | None = None
        self.update_count = 0
        self.last_evidence: EqualByteCodecEvidence | None = None

    def _pack_candidate(
        self,
        states: Mapping[int, torch.Tensor],
        query_ema: torch.Tensor,
    ) -> EqualByteCheckpoint:
        if self.codec == EXPANDED_RHT_Q4_Q8:
            return pack_expanded_rht_q4_q8(states, query_ema, layout=self.layout)
        if self.codec == RHT_Q4_Q6_Q8:
            return pack_rht_q4_q6_q8(states, query_ema, layout=self.layout)
        return pack_rht_residual_q4(states, query_ema, layout=self.layout)

    def commit(
        self,
        states: Mapping[int, torch.Tensor],
        queries: Mapping[int, torch.Tensor],
    ) -> EqualByteCheckpoint:
        previous_ema = None if self.checkpoint is None else self.checkpoint.query_energy_ema
        candidate_ema = update_causal_query_ema(
            previous_ema,
            queries,
            layout=self.layout,
        )
        candidate = self._pack_candidate(states, candidate_ema)
        candidate.validate()
        self.checkpoint = candidate
        self.update_count += 1
        self.last_evidence = candidate.evidence
        return candidate

    def materialize(self) -> dict[int, torch.Tensor]:
        if self.checkpoint is None:
            raise RuntimeError("equal-byte cache has no committed checkpoint")
        return self.checkpoint.materialize()

    def resident_bytes(self) -> int:
        return 0 if self.checkpoint is None else self.checkpoint.resident_bytes

    def reset(self) -> None:
        self.checkpoint = None
        self.update_count = 0
        self.last_evidence = None

    def to(self, device: torch.device | str) -> None:
        if self.checkpoint is not None:
            self.checkpoint = self.checkpoint.to(device)
