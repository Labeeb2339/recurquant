"""Physical mixed-group INT4/INT6/INT8 packing for recurrent states.

Each quantization group selects one of three symmetric signed precisions:
``0 = INT4``, ``1 = INT6``, and ``2 = INT8``. Four precision codes share one
byte, every group owns exactly one stored scale, and integer payloads are kept
in separate width-specific pools. ``storage_bytes`` counts the resident tensor
bytes exactly; Python metadata is intentionally excluded.

INT6 uses a little-endian bit stream within each group: four two's-complement
6-bit codes occupy three bytes. Consequently, a 128-element INT6 group occupies
exactly 96 payload bytes.
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

INT4_PRECISION_CODE = 0
INT6_PRECISION_CODE = 1
INT8_PRECISION_CODE = 2
_BITS_BY_PRECISION_CODE = (4, 6, 8)
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_multibit_specs(
    int4_spec: QuantizationSpec,
    int6_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
) -> None:
    specs = (int4_spec, int6_spec, int8_spec)
    expected_bits = _BITS_BY_PRECISION_CODE
    for label, spec, bits in zip(
        ("int4_spec", "int6_spec", "int8_spec"),
        specs,
        expected_bits,
        strict=True,
    ):
        if not isinstance(spec, QuantizationSpec):
            raise TypeError(f"{label} must be a QuantizationSpec")
        if spec.bits != bits:
            raise ValueError(f"{label} must use {bits} bits")

    shared_fields = (
        "group_size",
        "scale_bits",
        "flatten_last_dims",
        "rounding",
        "seed",
        "epsilon",
    )
    mismatched = [
        field
        for field in shared_fields
        if any(getattr(spec, field) != getattr(int4_spec, field) for spec in specs[1:])
    ]
    if mismatched:
        raise ValueError(
            "INT4, INT6, and INT8 specs must differ only in bits; mismatched fields: "
            + ", ".join(mismatched)
        )

    non_byte_widths = [
        bits for bits in expected_bits if (int4_spec.group_size * bits) % 8 != 0
    ]
    if non_byte_widths:
        widths = ", ".join(f"INT{bits}" for bits in non_byte_widths)
        raise ValueError(
            f"group_size={int4_spec.group_size} does not give whole-byte payloads for {widths}"
        )


def _validate_precision_code_shape(
    precision_codes: torch.Tensor,
    *,
    rows: int,
    groups_per_row: int,
) -> None:
    if not isinstance(precision_codes, torch.Tensor):
        raise TypeError("precision_codes must be a torch.Tensor")
    if precision_codes.dtype != torch.uint8:
        raise TypeError("precision_codes must use torch.uint8")

    expected_flat = (rows * groups_per_row,)
    expected_grouped = (rows, groups_per_row)
    if tuple(precision_codes.shape) not in (expected_flat, expected_grouped):
        raise ValueError(
            "precision_codes must have shape "
            f"{expected_flat} or {expected_grouped}, got {tuple(precision_codes.shape)}"
        )
    if (precision_codes > INT8_PRECISION_CODE).any().item():
        raise ValueError("precision_codes may contain only 0=INT4, 1=INT6, or 2=INT8")


def _pack_precision_codes(precision_codes: torch.Tensor) -> torch.Tensor:
    """Pack four canonical two-bit precision codes into each byte."""

    if not isinstance(precision_codes, torch.Tensor):
        raise TypeError("precision_codes must be a torch.Tensor")
    if precision_codes.dtype != torch.uint8:
        raise TypeError("precision_codes must use torch.uint8")
    if (precision_codes > INT8_PRECISION_CODE).any().item():
        raise ValueError("precision code 3 is reserved and must be rejected")

    flat = precision_codes.reshape(-1)
    padding = (-flat.numel()) % 4
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    chunks = flat.reshape(flat.numel() // 4, 4).to(torch.int16)
    packed = (
        chunks[:, 0]
        | torch.bitwise_left_shift(chunks[:, 1], 2)
        | torch.bitwise_left_shift(chunks[:, 2], 4)
        | torch.bitwise_left_shift(chunks[:, 3], 6)
    )
    return packed.to(torch.uint8).contiguous()


def _unpack_precision_codes(
    packed: torch.Tensor,
    total_groups: int,
) -> torch.Tensor:
    """Unpack and validate the canonical two-bit precision stream."""

    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed precision codes must be a torch.Tensor")
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise TypeError("packed precision codes must be a one-dimensional torch.uint8 tensor")
    if not isinstance(total_groups, int) or total_groups < 0:
        raise ValueError("total_groups must be a non-negative integer")

    expected_bytes = math.ceil(total_groups / 4)
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"packed precision stream must contain {expected_bytes} bytes, got {packed.numel()}"
        )

    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.int16, device=packed.device)
    expanded = torch.bitwise_right_shift(packed.to(torch.int16).unsqueeze(1), shifts)
    all_codes = torch.bitwise_and(expanded, 0x03).reshape(-1).to(torch.uint8)
    codes = all_codes[:total_groups]
    if (codes == 3).any().item():
        raise ValueError("packed precision stream contains reserved precision code 3")
    if (all_codes[total_groups:] != 0).any().item():
        raise ValueError("unused precision-code padding bits must be zero")
    return codes


def _validate_integer_groups(
    codes: torch.Tensor,
    *,
    bits: int,
) -> None:
    if not isinstance(codes, torch.Tensor):
        raise TypeError("integer codes must be a torch.Tensor")
    if codes.dtype not in _INTEGER_DTYPES:
        raise TypeError("integer codes must use an integer dtype")
    if codes.ndim != 2:
        raise ValueError("integer codes must have shape [groups, group_size]")
    if (codes.shape[1] * bits) % 8:
        raise ValueError(f"INT{bits} group payload must occupy a whole number of bytes")

    qmax = (1 << (bits - 1)) - 1
    if codes.numel() and (codes.min().item() < -qmax or codes.max().item() > qmax):
        raise ValueError(f"INT{bits} symmetric codes must lie in [{-qmax}, {qmax}]")


def _pack_int4_groups(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed symmetric INT4 groups, preserving each group boundary."""

    _validate_integer_groups(codes, bits=4)
    nibbles = torch.bitwise_and(codes.to(torch.int16), 0x0F)
    return (
        nibbles[:, 0::2] | torch.bitwise_left_shift(nibbles[:, 1::2], 4)
    ).to(torch.uint8).contiguous()


def _unpack_int4_groups(payload: torch.Tensor, group_size: int) -> torch.Tensor:
    if payload.dtype != torch.uint8 or payload.ndim != 2:
        raise TypeError("INT4 payload must be a two-dimensional torch.uint8 tensor")
    expected_width = group_size * 4 // 8
    if group_size <= 0 or group_size * 4 % 8 or payload.shape[1] != expected_width:
        raise ValueError("INT4 payload shape is inconsistent with group_size")

    low = torch.bitwise_and(payload, 0x0F).to(torch.int16)
    high = torch.bitwise_right_shift(payload, 4).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    codes = torch.empty(
        (payload.shape[0], group_size),
        dtype=torch.int16,
        device=payload.device,
    )
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    if codes.numel() and (codes == -8).any().item():
        raise ValueError("INT4 payload contains the reserved symmetric code -8")
    return codes


def _pack_int6_groups(codes: torch.Tensor) -> torch.Tensor:
    """Pack four signed two's-complement INT6 codes into three bytes."""

    _validate_integer_groups(codes, bits=6)
    unsigned = torch.bitwise_and(codes.to(torch.int16), 0x3F).reshape(
        codes.shape[0],
        codes.shape[1] // 4,
        4,
    )
    first = unsigned[:, :, 0]
    second = unsigned[:, :, 1]
    third = unsigned[:, :, 2]
    fourth = unsigned[:, :, 3]

    byte0 = first | torch.bitwise_left_shift(torch.bitwise_and(second, 0x03), 6)
    byte1 = torch.bitwise_right_shift(second, 2) | torch.bitwise_left_shift(
        torch.bitwise_and(third, 0x0F),
        4,
    )
    byte2 = torch.bitwise_right_shift(third, 4) | torch.bitwise_left_shift(fourth, 2)
    return (
        torch.stack((byte0, byte1, byte2), dim=-1)
        .reshape(codes.shape[0], codes.shape[1] * 6 // 8)
        .to(torch.uint8)
        .contiguous()
    )


def _unpack_int6_groups(payload: torch.Tensor, group_size: int) -> torch.Tensor:
    """Unpack group-aligned three-byte blocks into signed INT6 codes."""

    if payload.dtype != torch.uint8 or payload.ndim != 2:
        raise TypeError("INT6 payload must be a two-dimensional torch.uint8 tensor")
    expected_width = group_size * 6 // 8
    if group_size <= 0 or group_size * 6 % 8 or payload.shape[1] != expected_width:
        raise ValueError("INT6 payload shape is inconsistent with group_size")

    triples = payload.reshape(payload.shape[0], group_size // 4, 3).to(torch.int16)
    byte0 = triples[:, :, 0]
    byte1 = triples[:, :, 1]
    byte2 = triples[:, :, 2]
    first = torch.bitwise_and(byte0, 0x3F)
    second = torch.bitwise_right_shift(byte0, 6) | torch.bitwise_left_shift(
        torch.bitwise_and(byte1, 0x0F),
        2,
    )
    third = torch.bitwise_right_shift(byte1, 4) | torch.bitwise_left_shift(
        torch.bitwise_and(byte2, 0x03),
        4,
    )
    fourth = torch.bitwise_right_shift(byte2, 2)
    unsigned = torch.stack((first, second, third, fourth), dim=-1).reshape(
        payload.shape[0],
        group_size,
    )
    codes = torch.where(unsigned >= 32, unsigned - 64, unsigned).to(torch.int16)
    if codes.numel() and (codes == -32).any().item():
        raise ValueError("INT6 payload contains the reserved symmetric code -32")
    return codes


def _quantize_multibit_groups(
    grouped: torch.Tensor,
    precision_codes: torch.Tensor,
    spec: QuantizationSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax_lookup = torch.tensor((7.0, 31.0, 127.0), device=grouped.device)
    qmax = qmax_lookup[precision_codes.to(torch.long)].unsqueeze(1)
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    ideal_scales = torch.where(
        absmax > spec.epsilon,
        absmax / qmax,
        torch.ones_like(absmax),
    )
    scale_dtype = _scale_dtype(spec.scale_bits)
    if scale_dtype == torch.float16:
        ideal_scales = ideal_scales.clamp(
            min=2.0**-24,
            max=torch.finfo(torch.float16).max,
        )
    stored_scales = ideal_scales.to(scale_dtype)
    normalized = grouped / stored_scales.to(torch.float32)
    if spec.rounding == "nearest":
        quantized = torch.round(normalized)
    else:
        quantized = _stochastic_round(normalized, seed=spec.seed)
    quantized = torch.minimum(torch.maximum(quantized, -qmax), qmax)
    return quantized.to(torch.int16), stored_scales.squeeze(1).contiguous()


@dataclass(frozen=True, slots=True)
class PackedMultiBitQuantizedTensor:
    """A tensor with separate INT4, INT6, and INT8 group payload pools."""

    int4_payload: torch.Tensor
    int6_payload: torch.Tensor
    int8_payload: torch.Tensor
    scales: torch.Tensor
    packed_precision_codes: torch.Tensor
    int4_spec: QuantizationSpec
    int6_spec: QuantizationSpec
    int8_spec: QuantizationSpec
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    flattened_size: int
    padded_size: int
    rows: int
    groups_per_row: int

    def __post_init__(self) -> None:
        """Reject packed objects whose metadata and resident tensors disagree."""

        _validate_multibit_specs(self.int4_spec, self.int6_spec, self.int8_spec)
        metadata = {
            "flattened_size": self.flattened_size,
            "padded_size": self.padded_size,
            "rows": self.rows,
            "groups_per_row": self.groups_per_row,
        }
        for name, value in metadata.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if not isinstance(self.original_shape, tuple):
            raise TypeError("original_shape must be a tuple of integers")
        if not self.original_shape:
            raise ValueError("original_shape must not be empty")
        if any(
            not isinstance(dimension, int) or isinstance(dimension, bool)
            for dimension in self.original_shape
        ):
            raise TypeError("original_shape must be a tuple of integers")
        if any(dimension < 0 for dimension in self.original_shape):
            raise ValueError("original_shape dimensions must be non-negative")
        if len(self.original_shape) < self.int4_spec.flatten_last_dims:
            raise ValueError("original_shape has fewer dimensions than flatten_last_dims")

        flattened_dimensions = self.original_shape[-self.int4_spec.flatten_last_dims :]
        if any(dimension <= 0 for dimension in flattened_dimensions):
            raise ValueError("flattened original_shape dimensions must be positive")
        expected_flattened_size = math.prod(flattened_dimensions)
        if self.flattened_size != expected_flattened_size:
            raise ValueError(
                "flattened_size is inconsistent with original_shape and flatten_last_dims"
            )
        expected_groups_per_row = math.ceil(self.flattened_size / self.int4_spec.group_size)
        if self.groups_per_row != expected_groups_per_row:
            raise ValueError("groups_per_row is inconsistent with flattened_size and group_size")
        expected_padded_size = self.groups_per_row * self.int4_spec.group_size
        if self.padded_size != expected_padded_size:
            raise ValueError("padded_size is inconsistent with groups_per_row and group_size")
        expected_rows = math.prod(
            self.original_shape[: -self.int4_spec.flatten_last_dims]
        )
        if self.rows != expected_rows:
            raise ValueError("rows is inconsistent with original_shape and flatten_last_dims")

        if not isinstance(self.original_dtype, torch.dtype):
            raise TypeError("original_dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.original_dtype).is_floating_point():
            raise TypeError("original_dtype must be a floating-point dtype")

        resident_tensors = {
            "int4_payload": self.int4_payload,
            "int6_payload": self.int6_payload,
            "int8_payload": self.int8_payload,
            "scales": self.scales,
            "packed_precision_codes": self.packed_precision_codes,
        }
        for name, tensor in resident_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        resident_device = self.scales.device
        mismatched_devices = [
            name
            for name, tensor in resident_tensors.items()
            if tensor.device != resident_device
        ]
        if mismatched_devices:
            raise ValueError(
                "all resident tensors must share one device; mismatched: "
                + ", ".join(mismatched_devices)
            )

        expected_scale_dtype = _scale_dtype(self.int4_spec.scale_bits)
        if self.scales.dtype != expected_scale_dtype or self.scales.ndim != 1:
            raise TypeError(
                f"scales must be a one-dimensional {expected_scale_dtype} tensor"
            )
        if self.scales.numel() != self.total_groups:
            raise ValueError(
                f"scales must contain one value per group ({self.total_groups})"
            )
        if self.scales.numel():
            if not torch.isfinite(self.scales).all().item():
                raise ValueError("scales must contain only finite values")
            if (self.scales <= 0).any().item():
                raise ValueError("scales must be strictly positive")

        precision_codes = _unpack_precision_codes(
            self.packed_precision_codes,
            self.total_groups,
        )
        expected_counts = tuple(
            int((precision_codes == code).sum().item())
            for code in (
                INT4_PRECISION_CODE,
                INT6_PRECISION_CODE,
                INT8_PRECISION_CODE,
            )
        )
        payload_layouts = (
            ("int4_payload", self.int4_payload, torch.uint8, 4, expected_counts[0]),
            ("int6_payload", self.int6_payload, torch.uint8, 6, expected_counts[1]),
            ("int8_payload", self.int8_payload, torch.int8, 8, expected_counts[2]),
        )
        for name, payload, dtype, bits, expected_groups in payload_layouts:
            if payload.dtype != dtype or payload.ndim != 2:
                raise TypeError(f"{name} must be a two-dimensional {dtype} tensor")
            expected_width = self.int4_spec.group_size * bits // 8
            if tuple(payload.shape) != (expected_groups, expected_width):
                raise ValueError(
                    f"{name} must have shape {(expected_groups, expected_width)}, "
                    f"got {tuple(payload.shape)}"
                )

        if self.int4_payload.numel():
            low = torch.bitwise_and(self.int4_payload, 0x0F)
            high = torch.bitwise_right_shift(self.int4_payload, 4)
            if ((low == 8) | (high == 8)).any().item():
                raise ValueError("int4_payload contains the reserved symmetric code -8")
        if self.int6_payload.numel():
            _unpack_int6_groups(self.int6_payload, self.int4_spec.group_size)
        if self.int8_payload.numel() and (self.int8_payload == -128).any().item():
            raise ValueError("int8_payload contains the reserved symmetric code -128")

    @property
    def elements(self) -> int:
        return math.prod(self.original_shape)

    @property
    def total_groups(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def int4_groups(self) -> int:
        return self.int4_payload.shape[0]

    @property
    def int6_groups(self) -> int:
        return self.int6_payload.shape[0]

    @property
    def int8_groups(self) -> int:
        return self.int8_payload.shape[0]

    @property
    def payload_bytes(self) -> int:
        return sum(
            payload.numel() * payload.element_size()
            for payload in (self.int4_payload, self.int6_payload, self.int8_payload)
        )

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    @property
    def precision_code_bytes(self) -> int:
        return self.packed_precision_codes.numel() * self.packed_precision_codes.element_size()

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes + self.precision_code_bytes

    def precision_codes(self) -> torch.Tensor:
        """Return precision codes in ``[rows, groups_per_row]`` order."""

        return _unpack_precision_codes(
            self.packed_precision_codes,
            self.total_groups,
        ).reshape(self.rows, self.groups_per_row)

    def _integer_groups(self) -> torch.Tensor:
        precision_codes = self.precision_codes().reshape(-1)
        codes = torch.empty(
            (self.total_groups, self.int4_spec.group_size),
            dtype=torch.int16,
            device=self.scales.device,
        )
        codes[precision_codes == INT4_PRECISION_CODE] = _unpack_int4_groups(
            self.int4_payload,
            self.int4_spec.group_size,
        )
        codes[precision_codes == INT6_PRECISION_CODE] = _unpack_int6_groups(
            self.int6_payload,
            self.int4_spec.group_size,
        )
        int8_codes = self.int8_payload.to(torch.int16)
        if int8_codes.numel() and (int8_codes == -128).any().item():
            raise ValueError("INT8 payload contains the reserved symmetric code -128")
        codes[precision_codes == INT8_PRECISION_CODE] = int8_codes
        return codes

    def dequantize(self) -> torch.Tensor:
        """Materialize the mixed packed tensor in its original dtype and shape."""

        restored_groups = self._integer_groups().to(torch.float32) * self.scales.to(
            torch.float32
        ).unsqueeze(1)
        restored = restored_groups.reshape(self.rows, self.padded_size)[:, : self.flattened_size]
        return restored.reshape(self.original_shape).to(self.original_dtype)

    def to(self, device: torch.device | str) -> PackedMultiBitQuantizedTensor:
        """Move every resident tensor without materializing or requantizing."""

        return PackedMultiBitQuantizedTensor(
            int4_payload=self.int4_payload.to(device),
            int6_payload=self.int6_payload.to(device),
            int8_payload=self.int8_payload.to(device),
            scales=self.scales.to(device),
            packed_precision_codes=self.packed_precision_codes.to(device),
            int4_spec=self.int4_spec,
            int6_spec=self.int6_spec,
            int8_spec=self.int8_spec,
            original_shape=self.original_shape,
            original_dtype=self.original_dtype,
            flattened_size=self.flattened_size,
            padded_size=self.padded_size,
            rows=self.rows,
            groups_per_row=self.groups_per_row,
        )

    def reorder_batch(self, beam_idx: torch.LongTensor) -> PackedMultiBitQuantizedTensor:
        """Select packed batch entries without another quantization round trip."""

        if len(self.original_shape) <= self.int4_spec.flatten_last_dims:
            raise RuntimeError(
                "Cannot reorder multibit groups when quantization includes the batch dimension"
            )
        if not isinstance(beam_idx, torch.Tensor):
            raise TypeError("beam_idx must be a torch.Tensor")
        if beam_idx.ndim != 1 or beam_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("beam_idx must be a one-dimensional int32 or int64 tensor")

        batch_size = self.original_shape[0]
        if batch_size <= 0 or self.rows % batch_size:
            raise RuntimeError("multibit row metadata is inconsistent with its batch dimension")
        if beam_idx.numel() and (beam_idx.min().item() < 0 or beam_idx.max().item() >= batch_size):
            raise IndexError("beam_idx contains an out-of-range batch index")

        groups_per_batch = self.total_groups // batch_size
        selected = beam_idx.to(self.scales.device)
        selected_batch_size = selected.numel()
        codes = self._integer_groups().reshape(
            batch_size,
            groups_per_batch,
            self.int4_spec.group_size,
        )
        scales = self.scales.reshape(batch_size, groups_per_batch)
        precision_codes = self.precision_codes().reshape(batch_size, groups_per_batch)

        reordered_codes = codes.index_select(0, selected).reshape(
            selected_batch_size * groups_per_batch,
            self.int4_spec.group_size,
        )
        reordered_scales = scales.index_select(0, selected).reshape(-1).contiguous()
        reordered_precision = precision_codes.index_select(0, selected).reshape(-1)
        return _from_integer_groups(
            reordered_codes,
            reordered_scales,
            reordered_precision,
            int4_spec=self.int4_spec,
            int6_spec=self.int6_spec,
            int8_spec=self.int8_spec,
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
    precision_codes: torch.Tensor,
    *,
    int4_spec: QuantizationSpec,
    int6_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
    original_shape: tuple[int, ...],
    original_dtype: torch.dtype,
    flattened_size: int,
    padded_size: int,
    rows: int,
    groups_per_row: int,
) -> PackedMultiBitQuantizedTensor:
    flat_precision = precision_codes.reshape(-1)
    return PackedMultiBitQuantizedTensor(
        int4_payload=_pack_int4_groups(codes[flat_precision == INT4_PRECISION_CODE]),
        int6_payload=_pack_int6_groups(codes[flat_precision == INT6_PRECISION_CODE]),
        int8_payload=codes[flat_precision == INT8_PRECISION_CODE].to(torch.int8).contiguous(),
        scales=scales.contiguous(),
        packed_precision_codes=_pack_precision_codes(flat_precision),
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
        original_shape=original_shape,
        original_dtype=original_dtype,
        flattened_size=flattened_size,
        padded_size=padded_size,
        rows=rows,
        groups_per_row=groups_per_row,
    )


def quantize_pack_multibit(
    tensor: torch.Tensor,
    precision_codes: torch.Tensor,
    *,
    int4_spec: QuantizationSpec,
    int6_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
) -> PackedMultiBitQuantizedTensor:
    """Quantize groups into separate physical INT4, INT6, and INT8 pools.

    ``precision_codes`` may be flat or shaped ``[rows, groups_per_row]`` in the
    exact group order produced by the shared grouped quantizer. It must use
    ``torch.uint8`` and the canonical values 0, 1, and 2.
    """

    _validate_multibit_specs(int4_spec, int6_spec, int8_spec)
    working, original_dtype, flattened_size, padded_size, groups_per_row, grouped = _group_tensor(
        tensor,
        int4_spec,
    )
    rows = grouped.shape[0]
    _validate_precision_code_shape(
        precision_codes,
        rows=rows,
        groups_per_row=groups_per_row,
    )
    flat_precision = precision_codes.detach().to(grouped.device).reshape(-1)
    flat_groups = grouped.reshape(-1, int4_spec.group_size)
    codes, scales = _quantize_multibit_groups(
        flat_groups,
        flat_precision,
        int4_spec,
    )
    return _from_integer_groups(
        codes,
        scales,
        flat_precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
        original_shape=tuple(working.shape),
        original_dtype=original_dtype,
        flattened_size=flattened_size,
        padded_size=padded_size,
        rows=rows,
        groups_per_row=groups_per_row,
    )
