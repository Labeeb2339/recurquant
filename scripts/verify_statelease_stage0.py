#!/usr/bin/env python
"""Independent synthetic Stage-0 verifier for Experiment 010.

This file deliberately contains its own dense Gated DeltaNet recurrence,
randomized-Hadamard transform, signed bit packing, row quantizers, boundary
decision, storage accounting, and artifact checks.  It has no dependency on
the package implementation being verified.

The verifier consumes only synthetic tensors or caller-provided closed-schema
snapshots.  It never selects, formats, tokenizes, or evaluates model-quality
data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

FROZEN_SEED = 2339
SIGN_DOMAIN = b"recurquant.right-rht.signs.v1\0"
KEY_NORM_EPS = 1e-6
REPLAY_RELATIVE_L2_LIMIT = 3e-6
REPLAY_MAX_ABSOLUTE_LIMIT = 1e-5

LAYERS = 18
HEADS = 16
ROWS = 128
WIDTH = 128
REPLAY_CAPACITY = 5
STATE_ELEMENTS = LAYERS * HEADS * ROWS * WIDTH
STATE_ROWS = LAYERS * HEADS * ROWS
HIGH_PRECISION_ROWS = 1_976
LOW_PRECISION_ROWS = STATE_ROWS - HIGH_PRECISION_ROWS

CHECKPOINT_PAYLOAD_BYTES = 2_485_760
CHECKPOINT_SCALE_BYTES = 73_728
CHECKPOINT_MASK_BYTES = 4_608
CHECKPOINT_BYTES = 2_564_096
QUERY_EMA_BYTES = 147_456
KEY_BUFFER_BYTES = 368_640
UPDATE_BUFFER_BYTES = 368_640
LOG_DECAY_BUFFER_BYTES = 5_760
COUNT_BYTES = 72
STATELEASE_BYTES = 3_454_664

EXPANDED_Q48_BASE_BYTES = 2_585_088
EXPANDED_Q48_PROMOTIONS = 13_587
EXPANDED_Q48_PADDING_BYTES = 8
MULTIBIT_BASE_BYTES = 2_589_696
MULTIBIT_MARGINAL_STEPS = 27_030
MULTIBIT_PADDING_BYTES = 8
RESIDUAL_Q4_BASE_BYTES = 2_585_088
RESIDUAL_Q4_ROWS = 13_175
RESIDUAL_Q4_PADDING_BYTES = 26

ALLOWED_CONSUMED_DTYPES = frozenset(
    (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    )
)


class Stage0VerificationError(RuntimeError):
    """A fail-closed Stage-0 condition."""


def _fail(message: str) -> None:
    raise Stage0VerificationError(message)


def _require_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype | None = None,
    ndim: int | None = None,
    finite: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}; got {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got {value.ndim}")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if finite and value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def relative_l2_and_max_abs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    error = candidate.to(torch.float64) - reference.to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.to(torch.float64)).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(error) / denominator
    return float(relative_l2.item()), float(error.abs().max().item())


def assert_replay_matches(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    relative_l2_limit: float = REPLAY_RELATIVE_L2_LIMIT,
    max_absolute_limit: float = REPLAY_MAX_ABSOLUTE_LIMIT,
) -> tuple[float, float]:
    relative_l2, max_absolute = relative_l2_and_max_abs(reference, candidate)
    if relative_l2 > relative_l2_limit or max_absolute > max_absolute_limit:
        _fail(
            f"dense replay mismatch: relative_l2={relative_l2:.9g}, max_absolute={max_absolute:.9g}"
        )
    return relative_l2, max_absolute


def normalize_consumed_key(consumed_key: torch.Tensor) -> torch.Tensor:
    """Mirror pinned Qwen: normalize in input dtype, then convert to FP32."""

    key = _require_tensor(consumed_key, name="consumed_key")
    if key.dtype not in ALLOWED_CONSUMED_DTYPES:
        raise TypeError(
            "consumed_key must use float32, bfloat16, or float16 before FP32 normalization"
        )
    source = key.detach()
    normalized = source * torch.rsqrt((source * source).sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
    return normalized.to(torch.float32)


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    normalized_key: torch.Tensor
    update: torch.Tensor
    log_decay: torch.Tensor


def dense_transition_from_record(
    initial_state: torch.Tensor,
    record: ReplayRecord,
) -> torch.Tensor:
    state = _require_tensor(initial_state, name="initial_state", ndim=4).to(torch.float32)
    key = _require_tensor(
        record.normalized_key,
        name="normalized_key",
        ndim=3,
    ).to(torch.float32)
    update = _require_tensor(record.update, name="update", ndim=3).to(torch.float32)
    log_decay = _require_tensor(
        record.log_decay,
        name="log_decay",
        ndim=2,
    ).to(torch.float32)
    batch, heads, rows, width = state.shape
    if tuple(key.shape) != (batch, heads, rows):
        raise ValueError("normalized_key shape is incompatible with initial_state")
    if tuple(update.shape) != (batch, heads, width):
        raise ValueError("update shape is incompatible with initial_state")
    if tuple(log_decay.shape) != (batch, heads):
        raise ValueError("log_decay shape is incompatible with initial_state")
    return state * log_decay.exp().unsqueeze(-1).unsqueeze(-1) + key.unsqueeze(
        -1
    ) * update.unsqueeze(-2)


def derive_successful_record(
    *,
    initial_state: torch.Tensor,
    consumed_key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    successful_final_state: torch.Tensor,
) -> ReplayRecord:
    """Derive the exact post-correction record and verify the successful write."""

    state = _require_tensor(initial_state, name="initial_state", ndim=4).to(torch.float32)
    key = _require_tensor(consumed_key, name="consumed_key", ndim=4)
    value = _require_tensor(value, name="value", ndim=4).to(torch.float32)
    decay = _require_tensor(log_decay, name="log_decay", ndim=3).to(torch.float32)
    write = _require_tensor(beta, name="beta", ndim=3).to(torch.float32)
    final = _require_tensor(
        successful_final_state,
        name="successful_final_state",
        ndim=4,
    ).to(torch.float32)
    batch, heads, rows, width = state.shape
    if batch != 1:
        raise ValueError("Experiment 010 replay verification requires batch size one")
    if tuple(key.shape) != (batch, 1, heads, rows):
        raise ValueError("consumed_key must have shape [1, 1, heads, rows]")
    if tuple(value.shape) != (batch, 1, heads, width):
        raise ValueError("value must have shape [1, 1, heads, width]")
    if tuple(decay.shape) != (batch, 1, heads):
        raise ValueError("log_decay must have shape [1, 1, heads]")
    if tuple(write.shape) != (batch, 1, heads):
        raise ValueError("beta must have shape [1, 1, heads]")
    if tuple(final.shape) != tuple(state.shape):
        raise ValueError("successful_final_state shape differs from initial_state")
    if (decay > 0).any().item():
        raise ValueError("log_decay must be non-positive")
    if ((write < 0) | (write > 1)).any().item():
        raise ValueError("beta must lie in [0, 1]")

    normalized_key = normalize_consumed_key(key)[:, 0]
    decay_token = decay[:, 0]
    decayed = state * decay_token.exp().unsqueeze(-1).unsqueeze(-1)
    remembered = (decayed * normalized_key.unsqueeze(-1)).sum(dim=-2)
    update = (value[:, 0] - remembered) * write[:, 0].unsqueeze(-1)
    record = ReplayRecord(
        normalized_key=normalized_key,
        update=update,
        log_decay=decay_token,
    )
    reconstructed = dense_transition_from_record(state, record)
    assert_replay_matches(final, reconstructed)
    return record


def dense_replay(
    checkpoint: torch.Tensor,
    records: Sequence[ReplayRecord],
) -> torch.Tensor:
    state = _require_tensor(checkpoint, name="checkpoint", ndim=4).to(torch.float32)
    for record in records:
        state = dense_transition_from_record(state, record)
    return state


@dataclass(frozen=True, slots=True)
class ReplayBuffers:
    normalized_keys: torch.Tensor
    updates: torch.Tensor
    log_decays: torch.Tensor
    valid_count: torch.Tensor


def store_records_bf16(
    records: Sequence[ReplayRecord],
    *,
    capacity: int = REPLAY_CAPACITY,
) -> ReplayBuffers:
    if not 0 <= len(records) <= capacity:
        raise ValueError("record count is outside replay capacity")
    if not records:
        raise ValueError("at least one record is required to infer buffer geometry")
    first = records[0]
    batch, heads, rows = first.normalized_key.shape
    update_shape = first.update.shape
    if batch != 1 or update_shape[:2] != (batch, heads):
        raise ValueError("replay records require batch-one compatible geometry")
    width = update_shape[-1]
    device = first.normalized_key.device
    keys = torch.zeros((capacity, heads, rows), dtype=torch.bfloat16, device=device)
    updates = torch.zeros((capacity, heads, width), dtype=torch.bfloat16, device=device)
    decays = torch.zeros((capacity, heads), dtype=torch.float32, device=device)
    for index, record in enumerate(records):
        if (
            tuple(record.normalized_key.shape) != (1, heads, rows)
            or tuple(record.update.shape) != (1, heads, width)
            or tuple(record.log_decay.shape) != (1, heads)
        ):
            raise ValueError("replay record geometry changed within one buffer")
        keys[index].copy_(record.normalized_key[0].to(torch.bfloat16))
        updates[index].copy_(record.update[0].to(torch.bfloat16))
        decays[index].copy_(record.log_decay[0].to(torch.float32))
    count = torch.tensor([len(records)], dtype=torch.int32, device=device)
    return ReplayBuffers(keys, updates, decays, count)


def replay_stored_buffers(
    checkpoint: torch.Tensor,
    buffers: ReplayBuffers,
) -> torch.Tensor:
    keys = _require_tensor(
        buffers.normalized_keys,
        name="normalized_keys",
        dtype=torch.bfloat16,
        ndim=3,
    )
    updates = _require_tensor(
        buffers.updates,
        name="updates",
        dtype=torch.bfloat16,
        ndim=3,
    )
    decays = _require_tensor(
        buffers.log_decays,
        name="log_decays",
        dtype=torch.float32,
        ndim=2,
    )
    count_tensor = _require_tensor(
        buffers.valid_count,
        name="valid_count",
        dtype=torch.int32,
        ndim=1,
        finite=False,
    )
    if tuple(count_tensor.shape) != (1,):
        raise ValueError("valid_count must have shape [1]")
    count = int(count_tensor.item())
    if not 0 <= count <= keys.shape[0]:
        raise ValueError("valid_count is outside allocated storage")
    if keys.shape[0] != updates.shape[0] or keys.shape[0] != decays.shape[0]:
        raise ValueError("buffer capacities differ")
    records = [
        ReplayRecord(
            normalized_key=keys[index].to(torch.float32).unsqueeze(0),
            update=updates[index].to(torch.float32).unsqueeze(0),
            log_decay=decays[index].unsqueeze(0),
        )
        for index in range(count)
    ]
    return dense_replay(checkpoint, records)


def compact_full_buffer(
    buffers: ReplayBuffers,
    *,
    boundary: int,
) -> ReplayBuffers:
    if boundary not in (4, 5):
        raise ValueError("boundary must be 4 or 5")
    if int(buffers.valid_count.item()) != 5:
        raise ValueError("boundary compaction requires an explicit full H5 buffer")
    keys = buffers.normalized_keys.clone()
    updates = buffers.updates.clone()
    decays = buffers.log_decays.clone()
    if boundary == 4:
        retained_key = keys[4].clone()
        retained_update = updates[4].clone()
        retained_decay = decays[4].clone()
        keys.zero_()
        updates.zero_()
        decays.zero_()
        keys[0].copy_(retained_key)
        updates[0].copy_(retained_update)
        decays[0].copy_(retained_decay)
        count = 1
    else:
        keys.zero_()
        updates.zero_()
        decays.zero_()
        count = 0
    return ReplayBuffers(
        normalized_keys=keys,
        updates=updates,
        log_decays=decays,
        valid_count=torch.tensor(
            [count],
            dtype=torch.int32,
            device=buffers.valid_count.device,
        ),
    )


def _portable_sign_values(
    *,
    layer_index: int,
    head_index: int,
    width: int,
) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < width:
        message = b"".join(
            (
                SIGN_DOMAIN,
                FROZEN_SEED.to_bytes(8, "little", signed=False),
                layer_index.to_bytes(8, "little", signed=False),
                head_index.to_bytes(8, "little", signed=False),
                width.to_bytes(8, "little", signed=False),
                counter.to_bytes(8, "little", signed=False),
            )
        )
        digest = hashlib.sha256(message).digest()
        for byte in digest:
            for bit_index in range(8):
                values.append(1.0 if byte & (1 << bit_index) else -1.0)
                if len(values) == width:
                    return values
        counter += 1
    return values


def independent_rht_signs(
    *,
    layer_index: int,
    heads: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    if layer_index < 0 or heads <= 0 or width <= 0 or width & (width - 1):
        raise ValueError("invalid randomized-Hadamard geometry")
    values = [
        _portable_sign_values(
            layer_index=layer_index,
            head_index=head_index,
            width=width,
        )
        for head_index in range(heads)
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(
        1,
        heads,
        1,
        width,
    )


def _independent_fwht(value: torch.Tensor) -> torch.Tensor:
    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = value.to(torch.float32).reshape(-1, width)
    stride = 1
    while stride < width:
        blocks = result.reshape(-1, width // (2 * stride), 2 * stride)
        left = blocks[..., :stride]
        right = blocks[..., stride:]
        result = torch.cat((left + right, left - right), dim=-1).reshape(-1, width)
        stride *= 2
    return result.reshape(value.shape)


def independent_rht_encode(state: torch.Tensor, *, layer_index: int) -> torch.Tensor:
    state = _require_tensor(state, name="state", ndim=4).to(torch.float32)
    heads = state.shape[1]
    width = state.shape[-1]
    signs = independent_rht_signs(
        layer_index=layer_index,
        heads=heads,
        width=width,
        device=state.device,
    )
    return _independent_fwht(state * signs) / math.sqrt(width)


def independent_rht_decode(encoded: torch.Tensor, *, layer_index: int) -> torch.Tensor:
    encoded = _require_tensor(encoded, name="encoded", ndim=4).to(torch.float32)
    heads = encoded.shape[1]
    width = encoded.shape[-1]
    signs = independent_rht_signs(
        layer_index=layer_index,
        heads=heads,
        width=width,
        device=encoded.device,
    )
    return (_independent_fwht(encoded) / math.sqrt(width)) * signs


def _quantize_rows(rows: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _require_tensor(rows, name="rows", ndim=2).to(torch.float32)
    if bits not in (4, 6, 8):
        raise ValueError("row precision must be 4, 6, or 8")
    qmax = float((1 << (bits - 1)) - 1)
    absmax = rows.abs().amax(dim=-1, keepdim=True)
    ideal = torch.where(absmax > 1e-12, absmax / qmax, torch.ones_like(absmax))
    ideal = ideal.clamp(min=2.0**-24, max=torch.finfo(torch.float16).max)
    scales = ideal.to(torch.float16).squeeze(-1).contiguous()
    codes = torch.round(rows / scales.to(torch.float32).unsqueeze(-1)).clamp(
        -qmax,
        qmax,
    )
    return codes.to(torch.int16).contiguous(), scales


def _pack_signed_rows(codes: torch.Tensor, bits: int) -> torch.Tensor:
    codes = _require_tensor(codes, name="codes", ndim=2, finite=False)
    if bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6, or 8")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    codes16 = codes.to(device="cpu", dtype=torch.int16).contiguous()
    if ((codes16 < minimum) | (codes16 > maximum)).any().item():
        raise ValueError("signed code lies outside the requested bit width")
    bytes_per_row = math.ceil(codes16.shape[1] * bits / 8)
    packed = torch.zeros((codes16.shape[0], bytes_per_row), dtype=torch.uint8)
    mask = (1 << bits) - 1
    for row_index, row in enumerate(codes16.tolist()):
        accumulator = 0
        available = 0
        byte_index = 0
        for signed in row:
            accumulator |= (signed & mask) << available
            available += bits
            while available >= 8:
                packed[row_index, byte_index] = accumulator & 0xFF
                accumulator >>= 8
                available -= 8
                byte_index += 1
        if available:
            packed[row_index, byte_index] = accumulator & 0xFF
    return packed.contiguous()


def _unpack_signed_rows(
    payload: torch.Tensor,
    *,
    bits: int,
    width: int,
) -> torch.Tensor:
    payload = _require_tensor(
        payload,
        name="payload",
        dtype=torch.uint8,
        ndim=2,
        finite=False,
    )
    expected = math.ceil(width * bits / 8)
    if payload.shape[1] != expected:
        raise ValueError("payload row width does not match signed code geometry")
    output = torch.empty((payload.shape[0], width), dtype=torch.int16)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    modulus = 1 << bits
    for row_index, row in enumerate(payload.to("cpu").tolist()):
        accumulator = 0
        available = 0
        byte_index = 0
        for code_index in range(width):
            while available < bits:
                accumulator |= row[byte_index] << available
                available += 8
                byte_index += 1
            unsigned = accumulator & mask
            accumulator >>= bits
            available -= bits
            output[row_index, code_index] = unsigned - modulus if unsigned & sign_bit else unsigned
    return output.to(payload.device)


def _pack_small_codes(codes: torch.Tensor, *, bits: int) -> torch.Tensor:
    flat = _require_tensor(codes, name="codes", ndim=1, finite=False).to(
        device="cpu",
        dtype=torch.uint8,
    )
    maximum = (1 << bits) - 1
    if (flat > maximum).any().item():
        raise ValueError("precision code exceeds packed bit width")
    per_byte = 8 // bits
    padding = (-flat.numel()) % per_byte
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    chunks = flat.reshape(-1, per_byte).to(torch.int16)
    shifts = torch.arange(per_byte, dtype=torch.int16) * bits
    packed = torch.bitwise_left_shift(chunks, shifts).sum(dim=1)
    return packed.to(torch.uint8).contiguous()


def _unpack_small_codes(
    packed: torch.Tensor,
    *,
    bits: int,
    count: int,
) -> torch.Tensor:
    packed = _require_tensor(
        packed,
        name="packed",
        dtype=torch.uint8,
        ndim=1,
        finite=False,
    )
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, dtype=torch.int16, device=packed.device) * bits
    expanded = torch.bitwise_right_shift(packed.to(torch.int16).unsqueeze(1), shifts)
    return torch.bitwise_and(expanded, (1 << bits) - 1).reshape(-1)[:count].to(torch.uint8)


def _validate_code_stream_padding(
    packed: torch.Tensor,
    *,
    bits: int,
    count: int,
) -> None:
    per_byte = 8 // bits
    full_count = packed.numel() * per_byte
    if full_count < count:
        raise ValueError("packed precision stream is too short")
    all_codes = _unpack_small_codes(packed, bits=bits, count=full_count)
    if (all_codes[count:] != 0).any().item():
        _fail("packed precision stream has nonzero padding codes")


def _validate_physical_storage(
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> None:
    devices: set[torch.device] = set()
    identities: set[tuple[str, int | None, int]] = set()
    for name, tensor in tensors:
        _require_tensor(tensor, name=name)
        if not tensor.is_contiguous():
            _fail(f"{name} is not contiguous")
        devices.add(tensor.device)
        identity = _storage_identity(tensor)
        if tensor.numel() and identity in identities:
            _fail(f"{name} aliases another physical codec tensor")
        identities.add(identity)
    if len(devices) != 1:
        _fail("physical codec tensors do not share one device")


@dataclass(frozen=True, slots=True)
class PhysicalQ4Q8:
    low_payload: torch.Tensor
    high_payload: torch.Tensor
    scales: torch.Tensor
    precision_mask: torch.Tensor
    row_count: int
    width: int

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor_bytes(tensor)
            for tensor in (
                self.low_payload,
                self.high_payload,
                self.scales,
                self.precision_mask,
            )
        )

    def dequantize(self) -> torch.Tensor:
        _require_tensor(
            self.low_payload,
            name="low_payload",
            dtype=torch.uint8,
            ndim=2,
            finite=False,
        )
        _require_tensor(
            self.high_payload,
            name="high_payload",
            dtype=torch.int8,
            ndim=2,
            finite=False,
        )
        _require_tensor(
            self.scales,
            name="scales",
            dtype=torch.float16,
            ndim=1,
        )
        _require_tensor(
            self.precision_mask,
            name="precision_mask",
            dtype=torch.uint8,
            ndim=1,
            finite=False,
        )
        _validate_physical_storage(
            (
                ("low_payload", self.low_payload),
                ("high_payload", self.high_payload),
                ("scales", self.scales),
                ("precision_mask", self.precision_mask),
            )
        )
        if (
            self.row_count <= 0
            or self.width <= 0
            or self.low_payload.shape[1] != math.ceil(self.width * 4 / 8)
            or self.high_payload.shape[1] != self.width
            or self.scales.shape != (self.row_count,)
            or self.precision_mask.shape != (math.ceil(self.row_count / 8),)
        ):
            _fail("Q4/Q8 physical geometry is malformed")
        if (self.scales <= 0).any().item():
            _fail("Q4/Q8 scales must be positive")
        _validate_code_stream_padding(
            self.precision_mask,
            bits=1,
            count=self.row_count,
        )
        mask = _unpack_small_codes(
            self.precision_mask,
            bits=1,
            count=self.row_count,
        ).bool()
        if int(mask.sum().item()) != self.high_payload.shape[0]:
            _fail("Q4/Q8 precision mask and high-payload pool disagree")
        if int((~mask).sum().item()) != self.low_payload.shape[0]:
            _fail("Q4/Q8 precision mask and low-payload pool disagree")
        codes = torch.empty(
            (self.row_count, self.width),
            dtype=torch.int16,
            device=self.scales.device,
        )
        codes[~mask] = _unpack_signed_rows(
            self.low_payload,
            bits=4,
            width=self.width,
        )
        codes[mask] = self.high_payload.to(torch.int16)
        return codes.to(torch.float32) * self.scales.to(torch.float32).unsqueeze(-1)


def pack_physical_q4_q8(
    rows: torch.Tensor,
    high_precision_mask: torch.Tensor,
) -> PhysicalQ4Q8:
    rows = _require_tensor(rows, name="rows", ndim=2).to(torch.float32)
    mask = _require_tensor(
        high_precision_mask,
        name="high_precision_mask",
        dtype=torch.bool,
        ndim=1,
        finite=False,
    ).to(rows.device)
    if mask.numel() != rows.shape[0]:
        raise ValueError("high_precision_mask length differs from row count")
    low_codes, low_scales = _quantize_rows(rows[~mask], 4)
    high_codes, high_scales = _quantize_rows(rows[mask], 8)
    scales = torch.empty((rows.shape[0],), dtype=torch.float16, device=rows.device)
    scales[~mask] = low_scales
    scales[mask] = high_scales
    return PhysicalQ4Q8(
        low_payload=_pack_signed_rows(low_codes, 4).to(rows.device),
        high_payload=high_codes.to(torch.int8).contiguous(),
        scales=scales.contiguous(),
        precision_mask=_pack_small_codes(mask.to(torch.uint8), bits=1).to(rows.device),
        row_count=rows.shape[0],
        width=rows.shape[1],
    )


def stable_descending_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
    scores = _require_tensor(scores, name="scores", ndim=1).to(torch.float64).cpu()
    if not 0 <= count <= scores.numel():
        raise ValueError("selection count is outside score length")
    order = sorted(range(scores.numel()), key=lambda index: (-float(scores[index]), index))
    return torch.tensor(order[:count], dtype=torch.long)


def q4_q8_physical_benefit(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    rows = _require_tensor(rows, name="rows", ndim=2).to(torch.float32)
    weights = _require_tensor(weights, name="weights", ndim=1).to(torch.float32)
    if weights.shape != (rows.shape[0],):
        raise ValueError("weights length differs from row count")
    q4_codes, q4_scales = _quantize_rows(rows, 4)
    q8_codes, q8_scales = _quantize_rows(rows, 8)
    q4 = q4_codes.float() * q4_scales.float().unsqueeze(-1)
    q8 = q8_codes.float() * q8_scales.float().unsqueeze(-1)
    return weights * ((q4 - rows).square().mean(-1) - (q8 - rows).square().mean(-1))


@dataclass(frozen=True, slots=True)
class PhysicalQ4Q6Q8:
    q4_payload: torch.Tensor
    q6_payload: torch.Tensor
    q8_payload: torch.Tensor
    scales: torch.Tensor
    precision_codes: torch.Tensor
    row_count: int
    width: int

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor_bytes(tensor)
            for tensor in (
                self.q4_payload,
                self.q6_payload,
                self.q8_payload,
                self.scales,
                self.precision_codes,
            )
        )

    def dequantize(self) -> torch.Tensor:
        for name, payload, dtype in (
            ("q4_payload", self.q4_payload, torch.uint8),
            ("q6_payload", self.q6_payload, torch.uint8),
            ("q8_payload", self.q8_payload, torch.int8),
        ):
            _require_tensor(
                payload,
                name=name,
                dtype=dtype,
                ndim=2,
                finite=False,
            )
        _require_tensor(
            self.scales,
            name="scales",
            dtype=torch.float16,
            ndim=1,
        )
        _require_tensor(
            self.precision_codes,
            name="precision_codes",
            dtype=torch.uint8,
            ndim=1,
            finite=False,
        )
        _validate_physical_storage(
            (
                ("q4_payload", self.q4_payload),
                ("q6_payload", self.q6_payload),
                ("q8_payload", self.q8_payload),
                ("scales", self.scales),
                ("precision_codes", self.precision_codes),
            )
        )
        if (
            self.row_count <= 0
            or self.width <= 0
            or self.q4_payload.shape[1] != math.ceil(self.width * 4 / 8)
            or self.q6_payload.shape[1] != math.ceil(self.width * 6 / 8)
            or self.q8_payload.shape[1] != self.width
            or self.scales.shape != (self.row_count,)
            or self.precision_codes.shape != (math.ceil(self.row_count * 2 / 8),)
        ):
            _fail("Q4/Q6/Q8 physical geometry is malformed")
        if (self.scales <= 0).any().item():
            _fail("Q4/Q6/Q8 scales must be positive")
        _validate_code_stream_padding(
            self.precision_codes,
            bits=2,
            count=self.row_count,
        )
        precision = _unpack_small_codes(
            self.precision_codes,
            bits=2,
            count=self.row_count,
        )
        if (precision == 3).any().item():
            _fail("invalid Q4/Q6/Q8 precision code 3")
        output = torch.empty(
            (self.row_count, self.width),
            dtype=torch.int16,
            device=self.scales.device,
        )
        for code, bits, payload in (
            (0, 4, self.q4_payload),
            (1, 6, self.q6_payload),
            (2, 8, self.q8_payload),
        ):
            selected = precision == code
            expected = int(selected.sum().item())
            if payload.shape[0] != expected:
                _fail(f"Q{bits} payload pool and precision stream disagree")
            unpacked = (
                payload.to(torch.int16)
                if bits == 8
                else _unpack_signed_rows(payload, bits=bits, width=self.width)
            )
            output[selected] = unpacked
        return output.float() * self.scales.float().unsqueeze(-1)


def allocate_exact_multibit(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Small independent exact DP; exact ties favor earlier-row precision."""

    tensors = [
        _require_tensor(value, name=name, ndim=1).to(torch.float64).cpu()
        for value, name in ((d4, "d4"), (d6, "d6"), (d8, "d8"))
    ]
    if tensors[0].shape != tensors[1].shape or tensors[0].shape != tensors[2].shape:
        raise ValueError("multibit distortion shapes differ")
    rows = tensors[0].numel()
    if not 0 <= marginal_steps <= 2 * rows:
        raise ValueError("marginal_steps is outside [0, 2 * rows]")
    if rows * (marginal_steps + 1) > 2_000_000:
        raise ValueError("independent DP is intentionally limited to Stage-0 audit slices")
    costs = torch.stack(tensors, dim=1)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row_index in range(rows):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (cost, prefix) in states.items():
            for code in (0, 1, 2):
                total_steps = used + code
                if total_steps > marginal_steps:
                    continue
                candidate = (
                    cost + float(costs[row_index, code].item()),
                    (*prefix, code),
                )
                incumbent = next_states.get(total_steps)
                if (
                    incumbent is None
                    or candidate[0] < incumbent[0]
                    or (candidate[0] == incumbent[0] and candidate[1] > incumbent[1])
                ):
                    next_states[total_steps] = candidate
        states = next_states
    if marginal_steps not in states:
        _fail("multibit DP did not reach its exact marginal-step budget")
    return torch.tensor(states[marginal_steps][1], dtype=torch.uint8)


def pack_physical_q4_q6_q8(
    rows: torch.Tensor,
    precision: torch.Tensor,
) -> PhysicalQ4Q6Q8:
    rows = _require_tensor(rows, name="rows", ndim=2).to(torch.float32)
    precision = _require_tensor(
        precision,
        name="precision",
        dtype=torch.uint8,
        ndim=1,
        finite=False,
    ).to(rows.device)
    if precision.numel() != rows.shape[0] or (precision > 2).any().item():
        raise ValueError("precision must contain one Q4/Q6/Q8 code per row")
    payloads: dict[int, torch.Tensor] = {}
    scales = torch.empty((rows.shape[0],), dtype=torch.float16, device=rows.device)
    for code, bits in ((0, 4), (1, 6), (2, 8)):
        selected = precision == code
        codes, selected_scales = _quantize_rows(rows[selected], bits)
        scales[selected] = selected_scales
        payloads[bits] = (
            codes.to(torch.int8).contiguous()
            if bits == 8
            else _pack_signed_rows(codes, bits).to(rows.device)
        )
    return PhysicalQ4Q6Q8(
        q4_payload=payloads[4],
        q6_payload=payloads[6],
        q8_payload=payloads[8],
        scales=scales.contiguous(),
        precision_codes=_pack_small_codes(precision.cpu(), bits=2).to(rows.device),
        row_count=rows.shape[0],
        width=rows.shape[1],
    )


@dataclass(frozen=True, slots=True)
class PhysicalResidualQ4:
    base_payload: torch.Tensor
    base_scales: torch.Tensor
    residual_mask: torch.Tensor
    residual_payload: torch.Tensor
    residual_scales: torch.Tensor
    row_count: int
    width: int

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor_bytes(tensor)
            for tensor in (
                self.base_payload,
                self.base_scales,
                self.residual_mask,
                self.residual_payload,
                self.residual_scales,
            )
        )

    def dequantize(self) -> torch.Tensor:
        for name, payload in (
            ("base_payload", self.base_payload),
            ("residual_payload", self.residual_payload),
        ):
            _require_tensor(
                payload,
                name=name,
                dtype=torch.uint8,
                ndim=2,
                finite=False,
            )
        for name, scales in (
            ("base_scales", self.base_scales),
            ("residual_scales", self.residual_scales),
        ):
            _require_tensor(
                scales,
                name=name,
                dtype=torch.float16,
                ndim=1,
            )
        _require_tensor(
            self.residual_mask,
            name="residual_mask",
            dtype=torch.uint8,
            ndim=1,
            finite=False,
        )
        _validate_physical_storage(
            (
                ("base_payload", self.base_payload),
                ("base_scales", self.base_scales),
                ("residual_mask", self.residual_mask),
                ("residual_payload", self.residual_payload),
                ("residual_scales", self.residual_scales),
            )
        )
        if (
            self.row_count <= 0
            or self.width <= 0
            or self.base_payload.shape != (self.row_count, math.ceil(self.width * 4 / 8))
            or self.base_scales.shape != (self.row_count,)
            or self.residual_payload.shape[1] != math.ceil(self.width * 4 / 8)
            or self.residual_payload.shape[0] != self.residual_scales.shape[0]
            or self.residual_mask.shape != (math.ceil(self.row_count / 8),)
        ):
            _fail("residual-Q4 physical geometry is malformed")
        if (self.base_scales <= 0).any().item() or (self.residual_scales <= 0).any().item():
            _fail("residual-Q4 scales must be positive")
        _validate_code_stream_padding(
            self.residual_mask,
            bits=1,
            count=self.row_count,
        )
        base_codes = _unpack_signed_rows(self.base_payload, bits=4, width=self.width)
        result = base_codes.float() * self.base_scales.float().unsqueeze(-1)
        selected = _unpack_small_codes(
            self.residual_mask,
            bits=1,
            count=self.row_count,
        ).bool()
        if int(selected.sum().item()) != self.residual_payload.shape[0]:
            _fail("residual mask and residual payload pool disagree")
        residual_codes = _unpack_signed_rows(
            self.residual_payload,
            bits=4,
            width=self.width,
        )
        result[selected] += residual_codes.float() * self.residual_scales.float().unsqueeze(-1)
        return result


def pack_physical_residual_q4(
    rows: torch.Tensor,
    selected_mask: torch.Tensor,
) -> PhysicalResidualQ4:
    rows = _require_tensor(rows, name="rows", ndim=2).to(torch.float32)
    selected = _require_tensor(
        selected_mask,
        name="selected_mask",
        dtype=torch.bool,
        ndim=1,
        finite=False,
    ).to(rows.device)
    if selected.numel() != rows.shape[0]:
        raise ValueError("selected_mask length differs from row count")
    base_codes, base_scales = _quantize_rows(rows, 4)
    base = base_codes.float() * base_scales.float().unsqueeze(-1)
    residual_codes, residual_scales = _quantize_rows((rows - base)[selected], 4)
    return PhysicalResidualQ4(
        base_payload=_pack_signed_rows(base_codes, 4).to(rows.device),
        base_scales=base_scales,
        residual_mask=_pack_small_codes(selected.to(torch.uint8), bits=1).to(rows.device),
        residual_payload=_pack_signed_rows(residual_codes, 4).to(rows.device),
        residual_scales=residual_scales,
        row_count=rows.shape[0],
        width=rows.shape[1],
    )


def normalized_query_weights(query_ema: torch.Tensor) -> torch.Tensor:
    ema = _require_tensor(
        query_ema,
        name="query_ema",
        dtype=torch.float32,
        ndim=2,
    )
    row_sums = ema.sum(dim=-1, keepdim=True)
    if (row_sums <= 0).any().item() or not torch.isfinite(row_sums).all().item():
        raise ValueError("query_ema must have positive finite per-head mass")
    weights = ema / row_sums
    if not torch.isfinite(weights).all().item():
        raise ValueError("normalized query weights are non-finite")
    return weights


def query_weighted_handoff_risk(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    normalized_weights: torch.Tensor,
) -> torch.Tensor:
    reference = _require_tensor(reference, name="reference", ndim=4).to(torch.float32)
    approximation = _require_tensor(
        approximation,
        name="approximation",
        ndim=4,
    ).to(torch.float32)
    weights = _require_tensor(
        normalized_weights,
        name="normalized_weights",
        dtype=torch.float32,
        ndim=2,
    )
    if reference.shape != approximation.shape or reference.shape[0] != 1:
        raise ValueError("risk states must share batch-one geometry")
    if tuple(weights.shape) != tuple(reference.shape[1:3]):
        raise ValueError("risk weights do not match state head/row geometry")
    expected_mass = torch.ones(
        weights.shape[0],
        dtype=torch.float32,
        device=weights.device,
    )
    actual_mass = weights.sum(dim=-1)
    if not torch.equal(actual_mass, expected_mass) and not torch.allclose(
        actual_mass,
        expected_mass,
        rtol=0,
        atol=2e-6,
    ):
        raise ValueError("risk weights must sum to one per head")
    row_mse = (approximation - reference).square().mean(dim=-1).squeeze(0)
    risk = (weights * row_mse).sum(dim=-1).mean().to(torch.float32)
    if not torch.isfinite(risk).item() or risk.item() < 0:
        _fail("handoff risk is invalid")
    return risk


@dataclass(frozen=True, slots=True)
class BoundaryAudit:
    boundary: int
    cut4_candidate: torch.Tensor
    cut5_candidate: torch.Tensor
    cut4_risk: float
    cut5_risk: float
    tie: bool


def construct_and_choose_boundary(
    *,
    s4: torch.Tensor,
    raw_s5: torch.Tensor,
    record5: ReplayRecord,
    query_ema: torch.Tensor,
    pack_unpack: Callable[[torch.Tensor], torch.Tensor],
) -> BoundaryAudit:
    """Build both candidates from the same S5 and one shared weight view."""

    s4 = _require_tensor(s4, name="s4", ndim=4).to(torch.float32)
    raw_s5 = _require_tensor(raw_s5, name="raw_s5", ndim=4).to(torch.float32)
    direct_s5 = dense_transition_from_record(s4, record5)
    assert_replay_matches(raw_s5, direct_s5)
    shared_weights = normalized_query_weights(query_ema)
    cut4_checkpoint = pack_unpack(s4)
    cut4 = dense_transition_from_record(
        cut4_checkpoint,
        ReplayRecord(
            normalized_key=record5.normalized_key.to(torch.bfloat16).to(torch.float32),
            update=record5.update.to(torch.bfloat16).to(torch.float32),
            log_decay=record5.log_decay.to(torch.float32),
        ),
    )
    cut5 = pack_unpack(raw_s5)
    cut4_risk_tensor = query_weighted_handoff_risk(raw_s5, cut4, shared_weights)
    cut5_risk_tensor = query_weighted_handoff_risk(raw_s5, cut5, shared_weights)
    cut4_risk = float(cut4_risk_tensor.item())
    cut5_risk = float(cut5_risk_tensor.item())
    tie = cut4_risk == cut5_risk
    return BoundaryAudit(
        boundary=4 if cut4_risk < cut5_risk else 5,
        cut4_candidate=cut4,
        cut5_candidate=cut5,
        cut4_risk=cut4_risk,
        cut5_risk=cut5_risk,
        tie=tie,
    )


def frozen_storage_contract() -> dict[str, int | float]:
    payload = LOW_PRECISION_ROWS * 64 + HIGH_PRECISION_ROWS * 128
    scales = STATE_ROWS * 2
    mask = math.ceil(STATE_ROWS / 8)
    checkpoint = payload + scales + mask
    query_ema = LAYERS * HEADS * ROWS * 4
    key_buffer = LAYERS * REPLAY_CAPACITY * HEADS * ROWS * 2
    update_buffer = LAYERS * REPLAY_CAPACITY * HEADS * WIDTH * 2
    decay_buffer = LAYERS * REPLAY_CAPACITY * HEADS * 4
    counts = LAYERS * 4
    total = checkpoint + query_ema + key_buffer + update_buffer + decay_buffer + counts
    expected = (
        (payload, CHECKPOINT_PAYLOAD_BYTES, "checkpoint payload"),
        (scales, CHECKPOINT_SCALE_BYTES, "checkpoint scales"),
        (mask, CHECKPOINT_MASK_BYTES, "checkpoint mask"),
        (checkpoint, CHECKPOINT_BYTES, "checkpoint"),
        (query_ema, QUERY_EMA_BYTES, "query EMA"),
        (key_buffer, KEY_BUFFER_BYTES, "key buffer"),
        (update_buffer, UPDATE_BUFFER_BYTES, "update buffer"),
        (decay_buffer, LOG_DECAY_BUFFER_BYTES, "log-decay buffer"),
        (counts, COUNT_BYTES, "valid counts"),
        (total, STATELEASE_BYTES, "StateLease total"),
    )
    for actual, frozen, name in expected:
        if actual != frozen:
            _fail(f"{name} arithmetic drifted: {actual} != {frozen}")
    return {
        "state_elements": STATE_ELEMENTS,
        "fp32_state_bytes": STATE_ELEMENTS * 4,
        "checkpoint_payload_bytes": payload,
        "checkpoint_scale_bytes": scales,
        "checkpoint_mask_bytes": mask,
        "checkpoint_bytes": checkpoint,
        "query_ema_bytes": query_ema,
        "normalized_key_buffer_bytes": key_buffer,
        "update_buffer_bytes": update_buffer,
        "log_decay_buffer_bytes": decay_buffer,
        "count_bytes": counts,
        "resident_bytes": total,
        "bits_per_state_element": total * 8 / STATE_ELEMENTS,
        "fp32_compression_ratio": (STATE_ELEMENTS * 4) / total,
    }


def equal_byte_codec_contracts() -> dict[str, dict[str, int]]:
    q48_added = EXPANDED_Q48_PROMOTIONS * 64
    q48_total = EXPANDED_Q48_BASE_BYTES + q48_added + EXPANDED_Q48_PADDING_BYTES
    multibit_added = MULTIBIT_MARGINAL_STEPS * 32
    multibit_total = MULTIBIT_BASE_BYTES + multibit_added + MULTIBIT_PADDING_BYTES
    residual_added = RESIDUAL_Q4_ROWS * 66
    residual_total = RESIDUAL_Q4_BASE_BYTES + residual_added + RESIDUAL_Q4_PADDING_BYTES
    contracts = {
        "expanded_q4_q8": {
            "base_bytes": EXPANDED_Q48_BASE_BYTES,
            "allocation_units": EXPANDED_Q48_PROMOTIONS,
            "bytes_per_unit": 64,
            "added_bytes": q48_added,
            "padding_bytes": EXPANDED_Q48_PADDING_BYTES,
            "total_bytes": q48_total,
        },
        "q4_q6_q8": {
            "base_bytes": MULTIBIT_BASE_BYTES,
            "allocation_units": MULTIBIT_MARGINAL_STEPS,
            "bytes_per_unit": 32,
            "added_bytes": multibit_added,
            "padding_bytes": MULTIBIT_PADDING_BYTES,
            "total_bytes": multibit_total,
        },
        "residual_q4": {
            "base_bytes": RESIDUAL_Q4_BASE_BYTES,
            "allocation_units": RESIDUAL_Q4_ROWS,
            "bytes_per_unit": 66,
            "added_bytes": residual_added,
            "padding_bytes": RESIDUAL_Q4_PADDING_BYTES,
            "total_bytes": residual_total,
        },
    }
    for name, contract in contracts.items():
        if contract["total_bytes"] != STATELEASE_BYTES:
            _fail(f"{name} does not spend exactly {STATELEASE_BYTES} bytes")
    return contracts


@dataclass(frozen=True, slots=True)
class ResidentSnapshot:
    checkpoint_low_payloads: tuple[torch.Tensor, ...]
    checkpoint_high_payloads: tuple[torch.Tensor, ...]
    checkpoint_scales: tuple[torch.Tensor, ...]
    checkpoint_masks: tuple[torch.Tensor, ...]
    query_emas: tuple[torch.Tensor, ...]
    normalized_key_buffers: tuple[torch.Tensor, ...]
    update_buffers: tuple[torch.Tensor, ...]
    log_decay_buffers: tuple[torch.Tensor, ...]
    valid_counts: tuple[torch.Tensor, ...]
    extra_persistent_tensors: tuple[tuple[str, torch.Tensor], ...] = ()


def canonical_empty_resident_snapshot() -> ResidentSnapshot:
    """Allocate one closed-schema synthetic snapshot at the frozen geometry."""

    precision = torch.zeros((STATE_ROWS,), dtype=torch.uint8)
    precision[:HIGH_PRECISION_ROWS] = 1
    return ResidentSnapshot(
        checkpoint_low_payloads=(torch.zeros((LOW_PRECISION_ROWS, 64), dtype=torch.uint8),),
        checkpoint_high_payloads=(torch.zeros((HIGH_PRECISION_ROWS, WIDTH), dtype=torch.int8),),
        checkpoint_scales=(torch.ones((STATE_ROWS,), dtype=torch.float16),),
        checkpoint_masks=(_pack_small_codes(precision, bits=1),),
        query_emas=(torch.full((LAYERS, HEADS, ROWS), 1.0 / ROWS, dtype=torch.float32),),
        normalized_key_buffers=(
            torch.zeros(
                (LAYERS, REPLAY_CAPACITY, HEADS, ROWS),
                dtype=torch.bfloat16,
            ),
        ),
        update_buffers=(
            torch.zeros(
                (LAYERS, REPLAY_CAPACITY, HEADS, WIDTH),
                dtype=torch.bfloat16,
            ),
        ),
        log_decay_buffers=(torch.zeros((LAYERS, REPLAY_CAPACITY, HEADS), dtype=torch.float32),),
        valid_counts=(torch.zeros((LAYERS,), dtype=torch.int32),),
    )


def _component_numel(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)


def _validate_component(
    tensors: Sequence[torch.Tensor],
    *,
    name: str,
    dtype: torch.dtype,
    expected_numel: int,
    trailing_width: int | None = None,
) -> None:
    if not tensors:
        _fail(f"{name} is missing")
    for index, tensor in enumerate(tensors):
        _require_tensor(tensor, name=f"{name}[{index}]", dtype=dtype)
        if not tensor.is_contiguous():
            _fail(f"{name}[{index}] is not contiguous")
        if trailing_width is not None and (tensor.ndim != 2 or tensor.shape[-1] != trailing_width):
            _fail(f"{name}[{index}] has invalid payload geometry")
    actual = _component_numel(tensors)
    if actual != expected_numel:
        _fail(f"{name} has {actual} elements; expected {expected_numel}")


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    device_index = tensor.device.index
    return tensor.device.type, device_index, tensor.untyped_storage().data_ptr()


def _validate_layer_partitions(
    tensors: Sequence[torch.Tensor],
    *,
    name: str,
    per_layer_shape: tuple[int, ...],
) -> None:
    represented_layers = 0
    for index, tensor in enumerate(tensors):
        if tuple(tensor.shape) == per_layer_shape:
            represented_layers += 1
        elif tensor.ndim == len(per_layer_shape) + 1 and tuple(tensor.shape[1:]) == per_layer_shape:
            represented_layers += tensor.shape[0]
        else:
            _fail(
                f"{name}[{index}] must have one-layer shape {per_layer_shape} "
                "or a leading layer partition"
            )
    if represented_layers != LAYERS:
        _fail(f"{name} represents {represented_layers} layers; expected {LAYERS}")


def _validate_flat_layer_partitions(
    tensors: Sequence[torch.Tensor],
    *,
    name: str,
    values_per_layer: int,
) -> None:
    represented_layers = 0
    for index, tensor in enumerate(tensors):
        if tensor.ndim != 1 or tensor.numel() % values_per_layer:
            _fail(f"{name}[{index}] must be flat and contain a whole number of layer partitions")
        represented_layers += tensor.numel() // values_per_layer
    if represented_layers != LAYERS:
        _fail(f"{name} represents {represented_layers} layers; expected {LAYERS}")


def audit_resident_snapshot(snapshot: ResidentSnapshot) -> dict[str, int | float]:
    """Verify exact allocation, ownership, and absence of hidden tensor state."""

    if snapshot.extra_persistent_tensors:
        names = ", ".join(name for name, _ in snapshot.extra_persistent_tensors)
        _fail(f"unexpected persistent tensor fields: {names}")
    components: tuple[tuple[str, Sequence[torch.Tensor], torch.dtype, int, int | None], ...] = (
        (
            "checkpoint_low_payloads",
            snapshot.checkpoint_low_payloads,
            torch.uint8,
            LOW_PRECISION_ROWS * 64,
            64,
        ),
        (
            "checkpoint_high_payloads",
            snapshot.checkpoint_high_payloads,
            torch.int8,
            HIGH_PRECISION_ROWS * WIDTH,
            WIDTH,
        ),
        (
            "checkpoint_scales",
            snapshot.checkpoint_scales,
            torch.float16,
            STATE_ROWS,
            None,
        ),
        (
            "checkpoint_masks",
            snapshot.checkpoint_masks,
            torch.uint8,
            math.ceil(STATE_ROWS / 8),
            None,
        ),
        (
            "query_emas",
            snapshot.query_emas,
            torch.float32,
            LAYERS * HEADS * ROWS,
            None,
        ),
        (
            "normalized_key_buffers",
            snapshot.normalized_key_buffers,
            torch.bfloat16,
            LAYERS * REPLAY_CAPACITY * HEADS * ROWS,
            None,
        ),
        (
            "update_buffers",
            snapshot.update_buffers,
            torch.bfloat16,
            LAYERS * REPLAY_CAPACITY * HEADS * WIDTH,
            None,
        ),
        (
            "log_decay_buffers",
            snapshot.log_decay_buffers,
            torch.float32,
            LAYERS * REPLAY_CAPACITY * HEADS,
            None,
        ),
        (
            "valid_counts",
            snapshot.valid_counts,
            torch.int32,
            LAYERS,
            None,
        ),
    )
    all_tensors: list[tuple[str, torch.Tensor]] = []
    devices: set[torch.device] = set()
    for name, tensors, dtype, expected_numel, trailing_width in components:
        _validate_component(
            tensors,
            name=name,
            dtype=dtype,
            expected_numel=expected_numel,
            trailing_width=trailing_width,
        )
        for index, tensor in enumerate(tensors):
            all_tensors.append((f"{name}[{index}]", tensor))
            devices.add(tensor.device)
    _validate_flat_layer_partitions(
        snapshot.checkpoint_scales,
        name="checkpoint_scales",
        values_per_layer=HEADS * ROWS,
    )
    _validate_flat_layer_partitions(
        snapshot.checkpoint_masks,
        name="checkpoint_masks",
        values_per_layer=math.ceil(HEADS * ROWS / 8),
    )
    _validate_layer_partitions(
        snapshot.query_emas,
        name="query_emas",
        per_layer_shape=(HEADS, ROWS),
    )
    _validate_layer_partitions(
        snapshot.normalized_key_buffers,
        name="normalized_key_buffers",
        per_layer_shape=(REPLAY_CAPACITY, HEADS, ROWS),
    )
    _validate_layer_partitions(
        snapshot.update_buffers,
        name="update_buffers",
        per_layer_shape=(REPLAY_CAPACITY, HEADS, WIDTH),
    )
    _validate_layer_partitions(
        snapshot.log_decay_buffers,
        name="log_decay_buffers",
        per_layer_shape=(REPLAY_CAPACITY, HEADS),
    )
    _validate_flat_layer_partitions(
        snapshot.valid_counts,
        name="valid_counts",
        values_per_layer=1,
    )
    if len(devices) != 1:
        _fail("persistent tensors do not share one device")
    seen: dict[tuple[str, int | None, int], str] = {}
    for name, tensor in all_tensors:
        identity = _storage_identity(tensor)
        if identity in seen:
            _fail(f"persistent storage alias: {name} aliases {seen[identity]}")
        seen[identity] = name
        if tensor.storage_offset() != 0:
            _fail(f"{name} is a storage view with nonzero offset")
        if tensor.untyped_storage().nbytes() != tensor_bytes(tensor):
            _fail(f"{name} does not exclusively own its complete storage")
    counts = torch.cat([tensor.reshape(-1).cpu() for tensor in snapshot.valid_counts])
    if ((counts < 0) | (counts > REPLAY_CAPACITY)).any().item():
        _fail("valid count lies outside [0, 5]")
    checkpoint_mask = torch.cat([tensor.reshape(-1).cpu() for tensor in snapshot.checkpoint_masks])
    precision = _unpack_small_codes(
        checkpoint_mask,
        bits=1,
        count=STATE_ROWS,
    )
    if int(precision.sum().item()) != HIGH_PRECISION_ROWS:
        _fail(
            "checkpoint precision mask selects "
            f"{int(precision.sum().item())} rows; expected {HIGH_PRECISION_ROWS}"
        )
    result = frozen_storage_contract()
    actual = sum(tensor_bytes(tensor) for _, tensor in all_tensors)
    if actual != STATELEASE_BYTES:
        _fail(f"snapshot owns {actual} bytes; expected {STATELEASE_BYTES}")
    return result


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode())
    digest.update(byte_view.numpy().tobytes())
    return digest.hexdigest()


def resident_snapshot_digest(snapshot: ResidentSnapshot) -> str:
    digest = hashlib.sha256()
    for field_name in (
        "checkpoint_low_payloads",
        "checkpoint_high_payloads",
        "checkpoint_scales",
        "checkpoint_masks",
        "query_emas",
        "normalized_key_buffers",
        "update_buffers",
        "log_decay_buffers",
        "valid_counts",
    ):
        digest.update(field_name.encode())
        tensors = getattr(snapshot, field_name)
        for tensor in tensors:
            digest.update(_tensor_digest(tensor).encode())
    return digest.hexdigest()


def assert_rollback_preserved(
    before: ResidentSnapshot,
    after: ResidentSnapshot,
) -> str:
    before_digest = resident_snapshot_digest(before)
    after_digest = resident_snapshot_digest(after)
    if before_digest != after_digest:
        _fail("exception rollback changed persistent StateLease tensors")
    return before_digest


def verify_reset_snapshot(snapshot: ResidentSnapshot) -> None:
    audit_resident_snapshot(snapshot)
    counts = torch.cat([tensor.reshape(-1).cpu() for tensor in snapshot.valid_counts])
    if torch.count_nonzero(counts).item():
        _fail("reset left a nonzero replay count")
    for name, tensors in (
        ("normalized key", snapshot.normalized_key_buffers),
        ("update", snapshot.update_buffers),
        ("log decay", snapshot.log_decay_buffers),
    ):
        if any(torch.count_nonzero(tensor).item() for tensor in tensors):
            _fail(f"reset left nonzero {name} buffer contents")
    expected = 1.0 / ROWS
    if any(
        not torch.equal(tensor, torch.full_like(tensor, expected)) for tensor in snapshot.query_emas
    ):
        _fail("reset did not restore the uniform query EMA")


def verify_resume_integrity(
    *,
    prior_identity_hashes: Mapping[str, str],
    resumed_identity_hashes: Mapping[str, str],
    expected_identities: Sequence[str],
    completed_record_hashes: Mapping[str, str],
    resumed_completed_record_hashes: Mapping[str, str],
    resumed_remaining_identities: Sequence[str],
) -> None:
    if dict(prior_identity_hashes) != dict(resumed_identity_hashes):
        _fail("resume changed method, source, model, runtime, or identity hashes")
    if dict(completed_record_hashes) != dict(resumed_completed_record_hashes):
        _fail("resume changed an authenticated completed record")
    expected = tuple(expected_identities)
    if len(set(expected)) != len(expected):
        raise ValueError("expected identities contain duplicates")
    completed = set(completed_record_hashes)
    if not completed.issubset(expected):
        _fail("completed identities are outside the authenticated manifest")
    expected_remaining = tuple(identity for identity in expected if identity not in completed)
    if tuple(resumed_remaining_identities) != expected_remaining:
        _fail("resume does not omit exactly the authenticated completed identities")


def _compare_numeric(
    left: object,
    right: object,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            _fail(f"CC1 compatibility shape mismatch at {path}")
        if not torch.allclose(
            left.to(torch.float64),
            right.to(torch.float64),
            rtol=rtol,
            atol=atol,
        ):
            _fail(f"CC1 compatibility numeric mismatch at {path}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            _fail(f"CC1 compatibility mapping keys differ at {path}")
        for key in sorted(left, key=str):
            _compare_numeric(
                left[key],
                right[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
            )
        return
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            _fail(f"CC1 compatibility sequence length differs at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_numeric(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
            _fail(f"CC1 compatibility scalar mismatch at {path}")
        return
    if left != right:
        _fail(f"CC1 compatibility value mismatch at {path}")


def verify_cc1_compatibility(
    candidate: Mapping[str, object],
    anchor: Mapping[str, object],
    *,
    rtol: float,
    atol: float,
) -> None:
    required = frozenset(("trajectory", "aligned_metrics", "row_plan", "hashes"))
    if not required.issubset(candidate) or not required.issubset(anchor):
        _fail("CC1 artifact lacks trajectory, metrics, row plan, or hashes")
    _compare_numeric(
        candidate["trajectory"],
        anchor["trajectory"],
        path="trajectory",
        rtol=rtol,
        atol=atol,
    )
    _compare_numeric(
        candidate["aligned_metrics"],
        anchor["aligned_metrics"],
        path="aligned_metrics",
        rtol=rtol,
        atol=atol,
    )
    if candidate["row_plan"] != anchor["row_plan"]:
        _fail("CC1 row plans differ")
    if candidate["hashes"] != anchor["hashes"]:
        _fail("CC1 hashes differ")


def guard_protected_mbpp_window(
    *,
    stage: str,
    ranked_indices: Sequence[int] = (),
    task_ids: Sequence[str] = (),
    contains_quality_data: bool = False,
) -> None:
    """Reject protected ranks and any Stage-0 quality-data access."""

    normalized_stage = stage.strip().lower()
    if normalized_stage not in {"stage0", "stagea", "stageb", "stagec"}:
        raise ValueError("unknown Experiment 010 stage")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in ranked_indices):
        raise TypeError("ranked_indices must contain integers")
    if any(8 <= index < 16 for index in ranked_indices):
        _fail("protected ranked MBPP window [8, 16) was accessed")
    if normalized_stage == "stage0" and (ranked_indices or task_ids or contains_quality_data):
        _fail("Stage 0 may use only synthetic tensors and already captured traces")
    if normalized_stage == "stagea":
        if ranked_indices:
            _fail("Stage A may not resolve or inspect a new MBPP rank identity")
        if tuple(task_ids) != ("666",):
            _fail("Stage A is restricted to the already-open MBPP task 666")


def assert_independent_imports(path: Path) -> None:
    """Fail if this verifier acquires any package-under-test import."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        for module in modules:
            if module == "recurquant" or module.startswith("recurquant."):
                _fail(f"independent verifier imports package under test: {module}")


def _synthetic_record_sequence(
    *,
    dtype: torch.dtype,
    seed: int,
    count: int,
) -> tuple[torch.Tensor, list[ReplayRecord], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn((1, 2, 8, 8), generator=generator)
    initial = state.clone()
    records: list[ReplayRecord] = []
    for _ in range(count):
        key = torch.randn((1, 1, 2, 8), generator=generator).to(dtype)
        value = torch.randn((1, 1, 2, 8), generator=generator)
        decay = (-0.2 * torch.rand((1, 1, 2), generator=generator)).to(torch.float32)
        beta = torch.rand((1, 1, 2), generator=generator)
        normalized = normalize_consumed_key(key)[:, 0]
        decayed = state * decay[:, 0].exp().unsqueeze(-1).unsqueeze(-1)
        remembered = (decayed * normalized.unsqueeze(-1)).sum(dim=-2)
        update = (value[:, 0] - remembered) * beta[:, 0].unsqueeze(-1)
        final = decayed + normalized.unsqueeze(-1) * update.unsqueeze(-2)
        record = derive_successful_record(
            initial_state=state,
            consumed_key=key,
            value=value,
            log_decay=decay,
            beta=beta,
            successful_final_state=final,
        )
        records.append(record)
        state = final
    return initial, records, state


def run_synthetic_stage0() -> dict[str, object]:
    assert_independent_imports(Path(__file__))
    guard_protected_mbpp_window(stage="stage0")
    storage = frozen_storage_contract()
    comparators = equal_byte_codec_contracts()
    dtype_results: dict[str, dict[str, float]] = {}
    for offset, dtype in enumerate((torch.float32, torch.bfloat16, torch.float16)):
        initial, records, final = _synthetic_record_sequence(
            dtype=dtype,
            seed=FROZEN_SEED + offset,
            count=5,
        )
        replayed = dense_replay(initial, records)
        relative_l2, max_absolute = assert_replay_matches(final, replayed)
        buffers = store_records_bf16(records)
        buffered = replay_stored_buffers(initial, buffers)
        if not torch.isfinite(buffered).all().item():
            _fail("BF16 replay produced a non-finite state")
        dtype_results[str(dtype)] = {
            "unquantized_relative_l2": relative_l2,
            "unquantized_max_absolute": max_absolute,
            "bf16_buffer_relative_l2": relative_l2_and_max_abs(final, buffered)[0],
        }

    generator = torch.Generator().manual_seed(FROZEN_SEED)
    rows = torch.randn((8, 128), generator=generator)
    q48 = pack_physical_q4_q8(
        rows,
        torch.tensor([False, True, False, True, False, False, True, False]),
    )
    precision = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.uint8)
    q468 = pack_physical_q4_q6_q8(rows, precision)
    residual = pack_physical_residual_q4(
        rows,
        torch.tensor([True, False, True, False, False, True, False, True]),
    )
    for name, candidate in (
        ("q4_q8", q48.dequantize()),
        ("q4_q6_q8", q468.dequantize()),
        ("residual_q4", residual.dequantize()),
    ):
        if candidate.shape != rows.shape or not torch.isfinite(candidate).all().item():
            _fail(f"{name} physical codec failed its synthetic round trip")

    snapshot = canonical_empty_resident_snapshot()
    audit_resident_snapshot(snapshot)
    verify_reset_snapshot(snapshot)
    return {
        "status": "verifier_self_test_pass",
        "experiment_stage0_complete": False,
        "scope": "synthetic algebra, codec, storage, and guard self-test only",
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
        "independent_imports": True,
        "dtype_replay": dtype_results,
        "storage": storage,
        "equal_byte_comparators": comparators,
        "physical_codec_bytes": {
            "q4_q8": q48.storage_bytes,
            "q4_q6_q8": q468.storage_bytes,
            "residual_q4": residual.storage_bytes,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact rather than indented JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_synthetic_stage0()
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
