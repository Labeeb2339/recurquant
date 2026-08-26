"""Torch-free artifact contract for Experiment 013 static policies.

This module is intentionally limited to the Python standard library and NumPy.
It validates and reconstructs the metadata artifacts needed while the Stage-A
identity-capture import isolation forbids model frameworks.  It must not import
any other :mod:`recurquant` module: the capture runner authenticates this file
as a closed source unit before loading it.

The JSON schemas, canonical encodings, binary layouts, domain separators,
allocation tie rules, and frozen constants mirror ``static_q468.py``,
``static_q468_calibration.py``, and ``multibit_policy.py``.  Runtime modules can
adapt these immutable NumPy values to tensors outside capture isolation.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import numpy as np

# Static policy contract ----------------------------------------------------

STATIC_Q468_POLICY_SCHEMA = "recurquant.static-rht-q468-policy.v1"
STATIC_Q48_POLICY_SCHEMA = "recurquant.static-rht-q48-policy.v1"
STATIC_Q468_POLICY_REVISION = "experiment-013-static-policy-v1"
STATIC_Q48_POLICY_REVISION = "experiment-013-static-q48-policy-v1"
STATIC_Q468_ALLOCATOR_REVISION = "exact-multibit-o-nlogn-v1"
STATIC_Q468_CODEC_REVISION = "rht-q468-pools-u16-offsets-v1"
STATIC_Q48_SELECTOR_REVISION = "exact-top-q4-to-q8-benefit-v1"
STATIC_Q48_CODEC_REVISION = "rht-q48-pools-u16-offsets-v1"

STATIC_Q468_PRIMARY_METHOD = "rht_q468_static_k29334"
STATIC_Q468_ABLATION_METHOD = "rht_q468_static_k27030"
STATIC_Q468_MSE_METHOD = "rht_q468_static_mse_k29334"
STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD = (
    "rht_q468_static_diag_empirical_fisher_h1_k29334"
)
STATIC_Q468_UNIFORM_Q4_METHOD = "rht_q468_uniform_q4"
STATIC_Q468_UNIFORM_Q8_METHOD = "rht_q468_uniform_q8"
STATIC_Q48_COMPARATOR_METHOD = "rht_q48_static_p14739"

PRIMARY_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
PRIMARY_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PRIMARY_TOKENIZER_ID = PRIMARY_MODEL_ID
PRIMARY_TOKENIZER_REVISION = PRIMARY_MODEL_REVISION
FROZEN_TRANSFORMERS_VERSION = "5.14.1"

FROZEN_STATIC_Q468_PRIMARY_STEPS = 29_334
FROZEN_STATIC_Q468_ABLATION_STEPS = 27_030
FROZEN_STATIC_Q468_UNIFORM_Q4_STEPS = 0
FROZEN_STATIC_Q468_UNIFORM_Q8_STEPS = 73_728
FROZEN_STATIC_Q48_PROMOTIONS = 14_739
FROZEN_STATELEASE_RESIDENT_BYTES = 3_454_664
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
RHT_SEED = 2339

INT4_PRECISION_CODE = 0
INT6_PRECISION_CODE = 1
INT8_PRECISION_CODE = 2

StaticCodec: TypeAlias = Literal["q468", "q48"]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,127}")
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]*)?")


def _policy_canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Return the pretty canonical JSON used by research evidence artifacts."""

    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _validate_revision(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{name} must be a non-empty string of at most 128 characters")
    if not value.isascii() or any(character.isspace() for character in value):
        raise ValueError(f"{name} must be printable ASCII without whitespace")
    return value


def _validate_identity(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be a non-empty string of at most 256 characters")
    if (
        value != value.strip()
        or not value.isascii()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be stripped printable ASCII")
    return value


def _validate_git_revision(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable lowercase 40-hex Git revision")
    return value


def _validate_transformers_version(value: object) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ValueError("transformers_version must be a pinned semantic version")
    return value


def _validate_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _owned_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    ndim: int = 1,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype or value.ndim != ndim:
        raise TypeError(f"{name} must be a {ndim}-dimensional {dtype} array")
    normalized = np.ascontiguousarray(value, dtype=dtype)
    return _immutable_array_copy(normalized)


def _immutable_array_copy(value: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array whose immutable bytes prevent write re-enabling."""

    contiguous = np.ascontiguousarray(value)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _decode_b64(value: object, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a base64 string")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise ValueError(f"{name} is not canonical base64") from error
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(f"{name} is not canonical base64")
    return raw


def _array_b64(value: np.ndarray, *, dtype: np.dtype[Any]) -> str:
    array = np.ascontiguousarray(value).astype(dtype, copy=False)
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _expect_keys(value: object, *, name: str, expected: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys differ; missing={missing}, extra={extra}")
    return value


@dataclass(frozen=True, slots=True)
class StaticRhtQ468Geometry:
    layer_indices: tuple[int, ...]
    heads: int
    key_rows: int
    value_width: int
    target_resident_bytes: int

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
        _validate_integer(self.heads, name="heads", minimum=1)
        _validate_integer(self.key_rows, name="key_rows", minimum=1)
        width = _validate_integer(self.value_width, name="value_width", minimum=1)
        if width & (width - 1):
            raise ValueError("value_width must be a power of two for the frozen RHT")
        if width % 4:
            raise ValueError("value_width must make Q4, Q6, and Q8 rows byte aligned")
        _validate_integer(
            self.target_resident_bytes,
            name="target_resident_bytes",
            minimum=1,
        )
        if self.total_rows > 1 << 16:
            raise ValueError("uint16 pool offsets support at most 65,536 rows")

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

    def canonical_dict(self) -> dict[str, object]:
        return {
            "heads": self.heads,
            "key_rows": self.key_rows,
            "layer_indices": list(self.layer_indices),
            "target_resident_bytes": self.target_resident_bytes,
            "value_width": self.value_width,
        }

    @property
    def geometry_sha256(self) -> str:
        return _sha256_bytes(_policy_canonical_json(self.canonical_dict()))


FROZEN_QWEN35_STATIC_Q468_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=FROZEN_RECURRENT_LAYER_INDICES,
    heads=16,
    key_rows=128,
    value_width=128,
    target_resident_bytes=FROZEN_STATELEASE_RESIDENT_BYTES,
)


@dataclass(frozen=True, slots=True)
class StaticRhtByteLedger:
    method_id: str
    codec: StaticCodec
    selected_units: int
    payload_bytes: int
    scale_bytes: int
    precision_code_bytes: int
    pool_offset_bytes: int
    data_bytes: int
    alignment_bytes: int
    resident_bytes: int
    target_resident_bytes: int
    budget_delta_bytes: int
    exact_budget_eligible: bool

    def evidence_dict(self) -> dict[str, object]:
        return {
            "alignment_bytes": self.alignment_bytes,
            "budget_delta_bytes": self.budget_delta_bytes,
            "codec": self.codec,
            "data_bytes": self.data_bytes,
            "exact_budget_eligible": self.exact_budget_eligible,
            "method_id": self.method_id,
            "payload_bytes": self.payload_bytes,
            "pool_offset_bytes": self.pool_offset_bytes,
            "precision_code_bytes": self.precision_code_bytes,
            "resident_bytes": self.resident_bytes,
            "scale_bytes": self.scale_bytes,
            "selected_units": self.selected_units,
            "target_resident_bytes": self.target_resident_bytes,
        }


def _finish_ledger(
    *,
    method_id: str,
    codec: StaticCodec,
    selected_units: int,
    payload_bytes: int,
    scale_bytes: int,
    precision_code_bytes: int,
    pool_offset_bytes: int,
    target_resident_bytes: int,
) -> StaticRhtByteLedger:
    data_bytes = payload_bytes + scale_bytes + precision_code_bytes + pool_offset_bytes
    alignment_bytes = 8 if data_bytes + 8 == target_resident_bytes else 0
    resident_bytes = data_bytes + alignment_bytes
    delta = target_resident_bytes - resident_bytes
    return StaticRhtByteLedger(
        method_id=method_id,
        codec=codec,
        selected_units=selected_units,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        precision_code_bytes=precision_code_bytes,
        pool_offset_bytes=pool_offset_bytes,
        data_bytes=data_bytes,
        alignment_bytes=alignment_bytes,
        resident_bytes=resident_bytes,
        target_resident_bytes=target_resident_bytes,
        budget_delta_bytes=delta,
        exact_budget_eligible=delta == 0,
    )


def static_q468_byte_ledger(
    geometry: StaticRhtQ468Geometry,
    marginal_steps: int,
    *,
    method_id: str | None = None,
) -> StaticRhtByteLedger:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    steps = _validate_integer(marginal_steps, name="marginal_steps")
    if steps > 2 * geometry.total_rows:
        raise ValueError("marginal_steps exceeds two steps per state row")
    selected_method = method_id or f"rht_q468_static_k{steps}"
    if _METHOD_RE.fullmatch(selected_method) is None:
        raise ValueError("method_id must use lowercase identifier characters")
    return _finish_ledger(
        method_id=selected_method,
        codec="q468",
        selected_units=steps,
        payload_bytes=(
            geometry.state_elements * 4 // 8
            + steps * geometry.value_width * 2 // 8
        ),
        scale_bytes=geometry.total_rows * 2,
        precision_code_bytes=math.ceil(geometry.total_rows * 2 / 8),
        pool_offset_bytes=geometry.total_rows * 2,
        target_resident_bytes=geometry.target_resident_bytes,
    )


def static_q48_byte_ledger(
    geometry: StaticRhtQ468Geometry,
    promoted_rows: int,
    *,
    method_id: str | None = None,
) -> StaticRhtByteLedger:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    promotions = _validate_integer(promoted_rows, name="promoted_rows")
    if promotions > geometry.total_rows:
        raise ValueError("promoted_rows exceeds the number of state rows")
    selected_method = method_id or f"rht_q48_static_p{promotions}"
    if _METHOD_RE.fullmatch(selected_method) is None:
        raise ValueError("method_id must use lowercase identifier characters")
    return _finish_ledger(
        method_id=selected_method,
        codec="q48",
        selected_units=promotions,
        payload_bytes=(
            geometry.state_elements * 4 // 8
            + promotions * geometry.value_width * 4 // 8
        ),
        scale_bytes=geometry.total_rows * 2,
        precision_code_bytes=math.ceil(geometry.total_rows / 8),
        pool_offset_bytes=geometry.total_rows * 2,
        target_resident_bytes=geometry.target_resident_bytes,
    )


def pack_precision_codes(codes: np.ndarray) -> np.ndarray:
    normalized = _precision_codes(codes, name="precision_codes")
    padding = (-normalized.size) % 4
    if padding:
        normalized = np.pad(normalized, (0, padding), constant_values=0)
    chunks = normalized.reshape(-1, 4).astype(np.uint16, copy=False)
    packed = (
        chunks[:, 0]
        | np.left_shift(chunks[:, 1], 2)
        | np.left_shift(chunks[:, 2], 4)
        | np.left_shift(chunks[:, 3], 6)
    ).astype(np.uint8, copy=False)
    return np.ascontiguousarray(packed)


def unpack_precision_codes(packed: np.ndarray, total_rows: int) -> np.ndarray:
    if not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or packed.ndim != 1:
        raise TypeError("packed precision codes must be a one-dimensional uint8 array")
    rows = _validate_integer(total_rows, name="total_rows")
    expected_bytes = math.ceil(rows / 4)
    if packed.size != expected_bytes:
        raise ValueError(
            f"packed precision stream must contain {expected_bytes} bytes, got {packed.size}"
        )
    expanded = np.right_shift(
        packed.astype(np.uint16, copy=False)[:, None],
        np.asarray((0, 2, 4, 6), dtype=np.uint16),
    )
    all_codes = np.bitwise_and(expanded, 0x03).reshape(-1).astype(np.uint8, copy=False)
    codes = all_codes[:rows]
    if np.any(codes == 3):
        raise ValueError("packed precision stream contains reserved precision code 3")
    if np.any(all_codes[rows:] != 0):
        raise ValueError("unused precision-code padding bits must be zero")
    return np.ascontiguousarray(codes)


def pack_precision_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("precision mask must be a bool numpy.ndarray")
    flat = np.ascontiguousarray(mask.reshape(-1), dtype=np.uint8)
    padding = (-flat.size) % 8
    if padding:
        flat = np.pad(flat, (0, padding), constant_values=0)
    chunks = flat.reshape(-1, 8).astype(np.uint16, copy=False)
    weights = np.left_shift(np.ones(8, dtype=np.uint16), np.arange(8, dtype=np.uint16))
    return np.ascontiguousarray((chunks * weights).sum(axis=1).astype(np.uint8))


def unpack_precision_mask(packed: np.ndarray, total_rows: int) -> np.ndarray:
    if not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or packed.ndim != 1:
        raise TypeError("packed precision mask must be a one-dimensional uint8 array")
    rows = _validate_integer(total_rows, name="total_rows")
    expected_bytes = math.ceil(rows / 8)
    if packed.size != expected_bytes:
        raise ValueError(
            f"packed precision mask must contain {expected_bytes} bytes, got {packed.size}"
        )
    expanded = np.right_shift(
        packed.astype(np.uint16, copy=False)[:, None],
        np.arange(8, dtype=np.uint16),
    )
    all_bits = np.bitwise_and(expanded, 1).reshape(-1).astype(np.uint8, copy=False)
    if np.any(all_bits[rows:] != 0):
        raise ValueError("unused precision-mask padding bits must be zero")
    return np.ascontiguousarray(all_bits[:rows].astype(np.bool_))


def _precision_codes(
    value: object,
    *,
    name: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8")
    normalized = np.ascontiguousarray(value.reshape(-1), dtype=np.uint8)
    if normalized.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if expected_rows is not None and normalized.size != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} codes")
    if np.any(normalized > 2):
        raise ValueError(f"{name} may contain only Q4/Q6/Q8 codes 0, 1, and 2")
    return normalized.copy()


def _expected_pool_offsets(codes: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    flat = _precision_codes(codes, name="codes")
    offsets = np.empty(flat.size, dtype=np.uint16)
    counts: list[int] = []
    for code in (INT4_PRECISION_CODE, INT6_PRECISION_CODE, INT8_PRECISION_CODE):
        mask = flat == code
        count = int(np.count_nonzero(mask))
        counts.append(count)
        if count:
            offsets[mask] = np.arange(count, dtype=np.uint16)
    return np.ascontiguousarray(offsets), (counts[0], counts[1], counts[2])


def _expected_binary_pool_offsets(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("precision mask must be a bool numpy.ndarray")
    flat = np.ascontiguousarray(mask.reshape(-1), dtype=np.bool_)
    offsets = np.empty(flat.size, dtype=np.uint16)
    low_count = int(np.count_nonzero(~flat))
    high_count = int(np.count_nonzero(flat))
    if low_count:
        offsets[~flat] = np.arange(low_count, dtype=np.uint16)
    if high_count:
        offsets[flat] = np.arange(high_count, dtype=np.uint16)
    return np.ascontiguousarray(offsets), (low_count, high_count)


# Exact Q4/Q6/Q8 allocation -----------------------------------------------


def _validate_distortion_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"{name} must have non-empty shape [heads, rows]")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    shape = cast(tuple[int, int], tuple(value.shape))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"D4, D6, and D8 must have identical shapes; {name} has {shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(value < 0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return shape


def _exact_flat_distortions(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    shape = _validate_distortion_array(d4, name="D4", expected_shape=None)
    _validate_distortion_array(d6, name="D6", expected_shape=shape)
    _validate_distortion_array(d8, name="D8", expected_shape=shape)
    flat_values = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in (d4, d6, d8)],
        axis=1,
    )
    maximum_denominator_exponent = 0
    for value in flat_values.reshape(-1):
        _, denominator = float(value).as_integer_ratio()
        maximum_denominator_exponent = max(
            maximum_denominator_exponent,
            denominator.bit_length() - 1,
        )
    exact_rows: list[tuple[int, int, int]] = []
    common_trailing_zeros: int | None = None
    for row in flat_values:
        scaled: list[int] = []
        for value in row:
            numerator, denominator = float(value).as_integer_ratio()
            denominator_exponent = denominator.bit_length() - 1
            scaled.append(numerator << (maximum_denominator_exponent - denominator_exponent))
        row_minimum = min(scaled)
        reduced = cast(tuple[int, int, int], tuple(value - row_minimum for value in scaled))
        exact_rows.append(reduced)
        for value in reduced:
            if value:
                trailing_zeros = (value & -value).bit_length() - 1
                common_trailing_zeros = (
                    trailing_zeros
                    if common_trailing_zeros is None
                    else min(common_trailing_zeros, trailing_zeros)
                )
    if common_trailing_zeros:
        exact_rows = [
            cast(tuple[int, int, int], tuple(value >> common_trailing_zeros for value in row))
            for row in exact_rows
        ]
    return shape, exact_rows


def _validate_marginal_steps(marginal_steps: object, *, total_rows: int) -> int:
    if isinstance(marginal_steps, bool) or not isinstance(marginal_steps, int):
        raise TypeError("marginal_steps must be an integer")
    if not 0 <= marginal_steps <= 2 * total_rows:
        raise ValueError(
            f"marginal_steps must be in the exact representable range [0, {2 * total_rows}]"
        )
    return marginal_steps


class _RangeMinimum:
    def __init__(self, values: np.ndarray, *, sentinel: int) -> None:
        self.length = int(values.size)
        size = 1
        while size < self.length:
            size *= 2
        self.size = size
        self.sentinel = sentinel
        tree = np.full(2 * size, sentinel, dtype=np.int64)
        if self.length:
            tree[size : size + self.length] = values.astype(np.int64, copy=False)
        for index in range(size - 1, 0, -1):
            tree[index] = min(tree[2 * index], tree[2 * index + 1])
        self.tree = tree

    def query(self, start: int, stop: int) -> int:
        if not 0 <= start <= stop <= self.length:
            raise RuntimeError("internal range-minimum query is outside its domain")
        if start == stop:
            return self.sentinel
        left = start + self.size
        right = stop + self.size
        result = self.sentinel
        while left < right:
            if left & 1:
                result = min(result, int(self.tree[left]))
                left += 1
            if right & 1:
                right -= 1
                result = min(result, int(self.tree[right]))
            left //= 2
            right //= 2
        return result


@dataclass(frozen=True, slots=True)
class _FastAllocationCandidate:
    convex_steps: int
    bundle_count: int
    singleton_rank: int | None
    gain: int


@dataclass(frozen=True, slots=True)
class _FastAllocationStructure:
    total_rows: int
    convex_mask: np.ndarray
    convex_first_rank: np.ndarray
    convex_second_rank: np.ndarray
    nonconvex_rank: np.ndarray
    bundle_rows: np.ndarray
    convex_row_rmq: _RangeMinimum
    bundle_row_rmq: _RangeMinimum


def _candidate_bundle_limit(candidate: _FastAllocationCandidate) -> int:
    singleton = candidate.singleton_rank
    return candidate.bundle_count + int(
        singleton is not None and singleton < candidate.bundle_count
    )


def _candidate_nonconvex_code(candidate: _FastAllocationCandidate, rank: int) -> int:
    singleton = candidate.singleton_rank
    if rank < _candidate_bundle_limit(candidate) and rank != singleton:
        return 2
    if rank == singleton:
        return 1
    return 0


def _candidate_code_at_row(
    candidate: _FastAllocationCandidate,
    row: int,
    structure: _FastAllocationStructure,
) -> int:
    if structure.convex_mask[row]:
        return int(structure.convex_first_rank[row] < candidate.convex_steps) + int(
            structure.convex_second_rank[row] < candidate.convex_steps
        )
    return _candidate_nonconvex_code(candidate, int(structure.nonconvex_rank[row]))


def _candidate_lexicographically_greater(
    left: _FastAllocationCandidate,
    right: _FastAllocationCandidate,
    structure: _FastAllocationStructure,
) -> bool:
    sentinel = structure.total_rows
    earliest = sentinel
    if left.convex_steps != right.convex_steps:
        earliest = min(
            earliest,
            structure.convex_row_rmq.query(
                min(left.convex_steps, right.convex_steps),
                max(left.convex_steps, right.convex_steps),
            ),
        )
    nonconvex_rows = structure.bundle_rows.size
    if nonconvex_rows:
        boundaries = {
            0,
            nonconvex_rows,
            _candidate_bundle_limit(left),
            _candidate_bundle_limit(right),
        }
        for singleton in (left.singleton_rank, right.singleton_rank):
            if singleton is not None:
                boundaries.add(singleton)
                boundaries.add(singleton + 1)
        ordered = sorted(boundary for boundary in boundaries if 0 <= boundary <= nonconvex_rows)
        for start, stop in zip(ordered, ordered[1:], strict=False):
            if start == stop:
                continue
            if _candidate_nonconvex_code(left, start) != _candidate_nonconvex_code(right, start):
                earliest = min(earliest, structure.bundle_row_rmq.query(start, stop))
    if earliest == sentinel:
        return False
    left_code = _candidate_code_at_row(left, earliest, structure)
    right_code = _candidate_code_at_row(right, earliest, structure)
    if left_code == right_code:
        raise RuntimeError("implicit lexicographic comparison failed to find a differing row")
    return left_code > right_code


def _better_fast_candidate(
    candidate: _FastAllocationCandidate,
    incumbent: _FastAllocationCandidate | None,
    structure: _FastAllocationStructure,
) -> _FastAllocationCandidate:
    if incumbent is None or candidate.gain > incumbent.gain:
        return candidate
    if candidate.gain == incumbent.gain and _candidate_lexicographically_greater(
        candidate, incumbent, structure
    ):
        return candidate
    return incumbent


def allocate_exact_multibit_codes(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
    *,
    marginal_steps: int,
) -> np.ndarray:
    """Return the exact complete-state optimum with the frozen tie rule."""

    shape, flat_distortions = _exact_flat_distortions(d4, d6, d8)
    total_rows = math.prod(shape)
    steps = _validate_marginal_steps(marginal_steps, total_rows=total_rows)
    if steps == 0:
        return np.zeros(shape, dtype=np.uint8)
    if steps == 2 * total_rows:
        return np.full(shape, 2, dtype=np.uint8)

    first_gain = [row[0] - row[1] for row in flat_distortions]
    second_gain = [row[1] - row[2] for row in flat_distortions]
    convex_mask = np.fromiter(
        (first_gain[row] >= second_gain[row] for row in range(total_rows)),
        dtype=np.bool_,
        count=total_rows,
    )
    convex_rows = np.flatnonzero(convex_mask).astype(np.int64, copy=False)
    nonconvex_rows = np.flatnonzero(~convex_mask).astype(np.int64, copy=False)

    convex_count = convex_rows.size
    increments = [(first_gain[int(row)], int(row), 0) for row in convex_rows]
    increments.extend((second_gain[int(row)], int(row), 1) for row in convex_rows)
    increments.sort(key=lambda item: (-item[0], item[1], item[2]))
    ordered_increment_rows = np.fromiter(
        (row for _, row, _ in increments), dtype=np.int64, count=len(increments)
    )
    convex_prefix = [0]
    for gain, _, _ in increments:
        convex_prefix.append(convex_prefix[-1] + gain)
    convex_first_rank = np.full(total_rows, -1, dtype=np.int64)
    convex_second_rank = np.full(total_rows, -1, dtype=np.int64)
    for rank, (_, row, precision_step) in enumerate(increments):
        if precision_step == 0:
            convex_first_rank[row] = rank
        else:
            convex_second_rank[row] = rank

    bundles = sorted(
        (
            (first_gain[int(row)] + second_gain[int(row)], int(row), first_gain[int(row)])
            for row in nonconvex_rows
        ),
        key=lambda item: (-item[0], item[1]),
    )
    bundle_rows = np.fromiter(
        (row for _, row, _ in bundles), dtype=np.int64, count=len(bundles)
    )
    bundle_gains = [gain for gain, _, _ in bundles]
    bundle_first_gains = [gain for _, _, gain in bundles]
    bundle_prefix = [0]
    for gain in bundle_gains:
        bundle_prefix.append(bundle_prefix[-1] + gain)
    nonconvex_rank = np.full(total_rows, -1, dtype=np.int64)
    nonconvex_rank[bundle_rows] = np.arange(bundle_rows.size, dtype=np.int64)
    structure = _FastAllocationStructure(
        total_rows=total_rows,
        convex_mask=convex_mask,
        convex_first_rank=convex_first_rank,
        convex_second_rank=convex_second_rank,
        nonconvex_rank=nonconvex_rank,
        bundle_rows=bundle_rows,
        convex_row_rmq=_RangeMinimum(ordered_increment_rows, sentinel=total_rows),
        bundle_row_rmq=_RangeMinimum(bundle_rows, sentinel=total_rows),
    )

    bundle_count = bundle_rows.size
    prefix_singleton = np.full(bundle_count + 1, -1, dtype=np.int64)
    best_prefix = -1
    prefix_adjustment = [
        first - complete
        for first, complete in zip(bundle_first_gains, bundle_gains, strict=True)
    ]
    for stop in range(1, bundle_count + 1):
        rank = stop - 1
        if (
            best_prefix < 0
            or prefix_adjustment[rank] > prefix_adjustment[best_prefix]
            or (
                prefix_adjustment[rank] == prefix_adjustment[best_prefix]
                and bundle_rows[rank] > bundle_rows[best_prefix]
            )
        ):
            best_prefix = rank
        prefix_singleton[stop] = best_prefix

    suffix_singleton = np.full(bundle_count + 1, -1, dtype=np.int64)
    best_suffix = -1
    for start in range(bundle_count - 1, -1, -1):
        if (
            best_suffix < 0
            or bundle_first_gains[start] > bundle_first_gains[best_suffix]
            or (
                bundle_first_gains[start] == bundle_first_gains[best_suffix]
                and bundle_rows[start] < bundle_rows[best_suffix]
            )
        ):
            best_suffix = start
        suffix_singleton[start] = best_suffix

    best: _FastAllocationCandidate | None = None
    convex_increments = len(increments)
    minimum_bundles = max(0, math.ceil((steps - convex_increments) / 2))
    maximum_bundles = min(bundle_count, steps // 2)
    for selected_bundles in range(minimum_bundles, maximum_bundles + 1):
        selected_convex = steps - 2 * selected_bundles
        best = _better_fast_candidate(
            _FastAllocationCandidate(
                convex_steps=selected_convex,
                bundle_count=selected_bundles,
                singleton_rank=None,
                gain=convex_prefix[selected_convex] + bundle_prefix[selected_bundles],
            ),
            best,
            structure,
        )

    if bundle_count:
        minimum_bundles = max(0, math.ceil((steps - 1 - convex_increments) / 2))
        maximum_bundles = min(bundle_count - 1, (steps - 1) // 2)
        for selected_bundles in range(minimum_bundles, maximum_bundles + 1):
            selected_convex = steps - 2 * selected_bundles - 1
            if selected_bundles:
                rank = int(prefix_singleton[selected_bundles])
                best = _better_fast_candidate(
                    _FastAllocationCandidate(
                        convex_steps=selected_convex,
                        bundle_count=selected_bundles,
                        singleton_rank=rank,
                        gain=(
                            convex_prefix[selected_convex]
                            + bundle_prefix[selected_bundles + 1]
                            + prefix_adjustment[rank]
                        ),
                    ),
                    best,
                    structure,
                )
            rank = int(suffix_singleton[selected_bundles])
            if rank >= 0:
                best = _better_fast_candidate(
                    _FastAllocationCandidate(
                        convex_steps=selected_convex,
                        bundle_count=selected_bundles,
                        singleton_rank=rank,
                        gain=(
                            convex_prefix[selected_convex]
                            + bundle_prefix[selected_bundles]
                            + bundle_first_gains[rank]
                        ),
                    ),
                    best,
                    structure,
                )

    if best is None:
        raise RuntimeError("exact multibit allocator found no feasible allocation")
    flat_codes = np.zeros(total_rows, dtype=np.uint8)
    if convex_count:
        flat_codes[convex_rows] = (
            (convex_first_rank[convex_rows] < best.convex_steps).astype(np.uint8)
            + (convex_second_rank[convex_rows] < best.convex_steps).astype(np.uint8)
        )
    limit = _candidate_bundle_limit(best)
    if limit:
        flat_codes[bundle_rows[:limit]] = 2
    if best.singleton_rank is not None:
        flat_codes[bundle_rows[best.singleton_rank]] = 1
    if int(flat_codes.astype(np.int64).sum()) != steps:
        raise RuntimeError("exact multibit allocator changed the requested step budget")
    return np.ascontiguousarray(flat_codes.reshape(shape))


# Name used by the tensor-backed implementation; both names use the same core.
allocate_exact_multibit_codes_fast = allocate_exact_multibit_codes


def _normalize_distortion_triplet(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    expected_shape = tuple(d4.shape) if isinstance(d4, np.ndarray) else None
    normalized: list[np.ndarray] = []
    for name, value in (("D4", d4), ("D6", d6), ("D8", d8)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.size != geometry.total_rows:
            raise ValueError(f"{name} must contain exactly {geometry.total_rows} row distortions")
        if tuple(value.shape) != expected_shape:
            raise ValueError("D4, D6, and D8 must have identical shapes")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must use a floating-point dtype")
        array = np.ascontiguousarray(value, dtype=np.float64).reshape(1, geometry.total_rows)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name} must contain finite, non-negative values")
        normalized.append(array)
    return normalized[0], normalized[1], normalized[2]


def static_q468_distortion_sha256(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
) -> str:
    values = _normalize_distortion_triplet(d4, d6, d8, geometry=geometry)
    digest = hashlib.sha256()
    digest.update(b"recurquant.static-q468-distortions.v1\0")
    digest.update(_policy_canonical_json(geometry.canonical_dict()))
    for label, value in zip((b"D4\0", b"D6\0", b"D8\0"), values, strict=True):
        digest.update(label)
        digest.update(value.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StaticRhtQ468Policy:
    method_id: str
    policy_revision: str
    allocator_revision: str
    codec_revision: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_manifest_sha256: str
    transformers_version: str
    identity_artifact_sha256: str
    source_commit: str
    geometry: StaticRhtQ468Geometry
    calibration_manifest_sha256: str
    calibration_scores_sha256: str
    marginal_steps: int
    rht_seed: int
    packed_precision_codes: np.ndarray
    pool_offsets: np.ndarray
    pool_counts: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or _METHOD_RE.fullmatch(self.method_id) is None:
            raise ValueError("method_id must use lowercase identifier characters")
        _validate_revision(self.policy_revision, name="policy_revision")
        _validate_revision(self.allocator_revision, name="allocator_revision")
        _validate_revision(self.codec_revision, name="codec_revision")
        if self.policy_revision != STATIC_Q468_POLICY_REVISION:
            raise ValueError("unsupported static Q468 policy revision")
        if self.allocator_revision != STATIC_Q468_ALLOCATOR_REVISION:
            raise ValueError("unsupported static Q468 allocator revision")
        if self.codec_revision != STATIC_Q468_CODEC_REVISION:
            raise ValueError("unsupported static Q468 codec revision")
        _validate_identity(self.model_id, name="model_id")
        _validate_git_revision(self.model_revision, name="model_revision")
        _validate_identity(self.tokenizer_id, name="tokenizer_id")
        _validate_git_revision(self.tokenizer_revision, name="tokenizer_revision")
        _validate_sha256(self.tokenizer_manifest_sha256, name="tokenizer_manifest_sha256")
        _validate_transformers_version(self.transformers_version)
        _validate_sha256(self.identity_artifact_sha256, name="identity_artifact_sha256")
        _validate_git_revision(self.source_commit, name="source_commit")
        if not isinstance(self.geometry, StaticRhtQ468Geometry):
            raise TypeError("geometry must be a StaticRhtQ468Geometry")
        _validate_sha256(self.calibration_manifest_sha256, name="calibration_manifest_sha256")
        _validate_sha256(self.calibration_scores_sha256, name="calibration_scores_sha256")
        steps = _validate_integer(self.marginal_steps, name="marginal_steps")
        if steps > 2 * self.geometry.total_rows:
            raise ValueError("marginal_steps exceeds two steps per state row")
        if self.method_id == STATIC_Q48_COMPARATOR_METHOD:
            raise ValueError("reserved static Q48 method cannot identify a Q468 policy")
        frozen_steps = {
            STATIC_Q468_PRIMARY_METHOD: FROZEN_STATIC_Q468_PRIMARY_STEPS,
            STATIC_Q468_ABLATION_METHOD: FROZEN_STATIC_Q468_ABLATION_STEPS,
            STATIC_Q468_MSE_METHOD: FROZEN_STATIC_Q468_PRIMARY_STEPS,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD: FROZEN_STATIC_Q468_PRIMARY_STEPS,
            STATIC_Q468_UNIFORM_Q4_METHOD: FROZEN_STATIC_Q468_UNIFORM_Q4_STEPS,
            STATIC_Q468_UNIFORM_Q8_METHOD: FROZEN_STATIC_Q468_UNIFORM_Q8_STEPS,
        }.get(self.method_id)
        if frozen_steps is not None:
            if self.geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
                raise ValueError("reserved static Q468 method requires the frozen geometry")
            if steps != frozen_steps:
                raise ValueError("reserved static Q468 method has the wrong exact-K budget")
            if (
                self.model_id != PRIMARY_MODEL_ID
                or self.model_revision != PRIMARY_MODEL_REVISION
                or self.tokenizer_id != PRIMARY_TOKENIZER_ID
                or self.tokenizer_revision != PRIMARY_TOKENIZER_REVISION
                or self.transformers_version != FROZEN_TRANSFORMERS_VERSION
            ):
                raise ValueError("reserved static Q468 method requires the frozen model identity")
            if self.method_id in {
                STATIC_Q468_PRIMARY_METHOD,
                STATIC_Q468_MSE_METHOD,
                STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
            }:
                ledger = static_q468_byte_ledger(
                    self.geometry, steps, method_id=self.method_id
                )
                if (
                    ledger.resident_bytes != FROZEN_STATELEASE_RESIDENT_BYTES
                    or ledger.target_resident_bytes != FROZEN_STATELEASE_RESIDENT_BYTES
                    or not ledger.exact_budget_eligible
                ):
                    raise ValueError(
                        "reserved exact-budget static Q468 method has the wrong resident ledger"
                    )
        if self.rht_seed != RHT_SEED:
            raise ValueError(f"rht_seed must equal the codec seed {RHT_SEED}")

        packed = _owned_array(
            self.packed_precision_codes,
            name="packed_precision_codes",
            dtype=np.dtype("u1"),
        )
        offsets = _owned_array(
            self.pool_offsets,
            name="pool_offsets",
            dtype=np.dtype("uint16"),
        )
        if packed.size != math.ceil(self.geometry.total_rows * 2 / 8):
            raise ValueError("packed_precision_codes byte length does not match geometry")
        if offsets.size != self.geometry.total_rows:
            raise ValueError("pool_offsets must contain one uint16 value per state row")
        codes = unpack_precision_codes(packed, self.geometry.total_rows)
        if int(codes.astype(np.int64).sum()) != steps:
            raise ValueError("precision-code marginal sum does not match marginal_steps")
        expected_offsets, expected_counts = _expected_pool_offsets(codes)
        if not isinstance(self.pool_counts, tuple) or len(self.pool_counts) != 3:
            raise TypeError("pool_counts must be a three-integer tuple")
        for index, count in enumerate(self.pool_counts):
            _validate_integer(count, name=f"pool_counts[{index}]")
        if self.pool_counts != expected_counts:
            raise ValueError(
                f"pool_counts {self.pool_counts} do not match precision codes {expected_counts}"
            )
        if sum(self.pool_counts) != self.geometry.total_rows:
            raise ValueError("pool_counts do not cover every state row")
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("pool_offsets are not canonical per-pool prefix offsets")
        object.__setattr__(self, "packed_precision_codes", packed)
        object.__setattr__(self, "pool_offsets", offsets)

    def precision_codes(self) -> np.ndarray:
        return unpack_precision_codes(
            self.packed_precision_codes, self.geometry.total_rows
        ).reshape(self.geometry.layers, self.geometry.heads, self.geometry.key_rows)

    def _content_dict(self) -> dict[str, object]:
        return {
            "allocator_revision": self.allocator_revision,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_scores_sha256": self.calibration_scores_sha256,
            "code_map_sha256": self.code_map_sha256,
            "codec_revision": self.codec_revision,
            "geometry": self.geometry.canonical_dict(),
            "geometry_sha256": self.geometry.geometry_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "marginal_steps": self.marginal_steps,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "packed_precision_codes_b64": _array_b64(
                self.packed_precision_codes, dtype=np.dtype("u1")
            ),
            "policy_revision": self.policy_revision,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_le_b64": _array_b64(
                self.pool_offsets, dtype=np.dtype("<u2")
            ),
            "pool_offsets_sha256": self.pool_offsets_sha256,
            "rht_seed": self.rht_seed,
            "source_commit": self.source_commit,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "transformers_version": self.transformers_version,
        }

    @property
    def code_map_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"recurquant.static-q468-code-map.v1\0")
        digest.update(bytes.fromhex(self.geometry.geometry_sha256))
        digest.update(self.marginal_steps.to_bytes(8, "little", signed=False))
        digest.update(self.packed_precision_codes.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def pool_offsets_sha256(self) -> str:
        return _sha256_bytes(
            self.pool_offsets.astype("<u2", copy=False).tobytes(order="C")
        )

    @property
    def policy_sha256(self) -> str:
        return _sha256_bytes(_policy_canonical_json(self._content_dict()))


def build_static_rht_q468_policy(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
    marginal_steps: int,
    calibration_manifest_sha256: str,
    identity_artifact_sha256: str,
    tokenizer_manifest_sha256: str,
    source_commit: str,
    calibration_scores_sha256: str | None = None,
    method_id: str | None = None,
    policy_revision: str = STATIC_Q468_POLICY_REVISION,
    allocator_revision: str = STATIC_Q468_ALLOCATOR_REVISION,
    codec_revision: str = STATIC_Q468_CODEC_REVISION,
    model_id: str = PRIMARY_MODEL_ID,
    model_revision: str = PRIMARY_MODEL_REVISION,
    tokenizer_id: str = PRIMARY_TOKENIZER_ID,
    tokenizer_revision: str = PRIMARY_TOKENIZER_REVISION,
    transformers_version: str = FROZEN_TRANSFORMERS_VERSION,
    rht_seed: int = RHT_SEED,
) -> StaticRhtQ468Policy:
    normalized = _normalize_distortion_triplet(d4, d6, d8, geometry=geometry)
    scores_sha256 = static_q468_distortion_sha256(*normalized, geometry=geometry)
    if calibration_scores_sha256 is not None:
        _validate_sha256(calibration_scores_sha256, name="calibration_scores_sha256")
        if calibration_scores_sha256 != scores_sha256:
            raise ValueError("calibration_scores_sha256 does not match supplied distortions")
    codes = allocate_exact_multibit_codes(
        *normalized, marginal_steps=marginal_steps
    ).reshape(-1)
    offsets, counts = _expected_pool_offsets(codes)
    return StaticRhtQ468Policy(
        method_id=method_id or f"rht_q468_static_k{marginal_steps}",
        policy_revision=policy_revision,
        allocator_revision=allocator_revision,
        codec_revision=codec_revision,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        transformers_version=transformers_version,
        identity_artifact_sha256=identity_artifact_sha256,
        source_commit=source_commit,
        geometry=geometry,
        calibration_manifest_sha256=calibration_manifest_sha256,
        calibration_scores_sha256=scores_sha256,
        marginal_steps=marginal_steps,
        rht_seed=rht_seed,
        packed_precision_codes=pack_precision_codes(codes),
        pool_offsets=offsets,
        pool_counts=counts,
    )


def serialize_static_rht_q468_policy(policy: StaticRhtQ468Policy) -> bytes:
    if not isinstance(policy, StaticRhtQ468Policy):
        raise TypeError("policy must be a StaticRhtQ468Policy")
    content = policy._content_dict()
    return _policy_canonical_json(
        {
            "content": content,
            "policy_sha256": _sha256_bytes(_policy_canonical_json(content)),
            "schema": STATIC_Q468_POLICY_SCHEMA,
        }
    ) + b"\n"


def _parse_policy_geometry(value: object) -> StaticRhtQ468Geometry:
    geometry_dict = _expect_keys(
        value,
        name="geometry",
        expected={
            "heads",
            "key_rows",
            "layer_indices",
            "target_resident_bytes",
            "value_width",
        },
    )
    layer_indices = geometry_dict["layer_indices"]
    if not isinstance(layer_indices, list):
        raise TypeError("geometry.layer_indices must be a list")
    return StaticRhtQ468Geometry(
        layer_indices=tuple(layer_indices),
        heads=geometry_dict["heads"],
        key_rows=geometry_dict["key_rows"],
        value_width=geometry_dict["value_width"],
        target_resident_bytes=geometry_dict["target_resident_bytes"],
    )


def deserialize_static_rht_q468_policy(data: bytes) -> StaticRhtQ468Policy:
    if not isinstance(data, bytes):
        raise TypeError("serialized policy must be bytes")
    try:
        root = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("serialized policy is not valid UTF-8 JSON") from error
    envelope = _expect_keys(
        root,
        name="policy envelope",
        expected={"content", "policy_sha256", "schema"},
    )
    if envelope["schema"] != STATIC_Q468_POLICY_SCHEMA:
        raise ValueError("unsupported static Q468 policy schema")
    declared_digest = _validate_sha256(envelope["policy_sha256"], name="policy_sha256")
    content = _expect_keys(
        envelope["content"],
        name="policy content",
        expected={
            "allocator_revision",
            "calibration_manifest_sha256",
            "calibration_scores_sha256",
            "code_map_sha256",
            "codec_revision",
            "geometry",
            "geometry_sha256",
            "identity_artifact_sha256",
            "marginal_steps",
            "method_id",
            "model_id",
            "model_revision",
            "packed_precision_codes_b64",
            "policy_revision",
            "pool_counts",
            "pool_offsets_le_b64",
            "pool_offsets_sha256",
            "rht_seed",
            "source_commit",
            "tokenizer_id",
            "tokenizer_manifest_sha256",
            "tokenizer_revision",
            "transformers_version",
        },
    )
    if _sha256_bytes(_policy_canonical_json(content)) != declared_digest:
        raise ValueError("policy_sha256 does not authenticate policy content")
    geometry = _parse_policy_geometry(content["geometry"])
    if content["geometry_sha256"] != geometry.geometry_sha256:
        raise ValueError("geometry_sha256 does not authenticate geometry")
    code_bytes = _decode_b64(
        content["packed_precision_codes_b64"], name="packed_precision_codes_b64"
    )
    offset_bytes = _decode_b64(
        content["pool_offsets_le_b64"], name="pool_offsets_le_b64"
    )
    if len(code_bytes) != math.ceil(geometry.total_rows * 2 / 8):
        raise ValueError("packed precision code byte length does not match geometry")
    if len(offset_bytes) != geometry.total_rows * 2:
        raise ValueError("pool offset byte length does not match geometry")
    counts_value = content["pool_counts"]
    if not isinstance(counts_value, list):
        raise TypeError("pool_counts must be a list")
    policy = StaticRhtQ468Policy(
        method_id=content["method_id"],
        policy_revision=content["policy_revision"],
        allocator_revision=content["allocator_revision"],
        codec_revision=content["codec_revision"],
        model_id=content["model_id"],
        model_revision=content["model_revision"],
        tokenizer_id=content["tokenizer_id"],
        tokenizer_revision=content["tokenizer_revision"],
        tokenizer_manifest_sha256=content["tokenizer_manifest_sha256"],
        transformers_version=content["transformers_version"],
        identity_artifact_sha256=content["identity_artifact_sha256"],
        source_commit=content["source_commit"],
        geometry=geometry,
        calibration_manifest_sha256=content["calibration_manifest_sha256"],
        calibration_scores_sha256=content["calibration_scores_sha256"],
        marginal_steps=content["marginal_steps"],
        rht_seed=content["rht_seed"],
        packed_precision_codes=np.frombuffer(code_bytes, dtype="u1").copy(),
        pool_offsets=np.frombuffer(offset_bytes, dtype="<u2").astype(np.uint16, copy=True),
        pool_counts=tuple(counts_value),
    )
    if policy.policy_sha256 != declared_digest:
        raise ValueError("reconstructed policy digest differs from policy_sha256")
    if content["code_map_sha256"] != policy.code_map_sha256:
        raise ValueError("code_map_sha256 does not authenticate the precision map")
    if content["pool_offsets_sha256"] != policy.pool_offsets_sha256:
        raise ValueError("pool_offsets_sha256 does not authenticate pool offsets")
    if serialize_static_rht_q468_policy(policy) != data:
        raise ValueError("serialized policy is not in canonical form")
    return policy


def _normalize_q48_distortions(
    d4: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    expected_shape = tuple(d4.shape) if isinstance(d4, np.ndarray) else None
    normalized: list[np.ndarray] = []
    for name, value in (("D4", d4), ("D8", d8)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.size != geometry.total_rows:
            raise ValueError(f"{name} must contain exactly {geometry.total_rows} row distortions")
        if tuple(value.shape) != expected_shape:
            raise ValueError("D4 and D8 must have identical shapes")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must use a floating-point dtype")
        array = np.ascontiguousarray(value, dtype=np.float64).reshape(1, geometry.total_rows)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name} must contain finite, non-negative values")
        normalized.append(array)
    return normalized[0], normalized[1]


def static_q48_distortion_sha256(
    d4: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
) -> str:
    values = _normalize_q48_distortions(d4, d8, geometry=geometry)
    digest = hashlib.sha256()
    digest.update(b"recurquant.static-q48-distortions.v1\0")
    digest.update(_policy_canonical_json(geometry.canonical_dict()))
    for label, value in zip((b"D4\0", b"D8\0"), values, strict=True):
        digest.update(label)
        digest.update(value.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def allocate_exact_q48_mask(
    d4: np.ndarray,
    d8: np.ndarray,
    *,
    promoted_rows: int,
) -> np.ndarray:
    if not isinstance(d4, np.ndarray) or not isinstance(d8, np.ndarray):
        raise TypeError("D4 and D8 must be numpy.ndarray values")
    if d4.size == 0 or tuple(d4.shape) != tuple(d8.shape):
        raise ValueError("D4 and D8 must have one identical non-empty shape")
    for name, value in (("D4", d4), ("D8", d8)):
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must use a floating-point dtype")
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must contain finite, non-negative values")
    promotions = _validate_integer(promoted_rows, name="promoted_rows")
    if promotions > d4.size:
        raise ValueError("promoted_rows exceeds the number of distortion rows")
    flat_d4 = np.asarray(d4, dtype=np.float64).reshape(-1).tolist()
    flat_d8 = np.asarray(d8, dtype=np.float64).reshape(-1).tolist()
    benefits = [
        Fraction.from_float(value4) - Fraction.from_float(value8)
        for value4, value8 in zip(flat_d4, flat_d8, strict=True)
    ]
    ranked = sorted(range(len(benefits)), key=lambda index: (-benefits[index], index))
    mask = np.zeros(len(benefits), dtype=np.bool_)
    if promotions:
        mask[np.asarray(ranked[:promotions], dtype=np.int64)] = True
    return np.ascontiguousarray(mask.reshape(d4.shape))


@dataclass(frozen=True, slots=True)
class StaticRhtQ48Policy:
    method_id: str
    policy_revision: str
    selector_revision: str
    codec_revision: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_manifest_sha256: str
    transformers_version: str
    identity_artifact_sha256: str
    source_commit: str
    geometry: StaticRhtQ468Geometry
    calibration_manifest_sha256: str
    calibration_scores_sha256: str
    promoted_rows: int
    rht_seed: int
    packed_precision_mask: np.ndarray
    pool_offsets: np.ndarray
    pool_counts: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or _METHOD_RE.fullmatch(self.method_id) is None:
            raise ValueError("method_id must use lowercase identifier characters")
        _validate_revision(self.policy_revision, name="policy_revision")
        _validate_revision(self.selector_revision, name="selector_revision")
        _validate_revision(self.codec_revision, name="codec_revision")
        if self.policy_revision != STATIC_Q48_POLICY_REVISION:
            raise ValueError("unsupported static Q48 policy revision")
        if self.selector_revision != STATIC_Q48_SELECTOR_REVISION:
            raise ValueError("unsupported static Q48 selector revision")
        if self.codec_revision != STATIC_Q48_CODEC_REVISION:
            raise ValueError("unsupported static Q48 codec revision")
        _validate_identity(self.model_id, name="model_id")
        _validate_git_revision(self.model_revision, name="model_revision")
        _validate_identity(self.tokenizer_id, name="tokenizer_id")
        _validate_git_revision(self.tokenizer_revision, name="tokenizer_revision")
        _validate_sha256(self.tokenizer_manifest_sha256, name="tokenizer_manifest_sha256")
        _validate_transformers_version(self.transformers_version)
        _validate_sha256(self.identity_artifact_sha256, name="identity_artifact_sha256")
        _validate_git_revision(self.source_commit, name="source_commit")
        if not isinstance(self.geometry, StaticRhtQ468Geometry):
            raise TypeError("geometry must be a StaticRhtQ468Geometry")
        _validate_sha256(self.calibration_manifest_sha256, name="calibration_manifest_sha256")
        _validate_sha256(self.calibration_scores_sha256, name="calibration_scores_sha256")
        promotions = _validate_integer(self.promoted_rows, name="promoted_rows")
        if promotions > self.geometry.total_rows:
            raise ValueError("promoted_rows exceeds the number of state rows")
        if self.method_id in {
            STATIC_Q468_PRIMARY_METHOD,
            STATIC_Q468_ABLATION_METHOD,
            STATIC_Q468_MSE_METHOD,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
            STATIC_Q468_UNIFORM_Q4_METHOD,
            STATIC_Q468_UNIFORM_Q8_METHOD,
        }:
            raise ValueError("reserved static Q468 method cannot identify a Q48 policy")
        if self.method_id == STATIC_Q48_COMPARATOR_METHOD:
            if self.geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
                raise ValueError("reserved static Q48 method requires the frozen geometry")
            if promotions != FROZEN_STATIC_Q48_PROMOTIONS:
                raise ValueError("reserved static Q48 method has the wrong exact-P budget")
            if (
                self.model_id != PRIMARY_MODEL_ID
                or self.model_revision != PRIMARY_MODEL_REVISION
                or self.tokenizer_id != PRIMARY_TOKENIZER_ID
                or self.tokenizer_revision != PRIMARY_TOKENIZER_REVISION
                or self.transformers_version != FROZEN_TRANSFORMERS_VERSION
            ):
                raise ValueError("reserved static Q48 method requires the frozen model identity")
        if self.rht_seed != RHT_SEED:
            raise ValueError(f"rht_seed must equal the codec seed {RHT_SEED}")
        packed = _owned_array(
            self.packed_precision_mask,
            name="packed_precision_mask",
            dtype=np.dtype("u1"),
        )
        offsets = _owned_array(
            self.pool_offsets,
            name="pool_offsets",
            dtype=np.dtype("uint16"),
        )
        mask = unpack_precision_mask(packed, self.geometry.total_rows)
        if offsets.size != self.geometry.total_rows:
            raise ValueError("pool_offsets must contain one uint16 value per state row")
        if int(np.count_nonzero(mask)) != promotions:
            raise ValueError("precision-mask population does not match promoted_rows")
        expected_offsets, expected_counts = _expected_binary_pool_offsets(mask)
        if not isinstance(self.pool_counts, tuple) or len(self.pool_counts) != 2:
            raise TypeError("pool_counts must be a two-integer tuple")
        for index, count in enumerate(self.pool_counts):
            _validate_integer(count, name=f"pool_counts[{index}]")
        if self.pool_counts != expected_counts:
            raise ValueError(
                f"pool_counts {self.pool_counts} do not match precision mask {expected_counts}"
            )
        if sum(self.pool_counts) != self.geometry.total_rows:
            raise ValueError("pool_counts do not cover every state row")
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("pool_offsets are not canonical per-pool prefix offsets")
        object.__setattr__(self, "packed_precision_mask", packed)
        object.__setattr__(self, "pool_offsets", offsets)

    def high_precision_mask(self) -> np.ndarray:
        return unpack_precision_mask(
            self.packed_precision_mask, self.geometry.total_rows
        ).reshape(self.geometry.layers, self.geometry.heads, self.geometry.key_rows)

    @property
    def mask_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"recurquant.static-q48-mask.v1\0")
        digest.update(bytes.fromhex(self.geometry.geometry_sha256))
        digest.update(self.promoted_rows.to_bytes(8, "little", signed=False))
        digest.update(self.packed_precision_mask.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def pool_offsets_sha256(self) -> str:
        return _sha256_bytes(
            self.pool_offsets.astype("<u2", copy=False).tobytes(order="C")
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_scores_sha256": self.calibration_scores_sha256,
            "codec_revision": self.codec_revision,
            "geometry": self.geometry.canonical_dict(),
            "geometry_sha256": self.geometry.geometry_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "mask_sha256": self.mask_sha256,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "packed_precision_mask_b64": _array_b64(
                self.packed_precision_mask, dtype=np.dtype("u1")
            ),
            "policy_revision": self.policy_revision,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_le_b64": _array_b64(
                self.pool_offsets, dtype=np.dtype("<u2")
            ),
            "pool_offsets_sha256": self.pool_offsets_sha256,
            "promoted_rows": self.promoted_rows,
            "rht_seed": self.rht_seed,
            "selector_revision": self.selector_revision,
            "source_commit": self.source_commit,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "transformers_version": self.transformers_version,
        }

    @property
    def policy_sha256(self) -> str:
        return _sha256_bytes(_policy_canonical_json(self._content_dict()))


def build_static_rht_q48_policy(
    d4: np.ndarray,
    d8: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
    promoted_rows: int,
    calibration_manifest_sha256: str,
    identity_artifact_sha256: str,
    tokenizer_manifest_sha256: str,
    source_commit: str,
    calibration_scores_sha256: str | None = None,
    method_id: str | None = None,
    policy_revision: str = STATIC_Q48_POLICY_REVISION,
    selector_revision: str = STATIC_Q48_SELECTOR_REVISION,
    codec_revision: str = STATIC_Q48_CODEC_REVISION,
    model_id: str = PRIMARY_MODEL_ID,
    model_revision: str = PRIMARY_MODEL_REVISION,
    tokenizer_id: str = PRIMARY_TOKENIZER_ID,
    tokenizer_revision: str = PRIMARY_TOKENIZER_REVISION,
    transformers_version: str = FROZEN_TRANSFORMERS_VERSION,
    rht_seed: int = RHT_SEED,
) -> StaticRhtQ48Policy:
    normalized = _normalize_q48_distortions(d4, d8, geometry=geometry)
    scores_sha256 = static_q48_distortion_sha256(*normalized, geometry=geometry)
    if calibration_scores_sha256 is not None:
        _validate_sha256(calibration_scores_sha256, name="calibration_scores_sha256")
        if calibration_scores_sha256 != scores_sha256:
            raise ValueError("calibration_scores_sha256 does not match supplied distortions")
    mask = allocate_exact_q48_mask(*normalized, promoted_rows=promoted_rows).reshape(-1)
    offsets, counts = _expected_binary_pool_offsets(mask)
    return StaticRhtQ48Policy(
        method_id=method_id or f"rht_q48_static_p{promoted_rows}",
        policy_revision=policy_revision,
        selector_revision=selector_revision,
        codec_revision=codec_revision,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        transformers_version=transformers_version,
        identity_artifact_sha256=identity_artifact_sha256,
        source_commit=source_commit,
        geometry=geometry,
        calibration_manifest_sha256=calibration_manifest_sha256,
        calibration_scores_sha256=scores_sha256,
        promoted_rows=promoted_rows,
        rht_seed=rht_seed,
        packed_precision_mask=pack_precision_mask(mask),
        pool_offsets=offsets,
        pool_counts=counts,
    )


def serialize_static_rht_q48_policy(policy: StaticRhtQ48Policy) -> bytes:
    if not isinstance(policy, StaticRhtQ48Policy):
        raise TypeError("policy must be a StaticRhtQ48Policy")
    content = policy._content_dict()
    return _policy_canonical_json(
        {
            "content": content,
            "policy_sha256": _sha256_bytes(_policy_canonical_json(content)),
            "schema": STATIC_Q48_POLICY_SCHEMA,
        }
    ) + b"\n"


def deserialize_static_rht_q48_policy(data: bytes) -> StaticRhtQ48Policy:
    if not isinstance(data, bytes):
        raise TypeError("serialized policy must be bytes")
    try:
        root = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("serialized policy is not valid UTF-8 JSON") from error
    envelope = _expect_keys(
        root,
        name="policy envelope",
        expected={"content", "policy_sha256", "schema"},
    )
    if envelope["schema"] != STATIC_Q48_POLICY_SCHEMA:
        raise ValueError("unsupported static Q48 policy schema")
    declared_digest = _validate_sha256(envelope["policy_sha256"], name="policy_sha256")
    content = _expect_keys(
        envelope["content"],
        name="policy content",
        expected={
            "calibration_manifest_sha256",
            "calibration_scores_sha256",
            "codec_revision",
            "geometry",
            "geometry_sha256",
            "identity_artifact_sha256",
            "mask_sha256",
            "method_id",
            "model_id",
            "model_revision",
            "packed_precision_mask_b64",
            "policy_revision",
            "pool_counts",
            "pool_offsets_le_b64",
            "pool_offsets_sha256",
            "promoted_rows",
            "rht_seed",
            "selector_revision",
            "source_commit",
            "tokenizer_id",
            "tokenizer_manifest_sha256",
            "tokenizer_revision",
            "transformers_version",
        },
    )
    if _sha256_bytes(_policy_canonical_json(content)) != declared_digest:
        raise ValueError("policy_sha256 does not authenticate policy content")
    geometry = _parse_policy_geometry(content["geometry"])
    if content["geometry_sha256"] != geometry.geometry_sha256:
        raise ValueError("geometry_sha256 does not authenticate geometry")
    mask_bytes = _decode_b64(
        content["packed_precision_mask_b64"], name="packed_precision_mask_b64"
    )
    offset_bytes = _decode_b64(
        content["pool_offsets_le_b64"], name="pool_offsets_le_b64"
    )
    if len(mask_bytes) != math.ceil(geometry.total_rows / 8):
        raise ValueError("packed precision mask byte length does not match geometry")
    if len(offset_bytes) != geometry.total_rows * 2:
        raise ValueError("pool offset byte length does not match geometry")
    counts_value = content["pool_counts"]
    if not isinstance(counts_value, list):
        raise TypeError("pool_counts must be a list")
    policy = StaticRhtQ48Policy(
        method_id=content["method_id"],
        policy_revision=content["policy_revision"],
        selector_revision=content["selector_revision"],
        codec_revision=content["codec_revision"],
        model_id=content["model_id"],
        model_revision=content["model_revision"],
        tokenizer_id=content["tokenizer_id"],
        tokenizer_revision=content["tokenizer_revision"],
        tokenizer_manifest_sha256=content["tokenizer_manifest_sha256"],
        transformers_version=content["transformers_version"],
        identity_artifact_sha256=content["identity_artifact_sha256"],
        source_commit=content["source_commit"],
        geometry=geometry,
        calibration_manifest_sha256=content["calibration_manifest_sha256"],
        calibration_scores_sha256=content["calibration_scores_sha256"],
        promoted_rows=content["promoted_rows"],
        rht_seed=content["rht_seed"],
        packed_precision_mask=np.frombuffer(mask_bytes, dtype="u1").copy(),
        pool_offsets=np.frombuffer(offset_bytes, dtype="<u2").astype(np.uint16, copy=True),
        pool_counts=tuple(counts_value),
    )
    if policy.policy_sha256 != declared_digest:
        raise ValueError("reconstructed policy digest differs from policy_sha256")
    if content["mask_sha256"] != policy.mask_sha256:
        raise ValueError("mask_sha256 does not authenticate the precision mask")
    if content["pool_offsets_sha256"] != policy.pool_offsets_sha256:
        raise ValueError("pool_offsets_sha256 does not authenticate pool offsets")
    if serialize_static_rht_q48_policy(policy) != data:
        raise ValueError("serialized policy is not in canonical form")
    return policy


# Calibration score contract ------------------------------------------------

CalibrationFamily: TypeAlias = Literal["mbpp", "pg19", "ruler"]
RulerCategory: TypeAlias = Literal[
    "retrieval",
    "multi_hop_tracing",
    "aggregation",
    "question_answering",
]
SplitHalf: TypeAlias = Literal["a", "b"]
UnweightedSelectorProfile: TypeAlias = Literal[
    "rht_q468_static_mse_k29334",
    "rht_q468_static_diag_empirical_fisher_h1_k29334",
]

FROZEN_ANCHOR_COUNT = 16
CALIBRATION_FAMILY_ORDER = ("mbpp", "pg19", "ruler")
RULER_CATEGORY_ORDER: tuple[RulerCategory, ...] = (
    "retrieval",
    "multi_hop_tracing",
    "aggregation",
    "question_answering",
)
CALIBRATION_SPLIT_NAMESPACE = "recurquant.experiment013.calibration-split.v1\0"

MIN_SPLIT_HALF_SPEARMAN = 0.70
MIN_SPLIT_HALF_Q8_JACCARD = 0.50
MAX_LAYER_MEAN_BITWIDTH_SHIFT = 0.25

CALIBRATION_SCORE_ARTIFACT_KIND = "recurquant_experiment013_static_q468_scores"
CALIBRATION_SCORE_ARTIFACT_SCHEMA_VERSION = 1
CALIBRATION_SCORE_ARTIFACT_REVISION = "experiment-013-static-q468-scores-v1"
CALIBRATION_SCORE_ARTIFACT_PROFILE = (
    "experiment-013-qwen35-0.8b-static-q468-frozen-v1"
)
GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND = "recurquant_static_q468_scores_generic"
GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION = "static-q468-scores-generic-v1"
GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE = "generic-static-q468-calibration-v1"
CALIBRATION_SCORE_DTYPE = "float64-le"

SPLIT_HALF_STABILITY_ARTIFACT_KIND = (
    "recurquant_experiment013_static_q468_split_half_stability"
)
SPLIT_HALF_STABILITY_ARTIFACT_SCHEMA_VERSION = 1
SPLIT_HALF_STABILITY_ARTIFACT_REVISION = (
    "experiment-013-static-q468-split-half-stability-v1"
)
SPLIT_HALF_STABILITY_ARTIFACT_PROFILE = (
    "experiment-013-qwen35-0.8b-static-k29334-split-half-frozen-v1"
)

GENERIC_REDUCTION_PROFILE = "generic-anchor-row-reduction-v1"
FROZEN_REDUCTION_PROFILE = "experiment-013-qwen35-0.8b-anchor-reduction-v1"
FROZEN_UNWEIGHTED_MSE_PROFILE = "rht_q468_static_mse_k29334"
FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE = (
    "rht_q468_static_diag_empirical_fisher_h1_k29334"
)
FROZEN_FISHER_HORIZON = 1
FROZEN_SOURCE_AXIS_ORDER = ("anchor", "layer", "head", "key_row")
_SOURCE_TENSOR_NAMES = ("query_energy", "q4_mse", "q6_mse", "q8_mse")

COMPARATOR_SCORE_ARTIFACT_KIND = (
    "recurquant_experiment013_static_q468_comparator_scores"
)
COMPARATOR_SCORE_ARTIFACT_SCHEMA_VERSION = 1
COMPARATOR_SCORE_ARTIFACT_REVISION = (
    "experiment-013-static-q468-comparator-scores-v1"
)
COMPARATOR_SCORE_ARTIFACT_PROFILE = (
    "experiment-013-qwen35-0.8b-static-q468-comparators-frozen-v1"
)
FROZEN_COMPARATOR_PROFILE_ORDER: tuple[UnweightedSelectorProfile, ...] = (
    FROZEN_UNWEIGHTED_MSE_PROFILE,
    FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
)
FROZEN_COMPARATOR_ENDPOINT_AXIS_ORDER = (
    "endpoint_position",
    "layer",
    "head",
    "key_row",
)
FROZEN_COMPARATOR_POSITION_CONTRACTS = {
    FROZEN_UNWEIGHTED_MSE_PROFILE: "A(T)=frozen_anchor_positions(T)",
    FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE: "B(T)=frozen_anchor_positions(T-2)",
}

_SCORE_HASH_DOMAIN = b"recurquant.experiment013.sequence-scores.v1\0"
_CODE_MAP_HASH_DOMAIN = b"recurquant.static-q468-code-map.v1\0"
_IDENTITY_RECORD_HASH_DOMAIN = b"recurquant.experiment013.identity-record.v1\0"
_COMPARATOR_AGGREGATE_SCORE_HASH_DOMAIN = (
    b"recurquant.experiment013.comparator-aggregate-score.v1\0"
)
_TOKEN_SPAN_ORDER = (
    "prefill_start",
    "prefill_stop",
    "scored_start",
    "scored_stop",
    "cache_exposed_start",
    "cache_exposed_stop",
)
_IDENTITY_RECORD_PAYLOAD_FIELDS = frozenset(
    {
        "family",
        "canonical_id",
        "config",
        "selection_rank",
        "selection_sha256",
        "seed",
        "configured_length",
        "sequence_length",
        "ruler_category",
        "generator_receipt_sha256",
        "source_content_sha256",
        "formatted_content_sha256",
        "prompt_token_ids_sha256",
        "target_token_ids_sha256",
        "sequence_token_ids_sha256",
        "tokenizer_manifest_sha256",
        "token_span",
        "anchor_manifest_sha256",
        "fisher_boundary",
    }
)
_AGGREGATION_CONTRACT = {
    "accumulator": "cpu-float64",
    "anchor_reduction": "equal-weight arithmetic mean within each sequence",
    "bitwidths": [4, 6, 8],
    "broad_family_reduction": "equal-weight arithmetic mean of MBPP, PG19, and RULER",
    "ruler_reduction": (
        "equal-weight sequence mean within each of four categories, then "
        "equal-weight category macro"
    ),
    "sequence_reduction": "equal-weight arithmetic mean within MBPP and PG19",
}
_COMPARATOR_AGGREGATION_CONTRACT = {
    **_AGGREGATION_CONTRACT,
    "endpoint_reduction": "equal-weight CPU-float64 mean over frozen positions",
    "profiles": list(FROZEN_COMPARATOR_PROFILE_ORDER),
}


class CalibrationArtifactError(ValueError):
    """Raised when a calibration artifact fails closed."""


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _canonical_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{name} must use NFC Unicode normalization")
    if "\0" in value:
        raise ValueError(f"{name} cannot contain a NUL separator")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _metadata(
    *,
    family: object,
    config: object,
    ruler_category: object,
    canonical_id: object,
    seed: object,
    configured_length: object,
    token_count: object,
) -> tuple[CalibrationFamily, str, RulerCategory | None, str, int | None, int | None, int]:
    if family not in CALIBRATION_FAMILY_ORDER:
        raise ValueError("family must be one of mbpp, pg19, or ruler")
    normalized_family = cast(CalibrationFamily, family)
    normalized_config = _canonical_text(config, name="config")
    if normalized_family == "ruler":
        if ruler_category not in RULER_CATEGORY_ORDER:
            raise ValueError("ruler_category differs from the frozen category set")
        normalized_category = cast(RulerCategory, ruler_category)
        normalized_configured_length = _strict_positive_int(
            configured_length, name="configured_length"
        )
    else:
        if ruler_category is not None:
            raise ValueError("ruler_category must be None for MBPP and PG19")
        if configured_length is not None:
            raise ValueError("configured_length must be None for MBPP and PG19")
        normalized_category = None
        normalized_configured_length = None
    normalized_id = _canonical_text(canonical_id, name="canonical_id")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
        if seed < 0:
            raise ValueError("seed must be non-negative")
    normalized_tokens = _strict_positive_int(token_count, name="token_count")
    if (
        normalized_configured_length is not None
        and normalized_tokens > normalized_configured_length
    ):
        raise ValueError("token_count cannot exceed the RULER configured_length")
    return (
        normalized_family,
        normalized_config,
        normalized_category,
        normalized_id,
        cast(int | None, seed),
        normalized_configured_length,
        normalized_tokens,
    )


def _resolver_json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _resolver_json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolver_json_compatible(item) for item in value]
    return value


def _resolver_canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _resolver_json_compatible(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def calibration_identity_record_manifest_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    entries: list[dict[str, object]] = []
    for index, record in enumerate(records):
        try:
            family = record["family"]
            category = record["ruler_category"]
            config = record["config"]
            canonical_id = record["canonical_id"]
            seed = record["seed"]
            configured_length = record["configured_length"]
            sequence_length = record["sequence_length"]
            record_hash = record["identity_record_sha256"]
        except KeyError as exc:
            raise ValueError(f"identity manifest record {index} is incomplete") from exc
        _metadata(
            family=family,
            config=config,
            ruler_category=category,
            canonical_id=canonical_id,
            seed=seed,
            configured_length=configured_length,
            token_count=sequence_length,
        )
        entries.append(
            {
                "canonical_id": canonical_id,
                "config": config,
                "configured_length": configured_length,
                "family": family,
                "identity_record_sha256": _sha256(
                    record_hash, name=f"identity manifest record {index} SHA-256"
                ),
                "ruler_category": category,
                "seed": seed,
                "sequence_length": sequence_length,
            }
        )
    if not entries:
        raise ValueError("identity record manifest cannot be empty")
    entries.sort(
        key=lambda item: (
            CALIBRATION_FAMILY_ORDER.index(cast(str, item["family"])),
            "" if item["ruler_category"] is None else str(item["ruler_category"]),
            str(item["config"]),
            str(item["canonical_id"]),
            -1 if item["seed"] is None else cast(int, item["seed"]),
            -1
            if item["configured_length"] is None
            else cast(int, item["configured_length"]),
            cast(int, item["sequence_length"]),
        )
    )
    identities = [
        (
            item["family"],
            item["ruler_category"],
            item["config"],
            item["canonical_id"],
            item["seed"],
            item["configured_length"],
            item["sequence_length"],
        )
        for item in entries
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("identity record manifest contains duplicate sequence identities")
    return hashlib.sha256(_resolver_canonical_json_bytes(entries)).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationSourceTensorContract:
    reduction_profile: str
    axis_order: tuple[str, ...]
    trailing_shape: tuple[int, ...]
    dtypes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _canonical_text(self.reduction_profile, name="reduction_profile")
        if not isinstance(self.axis_order, tuple) or len(self.axis_order) < 2:
            raise ValueError("source axis_order must contain anchor and at least one row axis")
        for axis in self.axis_order:
            _canonical_text(axis, name="source axis name")
        if self.axis_order[0] != "anchor" or len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError("source axis_order must start with unique anchor axis")
        if (
            not isinstance(self.trailing_shape, tuple)
            or len(self.trailing_shape) != len(self.axis_order) - 1
        ):
            raise ValueError("source trailing_shape must match non-anchor axes")
        for index, extent in enumerate(self.trailing_shape):
            _strict_positive_int(extent, name=f"source trailing_shape[{index}]")
        if not isinstance(self.dtypes, tuple):
            raise TypeError("source dtypes must be a tuple")
        if tuple(name for name, _dtype in self.dtypes) != _SOURCE_TENSOR_NAMES:
            raise ValueError(
                "source dtypes must cover query_energy, q4_mse, q6_mse, and q8_mse"
            )
        for name, dtype in self.dtypes:
            _canonical_text(name, name="source tensor name")
            _canonical_text(dtype, name=f"{name} source dtype")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "axis_order": list(self.axis_order),
            "dtypes": {name: dtype for name, dtype in self.dtypes},
            "reduction_profile": self.reduction_profile,
            "trailing_shape": list(self.trailing_shape),
        }


FROZEN_SOURCE_TENSOR_CONTRACT = CalibrationSourceTensorContract(
    reduction_profile=FROZEN_REDUCTION_PROFILE,
    axis_order=FROZEN_SOURCE_AXIS_ORDER,
    trailing_shape=(
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY.layers,
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY.heads,
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY.key_rows,
    ),
    dtypes=tuple((name, "torch.float64") for name in _SOURCE_TENSOR_NAMES),
)


def _cpu_fp64_scores(
    value: object,
    *,
    name: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if expected_rows is not None and value.size != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} rows")
    normalized = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise ValueError(f"{name} must contain only finite, non-negative values")
    return _immutable_array_copy(normalized)


def _score_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).astype(
        "<f8", copy=False
    ).tobytes(order="C")


def _hash_score_triplet(
    d4: np.ndarray,
    d6: np.ndarray,
    d8: np.ndarray,
    *,
    domain: bytes,
    metadata: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(canonical_json_bytes(metadata))
    for label, value in zip((b"D4\0", b"D6\0", b"D8\0"), (d4, d6, d8), strict=True):
        digest.update(label)
        digest.update(_score_bytes(value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationAggregate:
    d4: np.ndarray
    d6: np.ndarray
    d8: np.ndarray
    family_sequence_counts: tuple[tuple[str, int], ...]
    ruler_category_sequence_counts: tuple[tuple[str, int], ...]
    sequence_score_manifest_sha256: str
    source_contract: CalibrationSourceTensorContract
    identity_record_manifest_sha256: str | None = None

    @property
    def row_count(self) -> int:
        return int(self.d4.size)

    def scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.d4, self.d6, self.d8


def _validate_aggregate(aggregate: object, *, expected_rows: int) -> CalibrationAggregate:
    if not isinstance(aggregate, CalibrationAggregate):
        raise TypeError("aggregate must be a CalibrationAggregate")
    values = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=expected_rows)
        for name, value in (("D4", aggregate.d4), ("D6", aggregate.d6), ("D8", aggregate.d8))
    )
    if not isinstance(aggregate.family_sequence_counts, tuple):
        raise ValueError("family_sequence_counts must be a tuple")
    if tuple(name for name, _count in aggregate.family_sequence_counts) != CALIBRATION_FAMILY_ORDER:
        raise ValueError("family_sequence_counts must use MBPP, PG19, RULER order")
    for name, count in aggregate.family_sequence_counts:
        _canonical_text(name, name="family count key")
        _strict_positive_int(count, name=f"{name} sequence count")
    if tuple(name for name, _count in aggregate.ruler_category_sequence_counts) != (
        RULER_CATEGORY_ORDER
    ):
        raise ValueError("RULER category counts must use the frozen category order")
    for name, count in aggregate.ruler_category_sequence_counts:
        _canonical_text(name, name="RULER category")
        _strict_positive_int(count, name=f"RULER {name} sequence count")
    family_count_map = dict(aggregate.family_sequence_counts)
    if family_count_map["ruler"] != sum(
        count for _name, count in aggregate.ruler_category_sequence_counts
    ):
        raise ValueError("RULER broad-family count must equal its four category counts")
    _sha256(
        aggregate.sequence_score_manifest_sha256,
        name="sequence_score_manifest_sha256",
    )
    if not isinstance(aggregate.source_contract, CalibrationSourceTensorContract):
        raise TypeError("source_contract must be a CalibrationSourceTensorContract")
    if math.prod(aggregate.source_contract.trailing_shape) != expected_rows:
        raise ValueError("source tensor contract does not match aggregate row count")
    if aggregate.source_contract == FROZEN_SOURCE_TENSOR_CONTRACT:
        identity_manifest_sha256 = _sha256(
            aggregate.identity_record_manifest_sha256,
            name="identity_record_manifest_sha256",
        )
    else:
        if aggregate.identity_record_manifest_sha256 is not None:
            raise ValueError("generic aggregate cannot claim a frozen identity-record manifest")
        identity_manifest_sha256 = None
    return CalibrationAggregate(
        d4=values[0],
        d6=values[1],
        d8=values[2],
        family_sequence_counts=aggregate.family_sequence_counts,
        ruler_category_sequence_counts=aggregate.ruler_category_sequence_counts,
        sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
        source_contract=aggregate.source_contract,
        identity_record_manifest_sha256=identity_manifest_sha256,
    )


def allocate_static_q468_code_map(
    aggregate: CalibrationAggregate,
    *,
    marginal_steps: int,
) -> np.ndarray:
    rows = aggregate.row_count
    normalized = _validate_aggregate(aggregate, expected_rows=rows)
    steps = _strict_nonnegative_int(marginal_steps, name="marginal_steps")
    if steps > 2 * rows:
        raise ValueError("marginal_steps exceeds two steps per row")
    codes = allocate_exact_multibit_codes(
        *(value.reshape(1, rows) for value in normalized.scores()),
        marginal_steps=steps,
    ).reshape(-1)
    if codes.dtype != np.uint8 or int(codes.astype(np.int64).sum()) != steps:
        raise RuntimeError("exact allocator did not satisfy the requested marginal budget")
    return np.ascontiguousarray(codes)


def static_q468_code_map_sha256(
    codes: np.ndarray,
    *,
    geometry: StaticRhtQ468Geometry,
    marginal_steps: int,
) -> str:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    normalized = _precision_codes(codes, name="codes", expected_rows=geometry.total_rows)
    steps = _strict_nonnegative_int(marginal_steps, name="marginal_steps")
    if int(normalized.astype(np.int64).sum()) != steps:
        raise ValueError("code-map marginal sum does not match marginal_steps")
    packed = pack_precision_codes(normalized)
    digest = hashlib.sha256()
    digest.update(_CODE_MAP_HASH_DOMAIN)
    digest.update(bytes.fromhex(geometry.geometry_sha256))
    digest.update(steps.to_bytes(8, "little", signed=False))
    digest.update(packed.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DecodedCalibrationScoreArtifact:
    artifact_kind: str
    artifact_profile: str
    artifact_revision: str
    aggregate: CalibrationAggregate
    geometry: StaticRhtQ468Geometry
    calibration_identity_sha256: str
    calibration_scores_sha256: str
    allocations: tuple[tuple[int, np.ndarray, str], ...]
    canonical_evidence_sha256: str
    file_sha256: str


def _reject_json_constant(value: str) -> None:
    raise CalibrationArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationArtifactError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CalibrationArtifactError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise CalibrationArtifactError(f"{context} must be a JSON array")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise CalibrationArtifactError(
            f"{context} fields drifted; missing={missing}, extra={extra}"
        )


def _artifact_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _artifact_sha(value: object, *, context: str) -> str:
    try:
        return _sha256(value, name=context)
    except (TypeError, ValueError) as exc:
        raise CalibrationArtifactError(str(exc)) from exc


def _parse_counts(
    value: object,
    *,
    context: str,
    key_name: str,
    expected_names: tuple[str, ...] | None = None,
) -> tuple[tuple[str, int], ...]:
    rows = _sequence(value, context=context)
    result: list[tuple[str, int]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, context=f"{context}[{index}]")
        _exact_keys(row, {key_name, "count"}, context=f"{context}[{index}]")
        try:
            name = _canonical_text(row[key_name], name=f"{context}[{index}].{key_name}")
        except (TypeError, ValueError) as exc:
            raise CalibrationArtifactError(str(exc)) from exc
        count = _artifact_int(row["count"], context=f"{context}[{index}].count", minimum=1)
        result.append((name, count))
    names = tuple(name for name, _count in result)
    if len(set(names)) != len(names):
        raise CalibrationArtifactError(f"{context} contains duplicate names")
    if expected_names is not None and names != expected_names:
        raise CalibrationArtifactError(f"{context} differs from the frozen family order")
    if expected_names is None and tuple(sorted(result)) != tuple(result):
        raise CalibrationArtifactError(f"{context} must be sorted canonically")
    return tuple(result)


def _parse_geometry(value: object) -> StaticRhtQ468Geometry:
    geometry = _mapping(value, context="evidence.geometry")
    _exact_keys(
        geometry,
        {"heads", "key_rows", "layer_indices", "target_resident_bytes", "value_width"},
        context="evidence.geometry",
    )
    raw_layers = _sequence(geometry["layer_indices"], context="evidence.geometry.layer_indices")
    layers = tuple(
        _artifact_int(item, context=f"evidence.geometry.layer_indices[{index}]")
        for index, item in enumerate(raw_layers)
    )
    try:
        return StaticRhtQ468Geometry(
            layer_indices=layers,
            heads=_artifact_int(geometry["heads"], context="evidence.geometry.heads", minimum=1),
            key_rows=_artifact_int(
                geometry["key_rows"], context="evidence.geometry.key_rows", minimum=1
            ),
            value_width=_artifact_int(
                geometry["value_width"], context="evidence.geometry.value_width", minimum=1
            ),
            target_resident_bytes=_artifact_int(
                geometry["target_resident_bytes"],
                context="evidence.geometry.target_resident_bytes",
                minimum=1,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationArtifactError(f"invalid geometry: {exc}") from exc


def _parse_source_tensor_contract(value: object) -> CalibrationSourceTensorContract:
    contract = _mapping(value, context="evidence.source_tensor_contract")
    _exact_keys(
        contract,
        {"axis_order", "dtypes", "reduction_profile", "trailing_shape"},
        context="evidence.source_tensor_contract",
    )
    raw_axes = _sequence(
        contract["axis_order"], context="evidence.source_tensor_contract.axis_order"
    )
    raw_shape = _sequence(
        contract["trailing_shape"], context="evidence.source_tensor_contract.trailing_shape"
    )
    raw_dtypes = _mapping(contract["dtypes"], context="evidence.source_tensor_contract.dtypes")
    if set(raw_dtypes) != set(_SOURCE_TENSOR_NAMES):
        raise CalibrationArtifactError(
            "source tensor dtypes must cover the exact source tensor names"
        )
    try:
        return CalibrationSourceTensorContract(
            reduction_profile=_canonical_text(
                contract["reduction_profile"], name="source reduction_profile"
            ),
            axis_order=tuple(
                _canonical_text(axis, name=f"source axis_order[{index}]")
                for index, axis in enumerate(raw_axes)
            ),
            trailing_shape=tuple(
                _artifact_int(
                    extent,
                    context=f"evidence.source_tensor_contract.trailing_shape[{index}]",
                    minimum=1,
                )
                for index, extent in enumerate(raw_shape)
            ),
            dtypes=tuple(
                (name, _canonical_text(raw_dtypes[name], name=f"source dtype {name}"))
                for name in _SOURCE_TENSOR_NAMES
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CalibrationArtifactError):
            raise
        raise CalibrationArtifactError(f"invalid source tensor contract: {exc}") from exc


def _decode_score_triplet(
    value: object,
    *,
    geometry: StaticRhtQ468Geometry,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = _mapping(value, context=context)
    _exact_keys(
        record,
        {"axis_order", "bitwidths", "data_base64", "dtype", "shape"},
        context=context,
    )
    if record["axis_order"] != ["bitwidth", "flattened_layer_head_key_row"]:
        raise CalibrationArtifactError(f"{context}.axis_order drifted")
    if record["bitwidths"] != [4, 6, 8]:
        raise CalibrationArtifactError(f"{context}.bitwidths must be Q4, Q6, Q8")
    if record["dtype"] != CALIBRATION_SCORE_DTYPE:
        raise CalibrationArtifactError(f"{context}.dtype must be float64-le")
    if record["shape"] != [3, geometry.total_rows]:
        raise CalibrationArtifactError(f"{context}.shape differs from geometry")
    encoded = record["data_base64"]
    if not isinstance(encoded, str):
        raise CalibrationArtifactError(f"{context}.data_base64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CalibrationArtifactError(f"{context}.data_base64 is invalid") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise CalibrationArtifactError(f"{context}.data_base64 is not canonical")
    expected_bytes = 3 * geometry.total_rows * 8
    if len(raw) != expected_bytes:
        raise CalibrationArtifactError(
            f"{context} byte length differs: expected {expected_bytes}, got {len(raw)}"
        )
    array = np.frombuffer(raw, dtype="<f8").copy().reshape(3, geometry.total_rows)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise CalibrationArtifactError(f"{context} arrays must be finite and non-negative")
    return cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple(np.ascontiguousarray(array[index], dtype=np.float64) for index in range(3)),
    )


def deserialize_calibration_score_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> DecodedCalibrationScoreArtifact:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    file_sha256 = hashlib.sha256(data).hexdigest()
    if expected_file_sha256 is not None:
        expected = _artifact_sha(expected_file_sha256, context="expected_file_sha256")
        if file_sha256 != expected:
            raise CalibrationArtifactError(
                f"file SHA-256 mismatch: expected {expected}, computed {file_sha256}"
            )
    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, CalibrationArtifactError):
            raise
        raise CalibrationArtifactError(f"artifact is not strict UTF-8 JSON: {exc}") from exc
    root = _mapping(document, context="artifact")
    _exact_keys(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="artifact",
    )
    if root["artifact_kind"] not in {
        CALIBRATION_SCORE_ARTIFACT_KIND,
        GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND,
    }:
        raise CalibrationArtifactError("artifact kind is not a supported calibration profile")
    if (
        _artifact_int(root["schema_version"], context="artifact.schema_version", minimum=1)
        != CALIBRATION_SCORE_ARTIFACT_SCHEMA_VERSION
    ):
        raise CalibrationArtifactError("artifact schema version differs from the frozen value")
    recorded_canonical = _artifact_sha(
        root["canonical_evidence_sha256"], context="artifact.canonical_evidence_sha256"
    )
    evidence = _mapping(root["evidence"], context="artifact.evidence")
    _exact_keys(
        evidence,
        {
            "aggregation_contract",
            "allocations",
            "artifact_profile",
            "artifact_revision",
            "calibration_identity_sha256",
            "calibration_scores_sha256",
            "family_sequence_counts",
            "geometry",
            "geometry_sha256",
            "identity_record_manifest_sha256",
            "ruler_category_sequence_counts",
            "scores",
            "sequence_score_manifest_sha256",
            "source_tensor_contract",
        },
        context="artifact.evidence",
    )
    computed_canonical = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    if recorded_canonical != computed_canonical:
        raise CalibrationArtifactError("canonical evidence SHA-256 mismatch")
    if canonical_json_bytes(root) != data:
        raise CalibrationArtifactError("artifact bytes are not canonical JSON")
    profile_triplet = (
        root["artifact_kind"],
        evidence["artifact_profile"],
        evidence["artifact_revision"],
    )
    official_profile = (
        CALIBRATION_SCORE_ARTIFACT_KIND,
        CALIBRATION_SCORE_ARTIFACT_PROFILE,
        CALIBRATION_SCORE_ARTIFACT_REVISION,
    )
    generic_profile = (
        GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND,
        GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE,
        GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION,
    )
    if profile_triplet != official_profile and profile_triplet != generic_profile:
        raise CalibrationArtifactError("artifact kind, profile, and revision are inconsistent")
    is_official = profile_triplet == official_profile
    if evidence["aggregation_contract"] != _AGGREGATION_CONTRACT:
        raise CalibrationArtifactError("aggregation contract differs from the frozen equation")

    identity_sha256 = _artifact_sha(
        evidence["calibration_identity_sha256"],
        context="evidence.calibration_identity_sha256",
    )
    sequence_manifest_sha256 = _artifact_sha(
        evidence["sequence_score_manifest_sha256"],
        context="evidence.sequence_score_manifest_sha256",
    )
    raw_identity_manifest_sha256 = evidence["identity_record_manifest_sha256"]
    if is_official:
        identity_record_manifest_sha256 = _artifact_sha(
            raw_identity_manifest_sha256,
            context="evidence.identity_record_manifest_sha256",
        )
    else:
        if raw_identity_manifest_sha256 is not None:
            raise CalibrationArtifactError(
                "generic score artifact cannot claim a frozen identity-record manifest"
            )
        identity_record_manifest_sha256 = None
    family_counts = _parse_counts(
        evidence["family_sequence_counts"],
        context="evidence.family_sequence_counts",
        key_name="family",
        expected_names=CALIBRATION_FAMILY_ORDER,
    )
    ruler_counts = _parse_counts(
        evidence["ruler_category_sequence_counts"],
        context="evidence.ruler_category_sequence_counts",
        key_name="category",
        expected_names=RULER_CATEGORY_ORDER,
    )
    geometry = _parse_geometry(evidence["geometry"])
    geometry_sha256 = _artifact_sha(
        evidence["geometry_sha256"], context="evidence.geometry_sha256"
    )
    if geometry.geometry_sha256 != geometry_sha256:
        raise CalibrationArtifactError("geometry SHA-256 mismatch")
    source_contract = _parse_source_tensor_contract(evidence["source_tensor_contract"])
    if is_official:
        if geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
            raise CalibrationArtifactError(
                "official Experiment 013 artifact requires the exact frozen geometry"
            )
        if family_counts != (("mbpp", 128), ("pg19", 16), ("ruler", 16)):
            raise CalibrationArtifactError(
                "official Experiment 013 artifact requires MBPP=128, PG19=16, RULER=16"
            )
        if ruler_counts != tuple((category, 4) for category in RULER_CATEGORY_ORDER):
            raise CalibrationArtifactError(
                "official Experiment 013 artifact requires four sequences per RULER category"
            )
        if source_contract != FROZEN_SOURCE_TENSOR_CONTRACT:
            raise CalibrationArtifactError(
                "official Experiment 013 artifact requires the frozen source tensor contract"
            )

    allocation_values = _sequence(evidence["allocations"], context="evidence.allocations")
    if is_official:
        official_budgets: list[int] = []
        for index, raw_allocation in enumerate(allocation_values):
            allocation = _mapping(raw_allocation, context=f"evidence.allocations[{index}]")
            if "marginal_steps" not in allocation:
                raise CalibrationArtifactError(
                    f"evidence.allocations[{index}] is missing marginal_steps"
                )
            official_budgets.append(
                _artifact_int(
                    allocation["marginal_steps"],
                    context=f"evidence.allocations[{index}].marginal_steps",
                )
            )
        if tuple(official_budgets) != (
            FROZEN_STATIC_Q468_ABLATION_STEPS,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
        ):
            raise CalibrationArtifactError(
                "official Experiment 013 artifact requires exactly K27030 and K29334"
            )

    score_values = _decode_score_triplet(
        evidence["scores"], geometry=geometry, context="evidence.scores"
    )
    aggregate = _validate_aggregate(
        CalibrationAggregate(
            d4=score_values[0],
            d6=score_values[1],
            d8=score_values[2],
            family_sequence_counts=family_counts,
            ruler_category_sequence_counts=ruler_counts,
            sequence_score_manifest_sha256=sequence_manifest_sha256,
            source_contract=source_contract,
            identity_record_manifest_sha256=identity_record_manifest_sha256,
        ),
        expected_rows=geometry.total_rows,
    )
    expected_score_sha256 = _artifact_sha(
        evidence["calibration_scores_sha256"],
        context="evidence.calibration_scores_sha256",
    )
    computed_score_sha256 = static_q468_distortion_sha256(
        *aggregate.scores(), geometry=geometry
    )
    if expected_score_sha256 != computed_score_sha256:
        raise CalibrationArtifactError("calibration score SHA-256 mismatch")
    if not allocation_values:
        raise CalibrationArtifactError("at least one exact allocation is required")
    allocations: list[tuple[int, np.ndarray, str]] = []
    prior_steps = -1
    for index, raw_allocation in enumerate(allocation_values):
        allocation = _mapping(raw_allocation, context=f"evidence.allocations[{index}]")
        _exact_keys(
            allocation,
            {
                "allocator_revision",
                "code_counts_q4_q6_q8",
                "code_map_sha256",
                "marginal_steps",
                "packed_precision_bytes",
            },
            context=f"evidence.allocations[{index}]",
        )
        if allocation["allocator_revision"] != STATIC_Q468_ALLOCATOR_REVISION:
            raise CalibrationArtifactError("allocator revision differs from the frozen value")
        steps = _artifact_int(
            allocation["marginal_steps"],
            context=f"evidence.allocations[{index}].marginal_steps",
        )
        if not prior_steps < steps <= 2 * geometry.total_rows:
            raise CalibrationArtifactError("allocation budgets must be unique and increasing")
        prior_steps = steps
        expected_packed_bytes = math.ceil(geometry.total_rows * 2 / 8)
        if (
            _artifact_int(
                allocation["packed_precision_bytes"],
                context=f"evidence.allocations[{index}].packed_precision_bytes",
            )
            != expected_packed_bytes
        ):
            raise CalibrationArtifactError("packed precision byte count differs from geometry")
        counts_raw = _sequence(
            allocation["code_counts_q4_q6_q8"],
            context=f"evidence.allocations[{index}].code_counts_q4_q6_q8",
        )
        if len(counts_raw) != 3:
            raise CalibrationArtifactError("code counts must contain Q4, Q6, and Q8 counts")
        counts = [
            _artifact_int(
                value,
                context=f"evidence.allocations[{index}].code_counts_q4_q6_q8[{position}]",
            )
            for position, value in enumerate(counts_raw)
        ]
        codes = allocate_static_q468_code_map(aggregate, marginal_steps=steps)
        computed_counts = [int(np.count_nonzero(codes == code)) for code in range(3)]
        if counts != computed_counts or sum(counts) != geometry.total_rows:
            raise CalibrationArtifactError("recorded code counts differ from exact allocation")
        recorded_code_hash = _artifact_sha(
            allocation["code_map_sha256"],
            context=f"evidence.allocations[{index}].code_map_sha256",
        )
        computed_code_hash = static_q468_code_map_sha256(
            codes, geometry=geometry, marginal_steps=steps
        )
        if recorded_code_hash != computed_code_hash:
            raise CalibrationArtifactError(
                "recorded code-map SHA-256 differs from allocation"
            )
        allocations.append((steps, _immutable_array_copy(codes), computed_code_hash))
    return DecodedCalibrationScoreArtifact(
        artifact_kind=cast(str, root["artifact_kind"]),
        artifact_profile=cast(str, evidence["artifact_profile"]),
        artifact_revision=cast(str, evidence["artifact_revision"]),
        aggregate=aggregate,
        geometry=geometry,
        calibration_identity_sha256=identity_sha256,
        calibration_scores_sha256=computed_score_sha256,
        allocations=tuple(allocations),
        canonical_evidence_sha256=computed_canonical,
        file_sha256=file_sha256,
    )


@dataclass(frozen=True, slots=True)
class ComparatorAggregate:
    selector_profile: UnweightedSelectorProfile
    d4: np.ndarray
    d6: np.ndarray
    d8: np.ndarray
    family_sequence_counts: tuple[tuple[str, int], ...]
    ruler_category_sequence_counts: tuple[tuple[str, int], ...]
    position_manifest_sha256: str
    sequence_score_manifest_sha256: str
    identity_record_manifest_sha256: str
    aggregate_scores_sha256: str

    @property
    def row_count(self) -> int:
        return int(self.d4.size)

    def scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.d4, self.d6, self.d8


def _comparator_aggregate_score_sha256(aggregate: ComparatorAggregate) -> str:
    metadata = {
        "family_sequence_counts": [
            {"count": count, "family": family}
            for family, count in aggregate.family_sequence_counts
        ],
        "identity_record_manifest_sha256": aggregate.identity_record_manifest_sha256,
        "position_manifest_sha256": aggregate.position_manifest_sha256,
        "row_count": aggregate.row_count,
        "ruler_category_sequence_counts": [
            {"category": category, "count": count}
            for category, count in aggregate.ruler_category_sequence_counts
        ],
        "selector_profile": aggregate.selector_profile,
        "sequence_score_manifest_sha256": aggregate.sequence_score_manifest_sha256,
    }
    return _hash_score_triplet(
        *aggregate.scores(),
        domain=_COMPARATOR_AGGREGATE_SCORE_HASH_DOMAIN,
        metadata=metadata,
    )


def _validate_frozen_comparator_aggregate(
    aggregate: object,
    *,
    expected_profile: UnweightedSelectorProfile,
) -> ComparatorAggregate:
    if not isinstance(aggregate, ComparatorAggregate):
        raise TypeError("aggregate must be a ComparatorAggregate")
    if aggregate.selector_profile != expected_profile:
        raise ValueError("comparator aggregate selector profile differs from its slot")
    expected_rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    scores = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=expected_rows)
        for name, value in (("D4", aggregate.d4), ("D6", aggregate.d6), ("D8", aggregate.d8))
    )
    if aggregate.family_sequence_counts != (
        ("mbpp", 128),
        ("pg19", 16),
        ("ruler", 16),
    ):
        raise ValueError("frozen comparator counts must be MBPP=128, PG19=16, RULER=16")
    if aggregate.ruler_category_sequence_counts != tuple(
        (category, 4) for category in RULER_CATEGORY_ORDER
    ):
        raise ValueError("each frozen comparator RULER category must contain four sequences")
    for name, value in (
        ("position_manifest_sha256", aggregate.position_manifest_sha256),
        ("sequence_score_manifest_sha256", aggregate.sequence_score_manifest_sha256),
        ("identity_record_manifest_sha256", aggregate.identity_record_manifest_sha256),
        ("aggregate_scores_sha256", aggregate.aggregate_scores_sha256),
    ):
        _sha256(value, name=name)
    normalized = ComparatorAggregate(
        selector_profile=aggregate.selector_profile,
        d4=scores[0],
        d6=scores[1],
        d8=scores[2],
        family_sequence_counts=aggregate.family_sequence_counts,
        ruler_category_sequence_counts=aggregate.ruler_category_sequence_counts,
        position_manifest_sha256=aggregate.position_manifest_sha256,
        sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
        identity_record_manifest_sha256=aggregate.identity_record_manifest_sha256,
        aggregate_scores_sha256=aggregate.aggregate_scores_sha256,
    )
    if normalized.aggregate_scores_sha256 != _comparator_aggregate_score_sha256(normalized):
        raise ValueError("comparator aggregate-score SHA-256 drifted")
    return normalized


def _allocate_frozen_comparator_codes(aggregate: ComparatorAggregate) -> np.ndarray:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    codes = allocate_exact_multibit_codes(
        *(value.reshape(1, rows) for value in aggregate.scores()),
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    ).reshape(-1)
    if (
        codes.dtype != np.uint8
        or codes.shape != (rows,)
        or int(codes.astype(np.int64).sum()) != FROZEN_STATIC_Q468_PRIMARY_STEPS
    ):
        raise RuntimeError("comparator allocator missed exact K29334")
    return np.ascontiguousarray(codes)


@dataclass(frozen=True, slots=True)
class ComparatorSelectorArtifact:
    method_id: UnweightedSelectorProfile
    aggregate: ComparatorAggregate
    position_manifest_sha256: str
    calibration_scores_sha256: str
    marginal_steps: int
    precision_codes: np.ndarray
    code_map_sha256: str


@dataclass(frozen=True, slots=True)
class ComparatorScoreArtifact:
    selectors: Mapping[str, ComparatorSelectorArtifact]
    calibration_identity_sha256: str
    canonical_evidence_sha256: str
    file_sha256: str


def deserialize_comparator_score_artifact(
    data: bytes,
    *,
    expected_calibration_identity_sha256: str | None = None,
) -> ComparatorScoreArtifact:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    file_sha256 = hashlib.sha256(data).hexdigest()
    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, CalibrationArtifactError):
            raise
        raise CalibrationArtifactError(
            f"comparator artifact is not strict UTF-8 JSON: {exc}"
        ) from exc
    root = _mapping(document, context="comparator artifact")
    _exact_keys(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="comparator artifact",
    )
    if root["artifact_kind"] != COMPARATOR_SCORE_ARTIFACT_KIND:
        raise CalibrationArtifactError("comparator artifact kind drifted")
    if (
        _artifact_int(
            root["schema_version"],
            context="comparator artifact.schema_version",
            minimum=1,
        )
        != COMPARATOR_SCORE_ARTIFACT_SCHEMA_VERSION
    ):
        raise CalibrationArtifactError("comparator artifact schema version drifted")
    evidence = _mapping(root["evidence"], context="comparator artifact.evidence")
    _exact_keys(
        evidence,
        {
            "aggregation_contract",
            "artifact_profile",
            "artifact_revision",
            "calibration_identity_sha256",
            "endpoint_tensor_contract",
            "geometry",
            "geometry_sha256",
            "identity_record_manifest_sha256",
            "selectors",
        },
        context="comparator artifact.evidence",
    )
    if evidence["artifact_profile"] != COMPARATOR_SCORE_ARTIFACT_PROFILE:
        raise CalibrationArtifactError("comparator artifact profile drifted")
    if evidence["artifact_revision"] != COMPARATOR_SCORE_ARTIFACT_REVISION:
        raise CalibrationArtifactError("comparator artifact revision drifted")
    if evidence["aggregation_contract"] != _COMPARATOR_AGGREGATION_CONTRACT:
        raise CalibrationArtifactError("comparator aggregation contract drifted")
    geometry = _parse_geometry(evidence["geometry"])
    if geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
        raise CalibrationArtifactError("comparator geometry differs from frozen Qwen3.5 geometry")
    geometry_hash = _artifact_sha(
        evidence["geometry_sha256"],
        context="comparator artifact.evidence.geometry_sha256",
    )
    if geometry_hash != geometry.geometry_sha256:
        raise CalibrationArtifactError("comparator geometry SHA-256 mismatch")
    expected_endpoint_contract = {
        "axis_order": list(FROZEN_COMPARATOR_ENDPOINT_AXIS_ORDER),
        "dtype": CALIBRATION_SCORE_DTYPE,
        "trailing_shape": list(FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape),
    }
    if evidence["endpoint_tensor_contract"] != expected_endpoint_contract:
        raise CalibrationArtifactError("comparator endpoint tensor contract drifted")
    calibration_identity_hash = _artifact_sha(
        evidence["calibration_identity_sha256"],
        context="comparator artifact.evidence.calibration_identity_sha256",
    )
    if expected_calibration_identity_sha256 is not None:
        expected_identity = _artifact_sha(
            expected_calibration_identity_sha256,
            context="expected_calibration_identity_sha256",
        )
        if calibration_identity_hash != expected_identity:
            raise CalibrationArtifactError(
                "comparator calibration identity SHA-256 differs from expected identity"
            )
    identity_manifest_hash = _artifact_sha(
        evidence["identity_record_manifest_sha256"],
        context="comparator artifact.evidence.identity_record_manifest_sha256",
    )
    recorded_canonical_hash = _artifact_sha(
        root["canonical_evidence_sha256"],
        context="comparator artifact.canonical_evidence_sha256",
    )
    computed_canonical_hash = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    if recorded_canonical_hash != computed_canonical_hash:
        raise CalibrationArtifactError("comparator canonical evidence SHA-256 mismatch")
    if canonical_json_bytes(root) != data:
        raise CalibrationArtifactError("comparator artifact bytes are not canonical JSON")

    selector_values = _sequence(
        evidence["selectors"], context="comparator artifact.evidence.selectors"
    )
    if len(selector_values) != len(FROZEN_COMPARATOR_PROFILE_ORDER):
        raise CalibrationArtifactError("comparator artifact must contain exactly two selectors")
    decoded_selectors: dict[str, ComparatorSelectorArtifact] = {}
    for index, (raw_selector, expected_method) in enumerate(
        zip(selector_values, FROZEN_COMPARATOR_PROFILE_ORDER, strict=True)
    ):
        context = f"comparator artifact.evidence.selectors[{index}]"
        selector = _mapping(raw_selector, context=context)
        _exact_keys(
            selector,
            {
                "allocation",
                "calibration_scores_sha256",
                "family_sequence_counts",
                "method_id",
                "position_contract",
                "position_manifest_sha256",
                "ruler_category_sequence_counts",
                "scores",
                "sequence_score_manifest_sha256",
            },
            context=context,
        )
        if selector["method_id"] != expected_method:
            raise CalibrationArtifactError(
                "comparator selectors must contain exactly MSE then diagonal Fisher H1"
            )
        if selector["position_contract"] != FROZEN_COMPARATOR_POSITION_CONTRACTS[expected_method]:
            raise CalibrationArtifactError(f"{context}.position_contract drifted")
        family_counts = _parse_counts(
            selector["family_sequence_counts"],
            context=f"{context}.family_sequence_counts",
            key_name="family",
            expected_names=CALIBRATION_FAMILY_ORDER,
        )
        ruler_counts = _parse_counts(
            selector["ruler_category_sequence_counts"],
            context=f"{context}.ruler_category_sequence_counts",
            key_name="category",
            expected_names=RULER_CATEGORY_ORDER,
        )
        scores = _decode_score_triplet(
            selector["scores"], geometry=geometry, context=f"{context}.scores"
        )
        position_hash = _artifact_sha(
            selector["position_manifest_sha256"],
            context=f"{context}.position_manifest_sha256",
        )
        sequence_manifest_hash = _artifact_sha(
            selector["sequence_score_manifest_sha256"],
            context=f"{context}.sequence_score_manifest_sha256",
        )
        aggregate_score_hash = _artifact_sha(
            selector["calibration_scores_sha256"],
            context=f"{context}.calibration_scores_sha256",
        )
        aggregate = _validate_frozen_comparator_aggregate(
            ComparatorAggregate(
                selector_profile=expected_method,
                d4=scores[0],
                d6=scores[1],
                d8=scores[2],
                family_sequence_counts=family_counts,
                ruler_category_sequence_counts=ruler_counts,
                position_manifest_sha256=position_hash,
                sequence_score_manifest_sha256=sequence_manifest_hash,
                identity_record_manifest_sha256=identity_manifest_hash,
                aggregate_scores_sha256=aggregate_score_hash,
            ),
            expected_profile=expected_method,
        )
        allocation = _mapping(selector["allocation"], context=f"{context}.allocation")
        _exact_keys(
            allocation,
            {
                "allocator_revision",
                "code_counts_q4_q6_q8",
                "code_map_sha256",
                "marginal_steps",
                "packed_precision_bytes",
            },
            context=f"{context}.allocation",
        )
        if allocation["allocator_revision"] != STATIC_Q468_ALLOCATOR_REVISION:
            raise CalibrationArtifactError(
                f"{context}.allocation allocator revision drifted"
            )
        marginal_steps = _artifact_int(
            allocation["marginal_steps"],
            context=f"{context}.allocation.marginal_steps",
        )
        if marginal_steps != FROZEN_STATIC_Q468_PRIMARY_STEPS:
            raise CalibrationArtifactError(f"{context} allocation must spend exact K29334")
        if (
            _artifact_int(
                allocation["packed_precision_bytes"],
                context=f"{context}.allocation.packed_precision_bytes",
            )
            != math.ceil(geometry.total_rows * 2 / 8)
        ):
            raise CalibrationArtifactError(f"{context} packed precision byte count drifted")
        raw_counts = _sequence(
            allocation["code_counts_q4_q6_q8"],
            context=f"{context}.allocation.code_counts_q4_q6_q8",
        )
        if len(raw_counts) != 3:
            raise CalibrationArtifactError(f"{context} allocation needs Q4/Q6/Q8 counts")
        counts = [
            _artifact_int(value, context=f"{context}.allocation.code_counts[{position}]")
            for position, value in enumerate(raw_counts)
        ]
        codes = _allocate_frozen_comparator_codes(aggregate)
        computed_counts = [int(np.count_nonzero(codes == code)) for code in range(3)]
        if counts != computed_counts or sum(counts) != geometry.total_rows:
            raise CalibrationArtifactError(f"{context} code counts differ from exact allocation")
        code_map_hash = _artifact_sha(
            allocation["code_map_sha256"],
            context=f"{context}.allocation.code_map_sha256",
        )
        computed_code_hash = static_q468_code_map_sha256(
            codes, geometry=geometry, marginal_steps=marginal_steps
        )
        if code_map_hash != computed_code_hash:
            raise CalibrationArtifactError(f"{context} code-map SHA-256 drifted")
        decoded_selectors[expected_method] = ComparatorSelectorArtifact(
            method_id=expected_method,
            aggregate=aggregate,
            position_manifest_sha256=position_hash,
            calibration_scores_sha256=aggregate_score_hash,
            marginal_steps=marginal_steps,
            precision_codes=_immutable_array_copy(codes),
            code_map_sha256=computed_code_hash,
        )
    if set(decoded_selectors) != set(FROZEN_COMPARATOR_PROFILE_ORDER):
        raise CalibrationArtifactError("comparator artifact is missing a frozen selector")
    return ComparatorScoreArtifact(
        selectors=MappingProxyType(decoded_selectors),
        calibration_identity_sha256=calibration_identity_hash,
        canonical_evidence_sha256=computed_canonical_hash,
        file_sha256=file_sha256,
    )


# Split-half stability contract --------------------------------------------


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


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    if len(left) < 2:
        raise ValueError("at least two paired values are required")
    left_ranks = np.asarray(_average_ranks(left), dtype=np.float64)
    right_ranks = np.asarray(_average_ranks(right), dtype=np.float64)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant input")
    return float(np.dot(left_centered, right_centered) / denominator)


def q8_set_jaccard(left_codes: np.ndarray, right_codes: np.ndarray) -> float:
    left = _precision_codes(left_codes, name="left_codes")
    right = _precision_codes(
        right_codes, name="right_codes", expected_rows=left.size
    )
    left_q8 = left == 2
    right_q8 = right == 2
    union = int(np.count_nonzero(np.logical_or(left_q8, right_q8)))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(np.logical_and(left_q8, right_q8)))
    return intersection / union


def per_layer_mean_bitwidth_shifts(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
    *,
    layer_indices: tuple[int, ...],
    rows_per_layer: int,
) -> tuple[tuple[int, float], ...]:
    if not isinstance(layer_indices, tuple) or not layer_indices:
        raise ValueError("layer_indices must be a non-empty tuple")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in layer_indices
    ) or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices must contain unique non-negative integers")
    layer_rows = _strict_positive_int(rows_per_layer, name="rows_per_layer")
    total_rows = len(layer_indices) * layer_rows
    left = _precision_codes(left_codes, name="left_codes", expected_rows=total_rows)
    right = _precision_codes(right_codes, name="right_codes", expected_rows=total_rows)
    left_layers = left.astype(np.float64).reshape(len(layer_indices), layer_rows)
    right_layers = right.astype(np.float64).reshape(len(layer_indices), layer_rows)
    shifts = 2.0 * np.abs(left_layers.mean(axis=1) - right_layers.mean(axis=1))
    return tuple(
        (layer_index, float(shift))
        for layer_index, shift in zip(layer_indices, shifts, strict=True)
    )


@dataclass(frozen=True, slots=True)
class PolicyStabilityResult:
    passed: bool
    spearman_average_ties: float | None
    q8_jaccard: float
    layer_mean_bitwidth_shifts: tuple[tuple[int, float], ...]
    checks: tuple[tuple[str, bool], ...]

    @property
    def max_layer_mean_bitwidth_shift(self) -> float:
        return max(shift for _layer, shift in self.layer_mean_bitwidth_shifts)


def evaluate_policy_stability(
    half_a_codes: np.ndarray,
    half_b_codes: np.ndarray,
    *,
    layer_indices: tuple[int, ...],
    rows_per_layer: int,
    expected_marginal_steps: int | None = None,
) -> PolicyStabilityResult:
    rows = len(layer_indices) * _strict_positive_int(
        rows_per_layer, name="rows_per_layer"
    )
    left = _precision_codes(half_a_codes, name="half_a_codes", expected_rows=rows)
    right = _precision_codes(half_b_codes, name="half_b_codes", expected_rows=rows)
    if expected_marginal_steps is not None:
        steps = _strict_nonnegative_int(
            expected_marginal_steps, name="expected_marginal_steps"
        )
        if steps > 2 * rows:
            raise ValueError("expected_marginal_steps exceeds two steps per row")
        if int(left.astype(np.int64).sum()) != steps:
            raise ValueError("half A code map does not satisfy the expected exact-K budget")
        if int(right.astype(np.int64).sum()) != steps:
            raise ValueError("half B code map does not satisfy the expected exact-K budget")
    try:
        spearman = _spearman_correlation(
            left.astype(np.float64).tolist(), right.astype(np.float64).tolist()
        )
    except ValueError as exc:
        if "constant input" not in str(exc):
            raise
        spearman = None
    jaccard = q8_set_jaccard(left, right)
    layer_shifts = per_layer_mean_bitwidth_shifts(
        left,
        right,
        layer_indices=layer_indices,
        rows_per_layer=rows_per_layer,
    )
    max_shift = max(shift for _layer, shift in layer_shifts)
    checks = (
        (
            "spearman_at_least_0_70",
            spearman is not None and spearman >= MIN_SPLIT_HALF_SPEARMAN,
        ),
        ("q8_jaccard_at_least_0_50", jaccard >= MIN_SPLIT_HALF_Q8_JACCARD),
        (
            "every_layer_mean_bitwidth_shift_at_most_0_25",
            max_shift <= MAX_LAYER_MEAN_BITWIDTH_SHIFT,
        ),
    )
    return PolicyStabilityResult(
        passed=all(passed for _name, passed in checks),
        spearman_average_ties=spearman,
        q8_jaccard=jaccard,
        layer_mean_bitwidth_shifts=layer_shifts,
        checks=checks,
    )


def _stability_threshold_record() -> dict[str, str]:
    return {
        "maximum_layer_mean_bitwidth_shift": MAX_LAYER_MEAN_BITWIDTH_SHIFT.hex(),
        "minimum_q8_jaccard": MIN_SPLIT_HALF_Q8_JACCARD.hex(),
        "minimum_spearman_average_ties": MIN_SPLIT_HALF_SPEARMAN.hex(),
    }


def _stability_metric_record(result: PolicyStabilityResult) -> dict[str, object]:
    spearman = (
        None
        if result.spearman_average_ties is None
        else result.spearman_average_ties.hex()
    )
    return {
        "checks": [{"name": name, "passed": passed} for name, passed in result.checks],
        "layer_mean_bitwidth_shifts": [
            {"layer_index": layer, "shift": shift.hex()}
            for layer, shift in result.layer_mean_bitwidth_shifts
        ],
        "maximum_layer_mean_bitwidth_shift": result.max_layer_mean_bitwidth_shift.hex(),
        "passed": result.passed,
        "q8_jaccard": result.q8_jaccard.hex(),
        "spearman_average_ties": spearman,
    }


@dataclass(frozen=True, slots=True)
class DecodedSplitHalfStabilityArtifact:
    identity_file_sha256: str
    canonical_identity_sha256: str
    resolver_assignment_sha256: str
    full_sequence_score_manifest_sha256: str
    full_calibration_scores_sha256: str
    half_a_aggregate: CalibrationAggregate
    half_b_aggregate: CalibrationAggregate
    half_a_codes: np.ndarray
    half_b_codes: np.ndarray
    stability: PolicyStabilityResult
    canonical_evidence_sha256: str
    file_sha256: str


def _decode_frozen_split_half(
    value: object,
    *,
    expected_half: SplitHalf,
) -> tuple[CalibrationAggregate, np.ndarray]:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    half = _mapping(value, context=f"evidence.halves[{expected_half}]")
    _exact_keys(
        half,
        {
            "calibration_scores_sha256",
            "code_map",
            "family_sequence_counts",
            "half",
            "identity_record_manifest_sha256",
            "ruler_category_sequence_counts",
            "scores",
            "sequence_score_manifest_sha256",
            "source_tensor_contract",
        },
        context=f"evidence.halves[{expected_half}]",
    )
    if half["half"] != expected_half:
        raise CalibrationArtifactError("split-half labels must be canonical A then B")
    family_counts = _parse_counts(
        half["family_sequence_counts"],
        context=f"evidence.halves[{expected_half}].family_sequence_counts",
        key_name="family",
        expected_names=CALIBRATION_FAMILY_ORDER,
    )
    if family_counts != (("mbpp", 64), ("pg19", 8), ("ruler", 8)):
        raise CalibrationArtifactError(
            "each split half must contain MBPP=64, PG19=8, RULER=8"
        )
    ruler_counts = _parse_counts(
        half["ruler_category_sequence_counts"],
        context=f"evidence.halves[{expected_half}].ruler_category_sequence_counts",
        key_name="category",
        expected_names=RULER_CATEGORY_ORDER,
    )
    if ruler_counts != tuple((category, 2) for category in RULER_CATEGORY_ORDER):
        raise CalibrationArtifactError(
            "each split half must contain two sequences per RULER category"
        )
    source_contract = _parse_source_tensor_contract(half["source_tensor_contract"])
    if source_contract != FROZEN_SOURCE_TENSOR_CONTRACT:
        raise CalibrationArtifactError("split-half source tensor contract drifted")
    sequence_manifest_sha256 = _artifact_sha(
        half["sequence_score_manifest_sha256"],
        context=f"evidence.halves[{expected_half}].sequence_score_manifest_sha256",
    )
    identity_manifest_sha256 = _artifact_sha(
        half["identity_record_manifest_sha256"],
        context=f"evidence.halves[{expected_half}].identity_record_manifest_sha256",
    )
    score_values = _decode_score_triplet(
        half["scores"],
        geometry=geometry,
        context=f"evidence.halves[{expected_half}].scores",
    )
    aggregate = _validate_aggregate(
        CalibrationAggregate(
            d4=score_values[0],
            d6=score_values[1],
            d8=score_values[2],
            family_sequence_counts=family_counts,
            ruler_category_sequence_counts=ruler_counts,
            sequence_score_manifest_sha256=sequence_manifest_sha256,
            source_contract=source_contract,
            identity_record_manifest_sha256=identity_manifest_sha256,
        ),
        expected_rows=geometry.total_rows,
    )
    recorded_scores_sha256 = _artifact_sha(
        half["calibration_scores_sha256"],
        context=f"evidence.halves[{expected_half}].calibration_scores_sha256",
    )
    computed_scores_sha256 = static_q468_distortion_sha256(
        *aggregate.scores(), geometry=geometry
    )
    if recorded_scores_sha256 != computed_scores_sha256:
        raise CalibrationArtifactError("split-half calibration score SHA-256 drifted")
    expected_codes = allocate_static_q468_code_map(
        aggregate, marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS
    )
    code_record = _mapping(
        half["code_map"], context=f"evidence.halves[{expected_half}].code_map"
    )
    _exact_keys(
        code_record,
        {"code_map_sha256", "codes_base64", "dtype", "shape"},
        context=f"evidence.halves[{expected_half}].code_map",
    )
    if code_record["dtype"] != "uint8" or code_record["shape"] != [geometry.total_rows]:
        raise CalibrationArtifactError("split-half code-map layout drifted")
    encoded_codes = code_record["codes_base64"]
    if not isinstance(encoded_codes, str):
        raise CalibrationArtifactError("split-half code map must be base64 text")
    try:
        raw_codes = base64.b64decode(encoded_codes, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CalibrationArtifactError("split-half codes_base64 is invalid") from exc
    if base64.b64encode(raw_codes).decode("ascii") != encoded_codes:
        raise CalibrationArtifactError("split-half codes_base64 is not canonical")
    if len(raw_codes) != geometry.total_rows:
        raise CalibrationArtifactError("split-half code-map byte length drifted")
    recorded_codes = np.frombuffer(raw_codes, dtype="u1").copy()
    if not np.array_equal(recorded_codes, expected_codes):
        raise CalibrationArtifactError("split-half code map differs from exact allocation")
    recorded_code_sha256 = _artifact_sha(
        code_record["code_map_sha256"],
        context=f"evidence.halves[{expected_half}].code_map_sha256",
    )
    computed_code_sha256 = static_q468_code_map_sha256(
        expected_codes,
        geometry=geometry,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    )
    if recorded_code_sha256 != computed_code_sha256:
        raise CalibrationArtifactError("split-half code-map SHA-256 drifted")
    return aggregate, _immutable_array_copy(expected_codes)


def deserialize_frozen_split_half_stability_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
    expected_identity_file_sha256: str | None = None,
    expected_canonical_identity_sha256: str | None = None,
    expected_resolver_assignment_sha256: str | None = None,
) -> DecodedSplitHalfStabilityArtifact:
    if not isinstance(data, bytes):
        raise TypeError("split-half stability artifact must be bytes")
    file_sha256 = hashlib.sha256(data).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != _artifact_sha(
        expected_file_sha256, context="expected_file_sha256"
    ):
        raise CalibrationArtifactError("split-half artifact file SHA-256 mismatch")
    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, CalibrationArtifactError):
            raise
        raise CalibrationArtifactError(
            f"split-half artifact is not strict UTF-8 JSON: {exc}"
        ) from exc
    root = _mapping(document, context="split-half artifact")
    _exact_keys(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="split-half artifact",
    )
    if root["artifact_kind"] != SPLIT_HALF_STABILITY_ARTIFACT_KIND:
        raise CalibrationArtifactError("split-half artifact kind drifted")
    if root["schema_version"] != SPLIT_HALF_STABILITY_ARTIFACT_SCHEMA_VERSION:
        raise CalibrationArtifactError("split-half schema version drifted")
    evidence = _mapping(root["evidence"], context="split-half evidence")
    _exact_keys(
        evidence,
        {
            "artifact_profile",
            "artifact_revision",
            "full_calibration",
            "geometry",
            "geometry_sha256",
            "halves",
            "identity",
            "marginal_steps",
            "metrics",
            "thresholds",
        },
        context="split-half evidence",
    )
    if (
        evidence["artifact_profile"] != SPLIT_HALF_STABILITY_ARTIFACT_PROFILE
        or evidence["artifact_revision"] != SPLIT_HALF_STABILITY_ARTIFACT_REVISION
    ):
        raise CalibrationArtifactError("split-half artifact profile or revision drifted")
    recorded_canonical = _artifact_sha(
        root["canonical_evidence_sha256"],
        context="split-half canonical evidence SHA-256",
    )
    computed_canonical = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    if recorded_canonical != computed_canonical:
        raise CalibrationArtifactError("split-half canonical evidence SHA-256 mismatch")
    if canonical_json_bytes(root) != data:
        raise CalibrationArtifactError("split-half artifact bytes are not canonical JSON")
    geometry = _parse_geometry(evidence["geometry"])
    if geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
        raise CalibrationArtifactError("split-half geometry differs from the frozen geometry")
    if evidence["geometry_sha256"] != geometry.geometry_sha256:
        raise CalibrationArtifactError("split-half geometry SHA-256 drifted")
    if evidence["marginal_steps"] != FROZEN_STATIC_Q468_PRIMARY_STEPS:
        raise CalibrationArtifactError("split-half policy must be exact K29334")
    if evidence["thresholds"] != _stability_threshold_record():
        raise CalibrationArtifactError("split-half stability thresholds drifted")

    identity = _mapping(evidence["identity"], context="split-half identity")
    _exact_keys(
        identity,
        {
            "canonical_identity_sha256",
            "identity_file_sha256",
            "resolver_assignment_sha256",
        },
        context="split-half identity",
    )
    identity_file_sha256 = _artifact_sha(
        identity["identity_file_sha256"], context="split identity file SHA-256"
    )
    canonical_identity_sha256 = _artifact_sha(
        identity["canonical_identity_sha256"],
        context="split canonical identity SHA-256",
    )
    resolver_assignment_sha256 = _artifact_sha(
        identity["resolver_assignment_sha256"],
        context="split resolver assignment SHA-256",
    )
    for expected_value, recorded_value, context in (
        (
            expected_identity_file_sha256,
            identity_file_sha256,
            "split identity file SHA-256",
        ),
        (
            expected_canonical_identity_sha256,
            canonical_identity_sha256,
            "split canonical identity SHA-256",
        ),
        (
            expected_resolver_assignment_sha256,
            resolver_assignment_sha256,
            "split resolver assignment SHA-256",
        ),
    ):
        if expected_value is not None and recorded_value != _artifact_sha(
            expected_value, context=f"expected {context}"
        ):
            raise CalibrationArtifactError(f"{context} differs from expected identity")
    full_calibration = _mapping(
        evidence["full_calibration"], context="split full calibration"
    )
    _exact_keys(
        full_calibration,
        {"calibration_scores_sha256", "sequence_score_manifest_sha256"},
        context="split full calibration",
    )
    full_scores_sha256 = _artifact_sha(
        full_calibration["calibration_scores_sha256"],
        context="split full calibration scores SHA-256",
    )
    full_manifest_sha256 = _artifact_sha(
        full_calibration["sequence_score_manifest_sha256"],
        context="split full sequence manifest SHA-256",
    )
    halves = _sequence(evidence["halves"], context="split halves")
    if len(halves) != 2:
        raise CalibrationArtifactError("split artifact must contain exactly halves A and B")
    aggregate_a, codes_a = _decode_frozen_split_half(halves[0], expected_half="a")
    aggregate_b, codes_b = _decode_frozen_split_half(halves[1], expected_half="b")
    stability = evaluate_policy_stability(
        codes_a,
        codes_b,
        layer_indices=geometry.layer_indices,
        rows_per_layer=geometry.rows_per_layer,
        expected_marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    )
    if not stability.passed:
        raise CalibrationArtifactError("recomputed split-half stability gate did not pass")
    if evidence["metrics"] != _stability_metric_record(stability):
        raise CalibrationArtifactError("split-half metrics or checks drifted")
    return DecodedSplitHalfStabilityArtifact(
        identity_file_sha256=identity_file_sha256,
        canonical_identity_sha256=canonical_identity_sha256,
        resolver_assignment_sha256=resolver_assignment_sha256,
        full_sequence_score_manifest_sha256=full_manifest_sha256,
        full_calibration_scores_sha256=full_scores_sha256,
        half_a_aggregate=aggregate_a,
        half_b_aggregate=aggregate_b,
        half_a_codes=codes_a,
        half_b_codes=codes_b,
        stability=stability,
        canonical_evidence_sha256=computed_canonical,
        file_sha256=file_sha256,
    )


__all__ = [
    "CALIBRATION_SCORE_ARTIFACT_KIND",
    "CalibrationAggregate",
    "CalibrationArtifactError",
    "CalibrationSourceTensorContract",
    "ComparatorAggregate",
    "ComparatorScoreArtifact",
    "ComparatorSelectorArtifact",
    "DecodedCalibrationScoreArtifact",
    "DecodedSplitHalfStabilityArtifact",
    "FROZEN_COMPARATOR_PROFILE_ORDER",
    "FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE",
    "FROZEN_QWEN35_STATIC_Q468_GEOMETRY",
    "FROZEN_STATELEASE_RESIDENT_BYTES",
    "FROZEN_STATIC_Q468_ABLATION_STEPS",
    "FROZEN_STATIC_Q468_PRIMARY_STEPS",
    "FROZEN_STATIC_Q48_PROMOTIONS",
    "FROZEN_TRANSFORMERS_VERSION",
    "FROZEN_UNWEIGHTED_MSE_PROFILE",
    "PRIMARY_MODEL_ID",
    "PRIMARY_MODEL_REVISION",
    "PRIMARY_TOKENIZER_ID",
    "PRIMARY_TOKENIZER_REVISION",
    "PolicyStabilityResult",
    "STATIC_Q468_ABLATION_METHOD",
    "STATIC_Q468_ALLOCATOR_REVISION",
    "STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD",
    "STATIC_Q468_MSE_METHOD",
    "STATIC_Q468_PRIMARY_METHOD",
    "STATIC_Q48_COMPARATOR_METHOD",
    "StaticRhtByteLedger",
    "StaticRhtQ468Geometry",
    "StaticRhtQ468Policy",
    "StaticRhtQ48Policy",
    "allocate_exact_multibit_codes",
    "allocate_exact_multibit_codes_fast",
    "allocate_exact_q48_mask",
    "allocate_static_q468_code_map",
    "build_static_rht_q468_policy",
    "build_static_rht_q48_policy",
    "calibration_identity_record_manifest_sha256",
    "deserialize_calibration_score_artifact",
    "deserialize_comparator_score_artifact",
    "deserialize_frozen_split_half_stability_artifact",
    "deserialize_static_rht_q468_policy",
    "deserialize_static_rht_q48_policy",
    "evaluate_policy_stability",
    "pack_precision_codes",
    "pack_precision_mask",
    "per_layer_mean_bitwidth_shifts",
    "q8_set_jaccard",
    "serialize_static_rht_q468_policy",
    "serialize_static_rht_q48_policy",
    "static_q468_byte_ledger",
    "static_q468_code_map_sha256",
    "static_q468_distortion_sha256",
    "static_q48_byte_ledger",
    "static_q48_distortion_sha256",
    "unpack_precision_codes",
    "unpack_precision_mask",
]
