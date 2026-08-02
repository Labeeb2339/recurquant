"""Static, packed RHT Q4/Q6/Q8 policies for Experiment 013.

The policy artifact freezes one precision code for every recurrent-state row.
Codes are selected once from calibration distortions and are not recomputed on
the inference path.  Packed states own three integer payload pools, one FP16
scale per row, a two-bit code stream, and a uint16 offset into the selected
pool.  No FP32 state mirror or runtime score tensor is resident.

This is a correctness-first reference format.  The explicit offsets are the
addressing contract for a packed-native kernel; this module does not claim that
the Python materialization path is an optimized runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from .mixed_quantization import (
    PackedMixedQuantizedTensor,
    _pack_precision_mask,
    _unpack_precision_mask,
    quantize_pack_mixed,
)
from .multibit_policy import allocate_exact_multibit_codes_fast
from .multibit_quantization import (
    INT4_PRECISION_CODE,
    INT6_PRECISION_CODE,
    INT8_PRECISION_CODE,
    PackedMultiBitQuantizedTensor,
    _pack_precision_codes,
    _unpack_precision_codes,
    quantize_pack_multibit,
)
from .quantization import QuantizationSpec
from .rht import RHT_SEED, right_rht_decode, right_rht_encode

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
STATIC_Q48_COMPARATOR_METHOD = "rht_q48_static_p14739"

PRIMARY_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
PRIMARY_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PRIMARY_TOKENIZER_ID = PRIMARY_MODEL_ID
PRIMARY_TOKENIZER_REVISION = PRIMARY_MODEL_REVISION
FROZEN_TRANSFORMERS_VERSION = "5.14.1"

FROZEN_STATIC_Q468_PRIMARY_STEPS = 29_334
FROZEN_STATIC_Q468_ABLATION_STEPS = 27_030
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

StaticCodec: TypeAlias = Literal["q468", "q48"]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,127}")
_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]*)?")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _validate_owned_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.storage_offset() != 0:
        raise ValueError(f"{name} must have zero storage offset")
    logical_bytes = tensor.numel() * tensor.element_size()
    if _storage_bytes(tensor) != logical_bytes:
        raise ValueError(
            f"{name} owns {_storage_bytes(tensor)} bytes but exposes {logical_bytes} bytes"
        )


@dataclass(frozen=True, slots=True)
class StaticRhtQ468Geometry:
    """Complete row geometry and resident-byte target for one static policy."""

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
        return _sha256_bytes(_canonical_json(self.canonical_dict()))


FROZEN_QWEN35_STATIC_Q468_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=FROZEN_RECURRENT_LAYER_INDICES,
    heads=16,
    key_rows=128,
    value_width=128,
    target_resident_bytes=FROZEN_STATELEASE_RESIDENT_BYTES,
)


@dataclass(frozen=True, slots=True)
class StaticRhtByteLedger:
    """Exact resident tensor arithmetic for one static packed format."""

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
    # The frozen exact-budget layouts reserve the remaining eight bytes.  An
    # under-budget ablation keeps its natural size instead of adding filler.
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
    """Return physical Q4/Q6/Q8 bytes for a frozen static code budget."""

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    steps = _validate_integer(marginal_steps, name="marginal_steps")
    if steps > 2 * geometry.total_rows:
        raise ValueError("marginal_steps exceeds two steps per state row")
    selected_method = method_id or f"rht_q468_static_k{steps}"
    if _METHOD_RE.fullmatch(selected_method) is None:
        raise ValueError("method_id must use lowercase identifier characters")

    base_q4 = geometry.state_elements * 4 // 8
    marginal_payload = steps * geometry.value_width * 2 // 8
    return _finish_ledger(
        method_id=selected_method,
        codec="q468",
        selected_units=steps,
        payload_bytes=base_q4 + marginal_payload,
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
    """Return physical Q4/Q8 comparator bytes with a one-bit precision map."""

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    promotions = _validate_integer(promoted_rows, name="promoted_rows")
    if promotions > geometry.total_rows:
        raise ValueError("promoted_rows exceeds the number of state rows")
    selected_method = method_id or f"rht_q48_static_p{promotions}"
    if _METHOD_RE.fullmatch(selected_method) is None:
        raise ValueError("method_id must use lowercase identifier characters")

    base_q4 = geometry.state_elements * 4 // 8
    promoted_payload = promotions * geometry.value_width * 4 // 8
    return _finish_ledger(
        method_id=selected_method,
        codec="q48",
        selected_units=promotions,
        payload_bytes=base_q4 + promoted_payload,
        scale_bytes=geometry.total_rows * 2,
        precision_code_bytes=math.ceil(geometry.total_rows / 8),
        pool_offset_bytes=geometry.total_rows * 2,
        target_resident_bytes=geometry.target_resident_bytes,
    )


def frozen_static_byte_accounting() -> dict[str, dict[str, object]]:
    """Return the three frozen Experiment 013 byte ledgers."""

    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    return {
        STATIC_Q468_PRIMARY_METHOD: static_q468_byte_ledger(
            geometry,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
            method_id=STATIC_Q468_PRIMARY_METHOD,
        ).evidence_dict(),
        STATIC_Q468_ABLATION_METHOD: static_q468_byte_ledger(
            geometry,
            FROZEN_STATIC_Q468_ABLATION_STEPS,
            method_id=STATIC_Q468_ABLATION_METHOD,
        ).evidence_dict(),
        STATIC_Q48_COMPARATOR_METHOD: static_q48_byte_ledger(
            geometry,
            FROZEN_STATIC_Q48_PROMOTIONS,
            method_id=STATIC_Q48_COMPARATOR_METHOD,
        ).evidence_dict(),
    }


def _expected_pool_offsets(codes: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    flat = codes.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    offsets = torch.empty(flat.numel(), dtype=torch.int64)
    counts: list[int] = []
    for code in (INT4_PRECISION_CODE, INT6_PRECISION_CODE, INT8_PRECISION_CODE):
        mask = flat == code
        count = int(mask.sum().item())
        counts.append(count)
        if count:
            offsets[mask] = torch.arange(count, dtype=torch.int64)
    return offsets.to(torch.uint16).contiguous(), (counts[0], counts[1], counts[2])


def _tensor_b64(tensor: torch.Tensor, *, dtype: np.dtype[Any]) -> str:
    array = tensor.detach().to("cpu").contiguous().numpy().astype(dtype, copy=False)
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _decode_b64(value: object, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"{name} is not canonical base64") from error


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    """Atomically publish ``payload`` while refusing to replace any path entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite policy artifact: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class StaticRhtQ468Policy:
    """Canonical static precision map and packed-native addressing metadata."""

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
    packed_precision_codes: torch.Tensor
    pool_offsets: torch.Tensor
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
        _validate_sha256(
            self.tokenizer_manifest_sha256,
            name="tokenizer_manifest_sha256",
        )
        _validate_transformers_version(self.transformers_version)
        _validate_sha256(
            self.identity_artifact_sha256,
            name="identity_artifact_sha256",
        )
        _validate_git_revision(self.source_commit, name="source_commit")
        if not isinstance(self.geometry, StaticRhtQ468Geometry):
            raise TypeError("geometry must be a StaticRhtQ468Geometry")
        _validate_sha256(
            self.calibration_manifest_sha256,
            name="calibration_manifest_sha256",
        )
        _validate_sha256(
            self.calibration_scores_sha256,
            name="calibration_scores_sha256",
        )
        steps = _validate_integer(self.marginal_steps, name="marginal_steps")
        if steps > 2 * self.geometry.total_rows:
            raise ValueError("marginal_steps exceeds two steps per state row")
        frozen_steps = {
            STATIC_Q468_PRIMARY_METHOD: FROZEN_STATIC_Q468_PRIMARY_STEPS,
            STATIC_Q468_ABLATION_METHOD: FROZEN_STATIC_Q468_ABLATION_STEPS,
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
        if self.rht_seed != RHT_SEED:
            raise ValueError(f"rht_seed must equal the codec seed {RHT_SEED}")

        _validate_owned_tensor(self.packed_precision_codes, name="packed_precision_codes")
        _validate_owned_tensor(self.pool_offsets, name="pool_offsets")
        if self.packed_precision_codes.device != self.pool_offsets.device:
            raise ValueError("precision codes and pool offsets must share one device")
        if (
            self.packed_precision_codes.dtype != torch.uint8
            or self.packed_precision_codes.ndim != 1
        ):
            raise TypeError("packed_precision_codes must be one-dimensional torch.uint8")
        if self.pool_offsets.dtype != torch.uint16 or self.pool_offsets.ndim != 1:
            raise TypeError("pool_offsets must be one-dimensional torch.uint16")
        if self.pool_offsets.numel() != self.geometry.total_rows:
            raise ValueError("pool_offsets must contain one uint16 value per state row")

        codes = _unpack_precision_codes(
            self.packed_precision_codes,
            self.geometry.total_rows,
        ).reshape(-1)
        if int(codes.to(torch.int64).sum().item()) != steps:
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
        if not torch.equal(self.pool_offsets.to("cpu"), expected_offsets):
            raise ValueError("pool_offsets are not canonical per-pool prefix offsets")

    def precision_codes(self) -> torch.Tensor:
        return _unpack_precision_codes(
            self.packed_precision_codes,
            self.geometry.total_rows,
        ).reshape(self.geometry.layers, self.geometry.heads, self.geometry.key_rows)

    def clone_to(self, device: torch.device | str) -> StaticRhtQ468Policy:
        """Copy the physical policy tensors to ``device`` without aliasing input storage."""

        return StaticRhtQ468Policy(
            method_id=self.method_id,
            policy_revision=self.policy_revision,
            allocator_revision=self.allocator_revision,
            codec_revision=self.codec_revision,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_manifest_sha256=self.tokenizer_manifest_sha256,
            transformers_version=self.transformers_version,
            identity_artifact_sha256=self.identity_artifact_sha256,
            source_commit=self.source_commit,
            geometry=self.geometry,
            calibration_manifest_sha256=self.calibration_manifest_sha256,
            calibration_scores_sha256=self.calibration_scores_sha256,
            marginal_steps=self.marginal_steps,
            rht_seed=self.rht_seed,
            packed_precision_codes=self.packed_precision_codes.detach().to(device).clone(),
            pool_offsets=self.pool_offsets.detach().to(device).clone(),
            pool_counts=self.pool_counts,
        )

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
            "packed_precision_codes_b64": _tensor_b64(
                self.packed_precision_codes,
                dtype=np.dtype("u1"),
            ),
            "policy_revision": self.policy_revision,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_le_b64": _tensor_b64(
                self.pool_offsets,
                dtype=np.dtype("<u2"),
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
        digest.update(
            self.packed_precision_codes.detach().to("cpu").contiguous().numpy().tobytes()
        )
        return digest.hexdigest()

    @property
    def pool_offsets_sha256(self) -> str:
        offsets = (
            self.pool_offsets.detach()
            .to("cpu")
            .contiguous()
            .numpy()
            .astype("<u2", copy=False)
        )
        return _sha256_bytes(offsets.tobytes(order="C"))

    @property
    def policy_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._content_dict()))

    def evidence_dict(self) -> dict[str, object]:
        ledger = static_q468_byte_ledger(
            self.geometry,
            self.marginal_steps,
            method_id=self.method_id,
        )
        return {
            "allocator_revision": self.allocator_revision,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_scores_sha256": self.calibration_scores_sha256,
            "code_map_sha256": self.code_map_sha256,
            "codec_revision": self.codec_revision,
            "geometry_sha256": self.geometry.geometry_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "ledger": ledger.evidence_dict(),
            "marginal_steps": self.marginal_steps,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "policy_revision": self.policy_revision,
            "policy_sha256": self.policy_sha256,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_sha256": self.pool_offsets_sha256,
            "rht_seed": self.rht_seed,
            "schema": STATIC_Q468_POLICY_SCHEMA,
            "source_commit": self.source_commit,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "transformers_version": self.transformers_version,
        }


def _normalize_static_distortions(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized: list[torch.Tensor] = []
    expected_shape = tuple(d4.shape) if isinstance(d4, torch.Tensor) else None
    for name, tensor in (("D4", d4), ("D6", d6), ("D8", d8)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.numel() != geometry.total_rows:
            raise ValueError(f"{name} must contain exactly {geometry.total_rows} row distortions")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError("D4, D6, and D8 must have identical shapes")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must be materialized")
        if not torch.isfinite(tensor).all().item() or (tensor < 0).any().item():
            raise ValueError(f"{name} must contain finite, non-negative values")
        normalized.append(tensor.detach().reshape(1, geometry.total_rows))
    return normalized[0], normalized[1], normalized[2]


def static_q468_distortion_sha256(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry,
) -> str:
    """Hash canonical CPU-FP64 calibration distortions and their geometry."""

    values = _normalize_static_distortions(d4, d6, d8, geometry=geometry)
    digest = hashlib.sha256()
    digest.update(b"recurquant.static-q468-distortions.v1\0")
    digest.update(_canonical_json(geometry.canonical_dict()))
    for label, tensor in zip((b"D4\0", b"D6\0", b"D8\0"), values, strict=True):
        digest.update(label)
        array = tensor.to(device="cpu", dtype=torch.float64).contiguous().numpy()
        digest.update(array.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def build_static_rht_q468_policy(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
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
    """Build the deterministic exact-allocation policy from calibration losses."""

    normalized = _normalize_static_distortions(d4, d6, d8, geometry=geometry)
    scores_sha256 = static_q468_distortion_sha256(*normalized, geometry=geometry)
    if calibration_scores_sha256 is not None:
        _validate_sha256(calibration_scores_sha256, name="calibration_scores_sha256")
        if calibration_scores_sha256 != scores_sha256:
            raise ValueError("calibration_scores_sha256 does not match supplied distortions")
    codes = allocate_exact_multibit_codes_fast(
        *normalized,
        marginal_steps=marginal_steps,
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
        packed_precision_codes=_pack_precision_codes(codes),
        pool_offsets=offsets,
        pool_counts=counts,
    )


def serialize_static_rht_q468_policy(policy: StaticRhtQ468Policy) -> bytes:
    """Serialize a policy as canonical, self-hashing JSON bytes."""

    if not isinstance(policy, StaticRhtQ468Policy):
        raise TypeError("policy must be a StaticRhtQ468Policy")
    content = policy._content_dict()
    envelope = {
        "content": content,
        "policy_sha256": _sha256_bytes(_canonical_json(content)),
        "schema": STATIC_Q468_POLICY_SCHEMA,
    }
    return _canonical_json(envelope) + b"\n"


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


def deserialize_static_rht_q468_policy(data: bytes) -> StaticRhtQ468Policy:
    """Load and independently verify canonical policy bytes."""

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
    if _sha256_bytes(_canonical_json(content)) != declared_digest:
        raise ValueError("policy_sha256 does not authenticate policy content")

    geometry_dict = _expect_keys(
        content["geometry"],
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
    geometry = StaticRhtQ468Geometry(
        layer_indices=tuple(layer_indices),
        heads=geometry_dict["heads"],
        key_rows=geometry_dict["key_rows"],
        value_width=geometry_dict["value_width"],
        target_resident_bytes=geometry_dict["target_resident_bytes"],
    )
    if content["geometry_sha256"] != geometry.geometry_sha256:
        raise ValueError("geometry_sha256 does not authenticate geometry")

    code_bytes = _decode_b64(
        content["packed_precision_codes_b64"],
        name="packed_precision_codes_b64",
    )
    offset_bytes = _decode_b64(
        content["pool_offsets_le_b64"],
        name="pool_offsets_le_b64",
    )
    expected_code_bytes = math.ceil(geometry.total_rows * 2 / 8)
    if len(code_bytes) != expected_code_bytes:
        raise ValueError("packed precision code byte length does not match geometry")
    if len(offset_bytes) != geometry.total_rows * 2:
        raise ValueError("pool offset byte length does not match geometry")
    packed_codes = torch.from_numpy(np.frombuffer(code_bytes, dtype="u1").copy())
    pool_offsets = torch.from_numpy(
        np.frombuffer(offset_bytes, dtype="<u2").astype(np.uint16, copy=True)
    )
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
        packed_precision_codes=packed_codes,
        pool_offsets=pool_offsets,
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


def save_static_rht_q468_policy(policy: StaticRhtQ468Policy, path: str | Path) -> None:
    """Atomically publish canonical policy bytes without replacing ``path``."""

    _atomic_publish_new(Path(path), serialize_static_rht_q468_policy(policy))


def load_static_rht_q468_policy(path: str | Path) -> StaticRhtQ468Policy:
    """Read and verify a canonical policy artifact from ``path``."""

    return deserialize_static_rht_q468_policy(Path(path).read_bytes())


def verify_static_rht_q468_policy(
    policy: StaticRhtQ468Policy,
    *,
    expected_policy_sha256: str | None = None,
) -> dict[str, object]:
    """Revalidate a policy and return its independently checkable evidence."""

    if not isinstance(policy, StaticRhtQ468Policy):
        raise TypeError("policy must be a StaticRhtQ468Policy")
    # Reconstructing exercises every fail-closed dataclass invariant again.
    verified = deserialize_static_rht_q468_policy(serialize_static_rht_q468_policy(policy))
    if expected_policy_sha256 is not None:
        _validate_sha256(expected_policy_sha256, name="expected_policy_sha256")
        if verified.policy_sha256 != expected_policy_sha256:
            raise ValueError("verified policy SHA-256 does not match the expected digest")
    return verified.evidence_dict()


def allocate_exact_q48_mask(
    d4: torch.Tensor,
    d8: torch.Tensor,
    *,
    promoted_rows: int,
) -> torch.Tensor:
    """Select the exact fixed-cardinality Q8 set with a stable row-order tie rule."""

    if not isinstance(d4, torch.Tensor) or not isinstance(d8, torch.Tensor):
        raise TypeError("D4 and D8 must be torch.Tensor values")
    if d4.numel() == 0 or tuple(d4.shape) != tuple(d8.shape):
        raise ValueError("D4 and D8 must have one identical non-empty shape")
    for name, tensor in (("D4", d4), ("D8", d8)):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must be materialized")
        if not torch.isfinite(tensor).all().item() or (tensor < 0).any().item():
            raise ValueError(f"{name} must contain finite, non-negative values")
    promotions = _validate_integer(promoted_rows, name="promoted_rows")
    if promotions > d4.numel():
        raise ValueError("promoted_rows exceeds the number of distortion rows")

    flat_d4 = d4.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    flat_d8 = d8.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    benefits = [
        Fraction.from_float(value4) - Fraction.from_float(value8)
        for value4, value8 in zip(flat_d4, flat_d8, strict=True)
    ]
    ranked = sorted(range(len(benefits)), key=lambda index: (-benefits[index], index))
    mask = torch.zeros(len(benefits), dtype=torch.bool)
    if promotions:
        mask[ranked[:promotions]] = True
    return mask.reshape(d4.shape)


def _expected_binary_pool_offsets(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    flat = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    offsets = torch.empty(flat.numel(), dtype=torch.int64)
    low_count = int((~flat).sum().item())
    high_count = int(flat.sum().item())
    if low_count:
        offsets[~flat] = torch.arange(low_count, dtype=torch.int64)
    if high_count:
        offsets[flat] = torch.arange(high_count, dtype=torch.int64)
    return offsets.to(torch.uint16).contiguous(), (low_count, high_count)


def _validate_packed_binary_mask(packed: torch.Tensor, *, total_rows: int) -> None:
    _validate_owned_tensor(packed, name="packed_precision_mask")
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise TypeError("packed_precision_mask must be one-dimensional torch.uint8")
    expected_bytes = math.ceil(total_rows / 8)
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"packed_precision_mask must contain {expected_bytes} bytes, got {packed.numel()}"
        )
    used_bits = total_rows % 8
    if used_bits and packed.numel():
        unused_mask = 0xFF ^ ((1 << used_bits) - 1)
        if int(packed[-1].item()) & unused_mask:
            raise ValueError("unused precision-mask padding bits must be zero")


def _normalize_static_q48_distortions(
    d4: torch.Tensor,
    d8: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    normalized: list[torch.Tensor] = []
    expected_shape = tuple(d4.shape) if isinstance(d4, torch.Tensor) else None
    for name, tensor in (("D4", d4), ("D8", d8)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.numel() != geometry.total_rows:
            raise ValueError(f"{name} must contain exactly {geometry.total_rows} row distortions")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError("D4 and D8 must have identical shapes")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must be materialized")
        if not torch.isfinite(tensor).all().item() or (tensor < 0).any().item():
            raise ValueError(f"{name} must contain finite, non-negative values")
        normalized.append(tensor.detach().reshape(1, geometry.total_rows))
    return normalized[0], normalized[1]


def static_q48_distortion_sha256(
    d4: torch.Tensor,
    d8: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry,
) -> str:
    """Hash canonical CPU-FP64 Q4/Q8 calibration distortions and geometry."""

    values = _normalize_static_q48_distortions(d4, d8, geometry=geometry)
    digest = hashlib.sha256()
    digest.update(b"recurquant.static-q48-distortions.v1\0")
    digest.update(_canonical_json(geometry.canonical_dict()))
    for label, tensor in zip((b"D4\0", b"D8\0"), values, strict=True):
        digest.update(label)
        array = tensor.to(device="cpu", dtype=torch.float64).contiguous().numpy()
        digest.update(array.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StaticRhtQ48Policy:
    """Canonical exact-cardinality Q4/Q8 mask with strict identity bindings."""

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
    packed_precision_mask: torch.Tensor
    pool_offsets: torch.Tensor
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
        _validate_sha256(
            self.tokenizer_manifest_sha256,
            name="tokenizer_manifest_sha256",
        )
        _validate_transformers_version(self.transformers_version)
        _validate_sha256(self.identity_artifact_sha256, name="identity_artifact_sha256")
        _validate_git_revision(self.source_commit, name="source_commit")
        if not isinstance(self.geometry, StaticRhtQ468Geometry):
            raise TypeError("geometry must be a StaticRhtQ468Geometry")
        _validate_sha256(
            self.calibration_manifest_sha256,
            name="calibration_manifest_sha256",
        )
        _validate_sha256(self.calibration_scores_sha256, name="calibration_scores_sha256")
        promotions = _validate_integer(self.promoted_rows, name="promoted_rows")
        if promotions > self.geometry.total_rows:
            raise ValueError("promoted_rows exceeds the number of state rows")
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

        _validate_packed_binary_mask(
            self.packed_precision_mask,
            total_rows=self.geometry.total_rows,
        )
        _validate_owned_tensor(self.pool_offsets, name="pool_offsets")
        if self.packed_precision_mask.device != self.pool_offsets.device:
            raise ValueError("precision mask and pool offsets must share one device")
        if self.pool_offsets.dtype != torch.uint16 or self.pool_offsets.ndim != 1:
            raise TypeError("pool_offsets must be one-dimensional torch.uint16")
        if self.pool_offsets.numel() != self.geometry.total_rows:
            raise ValueError("pool_offsets must contain one uint16 value per state row")

        mask = self.high_precision_mask().reshape(-1)
        if int(mask.sum().item()) != promotions:
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
        if not torch.equal(self.pool_offsets.to("cpu"), expected_offsets):
            raise ValueError("pool_offsets are not canonical per-pool prefix offsets")

    def high_precision_mask(self) -> torch.Tensor:
        return _unpack_precision_mask(
            self.packed_precision_mask,
            self.geometry.total_rows,
        ).reshape(self.geometry.layers, self.geometry.heads, self.geometry.key_rows)

    def clone_to(self, device: torch.device | str) -> StaticRhtQ48Policy:
        return StaticRhtQ48Policy(
            method_id=self.method_id,
            policy_revision=self.policy_revision,
            selector_revision=self.selector_revision,
            codec_revision=self.codec_revision,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_manifest_sha256=self.tokenizer_manifest_sha256,
            transformers_version=self.transformers_version,
            identity_artifact_sha256=self.identity_artifact_sha256,
            source_commit=self.source_commit,
            geometry=self.geometry,
            calibration_manifest_sha256=self.calibration_manifest_sha256,
            calibration_scores_sha256=self.calibration_scores_sha256,
            promoted_rows=self.promoted_rows,
            rht_seed=self.rht_seed,
            packed_precision_mask=self.packed_precision_mask.detach().to(device).clone(),
            pool_offsets=self.pool_offsets.detach().to(device).clone(),
            pool_counts=self.pool_counts,
        )

    @property
    def mask_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"recurquant.static-q48-mask.v1\0")
        digest.update(bytes.fromhex(self.geometry.geometry_sha256))
        digest.update(self.promoted_rows.to_bytes(8, "little", signed=False))
        digest.update(
            self.packed_precision_mask.detach().to("cpu").contiguous().numpy().tobytes()
        )
        return digest.hexdigest()

    @property
    def pool_offsets_sha256(self) -> str:
        offsets = (
            self.pool_offsets.detach()
            .to("cpu")
            .contiguous()
            .numpy()
            .astype("<u2", copy=False)
        )
        return _sha256_bytes(offsets.tobytes(order="C"))

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
            "packed_precision_mask_b64": _tensor_b64(
                self.packed_precision_mask,
                dtype=np.dtype("u1"),
            ),
            "policy_revision": self.policy_revision,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_le_b64": _tensor_b64(
                self.pool_offsets,
                dtype=np.dtype("<u2"),
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
        return _sha256_bytes(_canonical_json(self._content_dict()))

    def evidence_dict(self) -> dict[str, object]:
        ledger = static_q48_byte_ledger(
            self.geometry,
            self.promoted_rows,
            method_id=self.method_id,
        )
        return {
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_scores_sha256": self.calibration_scores_sha256,
            "codec_revision": self.codec_revision,
            "geometry_sha256": self.geometry.geometry_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "ledger": ledger.evidence_dict(),
            "mask_sha256": self.mask_sha256,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "policy_revision": self.policy_revision,
            "policy_sha256": self.policy_sha256,
            "pool_counts": list(self.pool_counts),
            "pool_offsets_sha256": self.pool_offsets_sha256,
            "promoted_rows": self.promoted_rows,
            "rht_seed": self.rht_seed,
            "schema": STATIC_Q48_POLICY_SCHEMA,
            "selector_revision": self.selector_revision,
            "source_commit": self.source_commit,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "transformers_version": self.transformers_version,
        }


def build_static_rht_q48_policy(
    d4: torch.Tensor,
    d8: torch.Tensor,
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
    """Build the deterministic exact-cardinality Q4/Q8 comparator policy."""

    normalized = _normalize_static_q48_distortions(d4, d8, geometry=geometry)
    scores_sha256 = static_q48_distortion_sha256(*normalized, geometry=geometry)
    if calibration_scores_sha256 is not None:
        _validate_sha256(calibration_scores_sha256, name="calibration_scores_sha256")
        if calibration_scores_sha256 != scores_sha256:
            raise ValueError("calibration_scores_sha256 does not match supplied distortions")
    mask = allocate_exact_q48_mask(
        *normalized,
        promoted_rows=promoted_rows,
    ).reshape(-1)
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
        packed_precision_mask=_pack_precision_mask(mask),
        pool_offsets=offsets,
        pool_counts=counts,
    )


def serialize_static_rht_q48_policy(policy: StaticRhtQ48Policy) -> bytes:
    """Serialize a Q4/Q8 policy as canonical, self-hashing JSON bytes."""

    if not isinstance(policy, StaticRhtQ48Policy):
        raise TypeError("policy must be a StaticRhtQ48Policy")
    content = policy._content_dict()
    envelope = {
        "content": content,
        "policy_sha256": _sha256_bytes(_canonical_json(content)),
        "schema": STATIC_Q48_POLICY_SCHEMA,
    }
    return _canonical_json(envelope) + b"\n"


def deserialize_static_rht_q48_policy(data: bytes) -> StaticRhtQ48Policy:
    """Load and independently verify canonical Q4/Q8 policy bytes."""

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
    if _sha256_bytes(_canonical_json(content)) != declared_digest:
        raise ValueError("policy_sha256 does not authenticate policy content")
    geometry_dict = _expect_keys(
        content["geometry"],
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
    geometry = StaticRhtQ468Geometry(
        layer_indices=tuple(layer_indices),
        heads=geometry_dict["heads"],
        key_rows=geometry_dict["key_rows"],
        value_width=geometry_dict["value_width"],
        target_resident_bytes=geometry_dict["target_resident_bytes"],
    )
    if content["geometry_sha256"] != geometry.geometry_sha256:
        raise ValueError("geometry_sha256 does not authenticate geometry")

    mask_bytes = _decode_b64(
        content["packed_precision_mask_b64"],
        name="packed_precision_mask_b64",
    )
    offset_bytes = _decode_b64(
        content["pool_offsets_le_b64"],
        name="pool_offsets_le_b64",
    )
    if len(mask_bytes) != math.ceil(geometry.total_rows / 8):
        raise ValueError("packed precision mask byte length does not match geometry")
    if len(offset_bytes) != geometry.total_rows * 2:
        raise ValueError("pool offset byte length does not match geometry")
    packed_mask = torch.from_numpy(np.frombuffer(mask_bytes, dtype="u1").copy())
    pool_offsets = torch.from_numpy(
        np.frombuffer(offset_bytes, dtype="<u2").astype(np.uint16, copy=True)
    )
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
        packed_precision_mask=packed_mask,
        pool_offsets=pool_offsets,
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


def save_static_rht_q48_policy(policy: StaticRhtQ48Policy, path: str | Path) -> None:
    """Atomically publish canonical Q4/Q8 policy bytes without replacement."""

    _atomic_publish_new(Path(path), serialize_static_rht_q48_policy(policy))


def load_static_rht_q48_policy(path: str | Path) -> StaticRhtQ48Policy:
    """Read and verify a canonical Q4/Q8 policy artifact from ``path``."""

    return deserialize_static_rht_q48_policy(Path(path).read_bytes())


def verify_static_rht_q48_policy(
    policy: StaticRhtQ48Policy,
    *,
    expected_policy_sha256: str | None = None,
) -> dict[str, object]:
    """Revalidate a Q4/Q8 policy and return checkable evidence."""

    if not isinstance(policy, StaticRhtQ48Policy):
        raise TypeError("policy must be a StaticRhtQ48Policy")
    verified = deserialize_static_rht_q48_policy(serialize_static_rht_q48_policy(policy))
    if expected_policy_sha256 is not None:
        _validate_sha256(expected_policy_sha256, name="expected_policy_sha256")
        if verified.policy_sha256 != expected_policy_sha256:
            raise ValueError("verified policy SHA-256 does not match the expected digest")
    return verified.evidence_dict()


def _q468_specs(geometry: StaticRhtQ468Geometry) -> tuple[QuantizationSpec, ...]:
    common = {
        "group_size": geometry.value_width,
        "scale_bits": 16,
        "flatten_last_dims": 1,
        "rounding": "nearest",
        "seed": RHT_SEED,
    }
    return (
        QuantizationSpec(bits=4, **common),
        QuantizationSpec(bits=6, **common),
        QuantizationSpec(bits=8, **common),
    )


def _q48_specs(
    geometry: StaticRhtQ468Geometry,
) -> tuple[QuantizationSpec, QuantizationSpec]:
    common = {
        "group_size": geometry.value_width,
        "scale_bits": 16,
        "flatten_last_dims": 1,
        "rounding": "nearest",
        "seed": RHT_SEED,
    }
    return QuantizationSpec(bits=4, **common), QuantizationSpec(bits=8, **common)


def _validate_states(
    states: Mapping[int, torch.Tensor],
    *,
    geometry: StaticRhtQ468Geometry,
) -> torch.device:
    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping from model-layer index to tensor")
    if set(states) != set(geometry.layer_indices):
        raise ValueError("states must contain exactly the policy recurrent-layer indices")
    expected_shape = (1, geometry.heads, geometry.key_rows, geometry.value_width)
    devices: set[torch.device] = set()
    for layer_index in geometry.layer_indices:
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
            raise RuntimeError("static packed recurrent states are inference-only")
        devices.add(state.device)
    if len(devices) != 1:
        raise ValueError("all state tensors must share one device")
    return next(iter(devices))


@dataclass(frozen=True, slots=True)
class StaticPackedRhtQ468State:
    """Physically packed complete recurrent state under one static policy."""

    policy: StaticRhtQ468Policy
    int4_payload: torch.Tensor
    int6_payload: torch.Tensor
    int8_payload: torch.Tensor
    scales: torch.Tensor
    padding: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.policy, StaticRhtQ468Policy):
            raise TypeError("policy must be a StaticRhtQ468Policy")
        tensors = self.persistent_tensors()
        devices: set[torch.device] = set()
        owners: dict[tuple[str, int], str] = {}
        for name, tensor in tensors:
            _validate_owned_tensor(tensor, name=name)
            if tensor.dtype in (torch.float32, torch.float64):
                raise ValueError(f"persistent tensor {name} may not use FP32 or FP64")
            devices.add(tensor.device)
            if tensor.numel():
                identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
                previous = owners.get(identity)
                if previous is not None:
                    raise ValueError(f"persistent tensor {name} aliases {previous}")
                owners[identity] = name
        if len(devices) != 1:
            raise ValueError("all persistent packed-state tensors must share one device")
        if self.padding.dtype != torch.uint8 or self.padding.ndim != 1:
            raise TypeError("padding must be a one-dimensional torch.uint8 tensor")
        if self.padding.numel() and self.padding.any().item():
            raise ValueError("reserved alignment bytes must be zero")

        ledger = self.ledger
        if self.padding.numel() != ledger.alignment_bytes:
            raise ValueError("padding length does not match the frozen byte ledger")
        q4_spec, q6_spec, q8_spec = _q468_specs(self.policy.geometry)
        packed = PackedMultiBitQuantizedTensor(
            int4_payload=self.int4_payload,
            int6_payload=self.int6_payload,
            int8_payload=self.int8_payload,
            scales=self.scales,
            packed_precision_codes=self.policy.packed_precision_codes,
            int4_spec=q4_spec,
            int6_spec=q6_spec,
            int8_spec=q8_spec,
            original_shape=(self.policy.geometry.total_rows, self.policy.geometry.value_width),
            original_dtype=torch.float32,
            flattened_size=self.policy.geometry.value_width,
            padded_size=self.policy.geometry.value_width,
            rows=self.policy.geometry.total_rows,
            groups_per_row=1,
        )
        if (
            packed.int4_groups,
            packed.int6_groups,
            packed.int8_groups,
        ) != self.policy.pool_counts:
            raise ValueError("physical payload pool counts do not match policy pool_counts")
        if self.resident_bytes != ledger.resident_bytes:
            raise ValueError("physical resident bytes do not match the static byte ledger")

    @property
    def ledger(self) -> StaticRhtByteLedger:
        return static_q468_byte_ledger(
            self.policy.geometry,
            self.policy.marginal_steps,
            method_id=self.policy.method_id,
        )

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("int4_payload", self.int4_payload),
            ("int6_payload", self.int6_payload),
            ("int8_payload", self.int8_payload),
            ("scales", self.scales),
            ("packed_precision_codes", self.policy.packed_precision_codes),
            ("pool_offsets", self.policy.pool_offsets),
            ("padding", self.padding),
        )

    @property
    def data_bytes(self) -> int:
        return sum(
            _storage_bytes(tensor)
            for name, tensor in self.persistent_tensors()
            if name != "padding"
        )

    @property
    def resident_bytes(self) -> int:
        return sum(_storage_bytes(tensor) for _, tensor in self.persistent_tensors())

    def _packed(self) -> PackedMultiBitQuantizedTensor:
        q4_spec, q6_spec, q8_spec = _q468_specs(self.policy.geometry)
        geometry = self.policy.geometry
        return PackedMultiBitQuantizedTensor(
            int4_payload=self.int4_payload,
            int6_payload=self.int6_payload,
            int8_payload=self.int8_payload,
            scales=self.scales,
            packed_precision_codes=self.policy.packed_precision_codes,
            int4_spec=q4_spec,
            int6_spec=q6_spec,
            int8_spec=q8_spec,
            original_shape=(geometry.total_rows, geometry.value_width),
            original_dtype=torch.float32,
            flattened_size=geometry.value_width,
            padded_size=geometry.value_width,
            rows=geometry.total_rows,
            groups_per_row=1,
        )

    def materialize(self) -> dict[int, torch.Tensor]:
        """Materialize decoded FP32 states without retaining an FP32 mirror."""

        geometry = self.policy.geometry
        encoded = self._packed().dequantize()
        result: dict[int, torch.Tensor] = {}
        for position, layer_index in enumerate(geometry.layer_indices):
            start = position * geometry.rows_per_layer
            stop = start + geometry.rows_per_layer
            layer = encoded[start:stop].reshape(
                1,
                geometry.heads,
                geometry.key_rows,
                geometry.value_width,
            )
            result[layer_index] = right_rht_decode(
                layer,
                layer_index=layer_index,
                expected_heads=geometry.heads,
                output_dtype=torch.float32,
            )
        return result

    def clone_to(self, device: torch.device | str) -> StaticPackedRhtQ468State:
        return StaticPackedRhtQ468State(
            policy=self.policy.clone_to(device),
            int4_payload=self.int4_payload.detach().to(device).clone(),
            int6_payload=self.int6_payload.detach().to(device).clone(),
            int8_payload=self.int8_payload.detach().to(device).clone(),
            scales=self.scales.detach().to(device).clone(),
            padding=self.padding.detach().to(device).clone(),
        )


def pack_static_rht_q468(
    states: Mapping[int, torch.Tensor],
    policy: StaticRhtQ468Policy,
) -> StaticPackedRhtQ468State:
    """RHT-encode and physically pack all recurrent rows under ``policy``."""

    if not isinstance(policy, StaticRhtQ468Policy):
        raise TypeError("policy must be a StaticRhtQ468Policy")
    geometry = policy.geometry
    device = _validate_states(states, geometry=geometry)
    encoded_layers = [
        right_rht_encode(
            states[layer_index],
            layer_index=layer_index,
            expected_heads=geometry.heads,
            output_dtype=torch.float32,
        ).reshape(geometry.rows_per_layer, geometry.value_width)
        for layer_index in geometry.layer_indices
    ]
    encoded = torch.cat(encoded_layers, dim=0)
    codes = policy.precision_codes().reshape(-1).to(device)
    q4_spec, q6_spec, q8_spec = _q468_specs(geometry)
    packed = quantize_pack_multibit(
        encoded,
        codes,
        int4_spec=q4_spec,
        int6_spec=q6_spec,
        int8_spec=q8_spec,
    )
    physical_policy = policy.clone_to(device)
    if not torch.equal(packed.packed_precision_codes, physical_policy.packed_precision_codes):
        raise RuntimeError("physical packer changed the canonical precision-code stream")
    ledger = static_q468_byte_ledger(
        geometry,
        policy.marginal_steps,
        method_id=policy.method_id,
    )
    return StaticPackedRhtQ468State(
        policy=physical_policy,
        int4_payload=packed.int4_payload,
        int6_payload=packed.int6_payload,
        int8_payload=packed.int8_payload,
        scales=packed.scales,
        padding=torch.zeros(ledger.alignment_bytes, dtype=torch.uint8, device=device),
    )


def verify_static_packed_rht_q468(
    state: StaticPackedRhtQ468State,
) -> dict[str, object]:
    """Revalidate a packed state and return storage-only evidence."""

    if not isinstance(state, StaticPackedRhtQ468State):
        raise TypeError("state must be a StaticPackedRhtQ468State")
    # Constructor replay validates payload shapes, offsets, counts, aliases,
    # storage ownership, and the absence of persistent FP32/FP64 tensors.
    verified = StaticPackedRhtQ468State(
        policy=state.policy,
        int4_payload=state.int4_payload,
        int6_payload=state.int6_payload,
        int8_payload=state.int8_payload,
        scales=state.scales,
        padding=state.padding,
    )
    evidence = verified.policy.evidence_dict()
    evidence["physical_data_bytes"] = verified.data_bytes
    evidence["physical_resident_bytes"] = verified.resident_bytes
    evidence["persistent_tensor_dtypes"] = {
        name: str(tensor.dtype) for name, tensor in verified.persistent_tensors()
    }
    return evidence


@dataclass(frozen=True, slots=True)
class StaticPackedRhtQ48State:
    """Physically packed complete recurrent state under a static Q4/Q8 policy."""

    policy: StaticRhtQ48Policy
    low_payload: torch.Tensor
    high_payload: torch.Tensor
    scales: torch.Tensor
    padding: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.policy, StaticRhtQ48Policy):
            raise TypeError("policy must be a StaticRhtQ48Policy")
        tensors = self.persistent_tensors()
        devices: set[torch.device] = set()
        owners: dict[tuple[str, int], str] = {}
        for name, tensor in tensors:
            _validate_owned_tensor(tensor, name=name)
            if tensor.dtype in (torch.float32, torch.float64):
                raise ValueError(f"persistent tensor {name} may not use FP32 or FP64")
            devices.add(tensor.device)
            if tensor.numel():
                identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
                previous = owners.get(identity)
                if previous is not None:
                    raise ValueError(f"persistent tensor {name} aliases {previous}")
                owners[identity] = name
        if len(devices) != 1:
            raise ValueError("all persistent packed-state tensors must share one device")

        geometry = self.policy.geometry
        low_count, high_count = self.policy.pool_counts
        expected_low_shape = (low_count, geometry.value_width * 4 // 8)
        expected_high_shape = (high_count, geometry.value_width)
        if (
            self.low_payload.dtype != torch.uint8
            or tuple(self.low_payload.shape) != expected_low_shape
        ):
            raise TypeError(
                f"low_payload must have shape {expected_low_shape} and dtype torch.uint8"
            )
        if (
            self.high_payload.dtype != torch.int8
            or tuple(self.high_payload.shape) != expected_high_shape
        ):
            raise TypeError(
                f"high_payload must have shape {expected_high_shape} and dtype torch.int8"
            )
        if self.low_payload.numel():
            low = torch.bitwise_and(self.low_payload, 0x0F)
            high = torch.bitwise_right_shift(self.low_payload, 4)
            if ((low == 8) | (high == 8)).any().item():
                raise ValueError("low_payload contains the reserved symmetric INT4 code -8")
        if self.high_payload.numel() and (self.high_payload == -128).any().item():
            raise ValueError("high_payload contains the reserved symmetric INT8 code -128")
        if self.scales.dtype != torch.float16 or tuple(self.scales.shape) != (
            geometry.total_rows,
        ):
            raise TypeError(
                f"scales must have shape {(geometry.total_rows,)} and dtype torch.float16"
            )
        if not torch.isfinite(self.scales).all().item() or (self.scales <= 0).any().item():
            raise ValueError("scales must contain finite, strictly positive values")
        if self.padding.dtype != torch.uint8 or self.padding.ndim != 1:
            raise TypeError("padding must be a one-dimensional torch.uint8 tensor")
        if self.padding.numel() and self.padding.any().item():
            raise ValueError("reserved alignment bytes must be zero")

        ledger = self.ledger
        if self.padding.numel() != ledger.alignment_bytes:
            raise ValueError("padding length does not match the frozen byte ledger")
        if self.data_bytes != ledger.data_bytes or self.resident_bytes != ledger.resident_bytes:
            raise ValueError("physical resident bytes do not match the static Q4/Q8 byte ledger")
        # Construct the shared packed representation as a final metadata and
        # dequantization compatibility check.
        self._packed()

    @property
    def ledger(self) -> StaticRhtByteLedger:
        return static_q48_byte_ledger(
            self.policy.geometry,
            self.policy.promoted_rows,
            method_id=self.policy.method_id,
        )

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("low_payload", self.low_payload),
            ("high_payload", self.high_payload),
            ("scales", self.scales),
            ("packed_precision_mask", self.policy.packed_precision_mask),
            ("pool_offsets", self.policy.pool_offsets),
            ("padding", self.padding),
        )

    @property
    def data_bytes(self) -> int:
        return sum(
            _storage_bytes(tensor)
            for name, tensor in self.persistent_tensors()
            if name != "padding"
        )

    @property
    def resident_bytes(self) -> int:
        return sum(_storage_bytes(tensor) for _, tensor in self.persistent_tensors())

    def _packed(self) -> PackedMixedQuantizedTensor:
        geometry = self.policy.geometry
        low_spec, high_spec = _q48_specs(geometry)
        return PackedMixedQuantizedTensor(
            low_payload=self.low_payload,
            high_payload=self.high_payload,
            scales=self.scales,
            precision_mask=self.policy.packed_precision_mask,
            low_spec=low_spec,
            high_spec=high_spec,
            original_shape=(geometry.total_rows, geometry.value_width),
            original_dtype=torch.float32,
            flattened_size=geometry.value_width,
            padded_size=geometry.value_width,
            rows=geometry.total_rows,
            groups_per_row=1,
        )

    def materialize(self) -> dict[int, torch.Tensor]:
        """Materialize decoded FP32 states without retaining an FP32 mirror."""

        geometry = self.policy.geometry
        encoded = self._packed().dequantize()
        result: dict[int, torch.Tensor] = {}
        for position, layer_index in enumerate(geometry.layer_indices):
            start = position * geometry.rows_per_layer
            stop = start + geometry.rows_per_layer
            layer = encoded[start:stop].reshape(
                1,
                geometry.heads,
                geometry.key_rows,
                geometry.value_width,
            )
            result[layer_index] = right_rht_decode(
                layer,
                layer_index=layer_index,
                expected_heads=geometry.heads,
                output_dtype=torch.float32,
            )
        return result

    def clone_to(self, device: torch.device | str) -> StaticPackedRhtQ48State:
        return StaticPackedRhtQ48State(
            policy=self.policy.clone_to(device),
            low_payload=self.low_payload.detach().to(device).clone(),
            high_payload=self.high_payload.detach().to(device).clone(),
            scales=self.scales.detach().to(device).clone(),
            padding=self.padding.detach().to(device).clone(),
        )


def pack_static_rht_q48(
    states: Mapping[int, torch.Tensor],
    policy: StaticRhtQ48Policy,
) -> StaticPackedRhtQ48State:
    """RHT-encode and physically pack all recurrent rows under a Q4/Q8 policy."""

    if not isinstance(policy, StaticRhtQ48Policy):
        raise TypeError("policy must be a StaticRhtQ48Policy")
    geometry = policy.geometry
    device = _validate_states(states, geometry=geometry)
    encoded_layers = [
        right_rht_encode(
            states[layer_index],
            layer_index=layer_index,
            expected_heads=geometry.heads,
            output_dtype=torch.float32,
        ).reshape(geometry.rows_per_layer, geometry.value_width)
        for layer_index in geometry.layer_indices
    ]
    encoded = torch.cat(encoded_layers, dim=0)
    mask = policy.high_precision_mask().reshape(-1).to(device)
    low_spec, high_spec = _q48_specs(geometry)
    packed = quantize_pack_mixed(
        encoded,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    physical_policy = policy.clone_to(device)
    if not torch.equal(packed.precision_mask, physical_policy.packed_precision_mask):
        raise RuntimeError("physical packer changed the canonical precision-mask stream")
    ledger = static_q48_byte_ledger(
        geometry,
        policy.promoted_rows,
        method_id=policy.method_id,
    )
    return StaticPackedRhtQ48State(
        policy=physical_policy,
        low_payload=packed.low_payload,
        high_payload=packed.high_payload,
        scales=packed.scales,
        padding=torch.zeros(ledger.alignment_bytes, dtype=torch.uint8, device=device),
    )


def verify_static_packed_rht_q48(
    state: StaticPackedRhtQ48State,
) -> dict[str, object]:
    """Revalidate a packed Q4/Q8 state and return storage-only evidence."""

    if not isinstance(state, StaticPackedRhtQ48State):
        raise TypeError("state must be a StaticPackedRhtQ48State")
    verified = StaticPackedRhtQ48State(
        policy=state.policy,
        low_payload=state.low_payload,
        high_payload=state.high_payload,
        scales=state.scales,
        padding=state.padding,
    )
    evidence = verified.policy.evidence_dict()
    evidence["physical_data_bytes"] = verified.data_bytes
    evidence["physical_resident_bytes"] = verified.resident_bytes
    evidence["persistent_tensor_dtypes"] = {
        name: str(tensor.dtype) for name, tensor in verified.persistent_tensors()
    }
    return evidence
