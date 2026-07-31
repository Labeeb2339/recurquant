#!/usr/bin/env python
"""Independent synthetic Stage-0 verifier for Experiment 012.

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
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
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
RAW_STATE_ELEMENTS_PER_LAYER = HEADS * ROWS * WIDTH
RAW_STATE_ELEMENTS_ALL_LAYERS = STATE_ELEMENTS

EXPANDED_Q48_BASE_BYTES = 2_585_088
EXPANDED_Q48_PROMOTIONS = 13_587
EXPANDED_Q48_PADDING_BYTES = 8
MULTIBIT_BASE_BYTES = 2_589_696
MULTIBIT_MARGINAL_STEPS = 27_030
MULTIBIT_PADDING_BYTES = 8
RESIDUAL_Q4_BASE_BYTES = 2_585_088
RESIDUAL_Q4_ROWS = 13_175
RESIDUAL_Q4_PADDING_BYTES = 26

EXPERIMENT_ID = "experiment012"
PRODUCTION_SCHEMA = f"recurquant.{EXPERIMENT_ID}.stage0.production.v1"
PRODUCTION_SCHEMA_VERSION = 1
GIT_REPOSITORY_BINDING_SCHEMA = "recurquant.git-repository-binding.v1"
PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256 = (
    "2466ad25043894fcd1604c97c373e5d5680061fdb7637f861b83d5c9465c31fe"
)
RUNTIME_PACKAGE_DISTRIBUTIONS = (
    "datasets",
    "fsspec",
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)
EFFECTIVE_PLAN_SHA256 = "6b7d8f6b7a4b1142f0363bf3387fa20f8d3e3b0656c4367680f84d76ee528640"
STATELEASE_SELECTION_METHOD = "statelease_cut4_cut5_right_rht_query_ema32_weighted_mse_fisher_quota"
LINEAR_LAYER_INDICES = (
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
LAYER_QUOTAS = {
    0: 355,
    1: 380,
    2: 269,
    4: 179,
    5: 185,
    6: 105,
    8: 80,
    9: 43,
    10: 84,
    12: 30,
    13: 62,
    14: 54,
    16: 45,
    17: 27,
    18: 7,
    20: 9,
    21: 7,
    22: 55,
}
EXPERIMENT012_SOURCE_PROVENANCE_PATHS = (
    "research/EXPERIMENT_012_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_012_STAGE_A_IDENTITY.md",
    "research/EXPERIMENT_011_STAGE_A_ADMINISTRATIVE_NULL.md",
    "evidence/experiment011-statelease-stage-a-administrative-null.json",
    "artifacts/experiment011-statelease-stage-a-666.attempt.json",
    "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_011_STAGE_A_IDENTITY.md",
    "research/EXPERIMENT_010_STAGE_A_ADMINISTRATIVE_NULL.md",
    "evidence/experiment010-statelease-stage-a-administrative-null.json",
    "artifacts/experiment010-statelease-stage-a-666.attempt.json",
    "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_010_STAGE_A_IDENTITY.md",
)
SOURCE_IDENTITY_PATHS = (
    "pyproject.toml",
    "scripts/capture_statelease_stage0.py",
    "scripts/screen_statelease_stage_a.py",
    "scripts/verify_statelease_stage0.py",
    *EXPERIMENT012_SOURCE_PROVENANCE_PATHS,
    "src/recurquant/__init__.py",
    "src/recurquant/cache.py",
    "src/recurquant/cli.py",
    "src/recurquant/confirmation.py",
    "src/recurquant/evaluation.py",
    "src/recurquant/evidence.py",
    "src/recurquant/finite_difference.py",
    "src/recurquant/fisher_sensitivity.py",
    "src/recurquant/horizon.py",
    "src/recurquant/horizon_calibration.py",
    "src/recurquant/intervention.py",
    "src/recurquant/metrics.py",
    "src/recurquant/mixed_quantization.py",
    "src/recurquant/model_fisher.py",
    "src/recurquant/multibit_policy.py",
    "src/recurquant/multibit_quantization.py",
    "src/recurquant/packed_cache.py",
    "src/recurquant/policies.py",
    "src/recurquant/public_data.py",
    "src/recurquant/quantization.py",
    "src/recurquant/query_energy.py",
    "src/recurquant/qwen35.py",
    "src/recurquant/qwen35_quickstart.py",
    "src/recurquant/rht.py",
    "src/recurquant/row_policy.py",
    "src/recurquant/signals.py",
    "src/recurquant/statelease.py",
    "src/recurquant/statelease_baselines.py",
    "src/recurquant/statelease_cache.py",
    "src/recurquant/statelease_equal_byte_baselines.py",
    "src/recurquant/statelease_equal_byte_cache.py",
    "src/recurquant/statelease_evaluation.py",
    "src/recurquant/statelease_observer.py",
    "src/recurquant/storage_boundary_validation.py",
    "src/recurquant/transformers_cache.py",
    "src/recurquant/transition_observer.py",
    "src/recurquant/triton_state.py",
    "tests/test_capture_statelease_stage0.py",
    "tests/test_mixed_quantization.py",
    "tests/test_multibit_policy.py",
    "tests/test_multibit_quantization.py",
    "tests/test_quantization.py",
    "tests/test_qwen35_factory.py",
    "tests/test_rht.py",
    "tests/test_right_rht_query_ema_cache.py",
    "tests/test_row_policy.py",
    "tests/test_statelease.py",
    "tests/test_statelease_baselines.py",
    "tests/test_statelease_cache.py",
    "tests/test_statelease_equal_byte_baselines.py",
    "tests/test_statelease_equal_byte_cache.py",
    "tests/test_statelease_evaluation.py",
    "tests/test_statelease_observer.py",
    "tests/test_screen_statelease_stage_a.py",
    "tests/test_verify_statelease_stage0.py",
)
REQUIRED_LOADED_RECURQUANT_MODULE_PATHS = (
    ("recurquant", "src/recurquant/__init__.py"),
    ("recurquant.evidence", "src/recurquant/evidence.py"),
    ("recurquant.finite_difference", "src/recurquant/finite_difference.py"),
    ("recurquant.fisher_sensitivity", "src/recurquant/fisher_sensitivity.py"),
    ("recurquant.horizon", "src/recurquant/horizon.py"),
    ("recurquant.horizon_calibration", "src/recurquant/horizon_calibration.py"),
    ("recurquant.intervention", "src/recurquant/intervention.py"),
    ("recurquant.mixed_quantization", "src/recurquant/mixed_quantization.py"),
    ("recurquant.model_fisher", "src/recurquant/model_fisher.py"),
    ("recurquant.multibit_policy", "src/recurquant/multibit_policy.py"),
    ("recurquant.multibit_quantization", "src/recurquant/multibit_quantization.py"),
    ("recurquant.packed_cache", "src/recurquant/packed_cache.py"),
    ("recurquant.quantization", "src/recurquant/quantization.py"),
    ("recurquant.query_energy", "src/recurquant/query_energy.py"),
    ("recurquant.qwen35", "src/recurquant/qwen35.py"),
    ("recurquant.rht", "src/recurquant/rht.py"),
    ("recurquant.row_policy", "src/recurquant/row_policy.py"),
    ("recurquant.statelease", "src/recurquant/statelease.py"),
    ("recurquant.statelease_baselines", "src/recurquant/statelease_baselines.py"),
    ("recurquant.statelease_cache", "src/recurquant/statelease_cache.py"),
    (
        "recurquant.statelease_equal_byte_baselines",
        "src/recurquant/statelease_equal_byte_baselines.py",
    ),
    (
        "recurquant.statelease_equal_byte_cache",
        "src/recurquant/statelease_equal_byte_cache.py",
    ),
    ("recurquant.statelease_evaluation", "src/recurquant/statelease_evaluation.py"),
    ("recurquant.statelease_observer", "src/recurquant/statelease_observer.py"),
    ("recurquant.transition_observer", "src/recurquant/transition_observer.py"),
)

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
        raise ValueError("Experiment 012 replay verification requires batch size one")
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
    if codes16.shape[0] == 0:
        return torch.empty((0, bytes_per_row), dtype=torch.uint8)
    mask = (1 << bits) - 1
    unsigned = torch.bitwise_and(codes16, mask)
    if bits == 4 and codes16.shape[1] % 2 == 0:
        pairs = unsigned.reshape(codes16.shape[0], -1, 2)
        return (
            (pairs[..., 0] | torch.bitwise_left_shift(pairs[..., 1], 4))
            .to(torch.uint8)
            .contiguous()
        )
    if bits == 6 and codes16.shape[1] % 4 == 0:
        groups = unsigned.reshape(codes16.shape[0], -1, 4)
        packed_groups = torch.empty(
            (codes16.shape[0], groups.shape[1], 3),
            dtype=torch.int16,
        )
        packed_groups[..., 0] = groups[..., 0] | torch.bitwise_left_shift(
            torch.bitwise_and(groups[..., 1], 0x03),
            6,
        )
        packed_groups[..., 1] = torch.bitwise_right_shift(
            groups[..., 1],
            2,
        ) | torch.bitwise_left_shift(
            torch.bitwise_and(groups[..., 2], 0x0F),
            4,
        )
        packed_groups[..., 2] = torch.bitwise_right_shift(
            groups[..., 2],
            4,
        ) | torch.bitwise_left_shift(groups[..., 3], 2)
        return packed_groups.to(torch.uint8).reshape(codes16.shape[0], -1).contiguous()
    packed = torch.zeros((codes16.shape[0], bytes_per_row), dtype=torch.uint8)
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
    if payload.shape[0] == 0:
        return torch.empty((0, width), dtype=torch.int16, device=payload.device)
    cpu_payload = payload.to("cpu")
    if bits == 4 and width % 2 == 0:
        expanded = torch.empty(
            (payload.shape[0], payload.shape[1], 2),
            dtype=torch.int16,
        )
        source = cpu_payload.to(torch.int16)
        expanded[..., 0] = torch.bitwise_and(source, 0x0F)
        expanded[..., 1] = torch.bitwise_right_shift(source, 4)
        unsigned = expanded.reshape(payload.shape[0], width)
        signed = torch.where(unsigned >= 8, unsigned - 16, unsigned)
        return signed.to(payload.device)
    if bits == 6 and width % 4 == 0:
        source = cpu_payload.reshape(payload.shape[0], -1, 3).to(torch.int16)
        expanded = torch.empty(
            (payload.shape[0], source.shape[1], 4),
            dtype=torch.int16,
        )
        expanded[..., 0] = torch.bitwise_and(source[..., 0], 0x3F)
        expanded[..., 1] = torch.bitwise_right_shift(
            source[..., 0],
            6,
        ) | torch.bitwise_left_shift(
            torch.bitwise_and(source[..., 1], 0x0F),
            2,
        )
        expanded[..., 2] = torch.bitwise_right_shift(
            source[..., 1],
            4,
        ) | torch.bitwise_left_shift(
            torch.bitwise_and(source[..., 2], 0x03),
            4,
        )
        expanded[..., 3] = torch.bitwise_right_shift(source[..., 2], 2)
        unsigned = expanded.reshape(payload.shape[0], width)
        signed = torch.where(unsigned >= 32, unsigned - 64, unsigned)
        return signed.to(payload.device)
    output = torch.empty((payload.shape[0], width), dtype=torch.int16)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    modulus = 1 << bits
    for row_index, row in enumerate(cpu_payload.tolist()):
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
        raise ValueError("unknown Experiment 012 stage")
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
    """Fail if this verifier imports the package or Stage-0 producer under test."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                modules.extend(alias.name for alias in node.names)
            else:
                modules.append(node.module)
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            if module == "recurquant" or module.startswith("recurquant."):
                _fail(f"independent verifier imports package under test: {module}")
            if module in {
                "capture_statelease_stage0",
                "scripts.capture_statelease_stage0",
            }:
                _fail(f"independent verifier imports Stage-0 producer under test: {module}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(value: object) -> str:
    """Hash the closed artifact schema without trusting torch serialization."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        item_type = type(item)
        if item is None:
            digest.update(b"n")
        elif item_type is bool:
            digest.update(b"b1" if item else b"b0")
        elif item_type is int:
            digest.update(b"i")
            digest.update(str(item).encode("ascii"))
            digest.update(b"\0")
        elif item_type is float:
            if not math.isfinite(item):
                _fail("artifact contains a non-finite scalar")
            digest.update(b"f")
            digest.update(item.hex().encode("ascii"))
            digest.update(b"\0")
        elif item_type is str:
            encoded = item.encode("utf-8")
            digest.update(b"s")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        elif item_type is torch.Tensor:
            digest.update(b"t")
            digest.update(_tensor_digest(item).encode("ascii"))
        elif item_type is dict:
            if any(type(key) is not str for key in item):
                _fail("artifact mapping keys must be plain strings")
            digest.update(b"d")
            digest.update(len(item).to_bytes(8, "little"))
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif item_type is list or item_type is tuple:
            digest.update(b"l" if item_type is list else b"t")
            digest.update(len(item).to_bytes(8, "little"))
            for child in item:
                visit(child)
        else:
            _fail(f"artifact contains unsupported type {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _require_exact_keys(
    value: object,
    *,
    name: str,
    keys: Sequence[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    actual = set(value)
    expected = set(keys)
    if actual != expected:
        _fail(
            f"{name} violates the closed schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if any(not isinstance(key, str) for key in value):
        _fail(f"{name} contains a non-string key")
    return value


def _read_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one stable regular-file handle and reject mutation during the read."""

    if path.is_symlink():
        _fail(f"{label} must be a regular non-symlink file")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                _fail(f"{label} must be a regular non-symlink file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise Stage0VerificationError(f"cannot read {label}: {type(error).__name__}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        _fail(f"{label} changed while its authenticated bytes were read")
    return payload


def _parse_sha256_sidecar(payload: bytes, artifact_path: Path) -> str:
    try:
        artifact_name = artifact_path.name.encode("ascii")
    except UnicodeEncodeError as error:
        raise Stage0VerificationError(
            "artifact filename is not representable in the closed ASCII sidecar syntax"
        ) from error
    digest_bytes = payload[:64]
    expected_payload = digest_bytes + b"  " + artifact_name + b"\n"
    if (
        len(digest_bytes) != 64
        or any(byte not in b"0123456789abcdef" for byte in digest_bytes)
        or payload != expected_payload
    ):
        _fail("artifact SHA-256 sidecar has invalid closed syntax")
    return digest_bytes.decode("ascii")


def _authenticated_artifact_bytes(
    artifact_path: Path,
    sidecar_path: Path,
) -> tuple[bytes, dict[str, str]]:
    """Read and authenticate one exact artifact/sidecar byte pair."""

    if (
        artifact_path.is_symlink()
        or sidecar_path.is_symlink()
        or not artifact_path.is_file()
        or not sidecar_path.is_file()
    ):
        _fail("artifact and SHA-256 sidecar must be regular non-symlink files")
    sidecar_payload = _read_regular_file_bytes(sidecar_path, label="SHA-256 sidecar")
    artifact_payload = _read_regular_file_bytes(artifact_path, label="serialized artifact")
    expected_hash = _parse_sha256_sidecar(sidecar_payload, artifact_path)
    artifact_hash = hashlib.sha256(artifact_payload).hexdigest()
    if artifact_hash != expected_hash:
        _fail("serialized artifact SHA-256 differs from its authenticated sidecar")
    return artifact_payload, {
        "artifact_file_sha256": artifact_hash,
        "sidecar_file_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
        "sidecar_declared_artifact_sha256": expected_hash,
    }


def _artifact_integrity_snapshot(
    artifact_path: Path,
    sidecar_path: Path,
) -> dict[str, str]:
    """Authenticate the exact current artifact/sidecar bytes."""

    _, integrity = _authenticated_artifact_bytes(artifact_path, sidecar_path)
    return integrity


def _load_authenticated_production_artifact_with_integrity(
    artifact_path: Path,
    sidecar: Path,
) -> tuple[Mapping[str, object], dict[str, str]]:
    artifact_payload, loaded_integrity = _authenticated_artifact_bytes(
        artifact_path,
        sidecar,
    )
    try:
        loaded = torch.load(
            io.BytesIO(artifact_payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise Stage0VerificationError(
            f"weights-only artifact loading failed: {type(error).__name__}"
        ) from error
    if _artifact_integrity_snapshot(artifact_path, sidecar) != loaded_integrity:
        _fail("artifact or sidecar changed during authenticated weights-only load")
    if type(loaded) is not dict:
        _fail("artifact root must be an exact plain dict")
    artifact = _require_exact_keys(
        loaded,
        name="artifact",
        keys=(
            "schema",
            "schema_version",
            "declarations",
            "source_identity",
            "runtime_identity",
            "method_identity",
            "production_trace",
            "successful_kernel_receipt",
            "resident_snapshot",
            "lifecycle",
            "cc1_compatibility",
            "equal_byte_comparators",
            "canonical_payload_sha256",
        ),
    )
    canonical = artifact["canonical_payload_sha256"]
    if (
        not isinstance(canonical, str)
        or len(canonical) != 64
        or any(character not in "0123456789abcdef" for character in canonical)
    ):
        _fail("artifact canonical payload digest is malformed")
    unhashed = {key: artifact[key] for key in artifact if key != "canonical_payload_sha256"}
    if canonical_payload_sha256(unhashed) != canonical:
        _fail("artifact canonical payload SHA-256 is invalid")
    return artifact, loaded_integrity


def load_authenticated_production_artifact(
    artifact_path: Path,
    *,
    sha256_path: Path | None = None,
) -> Mapping[str, object]:
    """Authenticate immutable bytes, then load only tensor-safe primitives."""

    if artifact_path.is_symlink() or (sha256_path is not None and sha256_path.is_symlink()):
        _fail("artifact and SHA-256 sidecar must be regular non-symlink files")
    artifact_path = artifact_path.resolve()
    sidecar = (
        artifact_path.with_suffix(artifact_path.suffix + ".sha256")
        if sha256_path is None
        else sha256_path.resolve()
    )
    artifact, _ = _load_authenticated_production_artifact_with_integrity(
        artifact_path,
        sidecar,
    )
    return artifact


def _recorded_source_snapshot(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    snapshot = _require_exact_keys(
        value,
        name=name,
        keys=(
            "repo_head",
            "repository_binding",
            "source_hashes",
            "source_set_sha256",
            "head_blob_hashes",
            "worktree_blob_hashes",
            "sources_match_head",
            "worktree_clean",
        ),
    )
    repo_head = snapshot["repo_head"]
    if (
        not isinstance(repo_head, str)
        or len(repo_head) != 40
        or any(character not in "0123456789abcdef" for character in repo_head)
    ):
        _fail(f"{name} lacks an exact repository HEAD")
    repository_binding = _recorded_repository_binding(
        snapshot["repository_binding"],
        name=f"{name}.repository_binding",
    )
    source_hashes = _require_exact_keys(
        snapshot["source_hashes"],
        name=f"{name}.source_hashes",
        keys=SOURCE_IDENTITY_PATHS,
    )
    malformed = [
        relative
        for relative, digest in source_hashes.items()
        if not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ]
    if malformed:
        _fail(f"{name} contains malformed source hashes: {malformed}")
    source_set_sha256 = snapshot["source_set_sha256"]
    if source_set_sha256 != canonical_payload_sha256(source_hashes):
        _fail(f"{name} source-set canonical digest differs")
    if snapshot["worktree_clean"] is not True:
        _fail(f"{name} does not attest a clean complete worktree")
    head_blob_hashes = _require_exact_keys(
        snapshot["head_blob_hashes"],
        name=f"{name}.head_blob_hashes",
        keys=SOURCE_IDENTITY_PATHS,
    )
    worktree_blob_hashes = _require_exact_keys(
        snapshot["worktree_blob_hashes"],
        name=f"{name}.worktree_blob_hashes",
        keys=SOURCE_IDENTITY_PATHS,
    )
    malformed_blobs = [
        f"{kind}:{relative}"
        for kind, hashes in (
            ("head", head_blob_hashes),
            ("worktree", worktree_blob_hashes),
        )
        for relative, digest in hashes.items()
        if not isinstance(digest, str)
        or len(digest) != 40
        or any(character not in "0123456789abcdef" for character in digest)
    ]
    if malformed_blobs:
        _fail(f"{name} contains malformed Git blob hashes: {malformed_blobs}")
    if snapshot["sources_match_head"] is not True or dict(head_blob_hashes) != dict(
        worktree_blob_hashes
    ):
        _fail(f"{name} source bytes do not equal their regular-file blobs at HEAD")
    return {
        "repo_head": repo_head,
        "repository_binding": repository_binding,
        "source_hashes": dict(source_hashes),
        "source_set_sha256": source_set_sha256,
        "head_blob_hashes": dict(head_blob_hashes),
        "worktree_blob_hashes": dict(worktree_blob_hashes),
        "sources_match_head": True,
        "worktree_clean": True,
    }


def _sanitized_verifier_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _verifier_git(
    repo_root: Path,
    *arguments: str,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.useReplaceRefs=false",
            "-c",
            f"core.attributesFile={os.devnull}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *arguments,
        ],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
        input=input_text,
        env=_sanitized_verifier_git_environment(),
    )


def _resolved_verifier_git_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _private_verifier_path_sha256(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve())).replace("\\", "/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _verified_local_git_config_sha256(repo_root: Path) -> str:
    process = _verifier_git(
        repo_root,
        "config",
        "--local",
        "--no-includes",
        "--null",
        "--list",
    )
    entries: list[dict[str, str]] = []
    for raw in process.stdout.split("\0"):
        if not raw:
            continue
        key, separator, value = raw.partition("\n")
        if not separator or not key:
            _fail("local Git config contains a malformed entry")
        normalized_key = key.lower()
        entries.append({"key": normalized_key, "value": value})
        forbidden = (
            normalized_key.startswith(("include.", "includeif.", "filter."))
            or normalized_key
            in {
                "core.alternaterefscommand",
                "core.alternaterefsprefixes",
                "core.attributesfile",
                "core.fsmonitor",
                "core.hookspath",
                "core.sparsecheckout",
                "core.sparsecheckoutcone",
                "core.untrackedcache",
                "core.worktree",
                "extensions.partialclone",
                "extensions.worktreeconfig",
                "index.sparse",
            }
            or (
                normalized_key.startswith("remote.")
                and normalized_key.endswith((".promisor", ".partialclonefilter"))
            )
        )
        if forbidden:
            _fail(f"unsafe local Git config key is present: {normalized_key}")
        if normalized_key == "core.usereplacerefs" and value.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            _fail("local Git config enables replacement objects")
    values_by_key: dict[str, list[str]] = {}
    for entry in entries:
        values_by_key.setdefault(entry["key"], []).append(entry["value"])
    if values_by_key.get("core.repositoryformatversion") != ["0"]:
        _fail("Git repository format version is not exactly zero")
    if values_by_key.get("core.bare") != ["false"]:
        _fail("Git repository must explicitly be non-bare")
    return canonical_payload_sha256(entries)


def _assert_verifier_index_has_no_hidden_flags(repo_root: Path) -> None:
    """Independently reject index flags that can conceal tracked-file drift."""

    try:
        process = _verifier_git(repo_root, "ls-files", "--cached", "-v", "-z")
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0VerificationError("cannot authenticate Git index visibility flags") from error
    records = [record for record in process.stdout.split("\0") if record]
    if any(len(record) < 3 or record[1] != " " for record in records):
        _fail("Git index visibility output is malformed")
    unsafe_tags = sorted({record[0] for record in records if record[0] != "H"})
    if unsafe_tags:
        _fail(f"Git index contains hidden or non-canonical tracked entries (tags={unsafe_tags})")


def _current_repository_binding(repo_root: Path) -> dict[str, object]:
    try:
        local_config_sha256 = _verified_local_git_config_sha256(repo_root)
        _assert_verifier_index_has_no_hidden_flags(repo_root)
        top_level = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--show-toplevel").stdout.strip(),
        )
        git_dir = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--absolute-git-dir").stdout.strip(),
        )
        common_dir = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--git-common-dir").stdout.strip(),
        )
        index_path = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--git-path", "index").stdout.strip(),
        )
        object_dir = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--git-path", "objects").stdout.strip(),
        )
        object_format = _verifier_git(
            repo_root,
            "rev-parse",
            "--show-object-format",
        ).stdout.strip()
        inside_worktree = _verifier_git(
            repo_root,
            "rev-parse",
            "--is-inside-work-tree",
        ).stdout.strip()
        bare = _verifier_git(
            repo_root,
            "rev-parse",
            "--is-bare-repository",
        ).stdout.strip()
        shallow = _verifier_git(
            repo_root,
            "rev-parse",
            "--is-shallow-repository",
        ).stdout.strip()
        shallow_path = _resolved_verifier_git_path(
            repo_root,
            _verifier_git(repo_root, "rev-parse", "--git-path", "shallow").stdout.strip(),
        )
        replace_refs = _verifier_git(
            repo_root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace/",
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0VerificationError(
            "cannot authenticate the Git repository/object view"
        ) from error

    expected_top_level = repo_root.resolve()
    if top_level != expected_top_level or inside_worktree != "true" or bare != "false":
        _fail("Git top-level/worktree identity differs from the expected repository")
    if object_format != "sha1":
        _fail("Stage-0 source identity requires the exact SHA-1 Git object format")
    if index_path != git_dir / "index":
        _fail("Git index is not the exact worktree index")
    if object_dir != common_dir / "objects" or not object_dir.is_dir():
        _fail("Git object directory is not the exact common object store")

    dot_git = expected_top_level / ".git"
    try:
        if dot_git.is_dir():
            if dot_git.is_symlink():
                _fail("main-worktree .git directory must not be a symlink")
            git_dir_kind = "main_worktree"
            if git_dir != dot_git.resolve() or common_dir != git_dir:
                _fail("main-worktree Git directory/common directory identity differs")
        elif dot_git.is_file():
            if dot_git.is_symlink():
                _fail("linked-worktree .git marker must not be a symlink")
            marker = dot_git.read_text(encoding="utf-8").strip()
            prefix = "gitdir: "
            if not marker.startswith(prefix):
                _fail("linked-worktree .git marker is malformed")
            declared_git_dir = Path(marker[len(prefix) :])
            if not declared_git_dir.is_absolute():
                declared_git_dir = expected_top_level / declared_git_dir
            if declared_git_dir.resolve() != git_dir:
                _fail("linked-worktree .git marker redirects to a different Git directory")
            if git_dir.parent != common_dir / "worktrees":
                _fail("linked-worktree Git directory is outside the common worktree registry")
            reverse_pointer = git_dir / "gitdir"
            if not reverse_pointer.is_file() or reverse_pointer.is_symlink():
                _fail("linked-worktree Git directory has no canonical reverse pointer")
            declared_dot_git = Path(reverse_pointer.read_text(encoding="utf-8").strip())
            if not declared_dot_git.is_absolute():
                declared_dot_git = git_dir / declared_dot_git
            if declared_dot_git.resolve() != dot_git.resolve():
                _fail("linked-worktree Git directory reverse pointer targets a different worktree")
            git_dir_kind = "linked_worktree"
        else:
            _fail("repository has neither a canonical .git directory nor marker")
    except OSError as error:
        raise Stage0VerificationError("cannot authenticate the worktree Git binding") from error

    unsafe_object_files = (
        object_dir / "info" / "alternates",
        object_dir / "info" / "http-alternates",
        git_dir / "info" / "grafts",
        common_dir / "info" / "grafts",
        shallow_path,
    )
    if any(path.exists() for path in unsafe_object_files):
        _fail("Git alternates, grafts, or shallow object view is not permitted")
    if shallow != "false":
        _fail("shallow Git history is not permitted")
    if replace_refs:
        _fail("Git replacement refs are not permitted")

    return {
        "schema": GIT_REPOSITORY_BINDING_SCHEMA,
        "top_level_path_sha256": _private_verifier_path_sha256(top_level),
        "worktree_path_sha256": _private_verifier_path_sha256(expected_top_level),
        "git_dir_path_sha256": _private_verifier_path_sha256(git_dir),
        "common_dir_path_sha256": _private_verifier_path_sha256(common_dir),
        "index_path_sha256": _private_verifier_path_sha256(index_path),
        "object_dir_path_sha256": _private_verifier_path_sha256(object_dir),
        "git_dir_kind": git_dir_kind,
        "object_format": object_format,
        "inside_worktree": True,
        "bare": False,
        "shallow": False,
        "alternates_absent": True,
        "grafts_absent": True,
        "replace_refs_absent": True,
        "unsafe_local_config_absent": True,
        "hidden_index_flags_absent": True,
        "local_config_sha256": local_config_sha256,
        "replacement_objects_disabled": True,
        "system_and_global_config_disabled": True,
        "fsmonitor_and_untracked_cache_disabled": True,
        "hooks_disabled": True,
        "worktree_gitdir_binding_verified": True,
        "raw_source_hash_mode": "git_hash_object_no_filters_stdin_paths",
    }


def _recorded_repository_binding(value: object, *, name: str) -> dict[str, object]:
    binding = _require_exact_keys(
        value,
        name=name,
        keys=(
            "schema",
            "top_level_path_sha256",
            "worktree_path_sha256",
            "git_dir_path_sha256",
            "common_dir_path_sha256",
            "index_path_sha256",
            "object_dir_path_sha256",
            "git_dir_kind",
            "object_format",
            "inside_worktree",
            "bare",
            "shallow",
            "alternates_absent",
            "grafts_absent",
            "replace_refs_absent",
            "unsafe_local_config_absent",
            "hidden_index_flags_absent",
            "local_config_sha256",
            "replacement_objects_disabled",
            "system_and_global_config_disabled",
            "fsmonitor_and_untracked_cache_disabled",
            "hooks_disabled",
            "worktree_gitdir_binding_verified",
            "raw_source_hash_mode",
        ),
    )
    digest_fields = (
        "top_level_path_sha256",
        "worktree_path_sha256",
        "git_dir_path_sha256",
        "common_dir_path_sha256",
        "index_path_sha256",
        "object_dir_path_sha256",
        "local_config_sha256",
    )
    malformed = [
        field
        for field in digest_fields
        if not isinstance(binding[field], str)
        or len(binding[field]) != 64
        or any(character not in "0123456789abcdef" for character in binding[field])
    ]
    if malformed:
        _fail(f"{name} contains malformed private path/config hashes: {malformed}")
    if binding["schema"] != GIT_REPOSITORY_BINDING_SCHEMA:
        _fail(f"{name} schema differs")
    if binding["object_format"] != "sha1":
        _fail(f"{name} object format differs")
    if binding["raw_source_hash_mode"] != "git_hash_object_no_filters_stdin_paths":
        _fail(f"{name} raw source hash mode differs")
    git_dir_kind = binding["git_dir_kind"]
    if not isinstance(git_dir_kind, str) or git_dir_kind not in {
        "main_worktree",
        "linked_worktree",
    }:
        _fail(f"{name} Git-directory kind is invalid")
    expected_booleans = {
        "inside_worktree": True,
        "bare": False,
        "shallow": False,
        "alternates_absent": True,
        "grafts_absent": True,
        "replace_refs_absent": True,
        "unsafe_local_config_absent": True,
        "hidden_index_flags_absent": True,
        "replacement_objects_disabled": True,
        "system_and_global_config_disabled": True,
        "fsmonitor_and_untracked_cache_disabled": True,
        "hooks_disabled": True,
        "worktree_gitdir_binding_verified": True,
    }
    if any(binding[field] is not expected for field, expected in expected_booleans.items()):
        _fail(f"{name} does not attest the exact safe Git/object view")
    if binding["top_level_path_sha256"] != binding["worktree_path_sha256"]:
        _fail(f"{name} top-level/worktree path binding differs")
    if (
        git_dir_kind == "main_worktree"
        and binding["git_dir_path_sha256"] != binding["common_dir_path_sha256"]
    ):
        _fail(f"{name} main-worktree Git/common-directory binding differs")
    if (
        git_dir_kind == "linked_worktree"
        and binding["git_dir_path_sha256"] == binding["common_dir_path_sha256"]
    ):
        _fail(f"{name} linked-worktree Git/common-directory binding is not distinct")
    return dict(binding)


def _git_blob_hashes_for_authenticated_sources(
    repo_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        tree_process = _verifier_git(
            repo_root,
            "ls-tree",
            "-r",
            "--full-tree",
            "HEAD",
            "--",
            *SOURCE_IDENTITY_PATHS,
        )
        worktree_process = _verifier_git(
            repo_root,
            "hash-object",
            "--no-filters",
            "--stdin-paths",
            input_text="\n".join(SOURCE_IDENTITY_PATHS),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0VerificationError(
            "cannot compare authenticated source bytes with repository HEAD"
        ) from error
    head_blobs: dict[str, str] = {}
    for line in tree_process.stdout.splitlines():
        metadata, separator, relative = line.partition("\t")
        parts = metadata.split()
        if (
            separator != "\t"
            or len(parts) != 3
            or parts[0] not in {"100644", "100755"}
            or parts[1] != "blob"
            or relative not in SOURCE_IDENTITY_PATHS
        ):
            _fail("authenticated source has an invalid repository tree entry")
        head_blobs[relative] = parts[2]
    if set(head_blobs) != set(SOURCE_IDENTITY_PATHS):
        missing = sorted(set(SOURCE_IDENTITY_PATHS) - set(head_blobs))
        _fail(f"authenticated source is not a regular tracked HEAD blob: {missing}")
    worktree_lines = worktree_process.stdout.splitlines()
    if len(worktree_lines) != len(SOURCE_IDENTITY_PATHS):
        _fail("Git did not hash every authenticated worktree source")
    worktree_blobs = dict(zip(SOURCE_IDENTITY_PATHS, worktree_lines, strict=True))
    malformed = [
        f"{kind}:{relative}"
        for kind, hashes in (("head", head_blobs), ("worktree", worktree_blobs))
        for relative, digest in hashes.items()
        if len(digest) != 40 or any(character not in "0123456789abcdef" for character in digest)
    ]
    if malformed:
        _fail(f"authenticated source Git blob identity is malformed: {malformed}")
    return head_blobs, worktree_blobs


def _current_repository_source_snapshot(repo_root: Path) -> dict[str, object]:
    repository_binding = _current_repository_binding(repo_root)
    missing = [
        relative
        for relative in SOURCE_IDENTITY_PATHS
        if not (repo_root / relative).is_file() or (repo_root / relative).is_symlink()
    ]
    if missing:
        _fail(f"authenticated Stage-0 source set is incomplete: {missing}")
    source_hashes = {
        relative: _file_sha256(repo_root / relative) for relative in SOURCE_IDENTITY_PATHS
    }
    head_blobs, worktree_blobs = _git_blob_hashes_for_authenticated_sources(repo_root)
    try:
        repo_head = _verifier_git(repo_root, "rev-parse", "HEAD").stdout.strip()
        status = _verifier_git(
            repo_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0VerificationError("cannot verify current repository source identity") from error
    if len(repo_head) != 40 or any(character not in "0123456789abcdef" for character in repo_head):
        _fail("current repository HEAD is not a lowercase SHA-1 commit identity")
    return {
        "repo_head": repo_head,
        "repository_binding": repository_binding,
        "source_hashes": source_hashes,
        "source_set_sha256": canonical_payload_sha256(source_hashes),
        "head_blob_hashes": head_blobs,
        "worktree_blob_hashes": worktree_blobs,
        "sources_match_head": head_blobs == worktree_blobs,
        "worktree_clean": status == "",
    }


def _verify_source_and_method_identity(
    artifact: Mapping[str, object],
    *,
    repo_root: Path,
) -> None:
    if artifact["schema"] != PRODUCTION_SCHEMA:
        _fail("production artifact schema identity differs")
    if artifact["schema_version"] != PRODUCTION_SCHEMA_VERSION:
        _fail("production artifact schema version differs")
    declarations = _require_exact_keys(
        artifact["declarations"],
        name="declarations",
        keys=(
            "stage",
            "synthetic_only",
            "quality_data_accessed",
            "protected_mbpp_window_accessed",
            "pretrained_checkpoint_loaded",
            "random_model_parameters_initialized",
            "tokenizer_loaded",
        ),
    )
    expected_declarations = {
        "stage": "stage0",
        "synthetic_only": True,
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
        "pretrained_checkpoint_loaded": False,
        "random_model_parameters_initialized": True,
        "tokenizer_loaded": False,
    }
    if dict(declarations) != expected_declarations:
        _fail("artifact declarations do not prove a synthetic-only Stage 0")

    source_identity = _require_exact_keys(
        artifact["source_identity"],
        name="source_identity",
        keys=(
            "repo_head",
            "repository_binding",
            "source_hashes",
            "source_set_sha256",
            "head_blob_hashes",
            "worktree_blob_hashes",
            "sources_match_head",
            "worktree_clean",
            "capture_start",
            "capture_end",
            "capture_start_equals_end",
            "loaded_local_source_paths",
            "loaded_recurquant_module_paths",
        ),
    )
    capture_start = _recorded_source_snapshot(
        source_identity["capture_start"],
        name="source_identity.capture_start",
    )
    capture_end = _recorded_source_snapshot(
        source_identity["capture_end"],
        name="source_identity.capture_end",
    )
    if capture_start != capture_end or source_identity["capture_start_equals_end"] is not True:
        _fail("authenticated capture start/end source snapshots differ")
    top_level_snapshot = {
        "repo_head": source_identity["repo_head"],
        "repository_binding": source_identity["repository_binding"],
        "source_hashes": source_identity["source_hashes"],
        "source_set_sha256": source_identity["source_set_sha256"],
        "head_blob_hashes": source_identity["head_blob_hashes"],
        "worktree_blob_hashes": source_identity["worktree_blob_hashes"],
        "sources_match_head": source_identity["sources_match_head"],
        "worktree_clean": source_identity["worktree_clean"],
    }
    if top_level_snapshot != capture_start:
        _fail("top-level source identity does not equal the recorded capture start")
    loaded_module_paths = source_identity["loaded_recurquant_module_paths"]
    if (
        type(loaded_module_paths) is not dict
        or any(type(key) is not str for key in loaded_module_paths)
        or any(type(value) is not str for value in loaded_module_paths.values())
    ):
        _fail("loaded RecurQuant module-name/source-path closure is not a plain mapping")
    required_module_paths = dict(REQUIRED_LOADED_RECURQUANT_MODULE_PATHS)
    missing_required = sorted(set(required_module_paths) - set(loaded_module_paths))
    mismatched_required = sorted(
        module_name
        for module_name in set(required_module_paths) & set(loaded_module_paths)
        if loaded_module_paths[module_name] != required_module_paths[module_name]
    )
    unauthenticated = sorted(
        module_name
        for module_name, relative in loaded_module_paths.items()
        if relative not in SOURCE_IDENTITY_PATHS
        or relative
        not in {
            f"src/{module_name.replace('.', '/')}.py",
            f"src/{module_name.replace('.', '/')}/__init__.py",
        }
        or (module_name != "recurquant" and not module_name.startswith("recurquant."))
    )
    if missing_required or mismatched_required or unauthenticated:
        _fail("loaded RecurQuant module-name/source-path closure differs")
    expected_loaded_paths = sorted(
        {"scripts/capture_statelease_stage0.py", *loaded_module_paths.values()}
    )
    loaded_paths = source_identity["loaded_local_source_paths"]
    if (
        not isinstance(loaded_paths, list)
        or any(not isinstance(path, str) for path in loaded_paths)
        or loaded_paths != expected_loaded_paths
        or not set(loaded_paths).issubset(SOURCE_IDENTITY_PATHS)
    ):
        _fail("loaded local production-source closure is invalid")
    current_snapshot = _current_repository_source_snapshot(repo_root)
    if current_snapshot["repo_head"] != capture_end["repo_head"]:
        _fail("artifact repository HEAD differs from current HEAD")
    if current_snapshot["worktree_clean"] is not True:
        _fail("current complete repository worktree is not clean")
    if current_snapshot["sources_match_head"] is not True:
        _fail("current authenticated source bytes differ from their HEAD blobs")
    if current_snapshot["source_hashes"] != capture_end["source_hashes"]:
        changed = [
            relative
            for relative in SOURCE_IDENTITY_PATHS
            if current_snapshot["source_hashes"][relative] != capture_end["source_hashes"][relative]
        ]
        _fail(f"authenticated production/verifier source changed: {changed}")
    if current_snapshot != capture_end:
        _fail("current repository/source identity differs from capture end")

    method = _require_exact_keys(
        artifact["method_identity"],
        name="method_identity",
        keys=(
            "method",
            "selection_method",
            "effective_plan_sha256",
            "seed",
            "replay_capacity",
            "boundary_rule",
            "key_normalization",
            "checkpoint_codec",
            "linear_layer_indices",
            "layer_quotas",
        ),
    )
    expected_method = {
        "method": "StateLease-H5",
        "selection_method": STATELEASE_SELECTION_METHOD,
        "effective_plan_sha256": EFFECTIVE_PLAN_SHA256,
        "seed": FROZEN_SEED,
        "replay_capacity": REPLAY_CAPACITY,
        "boundary_rule": "strictly_lower_c4_else_c5",
        "key_normalization": "consumed_dtype_then_fp32",
        "checkpoint_codec": "right_rht_sha256_signs_v1_q4_q8",
        "linear_layer_indices": list(LINEAR_LAYER_INDICES),
        "layer_quotas": {str(index): quota for index, quota in LAYER_QUOTAS.items()},
    }
    if dict(method) != expected_method:
        _fail("method or frozen effective-plan summary differs")


def _runtime_package_manifest() -> tuple[dict[str, str], str]:
    packages = {
        distribution: str(importlib.metadata.version(distribution))
        for distribution in RUNTIME_PACKAGE_DISTRIBUTIONS
    }
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")) + "\n"
    manifest_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if manifest_sha256 != PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256:
        _fail("current runtime package manifest differs from the Experiment 012 identity")
    return packages, manifest_sha256


def _verify_runtime_identity(value: object) -> dict[str, object]:
    runtime = _require_exact_keys(
        value,
        name="runtime_identity",
        keys=(
            "python_version",
            "python_implementation",
            "python_executable",
            "python_environment",
            "datasets_version",
            "fsspec_version",
            "huggingface_hub_version",
            "numpy_version",
            "pyarrow_version",
            "safetensors_version",
            "tokenizers_version",
            "torch_version",
            "transformers_version",
            "package_manifest_sha256",
            "platform",
            "system",
            "machine",
            "cuda_version",
            "cuda_available",
            "kernel_receipt_device",
            "default_dtype",
        ),
    )
    packages, package_manifest_sha256 = _runtime_package_manifest()
    expected = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "python_environment": Path(sys.prefix).name,
        "datasets_version": packages["datasets"],
        "fsspec_version": packages["fsspec"],
        "huggingface_hub_version": packages["huggingface-hub"],
        "numpy_version": packages["numpy"],
        "pyarrow_version": packages["pyarrow"],
        "safetensors_version": packages["safetensors"],
        "tokenizers_version": packages["tokenizers"],
        "torch_version": packages["torch"],
        "transformers_version": packages["transformers"],
        "package_manifest_sha256": package_manifest_sha256,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "kernel_receipt_device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
    }
    if dict(runtime) != expected:
        changed = [field for field in expected if runtime[field] != expected[field]]
        _fail(f"runtime identity differs: {changed}")
    return expected


def _assert_tensor_close(
    actual: object,
    expected: torch.Tensor,
    *,
    name: str,
    rtol: float = 2e-6,
    atol: float = 2e-6,
    exact: bool = False,
) -> torch.Tensor:
    tensor = _require_tensor(actual, name=name)
    if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
        _fail(
            f"{name} shape/dtype differs: "
            f"{tuple(tensor.shape)} {tensor.dtype} != "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    if exact:
        matches = torch.equal(tensor, expected)
    elif tensor.is_floating_point():
        matches = torch.allclose(tensor, expected, rtol=rtol, atol=atol)
    else:
        matches = torch.equal(tensor, expected)
    if not matches:
        maximum = (
            float((tensor.to(torch.float64) - expected.to(torch.float64)).abs().max().item())
            if tensor.numel()
            else 0.0
        )
        _fail(f"{name} differs; max_abs={maximum:.9g}")
    return tensor


def independent_query_ema(
    previous: torch.Tensor | None,
    query: torch.Tensor,
    *,
    expected_heads: int = HEADS,
    expected_rows: int = ROWS,
) -> torch.Tensor:
    query = _require_tensor(query, name="query", ndim=4).to(torch.float32)
    if query.shape[0] != 1 or query.shape[2:] != (
        expected_heads,
        expected_rows,
    ):
        raise ValueError("query has incompatible frozen geometry")
    squared = query.square()
    energy = squared / (squared.sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
    energy = energy.squeeze(0)
    prior = (
        torch.full(
            (expected_heads, expected_rows),
            1.0 / expected_rows,
            dtype=torch.float32,
        )
        if previous is None
        else _require_tensor(
            previous,
            name="previous_query_ema",
            dtype=torch.float32,
            ndim=2,
        )
    )
    decay = 0.5 ** (1.0 / 32.0)
    token_count = energy.shape[0]
    exponents = torch.arange(token_count - 1, -1, -1, dtype=torch.float32)
    weights = torch.pow(torch.tensor(decay, dtype=torch.float32), exponents)
    return (
        decay**token_count * prior + (1.0 - decay) * (energy * weights[:, None, None]).sum(dim=0)
    ).contiguous()


def _independent_q48_checkpoint(
    state: torch.Tensor,
    query_ema: torch.Tensor,
    *,
    layer_index: int,
    quota: int,
) -> tuple[PhysicalQ4Q8, torch.Tensor]:
    encoded = independent_rht_encode(state, layer_index=layer_index)
    rows = encoded.reshape(-1, encoded.shape[-1])
    benefit = q4_q8_physical_benefit(rows, query_ema.reshape(-1))
    selected = stable_descending_indices(benefit, quota)
    mask = torch.zeros(rows.shape[0], dtype=torch.bool)
    mask[selected] = True
    packed = pack_physical_q4_q8(rows, mask)
    restored = packed.dequantize().reshape_as(encoded)
    return packed, independent_rht_decode(restored, layer_index=layer_index)


def _record_from_artifact(value: object, *, name: str) -> ReplayRecord | None:
    record = _require_exact_keys(
        value,
        name=name,
        keys=("present", "normalized_key", "update", "log_decay"),
    )
    present = record["present"]
    if not isinstance(present, bool):
        _fail(f"{name}.present must be bool")
    if not present:
        for field in ("normalized_key", "update", "log_decay"):
            tensor = _require_tensor(record[field], name=f"{name}.{field}")
            if tensor.numel() != 0:
                _fail(f"{name}.{field} must be empty when record is absent")
        return None
    key = _require_tensor(
        record["normalized_key"],
        name=f"{name}.normalized_key",
        dtype=torch.float32,
        ndim=2,
    )
    update = _require_tensor(
        record["update"],
        name=f"{name}.update",
        dtype=torch.float32,
        ndim=2,
    )
    decay = _require_tensor(
        record["log_decay"],
        name=f"{name}.log_decay",
        dtype=torch.float32,
        ndim=1,
    )
    return ReplayRecord(
        normalized_key=key.unsqueeze(0),
        update=update.unsqueeze(0),
        log_decay=decay.unsqueeze(0),
    )


def _verify_trace_buffers(
    value: object,
    expected: ReplayBuffers,
    *,
    name: str,
) -> None:
    buffers = _require_exact_keys(
        value,
        name=name,
        keys=("normalized_keys", "updates", "log_decays", "valid_count"),
    )
    _assert_tensor_close(
        buffers["normalized_keys"],
        expected.normalized_keys,
        name=f"{name}.normalized_keys",
        exact=True,
    )
    _assert_tensor_close(
        buffers["updates"],
        expected.updates,
        name=f"{name}.updates",
        exact=True,
    )
    _assert_tensor_close(
        buffers["log_decays"],
        expected.log_decays,
        name=f"{name}.log_decays",
        exact=True,
    )
    _assert_tensor_close(
        buffers["valid_count"],
        expected.valid_count,
        name=f"{name}.valid_count",
        exact=True,
    )


def _empty_full_buffers() -> ReplayBuffers:
    return ReplayBuffers(
        normalized_keys=torch.zeros(
            (REPLAY_CAPACITY, HEADS, ROWS),
            dtype=torch.bfloat16,
        ),
        updates=torch.zeros(
            (REPLAY_CAPACITY, HEADS, WIDTH),
            dtype=torch.bfloat16,
        ),
        log_decays=torch.zeros(
            (REPLAY_CAPACITY, HEADS),
            dtype=torch.float32,
        ),
        valid_count=torch.zeros((1,), dtype=torch.int32),
    )


def _append_expected_record(
    buffers: ReplayBuffers,
    record: ReplayRecord,
) -> ReplayBuffers:
    count = int(buffers.valid_count.item())
    if count >= REPLAY_CAPACITY:
        raise ValueError("cannot append to a full expected buffer")
    keys = buffers.normalized_keys.clone()
    updates = buffers.updates.clone()
    decays = buffers.log_decays.clone()
    keys[count].copy_(record.normalized_key[0].to(torch.bfloat16))
    updates[count].copy_(record.update[0].to(torch.bfloat16))
    decays[count].copy_(record.log_decay[0].to(torch.float32))
    return ReplayBuffers(
        normalized_keys=keys,
        updates=updates,
        log_decays=decays,
        valid_count=torch.tensor([count + 1], dtype=torch.int32),
    )


def _dense_successful_transition(
    initial_state: torch.Tensor,
    *,
    consumed_key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    state = _require_tensor(initial_state, name="initial_state", ndim=4).to(torch.float32)
    key = _require_tensor(consumed_key, name="consumed_key", ndim=4)
    values = _require_tensor(value, name="value", ndim=4).to(torch.float32)
    decays = _require_tensor(log_decay, name="log_decay", ndim=3).to(torch.float32)
    writes = _require_tensor(beta, name="beta", ndim=3).to(torch.float32)
    if (
        key.shape[:3] != values.shape[:3]
        or key.shape[:3] != decays.shape
        or key.shape[:3] != writes.shape
        or key.shape[0] != 1
        or key.shape[2] != state.shape[1]
        or key.shape[3] != state.shape[2]
        or values.shape[3] != state.shape[3]
    ):
        raise ValueError("successful transition tensors have incompatible geometry")
    normalized = normalize_consumed_key(key)
    for token_index in range(key.shape[1]):
        decay = decays[:, token_index]
        key_token = normalized[:, token_index]
        decayed = state * decay.exp().unsqueeze(-1).unsqueeze(-1)
        remembered = (decayed * key_token.unsqueeze(-1)).sum(dim=-2)
        update = (values[:, token_index] - remembered) * writes[:, token_index].unsqueeze(-1)
        state = decayed + key_token.unsqueeze(-1) * update.unsqueeze(-2)
    return state


def _verify_successful_kernel_receipt(value: object) -> dict[str, object]:
    receipt = _require_exact_keys(
        value,
        name="successful_kernel_receipt",
        keys=(
            "observer",
            "model",
            "pretrained_checkpoint_loaded",
            "synthetic_token_ids_only",
            "successful_model_forwards",
            "successful_kernel_calls",
            "receipts",
        ),
    )
    expected_metadata = {
        "observer": "Qwen35StateLeaseObserver",
        "model": "randomly_initialized_tiny_Qwen3_5ForCausalLM",
        "pretrained_checkpoint_loaded": False,
        "synthetic_token_ids_only": True,
        "successful_model_forwards": 2,
        "successful_kernel_calls": 2,
    }
    for field, expected in expected_metadata.items():
        if receipt[field] != expected:
            _fail(f"successful-kernel receipt differs at {field}")
    calls = receipt["receipts"]
    if not isinstance(calls, list) or len(calls) != 2:
        _fail("successful-kernel receipt must contain exactly two calls")
    query_ema: torch.Tensor | None = None
    checkpoint: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    expected_buffers = ReplayBuffers(
        normalized_keys=torch.zeros((5, 2, 8), dtype=torch.bfloat16),
        updates=torch.zeros((5, 2, 8), dtype=torch.bfloat16),
        log_decays=torch.zeros((5, 2), dtype=torch.float32),
        valid_count=torch.zeros((1,), dtype=torch.int32),
    )
    for call_index, item in enumerate(calls):
        call = _require_exact_keys(
            item,
            name=f"successful_kernel_receipt.receipts[{call_index}]",
            keys=(
                "layer_index",
                "query",
                "consumed_key",
                "value",
                "log_decay",
                "beta",
                "initial_state_present",
                "initial_state",
                "successful_final_state",
                "call_index",
                "input_ids",
                "logits_shape",
                "logits_finite",
                "production_query_ema",
                "production_materialized_state",
                "production_buffers",
                "production_evidence",
            ),
        )
        if (
            call["layer_index"] != 0
            or call["call_index"] != call_index
            or call["logits_finite"] is not True
            or call["logits_shape"] != [1, 3 if call_index == 0 else 1, 128]
        ):
            _fail("successful-kernel call metadata differs")
        input_ids = _require_tensor(
            call["input_ids"],
            name="successful_kernel.input_ids",
            dtype=torch.long,
            ndim=2,
            finite=False,
        )
        expected_ids = (
            torch.tensor([[7, 11, 13]], dtype=torch.long)
            if call_index == 0
            else torch.tensor([[17]], dtype=torch.long)
        )
        _assert_tensor_close(
            input_ids,
            expected_ids,
            name="successful-kernel synthetic token IDs",
            exact=True,
        )
        initial_present = call["initial_state_present"]
        if initial_present is not (call_index == 1):
            _fail("successful-kernel initial-state presence differs")
        raw_initial = _require_tensor(
            call["initial_state"],
            name="successful_kernel.initial_state",
        )
        if call_index == 0:
            if raw_initial.numel():
                _fail("successful prefill kernel must have no initial state")
            initial = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
        else:
            if previous is None:
                _fail("decode kernel has no preceding resident state")
            _assert_tensor_close(
                raw_initial,
                previous,
                name="successful decode kernel initial state",
                exact=True,
            )
            initial = raw_initial
        expected_final = _dense_successful_transition(
            initial,
            consumed_key=call["consumed_key"],
            value=call["value"],
            log_decay=call["log_decay"],
            beta=call["beta"],
        )
        final = _assert_tensor_close(
            call["successful_final_state"],
            expected_final,
            name="successful observed kernel final state",
            rtol=3e-6,
            atol=2e-6,
        )
        query_ema = independent_query_ema(
            query_ema,
            call["query"],
            expected_heads=2,
            expected_rows=8,
        )
        saved_ema = _assert_tensor_close(
            call["production_query_ema"],
            query_ema,
            name="successful-kernel production query EMA",
            exact=True,
        )
        evidence = call["production_evidence"]
        if not isinstance(evidence, Mapping):
            _fail("successful-kernel production evidence must be a mapping")
        if call_index == 0:
            _, checkpoint = _independent_q48_checkpoint(
                final,
                saved_ema,
                layer_index=0,
                quota=5,
            )
            expected_materialized = checkpoint
            expected_action = "checkpoint_prefill"
        else:
            if checkpoint is None:
                _fail("successful decode receipt lacks its prefill checkpoint")
            record = derive_successful_record(
                initial_state=initial,
                consumed_key=call["consumed_key"],
                value=call["value"],
                log_decay=call["log_decay"],
                beta=call["beta"],
                successful_final_state=final,
            )
            expected_buffers = _append_expected_record(expected_buffers, record)
            expected_materialized = replay_stored_buffers(
                checkpoint,
                expected_buffers,
            )
            expected_action = "replay_append"
        if evidence.get("action") != expected_action:
            _fail("successful-kernel production action differs")
        _assert_tensor_close(
            call["production_materialized_state"],
            expected_materialized,
            name="successful-kernel resident state",
            rtol=4e-6,
            atol=3e-6,
        )
        _verify_trace_buffers(
            call["production_buffers"],
            expected_buffers,
            name=f"successful_kernel.buffers[{call_index}]",
        )
        previous = _require_tensor(
            call["production_materialized_state"],
            name="successful-kernel materialized state",
        )
    return {
        "observer": expected_metadata["observer"],
        "successful_model_forwards": 2,
        "successful_kernel_calls": 2,
        "consumed_record_verified": True,
    }


def _verify_production_trace(value: object) -> dict[str, object]:
    if not isinstance(value, list) or len(value) != 6:
        _fail("production trace must contain exactly six synthetic writes")
    query_ema: torch.Tensor | None = None
    checkpoint: torch.Tensor | None = None
    buffers = _empty_full_buffers()
    previous_materialized: torch.Tensor | None = None
    boundary_counts = {4: 0, 5: 0}
    trace_hashes: dict[str, str] = {}
    for step_index, item in enumerate(value):
        step = _require_exact_keys(
            item,
            name=f"production_trace[{step_index}]",
            keys=(
                "identity",
                "step_index",
                "signals",
                "production_record",
                "production_query_ema",
                "production_materialized_state",
                "production_buffers",
                "production_evidence",
            ),
        )
        identity = f"statelease_trace_step_{step_index}"
        if step["identity"] != identity or step["step_index"] != step_index:
            _fail("production trace identities or ordering differ")
        trace_hashes[identity] = canonical_payload_sha256(step)
        signals = _require_exact_keys(
            step["signals"],
            name=f"trace[{step_index}].signals",
            keys=(
                "query",
                "consumed_key",
                "value",
                "log_decay",
                "beta",
                "initial_state",
                "successful_final_state",
            ),
        )
        initial = _require_tensor(
            signals["initial_state"],
            name="initial_state",
            dtype=torch.float32,
            ndim=4,
        )
        if initial.shape != (1, HEADS, ROWS, WIDTH):
            _fail("trace initial state has wrong frozen geometry")
        if step_index == 0:
            if torch.count_nonzero(initial).item():
                _fail("trace prefill initial state must be zero")
        elif previous_materialized is None or not torch.equal(initial, previous_materialized):
            _fail("trace does not consume the preceding resident state")
        final = _require_tensor(
            signals["successful_final_state"],
            name="successful_final_state",
            dtype=torch.float32,
            ndim=4,
        )
        derived = derive_successful_record(
            initial_state=initial,
            consumed_key=signals["consumed_key"],
            value=signals["value"],
            log_decay=signals["log_decay"],
            beta=signals["beta"],
            successful_final_state=final,
        )
        production_record = _record_from_artifact(
            step["production_record"],
            name=f"trace[{step_index}].production_record",
        )
        if step_index == 0:
            if production_record is not None:
                _fail("prefill unexpectedly stored a replay record")
        else:
            if production_record is None:
                _fail("decode write did not expose its production record")
            _assert_tensor_close(
                production_record.normalized_key,
                derived.normalized_key,
                name="production consumed normalized key",
                exact=True,
            )
            _assert_tensor_close(
                production_record.update,
                derived.update,
                name="production post-correction update",
                exact=True,
            )
            _assert_tensor_close(
                production_record.log_decay,
                derived.log_decay,
                name="production log decay",
                exact=True,
            )

        query_ema = independent_query_ema(query_ema, signals["query"])
        saved_ema = _assert_tensor_close(
            step["production_query_ema"],
            query_ema,
            name=f"trace[{step_index}].query_ema",
            exact=True,
        )
        evidence = step["production_evidence"]
        if not isinstance(evidence, Mapping):
            _fail("production trace evidence must be a mapping")
        if evidence.get("update_index") != step_index:
            _fail("production evidence update index differs")

        if step_index == 0:
            _, checkpoint = _independent_q48_checkpoint(
                final,
                saved_ema,
                layer_index=0,
                quota=LAYER_QUOTAS[0],
            )
            buffers = _empty_full_buffers()
            expected_materialized = checkpoint
            expected_action = "checkpoint_prefill"
            expected_boundary = None
        elif step_index < 5:
            assert production_record is not None
            buffers = _append_expected_record(buffers, production_record)
            assert checkpoint is not None
            expected_materialized = replay_stored_buffers(checkpoint, buffers)
            expected_action = "replay_append"
            expected_boundary = None
        else:
            assert production_record is not None
            full = _append_expected_record(buffers, production_record)
            if int(full.valid_count.item()) != REPLAY_CAPACITY:
                _fail("fifth trace write did not exercise explicit slot four/count five")
            _, cut4_checkpoint = _independent_q48_checkpoint(
                initial,
                saved_ema,
                layer_index=0,
                quota=LAYER_QUOTAS[0],
            )
            cut4_buffers = compact_full_buffer(full, boundary=4)
            cut4_materialized = replay_stored_buffers(
                cut4_checkpoint,
                cut4_buffers,
            )
            _, cut5_materialized = _independent_q48_checkpoint(
                final,
                saved_ema,
                layer_index=0,
                quota=LAYER_QUOTAS[0],
            )
            weights = normalized_query_weights(saved_ema)
            cut4_risk = float(
                query_weighted_handoff_risk(
                    final,
                    cut4_materialized,
                    weights,
                ).item()
            )
            cut5_risk = float(
                query_weighted_handoff_risk(
                    final,
                    cut5_materialized,
                    weights,
                ).item()
            )
            expected_boundary = 4 if cut4_risk < cut5_risk else 5
            expected_action = f"boundary_{expected_boundary}"
            if expected_boundary == 4:
                checkpoint = cut4_checkpoint
                buffers = cut4_buffers
                expected_materialized = cut4_materialized
            else:
                checkpoint = cut5_materialized
                buffers = compact_full_buffer(full, boundary=5)
                expected_materialized = cut5_materialized
            boundary_counts[expected_boundary] += 1
            if not math.isclose(
                float(evidence.get("cut4_risk")),
                cut4_risk,
                rel_tol=2e-6,
                abs_tol=2e-7,
            ):
                _fail("production c4 handoff risk differs from dense verifier")
            if not math.isclose(
                float(evidence.get("cut5_risk")),
                cut5_risk,
                rel_tol=2e-6,
                abs_tol=2e-7,
            ):
                _fail("production c5 handoff risk differs from dense verifier")
            if bool(evidence.get("tie")) != (cut4_risk == cut5_risk):
                _fail("production tie receipt differs")
        if evidence.get("action") != expected_action:
            _fail("production trace action differs from independent state machine")
        if evidence.get("boundary") != expected_boundary:
            _fail("production boundary differs from strict c4/c5 rule")
        _assert_tensor_close(
            step["production_materialized_state"],
            expected_materialized,
            name=f"trace[{step_index}].materialized",
            rtol=4e-6,
            atol=3e-6,
        )
        _verify_trace_buffers(
            step["production_buffers"],
            buffers,
            name=f"trace[{step_index}].buffers",
        )
        previous_materialized = _require_tensor(
            step["production_materialized_state"],
            name="production_materialized_state",
        )
    if sum(boundary_counts.values()) != 1:
        _fail("production trace did not contain exactly one c4/c5 boundary")
    return {
        "trace_hashes": trace_hashes,
        "boundary4_count": boundary_counts[4],
        "boundary5_count": boundary_counts[5],
    }


def _resident_tensor_map(snapshot: ResidentSnapshot) -> dict[str, torch.Tensor]:
    if not all(
        len(getattr(snapshot, field)) == LAYERS
        for field in (
            "checkpoint_low_payloads",
            "checkpoint_high_payloads",
            "checkpoint_scales",
            "checkpoint_masks",
            "query_emas",
            "normalized_key_buffers",
            "update_buffers",
            "log_decay_buffers",
            "valid_counts",
        )
    ):
        _fail("resident snapshot must expose one tensor per recurrent layer")
    result: dict[str, torch.Tensor] = {}
    for position, layer_index in enumerate(LINEAR_LAYER_INDICES):
        result.update(
            {
                f"layer_{layer_index}.checkpoint.low_payload": (
                    snapshot.checkpoint_low_payloads[position]
                ),
                f"layer_{layer_index}.checkpoint.high_payload": (
                    snapshot.checkpoint_high_payloads[position]
                ),
                f"layer_{layer_index}.checkpoint.scales": (snapshot.checkpoint_scales[position]),
                f"layer_{layer_index}.checkpoint.precision_mask": (
                    snapshot.checkpoint_masks[position]
                ),
                f"layer_{layer_index}.query_energy_ema": snapshot.query_emas[position],
                f"layer_{layer_index}.normalized_key_buffer": (
                    snapshot.normalized_key_buffers[position]
                ),
                f"layer_{layer_index}.update_buffer": snapshot.update_buffers[position],
                f"layer_{layer_index}.log_decay_buffer": (snapshot.log_decay_buffers[position]),
                f"layer_{layer_index}.valid_count": snapshot.valid_counts[position],
            }
        )
    return result


def _candidate_graph_path(name: str) -> str:
    checkpoint = re.fullmatch(
        r"layer_(\d+)\.checkpoint\.(low_payload|high_payload|scales|precision_mask)",
        name,
    )
    if checkpoint is not None:
        return f"cache.layers[{int(checkpoint.group(1))}].packed_states[0].{checkpoint.group(2)}"
    direct = re.fullmatch(
        r"layer_(\d+)\.(query_energy_ema|normalized_key_buffer|update_buffer|"
        r"log_decay_buffer|valid_count)",
        name,
    )
    if direct is None:
        _fail(f"unknown StateLease persistent tensor name: {name}")
    return f"cache.layers[{int(direct.group(1))}].{direct.group(2)}"


def _inventory_tensor_view(path: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "path": path,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
        "storage_offset": tensor.storage_offset(),
        "sha256": _tensor_digest(tensor),
    }


def _expected_whole_cache_storage_audit(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    pending = [
        {
            "classification": "statelease_candidate",
            "allowed_tensor_name": name,
            "storage_nbytes": tensor.untyped_storage().nbytes(),
            "views": [_inventory_tensor_view(_candidate_graph_path(name), tensor)],
        }
        for name, tensor in tensors.items()
    ]
    pending.sort(key=lambda entry: entry["views"][0]["path"])
    inventory = [{"storage_index": index, **entry} for index, entry in enumerate(pending)]
    candidate_bytes = sum(int(entry["storage_nbytes"]) for entry in inventory)
    return {
        "inventory": inventory,
        "candidate_storage_count": len(inventory),
        "candidate_unique_storage_bytes": candidate_bytes,
        "shared_kv_unique_storage_bytes": 0,
        "shared_conv_unique_storage_bytes": 0,
        "unexplained_unique_storage_bytes": 0,
        "total_unique_storage_bytes": candidate_bytes,
        "unexplained_tensor_elements": 0,
        "raw_state_elements_per_layer": RAW_STATE_ELEMENTS_PER_LAYER,
        "raw_state_elements_all_layers": RAW_STATE_ELEMENTS_ALL_LAYERS,
        "unexplained_raw_state_equivalent_layer_floor": 0,
        "all_persistent_storage_classified": True,
        "no_unexplained_persistent_storage": True,
        "storage_deduplicated": True,
    }


def _verify_whole_cache_storage_audit(
    value: object,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    expected_tensor_names = {
        name
        for layer_index in LINEAR_LAYER_INDICES
        for name in (
            f"layer_{layer_index}.checkpoint.low_payload",
            f"layer_{layer_index}.checkpoint.high_payload",
            f"layer_{layer_index}.checkpoint.scales",
            f"layer_{layer_index}.checkpoint.precision_mask",
            f"layer_{layer_index}.query_energy_ema",
            f"layer_{layer_index}.normalized_key_buffer",
            f"layer_{layer_index}.update_buffer",
            f"layer_{layer_index}.log_decay_buffer",
            f"layer_{layer_index}.valid_count",
        )
    }
    if set(tensors) != expected_tensor_names:
        _fail("whole-cache audit did not receive the exact frozen candidate tensor schema")
    audit = _require_exact_keys(
        value,
        name="resident_snapshot.whole_cache_storage_audit",
        keys=(
            "inventory",
            "candidate_storage_count",
            "candidate_unique_storage_bytes",
            "shared_kv_unique_storage_bytes",
            "shared_conv_unique_storage_bytes",
            "unexplained_unique_storage_bytes",
            "total_unique_storage_bytes",
            "unexplained_tensor_elements",
            "raw_state_elements_per_layer",
            "raw_state_elements_all_layers",
            "unexplained_raw_state_equivalent_layer_floor",
            "all_persistent_storage_classified",
            "no_unexplained_persistent_storage",
            "storage_deduplicated",
        ),
    )
    expected = _expected_whole_cache_storage_audit(tensors)
    if dict(audit) != expected:
        _fail(
            "whole-cache all-dtype storage inventory does not reconcile with "
            "the closed StateLease tensor schema"
        )
    if expected["candidate_unique_storage_bytes"] != STATELEASE_BYTES:
        _fail("whole-cache candidate storage does not equal the frozen StateLease budget")
    return expected


def _verify_tensor_enumeration(
    value: object,
    tensors: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(tensors):
        _fail(f"{name} length differs from physical tensors")
    expected_names = sorted(tensors)
    actual_names: list[str] = []
    for index, item in enumerate(value):
        entry = _require_exact_keys(
            item,
            name=f"{name}[{index}]",
            keys=("name", "dtype", "shape", "nbytes", "sha256"),
        )
        tensor_name = entry["name"]
        if not isinstance(tensor_name, str) or tensor_name not in tensors:
            _fail(f"{name} contains an unknown tensor name")
        tensor = tensors[tensor_name]
        expected = {
            "name": tensor_name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "nbytes": tensor.numel() * tensor.element_size(),
            "sha256": _tensor_digest(tensor),
        }
        if dict(entry) != expected:
            _fail(f"{name} metadata differs for {tensor_name}")
        actual_names.append(tensor_name)
    if actual_names != expected_names:
        _fail(f"{name} is not a complete canonical tensor enumeration")


def _verify_resident_snapshot(value: object) -> dict[str, object]:
    mapping = _require_exact_keys(
        value,
        name="resident_snapshot",
        keys=(
            "checkpoint_low_payloads",
            "checkpoint_high_payloads",
            "checkpoint_scales",
            "checkpoint_masks",
            "query_emas",
            "normalized_key_buffers",
            "update_buffers",
            "log_decay_buffers",
            "valid_counts",
            "persistent_enumeration",
            "whole_cache_storage_audit",
            "no_hidden_persistent_state_mirror",
            "storage_summary",
        ),
    )
    tensor_fields = (
        "checkpoint_low_payloads",
        "checkpoint_high_payloads",
        "checkpoint_scales",
        "checkpoint_masks",
        "query_emas",
        "normalized_key_buffers",
        "update_buffers",
        "log_decay_buffers",
        "valid_counts",
    )
    for field in tensor_fields:
        if not isinstance(mapping[field], list) or any(
            not isinstance(tensor, torch.Tensor) for tensor in mapping[field]
        ):
            _fail(f"resident_snapshot.{field} must be a tensor list")
    snapshot = ResidentSnapshot(
        checkpoint_low_payloads=tuple(mapping["checkpoint_low_payloads"]),
        checkpoint_high_payloads=tuple(mapping["checkpoint_high_payloads"]),
        checkpoint_scales=tuple(mapping["checkpoint_scales"]),
        checkpoint_masks=tuple(mapping["checkpoint_masks"]),
        query_emas=tuple(mapping["query_emas"]),
        normalized_key_buffers=tuple(mapping["normalized_key_buffers"]),
        update_buffers=tuple(mapping["update_buffers"]),
        log_decay_buffers=tuple(mapping["log_decay_buffers"]),
        valid_counts=tuple(mapping["valid_counts"]),
    )
    storage = audit_resident_snapshot(snapshot)
    tensors = _resident_tensor_map(snapshot)
    _verify_tensor_enumeration(
        mapping["persistent_enumeration"],
        tensors,
        name="resident_snapshot.persistent_enumeration",
    )
    whole_cache_audit = _verify_whole_cache_storage_audit(
        mapping["whole_cache_storage_audit"],
        tensors,
    )
    if mapping["no_hidden_persistent_state_mirror"] is not True:
        _fail("producer did not declare absence of an all-dtype persistent state mirror")
    illegal_fp32 = [
        name
        for name, tensor in tensors.items()
        if tensor.dtype == torch.float32
        and ".query_energy_ema" not in name
        and ".log_decay_buffer" not in name
    ]
    if illegal_fp32:
        _fail(f"resident snapshot contains hidden FP32 tensors: {illegal_fp32}")
    summary = mapping["storage_summary"]
    if not isinstance(summary, Mapping):
        _fail("resident storage summary must be a mapping")
    expected_summary = {
        "selection_method": STATELEASE_SELECTION_METHOD,
        "experiment_identity_sha256": EFFECTIVE_PLAN_SHA256,
        "payload_bytes": CHECKPOINT_PAYLOAD_BYTES,
        "scale_bytes": CHECKPOINT_SCALE_BYTES,
        "mask_bytes": CHECKPOINT_MASK_BYTES,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "resident_bytes": CHECKPOINT_BYTES,
        "query_ema_bytes": QUERY_EMA_BYTES,
        "replay_capacity_bytes": (
            KEY_BUFFER_BYTES + UPDATE_BUFFER_BYTES + LOG_DECAY_BUFFER_BYTES + COUNT_BYTES
        ),
        "resident_bytes_including_statelease": STATELEASE_BYTES,
        "high_precision_groups": HIGH_PRECISION_ROWS,
        "full_precision_equivalent_bytes": STATE_ELEMENTS * 4,
        "physical_reduction_realized": True,
        "physical_reduction_realized_including_statelease": True,
        "forward_transaction_active": False,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            _fail(f"resident storage summary differs at {field}")
    if int(summary.get("replay_occupied_bytes", -1)) < COUNT_BYTES:
        _fail("resident storage summary has invalid occupied replay bytes")
    return {
        "storage": storage,
        "resident_digest": resident_snapshot_digest(snapshot),
        "tensor_count": len(tensors),
        "whole_cache_storage_audit": {
            key: value for key, value in whole_cache_audit.items() if key != "inventory"
        },
    }


def _verify_lifecycle(
    value: object,
    *,
    trace_hashes: Mapping[str, str],
) -> dict[str, object]:
    lifecycle = _require_exact_keys(
        value,
        name="lifecycle",
        keys=("rollback", "reset", "resume"),
    )
    rollback = _require_exact_keys(
        lifecycle["rollback"],
        name="lifecycle.rollback",
        keys=(
            "before_sha256",
            "mutated_sha256",
            "after_sha256",
            "before_update_index",
            "after_update_index",
            "before_evidence_count",
            "after_evidence_count",
            "transaction_active_after_rollback",
        ),
    )
    for field in ("before_sha256", "mutated_sha256", "after_sha256"):
        digest = rollback[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail(f"rollback {field} is not SHA-256")
    if (
        rollback["before_sha256"] == rollback["mutated_sha256"]
        or rollback["before_sha256"] != rollback["after_sha256"]
        or rollback["before_update_index"] != rollback["after_update_index"]
        or rollback["before_evidence_count"] != rollback["after_evidence_count"]
        or rollback["transaction_active_after_rollback"] is not False
    ):
        _fail("production rollback receipt does not prove exact restoration")

    reset = _require_exact_keys(
        lifecycle["reset"],
        name="lifecycle.reset",
        keys=(
            "resident_bytes_before_reset",
            "resident_bytes_after_reset",
            "update_index_after_reset",
            "evidence_count_after_reset",
            "all_has_previous_state_flags_cleared",
            "all_pending_observations_cleared",
            "post_reset_snapshot",
        ),
    )
    expected_reset_scalars = {
        "resident_bytes_before_reset": STATELEASE_BYTES,
        "resident_bytes_after_reset": STATELEASE_BYTES,
        "update_index_after_reset": 0,
        "evidence_count_after_reset": 0,
        "all_has_previous_state_flags_cleared": True,
        "all_pending_observations_cleared": True,
    }
    for field, expected in expected_reset_scalars.items():
        if reset[field] != expected:
            _fail(f"production reset receipt differs at {field}")
    _verify_resident_snapshot(reset["post_reset_snapshot"])
    post_reset = reset["post_reset_snapshot"]
    assert isinstance(post_reset, Mapping)
    for field in (
        "checkpoint_low_payloads",
        "checkpoint_high_payloads",
        "checkpoint_scales",
        "normalized_key_buffers",
        "update_buffers",
        "log_decay_buffers",
        "valid_counts",
    ):
        tensors = post_reset[field]
        if not isinstance(tensors, list) or any(
            torch.count_nonzero(tensor).item() for tensor in tensors
        ):
            _fail(f"production reset left nonzero {field}")
    for tensor in post_reset["query_emas"]:
        expected = torch.full_like(tensor, 1.0 / ROWS)
        if not torch.equal(tensor, expected):
            _fail("production reset did not restore uniform query EMA")

    resume = _require_exact_keys(
        lifecycle["resume"],
        name="lifecycle.resume",
        keys=(
            "prior_identity_hashes",
            "resumed_identity_hashes",
            "expected_identities",
            "completed_record_hashes",
            "resumed_completed_record_hashes",
            "resumed_remaining_identities",
        ),
    )
    verify_resume_integrity(
        prior_identity_hashes=resume["prior_identity_hashes"],
        resumed_identity_hashes=resume["resumed_identity_hashes"],
        expected_identities=resume["expected_identities"],
        completed_record_hashes=resume["completed_record_hashes"],
        resumed_completed_record_hashes=resume["resumed_completed_record_hashes"],
        resumed_remaining_identities=resume["resumed_remaining_identities"],
    )
    if list(resume["expected_identities"]) != list(trace_hashes):
        _fail("resume manifest differs from production trace identities")
    for identity, digest in resume["completed_record_hashes"].items():
        if trace_hashes.get(identity) != digest:
            _fail("resume completed record hash differs from trace")
    return {
        "rollback_preserved": True,
        "reset_cleared": True,
        "resume_integrity": True,
    }


def _synthetic_trajectory_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    error = candidate.to(torch.float64) - reference.to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.to(torch.float64)).clamp_min(1e-12)
    return {
        "synthetic_state_mse": float(error.square().mean().item()),
        "synthetic_state_relative_l2": float(
            (torch.linalg.vector_norm(error) / denominator).item()
        ),
    }


def _verify_cc1_production_compatibility(value: object) -> dict[str, object]:
    compatibility = _require_exact_keys(
        value,
        name="cc1_compatibility",
        keys=(
            "transitions",
            "fixed_cc1_trajectory",
            "rht_cqer_trajectory",
            "fixed_cc1_masks",
            "rht_cqer_masks",
            "fixed_cc1_metrics",
            "rht_cqer_metrics",
            "row_plan",
            "hashes",
        ),
    )
    transitions = compatibility["transitions"]
    if not isinstance(transitions, list) or len(transitions) != 4:
        _fail("CC1 compatibility trace must contain four transitions")
    fixed = _require_tensor(
        compatibility["fixed_cc1_trajectory"],
        name="fixed_cc1_trajectory",
        dtype=torch.float32,
        ndim=5,
    )
    anchor = _require_tensor(
        compatibility["rht_cqer_trajectory"],
        name="rht_cqer_trajectory",
        dtype=torch.float32,
        ndim=5,
    )
    if fixed.shape != (4, 1, HEADS, ROWS, WIDTH) or anchor.shape != fixed.shape:
        _fail("CC1 compatibility trajectory has wrong frozen geometry")
    _assert_tensor_close(
        fixed,
        anchor,
        name="fixed CC1 versus RHT-CQER trajectory",
        exact=True,
    )
    fixed_masks = compatibility["fixed_cc1_masks"]
    anchor_masks = compatibility["rht_cqer_masks"]
    if (
        not isinstance(fixed_masks, list)
        or not isinstance(anchor_masks, list)
        or len(fixed_masks) != 4
        or len(anchor_masks) != 4
    ):
        _fail("CC1 compatibility masks must contain four writes")
    query_ema: torch.Tensor | None = None
    expected_states: list[torch.Tensor] = []
    expected_masks: list[torch.Tensor] = []
    previous: torch.Tensor | None = None
    references: list[torch.Tensor] = []
    for index, item in enumerate(transitions):
        signals = _require_exact_keys(
            item,
            name=f"cc1.transitions[{index}]",
            keys=(
                "query",
                "consumed_key",
                "value",
                "log_decay",
                "beta",
                "initial_state",
                "successful_final_state",
            ),
        )
        initial = _require_tensor(
            signals["initial_state"],
            name="cc1.initial_state",
            dtype=torch.float32,
            ndim=4,
        )
        if index == 0:
            if torch.count_nonzero(initial).item():
                _fail("CC1 initial prefill state is not zero")
        elif previous is None or not torch.equal(initial, previous):
            _fail("CC1 trace did not consume its preceding materialized state")
        final = _require_tensor(
            signals["successful_final_state"],
            name="cc1.successful_final_state",
            dtype=torch.float32,
            ndim=4,
        )
        derive_successful_record(
            initial_state=initial,
            consumed_key=signals["consumed_key"],
            value=signals["value"],
            log_decay=signals["log_decay"],
            beta=signals["beta"],
            successful_final_state=final,
        )
        query_ema = independent_query_ema(query_ema, signals["query"])
        packed, materialized = _independent_q48_checkpoint(
            final,
            query_ema,
            layer_index=0,
            quota=LAYER_QUOTAS[0],
        )
        expected_states.append(materialized)
        expected_masks.append(packed.precision_mask)
        previous = materialized
        references.append(final)
    expected = torch.stack(expected_states)
    _assert_tensor_close(
        fixed,
        expected,
        name="fixed CC1 independently reconstructed trajectory",
        rtol=4e-6,
        atol=3e-6,
    )
    for index, expected_mask in enumerate(expected_masks):
        _assert_tensor_close(
            fixed_masks[index],
            expected_mask,
            name=f"fixed CC1 mask {index}",
            exact=True,
        )
        _assert_tensor_close(
            anchor_masks[index],
            expected_mask,
            name=f"RHT-CQER mask {index}",
            exact=True,
        )
    reference = torch.stack(references)
    expected_metrics = _synthetic_trajectory_metrics(expected, reference)
    for field, expected_value in expected_metrics.items():
        for metric_name in ("fixed_cc1_metrics", "rht_cqer_metrics"):
            metrics = compatibility[metric_name]
            if not isinstance(metrics, Mapping) or set(metrics) != set(expected_metrics):
                _fail(f"{metric_name} violates the closed metric schema")
            if not math.isclose(
                float(metrics[field]),
                expected_value,
                rel_tol=2e-12,
                abs_tol=2e-12,
            ):
                _fail(f"{metric_name}.{field} differs")
    row_plan = compatibility["row_plan"]
    expected_plan = {
        "effective_plan_sha256": EFFECTIVE_PLAN_SHA256,
        "layer_0_quota": LAYER_QUOTAS[0],
    }
    if row_plan != expected_plan:
        _fail("CC1 compatibility row plan differs")
    hashes = _require_exact_keys(
        compatibility["hashes"],
        name="cc1.hashes",
        keys=(
            "fixed_trajectory",
            "rht_cqer_trajectory",
            "fixed_masks",
            "rht_cqer_masks",
            "row_plan",
        ),
    )
    expected_hashes = {
        "fixed_trajectory": _tensor_digest(fixed),
        "rht_cqer_trajectory": _tensor_digest(anchor),
        "fixed_masks": canonical_payload_sha256(fixed_masks),
        "rht_cqer_masks": canonical_payload_sha256(anchor_masks),
        "row_plan": canonical_payload_sha256(row_plan),
    }
    if dict(hashes) != expected_hashes:
        _fail("CC1 compatibility hashes differ")
    return {
        "trajectory_steps": 4,
        "exact_match": True,
        "metrics": expected_metrics,
    }


def _quantized_row_reconstruction(
    rows: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    codes, scales = _quantize_rows(rows, bits)
    return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)


def _independent_allocate_multibit_fast(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Exact O(N log N) allocator over the binary64 distortion objective.

    Finite binary64 costs are lifted to integers under one common power-of-two
    denominator. Allocation sums and comparisons therefore cannot drift from
    floating accumulation or ambiguous equality.
    """

    values = [
        _require_tensor(tensor, name=name, ndim=1).to(torch.float64).cpu()
        for tensor, name in ((d4, "d4"), (d6, "d6"), (d8, "d8"))
    ]
    if values[0].shape != values[1].shape or values[0].shape != values[2].shape:
        raise ValueError("multibit distortion shapes differ")
    rows = values[0].numel()
    if not 0 <= marginal_steps <= 2 * rows:
        raise ValueError("multibit marginal steps are outside the exact domain")
    ratios = [
        [float(values[precision][row]).as_integer_ratio() for precision in range(3)]
        for row in range(rows)
    ]
    common_power = max(
        (denominator.bit_length() - 1 for row in ratios for _, denominator in row),
        default=0,
    )
    integer_costs = [
        [
            numerator << (common_power - (denominator.bit_length() - 1))
            for numerator, denominator in row
        ]
        for row in ratios
    ]
    first_gain = [integer_costs[row][0] - integer_costs[row][1] for row in range(rows)]
    second_gain = [integer_costs[row][1] - integer_costs[row][2] for row in range(rows)]
    convex_rows = [index for index in range(rows) if first_gain[index] >= second_gain[index]]
    nonconvex_rows = [index for index in range(rows) if first_gain[index] < second_gain[index]]
    increments = [(first_gain[row], row, 0) for row in convex_rows] + [
        (second_gain[row], row, 1) for row in convex_rows
    ]
    increments.sort(key=lambda item: (-item[0], item[1], item[2]))
    increment_prefix = [0]
    for gain, _, _ in increments:
        increment_prefix.append(increment_prefix[-1] + gain)

    bundles = [
        (
            first_gain[row] + second_gain[row],
            row,
            first_gain[row],
            second_gain[row],
        )
        for row in nonconvex_rows
    ]
    bundles.sort(key=lambda item: (-item[0], item[1]))
    bundle_prefix = [0]
    for gain, _, _, _ in bundles:
        bundle_prefix.append(bundle_prefix[-1] + gain)

    prefix_inside: list[int | None] = [None]
    best_inside: int | None = None
    for rank, (_, row, _, second) in enumerate(bundles):
        if best_inside is None:
            best_inside = rank
        else:
            _, best_row, _, best_second = bundles[best_inside]
            adjustment = -second
            best_adjustment = -best_second
            if adjustment > best_adjustment or (adjustment == best_adjustment and row > best_row):
                best_inside = rank
        prefix_inside.append(best_inside)

    suffix_outside: list[int | None] = [None] * (len(bundles) + 1)
    best_outside: int | None = None
    for rank in range(len(bundles) - 1, -1, -1):
        _, row, first, _ = bundles[rank]
        if best_outside is None:
            best_outside = rank
        else:
            _, best_row, best_first, _ = bundles[best_outside]
            if first > best_first or (first == best_first and row < best_row):
                best_outside = rank
        suffix_outside[rank] = best_outside

    Descriptor = tuple[int, int | None, int]

    def materialize(descriptor: Descriptor) -> torch.Tensor:
        bundle_count, singleton_rank, convex_steps = descriptor
        codes = torch.zeros(rows, dtype=torch.uint8)
        for _, row, _ in increments[:convex_steps]:
            codes[row] += 1
        bundle_limit = bundle_count + int(
            singleton_rank is not None and singleton_rank < bundle_count
        )
        for rank in range(bundle_limit):
            if rank != singleton_rank:
                codes[bundles[rank][1]] = 2
        if singleton_rank is not None:
            codes[bundles[singleton_rank][1]] = 1
        if int(codes.sum().item()) != marginal_steps:
            _fail("independent multibit allocator violated its exact budget")
        return codes

    def choose(
        incumbent: tuple[int, Descriptor] | None,
        gain: int,
        descriptor: Descriptor,
    ) -> tuple[int, Descriptor]:
        if incumbent is None or gain > incumbent[0]:
            return gain, descriptor
        if gain < incumbent[0]:
            return incumbent
        candidate_codes = materialize(descriptor)
        incumbent_codes = materialize(incumbent[1])
        differing = torch.nonzero(candidate_codes != incumbent_codes).reshape(-1)
        if differing.numel() and (
            int(candidate_codes[int(differing[0])]) > int(incumbent_codes[int(differing[0])])
        ):
            return gain, descriptor
        return incumbent

    best: tuple[int, Descriptor] | None = None
    bundle_count_max = min(len(bundles), marginal_steps // 2)
    for bundle_count in range(bundle_count_max + 1):
        convex_steps = marginal_steps - 2 * bundle_count
        if 0 <= convex_steps <= len(increments):
            gain = bundle_prefix[bundle_count] + increment_prefix[convex_steps]
            best = choose(
                best,
                gain,
                (bundle_count, None, convex_steps),
            )
        convex_with_singleton = marginal_steps - 2 * bundle_count - 1
        if not 0 <= convex_with_singleton <= len(increments):
            continue
        singleton_candidates: list[tuple[int, int]] = []
        outside = suffix_outside[bundle_count]
        if outside is not None:
            singleton_candidates.append((bundles[outside][2], outside))
        inside = prefix_inside[bundle_count]
        if inside is not None and bundle_count < len(bundles):
            adjustment = bundles[bundle_count][0] - bundles[inside][3]
            singleton_candidates.append((adjustment, inside))
        for adjustment, singleton_rank in singleton_candidates:
            gain = (
                bundle_prefix[bundle_count] + increment_prefix[convex_with_singleton] + adjustment
            )
            best = choose(
                best,
                gain,
                (bundle_count, singleton_rank, convex_with_singleton),
            )
    if best is None:
        _fail("independent multibit allocator found no exact allocation")
    return materialize(best[1])


def _selection_sha256(
    selections: Sequence[torch.Tensor],
) -> str:
    digest = hashlib.sha256(b"recurquant.statelease.equal-byte.selection.v1\0")
    for layer_index, selection in zip(
        LINEAR_LAYER_INDICES,
        selections,
        strict=True,
    ):
        digest.update(layer_index.to_bytes(4, "little", signed=False))
        digest.update(selection.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _global_masks(
    scores: torch.Tensor,
    *,
    selected_count: int,
) -> list[torch.Tensor]:
    indices = stable_descending_indices(scores.reshape(-1), selected_count)
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    mask[indices] = True
    return [chunk.reshape(HEADS, ROWS) for chunk in mask.split(HEADS * ROWS)]


def _physical_comparator_expectations(
    source_states: Mapping[int, torch.Tensor],
    query_ema: torch.Tensor,
    *,
    codec: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    encoded: list[torch.Tensor] = []
    row_sets: list[torch.Tensor] = []
    d4_sets: list[torch.Tensor] = []
    d6_sets: list[torch.Tensor] = []
    d8_sets: list[torch.Tensor] = []
    for position, layer_index in enumerate(LINEAR_LAYER_INDICES):
        transformed = independent_rht_encode(
            source_states[layer_index],
            layer_index=layer_index,
        )
        rows = transformed.reshape(-1, WIDTH)
        encoded.append(transformed)
        row_sets.append(rows)
        weight = query_ema[position].reshape(-1)
        distortions: list[torch.Tensor] = []
        for bits in (4, 6, 8):
            restored = _quantized_row_reconstruction(rows, bits)
            distortions.append(weight * (restored - rows).square().mean(dim=-1))
        d4_sets.append(distortions[0])
        d6_sets.append(distortions[1])
        d8_sets.append(distortions[2])

    expected: dict[str, torch.Tensor] = {}
    materialized: list[torch.Tensor] = []
    selection_streams: list[torch.Tensor] = []
    if codec == "expanded_rht_q4_q8":
        masks = _global_masks(
            torch.cat([d4 - d8 for d4, d8 in zip(d4_sets, d8_sets, strict=True)]),
            selected_count=EXPANDED_Q48_PROMOTIONS,
        )
        for position, (rows, mask, transformed) in enumerate(
            zip(row_sets, masks, encoded, strict=True)
        ):
            packed = pack_physical_q4_q8(rows, mask.reshape(-1))
            expected.update(
                {
                    f"layer_{position}.q4_payload": packed.low_payload,
                    f"layer_{position}.q8_payload": packed.high_payload,
                    f"layer_{position}.scales": packed.scales,
                    f"layer_{position}.precision_mask": packed.precision_mask,
                }
            )
            selection_streams.append(packed.precision_mask)
            decoded = independent_rht_decode(
                packed.dequantize().reshape_as(transformed),
                layer_index=LINEAR_LAYER_INDICES[position],
            )
            materialized.append(decoded)
        padding = EXPANDED_Q48_PADDING_BYTES
        selected_units = EXPANDED_Q48_PROMOTIONS
    elif codec == "rht_q4_q6_q8":
        precision = _independent_allocate_multibit_fast(
            torch.cat(d4_sets),
            torch.cat(d6_sets),
            torch.cat(d8_sets),
            marginal_steps=MULTIBIT_MARGINAL_STEPS,
        )
        codes = [chunk.reshape(HEADS, ROWS) for chunk in precision.split(HEADS * ROWS)]
        for position, (rows, layer_codes, transformed) in enumerate(
            zip(row_sets, codes, encoded, strict=True)
        ):
            packed = pack_physical_q4_q6_q8(rows, layer_codes.reshape(-1))
            expected.update(
                {
                    f"layer_{position}.q4_payload": packed.q4_payload,
                    f"layer_{position}.q6_payload": packed.q6_payload,
                    f"layer_{position}.q8_payload": packed.q8_payload,
                    f"layer_{position}.scales": packed.scales,
                    f"layer_{position}.precision_codes": packed.precision_codes,
                }
            )
            selection_streams.append(packed.precision_codes)
            decoded = independent_rht_decode(
                packed.dequantize().reshape_as(transformed),
                layer_index=LINEAR_LAYER_INDICES[position],
            )
            materialized.append(decoded)
        padding = MULTIBIT_PADDING_BYTES
        selected_units = MULTIBIT_MARGINAL_STEPS
    elif codec == "rht_residual_q4":
        benefits: list[torch.Tensor] = []
        for position, rows in enumerate(row_sets):
            base = _quantized_row_reconstruction(rows, 4)
            residual = rows - base
            correction = _quantized_row_reconstruction(residual, 4)
            weight = query_ema[position].reshape(-1)
            benefits.append(
                weight
                * (
                    (base - rows).square().mean(dim=-1)
                    - (base + correction - rows).square().mean(dim=-1)
                )
            )
        masks = _global_masks(
            torch.cat(benefits),
            selected_count=RESIDUAL_Q4_ROWS,
        )
        for position, (rows, mask, transformed) in enumerate(
            zip(row_sets, masks, encoded, strict=True)
        ):
            packed = pack_physical_residual_q4(rows, mask.reshape(-1))
            expected.update(
                {
                    f"layer_{position}.base_q4_payload": (packed.base_payload.reshape(-1)),
                    f"layer_{position}.base_scales": (packed.base_scales.reshape(HEADS, ROWS)),
                    f"layer_{position}.residual_q4_payload": (packed.residual_payload.reshape(-1)),
                    f"layer_{position}.residual_scales": (packed.residual_scales.reshape(-1, 1)),
                    f"layer_{position}.lease_mask": packed.residual_mask,
                }
            )
            selection_streams.append(packed.residual_mask)
            decoded = independent_rht_decode(
                packed.dequantize().reshape_as(transformed),
                layer_index=LINEAR_LAYER_INDICES[position],
            )
            materialized.append(decoded)
        padding = RESIDUAL_Q4_PADDING_BYTES
        selected_units = RESIDUAL_Q4_ROWS
    else:
        raise ValueError("unknown physical comparator codec")

    expected["query_energy_ema"] = query_ema
    expected["reserved_padding"] = torch.zeros(padding, dtype=torch.uint8)
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_source = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    for layer_index, candidate in zip(
        LINEAR_LAYER_INDICES,
        materialized,
        strict=True,
    ):
        error = candidate.to(torch.float64) - source_states[layer_index].to(torch.float64)
        squared_error += error.square().sum()
        squared_source += source_states[layer_index].to(torch.float64).square().sum()
        maximum = max(maximum, float(error.abs().max().item()))
    metrics = {
        "mean_squared_error": float((squared_error / STATE_ELEMENTS).item()),
        "relative_l2_error": float(
            (squared_error.sqrt() / squared_source.sqrt().clamp_min(1e-12)).item()
        ),
        "max_absolute_error": maximum,
        "selected_units": selected_units,
        "selection_sha256": _selection_sha256(selection_streams),
    }
    return expected, metrics


def _verify_equal_byte_comparators(value: object) -> dict[str, object]:
    comparators = _require_exact_keys(
        value,
        name="equal_byte_comparators",
        keys=("source_states", "query_energy_ema", "snapshots"),
    )
    raw_states = _require_exact_keys(
        comparators["source_states"],
        name="equal_byte.source_states",
        keys=tuple(str(index) for index in LINEAR_LAYER_INDICES),
    )
    source_states: dict[int, torch.Tensor] = {}
    for layer_index in LINEAR_LAYER_INDICES:
        state = _require_tensor(
            raw_states[str(layer_index)],
            name=f"equal_byte.source_states[{layer_index}]",
            dtype=torch.float32,
            ndim=4,
        )
        if state.shape != (1, HEADS, ROWS, WIDTH):
            _fail("equal-byte source state has wrong frozen geometry")
        source_states[layer_index] = state
    query_ema = _require_tensor(
        comparators["query_energy_ema"],
        name="equal_byte.query_energy_ema",
        dtype=torch.float32,
        ndim=3,
    )
    if query_ema.shape != (LAYERS, HEADS, ROWS):
        _fail("equal-byte query EMA has wrong frozen geometry")
    if not torch.isfinite(query_ema).all().item() or (query_ema < 0).any().item():
        _fail("equal-byte query EMA is invalid")
    snapshots = _require_exact_keys(
        comparators["snapshots"],
        name="equal_byte.snapshots",
        keys=("expanded_rht_q4_q8", "rht_q4_q6_q8", "rht_residual_q4"),
    )
    verified: dict[str, object] = {}
    for codec in ("expanded_rht_q4_q8", "rht_q4_q6_q8", "rht_residual_q4"):
        snapshot = _require_exact_keys(
            snapshots[codec],
            name=f"equal_byte.snapshots.{codec}",
            keys=(
                "tensors",
                "persistent_enumeration",
                "evidence",
                "no_replay",
                "no_hidden_persistent_state_mirror",
            ),
        )
        if (
            snapshot["no_replay"] is not True
            or snapshot["no_hidden_persistent_state_mirror"] is not True
        ):
            _fail(f"{codec} did not declare its no-replay/no-mirror contract")
        tensors = snapshot["tensors"]
        if not isinstance(tensors, Mapping) or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in tensors.items()
        ):
            _fail(f"{codec} tensors violate the closed schema")
        expected_tensors, expected_metrics = _physical_comparator_expectations(
            source_states,
            query_ema,
            codec=codec,
        )
        if set(tensors) != set(expected_tensors):
            _fail(f"{codec} physical tensor names differ")
        for name, expected in expected_tensors.items():
            _assert_tensor_close(
                tensors[name],
                expected,
                name=f"{codec}.{name}",
                exact=True,
            )
        _verify_tensor_enumeration(
            snapshot["persistent_enumeration"],
            tensors,
            name=f"{codec}.persistent_enumeration",
        )
        illegal_fp32 = [
            name
            for name, tensor in tensors.items()
            if tensor.dtype == torch.float32 and name != "query_energy_ema"
        ]
        if illegal_fp32:
            _fail(f"{codec} retains unexpected FP32 tensors: {illegal_fp32}")
        actual_bytes = sum(tensor_bytes(tensor) for tensor in tensors.values())
        if actual_bytes != STATELEASE_BYTES:
            _fail(f"{codec} owns {actual_bytes} bytes, expected {STATELEASE_BYTES}")
        evidence = snapshot["evidence"]
        if not isinstance(evidence, Mapping):
            _fail(f"{codec} evidence must be a mapping")
        if codec == "expanded_rht_q4_q8":
            payload_bytes = 3_228_864
            scale_bytes = 73_728
            precision_bytes = 4_608
            padding_bytes = EXPANDED_Q48_PADDING_BYTES
        elif codec == "rht_q4_q6_q8":
            payload_bytes = 3_224_256
            scale_bytes = 73_728
            precision_bytes = 9_216
            padding_bytes = MULTIBIT_PADDING_BYTES
        else:
            payload_bytes = 3_202_496
            scale_bytes = 100_078
            precision_bytes = 4_608
            padding_bytes = RESIDUAL_Q4_PADDING_BYTES
        expected_evidence = {
            "codec": codec,
            "state_elements": STATE_ELEMENTS,
            "fp32_state_bytes": STATE_ELEMENTS * 4,
            "payload_bytes": payload_bytes,
            "scale_bytes": scale_bytes,
            "precision_bytes": precision_bytes,
            "query_ema_bytes": QUERY_EMA_BYTES,
            "padding_bytes": padding_bytes,
            "resident_bytes": STATELEASE_BYTES,
            "selected_units": expected_metrics["selected_units"],
            "expected_selected_units": expected_metrics["selected_units"],
            "selection_sha256": expected_metrics["selection_sha256"],
            "mean_squared_error": expected_metrics["mean_squared_error"],
            "relative_l2_error": expected_metrics["relative_l2_error"],
            "max_absolute_error": expected_metrics["max_absolute_error"],
            "compression_ratio": (STATE_ELEMENTS * 4) / STATELEASE_BYTES,
        }
        if set(evidence) != set(expected_evidence):
            _fail(f"{codec} evidence violates the closed schema")
        for field, expected in expected_evidence.items():
            actual = evidence[field]
            if isinstance(expected, float):
                if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual),
                    expected,
                    rel_tol=3e-7,
                    abs_tol=3e-9,
                ):
                    _fail(f"{codec} evidence differs at {field}")
            elif actual != expected:
                _fail(f"{codec} evidence differs at {field}")
        verified[codec] = {
            "resident_bytes": actual_bytes,
            "selected_units": expected_metrics["selected_units"],
            "selection_sha256": expected_metrics["selection_sha256"],
            "relative_l2_error": expected_metrics["relative_l2_error"],
        }
    return verified


def verify_production_stage0(
    artifact_path: Path,
    *,
    sha256_path: Path | None = None,
) -> dict[str, object]:
    """Authenticate and independently recompute the full production Stage 0."""

    assert_independent_imports(Path(__file__))
    guard_protected_mbpp_window(stage="stage0")
    if artifact_path.is_symlink() or (sha256_path is not None and sha256_path.is_symlink()):
        _fail("artifact and SHA-256 sidecar must be regular non-symlink files")
    resolved_artifact = artifact_path.resolve()
    resolved_sidecar = (
        resolved_artifact.with_suffix(resolved_artifact.suffix + ".sha256")
        if sha256_path is None
        else sha256_path.resolve()
    )
    artifact, loaded_integrity = _load_authenticated_production_artifact_with_integrity(
        resolved_artifact,
        resolved_sidecar,
    )
    repo_root = Path(__file__).resolve().parents[1]
    _verify_source_and_method_identity(artifact, repo_root=repo_root)
    runtime_identity = _verify_runtime_identity(artifact["runtime_identity"])
    successful_kernel = _verify_successful_kernel_receipt(artifact["successful_kernel_receipt"])
    trace = _verify_production_trace(artifact["production_trace"])
    resident = _verify_resident_snapshot(artifact["resident_snapshot"])
    lifecycle = _verify_lifecycle(
        artifact["lifecycle"],
        trace_hashes=trace["trace_hashes"],
    )
    compatibility = _verify_cc1_production_compatibility(artifact["cc1_compatibility"])
    comparators = _verify_equal_byte_comparators(artifact["equal_byte_comparators"])
    integrity_after = _artifact_integrity_snapshot(resolved_artifact, resolved_sidecar)
    if integrity_after != loaded_integrity:
        _fail("artifact or sidecar changed during independent Stage-0 verification")
    return {
        "status": "production_stage0_pass",
        "experiment_stage0_complete": True,
        "scope": "authenticated production plus independent synthetic recomputation",
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
        "weights_only_load": True,
        "independent_imports": True,
        "artifact": _public_output_label(resolved_artifact, repo_root=repo_root),
        "sidecar": _public_output_label(resolved_sidecar, repo_root=repo_root),
        "artifact_file_sha256": integrity_after["artifact_file_sha256"],
        "sidecar_file_sha256": integrity_after["sidecar_file_sha256"],
        "canonical_payload_sha256": artifact["canonical_payload_sha256"],
        "repository_commit": artifact["source_identity"]["repo_head"],
        "runtime_identity": runtime_identity,
        "successful_kernel_receipt": successful_kernel,
        "trace": {key: value for key, value in trace.items() if key != "trace_hashes"},
        "storage": resident["storage"],
        "whole_cache_storage_audit": resident["whole_cache_storage_audit"],
        "resident_tensor_count": resident["tensor_count"],
        "resident_snapshot_sha256": resident["resident_digest"],
        "lifecycle": lifecycle,
        "cc1_compatibility": compatibility,
        "equal_byte_comparators": comparators,
    }


def _public_output_label(path: Path, *, repo_root: Path) -> str:
    """Return a stable report label without exposing an absolute local path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


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
        "--artifact",
        type=Path,
        help=(
            "authenticated production .pt artifact; omit to run only the "
            "independent algebra self-test"
        ),
    )
    parser.add_argument(
        "--sha256",
        type=Path,
        help="optional SHA-256 sidecar path (defaults to ARTIFACT.pt.sha256)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact rather than indented JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sha256 is not None and args.artifact is None:
        raise SystemExit("--sha256 requires --artifact")
    report = (
        run_synthetic_stage0()
        if args.artifact is None
        else verify_production_stage0(
            args.artifact,
            sha256_path=args.sha256,
        )
    )
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
