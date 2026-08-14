"""Pure calibration math and split-half gates for Experiment 013.

This module deliberately stops at deterministic CPU-FP64 score reduction,
exact code-map allocation, and policy-stability evidence.  It does not load a
dataset or model, resolve protected identities, quantize a live state, or open
an evaluation result.

The frozen reduction is::

    sequence = mean(anchor_energy * per_row_codec_mse, anchors)
    ruler = mean(mean(sequence, each of four categories), categories)
    D_b = mean(MBPP_b, PG19_b, RULER_b)

All input order is canonicalized before floating-point reductions.  Artifact
arrays use contiguous little-endian binary64 bytes rather than JSON decimal
renderings, and validation recomputes both exact allocations from those bytes.
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
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch

from .evidence import canonical_json_bytes
from .metrics import spearman_correlation
from .multibit_policy import allocate_exact_multibit_codes_fast
from .multibit_quantization import _pack_precision_codes
from .quantization import QuantizationSpec, quantize_dequantize
from .rht import RHT_SEED, right_rht_encode
from .static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    STATIC_Q468_ALLOCATOR_REVISION,
    StaticRhtQ468Geometry,
    static_q468_distortion_sha256,
)

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
CALIBRATION_SCORE_ARTIFACT_PROFILE = "experiment-013-qwen35-0.8b-static-q468-frozen-v1"
GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND = "recurquant_static_q468_scores_generic"
GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION = "static-q468-scores-generic-v1"
GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE = "generic-static-q468-calibration-v1"
CALIBRATION_SCORE_DTYPE = "float64-le"

SPLIT_HALF_STABILITY_ARTIFACT_KIND = "recurquant_experiment013_static_q468_split_half_stability"
SPLIT_HALF_STABILITY_ARTIFACT_SCHEMA_VERSION = 1
SPLIT_HALF_STABILITY_ARTIFACT_REVISION = "experiment-013-static-q468-split-half-stability-v1"
SPLIT_HALF_STABILITY_ARTIFACT_PROFILE = (
    "experiment-013-qwen35-0.8b-static-k29334-split-half-frozen-v1"
)

GENERIC_REDUCTION_PROFILE = "generic-anchor-row-reduction-v1"
FROZEN_REDUCTION_PROFILE = "experiment-013-qwen35-0.8b-anchor-reduction-v1"
FROZEN_UNWEIGHTED_MSE_PROFILE = "rht_q468_static_mse_k29334"
FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE = "rht_q468_static_diag_empirical_fisher_h1_k29334"
FROZEN_FISHER_HORIZON = 1
FROZEN_SOURCE_AXIS_ORDER = ("anchor", "layer", "head", "key_row")
_SOURCE_TENSOR_NAMES = ("query_energy", "q4_mse", "q6_mse", "q8_mse")

COMPARATOR_SCORE_ARTIFACT_KIND = "recurquant_experiment013_static_q468_comparator_scores"
COMPARATOR_SCORE_ARTIFACT_SCHEMA_VERSION = 1
COMPARATOR_SCORE_ARTIFACT_REVISION = "experiment-013-static-q468-comparator-scores-v1"
COMPARATOR_SCORE_ARTIFACT_PROFILE = "experiment-013-qwen35-0.8b-static-q468-comparators-frozen-v1"
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
    FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE: ("B(T)=frozen_anchor_positions(T-2)"),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SCORE_HASH_DOMAIN = b"recurquant.experiment013.sequence-scores.v1\0"
_ANCHOR_INPUT_HASH_DOMAIN = b"recurquant.experiment013.anchor-inputs.v1\0"
_CODE_MAP_HASH_DOMAIN = b"recurquant.static-q468-code-map.v1\0"
_IDENTITY_RECORD_HASH_DOMAIN = b"recurquant.experiment013.identity-record.v1\0"
_FISHER_BOUNDARY_HASH_DOMAIN = b"recurquant.experiment013.fisher-boundary.v1\0"
_FISHER_BOUNDARY_TOKEN_HASH_DOMAIN = b"recurquant.experiment013.fisher-boundary-token-sequence.v1\0"
_COMPARATOR_POSITION_HASH_DOMAIN = b"recurquant.experiment013.comparator-position-manifest.v1\0"
_COMPARATOR_ENDPOINT_INPUT_HASH_DOMAIN = b"recurquant.experiment013.comparator-endpoint-input.v1\0"
_COMPARATOR_TARGET_NLL_HASH_DOMAIN = b"recurquant.experiment013.comparator-target-nll.v1\0"
_COMPARATOR_SEQUENCE_SCORE_HASH_DOMAIN = b"recurquant.experiment013.comparator-sequence-score.v1\0"
_COMPARATOR_AGGREGATE_SCORE_HASH_DOMAIN = (
    b"recurquant.experiment013.comparator-aggregate-score.v1\0"
)
_COMPARATOR_SEQUENCE_MANIFEST_HASH_DOMAIN = (
    b"recurquant.experiment013.comparator-sequence-manifest.v1\0"
)
_COMPARATOR_POSITION_MANIFEST_HASH_DOMAIN = (
    b"recurquant.experiment013.comparator-ordered-position-manifest.v1\0"
)
_FISHER_BOUNDARY_SCHEMA = "recurquant.experiment013.fisher-boundary.v1"
_FISHER_BOUNDARY_FIELDS = frozenset(
    {
        "schema",
        "horizon",
        "boundary_positions",
        "input_positions",
        "target_positions",
        "input_token_ids_sha256",
        "target_token_ids_sha256",
        "fisher_boundary_sha256",
    }
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
    """Raised when a calibration score artifact fails closed."""


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
) -> tuple[
    CalibrationFamily,
    str,
    RulerCategory | None,
    str,
    int | None,
    int | None,
    int,
]:
    if family not in CALIBRATION_FAMILY_ORDER:
        raise ValueError("family must be one of mbpp, pg19, or ruler")
    normalized_family = cast(CalibrationFamily, family)
    normalized_config = _canonical_text(config, name="config")
    if normalized_family == "ruler":
        if ruler_category not in RULER_CATEGORY_ORDER:
            raise ValueError(
                "ruler_category must be one of retrieval, multi_hop_tracing, "
                "aggregation, or question_answering for RULER"
            )
        normalized_category = cast(RulerCategory, ruler_category)
        normalized_configured_length = _strict_positive_int(
            configured_length,
            name="configured_length",
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
        seed,
        normalized_configured_length,
        normalized_tokens,
    )


def frozen_anchor_positions(token_count: int) -> tuple[int, ...]:
    """Return the frozen unique zero-based post-token anchor positions.

    For ``T >= 16`` this is exactly
    ``floor((j + 1) * T / 16) - 1`` for ``j = 0,...,15``.  Shorter non-empty
    sequences capture every token.
    """

    tokens = _strict_positive_int(token_count, name="token_count")
    if tokens < FROZEN_ANCHOR_COUNT:
        positions = tuple(range(tokens))
    else:
        positions = tuple(
            (index + 1) * tokens // FROZEN_ANCHOR_COUNT - 1 for index in range(FROZEN_ANCHOR_COUNT)
        )
    if len(positions) != len(set(positions)):
        raise RuntimeError("frozen anchor equation produced duplicate positions")
    if not positions or positions[0] < 0 or positions[-1] >= tokens:
        raise RuntimeError("frozen anchor equation produced an out-of-range position")
    return positions


def fisher_h1_boundary_positions(token_count: int) -> tuple[int, ...]:
    """Return boundary positions with both causal input and target available.

    A boundary ``b`` stores ``S_b``; H=1 consumes ``x_(b+1)`` and scores the
    resulting logits against ``x_(b+2)``.  Therefore a length-``T`` sequence
    has exactly ``T - 2`` eligible boundaries before the frozen anchor equation
    is applied.
    """

    tokens = _strict_positive_int(token_count, name="token_count")
    if tokens < 3:
        raise ValueError("H=1 Fisher calibration requires at least three tokens")
    return frozen_anchor_positions(tokens - 2)


def _q468_endpoint_specs(geometry: StaticRhtQ468Geometry) -> tuple[QuantizationSpec, ...]:
    common = {
        "group_size": geometry.value_width,
        "scale_bits": 16,
        "flatten_last_dims": 1,
        "rounding": "nearest",
        "seed": RHT_SEED,
    }
    return tuple(QuantizationSpec(bits=bits, **common) for bits in (4, 6, 8))


def _endpoint_state_tensor(
    value: object,
    *,
    name: str,
    geometry: StaticRhtQ468Geometry,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = (
        len(geometry.layer_indices),
        geometry.heads,
        geometry.key_rows,
        geometry.value_width,
    )
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if value.device.type == "meta" or not value.is_floating_point():
        raise TypeError(f"{name} must be a materialized floating-point tensor")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")
    normalized = value.detach().to(torch.float32)
    if not torch.isfinite(normalized).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def compute_rht_unweighted_mse_endpoints(
    recurrent_state: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute equal-weight per-row RHT Q4/Q6/Q8 MSE without query proxies."""

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    state = _endpoint_state_tensor(
        recurrent_state,
        name="recurrent_state",
        geometry=geometry,
    )
    per_bit: list[list[torch.Tensor]] = [[], [], []]
    specifications = _q468_endpoint_specs(geometry)
    with torch.no_grad():
        for local_index, layer_index in enumerate(geometry.layer_indices):
            encoded = right_rht_encode(
                state[local_index].unsqueeze(0),
                layer_index=layer_index,
                expected_heads=geometry.heads,
                output_dtype=torch.float32,
            )
            for destination, specification in zip(per_bit, specifications, strict=True):
                restored = quantize_dequantize(encoded, specification).tensor
                error_fp64 = (
                    (restored - encoded)
                    .detach()
                    .to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )
                destination.append(error_fp64.square().mean(dim=-1).squeeze(0).contiguous())
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(torch.stack(rows, dim=0).contiguous() for rows in per_bit),
    )


def compute_rht_diagonal_empirical_fisher_h1_endpoints(
    recurrent_state: torch.Tensor,
    state_gradient: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``0.5 * sum_v((RHT grad)^2 * quantization_error^2)``.

    Both state and loss gradient are transformed by the same orthonormal right
    RHT.  Scores are returned as CPU-FP64 ``[layer, head, key_row]`` tensors and
    intentionally use neither query-energy weighting nor proxy normalization.
    """

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    state = _endpoint_state_tensor(
        recurrent_state,
        name="recurrent_state",
        geometry=geometry,
    )
    gradient = _endpoint_state_tensor(
        state_gradient,
        name="state_gradient",
        geometry=geometry,
        device=state.device,
    )
    per_bit: list[list[torch.Tensor]] = [[], [], []]
    specifications = _q468_endpoint_specs(geometry)
    with torch.no_grad():
        for local_index, layer_index in enumerate(geometry.layer_indices):
            encoded_state = right_rht_encode(
                state[local_index].unsqueeze(0),
                layer_index=layer_index,
                expected_heads=geometry.heads,
                output_dtype=torch.float32,
            )
            encoded_gradient = right_rht_encode(
                gradient[local_index].unsqueeze(0),
                layer_index=layer_index,
                expected_heads=geometry.heads,
                output_dtype=torch.float32,
            )
            gradient_fp64 = encoded_gradient.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            squared_gradient = gradient_fp64.square()
            for destination, specification in zip(per_bit, specifications, strict=True):
                restored = quantize_dequantize(encoded_state, specification).tensor
                error_fp64 = (
                    (restored - encoded_state)
                    .detach()
                    .to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )
                risk = 0.5 * (squared_gradient * error_fp64.square()).sum(dim=-1)
                destination.append(risk.squeeze(0).contiguous())
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(torch.stack(rows, dim=0).contiguous() for rows in per_bit),
    )


@dataclass(frozen=True, slots=True)
class UnweightedEndpointBatch:
    """Per-anchor Q4/Q6/Q8 endpoint scores for one static selector."""

    selector_profile: UnweightedSelectorProfile
    token_count: int
    anchor_positions: tuple[int, ...]
    q4_scores: torch.Tensor
    q6_scores: torch.Tensor
    q8_scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class FrozenComparatorEndpointBatch:
    """Identity-v5-bound endpoint scores for one frozen comparator sequence."""

    selector_profile: UnweightedSelectorProfile
    family: CalibrationFamily
    config: str
    ruler_category: RulerCategory | None
    canonical_id: str
    seed: int | None
    configured_length: int | None
    token_count: int
    endpoint_positions: tuple[int, ...]
    q4_scores: torch.Tensor
    q6_scores: torch.Tensor
    q8_scores: torch.Tensor
    sequence_token_ids: tuple[int, ...]
    identity_record: Mapping[str, object]
    target_nlls: torch.Tensor | None = None


def reduce_unweighted_endpoint_anchors(
    batch: UnweightedEndpointBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take an equal CPU-FP64 mean over anchors, with no proxy weighting."""

    if not isinstance(batch, UnweightedEndpointBatch):
        raise TypeError("batch must be an UnweightedEndpointBatch")
    if batch.selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE:
        expected_positions = frozen_anchor_positions(batch.token_count)
    elif batch.selector_profile == FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE:
        expected_positions = fisher_h1_boundary_positions(batch.token_count)
    else:
        raise ValueError("selector_profile is not a frozen unweighted selector")
    if not isinstance(batch.anchor_positions, tuple):
        raise TypeError("anchor_positions must be a tuple")
    if batch.anchor_positions != expected_positions:
        raise ValueError("anchor_positions differ from the selector's frozen equation")
    anchors = len(expected_positions)
    values = tuple(
        _cpu_fp64_matrix(value, name=name, anchors=anchors)
        for name, value in (
            ("q4_scores", batch.q4_scores),
            ("q6_scores", batch.q6_scores),
            ("q8_scores", batch.q8_scores),
        )
    )
    if values[1].shape != values[0].shape or values[2].shape != values[0].shape:
        raise ValueError("Q4/Q6/Q8 endpoint score shapes must match exactly")
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(value.mean(dim=0).contiguous() for value in values),
    )


def allocate_unweighted_endpoint_policy(
    batch: UnweightedEndpointBatch,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Allocate an exact Q4/Q6/Q8 code map from unweighted endpoint scores."""

    scores = reduce_unweighted_endpoint_anchors(batch)
    rows = scores[0].numel()
    steps = _strict_nonnegative_int(marginal_steps, name="marginal_steps")
    if steps > 2 * rows:
        raise ValueError("marginal_steps exceeds two steps per row")
    code_map = allocate_exact_multibit_codes_fast(
        *(value.reshape(1, rows) for value in scores),
        marginal_steps=steps,
    ).reshape(-1)
    if code_map.dtype != torch.uint8 or code_map.device.type != "cpu":
        raise RuntimeError("exact allocator returned a non-canonical code map")
    if int(code_map.to(torch.int64).sum().item()) != steps:
        raise RuntimeError("exact allocator did not satisfy the requested marginal budget")
    return code_map.contiguous()


def _cpu_fp64_matrix(
    value: object,
    *,
    name: str,
    anchors: int,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.ndim < 2 or value.shape[0] != anchors or value.numel() == 0:
        raise ValueError(f"{name} must have shape [anchor, row...] with {anchors} anchors")
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must match shape {expected_shape}")
    normalized = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(normalized).all().item():
        raise ValueError(f"{name} must contain only finite values")
    if (normalized < 0).any().item():
        raise ValueError(f"{name} must contain only non-negative values")
    return normalized.reshape(anchors, -1).clone()


def _cpu_fp64_scores(value: object, *, name: str, expected_rows: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if expected_rows is not None and value.numel() != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} rows")
    normalized = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1).contiguous()
    if not torch.isfinite(normalized).all().item():
        raise ValueError(f"{name} must contain only finite values")
    if (normalized < 0).any().item():
        raise ValueError(f"{name} must contain only non-negative values")
    return normalized.clone()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return array.astype("<f8", copy=False).tobytes(order="C")


def _domain_json_sha256(domain: bytes, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationSourceTensorContract:
    """Shape, axis, dtype, and reduction profile bound into score evidence."""

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
            raise ValueError("source dtypes must cover query_energy, q4_mse, q6_mse, and q8_mse")
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


def _hash_score_triplet(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    domain: bytes,
    metadata: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(canonical_json_bytes(metadata))
    for label, tensor in zip((b"D4\0", b"D6\0", b"D8\0"), (d4, d6, d8), strict=True):
        digest.update(label)
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _resolver_canonical_json_bytes(value: object) -> bytes:
    """Match the compact canonical JSON used by the identity resolver."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sequence_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash exact ordered token IDs with the capture/resolver representation."""

    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(token_ids, Sequence):
        raise TypeError("sequence_token_ids must be an integer sequence")
    normalized: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"sequence_token_ids[{index}] must be a non-negative integer")
        normalized.append(token_id)
    if not normalized:
        raise ValueError("sequence_token_ids cannot be empty")
    return hashlib.sha256(_resolver_canonical_json_bytes(normalized)).hexdigest()


def _fisher_token_ids_sha256(token_ids: Sequence[int], *, role: str) -> str:
    payload = {"role": role, "token_ids": list(token_ids)}
    return hashlib.sha256(
        _FISHER_BOUNDARY_TOKEN_HASH_DOMAIN + _resolver_canonical_json_bytes(payload)
    ).hexdigest()


def _validate_fisher_boundary(
    value: object,
    *,
    sequence_token_ids: tuple[int, ...],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _FISHER_BOUNDARY_FIELDS:
        raise ValueError("fisher_boundary must contain the exact frozen H=1 fields")
    if value["schema"] != _FISHER_BOUNDARY_SCHEMA:
        raise ValueError("fisher_boundary schema drifted")
    horizon = value["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != 1:
        raise ValueError("fisher_boundary horizon must equal H=1")
    boundary_positions = list(fisher_h1_boundary_positions(len(sequence_token_ids)))
    input_positions = [position + 1 for position in boundary_positions]
    target_positions = [position + 1 for position in input_positions]
    expected_positions = {
        "boundary_positions": boundary_positions,
        "input_positions": input_positions,
        "target_positions": target_positions,
    }
    for name, expected in expected_positions.items():
        if not isinstance(value[name], list) or value[name] != expected:
            raise ValueError(f"fisher_boundary.{name} differs from the causal H=1 contract")
    for name in (
        "input_token_ids_sha256",
        "target_token_ids_sha256",
        "fisher_boundary_sha256",
    ):
        _sha256(value[name], name=f"fisher_boundary.{name}")
    expected_input_hash = _fisher_token_ids_sha256(
        [sequence_token_ids[position] for position in input_positions],
        role="input",
    )
    expected_target_hash = _fisher_token_ids_sha256(
        [sequence_token_ids[position] for position in target_positions],
        role="target",
    )
    if value["input_token_ids_sha256"] != expected_input_hash:
        raise ValueError("fisher_boundary input token-ID hash differs from exact sequence tokens")
    if value["target_token_ids_sha256"] != expected_target_hash:
        raise ValueError("fisher_boundary target token-ID hash differs from exact sequence tokens")
    payload = {
        name: value[name] for name in sorted(_FISHER_BOUNDARY_FIELDS - {"fisher_boundary_sha256"})
    }
    expected_self_hash = hashlib.sha256(
        _FISHER_BOUNDARY_HASH_DOMAIN + _resolver_canonical_json_bytes(payload)
    ).hexdigest()
    if value["fisher_boundary_sha256"] != expected_self_hash:
        raise ValueError("fisher_boundary self-hash drifted")


def _normalize_token_span(value: object, *, sequence_length: int) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or set(value) != set(_TOKEN_SPAN_ORDER):
        raise ValueError("token_span must contain the exact six capture span fields")
    normalized: list[tuple[str, int]] = []
    for name in _TOKEN_SPAN_ORDER:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"token_span.{name} must be a non-negative integer")
        normalized.append((name, raw))
    span = dict(normalized)
    if (
        span["prefill_start"] != 0
        or span["prefill_stop"] != span["scored_start"]
        or span["prefill_stop"] < 1
        or span["scored_stop"] < span["scored_start"]
        or span["scored_stop"] != sequence_length
    ):
        raise ValueError("token_span is not contiguous and canonical")
    if (
        span["cache_exposed_start"] != span["scored_stop"]
        or span["cache_exposed_stop"] != span["scored_stop"]
    ):
        raise ValueError("calibration cache-exposed span must be empty at continuation stop")
    return tuple(normalized)


def identity_anchor_manifest_sha256(
    *,
    canonical_id: str,
    sequence_length: int,
    sequence_token_ids_sha256_value: str,
    token_span: tuple[tuple[str, int], ...] | Mapping[str, int],
) -> str:
    """Recompute the exact compact-JSON anchor hash emitted by capture."""

    normalized_id = _canonical_text(canonical_id, name="canonical_id")
    length = _strict_positive_int(sequence_length, name="sequence_length")
    token_hash = _sha256(
        sequence_token_ids_sha256_value,
        name="sequence_token_ids_sha256",
    )
    span = _normalize_token_span(dict(token_span), sequence_length=length)
    manifest = {
        "canonical_id": normalized_id,
        "positions": list(frozen_anchor_positions(length)),
        "sequence_token_ids_sha256": token_hash,
        "token_span": dict(span),
    }
    return hashlib.sha256(_resolver_canonical_json_bytes(manifest)).hexdigest()


def identity_record_sha256(record: Mapping[str, object]) -> str:
    """Recompute the domain-separated capture record self-hash."""

    missing = _IDENTITY_RECORD_PAYLOAD_FIELDS - set(record)
    if missing:
        raise ValueError(f"identity record is missing fields: {sorted(missing)}")
    payload = {name: record[name] for name in sorted(_IDENTITY_RECORD_PAYLOAD_FIELDS)}
    return hashlib.sha256(
        _IDENTITY_RECORD_HASH_DOMAIN + _resolver_canonical_json_bytes(payload)
    ).hexdigest()


def calibration_identity_record_manifest_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    """Hash ordered calibration identity-record commitments for score binding."""

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
                    record_hash,
                    name=f"identity manifest record {index} SHA-256",
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
            -1 if item["configured_length"] is None else cast(int, item["configured_length"]),
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
class AnchorDistortionBatch:
    """Unaggregated per-anchor energy and codec MSE for one sequence."""

    family: CalibrationFamily
    config: str
    ruler_category: RulerCategory | None
    canonical_id: str
    seed: int | None
    configured_length: int | None
    token_count: int
    anchor_positions: tuple[int, ...]
    query_energy: torch.Tensor
    q4_mse: torch.Tensor
    q6_mse: torch.Tensor
    q8_mse: torch.Tensor
    sequence_token_ids: tuple[int, ...] | None = None
    identity_record: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CalibrationSequenceScores:
    """CPU-FP64 per-row scores after the frozen within-sequence mean."""

    family: CalibrationFamily
    config: str
    ruler_category: RulerCategory | None
    canonical_id: str
    seed: int | None
    configured_length: int | None
    token_count: int
    anchor_positions: tuple[int, ...]
    anchor_manifest_sha256: str
    anchor_inputs_sha256: str
    sequence_scores_sha256: str
    d4: torch.Tensor
    d6: torch.Tensor
    d8: torch.Tensor
    source_contract: CalibrationSourceTensorContract
    source_shape: tuple[int, ...]
    sequence_token_ids_sha256: str | None = None
    token_span: tuple[tuple[str, int], ...] | None = None
    identity_anchor_manifest_sha256: str | None = None
    identity_record_sha256: str | None = None

    @property
    def row_count(self) -> int:
        return self.d4.numel()

    def identity_tuple(self) -> tuple[object, ...]:
        return (
            self.family,
            self.ruler_category,
            self.config,
            self.canonical_id,
            self.seed,
            self.configured_length,
            self.token_count,
        )

    def manifest_record(self) -> dict[str, object]:
        return {
            "anchor_inputs_sha256": self.anchor_inputs_sha256,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_positions": list(self.anchor_positions),
            "canonical_id": self.canonical_id,
            "config": self.config,
            "family": self.family,
            "ruler_category": self.ruler_category,
            "seed": self.seed,
            "configured_length": self.configured_length,
            "source_shape": list(self.source_shape),
            "source_tensor_contract": self.source_contract.canonical_dict(),
            "sequence_token_ids_sha256": self.sequence_token_ids_sha256,
            "token_span": None if self.token_span is None else dict(self.token_span),
            "identity_anchor_manifest_sha256": self.identity_anchor_manifest_sha256,
            "identity_record_sha256": self.identity_record_sha256,
            "sequence_scores_sha256": self.sequence_scores_sha256,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class _ValidatedIdentityLineage:
    sequence_token_ids_sha256: str
    token_span: tuple[tuple[str, int], ...]
    identity_anchor_manifest_sha256: str
    identity_record_sha256: str


def _validate_frozen_identity_lineage(
    batch: AnchorDistortionBatch | FrozenComparatorEndpointBatch,
) -> _ValidatedIdentityLineage:
    token_ids = batch.sequence_token_ids
    record = batch.identity_record
    if token_ids is None or record is None:
        raise ValueError(
            "frozen reduction requires sequence_token_ids and an exact capture identity_record"
        )
    if not isinstance(token_ids, tuple):
        raise TypeError("frozen sequence_token_ids must be a tuple")
    if not isinstance(record, Mapping):
        raise TypeError("frozen identity_record must be a mapping")
    allowed_fields = _IDENTITY_RECORD_PAYLOAD_FIELDS | {
        "identity_record_sha256",
        "anchor_positions",
        "anchor_positions_sha256",
    }
    if set(record) not in (
        _IDENTITY_RECORD_PAYLOAD_FIELDS | {"identity_record_sha256"},
        allowed_fields,
    ):
        missing = sorted(
            (_IDENTITY_RECORD_PAYLOAD_FIELDS | {"identity_record_sha256"}) - set(record)
        )
        extra = sorted(set(record) - allowed_fields)
        raise ValueError(f"identity_record fields drifted; missing={missing}, extra={extra}")

    token_hash = sequence_token_ids_sha256(token_ids)
    if len(token_ids) != batch.token_count or record["sequence_length"] != batch.token_count:
        raise ValueError("identity sequence length differs from the processed token count")
    if record["sequence_token_ids_sha256"] != token_hash:
        raise ValueError("sequence token-ID SHA-256 differs from the capture identity")
    _validate_fisher_boundary(
        record["fisher_boundary"],
        sequence_token_ids=token_ids,
    )
    for field, actual in (
        ("family", batch.family),
        ("config", batch.config),
        ("canonical_id", batch.canonical_id),
        ("ruler_category", batch.ruler_category),
        ("seed", batch.seed),
        ("configured_length", batch.configured_length),
    ):
        if record[field] != actual:
            raise ValueError(f"identity_record.{field} differs from the reduction batch")
    span = _normalize_token_span(record["token_span"], sequence_length=batch.token_count)
    computed_anchor_hash = identity_anchor_manifest_sha256(
        canonical_id=batch.canonical_id,
        sequence_length=batch.token_count,
        sequence_token_ids_sha256_value=token_hash,
        token_span=span,
    )
    recorded_anchor_hash = _sha256(
        record["anchor_manifest_sha256"],
        name="identity_record.anchor_manifest_sha256",
    )
    if recorded_anchor_hash != computed_anchor_hash:
        raise ValueError("identity anchor manifest SHA-256 does not recompute")
    recorded_record_hash = _sha256(
        record["identity_record_sha256"],
        name="identity_record.identity_record_sha256",
    )
    computed_record_hash = identity_record_sha256(record)
    if recorded_record_hash != computed_record_hash:
        raise ValueError("identity record SHA-256 does not recompute")

    positions = frozen_anchor_positions(batch.token_count)
    if "anchor_positions" in record:
        if record["anchor_positions"] != list(positions):
            raise ValueError("resolver anchor positions differ from the frozen anchor equation")
        positions_hash = hashlib.sha256(_resolver_canonical_json_bytes(positions)).hexdigest()
        if record["anchor_positions_sha256"] != positions_hash:
            raise ValueError("resolver anchor-position SHA-256 does not recompute")
    return _ValidatedIdentityLineage(
        sequence_token_ids_sha256=token_hash,
        token_span=span,
        identity_anchor_manifest_sha256=computed_anchor_hash,
        identity_record_sha256=computed_record_hash,
    )


def _comparator_expected_positions(
    selector_profile: object,
    token_count: int,
) -> tuple[int, ...]:
    if selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE:
        return frozen_anchor_positions(token_count)
    if selector_profile == FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE:
        return fisher_h1_boundary_positions(token_count)
    raise ValueError("selector_profile must be one of the two frozen comparator methods")


def _comparator_position_payload(
    *,
    selector_profile: UnweightedSelectorProfile,
    token_count: int,
    endpoint_positions: tuple[int, ...],
    sequence_token_ids_sha256_value: str,
    identity_anchor_manifest_sha256_value: str,
    identity_record_sha256_value: str,
    fisher_boundary_sha256: str,
) -> dict[str, object]:
    return {
        "endpoint_positions": list(endpoint_positions),
        "fisher_boundary_sha256": fisher_boundary_sha256,
        "identity_anchor_manifest_sha256": identity_anchor_manifest_sha256_value,
        "identity_record_sha256": identity_record_sha256_value,
        "position_contract": FROZEN_COMPARATOR_POSITION_CONTRACTS[selector_profile],
        "selector_profile": selector_profile,
        "sequence_token_ids_sha256": sequence_token_ids_sha256_value,
        "token_count": token_count,
    }


def _comparator_sequence_score_sha256(
    *,
    selector_profile: UnweightedSelectorProfile,
    position_manifest_sha256: str,
    endpoint_inputs_sha256: str,
    identity_record_sha256_value: str,
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
) -> str:
    metadata = {
        "endpoint_inputs_sha256": endpoint_inputs_sha256,
        "identity_record_sha256": identity_record_sha256_value,
        "position_manifest_sha256": position_manifest_sha256,
        "row_count": d4.numel(),
        "selector_profile": selector_profile,
    }
    return _hash_score_triplet(
        d4,
        d6,
        d8,
        domain=_COMPARATOR_SEQUENCE_SCORE_HASH_DOMAIN,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class ComparatorSequenceScores:
    """One identity-bound sequence reduced to comparator row scores."""

    selector_profile: UnweightedSelectorProfile
    family: CalibrationFamily
    config: str
    ruler_category: RulerCategory | None
    canonical_id: str
    seed: int | None
    configured_length: int | None
    token_count: int
    endpoint_positions: tuple[int, ...]
    position_manifest_sha256: str
    endpoint_inputs_sha256: str
    sequence_scores_sha256: str
    d4: torch.Tensor
    d6: torch.Tensor
    d8: torch.Tensor
    source_shape: tuple[int, ...]
    sequence_token_ids_sha256: str
    token_span: tuple[tuple[str, int], ...]
    identity_anchor_manifest_sha256: str
    identity_record_sha256: str
    fisher_boundary_sha256: str
    target_nlls_sha256: str | None

    @property
    def row_count(self) -> int:
        return self.d4.numel()

    def identity_tuple(self) -> tuple[object, ...]:
        return (
            self.family,
            self.ruler_category,
            self.config,
            self.canonical_id,
            self.seed,
            self.configured_length,
            self.token_count,
        )

    def position_manifest_record(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "config": self.config,
            "configured_length": self.configured_length,
            "family": self.family,
            "identity_record_sha256": self.identity_record_sha256,
            "position_manifest_sha256": self.position_manifest_sha256,
            "ruler_category": self.ruler_category,
            "seed": self.seed,
            "token_count": self.token_count,
        }

    def manifest_record(self) -> dict[str, object]:
        return {
            **self.position_manifest_record(),
            "endpoint_inputs_sha256": self.endpoint_inputs_sha256,
            "fisher_boundary_sha256": self.fisher_boundary_sha256,
            "identity_anchor_manifest_sha256": self.identity_anchor_manifest_sha256,
            "selector_profile": self.selector_profile,
            "sequence_scores_sha256": self.sequence_scores_sha256,
            "sequence_token_ids_sha256": self.sequence_token_ids_sha256,
            "source_shape": list(self.source_shape),
            "target_nlls_sha256": self.target_nlls_sha256,
            "token_span": dict(self.token_span),
        }


def reduce_frozen_comparator_endpoints(
    batch: FrozenComparatorEndpointBatch,
) -> ComparatorSequenceScores:
    """Reduce one strict Identity-v5 comparator endpoint batch on CPU-FP64."""

    if not isinstance(batch, FrozenComparatorEndpointBatch):
        raise TypeError("batch must be a FrozenComparatorEndpointBatch")
    (
        family,
        config,
        ruler_category,
        canonical_id,
        seed,
        configured_length,
        token_count,
    ) = _metadata(
        family=batch.family,
        config=batch.config,
        ruler_category=batch.ruler_category,
        canonical_id=batch.canonical_id,
        seed=batch.seed,
        configured_length=batch.configured_length,
        token_count=batch.token_count,
    )
    expected_positions = _comparator_expected_positions(batch.selector_profile, token_count)
    if not isinstance(batch.endpoint_positions, tuple):
        raise TypeError("endpoint_positions must be a tuple")
    if batch.endpoint_positions != expected_positions:
        raise ValueError(
            "endpoint_positions differ from the selector-specific frozen A(T)/B(T) equation"
        )
    if not isinstance(batch.sequence_token_ids, tuple):
        raise TypeError("sequence_token_ids must be a tuple")
    if not isinstance(batch.identity_record, Mapping):
        raise TypeError("identity_record must be a mapping")
    lineage = _validate_frozen_identity_lineage(batch)

    source_shape = (
        len(expected_positions),
        *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape,
    )
    endpoint_values: list[torch.Tensor] = []
    for name, value in (
        ("q4_scores", batch.q4_scores),
        ("q6_scores", batch.q6_scores),
        ("q8_scores", batch.q8_scores),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != source_shape:
            raise ValueError(f"{name} must have frozen source shape {source_shape}")
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise TypeError(f"{name} must be a CPU torch.float64 tensor")
        endpoint_values.append(
            _cpu_fp64_matrix(
                value,
                name=name,
                anchors=len(expected_positions),
                expected_shape=source_shape,
            )
        )

    target_nlls_sha256: str | None = None
    if batch.selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE:
        if batch.target_nlls is not None:
            raise ValueError("unweighted MSE comparator cannot carry target_nlls")
    else:
        if batch.target_nlls is None:
            raise ValueError("diagonal empirical-Fisher H1 comparator requires target_nlls")
        target_nlls = batch.target_nlls
        if not isinstance(target_nlls, torch.Tensor):
            raise TypeError("target_nlls must be a torch.Tensor or None")
        if (
            target_nlls.device.type != "cpu"
            or target_nlls.dtype != torch.float64
            or tuple(target_nlls.shape) != (len(expected_positions),)
        ):
            raise TypeError(
                "target_nlls must be a CPU torch.float64 vector matching endpoint positions"
            )
        if not torch.isfinite(target_nlls).all().item() or (target_nlls < 0).any().item():
            raise ValueError("target_nlls must contain only finite non-negative values")
        target_nlls_sha256 = hashlib.sha256(
            _COMPARATOR_TARGET_NLL_HASH_DOMAIN + _tensor_bytes(target_nlls)
        ).hexdigest()

    raw_boundary = batch.identity_record["fisher_boundary"]
    assert isinstance(raw_boundary, Mapping)
    fisher_boundary_hash = _sha256(
        raw_boundary["fisher_boundary_sha256"],
        name="identity_record.fisher_boundary.fisher_boundary_sha256",
    )
    position_payload = _comparator_position_payload(
        selector_profile=batch.selector_profile,
        token_count=token_count,
        endpoint_positions=expected_positions,
        sequence_token_ids_sha256_value=lineage.sequence_token_ids_sha256,
        identity_anchor_manifest_sha256_value=lineage.identity_anchor_manifest_sha256,
        identity_record_sha256_value=lineage.identity_record_sha256,
        fisher_boundary_sha256=fisher_boundary_hash,
    )
    position_manifest_sha256 = _domain_json_sha256(
        _COMPARATOR_POSITION_HASH_DOMAIN,
        position_payload,
    )
    endpoint_metadata = {
        "axis_order": list(FROZEN_COMPARATOR_ENDPOINT_AXIS_ORDER),
        "dtype": CALIBRATION_SCORE_DTYPE,
        "position_manifest_sha256": position_manifest_sha256,
        "selector_profile": batch.selector_profile,
        "shape": list(source_shape),
        "target_nlls_sha256": target_nlls_sha256,
    }
    endpoint_digest = hashlib.sha256()
    endpoint_digest.update(_COMPARATOR_ENDPOINT_INPUT_HASH_DOMAIN)
    endpoint_digest.update(canonical_json_bytes(endpoint_metadata))
    for label, value in zip(
        (b"Q4\0", b"Q6\0", b"Q8\0"),
        endpoint_values,
        strict=True,
    ):
        endpoint_digest.update(label)
        endpoint_digest.update(_tensor_bytes(value))
    endpoint_inputs_sha256 = endpoint_digest.hexdigest()
    reduced = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(value.mean(dim=0).contiguous() for value in endpoint_values),
    )
    sequence_scores_sha256 = _comparator_sequence_score_sha256(
        selector_profile=batch.selector_profile,
        position_manifest_sha256=position_manifest_sha256,
        endpoint_inputs_sha256=endpoint_inputs_sha256,
        identity_record_sha256_value=lineage.identity_record_sha256,
        d4=reduced[0],
        d6=reduced[1],
        d8=reduced[2],
    )
    return ComparatorSequenceScores(
        selector_profile=batch.selector_profile,
        family=family,
        config=config,
        ruler_category=ruler_category,
        canonical_id=canonical_id,
        seed=seed,
        configured_length=configured_length,
        token_count=token_count,
        endpoint_positions=expected_positions,
        position_manifest_sha256=position_manifest_sha256,
        endpoint_inputs_sha256=endpoint_inputs_sha256,
        sequence_scores_sha256=sequence_scores_sha256,
        d4=reduced[0],
        d6=reduced[1],
        d8=reduced[2],
        source_shape=source_shape,
        sequence_token_ids_sha256=lineage.sequence_token_ids_sha256,
        token_span=lineage.token_span,
        identity_anchor_manifest_sha256=lineage.identity_anchor_manifest_sha256,
        identity_record_sha256=lineage.identity_record_sha256,
        fisher_boundary_sha256=fisher_boundary_hash,
        target_nlls_sha256=target_nlls_sha256,
    )


def _reduce_anchor_distortions(
    batch: AnchorDistortionBatch,
    *,
    source_contract: CalibrationSourceTensorContract,
    identity_lineage: _ValidatedIdentityLineage | None,
) -> CalibrationSequenceScores:
    """Apply energy weighting after validating a declared source contract."""

    if not isinstance(batch, AnchorDistortionBatch):
        raise TypeError("batch must be an AnchorDistortionBatch")
    (
        family,
        config,
        ruler_category,
        canonical_id,
        seed,
        configured_length,
        token_count,
    ) = _metadata(
        family=batch.family,
        config=batch.config,
        ruler_category=batch.ruler_category,
        canonical_id=batch.canonical_id,
        seed=batch.seed,
        configured_length=batch.configured_length,
        token_count=batch.token_count,
    )
    expected_positions = frozen_anchor_positions(token_count)
    if not isinstance(batch.anchor_positions, tuple):
        raise TypeError("anchor_positions must be a tuple")
    if batch.anchor_positions != expected_positions:
        raise ValueError("anchor_positions differ from the frozen 16-anchor equation")
    if len(batch.anchor_positions) != len(set(batch.anchor_positions)):
        raise ValueError("anchor_positions must be unique")

    anchors = len(expected_positions)
    energy = _cpu_fp64_matrix(batch.query_energy, name="query_energy", anchors=anchors)
    source_shape = tuple(batch.query_energy.shape)
    if source_shape != (anchors, *source_contract.trailing_shape):
        raise ValueError(
            "query_energy shape differs from the declared source tensor contract: "
            f"expected {(anchors, *source_contract.trailing_shape)}, got {source_shape}"
        )
    source_dtypes = tuple((name, str(getattr(batch, name).dtype)) for name in _SOURCE_TENSOR_NAMES)
    if source_dtypes != source_contract.dtypes:
        raise ValueError("source tensor dtypes differ from the declared source tensor contract")
    q4 = _cpu_fp64_matrix(
        batch.q4_mse,
        name="q4_mse",
        anchors=anchors,
        expected_shape=source_shape,
    )
    q6 = _cpu_fp64_matrix(
        batch.q6_mse,
        name="q6_mse",
        anchors=anchors,
        expected_shape=source_shape,
    )
    q8 = _cpu_fp64_matrix(
        batch.q8_mse,
        name="q8_mse",
        anchors=anchors,
        expected_shape=source_shape,
    )

    weighted = tuple((energy * distortion).mean(dim=0) for distortion in (q4, q6, q8))
    if any(not torch.isfinite(value).all().item() for value in weighted):
        raise ValueError("energy-weighted anchor aggregation produced a non-finite score")

    identity = {
        "canonical_id": canonical_id,
        "config": config,
        "family": family,
        "ruler_category": ruler_category,
        "seed": seed,
        "configured_length": configured_length,
        "source_shape": list(source_shape),
        "source_tensor_contract": source_contract.canonical_dict(),
        "sequence_token_ids_sha256": (
            None if identity_lineage is None else identity_lineage.sequence_token_ids_sha256
        ),
        "token_span": (None if identity_lineage is None else dict(identity_lineage.token_span)),
        "identity_anchor_manifest_sha256": (
            None if identity_lineage is None else identity_lineage.identity_anchor_manifest_sha256
        ),
        "identity_record_sha256": (
            None if identity_lineage is None else identity_lineage.identity_record_sha256
        ),
        "token_count": token_count,
    }
    anchor_manifest = {
        **identity,
        "anchor_positions": list(expected_positions),
    }
    anchor_manifest_sha256 = hashlib.sha256(canonical_json_bytes(anchor_manifest)).hexdigest()

    input_digest = hashlib.sha256()
    input_digest.update(_ANCHOR_INPUT_HASH_DOMAIN)
    input_digest.update(canonical_json_bytes(anchor_manifest))
    for label, tensor in zip(
        (b"ENERGY\0", b"MSE4\0", b"MSE6\0", b"MSE8\0"),
        (energy, q4, q6, q8),
        strict=True,
    ):
        input_digest.update(label)
        input_digest.update(_tensor_bytes(tensor))

    score_metadata = {**anchor_manifest, "row_count": energy.shape[1]}
    score_sha256 = _hash_score_triplet(
        *weighted,
        domain=_SCORE_HASH_DOMAIN,
        metadata=score_metadata,
    )
    return CalibrationSequenceScores(
        family=family,
        config=config,
        ruler_category=ruler_category,
        canonical_id=canonical_id,
        seed=seed,
        configured_length=configured_length,
        token_count=token_count,
        anchor_positions=expected_positions,
        anchor_manifest_sha256=anchor_manifest_sha256,
        anchor_inputs_sha256=input_digest.hexdigest(),
        sequence_scores_sha256=score_sha256,
        d4=weighted[0].contiguous(),
        d6=weighted[1].contiguous(),
        d8=weighted[2].contiguous(),
        source_contract=source_contract,
        source_shape=source_shape,
        sequence_token_ids_sha256=(
            None if identity_lineage is None else identity_lineage.sequence_token_ids_sha256
        ),
        token_span=None if identity_lineage is None else identity_lineage.token_span,
        identity_anchor_manifest_sha256=(
            None if identity_lineage is None else identity_lineage.identity_anchor_manifest_sha256
        ),
        identity_record_sha256=(
            None if identity_lineage is None else identity_lineage.identity_record_sha256
        ),
    )


def reduce_anchor_distortions(batch: AnchorDistortionBatch) -> CalibrationSequenceScores:
    """Apply the generic, explicitly shape-bound equal-anchor CPU-FP64 reduction."""

    if not isinstance(batch, AnchorDistortionBatch):
        raise TypeError("batch must be an AnchorDistortionBatch")
    if not isinstance(batch.query_energy, torch.Tensor):
        raise TypeError("query_energy must be a torch.Tensor")
    shape = tuple(batch.query_energy.shape)
    if len(shape) < 2:
        raise ValueError("query_energy must have shape [anchor, row...]")
    axis_order = (
        ("anchor", "flattened_row")
        if len(shape) == 2
        else ("anchor", *(f"row_axis_{index}" for index in range(len(shape) - 1)))
    )
    dtypes: list[tuple[str, str]] = []
    for name in _SOURCE_TENSOR_NAMES:
        value = getattr(batch, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        dtypes.append((name, str(value.dtype)))
    contract = CalibrationSourceTensorContract(
        reduction_profile=GENERIC_REDUCTION_PROFILE,
        axis_order=axis_order,
        trailing_shape=shape[1:],
        dtypes=tuple(dtypes),
    )
    return _reduce_anchor_distortions(
        batch,
        source_contract=contract,
        identity_lineage=None,
    )


def reduce_frozen_anchor_distortions(
    batch: AnchorDistortionBatch,
) -> CalibrationSequenceScores:
    """Apply the official Experiment 013 reduction with exact CPU-FP64 source axes."""

    if not isinstance(batch, AnchorDistortionBatch):
        raise TypeError("batch must be an AnchorDistortionBatch")
    anchors = len(frozen_anchor_positions(batch.token_count))
    expected_shape = (anchors, *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape)
    for name in _SOURCE_TENSOR_NAMES:
        value = getattr(batch, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have frozen source shape {expected_shape}; got {tuple(value.shape)}"
            )
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise TypeError(f"{name} must be a CPU torch.float64 tensor for frozen reduction")
    identity_lineage = _validate_frozen_identity_lineage(batch)
    return _reduce_anchor_distortions(
        batch,
        source_contract=FROZEN_SOURCE_TENSOR_CONTRACT,
        identity_lineage=identity_lineage,
    )


def _sequence_sort_key(sequence: CalibrationSequenceScores) -> tuple[object, ...]:
    return (
        CALIBRATION_FAMILY_ORDER.index(sequence.family),
        "" if sequence.ruler_category is None else sequence.ruler_category,
        sequence.config,
        sequence.canonical_id,
        -1 if sequence.seed is None else sequence.seed,
        -1 if sequence.configured_length is None else sequence.configured_length,
        sequence.token_count,
    )


def _validate_sequence_score(sequence: object, *, expected_rows: int | None) -> None:
    if not isinstance(sequence, CalibrationSequenceScores):
        raise TypeError("sequences must contain CalibrationSequenceScores")
    _metadata(
        family=sequence.family,
        config=sequence.config,
        ruler_category=sequence.ruler_category,
        canonical_id=sequence.canonical_id,
        seed=sequence.seed,
        configured_length=sequence.configured_length,
        token_count=sequence.token_count,
    )
    if sequence.anchor_positions != frozen_anchor_positions(sequence.token_count):
        raise ValueError("sequence anchor positions differ from the frozen equation")
    _sha256(sequence.anchor_manifest_sha256, name="anchor_manifest_sha256")
    _sha256(sequence.anchor_inputs_sha256, name="anchor_inputs_sha256")
    _sha256(sequence.sequence_scores_sha256, name="sequence_scores_sha256")
    if not isinstance(sequence.source_contract, CalibrationSourceTensorContract):
        raise TypeError("source_contract must be a CalibrationSourceTensorContract")
    if not isinstance(sequence.source_shape, tuple) or len(sequence.source_shape) != len(
        sequence.source_contract.axis_order
    ):
        raise ValueError("source_shape must match the declared source axis order")
    for index, extent in enumerate(sequence.source_shape):
        _strict_positive_int(extent, name=f"source_shape[{index}]")
    if sequence.source_shape != (
        len(sequence.anchor_positions),
        *sequence.source_contract.trailing_shape,
    ):
        raise ValueError("source_shape differs from anchor count or source tensor contract")
    rows = expected_rows if expected_rows is not None else sequence.d4.numel()
    if math.prod(sequence.source_contract.trailing_shape) != rows:
        raise ValueError("source tensor contract does not flatten to the score row count")
    lineage_values = (
        sequence.sequence_token_ids_sha256,
        sequence.token_span,
        sequence.identity_anchor_manifest_sha256,
        sequence.identity_record_sha256,
    )
    if sequence.source_contract == FROZEN_SOURCE_TENSOR_CONTRACT:
        if any(value is None for value in lineage_values):
            raise ValueError("frozen sequence scores require complete capture identity lineage")
        token_hash = _sha256(
            sequence.sequence_token_ids_sha256,
            name="sequence_token_ids_sha256",
        )
        span = _normalize_token_span(
            dict(cast(tuple[tuple[str, int], ...], sequence.token_span)),
            sequence_length=sequence.token_count,
        )
        anchor_hash = identity_anchor_manifest_sha256(
            canonical_id=sequence.canonical_id,
            sequence_length=sequence.token_count,
            sequence_token_ids_sha256_value=token_hash,
            token_span=span,
        )
        if anchor_hash != sequence.identity_anchor_manifest_sha256:
            raise ValueError("identity anchor manifest SHA-256 does not match sequence lineage")
        _sha256(sequence.identity_record_sha256, name="identity_record_sha256")
    elif any(value is not None for value in lineage_values):
        raise ValueError("generic sequence scores cannot claim frozen capture identity lineage")
    values = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=rows)
        for name, value in (("D4", sequence.d4), ("D6", sequence.d6), ("D8", sequence.d8))
    )
    score_metadata = {
        "anchor_positions": list(sequence.anchor_positions),
        "canonical_id": sequence.canonical_id,
        "config": sequence.config,
        "family": sequence.family,
        "ruler_category": sequence.ruler_category,
        "row_count": rows,
        "seed": sequence.seed,
        "configured_length": sequence.configured_length,
        "source_shape": list(sequence.source_shape),
        "source_tensor_contract": sequence.source_contract.canonical_dict(),
        "sequence_token_ids_sha256": sequence.sequence_token_ids_sha256,
        "token_span": None if sequence.token_span is None else dict(sequence.token_span),
        "identity_anchor_manifest_sha256": sequence.identity_anchor_manifest_sha256,
        "identity_record_sha256": sequence.identity_record_sha256,
        "token_count": sequence.token_count,
    }
    computed = _hash_score_triplet(
        *values,
        domain=_SCORE_HASH_DOMAIN,
        metadata=score_metadata,
    )
    if computed != sequence.sequence_scores_sha256:
        raise ValueError("sequence_scores_sha256 does not match the score arrays")


@dataclass(frozen=True, slots=True)
class CalibrationAggregate:
    """Frozen broad-family macro scores and their ordered source manifest."""

    d4: torch.Tensor
    d6: torch.Tensor
    d8: torch.Tensor
    family_sequence_counts: tuple[tuple[str, int], ...]
    ruler_category_sequence_counts: tuple[tuple[str, int], ...]
    sequence_score_manifest_sha256: str
    source_contract: CalibrationSourceTensorContract
    identity_record_manifest_sha256: str | None = None

    @property
    def row_count(self) -> int:
        return self.d4.numel()

    def scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.d4, self.d6, self.d8


def _mean_score_rows(rows: list[torch.Tensor], *, context: str) -> torch.Tensor:
    if not rows:
        raise ValueError(f"{context} cannot be empty")
    result = torch.stack(rows, dim=0).mean(dim=0)
    if result.dtype != torch.float64 or result.device.type != "cpu":
        raise RuntimeError("calibration aggregation left the CPU-FP64 contract")
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{context} produced a non-finite aggregate")
    return result.contiguous()


def _family_balanced_score_aggregate(
    ordered: Sequence[CalibrationSequenceScores | ComparatorSequenceScores],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int], ...],
]:
    """Apply one shared MBPP/PG19/four-category-RULER macro equation."""

    grouped = {
        family: [sequence for sequence in ordered if sequence.family == family]
        for family in CALIBRATION_FAMILY_ORDER
    }
    if any(not grouped[family] for family in CALIBRATION_FAMILY_ORDER):
        raise ValueError("MBPP, PG19, and RULER must each contain at least one sequence")
    ruler_categories = tuple(
        category
        for category in RULER_CATEGORY_ORDER
        if any(sequence.ruler_category == category for sequence in grouped["ruler"])
    )
    if ruler_categories != RULER_CATEGORY_ORDER:
        raise ValueError("RULER calibration must contain all four frozen categories")

    aggregate_by_bit: list[torch.Tensor] = []
    for attribute in ("d4", "d6", "d8"):
        mbpp = _mean_score_rows(
            [getattr(sequence, attribute) for sequence in grouped["mbpp"]],
            context=f"MBPP {attribute}",
        )
        pg19 = _mean_score_rows(
            [getattr(sequence, attribute) for sequence in grouped["pg19"]],
            context=f"PG19 {attribute}",
        )
        ruler_category_means = [
            _mean_score_rows(
                [
                    getattr(sequence, attribute)
                    for sequence in grouped["ruler"]
                    if sequence.ruler_category == category
                ],
                context=f"RULER {category} {attribute}",
            )
            for category in ruler_categories
        ]
        ruler = _mean_score_rows(ruler_category_means, context=f"RULER macro {attribute}")
        aggregate_by_bit.append(
            _mean_score_rows([mbpp, pg19, ruler], context=f"broad-family macro {attribute}")
        )
    family_counts = tuple((family, len(grouped[family])) for family in CALIBRATION_FAMILY_ORDER)
    ruler_counts = tuple(
        (
            category,
            sum(sequence.ruler_category == category for sequence in grouped["ruler"]),
        )
        for category in ruler_categories
    )
    return (
        cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(aggregate_by_bit),
        ),
        family_counts,
        ruler_counts,
    )


def aggregate_calibration_scores(
    sequences: list[CalibrationSequenceScores] | tuple[CalibrationSequenceScores, ...],
) -> CalibrationAggregate:
    """Compute the frozen MBPP + PG19 + equal-RULER-category macro."""

    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError("sequences must be a non-empty list or tuple")
    ordered = sorted(sequences, key=_sequence_sort_key)
    expected_rows: int | None = None
    identities: set[tuple[object, ...]] = set()
    for sequence in ordered:
        _validate_sequence_score(sequence, expected_rows=expected_rows)
        expected_rows = sequence.row_count if expected_rows is None else expected_rows
        identity = sequence.identity_tuple()
        if identity in identities:
            raise ValueError(f"duplicate calibration sequence identity: {identity!r}")
        identities.add(identity)
    assert expected_rows is not None
    source_contracts = {sequence.source_contract for sequence in ordered}
    if len(source_contracts) != 1:
        raise ValueError("all calibration sequences must use one source tensor contract")
    source_contract = next(iter(source_contracts))
    if source_contract == FROZEN_SOURCE_TENSOR_CONTRACT:
        lineage_records = [
            {
                "family": sequence.family,
                "ruler_category": sequence.ruler_category,
                "config": sequence.config,
                "canonical_id": sequence.canonical_id,
                "seed": sequence.seed,
                "configured_length": sequence.configured_length,
                "sequence_length": sequence.token_count,
                "identity_record_sha256": sequence.identity_record_sha256,
            }
            for sequence in ordered
        ]
        identity_manifest_sha256 = calibration_identity_record_manifest_sha256(lineage_records)
    else:
        identity_manifest_sha256 = None

    aggregate_by_bit, family_counts, ruler_counts = _family_balanced_score_aggregate(ordered)

    manifest = [sequence.manifest_record() for sequence in ordered]
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return CalibrationAggregate(
        d4=aggregate_by_bit[0],
        d6=aggregate_by_bit[1],
        d8=aggregate_by_bit[2],
        family_sequence_counts=family_counts,
        ruler_category_sequence_counts=ruler_counts,
        sequence_score_manifest_sha256=manifest_sha256,
        source_contract=source_contract,
        identity_record_manifest_sha256=identity_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class ComparatorAggregate:
    """Family-balanced score arrays and complete hashed comparator lineage."""

    selector_profile: UnweightedSelectorProfile
    d4: torch.Tensor
    d6: torch.Tensor
    d8: torch.Tensor
    family_sequence_counts: tuple[tuple[str, int], ...]
    ruler_category_sequence_counts: tuple[tuple[str, int], ...]
    position_manifest_sha256: str
    sequence_score_manifest_sha256: str
    identity_record_manifest_sha256: str
    aggregate_scores_sha256: str

    @property
    def row_count(self) -> int:
        return self.d4.numel()

    def scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.d4, self.d6, self.d8


def _comparator_sequence_sort_key(sequence: ComparatorSequenceScores) -> tuple[object, ...]:
    return (
        CALIBRATION_FAMILY_ORDER.index(sequence.family),
        "" if sequence.ruler_category is None else sequence.ruler_category,
        sequence.config,
        sequence.canonical_id,
        -1 if sequence.seed is None else sequence.seed,
        -1 if sequence.configured_length is None else sequence.configured_length,
        sequence.token_count,
    )


def _validate_comparator_sequence_score(
    sequence: object,
    *,
    expected_rows: int | None,
) -> ComparatorSequenceScores:
    if not isinstance(sequence, ComparatorSequenceScores):
        raise TypeError("sequences must contain ComparatorSequenceScores")
    _metadata(
        family=sequence.family,
        config=sequence.config,
        ruler_category=sequence.ruler_category,
        canonical_id=sequence.canonical_id,
        seed=sequence.seed,
        configured_length=sequence.configured_length,
        token_count=sequence.token_count,
    )
    expected_positions = _comparator_expected_positions(
        sequence.selector_profile,
        sequence.token_count,
    )
    if sequence.endpoint_positions != expected_positions:
        raise ValueError("comparator sequence positions drifted from frozen A(T)/B(T)")
    expected_shape = (
        len(expected_positions),
        *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape,
    )
    if sequence.source_shape != expected_shape:
        raise ValueError("comparator sequence source shape drifted")
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("comparator sequence row count differs")
    for name, value in (("D4", sequence.d4), ("D6", sequence.d6), ("D8", sequence.d8)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or tuple(value.shape) != (rows,)
            or not value.is_contiguous()
        ):
            raise TypeError(f"{name} must be a contiguous CPU torch.float64 row vector")
    d4, d6, d8 = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(
            _cpu_fp64_scores(value, name=name, expected_rows=rows)
            for name, value in (
                ("D4", sequence.d4),
                ("D6", sequence.d6),
                ("D8", sequence.d8),
            )
        ),
    )
    token_hash = _sha256(
        sequence.sequence_token_ids_sha256,
        name="sequence_token_ids_sha256",
    )
    span = _normalize_token_span(
        dict(sequence.token_span),
        sequence_length=sequence.token_count,
    )
    identity_anchor_hash = identity_anchor_manifest_sha256(
        canonical_id=sequence.canonical_id,
        sequence_length=sequence.token_count,
        sequence_token_ids_sha256_value=token_hash,
        token_span=span,
    )
    if identity_anchor_hash != sequence.identity_anchor_manifest_sha256:
        raise ValueError("comparator identity anchor manifest SHA-256 drifted")
    identity_record_hash = _sha256(
        sequence.identity_record_sha256,
        name="identity_record_sha256",
    )
    fisher_boundary_hash = _sha256(
        sequence.fisher_boundary_sha256,
        name="fisher_boundary_sha256",
    )
    if sequence.target_nlls_sha256 is not None:
        _sha256(sequence.target_nlls_sha256, name="target_nlls_sha256")
    if (
        sequence.selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE
        and sequence.target_nlls_sha256 is not None
    ):
        raise ValueError("unweighted MSE sequence cannot claim target NLL input")
    if (
        sequence.selector_profile == FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE
        and sequence.target_nlls_sha256 is None
    ):
        raise ValueError("diagonal empirical-Fisher H1 sequence requires a target NLL receipt")
    endpoint_hash = _sha256(
        sequence.endpoint_inputs_sha256,
        name="endpoint_inputs_sha256",
    )
    position_payload = _comparator_position_payload(
        selector_profile=sequence.selector_profile,
        token_count=sequence.token_count,
        endpoint_positions=expected_positions,
        sequence_token_ids_sha256_value=token_hash,
        identity_anchor_manifest_sha256_value=identity_anchor_hash,
        identity_record_sha256_value=identity_record_hash,
        fisher_boundary_sha256=fisher_boundary_hash,
    )
    expected_position_hash = _domain_json_sha256(
        _COMPARATOR_POSITION_HASH_DOMAIN,
        position_payload,
    )
    if sequence.position_manifest_sha256 != expected_position_hash:
        raise ValueError("comparator position-manifest SHA-256 drifted")
    expected_score_hash = _comparator_sequence_score_sha256(
        selector_profile=sequence.selector_profile,
        position_manifest_sha256=expected_position_hash,
        endpoint_inputs_sha256=endpoint_hash,
        identity_record_sha256_value=identity_record_hash,
        d4=d4,
        d6=d6,
        d8=d8,
    )
    if sequence.sequence_scores_sha256 != expected_score_hash:
        raise ValueError("comparator sequence-score SHA-256 drifted")
    return sequence


def _comparator_aggregate_score_sha256(
    aggregate: ComparatorAggregate,
) -> str:
    metadata = {
        "family_sequence_counts": [
            {"count": count, "family": family} for family, count in aggregate.family_sequence_counts
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


def aggregate_comparator_scores(
    sequences: list[ComparatorSequenceScores] | tuple[ComparatorSequenceScores, ...],
) -> ComparatorAggregate:
    """Aggregate one comparator with the candidate's exact family-balanced macro."""

    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError("sequences must be a non-empty list or tuple")
    if any(not isinstance(sequence, ComparatorSequenceScores) for sequence in sequences):
        raise TypeError("sequences must contain ComparatorSequenceScores")
    selector_profiles = {sequence.selector_profile for sequence in sequences}
    if len(selector_profiles) != 1:
        raise ValueError("one comparator aggregate cannot mix selector profiles")
    selector_profile = next(iter(selector_profiles))
    _comparator_expected_positions(selector_profile, 3)
    expected_rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    for sequence in sequences:
        _validate_comparator_sequence_score(sequence, expected_rows=expected_rows)
    ordered = sorted(sequences, key=_comparator_sequence_sort_key)
    identities: set[tuple[object, ...]] = set()
    for sequence in ordered:
        identity = sequence.identity_tuple()
        if identity in identities:
            raise ValueError(f"duplicate comparator sequence identity: {identity!r}")
        identities.add(identity)

    aggregate_scores, family_counts, ruler_counts = _family_balanced_score_aggregate(ordered)
    identity_records = [
        {
            "canonical_id": sequence.canonical_id,
            "config": sequence.config,
            "configured_length": sequence.configured_length,
            "family": sequence.family,
            "identity_record_sha256": sequence.identity_record_sha256,
            "ruler_category": sequence.ruler_category,
            "seed": sequence.seed,
            "sequence_length": sequence.token_count,
        }
        for sequence in ordered
    ]
    identity_manifest_hash = calibration_identity_record_manifest_sha256(identity_records)
    position_manifest_hash = _domain_json_sha256(
        _COMPARATOR_POSITION_MANIFEST_HASH_DOMAIN,
        [sequence.position_manifest_record() for sequence in ordered],
    )
    sequence_manifest_hash = _domain_json_sha256(
        _COMPARATOR_SEQUENCE_MANIFEST_HASH_DOMAIN,
        [sequence.manifest_record() for sequence in ordered],
    )
    provisional = ComparatorAggregate(
        selector_profile=cast(UnweightedSelectorProfile, selector_profile),
        d4=aggregate_scores[0],
        d6=aggregate_scores[1],
        d8=aggregate_scores[2],
        family_sequence_counts=family_counts,
        ruler_category_sequence_counts=ruler_counts,
        position_manifest_sha256=position_manifest_hash,
        sequence_score_manifest_sha256=sequence_manifest_hash,
        identity_record_manifest_sha256=identity_manifest_hash,
        aggregate_scores_sha256="0" * 64,
    )
    return ComparatorAggregate(
        selector_profile=provisional.selector_profile,
        d4=provisional.d4,
        d6=provisional.d6,
        d8=provisional.d8,
        family_sequence_counts=provisional.family_sequence_counts,
        ruler_category_sequence_counts=provisional.ruler_category_sequence_counts,
        position_manifest_sha256=provisional.position_manifest_sha256,
        sequence_score_manifest_sha256=provisional.sequence_score_manifest_sha256,
        identity_record_manifest_sha256=provisional.identity_record_manifest_sha256,
        aggregate_scores_sha256=_comparator_aggregate_score_sha256(provisional),
    )


def calibration_sequence_rank_sha256(sequence: CalibrationSequenceScores) -> str:
    """Hash the exact domain-separated identity used by the frozen resolver."""

    _validate_sequence_score(sequence, expected_rows=None)
    return _calibration_sequence_rank_sha256_unchecked(sequence)


def _calibration_sequence_rank_sha256_unchecked(
    sequence: CalibrationSequenceScores,
) -> str:
    identity = "\0".join(
        (
            sequence.family,
            str(sequence.ruler_category),
            sequence.config,
            sequence.canonical_id,
            str(sequence.seed),
            str(sequence.configured_length),
            str(sequence.token_count),
        )
    )
    return hashlib.sha256(
        CALIBRATION_SPLIT_NAMESPACE.encode("utf-8") + identity.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    group: str
    canonical_id: str
    config: str
    ruler_category: RulerCategory | None
    configured_length: int | None
    sequence_length: int
    seed: int | None
    rank: int
    half: SplitHalf
    rank_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "config": self.config,
            "configured_length": self.configured_length,
            "group": self.group,
            "half": self.half,
            "rank": self.rank,
            "rank_sha256": self.rank_sha256,
            "ruler_category": self.ruler_category,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True, slots=True)
class CalibrationHalfSplit:
    half_a: tuple[CalibrationSequenceScores, ...]
    half_b: tuple[CalibrationSequenceScores, ...]
    assignments: tuple[SplitAssignment, ...]
    assignment_sha256: str


def balanced_sha_rank_halves(
    sequences: list[CalibrationSequenceScores] | tuple[CalibrationSequenceScores, ...],
) -> CalibrationHalfSplit:
    """Alternate SHA-ranked identities within MBPP, PG19, and each RULER family."""

    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError("sequences must be a non-empty list or tuple")
    ordered = sorted(sequences, key=_sequence_sort_key)
    expected_rows: int | None = None
    identities: set[tuple[object, ...]] = set()
    for sequence in ordered:
        _validate_sequence_score(sequence, expected_rows=expected_rows)
        expected_rows = sequence.row_count if expected_rows is None else expected_rows
        if sequence.identity_tuple() in identities:
            raise ValueError("calibration split contains a duplicate sequence identity")
        identities.add(sequence.identity_tuple())

    rank_hashes = {
        sequence.identity_tuple(): _calibration_sequence_rank_sha256_unchecked(sequence)
        for sequence in ordered
    }

    ruler_categories = tuple(
        category
        for category in RULER_CATEGORY_ORDER
        if any(
            sequence.family == "ruler" and sequence.ruler_category == category
            for sequence in ordered
        )
    )
    if ruler_categories != RULER_CATEGORY_ORDER:
        raise ValueError("calibration split requires all four frozen RULER categories")
    groups = (
        ("mbpp", [sequence for sequence in ordered if sequence.family == "mbpp"]),
        ("pg19", [sequence for sequence in ordered if sequence.family == "pg19"]),
        *(
            (
                f"ruler:{category}",
                [
                    sequence
                    for sequence in ordered
                    if sequence.family == "ruler" and sequence.ruler_category == category
                ],
            )
            for category in ruler_categories
        ),
    )

    half_a: list[CalibrationSequenceScores] = []
    half_b: list[CalibrationSequenceScores] = []
    assignments: list[SplitAssignment] = []
    for group, members in groups:
        if len(members) < 2:
            raise ValueError(f"split group {group!r} needs at least two sequences")
        ranked = sorted(
            members,
            key=lambda sequence: (
                rank_hashes[sequence.identity_tuple()],
                sequence.canonical_id,
            ),
        )
        for rank, sequence in enumerate(ranked):
            half: SplitHalf = "a" if rank % 2 == 0 else "b"
            (half_a if half == "a" else half_b).append(sequence)
            assignments.append(
                SplitAssignment(
                    group=group,
                    canonical_id=sequence.canonical_id,
                    config=sequence.config,
                    ruler_category=sequence.ruler_category,
                    configured_length=sequence.configured_length,
                    sequence_length=sequence.token_count,
                    seed=sequence.seed,
                    rank=rank,
                    half=half,
                    rank_sha256=rank_hashes[sequence.identity_tuple()],
                )
            )

    assignments.sort(key=lambda item: (item.group, item.rank))
    assignment_payload = [item.canonical_dict() for item in assignments]
    assignment_sha256 = hashlib.sha256(
        _resolver_canonical_json_bytes(assignment_payload)
    ).hexdigest()
    return CalibrationHalfSplit(
        half_a=tuple(sorted(half_a, key=_sequence_sort_key)),
        half_b=tuple(sorted(half_b, key=_sequence_sort_key)),
        assignments=tuple(assignments),
        assignment_sha256=assignment_sha256,
    )


def allocate_static_q468_code_map(
    aggregate: CalibrationAggregate,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Feed aggregate D4/D6/D8 directly to the existing exact allocator."""

    if not isinstance(aggregate, CalibrationAggregate):
        raise TypeError("aggregate must be a CalibrationAggregate")
    rows = aggregate.row_count
    steps = _strict_nonnegative_int(marginal_steps, name="marginal_steps")
    if steps > 2 * rows:
        raise ValueError("marginal_steps exceeds two steps per row")
    scores = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=rows).reshape(1, rows)
        for name, value in (("D4", aggregate.d4), ("D6", aggregate.d6), ("D8", aggregate.d8))
    )
    result = allocate_exact_multibit_codes_fast(*scores, marginal_steps=steps).reshape(-1)
    if result.dtype != torch.uint8 or result.device.type != "cpu":
        raise RuntimeError("exact allocator returned a non-canonical code map")
    if int(result.to(torch.int64).sum().item()) != steps:
        raise RuntimeError("exact allocator did not satisfy the requested marginal budget")
    return result.contiguous()


def allocate_frozen_static_q468_code_maps(
    aggregate: CalibrationAggregate,
) -> dict[int, torch.Tensor]:
    """Allocate the frozen K27030 diagnostic and K29334 primary maps."""

    if aggregate.row_count != FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows:
        raise ValueError(
            f"frozen Q468 maps require exactly {FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows} rows"
        )
    return {
        steps: allocate_static_q468_code_map(aggregate, marginal_steps=steps)
        for steps in (
            FROZEN_STATIC_Q468_ABLATION_STEPS,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
        )
    }


def _precision_codes(value: object, *, name: str, expected_rows: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must use torch.uint8")
    normalized = value.detach().to("cpu").reshape(-1).contiguous()
    if normalized.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if expected_rows is not None and normalized.numel() != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} codes")
    if (normalized > 2).any().item():
        raise ValueError(f"{name} may contain only Q4/Q6/Q8 codes 0, 1, and 2")
    return normalized.clone()


def q8_set_jaccard(left_codes: torch.Tensor, right_codes: torch.Tensor) -> float:
    """Jaccard similarity of code-2 sets; two empty sets are defined as 1."""

    left = _precision_codes(left_codes, name="left_codes")
    right = _precision_codes(right_codes, name="right_codes", expected_rows=left.numel())
    left_q8 = left == 2
    right_q8 = right == 2
    union = int(torch.logical_or(left_q8, right_q8).sum().item())
    if union == 0:
        return 1.0
    intersection = int(torch.logical_and(left_q8, right_q8).sum().item())
    return intersection / union


def per_layer_mean_bitwidth_shifts(
    left_codes: torch.Tensor,
    right_codes: torch.Tensor,
    *,
    layer_indices: tuple[int, ...],
    rows_per_layer: int,
) -> tuple[tuple[int, float], ...]:
    """Return absolute shifts between layer mean bitwidths in actual bits.

    A precision code is a two-bit step above Q4, so the mean-bitwidth shift is
    ``2 * abs(mean(code_A) - mean(code_B))``.  It is not the mean per-row
    churn; Q8-set Jaccard separately measures set overlap.
    """

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
    left_layers = left.to(torch.float64).reshape(len(layer_indices), layer_rows)
    right_layers = right.to(torch.float64).reshape(len(layer_indices), layer_rows)
    shifts = 2.0 * (left_layers.mean(dim=1) - right_layers.mean(dim=1)).abs()
    return tuple(
        (layer_index, float(shift.item()))
        for layer_index, shift in zip(layer_indices, shifts, strict=True)
    )


@dataclass(frozen=True, slots=True)
class PolicyStabilityResult:
    """Conjunctive split-half stability result for one exact-K policy."""

    passed: bool
    spearman_average_ties: float | None
    q8_jaccard: float
    layer_mean_bitwidth_shifts: tuple[tuple[int, float], ...]
    checks: tuple[tuple[str, bool], ...]

    @property
    def max_layer_mean_bitwidth_shift(self) -> float:
        return max(shift for _layer, shift in self.layer_mean_bitwidth_shifts)


def evaluate_policy_stability(
    half_a_codes: torch.Tensor,
    half_b_codes: torch.Tensor,
    *,
    layer_indices: tuple[int, ...],
    rows_per_layer: int,
    expected_marginal_steps: int | None = None,
) -> PolicyStabilityResult:
    """Evaluate the frozen Spearman, Q8-Jaccard, and layer-shift conjunction.

    Spearman uses average (mid-)ranks for exact ties.  A constant code vector
    has undefined correlation and therefore fails closed instead of receiving
    an arbitrary correlation.
    """

    rows = len(layer_indices) * _strict_positive_int(rows_per_layer, name="rows_per_layer")
    left = _precision_codes(half_a_codes, name="half_a_codes", expected_rows=rows)
    right = _precision_codes(half_b_codes, name="half_b_codes", expected_rows=rows)
    if expected_marginal_steps is not None:
        steps = _strict_nonnegative_int(
            expected_marginal_steps,
            name="expected_marginal_steps",
        )
        if steps > 2 * rows:
            raise ValueError("expected_marginal_steps exceeds two steps per row")
        if int(left.to(torch.int64).sum().item()) != steps:
            raise ValueError("half A code map does not satisfy the expected exact-K budget")
        if int(right.to(torch.int64).sum().item()) != steps:
            raise ValueError("half B code map does not satisfy the expected exact-K budget")

    try:
        spearman = spearman_correlation(
            left.to(torch.float64).tolist(),
            right.to(torch.float64).tolist(),
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


@dataclass(frozen=True, slots=True)
class SplitHalfPolicyFit:
    split: CalibrationHalfSplit
    half_a_aggregate: CalibrationAggregate
    half_b_aggregate: CalibrationAggregate
    half_a_codes: torch.Tensor
    half_b_codes: torch.Tensor
    stability: PolicyStabilityResult


def fit_split_half_policy(
    sequences: list[CalibrationSequenceScores] | tuple[CalibrationSequenceScores, ...],
    *,
    layer_indices: tuple[int, ...],
    rows_per_layer: int,
    marginal_steps: int,
) -> SplitHalfPolicyFit:
    """Split, independently aggregate, allocate exact-K maps, and gate them."""

    split = balanced_sha_rank_halves(sequences)
    half_a = aggregate_calibration_scores(split.half_a)
    half_b = aggregate_calibration_scores(split.half_b)
    codes_a = allocate_static_q468_code_map(half_a, marginal_steps=marginal_steps)
    codes_b = allocate_static_q468_code_map(half_b, marginal_steps=marginal_steps)
    stability = evaluate_policy_stability(
        codes_a,
        codes_b,
        layer_indices=layer_indices,
        rows_per_layer=rows_per_layer,
        expected_marginal_steps=marginal_steps,
    )
    return SplitHalfPolicyFit(
        split=split,
        half_a_aggregate=half_a,
        half_b_aggregate=half_b,
        half_a_codes=codes_a,
        half_b_codes=codes_b,
        stability=stability,
    )


def static_q468_code_map_sha256(
    codes: torch.Tensor,
    *,
    geometry: StaticRhtQ468Geometry,
    marginal_steps: int,
) -> str:
    """Return the code-map hash used by ``StaticRhtQ468Policy``."""

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    normalized = _precision_codes(codes, name="codes", expected_rows=geometry.total_rows)
    steps = _strict_nonnegative_int(marginal_steps, name="marginal_steps")
    if int(normalized.to(torch.int64).sum().item()) != steps:
        raise ValueError("code-map marginal sum does not match marginal_steps")
    packed = _pack_precision_codes(normalized)
    digest = hashlib.sha256()
    digest.update(_CODE_MAP_HASH_DOMAIN)
    digest.update(bytes.fromhex(geometry.geometry_sha256))
    digest.update(steps.to_bytes(8, "little", signed=False))
    digest.update(packed.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_aggregate(aggregate: object, *, expected_rows: int) -> CalibrationAggregate:
    if not isinstance(aggregate, CalibrationAggregate):
        raise TypeError("aggregate must be a CalibrationAggregate")
    values = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=expected_rows)
        for name, value in (("D4", aggregate.d4), ("D6", aggregate.d6), ("D8", aggregate.d8))
    )
    if not isinstance(aggregate.family_sequence_counts, tuple):
        raise ValueError("family_sequence_counts must be a tuple")
    expected_families = tuple(name for name, _count in aggregate.family_sequence_counts)
    if expected_families != CALIBRATION_FAMILY_ORDER:
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


def _budget_tuple(values: object, *, total_rows: int) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("marginal_steps must be a non-empty list or tuple")
    normalized = tuple(
        _strict_nonnegative_int(value, name="marginal_steps entry") for value in values
    )
    if any(value > 2 * total_rows for value in normalized):
        raise ValueError("a marginal_steps entry exceeds two steps per row")
    if len(set(normalized)) != len(normalized):
        raise ValueError("marginal_steps entries must be unique")
    return tuple(sorted(normalized))


def _score_data_b64(scores: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> str:
    raw = b"".join(_tensor_bytes(score) for score in scores)
    return base64.b64encode(raw).decode("ascii")


def _allocation_record(
    aggregate: CalibrationAggregate,
    *,
    geometry: StaticRhtQ468Geometry,
    marginal_steps: int,
) -> tuple[dict[str, object], torch.Tensor]:
    codes = allocate_static_q468_code_map(aggregate, marginal_steps=marginal_steps)
    counts = [int((codes == code).sum().item()) for code in range(3)]
    return (
        {
            "allocator_revision": STATIC_Q468_ALLOCATOR_REVISION,
            "code_counts_q4_q6_q8": counts,
            "code_map_sha256": static_q468_code_map_sha256(
                codes,
                geometry=geometry,
                marginal_steps=marginal_steps,
            ),
            "marginal_steps": marginal_steps,
            "packed_precision_bytes": math.ceil(geometry.total_rows * 2 / 8),
        },
        codes,
    )


def _build_calibration_score_artifact(
    aggregate: CalibrationAggregate,
    *,
    geometry: StaticRhtQ468Geometry,
    calibration_identity_sha256: str,
    marginal_steps: tuple[int, ...] | list[int],
    artifact_kind: str,
    artifact_profile: str,
    artifact_revision: str,
) -> bytes:
    """Build canonical, hash-bound score evidence for one explicit profile."""

    if not isinstance(geometry, StaticRhtQ468Geometry):
        raise TypeError("geometry must be a StaticRhtQ468Geometry")
    normalized = _validate_aggregate(aggregate, expected_rows=geometry.total_rows)
    identity_sha256 = _sha256(
        calibration_identity_sha256,
        name="calibration_identity_sha256",
    )
    budgets = _budget_tuple(marginal_steps, total_rows=geometry.total_rows)
    score_sha256 = static_q468_distortion_sha256(
        *normalized.scores(),
        geometry=geometry,
    )
    allocation_records = [
        _allocation_record(normalized, geometry=geometry, marginal_steps=steps)[0]
        for steps in budgets
    ]
    evidence = {
        "aggregation_contract": _AGGREGATION_CONTRACT,
        "allocations": allocation_records,
        "artifact_profile": artifact_profile,
        "artifact_revision": artifact_revision,
        "calibration_identity_sha256": identity_sha256,
        "calibration_scores_sha256": score_sha256,
        "family_sequence_counts": [
            {"count": count, "family": family}
            for family, count in normalized.family_sequence_counts
        ],
        "geometry": geometry.canonical_dict(),
        "geometry_sha256": geometry.geometry_sha256,
        "ruler_category_sequence_counts": [
            {"category": category, "count": count}
            for category, count in normalized.ruler_category_sequence_counts
        ],
        "scores": {
            "axis_order": ["bitwidth", "flattened_layer_head_key_row"],
            "bitwidths": [4, 6, 8],
            "data_base64": _score_data_b64(normalized.scores()),
            "dtype": CALIBRATION_SCORE_DTYPE,
            "shape": [3, geometry.total_rows],
        },
        "sequence_score_manifest_sha256": normalized.sequence_score_manifest_sha256,
        "identity_record_manifest_sha256": normalized.identity_record_manifest_sha256,
        "source_tensor_contract": normalized.source_contract.canonical_dict(),
    }
    canonical_evidence_sha256 = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    document = {
        "artifact_kind": artifact_kind,
        "canonical_evidence_sha256": canonical_evidence_sha256,
        "evidence": evidence,
        "schema_version": CALIBRATION_SCORE_ARTIFACT_SCHEMA_VERSION,
    }
    return canonical_json_bytes(document)


def build_calibration_score_artifact(
    aggregate: CalibrationAggregate,
    *,
    geometry: StaticRhtQ468Geometry,
    calibration_identity_sha256: str,
    marginal_steps: tuple[int, ...] | list[int],
) -> bytes:
    """Build a generic score artifact that cannot impersonate Experiment 013 evidence."""

    return _build_calibration_score_artifact(
        aggregate,
        geometry=geometry,
        calibration_identity_sha256=calibration_identity_sha256,
        marginal_steps=marginal_steps,
        artifact_kind=GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND,
        artifact_profile=GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE,
        artifact_revision=GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION,
    )


def build_frozen_calibration_score_artifact(
    aggregate: CalibrationAggregate,
    *,
    calibration_identity_sha256: str,
) -> bytes:
    """Build the frozen real-geometry K27030/K29334 score artifact."""

    if aggregate.family_sequence_counts != (
        ("mbpp", 128),
        ("pg19", 16),
        ("ruler", 16),
    ):
        raise ValueError("frozen calibration counts must be MBPP=128, PG19=16, RULER=16")
    if tuple(count for _name, count in aggregate.ruler_category_sequence_counts) != (
        4,
        4,
        4,
        4,
    ):
        raise ValueError("each frozen RULER category must contain four sequences")
    if aggregate.source_contract != FROZEN_SOURCE_TENSOR_CONTRACT:
        raise ValueError("frozen calibration requires the exact Experiment 013 source contract")
    return _build_calibration_score_artifact(
        aggregate,
        geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        calibration_identity_sha256=calibration_identity_sha256,
        marginal_steps=(
            FROZEN_STATIC_Q468_ABLATION_STEPS,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
        ),
        artifact_kind=CALIBRATION_SCORE_ARTIFACT_KIND,
        artifact_profile=CALIBRATION_SCORE_ARTIFACT_PROFILE,
        artifact_revision=CALIBRATION_SCORE_ARTIFACT_REVISION,
    )


@dataclass(frozen=True, slots=True)
class DecodedCalibrationScoreArtifact:
    artifact_kind: str
    artifact_profile: str
    artifact_revision: str
    aggregate: CalibrationAggregate
    geometry: StaticRhtQ468Geometry
    calibration_identity_sha256: str
    calibration_scores_sha256: str
    allocations: tuple[tuple[int, torch.Tensor, str], ...]
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
                geometry["key_rows"],
                context="evidence.geometry.key_rows",
                minimum=1,
            ),
            value_width=_artifact_int(
                geometry["value_width"],
                context="evidence.geometry.value_width",
                minimum=1,
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
        contract["axis_order"],
        context="evidence.source_tensor_contract.axis_order",
    )
    raw_shape = _sequence(
        contract["trailing_shape"],
        context="evidence.source_tensor_contract.trailing_shape",
    )
    raw_dtypes = _mapping(
        contract["dtypes"],
        context="evidence.source_tensor_contract.dtypes",
    )
    if set(raw_dtypes) != set(_SOURCE_TENSOR_NAMES):
        raise CalibrationArtifactError(
            "source tensor dtypes must cover the exact source tensor names"
        )
    try:
        return CalibrationSourceTensorContract(
            reduction_profile=_canonical_text(
                contract["reduction_profile"],
                name="source reduction_profile",
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
                (
                    name,
                    _canonical_text(raw_dtypes[name], name=f"source dtype {name}"),
                )
                for name in _SOURCE_TENSOR_NAMES
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CalibrationArtifactError):
            raise
        raise CalibrationArtifactError(f"invalid source tensor contract: {exc}") from exc


def deserialize_calibration_score_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> DecodedCalibrationScoreArtifact:
    """Strictly decode and recompute a canonical score artifact."""

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
        root["canonical_evidence_sha256"],
        context="artifact.canonical_evidence_sha256",
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
        evidence["geometry_sha256"],
        context="evidence.geometry_sha256",
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

    score_record = _mapping(evidence["scores"], context="evidence.scores")
    _exact_keys(
        score_record,
        {"axis_order", "bitwidths", "data_base64", "dtype", "shape"},
        context="evidence.scores",
    )
    if score_record["dtype"] != CALIBRATION_SCORE_DTYPE:
        raise CalibrationArtifactError("score dtype must be float64-le")
    if score_record["axis_order"] != ["bitwidth", "flattened_layer_head_key_row"]:
        raise CalibrationArtifactError("score axis order differs from the frozen layout")
    if score_record["bitwidths"] != [4, 6, 8]:
        raise CalibrationArtifactError("score bitwidth order must be Q4, Q6, Q8")
    if score_record["shape"] != [3, geometry.total_rows]:
        raise CalibrationArtifactError("score shape does not match geometry")
    encoded = score_record["data_base64"]
    if not isinstance(encoded, str):
        raise CalibrationArtifactError("score data_base64 must be a string")
    try:
        raw_scores = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CalibrationArtifactError("score data_base64 is invalid") from exc
    if base64.b64encode(raw_scores).decode("ascii") != encoded:
        raise CalibrationArtifactError("score data_base64 is not canonical")
    expected_bytes = 3 * geometry.total_rows * 8
    if len(raw_scores) != expected_bytes:
        raise CalibrationArtifactError(
            f"score byte length differs: expected {expected_bytes}, got {len(raw_scores)}"
        )
    array = np.frombuffer(raw_scores, dtype="<f8").copy().reshape(3, geometry.total_rows)
    if not np.isfinite(array).all() or (array < 0).any():
        raise CalibrationArtifactError("score arrays must be finite and non-negative")
    score_tensors = tuple(torch.from_numpy(array[index].copy()) for index in range(3))
    aggregate = _validate_aggregate(
        CalibrationAggregate(
            d4=score_tensors[0],
            d6=score_tensors[1],
            d8=score_tensors[2],
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
        *aggregate.scores(),
        geometry=geometry,
    )
    if expected_score_sha256 != computed_score_sha256:
        raise CalibrationArtifactError("calibration score SHA-256 mismatch")

    if not allocation_values:
        raise CalibrationArtifactError("at least one exact allocation is required")
    allocations: list[tuple[int, torch.Tensor, str]] = []
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
        computed_counts = [int((codes == code).sum().item()) for code in range(3)]
        if counts != computed_counts or sum(counts) != geometry.total_rows:
            raise CalibrationArtifactError("recorded code counts differ from exact allocation")
        recorded_code_hash = _artifact_sha(
            allocation["code_map_sha256"],
            context=f"evidence.allocations[{index}].code_map_sha256",
        )
        computed_code_hash = static_q468_code_map_sha256(
            codes,
            geometry=geometry,
            marginal_steps=steps,
        )
        if recorded_code_hash != computed_code_hash:
            raise CalibrationArtifactError("recorded code-map SHA-256 differs from allocation")
        allocations.append((steps, codes, computed_code_hash))

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


def verify_calibration_score_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible fail-closed verification report."""

    file_sha256 = hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else None
    try:
        artifact = deserialize_calibration_score_artifact(
            data,
            expected_file_sha256=expected_file_sha256,
        )
    except (TypeError, ValueError) as exc:
        return {
            "allocations": [],
            "calibration_scores_sha256": None,
            "canonical_evidence_sha256": None,
            "errors": [str(exc)],
            "file_sha256": file_sha256,
            "valid": False,
        }
    return {
        "allocations": [
            {"code_map_sha256": digest, "marginal_steps": steps}
            for steps, _codes, digest in artifact.allocations
        ],
        "calibration_scores_sha256": artifact.calibration_scores_sha256,
        "canonical_evidence_sha256": artifact.canonical_evidence_sha256,
        "errors": [],
        "file_sha256": artifact.file_sha256,
        "valid": True,
    }


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
    for name, value in (("D4", aggregate.d4), ("D6", aggregate.d6), ("D8", aggregate.d8)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or tuple(value.shape) != (expected_rows,)
            or not value.is_contiguous()
        ):
            raise TypeError(f"{name} must be a contiguous CPU torch.float64 row vector")
    scores = tuple(
        _cpu_fp64_scores(value, name=name, expected_rows=expected_rows)
        for name, value in (
            ("D4", aggregate.d4),
            ("D6", aggregate.d6),
            ("D8", aggregate.d8),
        )
    )
    if any(
        not torch.equal(left, right) for left, right in zip(scores, aggregate.scores(), strict=True)
    ):
        raise ValueError("comparator aggregate scores are not canonical CPU-FP64 vectors")
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
    if aggregate.aggregate_scores_sha256 != _comparator_aggregate_score_sha256(aggregate):
        raise ValueError("comparator aggregate-score SHA-256 drifted")
    return aggregate


def _allocate_frozen_comparator_codes(aggregate: ComparatorAggregate) -> torch.Tensor:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    codes = allocate_exact_multibit_codes_fast(
        *(value.reshape(1, rows) for value in aggregate.scores()),
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    ).reshape(-1)
    if codes.device.type != "cpu" or codes.dtype != torch.uint8 or codes.shape != (rows,):
        raise RuntimeError("comparator allocator returned a non-canonical code map")
    if int(codes.to(torch.int64).sum().item()) != FROZEN_STATIC_Q468_PRIMARY_STEPS:
        raise RuntimeError("comparator allocator missed exact K29334")
    return codes.contiguous()


def _frozen_comparator_selector_record(
    aggregate: ComparatorAggregate,
) -> dict[str, object]:
    codes = _allocate_frozen_comparator_codes(aggregate)
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    return {
        "allocation": {
            "allocator_revision": STATIC_Q468_ALLOCATOR_REVISION,
            "code_counts_q4_q6_q8": [int((codes == code).sum().item()) for code in range(3)],
            "code_map_sha256": static_q468_code_map_sha256(
                codes,
                geometry=geometry,
                marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
            ),
            "marginal_steps": FROZEN_STATIC_Q468_PRIMARY_STEPS,
            "packed_precision_bytes": math.ceil(geometry.total_rows * 2 / 8),
        },
        "calibration_scores_sha256": aggregate.aggregate_scores_sha256,
        "family_sequence_counts": [
            {"count": count, "family": family} for family, count in aggregate.family_sequence_counts
        ],
        "method_id": aggregate.selector_profile,
        "position_contract": FROZEN_COMPARATOR_POSITION_CONTRACTS[aggregate.selector_profile],
        "position_manifest_sha256": aggregate.position_manifest_sha256,
        "ruler_category_sequence_counts": [
            {"category": category, "count": count}
            for category, count in aggregate.ruler_category_sequence_counts
        ],
        "scores": {
            "axis_order": ["bitwidth", "flattened_layer_head_key_row"],
            "bitwidths": [4, 6, 8],
            "data_base64": _score_data_b64(aggregate.scores()),
            "dtype": CALIBRATION_SCORE_DTYPE,
            "shape": [3, geometry.total_rows],
        },
        "sequence_score_manifest_sha256": aggregate.sequence_score_manifest_sha256,
    }


def build_frozen_comparator_score_artifact(
    mse_aggregate: ComparatorAggregate,
    fisher_aggregate: ComparatorAggregate,
    *,
    calibration_identity_sha256: str,
) -> bytes:
    """Build canonical ``comparator-scores.json`` with both exact-K29334 methods."""

    identity_sha256 = _sha256(
        calibration_identity_sha256,
        name="calibration_identity_sha256",
    )
    normalized_mse = _validate_frozen_comparator_aggregate(
        mse_aggregate,
        expected_profile=FROZEN_UNWEIGHTED_MSE_PROFILE,
    )
    normalized_fisher = _validate_frozen_comparator_aggregate(
        fisher_aggregate,
        expected_profile=FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
    )
    if (
        normalized_mse.identity_record_manifest_sha256
        != normalized_fisher.identity_record_manifest_sha256
    ):
        raise ValueError("both comparator profiles must bind the same Identity-v5 records")
    if (
        normalized_mse.family_sequence_counts != normalized_fisher.family_sequence_counts
        or normalized_mse.ruler_category_sequence_counts
        != normalized_fisher.ruler_category_sequence_counts
    ):
        raise ValueError("both comparator profiles must cover the same frozen sequence counts")

    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    evidence = {
        "aggregation_contract": _COMPARATOR_AGGREGATION_CONTRACT,
        "artifact_profile": COMPARATOR_SCORE_ARTIFACT_PROFILE,
        "artifact_revision": COMPARATOR_SCORE_ARTIFACT_REVISION,
        "calibration_identity_sha256": identity_sha256,
        "endpoint_tensor_contract": {
            "axis_order": list(FROZEN_COMPARATOR_ENDPOINT_AXIS_ORDER),
            "dtype": CALIBRATION_SCORE_DTYPE,
            "trailing_shape": list(FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape),
        },
        "geometry": geometry.canonical_dict(),
        "geometry_sha256": geometry.geometry_sha256,
        "identity_record_manifest_sha256": (normalized_mse.identity_record_manifest_sha256),
        "selectors": [
            _frozen_comparator_selector_record(normalized_mse),
            _frozen_comparator_selector_record(normalized_fisher),
        ],
    }
    canonical_evidence_sha256 = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    return canonical_json_bytes(
        {
            "artifact_kind": COMPARATOR_SCORE_ARTIFACT_KIND,
            "canonical_evidence_sha256": canonical_evidence_sha256,
            "evidence": evidence,
            "schema_version": COMPARATOR_SCORE_ARTIFACT_SCHEMA_VERSION,
        }
    )


@dataclass(frozen=True, slots=True)
class ComparatorSelectorArtifact:
    method_id: UnweightedSelectorProfile
    aggregate: ComparatorAggregate
    position_manifest_sha256: str
    calibration_scores_sha256: str
    marginal_steps: int
    precision_codes: torch.Tensor
    code_map_sha256: str


@dataclass(frozen=True, slots=True)
class ComparatorScoreArtifact:
    selectors: Mapping[str, ComparatorSelectorArtifact]
    calibration_identity_sha256: str
    canonical_evidence_sha256: str
    file_sha256: str


def _decode_comparator_scores(
    value: object,
    *,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    record = _mapping(value, context=context)
    _exact_keys(
        record,
        {"axis_order", "bitwidths", "data_base64", "dtype", "shape"},
        context=context,
    )
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    if record["axis_order"] != ["bitwidth", "flattened_layer_head_key_row"]:
        raise CalibrationArtifactError(f"{context}.axis_order drifted")
    if record["bitwidths"] != [4, 6, 8]:
        raise CalibrationArtifactError(f"{context}.bitwidths must be Q4, Q6, Q8")
    if record["dtype"] != CALIBRATION_SCORE_DTYPE:
        raise CalibrationArtifactError(f"{context}.dtype must be float64-le")
    if record["shape"] != [3, geometry.total_rows]:
        raise CalibrationArtifactError(f"{context}.shape differs from frozen geometry")
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
    if not np.isfinite(array).all() or (array < 0).any():
        raise CalibrationArtifactError(f"{context} arrays must be finite and non-negative")
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(torch.from_numpy(array[index].copy()) for index in range(3)),
    )


def deserialize_comparator_score_artifact(
    data: bytes,
    *,
    expected_calibration_identity_sha256: str | None = None,
) -> ComparatorScoreArtifact:
    """Strictly decode and recompute canonical comparator-score evidence."""

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
        evidence["selectors"],
        context="comparator artifact.evidence.selectors",
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
        scores = _decode_comparator_scores(selector["scores"], context=f"{context}.scores")
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
        aggregate = ComparatorAggregate(
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
        )
        _validate_frozen_comparator_aggregate(
            aggregate,
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
            raise CalibrationArtifactError(f"{context}.allocation allocator revision drifted")
        marginal_steps = _artifact_int(
            allocation["marginal_steps"],
            context=f"{context}.allocation.marginal_steps",
        )
        if marginal_steps != FROZEN_STATIC_Q468_PRIMARY_STEPS:
            raise CalibrationArtifactError(f"{context} allocation must spend exact K29334")
        expected_packed_bytes = math.ceil(geometry.total_rows * 2 / 8)
        if (
            _artifact_int(
                allocation["packed_precision_bytes"],
                context=f"{context}.allocation.packed_precision_bytes",
            )
            != expected_packed_bytes
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
        computed_counts = [int((codes == code).sum().item()) for code in range(3)]
        if counts != computed_counts or sum(counts) != geometry.total_rows:
            raise CalibrationArtifactError(f"{context} code counts differ from exact allocation")
        code_map_hash = _artifact_sha(
            allocation["code_map_sha256"],
            context=f"{context}.allocation.code_map_sha256",
        )
        computed_code_hash = static_q468_code_map_sha256(
            codes,
            geometry=geometry,
            marginal_steps=marginal_steps,
        )
        if code_map_hash != computed_code_hash:
            raise CalibrationArtifactError(f"{context} code-map SHA-256 drifted")
        decoded_selectors[expected_method] = ComparatorSelectorArtifact(
            method_id=expected_method,
            aggregate=aggregate,
            position_manifest_sha256=position_hash,
            calibration_scores_sha256=aggregate_score_hash,
            marginal_steps=marginal_steps,
            precision_codes=codes,
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


def verify_comparator_score_artifact(
    data: bytes,
    *,
    expected_calibration_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible fail-closed comparator verification report."""

    file_sha256 = hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else None
    try:
        artifact = deserialize_comparator_score_artifact(
            data,
            expected_calibration_identity_sha256=expected_calibration_identity_sha256,
        )
    except (TypeError, ValueError) as exc:
        return {
            "errors": [str(exc)],
            "file_sha256": file_sha256,
            "selectors": [],
            "valid": False,
        }
    return {
        "errors": [],
        "file_sha256": artifact.file_sha256,
        "selectors": [
            {
                "code_map_sha256": artifact.selectors[method].code_map_sha256,
                "method_id": method,
            }
            for method in FROZEN_COMPARATOR_PROFILE_ORDER
        ],
        "valid": True,
    }


def _stability_threshold_record() -> dict[str, str]:
    return {
        "maximum_layer_mean_bitwidth_shift": MAX_LAYER_MEAN_BITWIDTH_SHIFT.hex(),
        "minimum_q8_jaccard": MIN_SPLIT_HALF_Q8_JACCARD.hex(),
        "minimum_spearman_average_ties": MIN_SPLIT_HALF_SPEARMAN.hex(),
    }


def _stability_metric_record(result: PolicyStabilityResult) -> dict[str, object]:
    if result.spearman_average_ties is None:
        spearman: str | None = None
    else:
        spearman = result.spearman_average_ties.hex()
    return {
        "checks": [{"name": name, "passed": passed} for name, passed in result.checks],
        "layer_mean_bitwidth_shifts": [
            {"layer_index": layer, "shift": shift.hex()}
            for layer, shift in result.layer_mean_bitwidth_shifts
        ],
        "maximum_layer_mean_bitwidth_shift": (result.max_layer_mean_bitwidth_shift.hex()),
        "passed": result.passed,
        "q8_jaccard": result.q8_jaccard.hex(),
        "spearman_average_ties": spearman,
    }


def _split_half_evidence_record(
    aggregate: CalibrationAggregate,
    *,
    half: SplitHalf,
) -> tuple[dict[str, object], torch.Tensor]:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    normalized = _validate_aggregate(aggregate, expected_rows=geometry.total_rows)
    if normalized.family_sequence_counts != (
        ("mbpp", 64),
        ("pg19", 8),
        ("ruler", 8),
    ):
        raise ValueError("each frozen split half must contain MBPP=64, PG19=8, RULER=8")
    if normalized.ruler_category_sequence_counts != tuple(
        (category, 2) for category in RULER_CATEGORY_ORDER
    ):
        raise ValueError("each frozen split half must contain two sequences per RULER category")
    if normalized.source_contract != FROZEN_SOURCE_TENSOR_CONTRACT:
        raise ValueError("split-half evidence requires the frozen source tensor contract")
    scores_sha256 = static_q468_distortion_sha256(
        *normalized.scores(),
        geometry=geometry,
    )
    codes = allocate_static_q468_code_map(
        normalized,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    )
    return (
        {
            "calibration_scores_sha256": scores_sha256,
            "code_map": {
                "code_map_sha256": static_q468_code_map_sha256(
                    codes,
                    geometry=geometry,
                    marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
                ),
                "codes_base64": base64.b64encode(codes.numpy().tobytes()).decode("ascii"),
                "dtype": "uint8",
                "shape": [geometry.total_rows],
            },
            "family_sequence_counts": [
                {"count": count, "family": family}
                for family, count in normalized.family_sequence_counts
            ],
            "half": half,
            "identity_record_manifest_sha256": (normalized.identity_record_manifest_sha256),
            "ruler_category_sequence_counts": [
                {"category": category, "count": count}
                for category, count in normalized.ruler_category_sequence_counts
            ],
            "scores": {
                "axis_order": ["bitwidth", "flattened_layer_head_key_row"],
                "bitwidths": [4, 6, 8],
                "data_base64": _score_data_b64(normalized.scores()),
                "dtype": CALIBRATION_SCORE_DTYPE,
                "shape": [3, geometry.total_rows],
            },
            "sequence_score_manifest_sha256": (normalized.sequence_score_manifest_sha256),
            "source_tensor_contract": normalized.source_contract.canonical_dict(),
        },
        codes,
    )


def build_frozen_split_half_stability_artifact(
    half_a_aggregate: CalibrationAggregate,
    half_b_aggregate: CalibrationAggregate,
    *,
    identity_file_sha256: str,
    canonical_identity_sha256: str,
    resolver_assignment_sha256: str,
    full_sequence_score_manifest_sha256: str,
    full_calibration_scores_sha256: str,
) -> bytes:
    """Build the strict passing K29334 split-half stability artifact."""

    half_a, codes_a = _split_half_evidence_record(half_a_aggregate, half="a")
    half_b, codes_b = _split_half_evidence_record(half_b_aggregate, half="b")
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    stability = evaluate_policy_stability(
        codes_a,
        codes_b,
        layer_indices=geometry.layer_indices,
        rows_per_layer=geometry.rows_per_layer,
        expected_marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    )
    if not stability.passed:
        raise ValueError("frozen K29334 split-half stability gate did not pass")
    evidence = {
        "artifact_profile": SPLIT_HALF_STABILITY_ARTIFACT_PROFILE,
        "artifact_revision": SPLIT_HALF_STABILITY_ARTIFACT_REVISION,
        "full_calibration": {
            "calibration_scores_sha256": _sha256(
                full_calibration_scores_sha256,
                name="full_calibration_scores_sha256",
            ),
            "sequence_score_manifest_sha256": _sha256(
                full_sequence_score_manifest_sha256,
                name="full_sequence_score_manifest_sha256",
            ),
        },
        "geometry": geometry.canonical_dict(),
        "geometry_sha256": geometry.geometry_sha256,
        "halves": [half_a, half_b],
        "identity": {
            "canonical_identity_sha256": _sha256(
                canonical_identity_sha256,
                name="canonical_identity_sha256",
            ),
            "identity_file_sha256": _sha256(
                identity_file_sha256,
                name="identity_file_sha256",
            ),
            "resolver_assignment_sha256": _sha256(
                resolver_assignment_sha256,
                name="resolver_assignment_sha256",
            ),
        },
        "marginal_steps": FROZEN_STATIC_Q468_PRIMARY_STEPS,
        "metrics": _stability_metric_record(stability),
        "thresholds": _stability_threshold_record(),
    }
    document = {
        "artifact_kind": SPLIT_HALF_STABILITY_ARTIFACT_KIND,
        "canonical_evidence_sha256": hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
        "evidence": evidence,
        "schema_version": SPLIT_HALF_STABILITY_ARTIFACT_SCHEMA_VERSION,
    }
    return canonical_json_bytes(document)


@dataclass(frozen=True, slots=True)
class DecodedSplitHalfStabilityArtifact:
    identity_file_sha256: str
    canonical_identity_sha256: str
    resolver_assignment_sha256: str
    full_sequence_score_manifest_sha256: str
    full_calibration_scores_sha256: str
    half_a_aggregate: CalibrationAggregate
    half_b_aggregate: CalibrationAggregate
    half_a_codes: torch.Tensor
    half_b_codes: torch.Tensor
    stability: PolicyStabilityResult
    canonical_evidence_sha256: str
    file_sha256: str


def _decode_frozen_split_half(
    value: object,
    *,
    expected_half: SplitHalf,
) -> tuple[CalibrationAggregate, torch.Tensor]:
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
        raise CalibrationArtifactError("each split half must contain MBPP=64, PG19=8, RULER=8")
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
    score_record = _mapping(
        half["scores"],
        context=f"evidence.halves[{expected_half}].scores",
    )
    _exact_keys(
        score_record,
        {"axis_order", "bitwidths", "data_base64", "dtype", "shape"},
        context=f"evidence.halves[{expected_half}].scores",
    )
    if (
        score_record["axis_order"] != ["bitwidth", "flattened_layer_head_key_row"]
        or score_record["bitwidths"] != [4, 6, 8]
        or score_record["dtype"] != CALIBRATION_SCORE_DTYPE
        or score_record["shape"] != [3, geometry.total_rows]
    ):
        raise CalibrationArtifactError("split-half score layout drifted")
    encoded_scores = score_record["data_base64"]
    if not isinstance(encoded_scores, str):
        raise CalibrationArtifactError("split-half score data must be base64 text")
    try:
        raw_scores = base64.b64decode(encoded_scores, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CalibrationArtifactError("split-half score data_base64 is invalid") from exc
    if base64.b64encode(raw_scores).decode("ascii") != encoded_scores:
        raise CalibrationArtifactError("split-half score data_base64 is not canonical")
    if len(raw_scores) != 3 * geometry.total_rows * 8:
        raise CalibrationArtifactError("split-half score byte length drifted")
    score_array = np.frombuffer(raw_scores, dtype="<f8").copy().reshape(3, geometry.total_rows)
    if not np.isfinite(score_array).all() or (score_array < 0).any():
        raise CalibrationArtifactError("split-half scores must be finite and non-negative")
    aggregate = _validate_aggregate(
        CalibrationAggregate(
            d4=torch.from_numpy(score_array[0].copy()),
            d6=torch.from_numpy(score_array[1].copy()),
            d8=torch.from_numpy(score_array[2].copy()),
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
    computed_scores_sha256 = static_q468_distortion_sha256(*aggregate.scores(), geometry=geometry)
    if recorded_scores_sha256 != computed_scores_sha256:
        raise CalibrationArtifactError("split-half calibration score SHA-256 drifted")
    expected_codes = allocate_static_q468_code_map(
        aggregate,
        marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
    )
    code_record = _mapping(
        half["code_map"],
        context=f"evidence.halves[{expected_half}].code_map",
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
    recorded_codes = torch.from_numpy(np.frombuffer(raw_codes, dtype="u1").copy())
    if not torch.equal(recorded_codes, expected_codes):
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
    return aggregate, expected_codes


def deserialize_frozen_split_half_stability_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
    expected_identity_file_sha256: str | None = None,
    expected_canonical_identity_sha256: str | None = None,
    expected_resolver_assignment_sha256: str | None = None,
) -> DecodedSplitHalfStabilityArtifact:
    """Strictly decode and recompute the frozen passing K29334 split artifact."""

    if not isinstance(data, bytes):
        raise TypeError("split-half stability artifact must be bytes")
    file_sha256 = hashlib.sha256(data).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != _artifact_sha(
        expected_file_sha256,
        context="expected_file_sha256",
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
    expected_identity_values = (
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
    )
    for expected_value, recorded_value, context in expected_identity_values:
        if expected_value is not None and recorded_value != _artifact_sha(
            expected_value,
            context=f"expected {context}",
        ):
            raise CalibrationArtifactError(f"{context} differs from expected identity")
    full_calibration = _mapping(evidence["full_calibration"], context="split full calibration")
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


def verify_frozen_split_half_stability_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
    expected_identity_file_sha256: str | None = None,
    expected_canonical_identity_sha256: str | None = None,
    expected_resolver_assignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed report for frozen split-half stability evidence."""

    file_sha256 = hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else None
    try:
        decoded = deserialize_frozen_split_half_stability_artifact(
            data,
            expected_file_sha256=expected_file_sha256,
            expected_identity_file_sha256=expected_identity_file_sha256,
            expected_canonical_identity_sha256=expected_canonical_identity_sha256,
            expected_resolver_assignment_sha256=expected_resolver_assignment_sha256,
        )
    except (TypeError, ValueError) as exc:
        return {
            "errors": [str(exc)],
            "file_sha256": file_sha256,
            "passed": False,
            "valid": False,
        }
    return {
        "errors": [],
        "file_sha256": decoded.file_sha256,
        "passed": decoded.stability.passed,
        "valid": True,
    }
