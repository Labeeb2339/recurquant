"""Independent offline verification for the frozen Experiment 012 artifact.

The verifier deliberately consumes only the published JSON artifact and
package modules.  It does not import the experiment runner under ``scripts/``
and it never trusts stored aggregate metrics, storage receipts, or gate flags.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .evaluation import TokenFidelity, fidelity_summary
from .evidence import canonical_json_bytes
from .statelease_evaluation import (
    EQUAL_BYTE_NO_REPLAY_METHODS,
    FIXED_REPLAY_METHODS,
    FROZEN_STAGE_A_ALIGNED_TOKENS,
    FROZEN_STAGE_A_PROMPT_TOKENS,
    FROZEN_STAGE_A_RECURRENT_LAYER_INDICES,
    FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS,
    FROZEN_STATELEASE_RESIDENT_BYTES,
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    TrajectoryNmseAccumulator,
    evaluate_statelease_stage_a_gate,
)

EXPERIMENT012_FILE_SHA256 = (
    "1e92b0bea176154496c7d5e45013bf051ef3f388352c1267d86910f81844fd22"
)
EXPERIMENT012_CANONICAL_EVIDENCE_SHA256 = (
    "d4bd2c89bb265e5e1dab81a7bf89d97e71fa4a6822bef5d56142428e16b897c1"
)
EXPERIMENT012_ARTIFACT_KIND = (
    "recurquant_experiment012_statelease_stage_a_falsification"
)
EXPERIMENT012_SCHEMA_VERSION = 1
EXPERIMENT012_TASK_ID = 666

FP32_METHOD = "fp32_reference"
QUALITY_METHODS = (
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    *FIXED_REPLAY_METHODS,
    *EQUAL_BYTE_NO_REPLAY_METHODS,
)
ALL_METHODS = (FP32_METHOD, *QUALITY_METHODS)

_FULL_PRECISION_STATE_BYTES = 18_874_368
_HISTORICAL_RHT_BYTES = 2_711_552
_HISTORICAL_RHT_CHECKPOINT_BYTES = 2_564_096
_SHARED_PERSISTENT_BYTES = 2_199_552
_EXACT_ROW_PLAN_SHA256 = (
    "018382fd7d946a58d7d91f02d5c710a1295a20086790311399e163ce28205da9"
)
_CANDIDATE_SCHEMA_SHA256 = {
    RHT_CQER_METHOD: "fa6c87a6053d734a5fbeb98f39cd7aea06b81c13e598b2e1a69d2fb1486a76d8",
    STATELEASE_METHOD: "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f",
    "fixed_cc1": "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f",
    "fixed_cc2": "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f",
    "fixed_cc4": "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f",
    "fixed_cc5": "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f",
    "fixed_cut4_in5": (
        "befd643add82995053158f8a917e42c633943f554b784b31b7cd2651f2dcd44f"
    ),
    "expanded_rht_q4_q8": (
        "cddd02dfc98c6f2d7070a0b49392a79d76c97f6ab401d97a61693df7b78993ed"
    ),
    "rht_q4_q6_q8": "a3307a5e68b44460c3a14104e5917b559a061f20e16888022962a89313d1713d",
    "rht_residual_q4": (
        "6de7b8c0079d6c80c8ced1cbd955befa1424df6929390be8c316fd814f12d4a9"
    ),
}
_SHARED_SCHEMA_SHA256 = "328aab59e8ff7daad4f2f21d45829c91521690b28ab04844a40e28c9e7660511"
_TASK_ROW_SHA256 = "b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9"
_PROMPT_TOKEN_IDS_SHA256 = (
    "729215c4c99cdf96b13ad73f6ac7b537ddf9e882409b77e479d609aee046bffa"
)
_CODE_TOKEN_IDS_SHA256 = (
    "a920370c4892513c8a5cdb9f88a33fd95d4c90201af39fdb7d517f3ad42a9d9a"
)
_TOKEN_HASH_SERIALIZATION = "sha256(recurquant.evidence.canonical_json_bytes(list[int]))"

_OUTER_KEYS = {
    "artifact_kind",
    "canonical_evidence_sha256",
    "evidence",
    "schema_version",
}
_EVIDENCE_KEYS = {
    "artifact_kind",
    "claim_boundary",
    "command",
    "created_at_utc",
    "cuda_memory",
    "dataset",
    "diagnostics",
    "environment",
    "input_authentication",
    "methods",
    "metric_contract",
    "metrics_aligned",
    "model",
    "one_authenticated_quality_run",
    "per_token_aligned",
    "prefill_identity",
    "protocol",
    "repository",
    "result_completeness",
    "schema_version",
    "screening_only",
    "source_files",
    "stage_a_gate",
    "storage",
    "storage_contracts",
    "trajectory_nmse",
    "trajectory_nmse_per_layer_write",
    "update_evidence",
}
_PER_TOKEN_KEYS = {
    "all_logits_finite",
    "candidate_nll",
    "delta_nll",
    "input_token_id",
    "kl",
    "reference_nll",
    "target_token_id",
    "top1_agreement",
    "write_index",
}
_TENSOR_SCHEMA_KEYS = {"dtype", "logical_bytes", "shape", "storage_bytes"}
_ELEMENT_BYTES = {
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.int64": 8,
}


class StateLeaseArtifactError(ValueError):
    """Raised internally when a frozen artifact condition is not satisfied."""


def _reject_constant(value: str) -> None:
    raise StateLeaseArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StateLeaseArtifactError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StateLeaseArtifactError(f"{context} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise StateLeaseArtifactError(
            f"{context} keys differ from the frozen schema: missing={missing}, extra={extra}"
        )


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateLeaseArtifactError(f"{context} must be an integer")
    return value


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateLeaseArtifactError(f"{context} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise StateLeaseArtifactError(f"{context} must be finite")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateLeaseArtifactError(message)


def _exact_value(actual: object, expected: object, *, context: str) -> None:
    """Compare JSON-compatible values with strict scalar types and complete keys."""

    if isinstance(expected, Mapping):
        actual_mapping = _mapping(actual, context=context)
        _exact_keys(actual_mapping, set(expected), context=context)
        for key, expected_value in expected.items():
            _exact_value(actual_mapping[key], expected_value, context=f"{context}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise StateLeaseArtifactError(f"{context} must be an array")
        if len(actual) != len(expected):
            raise StateLeaseArtifactError(
                f"{context} length differs: expected {len(expected)}, got {len(actual)}"
            )
        for index, expected_value in enumerate(expected):
            _exact_value(actual[index], expected_value, context=f"{context}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise StateLeaseArtifactError(
            f"{context} differs: expected {expected!r}, got {actual!r}"
        )


def _collect_exact_differences(
    actual: object,
    expected: object,
    *,
    context: str,
    errors: list[str],
) -> None:
    """Collect every gate difference instead of stopping at its top-level flag."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            errors.append(f"{context}: expected object, got {type(actual).__name__}")
            return
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            errors.append(f"{context}: missing keys {missing}")
        if extra:
            errors.append(f"{context}: extra keys {extra}")
        for key in sorted(set(expected) & set(actual)):
            _collect_exact_differences(
                actual[key],
                expected[key],
                context=f"{context}.{key}",
                errors=errors,
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            errors.append(f"{context}: expected array, got {type(actual).__name__}")
            return
        if len(actual) != len(expected):
            errors.append(f"{context}: expected length {len(expected)}, got {len(actual)}")
        for index in range(min(len(actual), len(expected))):
            _collect_exact_differences(
                actual[index],
                expected[index],
                context=f"{context}[{index}]",
                errors=errors,
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        errors.append(f"{context}: expected {expected!r}, got {actual!r}")


def _validate_artifact_contract(document: Mapping[str, object]) -> Mapping[str, object]:
    _exact_keys(document, _OUTER_KEYS, context="artifact")
    _require(
        document.get("schema_version") == EXPERIMENT012_SCHEMA_VERSION
        and not isinstance(document.get("schema_version"), bool),
        "artifact schema_version differs from the frozen value",
    )
    _require(
        document.get("artifact_kind") == EXPERIMENT012_ARTIFACT_KIND,
        "artifact kind differs from the frozen value",
    )
    evidence = _mapping(document.get("evidence"), context="artifact.evidence")
    _exact_keys(evidence, _EVIDENCE_KEYS, context="artifact.evidence")
    _require(
        evidence.get("schema_version") == EXPERIMENT012_SCHEMA_VERSION
        and not isinstance(evidence.get("schema_version"), bool),
        "evidence schema_version differs from the frozen value",
    )
    _require(
        evidence.get("artifact_kind") == EXPERIMENT012_ARTIFACT_KIND,
        "evidence artifact_kind differs from the frozen value",
    )
    _require(evidence.get("screening_only") is True, "screening_only must be true")
    _require(
        evidence.get("one_authenticated_quality_run") is True,
        "one_authenticated_quality_run must be true",
    )
    return evidence


def _validate_task_and_token_contract(evidence: Mapping[str, object]) -> None:
    _exact_value(evidence.get("methods"), list(ALL_METHODS), context="evidence.methods")
    _exact_value(
        evidence.get("metric_contract"),
        {
            "aligned_primary": "one-token decode writes after the shared FP32 prefill",
            "aligned_token_count": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "statelease_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
            "authenticated_forward_passes": len(ALL_METHODS)
            * (1 + FROZEN_STAGE_A_ALIGNED_TOKENS),
            "reference_aligned_trajectory": (
                "per-layer FP64 NMSE against the matched FP32 recurrent state at every write"
            ),
        },
        context="evidence.metric_contract",
    )
    _exact_value(
        evidence.get("result_completeness"),
        {
            "method_count": len(ALL_METHODS),
            "authenticated_forward_passes": len(ALL_METHODS)
            * (1 + FROZEN_STAGE_A_ALIGNED_TOKENS),
            "per_token_rows_per_method": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "trajectory_writes_per_method": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "trajectory_layers_per_write": len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES),
            "statelease_update_evidence_records": FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS,
            "aligned_summaries_recomputed_from_per_token_evidence": True,
            "aligned_aggregation_semantics": "recurquant.evaluation.fidelity_summary_fp32",
            "trajectory_summaries_recomputed_from_per_layer_write_evidence": True,
            "trajectory_aggregation_semantics": (
                "recurquant.statelease_evaluation.TrajectoryNmseAccumulator_fp64_neumaier"
            ),
            "stored_aggregates_exactly_reconciled": True,
            "all_expected_records_present_and_ordered": True,
        },
        context="evidence.result_completeness",
    )

    dataset = _mapping(evidence.get("dataset"), context="evidence.dataset")
    _require(dataset.get("phase") == "calibration", "dataset phase must be calibration")
    _require(
        dataset.get("selection_mode") == "exact_already_open_task_id",
        "dataset selection mode differs from the frozen value",
    )
    _require(
        dataset.get("protected_window_8_16_loaded_tokenized_or_evaluated") is False,
        "protected task window must remain unaccessed",
    )
    _exact_value(
        dataset.get("identity"),
        {
            "aligned_scored_tokens": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "authenticated_before_model_weights": True,
            "code_tokens": FROZEN_STAGE_A_ALIGNED_TOKENS + 1,
            "prompt_tokens": FROZEN_STAGE_A_PROMPT_TOKENS,
            "row_sha256": _TASK_ROW_SHA256,
            "task_id": EXPERIMENT012_TASK_ID,
        },
        context="evidence.dataset.identity",
    )
    _exact_value(
        dataset.get("runtime_token_manifest"),
        [
            {
                "aligned_scored_tokens": FROZEN_STAGE_A_ALIGNED_TOKENS,
                "code_tokens": FROZEN_STAGE_A_ALIGNED_TOKENS + 1,
                "full_code_scored_tokens": FROZEN_STAGE_A_ALIGNED_TOKENS + 1,
                "prompt_tokens": FROZEN_STAGE_A_PROMPT_TOKENS,
                "task_id": EXPERIMENT012_TASK_ID,
            }
        ],
        context="evidence.dataset.runtime_token_manifest",
    )
    _exact_value(
        dataset.get("runtime_token_ids"),
        {
            "code_shape": [1, FROZEN_STAGE_A_ALIGNED_TOKENS + 1],
            "code_token_ids_sha256": _CODE_TOKEN_IDS_SHA256,
            "prompt_shape": [1, FROZEN_STAGE_A_PROMPT_TOKENS],
            "prompt_token_ids_sha256": _PROMPT_TOKEN_IDS_SHA256,
            "token_id_hash_serialization": _TOKEN_HASH_SERIALIZATION,
        },
        context="evidence.dataset.runtime_token_ids",
    )

    protocol = _mapping(evidence.get("protocol"), context="evidence.protocol")
    for key, expected in {
        "name": "Experiment 012 Stage A",
        "method": "StateLease-H5",
        "task_locked": True,
        "thresholds_locked": True,
        "protected_ranked_window": [8, 16],
        "protected_window_accessed": False,
    }.items():
        _exact_value(protocol.get(key), expected, context=f"evidence.protocol.{key}")

    model = _mapping(evidence.get("model"), context="evidence.model")
    for key, expected in {
        "id": "Qwen/Qwen3.5-0.8B-Base",
        "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "dtype": "torch.bfloat16",
        "attn_implementation": "eager",
        "device": "cuda",
        "configuration_authenticated_before_task_or_model_weights": True,
        "configuration_loaded_from_authenticated_snapshot": True,
    }.items():
        _exact_value(model.get(key), expected, context=f"evidence.model.{key}")
    _exact_value(
        model.get("geometry"),
        {
            "batch_size": 1,
            "key_rows": 128,
            "linear_layer_indices": list(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES),
            "num_hidden_layers": 24,
            "recurrent_layers": len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES),
            "recurrent_state_dtype": "torch.float32",
            "value_heads": 16,
            "value_width": 128,
        },
        context="evidence.model.geometry",
    )

    diagnostics = _mapping(evidence.get("diagnostics"), context="evidence.diagnostics")
    updates = _mapping(evidence.get("update_evidence"), context="evidence.update_evidence")
    _exact_keys(diagnostics, set(QUALITY_METHODS), context="evidence.diagnostics")
    _exact_keys(updates, set(QUALITY_METHODS), context="evidence.update_evidence")


def _recompute_aligned_metrics(
    evidence: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    per_token = _mapping(
        evidence.get("per_token_aligned"),
        context="evidence.per_token_aligned",
    )
    stored = _mapping(evidence.get("metrics_aligned"), context="evidence.metrics_aligned")
    _exact_keys(per_token, set(ALL_METHODS), context="evidence.per_token_aligned")
    _exact_keys(stored, set(ALL_METHODS), context="evidence.metrics_aligned")

    reference_identity: list[tuple[int, int, float]] | None = None
    reference_code_ids: list[int] | None = None
    recomputed: dict[str, dict[str, object]] = {}
    for method in ALL_METHODS:
        raw_rows = per_token[method]
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise StateLeaseArtifactError(f"{method} per-token evidence must be an array")
        if len(raw_rows) != FROZEN_STAGE_A_ALIGNED_TOKENS:
            raise StateLeaseArtifactError(
                f"{method} must contain exactly {FROZEN_STAGE_A_ALIGNED_TOKENS} token rows"
            )
        identities: list[tuple[int, int, float]] = []
        kl_values: list[float] = []
        reference_nll_values: list[float] = []
        candidate_nll_values: list[float] = []
        top1_values: list[bool] = []
        finite_values: list[bool] = []
        for index, raw_row in enumerate(raw_rows):
            row = _mapping(raw_row, context=f"{method}.per_token[{index}]")
            _exact_keys(row, _PER_TOKEN_KEYS, context=f"{method}.per_token[{index}]")
            write_index = _strict_int(
                row.get("write_index"), context=f"{method}[{index}].write_index"
            )
            input_id = _strict_int(
                row.get("input_token_id"), context=f"{method}[{index}].input_token_id"
            )
            target_id = _strict_int(
                row.get("target_token_id"), context=f"{method}[{index}].target_token_id"
            )
            _require(write_index == index, f"{method} token rows are not ordered at {index}")
            kl = _finite_float(row.get("kl"), context=f"{method}[{index}].kl")
            reference_nll = _finite_float(
                row.get("reference_nll"), context=f"{method}[{index}].reference_nll"
            )
            candidate_nll = _finite_float(
                row.get("candidate_nll"), context=f"{method}[{index}].candidate_nll"
            )
            delta_nll = _finite_float(
                row.get("delta_nll"), context=f"{method}[{index}].delta_nll"
            )
            top1 = row.get("top1_agreement")
            finite = row.get("all_logits_finite")
            _require(isinstance(top1, bool), f"{method}[{index}].top1_agreement must be bool")
            _require(finite is True, f"{method}[{index}] reports non-finite logits")
            _require(
                delta_nll == candidate_nll - reference_nll,
                f"{method}[{index}].delta_nll does not equal candidate minus reference NLL",
            )
            _require(kl >= 0.0, f"{method}[{index}].kl must be non-negative")
            _require(
                reference_nll >= 0.0 and candidate_nll >= 0.0,
                f"{method}[{index}] NLL values must be non-negative",
            )
            if method == FP32_METHOD:
                _require(
                    kl == 0.0
                    and candidate_nll == reference_nll
                    and delta_nll == 0.0
                    and top1 is True,
                    f"{method}[{index}] is not an exact self-reference row",
                )
            identities.append((input_id, target_id, reference_nll))
            kl_values.append(kl)
            reference_nll_values.append(reference_nll)
            candidate_nll_values.append(candidate_nll)
            top1_values.append(top1)
            finite_values.append(finite)

        if reference_identity is None:
            reference_identity = identities
            reference_code_ids = [identities[0][0], *(row[1] for row in identities)]
            for index in range(1, len(identities)):
                _require(
                    identities[index][0] == identities[index - 1][1],
                    f"reference token chain breaks at write {index}",
                )
            code_hash = hashlib.sha256(canonical_json_bytes(reference_code_ids)).hexdigest()
            _require(
                code_hash == _CODE_TOKEN_IDS_SHA256,
                f"reconstructed code token hash differs: {code_hash}",
            )
        else:
            _require(
                identities == reference_identity,
                f"{method} per-token rows do not share the FP32 token/reference identity",
            )

        summary = fidelity_summary(
            TokenFidelity(
                kl=torch.tensor(kl_values, dtype=torch.float32),
                reference_nll=torch.tensor(reference_nll_values, dtype=torch.float32),
                candidate_nll=torch.tensor(candidate_nll_values, dtype=torch.float32),
                top1_agreement=torch.tensor(top1_values, dtype=torch.bool),
            )
        )
        normalized: dict[str, object] = {**summary, "all_logits_finite": all(finite_values)}
        _exact_value(
            stored[method],
            normalized,
            context=f"evidence.metrics_aligned.{method}",
        )
        recomputed[method] = normalized
    return recomputed


def _recompute_trajectory(
    evidence: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw_by_method = _mapping(
        evidence.get("trajectory_nmse_per_layer_write"),
        context="evidence.trajectory_nmse_per_layer_write",
    )
    stored = _mapping(evidence.get("trajectory_nmse"), context="evidence.trajectory_nmse")
    _exact_keys(
        raw_by_method,
        set(ALL_METHODS),
        context="evidence.trajectory_nmse_per_layer_write",
    )
    _exact_keys(stored, set(ALL_METHODS), context="evidence.trajectory_nmse")
    expected_layer_keys = {str(layer) for layer in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES}
    recomputed: dict[str, dict[str, object]] = {}
    for method in ALL_METHODS:
        rows = raw_by_method[method]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise StateLeaseArtifactError(f"{method} trajectory evidence must be an array")
        if len(rows) != FROZEN_STAGE_A_ALIGNED_TOKENS:
            raise StateLeaseArtifactError(
                f"{method} trajectory must contain {FROZEN_STAGE_A_ALIGNED_TOKENS} writes"
            )
        accumulator = TrajectoryNmseAccumulator()
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, context=f"{method}.trajectory[{index}]")
            _exact_keys(
                row,
                {"write_index", "per_layer_nmse", "layer_macro_nmse"},
                context=f"{method}.trajectory[{index}]",
            )
            _require(
                _strict_int(
                    row.get("write_index"), context=f"{method}.trajectory[{index}].write_index"
                )
                == index,
                f"{method} trajectory writes are not ordered at {index}",
            )
            per_layer = _mapping(
                row.get("per_layer_nmse"),
                context=f"{method}.trajectory[{index}].per_layer_nmse",
            )
            _exact_keys(
                per_layer,
                expected_layer_keys,
                context=f"{method}.trajectory[{index}].per_layer_nmse",
            )
            layer_values = {
                layer: _finite_float(
                    per_layer[str(layer)],
                    context=f"{method}.trajectory[{index}].layer[{layer}]",
                )
                for layer in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
            }
            _require(
                all(value >= 0 for value in layer_values.values()),
                f"{method} trajectory contains a negative NMSE at write {index}",
            )
            macro = _finite_float(
                row.get("layer_macro_nmse"),
                context=f"{method}.trajectory[{index}].layer_macro_nmse",
            )
            _require(
                macro == sum(layer_values.values()) / len(layer_values),
                f"{method} trajectory macro differs from its layers at write {index}",
            )
            if method == FP32_METHOD:
                _require(
                    macro == 0.0 and all(value == 0.0 for value in layer_values.values()),
                    f"{method} trajectory is not the exact zero reference at write {index}",
                )
            accumulator.append(layer_values)
        summary = dict(accumulator.summary())
        _exact_value(
            stored[method],
            summary,
            context=f"evidence.trajectory_nmse.{method}",
        )
        recomputed[method] = summary
    return recomputed


def _candidate_tensor_names(method: str) -> set[str]:
    if method == RHT_CQER_METHOD:
        suffixes = (
            "checkpoint.high_payload",
            "checkpoint.low_payload",
            "checkpoint.precision_mask",
            "checkpoint.scales",
            "query_energy_ema",
        )
        return {
            f"layer.{layer}.{suffix}"
            for layer in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
            for suffix in suffixes
        }
    if method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        suffixes = (
            "checkpoint.high_payload",
            "checkpoint.low_payload",
            "checkpoint.precision_mask",
            "checkpoint.scales",
            "log_decay_buffer",
            "normalized_key_buffer",
            "query_energy_ema",
            "update_buffer",
            "valid_count",
        )
        return {
            f"layer.{layer}.{suffix}"
            for layer in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
            for suffix in suffixes
        }
    if method == "expanded_rht_q4_q8":
        suffixes = ("precision_mask", "q4_payload", "q8_payload", "scales")
    elif method == "rht_q4_q6_q8":
        suffixes = ("precision_codes", "q4_payload", "q6_payload", "q8_payload", "scales")
    elif method == "rht_residual_q4":
        suffixes = (
            "base_q4_payload",
            "base_scales",
            "lease_mask",
            "residual_q4_payload",
            "residual_scales",
        )
    else:  # pragma: no cover - all callers are constrained by QUALITY_METHODS
        raise StateLeaseArtifactError(f"unknown storage method {method!r}")
    return {
        *(f"checkpoint.layer_{layer}.{suffix}" for layer in range(18) for suffix in suffixes),
        "checkpoint.query_energy_ema",
        "checkpoint.reserved_padding",
    }


def _shared_tensor_names() -> set[str]:
    recurrent = set(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)
    names = {f"layer.{layer}.conv.0" for layer in recurrent}
    for layer in set(range(24)) - recurrent:
        names.add(f"layer.{layer}.keys")
        names.add(f"layer.{layer}.values")
    return names


def _validate_tensor_schema(
    value: object,
    *,
    expected_names: set[str],
    context: str,
    allow_empty: bool,
    exact_storage: bool,
) -> int:
    schema = _mapping(value, context=context)
    _exact_keys(schema, expected_names, context=context)
    total = 0
    for name in sorted(expected_names):
        row = _mapping(schema[name], context=f"{context}.{name}")
        _exact_keys(row, _TENSOR_SCHEMA_KEYS, context=f"{context}.{name}")
        dtype = row.get("dtype")
        _require(dtype in _ELEMENT_BYTES, f"{context}.{name}.dtype is unsupported")
        shape = row.get("shape")
        if not isinstance(shape, list):
            raise StateLeaseArtifactError(f"{context}.{name}.shape must be an array")
        dimensions = [
            _strict_int(dimension, context=f"{context}.{name}.shape") for dimension in shape
        ]
        _require(
            all(dimension >= 0 for dimension in dimensions),
            f"{context}.{name}.shape contains a negative dimension",
        )
        logical = _strict_int(
            row.get("logical_bytes"), context=f"{context}.{name}.logical_bytes"
        )
        storage = _strict_int(
            row.get("storage_bytes"), context=f"{context}.{name}.storage_bytes"
        )
        _require(logical >= 0 and storage >= logical, f"{context}.{name} byte counts are invalid")
        _require(allow_empty or logical > 0, f"{context}.{name} must be non-empty")
        _require(
            logical == math.prod(dimensions) * _ELEMENT_BYTES[str(dtype)],
            f"{context}.{name}.logical_bytes does not match dtype and shape",
        )
        _require(
            not exact_storage or storage == logical,
            f"{context}.{name}.storage_bytes must equal logical_bytes",
        )
        total += storage
    return total


def _recompute_storage_contracts(
    evidence: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    storage = _mapping(evidence.get("storage"), context="evidence.storage")
    receipts = _mapping(evidence.get("storage_contracts"), context="evidence.storage_contracts")
    _exact_keys(storage, set(ALL_METHODS), context="evidence.storage")
    _exact_keys(receipts, set(QUALITY_METHODS), context="evidence.storage_contracts")
    _exact_value(
        storage[FP32_METHOD],
        {"recurrent_state_bytes": _FULL_PRECISION_STATE_BYTES},
        context=f"evidence.storage.{FP32_METHOD}",
    )
    shared_names = _shared_tensor_names()
    reference_shared_schema: object | None = None
    recomputed: dict[str, dict[str, object]] = {}
    for method in QUALITY_METHODS:
        row = _mapping(storage[method], context=f"evidence.storage.{method}")
        expected_bytes = (
            _HISTORICAL_RHT_BYTES if method == RHT_CQER_METHOD else FROZEN_STATELEASE_RESIDENT_BYTES
        )
        for key, expected in {
            "runtime_storage_contract_passed": True,
            "runtime_reachable_tensor_storage_closure_passed": True,
            "persistent_raw_state_bytes": 0,
            "persistent_fp32_state_mirror": False,
            "candidate_persistent_storage_bytes": expected_bytes,
            "shared_persistent_storage_bytes": _SHARED_PERSISTENT_BYTES,
            "shared_persistent_tensor_count": len(shared_names),
            "full_precision_equivalent_bytes": _FULL_PRECISION_STATE_BYTES,
        }.items():
            _exact_value(row.get(key), expected, context=f"evidence.storage.{method}.{key}")
        candidate_names = _candidate_tensor_names(method)
        _exact_value(
            row.get("candidate_persistent_tensor_count"),
            len(candidate_names),
            context=f"evidence.storage.{method}.candidate_persistent_tensor_count",
        )
        candidate_total = _validate_tensor_schema(
            row.get("candidate_tensor_schema"),
            expected_names=candidate_names,
            context=f"evidence.storage.{method}.candidate_tensor_schema",
            allow_empty=True,
            exact_storage=True,
        )
        shared_total = _validate_tensor_schema(
            row.get("shared_tensor_schema"),
            expected_names=shared_names,
            context=f"evidence.storage.{method}.shared_tensor_schema",
            allow_empty=False,
            exact_storage=False,
        )
        _require(candidate_total == expected_bytes, f"{method} candidate schema bytes drifted")
        _require(
            shared_total == _SHARED_PERSISTENT_BYTES,
            f"{method} shared schema bytes drifted",
        )
        candidate_schema_hash = hashlib.sha256(
            canonical_json_bytes(row.get("candidate_tensor_schema"))
        ).hexdigest()
        shared_schema_hash = hashlib.sha256(
            canonical_json_bytes(row.get("shared_tensor_schema"))
        ).hexdigest()
        _require(
            candidate_schema_hash == _CANDIDATE_SCHEMA_SHA256[method],
            f"{method} candidate tensor dtype/shape schema differs from the frozen identity",
        )
        _require(
            shared_schema_hash == _SHARED_SCHEMA_SHA256,
            f"{method} shared tensor dtype/shape schema differs from the frozen identity",
        )
        if reference_shared_schema is None:
            reference_shared_schema = row.get("shared_tensor_schema")
        else:
            _exact_value(
                row.get("shared_tensor_schema"),
                reference_shared_schema,
                context=f"evidence.storage.{method}.shared_tensor_schema_identity",
            )

        receipt: dict[str, object] = {"passed": True, "resident_bytes": expected_bytes}
        if method == RHT_CQER_METHOD:
            for key, expected in {
                "resident_bytes": _HISTORICAL_RHT_CHECKPOINT_BYTES,
                "resident_bytes_including_selector": _HISTORICAL_RHT_BYTES,
                "high_precision_groups": 1_976,
            }.items():
                _exact_value(row.get(key), expected, context=f"evidence.storage.{method}.{key}")
        elif method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
            for key, expected in {
                "resident_bytes_including_statelease": FROZEN_STATELEASE_RESIDENT_BYTES,
                "exact_row_plan_identity_verified": True,
                "authenticated_exact_row_plan_sha256": _EXACT_ROW_PLAN_SHA256,
                "forward_transaction_active": False,
                "stage0_independently_verified_no_hidden_fp32_state_mirror": True,
            }.items():
                _exact_value(row.get(key), expected, context=f"evidence.storage.{method}.{key}")
            receipt.update(
                {
                    "exact_row_plan_identity_verified": True,
                    "authenticated_exact_row_plan_sha256": _EXACT_ROW_PLAN_SHA256,
                }
            )
            if method in FIXED_REPLAY_METHODS:
                for key, expected in {
                    "logical_resident_capacity_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                    "capacity_fully_allocated": True,
                    "off_budget": False,
                    "baseline_mode": method,
                }.items():
                    _exact_value(
                        row.get(key), expected, context=f"evidence.storage.{method}.{key}"
                    )
        else:
            for key, expected in {
                "resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                "expected_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                "checkpoint_present": True,
                "codec": method,
                "forward_transaction_active": False,
                "stage0_independently_verified_no_hidden_fp32_state_mirror": True,
            }.items():
                _exact_value(row.get(key), expected, context=f"evidence.storage.{method}.{key}")
        _exact_value(
            receipts[method],
            receipt,
            context=f"evidence.storage_contracts.{method}",
        )
        recomputed[method] = receipt
    return recomputed


def _validate_provenance(evidence: Mapping[str, object]) -> bool:
    authentication = _mapping(
        evidence.get("input_authentication"), context="evidence.input_authentication"
    )
    _require(
        authentication.get("all_passed_before_one_run_seal_quality_data_or_model_weights")
        is True,
        "pre-seal authentication did not pass",
    )
    stage0 = _mapping(
        authentication.get("production_stage0"),
        context="evidence.input_authentication.production_stage0",
    )
    for key, expected in {
        "experiment_stage0_complete": True,
        "status": "production_stage0_pass",
        "independent_imports": True,
        "weights_only_load": True,
        "protected_mbpp_window_accessed": False,
        "quality_data_accessed": False,
    }.items():
        _exact_value(stage0.get(key), expected, context=f"production_stage0.{key}")

    repository = _mapping(evidence.get("repository"), context="evidence.repository")
    for key in (
        "commit_transition_authorized",
        "empty_tree_one_run_seal",
        "source_tree_stable",
    ):
        _require(repository.get(key) is True, f"evidence.repository.{key} must be true")
    for endpoint in ("start", "end"):
        state = _mapping(repository.get(endpoint), context=f"evidence.repository.{endpoint}")
        _require(state.get("worktree_clean") is True, f"repository {endpoint} was not clean")
        _exact_value(state.get("porcelain"), [], context=f"repository.{endpoint}.porcelain")

    sources = _mapping(evidence.get("source_files"), context="evidence.source_files")
    _require(sources.get("stable") is True, "source_files.stable must be true")
    _exact_value(
        sources.get("sha256_end"),
        sources.get("sha256_start"),
        context="evidence.source_files.sha256_end",
    )
    return True


def _step(
    name: str,
    function: Any,
    *,
    checks: dict[str, bool],
    errors: list[str],
) -> Any | None:
    try:
        result = function()
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        checks[name] = False
        errors.append(f"{name}: {type(error).__name__}: {error}")
        return None
    checks[name] = True
    return result


def _recompute_gate(
    evidence: Mapping[str, object],
    *,
    metrics: Mapping[str, object],
    trajectory: Mapping[str, object],
    storage_contracts: Mapping[str, object],
    provenance: object,
    artifact_integrity: bool,
) -> dict[str, object]:
    """Recompute the frozen gate while keeping malformed containers fail-closed."""

    del storage_contracts  # Its successful recomputation is bound by artifact_integrity.
    diagnostics = _mapping(evidence.get("diagnostics"), context="evidence.diagnostics")
    updates = _mapping(evidence.get("update_evidence"), context="evidence.update_evidence")
    storage = _mapping(evidence.get("storage"), context="evidence.storage")
    statelease_metrics = {
        method: _mapping(metrics.get(method), context=f"metrics.{method}")
        for method in QUALITY_METHODS
    }
    statelease_trajectory = {
        method: _mapping(trajectory.get(method), context=f"trajectory.{method}")
        for method in QUALITY_METHODS
    }
    return evaluate_statelease_stage_a_gate(
        aligned_metrics=statelease_metrics,
        trajectory_nmse_auc=statelease_trajectory,
        statelease_storage=_mapping(
            storage.get(STATELEASE_METHOD),
            context=f"evidence.storage.{STATELEASE_METHOD}",
        ),
        statelease_diagnostics=diagnostics.get(STATELEASE_METHOD),  # type: ignore[arg-type]
        statelease_update_evidence=updates.get(STATELEASE_METHOD),  # type: ignore[arg-type]
        stage0_complete=provenance is True,
        artifact_integrity=artifact_integrity,
    )


def verify_experiment012_statelease_stage_a(
    path: str | Path,
    *,
    expected_file_sha256: str = EXPERIMENT012_FILE_SHA256,
    expected_canonical_evidence_sha256: str = EXPERIMENT012_CANONICAL_EVIDENCE_SHA256,
) -> dict[str, Any]:
    """Verify the frozen Experiment 012 artifact without model or dataset access.

    The two expected hashes default to the committed public artifact.  Callers
    testing another byte-for-byte identity may explicitly supply both hashes;
    all frozen protocol, raw-evidence, storage, and gate checks still apply.
    The returned report is JSON serializable and fails closed via ``valid``.
    """

    artifact_path = Path(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        raw = artifact_path.read_bytes()
    except OSError as error:
        return {
            "artifact_path": str(artifact_path),
            "checks": {"readable": False},
            "computed_canonical_evidence_sha256": None,
            "errors": [f"readable: {type(error).__name__}: {error}"],
            "file_sha256": None,
            "recorded_canonical_evidence_sha256": None,
            "recomputed": None,
            "valid": False,
        }
    checks["readable"] = True
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if not _valid_sha256(expected_file_sha256):
        errors.append("file_hash: expected file SHA256 is not 64 hexadecimal characters")
        checks["file_hash"] = False
    elif file_sha256 != expected_file_sha256.lower():
        errors.append(
            "file_hash: expected "
            f"{expected_file_sha256.lower()}, computed {file_sha256}"
        )
        checks["file_hash"] = False
    else:
        checks["file_hash"] = True
    if not _valid_sha256(expected_canonical_evidence_sha256):
        errors.append("canonical_hash: expected canonical SHA256 is invalid")
        checks["canonical_hash"] = False

    try:
        document_value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        document = _mapping(document_value, context="artifact")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        checks["strict_json"] = False
        errors.append(f"strict_json: {type(error).__name__}: {error}")
        return {
            "artifact_path": str(artifact_path),
            "checks": checks,
            "computed_canonical_evidence_sha256": None,
            "errors": errors,
            "file_sha256": file_sha256,
            "recorded_canonical_evidence_sha256": None,
            "recomputed": None,
            "valid": False,
        }
    checks["strict_json"] = True

    evidence = _step(
        "artifact_contract",
        lambda: _validate_artifact_contract(document),
        checks=checks,
        errors=errors,
    )
    recorded_canonical = document.get("canonical_evidence_sha256")
    computed_canonical: str | None = None
    if isinstance(evidence, Mapping):
        computed_canonical = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
        canonical_ok = True
        if not _valid_sha256(recorded_canonical):
            errors.append("canonical_hash: recorded canonical SHA256 is invalid")
            canonical_ok = False
        elif str(recorded_canonical).lower() != computed_canonical:
            errors.append(
                "canonical_hash: recorded "
                f"{str(recorded_canonical).lower()}, computed {computed_canonical}"
            )
            canonical_ok = False
        if _valid_sha256(expected_canonical_evidence_sha256) and (
            computed_canonical != expected_canonical_evidence_sha256.lower()
        ):
            errors.append(
                "canonical_hash: expected "
                f"{expected_canonical_evidence_sha256.lower()}, computed {computed_canonical}"
            )
            canonical_ok = False
        checks["canonical_hash"] = canonical_ok
    else:
        checks.setdefault("canonical_hash", False)

    recomputed_metrics = None
    recomputed_trajectory = None
    recomputed_storage = None
    provenance = None
    recomputed_gate = None
    if isinstance(evidence, Mapping):
        _step(
            "task_and_token_contract",
            lambda: _validate_task_and_token_contract(evidence),
            checks=checks,
            errors=errors,
        )
        recomputed_metrics = _step(
            "aligned_metrics",
            lambda: _recompute_aligned_metrics(evidence),
            checks=checks,
            errors=errors,
        )
        recomputed_trajectory = _step(
            "trajectory",
            lambda: _recompute_trajectory(evidence),
            checks=checks,
            errors=errors,
        )
        recomputed_storage = _step(
            "storage_contracts",
            lambda: _recompute_storage_contracts(evidence),
            checks=checks,
            errors=errors,
        )
        provenance = _step(
            "provenance",
            lambda: _validate_provenance(evidence),
            checks=checks,
            errors=errors,
        )

    prerequisite_names = (
        "file_hash",
        "strict_json",
        "artifact_contract",
        "canonical_hash",
        "task_and_token_contract",
        "aligned_metrics",
        "trajectory",
        "storage_contracts",
        "provenance",
    )
    prerequisites = (
        isinstance(evidence, Mapping)
        and isinstance(recomputed_metrics, Mapping)
        and isinstance(recomputed_trajectory, Mapping)
        and isinstance(recomputed_storage, Mapping)
        and all(checks.get(name) is True for name in prerequisite_names)
    )
    if prerequisites:
        recomputed_gate = _step(
            "gate_recomputed",
            lambda: _recompute_gate(
                evidence,
                metrics=recomputed_metrics,
                trajectory=recomputed_trajectory,
                storage_contracts=recomputed_storage,
                provenance=provenance,
                artifact_integrity=True,
            ),
            checks=checks,
            errors=errors,
        )
        if isinstance(recomputed_gate, Mapping):
            gate_errors: list[str] = []
            _collect_exact_differences(
                evidence.get("stage_a_gate"),
                recomputed_gate,
                context="evidence.stage_a_gate",
                errors=gate_errors,
            )
            checks["recorded_gate_matches"] = not gate_errors
            errors.extend(f"recorded_gate_matches: {error}" for error in gate_errors)
        else:
            checks["recorded_gate_matches"] = False
            errors.append("recorded_gate_matches: gate recomputation failed")
    else:
        checks["gate_recomputed"] = False
        checks["recorded_gate_matches"] = False
        errors.append("gate_recomputed: raw inputs did not pass independent recomputation")

    valid = not errors and all(checks.values())
    return {
        "artifact_path": str(artifact_path),
        "checks": checks,
        "computed_canonical_evidence_sha256": computed_canonical,
        "errors": errors,
        "file_sha256": file_sha256,
        "recorded_canonical_evidence_sha256": (
            str(recorded_canonical).lower() if _valid_sha256(recorded_canonical) else None
        ),
        "recomputed": {
            "aligned_metrics": recomputed_metrics,
            "stage_a_gate": recomputed_gate,
            "storage_contracts": recomputed_storage,
            "trajectory_nmse": recomputed_trajectory,
        },
        "valid": valid,
    }


__all__ = [
    "EXPERIMENT012_ARTIFACT_KIND",
    "EXPERIMENT012_CANONICAL_EVIDENCE_SHA256",
    "EXPERIMENT012_FILE_SHA256",
    "EXPERIMENT012_SCHEMA_VERSION",
    "EXPERIMENT012_TASK_ID",
    "StateLeaseArtifactError",
    "verify_experiment012_statelease_stage_a",
]
