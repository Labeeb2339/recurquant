#!/usr/bin/env python3
"""Run the frozen Experiment 009 Stage-A right-RHT CQER screen.

This evaluator has deliberately narrow data access: it can load only the
already-open MBPP task 666.  It has no task, seed, threshold, or window flags.
Passing this one-task screen authorizes only the separately frozen Stage-B
development identity; it is not confirmation, novelty, or speed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.mixed_quantization import quantize_pack_mixed
from recurquant.packed_cache import (
    QueryEmaMixedPackedRecurrentStateCache,
    RightRhtQueryEmaMixedPackedRecurrentStateCache,
)
from recurquant.public_data import (
    load_mbpp_rows_by_task_ids,
    mbpp_manifest,
    mbpp_manifest_sha256,
    mbpp_row_sha256,
)
from recurquant.quantization import QuantizationSpec, quantize_dequantize
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.qwen35 import (
    create_qwen35_query_ema_exact_budget_cache,
    create_qwen35_right_rht_query_ema_exact_budget_cache,
)
from recurquant.rht import (
    right_rht_decode,
    right_rht_encode,
    right_rht_signs,
)
from recurquant.row_policy import ExactBudgetRowPlan

pilot = importlib.import_module(
    "scripts.pilot_evaluate_hrr" if __package__ else "pilot_evaluate_hrr"
)

SEED = 2339
ARTIFACT_KIND = "recurquant_rht_cqer32_stage_a_screen"
CQER_METHOD = "query_ema32_weighted_mse_target_fisher_quota"
RHT_METHOD = "right_rht_query_ema32_weighted_mse_target_fisher_quota"
METHODS = (CQER_METHOD, RHT_METHOD)
TARGET_FISHER_SCORE = "target_directional_fisher_difference_int4"

TASK_ID = 666
TASK_ROW_SHA256 = "b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9"
PROMPT_TOKENS = 69
CODE_TOKENS = 39
ALIGNED_TOKENS = 38
EXPECTED_STATE_WRITES = 39
EXPECTED_QUERY_TOKENS = PROMPT_TOKENS + ALIGNED_TOKENS

TARGET_PACKED_STATE_BYTES = 2_564_096
TARGET_SELECTOR_BYTES = 147_456
TARGET_TOTAL_BYTES = 2_711_552
TARGET_PAYLOAD_BYTES = 2_485_760
TARGET_SCALE_BYTES = 73_728
TARGET_MASK_BYTES = 4_608
TARGET_PROMOTED_ROWS = 1_976

MIN_STATE_SSE_REDUCTION = 0.50
MIN_DELTA_NLL_REDUCTION = 0.10
MAX_RHT_INVERSE_RELATIVE_L2 = 3e-7
EXPECTED_SIGN_SCHEDULE_SHA256 = (
    "2d5137b5ebeb325f100b34190618783b9e47bd2ce9b27b6cdf3cdc94459dabc3"
)

FROZEN_LAYER_QUOTAS = dict(pilot.CQER_FROZEN_LAYER_QUOTAS)
FROZEN_LINEAR_LAYERS = tuple(FROZEN_LAYER_QUOTAS)

SOURCE_FILES = (
    "research/EXPERIMENT_009_RHT_CQER_PROTOCOL.md",
    "scripts/screen_rht_cqer.py",
    "scripts/pilot_evaluate_hrr.py",
    "src/recurquant/__init__.py",
    "src/recurquant/cache.py",
    "src/recurquant/evaluation.py",
    "src/recurquant/evidence.py",
    "src/recurquant/metrics.py",
    "src/recurquant/mixed_quantization.py",
    "src/recurquant/packed_cache.py",
    "src/recurquant/public_data.py",
    "src/recurquant/quantization.py",
    "src/recurquant/query_energy.py",
    "src/recurquant/qwen35.py",
    "src/recurquant/rht.py",
    "src/recurquant/row_policy.py",
    "tests/test_rht.py",
    "tests/test_rht_mixed_quantization.py",
    "tests/test_right_rht_query_ema_cache.py",
    "tests/test_screen_rht_cqer.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Experiment 009 Stage-A screen on already-open MBPP task 666. "
            "No other task or threshold can be selected."
        )
    )
    parser.add_argument(
        "--selector",
        "--selector-artifact",
        dest="selector",
        type=Path,
        required=True,
        help="Authenticated frozen Experiment 007 HRR selector artifact.",
    )
    parser.add_argument(
        "--loss-selector",
        "--loss-selector-artifact",
        dest="loss_selector",
        type=Path,
        required=True,
        help="Authenticated frozen target-Fisher selector artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def validate_stage_a_identity(
    *,
    task_id: object,
    row_sha256: object,
    prompt_tokens: object,
    code_tokens: object,
    aligned_scored_tokens: object,
) -> dict[str, int | str | bool]:
    """Authenticate the immutable Stage-A row and token identity."""

    actual_task_id = _strict_int(task_id, context="task_id")
    actual_prompt = _strict_int(prompt_tokens, context="prompt_tokens")
    actual_code = _strict_int(code_tokens, context="code_tokens")
    actual_aligned = _strict_int(
        aligned_scored_tokens,
        context="aligned_scored_tokens",
    )
    if actual_task_id != TASK_ID:
        raise ValueError(f"Stage A is locked to task_id {TASK_ID}")
    if not isinstance(row_sha256, str) or row_sha256 != TASK_ROW_SHA256:
        raise ValueError("Stage-A MBPP row SHA-256 does not match the frozen identity")
    if (actual_prompt, actual_code, actual_aligned) != (
        PROMPT_TOKENS,
        CODE_TOKENS,
        ALIGNED_TOKENS,
    ):
        raise ValueError(
            "Stage-A token identity must be exactly "
            f"{PROMPT_TOKENS}/{CODE_TOKENS}/{ALIGNED_TOKENS}"
        )
    if actual_aligned != actual_code - 1:
        raise ValueError("aligned token count must equal code_tokens - 1")
    return {
        "task_id": actual_task_id,
        "row_sha256": row_sha256,
        "prompt_tokens": actual_prompt,
        "code_tokens": actual_code,
        "aligned_scored_tokens": actual_aligned,
        "authenticated_before_model_weights": True,
    }


def _evidence_mapping(record: object, *, index: int) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    evidence_dict = getattr(record, "evidence_dict", None)
    if not callable(evidence_dict):
        raise ValueError(f"state-error record {index} must be evidence or a mapping")
    converted = evidence_dict()
    if not isinstance(converted, Mapping):
        raise ValueError(f"state-error record {index} evidence must be an object")
    return dict(converted)


def aggregate_state_error_evidence(records: Sequence[object]) -> dict[str, Any]:
    """Aggregate record_evidence MSE into exact per-write state SSE.

    Cache evidence stores MSE, while the frozen gate is defined on SSE.  This
    function authenticates each shape and computes ``MSE * element_count``.
    It also assigns a causal write ordinal independently within each layer so
    the two codecs can be checked for identical closed-loop coverage.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("state-error evidence must be a sequence")
    if not records:
        raise ValueError("state-error evidence must not be empty")

    normalized: list[dict[str, Any]] = []
    layer_occurrences: defaultdict[tuple[int, int], int] = defaultdict(int)
    seen_update_indices: set[int] = set()
    layer_totals: defaultdict[int, dict[str, float | int]] = defaultdict(
        lambda: {"record_count": 0, "element_count": 0, "state_sse": 0.0}
    )
    write_totals: defaultdict[int, dict[str, float | int]] = defaultdict(
        lambda: {"record_count": 0, "element_count": 0, "state_sse": 0.0}
    )

    for record_index, raw_record in enumerate(records):
        record = _evidence_mapping(raw_record, index=record_index)
        update_index = _strict_int(
            record.get("update_index"),
            context=f"state-error record {record_index} update_index",
        )
        layer_index = _strict_int(
            record.get("layer_index"),
            context=f"state-error record {record_index} layer_index",
        )
        state_index = _strict_int(
            record.get("state_index"),
            context=f"state-error record {record_index} state_index",
        )
        if update_index < 0 or layer_index < 0 or state_index < 0:
            raise ValueError("state-error indices must be non-negative")
        if update_index in seen_update_indices:
            raise ValueError(f"duplicate state-error update_index {update_index}")
        seen_update_indices.add(update_index)

        raw_shape = record.get("shape")
        if (
            isinstance(raw_shape, (str, bytes))
            or not isinstance(raw_shape, Sequence)
            or not raw_shape
        ):
            raise ValueError(f"state-error record {record_index} shape is invalid")
        shape = tuple(
            _strict_int(dimension, context=f"state-error record {record_index} shape")
            for dimension in raw_shape
        )
        if any(dimension <= 0 for dimension in shape):
            raise ValueError("state-error shape dimensions must be positive")
        element_count = math.prod(shape)
        mse = _finite_float(
            record.get("mean_squared_error"),
            context=f"state-error record {record_index} mean_squared_error",
        )
        maximum = _finite_float(
            record.get("max_absolute_error"),
            context=f"state-error record {record_index} max_absolute_error",
        )
        relative_l2 = _finite_float(
            record.get("relative_l2_error"),
            context=f"state-error record {record_index} relative_l2_error",
        )
        if mse < 0 or maximum < 0 or relative_l2 < 0:
            raise ValueError("state-error magnitudes must be non-negative")
        state_sse = mse * element_count
        if not math.isfinite(state_sse):
            raise ValueError("state-error SSE must be finite")

        occurrence_key = (layer_index, state_index)
        write_ordinal = layer_occurrences[occurrence_key]
        layer_occurrences[occurrence_key] += 1
        if "element_count" in record and _strict_int(
            record["element_count"],
            context=f"state-error record {record_index} element_count",
        ) != element_count:
            raise ValueError("state-error element_count does not match its shape")
        if "write_ordinal" in record and _strict_int(
            record["write_ordinal"],
            context=f"state-error record {record_index} write_ordinal",
        ) != write_ordinal:
            raise ValueError("state-error write_ordinal does not match causal order")
        if "state_sse" in record and not math.isclose(
            _finite_float(
                record["state_sse"],
                context=f"state-error record {record_index} state_sse",
            ),
            state_sse,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("state-error state_sse does not match MSE times elements")
        normalized_record = {
            **record,
            "shape": list(shape),
            "element_count": element_count,
            "state_sse": state_sse,
            "write_ordinal": write_ordinal,
        }
        normalized.append(normalized_record)

        for totals in (layer_totals[layer_index], write_totals[write_ordinal]):
            totals["record_count"] = int(totals["record_count"]) + 1
            totals["element_count"] = int(totals["element_count"]) + element_count
            totals["state_sse"] = float(totals["state_sse"]) + state_sse

    normalized.sort(key=lambda record: int(record["update_index"]))
    coverage = [
        {
            "write_ordinal": int(record["write_ordinal"]),
            "layer_index": int(record["layer_index"]),
            "state_index": int(record["state_index"]),
            "shape": list(record["shape"]),
        }
        for record in normalized
    ]
    total_elements = sum(int(record["element_count"]) for record in normalized)
    total_sse = math.fsum(float(record["state_sse"]) for record in normalized)
    return {
        "record_count": len(normalized),
        "element_count": total_elements,
        "aggregate_state_sse": total_sse,
        "aggregate_state_mse": total_sse / total_elements,
        "coverage": coverage,
        "per_layer": {
            str(layer): {
                "record_count": int(values["record_count"]),
                "element_count": int(values["element_count"]),
                "state_sse": float(values["state_sse"]),
            }
            for layer, values in sorted(layer_totals.items())
        },
        "per_write": {
            str(write): {
                "record_count": int(values["record_count"]),
                "element_count": int(values["element_count"]),
                "state_sse": float(values["state_sse"]),
            }
            for write, values in sorted(write_totals.items())
        },
        "records": normalized,
    }


def validate_state_error_coverage(
    state_errors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute and authenticate exact layers, writes, states, shapes, and totals."""

    if set(state_errors) != set(METHODS):
        raise ValueError("state-error evidence must contain exactly CQER and RHT-CQER")
    expected_records = len(FROZEN_LINEAR_LAYERS) * EXPECTED_STATE_WRITES
    expected_coverage = [
        {
            "write_ordinal": write,
            "layer_index": layer,
            "state_index": 0,
            "shape": [1, 16, 128, 128],
        }
        for write in range(EXPECTED_STATE_WRITES)
        for layer in FROZEN_LINEAR_LAYERS
    ]
    final_masks: dict[str, dict[int, str]] = {}
    for method, record in state_errors.items():
        raw_records = record.get("records")
        if isinstance(raw_records, (str, bytes)) or not isinstance(
            raw_records, Sequence
        ):
            raise ValueError(f"{method} must retain raw per-write state-error records")
        recomputed = aggregate_state_error_evidence(raw_records)
        if int(record.get("record_count", -1)) != expected_records:
            raise ValueError(
                f"{method} must contain exactly {expected_records} state-error records"
            )
        if recomputed["record_count"] != expected_records:
            raise ValueError(f"{method} raw state-error record count drifted")
        if recomputed["coverage"] != expected_coverage:
            raise ValueError(f"{method} raw state-error geometry or causal order drifted")
        if record.get("coverage") != expected_coverage:
            raise ValueError(f"{method} reported state-error coverage drifted")
        if _strict_int(
            record.get("element_count"),
            context=f"{method} aggregate element_count",
        ) != recomputed["element_count"]:
            raise ValueError(f"{method} aggregate element count was not recomputed")
        for field in ("aggregate_state_sse", "aggregate_state_mse"):
            if not math.isclose(
                _finite_float(record.get(field), context=f"{method} {field}"),
                float(recomputed[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{method} {field} does not match raw records")
        per_layer = record.get("per_layer")
        per_write = record.get("per_write")
        if not isinstance(per_layer, Mapping) or set(per_layer) != {
            str(layer) for layer in FROZEN_LINEAR_LAYERS
        }:
            raise ValueError(f"{method} state-error layers do not match the frozen geometry")
        if not isinstance(per_write, Mapping) or set(per_write) != {
            str(write) for write in range(EXPECTED_STATE_WRITES)
        }:
            raise ValueError(f"{method} state-error writes do not match the frozen run")
        if any(
            int(per_layer[str(layer)].get("record_count", -1)) != EXPECTED_STATE_WRITES
            for layer in FROZEN_LINEAR_LAYERS
        ):
            raise ValueError(f"{method} did not record every layer on every write")
        if any(
            int(per_write[str(write)].get("record_count", -1))
            != len(FROZEN_LINEAR_LAYERS)
            for write in range(EXPECTED_STATE_WRITES)
        ):
            raise ValueError(f"{method} did not record every layer within every write")
        for grouping_name, reported, expected in (
            ("per-layer", per_layer, recomputed["per_layer"]),
            ("per-write", per_write, recomputed["per_write"]),
        ):
            for key, expected_values in expected.items():
                reported_values = reported[key]
                for field in ("record_count", "element_count"):
                    if _strict_int(
                        reported_values.get(field),
                        context=f"{method} {grouping_name} {key} {field}",
                    ) != int(expected_values[field]):
                        raise ValueError(
                            f"{method} {grouping_name} {key} {field} drifted"
                        )
                if not math.isclose(
                    _finite_float(
                        reported_values.get("state_sse"),
                        context=f"{method} {grouping_name} {key} state_sse",
                    ),
                    float(expected_values["state_sse"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{method} {grouping_name} {key} state_sse drifted"
                    )
        expected_selection_method = (
            "query_ema32_weighted_aligned_mse_reduction"
            if method == CQER_METHOD
            else RHT_METHOD
        )
        method_final_masks: dict[int, str] = {}
        for index, raw_record in enumerate(recomputed["records"]):
            layer = FROZEN_LINEAR_LAYERS[index % len(FROZEN_LINEAR_LAYERS)]
            write = index // len(FROZEN_LINEAR_LAYERS)
            if (
                raw_record.get("layer_index") != layer
                or raw_record.get("state_index") != 0
                or raw_record.get("shape") != [1, 16, 128, 128]
                or raw_record.get("update_index") != index
                or raw_record.get("write_ordinal") != write
                or raw_record.get("source_dtype") != "torch.float32"
                or raw_record.get("selection_method") != expected_selection_method
                or raw_record.get("total_groups") != 2048
                or raw_record.get("high_precision_groups")
                != FROZEN_LAYER_QUOTAS[layer]
            ):
                raise ValueError(
                    f"{method} raw state-error record {index} contract drifted"
                )
            mask_hash = raw_record.get("high_precision_mask_sha256")
            if (
                not isinstance(mask_hash, str)
                or len(mask_hash) != 64
                or any(character not in "0123456789abcdef" for character in mask_hash)
            ):
                raise ValueError(f"{method} raw state-error mask hash is invalid")
            if write == EXPECTED_STATE_WRITES - 1:
                method_final_masks[layer] = mask_hash
        final_masks[method] = method_final_masks
    if state_errors[CQER_METHOD].get("coverage") != state_errors[RHT_METHOD].get(
        "coverage"
    ):
        raise ValueError("CQER and RHT-CQER state-error coverage differs")
    return {
        "passed": True,
        "record_count_per_method": expected_records,
        "state_writes": EXPECTED_STATE_WRITES,
        "linear_layers": list(FROZEN_LINEAR_LAYERS),
        "final_mask_sha256s": {
            method: {str(layer): value for layer, value in masks.items()}
            for method, masks in final_masks.items()
        },
    }


def audit_selector_diagnostics(
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Authenticate exact quotas and one-stage/one-consume query handshakes."""

    if set(diagnostics) != set(METHODS):
        raise ValueError("selector diagnostics must contain exactly CQER and RHT-CQER")
    expected_selection_methods = {
        CQER_METHOD: "query_ema32_weighted_aligned_mse_reduction",
        RHT_METHOD: RHT_METHOD,
    }
    audited: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        records = diagnostics[method]
        if len(records) != len(FROZEN_LINEAR_LAYERS):
            raise ValueError(f"{method} must report every frozen recurrent layer")
        by_layer: dict[int, Mapping[str, Any]] = {}
        for record in records:
            layer = _strict_int(record.get("layer_index"), context="diagnostic layer_index")
            if layer in by_layer:
                raise ValueError(f"{method} repeats diagnostic layer {layer}")
            by_layer[layer] = record
        if set(by_layer) != set(FROZEN_LINEAR_LAYERS):
            raise ValueError(f"{method} diagnostic layer identity drifted")

        method_rows: list[dict[str, Any]] = []
        for layer in FROZEN_LINEAR_LAYERS:
            record = by_layer[layer]
            quota = _strict_int(record.get("quota"), context=f"{method} layer {layer} quota")
            selected = _strict_int(
                record.get("current_selected_count"),
                context=f"{method} layer {layer} current_selected_count",
            )
            updates = _strict_int(
                record.get("state_updates"),
                context=f"{method} layer {layer} state_updates",
            )
            staged = _strict_int(
                record.get("observations_staged"),
                context=f"{method} layer {layer} observations_staged",
            )
            committed = _strict_int(
                record.get("observations_committed"),
                context=f"{method} layer {layer} observations_committed",
            )
            observed = _strict_int(
                record.get("tokens_observed"),
                context=f"{method} layer {layer} tokens_observed",
            )
            if quota != FROZEN_LAYER_QUOTAS[layer] or selected != quota:
                raise ValueError(f"{method} layer {layer} did not realize its frozen quota")
            if (updates, staged, committed) != (
                EXPECTED_STATE_WRITES,
                EXPECTED_STATE_WRITES,
                EXPECTED_STATE_WRITES,
            ):
                raise ValueError(
                    f"{method} layer {layer} query observation handshake is incomplete"
                )
            if observed != EXPECTED_QUERY_TOKENS:
                raise ValueError(f"{method} layer {layer} query token count drifted")
            if record.get("pending_observation") is not False:
                raise ValueError(f"{method} layer {layer} retains a pending observation")
            if record.get("confirmation_two") is not False:
                raise ValueError(f"{method} layer {layer} unexpectedly enabled Confirmation-2")
            if record.get("selection_method") != expected_selection_methods[method]:
                raise ValueError(f"{method} layer {layer} selection method drifted")
            if method == RHT_METHOD and (
                record.get("state_codec") != "right_rht_sha256_signs_v1"
                or record.get("state_codec_seed") != SEED
                or record.get("state_codec_axis") != "value"
                or record.get("state_codec_normalization") != "orthonormal"
                or record.get("state_codec_persistent_tensor_bytes") != 0
            ):
                raise ValueError(f"{method} layer {layer} codec contract drifted")
            mask_hash = record.get("current_mask_sha256")
            if (
                not isinstance(mask_hash, str)
                or len(mask_hash) != 64
                or any(character not in "0123456789abcdef" for character in mask_hash)
            ):
                raise ValueError(f"{method} layer {layer} mask hash is invalid")
            method_rows.append(
                {
                    "layer_index": layer,
                    "quota": quota,
                    "state_updates": updates,
                    "observations_staged": staged,
                    "observations_committed": committed,
                    "tokens_observed": observed,
                    "selection_method": expected_selection_methods[method],
                    "current_mask_sha256": mask_hash,
                }
            )
        audited[method] = method_rows
    return {
        "passed": True,
        "quota_sum": sum(FROZEN_LAYER_QUOTAS.values()),
        "state_writes": EXPECTED_STATE_WRITES,
        "query_tokens": EXPECTED_QUERY_TOKENS,
        "methods": audited,
    }


def _metric_summary(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    expected_tokens: int,
    context: str,
) -> dict[str, dict[str, float | int | bool]]:
    if set(metrics) != set(METHODS):
        raise ValueError(f"{context} metrics must contain exactly CQER and RHT-CQER")
    normalized: dict[str, dict[str, float | int | bool]] = {}
    fields = (
        "mean_kl",
        "cvar95_kl",
        "max_kl",
        "top1_agreement",
        "reference_nll",
        "candidate_nll",
        "delta_nll",
    )
    for method in METHODS:
        record = metrics[method]
        token_count = _strict_int(
            record.get("token_count"),
            context=f"{context} {method} token_count",
        )
        if token_count != expected_tokens:
            raise ValueError(
                f"{context} {method} must score exactly {expected_tokens} tokens"
            )
        if record.get("all_logits_finite") is not True:
            raise ValueError(f"{context} {method} logits are not all finite")
        normalized[method] = {
            "token_count": token_count,
            "all_logits_finite": True,
            **{
                field: _finite_float(
                    record.get(field),
                    context=f"{context} {method} {field}",
                )
                for field in fields
            },
        }
        values = normalized[method]
        mean_kl = float(values["mean_kl"])
        cvar95_kl = float(values["cvar95_kl"])
        max_kl = float(values["max_kl"])
        top1 = float(values["top1_agreement"])
        reference_nll = float(values["reference_nll"])
        candidate_nll = float(values["candidate_nll"])
        delta_nll = float(values["delta_nll"])
        if mean_kl < 0 or cvar95_kl < 0 or max_kl < 0:
            raise ValueError(f"{context} {method} KL metrics must be non-negative")
        if mean_kl > cvar95_kl or cvar95_kl > max_kl:
            raise ValueError(f"{context} {method} KL summary ordering is invalid")
        if not 0.0 <= top1 <= 1.0:
            raise ValueError(f"{context} {method} top1_agreement must be in [0, 1]")
        if reference_nll < 0 or candidate_nll < 0:
            raise ValueError(f"{context} {method} NLL metrics must be non-negative")
        if not math.isclose(
            delta_nll,
            candidate_nll - reference_nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{context} {method} delta_nll does not equal candidate minus reference"
            )
    reference_values = [
        float(normalized[method]["reference_nll"]) for method in METHODS
    ]
    if any(
        not math.isclose(
            value,
            reference_values[0],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in reference_values[1:]
    ):
        raise ValueError(f"{context} methods do not share one reference NLL")
    return normalized


def _gate_check(function: Any) -> dict[str, Any]:
    try:
        result = function()
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "error": str(error)}
    if isinstance(result, Mapping):
        record = dict(result)
        record["passed"] = record.get("passed") is True
        return record
    return {"passed": bool(result)}


def evaluate_stage_a_gate(
    *,
    aligned_metrics: Mapping[str, Mapping[str, Any]],
    full_code_metrics: Mapping[str, Mapping[str, Any]],
    storage: Mapping[str, Mapping[str, Any]],
    selector_diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    state_errors: Mapping[str, Mapping[str, Any]],
    unit_evidence: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the frozen conjunction in the Experiment 009 protocol."""

    aligned: dict[str, dict[str, float | int | bool]] | None = None
    full: dict[str, dict[str, float | int | bool]] | None = None

    def repository_check() -> dict[str, Any]:
        fields = (
            "repository_clean_at_start",
            "repository_clean_at_end",
            "repository_commit_stable",
            "source_hashes_stable",
        )
        if any(integrity.get(field) is not True for field in fields):
            raise ValueError("repository cleanliness, commit, or source hashes were not stable")
        return {"passed": True, **{field: True for field in fields}}

    def identity_check() -> dict[str, Any]:
        if integrity.get("identity_authenticated_before_model_weights") is not True:
            raise ValueError("row and token identity was not authenticated before model weights")
        if integrity.get("protected_window_8_16_accessed") is not False:
            raise ValueError("protected ranked window [8, 16) was accessed")
        return {
            "passed": True,
            "identity_authenticated_before_model_weights": True,
            "protected_window_8_16_accessed": False,
        }

    def storage_check() -> dict[str, Any]:
        if set(storage) != set(METHODS):
            raise ValueError("storage must contain exactly CQER and RHT-CQER")
        expected = {
            "payload_bytes": TARGET_PAYLOAD_BYTES,
            "scale_bytes": TARGET_SCALE_BYTES,
            "mask_bytes": TARGET_MASK_BYTES,
            "resident_bytes": TARGET_PACKED_STATE_BYTES,
            "selector_auxiliary_bytes": TARGET_SELECTOR_BYTES,
            "resident_bytes_including_selector": TARGET_TOTAL_BYTES,
            "high_precision_groups": TARGET_PROMOTED_ROWS,
        }
        for method in METHODS:
            for field, value in expected.items():
                if _strict_int(
                    storage[method].get(field),
                    context=f"{method} storage {field}",
                ) != value:
                    raise ValueError(f"{method} storage {field} drifted")
        return {"passed": True, **expected}

    def finite_and_handshake_check() -> dict[str, Any]:
        nonlocal aligned, full
        aligned = _metric_summary(
            aligned_metrics,
            expected_tokens=ALIGNED_TOKENS,
            context="aligned",
        )
        full = _metric_summary(
            full_code_metrics,
            expected_tokens=CODE_TOKENS,
            context="full-code",
        )
        selector_audit = audit_selector_diagnostics(selector_diagnostics)
        coverage = validate_state_error_coverage(state_errors)
        for method in METHODS:
            diagnostic_masks = {
                str(record["layer_index"]): record["current_mask_sha256"]
                for record in selector_audit["methods"][method]
            }
            if diagnostic_masks != coverage["final_mask_sha256s"][method]:
                raise ValueError(
                    f"{method} final state-error masks do not match selector diagnostics"
                )
        return {
            "passed": True,
            "all_logits_and_metrics_finite": True,
            "selector_audit": selector_audit,
            "state_error_coverage": coverage,
        }

    def unit_check() -> dict[str, Any]:
        inverse = _finite_float(
            unit_evidence.get("inverse_relative_l2"),
            context="unit inverse_relative_l2",
        )
        if not inverse < MAX_RHT_INVERSE_RELATIVE_L2:
            raise ValueError("right-RHT inverse relative L2 did not meet the frozen bound")
        if unit_evidence.get("physical_pack_matches_transformed_qdq") is not True:
            raise ValueError("physical RHT pack did not exactly match transformed QDQ")
        if unit_evidence.get("physical_pack_max_abs_difference") != 0.0:
            raise ValueError("physical RHT pack reconstruction was not bit-exact")
        if unit_evidence.get("sign_schedule_sha256") != EXPECTED_SIGN_SCHEDULE_SHA256:
            raise ValueError("right-RHT sign schedule hash drifted")
        return {
            "passed": True,
            "inverse_relative_l2": inverse,
            "maximum_inverse_relative_l2": MAX_RHT_INVERSE_RELATIVE_L2,
            "physical_pack_matches_transformed_qdq": True,
            "sign_schedule_sha256": EXPECTED_SIGN_SCHEDULE_SHA256,
        }

    def state_sse_check() -> dict[str, Any]:
        baseline = _finite_float(
            state_errors[CQER_METHOD].get("aggregate_state_sse"),
            context="CQER aggregate state SSE",
        )
        candidate = _finite_float(
            state_errors[RHT_METHOD].get("aggregate_state_sse"),
            context="RHT aggregate state SSE",
        )
        if baseline <= 0 or candidate < 0:
            raise ValueError("state SSE baseline must be positive and candidate non-negative")
        reduction = (baseline - candidate) / baseline
        return {
            "passed": reduction >= MIN_STATE_SSE_REDUCTION,
            "cqer_state_sse": baseline,
            "rht_state_sse": candidate,
            "relative_reduction": reduction,
            "minimum_relative_reduction": MIN_STATE_SSE_REDUCTION,
        }

    def delta_nll_check() -> dict[str, Any]:
        normalized = aligned or _metric_summary(
            aligned_metrics,
            expected_tokens=ALIGNED_TOKENS,
            context="aligned",
        )
        baseline = float(normalized[CQER_METHOD]["delta_nll"])
        candidate = float(normalized[RHT_METHOD]["delta_nll"])
        if baseline <= 0:
            raise ValueError("CQER aligned excess NLL must be positive for a relative gate")
        reduction = (baseline - candidate) / baseline
        return {
            "passed": reduction >= MIN_DELTA_NLL_REDUCTION,
            "cqer_delta_nll": baseline,
            "rht_delta_nll": candidate,
            "relative_reduction": reduction,
            "minimum_relative_reduction": MIN_DELTA_NLL_REDUCTION,
        }

    def mean_kl_check() -> dict[str, Any]:
        normalized = aligned or _metric_summary(
            aligned_metrics,
            expected_tokens=ALIGNED_TOKENS,
            context="aligned",
        )
        baseline = float(normalized[CQER_METHOD]["mean_kl"])
        candidate = float(normalized[RHT_METHOD]["mean_kl"])
        return {
            "passed": candidate < baseline,
            "cqer_mean_kl": baseline,
            "rht_mean_kl": candidate,
        }

    def top1_check() -> dict[str, Any]:
        normalized = aligned or _metric_summary(
            aligned_metrics,
            expected_tokens=ALIGNED_TOKENS,
            context="aligned",
        )
        baseline = float(normalized[CQER_METHOD]["top1_agreement"])
        candidate = float(normalized[RHT_METHOD]["top1_agreement"])
        return {
            "passed": candidate >= baseline,
            "cqer_top1_agreement": baseline,
            "rht_top1_agreement": candidate,
        }

    checks = {
        "clean_stable_repository": _gate_check(repository_check),
        "frozen_task_identity_before_model_weights": _gate_check(identity_check),
        "exact_equal_storage": _gate_check(storage_check),
        "finite_metrics_exact_quotas_and_handshakes": _gate_check(
            finite_and_handshake_check
        ),
        "independent_rht_numeric_evidence": _gate_check(unit_check),
        "state_sse_relative_reduction": _gate_check(state_sse_check),
        "aligned_excess_nll_relative_reduction": _gate_check(delta_nll_check),
        "lower_aligned_mean_kl": _gate_check(mean_kl_check),
        "aligned_top1_not_lower": _gate_check(top1_check),
    }
    return {
        "schema": "recurquant.experiment009-stage-a-gate.v1",
        "applicable": True,
        "passed": all(check["passed"] is True for check in checks.values()),
        "checks": checks,
    }


def compute_unit_evidence() -> dict[str, Any]:
    """Recompute deterministic production-path self-consistency evidence on CPU."""

    generator = torch.Generator().manual_seed(SEED)
    state = torch.randn((1, 3, 7, 128), generator=generator, dtype=torch.float32)
    encoded = right_rht_encode(state, layer_index=5, expected_heads=3)
    restored = right_rht_decode(encoded, layer_index=5, expected_heads=3)
    inverse_relative_l2 = float(
        (
            torch.linalg.vector_norm(restored - state)
            / torch.linalg.vector_norm(state).clamp_min(1e-12)
        ).item()
    )

    mask = torch.zeros((3, 7), dtype=torch.bool)
    mask[0, 2] = True
    mask[2, 4:] = True
    low = QuantizationSpec(bits=4, group_size=128, flatten_last_dims=2)
    high = QuantizationSpec(bits=8, group_size=128, flatten_last_dims=2)
    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low,
        high_spec=high,
        right_rht_layer_index=5,
        right_rht_expected_heads=3,
    )
    low_qdq = quantize_dequantize(encoded, low).tensor
    high_qdq = quantize_dequantize(encoded, high).tensor
    transformed_qdq = torch.where(
        mask.reshape(1, 3, 7, 1),
        high_qdq,
        low_qdq,
    )
    expected = right_rht_decode(
        transformed_qdq,
        layer_index=5,
        expected_heads=3,
        output_dtype=state.dtype,
    )
    materialized = packed.dequantize()
    maximum_difference = float((materialized - expected).abs().max().item())

    schedule = torch.cat(
        [
            right_rht_signs(
                layer_index=layer,
                expected_heads=16,
                width=128,
                device="cpu",
            ).to(torch.int8).reshape(-1)
            for layer in FROZEN_LINEAR_LAYERS
        ]
    )
    sign_sha256 = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    return {
        "inverse_relative_l2": inverse_relative_l2,
        "inverse_relative_l2_threshold": MAX_RHT_INVERSE_RELATIVE_L2,
        "physical_pack_matches_transformed_qdq": torch.equal(materialized, expected),
        "physical_pack_max_abs_difference": maximum_difference,
        "physical_pack_storage_bytes": packed.storage_bytes,
        "sign_schedule_sha256": sign_sha256,
        "expected_sign_schedule_sha256": EXPECTED_SIGN_SCHEDULE_SHA256,
        "seed": SEED,
        "device": "cpu",
        "dtype": "torch.float32",
    }


def _append_metrics(
    accumulators: Mapping[str, Any],
    reference_logits: torch.Tensor,
    candidate_outputs: Mapping[str, Any],
    target: torch.Tensor,
) -> None:
    pilot._append_metrics(  # noqa: SLF001 - shared evaluator math is intentional.
        accumulators,
        reference_logits,
        {name: output.logits for name, output in candidate_outputs.items()},
        target,
    )


def evaluate_screen_task(
    model: Qwen3_5ForCausalLM,
    *,
    prompt_ids: torch.Tensor,
    code_ids: torch.Tensor,
    plan: ExactBudgetRowPlan,
) -> dict[str, Any]:
    """Evaluate reference, CQER, and RHT-CQER in one teacher-forced loop."""

    reference_cache = DynamicCache(config=model.config)
    cqer = create_qwen35_query_ema_exact_budget_cache(
        model,
        plan=plan,
        record_evidence=True,
    )
    rht = create_qwen35_right_rht_query_ema_exact_budget_cache(
        model,
        plan=plan,
        record_evidence=True,
    )
    caches: dict[
        str,
        QueryEmaMixedPackedRecurrentStateCache
        | RightRhtQueryEmaMixedPackedRecurrentStateCache,
    ] = {
        CQER_METHOD: cqer,
        RHT_METHOD: rht,
    }
    aligned = {name: pilot._TokenAccumulator.empty() for name in METHODS}  # noqa: SLF001
    full = {name: pilot._TokenAccumulator.empty() for name in METHODS}  # noqa: SLF001

    with Qwen35QueryEnergyObserver(model, caches=list(caches.values())):
        reference_output = model(
            prompt_ids,
            past_key_values=reference_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        candidate_outputs = {
            name: model(
                prompt_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            for name, cache in caches.items()
        }
        _append_metrics(full, reference_output.logits, candidate_outputs, code_ids[:, :1])

        for token_index in range(code_ids.shape[1] - 1):
            input_token = code_ids[:, token_index : token_index + 1]
            target_token = code_ids[:, token_index + 1 : token_index + 2]
            reference_output = model(
                input_token,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            candidate_outputs = {
                name: model(
                    input_token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                for name, cache in caches.items()
            }
            _append_metrics(
                aligned,
                reference_output.logits,
                candidate_outputs,
                target_token,
            )
            _append_metrics(
                full,
                reference_output.logits,
                candidate_outputs,
                target_token,
            )

    reference_bytes = sum(
        state.tensor.numel() * state.tensor.element_size()
        for state in iter_recurrent_states(reference_cache)
    )
    state_errors = {
        name: aggregate_state_error_evidence(cache.update_evidence)
        for name, cache in caches.items()
    }
    return {
        "aligned_metrics": {
            name: accumulator.summary() for name, accumulator in aligned.items()
        },
        "full_code_metrics": {
            name: accumulator.summary() for name, accumulator in full.items()
        },
        "storage": {name: cache.storage_summary() for name, cache in caches.items()},
        "selector_diagnostics": {
            name: cache.query_ema_diagnostics() for name, cache in caches.items()
        },
        "state_errors": state_errors,
        "reference_recurrent_state_bytes": reference_bytes,
    }


def _authenticate_selectors(
    selector_path: Path,
    loss_selector_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], ExactBudgetRowPlan]:
    selector, selector_file_sha = pilot.load_selector_artifact(
        selector_path,
        expected_kind=pilot.HRR_ARTIFACT_KIND,
    )
    loss_selector, loss_file_sha = pilot.load_selector_artifact(
        loss_selector_path,
        expected_kind=pilot.LOSS_ARTIFACT_KIND,
    )
    pilot.validate_compatible_selector(selector, loss_selector)
    pilot.validate_cqer_selector_artifacts(
        enabled=True,
        selectors=(selector, loss_selector),
    )
    plan = pilot.plan_from_artifact(loss_selector, TARGET_FISHER_SCORE)
    quotas = {
        layer: len(plan.groups_for_layer(layer))
        for layer, _heads, _rows in plan.score_shapes
    }
    pilot.validate_cqer_layer_quotas(enabled=True, quotas=quotas)
    if quotas != FROZEN_LAYER_QUOTAS:
        raise ValueError("target-Fisher quota vector drifted from Experiment 009")
    if plan.resident_bytes != TARGET_PACKED_STATE_BYTES:
        raise ValueError("target-Fisher plan does not realize the frozen byte budget")
    return (
        selector,
        loss_selector,
        {
            "selector_file_sha256": selector_file_sha,
            "loss_selector_file_sha256": loss_file_sha,
            "selector_canonical_evidence_sha256": pilot.sha256_bytes(
                canonical_json_bytes(selector)
            ),
            "loss_selector_canonical_evidence_sha256": pilot.sha256_bytes(
                canonical_json_bytes(loss_selector)
            ),
        },
        plan,
    )


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("datasets", "numpy", "safetensors", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _atomic_write(path: Path, payload: bytes) -> None:
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
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    repository_start = pilot.git_state()
    pilot.validate_cora_development_repository_start(repository_start)
    pilot.validate_heldout_output_path(args.output, repository_root)
    source_hashes_start = pilot.source_file_hashes(repository_root, SOURCE_FILES)

    # Selector authentication precedes all dataset access.
    selector, loss_selector, selector_hashes, plan = _authenticate_selectors(
        args.selector,
        args.loss_selector,
    )

    # The task-ID loader retains only task 666. It cannot select the protected
    # ranked [8, 16) window or any other calibration identity.
    rows = load_mbpp_rows_by_task_ids("calibration", task_ids=(TASK_ID,))
    if len(rows) != 1 or int(rows[0]["task_id"]) != TASK_ID:
        raise RuntimeError("Stage-A task-ID loader returned an unauthorized identity")
    row = rows[0]
    row_sha = mbpp_row_sha256(row)
    if row_sha != TASK_ROW_SHA256:
        raise ValueError("Stage-A row hash drifted before tokenization")
    manifest = mbpp_manifest(rows, phase="calibration")
    manifest_sha = mbpp_manifest_sha256(rows, phase="calibration")

    device = pilot.select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_record = selector["model"]
    if model_record["dtype"] != str(dtype):
        raise ValueError(
            "screen dtype does not match the frozen selector dtype: "
            f"{dtype} != {model_record['dtype']}"
        )
    model_id = str(model_record["id"])
    revision = str(model_record["revision"])

    # Tokenizer metadata is authenticated before model weights are opened.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    encoded_tasks, token_manifest = pilot.encode_task_rows(tokenizer, rows)
    if len(encoded_tasks) != 1 or len(token_manifest) != 1:
        raise RuntimeError("Stage-A encoding returned an unexpected task count")
    identity = validate_stage_a_identity(
        task_id=token_manifest[0]["task_id"],
        row_sha256=row_sha,
        prompt_tokens=token_manifest[0]["prompt_tokens"],
        code_tokens=token_manifest[0]["code_tokens"],
        aligned_scored_tokens=token_manifest[0]["aligned_scored_tokens"],
    )

    unit_evidence = compute_unit_evidence()
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    _row, prompt_cpu, code_cpu = encoded_tasks[0]
    torch.manual_seed(SEED)
    with torch.inference_mode():
        result = evaluate_screen_task(
            model,
            prompt_ids=prompt_cpu.to(device),
            code_ids=code_cpu.to(device),
            plan=plan,
        )

    repository_end = pilot.git_state()
    source_hashes_end = pilot.source_file_hashes(repository_root, SOURCE_FILES)
    pilot.validate_cora_development_repository_end(
        start_repository=repository_start,
        end_repository=repository_end,
        start_source_hashes=source_hashes_start,
        end_source_hashes=source_hashes_end,
    )
    integrity = {
        "repository_clean_at_start": repository_start["worktree_clean"] is True,
        "repository_clean_at_end": repository_end["worktree_clean"] is True,
        "repository_commit_stable": (
            repository_start["commit"] == repository_end["commit"]
        ),
        "source_hashes_stable": source_hashes_start == source_hashes_end,
        "identity_authenticated_before_model_weights": (
            identity["authenticated_before_model_weights"] is True
        ),
        "protected_window_8_16_accessed": False,
    }
    gate = evaluate_stage_a_gate(
        aligned_metrics=result["aligned_metrics"],
        full_code_metrics=result["full_code_metrics"],
        storage=result["storage"],
        selector_diagnostics=result["selector_diagnostics"],
        state_errors=result["state_errors"],
        unit_evidence=unit_evidence,
        integrity=integrity,
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "screening_only": True,
        "claim_boundary": (
            "This is a one-task, already-open falsification screen for a known "
            "right-RHT codec composed with CQER-32. Passing authorizes only the "
            "separately frozen 32-task development run. It is not novelty, speed, "
            "confirmation, state-of-the-art, or breakthrough evidence."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "name": "Experiment 009 Stage A",
            "method": RHT_METHOD,
            "task_locked": True,
            "thresholds_locked": True,
            "protected_ranked_window": [8, 16],
            "protected_window_accessed": False,
        },
        "selector_artifacts": {
            **selector_hashes,
            "selector_path": str(args.selector.resolve()),
            "loss_selector_path": str(args.loss_selector.resolve()),
            "frozen_canonical_evidence_sha256s": list(
                pilot.CQER_FROZEN_SELECTOR_CANONICAL_SHA256S
            ),
            "target_fisher_score": TARGET_FISHER_SCORE,
            "target_fisher_layer_quotas": {
                str(layer): quota for layer, quota in FROZEN_LAYER_QUOTAS.items()
            },
            "quota_sum": sum(FROZEN_LAYER_QUOTAS.values()),
            "authenticated": True,
        },
        "model": {
            "id": model_id,
            "revision": revision,
            "dtype": str(dtype),
            "device": str(device),
        },
        "dataset": {
            "phase": "calibration",
            "selection_mode": "exact_already_open_task_id",
            "identity": identity,
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "token_manifest": token_manifest,
            "identity_authenticated_before_model_weights": True,
            "protected_window_8_16_loaded_tokenized_or_evaluated": False,
        },
        "metric_contract": {
            "aligned_primary": "code transitions after recurrent-state storage",
            "aligned_token_count": ALIGNED_TOKENS,
            "full_code_secondary_token_count": CODE_TOKENS,
            "excluded_from_aligned": "prompt-to-first-code-token prediction",
        },
        "methods": list(METHODS),
        "metrics_aligned": result["aligned_metrics"],
        "metrics_full_code_secondary": result["full_code_metrics"],
        "storage": {
            "fp32_reference_recurrent_state_bytes": result[
                "reference_recurrent_state_bytes"
            ],
            "candidates": result["storage"],
        },
        "selector_diagnostics": result["selector_diagnostics"],
        "state_error": result["state_errors"],
        "unit_evidence": unit_evidence,
        "stage_a_gate": gate,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "repository": {
            "commit": repository_end["commit"],
            "start": repository_start,
            "end": repository_end,
            "stable_commit": repository_start["commit"] == repository_end["commit"],
        },
        "source_files": {
            "paths": list(SOURCE_FILES),
            "sha256_start": source_hashes_start,
            "sha256_end": source_hashes_end,
            "stable": source_hashes_start == source_hashes_end,
        },
        "command": [sys.executable, *sys.argv],
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "canonical_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest(),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    _atomic_write(args.output, payload)
    verification = verify_evidence_artifact(args.output)
    if verification["valid"] is not True:
        raise RuntimeError(
            "written Stage-A artifact failed verification: "
            + "; ".join(verification["errors"])
        )

    # Output is constrained to Git-ignored or external storage. Recheck that
    # even artifact publication did not alter the authenticated repository.
    post_write_repository = pilot.git_state()
    post_write_hashes = pilot.source_file_hashes(repository_root, SOURCE_FILES)
    pilot.validate_cora_development_repository_end(
        start_repository=repository_start,
        end_repository=post_write_repository,
        start_source_hashes=source_hashes_start,
        end_source_hashes=post_write_hashes,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "canonical_evidence_sha256": artifact[
                    "canonical_evidence_sha256"
                ],
                "stage_a_gate": gate,
                "post_write_repository_authenticated": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gate["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
