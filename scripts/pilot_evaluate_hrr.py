#!/usr/bin/env python3
"""Evaluate frozen row-selector plans on calibration rows as a diagnostic.

The script deliberately refuses development and confirmation data. Offset zero
checks the selector-task prefix; a positive offset checks a ranked calibration
holdout that must be disjoint from every selector artifact. Neither mode is
confirmation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.evaluation import (
    TokenFidelity,
    fidelity_summary,
    paired_bootstrap_mean_improvement,
    token_fidelity,
)
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.packed_cache import (
    AdaptiveMixedPackedRecurrentStateCache,
    MixedPackedRecurrentStateCache,
    PackedRecurrentStateCache,
    QueryEmaMixedPackedRecurrentStateCache,
    RankFusedMixedPackedRecurrentStateCache,
)
from recurquant.public_data import (
    MBPP_CALIBRATION_SIZE,
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_content_sha256,
    mbpp_manifest_sha256,
)
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.qwen35 import (
    create_qwen35_adaptive_exact_budget_cache,
    create_qwen35_exact_budget_cache,
    create_qwen35_packed_cache,
    create_qwen35_query_ema_exact_budget_cache,
    create_qwen35_rank_fused_exact_budget_cache,
    create_qwen35_v02_mixed_cache,
)
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget

SEED = 2339
RANDOM_ROW_SEED = 1101
HRR_ARTIFACT_KIND = "recurquant_hrr_calibration_diagnostic"
LOSS_ARTIFACT_KIND = "recurquant_loss_sensitivity_calibration_diagnostic"
STORAGE_BOUNDARY_ARTIFACT_KIND = "recurquant_storage_boundary_taylor_diagnostic"
LOSS_SELECTOR_PRIMARY = "signed_taylor_next_int4"
ADAPTIVE_H1 = "adaptive_mse_hrr_h1_quota"
ADAPTIVE_TARGET_FISHER = "adaptive_mse_target_directional_fisher_quota"
RANK_FUSION_METHODS = (
    ("rank_fusion_l025_target_fisher_adaptive_mse", 0.25),
    ("rank_fusion_l050_target_fisher_adaptive_mse", 0.50),
    ("rank_fusion_l075_target_fisher_adaptive_mse", 0.75),
)
RANK_FUSION_PRIMARY = "rank_fusion_l050_target_fisher_adaptive_mse"
QUERY_EMA_PRIMARY = "query_ema32_weighted_mse_target_fisher_quota"
QUERY_EMA_HALF_LIFE = 32
LOSS_SCORE_NAMES = (
    LOSS_SELECTOR_PRIMARY,
    "target_directional_fisher_difference_int4",
    "target_diagonal_fisher_difference_int4",
    "delta_direction_magnitude_int4",
)
QUANTIZER_FIELDS = (
    "bits",
    "group_size",
    "scale_bits",
    "flatten_last_dims",
    "rounding",
    "seed",
    "epsilon",
)
FROZEN_HOLDOUT_OFFSET = 8
FROZEN_HOLDOUT_LIMIT = 8
FROZEN_HRR_HORIZON = 32
FROZEN_BOOTSTRAP_SAMPLES = 10_000
CQER_DEVELOPMENT_LIMIT = 8
CQER_DEVELOPMENT_TASK_IDS = (945, 794, 657, 702, 651, 720, 903, 918)
CQER_FROZEN_SELECTOR_CANONICAL_SHA256S = (
    "7970961fd88b522998189ad64f26b333aed9c88ff5f653de5449fd9e01d8cbc8",
    "bff4e33253990b8115e1f35e74516c4975c2fe4aac5066475afe968eb8a64609",
)
CQER_FROZEN_LAYER_QUOTAS = (
    (0, 355),
    (1, 380),
    (2, 269),
    (4, 179),
    (5, 185),
    (6, 105),
    (8, 80),
    (9, 43),
    (10, 84),
    (12, 30),
    (13, 62),
    (14, 54),
    (16, 45),
    (17, 27),
    (18, 7),
    (20, 9),
    (21, 7),
    (22, 55),
)
TARGET_RESIDENT_BYTES = 2_564_096
MIN_RELATIVE_NLL_REDUCTION = 0.20
CQER_MIN_RELATIVE_NLL_REDUCTION_VS_ADAPTIVE = 0.05
TOP1_DISADVANTAGE_MARGIN = 0.01
CVAR95_DISADVANTAGE_MARGIN = 0.10
MAX_PER_TASK_NLL_DISADVANTAGE = 1.0
FROZEN_STATIC_COMPARATORS = (
    "v02_layer0_static",
    "hrr_h1",
    "hrr_h32",
    "row_mse",
    "random_rows_s1101",
    *LOSS_SCORE_NAMES,
)
EVALUATOR_SOURCE_FILES = (
    "scripts/pilot_evaluate_hrr.py",
    "src/recurquant/cache.py",
    "src/recurquant/evaluation.py",
    "src/recurquant/evidence.py",
    "src/recurquant/metrics.py",
    "src/recurquant/mixed_quantization.py",
    "src/recurquant/packed_cache.py",
    "src/recurquant/public_data.py",
    "src/recurquant/query_energy.py",
    "src/recurquant/quantization.py",
    "src/recurquant/qwen35.py",
    "src/recurquant/row_policy.py",
)

RankFusionCacheSpec = tuple[
    ExactBudgetRowPlan,
    Mapping[int, torch.Tensor],
    float,
]


@dataclass(slots=True)
class _TokenAccumulator:
    kl: list[torch.Tensor]
    reference_nll: list[torch.Tensor]
    candidate_nll: list[torch.Tensor]
    top1_agreement: list[torch.Tensor]
    outputs_finite: list[torch.Tensor]

    @classmethod
    def empty(cls) -> _TokenAccumulator:
        return cls([], [], [], [], [])

    def append(self, values: TokenFidelity, *, outputs_finite: torch.Tensor) -> None:
        cpu = values.to_cpu()
        self.kl.append(cpu.kl.reshape(-1))
        self.reference_nll.append(cpu.reference_nll.reshape(-1))
        self.candidate_nll.append(cpu.candidate_nll.reshape(-1))
        self.top1_agreement.append(cpu.top1_agreement.reshape(-1))
        self.outputs_finite.append(outputs_finite.detach().to("cpu").reshape(()))

    def summary(self) -> dict[str, float | int]:
        summary = fidelity_summary(
            TokenFidelity(
                kl=torch.cat(self.kl),
                reference_nll=torch.cat(self.reference_nll),
                candidate_nll=torch.cat(self.candidate_nll),
                top1_agreement=torch.cat(self.top1_agreement),
            )
        )
        summary["all_logits_finite"] = bool(torch.stack(self.outputs_finite).all().item())
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selector-prefix or heldout-calibration quality diagnostic; "
            "never confirmation evidence."
        )
    )
    parser.add_argument("--selector-artifact", type=Path, required=True)
    parser.add_argument("--loss-selector-artifact", type=Path)
    parser.add_argument("--storage-boundary-artifact", type=Path)
    parser.add_argument(
        "--rank-fusion",
        action="store_true",
        help=(
            "Run the frozen Experiment 006 same-calibration rank-fusion primary "
            "and its predeclared 0.25/0.75 ablations. Positive offsets remain "
            "disabled until the separate Experiment 006 prerequisite is implemented."
        ),
    )
    parser.add_argument(
        "--query-ema",
        action="store_true",
        help=(
            "Run the frozen Experiment 007 CQER-32 same-calibration primary. "
            "Positive offsets remain disabled until the candidate-aligned "
            "Experiment 007 prerequisite is implemented and passes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--calibration-offset",
        type=int,
        default=0,
        help=(
            "Start index in the frozen ranked calibration population. Positive offsets "
            "must select rows disjoint from every selector artifact."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_selector_artifact(
    path: Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    verification = verify_evidence_artifact(path)
    if not verification["valid"]:
        raise ValueError(
            "selector artifact failed evidence verification: " + "; ".join(verification["errors"])
        )
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selector artifact must be strict UTF-8 JSON") from error
    if not isinstance(artifact, dict):
        raise ValueError("selector artifact root must be an object")
    if artifact.get("artifact_kind") != expected_kind:
        raise ValueError("unexpected selector artifact kind")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("diagnostic_only") is not True:
        raise ValueError("selector artifact must be a calibration diagnostic")
    expected = artifact.get("canonical_evidence_sha256")
    actual = sha256_bytes(canonical_json_bytes(evidence))
    if expected != actual:
        raise ValueError("selector canonical evidence hash does not match")
    return evidence, sha256_bytes(payload)


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _selector_token_contract(evidence: Mapping[str, Any]) -> tuple[dict[str, int], ...]:
    """Normalize and validate the selector's ordered token manifest."""

    dataset = _require_mapping(evidence.get("dataset"), context="selector dataset")
    tasks = dataset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("selector dataset tasks must be a non-empty array")
    kind = evidence.get("artifact_kind")
    if kind == HRR_ARTIFACT_KIND:
        affected_field = "captured_decode_tokens"
    elif kind == LOSS_ARTIFACT_KIND:
        affected_field = "scored_transitions"
    else:
        raise ValueError("unexpected selector artifact kind")

    normalized: list[dict[str, int]] = []
    seen: set[int] = set()
    for index, raw_record in enumerate(tasks):
        record = _require_mapping(raw_record, context=f"selector task {index}")
        try:
            task_id = int(record["task_id"])
            prompt_tokens = int(record["prompt_tokens"])
            code_tokens = int(record["code_tokens"])
            affected_tokens = int(record[affected_field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"selector task {index} has invalid token metadata") from error
        if task_id in seen:
            raise ValueError(f"selector token manifest repeats task_id {task_id}")
        if prompt_tokens < 1 or code_tokens < 2:
            raise ValueError(f"selector task {task_id} has invalid token counts")
        if affected_tokens != code_tokens - 1:
            raise ValueError(
                f"selector task {task_id} affected-token count must equal code_tokens - 1"
            )
        seen.add(task_id)
        normalized.append(
            {
                "task_id": task_id,
                "prompt_tokens": prompt_tokens,
                "code_tokens": code_tokens,
                "affected_tokens": affected_tokens,
            }
        )
    return tuple(normalized)


def _selector_quantizer_contract(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict, explicit row-quantizer contract from a selector."""

    raw_contract = _require_mapping(
        evidence.get("quantizers"),
        context="selector quantizers",
    )
    axis_contract = raw_contract.get("axis_contract")
    if axis_contract != "one independent group per recurrent [head, key-row]":
        raise ValueError("selector quantizer axis contract is missing or unsupported")

    normalized: dict[str, Any] = {"axis_contract": axis_contract}
    for name, bits in (("int4", 4), ("int8", 8)):
        raw_spec = _require_mapping(
            raw_contract.get(name),
            context=f"selector {name} quantizer",
        )
        missing = [field for field in QUANTIZER_FIELDS if field not in raw_spec]
        if missing:
            raise ValueError(f"selector {name} quantizer lacks fields: {', '.join(missing)}")
        spec = {field: raw_spec[field] for field in QUANTIZER_FIELDS}
        if spec["bits"] != bits:
            raise ValueError(f"selector {name} quantizer must use bits={bits}")
        if spec["rounding"] not in ("nearest", "stochastic"):
            raise ValueError(f"selector {name} quantizer has unsupported rounding")
        if int(spec["group_size"]) <= 0 or int(spec["flatten_last_dims"]) <= 0:
            raise ValueError(f"selector {name} quantizer has invalid grouping")
        if int(spec["scale_bits"]) not in (16, 32):
            raise ValueError(f"selector {name} quantizer has invalid scale_bits")
        if float(spec["epsilon"]) <= 0:
            raise ValueError(f"selector {name} quantizer has invalid epsilon")
        normalized[name] = spec

    matched_fields = tuple(field for field in QUANTIZER_FIELDS if field != "bits")
    mismatches = [
        field for field in matched_fields if normalized["int4"][field] != normalized["int8"][field]
    ]
    if mismatches:
        raise ValueError("selector INT4/INT8 quantizers disagree on: " + ", ".join(mismatches))
    return normalized


def validate_selector_contract(evidence: dict[str, Any]) -> None:
    """Validate one selector's self-contained provenance and quantizer contract."""

    model = _require_mapping(evidence.get("model"), context="selector model")
    for field in ("id", "revision", "dtype"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"selector model {field} must be a non-empty string")
    layers = model.get("linear_attention_layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in layers
        )
    ):
        raise ValueError("selector model linear_attention_layers must be non-negative integers")
    if len(set(layers)) != len(layers):
        raise ValueError("selector model linear_attention_layers must be unique")

    dataset = _require_mapping(evidence.get("dataset"), context="selector dataset")
    manifest = _require_mapping(dataset.get("manifest"), context="selector dataset manifest")
    recorded_manifest_hash = dataset.get("manifest_sha256")
    actual_manifest_hash = mbpp_manifest_content_sha256(manifest)
    if recorded_manifest_hash != actual_manifest_hash:
        raise ValueError("selector dataset manifest hash does not match its content")
    if manifest.get("formatter_version") != "recurquant.mbpp-prompt-code.v1":
        raise ValueError("selector formatter version is unsupported")
    token_contract = _selector_token_contract(evidence)
    manifest_ids = {int(record["task_id"]) for record in manifest.get("rows", [])}
    if manifest_ids != {record["task_id"] for record in token_contract}:
        raise ValueError("selector token manifest and content manifest task IDs do not match")

    try:
        seed = int(evidence["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selector seed must be an integer") from error
    quantizers = _selector_quantizer_contract(evidence)
    budget = _require_mapping(evidence.get("byte_budget"), context="selector byte budget")
    for name in ("int4", "int8"):
        spec = quantizers[name]
        if int(spec["seed"]) != seed:
            raise ValueError(f"selector {name} seed does not match the artifact seed")
        if int(spec["group_size"]) != int(budget.get("group_size", -1)):
            raise ValueError(f"selector {name} group_size does not match the byte budget")
        if int(spec["scale_bits"]) != int(budget.get("scale_bits", -1)):
            raise ValueError(f"selector {name} scale_bits does not match the byte budget")

    if evidence.get("artifact_kind") == HRR_ARTIFACT_KIND:
        method = _require_mapping(evidence.get("method"), context="selector method")
        try:
            normalization_epsilon = float(method["normalization_epsilon"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("HRR selector lacks a valid normalization_epsilon") from error
        if normalization_epsilon <= 0:
            raise ValueError("HRR selector normalization_epsilon must be positive")


def validate_compatible_selector(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Require two calibration selectors to describe the same frozen inputs."""

    validate_selector_contract(reference)
    validate_selector_contract(candidate)
    for field in ("id", "revision", "dtype", "linear_attention_layers"):
        if candidate["model"][field] != reference["model"][field]:
            raise ValueError(f"selector model {field} values do not match")
    if candidate["dataset"]["manifest_sha256"] != reference["dataset"]["manifest_sha256"]:
        raise ValueError("selector calibration manifest hashes do not match")
    if candidate["dataset"]["manifest"] != reference["dataset"]["manifest"]:
        raise ValueError("selector calibration content manifests do not match")
    if _selector_token_contract(candidate) != _selector_token_contract(reference):
        raise ValueError("selector calibration token manifests do not match")
    if int(candidate["seed"]) != int(reference["seed"]):
        raise ValueError("selector seeds do not match")
    if _selector_quantizer_contract(candidate) != _selector_quantizer_contract(reference):
        raise ValueError("selector quantizer contracts do not match")
    budget_fields = (
        "target_resident_bytes",
        "group_size",
        "scale_bits",
        "precision_mask_bits_per_group",
    )
    for field in budget_fields:
        if int(candidate["byte_budget"][field]) != int(reference["byte_budget"][field]):
            raise ValueError(f"selector byte-budget {field} values do not match")


def validate_calibration_prefix(
    selector: Mapping[str, Any],
    *,
    actual_manifest: Mapping[str, Any],
    expected_task_ids: Sequence[int],
) -> None:
    """Verify a selected prefix against authenticated per-row content hashes."""

    dataset = _require_mapping(selector.get("dataset"), context="selector dataset")
    full_manifest = _require_mapping(
        dataset.get("manifest"),
        context="selector dataset manifest",
    )
    expected_id_set = set(expected_task_ids)
    if len(expected_id_set) != len(expected_task_ids):
        raise ValueError("selector prefix task IDs must be unique")
    full_rows = full_manifest.get("rows")
    if not isinstance(full_rows, list):
        raise ValueError("selector dataset manifest rows must be an array")
    selected_rows = [record for record in full_rows if int(record["task_id"]) in expected_id_set]
    if len(selected_rows) != len(expected_task_ids):
        raise ValueError("selector prefix is not fully represented in the content manifest")
    expected_manifest = dict(full_manifest)
    expected_manifest["row_count"] = len(selected_rows)
    expected_manifest["rows"] = selected_rows
    if dict(actual_manifest) != expected_manifest:
        raise ValueError("calibration prefix content manifest does not match the selector artifact")


def selector_task_ids(selectors: Sequence[Mapping[str, Any]]) -> frozenset[int]:
    """Return the union of authenticated task IDs used by selector artifacts."""

    if not selectors:
        raise ValueError("at least one selector artifact is required")
    return frozenset(
        record["task_id"] for selector in selectors for record in _selector_token_contract(selector)
    )


def select_calibration_window(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    offset: int,
    limit: int,
    selectors: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Slice ranked calibration rows and enforce selector disjointness for holdouts."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("calibration offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("calibration limit must be a positive integer")
    stop = offset + limit
    if stop > len(ranked_rows):
        raise ValueError(
            f"calibration window [{offset}:{stop}] exceeds {len(ranked_rows)} ranked rows"
        )
    selected = tuple(ranked_rows[offset:stop])
    if len(selected) != limit:
        raise RuntimeError("calibration window slicing returned an unexpected row count")
    if offset > 0:
        evaluation_ids = {int(row["task_id"]) for row in selected}
        overlap = evaluation_ids & selector_task_ids(selectors)
        if overlap:
            rendered = ", ".join(str(task_id) for task_id in sorted(overlap))
            raise ValueError(
                "heldout-calibration rows overlap selector artifact tasks: " + rendered
            )
    return selected


def validate_frozen_holdout_request(
    *,
    offset: int,
    limit: int,
    bootstrap_samples: int,
    selectors: Sequence[Mapping[str, Any]],
    loss_selector_present: bool,
    storage_boundary_present: bool,
    rank_fusion_enabled: bool = False,
    query_ema_enabled: bool = False,
) -> None:
    """Refuse any positive-offset run outside Experiment 005's frozen request."""

    if offset == 0:
        return
    if rank_fusion_enabled:
        raise ValueError(
            "Experiment 006 rank-fusion holdout remains closed until its separate "
            "candidate-aligned numeric prerequisite and frozen gate are implemented"
        )
    if query_ema_enabled:
        raise ValueError(
            "Experiment 007 CQER-32 holdout remains closed until its development "
            "rule and candidate-aligned numeric prerequisite pass"
        )
    if offset != FROZEN_HOLDOUT_OFFSET or limit != FROZEN_HOLDOUT_LIMIT:
        raise ValueError(
            "Experiment 005 heldout-calibration requires the exact ranked window "
            f"[{FROZEN_HOLDOUT_OFFSET}, "
            f"{FROZEN_HOLDOUT_OFFSET + FROZEN_HOLDOUT_LIMIT})"
        )
    if bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "Experiment 005 heldout-calibration requires exactly "
            f"{FROZEN_BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    if not loss_selector_present or len(selectors) != 2:
        raise ValueError("Experiment 005 heldout-calibration requires both HRR and loss selectors")
    if not storage_boundary_present:
        raise ValueError(
            "Experiment 005 heldout-calibration requires a passing storage-boundary artifact"
        )
    kinds = [selector.get("artifact_kind") for selector in selectors]
    if kinds != [HRR_ARTIFACT_KIND, LOSS_ARTIFACT_KIND]:
        raise ValueError("Experiment 005 requires HRR then loss selector artifacts")
    try:
        horizon = int(selectors[0]["method"]["horizon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Experiment 005 HRR selector lacks a valid horizon") from error
    if horizon != FROZEN_HRR_HORIZON:
        raise ValueError(
            f"Experiment 005 heldout-calibration requires HRR horizon {FROZEN_HRR_HORIZON}"
        )
    for selector in selectors:
        task_count = len(_selector_token_contract(selector))
        if task_count != FROZEN_HOLDOUT_LIMIT:
            raise ValueError(
                "each Experiment 005 selector must contain exactly "
                f"{FROZEN_HOLDOUT_LIMIT} tasks; found {task_count}"
            )


def validate_cqer_development_request(
    *,
    enabled: bool,
    offset: int,
    limit: int,
    bootstrap_samples: int,
) -> None:
    """Freeze Experiment 007 to the complete inspected selector partition."""

    if not enabled:
        return
    if (
        offset != 0
        or limit != CQER_DEVELOPMENT_LIMIT
        or bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES
    ):
        raise ValueError(
            "Experiment 007 CQER-32 development requires offset 0, exactly 8 tasks, "
            "and exactly 10000 bootstrap samples"
        )


def validate_cqer_selector_artifacts(
    *,
    enabled: bool,
    selectors: Sequence[Mapping[str, Any]],
) -> None:
    """Require CQER selectors to be the exact preregistered evidence pair."""

    if not enabled:
        return
    if len(selectors) != 2:
        raise ValueError(
            "Experiment 007 CQER-32 requires exactly two authenticated "
            "eight-task selector artifacts"
        )
    kinds = tuple(selector.get("artifact_kind") for selector in selectors)
    if kinds != (HRR_ARTIFACT_KIND, LOSS_ARTIFACT_KIND):
        raise ValueError("Experiment 007 CQER-32 requires HRR then loss selectors")
    token_contracts = tuple(_selector_token_contract(selector) for selector in selectors)
    if any(len(contract) != CQER_DEVELOPMENT_LIMIT for contract in token_contracts):
        raise ValueError(
            "Experiment 007 CQER-32 requires exactly two authenticated "
            "eight-task selector artifacts"
        )
    task_id_orders = tuple(
        tuple(int(record["task_id"]) for record in contract)
        for contract in token_contracts
    )
    if any(task_ids != CQER_DEVELOPMENT_TASK_IDS for task_ids in task_id_orders):
        raise ValueError(
            "Experiment 007 CQER-32 selector task IDs do not match the frozen order"
        )
    canonical_hashes = tuple(
        sha256_bytes(canonical_json_bytes(selector)) for selector in selectors
    )
    if canonical_hashes != CQER_FROZEN_SELECTOR_CANONICAL_SHA256S:
        raise ValueError(
            "Experiment 007 CQER-32 selector canonical hashes do not match the "
            "frozen artifacts"
        )


def validate_cqer_development_task_ids(
    *,
    enabled: bool,
    task_ids: Sequence[int],
) -> None:
    """Pin the exact ordered development tasks before model execution."""

    if enabled and tuple(int(task_id) for task_id in task_ids) != CQER_DEVELOPMENT_TASK_IDS:
        raise ValueError(
            "Experiment 007 CQER-32 task IDs do not match the frozen ordered prefix"
        )


def validate_cqer_layer_quotas(
    *,
    enabled: bool,
    quotas: Mapping[int, int],
) -> None:
    """Pin the preregistered target-Fisher layer allocation."""

    if not enabled:
        return
    normalized = tuple(sorted((int(layer), int(quota)) for layer, quota in quotas.items()))
    expected = tuple(sorted(CQER_FROZEN_LAYER_QUOTAS))
    if normalized != expected or sum(quota for _, quota in normalized) != 1_976:
        raise ValueError(
            "Experiment 007 CQER-32 layer quotas do not match the frozen "
            "target-Fisher allocation"
        )


def authenticate_selector_prefix(
    selector: Mapping[str, Any],
    *,
    ranked_prefix_manifest: Mapping[str, Any],
    ranked_prefix_task_ids: Sequence[int],
) -> None:
    """Authenticate a selector against the pinned ranked MBPP selector prefix."""

    token_contract = _selector_token_contract(selector)
    if len(token_contract) != FROZEN_HOLDOUT_LIMIT:
        raise ValueError(
            f"selector must contain exactly {FROZEN_HOLDOUT_LIMIT} frozen prefix tasks"
        )
    recorded_ids = [record["task_id"] for record in token_contract]
    if recorded_ids != list(ranked_prefix_task_ids):
        raise ValueError("selector ordered task IDs do not match the pinned ranked prefix")
    validate_calibration_prefix(
        selector,
        actual_manifest=ranked_prefix_manifest,
        expected_task_ids=ranked_prefix_task_ids,
    )


def validate_actual_token_manifest(
    selector: Mapping[str, Any],
    actual_records: Sequence[Mapping[str, int]],
) -> None:
    """Verify tokenizer output against the selector's ordered token manifest."""

    expected = _selector_token_contract(selector)
    normalized_actual = tuple(
        {
            "task_id": int(record["task_id"]),
            "prompt_tokens": int(record["prompt_tokens"]),
            "code_tokens": int(record["code_tokens"]),
            "affected_tokens": int(record["aligned_scored_tokens"]),
        }
        for record in actual_records
    )
    if normalized_actual != expected[: len(normalized_actual)]:
        raise ValueError("actual formatter/token manifest does not match the selector prefix")


def encode_task_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]],
    list[dict[str, int]],
]:
    """Tokenize rows and build the exact aligned/full-code token manifest."""

    encoded: list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]] = []
    manifest: list[dict[str, int]] = []
    for row in rows:
        formatted = format_mbpp_example(row)
        prompt_ids = tokenizer(
            formatted.prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        code_ids = tokenizer(
            formatted.code,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        code_tokens = int(code_ids.shape[1])
        if code_tokens < 2:
            raise RuntimeError(f"MBPP task {row['task_id']} has fewer than two code tokens")
        manifest.append(
            {
                "task_id": int(row["task_id"]),
                "prompt_tokens": int(prompt_ids.shape[1]),
                "code_tokens": code_tokens,
                "aligned_scored_tokens": code_tokens - 1,
                "full_code_scored_tokens": code_tokens,
            }
        )
        encoded.append((row, prompt_ids, code_ids))
    return encoded, manifest


def paired_contrast(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return bootstrap uncertainty only when at least two pairs exist."""

    if len(baseline_values) != len(candidate_values):
        raise ValueError("paired contrast values must have equal length")
    if not baseline_values:
        raise ValueError("paired contrast requires at least one example")
    if len(baseline_values) == 1:
        return {
            "paired_examples": 1,
            "mean_improvement": float(baseline_values[0] - candidate_values[0]),
            "confidence": None,
            "confidence_interval": None,
            "bootstrap_samples": 0,
            "seed": None,
            "note": "descriptive only; uncertainty is not estimable from one paired task",
        }
    return paired_bootstrap_mean_improvement(
        baseline_values,
        candidate_values,
        samples=samples,
        seed=seed,
    )


def aggregate_task_rows(
    per_task: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    """Compute task-macro summaries while retaining the exact token count."""

    return {
        name: {
            "task_count": len(task_rows),
            "macro_delta_nll": fmean(float(row["delta_nll"]) for row in task_rows),
            "macro_mean_kl": fmean(float(row["mean_kl"]) for row in task_rows),
            "macro_cvar95_kl": fmean(float(row["cvar95_kl"]) for row in task_rows),
            "macro_top1_agreement": fmean(float(row["top1_agreement"]) for row in task_rows),
            "token_count": sum(int(row["token_count"]) for row in task_rows),
        }
        for name, task_rows in per_task.items()
    }


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{context} must be finite")
    return rendered


def _paired_lower_bound(
    contrasts: Mapping[str, Mapping[str, Any]],
    name: str,
) -> float | None:
    record = contrasts.get(name)
    if not isinstance(record, Mapping):
        return None
    interval = record.get("confidence_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        return None
    try:
        lower = _finite_float(interval[0], context=f"{name} paired lower bound")
    except ValueError:
        return None
    return lower


def evaluate_frozen_holdout_gate(
    *,
    aggregates: Mapping[str, Mapping[str, float | int]],
    per_task: Mapping[str, Sequence[Mapping[str, float | int]]],
    contrasts: Mapping[str, Mapping[str, Any]],
    storage: Mapping[str, Mapping[str, int | float | bool]],
    primary_name: str,
) -> dict[str, Any]:
    """Evaluate the machine-checkable Experiment 005 heldout-calibration gate."""

    if primary_name != ADAPTIVE_TARGET_FISHER:
        raise ValueError(
            f"frozen heldout primary must be {ADAPTIVE_TARGET_FISHER}, got {primary_name}"
        )
    required = (primary_name, ADAPTIVE_H1)
    for name in required:
        if name not in aggregates or name not in per_task or name not in storage:
            raise ValueError(f"heldout gate lacks required method {name}")

    expected_methods = {
        "uniform_int4",
        *FROZEN_STATIC_COMPARATORS,
        ADAPTIVE_H1,
        ADAPTIVE_TARGET_FISHER,
    }
    if set(aggregates) != expected_methods:
        raise ValueError("heldout methods do not match the frozen Experiment 005 method set")
    all_static_names = list(FROZEN_STATIC_COMPARATORS)
    static_names = [
        name
        for name in all_static_names
        if name in storage and int(storage[name].get("resident_bytes", -1)) == TARGET_RESIDENT_BYTES
    ]
    if not static_names:
        raise ValueError("heldout gate requires at least one equal-byte static comparator")
    strongest_static = min(
        static_names,
        key=lambda name: _finite_float(
            aggregates[name]["macro_delta_nll"],
            context=f"{name} macro_delta_nll",
        ),
    )

    primary = aggregates[primary_name]
    static = aggregates[strongest_static]
    adaptive_h1 = aggregates[ADAPTIVE_H1]
    primary_nll = _finite_float(primary["macro_delta_nll"], context="primary macro_delta_nll")
    static_nll = _finite_float(
        static["macro_delta_nll"],
        context="strongest static macro_delta_nll",
    )
    adaptive_h1_nll = _finite_float(
        adaptive_h1["macro_delta_nll"],
        context="adaptive H1 macro_delta_nll",
    )
    static_nlls = {
        name: _finite_float(
            aggregates[name]["macro_delta_nll"],
            context=f"{name} macro_delta_nll",
        )
        for name in static_names
    }
    relative_reduction = (static_nll - primary_nll) / static_nll if static_nll > 0 else None

    primary_top1 = _finite_float(
        primary["macro_top1_agreement"],
        context="primary macro_top1_agreement",
    )
    static_top1 = _finite_float(
        static["macro_top1_agreement"],
        context="strongest static macro_top1_agreement",
    )
    primary_cvar = _finite_float(
        primary["macro_cvar95_kl"],
        context="primary macro_cvar95_kl",
    )
    static_cvar = _finite_float(
        static["macro_cvar95_kl"],
        context="strongest static macro_cvar95_kl",
    )

    primary_tasks = {
        int(row["task_id"]): _finite_float(
            row["delta_nll"],
            context=f"primary task {row['task_id']} delta_nll",
        )
        for row in per_task[primary_name]
    }
    static_tasks = {
        int(row["task_id"]): _finite_float(
            row["delta_nll"],
            context=f"static task {row['task_id']} delta_nll",
        )
        for row in per_task[strongest_static]
    }
    if not primary_tasks or primary_tasks.keys() != static_tasks.keys():
        raise ValueError("primary and strongest static per-task IDs must match")
    per_task_disadvantages = {
        str(task_id): primary_tasks[task_id] - static_tasks[task_id]
        for task_id in sorted(primary_tasks)
    }
    worst_task_id = max(per_task_disadvantages, key=per_task_disadvantages.__getitem__)
    worst_task_disadvantage = per_task_disadvantages[worst_task_id]

    exact_storage_observed = {
        name: int(summary.get("resident_bytes", -1))
        for name, summary in storage.items()
        if name != "uniform_int4"
    }
    exact_storage_passed = bool(exact_storage_observed) and all(
        value == TARGET_RESIDENT_BYTES for value in exact_storage_observed.values()
    )
    static_lower_bound = _paired_lower_bound(contrasts, strongest_static)
    adaptive_h1_lower_bound = _paired_lower_bound(contrasts, ADAPTIVE_H1)

    checks: dict[str, dict[str, Any]] = {
        "exact_resident_bytes": {
            "passed": exact_storage_passed,
            "required_bytes": TARGET_RESIDENT_BYTES,
            "observed": exact_storage_observed,
        },
        "lower_nll_than_every_equal_byte_static": {
            "passed": all(primary_nll < value for value in static_nlls.values()),
            "primary": primary_nll,
            "static_comparators": static_nlls,
        },
        "relative_nll_reduction_vs_strongest_static": {
            "passed": (
                relative_reduction is not None and relative_reduction >= MIN_RELATIVE_NLL_REDUCTION
            ),
            "observed": relative_reduction,
            "minimum": MIN_RELATIVE_NLL_REDUCTION,
            "denominator_contract": "strongest static excess NLL must be positive",
        },
        "lower_nll_than_adaptive_h1": {
            "passed": primary_nll < adaptive_h1_nll,
            "primary": primary_nll,
            "adaptive_h1": adaptive_h1_nll,
        },
        "paired_lower_ci_vs_strongest_static": {
            "passed": static_lower_bound is not None and static_lower_bound > 0,
            "observed_lower_bound": static_lower_bound,
            "required": "strictly greater than zero",
        },
        "paired_lower_ci_vs_adaptive_h1": {
            "passed": adaptive_h1_lower_bound is not None and adaptive_h1_lower_bound > 0,
            "observed_lower_bound": adaptive_h1_lower_bound,
            "required": "strictly greater than zero",
        },
        "top1_disadvantage_margin_vs_strongest_static": {
            "passed": primary_top1 >= static_top1 - TOP1_DISADVANTAGE_MARGIN,
            "observed_disadvantage": static_top1 - primary_top1,
            "maximum": TOP1_DISADVANTAGE_MARGIN,
        },
        "cvar95_disadvantage_margin_vs_strongest_static": {
            "passed": primary_cvar <= static_cvar + CVAR95_DISADVANTAGE_MARGIN,
            "observed_disadvantage": primary_cvar - static_cvar,
            "maximum": CVAR95_DISADVANTAGE_MARGIN,
        },
        "maximum_per_task_nll_disadvantage_vs_strongest_static": {
            "passed": worst_task_disadvantage <= MAX_PER_TASK_NLL_DISADVANTAGE,
            "observed": worst_task_disadvantage,
            "maximum": MAX_PER_TASK_NLL_DISADVANTAGE,
            "worst_task_id": int(worst_task_id),
            "per_task": per_task_disadvantages,
        },
    }
    return {
        "schema": "recurquant.experiment005-heldout-gate.v1",
        "applicable": True,
        "passed": all(check["passed"] is True for check in checks.values()),
        "primary": primary_name,
        "strongest_equal_byte_static": strongest_static,
        "adaptive_control": ADAPTIVE_H1,
        "thresholds": {
            "minimum_relative_nll_reduction": MIN_RELATIVE_NLL_REDUCTION,
            "maximum_top1_disadvantage": TOP1_DISADVANTAGE_MARGIN,
            "maximum_cvar95_kl_disadvantage": CVAR95_DISADVANTAGE_MARGIN,
            "maximum_per_task_nll_disadvantage": MAX_PER_TASK_NLL_DISADVANTAGE,
            "paired_lower_ci_must_be_strictly_positive": True,
        },
        "checks": checks,
    }


def evaluate_cqer_development_gate(
    *,
    aggregates: Mapping[str, Mapping[str, float | int]],
    per_task: Mapping[str, Sequence[Mapping[str, float | int]]],
    per_task_full_code: Mapping[str, Sequence[Mapping[str, float | int]]],
    storage: Mapping[str, Mapping[str, int | float | bool]],
    query_ema_diagnostics: Mapping[str, Sequence[Mapping[str, object]]],
    expected_quotas: Mapping[int, int],
    expected_packed_bytes: int,
    expected_selector_auxiliary_bytes: int,
) -> dict[str, Any]:
    """Evaluate Experiment 007's frozen same-calibration advancement filter."""

    required = (
        QUERY_EMA_PRIMARY,
        ADAPTIVE_TARGET_FISHER,
        "target_directional_fisher_difference_int4",
    )
    for name in required:
        if (
            name not in aggregates
            or name not in per_task
            or name not in per_task_full_code
            or name not in storage
        ):
            raise ValueError(f"CQER development gate lacks required method {name}")
    diagnostic_tasks = query_ema_diagnostics.get(QUERY_EMA_PRIMARY)
    if (
        not isinstance(diagnostic_tasks, Sequence)
        or len(diagnostic_tasks) != CQER_DEVELOPMENT_LIMIT
    ):
        raise ValueError(
            "CQER development gate requires query-EMA diagnostics for exactly "
            f"{CQER_DEVELOPMENT_LIMIT} tasks"
        )
    for name in required:
        if (
            len(per_task[name]) != CQER_DEVELOPMENT_LIMIT
            or len(per_task_full_code[name]) != CQER_DEVELOPMENT_LIMIT
        ):
            raise ValueError(
                "CQER development gate requires aligned and full-code metrics for "
                f"exactly {CQER_DEVELOPMENT_LIMIT} tasks per required method"
            )

    primary = aggregates[QUERY_EMA_PRIMARY]
    adaptive = aggregates[ADAPTIVE_TARGET_FISHER]
    static_name = "target_directional_fisher_difference_int4"
    static = aggregates[static_name]
    primary_nll = _finite_float(primary["macro_delta_nll"], context="CQER macro_delta_nll")
    adaptive_nll = _finite_float(
        adaptive["macro_delta_nll"],
        context="adaptive target-Fisher macro_delta_nll",
    )
    static_nll = _finite_float(
        static["macro_delta_nll"],
        context="static target-Fisher macro_delta_nll",
    )
    relative_vs_adaptive = (
        (adaptive_nll - primary_nll) / adaptive_nll if adaptive_nll > 0 else None
    )
    relative_vs_static = (
        (static_nll - primary_nll) / static_nll if static_nll > 0 else None
    )

    primary_top1 = _finite_float(
        primary["macro_top1_agreement"],
        context="CQER macro_top1_agreement",
    )
    better_comparator_top1 = max(
        _finite_float(
            adaptive["macro_top1_agreement"],
            context="adaptive target-Fisher macro_top1_agreement",
        ),
        _finite_float(
            static["macro_top1_agreement"],
            context="static target-Fisher macro_top1_agreement",
        ),
    )
    primary_cvar = _finite_float(
        primary["macro_cvar95_kl"],
        context="CQER macro_cvar95_kl",
    )
    lower_comparator_cvar = min(
        _finite_float(
            adaptive["macro_cvar95_kl"],
            context="adaptive target-Fisher macro_cvar95_kl",
        ),
        _finite_float(
            static["macro_cvar95_kl"],
            context="static target-Fisher macro_cvar95_kl",
        ),
    )

    expected_layers = dict(sorted(expected_quotas.items()))
    layer_audit: list[dict[str, object]] = []
    selector_contract_passed = True
    finite_diagnostics = True
    for task_record in diagnostic_tasks:
        task_id = int(task_record.get("task_id", -1))
        raw_layers = task_record.get("layers")
        if not isinstance(raw_layers, list):
            selector_contract_passed = False
            layer_audit.append({"task_id": task_id, "error": "layers must be an array"})
            continue
        observed_layers: dict[int, Mapping[str, object]] = {}
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, Mapping):
                selector_contract_passed = False
                continue
            layer_index = int(raw_layer.get("layer_index", -1))
            if layer_index in observed_layers:
                selector_contract_passed = False
                continue
            observed_layers[layer_index] = raw_layer
        task_passed = set(observed_layers) == set(expected_layers)
        observations = 0
        state_updates = 0
        for layer_index, quota in expected_layers.items():
            record = observed_layers.get(layer_index)
            if record is None:
                task_passed = False
                continue
            observed_quota = int(record.get("quota", -1))
            selected = int(record.get("current_selected_count", -1))
            committed = int(record.get("observations_committed", -1))
            updates = int(record.get("state_updates", -2))
            auxiliary = int(record.get("selector_auxiliary_bytes", -1))
            mask_sha256 = record.get("current_mask_sha256")
            pending = record.get("pending_observation")
            cutoff = record.get("last_cutoff_score_margin")
            if cutoff is not None:
                try:
                    _finite_float(cutoff, context="CQER cutoff score margin")
                except ValueError:
                    finite_diagnostics = False
            observations += max(committed, 0)
            state_updates += max(updates, 0)
            task_passed = task_passed and all(
                (
                    observed_quota == quota,
                    selected == quota,
                    committed == updates,
                    committed > 0,
                    auxiliary > 0,
                    isinstance(mask_sha256, str) and len(mask_sha256) == 64,
                    pending is False,
                )
            )
        selector_contract_passed = selector_contract_passed and task_passed
        layer_audit.append(
            {
                "task_id": task_id,
                "passed": task_passed,
                "observations_committed": observations,
                "state_updates": state_updates,
            }
        )

    summary = storage[QUERY_EMA_PRIMARY]
    observed_packed_bytes = int(summary.get("resident_bytes", -1))
    observed_auxiliary_bytes = int(summary.get("selector_auxiliary_bytes", -1))
    observed_total_bytes = int(summary.get("resident_bytes_including_selector", -1))
    observed_promotions = int(summary.get("high_precision_groups", -1))
    expected_promotions = sum(expected_layers.values())

    all_metrics_finite = finite_diagnostics
    for metric_partition in (per_task, per_task_full_code):
        for method_rows in metric_partition.values():
            for row in method_rows:
                logits_finite = row.get("all_logits_finite")
                if logits_finite is not True:
                    all_metrics_finite = False
                for key, value in row.items():
                    if key in ("task_id", "all_logits_finite"):
                        continue
                    try:
                        _finite_float(value, context=f"per-task metric {key}")
                    except ValueError:
                        all_metrics_finite = False

    checks: dict[str, dict[str, Any]] = {
        "exact_per_layer_quotas": {
            "passed": selector_contract_passed and observed_promotions == expected_promotions,
            "expected_quotas": {str(key): value for key, value in expected_layers.items()},
            "expected_total_promotions": expected_promotions,
            "observed_total_promotions": observed_promotions,
            "task_audit": layer_audit,
        },
        "exact_packed_and_selector_bytes": {
            "passed": (
                observed_packed_bytes == expected_packed_bytes
                and observed_auxiliary_bytes == expected_selector_auxiliary_bytes
                and observed_total_bytes
                == expected_packed_bytes + expected_selector_auxiliary_bytes
            ),
            "expected_packed_bytes": expected_packed_bytes,
            "observed_packed_bytes": observed_packed_bytes,
            "expected_selector_auxiliary_bytes": expected_selector_auxiliary_bytes,
            "observed_selector_auxiliary_bytes": observed_auxiliary_bytes,
            "expected_total_bytes": (
                expected_packed_bytes + expected_selector_auxiliary_bytes
            ),
            "observed_total_bytes": observed_total_bytes,
        },
        "lower_nll_than_both_components": {
            "passed": primary_nll < adaptive_nll and primary_nll < static_nll,
            "primary": primary_nll,
            "adaptive_component": adaptive_nll,
            "static_component": static_nll,
        },
        "relative_nll_reduction_vs_plain_adaptive": {
            "passed": (
                relative_vs_adaptive is not None
                and relative_vs_adaptive >= CQER_MIN_RELATIVE_NLL_REDUCTION_VS_ADAPTIVE
            ),
            "observed": relative_vs_adaptive,
            "minimum": CQER_MIN_RELATIVE_NLL_REDUCTION_VS_ADAPTIVE,
        },
        "relative_nll_reduction_vs_strongest_static": {
            "passed": (
                relative_vs_static is not None
                and relative_vs_static >= MIN_RELATIVE_NLL_REDUCTION
            ),
            "observed": relative_vs_static,
            "minimum": MIN_RELATIVE_NLL_REDUCTION,
        },
        "top1_disadvantage_margin": {
            "passed": primary_top1 >= better_comparator_top1 - TOP1_DISADVANTAGE_MARGIN,
            "observed_disadvantage": better_comparator_top1 - primary_top1,
            "maximum": TOP1_DISADVANTAGE_MARGIN,
        },
        "cvar95_disadvantage_margin": {
            "passed": primary_cvar <= lower_comparator_cvar + CVAR95_DISADVANTAGE_MARGIN,
            "observed_disadvantage": primary_cvar - lower_comparator_cvar,
            "maximum": CVAR95_DISADVANTAGE_MARGIN,
        },
        "all_values_finite": {
            "passed": all_metrics_finite,
        },
        "exact_stage_consume_handshake": {
            "passed": selector_contract_passed,
            "task_audit": layer_audit,
        },
    }
    return {
        "schema": "recurquant.experiment007-cqer32-development-gate.v1",
        "applicable": True,
        "passed": all(check["passed"] is True for check in checks.values()),
        "primary": QUERY_EMA_PRIMARY,
        "comparators": {
            "plain_adaptive": ADAPTIVE_TARGET_FISHER,
            "static": static_name,
        },
        "checks": checks,
    }


def primary_claim_text(primary_name: str) -> str:
    """Describe the actual primary without implying a missing loss selector."""

    if primary_name == QUERY_EMA_PRIMARY:
        return (
            "The actual primary uses target-directional-Fisher per-layer quotas "
            "and causally weights each per-write aligned INT4-to-INT8 row-MSE "
            "reduction by a normalized-query-energy EMA with a 32-token half-life."
        )
    if primary_name == RANK_FUSION_PRIMARY:
        return (
            "The actual primary uses target-directional-Fisher per-layer quotas "
            "and equal-weight ordinal rank fusion between calibrated static row "
            "sensitivity and causal per-write INT4-to-INT8 MSE reduction."
        )
    if primary_name == ADAPTIVE_TARGET_FISHER:
        return (
            "The actual primary uses target-directional-Fisher per-layer quotas "
            "with causal per-write row-MSE selection."
        )
    return (
        f"The actual primary is {primary_name}, a static HRR row policy; no "
        "loss-selector primary is present."
    )


def diagnostic_exit_code(*, heldout_calibration: bool, gate_passed: object) -> int:
    """Return the prespecified non-zero status for a failed frozen holdout gate."""

    return 2 if heldout_calibration and gate_passed is not True else 0


def scores_from_artifact(evidence: dict[str, Any], name: str) -> dict[int, torch.Tensor]:
    try:
        score_record = evidence["scores"][name]
        arrays = score_record["arrays"]
        expected_arrays_sha256 = score_record["canonical_arrays_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"selector artifact lacks score arrays for {name}") from error
    if not isinstance(arrays, dict):
        raise ValueError(f"selector score arrays for {name} must be an object")
    actual_arrays_sha256 = sha256_bytes(canonical_json_bytes(arrays))
    if expected_arrays_sha256 != actual_arrays_sha256:
        raise ValueError(f"selector score array hash does not match for {name}")
    scores: dict[int, torch.Tensor] = {}
    for raw_layer, values in arrays.items():
        try:
            layer_index = int(raw_layer)
            tensor = torch.tensor(values, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid selector scores for layer {raw_layer!r}") from error
        if str(layer_index) != raw_layer or layer_index < 0 or layer_index in scores:
            raise ValueError(f"selector score layer key {raw_layer!r} is not canonical and unique")
        if tensor.ndim != 2 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"selector layer {layer_index} scores must be a finite matrix")
        scores[layer_index] = tensor
    if not scores:
        raise ValueError(f"selector score arrays for {name} must not be empty")
    return scores


def plan_from_artifact(evidence: dict[str, Any], name: str) -> ExactBudgetRowPlan:
    scores = scores_from_artifact(evidence, name)
    budget = evidence["byte_budget"]
    plan = select_rows_exact_budget(
        scores,
        target_resident_bytes=int(budget["target_resident_bytes"]),
        group_size=int(budget["group_size"]),
        scale_bits=int(budget["scale_bits"]),
    )
    recorded = evidence["plans"][name]
    expected_locations = tuple(
        (
            int(location["layer_index"]),
            int(location["head_index"]),
            int(location["row_index"]),
        )
        for location in recorded["locations"]
    )
    actual_locations = tuple(
        (location.layer_index, location.head_index, location.row_index)
        for location in plan.high_precision_rows
    )
    if expected_locations != actual_locations:
        raise ValueError(f"recomputed {name} plan does not match the selector artifact")
    return plan


def random_plan_like(template: ExactBudgetRowPlan) -> ExactBudgetRowPlan:
    generator = torch.Generator().manual_seed(RANDOM_ROW_SEED)
    scores = {
        layer_index: torch.rand((heads, rows), generator=generator)
        for layer_index, heads, rows in template.score_shapes
    }
    return select_rows_exact_budget(
        scores,
        target_resident_bytes=template.target_resident_bytes,
        low_bits=template.low_bits,
        high_bits=template.high_bits,
        group_size=template.group_size,
        scale_bits=template.scale_bits,
    )


def make_caches(
    model: Qwen3_5ForCausalLM,
    *,
    plans: dict[str, ExactBudgetRowPlan],
    adaptive_plans: dict[str, ExactBudgetRowPlan],
    rank_fusion_specs: Mapping[str, RankFusionCacheSpec] | None = None,
    query_ema_plans: Mapping[str, ExactBudgetRowPlan] | None = None,
) -> dict[
    str,
    PackedRecurrentStateCache
    | MixedPackedRecurrentStateCache
    | AdaptiveMixedPackedRecurrentStateCache
    | QueryEmaMixedPackedRecurrentStateCache
    | RankFusedMixedPackedRecurrentStateCache,
]:
    caches: dict[
        str,
        PackedRecurrentStateCache
        | MixedPackedRecurrentStateCache
        | AdaptiveMixedPackedRecurrentStateCache
        | QueryEmaMixedPackedRecurrentStateCache
        | RankFusedMixedPackedRecurrentStateCache,
    ] = {
        "uniform_int4": create_qwen35_packed_cache(model, bits=4),
        "v02_layer0_static": create_qwen35_v02_mixed_cache(model),
    }
    for name, plan in plans.items():
        caches[name] = create_qwen35_exact_budget_cache(model, plan=plan)
    for name, plan in adaptive_plans.items():
        if name in caches:
            raise ValueError(f"adaptive cache name duplicates another method: {name}")
        caches[name] = create_qwen35_adaptive_exact_budget_cache(model, plan=plan)
    score_device = next(model.parameters()).device
    for name, (plan, static_scores, static_rank_weight) in (rank_fusion_specs or {}).items():
        if name in caches:
            raise ValueError(f"rank-fusion cache name duplicates another method: {name}")
        scores_on_device = {
            layer_index: scores.to(score_device)
            for layer_index, scores in static_scores.items()
        }
        caches[name] = create_qwen35_rank_fused_exact_budget_cache(
            model,
            plan=plan,
            static_scores_by_layer=scores_on_device,
            static_rank_weight=static_rank_weight,
        )
    for name, plan in (query_ema_plans or {}).items():
        if name in caches:
            raise ValueError(f"query-EMA cache name duplicates another method: {name}")
        caches[name] = create_qwen35_query_ema_exact_budget_cache(model, plan=plan)
    return caches


def _append_metrics(
    accumulators: dict[str, _TokenAccumulator],
    reference_logits: torch.Tensor,
    candidate_logits: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> None:
    reference_finite = torch.isfinite(reference_logits).all()
    for name, logits in candidate_logits.items():
        accumulators[name].append(
            token_fidelity(reference_logits, logits, target),
            outputs_finite=reference_finite & torch.isfinite(logits).all(),
        )


def evaluate_task(
    model: Qwen3_5ForCausalLM,
    *,
    prompt_ids: torch.Tensor,
    code_ids: torch.Tensor,
    plans: dict[str, ExactBudgetRowPlan],
    adaptive_plans: dict[str, ExactBudgetRowPlan],
    rank_fusion_specs: Mapping[str, RankFusionCacheSpec] | None = None,
    query_ema_plans: Mapping[str, ExactBudgetRowPlan] | None = None,
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, dict[str, float | int]],
    dict[str, dict[str, int | float | bool]],
    dict[str, object],
    int,
]:
    reference_cache = DynamicCache(config=model.config)
    caches = make_caches(
        model,
        plans=plans,
        adaptive_plans=adaptive_plans,
        rank_fusion_specs=rank_fusion_specs,
        query_ema_plans=query_ema_plans,
    )
    aligned_accumulators = {name: _TokenAccumulator.empty() for name in caches}
    full_code_accumulators = {name: _TokenAccumulator.empty() for name in caches}

    query_ema_caches = {
        name: cache
        for name, cache in caches.items()
        if isinstance(cache, QueryEmaMixedPackedRecurrentStateCache)
    }
    observer_context = (
        Qwen35QueryEnergyObserver(model, caches=list(query_ema_caches.values()))
        if query_ema_caches
        else nullcontext()
    )
    with observer_context:
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
        _append_metrics(
            full_code_accumulators,
            reference_output.logits,
            {name: output.logits for name, output in candidate_outputs.items()},
            code_ids[:, :1],
        )

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
                aligned_accumulators,
                reference_output.logits,
                {name: output.logits for name, output in candidate_outputs.items()},
                target_token,
            )
            _append_metrics(
                full_code_accumulators,
                reference_output.logits,
                {name: output.logits for name, output in candidate_outputs.items()},
                target_token,
            )

    reference_bytes = sum(
        state.tensor.numel() * state.tensor.element_size()
        for state in iter_recurrent_states(reference_cache)
    )
    return (
        {name: accumulator.summary() for name, accumulator in aligned_accumulators.items()},
        {name: accumulator.summary() for name, accumulator in full_code_accumulators.items()},
        {name: cache.storage_summary() for name, cache in caches.items()},
        {name: cache.query_ema_diagnostics() for name, cache in query_ema_caches.items()},
        reference_bytes,
    )


def git_state() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    ).stdout.splitlines()
    return {"commit": commit, "worktree_clean": not status, "status": status}


def source_file_hashes(
    repository_root: Path,
    relative_paths: Sequence[str] = EVALUATOR_SOURCE_FILES,
) -> dict[str, str]:
    """Hash the frozen evaluator implementation files in stable path order."""

    resolved_root = repository_root.resolve()
    hashes: dict[str, str] = {}
    for raw_path in relative_paths:
        path = (resolved_root / raw_path).resolve()
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source path escapes repository root: {raw_path}") from error
        if not path.is_file():
            raise ValueError(f"required evaluator source file is missing: {relative}")
        hashes[relative] = sha256_bytes(path.read_bytes())
    return hashes


def validate_heldout_repository_start(
    repository: Mapping[str, object],
    selectors: Sequence[Mapping[str, Any]],
) -> None:
    """Require a clean frozen commit shared by evaluator and selector artifacts."""

    commit = repository.get("commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("heldout repository commit is missing")
    if repository.get("worktree_clean") is not True or repository.get("status") != []:
        raise ValueError("heldout-calibration requires a clean worktree at start")
    for selector in selectors:
        selector_repository = _require_mapping(
            selector.get("repository"),
            context="selector repository",
        )
        if selector_repository.get("commit") != commit:
            raise ValueError("selector artifact commit does not match heldout evaluator commit")
        if (
            selector_repository.get("worktree_clean") is not True
            or selector_repository.get("status") != []
        ):
            raise ValueError("selector artifact was not generated from a clean worktree")


def validate_heldout_output_path(output: Path, repository_root: Path) -> None:
    """Require heldout output to stay outside Git state or under an ignore rule."""

    resolved_root = repository_root.resolve()
    resolved_output = output.resolve()
    try:
        relative = resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=resolved_root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "heldout output inside the repository must be Git-ignored so the worktree remains clean"
        )


def validate_heldout_repository_end(
    *,
    start_repository: Mapping[str, object],
    end_repository: Mapping[str, object],
    start_source_hashes: Mapping[str, str],
    end_source_hashes: Mapping[str, str],
) -> None:
    """Refuse heldout evidence if commit, worktree, or source files changed mid-run."""

    if end_repository.get("commit") != start_repository.get("commit"):
        raise RuntimeError("repository commit changed during heldout-calibration")
    if end_repository.get("worktree_clean") is not True or end_repository.get("status") != []:
        raise RuntimeError("worktree changed during heldout-calibration")
    if dict(end_source_hashes) != dict(start_source_hashes):
        raise RuntimeError("evaluator source files changed during heldout-calibration")


def validate_storage_boundary_prerequisite(
    evidence: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_model: Mapping[str, Any],
) -> None:
    """Authenticate and independently check the frozen local derivative gate summary."""

    if evidence.get("artifact_kind") != STORAGE_BOUNDARY_ARTIFACT_KIND:
        raise ValueError("unexpected storage-boundary artifact kind")
    model = _require_mapping(evidence.get("model"), context="storage-boundary model")
    if model.get("id") != expected_model.get("id") or model.get("revision") != expected_model.get(
        "revision"
    ):
        raise ValueError("storage-boundary model does not match selector model")
    if model.get("dtype") != "torch.float32":
        raise ValueError("storage-boundary gate must use torch.float32")

    gate = _require_mapping(
        evidence.get("derivative_gate"),
        context="storage-boundary derivative gate",
    )
    if gate.get("passed") is not True or gate.get("failures") != []:
        raise ValueError("storage-boundary derivative gate did not pass")
    thresholds = _require_mapping(gate.get("thresholds"), context="derivative thresholds")
    expected_thresholds = {
        "model_dtype": "torch.float32",
        "baseline_repeat_absolute_tolerance": 1e-7,
        "derivative_informative_floor": 1e-8,
        "near_zero_absolute_tolerance": 2e-7,
        "minimum_informative_rows": 3,
        "minimum_sign_agreement": 0.95,
        "maximum_median_relative_error": 0.10,
        "minimum_converged_row_fraction": 0.75,
    }
    if dict(thresholds) != expected_thresholds:
        raise ValueError("storage-boundary derivative thresholds do not match the freeze")
    observed = _require_mapping(gate.get("observed"), context="derivative observations")
    numeric_passed = (
        int(observed.get("rows", -1)) == 4
        and int(observed.get("informative_rows", -1)) >= 3
        and _finite_float(
            observed.get("maximum_baseline_repeat_absolute_error"),
            context="maximum baseline repeat error",
        )
        <= 1e-7
        and _finite_float(observed.get("sign_agreement"), context="sign agreement") >= 0.95
        and _finite_float(
            observed.get("median_relative_error"),
            context="median relative error",
        )
        <= 0.10
        and _finite_float(
            observed.get("converged_row_fraction"),
            context="converged row fraction",
        )
        >= 0.75
        and int(observed.get("near_zero_checks_passed", -1))
        == int(observed.get("near_zero_checks", -2))
    )
    if not numeric_passed:
        raise ValueError("storage-boundary numeric observations do not pass the frozen gate")

    implementation = _require_mapping(
        evidence.get("implementation"),
        context="storage-boundary implementation",
    )
    source_start = _require_mapping(
        implementation.get("source_hashes_start"),
        context="storage-boundary source hashes start",
    )
    source_end = _require_mapping(
        implementation.get("source_hashes_end"),
        context="storage-boundary source hashes end",
    )
    if (
        not source_start
        or dict(source_start) != dict(source_end)
        or implementation.get("unchanged_during_run") is not True
    ):
        raise ValueError("storage-boundary implementation source hashes are not stable")
    repository = _require_mapping(
        evidence.get("repository"),
        context="storage-boundary repository",
    )
    for phase in ("start", "end"):
        state = _require_mapping(
            repository.get(phase),
            context=f"storage-boundary repository {phase}",
        )
        if state.get("commit") != expected_commit:
            raise ValueError("storage-boundary commit does not match heldout evaluator")
        if state.get("worktree_clean") is not True or state.get("status") != []:
            raise ValueError("storage-boundary artifact was not generated cleanly")


def _package_versions() -> dict[str, str]:
    names = ("datasets", "numpy", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    args = parse_args()
    if args.limit is not None and not 1 <= args.limit <= 16:
        raise ValueError("--limit must be between 1 and 16")
    if args.calibration_offset < 0:
        raise ValueError("--calibration-offset must be non-negative")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.rank_fusion and args.query_ema:
        raise ValueError("--rank-fusion and --query-ema are mutually exclusive")
    heldout_calibration = args.calibration_offset > 0
    repository_root = Path(__file__).resolve().parents[1]
    repository_start = git_state()
    source_hashes_start = source_file_hashes(repository_root)

    selector, selector_sha256 = load_selector_artifact(
        args.selector_artifact,
        expected_kind=HRR_ARTIFACT_KIND,
    )
    validate_selector_contract(selector)
    loss_selector: dict[str, Any] | None = None
    loss_selector_sha256: str | None = None
    if args.loss_selector_artifact is not None:
        loss_selector, loss_selector_sha256 = load_selector_artifact(
            args.loss_selector_artifact,
            expected_kind=LOSS_ARTIFACT_KIND,
        )
        validate_compatible_selector(selector, loss_selector)
    if args.rank_fusion and loss_selector is None:
        raise ValueError("--rank-fusion requires --loss-selector-artifact")
    if args.query_ema and loss_selector is None:
        raise ValueError("--query-ema requires --loss-selector-artifact")
    storage_boundary: dict[str, Any] | None = None
    storage_boundary_sha256: str | None = None
    if args.storage_boundary_artifact is not None:
        storage_boundary, storage_boundary_sha256 = load_selector_artifact(
            args.storage_boundary_artifact,
            expected_kind=STORAGE_BOUNDARY_ARTIFACT_KIND,
        )
    selectors: list[dict[str, Any]] = [selector]
    if loss_selector is not None:
        selectors.append(loss_selector)
    validate_cqer_selector_artifacts(enabled=args.query_ema, selectors=selectors)
    all_selector_task_ids = selector_task_ids(selectors)
    task_records = selector["dataset"]["tasks"]
    available_tasks = len(task_records)
    limit = available_tasks if args.limit is None else args.limit
    validate_cqer_development_request(
        enabled=args.query_ema,
        offset=args.calibration_offset,
        limit=limit,
        bootstrap_samples=args.bootstrap_samples,
    )
    validate_frozen_holdout_request(
        offset=args.calibration_offset,
        limit=limit,
        bootstrap_samples=args.bootstrap_samples,
        selectors=selectors,
        loss_selector_present=loss_selector is not None,
        storage_boundary_present=storage_boundary is not None,
        rank_fusion_enabled=args.rank_fusion,
        query_ema_enabled=args.query_ema,
    )
    if heldout_calibration:
        validate_heldout_repository_start(repository_start, selectors)
        validate_heldout_output_path(args.output, repository_root)
        assert storage_boundary is not None
        validate_storage_boundary_prerequisite(
            storage_boundary,
            expected_commit=str(repository_start["commit"]),
            expected_model=selector["model"],
        )
    if args.calibration_offset == 0 and limit > available_tasks:
        raise ValueError(
            f"--limit={limit} exceeds the selector's {available_tasks} calibration tasks"
        )
    window_stop = args.calibration_offset + limit
    if window_stop > MBPP_CALIBRATION_SIZE:
        raise ValueError(
            "calibration window exceeds the frozen ranked calibration population: "
            f"{window_stop} > {MBPP_CALIBRATION_SIZE}"
        )
    ranked_rows = load_mbpp_rows("calibration", limit=window_stop)
    selector_prefix_rows = tuple(ranked_rows[:FROZEN_HOLDOUT_LIMIT])
    if heldout_calibration or args.query_ema:
        selector_prefix_ids = [int(row["task_id"]) for row in selector_prefix_rows]
        selector_prefix_manifest = mbpp_manifest(selector_prefix_rows, phase="calibration")
        for selector_evidence in selectors:
            authenticate_selector_prefix(
                selector_evidence,
                ranked_prefix_manifest=selector_prefix_manifest,
                ranked_prefix_task_ids=selector_prefix_ids,
            )
    rows = select_calibration_window(
        ranked_rows,
        offset=args.calibration_offset,
        limit=limit,
        selectors=selectors,
    )
    actual_ids = [row["task_id"] for row in rows]
    validate_cqer_development_task_ids(enabled=args.query_ema, task_ids=actual_ids)
    if args.calibration_offset == 0:
        expected_ids = [int(record["task_id"]) for record in task_records[:limit]]
        if actual_ids != expected_ids:
            raise ValueError("calibration task IDs do not match the selector artifact prefix")
    actual_manifest = mbpp_manifest(rows, phase="calibration")
    if args.calibration_offset == 0:
        validate_calibration_prefix(
            selector,
            actual_manifest=actual_manifest,
            expected_task_ids=actual_ids,
        )
    actual_manifest_sha256 = mbpp_manifest_sha256(rows, phase="calibration")
    if actual_manifest_sha256 != mbpp_manifest_content_sha256(actual_manifest):
        raise RuntimeError("calibration manifest helpers produced inconsistent hashes")

    horizon = int(selector["method"]["horizon"])
    hrr_primary_name = f"hrr_h{horizon}"
    h1_plan = plan_from_artifact(selector, "hrr_h1")
    hrr_primary_plan = plan_from_artifact(selector, hrr_primary_name)
    mse_plan = plan_from_artifact(selector, "row_mse")
    plans = {
        "hrr_h1": h1_plan,
        hrr_primary_name: hrr_primary_plan,
        "row_mse": mse_plan,
        "random_rows_s1101": random_plan_like(hrr_primary_plan),
    }
    adaptive_plans = {ADAPTIVE_H1: h1_plan}
    rank_fusion_specs: dict[str, RankFusionCacheSpec] = {}
    query_ema_plans: dict[str, ExactBudgetRowPlan] = {}
    primary_name = hrr_primary_name
    primary_plan = hrr_primary_plan
    if loss_selector is not None:
        plans.update({name: plan_from_artifact(loss_selector, name) for name in LOSS_SCORE_NAMES})
        adaptive_plans[ADAPTIVE_TARGET_FISHER] = plans["target_directional_fisher_difference_int4"]
        primary_name = ADAPTIVE_TARGET_FISHER
        primary_plan = adaptive_plans[primary_name]
        if args.rank_fusion:
            target_fisher_scores = scores_from_artifact(
                loss_selector,
                "target_directional_fisher_difference_int4",
            )
            rank_fusion_specs = {
                name: (primary_plan, target_fisher_scores, weight)
                for name, weight in RANK_FUSION_METHODS
            }
            primary_name = RANK_FUSION_PRIMARY
        if args.query_ema:
            query_ema_plans = {QUERY_EMA_PRIMARY: primary_plan}
            primary_name = QUERY_EMA_PRIMARY
    cqer_plan_quotas = {
        layer_index: len(primary_plan.groups_for_layer(layer_index))
        for layer_index, _, _ in primary_plan.score_shapes
    }
    validate_cqer_layer_quotas(enabled=args.query_ema, quotas=cqer_plan_quotas)

    torch.manual_seed(SEED)
    device = select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_record = selector["model"]
    if model_record["dtype"] != str(dtype):
        raise ValueError(
            "evaluation dtype does not match selector calibration dtype: "
            f"{dtype} != {model_record['dtype']}"
        )
    quantizer_contract = _selector_quantizer_contract(selector)
    for name in ("int4", "int8"):
        spec = quantizer_contract[name]
        if (
            spec["rounding"] != "nearest"
            or int(spec["seed"]) != SEED
            or float(spec["epsilon"]) != 1e-12
            or int(spec["flatten_last_dims"]) != 1
        ):
            raise ValueError(
                f"selector {name} quantizer does not match the physical evaluator contract"
            )
    model_id = str(model_record["id"])
    revision = str(model_record["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )

    encoded_tasks, token_manifest = encode_task_rows(tokenizer, rows)
    authenticated_selector_prefix: dict[str, object] | None = None
    if heldout_calibration or args.query_ema:
        _, selector_prefix_token_manifest = encode_task_rows(tokenizer, selector_prefix_rows)
        for selector_evidence in selectors:
            validate_actual_token_manifest(
                selector_evidence,
                selector_prefix_token_manifest,
            )
        authenticated_selector_prefix = {
            "manifest": selector_prefix_manifest,
            "manifest_sha256": mbpp_manifest_content_sha256(selector_prefix_manifest),
            "ordered_task_ids": selector_prefix_ids,
            "token_manifest": selector_prefix_token_manifest,
            "selector_count": len(selectors),
            "all_selectors_matched": True,
        }
    else:
        validate_actual_token_manifest(selector, token_manifest)
        if loss_selector is not None:
            validate_actual_token_manifest(loss_selector, token_manifest)

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

    per_task: dict[str, list[dict[str, float | int]]] = {}
    per_task_full_code: dict[str, list[dict[str, float | int]]] = {}
    per_task_query_ema_diagnostics: dict[str, list[dict[str, object]]] = {}
    storage_anchor: dict[str, dict[str, int | float | bool]] | None = None
    reference_state_bytes: int | None = None
    with torch.inference_mode():
        for task_number, (row, prompt_cpu, code_cpu) in enumerate(encoded_tasks, start=1):
            prompt_ids = prompt_cpu.to(device)
            code_ids = code_cpu.to(device)
            (
                summaries,
                full_code_summaries,
                storage,
                query_ema_diagnostics,
                task_reference_bytes,
            ) = evaluate_task(
                model,
                prompt_ids=prompt_ids,
                code_ids=code_ids,
                plans=plans,
                adaptive_plans=adaptive_plans,
                rank_fusion_specs=rank_fusion_specs,
                query_ema_plans=query_ema_plans,
            )
            if storage_anchor is None:
                storage_anchor = storage
                reference_state_bytes = task_reference_bytes
            elif storage != storage_anchor or task_reference_bytes != reference_state_bytes:
                raise RuntimeError("resident state storage changed between calibration tasks")
            for name, summary in summaries.items():
                per_task.setdefault(name, []).append({"task_id": row["task_id"], **summary})
            for name, summary in full_code_summaries.items():
                per_task_full_code.setdefault(name, []).append(
                    {"task_id": row["task_id"], **summary}
                )
            for name, diagnostics in query_ema_diagnostics.items():
                per_task_query_ema_diagnostics.setdefault(name, []).append(
                    {"task_id": row["task_id"], "layers": diagnostics}
                )
            print(
                f"[{task_number}/{len(rows)}] task={row['task_id']} "
                f"code_tokens={code_ids.shape[1]}",
                flush=True,
            )

    assert storage_anchor is not None
    assert reference_state_bytes is not None
    for name in plans:
        summary = storage_anchor[name]
        if summary["resident_bytes"] != plans[name].resident_bytes:
            raise RuntimeError(f"{name} did not realize its exact resident-byte plan")
    for name in adaptive_plans:
        summary = storage_anchor[name]
        if summary["resident_bytes"] != adaptive_plans[name].resident_bytes:
            raise RuntimeError(f"{name} did not realize its exact resident-byte plan")
    for name, (plan, _, _) in rank_fusion_specs.items():
        summary = storage_anchor[name]
        if summary["resident_bytes"] != plan.resident_bytes:
            raise RuntimeError(f"{name} did not realize its exact resident-byte plan")
    expected_selector_auxiliary_bytes = sum(
        heads * rows * torch.empty((), dtype=torch.float32).element_size()
        for _, heads, rows in primary_plan.score_shapes
    )
    for name, plan in query_ema_plans.items():
        summary = storage_anchor[name]
        if summary["resident_bytes"] != plan.resident_bytes:
            raise RuntimeError(f"{name} did not realize its exact packed-state byte plan")
        if summary["selector_auxiliary_bytes"] != expected_selector_auxiliary_bytes:
            raise RuntimeError(f"{name} did not realize the frozen selector auxiliary bytes")
        if summary["resident_bytes_including_selector"] != (
            plan.resident_bytes + expected_selector_auxiliary_bytes
        ):
            raise RuntimeError(f"{name} selector-aware resident byte total is inconsistent")
    if storage_anchor["v02_layer0_static"]["resident_bytes"] != primary_plan.resident_bytes:
        raise RuntimeError("v0.2 static and the primary row plan are not equal-byte")

    aggregates = aggregate_task_rows(per_task)
    aggregates_full_code = aggregate_task_rows(per_task_full_code)

    primary_values = [float(row["delta_nll"]) for row in per_task[primary_name]]
    contrasts = {
        name: paired_contrast(
            [float(row["delta_nll"]) for row in task_rows],
            primary_values,
            samples=args.bootstrap_samples,
            seed=SEED,
        )
        for name, task_rows in per_task.items()
        if name != primary_name
    }
    repository_end = git_state()
    source_hashes_end = source_file_hashes(repository_root)
    development_gate: dict[str, Any] | None = None
    if args.query_ema:
        expected_quotas = dict(CQER_FROZEN_LAYER_QUOTAS)
        development_gate = evaluate_cqer_development_gate(
            aggregates=aggregates,
            per_task=per_task,
            per_task_full_code=per_task_full_code,
            storage=storage_anchor,
            query_ema_diagnostics=per_task_query_ema_diagnostics,
            expected_quotas=expected_quotas,
            expected_packed_bytes=primary_plan.resident_bytes,
            expected_selector_auxiliary_bytes=expected_selector_auxiliary_bytes,
        )
        development_gate["checks"]["authenticated_repository_sources_and_manifests"] = {
            "passed": (
                repository_start["commit"] == repository_end["commit"]
                and repository_start["worktree_clean"] is True
                and repository_end["worktree_clean"] is True
                and source_hashes_start == source_hashes_end
            ),
            "repository_commit_stable": (
                repository_start["commit"] == repository_end["commit"]
            ),
            "worktree_clean_at_start_and_end": (
                repository_start["worktree_clean"] is True
                and repository_end["worktree_clean"] is True
            ),
            "source_hashes_stable": source_hashes_start == source_hashes_end,
            "selector_artifacts_authenticated": True,
            "evaluation_manifest_authenticated": True,
        }
        development_gate["passed"] = all(
            check["passed"] is True
            for check in development_gate["checks"].values()
        )
    if heldout_calibration:
        validate_heldout_repository_end(
            start_repository=repository_start,
            end_repository=repository_end,
            start_source_hashes=source_hashes_start,
            end_source_hashes=source_hashes_end,
        )
        heldout_gate = evaluate_frozen_holdout_gate(
            aggregates=aggregates,
            per_task=per_task,
            contrasts=contrasts,
            storage=storage_anchor,
            primary_name=primary_name,
        )
        heldout_gate["checks"]["authenticated_repository_sources_and_manifests"] = {
            "passed": True,
            "repository_commit_stable": True,
            "worktree_clean_at_start_and_end": True,
            "source_hashes_stable": True,
            "selector_prefixes_authenticated": True,
            "evaluation_manifest_authenticated": True,
            "storage_boundary_prerequisite_passed": True,
        }
        heldout_gate["passed"] = all(
            check["passed"] is True for check in heldout_gate["checks"].values()
        )
    else:
        heldout_gate = {
            "schema": (
                "recurquant.experiment007-heldout-gate.v1"
                if args.query_ema
                else (
                    "recurquant.experiment006-heldout-gate.v1"
                    if args.rank_fusion
                    else "recurquant.experiment005-heldout-gate.v1"
                )
            ),
            "applicable": False,
            "passed": None,
            "reason": (
                "same-calibration diagnostics cannot satisfy the frozen Experiment 007 "
                "holdout gate; CQER-32 positive offsets remain disabled"
                if args.query_ema
                else (
                    "same-calibration diagnostics cannot satisfy the frozen Experiment 006 "
                    "holdout gate; rank-fusion positive offsets remain disabled"
                    if args.rank_fusion
                    else "same-calibration diagnostics cannot satisfy the frozen holdout gate"
                )
            ),
        }
    selector_artifacts = {
        "hrr": {
            "path": str(args.selector_artifact.resolve()),
            "sha256": selector_sha256,
            "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(selector)),
        }
    }
    quality_artifact_kind = (
        "recurquant_hrr_heldout_calibration_quality_diagnostic"
        if heldout_calibration
        else "recurquant_hrr_same_calibration_quality_diagnostic"
    )
    if loss_selector is not None:
        assert args.loss_selector_artifact is not None
        assert loss_selector_sha256 is not None
        selector_artifacts["loss_sensitivity"] = {
            "path": str(args.loss_selector_artifact.resolve()),
            "sha256": loss_selector_sha256,
            "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(loss_selector)),
        }
        quality_artifact_kind = (
            "recurquant_adaptive_row_packing_heldout_calibration_quality_diagnostic"
            if heldout_calibration
            else "recurquant_adaptive_row_packing_same_calibration_quality_diagnostic"
        )
    if args.rank_fusion:
        quality_artifact_kind = "recurquant_rank_fusion_same_calibration_quality_diagnostic"
    if args.query_ema:
        quality_artifact_kind = "recurquant_cqer32_same_calibration_quality_diagnostic"
    prerequisite_artifacts: dict[str, dict[str, Any]] = {}
    if storage_boundary is not None:
        assert args.storage_boundary_artifact is not None
        assert storage_boundary_sha256 is not None
        prerequisite_artifacts["storage_boundary"] = {
            "path": str(args.storage_boundary_artifact.resolve()),
            "sha256": storage_boundary_sha256,
            "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(storage_boundary)),
            "artifact_kind": storage_boundary["artifact_kind"],
            "derivative_gate_passed": storage_boundary["derivative_gate"]["passed"],
        }
    primary_claim = primary_claim_text(primary_name)
    if heldout_calibration:
        claim_boundary = (
            "This is a heldout-calibration diagnostic: every evaluation task is a "
            "ranked calibration row disjoint from every selector artifact task, and "
            "all static plans and adaptive layer quotas come unchanged from those "
            "selector artifacts. It is not "
            "development or confirmation evidence and cannot establish final "
            "generalization, novelty, speed, or a breakthrough. "
            f"{primary_claim} The primary metric "
            "excludes the prompt-to-first-code-token prediction because no stored "
            "quantized recurrent state can affect that output."
        )
    else:
        claim_boundary = (
            "The selector and quality diagnostic use the same MBPP calibration tasks. "
            "This can catch implementation failures but cannot establish held-out "
            "generalization, novelty, speed, or a breakthrough. "
            f"{primary_claim} The primary metric "
            "excludes the prompt-to-first-code-token prediction because no stored "
            "quantized recurrent state can affect that output."
        )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": quality_artifact_kind,
        "diagnostic_only": True,
        "claim_boundary": claim_boundary,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selector_artifacts": selector_artifacts,
        "prerequisite_artifacts": prerequisite_artifacts,
        "model": {
            "id": model_id,
            "revision": revision,
            "dtype": str(dtype),
            "device": str(device),
        },
        "dataset": {
            "phase": "calibration",
            "manifest_sha256": actual_manifest_sha256,
            "content_manifest_sha256": actual_manifest_sha256,
            "manifest": actual_manifest,
            "task_count": len(rows),
            "tasks": token_manifest,
            "selection_mode": (
                "heldout_calibration" if heldout_calibration else "selector_task_prefix"
            ),
            "selection_window": {
                "calibration_offset": args.calibration_offset,
                "limit": limit,
                "stop_exclusive": window_stop,
                "resolved_before_tokenization_and_model_load": True,
            },
            "evaluation_task_ids": actual_ids,
            "selector_task_ids": sorted(all_selector_task_ids),
            "selector_task_prefix": not heldout_calibration and limit < available_tasks,
            "disjoint_from_all_selector_artifacts": (True if heldout_calibration else None),
            "authenticated_selector_prefix": authenticated_selector_prefix,
        },
        "metric_contract": {
            "primary": "calibration-aligned code transitions after recurrent-state storage",
            "primary_tokens_per_task": "code_tokens - 1",
            "excluded_from_primary": "prompt-to-first-code-token prediction",
            "secondary": "full reference-code tokens, including the unaffected first token",
            "contrasts": "paired task-macro baseline delta NLL minus primary delta NLL",
        },
        "methods": list(per_task),
        "primary": primary_name,
        "adaptive_policy_contracts": {
            ADAPTIVE_H1: {
                "selection": "per-update aligned INT4-to-INT8 row MSE reduction",
                "layer_quota_source": "hrr_h1 selector plan",
                "batch_size": 1,
                "resident_bytes": adaptive_plans[ADAPTIVE_H1].resident_bytes,
            },
            **(
                {
                    ADAPTIVE_TARGET_FISHER: {
                        "selection": "per-update aligned INT4-to-INT8 row MSE reduction",
                        "layer_quota_source": (
                            "target_directional_fisher_difference_int4 selector plan"
                        ),
                        "batch_size": 1,
                        "resident_bytes": adaptive_plans[ADAPTIVE_TARGET_FISHER].resident_bytes,
                    }
                }
                if loss_selector is not None
                else {}
            ),
            **{
                name: {
                    "selection": (
                        "per-layer ordinal rank fusion of calibrated target-directional-"
                        "Fisher scores and per-write aligned INT4-to-INT8 row MSE reduction"
                    ),
                    "static_rank_weight": weight,
                    "dynamic_rank_weight": 1.0 - weight,
                    "rank_normalization": (
                        "zero-best ordinal positions, stable flattened row ties"
                    ),
                    "layer_quota_source": (
                        "target_directional_fisher_difference_int4 selector plan"
                    ),
                    "batch_size": 1,
                    "resident_bytes": plan.resident_bytes,
                }
                for name, (plan, _, weight) in rank_fusion_specs.items()
            },
            **{
                name: {
                    "selection": (
                        "causal normalized-query-energy EMA multiplied by per-write "
                        "aligned INT4-to-INT8 row-MSE reduction"
                    ),
                    "query_normalization": "q / sqrt(sum(q^2) + 1e-6), computed in FP32",
                    "query_energy_half_life_tokens": QUERY_EMA_HALF_LIFE,
                    "query_energy_decay": 2.0 ** (-1.0 / QUERY_EMA_HALF_LIFE),
                    "initial_query_energy": "uniform 1 / key-row count",
                    "layer_quota_source": (
                        "target_directional_fisher_difference_int4 selector plan"
                    ),
                    "batch_size": 1,
                    "packed_recurrent_state_bytes": plan.resident_bytes,
                    "selector_auxiliary_bytes": expected_selector_auxiliary_bytes,
                    "resident_bytes_including_selector": (
                        plan.resident_bytes + expected_selector_auxiliary_bytes
                    ),
                }
                for name, plan in query_ema_plans.items()
            },
        },
        "storage": {
            "fp32_reference_recurrent_state_bytes": reference_state_bytes,
            "candidates": storage_anchor,
        },
        "aggregates": aggregates,
        "aggregates_full_code_secondary": aggregates_full_code,
        "contrasts_baseline_minus_primary_aligned_delta_nll": contrasts,
        "heldout_gate": heldout_gate,
        "development_gate": development_gate,
        "per_task": per_task,
        "per_task_full_code_secondary": per_task_full_code,
        "query_ema_diagnostics": per_task_query_ema_diagnostics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "repository": {
            "commit": repository_end["commit"],
            "worktree_clean": repository_end["worktree_clean"],
            "status": repository_end["status"],
            "start": repository_start,
            "end": repository_end,
            "stable_commit": repository_start["commit"] == repository_end["commit"],
        },
        "source_files": {
            "paths": list(EVALUATOR_SOURCE_FILES),
            "sha256_start": source_hashes_start,
            "sha256_end": source_hashes_end,
            "stable": source_hashes_start == source_hashes_end,
        },
        "command": [sys.executable, *sys.argv],
    }
    canonical_evidence = canonical_json_bytes(evidence)
    artifact = {
        "schema_version": 1,
        "artifact_kind": quality_artifact_kind,
        "canonical_evidence_sha256": sha256_bytes(canonical_evidence),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": sha256_bytes(payload),
                "primary": primary_name,
                "heldout_gate": heldout_gate,
                "development_gate": development_gate,
                "aggregates": aggregates,
                "contrasts": contrasts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return diagnostic_exit_code(
        heldout_calibration=heldout_calibration,
        gate_passed=heldout_gate["passed"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
