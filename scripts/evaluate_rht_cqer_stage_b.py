#!/usr/bin/env python3
"""Evaluate the frozen Experiment 009 Stage-B development window.

This evaluator accepts only a separately committed Stage-B identity artifact.
It has no task, rank-window, seed, bootstrap-count, method, or threshold flags.
The protected ranked MBPP window [8, 16) is never resolved by this script.
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
import string
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.evaluation import paired_bootstrap_mean_improvement
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.mixed_quantization import quantize_pack_mixed
from recurquant.public_data import mbpp_manifest_content_sha256
from recurquant.quantization import QuantizationSpec
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.qwen35 import (
    create_qwen35_adaptive_exact_budget_cache,
    create_qwen35_exact_budget_cache,
    create_qwen35_query_ema_exact_budget_cache,
    create_qwen35_right_rht_query_ema_exact_budget_cache,
)
from recurquant.rht import right_rht_encode, right_rht_signs
from recurquant.row_policy import ExactBudgetRowPlan, RowLocation

pilot = importlib.import_module(
    "scripts.pilot_evaluate_hrr" if __package__ else "pilot_evaluate_hrr"
)
screen = importlib.import_module(
    "scripts.screen_rht_cqer" if __package__ else "screen_rht_cqer"
)
identity_resolver = importlib.import_module(
    "scripts.resolve_rht_cqer_stage_b_identity"
    if __package__
    else "resolve_rht_cqer_stage_b_identity"
)

SEED = 2339
BOOTSTRAP_SAMPLES = 10_000
STAGE_B_OFFSET = 32
STAGE_B_LIMIT = 32
STAGE_B_STOP = 64
PROTECTED_WINDOW = (8, 16)

ARTIFACT_KIND = "recurquant_rht_cqer32_stage_b_development"
IDENTITY_ARTIFACT_KIND = "recurquant_rht_cqer32_stage_b_identity"
STAGE_A_ARTIFACT_KIND = "recurquant_rht_cqer32_stage_a_screen"
RESULT_CLAIM_BOUNDARY = (
    "This is the frozen 32-task Experiment 009 development result for a "
    "known right-RHT codec composed with CQER-32 on one pinned model and "
    "MBPP window. It is not confirmation, novelty, speed, state-of-the-art, "
    "deployment, or breakthrough evidence."
)
PRIMARY_STATE_ERROR_METRIC = (
    "closed-loop local codec reconstruction SSE: materialized packed "
    "state minus that candidate trajectory's own pre-pack source"
)
REFERENCE_ALIGNED_STATE_ERROR_METRIC = (
    "materialized candidate state minus matched FP32 recurrent state "
    "after each write; secondary and not an advancement gate"
)
PROTECTED_EVALUATION_FIELD = (
    "protected_window_8_16_content_selected_retained_canonicalized_"
    "formatted_tokenized_passed_to_model_or_evaluated"
)
DATA_ACCESS_TRANSPORT_LIMITATION = (
    "The Hugging Face streaming transport may deserialize complete source "
    "records before yielding mappings. These counters describe fields read "
    "and rows retained by RecurQuant application code."
)
STAGE_A_PATH_PRIVACY_LIMITATION = (
    "the immutable Stage-A artifact predates the shareability rule and "
    "contains local absolute paths internally; they are not copied here"
)

STATIC_METHOD = "target_directional_fisher_difference_int4"
ADAPTIVE_METHOD = "adaptive_mse_target_directional_fisher_quota"
CQER_METHOD = "query_ema32_weighted_mse_target_fisher_quota"
RHT_METHOD = "right_rht_query_ema32_weighted_mse_target_fisher_quota"
METHODS = (STATIC_METHOD, ADAPTIVE_METHOD, CQER_METHOD, RHT_METHOD)
QUERY_METHODS = (CQER_METHOD, RHT_METHOD)
TARGET_FISHER_SCORE = STATIC_METHOD

FROZEN_LAYER_QUOTAS = dict(pilot.CQER_FROZEN_LAYER_QUOTAS)
FROZEN_LINEAR_LAYERS = tuple(FROZEN_LAYER_QUOTAS)
TARGET_PROMOTED_ROWS = 1_976
TARGET_PAYLOAD_BYTES = 2_485_760
TARGET_SCALE_BYTES = 73_728
TARGET_MASK_BYTES = 4_608
TARGET_PACKED_STATE_BYTES = 2_564_096
TARGET_SELECTOR_BYTES = 147_456
TARGET_TOTAL_BYTES = 2_711_552
EXPECTED_SIGN_SCHEDULE_SHA256 = screen.EXPECTED_SIGN_SCHEDULE_SHA256
MAX_RHT_INVERSE_RELATIVE_L2 = 3e-7
PRODUCTION_SELF_CHECK_STORAGE_BYTES = 1_645
INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD = 2e-6
INDEPENDENT_PACK_MAX_ABS_THRESHOLD = 6e-6
INDEPENDENT_REFERENCE_DESCRIPTION = (
    "independent dense NumPy Hadamard, SHA-256 sign derivation, "
    "FP16-scale absmax quantizer, and decode"
)

MIN_DELTA_NLL_REDUCTION = 0.20
MIN_TASK_WINS = 20
MAX_TOP1_DISADVANTAGE = 0.005
MAX_TASK_NLL_DISADVANTAGE = 0.25
MIN_STATE_SSE_REDUCTION = 0.50

METRIC_FIELDS = (
    "mean_kl",
    "cvar95_kl",
    "max_kl",
    "top1_agreement",
    "reference_nll",
    "candidate_nll",
    "delta_nll",
)

SOURCE_FILES = identity_resolver.STAGE_B_SOURCE_FILES
RESULT_EVIDENCE_FIELDS = {
    "schema_version",
    "artifact_kind",
    "diagnostic_only",
    "claim_boundary",
    "created_at_utc",
    "protocol",
    "prerequisite_artifacts",
    "model",
    "dataset",
    "metric_contract",
    "methods",
    "storage",
    "aggregates",
    "aggregates_full_code_secondary",
    "per_task",
    "per_task_full_code_secondary",
    "per_task_token_primitives",
    "per_task_full_code_token_primitives",
    "paired_bootstrap_cqer_minus_rht_aligned_delta_nll",
    "selector_diagnostics",
    "state_error",
    "unit_evidence",
    "integrity_inputs",
    "stage_b_integrity",
    "stage_b_gate",
    "runtime_environment",
    "environment",
    "repository",
    "source_files",
    "command_template",
}
RESULT_IMPORTED_MODULE_PATHS = {
    "pilot_evaluate_hrr": "scripts/pilot_evaluate_hrr.py",
    "screen_rht_cqer": "scripts/screen_rht_cqer.py",
    "stage_b_identity_resolver": "scripts/resolve_rht_cqer_stage_b_identity.py",
    "recurquant_cache": "src/recurquant/cache.py",
    "recurquant_evaluation": "src/recurquant/evaluation.py",
    "recurquant_packed_cache": "src/recurquant/packed_cache.py",
    "recurquant_qwen35": "src/recurquant/qwen35.py",
    "recurquant_rht": "src/recurquant/rht.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Experiment 009 RHT-CQER 32-task development evaluation."
        )
    )
    parser.add_argument("--stage-a-artifact", type=Path, required=True)
    parser.add_argument("--identity-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{context} must be finite")
    return rendered


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _gate_check(function: Any) -> dict[str, Any]:
    try:
        result = function()
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "error": str(error)}
    if not isinstance(result, Mapping):
        return {"passed": bool(result)}
    normalized = dict(result)
    normalized["passed"] = normalized.get("passed") is True
    return normalized


def _load_evidence_artifact(
    path: Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    verification = verify_evidence_artifact(path)
    if verification.get("valid") is not True:
        raise ValueError(
            f"{expected_kind} artifact failed evidence verification: "
            + "; ".join(str(error) for error in verification.get("errors", []))
        )
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{expected_kind} artifact must be strict UTF-8 JSON") from error
    if not isinstance(artifact, dict) or artifact.get("artifact_kind") != expected_kind:
        raise ValueError(f"unexpected artifact kind; expected {expected_kind}")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{expected_kind} evidence must be an object")
    expected = artifact.get("canonical_evidence_sha256")
    actual = _sha256_bytes(canonical_json_bytes(evidence))
    if expected != actual:
        raise ValueError(f"{expected_kind} canonical evidence hash does not match")
    return evidence, _sha256_bytes(payload)


def _git(*arguments: str, repository_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )


def authenticate_committed_file(path: Path, repository_root: Path) -> dict[str, str]:
    """Require a file's current bytes to be identical to the current Git tree."""

    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("authenticated prerequisite must be inside the repository") from error
    tracked = _git("ls-files", "--error-unmatch", "--", relative, repository_root=root)
    if tracked.returncode != 0:
        raise ValueError(f"authenticated prerequisite is not tracked by Git: {relative}")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    current = resolved.read_bytes()
    if current != committed:
        raise ValueError(f"authenticated prerequisite differs from HEAD: {relative}")
    return {
        "path": relative,
        "sha256": _sha256_bytes(current),
        "git_blob_sha256": _sha256_bytes(committed),
    }


def authenticate_imported_module_paths(
    repository_root: Path,
    *,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Require evaluator imports to resolve to the authenticated repository tree."""

    identity_resolver.validate_all_imported_repository_modules_frozen(
        repository_root,
        modules=modules,
    )
    expected = {
        "pilot_evaluate_hrr": (
            pilot,
            "scripts/pilot_evaluate_hrr.py",
        ),
        "screen_rht_cqer": (
            screen,
            "scripts/screen_rht_cqer.py",
        ),
        "stage_b_identity_resolver": (
            identity_resolver,
            "scripts/resolve_rht_cqer_stage_b_identity.py",
        ),
        "recurquant_cache": (
            importlib.import_module("recurquant.cache"),
            "src/recurquant/cache.py",
        ),
        "recurquant_evaluation": (
            importlib.import_module("recurquant.evaluation"),
            "src/recurquant/evaluation.py",
        ),
        "recurquant_packed_cache": (
            importlib.import_module("recurquant.packed_cache"),
            "src/recurquant/packed_cache.py",
        ),
        "recurquant_qwen35": (
            importlib.import_module("recurquant.qwen35"),
            "src/recurquant/qwen35.py",
        ),
        "recurquant_rht": (
            importlib.import_module("recurquant.rht"),
            "src/recurquant/rht.py",
        ),
    }
    root = repository_root.resolve()
    authenticated: dict[str, str] = {}
    for name, (module, relative) in expected.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise ValueError(f"{name} does not expose an import path")
        actual = Path(raw_path).resolve()
        wanted = (root / relative).resolve()
        if actual != wanted:
            raise ValueError(
                f"{name} resolved outside the authenticated repository: {actual}"
            )
        authenticated[name] = actual.relative_to(root).as_posix()
    return authenticated


def _require_ancestor(commit: object, *, repository_root: Path, context: str) -> str:
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{context} must record a 40-character commit")
    result = _git(
        "merge-base",
        "--is-ancestor",
        commit,
        "HEAD",
        repository_root=repository_root,
    )
    if result.returncode != 0:
        raise ValueError(f"{context} commit is not an ancestor of the evaluator commit")
    return commit


def authenticate_stage_a_prerequisite(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Recompute and authenticate the committed Stage-A advancement result."""

    tracked = authenticate_committed_file(path, repository_root)
    evidence, file_sha256 = _load_evidence_artifact(
        path,
        expected_kind=STAGE_A_ARTIFACT_KIND,
    )
    if tracked["sha256"] != file_sha256:
        raise ValueError("Stage-A tracked-file hash does not match the artifact hash")
    if evidence.get("screening_only") is not True:
        raise ValueError("Stage-A prerequisite must be the frozen screening artifact")
    protocol = _mapping(evidence.get("protocol"), context="Stage-A protocol")
    if (
        protocol.get("method") != RHT_METHOD
        or protocol.get("protected_ranked_window") != list(PROTECTED_WINDOW)
        or protocol.get("protected_window_accessed") is not False
    ):
        raise ValueError("Stage-A protocol contract drifted")
    repository = _mapping(evidence.get("repository"), context="Stage-A repository")
    stage_a_commit = _require_ancestor(
        repository.get("commit"),
        repository_root=repository_root,
        context="Stage-A repository",
    )
    start = _mapping(repository.get("start"), context="Stage-A repository start")
    end = _mapping(repository.get("end"), context="Stage-A repository end")
    dataset = _mapping(evidence.get("dataset"), context="Stage-A dataset")
    integrity = {
        "repository_clean_at_start": start.get("worktree_clean") is True,
        "repository_clean_at_end": end.get("worktree_clean") is True,
        "repository_commit_stable": (
            start.get("commit") == end.get("commit") == stage_a_commit
        ),
        "source_hashes_stable": (
            _mapping(
                evidence.get("source_files"),
                context="Stage-A source files",
            ).get("stable")
            is True
        ),
        "identity_authenticated_before_model_weights": (
            dataset.get("identity_authenticated_before_model_weights") is True
        ),
        "protected_window_8_16_accessed": (
            dataset.get("protected_window_8_16_loaded_tokenized_or_evaluated")
            is not False
        ),
    }
    recomputed = screen.evaluate_stage_a_gate(
        aligned_metrics=_mapping(
            evidence.get("metrics_aligned"),
            context="Stage-A aligned metrics",
        ),
        full_code_metrics=_mapping(
            evidence.get("metrics_full_code_secondary"),
            context="Stage-A full-code metrics",
        ),
        storage=_mapping(
            _mapping(evidence.get("storage"), context="Stage-A storage").get(
                "candidates"
            ),
            context="Stage-A candidate storage",
        ),
        selector_diagnostics=_mapping(
            evidence.get("selector_diagnostics"),
            context="Stage-A selector diagnostics",
        ),
        state_errors=_mapping(
            evidence.get("state_error"),
            context="Stage-A state error",
        ),
        unit_evidence=_mapping(
            evidence.get("unit_evidence"),
            context="Stage-A unit evidence",
        ),
        integrity=integrity,
    )
    if recomputed.get("passed") is not True:
        raise ValueError("Stage-A advancement gate does not pass when recomputed")
    recorded_gate = _mapping(evidence.get("stage_a_gate"), context="recorded Stage-A gate")
    recorded_checks = _mapping(
        recorded_gate.get("checks"),
        context="recorded Stage-A checks",
    )
    if recorded_gate.get("passed") is not True or any(
        _mapping(check, context="recorded Stage-A check").get("passed") is not True
        for check in recorded_checks.values()
    ):
        raise ValueError("recorded Stage-A gate did not pass every frozen check")
    return evidence, {
        **tracked,
        "artifact_sha256": file_sha256,
        "canonical_evidence_sha256": _sha256_bytes(canonical_json_bytes(evidence)),
        "implementation_commit": stage_a_commit,
        "gate_recomputed_and_passed": "true",
    }


def _identity_task_records(
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    dataset = _mapping(identity.get("dataset"), context="Stage-B identity dataset")
    tasks = dataset.get("tasks")
    if (
        isinstance(tasks, (str, bytes))
        or not isinstance(tasks, Sequence)
        or len(tasks) != STAGE_B_LIMIT
    ):
        raise ValueError("Stage-B identity must contain exactly 32 task records")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    expected_fields = {
        "rank",
        "task_id",
        "row_sha256",
        "prompt_tokens",
        "code_tokens",
        "aligned_scored_tokens",
        "full_code_scored_tokens",
        "prompt_text_sha256",
        "code_text_sha256",
        "prompt_token_ids_sha256",
        "code_token_ids_sha256",
    }
    for offset, raw in enumerate(tasks):
        record = dict(_mapping(raw, context=f"Stage-B identity task {offset}"))
        if set(record) != expected_fields:
            raise ValueError(f"Stage-B identity task {offset} fields drifted")
        rank = _strict_int(record.get("rank"), context=f"identity task {offset} rank")
        task_id = _strict_int(
            record.get("task_id"),
            context=f"identity task {offset} task_id",
        )
        if rank != STAGE_B_OFFSET + offset:
            raise ValueError("Stage-B identity ranks must be the ordered window [32, 64)")
        if task_id in seen:
            raise ValueError("Stage-B identity task IDs must be unique")
        seen.add(task_id)
        for field in (
            "row_sha256",
            "prompt_text_sha256",
            "code_text_sha256",
            "prompt_token_ids_sha256",
            "code_token_ids_sha256",
        ):
            value = record.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"identity task {task_id} {field} is not a SHA-256")
        prompt_tokens = _strict_int(
            record.get("prompt_tokens"),
            context=f"identity task {task_id} prompt_tokens",
        )
        code_tokens = _strict_int(
            record.get("code_tokens"),
            context=f"identity task {task_id} code_tokens",
        )
        aligned = _strict_int(
            record.get("aligned_scored_tokens"),
            context=f"identity task {task_id} aligned_scored_tokens",
        )
        full = _strict_int(
            record.get("full_code_scored_tokens"),
            context=f"identity task {task_id} full_code_scored_tokens",
        )
        if prompt_tokens < 1 or code_tokens < 2:
            raise ValueError(f"identity task {task_id} has invalid token counts")
        if aligned != code_tokens - 1 or full != code_tokens:
            raise ValueError(f"identity task {task_id} aligned/full token contract drifted")
        normalized.append(record)
    ordered_ids = dataset.get("ordered_task_ids")
    if ordered_ids != [record["task_id"] for record in normalized]:
        raise ValueError("Stage-B ordered task IDs differ from its task records")
    if dataset.get("selection_window") != {
        "offset": STAGE_B_OFFSET,
        "limit": STAGE_B_LIMIT,
        "stop_exclusive": STAGE_B_STOP,
    }:
        raise ValueError("Stage-B identity selection window must equal [32, 64)")
    if dataset.get("protected_window") != {
        "offset": PROTECTED_WINDOW[0],
        "stop_exclusive": PROTECTED_WINDOW[1],
        "content_retained_canonicalized_or_tokenized": False,
    }:
        raise ValueError("Stage-B identity protected window must equal [8, 16)")
    return tuple(normalized)


def authenticate_stage_b_identity(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, str], tuple[dict[str, Any], ...]]:
    """Authenticate the committed identity amendment without opening model weights."""

    tracked = authenticate_committed_file(path, repository_root)
    identity, file_sha256 = identity_resolver.load_stage_b_identity_artifact(path)
    identity_resolver.validate_stage_b_identity_evidence(identity)
    if identity.get("artifact_kind") != IDENTITY_ARTIFACT_KIND:
        raise ValueError("Stage-B identity evidence kind drifted")
    if tracked["sha256"] != file_sha256:
        raise ValueError("Stage-B identity tracked-file hash does not match its artifact")
    model = _mapping(identity.get("model_contract"), context="identity model contract")
    if (
        model.get("id") != identity_resolver.MODEL_ID
        or model.get("revision") != identity_resolver.MODEL_REVISION
        or model.get("weights_loaded") is not False
    ):
        raise ValueError("Stage-B identity model/no-weights contract drifted")
    tokenizer = _mapping(
        identity.get("tokenizer_contract"),
        context="identity tokenizer contract",
    )
    if (
        tokenizer.get("source_id") != model["id"]
        or tokenizer.get("revision") != model["revision"]
        or tokenizer.get("trust_remote_code") is not False
        or tokenizer.get("prompt_add_special_tokens") is not True
        or tokenizer.get("code_add_special_tokens") is not False
    ):
        raise ValueError("Stage-B identity tokenizer contract drifted")
    tasks = _identity_task_records(identity)
    dataset = _mapping(identity.get("dataset"), context="identity dataset")
    manifest = _mapping(dataset.get("manifest"), context="identity content manifest")
    if (
        mbpp_manifest_content_sha256(manifest)
        != dataset.get("content_manifest_sha256")
    ):
        raise ValueError("Stage-B identity content-manifest hash does not match")
    integrity = _mapping(identity.get("integrity"), context="identity integrity")
    required_true = (
        "stage_a_authenticated_before_dataset_access",
        "repository_clean_at_start",
        "repository_clean_at_end",
        "repository_commit_stable",
        "source_hashes_stable",
        "task_id_only_ranking_pass",
        "only_stage_b_content_retained_canonicalized_and_tokenized",
    )
    if any(integrity.get(field) is not True for field in required_true):
        raise ValueError("Stage-B identity integrity prerequisites did not all pass")
    if (
        integrity.get(
            "protected_window_8_16_content_retained_canonicalized_or_tokenized"
        )
        is not False
        or integrity.get("model_weights_loaded") is not False
        or integrity.get("model_forward_pass_run") is not False
        or integrity.get("logits_or_quality_metrics_observed") is not False
    ):
        raise ValueError("Stage-B identity consumed forbidden data, weights, or metrics")
    repository = _mapping(identity.get("repository"), context="identity repository")
    identity_commit = _require_ancestor(
        repository.get("commit"),
        repository_root=repository_root,
        context="Stage-B identity repository",
    )
    return identity, {
        **tracked,
        "artifact_sha256": file_sha256,
        "canonical_evidence_sha256": _sha256_bytes(canonical_json_bytes(identity)),
        "resolver_commit": identity_commit,
    }, tasks


def authenticate_identity_source_freeze(
    identity: Mapping[str, Any],
    *,
    current_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Require the identity amendment to freeze the exact Stage-B runtime bytes."""

    source_files = _mapping(
        identity.get("source_files"),
        context="identity frozen source files",
    )
    expected_paths = list(SOURCE_FILES)
    if source_files.get("paths") != expected_paths:
        raise ValueError("identity did not freeze the exact Stage-B source path set")
    start = _mapping(
        source_files.get("sha256_start"),
        context="identity source hashes start",
    )
    end = _mapping(
        source_files.get("sha256_end"),
        context="identity source hashes end",
    )
    current = dict(current_source_hashes)
    if (
        source_files.get("stable") is not True
        or dict(start) != dict(end)
        or dict(start) != current
        or set(start) != set(expected_paths)
    ):
        raise ValueError(
            "current Stage-B evaluator/method/test bytes differ from the identity freeze"
        )
    return {
        "passed": True,
        "path_count": len(expected_paths),
        "paths": expected_paths,
        "sha256": current,
    }


def authenticate_stage_b_runtime_environment(
    stage_a: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> dict[str, Any]:
    """Match the live runtime to Stage A and the committed identity before data."""

    authenticated = identity_resolver.authenticate_runtime_environment(
        stage_a,
        local_files_only=local_files_only,
    )
    expected = _mapping(
        identity.get("environment"),
        context="Stage-B identity runtime environment",
    )
    if authenticated != dict(expected):
        raise ValueError(
            "authenticated runtime environment differs from the Stage-B identity"
        )
    return dict(authenticated)


def plan_from_identity(
    identity: Mapping[str, Any],
    *,
    stage_a: Mapping[str, Any],
) -> tuple[ExactBudgetRowPlan, dict[str, Any]]:
    """Reconstruct the sole Stage-B row plan from committed identity bytes."""

    raw_plan = dict(_mapping(identity.get("row_plan"), context="identity row plan"))
    recorded_hash = raw_plan.pop("canonical_plan_sha256", None)
    if recorded_hash != _sha256_bytes(canonical_json_bytes(raw_plan)):
        raise ValueError("identity row-plan canonical hash does not match")
    required_fields = {
        "schema",
        "method",
        "selector_binding",
        "model",
        "quantization",
        "accounting",
        "score_shapes",
        "layer_quotas",
        "high_precision_rows",
    }
    if set(raw_plan) != required_fields:
        raise ValueError("identity row-plan fields do not match the frozen schema")
    if (
        raw_plan["schema"] != "recurquant.experiment009-stage-b-row-plan.v1"
        or raw_plan["method"] != TARGET_FISHER_SCORE
    ):
        raise ValueError("identity row-plan schema or method drifted")

    stage_a_selectors = _mapping(
        stage_a.get("selector_artifacts"),
        context="Stage-A selector binding",
    )
    expected_selector_binding = {
        "selector_file_sha256": stage_a_selectors["selector_file_sha256"],
        "selector_canonical_evidence_sha256": stage_a_selectors[
            "selector_canonical_evidence_sha256"
        ],
        "loss_selector_file_sha256": stage_a_selectors[
            "loss_selector_file_sha256"
        ],
        "loss_selector_canonical_evidence_sha256": stage_a_selectors[
            "loss_selector_canonical_evidence_sha256"
        ],
    }
    if dict(_mapping(raw_plan["selector_binding"], context="row-plan selectors")) != (
        expected_selector_binding
    ):
        raise ValueError("identity row plan is not bound to the exact Stage-A selectors")
    model = _mapping(raw_plan["model"], context="row-plan model")
    identity_model = _mapping(identity.get("model_contract"), context="identity model")
    if dict(model) != {
        "id": identity_model["id"],
        "revision": identity_model["revision"],
    }:
        raise ValueError("identity row-plan model contract drifted")
    quantization = _mapping(raw_plan["quantization"], context="row-plan quantization")
    if dict(quantization) != {
        "low_bits": 4,
        "high_bits": 8,
        "group_size": 128,
        "scale_bits": 16,
    }:
        raise ValueError("identity row-plan quantization contract drifted")
    accounting = _mapping(raw_plan["accounting"], context="row-plan accounting")
    expected_accounting = {
        "total_groups": 36_864,
        "mask_bytes": TARGET_MASK_BYTES,
        "promotion_increment_bytes": 64,
        "target_resident_bytes": TARGET_PACKED_STATE_BYTES,
        "resident_bytes": TARGET_PACKED_STATE_BYTES,
        "promoted_group_count": TARGET_PROMOTED_ROWS,
    }
    if dict(accounting) != expected_accounting:
        raise ValueError("identity row-plan byte accounting drifted")
    expected_shapes = [
        {"layer_index": layer, "heads": 16, "rows": 128}
        for layer in FROZEN_LINEAR_LAYERS
    ]
    if raw_plan["score_shapes"] != expected_shapes:
        raise ValueError("identity row-plan geometry drifted")
    expected_quotas = {
        str(layer): quota for layer, quota in FROZEN_LAYER_QUOTAS.items()
    }
    if raw_plan["layer_quotas"] != expected_quotas:
        raise ValueError("identity row-plan layer quotas drifted")

    raw_locations = raw_plan["high_precision_rows"]
    if (
        isinstance(raw_locations, (str, bytes))
        or not isinstance(raw_locations, Sequence)
        or len(raw_locations) != TARGET_PROMOTED_ROWS
    ):
        raise ValueError("identity row plan must contain exactly 1,976 promoted rows")
    locations: list[RowLocation] = []
    seen: set[tuple[int, int, int]] = set()
    counts = {layer: 0 for layer in FROZEN_LINEAR_LAYERS}
    for index, raw_location in enumerate(raw_locations):
        location = _mapping(raw_location, context=f"row-plan location {index}")
        if set(location) != {"layer_index", "head_index", "row_index"}:
            raise ValueError(f"row-plan location {index} fields drifted")
        layer = _strict_int(location["layer_index"], context="plan layer")
        head = _strict_int(location["head_index"], context="plan head")
        row = _strict_int(location["row_index"], context="plan row")
        key = (layer, head, row)
        if layer not in counts or not 0 <= head < 16 or not 0 <= row < 128:
            raise ValueError(f"row-plan location {index} is outside frozen geometry")
        if key in seen:
            raise ValueError(f"row-plan location {index} is duplicated")
        seen.add(key)
        counts[layer] += 1
        locations.append(RowLocation(layer, head, row))
    if counts != FROZEN_LAYER_QUOTAS:
        raise ValueError("identity promoted rows do not realize the frozen quotas")

    plan = ExactBudgetRowPlan(
        low_bits=4,
        high_bits=8,
        group_size=128,
        scale_bits=16,
        total_groups=36_864,
        mask_bytes=TARGET_MASK_BYTES,
        promotion_increment_bytes=64,
        target_resident_bytes=TARGET_PACKED_STATE_BYTES,
        resident_bytes=TARGET_PACKED_STATE_BYTES,
        high_precision_rows=tuple(locations),
        score_shapes=tuple((layer, 16, 128) for layer in FROZEN_LINEAR_LAYERS),
    )
    if (
        plan.promoted_group_count != TARGET_PROMOTED_ROWS
        or {
            layer: len(plan.groups_for_layer(layer))
            for layer in FROZEN_LINEAR_LAYERS
        }
        != FROZEN_LAYER_QUOTAS
    ):
        raise ValueError("reconstructed identity row plan failed its exact quota contract")
    return plan, {
        "passed": True,
        "canonical_plan_sha256": recorded_hash,
        "method": TARGET_FISHER_SCORE,
        "promoted_group_count": TARGET_PROMOTED_ROWS,
        "resident_bytes": TARGET_PACKED_STATE_BYTES,
        "selector_binding": expected_selector_binding,
    }


def _normalize_metric_rows(
    per_task: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    task_records: Sequence[Mapping[str, Any]],
    token_field: str,
    context: str,
) -> dict[str, list[dict[str, float | int | bool]]]:
    if set(per_task) != set(METHODS):
        raise ValueError(f"{context} metrics must contain exactly the four frozen methods")
    expected_ids = [int(record["task_id"]) for record in task_records]
    normalized: dict[str, list[dict[str, float | int | bool]]] = {}
    for method in METHODS:
        rows = per_task[method]
        if len(rows) != STAGE_B_LIMIT:
            raise ValueError(f"{context} {method} must contain exactly 32 task rows")
        method_rows: list[dict[str, float | int | bool]] = []
        for index, (row, identity) in enumerate(zip(rows, task_records, strict=True)):
            task_id = _strict_int(
                row.get("task_id"),
                context=f"{context} {method} row {index} task_id",
            )
            if task_id != expected_ids[index]:
                raise ValueError(f"{context} {method} task order drifted")
            token_count = _strict_int(
                row.get("token_count"),
                context=f"{context} {method} task {task_id} token_count",
            )
            if token_count != int(identity[token_field]):
                raise ValueError(f"{context} {method} task {task_id} token count drifted")
            if row.get("all_logits_finite") is not True:
                raise ValueError(f"{context} {method} task {task_id} logits are not finite")
            values = {
                field: _finite_float(
                    row.get(field),
                    context=f"{context} {method} task {task_id} {field}",
                )
                for field in METRIC_FIELDS
            }
            if (
                values["mean_kl"] < 0
                or values["cvar95_kl"] < values["mean_kl"]
                or values["max_kl"] < values["cvar95_kl"]
            ):
                raise ValueError(
                    f"{context} {method} task {task_id} KL metrics are invalid or unordered"
                )
            if not 0 <= values["top1_agreement"] <= 1:
                raise ValueError(
                    f"{context} {method} task {task_id} top-1 agreement is outside [0, 1]"
                )
            if values["reference_nll"] < 0 or values["candidate_nll"] < 0:
                raise ValueError(
                    f"{context} {method} task {task_id} NLL metrics must be non-negative"
                )
            arithmetic_delta = values["candidate_nll"] - values["reference_nll"]
            if not math.isclose(
                values["delta_nll"],
                arithmetic_delta,
                rel_tol=1e-7,
                abs_tol=2e-6,
            ):
                raise ValueError(
                    f"{context} {method} task {task_id} delta_nll is not candidate-reference"
                )
            method_rows.append(
                {
                    "task_id": task_id,
                    "token_count": token_count,
                    "all_logits_finite": True,
                    **values,
                }
            )
        normalized[method] = method_rows
    for task_index, task_id in enumerate(expected_ids):
        reference_values = [
            float(normalized[method][task_index]["reference_nll"])
            for method in METHODS
        ]
        if any(
            not math.isclose(
                value,
                reference_values[0],
                rel_tol=0,
                abs_tol=1e-7,
            )
            for value in reference_values[1:]
        ):
            raise ValueError(
                f"{context} task {task_id} methods do not share one reference NLL"
            )
    return normalized


def _expected_aggregates(
    normalized: Mapping[str, Sequence[Mapping[str, float | int | bool]]],
) -> dict[str, dict[str, float | int]]:
    return {
        method: {
            "task_count": len(rows),
            "macro_delta_nll": fmean(float(row["delta_nll"]) for row in rows),
            "macro_mean_kl": fmean(float(row["mean_kl"]) for row in rows),
            "macro_cvar95_kl": fmean(float(row["cvar95_kl"]) for row in rows),
            "maximum_kl": max(float(row["max_kl"]) for row in rows),
            "macro_top1_agreement": fmean(
                float(row["top1_agreement"]) for row in rows
            ),
            "token_count": sum(int(row["token_count"]) for row in rows),
        }
        for method, rows in normalized.items()
    }


def _accumulator_trace(accumulator: Any) -> dict[str, Any]:
    """Retain compact per-token primitives so every task summary is reproducible."""

    fields = {
        "kl": torch.cat(accumulator.kl).to(torch.float32),
        "reference_nll": torch.cat(accumulator.reference_nll).to(torch.float32),
        "candidate_nll": torch.cat(accumulator.candidate_nll).to(torch.float32),
        "top1_agreement": torch.cat(accumulator.top1_agreement).to(torch.bool),
        "outputs_finite": torch.stack(accumulator.outputs_finite).to(torch.bool),
    }
    token_count = int(fields["kl"].numel())
    if token_count <= 0 or any(int(values.numel()) != token_count for values in fields.values()):
        raise RuntimeError("token primitive accumulators have inconsistent lengths")
    payload = {
        "token_count": token_count,
        **{
            name: (
                values.tolist()
                if name not in {"top1_agreement", "outputs_finite"}
                else [bool(value) for value in values.tolist()]
            )
            for name, values in fields.items()
        },
    }
    return {
        **payload,
        "canonical_primitives_sha256": _sha256_bytes(canonical_json_bytes(payload)),
    }


def _summary_from_trace(trace: Mapping[str, Any]) -> dict[str, float | int | bool]:
    token_count = _strict_int(trace.get("token_count"), context="trace token_count")
    if token_count <= 0:
        raise ValueError("trace token_count must be positive")
    numeric: dict[str, torch.Tensor] = {}
    for field in ("kl", "reference_nll", "candidate_nll"):
        raw = trace.get(field)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError(f"trace {field} must be an array")
        values = torch.tensor(
            [
                _finite_float(value, context=f"trace {field} value")
                for value in raw
            ],
            dtype=torch.float32,
        )
        if int(values.numel()) != token_count:
            raise ValueError(f"trace {field} length differs from token_count")
        numeric[field] = values
    boolean: dict[str, torch.Tensor] = {}
    for field in ("top1_agreement", "outputs_finite"):
        raw = trace.get(field)
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != token_count
            or any(not isinstance(value, bool) for value in raw)
        ):
            raise ValueError(f"trace {field} must be a token-aligned boolean array")
        boolean[field] = torch.tensor(raw, dtype=torch.bool)
    payload = {
        "token_count": token_count,
        **{
            field: list(trace[field])
            for field in (
                "kl",
                "reference_nll",
                "candidate_nll",
                "top1_agreement",
                "outputs_finite",
            )
        },
    }
    if trace.get("canonical_primitives_sha256") != _sha256_bytes(
        canonical_json_bytes(payload)
    ):
        raise ValueError("trace canonical primitive hash does not match")

    kl = numeric["kl"]
    reference = numeric["reference_nll"]
    candidate = numeric["candidate_nll"]
    tail_count = max(1, math.ceil(token_count * 0.05))
    reference_mean = reference.mean()
    candidate_mean = candidate.mean()
    return {
        "token_count": token_count,
        "mean_kl": float(kl.mean().item()),
        "cvar95_kl": float(torch.topk(kl, k=tail_count).values.mean().item()),
        "max_kl": float(kl.max().item()),
        "top1_agreement": float(
            boolean["top1_agreement"].to(torch.float32).mean().item()
        ),
        "reference_nll": float(reference_mean.item()),
        "candidate_nll": float(candidate_mean.item()),
        "delta_nll": float((candidate_mean - reference_mean).item()),
        "all_logits_finite": bool(boolean["outputs_finite"].all().item()),
    }


def audit_metric_traces(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    task_records: Sequence[Mapping[str, Any]],
    token_field: str,
    context: str,
) -> dict[str, Any]:
    """Recompute every task metric from stored token primitives and hashes."""

    if set(traces) != set(METHODS):
        raise ValueError(f"{context} traces must contain the four frozen methods")
    for method in METHODS:
        if len(traces[method]) != STAGE_B_LIMIT:
            raise ValueError(f"{context} {method} must contain exactly 32 traces")
    for task_index, identity in enumerate(task_records):
        task_id = int(identity["task_id"])
        expected_tokens = int(identity[token_field])
        reference_trace: Sequence[Any] | None = None
        for method in METHODS:
            trace = traces[method][task_index]
            if int(trace.get("task_id", -1)) != task_id:
                raise ValueError(f"{context} {method} trace task order drifted")
            recomputed = _summary_from_trace(trace)
            if recomputed["token_count"] != expected_tokens:
                raise ValueError(f"{context} {method} task {task_id} trace count drifted")
            expected_summary = {
                key: value
                for key, value in summaries[method][task_index].items()
                if key != "task_id"
            }
            if recomputed != expected_summary:
                raise ValueError(
                    f"{context} {method} task {task_id} summary differs from token primitives"
                )
            method_reference = trace.get("reference_nll")
            if reference_trace is None:
                reference_trace = method_reference
            elif method_reference != reference_trace:
                raise ValueError(
                    f"{context} task {task_id} methods do not share reference primitives"
                )
    return {
        "passed": True,
        "task_count": STAGE_B_LIMIT,
        "methods": list(METHODS),
        "token_primitive_hashes_verified": True,
        "summaries_recomputed": True,
    }


def _audit_aggregate_match(
    aggregates: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    if set(aggregates) != set(METHODS):
        raise ValueError(f"{context} aggregates must contain the four frozen methods")
    for method in METHODS:
        if dict(aggregates[method]) != dict(expected[method]):
            raise ValueError(f"{context} {method} aggregate differs from task-macro recomputation")
    return {
        "passed": True,
        "methods": list(METHODS),
        "task_count_per_method": STAGE_B_LIMIT,
    }


def audit_storage(storage: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(storage) != set(METHODS):
        raise ValueError("storage must contain exactly the four frozen methods")
    common = {
        "payload_bytes": TARGET_PAYLOAD_BYTES,
        "scale_bytes": TARGET_SCALE_BYTES,
        "mask_bytes": TARGET_MASK_BYTES,
        "resident_bytes": TARGET_PACKED_STATE_BYTES,
        "high_precision_groups": TARGET_PROMOTED_ROWS,
    }
    for method in METHODS:
        for field, expected in common.items():
            if _strict_int(
                storage[method].get(field),
                context=f"{method} storage {field}",
            ) != expected:
                raise ValueError(f"{method} storage {field} drifted")
    for method in QUERY_METHODS:
        if (
            _strict_int(
                storage[method].get("selector_auxiliary_bytes"),
                context=f"{method} selector auxiliary bytes",
            )
            != TARGET_SELECTOR_BYTES
            or _strict_int(
                storage[method].get("resident_bytes_including_selector"),
                context=f"{method} selector-aware resident bytes",
            )
            != TARGET_TOTAL_BYTES
        ):
            raise ValueError(f"{method} selector-aware storage drifted")
    return {
        "passed": True,
        **common,
        "query_selector_auxiliary_bytes": TARGET_SELECTOR_BYTES,
        "query_resident_bytes_including_selector": TARGET_TOTAL_BYTES,
    }


def audit_task_selector_diagnostics(
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    task_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(diagnostics) != set(QUERY_METHODS):
        raise ValueError("selector diagnostics must contain exactly CQER and RHT-CQER")
    methods: dict[str, list[dict[str, Any]]] = {}
    expected_selection = {
        CQER_METHOD: "query_ema32_weighted_aligned_mse_reduction",
        RHT_METHOD: RHT_METHOD,
    }
    for method in QUERY_METHODS:
        task_diagnostics = diagnostics[method]
        if len(task_diagnostics) != STAGE_B_LIMIT:
            raise ValueError(f"{method} must report diagnostics for exactly 32 tasks")
        audited_tasks: list[dict[str, Any]] = []
        for identity, task_record in zip(
            task_records,
            task_diagnostics,
            strict=True,
        ):
            task_id = _strict_int(task_record.get("task_id"), context="diagnostic task_id")
            if task_id != int(identity["task_id"]):
                raise ValueError(f"{method} diagnostic task order drifted")
            layers = task_record.get("layers")
            if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
                raise ValueError(f"{method} task {task_id} layers must be an array")
            if len(layers) != len(FROZEN_LINEAR_LAYERS):
                raise ValueError(f"{method} task {task_id} did not report every layer")
            by_layer: dict[int, Mapping[str, Any]] = {}
            for raw_layer in layers:
                layer_record = _mapping(
                    raw_layer,
                    context=f"{method} task {task_id} layer",
                )
                layer = _strict_int(
                    layer_record.get("layer_index"),
                    context=f"{method} task {task_id} layer_index",
                )
                if layer in by_layer:
                    raise ValueError(f"{method} task {task_id} repeats layer {layer}")
                by_layer[layer] = layer_record
            if set(by_layer) != set(FROZEN_LINEAR_LAYERS):
                raise ValueError(f"{method} task {task_id} layer identity drifted")

            expected_writes = int(identity["code_tokens"])
            expected_query_tokens = (
                int(identity["prompt_tokens"])
                + int(identity["aligned_scored_tokens"])
            )
            for layer in FROZEN_LINEAR_LAYERS:
                record = by_layer[layer]
                quota = _strict_int(
                    record.get("quota"),
                    context=f"{method} task {task_id} layer {layer} quota",
                )
                selected = _strict_int(
                    record.get("current_selected_count"),
                    context=f"{method} task {task_id} layer {layer} selection count",
                )
                updates = _strict_int(
                    record.get("state_updates"),
                    context=f"{method} task {task_id} layer {layer} updates",
                )
                staged = _strict_int(
                    record.get("observations_staged"),
                    context=f"{method} task {task_id} layer {layer} staged",
                )
                committed = _strict_int(
                    record.get("observations_committed"),
                    context=f"{method} task {task_id} layer {layer} committed",
                )
                tokens = _strict_int(
                    record.get("tokens_observed"),
                    context=f"{method} task {task_id} layer {layer} tokens",
                )
                if quota != FROZEN_LAYER_QUOTAS[layer] or selected != quota:
                    raise ValueError(f"{method} task {task_id} layer {layer} quota drifted")
                if (updates, staged, committed) != (
                    expected_writes,
                    expected_writes,
                    expected_writes,
                ):
                    raise ValueError(
                        f"{method} task {task_id} layer {layer} handshake drifted"
                    )
                if tokens != expected_query_tokens:
                    raise ValueError(
                        f"{method} task {task_id} layer {layer} query-token count drifted"
                    )
                if (
                    record.get("pending_observation") is not False
                    or record.get("confirmation_two") is not False
                    or record.get("selection_method") != expected_selection[method]
                ):
                    raise ValueError(
                        f"{method} task {task_id} layer {layer} method/handshake drifted"
                    )
                mask = record.get("current_mask_sha256")
                if (
                    not isinstance(mask, str)
                    or len(mask) != 64
                    or any(character not in "0123456789abcdef" for character in mask)
                ):
                    raise ValueError(
                        f"{method} task {task_id} layer {layer} mask hash is invalid"
                    )
                if method == RHT_METHOD and (
                    record.get("state_codec") != "right_rht_sha256_signs_v1"
                    or record.get("state_codec_seed") != SEED
                    or record.get("state_codec_axis") != "value"
                    or record.get("state_codec_normalization") != "orthonormal"
                    or record.get("state_codec_persistent_tensor_bytes") != 0
                ):
                    raise ValueError(
                        f"{method} task {task_id} layer {layer} codec contract drifted"
                    )
            audited_tasks.append(
                {
                    "task_id": task_id,
                    "state_writes": expected_writes,
                    "query_tokens": expected_query_tokens,
                    "layers": len(FROZEN_LINEAR_LAYERS),
                }
            )
        methods[method] = audited_tasks
    return {
        "passed": True,
        "quota_sum": sum(FROZEN_LAYER_QUOTAS.values()),
        "task_count": STAGE_B_LIMIT,
        "methods": methods,
    }


def validate_task_state_error_coverage(
    state_errors: Mapping[str, Mapping[str, Any]],
    *,
    expected_writes: int,
) -> dict[str, Any]:
    if set(state_errors) != set(QUERY_METHODS):
        raise ValueError("state-error evidence must contain exactly CQER and RHT-CQER")
    if state_errors[CQER_METHOD].get("coverage") != state_errors[RHT_METHOD].get(
        "coverage"
    ):
        raise ValueError("CQER and RHT-CQER state-error coverage differs")
    expected_records = len(FROZEN_LINEAR_LAYERS) * expected_writes
    for method in QUERY_METHODS:
        record = state_errors[method]
        if int(record.get("record_count", -1)) != expected_records:
            raise ValueError(f"{method} state-error record count drifted")
        per_layer = record.get("per_layer")
        per_write = record.get("per_write")
        if not isinstance(per_layer, Mapping) or set(per_layer) != {
            str(layer) for layer in FROZEN_LINEAR_LAYERS
        }:
            raise ValueError(f"{method} state-error layers drifted")
        if not isinstance(per_write, Mapping) or set(per_write) != {
            str(write) for write in range(expected_writes)
        }:
            raise ValueError(f"{method} state-error writes drifted")
        if any(
            int(per_layer[str(layer)].get("record_count", -1)) != expected_writes
            for layer in FROZEN_LINEAR_LAYERS
        ):
            raise ValueError(f"{method} did not record every layer on every write")
        if any(
            int(per_write[str(write)].get("record_count", -1))
            != len(FROZEN_LINEAR_LAYERS)
            for write in range(expected_writes)
        ):
            raise ValueError(f"{method} did not record every layer within each write")
    return {
        "passed": True,
        "state_writes": expected_writes,
        "records_per_method": expected_records,
    }


def recompute_local_codec_state_error_summary(
    summary: Mapping[str, Any],
    *,
    method: str,
    expected_writes: int,
    final_mask_hashes: Mapping[int, str],
) -> dict[str, Any]:
    """Recompute local pre-pack-to-materialized QDQ SSE from every raw record."""

    raw_records = summary.get("records")
    if (
        isinstance(raw_records, (str, bytes))
        or not isinstance(raw_records, Sequence)
        or len(raw_records) != len(FROZEN_LINEAR_LAYERS) * expected_writes
    ):
        raise ValueError(f"{method} raw local-codec records have the wrong length")
    expected_selection = (
        "query_ema32_weighted_aligned_mse_reduction"
        if method == CQER_METHOD
        else RHT_METHOD
    )
    normalized: list[dict[str, Any]] = []
    per_layer: dict[int, dict[str, float | int]] = {
        layer: {"record_count": 0, "element_count": 0, "state_sse": 0.0}
        for layer in FROZEN_LINEAR_LAYERS
    }
    per_write: dict[int, dict[str, float | int]] = {
        write: {"record_count": 0, "element_count": 0, "state_sse": 0.0}
        for write in range(expected_writes)
    }
    expected_shape = (1, 16, 128, 128)
    element_count = math.prod(expected_shape)
    for update_index, raw_record in enumerate(raw_records):
        record = dict(_mapping(raw_record, context=f"{method} raw state-error record"))
        write = update_index // len(FROZEN_LINEAR_LAYERS)
        layer = FROZEN_LINEAR_LAYERS[update_index % len(FROZEN_LINEAR_LAYERS)]
        if (
            _strict_int(record.get("update_index"), context="state update_index")
            != update_index
            or _strict_int(record.get("write_ordinal"), context="state write_ordinal")
            != write
            or _strict_int(record.get("layer_index"), context="state layer_index")
            != layer
            or _strict_int(record.get("state_index"), context="state state_index") != 0
        ):
            raise ValueError(f"{method} raw state-error write/layer order drifted")
        shape = record.get("shape")
        if list(shape) != list(expected_shape):
            raise ValueError(f"{method} layer {layer} recurrent-state geometry drifted")
        quota = FROZEN_LAYER_QUOTAS[layer]
        expected_payload = 2_048 * 64 + quota * 64
        expected_scale = 2_048 * 2
        expected_mask = 2_048 // 8
        expected_resident = expected_payload + expected_scale + expected_mask
        integer_contract = {
            "element_count": element_count,
            "low_bits": 4,
            "high_bits": 8,
            "group_size": 128,
            "scale_bits": 16,
            "total_groups": 2_048,
            "high_precision_groups": quota,
            "baseline_bytes": element_count * 4,
            "payload_bytes": expected_payload,
            "scale_bytes": expected_scale,
            "mask_bytes": expected_mask,
            "resident_bytes": expected_resident,
        }
        for field, expected in integer_contract.items():
            if _strict_int(
                record.get(field),
                context=f"{method} layer {layer} {field}",
            ) != expected:
                raise ValueError(f"{method} layer {layer} state-error {field} drifted")
        if (
            record.get("rounding") != "nearest"
            or record.get("source_dtype") != "torch.float32"
            or record.get("selection_method") != expected_selection
        ):
            raise ValueError(f"{method} layer {layer} state-error codec contract drifted")
        mask_hash = record.get("high_precision_mask_sha256")
        if (
            not isinstance(mask_hash, str)
            or len(mask_hash) != 64
            or any(character not in "0123456789abcdef" for character in mask_hash)
        ):
            raise ValueError(f"{method} layer {layer} state-error mask hash is invalid")
        mse = _finite_float(record.get("mean_squared_error"), context="state MSE")
        maximum = _finite_float(record.get("max_absolute_error"), context="state max error")
        relative_l2 = _finite_float(record.get("relative_l2_error"), context="state relative L2")
        recorded_sse = _finite_float(record.get("state_sse"), context="state SSE")
        if min(mse, maximum, relative_l2, recorded_sse) < 0:
            raise ValueError(f"{method} raw state-error magnitudes must be non-negative")
        recomputed_sse = mse * element_count
        if recorded_sse != recomputed_sse:
            raise ValueError(f"{method} raw state-error SSE is not MSE times elements")
        normalized.append(record)
        for totals in (per_layer[layer], per_write[write]):
            totals["record_count"] = int(totals["record_count"]) + 1
            totals["element_count"] = int(totals["element_count"]) + element_count
            totals["state_sse"] = float(totals["state_sse"]) + recomputed_sse

    for layer in FROZEN_LINEAR_LAYERS:
        final_record = normalized[
            (expected_writes - 1) * len(FROZEN_LINEAR_LAYERS)
            + FROZEN_LINEAR_LAYERS.index(layer)
        ]
        if final_record["high_precision_mask_sha256"] != final_mask_hashes.get(layer):
            raise ValueError(
                f"{method} layer {layer} final raw-record mask does not match diagnostics"
            )
    coverage = [
        {
            "write_ordinal": int(record["write_ordinal"]),
            "layer_index": int(record["layer_index"]),
            "state_index": int(record["state_index"]),
            "shape": list(record["shape"]),
        }
        for record in normalized
    ]
    total_elements = len(normalized) * element_count
    total_sse = math.fsum(float(record["state_sse"]) for record in normalized)
    recomputed = {
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
            for layer, values in per_layer.items()
        },
        "per_write": {
            str(write): {
                "record_count": int(values["record_count"]),
                "element_count": int(values["element_count"]),
                "state_sse": float(values["state_sse"]),
            }
            for write, values in per_write.items()
        },
        "records": normalized,
    }
    if dict(summary) != recomputed:
        raise ValueError(f"{method} state-error aggregates differ from raw recomputation")
    return recomputed


def audit_state_errors(
    per_task_state_errors: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    task_records: Sequence[Mapping[str, Any]],
    selector_diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(per_task_state_errors) != set(QUERY_METHODS):
        raise ValueError("per-task state errors must contain CQER and RHT-CQER")
    audits: list[dict[str, Any]] = []
    for index, identity in enumerate(task_records):
        task_id = int(identity["task_id"])
        task_pair: dict[str, Mapping[str, Any]] = {}
        for method in QUERY_METHODS:
            records = per_task_state_errors[method]
            if len(records) != STAGE_B_LIMIT:
                raise ValueError(f"{method} must contain state errors for exactly 32 tasks")
            task_record = records[index]
            if int(task_record.get("task_id", -1)) != task_id:
                raise ValueError(f"{method} state-error task order drifted")
            summary = _mapping(
                task_record.get("state_error"),
                context=f"{method} task {task_id} state error",
            )
            diagnostics = selector_diagnostics[method][index]
            layers = diagnostics.get("layers")
            if not isinstance(layers, Sequence):
                raise ValueError(f"{method} task {task_id} diagnostics are missing")
            final_masks = {
                int(layer["layer_index"]): str(layer["current_mask_sha256"])
                for layer in layers
            }
            task_pair[method] = recompute_local_codec_state_error_summary(
                summary,
                method=method,
                expected_writes=int(identity["code_tokens"]),
                final_mask_hashes=final_masks,
            )
        coverage = validate_task_state_error_coverage(
            task_pair,
            expected_writes=int(identity["code_tokens"]),
        )
        audits.append({"task_id": task_id, **coverage})
    return {"passed": True, "task_count": STAGE_B_LIMIT, "tasks": audits}


def aggregate_state_errors(
    per_task_state_errors: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for method in QUERY_METHODS:
        records = per_task_state_errors[method]
        element_count = sum(
            int(_mapping(record["state_error"], context="state error").get("element_count", -1))
            for record in records
        )
        state_sse = math.fsum(
            _finite_float(
                _mapping(record["state_error"], context="state error").get(
                    "aggregate_state_sse"
                ),
                context=f"{method} aggregate state SSE",
            )
            for record in records
        )
        record_count = sum(
            int(_mapping(record["state_error"], context="state error").get("record_count", -1))
            for record in records
        )
        if element_count <= 0 or record_count <= 0 or state_sse < 0:
            raise ValueError(f"{method} aggregate state-error evidence is invalid")
        aggregates[method] = {
            "task_count": len(records),
            "record_count": record_count,
            "element_count": element_count,
            "aggregate_state_sse": state_sse,
            "aggregate_state_mse": state_sse / element_count,
        }
    return aggregates


def audit_reference_aligned_state_errors(
    per_task_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    task_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate the FP64 candidate-versus-matched-FP32 secondary diagnostic."""

    if set(per_task_records) != set(QUERY_METHODS):
        raise ValueError("reference-aligned state errors must contain CQER and RHT-CQER")
    expected_shape = [1, 16, 128, 128]
    expected_elements = math.prod(expected_shape)
    for method in QUERY_METHODS:
        if len(per_task_records[method]) != STAGE_B_LIMIT:
            raise ValueError(f"{method} reference-aligned evidence must contain 32 tasks")
        for identity, task_wrapper in zip(
            task_records,
            per_task_records[method],
            strict=True,
        ):
            task_id = int(identity["task_id"])
            if int(task_wrapper.get("task_id", -1)) != task_id:
                raise ValueError(f"{method} reference-aligned task order drifted")
            summary = _mapping(
                task_wrapper.get("state_error"),
                context=f"{method} task {task_id} reference-aligned summary",
            )
            writes = summary.get("writes")
            expected_writes = int(identity["code_tokens"])
            if (
                summary.get("metric")
                != "candidate_materialized_state_minus_matched_fp32_state"
                or not isinstance(writes, Sequence)
                or len(writes) != expected_writes
            ):
                raise ValueError(f"{method} task {task_id} reference-aligned contract drifted")
            total_elements = 0
            total_sse_values: list[float] = []
            for write_ordinal, write in enumerate(writes):
                record = _mapping(
                    write,
                    context=f"{method} task {task_id} reference-aligned write",
                )
                layers = record.get("layers")
                if (
                    int(record.get("write_ordinal", -1)) != write_ordinal
                    or not isinstance(layers, Sequence)
                    or len(layers) != len(FROZEN_LINEAR_LAYERS)
                ):
                    raise ValueError(
                        f"{method} task {task_id} reference-aligned write coverage drifted"
                    )
                write_sse_values: list[float] = []
                for expected_layer, layer_record in zip(
                    FROZEN_LINEAR_LAYERS,
                    layers,
                    strict=True,
                ):
                    layer = _mapping(
                        layer_record,
                        context=f"{method} reference-aligned layer",
                    )
                    if (
                        int(layer.get("layer_index", -1)) != expected_layer
                        or int(layer.get("state_index", -1)) != 0
                        or layer.get("shape") != expected_shape
                        or int(layer.get("element_count", -1)) != expected_elements
                    ):
                        raise ValueError(
                            f"{method} task {task_id} reference-aligned geometry drifted"
                        )
                    sse = _finite_float(layer.get("state_sse"), context="reference SSE")
                    mse = _finite_float(layer.get("state_mse"), context="reference MSE")
                    relative = _finite_float(
                        layer.get("relative_l2_error"),
                        context="reference relative L2",
                    )
                    maximum = _finite_float(
                        layer.get("max_absolute_error"),
                        context="reference maximum error",
                    )
                    if min(sse, mse, relative, maximum) < 0 or mse != sse / expected_elements:
                        raise ValueError(
                            f"{method} task {task_id} reference-aligned metrics drifted"
                        )
                    write_sse_values.append(sse)
                write_sse = math.fsum(write_sse_values)
                write_elements = len(FROZEN_LINEAR_LAYERS) * expected_elements
                if (
                    int(record.get("record_count", -1)) != len(FROZEN_LINEAR_LAYERS)
                    or int(record.get("element_count", -1)) != write_elements
                    or _finite_float(
                        record.get("state_sse"),
                        context="reference write SSE",
                    )
                    != write_sse
                ):
                    raise ValueError(
                        f"{method} task {task_id} reference-aligned write totals drifted"
                    )
                total_elements += write_elements
                total_sse_values.append(write_sse)
            total_sse = math.fsum(total_sse_values)
            if (
                int(summary.get("write_count", -1)) != expected_writes
                or int(summary.get("record_count", -1))
                != expected_writes * len(FROZEN_LINEAR_LAYERS)
                or int(summary.get("element_count", -1)) != total_elements
                or _finite_float(
                    summary.get("aggregate_state_sse"),
                    context="reference aggregate SSE",
                )
                != total_sse
                or _finite_float(
                    summary.get("aggregate_state_mse"),
                    context="reference aggregate MSE",
                )
                != total_sse / total_elements
            ):
                raise ValueError(
                    f"{method} task {task_id} reference-aligned aggregate drifted"
                )
    return {
        "passed": True,
        "task_count": STAGE_B_LIMIT,
        "methods": list(QUERY_METHODS),
        "accumulation_dtype": "torch.float64",
        "advancement_gate_metric": False,
    }


def _dense_reference_signs(layer: int, heads: int, width: int) -> np.ndarray:
    signs = np.empty((1, heads, 1, width), dtype=np.float32)
    domain = b"recurquant.right-rht.signs.v1\0"
    for head in range(heads):
        values: list[float] = []
        counter = 0
        while len(values) < width:
            message = b"".join(
                (
                    domain,
                    SEED.to_bytes(8, "little"),
                    layer.to_bytes(8, "little"),
                    head.to_bytes(8, "little"),
                    width.to_bytes(8, "little"),
                    counter.to_bytes(8, "little"),
                )
            )
            for byte in hashlib.sha256(message).digest():
                for bit in range(8):
                    values.append(1.0 if byte & (1 << bit) else -1.0)
                    if len(values) == width:
                        break
                if len(values) == width:
                    break
            counter += 1
        signs[0, head, 0] = np.asarray(values, dtype=np.float32)
    return signs


def _dense_hadamard(width: int) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < width:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if matrix.shape != (width, width):
        raise ValueError("dense Hadamard reference requires a power-of-two width")
    return matrix


def _dense_reference_encode(state: np.ndarray, *, layer: int) -> np.ndarray:
    width = state.shape[-1]
    signed = state.astype(np.float32) * _dense_reference_signs(
        layer,
        state.shape[1],
        width,
    )
    return np.matmul(signed, _dense_hadamard(width)) / math.sqrt(width)


def _dense_reference_decode(encoded: np.ndarray, *, layer: int) -> np.ndarray:
    width = encoded.shape[-1]
    decoded = np.matmul(encoded.astype(np.float32), _dense_hadamard(width))
    decoded /= math.sqrt(width)
    return decoded * _dense_reference_signs(layer, encoded.shape[1], width)


def _dense_reference_mixed_qdq(
    state: np.ndarray,
    mask: np.ndarray,
    *,
    layer: int,
) -> np.ndarray:
    encoded = _dense_reference_encode(state, layer=layer)
    groups = encoded.reshape(-1, 128)
    qmax = np.where(mask.reshape(-1), 127.0, 7.0).reshape(-1, 1)
    ideal_scales = np.max(np.abs(groups), axis=1, keepdims=True) / qmax
    ideal_scales = np.where(ideal_scales > 1e-8, ideal_scales, 1.0)
    scales = np.clip(
        ideal_scales,
        2.0**-24,
        np.finfo(np.float16).max,
    ).astype(np.float16).astype(np.float32)
    codes = np.rint(groups / scales)
    codes = np.maximum(np.minimum(codes, qmax), -qmax)
    quantized = (codes * scales).reshape(encoded.shape)
    return _dense_reference_decode(quantized, layer=layer)


def compute_independent_dense_rht_evidence() -> dict[str, Any]:
    """Compare production RHT/packing with dense NumPy reference implementations."""

    encode_state = torch.randn(
        (1, 2, 3, 128),
        generator=torch.Generator().manual_seed(811),
        dtype=torch.float32,
    )
    reference_signs = _dense_reference_signs(7, 2, 128)
    production_signs = right_rht_signs(
        layer_index=7,
        expected_heads=2,
        width=128,
        device="cpu",
    ).numpy()
    reference_encoded = _dense_reference_encode(encode_state.numpy(), layer=7)
    production_encoded = right_rht_encode(
        encode_state,
        layer_index=7,
        expected_heads=2,
    ).numpy()
    encode_max_abs = float(np.max(np.abs(production_encoded - reference_encoded)))

    pack_state = torch.randn(
        (1, 2, 4, 128),
        generator=torch.Generator().manual_seed(823),
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[False, True, False, True], [True, False, False, True]],
        dtype=torch.bool,
    )
    packed = quantize_pack_mixed(
        pack_state,
        mask,
        low_spec=QuantizationSpec(bits=4, group_size=128, flatten_last_dims=2),
        high_spec=QuantizationSpec(bits=8, group_size=128, flatten_last_dims=2),
        right_rht_layer_index=9,
        right_rht_expected_heads=2,
    )
    reference_packed = _dense_reference_mixed_qdq(
        pack_state.numpy(),
        mask.numpy(),
        layer=9,
    )
    pack_max_abs = float(
        np.max(np.abs(packed.dequantize().numpy() - reference_packed))
    )
    return {
        "reference": INDEPENDENT_REFERENCE_DESCRIPTION,
        "signs_exact": bool(np.array_equal(production_signs, reference_signs)),
        "encode_max_abs_difference": encode_max_abs,
        "encode_max_abs_threshold": INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD,
        "physical_pack_max_abs_difference": pack_max_abs,
        "physical_pack_max_abs_threshold": INDEPENDENT_PACK_MAX_ABS_THRESHOLD,
        "passed": (
            np.array_equal(production_signs, reference_signs)
            and encode_max_abs <= INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD
            and pack_max_abs <= INDEPENDENT_PACK_MAX_ABS_THRESHOLD
        ),
    }


def validate_unit_evidence_schema(
    unit_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact production schemas and recompute every derived unit flag."""

    _require_exact_fields(
        unit_evidence,
        {"production_self_check", "independent_dense_reference"},
        context="Stage-B unit evidence",
    )
    production = _mapping(
        unit_evidence.get("production_self_check"),
        context="production RHT self-check",
    )
    _require_exact_fields(
        production,
        {
            "inverse_relative_l2",
            "inverse_relative_l2_threshold",
            "physical_pack_matches_transformed_qdq",
            "physical_pack_max_abs_difference",
            "physical_pack_storage_bytes",
            "sign_schedule_sha256",
            "expected_sign_schedule_sha256",
            "seed",
            "device",
            "dtype",
        },
        context="production RHT self-check",
    )
    independent = _mapping(
        unit_evidence.get("independent_dense_reference"),
        context="independent dense RHT reference",
    )
    _require_exact_fields(
        independent,
        {
            "reference",
            "signs_exact",
            "encode_max_abs_difference",
            "encode_max_abs_threshold",
            "physical_pack_max_abs_difference",
            "physical_pack_max_abs_threshold",
            "passed",
        },
        context="independent dense RHT reference",
    )

    inverse = _finite_float(
        production.get("inverse_relative_l2"),
        context="right-RHT inverse relative L2",
    )
    pack_difference = _finite_float(
        production.get("physical_pack_max_abs_difference"),
        context="right-RHT physical-pack maximum difference",
    )
    if inverse < 0 or pack_difference < 0:
        raise ValueError("production RHT self-check errors must be non-negative")
    if (
        production.get("inverse_relative_l2_threshold")
        != MAX_RHT_INVERSE_RELATIVE_L2
        or production.get("physical_pack_storage_bytes")
        != PRODUCTION_SELF_CHECK_STORAGE_BYTES
        or production.get("expected_sign_schedule_sha256")
        != EXPECTED_SIGN_SCHEDULE_SHA256
        or production.get("seed") != SEED
        or production.get("device") != "cpu"
        or production.get("dtype") != "torch.float32"
    ):
        raise ValueError("production RHT self-check frozen contract drifted")
    _require_sha256(
        production.get("sign_schedule_sha256"),
        context="production RHT sign schedule",
    )
    _require_sha256(
        production.get("expected_sign_schedule_sha256"),
        context="expected RHT sign schedule",
    )
    production_passed = (
        inverse < MAX_RHT_INVERSE_RELATIVE_L2
        and production.get("physical_pack_matches_transformed_qdq") is True
        and pack_difference == 0.0
        and production.get("sign_schedule_sha256") == EXPECTED_SIGN_SCHEDULE_SHA256
    )

    signs_exact = independent.get("signs_exact")
    recorded_passed = independent.get("passed")
    if not isinstance(signs_exact, bool) or not isinstance(recorded_passed, bool):
        raise ValueError("independent dense RHT boolean fields drifted")
    encode_difference = _finite_float(
        independent.get("encode_max_abs_difference"),
        context="independent encode maximum difference",
    )
    independent_pack_difference = _finite_float(
        independent.get("physical_pack_max_abs_difference"),
        context="independent pack maximum difference",
    )
    if (
        encode_difference < 0
        or independent_pack_difference < 0
        or independent.get("reference") != INDEPENDENT_REFERENCE_DESCRIPTION
        or independent.get("encode_max_abs_threshold")
        != INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD
        or independent.get("physical_pack_max_abs_threshold")
        != INDEPENDENT_PACK_MAX_ABS_THRESHOLD
    ):
        raise ValueError("independent dense RHT reference contract drifted")
    independent_passed = (
        signs_exact
        and encode_difference <= INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD
        and independent_pack_difference <= INDEPENDENT_PACK_MAX_ABS_THRESHOLD
    )
    if recorded_passed is not independent_passed:
        raise ValueError(
            "independent dense RHT passed flag contradicts its component conditions"
        )
    return {
        "production_passed": production_passed,
        "independent_passed": independent_passed,
        "inverse_relative_l2": inverse,
        "encode_max_abs_difference": encode_difference,
        "physical_pack_max_abs_difference": independent_pack_difference,
    }


def audit_unit_evidence(unit_evidence: Mapping[str, Any]) -> dict[str, Any]:
    audit = validate_unit_evidence_schema(unit_evidence)
    if audit["production_passed"] is not True:
        raise ValueError("right-RHT production self-check did not pass")
    if audit["independent_passed"] is not True:
        raise ValueError("independent dense RHT reference did not pass")
    return {
        "passed": True,
        "inverse_relative_l2": audit["inverse_relative_l2"],
        "maximum_inverse_relative_l2": MAX_RHT_INVERSE_RELATIVE_L2,
        "physical_pack_matches_transformed_qdq": True,
        "sign_schedule_sha256": EXPECTED_SIGN_SCHEDULE_SHA256,
        "independent_dense_reference_passed": True,
        "independent_encode_max_abs_difference": audit[
            "encode_max_abs_difference"
        ],
        "independent_pack_max_abs_difference": audit[
            "physical_pack_max_abs_difference"
        ],
    }


def evaluate_stage_b_integrity(
    *,
    per_task: Mapping[str, Sequence[Mapping[str, Any]]],
    per_task_full_code: Mapping[str, Sequence[Mapping[str, Any]]],
    per_task_token_traces: Mapping[str, Sequence[Mapping[str, Any]]],
    per_task_full_code_token_traces: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    aggregates: Mapping[str, Mapping[str, Any]],
    aggregates_full_code: Mapping[str, Mapping[str, Any]],
    storage: Mapping[str, Mapping[str, Any]],
    selector_diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    per_task_state_errors: Mapping[str, Sequence[Mapping[str, Any]]],
    per_task_reference_aligned_state_errors: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    task_records: Sequence[Mapping[str, Any]],
    unit_evidence: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Machine-check all Stage-A integrity conditions on the 32-task run."""

    normalized_aligned: dict[str, list[dict[str, float | int | bool]]] | None = None
    normalized_full: dict[str, list[dict[str, float | int | bool]]] | None = None

    def provenance_check() -> dict[str, Any]:
        required_true = (
            "repository_clean_at_start",
            "repository_clean_at_end",
            "repository_commit_stable",
            "source_hashes_stable",
            "stage_a_artifact_committed_authenticated_and_passed",
            "identity_artifact_committed_and_authenticated",
            "identity_authenticated_before_model_weights",
            "identity_row_plan_authenticated",
            "identity_source_freeze_matches_current_bytes",
            "imported_modules_resolved_to_authenticated_repository",
            "runtime_environment_authenticated_before_dataset_access",
        )
        if any(integrity.get(field) is not True for field in required_true):
            raise ValueError("Stage-B provenance prerequisites did not all pass")
        if (
            integrity.get(
                "protected_window_8_16_content_selected_retained_canonicalized_"
                "formatted_tokenized_passed_to_model_or_evaluated"
            )
            is not False
        ):
            raise ValueError(
                "protected ranked window [8, 16) entered an application content set"
            )
        return {
            "passed": True,
            **{field: True for field in required_true},
            (
                "protected_window_8_16_content_selected_retained_canonicalized_"
                "formatted_tokenized_passed_to_model_or_evaluated"
            ): False,
        }

    def metric_check() -> dict[str, Any]:
        nonlocal normalized_aligned, normalized_full
        normalized_aligned = _normalize_metric_rows(
            per_task,
            task_records=task_records,
            token_field="aligned_scored_tokens",
            context="aligned",
        )
        normalized_full = _normalize_metric_rows(
            per_task_full_code,
            task_records=task_records,
            token_field="full_code_scored_tokens",
            context="full-code",
        )
        expected_aligned = _expected_aggregates(normalized_aligned)
        expected_full = _expected_aggregates(normalized_full)
        _audit_aggregate_match(aggregates, expected_aligned, context="aligned")
        _audit_aggregate_match(
            aggregates_full_code,
            expected_full,
            context="full-code",
        )
        aligned_trace_audit = audit_metric_traces(
            per_task_token_traces,
            per_task,
            task_records=task_records,
            token_field="aligned_scored_tokens",
            context="aligned",
        )
        full_trace_audit = audit_metric_traces(
            per_task_full_code_token_traces,
            per_task_full_code,
            task_records=task_records,
            token_field="full_code_scored_tokens",
            context="full-code",
        )
        return {
            "passed": True,
            "task_count": STAGE_B_LIMIT,
            "methods": list(METHODS),
            "all_logits_and_metrics_finite": True,
            "task_macro_aggregates_recomputed": True,
            "aligned_token_trace_audit": aligned_trace_audit,
            "full_code_token_trace_audit": full_trace_audit,
        }

    checks = {
        "committed_clean_stable_provenance": _gate_check(provenance_check),
        "finite_exact_task_macro_metrics": _gate_check(metric_check),
        "exact_physical_storage": _gate_check(lambda: audit_storage(storage)),
        "exact_quotas_and_query_handshakes": _gate_check(
            lambda: audit_task_selector_diagnostics(
                selector_diagnostics,
                task_records=task_records,
            )
        ),
        "complete_matched_state_error_coverage": _gate_check(
            lambda: audit_state_errors(
                per_task_state_errors,
                task_records=task_records,
                selector_diagnostics=selector_diagnostics,
            )
        ),
        "authenticated_reference_aligned_state_secondary": _gate_check(
            lambda: audit_reference_aligned_state_errors(
                per_task_reference_aligned_state_errors,
                task_records=task_records,
            )
        ),
        "deterministic_rht_codec_self_check": _gate_check(
            lambda: audit_unit_evidence(unit_evidence)
        ),
    }
    return {
        "schema": "recurquant.experiment009-stage-b-integrity.v1",
        "passed": all(check.get("passed") is True for check in checks.values()),
        "checks": checks,
    }


def _validate_bootstrap(
    bootstrap: Mapping[str, Any],
    *,
    cqer_values: Sequence[float],
    rht_values: Sequence[float],
) -> dict[str, Any]:
    recomputed = paired_bootstrap_mean_improvement(
        list(cqer_values),
        list(rht_values),
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )
    if dict(bootstrap) != recomputed:
        raise ValueError("paired bootstrap differs from the frozen recomputation")
    if (
        bootstrap.get("paired_examples") != STAGE_B_LIMIT
        or bootstrap.get("bootstrap_samples") != BOOTSTRAP_SAMPLES
        or bootstrap.get("seed") != SEED
        or bootstrap.get("confidence") != 0.95
    ):
        raise ValueError("paired bootstrap contract drifted")
    interval = bootstrap.get("confidence_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("paired bootstrap confidence interval is invalid")
    lower = _finite_float(interval[0], context="paired bootstrap lower bound")
    upper = _finite_float(interval[1], context="paired bootstrap upper bound")
    return {
        "passed": lower > 0,
        "lower": lower,
        "upper": upper,
        "confidence": 0.95,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
    }


def evaluate_stage_b_gate(
    *,
    aggregates: Mapping[str, Mapping[str, Any]],
    per_task: Mapping[str, Sequence[Mapping[str, Any]]],
    paired_bootstrap: Mapping[str, Any],
    aggregate_state_error: Mapping[str, Mapping[str, Any]],
    per_task_state_errors: Mapping[str, Sequence[Mapping[str, Any]]],
    selector_diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    task_records: Sequence[Mapping[str, Any]],
    integrity_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and evaluate exactly the eight frozen Stage-B checks."""

    if set(aggregates) != set(METHODS) or set(per_task) != set(METHODS):
        raise ValueError("Stage-B gate requires exactly the four frozen methods")
    if len(per_task[CQER_METHOD]) != STAGE_B_LIMIT or len(per_task[RHT_METHOD]) != STAGE_B_LIMIT:
        raise ValueError("Stage-B gate requires exactly 32 paired task rows")

    normalized = _normalize_metric_rows(
        per_task,
        task_records=task_records,
        token_field="aligned_scored_tokens",
        context="Stage-B advancement",
    )
    expected_aggregates = _expected_aggregates(normalized)
    _audit_aggregate_match(
        aggregates,
        expected_aggregates,
        context="Stage-B advancement",
    )
    audit_state_errors(
        per_task_state_errors,
        task_records=task_records,
        selector_diagnostics=selector_diagnostics,
    )
    expected_state_error = aggregate_state_errors(per_task_state_errors)
    if {
        method: dict(_mapping(record, context=f"{method} state aggregate"))
        for method, record in aggregate_state_error.items()
    } != expected_state_error:
        raise ValueError(
            "recorded aggregate state error differs from raw state-error recomputation"
        )

    cqer = _mapping(aggregates[CQER_METHOD], context="CQER aggregate")
    rht = _mapping(aggregates[RHT_METHOD], context="RHT aggregate")
    cqer_values = [
        _finite_float(row.get("delta_nll"), context="CQER task delta NLL")
        for row in per_task[CQER_METHOD]
    ]
    rht_values = [
        _finite_float(row.get("delta_nll"), context="RHT task delta NLL")
        for row in per_task[RHT_METHOD]
    ]

    def nll_reduction_check() -> dict[str, Any]:
        baseline = _finite_float(cqer.get("macro_delta_nll"), context="CQER macro excess NLL")
        candidate = _finite_float(rht.get("macro_delta_nll"), context="RHT macro excess NLL")
        if baseline <= 0:
            raise ValueError("CQER macro excess NLL must be positive for the relative gate")
        reduction = (baseline - candidate) / baseline
        return {
            "passed": reduction >= MIN_DELTA_NLL_REDUCTION,
            "cqer_macro_delta_nll": baseline,
            "rht_macro_delta_nll": candidate,
            "relative_reduction": reduction,
            "minimum_relative_reduction": MIN_DELTA_NLL_REDUCTION,
        }

    def win_check() -> dict[str, Any]:
        wins = sum(
            candidate < baseline
            for baseline, candidate in zip(cqer_values, rht_values, strict=True)
        )
        ties = sum(
            candidate == baseline
            for baseline, candidate in zip(cqer_values, rht_values, strict=True)
        )
        return {
            "passed": wins >= MIN_TASK_WINS,
            "rht_wins": wins,
            "ties": ties,
            "task_count": STAGE_B_LIMIT,
            "minimum_wins": MIN_TASK_WINS,
        }

    def bootstrap_check() -> dict[str, Any]:
        cqer_macro = _finite_float(
            cqer.get("macro_delta_nll"),
            context="CQER macro excess NLL for paired bootstrap",
        )
        if cqer_macro <= 0:
            raise ValueError(
                "CQER macro excess NLL must be positive for the paired advancement gate"
            )
        return _validate_bootstrap(
            paired_bootstrap,
            cqer_values=cqer_values,
            rht_values=rht_values,
        )

    def mean_kl_check() -> dict[str, Any]:
        baseline = _finite_float(cqer.get("macro_mean_kl"), context="CQER macro mean KL")
        candidate = _finite_float(rht.get("macro_mean_kl"), context="RHT macro mean KL")
        return {
            "passed": candidate < baseline,
            "cqer_macro_mean_kl": baseline,
            "rht_macro_mean_kl": candidate,
        }

    def cvar_check() -> dict[str, Any]:
        baseline = _finite_float(cqer.get("macro_cvar95_kl"), context="CQER macro CVaR95 KL")
        candidate = _finite_float(rht.get("macro_cvar95_kl"), context="RHT macro CVaR95 KL")
        return {
            "passed": candidate <= baseline,
            "cqer_macro_cvar95_kl": baseline,
            "rht_macro_cvar95_kl": candidate,
        }

    def top1_check() -> dict[str, Any]:
        baseline = _finite_float(
            cqer.get("macro_top1_agreement"),
            context="CQER macro top-1 agreement",
        )
        candidate = _finite_float(
            rht.get("macro_top1_agreement"),
            context="RHT macro top-1 agreement",
        )
        disadvantage = baseline - candidate
        return {
            "passed": disadvantage <= MAX_TOP1_DISADVANTAGE,
            "cqer_macro_top1_agreement": baseline,
            "rht_macro_top1_agreement": candidate,
            "observed_disadvantage": disadvantage,
            "maximum_disadvantage": MAX_TOP1_DISADVANTAGE,
        }

    def tail_task_check() -> dict[str, Any]:
        disadvantages = [
            {
                "task_id": int(cqer_row["task_id"]),
                "rht_minus_cqer_delta_nll": candidate - baseline,
            }
            for cqer_row, baseline, candidate in zip(
                per_task[CQER_METHOD],
                cqer_values,
                rht_values,
                strict=True,
            )
        ]
        worst = max(disadvantages, key=lambda record: record["rht_minus_cqer_delta_nll"])
        return {
            "passed": worst["rht_minus_cqer_delta_nll"] <= MAX_TASK_NLL_DISADVANTAGE,
            "worst_task_id": worst["task_id"],
            "maximum_observed_disadvantage": worst["rht_minus_cqer_delta_nll"],
            "maximum_allowed_disadvantage": MAX_TASK_NLL_DISADVANTAGE,
            "per_task": disadvantages,
        }

    def state_sse_check() -> dict[str, Any]:
        if set(aggregate_state_error) != set(QUERY_METHODS):
            raise ValueError("aggregate state error must contain CQER and RHT-CQER")
        baseline = _finite_float(
            aggregate_state_error[CQER_METHOD].get("aggregate_state_sse"),
            context="CQER aggregate state SSE",
        )
        candidate = _finite_float(
            aggregate_state_error[RHT_METHOD].get("aggregate_state_sse"),
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

    advancement_checks = {
        "macro_excess_nll_relative_reduction": _gate_check(nll_reduction_check),
        "paired_95pct_lower_bound_above_zero": _gate_check(bootstrap_check),
        "at_least_20_task_level_excess_nll_wins": _gate_check(win_check),
        "lower_macro_mean_kl": _gate_check(mean_kl_check),
        "macro_cvar95_kl_not_higher": _gate_check(cvar_check),
        "macro_top1_disadvantage_at_most_0_005": _gate_check(top1_check),
        "maximum_task_excess_nll_disadvantage_at_most_0_25": _gate_check(
            tail_task_check
        ),
        "aggregate_state_sse_relative_reduction": _gate_check(state_sse_check),
    }
    if len(advancement_checks) != 8:
        raise AssertionError("Experiment 009 Stage-B must have exactly eight advancement checks")
    integrity_passed = integrity_gate.get("passed") is True
    return {
        "schema": "recurquant.experiment009-stage-b-gate.v1",
        "applicable": True,
        "passed": (
            integrity_passed
            and all(check.get("passed") is True for check in advancement_checks.values())
        ),
        "integrity_passed": integrity_passed,
        "advancement_checks": advancement_checks,
        "thresholds": {
            "minimum_macro_excess_nll_relative_reduction": MIN_DELTA_NLL_REDUCTION,
            "paired_confidence": 0.95,
            "paired_lower_bound_strictly_positive": True,
            "minimum_task_wins": MIN_TASK_WINS,
            "macro_mean_kl_strictly_lower": True,
            "macro_cvar95_kl_not_higher": True,
            "maximum_macro_top1_disadvantage": MAX_TOP1_DISADVANTAGE,
            "maximum_per_task_excess_nll_disadvantage": MAX_TASK_NLL_DISADVANTAGE,
            "minimum_aggregate_state_sse_relative_reduction": MIN_STATE_SSE_REDUCTION,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
        },
    }


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields drifted")


def _require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_git_sha(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value != value.lower()
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{context} must be a full lowercase Git SHA")
    return value


def _require_safe_repository_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a non-empty POSIX repository path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"{context} must be a normalized repository-relative path")
    return value


def _validate_result_protocol(evidence: Mapping[str, Any]) -> None:
    protocol = _mapping(evidence.get("protocol"), context="Stage-B result protocol")
    expected = {
        "name": "Experiment 009 Stage B",
        "ranked_window": [STAGE_B_OFFSET, STAGE_B_STOP],
        "protected_ranked_window": list(PROTECTED_WINDOW),
        "protected_window_application_content_intersection": False,
        "methods_locked": True,
        "thresholds_locked": True,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SEED,
    }
    if dict(protocol) != expected:
        raise ValueError("Stage-B result protocol contract drifted")

    metric_contract = _mapping(
        evidence.get("metric_contract"),
        context="Stage-B result metric contract",
    )
    if dict(metric_contract) != {
        "primary": "task-macro aligned excess next-token NLL versus FP32 state",
        "aligned_excludes": "prompt-to-first-code-token prediction",
        "secondary": "task-macro full-code metrics",
        "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "paired_bootstrap_seed": SEED,
    }:
        raise ValueError("Stage-B result metric contract drifted")


def _validate_result_prerequisites(
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    prerequisites = _mapping(
        evidence.get("prerequisite_artifacts"),
        context="Stage-B result prerequisites",
    )
    _require_exact_fields(
        prerequisites,
        {
            "stage_a",
            "stage_b_identity",
            "committed_row_plan",
            "identity_source_freeze",
        },
        context="Stage-B result prerequisites",
    )

    stage_a = _mapping(prerequisites["stage_a"], context="Stage-A prerequisite")
    _require_exact_fields(
        stage_a,
        {
            "path",
            "sha256",
            "git_blob_sha256",
            "artifact_sha256",
            "canonical_evidence_sha256",
            "implementation_commit",
            "gate_recomputed_and_passed",
            "artifact_kind",
            "historical_path_privacy_limitation",
        },
        context="Stage-A prerequisite",
    )
    if (
        _require_safe_repository_path(
            stage_a.get("path"),
            context="Stage-A prerequisite path",
        )
        != identity_resolver.STAGE_A_ARTIFACT_RELATIVE_PATH
        or stage_a.get("artifact_kind") != STAGE_A_ARTIFACT_KIND
        or stage_a.get("sha256") != identity_resolver.STAGE_A_FILE_SHA256
        or stage_a.get("git_blob_sha256") != identity_resolver.STAGE_A_FILE_SHA256
        or stage_a.get("artifact_sha256") != identity_resolver.STAGE_A_FILE_SHA256
        or stage_a.get("canonical_evidence_sha256")
        != identity_resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256
        or stage_a.get("implementation_commit")
        != identity_resolver.STAGE_A_IMPLEMENTATION_COMMIT
        or stage_a.get("gate_recomputed_and_passed") != "true"
        or stage_a.get("historical_path_privacy_limitation")
        != STAGE_A_PATH_PRIVACY_LIMITATION
    ):
        raise ValueError("Stage-A prerequisite contract drifted")

    identity = _mapping(
        prerequisites["stage_b_identity"],
        context="Stage-B identity prerequisite",
    )
    _require_exact_fields(
        identity,
        {
            "path",
            "sha256",
            "git_blob_sha256",
            "artifact_sha256",
            "canonical_evidence_sha256",
            "resolver_commit",
            "artifact_kind",
        },
        context="Stage-B identity prerequisite",
    )
    identity_path = _require_safe_repository_path(
        identity.get("path"),
        context="Stage-B identity prerequisite path",
    )
    identity_file_sha = _require_sha256(
        identity.get("sha256"),
        context="Stage-B identity file",
    )
    if (
        not identity_path.endswith(".json")
        or identity.get("artifact_kind") != IDENTITY_ARTIFACT_KIND
        or identity.get("git_blob_sha256") != identity_file_sha
        or identity.get("artifact_sha256") != identity_file_sha
    ):
        raise ValueError("Stage-B identity prerequisite contract drifted")
    _require_sha256(
        identity.get("canonical_evidence_sha256"),
        context="Stage-B identity canonical evidence",
    )
    _require_git_sha(
        identity.get("resolver_commit"),
        context="Stage-B identity resolver commit",
    )

    row_plan = _mapping(
        prerequisites["committed_row_plan"],
        context="Stage-B committed row-plan authentication",
    )
    _require_exact_fields(
        row_plan,
        {
            "passed",
            "canonical_plan_sha256",
            "method",
            "promoted_group_count",
            "resident_bytes",
            "selector_binding",
        },
        context="Stage-B committed row-plan authentication",
    )
    selectors = _mapping(
        row_plan.get("selector_binding"),
        context="Stage-B row-plan selector binding",
    )
    expected_selectors = {
        "selector_file_sha256": identity_resolver.SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": (
            identity_resolver.SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
        "loss_selector_file_sha256": identity_resolver.LOSS_SELECTOR_FILE_SHA256,
        "loss_selector_canonical_evidence_sha256": (
            identity_resolver.LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
    }
    if (
        row_plan.get("passed") is not True
        or row_plan.get("method") != TARGET_FISHER_SCORE
        or row_plan.get("promoted_group_count") != TARGET_PROMOTED_ROWS
        or row_plan.get("resident_bytes") != TARGET_PACKED_STATE_BYTES
        or dict(selectors) != expected_selectors
    ):
        raise ValueError("Stage-B committed row-plan contract drifted")
    _require_sha256(
        row_plan.get("canonical_plan_sha256"),
        context="Stage-B committed row plan",
    )

    source_freeze = _mapping(
        prerequisites["identity_source_freeze"],
        context="Stage-B identity source freeze",
    )
    _require_exact_fields(
        source_freeze,
        {"passed", "path_count", "paths", "sha256"},
        context="Stage-B identity source freeze",
    )
    paths = list(SOURCE_FILES)
    hashes = _mapping(
        source_freeze.get("sha256"),
        context="Stage-B identity source-freeze hashes",
    )
    if (
        source_freeze.get("passed") is not True
        or source_freeze.get("path_count") != len(paths)
        or source_freeze.get("paths") != paths
        or set(hashes) != set(paths)
    ):
        raise ValueError("Stage-B identity source-freeze contract drifted")
    for path in paths:
        _require_safe_repository_path(path, context="Stage-B frozen source path")
        _require_sha256(
            hashes[path],
            context=f"Stage-B frozen source {path}",
        )
    return dict(hashes)


def _validate_identity_data_access(
    value: object,
    *,
    ordered_ids: list[int],
    source_rows: int,
) -> dict[str, Any]:
    access = _mapping(value, context="Stage-B identity data access")
    _require_exact_fields(
        access,
        {
            "transport_limitation",
            "ranking_pass",
            "target_load_pass",
            "application_task_id_sets",
            "protected_window_intersection",
        },
        context="Stage-B identity data access",
    )
    if access.get("transport_limitation") != DATA_ACCESS_TRANSPORT_LIMITATION:
        raise ValueError("Stage-B identity transport limitation drifted")
    ranking = _mapping(
        access.get("ranking_pass"),
        context="Stage-B identity ranking access",
    )
    if dict(ranking) != {
        "transport_records_yielded": source_rows,
        "task_id_fields_inspected": source_rows,
        "non_task_id_fields_read_by_recurquant": 0,
        "row_mappings_retained": 0,
    }:
        raise ValueError("Stage-B identity ranking access drifted")
    target = _mapping(
        access.get("target_load_pass"),
        context="Stage-B identity target-load access",
    )
    target_yielded = _strict_int(
        target.get("transport_records_yielded"),
        context="Stage-B identity target-load records",
    )
    if target_yielded < STAGE_B_LIMIT or dict(target) != {
        "transport_records_yielded": target_yielded,
        "task_id_fields_inspected": target_yielded,
        "non_target_content_fields_read_by_recurquant": 0,
        "target_rows_retained_and_canonicalized": STAGE_B_LIMIT,
    }:
        raise ValueError("Stage-B identity target-load access drifted")
    application_keys = {
        "selected",
        "retained",
        "canonicalized",
        "formatted",
        "tokenized",
        "passed_to_model",
        "evaluated",
    }
    expected_sets = {
        "selected": ordered_ids,
        "retained": ordered_ids,
        "canonicalized": ordered_ids,
        "formatted": ordered_ids,
        "tokenized": ordered_ids,
        "passed_to_model": [],
        "evaluated": [],
    }
    application_sets = _mapping(
        access.get("application_task_id_sets"),
        context="Stage-B identity application task sets",
    )
    if set(application_sets) != application_keys or dict(application_sets) != expected_sets:
        raise ValueError("Stage-B identity application task sets drifted")
    protected = _mapping(
        access.get("protected_window_intersection"),
        context="Stage-B identity protected-window intersection",
    )
    if set(protected) != application_keys or any(
        item is not False for item in protected.values()
    ):
        raise ValueError("protected ranked window entered the identity access sets")
    return dict(access)


def _validate_result_identity_dataset(
    identity_dataset: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    expected_fields = {
        "id",
        "config",
        "revision",
        "phase",
        "source_split",
        "selection_namespace",
        "formatter_version",
        "selection_mode",
        "selection_window",
        "protected_window",
        "ordered_task_ids",
        "manifest",
        "content_manifest_sha256",
        "token_manifest_sha256",
        "ordered_identity_sha256",
        "tasks",
        "totals",
        "data_access",
    }
    _require_exact_fields(
        identity_dataset,
        expected_fields,
        context="Stage-B result identity dataset",
    )
    expected_contract = {
        "id": identity_resolver.MBPP_DATASET_ID,
        "config": identity_resolver.MBPP_CONFIG,
        "revision": identity_resolver.MBPP_REVISION,
        "phase": "calibration",
        "source_split": identity_resolver.mbpp_source_split("calibration"),
        "selection_namespace": identity_resolver.MBPP_SELECTION_NAMESPACE,
        "formatter_version": identity_resolver.MBPP_FORMATTER_VERSION,
        "selection_mode": "task_id_ranking_then_exact_task_id_stream",
    }
    if any(identity_dataset.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Stage-B result identity dataset contract drifted")
    task_records = _identity_task_records({"dataset": identity_dataset})
    ordered_ids = [int(record["task_id"]) for record in task_records]

    manifest = _mapping(
        identity_dataset.get("manifest"),
        context="Stage-B result identity content manifest",
    )
    manifest_fields = {
        "schema",
        "dataset_id",
        "config",
        "revision",
        "phase",
        "source_split",
        "selection_namespace",
        "formatter_version",
        "row_count",
        "rows",
    }
    _require_exact_fields(
        manifest,
        manifest_fields,
        context="Stage-B result identity content manifest",
    )
    expected_manifest_contract = {
        "schema": identity_resolver.MBPP_MANIFEST_SCHEMA,
        "dataset_id": identity_resolver.MBPP_DATASET_ID,
        "config": identity_resolver.MBPP_CONFIG,
        "revision": identity_resolver.MBPP_REVISION,
        "phase": "calibration",
        "source_split": identity_resolver.mbpp_source_split("calibration"),
        "selection_namespace": identity_resolver.MBPP_SELECTION_NAMESPACE,
        "formatter_version": identity_resolver.MBPP_FORMATTER_VERSION,
        "row_count": STAGE_B_LIMIT,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest_contract.items()):
        raise ValueError("Stage-B result identity manifest contract drifted")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != STAGE_B_LIMIT:
        raise ValueError("Stage-B result identity manifest rows drifted")
    normalized_rows: list[dict[str, Any]] = []
    seen_manifest_ids: set[int] = set()
    previous_task_id: int | None = None
    for offset, raw_row in enumerate(rows):
        row = _mapping(
            raw_row,
            context=f"Stage-B result identity manifest row {offset}",
        )
        _require_exact_fields(
            row,
            {"task_id", "sha256"},
            context=f"Stage-B result identity manifest row {offset}",
        )
        task_id = _strict_int(
            row.get("task_id"),
            context=f"Stage-B result identity manifest row {offset} task_id",
        )
        row_sha256 = _require_sha256(
            row.get("sha256"),
            context=f"Stage-B result identity manifest row {offset}",
        )
        if task_id in seen_manifest_ids:
            raise ValueError("Stage-B result identity manifest task IDs are duplicated")
        if previous_task_id is not None and task_id <= previous_task_id:
            raise ValueError(
                "Stage-B result identity manifest rows are not in canonical task-ID order"
            )
        seen_manifest_ids.add(task_id)
        previous_task_id = task_id
        normalized_rows.append({"task_id": task_id, "sha256": row_sha256})
    expected_rows = sorted(
        (
            {
                "task_id": int(record["task_id"]),
                "sha256": str(record["row_sha256"]),
            }
            for record in task_records
        ),
        key=lambda row: row["task_id"],
    )
    if normalized_rows != expected_rows:
        raise ValueError("Stage-B result identity manifest rows do not match tasks")
    content_hash = _require_sha256(
        identity_dataset.get("content_manifest_sha256"),
        context="Stage-B result identity content manifest",
    )
    if mbpp_manifest_content_sha256(manifest) != content_hash:
        raise ValueError("Stage-B result identity content-manifest hash drifted")
    token_hash = _require_sha256(
        identity_dataset.get("token_manifest_sha256"),
        context="Stage-B result identity token manifest",
    )
    if identity_resolver.token_manifest_sha256(task_records) != token_hash:
        raise ValueError("Stage-B result identity token-manifest hash drifted")
    ordered_hash = _require_sha256(
        identity_dataset.get("ordered_identity_sha256"),
        context="Stage-B result ordered identity",
    )
    if (
        identity_resolver.ordered_identity_sha256(
            content_manifest_sha256=content_hash,
            task_records=task_records,
        )
        != ordered_hash
    ):
        raise ValueError("Stage-B result ordered-identity hash drifted")

    totals = _mapping(
        identity_dataset.get("totals"),
        context="Stage-B result identity totals",
    )
    source_rows = _strict_int(
        totals.get("source_train_rows_seen_by_task_id_only"),
        context="Stage-B result identity source row count",
    )
    if source_rows < identity_resolver.MBPP_CALIBRATION_SIZE or dict(totals) != {
        "source_train_rows_seen_by_task_id_only": source_rows,
        "retained_rows": STAGE_B_LIMIT,
        "prompt_tokens": sum(int(record["prompt_tokens"]) for record in task_records),
        "code_tokens": sum(int(record["code_tokens"]) for record in task_records),
        "aligned_scored_tokens": sum(
            int(record["aligned_scored_tokens"]) for record in task_records
        ),
        "full_code_scored_tokens": sum(
            int(record["full_code_scored_tokens"]) for record in task_records
        ),
    }:
        raise ValueError("Stage-B result identity totals drifted")
    access = _validate_identity_data_access(
        identity_dataset.get("data_access"),
        ordered_ids=ordered_ids,
        source_rows=source_rows,
    )
    return task_records, access


def _validate_result_dataset(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    dataset = _mapping(evidence.get("dataset"), context="Stage-B result dataset")
    protected_field = PROTECTED_EVALUATION_FIELD
    _require_exact_fields(
        dataset,
        {
            "phase",
            "selection_mode",
            "identity",
            "identity_validation",
            "data_access",
            "identity_authenticated_before_model_weights",
            protected_field,
        },
        context="Stage-B result dataset",
    )
    if (
        dataset.get("phase") != "calibration"
        or dataset.get("selection_mode")
        != "committed_exact_task_ids_from_ranked_window_32_64"
        or dataset.get("identity_authenticated_before_model_weights") is not True
        or dataset.get(protected_field) is not False
    ):
        raise ValueError("Stage-B result dataset contract drifted")
    identity_dataset = _mapping(
        dataset.get("identity"),
        context="Stage-B result identity dataset",
    )
    task_records, identity_access = _validate_result_identity_dataset(identity_dataset)
    ordered_ids = [int(record["task_id"]) for record in task_records]

    identity_validation = _mapping(
        dataset.get("identity_validation"),
        context="Stage-B result runtime identity validation",
    )
    expected_validation = {
        "passed": True,
        "authenticated_before_model_weights": True,
        "ordered_task_ids": ordered_ids,
        "content_manifest_sha256": identity_dataset["content_manifest_sha256"],
        "token_manifest_sha256": identity_dataset["token_manifest_sha256"],
        "ordered_identity_sha256": identity_dataset["ordered_identity_sha256"],
    }
    if dict(identity_validation) != expected_validation:
        raise ValueError("Stage-B result runtime identity validation drifted")

    access = _mapping(
        dataset.get("data_access"),
        context="Stage-B result evaluation data access",
    )
    _require_exact_fields(
        access,
        {
            "transport_limitation",
            "identity_resolution",
            "evaluator_target_load",
            "evaluator_application_task_id_sets",
            "protected_window_intersection",
            "non_target_source_records",
        },
        context="Stage-B result evaluation data access",
    )
    if (
        access.get("transport_limitation") != DATA_ACCESS_TRANSPORT_LIMITATION
        or access.get("identity_resolution") != identity_access
    ):
        raise ValueError("Stage-B result evaluation data-access binding drifted")
    target = _mapping(
        access.get("evaluator_target_load"),
        context="Stage-B result evaluator target load",
    )
    target_yielded = _strict_int(
        target.get("transport_records_yielded"),
        context="Stage-B result evaluator target-load records",
    )
    if target_yielded < STAGE_B_LIMIT or dict(target) != {
        "transport_records_yielded": target_yielded,
        "task_id_fields_inspected": target_yielded,
        "non_target_content_fields_read_by_recurquant": 0,
        "target_rows_retained_and_canonicalized": STAGE_B_LIMIT,
    }:
        raise ValueError("Stage-B result evaluator target-load access drifted")
    application_keys = {
        "selected",
        "retained",
        "canonicalized",
        "formatted",
        "tokenized",
        "passed_to_model",
        "evaluated",
    }
    expected_sets = {key: ordered_ids for key in application_keys}
    application_sets = _mapping(
        access.get("evaluator_application_task_id_sets"),
        context="Stage-B result evaluator application task sets",
    )
    if set(application_sets) != application_keys or dict(application_sets) != expected_sets:
        raise ValueError("Stage-B result evaluator application task sets drifted")
    protected = _mapping(
        access.get("protected_window_intersection"),
        context="Stage-B result protected-window intersection",
    )
    if set(protected) != application_keys or any(
        item is not False for item in protected.values()
    ):
        raise ValueError("protected ranked window entered evaluator application sets")
    if access.get("non_target_source_records") != {
        "recurquant_fields_inspected": ["task_id"],
        "content_retained_canonicalized_formatted_tokenized_or_evaluated": False,
    }:
        raise ValueError("Stage-B result non-target access contract drifted")
    return task_records


def _validate_result_runtime_and_model(evidence: Mapping[str, Any]) -> None:
    runtime = _mapping(
        evidence.get("runtime_environment"),
        context="Stage-B result authenticated runtime",
    )
    local_files_only = runtime.get("local_files_only")
    if not isinstance(local_files_only, bool):
        raise ValueError("Stage-B result local_files_only must be boolean")
    python_version = ".".join(
        str(component) for component in identity_resolver.STAGE_A_PYTHON_VERSION
    )
    expected_runtime = {
        "schema": identity_resolver.RUNTIME_ENVIRONMENT_SCHEMA,
        "stage_a_binding": {
            "artifact_kind": STAGE_A_ARTIFACT_KIND,
            "file_sha256": identity_resolver.STAGE_A_FILE_SHA256,
            "canonical_evidence_sha256": (
                identity_resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256
            ),
        },
        "python": {
            "major": identity_resolver.STAGE_A_PYTHON_VERSION[0],
            "minor": identity_resolver.STAGE_A_PYTHON_VERSION[1],
            "micro": identity_resolver.STAGE_A_PYTHON_VERSION[2],
            "version": python_version,
        },
        "packages": dict(identity_resolver.STAGE_A_PACKAGE_VERSIONS),
        "cuda": dict(identity_resolver.STAGE_A_CUDA_CONTRACT),
        "runtime_matches_stage_a": True,
        "local_files_only": local_files_only,
    }
    if dict(runtime) != expected_runtime:
        raise ValueError("Stage-B result authenticated runtime contract drifted")

    model = _mapping(evidence.get("model"), context="Stage-B result model")
    if dict(model) != {
        "id": identity_resolver.MODEL_ID,
        "revision": identity_resolver.MODEL_REVISION,
        "dtype": "torch.bfloat16",
        "device": "cuda",
    }:
        raise ValueError("Stage-B result model contract drifted")

    environment = _mapping(
        evidence.get("environment"),
        context="Stage-B result observed environment",
    )
    _require_exact_fields(
        environment,
        {
            "python",
            "platform",
            "packages",
            "cuda_available",
            "cuda_runtime",
            "gpu",
        },
        context="Stage-B result observed environment",
    )
    observed_python = environment.get("python")
    observed_platform = environment.get("platform")
    if (
        not isinstance(observed_python, str)
        or observed_python.split(maxsplit=1)[0] != python_version
        or not isinstance(observed_platform, str)
        or not observed_platform
        or environment.get("packages") != expected_runtime["packages"]
        or environment.get("cuda_available") is not True
        or environment.get("cuda_runtime")
        != expected_runtime["cuda"]["runtime_version"]
        or not isinstance(environment.get("gpu"), str)
        or not environment.get("gpu")
    ):
        raise ValueError("Stage-B result observed environment drifted")


def _validate_result_repository_and_sources(
    evidence: Mapping[str, Any],
    *,
    frozen_source_hashes: Mapping[str, str],
) -> None:
    repository = _mapping(
        evidence.get("repository"),
        context="Stage-B result repository",
    )
    _require_exact_fields(
        repository,
        {"commit", "start", "end", "stable_commit"},
        context="Stage-B result repository",
    )
    commit = _require_git_sha(
        repository.get("commit"),
        context="Stage-B result repository commit",
    )
    for name in ("start", "end"):
        snapshot = _mapping(
            repository.get(name),
            context=f"Stage-B result repository {name}",
        )
        if dict(snapshot) != {
            "commit": commit,
            "worktree_clean": True,
            "status": [],
        }:
            raise ValueError("Stage-B result repository was not clean and stable")
    if repository.get("stable_commit") is not True:
        raise ValueError("Stage-B result repository commit was not stable")

    sources = _mapping(
        evidence.get("source_files"),
        context="Stage-B result source files",
    )
    _require_exact_fields(
        sources,
        {
            "paths",
            "sha256_start",
            "sha256_end",
            "stable",
            "imported_module_paths",
        },
        context="Stage-B result source files",
    )
    paths = list(SOURCE_FILES)
    start = _mapping(
        sources.get("sha256_start"),
        context="Stage-B result source hashes start",
    )
    end = _mapping(
        sources.get("sha256_end"),
        context="Stage-B result source hashes end",
    )
    imported = _mapping(
        sources.get("imported_module_paths"),
        context="Stage-B result imported module paths",
    )
    if (
        sources.get("paths") != paths
        or sources.get("stable") is not True
        or dict(start) != dict(end)
        or dict(start) != dict(frozen_source_hashes)
        or set(start) != set(paths)
        or dict(imported) != RESULT_IMPORTED_MODULE_PATHS
    ):
        raise ValueError("Stage-B result source-file authentication drifted")
    for path in paths:
        _require_safe_repository_path(path, context="Stage-B result source path")
        _require_sha256(start[path], context=f"Stage-B result source {path}")
    for name, path in imported.items():
        _require_safe_repository_path(
            path,
            context=f"Stage-B result imported module {name}",
        )
        if path not in SOURCE_FILES:
            raise ValueError("Stage-B result imported module is outside source freeze")


def _validate_result_command(evidence: Mapping[str, Any]) -> None:
    command = evidence.get("command_template")
    if not isinstance(command, list):
        raise ValueError("Stage-B result command template must be an array")
    expected_prefix = [
        "python",
        "scripts/evaluate_rht_cqer_stage_b.py",
        "--stage-a-artifact",
        "<committed-stage-a-artifact>",
        "--identity-artifact",
        "<committed-stage-b-identity-artifact>",
        "--output",
        "<ignored-or-external-output>",
        "--device",
    ]
    if (
        command[: len(expected_prefix)] != expected_prefix
        or len(command) not in {len(expected_prefix) + 1, len(expected_prefix) + 2}
        or command[len(expected_prefix)] not in {"auto", "cuda"}
    ):
        raise ValueError("Stage-B result command template drifted")
    suffix = command[len(expected_prefix) + 1 :]
    runtime = _mapping(
        evidence.get("runtime_environment"),
        context="Stage-B result authenticated runtime",
    )
    expected_suffix = ["--local-files-only"] if runtime["local_files_only"] else []
    if suffix != expected_suffix:
        raise ValueError("Stage-B result command/runtime binding drifted")


def validate_stage_b_result_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every semantic Stage-B result section from one evidence object."""

    _require_exact_fields(
        evidence,
        RESULT_EVIDENCE_FIELDS,
        context="Stage-B result evidence",
    )
    if (
        evidence.get("schema_version") != 1
        or evidence.get("artifact_kind") != ARTIFACT_KIND
        or evidence.get("diagnostic_only") is not True
        or evidence.get("claim_boundary") != RESULT_CLAIM_BOUNDARY
        or evidence.get("methods") != list(METHODS)
    ):
        raise ValueError("Stage-B result schema or claim boundary drifted")
    created_at = evidence.get("created_at_utc")
    if not isinstance(created_at, str):
        raise ValueError("Stage-B result creation time must be ISO-8601")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError("Stage-B result creation time is not valid ISO-8601") from error
    if (
        parsed_created_at.tzinfo is None
        or parsed_created_at.utcoffset() is None
        or parsed_created_at.utcoffset().total_seconds() != 0
    ):
        raise ValueError("Stage-B result creation time must be UTC")

    _validate_result_protocol(evidence)
    frozen_source_hashes = _validate_result_prerequisites(evidence)
    task_records = _validate_result_dataset(evidence)
    _validate_result_runtime_and_model(evidence)
    _validate_result_repository_and_sources(
        evidence,
        frozen_source_hashes=frozen_source_hashes,
    )
    _validate_result_command(evidence)

    per_task = _mapping(
        evidence.get("per_task"),
        context="Stage-B result aligned per-task metrics",
    )
    per_task_full_code = _mapping(
        evidence.get("per_task_full_code_secondary"),
        context="Stage-B result full-code per-task metrics",
    )
    per_task_token_traces = _mapping(
        evidence.get("per_task_token_primitives"),
        context="Stage-B result aligned token primitives",
    )
    per_task_full_code_token_traces = _mapping(
        evidence.get("per_task_full_code_token_primitives"),
        context="Stage-B result full-code token primitives",
    )
    aggregates = _mapping(
        evidence.get("aggregates"),
        context="Stage-B result aligned aggregates",
    )
    aggregates_full_code = _mapping(
        evidence.get("aggregates_full_code_secondary"),
        context="Stage-B result full-code aggregates",
    )
    storage_section = _mapping(
        evidence.get("storage"),
        context="Stage-B result storage",
    )
    _require_exact_fields(
        storage_section,
        {"fp32_reference_recurrent_state_bytes", "candidates"},
        context="Stage-B result storage",
    )
    if (
        _strict_int(
            storage_section.get("fp32_reference_recurrent_state_bytes"),
            context="Stage-B FP32 recurrent-state bytes",
        )
        <= 0
    ):
        raise ValueError("Stage-B FP32 recurrent-state bytes must be positive")
    storage = _mapping(
        storage_section.get("candidates"),
        context="Stage-B result candidate storage",
    )
    selector_diagnostics = _mapping(
        evidence.get("selector_diagnostics"),
        context="Stage-B result selector diagnostics",
    )
    state_error = _mapping(
        evidence.get("state_error"),
        context="Stage-B result state error",
    )
    _require_exact_fields(
        state_error,
        {
            "primary_gate_metric",
            "aggregates",
            "per_task",
            "reference_aligned_secondary",
        },
        context="Stage-B result state error",
    )
    if state_error.get("primary_gate_metric") != PRIMARY_STATE_ERROR_METRIC:
        raise ValueError("Stage-B primary state-error metric label drifted")
    aggregate_state_error = _mapping(
        state_error.get("aggregates"),
        context="Stage-B result state-error aggregates",
    )
    per_task_state_errors = _mapping(
        state_error.get("per_task"),
        context="Stage-B result per-task state errors",
    )
    reference_aligned = _mapping(
        state_error.get("reference_aligned_secondary"),
        context="Stage-B result reference-aligned state error",
    )
    _require_exact_fields(
        reference_aligned,
        {"metric", "per_task"},
        context="Stage-B result reference-aligned state error",
    )
    if (
        reference_aligned.get("metric")
        != REFERENCE_ALIGNED_STATE_ERROR_METRIC
    ):
        raise ValueError("Stage-B reference-aligned state-error metric label drifted")
    per_task_reference_aligned = _mapping(
        reference_aligned.get("per_task"),
        context="Stage-B result per-task reference-aligned state error",
    )
    unit_evidence = _mapping(
        evidence.get("unit_evidence"),
        context="Stage-B result unit evidence",
    )
    validate_unit_evidence_schema(unit_evidence)
    integrity_inputs = _mapping(
        evidence.get("integrity_inputs"),
        context="Stage-B result integrity inputs",
    )
    expected_integrity_input_fields = {
        "repository_clean_at_start",
        "repository_clean_at_end",
        "repository_commit_stable",
        "source_hashes_stable",
        "stage_a_artifact_committed_authenticated_and_passed",
        "identity_artifact_committed_and_authenticated",
        "identity_authenticated_before_model_weights",
        "identity_row_plan_authenticated",
        "identity_source_freeze_matches_current_bytes",
        "imported_modules_resolved_to_authenticated_repository",
        "runtime_environment_authenticated_before_dataset_access",
        PROTECTED_EVALUATION_FIELD,
    }
    if set(integrity_inputs) != expected_integrity_input_fields:
        raise ValueError("Stage-B result integrity-input fields drifted")

    recomputed_integrity = evaluate_stage_b_integrity(
        per_task=per_task,
        per_task_full_code=per_task_full_code,
        per_task_token_traces=per_task_token_traces,
        per_task_full_code_token_traces=per_task_full_code_token_traces,
        aggregates=aggregates,
        aggregates_full_code=aggregates_full_code,
        storage=storage,
        selector_diagnostics=selector_diagnostics,
        per_task_state_errors=per_task_state_errors,
        per_task_reference_aligned_state_errors=per_task_reference_aligned,
        task_records=task_records,
        unit_evidence=unit_evidence,
        integrity=integrity_inputs,
    )
    recorded_integrity = _mapping(
        evidence.get("stage_b_integrity"),
        context="recorded Stage-B integrity",
    )
    if dict(recorded_integrity) != recomputed_integrity:
        raise ValueError("recorded Stage-B integrity differs from recomputation")

    bootstrap = _mapping(
        evidence.get("paired_bootstrap_cqer_minus_rht_aligned_delta_nll"),
        context="Stage-B result paired bootstrap",
    )
    recomputed_gate = evaluate_stage_b_gate(
        aggregates=aggregates,
        per_task=per_task,
        paired_bootstrap=bootstrap,
        aggregate_state_error=aggregate_state_error,
        per_task_state_errors=per_task_state_errors,
        selector_diagnostics=selector_diagnostics,
        task_records=task_records,
        integrity_gate=recomputed_integrity,
    )
    recorded_gate = _mapping(
        evidence.get("stage_b_gate"),
        context="recorded Stage-B gate",
    )
    if dict(recorded_gate) != recomputed_gate:
        raise ValueError("recorded Stage-B gate differs from recomputation")
    if len(recomputed_gate.get("advancement_checks", {})) != 8:
        raise ValueError("Stage-B result does not contain exactly eight advancement gates")
    return {
        "passed": True,
        "integrity_passed": recomputed_integrity.get("passed") is True,
        "advancement_passed": recomputed_gate.get("passed") is True,
        "task_count": STAGE_B_LIMIT,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SEED,
        "advancement_check_count": 8,
    }


def load_and_validate_stage_b_result_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a canonical Stage-B artifact and re-run its complete semantic audit."""

    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    verification = verify_evidence_artifact(artifact_path)
    if verification.get("valid") is not True:
        raise ValueError(
            "Stage-B result artifact failed canonical verification: "
            + "; ".join(str(error) for error in verification.get("errors", []))
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage-B result artifact must be strict UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "artifact_kind",
            "canonical_evidence_sha256",
            "evidence",
        }
        or document.get("schema_version") != 1
        or document.get("artifact_kind") != ARTIFACT_KIND
        or not isinstance(document.get("evidence"), dict)
    ):
        raise ValueError("Stage-B result artifact wrapper schema drifted")
    if raw != canonical_json_bytes(document):
        raise ValueError("Stage-B result artifact is not canonically serialized")
    evidence = dict(document["evidence"])
    semantic = validate_stage_b_result_evidence(evidence)
    return evidence, {
        **semantic,
        "artifact_sha256": _sha256_bytes(raw),
        "canonical_evidence_sha256": document["canonical_evidence_sha256"],
        "canonical_round_trip": True,
    }


def _record_reference_aligned_state_error(
    reference_cache: object,
    candidates: Mapping[str, object],
    *,
    write_ordinal: int,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    """Record candidate-state divergence from the matched FP32 cache trajectory."""

    reference = {
        (state.layer_index, state.state_index): state.tensor.detach()
        for state in iter_recurrent_states(reference_cache)
    }
    expected_keys = {(layer, 0) for layer in FROZEN_LINEAR_LAYERS}
    if set(reference) != expected_keys:
        raise RuntimeError("FP32 recurrent-state geometry drifted")
    for method, cache in candidates.items():
        candidate = {
            (state.layer_index, state.state_index): state.tensor.detach()
            for state in iter_recurrent_states(cache)
        }
        if set(candidate) != expected_keys:
            raise RuntimeError(f"{method} recurrent-state geometry differs from FP32")
        layer_records: list[dict[str, Any]] = []
        for layer in FROZEN_LINEAR_LAYERS:
            key = (layer, 0)
            reference_tensor = reference[key].to(torch.float64)
            candidate_tensor = candidate[key].to(
                device=reference_tensor.device,
                dtype=torch.float64,
            )
            if candidate_tensor.shape != reference_tensor.shape:
                raise RuntimeError(f"{method} layer {layer} recurrent-state shape drifted")
            error = candidate_tensor - reference_tensor
            state_sse = float(error.square().sum().item())
            element_count = error.numel()
            relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
                reference_tensor
            ).clamp_min(1e-12)
            layer_records.append(
                {
                    "layer_index": layer,
                    "state_index": 0,
                    "shape": list(error.shape),
                    "element_count": element_count,
                    "state_sse": state_sse,
                    "state_mse": state_sse / element_count,
                    "relative_l2_error": float(relative_l2.item()),
                    "max_absolute_error": float(error.abs().max().item()),
                }
            )
        records[method].append(
            {
                "write_ordinal": write_ordinal,
                "record_count": len(layer_records),
                "element_count": sum(
                    int(record["element_count"]) for record in layer_records
                ),
                "state_sse": math.fsum(
                    float(record["state_sse"]) for record in layer_records
                ),
                "layers": layer_records,
            }
        )


def _aggregate_reference_aligned_state_error(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in QUERY_METHODS:
        writes = records[method]
        total_elements = sum(int(record["element_count"]) for record in writes)
        total_sse = math.fsum(float(record["state_sse"]) for record in writes)
        result[method] = {
            "metric": "candidate_materialized_state_minus_matched_fp32_state",
            "write_count": len(writes),
            "record_count": sum(int(record["record_count"]) for record in writes),
            "element_count": total_elements,
            "aggregate_state_sse": total_sse,
            "aggregate_state_mse": total_sse / total_elements,
            "writes": list(writes),
        }
    return result


def evaluate_task(
    model: Qwen3_5ForCausalLM,
    *,
    prompt_ids: torch.Tensor,
    code_ids: torch.Tensor,
    plan: ExactBudgetRowPlan,
) -> dict[str, Any]:
    """Evaluate FP32 reference and all four frozen methods in one teacher-forced loop."""

    reference_cache = DynamicCache(config=model.config)
    static = create_qwen35_exact_budget_cache(model, plan=plan)
    adaptive = create_qwen35_adaptive_exact_budget_cache(model, plan=plan)
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
    caches = {
        STATIC_METHOD: static,
        ADAPTIVE_METHOD: adaptive,
        CQER_METHOD: cqer,
        RHT_METHOD: rht,
    }
    aligned = {name: pilot._TokenAccumulator.empty() for name in METHODS}  # noqa: SLF001
    full = {name: pilot._TokenAccumulator.empty() for name in METHODS}  # noqa: SLF001
    observer = Qwen35QueryEnergyObserver(model, caches=[cqer, rht])
    reference_aligned_records = {method: [] for method in QUERY_METHODS}

    with observer:
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
        _record_reference_aligned_state_error(
            reference_cache,
            {CQER_METHOD: cqer, RHT_METHOD: rht},
            write_ordinal=0,
            records=reference_aligned_records,
        )
        pilot._append_metrics(  # noqa: SLF001
            full,
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
            _record_reference_aligned_state_error(
                reference_cache,
                {CQER_METHOD: cqer, RHT_METHOD: rht},
                write_ordinal=token_index + 1,
                records=reference_aligned_records,
            )
            logits = {name: output.logits for name, output in candidate_outputs.items()}
            pilot._append_metrics(  # noqa: SLF001
                aligned,
                reference_output.logits,
                logits,
                target_token,
            )
            pilot._append_metrics(  # noqa: SLF001
                full,
                reference_output.logits,
                logits,
                target_token,
            )

    reference_bytes = sum(
        state.tensor.numel() * state.tensor.element_size()
        for state in iter_recurrent_states(reference_cache)
    )
    return {
        "aligned_metrics": {name: values.summary() for name, values in aligned.items()},
        "full_code_metrics": {name: values.summary() for name, values in full.items()},
        "aligned_token_traces": {
            name: _accumulator_trace(values) for name, values in aligned.items()
        },
        "full_code_token_traces": {
            name: _accumulator_trace(values) for name, values in full.items()
        },
        "storage": {name: cache.storage_summary() for name, cache in caches.items()},
        "selector_diagnostics": {
            CQER_METHOD: cqer.query_ema_diagnostics(),
            RHT_METHOD: rht.query_ema_diagnostics(),
        },
        "state_errors": {
            CQER_METHOD: screen.aggregate_state_error_evidence(cqer.update_evidence),
            RHT_METHOD: screen.aggregate_state_error_evidence(rht.update_evidence),
        },
        "reference_aligned_state_errors": _aggregate_reference_aligned_state_error(
            reference_aligned_records
        ),
        "reference_recurrent_state_bytes": reference_bytes,
    }


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


def _evaluation_data_access_record(
    identity: Mapping[str, Any],
    runtime_identity: Any,
) -> dict[str, Any]:
    """Record transport limits and exact application-level Stage-B content sets."""

    selected = [int(task_id) for task_id in runtime_identity.ordered_task_ids]
    if selected != list(identity["dataset"]["ordered_task_ids"]):
        raise ValueError("runtime selected task IDs differ from the committed identity")
    if runtime_identity.access_audit is None:
        raise ValueError("runtime identity lacks task-ID stream access instrumentation")
    runtime_before_weights = runtime_identity.access_audit.as_dict(
        selected_task_ids=selected
    )
    protected = runtime_before_weights["protected_window_intersection"]
    if any(value is not False for value in protected.values()):
        raise ValueError("protected ranked window intersects a runtime application set")
    return {
        "transport_limitation": runtime_before_weights["transport_limitation"],
        "identity_resolution": identity["dataset"]["data_access"],
        "evaluator_target_load": runtime_before_weights["target_load_pass"],
        "evaluator_application_task_id_sets": {
            "selected": selected,
            "retained": selected,
            "canonicalized": selected,
            "formatted": selected,
            "tokenized": selected,
            "passed_to_model": selected,
            "evaluated": selected,
        },
        "protected_window_intersection": {
            "selected": False,
            "retained": False,
            "canonicalized": False,
            "formatted": False,
            "tokenized": False,
            "passed_to_model": False,
            "evaluated": False,
        },
        "non_target_source_records": {
            "recurquant_fields_inspected": ["task_id"],
            "content_retained_canonicalized_formatted_tokenized_or_evaluated": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    repository_start = pilot.git_state()
    pilot.validate_cora_development_repository_start(repository_start)
    pilot.validate_heldout_output_path(args.output, repository_root)
    source_hashes_start = pilot.source_file_hashes(repository_root, SOURCE_FILES)
    imported_module_paths = authenticate_imported_module_paths(repository_root)

    # Every prerequisite is authenticated before any dataset or tokenizer access.
    stage_a, stage_a_hashes = authenticate_stage_a_prerequisite(
        args.stage_a_artifact,
        repository_root=repository_root,
    )
    identity, identity_hashes, task_records = authenticate_stage_b_identity(
        args.identity_artifact,
        repository_root=repository_root,
    )
    identity_source_freeze = authenticate_identity_source_freeze(
        identity,
        current_source_hashes=source_hashes_start,
    )
    plan, row_plan_authentication = plan_from_identity(identity, stage_a=stage_a)
    authorization = _mapping(
        identity.get("authorization"),
        context="Stage-B identity authorization",
    )
    if (
        authorization.get("stage_a_file_sha256")
        != stage_a_hashes["artifact_sha256"]
        or authorization.get("stage_a_canonical_evidence_sha256")
        != stage_a_hashes["canonical_evidence_sha256"]
        or authorization.get("stage_a_gate_passed") is not True
    ):
        raise ValueError("Stage-B identity does not authorize the authenticated Stage-A pass")
    runtime_environment = authenticate_stage_b_runtime_environment(
        stage_a,
        identity,
        local_files_only=bool(args.local_files_only),
    )
    # Exact committed IDs are the only application-level content selector.
    authenticated_rows = identity_resolver.load_authenticated_stage_b_rows(identity)
    task_ids = authenticated_rows.ordered_task_ids
    if task_ids != tuple(int(record["task_id"]) for record in task_records):
        raise RuntimeError("Stage-B task-ID loader returned an unauthorized identity")

    device = pilot.select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    stage_a_model = _mapping(stage_a.get("model"), context="Stage-A model")
    if stage_a_model.get("dtype") != str(dtype):
        raise ValueError(
            f"Stage-B dtype {dtype} differs from Stage-A dtype {stage_a_model.get('dtype')}"
        )
    model_id = str(identity["model_contract"]["id"])
    revision = str(identity["model_contract"]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    runtime_identity = identity_resolver.authenticate_stage_b_runtime_identity(
        identity,
        authenticated_rows,
        tokenizer,
    )
    if runtime_identity.ordered_task_ids != task_ids:
        raise RuntimeError("runtime token identity changed the committed task order")
    encoded_tasks = tuple(
        (
            task.row,
            torch.tensor([task.prompt_token_ids], dtype=torch.int64),
            torch.tensor([task.code_token_ids], dtype=torch.int64),
        )
        for task in runtime_identity.tasks
    )
    identity_validation = {
        "passed": True,
        "authenticated_before_model_weights": True,
        "ordered_task_ids": list(runtime_identity.ordered_task_ids),
        "content_manifest_sha256": runtime_identity.content_manifest_sha256,
        "token_manifest_sha256": runtime_identity.token_manifest_sha256,
        "ordered_identity_sha256": runtime_identity.ordered_identity_sha256,
    }
    data_access = _evaluation_data_access_record(identity, runtime_identity)

    # Identity, text, tokenizer output, and all hashes now match before weights.
    unit_evidence = {
        "production_self_check": screen.compute_unit_evidence(),
        "independent_dense_reference": compute_independent_dense_rht_evidence(),
    }
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

    per_task = {method: [] for method in METHODS}
    per_task_full_code = {method: [] for method in METHODS}
    per_task_token_traces = {method: [] for method in METHODS}
    per_task_full_code_token_traces = {method: [] for method in METHODS}
    selector_diagnostics = {method: [] for method in QUERY_METHODS}
    per_task_state_errors = {method: [] for method in QUERY_METHODS}
    per_task_reference_aligned_state_errors = {
        method: [] for method in QUERY_METHODS
    }
    storage_anchor: dict[str, dict[str, Any]] | None = None
    reference_state_bytes: int | None = None

    with torch.inference_mode():
        for task_number, ((_row, prompt_cpu, code_cpu), identity_task) in enumerate(
            zip(encoded_tasks, task_records, strict=True),
            start=1,
        ):
            torch.manual_seed(SEED)
            result = evaluate_task(
                model,
                prompt_ids=prompt_cpu.to(device),
                code_ids=code_cpu.to(device),
                plan=plan,
            )
            storage = result["storage"]
            task_reference_bytes = int(result["reference_recurrent_state_bytes"])
            if storage_anchor is None:
                storage_anchor = storage
                reference_state_bytes = task_reference_bytes
            elif storage != storage_anchor or task_reference_bytes != reference_state_bytes:
                raise RuntimeError("resident state storage changed between Stage-B tasks")
            task_id = int(identity_task["task_id"])
            for method in METHODS:
                per_task[method].append(
                    {"task_id": task_id, **result["aligned_metrics"][method]}
                )
                per_task_full_code[method].append(
                    {"task_id": task_id, **result["full_code_metrics"][method]}
                )
                per_task_token_traces[method].append(
                    {"task_id": task_id, **result["aligned_token_traces"][method]}
                )
                per_task_full_code_token_traces[method].append(
                    {
                        "task_id": task_id,
                        **result["full_code_token_traces"][method],
                    }
                )
            for method in QUERY_METHODS:
                selector_diagnostics[method].append(
                    {
                        "task_id": task_id,
                        "layers": result["selector_diagnostics"][method],
                    }
                )
                per_task_state_errors[method].append(
                    {
                        "task_id": task_id,
                        "state_error": result["state_errors"][method],
                    }
                )
                per_task_reference_aligned_state_errors[method].append(
                    {
                        "task_id": task_id,
                        "state_error": result["reference_aligned_state_errors"][
                            method
                        ],
                    }
                )
            print(
                f"[{task_number}/{STAGE_B_LIMIT}] task={task_id} "
                f"code_tokens={code_cpu.shape[1]}",
                flush=True,
            )

    assert storage_anchor is not None
    assert reference_state_bytes is not None
    aggregates = pilot.aggregate_task_rows(per_task)
    aggregates_full_code = pilot.aggregate_task_rows(per_task_full_code)
    cqer_values = [float(row["delta_nll"]) for row in per_task[CQER_METHOD]]
    rht_values = [float(row["delta_nll"]) for row in per_task[RHT_METHOD]]
    bootstrap = paired_bootstrap_mean_improvement(
        cqer_values,
        rht_values,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )
    state_error_aggregates = aggregate_state_errors(per_task_state_errors)

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
        "repository_commit_stable": repository_start["commit"] == repository_end["commit"],
        "source_hashes_stable": source_hashes_start == source_hashes_end,
        "stage_a_artifact_committed_authenticated_and_passed": True,
        "identity_artifact_committed_and_authenticated": True,
        "identity_authenticated_before_model_weights": (
            identity_validation.get("passed") is True
        ),
        "identity_row_plan_authenticated": row_plan_authentication["passed"] is True,
        "identity_source_freeze_matches_current_bytes": (
            identity_source_freeze["passed"] is True
        ),
        "imported_modules_resolved_to_authenticated_repository": True,
        "runtime_environment_authenticated_before_dataset_access": True,
        (
            "protected_window_8_16_content_selected_retained_canonicalized_"
            "formatted_tokenized_passed_to_model_or_evaluated"
        ): False,
    }
    integrity_gate = evaluate_stage_b_integrity(
        per_task=per_task,
        per_task_full_code=per_task_full_code,
        per_task_token_traces=per_task_token_traces,
        per_task_full_code_token_traces=per_task_full_code_token_traces,
        aggregates=aggregates,
        aggregates_full_code=aggregates_full_code,
        storage=storage_anchor,
        selector_diagnostics=selector_diagnostics,
        per_task_state_errors=per_task_state_errors,
        per_task_reference_aligned_state_errors=(
            per_task_reference_aligned_state_errors
        ),
        task_records=task_records,
        unit_evidence=unit_evidence,
        integrity=integrity,
    )
    gate = evaluate_stage_b_gate(
        aggregates=aggregates,
        per_task=per_task,
        paired_bootstrap=bootstrap,
        aggregate_state_error=state_error_aggregates,
        per_task_state_errors=per_task_state_errors,
        selector_diagnostics=selector_diagnostics,
        task_records=task_records,
        integrity_gate=integrity_gate,
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "diagnostic_only": True,
        "claim_boundary": RESULT_CLAIM_BOUNDARY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "name": "Experiment 009 Stage B",
            "ranked_window": [STAGE_B_OFFSET, STAGE_B_STOP],
            "protected_ranked_window": list(PROTECTED_WINDOW),
            "protected_window_application_content_intersection": False,
            "methods_locked": True,
            "thresholds_locked": True,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SEED,
        },
        "prerequisite_artifacts": {
            "stage_a": {
                **stage_a_hashes,
                "artifact_kind": stage_a["artifact_kind"],
                "historical_path_privacy_limitation": STAGE_A_PATH_PRIVACY_LIMITATION,
            },
            "stage_b_identity": {
                **identity_hashes,
                "artifact_kind": IDENTITY_ARTIFACT_KIND,
            },
            "committed_row_plan": row_plan_authentication,
            "identity_source_freeze": identity_source_freeze,
        },
        "model": {
            "id": model_id,
            "revision": revision,
            "dtype": str(dtype),
            "device": str(device),
        },
        "dataset": {
            "phase": "calibration",
            "selection_mode": "committed_exact_task_ids_from_ranked_window_32_64",
            "identity": identity["dataset"],
            "identity_validation": identity_validation,
            "data_access": data_access,
            "identity_authenticated_before_model_weights": True,
            (
                "protected_window_8_16_content_selected_retained_canonicalized_"
                "formatted_tokenized_passed_to_model_or_evaluated"
            ): False,
        },
        "metric_contract": {
            "primary": "task-macro aligned excess next-token NLL versus FP32 state",
            "aligned_excludes": "prompt-to-first-code-token prediction",
            "secondary": "task-macro full-code metrics",
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "paired_bootstrap_seed": SEED,
        },
        "methods": list(METHODS),
        "storage": {
            "fp32_reference_recurrent_state_bytes": reference_state_bytes,
            "candidates": storage_anchor,
        },
        "aggregates": aggregates,
        "aggregates_full_code_secondary": aggregates_full_code,
        "per_task": per_task,
        "per_task_full_code_secondary": per_task_full_code,
        "per_task_token_primitives": per_task_token_traces,
        "per_task_full_code_token_primitives": per_task_full_code_token_traces,
        "paired_bootstrap_cqer_minus_rht_aligned_delta_nll": bootstrap,
        "selector_diagnostics": selector_diagnostics,
        "state_error": {
            "primary_gate_metric": PRIMARY_STATE_ERROR_METRIC,
            "aggregates": state_error_aggregates,
            "per_task": per_task_state_errors,
            "reference_aligned_secondary": {
                "metric": REFERENCE_ALIGNED_STATE_ERROR_METRIC,
                "per_task": per_task_reference_aligned_state_errors,
            },
        },
        "unit_evidence": unit_evidence,
        "integrity_inputs": integrity,
        "stage_b_integrity": integrity_gate,
        "stage_b_gate": gate,
        "runtime_environment": runtime_environment,
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
            "start": repository_start,
            "end": repository_end,
            "stable_commit": repository_start["commit"] == repository_end["commit"],
        },
        "source_files": {
            "paths": list(SOURCE_FILES),
            "sha256_start": source_hashes_start,
            "sha256_end": source_hashes_end,
            "stable": source_hashes_start == source_hashes_end,
            "imported_module_paths": imported_module_paths,
        },
        "command_template": [
            "python",
            "scripts/evaluate_rht_cqer_stage_b.py",
            "--stage-a-artifact",
            "<committed-stage-a-artifact>",
            "--identity-artifact",
            "<committed-stage-b-identity-artifact>",
            "--output",
            "<ignored-or-external-output>",
            "--device",
            args.device,
            *(["--local-files-only"] if args.local_files_only else []),
        ],
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "canonical_evidence_sha256": _sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    _atomic_write(args.output, payload)
    round_trip_evidence, semantic_verification = (
        load_and_validate_stage_b_result_artifact(args.output)
    )
    if round_trip_evidence != evidence:
        raise RuntimeError("written Stage-B evidence changed during canonical round-trip")

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
                "artifact_sha256": _sha256_bytes(payload),
                "canonical_evidence_sha256": artifact["canonical_evidence_sha256"],
                "stage_b_integrity": integrity_gate,
                "stage_b_gate": gate,
                "semantic_verification": semantic_verification,
                "post_write_repository_authenticated": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gate["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
