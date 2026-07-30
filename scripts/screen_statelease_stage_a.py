#!/usr/bin/env python3
"""Run the one permitted Experiment 010 StateLease Stage-A falsification screen.

The command is deliberately difficult to run accidentally.  It authenticates
the committed Experiment 009 task-666 evidence, both frozen selector
artifacts, an independently verified production Stage-0 artifact, the exact
repository source set, and the pinned model configuration before reserving the
single Stage-A attempt.  Only then may it read the already-open task 666,
tokenize it, or load model weights.

Stage A cannot support a public improvement, novelty, deployment, speed,
state-of-the-art, or breakthrough claim, even when every frozen gate passes.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from recurquant.cache import iter_recurrent_states
from recurquant.evaluation import TokenFidelity, fidelity_summary, token_fidelity
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.statelease_evaluation import (
    EQUAL_BYTE_NO_REPLAY_METHODS,
    FIXED_REPLAY_METHODS,
    FROZEN_STATELEASE_RESIDENT_BYTES,
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    TrajectoryNmseAccumulator,
    evaluate_statelease_stage_a_gate,
    reference_aligned_trajectory_nmse,
)

SEED = 2339
ARTIFACT_KIND = "recurquant_experiment010_statelease_stage_a_falsification"
ARTIFACT_SCHEMA_VERSION = 1

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MODEL_DTYPE = torch.bfloat16
TASK_ID = 666
TASK_ROW_SHA256 = "b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9"
PROMPT_TEXT_SHA256 = "b6f0f93b9d15b96ac42bbabbdb349a09d2d24e57667d47cafe900c1ea91fd64b"
CODE_TEXT_SHA256 = "d2701e79ccd968c9e5af78474af16256f3bbf39cdfedbec2199ac92e1a4f397e"
PROMPT_TOKEN_IDS_SHA256 = "729215c4c99cdf96b13ad73f6ac7b537ddf9e882409b77e479d609aee046bffa"
CODE_TOKEN_IDS_SHA256 = "a920370c4892513c8a5cdb9f88a33fd95d4c90201af39fdb7d517f3ad42a9d9a"
TOKEN_ID_HASH_SERIALIZATION = "sha256(recurquant.evidence.canonical_json_bytes(list[int]))"
PROMPT_TOKENS = 69
CODE_TOKENS = 39
ALIGNED_TOKENS = 38

ORIGINAL_RHT_METHOD = RHT_CQER_METHOD
FP32_METHOD = "fp32_reference"
QUALITY_METHODS = (
    ORIGINAL_RHT_METHOD,
    STATELEASE_METHOD,
    *FIXED_REPLAY_METHODS,
    *EQUAL_BYTE_NO_REPLAY_METHODS,
)
ALL_METHODS = (FP32_METHOD, *QUALITY_METHODS)

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
EXPECTED_GEOMETRY = {
    "num_hidden_layers": 24,
    "recurrent_layers": 18,
    "linear_layer_indices": list(LINEAR_LAYER_INDICES),
    "value_heads": 16,
    "key_rows": 128,
    "value_width": 128,
    "batch_size": 1,
    "recurrent_state_dtype": "torch.float32",
}

EXPERIMENT009_STAGE_A_RELATIVE_PATH = "evidence/experiment009-rht-cqer-stage-a-666-5be8d48.json"
EXPERIMENT009_STAGE_A_FILE_SHA256 = (
    "98a432843dc438f2d5fde34f8704f154ebc3ee12c93ba7c469369acfedfb15b5"
)
EXPERIMENT009_STAGE_A_CANONICAL_SHA256 = (
    "9e03a1e8cefb5801406a47a2e5e365686afb0a05e10e099a989cee616b505ed1"
)
EXPERIMENT009_MANIFEST_SHA256 = "3c21e7d6534bcdeb5b6af7fdf80560529639e27b77ffaa4f086491a7a3a50ea4"
EXPERIMENT009_MANIFEST_CANONICAL_SHA256 = (
    "f9d6715f9f280d5de83e72ff815987e7aff2be406ab501d99776b3f119434873"
)
EXPERIMENT009_TOKEN_MANIFEST_SHA256 = (
    "79732fe8d64bab58dae58f64f97426945a24f317ce4c6d25511fab82d0679643"
)
EXPERIMENT009_IDENTITY_SHA256 = "135a8fe824c8fd640f0738f23dc165b6fa17b7795c9832273a742d6bee8a850f"

SELECTOR_RELATIVE_PATH = "artifacts/experiment006-hrr-selector-8task-c2ad68b.json"
LOSS_SELECTOR_RELATIVE_PATH = "artifacts/experiment006-loss-selector-8task-c2ad68b.json"
SELECTOR_FILE_SHA256 = "d0c4267095ee3f5068627b189a1fd9f58cb02f6e25672d9b89dd0990e5b09330"
LOSS_SELECTOR_FILE_SHA256 = "95c16656edb32efbc985f2fea59e229634dd558f4f4bf04819b8efc37783a1d6"
SELECTOR_CANONICAL_SHA256 = "7970961fd88b522998189ad64f26b333aed9c88ff5f653de5449fd9e01d8cbc8"
LOSS_SELECTOR_CANONICAL_SHA256 = "bff4e33253990b8115e1f35e74516c4975c2fe4aac5066475afe968eb8a64609"

OUTPUT_RELATIVE_PATH = "artifacts/experiment010-statelease-stage-a-666.json"
ATTEMPT_RELATIVE_PATH = "artifacts/experiment010-statelease-stage-a-666.attempt.json"
IDENTITY_NOTE_RELATIVE_PATH = "research/EXPERIMENT_010_STAGE_A_IDENTITY.md"
IDENTITY_NOTE_FILE_SHA256 = "0bab7c8f416ce238071b9a87ed6b6dda6450d0e21265ee06ce5e47b1be36deb6"
ONE_RUN_MARKER = "RecurQuant-One-Run: experiment010-stage-a-task666-v1"
ONE_RUN_LIMITATION = (
    "The local Git commit plus reflog is tamper-evident for normal repository "
    "operations, not cryptographically non-bypassable against deliberate ref and "
    "reflog destruction; external append-only anchoring is outside this evaluator."
)

EXPECTED_DATASET_MANIFEST = {
    "config": "full",
    "dataset_id": "google-research-datasets/mbpp",
    "formatter_version": "recurquant.mbpp-prompt-code.v1",
    "phase": "calibration",
    "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "row_count": 1,
    "rows": [{"sha256": TASK_ROW_SHA256, "task_id": TASK_ID}],
    "schema": "recurquant.mbpp-manifest.v1",
    "selection_namespace": "rq-v0.2",
    "source_split": "train",
}
EXPECTED_TOKEN_MANIFEST = [
    {
        "aligned_scored_tokens": ALIGNED_TOKENS,
        "code_tokens": CODE_TOKENS,
        "full_code_scored_tokens": CODE_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "task_id": TASK_ID,
    }
]
EXPECTED_TASK_IDENTITY = {
    "aligned_scored_tokens": ALIGNED_TOKENS,
    "authenticated_before_model_weights": True,
    "code_tokens": CODE_TOKENS,
    "prompt_tokens": PROMPT_TOKENS,
    "row_sha256": TASK_ROW_SHA256,
    "task_id": TASK_ID,
}

SOURCE_FILES = (
    "pyproject.toml",
    "scripts/capture_statelease_stage0.py",
    "scripts/screen_rht_cqer.py",
    "scripts/screen_statelease_stage_a.py",
    "scripts/verify_statelease_stage0.py",
    IDENTITY_NOTE_RELATIVE_PATH,
    "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
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

CLAIM_BOUNDARY = (
    "Experiment 010 Stage A is one-task, already-open falsification evidence. "
    "Passing only authorizes the separately frozen development-identity step. "
    "It cannot support a public improvement, novelty, deployment, speed, "
    "state-of-the-art, or breakthrough claim."
)


class StageAAuthenticationError(RuntimeError):
    """An input, source, identity, or one-run condition failed closed."""


@dataclass(frozen=True, slots=True)
class StageAPreflight:
    repo_root: Path
    repository_start: dict[str, object]
    source_hashes_start: dict[str, str]
    identity_clarification: dict[str, object]
    anchor: dict[str, object]
    selector_identity: dict[str, object]
    plan: object
    stage0: dict[str, object]
    stage0_artifact: Path
    stage0_sha256: Path
    output_path: Path
    attempt_path: Path


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    config: object
    identity: dict[str, object]


@dataclass(frozen=True, slots=True)
class TokenizedTask:
    row: Mapping[str, object]
    prompt_ids: torch.Tensor
    code_ids: torch.Tensor
    token_manifest: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    receipt: dict[str, object]
    receipt_file_sha256: str
    h0_commit: str
    seal_commit: str
    tree: str
    seal_message_sha256: str


@dataclass(frozen=True, slots=True)
class AccessHooks:
    """Dependency-injected ordering contract used by production and tests."""

    authenticate: Callable[[], object]
    load_config: Callable[[object], object]
    reserve_attempt: Callable[[object, object], object]
    load_exact_task: Callable[[object], object]
    tokenize_task: Callable[[object, object], object]
    load_weights: Callable[[object, object], object]
    evaluate: Callable[[object, object, object, object], object]
    finalize: Callable[[object, object, object, object, object, object], object]
    record_failure: Callable[[object, BaseException], None]


def run_ordered_access(hooks: AccessHooks) -> object:
    """Enforce authentication-before-data and token-identity-before-weights."""

    authenticated = hooks.authenticate()
    configuration = hooks.load_config(authenticated)
    attempt = hooks.reserve_attempt(authenticated, configuration)
    try:
        row = hooks.load_exact_task(authenticated)
        tokenized = hooks.tokenize_task(authenticated, row)
        model = hooks.load_weights(configuration, authenticated)
        result = hooks.evaluate(model, tokenized, authenticated, configuration)
        return hooks.finalize(
            result,
            model,
            tokenized,
            authenticated,
            configuration,
            attempt,
        )
    except BaseException as error:
        hooks.record_failure(attempt, error)
        raise


def _script_module(name: str) -> Any:
    try:
        return importlib.import_module(f"scripts.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_mapping(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise StageAAuthenticationError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageAAuthenticationError(
            f"cannot read authenticated JSON {path.name}: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StageAAuthenticationError(f"{path.name} must contain a JSON object")
    return value


def _git(repo_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"git {' '.join(arguments)} failed: {message}")
    return process.stdout.strip()


def _repository_state(repo_root: Path) -> dict[str, object]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    branch = _git(repo_root, "branch", "--show-current")
    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": not status,
        "porcelain": status.splitlines(),
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise StageAAuthenticationError(f"Stage-A source set is incomplete: {missing}")
    return {relative: _file_sha256(repo_root / relative) for relative in SOURCE_FILES}


def _assert_loaded_local_modules_declared(repo_root: Path) -> tuple[str, ...]:
    root = repo_root.resolve()
    declared = set(SOURCE_FILES)
    loaded: set[str] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if relative.startswith("src/recurquant/") or relative in {
            "scripts/capture_statelease_stage0.py",
            "scripts/screen_rht_cqer.py",
            "scripts/screen_statelease_stage_a.py",
            "scripts/verify_statelease_stage0.py",
        }:
            loaded.add(relative)
    omitted = sorted(loaded - declared)
    if omitted:
        raise StageAAuthenticationError(
            f"loaded local modules are absent from the frozen source set: {omitted}"
        )
    return tuple(sorted(loaded))


def _assert_sources_match_head(repo_root: Path) -> None:
    for relative in SOURCE_FILES:
        tree_entry = _git(repo_root, "ls-tree", "HEAD", "--", relative)
        parts = tree_entry.split(maxsplit=3)
        if len(parts) != 4 or parts[0] not in {"100644", "100755"} or parts[1] != "blob":
            raise StageAAuthenticationError(
                f"Stage-A source is not a tracked regular file at HEAD: {relative}"
            )
        head_blob = parts[2]
        worktree_blob = _git(
            repo_root,
            "hash-object",
            f"--path={relative}",
            "--",
            relative,
        )
        if worktree_blob != head_blob:
            raise StageAAuthenticationError(
                f"Stage-A source bytes differ from HEAD despite Git status: {relative}"
            )


def _assert_no_prior_stage_a_seal(repo_root: Path) -> None:
    history = _git(
        repo_root,
        "log",
        "--all",
        "--reflog",
        "--format=%B%x00",
    )
    if ONE_RUN_MARKER in history:
        raise StageAAuthenticationError(
            "the durable one-run seal is already present in repository history"
        )


def _assert_repository_start(repo_root: Path) -> tuple[dict[str, object], dict[str, str]]:
    state = _repository_state(repo_root)
    if state["worktree_clean"] is not True:
        raise StageAAuthenticationError(
            "Stage A requires a clean tracked and untracked repository before any access"
        )
    _assert_sources_match_head(repo_root)
    _assert_no_prior_stage_a_seal(repo_root)
    return state, _source_hashes(repo_root)


def _assert_ignored_exact_output(repo_root: Path) -> tuple[Path, Path]:
    output = (repo_root / OUTPUT_RELATIVE_PATH).resolve()
    attempt = (repo_root / ATTEMPT_RELATIVE_PATH).resolve()
    expected_output = (repo_root.resolve() / OUTPUT_RELATIVE_PATH).resolve()
    expected_attempt = (repo_root.resolve() / ATTEMPT_RELATIVE_PATH).resolve()
    if output != expected_output or attempt != expected_attempt:
        raise StageAAuthenticationError("Stage-A output identity is not exact")
    for path in (output, attempt):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=repo_root,
            check=False,
        )
        if ignored.returncode != 0:
            raise StageAAuthenticationError(
                f"one-run artifact path is not Git-ignored: {path.name}"
            )
        if path.exists():
            raise StageAAuthenticationError(
                f"one-run artifact already exists; refusing another Stage-A run: {path.name}"
            )
    return output, attempt


def _validate_anchor_payload(artifact: Mapping[str, object]) -> dict[str, object]:
    if set(artifact) != {
        "schema_version",
        "artifact_kind",
        "canonical_evidence_sha256",
        "evidence",
    }:
        raise StageAAuthenticationError("Experiment 009 Stage-A artifact schema drifted")
    if artifact["artifact_kind"] != "recurquant_rht_cqer32_stage_a_screen":
        raise StageAAuthenticationError("Experiment 009 Stage-A artifact kind drifted")
    if artifact["canonical_evidence_sha256"] != EXPERIMENT009_STAGE_A_CANONICAL_SHA256:
        raise StageAAuthenticationError("Experiment 009 canonical evidence digest drifted")
    evidence = artifact["evidence"]
    if not isinstance(evidence, Mapping):
        raise StageAAuthenticationError("Experiment 009 evidence must be an object")
    if evidence.get("screening_only") is not True:
        raise StageAAuthenticationError("Experiment 009 artifact is not a screening artifact")
    gate = evidence.get("stage_a_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise StageAAuthenticationError("Experiment 009 authenticated Stage-A gate did not pass")

    dataset = evidence.get("dataset")
    if not isinstance(dataset, Mapping):
        raise StageAAuthenticationError("Experiment 009 dataset evidence is missing")
    manifest = dataset.get("manifest")
    token_manifest = dataset.get("token_manifest")
    identity = dataset.get("identity")
    if manifest != EXPECTED_DATASET_MANIFEST:
        raise StageAAuthenticationError("Experiment 009 task-666 manifest drifted")
    if token_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError("Experiment 009 task-666 token manifest drifted")
    if identity != EXPECTED_TASK_IDENTITY:
        raise StageAAuthenticationError("Experiment 009 task-666 identity drifted")
    if dataset.get("manifest_sha256") != EXPERIMENT009_MANIFEST_SHA256:
        raise StageAAuthenticationError("Experiment 009 manifest receipt drifted")
    if _canonical_sha256(manifest) != EXPERIMENT009_MANIFEST_CANONICAL_SHA256:
        raise StageAAuthenticationError("Experiment 009 manifest bytes failed authentication")
    if _canonical_sha256(token_manifest) != EXPERIMENT009_TOKEN_MANIFEST_SHA256:
        raise StageAAuthenticationError("Experiment 009 token manifest failed authentication")
    if _canonical_sha256(identity) != EXPERIMENT009_IDENTITY_SHA256:
        raise StageAAuthenticationError("Experiment 009 task identity failed authentication")

    model = evidence.get("model")
    if not isinstance(model, Mapping) or {
        "id": model.get("id"),
        "revision": model.get("revision"),
        "dtype": model.get("dtype"),
    } != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": str(MODEL_DTYPE),
    }:
        raise StageAAuthenticationError("Experiment 009 model identity drifted")
    selectors = evidence.get("selector_artifacts")
    if not isinstance(selectors, Mapping):
        raise StageAAuthenticationError("Experiment 009 selector receipts are missing")
    expected_selector_receipts = {
        "authenticated": True,
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": SELECTOR_CANONICAL_SHA256,
        "loss_selector_canonical_evidence_sha256": LOSS_SELECTOR_CANONICAL_SHA256,
        "quota_sum": 1976,
    }
    for key, expected in expected_selector_receipts.items():
        if selectors.get(key) != expected:
            raise StageAAuthenticationError(f"Experiment 009 selector receipt drifted: {key}")
    return {
        "artifact": dict(artifact),
        "evidence": dict(evidence),
        "dataset_manifest": dict(manifest),
        "token_manifest": [dict(item) for item in token_manifest],
        "task_identity": dict(identity),
        "selector_receipts": dict(selectors),
    }


def authenticate_experiment009_anchor(repo_root: Path) -> dict[str, object]:
    path = repo_root / EXPERIMENT009_STAGE_A_RELATIVE_PATH
    if _file_sha256(path) != EXPERIMENT009_STAGE_A_FILE_SHA256:
        raise StageAAuthenticationError("committed Experiment 009 Stage-A file hash drifted")
    verification = verify_evidence_artifact(path)
    if verification.get("valid") is not True:
        raise StageAAuthenticationError(
            "committed Experiment 009 Stage-A artifact failed canonical verification"
        )
    if (
        verification.get("file_sha256") != EXPERIMENT009_STAGE_A_FILE_SHA256
        or verification.get("computed_canonical_evidence_sha256")
        != EXPERIMENT009_STAGE_A_CANONICAL_SHA256
    ):
        raise StageAAuthenticationError("Experiment 009 verification receipt drifted")
    result = _validate_anchor_payload(_json_mapping(path))
    result["path"] = EXPERIMENT009_STAGE_A_RELATIVE_PATH
    result["file_sha256"] = EXPERIMENT009_STAGE_A_FILE_SHA256
    result["canonical_evidence_sha256"] = EXPERIMENT009_STAGE_A_CANONICAL_SHA256
    return result


def authenticate_stage_a_identity_clarification(
    repo_root: Path,
) -> dict[str, object]:
    path = repo_root / IDENTITY_NOTE_RELATIVE_PATH
    if not path.is_file() or _file_sha256(path) != IDENTITY_NOTE_FILE_SHA256:
        raise StageAAuthenticationError(
            "Experiment 010 Stage-A identity clarification hash drifted"
        )
    return {
        "path": IDENTITY_NOTE_RELATIVE_PATH,
        "file_sha256": IDENTITY_NOTE_FILE_SHA256,
        "task_id": TASK_ID,
        "row_sha256": TASK_ROW_SHA256,
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "code_text_sha256": CODE_TEXT_SHA256,
        "prompt_token_ids_sha256": PROMPT_TOKEN_IDS_SHA256,
        "code_token_ids_sha256": CODE_TOKEN_IDS_SHA256,
        "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
        "identity_only": True,
        "model_weights_or_quality_accessed": False,
    }


def authenticate_selectors(
    repo_root: Path,
    anchor: Mapping[str, object],
) -> tuple[dict[str, object], object]:
    selector_path = repo_root / SELECTOR_RELATIVE_PATH
    loss_selector_path = repo_root / LOSS_SELECTOR_RELATIVE_PATH
    actual_hashes = {
        "selector_file_sha256": _file_sha256(selector_path),
        "loss_selector_file_sha256": _file_sha256(loss_selector_path),
    }
    expected_hashes = {
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise StageAAuthenticationError("frozen selector artifact file hashes drifted")
    screen009 = _script_module("screen_rht_cqer")
    try:
        _selector, _loss_selector, hashes, plan = screen009._authenticate_selectors(  # noqa: SLF001
            selector_path,
            loss_selector_path,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise StageAAuthenticationError(
            f"selector semantic authentication failed: {error}"
        ) from error
    expected = {
        **expected_hashes,
        "selector_canonical_evidence_sha256": SELECTOR_CANONICAL_SHA256,
        "loss_selector_canonical_evidence_sha256": LOSS_SELECTOR_CANONICAL_SHA256,
    }
    if hashes != expected:
        raise StageAAuthenticationError("selector canonical identities drifted")
    receipts = anchor.get("selector_receipts")
    if not isinstance(receipts, Mapping) or any(
        receipts.get(key) != value for key, value in expected.items()
    ):
        raise StageAAuthenticationError(
            "selector identities do not match the committed task-666 anchor"
        )
    identity = {
        **expected,
        "selector_path": SELECTOR_RELATIVE_PATH,
        "loss_selector_path": LOSS_SELECTOR_RELATIVE_PATH,
        "target_fisher_score": receipts.get("target_fisher_score"),
        "target_fisher_layer_quotas": receipts.get("target_fisher_layer_quotas"),
        "quota_sum": receipts.get("quota_sum"),
        "authenticated": True,
    }
    return identity, plan


def authenticate_stage0(
    artifact_path: Path,
    sha256_path: Path | None,
    *,
    expected_repo_head: str | None = None,
) -> tuple[dict[str, object], Path, Path]:
    artifact = artifact_path.resolve()
    sidecar = (
        artifact.with_suffix(artifact.suffix + ".sha256")
        if sha256_path is None
        else sha256_path.resolve()
    )
    verifier = _script_module("verify_statelease_stage0")
    try:
        report = verifier.verify_production_stage0(
            artifact,
            sha256_path=sidecar,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise StageAAuthenticationError(
            f"production Stage 0 failed independent verification: {error}"
        ) from error
    required = {
        "status": "production_stage0_pass",
        "experiment_stage0_complete": True,
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
    }
    if not isinstance(report, dict) or any(
        report.get(key) != value for key, value in required.items()
    ):
        raise StageAAuthenticationError(
            "independent verifier did not authenticate a complete synthetic-only Stage 0"
        )
    if expected_repo_head is not None:
        artifact_head = report.get("repository_commit")
        if artifact_head != expected_repo_head:
            raise StageAAuthenticationError(
                "production Stage-0 repository HEAD does not equal the current Stage-A HEAD"
            )
    report = dict(report)
    report["artifact"] = artifact.name
    report["sidecar"] = sidecar.name
    report["sidecar_file_sha256"] = _file_sha256(sidecar)
    return report, artifact, sidecar


def authenticate_static_inputs(args: argparse.Namespace) -> StageAPreflight:
    repo_root = Path(__file__).resolve().parents[1]
    repository_start, source_hashes_start = _assert_repository_start(repo_root)
    output_path, attempt_path = _assert_ignored_exact_output(repo_root)
    identity_clarification = authenticate_stage_a_identity_clarification(repo_root)
    anchor = authenticate_experiment009_anchor(repo_root)
    selector_identity, plan = authenticate_selectors(repo_root, anchor)
    stage0, stage0_artifact, stage0_sha256 = authenticate_stage0(
        args.stage0_artifact,
        args.stage0_sha256,
        expected_repo_head=str(repository_start["commit"]),
    )
    _assert_loaded_local_modules_declared(repo_root)
    return StageAPreflight(
        repo_root=repo_root,
        repository_start=repository_start,
        source_hashes_start=source_hashes_start,
        identity_clarification=identity_clarification,
        anchor=anchor,
        selector_identity=selector_identity,
        plan=plan,
        stage0=stage0,
        stage0_artifact=stage0_artifact,
        stage0_sha256=stage0_sha256,
        output_path=output_path,
        attempt_path=attempt_path,
    )


def _validate_model_geometry(config: object) -> dict[str, object]:
    getter = getattr(config, "get_text_config", None)
    text_config = getter(decoder=True) if callable(getter) else config
    layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        raise StageAAuthenticationError("pinned Qwen configuration lacks layer_types")
    linear_indices = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    ]
    geometry = {
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "recurrent_layers": len(linear_indices),
        "linear_layer_indices": linear_indices,
        "value_heads": getattr(text_config, "linear_num_value_heads", None),
        "key_rows": getattr(text_config, "linear_key_head_dim", None),
        "value_width": getattr(text_config, "linear_value_head_dim", None),
        "batch_size": 1,
        "recurrent_state_dtype": "torch.float32",
    }
    if geometry != EXPECTED_GEOMETRY:
        raise StageAAuthenticationError(f"pinned Qwen recurrent geometry drifted: {geometry}")
    return geometry


def load_and_authenticate_config(
    preflight: StageAPreflight,
    *,
    local_files_only: bool,
) -> ModelConfiguration:
    del preflight
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    geometry = _validate_model_geometry(config)
    _assert_loaded_local_modules_declared(Path(__file__).resolve().parents[1])
    return ModelConfiguration(
        config=config,
        identity={
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": str(MODEL_DTYPE),
            "attn_implementation": "eager",
            "geometry": geometry,
            "configuration_authenticated_before_task_or_model_weights": True,
        },
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise StageAAuthenticationError(
            f"refusing to overwrite one-run artifact {path.name}"
        ) from error


def _atomic_replace_owned(path: Path, payload: bytes) -> None:
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


def _commit_tree(
    repo_root: Path,
    *,
    tree: str,
    parent: str,
    message: str,
) -> str:
    environment = dict(os.environ)
    name = _git(repo_root, "config", "--get", "user.name")
    email = _git(repo_root, "config", "--get", "user.email")
    if not name or not email or "\n" in name or "\n" in email:
        raise StageAAuthenticationError(
            "repository Git user.name and user.email are required for the one-run seal"
        )
    identity = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }
    environment.update(identity)
    process = subprocess.run(
        ["git", "commit-tree", tree, "-p", parent],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        input=f"{message}\n",
        env=environment,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"cannot create one-run seal commit: {detail}")
    commit = process.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise StageAAuthenticationError("one-run seal commit identity is malformed")
    return commit


def _seal_message(preflight: StageAPreflight, configuration: ModelConfiguration) -> str:
    payload = {
        "schema": "recurquant.experiment010.stage-a-one-run-seal.v1",
        "marker": ONE_RUN_MARKER,
        "h0_commit": preflight.repository_start["commit"],
        "source_set_sha256": _canonical_sha256(preflight.source_hashes_start),
        "identity_note_file_sha256": IDENTITY_NOTE_FILE_SHA256,
        "experiment009_file_sha256": EXPERIMENT009_STAGE_A_FILE_SHA256,
        "stage0_artifact_file_sha256": preflight.stage0.get("artifact_file_sha256"),
        "stage0_sidecar_file_sha256": preflight.stage0.get("sidecar_file_sha256"),
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "model": configuration.identity,
        "task_id": TASK_ID,
        "quality_data_accessed_before_seal": False,
        "model_weights_loaded_before_seal": False,
    }
    payload_bytes = canonical_json_bytes(payload)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    return (
        "chore: reserve Experiment 010 Stage-A one-run\n\n"
        f"{ONE_RUN_MARKER}\n"
        f"RecurQuant-One-Run-Payload-SHA256: {payload_hash}\n\n"
        f"{payload_bytes.decode('utf-8')}"
    ).rstrip("\r\n")


def _validate_one_run_seal(
    preflight: StageAPreflight,
    reservation: AttemptReservation,
    *,
    require_receipt: bool,
) -> dict[str, object]:
    state = _repository_state(preflight.repo_root)
    if state["worktree_clean"] is not True:
        raise StageAAuthenticationError("repository became dirty after one-run sealing")
    if state["commit"] != reservation.seal_commit:
        raise StageAAuthenticationError("repository no longer points at the one-run seal")
    if _git(preflight.repo_root, "show", "-s", "--format=%P", reservation.seal_commit) != (
        reservation.h0_commit
    ):
        raise StageAAuthenticationError("one-run seal parent drifted")
    actual_tree = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%T",
        reservation.seal_commit,
    )
    h0_tree = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%T",
        reservation.h0_commit,
    )
    if actual_tree != reservation.tree or h0_tree != reservation.tree:
        raise StageAAuthenticationError("one-run seal is not an empty-tree commit")
    message = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%B",
        reservation.seal_commit,
    )
    model_identity = reservation.receipt.get("model")
    if not isinstance(model_identity, Mapping):
        raise StageAAuthenticationError("one-run receipt model identity is malformed")
    expected_message = _seal_message(
        preflight,
        ModelConfiguration(config=None, identity=dict(model_identity)),
    )
    if (
        message != expected_message
        or ONE_RUN_MARKER not in message
        or hashlib.sha256(message.encode("utf-8")).hexdigest() != reservation.seal_message_sha256
    ):
        raise StageAAuthenticationError("one-run seal message drifted")
    _assert_sources_match_head(preflight.repo_root)
    if _source_hashes(preflight.repo_root) != preflight.source_hashes_start:
        raise StageAAuthenticationError("source hashes changed after one-run sealing")
    if require_receipt:
        try:
            payload = preflight.attempt_path.read_bytes()
        except OSError as error:
            raise StageAAuthenticationError("reserved attempt receipt is missing") from error
        if hashlib.sha256(payload).hexdigest() != reservation.receipt_file_sha256:
            raise StageAAuthenticationError("reserved attempt receipt bytes drifted")
    return {
        "h0_commit": reservation.h0_commit,
        "seal_commit": reservation.seal_commit,
        "tree": reservation.tree,
        "seal_message_sha256": reservation.seal_message_sha256,
        "empty_tree_commit": True,
        "durable_one_run_marker": ONE_RUN_MARKER,
        "local_seal_limitation": ONE_RUN_LIMITATION,
    }


def reserve_one_run(
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
) -> AttemptReservation:
    h0_commit = str(preflight.repository_start["commit"])
    if _git(preflight.repo_root, "rev-parse", "HEAD") != h0_commit:
        raise StageAAuthenticationError("repository HEAD changed before one-run sealing")
    tree = _git(preflight.repo_root, "show", "-s", "--format=%T", h0_commit)
    message = _seal_message(preflight, configuration)
    seal_commit = _commit_tree(
        preflight.repo_root,
        tree=tree,
        parent=h0_commit,
        message=message,
    )
    process = subprocess.run(
        ["git", "update-ref", "HEAD", seal_commit, h0_commit],
        cwd=preflight.repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"one-run HEAD compare-and-swap failed: {detail}")
    receipt = {
        "schema": "recurquant.experiment010.stage-a-attempt.v1",
        "status": "reserved_before_quality_data_or_model_weights",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attempt_number": 1,
        "task_id": TASK_ID,
        "output_path": OUTPUT_RELATIVE_PATH,
        "h0_repository_commit": h0_commit,
        "one_run_seal_commit": seal_commit,
        "one_run_seal_tree": tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_limitation": ONE_RUN_LIMITATION,
        "source_hashes": preflight.source_hashes_start,
        "experiment009_file_sha256": EXPERIMENT009_STAGE_A_FILE_SHA256,
        "stage0_artifact_file_sha256": preflight.stage0.get("artifact_file_sha256"),
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "model": configuration.identity,
        "completed_task_ids": [],
        "quality_aggregate_exposed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt_bytes = canonical_json_bytes(receipt)
    _exclusive_write(preflight.attempt_path, receipt_bytes)
    reservation = AttemptReservation(
        receipt=receipt,
        receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        h0_commit=h0_commit,
        seal_commit=seal_commit,
        tree=tree,
        seal_message_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
    )
    _validate_one_run_seal(preflight, reservation, require_receipt=True)
    return reservation


def record_attempt_failure(
    preflight: StageAPreflight,
    receipt: object,
    error: BaseException,
) -> None:
    if not isinstance(receipt, AttemptReservation):
        return
    _validate_one_run_seal(preflight, receipt, require_receipt=True)
    failed = {
        **receipt.receipt,
        "status": "failed_without_authenticated_stage_a_result",
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "completed_task_ids": [],
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
    }
    _atomic_replace_owned(preflight.attempt_path, canonical_json_bytes(failed))


def load_exact_authenticated_task(preflight: StageAPreflight) -> Mapping[str, object]:
    del preflight
    from recurquant.public_data import load_mbpp_rows_by_task_ids, mbpp_row_sha256

    verifier = _script_module("verify_statelease_stage0")
    verifier.guard_protected_mbpp_window(
        stage="stagea",
        task_ids=(str(TASK_ID),),
        contains_quality_data=True,
    )
    rows = load_mbpp_rows_by_task_ids("calibration", task_ids=(TASK_ID,))
    if len(rows) != 1 or int(rows[0].get("task_id", -1)) != TASK_ID:
        raise StageAAuthenticationError(
            "exact task-ID loader returned an unauthorized MBPP identity"
        )
    row = rows[0]
    if mbpp_row_sha256(row) != TASK_ROW_SHA256:
        raise StageAAuthenticationError("already-open task-666 row content drifted")
    return row


def tokenize_authenticated_task(
    preflight: StageAPreflight,
    row: Mapping[str, object],
    *,
    local_files_only: bool,
) -> TokenizedTask:
    from transformers import AutoTokenizer

    from recurquant.public_data import format_mbpp_example

    formatted = format_mbpp_example(row)
    if hashlib.sha256(formatted.prompt.encode("utf-8")).hexdigest() != PROMPT_TEXT_SHA256:
        raise StageAAuthenticationError("task-666 formatted prompt text drifted")
    if hashlib.sha256(formatted.code.encode("utf-8")).hexdigest() != CODE_TEXT_SHA256:
        raise StageAAuthenticationError("task-666 formatted code text drifted")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.__class__.__name__ != "Qwen2Tokenizer":
        raise StageAAuthenticationError("pinned tokenizer class drifted")
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
    token_manifest = [
        {
            "task_id": TASK_ID,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "code_tokens": int(code_ids.shape[1]),
            "aligned_scored_tokens": int(code_ids.shape[1]) - 1,
            "full_code_scored_tokens": int(code_ids.shape[1]),
        }
    ]
    if token_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError(
            "tokenized task 666 does not match the authenticated token manifest"
        )
    copied_manifest = preflight.anchor.get("token_manifest")
    if copied_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError(
            "runtime token identity differs from the copied Experiment 009 manifest"
        )
    if int(row.get("task_id", -1)) != TASK_ID:
        raise StageAAuthenticationError("encoded row identity drifted")
    _validate_token_id_hashes(prompt_ids, code_ids)
    return TokenizedTask(
        row=row,
        prompt_ids=prompt_ids,
        code_ids=code_ids,
        token_manifest=[dict(item) for item in token_manifest],
    )


def _canonical_token_ids_sha256(value: torch.Tensor) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.long
        or value.ndim != 2
        or value.shape[0] != 1
    ):
        raise StageAAuthenticationError("token IDs must be one batch of torch.long IDs")
    ids = [int(item) for item in value.detach().to("cpu").reshape(-1).tolist()]
    return hashlib.sha256(canonical_json_bytes(ids)).hexdigest()


def _validate_token_id_hashes(
    prompt_ids: torch.Tensor,
    code_ids: torch.Tensor,
) -> dict[str, object]:
    if tuple(prompt_ids.shape) != (1, PROMPT_TOKENS):
        raise StageAAuthenticationError("prompt token shape drifted")
    if tuple(code_ids.shape) != (1, CODE_TOKENS):
        raise StageAAuthenticationError("code token shape drifted")
    prompt_hash = _canonical_token_ids_sha256(prompt_ids)
    code_hash = _canonical_token_ids_sha256(code_ids)
    if prompt_hash != PROMPT_TOKEN_IDS_SHA256:
        raise StageAAuthenticationError("prompt token-ID hash drifted")
    if code_hash != CODE_TOKEN_IDS_SHA256:
        raise StageAAuthenticationError("code token-ID hash drifted")
    return {
        "prompt_token_ids_sha256": prompt_hash,
        "code_token_ids_sha256": code_hash,
        "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
        "prompt_shape": list(prompt_ids.shape),
        "code_shape": list(code_ids.shape),
    }


def _select_cuda_device(requested: str) -> torch.device:
    if requested not in {"auto", "cuda"}:
        raise StageAAuthenticationError("Stage A supports only the pinned CUDA BF16 path")
    if not torch.cuda.is_available():
        raise StageAAuthenticationError("the frozen Stage-A CUDA device is unavailable")
    return torch.device("cuda")


def load_model_weights(
    configuration: ModelConfiguration,
    preflight: StageAPreflight,
    *,
    device_name: str,
    local_files_only: bool,
) -> tuple[torch.nn.Module, torch.device]:
    del preflight
    from transformers import Qwen3_5ForCausalLM

    device = _select_cuda_device(device_name)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=local_files_only,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    loaded_geometry = _validate_model_geometry(model.config)
    if loaded_geometry != configuration.identity["geometry"]:
        raise StageAAuthenticationError(
            "loaded model geometry differs from the pre-weight configuration"
        )
    return model, device


@dataclass(slots=True)
class _AlignedAccumulator:
    kl: list[torch.Tensor]
    reference_nll: list[torch.Tensor]
    candidate_nll: list[torch.Tensor]
    top1: list[torch.Tensor]
    finite: list[bool]
    records: list[dict[str, object]]

    @classmethod
    def empty(cls) -> _AlignedAccumulator:
        return cls([], [], [], [], [], [])

    def append(
        self,
        *,
        token_index: int,
        reference_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        input_token: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        reference = reference_logits.detach().to("cpu")
        candidate = candidate_logits.detach().to("cpu")
        input_cpu = input_token.detach().to("cpu")
        target_cpu = target.detach().to("cpu")
        logits_finite = bool(
            torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item()
        )
        values = token_fidelity(reference, candidate, target_cpu).to_cpu()
        scalars = {
            "kl": float(values.kl.reshape(-1)[0].item()),
            "reference_nll": float(values.reference_nll.reshape(-1)[0].item()),
            "candidate_nll": float(values.candidate_nll.reshape(-1)[0].item()),
            "top1_agreement": bool(values.top1_agreement.reshape(-1)[0].item()),
        }
        if not logits_finite or not all(
            math.isfinite(float(value)) for key, value in scalars.items() if key != "top1_agreement"
        ):
            raise RuntimeError(f"aligned token {token_index} produced a non-finite value")
        self.kl.append(values.kl.reshape(-1))
        self.reference_nll.append(values.reference_nll.reshape(-1))
        self.candidate_nll.append(values.candidate_nll.reshape(-1))
        self.top1.append(values.top1_agreement.reshape(-1))
        self.finite.append(logits_finite)
        self.records.append(
            {
                "write_index": token_index,
                "input_token_id": int(input_cpu.reshape(-1)[0].item()),
                "target_token_id": int(target_cpu.reshape(-1)[0].item()),
                **scalars,
                "delta_nll": scalars["candidate_nll"] - scalars["reference_nll"],
                "all_logits_finite": logits_finite,
            }
        )

    def summary(self) -> dict[str, float | int | bool]:
        if not self.records:
            raise RuntimeError("aligned accumulator is empty")
        summary = fidelity_summary(
            TokenFidelity(
                kl=torch.cat(self.kl),
                reference_nll=torch.cat(self.reference_nll),
                candidate_nll=torch.cat(self.candidate_nll),
                top1_agreement=torch.cat(self.top1),
            )
        )
        return {**summary, "all_logits_finite": all(self.finite)}


def _snapshot_states_cpu(cache: object) -> dict[int, torch.Tensor]:
    checkpoint = getattr(cache, "checkpoint", None)
    layout = getattr(cache, "layout", None)
    materialize = getattr(checkpoint, "materialize", None)
    if checkpoint is not None and layout is not None and callable(materialize):
        states = materialize()
        if isinstance(states, Mapping):
            return {
                int(layer): tensor.detach().to("cpu").clone() for layer, tensor in states.items()
            }
    result: dict[int, torch.Tensor] = {}
    for state in iter_recurrent_states(cache):
        if state.state_index != 0:
            raise RuntimeError("Stage A requires one recurrent state per layer")
        if state.layer_index in result:
            raise RuntimeError("Stage A observed a duplicate recurrent layer")
        result[state.layer_index] = state.tensor.detach().to("cpu").clone()
    if tuple(sorted(result)) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"recurrent-state snapshot layers drifted: {sorted(result)}")
    return result


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().to("cpu")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_snapshot_digest(states: Mapping[int, torch.Tensor]) -> str:
    return _canonical_sha256(
        {
            str(layer): {
                "dtype": str(states[layer].dtype),
                "shape": list(states[layer].shape),
                "sha256": _tensor_digest(states[layer]),
            }
            for layer in sorted(states)
        }
    )


@contextlib.contextmanager
def _capture_prefill_writes(cache: object) -> Iterator[dict[int, torch.Tensor]]:
    original = getattr(cache, "update_recurrent_state", None)
    if not callable(original):
        raise TypeError("candidate cache does not expose update_recurrent_state")
    captured: dict[int, torch.Tensor] = {}

    def wrapped(
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        if state_idx != 0:
            raise RuntimeError("Stage A supports only recurrent state index zero")
        layer_index = int(layer_idx)
        if layer_index in captured:
            raise RuntimeError(f"prefill wrote recurrent layer {layer_index} twice")
        captured[layer_index] = recurrent_states.detach().to("cpu").clone()
        return original(
            recurrent_states,
            layer_idx,
            state_idx=state_idx,
            **kwargs,
        )

    cache.update_recurrent_state = wrapped  # type: ignore[attr-defined]
    try:
        yield captured
    finally:
        cache.update_recurrent_state = original  # type: ignore[attr-defined]


def _assert_shared_prefill(
    reference: Mapping[int, torch.Tensor],
    candidate: Mapping[int, torch.Tensor],
    *,
    method: str,
) -> dict[str, object]:
    expected_layers = set(LINEAR_LAYER_INDICES)
    if set(reference) != expected_layers or set(candidate) != expected_layers:
        raise RuntimeError(f"{method} prefill layer set differs from FP32")
    expected_shape = (
        1,
        int(EXPECTED_GEOMETRY["value_heads"]),
        int(EXPECTED_GEOMETRY["key_rows"]),
        int(EXPECTED_GEOMETRY["value_width"]),
    )
    for layer_index in sorted(reference):
        reference_tensor = reference[layer_index]
        candidate_tensor = candidate[layer_index]
        if (
            not isinstance(reference_tensor, torch.Tensor)
            or not isinstance(candidate_tensor, torch.Tensor)
            or reference_tensor.dtype != torch.float32
            or candidate_tensor.dtype != torch.float32
            or tuple(reference_tensor.shape) != expected_shape
            or tuple(candidate_tensor.shape) != expected_shape
        ):
            raise RuntimeError(
                f"{method} prefill layer {layer_index} is not the exact raw FP32 geometry"
            )
        if not torch.equal(reference_tensor, candidate_tensor):
            maximum = float((reference_tensor - candidate_tensor).abs().max().item())
            raise RuntimeError(
                f"{method} did not pack the identical FP32 prefill state "
                f"at layer {layer_index}; max_abs={maximum}"
            )
    return {
        "identical_raw_fp32_prefill_state": True,
        "layer_count": len(reference),
        "reference_digest": _state_snapshot_digest(reference),
        "candidate_digest": _state_snapshot_digest(candidate),
    }


def _snapshot_nonrecurrent_prefill(cache: object) -> dict[str, torch.Tensor]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or len(layers) != EXPECTED_GEOMETRY["num_hidden_layers"]:
        raise RuntimeError("prefill cache does not expose the frozen 24-layer geometry")
    result: dict[str, torch.Tensor] = {}
    linear = set(LINEAR_LAYER_INDICES)
    for layer_index, layer in enumerate(layers):
        if layer_index in linear:
            initialized = getattr(layer, "is_conv_states_initialized", None)
            conv_states = getattr(layer, "conv_states", None)
            try:
                is_initialized = bool(initialized[0])
                tensor = conv_states[0]
            except (IndexError, KeyError, TypeError, AttributeError) as error:
                raise RuntimeError(
                    f"linear layer {layer_index} lacks an auditable convolution cache"
                ) from error
            if (
                not is_initialized
                or not isinstance(tensor, torch.Tensor)
                or tensor.shape[0] != 1
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise RuntimeError(
                    f"linear layer {layer_index} convolution prefill cache is invalid"
                )
            result[f"layer.{layer_index}.conv.0"] = tensor.detach().to("cpu").clone()
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape[0] != 1
                or tensor.shape[-2] != PROMPT_TOKENS
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise RuntimeError(
                    f"full-attention layer {layer_index} {name} prefill cache is invalid"
                )
            result[f"layer.{layer_index}.{name}"] = tensor.detach().to("cpu").clone()
    expected_count = len(LINEAR_LAYER_INDICES) + 2 * (
        int(EXPECTED_GEOMETRY["num_hidden_layers"]) - len(LINEAR_LAYER_INDICES)
    )
    if len(result) != expected_count:
        raise RuntimeError("non-recurrent prefill cache tensor count drifted")
    return result


def _assert_shared_nonrecurrent_prefill(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    method: str,
) -> dict[str, object]:
    if set(reference) != set(candidate):
        raise RuntimeError(f"{method} non-recurrent prefill cache keys differ from FP32")
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left, right)
        ):
            raise RuntimeError(f"{method} non-recurrent prefill cache differs from FP32 at {name}")
    digest = _state_snapshot_digest(
        {index: reference[name] for index, name in enumerate(sorted(reference))}
    )
    return {
        "identical_full_attention_kv_and_convolution_prefill_state": True,
        "nonrecurrent_tensor_count": len(reference),
        "nonrecurrent_reference_digest": digest,
        "nonrecurrent_candidate_digest": _state_snapshot_digest(
            {index: candidate[name] for index, name in enumerate(sorted(candidate))}
        ),
    }


def _validate_prefill_cache_contract(
    method: str,
    cache: object,
) -> dict[str, object]:
    if method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        diagnostics = _diagnostics(cache)
        if len(diagnostics) != len(LINEAR_LAYER_INDICES):
            raise RuntimeError(f"{method} prefill diagnostics omitted recurrent layers")
        for row in diagnostics:
            if (
                row.get("state_updates") != 1
                or row.get("tokens_observed") != PROMPT_TOKENS
                or row.get("replay_valid_count") != 0
            ):
                raise RuntimeError(
                    f"{method} did not start decode from one packed prefill "
                    "checkpoint and an empty replay buffer"
                )
        records = _evidence_records(cache)
        if len(records) != len(LINEAR_LAYER_INDICES):
            raise RuntimeError(f"{method} prefill evidence count drifted")
        if any(
            record.get("token_count") != PROMPT_TOKENS
            or record.get("boundary") is not None
            or record.get("replay_valid_count") != 0
            or "prefill" not in str(record.get("action"))
            for record in records
        ):
            raise RuntimeError(f"{method} prefill checkpoint receipt drifted")
        return {
            "packed_prefill_checkpoint": True,
            "empty_replay_buffer": True,
            "prefill_evidence_records": len(records),
        }
    if method in EQUAL_BYTE_NO_REPLAY_METHODS:
        storage = _storage_summary(cache)
        records = _evidence_records(cache)
        if (
            storage.get("checkpoint_present") is not True
            or storage.get("update_count") != 1
            or storage.get("successful_tokens") != PROMPT_TOKENS
            or len(records) != 1
            or records[0].get("previous_checkpoint_present") is not False
            or records[0].get("token_count") != PROMPT_TOKENS
        ):
            raise RuntimeError(f"{method} did not install exactly one prefill checkpoint")
        return {
            "packed_prefill_checkpoint": True,
            "no_replay_buffer": True,
            "prefill_evidence_records": 1,
        }
    storage = _storage_summary(cache)
    if (
        storage.get("resident_bytes") != 2_564_096
        or storage.get("resident_bytes_including_selector") != 2_711_552
        or storage.get("high_precision_groups") != 1_976
    ):
        raise RuntimeError("historical RHT-CQER prefill storage identity drifted")
    return {
        "packed_prefill_checkpoint": True,
        "historical_anchor": True,
        "resident_bytes_including_selector": 2_711_552,
    }


def _evidence_records(cache: object) -> list[dict[str, object]]:
    records = getattr(cache, "update_evidence", ())
    result: list[dict[str, object]] = []
    for index, record in enumerate(records):
        converter = getattr(record, "evidence_dict", None)
        value = converter() if callable(converter) else record
        if not isinstance(value, Mapping):
            raise RuntimeError(f"cache evidence record {index} is not a mapping")
        result.append(_jsonable(dict(value)))
    return result


def _storage_summary(cache: object) -> dict[str, object]:
    callback = getattr(cache, "storage_summary", None)
    if not callable(callback):
        raise RuntimeError("candidate cache does not expose storage_summary")
    value = callback()
    if not isinstance(value, Mapping):
        raise RuntimeError("candidate storage summary is not a mapping")
    return _jsonable(dict(value))


TensorStorageKey = tuple[str, int | None, int, int]


def _expected_shared_tensor_names() -> frozenset[str]:
    linear = set(LINEAR_LAYER_INDICES)
    names = {f"layer.{layer_index}.conv.0" for layer_index in LINEAR_LAYER_INDICES}
    for layer_index in range(24):
        if layer_index not in linear:
            names.add(f"layer.{layer_index}.keys")
            names.add(f"layer.{layer_index}.values")
    return frozenset(names)


def _expected_candidate_tensor_names(method: str) -> frozenset[str]:
    if method == ORIGINAL_RHT_METHOD:
        per_layer = (
            "checkpoint.low_payload",
            "checkpoint.high_payload",
            "checkpoint.scales",
            "checkpoint.precision_mask",
            "query_energy_ema",
        )
        return frozenset(
            f"layer.{layer_index}.{field}"
            for layer_index in LINEAR_LAYER_INDICES
            for field in per_layer
        )
    if method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        per_layer = (
            "checkpoint.low_payload",
            "checkpoint.high_payload",
            "checkpoint.scales",
            "checkpoint.precision_mask",
            "query_energy_ema",
            "normalized_key_buffer",
            "update_buffer",
            "log_decay_buffer",
            "valid_count",
        )
        return frozenset(
            f"layer.{layer_index}.{field}"
            for layer_index in LINEAR_LAYER_INDICES
            for field in per_layer
        )
    equal_byte_fields = {
        "expanded_rht_q4_q8": (
            "q4_payload",
            "q8_payload",
            "scales",
            "precision_mask",
        ),
        "rht_q4_q6_q8": (
            "q4_payload",
            "q6_payload",
            "q8_payload",
            "scales",
            "precision_codes",
        ),
        "rht_residual_q4": (
            "base_q4_payload",
            "base_scales",
            "residual_q4_payload",
            "residual_scales",
            "lease_mask",
        ),
    }
    fields = equal_byte_fields.get(method)
    if fields is None:
        raise ValueError(f"{method} has no expected candidate tensor schema")
    names = {
        f"checkpoint.layer_{position}.{field}"
        for position in range(len(LINEAR_LAYER_INDICES))
        for field in fields
    }
    names.update(
        {
            "checkpoint.query_energy_ema",
            "checkpoint.reserved_padding",
        }
    )
    return frozenset(names)


EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES = _expected_shared_tensor_names()
EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT = len(EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES)
EXPECTED_CANDIDATE_PERSISTENT_TENSOR_COUNTS = {
    method: len(_expected_candidate_tensor_names(method)) for method in QUALITY_METHODS
}


def _tensor_storage_key(tensor: torch.Tensor) -> TensorStorageKey:
    if tensor.numel() == 0:
        raise RuntimeError("persistent tensor storage cannot be empty")
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        int(storage.nbytes()),
    )


def _tensor_schema(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": int(tensor.untyped_storage().nbytes()),
    }


def _shared_cache_tensors(cache: object) -> dict[str, torch.Tensor]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or len(layers) != 24:
        raise RuntimeError("cache does not expose the frozen 24-layer tensor layout")
    shared: dict[str, torch.Tensor] = {}
    linear = set(LINEAR_LAYER_INDICES)
    for layer_index, layer in enumerate(layers):
        if layer_index in linear:
            initialized = getattr(layer, "is_conv_states_initialized", None)
            states = getattr(layer, "conv_states", None)
            try:
                tensor = states[0]
                ready = bool(initialized[0])
            except (IndexError, KeyError, TypeError, AttributeError) as error:
                raise RuntimeError(
                    f"linear layer {layer_index} lacks shared convolution storage"
                ) from error
            if not ready or not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"linear layer {layer_index} convolution storage is not initialized"
                )
            shared[f"layer.{layer_index}.conv.0"] = tensor
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"full-attention layer {layer_index} lacks shared {name} storage"
                )
            shared[f"layer.{layer_index}.{name}"] = tensor
        sliding = getattr(layer, "_sliding_window_tensor", None)
        if isinstance(sliding, torch.Tensor):
            shared[f"layer.{layer_index}.sliding_window"] = sliding
    keys = [_tensor_storage_key(tensor) for tensor in shared.values()]
    if len(keys) != len(set(keys)):
        raise RuntimeError("shared cache tensors unexpectedly alias persistent storage")
    return shared


def _shared_cache_schema(cache: object) -> dict[str, dict[str, object]]:
    return {
        name: _tensor_schema(tensor)
        for name, tensor in sorted(_shared_cache_tensors(cache).items())
    }


def _assert_shared_cache_schema(
    cache: object,
    reference_schema: Mapping[str, object],
    *,
    method: str,
) -> dict[str, dict[str, object]]:
    actual = _shared_cache_schema(cache)
    if actual != reference_schema:
        raise RuntimeError(
            f"{method} post-decode shared KV/convolution storage schema differs from FP32"
        )
    return actual


def _statelease_candidate_tensors(
    cache: object,
    *,
    method: str,
) -> dict[str, torch.Tensor]:
    from recurquant.mixed_quantization import PackedMixedQuantizedTensor
    from recurquant.statelease_baselines import (
        FixedCC1RecurrentStateCache,
        FixedCC2RecurrentStateCache,
        FixedCC4RecurrentStateCache,
        FixedCC5RecurrentStateCache,
        FixedCut4In5RecurrentStateCache,
        FixedReplayLinearAttentionLayer,
    )
    from recurquant.statelease_cache import (
        StateLeaseLinearAttentionLayer,
        StateLeaseRecurrentStateCache,
        _StateLeaseStateView,
    )

    expected_cache_types = {
        STATELEASE_METHOD: StateLeaseRecurrentStateCache,
        "fixed_cc1": FixedCC1RecurrentStateCache,
        "fixed_cc2": FixedCC2RecurrentStateCache,
        "fixed_cc4": FixedCC4RecurrentStateCache,
        "fixed_cc5": FixedCC5RecurrentStateCache,
        "fixed_cut4_in5": FixedCut4In5RecurrentStateCache,
    }
    expected_cache = expected_cache_types[method]
    if type(cache) is not expected_cache:
        raise RuntimeError(
            f"{method} cache type drifted: {type(cache).__module__}.{type(cache).__qualname__}"
        )
    layers = tuple(cache.statelease_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"{method} recurrent layer identities drifted")
    expected_layer = (
        StateLeaseLinearAttentionLayer
        if method == STATELEASE_METHOD
        else FixedReplayLinearAttentionLayer
    )
    result: dict[str, torch.Tensor] = {}
    for layer_index, layer in layers:
        if type(layer) is not expected_layer or type(layer.recurrent_states) is not (
            _StateLeaseStateView
        ):
            raise RuntimeError(f"{method} layer {layer_index} type or state view drifted")
        packed = layer.packed_checkpoint
        if type(packed) is not PackedMixedQuantizedTensor:
            raise RuntimeError(f"{method} layer {layer_index} checkpoint type drifted")
        assert isinstance(packed, PackedMixedQuantizedTensor)
        if tuple(packed.original_shape) != (1, 16, 128, 128):
            raise RuntimeError(f"{method} layer {layer_index} checkpoint geometry drifted")
        packed_fields = {
            "low_payload": packed.low_payload,
            "high_payload": packed.high_payload,
            "scales": packed.scales,
            "precision_mask": packed.precision_mask,
        }
        expected_dtypes = {
            "low_payload": torch.uint8,
            "high_payload": torch.int8,
            "scales": torch.float16,
            "precision_mask": torch.uint8,
        }
        for name, tensor in packed_fields.items():
            if tensor.dtype != expected_dtypes[name]:
                raise RuntimeError(f"{method} layer {layer_index} checkpoint {name} dtype drifted")
            result[f"layer.{layer_index}.checkpoint.{name}"] = tensor
        auxiliaries = {
            "query_energy_ema": layer.query_energy_ema,
            "normalized_key_buffer": layer.normalized_key_buffer,
            "update_buffer": layer.update_buffer,
            "log_decay_buffer": layer.log_decay_buffer,
            "valid_count": layer.valid_count,
        }
        expected_auxiliary = {
            "query_energy_ema": (torch.float32, (16, 128)),
            "normalized_key_buffer": (torch.bfloat16, (5, 16, 128)),
            "update_buffer": (torch.bfloat16, (5, 16, 128)),
            "log_decay_buffer": (torch.float32, (5, 16)),
            "valid_count": (torch.int32, (1,)),
        }
        for name, value in auxiliaries.items():
            expected_dtype, expected_shape = expected_auxiliary[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != expected_dtype
                or tuple(value.shape) != expected_shape
            ):
                raise RuntimeError(f"{method} layer {layer_index} persistent {name} schema drifted")
            result[f"layer.{layer_index}.{name}"] = value
    return result


def _equal_byte_candidate_tensors(
    cache: object,
    *,
    method: str,
) -> dict[str, torch.Tensor]:
    from recurquant.statelease_equal_byte_baselines import EqualByteCheckpoint
    from recurquant.statelease_equal_byte_cache import (
        EqualByteLinearAttentionLayer,
        EqualByteQwen35Cache,
        _EqualByteStateView,
    )

    if type(cache) is not EqualByteQwen35Cache or getattr(cache, "codec", None) != method:
        raise RuntimeError(f"{method} equal-byte cache type or codec drifted")
    checkpoint = getattr(cache, "checkpoint", None)
    if type(checkpoint) is not EqualByteCheckpoint:
        raise RuntimeError(f"{method} equal-byte checkpoint type drifted")
    assert isinstance(checkpoint, EqualByteCheckpoint)
    checkpoint.validate()
    layers = tuple(cache.equal_byte_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"{method} equal-byte layer identities drifted")
    for layer_index, layer in layers:
        if (
            type(layer) is not EqualByteLinearAttentionLayer
            or type(layer.recurrent_states) is not _EqualByteStateView
        ):
            raise RuntimeError(f"{method} layer {layer_index} type or state view drifted")
    persistent = checkpoint.persistent_tensors()
    names = [name for name, _tensor in persistent]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{method} checkpoint tensor names are not unique")
    return {f"checkpoint.{name}": tensor for name, tensor in persistent}


def _rht_candidate_tensors(cache: object) -> dict[str, torch.Tensor]:
    from recurquant.mixed_quantization import PackedMixedQuantizedTensor
    from recurquant.packed_cache import (
        RightRhtQueryEmaMixedPackedLinearAttentionLayer,
        RightRhtQueryEmaMixedPackedRecurrentStateCache,
        _MixedPackedStateView,
    )

    if type(cache) is not RightRhtQueryEmaMixedPackedRecurrentStateCache:
        raise RuntimeError("historical RHT-CQER cache type drifted")
    layers = tuple(cache.mixed_packed_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError("historical RHT-CQER recurrent layer identities drifted")
    result: dict[str, torch.Tensor] = {}
    for layer_index, layer in layers:
        if (
            type(layer) is not RightRhtQueryEmaMixedPackedLinearAttentionLayer
            or type(layer.recurrent_states) is not _MixedPackedStateView
        ):
            raise RuntimeError(
                f"historical RHT-CQER layer {layer_index} type or state view drifted"
            )
        packed = layer.packed_states.get(0)
        if type(packed) is not PackedMixedQuantizedTensor:
            raise RuntimeError(f"historical RHT-CQER layer {layer_index} checkpoint type drifted")
        assert isinstance(packed, PackedMixedQuantizedTensor)
        if tuple(packed.original_shape) != (1, 16, 128, 128):
            raise RuntimeError(
                f"historical RHT-CQER layer {layer_index} checkpoint geometry drifted"
            )
        for name, expected_dtype in {
            "low_payload": torch.uint8,
            "high_payload": torch.int8,
            "scales": torch.float16,
            "precision_mask": torch.uint8,
        }.items():
            tensor = getattr(packed, name)
            if tensor.dtype != expected_dtype:
                raise RuntimeError(f"historical RHT-CQER layer {layer_index} {name} dtype drifted")
            result[f"layer.{layer_index}.checkpoint.{name}"] = tensor
        query_ema = layer.query_energy_ema
        if (
            not isinstance(query_ema, torch.Tensor)
            or query_ema.dtype != torch.float32
            or tuple(query_ema.shape) != (16, 128)
        ):
            raise RuntimeError(f"historical RHT-CQER layer {layer_index} query EMA drifted")
        if layer.previous_raw_mask_packed is not None:
            raise RuntimeError("historical RHT-CQER unexpectedly retained a raw mask")
        result[f"layer.{layer_index}.query_energy_ema"] = query_ema
    return result


def _candidate_persistent_tensors(
    method: str,
    cache: object,
) -> dict[str, torch.Tensor]:
    if method == ORIGINAL_RHT_METHOD:
        result = _rht_candidate_tensors(cache)
        expected_resident = 2_711_552
    elif method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        result = _statelease_candidate_tensors(cache, method=method)
        expected_resident = FROZEN_STATELEASE_RESIDENT_BYTES
    elif method in EQUAL_BYTE_NO_REPLAY_METHODS:
        result = _equal_byte_candidate_tensors(cache, method=method)
        expected_resident = FROZEN_STATELEASE_RESIDENT_BYTES
    else:
        raise ValueError(f"{method} has no strict candidate-storage schema")
    keys = [_tensor_storage_key(tensor) for tensor in result.values() if tensor.numel() != 0]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{method} candidate persistent tensors alias storage")
    for name, tensor in result.items():
        logical = tensor.numel() * tensor.element_size()
        storage_bytes = int(tensor.untyped_storage().nbytes())
        if tensor.numel() == 0:
            if storage_bytes != 0:
                raise RuntimeError(
                    f"{method} empty candidate tensor {name} retains backing storage"
                )
            continue
        if not tensor.is_contiguous() or tensor.storage_offset() != 0 or storage_bytes != logical:
            raise RuntimeError(f"{method} candidate tensor {name} does not own exact storage")
    resident = sum(key[3] for key in keys)
    if resident != expected_resident:
        raise RuntimeError(
            f"{method} explicit candidate tensors own {resident} bytes; expected "
            f"{expected_resident}"
        )
    return result


def _reachable_tensor_paths(cache: object) -> list[tuple[str, torch.Tensor]]:
    from recurquant.packed_cache import _MixedPackedStateView
    from recurquant.statelease_cache import _StateLeaseStateView
    from recurquant.statelease_equal_byte_cache import _EqualByteStateView

    state_view_types = (
        _MixedPackedStateView,
        _StateLeaseStateView,
        _EqualByteStateView,
    )
    tensors: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    visited_nodes = 0

    def visit(value: object, path: str) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > 100_000:
            raise RuntimeError("persistent cache traversal exceeded its closed bound")
        if isinstance(value, torch.Tensor):
            tensors.append((path, value))
            return
        if value is None or isinstance(
            value,
            (str, bytes, bytearray, int, float, bool, Path, torch.device, torch.dtype),
        ):
            return
        if isinstance(value, (types.ModuleType, type)):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) in state_view_types:
            attributes = getattr(value, "__dict__", {})
            for name, item in attributes.items():
                if name not in {"_owner", "owner"}:
                    visit(item, f"{path}.{name}")
            return
        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                visit(key, f"{path}.key[{index}]")
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            ordered = sorted(
                value,
                key=lambda item: repr(type(item)) + repr(item),
            )
            for index, item in enumerate(ordered):
                visit(item, f"{path}.set[{index}]")
            return
        if isinstance(value, types.MethodType):
            visit(value.__self__, f"{path}.__self__")
            visit(value.__func__, f"{path}.__func__")
            return
        if isinstance(value, types.FunctionType):
            visit(value.__defaults__, f"{path}.__defaults__")
            visit(value.__kwdefaults__, f"{path}.__kwdefaults__")
            if value.__closure__ is not None:
                for index, cell in enumerate(value.__closure__):
                    try:
                        contents = cell.cell_contents
                    except ValueError:
                        continue
                    visit(contents, f"{path}.__closure__[{index}]")
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                visit(item, f"{path}.{name}")
        slot_names: set[str] = set()
        for owner in type(value).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            slot_names.update(name for name in slots if name not in {"__dict__", "__weakref__"})
        for name in sorted(slot_names):
            if not isinstance(attributes, dict) or name not in attributes:
                try:
                    item = getattr(value, name)
                except AttributeError:
                    continue
                visit(item, f"{path}.{name}")

    visit(cache, "cache")
    return tensors


def _assert_reachable_tensor_storage_closure(
    *,
    method: str,
    reachable: Sequence[tuple[str, torch.Tensor]],
    shared: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> None:
    shared_keys = {_tensor_storage_key(tensor) for tensor in shared.values()}
    candidate_keys = {
        _tensor_storage_key(tensor) for tensor in candidate.values() if tensor.numel() != 0
    }
    if shared_keys & candidate_keys:
        raise RuntimeError(f"{method} candidate storage aliases shared KV/convolution storage")
    allowed = shared_keys | candidate_keys
    hidden_empty = [
        path
        for path, tensor in reachable
        if tensor.numel() == 0 and int(tensor.untyped_storage().nbytes()) != 0
    ]
    if hidden_empty:
        raise RuntimeError(
            f"{method} retains empty tensor views over undeclared storage: {sorted(hidden_empty)}"
        )
    paths_by_storage: dict[TensorStorageKey, list[str]] = {}
    unknown: dict[str, dict[str, object]] = {}
    for path, tensor in reachable:
        if tensor.numel() == 0:
            continue
        key = _tensor_storage_key(tensor)
        paths_by_storage.setdefault(key, []).append(path)
        if key not in allowed:
            unknown[path] = _tensor_schema(tensor)
    if unknown:
        details = [
            f"{path} ({schema['dtype']}, {schema['shape']})"
            for path, schema in sorted(unknown.items())
        ]
        raise RuntimeError(f"{method} retains undeclared persistent tensor storage: {details}")
    missing = allowed - set(paths_by_storage)
    if missing:
        raise RuntimeError(f"{method} explicit persistent tensor storage is unreachable")
    aliases = {key: sorted(paths) for key, paths in paths_by_storage.items() if len(paths) != 1}
    if aliases:
        raise RuntimeError(f"{method} persistent tensor storage has unexpected aliases")


def _audit_persistent_raw_state(
    cache: object,
    *,
    method: str,
    reference_shared_schema: Mapping[str, object],
) -> dict[str, object]:
    shared = _shared_cache_tensors(cache)
    shared_schema = _assert_shared_cache_schema(
        cache,
        reference_shared_schema,
        method=method,
    )
    candidate = _candidate_persistent_tensors(method, cache)
    shared_keys = {_tensor_storage_key(tensor) for tensor in shared.values()}
    candidate_keys = {
        _tensor_storage_key(tensor) for tensor in candidate.values() if tensor.numel() != 0
    }
    reachable = _reachable_tensor_paths(cache)
    _assert_reachable_tensor_storage_closure(
        method=method,
        reachable=reachable,
        shared=shared,
        candidate=candidate,
    )
    return {
        "persistent_fp32_state_mirror": False,
        "persistent_raw_state_bytes": 0,
        "candidate_persistent_tensor_count": len(candidate),
        "candidate_persistent_storage_bytes": sum(key[3] for key in candidate_keys),
        "shared_persistent_tensor_count": len(shared),
        "shared_persistent_storage_bytes": sum(key[3] for key in shared_keys),
        "candidate_tensor_schema": {
            name: _tensor_schema(tensor) for name, tensor in sorted(candidate.items())
        },
        "shared_tensor_schema": shared_schema,
        "runtime_reachable_tensor_storage_closure_passed": True,
    }


def _validated_storage_summary(
    method: str,
    cache: object,
    reference_shared_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary = _storage_summary(cache)
    transactional = method in {
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    }
    if transactional and summary.get("forward_transaction_active") is not False:
        raise RuntimeError(f"{method} ended with an active cache transaction")
    if method == ORIGINAL_RHT_METHOD:
        if (
            summary.get("resident_bytes") != 2_564_096
            or summary.get("resident_bytes_including_selector") != 2_711_552
            or summary.get("high_precision_groups") != 1_976
        ):
            raise RuntimeError("historical RHT-CQER storage identity drifted")
        if reference_shared_schema is None:
            raise RuntimeError("historical RHT-CQER lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
    elif method == STATELEASE_METHOD:
        if summary.get("resident_bytes_including_statelease") != (FROZEN_STATELEASE_RESIDENT_BYTES):
            raise RuntimeError("StateLease did not occupy the exact frozen byte budget")
        if reference_shared_schema is None:
            raise RuntimeError("StateLease lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    elif method in FIXED_REPLAY_METHODS:
        expected = FROZEN_STATELEASE_RESIDENT_BYTES
        if (
            summary.get("resident_bytes_including_statelease") != expected
            or summary.get("logical_resident_capacity_bytes") != expected
            or summary.get("capacity_fully_allocated") is not True
            or summary.get("off_budget") is not False
        ):
            raise RuntimeError(f"{method} is not an exact-byte resident comparator")
        if reference_shared_schema is None:
            raise RuntimeError(f"{method} lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    elif method in EQUAL_BYTE_NO_REPLAY_METHODS:
        expected = FROZEN_STATELEASE_RESIDENT_BYTES
        if (
            summary.get("resident_bytes") != expected
            or summary.get("expected_resident_bytes") != expected
            or summary.get("persistent_raw_state_bytes") != 0
            or summary.get("checkpoint_present") is not True
        ):
            raise RuntimeError(f"{method} is not an exact-byte resident comparator")
        if reference_shared_schema is None:
            raise RuntimeError(f"{method} lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    else:
        raise ValueError(f"unknown Stage-A storage method {method!r}")
    summary["runtime_storage_contract_passed"] = True
    return summary


def _diagnostics(cache: object) -> list[dict[str, object]]:
    callback = getattr(cache, "statelease_diagnostics", None)
    if callable(callback):
        value = callback()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RuntimeError("StateLease diagnostics are not a sequence")
        return [_jsonable(dict(item)) for item in value]
    query_callback = getattr(cache, "query_ema_diagnostics", None)
    if callable(query_callback):
        value = query_callback()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_jsonable(dict(item)) for item in value]
    generic = {
        name: getattr(cache, name)
        for name in (
            "update_count",
            "successful_tokens",
            "raw_state_workspace_peak_bytes",
            "query_workspace_peak_bytes",
        )
        if hasattr(cache, name)
    }
    return [_jsonable(generic)] if generic else []


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("large tensors cannot be embedded in Stage-A JSON")
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Stage-A evidence cannot contain non-finite floats")
        return value
    raise TypeError(f"Stage-A evidence cannot serialize {type(value).__name__}")


def _cuda_measurement_start(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_before": int(torch.cuda.memory_allocated(device)),
        "reserved_before": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_measurement_end(
    device: torch.device,
    start: Mapping[str, int],
) -> dict[str, int]:
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        **dict(start),
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "incremental_peak_allocated": max(
            0,
            peak_allocated - int(start["allocated_before"]),
        ),
        "incremental_peak_reserved": max(
            0,
            peak_reserved - int(start["reserved_before"]),
        ),
    }


def _reference_run(
    model: torch.nn.Module,
    tokenized: TokenizedTask,
    device: torch.device,
) -> dict[str, object]:
    from transformers import DynamicCache

    prompt = tokenized.prompt_ids.to(device)
    code = tokenized.code_ids.to(device)
    cache = DynamicCache(config=model.config)
    measurement = _cuda_measurement_start(device)
    logits: list[torch.Tensor] = []
    states: list[dict[int, torch.Tensor]] = []
    reference_accumulator = _AlignedAccumulator.empty()
    with torch.inference_mode():
        model(
            prompt,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_states = _snapshot_states_cpu(cache)
        nonrecurrent_prefill = _snapshot_nonrecurrent_prefill(cache)
        for token_index in range(ALIGNED_TOKENS):
            input_token = code[:, token_index : token_index + 1]
            target = code[:, token_index + 1 : token_index + 2]
            output = model(
                input_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cpu_logits = output.logits.detach().to("cpu").clone()
            logits.append(cpu_logits)
            states.append(_snapshot_states_cpu(cache))
            reference_accumulator.append(
                token_index=token_index,
                reference_logits=cpu_logits,
                candidate_logits=cpu_logits,
                input_token=input_token,
                target=target,
            )
    cuda = _cuda_measurement_end(device, measurement)
    shared_storage_schema = _shared_cache_schema(cache)
    reference_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in prefill_states.values()
    )
    if reference_bytes != 18_874_368:
        raise RuntimeError(f"FP32 recurrent-state byte identity drifted: {reference_bytes}")
    return {
        "prefill_states": prefill_states,
        "nonrecurrent_prefill": nonrecurrent_prefill,
        "aligned_logits": logits,
        "aligned_states": states,
        "aligned_metrics": reference_accumulator.summary(),
        "per_token": reference_accumulator.records,
        "trajectory": {
            "trajectory_nmse_auc": 0.0,
            "scored_write_count": ALIGNED_TOKENS,
            "layer_value_count": ALIGNED_TOKENS * len(LINEAR_LAYER_INDICES),
        },
        "recurrent_state_bytes": reference_bytes,
        "cuda": cuda,
        "shared_storage_schema": shared_storage_schema,
    }


def _candidate_factory(
    method: str,
    model: torch.nn.Module,
    plan: object,
) -> tuple[object, object]:
    from recurquant.query_energy import Qwen35QueryEnergyObserver
    from recurquant.qwen35 import (
        create_qwen35_experiment010_fixed_replay_cache,
        create_qwen35_experiment010_statelease_cache,
        create_qwen35_right_rht_query_ema_exact_budget_cache,
    )
    from recurquant.statelease_equal_byte_baselines import (
        EXPANDED_RHT_Q4_Q8,
        RHT_Q4_Q6_Q8,
        RHT_RESIDUAL_Q4,
    )
    from recurquant.statelease_equal_byte_cache import (
        Qwen35EqualByteObserver,
        create_qwen35_equal_byte_cache,
    )
    from recurquant.statelease_observer import Qwen35StateLeaseObserver

    if method == ORIGINAL_RHT_METHOD:
        cache = create_qwen35_right_rht_query_ema_exact_budget_cache(
            model,
            plan=plan,
            record_evidence=True,
        )
        return cache, Qwen35QueryEnergyObserver(model, caches=[cache])
    if method == STATELEASE_METHOD:
        cache = create_qwen35_experiment010_statelease_cache(
            model,
            plan=plan,
            record_evidence=True,
        )
        return cache, Qwen35StateLeaseObserver(model, caches=[cache])
    if method in FIXED_REPLAY_METHODS:
        cache = create_qwen35_experiment010_fixed_replay_cache(
            model,
            plan=plan,
            mode=method,
            record_evidence=True,
        )
        return cache, Qwen35StateLeaseObserver(model, caches=[cache])
    codec_by_method = {
        "expanded_rht_q4_q8": EXPANDED_RHT_Q4_Q8,
        "rht_q4_q6_q8": RHT_Q4_Q6_Q8,
        "rht_residual_q4": RHT_RESIDUAL_Q4,
    }
    try:
        codec = codec_by_method[method]
    except KeyError as error:
        raise ValueError(f"unknown Stage-A method {method!r}") from error
    cache = create_qwen35_equal_byte_cache(
        model,
        codec=codec,
        record_evidence=True,
    )
    return cache, Qwen35EqualByteObserver(model, caches=[cache])


def _run_candidate(
    method: str,
    model: torch.nn.Module,
    tokenized: TokenizedTask,
    reference: Mapping[str, object],
    plan: object,
    device: torch.device,
) -> dict[str, object]:
    gc.collect()
    measurement = _cuda_measurement_start(device)
    cache, observer = _candidate_factory(method, model, plan)
    prompt = tokenized.prompt_ids.to(device)
    code = tokenized.code_ids.to(device)
    accumulator = _AlignedAccumulator.empty()
    trajectory = TrajectoryNmseAccumulator()
    trajectory_records: list[dict[str, object]] = []
    reference_prefill = reference["prefill_states"]
    reference_logits = reference["aligned_logits"]
    reference_states = reference["aligned_states"]
    reference_nonrecurrent = reference["nonrecurrent_prefill"]
    if not isinstance(reference_prefill, Mapping):
        raise RuntimeError("FP32 prefill reference is invalid")
    if not isinstance(reference_logits, list) or not isinstance(reference_states, list):
        raise RuntimeError("FP32 aligned reference is invalid")

    with torch.inference_mode(), observer:
        with _capture_prefill_writes(cache) as raw_prefill:
            model(
                prompt,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        prefill_receipt = _assert_shared_prefill(
            reference_prefill,
            raw_prefill,
            method=method,
        )
        if not isinstance(reference_nonrecurrent, Mapping):
            raise RuntimeError("FP32 non-recurrent prefill reference is invalid")
        candidate_nonrecurrent = _snapshot_nonrecurrent_prefill(cache)
        prefill_receipt.update(
            _assert_shared_nonrecurrent_prefill(
                reference_nonrecurrent,
                candidate_nonrecurrent,
                method=method,
            )
        )
        prefill_receipt.update(_validate_prefill_cache_contract(method, cache))
        for token_index in range(ALIGNED_TOKENS):
            target = code[:, token_index + 1 : token_index + 2]
            output = model(
                code[:, token_index : token_index + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            accumulator.append(
                token_index=token_index,
                reference_logits=reference_logits[token_index],
                candidate_logits=output.logits,
                input_token=code[:, token_index : token_index + 1],
                target=target,
            )
            candidate_states = _snapshot_states_cpu(cache)
            layer_nmse = reference_aligned_trajectory_nmse(
                reference_states[token_index],
                candidate_states,
            )
            trajectory.append(layer_nmse)
            trajectory_records.append(
                {
                    "write_index": token_index,
                    "per_layer_nmse": {
                        str(layer): value for layer, value in sorted(layer_nmse.items())
                    },
                    "layer_macro_nmse": sum(layer_nmse.values()) / len(layer_nmse),
                }
            )

    result = {
        "aligned_metrics": accumulator.summary(),
        "per_token_aligned": accumulator.records,
        "trajectory": trajectory.summary(),
        "trajectory_per_write": trajectory_records,
        "storage": _validated_storage_summary(
            method,
            cache,
            reference.get("shared_storage_schema")
            if isinstance(reference.get("shared_storage_schema"), Mapping)
            else None,
        ),
        "update_evidence": _evidence_records(cache),
        "diagnostics": _diagnostics(cache),
        "prefill": prefill_receipt,
        "cuda": _cuda_measurement_end(device, measurement),
    }
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_frozen_stage_a(
    loaded_model: tuple[torch.nn.Module, torch.device],
    tokenized: TokenizedTask,
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
) -> dict[str, object]:
    del configuration
    model, device = loaded_model
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    reference = _reference_run(model, tokenized, device)
    candidates: dict[str, dict[str, object]] = {}
    for method in QUALITY_METHODS:
        candidates[method] = _run_candidate(
            method,
            model,
            tokenized,
            reference,
            preflight.plan,
            device,
        )
    return {
        "reference": reference,
        "candidates": candidates,
        "device": str(device),
    }


def _assert_end_integrity(
    preflight: StageAPreflight,
    reservation: AttemptReservation,
) -> dict[str, object]:
    seal = _validate_one_run_seal(
        preflight,
        reservation,
        require_receipt=True,
    )
    repository_end = _repository_state(preflight.repo_root)
    source_hashes_end = _source_hashes(preflight.repo_root)
    if repository_end["worktree_clean"] is not True:
        raise StageAAuthenticationError("repository became dirty during Stage A")
    if repository_end["commit"] != reservation.seal_commit:
        raise StageAAuthenticationError("repository left the durable one-run seal")
    if source_hashes_end != preflight.source_hashes_start:
        raise StageAAuthenticationError("authenticated source hashes changed during Stage A")
    if _file_sha256(preflight.repo_root / EXPERIMENT009_STAGE_A_RELATIVE_PATH) != (
        EXPERIMENT009_STAGE_A_FILE_SHA256
    ):
        raise StageAAuthenticationError("Experiment 009 anchor changed during Stage A")
    if _file_sha256(preflight.repo_root / IDENTITY_NOTE_RELATIVE_PATH) != (
        IDENTITY_NOTE_FILE_SHA256
    ):
        raise StageAAuthenticationError("Stage-A identity clarification changed during Stage A")
    selector_hashes = {
        "selector": _file_sha256(preflight.repo_root / SELECTOR_RELATIVE_PATH),
        "loss_selector": _file_sha256(preflight.repo_root / LOSS_SELECTOR_RELATIVE_PATH),
    }
    if selector_hashes != {
        "selector": SELECTOR_FILE_SHA256,
        "loss_selector": LOSS_SELECTOR_FILE_SHA256,
    }:
        raise StageAAuthenticationError("selector artifacts changed during Stage A")
    if _file_sha256(preflight.stage0_artifact) != preflight.stage0.get(
        "artifact_file_sha256"
    ) or _file_sha256(preflight.stage0_sha256) != preflight.stage0.get("sidecar_file_sha256"):
        raise StageAAuthenticationError("production Stage-0 identity changed during Stage A")
    return {
        "repository_end": repository_end,
        "source_hashes_end": source_hashes_end,
        "one_run_seal": seal,
        "stage0_file_hashes_reauthenticated": True,
        "stage0_independent_verification_completed_at_h0": True,
        "selectors_reauthenticated": True,
        "anchor_reauthenticated": True,
        "identity_clarification_reauthenticated": True,
        "artifact_integrity": True,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("datasets", "numpy", "safetensors", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_serialized_tensor_schemas(
    *,
    method: str,
    storage: Mapping[str, object],
) -> bool:
    expected_candidate_names = _expected_candidate_tensor_names(method)
    expected_candidate_count = len(expected_candidate_names)
    candidate_schema = storage.get("candidate_tensor_schema")
    shared_schema = storage.get("shared_tensor_schema")
    if (
        not isinstance(candidate_schema, Mapping)
        or set(candidate_schema) != expected_candidate_names
        or len(candidate_schema) != expected_candidate_count
        or storage.get("candidate_persistent_tensor_count") != expected_candidate_count
        or not isinstance(shared_schema, Mapping)
        or set(shared_schema) != EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES
        or len(shared_schema) != EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT
        or storage.get("shared_persistent_tensor_count") != EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT
    ):
        return False

    def valid_rows(
        rows: Mapping[object, object],
        *,
        allow_empty: bool,
        exact_storage: bool,
    ) -> tuple[bool, int]:
        element_sizes = {
            "torch.uint8": 1,
            "torch.int8": 1,
            "torch.float16": 2,
            "torch.bfloat16": 2,
            "torch.float32": 4,
            "torch.int32": 4,
            "torch.int64": 8,
        }
        storage_total = 0
        for name, raw_schema in rows.items():
            if not isinstance(name, str) or not isinstance(raw_schema, Mapping):
                return False, 0
            if set(raw_schema) != {
                "dtype",
                "shape",
                "logical_bytes",
                "storage_bytes",
            }:
                return False, 0
            dtype = raw_schema.get("dtype")
            shape = raw_schema.get("shape")
            logical_bytes = raw_schema.get("logical_bytes")
            storage_bytes = raw_schema.get("storage_bytes")
            if (
                not isinstance(dtype, str)
                or dtype not in element_sizes
                or not isinstance(shape, list)
                or any(
                    isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape
                )
                or isinstance(logical_bytes, bool)
                or not isinstance(logical_bytes, int)
                or logical_bytes < 0
                or isinstance(storage_bytes, bool)
                or not isinstance(storage_bytes, int)
                or storage_bytes < logical_bytes
                or (not allow_empty and logical_bytes == 0)
                or (exact_storage and storage_bytes != logical_bytes)
            ):
                return False, 0
            logical_elements = math.prod(shape)
            if logical_bytes != logical_elements * element_sizes[dtype]:
                return False, 0
            storage_total += storage_bytes
        return True, storage_total

    candidate_valid, candidate_total = valid_rows(
        candidate_schema,
        allow_empty=True,
        exact_storage=True,
    )
    shared_valid, shared_total = valid_rows(
        shared_schema,
        allow_empty=False,
        exact_storage=False,
    )
    return (
        candidate_valid
        and shared_valid
        and storage.get("candidate_persistent_storage_bytes") == candidate_total
        and storage.get("shared_persistent_storage_bytes") == shared_total
    )


def _validate_candidate_storage_results(
    candidates: Mapping[str, object],
) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for method in QUALITY_METHODS:
        candidate = candidates.get(method)
        if not isinstance(candidate, Mapping):
            raise RuntimeError(f"{method} candidate result is missing")
        storage = candidate.get("storage")
        if not isinstance(storage, Mapping):
            raise RuntimeError(f"{method} storage result is missing")
        if storage.get("runtime_storage_contract_passed") is not True:
            raise RuntimeError(f"{method} lacks a passed runtime storage contract")
        expected_candidate_bytes = (
            2_711_552 if method == ORIGINAL_RHT_METHOD else FROZEN_STATELEASE_RESIDENT_BYTES
        )
        common_passed = (
            storage.get("runtime_reachable_tensor_storage_closure_passed") is True
            and storage.get("candidate_persistent_storage_bytes") == expected_candidate_bytes
            and storage.get("persistent_raw_state_bytes") == 0
            and storage.get("persistent_fp32_state_mirror") is False
            and _validate_serialized_tensor_schemas(method=method, storage=storage)
        )
        if method == ORIGINAL_RHT_METHOD:
            passed = (
                common_passed
                and storage.get("resident_bytes") == 2_564_096
                and storage.get("resident_bytes_including_selector") == 2_711_552
                and storage.get("high_precision_groups") == 1_976
            )
        elif method == STATELEASE_METHOD:
            passed = (
                common_passed
                and storage.get("resident_bytes_including_statelease")
                == FROZEN_STATELEASE_RESIDENT_BYTES
            )
        elif method in FIXED_REPLAY_METHODS:
            passed = (
                common_passed
                and storage.get("resident_bytes_including_statelease")
                == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("logical_resident_capacity_bytes")
                == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("capacity_fully_allocated") is True
                and storage.get("off_budget") is False
            )
        else:
            passed = (
                common_passed
                and storage.get("resident_bytes") == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("expected_resident_bytes") == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("checkpoint_present") is True
            )
        if not passed:
            raise RuntimeError(f"{method} serialized storage contract drifted")
        receipts[method] = {
            "passed": True,
            "resident_bytes": (
                storage.get("resident_bytes_including_selector")
                if method == ORIGINAL_RHT_METHOD
                else storage.get(
                    "resident_bytes_including_statelease",
                    storage.get("resident_bytes"),
                )
            ),
        }
    return receipts


def _sanitized_command() -> list[str]:
    result = [Path(sys.executable).name, "scripts/screen_statelease_stage_a.py"]
    redact_next_path = False
    for argument in sys.argv[1:]:
        if redact_next_path:
            result.append(Path(argument).name)
            redact_next_path = False
            continue
        matched_inline_path = False
        for option in ("--stage0-artifact", "--stage0-sha256"):
            prefix = f"{option}="
            if argument.startswith(prefix):
                result.append(f"{prefix}{Path(argument[len(prefix) :]).name}")
                matched_inline_path = True
                break
        if matched_inline_path:
            continue
        result.append(argument)
        if argument in {"--stage0-artifact", "--stage0-sha256"}:
            redact_next_path = True
    return result


def _validate_public_artifact(value: object, *, repo_root: Path) -> None:
    forbidden_roots = {
        str(repo_root.resolve()).casefold(),
        str(Path.home().resolve()).casefold(),
    }
    sensitive_key_fragments = (
        "api_key",
        "access_token",
        "auth_token",
        "password",
        "private_key",
        "client_secret",
        "cookie",
    )
    secret_value_prefixes = (
        "sk-",
        "sk_proj-",
        "ghp_",
        "github_pat_",
        "hf_",
        "bearer ",
    )

    def walk(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized_key = str(key).casefold()
                if any(fragment in normalized_key for fragment in sensitive_key_fragments):
                    raise StageAAuthenticationError(
                        f"public artifact contains a sensitive field at {path}.{key}"
                    )
                walk(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if not isinstance(item, str):
            return
        normalized = item.strip().casefold()
        if (
            normalized.startswith(("\\\\", "/", "file://"))
            or (
                len(normalized) >= 3
                and normalized[0].isalpha()
                and normalized[1] == ":"
                and normalized[2] in {"\\", "/"}
            )
            or any(root and root in normalized for root in forbidden_roots)
        ):
            raise StageAAuthenticationError(
                f"public artifact contains an absolute local path at {path}"
            )
        if any(normalized.startswith(prefix) for prefix in secret_value_prefixes):
            raise StageAAuthenticationError(
                f"public artifact contains a secret-like value at {path}"
            )

    walk(value, "$")


def _build_artifact(
    result: Mapping[str, object],
    tokenized: TokenizedTask,
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
    integrity: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    candidates = result.get("candidates")
    reference = result.get("reference")
    if not isinstance(candidates, Mapping) or set(candidates) != set(QUALITY_METHODS):
        raise RuntimeError("Stage-A candidate result set is incomplete")
    if not isinstance(reference, Mapping):
        raise RuntimeError("Stage-A reference result is missing")
    storage_contracts = _validate_candidate_storage_results(candidates)

    gate_methods = (
        ORIGINAL_RHT_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    )
    gate_metrics = {method: candidates[method]["aligned_metrics"] for method in gate_methods}
    gate_trajectory = {method: candidates[method]["trajectory"] for method in gate_methods}
    statelease = candidates[STATELEASE_METHOD]
    gate = evaluate_statelease_stage_a_gate(
        aligned_metrics=gate_metrics,
        trajectory_nmse_auc=gate_trajectory,
        statelease_storage=statelease["storage"],
        statelease_diagnostics=statelease["diagnostics"],
        statelease_update_evidence=statelease["update_evidence"],
        stage0_complete=preflight.stage0.get("experiment_stage0_complete") is True,
        artifact_integrity=integrity.get("artifact_integrity") is True,
    )

    per_token = {
        FP32_METHOD: reference["per_token"],
        **{method: candidates[method]["per_token_aligned"] for method in QUALITY_METHODS},
    }
    aligned_metrics = {
        FP32_METHOD: reference["aligned_metrics"],
        **{method: candidates[method]["aligned_metrics"] for method in QUALITY_METHODS},
    }
    trajectory = {
        FP32_METHOD: reference["trajectory"],
        **{method: candidates[method]["trajectory"] for method in QUALITY_METHODS},
    }
    trajectory_per_write = {
        FP32_METHOD: [
            {
                "write_index": index,
                "per_layer_nmse": {str(layer): 0.0 for layer in LINEAR_LAYER_INDICES},
                "layer_macro_nmse": 0.0,
            }
            for index in range(ALIGNED_TOKENS)
        ],
        **{method: candidates[method]["trajectory_per_write"] for method in QUALITY_METHODS},
    }

    evidence: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "screening_only": True,
        "one_authenticated_quality_run": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol": {
            "name": "Experiment 010 Stage A",
            "method": "StateLease-H5",
            "task_locked": True,
            "thresholds_locked": True,
            "output_path_locked": OUTPUT_RELATIVE_PATH,
            "protected_ranked_window": [8, 16],
            "protected_window_accessed": False,
        },
        "input_authentication": {
            "stage_a_identity_clarification": preflight.identity_clarification,
            "experiment009_stage_a": {
                "path": preflight.anchor["path"],
                "file_sha256": preflight.anchor["file_sha256"],
                "canonical_evidence_sha256": preflight.anchor["canonical_evidence_sha256"],
            },
            "selectors": preflight.selector_identity,
            "production_stage0": preflight.stage0,
            "all_passed_before_quality_data_or_model_weights": True,
        },
        "model": {
            **configuration.identity,
            "device": result["device"],
        },
        "dataset": {
            "phase": "calibration",
            "selection_mode": "exact_already_open_task_id",
            "identity": EXPECTED_TASK_IDENTITY,
            "manifest_copied_from_authenticated_experiment009": preflight.anchor[
                "dataset_manifest"
            ],
            "manifest_sha256": EXPERIMENT009_MANIFEST_SHA256,
            "token_manifest_copied_from_authenticated_experiment009": preflight.anchor[
                "token_manifest"
            ],
            "runtime_token_manifest": tokenized.token_manifest,
            "runtime_text_sha256": {
                "prompt": PROMPT_TEXT_SHA256,
                "code": CODE_TEXT_SHA256,
            },
            "runtime_token_ids": _validate_token_id_hashes(
                tokenized.prompt_ids,
                tokenized.code_ids,
            ),
            "protected_window_8_16_loaded_tokenized_or_evaluated": False,
        },
        "metric_contract": {
            "aligned_primary": "one-token decode writes after the shared FP32 prefill",
            "aligned_token_count": ALIGNED_TOKENS,
            "statelease_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
            "reference_aligned_trajectory": (
                "per-layer FP64 NMSE against the matched FP32 recurrent state at every write"
            ),
        },
        "methods": list(ALL_METHODS),
        "metrics_aligned": aligned_metrics,
        "per_token_aligned": per_token,
        "trajectory_nmse": trajectory,
        "trajectory_nmse_per_layer_write": trajectory_per_write,
        "storage": {
            FP32_METHOD: {
                "recurrent_state_bytes": reference["recurrent_state_bytes"],
            },
            **{method: candidates[method]["storage"] for method in QUALITY_METHODS},
        },
        "storage_contracts": storage_contracts,
        "prefill_identity": {method: candidates[method]["prefill"] for method in QUALITY_METHODS},
        "update_evidence": {
            method: candidates[method]["update_evidence"] for method in QUALITY_METHODS
        },
        "diagnostics": {method: candidates[method]["diagnostics"] for method in QUALITY_METHODS},
        "cuda_memory": {
            FP32_METHOD: reference["cuda"],
            **{method: candidates[method]["cuda"] for method in QUALITY_METHODS},
            "scope": (
                "PyTorch allocator peaks for the correctness-first Python path; "
                "not a deployment or end-to-end process-memory claim"
            ),
            "model_load_peak_included": False,
            "tokenization_and_dataset_peak_included": False,
            "end_to_end_process_peak_available": False,
            "temporary_workspace_interpretation": (
                "method-scoped PyTorch CUDA allocator peaks and cache-reported "
                "workspace peaks only; untracked native or process-wide peaks are unavailable"
            ),
        },
        "stage_a_gate": gate,
        "repository": {
            "h0_commit": preflight.repository_start["commit"],
            "one_run_seal_commit": integrity["one_run_seal"]["seal_commit"],
            "one_run_seal_tree": integrity["one_run_seal"]["tree"],
            "start": preflight.repository_start,
            "end": integrity["repository_end"],
            "commit_transition_authorized": True,
            "empty_tree_one_run_seal": True,
            "source_tree_stable": True,
        },
        "source_files": {
            "paths": list(SOURCE_FILES),
            "loaded_local_modules": list(
                _assert_loaded_local_modules_declared(preflight.repo_root)
            ),
            "sha256_start": preflight.source_hashes_start,
            "sha256_end": integrity["source_hashes_end"],
            "stable": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(result["device"])),
        },
        "command": _sanitized_command(),
    }
    json_evidence = _jsonable(evidence)
    if not isinstance(json_evidence, dict):
        raise TypeError("Stage-A evidence did not normalize to a JSON object")
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "canonical_evidence_sha256": _canonical_sha256(json_evidence),
        "evidence": json_evidence,
    }
    return artifact, gate


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    """Atomically publish a new file without replacing an existing result."""

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
            raise StageAAuthenticationError(
                f"refusing to overwrite one-run artifact {path.name}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_completed_artifact(
    output_parent: Path,
    artifact: Mapping[str, object],
) -> tuple[bytes, str, str]:
    payload = canonical_json_bytes(artifact)
    output_parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".experiment010-stage-a-verify.",
        suffix=".json",
        dir=output_parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        verification = verify_evidence_artifact(temporary)
        if verification.get("valid") is not True:
            raise RuntimeError("temporary Stage-A artifact failed canonical verification")
        file_hash = hashlib.sha256(payload).hexdigest()
        canonical_hash = str(artifact["canonical_evidence_sha256"])
        if (
            verification.get("file_sha256") != file_hash
            or verification.get("computed_canonical_evidence_sha256") != canonical_hash
        ):
            raise RuntimeError("temporary Stage-A verification receipt drifted")
        return payload, file_hash, canonical_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_stage_a(
    result: object,
    loaded_model: object,
    tokenized: object,
    authenticated: object,
    configuration: object,
    attempt: object,
) -> dict[str, object]:
    del loaded_model
    if not isinstance(result, Mapping):
        raise TypeError("Stage-A evaluation result must be a mapping")
    if not isinstance(tokenized, TokenizedTask):
        raise TypeError("Stage-A token identity is invalid")
    if not isinstance(authenticated, StageAPreflight):
        raise TypeError("Stage-A authentication bundle is invalid")
    if not isinstance(configuration, ModelConfiguration):
        raise TypeError("Stage-A model configuration is invalid")
    if not isinstance(attempt, AttemptReservation):
        raise TypeError("Stage-A attempt receipt is invalid")
    integrity = _assert_end_integrity(authenticated, attempt)
    artifact, gate = _build_artifact(
        result,
        tokenized,
        authenticated,
        configuration,
        integrity,
    )
    _validate_public_artifact(artifact, repo_root=authenticated.repo_root)
    payload, file_hash, canonical_hash = _prepare_completed_artifact(
        authenticated.output_path.parent,
        artifact,
    )
    final_integrity = _assert_end_integrity(authenticated, attempt)
    if (
        final_integrity["repository_end"] != integrity["repository_end"]
        or final_integrity["source_hashes_end"] != integrity["source_hashes_end"]
    ):
        raise StageAAuthenticationError(
            "repository identity changed while preparing the Stage-A artifact"
        )
    _atomic_publish_new(authenticated.output_path, payload)
    completed = {
        **attempt.receipt,
        "status": "completed_with_authenticated_stage_a_result",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "completed_task_ids": [TASK_ID],
        "quality_aggregate_exposed": True,
        "output_file_sha256": file_hash,
        "output_canonical_evidence_sha256": canonical_hash,
        "stage_a_gate_passed": gate["passed"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _atomic_replace_owned(
        authenticated.attempt_path,
        canonical_json_bytes(completed),
    )
    return {
        "output": OUTPUT_RELATIVE_PATH,
        "artifact_file_sha256": file_hash,
        "canonical_evidence_sha256": canonical_hash,
        "stage_a_gate": gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def execute_stage_a(args: argparse.Namespace) -> dict[str, object]:
    holder: dict[str, StageAPreflight] = {}

    def authenticate() -> StageAPreflight:
        value = authenticate_static_inputs(args)
        holder["preflight"] = value
        return value

    hooks = AccessHooks(
        authenticate=authenticate,
        load_config=lambda preflight: load_and_authenticate_config(
            preflight,
            local_files_only=args.local_files_only,
        ),
        reserve_attempt=reserve_one_run,
        load_exact_task=load_exact_authenticated_task,
        tokenize_task=lambda preflight, row: tokenize_authenticated_task(
            preflight,
            row,
            local_files_only=args.local_files_only,
        ),
        load_weights=lambda configuration, preflight: load_model_weights(
            configuration,
            preflight,
            device_name=args.device,
            local_files_only=args.local_files_only,
        ),
        evaluate=evaluate_frozen_stage_a,
        finalize=finalize_stage_a,
        record_failure=lambda receipt, error: record_attempt_failure(
            holder["preflight"],
            receipt,
            error,
        ),
    )
    result = run_ordered_access(hooks)
    if not isinstance(result, dict):
        raise RuntimeError("Stage-A execution returned an invalid report")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage0-artifact",
        type=Path,
        required=True,
        help="Authenticated production Stage-0 torch artifact.",
    )
    parser.add_argument(
        "--stage0-sha256",
        type=Path,
        help="Optional explicit Stage-0 SHA-256 sidecar.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda"),
        default="auto",
        help="Frozen Stage A requires CUDA BF16; auto resolves only to CUDA.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Authenticate artifacts, sources, and model config without data or weights.",
    )
    mode.add_argument(
        "--execute-frozen-stage-a",
        action="store_true",
        help="Irreversibly reserve and execute the one permitted Stage-A quality run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        preflight = authenticate_static_inputs(args)
        configuration = load_and_authenticate_config(
            preflight,
            local_files_only=args.local_files_only,
        )
        print(
            json.dumps(
                {
                    "status": "stage_a_preflight_pass",
                    "quality_data_accessed": False,
                    "model_weights_loaded": False,
                    "one_run_reserved": False,
                    "repository_commit": preflight.repository_start["commit"],
                    "model": configuration.identity,
                    "output": OUTPUT_RELATIVE_PATH,
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = execute_stage_a(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stage_a_gate"]["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
