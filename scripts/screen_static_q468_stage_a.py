#!/usr/bin/env python3
"""Authenticated one-run Experiment 013 Stage-A falsification screen.

This evaluator is intentionally fail closed.  It authenticates the promoted
resolver-v6 Stage-A identity, its finalized capture provenance, the embedded
eight-dependency calibration binding, the
split-half pass, the complete H0 source inventory, the sealed runtime, the
model tree, and the checked-in Parquet identity before reserving the one run.
Only after an empty-diff seal commit wins an atomic HEAD compare-and-swap may
the canonical materializer open the twelve Stage-A rows.

Stage A is a falsification screen.  A pass is not confirmation, a novelty or
selector-superiority result, a deployment result, or a breakthrough claim.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Final, Protocol, cast

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
RUNNER_SOURCE_PATH: Final = "scripts/screen_static_q468_stage_a.py"
LAUNCHER_SOURCE_PATH: Final = "scripts/launch_static_q468_stage_a.py"
CALIBRATION_RUNNER_SOURCE_PATH: Final = "scripts/run_static_q468_calibration.py"
CAPTURE_SOURCE_PATH: Final = "scripts/capture_static_q468_identity_input.py"
RESOLVER_SOURCE_PATH: Final = "scripts/resolve_static_q468_identity.py"
SOURCE_MODULE_PATH: Final = "src/recurquant/experiment013_source.py"
STAGE_A_GATE_MODULE_PATH: Final = "src/recurquant/experiment013_stage_a.py"

RUNNER_REVISION: Final = "experiment-013-static-q468-stage-a-runner-v5"
ATTEMPT_SCHEMA: Final = "recurquant.experiment013.stage-a-attempt.v3"
IDENTITY_ATTEMPT_LOCK_SCHEMA: Final = "recurquant.experiment013.stage-a-identity-attempt-lock.v4"
IDENTITY_ATTEMPT_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "runner_revision",
        "created_at_utc",
        "attempt_number",
        "automatic_retry_authorized",
        "h0_source_commit",
        "h1_identity_commit",
        "identity_repository_path",
        "identity_file_sha256",
        "one_run_seal_commit",
        "one_run_seal_tree",
        "one_run_marker",
        "one_run_seal_message_sha256",
        "calibration_binding_file_sha256",
        "stage_a_capture_provenance_receipt_file_sha256",
        "source_manifest_file_sha256",
        "stage_a_input_bundle_manifest_file_sha256",
        "execution_bindings",
        "method_specs",
        "expected_forward_count",
        "claim_boundary",
        "output_path",
        "attempt_path",
        "complete_path",
    }
)
EXECUTION_ARTIFACT_KIND: Final = "recurquant_experiment013_stage_a_execution"
EXECUTION_ARTIFACT_SCHEMA: Final = 4
IDENTITY_SCHEMA_VERSION: Final = 6
BINDING_SCHEMA_VERSION: Final = 4
STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: Final = "stage_a_capture_provenance_receipt_file_sha256"
STAGE_A_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_stage_a_identity_capture_provenance"
)
STAGE_A_CAPTURE_PROVENANCE_STATUS: Final = (
    "captured_under_authenticated_runtime_and_launcher_finalized"
)
STAGE_A_CAPTURE_PUBLICATION_CONTRACT: Final = (
    "sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1"
)
STAGE_A_CAPTURE_RUNNER_REVISION: Final = "experiment-013-static-q468-calibration-runner-v13"
ONE_RUN_MARKER: Final = "RecurQuant-One-Run: experiment013-stage-a-v1"
CLAIM_BOUNDARY: Final = (
    "Stage A is a falsification screen only. Passage is not confirmation, selector "
    "superiority evidence, deployment evidence, state of the art, or a breakthrough claim."
)

FP32_METHOD: Final = "fp32_reference"
UNIFORM_Q4_METHOD: Final = "rht_q468_uniform_q4"
UNIFORM_Q8_METHOD: Final = "rht_q468_uniform_q8"
Q48_METHOD: Final = "rht_q48_static_p14739"
STATIC_K27030_METHOD: Final = "rht_q468_static_k27030"
DYNAMIC_K27030_METHOD: Final = "rht_q468_dynamic_k27030"
MSE_K29334_METHOD: Final = "rht_q468_static_mse_k29334"
FISHER_K29334_METHOD: Final = "rht_q468_static_diag_empirical_fisher_h1_k29334"
PRIMARY_K29334_METHOD: Final = "rht_q468_static_k29334"
METHOD_ORDER: Final = (
    FP32_METHOD,
    UNIFORM_Q4_METHOD,
    UNIFORM_Q8_METHOD,
    Q48_METHOD,
    STATIC_K27030_METHOD,
    DYNAMIC_K27030_METHOD,
    MSE_K29334_METHOD,
    FISHER_K29334_METHOD,
    PRIMARY_K29334_METHOD,
)
PRESEAL_ENGINE_SMOKE_PROFILE: Final = "experiment-013-stage-a-preseal-engine-smoke-v3"
PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT: Final = 4_096
PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_ID: Final = 1
PRESEAL_ENGINE_SMOKE_TARGET_TOKEN_COUNT: Final = 128
PRESEAL_ENGINE_SMOKE_TARGET: Final = (5,) * PRESEAL_ENGINE_SMOKE_TARGET_TOKEN_COUNT
EXPECTED_RECURRENT_RESIDENT_BYTES: Final = {
    FP32_METHOD: 18_874_368,
    UNIFORM_Q4_METHOD: 2_515_968,
    UNIFORM_Q8_METHOD: 4_875_264,
    Q48_METHOD: 3_454_664,
    STATIC_K27030_METHOD: 3_380_928,
    DYNAMIC_K27030_METHOD: 3_454_664,
    MSE_K29334_METHOD: 3_454_664,
    FISHER_K29334_METHOD: 3_454_664,
    PRIMARY_K29334_METHOD: 3_454_664,
}
RAW_TOKEN_EVIDENCE_FIELDS: Final = frozenset(
    {
        "family",
        "canonical_id",
        "selection_rank",
        "identity_record_sha256",
        "method_id",
        "transition_index",
        "input_position",
        "target_position",
        "authenticated_transition_sha256",
        "reference_nll",
        "method_nll",
        "excess_nll",
        "kl",
        "top1_agreement",
        "local_codec_sse",
        "trajectory_nmse",
        "decode_model_forward_latency_ns",
        "decode_cuda_diagnostic_peak_allocated_bytes",
        "decode_cuda_diagnostic_peak_reserved_bytes",
        "decode_logical_recurrent_resident_bytes",
        "method_cumulative_cache_reported_workspace_high_water_sum_bytes",
        "finite_checks",
    }
)
METHOD_RUNTIME_FIELDS: Final = frozenset(
    {
        "family",
        "canonical_id",
        "selection_rank",
        "identity_record_sha256",
        "method_id",
        "policy_file_sha256",
        "policy_origin",
        "prefill_diagnostics",
        "max_cuda_diagnostic_peak_allocated_bytes_across_prefill_and_decode",
        "max_cuda_diagnostic_peak_reserved_bytes_across_prefill_and_decode",
        "wall_time_with_diagnostics_ns",
        "storage",
    }
)
PREFILL_DIAGNOSTIC_FIELDS: Final = frozenset(
    {
        "model_forward_latency_ns",
        "cuda_diagnostic_peak_allocated_bytes",
        "cuda_diagnostic_peak_reserved_bytes",
        "logical_recurrent_resident_bytes",
        "cache_reported_workspace_high_water_sum_bytes",
    }
)
STORAGE_RECEIPT_FIELDS: Final = frozenset(
    {
        "logical_recurrent_resident_bytes",
        "cache_reported_resident_bytes",
        "cache_reported_expected_resident_bytes",
        "raw_state_workspace_high_water_bytes",
        "query_workspace_high_water_bytes",
        "workspace_scope",
    }
)

EXECUTION_BINDING_FIELDS: Final = frozenset(
    {
        "repository_source_manifest_file_sha256",
        "calibration_runtime_manifest_file_sha256",
        "model_file_manifest_file_sha256",
        "parquet_materialization_manifest_file_sha256",
    }
)
BINDING_DEPENDENCY_NAMES: Final = frozenset(
    {
        "calibration_score_artifact",
        "comparator_score_artifact",
        "frozen_identity_artifact",
        "split_half_stability_artifact",
        "static_fisher_k29334_policy_artifact",
        "static_k27030_policy_artifact",
        "static_k29334_policy_artifact",
        "static_mse_k29334_policy_artifact",
    }
)
BINDING_FIELDS: Final = frozenset(
    {
        "calibration_authorization_file_sha256",
        "calibration_identity_file_sha256",
        "calibration_score_artifact_file_sha256",
        "comparator_score_artifact_file_sha256",
        "split_half_stability_artifact_file_sha256",
        "static_fisher_k29334_policy_file_sha256",
        "static_k27030_policy_file_sha256",
        "static_k29334_policy_file_sha256",
        "static_mse_k29334_policy_file_sha256",
    }
)
REQUIRED_SOURCE_PATHS: Final = frozenset(
    {
        RUNNER_SOURCE_PATH,
        LAUNCHER_SOURCE_PATH,
        CALIBRATION_RUNNER_SOURCE_PATH,
        CAPTURE_SOURCE_PATH,
        RESOLVER_SOURCE_PATH,
        SOURCE_MODULE_PATH,
        STAGE_A_GATE_MODULE_PATH,
    }
)

OUTPUT_FILENAME: Final = "stage-a-result.json"
ATTEMPT_FILENAME: Final = "stage-a-attempt.json"
COMPLETE_FILENAME: Final = "STAGE_A_COMPLETE"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}")
_WINDOWS_REPARSE_POINT: Final = 0x400


class StageAError(RuntimeError):
    """Raised when an authentication, execution, or one-run invariant fails."""


class StageAStabilityFailure(StageAError):
    """Raised before reservation when the frozen split-half gate did not pass."""


@dataclass(frozen=True, slots=True)
class StageAConfig:
    frozen_identity_path: Path
    calibration_binding_path: Path
    stage_a_capture_provenance_receipt_path: Path
    repository_source_manifest_path: Path
    runtime_manifest_path: Path
    model_file_manifest_path: Path
    parquet_materialization_manifest_path: Path
    model_root: Path
    cache_root: Path
    ruler_root: Path
    input_bundle_root: Path
    repository_root: Path
    source_commit: str
    identity_commit: str
    output_dir: Path
    expected_runtime_manifest_sha256: str
    expected_model_file_manifest_sha256: str
    expected_parquet_materialization_manifest_sha256: str
    expected_stage_a_capture_provenance_receipt_sha256: str
    base_runtime_root: Path | None = None
    package_roots: Mapping[str, Path] = field(default_factory=dict)
    package_import_paths: Mapping[str, str] = field(default_factory=dict)
    interpreter_path: Path | None = None
    git_executable_path: Path | None = None
    pycache_prefix: Path | None = None

    @property
    def output_path(self) -> Path:
        return self.output_dir / OUTPUT_FILENAME

    @property
    def attempt_path(self) -> Path:
        return self.output_dir / ATTEMPT_FILENAME

    @property
    def complete_path(self) -> Path:
        return self.output_dir / COMPLETE_FILENAME


@dataclass(frozen=True, slots=True)
class BootstrapIdentity:
    file_sha256: str
    canonical_evidence_sha256: str
    execution_bindings: Mapping[str, str]
    calibration_binding: Mapping[str, str]
    identity_input_file_sha256: str
    stage_a_capture_provenance_receipt_file_sha256: str
    expected_forward_count: int


@dataclass(frozen=True, slots=True)
class StageAMethodSpec:
    method_id: str
    policy: object | None
    policy_file_sha256: str | None
    origin: str


@dataclass(frozen=True, slots=True)
class AuthenticatedStageA:
    bootstrap_identity: BootstrapIdentity
    identity: object
    binding: object
    capture_provenance_receipt: object
    capture_provenance_receipt_bytes: bytes
    dependency_bytes: Mapping[str, bytes]
    execution_artifact_bytes: Mapping[str, bytes]
    source_manifest: Mapping[str, object]
    source_manifest_file_sha256: str
    source_commit: str
    input_bundle: object | None
    input_bundle_manifest_file_sha256: str | None
    model_manifest: object
    authenticated_model_files: object
    runtime_manifest: object
    authenticated_runtime: object
    resolver: ModuleType
    capture: ModuleType
    calibration_runner: ModuleType
    source_module: ModuleType
    methods: tuple[StageAMethodSpec, ...]


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    receipt: Mapping[str, object]
    h1_commit: str
    seal_commit: str
    tree: str


@dataclass(frozen=True, slots=True)
class ForwardObservation:
    """One authenticated model forward, including private comparison logits."""

    position: int
    target_token_id: int
    comparison_logits: object = field(repr=False)
    target_nll: float = 0.0
    top1_token_id: int = 0
    local_codec_sse: float = 0.0
    trajectory_nmse: float = 0.0
    latency_ns: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    resident_bytes: int = 0
    transient_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _PresealSmokeSequence:
    identity_record_sha256: str
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StageAEvaluation:
    examples: tuple[object, ...]
    gate_rows: tuple[object, ...]
    raw_rows: tuple[Mapping[str, object], ...]
    forward_count: int
    method_runtime: tuple[Mapping[str, object], ...]
    device_runtime: Mapping[str, object]


class StageAEngine(Protocol):
    """Model-specific surface; orchestration owns order and forward counting."""

    def load_model(self, authenticated_model_files: object) -> object: ...

    def close_model(self, model: object) -> None: ...

    def begin_method(
        self,
        model: object,
        method: StageAMethodSpec,
        sequence: object,
    ) -> object: ...

    def prefill(
        self,
        session: object,
        *,
        prompt_token_ids: tuple[int, ...],
        first_target_token_id: int,
        position: int,
    ) -> ForwardObservation: ...

    def step(
        self,
        session: object,
        *,
        input_token_id: int,
        target_token_id: int,
        position: int,
    ) -> ForwardObservation: ...

    def end_method(self, session: object) -> Mapping[str, object]: ...

    def runtime_snapshot(self, model: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class StageAServices:
    authenticate: Callable[[StageAConfig], AuthenticatedStageA]
    reauthenticate: Callable[[StageAConfig, AuthenticatedStageA, AttemptReservation | None], None]
    preseal_smoke: Callable[[AuthenticatedStageA], Mapping[str, object]]
    reserve: Callable[[StageAConfig, AuthenticatedStageA], AttemptReservation]
    materialize: Callable[[StageAConfig, AuthenticatedStageA], object]
    engine: StageAEngine
    persist_receipt: Callable[
        [StageAConfig, AttemptReservation, Mapping[str, object]], AttemptReservation
    ]
    publish: Callable[
        [StageAConfig, AttemptReservation, bytes, Mapping[str, object]], Mapping[str, object]
    ]
    record_failure: Callable[[StageAConfig, AttemptReservation | None, BaseException, str], None]


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StageAError("value is not finite canonical JSON data") from error


def pretty_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StageAError("value is not finite pretty JSON data") from error


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageAError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_sha1(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise StageAError(f"{context} must be a lowercase SHA-1 object ID")
    return value


def _strict_json(data: bytes, *, context: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{context} must be bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageAError(f"{context} contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise StageAError(f"{context} contains a non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageAError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StageAError(f"{context} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, context: str
) -> None:
    if set(value) != set(expected):
        raise StageAError(f"{context} fields differ from the frozen schema")


def _safe_relative_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StageAError(f"{context} is not a canonical relative path")
    if any(character in value for character in ("\\", "\0", "\n", "\r", ":")):
        raise StageAError(f"{context} is not a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StageAError(f"{context} is not a canonical relative path")
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as error:
        raise StageAError(f"required path is unavailable: {path}") from error
    return path.is_symlink() or bool(
        getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _stable_file_bytes(path: Path, *, context: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    if _is_link_or_reparse(absolute):
        raise StageAError(f"{context} is a link or reparse point")
    try:
        before = absolute.stat()
        data = absolute.read_bytes()
        after = absolute.stat()
    except OSError as error:
        raise StageAError(f"cannot read {context}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != after.st_size
        or _is_link_or_reparse(absolute)
    ):
        raise StageAError(f"{context} changed while it was authenticated")
    return data


def bootstrap_stage_a_identity(data: bytes) -> BootstrapIdentity:
    """Strict stdlib-only v6 promotion check performed before dependency imports."""

    root = _strict_json(data, context="frozen Stage-A identity")
    _exact_fields(root, {"canonical_evidence_sha256", "evidence"}, context="identity wrapper")
    if canonical_json_bytes(root) != data:
        raise StageAError("frozen Stage-A identity is not canonical JSON")
    evidence = root.get("evidence")
    if not isinstance(evidence, dict):
        raise StageAError("frozen Stage-A identity evidence is missing")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != IDENTITY_SCHEMA_VERSION
        or evidence.get("identity_schema") != "recurquant.experiment013.identity-frozen.v6"
        or evidence.get("status") != "frozen"
        or evidence.get("phase") != "stage_a"
        or evidence.get("identity_only") is not True
        or evidence.get("promotion_required") is not False
    ):
        raise StageAError("identity is not the promoted resolver-v6 Stage-A artifact")
    promotion = evidence.get("promotion")
    if not isinstance(promotion, dict):
        raise StageAError("Stage-A identity lacks an explicit promotion")
    _exact_fields(
        promotion,
        {
            "candidate_file_sha256",
            "candidate_canonical_evidence_sha256",
            "explicit",
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD,
        },
        context="Stage-A identity promotion",
    )
    if promotion.get("explicit") is not True:
        raise StageAError("Stage-A identity lacks an explicit promotion")
    canonical_hash = _require_sha256(
        root.get("canonical_evidence_sha256"), context="identity canonical evidence SHA-256"
    )
    if canonical_hash != sha256_bytes(canonical_json_bytes(evidence)):
        raise StageAError("frozen Stage-A identity canonical evidence hash drifted")
    execution = evidence.get("execution_bindings")
    if not isinstance(execution, dict):
        raise StageAError("frozen Stage-A identity execution bindings are missing")
    _exact_fields(execution, EXECUTION_BINDING_FIELDS, context="identity execution bindings")
    normalized_execution = {
        name: _require_sha256(execution[name], context=f"execution binding {name}")
        for name in sorted(EXECUTION_BINDING_FIELDS)
    }
    calibration = evidence.get("calibration_binding")
    if not isinstance(calibration, dict):
        raise StageAError("frozen Stage-A identity calibration binding is missing")
    _exact_fields(calibration, BINDING_FIELDS, context="identity calibration binding")
    normalized_calibration = {
        name: _require_sha256(calibration[name], context=f"calibration binding {name}")
        for name in sorted(BINDING_FIELDS)
    }
    identity_input_file_sha256 = _require_sha256(
        evidence.get("source_manifest_sha256"),
        context="Stage-A identity input file SHA-256",
    )
    capture_provenance_sha256 = _require_sha256(
        evidence.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD),
        context="Stage-A capture provenance receipt file SHA-256",
    )
    if capture_provenance_sha256 != _require_sha256(
        promotion.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD),
        context="promoted Stage-A capture provenance receipt file SHA-256",
    ):
        raise StageAError("Stage-A promotion binds a different capture provenance receipt")
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise StageAError("frozen Stage-A identity must contain exactly twelve records")
    expected_order = [
        (family, rank) for family in ("pg19", "ruler", "humaneval_plus") for rank in range(4)
    ]
    observed_order: list[tuple[object, object]] = []
    continuation_counts: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise StageAError(f"identity records[{index}] must be an object")
        observed_order.append((record.get("family"), record.get("selection_rank")))
        span = record.get("token_span")
        if not isinstance(span, dict):
            raise StageAError(f"identity records[{index}] token span is missing")
        scored_start = span.get("scored_start")
        scored_stop = span.get("scored_stop")
        cache_start = span.get("cache_exposed_start")
        cache_stop = span.get("cache_exposed_stop")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (scored_start, scored_stop, cache_start, cache_stop)
        ):
            raise StageAError(f"identity records[{index}] token span is invalid")
        assert isinstance(scored_start, int) and isinstance(scored_stop, int)
        assert isinstance(cache_start, int) and isinstance(cache_stop, int)
        continuation = scored_stop - scored_start
        if continuation < 2 or cache_start != scored_start + 1 or cache_stop != scored_stop:
            raise StageAError(f"identity records[{index}] cache-exposed span drifted")
        continuation_counts.append(continuation)
    if observed_order != expected_order:
        raise StageAError("Stage-A identity record order drifted")
    expected_forward_count = len(METHOD_ORDER) * sum(1 + count - 1 for count in continuation_counts)
    return BootstrapIdentity(
        file_sha256=sha256_bytes(data),
        canonical_evidence_sha256=canonical_hash,
        execution_bindings=MappingProxyType(normalized_execution),
        calibration_binding=MappingProxyType(normalized_calibration),
        identity_input_file_sha256=identity_input_file_sha256,
        stage_a_capture_provenance_receipt_file_sha256=capture_provenance_sha256,
        expected_forward_count=expected_forward_count,
    )


def bootstrap_stage_a_capture_provenance_receipt(
    data: bytes,
    *,
    expected_file_sha256: str,
    calibration_binding_artifact: bytes,
    identity: BootstrapIdentity,
    expected_source_commit: str,
) -> Mapping[str, object]:
    """Verify the finalized flat receipt before imports, model paths, or protected rows."""

    expected_sha256 = _require_sha256(
        expected_file_sha256,
        context="expected Stage-A capture provenance receipt SHA-256",
    )
    file_sha256 = sha256_bytes(data)
    if (
        file_sha256 != expected_sha256
        or file_sha256 != identity.stage_a_capture_provenance_receipt_file_sha256
    ):
        raise StageAError("Stage-A capture provenance receipt differs from authenticated custody")
    root = _strict_json(data, context="Stage-A capture provenance receipt")
    _exact_fields(
        root,
        {
            "artifact_kind",
            "calibration_authorization_file_sha256",
            "calibration_binding_file_sha256",
            "capture_source",
            "capture_version",
            "critical_module_origins",
            "excluded_runtime_modules",
            "execution_bindings",
            "identity_input_file_sha256",
            "phase",
            "publication_contract",
            "runner_revision",
            "schema_version",
            "source_commit",
            "status",
        },
        context="Stage-A capture provenance receipt",
    )
    if canonical_json_bytes(root) != data:
        raise StageAError("Stage-A capture provenance receipt is not canonical JSON")
    if (
        root.get("artifact_kind") != STAGE_A_CAPTURE_PROVENANCE_KIND
        or type(root.get("schema_version")) is not int
        or root.get("schema_version") != 1
        or type(root.get("capture_version")) is not int
        or root.get("capture_version") != 6
        or root.get("runner_revision") != STAGE_A_CAPTURE_RUNNER_REVISION
        or root.get("phase") != "stage_a"
        or root.get("status") != STAGE_A_CAPTURE_PROVENANCE_STATUS
        or root.get("publication_contract") != STAGE_A_CAPTURE_PUBLICATION_CONTRACT
    ):
        raise StageAError("Stage-A capture provenance finalized envelope drifted")
    source_commit = _require_sha1(
        root.get("source_commit"), context="Stage-A capture provenance H0"
    )
    if source_commit != _require_sha1(
        expected_source_commit, context="expected Stage-A capture provenance H0"
    ):
        raise StageAError("Stage-A capture provenance binds a different H0")
    if (
        _require_sha256(
            root.get("identity_input_file_sha256"),
            context="Stage-A capture provenance identity input SHA-256",
        )
        != identity.identity_input_file_sha256
    ):
        raise StageAError("Stage-A capture provenance binds a different identity input")
    if _require_sha256(
        root.get("calibration_binding_file_sha256"),
        context="Stage-A capture provenance calibration binding SHA-256",
    ) != sha256_bytes(calibration_binding_artifact):
        raise StageAError("Stage-A capture provenance binds a different calibration binding")
    if (
        _require_sha256(
            root.get("calibration_authorization_file_sha256"),
            context="Stage-A capture provenance authorization SHA-256",
        )
        != identity.calibration_binding["calibration_authorization_file_sha256"]
    ):
        raise StageAError("Stage-A capture provenance binds a different authorization")
    execution = root.get("execution_bindings")
    if not isinstance(execution, Mapping):
        raise StageAError("Stage-A capture provenance execution bindings are missing")
    _exact_fields(execution, EXECUTION_BINDING_FIELDS, context="capture execution bindings")
    normalized_execution = {
        name: _require_sha256(execution[name], context=f"capture execution binding {name}")
        for name in sorted(EXECUTION_BINDING_FIELDS)
    }
    if normalized_execution != dict(identity.execution_bindings):
        raise StageAError("Stage-A capture provenance execution bindings drifted")
    capture_source = root.get("capture_source")
    if not isinstance(capture_source, Mapping):
        raise StageAError("Stage-A capture provenance source record is missing")
    _exact_fields(capture_source, {"path", "sha256"}, context="capture source record")
    if capture_source.get("path") != CAPTURE_SOURCE_PATH:
        raise StageAError("Stage-A capture provenance source path drifted")
    _require_sha256(capture_source.get("sha256"), context="capture source SHA-256")
    if not isinstance(root.get("critical_module_origins"), list):
        raise StageAError("Stage-A capture provenance critical origins are missing")
    if root.get("excluded_runtime_modules") != ["pkg_resources", "setuptools"]:
        raise StageAError("Stage-A capture provenance exclusion inventory drifted")
    return MappingProxyType(root)


def _bootstrap_source_manifest(data: bytes) -> dict[str, object]:
    root = _strict_json(data, context="repository source manifest")
    _exact_fields(
        root,
        {
            "canonical_manifest_sha256",
            "git_executable",
            "object_format",
            "paths",
            "profile",
            "repository_binding",
            "schema",
            "source_commit",
        },
        context="repository source manifest",
    )
    if pretty_json_bytes(root) != data:
        raise StageAError("repository source manifest is not canonical pretty JSON")
    if (
        root.get("schema") != "recurquant.experiment013.source-manifest.v2"
        or root.get("profile") != "experiment-013-static-q468-frozen-source-v2"
        or root.get("object_format") != "sha1"
    ):
        raise StageAError("repository source manifest object format drifted")
    raw_git = root.get("git_executable")
    if not isinstance(raw_git, dict):
        raise StageAError("repository source Git executable identity is missing")
    _exact_fields(raw_git, {"sha256", "size_bytes"}, context="source Git executable")
    size_bytes = raw_git["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise StageAError("repository source Git executable size is invalid")
    git_executable = {
        "sha256": _require_sha256(raw_git["sha256"], context="source Git executable SHA-256"),
        "size_bytes": size_bytes,
    }
    _require_sha1(root.get("source_commit"), context="source commit")
    claimed = _require_sha256(
        root.get("canonical_manifest_sha256"), context="source manifest self-hash"
    )
    payload = dict(root)
    payload.pop("canonical_manifest_sha256")
    if claimed != sha256_bytes(pretty_json_bytes(payload)):
        raise StageAError("repository source manifest self-hash drifted")
    raw_paths = root.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise StageAError("repository source manifest has no path inventory")
    paths: list[dict[str, object]] = []
    for index, entry in enumerate(raw_paths):
        if not isinstance(entry, dict):
            raise StageAError(f"source paths[{index}] must be an object")
        _exact_fields(
            entry,
            {"git_blob_oid", "index_blob_oid", "mode", "path", "raw_sha256", "worktree_blob_oid"},
            context=f"source paths[{index}]",
        )
        relative = _safe_relative_path(entry["path"], context=f"source paths[{index}].path")
        identities = {
            _require_sha1(entry[name], context=f"source paths[{index}].{name}")
            for name in ("git_blob_oid", "index_blob_oid", "worktree_blob_oid")
        }
        if len(identities) != 1:
            raise StageAError(f"source paths[{index}] commit/index/worktree identities differ")
        if entry["mode"] not in {"100644", "100755"}:
            raise StageAError(f"source paths[{index}] mode is not a regular file")
        paths.append(
            {
                "path": relative,
                "raw_sha256": _require_sha256(
                    entry["raw_sha256"], context=f"source paths[{index}].raw_sha256"
                ),
            }
        )
    rendered = [cast(str, entry["path"]) for entry in paths]
    if rendered != sorted(rendered) or len({item.casefold() for item in rendered}) != len(rendered):
        raise StageAError("repository source path inventory is not unique and sorted")
    if not set(rendered) >= REQUIRED_SOURCE_PATHS:
        raise StageAError("repository source manifest omits Stage-A implementation paths")
    return {
        "document": root,
        "entries": paths,
        "git_executable": git_executable,
        "source_commit": root["source_commit"],
    }


def _source_entries(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        cast(str, entry["path"]): cast(Mapping[str, object], entry)
        for entry in cast(Sequence[Mapping[str, object]], manifest["entries"])
    }


def _verify_source_bytes(manifest: Mapping[str, object], repository_root: Path) -> None:
    root = Path(os.path.abspath(repository_root))
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise StageAError("repository root is unavailable") from error
    if not resolved_root.is_dir() or _is_link_or_reparse(resolved_root):
        raise StageAError("repository root is not a stable directory")
    for entry in cast(Sequence[Mapping[str, object]], manifest["entries"]):
        relative = cast(str, entry["path"])
        path = resolved_root.joinpath(*PurePosixPath(relative).parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise StageAError(f"source path escapes or is unavailable: {relative}") from error
        if (
            sha256_bytes(_stable_file_bytes(resolved, context=f"source file {relative}"))
            != entry["raw_sha256"]
        ):
            raise StageAError(f"source bytes drifted: {relative}")


def _install_source_namespace(repository_root: Path) -> None:
    existing = sys.modules.get("recurquant")
    package_root = repository_root.resolve(strict=True) / "src" / "recurquant"
    if existing is not None:
        paths = getattr(existing, "__path__", None)
        if paths is None or [str(package_root)] != list(paths):
            raise StageAError("recurquant namespace was preloaded from an unauthenticated path")
        return
    package = ModuleType("recurquant")
    package.__package__ = "recurquant"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__spec__ = importlib.machinery.ModuleSpec("recurquant", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = [str(package_root)]
    sys.modules["recurquant"] = package


def _load_exact_module(
    name: str,
    relative: str,
    *,
    repository_root: Path,
    entries: Mapping[str, Mapping[str, object]],
) -> ModuleType:
    if name in sys.modules:
        raise StageAError(f"module {name} name is already occupied before authenticated load")
    resolved_root = repository_root.resolve(strict=True)
    path = (resolved_root / PurePosixPath(relative)).resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise StageAError(f"module source escapes repository root: {relative}") from error
    authenticated_bytes = _stable_file_bytes(path, context=f"module {name}")
    if sha256_bytes(authenticated_bytes) != entries[relative]["raw_sha256"]:
        raise StageAError(f"module source bytes drifted before load: {relative}")
    try:
        code = compile(authenticated_bytes, str(path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise StageAError(f"authenticated module cannot compile: {relative}") from error
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _decode_binding_dependencies(binding: object) -> dict[str, bytes]:
    """Use only dependencies released by the resolver's v4 authorization verifier."""

    raw = getattr(binding, "calibration_dependencies", None)
    if not isinstance(raw, Mapping) or set(raw) != BINDING_DEPENDENCY_NAMES:
        raise StageAError(
            "Stage-A calibration binding did not release the exact authorized dependencies"
        )
    result: dict[str, bytes] = {}
    for name in sorted(BINDING_DEPENDENCY_NAMES):
        value = raw[name]
        if not isinstance(value, bytes):
            raise StageAError(f"authorized calibration dependency {name} is not bytes")
        result[name] = value
    return result


def _binding_field_for_dependency(name: str) -> str:
    return {
        "frozen_identity_artifact": "calibration_identity_file_sha256",
        "calibration_score_artifact": "calibration_score_artifact_file_sha256",
        "split_half_stability_artifact": "split_half_stability_artifact_file_sha256",
        "static_k27030_policy_artifact": "static_k27030_policy_file_sha256",
        "static_k29334_policy_artifact": "static_k29334_policy_file_sha256",
        "comparator_score_artifact": "comparator_score_artifact_file_sha256",
        "static_fisher_k29334_policy_artifact": ("static_fisher_k29334_policy_file_sha256"),
        "static_mse_k29334_policy_artifact": "static_mse_k29334_policy_file_sha256",
    }[name]


def reconstruct_stage_a_methods(
    *,
    dependency_bytes: Mapping[str, bytes],
    frozen_stage_a_identity: object,
    source_commit: str,
) -> tuple[StageAMethodSpec, ...]:
    """Reconstruct uniform and Q48 policies only from bound candidate scores."""

    if set(dependency_bytes) != BINDING_DEPENDENCY_NAMES:
        raise StageAError("Stage-A reconstruction requires the exact authorized dependencies")
    static = importlib.import_module("recurquant.static_q468")
    calibration = importlib.import_module("recurquant.static_q468_calibration")
    scores = calibration.deserialize_calibration_score_artifact(
        dependency_bytes["calibration_score_artifact"]
    )
    identity_hash = cast(
        str, frozen_stage_a_identity.calibration_binding["calibration_identity_file_sha256"]
    )
    tokenizer_hash = cast(str, frozen_stage_a_identity.tokenizer_manifest_sha256)
    if scores.calibration_identity_sha256 != identity_hash:
        raise StageAError("candidate scores differ from the Stage-A calibration identity binding")
    geometry = scores.geometry
    common = {
        "geometry": geometry,
        "calibration_manifest_sha256": scores.aggregate.sequence_score_manifest_sha256,
        "identity_artifact_sha256": identity_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "source_commit": source_commit,
        "calibration_scores_sha256": scores.calibration_scores_sha256,
    }
    uniform_q4 = static.build_static_rht_q468_policy(
        scores.aggregate.d4,
        scores.aggregate.d6,
        scores.aggregate.d8,
        marginal_steps=0,
        method_id=UNIFORM_Q4_METHOD,
        **common,
    )
    uniform_q8 = static.build_static_rht_q468_policy(
        scores.aggregate.d4,
        scores.aggregate.d6,
        scores.aggregate.d8,
        marginal_steps=2 * geometry.total_rows,
        method_id=UNIFORM_Q8_METHOD,
        **common,
    )
    q48_scores_hash = static.static_q48_distortion_sha256(
        scores.aggregate.d4,
        scores.aggregate.d8,
        geometry=geometry,
    )
    q48 = static.build_static_rht_q48_policy(
        scores.aggregate.d4,
        scores.aggregate.d8,
        geometry=geometry,
        promoted_rows=static.FROZEN_STATIC_Q48_PROMOTIONS,
        calibration_manifest_sha256=scores.aggregate.sequence_score_manifest_sha256,
        identity_artifact_sha256=identity_hash,
        tokenizer_manifest_sha256=tokenizer_hash,
        source_commit=source_commit,
        calibration_scores_sha256=q48_scores_hash,
        method_id=Q48_METHOD,
    )
    supplied = {
        STATIC_K27030_METHOD: dependency_bytes["static_k27030_policy_artifact"],
        MSE_K29334_METHOD: dependency_bytes["static_mse_k29334_policy_artifact"],
        FISHER_K29334_METHOD: dependency_bytes["static_fisher_k29334_policy_artifact"],
        PRIMARY_K29334_METHOD: dependency_bytes["static_k29334_policy_artifact"],
    }
    decoded: dict[str, object] = {}
    for method_id, payload in supplied.items():
        policy = static.deserialize_static_rht_q468_policy(payload)
        if policy.method_id != method_id or policy.source_commit != source_commit:
            raise StageAError(f"bound policy identity drifted for {method_id}")
        decoded[method_id] = policy
    policies: dict[str, object | None] = {
        FP32_METHOD: None,
        UNIFORM_Q4_METHOD: uniform_q4,
        UNIFORM_Q8_METHOD: uniform_q8,
        Q48_METHOD: q48,
        STATIC_K27030_METHOD: decoded[STATIC_K27030_METHOD],
        DYNAMIC_K27030_METHOD: None,
        MSE_K29334_METHOD: decoded[MSE_K29334_METHOD],
        FISHER_K29334_METHOD: decoded[FISHER_K29334_METHOD],
        PRIMARY_K29334_METHOD: decoded[PRIMARY_K29334_METHOD],
    }
    origins = {
        FP32_METHOD: "runtime_reference",
        UNIFORM_Q4_METHOD: "reconstructed_candidate_scores_k0",
        UNIFORM_Q8_METHOD: f"reconstructed_candidate_scores_k{2 * geometry.total_rows}",
        Q48_METHOD: "reconstructed_candidate_scores_p14739",
        STATIC_K27030_METHOD: "embedded_binding_v3",
        DYNAMIC_K27030_METHOD: "frozen_runtime_baseline",
        MSE_K29334_METHOD: "embedded_binding_v3",
        FISHER_K29334_METHOD: "embedded_binding_v3",
        PRIMARY_K29334_METHOD: "embedded_binding_v3",
    }
    result: list[StageAMethodSpec] = []
    for method_id in METHOD_ORDER:
        policy = policies[method_id]
        if method_id in {UNIFORM_Q4_METHOD, UNIFORM_Q8_METHOD}:
            serialized = static.serialize_static_rht_q468_policy(policy)
            digest = sha256_bytes(serialized)
        elif method_id == Q48_METHOD:
            serialized = static.serialize_static_rht_q48_policy(policy)
            digest = sha256_bytes(serialized)
        elif method_id in supplied:
            digest = sha256_bytes(supplied[method_id])
        else:
            digest = None
        result.append(StageAMethodSpec(method_id, policy, digest, origins[method_id]))
    if tuple(item.method_id for item in result) != METHOD_ORDER:
        raise RuntimeError("internal Stage-A method order drifted")
    return tuple(result)


def _git_environment() -> dict[str, str]:
    inherited = {key.upper(): (key, value) for key, value in os.environ.items()}
    environment = {
        inherited[name][0]: inherited[name][1]
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC")
        if name in inherited
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": os.devnull,
            "GIT_CONFIG_KEY_1": "core.fsmonitor",
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_AUTHOR_NAME": "RecurQuant Experiment 013",
            "GIT_AUTHOR_EMAIL": "experiment013@invalid",
            "GIT_COMMITTER_NAME": "RecurQuant Experiment 013",
            "GIT_COMMITTER_EMAIL": "experiment013@invalid",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _authenticated_git_executable(path: Path | None) -> Path:
    selected: str | os.PathLike[str]
    if path is None:
        discovered = shutil.which("git")
        if discovered is None:
            raise StageAError("Git executable is unavailable")
        selected = discovered
    else:
        selected = path
    try:
        resolved = Path(selected).resolve(strict=True)
    except OSError as error:
        raise StageAError("Git executable is unavailable") from error
    if resolved.name.casefold() == "git.exe" and resolved.parent.name.casefold() == "cmd":
        try:
            resolved = (resolved.parent.parent / "mingw64" / "bin" / "git.exe").resolve(strict=True)
        except OSError as error:
            raise StageAError(
                "Git-for-Windows cmd shim has no canonical mingw64 executable"
            ) from error
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            raise StageAError("Git executable traverses a link or reparse point")
    if not resolved.is_file() or _is_link_or_reparse(resolved):
        raise StageAError("Git executable must be a regular non-link file")
    return resolved


def _git_process(
    git_executable_path: Path | None,
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    git_executable = _authenticated_git_executable(git_executable_path)
    try:
        return subprocess.run(
            [str(git_executable), *arguments],
            cwd=root,
            input=input_bytes,
            capture_output=True,
            check=False,
            env=_git_environment(),
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StageAError(f"git {arguments[0]} could not be executed") from error


def _git(
    git_executable_path: Path | None,
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> str:
    result = _git_process(git_executable_path, root, *arguments, input_bytes=input_bytes)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise StageAError(f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else ""))
    try:
        return result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise StageAError("Git returned non-UTF-8 output") from error


def _identity_attempt_lock_path(
    root: Path,
    identity_file_sha256: str,
    *,
    git_executable_path: Path | None,
) -> Path:
    identity_hash = _require_sha256(
        identity_file_sha256,
        context="identity-scoped attempt lock identity SHA-256",
    )
    raw_common = _git(
        git_executable_path,
        root,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    )
    common = Path(raw_common)
    if not common.is_absolute():
        raise StageAError("Git common directory is not absolute")
    common = Path(os.path.abspath(common))
    if not common.is_dir() or _is_link_or_reparse(common):
        raise StageAError("Git common directory is not a stable directory")
    parent = common
    for name in ("recurquant", "experiment013-stage-a"):
        child = parent / name
        try:
            child.mkdir()
        except FileExistsError:
            pass
        except OSError as error:
            raise StageAError("cannot create the identity-scoped attempt-lock directory") from error
        if not child.is_dir() or _is_link_or_reparse(child):
            raise StageAError("identity-scoped attempt-lock directory is unsafe")
        parent = child
    return parent / f"{identity_hash}.attempt.json"


def _reject_prior_identity_seal(
    root: Path,
    *,
    identity_file_sha256: str,
    git_executable_path: Path | None,
) -> None:
    identity_line = f"Stage-A-Identity: {identity_file_sha256}"
    prior = _git(
        git_executable_path,
        root,
        "log",
        "--all",
        "--reflog",
        "--format=%H",
        "--fixed-strings",
        "--all-match",
        f"--grep={ONE_RUN_MARKER}",
        f"--grep={identity_line}",
    )
    if prior:
        raise StageAError("this Stage-A identity already has a one-run seal in Git history")


def _assert_tracked_identity_bytes(config: StageAConfig, identity_bytes: bytes) -> str:
    root = config.repository_root.resolve(strict=True)
    identity = config.frozen_identity_path.resolve(strict=True)
    try:
        relative = identity.relative_to(root).as_posix()
    except ValueError as error:
        raise StageAError(
            "frozen Stage-A identity must be tracked inside the repository"
        ) from error
    relative = _safe_relative_path(relative, context="tracked Stage-A identity path")
    head = _require_sha1(
        _git(config.git_executable_path, root, "rev-parse", "HEAD"),
        context="repository HEAD",
    )
    identity_commit = _require_sha1(config.identity_commit, context="identity commit")
    if head != identity_commit:
        raise StageAError("HEAD must equal the explicit Stage-A identity authorization commit")
    show = _git_process(
        config.git_executable_path,
        root,
        "show",
        f"{identity_commit}:{relative}",
    )
    if show.returncode != 0 or show.stdout != identity_bytes:
        raise StageAError("tracked identity bytes differ from the identity authorization commit")
    index = _git_process(config.git_executable_path, root, "show", f":{relative}")
    if index.returncode != 0 or index.stdout != identity_bytes:
        raise StageAError("tracked identity bytes differ from the Git index")
    if _stable_file_bytes(identity, context="tracked Stage-A identity") != identity_bytes:
        raise StageAError("tracked identity bytes differ from the worktree")
    return relative


def _assert_output_paths_isolated(config: StageAConfig) -> None:
    """Require a stable output directory and ignore repository-local evidence."""

    root = config.repository_root.resolve(strict=True)
    output_candidate = Path(os.path.abspath(config.output_dir))

    def nested(path: Path, possible_parent: Path) -> bool:
        try:
            path.relative_to(possible_parent)
        except ValueError:
            return False
        return True

    output_paths = (
        output_candidate,
        Path(os.path.abspath(config.output_path)),
        Path(os.path.abspath(config.attempt_path)),
        Path(os.path.abspath(config.complete_path)),
    )
    protected_roots = {
        "input bundle": Path(os.path.abspath(config.input_bundle_root)),
        "model": Path(os.path.abspath(config.model_root)),
        **(
            {"base runtime": Path(os.path.abspath(config.base_runtime_root))}
            if config.base_runtime_root is not None
            else {}
        ),
        **{
            f"package runtime {name}": Path(os.path.abspath(path))
            for name, path in config.package_roots.items()
        },
    }
    for protected_name, protected_root in protected_roots.items():
        for path in output_paths:
            if nested(path, protected_root) or nested(protected_root, path):
                raise StageAError(
                    f"Stage-A {protected_name} and output evidence paths must not overlap"
                )
    output_dir = _safe_directory(config.output_dir, create=True)
    for path in (config.output_path, config.attempt_path, config.complete_path):
        absolute = Path(os.path.abspath(path))
        if absolute.parent != output_dir:
            raise StageAError("Stage-A output paths escaped the authenticated output directory")
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError:
            continue
        relative = _safe_relative_path(relative, context="repository-local Stage-A output path")
        ignored = _git_process(
            config.git_executable_path,
            root,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative,
        )
        if ignored.returncode != 0:
            raise StageAError(
                "repository-local Stage-A outputs must be ignored before one-run reservation"
            )


def _safe_directory(path: Path, *, create: bool) -> Path:
    """Validate every directory component without resolving through links/reparse points."""

    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or not absolute.anchor:
        raise StageAError("Stage-A directory is not absolute")
    current = Path(absolute.anchor)
    components = absolute.parts[1:]
    for component in components:
        current = current / component
        if not os.path.lexists(current):
            if not create:
                raise StageAError(f"required Stage-A directory is absent: {current}")
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as error:
                raise StageAError(f"cannot create Stage-A directory: {current}") from error
        if _is_link_or_reparse(current):
            raise StageAError(f"Stage-A directory is a link or reparse point: {current}")
        try:
            status = current.stat()
        except OSError as error:
            raise StageAError(f"cannot authenticate Stage-A directory: {current}") from error
        if not stat.S_ISDIR(status.st_mode):
            raise StageAError(f"Stage-A directory component is not a directory: {current}")
    return absolute


def _exclusive_write(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(path))
    parent = _safe_directory(absolute.parent, create=True)
    if absolute.parent != parent:
        raise StageAError("exclusive write path escaped its authenticated parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(absolute, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_directory(parent, create=False)
        if _is_link_or_reparse(absolute) or not absolute.is_file():
            raise StageAError("exclusive write did not create a stable regular file")
    except BaseException:
        with contextlib.suppress(OSError):
            absolute.unlink()
        raise


def _atomic_replace_owned(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(path))
    parent = _safe_directory(absolute.parent, create=False)
    if not absolute.is_file() or _is_link_or_reparse(absolute):
        raise StageAError("owned receipt disappeared or became unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_directory(parent, create=False)
        os.replace(temporary, absolute)
        _safe_directory(parent, create=False)
        if _is_link_or_reparse(absolute) or not absolute.is_file():
            raise StageAError("owned receipt replacement became unsafe")
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(path))
    parent = _safe_directory(absolute.parent, create=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _safe_directory(parent, create=False)
            os.link(temporary, absolute)
        except FileExistsError as error:
            raise StageAError(f"refusing to overwrite published output: {absolute}") from error
        temporary.unlink()
        _safe_directory(parent, create=False)
        if _is_link_or_reparse(absolute) or not absolute.is_file():
            raise StageAError("published output became unsafe")
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _seal_message_values(
    *,
    identity_file_sha256: str,
    calibration_binding_file_sha256: str,
    stage_a_capture_provenance_receipt_file_sha256: str,
    source_manifest_file_sha256: str,
    input_bundle_manifest_file_sha256: str,
    expected_forward_count: int,
) -> str:
    return (
        "Experiment 013 Stage-A one-run seal\n\n"
        f"{ONE_RUN_MARKER}\n"
        f"Stage-A-Identity: {identity_file_sha256}\n"
        f"Calibration-Binding: {calibration_binding_file_sha256}\n"
        "Stage-A-Capture-Provenance: "
        f"{stage_a_capture_provenance_receipt_file_sha256}\n"
        f"Source-Manifest: {source_manifest_file_sha256}\n"
        f"Stage-A-Input-Bundle: {input_bundle_manifest_file_sha256}\n"
        f"Expected-Forwards: {expected_forward_count}\n"
    )


def _seal_message(authenticated: AuthenticatedStageA) -> str:
    bundle_hash = _require_sha256(
        authenticated.input_bundle_manifest_file_sha256,
        context="authenticated Stage-A input bundle manifest SHA-256",
    )
    return _seal_message_values(
        identity_file_sha256=authenticated.bootstrap_identity.file_sha256,
        calibration_binding_file_sha256=authenticated.binding.file_sha256,
        stage_a_capture_provenance_receipt_file_sha256=(
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        source_manifest_file_sha256=authenticated.source_manifest_file_sha256,
        input_bundle_manifest_file_sha256=bundle_hash,
        expected_forward_count=authenticated.bootstrap_identity.expected_forward_count,
    )


def _method_spec_receipts(
    methods: Sequence[StageAMethodSpec],
) -> list[dict[str, object]]:
    if tuple(method.method_id for method in methods) != METHOD_ORDER:
        raise StageAError("Stage-A method specifications are incomplete or reordered")
    receipts: list[dict[str, object]] = []
    for method in methods:
        if method.method_id in {FP32_METHOD, DYNAMIC_K27030_METHOD}:
            if method.policy_file_sha256 is not None:
                raise StageAError(f"{method.method_id} must not bind a static policy file")
        else:
            _require_sha256(
                method.policy_file_sha256,
                context=f"{method.method_id} policy file SHA-256",
            )
        receipts.append(
            {
                "method_id": method.method_id,
                "policy_file_sha256": method.policy_file_sha256,
                "policy_origin": _bounded_text(
                    method.origin,
                    context=f"{method.method_id} policy origin",
                ),
            }
        )
    return receipts


def reserve_one_run(config: StageAConfig, authenticated: AuthenticatedStageA) -> AttemptReservation:
    _assert_output_paths_isolated(config)
    if config.output_path.exists() or config.complete_path.exists() or config.attempt_path.exists():
        raise StageAError("Stage-A output or attempt already exists; automatic retry is forbidden")
    identity_bytes = _stable_file_bytes(config.frozen_identity_path, context="frozen identity")
    identity_path = _assert_tracked_identity_bytes(config, identity_bytes)
    identity_file_sha256 = authenticated.bootstrap_identity.file_sha256
    _reject_prior_identity_seal(
        config.repository_root,
        identity_file_sha256=identity_file_sha256,
        git_executable_path=config.git_executable_path,
    )
    h1 = _require_sha1(
        _git(config.git_executable_path, config.repository_root, "rev-parse", "HEAD"),
        context="H1",
    )
    tree = _require_sha1(
        _git(
            config.git_executable_path,
            config.repository_root,
            "show",
            "-s",
            "--format=%T",
            h1,
        ),
        context="H1 tree",
    )
    message = _seal_message(authenticated)
    seal = _require_sha1(
        _git(
            config.git_executable_path,
            config.repository_root,
            "commit-tree",
            tree,
            "-p",
            h1,
            input_bytes=message.encode("utf-8"),
        ),
        context="one-run seal commit",
    )
    global_lock = _identity_attempt_lock_path(
        config.repository_root,
        identity_file_sha256,
        git_executable_path=config.git_executable_path,
    )
    created_at_utc = datetime.now(UTC).isoformat()
    method_specs = _method_spec_receipts(authenticated.methods)
    global_lock_document = {
        "schema": IDENTITY_ATTEMPT_LOCK_SCHEMA,
        "runner_revision": RUNNER_REVISION,
        "created_at_utc": created_at_utc,
        "attempt_number": 1,
        "automatic_retry_authorized": False,
        "h0_source_commit": authenticated.source_commit,
        "h1_identity_commit": h1,
        "identity_repository_path": identity_path,
        "identity_file_sha256": identity_file_sha256,
        "one_run_seal_commit": seal,
        "one_run_seal_tree": tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": sha256_bytes(message.encode("utf-8")),
        "calibration_binding_file_sha256": authenticated.binding.file_sha256,
        STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        "source_manifest_file_sha256": authenticated.source_manifest_file_sha256,
        "stage_a_input_bundle_manifest_file_sha256": _require_sha256(
            authenticated.input_bundle_manifest_file_sha256,
            context="authenticated Stage-A input bundle manifest SHA-256",
        ),
        "execution_bindings": dict(authenticated.bootstrap_identity.execution_bindings),
        "method_specs": method_specs,
        "expected_forward_count": authenticated.bootstrap_identity.expected_forward_count,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_path": str(Path(os.path.abspath(config.output_path))),
        "attempt_path": str(Path(os.path.abspath(config.attempt_path))),
        "complete_path": str(Path(os.path.abspath(config.complete_path))),
    }
    global_lock_bytes = canonical_json_bytes(global_lock_document)
    try:
        _exclusive_write(global_lock, global_lock_bytes)
    except FileExistsError as error:
        raise StageAError(
            "this Stage-A identity already has a consumed identity-scoped attempt lock"
        ) from error
    if _stable_file_bytes(global_lock, context="identity-scoped attempt lock") != (
        global_lock_bytes
    ):
        raise StageAError("identity-scoped attempt lock changed after exclusive creation")
    prepared: dict[str, object] = {
        "schema": ATTEMPT_SCHEMA,
        "status": "prepared_before_head_cas",
        "runner_revision": RUNNER_REVISION,
        "created_at_utc": created_at_utc,
        "attempt_number": 1,
        "h0_source_commit": authenticated.source_commit,
        "h1_identity_commit": h1,
        "identity_repository_path": identity_path,
        "one_run_seal_commit": seal,
        "one_run_seal_tree": tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": sha256_bytes(message.encode("utf-8")),
        "identity_file_sha256": identity_file_sha256,
        "identity_scoped_attempt_lock_file_sha256": sha256_bytes(global_lock_bytes),
        "calibration_binding_file_sha256": authenticated.binding.file_sha256,
        STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        "source_manifest_file_sha256": authenticated.source_manifest_file_sha256,
        "stage_a_input_bundle_manifest_file_sha256": _require_sha256(
            authenticated.input_bundle_manifest_file_sha256,
            context="authenticated Stage-A input bundle manifest SHA-256",
        ),
        "execution_bindings": dict(authenticated.bootstrap_identity.execution_bindings),
        "method_specs": method_specs,
        "expected_forward_count": authenticated.bootstrap_identity.expected_forward_count,
        "observed_forward_count": 0,
        "content_materialized": False,
        "model_load_count": 0,
        "evaluation_complete": False,
        "result_available": False,
        "automatic_retry_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    prepared_bytes = canonical_json_bytes(prepared)
    _exclusive_write(config.attempt_path, prepared_bytes)
    if (
        _stable_file_bytes(config.attempt_path, context="prepared attempt receipt")
        != prepared_bytes
    ):
        raise StageAError("prepared attempt receipt changed before HEAD CAS")
    cas = _git_process(
        config.git_executable_path,
        config.repository_root,
        "update-ref",
        "HEAD",
        seal,
        h1,
    )
    if cas.returncode != 0:
        detail = cas.stderr.decode("utf-8", errors="replace").strip()
        raise StageAError(
            "one-run HEAD compare-and-swap failed" + (f": {detail}" if detail else "")
        )
    if (
        _git(
            config.git_executable_path,
            config.repository_root,
            "rev-parse",
            "HEAD",
        )
        != seal
    ):
        raise StageAError("one-run HEAD compare-and-swap did not reach the seal")
    reserved = {**prepared, "status": "reserved_before_stage_a_content_access"}
    _atomic_replace_owned(config.attempt_path, canonical_json_bytes(reserved))
    return AttemptReservation(reserved, h1, seal, tree)


def persist_receipt(
    config: StageAConfig,
    reservation: AttemptReservation,
    updates: Mapping[str, object],
) -> AttemptReservation:
    disk = _strict_json(
        _stable_file_bytes(config.attempt_path, context="Stage-A attempt receipt"),
        context="Stage-A attempt receipt",
    )
    if disk != dict(reservation.receipt):
        raise StageAError("Stage-A attempt receipt changed outside the evaluator")
    updated = {**disk, **dict(updates), "automatic_retry_authorized": False}
    payload = canonical_json_bytes(updated)
    _atomic_replace_owned(config.attempt_path, payload)
    if _stable_file_bytes(config.attempt_path, context="updated attempt receipt") != payload:
        raise StageAError("Stage-A attempt receipt persistence drifted")
    return dataclasses.replace(reservation, receipt=MappingProxyType(updated))


def _transition_hash(
    *,
    identity_record_sha256: str,
    method_id: str,
    transition_index: int,
    input_position: int,
    target_position: int,
) -> str:
    return sha256_bytes(
        b"recurquant.experiment013.stage-a-transition.v2\0"
        + canonical_json_bytes(
            {
                "identity_record_sha256": identity_record_sha256,
                "input_position": input_position,
                "method_id": method_id,
                "target_position": target_position,
                "transition_index": transition_index,
            }
        )
    )


def _finite_observation(
    observation: ForwardObservation,
    *,
    expected_position: int,
    expected_target: int,
) -> ForwardObservation:
    if not isinstance(observation, ForwardObservation):
        raise StageAError("Stage-A engine returned a non-observation")
    if observation.position != expected_position or observation.target_token_id != expected_target:
        raise StageAError("Stage-A forward position or target identity drifted")
    torch = importlib.import_module("torch")
    logits = observation.comparison_logits
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or logits.numel() == 0
        or logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or logits.requires_grad
        or not torch.isfinite(logits).all().item()
    ):
        raise StageAError(
            "Stage-A comparison logits must be a finite contiguous CPU float32 tensor"
        )
    values = (
        observation.target_nll,
        observation.local_codec_sse,
        observation.trajectory_nmse,
    )
    if (
        any(not math.isfinite(value) for value in values)
        or observation.target_nll < 0.0
        or observation.local_codec_sse < 0.0
        or observation.trajectory_nmse < 0.0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                observation.latency_ns,
                observation.peak_allocated_bytes,
                observation.peak_reserved_bytes,
                observation.resident_bytes,
                observation.transient_bytes,
            )
        )
        or isinstance(observation.top1_token_id, bool)
        or not isinstance(observation.top1_token_id, int)
        or observation.top1_token_id < 0
        or observation.top1_token_id >= logits.numel()
        or isinstance(observation.target_token_id, bool)
        or not isinstance(observation.target_token_id, int)
        or observation.target_token_id < 0
        or observation.target_token_id >= logits.numel()
    ):
        raise StageAError("Stage-A forward observation contains invalid or non-finite evidence")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    expected_nll = -float(log_probabilities[expected_target].item())
    expected_top1 = int(torch.argmax(logits).item())
    if observation.target_nll != expected_nll or observation.top1_token_id != expected_top1:
        raise StageAError("Stage-A NLL or top-1 evidence differs from its comparison logits")
    return observation


def _kl(reference: object, candidate: object) -> float:
    """Reviewed FP32 token KL equation over compact CPU logits."""

    torch = importlib.import_module("torch")
    if (
        not isinstance(reference, torch.Tensor)
        or not isinstance(candidate, torch.Tensor)
        or reference.ndim != 1
        or candidate.ndim != 1
        or reference.shape != candidate.shape
        or reference.dtype != torch.float32
        or candidate.dtype != torch.float32
        or reference.device.type != "cpu"
        or candidate.device.type != "cpu"
        or not reference.is_contiguous()
        or not candidate.is_contiguous()
    ):
        raise StageAError("Stage-A reference and candidate vocabulary dimensions differ")
    if not torch.isfinite(reference).all().item() or not torch.isfinite(candidate).all().item():
        raise StageAError("Stage-A comparison logits are non-finite")
    reference_log_probabilities = torch.log_softmax(reference, dim=-1)
    candidate_log_probabilities = torch.log_softmax(candidate, dim=-1)
    value = float(
        (
            reference_log_probabilities.exp()
            * (reference_log_probabilities - candidate_log_probabilities)
        )
        .sum(dim=-1)
        .item()
    )
    if not math.isfinite(value):
        raise StageAError("Stage-A KL became non-finite")
    if value < -1.0e-6:
        raise StageAError("Stage-A KL is materially negative")
    return max(0.0, value)


def _validated_device_runtime(value: Mapping[str, object]) -> Mapping[str, object]:
    expected = {
        "attention_implementation",
        "capability",
        "cuda_runtime",
        "device_index",
        "model_class",
        "model_config_class",
        "model_parameter_dtype",
        "name",
        "torch_version",
        "total_memory_bytes",
    }
    _exact_fields(value, expected, context="Stage-A device runtime")
    capability = value.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
    ):
        raise StageAError("Stage-A CUDA capability is invalid")
    for field_name in (
        "attention_implementation",
        "cuda_runtime",
        "model_class",
        "model_config_class",
        "model_parameter_dtype",
        "name",
        "torch_version",
    ):
        item = value.get(field_name)
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 256
            or any(ord(character) < 32 for character in item)
        ):
            raise StageAError(f"Stage-A device runtime {field_name} is invalid")
    for field_name in ("device_index", "total_memory_bytes"):
        item = value.get(field_name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StageAError(f"Stage-A device runtime {field_name} is invalid")
    if (
        value.get("attention_implementation") != "eager"
        or value.get("model_class") != "Qwen3_5ForCausalLM"
        or value.get("model_config_class") != "Qwen3_5TextConfig"
        or value.get("model_parameter_dtype") != "torch.bfloat16"
    ):
        raise StageAError(
            "Stage-A model class, config, eager attention, or BF16 dtype contract drifted"
        )
    return MappingProxyType({**dict(value), "capability": list(capability)})


def _validated_authenticated_runtime(
    value: Mapping[str, object],
    *,
    expected_manifest_file_sha256: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StageAError("Stage-A authenticated runtime evidence is missing")
    _exact_fields(
        value,
        {
            "base_runtime_file_count",
            "distribution_count",
            "distributions",
            "file_count",
            "git_executable_absolute_path_sha256",
            "git_executable_sha256",
            "git_executable_size_bytes",
            "interpreter_sha256",
            "machine_name",
            "manifest_file_sha256",
            "package_root_count",
            "python_cache_tag",
            "python_implementation",
            "python_version",
        },
        context="Stage-A authenticated runtime evidence",
    )
    manifest_hash = _require_sha256(
        expected_manifest_file_sha256,
        context="expected Stage-A runtime manifest file SHA-256",
    )
    if value.get("manifest_file_sha256") != manifest_hash:
        raise StageAError("Stage-A authenticated runtime differs from its execution binding")
    for name in (
        "git_executable_absolute_path_sha256",
        "git_executable_sha256",
        "interpreter_sha256",
    ):
        _require_sha256(value.get(name), context=f"Stage-A authenticated runtime {name}")
    for name in (
        "base_runtime_file_count",
        "distribution_count",
        "file_count",
        "git_executable_size_bytes",
        "package_root_count",
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StageAError(f"Stage-A authenticated runtime {name} is invalid")
    for name in (
        "machine_name",
        "python_cache_tag",
        "python_implementation",
        "python_version",
    ):
        _bounded_text(value.get(name), context=f"Stage-A authenticated runtime {name}")
    distributions = value.get("distributions")
    if not isinstance(distributions, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or any(
            not isinstance(part, str)
            or not part
            or len(part) > 512
            or part != part.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            for part in item
        )
        for item in distributions
    ):
        raise StageAError("Stage-A authenticated distributions are invalid")
    if value.get("distribution_count") != len(distributions):
        raise StageAError("Stage-A authenticated distribution count drifted")
    return MappingProxyType(
        {**dict(value), "distributions": [list(item) for item in distributions]}
    )


def _authenticated_runtime_record(
    runtime: object,
    *,
    expected_manifest_file_sha256: str,
) -> Mapping[str, object]:
    return _validated_authenticated_runtime(
        {
            "base_runtime_file_count": getattr(runtime, "base_runtime_file_count", None),
            "distribution_count": getattr(runtime, "distribution_count", None),
            "distributions": [list(item) for item in getattr(runtime, "distributions", ())],
            "file_count": getattr(runtime, "file_count", None),
            "git_executable_absolute_path_sha256": getattr(
                runtime,
                "git_executable_absolute_path_sha256",
                None,
            ),
            "git_executable_sha256": getattr(runtime, "git_executable_sha256", None),
            "git_executable_size_bytes": getattr(
                runtime,
                "git_executable_size_bytes",
                None,
            ),
            "interpreter_sha256": getattr(runtime, "interpreter_sha256", None),
            "machine_name": getattr(runtime, "machine_name", None),
            "manifest_file_sha256": getattr(runtime, "manifest_file_sha256", None),
            "package_root_count": getattr(runtime, "package_root_count", None),
            "python_cache_tag": getattr(runtime, "python_cache_tag", None),
            "python_implementation": getattr(runtime, "python_implementation", None),
            "python_version": getattr(runtime, "python_version", None),
        },
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )


def _sequence_fields(
    sequence: object,
) -> tuple[Mapping[str, object], tuple[int, ...], tuple[int, ...]]:
    record = getattr(sequence, "identity_record", None)
    prompt = getattr(sequence, "prompt_token_ids", None)
    target = getattr(sequence, "target_token_ids", None)
    if (
        not isinstance(record, Mapping)
        or not isinstance(prompt, tuple)
        or not isinstance(target, tuple)
    ):
        raise StageAError("Stage-A materializer returned an invalid sequence")
    if not prompt or len(target) < 2:
        raise StageAError("Stage-A sequence does not satisfy prompt/continuation bounds")
    span = record.get("token_span")
    if not isinstance(span, Mapping):
        raise StageAError("Stage-A identity record has no token span")
    expected_span = {
        "prefill_start": 0,
        "prefill_stop": len(prompt),
        "scored_start": len(prompt),
        "scored_stop": len(prompt) + len(target),
        "cache_exposed_start": len(prompt) + 1,
        "cache_exposed_stop": len(prompt) + len(target),
    }
    if dict(span) != expected_span:
        raise StageAError("Stage-A materialized token span differs from the identity")
    return record, prompt, target


def _storage_receipt(method_id: str, summary: Mapping[str, object]) -> Mapping[str, object]:
    if method_id not in METHOD_ORDER:
        raise StageAError("Stage-A storage receipt method is unknown")
    expected_resident = EXPECTED_RECURRENT_RESIDENT_BYTES[method_id]
    reported_resident = summary.get("resident_bytes", expected_resident)
    reported_expected = summary.get("expected_resident_bytes", expected_resident)
    raw_workspace = summary.get("raw_state_workspace_peak_bytes", 0)
    query_workspace = summary.get("query_workspace_peak_bytes", 0)
    for name, value in (
        ("cache-reported resident bytes", reported_resident),
        ("cache-reported expected resident bytes", reported_expected),
        ("raw-state workspace high-water bytes", raw_workspace),
        ("query workspace high-water bytes", query_workspace),
    ):
        _nonnegative_int(value, context=f"Stage-A {name}")
    if reported_resident != expected_resident or reported_expected != expected_resident:
        raise StageAError("Stage-A cache storage differs from the frozen resident-byte ledger")
    return MappingProxyType(
        {
            "logical_recurrent_resident_bytes": expected_resident,
            "cache_reported_resident_bytes": reported_resident,
            "cache_reported_expected_resident_bytes": reported_expected,
            "raw_state_workspace_high_water_bytes": raw_workspace,
            "query_workspace_high_water_bytes": query_workspace,
            "workspace_scope": "method_lifetime_high_water_since_cache_creation",
        }
    )


def evaluate_materialized_stage_a(
    authenticated: AuthenticatedStageA,
    materialization: object,
    engine: StageAEngine,
    model: object,
) -> StageAEvaluation:
    """Execute the fixed 9-method grid and independently count every forward."""

    gate = importlib.import_module("recurquant.experiment013_stage_a")
    if tuple(gate.STAGE_A_METHOD_ORDER) != METHOD_ORDER:
        raise StageAError("Stage-A gate method order differs from the runner")
    if tuple(method.method_id for method in authenticated.methods) != METHOD_ORDER:
        raise StageAError("authenticated Stage-A methods are missing, duplicated, or reordered")
    sequences = getattr(materialization, "sequences", None)
    if not isinstance(sequences, tuple) or len(sequences) != 12:
        raise StageAError("Stage-A materialization must contain exactly twelve sequences")
    examples: list[object] = []
    gate_rows: list[object] = []
    raw_rows: list[Mapping[str, object]] = []
    method_runtime: list[Mapping[str, object]] = []
    forwards = 0
    for sequence in sequences:
        record, prompt_ids, target_ids = _sequence_fields(sequence)
        identity_hash = _require_sha256(
            record.get("identity_record_sha256"), context="Stage-A identity record SHA-256"
        )
        family = record.get("family")
        canonical_id = record.get("canonical_id")
        rank = record.get("selection_rank")
        if family not in {"pg19", "ruler", "humaneval_plus"}:
            raise StageAError("Stage-A sequence family drifted")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise StageAError("Stage-A canonical ID is invalid")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(4):
            raise StageAError("Stage-A selection rank is invalid")
        examples.append(
            gate.StageAExample(
                family=family,
                canonical_id=canonical_id,
                selection_rank=rank,
                continuation_token_count=len(target_ids),
                identity_record_sha256=identity_hash,
            )
        )
        references: list[ForwardObservation] = []
        for method in authenticated.methods:
            session = engine.begin_method(model, method, sequence)
            method_started = time.perf_counter_ns()
            try:
                prefill_position = len(prompt_ids) - 1
                prefill = _finite_observation(
                    engine.prefill(
                        session,
                        prompt_token_ids=prompt_ids,
                        first_target_token_id=target_ids[0],
                        position=prefill_position,
                    ),
                    expected_position=prefill_position,
                    expected_target=target_ids[0],
                )
                forwards += 1
                current_rows: list[ForwardObservation] = []
                for transition_index in range(len(target_ids) - 1):
                    input_position = len(prompt_ids) + transition_index
                    target_position = input_position + 1
                    observation = _finite_observation(
                        engine.step(
                            session,
                            input_token_id=target_ids[transition_index],
                            target_token_id=target_ids[transition_index + 1],
                            position=input_position,
                        ),
                        expected_position=input_position,
                        expected_target=target_ids[transition_index + 1],
                    )
                    forwards += 1
                    current_rows.append(observation)
                    if method.method_id == FP32_METHOD:
                        reference = observation
                        references.append(observation)
                        kl = 0.0
                        agreement = True
                    else:
                        if transition_index >= len(references):
                            raise StageAError("candidate method ran before its FP32 reference")
                        reference = references[transition_index]
                        kl = _kl(reference.comparison_logits, observation.comparison_logits)
                        agreement = observation.top1_token_id == reference.top1_token_id
                    transition_hash = _transition_hash(
                        identity_record_sha256=identity_hash,
                        method_id=method.method_id,
                        transition_index=transition_index,
                        input_position=input_position,
                        target_position=target_position,
                    )
                    gate_rows.append(
                        gate.StageATokenRow(
                            family=family,
                            canonical_id=canonical_id,
                            selection_rank=rank,
                            identity_record_sha256=identity_hash,
                            method_id=method.method_id,
                            transition_index=transition_index,
                            reference_nll=reference.target_nll,
                            method_nll=observation.target_nll,
                            kl=kl,
                            top1_agreement=agreement,
                        )
                    )
                    raw_rows.append(
                        MappingProxyType(
                            {
                                "family": family,
                                "canonical_id": canonical_id,
                                "selection_rank": rank,
                                "identity_record_sha256": identity_hash,
                                "method_id": method.method_id,
                                "transition_index": transition_index,
                                "input_position": input_position,
                                "target_position": target_position,
                                "authenticated_transition_sha256": transition_hash,
                                "reference_nll": reference.target_nll,
                                "method_nll": observation.target_nll,
                                "excess_nll": observation.target_nll - reference.target_nll,
                                "kl": kl,
                                "top1_agreement": agreement,
                                "local_codec_sse": observation.local_codec_sse,
                                "trajectory_nmse": observation.trajectory_nmse,
                                "decode_model_forward_latency_ns": observation.latency_ns,
                                "decode_cuda_diagnostic_peak_allocated_bytes": (
                                    observation.peak_allocated_bytes
                                ),
                                "decode_cuda_diagnostic_peak_reserved_bytes": (
                                    observation.peak_reserved_bytes
                                ),
                                "decode_logical_recurrent_resident_bytes": (
                                    observation.resident_bytes
                                ),
                                "method_cumulative_cache_reported_workspace_high_water_sum_bytes": (
                                    observation.transient_bytes
                                ),
                                "finite_checks": {
                                    "comparison_logits": True,
                                    "nll": True,
                                    "kl": True,
                                    "local_codec_sse": True,
                                    "trajectory_nmse": True,
                                },
                            }
                        )
                    )
                if len(current_rows) != len(target_ids) - 1:
                    raise StageAError("Stage-A method omitted a cache-exposed transition")
            except BaseException as error:
                try:
                    engine.end_method(session)
                except BaseException as cleanup_error:
                    error.add_note(f"Stage-A cache observer cleanup also failed: {cleanup_error!r}")
                raise
            else:
                summary = dict(engine.end_method(session))
            method_runtime.append(
                MappingProxyType(
                    {
                        "family": family,
                        "canonical_id": canonical_id,
                        "selection_rank": rank,
                        "identity_record_sha256": identity_hash,
                        "method_id": method.method_id,
                        "policy_file_sha256": method.policy_file_sha256,
                        "policy_origin": method.origin,
                        "prefill_diagnostics": {
                            "model_forward_latency_ns": prefill.latency_ns,
                            "cuda_diagnostic_peak_allocated_bytes": (prefill.peak_allocated_bytes),
                            "cuda_diagnostic_peak_reserved_bytes": prefill.peak_reserved_bytes,
                            "logical_recurrent_resident_bytes": prefill.resident_bytes,
                            "cache_reported_workspace_high_water_sum_bytes": (
                                prefill.transient_bytes
                            ),
                        },
                        "max_cuda_diagnostic_peak_allocated_bytes_across_prefill_and_decode": max(
                            [prefill.peak_allocated_bytes]
                            + [row.peak_allocated_bytes for row in current_rows]
                        ),
                        "max_cuda_diagnostic_peak_reserved_bytes_across_prefill_and_decode": max(
                            [prefill.peak_reserved_bytes]
                            + [row.peak_reserved_bytes for row in current_rows]
                        ),
                        "wall_time_with_diagnostics_ns": (time.perf_counter_ns() - method_started),
                        "storage": dict(_storage_receipt(method.method_id, summary)),
                    }
                )
            )
        if len(references) != len(target_ids) - 1:
            raise StageAError("FP32 reference trace is incomplete")
        references.clear()
    if forwards != authenticated.bootstrap_identity.expected_forward_count:
        raise StageAError(
            "Stage-A forward count differs from 9*sum(1+m-1): "
            f"expected {authenticated.bootstrap_identity.expected_forward_count}, "
            f"observed {forwards}"
        )
    device_runtime = _validated_device_runtime(dict(engine.runtime_snapshot(model)))
    return StageAEvaluation(
        examples=tuple(examples),
        gate_rows=tuple(gate_rows),
        raw_rows=tuple(raw_rows),
        forward_count=forwards,
        method_runtime=tuple(method_runtime),
        device_runtime=device_runtime,
    )


def _expected_execution_contract(expected_forward_count: int) -> dict[str, object]:
    if (
        isinstance(expected_forward_count, bool)
        or not isinstance(expected_forward_count, int)
        or expected_forward_count <= 0
    ):
        raise StageAError("Stage-A expected forward count is invalid")
    return {
        "method_order": list(METHOD_ORDER),
        "model_load_count": 1,
        "forward_formula": "9*sum(1+m-1)",
        "expected_forward_count": expected_forward_count,
        "observed_forward_count": expected_forward_count,
        "one_prefill_per_method_example": True,
        "one_token_transitions_per_method_example": True,
        "trajectory_nmse": (
            "per-token mean of per-layer FP64 NMSE against the matched FP32 recurrent trajectory"
        ),
        "measurement_scope": {
            "decode_logical_recurrent_resident_bytes": (
                "logical persistent recurrent-state bytes exposed after each scored decode; "
                "FP32 is the exact 18,874,368-byte tensor ledger and packed methods use "
                "checkpoint bytes"
            ),
            "method_cumulative_cache_reported_workspace_high_water_sum_bytes": (
                "sum of independently tracked cache workspace high-water marks accumulated "
                "since method start, including prefill; not per-decode transient memory and "
                "not a simultaneous allocator peak"
            ),
            "decode_cuda_diagnostic_peak_bytes": (
                "CUDA allocator peak since the per-forward reset for a scored decode, "
                "including observer/cache and diagnostic state materialization; not "
                "deployment HBM"
            ),
            "decode_model_forward_latency_ns": (
                "synchronized scored-decode model forward including observer/cache update, "
                "excluding post-forward trajectory diagnostics"
            ),
            "prefill_diagnostics": (
                "separate synchronized 4096-or-identity-length prefill forward and CUDA "
                "allocator diagnostics; no prefill quality metric is scored"
            ),
            "method_runtime": (
                "wall time includes prefill, scored decodes, cache observers, and trajectory "
                "diagnostics; max CUDA fields cover both prefill and scored decodes"
            ),
        },
        "automatic_retry_authorized": False,
    }


def _materialization_receipt(materialization: object) -> dict[str, object]:
    receipt = {
        "sequence_count": 12,
        "capture_input_sha256": getattr(materialization, "capture_input_sha256", None),
        "token_sequence_manifest_sha256": getattr(
            materialization,
            "token_sequence_manifest_sha256",
            None,
        ),
        "tokenizer_manifest_sha256": getattr(
            materialization,
            "tokenizer_manifest_sha256",
            None,
        ),
    }
    for name in (
        "capture_input_sha256",
        "token_sequence_manifest_sha256",
        "tokenizer_manifest_sha256",
    ):
        _require_sha256(receipt[name], context=f"Stage-A materialization {name}")
    return receipt


def build_execution_artifact(
    authenticated: AuthenticatedStageA,
    materialization: object,
    evaluation: StageAEvaluation,
    reservation: AttemptReservation,
) -> bytes:
    gate = importlib.import_module("recurquant.experiment013_stage_a")
    gate_bytes = gate.build_stage_a_evidence_artifact(
        evaluation.examples,
        evaluation.gate_rows,
        stage_a_identity_file_sha256=authenticated.bootstrap_identity.file_sha256,
        stage_a_calibration_binding_file_sha256=authenticated.binding.file_sha256,
    )
    verified_gate = gate.deserialize_stage_a_evidence_artifact(
        gate_bytes,
        expected_stage_a_identity_file_sha256=authenticated.bootstrap_identity.file_sha256,
        expected_stage_a_calibration_binding_file_sha256=authenticated.binding.file_sha256,
    )
    receipt = reservation.receipt
    expected_runtime_manifest_hash = _require_sha256(
        authenticated.bootstrap_identity.execution_bindings.get(
            "calibration_runtime_manifest_file_sha256"
        ),
        context="Stage-A runtime execution binding",
    )
    live_runtime_record = _authenticated_runtime_record(
        authenticated.authenticated_runtime,
        expected_manifest_file_sha256=expected_runtime_manifest_hash,
    )
    receipt_runtime = receipt.get("post_load_authenticated_runtime")
    if not isinstance(receipt_runtime, Mapping):
        raise StageAError("Stage-A attempt receipt does not bind the post-load runtime")
    runtime_record = _validated_authenticated_runtime(
        receipt_runtime,
        expected_manifest_file_sha256=expected_runtime_manifest_hash,
    )
    if dict(runtime_record) != dict(live_runtime_record):
        raise StageAError("Stage-A authenticated runtime drifted after its post-load receipt")
    receipt_device = receipt.get("post_load_device_runtime")
    if not isinstance(receipt_device, Mapping):
        raise StageAError("Stage-A attempt receipt does not bind the post-load device")
    device_record = _validated_device_runtime(receipt_device)
    if dict(device_record) != dict(evaluation.device_runtime):
        raise StageAError("Stage-A device runtime drifted after its post-load receipt")
    raw_smoke = receipt.get("preseal_engine_smoke")
    if not isinstance(raw_smoke, Mapping):
        raise StageAError("Stage-A attempt receipt does not bind the pre-seal engine smoke")
    preseal_smoke = _validated_preseal_engine_smoke(raw_smoke)
    preseal_smoke_sha256 = sha256_bytes(canonical_json_bytes(dict(preseal_smoke)))
    if preseal_smoke_sha256 != _require_sha256(
        receipt.get("preseal_engine_smoke_sha256"),
        context="attempt pre-seal engine smoke SHA-256",
    ):
        raise StageAError("Stage-A pre-seal engine smoke receipt hash drifted")
    one_run = {
        "attempt_schema": receipt.get("schema"),
        "automatic_retry_authorized": receipt.get("automatic_retry_authorized"),
        "h0_source_commit": receipt.get("h0_source_commit"),
        "h1_identity_commit": reservation.h1_commit,
        "identity_scoped_attempt_lock_file_sha256": receipt.get(
            "identity_scoped_attempt_lock_file_sha256"
        ),
        "one_run_marker": receipt.get("one_run_marker"),
        "one_run_seal_commit": reservation.seal_commit,
        "one_run_seal_message_sha256": receipt.get("one_run_seal_message_sha256"),
        "one_run_seal_tree": reservation.tree,
        "preseal_engine_smoke_sha256": preseal_smoke_sha256,
        STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        "stage_a_input_bundle_manifest_file_sha256": _require_sha256(
            authenticated.input_bundle_manifest_file_sha256,
            context="authenticated Stage-A input bundle manifest SHA-256",
        ),
    }
    evidence = {
        "artifact_revision": RUNNER_REVISION,
        "claim_boundary": CLAIM_BOUNDARY,
        "dependencies": {
            "stage_a_identity_file_sha256": authenticated.bootstrap_identity.file_sha256,
            "stage_a_calibration_binding_file_sha256": authenticated.binding.file_sha256,
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
                authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
            ),
            "repository_source_manifest_file_sha256": authenticated.source_manifest_file_sha256,
            "stage_a_input_bundle_manifest_file_sha256": _require_sha256(
                authenticated.input_bundle_manifest_file_sha256,
                context="authenticated Stage-A input bundle manifest SHA-256",
            ),
            "execution_bindings": dict(authenticated.bootstrap_identity.execution_bindings),
            "method_specs": _method_spec_receipts(authenticated.methods),
        },
        "execution_contract": _expected_execution_contract(evaluation.forward_count),
        "one_run": one_run,
        "preseal_engine_smoke": dict(preseal_smoke),
        "materialization": _materialization_receipt(materialization),
        "runtime": {
            "authenticated_runtime": dict(runtime_record),
            "device": dict(device_record),
        },
        "method_runtime": [dict(row) for row in evaluation.method_runtime],
        "raw_token_evidence": [dict(row) for row in evaluation.raw_rows],
        "stage_a_gate_artifact": _strict_json(gate_bytes, context="Stage-A gate artifact"),
        "stage_a_gate_file_sha256": verified_gate.file_sha256,
        "stage_a_passed": verified_gate.passed,
    }
    document = {
        "artifact_kind": EXECUTION_ARTIFACT_KIND,
        "schema_version": EXECUTION_ARTIFACT_SCHEMA,
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(document)
    verify_execution_artifact(
        payload,
        expected_identity_file_sha256=authenticated.bootstrap_identity.file_sha256,
        expected_calibration_binding_file_sha256=authenticated.binding.file_sha256,
        expected_stage_a_capture_provenance_receipt_file_sha256=(
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        expected_h1_commit=reservation.h1_commit,
        expected_seal_commit=reservation.seal_commit,
        expected_source_commit=authenticated.source_commit,
        expected_source_manifest_file_sha256=authenticated.source_manifest_file_sha256,
        expected_input_bundle_manifest_file_sha256=_require_sha256(
            authenticated.input_bundle_manifest_file_sha256,
            context="authenticated Stage-A input bundle manifest SHA-256",
        ),
        expected_execution_bindings=authenticated.bootstrap_identity.execution_bindings,
        expected_method_specs=_method_spec_receipts(authenticated.methods),
        expected_materialization=_materialization_receipt(materialization),
        expected_seal_tree=reservation.tree,
        expected_seal_message_sha256=_require_sha256(
            receipt.get("one_run_seal_message_sha256"),
            context="attempt one-run seal message SHA-256",
        ),
        expected_attempt_lock_file_sha256=_require_sha256(
            receipt.get("identity_scoped_attempt_lock_file_sha256"),
            context="attempt identity-scoped lock SHA-256",
        ),
        expected_authenticated_runtime=runtime_record,
        expected_device_runtime=device_record,
        expected_forward_count=authenticated.bootstrap_identity.expected_forward_count,
    )
    return payload


def _nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StageAError(f"{context} is not a nonnegative integer")
    return value


def _finite_number(
    value: object,
    *,
    context: str,
    nonnegative: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageAError(f"{context} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise StageAError(f"{context} is invalid or non-finite")
    return result


def _bounded_text(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise StageAError(f"{context} is invalid")
    return value


def verify_execution_artifact(
    data: bytes,
    *,
    expected_identity_file_sha256: str,
    expected_calibration_binding_file_sha256: str,
    expected_stage_a_capture_provenance_receipt_file_sha256: str,
    expected_h1_commit: str,
    expected_seal_commit: str,
    expected_source_commit: str,
    expected_source_manifest_file_sha256: str,
    expected_input_bundle_manifest_file_sha256: str,
    expected_execution_bindings: Mapping[str, str],
    expected_method_specs: Sequence[Mapping[str, object]],
    expected_materialization: Mapping[str, object],
    expected_seal_tree: str,
    expected_seal_message_sha256: str,
    expected_attempt_lock_file_sha256: str,
    expected_authenticated_runtime: Mapping[str, object],
    expected_device_runtime: Mapping[str, object],
    expected_forward_count: int,
) -> Mapping[str, object]:
    identity_hash = _require_sha256(
        expected_identity_file_sha256,
        context="expected Stage-A identity file SHA-256",
    )
    binding_hash = _require_sha256(
        expected_calibration_binding_file_sha256,
        context="expected Stage-A calibration binding file SHA-256",
    )
    capture_provenance_hash = _require_sha256(
        expected_stage_a_capture_provenance_receipt_file_sha256,
        context="expected Stage-A capture provenance receipt file SHA-256",
    )
    h1 = _require_sha1(expected_h1_commit, context="expected Stage-A H1 commit")
    seal = _require_sha1(expected_seal_commit, context="expected Stage-A seal commit")
    source_commit = _require_sha1(
        expected_source_commit,
        context="expected Stage-A H0 source commit",
    )
    source_manifest_hash = _require_sha256(
        expected_source_manifest_file_sha256,
        context="expected Stage-A source manifest file SHA-256",
    )
    input_bundle_manifest_hash = _require_sha256(
        expected_input_bundle_manifest_file_sha256,
        context="expected Stage-A input bundle manifest file SHA-256",
    )
    seal_tree = _require_sha1(expected_seal_tree, context="expected Stage-A seal tree")
    seal_message_hash = _require_sha256(
        expected_seal_message_sha256,
        context="expected Stage-A seal message SHA-256",
    )
    attempt_lock_hash = _require_sha256(
        expected_attempt_lock_file_sha256,
        context="expected Stage-A attempt lock SHA-256",
    )
    expected_bindings = dict(expected_execution_bindings)
    _exact_fields(
        expected_bindings,
        EXECUTION_BINDING_FIELDS,
        context="expected Stage-A execution bindings",
    )
    for name, digest in expected_bindings.items():
        _require_sha256(digest, context=f"expected Stage-A execution binding {name}")
    trusted_runtime = _validated_authenticated_runtime(
        expected_authenticated_runtime,
        expected_manifest_file_sha256=expected_bindings["calibration_runtime_manifest_file_sha256"],
    )
    trusted_device = _validated_device_runtime(expected_device_runtime)
    if isinstance(expected_method_specs, (str, bytes, bytearray)):
        raise StageAError("expected Stage-A method specifications are invalid")
    expected_specs: list[dict[str, object]] = []
    if len(expected_method_specs) != len(METHOD_ORDER):
        raise StageAError("expected Stage-A method specifications are incomplete")
    for method_id, raw_spec in zip(METHOD_ORDER, expected_method_specs, strict=True):
        if not isinstance(raw_spec, Mapping):
            raise StageAError("expected Stage-A method specification is invalid")
        _exact_fields(
            raw_spec,
            {"method_id", "policy_file_sha256", "policy_origin"},
            context=f"expected Stage-A method specification {method_id}",
        )
        if raw_spec.get("method_id") != method_id:
            raise StageAError("expected Stage-A method specifications are reordered")
        policy_hash = raw_spec.get("policy_file_sha256")
        if method_id in {FP32_METHOD, DYNAMIC_K27030_METHOD}:
            if policy_hash is not None:
                raise StageAError(f"expected {method_id} policy hash must be null")
        else:
            _require_sha256(policy_hash, context=f"expected {method_id} policy SHA-256")
        _bounded_text(raw_spec.get("policy_origin"), context=f"expected {method_id} origin")
        expected_specs.append(dict(raw_spec))
    if (
        isinstance(expected_forward_count, bool)
        or not isinstance(expected_forward_count, int)
        or expected_forward_count <= 0
    ):
        raise StageAError("expected Stage-A forward count is invalid")
    expected_materialization_receipt = dict(expected_materialization)
    _exact_fields(
        expected_materialization_receipt,
        {
            "sequence_count",
            "capture_input_sha256",
            "token_sequence_manifest_sha256",
            "tokenizer_manifest_sha256",
        },
        context="expected Stage-A materialization receipt",
    )
    if expected_materialization_receipt.get("sequence_count") != 12:
        raise StageAError("expected Stage-A materialization count drifted")
    for name in (
        "capture_input_sha256",
        "token_sequence_manifest_sha256",
        "tokenizer_manifest_sha256",
    ):
        _require_sha256(
            expected_materialization_receipt.get(name),
            context=f"expected Stage-A materialization {name}",
        )
    root = _strict_json(data, context="Stage-A execution artifact")
    _exact_fields(
        root,
        {"artifact_kind", "schema_version", "canonical_evidence_sha256", "evidence"},
        context="Stage-A execution artifact",
    )
    if canonical_json_bytes(root) != data:
        raise StageAError("Stage-A execution artifact is not canonical JSON")
    if (
        root["artifact_kind"] != EXECUTION_ARTIFACT_KIND
        or root["schema_version"] != EXECUTION_ARTIFACT_SCHEMA
    ):
        raise StageAError("Stage-A execution artifact kind or schema drifted")
    evidence = root.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StageAError("Stage-A execution evidence is missing")
    _exact_fields(
        evidence,
        {
            "artifact_revision",
            "claim_boundary",
            "dependencies",
            "execution_contract",
            "materialization",
            "method_runtime",
            "one_run",
            "preseal_engine_smoke",
            "raw_token_evidence",
            "runtime",
            "stage_a_gate_artifact",
            "stage_a_gate_file_sha256",
            "stage_a_passed",
        },
        context="Stage-A execution evidence",
    )
    if root.get("canonical_evidence_sha256") != sha256_bytes(canonical_json_bytes(evidence)):
        raise StageAError("Stage-A execution evidence SHA-256 drifted")
    if (
        evidence.get("artifact_revision") != RUNNER_REVISION
        or evidence.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise StageAError("Stage-A execution artifact revision or claim boundary drifted")

    dependencies = evidence.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise StageAError("Stage-A execution dependencies are missing")
    _exact_fields(
        dependencies,
        {
            "stage_a_identity_file_sha256",
            "stage_a_calibration_binding_file_sha256",
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD,
            "repository_source_manifest_file_sha256",
            "stage_a_input_bundle_manifest_file_sha256",
            "execution_bindings",
            "method_specs",
        },
        context="Stage-A execution dependencies",
    )
    if (
        dependencies.get("stage_a_identity_file_sha256") != identity_hash
        or dependencies.get("stage_a_calibration_binding_file_sha256") != binding_hash
        or dependencies.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD) != capture_provenance_hash
    ):
        raise StageAError("Stage-A result identity, binding, or capture provenance drifted")
    if dependencies.get("repository_source_manifest_file_sha256") != source_manifest_hash:
        raise StageAError("Stage-A result source manifest identity drifted")
    if dependencies.get("stage_a_input_bundle_manifest_file_sha256") != input_bundle_manifest_hash:
        raise StageAError("Stage-A result input bundle identity drifted")
    execution_bindings = dependencies.get("execution_bindings")
    if not isinstance(execution_bindings, Mapping):
        raise StageAError("Stage-A result execution bindings are missing")
    _exact_fields(
        execution_bindings,
        EXECUTION_BINDING_FIELDS,
        context="Stage-A result execution bindings",
    )
    for name, digest in execution_bindings.items():
        _require_sha256(digest, context=f"Stage-A result execution binding {name}")
    if dict(execution_bindings) != expected_bindings:
        raise StageAError("Stage-A result execution bindings drifted")
    method_specs = dependencies.get("method_specs")
    if not isinstance(method_specs, list) or method_specs != expected_specs:
        raise StageAError("Stage-A result method specifications drifted")

    one_run = evidence.get("one_run")
    if not isinstance(one_run, Mapping):
        raise StageAError("Stage-A result one-run identity is missing")
    _exact_fields(
        one_run,
        {
            "attempt_schema",
            "automatic_retry_authorized",
            "h0_source_commit",
            "h1_identity_commit",
            "identity_scoped_attempt_lock_file_sha256",
            "one_run_marker",
            "one_run_seal_commit",
            "one_run_seal_message_sha256",
            "one_run_seal_tree",
            "preseal_engine_smoke_sha256",
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD,
            "stage_a_input_bundle_manifest_file_sha256",
        },
        context="Stage-A result one-run identity",
    )
    if (
        one_run.get("attempt_schema") != ATTEMPT_SCHEMA
        or one_run.get("automatic_retry_authorized") is not False
        or one_run.get("one_run_marker") != ONE_RUN_MARKER
        or one_run.get("h0_source_commit") != source_commit
        or one_run.get("h1_identity_commit") != h1
        or one_run.get("one_run_seal_commit") != seal
        or one_run.get("one_run_seal_tree") != seal_tree
        or one_run.get("one_run_seal_message_sha256") != seal_message_hash
        or one_run.get("identity_scoped_attempt_lock_file_sha256") != attempt_lock_hash
        or one_run.get("stage_a_input_bundle_manifest_file_sha256") != input_bundle_manifest_hash
        or one_run.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD) != capture_provenance_hash
    ):
        raise StageAError("Stage-A result one-run identity drifted")
    preseal_smoke = evidence.get("preseal_engine_smoke")
    if not isinstance(preseal_smoke, Mapping):
        raise StageAError("Stage-A result pre-seal engine smoke is missing")
    _validated_preseal_engine_smoke(preseal_smoke)
    if sha256_bytes(canonical_json_bytes(dict(preseal_smoke))) != _require_sha256(
        one_run.get("preseal_engine_smoke_sha256"),
        context="Stage-A result pre-seal engine smoke SHA-256",
    ):
        raise StageAError("Stage-A result pre-seal engine smoke hash drifted")

    execution_contract = evidence.get("execution_contract")
    if not isinstance(execution_contract, Mapping) or dict(execution_contract) != (
        _expected_execution_contract(expected_forward_count)
    ):
        raise StageAError("Stage-A execution contract drifted")
    materialization = evidence.get("materialization")
    if not isinstance(materialization, Mapping):
        raise StageAError("Stage-A materialization receipt is missing")
    _exact_fields(
        materialization,
        {
            "sequence_count",
            "capture_input_sha256",
            "token_sequence_manifest_sha256",
            "tokenizer_manifest_sha256",
        },
        context="Stage-A materialization receipt",
    )
    if materialization.get("sequence_count") != 12:
        raise StageAError("Stage-A materialization sequence count drifted")
    for name in (
        "capture_input_sha256",
        "token_sequence_manifest_sha256",
        "tokenizer_manifest_sha256",
    ):
        _require_sha256(
            materialization.get(name),
            context=f"Stage-A materialization {name}",
        )
    if dict(materialization) != expected_materialization_receipt:
        raise StageAError("Stage-A materialization receipt drifted")

    runtime = evidence.get("runtime")
    if not isinstance(runtime, Mapping):
        raise StageAError("Stage-A result runtime evidence is missing")
    _exact_fields(runtime, {"authenticated_runtime", "device"}, context="Stage-A runtime")
    authenticated_runtime = runtime.get("authenticated_runtime")
    if not isinstance(authenticated_runtime, Mapping):
        raise StageAError("Stage-A authenticated runtime evidence is missing")
    observed_runtime = _validated_authenticated_runtime(
        authenticated_runtime,
        expected_manifest_file_sha256=expected_bindings["calibration_runtime_manifest_file_sha256"],
    )
    if dict(observed_runtime) != dict(trusted_runtime):
        raise StageAError("Stage-A authenticated runtime evidence drifted")
    device = runtime.get("device")
    if not isinstance(device, Mapping):
        raise StageAError("Stage-A device evidence is missing")
    observed_device = _validated_device_runtime(device)
    if dict(observed_device) != dict(trusted_device):
        raise StageAError("Stage-A device runtime evidence drifted")

    rows = evidence.get("raw_token_evidence")
    if not isinstance(rows, list) or not rows:
        raise StageAError("Stage-A result has no raw transition evidence")
    gate_artifact = evidence.get("stage_a_gate_artifact")
    if not isinstance(gate_artifact, Mapping):
        raise StageAError("Stage-A gate artifact is missing")
    gate = importlib.import_module("recurquant.experiment013_stage_a")
    gate_bytes = gate.canonical_json_bytes(dict(gate_artifact))
    if sha256_bytes(gate_bytes) != _require_sha256(
        evidence.get("stage_a_gate_file_sha256"),
        context="Stage-A gate artifact file SHA-256",
    ):
        raise StageAError("Stage-A gate artifact file identity drifted")
    try:
        verified_gate = gate.deserialize_stage_a_evidence_artifact(
            gate_bytes,
            expected_stage_a_identity_file_sha256=identity_hash,
            expected_stage_a_calibration_binding_file_sha256=binding_hash,
        )
    except (TypeError, ValueError) as error:
        raise StageAError("Stage-A gate artifact reconstruction failed") from error
    if (
        not isinstance(evidence.get("stage_a_passed"), bool)
        or evidence.get("stage_a_passed") is not verified_gate.passed
    ):
        raise StageAError("Stage-A result passage scalar is invalid")
    gate_evidence = verified_gate.evidence
    gate_examples = gate_evidence.get("examples")
    gate_rows = gate_evidence.get("token_rows")
    if (
        not isinstance(gate_examples, tuple)
        or len(gate_examples) != 12
        or not isinstance(gate_rows, tuple)
        or len(gate_rows) != len(rows)
    ):
        raise StageAError("Stage-A outer and gate row counts differ")
    derived_forward_count = 0
    for example in gate_examples:
        if not isinstance(example, Mapping):
            raise StageAError("Stage-A gate example is invalid")
        continuation_count = _nonnegative_int(
            example.get("continuation_token_count"),
            context="Stage-A gate continuation token count",
        )
        if continuation_count < 2:
            raise StageAError("Stage-A gate continuation is too short")
        derived_forward_count += len(METHOD_ORDER) * continuation_count
    if derived_forward_count != expected_forward_count:
        raise StageAError("Stage-A gate examples differ from the authenticated forward count")
    shared_fields = (
        "canonical_id",
        "family",
        "identity_record_sha256",
        "kl",
        "method_id",
        "method_nll",
        "reference_nll",
        "selection_rank",
        "top1_agreement",
        "transition_index",
    )
    position_by_transition: dict[tuple[object, ...], tuple[int, int]] = {}
    first_input_by_method_example: dict[tuple[object, ...], int] = {}
    rows_by_method_example: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    finite_checks = {
        "comparison_logits": True,
        "nll": True,
        "kl": True,
        "local_codec_sse": True,
        "trajectory_nmse": True,
    }
    for row_index, (outer_row, gate_row) in enumerate(zip(rows, gate_rows, strict=True)):
        if not isinstance(outer_row, Mapping) or not isinstance(gate_row, Mapping):
            raise StageAError("Stage-A transition evidence is invalid")
        if any(
            name in outer_row
            for name in (
                "input_token_id",
                "input_token_ids_sha256",
                "target_token_id",
                "target_token_ids_sha256",
            )
        ):
            raise StageAError("Stage-A result exposes low-entropy per-token evidence")
        _exact_fields(
            outer_row,
            RAW_TOKEN_EVIDENCE_FIELDS,
            context=f"Stage-A raw transition evidence {row_index}",
        )
        if any(outer_row.get(name) != gate_row.get(name) for name in shared_fields):
            raise StageAError("Stage-A outer transition evidence differs from the gate")
        method_id = outer_row.get("method_id")
        if method_id not in METHOD_ORDER:
            raise StageAError("Stage-A raw transition method is unknown")
        transition_index = _nonnegative_int(
            outer_row.get("transition_index"),
            context="Stage-A transition index",
        )
        input_position = _nonnegative_int(
            outer_row.get("input_position"),
            context="Stage-A input position",
        )
        target_position = _nonnegative_int(
            outer_row.get("target_position"),
            context="Stage-A target position",
        )
        if target_position != input_position + 1:
            raise StageAError("Stage-A transition positions are not causal")
        example_key = (
            outer_row.get("family"),
            outer_row.get("canonical_id"),
            outer_row.get("selection_rank"),
            outer_row.get("identity_record_sha256"),
        )
        method_example_key = (*example_key, method_id)
        first_input = first_input_by_method_example.setdefault(
            method_example_key,
            input_position - transition_index,
        )
        if first_input < 1 or input_position != first_input + transition_index:
            raise StageAError("Stage-A transition positions are incomplete or reordered")
        transition_key = (*example_key, transition_index)
        reference_positions = position_by_transition.setdefault(
            transition_key,
            (input_position, target_position),
        )
        if reference_positions != (input_position, target_position):
            raise StageAError("Stage-A methods used different causal positions")
        if outer_row.get("authenticated_transition_sha256") != _transition_hash(
            identity_record_sha256=_require_sha256(
                outer_row.get("identity_record_sha256"),
                context="Stage-A raw identity record SHA-256",
            ),
            method_id=cast(str, method_id),
            transition_index=transition_index,
            input_position=input_position,
            target_position=target_position,
        ):
            raise StageAError("Stage-A transition commitment drifted")
        reference_nll = _finite_number(
            outer_row.get("reference_nll"),
            context="Stage-A reference NLL",
            nonnegative=True,
        )
        method_nll = _finite_number(
            outer_row.get("method_nll"),
            context="Stage-A method NLL",
            nonnegative=True,
        )
        excess_nll = _finite_number(
            outer_row.get("excess_nll"),
            context="Stage-A excess NLL",
            nonnegative=False,
        )
        if excess_nll != method_nll - reference_nll:
            raise StageAError("Stage-A excess NLL differs from its component values")
        for name in ("kl", "local_codec_sse", "trajectory_nmse"):
            _finite_number(
                outer_row.get(name),
                context=f"Stage-A {name}",
                nonnegative=True,
            )
        if not isinstance(outer_row.get("top1_agreement"), bool):
            raise StageAError("Stage-A top-1 agreement is not boolean")
        for name in (
            "decode_model_forward_latency_ns",
            "decode_cuda_diagnostic_peak_allocated_bytes",
            "decode_cuda_diagnostic_peak_reserved_bytes",
            "decode_logical_recurrent_resident_bytes",
            "method_cumulative_cache_reported_workspace_high_water_sum_bytes",
        ):
            _nonnegative_int(outer_row.get(name), context=f"Stage-A {name}")
        if (
            outer_row.get("decode_logical_recurrent_resident_bytes")
            != (EXPECTED_RECURRENT_RESIDENT_BYTES[cast(str, method_id)])
        ):
            raise StageAError("Stage-A recurrent resident-byte ledger drifted")
        checks = outer_row.get("finite_checks")
        if not isinstance(checks, Mapping) or dict(checks) != finite_checks:
            raise StageAError("Stage-A finite-check receipt drifted")
        rows_by_method_example.setdefault(method_example_key, []).append(outer_row)

    method_runtime = evidence.get("method_runtime")
    expected_runtime_count = len(gate_examples) * len(METHOD_ORDER)
    if not isinstance(method_runtime, list) or len(method_runtime) != expected_runtime_count:
        raise StageAError("Stage-A method runtime inventory is incomplete")
    runtime_index = 0
    for example in gate_examples:
        continuation_count = cast(int, example["continuation_token_count"])
        example_key = (
            example["family"],
            example["canonical_id"],
            example["selection_rank"],
            example["identity_record_sha256"],
        )
        for method_id in METHOD_ORDER:
            runtime_row = method_runtime[runtime_index]
            runtime_index += 1
            if not isinstance(runtime_row, Mapping):
                raise StageAError("Stage-A method runtime row is invalid")
            _exact_fields(
                runtime_row,
                METHOD_RUNTIME_FIELDS,
                context=f"Stage-A method runtime row {runtime_index - 1}",
            )
            if (
                tuple(
                    runtime_row.get(name)
                    for name in (
                        "family",
                        "canonical_id",
                        "selection_rank",
                        "identity_record_sha256",
                    )
                )
                != example_key
                or runtime_row.get("method_id") != method_id
            ):
                raise StageAError("Stage-A method runtime rows are reordered or misbound")
            policy_hash = runtime_row.get("policy_file_sha256")
            if policy_hash is not None:
                _require_sha256(policy_hash, context="Stage-A method policy file SHA-256")
            policy_origin = _bounded_text(
                runtime_row.get("policy_origin"),
                context="Stage-A policy origin",
            )
            expected_spec = expected_specs[METHOD_ORDER.index(method_id)]
            if (
                policy_hash != expected_spec["policy_file_sha256"]
                or policy_origin != expected_spec["policy_origin"]
            ):
                raise StageAError("Stage-A method runtime policy identity drifted")
            _nonnegative_int(
                runtime_row.get("wall_time_with_diagnostics_ns"),
                context="Stage-A method wall time",
            )
            prefill = runtime_row.get("prefill_diagnostics")
            if not isinstance(prefill, Mapping):
                raise StageAError("Stage-A prefill diagnostics are missing")
            _exact_fields(
                prefill,
                PREFILL_DIAGNOSTIC_FIELDS,
                context="Stage-A prefill diagnostics",
            )
            for name in PREFILL_DIAGNOSTIC_FIELDS:
                _nonnegative_int(prefill.get(name), context=f"Stage-A prefill {name}")
            expected_resident = EXPECTED_RECURRENT_RESIDENT_BYTES[method_id]
            if prefill.get("logical_recurrent_resident_bytes") != expected_resident:
                raise StageAError("Stage-A prefill resident-byte ledger drifted")
            group = rows_by_method_example.get((*example_key, method_id), [])
            if len(group) != continuation_count - 1:
                raise StageAError("Stage-A method runtime has an incomplete decode trajectory")
            max_allocated = _nonnegative_int(
                runtime_row.get(
                    "max_cuda_diagnostic_peak_allocated_bytes_across_prefill_and_decode"
                ),
                context="Stage-A method maximum allocated bytes",
            )
            max_reserved = _nonnegative_int(
                runtime_row.get(
                    "max_cuda_diagnostic_peak_reserved_bytes_across_prefill_and_decode"
                ),
                context="Stage-A method maximum reserved bytes",
            )
            expected_max_allocated = max(
                cast(int, prefill["cuda_diagnostic_peak_allocated_bytes"]),
                *(cast(int, row["decode_cuda_diagnostic_peak_allocated_bytes"]) for row in group),
            )
            expected_max_reserved = max(
                cast(int, prefill["cuda_diagnostic_peak_reserved_bytes"]),
                *(cast(int, row["decode_cuda_diagnostic_peak_reserved_bytes"]) for row in group),
            )
            if max_allocated != expected_max_allocated or max_reserved != expected_max_reserved:
                raise StageAError("Stage-A method CUDA maximum diagnostics drifted")
            storage = runtime_row.get("storage")
            if not isinstance(storage, Mapping):
                raise StageAError("Stage-A method storage receipt is missing")
            _exact_fields(
                storage,
                STORAGE_RECEIPT_FIELDS,
                context="Stage-A method storage receipt",
            )
            for name in STORAGE_RECEIPT_FIELDS - {"workspace_scope"}:
                _nonnegative_int(storage.get(name), context=f"Stage-A storage {name}")
            if (
                storage.get("logical_recurrent_resident_bytes") != expected_resident
                or storage.get("cache_reported_resident_bytes") != expected_resident
                or storage.get("cache_reported_expected_resident_bytes") != expected_resident
                or storage.get("workspace_scope")
                != "method_lifetime_high_water_since_cache_creation"
            ):
                raise StageAError("Stage-A method storage ledger drifted")
            final_workspace = cast(
                int,
                group[-1]["method_cumulative_cache_reported_workspace_high_water_sum_bytes"],
            )
            if final_workspace != cast(int, storage["raw_state_workspace_high_water_bytes"]) + cast(
                int, storage["query_workspace_high_water_bytes"]
            ):
                raise StageAError("Stage-A method workspace high-water receipt drifted")
    return MappingProxyType(dict(root))


def _completion_marker_bytes(
    config: StageAConfig,
    *,
    result_file_sha256: str,
) -> bytes:
    result_hash = _require_sha256(
        result_file_sha256,
        context="completion marker result SHA-256",
    )
    attempt_bytes = _stable_file_bytes(
        config.attempt_path,
        context="completed Stage-A attempt receipt",
    )
    return canonical_json_bytes(
        {
            "attempt_file_sha256": sha256_bytes(attempt_bytes),
            "result_file_sha256": result_hash,
            "status": "complete",
        }
    )


def publish_result(
    config: StageAConfig,
    reservation: AttemptReservation,
    payload: bytes,
    summary: Mapping[str, object],
) -> Mapping[str, object]:
    verify_execution_artifact(
        payload,
        expected_identity_file_sha256=_require_sha256(
            reservation.receipt.get("identity_file_sha256"),
            context="attempt identity file SHA-256",
        ),
        expected_calibration_binding_file_sha256=_require_sha256(
            reservation.receipt.get("calibration_binding_file_sha256"),
            context="attempt calibration binding file SHA-256",
        ),
        expected_stage_a_capture_provenance_receipt_file_sha256=_require_sha256(
            reservation.receipt.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD),
            context="attempt Stage-A capture provenance receipt SHA-256",
        ),
        expected_h1_commit=reservation.h1_commit,
        expected_seal_commit=reservation.seal_commit,
        expected_source_commit=_require_sha1(
            reservation.receipt.get("h0_source_commit"),
            context="attempt H0 source commit",
        ),
        expected_source_manifest_file_sha256=_require_sha256(
            reservation.receipt.get("source_manifest_file_sha256"),
            context="attempt source manifest file SHA-256",
        ),
        expected_input_bundle_manifest_file_sha256=_require_sha256(
            reservation.receipt.get("stage_a_input_bundle_manifest_file_sha256"),
            context="attempt Stage-A input bundle manifest SHA-256",
        ),
        expected_execution_bindings=cast(
            Mapping[str, str],
            reservation.receipt.get("execution_bindings"),
        ),
        expected_method_specs=cast(
            Sequence[Mapping[str, object]],
            reservation.receipt.get("method_specs"),
        ),
        expected_materialization={
            "sequence_count": 12,
            "capture_input_sha256": reservation.receipt.get("capture_input_sha256"),
            "token_sequence_manifest_sha256": reservation.receipt.get(
                "token_sequence_manifest_sha256"
            ),
            "tokenizer_manifest_sha256": reservation.receipt.get("tokenizer_manifest_sha256"),
        },
        expected_seal_tree=reservation.tree,
        expected_seal_message_sha256=_require_sha256(
            reservation.receipt.get("one_run_seal_message_sha256"),
            context="attempt seal message SHA-256",
        ),
        expected_attempt_lock_file_sha256=_require_sha256(
            reservation.receipt.get("identity_scoped_attempt_lock_file_sha256"),
            context="attempt identity-scoped lock SHA-256",
        ),
        expected_authenticated_runtime=cast(
            Mapping[str, object],
            reservation.receipt.get("post_load_authenticated_runtime"),
        ),
        expected_device_runtime=cast(
            Mapping[str, object],
            reservation.receipt.get("post_load_device_runtime"),
        ),
        expected_forward_count=_nonnegative_int(
            reservation.receipt.get("expected_forward_count"),
            context="attempt expected forward count",
        ),
    )
    file_hash = sha256_bytes(payload)
    document = _strict_json(payload, context="prepared Stage-A result")
    canonical_hash = _require_sha256(
        document.get("canonical_evidence_sha256"), context="result canonical evidence SHA-256"
    )
    prepared = persist_receipt(
        config,
        reservation,
        {
            "status": "result_prepared_before_atomic_publication",
            "observed_forward_count": reservation.receipt["expected_forward_count"],
            "content_materialized": True,
            "model_load_count": 1,
            "evaluation_complete": True,
            "result_available": False,
            "result_file_sha256": file_hash,
            "result_canonical_evidence_sha256": canonical_hash,
            "stage_a_passed": bool(summary["stage_a_passed"]),
        },
    )
    _atomic_publish_new(config.output_path, payload)
    if _stable_file_bytes(config.output_path, context="published Stage-A result") != payload:
        raise StageAError("published Stage-A output changed on disk")
    completed = persist_receipt(
        config,
        prepared,
        {
            "status": "completed_with_authenticated_stage_a_result",
            "result_available": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    marker = _completion_marker_bytes(config, result_file_sha256=file_hash)
    _atomic_publish_new(config.complete_path, marker)
    return {
        "status": completed.receipt["status"],
        "output": str(config.output_path.resolve()),
        "artifact_file_sha256": file_hash,
        "canonical_evidence_sha256": canonical_hash,
        "stage_a_passed": summary["stage_a_passed"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def record_failure(
    config: StageAConfig,
    reservation: AttemptReservation | None,
    error: BaseException,
    phase: str,
) -> None:
    if reservation is None or not config.attempt_path.is_file():
        return
    try:
        disk = _strict_json(
            _stable_file_bytes(config.attempt_path, context="Stage-A attempt receipt"),
            context="Stage-A attempt receipt",
        )
        for field in (
            "schema",
            "h1_identity_commit",
            "one_run_seal_commit",
            "identity_file_sha256",
            "identity_scoped_attempt_lock_file_sha256",
        ):
            if disk.get(field) != reservation.receipt.get(field):
                raise StageAError("Stage-A attempt identity changed before failure recording")
        current = dataclasses.replace(reservation, receipt=MappingProxyType(disk))
        detail_hash = sha256_bytes(str(error).encode("utf-8", errors="replace"))
        persist_receipt(
            config,
            current,
            {
                "status": "consumed_attempt_failed_no_automatic_retry",
                "failure_phase": phase,
                "failure_type": type(error).__name__,
                "failure_detail_sha256": detail_hash,
                "failure_detail_recorded": False,
                "result_available": config.output_path.is_file(),
            },
        )
    except BaseException as receipt_error:
        raise StageAError("could not preserve an authenticated failure receipt") from receipt_error


def _validated_preseal_engine_smoke(value: Mapping[str, object]) -> Mapping[str, object]:
    _exact_fields(
        value,
        {
            "profile",
            "passed",
            "stage_a_content_accessed",
            "input_profile",
            "prompt_token_count",
            "continuation_token_count",
            "model_load_count",
            "method_order",
            "forward_count",
            "method_receipts",
            "device",
        },
        context="pre-seal engine smoke",
    )
    if (
        value.get("profile") != PRESEAL_ENGINE_SMOKE_PROFILE
        or value.get("passed") is not True
        or value.get("stage_a_content_accessed") is not False
        or value.get("input_profile") != "fixed_public_synthetic_4096_plus_128_v3"
        or value.get("prompt_token_count") != PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT
        or value.get("continuation_token_count") != PRESEAL_ENGINE_SMOKE_TARGET_TOKEN_COUNT
        or value.get("model_load_count") != 1
        or value.get("method_order") != list(METHOD_ORDER)
        or value.get("forward_count") != len(METHOD_ORDER) * len(PRESEAL_ENGINE_SMOKE_TARGET)
    ):
        raise StageAError("pre-seal engine smoke contract drifted")
    receipts = value.get("method_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(METHOD_ORDER):
        raise StageAError("pre-seal engine smoke method receipts are incomplete")
    for method_id, receipt in zip(METHOD_ORDER, receipts, strict=True):
        if not isinstance(receipt, Mapping):
            raise StageAError("pre-seal engine smoke method receipt is invalid")
        _exact_fields(
            receipt,
            {
                "method_id",
                "forward_count",
                "logical_recurrent_resident_bytes",
                "equal_byte_observer_required",
            },
            context=f"pre-seal engine smoke {method_id}",
        )
        if (
            receipt.get("method_id") != method_id
            or receipt.get("forward_count") != len(PRESEAL_ENGINE_SMOKE_TARGET)
            or receipt.get("logical_recurrent_resident_bytes")
            != EXPECTED_RECURRENT_RESIDENT_BYTES[method_id]
            or receipt.get("equal_byte_observer_required") is not (method_id != FP32_METHOD)
        ):
            raise StageAError(f"pre-seal engine smoke {method_id} receipt drifted")
    device = value.get("device")
    if not isinstance(device, Mapping):
        raise StageAError("pre-seal engine smoke device receipt is missing")
    _validated_device_runtime(device)
    return MappingProxyType(
        _strict_json(canonical_json_bytes(value), context="pre-seal engine smoke")
    )


def run_stage_a(config: StageAConfig, services: StageAServices) -> Mapping[str, object]:
    """Order auth -> seal -> materialize -> load once -> evaluate -> publish."""

    reservation: AttemptReservation | None = None
    phase = "metadata_authentication"
    authenticated = services.authenticate(config)
    try:
        phase = "preseal_engine_smoke"
        smoke = _validated_preseal_engine_smoke(dict(services.preseal_smoke(authenticated)))
        phase = "post_preseal_engine_smoke_reauthentication"
        services.reauthenticate(config, authenticated, None)
        phase = "one_run_reservation"
        reservation = services.reserve(config, authenticated)
        smoke_bytes = canonical_json_bytes(dict(smoke))
        reservation = services.persist_receipt(
            config,
            reservation,
            {
                "status": "preseal_engine_smoke_bound_before_materialization",
                "preseal_engine_smoke": dict(smoke),
                "preseal_engine_smoke_sha256": sha256_bytes(smoke_bytes),
            },
        )
        phase = "pre_materialization_reauthentication"
        services.reauthenticate(config, authenticated, reservation)
        reservation = services.persist_receipt(
            config,
            reservation,
            {"status": "stage_a_materialization_entered"},
        )
        phase = "stage_a_materialization"
        materialization = services.materialize(config, authenticated)
        if getattr(materialization, "frozen_identity_file_sha256", None) != (
            authenticated.bootstrap_identity.file_sha256
        ):
            raise StageAError("materializer returned a different frozen Stage-A identity")
        if (
            getattr(materialization, "calibration_binding_file_sha256", None)
            != authenticated.binding.file_sha256
        ):
            raise StageAError("materializer returned a different calibration binding")
        reservation = services.persist_receipt(
            config,
            reservation,
            {
                "status": "stage_a_content_materialized_before_model_load",
                "content_materialized": True,
                "capture_input_sha256": materialization.capture_input_sha256,
                "token_sequence_manifest_sha256": materialization.token_sequence_manifest_sha256,
                "tokenizer_manifest_sha256": materialization.tokenizer_manifest_sha256,
            },
        )
        phase = "pre_model_reauthentication"
        services.reauthenticate(config, authenticated, reservation)
        phase = "model_load"
        model = services.engine.load_model(authenticated.authenticated_model_files)
        try:
            phase = "post_model_load_reauthentication"
            services.reauthenticate(config, authenticated, reservation)
            runtime_manifest_hash = _require_sha256(
                authenticated.bootstrap_identity.execution_bindings.get(
                    "calibration_runtime_manifest_file_sha256"
                ),
                context="Stage-A runtime execution binding",
            )
            post_load_authenticated_runtime = _authenticated_runtime_record(
                authenticated.authenticated_runtime,
                expected_manifest_file_sha256=runtime_manifest_hash,
            )
            post_load_device_runtime = _validated_device_runtime(
                dict(services.engine.runtime_snapshot(model))
            )
            phase = "post_model_load_receipt"
            reservation = services.persist_receipt(
                config,
                reservation,
                {
                    "status": "model_loaded_once_before_evaluation",
                    "model_load_count": 1,
                    "post_load_authenticated_runtime": dict(post_load_authenticated_runtime),
                    "post_load_device_runtime": dict(post_load_device_runtime),
                },
            )
            phase = "evaluation"
            evaluation = evaluate_materialized_stage_a(
                authenticated,
                materialization,
                services.engine,
                model,
            )
            phase = "post_evaluation_device_reauthentication"
            if dict(evaluation.device_runtime) != dict(post_load_device_runtime):
                raise StageAError(
                    "Stage-A device runtime drifted between model load and evaluation end"
                )
        finally:
            services.engine.close_model(model)
        reservation = services.persist_receipt(
            config,
            reservation,
            {
                "status": "evaluation_returned_before_result_build",
                "observed_forward_count": evaluation.forward_count,
                "evaluation_complete": True,
            },
        )
        phase = "post_evaluation_reauthentication"
        services.reauthenticate(config, authenticated, reservation)
        phase = "artifact_build"
        payload = build_execution_artifact(
            authenticated,
            materialization,
            evaluation,
            reservation,
        )
        document = _strict_json(payload, context="Stage-A execution artifact")
        evidence = cast(Mapping[str, object], document["evidence"])
        phase = "pre_publication_reauthentication"
        services.reauthenticate(config, authenticated, reservation)
        phase = "result_publication"
        return services.publish(
            config,
            reservation,
            payload,
            {"stage_a_passed": evidence["stage_a_passed"]},
        )
    except BaseException as error:
        services.record_failure(config, reservation, error, phase)
        raise


def _runtime_context(config: StageAConfig) -> dict[str, object]:
    if (
        config.base_runtime_root is None
        or config.git_executable_path is None
        or config.interpreter_path is None
    ):
        raise StageAError("sealed runner did not receive its runtime authentication context")
    return {
        "base_runtime_root": config.base_runtime_root,
        "git_executable": config.git_executable_path,
        "staged_interpreter": config.interpreter_path,
        "package_runtime_roots": dict(config.package_roots),
        "package_import_paths": dict(config.package_import_paths),
    }


def _read_execution_artifacts(config: StageAConfig) -> dict[str, bytes]:
    paths = {
        "repository_source_manifest_file_sha256": config.repository_source_manifest_path,
        "calibration_runtime_manifest_file_sha256": config.runtime_manifest_path,
        "model_file_manifest_file_sha256": config.model_file_manifest_path,
        "parquet_materialization_manifest_file_sha256": (
            config.parquet_materialization_manifest_path
        ),
    }
    return {
        name: _stable_file_bytes(path, context=f"execution artifact {name}")
        for name, path in paths.items()
    }


def authenticate_production(
    config: StageAConfig,
    *,
    require_input_bundle: bool = True,
) -> AuthenticatedStageA:
    _assert_output_paths_isolated(config)
    identity_bytes = _stable_file_bytes(config.frozen_identity_path, context="frozen identity")
    bootstrap = bootstrap_stage_a_identity(identity_bytes)
    binding_bytes = _stable_file_bytes(
        config.calibration_binding_path, context="Stage-A calibration binding"
    )
    capture_provenance_receipt_bytes = _stable_file_bytes(
        config.stage_a_capture_provenance_receipt_path,
        context="finalized Stage-A capture provenance receipt",
    )
    bootstrap_capture = bootstrap_stage_a_capture_provenance_receipt(
        capture_provenance_receipt_bytes,
        expected_file_sha256=(config.expected_stage_a_capture_provenance_receipt_sha256),
        calibration_binding_artifact=binding_bytes,
        identity=bootstrap,
        expected_source_commit=config.source_commit,
    )
    execution = _read_execution_artifacts(config)
    if {name: sha256_bytes(data) for name, data in execution.items()} != dict(
        bootstrap.execution_bindings
    ):
        raise StageAError("one or more execution artifacts differ from Stage-A identity v6")
    expected_cli = {
        "calibration_runtime_manifest_file_sha256": _require_sha256(
            config.expected_runtime_manifest_sha256, context="expected runtime manifest SHA-256"
        ),
        "model_file_manifest_file_sha256": _require_sha256(
            config.expected_model_file_manifest_sha256, context="expected model manifest SHA-256"
        ),
        "parquet_materialization_manifest_file_sha256": _require_sha256(
            config.expected_parquet_materialization_manifest_sha256,
            context="expected Parquet manifest SHA-256",
        ),
    }
    if any(bootstrap.execution_bindings[name] != digest for name, digest in expected_cli.items()):
        raise StageAError("explicit expected execution-artifact digest differs from identity v6")
    source_bootstrap = _bootstrap_source_manifest(
        execution["repository_source_manifest_file_sha256"]
    )
    if source_bootstrap["source_commit"] != _require_sha1(
        config.source_commit, context="explicit H0 source commit"
    ):
        raise StageAError("explicit H0 differs from the authenticated source manifest")
    source_entries = _source_entries(source_bootstrap)
    capture_source = cast(Mapping[str, object], bootstrap_capture["capture_source"])
    if capture_source.get("sha256") != source_entries[CAPTURE_SOURCE_PATH]["raw_sha256"]:
        raise StageAError("Stage-A capture provenance source differs from authenticated H0")
    _verify_source_bytes(source_bootstrap, config.repository_root)
    entries = source_entries
    _install_source_namespace(config.repository_root)
    source_module = _load_exact_module(
        "recurquant.experiment013_source",
        SOURCE_MODULE_PATH,
        repository_root=config.repository_root,
        entries=entries,
    )
    calibration_runner = _load_exact_module(
        "_recurquant_experiment013_calibration_runner_for_stage_a",
        CALIBRATION_RUNNER_SOURCE_PATH,
        repository_root=config.repository_root,
        entries=entries,
    )
    resolver = _load_exact_module(
        "recurquant_experiment013_identity_resolver",
        RESOLVER_SOURCE_PATH,
        repository_root=config.repository_root,
        entries=entries,
    )
    normalized_source = source_module.validate_experiment013_source_manifest(
        source_bootstrap["document"]
    )
    if (
        source_module.verify_experiment013_source_manifest(
            normalized_source,
            config.repository_root,
            git_executable=config.git_executable_path,
        )
        != normalized_source
    ):
        raise StageAError("source verifier returned a different H0 manifest")
    _assert_tracked_identity_bytes(config, identity_bytes)
    binding = resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)
    capture_provenance_receipt = resolver.deserialize_stage_a_capture_provenance_receipt(
        capture_provenance_receipt_bytes,
        expected_file_sha256=config.expected_stage_a_capture_provenance_receipt_sha256,
        calibration_binding_artifact=binding_bytes,
        expected_identity_input_file_sha256=bootstrap.identity_input_file_sha256,
    )
    if dict(binding.execution_bindings) != dict(bootstrap.execution_bindings):
        raise StageAError(
            "calibration authorization and Stage-A identity bind different execution artifacts"
        )
    if binding.source_commit != source_bootstrap["source_commit"]:
        raise StageAError("calibration authorization and Stage-A source manifest bind different H0")
    dependencies = _decode_binding_dependencies(binding)
    if any(
        sha256_bytes(dependencies[name])
        != bootstrap.calibration_binding[_binding_field_for_dependency(name)]
        for name in BINDING_DEPENDENCY_NAMES
    ):
        raise StageAError("identity v6 binding differs from authorized calibration dependencies")
    if (
        binding.authorization_file_sha256
        != bootstrap.calibration_binding["calibration_authorization_file_sha256"]
    ):
        raise StageAError("identity v6 differs from the calibration authorization receipt")
    identity = resolver.deserialize_frozen_stage_a_identity_artifact(
        identity_bytes,
        calibration_binding_artifact=binding_bytes,
        stage_a_capture_provenance_receipt=capture_provenance_receipt_bytes,
        expected_stage_a_capture_provenance_receipt_sha256=(
            config.expected_stage_a_capture_provenance_receipt_sha256
        ),
        expected_file_sha256=bootstrap.file_sha256,
    )
    calibration = importlib.import_module("recurquant.static_q468_calibration")
    split = calibration.deserialize_frozen_split_half_stability_artifact(
        dependencies["split_half_stability_artifact"]
    )
    if split.stability.passed is not True:
        raise StageAStabilityFailure("frozen split-half stability gate did not pass")
    runtime_manifest = calibration_runner.parse_calibration_runtime_manifest(
        execution["calibration_runtime_manifest_file_sha256"]
    )
    if source_bootstrap["git_executable"] != {
        "sha256": runtime_manifest.git_executable_sha256,
        "size_bytes": runtime_manifest.git_executable_size_bytes,
    }:
        raise StageAError("source and runtime manifests bind different Git executable bytes")
    authenticated_runtime = calibration_runner.authenticate_calibration_runtime(
        runtime_manifest,
        base_runtime_root=config.base_runtime_root,
        package_roots=config.package_roots,
        interpreter_path=config.interpreter_path,
        git_executable_path=config.git_executable_path,
    )
    model_manifest = calibration_runner.parse_model_file_manifest(
        execution["model_file_manifest_file_sha256"]
    )
    authenticated_model = calibration_runner.authenticate_local_model_files(
        config.model_root,
        model_manifest,
    )
    capture = _load_exact_module(
        "_recurquant_experiment013_stage_a_capture",
        CAPTURE_SOURCE_PATH,
        repository_root=config.repository_root,
        entries=entries,
    )
    input_bundle = None
    input_bundle_manifest_file_sha256 = None
    if require_input_bundle:
        input_bundle = capture.authenticate_stage_a_input_bundle(
            config.input_bundle_root,
            frozen_stage_a_identity_artifact=identity_bytes,
            calibration_binding_artifact=binding_bytes,
            stage_a_capture_provenance_receipt=capture_provenance_receipt_bytes,
            expected_stage_a_capture_provenance_receipt_sha256=(
                config.expected_stage_a_capture_provenance_receipt_sha256
            ),
            execution_binding_artifacts=execution,
        )
        input_bundle_manifest_file_sha256 = _require_sha256(
            getattr(input_bundle, "manifest_file_sha256", None),
            context="authenticated Stage-A input bundle manifest SHA-256",
        )
    methods = reconstruct_stage_a_methods(
        dependency_bytes=dependencies,
        frozen_stage_a_identity=identity,
        source_commit=cast(str, source_bootstrap["source_commit"]),
    )
    return AuthenticatedStageA(
        bootstrap_identity=bootstrap,
        identity=identity,
        binding=binding,
        capture_provenance_receipt=capture_provenance_receipt,
        capture_provenance_receipt_bytes=capture_provenance_receipt_bytes,
        dependency_bytes=MappingProxyType(dependencies),
        execution_artifact_bytes=MappingProxyType(execution),
        source_manifest=MappingProxyType(normalized_source),
        source_manifest_file_sha256=sha256_bytes(
            execution["repository_source_manifest_file_sha256"]
        ),
        source_commit=cast(str, source_bootstrap["source_commit"]),
        input_bundle=input_bundle,
        input_bundle_manifest_file_sha256=input_bundle_manifest_file_sha256,
        model_manifest=model_manifest,
        authenticated_model_files=authenticated_model,
        runtime_manifest=runtime_manifest,
        authenticated_runtime=authenticated_runtime,
        resolver=resolver,
        capture=capture,
        calibration_runner=calibration_runner,
        source_module=source_module,
        methods=methods,
    )


def reauthenticate_production(
    config: StageAConfig,
    previous: AuthenticatedStageA,
    reservation: AttemptReservation | None,
) -> None:
    execution = _read_execution_artifacts(config)
    if execution != dict(previous.execution_artifact_bytes):
        raise StageAError("execution artifact bytes changed during Stage A")
    identity_bytes = _stable_file_bytes(config.frozen_identity_path, context="frozen identity")
    if sha256_bytes(identity_bytes) != previous.bootstrap_identity.file_sha256:
        raise StageAError("frozen Stage-A identity changed during evaluation")
    binding_bytes = _stable_file_bytes(
        config.calibration_binding_path,
        context="Stage-A calibration binding",
    )
    receipt_bytes = _stable_file_bytes(
        config.stage_a_capture_provenance_receipt_path,
        context="finalized Stage-A capture provenance receipt",
    )
    if receipt_bytes != previous.capture_provenance_receipt_bytes:
        raise StageAError("finalized Stage-A capture provenance changed during evaluation")
    bootstrap_stage_a_capture_provenance_receipt(
        receipt_bytes,
        expected_file_sha256=config.expected_stage_a_capture_provenance_receipt_sha256,
        calibration_binding_artifact=binding_bytes,
        identity=previous.bootstrap_identity,
        expected_source_commit=config.source_commit,
    )
    previous.resolver.deserialize_stage_a_calibration_binding_artifact(
        binding_bytes,
        expected_file_sha256=previous.binding.file_sha256,
    )
    repeated_receipt = previous.resolver.deserialize_stage_a_capture_provenance_receipt(
        receipt_bytes,
        expected_file_sha256=config.expected_stage_a_capture_provenance_receipt_sha256,
        calibration_binding_artifact=binding_bytes,
        expected_identity_input_file_sha256=(
            previous.bootstrap_identity.identity_input_file_sha256
        ),
    )
    if repeated_receipt != previous.capture_provenance_receipt:
        raise StageAError("finalized Stage-A capture provenance reauthentication drifted")
    previous.resolver.deserialize_frozen_stage_a_identity_artifact(
        identity_bytes,
        calibration_binding_artifact=binding_bytes,
        stage_a_capture_provenance_receipt=receipt_bytes,
        expected_stage_a_capture_provenance_receipt_sha256=(
            config.expected_stage_a_capture_provenance_receipt_sha256
        ),
        expected_file_sha256=previous.bootstrap_identity.file_sha256,
    )
    if reservation is None:
        _assert_tracked_identity_bytes(config, identity_bytes)
    else:
        _assert_tracked_identity_bytes_after_seal(
            config,
            identity_bytes,
            previous,
            reservation,
        )
    if previous.source_module.verify_experiment013_source_manifest(
        previous.source_manifest,
        config.repository_root,
        git_executable=config.git_executable_path,
    ) != dict(previous.source_manifest):
        raise StageAError("H0 source identity changed during Stage A")
    runtime = previous.calibration_runner.authenticate_calibration_runtime(
        previous.runtime_manifest,
        base_runtime_root=config.base_runtime_root,
        package_roots=config.package_roots,
        interpreter_path=config.interpreter_path,
        git_executable_path=config.git_executable_path,
    )
    if runtime != previous.authenticated_runtime:
        raise StageAError("sealed runtime changed during Stage A")
    model = previous.calibration_runner.authenticate_local_model_files(
        config.model_root, previous.model_manifest
    )
    if model != previous.authenticated_model_files:
        raise StageAError("authenticated model files changed during Stage A")
    if previous.input_bundle is None or previous.input_bundle_manifest_file_sha256 is None:
        raise StageAError("authenticated Stage-A input bundle is unavailable")
    bundle = previous.capture.authenticate_stage_a_input_bundle(
        config.input_bundle_root,
        frozen_stage_a_identity_artifact=identity_bytes,
        calibration_binding_artifact=binding_bytes,
        stage_a_capture_provenance_receipt=receipt_bytes,
        expected_stage_a_capture_provenance_receipt_sha256=(
            config.expected_stage_a_capture_provenance_receipt_sha256
        ),
        execution_binding_artifacts=execution,
    )
    if getattr(bundle, "manifest_file_sha256", None) != (
        previous.input_bundle_manifest_file_sha256
    ):
        raise StageAError("Stage-A input bundle manifest changed during Stage A")


def _assert_tracked_identity_bytes_after_seal(
    config: StageAConfig,
    identity_bytes: bytes,
    authenticated: AuthenticatedStageA,
    reservation: AttemptReservation,
) -> None:
    head = _require_sha1(
        _git(config.git_executable_path, config.repository_root, "rev-parse", "HEAD"),
        context="HEAD",
    )
    if head != reservation.seal_commit:
        raise StageAError("HEAD differs from the reserved one-run seal commit")
    parents = _git(
        config.git_executable_path,
        config.repository_root,
        "show",
        "-s",
        "--format=%P",
        head,
    ).split()
    if (
        len(parents) != 1
        or parents[0] != config.identity_commit
        or parents[0] != reservation.h1_commit
    ):
        raise StageAError("HEAD is not the one-run seal child of the identity commit")
    identity = config.frozen_identity_path.resolve(strict=True)
    relative = identity.relative_to(config.repository_root.resolve(strict=True)).as_posix()
    commit = _git_process(
        config.git_executable_path,
        config.repository_root,
        "show",
        f"{head}:{relative}",
    )
    index = _git_process(
        config.git_executable_path,
        config.repository_root,
        "show",
        f":{relative}",
    )
    if commit.returncode != 0 or index.returncode != 0:
        raise StageAError("sealed identity disappeared from Git")
    if commit.stdout != identity_bytes or index.stdout != identity_bytes:
        raise StageAError("sealed identity differs across HEAD, index, and worktree")
    if _git(
        config.git_executable_path,
        config.repository_root,
        "show",
        "-s",
        "--format=%T",
        head,
    ) != _git(
        config.git_executable_path,
        config.repository_root,
        "show",
        "-s",
        "--format=%T",
        config.identity_commit,
    ):
        raise StageAError("one-run seal is not an empty-diff commit")
    expected_message = _seal_message(authenticated).encode("utf-8")
    commit = _git_process(
        config.git_executable_path,
        config.repository_root,
        "cat-file",
        "commit",
        head,
    )
    if commit.returncode != 0:
        raise StageAError("cannot read the reserved one-run seal commit")
    _headers, separator, message = commit.stdout.partition(b"\n\n")
    if (
        not separator
        or message != expected_message
        or sha256_bytes(expected_message) != reservation.receipt.get("one_run_seal_message_sha256")
    ):
        raise StageAError("reserved one-run seal message drifted")
    if authenticated.source_commit != config.source_commit:
        raise StageAError("authenticated H0 source commit changed")


def materialize_production(config: StageAConfig, authenticated: AuthenticatedStageA) -> object:
    if authenticated.input_bundle is None:
        raise StageAError("authenticated Stage-A input bundle is unavailable")
    source = authenticated.capture.StagedCaptureSource(authenticated.input_bundle)
    materialization = authenticated.capture.materialize_stage_a_identity_sequences(
        source=source,
        frozen_stage_a_identity_artifact=_stable_file_bytes(
            config.frozen_identity_path, context="frozen Stage-A identity"
        ),
        calibration_binding_artifact=_stable_file_bytes(
            config.calibration_binding_path, context="Stage-A calibration binding"
        ),
        stage_a_capture_provenance_receipt=_stable_file_bytes(
            config.stage_a_capture_provenance_receipt_path,
            context="finalized Stage-A capture provenance receipt",
        ),
        expected_stage_a_capture_provenance_receipt_sha256=(
            config.expected_stage_a_capture_provenance_receipt_sha256
        ),
        expected_frozen_stage_a_identity_file_sha256=(authenticated.bootstrap_identity.file_sha256),
        execution_binding_artifacts=dict(authenticated.execution_artifact_bytes),
        runtime_authentication_context=_runtime_context(config),
    )
    return materialization


def prepare_inputs_production(config: StageAConfig) -> Mapping[str, object]:
    """Stage opaque public bytes before the one-run seal in this network-only child."""

    authenticated = authenticate_production(config, require_input_bundle=False)
    identity_bytes = _stable_file_bytes(config.frozen_identity_path, context="frozen identity")
    binding_bytes = _stable_file_bytes(
        config.calibration_binding_path,
        context="Stage-A calibration binding",
    )
    bundle = authenticated.capture.stage_stage_a_input_bundle(
        bundle_root=config.input_bundle_root,
        cache_dir=config.cache_root,
        ruler_receipt_dir=config.ruler_root,
        frozen_stage_a_identity_artifact=identity_bytes,
        calibration_binding_artifact=binding_bytes,
        stage_a_capture_provenance_receipt=(authenticated.capture_provenance_receipt_bytes),
        expected_stage_a_capture_provenance_receipt_sha256=(
            config.expected_stage_a_capture_provenance_receipt_sha256
        ),
        execution_binding_artifacts=dict(authenticated.execution_artifact_bytes),
        runtime_authentication_context=_runtime_context(config),
    )
    return MappingProxyType(
        {
            "status": "stage_a_input_bundle_prepared",
            "stage_a_input_bundle_manifest_file_sha256": _require_sha256(
                getattr(bundle, "manifest_file_sha256", None),
                context="prepared Stage-A input bundle manifest SHA-256",
            ),
            "identity_file_sha256": authenticated.bootstrap_identity.file_sha256,
            "calibration_binding_file_sha256": authenticated.binding.file_sha256,
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
                authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
            ),
            "source_commit": authenticated.source_commit,
            "content_materialized": False,
            "model_instantiated": False,
            "one_run_reserved": False,
        }
    )


class TorchStageAEngine:
    """Reviewed Qwen3.5 execution path; all imports and weights are late."""

    def __init__(self) -> None:
        self._torch: Any = None
        self._model: object | None = None
        self._reference_states: dict[str, list[dict[int, object]]] = {}

    @staticmethod
    def _assert_loaded_model_contract(
        model: object,
        *,
        torch: object,
        transformers: object,
        device: object,
    ) -> None:
        if type(model) is not getattr(transformers, "Qwen3_5ForCausalLM", None):
            raise StageAError("loaded model is not the pinned Qwen3.5 causal-LM class")
        model_config = getattr(model, "config", None)
        if type(model_config) is not getattr(transformers, "Qwen3_5TextConfig", None):
            raise StageAError("loaded model config is not the pinned Qwen3.5 text config class")
        if (
            getattr(model_config, "_attn_implementation", None) != "eager"
            or getattr(model_config, "_attn_implementation_internal", None) != "eager"
        ):
            raise StageAError("loaded Stage-A model did not retain eager attention")
        parameters = tuple(model.parameters())
        if not parameters:
            raise StageAError("loaded Stage-A model has no parameters")
        floating_parameter_count = 0
        for index, parameter in enumerate(parameters):
            if getattr(parameter, "device", None) != device:
                raise StageAError(f"loaded Stage-A parameter {index} is not on the one CUDA device")
            is_floating_point = getattr(parameter, "is_floating_point", None)
            if not callable(is_floating_point):
                raise StageAError(f"loaded Stage-A parameter {index} has no dtype contract")
            if is_floating_point():
                floating_parameter_count += 1
                if getattr(parameter, "dtype", None) != getattr(torch, "bfloat16", None):
                    raise StageAError(f"loaded Stage-A floating parameter {index} is not BF16")
        if floating_parameter_count == 0:
            raise StageAError("loaded Stage-A model has no floating parameters")

    def load_model(self, authenticated_model_files: object) -> object:
        if self._model is not None:
            raise StageAError("Stage-A model may be loaded exactly once")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        if not torch.cuda.is_available():
            raise StageAError("official Stage A requires CUDA")
        root = getattr(authenticated_model_files, "model_root", None)
        if not isinstance(root, Path):
            raise StageAError("authenticated model root is unavailable")
        config = transformers.Qwen3_5TextConfig.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        loaded = transformers.Qwen3_5ForCausalLM.from_pretrained(
            str(root),
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            weights_only=True,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if type(loaded) is not tuple or len(loaded) != 2:
            raise StageAError("Transformers did not return model loading diagnostics")
        model, diagnostics = loaded
        if not isinstance(diagnostics, dict) or any(diagnostics.values()):
            raise StageAError("authenticated model load reported missing or unexpected weights")
        device = torch.device("cuda", torch.cuda.current_device())
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self._assert_loaded_model_contract(
            model,
            torch=torch,
            transformers=transformers,
            device=device,
        )
        self._torch = torch
        self._model = model
        return model

    def close_model(self, model: object) -> None:
        if model is not self._model:
            raise StageAError("attempted to close a different Stage-A model")
        self._model = None
        self._reference_states.clear()
        with contextlib.suppress(Exception):
            self._torch.cuda.empty_cache()

    def runtime_snapshot(self, model: object) -> Mapping[str, object]:
        if model is not self._model:
            raise StageAError("cannot snapshot a different Stage-A model")
        torch = self._torch
        parameter = next(model.parameters())
        device = parameter.device
        if device.type != "cuda" or device.index is None:
            raise StageAError("official Stage A model is not on one explicit CUDA device")
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        cuda_runtime = torch.version.cuda
        if not isinstance(cuda_runtime, str) or not cuda_runtime:
            raise StageAError("Stage-A Torch runtime has no CUDA version")
        return _validated_device_runtime(
            {
                "attention_implementation": str(model.config._attn_implementation),
                "capability": list(torch.cuda.get_device_capability(device)),
                "cuda_runtime": cuda_runtime,
                "device_index": int(device.index),
                "model_class": type(model).__name__,
                "model_config_class": type(model.config).__name__,
                "model_parameter_dtype": str(parameter.dtype),
                "name": str(torch.cuda.get_device_name(device)),
                "torch_version": str(torch.__version__),
                "total_memory_bytes": int(properties.total_memory),
            }
        )

    def _cache(self, model: object, method: StageAMethodSpec) -> tuple[object, object | None]:
        transformers = importlib.import_module("transformers")
        if method.method_id == FP32_METHOD:
            return transformers.DynamicCache(config=model.config), None
        cache_module = importlib.import_module("recurquant.static_q468_cache")
        if method.method_id == DYNAMIC_K27030_METHOD:
            cache = cache_module.create_qwen35_dynamic_q468_baseline_cache(
                model, record_evidence=True
            )
        else:
            if method.policy is None or method.policy_file_sha256 is None:
                raise StageAError(f"static method {method.method_id} has no authenticated policy")
            cache = cache_module.create_qwen35_static_rht_cache(
                model,
                policy=method.policy,
                expected_policy_sha256=method.policy.policy_sha256,
                record_evidence=True,
            )
        observer_module = importlib.import_module("recurquant.statelease_equal_byte_cache")
        observer = observer_module.Qwen35EqualByteObserver(model, caches=[cache])
        return cache, observer

    def begin_method(self, model: object, method: StageAMethodSpec, sequence: object) -> object:
        if model is not self._model:
            raise StageAError("Stage-A engine received a different model")
        identity = getattr(sequence, "identity_record_sha256", None)
        target_ids = getattr(sequence, "target_token_ids", None)
        if not isinstance(identity, str) or _SHA256_RE.fullmatch(identity) is None:
            raise StageAError("Stage-A sequence has no authenticated identity hash")
        if not isinstance(target_ids, tuple) or len(target_ids) < 2:
            raise StageAError("Stage-A sequence has no valid continuation")
        expected_steps = len(target_ids) - 1
        if method.method_id == FP32_METHOD:
            if identity in self._reference_states:
                raise StageAError("FP32 trajectory identity was reused")
            self._reference_states[identity] = []
        else:
            reference = self._reference_states.get(identity)
            if reference is None or len(reference) != expected_steps:
                raise StageAError("candidate method has no complete matched FP32 trajectory")
        device = next(model.parameters()).device
        self._torch.cuda.synchronize(device)
        self._torch.cuda.empty_cache()
        cache, observer = self._cache(model, method)
        if observer is not None:
            enter = getattr(observer, "__enter__", None)
            if not callable(enter) or enter() is not observer:
                raise StageAError("Stage-A equal-byte observer did not install exactly once")
        return {
            "model": model,
            "method": method,
            "cache": cache,
            "observer": observer,
            "identity_record_sha256": identity,
            "expected_steps": expected_steps,
            "step_index": 0,
        }

    def _recurrent_states(self, cache: object, *, packed: bool) -> dict[int, object]:
        torch = self._torch
        static = importlib.import_module("recurquant.static_q468")
        expected_layers = tuple(static.FROZEN_RECURRENT_LAYER_INDICES)
        geometry = static.FROZEN_QWEN35_STATIC_Q468_GEOMETRY
        expected_shape = (1, geometry.heads, geometry.key_rows, geometry.value_width)
        states: dict[int, object] = {}
        if packed:
            if self._model is None:
                raise StageAError("packed recurrent-state capture has no loaded model")
            expected_device = next(self._model.parameters()).device
            checkpoint = getattr(cache, "checkpoint", None)
            materialize_all = getattr(checkpoint, "materialize", None)
            if callable(materialize_all):
                raw_states = materialize_all()
                if not isinstance(raw_states, Mapping):
                    raise StageAError("packed cache materialization returned an invalid layer map")
                source = raw_states.items()
            else:
                materialize_one = getattr(cache, "materialize_recurrent_state", None)
                if not callable(materialize_one):
                    raise StageAError("packed cache cannot materialize its recurrent state")
                source = (
                    (layer_index, materialize_one(layer_index)) for layer_index in expected_layers
                )
            for raw_layer_index, tensor in source:
                layer_index = int(raw_layer_index)
                if layer_index not in expected_layers or layer_index in states:
                    raise StageAError("packed cache recurrent layer inventory drifted")
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    raise StageAError("packed cache materialized a non-floating recurrent state")
                if tensor.dtype is not torch.float32:
                    raise StageAError("packed cache materialized a non-FP32 recurrent state")
                if tensor.device != expected_device:
                    raise StageAError(
                        "packed cache recurrent state is not on the model CUDA device"
                    )
                if tuple(tensor.shape) != expected_shape:
                    raise StageAError("packed cache recurrent-state geometry drifted")
                states[layer_index] = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        else:
            cache_module = importlib.import_module("recurquant.cache")
            if self._model is None:
                raise StageAError("FP32 recurrent-state capture has no loaded model")
            expected_device = next(self._model.parameters()).device
            fp32_resident_bytes = 0
            for state in cache_module.iter_recurrent_states(cache):
                if state.layer_index not in expected_layers or state.state_index != 0:
                    raise StageAError("FP32 cache recurrent layer/state inventory drifted")
                if state.layer_index in states:
                    raise StageAError("FP32 cache exposed a duplicate recurrent layer")
                tensor = state.tensor
                if tensor.dtype is not torch.float32:
                    raise StageAError("FP32 reference cache exposed a non-FP32 recurrent state")
                if tensor.device != expected_device:
                    raise StageAError("FP32 reference cache state is not on the model CUDA device")
                if tuple(tensor.shape) != expected_shape:
                    raise StageAError("FP32 reference cache recurrent-state geometry drifted")
                fp32_resident_bytes += int(tensor.numel()) * int(tensor.element_size())
                states[state.layer_index] = (
                    tensor.detach().to(device="cpu", dtype=torch.float32).clone()
                )
            if fp32_resident_bytes != EXPECTED_RECURRENT_RESIDENT_BYTES[FP32_METHOD]:
                raise StageAError("FP32 reference cache byte ledger drifted")
        if tuple(sorted(states)) != tuple(sorted(expected_layers)):
            raise StageAError("recurrent trajectory omitted or added a frozen layer")
        return states

    @staticmethod
    def _cache_length(cache: object) -> int:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            raise StageAError("Stage-A cache does not expose get_seq_length")
        length = get_seq_length()
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise StageAError("Stage-A cache returned an invalid sequence length")
        return length

    def _trajectory_nmse(self, session: dict[str, object]) -> float:
        torch = self._torch
        method = cast(StageAMethodSpec, session["method"])
        identity = cast(str, session["identity_record_sha256"])
        step_index = cast(int, session["step_index"])
        expected_steps = cast(int, session["expected_steps"])
        if step_index >= expected_steps:
            raise StageAError("Stage-A trajectory received an extra scored transition")
        states = self._recurrent_states(
            session["cache"],
            packed=method.method_id != FP32_METHOD,
        )
        if method.method_id == FP32_METHOD:
            self._reference_states[identity].append(states)
            return 0.0
        reference_trace = self._reference_states.get(identity)
        if reference_trace is None or len(reference_trace) != expected_steps:
            raise StageAError("matched FP32 trajectory changed before candidate comparison")
        reference = reference_trace[step_index]
        if set(reference) != set(states):
            raise StageAError("candidate trajectory layer inventory differs from FP32")
        layer_values: list[float] = []
        for layer_index in sorted(reference):
            reference_tensor = reference[layer_index]
            candidate_tensor = states[layer_index]
            if not isinstance(reference_tensor, torch.Tensor) or not isinstance(
                candidate_tensor, torch.Tensor
            ):
                raise StageAError("trajectory state is not a tensor")
            if reference_tensor.shape != candidate_tensor.shape:
                raise StageAError("candidate trajectory shape differs from matched FP32")
            reference_fp64 = reference_tensor.to(dtype=torch.float64)
            candidate_fp64 = candidate_tensor.to(dtype=torch.float64)
            if (
                not torch.isfinite(reference_fp64).all().item()
                or not torch.isfinite(candidate_fp64).all().item()
            ):
                raise StageAError("trajectory state contains non-finite values")
            numerator = (candidate_fp64 - reference_fp64).square().sum(dtype=torch.float64)
            denominator = reference_fp64.square().sum(dtype=torch.float64) + 1.0e-12
            value = float((numerator / denominator).item())
            if not math.isfinite(value) or value < 0.0:
                raise StageAError("trajectory NMSE is invalid")
            layer_values.append(value)
        if not layer_values:
            raise StageAError("trajectory NMSE has no recurrent layers")
        return math.fsum(layer_values) / len(layer_values)

    def _forward(
        self,
        session: dict[str, object],
        token_ids: tuple[int, ...],
        *,
        target_token_id: int,
        position: int,
        scored: bool,
    ) -> ForwardObservation:
        torch = self._torch
        model = session["model"]
        cache = session["cache"]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or isinstance(target_token_id, bool)
            or not isinstance(target_token_id, int)
            or target_token_id < 0
            or not token_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in token_ids
            )
        ):
            raise StageAError("Stage-A forward position or input token IDs are invalid")
        expected_before = position + 1 - len(token_ids)
        if expected_before < 0:
            raise StageAError("Stage-A forward token span starts before position zero")
        if self._cache_length(cache) != expected_before:
            raise StageAError("Stage-A cache length drifted before the causal forward")
        device = next(model.parameters()).device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        cache_position = torch.arange(
            expected_before,
            position + 1,
            dtype=torch.long,
            device=device,
        )
        position_ids = cache_position.unsqueeze(0)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        if getattr(output, "past_key_values", None) is not cache:
            raise StageAError("Qwen3.5 returned a different Stage-A cache object")
        if self._cache_length(cache) != position + 1:
            raise StageAError("Stage-A cache did not advance by the exact causal token count")
        torch.cuda.synchronize(device)
        latency = time.perf_counter_ns() - started
        logits = output.logits[0, -1].detach().to(device="cpu", dtype=torch.float32).contiguous()
        log_probabilities_tensor = torch.log_softmax(logits, dim=-1)
        if target_token_id >= log_probabilities_tensor.numel():
            raise StageAError("target token is outside the authenticated model vocabulary")
        target_nll = -float(log_probabilities_tensor[target_token_id].item())
        top1 = int(torch.argmax(log_probabilities_tensor).item())
        evidence = getattr(cache, "last_evidence", None)
        static = importlib.import_module("recurquant.static_q468")
        local_sse = (
            0.0
            if evidence is None
            else float(getattr(evidence, "mean_squared_error", 0.0))
            * int(static.FROZEN_QWEN35_STATIC_Q468_GEOMETRY.state_elements)
        )
        trajectory_nmse = self._trajectory_nmse(session) if scored else 0.0
        if scored:
            session["step_index"] = cast(int, session["step_index"]) + 1
        storage_method = getattr(cache, "storage_summary", None)
        storage = {} if not callable(storage_method) else dict(storage_method())
        method = cast(StageAMethodSpec, session["method"])
        if method.method_id == FP32_METHOD:
            resident = EXPECTED_RECURRENT_RESIDENT_BYTES[FP32_METHOD]
        else:
            resident = int(storage.get("resident_bytes", 0))
        transient = int(storage.get("raw_state_workspace_peak_bytes", 0)) + int(
            storage.get("query_workspace_peak_bytes", 0)
        )
        return ForwardObservation(
            position=position,
            target_token_id=target_token_id,
            comparison_logits=logits,
            target_nll=target_nll,
            top1_token_id=top1,
            local_codec_sse=local_sse,
            trajectory_nmse=trajectory_nmse,
            latency_ns=latency,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            resident_bytes=resident,
            transient_bytes=transient,
        )

    def prefill(
        self,
        session: object,
        *,
        prompt_token_ids: tuple[int, ...],
        first_target_token_id: int,
        position: int,
    ) -> ForwardObservation:
        return self._forward(
            cast(dict[str, object], session),
            prompt_token_ids,
            target_token_id=first_target_token_id,
            position=position,
            scored=False,
        )

    def step(
        self,
        session: object,
        *,
        input_token_id: int,
        target_token_id: int,
        position: int,
    ) -> ForwardObservation:
        return self._forward(
            cast(dict[str, object], session),
            (input_token_id,),
            target_token_id=target_token_id,
            position=position,
            scored=True,
        )

    def end_method(self, session: object) -> Mapping[str, object]:
        values = cast(dict[str, object], session)
        observer = values.get("observer")
        try:
            step_index = cast(int, values["step_index"])
            expected_steps = cast(int, values["expected_steps"])
            if step_index != expected_steps:
                raise StageAError("Stage-A method did not complete its exact scored trajectory")
            method = cast(StageAMethodSpec, values["method"])
            identity = cast(str, values["identity_record_sha256"])
            if method.method_id == PRIMARY_K29334_METHOD:
                reference = self._reference_states.pop(identity, None)
                if reference is None or len(reference) != expected_steps:
                    raise StageAError("completed method grid has no exact FP32 trajectory")
            cache = values["cache"]
            summary = getattr(cache, "storage_summary", None)
            storage = {} if not callable(summary) else dict(summary())
            if method.method_id == FP32_METHOD:
                storage.update(
                    {
                        "resident_bytes": EXPECTED_RECURRENT_RESIDENT_BYTES[FP32_METHOD],
                        "resident_byte_scope": "logical_fp32_recurrent_state_only",
                    }
                )
            return storage
        finally:
            try:
                if observer is not None:
                    remove = getattr(observer, "remove", None)
                    if not callable(remove):
                        raise StageAError("Stage-A equal-byte observer cannot be removed")
                    remove()
            finally:
                # The caller keeps the session variable alive until the next method
                # assignment.  Clear ownership here so the completed cache/model can
                # be released before the next method resets allocator diagnostics.
                values.clear()


def run_preseal_engine_smoke(authenticated: AuthenticatedStageA) -> Mapping[str, object]:
    """Exercise all nine real cache paths on fixed public tokens before the one-run seal."""

    if tuple(method.method_id for method in authenticated.methods) != METHOD_ORDER:
        raise StageAError("pre-seal smoke method order differs from the frozen grid")
    engine = TorchStageAEngine()
    sequence = _PresealSmokeSequence(
        identity_record_sha256=sha256_bytes(
            b"recurquant.experiment013.stage-a-preseal-engine-smoke.v1"
        ),
        target_token_ids=PRESEAL_ENGINE_SMOKE_TARGET,
    )
    prompt_token_ids = (PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_ID,) * (
        PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT
    )
    model = engine.load_model(authenticated.authenticated_model_files)
    reference_logits: list[object] = []
    method_receipts: list[dict[str, object]] = []
    total_forwards = 0
    try:
        for method in authenticated.methods:
            session = engine.begin_method(model, method, sequence)
            try:
                prefill = _finite_observation(
                    engine.prefill(
                        session,
                        prompt_token_ids=prompt_token_ids,
                        first_target_token_id=PRESEAL_ENGINE_SMOKE_TARGET[0],
                        position=PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT - 1,
                    ),
                    expected_position=PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT - 1,
                    expected_target=PRESEAL_ENGINE_SMOKE_TARGET[0],
                )
                del prefill
                total_forwards += 1
                current_logits: list[object] = []
                for transition_index in range(len(PRESEAL_ENGINE_SMOKE_TARGET) - 1):
                    position = PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT + transition_index
                    observation = _finite_observation(
                        engine.step(
                            session,
                            input_token_id=PRESEAL_ENGINE_SMOKE_TARGET[transition_index],
                            target_token_id=PRESEAL_ENGINE_SMOKE_TARGET[transition_index + 1],
                            position=position,
                        ),
                        expected_position=position,
                        expected_target=PRESEAL_ENGINE_SMOKE_TARGET[transition_index + 1],
                    )
                    total_forwards += 1
                    current_logits.append(observation.comparison_logits)
                    if method.method_id == FP32_METHOD:
                        reference_logits.append(observation.comparison_logits)
                    else:
                        if transition_index >= len(reference_logits):
                            raise StageAError("pre-seal cache path ran before the FP32 reference")
                        _kl(reference_logits[transition_index], observation.comparison_logits)
            except BaseException as error:
                try:
                    engine.end_method(session)
                except BaseException as cleanup_error:
                    error.add_note(
                        f"pre-seal cache observer cleanup also failed: {cleanup_error!r}"
                    )
                raise
            else:
                summary = dict(engine.end_method(session))
            resident = summary.get("resident_bytes")
            expected_resident = EXPECTED_RECURRENT_RESIDENT_BYTES[method.method_id]
            if (
                isinstance(resident, bool)
                or not isinstance(resident, int)
                or (resident != expected_resident)
            ):
                raise StageAError(
                    f"pre-seal {method.method_id} resident bytes differ from the frozen ledger"
                )
            method_receipts.append(
                {
                    "method_id": method.method_id,
                    "forward_count": len(PRESEAL_ENGINE_SMOKE_TARGET),
                    "logical_recurrent_resident_bytes": resident,
                    "equal_byte_observer_required": method.method_id != FP32_METHOD,
                }
            )
            current_logits.clear()
        device = dict(engine.runtime_snapshot(model))
    finally:
        reference_logits.clear()
        engine.close_model(model)
        del model
        engine._torch.cuda.synchronize()
        engine._torch.cuda.empty_cache()
    expected_forwards = len(METHOD_ORDER) * len(PRESEAL_ENGINE_SMOKE_TARGET)
    if total_forwards != expected_forwards:
        raise StageAError("pre-seal engine smoke forward count drifted")
    report = {
        "profile": PRESEAL_ENGINE_SMOKE_PROFILE,
        "passed": True,
        "stage_a_content_accessed": False,
        "input_profile": "fixed_public_synthetic_4096_plus_128_v3",
        "prompt_token_count": PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT,
        "continuation_token_count": PRESEAL_ENGINE_SMOKE_TARGET_TOKEN_COUNT,
        "model_load_count": 1,
        "method_order": list(METHOD_ORDER),
        "forward_count": total_forwards,
        "method_receipts": method_receipts,
        "device": device,
    }
    return MappingProxyType(_strict_json(canonical_json_bytes(report), context="pre-seal smoke"))


def default_services() -> StageAServices:
    return StageAServices(
        authenticate=authenticate_production,
        reauthenticate=reauthenticate_production,
        preseal_smoke=run_preseal_engine_smoke,
        reserve=reserve_one_run,
        materialize=materialize_production,
        engine=TorchStageAEngine(),
        persist_receipt=persist_receipt,
        publish=publish_result,
        record_failure=record_failure,
    )


def _authenticate_recovery_boundary(
    config: StageAConfig,
    receipt: Mapping[str, object],
    *,
    allow_pre_cas_head: bool = False,
) -> tuple[BootstrapIdentity, str, str]:
    identity_bytes = _stable_file_bytes(
        config.frozen_identity_path,
        context="recovery frozen Stage-A identity",
    )
    bootstrap = bootstrap_stage_a_identity(identity_bytes)
    if bootstrap.file_sha256 != _require_sha256(
        receipt.get("identity_file_sha256"),
        context="recovery receipt identity file SHA-256",
    ):
        raise StageAError("recovery identity differs from the durable receipt")
    if (
        receipt.get("runner_revision") != RUNNER_REVISION
        or receipt.get("attempt_number") != 1
        or receipt.get("one_run_marker") != ONE_RUN_MARKER
        or receipt.get("automatic_retry_authorized") is not False
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or receipt.get("execution_bindings") != dict(bootstrap.execution_bindings)
        or receipt.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD)
        != bootstrap.stage_a_capture_provenance_receipt_file_sha256
        or receipt.get("expected_forward_count") != bootstrap.expected_forward_count
    ):
        raise StageAError("recovery receipt contract differs from the frozen identity")
    h1 = _require_sha1(config.identity_commit, context="recovery identity commit")
    if h1 != _require_sha1(
        receipt.get("h1_identity_commit"),
        context="recovery receipt H1 commit",
    ):
        raise StageAError("recovery H1 differs from the durable receipt")
    seal = _require_sha1(
        receipt.get("one_run_seal_commit"),
        context="recovery receipt seal commit",
    )
    head = _require_sha1(
        _git(config.git_executable_path, config.repository_root, "rev-parse", "HEAD"),
        context="HEAD",
    )
    if head != seal and not (allow_pre_cas_head and head == h1):
        raise StageAError("recovery requires HEAD to remain at the authenticated one-run seal")
    parents = _git(
        config.git_executable_path,
        config.repository_root,
        "show",
        "-s",
        "--format=%P",
        seal,
    ).split()
    if parents != [h1]:
        raise StageAError("recovery seal is not the exact child of H1")
    seal_tree = _require_sha1(
        _git(
            config.git_executable_path,
            config.repository_root,
            "show",
            "-s",
            "--format=%T",
            seal,
        ),
        context="recovery seal tree",
    )
    h1_tree = _require_sha1(
        _git(
            config.git_executable_path,
            config.repository_root,
            "show",
            "-s",
            "--format=%T",
            h1,
        ),
        context="recovery H1 tree",
    )
    if seal_tree != h1_tree or seal_tree != _require_sha1(
        receipt.get("one_run_seal_tree"),
        context="recovery receipt seal tree",
    ):
        raise StageAError("recovery seal is not the recorded empty-diff commit")

    binding_bytes = _stable_file_bytes(
        config.calibration_binding_path,
        context="recovery Stage-A calibration binding",
    )
    binding_hash = sha256_bytes(binding_bytes)
    if binding_hash != _require_sha256(
        receipt.get("calibration_binding_file_sha256"),
        context="recovery receipt calibration binding SHA-256",
    ):
        raise StageAError("recovery calibration binding differs from the durable receipt")
    capture_receipt_bytes = _stable_file_bytes(
        config.stage_a_capture_provenance_receipt_path,
        context="recovery finalized Stage-A capture provenance receipt",
    )
    bootstrap_capture = bootstrap_stage_a_capture_provenance_receipt(
        capture_receipt_bytes,
        expected_file_sha256=config.expected_stage_a_capture_provenance_receipt_sha256,
        calibration_binding_artifact=binding_bytes,
        identity=bootstrap,
        expected_source_commit=config.source_commit,
    )
    execution = _read_execution_artifacts(config)
    observed_execution = {name: sha256_bytes(data) for name, data in execution.items()}
    if observed_execution != dict(bootstrap.execution_bindings):
        raise StageAError("recovery execution artifacts differ from the frozen identity")
    source_hash = sha256_bytes(execution["repository_source_manifest_file_sha256"])
    if source_hash != _require_sha256(
        receipt.get("source_manifest_file_sha256"),
        context="recovery receipt source manifest SHA-256",
    ):
        raise StageAError("recovery source manifest differs from the durable receipt")
    input_bundle_hash = _require_sha256(
        receipt.get("stage_a_input_bundle_manifest_file_sha256"),
        context="recovery receipt Stage-A input bundle manifest SHA-256",
    )
    source = _bootstrap_source_manifest(execution["repository_source_manifest_file_sha256"])
    recovery_source_entries = _source_entries(source)
    recovery_capture_source = cast(Mapping[str, object], bootstrap_capture["capture_source"])
    if (
        recovery_capture_source.get("sha256")
        != recovery_source_entries[CAPTURE_SOURCE_PATH]["raw_sha256"]
    ):
        raise StageAError("recovery capture provenance source differs from authenticated H0")
    source_commit = _require_sha1(source.get("source_commit"), context="recovery source commit")
    if source_commit != _require_sha1(
        config.source_commit, context="recovery CLI H0 commit"
    ) or source_commit != _require_sha1(
        receipt.get("h0_source_commit"), context="recovery receipt H0 commit"
    ):
        raise StageAError("recovery H0 source identity drifted")
    _verify_source_bytes(source, config.repository_root)

    identity = config.frozen_identity_path.resolve(strict=True)
    root = config.repository_root.resolve(strict=True)
    try:
        relative = identity.relative_to(root).as_posix()
    except ValueError as error:
        raise StageAError("recovery identity is outside the repository") from error
    relative = _safe_relative_path(relative, context="recovery identity repository path")
    committed = _git_process(
        config.git_executable_path,
        root,
        "show",
        f"{seal}:{relative}",
    )
    indexed = _git_process(config.git_executable_path, root, "show", f":{relative}")
    if (
        committed.returncode != 0
        or indexed.returncode != 0
        or committed.stdout != identity_bytes
        or indexed.stdout != identity_bytes
    ):
        raise StageAError("recovery identity differs across seal, index, and worktree")

    expected_message = _seal_message_values(
        identity_file_sha256=bootstrap.file_sha256,
        calibration_binding_file_sha256=binding_hash,
        stage_a_capture_provenance_receipt_file_sha256=(
            bootstrap.stage_a_capture_provenance_receipt_file_sha256
        ),
        source_manifest_file_sha256=source_hash,
        input_bundle_manifest_file_sha256=input_bundle_hash,
        expected_forward_count=bootstrap.expected_forward_count,
    ).encode("utf-8")
    commit_object = _git_process(
        config.git_executable_path,
        root,
        "cat-file",
        "commit",
        seal,
    )
    if commit_object.returncode != 0:
        raise StageAError("recovery cannot read the seal commit object")
    _headers, separator, message = commit_object.stdout.partition(b"\n\n")
    if (
        not separator
        or message != expected_message
        or sha256_bytes(expected_message) != (receipt.get("one_run_seal_message_sha256"))
    ):
        raise StageAError("recovery one-run seal message drifted")

    lock_path = _identity_attempt_lock_path(
        root,
        bootstrap.file_sha256,
        git_executable_path=config.git_executable_path,
    )
    lock_bytes = _stable_file_bytes(lock_path, context="recovery identity-scoped attempt lock")
    if sha256_bytes(lock_bytes) != _require_sha256(
        receipt.get("identity_scoped_attempt_lock_file_sha256"),
        context="recovery receipt attempt-lock SHA-256",
    ):
        raise StageAError("recovery identity-scoped attempt lock drifted")
    lock = _strict_json(lock_bytes, context="recovery identity-scoped attempt lock")
    _exact_fields(
        lock,
        IDENTITY_ATTEMPT_LOCK_FIELDS,
        context="recovery identity-scoped attempt lock",
    )
    expected_lock = {
        "schema": IDENTITY_ATTEMPT_LOCK_SCHEMA,
        "runner_revision": RUNNER_REVISION,
        "created_at_utc": receipt.get("created_at_utc"),
        "attempt_number": 1,
        "automatic_retry_authorized": False,
        "h0_source_commit": source_commit,
        "h1_identity_commit": h1,
        "identity_repository_path": relative,
        "identity_file_sha256": bootstrap.file_sha256,
        "one_run_seal_commit": seal,
        "one_run_seal_tree": seal_tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": sha256_bytes(expected_message),
        "calibration_binding_file_sha256": binding_hash,
        STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
            bootstrap.stage_a_capture_provenance_receipt_file_sha256
        ),
        "source_manifest_file_sha256": source_hash,
        "stage_a_input_bundle_manifest_file_sha256": input_bundle_hash,
        "execution_bindings": dict(bootstrap.execution_bindings),
        "method_specs": receipt.get("method_specs"),
        "expected_forward_count": bootstrap.expected_forward_count,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_path": str(Path(os.path.abspath(config.output_path))),
        "attempt_path": str(Path(os.path.abspath(config.attempt_path))),
        "complete_path": str(Path(os.path.abspath(config.complete_path))),
    }
    if lock != expected_lock:
        raise StageAError("recovery identity-scoped attempt lock semantics drifted")
    return bootstrap, binding_hash, seal


def _recover_lock_only_attempt(config: StageAConfig) -> None:
    """Materialize an administrative no-retry receipt after lock-before-receipt failure."""

    identity_bytes = _stable_file_bytes(
        config.frozen_identity_path,
        context="lock-only recovery frozen Stage-A identity",
    )
    bootstrap = bootstrap_stage_a_identity(identity_bytes)
    lock_path = _identity_attempt_lock_path(
        config.repository_root,
        bootstrap.file_sha256,
        git_executable_path=config.git_executable_path,
    )
    if not lock_path.is_file():
        raise StageAError("there is no Stage-A attempt receipt or identity-scoped lock to recover")
    lock_bytes = _stable_file_bytes(
        lock_path,
        context="lock-only recovery identity-scoped attempt lock",
    )
    lock = _strict_json(lock_bytes, context="lock-only recovery attempt lock")
    if canonical_json_bytes(lock) != lock_bytes:
        raise StageAError("lock-only recovery attempt lock is not canonical JSON")
    _exact_fields(
        lock,
        IDENTITY_ATTEMPT_LOCK_FIELDS,
        context="lock-only recovery attempt lock",
    )
    if (
        lock.get("schema") != IDENTITY_ATTEMPT_LOCK_SCHEMA
        or lock.get("identity_file_sha256") != bootstrap.file_sha256
        or lock.get("output_path") != str(Path(os.path.abspath(config.output_path)))
        or lock.get("attempt_path") != str(Path(os.path.abspath(config.attempt_path)))
        or lock.get("complete_path") != str(Path(os.path.abspath(config.complete_path)))
        or lock.get("automatic_retry_authorized") is not False
    ):
        raise StageAError("lock-only recovery attempt lock identity drifted")
    if config.output_path.exists() or config.complete_path.exists():
        raise StageAError("lock-only recovery found output without an attempt receipt")
    receipt: dict[str, object] = {
        "schema": ATTEMPT_SCHEMA,
        "status": "prepared_before_head_cas",
        "runner_revision": lock.get("runner_revision"),
        "created_at_utc": lock.get("created_at_utc"),
        "attempt_number": lock.get("attempt_number"),
        "h0_source_commit": lock.get("h0_source_commit"),
        "h1_identity_commit": lock.get("h1_identity_commit"),
        "identity_repository_path": lock.get("identity_repository_path"),
        "one_run_seal_commit": lock.get("one_run_seal_commit"),
        "one_run_seal_tree": lock.get("one_run_seal_tree"),
        "one_run_marker": lock.get("one_run_marker"),
        "one_run_seal_message_sha256": lock.get("one_run_seal_message_sha256"),
        "identity_file_sha256": lock.get("identity_file_sha256"),
        "identity_scoped_attempt_lock_file_sha256": sha256_bytes(lock_bytes),
        "calibration_binding_file_sha256": lock.get("calibration_binding_file_sha256"),
        STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: lock.get(
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD
        ),
        "source_manifest_file_sha256": lock.get("source_manifest_file_sha256"),
        "stage_a_input_bundle_manifest_file_sha256": lock.get(
            "stage_a_input_bundle_manifest_file_sha256"
        ),
        "execution_bindings": lock.get("execution_bindings"),
        "method_specs": lock.get("method_specs"),
        "expected_forward_count": lock.get("expected_forward_count"),
        "observed_forward_count": 0,
        "content_materialized": False,
        "model_load_count": 0,
        "evaluation_complete": False,
        "result_available": False,
        "automatic_retry_authorized": False,
        "claim_boundary": lock.get("claim_boundary"),
        "lock_only_recovered_at_utc": datetime.now(UTC).isoformat(),
    }
    _authenticate_recovery_boundary(
        config,
        receipt,
        allow_pre_cas_head=True,
    )
    payload = canonical_json_bytes(receipt)
    _exclusive_write(config.attempt_path, payload)
    if (
        _stable_file_bytes(
            config.attempt_path,
            context="lock-only recovered Stage-A attempt receipt",
        )
        != payload
    ):
        raise StageAError("lock-only recovered Stage-A receipt changed after publication")


def recover_interrupted(config: StageAConfig) -> Mapping[str, object]:
    """Reconcile durable receipts without re-entering content or model evaluation."""

    output_dir = _safe_directory(config.output_dir, create=False)
    if any(
        Path(os.path.abspath(path)).parent != output_dir
        for path in (config.output_path, config.attempt_path, config.complete_path)
    ):
        raise StageAError("Stage-A recovery paths escaped the authenticated output directory")
    if not config.attempt_path.is_file():
        _recover_lock_only_attempt(config)
    receipt = _strict_json(
        _stable_file_bytes(config.attempt_path, context="Stage-A attempt receipt"),
        context="Stage-A attempt receipt",
    )
    if (
        receipt.get("schema") != ATTEMPT_SCHEMA
        or receipt.get("automatic_retry_authorized") is not False
    ):
        raise StageAError("Stage-A attempt receipt identity drifted")
    status = receipt.get("status")
    seal = _require_sha1(receipt.get("one_run_seal_commit"), context="receipt seal commit")
    h1 = _require_sha1(receipt.get("h1_identity_commit"), context="receipt H1 commit")
    head = _require_sha1(
        _git(config.git_executable_path, config.repository_root, "rev-parse", "HEAD"),
        context="HEAD",
    )
    pre_cas_statuses = {
        "prepared_before_head_cas",
        "pre_cas_attempt_receipt_present_no_automatic_retry",
    }
    pre_cas = status in pre_cas_statuses and head == h1 and not config.output_path.exists()
    bootstrap, binding_hash, authenticated_seal = _authenticate_recovery_boundary(
        config,
        receipt,
        allow_pre_cas_head=pre_cas,
    )
    if seal != authenticated_seal:
        raise StageAError("recovery seal authentication returned a different commit")
    if config.output_path.is_file():
        payload = _stable_file_bytes(config.output_path, context="published Stage-A result")
        result_file_sha256 = _require_sha256(
            receipt.get("result_file_sha256"),
            context="receipt result file SHA-256",
        )
        if sha256_bytes(payload) != result_file_sha256:
            raise StageAError("published Stage-A result differs from its receipt")
        result = _strict_json(payload, context="published Stage-A result")
        if canonical_json_bytes(result) != payload:
            raise StageAError("published Stage-A result is not canonical JSON")
        verify_execution_artifact(
            payload,
            expected_identity_file_sha256=bootstrap.file_sha256,
            expected_calibration_binding_file_sha256=binding_hash,
            expected_stage_a_capture_provenance_receipt_file_sha256=_require_sha256(
                receipt.get(STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD),
                context="recovery receipt capture provenance SHA-256",
            ),
            expected_h1_commit=h1,
            expected_seal_commit=authenticated_seal,
            expected_source_commit=_require_sha1(
                receipt.get("h0_source_commit"),
                context="recovery receipt H0 source commit",
            ),
            expected_source_manifest_file_sha256=_require_sha256(
                receipt.get("source_manifest_file_sha256"),
                context="recovery receipt source manifest SHA-256",
            ),
            expected_input_bundle_manifest_file_sha256=_require_sha256(
                receipt.get("stage_a_input_bundle_manifest_file_sha256"),
                context="recovery receipt Stage-A input bundle manifest SHA-256",
            ),
            expected_execution_bindings=cast(
                Mapping[str, str],
                receipt.get("execution_bindings"),
            ),
            expected_method_specs=cast(
                Sequence[Mapping[str, object]],
                receipt.get("method_specs"),
            ),
            expected_materialization={
                "sequence_count": 12,
                "capture_input_sha256": receipt.get("capture_input_sha256"),
                "token_sequence_manifest_sha256": receipt.get("token_sequence_manifest_sha256"),
                "tokenizer_manifest_sha256": receipt.get("tokenizer_manifest_sha256"),
            },
            expected_seal_tree=_require_sha1(
                receipt.get("one_run_seal_tree"),
                context="recovery receipt seal tree",
            ),
            expected_seal_message_sha256=_require_sha256(
                receipt.get("one_run_seal_message_sha256"),
                context="recovery receipt seal message SHA-256",
            ),
            expected_attempt_lock_file_sha256=_require_sha256(
                receipt.get("identity_scoped_attempt_lock_file_sha256"),
                context="recovery receipt attempt-lock SHA-256",
            ),
            expected_authenticated_runtime=cast(
                Mapping[str, object],
                receipt.get("post_load_authenticated_runtime"),
            ),
            expected_device_runtime=cast(
                Mapping[str, object],
                receipt.get("post_load_device_runtime"),
            ),
            expected_forward_count=_nonnegative_int(
                receipt.get("expected_forward_count"),
                context="recovery receipt expected forward count",
            ),
        )
        evidence = result.get("evidence")
        if not isinstance(evidence, Mapping):
            raise StageAError("published Stage-A result evidence is missing")
        result_smoke = evidence.get("preseal_engine_smoke")
        receipt_smoke = receipt.get("preseal_engine_smoke")
        if (
            not isinstance(result_smoke, Mapping)
            or not isinstance(receipt_smoke, Mapping)
            or dict(result_smoke) != dict(receipt_smoke)
            or sha256_bytes(canonical_json_bytes(dict(result_smoke)))
            != _require_sha256(
                receipt.get("preseal_engine_smoke_sha256"),
                context="recovery receipt pre-seal smoke SHA-256",
            )
        ):
            raise StageAError("published Stage-A pre-seal smoke differs from the receipt")
        canonical_evidence_sha256 = _require_sha256(
            result.get("canonical_evidence_sha256"),
            context="published result canonical evidence SHA-256",
        )
        if canonical_evidence_sha256 != receipt.get("result_canonical_evidence_sha256"):
            raise StageAError("published Stage-A result canonical identity differs from receipt")
        completed_statuses = {
            "completed_with_authenticated_stage_a_result",
            "completed_result_published_receipt_recovered",
        }
        if status not in completed_statuses:
            updated = {
                **receipt,
                "status": "completed_result_published_receipt_recovered",
                "result_available": True,
                "recovered_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_replace_owned(config.attempt_path, canonical_json_bytes(updated))
            status = updated["status"]
        elif receipt.get("result_available") is not True:
            raise StageAError("completed Stage-A receipt does not mark the result available")
        marker = _completion_marker_bytes(
            config,
            result_file_sha256=result_file_sha256,
        )
        if config.complete_path.exists():
            if (
                not config.complete_path.is_file()
                or _stable_file_bytes(
                    config.complete_path,
                    context="Stage-A completion marker",
                )
                != marker
            ):
                raise StageAError("Stage-A completion marker differs from recovered evidence")
        else:
            _atomic_publish_new(config.complete_path, marker)
        if (
            _stable_file_bytes(
                config.complete_path,
                context="recovered Stage-A completion marker",
            )
            != marker
        ):
            raise StageAError("recovered Stage-A completion marker changed after publication")
        return {
            "status": status,
            "result_available": True,
            "completion_marker_available": True,
            "automatic_retry_authorized": False,
        }
    if config.complete_path.exists():
        raise StageAError("Stage-A completion marker exists without its published result")
    completed_statuses = {
        "completed_with_authenticated_stage_a_result",
        "completed_result_published_receipt_recovered",
    }
    if status in completed_statuses or receipt.get("result_available") is True:
        raise StageAError("completed Stage-A evidence is missing its published result")
    stable_no_result_statuses = {
        "consumed_attempt_interrupted_no_result",
        "consumed_attempt_failed_no_automatic_retry",
        "pre_cas_attempt_receipt_present_no_automatic_retry",
    }
    if head == seal:
        state = (
            status
            if status
            in stable_no_result_statuses - {"pre_cas_attempt_receipt_present_no_automatic_retry"}
            else "consumed_attempt_interrupted_no_result"
        )
    elif head == h1 and status in pre_cas_statuses:
        state = "pre_cas_attempt_receipt_present_no_automatic_retry"
    else:
        raise StageAError("receipt, HEAD, and one-run seal cannot be reconciled")
    if state == status:
        return {
            "status": state,
            "result_available": False,
            "automatic_retry_authorized": False,
        }
    updated = {
        **receipt,
        "status": state,
        "result_available": False,
        "automatic_retry_authorized": False,
        "recovered_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_replace_owned(config.attempt_path, canonical_json_bytes(updated))
    return {"status": state, "result_available": False, "automatic_retry_authorized": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare-inputs", "preflight", "execute", "recover"),
    )
    parser.add_argument("--frozen-identity", required=True, type=Path)
    parser.add_argument("--stage-a-calibration-binding", required=True, type=Path)
    parser.add_argument("--stage-a-capture-provenance-receipt", required=True, type=Path)
    parser.add_argument(
        "--expected-stage-a-capture-provenance-receipt-sha256",
        required=True,
    )
    parser.add_argument("--repository-source-manifest", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--model-file-manifest", required=True, type=Path)
    parser.add_argument("--expected-model-file-manifest-sha256", required=True)
    parser.add_argument("--parquet-materialization-manifest", required=True, type=Path)
    parser.add_argument("--expected-parquet-materialization-manifest-sha256", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--ruler-root", required=True, type=Path)
    parser.add_argument("--input-bundle-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--identity-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _config_from_args(
    args: argparse.Namespace,
    *,
    base_runtime_root: Path | None,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
    interpreter_path: Path | None,
    git_executable_path: Path | None,
    pycache_prefix: Path | None,
) -> StageAConfig:
    return StageAConfig(
        frozen_identity_path=args.frozen_identity,
        calibration_binding_path=args.stage_a_calibration_binding,
        stage_a_capture_provenance_receipt_path=(args.stage_a_capture_provenance_receipt),
        repository_source_manifest_path=args.repository_source_manifest,
        runtime_manifest_path=args.runtime_manifest,
        model_file_manifest_path=args.model_file_manifest,
        parquet_materialization_manifest_path=args.parquet_materialization_manifest,
        model_root=args.model_root,
        cache_root=args.cache_root,
        ruler_root=args.ruler_root,
        input_bundle_root=args.input_bundle_root,
        repository_root=args.repository_root,
        source_commit=args.source_commit,
        identity_commit=args.identity_commit,
        output_dir=args.output_dir,
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        expected_model_file_manifest_sha256=args.expected_model_file_manifest_sha256,
        expected_parquet_materialization_manifest_sha256=(
            args.expected_parquet_materialization_manifest_sha256
        ),
        expected_stage_a_capture_provenance_receipt_sha256=(
            args.expected_stage_a_capture_provenance_receipt_sha256
        ),
        base_runtime_root=base_runtime_root,
        package_roots=MappingProxyType(dict(package_roots)),
        package_import_paths=MappingProxyType(dict(package_import_paths)),
        interpreter_path=interpreter_path,
        git_executable_path=git_executable_path,
        pycache_prefix=pycache_prefix,
    )


def sealed_main(
    argv: Sequence[str],
    *,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
    interpreter_path: Path,
    git_executable_path: Path,
    pycache_prefix: Path,
) -> int:
    args = _parser().parse_args(list(argv))
    config = _config_from_args(
        args,
        base_runtime_root=base_runtime_root,
        package_roots=package_roots,
        package_import_paths=package_import_paths,
        interpreter_path=interpreter_path,
        git_executable_path=git_executable_path,
        pycache_prefix=pycache_prefix,
    )
    if args.mode == "prepare-inputs":
        report = prepare_inputs_production(config)
        print(json.dumps(dict(report), indent=2, sort_keys=True))
        return 0
    if args.mode == "recover":
        report = recover_interrupted(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    services = default_services()
    if args.mode == "preflight":
        authenticated = services.authenticate(config)
        print(
            json.dumps(
                {
                    "status": "stage_a_preflight_pass",
                    "identity_file_sha256": authenticated.bootstrap_identity.file_sha256,
                    "calibration_binding_file_sha256": authenticated.binding.file_sha256,
                    "source_commit": authenticated.source_commit,
                    "stage_a_input_bundle_manifest_file_sha256": (
                        authenticated.input_bundle_manifest_file_sha256
                    ),
                    "expected_forward_count": (
                        authenticated.bootstrap_identity.expected_forward_count
                    ),
                    "content_materialized": False,
                    "model_weight_files_authenticated": True,
                    "model_instantiated": False,
                    "one_run_reserved": False,
                    "method_order": list(METHOD_ORDER),
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = run_stage_a(config, services)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stage_a_passed"] is True else 2


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise StageAError(
        "Stage A must run through launch_static_q468_stage_a.py in the authenticated sealed runtime"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
