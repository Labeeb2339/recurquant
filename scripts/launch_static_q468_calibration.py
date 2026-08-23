#!/usr/bin/env python3
"""Launch Experiment 013 in an authenticated, site-free staged Python runtime.

This host process is metadata-only.  It verifies the frozen runtime, source, and
identity bindings, starts the staged interpreter with an exact isolation argv,
and lets the child repeat every material verification before loading the runner.
Model weights and calibration datasets are intentionally outside this module.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

RUNTIME_MANIFEST_KIND: Final = "recurquant_experiment013_calibration_runtime_manifest"
RUNTIME_MANIFEST_SCHEMA: Final = 6
IDENTITY_SCHEMA: Final = 5
BASE_RUNTIME_ROOT_NAME: Final = "base-runtime"
RUNNER_SOURCE_PATH: Final = "scripts/run_static_q468_calibration.py"
CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH: Final = "scripts/capture_static_q468_identity_input.py"
RUNNER_MODULE_NAME: Final = "_recurquant_experiment013_sealed_runner"
RUNNER_REVISION: Final = "experiment-013-static-q468-calibration-runner-v13"
RUN_REPORT_KIND: Final = "recurquant_experiment013_calibration_run"
RUN_REPORT_SCHEMA: Final = 3
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_calibration_identity_capture_provenance"
)
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA: Final = 2
CALIBRATION_IDENTITY_CAPTURE_VERSION: Final = 6
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS: Final = (
    "captured_under_authenticated_runtime_and_launcher_finalized"
)
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT: Final = (
    "sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1"
)
STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_stage_a_identity_capture_provenance"
)
STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA: Final = 1
STAGE_A_CALIBRATION_BINDING_KIND: Final = "recurquant_experiment013_stage_a_calibration_binding"
STAGE_A_CALIBRATION_BINDING_SCHEMA: Final = 4
STAGE_A_CALIBRATION_BINDING_REVISION: Final = "experiment-013-stage-a-calibration-binding-v4"
STAGE_A_CALIBRATION_AUTHORIZATION_KIND: Final = (
    "recurquant_experiment013_stage_a_calibration_authorization"
)
STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA: Final = 1
STAGE_A_CALIBRATION_AUTHORIZATION_REVISION: Final = (
    "experiment-013-stage-a-calibration-authorization-v1"
)
STAGE_A_CALIBRATION_AUTHORIZATION_STATUS: Final = "authorized_for_stage_a"
CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES: Final = (
    "pkg_resources",
    "setuptools",
)
CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS: Final = {
    "datasets": "datasets",
    "fsspec": "fsspec",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "six": "six",
    "tokenizers": "tokenizers",
    "transformers": "transformers",
}
FISHER_SMOKE_COMPLETE_BYTES: Final = b"recurquant-experiment013-fisher-h1-smoke-complete-v1\n"
_WINDOWS_REPARSE_POINT: Final = 0x400
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}")
_ROOT_NAME_RE: Final = re.compile(r"[a-z][a-z0-9-]{0,63}")
_FORBIDDEN_SUFFIXES: Final = frozenset({".egg-link", ".pth", ".pyc", ".pyo", "._pth"})
_FORBIDDEN_DIRECTORIES: Final = frozenset({"__pycache__"})
_FORBIDDEN_FILENAMES: Final = frozenset({"pyvenv.cfg", "sitecustomize.py", "usercustomize.py"})
_WINDOWS_RESERVED_NAMES: Final = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
EXECUTABLE_CUSTODY_MODE: Final = "platform-held-launch-handles-v1"
SEALED_LAUNCH_POLICY: Final = {
    "bootstrap_mode": "stdlib-only-exact-runner-and-capture-v3",
    "cache_confinement_mode": (
        "private-scratch-plus-explicit-dataset-and-capture-hub-root-v2"
    ),
    "child_cwd_mode": "authenticated-launcher-owned-scratch-v1",
    "dont_write_bytecode": 1,
    "executable_custody_mode": EXECUTABLE_CUSTODY_MODE,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "package_path_mode": "authenticated-record-only-roots-v1",
    "pycache_mode": "new-verified-empty-prefix-v1",
    "safe_path": True,
    "site_loaded": False,
    "sys_path_mode": "staged-base-then-authenticated-packages-v1",
    "utf8_mode": 1,
    "virtualenv_hook_loaded": False,
}
_BOUND_ARTIFACT_OPTIONS: Final = {
    "calibration_runtime_manifest_file_sha256": "--runtime-manifest",
    "model_file_manifest_file_sha256": "--model-file-manifest",
    "parquet_materialization_manifest_file_sha256": "--parquet-materialization-manifest",
    "repository_source_manifest_file_sha256": "--repository-source-manifest",
}
_EXPECTED_BOUND_DIGEST_OPTIONS: Final = {
    "calibration_runtime_manifest_file_sha256": "--expected-runtime-manifest-sha256",
    "model_file_manifest_file_sha256": "--expected-model-file-manifest-sha256",
    "parquet_materialization_manifest_file_sha256": (
        "--expected-parquet-materialization-manifest-sha256"
    ),
}
_STAGE_A_CALIBRATION_BINDING_FIELDS: Final = frozenset(
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
_CALIBRATION_CAPTURE_COMMAND: Final = "capture-calibration-identity"
_STAGE_A_CAPTURE_COMMAND: Final = "capture-stage-a-identity"
_CAPTURE_COMMANDS: Final = frozenset({_CALIBRATION_CAPTURE_COMMAND, _STAGE_A_CAPTURE_COMMAND})
_CAPTURE_EXPECTED_BOUND_DIGEST_OPTIONS: Final = {
    **_EXPECTED_BOUND_DIGEST_OPTIONS,
    "repository_source_manifest_file_sha256": ("--expected-repository-source-manifest-sha256"),
}
_CALIBRATION_CAPTURE_REQUIRED_RUNNER_OPTIONS: Final = frozenset(
    {
        "--cache-root",
        "--capture-provenance-receipt-output",
        "--output",
        "--repository-root",
        "--ruler-receipt-dir",
        "--source-commit",
        *_BOUND_ARTIFACT_OPTIONS.values(),
        *_CAPTURE_EXPECTED_BOUND_DIGEST_OPTIONS.values(),
    }
)
_STAGE_A_CAPTURE_REQUIRED_RUNNER_OPTIONS: Final = frozenset(
    {
        *_CALIBRATION_CAPTURE_REQUIRED_RUNNER_OPTIONS,
        "--expected-stage-a-calibration-binding-sha256",
        "--stage-a-calibration-binding",
    }
)
_REQUIRED_RUNNER_OPTIONS: Final = frozenset(
    {
        "--capture-provenance-receipt",
        "--expected-capture-provenance-receipt-sha256",
        "--frozen-identity",
        "--cache-root",
        "--repository-root",
        "--ruler-receipt-dir",
        "--source-commit",
        *_BOUND_ARTIFACT_OPTIONS.values(),
        *_EXPECTED_BOUND_DIGEST_OPTIONS.values(),
    }
)
_FORBIDDEN_RUNNER_OPTIONS: Final = frozenset({"--ruler-root"})
_SMOKE_PREREQUISITE_OPTIONS: Final = frozenset(
    {
        "--prior-fisher-h1-smoke-report",
        "--prior-fisher-h1-smoke-complete-marker",
    }
)
_SENSITIVE_MODULES: Final = frozenset(
    {
        "_virtualenv",
        RUNNER_MODULE_NAME,
        "recurquant",
        "recurquant.experiment013_calibration_api",
        "recurquant.experiment013_qwen35_adapter",
        "recurquant.experiment013_source",
        "recurquant_experiment013_calibration_identity_capture",
        "_recurquant_experiment013_calibration_runner_for_capture",
        "recurquant_experiment013_identity_resolver",
        "datasets",
        "fsspec",
        "huggingface_hub",
        "numpy",
        "pkg_resources",
        "pyarrow",
        "setuptools",
        "tokenizers",
        "torch",
        "transformers",
        "site",
    }
)


class SealedLaunchError(RuntimeError):
    """Raised when the staged launch cannot be authenticated safely."""


def _canonical_json_bytes(value: object) -> bytes:
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


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(data: bytes, *, context: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{context} must be bytes")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SealedLaunchError(f"{context} contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SealedLaunchError(f"{context} contains a non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedLaunchError(f"{context} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SealedLaunchError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise SealedLaunchError(f"{context} fields differ from the frozen schema")


def _canonical_base64(value: object, *, context: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise SealedLaunchError(f"{context} must be canonical base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SealedLaunchError(f"{context} must be canonical base64 text") from error
    if not decoded or base64.b64encode(decoded).decode("ascii") != value:
        raise SealedLaunchError(f"{context} must be canonical base64 text")
    return decoded


def _parse_stage_a_calibration_binding_envelope(
    data: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, object]:
    """Bootstrap the exact v4 envelope; the sealed child rederives the full chain."""

    actual = _sha256_bytes(data)
    if actual != _sha256(expected_file_sha256, context="expected Stage-A binding SHA-256"):
        raise SealedLaunchError("Stage-A calibration binding differs from its explicit SHA-256")
    root = _strict_json(data, context="Stage-A calibration binding")
    _exact_fields(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="Stage-A calibration binding",
    )
    if _canonical_json_bytes(root) != data:
        raise SealedLaunchError("Stage-A calibration binding is not canonical JSON")
    if (
        root["artifact_kind"] != STAGE_A_CALIBRATION_BINDING_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != STAGE_A_CALIBRATION_BINDING_SCHEMA
    ):
        raise SealedLaunchError("Stage-A calibration binding kind or schema drifted")
    evidence = root.get("evidence")
    if not isinstance(evidence, dict):
        raise SealedLaunchError("Stage-A calibration binding evidence is missing")
    _exact_fields(
        evidence,
        {"artifact_revision", "binding", "dependencies_base64", "dependency_file_sha256"},
        context="Stage-A calibration binding evidence",
    )
    if evidence["artifact_revision"] != STAGE_A_CALIBRATION_BINDING_REVISION:
        raise SealedLaunchError("Stage-A calibration binding revision drifted")
    if _sha256(
        root["canonical_evidence_sha256"], context="Stage-A binding evidence SHA-256"
    ) != _sha256_bytes(_canonical_json_bytes(evidence)):
        raise SealedLaunchError("Stage-A calibration binding evidence hash drifted")
    binding = evidence.get("binding")
    encoded = evidence.get("dependencies_base64")
    hashes = evidence.get("dependency_file_sha256")
    if (
        not isinstance(binding, dict)
        or not isinstance(encoded, dict)
        or not isinstance(hashes, dict)
    ):
        raise SealedLaunchError("Stage-A calibration binding dependencies are missing")
    _exact_fields(binding, set(_STAGE_A_CALIBRATION_BINDING_FIELDS), context="Stage-A binding")
    for name in sorted(_STAGE_A_CALIBRATION_BINDING_FIELDS):
        _sha256(binding[name], context=f"Stage-A binding {name}")
    dependency_name = "calibration_authorization_artifact"
    _exact_fields(encoded, {dependency_name}, context="Stage-A binding dependencies")
    _exact_fields(hashes, {dependency_name}, context="Stage-A binding dependency hashes")
    authorization_bytes = _canonical_base64(
        encoded[dependency_name], context="Stage-A calibration authorization"
    )
    authorization_sha256 = _sha256(
        hashes[dependency_name], context="Stage-A calibration authorization SHA-256"
    )
    if _sha256_bytes(authorization_bytes) != authorization_sha256:
        raise SealedLaunchError("Stage-A calibration authorization bytes differ from their hash")
    if binding["calibration_authorization_file_sha256"] != authorization_sha256:
        raise SealedLaunchError("Stage-A binding carries a different authorization hash")
    authorization = _strict_json(authorization_bytes, context="Stage-A calibration authorization")
    _exact_fields(
        authorization,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="Stage-A calibration authorization",
    )
    if _canonical_json_bytes(authorization) != authorization_bytes:
        raise SealedLaunchError("Stage-A calibration authorization is not canonical JSON")
    if (
        authorization["artifact_kind"] != STAGE_A_CALIBRATION_AUTHORIZATION_KIND
        or type(authorization["schema_version"]) is not int
        or authorization["schema_version"] != STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA
    ):
        raise SealedLaunchError("Stage-A calibration authorization kind or schema drifted")
    authorization_evidence = authorization.get("evidence")
    if not isinstance(authorization_evidence, dict):
        raise SealedLaunchError("Stage-A calibration authorization evidence is missing")
    _exact_fields(
        authorization_evidence,
        {
            "artifact_revision",
            "authorized_output_file_sha256",
            "bindings",
            "dependencies_base64",
            "dependency_file_sha256",
            "status",
        },
        context="Stage-A calibration authorization evidence",
    )
    if (
        authorization_evidence["artifact_revision"] != STAGE_A_CALIBRATION_AUTHORIZATION_REVISION
        or authorization_evidence["status"] != STAGE_A_CALIBRATION_AUTHORIZATION_STATUS
        or _sha256(
            authorization["canonical_evidence_sha256"],
            context="Stage-A authorization evidence SHA-256",
        )
        != _sha256_bytes(_canonical_json_bytes(authorization_evidence))
    ):
        raise SealedLaunchError("Stage-A calibration authorization envelope drifted")
    custody = authorization_evidence.get("bindings")
    if not isinstance(custody, dict):
        raise SealedLaunchError("Stage-A calibration authorization custody bindings are missing")
    _exact_fields(
        custody,
        {
            "calibration_core_binding_file_sha256",
            "calibration_run_report_file_sha256",
            "capture_provenance_receipt_file_sha256",
            "execution_bindings",
            "fisher_h1_smoke_report_file_sha256",
            "frozen_calibration_identity_file_sha256",
            "identity_input_manifest_sha256",
            "source_commit",
            "static_q48_policy_file_sha256",
        },
        context="Stage-A calibration authorization custody bindings",
    )
    execution_bindings = custody.get("execution_bindings")
    if not isinstance(execution_bindings, dict):
        raise SealedLaunchError("Stage-A authorization execution bindings are missing")
    _exact_fields(
        execution_bindings,
        set(_BOUND_ARTIFACT_OPTIONS),
        context="Stage-A authorization execution bindings",
    )
    normalized_execution = {
        name: _sha256(execution_bindings[name], context=f"Stage-A authorization {name}")
        for name in sorted(execution_bindings)
    }
    source_commit = custody.get("source_commit")
    if not isinstance(source_commit, str) or _SHA1_RE.fullmatch(source_commit) is None:
        raise SealedLaunchError("Stage-A authorization source commit must be a SHA-1")
    return {
        "authorization_file_sha256": authorization_sha256,
        "execution_bindings": normalized_execution,
        "file_sha256": actual,
        "source_commit": source_commit,
    }


def _exact_typed_mapping(
    value: object,
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise SealedLaunchError(f"{context} fields differ from the frozen schema")
    if any(
        type(value[name]) is not type(expected[name]) or value[name] != expected[name]
        for name in expected
    ):
        raise SealedLaunchError(f"{context} value or JSON type drifted")


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SealedLaunchError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SealedLaunchError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SealedLaunchError(f"{context} must be a non-negative integer")
    return value


def _canonical_relative_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SealedLaunchError(f"{context} is not a canonical relative path")
    if any(character in value for character in ("\\", "\0", "\n", "\r", ":")):
        raise SealedLaunchError(f"{context} is not a safe POSIX path")
    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SealedLaunchError(f"{context} is not a canonical relative path")
    for part in path.parts:
        if part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise SealedLaunchError(f"{context} is unsafe on Windows")
    return value


def _canonical_base_sys_path_entry(value: object, *, context: str) -> str:
    # CPython's Windows runtime layout uses the exact ``.`` sentinel for the
    # base-runtime root.  Keep that exception local to base_sys_path: every
    # other manifest path must remain a non-dot canonical relative path.
    if value == ".":
        return "."
    return _canonical_relative_path(value, context=context)


def _root_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _ROOT_NAME_RE.fullmatch(value) is None:
        raise SealedLaunchError(f"{context} is not a canonical runtime-root name")
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as exc:
        raise SealedLaunchError(f"required path is unavailable: {path}") from exc
    return path.is_symlink() or bool(
        getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _absolute_directory(path: Path, *, context: str) -> Path:
    candidate = Path(os.path.abspath(path))
    if _is_link_or_reparse(candidate):
        raise SealedLaunchError(f"{context} is a link or reparse point")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SealedLaunchError(f"{context} is unavailable") from exc
    if not resolved.is_dir():
        raise SealedLaunchError(f"{context} is not a directory")
    return resolved


def _safe_join(root: Path, relative: str, *, context: str, directory: bool = False) -> Path:
    candidate = root
    for part in PurePosixPath(_canonical_relative_path(relative, context=context)).parts:
        candidate /= part
        if _is_link_or_reparse(candidate):
            raise SealedLaunchError(f"{context} traverses a link or reparse point")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SealedLaunchError(f"{context} escapes its authenticated root") from exc
    if directory and not resolved.is_dir():
        raise SealedLaunchError(f"{context} is not a directory")
    if not directory and not resolved.is_file():
        raise SealedLaunchError(f"{context} is not a regular file")
    return resolved


def _forbidden_runtime_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(part.casefold() in _FORBIDDEN_DIRECTORIES for part in path.parts)
        or path.name.casefold() in _FORBIDDEN_FILENAMES
        or path.suffix.casefold() in _FORBIDDEN_SUFFIXES
    )


def _stable_file_record(path: Path, *, relative: str, context: str) -> dict[str, object]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or _is_link_or_reparse(path):
        raise SealedLaunchError(f"{context} is not a stable regular file")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise SealedLaunchError(f"cannot read {context}") from exc
    after = path.stat()
    if (
        not stat.S_ISREG(after.st_mode)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size != after.st_size
        or _is_link_or_reparse(path)
    ):
        raise SealedLaunchError(f"{context} changed while it was authenticated")
    return {"path": relative, "sha256": digest.hexdigest(), "size_bytes": size}


def _stable_file_bytes(path: Path, *, context: str) -> bytes:
    before = _stable_file_record(path, relative=path.name, context=context)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SealedLaunchError(f"cannot read {context}") from exc
    after = _stable_file_record(path, relative=path.name, context=context)
    if (
        before != after
        or _sha256_bytes(data) != before["sha256"]
        or len(data) != before["size_bytes"]
    ):
        raise SealedLaunchError(f"{context} changed while it was read")
    return data


CaptureArtifactSnapshot = tuple[
    Path,
    tuple[tuple[str, int, int, int], ...],
]


def _new_capture_artifact_snapshot(
    path: Path,
    *,
    context: str,
) -> CaptureArtifactSnapshot:
    raw = Path(path)
    if not raw.is_absolute():
        raise SealedLaunchError(f"{context} must be absolute")
    destination = Path(os.path.abspath(raw))
    if not destination.name:
        raise SealedLaunchError(f"{context} cannot be a filesystem root")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing {context}: {destination}")
    parent_identities = _non_link_directory_identity_chain(
        destination.parent,
        context=f"{context} parent",
    )
    parent = _absolute_directory(destination.parent, context=f"{context} parent")
    return parent / destination.name, parent_identities


def _revalidate_capture_artifact_snapshot(
    snapshot: CaptureArtifactSnapshot,
    *,
    context: str,
    expect_absent: bool,
) -> Path:
    destination, expected_parent_identities = snapshot
    parent_identities = _non_link_directory_identity_chain(
        destination.parent,
        context=f"{context} parent",
    )
    parent = _absolute_directory(destination.parent, context=f"{context} parent")
    if parent / destination.name != destination or parent_identities != expected_parent_identities:
        raise SealedLaunchError(f"{context} parent changed during sealed capture")
    exists = os.path.lexists(destination)
    if expect_absent and exists:
        raise FileExistsError(f"refusing to overwrite existing {context}: {destination}")
    if not expect_absent and not exists:
        raise SealedLaunchError(f"{context} was not published by the sealed child")
    return destination


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        pass
    else:
        return True
    try:
        right.relative_to(left)
    except ValueError:
        return False
    return True


def _validate_capture_artifact_disjointness(
    snapshots: Sequence[CaptureArtifactSnapshot],
    *,
    forbidden_roots: Sequence[Path],
) -> None:
    destinations = [snapshot[0] for snapshot in snapshots]
    if len(set(destinations)) != len(destinations):
        raise SealedLaunchError("capture identity and provenance output paths must differ")
    for destination in destinations:
        for root in forbidden_roots:
            authenticated_root = _absolute_directory(root, context="capture forbidden root")
            if _paths_overlap(destination.parent, authenticated_root):
                raise SealedLaunchError(
                    "capture output parent overlaps an authenticated runtime or cache root"
                )


def _atomic_publish_capture_receipt(
    snapshot: CaptureArtifactSnapshot,
    payload: bytes,
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("capture provenance receipt payload must be bytes")
    destination = _revalidate_capture_artifact_snapshot(
        snapshot,
        context="calibration identity capture provenance receipt",
        expect_absent=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _revalidate_capture_artifact_snapshot(
            snapshot,
            context="calibration identity capture provenance receipt",
            expect_absent=True,
        )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing capture provenance receipt: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _absolute_path_sha256(path: Path) -> str:
    return _sha256_bytes(os.path.normcase(str(path.resolve(strict=True))).encode("utf-8"))


class _WindowsFileTime(ctypes.Structure):
    _fields_ = [
        ("low", ctypes.c_uint32),
        ("high", ctypes.c_uint32),
    ]


class _WindowsByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", ctypes.c_uint32),
        ("creation_time", _WindowsFileTime),
        ("last_access_time", _WindowsFileTime),
        ("last_write_time", _WindowsFileTime),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    ]


def _windows_handle_identity(handle: int) -> dict[str, int]:
    """Return the kernel identity of a live Windows custody handle."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileInformationByHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = ctypes.c_int
    information = _WindowsByHandleFileInformation()
    ctypes.set_last_error(0)
    if not kernel32.GetFileInformationByHandle(ctypes.c_void_p(handle), ctypes.byref(information)):
        raise SealedLaunchError("cannot query live Windows executable custody handle") from (
            ctypes.WinError(ctypes.get_last_error())
        )
    return {
        "file_attributes": int(information.file_attributes),
        "file_index": (int(information.file_index_high) << 32) | int(information.file_index_low),
        "size_bytes": (int(information.file_size_high) << 32) | int(information.file_size_low),
        "volume_serial_number": int(information.volume_serial_number),
    }


class _HeldExecutableCustody:
    """Hold authenticated launch executables stable until the child chain returns.

    On Windows the underlying handles permit only additional readers, so new
    data/append-write and delete/rename opens are rejected by the kernel while
    custody is active. POSIX descriptors keep the exact inode alive for
    portability tests; a non-Windows runtime is never protocol-eligible
    Experiment 013 evidence.
    """

    def __init__(
        self,
        *,
        entries: tuple[dict[str, object], ...],
        windows_handles: tuple[int, ...] = (),
        posix_descriptors: tuple[int, ...] = (),
    ) -> None:
        self._entries = entries
        self._windows_handles = list(windows_handles)
        self._posix_descriptors = list(posix_descriptors)
        self._closed = False

    @property
    def record(self) -> dict[str, object]:
        return {
            "entries": [
                {key: value for key, value in entry.items() if key not in {"path", "path_identity"}}
                for entry in self._entries
            ],
            "mode": EXECUTABLE_CUSTODY_MODE,
            "protocol_eligible": os.name == "nt",
        }

    def verify(self) -> None:
        if self._closed:
            raise SealedLaunchError("executable custody was released before launch completion")
        roles = tuple(entry.get("role") for entry in self._entries)
        expected_roles = ("staged Python executable", "Git executable")
        if roles != expected_roles:
            raise SealedLaunchError(
                "executable custody does not contain the authenticated Python and Git roles"
            )
        if os.name == "nt":
            if len(self._windows_handles) != len(expected_roles) or self._posix_descriptors:
                raise SealedLaunchError("Windows executable custody handle set is incomplete")
        elif len(self._posix_descriptors) != len(expected_roles) or self._windows_handles:
            raise SealedLaunchError("POSIX executable custody descriptor set is incomplete")
        for index, entry in enumerate(self._entries):
            path = entry["path"]
            assert isinstance(path, Path)
            context = str(entry["role"])
            try:
                status = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise SealedLaunchError(
                    f"{context} changed while executable custody was held"
                ) from exc
            current_identity = (
                int(status.st_dev),
                int(status.st_ino),
                int(status.st_size),
                stat.S_IFMT(status.st_mode),
            )
            if current_identity != entry["path_identity"]:
                raise SealedLaunchError(f"{context} identity changed while custody was held")
            if os.name == "nt":
                held_identity = _windows_handle_identity(self._windows_handles[index])
                if held_identity != entry.get("handle_identity"):
                    raise SealedLaunchError(f"{context} custody handle identity changed")
                if (
                    held_identity["volume_serial_number"] != int(status.st_dev)
                    or held_identity["file_index"] != int(status.st_ino)
                    or held_identity["size_bytes"] != int(status.st_size)
                    or held_identity["file_attributes"] & 0x410
                ):
                    raise SealedLaunchError(
                        f"{context} custody handle is not bound to the authenticated file"
                    )
            else:
                try:
                    held_status = os.fstat(self._posix_descriptors[index])
                except OSError as exc:
                    raise SealedLaunchError(
                        f"{context} POSIX custody descriptor is not live"
                    ) from exc
                held_identity = (
                    int(held_status.st_dev),
                    int(held_status.st_ino),
                    int(held_status.st_size),
                    stat.S_IFMT(held_status.st_mode),
                )
                if held_identity != current_identity:
                    raise SealedLaunchError(
                        f"{context} POSIX custody descriptor is not bound to the path"
                    )
            record = _stable_file_record(path, relative=path.name, context=context)
            if record["sha256"] != entry["sha256"] or record["size_bytes"] != entry["size_bytes"]:
                raise SealedLaunchError(f"{context} bytes changed while custody was held")
            if os.name == "nt":
                write_error = _windows_conflicting_open_error(path, desired_access=0x40000000)
                delete_error = _windows_conflicting_open_error(path, desired_access=0x00010000)
                if write_error not in {5, 32} or delete_error not in {5, 32}:
                    raise SealedLaunchError(
                        f"{context} custody does not block data-write and delete access"
                    )

    def close(self) -> None:
        if self._closed:
            return
        failures: list[OSError] = []
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            for handle in reversed(self._windows_handles):
                if not kernel32.CloseHandle(ctypes.c_void_p(handle)):
                    failures.append(ctypes.WinError(ctypes.get_last_error()))
        else:
            for descriptor in reversed(self._posix_descriptors):
                try:
                    os.close(descriptor)
                except OSError as exc:
                    failures.append(exc)
        self._windows_handles.clear()
        self._posix_descriptors.clear()
        self._closed = True
        if failures:
            raise SealedLaunchError(
                "failed to release authenticated executable custody"
            ) from failures[0]

    def __enter__(self) -> _HeldExecutableCustody:
        try:
            self.verify()
        except BaseException as primary_error:
            try:
                self.close()
            except BaseException as close_error:
                raise BaseExceptionGroup(
                    "executable custody release failed after entry verification failed",
                    [primary_error, close_error],
                ) from None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            assert isinstance(exc, BaseException)
            raise BaseExceptionGroup(
                "executable custody release failed after a primary launch failure",
                [exc, close_error],
            ) from None
        return False


def _windows_open_custody_handle(path: Path) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    ctypes.set_last_error(0)
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ; deny data-write and delete/rename sharing
        None,
        3,  # OPEN_EXISTING
        0x00000080 | 0x00200000,  # FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in {None, invalid}:
        raise SealedLaunchError("cannot acquire Windows executable custody") from ctypes.WinError(
            ctypes.get_last_error()
        )
    return int(handle)


def _windows_conflicting_open_error(path: Path, *, desired_access: int) -> int:
    """Return the error proving a conflicting data-write/delete open was blocked."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    ctypes.set_last_error(0)
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle not in {None, invalid}:
        kernel32.CloseHandle(handle)
        return 0
    return int(ctypes.get_last_error())


def _acquire_executable_custody(
    *,
    interpreter: Path,
    git_executable: Path,
    runtime_manifest: Mapping[str, object],
) -> _HeldExecutableCustody:
    """Acquire and verify the platform launch handles bound by runtime schema v6."""

    machine = runtime_manifest.get("machine")
    if not isinstance(machine, dict) or not isinstance(machine.get("system"), str):
        raise SealedLaunchError("runtime machine identity is unavailable for custody")
    if (os.name == "nt") != (machine["system"] == "Windows"):
        raise SealedLaunchError("host and runtime platforms differ for executable custody")

    expected = {
        "staged Python executable": runtime_manifest["interpreter"],
        "Git executable": runtime_manifest["git_executable"],
    }
    paths = {
        "staged Python executable": interpreter,
        "Git executable": git_executable,
    }
    windows_handles: list[int] = []
    posix_descriptors: list[int] = []
    entries: list[dict[str, object]] = []
    try:
        for role in ("staged Python executable", "Git executable"):
            path = paths[role].resolve(strict=True)
            if _is_link_or_reparse(path):
                raise SealedLaunchError(f"{role} is a link or reparse point")
            if os.name == "nt":
                windows_handles.append(_windows_open_custody_handle(path))
            else:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                posix_descriptors.append(os.open(path, flags))
            status = path.stat(follow_symlinks=False)
            handle_identity: dict[str, int] | None = None
            if os.name == "nt":
                handle_identity = _windows_handle_identity(windows_handles[-1])
                if (
                    handle_identity["volume_serial_number"] != int(status.st_dev)
                    or handle_identity["file_index"] != int(status.st_ino)
                    or handle_identity["size_bytes"] != int(status.st_size)
                    or handle_identity["file_attributes"] & 0x410
                ):
                    raise SealedLaunchError(
                        f"{role} custody handle is not bound to the authenticated file"
                    )
            record = _stable_file_record(path, relative=path.name, context=role)
            expected_record = expected[role]
            if not isinstance(expected_record, dict):
                raise SealedLaunchError(f"runtime {role} record is missing")
            if record["sha256"] != expected_record.get("sha256") or record[
                "size_bytes"
            ] != expected_record.get("size_bytes"):
                raise SealedLaunchError(f"{role} differs from the runtime manifest")
            if role == "Git executable" and _absolute_path_sha256(path) != expected_record.get(
                "absolute_path_sha256"
            ):
                raise SealedLaunchError("Git executable path differs from the runtime manifest")
            entry: dict[str, object] = {
                "path": path,
                "path_identity": (
                    int(status.st_dev),
                    int(status.st_ino),
                    int(status.st_size),
                    stat.S_IFMT(status.st_mode),
                ),
                "role": role,
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            if os.name == "nt":
                assert handle_identity is not None
                entry["handle_identity"] = handle_identity
                write_error = _windows_conflicting_open_error(path, desired_access=0x40000000)
                delete_error = _windows_conflicting_open_error(path, desired_access=0x00010000)
                if write_error not in {5, 32} or delete_error not in {5, 32}:
                    raise SealedLaunchError(
                        f"{role} custody failed to block data-write and delete access"
                    )
                entry["delete_open_block_error"] = delete_error
                entry["write_open_block_error"] = write_error
            entries.append(entry)
        custody = _HeldExecutableCustody(
            entries=tuple(entries),
            windows_handles=tuple(windows_handles),
            posix_descriptors=tuple(posix_descriptors),
        )
        custody.verify()
        return custody
    except BaseException as primary_error:
        partial = _HeldExecutableCustody(
            entries=tuple(entries),
            windows_handles=tuple(windows_handles),
            posix_descriptors=tuple(posix_descriptors),
        )
        try:
            partial.close()
        except BaseException as close_error:
            raise BaseExceptionGroup(
                "partial executable custody release failed after acquisition failed",
                [primary_error, close_error],
            ) from None
        raise


def _authenticated_git_executable(path: Path) -> tuple[Path, dict[str, object]]:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as exc:
        raise SealedLaunchError("Git executable is unavailable") from exc
    if resolved.name.casefold() == "git.exe" and resolved.parent.name.casefold() == "cmd":
        try:
            resolved = (resolved.parent.parent / "mingw64" / "bin" / "git.exe").resolve(strict=True)
        except OSError as exc:
            raise SealedLaunchError(
                "Git-for-Windows cmd shim has no canonical mingw64 executable"
            ) from exc
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            raise SealedLaunchError("Git executable traverses a link or reparse point")
    record = _stable_file_record(resolved, relative=resolved.name, context="Git executable")
    if int(record["size_bytes"]) <= 0:
        raise SealedLaunchError("Git executable is empty")
    return resolved, {
        "absolute_path_sha256": _absolute_path_sha256(resolved),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def _tree_files(root: Path, *, context: str) -> tuple[dict[str, object], ...]:
    root = _absolute_directory(root, context=f"{context} root")
    stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    files: list[dict[str, object]] = []
    folded: set[str] = set()
    while stack:
        directory, parents = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise SealedLaunchError(f"cannot enumerate {context}") from exc
        directory_names: set[str] = set()
        for entry in entries:
            relative = _canonical_relative_path(
                PurePosixPath(*parents, entry.name).as_posix(),
                context=f"{context} path",
            )
            folded_relative = relative.casefold()
            if folded_relative in folded or folded_relative in directory_names:
                raise SealedLaunchError(f"{context} has a case-insensitive path collision")
            directory_names.add(folded_relative)
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SealedLaunchError(f"{context} path is unavailable") from exc
            if entry.is_symlink() or bool(
                getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise SealedLaunchError(f"{context} contains a link or reparse point")
            if stat.S_ISDIR(status.st_mode):
                if entry.name.casefold() in _FORBIDDEN_DIRECTORIES:
                    raise SealedLaunchError(f"{context} contains a forbidden cache directory")
                stack.append((Path(entry.path), (*parents, entry.name)))
                continue
            if not stat.S_ISREG(status.st_mode):
                raise SealedLaunchError(f"{context} contains a non-regular path")
            if _forbidden_runtime_path(relative):
                raise SealedLaunchError(f"{context} contains a forbidden runtime file")
            folded.add(folded_relative)
            files.append(
                _stable_file_record(
                    Path(entry.path),
                    relative=relative,
                    context=f"{context} file",
                )
            )
    files.sort(key=lambda item: str(item["path"]))
    if not files:
        raise SealedLaunchError(f"{context} has no files")
    return tuple(files)


def _parse_runtime_manifest(data: bytes) -> dict[str, object]:
    root = _strict_json(data, context="runtime manifest")
    _exact_fields(
        root,
        {
            "artifact_kind",
            "base_runtime_root",
            "base_sys_path",
            "distributions",
            "git_executable",
            "interpreter",
            "launch_policy",
            "machine",
            "package_roots",
            "python",
            "runtime_trees",
            "schema_version",
        },
        context="runtime manifest",
    )
    if _canonical_json_bytes(root) != data:
        raise SealedLaunchError("runtime manifest is not canonical JSON")
    if (
        root["artifact_kind"] != RUNTIME_MANIFEST_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != RUNTIME_MANIFEST_SCHEMA
    ):
        raise SealedLaunchError("runtime manifest kind or schema drifted")
    _exact_typed_mapping(
        root["launch_policy"],
        SEALED_LAUNCH_POLICY,
        context="runtime manifest launch policy",
    )

    git_executable = root["git_executable"]
    if not isinstance(git_executable, dict):
        raise SealedLaunchError("runtime Git executable record must be an object")
    _exact_fields(
        git_executable,
        {"absolute_path_sha256", "sha256", "size_bytes"},
        context="runtime Git executable",
    )
    normalized_git_executable = {
        "absolute_path_sha256": _sha256(
            git_executable["absolute_path_sha256"],
            context="runtime Git executable path SHA-256",
        ),
        "sha256": _sha256(git_executable["sha256"], context="runtime Git executable SHA-256"),
        "size_bytes": _positive_int(
            git_executable["size_bytes"], context="runtime Git executable size"
        ),
    }

    python_record = root["python"]
    if not isinstance(python_record, dict):
        raise SealedLaunchError("runtime Python record must be an object")
    _exact_fields(
        python_record,
        {"abi_flags", "cache_tag", "implementation", "version"},
        context="runtime Python record",
    )
    for field in ("abi_flags", "cache_tag", "implementation", "version"):
        value = python_record[field]
        if (
            not isinstance(value, str)
            or value != value.strip()
            or (field != "abi_flags" and not value)
        ):
            raise SealedLaunchError("runtime Python identity is invalid")

    machine = root["machine"]
    if not isinstance(machine, dict):
        raise SealedLaunchError("runtime machine record must be an object")
    _exact_fields(
        machine,
        {"architecture", "byteorder", "machine", "pointer_bits", "system"},
        context="runtime machine record",
    )
    for field in ("architecture", "machine", "system"):
        value = machine[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise SealedLaunchError("runtime machine identity is invalid")
    if machine["byteorder"] not in {"big", "little"}:
        raise SealedLaunchError("runtime byte order is invalid")
    _positive_int(machine["pointer_bits"], context="runtime pointer bits")

    if _root_name(root["base_runtime_root"], context="base runtime root") != (
        BASE_RUNTIME_ROOT_NAME
    ):
        raise SealedLaunchError("base runtime root name drifted")
    raw_base_sys_path = root["base_sys_path"]
    if not isinstance(raw_base_sys_path, list) or not raw_base_sys_path:
        raise SealedLaunchError("base sys.path must be a non-empty list")
    base_sys_path = [
        _canonical_base_sys_path_entry(item, context="base sys.path entry")
        for item in raw_base_sys_path
    ]
    if len({item.casefold() for item in base_sys_path}) != len(base_sys_path):
        raise SealedLaunchError("base sys.path entries collide")

    raw_package_roots = root["package_roots"]
    if not isinstance(raw_package_roots, list) or not raw_package_roots:
        raise SealedLaunchError("runtime manifest has no package roots")
    package_roots: list[dict[str, str]] = []
    for item in raw_package_roots:
        if not isinstance(item, dict):
            raise SealedLaunchError("package root record must be an object")
        _exact_fields(item, {"import_path", "name"}, context="package root record")
        package_roots.append(
            {
                "name": _root_name(item["name"], context="package root name"),
                "import_path": _canonical_relative_path(
                    item["import_path"], context="package import path"
                ),
            }
        )
    names = [item["name"] for item in package_roots]
    if names != sorted(names) or len(set(names)) != len(names) or BASE_RUNTIME_ROOT_NAME in names:
        raise SealedLaunchError("package roots are not unique and sorted")

    raw_trees = root["runtime_trees"]
    if not isinstance(raw_trees, list) or len(raw_trees) != len(package_roots) + 1:
        raise SealedLaunchError("runtime tree inventory differs from declared roots")
    trees: list[dict[str, object]] = []
    expected_tree_names = [BASE_RUNTIME_ROOT_NAME, *names]
    for index, raw_tree in enumerate(raw_trees):
        if not isinstance(raw_tree, dict):
            raise SealedLaunchError("runtime tree record must be an object")
        _exact_fields(raw_tree, {"files", "kind", "name"}, context="runtime tree record")
        name = _root_name(raw_tree["name"], context="runtime tree name")
        kind = raw_tree["kind"]
        expected_kind = "base-runtime" if index == 0 else "packages"
        if name != expected_tree_names[index] or kind != expected_kind:
            raise SealedLaunchError("runtime tree order or kind drifted")
        raw_files = raw_tree["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise SealedLaunchError("runtime tree has no files")
        files: list[dict[str, object]] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise SealedLaunchError("runtime file record must be an object")
            _exact_fields(
                raw_file,
                {"path", "sha256", "size_bytes"},
                context="runtime file record",
            )
            relative = _canonical_relative_path(raw_file["path"], context="runtime file path")
            if _forbidden_runtime_path(relative):
                raise SealedLaunchError("runtime manifest contains a forbidden file")
            files.append(
                {
                    "path": relative,
                    "sha256": _sha256(raw_file["sha256"], context="runtime file SHA-256"),
                    "size_bytes": _nonnegative_int(
                        raw_file["size_bytes"], context="runtime file size"
                    ),
                }
            )
        paths = [str(item["path"]) for item in files]
        if paths != sorted(paths) or len({item.casefold() for item in paths}) != len(paths):
            raise SealedLaunchError("runtime file inventory is not unique and sorted")
        trees.append({"files": files, "kind": kind, "name": name})

    interpreter = root["interpreter"]
    if not isinstance(interpreter, dict):
        raise SealedLaunchError("runtime interpreter record must be an object")
    _exact_fields(
        interpreter,
        {"relative_path", "root", "sha256", "size_bytes"},
        context="runtime interpreter record",
    )
    interpreter_path = _canonical_relative_path(
        interpreter["relative_path"], context="runtime interpreter path"
    )
    normalized_interpreter = {
        "relative_path": interpreter_path,
        "root": interpreter["root"],
        "sha256": _sha256(interpreter["sha256"], context="runtime interpreter SHA-256"),
        "size_bytes": _positive_int(interpreter["size_bytes"], context="runtime interpreter size"),
    }
    if normalized_interpreter["root"] != BASE_RUNTIME_ROOT_NAME:
        raise SealedLaunchError("runtime interpreter is not in the base tree")
    base_files = {item["path"]: item for item in trees[0]["files"]}
    expected_interpreter_file = {
        "path": interpreter_path,
        "sha256": normalized_interpreter["sha256"],
        "size_bytes": normalized_interpreter["size_bytes"],
    }
    if base_files.get(interpreter_path) != expected_interpreter_file:
        raise SealedLaunchError("runtime interpreter differs from its base-tree record")
    for entry in base_sys_path:
        present = (
            entry == "."
            or entry in base_files
            or any(str(path).startswith(f"{entry}/") for path in base_files)
        )
        optional_zip = re.fullmatch(r"python[0-9]+\.zip", entry) is not None
        if not present and not optional_zip:
            raise SealedLaunchError("base sys.path entry is absent from the base tree")

    raw_distributions = root["distributions"]
    if not isinstance(raw_distributions, list) or not raw_distributions:
        raise SealedLaunchError("runtime manifest has no distributions")
    distributions: list[dict[str, object]] = []
    ownership = {name: set() for name in names}
    for raw_distribution in raw_distributions:
        if not isinstance(raw_distribution, dict):
            raise SealedLaunchError("runtime distribution record must be an object")
        _exact_fields(
            raw_distribution,
            {"files", "name", "package_root", "version"},
            context="runtime distribution record",
        )
        name = _canonical_distribution_name(raw_distribution["name"])
        if name != raw_distribution["name"]:
            raise SealedLaunchError("distribution name is not canonical")
        version = raw_distribution["version"]
        package_root = _root_name(
            raw_distribution["package_root"], context="distribution package root"
        )
        if (
            not isinstance(version, str)
            or not version
            or version != version.strip()
            or package_root not in ownership
        ):
            raise SealedLaunchError("runtime distribution identity is invalid")
        raw_files = raw_distribution["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise SealedLaunchError("runtime distribution has no RECORD files")
        files = [
            _canonical_relative_path(item, context="distribution RECORD path") for item in raw_files
        ]
        if files != sorted(files) or len({item.casefold() for item in files}) != len(files):
            raise SealedLaunchError("distribution RECORD paths are not unique and sorted")
        overlap = ownership[package_root].intersection(files)
        if overlap:
            raise SealedLaunchError("runtime distributions claim the same installed file")
        ownership[package_root].update(files)
        distributions.append(
            {"files": files, "name": name, "package_root": package_root, "version": version}
        )
    distribution_names = [str(item["name"]) for item in distributions]
    if distribution_names != sorted(distribution_names) or len(set(distribution_names)) != len(
        distribution_names
    ):
        raise SealedLaunchError("runtime distributions are not unique and sorted")
    tree_by_name = {str(item["name"]): item for item in trees}
    for name in names:
        if ownership[name] != {str(item["path"]) for item in tree_by_name[name]["files"]}:
            raise SealedLaunchError("package tree differs from exact RECORD ownership")

    return {
        "base_sys_path": base_sys_path,
        "distributions": distributions,
        "file_sha256": _sha256_bytes(data),
        "git_executable": normalized_git_executable,
        "interpreter": normalized_interpreter,
        "machine": machine,
        "package_roots": package_roots,
        "python": python_record,
        "runtime_trees": trees,
    }


def _canonical_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SealedLaunchError("distribution has no canonical name")
    normalized = re.sub(r"[-_.]+", "-", value.strip()).lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None:
        raise SealedLaunchError("distribution name is invalid")
    return normalized


def _parse_package_root(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("--package-root must use name=absolute-path")
    try:
        canonical_name = _root_name(name, context="package root name")
    except SealedLaunchError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--package-root path must be absolute")
    return canonical_name, path


def _runtime_roots(
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    manifest: Mapping[str, object],
) -> tuple[Path, dict[str, Path], dict[str, str]]:
    base = _absolute_directory(base_runtime_root, context="base runtime root")
    expected_roots = {
        str(item["name"]): str(item["import_path"])
        for item in manifest["package_roots"]  # type: ignore[union-attr]
    }
    if set(package_roots) != set(expected_roots):
        raise SealedLaunchError("CLI package roots differ from the runtime manifest")
    packages = {
        name: _absolute_directory(package_roots[name], context=f"package root {name}")
        for name in sorted(package_roots)
    }
    all_roots = [(BASE_RUNTIME_ROOT_NAME, base), *packages.items()]
    for index, (left_name, left) in enumerate(all_roots):
        for right_name, right in all_roots[index + 1 :]:
            if left == right:
                raise SealedLaunchError(f"runtime roots alias: {left_name}, {right_name}")
            for outer, inner in ((left, right), (right, left)):
                try:
                    inner.relative_to(outer)
                except ValueError:
                    continue
                raise SealedLaunchError("runtime roots must not be nested")
    for name, relative in expected_roots.items():
        _safe_join(
            packages[name],
            relative,
            context=f"package root {name} import path",
            directory=True,
        )
    return base, packages, expected_roots


def _distribution_inventory(
    package_roots: Mapping[str, Path],
    import_paths: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    result: dict[str, dict[str, object]] = {}
    for root_name in sorted(package_roots):
        search_root = _safe_join(
            package_roots[root_name],
            import_paths[root_name],
            context=f"package root {root_name} import path",
            directory=True,
        )
        for distribution in importlib.metadata.distributions(path=[str(search_root)]):
            name = _canonical_distribution_name(distribution.metadata.get("Name"))
            if name in result:
                raise SealedLaunchError("staged package roots contain a duplicate distribution")
            raw_files = distribution.files
            if raw_files is None:
                raise SealedLaunchError(f"distribution {name} has no RECORD inventory")
            normalized: list[str] = []
            for raw_file in raw_files:
                candidate = Path(distribution.locate_file(raw_file))
                try:
                    resolved = candidate.resolve(strict=True)
                    relative = resolved.relative_to(package_roots[root_name]).as_posix()
                except (OSError, ValueError) as exc:
                    raise SealedLaunchError(
                        f"distribution {name} RECORD path escapes its package tree"
                    ) from exc
                _safe_join(
                    package_roots[root_name],
                    relative,
                    context=f"distribution {name} RECORD path",
                )
                normalized.append(
                    _canonical_relative_path(relative, context="normalized RECORD path")
                )
            normalized.sort()
            if not normalized or len({item.casefold() for item in normalized}) != len(normalized):
                raise SealedLaunchError(f"distribution {name} RECORD inventory is invalid")
            record_paths = [
                item for item in normalized if item.casefold().endswith(".dist-info/record")
            ]
            if len(record_paths) != 1 or not distribution.read_text("RECORD"):
                raise SealedLaunchError(f"distribution {name} must contain exactly one RECORD")
            result[name] = {
                "files": normalized,
                "name": name,
                "package_root": root_name,
                "version": str(distribution.version),
            }
    if not result:
        raise SealedLaunchError("staged runtime contains no distributions")
    return tuple(result[name] for name in sorted(result))


def _verify_runtime(
    manifest: Mapping[str, object],
    *,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    git_executable_path: Path,
    require_current_process: bool,
) -> tuple[Path, dict[str, Path], dict[str, str], Path, Path]:
    base, packages, import_paths = _runtime_roots(
        base_runtime_root,
        package_roots,
        manifest,
    )
    tree_roots = {BASE_RUNTIME_ROOT_NAME: base, **packages}
    for expected_tree in manifest["runtime_trees"]:  # type: ignore[union-attr]
        name = str(expected_tree["name"])
        actual = _tree_files(tree_roots[name], context=f"runtime tree {name}")
        if list(actual) != expected_tree["files"]:
            raise SealedLaunchError(f"runtime tree {name} differs from its frozen identity")
    if list(_distribution_inventory(packages, import_paths)) != manifest["distributions"]:
        raise SealedLaunchError("staged distribution identity differs from the runtime manifest")
    git_executable, git_record = _authenticated_git_executable(git_executable_path)
    if git_record != manifest["git_executable"]:
        raise SealedLaunchError("Git executable differs from the runtime manifest")
    interpreter = _safe_join(
        base,
        str(manifest["interpreter"]["relative_path"]),  # type: ignore[index]
        context="staged interpreter",
    )
    interpreter_record = _stable_file_record(
        interpreter,
        relative=str(manifest["interpreter"]["relative_path"]),  # type: ignore[index]
        context="staged interpreter",
    )
    if interpreter_record != {
        "path": manifest["interpreter"]["relative_path"],  # type: ignore[index]
        "sha256": manifest["interpreter"]["sha256"],  # type: ignore[index]
        "size_bytes": manifest["interpreter"]["size_bytes"],  # type: ignore[index]
    }:
        raise SealedLaunchError("staged interpreter identity drifted")
    if require_current_process:
        python = manifest["python"]
        machine = manifest["machine"]
        import struct

        if (
            platform.python_implementation() != python["implementation"]
            or platform.python_version() != python["version"]
            or sys.implementation.cache_tag != python["cache_tag"]
            or getattr(sys, "abiflags", "") != python["abi_flags"]
            or platform.system() != machine["system"]
            or f"{8 * struct.calcsize('P')}bit" != machine["architecture"]
            or platform.machine() != machine["machine"]
            or sys.byteorder != machine["byteorder"]
            or 8 * struct.calcsize("P") != machine["pointer_bits"]
            or Path(sys.executable).resolve(strict=True) != interpreter
        ):
            raise SealedLaunchError("point-used Python or machine identity drifted")
    return base, packages, import_paths, interpreter, git_executable


def _extract_runner_options(arguments: Sequence[str]) -> dict[str, str]:
    if arguments and arguments[0] in _CAPTURE_COMMANDS:
        command = arguments[0]
        required_options = (
            _CALIBRATION_CAPTURE_REQUIRED_RUNNER_OPTIONS
            if command == _CALIBRATION_CAPTURE_COMMAND
            else _STAGE_A_CAPTURE_REQUIRED_RUNNER_OPTIONS
        )
        remainder = list(arguments[1:])
        if len(remainder) != 2 * len(required_options):
            raise SealedLaunchError(
                "sealed calibration identity capture arguments are not an exact option profile"
            )
        capture_result: dict[str, str] = {}
        for index in range(0, len(remainder), 2):
            option = remainder[index]
            value = remainder[index + 1]
            if (
                option not in required_options
                or option in capture_result
                or value.startswith("--")
                or not value
            ):
                raise SealedLaunchError(
                    "sealed calibration identity capture arguments are mixed, duplicated, "
                    "or incomplete"
                )
            capture_result[option] = value
        if set(capture_result) != set(required_options):
            raise SealedLaunchError("sealed calibration identity capture inputs are incomplete")
        for option in (
            "--cache-root",
            "--capture-provenance-receipt-output",
            "--model-file-manifest",
            "--output",
            "--parquet-materialization-manifest",
            "--repository-root",
            "--repository-source-manifest",
            "--ruler-receipt-dir",
            "--runtime-manifest",
            "--stage-a-calibration-binding",
        ):
            if option in capture_result and not Path(capture_result[option]).is_absolute():
                raise SealedLaunchError(f"capture runner option must be absolute: {option}")
        return capture_result
    result: dict[str, str] = {}
    value_options = _REQUIRED_RUNNER_OPTIONS | _SMOKE_PREREQUISITE_OPTIONS
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option in _FORBIDDEN_RUNNER_OPTIONS or any(
            option.startswith(f"{forbidden}=") for forbidden in _FORBIDDEN_RUNNER_OPTIONS
        ):
            raise SealedLaunchError(f"legacy runner option is forbidden: {option}")
        if option in value_options:
            if option in result or index + 1 >= len(arguments):
                raise SealedLaunchError(f"runner option is duplicated or incomplete: {option}")
            raw_value = arguments[index + 1]
            if raw_value.startswith("--"):
                raise SealedLaunchError(f"runner option has no value: {option}")
            result[option] = raw_value
            index += 2
            continue
        index += 1
    missing = sorted(_REQUIRED_RUNNER_OPTIONS - set(result))
    if missing:
        raise SealedLaunchError(f"runner arguments omit required sealed inputs: {missing}")
    smoke_flag_count = sum(option == "--fisher-h1-smoke" for option in arguments)
    if smoke_flag_count > 1:
        raise SealedLaunchError("runner Fisher H=1 smoke flag is duplicated")
    supplied_prerequisites = _SMOKE_PREREQUISITE_OPTIONS & set(result)
    if smoke_flag_count == 1 and supplied_prerequisites:
        raise SealedLaunchError("smoke mode forbids prior Fisher H=1 smoke prerequisites")
    if smoke_flag_count == 0 and supplied_prerequisites != _SMOKE_PREREQUISITE_OPTIONS:
        raise SealedLaunchError(
            "full calibration requires both prior Fisher H=1 smoke prerequisite paths"
        )
    if not Path(result["--capture-provenance-receipt"]).is_absolute():
        raise SealedLaunchError("capture provenance receipt path must be absolute")
    return result


def _verify_capture_provenance_envelope(runner_options: Mapping[str, str]) -> str:
    receipt_bytes = _stable_file_bytes(
        Path(runner_options["--capture-provenance-receipt"]),
        context="capture provenance receipt",
    )
    actual = _sha256_bytes(receipt_bytes)
    expected = _sha256(
        runner_options["--expected-capture-provenance-receipt-sha256"],
        context="expected capture provenance receipt SHA-256",
    )
    if actual != expected:
        raise SealedLaunchError("capture provenance receipt differs from its explicit SHA-256")
    root = _strict_json(receipt_bytes, context="capture provenance receipt")
    _exact_fields(
        root,
        {
            "artifact_kind",
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
        context="capture provenance receipt",
    )
    if _canonical_json_bytes(root) != receipt_bytes:
        raise SealedLaunchError("capture provenance receipt is not canonical JSON")
    if (
        root["artifact_kind"] != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
        or type(root["capture_version"]) is not int
        or root["capture_version"] != CALIBRATION_IDENTITY_CAPTURE_VERSION
        or root["runner_revision"] != RUNNER_REVISION
        or root["phase"] != "calibration"
        or root["publication_contract"]
        != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT
        or root["status"] != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS
        or root["source_commit"] != runner_options["--source-commit"]
    ):
        raise SealedLaunchError("capture provenance receipt finalized envelope drifted")
    return actual


def _verify_fisher_smoke_prerequisite_files(runner_options: Mapping[str, str]) -> None:
    capture_provenance_sha256 = _verify_capture_provenance_envelope(runner_options)
    if "--prior-fisher-h1-smoke-report" not in runner_options:
        return
    try:
        marker = Path(runner_options["--prior-fisher-h1-smoke-complete-marker"]).read_bytes()
        report_bytes = Path(runner_options["--prior-fisher-h1-smoke-report"]).read_bytes()
    except OSError as exc:
        raise SealedLaunchError("prior Fisher H=1 smoke prerequisite is unavailable") from exc
    if marker != FISHER_SMOKE_COMPLETE_BYTES:
        raise SealedLaunchError("prior Fisher H=1 smoke completion marker drifted")
    root = _strict_json(report_bytes, context="prior Fisher H=1 smoke report")
    _exact_fields(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="prior Fisher H=1 smoke report",
    )
    if _canonical_json_bytes(root) != report_bytes:
        raise SealedLaunchError("prior Fisher H=1 smoke report is not canonical JSON")
    evidence = root["evidence"]
    if (
        root["artifact_kind"] != RUN_REPORT_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != RUN_REPORT_SCHEMA
        or not isinstance(evidence, dict)
        or evidence.get("status") != "fisher_h1_smoke_passed"
        or evidence.get("runner_revision") != RUNNER_REVISION
        or evidence.get("prerequisites")
        != {
            "capture_provenance_receipt_file_sha256": capture_provenance_sha256,
            "fisher_h1_smoke_report_file_sha256": None,
        }
        or _sha256(
            root["canonical_evidence_sha256"],
            context="prior Fisher H=1 smoke evidence SHA-256",
        )
        != _sha256_bytes(_canonical_json_bytes(evidence))
    ):
        raise SealedLaunchError("prior Fisher H=1 smoke report authentication failed")


def _parse_identity(data: bytes) -> dict[str, str]:
    root = _strict_json(data, context="frozen identity")
    _exact_fields(root, {"canonical_evidence_sha256", "evidence"}, context="frozen identity")
    if _canonical_json_bytes(root) != data:
        raise SealedLaunchError("frozen identity is not canonical JSON")
    evidence = root["evidence"]
    if not isinstance(evidence, dict):
        raise SealedLaunchError("frozen identity evidence is missing")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != IDENTITY_SCHEMA
        or evidence.get("status") != "frozen"
        or evidence.get("phase") != "calibration"
        or evidence.get("identity_only") is not True
        or evidence.get("promotion_required") is not False
    ):
        raise SealedLaunchError("frozen identity state or schema drifted")
    claimed = _sha256(root["canonical_evidence_sha256"], context="identity evidence SHA-256")
    if claimed != _sha256_bytes(_canonical_json_bytes(evidence)):
        raise SealedLaunchError("frozen identity evidence hash drifted")
    bindings = evidence.get("execution_bindings")
    if not isinstance(bindings, dict):
        raise SealedLaunchError("frozen identity execution bindings are missing")
    _exact_fields(bindings, set(_BOUND_ARTIFACT_OPTIONS), context="identity execution bindings")
    return {
        name: _sha256(bindings[name], context=f"identity binding {name}")
        for name in sorted(bindings)
    }


def _parse_source_manifest(data: bytes) -> dict[str, object]:
    root = _strict_json(data, context="source manifest")
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
        context="source manifest",
    )
    if _pretty_json_bytes(root) != data:
        raise SealedLaunchError("source manifest is not canonical JSON")
    if root["schema"] != "recurquant.experiment013.source-manifest.v2":
        raise SealedLaunchError("source manifest schema drifted")
    if root["profile"] != "experiment-013-static-q468-frozen-source-v2":
        raise SealedLaunchError("source manifest profile drifted")
    if root["object_format"] != "sha1" or not isinstance(root["source_commit"], str):
        raise SealedLaunchError("source manifest Git identity drifted")
    if _SHA1_RE.fullmatch(str(root["source_commit"])) is None:
        raise SealedLaunchError("source manifest commit is invalid")
    raw_git = root["git_executable"]
    if not isinstance(raw_git, dict):
        raise SealedLaunchError("source manifest Git executable record is missing")
    _exact_fields(raw_git, {"sha256", "size_bytes"}, context="source Git executable")
    git_executable = {
        "sha256": _sha256(raw_git["sha256"], context="source Git executable SHA-256"),
        "size_bytes": _positive_int(raw_git["size_bytes"], context="source Git executable size"),
    }
    payload = dict(root)
    claimed = _sha256(payload.pop("canonical_manifest_sha256"), context="source self-hash")
    if claimed != _sha256_bytes(_pretty_json_bytes(payload)):
        raise SealedLaunchError("source manifest self-hash drifted")
    raw_paths = root["paths"]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise SealedLaunchError("source manifest has no paths")
    paths: list[dict[str, object]] = []
    for raw_entry in raw_paths:
        if not isinstance(raw_entry, dict):
            raise SealedLaunchError("source path record must be an object")
        _exact_fields(
            raw_entry,
            {"git_blob_oid", "index_blob_oid", "mode", "path", "raw_sha256", "worktree_blob_oid"},
            context="source path record",
        )
        relative = _canonical_relative_path(raw_entry["path"], context="source path")
        for oid_field in ("git_blob_oid", "index_blob_oid", "worktree_blob_oid"):
            if (
                not isinstance(raw_entry[oid_field], str)
                or _SHA1_RE.fullmatch(raw_entry[oid_field]) is None
            ):
                raise SealedLaunchError("source Git object identity is invalid")
        git_object_ids = {
            raw_entry[name] for name in ("git_blob_oid", "index_blob_oid", "worktree_blob_oid")
        }
        if len(git_object_ids) != 1:
            raise SealedLaunchError("source Git object identities disagree")
        if raw_entry["mode"] not in {"100644", "100755"}:
            raise SealedLaunchError("source file mode is invalid")
        paths.append(
            {
                "path": relative,
                "raw_sha256": _sha256(raw_entry["raw_sha256"], context="source raw SHA-256"),
            }
        )
    rendered = [str(item["path"]) for item in paths]
    if rendered != sorted(rendered) or len({item.casefold() for item in rendered}) != len(rendered):
        raise SealedLaunchError("source path inventory is not unique and sorted")
    if RUNNER_SOURCE_PATH not in rendered:
        raise SealedLaunchError("source manifest omits the calibration runner")
    return {
        "file_sha256": _sha256_bytes(data),
        "git_executable": git_executable,
        "paths": paths,
        "source_commit": str(root["source_commit"]),
    }


def _validate_capture_provenance_candidate(
    payload: bytes,
    *,
    bindings: Mapping[str, str],
    identity_input_file_sha256: str,
    runtime_manifest: Mapping[str, object],
    source_manifest: Mapping[str, object],
    phase: str = "calibration",
    stage_a_binding_envelope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = _strict_json(payload, context="capture provenance candidate")
    expected_fields = {
        "artifact_kind",
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
    }
    if phase == "stage_a":
        expected_fields.update(
            {
                "calibration_authorization_file_sha256",
                "calibration_binding_file_sha256",
            }
        )
    _exact_fields(root, expected_fields, context="capture provenance candidate")
    if _canonical_json_bytes(root) != payload:
        raise SealedLaunchError("capture provenance candidate is not canonical JSON")
    if (
        root["artifact_kind"]
        != (
            CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND
            if phase == "calibration"
            else STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND
        )
        or type(root["capture_version"]) is not int
        or root["capture_version"] != CALIBRATION_IDENTITY_CAPTURE_VERSION
        or type(root["schema_version"]) is not int
        or root["schema_version"]
        != (
            CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
            if phase == "calibration"
            else STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
        )
        or root["runner_revision"] != RUNNER_REVISION
        or root["phase"] != phase
        or root["publication_contract"]
        != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT
        or root["status"] != CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS
    ):
        raise SealedLaunchError("capture provenance candidate identity drifted")
    if phase == "stage_a":
        if stage_a_binding_envelope is None:
            raise SealedLaunchError("Stage-A capture provenance lacks a verified binding envelope")
        if (
            _sha256(
                root["calibration_binding_file_sha256"],
                context="capture provenance Stage-A binding SHA-256",
            )
            != stage_a_binding_envelope["file_sha256"]
        ):
            raise SealedLaunchError("capture provenance Stage-A binding drifted")
        if (
            _sha256(
                root["calibration_authorization_file_sha256"],
                context="capture provenance Stage-A authorization SHA-256",
            )
            != stage_a_binding_envelope["authorization_file_sha256"]
        ):
            raise SealedLaunchError("capture provenance Stage-A authorization drifted")
    elif stage_a_binding_envelope is not None:
        raise SealedLaunchError("calibration capture cannot carry a Stage-A binding envelope")
    if root["excluded_runtime_modules"] != list(CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES):
        raise SealedLaunchError("capture provenance candidate exclusion policy drifted")
    if root["source_commit"] != source_manifest["source_commit"]:
        raise SealedLaunchError("capture provenance candidate source commit drifted")
    if _sha256(
        root["identity_input_file_sha256"],
        context="capture provenance identity input SHA-256",
    ) != _sha256(
        identity_input_file_sha256,
        context="published identity input SHA-256",
    ):
        raise SealedLaunchError("capture provenance candidate binds a different identity input")

    raw_bindings = root["execution_bindings"]
    if not isinstance(raw_bindings, dict):
        raise SealedLaunchError("capture provenance candidate bindings are missing")
    _exact_fields(
        raw_bindings,
        set(_BOUND_ARTIFACT_OPTIONS),
        context="capture provenance candidate bindings",
    )
    normalized_bindings = {
        name: _sha256(raw_bindings[name], context=f"capture provenance binding {name}")
        for name in sorted(raw_bindings)
    }
    if normalized_bindings != dict(bindings):
        raise SealedLaunchError("capture provenance candidate bindings drifted")

    capture_source = root["capture_source"]
    if not isinstance(capture_source, dict):
        raise SealedLaunchError("capture provenance candidate source record is missing")
    _exact_fields(
        capture_source,
        {"path", "sha256"},
        context="capture provenance candidate source record",
    )
    source_entries = {
        str(item["path"]): str(item["raw_sha256"])
        for item in source_manifest["paths"]  # type: ignore[union-attr]
    }
    if capture_source["path"] != CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH or _sha256(
        capture_source["sha256"],
        context="capture provenance source SHA-256",
    ) != source_entries.get(CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH):
        raise SealedLaunchError("capture provenance candidate source drifted")

    raw_origins = root["critical_module_origins"]
    if not isinstance(raw_origins, list):
        raise SealedLaunchError("capture provenance candidate origins are missing")
    expected_modules = sorted(CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS)
    if [item.get("module") if isinstance(item, dict) else None for item in raw_origins] != (
        expected_modules
    ):
        raise SealedLaunchError("capture provenance candidate origin inventory drifted")
    distributions = {
        str(item["name"]): item
        for item in runtime_manifest["distributions"]  # type: ignore[union-attr]
    }
    trees = {
        str(item["name"]): {
            str(record["path"]): record
            for record in item["files"]  # type: ignore[index]
        }
        for item in runtime_manifest["runtime_trees"]  # type: ignore[union-attr]
    }
    import_paths = {
        str(item["name"]): str(item["import_path"])
        for item in runtime_manifest["package_roots"]  # type: ignore[union-attr]
    }
    for item in raw_origins:
        assert isinstance(item, dict)
        _exact_fields(
            item,
            {
                "distribution",
                "module",
                "package_root",
                "relative_path",
                "sha256",
                "size_bytes",
                "version",
            },
            context="capture provenance candidate origin",
        )
        module_name = str(item["module"])
        distribution_name = CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS[module_name]
        distribution = distributions.get(distribution_name)
        if distribution is None or item["distribution"] != distribution_name:
            raise SealedLaunchError("capture provenance candidate distribution drifted")
        package_root = str(item["package_root"])
        relative_path = _canonical_relative_path(
            item["relative_path"],
            context=f"capture provenance {module_name} relative path",
        )
        import_path = import_paths.get(package_root)
        if (
            package_root != distribution["package_root"]
            or item["version"] != distribution["version"]
            or relative_path not in distribution["files"]
            or import_path is None
        ):
            raise SealedLaunchError("capture provenance candidate runtime identity drifted")
        try:
            module_relative = PurePosixPath(relative_path).relative_to(PurePosixPath(import_path))
        except ValueError as exc:
            raise SealedLaunchError(
                "capture provenance candidate origin is outside its import root"
            ) from exc
        if not module_relative.parts or (
            module_relative != PurePosixPath("six.py")
            if module_name == "six"
            else module_relative.parts[0] != module_name
        ):
            raise SealedLaunchError("capture provenance candidate module is shadowed")
        runtime_file = trees.get(package_root, {}).get(relative_path)
        if runtime_file != {
            "path": relative_path,
            "sha256": _sha256(
                item["sha256"],
                context=f"capture provenance {module_name} file SHA-256",
            ),
            "size_bytes": _nonnegative_int(
                item["size_bytes"],
                context=f"capture provenance {module_name} file size",
            ),
        }:
            raise SealedLaunchError("capture provenance candidate runtime file drifted")
    return root


def _verify_source(source_manifest: Mapping[str, object], repository_root: Path) -> Path:
    root = _absolute_directory(repository_root, context="repository root")
    runner_path: Path | None = None
    for entry in source_manifest["paths"]:  # type: ignore[union-attr]
        relative = str(entry["path"])
        path = _safe_join(root, relative, context=f"source file {relative}")
        actual = _stable_file_record(path, relative=relative, context=f"source file {relative}")
        if actual["sha256"] != entry["raw_sha256"]:
            raise SealedLaunchError(f"source bytes drifted: {relative}")
        if relative == RUNNER_SOURCE_PATH:
            runner_path = path
    assert runner_path is not None
    return runner_path


def _verify_bound_artifacts(
    runner_options: Mapping[str, str],
    *,
    runtime_manifest_path: Path,
) -> tuple[dict[str, str], dict[str, object], Path]:
    capture_profile = "--capture-provenance-receipt-output" in runner_options
    if capture_profile:
        bindings: dict[str, str] = {}
        for binding, option in _BOUND_ARTIFACT_OPTIONS.items():
            artifact_path = Path(runner_options[option])
            artifact_bytes = _stable_file_bytes(
                artifact_path,
                context=f"capture bound artifact {option}",
            )
            actual = _sha256_bytes(artifact_bytes)
            expected_option = _CAPTURE_EXPECTED_BOUND_DIGEST_OPTIONS[binding]
            expected = _sha256(
                runner_options[expected_option],
                context=f"capture runner option {expected_option}",
            )
            if actual != expected:
                raise SealedLaunchError(f"capture artifact digest mismatch: {option}")
            bindings[binding] = actual
        try:
            if runtime_manifest_path.resolve(strict=True) != Path(
                runner_options["--runtime-manifest"]
            ).resolve(strict=True):
                raise SealedLaunchError("host and capture runtime-manifest paths differ")
        except OSError as exc:
            raise SealedLaunchError("capture runtime manifest path is unavailable") from exc
        source_bytes = _stable_file_bytes(
            Path(runner_options["--repository-source-manifest"]),
            context="capture repository source manifest",
        )
        source_manifest = _parse_source_manifest(source_bytes)
        if runner_options["--source-commit"] != source_manifest["source_commit"]:
            raise SealedLaunchError("capture source commit differs from source-manifest H0")
        runner_path = _verify_source(
            source_manifest,
            Path(runner_options["--repository-root"]),
        )
        return bindings, source_manifest, runner_path
    identity_path = Path(runner_options["--frozen-identity"])
    identity_bytes = identity_path.read_bytes()
    bindings = _parse_identity(identity_bytes)
    for binding, option in _BOUND_ARTIFACT_OPTIONS.items():
        artifact_path = Path(runner_options[option])
        try:
            artifact_bytes = artifact_path.read_bytes()
        except OSError as exc:
            raise SealedLaunchError(f"bound artifact is unavailable: {option}") from exc
        if _sha256_bytes(artifact_bytes) != bindings[binding]:
            raise SealedLaunchError(f"identity binding mismatch: {option}")
    for binding, option in _EXPECTED_BOUND_DIGEST_OPTIONS.items():
        expected = _sha256(runner_options[option], context=f"runner option {option}")
        if expected != bindings[binding]:
            raise SealedLaunchError(f"runner digest binding mismatch: {option}")
    try:
        if runtime_manifest_path.resolve(strict=True) != Path(
            runner_options["--runtime-manifest"]
        ).resolve(strict=True):
            raise SealedLaunchError("host and runner runtime-manifest paths differ")
    except OSError as exc:
        raise SealedLaunchError("runtime manifest path is unavailable") from exc
    source_bytes = Path(runner_options["--repository-source-manifest"]).read_bytes()
    source_manifest = _parse_source_manifest(source_bytes)
    if runner_options["--source-commit"] != source_manifest["source_commit"]:
        raise SealedLaunchError("runner source commit differs from source-manifest H0")
    runner_path = _verify_source(
        source_manifest,
        Path(runner_options["--repository-root"]),
    )
    _verify_fisher_smoke_prerequisite_files(runner_options)
    return bindings, source_manifest, runner_path


def _verify_stage_a_capture_binding(
    runner_options: Mapping[str, str],
    *,
    execution_bindings: Mapping[str, str],
    source_commit: str,
) -> tuple[bytes, dict[str, object]]:
    if "--stage-a-calibration-binding" not in runner_options:
        raise SealedLaunchError("Stage-A capture binding is missing")
    binding_bytes = _stable_file_bytes(
        Path(runner_options["--stage-a-calibration-binding"]),
        context="Stage-A calibration binding",
    )
    envelope = _parse_stage_a_calibration_binding_envelope(
        binding_bytes,
        expected_file_sha256=runner_options["--expected-stage-a-calibration-binding-sha256"],
    )
    if envelope["execution_bindings"] != dict(execution_bindings):
        raise SealedLaunchError(
            "Stage-A calibration authorization execution bindings differ from capture"
        )
    if envelope["source_commit"] != source_commit:
        raise SealedLaunchError("Stage-A calibration authorization source commit differs from H0")
    return binding_bytes, envelope


def _verify_empty_pycache(path: Path) -> Path:
    root = _absolute_directory(path, context="pycache prefix")
    try:
        if any(os.scandir(root)):
            raise SealedLaunchError("pycache prefix is not empty")
    except OSError as exc:
        raise SealedLaunchError("cannot enumerate pycache prefix") from exc
    return root


def _verify_empty_scratch(path: Path) -> Path:
    root = _absolute_directory(path, context="sealed scratch directory")
    try:
        if any(os.scandir(root)):
            raise SealedLaunchError("sealed scratch directory is not empty")
    except OSError as exc:
        raise SealedLaunchError("cannot enumerate sealed scratch directory") from exc
    return root


def _verified_dataset_cache_root(
    path: Path,
    *,
    runtime_roots: Sequence[Path],
) -> Path:
    """Authenticate the explicit writable data cache and keep it off runtime trees."""

    raw = Path(path)
    _non_link_directory_identity_chain(raw, context="dataset cache root")
    root = _absolute_directory(raw, context="dataset cache root")
    if root == Path(root.anchor):
        raise SealedLaunchError("dataset cache root cannot be a filesystem root")
    for runtime_root in runtime_roots:
        authenticated_runtime_root = _absolute_directory(
            runtime_root,
            context="authenticated runtime root",
        )
        if root == authenticated_runtime_root:
            raise SealedLaunchError("dataset cache root overlaps an authenticated runtime root")
        try:
            root.relative_to(authenticated_runtime_root)
        except ValueError:
            pass
        else:
            raise SealedLaunchError("dataset cache root overlaps an authenticated runtime root")
        try:
            authenticated_runtime_root.relative_to(root)
        except ValueError:
            pass
        else:
            raise SealedLaunchError("dataset cache root overlaps an authenticated runtime root")
    return root


def _verified_capture_hub_cache_root(dataset_cache_root: Path) -> Path:
    """Authenticate the exact persistent Hub endpoint used only by capture children."""

    cache = _verified_dataset_cache_root(dataset_cache_root, runtime_roots=())
    candidate = cache / "hub"
    _non_link_directory_identity_chain(candidate, context="capture Hub cache root")
    root = _absolute_directory(candidate, context="capture Hub cache root")
    if root.parent != cache:
        raise SealedLaunchError("capture Hub cache root escaped the dataset cache root")
    return root


def _prepare_capture_hub_cache_root(dataset_cache_root: Path) -> Path:
    """Create the fixed capture Hub endpoint without accepting a creation race."""

    cache = _verified_dataset_cache_root(dataset_cache_root, runtime_roots=())
    candidate = cache / "hub"
    if not os.path.lexists(candidate):
        try:
            os.mkdir(candidate, mode=0o700)
        except FileExistsError as exc:
            raise SealedLaunchError(
                "capture Hub cache root appeared during creation"
            ) from exc
        except OSError as exc:
            raise SealedLaunchError("cannot create capture Hub cache root") from exc
    return _verified_capture_hub_cache_root(cache)


def _verify_capture_hub_cache_identity(
    dataset_cache_root: Path,
    *,
    expected_identity: tuple[tuple[str, int, int, int], ...],
) -> Path:
    root = _verified_capture_hub_cache_root(dataset_cache_root)
    if (
        _non_link_directory_identity_chain(root, context="capture Hub cache root")
        != expected_identity
    ):
        raise SealedLaunchError("capture Hub cache root identity changed")
    return root


def _non_link_directory_identity_chain(
    path: Path,
    *,
    context: str,
) -> tuple[tuple[str, int, int, int], ...]:
    """Return the ordered identity of every existing component without following links."""

    raw = Path(path)
    if not raw.is_absolute():
        raise SealedLaunchError(f"{context} must be an absolute path")
    candidate = Path(os.path.abspath(raw))
    current = Path(candidate.anchor)
    components = [current]
    for part in candidate.parts[1:]:
        current /= part
        components.append(current)
    identities: list[tuple[str, int, int, int]] = []
    for component in components:
        try:
            status = component.lstat()
        except OSError as exc:
            raise SealedLaunchError(f"{context} component is unavailable") from exc
        if component.is_symlink() or bool(
            getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
        ):
            raise SealedLaunchError(f"{context} traverses a link or reparse point")
        if not stat.S_ISDIR(status.st_mode):
            raise SealedLaunchError(f"{context} component is not a directory")
        identities.append(
            (
                os.path.normcase(str(component)),
                int(status.st_dev),
                int(status.st_ino),
                stat.S_IFMT(status.st_mode),
            )
        )
    return tuple(identities)


def _temporary_directory_identity(path: Path, *, context: str) -> tuple[int, int, int]:
    root = _absolute_directory(path, context=context)
    try:
        status = root.lstat()
    except OSError as exc:
        raise SealedLaunchError(f"cannot identify {context}") from exc
    return (int(status.st_dev), int(status.st_ino), stat.S_IFMT(status.st_mode))


def _assert_owned_temporary_tree_has_no_reparse(path: Path, *, context: str) -> None:
    stack = [path]
    while stack:
        directory = stack.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise SealedLaunchError(f"cannot enumerate {context} during cleanup") from exc
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SealedLaunchError(f"cannot inspect {context} during cleanup") from exc
            if entry.is_symlink() or bool(
                getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise SealedLaunchError(f"{context} contains a link or reparse point")
            if stat.S_ISDIR(status.st_mode):
                stack.append(Path(entry.path))
            elif not stat.S_ISREG(status.st_mode):
                raise SealedLaunchError(f"{context} contains a non-regular path")


def _cleanup_owned_temporary_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int, int],
    context: str,
) -> None:
    """Remove one launcher-owned tree without following redirected content."""

    root = Path(os.path.abspath(path))
    if not os.path.lexists(root):
        return
    observed_identity = _temporary_directory_identity(root, context=context)
    if observed_identity != expected_identity:
        raise SealedLaunchError(f"{context} identity changed before cleanup")
    _assert_owned_temporary_tree_has_no_reparse(root, context=context)
    try:
        shutil.rmtree(root, ignore_errors=False)
    except OSError as exc:
        raise SealedLaunchError(f"cannot remove {context}") from exc
    if os.path.lexists(root):
        raise SealedLaunchError(f"{context} survived cleanup")


def _failure_summary(failures: Sequence[tuple[str, BaseException]]) -> str:
    return "; ".join(
        f"{context}: {error.__class__.__name__}: {error}" for context, error in failures
    )


def _postcondition_error(failures: Sequence[tuple[str, BaseException]]) -> SealedLaunchError:
    if not failures:
        raise ValueError("postcondition failure inventory cannot be empty")
    error = SealedLaunchError(f"sealed child postcondition failed: {_failure_summary(failures)}")
    error.__cause__ = failures[0][1]
    return error


def _surface_secondary_failures(
    failures: Sequence[tuple[str, BaseException]],
    *,
    primary_error: BaseException | None,
    child_returncode: int | None,
) -> None:
    if not failures:
        return
    message = f"sealed launcher secondary failure: {_failure_summary(failures)}"
    if primary_error is not None:
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(message)
        else:  # pragma: no cover - supported Python versions expose add_note
            print(message, file=sys.stderr, flush=True)
        return
    if child_returncode is not None and child_returncode != 0:
        print(
            f"{message}; preserving child return code {child_returncode}",
            file=sys.stderr,
            flush=True,
        )
        return
    raise _postcondition_error(failures)


def _sealed_environment(
    *,
    scratch_directory: Path,
    dataset_cache_root: Path,
    capture_profile: bool = False,
) -> dict[str, str]:
    scratch = _absolute_directory(scratch_directory, context="sealed scratch directory")
    cache = _verified_dataset_cache_root(dataset_cache_root, runtime_roots=())
    private_home = scratch / "private-home"
    xdg_cache = scratch / "xdg-cache"
    hf_home = scratch / "huggingface"
    hf_hub_cache = (
        _verified_capture_hub_cache_root(cache) if capture_profile else hf_home / "hub"
    )
    hf_assets_cache = hf_home / "assets"
    hf_xet_cache = hf_home / "xet"
    hf_modules_cache = hf_home / "modules"
    hf_token_path = hf_home / "token"
    transformers_cache = scratch / "transformers"
    torch_home = scratch / "torch"
    torch_kernel_cache = torch_home / "kernels"
    torch_extensions = torch_home / "extensions"
    torch_inductor = torch_home / "inductor"
    triton_cache = torch_home / "triton"
    datasets_cache = cache / "datasets"
    datasets_downloads = datasets_cache / "downloads"
    datasets_extracted = datasets_downloads / "extracted"
    inherited = {key.upper(): (key, value) for key, value in os.environ.items()}
    environment = {
        inherited[name][0]: inherited[name][1]
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_ARCHITEW6432",
        )
        if name in inherited
    }
    environment.update(
        {
            "LANG": "C",
            "LC_ALL": "C",
            "HOME": str(private_home),
            "USERPROFILE": str(private_home),
            "TEMP": str(scratch),
            "TMP": str(scratch),
            "TZ": "UTC",
            "XDG_CACHE_HOME": str(xdg_cache),
            "HF_HOME": str(hf_home),
            "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
            "HF_HUB_CACHE": str(hf_hub_cache),
            "HUGGINGFACE_ASSETS_CACHE": str(hf_assets_cache),
            "HF_ASSETS_CACHE": str(hf_assets_cache),
            "HF_XET_CACHE": str(hf_xet_cache),
            "HF_MODULES_CACHE": str(hf_modules_cache),
            "HF_TOKEN_PATH": str(hf_token_path),
            "TRANSFORMERS_CACHE": str(transformers_cache),
            "TORCH_HOME": str(torch_home),
            "PYTORCH_KERNEL_CACHE_PATH": str(torch_kernel_cache),
            "TORCH_EXTENSIONS_DIR": str(torch_extensions),
            "TORCHINDUCTOR_CACHE_DIR": str(torch_inductor),
            "TRITON_CACHE_DIR": str(triton_cache),
            "HF_DATASETS_CACHE": str(datasets_cache),
            "HF_DATASETS_DOWNLOADED_DATASETS_PATH": str(datasets_downloads),
            "HF_DATASETS_EXTRACTED_DATASETS_PATH": str(datasets_extracted),
            "DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_UPDATE_CHECK": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    return environment


def _authenticated_stdin_loader(payload: bytes) -> str:
    """Return a small ``-c`` loader bound to one exact stdin payload."""
    digest = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    return (
        "import hashlib as _h,sys as _s\n"
        f"_p=_s.stdin.buffer.read({size + 1})\n"
        f"if len(_p)!={size} or _h.sha256(_p).hexdigest()!='{digest}':"
        " raise RuntimeError('sealed bootstrap stdin authentication failed')\n"
        "exec(compile(_p,'<recurquant-sealed-bootstrap>','exec',dont_inherit=True))"
    )


def _sealed_argv(
    *,
    interpreter: Path,
    runtime_manifest: Path,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    git_executable: Path,
    pycache_prefix: Path,
    scratch_directory: Path,
    runner_arguments: Sequence[str],
) -> list[str]:
    serialized_roots = json.dumps(
        {name: str(package_roots[name]) for name in sorted(package_roots)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        str(interpreter),
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        "-X",
        "utf8",
        "-c",
        SEALED_STDIN_LOADER,
        str(runtime_manifest),
        str(base_runtime_root),
        serialized_roots,
        str(pycache_prefix),
        str(git_executable),
        str(scratch_directory),
        *runner_arguments,
    ]


# This is intentionally standalone.  The child starts with no repository or
# package path and cannot import this host module safely.  Keep its verification
# semantics aligned with the host functions above and test both boundaries.
SEALED_BOOTSTRAP: Final = r"""
import sys as _s
_sensitive = {
    "site", "_virtualenv", "recurquant",
    "recurquant.experiment013_calibration_api",
    "recurquant.experiment013_qwen35_adapter",
    "recurquant.experiment013_source",
    "recurquant_experiment013_calibration_identity_capture",
    "_recurquant_experiment013_calibration_runner_for_capture",
    "recurquant_experiment013_identity_resolver",
    "datasets", "fsspec", "huggingface_hub", "numpy", "pkg_resources",
    "pyarrow", "setuptools", "six", "tokenizers", "torch", "transformers",
    "_recurquant_experiment013_sealed_runner",
}
if _sensitive.intersection(_s.modules):
    raise RuntimeError("sealed bootstrap found a preloaded sensitive module")
if (
    _s.flags.isolated != 1
    or _s.flags.ignore_environment != 1
    or _s.flags.no_user_site != 1
    or _s.flags.no_site != 1
    or _s.flags.dont_write_bytecode != 1
    or _s.flags.safe_path is not True
    or _s.flags.utf8_mode != 1
):
    raise RuntimeError("sealed bootstrap startup flags drifted")
if (
    set(_s._xoptions) != {"pycache_prefix", "utf8"}
    or _s._xoptions.get("utf8") is not True
    or not isinstance(_s._xoptions.get("pycache_prefix"), str)
):
    raise RuntimeError("sealed bootstrap -X options drifted")

import hashlib as _h
import importlib.metadata as _md
import json as _j
import os as _o
import pathlib as _p
import platform as _platform
import re as _re
import stat as _stat
import struct as _struct
import types as _types

_rp = 0x400
_sha = _re.compile(r"[0-9a-f]{64}")
_sha1 = _re.compile(r"[0-9a-f]{40}")
_root_re = _re.compile(r"[a-z][a-z0-9-]{0,63}")
_bad_suffix = {".egg-link", ".pth", ".pyc", ".pyo", "._pth"}
_bad_dir = {"__pycache__"}
_bad_name = {"pyvenv.cfg", "sitecustomize.py", "usercustomize.py"}
_reserved = {
    "aux", "clock$", "con", "nul", "prn",
    *{"com" + str(i) for i in range(1, 10)},
    *{"lpt" + str(i) for i in range(1, 10)},
}
_policy = {
    "bootstrap_mode": "stdlib-only-exact-runner-and-capture-v3",
    "cache_confinement_mode": "private-scratch-plus-explicit-dataset-and-capture-hub-root-v2",
    "child_cwd_mode": "authenticated-launcher-owned-scratch-v1",
    "dont_write_bytecode": 1,
    "executable_custody_mode": "platform-held-launch-handles-v1",
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "package_path_mode": "authenticated-record-only-roots-v1",
    "pycache_mode": "new-verified-empty-prefix-v1",
    "safe_path": True,
    "site_loaded": False,
    "sys_path_mode": "staged-base-then-authenticated-packages-v1",
    "utf8_mode": 1,
    "virtualenv_hook_loaded": False,
}
_binding_options = {
    "calibration_runtime_manifest_file_sha256": "--runtime-manifest",
    "model_file_manifest_file_sha256": "--model-file-manifest",
    "parquet_materialization_manifest_file_sha256": "--parquet-materialization-manifest",
    "repository_source_manifest_file_sha256": "--repository-source-manifest",
}
_expected_digest_options = {
    "calibration_runtime_manifest_file_sha256": "--expected-runtime-manifest-sha256",
    "model_file_manifest_file_sha256": "--expected-model-file-manifest-sha256",
    "parquet_materialization_manifest_file_sha256":
        "--expected-parquet-materialization-manifest-sha256",
}
_capture_expected_digest_options = dict(_expected_digest_options)
_capture_expected_digest_options["repository_source_manifest_file_sha256"] = (
    "--expected-repository-source-manifest-sha256"
)
_calibration_capture_command = "capture-calibration-identity"
_stage_a_capture_command = "capture-stage-a-identity"
_capture_commands = {_calibration_capture_command, _stage_a_capture_command}
_calibration_capture_required = {
    "--cache-root", "--capture-provenance-receipt-output", "--output",
    "--repository-root", "--ruler-receipt-dir", "--source-commit",
    *_binding_options.values(), *_capture_expected_digest_options.values(),
}
_stage_a_capture_required = _calibration_capture_required | {
    "--expected-stage-a-calibration-binding-sha256",
    "--stage-a-calibration-binding",
}
_smoke_options = {
    "--prior-fisher-h1-smoke-report",
    "--prior-fisher-h1-smoke-complete-marker",
}
_forbidden_runner_options = {"--ruler-root"}
_smoke_marker = b"recurquant-experiment013-fisher-h1-smoke-complete-v1\n"

def _fail(message):
    raise RuntimeError(message)

def _surface_postcondition_failures(primary, result, failures):
    if not failures:
        return
    details = "; ".join(
        label + ": " + error.__class__.__name__ + ": " + str(error)
        for label, error in failures
    )
    message = "sealed bootstrap secondary postcondition failure: " + details
    if primary is not None:
        primary.add_note(message)
    elif type(result) is int and result != 0:
        _s.stderr.write(
            message + "; preserving sealed_main return code " + str(result) + "\n"
        )
        _s.stderr.flush()
    else:
        _fail(message)

def _canonical(value):
    return (_j.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                     separators=(",", ":")) + "\n").encode("utf-8")

def _pretty(value):
    return (_j.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

def _json(data, context):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                _fail(context + " contains a duplicate JSON key")
            out[key] = value
        return out
    def constant(value):
        _fail(context + " contains a non-finite JSON constant")
    try:
        value = _j.loads(data.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=constant)
    except (UnicodeDecodeError, _j.JSONDecodeError) as error:
        raise RuntimeError(context + " is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        _fail(context + " must be an object")
    return value

def _fields(value, expected, context):
    if set(value) != set(expected):
        _fail(context + " fields differ from the frozen schema")

def _typed(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(context + " fields differ from the frozen schema")
    if any(type(value[key]) is not type(expected[key]) or value[key] != expected[key]
           for key in expected):
        _fail(context + " value or JSON type drifted")

def _digest(value, context):
    if not isinstance(value, str) or _sha.fullmatch(value) is None:
        _fail(context + " is not a SHA-256 digest")
    return value

def _positive(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(context + " is not a positive integer")
    return value

def _nonnegative(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(context + " is not a non-negative integer")
    return value

def _relative(value, context):
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context + " is not a canonical relative path")
    if any(c in value for c in ("\\", "\0", "\n", "\r", ":")):
        _fail(context + " is not a safe POSIX path")
    path = _p.PurePosixPath(value)
    if value == "." or path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        _fail(context + " is not a canonical relative path")
    for part in path.parts:
        if part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in _reserved:
            _fail(context + " is unsafe on Windows")
    return value

def _base_path(value, context):
    if value == ".":
        return "."
    return _relative(value, context)

def _link(path):
    try:
        status = path.lstat()
    except OSError as error:
        raise RuntimeError("required path is unavailable") from error
    return path.is_symlink() or bool(getattr(status, "st_file_attributes", 0) & _rp)

def _directory(raw, context):
    path = _p.Path(_o.path.abspath(raw))
    if _link(path):
        _fail(context + " is a link or reparse point")
    try:
        result = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(context + " is unavailable") from error
    if not result.is_dir():
        _fail(context + " is not a directory")
    return result

def _temporary_identity(path, context):
    root = _directory(path, context)
    try:
        status = root.lstat()
    except OSError as error:
        raise RuntimeError("cannot identify " + context) from error
    return (int(status.st_dev), int(status.st_ino), _stat.S_IFMT(status.st_mode))

def _directory_chain(raw, context):
    path = _p.Path(raw)
    if not path.is_absolute():
        _fail(context + " must be an absolute path")
    candidate = _p.Path(_o.path.abspath(path))
    current = _p.Path(candidate.anchor)
    components = [current]
    for part in candidate.parts[1:]:
        current /= part
        components.append(current)
    identities = []
    for component in components:
        try:
            status = component.lstat()
        except OSError as error:
            raise RuntimeError(context + " component is unavailable") from error
        if component.is_symlink() or bool(
                getattr(status, "st_file_attributes", 0) & _rp):
            _fail(context + " traverses a link or reparse point")
        if not _stat.S_ISDIR(status.st_mode):
            _fail(context + " component is not a directory")
        identities.append((_o.path.normcase(str(component)), int(status.st_dev),
                           int(status.st_ino), _stat.S_IFMT(status.st_mode)))
    return tuple(identities)

def _assert_private_tree(path, context):
    stack = [path]
    while stack:
        directory = stack.pop()
        try:
            entries = tuple(_o.scandir(directory))
        except OSError as error:
            raise RuntimeError("cannot enumerate " + context) from error
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError("cannot inspect " + context) from error
            if entry.is_symlink() or bool(
                    getattr(status, "st_file_attributes", 0) & _rp):
                _fail(context + " contains a link or reparse point")
            if _stat.S_ISDIR(status.st_mode):
                stack.append(_p.Path(entry.path))
            elif not _stat.S_ISREG(status.st_mode):
                _fail(context + " contains a non-regular path")

def _assert_isolated_cache(cache, runtime_roots):
    if cache == _p.Path(cache.anchor):
        _fail("dataset cache root cannot be a filesystem root")
    for runtime in runtime_roots:
        if cache == runtime:
            _fail("dataset cache root overlaps an authenticated runtime root")
        try:
            cache.relative_to(runtime)
        except ValueError:
            pass
        else:
            _fail("dataset cache root overlaps an authenticated runtime root")
        try:
            runtime.relative_to(cache)
        except ValueError:
            pass
        else:
            _fail("dataset cache root overlaps an authenticated runtime root")

def _join(root, relative, context, directory=False):
    path = root
    for part in _p.PurePosixPath(_relative(relative, context)).parts:
        path /= part
        if _link(path):
            _fail(context + " traverses a link or reparse point")
    try:
        result = path.resolve(strict=True)
        result.relative_to(root)
    except (OSError, ValueError) as error:
        raise RuntimeError(context + " escapes its authenticated root") from error
    if directory and not result.is_dir():
        _fail(context + " is not a directory")
    if not directory and not result.is_file():
        _fail(context + " is not a regular file")
    return result

def _forbidden(relative):
    path = _p.PurePosixPath(relative)
    return (any(part.casefold() in _bad_dir for part in path.parts)
            or path.name.casefold() in _bad_name
            or path.suffix.casefold() in _bad_suffix)

def _file(path, relative, context):
    before = path.stat()
    if not _stat.S_ISREG(before.st_mode) or _link(path):
        _fail(context + " is not a stable regular file")
    digest = _h.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or size != after.st_size or _link(path)):
        _fail(context + " changed during authentication")
    return {"path": relative, "sha256": digest.hexdigest(), "size_bytes": size}

def _bytes(path, context):
    before = _file(path, path.name, context)
    data = path.read_bytes()
    after = _file(path, path.name, context)
    if (before != after or _h.sha256(data).hexdigest() != before["sha256"]
            or len(data) != before["size_bytes"]):
        _fail(context + " changed while it was read")
    return data

def _git(raw):
    try:
        path = _p.Path(raw).resolve(strict=True)
    except OSError as error:
        raise RuntimeError("Git executable is unavailable") from error
    if path.name.casefold() == "git.exe" and path.parent.name.casefold() == "cmd":
        try:
            path = (path.parent.parent / "mingw64" / "bin" / "git.exe").resolve(strict=True)
        except OSError as error:
            raise RuntimeError("Git-for-Windows cmd shim has no implementation") from error
    current = _p.Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _link(current):
            _fail("Git executable traverses a link or reparse point")
    record = _file(path, path.name, "Git executable")
    if record["size_bytes"] <= 0:
        _fail("Git executable is empty")
    return path, {"absolute_path_sha256": _h.sha256(
        _o.path.normcase(str(path)).encode("utf-8")
    ).hexdigest(), "sha256": record["sha256"], "size_bytes": record["size_bytes"]}

def _tree(root, context):
    root = _directory(root, context + " root")
    stack = [(root, ())]
    files = []
    folded = set()
    while stack:
        directory, parents = stack.pop()
        entries = sorted(_o.scandir(directory), key=lambda item: item.name.casefold())
        for entry in entries:
            relative = _relative(_p.PurePosixPath(*parents, entry.name).as_posix(),
                                 context + " path")
            status = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or bool(getattr(status, "st_file_attributes", 0) & _rp):
                _fail(context + " contains a link or reparse point")
            if _stat.S_ISDIR(status.st_mode):
                if entry.name.casefold() in _bad_dir:
                    _fail(context + " contains a forbidden cache directory")
                stack.append((_p.Path(entry.path), (*parents, entry.name)))
                continue
            if not _stat.S_ISREG(status.st_mode) or _forbidden(relative):
                _fail(context + " contains a forbidden or non-regular file")
            if relative.casefold() in folded:
                _fail(context + " contains a case-insensitive path collision")
            folded.add(relative.casefold())
            files.append(_file(_p.Path(entry.path), relative, context + " file"))
    files.sort(key=lambda item: item["path"])
    if not files:
        _fail(context + " has no files")
    return files

def _name(value):
    if not isinstance(value, str) or not value.strip():
        _fail("distribution has no canonical name")
    result = _re.sub(r"[-_.]+", "-", value.strip()).lower()
    if _re.fullmatch(r"[a-z0-9][a-z0-9-]*", result) is None:
        _fail("distribution name is invalid")
    return result

def _options(arguments):
    if arguments and arguments[0] in _capture_commands:
        required_capture = (_calibration_capture_required
            if arguments[0] == _calibration_capture_command
            else _stage_a_capture_required)
        remainder = arguments[1:]
        if len(remainder) != 2 * len(required_capture):
            _fail("sealed capture arguments are not an exact option profile")
        captured = {}
        for index in range(0, len(remainder), 2):
            option = remainder[index]
            value = remainder[index + 1]
            if (option not in required_capture or option in captured
                    or not value or value.startswith("--")):
                _fail("sealed capture arguments are mixed, duplicated, or incomplete")
            captured[option] = value
        if set(captured) != required_capture:
            _fail("sealed capture inputs are incomplete")
        for option in {
            "--cache-root", "--capture-provenance-receipt-output",
            "--model-file-manifest", "--output",
            "--parquet-materialization-manifest", "--repository-root",
            "--repository-source-manifest", "--ruler-receipt-dir",
            "--runtime-manifest", "--stage-a-calibration-binding",
        }:
            if option in captured and not _p.Path(captured[option]).is_absolute():
                _fail("sealed capture path is not absolute: " + option)
        return captured
    required = {
        "--capture-provenance-receipt",
        "--expected-capture-provenance-receipt-sha256",
        "--frozen-identity", "--cache-root", "--repository-root", "--source-commit",
        *_binding_options.values(), *_expected_digest_options.values(),
        "--ruler-receipt-dir",
    }
    value_options = required | _smoke_options
    result = {}
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if (item in _forbidden_runner_options
                or any(item.startswith(value + "=") for value in _forbidden_runner_options)):
            _fail("legacy runner option is forbidden: " + item)
        if item in value_options:
            if (
                item in result
                or index + 1 >= len(arguments)
                or arguments[index + 1].startswith("--")
            ):
                _fail("runner option is duplicated or incomplete: " + item)
            result[item] = arguments[index + 1]
            index += 2
        else:
            index += 1
    if set(result) != required:
        if not required.issubset(result):
            _fail("runner arguments omit required sealed inputs")
    smoke_count = sum(item == "--fisher-h1-smoke" for item in arguments)
    if smoke_count > 1:
        _fail("runner Fisher H=1 smoke flag is duplicated")
    supplied = _smoke_options.intersection(result)
    if smoke_count == 1 and supplied:
        _fail("smoke mode forbids prior Fisher H=1 smoke prerequisites")
    if smoke_count == 0 and supplied != _smoke_options:
        _fail("full calibration requires prior Fisher H=1 smoke prerequisites")
    if not _p.Path(result["--capture-provenance-receipt"]).is_absolute():
        _fail("capture provenance receipt path must be absolute")
    return result

def _smoke(options):
    receipt = _bytes(
        _p.Path(options["--capture-provenance-receipt"]),
        "capture provenance receipt",
    )
    receipt_sha256 = _h.sha256(receipt).hexdigest()
    if receipt_sha256 != _digest(
            options["--expected-capture-provenance-receipt-sha256"],
            "capture provenance receipt SHA-256"):
        _fail("capture provenance receipt differs from its explicit SHA-256")
    receipt_root = _json(receipt, "capture provenance receipt")
    _fields(receipt_root, {
        "artifact_kind", "capture_source", "capture_version",
        "critical_module_origins", "excluded_runtime_modules",
        "execution_bindings", "identity_input_file_sha256", "phase",
        "publication_contract", "runner_revision", "schema_version",
        "source_commit", "status",
    }, "capture provenance receipt")
    if (_canonical(receipt_root) != receipt
            or receipt_root["artifact_kind"]
            != "recurquant_experiment013_calibration_identity_capture_provenance"
            or type(receipt_root["schema_version"]) is not int
            or receipt_root["schema_version"] != 2
            or type(receipt_root["capture_version"]) is not int
            or receipt_root["capture_version"] != 6
            or receipt_root["runner_revision"]
            != "experiment-013-static-q468-calibration-runner-v13"
            or receipt_root["phase"] != "calibration"
            or receipt_root["publication_contract"]
            != "sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1"
            or receipt_root["status"]
            != "captured_under_authenticated_runtime_and_launcher_finalized"
            or receipt_root["source_commit"] != options["--source-commit"]):
        _fail("capture provenance receipt finalized envelope drifted")
    if "--prior-fisher-h1-smoke-report" not in options:
        return
    if _p.Path(options["--prior-fisher-h1-smoke-complete-marker"]).read_bytes() != _smoke_marker:
        _fail("prior Fisher H=1 smoke completion marker drifted")
    data = _p.Path(options["--prior-fisher-h1-smoke-report"]).read_bytes()
    root = _json(data, "prior Fisher H=1 smoke report")
    _fields(root, {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
            "prior Fisher H=1 smoke report")
    evidence = root["evidence"]
    if (_canonical(root) != data
            or root["artifact_kind"] != "recurquant_experiment013_calibration_run"
            or type(root["schema_version"]) is not int or root["schema_version"] != 3
            or not isinstance(evidence, dict)
            or evidence.get("status") != "fisher_h1_smoke_passed"
            or evidence.get("runner_revision")
            != "experiment-013-static-q468-calibration-runner-v13"
            or evidence.get("prerequisites") != {
                "capture_provenance_receipt_file_sha256": receipt_sha256,
                "fisher_h1_smoke_report_file_sha256": None,
            }
            or _digest(root["canonical_evidence_sha256"], "smoke evidence hash")
            != _h.sha256(_canonical(evidence)).hexdigest()):
        _fail("prior Fisher H=1 smoke report authentication failed")

def _identity(data):
    root = _json(data, "frozen identity")
    _fields(root, {"canonical_evidence_sha256", "evidence"}, "frozen identity")
    if _canonical(root) != data:
        _fail("frozen identity is not canonical JSON")
    evidence = root["evidence"]
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 5
            or evidence.get("status") != "frozen" or evidence.get("phase") != "calibration"
            or evidence.get("identity_only") is not True
            or evidence.get("promotion_required") is not False):
        _fail("frozen identity state or schema drifted")
    if _digest(root["canonical_evidence_sha256"], "identity evidence hash") != _h.sha256(
        _canonical(evidence)
    ).hexdigest():
        _fail("frozen identity evidence hash drifted")
    bindings = evidence.get("execution_bindings")
    if not isinstance(bindings, dict):
        _fail("frozen identity bindings are missing")
    _fields(bindings, set(_binding_options), "identity bindings")
    return {key: _digest(value, "identity binding") for key, value in bindings.items()}

def _source(data):
    root = _json(data, "source manifest")
    _fields(root, {"canonical_manifest_sha256", "git_executable", "object_format",
                   "paths", "profile", "repository_binding", "schema",
                   "source_commit"}, "source manifest")
    if _pretty(root) != data:
        _fail("source manifest is not canonical JSON")
    payload = dict(root)
    claimed = _digest(payload.pop("canonical_manifest_sha256"), "source self-hash")
    if claimed != _h.sha256(_pretty(payload)).hexdigest():
        _fail("source manifest self-hash drifted")
    if (root["schema"] != "recurquant.experiment013.source-manifest.v2"
            or root["profile"] != "experiment-013-static-q468-frozen-source-v2"
            or root["object_format"] != "sha1"
            or not isinstance(root["source_commit"], str)
            or _sha1.fullmatch(root["source_commit"]) is None):
        _fail("source manifest profile drifted")
    git = root["git_executable"]
    if not isinstance(git, dict):
        _fail("source Git executable record is invalid")
    _fields(git, {"sha256", "size_bytes"}, "source Git executable")
    git = {"sha256": _digest(git["sha256"], "source Git executable hash"),
           "size_bytes": _positive(git["size_bytes"], "source Git executable size")}
    paths = []
    for entry in root["paths"]:
        _fields(entry, {"git_blob_oid", "index_blob_oid", "mode", "path", "raw_sha256",
                        "worktree_blob_oid"}, "source path")
        paths.append({"path": _relative(entry["path"], "source path"),
                      "raw_sha256": _digest(entry["raw_sha256"], "source SHA-256")})
    rendered = [item["path"] for item in paths]
    if rendered != sorted(rendered) or "scripts/run_static_q468_calibration.py" not in rendered:
        _fail("source path inventory drifted")
    return {"file_sha256": _h.sha256(data).hexdigest(), "git_executable": git,
            "paths": paths, "source_commit": root["source_commit"]}

def _verify_source(manifest, root):
    root = _directory(root, "repository root")
    runner = None
    for entry in manifest["paths"]:
        path = _join(root, entry["path"], "source file")
        record = _file(path, entry["path"], "source file")
        if record["sha256"] != entry["raw_sha256"]:
            _fail("source bytes drifted")
        if entry["path"] == "scripts/run_static_q468_calibration.py":
            runner = path
    if runner is None:
        _fail("source manifest omitted runner")
    return runner

def _distributions(packages, imports):
    found = {}
    for root_name in sorted(packages):
        search = _join(packages[root_name], imports[root_name], "package import path", True)
        for distribution in _md.distributions(path=[str(search)]):
            name = _name(distribution.metadata.get("Name"))
            if name in found or distribution.files is None:
                _fail("staged distribution inventory is invalid")
            files = []
            for item in distribution.files:
                candidate = _p.Path(distribution.locate_file(item)).resolve(strict=True)
                try:
                    relative = candidate.relative_to(packages[root_name]).as_posix()
                except ValueError as error:
                    raise RuntimeError("RECORD path escapes package tree") from error
                _join(packages[root_name], relative, "RECORD path")
                files.append(_relative(relative, "normalized RECORD path"))
            files.sort()
            if len([item for item in files if item.casefold().endswith(".dist-info/record")]) != 1:
                _fail("distribution must contain exactly one RECORD")
            found[name] = {"files": files, "name": name, "package_root": root_name,
                           "version": str(distribution.version)}
    return [found[name] for name in sorted(found)]

def _manifest(data):
    root = _json(data, "runtime manifest")
    _fields(root, {"artifact_kind", "base_runtime_root", "base_sys_path", "distributions",
                   "git_executable",
                   "interpreter", "launch_policy", "machine", "package_roots", "python",
                   "runtime_trees", "schema_version"}, "runtime manifest")
    if _canonical(root) != data or root["artifact_kind"] != (
        "recurquant_experiment013_calibration_runtime_manifest"
    ) or type(root["schema_version"]) is not int or root["schema_version"] != 6:
        _fail("runtime manifest identity or policy drifted")
    _typed(root["launch_policy"], _policy, "runtime launch policy")
    if root["base_runtime_root"] != "base-runtime":
        _fail("base runtime name drifted")

    git = root["git_executable"]
    if not isinstance(git, dict):
        _fail("runtime Git executable record is invalid")
    _fields(git, {"absolute_path_sha256", "sha256", "size_bytes"},
            "runtime Git executable")
    git = {"absolute_path_sha256": _digest(
               git["absolute_path_sha256"], "runtime Git executable path hash"),
           "sha256": _digest(git["sha256"], "runtime Git executable hash"),
           "size_bytes": _positive(git["size_bytes"], "runtime Git executable size")}

    python = root["python"]
    if not isinstance(python, dict):
        _fail("runtime Python record is invalid")
    _fields(python, {"abi_flags", "cache_tag", "implementation", "version"},
            "runtime Python record")
    for field in ("abi_flags", "cache_tag", "implementation", "version"):
        value = python[field]
        if (not isinstance(value, str) or value != value.strip()
                or (field != "abi_flags" and not value)):
            _fail("runtime Python identity is invalid")

    machine = root["machine"]
    if not isinstance(machine, dict):
        _fail("runtime machine record is invalid")
    _fields(machine, {"architecture", "byteorder", "machine", "pointer_bits", "system"},
            "runtime machine record")
    for field in ("architecture", "machine", "system"):
        value = machine[field]
        if not isinstance(value, str) or not value or value != value.strip():
            _fail("runtime machine identity is invalid")
    if machine["byteorder"] not in {"big", "little"}:
        _fail("runtime machine byte order is invalid")
    _positive(machine["pointer_bits"], "runtime pointer bits")

    raw_roots = root["package_roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        _fail("runtime package roots are invalid")
    roots = []
    for item in raw_roots:
        if not isinstance(item, dict):
            _fail("package root is not an object")
        _fields(item, {"import_path", "name"}, "package root")
        name = item["name"]
        if not isinstance(name, str) or _root_re.fullmatch(name) is None:
            _fail("package root name is invalid")
        roots.append({"name": name, "import_path": _relative(item["import_path"], "import path")})
    names = [item["name"] for item in roots]
    if (names != sorted(names) or len(set(names)) != len(names)
            or "base-runtime" in names):
        _fail("package roots drifted")

    raw_base_paths = root["base_sys_path"]
    if not isinstance(raw_base_paths, list) or not raw_base_paths:
        _fail("base sys.path is invalid")
    base_paths = [_base_path(item, "base sys.path") for item in raw_base_paths]
    if len({item.casefold() for item in base_paths}) != len(base_paths):
        _fail("base sys.path entries collide")

    trees = root["runtime_trees"]
    if not isinstance(trees, list) or len(trees) != len(roots) + 1:
        _fail("runtime tree count drifted")
    expected_names = ["base-runtime", *names]
    normalized_trees = []
    for index, tree in enumerate(trees):
        if not isinstance(tree, dict):
            _fail("runtime tree is not an object")
        _fields(tree, {"files", "kind", "name"}, "runtime tree")
        if tree["name"] != expected_names[index] or tree["kind"] != (
            "base-runtime" if index == 0 else "packages"
        ):
            _fail("runtime tree order drifted")
        if not isinstance(tree["files"], list) or not tree["files"]:
            _fail("runtime tree file inventory is invalid")
        files = []
        for item in tree["files"]:
            if not isinstance(item, dict):
                _fail("runtime file is not an object")
            _fields(item, {"path", "sha256", "size_bytes"}, "runtime file")
            relative = _relative(item["path"], "runtime file")
            if _forbidden(relative):
                _fail("runtime manifest contains forbidden file")
            files.append({"path": relative, "sha256": _digest(item["sha256"], "file hash"),
                          "size_bytes": _nonnegative(item["size_bytes"], "file size")})
        paths = [item["path"] for item in files]
        if paths != sorted(paths) or len({item.casefold() for item in paths}) != len(paths):
            _fail("runtime file inventory is not unique and sorted")
        normalized_trees.append({"name": tree["name"], "kind": tree["kind"], "files": files})

    interpreter = root["interpreter"]
    if not isinstance(interpreter, dict):
        _fail("runtime interpreter is not an object")
    _fields(interpreter, {"relative_path", "root", "sha256", "size_bytes"}, "interpreter")
    normalized_interpreter = {
        "relative_path": _relative(interpreter["relative_path"], "interpreter path"),
        "root": interpreter["root"],
        "sha256": _digest(interpreter["sha256"], "interpreter hash"),
        "size_bytes": _positive(interpreter["size_bytes"], "interpreter size"),
    }
    if normalized_interpreter["root"] != "base-runtime":
        _fail("runtime interpreter root drifted")
    base_files = {item["path"]: item for item in normalized_trees[0]["files"]}
    expected_interpreter = {
        "path": normalized_interpreter["relative_path"],
        "sha256": normalized_interpreter["sha256"],
        "size_bytes": normalized_interpreter["size_bytes"],
    }
    if base_files.get(normalized_interpreter["relative_path"]) != expected_interpreter:
        _fail("runtime interpreter differs from the base tree")
    for entry in base_paths:
        present = (entry == "." or entry in base_files
                   or any(path.startswith(entry + "/") for path in base_files))
        if not present and _re.fullmatch(r"python[0-9]+\.zip", entry) is None:
            _fail("base sys.path entry is absent from the base tree")

    raw_distributions = root["distributions"]
    if not isinstance(raw_distributions, list) or not raw_distributions:
        _fail("runtime distributions are invalid")
    distributions = []
    ownership = {name: set() for name in names}
    for item in raw_distributions:
        if not isinstance(item, dict):
            _fail("runtime distribution is not an object")
        _fields(item, {"files", "name", "package_root", "version"},
                "runtime distribution")
        name = _name(item["name"])
        package_root = item["package_root"]
        version = item["version"]
        if (name != item["name"] or package_root not in ownership
                or not isinstance(version, str) or not version
                or version != version.strip()):
            _fail("runtime distribution identity is invalid")
        raw_files = item["files"]
        if not isinstance(raw_files, list) or not raw_files:
            _fail("runtime distribution files are invalid")
        files = [_relative(path, "distribution RECORD path") for path in raw_files]
        if files != sorted(files) or len({path.casefold() for path in files}) != len(files):
            _fail("distribution RECORD paths are not unique and sorted")
        if ownership[package_root].intersection(files):
            _fail("runtime distributions have overlapping RECORD ownership")
        ownership[package_root].update(files)
        distributions.append({"files": files, "name": name,
                              "package_root": package_root, "version": version})
    distribution_names = [item["name"] for item in distributions]
    if (distribution_names != sorted(distribution_names)
            or len(set(distribution_names)) != len(distribution_names)):
        _fail("runtime distributions are not unique and sorted")
    tree_by_name = {item["name"]: item for item in normalized_trees}
    for name in names:
        if ownership[name] != {item["path"] for item in tree_by_name[name]["files"]}:
            _fail("package tree differs from exact RECORD ownership")

    return {"base_sys_path": base_paths, "distributions": distributions,
            "file_sha256": _h.sha256(data).hexdigest(),
            "git_executable": git,
            "interpreter": normalized_interpreter,
            "machine": machine, "package_roots": roots, "python": python,
            "runtime_trees": normalized_trees}

def _verify_runtime(manifest, base_raw, package_raw, git_raw, packages_appended=False):
    base = _directory(base_raw, "base runtime")
    declared = {item["name"]: item["import_path"] for item in manifest["package_roots"]}
    if set(package_raw) != set(declared):
        _fail("CLI package roots differ from manifest")
    packages = {name: _directory(package_raw[name], "package root") for name in sorted(package_raw)}
    tree_roots = {"base-runtime": base, **packages}
    for tree in manifest["runtime_trees"]:
        if _tree(tree_roots[tree["name"]], "runtime tree") != tree["files"]:
            _fail("complete runtime tree identity drifted")
    if _distributions(packages, declared) != manifest["distributions"]:
        _fail("distribution RECORD identity drifted")
    git, git_record = _git(git_raw)
    if git_record != manifest["git_executable"]:
        _fail("Git executable identity drifted")
    interpreter = _join(base, manifest["interpreter"]["relative_path"], "interpreter")
    if _file(
        interpreter,
        manifest["interpreter"]["relative_path"],
        "interpreter",
    ) != {
        "path": manifest["interpreter"]["relative_path"],
        "sha256": manifest["interpreter"]["sha256"],
        "size_bytes": manifest["interpreter"]["size_bytes"],
    }:
        _fail("point-used interpreter identity drifted")
    if _p.Path(_s.executable).resolve(strict=True) != interpreter:
        _fail("point-used interpreter path drifted")
    python = manifest["python"]
    machine = manifest["machine"]
    if (_platform.python_implementation() != python["implementation"]
            or _platform.python_version() != python["version"]
            or _s.implementation.cache_tag != python["cache_tag"]
            or getattr(_s, "abiflags", "") != python["abi_flags"]
            or _platform.system() != machine["system"]
            or f"{8 * _struct.calcsize('P')}bit" != machine["architecture"]
            or _platform.machine() != machine["machine"]
            or _s.byteorder != machine["byteorder"]
            or 8 * _struct.calcsize("P") != machine["pointer_bits"]):
        _fail("point-used Python or machine identity drifted")
    imports = {name: str(_join(packages[name], declared[name], "import path", True))
               for name in sorted(packages)}
    expected_path = [str(base / _p.PurePosixPath(item))
                     for item in manifest["base_sys_path"]]
    if packages_appended:
        expected_path.extend(imports[name] for name in sorted(imports))
    if [_o.path.normcase(_o.path.abspath(item)) for item in _s.path] != [
        _o.path.normcase(_o.path.abspath(item)) for item in expected_path
    ]:
        _fail("authenticated sys.path differs from the frozen runtime")
    return base, packages, declared, imports, interpreter, git

_runtime_path = _p.Path(_s.argv[1])
_base_raw = _p.Path(_s.argv[2])
try:
    _package_raw = {name: _p.Path(path) for name, path in _j.loads(_s.argv[3]).items()}
except Exception as error:
    raise RuntimeError("package-root bootstrap argument is invalid") from error
_pycache = _directory(_s.argv[4], "pycache prefix")
if any(_pycache.iterdir()) or _s.pycache_prefix is None or _p.Path(
    _s.pycache_prefix
).resolve(strict=True) != _pycache:
    _fail("sealed pycache prefix is not exact and empty")
_git_raw = _p.Path(_s.argv[5])
_scratch = _directory(_s.argv[6], "sealed scratch directory")
if any(_scratch.iterdir()):
    _fail("sealed scratch directory is not initially empty")
_scratch_identity = _temporary_identity(_scratch, "sealed scratch directory")
if _p.Path.cwd().resolve(strict=True) != _scratch:
    _fail("sealed child cwd differs from the launcher-owned scratch directory")
_runner_args = list(_s.argv[7:])
_runner_options = _options(_runner_args)
_capture_profile = bool(_runner_args and _runner_args[0] in _capture_commands)
_stage_a_capture_profile = bool(
    _runner_args and _runner_args[0] == _stage_a_capture_command)
_cache_raw = _p.Path(_runner_options["--cache-root"])
if not _cache_raw.is_absolute():
    _fail("dataset cache root must be an absolute path")
_cache_root = _directory(_cache_raw, "dataset cache root")
_cache_identity = _directory_chain(_cache_raw, "dataset cache root")
_private_home = _scratch / "private-home"
_hf_home = _scratch / "huggingface"
if _capture_profile:
    _hf_hub_raw = _cache_root / "hub"
    _hf_hub_identity = _directory_chain(_hf_hub_raw, "capture Hub cache root")
    _hf_hub_cache = _directory(_hf_hub_raw, "capture Hub cache root")
    if _hf_hub_cache.parent != _cache_root:
        _fail("capture Hub cache root escaped the dataset cache root")
else:
    _hf_hub_cache = _hf_home / "hub"
    _hf_hub_identity = None
_torch_home = _scratch / "torch"
_datasets_cache = _cache_root / "datasets"
_expected_environment = {
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_ASSETS_CACHE": str(_hf_home / "assets"),
    "HF_DATASETS_CACHE": str(_datasets_cache),
    "HF_DATASETS_DOWNLOADED_DATASETS_PATH": str(_datasets_cache / "downloads"),
    "HF_DATASETS_EXTRACTED_DATASETS_PATH": str(_datasets_cache / "downloads" / "extracted"),
    "HF_HOME": str(_hf_home),
    "HF_HUB_CACHE": str(_hf_hub_cache),
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "HF_HUB_DISABLE_XET": "1",
    "HF_MODULES_CACHE": str(_hf_home / "modules"),
    "HF_TOKEN_PATH": str(_hf_home / "token"),
    "HF_XET_CACHE": str(_hf_home / "xet"),
    "HOME": str(_private_home),
    "HUGGINGFACE_ASSETS_CACHE": str(_hf_home / "assets"),
    "HUGGINGFACE_HUB_CACHE": str(_hf_hub_cache),
    "LANG": "C",
    "LC_ALL": "C",
    "PYTORCH_KERNEL_CACHE_PATH": str(_torch_home / "kernels"),
    "TEMP": str(_scratch),
    "TMP": str(_scratch),
    "TORCH_EXTENSIONS_DIR": str(_torch_home / "extensions"),
    "TORCH_HOME": str(_torch_home),
    "TORCHINDUCTOR_CACHE_DIR": str(_torch_home / "inductor"),
    "TRANSFORMERS_CACHE": str(_scratch / "transformers"),
    "TRITON_CACHE_DIR": str(_torch_home / "triton"),
    "TZ": "UTC",
    "USERPROFILE": str(_private_home),
    "XDG_CACHE_HOME": str(_scratch / "xdg-cache"),
}
_environment = {key.upper(): value for key, value in _o.environ.items()}
_os_environment = {
    "SYSTEMROOT", "WINDIR", "COMSPEC",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432",
}
_allowed_environment = set(_expected_environment) | _os_environment
if (set(_environment) - _os_environment != set(_expected_environment)
        or any(_environment.get(name) != value
               for name, value in _expected_environment.items())
        or any(not value or "\0" in value or "\n" in value or "\r" in value
               for key, value in _environment.items()
               if key in _os_environment)):
    _fail("sealed child environment differs from the private cache contract")
if not _capture_profile:
    _smoke(_runner_options)
_runtime_bytes = _bytes(_runtime_path, "runtime manifest")
_runtime = _manifest(_runtime_bytes)
if _capture_profile:
    _bindings = {}
    for _binding, _option in _binding_options.items():
        _artifact = _bytes(_p.Path(_runner_options[_option]), "capture bound artifact")
        _actual = _h.sha256(_artifact).hexdigest()
        _expected_option = _capture_expected_digest_options[_binding]
        if _actual != _digest(_runner_options[_expected_option], "capture digest binding"):
            _fail("capture artifact digest mismatch: " + _option)
        _bindings[_binding] = _actual
    if _stage_a_capture_profile:
        _stage_a_binding = _bytes(
            _p.Path(_runner_options["--stage-a-calibration-binding"]),
            "Stage-A calibration binding",
        )
        if _h.sha256(_stage_a_binding).hexdigest() != _digest(
                _runner_options["--expected-stage-a-calibration-binding-sha256"],
                "expected Stage-A calibration binding SHA-256"):
            _fail("Stage-A calibration binding differs from its explicit SHA-256")
else:
    _identity_path = _p.Path(_runner_options["--frozen-identity"])
    _bindings = _identity(_bytes(_identity_path, "frozen identity"))
    for _binding, _option in _binding_options.items():
        if _h.sha256(_bytes(
                _p.Path(_runner_options[_option]), "identity-bound artifact"
        )).hexdigest() != _bindings[_binding]:
            _fail("identity binding mismatch: " + _option)
    for _binding, _option in _expected_digest_options.items():
        if _digest(_runner_options[_option], "runner digest binding") != _bindings[_binding]:
            _fail("runner digest binding mismatch: " + _option)
if _runtime_path.resolve(strict=True) != _p.Path(
    _runner_options["--runtime-manifest"]
).resolve(strict=True):
    _fail("host and runner runtime manifests differ")
_source_manifest = _source(
    _bytes(_p.Path(_runner_options["--repository-source-manifest"]), "source manifest")
)
if _runner_options["--source-commit"] != _source_manifest["source_commit"]:
    _fail("runner source commit differs from source-manifest H0")
if _source_manifest["git_executable"] != {
    "sha256": _runtime["git_executable"]["sha256"],
    "size_bytes": _runtime["git_executable"]["size_bytes"],
}:
    _fail("source and runtime manifests bind different Git bytes")
_repository_root = _p.Path(_runner_options["--repository-root"])
_runner_path = _verify_source(_source_manifest, _repository_root)
_base, _packages, _import_rel, _imports, _interpreter, _git_executable = _verify_runtime(
    _runtime, _base_raw, _package_raw, _git_raw
)
_assert_isolated_cache(_cache_root, [_base, *_packages.values(), _pycache, _scratch])
_s.path.extend(_imports[name] for name in sorted(_imports))
if _s.path != [str(_base / _p.PurePosixPath(item)) for item in _runtime["base_sys_path"]] + [
    _imports[name] for name in sorted(_imports)
]:
    _fail("authenticated package paths were not appended exactly")
if _sensitive.intersection(_s.modules):
    _fail("sensitive module appeared before exact runner load")
_before = _file(_runner_path, "scripts/run_static_q468_calibration.py", "runner source")
_payload = _bytes(_runner_path, "runner source")
if _h.sha256(_payload).hexdigest() != _before["sha256"]:
    _fail("runner source changed before compile")
_module = _types.ModuleType("_recurquant_experiment013_sealed_runner")
_module.__file__ = str(_runner_path)
_module.__package__ = ""
_s.modules[_module.__name__] = _module
_result = None
try:
    try:
        _code = compile(_payload, str(_runner_path), "exec", dont_inherit=True)
        exec(_code, _module.__dict__)
        if _file(
            _runner_path,
            "scripts/run_static_q468_calibration.py",
            "runner source",
        ) != _before:
            _fail("runner source changed during exact load")
        _sealed_main = getattr(_module, "sealed_main", None)
        if not callable(_sealed_main):
            _fail("authenticated runner has no sealed_main entrypoint")
        _result = _sealed_main(
            _runner_args,
            base_runtime_root=_base,
            package_roots=_packages,
            package_import_paths=_import_rel,
            interpreter_path=_interpreter,
            git_executable_path=_git_executable,
            pycache_prefix=_pycache,
        )
        if not isinstance(_result, int) or isinstance(_result, bool):
            _fail("sealed_main returned a non-integer status")
    finally:
        _primary = _s.exception()
        _postcondition_failures = []
        if _capture_profile:
            try:
                _repeated_hub_raw = _cache_root / "hub"
                _repeated_hub_identity = _directory_chain(
                    _repeated_hub_raw, "capture Hub cache root")
                _repeated_hub = _directory(
                    _repeated_hub_raw, "capture Hub cache root")
                if (_repeated_hub.parent != _cache_root
                        or _repeated_hub_identity != _hf_hub_identity):
                    raise RuntimeError("capture Hub cache root identity changed")
            except Exception as error:
                _postcondition_failures.append(
                    ("capture Hub cache root reauthentication", error)
                )
        try:
            if any(_pycache.iterdir()):
                raise RuntimeError("sealed pycache prefix changed during calibration")
        except Exception as error:
            _postcondition_failures.append(("pycache", error))
        try:
            if _temporary_identity(
                    _scratch, "sealed scratch directory") != _scratch_identity:
                raise RuntimeError("sealed scratch directory identity changed")
            _assert_private_tree(_scratch, "sealed scratch directory")
            if any(_scratch.iterdir()):
                raise RuntimeError("sealed scratch directory was not left empty")
        except Exception as error:
            _postcondition_failures.append(("scratch containment", error))
        try:
            if _directory_chain(
                    _cache_root, "dataset cache root") != _cache_identity:
                raise RuntimeError("dataset cache root identity changed")
            _assert_isolated_cache(
                _cache_root, [_base, *_packages.values(), _pycache, _scratch]
            )
        except Exception as error:
            _postcondition_failures.append(("dataset cache root reauthentication", error))
        try:
            _verify_source(_source_manifest, _repository_root)
        except Exception as error:
            _postcondition_failures.append(("repository source reauthentication", error))
        try:
            _verify_runtime(
                _runtime, _base, _packages, _git_executable, packages_appended=True
            )
        except Exception as error:
            _postcondition_failures.append(("runtime reauthentication", error))
        _surface_postcondition_failures(_primary, _result, _postcondition_failures)
finally:
    _s.modules.pop("_recurquant_experiment013_sealed_runner", None)
raise SystemExit(_result)
""".strip()

SEALED_BOOTSTRAP_BYTES: Final = SEALED_BOOTSTRAP.encode("utf-8")
SEALED_STDIN_LOADER: Final = _authenticated_stdin_loader(SEALED_BOOTSTRAP_BYTES)


def _split_host_and_runner_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    positions = [index for index, value in enumerate(argv) if value == "--"]
    if len(positions) != 1:
        raise SealedLaunchError("launcher requires exactly one -- separator")
    separator = positions[0]
    host = list(argv[:separator])
    runner = list(argv[separator + 1 :])
    if not runner:
        raise SealedLaunchError("launcher requires exact runner arguments after --")
    return host, runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-root", required=True, type=Path)
    parser.add_argument("--git-executable", required=True, type=Path)
    parser.add_argument("--package-root", required=True, action="append", type=_parse_package_root)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    return parser


def _launch_with_custody(
    argv: Sequence[str],
    *,
    _executable_custody: _HeldExecutableCustody | None = None,
) -> int:
    if list(argv) in (["-h"], ["--help"]):
        _parser().print_help()
        print("\nAppend -- followed by the exact run_static_q468_calibration.py arguments.")
        return 0
    host_arguments, runner_arguments = _split_host_and_runner_args(argv)
    args = _parser().parse_args(host_arguments)
    package_roots: dict[str, Path] = {}
    for name, path in args.package_root:
        if name in package_roots or name == BASE_RUNTIME_ROOT_NAME:
            raise SealedLaunchError(f"duplicate or reserved package root: {name}")
        package_roots[name] = path

    runtime_bytes = _stable_file_bytes(
        args.runtime_manifest,
        context="runtime manifest",
    )
    runtime_manifest = _parse_runtime_manifest(runtime_bytes)
    runner_options = _extract_runner_options(runner_arguments)
    bindings, source_manifest, _runner_path = _verify_bound_artifacts(
        runner_options,
        runtime_manifest_path=args.runtime_manifest,
    )
    capture_command = runner_arguments[0] if runner_arguments else None
    capture_profile = capture_command in _CAPTURE_COMMANDS
    capture_phase = "stage_a" if capture_command == _STAGE_A_CAPTURE_COMMAND else "calibration"
    capture_context = "Stage-A" if capture_phase == "stage_a" else "calibration"
    stage_a_binding_bytes: bytes | None = None
    stage_a_binding_envelope: dict[str, object] | None = None
    if capture_phase == "stage_a":
        stage_a_binding_bytes, stage_a_binding_envelope = _verify_stage_a_capture_binding(
            runner_options,
            execution_bindings=bindings,
            source_commit=str(source_manifest["source_commit"]),
        )
    if source_manifest["git_executable"] != {
        "sha256": runtime_manifest["git_executable"]["sha256"],
        "size_bytes": runtime_manifest["git_executable"]["size_bytes"],
    }:
        raise SealedLaunchError("source and runtime manifests bind different Git bytes")
    base, packages, _import_paths, interpreter, git_executable = _verify_runtime(
        runtime_manifest,
        base_runtime_root=args.base_runtime_root,
        package_roots=package_roots,
        git_executable_path=args.git_executable,
        require_current_process=False,
    )
    if _executable_custody is None:
        with _acquire_executable_custody(
            interpreter=interpreter,
            git_executable=git_executable,
            runtime_manifest=runtime_manifest,
        ) as custody:
            result = _launch_with_custody(argv, _executable_custody=custody)
            custody.verify()
            return result
    _executable_custody.verify()
    dataset_cache_root = _verified_dataset_cache_root(
        Path(runner_options["--cache-root"]),
        runtime_roots=(base, *packages.values()),
    )
    dataset_cache_identity = _non_link_directory_identity_chain(
        dataset_cache_root,
        context="dataset cache root",
    )
    capture_hub_cache_root: Path | None = None
    capture_hub_cache_identity: tuple[tuple[str, int, int, int], ...] | None = None
    if capture_profile:
        capture_hub_cache_root = _prepare_capture_hub_cache_root(dataset_cache_root)
        capture_hub_cache_identity = _non_link_directory_identity_chain(
            capture_hub_cache_root,
            context="capture Hub cache root",
        )
    capture_output_snapshot: CaptureArtifactSnapshot | None = None
    capture_receipt_snapshot: CaptureArtifactSnapshot | None = None
    if capture_profile:
        capture_output_snapshot = _new_capture_artifact_snapshot(
            Path(runner_options["--output"]),
            context=f"{capture_context} identity output",
        )
        capture_receipt_snapshot = _new_capture_artifact_snapshot(
            Path(runner_options["--capture-provenance-receipt-output"]),
            context=f"{capture_context} identity capture provenance receipt",
        )
        _validate_capture_artifact_disjointness(
            (capture_output_snapshot, capture_receipt_snapshot),
            forbidden_roots=(base, *packages.values(), dataset_cache_root),
        )

    pycache: Path | None = None
    scratch: Path | None = None
    pycache_identity: tuple[int, int, int] | None = None
    scratch_identity: tuple[int, int, int] | None = None
    completed: subprocess.CompletedProcess[bytes] | None = None
    child_returncode: int | None = None
    primary_error: BaseException | None = None
    secondary_failures: list[tuple[str, BaseException]] = []
    try:
        pycache = Path(tempfile.mkdtemp(prefix="recurquant-exp013-sealed-pycache-"))
        pycache_identity = _temporary_directory_identity(pycache, context="pycache prefix")
        pycache = _verify_empty_pycache(pycache)
        scratch = Path(tempfile.mkdtemp(prefix="recurquant-exp013-sealed-scratch-"))
        scratch_identity = _temporary_directory_identity(
            scratch,
            context="sealed scratch directory",
        )
        scratch = _verify_empty_scratch(scratch)
        _verified_dataset_cache_root(
            dataset_cache_root,
            runtime_roots=(base, *packages.values(), pycache, scratch),
        )
        if capture_profile:
            assert capture_output_snapshot is not None
            assert capture_receipt_snapshot is not None
            _validate_capture_artifact_disjointness(
                (capture_output_snapshot, capture_receipt_snapshot),
                forbidden_roots=(
                    base,
                    *packages.values(),
                    dataset_cache_root,
                    pycache,
                    scratch,
                ),
            )
        command = _sealed_argv(
            interpreter=interpreter,
            runtime_manifest=args.runtime_manifest.resolve(strict=True),
            base_runtime_root=base,
            package_roots=packages,
            git_executable=git_executable,
            pycache_prefix=pycache,
            scratch_directory=scratch,
            runner_arguments=runner_arguments,
        )
        run_arguments = {
            "check": False,
            "cwd": scratch,
            "env": _sealed_environment(
                scratch_directory=scratch,
                dataset_cache_root=dataset_cache_root,
                capture_profile=capture_profile,
            ),
            "input": SEALED_BOOTSTRAP_BYTES,
        }
        if capture_profile:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                **run_arguments,
            )
        else:
            completed = subprocess.run(command, **run_arguments)
        child_returncode = int(completed.returncode)
        if capture_profile:
            assert capture_hub_cache_identity is not None
            try:
                _verify_capture_hub_cache_identity(
                    dataset_cache_root,
                    expected_identity=capture_hub_cache_identity,
                )
            except Exception as exc:
                secondary_failures.append(
                    ("capture Hub cache root immediate reauthentication", exc)
                )
        try:
            _verify_empty_pycache(pycache)
        except Exception as exc:
            secondary_failures.append(("pycache postcondition", exc))
        try:
            if (
                _temporary_directory_identity(
                    scratch,
                    context="sealed scratch directory",
                )
                != scratch_identity
            ):
                raise SealedLaunchError(
                    "sealed scratch directory identity changed during execution"
                )
            _assert_owned_temporary_tree_has_no_reparse(
                scratch,
                context="sealed scratch directory",
            )
            _verify_empty_scratch(scratch)
        except Exception as exc:
            secondary_failures.append(("scratch containment postcondition", exc))
        try:
            repeated_cache_root = _verified_dataset_cache_root(
                dataset_cache_root,
                runtime_roots=(base, *packages.values(), pycache, scratch),
            )
            if (
                _non_link_directory_identity_chain(
                    repeated_cache_root,
                    context="dataset cache root",
                )
                != dataset_cache_identity
            ):
                raise SealedLaunchError("dataset cache root identity changed during execution")
        except Exception as exc:
            secondary_failures.append(("dataset cache root reauthentication", exc))
        try:
            repeated_bindings, repeated_source, _runner_path = _verify_bound_artifacts(
                runner_options,
                runtime_manifest_path=args.runtime_manifest,
            )
            if repeated_bindings != bindings or repeated_source != source_manifest:
                raise SealedLaunchError("bound artifacts changed during execution")
            if capture_phase == "stage_a":
                repeated_binding_bytes, repeated_binding_envelope = _verify_stage_a_capture_binding(
                    runner_options,
                    execution_bindings=repeated_bindings,
                    source_commit=str(repeated_source["source_commit"]),
                )
                if (
                    repeated_binding_bytes != stage_a_binding_bytes
                    or repeated_binding_envelope != stage_a_binding_envelope
                ):
                    raise SealedLaunchError("Stage-A calibration binding changed during execution")
        except Exception as exc:
            secondary_failures.append(("bound-artifact reauthentication", exc))
        try:
            _verify_runtime(
                runtime_manifest,
                base_runtime_root=base,
                package_roots=packages,
                git_executable_path=git_executable,
                require_current_process=False,
            )
        except Exception as exc:
            secondary_failures.append(("runtime reauthentication", exc))
        if child_returncode == 0 and secondary_failures:
            failures = tuple(secondary_failures)
            secondary_failures.clear()
            raise _postcondition_error(failures)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        for path, expected_identity, context in (
            (scratch, scratch_identity, "sealed scratch directory"),
            (pycache, pycache_identity, "pycache prefix"),
        ):
            if path is None or expected_identity is None:
                continue
            try:
                _cleanup_owned_temporary_directory(
                    path,
                    expected_identity=expected_identity,
                    context=context,
                )
            except Exception as exc:
                secondary_failures.append((f"{context} cleanup", exc))
        _surface_secondary_failures(
            secondary_failures,
            primary_error=primary_error,
            child_returncode=child_returncode,
        )

    if child_returncode is None or completed is None:
        raise SealedLaunchError("sealed child completed without a return code")
    if capture_profile and child_returncode == 0:
        assert capture_output_snapshot is not None
        assert capture_receipt_snapshot is not None
        assert capture_hub_cache_identity is not None
        _verify_capture_hub_cache_identity(
            dataset_cache_root,
            expected_identity=capture_hub_cache_identity,
        )
        final_bindings, final_source, _runner_path = _verify_bound_artifacts(
            runner_options,
            runtime_manifest_path=args.runtime_manifest,
        )
        if final_bindings != bindings or final_source != source_manifest:
            raise SealedLaunchError("bound artifacts changed after owned-root cleanup")
        if capture_phase == "stage_a":
            final_binding_bytes, final_binding_envelope = _verify_stage_a_capture_binding(
                runner_options,
                execution_bindings=final_bindings,
                source_commit=str(final_source["source_commit"]),
            )
            if (
                final_binding_bytes != stage_a_binding_bytes
                or final_binding_envelope != stage_a_binding_envelope
            ):
                raise SealedLaunchError(
                    "Stage-A calibration binding changed after owned-root cleanup"
                )
        final_cache_root = _verified_dataset_cache_root(
            dataset_cache_root,
            runtime_roots=(base, *packages.values()),
        )
        if (
            _non_link_directory_identity_chain(
                final_cache_root,
                context="dataset cache root",
            )
            != dataset_cache_identity
        ):
            raise SealedLaunchError("dataset cache root changed after owned-root cleanup")
        _verify_runtime(
            runtime_manifest,
            base_runtime_root=base,
            package_roots=packages,
            git_executable_path=git_executable,
            require_current_process=False,
        )
        identity_output = _revalidate_capture_artifact_snapshot(
            capture_output_snapshot,
            context=f"{capture_context} identity output",
            expect_absent=False,
        )
        _revalidate_capture_artifact_snapshot(
            capture_receipt_snapshot,
            context=f"{capture_context} identity capture provenance receipt",
            expect_absent=True,
        )
        identity_bytes = _stable_file_bytes(
            identity_output,
            context=f"published {capture_context} identity input",
        )
        candidate = completed.stdout
        if not isinstance(candidate, bytes):
            raise SealedLaunchError("sealed capture emitted no provenance candidate")
        _validate_capture_provenance_candidate(
            candidate,
            bindings=bindings,
            identity_input_file_sha256=_sha256_bytes(identity_bytes),
            runtime_manifest=runtime_manifest,
            source_manifest=source_manifest,
            phase=capture_phase,
            stage_a_binding_envelope=stage_a_binding_envelope,
        )
        identity_output = _revalidate_capture_artifact_snapshot(
            capture_output_snapshot,
            context=f"{capture_context} identity output",
            expect_absent=False,
        )
        repeated_identity_bytes = _stable_file_bytes(
            identity_output,
            context=f"final published {capture_context} identity input",
        )
        if repeated_identity_bytes != identity_bytes:
            raise SealedLaunchError(
                f"published {capture_context} identity changed before receipt publication"
            )
        _revalidate_capture_artifact_snapshot(
            capture_output_snapshot,
            context=f"{capture_context} identity output",
            expect_absent=False,
        )
        _verify_capture_hub_cache_identity(
            dataset_cache_root,
            expected_identity=capture_hub_cache_identity,
        )
        _atomic_publish_capture_receipt(capture_receipt_snapshot, candidate)
        published_receipt = _revalidate_capture_artifact_snapshot(
            capture_receipt_snapshot,
            context=f"{capture_context} identity capture provenance receipt",
            expect_absent=False,
        )
        published_candidate = _stable_file_bytes(
            published_receipt,
            context=f"published {capture_context} identity capture provenance receipt",
        )
        if published_candidate != candidate:
            raise SealedLaunchError("published capture provenance receipt bytes changed")
        _validate_capture_provenance_candidate(
            published_candidate,
            bindings=bindings,
            identity_input_file_sha256=_sha256_bytes(identity_bytes),
            runtime_manifest=runtime_manifest,
            source_manifest=source_manifest,
            phase=capture_phase,
            stage_a_binding_envelope=stage_a_binding_envelope,
        )
        _verify_capture_hub_cache_identity(
            dataset_cache_root,
            expected_identity=capture_hub_cache_identity,
        )
        post_publication_identity = _revalidate_capture_artifact_snapshot(
            capture_output_snapshot,
            context=f"post-publication {capture_context} identity output",
            expect_absent=False,
        )
        post_publication_identity_bytes = _stable_file_bytes(
            post_publication_identity,
            context=f"post-publication {capture_context} identity input",
        )
        if post_publication_identity_bytes != identity_bytes or _sha256_bytes(
            post_publication_identity_bytes
        ) != _sha256_bytes(identity_bytes):
            raise SealedLaunchError(
                f"published {capture_context} identity changed after receipt publication"
            )
        print(
            _canonical_json_bytes(
                {
                    "capture_provenance_receipt_file_sha256": _sha256_bytes(candidate),
                    "identity_input_file_sha256": _sha256_bytes(identity_bytes),
                    "runner_revision": RUNNER_REVISION,
                    "status": (
                        "captured_stage_a_identity_with_launcher_finalization"
                        if capture_phase == "stage_a"
                        else "captured_calibration_identity_with_launcher_finalization"
                    ),
                }
            ).decode("utf-8"),
            end="",
        )
    return child_returncode


def launch(argv: Sequence[str]) -> int:
    """Run through the sole public entry point, which always acquires custody."""

    return _launch_with_custody(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return launch(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
