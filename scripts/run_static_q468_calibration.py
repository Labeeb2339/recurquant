#!/usr/bin/env python3
"""Run the fail-closed Experiment 013 static-Q468 calibration.

The orchestration core is deliberately separated from the live Qwen adapter.
It authenticates the promoted calibration identity, repository source, exact
local model files, and every materialized token sequence before a model is
loaded.  A live adapter must expose one-token causal observations; this module
does not silently fall back to chunked inference or an unauthenticated Hub
download.

The command line accepts only the reviewed Qwen3.5 adapter. Generic adapters
remain test-injection surfaces. Unit tests exercise the complete ordering, EMA,
anchor, output, and failure semantics with small in-memory adapters and never
load model weights or data.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final, Protocol, cast

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
IDENTITY_RESOLVER_PATH: Final = REPOSITORY_ROOT / "scripts" / "resolve_static_q468_identity.py"
IDENTITY_RESOLVER_MODULE: Final = "recurquant_experiment013_identity_resolver"
CALIBRATION_API_MODULE: Final = "recurquant.experiment013_calibration_api"
CALIBRATION_API_PATH: Final = "src/recurquant/experiment013_calibration_api.py"
CALIBRATION_REQUIREMENTS_PATH: Final = "requirements/experiment013-calibration.txt"
SOURCE_VERIFIER_PATH: Final = "src/recurquant/experiment013_source.py"
SOURCE_CAPTURE_MODULE: Final = "recurquant_experiment013_source_capture"
CALIBRATION_IDENTITY_CAPTURE_MODULE: Final = "recurquant_experiment013_calibration_identity_capture"
CALIBRATION_IDENTITY_CAPTURE_RUNNER_MODULE: Final = (
    "_recurquant_experiment013_calibration_runner_for_capture"
)
CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE: Final = "recurquant.experiment013_source"
CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE: Final = "recurquant.experiment013_parquet"
CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH: Final = "scripts/capture_static_q468_identity_input.py"
PARQUET_SOURCE_PATH: Final = "src/recurquant/experiment013_parquet.py"
MODEL_STAGING_SOURCE_MODULE: Final = "recurquant_experiment013_source_for_model_staging"
MODEL_STAGING_RESOLVER_MODULE: Final = "recurquant_experiment013_resolver_for_model_staging"
RUNNER_SOURCE_PATH: Final = "scripts/run_static_q468_calibration.py"
IDENTITY_RESOLVER_SOURCE_PATH: Final = "scripts/resolve_static_q468_identity.py"
CANONICAL_ADAPTER_SPEC: Final = "recurquant.experiment013_qwen35_adapter:create_adapter"
CANONICAL_ADAPTER_MODULE: Final = "recurquant.experiment013_qwen35_adapter"
CANONICAL_ADAPTER_PATH: Final = "src/recurquant/experiment013_qwen35_adapter.py"

RUNNER_REVISION: Final = "experiment-013-static-q468-calibration-runner-v13"
FROZEN_IDENTITY_SCHEMA_VERSION: Final = 5
FISHER_BOUNDARY_SCHEMA: Final = "recurquant.experiment013.fisher-boundary.v1"
FISHER_BOUNDARY_NAMESPACE: Final = b"recurquant.experiment013.fisher-boundary.v1\0"
FISHER_BOUNDARY_HORIZON: Final = 1
FISHER_BOUNDARY_FIELDS: Final = {
    "boundary_positions",
    "fisher_boundary_sha256",
    "horizon",
    "input_positions",
    "input_token_ids_sha256",
    "schema",
    "target_positions",
    "target_token_ids_sha256",
}
MODEL_FILE_MANIFEST_KIND: Final = "recurquant_experiment013_model_file_manifest"
MODEL_FILE_MANIFEST_SCHEMA: Final = 1
MODEL_FILE_MANIFEST_DERIVATION: Final = "huggingface-hub-pinned-tree-lfs-v1"
MODEL_FILE_SELECTION_PROFILE: Final = "qwen35-config-index-safetensors-v1"
FROZEN_IDENTITY_CONTRACT_KIND: Final = (
    "recurquant_experiment013_frozen_identity_contract_verification"
)
FROZEN_IDENTITY_CONTRACT_SCHEMA: Final = 2
MODEL_STAGING_AUTHORIZATION_KIND: Final = "recurquant_experiment013_model_staging_authorization"
MODEL_STAGING_AUTHORIZATION_SCHEMA: Final = 2
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_calibration_identity_capture_provenance"
)
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA: Final = 2
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS: Final = (
    "captured_under_authenticated_runtime_and_launcher_finalized"
)
CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT: Final = (
    "sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1"
)
CALIBRATION_IDENTITY_CAPTURE_VERSION: Final = 6
CALIBRATION_IDENTITY_INPUT_SCHEMA: Final = "recurquant.experiment013.identity-input.v5"
STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_stage_a_identity_capture_provenance"
)
STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA: Final = 1
STAGE_A_CALIBRATION_BINDING_SCHEMA: Final = 4
STAGE_A_CALIBRATION_BINDING_REVISION: Final = "experiment-013-stage-a-calibration-binding-v4"
MODEL_STAGING_PATHS_KIND: Final = "recurquant_experiment013_model_staging_paths_verification"
MODEL_STAGING_PATHS_SCHEMA: Final = 1
RUNTIME_MANIFEST_KIND: Final = "recurquant_experiment013_calibration_runtime_manifest"
RUNTIME_MANIFEST_SCHEMA: Final = 6
OFFICIAL_DATASETS_DISTRIBUTION_VERSION: Final = "4.8.5"
RUN_REPORT_KIND: Final = "recurquant_experiment013_calibration_run"
RUN_REPORT_SCHEMA: Final = 3
CANONICAL_ADAPTER_REVISION: Final = "experiment-013-qwen35-live-adapter-v2"
CANONICAL_ADAPTER_KERNEL_BACKEND: Final = "transformers_pure_torch_gated_delta_rule"
CANONICAL_ADAPTER_MODEL_DTYPE: Final = "bfloat16"
CANONICAL_TORCH_DISTRIBUTION_VERSION: Final = "2.13.0+cu130"
CANONICAL_TORCH_RUNTIME_VERSION: Final = "2.13.0+cu130"
CANONICAL_CUDA_RUNTIME_VERSION: Final = "13.0"
CANONICAL_ADAPTER_QUERY_SHAPE: Final = (1, 1, 16, 128)
CANONICAL_ADAPTER_STATE_SHAPE: Final = (1, 16, 128, 128)
CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES: Final = (
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
CANONICAL_ADAPTER_LOADING_DIAGNOSTICS: Final = (
    "error_msgs",
    "mismatched_keys",
    "missing_keys",
    "unexpected_keys",
)

QUERY_EMA_DECAY: Final = 2.0 ** (-1.0 / 32.0)
QUERY_ENERGY_EPSILON: Final = 1.0e-6
RHT_SEED: Final = 2_339

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE: Final = re.compile(r"[0-9a-f]{40}")
_SAFE_MODEL_FILE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_MODEL_STAGING_OUTPUT_ROOT_NAME_RE: Final = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?"
)
_WINDOWS_RESERVED_BASENAMES: Final = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_WEIGHT_FILE_RE: Final = re.compile(
    r"(?:^|/)(?:model(?:-[0-9]+-of-[0-9]+)?|"
    r"model\.safetensors-[0-9]+-of-[0-9]+)\.safetensors$"
)

SCORE_FILENAME: Final = "calibration-scores.json"
COMPARATOR_SCORE_FILENAME: Final = "comparator-scores.json"
SPLIT_FILENAME: Final = "split-half-stability.json"
K27030_FILENAME: Final = "static-k27030-policy.json"
K29334_FILENAME: Final = "static-k29334-policy.json"
MSE_K29334_FILENAME: Final = "static-mse-k29334-policy.json"
FISHER_K29334_FILENAME: Final = "static-diagonal-fisher-h1-k29334-policy.json"
Q48_FILENAME: Final = "static-q48-p14739-policy.json"
CORE_BINDING_FILENAME: Final = "stage-a-calibration-core-binding.json"
BINDING_FILENAME: Final = "stage-a-calibration-binding.json"
AUTHORIZATION_FILENAME: Final = "stage-a-calibration-authorization.json"
AUTHORIZATION_COMPLETE_FILENAME: Final = "STAGE_A_CALIBRATION_AUTHORIZED"
AUTHORIZATION_COMPLETE_BYTES: Final = (
    b"recurquant-experiment013-stage-a-calibration-authorized-v1\n"
)
REPORT_FILENAME: Final = "calibration-run-report.json"
COMPLETE_FILENAME: Final = "CALIBRATION_COMPLETE"
CALIBRATION_COMPLETE_BYTES: Final = b"recurquant-experiment013-calibration-complete-v1\n"
FISHER_SMOKE_REPORT_FILENAME: Final = "fisher-h1-smoke-report.json"
FISHER_SMOKE_COMPLETE_FILENAME: Final = "FISHER_H1_SMOKE_COMPLETE"
FISHER_SMOKE_COMPLETE_BYTES: Final = b"recurquant-experiment013-fisher-h1-smoke-complete-v1\n"
RULER_RECEIPT_DIRECTORY_FILENAMES: Final = (
    "aggregation__cwe__l2048__s12340.json",
    "aggregation__cwe__l4096__s12340.json",
    "aggregation__fwe__l2048__s12339.json",
    "aggregation__fwe__l4096__s12339.json",
    "aggregation__fwe__l4096__s2343.json",
    "generation-manifest.json",
    "multi_hop_tracing__vt__l2048__s12339.json",
    "multi_hop_tracing__vt__l2048__s12340.json",
    "multi_hop_tracing__vt__l4096__s12339.json",
    "multi_hop_tracing__vt__l4096__s12340.json",
    "multi_hop_tracing__vt__l4096__s2343.json",
    "question_answering__qa_1__l2048__s12339.json",
    "question_answering__qa_1__l4096__s12339.json",
    "question_answering__qa_1__l4096__s2343.json",
    "question_answering__qa_2__l2048__s12340.json",
    "question_answering__qa_2__l4096__s12340.json",
    "retrieval__niah_multikey_2__l2048__s12340.json",
    "retrieval__niah_multiquery__l2048__s12339.json",
    "retrieval__niah_multiquery__l4096__s2343.json",
    "retrieval__niah_multivalue__l4096__s12340.json",
    "retrieval__niah_single_1__l4096__s12339.json",
)
RULER_GENERATION_MANIFEST_FILE_SHA256: Final = (
    "979f91848b6c0692160419c3e5e9ee555aa94d9e7add3092067f003ea0543e80"
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
CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES: Final = (
    "recurquant.experiment013_calibration_api",
    "recurquant.experiment013_qwen35_adapter",
    "torch",
    "transformers.modeling_utils",
)
CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES: Final = ("torch",)
CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES: Final = (
    "pkg_resources",
    "setuptools",
)
PREPARED_RUNTIME_MANIFEST_FILENAME: Final = "calibration-runtime-manifest.json"
PREPARED_RUNTIME_COMPLETE_FILENAME: Final = "RUNTIME_PREPARED"
DEFAULT_PACKAGE_RUNTIME_ROOT_NAME: Final = "calibration-packages"
DEFAULT_PACKAGE_IMPORT_PATH: Final = "Lib/site-packages"
STAGING_EXCLUDED_DISTRIBUTIONS: Final = frozenset({"pip", "setuptools"})

_WINDOWS_REPARSE_POINT: Final = 0x400
_RUNTIME_ROOT_NAME_RE: Final = re.compile(r"[a-z][a-z0-9-]{0,63}")
BASE_RUNTIME_ROOT_NAME: Final = "base-runtime"
_FORBIDDEN_RUNTIME_SUFFIXES: Final = frozenset({"._pth", ".egg-link", ".pth", ".pyc", ".pyo"})
_FORBIDDEN_RUNTIME_DIRECTORY_NAMES: Final = frozenset({"__pycache__"})
_FORBIDDEN_RUNTIME_FILENAMES: Final = frozenset(
    {"pyvenv.cfg", "sitecustomize.py", "usercustomize.py"}
)
SEALED_LAUNCH_POLICY: Final = {
    "bootstrap_mode": "stdlib-only-exact-runner-and-capture-v3",
    "cache_confinement_mode": (
        "private-scratch-plus-explicit-dataset-and-capture-hub-root-v2"
    ),
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


class CalibrationRunError(RuntimeError):
    """Raised when the authenticated calibration cannot safely continue."""


class CalibrationStabilityFailure(CalibrationRunError):
    """Raised after publishing a failure-only report for an unstable policy."""


_AUTHENTICATED_IDENTITY_RESOLVER: ModuleType | None = None
_AUTHENTICATED_SOURCE_VERIFIER: ModuleType | None = None
_AUTHENTICATED_CALIBRATION_API: ModuleType | None = None


def canonical_json_bytes(value: object) -> bytes:
    """Return the newline-terminated canonical JSON used by run evidence."""

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_bytes(data: bytes, *, context: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{context} must be bytes")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_typed_mapping(
    value: object,
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{context} fields differ from the frozen schema")
    if any(
        type(value[name]) is not type(expected[name]) or value[name] != expected[name]
        for name in expected
    ):
        raise ValueError(f"{context} value or JSON type drifted")


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _git_revision(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an immutable lowercase 40-hex revision")
    return value


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _canonical_nonnegative_float_hex(value: object, *, context: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a canonical hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{context} must be a canonical hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or value.startswith("-") or parsed.hex() != value:
        raise ValueError(f"{context} must be finite, non-negative, and canonical")
    return parsed


@dataclass(frozen=True, slots=True)
class BootstrapIdentityBindings:
    repository_source_manifest_file_sha256: str
    runtime_manifest_file_sha256: str
    model_file_manifest_file_sha256: str
    parquet_materialization_manifest_file_sha256: str
    identity_input_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class BootstrapSource:
    manifest: dict[str, object]
    source_commit: str
    entries: dict[str, dict[str, object]]


def _bootstrap_identity_bindings(data: bytes) -> BootstrapIdentityBindings:
    """Strictly extract only the v5 execution bindings using stdlib code.

    Full semantic decoding remains the authenticated resolver's job.  This
    minimal pass exists solely to authenticate the code that performs it.
    """

    root = _strict_json_bytes(data, context="frozen calibration identity bootstrap")
    _exact_fields(
        root,
        {"canonical_evidence_sha256", "evidence"},
        context="frozen calibration identity bootstrap wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise CalibrationRunError("frozen calibration identity bootstrap bytes are not canonical")
    evidence = root["evidence"]
    if not isinstance(evidence, dict):
        raise CalibrationRunError("frozen calibration identity bootstrap evidence is missing")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != FROZEN_IDENTITY_SCHEMA_VERSION
        or evidence.get("status") != "frozen"
        or evidence.get("phase") != "calibration"
        or evidence.get("identity_only") is not True
        or evidence.get("promotion_required") is not False
    ):
        raise CalibrationRunError("frozen calibration identity bootstrap state is invalid")
    canonical_evidence_sha256 = _sha256(
        root["canonical_evidence_sha256"],
        context="frozen identity canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise CalibrationRunError("frozen identity canonical evidence SHA-256 drifted")
    bindings = evidence.get("execution_bindings")
    if not isinstance(bindings, dict):
        raise CalibrationRunError("frozen identity execution bindings are missing")
    _exact_fields(
        bindings,
        {
            "calibration_runtime_manifest_file_sha256",
            "model_file_manifest_file_sha256",
            "parquet_materialization_manifest_file_sha256",
            "repository_source_manifest_file_sha256",
        },
        context="frozen identity bootstrap execution bindings",
    )
    return BootstrapIdentityBindings(
        repository_source_manifest_file_sha256=_sha256(
            bindings["repository_source_manifest_file_sha256"],
            context="bootstrap repository source manifest file SHA-256",
        ),
        runtime_manifest_file_sha256=_sha256(
            bindings["calibration_runtime_manifest_file_sha256"],
            context="bootstrap runtime manifest file SHA-256",
        ),
        model_file_manifest_file_sha256=_sha256(
            bindings["model_file_manifest_file_sha256"],
            context="bootstrap model manifest file SHA-256",
        ),
        parquet_materialization_manifest_file_sha256=_sha256(
            bindings["parquet_materialization_manifest_file_sha256"],
            context="bootstrap parquet materialization manifest file SHA-256",
        ),
        identity_input_manifest_sha256=_sha256(
            evidence.get("source_manifest_sha256"),
            context="bootstrap identity input manifest SHA-256",
        ),
    )


def _source_manifest_canonical_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("canonical_manifest_sha256", None)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def _canonical_relative_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CalibrationRunError(f"{context} must be a non-empty canonical path")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise CalibrationRunError(f"{context} must be a single-line POSIX path")
    path = PurePosixPath(value)
    parts = path.parts
    if (
        not parts
        or path.is_absolute()
        or parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in parts)
        or path.as_posix() != value
    ):
        raise CalibrationRunError(f"{context} must be repository-relative")
    return value


def _canonical_base_sys_path_entry(value: object, *, context: str) -> str:
    """Accept only the runtime-root sentinel beyond canonical relative paths."""

    if isinstance(value, str) and value == ".":
        return "."
    return _canonical_relative_path(value, context=context)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise CalibrationRunError(f"required path is unavailable: {path}") from exc
    return path.is_symlink() or bool(
        getattr(stat_result, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _assert_no_link_components(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    if _is_link_or_reparse(candidate):
        raise CalibrationRunError(f"authenticated root is a link or reparse point: {root}")
    for part in relative.parts:
        candidate = candidate / part
        if _is_link_or_reparse(candidate):
            raise CalibrationRunError(
                f"authenticated path traverses a link or reparse point: {relative}"
            )
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CalibrationRunError(f"authenticated path escapes its root: {relative}") from exc
    if not resolved.is_file():
        raise CalibrationRunError(f"authenticated path is not a regular file: {relative}")
    return resolved


def _bootstrap_source_manifest(
    data: bytes,
    *,
    repository_root: Path,
    require_adapter: bool,
) -> BootstrapSource:
    manifest = _strict_json_bytes(data, context="repository source manifest bootstrap")
    _exact_fields(
        manifest,
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
        context="repository source manifest bootstrap",
    )
    claimed = _sha256(
        manifest["canonical_manifest_sha256"],
        context="repository source manifest canonical SHA-256",
    )
    if claimed != _source_manifest_canonical_sha256(manifest):
        raise CalibrationRunError("repository source manifest canonical SHA-256 drifted")
    if (
        manifest["schema"] != "recurquant.experiment013.source-manifest.v2"
        or manifest["profile"] != "experiment-013-static-q468-frozen-source-v2"
        or manifest["object_format"] != "sha1"
    ):
        raise CalibrationRunError("repository source manifest profile drifted")
    git_record = manifest["git_executable"]
    if not isinstance(git_record, dict):
        raise CalibrationRunError("repository source Git executable record is missing")
    _exact_fields(
        git_record,
        {"sha256", "size_bytes"},
        context="repository source Git executable",
    )
    _sha256(git_record["sha256"], context="repository source Git executable SHA-256")
    _positive_int(git_record["size_bytes"], context="repository source Git executable size")
    source_commit = _git_revision(
        manifest["source_commit"],
        context="repository source manifest source commit",
    )
    raw_paths = manifest["paths"]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise CalibrationRunError("repository source manifest paths must be non-empty")
    entries: dict[str, dict[str, object]] = {}
    for index, raw_entry in enumerate(raw_paths):
        if not isinstance(raw_entry, dict):
            raise CalibrationRunError(f"repository source paths[{index}] must be an object")
        _exact_fields(
            raw_entry,
            {
                "git_blob_oid",
                "index_blob_oid",
                "mode",
                "path",
                "raw_sha256",
                "worktree_blob_oid",
            },
            context=f"repository source paths[{index}]",
        )
        relative = _canonical_relative_path(
            raw_entry["path"], context=f"repository source paths[{index}].path"
        )
        if relative in entries:
            raise CalibrationRunError(f"duplicate repository source path: {relative}")
        _sha256(raw_entry["raw_sha256"], context=f"repository source {relative} SHA-256")
        entries[relative] = dict(raw_entry)
    required = {
        CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
        RUNNER_SOURCE_PATH,
        IDENTITY_RESOLVER_SOURCE_PATH,
        SOURCE_VERIFIER_PATH,
        PARQUET_SOURCE_PATH,
        CALIBRATION_API_PATH,
        CALIBRATION_REQUIREMENTS_PATH,
    }
    if require_adapter:
        required.add(CANONICAL_ADAPTER_PATH)
    missing = sorted(required - set(entries))
    if missing:
        raise CalibrationRunError(f"repository source manifest omits bootstrap paths: {missing}")
    root = Path(os.path.abspath(repository_root))
    if not root.is_dir():
        raise CalibrationRunError("repository root is unavailable")
    for relative in sorted(required):
        source_path = _assert_no_link_components(root, PurePosixPath(relative))
        digest, _size = _stream_file_sha256(source_path)
        if digest != entries[relative]["raw_sha256"]:
            raise CalibrationRunError(f"bootstrap source bytes drifted: {relative}")
    runner_path = _assert_no_link_components(root, PurePosixPath(RUNNER_SOURCE_PATH))
    if Path(__file__).resolve(strict=True) != runner_path:
        raise CalibrationRunError("executing runner is not the authenticated repository runner")
    return BootstrapSource(manifest=manifest, source_commit=source_commit, entries=entries)


def _load_exact_source_module(
    module_name: str,
    relative_path: str,
    *,
    repository_root: Path,
    entry: Mapping[str, object],
) -> ModuleType:
    if module_name in sys.modules:
        raise CalibrationRunError(f"refusing preloaded authenticated module: {module_name}")
    source_path = _assert_no_link_components(
        Path(os.path.abspath(repository_root)), PurePosixPath(relative_path)
    )
    expected_sha256 = _sha256(entry.get("raw_sha256"), context=f"{module_name} source SHA-256")
    source_bytes = _read_stable_regular_bytes(
        source_path,
        context=f"authenticated module source {module_name}",
    )
    actual_sha256 = sha256_bytes(source_bytes)
    if actual_sha256 != expected_sha256:
        raise CalibrationRunError(
            f"authenticated module bytes drifted before import: {module_name}"
        )
    try:
        code = compile(source_bytes, str(source_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise CalibrationRunError(f"cannot compile authenticated source: {module_name}") from exc
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        origin=str(source_path),
    )
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
        declared = Path(cast(str, getattr(module, "__file__", ""))).resolve(strict=True)
        after_sha256, _after_size = _stream_file_sha256(source_path)
        if declared != source_path or after_sha256 != expected_sha256:
            raise CalibrationRunError(
                f"authenticated module identity drifted on import: {module_name}"
            )
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_source_capture_module(repository_root: Path) -> ModuleType:
    """Load the committed source-manifest implementation without package import hooks."""

    if SOURCE_CAPTURE_MODULE in sys.modules:
        raise CalibrationRunError("refusing a preloaded source-manifest capture module")
    root = Path(os.path.abspath(repository_root))
    source_path = _assert_no_link_components(root, PurePosixPath(SOURCE_VERIFIER_PATH))
    source_bytes = source_path.read_bytes()
    before_sha256 = sha256_bytes(source_bytes)
    before_size = len(source_bytes)
    try:
        code = compile(source_bytes, str(source_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise CalibrationRunError("cannot compile source-manifest implementation") from exc
    module = ModuleType(SOURCE_CAPTURE_MODULE)
    module.__file__ = str(source_path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        SOURCE_CAPTURE_MODULE,
        loader=None,
        origin=str(source_path),
    )
    sys.modules[SOURCE_CAPTURE_MODULE] = module
    try:
        exec(code, module.__dict__)
        declared = Path(cast(str, getattr(module, "__file__", ""))).resolve(strict=True)
        after_sha256, after_size = _stream_file_sha256(source_path)
        if declared != source_path or after_sha256 != before_sha256 or after_size != before_size:
            raise CalibrationRunError("source-manifest implementation changed during import")
    except BaseException:
        sys.modules.pop(SOURCE_CAPTURE_MODULE, None)
        raise
    return module


def _sanitized_git_environment() -> dict[str, str]:
    inherited = {key.upper(): (key, value) for key, value in os.environ.items()}
    environment = {
        inherited[name][0]: inherited[name][1]
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC")
        if name in inherited
    }
    environment.update(
        {
            "GIT_AUTHOR_NAME": "RecurQuant Experiment 013",
            "GIT_AUTHOR_EMAIL": "experiment013@invalid",
            "GIT_COMMITTER_NAME": "RecurQuant Experiment 013",
            "GIT_COMMITTER_EMAIL": "experiment013@invalid",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _assert_source_manifest_output_location(
    repository_root: Path,
    output: Path,
    *,
    git_executable: AuthenticatedGitExecutable,
) -> Path:
    """Allow source manifests only outside the repository or at an ignored path."""

    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise CalibrationRunError("repository root is unavailable") from exc
    if not root.is_dir() or _is_link_or_reparse(root):
        raise CalibrationRunError("repository root is not a regular directory")
    resolved_output = Path(output).resolve(strict=False)
    if resolved_output.exists() or resolved_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {resolved_output}")
    try:
        relative = resolved_output.relative_to(root)
    except ValueError:
        return resolved_output
    if not relative.parts or relative.parts[0].casefold() == ".git":
        raise CalibrationRunError("source manifest output cannot be repository metadata")
    process = subprocess.run(
        [
            str(git_executable.path),
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative.as_posix(),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        env=_sanitized_git_environment(),
    )
    if process.returncode == 0:
        return resolved_output
    if process.returncode == 1:
        raise CalibrationRunError(
            "source manifest output inside the repository must be ignored by Git"
        )
    raise CalibrationRunError("cannot verify source manifest output ignore status")


def _torch_runtime() -> Any:
    """Import torch only after the caller has authenticated the runtime."""

    return importlib.import_module("torch")


@dataclass(frozen=True, slots=True)
class Geometry:
    """Minimal recurrent-state geometry required by the causal runner."""

    layer_indices: tuple[int, ...]
    heads: int
    key_rows: int
    value_width: int

    def __post_init__(self) -> None:
        if (
            not self.layer_indices
            or any(index < 0 for index in self.layer_indices)
            or len(set(self.layer_indices)) != len(self.layer_indices)
        ):
            raise ValueError("layer_indices must be unique non-negative integers")
        for name in ("heads", "key_rows", "value_width"):
            _positive_int(getattr(self, name), context=name)
        if self.value_width & (self.value_width - 1):
            raise ValueError("value_width must be a power of two")

    @property
    def layers(self) -> int:
        return len(self.layer_indices)

    @property
    def rows_per_layer(self) -> int:
        return self.heads * self.key_rows

    @property
    def total_rows(self) -> int:
        return self.layers * self.rows_per_layer


@dataclass(frozen=True, slots=True)
class FrozenCalibrationIdentity:
    """Runner-facing view of the resolver's strictly decoded frozen identity."""

    file_sha256: str
    canonical_evidence_sha256: str
    records: tuple[dict[str, object], ...]
    assignment: tuple[dict[str, object], ...]
    assignment_sha256: str
    tokenizer_manifest_sha256: str
    identity_input_manifest_sha256: str
    repository_source_manifest_file_sha256: str
    runtime_manifest_file_sha256: str
    model_file_manifest_file_sha256: str
    parquet_materialization_manifest_file_sha256: str
    model_id: str
    model_revision: str
    transformers_version: str
    artifact_bytes: bytes


@dataclass(frozen=True, slots=True)
class CapturedSequence:
    """One causal pass worth of post-token and H=1 endpoint tensors."""

    anchor_positions: tuple[int, ...]
    query_energy: Any
    q4_mse: Any
    q6_mse: Any
    q8_mse: Any
    fisher_boundary_positions: tuple[int, ...]
    fisher_q4_risk: Any
    fisher_q6_risk: Any
    fisher_q8_risk: Any
    fisher_target_nlls: Any


@dataclass(frozen=True, slots=True)
class ReducedSequenceScores:
    """Candidate, unweighted-MSE, and H=1 Fisher scores for one identity row."""

    candidate: Any
    mse: Any
    fisher: Any


@dataclass(frozen=True, slots=True)
class ModelFileRecord:
    name: str
    size_bytes: int
    sha256: str | None
    git_blob_oid: str
    lfs_sha256: str | None
    lfs_size_bytes: int | None


@dataclass(frozen=True, slots=True)
class ModelFileManifest:
    model_id: str
    revision: str
    transformers_version: str
    files: tuple[ModelFileRecord, ...]
    hub_tree_manifest_sha256: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ModelStagingAuthorization:
    identity: FrozenCalibrationIdentity
    model_manifest: ModelFileManifest
    frozen_identity_file_sha256: str
    capture_provenance_receipt_file_sha256: str
    identity_commit: str
    source_commit: str


@dataclass(frozen=True, slots=True)
class DirectoryComponentIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True, slots=True)
class NewCaptureArtifactPath:
    path: Path
    parent: Path
    parent_component_identities: tuple[DirectoryComponentIdentity, ...]


@dataclass(frozen=True, slots=True)
class ModelStagingPaths:
    repository_root: Path
    repository_component_identities: tuple[DirectoryComponentIdentity, ...]
    hub_cache_root: Path
    hub_cache_component_identities: tuple[DirectoryComponentIdentity, ...]
    output_root: Path
    output_parent_component_identities: tuple[DirectoryComponentIdentity, ...]


@dataclass(frozen=True, slots=True)
class FrozenIdentitySourceAuthorization:
    identity: FrozenCalibrationIdentity
    bindings: BootstrapIdentityBindings
    identity_bytes: bytes
    frozen_identity_file_sha256: str
    source_commit: str


@dataclass(frozen=True, slots=True)
class RuntimeFileRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeDistributionRecord:
    name: str
    version: str
    package_root: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeTreeRecord:
    name: str
    kind: str
    files: tuple[RuntimeFileRecord, ...]


@dataclass(frozen=True, slots=True)
class RuntimePackageRootRecord:
    name: str
    import_path: str


@dataclass(frozen=True, slots=True)
class RuntimeInterpreterProbe:
    python_implementation: str
    python_version: str
    python_cache_tag: str
    python_abi_flags: str
    machine_system: str
    machine_architecture: str
    machine_name: str
    machine_byteorder: str
    machine_pointer_bits: int
    base_sys_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class AuthenticatedGitExecutable:
    path: Path
    absolute_path_sha256: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CalibrationRuntimeManifest:
    python_implementation: str
    python_version: str
    python_cache_tag: str
    python_abi_flags: str
    machine_system: str
    machine_architecture: str
    machine_name: str
    machine_byteorder: str
    machine_pointer_bits: int
    launch_policy: Mapping[str, object]
    base_sys_path: tuple[str, ...]
    base_runtime_root: str
    package_roots: tuple[RuntimePackageRootRecord, ...]
    interpreter_root: str
    interpreter_relative_path: str
    interpreter_size_bytes: int
    interpreter_sha256: str
    git_executable_absolute_path_sha256: str
    git_executable_sha256: str
    git_executable_size_bytes: int
    runtime_trees: tuple[RuntimeTreeRecord, ...]
    distributions: tuple[RuntimeDistributionRecord, ...]
    file_sha256: str


@dataclass(frozen=True, slots=True)
class AuthenticatedRuntime:
    manifest_file_sha256: str
    python_implementation: str
    python_version: str
    python_cache_tag: str
    interpreter_sha256: str
    git_executable_absolute_path_sha256: str
    git_executable_sha256: str
    git_executable_size_bytes: int
    machine_name: str
    base_runtime_file_count: int
    package_root_count: int
    distributions: tuple[tuple[str, str], ...]
    distribution_count: int
    file_count: int


@dataclass(frozen=True, slots=True)
class SealedRuntimeContext:
    manifest_file_sha256: str
    base_runtime_root: Path
    package_roots: Mapping[str, Path]
    package_import_paths: Mapping[str, str]
    git_executable_path: Path
    pycache_prefix: Path


@dataclass(frozen=True, slots=True)
class AuthenticatedCaptureSix:
    """Exact six module/importer state admitted before capture isolation."""

    module: ModuleType
    module_spec: object
    importer: object
    importer_type: type
    origin_path: Path
    origin_sha256: str
    origin_size_bytes: int
    meta_path: list[object]
    topology_before_import: tuple[object, ...]
    topology_with_importer: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    """In-memory pass artifacts; no publication occurs until all are built."""

    score: bytes
    comparator_score: bytes
    split_half: bytes
    static_k27030: bytes
    static_k29334: bytes
    static_mse_k29334: bytes
    static_fisher_k29334: bytes
    static_q48: bytes
    calibration_core_binding: bytes
    stability: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    passed: bool
    stability: Mapping[str, object]
    artifacts: CalibrationArtifacts | None


@dataclass(frozen=True, slots=True)
class CalibrationRunConfig:
    frozen_identity_bytes: bytes
    repository_source_manifest_bytes: bytes
    model_file_manifest_bytes: bytes
    parquet_materialization_manifest_bytes: bytes
    runtime_manifest_bytes: bytes
    model_root: Path
    repository_root: Path
    expected_source_commit: str
    expected_model_file_manifest_sha256: str
    expected_parquet_materialization_manifest_sha256: str
    expected_runtime_manifest_sha256: str
    capture_provenance_receipt_bytes: bytes
    expected_capture_provenance_receipt_sha256: str
    output_dir: Path
    require_cuda: bool = True
    fisher_h1_smoke: bool = False
    prior_fisher_h1_smoke_report_bytes: bytes | None = None
    prior_fisher_h1_smoke_complete_bytes: bytes | None = None


class CalibrationBackend(Protocol):
    geometry: Geometry

    def decode_identity(self, data: bytes) -> FrozenCalibrationIdentity: ...

    def reduce_sequence(
        self,
        record: Mapping[str, object],
        token_ids: tuple[int, ...],
        captured: CapturedSequence,
    ) -> object: ...

    def finalize(
        self,
        scores: Sequence[object],
        *,
        identity: FrozenCalibrationIdentity,
        source_commit: str,
    ) -> FinalizationResult: ...


SourceVerifier = Callable[[Mapping[str, object], Path], tuple[dict[str, object], str]]
AdapterValidator = Callable[[Any], None]
DistortionFunction = Callable[[Any, Geometry], tuple[Any, Any, Any]]
FisherDistortionFunction = Callable[[Any, Any, Geometry], tuple[Any, Any, Any]]
ModelAuthenticator = Callable[[Path, ModelFileManifest], Any]
RuntimeAuthenticator = Callable[[CalibrationRuntimeManifest], AuthenticatedRuntime]


@dataclass(frozen=True, slots=True)
class RunnerServices:
    backend: CalibrationBackend
    calibration_api: ModuleType
    identity_resolver: Any | None
    verify_repository_source: SourceVerifier
    validate_adapter: AdapterValidator
    distortion_function: DistortionFunction
    fisher_distortion_function: FisherDistortionFunction
    authenticate_model_files: ModelAuthenticator
    authenticate_runtime: RuntimeAuthenticator


def _selected_model_tree_path(path: str) -> bool:
    if "/" in path:
        return False
    return path in {"config.json", "model.safetensors.index.json"} or bool(
        _WEIGHT_FILE_RE.fullmatch(path)
    )


def parse_model_file_manifest(data: bytes) -> ModelFileManifest:
    """Strictly decode a canonical, immutable local-model file manifest."""

    root = _strict_json_bytes(data, context="model file manifest")
    _exact_fields(
        root,
        {
            "artifact_kind",
            "files",
            "hub_tree_manifest_sha256",
            "metadata_derivation",
            "selection_profile",
            "model_id",
            "revision",
            "schema_version",
            "transformers_version",
        },
        context="model file manifest",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("model file manifest is not canonical newline-terminated JSON")
    if (
        root["artifact_kind"] != MODEL_FILE_MANIFEST_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != MODEL_FILE_MANIFEST_SCHEMA
    ):
        raise ValueError("model file manifest kind or schema drifted")
    if root["metadata_derivation"] != MODEL_FILE_MANIFEST_DERIVATION:
        raise ValueError("model file manifest metadata derivation drifted")
    if root["selection_profile"] != MODEL_FILE_SELECTION_PROFILE:
        raise ValueError("model file manifest selection profile drifted")
    model_id = root["model_id"]
    transformers_version = root["transformers_version"]
    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ValueError("model file manifest model_id is invalid")
    revision = _git_revision(root["revision"], context="model file manifest revision")
    if not isinstance(transformers_version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", transformers_version
    ):
        raise ValueError("model file manifest Transformers version must be exact semver")
    raw_files = root["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("model file manifest files must be a non-empty list")
    files: list[ModelFileRecord] = []
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise ValueError(f"model file manifest files[{index}] must be an object")
        _exact_fields(
            item,
            {
                "git_blob_oid",
                "lfs_sha256",
                "lfs_size_bytes",
                "name",
                "sha256",
                "size_bytes",
            },
            context=f"files[{index}]",
        )
        name = item["name"]
        if not isinstance(name, str) or _SAFE_MODEL_FILE_RE.fullmatch(name) is None:
            raise ValueError(f"model file manifest files[{index}].name is invalid")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in name:
            raise ValueError("model file names must be canonical relative POSIX paths")
        size_bytes = _positive_int(item["size_bytes"], context=f"files[{index}].size_bytes")
        raw_file_sha256 = item["sha256"]
        git_blob_oid = _git_revision(item["git_blob_oid"], context=f"files[{index}].git_blob_oid")
        raw_lfs_sha256 = item["lfs_sha256"]
        raw_lfs_size = item["lfs_size_bytes"]
        if raw_lfs_sha256 is None and raw_lfs_size is None:
            lfs_sha256 = None
            lfs_size_bytes = None
            if raw_file_sha256 is not None:
                raise ValueError("ordinary Git blobs must use null SHA-256 and their Git blob OID")
            file_sha256 = None
        elif raw_lfs_sha256 is not None and raw_lfs_size is not None:
            lfs_sha256 = _sha256(raw_lfs_sha256, context=f"files[{index}].lfs_sha256")
            file_sha256 = _sha256(raw_file_sha256, context=f"files[{index}].sha256")
            lfs_size_bytes = _positive_int(
                raw_lfs_size,
                context=f"files[{index}].lfs_size_bytes",
            )
            if lfs_sha256 != file_sha256 or lfs_size_bytes != size_bytes:
                raise ValueError("model LFS identity must equal the local content contract")
        else:
            raise ValueError("model LFS SHA-256 and size must either both be null or both be set")
        is_weight = _WEIGHT_FILE_RE.fullmatch(name) is not None
        if is_weight and lfs_sha256 is None:
            raise ValueError("safetensors weights require a pinned Hub LFS identity")
        if not is_weight and lfs_sha256 is not None:
            raise ValueError("model config and index files must be ordinary Git blobs")
        files.append(
            ModelFileRecord(
                name=name,
                size_bytes=size_bytes,
                sha256=file_sha256,
                git_blob_oid=git_blob_oid,
                lfs_sha256=lfs_sha256,
                lfs_size_bytes=lfs_size_bytes,
            )
        )
    names = [item.name for item in files]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("model file manifest names must be unique and sorted")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("model file manifest names contain a case-insensitive collision")
    if any(not _selected_model_tree_path(name) for name in names):
        raise ValueError("model file manifest contains a file outside its selection profile")
    if "config.json" not in names:
        raise ValueError("model file manifest must authenticate config.json")
    if "model.safetensors.index.json" not in names:
        raise ValueError("model file manifest must authenticate model.safetensors.index.json")
    if not any(_WEIGHT_FILE_RE.fullmatch(name) for name in names):
        raise ValueError(
            "model file manifest must authenticate at least one safetensors weight file"
        )
    tree_payload = [
        {
            "git_blob_oid": item.git_blob_oid,
            "lfs_sha256": item.lfs_sha256,
            "lfs_size_bytes": item.lfs_size_bytes,
            "name": item.name,
        }
        for item in files
    ]
    hub_tree_manifest_sha256 = _sha256(
        root["hub_tree_manifest_sha256"],
        context="model Hub tree manifest SHA-256",
    )
    if hub_tree_manifest_sha256 != sha256_bytes(canonical_json_bytes(tree_payload)):
        raise ValueError("model Hub tree metadata manifest SHA-256 drifted")
    return ModelFileManifest(
        model_id=model_id,
        revision=revision,
        transformers_version=transformers_version,
        files=tuple(files),
        hub_tree_manifest_sha256=hub_tree_manifest_sha256,
        file_sha256=sha256_bytes(data),
    )


def _hub_value(value: object, name: str, *, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def capture_model_file_manifest_from_hub(
    model_id: str,
    revision: str,
    *,
    transformers_version: str,
    api: object | None = None,
    tree_entries: Sequence[object] | None = None,
    resolved_revision: str | None = None,
    token: bool = False,
) -> bytes:
    """Build the local-file contract using only pinned Hub tree/LFS metadata.

    The function never downloads or opens model files. Ordinary files are
    authenticated by their Git blob OID. LFS files additionally bind the
    content SHA-256 and byte size advertised by the pinned Hub revision.
    """

    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ValueError("model_id must be a non-empty canonical string")
    if token is not False:
        raise CalibrationRunError("Experiment 013 Hub metadata access must be unauthenticated")
    pinned_revision = _git_revision(revision, context="model Hub revision")
    if (
        not isinstance(transformers_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", transformers_version) is None
    ):
        raise ValueError("Transformers version must be exact semver")
    if tree_entries is None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=False, endpoint="https://huggingface.co")
        info = api.model_info(  # type: ignore[attr-defined]
            model_id,
            revision=pinned_revision,
            files_metadata=False,
            token=False,
        )
        resolved = _git_revision(
            getattr(info, "sha", None),
            context="resolved Hub model revision",
        )
        if resolved != pinned_revision:
            raise CalibrationRunError("Hub resolved a different immutable model revision")
        tree_entries = tuple(
            api.list_repo_tree(  # type: ignore[attr-defined]
                model_id,
                recursive=True,
                expand=False,
                revision=pinned_revision,
                repo_type="model",
                token=False,
            )
        )
    else:
        resolved = _git_revision(
            resolved_revision,
            context="mocked resolved Hub model revision",
        )
        if resolved != pinned_revision:
            raise CalibrationRunError("Hub tree metadata revision differs from the pinned revision")

    selected: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(tree_entries):
        raw_path = _hub_value(entry, "path")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or "\0" in raw_path:
            raise ValueError(f"Hub tree entry {index} has an invalid path")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("Hub tree paths must be canonical repository-relative POSIX paths")
        if raw_path in seen_paths:
            raise ValueError(f"Hub tree metadata contains duplicate path: {raw_path}")
        seen_paths.add(raw_path)
        if not _selected_model_tree_path(raw_path):
            continue
        size_bytes = _positive_int(
            _hub_value(entry, "size"),
            context=f"Hub tree {raw_path} size",
        )
        git_blob_oid = _git_revision(
            _hub_value(entry, "blob_id"),
            context=f"Hub tree {raw_path} Git blob OID",
        )
        raw_lfs = _hub_value(entry, "lfs")
        if raw_lfs is None:
            lfs_sha256 = None
            lfs_size_bytes = None
            file_sha256 = None
        else:
            lfs_sha256 = _sha256(
                _hub_value(raw_lfs, "sha256"),
                context=f"Hub tree {raw_path} LFS SHA-256",
            )
            lfs_size_bytes = _positive_int(
                _hub_value(raw_lfs, "size"),
                context=f"Hub tree {raw_path} LFS size",
            )
            if lfs_size_bytes != size_bytes:
                raise ValueError(f"Hub tree {raw_path} size differs from LFS metadata")
            file_sha256 = lfs_sha256
        if _WEIGHT_FILE_RE.fullmatch(raw_path) and lfs_sha256 is None:
            raise ValueError(f"safetensors weight lacks pinned LFS metadata: {raw_path}")
        selected.append(
            {
                "git_blob_oid": git_blob_oid,
                "lfs_sha256": lfs_sha256,
                "lfs_size_bytes": lfs_size_bytes,
                "name": raw_path,
                "sha256": file_sha256,
                "size_bytes": size_bytes,
            }
        )
    selected.sort(key=lambda item: cast(str, item["name"]))
    names = [cast(str, item["name"]) for item in selected]
    if "config.json" not in names:
        raise ValueError("pinned Hub tree has no root config.json")
    if not any(_WEIGHT_FILE_RE.fullmatch(name) for name in names):
        raise ValueError("pinned Hub tree has no root safetensors weight files")
    tree_payload = [
        {
            "git_blob_oid": item["git_blob_oid"],
            "lfs_sha256": item["lfs_sha256"],
            "lfs_size_bytes": item["lfs_size_bytes"],
            "name": item["name"],
        }
        for item in selected
    ]
    document = {
        "artifact_kind": MODEL_FILE_MANIFEST_KIND,
        "files": selected,
        "hub_tree_manifest_sha256": sha256_bytes(canonical_json_bytes(tree_payload)),
        "metadata_derivation": MODEL_FILE_MANIFEST_DERIVATION,
        "model_id": model_id,
        "revision": pinned_revision,
        "schema_version": MODEL_FILE_MANIFEST_SCHEMA,
        "selection_profile": MODEL_FILE_SELECTION_PROFILE,
        "transformers_version": transformers_version,
    }
    payload = canonical_json_bytes(document)
    parse_model_file_manifest(payload)
    return payload


def _stream_file_sha256(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or size != after.st_size
    ):
        raise CalibrationRunError(f"model file changed while hashing: {path}")
    return digest.hexdigest(), size


def _absolute_path_sha256(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve(strict=True)))
    return sha256_bytes(normalized.encode("utf-8"))


def _assert_absolute_path_components_not_links(path: Path, *, context: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            raise CalibrationRunError(f"{context} traverses a link or reparse point")


def _authenticate_git_executable(
    executable: str | os.PathLike[str] | None,
) -> AuthenticatedGitExecutable:
    selected: str | os.PathLike[str]
    if executable is None:
        discovered = shutil.which("git")
        if discovered is None:
            raise CalibrationRunError("Git executable is unavailable")
        selected = discovered
    else:
        selected = executable
    try:
        resolved = Path(selected).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise CalibrationRunError("Git executable is unavailable") from exc
    if resolved.name.casefold() == "git.exe" and resolved.parent.name.casefold() == "cmd":
        try:
            resolved = (resolved.parent.parent / "mingw64" / "bin" / "git.exe").resolve(strict=True)
        except OSError as exc:
            raise CalibrationRunError(
                "Git-for-Windows cmd shim has no canonical mingw64 executable"
            ) from exc
    _assert_absolute_path_components_not_links(resolved, context="Git executable")
    if not resolved.is_file() or _is_link_or_reparse(resolved):
        raise CalibrationRunError("Git executable must be a regular non-link file")
    digest, size = _stream_file_sha256(resolved)
    if size <= 0:
        raise CalibrationRunError("Git executable must be non-empty")
    return AuthenticatedGitExecutable(
        path=resolved,
        absolute_path_sha256=_absolute_path_sha256(resolved),
        sha256=digest,
        size_bytes=size,
    )


def _stream_model_file_identity(path: Path) -> tuple[str, str, int]:
    before = path.stat()
    size = before.st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {size}\0".encode("ascii"))
    streamed = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
            streamed += len(chunk)
    after = path.stat()
    if (
        streamed != size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise CalibrationRunError(f"model file changed while hashing: {path}")
    return sha256.hexdigest(), git_blob.hexdigest(), size


def _verify_exact_local_model_tree(
    model_root: Path,
    manifest: ModelFileManifest,
) -> Path:
    """Authenticate one exact, regular, link-free local model tree."""

    root = Path(os.path.abspath(model_root))
    if not root.is_dir():
        raise ValueError("model_root must be a directory")
    if _is_link_or_reparse(root):
        raise CalibrationRunError("model_root must not be a symlink or reparse point")
    actual_names: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_link_or_reparse(path):
            raise CalibrationRunError(f"local model path is a link or reparse point: {relative}")
        if path.is_file():
            actual_names.append(relative)
        elif not path.is_dir():
            raise CalibrationRunError(
                f"local model path is not a regular file/directory: {relative}"
            )
    actual_names.sort()
    if len({name.casefold() for name in actual_names}) != len(actual_names):
        raise CalibrationRunError("local model file set has a case-insensitive collision")
    expected_names = [item.name for item in manifest.files]
    if actual_names != expected_names:
        raise CalibrationRunError(
            "local model file set differs from the authenticated manifest; "
            f"missing={sorted(set(expected_names) - set(actual_names))}, "
            f"extra={sorted(set(actual_names) - set(expected_names))}"
        )
    for item in manifest.files:
        candidate = _assert_no_link_components(root, PurePosixPath(item.name))
        sha256, git_blob_oid, size = _stream_model_file_identity(candidate)
        content_matches = (
            sha256 == item.lfs_sha256
            if item.lfs_sha256 is not None
            else git_blob_oid == item.git_blob_oid
        )
        if size != item.size_bytes or not content_matches:
            raise CalibrationRunError(f"local model file authentication failed: {item.name}")
    return root


def authenticate_local_model_files(
    model_root: Path,
    manifest: ModelFileManifest,
    *,
    calibration_api: ModuleType,
) -> Any:
    """Hash every exact local model file immediately before model loading."""

    root = _verify_exact_local_model_tree(model_root, manifest)
    identities = tuple(
        calibration_api.ModelFileIdentity(
            name=item.name,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
            git_blob_oid=item.git_blob_oid,
            lfs_sha256=item.lfs_sha256,
            lfs_size_bytes=item.lfs_size_bytes,
        )
        for item in manifest.files
    )
    return calibration_api.AuthenticatedModelFiles(
        model_root=root,
        model_id=manifest.model_id,
        revision=manifest.revision,
        transformers_version=manifest.transformers_version,
        files=identities,
        hub_tree_manifest_sha256=manifest.hub_tree_manifest_sha256,
        manifest_file_sha256=manifest.file_sha256,
    )


def _read_stable_regular_bytes(path: Path, *, context: str) -> bytes:
    candidate = Path(os.path.abspath(path))
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise CalibrationRunError(f"{context} is unavailable") from exc
    if _is_link_or_reparse(candidate) or not stat.S_ISREG(before.st_mode):
        raise CalibrationRunError(f"{context} must be a regular non-link file")
    try:
        data = candidate.read_bytes()
        after = candidate.lstat()
    except OSError as exc:
        raise CalibrationRunError(f"cannot read {context}") from exc
    if (
        _is_link_or_reparse(candidate)
        or not stat.S_ISREG(after.st_mode)
        or len(data) != after.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise CalibrationRunError(f"{context} changed while it was read")
    return data


def _git_blob_oid_bytes(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _run_model_staging_git(
    git_executable: AuthenticatedGitExecutable,
    repository_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(git_executable.path), "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
        env=_sanitized_git_environment(),
    )


def _one_nul_git_record(process: subprocess.CompletedProcess[bytes], *, context: str) -> bytes:
    if process.returncode != 0:
        raise CalibrationRunError(f"cannot authenticate {context}")
    records = process.stdout.split(b"\0")
    if records[-1:] != [b""] or len(records) != 2 or not records[0]:
        raise CalibrationRunError(f"{context} is not exactly one Git record")
    return records[0]


def _verify_committed_frozen_identity(
    git_executable: AuthenticatedGitExecutable,
    repository_root: Path,
    identity_path: Path,
    identity_bytes: bytes,
    *,
    identity_commit: str,
) -> str:
    """Require the promoted identity to be the exact clean blob at current HEAD."""

    root = Path(os.path.abspath(repository_root)).resolve(strict=True)
    declared = Path(os.path.abspath(identity_path))
    try:
        relative_path = declared.relative_to(root).as_posix()
    except ValueError as exc:
        raise CalibrationRunError(
            "frozen identity must be committed inside the repository"
        ) from exc
    if not relative_path or "\t" in relative_path:
        raise CalibrationRunError("frozen identity repository path is not canonical")
    authenticated_path = _assert_no_link_components(root, PurePosixPath(relative_path))
    if authenticated_path.read_bytes() != identity_bytes:
        raise CalibrationRunError("committed frozen identity changed after authorization")

    expected_commit = _git_revision(identity_commit, context="identity commit")
    head = _run_model_staging_git(
        git_executable,
        root,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    )
    if head.returncode != 0:
        raise CalibrationRunError("cannot resolve current HEAD for frozen identity provenance")
    actual_head = _git_revision(
        head.stdout.decode("ascii", errors="strict").strip(),
        context="current identity commit",
    )
    if actual_head != expected_commit:
        raise CalibrationRunError("current HEAD differs from the explicit identity commit")

    tree_record = _one_nul_git_record(
        _run_model_staging_git(
            git_executable,
            root,
            "ls-tree",
            "-z",
            "--full-tree",
            expected_commit,
            "--",
            relative_path,
        ),
        context="frozen identity HEAD entry",
    )
    try:
        tree_header, tree_path = tree_record.split(b"\t", 1)
        mode, object_type, head_oid = tree_header.decode("ascii").split(" ", 2)
        decoded_tree_path = tree_path.decode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise CalibrationRunError("frozen identity HEAD entry is malformed") from exc
    if mode != "100644" or object_type != "blob" or decoded_tree_path != relative_path:
        raise CalibrationRunError("frozen identity HEAD entry has the wrong mode, type, or path")

    index_record = _one_nul_git_record(
        _run_model_staging_git(
            git_executable,
            root,
            "ls-files",
            "--stage",
            "-z",
            "--",
            relative_path,
        ),
        context="frozen identity index entry",
    )
    try:
        index_header, index_path = index_record.split(b"\t", 1)
        index_mode, index_oid, stage = index_header.decode("ascii").split(" ", 2)
        decoded_index_path = index_path.decode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise CalibrationRunError("frozen identity index entry is malformed") from exc
    expected_oid = _git_blob_oid_bytes(identity_bytes)
    if (
        index_mode != "100644"
        or stage != "0"
        or decoded_index_path != relative_path
        or head_oid != expected_oid
        or index_oid != expected_oid
    ):
        raise CalibrationRunError("frozen identity HEAD, index, and worktree bytes differ")
    return actual_head


def _authenticate_frozen_identity_source_contract(
    *,
    git_executable: AuthenticatedGitExecutable,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    repository_root: Path,
    repository_source_manifest_path: Path,
    source_commit: str,
) -> FrozenIdentitySourceAuthorization:
    """Authenticate one promoted identity against its exact H0 source contract."""

    expected_identity_sha256 = _sha256(
        expected_frozen_identity_sha256,
        context="expected frozen identity SHA-256",
    )
    identity_bytes = _read_stable_regular_bytes(
        frozen_identity_path,
        context="frozen identity",
    )
    if sha256_bytes(identity_bytes) != expected_identity_sha256:
        raise CalibrationRunError("frozen identity bytes differ from the explicit SHA-256")
    bindings = _bootstrap_identity_bindings(identity_bytes)

    source_manifest_bytes = _read_stable_regular_bytes(
        repository_source_manifest_path,
        context="repository source manifest",
    )
    if sha256_bytes(source_manifest_bytes) != bindings.repository_source_manifest_file_sha256:
        raise CalibrationRunError("repository source manifest differs from the frozen identity")
    bootstrap_source = _bootstrap_source_manifest(
        source_manifest_bytes,
        repository_root=repository_root,
        require_adapter=False,
    )
    requested_source_commit = _git_revision(source_commit, context="requested source commit")
    if requested_source_commit != bootstrap_source.source_commit:
        raise CalibrationRunError("requested source commit differs from the identity-bound H0")

    source_module = _load_exact_source_module(
        MODEL_STAGING_SOURCE_MODULE,
        SOURCE_VERIFIER_PATH,
        repository_root=repository_root,
        entry=bootstrap_source.entries[SOURCE_VERIFIER_PATH],
    )
    resolver_module: ModuleType | None = None
    try:
        verified_source = source_module.verify_experiment013_source_manifest(
            bootstrap_source.manifest,
            repo_root=repository_root,
            git_executable=git_executable.path,
        )
        if verified_source != bootstrap_source.manifest:
            raise CalibrationRunError("source verifier returned different frozen-identity evidence")
        if verified_source.get("source_commit") != requested_source_commit:
            raise CalibrationRunError(
                "verified source-manifest commit differs from requested frozen source commit"
            )
        resolver_module = _load_exact_source_module(
            MODEL_STAGING_RESOLVER_MODULE,
            IDENTITY_RESOLVER_SOURCE_PATH,
            repository_root=repository_root,
            entry=bootstrap_source.entries[IDENTITY_RESOLVER_SOURCE_PATH],
        )
        identity = _identity_view_from_resolver(
            identity_bytes,
            resolver_module,
            expected_file_sha256=expected_identity_sha256,
        )
    finally:
        if resolver_module is not None:
            sys.modules.pop(MODEL_STAGING_RESOLVER_MODULE, None)
        sys.modules.pop(MODEL_STAGING_SOURCE_MODULE, None)
    if (
        identity.repository_source_manifest_file_sha256
        != bindings.repository_source_manifest_file_sha256
        or identity.runtime_manifest_file_sha256 != bindings.runtime_manifest_file_sha256
        or identity.model_file_manifest_file_sha256 != bindings.model_file_manifest_file_sha256
        or identity.parquet_materialization_manifest_file_sha256
        != bindings.parquet_materialization_manifest_file_sha256
    ):
        raise CalibrationRunError("full frozen identity differs from its bootstrap bindings")

    return FrozenIdentitySourceAuthorization(
        identity=identity,
        bindings=bindings,
        identity_bytes=identity_bytes,
        frozen_identity_file_sha256=expected_identity_sha256,
        source_commit=requested_source_commit,
    )


def _source_manifest_entry_sha256(
    source_manifest_bytes: bytes,
    *,
    relative_path: str,
) -> str:
    """Read one already-authenticated source entry without importing source code."""

    manifest = _strict_json_bytes(
        source_manifest_bytes,
        context="repository source manifest for capture provenance",
    )
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, list):
        raise CalibrationRunError("repository source manifest path inventory is missing")
    matches = [
        item for item in raw_paths if isinstance(item, dict) and item.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise CalibrationRunError(
            f"repository source manifest does not bind exactly one {relative_path}"
        )
    return _sha256(
        matches[0].get("raw_sha256"),
        context=f"repository source {relative_path} SHA-256",
    )


def _authenticate_calibration_identity_capture_provenance_bytes_unchecked(
    *,
    receipt_bytes: bytes,
    expected_receipt_sha256: str,
    runtime_manifest_bytes: bytes,
    expected_runtime_manifest_sha256: str,
    source_manifest_bytes: bytes,
    expected_identity_input_sha256: str,
    expected_bindings: BootstrapIdentityBindings,
    expected_source_commit: str,
) -> str:
    """Authenticate the content-addressed sealed-capture custody receipt.

    The receipt attests no general dependency closure.  It records and checks
    only the fixed application import surface used by calibration identity
    capture against the runtime-v5 tree and distribution RECORD inventories.
    """

    if not isinstance(receipt_bytes, bytes) or not isinstance(runtime_manifest_bytes, bytes):
        raise TypeError("capture provenance receipt and runtime manifest must be bytes")
    expected_receipt = _sha256(
        expected_receipt_sha256,
        context="expected calibration identity capture provenance receipt SHA-256",
    )
    actual_receipt = sha256_bytes(receipt_bytes)
    if actual_receipt != expected_receipt:
        raise CalibrationRunError(
            "calibration identity capture provenance receipt differs from its explicit SHA-256"
        )
    root = _strict_json_bytes(
        receipt_bytes,
        context="calibration identity capture provenance receipt",
    )
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
        context="calibration identity capture provenance receipt",
    )
    if canonical_json_bytes(root) != receipt_bytes:
        raise CalibrationRunError(
            "calibration identity capture provenance receipt is not canonical JSON"
        )
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
    ):
        raise CalibrationRunError(
            "calibration identity capture provenance receipt identity drifted"
        )
    if root["excluded_runtime_modules"] != list(CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES):
        raise CalibrationRunError("capture provenance excluded-module policy drifted")
    if _git_revision(root["source_commit"], context="capture provenance source commit") != (
        _git_revision(expected_source_commit, context="expected capture provenance source commit")
    ):
        raise CalibrationRunError("capture provenance source commit differs from H0")
    if _sha256(
        root["identity_input_file_sha256"],
        context="capture provenance identity input SHA-256",
    ) != _sha256(
        expected_identity_input_sha256,
        context="expected capture provenance identity input SHA-256",
    ):
        raise CalibrationRunError("capture provenance binds a different identity input")

    expected_binding_values = {
        "calibration_runtime_manifest_file_sha256": expected_bindings.runtime_manifest_file_sha256,
        "model_file_manifest_file_sha256": expected_bindings.model_file_manifest_file_sha256,
        "parquet_materialization_manifest_file_sha256": (
            expected_bindings.parquet_materialization_manifest_file_sha256
        ),
        "repository_source_manifest_file_sha256": (
            expected_bindings.repository_source_manifest_file_sha256
        ),
    }
    raw_bindings = root["execution_bindings"]
    if not isinstance(raw_bindings, dict):
        raise CalibrationRunError("capture provenance execution bindings are missing")
    _exact_fields(
        raw_bindings,
        set(expected_binding_values),
        context="capture provenance execution bindings",
    )
    normalized_bindings = {
        name: _sha256(raw_bindings[name], context=f"capture provenance {name}")
        for name in sorted(raw_bindings)
    }
    if normalized_bindings != expected_binding_values:
        raise CalibrationRunError("capture provenance execution bindings differ from identity")
    if (
        sha256_bytes(source_manifest_bytes)
        != expected_bindings.repository_source_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "repository source manifest for capture provenance differs from identity"
        )

    capture_source = root["capture_source"]
    if not isinstance(capture_source, dict):
        raise CalibrationRunError("capture provenance source record is missing")
    _exact_fields(
        capture_source,
        {"path", "sha256"},
        context="capture provenance source record",
    )
    if capture_source["path"] != CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH:
        raise CalibrationRunError("capture provenance source path drifted")
    expected_capture_source_sha256 = _source_manifest_entry_sha256(
        source_manifest_bytes,
        relative_path=CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
    )
    if (
        _sha256(
            capture_source["sha256"],
            context="capture provenance source SHA-256",
        )
        != expected_capture_source_sha256
    ):
        raise CalibrationRunError("capture provenance source differs from H0")

    expected_runtime = _sha256(
        expected_runtime_manifest_sha256,
        context="expected runtime manifest SHA-256 for capture provenance",
    )
    actual_runtime = sha256_bytes(runtime_manifest_bytes)
    if (
        actual_runtime != expected_runtime
        or actual_runtime != expected_bindings.runtime_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "capture provenance runtime manifest differs from identity/CLI binding"
        )
    runtime_manifest = parse_calibration_runtime_manifest(runtime_manifest_bytes)
    tree_by_name = {
        tree.name: {item.path: item for item in tree.files}
        for tree in runtime_manifest.runtime_trees
    }
    distribution_by_name = {item.name: item for item in runtime_manifest.distributions}
    import_path_by_root = {item.name: item.import_path for item in runtime_manifest.package_roots}

    raw_origins = root["critical_module_origins"]
    if not isinstance(raw_origins, list):
        raise CalibrationRunError("capture provenance critical module origins are missing")
    expected_modules = sorted(CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS)
    if [item.get("module") if isinstance(item, dict) else None for item in raw_origins] != (
        expected_modules
    ):
        raise CalibrationRunError(
            "capture provenance critical module inventory is not exact and sorted"
        )
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
            context="capture provenance critical module origin",
        )
        module_name = cast(str, item["module"])
        distribution_name = CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS[module_name]
        if item["distribution"] != distribution_name:
            raise CalibrationRunError(
                f"capture provenance distribution mapping drifted: {module_name}"
            )
        distribution = distribution_by_name.get(distribution_name)
        if distribution is None:
            raise CalibrationRunError(
                f"capture provenance runtime omits distribution: {distribution_name}"
            )
        package_root = _runtime_root_name(
            item["package_root"],
            context=f"capture provenance {module_name} package root",
        )
        relative_path = _canonical_relative_path(
            item["relative_path"],
            context=f"capture provenance {module_name} relative path",
        )
        if package_root != distribution.package_root or item["version"] != distribution.version:
            raise CalibrationRunError(
                f"capture provenance distribution identity drifted: {module_name}"
            )
        import_path = import_path_by_root.get(package_root)
        if import_path is None:
            raise CalibrationRunError(
                f"capture provenance package root is not importable: {module_name}"
            )
        try:
            module_relative = PurePosixPath(relative_path).relative_to(PurePosixPath(import_path))
        except ValueError as exc:
            raise CalibrationRunError(
                f"capture provenance module is outside its import root: {module_name}"
            ) from exc
        if not _runtime_module_relative_origin_matches(module_name, module_relative):
            raise CalibrationRunError(f"capture provenance module path is shadowed: {module_name}")
        if relative_path not in distribution.files:
            raise CalibrationRunError(
                f"capture provenance module lacks RECORD ownership: {module_name}"
            )
        runtime_file = tree_by_name.get(package_root, {}).get(relative_path)
        if runtime_file is None or runtime_file != RuntimeFileRecord(
            path=relative_path,
            sha256=_sha256(
                item["sha256"],
                context=f"capture provenance {module_name} file SHA-256",
            ),
            size_bytes=_nonnegative_int(
                item["size_bytes"],
                context=f"capture provenance {module_name} file size",
            ),
        ):
            raise CalibrationRunError(
                f"capture provenance module differs from runtime inventory: {module_name}"
            )
    return actual_receipt


def _authenticate_calibration_identity_capture_provenance_bytes(
    *,
    receipt_bytes: bytes,
    expected_receipt_sha256: str,
    runtime_manifest_bytes: bytes,
    expected_runtime_manifest_sha256: str,
    source_manifest_bytes: bytes,
    expected_identity_input_sha256: str,
    expected_bindings: BootstrapIdentityBindings,
    expected_source_commit: str,
) -> str:
    """Fail closed with one public error type for malformed provenance bytes."""

    try:
        return _authenticate_calibration_identity_capture_provenance_bytes_unchecked(
            receipt_bytes=receipt_bytes,
            expected_receipt_sha256=expected_receipt_sha256,
            runtime_manifest_bytes=runtime_manifest_bytes,
            expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
            source_manifest_bytes=source_manifest_bytes,
            expected_identity_input_sha256=expected_identity_input_sha256,
            expected_bindings=expected_bindings,
            expected_source_commit=expected_source_commit,
        )
    except CalibrationRunError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationRunError(str(exc)) from exc


def _authenticate_calibration_identity_capture_provenance(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
    source_manifest_bytes: bytes,
    expected_identity_input_sha256: str,
    expected_bindings: BootstrapIdentityBindings,
    expected_source_commit: str,
) -> str:
    receipt_bytes = _read_stable_regular_bytes(
        receipt_path,
        context="calibration identity capture provenance receipt",
    )
    runtime_manifest_bytes = _read_stable_regular_bytes(
        runtime_manifest_path,
        context="calibration runtime manifest for capture provenance",
    )
    return _authenticate_calibration_identity_capture_provenance_bytes(
        receipt_bytes=receipt_bytes,
        expected_receipt_sha256=expected_receipt_sha256,
        runtime_manifest_bytes=runtime_manifest_bytes,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        source_manifest_bytes=source_manifest_bytes,
        expected_identity_input_sha256=expected_identity_input_sha256,
        expected_bindings=expected_bindings,
        expected_source_commit=expected_source_commit,
    )


def verify_frozen_identity_contract(
    *,
    git_executable_path: Path | None = None,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    repository_root: Path,
    repository_source_manifest_path: Path,
    source_commit: str,
    capture_provenance_receipt_path: Path,
    expected_capture_provenance_receipt_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
) -> dict[str, object]:
    """Verify a promoted identity against H0 without writes or model access."""

    git_executable = _authenticate_git_executable(git_executable_path)
    authorization = _authenticate_frozen_identity_source_contract(
        git_executable=git_executable,
        frozen_identity_path=frozen_identity_path,
        expected_frozen_identity_sha256=expected_frozen_identity_sha256,
        repository_root=repository_root,
        repository_source_manifest_path=repository_source_manifest_path,
        source_commit=source_commit,
    )
    identity = authorization.identity
    bindings = authorization.bindings
    source_manifest_bytes = _read_stable_regular_bytes(
        repository_source_manifest_path,
        context="repository source manifest for capture provenance",
    )
    capture_provenance_sha256 = _authenticate_calibration_identity_capture_provenance(
        receipt_path=capture_provenance_receipt_path,
        expected_receipt_sha256=expected_capture_provenance_receipt_sha256,
        runtime_manifest_path=runtime_manifest_path,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        source_manifest_bytes=source_manifest_bytes,
        expected_identity_input_sha256=identity.identity_input_manifest_sha256,
        expected_bindings=bindings,
        expected_source_commit=authorization.source_commit,
    )
    return {
        "artifact_kind": FROZEN_IDENTITY_CONTRACT_KIND,
        "assignment_sha256": identity.assignment_sha256,
        "canonical_evidence_sha256": identity.canonical_evidence_sha256,
        "capture_provenance_receipt_file_sha256": capture_provenance_sha256,
        "execution_bindings": {
            "calibration_runtime_manifest_file_sha256": bindings.runtime_manifest_file_sha256,
            "model_file_manifest_file_sha256": bindings.model_file_manifest_file_sha256,
            "parquet_materialization_manifest_file_sha256": (
                bindings.parquet_materialization_manifest_file_sha256
            ),
            "repository_source_manifest_file_sha256": (
                bindings.repository_source_manifest_file_sha256
            ),
        },
        "frozen_identity_file_sha256": authorization.frozen_identity_file_sha256,
        "git_executable": {
            "sha256": git_executable.sha256,
            "size_bytes": git_executable.size_bytes,
        },
        "identity_input_manifest_sha256": identity.identity_input_manifest_sha256,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "record_count": len(identity.records),
        "runner_revision": RUNNER_REVISION,
        "schema_version": FROZEN_IDENTITY_CONTRACT_SCHEMA,
        "source_commit": authorization.source_commit,
        "status": "verified_frozen_identity_contract",
        "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
        "transformers_version": identity.transformers_version,
    }


def _authenticate_model_staging_authorization(
    *,
    git_executable: AuthenticatedGitExecutable,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    identity_commit: str,
    repository_root: Path,
    repository_source_manifest_path: Path,
    source_commit: str,
    model_file_manifest_path: Path,
    expected_model_file_manifest_sha256: str,
    capture_provenance_receipt_path: Path,
    expected_capture_provenance_receipt_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
) -> ModelStagingAuthorization:
    """Authenticate promotion, committed provenance, source, and model metadata."""

    source_authorization = _authenticate_frozen_identity_source_contract(
        git_executable=git_executable,
        frozen_identity_path=frozen_identity_path,
        expected_frozen_identity_sha256=expected_frozen_identity_sha256,
        repository_root=repository_root,
        repository_source_manifest_path=repository_source_manifest_path,
        source_commit=source_commit,
    )

    committed_at = _verify_committed_frozen_identity(
        git_executable,
        repository_root,
        frozen_identity_path,
        source_authorization.identity_bytes,
        identity_commit=identity_commit,
    )
    model_manifest_bytes = _read_stable_regular_bytes(
        model_file_manifest_path,
        context="model file manifest",
    )
    expected_model_sha256 = _sha256(
        expected_model_file_manifest_sha256,
        context="expected model file manifest SHA-256",
    )
    actual_model_sha256 = sha256_bytes(model_manifest_bytes)
    if (
        actual_model_sha256 != expected_model_sha256
        or actual_model_sha256 != source_authorization.bindings.model_file_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "model file manifest differs from the frozen identity/CLI binding"
        )
    model_manifest = parse_model_file_manifest(model_manifest_bytes)
    _model_contract_matches(source_authorization.identity, model_manifest)
    source_manifest_bytes = _read_stable_regular_bytes(
        repository_source_manifest_path,
        context="repository source manifest for capture provenance",
    )
    capture_provenance_sha256 = _authenticate_calibration_identity_capture_provenance(
        receipt_path=capture_provenance_receipt_path,
        expected_receipt_sha256=expected_capture_provenance_receipt_sha256,
        runtime_manifest_path=runtime_manifest_path,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        source_manifest_bytes=source_manifest_bytes,
        expected_identity_input_sha256=(
            source_authorization.identity.identity_input_manifest_sha256
        ),
        expected_bindings=source_authorization.bindings,
        expected_source_commit=source_authorization.source_commit,
    )
    return ModelStagingAuthorization(
        identity=source_authorization.identity,
        model_manifest=model_manifest,
        frozen_identity_file_sha256=source_authorization.frozen_identity_file_sha256,
        capture_provenance_receipt_file_sha256=capture_provenance_sha256,
        identity_commit=committed_at,
        source_commit=source_authorization.source_commit,
    )


def verify_identity_bound_model_staging_authorization(
    *,
    git_executable_path: Path | None = None,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    identity_commit: str,
    repository_root: Path,
    repository_source_manifest_path: Path,
    source_commit: str,
    model_file_manifest_path: Path,
    expected_model_file_manifest_sha256: str,
    capture_provenance_receipt_path: Path,
    expected_capture_provenance_receipt_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
) -> dict[str, object]:
    """Verify model-staging authorization without accessing model payloads."""

    git_executable = _authenticate_git_executable(git_executable_path)
    authorization = _authenticate_model_staging_authorization(
        git_executable=git_executable,
        frozen_identity_path=frozen_identity_path,
        expected_frozen_identity_sha256=expected_frozen_identity_sha256,
        identity_commit=identity_commit,
        repository_root=repository_root,
        repository_source_manifest_path=repository_source_manifest_path,
        source_commit=source_commit,
        model_file_manifest_path=model_file_manifest_path,
        expected_model_file_manifest_sha256=expected_model_file_manifest_sha256,
        capture_provenance_receipt_path=capture_provenance_receipt_path,
        expected_capture_provenance_receipt_sha256=(expected_capture_provenance_receipt_sha256),
        runtime_manifest_path=runtime_manifest_path,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
    )
    return {
        "artifact_kind": MODEL_STAGING_AUTHORIZATION_KIND,
        "capture_provenance_receipt_file_sha256": (
            authorization.capture_provenance_receipt_file_sha256
        ),
        "file_count": len(authorization.model_manifest.files),
        "frozen_identity_file_sha256": authorization.frozen_identity_file_sha256,
        "hub_tree_manifest_sha256": authorization.model_manifest.hub_tree_manifest_sha256,
        "identity_commit": authorization.identity_commit,
        "model_id": authorization.model_manifest.model_id,
        "model_manifest_file_sha256": authorization.model_manifest.file_sha256,
        "revision": authorization.model_manifest.revision,
        "repository_source_manifest_file_sha256": (
            authorization.identity.repository_source_manifest_file_sha256
        ),
        "runner_revision": RUNNER_REVISION,
        "schema_version": MODEL_STAGING_AUTHORIZATION_SCHEMA,
        "source_commit": authorization.source_commit,
        "status": "verified_identity_bound_model_staging_authorization",
        "total_size_bytes": sum(item.size_bytes for item in authorization.model_manifest.files),
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_existing_regular_directory(
    path: Path,
    *,
    context: str,
) -> tuple[Path, tuple[DirectoryComponentIdentity, ...]]:
    """Normalize one existing directory without creating or following links."""

    try:
        absolute = Path(os.path.abspath(path))
    except (OSError, TypeError, ValueError) as exc:
        raise CalibrationRunError(f"{context} is unavailable") from exc
    if not absolute.anchor:
        raise CalibrationRunError(f"{context} is not absolute after normalization")
    component = Path(absolute.anchor)
    components = [component]
    for part in absolute.parts[1:]:
        component /= part
        components.append(component)

    def snapshot() -> tuple[DirectoryComponentIdentity, ...]:
        identities: list[DirectoryComponentIdentity] = []
        for candidate in components:
            if not os.path.lexists(candidate):
                raise CalibrationRunError(f"{context} must already exist")
            try:
                status = candidate.lstat()
            except OSError as exc:
                raise CalibrationRunError(f"{context} is unavailable") from exc
            if stat.S_ISLNK(status.st_mode) or bool(
                getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise CalibrationRunError(f"{context} traverses a link or non-directory")
            if not stat.S_ISDIR(status.st_mode):
                raise CalibrationRunError(f"{context} traverses a link or non-directory")
            identities.append(
                DirectoryComponentIdentity(
                    device=status.st_dev,
                    inode=status.st_ino,
                    mode=status.st_mode,
                )
            )
        return tuple(identities)

    before = snapshot()
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CalibrationRunError(f"{context} is unavailable") from exc
    after = snapshot()
    if after != before:
        raise CalibrationRunError(f"{context} changed while it was validated")
    return resolved, after


def _verify_ruler_receipt_directory_precondition(path: Path) -> Path:
    """Validate only the frozen receipt-directory shape without reading file bytes."""

    raw = Path(path)
    if not raw.is_absolute():
        raise CalibrationRunError("RULER receipt directory must be absolute")
    root, component_identities = _require_existing_regular_directory(
        raw,
        context="RULER receipt directory",
    )

    def snapshot() -> tuple[tuple[str, int, int, int, int], ...]:
        try:
            with os.scandir(root) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise CalibrationRunError("RULER receipt directory is unavailable") from exc
        names = [entry.name for entry in entries]
        if len({name.casefold() for name in names}) != len(names):
            raise CalibrationRunError("RULER receipt directory has case-colliding names")
        expected = set(RULER_RECEIPT_DIRECTORY_FILENAMES)
        observed = set(names)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise CalibrationRunError(
                "RULER receipt directory inventory drifted: "
                f"missing={missing}, unexpected={unexpected}"
            )
        identities: list[tuple[str, int, int, int, int]] = []
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CalibrationRunError(
                    f"RULER receipt entry is unavailable: {entry.name}"
                ) from exc
            if (
                entry.is_symlink()
                or bool(getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)
                or not stat.S_ISREG(status.st_mode)
            ):
                raise CalibrationRunError(
                    f"RULER receipt entry must be a regular non-link file: {entry.name}"
                )
            identities.append(
                (entry.name, status.st_dev, status.st_ino, status.st_mode, status.st_size)
            )
        return tuple(identities)

    before = snapshot()
    repeated_root, repeated_components = _require_existing_regular_directory(
        raw,
        context="RULER receipt directory",
    )
    after = snapshot()
    if repeated_root != root or repeated_components != component_identities or after != before:
        raise CalibrationRunError("RULER receipt directory changed while it was validated")
    return root


def _capture_hub_cache_root_precondition(path: Path) -> Path:
    """Bind the explicit Hub endpoint already authenticated by the sealed launcher."""

    cache_root = Path(os.path.abspath(path))
    hub_root, _component_identities = _require_existing_regular_directory(
        cache_root / "hub",
        context="capture Hub cache root",
    )
    if hub_root.parent != cache_root:
        raise CalibrationRunError("capture Hub cache root escaped the explicit cache root")
    environment = {name.upper(): value for name, value in os.environ.items()}
    expected = str(hub_root)
    if (
        environment.get("HF_HUB_CACHE") != expected
        or environment.get("HUGGINGFACE_HUB_CACHE") != expected
    ):
        raise CalibrationRunError("capture Hub cache environment differs from its endpoint")
    return hub_root


def _normalized_absolute_path_sha256(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError("path digest input must be absolute")
    normalized = os.path.normcase(os.path.normpath(str(path)))
    return sha256_bytes(normalized.encode("utf-8"))


def _directory_component_identities_sha256(
    identities: tuple[DirectoryComponentIdentity, ...],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "device": item.device,
                    "inode": item.inode,
                    "mode": item.mode,
                }
                for item in identities
            ]
        )
    )


def _model_staging_path_contract(paths: ModelStagingPaths) -> dict[str, object]:
    return {
        "hub_cache_component_identities_sha256": _directory_component_identities_sha256(
            paths.hub_cache_component_identities
        ),
        "hub_cache_root_absolute_path_sha256": _normalized_absolute_path_sha256(
            paths.hub_cache_root
        ),
        "hub_cache_root_state": "existing_regular_non_link_directory",
        "output_parent_absolute_path_sha256": _normalized_absolute_path_sha256(
            paths.output_root.parent
        ),
        "output_parent_component_identities_sha256": _directory_component_identities_sha256(
            paths.output_parent_component_identities
        ),
        "output_parent_state": "existing_regular_non_link_directory",
        "output_root_absolute_path_sha256": _normalized_absolute_path_sha256(paths.output_root),
        "output_root_state": "absent",
        "repository_component_identities_sha256": _directory_component_identities_sha256(
            paths.repository_component_identities
        ),
        "repository_root_absolute_path_sha256": _normalized_absolute_path_sha256(
            paths.repository_root
        ),
        "repository_root_state": "existing_regular_non_link_directory",
    }


def _model_staging_path_contract_sha256(paths: ModelStagingPaths) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": MODEL_STAGING_PATHS_SCHEMA,
                **_model_staging_path_contract(paths),
            }
        )
    )


def _validate_model_staging_roots(
    *,
    repository_root: Path,
    hub_cache_root: Path,
    output_root: Path,
) -> ModelStagingPaths:
    """Validate and snapshot the model-staging path boundary without writes."""

    repository, repository_identities = _require_existing_regular_directory(
        repository_root,
        context="repository root",
    )
    try:
        lexical_destination = Path(output_root)
        lexical_name = lexical_destination.name
        destination = Path(os.path.abspath(lexical_destination))
    except (OSError, TypeError, ValueError) as exc:
        raise CalibrationRunError("model output root is unavailable") from exc
    if not lexical_name or not destination.name:
        raise CalibrationRunError("model output root cannot be a filesystem root")
    if (
        _MODEL_STAGING_OUTPUT_ROOT_NAME_RE.fullmatch(lexical_name) is None
        or lexical_name.endswith((".", " "))
        or lexical_name.casefold().partition(".")[0] in _WINDOWS_RESERVED_BASENAMES
        or destination.name != lexical_name
    ):
        raise CalibrationRunError(
            "model output root name must be a canonical Windows-safe basename"
        )
    parent, output_parent_identities = _require_existing_regular_directory(
        destination.parent,
        context="model output parent",
    )
    cache, cache_identities = _require_existing_regular_directory(
        hub_cache_root,
        context="Hub cache root",
    )
    if not parent.name:
        raise CalibrationRunError("model output parent cannot be a filesystem root")
    if not cache.name:
        raise CalibrationRunError("Hub cache root cannot be a filesystem root")
    resolved_destination = parent / destination.name
    if os.path.lexists(resolved_destination):
        raise FileExistsError(f"refusing to overwrite staged model root: {resolved_destination}")
    repository_identity = repository_identities[-1]
    cache_identity = cache_identities[-1]
    if (
        _path_is_within(resolved_destination, repository)
        or repository_identity in output_parent_identities
    ):
        raise CalibrationRunError("model output root must be outside the repository")
    if (
        _path_is_within(cache, repository)
        or _path_is_within(repository, cache)
        or repository_identity in cache_identities
        or cache_identity in repository_identities
    ):
        raise CalibrationRunError("Hub cache root must not overlap the repository")
    if (
        _path_is_within(resolved_destination, cache)
        or _path_is_within(cache, resolved_destination)
        or cache_identity in output_parent_identities
    ):
        raise CalibrationRunError("Hub cache and staged model roots must not be nested")
    return ModelStagingPaths(
        repository_root=repository,
        repository_component_identities=repository_identities,
        hub_cache_root=cache,
        hub_cache_component_identities=cache_identities,
        output_root=resolved_destination,
        output_parent_component_identities=output_parent_identities,
    )


def verify_model_staging_paths(
    *,
    repository_root: Path,
    hub_cache_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Verify model-staging roots without authorization, imports, or writes."""

    paths = _validate_model_staging_roots(
        repository_root=repository_root,
        hub_cache_root=hub_cache_root,
        output_root=output_root,
    )
    contract = _model_staging_path_contract(paths)
    return {
        "artifact_kind": MODEL_STAGING_PATHS_KIND,
        **contract,
        "path_contract_sha256": _model_staging_path_contract_sha256(paths),
        "runner_revision": RUNNER_REVISION,
        "schema_version": MODEL_STAGING_PATHS_SCHEMA,
        "status": "verified_model_staging_paths",
    }


def _assert_regular_cache_payload(cache_root: Path, returned_path: object) -> Path:
    if not isinstance(returned_path, (str, os.PathLike)):
        raise CalibrationRunError("Hub downloader did not return a filesystem path")
    cache = cache_root.resolve(strict=True)
    returned = Path(os.path.abspath(returned_path))
    if not _path_is_within(returned, cache):
        raise CalibrationRunError("Hub downloader returned a path outside the explicit cache")
    candidate = cache
    relative = returned.relative_to(cache)
    if not relative.parts:
        raise CalibrationRunError("Hub downloader returned the cache root instead of a file")
    for part in relative.parts[:-1]:
        candidate /= part
        if _is_link_or_reparse(candidate) or not candidate.is_dir():
            raise CalibrationRunError("Hub cache payload path traverses a link or non-directory")
    if not os.path.lexists(returned):
        raise CalibrationRunError("Hub downloader returned an unavailable cache path")
    try:
        payload = returned.resolve(strict=True)
        payload.relative_to(cache)
    except (OSError, ValueError) as exc:
        raise CalibrationRunError("Hub cache pointer escapes the explicit cache") from exc
    payload_relative = payload.relative_to(cache)
    candidate = cache
    for part in payload_relative.parts:
        candidate /= part
        if _is_link_or_reparse(candidate):
            raise CalibrationRunError(
                "resolved Hub cache payload traverses a link or reparse point"
            )
    try:
        status = payload.stat()
    except OSError as exc:
        raise CalibrationRunError("resolved Hub cache payload is unavailable") from exc
    if not stat.S_ISREG(status.st_mode):
        raise CalibrationRunError("resolved Hub cache payload is not a regular file")
    return payload


def _copy_authenticated_model_payload(
    source: Path,
    destination: Path,
    record: ModelFileRecord,
) -> None:
    try:
        before = source.stat()
    except OSError as exc:
        raise CalibrationRunError(f"cannot stat cached model file: {record.name}") from exc
    if not stat.S_ISREG(before.st_mode) or _is_link_or_reparse(source):
        raise CalibrationRunError(f"cached model payload is not a regular file: {record.name}")
    if before.st_size != record.size_bytes:
        raise CalibrationRunError(f"cached model file size differs from manifest: {record.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {record.size_bytes}\0".encode("ascii"))
    copied = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
                writer.write(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
    except OSError as exc:
        raise CalibrationRunError(f"cannot stage cached model file: {record.name}") from exc
    if (
        copied != record.size_bytes
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise CalibrationRunError(f"cached model file changed while staging: {record.name}")
    content_matches = (
        sha256.hexdigest() == record.lfs_sha256
        if record.lfs_sha256 is not None
        else git_blob.hexdigest() == record.git_blob_oid
    )
    if not content_matches:
        raise CalibrationRunError(f"cached model file authentication failed: {record.name}")


def _atomic_rename_directory_no_overwrite(source: Path, destination: Path) -> None:
    """Publish a sibling directory atomically without replacement races."""

    if source.parent != destination.parent:
        raise CalibrationRunError("atomic model publication requires sibling directories")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite staged model root: {destination}")
    if os.name == "nt":
        try:
            source.rename(destination)
        except OSError as exc:
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"refusing to overwrite staged model root: {destination}"
                ) from exc
            raise
        return
    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise CalibrationRunError("atomic no-replace directory publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"refusing to overwrite staged model root: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))
    raise CalibrationRunError("atomic no-replace directory publication is unsupported")


def stage_identity_bound_model(
    *,
    git_executable_path: Path | None = None,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    identity_commit: str,
    repository_root: Path,
    repository_source_manifest_path: Path,
    source_commit: str,
    model_file_manifest_path: Path,
    expected_model_file_manifest_sha256: str,
    capture_provenance_receipt_path: Path,
    expected_capture_provenance_receipt_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
    expected_model_staging_path_contract_sha256: str,
    hub_cache_root: Path,
    output_root: Path,
    local_files_only: bool = False,
    downloader: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Stage only identity-bound model files after committed promotion."""

    if type(local_files_only) is not bool:
        raise TypeError("local_files_only must be bool")
    initial_paths = _validate_model_staging_roots(
        repository_root=repository_root,
        hub_cache_root=hub_cache_root,
        output_root=output_root,
    )
    expected_path_contract_sha256 = _sha256(
        expected_model_staging_path_contract_sha256,
        context="expected model-staging path contract SHA-256",
    )
    actual_path_contract_sha256 = _model_staging_path_contract_sha256(initial_paths)
    if actual_path_contract_sha256 != expected_path_contract_sha256:
        raise CalibrationRunError("model-staging path contract differs from the CLI binding")
    git_executable = _authenticate_git_executable(git_executable_path)
    authorization = _authenticate_model_staging_authorization(
        git_executable=git_executable,
        frozen_identity_path=frozen_identity_path,
        expected_frozen_identity_sha256=expected_frozen_identity_sha256,
        identity_commit=identity_commit,
        repository_root=initial_paths.repository_root,
        repository_source_manifest_path=repository_source_manifest_path,
        source_commit=source_commit,
        model_file_manifest_path=model_file_manifest_path,
        expected_model_file_manifest_sha256=expected_model_file_manifest_sha256,
        capture_provenance_receipt_path=capture_provenance_receipt_path,
        expected_capture_provenance_receipt_sha256=(expected_capture_provenance_receipt_sha256),
        runtime_manifest_path=runtime_manifest_path,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
    )
    confirmed_paths = _validate_model_staging_roots(
        repository_root=repository_root,
        hub_cache_root=hub_cache_root,
        output_root=output_root,
    )
    if confirmed_paths != initial_paths:
        raise CalibrationRunError("model-staging roots changed during authorization")
    cache = confirmed_paths.hub_cache_root
    destination = confirmed_paths.output_root
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    prefix = f".{destination.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=destination.parent))
    owned_staging = True
    staging_component_identities: tuple[DirectoryComponentIdentity, ...] | None = None
    try:
        staging, staging_component_identities = _require_existing_regular_directory(
            staging,
            context="owned model staging directory",
        )
        for record in authorization.model_manifest.files:
            returned = downloader(
                repo_id=authorization.model_manifest.model_id,
                filename=record.name,
                repo_type="model",
                revision=authorization.model_manifest.revision,
                cache_dir=cache,
                local_files_only=local_files_only,
                token=False,
                endpoint="https://huggingface.co",
            )
            source = _assert_regular_cache_payload(cache, returned)
            _copy_authenticated_model_payload(source, staging / record.name, record)
        _verify_exact_local_model_tree(staging, authorization.model_manifest)

        repeated = _authenticate_model_staging_authorization(
            git_executable=git_executable,
            frozen_identity_path=frozen_identity_path,
            expected_frozen_identity_sha256=expected_frozen_identity_sha256,
            identity_commit=identity_commit,
            repository_root=initial_paths.repository_root,
            repository_source_manifest_path=repository_source_manifest_path,
            source_commit=source_commit,
            model_file_manifest_path=model_file_manifest_path,
            expected_model_file_manifest_sha256=expected_model_file_manifest_sha256,
            capture_provenance_receipt_path=capture_provenance_receipt_path,
            expected_capture_provenance_receipt_sha256=(expected_capture_provenance_receipt_sha256),
            runtime_manifest_path=runtime_manifest_path,
            expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        )
        if repeated != authorization:
            raise CalibrationRunError("model-staging authorization changed before publication")
        _verify_exact_local_model_tree(staging, repeated.model_manifest)
        publication_paths = _validate_model_staging_roots(
            repository_root=repository_root,
            hub_cache_root=hub_cache_root,
            output_root=output_root,
        )
        if publication_paths != initial_paths:
            raise CalibrationRunError("model-staging roots changed before publication")
        _atomic_rename_directory_no_overwrite(staging, destination)
        owned_staging = False
    finally:
        if owned_staging:
            try:
                staging.relative_to(destination.parent)
            except ValueError as exc:
                raise RuntimeError("owned model staging directory escaped its parent") from exc
            if not staging.name.startswith(prefix):
                raise RuntimeError("owned model staging directory name drifted")
            if staging_component_identities is None:
                raise RuntimeError("owned model staging directory identity was not captured")
            try:
                current_staging, current_identities = _require_existing_regular_directory(
                    staging,
                    context="owned model staging directory",
                )
            except CalibrationRunError as exc:
                raise RuntimeError(
                    "refusing to clean an unauthenticated model staging directory"
                ) from exc
            if current_staging != staging or current_identities != staging_component_identities:
                raise RuntimeError("refusing to clean a replaced model staging directory")
            shutil.rmtree(staging, ignore_errors=False)
    _verify_exact_local_model_tree(destination, authorization.model_manifest)
    return {
        "capture_provenance_receipt_file_sha256": (
            authorization.capture_provenance_receipt_file_sha256
        ),
        "file_count": len(authorization.model_manifest.files),
        "frozen_identity_file_sha256": authorization.frozen_identity_file_sha256,
        "identity_commit": authorization.identity_commit,
        "model_id": authorization.model_manifest.model_id,
        "model_manifest_file_sha256": authorization.model_manifest.file_sha256,
        "model_root": str(destination),
        "model_staging_path_contract_sha256": expected_path_contract_sha256,
        "revision": authorization.model_manifest.revision,
        "source_commit": authorization.source_commit,
        "status": "staged_authenticated_model",
        "total_size_bytes": sum(item.size_bytes for item in authorization.model_manifest.files),
    }


def _normalized_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationRunError("installed distribution has no canonical name")
    normalized = re.sub(r"[-_.]+", "-", value.strip()).lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None:
        raise CalibrationRunError(f"installed distribution name is invalid: {value!r}")
    return normalized


def _runtime_root_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _RUNTIME_ROOT_NAME_RE.fullmatch(value) is None:
        raise CalibrationRunError(f"{context} is not a canonical runtime-root name")
    return value


def _absolute_runtime_root(path: Path, *, context: str) -> Path:
    root = Path(os.path.abspath(path))
    if _is_link_or_reparse(root):
        raise CalibrationRunError(f"{context} is a link or reparse point")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CalibrationRunError(f"{context} is unavailable") from exc
    if not resolved.is_dir():
        raise CalibrationRunError(f"{context} is not a directory")
    return resolved


def _runtime_root_map(
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
) -> dict[str, Path]:
    if not isinstance(package_roots, Mapping) or not package_roots:
        raise CalibrationRunError("at least one explicit package root is required")
    roots = {
        BASE_RUNTIME_ROOT_NAME: _absolute_runtime_root(
            base_runtime_root,
            context="base runtime root",
        )
    }
    for raw_name, raw_path in sorted(package_roots.items()):
        name = _runtime_root_name(raw_name, context="package root name")
        if name == BASE_RUNTIME_ROOT_NAME or name in roots:
            raise CalibrationRunError(f"duplicate or reserved package root name: {name}")
        roots[name] = _absolute_runtime_root(
            Path(raw_path),
            context=f"package root {name}",
        )
    resolved = list(roots.items())
    for index, (left_name, left) in enumerate(resolved):
        for right_name, right in resolved[index + 1 :]:
            if left == right:
                raise CalibrationRunError(
                    f"runtime roots resolve to the same directory: {left_name}, {right_name}"
                )
            for outer_name, outer, inner_name, inner in (
                (left_name, left, right_name, right),
                (right_name, right, left_name, left),
            ):
                try:
                    inner.relative_to(outer)
                except ValueError:
                    continue
                raise CalibrationRunError(
                    f"runtime roots must not be nested: {outer_name}, {inner_name}"
                )
    return roots


def _normalized_package_import_paths(
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(package_import_paths, Mapping) or set(package_import_paths) != set(
        package_roots
    ):
        raise CalibrationRunError("package import paths must exactly match the named package roots")
    normalized: dict[str, str] = {}
    for name in sorted(package_roots):
        relative = _canonical_relative_path(
            package_import_paths[name],
            context=f"package root {name} import path",
        )
        candidate = package_roots[name]
        for part in PurePosixPath(relative).parts:
            candidate /= part
            if _is_link_or_reparse(candidate):
                raise CalibrationRunError(
                    f"package root {name} import path traverses a link or reparse"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(package_roots[name])
        except (OSError, ValueError) as exc:
            raise CalibrationRunError(
                f"package root {name} import path is outside its runtime tree"
            ) from exc
        if not resolved.is_dir():
            raise CalibrationRunError(f"package root {name} import path is not a directory")
        normalized[name] = relative
    return normalized


def _capture_base_sys_path(
    base_runtime_root: Path,
    supplied: Sequence[str] | None,
) -> tuple[str, ...]:
    if supplied is not None:
        values = tuple(
            _canonical_base_sys_path_entry(item, context="base sys.path entry") for item in supplied
        )
    else:
        root = base_runtime_root.resolve(strict=True)
        if (
            Path(sys.prefix).resolve(strict=True) != root
            or Path(sys.base_prefix).resolve(strict=True) != root
        ):
            raise CalibrationRunError(
                "runtime capture must run from the staged base interpreter with no virtualenv"
            )
        captured: list[str] = []
        for raw_entry in sys.path:
            if not isinstance(raw_entry, str) or not raw_entry or not Path(raw_entry).is_absolute():
                raise CalibrationRunError("isolated base sys.path contains a relative entry")
            try:
                relative = Path(os.path.abspath(raw_entry)).relative_to(root).as_posix()
            except ValueError as exc:
                raise CalibrationRunError(
                    "isolated base sys.path escapes the staged base runtime"
                ) from exc
            captured.append(_canonical_base_sys_path_entry(relative, context="base sys.path entry"))
        values = tuple(captured)
    if not values or len(set(item.casefold() for item in values)) != len(values):
        raise CalibrationRunError("base sys.path entries must be non-empty and unique")
    return values


def _verify_runtime_capture_launch_state() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.utf8_mode != 1
        or sys.flags.safe_path is not True
    ):
        raise CalibrationRunError(
            "runtime capture requires -I -S -B -X utf8 on the staged interpreter"
        )
    if sys._xoptions != {"utf8": True}:
        raise CalibrationRunError("runtime capture Python -X options drifted")
    if "site" in sys.modules or "_virtualenv" in sys.modules:
        raise CalibrationRunError("runtime capture loaded site or virtualenv startup hooks")


def _runtime_path_is_forbidden(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(part.casefold() in _FORBIDDEN_RUNTIME_DIRECTORY_NAMES for part in path.parts)
        or path.name.casefold() in _FORBIDDEN_RUNTIME_FILENAMES
        or path.suffix.casefold() in _FORBIDDEN_RUNTIME_SUFFIXES
    )


def _runtime_tree_files(root: Path, *, kind: str) -> tuple[RuntimeFileRecord, ...]:
    """Hash one complete staged tree without following links or reparses."""

    root = _absolute_runtime_root(root, context=f"{kind} tree root")
    stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    files: list[RuntimeFileRecord] = []
    casefolded_paths: set[str] = set()
    while stack:
        directory, relative_parts = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CalibrationRunError(f"cannot enumerate {kind} tree") from exc
        for entry in entries:
            parts = (*relative_parts, entry.name)
            relative = _canonical_relative_path(
                PurePosixPath(*parts).as_posix(),
                context=f"{kind} tree path",
            )
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CalibrationRunError(f"runtime tree path is unavailable: {relative}") from exc
            if entry.is_symlink() or (
                getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise CalibrationRunError(
                    f"runtime tree path is a link or reparse point: {relative}"
                )
            if stat.S_ISDIR(status.st_mode):
                if entry.name.casefold() in _FORBIDDEN_RUNTIME_DIRECTORY_NAMES:
                    raise CalibrationRunError(
                        f"runtime tree contains forbidden cache directory: {relative}"
                    )
                stack.append((Path(entry.path), parts))
                continue
            if not stat.S_ISREG(status.st_mode):
                raise CalibrationRunError(f"runtime tree path is not regular: {relative}")
            if _runtime_path_is_forbidden(relative):
                raise CalibrationRunError(f"runtime tree contains forbidden file: {relative}")
            folded = relative.casefold()
            if folded in casefolded_paths:
                raise CalibrationRunError(
                    f"runtime tree contains a case-insensitive duplicate path: {relative}"
                )
            casefolded_paths.add(folded)
            path = Path(entry.path)
            digest, size = _stream_file_sha256(path)
            if _is_link_or_reparse(path):
                raise CalibrationRunError(
                    f"runtime tree path became a link or reparse point: {relative}"
                )
            try:
                path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise CalibrationRunError(
                    f"runtime tree path escapes its root: {relative}"
                ) from exc
            files.append(RuntimeFileRecord(path=relative, size_bytes=size, sha256=digest))
    files.sort(key=lambda item: item.path)
    if not files:
        raise CalibrationRunError(f"{kind} tree contains no files")
    return tuple(files)


def _raw_record_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CalibrationRunError(f"{context} is empty or non-canonical")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise CalibrationRunError(f"{context} is not a POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts[0].endswith(":") or path.as_posix() != value:
        raise CalibrationRunError(f"{context} is absolute or non-canonical")
    return value


def _record_paths(distribution: Any, *, name: str) -> tuple[str, ...]:
    read_text = getattr(distribution, "read_text", None)
    if not callable(read_text):
        raise CalibrationRunError(f"distribution has no RECORD reader: {name}")
    record_text = read_text("RECORD")
    if not isinstance(record_text, str) or not record_text:
        raise CalibrationRunError(f"distribution has no wheel RECORD inventory: {name}")
    parsed: list[str] = []
    try:
        for index, row in enumerate(csv.reader(StringIO(record_text, newline=""))):
            if len(row) != 3:
                raise CalibrationRunError(f"distribution {name} RECORD row {index} is malformed")
            parsed.append(
                _raw_record_path(
                    row[0],
                    context=f"distribution {name} RECORD path",
                )
            )
    except csv.Error as exc:
        raise CalibrationRunError(f"distribution {name} RECORD is malformed") from exc
    if not parsed:
        raise CalibrationRunError(f"distribution has an empty RECORD inventory: {name}")
    if len({path.casefold() for path in parsed}) != len(parsed):
        raise CalibrationRunError(
            f"distribution {name} RECORD paths must be case-insensitively unique"
        )
    raw_files = getattr(distribution, "files", None)
    if raw_files is None:
        raise CalibrationRunError(f"distribution has no RECORD file inventory: {name}")
    advertised = tuple(
        _raw_record_path(
            PurePosixPath(str(path)).as_posix(),
            context=f"distribution {name} metadata path",
        )
        for path in raw_files
    )
    if sorted(advertised) != sorted(parsed):
        raise CalibrationRunError(
            f"distribution {name} metadata files differ from its exact RECORD"
        )
    return tuple(sorted(parsed))


def _distribution_record(
    distribution: Any,
    *,
    name: str,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
) -> RuntimeDistributionRecord:
    raw_paths = _record_paths(distribution, name=name)
    selected_root: str | None = None
    rendered_paths: list[str] = []
    for record_path in raw_paths:
        located = Path(distribution.locate_file(record_path))
        matches: list[tuple[str, Path]] = []
        for root_name, root in package_roots.items():
            try:
                candidate = located.resolve(strict=True)
                candidate.relative_to(root)
            except (OSError, ValueError):
                continue
            if _is_link_or_reparse(candidate) or not candidate.is_file():
                raise CalibrationRunError(
                    f"distribution {name} RECORD path is not a regular authenticated file"
                )
            matches.append((root_name, candidate))
        if len(matches) != 1:
            raise CalibrationRunError(
                f"distribution {name} RECORD path is outside or ambiguous across package roots"
            )
        matched_root, matched_path = matches[0]
        if selected_root is None:
            selected_root = matched_root
        elif selected_root != matched_root:
            raise CalibrationRunError(
                f"distribution {name} spans more than one authenticated package root"
            )
        rendered_paths.append(matched_path.relative_to(package_roots[matched_root]).as_posix())
    assert selected_root is not None
    import_root = package_roots[selected_root] / PurePosixPath(package_import_paths[selected_root])
    try:
        import_root.resolve(strict=True).relative_to(package_roots[selected_root])
    except (OSError, ValueError) as exc:
        raise CalibrationRunError(
            f"distribution {name} package import path escapes its tree root"
        ) from exc
    if len({path.casefold() for path in rendered_paths}) != len(rendered_paths):
        raise CalibrationRunError(
            f"distribution {name} RECORD paths collide after tree-root normalization"
        )
    version = str(distribution.version)
    if not version or version != version.strip():
        raise CalibrationRunError(f"distribution {name} has an invalid version")
    return RuntimeDistributionRecord(
        name=name,
        version=version,
        package_root=selected_root,
        files=tuple(sorted(rendered_paths)),
    )


def _installed_distribution_map(
    distributions: Sequence[Any] | None = None,
    *,
    package_roots: Mapping[str, Path] | None = None,
    package_import_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if distributions is None:
        if not package_roots or not package_import_paths:
            raise CalibrationRunError(
                "installed distribution discovery requires explicit package roots"
            )
        selected = list(
            importlib.metadata.distributions(
                path=[
                    str(package_roots[name] / PurePosixPath(package_import_paths[name]))
                    for name in sorted(package_roots)
                ]
            )
        )
    else:
        selected = list(distributions)
    result: dict[str, Any] = {}
    for distribution in selected:
        name = _normalized_distribution_name(distribution.metadata.get("Name"))
        if name in result:
            raise CalibrationRunError(f"duplicate installed distribution identity: {name}")
        result[name] = distribution
    if not result:
        raise CalibrationRunError("calibration runtime contains no installed distributions")
    return result


def capture_calibration_runtime_manifest(
    *,
    git_executable_path: Path | None = None,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
    base_sys_path: Sequence[str] | None = None,
    interpreter_relative_path: str | None = None,
    distributions: Sequence[Any] | None = None,
    interpreter_path: Path | None = None,
    runtime_probe: RuntimeInterpreterProbe | None = None,
) -> bytes:
    """Capture complete staged base and RECORD-only package trees.

    Capture itself is not authorization to load weights. The canonical bytes
    must be frozen into the promoted identity before ``run_calibration`` can
    accept them.
    """

    git_executable = _authenticate_git_executable(git_executable_path)
    roots = _runtime_root_map(base_runtime_root, package_roots)
    packages = {name: roots[name] for name in sorted(roots) if name != BASE_RUNTIME_ROOT_NAME}
    import_paths = _normalized_package_import_paths(packages, package_import_paths)
    if runtime_probe is not None and not isinstance(runtime_probe, RuntimeInterpreterProbe):
        raise TypeError("runtime_probe must be RuntimeInterpreterProbe")
    if runtime_probe is not None and base_sys_path is not None:
        raise CalibrationRunError("runtime probe and explicit base sys.path are mutually exclusive")
    if runtime_probe is None and base_sys_path is None:
        _verify_runtime_capture_launch_state()
    frozen_base_sys_path = _capture_base_sys_path(
        roots[BASE_RUNTIME_ROOT_NAME],
        runtime_probe.base_sys_path if runtime_probe is not None else base_sys_path,
    )
    trees: list[RuntimeTreeRecord] = [
        RuntimeTreeRecord(
            name=BASE_RUNTIME_ROOT_NAME,
            kind="base-runtime",
            files=_runtime_tree_files(roots[BASE_RUNTIME_ROOT_NAME], kind="base-runtime"),
        )
    ]
    for name, root in packages.items():
        trees.append(
            RuntimeTreeRecord(
                name=name,
                kind="packages",
                files=_runtime_tree_files(root, kind=f"package root {name}"),
            )
        )
    selected_interpreter = (
        Path(sys.executable) if interpreter_path is None else Path(interpreter_path)
    )
    try:
        resolved_interpreter = selected_interpreter.resolve(strict=True)
        derived_relative = resolved_interpreter.relative_to(
            roots[BASE_RUNTIME_ROOT_NAME]
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise CalibrationRunError("interpreter is outside the staged base runtime root") from exc
    relative_interpreter = (
        derived_relative
        if interpreter_relative_path is None
        else _canonical_relative_path(
            interpreter_relative_path,
            context="runtime interpreter relative path",
        )
    )
    if relative_interpreter != derived_relative:
        raise CalibrationRunError("interpreter relative path differs from its staged location")
    base_files = {item.path: item for item in trees[0].files}
    if relative_interpreter not in base_files:
        raise CalibrationRunError("staged interpreter is absent from the complete base tree")

    installed = _installed_distribution_map(
        distributions,
        package_roots=packages,
        package_import_paths=import_paths,
    )
    distribution_records: list[RuntimeDistributionRecord] = []
    for name, distribution in sorted(installed.items()):
        distribution_records.append(
            _distribution_record(
                distribution,
                name=name,
                package_roots=packages,
                package_import_paths=import_paths,
            )
        )
    tree_by_name = {tree.name: tree for tree in trees}
    for root_name in packages:
        expected = {item.path for item in tree_by_name[root_name].files}
        claimed: set[str] = set()
        for distribution in distribution_records:
            if distribution.package_root != root_name:
                continue
            overlap = claimed.intersection(distribution.files)
            if overlap:
                raise CalibrationRunError(
                    f"package root {root_name} has duplicate RECORD ownership: {sorted(overlap)}"
                )
            claimed.update(distribution.files)
        if claimed != expected:
            raise CalibrationRunError(
                f"package root {root_name} differs from the complete RECORD inventory; "
                f"missing={sorted(expected - claimed)}, extra={sorted(claimed - expected)}"
            )
    interpreter = base_files[relative_interpreter]
    current_machine = _current_machine_identity()
    machine_identity = (
        (
            runtime_probe.machine_system,
            runtime_probe.machine_architecture,
            runtime_probe.machine_name,
            runtime_probe.machine_byteorder,
            runtime_probe.machine_pointer_bits,
        )
        if runtime_probe is not None
        else current_machine
    )
    python_identity = (
        (
            runtime_probe.python_implementation,
            runtime_probe.python_version,
            runtime_probe.python_cache_tag,
            runtime_probe.python_abi_flags,
        )
        if runtime_probe is not None
        else (
            platform.python_implementation(),
            platform.python_version(),
            sys.implementation.cache_tag,
            getattr(sys, "abiflags", ""),
        )
    )
    document = {
        "artifact_kind": RUNTIME_MANIFEST_KIND,
        "base_sys_path": list(frozen_base_sys_path),
        "base_runtime_root": BASE_RUNTIME_ROOT_NAME,
        "distributions": [
            {
                "files": list(item.files),
                "name": item.name,
                "package_root": item.package_root,
                "version": item.version,
            }
            for item in distribution_records
        ],
        "interpreter": {
            "relative_path": relative_interpreter,
            "root": BASE_RUNTIME_ROOT_NAME,
            "sha256": interpreter.sha256,
            "size_bytes": interpreter.size_bytes,
        },
        "git_executable": {
            "absolute_path_sha256": git_executable.absolute_path_sha256,
            "sha256": git_executable.sha256,
            "size_bytes": git_executable.size_bytes,
        },
        "launch_policy": dict(SEALED_LAUNCH_POLICY),
        "machine": dict(
            zip(
                ("system", "architecture", "machine", "byteorder", "pointer_bits"),
                machine_identity,
                strict=True,
            )
        ),
        "package_roots": [{"import_path": import_paths[name], "name": name} for name in packages],
        "python": {
            "abi_flags": python_identity[3],
            "cache_tag": python_identity[2],
            "implementation": python_identity[0],
            "version": python_identity[1],
        },
        "runtime_trees": [
            {
                "files": [
                    {
                        "path": file.path,
                        "sha256": file.sha256,
                        "size_bytes": file.size_bytes,
                    }
                    for file in tree.files
                ],
                "kind": tree.kind,
                "name": tree.name,
            }
            for tree in trees
        ],
        "schema_version": RUNTIME_MANIFEST_SCHEMA,
    }
    payload = canonical_json_bytes(document)
    parse_calibration_runtime_manifest(payload)
    return payload


def _parse_runtime_tree(raw_tree: object, *, index: int) -> RuntimeTreeRecord:
    if not isinstance(raw_tree, dict):
        raise ValueError(f"runtime_trees[{index}] must be an object")
    _exact_fields(raw_tree, {"files", "kind", "name"}, context=f"runtime_trees[{index}]")
    name = _runtime_root_name(raw_tree["name"], context=f"runtime_trees[{index}].name")
    kind = raw_tree["kind"]
    if kind not in {"base-runtime", "packages"}:
        raise ValueError(f"runtime_trees[{index}].kind is invalid")
    raw_files = raw_tree["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"runtime_trees[{index}].files must be non-empty")
    files: list[RuntimeFileRecord] = []
    for file_index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ValueError(f"runtime_trees[{index}].files[{file_index}] must be an object")
        _exact_fields(
            raw_file,
            {"path", "sha256", "size_bytes"},
            context=f"runtime_trees[{index}].files[{file_index}]",
        )
        path = _canonical_relative_path(
            raw_file["path"],
            context=f"runtime tree {name} file path",
        )
        if _runtime_path_is_forbidden(path):
            raise ValueError(f"runtime tree {name} contains a forbidden file")
        files.append(
            RuntimeFileRecord(
                path=path,
                size_bytes=_nonnegative_int(
                    raw_file["size_bytes"], context=f"runtime tree {name} file size"
                ),
                sha256=_sha256(raw_file["sha256"], context=f"runtime tree {name} file SHA-256"),
            )
        )
    if [item.path for item in files] != sorted(item.path for item in files) or len(
        {item.path.casefold() for item in files}
    ) != len(files):
        raise ValueError(f"runtime tree {name} file paths must be unique and sorted")
    return RuntimeTreeRecord(name=name, kind=cast(str, kind), files=tuple(files))


def parse_calibration_runtime_manifest(data: bytes) -> CalibrationRuntimeManifest:
    root = _strict_json_bytes(data, context="calibration runtime manifest")
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
        context="calibration runtime manifest",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("calibration runtime manifest is not canonical JSON")
    if (
        root["artifact_kind"] != RUNTIME_MANIFEST_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != RUNTIME_MANIFEST_SCHEMA
    ):
        raise ValueError("calibration runtime manifest kind or schema drifted")
    _exact_typed_mapping(
        root["launch_policy"],
        SEALED_LAUNCH_POLICY,
        context="calibration runtime launch policy",
    )

    raw_git_executable = root["git_executable"]
    if not isinstance(raw_git_executable, dict):
        raise ValueError("calibration runtime Git executable record must be an object")
    _exact_fields(
        raw_git_executable,
        {"absolute_path_sha256", "sha256", "size_bytes"},
        context="calibration runtime Git executable",
    )
    git_absolute_path_sha256 = _sha256(
        raw_git_executable["absolute_path_sha256"],
        context="runtime Git executable absolute-path SHA-256",
    )
    git_sha256 = _sha256(
        raw_git_executable["sha256"],
        context="runtime Git executable SHA-256",
    )
    git_size_bytes = _positive_int(
        raw_git_executable["size_bytes"],
        context="runtime Git executable size",
    )

    python_record = root["python"]
    if not isinstance(python_record, dict):
        raise ValueError("calibration runtime python record must be an object")
    _exact_fields(
        python_record,
        {"abi_flags", "cache_tag", "implementation", "version"},
        context="calibration runtime python record",
    )
    for field in ("abi_flags", "cache_tag", "implementation", "version"):
        value = python_record[field]
        invalid_empty = field != "abi_flags" and (not value or value != value.strip())
        if not isinstance(value, str) or invalid_empty:
            raise ValueError(f"calibration runtime python {field} is invalid")

    machine = root["machine"]
    if not isinstance(machine, dict):
        raise ValueError("calibration runtime machine record must be an object")
    _exact_fields(
        machine,
        {"architecture", "byteorder", "machine", "pointer_bits", "system"},
        context="calibration runtime machine record",
    )
    for field in ("architecture", "machine", "system"):
        value = machine[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"calibration runtime machine {field} is invalid")
    if machine["byteorder"] not in {"little", "big"}:
        raise ValueError("calibration runtime machine byteorder is invalid")
    pointer_bits = _positive_int(machine["pointer_bits"], context="machine pointer_bits")

    base_root = _runtime_root_name(root["base_runtime_root"], context="base runtime root")
    if base_root != BASE_RUNTIME_ROOT_NAME:
        raise ValueError("base runtime root name drifted")
    raw_base_sys_path = root["base_sys_path"]
    if not isinstance(raw_base_sys_path, list) or not raw_base_sys_path:
        raise ValueError("base_sys_path must be a non-empty list")
    base_sys_path = tuple(
        _canonical_base_sys_path_entry(item, context="base sys.path entry")
        for item in raw_base_sys_path
    )
    if len({item.casefold() for item in base_sys_path}) != len(base_sys_path):
        raise ValueError("base_sys_path entries must be case-insensitively unique")
    raw_package_roots = root["package_roots"]
    if not isinstance(raw_package_roots, list) or not raw_package_roots:
        raise ValueError("package_roots must be a non-empty list")
    parsed_package_roots: list[RuntimePackageRootRecord] = []
    for index, item in enumerate(raw_package_roots):
        if not isinstance(item, dict):
            raise ValueError(f"package_roots[{index}] must be an object")
        _exact_fields(
            item,
            {"import_path", "name"},
            context=f"package_roots[{index}]",
        )
        parsed_package_roots.append(
            RuntimePackageRootRecord(
                name=_runtime_root_name(item["name"], context="package root name"),
                import_path=_canonical_relative_path(
                    item["import_path"],
                    context="package root import path",
                ),
            )
        )
    package_root_names = tuple(item.name for item in parsed_package_roots)
    if (
        list(package_root_names) != sorted(package_root_names)
        or len(set(package_root_names)) != len(package_root_names)
        or base_root in package_root_names
    ):
        raise ValueError("package_roots must be unique, sorted, and distinct from base runtime")

    raw_trees = root["runtime_trees"]
    if not isinstance(raw_trees, list):
        raise ValueError("runtime_trees must be a list")
    trees = tuple(_parse_runtime_tree(item, index=index) for index, item in enumerate(raw_trees))
    if tuple(item.name for item in trees) != (base_root, *package_root_names):
        raise ValueError("runtime tree order or exact root inventory drifted")
    if trees[0].kind != "base-runtime" or any(tree.kind != "packages" for tree in trees[1:]):
        raise ValueError("runtime tree kinds drifted")

    interpreter = root["interpreter"]
    if not isinstance(interpreter, dict):
        raise ValueError("calibration runtime interpreter record must be an object")
    _exact_fields(
        interpreter,
        {"relative_path", "root", "sha256", "size_bytes"},
        context="runtime interpreter",
    )
    if interpreter["root"] != base_root:
        raise ValueError("runtime interpreter is not bound to the base runtime root")
    interpreter_path = _canonical_relative_path(
        interpreter["relative_path"], context="runtime interpreter relative path"
    )
    interpreter_sha256 = _sha256(interpreter["sha256"], context="runtime interpreter SHA-256")
    interpreter_size = _positive_int(interpreter["size_bytes"], context="runtime interpreter size")
    base_files = {item.path: item for item in trees[0].files}
    for sys_path_entry in base_sys_path:
        present = (
            sys_path_entry == "."
            or sys_path_entry in base_files
            or any(path.startswith(f"{sys_path_entry}/") for path in base_files)
        )
        optional_zip = re.fullmatch(r"python[0-9]+\.zip", sys_path_entry) is not None
        if not present and not optional_zip:
            raise ValueError("base_sys_path entry is absent from the complete base tree")
    if base_files.get(interpreter_path) != RuntimeFileRecord(
        interpreter_path,
        interpreter_size,
        interpreter_sha256,
    ):
        raise ValueError("runtime interpreter identity differs from the complete base tree")

    raw_distributions = root["distributions"]
    if not isinstance(raw_distributions, list) or not raw_distributions:
        raise ValueError("calibration runtime distributions must be a non-empty list")
    parsed: list[RuntimeDistributionRecord] = []
    ownership: dict[str, set[str]] = {name: set() for name in package_root_names}
    for index, raw_distribution in enumerate(raw_distributions):
        if not isinstance(raw_distribution, dict):
            raise ValueError(f"runtime distributions[{index}] must be an object")
        _exact_fields(
            raw_distribution,
            {"files", "name", "package_root", "version"},
            context=f"runtime distributions[{index}]",
        )
        name = _normalized_distribution_name(raw_distribution["name"])
        if name != raw_distribution["name"]:
            raise ValueError("runtime distribution names must already be canonical")
        version = raw_distribution["version"]
        if not isinstance(version, str) or not version or version != version.strip():
            raise ValueError(f"runtime distribution {name} version is invalid")
        package_root = _runtime_root_name(
            raw_distribution["package_root"],
            context=f"runtime distribution {name} package_root",
        )
        if package_root not in ownership:
            raise ValueError(f"runtime distribution {name} uses an unknown package root")
        raw_files = raw_distribution["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"runtime distribution {name} files must be non-empty")
        files = tuple(
            _canonical_relative_path(path, context=f"runtime distribution {name} file path")
            for path in raw_files
        )
        if list(files) != sorted(files) or len({path.casefold() for path in files}) != len(files):
            raise ValueError(f"runtime distribution {name} files must be unique and sorted")
        overlap = ownership[package_root].intersection(files)
        if overlap:
            raise ValueError(
                f"runtime package file has duplicate distribution ownership: {sorted(overlap)}"
            )
        ownership[package_root].update(files)
        parsed.append(RuntimeDistributionRecord(name, cast(str, version), package_root, files))
    if [item.name for item in parsed] != sorted(item.name for item in parsed) or len(
        {item.name for item in parsed}
    ) != len(parsed):
        raise ValueError("runtime distributions must be unique and sorted")
    tree_by_name = {item.name: item for item in trees}
    for root_name in package_root_names:
        if ownership[root_name] != {item.path for item in tree_by_name[root_name].files}:
            raise ValueError(
                f"package tree {root_name} differs from its complete distribution inventory"
            )

    return CalibrationRuntimeManifest(
        python_implementation=cast(str, python_record["implementation"]),
        python_version=cast(str, python_record["version"]),
        python_cache_tag=cast(str, python_record["cache_tag"]),
        python_abi_flags=cast(str, python_record["abi_flags"]),
        machine_system=cast(str, machine["system"]),
        machine_architecture=cast(str, machine["architecture"]),
        machine_name=cast(str, machine["machine"]),
        machine_byteorder=cast(str, machine["byteorder"]),
        machine_pointer_bits=pointer_bits,
        launch_policy=dict(SEALED_LAUNCH_POLICY),
        base_sys_path=base_sys_path,
        base_runtime_root=base_root,
        package_roots=tuple(parsed_package_roots),
        interpreter_root=base_root,
        interpreter_relative_path=interpreter_path,
        interpreter_size_bytes=interpreter_size,
        interpreter_sha256=interpreter_sha256,
        git_executable_absolute_path_sha256=git_absolute_path_sha256,
        git_executable_sha256=git_sha256,
        git_executable_size_bytes=git_size_bytes,
        runtime_trees=trees,
        distributions=tuple(parsed),
        file_sha256=sha256_bytes(data),
    )


def authenticate_calibration_runtime(
    manifest: CalibrationRuntimeManifest,
    *,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    distributions: Sequence[Any] | None = None,
    interpreter_path: Path | None = None,
    git_executable_path: Path | None = None,
) -> AuthenticatedRuntime:
    """Rehash both complete staged trees and exact RECORD inventories."""

    if not isinstance(manifest, CalibrationRuntimeManifest):
        raise TypeError("manifest must be CalibrationRuntimeManifest")
    git_executable = _authenticate_git_executable(git_executable_path)
    if (
        git_executable.absolute_path_sha256 != manifest.git_executable_absolute_path_sha256
        or git_executable.sha256 != manifest.git_executable_sha256
        or git_executable.size_bytes != manifest.git_executable_size_bytes
    ):
        raise CalibrationRunError("Git executable differs from the frozen runtime manifest")
    roots = _runtime_root_map(Path(base_runtime_root), package_roots)
    packages = {name: roots[name] for name in sorted(roots) if name != BASE_RUNTIME_ROOT_NAME}
    manifest_package_names = tuple(item.name for item in manifest.package_roots)
    if tuple(packages) != manifest_package_names:
        raise CalibrationRunError("point-used package roots differ from the frozen manifest")
    import_paths = _normalized_package_import_paths(
        packages,
        {item.name: item.import_path for item in manifest.package_roots},
    )
    if (
        manifest.python_implementation != platform.python_implementation()
        or manifest.python_version != platform.python_version()
        or manifest.python_cache_tag != sys.implementation.cache_tag
        or manifest.python_abi_flags != getattr(sys, "abiflags", "")
    ):
        raise CalibrationRunError("Python runtime differs from the frozen runtime manifest")
    if _current_machine_identity() != (
        manifest.machine_system,
        manifest.machine_architecture,
        manifest.machine_name,
        manifest.machine_byteorder,
        manifest.machine_pointer_bits,
    ):
        raise CalibrationRunError("machine identity differs from the frozen runtime manifest")

    actual_trees: list[RuntimeTreeRecord] = [
        RuntimeTreeRecord(
            BASE_RUNTIME_ROOT_NAME,
            "base-runtime",
            _runtime_tree_files(roots[BASE_RUNTIME_ROOT_NAME], kind="base-runtime"),
        )
    ]
    actual_trees.extend(
        RuntimeTreeRecord(
            name,
            "packages",
            _runtime_tree_files(path, kind=f"package root {name}"),
        )
        for name, path in packages.items()
    )
    if tuple(actual_trees) != manifest.runtime_trees:
        raise CalibrationRunError("complete staged runtime tree identity drifted")

    expected_interpreter = _assert_no_link_components(
        roots[BASE_RUNTIME_ROOT_NAME],
        PurePosixPath(manifest.interpreter_relative_path),
    )
    actual_interpreter = (
        Path(sys.executable).resolve(strict=True)
        if interpreter_path is None
        else Path(interpreter_path).resolve(strict=True)
    )
    if actual_interpreter != expected_interpreter:
        raise CalibrationRunError("point-used interpreter path differs from the runtime manifest")
    interpreter_sha256, interpreter_size = _stream_file_sha256(actual_interpreter)
    if (
        interpreter_sha256 != manifest.interpreter_sha256
        or interpreter_size != manifest.interpreter_size_bytes
    ):
        raise CalibrationRunError("Python interpreter bytes differ from the runtime manifest")

    installed = _installed_distribution_map(
        distributions,
        package_roots=packages,
        package_import_paths=import_paths,
    )
    if sorted(installed) != [item.name for item in manifest.distributions]:
        raise CalibrationRunError("installed distribution set differs from the frozen manifest")
    for expected in manifest.distributions:
        actual = _distribution_record(
            installed[expected.name],
            name=expected.name,
            package_roots=packages,
            package_import_paths=import_paths,
        )
        if actual != expected:
            raise CalibrationRunError(f"installed distribution drifted: {expected.name}")
    total_files = sum(len(tree.files) for tree in manifest.runtime_trees)
    return AuthenticatedRuntime(
        manifest_file_sha256=manifest.file_sha256,
        python_implementation=manifest.python_implementation,
        python_version=manifest.python_version,
        python_cache_tag=manifest.python_cache_tag,
        interpreter_sha256=manifest.interpreter_sha256,
        git_executable_absolute_path_sha256=manifest.git_executable_absolute_path_sha256,
        git_executable_sha256=manifest.git_executable_sha256,
        git_executable_size_bytes=manifest.git_executable_size_bytes,
        machine_name=manifest.machine_name,
        base_runtime_file_count=len(manifest.runtime_trees[0].files),
        package_root_count=len(manifest.package_roots),
        distributions=tuple((item.name, item.version) for item in manifest.distributions),
        distribution_count=len(manifest.distributions),
        file_count=total_files,
    )


def _current_machine_identity() -> tuple[str, str, str, str, int]:
    pointer_bits = 8 * struct.calcsize("P")
    return (
        platform.system(),
        f"{pointer_bits}bit",
        platform.machine(),
        sys.byteorder,
        pointer_bits,
    )


def _parse_named_path_arguments(values: Sequence[str], *, context: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise CalibrationRunError(f"{context} must use NAME=PATH")
        raw_name, raw_path = value.split("=", 1)
        name = _runtime_root_name(raw_name, context=f"{context} name")
        if name in result or not raw_path or raw_path != raw_path.strip():
            raise CalibrationRunError(f"{context} contains a duplicate name or invalid path")
        path = Path(raw_path)
        if not path.is_absolute():
            raise CalibrationRunError(f"{context} paths must be absolute")
        result[name] = path
    if not result:
        raise CalibrationRunError(f"{context} must contain at least one entry")
    return {name: result[name] for name in sorted(result)}


def _parse_named_import_path_arguments(
    values: Sequence[str],
    *,
    context: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise CalibrationRunError(f"{context} must use NAME=RELATIVE_PATH")
        raw_name, raw_path = value.split("=", 1)
        name = _runtime_root_name(raw_name, context=f"{context} name")
        if name in result:
            raise CalibrationRunError(f"{context} contains a duplicate name")
        result[name] = _canonical_relative_path(
            raw_path,
            context=f"{context} relative path",
        )
    if not result:
        raise CalibrationRunError(f"{context} must contain at least one entry")
    return {name: result[name] for name in sorted(result)}


def _read_pyvenv_config(path: Path) -> dict[str, str]:
    if _is_link_or_reparse(path):
        raise CalibrationRunError("source pyvenv.cfg is a link or reparse point")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CalibrationRunError("source pyvenv.cfg is not readable UTF-8") from exc
    result: dict[str, str] = {}
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise CalibrationRunError(f"source pyvenv.cfg line {index} is malformed")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip().casefold()
        value = raw_value.strip()
        if not key or not value or key in result:
            raise CalibrationRunError("source pyvenv.cfg has a duplicate or empty field")
        result[key] = value
    return result


def _source_runtime_layout(source_python: Path) -> tuple[Path, Path, str]:
    executable = Path(os.path.abspath(source_python))
    if _is_link_or_reparse(executable):
        raise CalibrationRunError("source interpreter is a link or reparse point")
    try:
        executable = executable.resolve(strict=True)
    except OSError as exc:
        raise CalibrationRunError("source interpreter is unavailable") from exc
    if not executable.is_file():
        raise CalibrationRunError("source interpreter is not a regular file")

    virtual_root = executable.parent.parent
    config_path = virtual_root / "pyvenv.cfg"
    if executable.parent.name.casefold() == "scripts" and config_path.is_file():
        _assert_no_link_components(virtual_root, PurePosixPath("pyvenv.cfg"))
        config = _read_pyvenv_config(config_path)
        raw_home = config.get("home")
        if raw_home is None or not Path(raw_home).is_absolute():
            raise CalibrationRunError("source pyvenv.cfg has no absolute base home")
        try:
            physical_home = Path(raw_home).resolve(strict=True)
        except OSError as exc:
            raise CalibrationRunError("source pyvenv.cfg base home is unavailable") from exc
        base_root = _absolute_runtime_root(physical_home, context="source base runtime root")
        package_root = _absolute_runtime_root(virtual_root, context="source virtualenv root")
    else:
        base_root = _absolute_runtime_root(executable.parent, context="source base runtime root")
        package_root = base_root
    import_relative = DEFAULT_PACKAGE_IMPORT_PATH
    import_root = package_root / PurePosixPath(import_relative)
    if not import_root.is_dir() or _is_link_or_reparse(import_root):
        raise CalibrationRunError("source environment has no regular Lib/site-packages")
    return base_root, package_root, import_relative


def _wheel_requirement_version(name: str, url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    fragment = urllib.parse.parse_qs(parsed.fragment, strict_parsing=True)
    hashes = fragment.get("sha256")
    if parsed.scheme != "https" or hashes is None or len(hashes) != 1:
        raise CalibrationRunError(f"direct requirement {name} must have one HTTPS SHA-256")
    _sha256(hashes[0], context=f"direct requirement {name} SHA-256")
    filename = PurePosixPath(urllib.parse.unquote(parsed.path)).name
    match = re.fullmatch(
        r"(?P<distribution>.+)-(?P<version>[^-]+)-[^-]+-[^-]+-[^-]+\.whl",
        filename,
    )
    if match is None or _normalized_distribution_name(match.group("distribution")) != name:
        raise CalibrationRunError(f"direct requirement {name} has an invalid wheel URL")
    return match.group("version")


def _parse_runtime_requirements(path: Path) -> tuple[RuntimeRequirement, ...]:
    requirements_path = Path(os.path.abspath(path))
    if _is_link_or_reparse(requirements_path):
        raise CalibrationRunError("runtime requirements file is a link or reparse point")
    try:
        text = requirements_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CalibrationRunError("runtime requirements file is not readable UTF-8") from exc
    selected: dict[str, RuntimeRequirement] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pinned = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)", line)
        direct = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*@\s*(\S+)", line)
        if pinned is not None:
            name = _normalized_distribution_name(pinned.group(1))
            version = pinned.group(2)
        elif direct is not None:
            name = _normalized_distribution_name(direct.group(1))
            version = _wheel_requirement_version(name, direct.group(2))
        else:
            raise CalibrationRunError(
                f"runtime requirement line {line_number} is not one exact pin"
            )
        if name in selected:
            raise CalibrationRunError(f"runtime requirements repeat distribution {name}")
        if name not in STAGING_EXCLUDED_DISTRIBUTIONS:
            selected[name] = RuntimeRequirement(name=name, version=version)
    if not selected:
        raise CalibrationRunError("runtime requirements select no loadable distributions")
    return tuple(selected[name] for name in sorted(selected))


def _source_regular_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    if _is_link_or_reparse(candidate):
        raise CalibrationRunError("source runtime root is a link or reparse point")
    for part in relative.parts:
        candidate /= part
        if _is_link_or_reparse(candidate):
            raise CalibrationRunError(
                f"source runtime path traverses a link or reparse point: {relative.as_posix()}"
            )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CalibrationRunError(
            f"source runtime path escapes its explicit root: {relative.as_posix()}"
        ) from exc
    if not resolved.is_file():
        raise CalibrationRunError(
            f"source runtime path is not a regular file: {relative.as_posix()}"
        )
    return resolved


def _copy_independent_runtime_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise CalibrationRunError(f"runtime staging path collision: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if os.path.samefile(source, destination):
        raise CalibrationRunError("runtime staging created a shared file instead of copied bytes")
    source_sha256, source_size = _stream_file_sha256(source)
    destination_sha256, destination_size = _stream_file_sha256(destination)
    if (source_sha256, source_size) != (destination_sha256, destination_size):
        raise CalibrationRunError("runtime staging changed copied file bytes")


def _copy_base_runtime(source_root: Path, destination_root: Path) -> None:
    selected_directories = frozenset({"dlls", "lib"})
    stack: list[tuple[Path, tuple[str, ...]]] = [(source_root, ())]
    copied = 0
    while stack:
        directory, parts = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CalibrationRunError("cannot enumerate source base runtime") from exc
        for entry in entries:
            child_parts = (*parts, entry.name)
            relative = PurePosixPath(*child_parts).as_posix()
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CalibrationRunError("source base runtime changed during staging") from exc
            if entry.is_symlink() or (
                getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise CalibrationRunError(
                    f"source base runtime contains a link or reparse point: {relative}"
                )
            if stat.S_ISDIR(status.st_mode):
                if not parts and entry.name.casefold() not in selected_directories:
                    continue
                if (
                    len(parts) == 1
                    and parts[0].casefold() == "lib"
                    and entry.name.casefold() == "site-packages"
                ):
                    continue
                if entry.name.casefold() in _FORBIDDEN_RUNTIME_DIRECTORY_NAMES:
                    continue
                stack.append((Path(entry.path), child_parts))
                continue
            if not stat.S_ISREG(status.st_mode):
                raise CalibrationRunError(f"source base runtime path is not regular: {relative}")
            if _runtime_path_is_forbidden(relative):
                continue
            source = _source_regular_file(source_root, PurePosixPath(*child_parts))
            _copy_independent_runtime_file(source, destination_root / Path(*child_parts))
            copied += 1
    if copied == 0:
        raise CalibrationRunError("source base runtime contributed no staged files")


def _selected_source_distributions(
    source_package_root: Path,
    import_path: str,
    requirements: Sequence[RuntimeRequirement],
) -> tuple[Any, ...]:
    import_root = source_package_root / PurePosixPath(import_path)
    installed = _installed_distribution_map(
        list(importlib.metadata.distributions(path=[str(import_root)]))
    )
    result: list[Any] = []
    for requirement in requirements:
        distribution = installed.get(requirement.name)
        if distribution is None:
            raise CalibrationRunError(
                f"source environment lacks required distribution {requirement.name}"
            )
        if str(distribution.version) != requirement.version:
            raise CalibrationRunError(
                f"source distribution version differs from exact pin: {requirement.name}"
            )
        result.append(distribution)
    return tuple(result)


def _copy_record_only_packages(
    source_root: Path,
    destination_root: Path,
    distributions: Sequence[Any],
) -> None:
    copied: set[str] = set()
    for distribution in distributions:
        name = _normalized_distribution_name(distribution.metadata.get("Name"))
        for record_path in _record_paths(distribution, name=name):
            located = Path(distribution.locate_file(record_path))
            try:
                resolved = located.resolve(strict=True)
                relative = resolved.relative_to(source_root)
            except (OSError, ValueError) as exc:
                raise CalibrationRunError(
                    f"distribution {name} RECORD path escapes the source environment"
                ) from exc
            posix_relative = _canonical_relative_path(
                relative.as_posix(),
                context=f"distribution {name} staged path",
            )
            if _runtime_path_is_forbidden(posix_relative):
                raise CalibrationRunError(
                    f"distribution {name} owns a forbidden startup or bytecode file"
                )
            folded = posix_relative.casefold()
            if folded in copied:
                raise CalibrationRunError(
                    f"selected distributions have duplicate RECORD ownership: {posix_relative}"
                )
            source = _source_regular_file(source_root, PurePosixPath(posix_relative))
            _copy_independent_runtime_file(
                source,
                destination_root / Path(*PurePosixPath(posix_relative).parts),
            )
            copied.add(folded)
    if not copied:
        raise CalibrationRunError("selected distributions contributed no staged package files")


_RUNTIME_PROBE_SOURCE: Final = r"""
import json
import os
import platform
import struct
import sys

root = os.path.realpath(sys.argv[1])
if os.path.realpath(sys.prefix) != root or os.path.realpath(sys.base_prefix) != root:
    raise SystemExit("prefix drift")
if "site" in sys.modules or "_virtualenv" in sys.modules:
    raise SystemExit("startup hook loaded")
if not (
    sys.flags.isolated == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.no_user_site == 1
    and sys.flags.no_site == 1
    and sys.flags.dont_write_bytecode == 1
    and sys.flags.utf8_mode == 1
    and sys.flags.safe_path is True
):
    raise SystemExit("flag drift")
if sys._xoptions != {"utf8": True}:
    raise SystemExit("xoption drift")
paths = []
for item in sys.path:
    if not isinstance(item, str) or not item or not os.path.isabs(item):
        raise SystemExit("relative sys.path")
    relative = os.path.relpath(os.path.realpath(item), root)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise SystemExit("sys.path escape")
    paths.append(relative.replace(os.sep, "/"))
bits = 8 * struct.calcsize("P")
payload = {
    "base_sys_path": paths,
    "machine": {
        "architecture": f"{bits}bit",
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "pointer_bits": bits,
        "system": platform.system(),
    },
    "python": {
        "abi_flags": getattr(sys, "abiflags", ""),
        "cache_tag": sys.implementation.cache_tag,
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    },
}
print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
""".strip()


def _probe_staged_interpreter(
    interpreter: Path,
    base_runtime_root: Path,
) -> RuntimeInterpreterProbe:
    process = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            "-c",
            _RUNTIME_PROBE_SOURCE,
            str(base_runtime_root),
        ],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if process.returncode != 0 or process.stderr:
        raise CalibrationRunError("staged interpreter failed the isolated stdlib probe")
    root = _strict_json_bytes(process.stdout, context="staged interpreter probe")
    _exact_fields(root, {"base_sys_path", "machine", "python"}, context="runtime probe")
    raw_python = root["python"]
    raw_machine = root["machine"]
    raw_paths = root["base_sys_path"]
    if not isinstance(raw_python, dict) or not isinstance(raw_machine, dict):
        raise CalibrationRunError("staged interpreter probe identity is malformed")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise CalibrationRunError("staged interpreter probe sys.path is malformed")
    _exact_fields(
        raw_python,
        {"abi_flags", "cache_tag", "implementation", "version"},
        context="runtime probe python",
    )
    _exact_fields(
        raw_machine,
        {"architecture", "byteorder", "machine", "pointer_bits", "system"},
        context="runtime probe machine",
    )
    return RuntimeInterpreterProbe(
        python_implementation=cast(str, raw_python["implementation"]),
        python_version=cast(str, raw_python["version"]),
        python_cache_tag=cast(str, raw_python["cache_tag"]),
        python_abi_flags=cast(str, raw_python["abi_flags"]),
        machine_system=cast(str, raw_machine["system"]),
        machine_architecture=cast(str, raw_machine["architecture"]),
        machine_name=cast(str, raw_machine["machine"]),
        machine_byteorder=cast(str, raw_machine["byteorder"]),
        machine_pointer_bits=_positive_int(
            raw_machine["pointer_bits"],
            context="runtime probe pointer_bits",
        ),
        base_sys_path=tuple(
            _canonical_base_sys_path_entry(item, context="runtime probe base sys.path")
            for item in raw_paths
        ),
    )


def prepare_calibration_runtime(
    *,
    source_python: Path,
    requirements_file: Path,
    output_root: Path,
    git_executable_path: Path | None = None,
    package_root_name: str = DEFAULT_PACKAGE_RUNTIME_ROOT_NAME,
) -> dict[str, object]:
    """Stage independent base bytes and only exact wheel-RECORD package bytes."""

    name = _runtime_root_name(package_root_name, context="prepared package root name")
    if name == BASE_RUNTIME_ROOT_NAME:
        raise CalibrationRunError("prepared package root name is reserved")
    destination = Path(os.path.abspath(output_root))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prepared runtime: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_base, source_packages, import_path = _source_runtime_layout(source_python)
    requirements = _parse_runtime_requirements(requirements_file)
    distributions = _selected_source_distributions(
        source_packages,
        import_path,
        requirements,
    )

    prefix = f".{destination.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=destination.parent))
    owned_staging = True
    try:
        base_root = staging / BASE_RUNTIME_ROOT_NAME
        package_root = staging / name
        base_root.mkdir()
        package_root.mkdir()
        _copy_base_runtime(source_base, base_root)
        _copy_record_only_packages(source_packages, package_root, distributions)
        source_executable_name = Path(source_python).name
        staged_interpreter = base_root / source_executable_name
        if not staged_interpreter.is_file():
            raise CalibrationRunError("staged base runtime omitted the selected interpreter")
        probe = _probe_staged_interpreter(staged_interpreter, base_root)
        package_roots = {name: package_root}
        package_import_paths = {name: import_path}
        payload = capture_calibration_runtime_manifest(
            git_executable_path=git_executable_path,
            base_runtime_root=base_root,
            package_roots=package_roots,
            package_import_paths=package_import_paths,
            interpreter_path=staged_interpreter,
            runtime_probe=probe,
        )
        manifest_path = staging / PREPARED_RUNTIME_MANIFEST_FILENAME
        _atomic_publish_new(manifest_path, payload)
        if manifest_path.read_bytes() != payload:
            raise CalibrationRunError("prepared runtime manifest changed after publication")
        repeated = capture_calibration_runtime_manifest(
            git_executable_path=git_executable_path,
            base_runtime_root=base_root,
            package_roots=package_roots,
            package_import_paths=package_import_paths,
            interpreter_path=staged_interpreter,
            runtime_probe=probe,
        )
        if repeated != payload:
            raise CalibrationRunError("prepared runtime trees changed before publication")
        _atomic_publish_new(
            staging / PREPARED_RUNTIME_COMPLETE_FILENAME,
            b"recurquant-experiment013-runtime-prepared-v1\n",
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite prepared runtime: {destination}")
        staging.rename(destination)
        owned_staging = False
    finally:
        if owned_staging:
            try:
                staging.relative_to(destination.parent)
            except ValueError as exc:
                raise RuntimeError("owned runtime staging directory escaped its parent") from exc
            if not staging.name.startswith(prefix):
                raise RuntimeError("owned runtime staging directory name drifted")
            shutil.rmtree(staging, ignore_errors=False)
    return {
        "base_runtime_root": str(destination / BASE_RUNTIME_ROOT_NAME),
        "excluded_distributions": sorted(STAGING_EXCLUDED_DISTRIBUTIONS),
        "manifest_file_sha256": sha256_bytes(payload),
        "package_import_path": import_path,
        "package_root": str(destination / name),
        "package_root_name": name,
        "prepared_runtime_root": str(destination),
        "status": "prepared_record_only_runtime",
    }


def _verify_empty_pycache_prefix(path: Path) -> Path:
    root = _absolute_runtime_root(path, context="pycache prefix")
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise CalibrationRunError("cannot enumerate pycache prefix") from exc
    if entries:
        raise CalibrationRunError("pycache prefix is not empty")
    return root


def _verify_sealed_launch_state(
    pycache_prefix: Path,
    *,
    manifest: CalibrationRuntimeManifest,
    roots: Mapping[str, Path],
) -> Path:
    if (
        sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.utf8_mode != 1
        or sys.flags.safe_path is not True
    ):
        raise CalibrationRunError("sealed Python startup flags drifted")
    if "site" in sys.modules or "_virtualenv" in sys.modules:
        raise CalibrationRunError("site or virtualenv startup hook is loaded")
    if sys.pycache_prefix is None:
        raise CalibrationRunError("sealed Python startup has no pycache prefix")
    expected = Path(os.path.abspath(pycache_prefix))
    actual = Path(os.path.abspath(sys.pycache_prefix))
    if actual != expected:
        raise CalibrationRunError("sealed Python pycache prefix drifted")
    if set(sys._xoptions) != {"pycache_prefix", "utf8"} or (
        Path(os.path.abspath(str(sys._xoptions["pycache_prefix"]))) != expected
        or sys._xoptions["utf8"] is not True
    ):
        raise CalibrationRunError("sealed Python -X options drifted")
    base_root = roots[BASE_RUNTIME_ROOT_NAME]
    if (
        Path(sys.prefix).resolve(strict=True) != base_root
        or Path(sys.base_prefix).resolve(strict=True) != base_root
    ):
        raise CalibrationRunError("sealed Python prefix differs from the staged base runtime")
    expected_sys_path = [
        str(base_root / PurePosixPath(relative)) for relative in manifest.base_sys_path
    ]
    expected_sys_path.extend(
        str(roots[item.name] / PurePosixPath(item.import_path)) for item in manifest.package_roots
    )
    if [os.path.abspath(item) for item in sys.path] != [
        os.path.abspath(item) for item in expected_sys_path
    ]:
        raise CalibrationRunError("sealed Python sys.path differs from authenticated roots")
    return _verify_empty_pycache_prefix(expected)


def _authenticate_sealed_runtime_context(
    runtime_manifest_bytes: bytes,
    *,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
    interpreter_path: Path,
    git_executable_path: Path,
    pycache_prefix: Path,
) -> tuple[CalibrationRuntimeManifest, SealedRuntimeContext, AuthenticatedRuntime]:
    """Reauthenticate explicit launcher inputs without copying them to globals."""

    manifest = parse_calibration_runtime_manifest(runtime_manifest_bytes)
    roots = _runtime_root_map(base_runtime_root, package_roots)
    declared_names = tuple(item.name for item in manifest.package_roots)
    actual_names = tuple(name for name in roots if name != BASE_RUNTIME_ROOT_NAME)
    if actual_names != declared_names:
        raise CalibrationRunError("bootstrap package roots differ from the frozen runtime manifest")
    normalized_import_paths = _normalized_package_import_paths(
        {item.name: roots[item.name] for item in manifest.package_roots},
        package_import_paths,
    )
    frozen_import_paths = {item.name: item.import_path for item in manifest.package_roots}
    if normalized_import_paths != frozen_import_paths:
        raise CalibrationRunError(
            "bootstrap package import paths differ from the frozen runtime manifest"
        )
    verified_pycache = _verify_sealed_launch_state(
        pycache_prefix,
        manifest=manifest,
        roots=roots,
    )
    authenticated = authenticate_calibration_runtime(
        manifest,
        base_runtime_root=base_runtime_root,
        package_roots=package_roots,
        interpreter_path=interpreter_path,
        git_executable_path=git_executable_path,
    )
    context = SealedRuntimeContext(
        manifest_file_sha256=manifest.file_sha256,
        base_runtime_root=roots[BASE_RUNTIME_ROOT_NAME],
        package_roots={item.name: roots[item.name] for item in manifest.package_roots},
        package_import_paths=normalized_import_paths,
        git_executable_path=_authenticate_git_executable(git_executable_path).path,
        pycache_prefix=verified_pycache,
    )
    return manifest, context, authenticated


def _load_identity_resolver(repository_root: Path) -> Any:
    module = _AUTHENTICATED_IDENTITY_RESOLVER
    if module is None:
        raise CalibrationRunError("identity resolver was not bootstrap-authenticated")
    expected = (repository_root / IDENTITY_RESOLVER_SOURCE_PATH).resolve(strict=True)
    actual = Path(cast(str, getattr(module, "__file__", ""))).resolve(strict=True)
    if actual != expected:
        raise CalibrationRunError("authenticated identity resolver path drifted")
    return module


def _identity_records_with_fisher_boundary(
    decoded: object,
    evidence: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Preserve each resolver-validated schema-v5 Fisher boundary exactly."""

    raw_records = evidence.get("records")
    decoded_records = getattr(decoded, "records", None)
    if not isinstance(raw_records, list) or not isinstance(decoded_records, (list, tuple)):
        raise CalibrationRunError("schema-v5 frozen identity records are missing")
    if len(raw_records) != len(decoded_records):
        raise CalibrationRunError("decoded schema-v5 record inventory differs from evidence")

    normalized_records: list[dict[str, object]] = []
    for index, (raw_record, decoded_record) in enumerate(
        zip(raw_records, decoded_records, strict=True)
    ):
        if not isinstance(raw_record, Mapping) or not isinstance(decoded_record, Mapping):
            raise CalibrationRunError(f"schema-v5 records[{index}] is not an object")
        raw_boundary = raw_record.get("fisher_boundary")
        decoded_boundary = decoded_record.get("fisher_boundary")
        if not isinstance(raw_boundary, Mapping) or not isinstance(decoded_boundary, Mapping):
            raise CalibrationRunError(f"schema-v5 records[{index}].fisher_boundary is missing")
        try:
            _exact_fields(
                raw_boundary,
                FISHER_BOUNDARY_FIELDS,
                context=f"identity evidence records[{index}].fisher_boundary",
            )
            _exact_fields(
                decoded_boundary,
                FISHER_BOUNDARY_FIELDS,
                context=f"decoded records[{index}].fisher_boundary",
            )
        except ValueError as exc:
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary fields drifted"
            ) from exc
        if dict(decoded_boundary) != dict(raw_boundary):
            raise CalibrationRunError(
                f"decoded records[{index}].fisher_boundary differs from identity evidence"
            )
        if decoded_boundary["schema"] != FISHER_BOUNDARY_SCHEMA:
            raise CalibrationRunError(f"schema-v5 records[{index}].fisher_boundary schema drifted")
        horizon = decoded_boundary["horizon"]
        if type(horizon) is not int or horizon != FISHER_BOUNDARY_HORIZON:
            raise CalibrationRunError(f"schema-v5 records[{index}].fisher_boundary horizon drifted")

        positions: dict[str, list[int]] = {}
        for name in ("boundary_positions", "input_positions", "target_positions"):
            values = decoded_boundary[name]
            if (
                isinstance(values, (str, bytes, bytearray))
                or not isinstance(values, Sequence)
                or not values
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise CalibrationRunError(
                    f"schema-v5 records[{index}].fisher_boundary {name} is invalid"
                )
            positions[name] = list(values)
        normalized_boundary = dict(decoded_boundary)
        normalized_boundary.update(positions)
        if not (
            len(positions["boundary_positions"])
            == len(positions["input_positions"])
            == len(positions["target_positions"])
        ):
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary position lengths drifted"
            )
        if positions["input_positions"] != [
            value + FISHER_BOUNDARY_HORIZON for value in positions["boundary_positions"]
        ] or positions["target_positions"] != [value + 1 for value in positions["input_positions"]]:
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary H=1 positions drifted"
            )
        sequence_length = decoded_record.get("sequence_length")
        if type(sequence_length) is not int or sequence_length < 3:
            raise CalibrationRunError(
                f"schema-v5 records[{index}] cannot support an H=1 Fisher boundary"
            )
        if positions["boundary_positions"] != list(frozen_anchor_positions(sequence_length - 2)):
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary B(T) positions drifted"
            )
        try:
            for name in (
                "input_token_ids_sha256",
                "target_token_ids_sha256",
                "fisher_boundary_sha256",
            ):
                _sha256(
                    decoded_boundary[name],
                    context=f"records[{index}].fisher_boundary.{name}",
                )
        except ValueError as exc:
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary hash is invalid"
            ) from exc
        boundary_payload = {
            name: normalized_boundary[name]
            for name in FISHER_BOUNDARY_FIELDS - {"fisher_boundary_sha256"}
        }
        if normalized_boundary["fisher_boundary_sha256"] != sha256_bytes(
            FISHER_BOUNDARY_NAMESPACE + canonical_json_bytes(boundary_payload)
        ):
            raise CalibrationRunError(
                f"schema-v5 records[{index}].fisher_boundary self-hash drifted"
            )

        normalized_record = dict(decoded_record)
        normalized_record["fisher_boundary"] = normalized_boundary
        normalized_records.append(normalized_record)
    return tuple(normalized_records)


def _identity_view_from_resolver(
    data: bytes,
    resolver: Any,
    *,
    expected_file_sha256: str | None = None,
) -> FrozenCalibrationIdentity:
    if expected_file_sha256 is None:
        decoded = resolver.deserialize_frozen_calibration_identity_artifact(data)
    else:
        decoded = resolver.deserialize_frozen_calibration_identity_artifact(
            data,
            expected_file_sha256=expected_file_sha256,
        )
    root = _strict_json_bytes(data, context="frozen calibration identity")
    evidence = root.get("evidence")
    if not isinstance(evidence, dict):  # independently decoded above; defensive only
        raise ValueError("frozen identity evidence is missing")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != FROZEN_IDENTITY_SCHEMA_VERSION
    ):
        raise CalibrationRunError("frozen identity is not strict schema v5")
    model_contracts = cast(dict[str, object], evidence["model_contracts"])
    primary = cast(dict[str, object], model_contracts["primary"])
    tokenizer = cast(dict[str, object], evidence["tokenizer"])
    execution_bindings = getattr(decoded, "execution_bindings", None)
    if not isinstance(execution_bindings, Mapping):
        raise CalibrationRunError(
            "frozen identity does not contain schema-v5 execution_bindings; "
            "runtime/model access remains unauthorized"
        )
    _exact_fields(
        execution_bindings,
        {
            "calibration_runtime_manifest_file_sha256",
            "model_file_manifest_file_sha256",
            "parquet_materialization_manifest_file_sha256",
            "repository_source_manifest_file_sha256",
        },
        context="frozen identity execution bindings",
    )
    if evidence.get("execution_bindings") != dict(execution_bindings):
        raise CalibrationRunError("decoded execution bindings differ from identity evidence")
    records = _identity_records_with_fisher_boundary(decoded, evidence)
    return FrozenCalibrationIdentity(
        file_sha256=decoded.file_sha256,
        canonical_evidence_sha256=decoded.canonical_evidence_sha256,
        records=records,
        assignment=tuple(dict(item) for item in decoded.assignment),
        assignment_sha256=decoded.assignment_sha256,
        tokenizer_manifest_sha256=decoded.tokenizer_manifest_sha256,
        identity_input_manifest_sha256=_sha256(
            evidence["source_manifest_sha256"],
            context="identity input manifest SHA-256",
        ),
        repository_source_manifest_file_sha256=_sha256(
            execution_bindings["repository_source_manifest_file_sha256"],
            context="identity repository source manifest file SHA-256",
        ),
        runtime_manifest_file_sha256=_sha256(
            execution_bindings["calibration_runtime_manifest_file_sha256"],
            context="identity runtime manifest file SHA-256",
        ),
        model_file_manifest_file_sha256=_sha256(
            execution_bindings["model_file_manifest_file_sha256"],
            context="identity model manifest file SHA-256",
        ),
        parquet_materialization_manifest_file_sha256=_sha256(
            execution_bindings["parquet_materialization_manifest_file_sha256"],
            context="identity parquet materialization manifest file SHA-256",
        ),
        model_id=cast(str, primary["id"]),
        model_revision=_git_revision(primary["revision"], context="identity model revision"),
        transformers_version=cast(str, tokenizer["transformers_version"]),
        artifact_bytes=data,
    )


def _identity_view(data: bytes, repository_root: Path) -> FrozenCalibrationIdentity:
    return _identity_view_from_resolver(data, _load_identity_resolver(repository_root))


def verify_repository_source_manifest(
    expected: Mapping[str, object],
    repository_root: Path,
    *,
    git_executable_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Use the frozen source API to reauthenticate code at point of use."""

    module = _AUTHENTICATED_SOURCE_VERIFIER
    if module is None:
        raise CalibrationRunError("repository source verifier was not bootstrap-authenticated")
    normalized_expected = module.validate_experiment013_source_manifest(expected)
    verified = module.verify_experiment013_source_manifest(
        normalized_expected,
        repo_root=repository_root,
        git_executable=git_executable_path,
    )
    if verified != normalized_expected:
        raise CalibrationRunError("repository source verification returned a different manifest")
    normalized = dict(verified)
    claimed = normalized.pop("canonical_manifest_sha256", None)
    if claimed is None:
        raise CalibrationRunError("repository source manifest is missing its canonical hash")
    digest = module.canonical_experiment013_source_manifest_sha256(normalized)
    if claimed != digest:
        raise CalibrationRunError("repository source manifest self-hash drifted")
    return dict(verified), digest


def validate_adapter_contract(adapter: Any, *, calibration_api: ModuleType) -> None:
    """Validate the reviewed adapter structurally against the authenticated API."""

    if not isinstance(adapter, calibration_api.CalibrationAdapter):
        raise TypeError("reviewed adapter does not implement CalibrationAdapter")


def _record_int(record: Mapping[str, object], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationRunError(f"identity record {name} must be a non-negative integer")
    return value


def _token_ids_sha256(token_ids: Sequence[int], *, allow_empty: bool = False) -> str:
    values: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise CalibrationRunError(f"token_ids[{index}] must be a non-negative integer")
        values.append(token_id)
    if not values and not allow_empty:
        raise CalibrationRunError("materialized token sequence cannot be empty")
    return sha256_bytes(canonical_json_bytes(values))


def validate_materialized_sequence(
    record: Mapping[str, object],
    materialized: Any,
    *,
    calibration_api: ModuleType,
    identity_resolver: Any,
) -> tuple[int, ...]:
    """Reauthenticate all sequence commitments without retaining source text."""

    if not isinstance(materialized, calibration_api.AuthenticatedSequence):
        raise TypeError("adapter must return AuthenticatedSequence")
    token_ids = tuple(materialized.token_ids)
    expected_length = _record_int(record, "sequence_length")
    if len(token_ids) != expected_length:
        raise CalibrationRunError("materialized sequence length differs from frozen identity")
    if _token_ids_sha256(token_ids) != record.get("sequence_token_ids_sha256"):
        raise CalibrationRunError("materialized token IDs differ from frozen identity")
    span = record.get("token_span")
    if not isinstance(span, Mapping):
        raise CalibrationRunError("identity token_span is missing")
    prefill_stop = span.get("prefill_stop")
    scored_stop = span.get("scored_stop")
    if (
        isinstance(prefill_stop, bool)
        or not isinstance(prefill_stop, int)
        or not 0 < prefill_stop <= len(token_ids)
        or scored_stop != len(token_ids)
    ):
        raise CalibrationRunError("identity token span is invalid for materialized sequence")
    if _token_ids_sha256(token_ids[:prefill_stop], allow_empty=True) != record.get(
        "prompt_token_ids_sha256"
    ):
        raise CalibrationRunError("materialized prompt token IDs differ from frozen identity")
    if _token_ids_sha256(token_ids[prefill_stop:], allow_empty=True) != record.get(
        "target_token_ids_sha256"
    ):
        raise CalibrationRunError("materialized target token IDs differ from frozen identity")
    exact = {
        "source_content_sha256": materialized.source_content_sha256,
        "formatted_content_sha256": materialized.formatted_content_sha256,
        "generator_receipt_sha256": materialized.generator_receipt_sha256,
        "tokenizer_manifest_sha256": materialized.tokenizer_manifest_sha256,
    }
    for name, actual in exact.items():
        if actual != record.get(name):
            raise CalibrationRunError(f"materialized {name} differs from frozen identity")
    build_fisher_boundary = getattr(
        identity_resolver,
        "build_fisher_boundary_contract",
        None,
    )
    if not callable(build_fisher_boundary):
        raise CalibrationRunError(
            "authenticated identity resolver lacks the Fisher-boundary contract builder"
        )
    try:
        expected_fisher_boundary = build_fisher_boundary(token_ids)
    except (TypeError, ValueError) as exc:
        raise CalibrationRunError(
            "materialized token IDs cannot satisfy the frozen Fisher-boundary contract"
        ) from exc
    if not isinstance(expected_fisher_boundary, Mapping) or dict(
        expected_fisher_boundary
    ) != record.get("fisher_boundary"):
        raise CalibrationRunError(
            "materialized Fisher input/target tokens differ from frozen identity"
        )
    return token_ids


def frozen_anchor_positions(token_count: int) -> tuple[int, ...]:
    tokens = _positive_int(token_count, context="token_count")
    if tokens < 16:
        return tuple(range(tokens))
    positions = tuple((index + 1) * tokens // 16 - 1 for index in range(16))
    if len(set(positions)) != len(positions):
        raise RuntimeError("frozen anchor equation produced duplicate positions")
    return positions


def compute_anchor_distortions(
    state: Any,
    geometry: Geometry,
) -> tuple[Any, Any, Any]:
    """Return Q4/Q6/Q8 endpoint MSE with deterministic CPU-FP64 reduction."""

    from recurquant.static_q468 import StaticRhtQ468Geometry
    from recurquant.static_q468_calibration import compute_rht_unweighted_mse_endpoints

    endpoint_geometry = StaticRhtQ468Geometry(
        layer_indices=geometry.layer_indices,
        heads=geometry.heads,
        key_rows=geometry.key_rows,
        value_width=geometry.value_width,
        # Endpoint score math does not consume the packing target.  A positive
        # inert value keeps the shared geometry validator authoritative.
        target_resident_bytes=1,
    )
    try:
        return cast(
            tuple[Any, Any, Any],
            compute_rht_unweighted_mse_endpoints(
                state,
                geometry=endpoint_geometry,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationRunError("anchor state violates the frozen endpoint contract") from exc


def compute_fisher_distortions(
    source_state: Any,
    source_gradient: Any,
    geometry: Geometry,
) -> tuple[Any, Any, Any]:
    """Return causal H=1 diagonal-Fisher Q4/Q6/Q8 endpoint risks."""

    from recurquant.static_q468 import StaticRhtQ468Geometry
    from recurquant.static_q468_calibration import (
        compute_rht_diagonal_empirical_fisher_h1_endpoints,
    )

    endpoint_geometry = StaticRhtQ468Geometry(
        layer_indices=geometry.layer_indices,
        heads=geometry.heads,
        key_rows=geometry.key_rows,
        value_width=geometry.value_width,
        target_resident_bytes=1,
    )
    try:
        return cast(
            tuple[Any, Any, Any],
            compute_rht_diagonal_empirical_fisher_h1_endpoints(
                source_state,
                source_gradient,
                geometry=endpoint_geometry,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationRunError(
            "Fisher source state/gradient violates the frozen endpoint contract"
        ) from exc


def capture_sequence_causally(
    adapter: Any,
    model: object,
    record: Mapping[str, object],
    token_ids: tuple[int, ...],
    *,
    geometry: Geometry,
    calibration_api: ModuleType,
    require_cuda: bool,
    distortion_function: DistortionFunction = compute_anchor_distortions,
    fisher_distortion_function: FisherDistortionFunction = compute_fisher_distortions,
) -> CapturedSequence:
    """Run exactly one forward per token and collect both frozen endpoint sets."""

    torch = _torch_runtime()
    anchors = frozen_anchor_positions(len(token_ids))
    anchor_set = set(anchors)
    boundary = record.get("fisher_boundary")
    if not isinstance(boundary, Mapping):
        raise CalibrationRunError("identity record is missing its Fisher-boundary contract")
    try:
        fisher_boundaries = tuple(cast(Sequence[int], boundary["boundary_positions"]))
        fisher_inputs = tuple(cast(Sequence[int], boundary["input_positions"]))
        fisher_targets = tuple(cast(Sequence[int], boundary["target_positions"]))
    except (KeyError, TypeError) as exc:
        raise CalibrationRunError("identity Fisher-boundary positions are malformed") from exc
    expected_boundaries = frozen_anchor_positions(len(token_ids) - 2)
    expected_inputs = tuple(position + 1 for position in expected_boundaries)
    expected_targets = tuple(position + 1 for position in expected_inputs)
    if (
        fisher_boundaries != expected_boundaries
        or fisher_inputs != expected_inputs
        or fisher_targets != expected_targets
    ):
        raise CalibrationRunError("identity Fisher-boundary positions differ from B(T), H=1")
    fisher_by_input = dict(zip(fisher_inputs, fisher_boundaries, strict=True))
    query_ema: Any | None = None
    energies: list[Any] = []
    q4_rows: list[Any] = []
    q6_rows: list[Any] = []
    q8_rows: list[Any] = []
    fisher_q4_rows: list[Any] = []
    fisher_q6_rows: list[Any] = []
    fisher_q8_rows: list[Any] = []
    fisher_target_nlls: list[float] = []
    expected_rows = (geometry.layers, geometry.heads, geometry.key_rows)
    expected_state_shape = (*expected_rows, geometry.value_width)

    def validate_endpoint_triplet(
        values: tuple[Any, Any, Any],
        *,
        context: str,
    ) -> tuple[Any, Any, Any]:
        for name, tensor in zip(("Q4", "Q6", "Q8"), values, strict=True):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != expected_rows
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float64
                or not torch.isfinite(tensor).all().item()
                or (tensor < 0).any().item()
            ):
                raise CalibrationRunError(
                    f"{context} {name} endpoint must be finite non-negative CPU FP64 "
                    f"{expected_rows}"
                )
        return values

    adapter.begin_sequence(model, record)
    completed = False
    try:
        for position, token_id in enumerate(token_ids):
            capture_state = position in anchor_set
            fisher_observation: Any | None = None
            if position in fisher_by_input:
                target_position = position + 1
                fisher_observation = adapter.step_token_with_fisher(
                    model,
                    token_id=token_id,
                    position=position,
                    target_token_id=token_ids[target_position],
                    capture_state=capture_state,
                )
                if not isinstance(
                    fisher_observation,
                    calibration_api.FisherStepObservation,
                ):
                    raise TypeError(
                        "adapter.step_token_with_fisher must return FisherStepObservation"
                    )
                if (
                    fisher_observation.boundary_position != fisher_by_input[position]
                    or fisher_observation.input_position != position
                    or fisher_observation.target_position != target_position
                    or fisher_observation.input_token_id != token_id
                    or fisher_observation.target_token_id != token_ids[target_position]
                ):
                    raise CalibrationRunError(
                        "adapter Fisher observation differs from the frozen H=1 causal pair"
                    )
                observation = fisher_observation.step_observation
            else:
                observation = adapter.step_token(
                    model,
                    token_id=token_id,
                    position=position,
                    capture_state=capture_state,
                )
            if not isinstance(observation, calibration_api.StepObservation):
                raise TypeError("adapter causal step must return StepObservation")
            if (
                observation.position != position
                or observation.token_id != token_id
                or observation.layer_indices != geometry.layer_indices
                or observation.successful_kernel_calls_per_layer != (1,) * geometry.layers
            ):
                raise CalibrationRunError("adapter did not prove one successful causal kernel call")
            query = observation.recurrence_query
            expected_query_shape = (geometry.layers, geometry.heads, geometry.key_rows)
            if (
                not isinstance(query, torch.Tensor)
                or tuple(query.shape) != expected_query_shape
                or not query.is_floating_point()
                or not torch.isfinite(query).all().item()
            ):
                raise CalibrationRunError(
                    f"recurrence query must be finite floating point {expected_query_shape}"
                )
            if require_cuda and query.device.type != "cuda":
                raise CalibrationRunError("official recurrence queries must be actual CUDA tensors")
            query32 = query.detach().to(torch.float32)
            squared = query32.square()
            energy = squared / (squared.sum(dim=-1, keepdim=True) + QUERY_ENERGY_EPSILON)
            if query_ema is None:
                query_ema = torch.full_like(energy, 1.0 / geometry.key_rows)
            query_ema = QUERY_EMA_DECAY * query_ema + (1.0 - QUERY_EMA_DECAY) * energy
            if not torch.isfinite(query_ema).all().item() or (query_ema < 0).any().item():
                raise CalibrationRunError("normalized-query-energy EMA became invalid")

            if fisher_observation is not None:
                source_state = fisher_observation.source_recurrent_state
                source_gradient = fisher_observation.source_state_gradient
                for name, tensor in (
                    ("source recurrent state", source_state),
                    ("source state gradient", source_gradient),
                ):
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tuple(tensor.shape) != expected_state_shape
                        or tensor.dtype != torch.float32
                        or tensor.device != query.device
                        or not torch.isfinite(tensor).all().item()
                    ):
                        raise CalibrationRunError(
                            f"Fisher {name} must be finite FP32 {expected_state_shape} "
                            "on the recurrence-query device"
                        )
                    if require_cuda and tensor.device.type != "cuda":
                        raise CalibrationRunError(
                            f"official Fisher {name} must be an actual CUDA tensor"
                        )
                target_nll = fisher_observation.target_nll
                if type(target_nll) is not float or not math.isfinite(target_nll) or target_nll < 0:
                    raise CalibrationRunError("Fisher target-token NLL must be finite non-negative")
                fisher_values = validate_endpoint_triplet(
                    fisher_distortion_function(source_state, source_gradient, geometry),
                    context="Fisher",
                )
                fisher_q4_rows.append(fisher_values[0])
                fisher_q6_rows.append(fisher_values[1])
                fisher_q8_rows.append(fisher_values[2])
                fisher_target_nlls.append(target_nll)

            if capture_state:
                if observation.recurrent_state is None:
                    raise CalibrationRunError("adapter omitted recurrent state at a frozen anchor")
                state = observation.recurrent_state
                if not isinstance(state, torch.Tensor):
                    raise CalibrationRunError("adapter recurrent state must be a tensor")
                if state.device != query.device:
                    raise CalibrationRunError(
                        "anchor query and recurrent state use different devices"
                    )
                if require_cuda and state.device.type != "cuda":
                    raise CalibrationRunError(
                        "official recurrent states must be actual CUDA tensors"
                    )
                if state.dtype != torch.float32:
                    raise CalibrationRunError("reference recurrent state must be FP32")
                d4, d6, d8 = validate_endpoint_triplet(
                    distortion_function(state, geometry),
                    context="MSE",
                )
                energies.append(query_ema.detach().to(device="cpu", dtype=torch.float64))
                q4_rows.append(d4)
                q6_rows.append(d6)
                q8_rows.append(d8)
            elif observation.recurrent_state is not None:
                raise CalibrationRunError("adapter retained/exposed full state outside an anchor")
        completed = True
    finally:
        adapter.end_sequence(model, record)
    if (
        not completed
        or len(energies) != len(anchors)
        or len(fisher_q4_rows) != len(fisher_boundaries)
        or len(fisher_q6_rows) != len(fisher_boundaries)
        or len(fisher_q8_rows) != len(fisher_boundaries)
        or len(fisher_target_nlls) != len(fisher_boundaries)
    ):
        raise CalibrationRunError(
            "causal sequence capture did not complete every post-token and Fisher endpoint"
        )
    return CapturedSequence(
        anchor_positions=anchors,
        query_energy=torch.stack(energies).contiguous(),
        q4_mse=torch.stack(q4_rows).contiguous(),
        q6_mse=torch.stack(q6_rows).contiguous(),
        q8_mse=torch.stack(q8_rows).contiguous(),
        fisher_boundary_positions=fisher_boundaries,
        fisher_q4_risk=torch.stack(fisher_q4_rows).contiguous(),
        fisher_q6_risk=torch.stack(fisher_q6_rows).contiguous(),
        fisher_q8_risk=torch.stack(fisher_q8_rows).contiguous(),
        fisher_target_nlls=torch.tensor(fisher_target_nlls, dtype=torch.float64),
    )


def _stability_record(value: object) -> dict[str, object]:
    checks = getattr(value, "checks", ())
    shifts = getattr(value, "layer_mean_bitwidth_shifts", ())
    spearman = getattr(value, "spearman_average_ties", None)
    jaccard = getattr(value, "q8_jaccard", None)
    passed = getattr(value, "passed", None)
    if not isinstance(passed, bool):
        raise TypeError("stability result must expose a boolean passed field")
    return {
        "checks": [{"name": str(name), "passed": bool(ok)} for name, ok in checks],
        "layer_mean_bitwidth_shifts": [
            {"layer_index": int(layer), "shift_hex": float(shift).hex()} for layer, shift in shifts
        ],
        "passed": passed,
        "q8_jaccard_hex": None if jaccard is None else float(jaccard).hex(),
        "spearman_average_ties_hex": None if spearman is None else float(spearman).hex(),
    }


class Experiment013Backend:
    """Production wrapper around the frozen resolver, math, codecs, and gates."""

    def __init__(self, repository_root: Path = REPOSITORY_ROOT) -> None:
        self.repository_root = repository_root
        self._geometry: Geometry | None = None

    @property
    def geometry(self) -> Geometry:
        if self._geometry is None:
            from recurquant.static_q468 import FROZEN_QWEN35_STATIC_Q468_GEOMETRY

            frozen = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
            self._geometry = Geometry(
                layer_indices=frozen.layer_indices,
                heads=frozen.heads,
                key_rows=frozen.key_rows,
                value_width=frozen.value_width,
            )
        return self._geometry

    def decode_identity(self, data: bytes) -> FrozenCalibrationIdentity:
        return _identity_view(data, self.repository_root)

    def reduce_sequence(
        self,
        record: Mapping[str, object],
        token_ids: tuple[int, ...],
        captured: CapturedSequence,
    ) -> object:
        from recurquant.static_q468_calibration import (
            FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            FROZEN_UNWEIGHTED_MSE_PROFILE,
            AnchorDistortionBatch,
            FrozenComparatorEndpointBatch,
            reduce_frozen_anchor_distortions,
            reduce_frozen_comparator_endpoints,
        )

        metadata = {
            "family": cast(Any, record["family"]),
            "config": cast(str, record["config"]),
            "ruler_category": cast(Any, record["ruler_category"]),
            "canonical_id": cast(str, record["canonical_id"]),
            "seed": cast(int | None, record["seed"]),
            "configured_length": cast(int | None, record["configured_length"]),
            "token_count": len(token_ids),
        }
        candidate_batch = AnchorDistortionBatch(
            **metadata,
            anchor_positions=captured.anchor_positions,
            query_energy=captured.query_energy,
            q4_mse=captured.q4_mse,
            q6_mse=captured.q6_mse,
            q8_mse=captured.q8_mse,
            sequence_token_ids=token_ids,
            identity_record=record,
        )
        mse_batch = FrozenComparatorEndpointBatch(
            selector_profile=FROZEN_UNWEIGHTED_MSE_PROFILE,
            **metadata,
            endpoint_positions=captured.anchor_positions,
            q4_scores=captured.q4_mse,
            q6_scores=captured.q6_mse,
            q8_scores=captured.q8_mse,
            sequence_token_ids=token_ids,
            identity_record=record,
        )
        fisher_batch = FrozenComparatorEndpointBatch(
            selector_profile=FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            **metadata,
            endpoint_positions=captured.fisher_boundary_positions,
            q4_scores=captured.fisher_q4_risk,
            q6_scores=captured.fisher_q6_risk,
            q8_scores=captured.fisher_q8_risk,
            sequence_token_ids=token_ids,
            identity_record=record,
            target_nlls=captured.fisher_target_nlls,
        )
        return ReducedSequenceScores(
            candidate=reduce_frozen_anchor_distortions(candidate_batch),
            mse=reduce_frozen_comparator_endpoints(mse_batch),
            fisher=reduce_frozen_comparator_endpoints(fisher_batch),
        )

    def finalize(
        self,
        scores: Sequence[object],
        *,
        identity: FrozenCalibrationIdentity,
        source_commit: str,
    ) -> FinalizationResult:
        from recurquant.static_q468 import (
            FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
            FROZEN_STATIC_Q48_PROMOTIONS,
            FROZEN_STATIC_Q468_ABLATION_STEPS,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
            STATIC_Q48_COMPARATOR_METHOD,
            STATIC_Q468_ABLATION_METHOD,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
            STATIC_Q468_MSE_METHOD,
            STATIC_Q468_PRIMARY_METHOD,
            build_static_rht_q48_policy,
            build_static_rht_q468_policy,
            deserialize_static_rht_q48_policy,
            deserialize_static_rht_q468_policy,
            serialize_static_rht_q48_policy,
            serialize_static_rht_q468_policy,
        )
        from recurquant.static_q468_calibration import (
            FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            FROZEN_UNWEIGHTED_MSE_PROFILE,
            aggregate_calibration_scores,
            aggregate_comparator_scores,
            build_frozen_calibration_score_artifact,
            build_frozen_comparator_score_artifact,
            build_frozen_split_half_stability_artifact,
            deserialize_calibration_score_artifact,
            deserialize_comparator_score_artifact,
            deserialize_frozen_split_half_stability_artifact,
            fit_split_half_policy,
        )

        resolver = _load_identity_resolver(self.repository_root)
        if not scores or any(not isinstance(item, ReducedSequenceScores) for item in scores):
            raise TypeError("scores must contain non-empty ReducedSequenceScores")
        typed_scores = cast(list[ReducedSequenceScores], list(scores))
        candidate_scores = [item.candidate for item in typed_scores]
        mse_scores = [item.mse for item in typed_scores]
        fisher_scores = [item.fisher for item in typed_scores]
        aggregate = aggregate_calibration_scores(candidate_scores)
        mse_aggregate = aggregate_comparator_scores(mse_scores)
        fisher_aggregate = aggregate_comparator_scores(fisher_scores)
        geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
        fit = fit_split_half_policy(
            candidate_scores,
            layer_indices=geometry.layer_indices,
            rows_per_layer=geometry.rows_per_layer,
            marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        )
        expected_assignment = [dict(item) for item in identity.assignment]
        actual_assignment = [item.canonical_dict() for item in fit.split.assignments]
        if (
            fit.split.assignment_sha256 != identity.assignment_sha256
            or actual_assignment != expected_assignment
        ):
            raise CalibrationRunError(
                "calibration split differs from the frozen resolver assignment"
            )
        stability = _stability_record(fit.stability)
        if not fit.stability.passed:
            return FinalizationResult(passed=False, stability=stability, artifacts=None)

        score_bytes = build_frozen_calibration_score_artifact(
            aggregate,
            calibration_identity_sha256=identity.file_sha256,
        )
        decoded_score = deserialize_calibration_score_artifact(score_bytes)
        comparator_score_bytes = build_frozen_comparator_score_artifact(
            mse_aggregate,
            fisher_aggregate,
            calibration_identity_sha256=identity.file_sha256,
        )
        decoded_comparator_score = deserialize_comparator_score_artifact(
            comparator_score_bytes,
            expected_calibration_identity_sha256=identity.file_sha256,
        )
        mse_selector = decoded_comparator_score.selectors[FROZEN_UNWEIGHTED_MSE_PROFILE]
        fisher_selector = decoded_comparator_score.selectors[
            FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE
        ]
        split_bytes = build_frozen_split_half_stability_artifact(
            fit.half_a_aggregate,
            fit.half_b_aggregate,
            identity_file_sha256=identity.file_sha256,
            canonical_identity_sha256=identity.canonical_evidence_sha256,
            resolver_assignment_sha256=identity.assignment_sha256,
            full_sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
            full_calibration_scores_sha256=decoded_score.calibration_scores_sha256,
        )
        deserialize_frozen_split_half_stability_artifact(
            split_bytes,
            expected_identity_file_sha256=identity.file_sha256,
            expected_canonical_identity_sha256=identity.canonical_evidence_sha256,
            expected_resolver_assignment_sha256=identity.assignment_sha256,
        )
        policy_common = {
            "geometry": geometry,
            "calibration_manifest_sha256": aggregate.sequence_score_manifest_sha256,
            "identity_artifact_sha256": identity.file_sha256,
            "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
            "source_commit": source_commit,
            "calibration_scores_sha256": decoded_score.calibration_scores_sha256,
        }
        k27030 = build_static_rht_q468_policy(
            aggregate.d4,
            aggregate.d6,
            aggregate.d8,
            marginal_steps=FROZEN_STATIC_Q468_ABLATION_STEPS,
            method_id=STATIC_Q468_ABLATION_METHOD,
            **policy_common,
        )
        k29334 = build_static_rht_q468_policy(
            aggregate.d4,
            aggregate.d6,
            aggregate.d8,
            marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
            method_id=STATIC_Q468_PRIMARY_METHOD,
            **policy_common,
        )
        comparator_policy_common = {
            "geometry": geometry,
            "identity_artifact_sha256": identity.file_sha256,
            "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
            "source_commit": source_commit,
            "marginal_steps": FROZEN_STATIC_Q468_PRIMARY_STEPS,
        }
        mse_k29334 = build_static_rht_q468_policy(
            mse_selector.aggregate.d4,
            mse_selector.aggregate.d6,
            mse_selector.aggregate.d8,
            calibration_manifest_sha256=(mse_selector.aggregate.sequence_score_manifest_sha256),
            method_id=STATIC_Q468_MSE_METHOD,
            **comparator_policy_common,
        )
        fisher_k29334 = build_static_rht_q468_policy(
            fisher_selector.aggregate.d4,
            fisher_selector.aggregate.d6,
            fisher_selector.aggregate.d8,
            calibration_manifest_sha256=(fisher_selector.aggregate.sequence_score_manifest_sha256),
            method_id=STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
            **comparator_policy_common,
        )
        q48 = build_static_rht_q48_policy(
            aggregate.d4,
            aggregate.d8,
            geometry=geometry,
            promoted_rows=FROZEN_STATIC_Q48_PROMOTIONS,
            calibration_manifest_sha256=aggregate.sequence_score_manifest_sha256,
            identity_artifact_sha256=identity.file_sha256,
            tokenizer_manifest_sha256=identity.tokenizer_manifest_sha256,
            source_commit=source_commit,
            method_id=STATIC_Q48_COMPARATOR_METHOD,
        )
        k27030_bytes = serialize_static_rht_q468_policy(k27030)
        k29334_bytes = serialize_static_rht_q468_policy(k29334)
        mse_k29334_bytes = serialize_static_rht_q468_policy(mse_k29334)
        fisher_k29334_bytes = serialize_static_rht_q468_policy(fisher_k29334)
        q48_bytes = serialize_static_rht_q48_policy(q48)
        deserialize_static_rht_q468_policy(k27030_bytes)
        deserialize_static_rht_q468_policy(k29334_bytes)
        deserialize_static_rht_q468_policy(mse_k29334_bytes)
        deserialize_static_rht_q468_policy(fisher_k29334_bytes)
        deserialize_static_rht_q48_policy(q48_bytes)
        binding_bytes = resolver.build_stage_a_calibration_core_binding_artifact(
            frozen_identity_artifact=identity.artifact_bytes,
            calibration_score_artifact=score_bytes,
            split_half_stability_artifact=split_bytes,
            static_k27030_policy_artifact=k27030_bytes,
            static_k29334_policy_artifact=k29334_bytes,
            comparator_score_artifact=comparator_score_bytes,
            static_fisher_k29334_policy_artifact=fisher_k29334_bytes,
            static_mse_k29334_policy_artifact=mse_k29334_bytes,
        )
        resolver.deserialize_stage_a_calibration_core_binding_artifact(binding_bytes)
        return FinalizationResult(
            passed=True,
            stability=stability,
            artifacts=CalibrationArtifacts(
                score=score_bytes,
                comparator_score=comparator_score_bytes,
                split_half=split_bytes,
                static_k27030=k27030_bytes,
                static_k29334=k29334_bytes,
                static_mse_k29334=mse_k29334_bytes,
                static_fisher_k29334=fisher_k29334_bytes,
                static_q48=q48_bytes,
                calibration_core_binding=binding_bytes,
                stability=stability,
            ),
        )


def _model_contract_matches(
    identity: FrozenCalibrationIdentity,
    manifest: ModelFileManifest,
) -> None:
    expected = (
        identity.model_id,
        identity.model_revision,
        identity.transformers_version,
    )
    actual = (manifest.model_id, manifest.revision, manifest.transformers_version)
    if actual != expected:
        raise CalibrationRunError(
            "model file manifest differs from frozen identity model/tokenizer runtime contract"
        )


def _runtime_environment(
    adapter_metadata: Mapping[str, object],
    *,
    elapsed_seconds: float,
    authenticated_runtime: AuthenticatedRuntime,
) -> dict[str, object]:
    torch = _torch_runtime()
    result: dict[str, object] = {
        "adapter": dict(adapter_metadata),
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "authenticated_distribution_count": authenticated_runtime.distribution_count,
        "authenticated_file_count": authenticated_runtime.file_count,
        "packages": dict(authenticated_runtime.distributions),
        "runtime_manifest_file_sha256": authenticated_runtime.manifest_file_sha256,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        result["gpu"] = {
            "capability": list(torch.cuda.get_device_capability(device)),
            "device_index": device,
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    else:
        result["gpu"] = None
    return result


def _report_bytes(
    *,
    status: str,
    identity: FrozenCalibrationIdentity,
    source_commit: str,
    source_manifest_sha256: str,
    source_manifest_file_sha256: str,
    model_files: Any,
    sequence_count: int,
    token_count: int,
    post_token_anchor_count: int,
    fisher_boundary_count: int,
    observed_fisher_step_count: int,
    stability: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    runtime: Mapping[str, object],
    capture_provenance_receipt_file_sha256: str,
    fisher_h1_smoke_report_file_sha256: str | None,
) -> bytes:
    evidence = {
        "artifacts": {name: sha256_bytes(payload) for name, payload in sorted(artifacts.items())},
        "calibration": {
            "expected_fisher_step_count": fisher_boundary_count,
            "observed_fisher_step_count": observed_fisher_step_count,
            "fisher_boundary_count": fisher_boundary_count,
            "post_token_anchor_count": post_token_anchor_count,
            "sequence_count": sequence_count,
            "token_count": token_count,
        },
        "identity": {
            "canonical_evidence_sha256": identity.canonical_evidence_sha256,
            "file_sha256": identity.file_sha256,
            "identity_input_manifest_sha256": identity.identity_input_manifest_sha256,
            "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
            "execution_bindings": {
                "calibration_runtime_manifest_file_sha256": (identity.runtime_manifest_file_sha256),
                "model_file_manifest_file_sha256": (identity.model_file_manifest_file_sha256),
                "parquet_materialization_manifest_file_sha256": (
                    identity.parquet_materialization_manifest_file_sha256
                ),
                "repository_source_manifest_file_sha256": (
                    identity.repository_source_manifest_file_sha256
                ),
            },
        },
        "model_files": {
            "file_count": len(model_files.files),
            "hub_tree_manifest_sha256": model_files.hub_tree_manifest_sha256,
            "manifest_file_sha256": model_files.manifest_file_sha256,
            "model_id": model_files.model_id,
            "revision": model_files.revision,
            "transformers_version": model_files.transformers_version,
        },
        "query_energy_ema": {
            "decay_hex": QUERY_EMA_DECAY.hex(),
            "epsilon_hex": QUERY_ENERGY_EPSILON.hex(),
            "prior": "uniform_1_over_key_rows",
        },
        "prerequisites": {
            "capture_provenance_receipt_file_sha256": (capture_provenance_receipt_file_sha256),
            "fisher_h1_smoke_report_file_sha256": (fisher_h1_smoke_report_file_sha256),
        },
        "repository": {
            "source_commit": source_commit,
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        },
        "runner_revision": RUNNER_REVISION,
        "runtime": dict(runtime),
        "stability": dict(stability),
        "status": status,
    }
    document = {
        "artifact_kind": RUN_REPORT_KIND,
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
        "schema_version": RUN_REPORT_SCHEMA,
    }
    return canonical_json_bytes(document)


def _frozen_token_sequence_manifest_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    commitments: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"frozen calibration record {index} is not a mapping")
        commitments.append(
            {
                "identity_record_sha256": _sha256(
                    record.get("identity_record_sha256"),
                    context=f"frozen calibration record {index} identity SHA-256",
                ),
                "prompt_token_ids_sha256": _sha256(
                    record.get("prompt_token_ids_sha256"),
                    context=f"frozen calibration record {index} prompt-token SHA-256",
                ),
                "sequence_length": _positive_int(
                    record.get("sequence_length"),
                    context=f"frozen calibration record {index} sequence length",
                ),
                "sequence_token_ids_sha256": _sha256(
                    record.get("sequence_token_ids_sha256"),
                    context=f"frozen calibration record {index} sequence-token SHA-256",
                ),
                "target_token_ids_sha256": _sha256(
                    record.get("target_token_ids_sha256"),
                    context=f"frozen calibration record {index} target-token SHA-256",
                ),
            }
        )
    return sha256_bytes(canonical_json_bytes(commitments))


def _authenticate_fisher_h1_smoke_prerequisite_unchecked(
    report_bytes: bytes,
    complete_marker_bytes: bytes,
    *,
    identity: FrozenCalibrationIdentity,
    source_commit: str,
    source_manifest_sha256: str,
    source_manifest_file_sha256: str,
    model_manifest: ModelFileManifest,
    authenticated_runtime: AuthenticatedRuntime,
    expected_capture_provenance_receipt_sha256: str,
) -> str:
    """Authenticate the mandatory one-sequence Fisher H=1 smoke receipt."""

    if not isinstance(report_bytes, bytes) or not isinstance(complete_marker_bytes, bytes):
        raise TypeError("Fisher H=1 smoke prerequisite files must be bytes")
    if complete_marker_bytes != FISHER_SMOKE_COMPLETE_BYTES:
        raise CalibrationRunError("Fisher H=1 smoke completion marker drifted")
    root = _strict_json_bytes(report_bytes, context="Fisher H=1 smoke report")
    _exact_fields(
        root,
        {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"},
        context="Fisher H=1 smoke report",
    )
    if canonical_json_bytes(root) != report_bytes:
        raise CalibrationRunError("Fisher H=1 smoke report is not canonical JSON")
    if (
        root["artifact_kind"] != RUN_REPORT_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != RUN_REPORT_SCHEMA
    ):
        raise CalibrationRunError("Fisher H=1 smoke report kind or schema drifted")
    evidence = root["evidence"]
    if not isinstance(evidence, dict):
        raise CalibrationRunError("Fisher H=1 smoke evidence is missing")
    _exact_fields(
        evidence,
        {
            "artifacts",
            "calibration",
            "identity",
            "model_files",
            "prerequisites",
            "query_energy_ema",
            "repository",
            "runner_revision",
            "runtime",
            "stability",
            "status",
        },
        context="Fisher H=1 smoke evidence",
    )
    canonical_hash = _sha256(
        root["canonical_evidence_sha256"],
        context="Fisher H=1 smoke canonical evidence SHA-256",
    )
    if canonical_hash != sha256_bytes(canonical_json_bytes(evidence)):
        raise CalibrationRunError("Fisher H=1 smoke canonical evidence SHA-256 drifted")
    if (
        evidence["status"] != "fisher_h1_smoke_passed"
        or evidence["runner_revision"] != RUNNER_REVISION
        or evidence["artifacts"] != {}
        or evidence["prerequisites"]
        != {
            "capture_provenance_receipt_file_sha256": (expected_capture_provenance_receipt_sha256),
            "fisher_h1_smoke_report_file_sha256": None,
        }
    ):
        raise CalibrationRunError("Fisher H=1 smoke status or provenance drifted")
    _exact_typed_mapping(
        evidence["stability"],
        {
            "checks": [],
            "evaluated": False,
            "passed": None,
            "scope": "smoke_only",
        },
        context="Fisher H=1 smoke stability",
    )
    _exact_typed_mapping(
        evidence["query_energy_ema"],
        {
            "decay_hex": QUERY_EMA_DECAY.hex(),
            "epsilon_hex": QUERY_ENERGY_EPSILON.hex(),
            "prior": "uniform_1_over_key_rows",
        },
        context="Fisher H=1 smoke query-energy contract",
    )

    first_record = identity.records[0]
    token_count = _positive_int(
        first_record.get("sequence_length"),
        context="first frozen smoke sequence length",
    )
    raw_boundary = first_record.get("fisher_boundary")
    if not isinstance(raw_boundary, Mapping):
        raise CalibrationRunError("first frozen smoke sequence has no Fisher boundary")
    boundary_positions = raw_boundary.get("boundary_positions")
    if not isinstance(boundary_positions, list):
        raise CalibrationRunError("first frozen smoke Fisher boundary positions are missing")
    fisher_count = len(boundary_positions)
    if fisher_count <= 0:
        raise CalibrationRunError("first frozen smoke sequence has no Fisher H=1 steps")
    _exact_typed_mapping(
        evidence["calibration"],
        {
            "expected_fisher_step_count": fisher_count,
            "observed_fisher_step_count": fisher_count,
            "fisher_boundary_count": fisher_count,
            "post_token_anchor_count": len(frozen_anchor_positions(token_count)),
            "sequence_count": 1,
            "token_count": token_count,
        },
        context="Fisher H=1 smoke calibration receipt",
    )
    _exact_typed_mapping(
        evidence["identity"],
        {
            "canonical_evidence_sha256": identity.canonical_evidence_sha256,
            "file_sha256": identity.file_sha256,
            "identity_input_manifest_sha256": identity.identity_input_manifest_sha256,
            "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
            "execution_bindings": {
                "calibration_runtime_manifest_file_sha256": (identity.runtime_manifest_file_sha256),
                "model_file_manifest_file_sha256": identity.model_file_manifest_file_sha256,
                "parquet_materialization_manifest_file_sha256": (
                    identity.parquet_materialization_manifest_file_sha256
                ),
                "repository_source_manifest_file_sha256": (
                    identity.repository_source_manifest_file_sha256
                ),
            },
        },
        context="Fisher H=1 smoke identity receipt",
    )
    _exact_typed_mapping(
        evidence["model_files"],
        {
            "file_count": len(model_manifest.files),
            "hub_tree_manifest_sha256": model_manifest.hub_tree_manifest_sha256,
            "manifest_file_sha256": model_manifest.file_sha256,
            "model_id": model_manifest.model_id,
            "revision": model_manifest.revision,
            "transformers_version": model_manifest.transformers_version,
        },
        context="Fisher H=1 smoke model receipt",
    )
    _exact_typed_mapping(
        evidence["repository"],
        {
            "source_commit": source_commit,
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        },
        context="Fisher H=1 smoke source receipt",
    )
    runtime = evidence["runtime"]
    if not isinstance(runtime, Mapping):
        raise CalibrationRunError("Fisher H=1 smoke runtime receipt is missing")
    _exact_fields(
        runtime,
        {
            "adapter",
            "authenticated_distribution_count",
            "authenticated_file_count",
            "cuda_available",
            "cuda_runtime",
            "elapsed_seconds_hex",
            "gpu",
            "packages",
            "platform",
            "python",
            "runtime_manifest_file_sha256",
            "torch",
        },
        context="Fisher H=1 smoke runtime receipt",
    )
    expected_packages = dict(authenticated_runtime.distributions)
    if (
        runtime["runtime_manifest_file_sha256"] != authenticated_runtime.manifest_file_sha256
        or runtime["authenticated_distribution_count"] != authenticated_runtime.distribution_count
        or type(runtime["authenticated_distribution_count"]) is not int
        or runtime["authenticated_file_count"] != authenticated_runtime.file_count
        or type(runtime["authenticated_file_count"]) is not int
        or runtime["packages"] != expected_packages
        or runtime["python"] != authenticated_runtime.python_version
        or expected_packages.get("torch") != CANONICAL_TORCH_DISTRIBUTION_VERSION
        or runtime["torch"] != CANONICAL_TORCH_RUNTIME_VERSION
        or runtime["cuda_available"] is not True
        or runtime["cuda_runtime"] != CANONICAL_CUDA_RUNTIME_VERSION
        or not isinstance(runtime["platform"], str)
        or not runtime["platform"]
    ):
        raise CalibrationRunError("Fisher H=1 smoke runtime identity drifted")
    _canonical_nonnegative_float_hex(
        runtime["elapsed_seconds_hex"],
        context="Fisher H=1 smoke elapsed seconds",
    )

    gpu = runtime["gpu"]
    if not isinstance(gpu, Mapping):
        raise CalibrationRunError("Fisher H=1 smoke GPU receipt is missing")
    _exact_fields(
        gpu,
        {
            "capability",
            "device_index",
            "name",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        },
        context="Fisher H=1 smoke GPU receipt",
    )
    device_index = _nonnegative_int(
        gpu["device_index"],
        context="Fisher H=1 smoke GPU device index",
    )
    capability = gpu["capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or type(capability[0]) is not int
        or capability[0] <= 0
        or type(capability[1]) is not int
        or capability[1] < 0
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
    ):
        raise CalibrationRunError("Fisher H=1 smoke GPU identity drifted")
    peak_allocated = _nonnegative_int(
        gpu["peak_allocated_bytes"],
        context="Fisher H=1 smoke GPU peak allocated bytes",
    )
    peak_reserved = _nonnegative_int(
        gpu["peak_reserved_bytes"],
        context="Fisher H=1 smoke GPU peak reserved bytes",
    )
    if peak_reserved < peak_allocated:
        raise CalibrationRunError("Fisher H=1 smoke GPU peak counters are inconsistent")

    adapter = runtime.get("adapter")
    if not isinstance(adapter, Mapping):
        raise CalibrationRunError("Fisher H=1 smoke adapter receipt is missing")
    _exact_fields(
        adapter,
        {
            "adapter_revision",
            "capture_input_sha256",
            "device",
            "fisher_step_count",
            "kernel_backend",
            "materialization_attempted",
            "materialized_sequence_count",
            "model_dtype",
            "model_id",
            "model_loaded",
            "model_loading_diagnostic_counts",
            "model_revision",
            "query_shape",
            "recurrent_layer_indices",
            "state_shape",
            "token_sequence_manifest_sha256",
            "transformers_version",
        },
        context="Fisher H=1 smoke adapter receipt",
    )
    diagnostics = adapter["model_loading_diagnostic_counts"]
    if not isinstance(diagnostics, Mapping):
        raise CalibrationRunError("Fisher H=1 smoke model diagnostics are missing")
    _exact_fields(
        diagnostics,
        set(CANONICAL_ADAPTER_LOADING_DIAGNOSTICS),
        context="Fisher H=1 smoke model diagnostics",
    )
    if any(type(diagnostics[name]) is not int or diagnostics[name] != 0 for name in diagnostics):
        raise CalibrationRunError("Fisher H=1 smoke model diagnostics are not empty")
    expected_token_sequence_manifest_sha256 = _frozen_token_sequence_manifest_sha256(
        identity.records
    )
    if (
        adapter["adapter_revision"] != CANONICAL_ADAPTER_REVISION
        or adapter["kernel_backend"] != CANONICAL_ADAPTER_KERNEL_BACKEND
        or adapter["model_dtype"] != CANONICAL_ADAPTER_MODEL_DTYPE
        or adapter["model_id"] != model_manifest.model_id
        or adapter["model_revision"] != model_manifest.revision
        or adapter["transformers_version"] != model_manifest.transformers_version
        or adapter["device"] != f"cuda:{device_index}"
        or adapter["fisher_step_count"] != fisher_count
        or type(adapter["fisher_step_count"]) is not int
        or adapter["materialization_attempted"] is not True
        or adapter["materialized_sequence_count"] != len(identity.records)
        or type(adapter["materialized_sequence_count"]) is not int
        or adapter["model_loaded"] is not True
        or adapter["query_shape"] != list(CANONICAL_ADAPTER_QUERY_SHAPE)
        or adapter["recurrent_layer_indices"] != list(CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES)
        or adapter["state_shape"] != list(CANONICAL_ADAPTER_STATE_SHAPE)
        or adapter["capture_input_sha256"] != identity.identity_input_manifest_sha256
        or adapter["token_sequence_manifest_sha256"] != expected_token_sequence_manifest_sha256
    ):
        raise CalibrationRunError("Fisher H=1 smoke adapter identity drifted")
    return sha256_bytes(report_bytes)


def authenticate_fisher_h1_smoke_prerequisite(
    report_bytes: bytes,
    complete_marker_bytes: bytes,
    *,
    identity: FrozenCalibrationIdentity,
    source_commit: str,
    source_manifest_sha256: str,
    source_manifest_file_sha256: str,
    model_manifest: ModelFileManifest,
    authenticated_runtime: AuthenticatedRuntime,
    expected_capture_provenance_receipt_sha256: str,
) -> str:
    """Fail closed with one public error type for malformed smoke evidence."""

    try:
        return _authenticate_fisher_h1_smoke_prerequisite_unchecked(
            report_bytes,
            complete_marker_bytes,
            identity=identity,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            source_manifest_file_sha256=source_manifest_file_sha256,
            model_manifest=model_manifest,
            authenticated_runtime=authenticated_runtime,
            expected_capture_provenance_receipt_sha256=(expected_capture_provenance_receipt_sha256),
        )
    except CalibrationRunError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationRunError(str(exc)) from exc


def _atomic_publish_new(
    path: Path,
    payload: bytes,
    *,
    capture_path_snapshot: NewCaptureArtifactPath | None = None,
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("artifact payload must be bytes")
    if capture_path_snapshot is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if path != capture_path_snapshot.path:
            raise CalibrationRunError("capture publication path differs from its snapshot")
        _revalidate_new_capture_artifact_path(
            capture_path_snapshot,
            context="capture artifact",
        )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _publish_output_directory(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    complete_filename: str = COMPLETE_FILENAME,
) -> None:
    if complete_filename not in {
        COMPLETE_FILENAME,
        FISHER_SMOKE_COMPLETE_FILENAME,
        AUTHORIZATION_COMPLETE_FILENAME,
    }:
        raise ValueError("completion marker is not a frozen Experiment 013 marker")
    resolved = Path(os.path.abspath(output_dir))
    parent = resolved.parent
    parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing calibration output: {resolved}")
    prefix = f".{resolved.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    owned_staging = True
    # Custody receipts and the completion marker are deliberately last. The
    # public directory appears only after every dependency is durable.
    ordered = [
        name
        for name in sorted(payloads)
        if name not in {REPORT_FILENAME, AUTHORIZATION_FILENAME, BINDING_FILENAME}
    ]
    if REPORT_FILENAME in payloads:
        ordered.append(REPORT_FILENAME)
    if AUTHORIZATION_FILENAME in payloads:
        ordered.append(AUTHORIZATION_FILENAME)
    if BINDING_FILENAME in payloads:
        ordered.append(BINDING_FILENAME)
    try:
        for name in ordered:
            _atomic_publish_new(staging / name, payloads[name])
        _atomic_publish_new(
            staging / complete_filename,
            {
                COMPLETE_FILENAME: CALIBRATION_COMPLETE_BYTES,
                FISHER_SMOKE_COMPLETE_FILENAME: FISHER_SMOKE_COMPLETE_BYTES,
                AUTHORIZATION_COMPLETE_FILENAME: AUTHORIZATION_COMPLETE_BYTES,
            }[complete_filename],
        )
        if resolved.exists():
            raise FileExistsError(f"refusing to overwrite existing calibration output: {resolved}")
        staging.rename(resolved)
        owned_staging = False
        try:
            descriptor = os.open(parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if owned_staging:
            try:
                staging.relative_to(parent)
            except ValueError as exc:  # defensive: never recursively remove outside parent
                raise RuntimeError("owned staging directory escaped its parent") from exc
            if not staging.name.startswith(prefix):
                raise RuntimeError("owned staging directory name drifted")
            shutil.rmtree(staging, ignore_errors=False)


def run_calibration(
    config: CalibrationRunConfig,
    adapter: Any,
    *,
    services: RunnerServices,
) -> dict[str, object]:
    """Execute the authenticated calibration and publish one no-overwrite result set."""

    started = time.perf_counter()
    if not isinstance(config.require_cuda, bool) or not isinstance(config.fisher_h1_smoke, bool):
        raise TypeError("calibration mode flags must be booleans")
    for name, value in (
        ("capture_provenance_receipt_bytes", config.capture_provenance_receipt_bytes),
        ("prior_fisher_h1_smoke_report_bytes", config.prior_fisher_h1_smoke_report_bytes),
        (
            "prior_fisher_h1_smoke_complete_bytes",
            config.prior_fisher_h1_smoke_complete_bytes,
        ),
    ):
        if value is not None and not isinstance(value, bytes):
            raise TypeError(f"{name} must be bytes or None")
    bootstrap_bindings = _bootstrap_identity_bindings(config.frozen_identity_bytes)
    source_commit = _git_revision(
        config.expected_source_commit,
        context="expected source commit",
    )
    capture_provenance_receipt_file_sha256 = (
        _authenticate_calibration_identity_capture_provenance_bytes(
            receipt_bytes=config.capture_provenance_receipt_bytes,
            expected_receipt_sha256=(config.expected_capture_provenance_receipt_sha256),
            runtime_manifest_bytes=config.runtime_manifest_bytes,
            expected_runtime_manifest_sha256=(config.expected_runtime_manifest_sha256),
            source_manifest_bytes=config.repository_source_manifest_bytes,
            expected_identity_input_sha256=(bootstrap_bindings.identity_input_manifest_sha256),
            expected_bindings=bootstrap_bindings,
            expected_source_commit=source_commit,
        )
    )
    # After the metadata-only provenance gate, strictly decode the promoted
    # identity before touching an adapter, model path, or output path.
    identity = services.backend.decode_identity(config.frozen_identity_bytes)
    if (
        identity.repository_source_manifest_file_sha256
        != bootstrap_bindings.repository_source_manifest_file_sha256
        or identity.runtime_manifest_file_sha256 != bootstrap_bindings.runtime_manifest_file_sha256
        or identity.model_file_manifest_file_sha256
        != bootstrap_bindings.model_file_manifest_file_sha256
        or identity.parquet_materialization_manifest_file_sha256
        != bootstrap_bindings.parquet_materialization_manifest_file_sha256
        or identity.identity_input_manifest_sha256
        != bootstrap_bindings.identity_input_manifest_sha256
    ):
        raise CalibrationRunError("decoded identity differs from bootstrap provenance bindings")
    if config.output_dir.resolve().exists():
        raise FileExistsError(
            f"refusing to overwrite existing calibration output: {config.output_dir.resolve()}"
        )
    if config.fisher_h1_smoke:
        if (
            config.prior_fisher_h1_smoke_report_bytes is not None
            or config.prior_fisher_h1_smoke_complete_bytes is not None
        ):
            raise CalibrationRunError("Fisher H=1 smoke mode forbids a prior smoke prerequisite")
    elif (
        config.prior_fisher_h1_smoke_report_bytes is None
        or config.prior_fisher_h1_smoke_complete_bytes is None
    ):
        raise CalibrationRunError(
            "full calibration requires the prior Fisher H=1 smoke report and marker"
        )

    # H0 remains the policy/report provenance even when the authenticated
    # verifier accepts a later HEAD whose complete frozen source inventory is
    # byte-identical to H0.  The verifier below proves ancestry and equality;
    # a raw HEAD == H0 check would incorrectly forbid committing the promoted
    # identity before opening weights.
    source_manifest_file_sha256 = sha256_bytes(config.repository_source_manifest_bytes)
    if source_manifest_file_sha256 != identity.repository_source_manifest_file_sha256:
        raise CalibrationRunError(
            "repository source manifest bytes differ from the frozen identity binding"
        )
    source_manifest_input = _strict_json_bytes(
        config.repository_source_manifest_bytes,
        context="repository source manifest",
    )
    source_manifest, source_manifest_sha256 = services.verify_repository_source(
        source_manifest_input,
        config.repository_root,
    )
    if source_manifest.get("source_commit") != source_commit:
        raise CalibrationRunError(
            "reported source commit must equal the authenticated source-manifest commit"
        )
    services.validate_adapter(adapter)

    expected_runtime_manifest_sha256 = _sha256(
        config.expected_runtime_manifest_sha256,
        context="expected calibration runtime manifest SHA-256",
    )
    runtime_manifest_file_sha256 = sha256_bytes(config.runtime_manifest_bytes)
    if (
        runtime_manifest_file_sha256 != expected_runtime_manifest_sha256
        or runtime_manifest_file_sha256 != identity.runtime_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "calibration runtime manifest bytes differ from the frozen identity/config binding"
        )
    runtime_manifest = parse_calibration_runtime_manifest(config.runtime_manifest_bytes)
    authenticated_runtime = services.authenticate_runtime(runtime_manifest)
    if authenticated_runtime.manifest_file_sha256 != runtime_manifest_file_sha256:
        raise CalibrationRunError("runtime authenticator returned a different manifest identity")
    runtime_versions = dict(authenticated_runtime.distributions)
    if runtime_versions.get("transformers") != identity.transformers_version:
        raise CalibrationRunError(
            "authenticated Transformers version differs from the frozen identity contract"
        )

    expected_model_manifest_sha256 = _sha256(
        config.expected_model_file_manifest_sha256,
        context="expected model file manifest SHA-256",
    )
    model_manifest_file_sha256 = sha256_bytes(config.model_file_manifest_bytes)
    if (
        model_manifest_file_sha256 != expected_model_manifest_sha256
        or model_manifest_file_sha256 != identity.model_file_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "model file manifest bytes differ from the frozen identity/config binding"
        )
    model_manifest = parse_model_file_manifest(config.model_file_manifest_bytes)
    _model_contract_matches(identity, model_manifest)

    expected_parquet_manifest_sha256 = _sha256(
        config.expected_parquet_materialization_manifest_sha256,
        context="expected parquet materialization manifest SHA-256",
    )
    parquet_manifest_file_sha256 = sha256_bytes(config.parquet_materialization_manifest_bytes)
    if (
        parquet_manifest_file_sha256 != expected_parquet_manifest_sha256
        or parquet_manifest_file_sha256 != identity.parquet_materialization_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "parquet materialization manifest bytes differ from the frozen identity/config binding"
        )

    runtime_before_data = services.authenticate_runtime(runtime_manifest)
    if runtime_before_data != authenticated_runtime:
        raise CalibrationRunError("calibration runtime identity changed before data access")

    fisher_h1_smoke_report_file_sha256: str | None = None
    if not config.fisher_h1_smoke:
        assert config.prior_fisher_h1_smoke_report_bytes is not None
        assert config.prior_fisher_h1_smoke_complete_bytes is not None
        fisher_h1_smoke_report_file_sha256 = authenticate_fisher_h1_smoke_prerequisite(
            config.prior_fisher_h1_smoke_report_bytes,
            config.prior_fisher_h1_smoke_complete_bytes,
            identity=identity,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            source_manifest_file_sha256=source_manifest_file_sha256,
            model_manifest=model_manifest,
            authenticated_runtime=authenticated_runtime,
            expected_capture_provenance_receipt_sha256=(capture_provenance_receipt_file_sha256),
        )

    materialized: list[tuple[dict[str, object], tuple[int, ...]]] = []
    identity_resolver = services.identity_resolver
    if identity_resolver is None:
        identity_resolver = _load_identity_resolver(config.repository_root)
    for record in identity.records:
        candidate = adapter.materialize_sequence(record)
        materialized.append(
            (
                record,
                validate_materialized_sequence(
                    record,
                    candidate,
                    calibration_api=services.calibration_api,
                    identity_resolver=identity_resolver,
                ),
            )
        )

    # Reverify source after data/tokenizer adapter use and immediately before
    # opening the local model files.
    _verified_again, second_source_sha256 = services.verify_repository_source(
        source_manifest_input,
        config.repository_root,
    )
    if second_source_sha256 != source_manifest_sha256:
        raise CalibrationRunError("repository source changed during sequence materialization")
    runtime_before_model = services.authenticate_runtime(runtime_manifest)
    if runtime_before_model != authenticated_runtime:
        raise CalibrationRunError("calibration runtime identity changed before model access")

    authenticated_model = services.authenticate_model_files(config.model_root, model_manifest)
    torch = _torch_runtime()
    if config.require_cuda and not torch.cuda.is_available():
        raise CalibrationRunError("official Experiment 013 calibration requires CUDA")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model: object | None = None
    adapter_runtime_metadata: Mapping[str, object] | None = None
    scores: list[object] = []
    total_tokens = 0
    total_anchors = 0
    total_fisher_boundaries = 0
    try:
        # The adapter may call AutoModel only inside this method. The exact local
        # file set has already been hashed, revision checked, and source verified.
        model = adapter.load_model(authenticated_model)
        authenticated_after_load = services.authenticate_model_files(
            config.model_root, model_manifest
        )
        if authenticated_after_load != authenticated_model:
            raise CalibrationRunError("local model identity changed while loading weights")
        selected_materialized = materialized[:1] if config.fisher_h1_smoke else materialized
        for record, token_ids in selected_materialized:
            captured = capture_sequence_causally(
                adapter,
                model,
                record,
                token_ids,
                geometry=services.backend.geometry,
                calibration_api=services.calibration_api,
                require_cuda=config.require_cuda,
                distortion_function=services.distortion_function,
                fisher_distortion_function=services.fisher_distortion_function,
            )
            scores.append(services.backend.reduce_sequence(record, token_ids, captured))
            total_tokens += len(token_ids)
            total_anchors += len(captured.anchor_positions)
            total_fisher_boundaries += len(captured.fisher_boundary_positions)
        result = (
            None
            if config.fisher_h1_smoke
            else services.backend.finalize(
                scores,
                identity=identity,
                source_commit=source_commit,
            )
        )
        # Snapshot run metadata while the authenticated model/device/observer
        # are still live. Cleanup follows immediately and every external
        # identity is then reauthenticated before publication.
        adapter_runtime_metadata = dict(adapter.runtime_metadata())
    finally:
        if model is not None:
            adapter.close_model(model)

    if adapter_runtime_metadata is None:
        raise RuntimeError("successful calibration omitted live adapter runtime metadata")
    observed_fisher_steps = adapter_runtime_metadata.get("fisher_step_count")
    if type(observed_fisher_steps) is not int or observed_fisher_steps != total_fisher_boundaries:
        raise CalibrationRunError(
            "adapter Fisher-step count differs from the frozen boundary inventory"
        )

    authenticated_after_run = services.authenticate_model_files(config.model_root, model_manifest)
    if authenticated_after_run != authenticated_model:
        raise CalibrationRunError("local model identity changed during calibration")

    # Adapter callbacks finish before the final source/runtime/model checks.
    runtime = _runtime_environment(
        adapter_runtime_metadata,
        elapsed_seconds=time.perf_counter() - started,
        authenticated_runtime=authenticated_runtime,
    )
    _verified_final, final_source_sha256 = services.verify_repository_source(
        source_manifest_input,
        config.repository_root,
    )
    if final_source_sha256 != source_manifest_sha256:
        raise CalibrationRunError("repository source changed during calibration")
    final_runtime = services.authenticate_runtime(runtime_manifest)
    if final_runtime != authenticated_runtime:
        raise CalibrationRunError("calibration runtime identity changed during calibration")
    final_model = services.authenticate_model_files(config.model_root, model_manifest)
    if final_model != authenticated_model:
        raise CalibrationRunError("local model identity changed before publication")

    if config.fisher_h1_smoke:
        if len(scores) != 1:
            raise RuntimeError("Fisher H=1 smoke must cover exactly one frozen sequence")
        report = _report_bytes(
            status="fisher_h1_smoke_passed",
            identity=identity,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            source_manifest_file_sha256=source_manifest_file_sha256,
            model_files=authenticated_model,
            sequence_count=len(scores),
            token_count=total_tokens,
            post_token_anchor_count=total_anchors,
            fisher_boundary_count=total_fisher_boundaries,
            observed_fisher_step_count=cast(int, observed_fisher_steps),
            stability={
                "checks": [],
                "evaluated": False,
                "passed": None,
                "scope": "smoke_only",
            },
            artifacts={},
            runtime=runtime,
            capture_provenance_receipt_file_sha256=(capture_provenance_receipt_file_sha256),
            fisher_h1_smoke_report_file_sha256=None,
        )
        smoke_payloads = {FISHER_SMOKE_REPORT_FILENAME: report}
        _publish_output_directory(
            config.output_dir,
            smoke_payloads,
            complete_filename=FISHER_SMOKE_COMPLETE_FILENAME,
        )
        return {
            "artifact_sha256": {FISHER_SMOKE_REPORT_FILENAME: sha256_bytes(report)},
            "fisher_boundary_count": total_fisher_boundaries,
            "output_dir": str(config.output_dir.resolve()),
            "sequence_count": len(scores),
            "status": "fisher_h1_smoke_passed",
            "token_count": total_tokens,
        }

    if result is None:
        raise RuntimeError("full calibration omitted its finalization result")
    if not result.passed:
        report = _report_bytes(
            status="stability_failed",
            identity=identity,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            source_manifest_file_sha256=source_manifest_file_sha256,
            model_files=authenticated_model,
            sequence_count=len(scores),
            token_count=total_tokens,
            post_token_anchor_count=total_anchors,
            fisher_boundary_count=total_fisher_boundaries,
            observed_fisher_step_count=cast(int, observed_fisher_steps),
            stability=result.stability,
            artifacts={},
            runtime=runtime,
            capture_provenance_receipt_file_sha256=(capture_provenance_receipt_file_sha256),
            fisher_h1_smoke_report_file_sha256=(fisher_h1_smoke_report_file_sha256),
        )
        _publish_output_directory(config.output_dir, {REPORT_FILENAME: report})
        report_path = config.output_dir / REPORT_FILENAME
        raise CalibrationStabilityFailure(
            f"split-half stability gate failed; failure report: {report_path}"
        )
    if result.artifacts is None:
        raise RuntimeError("passing finalization omitted calibration artifacts")
    artifacts = result.artifacts
    payloads = {
        SCORE_FILENAME: artifacts.score,
        COMPARATOR_SCORE_FILENAME: artifacts.comparator_score,
        SPLIT_FILENAME: artifacts.split_half,
        K27030_FILENAME: artifacts.static_k27030,
        K29334_FILENAME: artifacts.static_k29334,
        MSE_K29334_FILENAME: artifacts.static_mse_k29334,
        FISHER_K29334_FILENAME: artifacts.static_fisher_k29334,
        Q48_FILENAME: artifacts.static_q48,
        CORE_BINDING_FILENAME: artifacts.calibration_core_binding,
    }
    report = _report_bytes(
        status="passed",
        identity=identity,
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
        source_manifest_file_sha256=source_manifest_file_sha256,
        model_files=authenticated_model,
        sequence_count=len(scores),
        token_count=total_tokens,
        post_token_anchor_count=total_anchors,
        fisher_boundary_count=total_fisher_boundaries,
        observed_fisher_step_count=cast(int, observed_fisher_steps),
        stability=result.stability,
        artifacts=payloads,
        runtime=runtime,
        capture_provenance_receipt_file_sha256=(capture_provenance_receipt_file_sha256),
        fisher_h1_smoke_report_file_sha256=fisher_h1_smoke_report_file_sha256,
    )
    payloads[REPORT_FILENAME] = report
    _publish_output_directory(config.output_dir, payloads)
    return {
        "artifact_sha256": {name: sha256_bytes(payload) for name, payload in payloads.items()},
        "output_dir": str(config.output_dir.resolve()),
        "sequence_count": len(scores),
        "status": "passed",
    }


def default_services(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    base_runtime_root: Path,
    calibration_api: ModuleType | None = None,
    interpreter_path: Path,
    package_roots: Mapping[str, Path],
    git_executable_path: Path | None = None,
) -> RunnerServices:
    api = _AUTHENTICATED_CALIBRATION_API if calibration_api is None else calibration_api
    if api is None:
        raise CalibrationRunError("calibration API was not bootstrap-authenticated")
    backend = Experiment013Backend(repository_root)
    return RunnerServices(
        backend=backend,
        calibration_api=api,
        # The stdlib bootstrap authenticates and installs the resolver before
        # run_calibration.  Keep service construction import-free.
        identity_resolver=None,
        verify_repository_source=lambda manifest, root: verify_repository_source_manifest(
            manifest,
            root,
            git_executable_path=git_executable_path,
        ),
        validate_adapter=lambda adapter: validate_adapter_contract(
            adapter,
            calibration_api=api,
        ),
        distortion_function=compute_anchor_distortions,
        fisher_distortion_function=compute_fisher_distortions,
        authenticate_model_files=lambda root, manifest: authenticate_local_model_files(
            root,
            manifest,
            calibration_api=api,
        ),
        authenticate_runtime=lambda manifest: authenticate_calibration_runtime(
            manifest,
            base_runtime_root=base_runtime_root,
            package_roots=package_roots,
            interpreter_path=interpreter_path,
            git_executable_path=git_executable_path,
        ),
    )


def _install_authenticated_recurquant_namespace(repository_root: Path) -> ModuleType:
    if "recurquant" in sys.modules:
        raise CalibrationRunError("refusing a preloaded recurquant package before adapter import")
    package_root = Path(os.path.abspath(repository_root)) / "src" / "recurquant"
    if not package_root.is_dir() or _is_link_or_reparse(package_root):
        raise CalibrationRunError("authenticated recurquant package root is unavailable")
    package = ModuleType("recurquant")
    package.__package__ = "recurquant"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "recurquant",
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [str(package_root)]
    sys.modules["recurquant"] = package
    return package


_CALIBRATION_IDENTITY_CAPTURE_VALUE_OPTIONS: Final = frozenset(
    {
        "--cache-root",
        "--capture-provenance-receipt-output",
        "--expected-model-file-manifest-sha256",
        "--expected-parquet-materialization-manifest-sha256",
        "--expected-repository-source-manifest-sha256",
        "--expected-runtime-manifest-sha256",
        "--model-file-manifest",
        "--output",
        "--parquet-materialization-manifest",
        "--repository-root",
        "--repository-source-manifest",
        "--ruler-receipt-dir",
        "--runtime-manifest",
        "--source-commit",
    }
)
_STAGE_A_IDENTITY_CAPTURE_VALUE_OPTIONS: Final = frozenset(
    {
        *_CALIBRATION_IDENTITY_CAPTURE_VALUE_OPTIONS,
        "--expected-stage-a-calibration-binding-sha256",
        "--stage-a-calibration-binding",
    }
)
_IDENTITY_CAPTURE_COMMANDS: Final = {
    "capture-calibration-identity": (
        "calibration",
        _CALIBRATION_IDENTITY_CAPTURE_VALUE_OPTIONS,
    ),
    "capture-stage-a-identity": (
        "stage_a",
        _STAGE_A_IDENTITY_CAPTURE_VALUE_OPTIONS,
    ),
}


def _parse_identity_capture_arguments(
    arguments: Sequence[str],
) -> argparse.Namespace:
    values = list(arguments)
    if not values or values[0] not in _IDENTITY_CAPTURE_COMMANDS:
        raise CalibrationRunError("sealed identity capture command is missing")
    command = values[0]
    capture_phase, value_options = _IDENTITY_CAPTURE_COMMANDS[command]
    remainder = values[1:]
    if len(remainder) != 2 * len(value_options):
        raise CalibrationRunError(
            "sealed identity capture arguments are not an exact option profile"
        )
    parsed: dict[str, str] = {}
    for index in range(0, len(remainder), 2):
        option = remainder[index]
        value = remainder[index + 1]
        if option not in value_options or option in parsed or value.startswith("--") or not value:
            raise CalibrationRunError(
                "sealed identity capture arguments are mixed, duplicated, or incomplete"
            )
        parsed[option] = value
    if set(parsed) != set(value_options):
        raise CalibrationRunError("sealed identity capture inputs are incomplete")
    path_options = {
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
    }
    normalized: dict[str, object] = {
        "capture_command": command,
        "capture_phase": capture_phase,
    }
    for option, value in parsed.items():
        name = option[2:].replace("-", "_")
        if option in path_options:
            path = Path(value)
            if not path.is_absolute():
                raise CalibrationRunError(f"sealed identity capture requires an absolute {option}")
            normalized[name] = path
        else:
            normalized[name] = value
    return argparse.Namespace(**normalized)


def _parse_calibration_identity_capture_arguments(
    arguments: Sequence[str],
) -> argparse.Namespace:
    parsed = _parse_identity_capture_arguments(arguments)
    if parsed.capture_phase != "calibration":
        raise CalibrationRunError("sealed calibration identity capture command is missing")
    return parsed


def _parse_stage_a_identity_capture_arguments(
    arguments: Sequence[str],
) -> argparse.Namespace:
    parsed = _parse_identity_capture_arguments(arguments)
    if parsed.capture_phase != "stage_a":
        raise CalibrationRunError("sealed Stage-A identity capture command is missing")
    return parsed


def _new_capture_artifact_path(path: Path, *, context: str) -> NewCaptureArtifactPath:
    raw = Path(path)
    if not raw.is_absolute():
        raise CalibrationRunError(f"{context} must be absolute")
    absolute = Path(os.path.abspath(raw))
    if os.path.lexists(absolute):
        raise FileExistsError(f"refusing to overwrite existing {context}: {absolute}")
    parent, identities = _require_existing_regular_directory(
        absolute.parent,
        context=f"{context} parent",
    )
    return NewCaptureArtifactPath(
        path=parent / absolute.name,
        parent=parent,
        parent_component_identities=identities,
    )


def _revalidate_new_capture_artifact_path(
    snapshot: NewCaptureArtifactPath,
    *,
    context: str,
) -> None:
    parent, identities = _require_existing_regular_directory(
        snapshot.parent,
        context=f"{context} parent",
    )
    if parent != snapshot.parent or identities != snapshot.parent_component_identities:
        raise CalibrationRunError(f"{context} parent changed before publication")
    if os.path.lexists(snapshot.path):
        raise FileExistsError(f"refusing to overwrite existing {context}: {snapshot.path}")


def _runtime_module_relative_origin_matches(
    module_name: str,
    module_relative: PurePosixPath,
) -> bool:
    if not module_relative.parts:
        return False
    if module_name == "six":
        return module_relative == PurePosixPath("six.py")
    return module_relative.parts[0] == module_name


def _runtime_module_origin_record(
    *,
    module_name: str,
    origin_path: Path,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
) -> dict[str, object]:
    distribution_name = CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS[module_name]
    distribution = next(
        (item for item in manifest.distributions if item.name == distribution_name),
        None,
    )
    if distribution is None:
        raise CalibrationRunError(
            f"calibration identity runtime omits critical distribution: {distribution_name}"
        )
    matches: list[tuple[str, str]] = []
    for package_root, root in runtime_context.package_roots.items():
        try:
            relative = origin_path.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        matches.append((package_root, relative))
    if len(matches) != 1:
        raise CalibrationRunError(
            "critical module origin is outside or aliases authenticated package roots: "
            f"{module_name}"
        )
    package_root, relative_path = matches[0]
    if package_root != distribution.package_root:
        raise CalibrationRunError(
            f"critical module origin uses the wrong package root: {module_name}"
        )
    canonical_relative = _canonical_relative_path(
        relative_path,
        context=f"critical module {module_name} origin",
    )
    import_path = runtime_context.package_import_paths[package_root]
    try:
        module_relative = PurePosixPath(canonical_relative).relative_to(PurePosixPath(import_path))
    except ValueError as exc:
        raise CalibrationRunError(
            f"critical module origin is outside its authenticated import root: {module_name}"
        ) from exc
    if not _runtime_module_relative_origin_matches(module_name, module_relative):
        raise CalibrationRunError(f"critical module is shadowed: {module_name}")
    if canonical_relative not in distribution.files:
        raise CalibrationRunError(
            f"critical module origin lacks distribution RECORD ownership: {module_name}"
        )
    tree = next(item for item in manifest.runtime_trees if item.name == package_root)
    expected_file = next(
        (item for item in tree.files if item.path == canonical_relative),
        None,
    )
    if expected_file is None:
        raise CalibrationRunError(
            f"critical module origin is absent from the runtime tree: {module_name}"
        )
    authenticated_path = _assert_no_link_components(
        runtime_context.package_roots[package_root],
        PurePosixPath(canonical_relative),
    )
    if authenticated_path != origin_path.resolve(strict=True):
        raise CalibrationRunError(f"critical module origin path changed: {module_name}")
    digest, size = _stream_file_sha256(authenticated_path)
    if digest != expected_file.sha256 or size != expected_file.size_bytes:
        raise CalibrationRunError(
            f"critical module origin differs from the runtime tree: {module_name}"
        )
    return {
        "distribution": distribution.name,
        "module": module_name,
        "package_root": package_root,
        "relative_path": canonical_relative,
        "sha256": digest,
        "size_bytes": size,
        "version": distribution.version,
    }


def _preflight_calibration_identity_import_surface(
    *,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
) -> dict[str, Path]:
    forbidden_loaded = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0]
        in {
            *CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS,
            *CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES,
        }
    )
    if forbidden_loaded:
        raise CalibrationRunError(
            f"calibration identity critical module was preloaded: {forbidden_loaded}"
        )
    distribution_names = {item.name for item in manifest.distributions}
    if distribution_names.intersection(CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES):
        raise CalibrationRunError(
            "calibration identity runtime unexpectedly stages setuptools/pkg_resources"
        )
    for module_name in CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            raise CalibrationRunError(
                f"excluded calibration identity module remains importable: {module_name}"
            )
    origins: dict[str, Path] = {}
    for module_name in sorted(CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS):
        specification = importlib.util.find_spec(module_name)
        raw_origin = None if specification is None else specification.origin
        if not isinstance(raw_origin, str) or raw_origin in {"built-in", "frozen"}:
            raise CalibrationRunError(
                f"critical calibration identity module is not importable: {module_name}"
            )
        origin = Path(raw_origin)
        _runtime_module_origin_record(
            module_name=module_name,
            origin_path=origin,
            manifest=manifest,
            runtime_context=runtime_context,
        )
        origins[module_name] = origin.resolve(strict=True)
    return origins


def _capture_calibration_identity_module_origins(
    *,
    expected_origins: Mapping[str, Path],
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
) -> list[dict[str, object]]:
    origins: list[dict[str, object]] = []
    for module_name in sorted(CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS):
        module = sys.modules.get(module_name)
        raw_file = None if module is None else getattr(module, "__file__", None)
        specification = None if module is None else getattr(module, "__spec__", None)
        raw_origin = None if specification is None else specification.origin
        if not isinstance(raw_file, str) or not isinstance(raw_origin, str):
            raise CalibrationRunError(
                f"critical calibration identity module was not loaded: {module_name}"
            )
        file_path = Path(raw_file).resolve(strict=True)
        if (
            file_path != Path(raw_origin).resolve(strict=True)
            or file_path != expected_origins[module_name]
        ):
            raise CalibrationRunError(
                f"critical calibration identity module origin changed: {module_name}"
            )
        origins.append(
            _runtime_module_origin_record(
                module_name=module_name,
                origin_path=file_path,
                manifest=manifest,
                runtime_context=runtime_context,
            )
        )
    return origins


def _is_six_meta_path_importer(value: object) -> bool:
    return (
        type(value).__name__ == "_SixMetaPathImporter"
        and type(value).__module__ == "six"
        and getattr(value, "name", None) == "six"
    )


def _assert_authenticated_capture_six(
    binding: AuthenticatedCaptureSix,
    *,
    hash_origin: bool,
) -> None:
    module = binding.module
    if sys.modules.get("six") is not module:
        raise CalibrationRunError("authenticated six module identity changed")
    if getattr(module, "__spec__", None) is not binding.module_spec:
        raise CalibrationRunError("authenticated six module specification changed")
    raw_file = getattr(module, "__file__", None)
    raw_origin = getattr(binding.module_spec, "origin", None)
    if not isinstance(raw_file, str) or not isinstance(raw_origin, str):
        raise CalibrationRunError("authenticated six module origin is missing")
    try:
        declared_file = Path(raw_file).resolve(strict=True)
        declared_origin = Path(raw_origin).resolve(strict=True)
    except OSError as exc:
        raise CalibrationRunError("authenticated six module origin is unavailable") from exc
    if declared_file != binding.origin_path or declared_origin != binding.origin_path:
        raise CalibrationRunError("authenticated six module origin changed")
    if getattr(module, "_importer", None) is not binding.importer:
        raise CalibrationRunError("authenticated six importer identity changed")
    if (
        type(binding.importer) is not binding.importer_type
        or not _is_six_meta_path_importer(binding.importer)
    ):
        raise CalibrationRunError("authenticated six importer contract changed")
    if sum(item is binding.importer for item in sys.meta_path) != 1:
        raise CalibrationRunError("authenticated six importer topology changed")
    if hash_origin:
        try:
            digest, size = _stream_file_sha256(binding.origin_path)
        except OSError as exc:
            raise CalibrationRunError(
                "authenticated six source became unavailable during capture"
            ) from exc
        if digest != binding.origin_sha256 or size != binding.origin_size_bytes:
            raise CalibrationRunError("authenticated six source changed during capture")


def _preload_authenticated_capture_six(
    *,
    expected_origin: Path,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
) -> AuthenticatedCaptureSix:
    """Load exact six bytes and admit only its single documented importer append."""

    if "six" in sys.modules or any(name.startswith("six.") for name in sys.modules):
        raise CalibrationRunError("calibration identity six module was preloaded")
    if not isinstance(sys.meta_path, list):
        raise CalibrationRunError("capture six preload requires mutable import topology")
    if any(_is_six_meta_path_importer(item) for item in sys.meta_path):
        raise CalibrationRunError("capture import topology already contains a six importer")
    original_meta_path = sys.meta_path
    topology_before_import = tuple(original_meta_path)
    forbidden_blocker = _ForbiddenCalibrationIdentityImportBlocker()
    origin_path = Path(expected_origin).resolve(strict=True)
    origin_record = _runtime_module_origin_record(
        module_name="six",
        origin_path=origin_path,
        manifest=manifest,
        runtime_context=runtime_context,
    )
    source_bytes = _read_stable_regular_bytes(
        origin_path,
        context="authenticated capture six source",
    )
    source_sha256 = sha256_bytes(source_bytes)
    source_size = len(source_bytes)
    if (
        source_sha256 != origin_record["sha256"]
        or source_size != origin_record["size_bytes"]
    ):
        raise CalibrationRunError("authenticated capture six source drifted before import")
    try:
        code = compile(source_bytes, str(origin_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise CalibrationRunError("cannot compile authenticated capture six source") from exc
    module = ModuleType("six")
    module.__file__ = str(origin_path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        "six",
        loader=None,
        origin=str(origin_path),
    )
    sys.modules["six"] = module
    try:
        original_meta_path.insert(0, forbidden_blocker)
        if tuple(original_meta_path) != (forbidden_blocker, *topology_before_import):
            raise CalibrationRunError("capture six preload blocker topology changed")
        exec(code, module.__dict__)
        importer = getattr(module, "_importer", None)
        if sys.meta_path is not original_meta_path or tuple(original_meta_path) != (
            forbidden_blocker,
            *topology_before_import,
            importer,
        ):
            raise CalibrationRunError(
                "authenticated six import changed meta-path beyond its exact importer append"
            )
        if forbidden_blocker.attempts:
            raise CalibrationRunError(
                "authenticated six attempted forbidden model/CUDA imports: "
                f"{sorted(forbidden_blocker.attempts)}"
            )
        if not _is_six_meta_path_importer(importer):
            raise CalibrationRunError("authenticated six did not install its exact importer")
        del original_meta_path[0]
        topology_with_importer = tuple(original_meta_path)
        if topology_with_importer != (*topology_before_import, importer):
            raise CalibrationRunError("capture six preload blocker removal changed topology")
        repeated_record = _runtime_module_origin_record(
            module_name="six",
            origin_path=origin_path,
            manifest=manifest,
            runtime_context=runtime_context,
        )
        if repeated_record != origin_record:
            raise CalibrationRunError("authenticated six runtime origin changed on import")
        binding = AuthenticatedCaptureSix(
            module=module,
            module_spec=module.__spec__,
            importer=importer,
            importer_type=type(importer),
            origin_path=origin_path,
            origin_sha256=source_sha256,
            origin_size_bytes=source_size,
            meta_path=original_meta_path,
            topology_before_import=topology_before_import,
            topology_with_importer=topology_with_importer,
        )
        _assert_authenticated_capture_six(binding, hash_origin=True)
        _assert_capture_forbidden_modules_absent()
        return binding
    except BaseException:
        for name in tuple(sys.modules):
            if name == "six" or name.startswith("six."):
                sys.modules.pop(name, None)
        original_meta_path[:] = list(topology_before_import)
        sys.meta_path = original_meta_path
        raise


def _restore_authenticated_capture_six(
    binding: AuthenticatedCaptureSix,
    *,
    primary_error: BaseException | None,
) -> None:
    failures: list[str] = []
    try:
        _assert_authenticated_capture_six(binding, hash_origin=True)
    except CalibrationRunError as error:
        failures.append(str(error))
    if sys.meta_path is not binding.meta_path:
        failures.append("authenticated six meta-path object changed before restoration")
    elif tuple(binding.meta_path) != binding.topology_with_importer:
        failures.append("authenticated six meta-path topology changed before restoration")
    for name in tuple(sys.modules):
        if name == "six" or name.startswith("six."):
            sys.modules.pop(name, None)
    binding.meta_path[:] = list(binding.topology_before_import)
    sys.meta_path = binding.meta_path
    if failures:
        error = CalibrationRunError("; ".join(failures))
        if primary_error is None:
            raise error
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(str(error))


class _ExcludedCalibrationIdentityImportBlocker:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if fullname.split(".", 1)[0] in CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES:
            self.attempts.append(fullname)
            raise CalibrationRunError(f"capture attempted excluded runtime import: {fullname}")
        return None


def _module_matches_prefixes(fullname: str, prefixes: Sequence[str]) -> bool:
    return any(
        fullname == prefix or fullname.startswith(f"{prefix}.") for prefix in prefixes
    )


class _ForbiddenCalibrationIdentityImportBlocker:
    """Reject real imports of every model/CUDA surface during identity capture."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if _module_matches_prefixes(
            fullname,
            CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES,
        ):
            self.attempts.append(fullname)
            raise CalibrationRunError(
                f"capture attempted forbidden model/CUDA import: {fullname}"
            )
        return None


class _GuardedCalibrationIdentityMetaPath(Sequence[object]):
    """A fixed import-finder topology that records and rejects mutation attempts."""

    def __init__(self, entries: Sequence[object]) -> None:
        self._entries = tuple(entries)
        self.mutation_attempts: list[str] = []

    def __getitem__(self, index: int | slice) -> object:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

    def _reject(self, operation: str) -> None:
        self.mutation_attempts.append(operation)
        raise CalibrationRunError(
            f"capture import-finder topology mutation was attempted: {operation}"
        )

    def __delitem__(self, key: object) -> None:
        del key
        self._reject("delete")

    def __iadd__(self, values: object) -> _GuardedCalibrationIdentityMetaPath:
        del values
        self._reject("in-place add")

    def __imul__(self, count: object) -> _GuardedCalibrationIdentityMetaPath:
        del count
        self._reject("in-place multiply")

    def __setitem__(self, key: object, value: object) -> None:
        del key, value
        self._reject("assignment")

    def append(self, value: object) -> None:
        del value
        self._reject("append")

    def clear(self) -> None:
        self._reject("clear")

    def extend(self, values: object) -> None:
        del values
        self._reject("extend")

    def insert(self, index: int, value: object) -> None:
        del index, value
        self._reject("insert")

    def pop(self, index: int = -1) -> object:
        del index
        self._reject("pop")

    def remove(self, value: object) -> None:
        del value
        self._reject("remove")

    def reverse(self) -> None:
        self._reject("reverse")

    def sort(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._reject("sort")


class _CalibrationIdentityImportIsolation:
    """Hide Torch availability probes while hard-stopping actual Torch imports."""

    def __init__(self) -> None:
        self.availability_probes: list[str] = []
        self.blocker = _ForbiddenCalibrationIdentityImportBlocker()
        self._original_find_spec: Callable[..., object] | None = None
        self._scoped_find_spec: Callable[..., object] | None = None
        self._original_import: Callable[..., object] | None = None
        self._scoped_import: Callable[..., object] | None = None
        self._bootstrap_module: ModuleType | None = None
        self._original_find_and_load: Callable[..., object] | None = None
        self._scoped_find_and_load: Callable[..., object] | None = None
        self._original_meta_path: list[object] | None = None
        self._original_meta_path_topology: tuple[object, ...] | None = None
        self._guarded_meta_path: _GuardedCalibrationIdentityMetaPath | None = None
        self._authenticated_six: AuthenticatedCaptureSix | None = None

    def _forbidden_import(self, fullname: object) -> bool:
        return isinstance(fullname, str) and _module_matches_prefixes(
            fullname,
            CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES,
        )

    def _assert_guard_state(self) -> None:
        guarded = self._guarded_meta_path
        scoped_find_spec = self._scoped_find_spec
        scoped_import = self._scoped_import
        bootstrap = self._bootstrap_module
        scoped_find_and_load = self._scoped_find_and_load
        if (
            guarded is None
            or scoped_find_spec is None
            or scoped_import is None
            or bootstrap is None
            or scoped_find_and_load is None
        ):
            raise CalibrationRunError("capture import isolation is not active")
        if sys.meta_path is not guarded:
            raise CalibrationRunError("capture import-finder topology object changed")
        if (
            not guarded
            or guarded[0] is not self.blocker
            or sum(item is self.blocker for item in guarded) != 1
            or tuple(guarded) != (
                self.blocker,
                *(self._original_meta_path_topology or ()),
            )
        ):
            raise CalibrationRunError("capture import-finder topology/order changed")
        if guarded.mutation_attempts:
            raise CalibrationRunError(
                "capture import-finder topology mutation was attempted: "
                f"{sorted(guarded.mutation_attempts)}"
            )
        if self._authenticated_six is not None:
            _assert_authenticated_capture_six(
                self._authenticated_six,
                hash_origin=False,
            )
        if importlib.util.find_spec is not scoped_find_spec:
            raise CalibrationRunError("capture availability isolation changed during execution")
        if builtins.__import__ is not scoped_import:
            raise CalibrationRunError("capture guarded import entrypoint changed during execution")
        if getattr(bootstrap, "_find_and_load", None) is not scoped_find_and_load:
            raise CalibrationRunError("capture import loader entrypoint changed during execution")

    def _guard_import_request(self, fullname: object) -> None:
        self._assert_guard_state()
        if self._forbidden_import(fullname):
            assert isinstance(fullname, str)
            self.blocker.attempts.append(fullname)
            raise CalibrationRunError(
                f"capture attempted forbidden model/CUDA import: {fullname}"
            )

    def activate(
        self,
        *,
        authenticated_six: AuthenticatedCaptureSix | None = None,
    ) -> None:
        if any(
            item is not None
            for item in (
                self._original_find_spec,
                self._scoped_find_spec,
                self._original_import,
                self._scoped_import,
                self._bootstrap_module,
                self._original_find_and_load,
                self._scoped_find_and_load,
                self._original_meta_path,
                self._original_meta_path_topology,
                self._guarded_meta_path,
                self._authenticated_six,
            )
        ):
            raise CalibrationRunError("capture import isolation was activated twice")
        preloaded = sorted(
            name
            for name in sys.modules
            if _module_matches_prefixes(
                name,
                CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES,
            )
        )
        if preloaded:
            raise CalibrationRunError(
                f"capture availability-hidden module was preloaded: {preloaded}"
            )
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__
        bootstrap = getattr(importlib, "_bootstrap", None)
        original_find_and_load = getattr(bootstrap, "_find_and_load", None)
        if not isinstance(sys.meta_path, list) or not callable(original_find_and_load):
            raise CalibrationRunError("capture import machinery is unavailable")
        original_meta_path = sys.meta_path
        original_meta_path_topology = tuple(original_meta_path)
        if authenticated_six is not None:
            _assert_authenticated_capture_six(authenticated_six, hash_origin=True)
            if (
                original_meta_path is not authenticated_six.meta_path
                or original_meta_path_topology
                != authenticated_six.topology_with_importer
            ):
                raise CalibrationRunError(
                    "capture isolation did not receive the exact authenticated six topology"
                )
        guarded_meta_path = _GuardedCalibrationIdentityMetaPath(
            (self.blocker, *original_meta_path_topology)
        )

        def scoped_find_spec(fullname: str, package: str | None = None) -> object:
            self._assert_guard_state()
            if _module_matches_prefixes(
                fullname,
                CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES,
            ):
                self.availability_probes.append(fullname)
                return None
            return original_find_spec(fullname, package)

        def scoped_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            self._guard_import_request(name if level == 0 else None)
            return original_import(name, globals, locals, fromlist, level)

        def scoped_find_and_load(name: str, import_: object) -> object:
            self._guard_import_request(name)
            return original_find_and_load(name, import_)

        self._original_find_spec = original_find_spec
        self._scoped_find_spec = scoped_find_spec
        self._original_import = original_import
        self._scoped_import = scoped_import
        self._bootstrap_module = bootstrap
        self._original_find_and_load = original_find_and_load
        self._scoped_find_and_load = scoped_find_and_load
        self._original_meta_path = original_meta_path
        self._original_meta_path_topology = original_meta_path_topology
        self._guarded_meta_path = guarded_meta_path
        self._authenticated_six = authenticated_six
        try:
            sys.meta_path = guarded_meta_path
            importlib.util.find_spec = scoped_find_spec  # type: ignore[assignment]
            bootstrap._find_and_load = scoped_find_and_load  # type: ignore[attr-defined]
            builtins.__import__ = scoped_import  # type: ignore[assignment]
            self._assert_guard_state()
        except BaseException as error:
            self.restore(primary_error=error)
            raise

    def assert_intact(self) -> None:
        self._assert_guard_state()
        if self._authenticated_six is not None:
            _assert_authenticated_capture_six(
                self._authenticated_six,
                hash_origin=True,
            )
        if self.blocker.attempts:
            raise CalibrationRunError(
                "capture attempted forbidden model/CUDA imports: "
                f"{sorted(self.blocker.attempts)}"
            )

    def restore(self, *, primary_error: BaseException | None) -> None:
        failures: list[str] = []
        original_find_spec = self._original_find_spec
        scoped_find_spec = self._scoped_find_spec
        original_import = self._original_import
        scoped_import = self._scoped_import
        bootstrap = self._bootstrap_module
        original_find_and_load = self._original_find_and_load
        scoped_find_and_load = self._scoped_find_and_load
        original_meta_path = self._original_meta_path
        original_meta_path_topology = self._original_meta_path_topology
        guarded_meta_path = self._guarded_meta_path
        authenticated_six = self._authenticated_six
        state = (
            original_find_spec,
            scoped_find_spec,
            original_import,
            scoped_import,
            bootstrap,
            original_find_and_load,
            scoped_find_and_load,
            original_meta_path,
            original_meta_path_topology,
            guarded_meta_path,
            authenticated_six,
        )
        if all(item is None for item in state):
            return
        if original_find_spec is None or scoped_find_spec is None:
            failures.append("capture availability isolation state was incomplete")
        else:
            if importlib.util.find_spec is not scoped_find_spec:
                failures.append("capture availability isolation changed before restoration")
            importlib.util.find_spec = original_find_spec  # type: ignore[assignment]
        if original_import is None or scoped_import is None:
            failures.append("capture guarded import state was incomplete")
        else:
            if builtins.__import__ is not scoped_import:
                failures.append("capture guarded import changed before restoration")
            builtins.__import__ = original_import  # type: ignore[assignment]
        if (
            bootstrap is None
            or original_find_and_load is None
            or scoped_find_and_load is None
        ):
            failures.append("capture import loader state was incomplete")
        else:
            if getattr(bootstrap, "_find_and_load", None) is not scoped_find_and_load:
                failures.append("capture import loader changed before restoration")
            bootstrap._find_and_load = original_find_and_load  # type: ignore[attr-defined]
        if (
            original_meta_path is None
            or original_meta_path_topology is None
            or guarded_meta_path is None
        ):
            failures.append("capture import-finder topology state was incomplete")
        else:
            if sys.meta_path is not guarded_meta_path:
                failures.append("capture import-finder topology object changed before restoration")
            elif tuple(guarded_meta_path) != (
                self.blocker,
                *original_meta_path_topology,
            ):
                failures.append("capture import-finder topology/order changed before restoration")
            if guarded_meta_path.mutation_attempts:
                failures.append("capture import-finder topology mutation was attempted")
            if tuple(original_meta_path) != original_meta_path_topology:
                failures.append("saved capture import-finder topology changed")
                original_meta_path[:] = list(original_meta_path_topology)
            sys.meta_path = original_meta_path
        if self.blocker.attempts:
            failures.append(
                "capture attempted forbidden model/CUDA imports before restoration: "
                f"{sorted(self.blocker.attempts)}"
            )
        if authenticated_six is not None:
            try:
                _assert_authenticated_capture_six(
                    authenticated_six,
                    hash_origin=True,
                )
            except CalibrationRunError as error:
                failures.append(str(error))
        self._original_find_spec = None
        self._scoped_find_spec = None
        self._original_import = None
        self._scoped_import = None
        self._bootstrap_module = None
        self._original_find_and_load = None
        self._scoped_find_and_load = None
        self._original_meta_path = None
        self._original_meta_path_topology = None
        self._guarded_meta_path = None
        self._authenticated_six = None
        if failures:
            error = CalibrationRunError("; ".join(failures))
            if primary_error is None:
                raise error
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(str(error))


def _snapshot_phase_ruler_file_metadata(
    *,
    capture_module: ModuleType,
    live_source: object,
    phase: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, int],
    tuple[tuple[str, str, int], ...],
]:
    """Hash only the generation manifest and this phase's receipt bodies."""

    if phase not in {"calibration", "stage_a"}:
        raise CalibrationRunError("RULER metadata snapshot phase is invalid")
    required_provider = getattr(capture_module, "required_ruler_receipts", None)
    generation_reader = getattr(live_source, "ruler_generation_manifest_bytes", None)
    receipt_reader = getattr(live_source, "ruler_receipt_bytes", None)
    if not callable(required_provider) or not callable(generation_reader) or not callable(
        receipt_reader
    ):
        raise CalibrationRunError("authenticated RULER metadata surface is incomplete")
    required = required_provider()
    if not isinstance(required, tuple) or len(required) != 20:
        raise CalibrationRunError("authenticated RULER receipt schedule is not exact")
    fields = {
        "category",
        "config",
        "configured_length",
        "filename",
        "phase",
        "sample_index",
        "seed",
    }
    normalized: list[dict[str, object]] = []
    for raw in required:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise CalibrationRunError("authenticated RULER receipt schedule fields drifted")
        item_phase = raw["phase"]
        category = raw["category"]
        config = raw["config"]
        filename = raw["filename"]
        configured_length = raw["configured_length"]
        seed = raw["seed"]
        sample_index = raw["sample_index"]
        if (
            item_phase not in {"calibration", "stage_a"}
            or not isinstance(category, str)
            or not category
            or not isinstance(config, str)
            or not config
            or not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or isinstance(configured_length, bool)
            or not isinstance(configured_length, int)
            or configured_length <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or sample_index != 0
        ):
            raise CalibrationRunError("authenticated RULER receipt schedule value drifted")
        normalized.append(dict(raw))
    names = tuple(
        sorted(
            ["generation-manifest.json", *(str(item["filename"]) for item in normalized)]
        )
    )
    if (
        len(names) != 21
        or len({name.casefold() for name in names}) != 21
        or names != tuple(sorted(RULER_RECEIPT_DIRECTORY_FILENAMES))
    ):
        raise CalibrationRunError("authenticated RULER 21-file inventory drifted")
    phase_items = sorted(
        (item for item in normalized if item["phase"] == phase),
        key=lambda item: str(item["filename"]),
    )
    expected_count = 16 if phase == "calibration" else 4
    if len(phase_items) != expected_count:
        raise CalibrationRunError("authenticated RULER phase receipt count drifted")

    generation_bytes = generation_reader()
    if not isinstance(generation_bytes, bytes) or not generation_bytes:
        raise CalibrationRunError("RULER generation manifest bytes are unavailable")
    generation_sha256 = sha256_bytes(generation_bytes)
    if generation_sha256 != RULER_GENERATION_MANIFEST_FILE_SHA256:
        raise CalibrationRunError("RULER generation manifest file SHA-256 drifted")

    receipt_records: list[tuple[str, str, int]] = []
    for item in phase_items:
        receipt_bytes = receipt_reader(
            category=str(item["category"]),
            config=str(item["config"]),
            configured_length=int(item["configured_length"]),
            seed=int(item["seed"]),
        )
        if not isinstance(receipt_bytes, bytes) or not receipt_bytes:
            raise CalibrationRunError("RULER phase receipt bytes are unavailable")
        receipt_records.append(
            (
                str(item["filename"]),
                sha256_bytes(receipt_bytes),
                len(receipt_bytes),
            )
        )
    return names, (generation_sha256, len(generation_bytes)), tuple(receipt_records)


def _assert_capture_forbidden_modules_absent() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES
        )
    )
    if loaded:
        raise CalibrationRunError(
            f"calibration identity capture crossed a model/CUDA surface: {loaded}"
        )


def _validate_identity_input_payload(
    payload: bytes,
    *,
    phase: str,
    expected_bindings: Mapping[str, str],
    expected_calibration_binding: Mapping[str, str] | None,
) -> dict[str, object]:
    root = _strict_json_bytes(payload, context=f"{phase} identity input")
    if canonical_json_bytes(root) != payload:
        raise CalibrationRunError(f"{phase} identity input is not canonical JSON")
    if (
        root.get("schema") != CALIBRATION_IDENTITY_INPUT_SCHEMA
        or root.get("phase") != phase
        or root.get("model_weights_loaded") is not False
        or root.get("execution_bindings") != dict(expected_bindings)
    ):
        raise CalibrationRunError(f"{phase} identity input custody fields drifted")
    if phase == "calibration":
        if "calibration_binding" in root or expected_calibration_binding is not None:
            raise CalibrationRunError("calibration identity input carries a Stage-A binding")
    elif (
        phase != "stage_a"
        or expected_calibration_binding is None
        or root.get("calibration_binding") != dict(expected_calibration_binding)
    ):
        raise CalibrationRunError("Stage-A identity input calibration binding drifted")
    return root


def _validate_calibration_identity_input_payload(
    payload: bytes,
    *,
    expected_bindings: Mapping[str, str],
) -> dict[str, object]:
    return _validate_identity_input_payload(
        payload,
        phase="calibration",
        expected_bindings=expected_bindings,
        expected_calibration_binding=None,
    )


def _sealed_capture_identity(
    arguments: Sequence[str],
    *,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
    authenticated_runtime: AuthenticatedRuntime,
    interpreter_path: Path,
) -> int:
    args = _parse_identity_capture_arguments(arguments)
    phase = str(args.capture_phase)
    context_prefix = "calibration" if phase == "calibration" else "Stage-A"
    binding_bytes: bytes | None = None
    binding_sha256: str | None = None
    if phase == "stage_a":
        binding_bytes = _read_stable_regular_bytes(
            args.stage_a_calibration_binding,
            context="Stage-A calibration binding",
        )
        binding_sha256 = sha256_bytes(binding_bytes)
        if binding_sha256 != _sha256(
            args.expected_stage_a_calibration_binding_sha256,
            context="expected Stage-A calibration binding SHA-256",
        ):
            raise CalibrationRunError(
                "Stage-A calibration binding differs from its explicit SHA-256"
            )
    ruler_receipt_dir = _verify_ruler_receipt_directory_precondition(args.ruler_receipt_dir)
    output_snapshot = _new_capture_artifact_path(
        args.output,
        context=f"{context_prefix} identity output",
    )
    receipt_output_snapshot = _new_capture_artifact_path(
        args.capture_provenance_receipt_output,
        context=f"{context_prefix} identity capture provenance receipt",
    )
    output = output_snapshot.path
    receipt_output = receipt_output_snapshot.path
    if output == receipt_output:
        raise CalibrationRunError(f"{context_prefix} identity output and receipt paths must differ")

    artifact_paths = {
        "repository_source_manifest_file_sha256": args.repository_source_manifest,
        "calibration_runtime_manifest_file_sha256": args.runtime_manifest,
        "model_file_manifest_file_sha256": args.model_file_manifest,
        "parquet_materialization_manifest_file_sha256": (args.parquet_materialization_manifest),
    }
    expected_digests = {
        "repository_source_manifest_file_sha256": args.expected_repository_source_manifest_sha256,
        "calibration_runtime_manifest_file_sha256": args.expected_runtime_manifest_sha256,
        "model_file_manifest_file_sha256": args.expected_model_file_manifest_sha256,
        "parquet_materialization_manifest_file_sha256": (
            args.expected_parquet_materialization_manifest_sha256
        ),
    }
    artifact_bytes: dict[str, bytes] = {}
    bindings: dict[str, str] = {}
    for name in sorted(artifact_paths):
        data = _read_stable_regular_bytes(
            artifact_paths[name],
            context=f"{context_prefix} identity capture {name}",
        )
        actual = sha256_bytes(data)
        expected = _sha256(expected_digests[name], context=f"expected {name}")
        if actual != expected:
            raise CalibrationRunError(f"{context_prefix} identity capture {name} drifted")
        artifact_bytes[name] = data
        bindings[name] = actual

    def reauthenticate_artifact_bytes() -> None:
        for artifact_name in sorted(artifact_paths):
            repeated_bytes = _read_stable_regular_bytes(
                artifact_paths[artifact_name],
                context=f"repeated {context_prefix} identity capture {artifact_name}",
            )
            if (
                repeated_bytes != artifact_bytes[artifact_name]
                or sha256_bytes(repeated_bytes) != bindings[artifact_name]
            ):
                raise CalibrationRunError(
                    f"{context_prefix} identity capture artifact changed: {artifact_name}"
                )
        if phase == "stage_a":
            assert binding_bytes is not None
            repeated_binding = _read_stable_regular_bytes(
                args.stage_a_calibration_binding,
                context="repeated Stage-A calibration binding",
            )
            if (
                repeated_binding != binding_bytes
                or sha256_bytes(repeated_binding) != binding_sha256
            ):
                raise CalibrationRunError("Stage-A calibration binding changed during capture")

    if bindings["calibration_runtime_manifest_file_sha256"] != manifest.file_sha256:
        raise CalibrationRunError("capture runtime differs from the sealed launcher runtime")
    if authenticated_runtime.manifest_file_sha256 != manifest.file_sha256:
        raise CalibrationRunError("sealed runtime authentication context drifted")
    parse_model_file_manifest(artifact_bytes["model_file_manifest_file_sha256"])

    bootstrap_source = _bootstrap_source_manifest(
        artifact_bytes["repository_source_manifest_file_sha256"],
        repository_root=args.repository_root,
        require_adapter=False,
    )
    requested_commit = _git_revision(args.source_commit, context="capture source commit")
    if requested_commit != bootstrap_source.source_commit:
        raise CalibrationRunError("capture source commit differs from source-manifest H0")
    source_git = bootstrap_source.manifest["git_executable"]
    runtime_git = _authenticate_git_executable(runtime_context.git_executable_path)
    if source_git != {"sha256": runtime_git.sha256, "size_bytes": runtime_git.size_bytes}:
        raise CalibrationRunError("capture source and runtime bind different Git bytes")
    requirements_path = _assert_no_link_components(
        Path(os.path.abspath(args.repository_root)),
        PurePosixPath(CALIBRATION_REQUIREMENTS_PATH),
    )
    requirements_sha256, _requirements_size = _stream_file_sha256(requirements_path)
    if requirements_sha256 != bootstrap_source.entries[CALIBRATION_REQUIREMENTS_PATH]["raw_sha256"]:
        raise CalibrationRunError("calibration requirements changed before capture preflight")
    _preflight_runtime_requirements(manifest, _parse_runtime_requirements(requirements_path))
    expected_origins = _preflight_calibration_identity_import_surface(
        manifest=manifest,
        runtime_context=runtime_context,
    )
    _assert_capture_forbidden_modules_absent()

    exact_module_names = (
        "recurquant",
        CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE,
        CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE,
        IDENTITY_RESOLVER_MODULE,
        CALIBRATION_IDENTITY_CAPTURE_MODULE,
        CALIBRATION_IDENTITY_CAPTURE_RUNNER_MODULE,
    )
    preloaded = sorted(name for name in exact_module_names if name in sys.modules)
    if preloaded:
        raise CalibrationRunError(f"calibration identity source module was preloaded: {preloaded}")
    namespace: ModuleType | None = None
    source_module: ModuleType | None = None
    parquet_module: ModuleType | None = None
    capture_module: ModuleType | None = None
    resolver_module: ModuleType | None = None
    blocker = _ExcludedCalibrationIdentityImportBlocker()
    import_isolation = _CalibrationIdentityImportIsolation()
    authenticated_six: AuthenticatedCaptureSix | None = None
    payload: bytes | None = None
    origins: list[dict[str, object]] | None = None
    normalized_calibration_binding: dict[str, str] | None = None
    calibration_authorization_sha256: str | None = None
    preexisting_policy_modules = frozenset(sys.modules)
    try:
        sys.meta_path.insert(0, blocker)
        authenticated_six = _preload_authenticated_capture_six(
            expected_origin=expected_origins["six"],
            manifest=manifest,
            runtime_context=runtime_context,
        )
        import_isolation.activate(authenticated_six=authenticated_six)
        namespace = _install_authenticated_recurquant_namespace(args.repository_root)
        source_module = _load_exact_source_module(
            CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE,
            SOURCE_VERIFIER_PATH,
            repository_root=args.repository_root,
            entry=bootstrap_source.entries[SOURCE_VERIFIER_PATH],
        )
        namespace.experiment013_source = source_module
        parquet_module = _load_exact_source_module(
            CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE,
            PARQUET_SOURCE_PATH,
            repository_root=args.repository_root,
            entry=bootstrap_source.entries[PARQUET_SOURCE_PATH],
        )
        namespace.experiment013_parquet = parquet_module
        resolver_module = _load_exact_source_module(
            IDENTITY_RESOLVER_MODULE,
            IDENTITY_RESOLVER_SOURCE_PATH,
            repository_root=args.repository_root,
            entry=bootstrap_source.entries[IDENTITY_RESOLVER_SOURCE_PATH],
        )
        if getattr(resolver_module, "CALIBRATION_RUNNER_REVISION", None) != RUNNER_REVISION:
            raise CalibrationRunError("authenticated identity resolver runner revision drifted")
        if phase == "stage_a":
            assert binding_bytes is not None
            assert binding_sha256 is not None
            try:
                verified_binding = resolver_module.deserialize_stage_a_calibration_binding_artifact(
                    binding_bytes,
                    expected_file_sha256=binding_sha256,
                )
            except (TypeError, ValueError) as error:
                raise CalibrationRunError(
                    "Stage-A calibration binding authentication failed"
                ) from error
            normalized_calibration_binding = dict(verified_binding.binding)
            calibration_authorization_sha256 = str(verified_binding.authorization_file_sha256)
            if verified_binding.source_commit != requested_commit:
                raise CalibrationRunError(
                    "Stage-A calibration authorization source commit differs from H0"
                )
            if dict(verified_binding.execution_bindings) != bindings:
                raise CalibrationRunError(
                    "Stage-A calibration authorization execution bindings differ from capture"
                )
        capture_module = _load_exact_source_module(
            CALIBRATION_IDENTITY_CAPTURE_MODULE,
            CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
            repository_root=args.repository_root,
            entry=bootstrap_source.entries[CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH],
        )
        if getattr(capture_module, "CAPTURE_VERSION", None) != (
            CALIBRATION_IDENTITY_CAPTURE_VERSION
        ):
            raise CalibrationRunError(f"{context_prefix} identity capture version drifted")
        verified_source = source_module.verify_experiment013_source_manifest(
            bootstrap_source.manifest,
            repo_root=args.repository_root,
            git_executable=runtime_context.git_executable_path,
        )
        if verified_source != bootstrap_source.manifest:
            raise CalibrationRunError("capture source verifier returned different evidence")
        capture_hub_cache_root = _capture_hub_cache_root_precondition(args.cache_root)
        live_source = capture_module.LiveCaptureSource(
            cache_dir=capture_hub_cache_root,
            ruler_receipt_dir=ruler_receipt_dir,
        )
        ruler_file_snapshot = _snapshot_phase_ruler_file_metadata(
            capture_module=capture_module,
            live_source=live_source,
            phase=phase,
        )
        captured = capture_module.capture_identity_input(
            phase=phase,
            source=live_source,
            calibration_binding=binding_bytes,
            execution_binding_artifacts=artifact_bytes,
            runtime_authentication_context={
                "base_runtime_root": runtime_context.base_runtime_root,
                "git_executable": runtime_context.git_executable_path,
                "staged_interpreter": Path(interpreter_path),
                "package_runtime_roots": dict(runtime_context.package_roots),
                "package_import_paths": dict(runtime_context.package_import_paths),
            },
        )
        repeated_ruler_file_snapshot = _snapshot_phase_ruler_file_metadata(
            capture_module=capture_module,
            live_source=live_source,
            phase=phase,
        )
        if repeated_ruler_file_snapshot != ruler_file_snapshot:
            raise CalibrationRunError("phase-scoped RULER files changed during capture")
        if blocker.attempts:
            raise CalibrationRunError(
                f"capture attempted excluded runtime imports: {sorted(blocker.attempts)}"
            )
        if any(
            name.split(".", 1)[0] in CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES
            for name in sys.modules
        ):
            raise CalibrationRunError("capture left an excluded runtime module loaded")
        _assert_capture_forbidden_modules_absent()
        import_isolation.assert_intact()
        if phase == "stage_a":
            assert binding_bytes is not None
            assert binding_sha256 is not None
            try:
                resolver_module.validate_stage_a_identity_input_for_capture(
                    captured,
                    calibration_binding_artifact=binding_bytes,
                    expected_calibration_binding_file_sha256=binding_sha256,
                )
            except (TypeError, ValueError) as error:
                raise CalibrationRunError(
                    "Stage-A identity input failed authenticated pre-finalization validation"
                ) from error
        payload = canonical_json_bytes(captured)
        _validate_identity_input_payload(
            payload,
            phase=phase,
            expected_bindings=bindings,
            expected_calibration_binding=normalized_calibration_binding,
        )
        origins = _capture_calibration_identity_module_origins(
            expected_origins=expected_origins,
            manifest=manifest,
            runtime_context=runtime_context,
        )
        reauthenticate_artifact_bytes()
        capture_source_sha256, _capture_source_size = _stream_file_sha256(
            Path(args.repository_root) / CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH
        )
        if (
            capture_source_sha256
            != bootstrap_source.entries[CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH]["raw_sha256"]
        ):
            raise CalibrationRunError("capture source changed during execution")
        repeated_source = source_module.verify_experiment013_source_manifest(
            bootstrap_source.manifest,
            repo_root=args.repository_root,
            git_executable=runtime_context.git_executable_path,
        )
        if repeated_source != bootstrap_source.manifest:
            raise CalibrationRunError("repository source changed during identity capture")
        authenticate_calibration_runtime(
            manifest,
            base_runtime_root=runtime_context.base_runtime_root,
            package_roots=runtime_context.package_roots,
            interpreter_path=interpreter_path,
            git_executable_path=runtime_context.git_executable_path,
        )
        _revalidate_new_capture_artifact_path(
            output_snapshot,
            context=f"{context_prefix} identity output",
        )
        _atomic_publish_new(
            output,
            payload,
            capture_path_snapshot=output_snapshot,
        )
        if (
            _read_stable_regular_bytes(output, context=f"published {context_prefix} identity")
            != payload
        ):
            raise CalibrationRunError(f"published {context_prefix} identity bytes changed")
        repeated_source = source_module.verify_experiment013_source_manifest(
            bootstrap_source.manifest,
            repo_root=args.repository_root,
            git_executable=runtime_context.git_executable_path,
        )
        if repeated_source != bootstrap_source.manifest:
            raise CalibrationRunError("repository source changed before receipt publication")
        authenticate_calibration_runtime(
            manifest,
            base_runtime_root=runtime_context.base_runtime_root,
            package_roots=runtime_context.package_roots,
            interpreter_path=interpreter_path,
            git_executable_path=runtime_context.git_executable_path,
        )
        if (
            _read_stable_regular_bytes(output, context=f"published {context_prefix} identity")
            != payload
        ):
            raise CalibrationRunError(
                f"{context_prefix} identity changed before receipt publication"
            )
        reauthenticate_artifact_bytes()
        if blocker.attempts or any(
            name.split(".", 1)[0] in CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES
            for name in sys.modules
        ):
            raise CalibrationRunError(
                "excluded runtime module policy changed before receipt publication"
            )
        import_isolation.assert_intact()
        receipt: dict[str, object] = {
            "artifact_kind": (
                CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND
                if phase == "calibration"
                else STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND
            ),
            "capture_source": {
                "path": CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
                "sha256": capture_source_sha256,
            },
            "capture_version": CALIBRATION_IDENTITY_CAPTURE_VERSION,
            "critical_module_origins": origins,
            "excluded_runtime_modules": list(CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES),
            "execution_bindings": bindings,
            "identity_input_file_sha256": sha256_bytes(payload),
            "phase": phase,
            "publication_contract": (CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT),
            "runner_revision": RUNNER_REVISION,
            "schema_version": (
                CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
                if phase == "calibration"
                else STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
            ),
            "source_commit": requested_commit,
            "status": CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS,
        }
        if phase == "stage_a":
            assert binding_sha256 is not None
            assert calibration_authorization_sha256 is not None
            receipt["calibration_binding_file_sha256"] = binding_sha256
            receipt["calibration_authorization_file_sha256"] = calibration_authorization_sha256
        receipt_payload = canonical_json_bytes(receipt)
        _revalidate_new_capture_artifact_path(
            receipt_output_snapshot,
            context=f"{context_prefix} identity capture provenance receipt",
        )
        print(receipt_payload.decode("utf-8"), end="")
        return 0
    finally:
        try:
            import_isolation.restore(primary_error=sys.exception())
        finally:
            try:
                if authenticated_six is not None:
                    _restore_authenticated_capture_six(
                        authenticated_six,
                        primary_error=sys.exception(),
                    )
            finally:
                if blocker in sys.meta_path:
                    sys.meta_path.remove(blocker)
                for name in tuple(sys.modules):
                    if (
                        name not in preexisting_policy_modules
                        and (
                            name.split(".", 1)[0]
                            in CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES
                            or _module_matches_prefixes(
                                name,
                                CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES,
                            )
                        )
                    ):
                        sys.modules.pop(name, None)
                for name in (
                    CALIBRATION_IDENTITY_CAPTURE_RUNNER_MODULE,
                    CALIBRATION_IDENTITY_CAPTURE_MODULE,
                    IDENTITY_RESOLVER_MODULE,
                    CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE,
                    CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE,
                    "recurquant",
                ):
                    sys.modules.pop(name, None)


def _sealed_capture_calibration_identity(
    arguments: Sequence[str],
    *,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
    authenticated_runtime: AuthenticatedRuntime,
    interpreter_path: Path,
) -> int:
    if not arguments or arguments[0] != "capture-calibration-identity":
        raise CalibrationRunError("sealed calibration identity capture command is missing")
    return _sealed_capture_identity(
        arguments,
        manifest=manifest,
        runtime_context=runtime_context,
        authenticated_runtime=authenticated_runtime,
        interpreter_path=interpreter_path,
    )


def _sealed_capture_stage_a_identity(
    arguments: Sequence[str],
    *,
    manifest: CalibrationRuntimeManifest,
    runtime_context: SealedRuntimeContext,
    authenticated_runtime: AuthenticatedRuntime,
    interpreter_path: Path,
) -> int:
    if not arguments or arguments[0] != "capture-stage-a-identity":
        raise CalibrationRunError("sealed Stage-A identity capture command is missing")
    return _sealed_capture_identity(
        arguments,
        manifest=manifest,
        runtime_context=runtime_context,
        authenticated_runtime=authenticated_runtime,
        interpreter_path=interpreter_path,
    )


def _adapter_construction_context(
    *,
    calibration_api: ModuleType,
    repository_root: Path,
    model_root: Path,
    cache_root: Path,
    ruler_root: Path,
    repository_source_manifest_bytes: bytes,
    calibration_runtime_manifest_bytes: bytes,
    model_file_manifest_bytes: bytes,
    parquet_materialization_manifest_bytes: bytes,
    runtime_context: SealedRuntimeContext,
    interpreter_path: Path,
) -> Any:
    """Build the exact authenticated context consumed by the reviewed adapter."""

    return calibration_api.AdapterConstructionContext(
        repository_root=Path(repository_root),
        model_root=Path(model_root),
        cache_root=Path(cache_root),
        ruler_root=Path(ruler_root),
        execution_binding_artifacts={
            "repository_source_manifest_bytes": bytes(repository_source_manifest_bytes),
            "calibration_runtime_manifest_bytes": bytes(calibration_runtime_manifest_bytes),
            "model_file_manifest_bytes": bytes(model_file_manifest_bytes),
            "parquet_materialization_manifest_bytes": bytes(parquet_materialization_manifest_bytes),
        },
        runtime_authentication_context={
            "base_runtime_root": runtime_context.base_runtime_root,
            "git_executable": runtime_context.git_executable_path,
            "staged_interpreter": Path(interpreter_path),
            "package_runtime_roots": dict(runtime_context.package_roots),
            "package_import_paths": dict(runtime_context.package_import_paths),
        },
    )


def _load_adapter(
    specification: str,
    *,
    repository_root: Path,
    source_entry: Mapping[str, object],
    calibration_api: ModuleType,
    context: Any,
) -> Any:
    if specification != CANONICAL_ADAPTER_SPEC:
        raise CalibrationRunError(f"official calibration requires exactly {CANONICAL_ADAPTER_SPEC}")
    if CANONICAL_ADAPTER_MODULE in sys.modules:
        raise CalibrationRunError("refusing a preloaded reviewed adapter module")
    _install_authenticated_recurquant_namespace(repository_root)
    module = _load_exact_source_module(
        CANONICAL_ADAPTER_MODULE,
        CANONICAL_ADAPTER_PATH,
        repository_root=repository_root,
        entry=source_entry,
    )
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise TypeError("reviewed adapter create_adapter factory is not callable")
    adapter = factory(context)
    validate_adapter_contract(adapter, calibration_api=calibration_api)
    adapter_path = _assert_no_link_components(
        Path(os.path.abspath(repository_root)),
        PurePosixPath(CANONICAL_ADAPTER_PATH),
    )
    digest, _size = _stream_file_sha256(adapter_path)
    if digest != source_entry.get("raw_sha256"):
        raise CalibrationRunError("reviewed adapter source changed during construction")
    return adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--frozen-identity", required=True, type=Path)
    parser.add_argument("--capture-provenance-receipt", required=True, type=Path)
    parser.add_argument(
        "--expected-capture-provenance-receipt-sha256",
        required=True,
    )
    parser.add_argument("--repository-source-manifest", required=True, type=Path)
    parser.add_argument("--model-file-manifest", required=True, type=Path)
    parser.add_argument("--expected-model-file-manifest-sha256", required=True)
    parser.add_argument("--parquet-materialization-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-parquet-materialization-manifest-sha256",
        required=True,
    )
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--ruler-receipt-dir",
        required=True,
        type=Path,
        help=(
            "Absolute directory containing exactly the frozen RULER generation manifest "
            "and 20 receipt JSON files; this is not the NVIDIA/RULER source checkout."
        ),
    )
    parser.add_argument("--repository-root", default=REPOSITORY_ROOT, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fisher-h1-smoke",
        action="store_true",
        help=(
            "Run exactly the first frozen calibration sequence through the full causal H=1 "
            "path and publish only its authenticated smoke receipt."
        ),
    )
    parser.add_argument(
        "--prior-fisher-h1-smoke-report",
        type=Path,
        help=(
            "Canonical smoke report from the required earlier --fisher-h1-smoke run; "
            "mandatory for full calibration and forbidden in smoke mode."
        ),
    )
    parser.add_argument(
        "--prior-fisher-h1-smoke-complete-marker",
        type=Path,
        help=(
            "FISHER_H1_SMOKE_COMPLETE marker paired with the prior smoke report; "
            "mandatory for full calibration and forbidden in smoke mode."
        ),
    )
    parser.add_argument(
        "--adapter",
        choices=[CANONICAL_ADAPTER_SPEC],
        default=CANONICAL_ADAPTER_SPEC,
        help="The single reviewed live adapter (generic adapters are test-injection only).",
    )
    return parser


def _read_exact_regular_directory(
    directory: Path,
    *,
    expected_filenames: set[str],
    context: str,
) -> dict[str, bytes]:
    root = Path(os.path.abspath(directory))
    try:
        before = root.lstat()
    except OSError as exc:
        raise CalibrationRunError(f"{context} is unavailable") from exc
    if _is_link_or_reparse(root) or not stat.S_ISDIR(before.st_mode):
        raise CalibrationRunError(f"{context} must be a regular non-link directory")
    try:
        names = {item.name for item in root.iterdir()}
    except OSError as exc:
        raise CalibrationRunError(f"cannot enumerate {context}") from exc
    if names != expected_filenames:
        raise CalibrationRunError(
            f"{context} inventory drifted; missing={sorted(expected_filenames - names)}, "
            f"extra={sorted(names - expected_filenames)}"
        )
    payloads = {
        name: _read_stable_regular_bytes(root / name, context=f"{context} {name}")
        for name in sorted(expected_filenames)
    }
    try:
        after = root.lstat()
        repeated_names = {item.name for item in root.iterdir()}
    except OSError as exc:
        raise CalibrationRunError(f"cannot reauthenticate {context}") from exc
    if (
        _is_link_or_reparse(root)
        or not stat.S_ISDIR(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or names != repeated_names
    ):
        raise CalibrationRunError(f"{context} changed while it was authenticated")
    return payloads


def _load_authorization_identity_resolver(
    bootstrap_source: BootstrapSource,
) -> ModuleType:
    """Load authorization resolver only from source-manifest-authenticated bytes."""

    module_name = "_recurquant_experiment013_identity_resolver_for_authorization"
    entry = bootstrap_source.entries.get(IDENTITY_RESOLVER_SOURCE_PATH)
    if not isinstance(entry, Mapping):
        raise CalibrationRunError(
            "calibration authorization source manifest omits the identity resolver"
        )
    return _load_exact_source_module(
        module_name,
        IDENTITY_RESOLVER_SOURCE_PATH,
        repository_root=REPOSITORY_ROOT,
        entry=entry,
    )


def authorize_stage_a_calibration(
    *,
    calibration_output_dir: Path,
    fisher_h1_smoke_output_dir: Path,
    capture_provenance_receipt_path: Path,
    expected_capture_provenance_receipt_sha256: str,
    frozen_identity_path: Path,
    expected_frozen_identity_sha256: str,
    repository_source_manifest_path: Path,
    expected_repository_source_manifest_sha256: str,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
    model_file_manifest_path: Path,
    expected_model_file_manifest_sha256: str,
    expected_full_run_report_sha256: str,
    expected_fisher_h1_smoke_report_sha256: str,
    source_commit: str,
    output_dir: Path,
    identity_resolver: Any | None = None,
) -> dict[str, object]:
    """Publish the post-calibration receipt and only Stage-A-eligible binding."""

    expected_source_commit = _git_revision(source_commit, context="authorization H0")
    expected_full_report = _sha256(
        expected_full_run_report_sha256,
        context="expected full calibration report SHA-256",
    )
    expected_smoke_report = _sha256(
        expected_fisher_h1_smoke_report_sha256,
        context="expected Fisher H=1 smoke report SHA-256",
    )
    expected_receipt = _sha256(
        expected_capture_provenance_receipt_sha256,
        context="expected capture provenance receipt SHA-256",
    )
    expected_identity = _sha256(
        expected_frozen_identity_sha256,
        context="expected frozen calibration identity SHA-256",
    )
    expected_repository_source_manifest = _sha256(
        expected_repository_source_manifest_sha256,
        context="expected repository source manifest SHA-256",
    )
    expected_runtime_manifest = _sha256(
        expected_runtime_manifest_sha256,
        context="expected calibration runtime manifest SHA-256",
    )
    expected_model_manifest = _sha256(
        expected_model_file_manifest_sha256,
        context="expected model file manifest SHA-256",
    )
    full_filenames = {
        SCORE_FILENAME,
        COMPARATOR_SCORE_FILENAME,
        SPLIT_FILENAME,
        K27030_FILENAME,
        K29334_FILENAME,
        MSE_K29334_FILENAME,
        FISHER_K29334_FILENAME,
        Q48_FILENAME,
        CORE_BINDING_FILENAME,
        REPORT_FILENAME,
        COMPLETE_FILENAME,
    }
    full = _read_exact_regular_directory(
        calibration_output_dir,
        expected_filenames=full_filenames,
        context="finalized full calibration output",
    )
    smoke = _read_exact_regular_directory(
        fisher_h1_smoke_output_dir,
        expected_filenames={FISHER_SMOKE_REPORT_FILENAME, FISHER_SMOKE_COMPLETE_FILENAME},
        context="finalized Fisher H=1 smoke output",
    )
    receipt_bytes = _read_stable_regular_bytes(
        capture_provenance_receipt_path,
        context="finalized calibration capture provenance receipt",
    )
    frozen_identity_bytes = _read_stable_regular_bytes(
        frozen_identity_path,
        context="frozen calibration identity for authorization",
    )
    repository_source_manifest_bytes = _read_stable_regular_bytes(
        repository_source_manifest_path,
        context="repository source manifest for authorization",
    )
    runtime_manifest_bytes = _read_stable_regular_bytes(
        runtime_manifest_path,
        context="calibration runtime manifest for authorization",
    )
    model_file_manifest_bytes = _read_stable_regular_bytes(
        model_file_manifest_path,
        context="model file manifest for authorization",
    )
    observed = {
        "capture provenance receipt": (sha256_bytes(receipt_bytes), expected_receipt),
        "frozen calibration identity": (sha256_bytes(frozen_identity_bytes), expected_identity),
        "repository source manifest": (
            sha256_bytes(repository_source_manifest_bytes),
            expected_repository_source_manifest,
        ),
        "calibration runtime manifest": (
            sha256_bytes(runtime_manifest_bytes),
            expected_runtime_manifest,
        ),
        "model file manifest": (
            sha256_bytes(model_file_manifest_bytes),
            expected_model_manifest,
        ),
        "full calibration report": (sha256_bytes(full[REPORT_FILENAME]), expected_full_report),
        "Fisher H=1 smoke report": (
            sha256_bytes(smoke[FISHER_SMOKE_REPORT_FILENAME]),
            expected_smoke_report,
        ),
    }
    for name, (actual, expected) in observed.items():
        if actual != expected:
            raise CalibrationRunError(f"{name} differs from its explicit SHA-256")

    owned_resolver = identity_resolver is None
    if owned_resolver:
        bootstrap_source = _bootstrap_source_manifest(
            repository_source_manifest_bytes,
            repository_root=REPOSITORY_ROOT,
            require_adapter=False,
        )
        if bootstrap_source.source_commit != expected_source_commit:
            raise CalibrationRunError("repository source manifest differs from authorization H0")
        resolver = _load_authorization_identity_resolver(bootstrap_source)
    else:
        resolver = identity_resolver
    try:
        core = resolver.deserialize_stage_a_calibration_core_binding_artifact(
            full[CORE_BINDING_FILENAME]
        )
        embedded_identity = core.calibration_dependencies["frozen_identity_artifact"]
        if embedded_identity != frozen_identity_bytes:
            raise CalibrationRunError(
                "calibration core binding embeds a different frozen calibration identity"
            )
        authorization = resolver.build_stage_a_calibration_authorization_artifact(
            calibration_run_report=full[REPORT_FILENAME],
            calibration_complete_marker=full[COMPLETE_FILENAME],
            capture_provenance_receipt=receipt_bytes,
            fisher_h1_smoke_report=smoke[FISHER_SMOKE_REPORT_FILENAME],
            fisher_h1_smoke_complete_marker=smoke[FISHER_SMOKE_COMPLETE_FILENAME],
            calibration_core_binding_artifact=full[CORE_BINDING_FILENAME],
            calibration_runtime_manifest=runtime_manifest_bytes,
            model_file_manifest=model_file_manifest_bytes,
            repository_source_manifest=repository_source_manifest_bytes,
            static_q48_policy_artifact=full[Q48_FILENAME],
        )
        verified_authorization = resolver.deserialize_stage_a_calibration_authorization_artifact(
            authorization
        )
        if verified_authorization.source_commit != expected_source_commit:
            raise CalibrationRunError("calibration authorization differs from explicit H0")
        binding = resolver.build_stage_a_calibration_binding_artifact(
            calibration_authorization_artifact=authorization
        )
        verified_binding = resolver.deserialize_stage_a_calibration_binding_artifact(binding)
        if verified_binding.authorization_file_sha256 != sha256_bytes(authorization):
            raise CalibrationRunError("Stage-A binding lost its authorization dependency")
    except CalibrationRunError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationRunError(str(exc)) from exc
    finally:
        if owned_resolver:
            sys.modules.pop(resolver.__name__, None)

    _publish_output_directory(
        output_dir,
        {
            AUTHORIZATION_FILENAME: authorization,
            BINDING_FILENAME: binding,
        },
        complete_filename=AUTHORIZATION_COMPLETE_FILENAME,
    )
    return {
        "artifact_sha256": {
            AUTHORIZATION_FILENAME: sha256_bytes(authorization),
            BINDING_FILENAME: sha256_bytes(binding),
        },
        "authorized_calibration_output_dir": str(Path(calibration_output_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "source_commit": expected_source_commit,
        "status": "authorized_for_stage_a",
    }


def _capture_manifest_mode(arguments: Sequence[str]) -> int | None:
    if not arguments or arguments[0] not in {
        "capture-source-manifest",
        "capture-runtime-manifest",
        "authorize-stage-a-calibration",
        "capture-model-manifest",
        "prepare-runtime",
        "stage-model",
        "verify-frozen-identity-contract",
        "verify-model-staging-authorization",
        "verify-model-staging-paths",
    }:
        return None
    command = arguments[0]
    parser = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} {command}",
        allow_abbrev=False,
    )
    if command == "authorize-stage-a-calibration":
        parser.add_argument("--calibration-output-dir", required=True, type=Path)
        parser.add_argument("--fisher-h1-smoke-output-dir", required=True, type=Path)
        parser.add_argument("--capture-provenance-receipt", required=True, type=Path)
        parser.add_argument(
            "--expected-capture-provenance-receipt-sha256",
            required=True,
        )
        parser.add_argument("--frozen-identity", required=True, type=Path)
        parser.add_argument("--expected-frozen-identity-sha256", required=True)
        parser.add_argument("--repository-source-manifest", required=True, type=Path)
        parser.add_argument("--expected-repository-source-manifest-sha256", required=True)
        parser.add_argument("--runtime-manifest", required=True, type=Path)
        parser.add_argument("--expected-runtime-manifest-sha256", required=True)
        parser.add_argument("--model-file-manifest", required=True, type=Path)
        parser.add_argument("--expected-model-file-manifest-sha256", required=True)
        parser.add_argument("--expected-full-run-report-sha256", required=True)
        parser.add_argument("--expected-fisher-h1-smoke-report-sha256", required=True)
        parser.add_argument("--source-commit", required=True)
        parser.add_argument("--output-dir", required=True, type=Path)
    elif command == "prepare-runtime":
        parser.add_argument("--git-executable", required=True, type=Path)
        parser.add_argument("--source-python", required=True, type=Path)
        parser.add_argument("--requirements", required=True, type=Path)
        parser.add_argument("--output-root", required=True, type=Path)
        parser.add_argument(
            "--package-root-name",
            default=DEFAULT_PACKAGE_RUNTIME_ROOT_NAME,
        )
    elif command == "verify-frozen-identity-contract":
        parser.add_argument("--git-executable", required=True, type=Path)
        parser.add_argument("--frozen-identity", required=True, type=Path)
        parser.add_argument("--expected-frozen-identity-sha256", required=True)
        parser.add_argument("--repository-root", required=True, type=Path)
        parser.add_argument("--repository-source-manifest", required=True, type=Path)
        parser.add_argument("--source-commit", required=True)
        parser.add_argument("--capture-provenance-receipt", required=True, type=Path)
        parser.add_argument(
            "--expected-capture-provenance-receipt-sha256",
            required=True,
        )
        parser.add_argument("--runtime-manifest", required=True, type=Path)
        parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    elif command == "verify-model-staging-paths":
        parser.add_argument("--repository-root", required=True, type=Path)
        parser.add_argument("--hub-cache-root", required=True, type=Path)
        parser.add_argument("--output-root", required=True, type=Path)
    elif command in {"stage-model", "verify-model-staging-authorization"}:
        parser.add_argument("--git-executable", required=True, type=Path)
        parser.add_argument("--frozen-identity", required=True, type=Path)
        parser.add_argument("--expected-frozen-identity-sha256", required=True)
        parser.add_argument("--identity-commit", required=True)
        parser.add_argument("--repository-root", required=True, type=Path)
        parser.add_argument("--repository-source-manifest", required=True, type=Path)
        parser.add_argument("--source-commit", required=True)
        parser.add_argument("--model-file-manifest", required=True, type=Path)
        parser.add_argument("--expected-model-file-manifest-sha256", required=True)
        parser.add_argument("--capture-provenance-receipt", required=True, type=Path)
        parser.add_argument(
            "--expected-capture-provenance-receipt-sha256",
            required=True,
        )
        parser.add_argument("--runtime-manifest", required=True, type=Path)
        parser.add_argument("--expected-runtime-manifest-sha256", required=True)
        if command == "stage-model":
            parser.add_argument("--hub-cache-root", required=True, type=Path)
            parser.add_argument("--output-root", required=True, type=Path)
            parser.add_argument(
                "--expected-model-staging-path-contract-sha256",
                required=True,
            )
            parser.add_argument("--local-files-only", action="store_true")
    else:
        parser.add_argument("--output", required=True, type=Path)
    if command == "capture-source-manifest":
        parser.add_argument("--git-executable", required=True, type=Path)
        parser.add_argument("--repository-root", required=True, type=Path)
    if command == "capture-runtime-manifest":
        parser.add_argument("--git-executable", required=True, type=Path)
        parser.add_argument("--base-runtime-root", required=True, type=Path)
        parser.add_argument("--staged-interpreter", required=True, type=Path)
        parser.add_argument("--package-root", required=True, action="append")
        parser.add_argument("--package-import-path", required=True, action="append")
    if command == "capture-model-manifest":
        parser.add_argument("--model-id", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--transformers-version", required=True)
    args = parser.parse_args(arguments[1:])
    if command == "authorize-stage-a-calibration":
        details = authorize_stage_a_calibration(
            calibration_output_dir=args.calibration_output_dir,
            fisher_h1_smoke_output_dir=args.fisher_h1_smoke_output_dir,
            capture_provenance_receipt_path=args.capture_provenance_receipt,
            expected_capture_provenance_receipt_sha256=(
                args.expected_capture_provenance_receipt_sha256
            ),
            frozen_identity_path=args.frozen_identity,
            expected_frozen_identity_sha256=args.expected_frozen_identity_sha256,
            repository_source_manifest_path=args.repository_source_manifest,
            expected_repository_source_manifest_sha256=(
                args.expected_repository_source_manifest_sha256
            ),
            runtime_manifest_path=args.runtime_manifest,
            expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
            model_file_manifest_path=args.model_file_manifest,
            expected_model_file_manifest_sha256=args.expected_model_file_manifest_sha256,
            expected_full_run_report_sha256=args.expected_full_run_report_sha256,
            expected_fisher_h1_smoke_report_sha256=(args.expected_fisher_h1_smoke_report_sha256),
            source_commit=args.source_commit,
            output_dir=args.output_dir,
        )
        print(canonical_json_bytes(details).decode("utf-8"), end="")
        return 0
    if command == "prepare-runtime":
        details = prepare_calibration_runtime(
            git_executable_path=args.git_executable,
            source_python=args.source_python,
            requirements_file=args.requirements,
            output_root=args.output_root,
            package_root_name=args.package_root_name,
        )
        print(json.dumps(details, sort_keys=True))
        return 0
    if command == "stage-model":
        details = stage_identity_bound_model(
            git_executable_path=args.git_executable,
            frozen_identity_path=args.frozen_identity,
            expected_frozen_identity_sha256=args.expected_frozen_identity_sha256,
            identity_commit=args.identity_commit,
            repository_root=args.repository_root,
            repository_source_manifest_path=args.repository_source_manifest,
            source_commit=args.source_commit,
            model_file_manifest_path=args.model_file_manifest,
            expected_model_file_manifest_sha256=args.expected_model_file_manifest_sha256,
            capture_provenance_receipt_path=args.capture_provenance_receipt,
            expected_capture_provenance_receipt_sha256=(
                args.expected_capture_provenance_receipt_sha256
            ),
            runtime_manifest_path=args.runtime_manifest,
            expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
            expected_model_staging_path_contract_sha256=(
                args.expected_model_staging_path_contract_sha256
            ),
            hub_cache_root=args.hub_cache_root,
            output_root=args.output_root,
            local_files_only=args.local_files_only,
        )
        print(json.dumps(details, sort_keys=True))
        return 0
    if command == "verify-model-staging-paths":
        details = verify_model_staging_paths(
            repository_root=args.repository_root,
            hub_cache_root=args.hub_cache_root,
            output_root=args.output_root,
        )
        print(canonical_json_bytes(details).decode("utf-8"), end="")
        return 0
    if command == "verify-frozen-identity-contract":
        details = verify_frozen_identity_contract(
            git_executable_path=args.git_executable,
            frozen_identity_path=args.frozen_identity,
            expected_frozen_identity_sha256=args.expected_frozen_identity_sha256,
            repository_root=args.repository_root,
            repository_source_manifest_path=args.repository_source_manifest,
            source_commit=args.source_commit,
            capture_provenance_receipt_path=args.capture_provenance_receipt,
            expected_capture_provenance_receipt_sha256=(
                args.expected_capture_provenance_receipt_sha256
            ),
            runtime_manifest_path=args.runtime_manifest,
            expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        )
        print(canonical_json_bytes(details).decode("utf-8"), end="")
        return 0
    if command == "verify-model-staging-authorization":
        details = verify_identity_bound_model_staging_authorization(
            git_executable_path=args.git_executable,
            frozen_identity_path=args.frozen_identity,
            expected_frozen_identity_sha256=args.expected_frozen_identity_sha256,
            identity_commit=args.identity_commit,
            repository_root=args.repository_root,
            repository_source_manifest_path=args.repository_source_manifest,
            source_commit=args.source_commit,
            model_file_manifest_path=args.model_file_manifest,
            expected_model_file_manifest_sha256=args.expected_model_file_manifest_sha256,
            capture_provenance_receipt_path=args.capture_provenance_receipt,
            expected_capture_provenance_receipt_sha256=(
                args.expected_capture_provenance_receipt_sha256
            ),
            runtime_manifest_path=args.runtime_manifest,
            expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        )
        print(canonical_json_bytes(details).decode("utf-8"), end="")
        return 0
    status = "captured_metadata_only"
    details: dict[str, object] = {}
    if command == "capture-source-manifest":
        source_module = _load_source_capture_module(args.repository_root)
        try:
            captured = source_module.capture_experiment013_source_manifest(
                args.repository_root,
                git_executable=args.git_executable,
            )
            normalized = source_module.validate_experiment013_source_manifest(captured)
            payload = source_module.canonical_experiment013_source_manifest_bytes(normalized)
            git_executable = _authenticate_git_executable(args.git_executable)
            output = _assert_source_manifest_output_location(
                args.repository_root,
                args.output,
                git_executable=git_executable,
            )
            before_publish = source_module.verify_experiment013_source_manifest(
                normalized,
                args.repository_root,
                git_executable=git_executable.path,
            )
            if before_publish != normalized:
                raise CalibrationRunError("source verifier changed the captured manifest")
            _atomic_publish_new(output, payload)
            if output.read_bytes() != payload:
                raise CalibrationRunError("published source manifest bytes changed on disk")
            after_publish = source_module.verify_experiment013_source_manifest(
                normalized,
                args.repository_root,
                git_executable=git_executable.path,
            )
            if after_publish != normalized:
                raise CalibrationRunError("source identity changed after manifest publication")
            details = {
                "canonical_manifest_sha256": normalized["canonical_manifest_sha256"],
                "source_commit": normalized["source_commit"],
            }
            status = "captured_verified_source_metadata"
        finally:
            sys.modules.pop(SOURCE_CAPTURE_MODULE, None)
    elif command == "capture-runtime-manifest":
        package_roots = _parse_named_path_arguments(
            args.package_root,
            context="package root",
        )
        package_import_paths = _parse_named_import_path_arguments(
            args.package_import_path,
            context="package import path",
        )
        payload = capture_calibration_runtime_manifest(
            git_executable_path=args.git_executable,
            base_runtime_root=args.base_runtime_root,
            package_roots=package_roots,
            package_import_paths=package_import_paths,
            interpreter_path=args.staged_interpreter,
        )
    else:
        payload = capture_model_file_manifest_from_hub(
            args.model_id,
            args.revision,
            transformers_version=args.transformers_version,
            token=False,
        )
    if command != "capture-source-manifest":
        output = args.output.resolve()
        _atomic_publish_new(output, payload)
    print(
        json.dumps(
            {
                **details,
                "file_sha256": sha256_bytes(payload),
                "output": str(output),
                "status": status,
            },
            sort_keys=True,
        )
    )
    return 0


def _official_main(
    argv: Sequence[str],
    *,
    runtime_context: SealedRuntimeContext,
    interpreter_path: Path,
) -> int:
    global _AUTHENTICATED_CALIBRATION_API
    global _AUTHENTICATED_IDENTITY_RESOLVER
    global _AUTHENTICATED_SOURCE_VERIFIER

    arguments = list(argv)
    if any(
        module is not None
        for module in (
            _AUTHENTICATED_CALIBRATION_API,
            _AUTHENTICATED_IDENTITY_RESOLVER,
            _AUTHENTICATED_SOURCE_VERIFIER,
        )
    ):
        raise CalibrationRunError("authenticated runner modules were already loaded")
    args = _parser().parse_args(arguments)
    ruler_receipt_dir = _verify_ruler_receipt_directory_precondition(args.ruler_receipt_dir)
    identity_bytes = args.frozen_identity.read_bytes()
    bindings = _bootstrap_identity_bindings(identity_bytes)
    source_manifest_bytes = args.repository_source_manifest.read_bytes()
    runtime_manifest_bytes = args.runtime_manifest.read_bytes()
    model_manifest_bytes = args.model_file_manifest.read_bytes()
    parquet_manifest_bytes = args.parquet_materialization_manifest.read_bytes()
    exact_manifest_hashes = {
        "repository source": (
            sha256_bytes(source_manifest_bytes),
            bindings.repository_source_manifest_file_sha256,
        ),
        "runtime": (sha256_bytes(runtime_manifest_bytes), bindings.runtime_manifest_file_sha256),
        "model": (sha256_bytes(model_manifest_bytes), bindings.model_file_manifest_file_sha256),
        "parquet materialization": (
            sha256_bytes(parquet_manifest_bytes),
            bindings.parquet_materialization_manifest_file_sha256,
        ),
    }
    for name, (actual, expected) in exact_manifest_hashes.items():
        if actual != expected:
            raise CalibrationRunError(
                f"{name} manifest bytes differ from the frozen identity bootstrap binding"
            )
    if args.fisher_h1_smoke:
        if (
            args.prior_fisher_h1_smoke_report is not None
            or args.prior_fisher_h1_smoke_complete_marker is not None
        ):
            raise CalibrationRunError(
                "Fisher H=1 smoke mode forbids prior smoke prerequisite paths"
            )
        prior_smoke_report_bytes = None
        prior_smoke_complete_bytes = None
    else:
        if (
            args.prior_fisher_h1_smoke_report is None
            or args.prior_fisher_h1_smoke_complete_marker is None
        ):
            raise CalibrationRunError(
                "full calibration requires prior Fisher H=1 smoke report and marker paths"
            )
        prior_smoke_report_bytes = args.prior_fisher_h1_smoke_report.read_bytes()
        prior_smoke_complete_bytes = args.prior_fisher_h1_smoke_complete_marker.read_bytes()
    if (
        _sha256(
            args.expected_runtime_manifest_sha256,
            context="expected runtime manifest SHA-256",
        )
        != bindings.runtime_manifest_file_sha256
    ):
        raise CalibrationRunError("CLI runtime manifest SHA-256 differs from frozen identity")
    if (
        _sha256(
            args.expected_model_file_manifest_sha256,
            context="expected model manifest SHA-256",
        )
        != bindings.model_file_manifest_file_sha256
    ):
        raise CalibrationRunError("CLI model manifest SHA-256 differs from frozen identity")
    if (
        _sha256(
            args.expected_parquet_materialization_manifest_sha256,
            context="expected parquet materialization manifest SHA-256",
        )
        != bindings.parquet_materialization_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "CLI parquet materialization manifest SHA-256 differs from frozen identity"
        )

    bootstrap_source = _bootstrap_source_manifest(
        source_manifest_bytes,
        repository_root=args.repository_root,
        require_adapter=True,
    )
    requested_commit = _git_revision(args.source_commit, context="requested source commit")
    if requested_commit != bootstrap_source.source_commit:
        raise CalibrationRunError("CLI source commit differs from source-manifest commit")
    source_git = bootstrap_source.manifest["git_executable"]
    runtime_git = _authenticate_git_executable(runtime_context.git_executable_path)
    if source_git != {
        "sha256": runtime_git.sha256,
        "size_bytes": runtime_git.size_bytes,
    }:
        raise CalibrationRunError(
            "source and runtime manifests bind different Git executable bytes"
        )

    capture_provenance_receipt_file_sha256 = _authenticate_calibration_identity_capture_provenance(
        receipt_path=args.capture_provenance_receipt,
        expected_receipt_sha256=(args.expected_capture_provenance_receipt_sha256),
        runtime_manifest_path=args.runtime_manifest,
        expected_runtime_manifest_sha256=(args.expected_runtime_manifest_sha256),
        source_manifest_bytes=source_manifest_bytes,
        expected_identity_input_sha256=(bindings.identity_input_manifest_sha256),
        expected_bindings=bindings,
        expected_source_commit=requested_commit,
    )
    capture_provenance_receipt_bytes = _read_stable_regular_bytes(
        args.capture_provenance_receipt,
        context="calibration identity capture provenance receipt after authentication",
    )
    if sha256_bytes(capture_provenance_receipt_bytes) != (capture_provenance_receipt_file_sha256):
        raise CalibrationRunError("capture provenance receipt changed after authentication")

    requirements_path = _assert_no_link_components(
        Path(os.path.abspath(args.repository_root)),
        PurePosixPath(CALIBRATION_REQUIREMENTS_PATH),
    )
    requirements_sha256, _requirements_size = _stream_file_sha256(requirements_path)
    if requirements_sha256 != bootstrap_source.entries[CALIBRATION_REQUIREMENTS_PATH]["raw_sha256"]:
        raise CalibrationRunError("calibration requirements changed before runtime preflight")
    runtime_manifest = parse_calibration_runtime_manifest(runtime_manifest_bytes)
    _preflight_runtime_requirements(
        runtime_manifest,
        _parse_runtime_requirements(requirements_path),
    )

    _AUTHENTICATED_CALIBRATION_API = _load_exact_source_module(
        CALIBRATION_API_MODULE,
        CALIBRATION_API_PATH,
        repository_root=args.repository_root,
        entry=bootstrap_source.entries[CALIBRATION_API_PATH],
    )
    _AUTHENTICATED_IDENTITY_RESOLVER = _load_exact_source_module(
        IDENTITY_RESOLVER_MODULE,
        IDENTITY_RESOLVER_SOURCE_PATH,
        repository_root=args.repository_root,
        entry=bootstrap_source.entries[IDENTITY_RESOLVER_SOURCE_PATH],
    )
    _AUTHENTICATED_SOURCE_VERIFIER = _load_exact_source_module(
        "recurquant.experiment013_source",
        SOURCE_VERIFIER_PATH,
        repository_root=args.repository_root,
        entry=bootstrap_source.entries[SOURCE_VERIFIER_PATH],
    )
    services = default_services(
        args.repository_root,
        base_runtime_root=runtime_context.base_runtime_root,
        calibration_api=_AUTHENTICATED_CALIBRATION_API,
        interpreter_path=interpreter_path,
        package_roots=runtime_context.package_roots,
        git_executable_path=runtime_context.git_executable_path,
    )
    identity = services.backend.decode_identity(identity_bytes)
    if (
        identity.repository_source_manifest_file_sha256
        != bindings.repository_source_manifest_file_sha256
        or identity.runtime_manifest_file_sha256 != bindings.runtime_manifest_file_sha256
        or identity.model_file_manifest_file_sha256 != bindings.model_file_manifest_file_sha256
        or identity.parquet_materialization_manifest_file_sha256
        != bindings.parquet_materialization_manifest_file_sha256
        or identity.identity_input_manifest_sha256 != bindings.identity_input_manifest_sha256
    ):
        raise CalibrationRunError("full identity decode differs from bootstrap identity bindings")
    verified_source, _source_sha256 = services.verify_repository_source(
        bootstrap_source.manifest,
        args.repository_root,
    )
    if verified_source.get("source_commit") != requested_commit:
        raise CalibrationRunError(
            "verified source-manifest commit differs from requested frozen source commit"
        )
    authenticated_runtime = services.authenticate_runtime(runtime_manifest)
    if authenticated_runtime.manifest_file_sha256 != bindings.runtime_manifest_file_sha256:
        raise CalibrationRunError("runtime authenticator returned a different manifest identity")
    model_manifest = parse_model_file_manifest(model_manifest_bytes)
    _model_contract_matches(identity, model_manifest)
    context = _adapter_construction_context(
        calibration_api=_AUTHENTICATED_CALIBRATION_API,
        repository_root=args.repository_root,
        model_root=args.model_root,
        cache_root=args.cache_root,
        ruler_root=ruler_receipt_dir,
        repository_source_manifest_bytes=source_manifest_bytes,
        calibration_runtime_manifest_bytes=runtime_manifest_bytes,
        model_file_manifest_bytes=model_manifest_bytes,
        parquet_materialization_manifest_bytes=parquet_manifest_bytes,
        runtime_context=runtime_context,
        interpreter_path=interpreter_path,
    )
    adapter = _load_adapter(
        args.adapter,
        repository_root=args.repository_root,
        source_entry=bootstrap_source.entries[CANONICAL_ADAPTER_PATH],
        calibration_api=_AUTHENTICATED_CALIBRATION_API,
        context=context,
    )
    config = CalibrationRunConfig(
        frozen_identity_bytes=identity_bytes,
        repository_source_manifest_bytes=source_manifest_bytes,
        model_file_manifest_bytes=model_manifest_bytes,
        parquet_materialization_manifest_bytes=parquet_manifest_bytes,
        runtime_manifest_bytes=runtime_manifest_bytes,
        model_root=args.model_root,
        repository_root=args.repository_root,
        expected_source_commit=args.source_commit,
        expected_model_file_manifest_sha256=args.expected_model_file_manifest_sha256,
        expected_parquet_materialization_manifest_sha256=(
            args.expected_parquet_materialization_manifest_sha256
        ),
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        capture_provenance_receipt_bytes=capture_provenance_receipt_bytes,
        expected_capture_provenance_receipt_sha256=(capture_provenance_receipt_file_sha256),
        output_dir=args.output_dir,
        require_cuda=True,
        fisher_h1_smoke=args.fisher_h1_smoke,
        prior_fisher_h1_smoke_report_bytes=prior_smoke_report_bytes,
        prior_fisher_h1_smoke_complete_bytes=prior_smoke_complete_bytes,
    )
    result = run_calibration(
        config,
        adapter,
        services=services,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _preflight_official_runtime_distributions(runtime: AuthenticatedRuntime) -> None:
    """Fail before adapter or model access unless the frozen datasets wheel is present."""

    if "datasets" in sys.modules:
        raise CalibrationRunError("datasets was imported before the official runtime preflight")
    distributions = dict(runtime.distributions)
    if distributions.get("datasets") != OFFICIAL_DATASETS_DISTRIBUTION_VERSION:
        raise CalibrationRunError(
            "official calibration runtime requires exactly "
            f"datasets=={OFFICIAL_DATASETS_DISTRIBUTION_VERSION}"
        )


def _preflight_runtime_requirements(
    manifest: CalibrationRuntimeManifest,
    requirements: Sequence[RuntimeRequirement],
) -> None:
    """Bind every authenticated runtime distribution to the source-frozen pins."""

    expected = tuple((item.name, item.version) for item in requirements)
    observed = tuple((item.name, item.version) for item in manifest.distributions)
    if observed != expected:
        raise CalibrationRunError(
            "official runtime distributions differ from the source-bound calibration requirements"
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
    """Run only after the stdlib bootstrap supplies explicit authenticated roots."""

    arguments = list(argv)
    capture_mode = bool(arguments and arguments[0] in _IDENTITY_CAPTURE_COMMANDS)
    args = (
        _parse_identity_capture_arguments(arguments)
        if capture_mode
        else _parser().parse_args(arguments)
    )
    _verify_ruler_receipt_directory_precondition(args.ruler_receipt_dir)
    runtime_manifest_bytes = args.runtime_manifest.read_bytes()
    manifest, runtime_context, authenticated = _authenticate_sealed_runtime_context(
        runtime_manifest_bytes,
        base_runtime_root=base_runtime_root,
        package_roots=package_roots,
        package_import_paths=package_import_paths,
        interpreter_path=interpreter_path,
        git_executable_path=git_executable_path,
        pycache_prefix=pycache_prefix,
    )
    if manifest.file_sha256 != _sha256(
        args.expected_runtime_manifest_sha256,
        context="expected runtime manifest SHA-256",
    ):
        raise CalibrationRunError("sealed runtime manifest differs from the CLI binding")
    _preflight_official_runtime_distributions(authenticated)
    result = (
        _sealed_capture_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=Path(interpreter_path),
        )
        if capture_mode
        else _official_main(
            arguments,
            runtime_context=runtime_context,
            interpreter_path=Path(interpreter_path),
        )
    )
    _verify_sealed_launch_state(
        runtime_context.pycache_prefix,
        manifest=manifest,
        roots={
            BASE_RUNTIME_ROOT_NAME: runtime_context.base_runtime_root,
            **dict(runtime_context.package_roots),
        },
    )
    authenticate_calibration_runtime(
        manifest,
        base_runtime_root=runtime_context.base_runtime_root,
        package_roots=runtime_context.package_roots,
        interpreter_path=interpreter_path,
        git_executable_path=git_executable_path,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose authenticated preparation only; official runs require the sealed launcher."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    capture_result = _capture_manifest_mode(arguments)
    if capture_result is not None:
        return capture_result
    raise CalibrationRunError(
        "official calibration must be started with launch_static_q468_calibration.py"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
