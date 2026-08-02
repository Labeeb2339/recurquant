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
import csv
import hashlib
import importlib
import importlib.metadata
import json
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
SOURCE_VERIFIER_PATH: Final = "src/recurquant/experiment013_source.py"
SOURCE_CAPTURE_MODULE: Final = "recurquant_experiment013_source_capture"
RUNNER_SOURCE_PATH: Final = "scripts/run_static_q468_calibration.py"
IDENTITY_RESOLVER_SOURCE_PATH: Final = "scripts/resolve_static_q468_identity.py"
CANONICAL_ADAPTER_SPEC: Final = "recurquant.experiment013_qwen35_adapter:create_adapter"
CANONICAL_ADAPTER_MODULE: Final = "recurquant.experiment013_qwen35_adapter"
CANONICAL_ADAPTER_PATH: Final = "src/recurquant/experiment013_qwen35_adapter.py"

RUNNER_REVISION: Final = "experiment-013-static-q468-calibration-runner-v1"
MODEL_FILE_MANIFEST_KIND: Final = "recurquant_experiment013_model_file_manifest"
MODEL_FILE_MANIFEST_SCHEMA: Final = 1
MODEL_FILE_MANIFEST_DERIVATION: Final = "huggingface-hub-pinned-tree-lfs-v1"
MODEL_FILE_SELECTION_PROFILE: Final = "qwen35-config-index-safetensors-v1"
RUNTIME_MANIFEST_KIND: Final = "recurquant_experiment013_calibration_runtime_manifest"
RUNTIME_MANIFEST_SCHEMA: Final = 3
RUN_REPORT_KIND: Final = "recurquant_experiment013_calibration_run"
RUN_REPORT_SCHEMA: Final = 1

QUERY_EMA_DECAY: Final = 2.0 ** (-1.0 / 32.0)
QUERY_ENERGY_EPSILON: Final = 1.0e-6
RHT_SEED: Final = 2_339

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE: Final = re.compile(r"[0-9a-f]{40}")
_SAFE_MODEL_FILE_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_WEIGHT_FILE_RE: Final = re.compile(
    r"(?:^|/)(?:model(?:-[0-9]+-of-[0-9]+)?|"
    r"model\.safetensors-[0-9]+-of-[0-9]+)\.safetensors$"
)

SCORE_FILENAME: Final = "calibration-scores.json"
SPLIT_FILENAME: Final = "split-half-stability.json"
K27030_FILENAME: Final = "static-k27030-policy.json"
K29334_FILENAME: Final = "static-k29334-policy.json"
Q48_FILENAME: Final = "static-q48-p14739-policy.json"
BINDING_FILENAME: Final = "stage-a-calibration-binding.json"
REPORT_FILENAME: Final = "calibration-run-report.json"
COMPLETE_FILENAME: Final = "CALIBRATION_COMPLETE"
PREPARED_RUNTIME_MANIFEST_FILENAME: Final = "calibration-runtime-manifest.json"
PREPARED_RUNTIME_COMPLETE_FILENAME: Final = "RUNTIME_PREPARED"
DEFAULT_PACKAGE_RUNTIME_ROOT_NAME: Final = "calibration-packages"
DEFAULT_PACKAGE_IMPORT_PATH: Final = "Lib/site-packages"
STAGING_EXCLUDED_DISTRIBUTIONS: Final = frozenset({"pip", "setuptools"})

_WINDOWS_REPARSE_POINT: Final = 0x400
_RUNTIME_ROOT_NAME_RE: Final = re.compile(r"[a-z][a-z0-9-]{0,63}")
BASE_RUNTIME_ROOT_NAME: Final = "base-runtime"
_FORBIDDEN_RUNTIME_SUFFIXES: Final = frozenset(
    {"._pth", ".egg-link", ".pth", ".pyc", ".pyo"}
)
_FORBIDDEN_RUNTIME_DIRECTORY_NAMES: Final = frozenset({"__pycache__"})
_FORBIDDEN_RUNTIME_FILENAMES: Final = frozenset(
    {"pyvenv.cfg", "sitecustomize.py", "usercustomize.py"}
)
SEALED_LAUNCH_POLICY: Final = {
    "bootstrap_mode": "stdlib-only-exact-runner-v1",
    "dont_write_bytecode": 1,
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


@dataclass(frozen=True, slots=True)
class BootstrapIdentityBindings:
    repository_source_manifest_file_sha256: str
    runtime_manifest_file_sha256: str
    model_file_manifest_file_sha256: str
    parquet_materialization_manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class BootstrapSource:
    manifest: dict[str, object]
    source_commit: str
    entries: dict[str, dict[str, object]]


def _bootstrap_identity_bindings(data: bytes) -> BootstrapIdentityBindings:
    """Strictly extract only the v4 execution bindings using stdlib code.

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
        or evidence.get("schema_version") != 4
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
    if (
        path.is_absolute()
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CalibrationRunError(f"{context} must be repository-relative")
    return value


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
        RUNNER_SOURCE_PATH,
        IDENTITY_RESOLVER_SOURCE_PATH,
        SOURCE_VERIFIER_PATH,
        CALIBRATION_API_PATH,
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
    source_bytes = source_path.read_bytes()
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
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _assert_source_manifest_output_location(repository_root: Path, output: Path) -> Path:
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
        ["git", "check-ignore", "--quiet", "--no-index", "--", relative.as_posix()],
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
    """Only the current sequence's frozen anchor tensors, all on CPU FP64."""

    anchor_positions: tuple[int, ...]
    query_energy: Any
    q4_mse: Any
    q6_mse: Any
    q8_mse: Any


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
    pycache_prefix: Path


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    """In-memory pass artifacts; no publication occurs until all are built."""

    score: bytes
    split_half: bytes
    static_k27030: bytes
    static_k29334: bytes
    static_q48: bytes
    stage_a_binding: bytes
    stability: Mapping[str, object]
    calibration_scores_sha256: str
    sequence_score_manifest_sha256: str


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
    output_dir: Path
    require_cuda: bool = True


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
ModelAuthenticator = Callable[[Path, ModelFileManifest], Any]
RuntimeAuthenticator = Callable[[CalibrationRuntimeManifest], AuthenticatedRuntime]


@dataclass(frozen=True, slots=True)
class RunnerServices:
    backend: CalibrationBackend
    calibration_api: ModuleType
    verify_repository_source: SourceVerifier
    validate_adapter: AdapterValidator
    distortion_function: DistortionFunction
    authenticate_model_files: ModelAuthenticator
    authenticate_runtime: RuntimeAuthenticator


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
        if _WEIGHT_FILE_RE.search(name) and lfs_sha256 is None:
            raise ValueError("safetensors weights require a pinned Hub LFS identity")
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
    if "config.json" not in names:
        raise ValueError("model file manifest must authenticate config.json")
    if not any(_WEIGHT_FILE_RE.search(name) for name in names):
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


def _selected_model_tree_path(path: str) -> bool:
    if "/" in path:
        return False
    return path in {"config.json", "model.safetensors.index.json"} or bool(
        _WEIGHT_FILE_RE.fullmatch(path)
    )


def capture_model_file_manifest_from_hub(
    model_id: str,
    revision: str,
    *,
    transformers_version: str,
    api: object | None = None,
    tree_entries: Sequence[object] | None = None,
    resolved_revision: str | None = None,
    token: str | bool | None = None,
) -> bytes:
    """Build the local-file contract using only pinned Hub tree/LFS metadata.

    The function never downloads or opens model files. Ordinary files are
    authenticated by their Git blob OID. LFS files additionally bind the
    content SHA-256 and byte size advertised by the pinned Hub revision.
    """

    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ValueError("model_id must be a non-empty canonical string")
    pinned_revision = _git_revision(revision, context="model Hub revision")
    if (
        not isinstance(transformers_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", transformers_version) is None
    ):
        raise ValueError("Transformers version must be exact semver")
    if tree_entries is None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        info = api.model_info(  # type: ignore[attr-defined]
            model_id,
            revision=pinned_revision,
            files_metadata=False,
            token=token,
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
                token=token,
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


def authenticate_local_model_files(
    model_root: Path,
    manifest: ModelFileManifest,
    *,
    calibration_api: ModuleType,
) -> Any:
    """Hash every exact local model file immediately before model loading."""

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
        raise CalibrationRunError(
            "package import paths must exactly match the named package roots"
        )
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
            _canonical_relative_path(item, context="base sys.path entry") for item in supplied
        )
    else:
        root = base_runtime_root.resolve(strict=True)
        if Path(sys.prefix).resolve(strict=True) != root or Path(sys.base_prefix).resolve(
            strict=True
        ) != root:
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
            captured.append(
                _canonical_relative_path(relative, context="base sys.path entry")
            )
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
                raise CalibrationRunError(
                    f"runtime tree path is unavailable: {relative}"
                ) from exc
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
                raise CalibrationRunError(
                    f"distribution {name} RECORD row {index} is malformed"
                )
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
    import_root = package_roots[selected_root] / PurePosixPath(
        package_import_paths[selected_root]
    )
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
        "launch_policy": dict(SEALED_LAUNCH_POLICY),
        "machine": dict(
            zip(
                ("system", "architecture", "machine", "byteorder", "pointer_bits"),
                machine_identity,
                strict=True,
            )
        ),
        "package_roots": [
            {"import_path": import_paths[name], "name": name} for name in packages
        ],
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
                sha256=_sha256(
                    raw_file["sha256"], context=f"runtime tree {name} file SHA-256"
                ),
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
        _canonical_relative_path(item, context="base sys.path entry")
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
    interpreter_sha256 = _sha256(
        interpreter["sha256"], context="runtime interpreter SHA-256"
    )
    interpreter_size = _positive_int(
        interpreter["size_bytes"], context="runtime interpreter size"
    )
    base_files = {item.path: item for item in trees[0].files}
    for sys_path_entry in base_sys_path:
        present = sys_path_entry in base_files or any(
            path.startswith(f"{sys_path_entry}/") for path in base_files
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
) -> AuthenticatedRuntime:
    """Rehash both complete staged trees and exact RECORD inventories."""

    if not isinstance(manifest, CalibrationRuntimeManifest):
        raise TypeError("manifest must be CalibrationRuntimeManifest")
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


_RUNTIME_PROBE_SOURCE: Final = r'''
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
'''.strip()


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
            _canonical_relative_path(item, context="runtime probe base sys.path")
            for item in raw_paths
        ),
    )


def prepare_calibration_runtime(
    *,
    source_python: Path,
    requirements_file: Path,
    output_root: Path,
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
        str(roots[item.name] / PurePosixPath(item.import_path))
        for item in manifest.package_roots
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
    pycache_prefix: Path,
) -> tuple[CalibrationRuntimeManifest, SealedRuntimeContext, AuthenticatedRuntime]:
    """Reauthenticate explicit launcher inputs without copying them to globals."""

    manifest = parse_calibration_runtime_manifest(runtime_manifest_bytes)
    roots = _runtime_root_map(base_runtime_root, package_roots)
    declared_names = tuple(item.name for item in manifest.package_roots)
    actual_names = tuple(name for name in roots if name != BASE_RUNTIME_ROOT_NAME)
    if actual_names != declared_names:
        raise CalibrationRunError(
            "bootstrap package roots differ from the frozen runtime manifest"
        )
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
    )
    context = SealedRuntimeContext(
        manifest_file_sha256=manifest.file_sha256,
        base_runtime_root=roots[BASE_RUNTIME_ROOT_NAME],
        package_roots={item.name: roots[item.name] for item in manifest.package_roots},
        package_import_paths=normalized_import_paths,
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


def _identity_view(data: bytes, repository_root: Path) -> FrozenCalibrationIdentity:
    resolver = _load_identity_resolver(repository_root)
    decoded = resolver.deserialize_frozen_calibration_identity_artifact(data)
    root = _strict_json_bytes(data, context="frozen calibration identity")
    evidence = root.get("evidence")
    if not isinstance(evidence, dict):  # independently decoded above; defensive only
        raise ValueError("frozen identity evidence is missing")
    model_contracts = cast(dict[str, object], evidence["model_contracts"])
    primary = cast(dict[str, object], model_contracts["primary"])
    tokenizer = cast(dict[str, object], evidence["tokenizer"])
    execution_bindings = getattr(decoded, "execution_bindings", None)
    if not isinstance(execution_bindings, Mapping):
        raise CalibrationRunError(
            "frozen identity does not contain schema-v4 execution_bindings; "
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
    return FrozenCalibrationIdentity(
        file_sha256=decoded.file_sha256,
        canonical_evidence_sha256=decoded.canonical_evidence_sha256,
        records=tuple(dict(record) for record in decoded.records),
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


def verify_repository_source_manifest(
    expected: Mapping[str, object],
    repository_root: Path,
) -> tuple[dict[str, object], str]:
    """Use the frozen source API to reauthenticate code at point of use."""

    module = _AUTHENTICATED_SOURCE_VERIFIER
    if module is None:
        raise CalibrationRunError("repository source verifier was not bootstrap-authenticated")
    normalized_expected = module.validate_experiment013_source_manifest(expected)
    verified = module.verify_experiment013_source_manifest(
        normalized_expected,
        repo_root=repository_root,
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


def _verify_repository_commit(repository_root: Path, expected: str) -> str:
    revision = _git_revision(expected, context="expected source commit")
    process = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise CalibrationRunError("cannot resolve repository HEAD for policy provenance")
    actual = process.stdout.strip()
    if actual != revision:
        raise CalibrationRunError(
            f"repository HEAD differs from expected source commit: {actual!r} != {revision!r}"
        )
    return revision


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
    """Return per-row RHT Q4/Q6/Q8 MSE, releasing layer workspaces eagerly."""

    torch = _torch_runtime()
    from recurquant.quantization import QuantizationSpec, quantize_dequantize
    from recurquant.rht import right_rht_encode

    expected = (
        geometry.layers,
        geometry.heads,
        geometry.key_rows,
        geometry.value_width,
    )
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != expected:
        raise CalibrationRunError(f"anchor state must have shape {expected}")
    if not state.is_floating_point() or not torch.isfinite(state).all().item():
        raise CalibrationRunError("anchor state must be finite floating point")
    specifications = tuple(
        QuantizationSpec(
            bits=bits,
            group_size=geometry.value_width,
            scale_bits=16,
            flatten_last_dims=1,
            rounding="nearest",
            seed=RHT_SEED,
        )
        for bits in (4, 6, 8)
    )
    per_bit: list[list[Any]] = [[], [], []]
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
                mse = (restored - encoded).square().mean(dim=-1).squeeze(0)
                destination.append(mse.detach().to(device="cpu", dtype=torch.float64))
    return cast(
        tuple[Any, Any, Any],
        tuple(torch.stack(rows, dim=0).contiguous() for rows in per_bit),
    )


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
) -> CapturedSequence:
    """Process exactly one token per adapter call and retain only anchor tensors."""

    torch = _torch_runtime()
    anchors = frozen_anchor_positions(len(token_ids))
    anchor_set = set(anchors)
    query_ema: Any | None = None
    energies: list[Any] = []
    q4_rows: list[Any] = []
    q6_rows: list[Any] = []
    q8_rows: list[Any] = []
    adapter.begin_sequence(model, record)
    completed = False
    try:
        for position, token_id in enumerate(token_ids):
            capture_state = position in anchor_set
            observation = adapter.step_token(
                model,
                token_id=token_id,
                position=position,
                capture_state=capture_state,
            )
            if not isinstance(observation, calibration_api.StepObservation):
                raise TypeError("adapter.step_token must return StepObservation")
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
                d4, d6, d8 = distortion_function(state, geometry)
                expected_rows = (geometry.layers, geometry.heads, geometry.key_rows)
                for name, tensor in (
                    ("D4", d4),
                    ("D6", d6),
                    ("D8", d8),
                ):
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tuple(tensor.shape) != expected_rows
                        or tensor.device.type != "cpu"
                        or tensor.dtype != torch.float64
                        or not torch.isfinite(tensor).all().item()
                        or (tensor < 0).any().item()
                    ):
                        raise CalibrationRunError(
                            f"{name} distortion must be finite non-negative CPU FP64 "
                            f"{expected_rows}"
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
    if not completed or len(energies) != len(anchors):
        raise CalibrationRunError("causal sequence capture did not complete every frozen anchor")
    return CapturedSequence(
        anchor_positions=anchors,
        query_energy=torch.stack(energies).contiguous(),
        q4_mse=torch.stack(q4_rows).contiguous(),
        q6_mse=torch.stack(q6_rows).contiguous(),
        q8_mse=torch.stack(q8_rows).contiguous(),
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
            AnchorDistortionBatch,
            reduce_frozen_anchor_distortions,
        )

        batch = AnchorDistortionBatch(
            family=cast(Any, record["family"]),
            config=cast(str, record["config"]),
            ruler_category=cast(Any, record["ruler_category"]),
            canonical_id=cast(str, record["canonical_id"]),
            seed=cast(int | None, record["seed"]),
            configured_length=cast(int | None, record["configured_length"]),
            token_count=len(token_ids),
            anchor_positions=captured.anchor_positions,
            query_energy=captured.query_energy,
            q4_mse=captured.q4_mse,
            q6_mse=captured.q6_mse,
            q8_mse=captured.q8_mse,
            sequence_token_ids=token_ids,
            identity_record=record,
        )
        return reduce_frozen_anchor_distortions(batch)

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
            STATIC_Q468_PRIMARY_METHOD,
            build_static_rht_q48_policy,
            build_static_rht_q468_policy,
            deserialize_static_rht_q48_policy,
            deserialize_static_rht_q468_policy,
            serialize_static_rht_q48_policy,
            serialize_static_rht_q468_policy,
        )
        from recurquant.static_q468_calibration import (
            aggregate_calibration_scores,
            build_frozen_calibration_score_artifact,
            build_frozen_split_half_stability_artifact,
            deserialize_calibration_score_artifact,
            deserialize_frozen_split_half_stability_artifact,
            fit_split_half_policy,
        )

        resolver = _load_identity_resolver(self.repository_root)
        typed_scores = cast(list[Any], list(scores))
        aggregate = aggregate_calibration_scores(typed_scores)
        geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
        fit = fit_split_half_policy(
            typed_scores,
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
        q48_bytes = serialize_static_rht_q48_policy(q48)
        deserialize_static_rht_q468_policy(k27030_bytes)
        deserialize_static_rht_q468_policy(k29334_bytes)
        deserialize_static_rht_q48_policy(q48_bytes)
        binding_bytes = resolver.build_stage_a_calibration_binding_artifact(
            frozen_identity_artifact=identity.artifact_bytes,
            calibration_score_artifact=score_bytes,
            split_half_stability_artifact=split_bytes,
            static_k27030_policy_artifact=k27030_bytes,
            static_k29334_policy_artifact=k29334_bytes,
        )
        resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)
        return FinalizationResult(
            passed=True,
            stability=stability,
            artifacts=CalibrationArtifacts(
                score=score_bytes,
                split_half=split_bytes,
                static_k27030=k27030_bytes,
                static_k29334=k29334_bytes,
                static_q48=q48_bytes,
                stage_a_binding=binding_bytes,
                stability=stability,
                calibration_scores_sha256=decoded_score.calibration_scores_sha256,
                sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
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
    anchor_count: int,
    stability: Mapping[str, object],
    artifacts: Mapping[str, bytes],
    runtime: Mapping[str, object],
) -> bytes:
    evidence = {
        "artifacts": {name: sha256_bytes(payload) for name, payload in sorted(artifacts.items())},
        "calibration": {
            "anchor_count": anchor_count,
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


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("artifact payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _publish_output_directory(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    resolved = Path(os.path.abspath(output_dir))
    parent = resolved.parent
    parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing calibration output: {resolved}")
    prefix = f".{resolved.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    owned_staging = True
    # The Stage-A binding and completion marker are deliberately last. The
    # public directory appears only after every dependency is durable.
    ordered = [name for name in sorted(payloads) if name not in {REPORT_FILENAME, BINDING_FILENAME}]
    if REPORT_FILENAME in payloads:
        ordered.append(REPORT_FILENAME)
    if BINDING_FILENAME in payloads:
        ordered.append(BINDING_FILENAME)
    try:
        for name in ordered:
            _atomic_publish_new(staging / name, payloads[name])
        _atomic_publish_new(
            staging / COMPLETE_FILENAME,
            b"recurquant-experiment013-calibration-complete-v1\n",
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
    # First executable boundary: a strict promoted identity decode. No source
    # adapter, model path, repository command, or output path is touched first.
    identity = services.backend.decode_identity(config.frozen_identity_bytes)
    if config.output_dir.resolve().exists():
        raise FileExistsError(
            f"refusing to overwrite existing calibration output: {config.output_dir.resolve()}"
        )

    source_commit = _verify_repository_commit(
        config.repository_root,
        config.expected_source_commit,
    )
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
    parquet_manifest_file_sha256 = sha256_bytes(
        config.parquet_materialization_manifest_bytes
    )
    if (
        parquet_manifest_file_sha256 != expected_parquet_manifest_sha256
        or parquet_manifest_file_sha256
        != identity.parquet_materialization_manifest_file_sha256
    ):
        raise CalibrationRunError(
            "parquet materialization manifest bytes differ from the frozen identity/config "
            "binding"
        )

    runtime_before_data = services.authenticate_runtime(runtime_manifest)
    if runtime_before_data != authenticated_runtime:
        raise CalibrationRunError("calibration runtime identity changed before data access")

    materialized: list[tuple[dict[str, object], tuple[int, ...]]] = []
    for record in identity.records:
        candidate = adapter.materialize_sequence(record)
        materialized.append(
            (
                record,
                validate_materialized_sequence(
                    record,
                    candidate,
                    calibration_api=services.calibration_api,
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
    try:
        # The adapter may call AutoModel only inside this method. The exact local
        # file set has already been hashed, revision checked, and source verified.
        model = adapter.load_model(authenticated_model)
        authenticated_after_load = services.authenticate_model_files(
            config.model_root, model_manifest
        )
        if authenticated_after_load != authenticated_model:
            raise CalibrationRunError("local model identity changed while loading weights")
        for record, token_ids in materialized:
            captured = capture_sequence_causally(
                adapter,
                model,
                record,
                token_ids,
                geometry=services.backend.geometry,
                calibration_api=services.calibration_api,
                require_cuda=config.require_cuda,
                distortion_function=services.distortion_function,
            )
            scores.append(services.backend.reduce_sequence(record, token_ids, captured))
            total_tokens += len(token_ids)
            total_anchors += len(captured.anchor_positions)
        result = services.backend.finalize(
            scores,
            identity=identity,
            source_commit=source_commit,
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
            anchor_count=total_anchors,
            stability=result.stability,
            artifacts={},
            runtime=runtime,
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
        SPLIT_FILENAME: artifacts.split_half,
        K27030_FILENAME: artifacts.static_k27030,
        K29334_FILENAME: artifacts.static_k29334,
        Q48_FILENAME: artifacts.static_q48,
        BINDING_FILENAME: artifacts.stage_a_binding,
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
        anchor_count=total_anchors,
        stability=result.stability,
        artifacts=payloads,
        runtime=runtime,
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
) -> RunnerServices:
    api = _AUTHENTICATED_CALIBRATION_API if calibration_api is None else calibration_api
    if api is None:
        raise CalibrationRunError("calibration API was not bootstrap-authenticated")
    backend = Experiment013Backend(repository_root)
    return RunnerServices(
        backend=backend,
        calibration_api=api,
        verify_repository_source=verify_repository_source_manifest,
        validate_adapter=lambda adapter: validate_adapter_contract(
            adapter,
            calibration_api=api,
        ),
        distortion_function=compute_anchor_distortions,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-identity", required=True, type=Path)
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
    parser.add_argument("--ruler-root", required=True, type=Path)
    parser.add_argument("--repository-root", default=REPOSITORY_ROOT, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        choices=[CANONICAL_ADAPTER_SPEC],
        default=CANONICAL_ADAPTER_SPEC,
        help="The single reviewed live adapter (generic adapters are test-injection only).",
    )
    return parser


def _capture_manifest_mode(arguments: Sequence[str]) -> int | None:
    if not arguments or arguments[0] not in {
        "capture-source-manifest",
        "capture-runtime-manifest",
        "capture-model-manifest",
        "prepare-runtime",
    }:
        return None
    command = arguments[0]
    parser = argparse.ArgumentParser(prog=f"{Path(__file__).name} {command}")
    if command == "prepare-runtime":
        parser.add_argument("--source-python", required=True, type=Path)
        parser.add_argument("--requirements", required=True, type=Path)
        parser.add_argument("--output-root", required=True, type=Path)
        parser.add_argument(
            "--package-root-name",
            default=DEFAULT_PACKAGE_RUNTIME_ROOT_NAME,
        )
    else:
        parser.add_argument("--output", required=True, type=Path)
    if command == "capture-source-manifest":
        parser.add_argument("--repository-root", required=True, type=Path)
    if command == "capture-runtime-manifest":
        parser.add_argument("--base-runtime-root", required=True, type=Path)
        parser.add_argument("--staged-interpreter", required=True, type=Path)
        parser.add_argument("--package-root", required=True, action="append")
        parser.add_argument("--package-import-path", required=True, action="append")
    if command == "capture-model-manifest":
        parser.add_argument("--model-id", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--transformers-version", required=True)
    args = parser.parse_args(arguments[1:])
    if command == "prepare-runtime":
        details = prepare_calibration_runtime(
            source_python=args.source_python,
            requirements_file=args.requirements,
            output_root=args.output_root,
            package_root_name=args.package_root_name,
        )
        print(json.dumps(details, sort_keys=True))
        return 0
    status = "captured_metadata_only"
    details: dict[str, object] = {}
    if command == "capture-source-manifest":
        source_module = _load_source_capture_module(args.repository_root)
        try:
            captured = source_module.capture_experiment013_source_manifest(args.repository_root)
            normalized = source_module.validate_experiment013_source_manifest(captured)
            payload = source_module.canonical_experiment013_source_manifest_bytes(normalized)
            output = _assert_source_manifest_output_location(args.repository_root, args.output)
            before_publish = source_module.verify_experiment013_source_manifest(
                normalized,
                args.repository_root,
            )
            if before_publish != normalized:
                raise CalibrationRunError("source verifier changed the captured manifest")
            _atomic_publish_new(output, payload)
            if output.read_bytes() != payload:
                raise CalibrationRunError("published source manifest bytes changed on disk")
            after_publish = source_module.verify_experiment013_source_manifest(
                normalized,
                args.repository_root,
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
    _verify_repository_commit(args.repository_root, requested_commit)

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
    )
    identity = services.backend.decode_identity(identity_bytes)
    if (
        identity.repository_source_manifest_file_sha256
        != bindings.repository_source_manifest_file_sha256
        or identity.runtime_manifest_file_sha256 != bindings.runtime_manifest_file_sha256
        or identity.model_file_manifest_file_sha256 != bindings.model_file_manifest_file_sha256
        or identity.parquet_materialization_manifest_file_sha256
        != bindings.parquet_materialization_manifest_file_sha256
    ):
        raise CalibrationRunError("full identity decode differs from bootstrap execution bindings")
    verified_source, _source_sha256 = services.verify_repository_source(
        bootstrap_source.manifest,
        args.repository_root,
    )
    if verified_source.get("source_commit") != requested_commit:
        raise CalibrationRunError("verified source-manifest commit differs from current HEAD")
    runtime_manifest = parse_calibration_runtime_manifest(runtime_manifest_bytes)
    authenticated_runtime = services.authenticate_runtime(runtime_manifest)
    if authenticated_runtime.manifest_file_sha256 != bindings.runtime_manifest_file_sha256:
        raise CalibrationRunError("runtime authenticator returned a different manifest identity")
    model_manifest = parse_model_file_manifest(model_manifest_bytes)
    _model_contract_matches(identity, model_manifest)
    context = _AUTHENTICATED_CALIBRATION_API.AdapterConstructionContext(
        repository_root=Path(args.repository_root),
        model_root=Path(args.model_root),
        cache_root=Path(args.cache_root),
        ruler_root=Path(args.ruler_root),
        execution_binding_artifacts={
            "repository_source_manifest_bytes": bytes(source_manifest_bytes),
            "calibration_runtime_manifest_bytes": bytes(runtime_manifest_bytes),
            "model_file_manifest_bytes": bytes(model_manifest_bytes),
            "parquet_materialization_manifest_bytes": bytes(parquet_manifest_bytes),
        },
        runtime_authentication_context={
            "base_runtime_root": runtime_context.base_runtime_root,
            "staged_interpreter": Path(interpreter_path),
            "package_runtime_roots": dict(runtime_context.package_roots),
            "package_import_paths": dict(runtime_context.package_import_paths),
        },
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
        output_dir=args.output_dir,
        require_cuda=True,
    )
    result = run_calibration(
        config,
        adapter,
        services=services,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def sealed_main(
    argv: Sequence[str],
    *,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    package_import_paths: Mapping[str, str],
    interpreter_path: Path,
    pycache_prefix: Path,
) -> int:
    """Run only after the stdlib bootstrap supplies explicit authenticated roots."""

    arguments = list(argv)
    args = _parser().parse_args(arguments)
    runtime_manifest_bytes = args.runtime_manifest.read_bytes()
    manifest, runtime_context, _authenticated = _authenticate_sealed_runtime_context(
        runtime_manifest_bytes,
        base_runtime_root=base_runtime_root,
        package_roots=package_roots,
        package_import_paths=package_import_paths,
        interpreter_path=interpreter_path,
        pycache_prefix=pycache_prefix,
    )
    if manifest.file_sha256 != _sha256(
        args.expected_runtime_manifest_sha256,
        context="expected runtime manifest SHA-256",
    ):
        raise CalibrationRunError("sealed runtime manifest differs from the CLI binding")
    result = _official_main(
        arguments,
        runtime_context=runtime_context,
        interpreter_path=Path(interpreter_path),
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
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose metadata preparation only; official runs require the sealed launcher."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    capture_result = _capture_manifest_mode(arguments)
    if capture_result is not None:
        return capture_result
    raise CalibrationRunError(
        "official calibration must be started with launch_static_q468_calibration.py"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
