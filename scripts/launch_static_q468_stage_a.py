#!/usr/bin/env python3
"""Launch Experiment 013 Stage A in the authenticated sealed runtime.

The host process is metadata-only.  It verifies the promoted v5 Stage-A
identity bindings, exact H0 source files, and complete sealed runtime before
starting the staged interpreter.  The isolated child repeats those checks and
loads exactly ``screen_static_q468_stage_a.py`` from authenticated bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Final

RUNNER_SOURCE_PATH: Final = "scripts/screen_static_q468_stage_a.py"
LAUNCHER_SOURCE_PATH: Final = "scripts/launch_static_q468_stage_a.py"
CALIBRATION_LAUNCHER_SOURCE_PATH: Final = "scripts/launch_static_q468_calibration.py"
RUNNER_MODULE_NAME: Final = "_recurquant_experiment013_sealed_stage_a_runner"
CALIBRATION_LAUNCHER_MODULE_NAME: Final = (
    "_recurquant_experiment013_calibration_launcher_for_stage_a"
)
IDENTITY_SCHEMA: Final = 5
BASE_RUNTIME_ROOT_NAME: Final = "base-runtime"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_BOUND_ARTIFACT_OPTIONS: Final = {
    "calibration_runtime_manifest_file_sha256": "--runtime-manifest",
    "model_file_manifest_file_sha256": "--model-file-manifest",
    "parquet_materialization_manifest_file_sha256": ("--parquet-materialization-manifest"),
    "repository_source_manifest_file_sha256": "--repository-source-manifest",
}
_EXPECTED_DIGEST_OPTIONS: Final = {
    "calibration_runtime_manifest_file_sha256": "--expected-runtime-manifest-sha256",
    "model_file_manifest_file_sha256": "--expected-model-file-manifest-sha256",
    "parquet_materialization_manifest_file_sha256": (
        "--expected-parquet-materialization-manifest-sha256"
    ),
}
_REQUIRED_OPTIONS: Final = frozenset(
    {
        "--frozen-identity",
        "--stage-a-calibration-binding",
        "--repository-root",
        "--source-commit",
        "--identity-commit",
        "--model-root",
        "--cache-root",
        "--input-bundle-root",
        "--ruler-root",
        "--output-dir",
        *_BOUND_ARTIFACT_OPTIONS.values(),
        *_EXPECTED_DIGEST_OPTIONS.values(),
    }
)


class SealedStageALaunchError(RuntimeError):
    """Raised before the protected Stage-A runner can be entered."""


def _canonical_json_bytes(value: object) -> bytes:
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
        raise SealedStageALaunchError("value is not canonical JSON data") from error


def _pretty_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SealedStageALaunchError("value is not pretty JSON data") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SealedStageALaunchError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _strict_json(data: bytes, *, context: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SealedStageALaunchError(f"{context} contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SealedStageALaunchError(f"{context} contains a non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedStageALaunchError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SealedStageALaunchError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise SealedStageALaunchError(f"{context} fields differ from the frozen schema")


def _relative_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SealedStageALaunchError(f"{context} is not a canonical path")
    if any(character in value for character in ("\\", "\0", "\n", "\r", ":")):
        raise SealedStageALaunchError(f"{context} is unsafe")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SealedStageALaunchError(f"{context} is not a canonical relative path")
    return value


def _extract_options(arguments: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item in _REQUIRED_OPTIONS:
            if (
                item in result
                or index + 1 >= len(arguments)
                or arguments[index + 1].startswith("--")
            ):
                raise SealedStageALaunchError(f"runner option is duplicated or incomplete: {item}")
            result[item] = arguments[index + 1]
            index += 2
        elif item.startswith("--"):
            raise SealedStageALaunchError(f"runner option is not in the frozen CLI: {item}")
        else:
            index += 1
    if set(result) != _REQUIRED_OPTIONS:
        missing = sorted(_REQUIRED_OPTIONS - set(result))
        raise SealedStageALaunchError(f"runner arguments omit required inputs: {missing}")
    if not arguments or arguments[0] not in {
        "prepare-inputs",
        "preflight",
        "execute",
        "recover",
    }:
        raise SealedStageALaunchError(
            "runner mode must be prepare-inputs, preflight, execute, or recover"
        )
    return result


def _parse_identity(data: bytes) -> dict[str, str]:
    root = _strict_json(data, context="frozen Stage-A identity")
    _exact_fields(root, {"canonical_evidence_sha256", "evidence"}, context="identity")
    if _canonical_json_bytes(root) != data:
        raise SealedStageALaunchError("frozen Stage-A identity is not canonical JSON")
    evidence = root.get("evidence")
    if not isinstance(evidence, dict):
        raise SealedStageALaunchError("frozen Stage-A identity evidence is missing")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != IDENTITY_SCHEMA
        or evidence.get("identity_schema") != "recurquant.experiment013.identity-frozen.v5"
        or evidence.get("status") != "frozen"
        or evidence.get("phase") != "stage_a"
        or evidence.get("identity_only") is not True
        or evidence.get("promotion_required") is not False
    ):
        raise SealedStageALaunchError("identity is not a promoted Stage-A resolver-v5 artifact")
    promotion = evidence.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("explicit") is not True:
        raise SealedStageALaunchError("Stage-A identity lacks explicit promotion")
    recorded = _sha256(
        root.get("canonical_evidence_sha256"), context="identity canonical evidence SHA-256"
    )
    if recorded != _sha256_bytes(_canonical_json_bytes(evidence)):
        raise SealedStageALaunchError("identity canonical evidence SHA-256 drifted")
    bindings = evidence.get("execution_bindings")
    if not isinstance(bindings, dict):
        raise SealedStageALaunchError("identity execution bindings are missing")
    _exact_fields(bindings, set(_BOUND_ARTIFACT_OPTIONS), context="identity bindings")
    return {
        name: _sha256(bindings[name], context=f"identity binding {name}")
        for name in sorted(bindings)
    }


def _parse_source(data: bytes) -> dict[str, object]:
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
    if _pretty_json_bytes(root) != data:
        raise SealedStageALaunchError("repository source manifest is not canonical JSON")
    payload = dict(root)
    claimed = _sha256(payload.pop("canonical_manifest_sha256"), context="source self-hash")
    if claimed != _sha256_bytes(_pretty_json_bytes(payload)):
        raise SealedStageALaunchError("repository source manifest self-hash drifted")
    if (
        root["schema"] != "recurquant.experiment013.source-manifest.v2"
        or root["profile"] != "experiment-013-static-q468-frozen-source-v2"
        or root["object_format"] != "sha1"
    ):
        raise SealedStageALaunchError("repository source manifest profile drifted")
    raw_git = root["git_executable"]
    if not isinstance(raw_git, dict):
        raise SealedStageALaunchError("repository source Git executable record is missing")
    _exact_fields(raw_git, {"sha256", "size_bytes"}, context="source Git executable")
    if (
        isinstance(raw_git["size_bytes"], bool)
        or not isinstance(raw_git["size_bytes"], int)
        or raw_git["size_bytes"] <= 0
    ):
        raise SealedStageALaunchError("repository source Git executable size is invalid")
    git_executable = {
        "sha256": _sha256(raw_git["sha256"], context="source Git executable SHA-256"),
        "size_bytes": raw_git["size_bytes"],
    }
    raw_paths = root.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise SealedStageALaunchError("repository source manifest has no paths")
    paths: list[dict[str, str]] = []
    for index, entry in enumerate(raw_paths):
        if not isinstance(entry, dict):
            raise SealedStageALaunchError(f"source paths[{index}] must be an object")
        _exact_fields(
            entry,
            {
                "git_blob_oid",
                "index_blob_oid",
                "mode",
                "path",
                "raw_sha256",
                "worktree_blob_oid",
            },
            context=f"source paths[{index}]",
        )
        path = _relative_path(entry["path"], context=f"source paths[{index}].path")
        paths.append(
            {
                "path": path,
                "raw_sha256": _sha256(
                    entry["raw_sha256"], context=f"source paths[{index}].raw_sha256"
                ),
            }
        )
    names = [entry["path"] for entry in paths]
    if names != sorted(names) or len({name.casefold() for name in names}) != len(names):
        raise SealedStageALaunchError("source path inventory is not unique and sorted")
    required = {RUNNER_SOURCE_PATH, LAUNCHER_SOURCE_PATH, CALIBRATION_LAUNCHER_SOURCE_PATH}
    if not set(names) >= required:
        raise SealedStageALaunchError("source manifest omits a sealed Stage-A launch path")
    return {"document": root, "git_executable": git_executable, "paths": paths}


def _verify_source(source: Mapping[str, object], repository_root: Path) -> Path:
    root = repository_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SealedStageALaunchError("repository root is not a stable directory")
    runner: Path | None = None
    for entry in source["paths"]:  # type: ignore[union-attr]
        relative = str(entry["path"])
        path = (root / PurePosixPath(relative)).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SealedStageALaunchError("source path escapes repository root") from error
        if path.is_symlink() or not path.is_file():
            raise SealedStageALaunchError(f"source path is not a regular file: {relative}")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise SealedStageALaunchError(f"source path changed while read: {relative}")
        if _sha256_bytes(data) != entry["raw_sha256"]:
            raise SealedStageALaunchError(f"source bytes drifted: {relative}")
        if relative == RUNNER_SOURCE_PATH:
            runner = path
    if runner is None:
        raise SealedStageALaunchError("source manifest omits Stage-A runner")
    return runner


def _load_calibration_launcher(
    repository_root: Path,
    source: Mapping[str, object],
) -> ModuleType:
    entries = {str(item["path"]): item for item in source["paths"]}  # type: ignore[index]
    path = (repository_root / PurePosixPath(CALIBRATION_LAUNCHER_SOURCE_PATH)).resolve(strict=True)
    if _sha256_bytes(path.read_bytes()) != entries[CALIBRATION_LAUNCHER_SOURCE_PATH]["raw_sha256"]:
        raise SealedStageALaunchError("calibration launcher source bytes drifted")
    if CALIBRATION_LAUNCHER_MODULE_NAME in sys.modules:
        raise SealedStageALaunchError("calibration launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(CALIBRATION_LAUNCHER_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise SealedStageALaunchError("cannot load authenticated calibration launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[CALIBRATION_LAUNCHER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(CALIBRATION_LAUNCHER_MODULE_NAME, None)
        raise
    return module


def _stage_a_bootstrap(calibration_launcher: ModuleType) -> str:
    source = getattr(calibration_launcher, "SEALED_BOOTSTRAP", None)
    if not isinstance(source, str):
        raise SealedStageALaunchError("authenticated calibration launcher has no bootstrap")
    stage_a_options = """_stage_a_options = {
    "--cache-root",
    "--expected-model-file-manifest-sha256",
    "--expected-parquet-materialization-manifest-sha256",
    "--expected-runtime-manifest-sha256",
    "--frozen-identity",
    "--identity-commit",
    "--input-bundle-root",
    "--model-file-manifest",
    "--model-root",
    "--output-dir",
    "--parquet-materialization-manifest",
    "--repository-root",
    "--repository-source-manifest",
    "--ruler-root",
    "--runtime-manifest",
    "--source-commit",
    "--stage-a-calibration-binding",
}

def _options(arguments):
    if not arguments or arguments[0] not in {
            "prepare-inputs", "preflight", "execute", "recover"}:
        _fail("Stage-A runner mode drifted")
    result = {}
    index = 1
    while index < len(arguments):
        item = arguments[index]
        if item not in _stage_a_options:
            _fail("runner option is not in the frozen Stage-A CLI: " + item)
        if (item in result or index + 1 >= len(arguments)
                or arguments[index + 1].startswith("--")):
            _fail("runner option is duplicated or incomplete: " + item)
        result[item] = arguments[index + 1]
        index += 2
    if set(result) != _stage_a_options:
        _fail("runner arguments omit required Stage-A inputs")
    return result
"""
    stage_a_identity = """def _identity(data):
    root = _json(data, "frozen Stage-A identity")
    _fields(root, {"canonical_evidence_sha256", "evidence"}, "frozen Stage-A identity")
    if _canonical(root) != data:
        _fail("frozen Stage-A identity is not canonical JSON")
    evidence = root["evidence"]
    promotion = evidence.get("promotion") if isinstance(evidence, dict) else None
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 5
            or evidence.get("identity_schema") != "recurquant.experiment013.identity-frozen.v5"
            or evidence.get("status") != "frozen" or evidence.get("phase") != "stage_a"
            or evidence.get("identity_only") is not True
            or evidence.get("promotion_required") is not False
            or not isinstance(promotion, dict) or promotion.get("explicit") is not True):
        _fail("frozen Stage-A identity state, schema, or promotion drifted")
    if _digest(root["canonical_evidence_sha256"], "identity evidence hash") != _h.sha256(
        _canonical(evidence)
    ).hexdigest():
        _fail("frozen Stage-A identity evidence hash drifted")
    bindings = evidence.get("execution_bindings")
    if not isinstance(bindings, dict):
        _fail("frozen Stage-A identity bindings are missing")
    _fields(bindings, set(_binding_options), "identity bindings")
    return {key: _digest(value, "identity binding") for key, value in bindings.items()}
"""
    environment_anchor = (
        "_environment = {key.upper(): value for key, value in _o.environ.items()}\n"
    )
    stage_a_environment = """_stage_a_mode = _runner_args[0] if _runner_args else None
_stage_a_offline_values = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_stage_a_offline = _stage_a_mode != "prepare-inputs"
if _stage_a_offline:
    _expected_environment.update(_stage_a_offline_values)
_environment = {key.upper(): value for key, value in _o.environ.items()}
if _stage_a_offline:
    _forbidden_network_events = {
        "socket.connect", "socket.connect_ex", "socket.getaddrinfo",
        "socket.gethostbyaddr", "socket.gethostbyname", "socket.gethostbyname_ex",
        "socket.getnameinfo", "socket.sendto",
    }
    def _reject_network(event, _arguments):
        if event in _forbidden_network_events:
            raise RuntimeError("sealed offline Stage-A child forbids network access: " + event)
    _s.addaudithook(_reject_network)
"""

    def replace_exact(value: str, old: str, new: str, *, count: int) -> str:
        if value.count(old) != count:
            raise SealedStageALaunchError(
                "calibration bootstrap transformation anchor is missing or duplicated"
            )
        return value.replace(old, new)

    def replace_section(value: str, start: str, end: str, replacement: str) -> str:
        if value.count(start) != 1 or value.count(end) != 1:
            raise SealedStageALaunchError(
                "calibration bootstrap section anchor is missing or duplicated"
            )
        prefix, remainder = value.split(start, 1)
        _discarded, suffix = remainder.split(end, 1)
        return prefix + replacement + "\n" + end + suffix

    result = source
    result = replace_exact(
        result,
        '"_recurquant_experiment013_sealed_runner"',
        f'"{RUNNER_MODULE_NAME}"',
        count=3,
    )
    result = replace_exact(
        result,
        "scripts/run_static_q468_calibration.py",
        RUNNER_SOURCE_PATH,
        count=4,
    )
    result = replace_section(result, "_smoke_options = {", "def _fail(message):", "")
    result = replace_section(
        result,
        "def _options(arguments):",
        "def _identity(data):",
        stage_a_options,
    )
    result = replace_section(
        result,
        "def _identity(data):",
        "def _source(data):",
        stage_a_identity,
    )
    result = replace_exact(
        result,
        environment_anchor,
        stage_a_environment,
        count=1,
    )
    result = replace_exact(result, "_smoke(_runner_options)\n", "", count=1)
    forbidden = (
        "scripts/run_static_q468_calibration.py",
        "--fisher-h1-smoke",
        "--prior-fisher-h1-smoke-report",
        "--prior-fisher-h1-smoke-complete-marker",
        "full calibration",
    )
    if any(value in result for value in forbidden):
        raise SealedStageALaunchError("Stage-A bootstrap retained calibration-only runner logic")
    return result


def _split_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    positions = [index for index, value in enumerate(argv) if value == "--"]
    if len(positions) != 1:
        raise SealedStageALaunchError("launcher requires exactly one -- separator")
    position = positions[0]
    host = list(argv[:position])
    runner = list(argv[position + 1 :])
    if not runner:
        raise SealedStageALaunchError("launcher requires exact runner arguments after --")
    return host, runner


def _parse_package_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("package roots use NAME=PATH")
    name, path = value.split("=", 1)
    if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None or not path:
        raise argparse.ArgumentTypeError("package root name or path is invalid")
    return name, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runtime-root", required=True, type=Path)
    parser.add_argument("--git-executable", required=True, type=Path)
    parser.add_argument("--package-root", required=True, action="append", type=_parse_package_root)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    return parser


def _sealed_argv(
    *,
    interpreter: Path,
    bootstrap: bytes,
    runtime_manifest: Path,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    git_executable: Path,
    pycache_prefix: Path,
    scratch_directory: Path,
    runner_arguments: Sequence[str],
) -> list[str]:
    roots = json.dumps(
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
        _authenticated_stdin_loader(bootstrap),
        str(runtime_manifest),
        str(base_runtime_root),
        roots,
        str(pycache_prefix),
        str(git_executable),
        str(scratch_directory),
        *runner_arguments,
    ]


def _sealed_environment(
    *,
    scratch_directory: Path,
    dataset_cache_root: Path,
    offline: bool,
) -> dict[str, str]:
    scratch = scratch_directory.resolve(strict=True)
    cache = dataset_cache_root.resolve(strict=True)
    private_home = scratch / "private-home"
    xdg_cache = scratch / "xdg-cache"
    hf_home = scratch / "huggingface"
    hf_hub_cache = hf_home / "hub"
    hf_assets_cache = hf_home / "assets"
    hf_xet_cache = hf_home / "xet"
    hf_modules_cache = hf_home / "modules"
    hf_token_path = hf_home / "token"
    transformers_cache = scratch / "transformers"
    torch_home = scratch / "torch"
    datasets_cache = cache / "datasets"
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
            "PYTORCH_KERNEL_CACHE_PATH": str(torch_home / "kernels"),
            "TORCH_EXTENSIONS_DIR": str(torch_home / "extensions"),
            "TORCHINDUCTOR_CACHE_DIR": str(torch_home / "inductor"),
            "TRITON_CACHE_DIR": str(torch_home / "triton"),
            "HF_DATASETS_CACHE": str(datasets_cache),
            "HF_DATASETS_DOWNLOADED_DATASETS_PATH": str(datasets_cache / "downloads"),
            "HF_DATASETS_EXTRACTED_DATASETS_PATH": str(datasets_cache / "downloads" / "extracted"),
            "DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_UPDATE_CHECK": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    if offline:
        environment.update(
            {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return environment


def _authenticated_stdin_loader(payload: bytes) -> str:
    """Return a small ``-c`` loader bound to the exact Stage-A bootstrap."""
    digest = _sha256_bytes(payload)
    size = len(payload)
    return (
        "import hashlib as _h,sys as _s\n"
        f"_p=_s.stdin.buffer.read({size + 1})\n"
        f"if len(_p)!={size} or _h.sha256(_p).hexdigest()!='{digest}':"
        " raise RuntimeError('sealed Stage-A bootstrap stdin authentication failed')\n"
        "exec(compile(_p,'<recurquant-stage-a-bootstrap>','exec',dont_inherit=True))"
    )


def _verify_bound_inputs(
    options: Mapping[str, str],
    *,
    runtime_manifest_path: Path,
) -> tuple[dict[str, str], dict[str, object], Path]:
    identity_bytes = Path(options["--frozen-identity"]).read_bytes()
    bindings = _parse_identity(identity_bytes)
    for binding, option in _BOUND_ARTIFACT_OPTIONS.items():
        try:
            data = Path(options[option]).read_bytes()
        except OSError as error:
            raise SealedStageALaunchError(f"bound artifact is unavailable: {option}") from error
        if _sha256_bytes(data) != bindings[binding]:
            raise SealedStageALaunchError(f"identity binding mismatch: {option}")
    for binding, option in _EXPECTED_DIGEST_OPTIONS.items():
        if _sha256(options[option], context=f"runner option {option}") != bindings[binding]:
            raise SealedStageALaunchError(f"runner digest binding mismatch: {option}")
    if runtime_manifest_path.resolve(strict=True) != Path(options["--runtime-manifest"]).resolve(
        strict=True
    ):
        raise SealedStageALaunchError("host and runner runtime-manifest paths differ")
    source_bytes = Path(options["--repository-source-manifest"]).read_bytes()
    source = _parse_source(source_bytes)
    runner_path = _verify_source(source, Path(options["--repository-root"]))
    return bindings, source, runner_path


def _run_sealed_child(
    *,
    calibration_launcher: ModuleType,
    bootstrap: bytes,
    runtime_manifest_path: Path,
    runtime_manifest: Mapping[str, object],
    interpreter: Path,
    base_runtime_root: Path,
    package_roots: Mapping[str, Path],
    git_executable: Path,
    source: Mapping[str, object],
    options: Mapping[str, str],
    runner_arguments: Sequence[str],
    dataset_cache_root: Path,
    offline: bool,
) -> int:
    pycache: Path | None = None
    scratch: Path | None = None
    pycache_identity: tuple[int, int, int] | None = None
    scratch_identity: tuple[int, int, int] | None = None
    dataset_cache_identity: tuple[tuple[str, int, int, int], ...] | None = None
    completed: subprocess.CompletedProcess[bytes] | None = None
    primary_error: BaseException | None = None
    secondary_failures: list[tuple[str, BaseException]] = []
    try:
        pycache = Path(tempfile.mkdtemp(prefix="recurquant-exp013-stage-a-pycache-"))
        pycache_identity = calibration_launcher._temporary_directory_identity(
            pycache,
            context="Stage-A pycache prefix",
        )
        calibration_launcher._verify_empty_pycache(pycache)
        scratch = Path(tempfile.mkdtemp(prefix="recurquant-exp013-stage-a-scratch-"))
        scratch_identity = calibration_launcher._temporary_directory_identity(
            scratch,
            context="Stage-A sealed scratch directory",
        )
        calibration_launcher._verify_empty_scratch(scratch)
        confirmed_cache_root = calibration_launcher._verified_dataset_cache_root(
            dataset_cache_root,
            runtime_roots=(base_runtime_root, *package_roots.values(), pycache, scratch),
        )
        dataset_cache_identity = calibration_launcher._non_link_directory_identity_chain(
            confirmed_cache_root,
            context="dataset cache root",
        )
        command = _sealed_argv(
            interpreter=interpreter,
            bootstrap=bootstrap,
            runtime_manifest=runtime_manifest_path,
            base_runtime_root=base_runtime_root,
            package_roots=package_roots,
            git_executable=git_executable,
            pycache_prefix=pycache,
            scratch_directory=scratch,
            runner_arguments=runner_arguments,
        )
        completed = subprocess.run(
            command,
            check=False,
            cwd=scratch,
            env=_sealed_environment(
                scratch_directory=scratch,
                dataset_cache_root=dataset_cache_root,
                offline=offline,
            ),
            input=bootstrap,
        )
        try:
            calibration_launcher._verify_empty_pycache(pycache)
        except Exception as error:
            secondary_failures.append(("Stage-A pycache postcondition", error))
        try:
            if (
                calibration_launcher._temporary_directory_identity(
                    scratch,
                    context="Stage-A sealed scratch directory",
                )
                != scratch_identity
            ):
                raise SealedStageALaunchError(
                    "Stage-A sealed scratch directory identity changed during execution"
                )
            calibration_launcher._assert_owned_temporary_tree_has_no_reparse(
                scratch,
                context="Stage-A sealed scratch directory",
            )
            calibration_launcher._verify_empty_scratch(scratch)
        except Exception as error:
            secondary_failures.append(("Stage-A scratch containment postcondition", error))
        try:
            repeated_cache_root = calibration_launcher._verified_dataset_cache_root(
                dataset_cache_root,
                runtime_roots=(base_runtime_root, *package_roots.values(), pycache, scratch),
            )
            if (
                calibration_launcher._non_link_directory_identity_chain(
                    repeated_cache_root,
                    context="dataset cache root",
                )
                != dataset_cache_identity
            ):
                raise SealedStageALaunchError(
                    "Stage-A dataset cache root identity changed during execution"
                )
        except Exception as error:
            secondary_failures.append(("Stage-A dataset cache root reauthentication", error))
        try:
            _bindings, repeated_source, _runner = _verify_bound_inputs(
                options,
                runtime_manifest_path=runtime_manifest_path,
            )
            if repeated_source["git_executable"] != source["git_executable"]:
                raise SealedStageALaunchError(
                    "source Git executable binding changed during Stage-A execution"
                )
        except Exception as error:
            secondary_failures.append(("Stage-A bound-input reauthentication", error))
        try:
            calibration_launcher._verify_runtime(
                runtime_manifest,
                base_runtime_root=base_runtime_root,
                package_roots=package_roots,
                git_executable_path=git_executable,
                require_current_process=False,
            )
            calibration_launcher._verified_dataset_cache_root(
                dataset_cache_root,
                runtime_roots=(base_runtime_root, *package_roots.values()),
            )
        except Exception as error:
            secondary_failures.append(("Stage-A runtime reauthentication", error))
        if completed.returncode == 0 and secondary_failures:
            failures = tuple(secondary_failures)
            secondary_failures.clear()
            raise calibration_launcher._postcondition_error(failures)
        return int(completed.returncode)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        for path, expected_identity, context in (
            (scratch, scratch_identity, "Stage-A sealed scratch directory"),
            (pycache, pycache_identity, "Stage-A pycache prefix"),
        ):
            if path is None or expected_identity is None:
                continue
            try:
                calibration_launcher._cleanup_owned_temporary_directory(
                    path,
                    expected_identity=expected_identity,
                    context=context,
                )
            except Exception as error:
                secondary_failures.append((f"{context} cleanup", error))
        calibration_launcher._surface_secondary_failures(
            secondary_failures,
            primary_error=primary_error,
            child_returncode=None if completed is None else int(completed.returncode),
        )


def launch(argv: Sequence[str]) -> int:
    if list(argv) in (["-h"], ["--help"]):
        _parser().print_help()
        print("\nAppend -- followed by exact screen_static_q468_stage_a.py arguments.")
        return 0
    host_arguments, runner_arguments = _split_arguments(argv)
    args = _parser().parse_args(host_arguments)
    options = _extract_options(runner_arguments)
    package_roots: dict[str, Path] = {}
    for name, path in args.package_root:
        if name in package_roots or name == BASE_RUNTIME_ROOT_NAME:
            raise SealedStageALaunchError(f"duplicate or reserved package root: {name}")
        package_roots[name] = path
    _bindings, source, _runner = _verify_bound_inputs(
        options, runtime_manifest_path=args.runtime_manifest
    )
    repository_root = Path(options["--repository-root"])
    calibration_launcher = _load_calibration_launcher(repository_root, source)
    try:
        runtime_bytes = args.runtime_manifest.read_bytes()
        runtime_manifest = calibration_launcher._parse_runtime_manifest(runtime_bytes)
        if source["git_executable"] != {
            "sha256": runtime_manifest["git_executable"]["sha256"],
            "size_bytes": runtime_manifest["git_executable"]["size_bytes"],
        }:
            raise SealedStageALaunchError(
                "source and runtime manifests bind different Git executable bytes"
            )
        base, packages, _import_paths, interpreter, git_executable = (
            calibration_launcher._verify_runtime(
                runtime_manifest,
                base_runtime_root=args.base_runtime_root,
                package_roots=package_roots,
                git_executable_path=args.git_executable,
                require_current_process=False,
            )
        )
        dataset_cache_root = calibration_launcher._verified_dataset_cache_root(
            Path(options["--cache-root"]),
            runtime_roots=(base, *packages.values()),
        )
        bootstrap = _stage_a_bootstrap(calibration_launcher).encode("utf-8")
        runtime_manifest_path = args.runtime_manifest.resolve(strict=True)
        mode = runner_arguments[0]
        runs: list[tuple[list[str], bool]] = []
        if mode in {"execute", "preflight"}:
            runs.append((["prepare-inputs", *runner_arguments[1:]], False))
        runs.append((list(runner_arguments), mode != "prepare-inputs"))
        for index, (child_arguments, offline) in enumerate(runs):
            return_code = _run_sealed_child(
                calibration_launcher=calibration_launcher,
                bootstrap=bootstrap,
                runtime_manifest_path=runtime_manifest_path,
                runtime_manifest=runtime_manifest,
                interpreter=interpreter,
                base_runtime_root=base,
                package_roots=packages,
                git_executable=git_executable,
                source=source,
                options=options,
                runner_arguments=child_arguments,
                dataset_cache_root=dataset_cache_root,
                offline=offline,
            )
            if return_code != 0 or index == len(runs) - 1:
                return return_code
        raise AssertionError("Stage-A child execution schedule was empty")
    finally:
        sys.modules.pop(CALIBRATION_LAUNCHER_MODULE_NAME, None)


def main(argv: Sequence[str] | None = None) -> int:
    return launch(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
