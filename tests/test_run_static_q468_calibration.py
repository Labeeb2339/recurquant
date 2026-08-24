from __future__ import annotations

import builtins
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_static_q468_calibration.py"
API_SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "recurquant" / "experiment013_calibration_api.py"
)
API_SPEC = importlib.util.spec_from_file_location(
    "experiment013_calibration_api_for_tests", API_SCRIPT
)
assert API_SPEC is not None and API_SPEC.loader is not None
api = importlib.util.module_from_spec(API_SPEC)
sys.modules[API_SPEC.name] = api
API_SPEC.loader.exec_module(api)
RESOLVER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "resolve_static_q468_identity.py"
)
RESOLVER_SPEC = importlib.util.spec_from_file_location(
    "resolve_static_q468_identity_for_runner_tests",
    RESOLVER_SCRIPT,
)
assert RESOLVER_SPEC is not None and RESOLVER_SPEC.loader is not None
identity_resolver = importlib.util.module_from_spec(RESOLVER_SPEC)
sys.modules[RESOLVER_SPEC.name] = identity_resolver
RESOLVER_SPEC.loader.exec_module(identity_resolver)
SPEC = importlib.util.spec_from_file_location("run_static_q468_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_sequence_manifest_digest(records: Sequence[Mapping[str, object]]) -> str:
    commitments = [
        {
            "identity_record_sha256": item["identity_record_sha256"],
            "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
            "sequence_length": item["sequence_length"],
            "sequence_token_ids_sha256": item["sequence_token_ids_sha256"],
            "target_token_ids_sha256": item["target_token_ids_sha256"],
        }
        for item in records
    ]
    return digest(runner.canonical_json_bytes(commitments))


def token_digest(values: Sequence[int]) -> str:
    return digest(runner.canonical_json_bytes(list(values)))


def current_head() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def authenticated_git_path() -> Path:
    return runner._authenticate_git_executable(None).path


def model_staging_path_contract_sha256(
    repository_root: Path,
    hub_cache_root: Path,
    output_root: Path,
) -> str:
    return runner._model_staging_path_contract_sha256(
        runner._validate_model_staging_roots(
            repository_root=repository_root,
            hub_cache_root=hub_cache_root,
            output_root=output_root,
        )
    )


def capture_provenance_gate_kwargs(root: Path) -> dict[str, object]:
    return {
        "capture_provenance_receipt_path": root / "capture-provenance.json",
        "expected_capture_provenance_receipt_sha256": "5" * 64,
        "runtime_manifest_path": root / "runtime.json",
        "expected_runtime_manifest_sha256": "8" * 64,
    }


def capture_provenance_gate_cli(root: Path) -> list[str]:
    return [
        "--capture-provenance-receipt",
        str(root / "capture-provenance.json"),
        "--expected-capture-provenance-receipt-sha256",
        "5" * 64,
        "--runtime-manifest",
        str(root / "runtime.json"),
        "--expected-runtime-manifest-sha256",
        "8" * 64,
    ]


def ruler_receipt_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    for filename in runner.RULER_RECEIPT_DIRECTORY_FILENAMES:
        (path / filename).write_bytes(b"fixture\n")
    return path


def official_cli_arguments(tmp_path: Path, *, ruler_receipts: Path) -> list[str]:
    return [
        "--frozen-identity",
        str(tmp_path / "identity.json"),
        "--capture-provenance-receipt",
        str(tmp_path / "capture-provenance.json"),
        "--expected-capture-provenance-receipt-sha256",
        "5" * 64,
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--model-file-manifest",
        str(tmp_path / "model.json"),
        "--expected-model-file-manifest-sha256",
        "1" * 64,
        "--parquet-materialization-manifest",
        str(tmp_path / "parquet.json"),
        "--expected-parquet-materialization-manifest-sha256",
        "2" * 64,
        "--runtime-manifest",
        str(tmp_path / "runtime.json"),
        "--expected-runtime-manifest-sha256",
        "3" * 64,
        "--model-root",
        str(tmp_path / "model-root"),
        "--cache-root",
        str(tmp_path / "cache-root"),
        "--ruler-receipt-dir",
        str(ruler_receipts),
        "--repository-root",
        str(tmp_path / "repository"),
        "--source-commit",
        "4" * 40,
        "--output-dir",
        str(tmp_path / "output"),
        "--fisher-h1-smoke",
    ]


def sealed_capture_cli_arguments(tmp_path: Path, *, ruler_receipts: Path) -> list[str]:
    return [
        "capture-calibration-identity",
        "--repository-root",
        str(tmp_path / "repository"),
        "--source-commit",
        "4" * 40,
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--expected-repository-source-manifest-sha256",
        "1" * 64,
        "--runtime-manifest",
        str(tmp_path / "runtime.json"),
        "--expected-runtime-manifest-sha256",
        "2" * 64,
        "--model-file-manifest",
        str(tmp_path / "model.json"),
        "--expected-model-file-manifest-sha256",
        "3" * 64,
        "--parquet-materialization-manifest",
        str(tmp_path / "parquet.json"),
        "--expected-parquet-materialization-manifest-sha256",
        "4" * 64,
        "--cache-root",
        str(tmp_path / "cache"),
        "--ruler-receipt-dir",
        str(ruler_receipts),
        "--output",
        str(tmp_path / "identity-input.json"),
        "--capture-provenance-receipt-output",
        str(tmp_path / "capture-provenance.json"),
    ]


@contextmanager
def isolated_sealed_capture_modules() -> Iterator[None]:
    module_names = (
        "recurquant",
        runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE,
        runner.IDENTITY_RESOLVER_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_RUNNER_MODULE,
    )
    previous = {name: sys.modules[name] for name in module_names if name in sys.modules}
    for name in module_names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def fisher_boundary_contract(
    token_ids: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    return identity_resolver.build_fisher_boundary_contract(token_ids)


def record(
    token_ids: tuple[int, ...] = (1, 2, 3),
    *,
    canonical_id: str = "item-1",
) -> dict[str, object]:
    prompt_stop = max(1, len(token_ids) - 1)
    item: dict[str, object] = {
        "anchor_manifest_sha256": "e" * 64,
        "canonical_id": canonical_id,
        "config": "default",
        "family": "mbpp",
        "formatted_content_sha256": "b" * 64,
        "fisher_boundary": fisher_boundary_contract(token_ids),
        "generator_receipt_sha256": None,
        "prompt_token_ids_sha256": token_digest(token_ids[:prompt_stop]),
        "ruler_category": None,
        "seed": None,
        "selection_rank": 0,
        "selection_sha256": "d" * 64,
        "configured_length": None,
        "sequence_length": len(token_ids),
        "sequence_token_ids_sha256": token_digest(token_ids),
        "source_content_sha256": "a" * 64,
        "target_token_ids_sha256": token_digest(token_ids[prompt_stop:]),
        "token_span": {
            "prefill_start": 0,
            "prefill_stop": prompt_stop,
            "scored_start": prompt_stop,
            "scored_stop": len(token_ids),
            "cache_exposed_start": len(token_ids),
            "cache_exposed_stop": len(token_ids),
        },
        "tokenizer_manifest_sha256": "c" * 64,
    }
    item["identity_record_sha256"] = identity_resolver.identity_record_sha256(item)
    return item


def materialized(item: Mapping[str, object], token_ids: tuple[int, ...]) -> Any:
    return api.AuthenticatedSequence(
        token_ids=token_ids,
        source_content_sha256=item["source_content_sha256"],
        formatted_content_sha256=item["formatted_content_sha256"],
        generator_receipt_sha256=item["generator_receipt_sha256"],
        tokenizer_manifest_sha256=item["tokenizer_manifest_sha256"],
    )


def identity(
    records: Sequence[dict[str, object]],
    *,
    source_manifest_sha256: str = "7" * 64,
    runtime_manifest_sha256: str = "8" * 64,
    model_manifest_sha256: str = "9" * 64,
    parquet_manifest_sha256: str = "a" * 64,
) -> Any:
    return runner.FrozenCalibrationIdentity(
        file_sha256="d" * 64,
        canonical_evidence_sha256="e" * 64,
        records=tuple(records),
        assignment=(),
        assignment_sha256="f" * 64,
        tokenizer_manifest_sha256="c" * 64,
        identity_input_manifest_sha256="1" * 64,
        repository_source_manifest_file_sha256=source_manifest_sha256,
        runtime_manifest_file_sha256=runtime_manifest_sha256,
        model_file_manifest_file_sha256=model_manifest_sha256,
        parquet_materialization_manifest_file_sha256=parquet_manifest_sha256,
        model_id="example/model",
        model_revision="2" * 40,
        transformers_version="5.14.1",
        artifact_bytes=b"frozen-identity",
    )


def bootstrap_identity_bytes(
    *,
    source_manifest_sha256: str,
    runtime_manifest_sha256: str,
    model_manifest_sha256: str,
    parquet_manifest_sha256: str,
    identity_input_manifest_sha256: str = "1" * 64,
) -> bytes:
    evidence = {
        "execution_bindings": {
            "calibration_runtime_manifest_file_sha256": runtime_manifest_sha256,
            "model_file_manifest_file_sha256": model_manifest_sha256,
            "parquet_materialization_manifest_file_sha256": parquet_manifest_sha256,
            "repository_source_manifest_file_sha256": source_manifest_sha256,
        },
        "identity_only": True,
        "phase": "calibration",
        "promotion_required": False,
        "schema_version": runner.FROZEN_IDENTITY_SCHEMA_VERSION,
        "source_manifest_sha256": identity_input_manifest_sha256,
        "status": "frozen",
    }
    return runner.canonical_json_bytes(
        {
            "canonical_evidence_sha256": digest(runner.canonical_json_bytes(evidence)),
            "evidence": evidence,
        }
    )


def model_manifest_bytes(files: Mapping[str, bytes]) -> bytes:
    file_records = []
    tree_records = []
    for index, (name, content) in enumerate(sorted(files.items())):
        is_weight = name.endswith(".safetensors")
        content_sha256 = digest(content)
        git_blob_oid = hashlib.sha1(
            f"blob {len(content)}\0".encode("ascii") + content,
            usedforsecurity=False,
        ).hexdigest()
        item = {
            "git_blob_oid": f"{index + 1:040x}" if is_weight else git_blob_oid,
            "lfs_sha256": content_sha256 if is_weight else None,
            "lfs_size_bytes": len(content) if is_weight else None,
            "name": name,
            "sha256": content_sha256 if is_weight else None,
            "size_bytes": len(content),
        }
        file_records.append(item)
        tree_records.append(
            {
                "git_blob_oid": item["git_blob_oid"],
                "lfs_sha256": item["lfs_sha256"],
                "lfs_size_bytes": item["lfs_size_bytes"],
                "name": name,
            }
        )
    payload = {
        "artifact_kind": runner.MODEL_FILE_MANIFEST_KIND,
        "files": file_records,
        "hub_tree_manifest_sha256": digest(runner.canonical_json_bytes(tree_records)),
        "metadata_derivation": runner.MODEL_FILE_MANIFEST_DERIVATION,
        "model_id": "example/model",
        "revision": "2" * 40,
        "schema_version": runner.MODEL_FILE_MANIFEST_SCHEMA,
        "selection_profile": runner.MODEL_FILE_SELECTION_PROFILE,
        "transformers_version": "5.14.1",
    }
    return runner.canonical_json_bytes(payload)


def staged_model_files() -> dict[str, bytes]:
    return {
        "config.json": b"{}",
        "model.safetensors": b"authenticated-weight-placeholder",
        "model.safetensors.index.json": b"{}",
    }


def model_staging_authorization(
    files: Mapping[str, bytes] | None = None,
) -> Any:
    payload = model_manifest_bytes(files or staged_model_files())
    manifest = runner.parse_model_file_manifest(payload)
    frozen = identity((), model_manifest_sha256=digest(payload))
    return runner.ModelStagingAuthorization(
        identity=frozen,
        model_manifest=manifest,
        frozen_identity_file_sha256="d" * 64,
        capture_provenance_receipt_file_sha256="5" * 64,
        identity_commit="3" * 40,
        source_commit="1" * 40,
    )


def frozen_identity_source_authorization() -> Any:
    frozen = identity(())
    return runner.FrozenIdentitySourceAuthorization(
        identity=frozen,
        bindings=runner.BootstrapIdentityBindings(
            repository_source_manifest_file_sha256=(frozen.repository_source_manifest_file_sha256),
            runtime_manifest_file_sha256=frozen.runtime_manifest_file_sha256,
            model_file_manifest_file_sha256=frozen.model_file_manifest_file_sha256,
            parquet_materialization_manifest_file_sha256=(
                frozen.parquet_materialization_manifest_file_sha256
            ),
            identity_input_manifest_sha256=frozen.identity_input_manifest_sha256,
        ),
        identity_bytes=b"frozen-identity",
        frozen_identity_file_sha256="d" * 64,
        source_commit="1" * 40,
    )


def runtime_manifest_bytes() -> bytes:
    git_executable = runner._authenticate_git_executable(None)
    interpreter_sha256, interpreter_size = runner._stream_file_sha256(
        Path(sys.executable).resolve(strict=True)
    )
    machine = runner._current_machine_identity()
    base_files = [
        {
            "path": "Lib/os.py",
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        {
            "path": "python.exe",
            "sha256": interpreter_sha256,
            "size_bytes": interpreter_size,
        },
    ]
    package_files = [
        {
            "path": "Lib/site-packages/transformers-5.14.1.dist-info/RECORD",
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
        {
            "path": "Lib/site-packages/transformers/__init__.py",
            "sha256": "d" * 64,
            "size_bytes": 1,
        },
    ]
    payload = {
        "artifact_kind": runner.RUNTIME_MANIFEST_KIND,
        "base_runtime_root": runner.BASE_RUNTIME_ROOT_NAME,
        "base_sys_path": ["Lib"],
        "distributions": [
            {
                "files": [item["path"] for item in package_files],
                "name": "transformers",
                "package_root": "packages",
                "version": "5.14.1",
            }
        ],
        "interpreter": {
            "relative_path": "python.exe",
            "root": runner.BASE_RUNTIME_ROOT_NAME,
            "sha256": interpreter_sha256,
            "size_bytes": interpreter_size,
        },
        "git_executable": {
            "absolute_path_sha256": git_executable.absolute_path_sha256,
            "sha256": git_executable.sha256,
            "size_bytes": git_executable.size_bytes,
        },
        "launch_policy": dict(runner.SEALED_LAUNCH_POLICY),
        "machine": dict(
            zip(
                ("system", "architecture", "machine", "byteorder", "pointer_bits"),
                machine,
                strict=True,
            )
        ),
        "package_roots": [{"import_path": "Lib/site-packages", "name": "packages"}],
        "python": {
            "abi_flags": getattr(sys, "abiflags", ""),
            "cache_tag": sys.implementation.cache_tag,
            "implementation": runner.platform.python_implementation(),
            "version": runner.platform.python_version(),
        },
        "runtime_trees": [
            {
                "files": base_files,
                "kind": "base-runtime",
                "name": runner.BASE_RUNTIME_ROOT_NAME,
            },
            {
                "files": package_files,
                "kind": "packages",
                "name": "packages",
            },
        ],
        "schema_version": runner.RUNTIME_MANIFEST_SCHEMA,
    }
    return runner.canonical_json_bytes(payload)


def capture_provenance_runtime_manifest_bytes() -> tuple[bytes, list[dict[str, object]]]:
    git_executable = runner._authenticate_git_executable(None)
    interpreter_sha256, interpreter_size = runner._stream_file_sha256(
        Path(sys.executable).resolve(strict=True)
    )
    machine = runner._current_machine_identity()
    base_files = [
        {"path": "Lib/os.py", "sha256": "b" * 64, "size_bytes": 1},
        {
            "path": "python.exe",
            "sha256": interpreter_sha256,
            "size_bytes": interpreter_size,
        },
    ]
    package_files: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    origins: list[dict[str, object]] = []
    for index, module_name in enumerate(
        sorted(runner.CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS),
        start=1,
    ):
        distribution_name = runner.CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS[module_name]
        module_path = (
            "Lib/site-packages/six.py"
            if module_name == "six"
            else f"Lib/site-packages/{module_name}/__init__.py"
        )
        record_path = (
            f"Lib/site-packages/{distribution_name.replace('-', '_')}-1.0.dist-info/RECORD"
        )
        module_sha256 = f"{index:064x}"
        record_sha256 = f"{index + 20:064x}"
        package_files.extend(
            [
                {"path": module_path, "sha256": module_sha256, "size_bytes": index},
                {"path": record_path, "sha256": record_sha256, "size_bytes": index + 20},
            ]
        )
        distributions.append(
            {
                "files": sorted([module_path, record_path]),
                "name": distribution_name,
                "package_root": "packages",
                "version": "1.0",
            }
        )
        origins.append(
            {
                "distribution": distribution_name,
                "module": module_name,
                "package_root": "packages",
                "relative_path": module_path,
                "sha256": module_sha256,
                "size_bytes": index,
                "version": "1.0",
            }
        )
    package_files.sort(key=lambda item: str(item["path"]))
    distributions.sort(key=lambda item: str(item["name"]))
    payload = {
        "artifact_kind": runner.RUNTIME_MANIFEST_KIND,
        "base_runtime_root": runner.BASE_RUNTIME_ROOT_NAME,
        "base_sys_path": ["Lib"],
        "distributions": distributions,
        "interpreter": {
            "relative_path": "python.exe",
            "root": runner.BASE_RUNTIME_ROOT_NAME,
            "sha256": interpreter_sha256,
            "size_bytes": interpreter_size,
        },
        "git_executable": {
            "absolute_path_sha256": git_executable.absolute_path_sha256,
            "sha256": git_executable.sha256,
            "size_bytes": git_executable.size_bytes,
        },
        "launch_policy": dict(runner.SEALED_LAUNCH_POLICY),
        "machine": dict(
            zip(
                ("system", "architecture", "machine", "byteorder", "pointer_bits"),
                machine,
                strict=True,
            )
        ),
        "package_roots": [{"import_path": "Lib/site-packages", "name": "packages"}],
        "python": {
            "abi_flags": getattr(sys, "abiflags", ""),
            "cache_tag": sys.implementation.cache_tag,
            "implementation": runner.platform.python_implementation(),
            "version": runner.platform.python_version(),
        },
        "runtime_trees": [
            {"files": base_files, "kind": "base-runtime", "name": "base-runtime"},
            {"files": package_files, "kind": "packages", "name": "packages"},
        ],
        "schema_version": runner.RUNTIME_MANIFEST_SCHEMA,
    }
    return runner.canonical_json_bytes(payload), origins


def capture_provenance_receipt_document(
    *,
    origins: Sequence[Mapping[str, object]],
    bindings: Any,
    capture_source_sha256: str,
    identity_input_manifest_sha256: str,
    source_commit: str,
) -> dict[str, object]:
    return {
        "artifact_kind": runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND,
        "capture_source": {
            "path": runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
            "sha256": capture_source_sha256,
        },
        "capture_version": runner.CALIBRATION_IDENTITY_CAPTURE_VERSION,
        "critical_module_origins": origins,
        "excluded_runtime_modules": list(runner.CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES),
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
        "identity_input_file_sha256": identity_input_manifest_sha256,
        "phase": "calibration",
        "publication_contract": (
            runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT
        ),
        "runner_revision": runner.RUNNER_REVISION,
        "schema_version": runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA,
        "source_commit": source_commit,
        "status": runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS,
    }


def capture_provenance_receipt_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime_bytes, origins = capture_provenance_runtime_manifest_bytes()
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(runtime_bytes)
    capture_source_sha256 = "c" * 64
    source_bytes = runner.canonical_json_bytes(
        {
            "paths": [
                {
                    "path": runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
                    "raw_sha256": capture_source_sha256,
                }
            ]
        }
    )
    bindings = runner.BootstrapIdentityBindings(
        repository_source_manifest_file_sha256=digest(source_bytes),
        runtime_manifest_file_sha256=digest(runtime_bytes),
        model_file_manifest_file_sha256="9" * 64,
        parquet_materialization_manifest_file_sha256="a" * 64,
        identity_input_manifest_sha256="1" * 64,
    )
    receipt = capture_provenance_receipt_document(
        origins=origins,
        bindings=bindings,
        capture_source_sha256=capture_source_sha256,
        identity_input_manifest_sha256=bindings.identity_input_manifest_sha256,
        source_commit="1" * 40,
    )
    receipt_bytes = runner.canonical_json_bytes(receipt)
    receipt_path = tmp_path / "capture-provenance.json"
    receipt_path.write_bytes(receipt_bytes)
    arguments = {
        "receipt_path": receipt_path,
        "expected_receipt_sha256": digest(receipt_bytes),
        "runtime_manifest_path": runtime_path,
        "expected_runtime_manifest_sha256": digest(runtime_bytes),
        "source_manifest_bytes": source_bytes,
        "expected_identity_input_sha256": "1" * 64,
        "expected_bindings": bindings,
        "expected_source_commit": "1" * 40,
    }
    return arguments, receipt


def prepared_fake_sealed_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_a: bool = False,
) -> tuple[
    list[str],
    Any,
    runner.CalibrationRuntimeManifest,
    runner.SealedRuntimeContext,
    dict[str, object],
]:
    repository = tmp_path / "repository"
    (repository / "requirements").mkdir(parents=True)
    (repository / "scripts").mkdir()
    (repository / "src" / "recurquant").mkdir(parents=True)
    requirements_path = repository / runner.CALIBRATION_REQUIREMENTS_PATH
    requirements_path.write_bytes(b"authenticated requirements\n")
    capture_source_path = repository / runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH
    capture_source_path.write_bytes(b"authenticated unchanged capture source\n")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    capture_hub_cache_root = cache_root / "hub"
    capture_hub_cache_root.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(capture_hub_cache_root))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(capture_hub_cache_root))
    ruler_root = ruler_receipt_directory(tmp_path / "ruler-receipts")
    runtime_bytes, origins = capture_provenance_runtime_manifest_bytes()
    model_bytes = model_manifest_bytes(staged_model_files())
    artifact_bytes = {
        "repository_source_manifest_file_sha256": b"authenticated source manifest\n",
        "calibration_runtime_manifest_file_sha256": runtime_bytes,
        "model_file_manifest_file_sha256": model_bytes,
        "parquet_materialization_manifest_file_sha256": b"authenticated parquet manifest\n",
    }
    paths = {
        "repository_source_manifest_file_sha256": tmp_path / "source.json",
        "calibration_runtime_manifest_file_sha256": tmp_path / "runtime.json",
        "model_file_manifest_file_sha256": tmp_path / "model.json",
        "parquet_materialization_manifest_file_sha256": tmp_path / "parquet.json",
    }
    for name, path in paths.items():
        path.write_bytes(artifact_bytes[name])
    bindings = {name: digest(data) for name, data in artifact_bytes.items()}
    binding_values = {
        name: f"{index:064x}"
        for index, name in enumerate(
            sorted(
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
            ),
            start=1,
        )
    }
    binding_bytes = b"authenticated Stage-A calibration binding\n"
    binding_path = tmp_path / "stage-a-calibration-binding.json"
    if stage_a:
        binding_path.write_bytes(binding_bytes)
    arguments = [
        "capture-stage-a-identity" if stage_a else "capture-calibration-identity",
        "--repository-root",
        str(repository),
        "--source-commit",
        "4" * 40,
        "--repository-source-manifest",
        str(paths["repository_source_manifest_file_sha256"]),
        "--expected-repository-source-manifest-sha256",
        bindings["repository_source_manifest_file_sha256"],
        "--runtime-manifest",
        str(paths["calibration_runtime_manifest_file_sha256"]),
        "--expected-runtime-manifest-sha256",
        bindings["calibration_runtime_manifest_file_sha256"],
        "--model-file-manifest",
        str(paths["model_file_manifest_file_sha256"]),
        "--expected-model-file-manifest-sha256",
        bindings["model_file_manifest_file_sha256"],
        "--parquet-materialization-manifest",
        str(paths["parquet_materialization_manifest_file_sha256"]),
        "--expected-parquet-materialization-manifest-sha256",
        bindings["parquet_materialization_manifest_file_sha256"],
        "--cache-root",
        str(cache_root),
        "--ruler-receipt-dir",
        str(ruler_root),
        "--output",
        str(tmp_path / "identity-input.json"),
        "--capture-provenance-receipt-output",
        str(tmp_path / "capture-provenance.json"),
    ]
    if stage_a:
        arguments.extend(
            [
                "--stage-a-calibration-binding",
                str(binding_path),
                "--expected-stage-a-calibration-binding-sha256",
                digest(binding_bytes),
            ]
        )
    manifest = runner.parse_calibration_runtime_manifest(runtime_bytes)
    (tmp_path / "base").mkdir()
    (tmp_path / "packages").mkdir()
    runtime_context = runner.SealedRuntimeContext(
        manifest_file_sha256=manifest.file_sha256,
        base_runtime_root=tmp_path / "base",
        package_roots={"packages": tmp_path / "packages"},
        package_import_paths={"packages": "Lib/site-packages"},
        git_executable_path=tmp_path / "git.exe",
        pycache_prefix=tmp_path / "pycache",
    )
    runtime_git = {
        "sha256": manifest.git_executable_sha256,
        "size_bytes": manifest.git_executable_size_bytes,
    }
    bootstrap = runner.BootstrapSource(
        manifest={"git_executable": runtime_git},
        source_commit="4" * 40,
        entries={
            runner.CALIBRATION_REQUIREMENTS_PATH: {
                "raw_sha256": digest(requirements_path.read_bytes())
            },
            runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH: {
                "raw_sha256": digest(capture_source_path.read_bytes())
            },
            runner.SOURCE_VERIFIER_PATH: {"raw_sha256": "1" * 64},
            runner.PARQUET_SOURCE_PATH: {"raw_sha256": "2" * 64},
            runner.IDENTITY_RESOLVER_SOURCE_PATH: {"raw_sha256": "3" * 64},
        },
    )
    observations: dict[str, object] = {
        "capture_hub_cache_root": capture_hub_cache_root,
        "runtime_authentications": 0,
        "ruler_reads": [],
        "source_verifications": 0,
    }

    def verify_source(manifest_value: object, **kwargs: object) -> object:
        assert kwargs == {
            "repo_root": repository,
            "git_executable": runtime_context.git_executable_path,
        }
        observations["source_verifications"] = int(observations["source_verifications"]) + 1
        return manifest_value

    source_module = SimpleNamespace(verify_experiment013_source_manifest=verify_source)

    def validate_stage_a_identity_input_for_capture(
        source: object,
        **kwargs: object,
    ) -> None:
        assert kwargs == {
            "calibration_binding_artifact": binding_bytes,
            "expected_calibration_binding_file_sha256": digest(binding_bytes),
        }
        observations["stage_a_pre_finalization_input"] = source

    resolver_module = SimpleNamespace(
        CALIBRATION_RUNNER_REVISION=runner.RUNNER_REVISION,
        deserialize_stage_a_calibration_binding_artifact=lambda data, **kwargs: (
            SimpleNamespace(
                authorization_file_sha256=binding_values["calibration_authorization_file_sha256"],
                binding=binding_values,
                execution_bindings=bindings,
                source_commit="4" * 40,
            )
            if data == binding_bytes and kwargs == {"expected_file_sha256": digest(binding_bytes)}
            else pytest.fail("Stage-A binding was not authenticated exactly")
        ),
        validate_stage_a_identity_input_for_capture=(validate_stage_a_identity_input_for_capture),
    )
    observations["resolver_module"] = resolver_module

    def capture_identity_input(**kwargs: object) -> dict[str, object]:
        observations["capture_kwargs"] = dict(kwargs)
        during_capture = observations.get("during_capture")
        if callable(during_capture):
            during_capture()
        result = {
            "datasets": {},
            "execution_bindings": bindings,
            "model_weights_loaded": False,
            "phase": "stage_a" if stage_a else "calibration",
            "records": [],
            "schema": runner.CALIBRATION_IDENTITY_INPUT_SCHEMA,
            "tokenizer": {},
        }
        if stage_a:
            result["calibration_binding"] = binding_values
        return result

    receipt_names = [
        name
        for name in runner.RULER_RECEIPT_DIRECTORY_FILENAMES
        if name != "generation-manifest.json"
    ]
    required_ruler: list[dict[str, object]] = []
    ruler_payloads: dict[tuple[str, str, int, int], bytes] = {}
    ruler_filenames: dict[tuple[str, str, int, int], str] = {}
    for index, filename in enumerate(receipt_names):
        category, config, raw_length, raw_seed = filename.removesuffix(".json").split("__")
        key = (category, config, int(raw_length.removeprefix("l")), int(raw_seed.removeprefix("s")))
        required_ruler.append(
            {
                "category": category,
                "config": config,
                "configured_length": key[2],
                "filename": filename,
                "phase": "calibration" if index < 16 else "stage_a",
                "sample_index": 0,
                "seed": key[3],
            }
        )
        ruler_payloads[key] = f"fixture receipt {filename}\n".encode()
        ruler_filenames[key] = filename
    generation_manifest_bytes = b"fixture generation manifest\n"
    observations["required_ruler"] = tuple(required_ruler)
    observations["ruler_payloads"] = ruler_payloads
    observations["generation_manifest_bytes"] = generation_manifest_bytes

    class FakeLiveCaptureSource:
        def __init__(self, **kwargs: object) -> None:
            self.arguments = kwargs
            observations["live_source"] = self

        def ruler_generation_manifest_bytes(self) -> bytes:
            reads = observations["ruler_reads"]
            assert isinstance(reads, list)
            reads.append("generation-manifest.json")
            value = observations["generation_manifest_bytes"]
            assert isinstance(value, bytes)
            return value

        def ruler_receipt_bytes(
            self,
            *,
            category: str,
            config: str,
            configured_length: int,
            seed: int,
        ) -> bytes:
            key = (category, config, configured_length, seed)
            reads = observations["ruler_reads"]
            assert isinstance(reads, list)
            reads.append(ruler_filenames[key])
            return ruler_payloads[key]

    capture_module = SimpleNamespace(
        CAPTURE_VERSION=runner.CALIBRATION_IDENTITY_CAPTURE_VERSION,
        LiveCaptureSource=FakeLiveCaptureSource,
        capture_identity_input=capture_identity_input,
        required_ruler_receipts=lambda: tuple(required_ruler),
    )

    def load_module(
        _module_name: str,
        relative_path: str,
        **_kwargs: object,
    ) -> object:
        return {
            runner.SOURCE_VERIFIER_PATH: source_module,
            runner.PARQUET_SOURCE_PATH: SimpleNamespace(),
            runner.IDENTITY_RESOLVER_SOURCE_PATH: resolver_module,
            runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH: capture_module,
        }[relative_path]

    monkeypatch.setattr(runner, "_bootstrap_source_manifest", lambda *_args, **_kwargs: bootstrap)
    monkeypatch.setattr(runner, "_load_exact_source_module", load_module)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: SimpleNamespace(path=tmp_path / "git.exe", **runtime_git),
    )
    monkeypatch.setattr(runner, "_parse_runtime_requirements", lambda _path: ())
    monkeypatch.setattr(runner, "_preflight_runtime_requirements", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "_preflight_calibration_identity_import_surface",
        lambda **_kwargs: {name: tmp_path / name for name in origins_by_module(origins)},
    )
    monkeypatch.setattr(
        runner,
        "_preload_authenticated_capture_six",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_assert_capture_forbidden_modules_absent", lambda: None)
    monkeypatch.setattr(
        runner,
        "_capture_calibration_identity_module_origins",
        lambda **_kwargs: origins,
    )
    monkeypatch.setattr(
        runner,
        "authenticate_calibration_runtime",
        lambda *_args, **_kwargs: observations.__setitem__(
            "runtime_authentications",
            int(observations["runtime_authentications"]) + 1,
        ),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES",
        ("never_present_capture_dependency",),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        ("never_present_capture_model_runtime",),
    )
    monkeypatch.setattr(
        runner,
        "RULER_GENERATION_MANIFEST_FILE_SHA256",
        digest(generation_manifest_bytes),
    )
    authenticated = SimpleNamespace(manifest_file_sha256=manifest.file_sha256)
    return arguments, authenticated, manifest, runtime_context, observations


def origins_by_module(origins: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(item["module"]): item for item in origins}


def test_capture_provenance_receipt_authenticates_exact_runtime_record_origins(
    tmp_path: Path,
) -> None:
    arguments, _receipt = capture_provenance_receipt_fixture(tmp_path)

    assert (
        runner._authenticate_calibration_identity_capture_provenance(**arguments)
        == (arguments["expected_receipt_sha256"])
    )


def test_capture_provenance_receipt_rejects_replaced_source_manifest(
    tmp_path: Path,
) -> None:
    arguments, _receipt = capture_provenance_receipt_fixture(tmp_path)
    replacement = json.loads(arguments["source_manifest_bytes"])
    replacement["unrelated_authenticated_path"] = "replacement"
    arguments["source_manifest_bytes"] = runner.canonical_json_bytes(replacement)

    with pytest.raises(runner.CalibrationRunError, match="source manifest.*differs"):
        runner._authenticate_calibration_identity_capture_provenance(**arguments)


@pytest.mark.parametrize(
    "mutation",
    [
        "h0",
        "input",
        "binding",
        "module-version",
        "module-path",
        "excluded-policy",
        "publication-contract",
        "schema-v1",
        "status",
    ],
)
def test_capture_provenance_receipt_rejects_every_custody_drift(
    mutation: str,
    tmp_path: Path,
) -> None:
    arguments, receipt = capture_provenance_receipt_fixture(tmp_path)
    if mutation == "h0":
        receipt["source_commit"] = "2" * 40
    elif mutation == "input":
        receipt["identity_input_file_sha256"] = "2" * 64
    elif mutation == "binding":
        receipt["execution_bindings"][  # type: ignore[index]
            "model_file_manifest_file_sha256"
        ] = "2" * 64
    elif mutation == "module-version":
        receipt["critical_module_origins"][0]["version"] = "2.0"  # type: ignore[index]
    elif mutation == "module-path":
        receipt["critical_module_origins"][0][  # type: ignore[index]
            "relative_path"
        ] = "Lib/site-packages/shadow/__init__.py"
    elif mutation == "excluded-policy":
        receipt["excluded_runtime_modules"] = ["setuptools"]
    elif mutation == "publication-contract":
        receipt["publication_contract"] = "child-published-before-host-cleanup"
    elif mutation == "schema-v1":
        receipt["schema_version"] = 1
    else:
        receipt["status"] = "captured_under_authenticated_runtime"
    mutated = runner.canonical_json_bytes(receipt)
    Path(arguments["receipt_path"]).write_bytes(mutated)
    arguments["expected_receipt_sha256"] = digest(mutated)

    with pytest.raises(runner.CalibrationRunError):
        runner._authenticate_calibration_identity_capture_provenance(**arguments)


def test_capture_provenance_receipt_rejects_noncanonical_or_missing_receipt(
    tmp_path: Path,
) -> None:
    arguments, receipt = capture_provenance_receipt_fixture(tmp_path)
    noncanonical = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    Path(arguments["receipt_path"]).write_bytes(noncanonical)
    arguments["expected_receipt_sha256"] = digest(noncanonical)
    with pytest.raises(runner.CalibrationRunError, match="canonical"):
        runner._authenticate_calibration_identity_capture_provenance(**arguments)

    Path(arguments["receipt_path"]).unlink()
    with pytest.raises(runner.CalibrationRunError, match="unavailable"):
        runner._authenticate_calibration_identity_capture_provenance(**arguments)


def test_exact_stdout_bytes_bypasses_text_newline_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TranslatingTextStream:
        def __init__(self) -> None:
            self.buffer = io.BytesIO()
            self.text_writes: list[str] = []

        def write(self, value: str) -> int:
            self.text_writes.append(value)
            translated = value.replace("\n", "\r\n").encode("utf-8")
            self.buffer.write(translated)
            return len(value)

        def flush(self) -> None:
            pass

    stream = TranslatingTextStream()
    monkeypatch.setattr(sys, "stdout", stream)
    payload = b'{"status":"canonical"}\n'

    runner._write_exact_stdout_bytes(payload)

    assert stream.text_writes == []
    assert stream.buffer.getvalue() == payload


def test_exact_stdout_bytes_rejects_incomplete_binary_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShortBinaryStream(io.BytesIO):
        def write(self, value: bytes) -> int:
            return super().write(value[:-1])

    class TextStream:
        def __init__(self) -> None:
            self.buffer = ShortBinaryStream()

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", TextStream())

    with pytest.raises(runner.CalibrationRunError, match="incomplete"):
        runner._write_exact_stdout_bytes(b'{"status":"canonical"}\n')


def test_exact_stdout_bytes_rejects_unavailable_binary_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TextOnlyStream:
        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", TextOnlyStream())

    with pytest.raises(runner.CalibrationRunError, match="binary stdout custody"):
        runner._write_exact_stdout_bytes(b'{"status":"canonical"}\n')


def test_exact_stdout_bytes_rejects_binary_flush_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingFlushStream(io.BytesIO):
        def flush(self) -> None:
            raise OSError("simulated flush failure")

    class TextStream:
        def __init__(self) -> None:
            self.buffer = FailingFlushStream()

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", TextStream())

    with pytest.raises(runner.CalibrationRunError, match="binary stdout custody"):
        runner._write_exact_stdout_bytes(b'{"status":"canonical"}\n')


def test_exact_stdout_bytes_survives_real_windows_pipe_translation() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('isolated_binary_stdout_runner', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._write_exact_stdout_bytes(b'{{"status":"canonical"}}\\n')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
    )

    assert completed.stdout == b'{"status":"canonical"}\n'
    assert completed.stderr == b""


def test_sealed_capture_emits_receipt_candidate_without_publishing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )

    with isolated_sealed_capture_modules():
        assert (
            runner._sealed_capture_calibration_identity(
                arguments,
                manifest=manifest,
                runtime_context=runtime_context,
                authenticated_runtime=authenticated,
                interpreter_path=tmp_path / "python.exe",
            )
            == 0
        )
    identity_bytes = (tmp_path / "identity-input.json").read_bytes()
    assert not (tmp_path / "capture-provenance.json").exists()
    receipt_bytes = capsys.readouterr().out.encode("utf-8")
    receipt = json.loads(receipt_bytes)
    assert receipt_bytes == runner.canonical_json_bytes(receipt)
    assert receipt["identity_input_file_sha256"] == digest(identity_bytes)
    assert receipt["excluded_runtime_modules"] == ["never_present_capture_dependency"]
    assert receipt["publication_contract"] == (
        runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT
    )
    assert receipt["status"] == runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS
    assert observations["runtime_authentications"] == 2
    assert observations["source_verifications"] == 3
    live_source = observations["live_source"]
    assert live_source.arguments["cache_dir"] == observations["capture_hub_cache_root"]
    assert live_source.arguments["cache_dir"] != Path(
        arguments[arguments.index("--cache-root") + 1]
    )
    capture_kwargs = observations["capture_kwargs"]
    assert isinstance(capture_kwargs, dict)
    assert capture_kwargs["phase"] == "calibration"
    assert capture_kwargs["calibration_binding"] is None
    assert "model_root" not in capture_kwargs
    assert "adapter" not in capture_kwargs
    runtime_authentication_context = capture_kwargs["runtime_authentication_context"]
    assert runtime_authentication_context["staged_interpreter"] == tmp_path / "python.exe"
    assert runtime_authentication_context["package_runtime_roots"] == dict(
        runtime_context.package_roots
    )
    required_ruler = observations["required_ruler"]
    assert isinstance(required_ruler, tuple)
    calibration_names = sorted(
        str(item["filename"]) for item in required_ruler if item["phase"] == "calibration"
    )
    assert observations["ruler_reads"] == [
        "generation-manifest.json",
        *calibration_names,
        "generation-manifest.json",
        *calibration_names,
    ]


def test_sealed_stage_a_capture_authenticates_binding_and_emits_only_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch, stage_a=True)
    )

    with isolated_sealed_capture_modules():
        assert (
            runner._sealed_capture_stage_a_identity(
                arguments,
                manifest=manifest,
                runtime_context=runtime_context,
                authenticated_runtime=authenticated,
                interpreter_path=tmp_path / "python.exe",
            )
            == 0
        )

    identity_bytes = (tmp_path / "identity-input.json").read_bytes()
    assert not (tmp_path / "capture-provenance.json").exists()
    receipt_bytes = capsys.readouterr().out.encode("utf-8")
    receipt = json.loads(receipt_bytes)
    assert receipt_bytes == runner.canonical_json_bytes(receipt)
    assert receipt["artifact_kind"] == runner.STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND
    assert receipt["schema_version"] == runner.STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
    assert receipt["phase"] == "stage_a"
    assert receipt["identity_input_file_sha256"] == digest(identity_bytes)
    assert receipt["calibration_binding_file_sha256"] == digest(
        (tmp_path / "stage-a-calibration-binding.json").read_bytes()
    )
    capture_kwargs = observations["capture_kwargs"]
    assert isinstance(capture_kwargs, dict)
    assert capture_kwargs["phase"] == "stage_a"
    assert (
        capture_kwargs["calibration_binding"]
        == (tmp_path / "stage-a-calibration-binding.json").read_bytes()
    )
    assert "model_root" not in capture_kwargs
    assert observations["stage_a_pre_finalization_input"] is not None
    required_ruler = observations["required_ruler"]
    assert isinstance(required_ruler, tuple)
    stage_a_names = sorted(
        str(item["filename"]) for item in required_ruler if item["phase"] == "stage_a"
    )
    assert observations["ruler_reads"] == [
        "generation-manifest.json",
        *stage_a_names,
        "generation-manifest.json",
        *stage_a_names,
    ]


def test_sealed_capture_rejects_hub_environment_that_bypasses_authenticated_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    cache_root = Path(arguments[arguments.index("--cache-root") + 1])
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    with (
        isolated_sealed_capture_modules(),
        pytest.raises(runner.CalibrationRunError, match="Hub cache environment"),
    ):
        runner._sealed_capture_calibration_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert "live_source" not in observations
    assert not (tmp_path / "identity-input.json").exists()
    assert not (tmp_path / "capture-provenance.json").exists()


@pytest.mark.parametrize("mutation", ["generation-manifest", "phase-receipt"])
def test_sealed_capture_rejects_phase_scoped_ruler_file_mutation(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    required_ruler = observations["required_ruler"]
    payloads = observations["ruler_payloads"]
    assert isinstance(required_ruler, tuple)
    assert isinstance(payloads, dict)

    def mutate() -> None:
        if mutation == "generation-manifest":
            observations["generation_manifest_bytes"] = b"changed generation manifest\n"
            return
        selected = next(item for item in required_ruler if item["phase"] == "calibration")
        key = (
            selected["category"],
            selected["config"],
            selected["configured_length"],
            selected["seed"],
        )
        payloads[key] = b"changed phase receipt\n"

    observations["during_capture"] = mutate
    with (
        isolated_sealed_capture_modules(),
        pytest.raises(runner.CalibrationRunError, match="RULER files changed|SHA-256 drifted"),
    ):
        runner._sealed_capture_calibration_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert not (tmp_path / "identity-input.json").exists()
    assert not (tmp_path / "capture-provenance.json").exists()
    assert capsys.readouterr().out == ""


def test_sealed_capture_does_not_open_or_hash_other_phase_ruler_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    required_ruler = observations["required_ruler"]
    payloads = observations["ruler_payloads"]
    assert isinstance(required_ruler, tuple)
    assert isinstance(payloads, dict)
    other = next(item for item in required_ruler if item["phase"] == "stage_a")
    other_key = (
        other["category"],
        other["config"],
        other["configured_length"],
        other["seed"],
    )

    def mutate_other_phase() -> None:
        payloads[other_key] = b"changed other-phase receipt\n"

    observations["during_capture"] = mutate_other_phase
    with isolated_sealed_capture_modules():
        assert (
            runner._sealed_capture_calibration_identity(
                arguments,
                manifest=manifest,
                runtime_context=runtime_context,
                authenticated_runtime=authenticated,
                interpreter_path=tmp_path / "python.exe",
            )
            == 0
        )
    capsys.readouterr()

    reads = observations["ruler_reads"]
    assert isinstance(reads, list)
    assert str(other["filename"]) not in reads


def test_sealed_stage_a_capture_rejects_binding_digest_before_live_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch, stage_a=True)
    )
    expected_index = arguments.index("--expected-stage-a-calibration-binding-sha256") + 1
    arguments[expected_index] = "f" * 64

    with (
        isolated_sealed_capture_modules(),
        pytest.raises(runner.CalibrationRunError, match="explicit SHA-256"),
    ):
        runner._sealed_capture_stage_a_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )
    assert "capture_kwargs" not in observations
    assert not (tmp_path / "identity-input.json").exists()
    assert not (tmp_path / "capture-provenance.json").exists()


def test_sealed_stage_a_capture_rejects_resolver_runner_revision_before_live_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch, stage_a=True)
    )
    resolver_module = observations["resolver_module"]
    assert isinstance(resolver_module, SimpleNamespace)
    resolver_module.CALIBRATION_RUNNER_REVISION = "experiment-013-static-q468-calibration-runner-v9"

    with (
        isolated_sealed_capture_modules(),
        pytest.raises(
            runner.CalibrationRunError,
            match="authenticated identity resolver runner revision drifted",
        ),
    ):
        runner._sealed_capture_stage_a_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert "capture_kwargs" not in observations
    assert not (tmp_path / "identity-input.json").exists()
    assert not (tmp_path / "capture-provenance.json").exists()


def test_sealed_stage_a_capture_rejects_resolver_pre_finalization_failure_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, authenticated, manifest, runtime_context, observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch, stage_a=True)
    )
    resolver_module = observations["resolver_module"]
    assert isinstance(resolver_module, SimpleNamespace)

    def reject_invalid_stage_a_input(_source: object, **_kwargs: object) -> None:
        raise ValueError("records are forged")

    resolver_module.validate_stage_a_identity_input_for_capture = reject_invalid_stage_a_input

    with (
        isolated_sealed_capture_modules(),
        pytest.raises(
            runner.CalibrationRunError,
            match="authenticated pre-finalization validation",
        ),
    ):
        runner._sealed_capture_stage_a_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert not (tmp_path / "identity-input.json").exists()
    assert not (tmp_path / "capture-provenance.json").exists()
    assert capsys.readouterr().out == ""


def test_sealed_capture_never_attempts_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    downstream = tmp_path / "downstream"
    downstream.mkdir()
    provenance_arguments, _receipt = capture_provenance_receipt_fixture(downstream)
    arguments, authenticated, manifest, runtime_context, _observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    publish = runner._atomic_publish_new

    def fail_receipt(path: Path, payload: bytes, **kwargs: object) -> None:
        if path.name == "capture-provenance.json":
            raise OSError("injected receipt publication failure")
        publish(path, payload, **kwargs)

    monkeypatch.setattr(runner, "_atomic_publish_new", fail_receipt)
    with isolated_sealed_capture_modules():
        assert (
            runner._sealed_capture_calibration_identity(
                arguments,
                manifest=manifest,
                runtime_context=runtime_context,
                authenticated_runtime=authenticated,
                interpreter_path=tmp_path / "python.exe",
            )
            == 0
        )
    assert (tmp_path / "identity-input.json").is_file()
    assert not (tmp_path / "capture-provenance.json").exists()
    assert json.loads(capsys.readouterr().out)["artifact_kind"] == (
        runner.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND
    )

    Path(provenance_arguments["receipt_path"]).unlink()
    with pytest.raises(runner.CalibrationRunError, match="unavailable"):
        runner._authenticate_calibration_identity_capture_provenance(**provenance_arguments)


def test_sealed_capture_rejects_output_parent_replacement_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, _observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output_path = output_parent / "identity-input.json"
    arguments[arguments.index("--output") + 1] = str(output_path)
    displaced_parent = tmp_path / "displaced-output-parent"
    publish = runner._atomic_publish_new

    def replace_parent(path: Path, payload: bytes, **kwargs: object) -> None:
        if path == output_path:
            output_parent.rename(displaced_parent)
            output_parent.mkdir()
        publish(path, payload, **kwargs)

    monkeypatch.setattr(runner, "_atomic_publish_new", replace_parent)
    with (
        isolated_sealed_capture_modules(),
        pytest.raises(
            runner.CalibrationRunError,
            match="parent changed",
        ),
    ):
        runner._sealed_capture_calibration_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert not output_path.exists()
    assert not (displaced_parent / output_path.name).exists()
    assert not (tmp_path / "capture-provenance.json").exists()


def test_sealed_capture_rejects_receipt_parent_replacement_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, _observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    receipt_parent = tmp_path / "receipt-parent"
    receipt_parent.mkdir()
    receipt_path = receipt_parent / "capture-provenance.json"
    arguments[arguments.index("--capture-provenance-receipt-output") + 1] = str(receipt_path)
    displaced_parent = tmp_path / "displaced-receipt-parent"
    revalidate = runner._revalidate_new_capture_artifact_path
    replaced = False

    def replace_parent(
        snapshot: runner.NewCaptureArtifactPath,
        *,
        context: str,
    ) -> None:
        nonlocal replaced
        if snapshot.path == receipt_path and not replaced:
            receipt_parent.rename(displaced_parent)
            receipt_parent.mkdir()
            replaced = True
        revalidate(snapshot, context=context)

    monkeypatch.setattr(runner, "_revalidate_new_capture_artifact_path", replace_parent)
    with (
        isolated_sealed_capture_modules(),
        pytest.raises(
            runner.CalibrationRunError,
            match="parent changed",
        ),
    ):
        runner._sealed_capture_calibration_identity(
            arguments,
            manifest=manifest,
            runtime_context=runtime_context,
            authenticated_runtime=authenticated,
            interpreter_path=tmp_path / "python.exe",
        )

    assert (tmp_path / "identity-input.json").is_file()
    assert not receipt_path.exists()
    assert not (displaced_parent / receipt_path.name).exists()


def test_capture_excluded_import_blocker_is_a_hard_stop_without_partial_module() -> None:
    blocker = runner._ExcludedCalibrationIdentityImportBlocker()
    module_name = runner.CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES[0]
    sys.modules.pop(module_name, None)

    with pytest.raises(runner.CalibrationRunError, match="excluded runtime import"):
        blocker.find_spec(module_name)
    assert blocker.attempts == [module_name]
    assert module_name not in sys.modules


def test_capture_import_isolation_hides_probe_blocks_import_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_forbidden_probe_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES",
        (module_name,),
    )
    original_find_spec = importlib.util.find_spec
    isolation = runner._CalibrationIdentityImportIsolation()

    isolation.activate()
    try:
        assert importlib.util.find_spec(module_name) is None
        with pytest.raises(runner.CalibrationRunError, match="forbidden model/CUDA import"):
            __import__(module_name)
        with pytest.raises(
            runner.CalibrationRunError,
            match="attempted forbidden model/CUDA imports",
        ):
            isolation.assert_intact()
        assert isolation.availability_probes == [module_name]
        assert isolation.blocker.attempts == [module_name]
        assert module_name not in sys.modules
    finally:
        with pytest.raises(
            runner.CalibrationRunError,
            match="attempted forbidden model/CUDA imports before restoration",
        ):
            isolation.restore(primary_error=None)

    assert importlib.util.find_spec is original_find_spec
    assert isolation.blocker not in sys.meta_path


def test_capture_import_isolation_rejects_a_swallowed_forbidden_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_swallowed_forbidden_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES",
        (module_name,),
    )
    isolation = runner._CalibrationIdentityImportIsolation()

    isolation.activate()
    try:
        with suppress(runner.CalibrationRunError):
            __import__(module_name)
        with pytest.raises(
            runner.CalibrationRunError,
            match="attempted forbidden model/CUDA imports",
        ):
            isolation.assert_intact()
    finally:
        with pytest.raises(
            runner.CalibrationRunError,
            match="attempted forbidden model/CUDA imports before restoration",
        ):
            isolation.restore(primary_error=None)


def test_capture_import_isolation_restores_after_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_restore_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    original_find_spec = importlib.util.find_spec
    isolation = runner._CalibrationIdentityImportIsolation()

    with pytest.raises(RuntimeError, match="injected capture failure"):
        try:
            isolation.activate()
            raise RuntimeError("injected capture failure")
        finally:
            isolation.restore(primary_error=sys.exception())

    assert importlib.util.find_spec is original_find_spec
    assert isolation.blocker not in sys.meta_path


def test_capture_import_isolation_prevents_transient_preceding_finder_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_transient_finder_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES",
        (module_name,),
    )
    original_meta_path = sys.meta_path
    original_import = builtins.__import__
    bootstrap = runner.importlib._bootstrap
    original_find_and_load = bootstrap._find_and_load
    isolation = runner._CalibrationIdentityImportIsolation()

    class SelfRemovingFinder:
        called = False

        def find_spec(
            self,
            fullname: str,
            path: object = None,
            target: object = None,
        ) -> None:
            del path, target
            if fullname == module_name:
                self.called = True
                sys.meta_path = [item for item in sys.meta_path if item is not self]
            return None

    isolation.activate()
    retained_import = isolation._original_import
    assert callable(retained_import)
    finder = SelfRemovingFinder()
    primary_error: BaseException | None = None
    try:
        sys.meta_path = [finder, *sys.meta_path]
        with pytest.raises(
            runner.CalibrationRunError,
            match="import-finder topology object changed",
        ) as captured:
            retained_import(module_name)
        primary_error = captured.value
        assert finder.called is False
        assert module_name not in sys.modules
    finally:
        isolation.restore(primary_error=primary_error)

    assert sys.meta_path is original_meta_path
    assert builtins.__import__ is original_import
    assert bootstrap._find_and_load is original_find_and_load
    assert isolation.blocker not in sys.meta_path
    assert primary_error is not None
    assert any(
        "topology object changed before restoration" in note
        for note in getattr(primary_error, "__notes__", ())
    )


def test_capture_import_isolation_rejects_reassigned_empty_meta_path_via_importlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_empty_meta_path_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_FORBIDDEN_MODULE_PREFIXES",
        (module_name,),
    )
    original_meta_path = sys.meta_path
    isolation = runner._CalibrationIdentityImportIsolation()
    primary_error: BaseException | None = None

    isolation.activate()
    try:
        sys.meta_path = []
        with pytest.raises(
            runner.CalibrationRunError,
            match="import-finder topology object changed",
        ) as captured:
            importlib.import_module(module_name)
        primary_error = captured.value
        assert module_name not in sys.modules
    finally:
        isolation.restore(primary_error=primary_error)

    assert sys.meta_path is original_meta_path
    assert primary_error is not None


def test_capture_import_isolation_meta_path_is_not_a_list_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_tuple_meta_path_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    isolation = runner._CalibrationIdentityImportIsolation()

    isolation.activate()
    try:
        sealed_meta_path = sys.meta_path
        before = tuple(sealed_meta_path)
        assert not isinstance(sealed_meta_path, list)
        with pytest.raises(TypeError):
            list.insert(sealed_meta_path, 0, object())  # type: ignore[arg-type]
        assert tuple(sealed_meta_path) == before
        isolation.assert_intact()
    finally:
        isolation.restore(primary_error=sys.exception())


def test_capture_import_isolation_restores_mutated_detached_original_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_detached_meta_path_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    original_meta_path = sys.meta_path
    original_topology = tuple(original_meta_path)
    isolation = runner._CalibrationIdentityImportIsolation()
    primary_error = runner.CalibrationRunError("injected primary capture failure")

    isolation.activate()
    original_meta_path.insert(0, object())
    isolation.restore(primary_error=primary_error)

    assert sys.meta_path is original_meta_path
    assert tuple(sys.meta_path) == original_topology
    assert any(
        "saved capture import-finder topology changed" in note
        for note in getattr(primary_error, "__notes__", ())
    )


def test_capture_import_isolation_partial_activation_failure_restores_every_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_partial_activation_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    original_meta_path = sys.meta_path
    original_find_spec = importlib.util.find_spec
    original_import = builtins.__import__
    bootstrap = runner.importlib._bootstrap
    original_find_and_load = bootstrap._find_and_load
    isolation = runner._CalibrationIdentityImportIsolation()

    monkeypatch.setattr(
        isolation,
        "_assert_guard_state",
        lambda: (_ for _ in ()).throw(runner.CalibrationRunError("injected activation failure")),
    )

    with pytest.raises(runner.CalibrationRunError, match="injected activation failure"):
        isolation.activate()

    assert sys.meta_path is original_meta_path
    assert importlib.util.find_spec is original_find_spec
    assert builtins.__import__ is original_import
    assert bootstrap._find_and_load is original_find_and_load
    isolation.restore(primary_error=None)


def test_capture_import_isolation_records_and_rejects_meta_path_mutation_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_guarded_meta_path_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    isolation = runner._CalibrationIdentityImportIsolation()
    primary_error: BaseException | None = None

    isolation.activate()
    try:
        with pytest.raises(
            runner.CalibrationRunError,
            match="topology mutation was attempted: insert",
        ) as captured:
            sys.meta_path.insert(0, object())
        primary_error = captured.value
        with pytest.raises(
            runner.CalibrationRunError,
            match="topology mutation was attempted",
        ):
            isolation.assert_intact()
    finally:
        isolation.restore(primary_error=primary_error)

    assert isolation.blocker not in sys.meta_path
    assert primary_error is not None
    assert any(
        "topology mutation was attempted" in note
        for note in getattr(primary_error, "__notes__", ())
    )


def test_capture_import_isolation_latches_self_removing_finder_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "recurquant_capture_self_removing_finder_fixture"
    monkeypatch.setattr(
        runner,
        "CALIBRATION_IDENTITY_HIDDEN_AVAILABILITY_MODULES",
        (module_name,),
    )
    original_meta_path = sys.meta_path

    class SelfRemovingFinder:
        called = False
        swallowed = False

        def find_spec(
            self,
            fullname: str,
            path: object = None,
            target: object = None,
        ) -> None:
            del path, target
            if fullname == module_name:
                self.called = True
                try:
                    sys.meta_path.remove(self)
                except runner.CalibrationRunError:
                    self.swallowed = True
            return None

    finder = SelfRemovingFinder()
    original_meta_path.insert(0, finder)
    isolation = runner._CalibrationIdentityImportIsolation()
    primary_error: BaseException | None = None
    try:
        isolation.activate()
        try:
            with suppress(ModuleNotFoundError):
                importlib.import_module(module_name)
            assert finder.called is True
            assert finder.swallowed is True
            with pytest.raises(
                runner.CalibrationRunError,
                match="topology mutation was attempted",
            ) as captured:
                isolation.assert_intact()
            primary_error = captured.value
        finally:
            isolation.restore(primary_error=primary_error)
    finally:
        if finder in original_meta_path:
            original_meta_path.remove(finder)

    assert sys.meta_path is original_meta_path
    assert primary_error is not None
    assert any(
        "topology mutation was attempted" in note
        for note in getattr(primary_error, "__notes__", ())
    )


def test_authenticated_six_preload_supports_dataset_and_tokenizer_under_isolation() -> None:
    code = f"""
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('isolated_six_capture_runner', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'six' not in sys.modules
assert 'torch' not in sys.modules

six_spec = importlib.util.find_spec('six')
assert six_spec is not None and isinstance(six_spec.origin, str)
six_path = Path(six_spec.origin).resolve(strict=True)
package_root = six_path.parent
payload = six_path.read_bytes()
relative = 'six.py'
file_record = module.RuntimeFileRecord(
    path=relative,
    sha256=hashlib.sha256(payload).hexdigest(),
    size_bytes=len(payload),
)
manifest = module.CalibrationRuntimeManifest(
    python_implementation='CPython',
    python_version='fixture',
    python_cache_tag='fixture',
    python_abi_flags='',
    machine_system='fixture',
    machine_architecture='fixture',
    machine_name='fixture',
    machine_byteorder=sys.byteorder,
    machine_pointer_bits=64,
    launch_policy={{}},
    base_sys_path=(),
    base_runtime_root='base-runtime',
    package_roots=(module.RuntimePackageRootRecord(name='packages', import_path='.'),),
    interpreter_root='base-runtime',
    interpreter_relative_path='python.exe',
    interpreter_size_bytes=1,
    interpreter_sha256='0' * 64,
    git_executable_absolute_path_sha256='1' * 64,
    git_executable_sha256='2' * 64,
    git_executable_size_bytes=1,
    runtime_trees=(
        module.RuntimeTreeRecord(name='packages', kind='packages', files=(file_record,)),
    ),
    distributions=(
        module.RuntimeDistributionRecord(
            name='six',
            version='1.17.0',
            package_root='packages',
            files=(relative,),
        ),
    ),
    file_sha256='3' * 64,
)
context = module.SealedRuntimeContext(
    manifest_file_sha256=manifest.file_sha256,
    base_runtime_root=package_root,
    package_roots={{'packages': package_root}},
    package_import_paths={{'packages': '.'}},
    git_executable_path=package_root / 'git',
    pycache_prefix=package_root / 'unused-pycache',
)

original_meta_path = sys.meta_path
original_topology = tuple(original_meta_path)
excluded = module._ExcludedCalibrationIdentityImportBlocker()
isolation = module._CalibrationIdentityImportIsolation()
binding = None
primary_error = None
try:
    sys.meta_path.insert(0, excluded)
    binding = module._preload_authenticated_capture_six(
        expected_origin=six_path,
        manifest=manifest,
        runtime_context=context,
    )
    assert tuple(sys.meta_path) == binding.topology_with_importer
    assert binding.topology_with_importer == (
        *binding.topology_before_import,
        binding.importer,
    )
    isolation.activate(authenticated_six=binding)

    from datasets import Dataset
    import pandas
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    assert Dataset.from_dict({{'text': ['hello']}})[0]['text'] == 'hello'
    assert pandas.DataFrame({{'value': [1]}}).iloc[0, 0] == 1
    tokenizer = Tokenizer(models.WordLevel({{'[UNK]': 0, 'hello': 1}}, unk_token='[UNK]'))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    with tempfile.TemporaryDirectory() as directory:
        fast.save_pretrained(directory)
        loaded = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        assert loaded.encode('hello', add_special_tokens=False) == [1]
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    isolation.assert_intact()
    try:
        sys.meta_path.append(object())
    except module.CalibrationRunError as error:
        primary_error = error
    else:
        raise AssertionError('sealed meta-path append was accepted')
finally:
    isolation.restore(primary_error=primary_error)
    if binding is not None:
        module._restore_authenticated_capture_six(binding, primary_error=primary_error)
    if excluded in sys.meta_path:
        sys.meta_path.remove(excluded)

assert primary_error is not None
assert sys.meta_path is original_meta_path
assert tuple(sys.meta_path) == original_topology
assert 'six' not in sys.modules
assert not any(name.startswith('six.') for name in sys.modules)
assert any(
    'topology mutation was attempted' in note
    for note in getattr(primary_error, '__notes__', ())
)
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_authenticated_six_preload_rejects_adversarial_state_and_cleans_exactly() -> None:
    code = f"""
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('isolated_adversarial_six_runner', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def runtime_for(source, package_root, relative):
    payload = source.read_bytes()
    record = module.RuntimeFileRecord(
        path=relative,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    manifest = module.CalibrationRuntimeManifest(
        python_implementation='CPython', python_version='fixture',
        python_cache_tag='fixture', python_abi_flags='', machine_system='fixture',
        machine_architecture='fixture', machine_name='fixture',
        machine_byteorder=sys.byteorder, machine_pointer_bits=64, launch_policy={{}},
        base_sys_path=(), base_runtime_root='base-runtime',
        package_roots=(module.RuntimePackageRootRecord(name='packages', import_path='.'),),
        interpreter_root='base-runtime', interpreter_relative_path='python.exe',
        interpreter_size_bytes=1, interpreter_sha256='0' * 64,
        git_executable_absolute_path_sha256='1' * 64,
        git_executable_sha256='2' * 64, git_executable_size_bytes=1,
        runtime_trees=(
            module.RuntimeTreeRecord(name='packages', kind='packages', files=(record,)),
        ),
        distributions=(
            module.RuntimeDistributionRecord(
                name='six', version='1.17.0', package_root='packages', files=(relative,),
            ),
        ),
        file_sha256='3' * 64,
    )
    context = module.SealedRuntimeContext(
        manifest_file_sha256=manifest.file_sha256,
        base_runtime_root=package_root,
        package_roots={{'packages': package_root}},
        package_import_paths={{'packages': '.'}},
        git_executable_path=package_root / 'git',
        pycache_prefix=package_root / 'unused-pycache',
    )
    return manifest, context

valid_tail = '''
class _SixMetaPathImporter:
    def __init__(self):
        self.name = __name__
_importer = _SixMetaPathImporter()
sys.meta_path.append(_importer)
'''

original_meta_path = sys.meta_path
original_topology = tuple(original_meta_path)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / 'six.py'

    source.write_text('import sys\\n' + valid_tail, encoding='utf-8')
    manifest, context = runtime_for(source, root, 'six.py')
    sys.modules['six.moves'] = ModuleType('six.moves')
    try:
        try:
            module._preload_authenticated_capture_six(
                expected_origin=source, manifest=manifest, runtime_context=context,
            )
        except module.CalibrationRunError as error:
            assert 'preloaded' in str(error)
        else:
            raise AssertionError('orphan six.moves was accepted')
    finally:
        sys.modules.pop('six.moves', None)
    assert tuple(sys.meta_path) == original_topology

    Orphan = type(
        '_SixMetaPathImporter',
        (),
        {{'__module__': 'six', '__init__': lambda self: setattr(self, 'name', 'six')}},
    )
    orphan = Orphan()
    sys.meta_path.append(orphan)
    try:
        try:
            module._preload_authenticated_capture_six(
                expected_origin=source, manifest=manifest, runtime_context=context,
            )
        except module.CalibrationRunError as error:
            assert 'already contains a six importer' in str(error)
        else:
            raise AssertionError('orphan six importer was accepted')
    finally:
        sys.meta_path.remove(orphan)
    assert tuple(sys.meta_path) == original_topology

    source.write_text(
        'import sys\\ntry:\\n    import torch\\nexcept BaseException:\\n    pass\\n' + valid_tail,
        encoding='utf-8',
    )
    manifest, context = runtime_for(source, root, 'six.py')
    excluded = module._ExcludedCalibrationIdentityImportBlocker()
    sys.meta_path.insert(0, excluded)
    topology_with_excluded = tuple(sys.meta_path)
    try:
        try:
            module._preload_authenticated_capture_six(
                expected_origin=source, manifest=manifest, runtime_context=context,
            )
        except module.CalibrationRunError as error:
            assert 'attempted forbidden model/CUDA imports' in str(error)
        else:
            raise AssertionError('swallowed forbidden import was accepted')
        assert tuple(sys.meta_path) == topology_with_excluded
        assert 'six' not in sys.modules
        assert 'torch' not in sys.modules
    finally:
        sys.meta_path.remove(excluded)
    assert tuple(sys.meta_path) == original_topology

    source.write_text('import sys\\n' + valid_tail + '\\nraise RuntimeError("partial")\\n')
    manifest, context = runtime_for(source, root, 'six.py')
    try:
        module._preload_authenticated_capture_six(
            expected_origin=source, manifest=manifest, runtime_context=context,
        )
    except RuntimeError as error:
        assert str(error) == 'partial'
    else:
        raise AssertionError('partial six initialization was accepted')
    assert tuple(sys.meta_path) == original_topology
    assert 'six' not in sys.modules

    package_source = root / 'six' / '__init__.py'
    package_source.parent.mkdir()
    package_source.write_text('import sys\\n' + valid_tail, encoding='utf-8')
    package_manifest, package_context = runtime_for(
        package_source, root, 'six/__init__.py'
    )
    try:
        module._preload_authenticated_capture_six(
            expected_origin=package_source,
            manifest=package_manifest,
            runtime_context=package_context,
        )
    except module.CalibrationRunError as error:
        assert 'critical module is shadowed: six' in str(error)
    else:
        raise AssertionError('package-form six origin was accepted')
    assert tuple(sys.meta_path) == original_topology

    source.write_text('import sys\\n' + valid_tail, encoding='utf-8')
    manifest, context = runtime_for(source, root, 'six.py')
    binding = module._preload_authenticated_capture_six(
        expected_origin=source, manifest=manifest, runtime_context=context,
    )
    isolation = module._CalibrationIdentityImportIsolation()
    isolation.activate(authenticated_six=binding)
    replacement_error = None
    try:
        sys.modules['six'] = ModuleType('six')
        try:
            isolation.assert_intact()
        except module.CalibrationRunError as error:
            replacement_error = error
        else:
            raise AssertionError('post-seal six module replacement was accepted')
    finally:
        isolation.restore(primary_error=replacement_error)
        module._restore_authenticated_capture_six(binding, primary_error=replacement_error)
    assert replacement_error is not None
    assert tuple(sys.meta_path) == original_topology
    assert 'six' not in sys.modules
    assert any(
        'six module identity changed' in note
        for note in getattr(replacement_error, '__notes__', ())
    )

    binding = module._preload_authenticated_capture_six(
        expected_origin=source, manifest=manifest, runtime_context=context,
    )
    unreadable_error = module.CalibrationRunError('injected primary capture failure')
    original_stream_file_sha256 = module._stream_file_sha256
    def deny_six_hash(_path):
        raise PermissionError('injected unreadable six source')
    module._stream_file_sha256 = deny_six_hash
    try:
        module._restore_authenticated_capture_six(
            binding, primary_error=unreadable_error,
        )
    finally:
        module._stream_file_sha256 = original_stream_file_sha256
    assert tuple(sys.meta_path) == original_topology
    assert 'six' not in sys.modules
    assert any(
        'six source became unavailable during capture' in note
        for note in getattr(unreadable_error, '__notes__', ())
    )

assert sys.meta_path is original_meta_path
assert tuple(sys.meta_path) == original_topology
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_sealed_capture_restoration_error_cannot_skip_other_policy_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, authenticated, manifest, runtime_context, _observations = (
        prepared_fake_sealed_capture(tmp_path, monkeypatch)
    )
    original_restore = runner._CalibrationIdentityImportIsolation.restore
    module_names = (
        runner.CALIBRATION_IDENTITY_CAPTURE_RUNNER_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_MODULE,
        runner.IDENTITY_RESOLVER_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_PARQUET_MODULE,
        runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_MODULE,
        "recurquant",
    )
    preexisting = {name: sys.modules.get(name) for name in module_names}

    def fail_after_restore(
        self: runner._CalibrationIdentityImportIsolation,
        *,
        primary_error: BaseException | None,
    ) -> None:
        original_restore(self, primary_error=primary_error)
        raise runner.CalibrationRunError("injected restoration failure")

    monkeypatch.setattr(runner._CalibrationIdentityImportIsolation, "restore", fail_after_restore)

    with isolated_sealed_capture_modules():
        with pytest.raises(runner.CalibrationRunError, match="injected restoration failure"):
            runner._sealed_capture_calibration_identity(
                arguments,
                manifest=manifest,
                runtime_context=runtime_context,
                authenticated_runtime=authenticated,
                interpreter_path=tmp_path / "python.exe",
            )
        assert not any(
            isinstance(item, runner._ExcludedCalibrationIdentityImportBlocker)
            for item in sys.meta_path
        )
        for name in module_names:
            assert name not in sys.modules

    assert {name: sys.modules.get(name) for name in module_names} == preexisting


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 3.0), ("dont_write_bytecode", True)],
)
def test_runtime_manifest_rejects_equality_compatible_json_types(
    field: str,
    value: object,
) -> None:
    document = json.loads(runtime_manifest_bytes())
    if field == "schema_version":
        document[field] = value
    else:
        document["launch_policy"][field] = value

    with pytest.raises(ValueError):
        runner.parse_calibration_runtime_manifest(runner.canonical_json_bytes(document))


def test_bootstrap_identity_rejects_float_schema_version() -> None:
    data = bootstrap_identity_bytes(
        source_manifest_sha256="7" * 64,
        runtime_manifest_sha256="8" * 64,
        model_manifest_sha256="9" * 64,
        parquet_manifest_sha256="a" * 64,
    )
    document = json.loads(data)
    document["evidence"]["schema_version"] = 5.0
    document["canonical_evidence_sha256"] = digest(
        runner.canonical_json_bytes(document["evidence"])
    )

    with pytest.raises(runner.CalibrationRunError, match="state"):
        runner._bootstrap_identity_bindings(runner.canonical_json_bytes(document))


def test_bootstrap_identity_rejects_obsolete_schema_v4() -> None:
    data = bootstrap_identity_bytes(
        source_manifest_sha256="7" * 64,
        runtime_manifest_sha256="8" * 64,
        model_manifest_sha256="9" * 64,
        parquet_manifest_sha256="a" * 64,
    )
    document = json.loads(data)
    document["evidence"]["schema_version"] = 4
    document["canonical_evidence_sha256"] = digest(
        runner.canonical_json_bytes(document["evidence"])
    )

    with pytest.raises(runner.CalibrationRunError, match="state"):
        runner._bootstrap_identity_bindings(runner.canonical_json_bytes(document))


def test_model_manifest_rejects_float_schema_version() -> None:
    document = json.loads(
        model_manifest_bytes(
            {
                "config.json": b"{}",
                "model.safetensors.index.json": b"{}",
                "model.safetensors-00001-of-00001.safetensors": b"weights",
            }
        )
    )
    document["schema_version"] = float(runner.MODEL_FILE_MANIFEST_SCHEMA)

    with pytest.raises(ValueError, match="schema"):
        runner.parse_model_file_manifest(runner.canonical_json_bytes(document))


def write_model_root(path: Path, files: Mapping[str, bytes]) -> None:
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


class FakeBackend:
    geometry = runner.Geometry(layer_indices=(0, 2), heads=2, key_rows=4, value_width=4)

    def __init__(
        self,
        frozen_identity: Any,
        events: list[str],
        *,
        stability_passed: bool = True,
        decode_error: BaseException | None = None,
        expected_source_commit: str | None = None,
        expected_identity_bytes: bytes = b"identity",
    ) -> None:
        self.frozen_identity = frozen_identity
        self.events = events
        self.stability_passed = stability_passed
        self.decode_error = decode_error
        self.expected_source_commit = expected_source_commit or current_head()
        self.expected_identity_bytes = expected_identity_bytes
        self.captured: list[Any] = []

    def decode_identity(self, data: bytes) -> Any:
        self.events.append("decode_identity")
        assert data == self.expected_identity_bytes
        if self.decode_error is not None:
            raise self.decode_error
        return self.frozen_identity

    def reduce_sequence(
        self,
        item: Mapping[str, object],
        token_ids: tuple[int, ...],
        captured: Any,
    ) -> object:
        self.events.append("reduce_sequence")
        self.captured.append((item, token_ids, captured))
        return {"canonical_id": item["canonical_id"]}

    def finalize(
        self,
        scores: Sequence[object],
        *,
        identity: Any,
        source_commit: str,
    ) -> Any:
        self.events.append("finalize")
        assert len(scores) == len(identity.records)
        assert source_commit == self.expected_source_commit
        stability = {
            "checks": [{"name": "fake", "passed": self.stability_passed}],
            "passed": self.stability_passed,
        }
        if not self.stability_passed:
            return runner.FinalizationResult(
                passed=False,
                stability=stability,
                artifacts=None,
            )
        artifacts = runner.CalibrationArtifacts(
            score=b"score",
            comparator_score=b"comparator-score",
            split_half=b"split",
            static_k27030=b"k27030",
            static_k29334=b"k29334",
            static_mse_k29334=b"mse-k29334",
            static_fisher_k29334=b"fisher-k29334",
            static_q48=b"q48",
            calibration_core_binding=b"binding",
            stability=stability,
        )
        return runner.FinalizationResult(
            passed=True,
            stability=stability,
            artifacts=artifacts,
        )


class FakeAdapter:
    def __init__(
        self,
        sequence_by_id: Mapping[str, Any],
        events: list[str],
        *,
        invalid_kernel_receipt: bool = False,
    ) -> None:
        self.sequence_by_id = sequence_by_id
        self.events = events
        self.invalid_kernel_receipt = invalid_kernel_receipt
        self.closed = False
        self.capture_flags: list[bool] = []
        self.fisher_calls: list[tuple[int, int, int]] = []
        self.geometry = FakeBackend.geometry

    def materialize_sequence(self, item: Mapping[str, object]) -> Any:
        self.events.append("materialize_sequence")
        return self.sequence_by_id[str(item["canonical_id"])]

    def load_model(self, authenticated: Any) -> object:
        self.events.append("load_model")
        assert authenticated.revision == "2" * 40
        return object()

    def begin_sequence(self, model: object, item: Mapping[str, object]) -> None:
        del model, item
        self.events.append("begin_sequence")

    def step_token(
        self,
        model: object,
        *,
        token_id: int,
        position: int,
        capture_state: bool,
    ) -> Any:
        del model
        self.events.append("step_token")
        self.capture_flags.append(capture_state)
        geometry = self.geometry
        return api.StepObservation(
            position=position,
            token_id=token_id,
            layer_indices=geometry.layer_indices,
            recurrence_query=torch.ones(
                geometry.layers,
                geometry.heads,
                geometry.key_rows,
            ),
            recurrent_state=(
                torch.ones(
                    geometry.layers,
                    geometry.heads,
                    geometry.key_rows,
                    geometry.value_width,
                )
                if capture_state
                else None
            ),
            successful_kernel_calls_per_layer=(
                (2,) * geometry.layers if self.invalid_kernel_receipt else (1,) * geometry.layers
            ),
        )

    def step_token_with_fisher(
        self,
        model: object,
        *,
        token_id: int,
        position: int,
        target_token_id: int,
        capture_state: bool,
    ) -> object:
        self.events.append("step_token_with_fisher")
        self.fisher_calls.append((position, token_id, target_token_id))
        observation = self.step_token(
            model,
            token_id=token_id,
            position=position,
            capture_state=capture_state,
        )
        geometry = self.geometry
        source = torch.ones(
            geometry.layers,
            geometry.heads,
            geometry.key_rows,
            geometry.value_width,
        )
        return api.FisherStepObservation(
            boundary_position=position - 1,
            input_position=position,
            target_position=position + 1,
            input_token_id=token_id,
            target_token_id=target_token_id,
            step_observation=observation,
            source_recurrent_state=source,
            source_state_gradient=torch.full_like(source, 0.25),
            target_nll=1.0,
        )

    def end_sequence(self, model: object, item: Mapping[str, object]) -> None:
        del model, item
        self.events.append("end_sequence")

    def close_model(self, model: object) -> None:
        del model
        self.events.append("close_model")
        self.closed = True

    def runtime_metadata(self) -> Mapping[str, object]:
        self.events.append("runtime_metadata")
        return {
            "fisher_step_count": len(self.fisher_calls),
            "model_open": not self.closed,
            "name": "fake",
            "one_token_calls": self.events.count("step_token"),
        }


def fake_distortions(state: torch.Tensor, geometry: Any) -> Any:
    assert tuple(state.shape) == (
        geometry.layers,
        geometry.heads,
        geometry.key_rows,
        geometry.value_width,
    )
    shape = (geometry.layers, geometry.heads, geometry.key_rows)
    return tuple(torch.full(shape, value, dtype=torch.float64) for value in (3.0, 2.0, 1.0))


def fake_fisher_distortions(
    state: torch.Tensor,
    gradient: torch.Tensor,
    geometry: Any,
) -> Any:
    assert state.shape == gradient.shape
    return fake_distortions(state, geometry)


def source_verifier(
    events: list[str],
    *,
    fail_on_call: int | None = None,
    source_commit: str | None = None,
) -> Any:
    calls = 0
    expected_source_commit = source_commit or current_head()

    def verify(expected: Mapping[str, object], root: Path) -> Any:
        nonlocal calls
        calls += 1
        events.append("verify_source")
        assert expected == {
            "manifest": "expected",
            "paths": [
                {
                    "path": runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
                    "raw_sha256": "c" * 64,
                }
            ],
            "source_commit": expected_source_commit,
        }
        assert root == SCRIPT.parents[1]
        if fail_on_call == calls:
            raise runner.CalibrationRunError("source drift")
        return {"manifest": "expected", "source_commit": expected_source_commit}, "5" * 64

    return verify


def configured_run(
    tmp_path: Path,
    *,
    records: Sequence[dict[str, object]] | None = None,
    stability_passed: bool = True,
    source_fail_on_call: int | None = None,
    decode_error: BaseException | None = None,
    source_commit: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    selected_records = list(records or [record()])
    expected_source_commit = source_commit or current_head()
    files = {
        "config.json": b"{}",
        "model.safetensors": b"safe-test-placeholder",
        "model.safetensors.index.json": b"{}",
    }
    capture_source_sha256 = "c" * 64
    source_bytes = runner.canonical_json_bytes(
        {
            "manifest": "expected",
            "paths": [
                {
                    "path": runner.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
                    "raw_sha256": capture_source_sha256,
                }
            ],
            "source_commit": expected_source_commit,
        }
    )
    model_bytes = model_manifest_bytes(files)
    runtime_bytes, provenance_origins = capture_provenance_runtime_manifest_bytes()
    parquet_bytes = b'{"artifact_kind":"test-parquet-materializations"}\n'
    frozen = identity(
        selected_records,
        source_manifest_sha256=digest(source_bytes),
        runtime_manifest_sha256=digest(runtime_bytes),
        model_manifest_sha256=digest(model_bytes),
        parquet_manifest_sha256=digest(parquet_bytes),
    )
    identity_bytes = bootstrap_identity_bytes(
        source_manifest_sha256=digest(source_bytes),
        runtime_manifest_sha256=digest(runtime_bytes),
        model_manifest_sha256=digest(model_bytes),
        parquet_manifest_sha256=digest(parquet_bytes),
        identity_input_manifest_sha256=frozen.identity_input_manifest_sha256,
    )
    provenance_bindings = runner.BootstrapIdentityBindings(
        repository_source_manifest_file_sha256=digest(source_bytes),
        runtime_manifest_file_sha256=digest(runtime_bytes),
        model_file_manifest_file_sha256=digest(model_bytes),
        parquet_materialization_manifest_file_sha256=digest(parquet_bytes),
        identity_input_manifest_sha256=frozen.identity_input_manifest_sha256,
    )
    capture_provenance_receipt_bytes = runner.canonical_json_bytes(
        capture_provenance_receipt_document(
            origins=provenance_origins,
            bindings=provenance_bindings,
            capture_source_sha256=capture_source_sha256,
            identity_input_manifest_sha256=frozen.identity_input_manifest_sha256,
            source_commit=expected_source_commit,
        )
    )
    model_root = tmp_path / "model"
    write_model_root(model_root, files)
    events: list[str] = []
    backend = FakeBackend(
        frozen,
        events,
        stability_passed=stability_passed,
        decode_error=decode_error,
        expected_source_commit=expected_source_commit,
        expected_identity_bytes=identity_bytes,
    )
    sequence_map = {
        str(item["canonical_id"]): materialized(
            item,
            tuple(range(1, int(item["sequence_length"]) + 1)),
        )
        for item in selected_records
    }
    adapter = FakeAdapter(sequence_map, events)

    def authenticate(root: Path, manifest: Any) -> Any:
        events.append("authenticate_model_files")
        return runner.authenticate_local_model_files(root, manifest, calibration_api=api)

    def authenticate_runtime(manifest: Any) -> Any:
        events.append("authenticate_runtime")
        return runner.AuthenticatedRuntime(
            manifest_file_sha256=manifest.file_sha256,
            python_implementation=manifest.python_implementation,
            python_version=manifest.python_version,
            python_cache_tag=manifest.python_cache_tag,
            interpreter_sha256=manifest.interpreter_sha256,
            git_executable_absolute_path_sha256=(manifest.git_executable_absolute_path_sha256),
            git_executable_sha256=manifest.git_executable_sha256,
            git_executable_size_bytes=manifest.git_executable_size_bytes,
            machine_name=manifest.machine_name,
            base_runtime_file_count=len(manifest.runtime_trees[0].files),
            package_root_count=len(manifest.package_roots),
            distributions=(
                ("torch", runner.CANONICAL_TORCH_DISTRIBUTION_VERSION),
                ("transformers", "5.14.1"),
            ),
            distribution_count=2,
            file_count=sum(len(tree.files) for tree in manifest.runtime_trees),
        )

    services = runner.RunnerServices(
        backend=backend,
        calibration_api=api,
        identity_resolver=identity_resolver,
        verify_repository_source=source_verifier(
            events,
            fail_on_call=source_fail_on_call,
            source_commit=expected_source_commit,
        ),
        validate_adapter=lambda _adapter: events.append("validate_adapter"),
        distortion_function=fake_distortions,
        fisher_distortion_function=fake_fisher_distortions,
        authenticate_model_files=authenticate,
        authenticate_runtime=authenticate_runtime,
    )
    config = runner.CalibrationRunConfig(
        frozen_identity_bytes=identity_bytes,
        repository_source_manifest_bytes=source_bytes,
        model_file_manifest_bytes=model_bytes,
        parquet_materialization_manifest_bytes=parquet_bytes,
        runtime_manifest_bytes=runtime_bytes,
        model_root=model_root,
        repository_root=SCRIPT.parents[1],
        expected_source_commit=expected_source_commit,
        expected_model_file_manifest_sha256=digest(model_bytes),
        expected_parquet_materialization_manifest_sha256=digest(parquet_bytes),
        expected_runtime_manifest_sha256=digest(runtime_bytes),
        capture_provenance_receipt_bytes=capture_provenance_receipt_bytes,
        expected_capture_provenance_receipt_sha256=digest(capture_provenance_receipt_bytes),
        output_dir=tmp_path / "output",
        require_cuda=False,
    )
    parsed_runtime = runner.parse_calibration_runtime_manifest(runtime_bytes)
    authenticated_runtime = authenticate_runtime(parsed_runtime)
    events.clear()
    parsed_model = runner.parse_model_file_manifest(model_bytes)
    authenticated_model = runner.authenticate_local_model_files(
        model_root,
        parsed_model,
        calibration_api=api,
    )
    first_record = frozen.records[0]
    first_token_count = int(first_record["sequence_length"])
    first_fisher_count = len(first_record["fisher_boundary"]["boundary_positions"])
    smoke_report = runner._report_bytes(
        status="fisher_h1_smoke_passed",
        identity=frozen,
        source_commit=expected_source_commit,
        source_manifest_sha256="5" * 64,
        source_manifest_file_sha256=digest(source_bytes),
        model_files=authenticated_model,
        sequence_count=1,
        token_count=first_token_count,
        post_token_anchor_count=len(runner.frozen_anchor_positions(first_token_count)),
        fisher_boundary_count=first_fisher_count,
        observed_fisher_step_count=first_fisher_count,
        stability={
            "checks": [],
            "evaluated": False,
            "passed": None,
            "scope": "smoke_only",
        },
        artifacts={},
        runtime={
            "adapter": {
                "adapter_revision": runner.CANONICAL_ADAPTER_REVISION,
                "capture_input_sha256": frozen.identity_input_manifest_sha256,
                "device": "cuda:0",
                "fisher_step_count": first_fisher_count,
                "kernel_backend": runner.CANONICAL_ADAPTER_KERNEL_BACKEND,
                "materialization_attempted": True,
                "materialized_sequence_count": len(frozen.records),
                "model_dtype": runner.CANONICAL_ADAPTER_MODEL_DTYPE,
                "model_id": parsed_model.model_id,
                "model_loaded": True,
                "model_loading_diagnostic_counts": {
                    name: 0 for name in runner.CANONICAL_ADAPTER_LOADING_DIAGNOSTICS
                },
                "model_revision": parsed_model.revision,
                "query_shape": list(runner.CANONICAL_ADAPTER_QUERY_SHAPE),
                "recurrent_layer_indices": list(runner.CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES),
                "state_shape": list(runner.CANONICAL_ADAPTER_STATE_SHAPE),
                "token_sequence_manifest_sha256": token_sequence_manifest_digest(frozen.records),
                "transformers_version": parsed_model.transformers_version,
            },
            "authenticated_distribution_count": authenticated_runtime.distribution_count,
            "authenticated_file_count": authenticated_runtime.file_count,
            "cuda_available": True,
            "cuda_runtime": runner.CANONICAL_CUDA_RUNTIME_VERSION,
            "elapsed_seconds_hex": (0.0).hex(),
            "gpu": {
                "capability": [12, 0],
                "device_index": 0,
                "name": "test-gpu",
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 2,
            },
            "packages": dict(authenticated_runtime.distributions),
            "platform": "test",
            "python": authenticated_runtime.python_version,
            "runtime_manifest_file_sha256": authenticated_runtime.manifest_file_sha256,
            "torch": runner.CANONICAL_TORCH_RUNTIME_VERSION,
        },
        capture_provenance_receipt_file_sha256=digest(capture_provenance_receipt_bytes),
        fisher_h1_smoke_report_file_sha256=None,
        fisher_h1_smoke_launch_finalization_file_sha256=None,
    )
    prior_smoke_output_dir = tmp_path / "prior-smoke-output"
    prior_smoke_output_dir.mkdir()
    prior_smoke_output_directory_absolute_path_sha256 = runner._absolute_path_sha256(
        prior_smoke_output_dir
    )
    smoke_launch_finalization = runner.canonical_json_bytes(
        {
            "artifact_kind": runner.RUN_LAUNCH_FINALIZATION_KIND,
            "capture_provenance_receipt_file_sha256": digest(capture_provenance_receipt_bytes),
            "child_output_file_sha256": {runner.FISHER_SMOKE_REPORT_FILENAME: digest(smoke_report)},
            "child_output_size_bytes": {runner.FISHER_SMOKE_REPORT_FILENAME: len(smoke_report)},
            "completion_marker_filename": runner.FISHER_SMOKE_COMPLETE_FILENAME,
            "completion_marker_sha256": digest(runner.FISHER_SMOKE_COMPLETE_BYTES),
            "execution_bindings": {
                "calibration_runtime_manifest_file_sha256": (frozen.runtime_manifest_file_sha256),
                "model_file_manifest_file_sha256": frozen.model_file_manifest_file_sha256,
                "parquet_materialization_manifest_file_sha256": (
                    frozen.parquet_materialization_manifest_file_sha256
                ),
                "repository_source_manifest_file_sha256": (
                    frozen.repository_source_manifest_file_sha256
                ),
            },
            "frozen_identity_file_sha256": frozen.file_sha256,
            "launch_policy": dict(runner.SEALED_LAUNCH_POLICY),
            "mode": "fisher_h1_smoke",
            "output_directory_absolute_path_sha256": (
                prior_smoke_output_directory_absolute_path_sha256
            ),
            "prior_fisher_h1_smoke_launch_finalization_file_sha256": None,
            "publication_contract": runner.RUN_LAUNCH_FINALIZATION_PUBLICATION_CONTRACT,
            "runner_revision": runner.RUNNER_REVISION,
            "schema_version": runner.RUN_LAUNCH_FINALIZATION_SCHEMA,
            "source_commit": expected_source_commit,
            "status": "fisher_h1_smoke_launcher_finalized",
        }
    )
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=smoke_report,
        prior_fisher_h1_smoke_complete_bytes=runner.FISHER_SMOKE_COMPLETE_BYTES,
        prior_fisher_h1_smoke_launch_finalization_bytes=smoke_launch_finalization,
        expected_prior_fisher_h1_smoke_launch_finalization_sha256=digest(smoke_launch_finalization),
        expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256=(
            prior_smoke_output_directory_absolute_path_sha256
        ),
    )
    return config, adapter, services, events


def test_success_authenticates_every_boundary_and_publishes_complete_set(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)

    result = runner.run_calibration(config, adapter, services=services)

    assert result["status"] == "passed"
    assert set(path.name for path in config.output_dir.iterdir()) == {
        runner.SCORE_FILENAME,
        runner.COMPARATOR_SCORE_FILENAME,
        runner.SPLIT_FILENAME,
        runner.K27030_FILENAME,
        runner.K29334_FILENAME,
        runner.MSE_K29334_FILENAME,
        runner.FISHER_K29334_FILENAME,
        runner.Q48_FILENAME,
        runner.CORE_BINDING_FILENAME,
        runner.REPORT_FILENAME,
    }
    assert events[0] == "decode_identity"
    assert events.index("verify_source") < events.index("validate_adapter")
    assert events.index("validate_adapter") < events.index("authenticate_runtime")
    assert events.index("authenticate_runtime") < events.index("materialize_sequence")
    assert events.index("materialize_sequence") < events.index("authenticate_model_files")
    assert events.index("authenticate_model_files") < events.index("load_model")
    assert events.count("verify_source") == 3
    assert events.count("authenticate_runtime") == 4
    assert events[-6:] == [
        "runtime_metadata",
        "close_model",
        "authenticate_model_files",
        "verify_source",
        "authenticate_runtime",
        "authenticate_model_files",
    ]
    assert adapter.closed
    report = json.loads((config.output_dir / runner.REPORT_FILENAME).read_text())
    assert report["schema_version"] == runner.RUN_REPORT_SCHEMA
    assert report["evidence"]["status"] == "passed"
    assert report["evidence"]["runner_revision"] == runner.RUNNER_REVISION
    assert report["evidence"]["calibration"] == {
        "expected_fisher_step_count": 1,
        "observed_fisher_step_count": 1,
        "fisher_boundary_count": 1,
        "post_token_anchor_count": 3,
        "sequence_count": 1,
        "token_count": 3,
    }
    assert report["evidence"]["runtime"]["adapter"]["model_open"] is True
    assert report["evidence"]["identity"]["execution_bindings"] == {
        "calibration_runtime_manifest_file_sha256": digest(config.runtime_manifest_bytes),
        "model_file_manifest_file_sha256": digest(config.model_file_manifest_bytes),
        "parquet_materialization_manifest_file_sha256": digest(
            config.parquet_materialization_manifest_bytes
        ),
        "repository_source_manifest_file_sha256": digest(config.repository_source_manifest_bytes),
    }
    assert report["evidence"]["prerequisites"] == {
        "capture_provenance_receipt_file_sha256": (
            config.expected_capture_provenance_receipt_sha256
        ),
        "fisher_h1_smoke_launch_finalization_file_sha256": digest(
            config.prior_fisher_h1_smoke_launch_finalization_bytes
        ),
        "fisher_h1_smoke_report_file_sha256": digest(config.prior_fisher_h1_smoke_report_bytes),
    }


def test_post_calibration_authorizer_requires_exact_inputs_and_publishes_v4_pair(
    tmp_path: Path,
) -> None:
    source_commit = "a" * 40
    calibration_dir = tmp_path / "calibration"
    smoke_dir = tmp_path / "smoke"
    calibration_dir.mkdir()
    smoke_dir.mkdir()
    frozen_identity = b"frozen-identity"
    receipt = b"capture-receipt"
    full_payloads = {
        runner.SCORE_FILENAME: b"score",
        runner.COMPARATOR_SCORE_FILENAME: b"comparators",
        runner.SPLIT_FILENAME: b"split",
        runner.K27030_FILENAME: b"k27030",
        runner.K29334_FILENAME: b"k29334",
        runner.MSE_K29334_FILENAME: b"mse",
        runner.FISHER_K29334_FILENAME: b"fisher",
        runner.Q48_FILENAME: b"q48",
        runner.CORE_BINDING_FILENAME: b"core-binding",
        runner.REPORT_FILENAME: b"full-report",
        runner.COMPLETE_FILENAME: runner.CALIBRATION_COMPLETE_BYTES,
        runner.RUN_LAUNCH_FINALIZATION_FILENAME: b"full-launch-finalization",
    }
    for name, payload in full_payloads.items():
        (calibration_dir / name).write_bytes(payload)
    smoke_report = b"smoke-report"
    (smoke_dir / runner.FISHER_SMOKE_REPORT_FILENAME).write_bytes(smoke_report)
    (smoke_dir / runner.FISHER_SMOKE_COMPLETE_FILENAME).write_bytes(
        runner.FISHER_SMOKE_COMPLETE_BYTES
    )
    smoke_launch_finalization = b"smoke-launch-finalization"
    (smoke_dir / runner.RUN_LAUNCH_FINALIZATION_FILENAME).write_bytes(smoke_launch_finalization)
    receipt_path = tmp_path / "capture-receipt.json"
    identity_path = tmp_path / "frozen-identity.json"
    source_manifest_path = tmp_path / "repository-source-manifest.json"
    runtime_manifest_path = tmp_path / "calibration-runtime-manifest.json"
    model_manifest_path = tmp_path / "model-file-manifest.json"
    source_manifest = b"repository-source-manifest"
    runtime_manifest = b"calibration-runtime-manifest"
    model_manifest = b"model-file-manifest"
    receipt_path.write_bytes(receipt)
    identity_path.write_bytes(frozen_identity)
    source_manifest_path.write_bytes(source_manifest)
    runtime_manifest_path.write_bytes(runtime_manifest)
    model_manifest_path.write_bytes(model_manifest)
    authorization_bytes = b"authorization"
    binding_bytes = b"authorized-binding"

    class FakeResolver:
        @staticmethod
        def deserialize_stage_a_calibration_core_binding_artifact(data: bytes) -> Any:
            assert data == full_payloads[runner.CORE_BINDING_FILENAME]
            return SimpleNamespace(
                calibration_dependencies={"frozen_identity_artifact": frozen_identity}
            )

        @staticmethod
        def build_stage_a_calibration_authorization_artifact(**kwargs: object) -> bytes:
            assert kwargs["calibration_run_report"] == full_payloads[runner.REPORT_FILENAME]
            assert (
                kwargs["calibration_run_launch_finalization"]
                == full_payloads[runner.RUN_LAUNCH_FINALIZATION_FILENAME]
            )
            assert kwargs["capture_provenance_receipt"] == receipt
            assert kwargs["fisher_h1_smoke_report"] == smoke_report
            assert kwargs["fisher_h1_smoke_launch_finalization"] == (smoke_launch_finalization)
            assert kwargs["fisher_h1_smoke_complete_marker"] == (runner.FISHER_SMOKE_COMPLETE_BYTES)
            assert kwargs["calibration_complete_marker"] == runner.CALIBRATION_COMPLETE_BYTES
            assert kwargs["repository_source_manifest"] == source_manifest
            assert kwargs["calibration_runtime_manifest"] == runtime_manifest
            assert kwargs["model_file_manifest"] == model_manifest
            assert kwargs[
                "expected_calibration_output_directory_absolute_path_sha256"
            ] == runner._absolute_path_sha256(calibration_dir)
            assert kwargs[
                "expected_fisher_h1_smoke_output_directory_absolute_path_sha256"
            ] == runner._absolute_path_sha256(smoke_dir)
            return authorization_bytes

        @staticmethod
        def deserialize_stage_a_calibration_authorization_artifact(data: bytes) -> Any:
            assert data == authorization_bytes
            return SimpleNamespace(
                source_commit=source_commit,
                authorized_output_file_sha256={
                    name: digest(full_payloads[name])
                    for name in (
                        runner.SCORE_FILENAME,
                        runner.COMPARATOR_SCORE_FILENAME,
                        runner.SPLIT_FILENAME,
                        runner.K27030_FILENAME,
                        runner.K29334_FILENAME,
                        runner.MSE_K29334_FILENAME,
                        runner.FISHER_K29334_FILENAME,
                        runner.Q48_FILENAME,
                        runner.CORE_BINDING_FILENAME,
                    )
                },
            )

        @staticmethod
        def build_stage_a_calibration_binding_artifact(**kwargs: bytes) -> bytes:
            assert kwargs == {"calibration_authorization_artifact": authorization_bytes}
            return binding_bytes

        @staticmethod
        def deserialize_stage_a_calibration_binding_artifact(data: bytes) -> Any:
            assert data == binding_bytes
            return SimpleNamespace(
                authorization_file_sha256=runner.sha256_bytes(authorization_bytes)
            )

    output_dir = tmp_path / "authorized"
    result = runner.authorize_stage_a_calibration(
        calibration_output_dir=calibration_dir,
        fisher_h1_smoke_output_dir=smoke_dir,
        capture_provenance_receipt_path=receipt_path,
        expected_capture_provenance_receipt_sha256=runner.sha256_bytes(receipt),
        frozen_identity_path=identity_path,
        expected_frozen_identity_sha256=runner.sha256_bytes(frozen_identity),
        repository_source_manifest_path=source_manifest_path,
        expected_repository_source_manifest_sha256=runner.sha256_bytes(source_manifest),
        runtime_manifest_path=runtime_manifest_path,
        expected_runtime_manifest_sha256=runner.sha256_bytes(runtime_manifest),
        model_file_manifest_path=model_manifest_path,
        expected_model_file_manifest_sha256=runner.sha256_bytes(model_manifest),
        expected_full_run_report_sha256=runner.sha256_bytes(full_payloads[runner.REPORT_FILENAME]),
        expected_calibration_run_launch_finalization_sha256=runner.sha256_bytes(
            full_payloads[runner.RUN_LAUNCH_FINALIZATION_FILENAME]
        ),
        expected_fisher_h1_smoke_report_sha256=runner.sha256_bytes(smoke_report),
        expected_fisher_h1_smoke_launch_finalization_sha256=runner.sha256_bytes(
            smoke_launch_finalization
        ),
        source_commit=source_commit,
        output_dir=output_dir,
        identity_resolver=FakeResolver(),
    )

    assert result["status"] == "authorized_for_stage_a"
    assert {item.name for item in output_dir.iterdir()} == {
        runner.AUTHORIZATION_FILENAME,
        runner.BINDING_FILENAME,
        runner.AUTHORIZATION_COMPLETE_FILENAME,
    }
    assert (output_dir / runner.AUTHORIZATION_FILENAME).read_bytes() == authorization_bytes
    assert (output_dir / runner.BINDING_FILENAME).read_bytes() == binding_bytes
    assert (output_dir / runner.AUTHORIZATION_COMPLETE_FILENAME).read_bytes() == (
        runner.AUTHORIZATION_COMPLETE_BYTES
    )

    (calibration_dir / "unexpected.txt").write_bytes(b"extra")
    with pytest.raises(runner.CalibrationRunError, match="inventory drifted"):
        runner.authorize_stage_a_calibration(
            calibration_output_dir=calibration_dir,
            fisher_h1_smoke_output_dir=smoke_dir,
            capture_provenance_receipt_path=receipt_path,
            expected_capture_provenance_receipt_sha256=runner.sha256_bytes(receipt),
            frozen_identity_path=identity_path,
            expected_frozen_identity_sha256=runner.sha256_bytes(frozen_identity),
            repository_source_manifest_path=source_manifest_path,
            expected_repository_source_manifest_sha256=runner.sha256_bytes(source_manifest),
            runtime_manifest_path=runtime_manifest_path,
            expected_runtime_manifest_sha256=runner.sha256_bytes(runtime_manifest),
            model_file_manifest_path=model_manifest_path,
            expected_model_file_manifest_sha256=runner.sha256_bytes(model_manifest),
            expected_full_run_report_sha256=runner.sha256_bytes(
                full_payloads[runner.REPORT_FILENAME]
            ),
            expected_calibration_run_launch_finalization_sha256=runner.sha256_bytes(
                full_payloads[runner.RUN_LAUNCH_FINALIZATION_FILENAME]
            ),
            expected_fisher_h1_smoke_report_sha256=runner.sha256_bytes(smoke_report),
            expected_fisher_h1_smoke_launch_finalization_sha256=runner.sha256_bytes(
                smoke_launch_finalization
            ),
            source_commit=source_commit,
            output_dir=tmp_path / "forbidden-output",
            identity_resolver=FakeResolver(),
        )


def test_post_calibration_authorizer_rejects_output_overlapping_evidence(
    tmp_path: Path,
) -> None:
    calibration_dir = tmp_path / "calibration"
    smoke_dir = tmp_path / "smoke"
    calibration_dir.mkdir()
    smoke_dir.mkdir()
    protected_files = {
        "capture_provenance_receipt_path": tmp_path / "capture.json",
        "frozen_identity_path": tmp_path / "identity.json",
        "repository_source_manifest_path": tmp_path / "source.json",
        "runtime_manifest_path": tmp_path / "runtime.json",
        "model_file_manifest_path": tmp_path / "model.json",
    }
    for path in protected_files.values():
        path.write_bytes(b"fixture")
    overlapping_output = calibration_dir / "stage-a-authorization"

    with pytest.raises(
        runner.CalibrationRunError,
        match="Stage-A authorization output overlaps authenticated calibration evidence",
    ):
        runner.authorize_stage_a_calibration(
            calibration_output_dir=calibration_dir,
            fisher_h1_smoke_output_dir=smoke_dir,
            **protected_files,
            expected_capture_provenance_receipt_sha256="1" * 64,
            expected_frozen_identity_sha256="2" * 64,
            expected_repository_source_manifest_sha256="3" * 64,
            expected_runtime_manifest_sha256="4" * 64,
            expected_model_file_manifest_sha256="5" * 64,
            expected_full_run_report_sha256="6" * 64,
            expected_calibration_run_launch_finalization_sha256="7" * 64,
            expected_fisher_h1_smoke_report_sha256="8" * 64,
            expected_fisher_h1_smoke_launch_finalization_sha256="9" * 64,
            source_commit="a" * 40,
            output_dir=overlapping_output,
            identity_resolver=SimpleNamespace(),
        )

    assert not overlapping_output.exists()


def _stage_a_authorization_path_gate_kwargs(
    tmp_path: Path,
    *,
    calibration_output_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    smoke_dir = tmp_path / "path-gate-smoke"
    smoke_dir.mkdir(exist_ok=True)
    protected_files = {
        "capture_provenance_receipt_path": tmp_path / "path-gate-capture.json",
        "frozen_identity_path": tmp_path / "path-gate-identity.json",
        "repository_source_manifest_path": tmp_path / "path-gate-source.json",
        "runtime_manifest_path": tmp_path / "path-gate-runtime.json",
        "model_file_manifest_path": tmp_path / "path-gate-model.json",
    }
    for path in protected_files.values():
        path.write_bytes(b"path-gate-fixture")
    return {
        "calibration_output_dir": calibration_output_dir,
        "fisher_h1_smoke_output_dir": smoke_dir,
        **protected_files,
        "expected_capture_provenance_receipt_sha256": "1" * 64,
        "expected_frozen_identity_sha256": "2" * 64,
        "expected_repository_source_manifest_sha256": "3" * 64,
        "expected_runtime_manifest_sha256": "4" * 64,
        "expected_model_file_manifest_sha256": "5" * 64,
        "expected_full_run_report_sha256": "6" * 64,
        "expected_calibration_run_launch_finalization_sha256": "7" * 64,
        "expected_fisher_h1_smoke_report_sha256": "8" * 64,
        "expected_fisher_h1_smoke_launch_finalization_sha256": "9" * 64,
        "source_commit": "a" * 40,
        "output_dir": output_dir,
        "identity_resolver": SimpleNamespace(),
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path spellings only")
@pytest.mark.parametrize("spelling", ["extended", "device", "unc"])
def test_post_calibration_authorizer_rejects_nonordinary_windows_output_spelling(
    tmp_path: Path,
    spelling: str,
) -> None:
    drive = tmp_path.drive
    assert len(drive) == 2 and drive[1] == ":"
    if spelling == "extended":
        output_dir = Path("\\\\?\\" + drive + "\\recurquant-stage-a-output")
    elif spelling == "device":
        output_dir = Path("\\\\.\\" + drive + "\\recurquant-stage-a-output")
    else:
        output_dir = Path("\\\\localhost\\" + drive[0] + "$\\recurquant-stage-a-output")

    with pytest.raises(
        runner.CalibrationRunError,
        match="must use an ordinary local-drive absolute path",
    ):
        runner.authorize_stage_a_calibration(
            **_stage_a_authorization_path_gate_kwargs(
                tmp_path,
                calibration_output_dir=tmp_path,
                output_dir=output_dir,
            )
        )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows admin-share aliases only")
def test_post_calibration_authorizer_rejects_unc_alias_of_local_evidence_parent(
    tmp_path: Path,
) -> None:
    evidence_parent = tmp_path / "unc-aliased-evidence"
    evidence_parent.mkdir()
    local_parent = evidence_parent.resolve(strict=True)
    drive = local_parent.drive
    assert len(drive) == 2 and drive[1] == ":"
    relative = local_parent.relative_to(Path(drive + "\\"))
    unc_parent = Path("\\\\localhost\\" + drive[0] + "$\\" + str(relative))
    try:
        unc_resolved = unc_parent.resolve(strict=True)
        local_status = local_parent.stat()
        unc_status = unc_resolved.stat()
    except OSError:
        pytest.skip("local administrative share is unavailable")
    if (local_status.st_dev, local_status.st_ino) != (
        unc_status.st_dev,
        unc_status.st_ino,
    ):
        pytest.skip("administrative share does not expose the local directory identity")

    output_dir = local_parent / "stage-a-authorization"
    destination, _identities = runner._normalized_new_output_destination(
        output_dir,
        context="test Stage-A authorization output",
    )
    assert not runner._paths_overlap(destination, unc_resolved)

    with pytest.raises(
        runner.CalibrationRunError,
        match="Stage-A authorization output overlaps authenticated calibration evidence",
    ):
        runner.authorize_stage_a_calibration(
            **_stage_a_authorization_path_gate_kwargs(
                tmp_path,
                calibration_output_dir=unc_parent,
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_authorization_resolver_path_swap_executes_no_unauthenticated_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    resolver_path = repository / runner.IDENTITY_RESOLVER_SOURCE_PATH
    resolver_path.parent.mkdir(parents=True)
    authenticated_bytes = b"AUTHENTICATED_SENTINEL = 'exact-buffer'\n"
    side_effect = tmp_path / "unauthenticated-resolver-side-effect.txt"
    malicious_bytes = (
        "from pathlib import Path\n"
        f"Path({str(side_effect)!r}).write_text('executed', encoding='utf-8')\n"
    ).encode()
    resolver_path.write_bytes(authenticated_bytes)
    bootstrap = runner.BootstrapSource(
        manifest={},
        source_commit="a" * 40,
        entries={
            runner.IDENTITY_RESOLVER_SOURCE_PATH: {
                "raw_sha256": digest(authenticated_bytes),
            }
        },
    )
    stable_read = runner._read_stable_regular_bytes

    def read_then_swap(path: Path, *, context: str) -> bytes:
        payload = stable_read(path, context=context)
        if Path(path) == resolver_path:
            resolver_path.write_bytes(malicious_bytes)
        return payload

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runner, "_read_stable_regular_bytes", read_then_swap)

    with pytest.raises(runner.CalibrationRunError, match="identity drifted on import"):
        runner._load_authorization_identity_resolver(bootstrap)

    assert not side_effect.exists()
    assert "_recurquant_experiment013_identity_resolver_for_authorization" not in sys.modules


def test_forged_authorization_resolver_is_rejected_before_output_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "a" * 40
    full = {
        name: f"fixture-{name}".encode()
        for name in {
            runner.SCORE_FILENAME,
            runner.COMPARATOR_SCORE_FILENAME,
            runner.SPLIT_FILENAME,
            runner.K27030_FILENAME,
            runner.K29334_FILENAME,
            runner.MSE_K29334_FILENAME,
            runner.FISHER_K29334_FILENAME,
            runner.Q48_FILENAME,
            runner.CORE_BINDING_FILENAME,
            runner.REPORT_FILENAME,
            runner.COMPLETE_FILENAME,
            runner.RUN_LAUNCH_FINALIZATION_FILENAME,
        }
    }
    smoke = {
        runner.FISHER_SMOKE_REPORT_FILENAME: b"smoke-report",
        runner.FISHER_SMOKE_COMPLETE_FILENAME: runner.FISHER_SMOKE_COMPLETE_BYTES,
        runner.RUN_LAUNCH_FINALIZATION_FILENAME: b"smoke-launch-finalization",
    }
    receipt = b"capture-receipt"
    frozen_identity = b"frozen-identity"
    source_manifest = b"source-manifest"
    runtime_manifest = b"runtime-manifest"
    model_manifest = b"model-manifest"
    (tmp_path / "calibration").mkdir()
    (tmp_path / "smoke").mkdir()
    path_payloads = {
        tmp_path / "receipt.json": receipt,
        tmp_path / "identity.json": frozen_identity,
        tmp_path / "source.json": source_manifest,
        tmp_path / "runtime.json": runtime_manifest,
        tmp_path / "model.json": model_manifest,
    }
    for path, payload in path_payloads.items():
        path.write_bytes(payload)
    stable_read = runner._read_stable_regular_bytes

    def read_fixture_or_source(path: Path, *, context: str) -> bytes:
        fixture = path_payloads.get(Path(path))
        if fixture is not None:
            return fixture
        return stable_read(path, context=context)

    def read_directory(
        _path: Path,
        *,
        expected_filenames: set[str],
        context: str,
    ) -> dict[str, bytes]:
        payloads = smoke if "smoke" in context.casefold() else full
        assert set(payloads) == expected_filenames
        return dict(payloads)

    forged_bootstrap = runner.BootstrapSource(
        manifest={},
        source_commit=source_commit,
        entries={
            runner.IDENTITY_RESOLVER_SOURCE_PATH: {
                "raw_sha256": "0" * 64,
            }
        },
    )
    monkeypatch.setattr(runner, "_read_stable_regular_bytes", read_fixture_or_source)
    monkeypatch.setattr(runner, "_read_exact_regular_directory", read_directory)
    monkeypatch.setattr(
        runner,
        "_bootstrap_source_manifest",
        lambda *_args, **_kwargs: forged_bootstrap,
    )
    output_dir = tmp_path / "authorization-output"

    with pytest.raises(runner.CalibrationRunError, match="bytes drifted before import"):
        runner.authorize_stage_a_calibration(
            calibration_output_dir=tmp_path / "calibration",
            fisher_h1_smoke_output_dir=tmp_path / "smoke",
            capture_provenance_receipt_path=tmp_path / "receipt.json",
            expected_capture_provenance_receipt_sha256=digest(receipt),
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256=digest(frozen_identity),
            repository_source_manifest_path=tmp_path / "source.json",
            expected_repository_source_manifest_sha256=digest(source_manifest),
            runtime_manifest_path=tmp_path / "runtime.json",
            expected_runtime_manifest_sha256=digest(runtime_manifest),
            model_file_manifest_path=tmp_path / "model.json",
            expected_model_file_manifest_sha256=digest(model_manifest),
            expected_full_run_report_sha256=digest(full[runner.REPORT_FILENAME]),
            expected_calibration_run_launch_finalization_sha256=digest(
                full[runner.RUN_LAUNCH_FINALIZATION_FILENAME]
            ),
            expected_fisher_h1_smoke_report_sha256=digest(
                smoke[runner.FISHER_SMOKE_REPORT_FILENAME]
            ),
            expected_fisher_h1_smoke_launch_finalization_sha256=digest(
                smoke[runner.RUN_LAUNCH_FINALIZATION_FILENAME]
            ),
            source_commit=source_commit,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_post_calibration_authorizer_cli_forwards_all_authenticated_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_authorize(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"status": "authorized_for_stage_a"}

    monkeypatch.setattr(runner, "authorize_stage_a_calibration", fake_authorize)
    paths = {
        "calibration_output_dir": tmp_path / "calibration",
        "fisher_h1_smoke_output_dir": tmp_path / "smoke",
        "capture_provenance_receipt_path": tmp_path / "capture.json",
        "frozen_identity_path": tmp_path / "identity.json",
        "repository_source_manifest_path": tmp_path / "source.json",
        "runtime_manifest_path": tmp_path / "runtime.json",
        "model_file_manifest_path": tmp_path / "model.json",
        "output_dir": tmp_path / "authorization",
    }
    arguments = [
        "authorize-stage-a-calibration",
        "--calibration-output-dir",
        str(paths["calibration_output_dir"]),
        "--fisher-h1-smoke-output-dir",
        str(paths["fisher_h1_smoke_output_dir"]),
        "--capture-provenance-receipt",
        str(paths["capture_provenance_receipt_path"]),
        "--expected-capture-provenance-receipt-sha256",
        "1" * 64,
        "--frozen-identity",
        str(paths["frozen_identity_path"]),
        "--expected-frozen-identity-sha256",
        "2" * 64,
        "--repository-source-manifest",
        str(paths["repository_source_manifest_path"]),
        "--expected-repository-source-manifest-sha256",
        "3" * 64,
        "--runtime-manifest",
        str(paths["runtime_manifest_path"]),
        "--expected-runtime-manifest-sha256",
        "4" * 64,
        "--model-file-manifest",
        str(paths["model_file_manifest_path"]),
        "--expected-model-file-manifest-sha256",
        "5" * 64,
        "--expected-full-run-report-sha256",
        "6" * 64,
        "--expected-calibration-run-launch-finalization-sha256",
        "9" * 64,
        "--expected-fisher-h1-smoke-report-sha256",
        "7" * 64,
        "--expected-fisher-h1-smoke-launch-finalization-sha256",
        "a" * 64,
        "--source-commit",
        "8" * 40,
        "--output-dir",
        str(paths["output_dir"]),
    ]

    assert runner._capture_manifest_mode(arguments) == 0

    assert received == {
        **paths,
        "expected_capture_provenance_receipt_sha256": "1" * 64,
        "expected_frozen_identity_sha256": "2" * 64,
        "expected_repository_source_manifest_sha256": "3" * 64,
        "expected_runtime_manifest_sha256": "4" * 64,
        "expected_model_file_manifest_sha256": "5" * 64,
        "expected_full_run_report_sha256": "6" * 64,
        "expected_calibration_run_launch_finalization_sha256": "9" * 64,
        "expected_fisher_h1_smoke_report_sha256": "7" * 64,
        "expected_fisher_h1_smoke_launch_finalization_sha256": "a" * 64,
        "source_commit": "8" * 40,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "authorized_for_stage_a"}


def test_fisher_h1_smoke_runs_first_frozen_sequence_and_publishes_only_receipt(
    tmp_path: Path,
) -> None:
    first = record()
    second = record(canonical_id="item-2")
    config, adapter, services, events = configured_run(
        tmp_path,
        records=[first, second],
    )
    config = replace(
        config,
        fisher_h1_smoke=True,
        prior_fisher_h1_smoke_report_bytes=None,
        prior_fisher_h1_smoke_complete_bytes=None,
        prior_fisher_h1_smoke_launch_finalization_bytes=None,
        expected_prior_fisher_h1_smoke_launch_finalization_sha256=None,
        expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256=None,
    )

    result = runner.run_calibration(config, adapter, services=services)

    assert result["status"] == "fisher_h1_smoke_passed"
    assert result["sequence_count"] == 1
    assert result["token_count"] == 3
    assert result["fisher_boundary_count"] == 1
    assert events.count("materialize_sequence") == 2
    assert events.count("reduce_sequence") == 1
    assert "finalize" not in events
    assert adapter.closed
    assert {path.name for path in config.output_dir.iterdir()} == {
        runner.FISHER_SMOKE_REPORT_FILENAME,
    }
    report = json.loads((config.output_dir / runner.FISHER_SMOKE_REPORT_FILENAME).read_text())
    assert report["schema_version"] == runner.RUN_REPORT_SCHEMA
    assert report["evidence"]["status"] == "fisher_h1_smoke_passed"
    assert report["evidence"]["runner_revision"] == runner.RUNNER_REVISION
    assert report["evidence"]["calibration"] == {
        "expected_fisher_step_count": 1,
        "observed_fisher_step_count": 1,
        "fisher_boundary_count": 1,
        "post_token_anchor_count": 3,
        "sequence_count": 1,
        "token_count": 3,
    }
    assert report["evidence"]["artifacts"] == {}
    assert report["evidence"]["prerequisites"] == {
        "capture_provenance_receipt_file_sha256": (
            config.expected_capture_provenance_receipt_sha256
        ),
        "fisher_h1_smoke_launch_finalization_file_sha256": None,
        "fisher_h1_smoke_report_file_sha256": None,
    }


@pytest.mark.parametrize("fisher_h1_smoke", [False, True], ids=["full", "smoke"])
def test_smoke_and_full_require_semantic_capture_provenance_before_identity_decode(
    tmp_path: Path,
    fisher_h1_smoke: bool,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    malformed_document = json.loads(config.capture_provenance_receipt_bytes)
    malformed_document["status"] = "captured_before_launcher_finalization"
    malformed_receipt = runner.canonical_json_bytes(malformed_document)
    config = replace(
        config,
        capture_provenance_receipt_bytes=malformed_receipt,
        expected_capture_provenance_receipt_sha256=digest(malformed_receipt),
        fisher_h1_smoke=fisher_h1_smoke,
        prior_fisher_h1_smoke_report_bytes=(
            None if fisher_h1_smoke else config.prior_fisher_h1_smoke_report_bytes
        ),
        prior_fisher_h1_smoke_complete_bytes=(
            None if fisher_h1_smoke else config.prior_fisher_h1_smoke_complete_bytes
        ),
        prior_fisher_h1_smoke_launch_finalization_bytes=(
            None if fisher_h1_smoke else config.prior_fisher_h1_smoke_launch_finalization_bytes
        ),
        expected_prior_fisher_h1_smoke_launch_finalization_sha256=(
            None
            if fisher_h1_smoke
            else config.expected_prior_fisher_h1_smoke_launch_finalization_sha256
        ),
        expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256=(
            None
            if fisher_h1_smoke
            else config.expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256
        ),
    )

    with pytest.raises(runner.CalibrationRunError, match="capture provenance receipt"):
        runner.run_calibration(config, adapter, services=services)

    assert events == []
    assert not config.output_dir.exists()


def test_full_calibration_requires_authenticated_prior_fisher_smoke_before_data(
    tmp_path: Path,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=None,
        prior_fisher_h1_smoke_complete_bytes=None,
        prior_fisher_h1_smoke_launch_finalization_bytes=None,
        expected_prior_fisher_h1_smoke_launch_finalization_sha256=None,
        expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256=None,
    )

    with pytest.raises(runner.CalibrationRunError, match="requires the prior Fisher H=1"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity"]
    assert not config.output_dir.exists()


def test_smoke_mode_forbids_prior_launcher_finalization_before_data(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(
        config,
        fisher_h1_smoke=True,
        prior_fisher_h1_smoke_report_bytes=None,
        prior_fisher_h1_smoke_complete_bytes=None,
    )

    with pytest.raises(runner.CalibrationRunError, match="forbids a prior smoke prerequisite"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity"]
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("status", "child_only_passed"),
        ("completion_marker_sha256", "0" * 64),
        ("prior_fisher_h1_smoke_launch_finalization_file_sha256", "0" * 64),
        ("launch_policy", {}),
    ],
)
def test_full_calibration_rejects_rehashed_smoke_launch_finalization_mutation_before_data(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    document = json.loads(config.prior_fisher_h1_smoke_launch_finalization_bytes)
    document[field] = replacement
    mutated = runner.canonical_json_bytes(document)
    config = replace(
        config,
        prior_fisher_h1_smoke_launch_finalization_bytes=mutated,
        expected_prior_fisher_h1_smoke_launch_finalization_sha256=digest(mutated),
    )

    with pytest.raises(runner.CalibrationRunError, match="launch finalization drifted"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "authenticate_model_files" not in events
    assert "load_model" not in events
    assert not config.output_dir.exists()


def test_full_calibration_rejects_smoke_finalization_copied_to_another_path_before_data(
    tmp_path: Path,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(
        config,
        expected_prior_fisher_h1_smoke_output_directory_absolute_path_sha256=digest(
            b"copied-prior-smoke-output-directory"
        ),
    )

    with pytest.raises(runner.CalibrationRunError, match="launch finalization drifted"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "authenticate_model_files" not in events
    assert "load_model" not in events
    assert not config.output_dir.exists()


def test_full_calibration_rejects_rehashed_smoke_from_another_identity_before_data(
    tmp_path: Path,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    document = json.loads(config.prior_fisher_h1_smoke_report_bytes)
    document["evidence"]["identity"]["file_sha256"] = "0" * 64
    document["canonical_evidence_sha256"] = digest(
        runner.canonical_json_bytes(document["evidence"])
    )
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=runner.canonical_json_bytes(document),
    )

    with pytest.raises(runner.CalibrationRunError, match="smoke identity receipt"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "authenticate_model_files" not in events
    assert "load_model" not in events


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("runtime", "elapsed_seconds_hex"), "not-a-float"),
        (("runtime", "elapsed_seconds_hex"), "0x1p999999999"),
        (("runtime", "forged"), True),
        (("runtime", "gpu"), {}),
        (("runtime", "gpu", "name"), ""),
        (("runtime", "gpu", "peak_allocated_bytes"), -1),
        (("runtime", "gpu", "peak_reserved_bytes"), 0),
        (("runtime", "adapter", "forged"), True),
        (("runtime", "adapter", "capture_input_sha256"), "invalid"),
        (("runtime", "adapter", "capture_input_sha256"), "0" * 64),
        (("runtime", "adapter", "fisher_step_count"), True),
        (("runtime", "adapter", "materialization_attempted"), False),
        (("runtime", "adapter", "materialized_sequence_count"), 0),
        (("runtime", "adapter", "model_id"), "forged/model"),
        (("runtime", "adapter", "model_loaded"), False),
        (("runtime", "adapter", "token_sequence_manifest_sha256"), "0" * 64),
        (
            ("runtime", "adapter", "model_loading_diagnostic_counts", "missing_keys"),
            1,
        ),
        (("prerequisites", "capture_provenance_receipt_file_sha256"), "0" * 64),
    ],
)
def test_full_calibration_rejects_rehashed_malformed_smoke_evidence_before_data(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    document = json.loads(config.prior_fisher_h1_smoke_report_bytes)
    target: dict[str, Any] = document["evidence"]
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = replacement
    document["canonical_evidence_sha256"] = digest(
        runner.canonical_json_bytes(document["evidence"])
    )
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=runner.canonical_json_bytes(document),
    )

    with pytest.raises(runner.CalibrationRunError, match="Fisher H=1 smoke"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "load_model" not in events
    assert not config.output_dir.exists()


def test_authenticated_unchanged_descendant_retains_h0_provenance(tmp_path: Path) -> None:
    h0 = "1" * 40
    config, adapter, services, _events = configured_run(tmp_path, source_commit=h0)

    runner.run_calibration(config, adapter, services=services)

    report = json.loads((config.output_dir / runner.REPORT_FILENAME).read_text())
    assert report["evidence"]["repository"]["source_commit"] == h0
    assert report["evidence"]["repository"]["source_commit"] != current_head()


def test_identity_view_consumes_schema_v5_bindings_and_preserves_fisher_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {
        "calibration_runtime_manifest_file_sha256": "1" * 64,
        "model_file_manifest_file_sha256": "2" * 64,
        "parquet_materialization_manifest_file_sha256": "9" * 64,
        "repository_source_manifest_file_sha256": "3" * 64,
    }
    evidence_record = record()
    boundary = evidence_record["fisher_boundary"]
    assert isinstance(boundary, dict)
    evidence = {
        "execution_bindings": bindings,
        "model_contracts": {"primary": {"id": "example/model", "revision": "4" * 40}},
        "records": [evidence_record],
        "schema_version": runner.FROZEN_IDENTITY_SCHEMA_VERSION,
        "source_manifest_sha256": "5" * 64,
        "tokenizer": {"transformers_version": "5.14.1"},
    }
    payload = runner.canonical_json_bytes({"evidence": evidence})
    decoded_artifact = identity_resolver.FrozenCalibrationIdentityArtifact(
        file_sha256=digest(payload),
        canonical_evidence_sha256="6" * 64,
        records=(evidence_record,),
        assignment=(),
        assignment_sha256="7" * 64,
        tokenizer_manifest_sha256="8" * 64,
        parquet_materialization_manifest_file_sha256="9" * 64,
        execution_bindings=bindings,
    )
    frozen_boundary = decoded_artifact.records[0]["fisher_boundary"]
    assert isinstance(frozen_boundary, Mapping)
    assert not isinstance(frozen_boundary["boundary_positions"], list)

    class Resolver:
        @staticmethod
        def deserialize_frozen_calibration_identity_artifact(
            data: bytes,
        ) -> identity_resolver.FrozenCalibrationIdentityArtifact:
            assert data == payload
            return decoded_artifact

    monkeypatch.setattr(runner, "_load_identity_resolver", lambda _root: Resolver())
    decoded = runner._identity_view(payload, tmp_path)

    assert decoded.repository_source_manifest_file_sha256 == "3" * 64
    assert decoded.runtime_manifest_file_sha256 == "1" * 64
    assert decoded.model_file_manifest_file_sha256 == "2" * 64
    assert decoded.parquet_materialization_manifest_file_sha256 == "9" * 64
    assert decoded.records[0]["fisher_boundary"] == boundary
    assert decoded.records[0]["fisher_boundary"] is not boundary
    for name in ("boundary_positions", "input_positions", "target_positions"):
        assert isinstance(decoded.records[0]["fisher_boundary"][name], list)
    assert isinstance(decoded.records[0]["token_span"], Mapping)
    from recurquant.static_q468_calibration import identity_record_sha256

    assert identity_record_sha256(decoded.records[0]) == evidence_record["identity_record_sha256"]


@pytest.mark.parametrize(
    "invalid_positions",
    ["0", b"0", [], [True], [-1]],
    ids=("text", "bytes", "empty", "boolean", "negative"),
)
def test_identity_view_rejects_invalid_fisher_boundary_position_sequences(
    invalid_positions: object,
) -> None:
    item = {
        "canonical_id": "item-1",
        "fisher_boundary": fisher_boundary_contract(),
        "sequence_length": 3,
    }
    item["fisher_boundary"]["boundary_positions"] = invalid_positions

    class Decoded:
        records = (item,)

    with pytest.raises(
        runner.CalibrationRunError,
        match=r"fisher_boundary boundary_positions is invalid",
    ):
        runner._identity_records_with_fisher_boundary(Decoded(), {"records": [item]})


def test_identity_view_rejects_resolver_record_that_drops_fisher_boundary() -> None:
    bindings = {
        "calibration_runtime_manifest_file_sha256": "1" * 64,
        "model_file_manifest_file_sha256": "2" * 64,
        "parquet_materialization_manifest_file_sha256": "9" * 64,
        "repository_source_manifest_file_sha256": "3" * 64,
    }
    evidence_record = {
        "canonical_id": "item-1",
        "fisher_boundary": fisher_boundary_contract(),
        "sequence_length": 3,
    }
    evidence = {
        "execution_bindings": bindings,
        "model_contracts": {"primary": {"id": "example/model", "revision": "4" * 40}},
        "records": [evidence_record],
        "schema_version": runner.FROZEN_IDENTITY_SCHEMA_VERSION,
        "source_manifest_sha256": "5" * 64,
        "tokenizer": {"transformers_version": "5.14.1"},
    }
    payload = runner.canonical_json_bytes({"evidence": evidence})

    class Decoded:
        file_sha256 = digest(payload)
        canonical_evidence_sha256 = "6" * 64
        records = ({"canonical_id": "item-1"},)
        assignment = ()
        assignment_sha256 = "7" * 64
        tokenizer_manifest_sha256 = "8" * 64
        execution_bindings = bindings

    class Resolver:
        @staticmethod
        def deserialize_frozen_calibration_identity_artifact(data: bytes) -> Decoded:
            assert data == payload
            return Decoded()

    with pytest.raises(runner.CalibrationRunError, match="fisher_boundary"):
        runner._identity_view_from_resolver(payload, Resolver())


def test_failed_stability_publishes_only_report_and_never_binding(tmp_path: Path) -> None:
    config, adapter, services, _events = configured_run(tmp_path, stability_passed=False)

    with pytest.raises(runner.CalibrationStabilityFailure, match="stability gate failed"):
        runner.run_calibration(config, adapter, services=services)

    assert {path.name for path in config.output_dir.iterdir()} == {
        runner.REPORT_FILENAME,
    }
    report = json.loads((config.output_dir / runner.REPORT_FILENAME).read_text())
    assert report["evidence"]["status"] == "stability_failed"
    assert report["evidence"]["artifacts"] == {}
    assert adapter.closed


def test_identity_decode_is_first_and_failure_touches_no_other_boundary(tmp_path: Path) -> None:
    error = ValueError("identity invalid")
    config, _adapter, services, events = configured_run(tmp_path, decode_error=error)
    config = replace(
        config,
        repository_root=tmp_path / "does-not-exist",
        model_root=tmp_path / "also-missing",
    )

    with pytest.raises(ValueError, match="identity invalid"):
        runner.run_calibration(config, FakeAdapter({}, events), services=services)

    assert events == ["decode_identity"]
    assert not config.output_dir.exists()


def test_existing_output_stops_after_identity_before_source_or_data(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config.output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity"]


def test_source_drift_stops_before_dataset_and_model_access(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path, source_fail_on_call=1)

    with pytest.raises(runner.CalibrationRunError, match="source drift"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity", "verify_source"]
    assert not config.output_dir.exists()


def test_source_manifest_commit_must_equal_requested_frozen_commit(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)

    def wrong_commit(expected: Mapping[str, object], root: Path) -> Any:
        del root
        events.append("verify_source")
        return {**expected, "source_commit": "0" * 40}, "5" * 64

    services = replace(services, verify_repository_source=wrong_commit)
    with pytest.raises(runner.CalibrationRunError, match="must equal"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events


def test_source_manifest_exact_bytes_are_identity_bound_before_verifier(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(config, repository_source_manifest_bytes=b'{"manifest":"changed"}\n')

    with pytest.raises(
        runner.CalibrationRunError, match="capture provenance differs from identity"
    ):
        runner.run_calibration(config, adapter, services=services)

    assert events == []


def test_materialized_token_mismatch_stops_before_model_file_open_or_load(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    item = services.backend.frozen_identity.records[0]
    adapter.sequence_by_id["item-1"] = materialized(item, (9, 9, 9))
    config.model_root.joinpath("model.safetensors").unlink()

    with pytest.raises(runner.CalibrationRunError, match="token IDs differ"):
        runner.run_calibration(config, adapter, services=services)

    assert "authenticate_model_files" not in events
    assert "load_model" not in events


def test_materialized_fisher_token_hash_mismatch_reaches_exact_boundary_check(
    tmp_path: Path,
) -> None:
    item = record()
    boundary = dict(item["fisher_boundary"])
    boundary["input_token_ids_sha256"] = "0" * 64
    boundary_without_hash = {
        name: value for name, value in boundary.items() if name != "fisher_boundary_sha256"
    }
    boundary["fisher_boundary_sha256"] = digest(
        identity_resolver.FISHER_BOUNDARY_NAMESPACE.encode("utf-8")
        + runner.canonical_json_bytes(boundary_without_hash)
    )
    item["fisher_boundary"] = boundary
    config, adapter, services, events = configured_run(tmp_path, records=[item])

    with pytest.raises(runner.CalibrationRunError, match="Fisher input/target tokens"):
        runner.run_calibration(config, adapter, services=services)

    assert "authenticate_model_files" not in events
    assert "load_model" not in events


def test_model_manifest_commitment_is_checked_before_data_access(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(config, expected_model_file_manifest_sha256="9" * 64)

    with pytest.raises(runner.CalibrationRunError, match="identity/config binding"):
        runner.run_calibration(config, adapter, services=services)

    assert "validate_adapter" in events
    assert "materialize_sequence" not in events
    assert "authenticate_model_files" not in events
    assert "load_model" not in events


def test_parquet_manifest_commitment_is_checked_before_data_access(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(
        config,
        expected_parquet_materialization_manifest_sha256="9" * 64,
    )

    with pytest.raises(runner.CalibrationRunError, match="parquet materialization manifest"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "authenticate_model_files" not in events
    assert "load_model" not in events


def test_runtime_manifest_commitment_and_authentication_precede_data(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(config, expected_runtime_manifest_sha256="9" * 64)

    with pytest.raises(
        runner.CalibrationRunError,
        match="capture provenance runtime manifest differs from identity/CLI binding",
    ):
        runner.run_calibration(config, adapter, services=services)

    assert "decode_identity" not in events
    assert "authenticate_runtime" not in events
    assert "materialize_sequence" not in events
    assert "load_model" not in events


def test_point_used_runtime_drift_stops_before_data_access(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    original = services.authenticate_runtime
    calls = 0

    def drifting(manifest: Any) -> Any:
        nonlocal calls
        calls += 1
        authenticated = original(manifest)
        return replace(authenticated, machine_name="drifted") if calls == 2 else authenticated

    services = replace(services, authenticate_runtime=drifting)
    with pytest.raises(runner.CalibrationRunError, match="before data access"):
        runner.run_calibration(config, adapter, services=services)

    assert "materialize_sequence" not in events
    assert "load_model" not in events


def test_empty_calibration_target_is_hash_checked_and_allowed() -> None:
    token_ids = (10, 11, 12)
    item = record(token_ids)
    span = dict(item["token_span"])
    span["prefill_stop"] = len(token_ids)
    span["scored_start"] = len(token_ids)
    item["token_span"] = span
    item["prompt_token_ids_sha256"] = token_digest(token_ids)
    item["target_token_ids_sha256"] = token_digest(())
    candidate = materialized(item, token_ids)

    assert (
        runner.validate_materialized_sequence(
            item,
            candidate,
            calibration_api=api,
            identity_resolver=identity_resolver,
        )
        == token_ids
    )


def test_exact_local_model_file_mismatch_stops_before_model_load(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config.model_root.joinpath("model.safetensors").write_bytes(b"tampered")

    with pytest.raises(runner.CalibrationRunError, match="authentication failed"):
        runner.run_calibration(config, adapter, services=services)

    assert "authenticate_model_files" in events
    assert "load_model" not in events


def test_model_files_are_reauthenticated_immediately_after_load(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    original_load = adapter.load_model

    def mutating_load(authenticated: Any) -> object:
        model = original_load(authenticated)
        config.model_root.joinpath("model.safetensors").write_bytes(b"changed-after-auth")
        return model

    adapter.load_model = mutating_load  # type: ignore[method-assign]
    with pytest.raises(runner.CalibrationRunError, match="authentication failed"):
        runner.run_calibration(config, adapter, services=services)

    assert "load_model" in events
    assert "step_token" not in events
    assert adapter.closed


def test_model_authentication_rejects_reparse_or_symlink_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "model.safetensors.index.json": b"{}",
    }
    root = tmp_path / "model"
    write_model_root(root, files)
    manifest = runner.parse_model_file_manifest(model_manifest_bytes(files))
    real_check = runner._is_link_or_reparse

    def selected_link(path: Path) -> bool:
        return path.name == "model.safetensors" or real_check(path)

    monkeypatch.setattr(runner, "_is_link_or_reparse", selected_link)
    with pytest.raises(runner.CalibrationRunError, match="link or reparse"):
        runner.authenticate_local_model_files(root, manifest, calibration_api=api)


def test_second_source_verification_stops_before_model_file_authentication(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path, source_fail_on_call=2)

    with pytest.raises(runner.CalibrationRunError, match="source drift"):
        runner.run_calibration(config, adapter, services=services)

    assert events.count("materialize_sequence") == 1
    assert "authenticate_model_files" not in events
    assert "load_model" not in events


def test_bad_kernel_receipt_closes_model_and_publishes_nothing(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    adapter.invalid_kernel_receipt = True

    with pytest.raises(runner.CalibrationRunError, match="one successful causal kernel"):
        runner.run_calibration(config, adapter, services=services)

    assert events[-2:] == ["end_sequence", "close_model"]
    assert adapter.closed
    assert not config.output_dir.exists()


def test_capture_uses_frozen_ema_and_only_requests_state_at_anchors() -> None:
    geometry = runner.Geometry(layer_indices=(0,), heads=1, key_rows=4, value_width=4)
    tokens = tuple(range(17))
    item = record(tokens)
    events: list[str] = []

    class QueryAdapter(FakeAdapter):
        def step_token(
            self,
            model: object,
            *,
            token_id: int,
            position: int,
            capture_state: bool,
        ) -> Any:
            del model
            self.capture_flags.append(capture_state)
            query = torch.zeros(1, 1, 4)
            query[..., position % 4] = 1.0
            return api.StepObservation(
                position=position,
                token_id=token_id,
                layer_indices=(0,),
                recurrence_query=query,
                recurrent_state=torch.ones(1, 1, 4, 4) if capture_state else None,
                successful_kernel_calls_per_layer=(1,),
            )

    adapter = QueryAdapter({}, events)
    adapter.geometry = geometry
    captured = runner.capture_sequence_causally(
        adapter,
        object(),
        item,
        tokens,
        geometry=geometry,
        calibration_api=api,
        require_cuda=False,
        distortion_function=fake_distortions,
        fisher_distortion_function=fake_fisher_distortions,
    )

    assert captured.anchor_positions == runner.frozen_anchor_positions(17)
    assert captured.fisher_boundary_positions == runner.frozen_anchor_positions(15)
    assert len(adapter.capture_flags) == len(tokens)
    assert events.count("step_token_with_fisher") == len(captured.fisher_boundary_positions)
    assert adapter.capture_flags[15] is False
    assert sum(adapter.capture_flags) == 16
    first_energy = torch.tensor(
        [1.0 / (1.0 + runner.QUERY_ENERGY_EPSILON), 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    expected = runner.QUERY_EMA_DECAY * torch.full((4,), 0.25, dtype=torch.float64)
    expected += (1.0 - runner.QUERY_EMA_DECAY) * first_energy
    torch.testing.assert_close(captured.query_energy[0, 0, 0], expected, rtol=1e-6, atol=1e-8)
    assert captured.query_energy.dtype == torch.float64
    assert captured.q4_mse.shape == (16, 1, 1, 4)
    assert captured.fisher_q4_risk.shape == (15, 1, 1, 4)
    assert captured.fisher_target_nlls.shape == (15,)
    assert captured.fisher_target_nlls.dtype == torch.float64


def test_three_token_capture_uses_one_exact_h1_fisher_step_and_three_forwards() -> None:
    tokens = (10, 11, 12)
    events: list[str] = []

    class CountingModel:
        def __init__(self) -> None:
            self.forward_count = 0

        def forward(self) -> None:
            self.forward_count += 1

    class CountingAdapter(FakeAdapter):
        def step_token(self, model: object, **kwargs: object) -> Any:
            assert isinstance(model, CountingModel)
            model.forward()
            return super().step_token(model, **kwargs)

    adapter = CountingAdapter({}, events)
    model = CountingModel()

    captured = runner.capture_sequence_causally(
        adapter,
        model,
        record(tokens),
        tokens,
        geometry=FakeBackend.geometry,
        calibration_api=api,
        require_cuda=False,
        distortion_function=fake_distortions,
        fisher_distortion_function=fake_fisher_distortions,
    )

    assert captured.fisher_boundary_positions == (0,)
    assert events.count("step_token") == len(tokens)
    assert events.count("step_token_with_fisher") == 1
    assert adapter.fisher_calls == [(1, 11, 12)]
    assert len(adapter.capture_flags) == len(tokens)
    assert model.forward_count == len(tokens)


def test_official_capture_rejects_cpu_queries_even_when_cuda_might_exist() -> None:
    geometry = FakeBackend.geometry
    tokens = (1, 2, 3)
    adapter = FakeAdapter({}, [])

    with pytest.raises(runner.CalibrationRunError, match="actual CUDA"):
        runner.capture_sequence_causally(
            adapter,
            object(),
            record(tokens),
            tokens,
            geometry=geometry,
            calibration_api=api,
            require_cuda=True,
            distortion_function=fake_distortions,
            fisher_distortion_function=fake_fisher_distortions,
        )


def test_capture_rejects_non_fp32_reference_state() -> None:
    geometry = runner.Geometry(layer_indices=(0,), heads=1, key_rows=4, value_width=4)

    class Bf16Adapter(FakeAdapter):
        def step_token(self, model: object, **kwargs: object) -> Any:
            del model
            return api.StepObservation(
                position=kwargs["position"],
                token_id=kwargs["token_id"],
                layer_indices=(0,),
                recurrence_query=torch.ones(1, 1, 4),
                recurrent_state=torch.ones(1, 1, 4, 4, dtype=torch.bfloat16),
                successful_kernel_calls_per_layer=(1,),
            )

    adapter = Bf16Adapter({}, [])
    adapter.geometry = geometry
    with pytest.raises(runner.CalibrationRunError, match="must be FP32"):
        runner.capture_sequence_causally(
            adapter,
            object(),
            record((1, 2, 3)),
            (1, 2, 3),
            geometry=geometry,
            calibration_api=api,
            require_cuda=False,
            distortion_function=fake_distortions,
            fisher_distortion_function=fake_fisher_distortions,
        )


def test_compute_anchor_distortions_is_cpu_fp64_and_precision_ordered() -> None:
    geometry = runner.Geometry(layer_indices=(3,), heads=1, key_rows=2, value_width=4)
    state = torch.tensor(
        [[[[1.2, -0.1, 0.4, 2.3], [0.2, 0.7, -1.4, 0.5]]]],
        dtype=torch.float32,
    )

    d4, d6, d8 = runner.compute_anchor_distortions(state, geometry)

    for tensor in (d4, d6, d8):
        assert tensor.shape == (1, 1, 2)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert torch.isfinite(tensor).all()
        assert (tensor >= 0).all()
    assert torch.all(d8 <= d4)


def test_compute_fisher_distortions_delegates_exact_cpu_fp64_endpoint_math() -> None:
    from recurquant.static_q468 import StaticRhtQ468Geometry
    from recurquant.static_q468_calibration import (
        compute_rht_diagonal_empirical_fisher_h1_endpoints,
    )

    geometry = runner.Geometry(layer_indices=(3,), heads=1, key_rows=2, value_width=4)
    source = torch.tensor(
        [[[[1.2, -0.1, 0.4, 2.3], [0.2, 0.7, -1.4, 0.5]]]],
        dtype=torch.float32,
    )
    gradient = torch.tensor(
        [[[[0.3, -0.2, 0.1, 0.4], [0.5, 0.1, -0.3, 0.2]]]],
        dtype=torch.float32,
    )
    endpoint_geometry = StaticRhtQ468Geometry(
        layer_indices=geometry.layer_indices,
        heads=geometry.heads,
        key_rows=geometry.key_rows,
        value_width=geometry.value_width,
        target_resident_bytes=1,
    )

    actual = runner.compute_fisher_distortions(source, gradient, geometry)
    expected = compute_rht_diagonal_empirical_fisher_h1_endpoints(
        source,
        gradient,
        geometry=endpoint_geometry,
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.device.type == "cpu"
        assert actual_tensor.dtype == torch.float64
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    with pytest.raises(runner.CalibrationRunError, match="source state/gradient"):
        runner.compute_fisher_distortions(source, gradient[..., :3], geometry)


def test_model_manifest_rejects_noncanonical_duplicate_and_traversal() -> None:
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "model.safetensors.index.json": b"{}",
    }
    valid = model_manifest_bytes(files)
    parsed = runner.parse_model_file_manifest(valid)
    assert parsed.file_sha256 == digest(valid)
    assert [item.name for item in parsed.files] == [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ]

    with pytest.raises(ValueError, match="canonical"):
        runner.parse_model_file_manifest(valid.rstrip(b"\n") + b" \n")
    duplicate = valid.replace(
        b'"model_id":"example/model"', b'"model_id":"x","model_id":"example/model"'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.parse_model_file_manifest(duplicate)
    payload = json.loads(valid)
    payload["files"][0]["name"] = "../config.json"
    with pytest.raises(ValueError, match="name is invalid|relative POSIX"):
        runner.parse_model_file_manifest(runner.canonical_json_bytes(payload))


def test_model_manifest_enforces_exact_selection_profile_and_portable_names() -> None:
    required = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "model.safetensors.index.json": b"{}",
    }

    with pytest.raises(ValueError, match="case-insensitive collision"):
        runner.parse_model_file_manifest(
            model_manifest_bytes({**required, "CONFIG.JSON": b"other"})
        )
    with pytest.raises(ValueError, match="selection profile"):
        runner.parse_model_file_manifest(model_manifest_bytes({**required, "unbound.json": b"x"}))
    with pytest.raises(ValueError, match="index"):
        runner.parse_model_file_manifest(
            model_manifest_bytes({"config.json": b"{}", "model.safetensors": b"weights"})
        )


def hub_tree_entries() -> list[dict[str, object]]:
    return [
        {
            "blob_id": "1" * 40,
            "lfs": None,
            "path": "config.json",
            "size": 123,
        },
        {
            "blob_id": "2" * 40,
            "lfs": {"sha256": "a" * 64, "size": 1_024},
            "path": "model.safetensors",
            "size": 1_024,
        },
        {
            "blob_id": "3" * 40,
            "lfs": None,
            "path": "model.safetensors.index.json",
            "size": 321,
        },
        {
            "blob_id": "4" * 40,
            "lfs": None,
            "path": "README.md",
            "size": 50,
        },
    ]


def test_model_manifest_capture_uses_only_pinned_hub_tree_lfs_metadata() -> None:
    revision = "4" * 40
    payload = runner.capture_model_file_manifest_from_hub(
        "example/model",
        revision,
        transformers_version="5.14.1",
        tree_entries=hub_tree_entries(),
        resolved_revision=revision,
    )
    parsed = runner.parse_model_file_manifest(payload)

    assert [item.name for item in parsed.files] == [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ]
    assert parsed.files[0].sha256 is None
    assert parsed.files[0].git_blob_oid == "1" * 40
    assert parsed.files[1].sha256 == "a" * 64
    assert parsed.files[1].lfs_sha256 == "a" * 64
    assert b"README.md" not in payload


def test_model_manifest_accepts_the_pinned_qwen35_weight_filename() -> None:
    revision = "4" * 40
    weight_name = "model.safetensors-00001-of-00001.safetensors"
    entries = [
        hub_tree_entries()[0],
        {
            "blob_id": "2" * 40,
            "lfs": {"sha256": "a" * 64, "size": 1_024},
            "path": weight_name,
            "size": 1_024,
        },
        {
            "blob_id": "3" * 40,
            "lfs": None,
            "path": "model.safetensors.index.json",
            "size": 321,
        },
    ]

    payload = runner.capture_model_file_manifest_from_hub(
        "Qwen/Qwen3.5-0.8B-Base",
        revision,
        transformers_version="5.14.1",
        tree_entries=entries,
        resolved_revision=revision,
    )
    parsed = runner.parse_model_file_manifest(payload)

    assert [item.name for item in parsed.files] == [
        "config.json",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    ]
    assert parsed.files[1].lfs_sha256 == "a" * 64


def test_model_manifest_capture_calls_only_metadata_api_surfaces() -> None:
    revision = "4" * 40
    calls: list[str] = []

    class Info:
        sha = revision

    class Api:
        @staticmethod
        def model_info(*args: object, **kwargs: object) -> Info:
            del args
            assert kwargs["token"] is False
            calls.append("model_info")
            return Info()

        @staticmethod
        def list_repo_tree(*args: object, **kwargs: object) -> list[dict[str, object]]:
            del args
            assert kwargs["token"] is False
            calls.append("list_repo_tree")
            return hub_tree_entries()

    runner.capture_model_file_manifest_from_hub(
        "example/model",
        revision,
        transformers_version="5.14.1",
        api=Api(),
    )

    assert calls == ["model_info", "list_repo_tree"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda entries: entries.append(dict(entries[0])),
            "duplicate path",
        ),
        (
            lambda entries: entries.append(
                {"blob_id": "5" * 40, "lfs": None, "path": "../escape", "size": 1}
            ),
            "repository-relative",
        ),
        (
            lambda entries: entries[1].update({"lfs": None}),
            "lacks pinned LFS",
        ),
        (
            lambda entries: entries[1].update({"lfs": {"sha256": "a" * 64, "size": 999}}),
            "differs from LFS",
        ),
    ],
)
def test_model_manifest_capture_rejects_malformed_hub_metadata(
    mutator: Any,
    message: str,
) -> None:
    entries = hub_tree_entries()
    mutator(entries)

    with pytest.raises(ValueError, match=message):
        runner.capture_model_file_manifest_from_hub(
            "example/model",
            "4" * 40,
            transformers_version="5.14.1",
            tree_entries=entries,
            resolved_revision="4" * 40,
        )


def test_model_manifest_parser_detects_tree_metadata_tamper() -> None:
    payload = runner.capture_model_file_manifest_from_hub(
        "example/model",
        "4" * 40,
        transformers_version="5.14.1",
        tree_entries=hub_tree_entries(),
        resolved_revision="4" * 40,
    )
    document = json.loads(payload)
    document["files"][0]["git_blob_oid"] = "9" * 40

    with pytest.raises(ValueError, match="tree metadata manifest"):
        runner.parse_model_file_manifest(runner.canonical_json_bytes(document))


def test_model_authentication_rejects_extra_files(tmp_path: Path) -> None:
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "model.safetensors.index.json": b"{}",
    }
    root = tmp_path / "model"
    write_model_root(root, {**files, "unbound.txt": b"extra"})
    manifest = runner.parse_model_file_manifest(model_manifest_bytes(files))

    with pytest.raises(runner.CalibrationRunError, match="file set differs"):
        runner.authenticate_local_model_files(root, manifest, calibration_api=api)


def test_stage_model_auth_failure_does_not_import_download_or_write(
    tmp_path: Path,
) -> None:
    candidate = runner.canonical_json_bytes(
        {
            "canonical_evidence_sha256": "0" * 64,
            "evidence": {
                "identity_only": True,
                "phase": "calibration",
                "promotion_required": True,
                "schema_version": runner.FROZEN_IDENTITY_SCHEMA_VERSION,
                "status": "candidate",
            },
        }
    )
    identity_path = tmp_path / "candidate.json"
    identity_path.write_bytes(candidate)
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_marker = cache / "preexisting.txt"
    cache_marker.write_text("untouched", encoding="utf-8")
    output = tmp_path / "model"
    calls: list[dict[str, object]] = []

    with pytest.raises(runner.CalibrationRunError, match="state"):
        runner.stage_identity_bound_model(
            frozen_identity_path=identity_path,
            expected_frozen_identity_sha256=digest(candidate),
            identity_commit="3" * 40,
            repository_root=SCRIPT.parents[1],
            repository_source_manifest_path=tmp_path / "missing-source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "missing-model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=(
                model_staging_path_contract_sha256(SCRIPT.parents[1], cache, output)
            ),
            hub_cache_root=cache,
            output_root=output,
            downloader=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert cache_marker.read_text(encoding="utf-8") == "untouched"
    assert list(cache.iterdir()) == [cache_marker]
    assert not output.exists()


def test_verify_frozen_identity_contract_is_deterministic_read_only_and_exactly_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = frozen_identity_source_authorization()
    git_executable = runner.AuthenticatedGitExecutable(
        path=tmp_path / "git.exe",
        absolute_path_sha256="a" * 64,
        sha256="b" * 64,
        size_bytes=1,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_authenticate_git_executable", lambda _path: git_executable)

    def authenticate(**kwargs: object) -> runner.FrozenIdentitySourceAuthorization:
        calls.append(dict(kwargs))
        return authorization

    monkeypatch.setattr(runner, "_authenticate_frozen_identity_source_contract", authenticate)
    monkeypatch.setattr(
        runner,
        "_read_stable_regular_bytes",
        lambda _path, *, context: (
            b"authenticated-source" if "source manifest" in context else pytest.fail(context)
        ),
    )
    provenance_calls: list[dict[str, object]] = []

    def authenticate_provenance(**kwargs: object) -> str:
        provenance_calls.append(dict(kwargs))
        return "5" * 64

    monkeypatch.setattr(
        runner,
        "_authenticate_calibration_identity_capture_provenance",
        authenticate_provenance,
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("frozen-identity preflight imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    untouched = tmp_path / "untouched"
    arguments = {
        "git_executable_path": tmp_path / "requested-git.exe",
        "frozen_identity_path": untouched / "identity.json",
        "expected_frozen_identity_sha256": "d" * 64,
        "repository_root": untouched / "repository",
        "repository_source_manifest_path": untouched / "source.json",
        "source_commit": "1" * 40,
        **capture_provenance_gate_kwargs(untouched),
    }

    first = runner.verify_frozen_identity_contract(**arguments)
    second = runner.verify_frozen_identity_contract(**arguments)

    expected = {
        "artifact_kind": runner.FROZEN_IDENTITY_CONTRACT_KIND,
        "assignment_sha256": "f" * 64,
        "canonical_evidence_sha256": "e" * 64,
        "capture_provenance_receipt_file_sha256": "5" * 64,
        "execution_bindings": {
            "calibration_runtime_manifest_file_sha256": "8" * 64,
            "model_file_manifest_file_sha256": "9" * 64,
            "parquet_materialization_manifest_file_sha256": "a" * 64,
            "repository_source_manifest_file_sha256": "7" * 64,
        },
        "frozen_identity_file_sha256": "d" * 64,
        "git_executable": {
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "identity_input_manifest_sha256": "1" * 64,
        "model_id": "example/model",
        "model_revision": "2" * 40,
        "record_count": 0,
        "runner_revision": runner.RUNNER_REVISION,
        "schema_version": runner.FROZEN_IDENTITY_CONTRACT_SCHEMA,
        "source_commit": "1" * 40,
        "status": "verified_frozen_identity_contract",
        "tokenizer_manifest_sha256": "c" * 64,
        "transformers_version": "5.14.1",
    }
    assert first == expected
    assert runner.canonical_json_bytes(first) == runner.canonical_json_bytes(second)
    assert str(tmp_path).encode("utf-8") not in runner.canonical_json_bytes(first)
    expected_call = {
        "expected_frozen_identity_sha256": "d" * 64,
        "frozen_identity_path": untouched / "identity.json",
        "git_executable": git_executable,
        "repository_root": untouched / "repository",
        "repository_source_manifest_path": untouched / "source.json",
        "source_commit": "1" * 40,
    }
    assert calls == [expected_call, expected_call]
    assert len(provenance_calls) == 2
    assert all(
        call["expected_identity_input_sha256"] == "1" * 64
        and call["expected_receipt_sha256"] == "5" * 64
        and call["expected_runtime_manifest_sha256"] == "8" * 64
        for call in provenance_calls
    )
    assert not untouched.exists()


def test_verify_frozen_identity_contract_failure_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    untouched = tmp_path / "untouched"
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("failed frozen-identity preflight imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: runner.AuthenticatedGitExecutable(
            path=tmp_path / "git.exe",
            absolute_path_sha256="a" * 64,
            sha256="b" * 64,
            size_bytes=1,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_frozen_identity_source_contract",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.CalibrationRunError("injected frozen-identity authorization failure")
        ),
    )

    with pytest.raises(
        runner.CalibrationRunError,
        match="injected frozen-identity authorization failure",
    ):
        runner.verify_frozen_identity_contract(
            git_executable_path=tmp_path / "git.exe",
            frozen_identity_path=untouched / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            repository_root=untouched / "repository",
            repository_source_manifest_path=untouched / "source.json",
            source_commit="1" * 40,
            **capture_provenance_gate_kwargs(untouched),
        )

    assert not untouched.exists()


def test_frozen_identity_source_contract_consumes_real_resolver_frozen_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest_bytes = b"authenticated-source-manifest"
    source_manifest_sha256 = digest(source_manifest_bytes)
    bindings = {
        "calibration_runtime_manifest_file_sha256": "1" * 64,
        "model_file_manifest_file_sha256": "2" * 64,
        "parquet_materialization_manifest_file_sha256": "9" * 64,
        "repository_source_manifest_file_sha256": source_manifest_sha256,
    }
    first_boundary = fisher_boundary_contract()
    second_boundary = fisher_boundary_contract((1, 2, 3, 4))
    evidence_records = [
        {
            "canonical_id": "item-1",
            "fisher_boundary": first_boundary,
            "sequence_length": 3,
        },
        {
            "canonical_id": "item-2",
            "fisher_boundary": second_boundary,
            "sequence_length": 4,
        },
    ]
    evidence = {
        "execution_bindings": bindings,
        "identity_only": True,
        "model_contracts": {"primary": {"id": "example/model", "revision": "4" * 40}},
        "phase": "calibration",
        "promotion_required": False,
        "records": evidence_records,
        "schema_version": runner.FROZEN_IDENTITY_SCHEMA_VERSION,
        "source_manifest_sha256": "5" * 64,
        "status": "frozen",
        "tokenizer": {"transformers_version": "5.14.1"},
    }
    identity_bytes = runner.canonical_json_bytes(
        {
            "canonical_evidence_sha256": digest(runner.canonical_json_bytes(evidence)),
            "evidence": evidence,
        }
    )
    identity_sha256 = digest(identity_bytes)
    decoded_artifact = identity_resolver.FrozenCalibrationIdentityArtifact(
        file_sha256=identity_sha256,
        canonical_evidence_sha256=digest(runner.canonical_json_bytes(evidence)),
        records=tuple(evidence_records),
        assignment=(),
        assignment_sha256="7" * 64,
        tokenizer_manifest_sha256="8" * 64,
        parquet_materialization_manifest_file_sha256="9" * 64,
        execution_bindings=bindings,
    )
    for item in decoded_artifact.records:
        frozen_boundary = item["fisher_boundary"]
        assert isinstance(frozen_boundary, Mapping)
        assert not isinstance(frozen_boundary["boundary_positions"], list)

    class Resolver:
        @staticmethod
        def deserialize_frozen_calibration_identity_artifact(
            data: bytes,
            *,
            expected_file_sha256: str | None = None,
        ) -> identity_resolver.FrozenCalibrationIdentityArtifact:
            assert data == identity_bytes
            assert expected_file_sha256 == identity_sha256
            return decoded_artifact

    source_commit = "a" * 40
    source_manifest = {"source_commit": source_commit}

    class SourceVerifier:
        @staticmethod
        def verify_experiment013_source_manifest(
            manifest: Mapping[str, object],
            *,
            repo_root: Path,
            git_executable: Path,
        ) -> Mapping[str, object]:
            assert manifest == source_manifest
            assert repo_root == tmp_path / "repository"
            assert git_executable == tmp_path / "git.exe"
            return manifest

    identity_path = tmp_path / "identity.json"
    source_path = tmp_path / "source.json"
    identity_path.write_bytes(identity_bytes)
    source_path.write_bytes(source_manifest_bytes)
    bootstrap = runner.BootstrapSource(
        manifest=source_manifest,
        source_commit=source_commit,
        entries={
            runner.SOURCE_VERIFIER_PATH: {"raw_sha256": "a" * 64},
            runner.IDENTITY_RESOLVER_SOURCE_PATH: {"raw_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        runner,
        "_bootstrap_source_manifest",
        lambda data, *, repository_root, require_adapter: (
            bootstrap
            if data == source_manifest_bytes
            and repository_root == tmp_path / "repository"
            and require_adapter is False
            else pytest.fail("source bootstrap inputs drifted")
        ),
    )

    def load_module(
        _module_name: str,
        relative_path: str,
        *,
        repository_root: Path,
        entry: Mapping[str, object],
    ) -> object:
        assert repository_root == tmp_path / "repository"
        assert entry == bootstrap.entries[relative_path]
        if relative_path == runner.SOURCE_VERIFIER_PATH:
            return SourceVerifier()
        if relative_path == runner.IDENTITY_RESOLVER_SOURCE_PATH:
            return Resolver()
        pytest.fail(f"unexpected authenticated source module: {relative_path}")

    monkeypatch.setattr(runner, "_load_exact_source_module", load_module)
    authorization = runner._authenticate_frozen_identity_source_contract(
        git_executable=runner.AuthenticatedGitExecutable(
            path=tmp_path / "git.exe",
            absolute_path_sha256="a" * 64,
            sha256="b" * 64,
            size_bytes=1,
        ),
        frozen_identity_path=identity_path,
        expected_frozen_identity_sha256=identity_sha256,
        repository_root=tmp_path / "repository",
        repository_source_manifest_path=source_path,
        source_commit=source_commit,
    )

    assert len(authorization.identity.records) == 2
    for index, expected_boundary in enumerate((first_boundary, second_boundary)):
        assert authorization.identity.records[index]["fisher_boundary"] == expected_boundary
        for name in ("boundary_positions", "input_positions", "target_positions"):
            assert isinstance(
                authorization.identity.records[index]["fisher_boundary"][name],
                list,
            )


@pytest.mark.parametrize(
    "field",
    [
        "repository_source_manifest_file_sha256",
        "runtime_manifest_file_sha256",
        "model_file_manifest_file_sha256",
        "parquet_materialization_manifest_file_sha256",
    ],
)
def test_frozen_identity_source_contract_rejects_each_full_binding_mismatch(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_bytes = b"identity"
    source_manifest_bytes = b"source"
    bindings = runner.BootstrapIdentityBindings(
        repository_source_manifest_file_sha256=digest(source_manifest_bytes),
        runtime_manifest_file_sha256="1" * 64,
        model_file_manifest_file_sha256="2" * 64,
        parquet_materialization_manifest_file_sha256="3" * 64,
        identity_input_manifest_sha256="1" * 64,
    )
    decoded_identity = identity(
        (),
        source_manifest_sha256=bindings.repository_source_manifest_file_sha256,
        runtime_manifest_sha256=bindings.runtime_manifest_file_sha256,
        model_manifest_sha256=bindings.model_file_manifest_file_sha256,
        parquet_manifest_sha256=bindings.parquet_materialization_manifest_file_sha256,
    )
    decoded_identity = replace(decoded_identity, **{field: "f" * 64})
    source_commit = "a" * 40
    source_manifest = {"source_commit": source_commit}
    bootstrap = runner.BootstrapSource(
        manifest=source_manifest,
        source_commit=source_commit,
        entries={
            runner.SOURCE_VERIFIER_PATH: {},
            runner.IDENTITY_RESOLVER_SOURCE_PATH: {},
        },
    )

    class SourceVerifier:
        @staticmethod
        def verify_experiment013_source_manifest(
            manifest: Mapping[str, object],
            *,
            repo_root: Path,
            git_executable: Path,
        ) -> Mapping[str, object]:
            del repo_root, git_executable
            return manifest

    monkeypatch.setattr(
        runner,
        "_read_stable_regular_bytes",
        lambda path, *, context: (
            identity_bytes if context == "frozen identity" else source_manifest_bytes
        ),
    )
    monkeypatch.setattr(runner, "_bootstrap_identity_bindings", lambda _data: bindings)
    monkeypatch.setattr(runner, "_bootstrap_source_manifest", lambda *_args, **_kwargs: bootstrap)
    monkeypatch.setattr(
        runner,
        "_load_exact_source_module",
        lambda _module_name, relative_path, **_kwargs: (
            SourceVerifier() if relative_path == runner.SOURCE_VERIFIER_PATH else object()
        ),
    )
    monkeypatch.setattr(
        runner,
        "_identity_view_from_resolver",
        lambda *_args, **_kwargs: decoded_identity,
    )

    with pytest.raises(
        runner.CalibrationRunError,
        match="full frozen identity differs from its bootstrap bindings",
    ):
        runner._authenticate_frozen_identity_source_contract(
            git_executable=runner.AuthenticatedGitExecutable(
                path=tmp_path / "git.exe",
                absolute_path_sha256="a" * 64,
                sha256="b" * 64,
                size_bytes=1,
            ),
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256=digest(identity_bytes),
            repository_root=tmp_path / "repository",
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit=source_commit,
        )


def test_model_staging_authorization_reuses_frozen_identity_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_bytes = model_manifest_bytes(staged_model_files())
    source_authorization = frozen_identity_source_authorization()
    source_authorization = replace(
        source_authorization,
        identity=identity((), model_manifest_sha256=digest(model_bytes)),
        bindings=replace(
            source_authorization.bindings,
            model_file_manifest_file_sha256=digest(model_bytes),
        ),
    )
    git_executable = runner.AuthenticatedGitExecutable(
        path=tmp_path / "git.exe",
        absolute_path_sha256="a" * 64,
        sha256="b" * 64,
        size_bytes=1,
    )
    common_calls: list[dict[str, object]] = []

    def authenticate_common(**kwargs: object) -> runner.FrozenIdentitySourceAuthorization:
        common_calls.append(dict(kwargs))
        return source_authorization

    monkeypatch.setattr(
        runner,
        "_authenticate_frozen_identity_source_contract",
        authenticate_common,
    )

    def verify_commit(
        received_git: runner.AuthenticatedGitExecutable,
        repository_root: Path,
        frozen_identity_path: Path,
        identity_bytes: bytes,
        *,
        identity_commit: str,
    ) -> str:
        assert received_git == git_executable
        assert repository_root == tmp_path / "repository"
        assert frozen_identity_path == tmp_path / "identity.json"
        assert identity_bytes == b"frozen-identity"
        assert identity_commit == "3" * 40
        return identity_commit

    monkeypatch.setattr(runner, "_verify_committed_frozen_identity", verify_commit)
    monkeypatch.setattr(
        runner,
        "_read_stable_regular_bytes",
        lambda path, *, context: (
            model_bytes
            if path == tmp_path / "model.json" and context == "model file manifest"
            else b"authenticated-source"
            if path == tmp_path / "source.json"
            and context == "repository source manifest for capture provenance"
            else pytest.fail("unexpected stable read outside common authorization")
        ),
    )
    provenance_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "_authenticate_calibration_identity_capture_provenance",
        lambda **kwargs: provenance_calls.append(dict(kwargs)) or "5" * 64,
    )
    result = runner._authenticate_model_staging_authorization(
        git_executable=git_executable,
        frozen_identity_path=tmp_path / "identity.json",
        expected_frozen_identity_sha256="d" * 64,
        identity_commit="3" * 40,
        repository_root=tmp_path / "repository",
        repository_source_manifest_path=tmp_path / "source.json",
        source_commit="1" * 40,
        model_file_manifest_path=tmp_path / "model.json",
        expected_model_file_manifest_sha256=digest(model_bytes),
        **capture_provenance_gate_kwargs(tmp_path),
    )

    assert result.identity == source_authorization.identity
    assert result.identity_commit == "3" * 40
    assert result.capture_provenance_receipt_file_sha256 == "5" * 64
    assert len(provenance_calls) == 1
    assert common_calls == [
        {
            "expected_frozen_identity_sha256": "d" * 64,
            "frozen_identity_path": tmp_path / "identity.json",
            "git_executable": git_executable,
            "repository_root": tmp_path / "repository",
            "repository_source_manifest_path": tmp_path / "source.json",
            "source_commit": "1" * 40,
        }
    ]


def test_verify_model_staging_authorization_is_deterministic_read_only_and_exactly_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = model_staging_authorization()
    git_executable = runner.AuthenticatedGitExecutable(
        path=tmp_path / "git.exe",
        absolute_path_sha256="a" * 64,
        sha256="b" * 64,
        size_bytes=1,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_authenticate_git_executable", lambda _path: git_executable)

    def authenticate(**kwargs: object) -> runner.ModelStagingAuthorization:
        calls.append(dict(kwargs))
        return authorization

    monkeypatch.setattr(runner, "_authenticate_model_staging_authorization", authenticate)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("authorization preflight imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    untouched = tmp_path / "untouched"
    arguments = {
        "git_executable_path": tmp_path / "requested-git.exe",
        "frozen_identity_path": untouched / "identity.json",
        "expected_frozen_identity_sha256": "d" * 64,
        "identity_commit": "3" * 40,
        "repository_root": untouched / "repository",
        "repository_source_manifest_path": untouched / "source.json",
        "source_commit": "1" * 40,
        "model_file_manifest_path": untouched / "model.json",
        "expected_model_file_manifest_sha256": authorization.model_manifest.file_sha256,
        **capture_provenance_gate_kwargs(untouched),
    }

    first = runner.verify_identity_bound_model_staging_authorization(**arguments)
    second = runner.verify_identity_bound_model_staging_authorization(**arguments)

    expected = {
        "artifact_kind": runner.MODEL_STAGING_AUTHORIZATION_KIND,
        "capture_provenance_receipt_file_sha256": "5" * 64,
        "file_count": len(authorization.model_manifest.files),
        "frozen_identity_file_sha256": "d" * 64,
        "hub_tree_manifest_sha256": authorization.model_manifest.hub_tree_manifest_sha256,
        "identity_commit": "3" * 40,
        "model_id": "example/model",
        "model_manifest_file_sha256": authorization.model_manifest.file_sha256,
        "revision": "2" * 40,
        "repository_source_manifest_file_sha256": "7" * 64,
        "runner_revision": runner.RUNNER_REVISION,
        "schema_version": runner.MODEL_STAGING_AUTHORIZATION_SCHEMA,
        "source_commit": "1" * 40,
        "status": "verified_identity_bound_model_staging_authorization",
        "total_size_bytes": sum(item.size_bytes for item in authorization.model_manifest.files),
    }
    assert first == expected
    assert runner.canonical_json_bytes(first) == runner.canonical_json_bytes(second)
    assert calls == [
        {
            "expected_frozen_identity_sha256": "d" * 64,
            "expected_model_file_manifest_sha256": authorization.model_manifest.file_sha256,
            "frozen_identity_path": untouched / "identity.json",
            "git_executable": git_executable,
            "identity_commit": "3" * 40,
            "model_file_manifest_path": untouched / "model.json",
            "repository_root": untouched / "repository",
            "repository_source_manifest_path": untouched / "source.json",
            "source_commit": "1" * 40,
            **capture_provenance_gate_kwargs(untouched),
        },
        {
            "expected_frozen_identity_sha256": "d" * 64,
            "expected_model_file_manifest_sha256": authorization.model_manifest.file_sha256,
            "frozen_identity_path": untouched / "identity.json",
            "git_executable": git_executable,
            "identity_commit": "3" * 40,
            "model_file_manifest_path": untouched / "model.json",
            "repository_root": untouched / "repository",
            "repository_source_manifest_path": untouched / "source.json",
            "source_commit": "1" * 40,
            **capture_provenance_gate_kwargs(untouched),
        },
    ]
    assert not untouched.exists()


def test_verify_model_staging_authorization_failure_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    untouched = tmp_path / "untouched"
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("failed authorization preflight imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: runner.AuthenticatedGitExecutable(
            path=tmp_path / "git.exe",
            absolute_path_sha256="a" * 64,
            sha256="b" * 64,
            size_bytes=1,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.CalibrationRunError("injected authorization failure")
        ),
    )

    with pytest.raises(runner.CalibrationRunError, match="injected authorization failure"):
        runner.verify_identity_bound_model_staging_authorization(
            git_executable_path=tmp_path / "git.exe",
            frozen_identity_path=untouched / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=untouched / "repository",
            repository_source_manifest_path=untouched / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=untouched / "model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(untouched),
        )

    assert not untouched.exists()


def test_committed_frozen_identity_requires_exact_head_index_and_worktree_blob(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    identity_path = repository / "evidence" / "identity.json"
    identity_path.parent.mkdir()
    identity_bytes = b'{"status":"frozen"}\n'
    identity_path.write_bytes(identity_bytes)
    subprocess.run(["git", "add", "--", "evidence/identity.json"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "freeze identity"],
        cwd=repository,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        runner._verify_committed_frozen_identity(
            runner._authenticate_git_executable(authenticated_git_path()),
            repository,
            identity_path,
            identity_bytes,
            identity_commit=head,
        )
        == head
    )
    identity_path.write_bytes(b'{"status":"changed"}\n')
    with pytest.raises(runner.CalibrationRunError, match="changed|differ"):
        runner._verify_committed_frozen_identity(
            runner._authenticate_git_executable(authenticated_git_path()),
            repository,
            identity_path,
            identity_bytes,
            identity_commit=head,
        )


def test_model_staging_path_preflight_is_bound_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "hub-cache"
    repository.mkdir()
    cache.mkdir()
    marker = cache / "preexisting.txt"
    marker.write_text("untouched", encoding="utf-8")
    output = tmp_path / "published-model"

    first = runner.verify_model_staging_paths(
        repository_root=repository,
        hub_cache_root=cache,
        output_root=output,
    )
    second = runner.verify_model_staging_paths(
        repository_root=repository,
        hub_cache_root=cache,
        output_root=output,
    )
    paths = runner._validate_model_staging_roots(
        repository_root=repository,
        hub_cache_root=cache,
        output_root=output,
    )

    assert set(first) == {
        "artifact_kind",
        "hub_cache_component_identities_sha256",
        "hub_cache_root_absolute_path_sha256",
        "hub_cache_root_state",
        "output_parent_absolute_path_sha256",
        "output_parent_component_identities_sha256",
        "output_parent_state",
        "output_root_absolute_path_sha256",
        "output_root_state",
        "path_contract_sha256",
        "repository_component_identities_sha256",
        "repository_root_absolute_path_sha256",
        "repository_root_state",
        "runner_revision",
        "schema_version",
        "status",
    }
    assert first == {
        "artifact_kind": runner.MODEL_STAGING_PATHS_KIND,
        "hub_cache_component_identities_sha256": (
            runner._directory_component_identities_sha256(paths.hub_cache_component_identities)
        ),
        "hub_cache_root_absolute_path_sha256": runner._normalized_absolute_path_sha256(
            cache.resolve()
        ),
        "hub_cache_root_state": "existing_regular_non_link_directory",
        "output_parent_absolute_path_sha256": runner._normalized_absolute_path_sha256(
            output.parent.resolve()
        ),
        "output_parent_component_identities_sha256": (
            runner._directory_component_identities_sha256(paths.output_parent_component_identities)
        ),
        "output_parent_state": "existing_regular_non_link_directory",
        "output_root_absolute_path_sha256": runner._normalized_absolute_path_sha256(
            output.parent.resolve() / output.name
        ),
        "output_root_state": "absent",
        "path_contract_sha256": runner._model_staging_path_contract_sha256(paths),
        "repository_component_identities_sha256": (
            runner._directory_component_identities_sha256(paths.repository_component_identities)
        ),
        "repository_root_absolute_path_sha256": runner._normalized_absolute_path_sha256(
            repository.resolve()
        ),
        "repository_root_state": "existing_regular_non_link_directory",
        "runner_revision": runner.RUNNER_REVISION,
        "schema_version": runner.MODEL_STAGING_PATHS_SCHEMA,
        "status": "verified_model_staging_paths",
    }
    assert runner.canonical_json_bytes(first) == runner.canonical_json_bytes(second)
    assert str(tmp_path).encode("utf-8") not in runner.canonical_json_bytes(first)
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert list(cache.iterdir()) == [marker]
    assert not output.exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-cache", "Hub cache root must already exist"),
        ("cache-file", "Hub cache root traverses a link or non-directory"),
        ("missing-output-parent", "model output parent must already exist"),
        ("existing-output-file", "refusing to overwrite"),
        ("existing-output-directory", "refusing to overwrite"),
        ("repo-local-cache", "Hub cache root must not overlap the repository"),
        ("cache-contains-repository", "Hub cache root must not overlap the repository"),
        ("repo-local-output", "model output root must be outside the repository"),
        ("output-inside-cache", "must not be nested"),
        ("lexical-output-inside-cache", "must not be nested"),
    ],
)
def test_model_staging_path_preflight_rejects_invalid_roots_without_writes(
    case: str,
    message: str,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    output_parent = outside / "output-parent"
    repository.mkdir()
    output_parent.mkdir(parents=True)
    cache = outside / "hub-cache"
    cache.mkdir()
    output = output_parent / "model"

    if case == "missing-cache":
        cache = outside / "missing-cache"
    elif case == "cache-file":
        cache = outside / "cache-file"
        cache.write_text("not a directory", encoding="utf-8")
    elif case == "missing-output-parent":
        output = outside / "missing-parent" / "model"
    elif case == "existing-output-file":
        output.write_text("occupied", encoding="utf-8")
    elif case == "existing-output-directory":
        output.mkdir()
    elif case == "repo-local-cache":
        cache = repository / "cache"
        cache.mkdir()
    elif case == "cache-contains-repository":
        cache = tmp_path
    elif case == "repo-local-output":
        output = repository / "model"
    elif case == "output-inside-cache":
        output = cache / "model"
    elif case == "lexical-output-inside-cache":
        output = cache / "unused" / ".." / "model"
    else:  # pragma: no cover - guards the table itself
        raise AssertionError(case)

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises((FileExistsError, runner.CalibrationRunError), match=message):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=cache,
            output_root=output,
        )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert after == before


def test_model_staging_path_preflight_rejects_filesystem_roots(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    repository.mkdir()
    cache.mkdir()
    root = Path(tmp_path.anchor)

    with pytest.raises(runner.CalibrationRunError, match="output parent.*filesystem root"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=cache,
            output_root=root / f"recurquant-output-{tmp_path.name}",
        )
    with pytest.raises(runner.CalibrationRunError, match="cache root.*filesystem root"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=root,
            output_root=tmp_path / "model",
        )


@pytest.mark.parametrize(
    "name",
    ("model:stream", "CON", "aux.json", "model.", "model ", "bad name", "\x01model"),
)
def test_model_staging_path_preflight_requires_windows_safe_output_basename(
    name: str,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    repository.mkdir()
    cache.mkdir()

    with pytest.raises(runner.CalibrationRunError, match="Windows-safe basename"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=cache,
            output_root=tmp_path / name,
        )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows case aliases only")
def test_model_staging_path_preflight_rejects_case_alias_repository_cache(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    case_alias = Path(str(repository).swapcase())

    with pytest.raises(runner.CalibrationRunError, match="must not overlap"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=case_alias,
            output_root=tmp_path / "model",
        )


def test_model_staging_path_preflight_rejects_links_and_dangling_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    real_cache = tmp_path / "real-cache"
    repository.mkdir()
    real_cache.mkdir()
    cache_link = tmp_path / "cache-link"
    dangling_output = tmp_path / "dangling-output"
    try:
        cache_link.symlink_to(real_cache, target_is_directory=True)
        dangling_output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    with pytest.raises(runner.CalibrationRunError, match="link or non-directory"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=cache_link,
            output_root=tmp_path / "model",
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=real_cache,
            output_root=dangling_output,
        )


def test_model_staging_path_snapshot_rejects_in_validation_component_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    displaced = tmp_path / "displaced-cache"
    repository.mkdir()
    cache.mkdir()
    original_resolve = Path.resolve
    replaced = False

    def replace_after_resolve(path: Path, strict: bool = False) -> Path:
        nonlocal replaced
        resolved = original_resolve(path, strict=strict)
        if path == cache and not replaced:
            replaced = True
            cache.rename(displaced)
            cache.mkdir()
        return resolved

    monkeypatch.setattr(Path, "resolve", replace_after_resolve)
    with pytest.raises(runner.CalibrationRunError, match="changed while it was validated"):
        runner.verify_model_staging_paths(
            repository_root=repository,
            hub_cache_root=cache,
            output_root=tmp_path / "model",
        )


def test_repo_local_cache_fails_before_git_h1_import_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    cache = repository / "cache"
    output_parent = tmp_path / "outside"
    cache.mkdir(parents=True)
    output_parent.mkdir()
    output = output_parent / "model"
    authentication_calls: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("invalid staging roots imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: authentication_calls.append("git"),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authentication_calls.append("h1"),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(runner.CalibrationRunError, match="must not overlap"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "missing-identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "missing-source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "missing-model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256="0" * 64,
            hub_cache_root=cache,
            output_root=output,
        )

    assert authentication_calls == []
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not output.exists()


def test_stage_model_path_contract_mismatch_fails_before_git_h1_hub_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    repository.mkdir()
    cache.mkdir()
    marker = cache / "preexisting.txt"
    marker.write_text("untouched", encoding="utf-8")
    output = tmp_path / "model"
    authentication_calls: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("path-contract mismatch imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: authentication_calls.append("git"),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authentication_calls.append("h1"),
    )

    with pytest.raises(runner.CalibrationRunError, match="differs from the CLI binding"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "missing-identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "missing-source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "missing-model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256="0" * 64,
            hub_cache_root=cache,
            output_root=output,
        )

    assert authentication_calls == []
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert list(cache.iterdir()) == [marker]
    assert not output.exists()
    assert not list(tmp_path.glob(".model.staging-*"))


@pytest.mark.parametrize(
    "invalid_sha256",
    ("A" * 64, "0" * 63, "g" * 64),
    ids=("uppercase", "short", "non-hex"),
)
def test_stage_model_requires_exact_lowercase_path_contract_sha256_before_auth(
    invalid_sha256: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    repository.mkdir()
    cache.mkdir()
    output = tmp_path / "model"
    authentication_calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: authentication_calls.append("git"),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authentication_calls.append("h1"),
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=invalid_sha256,
            hub_cache_root=cache,
            output_root=output,
        )

    assert authentication_calls == []
    assert not output.exists()


def test_stage_model_rejects_path_snapshot_mismatch_before_hub_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    other_cache = tmp_path / "other-cache"
    repository.mkdir()
    cache.mkdir()
    other_cache.mkdir()
    output = tmp_path / "model"
    original_validate = runner._validate_model_staging_roots
    expected_path_contract_sha256 = model_staging_path_contract_sha256(
        repository,
        cache,
        output,
    )
    validations = 0
    authorizations = 0
    original_import = builtins.__import__

    def validate(**kwargs: object) -> runner.ModelStagingPaths:
        nonlocal validations
        validations += 1
        result = original_validate(**kwargs)
        if validations == 2:
            return replace(result, hub_cache_root=other_cache.resolve())
        return result

    def authenticate(**_kwargs: object) -> runner.ModelStagingAuthorization:
        nonlocal authorizations
        authorizations += 1
        return model_staging_authorization()

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            pytest.fail("path mismatch imported a Hugging Face client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runner, "_validate_model_staging_roots", validate)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: runner.AuthenticatedGitExecutable(
            path=tmp_path / "git.exe",
            absolute_path_sha256="a" * 64,
            sha256="b" * 64,
            size_bytes=1,
        ),
    )
    monkeypatch.setattr(runner, "_authenticate_model_staging_authorization", authenticate)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(runner.CalibrationRunError, match="changed during authorization"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "model.json",
            expected_model_file_manifest_sha256="2" * 64,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=expected_path_contract_sha256,
            hub_cache_root=cache,
            output_root=output,
        )

    assert validations == 2
    assert authorizations == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".model.staging-*"))


def test_stage_model_downloads_only_exact_bound_files_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = staged_model_files()
    authorization = model_staging_authorization(files)
    authorizations: list[str] = []

    def authenticate(**kwargs: object) -> Any:
        del kwargs
        authorizations.append("authenticated")
        return authorization

    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        name = str(kwargs["filename"])
        cache = Path(kwargs["cache_dir"])
        target = cache / "models--example--model" / "snapshots" / ("2" * 40) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return str(target)

    monkeypatch.setattr(runner, "_authenticate_model_staging_authorization", authenticate)
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    output = tmp_path / "published-model"
    expected_path_contract_sha256 = runner._model_staging_path_contract_sha256(
        runner._validate_model_staging_roots(
            repository_root=SCRIPT.parents[1],
            hub_cache_root=cache,
            output_root=output,
        )
    )
    result = runner.stage_identity_bound_model(
        frozen_identity_path=tmp_path / "identity.json",
        expected_frozen_identity_sha256="d" * 64,
        identity_commit="3" * 40,
        repository_root=SCRIPT.parents[1],
        repository_source_manifest_path=tmp_path / "source.json",
        source_commit="1" * 40,
        model_file_manifest_path=tmp_path / "model-manifest.json",
        expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
        **capture_provenance_gate_kwargs(tmp_path),
        expected_model_staging_path_contract_sha256=expected_path_contract_sha256,
        hub_cache_root=cache,
        output_root=output,
        local_files_only=True,
        downloader=download,
    )

    assert authorizations == ["authenticated", "authenticated"]
    assert [call["filename"] for call in calls] == sorted(files)
    assert all(
        call
        == {
            "cache_dir": cache.resolve(),
            "endpoint": "https://huggingface.co",
            "filename": call["filename"],
            "local_files_only": True,
            "repo_id": "example/model",
            "repo_type": "model",
            "revision": "2" * 40,
            "token": False,
        }
        for call in calls
    )
    assert {path.name for path in output.iterdir()} == set(files)
    assert all((output / name).read_bytes() == content for name, content in files.items())
    assert not list(tmp_path.glob(".published-model.staging-*"))
    assert result["status"] == "staged_authenticated_model"
    assert result["source_commit"] == "1" * 40
    assert result["model_staging_path_contract_sha256"] == expected_path_contract_sha256


def test_stage_model_revalidates_root_identities_immediately_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = staged_model_files()
    authorization = model_staging_authorization(files)
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    displaced_cache = tmp_path / "displaced-cache"
    repository.mkdir()
    cache.mkdir()
    output = tmp_path / "published-model"
    expected_path_contract_sha256 = model_staging_path_contract_sha256(
        repository,
        cache,
        output,
    )
    authorization_calls = 0

    def authenticate(**_kwargs: object) -> runner.ModelStagingAuthorization:
        nonlocal authorization_calls
        authorization_calls += 1
        if authorization_calls == 2:
            cache.rename(displaced_cache)
            cache.mkdir()
        return authorization

    def download(**kwargs: object) -> str:
        name = str(kwargs["filename"])
        target = Path(kwargs["cache_dir"]) / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return str(target)

    monkeypatch.setattr(runner, "_authenticate_model_staging_authorization", authenticate)

    with pytest.raises(runner.CalibrationRunError, match="changed before publication"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "manifest.json",
            expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=expected_path_contract_sha256,
            hub_cache_root=cache,
            output_root=output,
            downloader=download,
        )

    assert authorization_calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".published-model.staging-*"))
    assert displaced_cache.is_dir()


def test_stage_model_failure_cleans_owned_staging_and_never_exposes_final_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = staged_model_files()
    authorization = model_staging_authorization(files)
    calls = 0

    def download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected download failure")
        name = str(kwargs["filename"])
        target = Path(kwargs["cache_dir"]) / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return str(target)

    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authorization,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "published-model"
    with pytest.raises(OSError, match="injected"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=SCRIPT.parents[1],
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "manifest.json",
            expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=(
                model_staging_path_contract_sha256(SCRIPT.parents[1], cache, output)
            ),
            hub_cache_root=cache,
            output_root=output,
            downloader=download,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".published-model.staging-*"))


def test_stage_model_refuses_to_clean_replaced_owned_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = model_staging_authorization()
    repository = tmp_path / "repository"
    cache = tmp_path / "cache"
    repository.mkdir()
    cache.mkdir()
    output = tmp_path / "published-model"
    displaced = tmp_path / "displaced-owned-staging"
    replacement: Path | None = None

    def download(**_kwargs: object) -> str:
        nonlocal replacement
        [owned] = list(tmp_path.glob(".published-model.staging-*"))
        owned.rename(displaced)
        owned.mkdir()
        (owned / "replacement-owner.txt").write_text("preserve", encoding="utf-8")
        replacement = owned
        raise OSError("injected after staging replacement")

    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authorization,
    )

    with pytest.raises(RuntimeError, match="refusing to clean a replaced"):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=repository,
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "manifest.json",
            expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=(
                model_staging_path_contract_sha256(repository, cache, output)
            ),
            hub_cache_root=cache,
            output_root=output,
            downloader=download,
        )

    assert replacement is not None
    assert (replacement / "replacement-owner.txt").read_text(encoding="utf-8") == "preserve"
    assert displaced.is_dir()
    assert not output.exists()


@pytest.mark.parametrize("outside_kind", ("outside", "wrong-content"))
def test_stage_model_rejects_untrusted_cache_payload_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside_kind: str,
) -> None:
    files = staged_model_files()
    authorization = model_staging_authorization(files)
    monkeypatch.setattr(
        runner,
        "_authenticate_model_staging_authorization",
        lambda **_kwargs: authorization,
    )
    cache = tmp_path / "cache"
    cache.mkdir()

    def download(**kwargs: object) -> str:
        name = str(kwargs["filename"])
        if outside_kind == "outside":
            target = tmp_path / "outside" / name
            content = files[name]
        else:
            target = Path(kwargs["cache_dir"]) / "snapshot" / name
            content = b"tampered" if name.endswith(".safetensors") else files[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    output = tmp_path / "published-model"
    message = "outside" if outside_kind == "outside" else "size differs|authentication failed"
    with pytest.raises(runner.CalibrationRunError, match=message):
        runner.stage_identity_bound_model(
            frozen_identity_path=tmp_path / "identity.json",
            expected_frozen_identity_sha256="d" * 64,
            identity_commit="3" * 40,
            repository_root=SCRIPT.parents[1],
            repository_source_manifest_path=tmp_path / "source.json",
            source_commit="1" * 40,
            model_file_manifest_path=tmp_path / "manifest.json",
            expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
            **capture_provenance_gate_kwargs(tmp_path),
            expected_model_staging_path_contract_sha256=(
                model_staging_path_contract_sha256(SCRIPT.parents[1], cache, output)
            ),
            hub_cache_root=cache,
            output_root=output,
            downloader=download,
        )
    assert not output.exists()


def test_cache_pointer_is_confined_and_never_published_as_a_link(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    blob = cache / "blobs" / "payload"
    pointer = cache / "snapshots" / ("2" * 40) / "model.safetensors"
    blob.parent.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    blob.write_bytes(b"payload")
    try:
        pointer.symlink_to(blob)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    assert runner._assert_regular_cache_payload(cache, pointer) == blob.resolve()
    pointer.unlink()
    outside = tmp_path / "outside-payload"
    outside.write_bytes(b"payload")
    pointer.symlink_to(outside)
    with pytest.raises(runner.CalibrationRunError, match="escapes"):
        runner._assert_regular_cache_payload(cache, pointer)


def test_atomic_model_directory_publish_never_replaces_racing_destination(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "model"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    (destination / "owner.txt").write_text("owner", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._atomic_rename_directory_no_overwrite(source, destination)

    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "owner"


def test_runtime_probe_and_manifest_accept_exact_runtime_root_sys_path_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_payload = runner.canonical_json_bytes(
        {
            "base_sys_path": ["Lib", "."],
            "machine": {
                "architecture": "64bit",
                "byteorder": "little",
                "machine": "test-machine",
                "pointer_bits": 64,
                "system": "TestOS",
            },
            "python": {
                "abi_flags": "",
                "cache_tag": "cpython-311",
                "implementation": "CPython",
                "version": "3.11.0",
            },
        }
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=probe_payload,
            stderr=b"",
        ),
    )

    probe = runner._probe_staged_interpreter(tmp_path / "python", tmp_path / "runtime")
    assert probe.base_sys_path == ("Lib", ".")

    manifest_payload = json.loads(runtime_manifest_bytes())
    manifest_payload["base_sys_path"] = ["."]
    manifest = runner.parse_calibration_runtime_manifest(
        runner.canonical_json_bytes(manifest_payload)
    )
    assert manifest.base_sys_path == (".",)


@pytest.mark.parametrize("value", ["", False, "/absolute", "../escape"])
def test_runtime_root_sys_path_normalizer_rejects_malformed_entries(value: object) -> None:
    with pytest.raises(runner.CalibrationRunError):
        runner._canonical_base_sys_path_entry(value, context="test base sys.path")


def test_runtime_root_sentinel_remains_invalid_for_ordinary_repository_paths() -> None:
    with pytest.raises(runner.CalibrationRunError):
        runner._canonical_relative_path(".", context="ordinary repository path")


def test_runtime_manifest_hashes_complete_record_inventory(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    package_root = tmp_path / "packages"
    import_root = package_root / "Lib" / "site-packages"
    interpreter = base_root / "python.exe"
    stdlib_file = base_root / "Lib" / "os.py"
    first = import_root / "package" / "__init__.py"
    second = import_root / "package-1.0.dist-info" / "RECORD"
    interpreter.parent.mkdir(parents=True)
    stdlib_file.parent.mkdir(parents=True)
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fake-python")
    stdlib_file.write_bytes(b"stdlib")
    first.write_bytes(b"VALUE = 1\n")
    second.write_text(
        "package/__init__.py,,\npackage-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    class Distribution:
        metadata = {"Name": "Package"}
        version = "1.0"
        files = ("package/__init__.py", "package-1.0.dist-info/RECORD")

        @staticmethod
        def locate_file(path: object) -> Path:
            return import_root / str(path)

        @staticmethod
        def read_text(name: str) -> str | None:
            return second.read_text(encoding="utf-8") if name == "RECORD" else None

    distribution = Distribution()
    payload = runner.capture_calibration_runtime_manifest(
        base_runtime_root=base_root,
        base_sys_path=("Lib",),
        distributions=[distribution],
        interpreter_path=interpreter,
        package_import_paths={"packages": "Lib/site-packages"},
        package_roots={"packages": package_root},
    )
    parsed = runner.parse_calibration_runtime_manifest(payload)
    authenticated = runner.authenticate_calibration_runtime(
        parsed,
        base_runtime_root=base_root,
        distributions=[distribution],
        interpreter_path=interpreter,
        package_roots={"packages": package_root},
    )

    assert authenticated.distributions == (("package", "1.0"),)
    assert authenticated.file_count == 4
    first.write_bytes(b"VALUE = 2\n")
    with pytest.raises(runner.CalibrationRunError, match="tree identity drifted"):
        runner.authenticate_calibration_runtime(
            parsed,
            base_runtime_root=base_root,
            distributions=[distribution],
            interpreter_path=interpreter,
            package_roots={"packages": package_root},
        )


@pytest.mark.parametrize("unsafe", ["../escape.py", "C:/escape.py", "/escape.py"])
def test_runtime_manifest_rejects_unsafe_serialized_paths(unsafe: str) -> None:
    payload = json.loads(runtime_manifest_bytes())
    payload["runtime_trees"][1]["files"][0]["path"] = unsafe

    with pytest.raises((ValueError, runner.CalibrationRunError), match="repository-relative"):
        runner.parse_calibration_runtime_manifest(runner.canonical_json_bytes(payload))


@pytest.mark.parametrize(
    "unsafe",
    [
        "Lib/site-packages/startup.pth",
        "Lib/site-packages/module.pyc",
        "Lib/site-packages/__pycache__/module.py",
    ],
)
def test_runtime_manifest_rejects_startup_hooks_and_bytecode(unsafe: str) -> None:
    payload = json.loads(runtime_manifest_bytes())
    payload["runtime_trees"][1]["files"][0]["path"] = unsafe

    with pytest.raises(ValueError, match="forbidden"):
        runner.parse_calibration_runtime_manifest(runner.canonical_json_bytes(payload))


def test_runtime_manifest_rejects_unrecorded_importable_package_file() -> None:
    payload = json.loads(runtime_manifest_bytes())
    payload["runtime_trees"][1]["files"].append(
        {
            "path": "Lib/site-packages/unrecorded.py",
            "sha256": "e" * 64,
            "size_bytes": 1,
        }
    )

    with pytest.raises(ValueError, match="distribution inventory"):
        runner.parse_calibration_runtime_manifest(runner.canonical_json_bytes(payload))


def test_runtime_tree_rejects_link_or_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    linked = root / "module.py"
    linked.write_bytes(b"value = 1\n")
    real_check = runner._is_link_or_reparse

    def selected_link(path: Path) -> bool:
        return path.name == "module.py" or real_check(path)

    monkeypatch.setattr(runner, "_is_link_or_reparse", selected_link)
    with pytest.raises(runner.CalibrationRunError, match="link or reparse"):
        runner._runtime_tree_files(root, kind="packages")


def test_nonempty_pycache_prefix_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "pycache"
    root.mkdir()
    root.joinpath("injected.pyc").write_bytes(b"bytecode")

    with pytest.raises(runner.CalibrationRunError, match="not empty"):
        runner._verify_empty_pycache_prefix(root)


def test_official_runtime_preflight_requires_datasets_before_adapter_access() -> None:
    runtime = runner.AuthenticatedRuntime(
        manifest_file_sha256="a" * 64,
        python_implementation="CPython",
        python_version="3.11.15",
        python_cache_tag="cpython-311",
        interpreter_sha256="b" * 64,
        git_executable_absolute_path_sha256="c" * 64,
        git_executable_sha256="d" * 64,
        git_executable_size_bytes=1,
        machine_name="AMD64",
        base_runtime_file_count=1,
        package_root_count=1,
        distributions=(("transformers", "5.14.1"),),
        distribution_count=1,
        file_count=1,
    )

    with pytest.raises(runner.CalibrationRunError, match="datasets==4.8.5"):
        runner._preflight_official_runtime_distributions(runtime)

    runner._preflight_official_runtime_distributions(
        replace(
            runtime,
            distributions=(("datasets", "4.8.5"),),
            distribution_count=1,
        )
    )


def test_source_bound_requirements_match_exact_54_distribution_runtime() -> None:
    requirements = runner._parse_runtime_requirements(
        SCRIPT.parents[1] / runner.CALIBRATION_REQUIREMENTS_PATH
    )
    assert len(requirements) == 54
    assert tuple(item.name for item in requirements) == tuple(
        sorted(item.name for item in requirements)
    )
    pins = {item.name: item.version for item in requirements}
    assert {name: pins[name] for name in ("datasets", "fsspec", "xxhash")} == {
        "datasets": "4.8.5",
        "fsspec": "2026.2.0",
        "xxhash": "3.8.1",
    }

    manifest = runner.parse_calibration_runtime_manifest(runtime_manifest_bytes())
    exact_distributions = tuple(
        runner.RuntimeDistributionRecord(
            name=item.name,
            version=item.version,
            package_root="packages",
            files=(f"Lib/site-packages/{item.name}.dist-info/RECORD",),
        )
        for item in requirements
    )
    exact_manifest = replace(manifest, distributions=exact_distributions)
    runner._preflight_runtime_requirements(exact_manifest, requirements)

    drifted = replace(
        exact_manifest,
        distributions=(
            replace(exact_distributions[0], version="0.0.0"),
            *exact_distributions[1:],
        ),
    )
    with pytest.raises(runner.CalibrationRunError, match="source-bound"):
        runner._preflight_runtime_requirements(drifted, requirements)


def test_base_runtime_staging_omits_bytecode_hooks_and_global_packages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    for relative, content in {
        "python.exe": b"python",
        "python311.dll": b"dll",
        "DLLs/_ssl.pyd": b"ssl",
        "Lib/os.py": b"os",
        "Lib/startup.pth": b"hook",
        "Lib/sitecustomize.py": b"hook",
        "Lib/__pycache__/os.pyc": b"bytecode",
        "Lib/site-packages/untrusted.py": b"untrusted",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    destination.mkdir()

    runner._copy_base_runtime(source.resolve(), destination)

    copied = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert copied == {
        "DLLs/_ssl.pyd",
        "Lib/os.py",
        "python.exe",
        "python311.dll",
    }


def test_record_only_staging_preserves_scripts_outside_site_packages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    site = source / "Lib" / "site-packages"
    package = site / "demo" / "__init__.py"
    record_path = site / "demo-1.0.dist-info" / "RECORD"
    script = source / "Scripts" / "demo.exe"
    for path, content in ((package, b"demo"), (script, b"script")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_text = "demo/__init__.py,,\ndemo-1.0.dist-info/RECORD,,\n../../Scripts/demo.exe,,\n"
    record_path.write_text(record_text, encoding="utf-8")
    destination.mkdir()

    class Distribution:
        metadata = {"Name": "demo"}
        version = "1.0"
        files = (
            "demo/__init__.py",
            "demo-1.0.dist-info/RECORD",
            "../../Scripts/demo.exe",
        )

        @staticmethod
        def locate_file(path: object) -> Path:
            return site / str(path)

        @staticmethod
        def read_text(name: str) -> str | None:
            return record_text if name == "RECORD" else None

    runner._copy_record_only_packages(source.resolve(), destination, [Distribution()])

    assert (destination / "Scripts" / "demo.exe").read_bytes() == b"script"
    assert not list(destination.rglob("*.pth"))


def test_prepare_runtime_is_strictly_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "runtime"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.prepare_calibration_runtime(
            source_python=tmp_path / "missing-python.exe",
            requirements_file=tmp_path / "missing-requirements.txt",
            output_root=output,
        )


def test_adapter_contract_uses_the_shared_api_identity() -> None:
    events: list[str] = []
    adapter = FakeAdapter({}, events)

    runner.validate_adapter_contract(adapter, calibration_api=api)
    with pytest.raises(TypeError, match="does not implement"):
        runner.validate_adapter_contract(object(), calibration_api=api)


def test_atomic_publish_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    runner._atomic_publish_new(path, b"first")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._atomic_publish_new(path, b"second")

    assert path.read_bytes() == b"first"


def test_output_directory_fault_never_exposes_a_partial_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "final"
    real_publish = runner._atomic_publish_new
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_publish(path, payload)

    monkeypatch.setattr(runner, "_atomic_publish_new", fail_second)
    with pytest.raises(OSError, match="injected"):
        runner._publish_output_directory(output, {"a.json": b"a", "b.json": b"b"})

    assert not output.exists()
    assert not list(tmp_path.glob(".final.staging-*"))


def test_output_directory_snapshot_revalidates_parent_identity_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authorization-parent"
    parent.mkdir()
    output = parent / "stage-a-authorization"
    snapshot = runner._new_capture_artifact_path(
        output,
        context="Stage-A authorization output directory",
    )
    require_directory = runner._require_existing_regular_directory
    revalidation_calls = 0

    def inject_identity_drift(
        path: Path,
        *,
        context: str,
    ) -> tuple[Path, tuple[runner.DirectoryComponentIdentity, ...]]:
        nonlocal revalidation_calls
        resolved, identities = require_directory(path, context=context)
        if context.startswith("published output directory"):
            revalidation_calls += 1
            if revalidation_calls == 2:
                final = identities[-1]
                identities = (
                    *identities[:-1],
                    replace(final, inode=final.inode + 1),
                )
        return resolved, identities

    monkeypatch.setattr(
        runner,
        "_require_existing_regular_directory",
        inject_identity_drift,
    )
    with pytest.raises(
        runner.CalibrationRunError,
        match="parent changed before publication",
    ):
        runner._publish_output_directory(
            output,
            {
                runner.AUTHORIZATION_FILENAME: b"authorization",
                runner.BINDING_FILENAME: b"binding",
            },
            complete_filename=runner.AUTHORIZATION_COMPLETE_FILENAME,
            output_path_snapshot=snapshot,
        )

    assert revalidation_calls == 2
    assert not output.exists()
    assert not list(parent.glob(".stage-a-authorization.staging-*"))


def test_capture_manifest_cli_modes_are_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    source_output = tmp_path / "source.json"
    runtime_output = tmp_path / "runtime.json"
    model_output = tmp_path / "model.json"
    source_manifest = {
        "canonical_manifest_sha256": "6" * 64,
        "source_commit": "4" * 40,
    }
    source_payload = b'{"source":"verified"}\n'
    verification_calls: list[dict[str, object]] = []

    class FakeSourceCapture:
        @staticmethod
        def capture_experiment013_source_manifest(
            root: Path,
            *,
            git_executable: Path,
        ) -> dict[str, object]:
            assert root == repository_root
            assert git_executable == authenticated_git_path()
            return source_manifest

        @staticmethod
        def validate_experiment013_source_manifest(
            manifest: Mapping[str, object],
        ) -> dict[str, object]:
            assert manifest is source_manifest
            return source_manifest

        @staticmethod
        def canonical_experiment013_source_manifest_bytes(
            manifest: Mapping[str, object],
        ) -> bytes:
            assert manifest is source_manifest
            return source_payload

        @staticmethod
        def verify_experiment013_source_manifest(
            manifest: Mapping[str, object],
            root: Path,
            *,
            git_executable: Path,
        ) -> dict[str, object]:
            assert manifest is source_manifest
            assert root == repository_root
            assert git_executable == authenticated_git_path()
            verification_calls.append(dict(manifest))
            return source_manifest

    monkeypatch.setattr(runner, "_load_source_capture_module", lambda _root: FakeSourceCapture)
    runtime_calls: list[dict[str, object]] = []

    def runtime_capture(**kwargs: object) -> bytes:
        runtime_calls.append(dict(kwargs))
        return b"runtime\n"

    monkeypatch.setattr(runner, "capture_calibration_runtime_manifest", runtime_capture)

    assert (
        runner.main(
            [
                "capture-source-manifest",
                "--git-executable",
                str(authenticated_git_path()),
                "--repository-root",
                str(repository_root),
                "--output",
                str(source_output),
            ]
        )
        == 0
    )
    assert source_output.read_bytes() == source_payload
    assert len(verification_calls) == 2
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.main(
            [
                "capture-source-manifest",
                "--git-executable",
                str(authenticated_git_path()),
                "--repository-root",
                str(repository_root),
                "--output",
                str(source_output),
            ]
        )

    runtime_args = [
        "capture-runtime-manifest",
        "--git-executable",
        str(authenticated_git_path()),
        "--output",
        str(runtime_output),
        "--base-runtime-root",
        str(tmp_path / "base"),
        "--staged-interpreter",
        str(tmp_path / "base" / "python.exe"),
        "--package-root",
        f"packages={tmp_path / 'packages'}",
        "--package-import-path",
        "packages=Lib/site-packages",
    ]
    assert runner.main(runtime_args) == 0
    assert runtime_output.read_bytes() == b"runtime\n"
    assert runtime_calls == [
        {
            "base_runtime_root": tmp_path / "base",
            "git_executable_path": authenticated_git_path(),
            "interpreter_path": tmp_path / "base" / "python.exe",
            "package_import_paths": {"packages": "Lib/site-packages"},
            "package_roots": {"packages": tmp_path / "packages"},
        }
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.main(runtime_args)

    def model_capture(
        model_id: str,
        revision: str,
        *,
        transformers_version: str,
        token: bool,
    ) -> bytes:
        assert (model_id, revision, transformers_version) == (
            "example/model",
            "4" * 40,
            "5.14.1",
        )
        assert token is False
        return b"model\n"

    monkeypatch.setattr(runner, "capture_model_file_manifest_from_hub", model_capture)
    assert (
        runner.main(
            [
                "capture-model-manifest",
                "--output",
                str(model_output),
                "--model-id",
                "example/model",
                "--revision",
                "4" * 40,
                "--transformers-version",
                "5.14.1",
            ]
        )
        == 0
    )
    assert model_output.read_bytes() == b"model\n"


def test_stage_model_cli_forwards_only_explicit_authorization_and_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def stage(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "staged_authenticated_model"}

    monkeypatch.setattr(runner, "stage_identity_bound_model", stage)
    arguments = [
        "stage-model",
        "--git-executable",
        str(authenticated_git_path()),
        "--frozen-identity",
        str(tmp_path / "identity.json"),
        "--expected-frozen-identity-sha256",
        "1" * 64,
        "--identity-commit",
        "2" * 40,
        "--repository-root",
        str(tmp_path / "repository"),
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--source-commit",
        "3" * 40,
        "--model-file-manifest",
        str(tmp_path / "model.json"),
        "--expected-model-file-manifest-sha256",
        "4" * 64,
        *capture_provenance_gate_cli(tmp_path),
        "--hub-cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "output"),
        "--expected-model-staging-path-contract-sha256",
        "5" * 64,
        "--local-files-only",
    ]

    assert runner.main(arguments) == 0
    assert calls == [
        {
            "expected_frozen_identity_sha256": "1" * 64,
            "expected_model_file_manifest_sha256": "4" * 64,
            "expected_model_staging_path_contract_sha256": "5" * 64,
            "capture_provenance_receipt_path": tmp_path / "capture-provenance.json",
            "expected_capture_provenance_receipt_sha256": "5" * 64,
            "runtime_manifest_path": tmp_path / "runtime.json",
            "expected_runtime_manifest_sha256": "8" * 64,
            "frozen_identity_path": tmp_path / "identity.json",
            "git_executable_path": authenticated_git_path(),
            "hub_cache_root": tmp_path / "cache",
            "identity_commit": "2" * 40,
            "local_files_only": True,
            "model_file_manifest_path": tmp_path / "model.json",
            "output_root": tmp_path / "output",
            "repository_root": tmp_path / "repository",
            "repository_source_manifest_path": tmp_path / "source.json",
            "source_commit": "3" * 40,
        }
    ]

    missing_contract_arguments = arguments[:-3] + arguments[-1:]
    with pytest.raises(SystemExit):
        runner.main(missing_contract_arguments)
    assert len(calls) == 1


def test_verify_model_staging_paths_cli_is_canonical_and_exactly_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    receipt = {
        "artifact_kind": runner.MODEL_STAGING_PATHS_KIND,
        "schema_version": runner.MODEL_STAGING_PATHS_SCHEMA,
        "status": "verified_model_staging_paths",
    }

    def verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return receipt

    monkeypatch.setattr(runner, "verify_model_staging_paths", verify)
    arguments = [
        "verify-model-staging-paths",
        "--repository-root",
        str(tmp_path / "repository"),
        "--hub-cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "output"),
    ]

    assert runner.main(arguments) == 0
    assert capsys.readouterr().out == runner.canonical_json_bytes(receipt).decode("utf-8")
    assert calls == [
        {
            "hub_cache_root": tmp_path / "cache",
            "output_root": tmp_path / "output",
            "repository_root": tmp_path / "repository",
        }
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--git-executable", "git"),
        ("--frozen-identity", "identity.json"),
        ("--expected-frozen-identity-sha256", "1" * 64),
        ("--identity-commit", "2" * 40),
        ("--repository-source-manifest", "source.json"),
        ("--source-commit", "3" * 40),
        ("--model-file-manifest", "model.json"),
        ("--expected-model-file-manifest-sha256", "4" * 64),
        ("--expected-model-staging-path-contract-sha256", "5" * 64),
        ("--local-files-only",),
        ("--model-root", "model"),
        ("--cache-root", "cache"),
        ("--output", "receipt.json"),
        ("--output-dir", "output"),
    ],
    ids=(
        "git",
        "identity",
        "identity-hash",
        "h1",
        "source-manifest",
        "h0",
        "model-manifest",
        "model-manifest-hash",
        "staging-path-contract-hash",
        "local-hub-mode",
        "model-root",
        "generic-cache",
        "persisted-output",
        "calibration-output",
    ),
)
def test_verify_model_staging_paths_cli_rejects_every_non_path_surface(
    forbidden: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "verify_model_staging_paths",
        lambda **kwargs: calls.append(dict(kwargs)),
    )
    arguments = [
        "verify-model-staging-paths",
        "--repository-root",
        str(tmp_path / "repository"),
        "--hub-cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "output"),
    ]

    with pytest.raises(SystemExit):
        runner.main([*arguments, *forbidden])
    assert calls == []


def test_verify_frozen_identity_contract_cli_is_canonical_and_exactly_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    receipt = {
        "artifact_kind": runner.FROZEN_IDENTITY_CONTRACT_KIND,
        "schema_version": runner.FROZEN_IDENTITY_CONTRACT_SCHEMA,
        "status": "verified_frozen_identity_contract",
    }

    def verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return receipt

    monkeypatch.setattr(runner, "verify_frozen_identity_contract", verify)
    arguments = [
        "verify-frozen-identity-contract",
        "--git-executable",
        str(authenticated_git_path()),
        "--frozen-identity",
        str(tmp_path / "identity.json"),
        "--expected-frozen-identity-sha256",
        "1" * 64,
        "--repository-root",
        str(tmp_path / "repository"),
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--source-commit",
        "3" * 40,
        *capture_provenance_gate_cli(tmp_path),
    ]

    assert runner.main(arguments) == 0
    assert capsys.readouterr().out == runner.canonical_json_bytes(receipt).decode("utf-8")
    assert calls == [
        {
            "expected_frozen_identity_sha256": "1" * 64,
            "frozen_identity_path": tmp_path / "identity.json",
            "git_executable_path": authenticated_git_path(),
            "repository_root": tmp_path / "repository",
            "repository_source_manifest_path": tmp_path / "source.json",
            "source_commit": "3" * 40,
            "capture_provenance_receipt_path": tmp_path / "capture-provenance.json",
            "expected_capture_provenance_receipt_sha256": "5" * 64,
            "runtime_manifest_path": tmp_path / "runtime.json",
            "expected_runtime_manifest_sha256": "8" * 64,
        }
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--identity-commit", "2" * 40),
        ("--model-file-manifest", "model.json"),
        ("--expected-model-file-manifest-sha256", "4" * 64),
        ("--expected-model-staging-path-contract-sha256", "5" * 64),
        ("--hub-cache-root", "cache"),
        ("--output-root", "output"),
        ("--output", "output.json"),
        ("--cache-root", "cache"),
        ("--local-files-only",),
    ],
    ids=(
        "h1",
        "model-manifest",
        "model-manifest-hash",
        "staging-path-contract-hash",
        "hub-cache",
        "output-root",
        "output",
        "cache-root",
        "local-hub-mode",
    ),
)
def test_verify_frozen_identity_contract_cli_rejects_forbidden_surfaces(
    forbidden: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "verify_frozen_identity_contract",
        lambda **kwargs: calls.append(dict(kwargs)),
    )
    arguments = [
        "verify-frozen-identity-contract",
        "--git-executable",
        str(authenticated_git_path()),
        "--frozen-identity",
        str(tmp_path / "identity.json"),
        "--expected-frozen-identity-sha256",
        "1" * 64,
        "--repository-root",
        str(tmp_path / "repository"),
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--source-commit",
        "3" * 40,
        *capture_provenance_gate_cli(tmp_path),
    ]

    with pytest.raises(SystemExit):
        runner.main([*arguments, *forbidden])
    assert calls == []


def test_verify_model_staging_authorization_cli_has_no_cache_or_output_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    receipt = {
        "artifact_kind": runner.MODEL_STAGING_AUTHORIZATION_KIND,
        "status": "verified_identity_bound_model_staging_authorization",
    }

    def verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return receipt

    monkeypatch.setattr(runner, "verify_identity_bound_model_staging_authorization", verify)
    arguments = [
        "verify-model-staging-authorization",
        "--git-executable",
        str(authenticated_git_path()),
        "--frozen-identity",
        str(tmp_path / "identity.json"),
        "--expected-frozen-identity-sha256",
        "1" * 64,
        "--identity-commit",
        "2" * 40,
        "--repository-root",
        str(tmp_path / "repository"),
        "--repository-source-manifest",
        str(tmp_path / "source.json"),
        "--source-commit",
        "3" * 40,
        "--model-file-manifest",
        str(tmp_path / "model.json"),
        "--expected-model-file-manifest-sha256",
        "4" * 64,
        *capture_provenance_gate_cli(tmp_path),
    ]

    assert runner.main(arguments) == 0
    assert capsys.readouterr().out == runner.canonical_json_bytes(receipt).decode("utf-8")
    assert calls == [
        {
            "expected_frozen_identity_sha256": "1" * 64,
            "expected_model_file_manifest_sha256": "4" * 64,
            "capture_provenance_receipt_path": tmp_path / "capture-provenance.json",
            "expected_capture_provenance_receipt_sha256": "5" * 64,
            "runtime_manifest_path": tmp_path / "runtime.json",
            "expected_runtime_manifest_sha256": "8" * 64,
            "frozen_identity_path": tmp_path / "identity.json",
            "git_executable_path": authenticated_git_path(),
            "identity_commit": "2" * 40,
            "model_file_manifest_path": tmp_path / "model.json",
            "repository_root": tmp_path / "repository",
            "repository_source_manifest_path": tmp_path / "source.json",
            "source_commit": "3" * 40,
        }
    ]

    for forbidden in (
        ("--hub-cache-root", str(tmp_path / "forbidden-cache")),
        ("--output-root", str(tmp_path / "forbidden-output")),
        ("--expected-model-staging-path-contract-sha256", "5" * 64),
        ("--local-files-only",),
    ):
        with pytest.raises(SystemExit):
            runner.main([*arguments, *forbidden])
    assert len(calls) == 1


def test_source_manifest_output_location_allows_only_external_or_ignored_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")

    outside = tmp_path / "outside.json"
    ignored = repository / "artifacts" / "source.json"
    git_executable = runner._authenticate_git_executable(authenticated_git_path())
    assert (
        runner._assert_source_manifest_output_location(
            repository, outside, git_executable=git_executable
        )
        == outside.resolve()
    )
    assert (
        runner._assert_source_manifest_output_location(
            repository, ignored, git_executable=git_executable
        )
        == ignored.resolve()
    )

    with pytest.raises(runner.CalibrationRunError, match="must be ignored"):
        runner._assert_source_manifest_output_location(
            repository,
            repository / "source.json",
            git_executable=git_executable,
        )
    with pytest.raises(runner.CalibrationRunError, match="repository metadata"):
        runner._assert_source_manifest_output_location(
            repository,
            repository / ".git" / "new.json",
            git_executable=git_executable,
        )


def test_official_cli_uses_only_unambiguous_ruler_receipt_directory_option(
    tmp_path: Path,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    arguments = official_cli_arguments(tmp_path, ruler_receipts=receipt_dir)

    parsed = runner._parser().parse_args(arguments)

    assert parsed.ruler_receipt_dir == receipt_dir
    assert parsed.capture_provenance_receipt == tmp_path / "capture-provenance.json"
    assert parsed.expected_capture_provenance_receipt_sha256 == "5" * 64
    assert "--ruler-receipt-dir" in runner._parser().format_help()
    assert "--ruler-root" not in runner._parser().format_help()
    for required in (
        "--capture-provenance-receipt",
        "--expected-capture-provenance-receipt-sha256",
    ):
        missing = list(arguments)
        index = missing.index(required)
        del missing[index : index + 2]
        with pytest.raises(SystemExit):
            runner._parser().parse_args(missing)
    legacy = list(arguments)
    legacy[legacy.index("--ruler-receipt-dir")] = "--ruler-root"
    with pytest.raises(SystemExit):
        runner._parser().parse_args(legacy)


def test_sealed_capture_parser_is_exact_nonmixable_and_hard_codes_phase(
    tmp_path: Path,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    arguments = sealed_capture_cli_arguments(tmp_path, ruler_receipts=receipt_dir)
    parsed = runner._parse_calibration_identity_capture_arguments(arguments)

    assert parsed.repository_root == tmp_path / "repository"
    assert not hasattr(parsed, "phase")
    assert not hasattr(parsed, "model_root")
    assert not hasattr(parsed, "adapter")
    for forbidden in (
        ["--phase", "stage_a"],
        ["--model-root", str(tmp_path / "model")],
        ["--adapter", runner.CANONICAL_ADAPTER_SPEC],
        ["--fisher-h1-smoke", "1"],
        ["--package-root", f"packages={tmp_path}"],
    ):
        with pytest.raises(runner.CalibrationRunError, match="exact|mixed"):
            runner._parse_calibration_identity_capture_arguments([*arguments, *forbidden])
    duplicated = [*arguments, "--output", str(tmp_path / "other.json")]
    with pytest.raises(runner.CalibrationRunError, match="exact|mixed"):
        runner._parse_calibration_identity_capture_arguments(duplicated)


def test_sealed_stage_a_capture_parser_is_exact_nonmixable_and_hard_codes_phase(
    tmp_path: Path,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    arguments = sealed_capture_cli_arguments(tmp_path, ruler_receipts=receipt_dir)
    arguments[0] = "capture-stage-a-identity"
    binding = tmp_path / "stage-a-binding.json"
    arguments.extend(
        [
            "--stage-a-calibration-binding",
            str(binding),
            "--expected-stage-a-calibration-binding-sha256",
            "6" * 64,
        ]
    )
    parsed = runner._parse_stage_a_identity_capture_arguments(arguments)

    assert parsed.capture_phase == "stage_a"
    assert parsed.stage_a_calibration_binding == binding
    assert not hasattr(parsed, "phase")
    for forbidden in (
        ["--phase", "stage_a"],
        ["--model-root", str(tmp_path / "model")],
        ["--fisher-h1-smoke", "1"],
        ["--capture-provenance-receipt", str(tmp_path / "old.json")],
    ):
        with pytest.raises(runner.CalibrationRunError, match="exact|mixed"):
            runner._parse_stage_a_identity_capture_arguments([*arguments, *forbidden])
    with pytest.raises(runner.CalibrationRunError, match="calibration.*command"):
        runner._parse_calibration_identity_capture_arguments(arguments)


def test_unsealed_main_rejects_calibration_identity_capture(
    tmp_path: Path,
) -> None:
    arguments = sealed_capture_cli_arguments(
        tmp_path,
        ruler_receipts=tmp_path / "missing-ruler-receipts",
    )
    with pytest.raises(
        runner.CalibrationRunError,
        match="started with launch_static_q468_calibration.py",
    ):
        runner.main(arguments)


def test_sealed_capture_checks_ruler_inventory_before_runtime_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_checkout = tmp_path / "RULER"
    source_checkout.mkdir()
    (source_checkout / "README.md").write_text("source checkout", encoding="utf-8")
    arguments = sealed_capture_cli_arguments(tmp_path, ruler_receipts=source_checkout)
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "_authenticate_sealed_runtime_context",
        lambda *_args, **_kwargs: calls.append("runtime") or pytest.fail("runtime accessed"),
    )

    with pytest.raises(runner.CalibrationRunError, match="inventory drifted"):
        runner.sealed_main(
            arguments,
            base_runtime_root=tmp_path / "base",
            package_roots={"packages": tmp_path / "packages"},
            package_import_paths={"packages": "Lib/site-packages"},
            interpreter_path=tmp_path / "python.exe",
            git_executable_path=tmp_path / "git.exe",
            pycache_prefix=tmp_path / "pycache",
        )
    assert calls == []


def test_ruler_receipt_directory_precondition_reads_no_file_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("receipt precondition must not read file bytes"),
    )

    assert runner._verify_ruler_receipt_directory_precondition(receipt_dir) == (
        receipt_dir.resolve(strict=True)
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_ruler_receipt_directory_precondition_rejects_inventory_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    if mutation == "missing":
        (receipt_dir / runner.RULER_RECEIPT_DIRECTORY_FILENAMES[-1]).unlink()
    else:
        (receipt_dir / "unexpected.json").write_bytes(b"unexpected\n")

    with pytest.raises(runner.CalibrationRunError, match="inventory drifted"):
        runner._verify_ruler_receipt_directory_precondition(receipt_dir)


def test_ruler_receipt_directory_precondition_rejects_case_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    names = [*runner.RULER_RECEIPT_DIRECTORY_FILENAMES, "GENERATION-MANIFEST.JSON"]

    class FakeScandir:
        def __enter__(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name) for name in names]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(runner.os, "scandir", lambda _root: FakeScandir())

    with pytest.raises(runner.CalibrationRunError, match="case-colliding"):
        runner._verify_ruler_receipt_directory_precondition(receipt_dir)


def test_ruler_receipt_directory_precondition_rejects_nonregular_entry(
    tmp_path: Path,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    entry = receipt_dir / runner.RULER_RECEIPT_DIRECTORY_FILENAMES[-1]
    entry.unlink()
    entry.mkdir()

    with pytest.raises(runner.CalibrationRunError, match="regular non-link file"):
        runner._verify_ruler_receipt_directory_precondition(receipt_dir)


def test_ruler_receipt_directory_precondition_rejects_symlink_entry(
    tmp_path: Path,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    entry = receipt_dir / runner.RULER_RECEIPT_DIRECTORY_FILENAMES[-1]
    target = tmp_path / "outside.json"
    target.write_bytes(b"outside\n")
    entry.unlink()
    try:
        entry.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    with pytest.raises(runner.CalibrationRunError, match="regular non-link file"):
        runner._verify_ruler_receipt_directory_precondition(receipt_dir)


def test_ruler_receipt_directory_precondition_rejects_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_dir = ruler_receipt_directory(tmp_path / "ruler-receipts")
    target = runner.RULER_RECEIPT_DIRECTORY_FILENAMES[-1]

    class FakeEntry:
        def __init__(self, name: str) -> None:
            self.name = name

        @staticmethod
        def is_symlink() -> bool:
            return False

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(
                st_dev=1,
                st_file_attributes=(runner._WINDOWS_REPARSE_POINT if self.name == target else 0),
                st_ino=runner.RULER_RECEIPT_DIRECTORY_FILENAMES.index(self.name) + 1,
                st_mode=runner.stat.S_IFREG | 0o644,
                st_size=8,
            )

    class FakeScandir:
        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry(name) for name in runner.RULER_RECEIPT_DIRECTORY_FILENAMES]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(runner.os, "scandir", lambda _root: FakeScandir())

    with pytest.raises(runner.CalibrationRunError, match="regular non-link file"):
        runner._verify_ruler_receipt_directory_precondition(receipt_dir)


def test_ruler_receipt_directory_precondition_rejects_linked_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    receipt_dir = ruler_receipt_directory(real_parent / "ruler-receipts")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    with pytest.raises(runner.CalibrationRunError, match="link or non-directory"):
        runner._verify_ruler_receipt_directory_precondition(linked_parent / receipt_dir.name)


@pytest.mark.parametrize("kind", ["relative", "missing"])
def test_ruler_receipt_directory_precondition_rejects_invalid_path_before_access(
    tmp_path: Path,
    kind: str,
) -> None:
    path = Path("relative-ruler-receipts") if kind == "relative" else tmp_path / "missing"
    message = "must be absolute" if kind == "relative" else "must already exist"

    with pytest.raises(runner.CalibrationRunError, match=message):
        runner._verify_ruler_receipt_directory_precondition(path)


def test_sealed_main_rejects_ruler_source_checkout_before_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_checkout = tmp_path / "ruler-source-checkout"
    (source_checkout / ".git").mkdir(parents=True)
    (source_checkout / "scripts").mkdir()
    (source_checkout / "README.md").write_bytes(b"RULER source checkout\n")
    arguments = official_cli_arguments(tmp_path, ruler_receipts=source_checkout)
    monkeypatch.setattr(
        runner,
        "_authenticate_sealed_runtime_context",
        lambda *_args, **_kwargs: pytest.fail("runtime authentication must not begin"),
    )
    monkeypatch.setattr(
        runner,
        "_official_main",
        lambda *_args, **_kwargs: pytest.fail("materialization boundary must not be entered"),
    )

    with pytest.raises(runner.CalibrationRunError, match="receipt directory inventory drifted"):
        runner.sealed_main(
            arguments,
            base_runtime_root=tmp_path / "unopened-base-runtime",
            package_roots={"packages": tmp_path / "unopened-packages"},
            package_import_paths={"packages": "Lib/site-packages"},
            interpreter_path=tmp_path / "unopened-base-runtime" / "python.exe",
            git_executable_path=tmp_path / "unopened-git.exe",
            pycache_prefix=tmp_path / "unopened-pycache",
        )

    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "identity.json").exists()
    assert not (tmp_path / "model-root").exists()
    assert not (tmp_path / "output").exists()


def test_public_main_binds_manifest_bytes_before_adapter_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bytes = b"not-the-frozen-source\n"
    runtime_bytes = b"runtime-metadata\n"
    model_bytes = b"model-metadata\n"
    parquet_bytes = b"parquet-metadata\n"
    identity_bytes = bootstrap_identity_bytes(
        source_manifest_sha256="0" * 64,
        runtime_manifest_sha256=digest(runtime_bytes),
        model_manifest_sha256=digest(model_bytes),
        parquet_manifest_sha256=digest(parquet_bytes),
    )
    paths = {
        "identity": tmp_path / "identity.json",
        "source": tmp_path / "source.json",
        "runtime": tmp_path / "runtime.json",
        "model": tmp_path / "model.json",
        "parquet": tmp_path / "parquet.json",
    }
    paths["identity"].write_bytes(identity_bytes)
    paths["source"].write_bytes(source_bytes)
    paths["runtime"].write_bytes(runtime_bytes)
    paths["model"].write_bytes(model_bytes)
    paths["parquet"].write_bytes(parquet_bytes)
    receipt_dir = ruler_receipt_directory(tmp_path / "unopened-ruler-receipts")
    imported: list[bool] = []
    monkeypatch.setattr(runner, "_load_adapter", lambda *args, **kwargs: imported.append(True))

    with pytest.raises(runner.CalibrationRunError, match="bootstrap binding"):
        runner._official_main(
            [
                "--frozen-identity",
                str(paths["identity"]),
                "--capture-provenance-receipt",
                str(tmp_path / "unopened-capture-provenance.json"),
                "--expected-capture-provenance-receipt-sha256",
                "5" * 64,
                "--repository-source-manifest",
                str(paths["source"]),
                "--runtime-manifest",
                str(paths["runtime"]),
                "--expected-runtime-manifest-sha256",
                digest(runtime_bytes),
                "--model-file-manifest",
                str(paths["model"]),
                "--expected-model-file-manifest-sha256",
                digest(model_bytes),
                "--parquet-materialization-manifest",
                str(paths["parquet"]),
                "--expected-parquet-materialization-manifest-sha256",
                digest(parquet_bytes),
                "--model-root",
                str(tmp_path / "unopened-model"),
                "--cache-root",
                str(tmp_path / "unopened-cache"),
                "--ruler-receipt-dir",
                str(receipt_dir),
                "--repository-root",
                str(tmp_path / "unopened-repository"),
                "--source-commit",
                "1" * 40,
                "--output-dir",
                str(tmp_path / "output"),
            ],
            runtime_context=runner.SealedRuntimeContext(
                manifest_file_sha256=digest(runtime_bytes),
                base_runtime_root=tmp_path / "base",
                package_roots={"packages": tmp_path / "packages"},
                package_import_paths={"packages": "Lib/site-packages"},
                git_executable_path=authenticated_git_path(),
                pycache_prefix=tmp_path / "pycache",
            ),
            interpreter_path=tmp_path / "base" / "python.exe",
        )

    assert imported == []
    assert not (tmp_path / "unopened-model").exists()


def test_official_main_rejects_nonfinal_capture_receipt_before_source_module_or_adapter_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _adapter, _services, _events = configured_run(tmp_path / "configured")
    ruler_receipts = ruler_receipt_directory(tmp_path / "ruler-receipts")
    artifacts = {
        "identity.json": config.frozen_identity_bytes,
        "source.json": config.repository_source_manifest_bytes,
        "runtime.json": config.runtime_manifest_bytes,
        "model.json": config.model_file_manifest_bytes,
        "parquet.json": config.parquet_materialization_manifest_bytes,
    }
    for name, payload in artifacts.items():
        (tmp_path / name).write_bytes(payload)
    malformed_receipt = runner.canonical_json_bytes({})
    (tmp_path / "capture-provenance.json").write_bytes(malformed_receipt)

    git_record = {"sha256": "d" * 64, "size_bytes": 123}
    bootstrap = runner.BootstrapSource(
        manifest={"git_executable": git_record},
        source_commit=config.expected_source_commit,
        entries={},
    )
    monkeypatch.setattr(runner, "_bootstrap_source_manifest", lambda *_args, **_kwargs: bootstrap)
    monkeypatch.setattr(
        runner,
        "_authenticate_git_executable",
        lambda _path: SimpleNamespace(path=tmp_path / "git.exe", **git_record),
    )
    touched: list[str] = []
    monkeypatch.setattr(
        runner,
        "_load_exact_source_module",
        lambda *_args, **_kwargs: (
            touched.append("source-module") or pytest.fail("source modules must not load")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_adapter",
        lambda *_args, **_kwargs: touched.append("adapter") or pytest.fail("adapter must not load"),
    )
    monkeypatch.setattr(runner, "_AUTHENTICATED_CALIBRATION_API", None)
    monkeypatch.setattr(runner, "_AUTHENTICATED_IDENTITY_RESOLVER", None)
    monkeypatch.setattr(runner, "_AUTHENTICATED_SOURCE_VERIFIER", None)

    arguments = official_cli_arguments(tmp_path, ruler_receipts=ruler_receipts)
    replacements = {
        "--expected-capture-provenance-receipt-sha256": digest(malformed_receipt),
        "--expected-model-file-manifest-sha256": digest(config.model_file_manifest_bytes),
        "--expected-parquet-materialization-manifest-sha256": digest(
            config.parquet_materialization_manifest_bytes
        ),
        "--expected-runtime-manifest-sha256": digest(config.runtime_manifest_bytes),
        "--source-commit": config.expected_source_commit,
    }
    for option, value in replacements.items():
        arguments[arguments.index(option) + 1] = value

    with pytest.raises(runner.CalibrationRunError, match="capture provenance receipt"):
        runner._official_main(
            arguments,
            runtime_context=runner.SealedRuntimeContext(
                manifest_file_sha256=digest(config.runtime_manifest_bytes),
                base_runtime_root=tmp_path / "base",
                package_roots={"packages": tmp_path / "packages"},
                package_import_paths={"packages": "Lib/site-packages"},
                git_executable_path=tmp_path / "git.exe",
                pycache_prefix=tmp_path / "pycache",
            ),
            interpreter_path=tmp_path / "base" / "python.exe",
        )

    assert touched == []
    assert not (tmp_path / "output").exists()


def test_public_main_rejects_unsealed_official_execution() -> None:
    with pytest.raises(runner.CalibrationRunError, match="launch_static_q468"):
        runner.main(["--frozen-identity", "identity.json"])


def test_official_loader_rejects_generic_and_preloaded_adapters() -> None:
    kwargs = {
        "repository_root": SCRIPT.parents[1],
        "source_entry": {},
        "calibration_api": api,
        "context": object(),
    }
    with pytest.raises(runner.CalibrationRunError, match="requires exactly"):
        runner._load_adapter("untrusted.module:factory", **kwargs)

    sys.modules[runner.CANONICAL_ADAPTER_MODULE] = object()  # type: ignore[assignment]
    try:
        with pytest.raises(runner.CalibrationRunError, match="preloaded reviewed adapter"):
            runner._load_adapter(runner.CANONICAL_ADAPTER_SPEC, **kwargs)
    finally:
        sys.modules.pop(runner.CANONICAL_ADAPTER_MODULE, None)


def test_runner_runtime_context_constructs_the_real_reviewed_adapter() -> None:
    code = f"""
import hashlib
import importlib.util
import sys
from pathlib import Path

repository_root = Path({str(SCRIPT.parents[1])!r})
runner_path = repository_root / 'scripts' / 'run_static_q468_calibration.py'
spec = importlib.util.spec_from_file_location('isolated_runner_context_integration', runner_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def source_entry(relative_path):
    payload = (repository_root / relative_path).read_bytes()
    return {{'raw_sha256': hashlib.sha256(payload).hexdigest()}}

calibration_api = module._load_exact_source_module(
    module.CALIBRATION_API_MODULE,
    module.CALIBRATION_API_PATH,
    repository_root=repository_root,
    entry=source_entry(module.CALIBRATION_API_PATH),
)
git_executable = repository_root / 'authenticated-tools' / 'git.exe'
runtime_context = module.SealedRuntimeContext(
    manifest_file_sha256='1' * 64,
    base_runtime_root=repository_root / 'runtime' / 'base',
    git_executable_path=git_executable,
    package_roots={{
        'calibration-packages': repository_root / 'runtime' / 'packages'
    }},
    package_import_paths={{'calibration-packages': 'Lib/site-packages'}},
    pycache_prefix=repository_root / 'unopened-pycache',
)
context = module._adapter_construction_context(
    calibration_api=calibration_api,
    repository_root=repository_root,
    model_root=repository_root / 'unopened-model',
    cache_root=repository_root / 'unopened-cache',
    ruler_root=repository_root / 'unopened-ruler',
    repository_source_manifest_bytes=b'source',
    calibration_runtime_manifest_bytes=b'runtime',
    model_file_manifest_bytes=b'model',
    parquet_materialization_manifest_bytes=b'parquet',
    runtime_context=runtime_context,
    interpreter_path=repository_root / 'runtime' / 'base' / 'python.exe',
)
assert set(context.runtime_authentication_context) == (
    calibration_api.RUNTIME_AUTHENTICATION_CONTEXT_KEYS
)
adapter = module._load_adapter(
    module.CANONICAL_ADAPTER_SPEC,
    repository_root=repository_root,
    source_entry=source_entry(module.CANONICAL_ADAPTER_PATH),
    calibration_api=calibration_api,
    context=context,
)
assert adapter._runtime_authentication_context['git_executable'] == git_executable
assert adapter._runtime_authentication_context['package_runtime_roots'] == {{
    'calibration-packages': repository_root / 'runtime' / 'packages'
}}
assert not (repository_root / 'unopened-model').exists()
assert not (repository_root / 'unopened-cache').exists()
assert not (repository_root / 'unopened-ruler').exists()
assert not (repository_root / 'unopened-pycache').exists()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_runner_context_reaches_real_capture_authentication_before_data_access() -> None:
    code = f"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

repository_root = Path({str(SCRIPT.parents[1])!r})
runner_path = repository_root / 'scripts' / 'run_static_q468_calibration.py'
spec = importlib.util.spec_from_file_location('isolated_runner_capture_integration', runner_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def source_entry(relative_path):
    payload = (repository_root / relative_path).read_bytes()
    return {{'raw_sha256': hashlib.sha256(payload).hexdigest()}}

calibration_api = module._load_exact_source_module(
    module.CALIBRATION_API_MODULE,
    module.CALIBRATION_API_PATH,
    repository_root=repository_root,
    entry=source_entry(module.CALIBRATION_API_PATH),
)
capture_source_path = 'scripts/capture_static_q468_identity_input.py'
capture_payload = (repository_root / capture_source_path).read_bytes()
source_manifest_bytes = json.dumps(
    {{
        'paths': [
            {{
                'path': capture_source_path,
                'raw_sha256': hashlib.sha256(capture_payload).hexdigest(),
            }}
        ],
        'schema': 'recurquant.experiment013.source-manifest.v2',
    }},
    sort_keys=True,
    separators=(',', ':'),
).encode('utf-8')

base_runtime_root = repository_root / 'unopened-boundary-runtime'
package_runtime_root = repository_root / 'unopened-boundary-packages'
git_executable = repository_root / 'unopened-boundary-tools' / 'git.exe'
model_root = repository_root / 'unopened-boundary-model'
cache_root = repository_root / 'unopened-boundary-cache'
ruler_root = repository_root / 'unopened-boundary-ruler'
pycache_root = repository_root / 'unopened-boundary-pycache'
runtime_context = module.SealedRuntimeContext(
    manifest_file_sha256='1' * 64,
    base_runtime_root=base_runtime_root,
    git_executable_path=git_executable,
    package_roots={{'calibration-packages': package_runtime_root}},
    package_import_paths={{'calibration-packages': 'Lib/site-packages'}},
    pycache_prefix=pycache_root,
)
context = module._adapter_construction_context(
    calibration_api=calibration_api,
    repository_root=repository_root,
    model_root=model_root,
    cache_root=cache_root,
    ruler_root=ruler_root,
    repository_source_manifest_bytes=source_manifest_bytes,
    calibration_runtime_manifest_bytes=b'runtime-manifest',
    model_file_manifest_bytes=b'model-manifest',
    parquet_materialization_manifest_bytes=b'parquet-manifest',
    runtime_context=runtime_context,
    interpreter_path=base_runtime_root / 'python.exe',
)
adapter = module._load_adapter(
    module.CANONICAL_ADAPTER_SPEC,
    repository_root=repository_root,
    source_entry=source_entry(module.CANONICAL_ADAPTER_PATH),
    calibration_api=calibration_api,
    context=context,
)
adapter_module = sys.modules[module.CANONICAL_ADAPTER_MODULE]
assert adapter_module.CAPTURE_SOURCE_PATH == capture_source_path
assert adapter_module.SOURCE_MANIFEST_SCHEMA == (
    'recurquant.experiment013.source-manifest.v2'
)
capture_binding = adapter_module._load_capture_module(
    repository_root,
    source_manifest_bytes,
)

observed = {{}}

class CaptureAuthenticationBoundaryReached(RuntimeError):
    pass

def stop_before_artifact_hub_or_data_access(artifacts, *, runtime_context, **kwargs):
    assert kwargs == {{}}
    observed['artifacts'] = dict(artifacts)
    observed['runtime_context_type'] = type(runtime_context).__name__
    observed['base_runtime_root'] = runtime_context.base_runtime_root
    observed['git_executable'] = runtime_context.git_executable
    observed['staged_interpreter'] = runtime_context.staged_interpreter
    observed['package_runtime_roots'] = dict(runtime_context.package_runtime_roots)
    observed['package_import_paths'] = dict(runtime_context.package_import_paths)
    raise CaptureAuthenticationBoundaryReached

capture_binding.module._authenticate_execution_binding_artifacts = (
    stop_before_artifact_hub_or_data_access
)
adapter_module._load_capture_module = lambda _root, _manifest: capture_binding

try:
    adapter.materialize_sequence({{'identity_record_sha256': '0' * 64}})
except CaptureAuthenticationBoundaryReached:
    pass
else:
    raise AssertionError('real capture authentication boundary was not reached')

assert observed == {{
    'artifacts': {{
        'calibration_runtime_manifest_file_sha256': b'runtime-manifest',
        'model_file_manifest_file_sha256': b'model-manifest',
        'parquet_materialization_manifest_file_sha256': b'parquet-manifest',
        'repository_source_manifest_file_sha256': source_manifest_bytes,
    }},
    'runtime_context_type': '_RuntimeAuthenticationContext',
    'base_runtime_root': base_runtime_root,
    'git_executable': git_executable,
    'staged_interpreter': base_runtime_root / 'python.exe',
    'package_runtime_roots': {{'calibration-packages': package_runtime_root}},
    'package_import_paths': {{'calibration-packages': 'Lib/site-packages'}},
}}
assert adapter._execution_binding_artifacts is None
assert adapter._runtime_authentication_context is None
assert capture_binding.module._CALIBRATION_RUNNER_MODULE_NAME not in sys.modules
assert {{'datasets', 'huggingface_hub', 'transformers'}}.isdisjoint(sys.modules)
assert not any(
    path.exists()
    for path in (
        base_runtime_root,
        package_runtime_root,
        git_executable.parent,
        model_root,
        cache_root,
        ruler_root,
        pycache_root,
    )
)
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_default_services_do_not_eagerly_import_static_calibration_modules() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path
script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('isolated_runner', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'recurquant.static_q468' not in sys.modules
api_script = script.parents[1] / 'src' / 'recurquant' / 'experiment013_calibration_api.py'
api_spec = importlib.util.spec_from_file_location('isolated_calibration_api', api_script)
api_module = importlib.util.module_from_spec(api_spec)
sys.modules[api_spec.name] = api_module
api_spec.loader.exec_module(api_module)
module.default_services(
    script.parents[1],
    base_runtime_root=script.parents[1] / 'base',
    calibration_api=api_module,
    interpreter_path=script.parents[1] / 'base' / 'python.exe',
    package_roots={{'packages': script.parents[1] / 'packages'}},
)
assert 'recurquant.static_q468' not in sys.modules
assert 'recurquant.static_q468_calibration' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
