from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch_static_q468_calibration.py"
SPEC = importlib.util.spec_from_file_location("launch_static_q468_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sealed_fixture(tmp_path: Path) -> dict[str, Any]:
    base = tmp_path / "base"
    packages = tmp_path / "packages"
    repository = tmp_path / "repository"
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    artifacts = tmp_path / "artifacts"
    git_executable = _write(tmp_path / "toolchain" / "git.exe", b"fixture git executable\n")
    git_executable, git_record = launcher._authenticated_git_executable(git_executable)

    interpreter = _write(base / "python.exe", b"staged interpreter\n")
    _write(base / "Lib" / "marker.py", b"BASE = True\n")
    import_root = packages / "Lib" / "site-packages"
    _write(import_root / "demo" / "__init__.py", b"VALUE = 1\n")
    metadata = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n"
    _write(import_root / "demo-1.0.dist-info" / "METADATA", metadata)
    record = b"demo-1.0.dist-info/METADATA,,\ndemo-1.0.dist-info/RECORD,,\ndemo/__init__.py,,\n"
    _write(import_root / "demo-1.0.dist-info" / "RECORD", record)
    for (
        module_name,
        distribution_name,
    ) in launcher.CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS.items():
        dist_info_name = f"{distribution_name.replace('-', '_')}-1.0.dist-info"
        module_relative_path = (
            f"{module_name}.py" if module_name == "six" else f"{module_name}/__init__.py"
        )
        _write(
            import_root / module_relative_path,
            f"NAME = {module_name!r}\n".encode(),
        )
        _write(
            import_root / dist_info_name / "METADATA",
            (f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 1.0\n\n").encode(),
        )
        _write(
            import_root / dist_info_name / "RECORD",
            (
                f"{dist_info_name}/METADATA,,\n"
                f"{dist_info_name}/RECORD,,\n"
                f"{module_relative_path},,\n"
            ).encode(),
        )

    package_roots = {"packages": packages.resolve(strict=True)}
    import_paths = {"packages": "Lib/site-packages"}
    base_files = list(launcher._tree_files(base, context="fixture base"))
    package_files = list(launcher._tree_files(packages, context="fixture packages"))
    interpreter_record = launcher._stable_file_record(
        interpreter,
        relative="python.exe",
        context="fixture interpreter",
    )
    runtime = {
        "artifact_kind": launcher.RUNTIME_MANIFEST_KIND,
        "base_runtime_root": launcher.BASE_RUNTIME_ROOT_NAME,
        "base_sys_path": ["Lib", "."],
        "distributions": list(launcher._distribution_inventory(package_roots, import_paths)),
        "git_executable": git_record,
        "interpreter": {
            "relative_path": "python.exe",
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": interpreter_record["sha256"],
            "size_bytes": interpreter_record["size_bytes"],
        },
        "launch_policy": launcher.SEALED_LAUNCH_POLICY,
        "machine": {
            "architecture": platform.architecture()[0],
            "byteorder": sys.byteorder,
            "machine": platform.machine(),
            "pointer_bits": 8 * struct.calcsize("P"),
            "system": platform.system(),
        },
        "package_roots": [{"import_path": "Lib/site-packages", "name": "packages"}],
        "python": {
            "abi_flags": getattr(sys, "abiflags", ""),
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "runtime_trees": [
            {"files": base_files, "kind": "base-runtime", "name": "base-runtime"},
            {"files": package_files, "kind": "packages", "name": "packages"},
        ],
        "schema_version": launcher.RUNTIME_MANIFEST_SCHEMA,
    }
    runtime_path = _write(
        artifacts / "runtime.json",
        launcher._canonical_json_bytes(runtime),
    )

    runner_path = _write(
        repository / launcher.RUNNER_SOURCE_PATH,
        (
            b"def sealed_main(argv, *, base_runtime_root, package_roots, "
            b"package_import_paths, interpreter_path, pycache_prefix, "
            b"git_executable_path):\n"
            b"    return 0\n"
        ),
    )
    capture_source_path = _write(
        repository / launcher.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
        b"CAPTURE_VERSION = 6\n",
    )
    source_payload = {
        "git_executable": {
            "sha256": git_record["sha256"],
            "size_bytes": git_record["size_bytes"],
        },
        "object_format": "sha1",
        "paths": sorted(
            [
                {
                    "git_blob_oid": "c" * 40,
                    "index_blob_oid": "c" * 40,
                    "mode": "100644",
                    "path": launcher.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
                    "raw_sha256": _sha256(capture_source_path.read_bytes()),
                    "worktree_blob_oid": "c" * 40,
                },
                {
                    "git_blob_oid": "b" * 40,
                    "index_blob_oid": "b" * 40,
                    "mode": "100644",
                    "path": launcher.RUNNER_SOURCE_PATH,
                    "raw_sha256": _sha256(runner_path.read_bytes()),
                    "worktree_blob_oid": "b" * 40,
                },
            ],
            key=lambda item: item["path"],
        ),
        "profile": "experiment-013-static-q468-frozen-source-v2",
        "repository_binding": {},
        "schema": "recurquant.experiment013.source-manifest.v2",
        "source_commit": "a" * 40,
    }
    source = {
        **source_payload,
        "canonical_manifest_sha256": _sha256(launcher._pretty_json_bytes(source_payload)),
    }
    source_path = _write(
        artifacts / "source.json",
        launcher._pretty_json_bytes(source),
    )
    model_path = _write(artifacts / "model.json", b'{"model":"fixture"}\n')
    parquet_path = _write(artifacts / "parquet.json", b'{"parquet":"fixture"}\n')
    ruler_receipt_dir = tmp_path / "ruler-receipts"
    ruler_receipt_dir.mkdir()
    bindings = {
        "calibration_runtime_manifest_file_sha256": _sha256(runtime_path.read_bytes()),
        "model_file_manifest_file_sha256": _sha256(model_path.read_bytes()),
        "parquet_materialization_manifest_file_sha256": _sha256(parquet_path.read_bytes()),
        "repository_source_manifest_file_sha256": _sha256(source_path.read_bytes()),
    }
    identity_input_bytes = launcher._canonical_json_bytes({"identity_input": "fixture"})
    evidence = {
        "execution_bindings": bindings,
        "identity_only": True,
        "phase": "calibration",
        "promotion_required": False,
        "schema_version": launcher.IDENTITY_SCHEMA,
        "source_manifest_sha256": _sha256(identity_input_bytes),
        "status": "frozen",
    }
    identity = {
        "canonical_evidence_sha256": _sha256(launcher._canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    identity_path = _write(
        artifacts / "identity.json",
        launcher._canonical_json_bytes(identity),
    )
    receipt_bytes = _capture_candidate(
        {
            "bindings": bindings,
            "runtime_manifest": runtime,
            "source_manifest": {
                "paths": [
                    {"path": item["path"], "raw_sha256": item["raw_sha256"]}
                    for item in source["paths"]
                ],
                "source_commit": source["source_commit"],
            },
        },
        identity_input_bytes,
    )
    receipt_path = _write(artifacts / "capture-provenance.json", receipt_bytes)
    runner_arguments = [
        "--fisher-h1-smoke",
        "--frozen-identity",
        str(identity_path),
        "--capture-provenance-receipt",
        str(receipt_path),
        "--expected-capture-provenance-receipt-sha256",
        _sha256(receipt_bytes),
        "--cache-root",
        str(cache_root),
        "--repository-source-manifest",
        str(source_path),
        "--model-file-manifest",
        str(model_path),
        "--expected-model-file-manifest-sha256",
        bindings["model_file_manifest_file_sha256"],
        "--parquet-materialization-manifest",
        str(parquet_path),
        "--expected-parquet-materialization-manifest-sha256",
        bindings["parquet_materialization_manifest_file_sha256"],
        "--runtime-manifest",
        str(runtime_path),
        "--expected-runtime-manifest-sha256",
        bindings["calibration_runtime_manifest_file_sha256"],
        "--repository-root",
        str(repository),
        "--source-commit",
        source["source_commit"],
        "--ruler-receipt-dir",
        str(ruler_receipt_dir),
    ]
    host_arguments = [
        "--base-runtime-root",
        str(base),
        "--git-executable",
        str(git_executable),
        "--package-root",
        f"packages={packages.resolve(strict=True)}",
        "--runtime-manifest",
        str(runtime_path),
        "--",
        *runner_arguments,
    ]
    return {
        "base": base,
        "bindings": bindings,
        "cache_root": cache_root,
        "git_executable": git_executable,
        "host_arguments": host_arguments,
        "identity_input_bytes": identity_input_bytes,
        "model_path": model_path,
        "packages": packages,
        "parquet_path": parquet_path,
        "repository": repository,
        "capture_provenance_receipt_bytes": receipt_bytes,
        "capture_provenance_receipt_path": receipt_path,
        "ruler_receipt_dir": ruler_receipt_dir,
        "runner_arguments": runner_arguments,
        "runtime_manifest": runtime,
        "runtime_path": runtime_path,
        "source_manifest": {
            "file_sha256": _sha256(source_path.read_bytes()),
            "git_executable": source["git_executable"],
            "paths": [
                {"path": item["path"], "raw_sha256": item["raw_sha256"]} for item in source["paths"]
            ],
            "source_commit": source["source_commit"],
        },
        "source_path": source_path,
    }


def _stage_a_binding_bytes(fixture: dict[str, Any]) -> tuple[bytes, str]:
    authorization_evidence = {
        "artifact_revision": launcher.STAGE_A_CALIBRATION_AUTHORIZATION_REVISION,
        "authorized_output_file_sha256": {},
        "bindings": {
            "calibration_core_binding_file_sha256": "1" * 64,
            "calibration_run_report_file_sha256": "2" * 64,
            "capture_provenance_receipt_file_sha256": "3" * 64,
            "execution_bindings": fixture["bindings"],
            "fisher_h1_smoke_report_file_sha256": "4" * 64,
            "frozen_calibration_identity_file_sha256": "5" * 64,
            "identity_input_manifest_sha256": "6" * 64,
            "source_commit": fixture["source_manifest"]["source_commit"],
            "static_q48_policy_file_sha256": "7" * 64,
        },
        "dependencies_base64": {},
        "dependency_file_sha256": {},
        "status": launcher.STAGE_A_CALIBRATION_AUTHORIZATION_STATUS,
    }
    authorization_bytes = launcher._canonical_json_bytes(
        {
            "artifact_kind": launcher.STAGE_A_CALIBRATION_AUTHORIZATION_KIND,
            "canonical_evidence_sha256": _sha256(
                launcher._canonical_json_bytes(authorization_evidence)
            ),
            "evidence": authorization_evidence,
            "schema_version": launcher.STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA,
        }
    )
    authorization_sha256 = _sha256(authorization_bytes)
    binding_values = {
        name: f"{index:064x}"
        for index, name in enumerate(sorted(launcher._STAGE_A_CALIBRATION_BINDING_FIELDS), start=10)
    }
    binding_values["calibration_authorization_file_sha256"] = authorization_sha256
    binding_evidence = {
        "artifact_revision": launcher.STAGE_A_CALIBRATION_BINDING_REVISION,
        "binding": binding_values,
        "dependencies_base64": {
            "calibration_authorization_artifact": base64.b64encode(authorization_bytes).decode(
                "ascii"
            )
        },
        "dependency_file_sha256": {"calibration_authorization_artifact": authorization_sha256},
    }
    return (
        launcher._canonical_json_bytes(
            {
                "artifact_kind": launcher.STAGE_A_CALIBRATION_BINDING_KIND,
                "canonical_evidence_sha256": _sha256(
                    launcher._canonical_json_bytes(binding_evidence)
                ),
                "evidence": binding_evidence,
                "schema_version": launcher.STAGE_A_CALIBRATION_BINDING_SCHEMA,
            }
        ),
        authorization_sha256,
    )


def _capture_fixture(tmp_path: Path, *, stage_a: bool = False) -> dict[str, Any]:
    fixture = _sealed_fixture(tmp_path)
    output_parent = tmp_path / "capture-output"
    output_parent.mkdir()
    identity_output = output_parent / "identity-input.json"
    receipt_output = output_parent / "capture-provenance.json"
    capture_arguments = [
        "capture-stage-a-identity" if stage_a else "capture-calibration-identity",
        "--repository-root",
        str(fixture["repository"]),
        "--source-commit",
        str(fixture["source_manifest"]["source_commit"]),
        "--repository-source-manifest",
        str(fixture["source_path"]),
        "--expected-repository-source-manifest-sha256",
        fixture["bindings"]["repository_source_manifest_file_sha256"],
        "--runtime-manifest",
        str(fixture["runtime_path"]),
        "--expected-runtime-manifest-sha256",
        fixture["bindings"]["calibration_runtime_manifest_file_sha256"],
        "--model-file-manifest",
        str(fixture["model_path"]),
        "--expected-model-file-manifest-sha256",
        fixture["bindings"]["model_file_manifest_file_sha256"],
        "--parquet-materialization-manifest",
        str(fixture["parquet_path"]),
        "--expected-parquet-materialization-manifest-sha256",
        fixture["bindings"]["parquet_materialization_manifest_file_sha256"],
        "--cache-root",
        str(fixture["cache_root"]),
        "--ruler-receipt-dir",
        str(fixture["ruler_receipt_dir"]),
        "--output",
        str(identity_output),
        "--capture-provenance-receipt-output",
        str(receipt_output),
    ]
    if stage_a:
        binding_bytes, authorization_sha256 = _stage_a_binding_bytes(fixture)
        binding_path = _write(output_parent / "stage-a-binding.json", binding_bytes)
        capture_arguments.extend(
            [
                "--stage-a-calibration-binding",
                str(binding_path),
                "--expected-stage-a-calibration-binding-sha256",
                _sha256(binding_bytes),
            ]
        )
    separator = fixture["host_arguments"].index("--")
    fixture.update(
        {
            "capture_arguments": capture_arguments,
            "capture_host_arguments": [
                *fixture["host_arguments"][: separator + 1],
                *capture_arguments,
            ],
            "identity_output": identity_output,
            "receipt_output": receipt_output,
            "stage_a": stage_a,
        }
    )
    if stage_a:
        fixture["stage_a_binding_bytes"] = binding_bytes
        fixture["stage_a_binding_path"] = binding_path
        fixture["stage_a_authorization_sha256"] = authorization_sha256
    return fixture


def _capture_candidate(fixture: dict[str, Any], identity_bytes: bytes) -> bytes:
    runtime_manifest = fixture["runtime_manifest"]
    distributions = {item["name"]: item for item in runtime_manifest["distributions"]}
    runtime_trees = {
        item["name"]: {record["path"]: record for record in item["files"]}
        for item in runtime_manifest["runtime_trees"]
    }
    source_entries = {
        item["path"]: item["raw_sha256"] for item in fixture["source_manifest"]["paths"]
    }
    origins: list[dict[str, object]] = []
    for module_name in sorted(launcher.CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS):
        distribution_name = launcher.CALIBRATION_IDENTITY_CRITICAL_MODULE_DISTRIBUTIONS[module_name]
        distribution = distributions[distribution_name]
        relative_path = (
            "Lib/site-packages/six.py"
            if module_name == "six"
            else f"Lib/site-packages/{module_name}/__init__.py"
        )
        file_record = runtime_trees[distribution["package_root"]][relative_path]
        origins.append(
            {
                "distribution": distribution_name,
                "module": module_name,
                "package_root": distribution["package_root"],
                "relative_path": relative_path,
                "sha256": file_record["sha256"],
                "size_bytes": file_record["size_bytes"],
                "version": distribution["version"],
            }
        )
    phase = "stage_a" if fixture.get("stage_a") else "calibration"
    candidate: dict[str, object] = {
        "artifact_kind": (
            launcher.STAGE_A_IDENTITY_CAPTURE_PROVENANCE_KIND
            if phase == "stage_a"
            else launcher.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_KIND
        ),
        "capture_source": {
            "path": launcher.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH,
            "sha256": source_entries[launcher.CALIBRATION_IDENTITY_CAPTURE_SOURCE_PATH],
        },
        "capture_version": launcher.CALIBRATION_IDENTITY_CAPTURE_VERSION,
        "critical_module_origins": origins,
        "excluded_runtime_modules": list(launcher.CALIBRATION_IDENTITY_EXCLUDED_RUNTIME_MODULES),
        "execution_bindings": fixture["bindings"],
        "identity_input_file_sha256": _sha256(identity_bytes),
        "phase": phase,
        "publication_contract": (
            launcher.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_PUBLICATION_CONTRACT
        ),
        "runner_revision": launcher.RUNNER_REVISION,
        "schema_version": (
            launcher.STAGE_A_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
            if phase == "stage_a"
            else launcher.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_SCHEMA
        ),
        "source_commit": fixture["source_manifest"]["source_commit"],
        "status": launcher.CALIBRATION_IDENTITY_CAPTURE_PROVENANCE_STATUS,
    }
    if phase == "stage_a":
        candidate["calibration_binding_file_sha256"] = _sha256(fixture["stage_a_binding_bytes"])
        candidate["calibration_authorization_file_sha256"] = fixture["stage_a_authorization_sha256"]
    return launcher._canonical_json_bytes(candidate)


def _run_embedded_manifest_boundary(
    fixture: dict[str, Any],
    *,
    pycache: Path,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    pycache.mkdir()
    scratch = pycache.parent / f"{pycache.name}-scratch"
    scratch.mkdir()
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={pycache}",
        "-X",
        "utf8",
        "-c",
        launcher.SEALED_STDIN_LOADER,
        str(fixture["runtime_path"]),
        str(fixture["base"]),
        json.dumps(
            {"packages": str(fixture["packages"])},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        str(pycache),
        str(fixture["git_executable"]),
        str(scratch),
        *fixture["runner_arguments"],
    ]
    capture_profile = bool(
        fixture["runner_arguments"]
        and fixture["runner_arguments"][0] in launcher._CAPTURE_COMMANDS
    )
    if capture_profile:
        launcher._prepare_capture_hub_cache_root(fixture["cache_root"])
    environment = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=fixture["cache_root"],
        capture_profile=capture_profile,
    )
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        cwd=scratch,
        env=environment,
        input=launcher.SEALED_BOOTSTRAP_BYTES,
    )


def test_launch_uses_exact_isolated_command_and_reauthenticates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "untrusted-venv"))
    monkeypatch.setenv("VIRTUAL_ENV_PROMPT", "untrusted")
    monkeypatch.setenv("PATH", str(tmp_path / "fake-path"))
    events: list[str] = []
    commands: list[tuple[list[str], Path, dict[str, str], bytes]] = []
    original_git_bytes = fixture["git_executable"].read_bytes()
    verify_bound = launcher._verify_bound_artifacts
    verify_runtime = launcher._verify_runtime
    verify_pycache = launcher._verify_empty_pycache
    cleanup_temporary = launcher._cleanup_owned_temporary_directory

    def assert_git_locked() -> None:
        if os.name == "nt":
            with pytest.raises(OSError):
                fixture["git_executable"].write_bytes(b"mutated outside custody\n")

    def bound_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("bound")
        if events.count("bound") >= 2:
            assert_git_locked()
        return verify_bound(*args, **kwargs)

    def runtime_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("runtime")
        if events.count("runtime") >= 2:
            assert_git_locked()
        return verify_runtime(*args, **kwargs)

    def pycache_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("pycache")
        assert_git_locked()
        return verify_pycache(*args, **kwargs)

    def cleanup_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("cleanup")
        assert_git_locked()
        return cleanup_temporary(*args, **kwargs)

    def run_wrapper(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
        input: bytes,
    ) -> subprocess.CompletedProcess[str]:
        events.append("run")
        assert check is False
        assert_git_locked()
        commands.append((command, cwd, env, input))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher, "_verify_bound_artifacts", bound_wrapper)
    monkeypatch.setattr(launcher, "_verify_runtime", runtime_wrapper)
    monkeypatch.setattr(launcher, "_verify_empty_pycache", pycache_wrapper)
    monkeypatch.setattr(launcher, "_cleanup_owned_temporary_directory", cleanup_wrapper)
    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    assert launcher.launch(fixture["host_arguments"]) == 0
    assert events == [
        "bound",
        "runtime",
        "bound",
        "runtime",
        "pycache",
        "run",
        "pycache",
        "bound",
        "runtime",
        "cleanup",
        "cleanup",
    ]
    assert len(commands) == 1
    command, cwd, environment, stdin_payload = commands[0]
    assert command[0] == str((fixture["base"] / "python.exe").resolve(strict=True))
    assert command[1:5] == ["-I", "-S", "-B", "-X"]
    assert command[5].startswith("pycache_prefix=")
    assert command[6:10] == ["-X", "utf8", "-c", launcher.SEALED_STDIN_LOADER]
    assert launcher.SEALED_BOOTSTRAP not in command
    assert len(subprocess.list2cmdline(command)) < 32_767
    assert command[10] == str(fixture["runtime_path"].resolve(strict=True))
    assert command[11] == str(fixture["base"].resolve(strict=True))
    assert json.loads(command[12]) == {"packages": str(fixture["packages"].resolve(strict=True))}
    assert command[14] == str(fixture["git_executable"])
    assert command[16:] == fixture["runner_arguments"]
    assert not Path(command[13]).exists()
    assert not Path(command[15]).exists()
    assert cwd == Path(command[15])
    assert all(not key.upper().startswith("PYTHON") for key in environment)
    assert "VIRTUAL_ENV" not in environment
    assert "VIRTUAL_ENV_PROMPT" not in environment
    assert "PATH" not in environment
    assert environment["TEMP"] == environment["TMP"] == command[15]
    assert (
        environment["HOME"] == environment["USERPROFILE"] == str(Path(command[15]) / "private-home")
    )
    assert environment["HF_DATASETS_CACHE"] == str(fixture["cache_root"] / "datasets")
    assert stdin_payload == launcher.SEALED_BOOTSTRAP_BYTES
    fixture["git_executable"].write_bytes(original_git_bytes)
    assert fixture["git_executable"].read_bytes() == original_git_bytes


def test_executable_custody_holds_authenticated_files_until_release(tmp_path: Path) -> None:
    python_path = Path(sys.executable).resolve(strict=True)
    git_path = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    python_record = launcher._stable_file_record(
        python_path,
        relative=python_path.name,
        context="test Python executable",
    )
    git_path, git_record = launcher._authenticated_git_executable(git_path)
    runtime = {
        "git_executable": git_record,
        "interpreter": {
            "relative_path": python_path.name,
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": python_record["sha256"],
            "size_bytes": python_record["size_bytes"],
        },
        "machine": {"system": platform.system()},
    }
    original_git = git_path.read_bytes()
    replacement = _write(git_path.with_name("replacement.exe"), b"replacement\n")
    moved = git_path.with_name("moved.exe")

    with launcher._acquire_executable_custody(
        interpreter=python_path,
        git_executable=git_path,
        runtime_manifest=runtime,
    ) as custody:
        custody.verify()
        assert custody.record["mode"] == launcher.EXECUTABLE_CUSTODY_MODE
        assert custody.record["protocol_eligible"] is (os.name == "nt")
        if os.name == "nt":
            with pytest.raises(OSError):
                git_path.write_bytes(b"mutated while held\n")
            with pytest.raises(OSError):
                git_path.unlink()
            with pytest.raises(OSError):
                git_path.rename(moved)
            with pytest.raises(OSError):
                os.replace(replacement, git_path)
            completed = subprocess.run(
                [str(python_path), "-I", "-S", "-B", "-c", "print('custody-ok')"],
                check=True,
                capture_output=True,
                text=True,
            )
            assert completed.stdout == "custody-ok\n"

        custody.close()
        with pytest.raises(launcher.SealedLaunchError, match="released"):
            custody.verify()

    if os.name == "nt":
        git_path.rename(moved)
        moved.rename(git_path)
        os.replace(replacement, git_path)
        assert git_path.read_bytes() == b"replacement\n"
        git_path.unlink()
    git_path.write_bytes(original_git)
    assert git_path.read_bytes() == original_git


def test_executable_custody_allows_authenticated_python_and_git_launches() -> None:
    git_candidate = shutil.which("git")
    assert git_candidate is not None
    python_path = Path(sys.executable).resolve(strict=True)
    git_path, git_record = launcher._authenticated_git_executable(Path(git_candidate))
    python_record = launcher._stable_file_record(
        python_path,
        relative=python_path.name,
        context="test Python executable",
    )
    runtime = {
        "git_executable": git_record,
        "interpreter": {
            "relative_path": python_path.name,
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": python_record["sha256"],
            "size_bytes": python_record["size_bytes"],
        },
        "machine": {"system": platform.system()},
    }

    with launcher._acquire_executable_custody(
        interpreter=python_path,
        git_executable=git_path,
        runtime_manifest=runtime,
    ) as custody:
        python_result = subprocess.run(
            [str(python_path), "-I", "-S", "-B", "-c", "print('custody-python-ok')"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_result = subprocess.run(
            [str(git_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        custody.verify()

    assert python_result.stdout == "custody-python-ok\n"
    assert git_result.stdout.startswith("git version ")


def test_executable_custody_rejects_an_empty_unacquired_handle_set() -> None:
    custody = launcher._HeldExecutableCustody(entries=())

    with pytest.raises(
        launcher.SealedLaunchError,
        match="does not contain the authenticated Python and Git roles",
    ):
        custody.verify()


def test_executable_custody_rejects_dead_handles_even_for_exact_roles(tmp_path: Path) -> None:
    python_path = Path(sys.executable).resolve(strict=True)
    git_path = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    python_record = launcher._stable_file_record(
        python_path,
        relative=python_path.name,
        context="test Python executable",
    )
    git_path, git_record = launcher._authenticated_git_executable(git_path)
    runtime = {
        "git_executable": git_record,
        "interpreter": {
            "relative_path": python_path.name,
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": python_record["sha256"],
            "size_bytes": python_record["size_bytes"],
        },
        "machine": {"system": platform.system()},
    }

    with launcher._acquire_executable_custody(
        interpreter=python_path,
        git_executable=git_path,
        runtime_manifest=runtime,
    ) as custody:
        if os.name == "nt":
            forged = launcher._HeldExecutableCustody(
                entries=custody._entries,
                windows_handles=(0, 0),
            )
            error_match = "Windows executable custody handle"
        else:
            forged = launcher._HeldExecutableCustody(
                entries=custody._entries,
                posix_descriptors=(-1, -1),
            )
            error_match = "POSIX custody descriptor"
        with pytest.raises(launcher.SealedLaunchError, match=error_match):
            forged.verify()


def test_executable_custody_entry_failure_releases_all_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = Path(sys.executable).resolve(strict=True)
    git_path = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    python_record = launcher._stable_file_record(
        python_path,
        relative=python_path.name,
        context="test Python executable",
    )
    git_path, git_record = launcher._authenticated_git_executable(git_path)
    runtime = {
        "git_executable": git_record,
        "interpreter": {
            "relative_path": python_path.name,
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": python_record["sha256"],
            "size_bytes": python_record["size_bytes"],
        },
        "machine": {"system": platform.system()},
    }
    custody = launcher._acquire_executable_custody(
        interpreter=python_path,
        git_executable=git_path,
        runtime_manifest=runtime,
    )

    def fail_entry_verification() -> None:
        raise launcher.SealedLaunchError("forced custody entry failure")

    monkeypatch.setattr(custody, "verify", fail_entry_verification)
    with pytest.raises(launcher.SealedLaunchError, match="forced custody entry failure"):
        custody.__enter__()

    assert custody._closed is True
    git_path.write_bytes(b"write proves entry-failure handles were released\n")


def test_executable_custody_releases_partial_handles_after_git_path_mismatch(
    tmp_path: Path,
) -> None:
    python_path = Path(sys.executable).resolve(strict=True)
    git_path = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    python_record = launcher._stable_file_record(
        python_path,
        relative=python_path.name,
        context="test Python executable",
    )
    git_path, git_record = launcher._authenticated_git_executable(git_path)
    runtime = {
        "git_executable": {**git_record, "absolute_path_sha256": "0" * 64},
        "interpreter": {
            "relative_path": python_path.name,
            "root": launcher.BASE_RUNTIME_ROOT_NAME,
            "sha256": python_record["sha256"],
            "size_bytes": python_record["size_bytes"],
        },
        "machine": {"system": platform.system()},
    }

    with pytest.raises(launcher.SealedLaunchError, match="path differs"):
        launcher._acquire_executable_custody(
            interpreter=python_path,
            git_executable=git_path,
            runtime_manifest=runtime,
        )

    git_path.write_bytes(b"write proves partial handles were released\n")


def test_capture_candidate_authenticates_exact_source_runtime_and_bindings(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)

    assert launcher._validate_capture_provenance_candidate(
        candidate,
        bindings=fixture["bindings"],
        identity_input_file_sha256=_sha256(identity_bytes),
        runtime_manifest=fixture["runtime_manifest"],
        source_manifest=fixture["source_manifest"],
    ) == json.loads(candidate)

    stale = json.loads(candidate)
    stale["schema_version"] = 1
    with pytest.raises(launcher.SealedLaunchError, match="identity drifted"):
        launcher._validate_capture_provenance_candidate(
            launcher._canonical_json_bytes(stale),
            bindings=fixture["bindings"],
            identity_input_file_sha256=_sha256(identity_bytes),
            runtime_manifest=fixture["runtime_manifest"],
            source_manifest=fixture["source_manifest"],
        )


def test_capture_candidate_rejects_text_translation_and_trailing_bytes(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)

    for mutated_candidate in (candidate[:-1] + b"\r\n", candidate + b" "):
        with pytest.raises(launcher.SealedLaunchError, match="not canonical JSON"):
            launcher._validate_capture_provenance_candidate(
                mutated_candidate,
                bindings=fixture["bindings"],
                identity_input_file_sha256=_sha256(identity_bytes),
                runtime_manifest=fixture["runtime_manifest"],
                source_manifest=fixture["source_manifest"],
            )


def test_stage_a_capture_candidate_binds_v4_authorization_and_exact_chain(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(tmp_path, stage_a=True)
    identity_bytes = launcher._canonical_json_bytes({"identity": "stage-a-fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    binding_bytes, envelope = launcher._verify_stage_a_capture_binding(
        launcher._extract_runner_options(fixture["capture_arguments"]),
        execution_bindings=fixture["bindings"],
        source_commit=fixture["source_manifest"]["source_commit"],
    )
    assert binding_bytes == fixture["stage_a_binding_bytes"]
    assert launcher._validate_capture_provenance_candidate(
        candidate,
        bindings=fixture["bindings"],
        identity_input_file_sha256=_sha256(identity_bytes),
        runtime_manifest=fixture["runtime_manifest"],
        source_manifest=fixture["source_manifest"],
        phase="stage_a",
        stage_a_binding_envelope=envelope,
    ) == json.loads(candidate)

    for field in (
        "calibration_binding_file_sha256",
        "calibration_authorization_file_sha256",
    ):
        tampered = json.loads(candidate)
        tampered[field] = "0" * 64
        with pytest.raises(launcher.SealedLaunchError, match="Stage-A"):
            launcher._validate_capture_provenance_candidate(
                launcher._canonical_json_bytes(tampered),
                bindings=fixture["bindings"],
                identity_input_file_sha256=_sha256(identity_bytes),
                runtime_manifest=fixture["runtime_manifest"],
                source_manifest=fixture["source_manifest"],
                phase="stage_a",
                stage_a_binding_envelope=envelope,
            )


def test_stage_a_capture_rejects_binding_downgrade_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path, stage_a=True)
    binding = json.loads(fixture["stage_a_binding_bytes"])
    binding["schema_version"] = 3
    downgraded = launcher._canonical_json_bytes(binding)
    fixture["stage_a_binding_path"].write_bytes(downgraded)
    expected_index = (
        fixture["capture_host_arguments"].index("--expected-stage-a-calibration-binding-sha256") + 1
    )
    fixture["capture_host_arguments"][expected_index] = _sha256(downgraded)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess started before binding rejection"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="kind or schema"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["identity_output"].exists()
    assert not fixture["receipt_output"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "binding",
        "capture-source",
        "excluded-policy",
        "extra-field",
        "identity-input",
        "module-hash",
        "module-path",
        "module-size",
        "module-version",
        "publication-contract",
        "runner-revision",
        "schema-v1",
        "source-commit",
        "status",
    ],
)
def test_capture_candidate_rejects_every_finalization_drift(
    mutation: str,
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = json.loads(_capture_candidate(fixture, identity_bytes))
    if mutation == "binding":
        candidate["execution_bindings"]["model_file_manifest_file_sha256"] = "0" * 64
    elif mutation == "capture-source":
        candidate["capture_source"]["sha256"] = "0" * 64
    elif mutation == "excluded-policy":
        candidate["excluded_runtime_modules"] = ["setuptools"]
    elif mutation == "extra-field":
        candidate["launcher_finalized"] = True
    elif mutation == "identity-input":
        candidate["identity_input_file_sha256"] = "0" * 64
    elif mutation == "module-hash":
        candidate["critical_module_origins"][0]["sha256"] = "0" * 64
    elif mutation == "module-path":
        candidate["critical_module_origins"][0]["relative_path"] = (
            "Lib/site-packages/shadow/__init__.py"
        )
    elif mutation == "module-size":
        candidate["critical_module_origins"][0]["size_bytes"] += 1
    elif mutation == "module-version":
        candidate["critical_module_origins"][0]["version"] = "2.0"
    elif mutation == "publication-contract":
        candidate["publication_contract"] = "child-published-before-cleanup"
    elif mutation == "runner-revision":
        candidate["runner_revision"] = "experiment-013-static-q468-calibration-runner-v7"
    elif mutation == "schema-v1":
        candidate["schema_version"] = 1
    elif mutation == "source-commit":
        candidate["source_commit"] = "0" * 40
    else:
        candidate["status"] = "captured_under_authenticated_runtime"

    with pytest.raises(launcher.SealedLaunchError):
        launcher._validate_capture_provenance_candidate(
            launcher._canonical_json_bytes(candidate),
            bindings=fixture["bindings"],
            identity_input_file_sha256=_sha256(identity_bytes),
            runtime_manifest=fixture["runtime_manifest"],
            source_manifest=fixture["source_manifest"],
        )


def test_capture_receipt_is_published_only_after_postconditions_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    temporary_roots: list[Path] = []
    original_publish = launcher._atomic_publish_capture_receipt
    original_verify_hub = launcher._verify_capture_hub_cache_identity
    hub_reauthentications: list[Path] = []

    def verify_hub_wrapper(*args: Any, **kwargs: Any) -> Path:
        result = original_verify_hub(*args, **kwargs)
        hub_reauthentications.append(result)
        return result

    def run_wrapper(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["env"]["HF_HUB_CACHE"] == str(fixture["cache_root"] / "hub")
        assert kwargs["env"]["HUGGINGFACE_HUB_CACHE"] == str(
            fixture["cache_root"] / "hub"
        )
        assert kwargs["env"]["HF_HOME"] == str(Path(command[15]) / "huggingface")
        assert not fixture["receipt_output"].exists()
        temporary_roots.extend((Path(command[13]), Path(command[15])))
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def publish_wrapper(
        snapshot: launcher.CaptureArtifactSnapshot,
        payload: bytes,
    ) -> None:
        assert temporary_roots and all(not path.exists() for path in temporary_roots)
        assert len(hub_reauthentications) >= 2
        assert fixture["identity_output"].read_bytes() == identity_bytes
        assert not fixture["receipt_output"].exists()
        if os.name == "nt":
            with pytest.raises(OSError):
                fixture["git_executable"].write_bytes(b"mutated during publication\n")
        original_publish(snapshot, payload)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(launcher, "_atomic_publish_capture_receipt", publish_wrapper)
    monkeypatch.setattr(
        launcher,
        "_verify_capture_hub_cache_identity",
        verify_hub_wrapper,
    )

    assert launcher.launch(fixture["capture_host_arguments"]) == 0
    assert fixture["receipt_output"].read_bytes() == candidate
    assert len(hub_reauthentications) == 4
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "capture_provenance_receipt_file_sha256": _sha256(candidate),
        "identity_input_file_sha256": _sha256(identity_bytes),
        "runner_revision": launcher.RUNNER_REVISION,
        "status": "captured_calibration_identity_with_launcher_finalization",
    }


def test_capture_fails_if_identity_changes_during_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    changed_identity_bytes = launcher._canonical_json_bytes({"identity": "mutated"})
    candidate = _capture_candidate(fixture, identity_bytes)
    original_publish = launcher._atomic_publish_capture_receipt

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def publish_and_mutate_identity(
        snapshot: launcher.CaptureArtifactSnapshot,
        payload: bytes,
    ) -> None:
        original_publish(snapshot, payload)
        fixture["identity_output"].write_bytes(changed_identity_bytes)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(
        launcher,
        "_atomic_publish_capture_receipt",
        publish_and_mutate_identity,
    )

    with pytest.raises(
        launcher.SealedLaunchError,
        match="identity changed after receipt publication",
    ):
        launcher.launch(fixture["capture_host_arguments"])

    assert fixture["receipt_output"].read_bytes() == candidate
    assert fixture["identity_output"].read_bytes() == changed_identity_bytes
    assert capsys.readouterr().out == ""


def test_stage_a_receipt_is_host_finalized_after_cleanup_and_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _capture_fixture(tmp_path, stage_a=True)
    identity_bytes = launcher._canonical_json_bytes({"identity": "stage-a-fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    temporary_roots: list[Path] = []
    stable_receipt_reads = 0
    stable_read = launcher._stable_file_bytes

    def run_wrapper(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["stdout"] == subprocess.PIPE
        temporary_roots.extend((Path(command[13]), Path(command[15])))
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def stable_read_wrapper(path: Path, *, context: str) -> bytes:
        nonlocal stable_receipt_reads
        if path == fixture["receipt_output"]:
            assert temporary_roots and all(not item.exists() for item in temporary_roots)
            stable_receipt_reads += 1
        return stable_read(path, context=context)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(launcher, "_stable_file_bytes", stable_read_wrapper)

    assert launcher.launch(fixture["capture_host_arguments"]) == 0
    assert fixture["receipt_output"].read_bytes() == candidate
    assert stable_receipt_reads == 1
    assert json.loads(capsys.readouterr().out)["status"] == (
        "captured_stage_a_identity_with_launcher_finalization"
    )


def test_capture_child_failure_or_postcondition_failure_never_publishes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)

    def failed_child(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 37, stdout=candidate)

    monkeypatch.setattr(launcher.subprocess, "run", failed_child)
    assert launcher.launch(fixture["capture_host_arguments"]) == 37
    assert fixture["identity_output"].is_file()
    assert not fixture["receipt_output"].exists()

    fixture["identity_output"].unlink()

    def residue_child(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        _write(Path(command[15]) / "residue.txt", b"must fail closed\n")
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    monkeypatch.setattr(launcher.subprocess, "run", residue_child)
    with pytest.raises(launcher.SealedLaunchError, match="postcondition failed"):
        launcher.launch(fixture["capture_host_arguments"])
    assert fixture["identity_output"].is_file()
    assert not fixture["receipt_output"].exists()


def test_capture_cleanup_failure_never_publishes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    temporary_roots: list[Path] = []
    cleanup = launcher._cleanup_owned_temporary_directory

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        temporary_roots.extend((Path(command[13]), Path(command[15])))
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def fail_scratch_cleanup(path: Path, **kwargs: Any) -> None:
        if kwargs["context"] == "sealed scratch directory":
            raise launcher.SealedLaunchError("injected capture scratch cleanup failure")
        cleanup(path, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(launcher, "_cleanup_owned_temporary_directory", fail_scratch_cleanup)

    with pytest.raises(launcher.SealedLaunchError, match="scratch cleanup failure"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["receipt_output"].exists()
    pycache, scratch = temporary_roots
    assert not pycache.exists()
    assert scratch.exists()
    cleanup(
        scratch,
        expected_identity=launcher._temporary_directory_identity(
            scratch,
            context="sealed scratch directory",
        ),
        context="sealed scratch directory",
    )


def test_capture_invalid_candidate_is_rejected_after_cleanup_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = json.loads(_capture_candidate(fixture, identity_bytes))
    candidate["publication_contract"] = "child-finalized-before-cleanup"
    stale_candidate = launcher._canonical_json_bytes(candidate)
    temporary_roots: list[Path] = []

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        temporary_roots.extend((Path(command[13]), Path(command[15])))
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=stale_candidate)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    with pytest.raises(launcher.SealedLaunchError, match="identity drifted"):
        launcher.launch(fixture["capture_host_arguments"])
    assert temporary_roots and all(not path.exists() for path in temporary_roots)
    assert fixture["identity_output"].is_file()
    assert not fixture["receipt_output"].exists()


def test_capture_identity_mutation_during_candidate_validation_prevents_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    validate = launcher._validate_capture_provenance_candidate

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def mutate_identity_during_validation(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = validate(*args, **kwargs)
        fixture["identity_output"].write_bytes(
            launcher._canonical_json_bytes({"identity": "mutated"})
        )
        return result

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(
        launcher,
        "_validate_capture_provenance_candidate",
        mutate_identity_during_validation,
    )

    with pytest.raises(launcher.SealedLaunchError, match="changed before receipt publication"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["receipt_output"].exists()


def test_capture_receipt_collision_during_validation_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    validate = launcher._validate_capture_provenance_candidate
    competing_bytes = b"competing receipt bytes\n"

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def inject_competing_receipt(*args: Any, **kwargs: Any) -> dict[str, object]:
        result = validate(*args, **kwargs)
        fixture["receipt_output"].write_bytes(competing_bytes)
        return result

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(
        launcher,
        "_validate_capture_provenance_candidate",
        inject_competing_receipt,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launcher.launch(fixture["capture_host_arguments"])
    assert fixture["receipt_output"].read_bytes() == competing_bytes


def test_capture_success_without_identity_output_leaves_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=candidate),
    )

    with pytest.raises(launcher.SealedLaunchError, match="was not published"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["receipt_output"].exists()


def test_capture_rejects_output_collision_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    arguments = list(fixture["capture_host_arguments"])
    output_index = arguments.index("--output") + 1
    arguments[output_index] = str(fixture["receipt_output"])
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="output paths must differ"):
        launcher.launch(arguments)


def test_failed_child_return_code_survives_residue_and_owned_roots_are_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _sealed_fixture(tmp_path)
    temporary_roots: list[Path] = []

    def run_wrapper(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        pycache = Path(command[13])
        scratch = Path(command[15])
        temporary_roots.extend((pycache, scratch))
        _write(pycache / "late.pyc", b"residue")
        _write(scratch / "tokenizer" / "temporary.json", b"residue")
        return subprocess.CompletedProcess(command, 37)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    assert launcher.launch(fixture["host_arguments"]) == 37

    assert temporary_roots and all(not path.exists() for path in temporary_roots)
    diagnostic = capsys.readouterr().err
    assert "sealed launcher secondary failure" in diagnostic
    assert "pycache postcondition" in diagnostic
    assert "scratch containment postcondition" in diagnostic
    assert "preserving child return code 37" in diagnostic


def test_successful_child_residue_fails_closed_after_owned_roots_are_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    temporary_roots: list[Path] = []

    def run_wrapper(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        pycache = Path(command[13])
        scratch = Path(command[15])
        temporary_roots.extend((pycache, scratch))
        _write(pycache / "late.pyc", b"residue")
        _write(scratch / "temporary.txt", b"residue")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    with pytest.raises(
        launcher.SealedLaunchError, match="sealed child postcondition failed"
    ) as caught:
        launcher.launch(fixture["host_arguments"])

    assert "pycache postcondition" in str(caught.value)
    assert "scratch containment postcondition" in str(caught.value)
    assert temporary_roots and all(not path.exists() for path in temporary_roots)


def test_subprocess_exception_remains_primary_while_owned_roots_are_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    temporary_roots: list[Path] = []

    def run_wrapper(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        pycache = Path(command[13])
        scratch = Path(command[15])
        temporary_roots.extend((pycache, scratch))
        _write(scratch / "temporary.txt", b"residue")
        raise OSError("primary subprocess failure")

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    with pytest.raises(OSError, match="primary subprocess failure"):
        launcher.launch(fixture["host_arguments"])

    assert temporary_roots and all(not path.exists() for path in temporary_roots)


def test_cleanup_failure_is_secondary_note_on_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    created: list[Path] = []
    original_cleanup = launcher._cleanup_owned_temporary_directory

    def mkdtemp(*, prefix: str) -> str:
        path = tmp_path / f"{prefix}{len(created)}"
        path.mkdir()
        created.append(path)
        return str(path)

    def cleanup_wrapper(path: Path, **kwargs: Any) -> None:
        if kwargs["context"] == "sealed scratch directory":
            raise launcher.SealedLaunchError("simulated scratch cleanup failure")
        original_cleanup(path, **kwargs)

    def run_wrapper(_command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise OSError("primary subprocess failure")

    monkeypatch.setattr(launcher.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(launcher, "_cleanup_owned_temporary_directory", cleanup_wrapper)
    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    with pytest.raises(OSError, match="primary subprocess failure") as caught:
        launcher.launch(fixture["host_arguments"])

    assert any("simulated scratch cleanup failure" in note for note in caught.value.__notes__)
    scratch = next(path for path in created if "scratch" in path.name)
    assert scratch.exists()
    assert all(not path.exists() for path in created if path != scratch)
    original_cleanup(
        scratch,
        expected_identity=launcher._temporary_directory_identity(
            scratch,
            context="sealed scratch directory",
        ),
        context="sealed scratch directory",
    )


def test_cleanup_refuses_replaced_temporary_root_identity(tmp_path: Path) -> None:
    root = tmp_path / "owned-temporary-root"
    root.mkdir()
    (root / "replacement-sentinel.txt").write_bytes(b"replacement must survive\n")
    observed = launcher._temporary_directory_identity(root, context="temporary root")
    replaced_identity = (observed[0], observed[1] + 1, observed[2])

    with pytest.raises(launcher.SealedLaunchError, match="identity changed before cleanup"):
        launcher._cleanup_owned_temporary_directory(
            root,
            expected_identity=replaced_identity,
            context="temporary root",
        )

    assert (root / "replacement-sentinel.txt").read_bytes() == b"replacement must survive\n"


def test_cleanup_refuses_link_entry_and_preserves_external_target(tmp_path: Path) -> None:
    root = tmp_path / "owned-temporary-root"
    root.mkdir()
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_bytes(b"outside must survive\n")
    link = root / "redirect"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")
    identity = launcher._temporary_directory_identity(root, context="temporary root")

    with pytest.raises(launcher.SealedLaunchError, match="link or reparse point"):
        launcher._cleanup_owned_temporary_directory(
            root,
            expected_identity=identity,
            context="temporary root",
        )

    assert link.is_symlink()
    assert outside.read_bytes() == b"outside must survive\n"


def test_partial_temporary_root_creation_preserves_error_and_cleans_first_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    created: list[Path] = []

    def mkdtemp(*, prefix: str) -> str:
        if created:
            raise OSError("second temporary-root creation failed")
        path = tmp_path / f"{prefix}first"
        path.mkdir()
        created.append(path)
        return str(path)

    monkeypatch.setattr(launcher.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(OSError, match="second temporary-root creation failed"):
        launcher.launch(fixture["host_arguments"])

    assert len(created) == 1
    assert not created[0].exists()


def test_launcher_requires_receipt_directory_and_rejects_legacy_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )
    missing = list(fixture["host_arguments"])
    option_index = missing.index("--ruler-receipt-dir")
    del missing[option_index : option_index + 2]

    with pytest.raises(launcher.SealedLaunchError, match="--ruler-receipt-dir"):
        launcher.launch(missing)

    legacy = list(fixture["host_arguments"])
    legacy[legacy.index("--ruler-receipt-dir")] = "--ruler-root"
    with pytest.raises(launcher.SealedLaunchError, match="legacy runner option is forbidden"):
        launcher.launch(legacy)


def test_embedded_bootstrap_keeps_cleanup_failure_secondary_to_child_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = ast.parse(launcher.SEALED_BOOTSTRAP)
    handler_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_surface_postcondition_failures"
    )
    handler_module = ast.fix_missing_locations(ast.Module(body=[handler_node], type_ignores=[]))

    def fail(message: str) -> None:
        raise RuntimeError(message)

    namespace: dict[str, Any] = {"_fail": fail, "_s": sys}
    exec(compile(handler_module, "<bootstrap-postcondition-handler>", "exec"), namespace)
    handler = namespace["_surface_postcondition_failures"]
    failures = [("scratch", RuntimeError("scratch residue"))]

    primary = ValueError("primary child failure")
    handler(primary, None, failures)
    assert any("scratch residue" in note for note in primary.__notes__)

    handler(None, 37, failures)
    diagnostic = capsys.readouterr().err
    assert "scratch residue" in diagnostic
    assert "preserving sealed_main return code 37" in diagnostic

    with pytest.raises(RuntimeError, match="scratch residue"):
        handler(None, 0, failures)


def test_sealed_environment_omits_auth_network_and_compute_modifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    forbidden = {
        "HF_TOKEN": "secret",
        "HUGGING_FACE_HUB_TOKEN": "secret",
        "HF_ENDPOINT": "https://attacker.invalid",
        "HTTPS_PROXY": "http://proxy.invalid",
        "NO_PROXY": "*",
        "CUDA_VISIBLE_DEVICES": "7",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_LOGS": "+all",
        "NCCL_DEBUG": "INFO",
        "OMP_NUM_THREADS": "99",
    }
    for name, value in forbidden.items():
        monkeypatch.setenv(name, value)

    environment = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
    )

    assert not set(forbidden).intersection({name.upper() for name in environment})
    assert environment["TEMP"] == environment["TMP"] == str(scratch.resolve(strict=True))
    assert environment["LANG"] == environment["LC_ALL"] == "C"
    assert environment["TZ"] == "UTC"
    assert environment["HOME"] == environment["USERPROFILE"] == str(scratch / "private-home")
    assert environment["HF_HOME"] == str(scratch / "huggingface")
    assert environment["HF_HUB_CACHE"] == str(cache_root)
    assert environment["HUGGINGFACE_HUB_CACHE"] == str(cache_root)
    assert environment["TORCH_HOME"] == str(scratch / "torch")
    assert environment["HF_DATASETS_CACHE"] == str(cache_root / "datasets")
    assert environment["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] == str(
        cache_root / "datasets" / "downloads"
    )
    assert environment["HF_DATASETS_EXTRACTED_DATASETS_PATH"] == str(
        cache_root / "datasets" / "downloads" / "extracted"
    )
    assert {
        "DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_UPDATE_CHECK": "1",
        "HF_HUB_DISABLE_XET": "1",
    }.items() <= environment.items()
    assert not (scratch / "huggingface" / "hub").exists()


def test_external_hub_snapshot_link_does_not_contaminate_owned_scratch(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cache_root = tmp_path / "dataset-cache"
    blob = cache_root / "datasets--google-research-datasets--mbpp" / "blobs" / ("a" * 40)
    snapshot = (
        cache_root
        / "datasets--google-research-datasets--mbpp"
        / "snapshots"
        / ("b" * 40)
    )
    blob.parent.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob.write_bytes(b"dataset card\n")
    link = snapshot / "README.md"
    try:
        link.symlink_to(Path("..") / ".." / "blobs" / blob.name)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    environment = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
    )
    assert environment["HF_HUB_CACHE"] == str(cache_root)
    assert link.is_symlink()
    (scratch / "private-home").mkdir()
    (scratch / "private-home" / "ordinary.cache").write_bytes(b"contained\n")
    scratch_identity = launcher._temporary_directory_identity(
        scratch,
        context="sealed scratch directory",
    )

    launcher._cleanup_owned_temporary_directory(
        scratch,
        expected_identity=scratch_identity,
        context="sealed scratch directory",
    )

    assert not scratch.exists()
    assert link.is_symlink()
    assert link.read_bytes() == b"dataset card\n"


def test_capture_environment_routes_only_hub_cache_to_explicit_cache_root(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    hub_root = launcher._prepare_capture_hub_cache_root(cache_root)

    environment = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
        capture_profile=True,
    )

    assert environment["HF_HUB_CACHE"] == str(cache_root / "hub")
    assert environment["HUGGINGFACE_HUB_CACHE"] == str(cache_root / "hub")
    assert environment["HF_HOME"] == str(scratch / "huggingface")
    assert environment["HF_ASSETS_CACHE"] == str(scratch / "huggingface" / "assets")
    assert environment["HF_XET_CACHE"] == str(scratch / "huggingface" / "xet")
    assert environment["HF_MODULES_CACHE"] == str(scratch / "huggingface" / "modules")
    assert environment["HF_TOKEN_PATH"] == str(scratch / "huggingface" / "token")
    assert environment["TRANSFORMERS_CACHE"] == str(scratch / "transformers")
    assert hub_root == cache_root / "hub"
    assert hub_root.is_dir()


def test_capture_hub_cache_root_rejects_a_link(tmp_path: Path) -> None:
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    outside = tmp_path / "outside-hub"
    outside.mkdir()
    try:
        (cache_root / "hub").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(launcher.SealedLaunchError, match="link or reparse"):
        launcher._prepare_capture_hub_cache_root(cache_root)


def test_capture_hub_cache_root_rejects_a_creation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    outside = tmp_path / "outside-hub"
    outside.mkdir()
    candidate = cache_root / "hub"
    original_mkdir = launcher.os.mkdir

    def racing_mkdir(path: object, mode: int = 0o777, **kwargs: Any) -> None:
        if Path(path) == candidate:
            try:
                candidate.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"directory symlinks are unavailable: {error}")
            raise FileExistsError(str(candidate))
        original_mkdir(path, mode, **kwargs)

    monkeypatch.setattr(launcher.os, "mkdir", racing_mkdir)

    with pytest.raises(launcher.SealedLaunchError, match="appeared during creation"):
        launcher._prepare_capture_hub_cache_root(cache_root)


def test_capture_hub_cache_root_rejects_identity_replacement(tmp_path: Path) -> None:
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    hub_root = launcher._prepare_capture_hub_cache_root(cache_root)
    identity = launcher._non_link_directory_identity_chain(
        hub_root,
        context="capture Hub cache root",
    )
    archived = cache_root / "hub-original"
    hub_root.rename(archived)
    hub_root.mkdir()

    with pytest.raises(launcher.SealedLaunchError, match="identity changed"):
        launcher._verify_capture_hub_cache_identity(
            cache_root,
            expected_identity=identity,
        )


def test_capture_hub_cache_replacement_prevents_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        hub_root = fixture["cache_root"] / "hub"
        hub_root.rename(fixture["cache_root"] / "hub-original")
        hub_root.mkdir()
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    with pytest.raises(launcher.SealedLaunchError, match="capture Hub cache root"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["receipt_output"].exists()


def test_capture_hub_cache_replacement_immediately_before_receipt_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(tmp_path)
    identity_bytes = launcher._canonical_json_bytes({"identity": "fixture"})
    candidate = _capture_candidate(fixture, identity_bytes)
    validate = launcher._validate_capture_provenance_candidate
    replaced = False

    def run_wrapper(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        fixture["identity_output"].write_bytes(identity_bytes)
        return subprocess.CompletedProcess(command, 0, stdout=candidate)

    def validate_and_replace_hub(*args: Any, **kwargs: Any) -> None:
        nonlocal replaced
        validate(*args, **kwargs)
        if not replaced:
            hub_root = fixture["cache_root"] / "hub"
            hub_root.rename(fixture["cache_root"] / "hub-original")
            hub_root.mkdir()
            replaced = True

    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)
    monkeypatch.setattr(
        launcher,
        "_validate_capture_provenance_candidate",
        validate_and_replace_hub,
    )

    with pytest.raises(launcher.SealedLaunchError, match="capture Hub cache root"):
        launcher.launch(fixture["capture_host_arguments"])
    assert not fixture["receipt_output"].exists()


def test_embedded_bootstrap_binds_capture_hub_root_before_and_after_runner() -> None:
    assert (
        '_hf_hub_identity = _directory_chain(_hf_hub_raw, "capture Hub cache root")'
        in launcher.SEALED_BOOTSTRAP
    )
    assert "_repeated_hub_identity != _hf_hub_identity" in launcher.SEALED_BOOTSTRAP
    assert "_hf_hub_cache = _cache_root" in launcher.SEALED_BOOTSTRAP
    assert (
        "private-scratch-plus-explicit-dataset-and-phase-hub-roots-v3"
        in launcher.SEALED_BOOTSTRAP
    )


def test_private_home_prevents_literal_tilde_runtime_writes(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    cache_root = tmp_path / "dataset-cache"
    scratch.mkdir()
    cache_root.mkdir()
    environment = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
    )
    code = (
        "from pathlib import Path\n"
        "home = Path('~').expanduser()\n"
        "assert home.is_absolute()\n"
        "target = home / '.cache' / 'huggingface' / '.agent_harnesses.json'\n"
        "target.parent.mkdir(parents=True)\n"
        "target.write_text('contained\\n', encoding='utf-8')\n"
    )

    subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        check=True,
        cwd=scratch,
        env=environment,
    )

    assert (scratch / "private-home" / ".cache" / "huggingface" / ".agent_harnesses.json").is_file()
    assert not (scratch / "~").exists()


def test_dataset_cache_root_is_absolute_non_link_and_disjoint(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    nested_cache = runtime / "cache"
    nested_cache.mkdir()
    with pytest.raises(launcher.SealedLaunchError, match="absolute path"):
        launcher._verified_dataset_cache_root(Path("relative-cache"), runtime_roots=(runtime,))
    with pytest.raises(launcher.SealedLaunchError, match="overlaps"):
        launcher._verified_dataset_cache_root(nested_cache, runtime_roots=(runtime,))
    with pytest.raises(launcher.SealedLaunchError, match="overlaps"):
        launcher._verified_dataset_cache_root(tmp_path, runtime_roots=(runtime,))


def test_dataset_cache_root_rejects_redirected_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "target"
    cache = target / "cache"
    cache.mkdir(parents=True)
    redirect = tmp_path / "redirect"
    try:
        redirect.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(launcher.SealedLaunchError, match="link or reparse"):
        launcher._verified_dataset_cache_root(redirect / "cache", runtime_roots=())


def test_embedded_bootstrap_rejects_an_extra_credential_variable(tmp_path: Path) -> None:
    fixture = _sealed_fixture(tmp_path)

    completed = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "credential-pycache",
        extra_environment={"HF_TOKEN": "must-not-cross"},
    )

    assert completed.returncode != 0
    assert b"sealed child environment differs from the private cache contract" in completed.stderr


@pytest.mark.parametrize("stage_a", [False, True])
def test_embedded_capture_profiles_accept_explicit_hub_cache_contract(
    tmp_path: Path,
    stage_a: bool,
) -> None:
    fixture = _capture_fixture(tmp_path, stage_a=stage_a)

    completed = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "capture-environment-pycache",
    )

    assert completed.returncode != 0
    assert b"sealed child environment differs from the private cache contract" not in (
        completed.stderr
    )
    assert b"point-used interpreter path drifted" in completed.stderr


def test_help_does_not_require_the_runner_separator(capsys: pytest.CaptureFixture[str]) -> None:
    assert launcher.launch(["--help"]) == 0
    assert "exact run_static_q468_calibration.py arguments" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option",
    [
        "--capture-provenance-receipt",
        "--expected-capture-provenance-receipt-sha256",
    ],
)
def test_launch_requires_finalized_capture_provenance_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    arguments = list(fixture["host_arguments"])
    option_index = arguments.index(option)
    del arguments[option_index : option_index + 2]
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="omit required sealed inputs"):
        launcher.launch(arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("runner_revision", "experiment-013-static-q468-calibration-runner-v8"),
        ("status", "captured_under_authenticated_runtime"),
        ("source_commit", "b" * 40),
    ],
)
def test_launch_rejects_nonfinal_capture_provenance_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    receipt = json.loads(fixture["capture_provenance_receipt_bytes"])
    receipt[field] = value
    receipt_bytes = launcher._canonical_json_bytes(receipt)
    fixture["capture_provenance_receipt_path"].write_bytes(receipt_bytes)
    arguments = list(fixture["host_arguments"])
    expected_index = arguments.index("--expected-capture-provenance-receipt-sha256") + 1
    arguments[expected_index] = _sha256(receipt_bytes)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="finalized envelope drifted"):
        launcher.launch(arguments)


def test_embedded_bootstrap_requires_exact_finalized_capture_provenance_digest(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    expected_index = (
        fixture["runner_arguments"].index("--expected-capture-provenance-receipt-sha256") + 1
    )
    fixture["runner_arguments"][expected_index] = "0" * 64

    completed = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "provenance-digest-pycache",
    )

    assert completed.returncode != 0
    assert b"capture provenance receipt differs from its explicit SHA-256" in completed.stderr
    assert b"point-used interpreter path drifted" not in completed.stderr


def test_full_launch_requires_both_prior_fisher_smoke_paths_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    arguments = list(fixture["host_arguments"])
    arguments.remove("--fisher-h1-smoke")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="requires both prior Fisher"):
        launcher.launch(arguments)


def test_full_launch_authenticates_prior_fisher_smoke_report_and_marker(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    evidence = {
        "prerequisites": {
            "capture_provenance_receipt_file_sha256": _sha256(
                fixture["capture_provenance_receipt_bytes"]
            ),
            "fisher_h1_smoke_report_file_sha256": None,
        },
        "runner_revision": launcher.RUNNER_REVISION,
        "status": "fisher_h1_smoke_passed",
    }
    report = launcher._canonical_json_bytes(
        {
            "artifact_kind": launcher.RUN_REPORT_KIND,
            "canonical_evidence_sha256": _sha256(launcher._canonical_json_bytes(evidence)),
            "evidence": evidence,
            "schema_version": launcher.RUN_REPORT_SCHEMA,
        }
    )
    report_path = _write(tmp_path / "smoke" / "report.json", report)
    marker_path = _write(
        tmp_path / "smoke" / "FISHER_H1_SMOKE_COMPLETE",
        launcher.FISHER_SMOKE_COMPLETE_BYTES,
    )
    runner_arguments = list(fixture["runner_arguments"])
    runner_arguments.remove("--fisher-h1-smoke")
    runner_arguments.extend(
        [
            "--prior-fisher-h1-smoke-report",
            str(report_path),
            "--prior-fisher-h1-smoke-complete-marker",
            str(marker_path),
        ]
    )

    options = launcher._extract_runner_options(runner_arguments)
    launcher._verify_bound_artifacts(
        options,
        runtime_manifest_path=fixture["runtime_path"],
    )


def test_full_launch_rejects_rehashed_smoke_bound_to_another_capture_receipt(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    evidence = {
        "prerequisites": {
            "capture_provenance_receipt_file_sha256": "0" * 64,
            "fisher_h1_smoke_report_file_sha256": None,
        },
        "runner_revision": launcher.RUNNER_REVISION,
        "status": "fisher_h1_smoke_passed",
    }
    report = launcher._canonical_json_bytes(
        {
            "artifact_kind": launcher.RUN_REPORT_KIND,
            "canonical_evidence_sha256": _sha256(launcher._canonical_json_bytes(evidence)),
            "evidence": evidence,
            "schema_version": launcher.RUN_REPORT_SCHEMA,
        }
    )
    report_path = _write(tmp_path / "wrong-smoke" / "report.json", report)
    marker_path = _write(
        tmp_path / "wrong-smoke" / "FISHER_H1_SMOKE_COMPLETE",
        launcher.FISHER_SMOKE_COMPLETE_BYTES,
    )
    runner_arguments = list(fixture["runner_arguments"])
    runner_arguments.remove("--fisher-h1-smoke")
    runner_arguments.extend(
        [
            "--prior-fisher-h1-smoke-report",
            str(report_path),
            "--prior-fisher-h1-smoke-complete-marker",
            str(marker_path),
        ]
    )

    with pytest.raises(launcher.SealedLaunchError, match="authentication failed"):
        launcher._verify_bound_artifacts(
            launcher._extract_runner_options(runner_arguments),
            runtime_manifest_path=fixture["runtime_path"],
        )


def test_embedded_bootstrap_authenticates_smoke_prerequisite_before_runner_load() -> None:
    assert "if not _capture_profile:\n    _smoke(_runner_options)" in launcher.SEALED_BOOTSTRAP
    assert "full calibration requires prior Fisher H=1 smoke prerequisites" in (
        launcher.SEALED_BOOTSTRAP
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 3.0), ("dont_write_bytecode", True)],
)
def test_runtime_parser_rejects_equality_compatible_json_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    document = json.loads(fixture["runtime_path"].read_bytes())
    if field == "schema_version":
        document[field] = value
    else:
        document["launch_policy"][field] = value

    with pytest.raises(launcher.SealedLaunchError):
        launcher._parse_runtime_manifest(launcher._canonical_json_bytes(document))


def test_runtime_parser_accepts_exact_dot_only_as_base_sys_path_sentinel(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    document = json.loads(fixture["runtime_path"].read_bytes())

    parsed = launcher._parse_runtime_manifest(launcher._canonical_json_bytes(document))

    assert parsed["base_sys_path"] == ["Lib", "."]

    document["package_roots"][0]["import_path"] = "."
    with pytest.raises(launcher.SealedLaunchError, match="canonical relative path"):
        launcher._parse_runtime_manifest(launcher._canonical_json_bytes(document))


def test_embedded_runtime_parser_applies_the_same_exact_dot_boundary(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)

    accepted = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "accepted-pycache",
    )
    assert accepted.returncode != 0
    assert b"base sys.path is not a canonical relative path" not in accepted.stderr
    assert b"point-used interpreter path drifted" in accepted.stderr

    document = json.loads(fixture["runtime_path"].read_bytes())
    document["package_roots"][0]["import_path"] = "."
    fixture["runtime_path"].write_bytes(launcher._canonical_json_bytes(document))
    rejected_elsewhere = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "elsewhere-pycache",
    )
    assert rejected_elsewhere.returncode != 0
    assert b"import path is not a canonical relative path" in rejected_elsewhere.stderr

    document["package_roots"][0]["import_path"] = "Lib/site-packages"
    document["base_sys_path"] = ["Lib", "./"]
    fixture["runtime_path"].write_bytes(launcher._canonical_json_bytes(document))
    rejected_unsafe = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "unsafe-pycache",
    )
    assert rejected_unsafe.returncode != 0
    assert b"base sys.path is not a canonical relative path" in rejected_unsafe.stderr


@pytest.mark.parametrize("entry", ["./", "./Lib", "..", "Lib/.."])
def test_runtime_parser_rejects_noncanonical_base_sys_path_dot_forms(
    tmp_path: Path,
    entry: str,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    document = json.loads(fixture["runtime_path"].read_bytes())
    document["base_sys_path"] = ["Lib", entry]

    with pytest.raises(launcher.SealedLaunchError, match="canonical relative path"):
        launcher._parse_runtime_manifest(launcher._canonical_json_bytes(document))


def test_identity_parser_rejects_float_schema_version(tmp_path: Path) -> None:
    fixture = _sealed_fixture(tmp_path)
    identity_path = fixture["runtime_path"].parent / "identity.json"
    document = json.loads(identity_path.read_bytes())
    document["evidence"]["schema_version"] = 4.0
    document["canonical_evidence_sha256"] = _sha256(
        launcher._canonical_json_bytes(document["evidence"])
    )

    with pytest.raises(launcher.SealedLaunchError, match="schema"):
        launcher._parse_identity(launcher._canonical_json_bytes(document))


def test_launcher_and_embedded_bootstrap_require_identity_v5() -> None:
    assert launcher.IDENTITY_SCHEMA == 5
    assert 'evidence.get("schema_version") != 5' in launcher.SEALED_BOOTSTRAP


def test_embedded_bootstrap_repeats_exact_json_type_checks() -> None:
    assert 'type(evidence.get("schema_version")) is not int' in launcher.SEALED_BOOTSTRAP
    assert '_typed(root["launch_policy"], _policy' in launcher.SEALED_BOOTSTRAP


def test_identity_binding_mismatch_stops_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    fixture["model_path"].write_bytes(b"mutated model manifest\n")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="identity binding mismatch"):
        launcher.launch(fixture["host_arguments"])


def test_expected_parquet_digest_mismatch_stops_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    arguments = list(fixture["host_arguments"])
    index = arguments.index("--expected-parquet-materialization-manifest-sha256")
    arguments[index + 1] = "0" * 64
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="runner digest binding mismatch"):
        launcher.launch(arguments)


def test_complete_tree_mutation_stops_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    package_file = fixture["packages"] / "Lib" / "site-packages" / "demo" / "__init__.py"
    package_file.write_bytes(b"VALUE = 2\n")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )

    with pytest.raises(launcher.SealedLaunchError, match="runtime tree packages differs"):
        launcher.launch(fixture["host_arguments"])


def test_nonempty_pycache_is_rejected(tmp_path: Path) -> None:
    pycache = tmp_path / "pycache"
    _write(pycache / "unexpected.pyc", b"not allowed")

    with pytest.raises(launcher.SealedLaunchError, match="pycache prefix is not empty"):
        launcher._verify_empty_pycache(pycache)


def test_bootstrap_rejects_flag_drift() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "-c", launcher.SEALED_STDIN_LOADER],
        check=False,
        capture_output=True,
        input=launcher.SEALED_BOOTSTRAP_BYTES,
    )

    assert completed.returncode != 0
    assert b"sealed bootstrap startup flags drifted" in completed.stderr


def test_authenticated_stdin_loader_crosses_a_real_child_boundary(
    tmp_path: Path,
) -> None:
    pycache = tmp_path / "empty-pycache"
    pycache.mkdir()
    payload = b"import sys\nprint('authenticated-child:' + sys.argv[1])\nraise SystemExit(23)\n"
    loader = launcher._authenticated_stdin_loader(payload)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={pycache}",
            "-X",
            "utf8",
            "-c",
            loader,
            "ok",
        ],
        check=False,
        capture_output=True,
        input=payload,
    )

    assert completed.returncode == 23
    assert completed.stdout.splitlines() == [b"authenticated-child:ok"]
    assert completed.stderr == b""
    assert not any(pycache.iterdir())


def test_authenticated_stdin_loader_rejects_modified_payload() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", launcher.SEALED_STDIN_LOADER],
        check=False,
        capture_output=True,
        input=launcher.SEALED_BOOTSTRAP_BYTES + b"\n",
    )

    assert completed.returncode != 0
    assert b"sealed bootstrap stdin authentication failed" in completed.stderr


@pytest.mark.parametrize("module_name", ["_virtualenv", "six"])
def test_bootstrap_rejects_preloaded_sensitive_module(
    tmp_path: Path,
    module_name: str,
) -> None:
    pycache = tmp_path / "empty-pycache"
    pycache.mkdir()
    prefix = f"import sys; sys.modules[{module_name!r}] = object()\n"
    payload = (prefix + launcher.SEALED_BOOTSTRAP).encode("utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={pycache}",
            "-X",
            "utf8",
            "-c",
            launcher._authenticated_stdin_loader(payload),
        ],
        check=False,
        capture_output=True,
        input=payload,
    )

    assert completed.returncode != 0
    assert b"sealed bootstrap found a preloaded sensitive module" in completed.stderr
    assert not any(pycache.iterdir())


@pytest.mark.parametrize("arguments", [[], ["--", "--", "runner"]])
def test_launcher_requires_one_separator(arguments: list[str]) -> None:
    with pytest.raises(launcher.SealedLaunchError, match="exactly one -- separator"):
        launcher._split_host_and_runner_args(arguments)
