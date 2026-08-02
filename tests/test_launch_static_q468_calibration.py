from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_static_q468_calibration.py"
)
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
    artifacts = tmp_path / "artifacts"

    interpreter = _write(base / "python.exe", b"staged interpreter\n")
    _write(base / "Lib" / "marker.py", b"BASE = True\n")
    import_root = packages / "Lib" / "site-packages"
    _write(import_root / "demo" / "__init__.py", b"VALUE = 1\n")
    metadata = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n"
    _write(import_root / "demo-1.0.dist-info" / "METADATA", metadata)
    record = (
        b"demo-1.0.dist-info/METADATA,,\n"
        b"demo-1.0.dist-info/RECORD,,\n"
        b"demo/__init__.py,,\n"
    )
    _write(import_root / "demo-1.0.dist-info" / "RECORD", record)

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
        "base_sys_path": ["Lib"],
        "distributions": list(
            launcher._distribution_inventory(package_roots, import_paths)
        ),
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
        "package_roots": [
            {"import_path": "Lib/site-packages", "name": "packages"}
        ],
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
            b"package_import_paths, interpreter_path, pycache_prefix):\n"
            b"    return 0\n"
        ),
    )
    source_payload = {
        "object_format": "sha1",
        "paths": [
            {
                "git_blob_oid": "b" * 40,
                "index_blob_oid": "b" * 40,
                "mode": "100644",
                "path": launcher.RUNNER_SOURCE_PATH,
                "raw_sha256": _sha256(runner_path.read_bytes()),
                "worktree_blob_oid": "b" * 40,
            }
        ],
        "profile": "experiment-013-static-q468-frozen-source-v1",
        "repository_binding": {},
        "schema": "recurquant.experiment013.source-manifest.v1",
        "source_commit": "a" * 40,
    }
    source = {
        **source_payload,
        "canonical_manifest_sha256": _sha256(
            launcher._pretty_json_bytes(source_payload)
        ),
    }
    source_path = _write(
        artifacts / "source.json",
        launcher._pretty_json_bytes(source),
    )
    model_path = _write(artifacts / "model.json", b'{"model":"fixture"}\n')
    parquet_path = _write(artifacts / "parquet.json", b'{"parquet":"fixture"}\n')
    bindings = {
        "calibration_runtime_manifest_file_sha256": _sha256(runtime_path.read_bytes()),
        "model_file_manifest_file_sha256": _sha256(model_path.read_bytes()),
        "parquet_materialization_manifest_file_sha256": _sha256(
            parquet_path.read_bytes()
        ),
        "repository_source_manifest_file_sha256": _sha256(source_path.read_bytes()),
    }
    evidence = {
        "execution_bindings": bindings,
        "identity_only": True,
        "phase": "calibration",
        "promotion_required": False,
        "schema_version": launcher.IDENTITY_SCHEMA,
        "status": "frozen",
    }
    identity = {
        "canonical_evidence_sha256": _sha256(
            launcher._canonical_json_bytes(evidence)
        ),
        "evidence": evidence,
    }
    identity_path = _write(
        artifacts / "identity.json",
        launcher._canonical_json_bytes(identity),
    )
    runner_arguments = [
        "--frozen-identity",
        str(identity_path),
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
    ]
    host_arguments = [
        "--base-runtime-root",
        str(base),
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
        "host_arguments": host_arguments,
        "model_path": model_path,
        "packages": packages,
        "parquet_path": parquet_path,
        "repository": repository,
        "runner_arguments": runner_arguments,
        "runtime_path": runtime_path,
    }


def test_launch_uses_exact_isolated_command_and_reauthenticates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "untrusted-venv"))
    monkeypatch.setenv("VIRTUAL_ENV_PROMPT", "untrusted")
    events: list[str] = []
    commands: list[tuple[list[str], Path, dict[str, str]]] = []
    verify_bound = launcher._verify_bound_artifacts
    verify_runtime = launcher._verify_runtime
    verify_pycache = launcher._verify_empty_pycache

    def bound_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("bound")
        return verify_bound(*args, **kwargs)

    def runtime_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("runtime")
        return verify_runtime(*args, **kwargs)

    def pycache_wrapper(*args: Any, **kwargs: Any) -> Any:
        events.append("pycache")
        return verify_pycache(*args, **kwargs)

    def run_wrapper(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        events.append("run")
        assert check is False
        commands.append((command, cwd, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher, "_verify_bound_artifacts", bound_wrapper)
    monkeypatch.setattr(launcher, "_verify_runtime", runtime_wrapper)
    monkeypatch.setattr(launcher, "_verify_empty_pycache", pycache_wrapper)
    monkeypatch.setattr(launcher.subprocess, "run", run_wrapper)

    assert launcher.launch(fixture["host_arguments"]) == 0
    assert events == [
        "bound",
        "runtime",
        "pycache",
        "run",
        "pycache",
        "bound",
        "runtime",
        "pycache",
    ]
    assert len(commands) == 1
    command, cwd, environment = commands[0]
    assert command[0] == str((fixture["base"] / "python.exe").resolve(strict=True))
    assert command[1:5] == ["-I", "-S", "-B", "-X"]
    assert command[5].startswith("pycache_prefix=")
    assert command[6:10] == ["-X", "utf8", "-c", launcher.SEALED_BOOTSTRAP]
    assert command[10] == str(fixture["runtime_path"].resolve(strict=True))
    assert command[11] == str(fixture["base"].resolve(strict=True))
    assert json.loads(command[12]) == {
        "packages": str(fixture["packages"].resolve(strict=True))
    }
    assert command[14:] == fixture["runner_arguments"]
    assert not Path(command[13]).exists()
    assert cwd == fixture["base"].resolve(strict=True)
    assert all(not key.upper().startswith("PYTHON") for key in environment)
    assert "VIRTUAL_ENV" not in environment
    assert "VIRTUAL_ENV_PROMPT" not in environment


def test_help_does_not_require_the_runner_separator(capsys: pytest.CaptureFixture[str]) -> None:
    assert launcher.launch(["--help"]) == 0
    assert "exact run_static_q468_calibration.py arguments" in capsys.readouterr().out


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
        [sys.executable, "-S", "-c", launcher.SEALED_BOOTSTRAP],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "sealed bootstrap startup flags drifted" in completed.stderr


def test_bootstrap_rejects_preloaded_sensitive_module(tmp_path: Path) -> None:
    pycache = tmp_path / "empty-pycache"
    pycache.mkdir()
    prefix = "import sys; sys.modules['_virtualenv'] = object()\n"
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
            prefix + launcher.SEALED_BOOTSTRAP,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "sealed bootstrap found a preloaded sensitive module" in completed.stderr
    assert not any(pycache.iterdir())


@pytest.mark.parametrize("arguments", [[], ["--", "--", "runner"]])
def test_launcher_requires_one_separator(arguments: list[str]) -> None:
    with pytest.raises(launcher.SealedLaunchError, match="exactly one -- separator"):
        launcher._split_host_and_runner_args(arguments)
