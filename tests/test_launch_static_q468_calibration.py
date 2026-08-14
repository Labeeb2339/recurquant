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
    source_payload = {
        "git_executable": {
            "sha256": git_record["sha256"],
            "size_bytes": git_record["size_bytes"],
        },
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
    bindings = {
        "calibration_runtime_manifest_file_sha256": _sha256(runtime_path.read_bytes()),
        "model_file_manifest_file_sha256": _sha256(model_path.read_bytes()),
        "parquet_materialization_manifest_file_sha256": _sha256(parquet_path.read_bytes()),
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
        "canonical_evidence_sha256": _sha256(launcher._canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    identity_path = _write(
        artifacts / "identity.json",
        launcher._canonical_json_bytes(identity),
    )
    runner_arguments = [
        "--fisher-h1-smoke",
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
        "git_executable": git_executable,
        "host_arguments": host_arguments,
        "model_path": model_path,
        "packages": packages,
        "parquet_path": parquet_path,
        "repository": repository,
        "runner_arguments": runner_arguments,
        "runtime_path": runtime_path,
    }


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
    environment = launcher._sealed_environment(scratch_directory=scratch)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        cwd=fixture["base"],
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
        input: bytes,
    ) -> subprocess.CompletedProcess[str]:
        events.append("run")
        assert check is False
        commands.append((command, cwd, env, input))
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
    assert cwd == fixture["base"].resolve(strict=True)
    assert all(not key.upper().startswith("PYTHON") for key in environment)
    assert "VIRTUAL_ENV" not in environment
    assert "VIRTUAL_ENV_PROMPT" not in environment
    assert "PATH" not in environment
    assert environment["TEMP"] == environment["TMP"] == command[15]
    assert stdin_payload == launcher.SEALED_BOOTSTRAP_BYTES


def test_sealed_environment_omits_auth_network_and_compute_modifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
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

    environment = launcher._sealed_environment(scratch_directory=scratch)

    assert not set(forbidden).intersection({name.upper() for name in environment})
    assert environment["TEMP"] == environment["TMP"] == str(scratch.resolve(strict=True))
    assert environment["LANG"] == environment["LC_ALL"] == "C"
    assert environment["TZ"] == "UTC"


def test_embedded_bootstrap_rejects_an_extra_credential_variable(tmp_path: Path) -> None:
    fixture = _sealed_fixture(tmp_path)

    completed = _run_embedded_manifest_boundary(
        fixture,
        pycache=tmp_path / "credential-pycache",
        extra_environment={"HF_TOKEN": "must-not-cross"},
    )

    assert completed.returncode != 0
    assert b"sealed child environment differs from the minimal contract" in completed.stderr


def test_help_does_not_require_the_runner_separator(capsys: pytest.CaptureFixture[str]) -> None:
    assert launcher.launch(["--help"]) == 0
    assert "exact run_static_q468_calibration.py arguments" in capsys.readouterr().out


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
    evidence = {"status": "fisher_h1_smoke_passed"}
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


def test_embedded_bootstrap_authenticates_smoke_prerequisite_before_runner_load() -> None:
    assert "_smoke(_runner_options)" in launcher.SEALED_BOOTSTRAP
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


def test_bootstrap_rejects_preloaded_sensitive_module(tmp_path: Path) -> None:
    pycache = tmp_path / "empty-pycache"
    pycache.mkdir()
    prefix = "import sys; sys.modules['_virtualenv'] = object()\n"
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
