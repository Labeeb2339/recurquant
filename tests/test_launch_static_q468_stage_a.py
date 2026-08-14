from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "launch_static_q468_stage_a.py"
SPEC = importlib.util.spec_from_file_location("launch_static_q468_stage_a_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _identity(bindings: dict[str, str]) -> bytes:
    evidence = {
        "schema_version": 5,
        "identity_schema": "recurquant.experiment013.identity-frozen.v5",
        "status": "frozen",
        "phase": "stage_a",
        "identity_only": True,
        "promotion_required": False,
        "promotion": {"explicit": True},
        "execution_bindings": bindings,
    }
    return launcher._canonical_json_bytes(
        {
            "canonical_evidence_sha256": _digest(launcher._canonical_json_bytes(evidence)),
            "evidence": evidence,
        }
    )


def _source_manifest(root: Path) -> bytes:
    git_bytes = b"fixture Git executable\n"
    _write(root / "toolchain" / "git.exe", git_bytes)
    paths = []
    for relative in sorted(
        {
            launcher.RUNNER_SOURCE_PATH,
            launcher.LAUNCHER_SOURCE_PATH,
            launcher.CALIBRATION_LAUNCHER_SOURCE_PATH,
        }
    ):
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = relative.encode()
        path.write_bytes(payload)
        oid = "1" * 40
        paths.append(
            {
                "git_blob_oid": oid,
                "index_blob_oid": oid,
                "mode": "100644",
                "path": relative,
                "raw_sha256": _digest(payload),
                "worktree_blob_oid": oid,
            }
        )
    document = {
        "schema": "recurquant.experiment013.source-manifest.v2",
        "profile": "experiment-013-static-q468-frozen-source-v2",
        "object_format": "sha1",
        "source_commit": "2" * 40,
        "git_executable": {
            "sha256": _digest(git_bytes),
            "size_bytes": len(git_bytes),
        },
        "repository_binding": {},
        "paths": paths,
    }
    document["canonical_manifest_sha256"] = _digest(launcher._pretty_json_bytes(document))
    return launcher._pretty_json_bytes(document)


def _runner_arguments(paths: dict[str, Path], digests: dict[str, str]) -> list[str]:
    values = {
        "--frozen-identity": paths["identity"],
        "--stage-a-calibration-binding": paths["binding"],
        "--repository-root": paths["root"],
        "--source-commit": "2" * 40,
        "--identity-commit": "3" * 40,
        "--model-root": paths["root"],
        "--cache-root": paths["root"],
        "--input-bundle-root": paths["root"] / "bundle",
        "--ruler-root": paths["root"],
        "--output-dir": paths["root"] / "out",
        "--runtime-manifest": paths["runtime"],
        "--model-file-manifest": paths["model"],
        "--parquet-materialization-manifest": paths["parquet"],
        "--repository-source-manifest": paths["source"],
        "--expected-runtime-manifest-sha256": digests["runtime"],
        "--expected-model-file-manifest-sha256": digests["model"],
        "--expected-parquet-materialization-manifest-sha256": digests["parquet"],
    }
    result = ["preflight"]
    for option in sorted(values):
        result.extend((option, str(values[option])))
    return result


def _load_calibration_launcher(name: str) -> Any:
    path = ROOT / launcher.CALIBRATION_LAUNCHER_SOURCE_PATH
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _copy_test_runtime(destination: Path) -> Path:
    source = Path(sys.base_prefix).resolve(strict=True)
    destination.mkdir(parents=True)
    interpreter_source = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    interpreter = destination / interpreter_source.name
    shutil.copy2(interpreter_source, interpreter)
    for path in source.iterdir():
        if path.is_file() and path.suffix.casefold() == ".dll":
            shutil.copy2(path, destination / path.name)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name == "__pycache__"
            or name == "site-packages"
            or Path(name).suffix.casefold() in {".pyc", ".pyo", ".pth"}
        }

    shutil.copytree(source / "Lib", destination / "Lib", ignore=ignore)
    shutil.copytree(source / "DLLs", destination / "DLLs", ignore=ignore)
    return interpreter


def _embedded_bootstrap_fixture(tmp_path: Path) -> dict[str, Any]:
    calibration = _load_calibration_launcher("calibration_launcher_stage_a_embedded_test")
    base = tmp_path / "base"
    interpreter = _copy_test_runtime(base)
    package_root = tmp_path / "packages"
    git_executable = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    git_executable, git_record = calibration._authenticated_git_executable(git_executable)
    import_root = package_root / "Lib" / "site-packages"
    _write(import_root / "demo" / "__init__.py", b"VALUE = 1\n")
    _write(
        import_root / "demo-1.0.dist-info" / "METADATA",
        b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n",
    )
    _write(
        import_root / "demo-1.0.dist-info" / "RECORD",
        (b"demo-1.0.dist-info/METADATA,,\ndemo-1.0.dist-info/RECORD,,\ndemo/__init__.py,,\n"),
    )

    probe = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            "-c",
            "import json,sys;print(json.dumps(sys.path))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    base_resolved = base.resolve(strict=True)
    base_sys_path: list[str] = []
    for raw_path in json.loads(probe.stdout):
        path = Path(raw_path).resolve(strict=False)
        if path == base_resolved:
            base_sys_path.append(".")
        else:
            base_sys_path.append(path.relative_to(base_resolved).as_posix())

    package_roots = {"packages": package_root.resolve(strict=True)}
    import_paths = {"packages": "Lib/site-packages"}
    interpreter_record = calibration._stable_file_record(
        interpreter,
        relative=interpreter.name,
        context="Stage-A embedded-test interpreter",
    )
    runtime = {
        "artifact_kind": calibration.RUNTIME_MANIFEST_KIND,
        "base_runtime_root": calibration.BASE_RUNTIME_ROOT_NAME,
        "base_sys_path": base_sys_path,
        "distributions": list(calibration._distribution_inventory(package_roots, import_paths)),
        "git_executable": git_record,
        "interpreter": {
            "relative_path": interpreter.name,
            "root": calibration.BASE_RUNTIME_ROOT_NAME,
            "sha256": interpreter_record["sha256"],
            "size_bytes": interpreter_record["size_bytes"],
        },
        "launch_policy": calibration.SEALED_LAUNCH_POLICY,
        "machine": {
            "architecture": platform.architecture()[0],
            "byteorder": sys.byteorder,
            "machine": platform.machine(),
            "pointer_bits": 8 * struct.calcsize("P"),
            "system": platform.system(),
        },
        "package_roots": [{"import_path": import_paths["packages"], "name": "packages"}],
        "python": {
            "abi_flags": getattr(sys, "abiflags", ""),
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "runtime_trees": [
            {
                "files": list(calibration._tree_files(base, context="embedded-test base")),
                "kind": "base-runtime",
                "name": "base-runtime",
            },
            {
                "files": list(
                    calibration._tree_files(package_root, context="embedded-test packages")
                ),
                "kind": "packages",
                "name": "packages",
            },
        ],
        "schema_version": calibration.RUNTIME_MANIFEST_SCHEMA,
    }
    artifacts = tmp_path / "artifacts"
    runtime_path = _write(artifacts / "runtime.json", calibration._canonical_json_bytes(runtime))

    repository = tmp_path / "repository"
    sentinel = tmp_path / "runner-boundary-reached.txt"
    _write(
        repository / launcher.RUNNER_SOURCE_PATH,
        (
            "from pathlib import Path\n"
            "def sealed_main(argv, *, base_runtime_root, package_roots, "
            "package_import_paths, interpreter_path, pycache_prefix, "
            "git_executable_path):\n"
            "    options = dict(zip(argv[1::2], argv[2::2], strict=True))\n"
            f"    Path({str(sentinel)!r}).write_text('stage-a-boundary\\n', encoding='utf-8')\n"
            "    return 37\n"
        ).encode(),
    )
    _write(repository / launcher.LAUNCHER_SOURCE_PATH, b"# authenticated fixture\n")
    _write(
        repository / launcher.CALIBRATION_LAUNCHER_SOURCE_PATH,
        b"# authenticated fixture\n",
    )
    source_paths = []
    for relative in sorted(
        {
            launcher.RUNNER_SOURCE_PATH,
            launcher.LAUNCHER_SOURCE_PATH,
            launcher.CALIBRATION_LAUNCHER_SOURCE_PATH,
        }
    ):
        path = repository / Path(relative)
        source_paths.append(
            {
                "git_blob_oid": "b" * 40,
                "index_blob_oid": "b" * 40,
                "mode": "100644",
                "path": relative,
                "raw_sha256": _digest(path.read_bytes()),
                "worktree_blob_oid": "b" * 40,
            }
        )
    source_payload = {
        "git_executable": {
            "sha256": git_record["sha256"],
            "size_bytes": git_record["size_bytes"],
        },
        "object_format": "sha1",
        "paths": source_paths,
        "profile": "experiment-013-static-q468-frozen-source-v2",
        "repository_binding": {},
        "schema": "recurquant.experiment013.source-manifest.v2",
        "source_commit": "2" * 40,
    }
    source = {
        **source_payload,
        "canonical_manifest_sha256": _digest(launcher._pretty_json_bytes(source_payload)),
    }
    source_path = _write(artifacts / "source.json", launcher._pretty_json_bytes(source))
    model_path = _write(artifacts / "model.json", b'{"model":"fixture"}\n')
    parquet_path = _write(artifacts / "parquet.json", b'{"parquet":"fixture"}\n')
    binding_path = _write(artifacts / "binding.json", b'{"binding":"fixture"}\n')
    bindings = {
        "calibration_runtime_manifest_file_sha256": _digest(runtime_path.read_bytes()),
        "model_file_manifest_file_sha256": _digest(model_path.read_bytes()),
        "parquet_materialization_manifest_file_sha256": _digest(parquet_path.read_bytes()),
        "repository_source_manifest_file_sha256": _digest(source_path.read_bytes()),
    }
    identity_path = _write(artifacts / "identity.json", _identity(bindings))
    paths = {
        "binding": binding_path,
        "identity": identity_path,
        "model": model_path,
        "parquet": parquet_path,
        "root": repository,
        "runtime": runtime_path,
        "source": source_path,
    }
    runner_arguments = _runner_arguments(
        paths,
        {
            "model": bindings["model_file_manifest_file_sha256"],
            "parquet": bindings["parquet_materialization_manifest_file_sha256"],
            "runtime": bindings["calibration_runtime_manifest_file_sha256"],
        },
    )
    return {
        "base": base,
        "bootstrap": launcher._stage_a_bootstrap(calibration).encode("utf-8"),
        "calibration": calibration,
        "interpreter": interpreter,
        "git_executable": git_executable,
        "identity_path": identity_path,
        "model_path": model_path,
        "package_root": package_root,
        "repository": repository,
        "runner_arguments": runner_arguments,
        "runner_path": repository / launcher.RUNNER_SOURCE_PATH,
        "runtime_path": runtime_path,
        "sentinel": sentinel,
        "source_path": source_path,
    }


def test_identity_parser_accepts_only_promoted_stage_a_v5() -> None:
    bindings = {name: _digest(name) for name in launcher._BOUND_ARTIFACT_OPTIONS}
    assert launcher._parse_identity(_identity(bindings)) == {
        name: bindings[name] for name in sorted(bindings)
    }

    root = launcher._strict_json(_identity(bindings), context="identity")
    root["evidence"]["phase"] = "calibration"
    root["canonical_evidence_sha256"] = _digest(launcher._canonical_json_bytes(root["evidence"]))
    with pytest.raises(launcher.SealedStageALaunchError, match="promoted Stage-A"):
        launcher._parse_identity(launcher._canonical_json_bytes(root))


def test_runner_cli_forbids_method_seed_threshold_and_policy_options(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in ("identity", "binding", "runtime", "model", "parquet", "source")
    }
    paths["root"] = tmp_path
    digests = {name: _digest(name) for name in ("runtime", "model", "parquet")}
    arguments = _runner_arguments(paths, digests)
    assert set(launcher._extract_options(arguments)) == launcher._REQUIRED_OPTIONS
    for option in ("--methods", "--seed", "--threshold", "--q48-policy", "--uniform-policy"):
        with pytest.raises(launcher.SealedStageALaunchError, match="frozen CLI"):
            launcher._extract_options([*arguments, option, "bad"])


def test_source_bootstrap_and_verifier_detect_byte_tamper(tmp_path: Path) -> None:
    data = _source_manifest(tmp_path)
    decoded = launcher._parse_source(data)
    runner = launcher._verify_source(decoded, tmp_path)
    assert runner == (tmp_path / Path(launcher.RUNNER_SOURCE_PATH)).resolve()
    runner.write_bytes(b"tampered")
    with pytest.raises(launcher.SealedStageALaunchError, match="source bytes drifted"):
        launcher._verify_source(decoded, tmp_path)


def test_bound_artifacts_are_checked_before_runner_load(tmp_path: Path) -> None:
    runtime = b"runtime"
    model = b"model"
    parquet = b"parquet"
    source = _source_manifest(tmp_path)
    files = {
        "runtime": runtime,
        "model": model,
        "parquet": parquet,
        "source": source,
        "binding": b"binding-v3",
    }
    paths = {name: tmp_path / f"{name}.json" for name in files}
    paths["root"] = tmp_path
    for name, payload in files.items():
        paths[name].write_bytes(payload)
    bindings = {
        "calibration_runtime_manifest_file_sha256": _digest(runtime),
        "model_file_manifest_file_sha256": _digest(model),
        "parquet_materialization_manifest_file_sha256": _digest(parquet),
        "repository_source_manifest_file_sha256": _digest(source),
    }
    paths["identity"] = tmp_path / "identity.json"
    paths["identity"].write_bytes(_identity(bindings))
    arguments = _runner_arguments(
        paths,
        {"runtime": _digest(runtime), "model": _digest(model), "parquet": _digest(parquet)},
    )
    options = launcher._extract_options(arguments)
    parsed, _source, runner = launcher._verify_bound_inputs(
        options, runtime_manifest_path=paths["runtime"]
    )
    assert parsed == bindings
    assert runner.name == "screen_static_q468_stage_a.py"

    paths["model"].write_bytes(b"tampered")
    with pytest.raises(launcher.SealedStageALaunchError, match="identity binding mismatch"):
        launcher._verify_bound_inputs(options, runtime_manifest_path=paths["runtime"])


def test_bootstrap_derivation_switches_schema_phase_runner_and_keeps_isolation() -> None:
    calibration_path = ROOT / launcher.CALIBRATION_LAUNCHER_SOURCE_PATH
    spec = importlib.util.spec_from_file_location(
        "calibration_launcher_stage_a_test", calibration_path
    )
    assert spec is not None and spec.loader is not None
    calibration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = calibration
    try:
        spec.loader.exec_module(calibration)
        bootstrap = launcher._stage_a_bootstrap(calibration)
    finally:
        sys.modules.pop(spec.name, None)
    assert 'evidence.get("schema_version") != 5' in bootstrap
    assert 'evidence.get("phase") != "stage_a"' in bootstrap
    assert launcher.RUNNER_SOURCE_PATH in bootstrap
    assert "scripts/run_static_q468_calibration.py" not in bootstrap
    assert "_stage_a_options" in bootstrap
    assert "--stage-a-calibration-binding" in bootstrap
    assert "--fisher-h1-smoke" not in bootstrap
    assert "--prior-fisher-h1-smoke-report" not in bootstrap
    assert "full calibration" not in bootstrap
    assert 'promotion.get("explicit") is not True' in bootstrap
    assert "isolated != 1" in bootstrap
    assert "no_site != 1" in bootstrap
    assert "dont_write_bytecode != 1" in bootstrap


def test_authenticated_stdin_loader_rejects_modified_bootstrap() -> None:
    payload = b"raise SystemExit(23)"
    loader = launcher._authenticated_stdin_loader(payload)
    accepted = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", loader],
        check=False,
        input=payload,
        capture_output=True,
    )
    assert accepted.returncode == 23

    rejected = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", loader],
        check=False,
        input=payload + b"\n",
        capture_output=True,
    )
    assert rejected.returncode != 23
    assert b"sealed Stage-A bootstrap stdin authentication failed" in rejected.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="sealed runtime fixture is Windows-only")
def test_embedded_bootstrap_reaches_runner_boundary_and_rejects_bound_tamper(
    tmp_path: Path,
) -> None:
    fixture = _embedded_bootstrap_fixture(tmp_path)

    def run_bootstrap(
        pycache: Path,
        *,
        offline: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        pycache.mkdir()
        scratch = pycache.parent / f"{pycache.name}-scratch"
        scratch.mkdir()
        command = launcher._sealed_argv(
            interpreter=fixture["interpreter"],
            bootstrap=fixture["bootstrap"],
            runtime_manifest=fixture["runtime_path"],
            base_runtime_root=fixture["base"],
            package_roots={"packages": fixture["package_root"]},
            git_executable=fixture["git_executable"],
            pycache_prefix=pycache,
            scratch_directory=scratch,
            runner_arguments=fixture["runner_arguments"],
        )
        return subprocess.run(
            command,
            check=False,
            cwd=fixture["base"],
            env=launcher._sealed_environment(scratch_directory=scratch, offline=offline),
            capture_output=True,
            input=fixture["bootstrap"],
        )

    completed = run_bootstrap(tmp_path / "pycache-valid")
    assert completed.returncode == 37, completed.stderr
    assert fixture["sentinel"].read_text(encoding="utf-8") == "stage-a-boundary\n"

    fixture["sentinel"].unlink()
    missing_offline = run_bootstrap(tmp_path / "pycache-missing-offline", offline=False)
    assert missing_offline.returncode != 37
    assert b"mode-specific minimal contract" in missing_offline.stderr
    assert not fixture["sentinel"].exists()

    fixture["model_path"].write_bytes(b"tampered\n")
    rejected = run_bootstrap(tmp_path / "pycache-tampered")
    assert rejected.returncode != 37
    assert b"identity binding mismatch: --model-file-manifest" in rejected.stderr
    assert not fixture["sentinel"].exists()


def test_mode_specific_child_environments_strip_credentials_and_proxies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in (
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.setenv(name, "must-not-cross-child-boundary")

    networked = launcher._sealed_environment(scratch_directory=scratch, offline=False)
    offline = launcher._sealed_environment(scratch_directory=scratch, offline=True)

    assert not {"HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} & set(networked)
    assert {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items() <= offline.items()
    forbidden = {"GITHUB_TOKEN", "HF_TOKEN", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"}
    assert not forbidden & set(networked)
    assert not forbidden & set(offline)


@pytest.mark.skipif(sys.platform != "win32", reason="sealed runtime fixture is Windows-only")
def test_offline_child_fatally_rejects_socket_connect_before_runner_side_effect(
    tmp_path: Path,
) -> None:
    fixture = _embedded_bootstrap_fixture(tmp_path)
    fixture["runner_path"].write_bytes(
        (
            "import socket\n"
            "from pathlib import Path\n"
            "def sealed_main(argv, *, base_runtime_root, package_roots, "
            "package_import_paths, interpreter_path, pycache_prefix, "
            "git_executable_path):\n"
            "    connection = socket.socket()\n"
            "    try:\n"
            "        connection.connect(('127.0.0.1', 9))\n"
            "    finally:\n"
            "        connection.close()\n"
            f"    Path({str(fixture['sentinel'])!r}).write_text('network-ran\\n')\n"
            "    return 37\n"
        ).encode()
    )
    source = launcher._strict_json(fixture["source_path"].read_bytes(), context="source")
    for entry in source["paths"]:
        if entry["path"] == launcher.RUNNER_SOURCE_PATH:
            entry["raw_sha256"] = _digest(fixture["runner_path"].read_bytes())
    source_payload = dict(source)
    source_payload.pop("canonical_manifest_sha256")
    source["canonical_manifest_sha256"] = _digest(launcher._pretty_json_bytes(source_payload))
    fixture["source_path"].write_bytes(launcher._pretty_json_bytes(source))
    identity = launcher._strict_json(
        fixture["identity_path"].read_bytes(),
        context="identity",
    )
    identity["evidence"]["execution_bindings"]["repository_source_manifest_file_sha256"] = _digest(
        fixture["source_path"].read_bytes()
    )
    identity["canonical_evidence_sha256"] = _digest(
        launcher._canonical_json_bytes(identity["evidence"])
    )
    fixture["identity_path"].write_bytes(launcher._canonical_json_bytes(identity))
    pycache = tmp_path / "network-pycache"
    scratch = tmp_path / "network-scratch"
    pycache.mkdir()
    scratch.mkdir()
    command = launcher._sealed_argv(
        interpreter=fixture["interpreter"],
        bootstrap=fixture["bootstrap"],
        runtime_manifest=fixture["runtime_path"],
        base_runtime_root=fixture["base"],
        package_roots={"packages": fixture["package_root"]},
        git_executable=fixture["git_executable"],
        pycache_prefix=pycache,
        scratch_directory=scratch,
        runner_arguments=fixture["runner_arguments"],
    )

    completed = subprocess.run(
        command,
        check=False,
        cwd=fixture["base"],
        env=launcher._sealed_environment(scratch_directory=scratch, offline=True),
        capture_output=True,
        input=fixture["bootstrap"],
    )

    assert completed.returncode != 37
    assert b"forbids network access: socket.connect" in completed.stderr
    assert not fixture["sentinel"].exists()


def test_split_launcher_arguments_requires_one_separator() -> None:
    with pytest.raises(launcher.SealedStageALaunchError, match="exactly one"):
        launcher._split_arguments(["a"])
    with pytest.raises(launcher.SealedStageALaunchError, match="exactly one"):
        launcher._split_arguments(["a", "--", "b", "--", "c"])
    assert launcher._split_arguments(["host", "--", "preflight"]) == (
        ["host"],
        ["preflight"],
    )


def test_launch_reauthenticates_after_child_and_uses_isolated_argv(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(b"runtime")
    repository = tmp_path / "repo"
    repository.mkdir()
    base = tmp_path / "base"
    packages = tmp_path / "packages"
    base.mkdir()
    packages.mkdir()
    git_executable = _write(tmp_path / "toolchain" / "git.exe", b"fixture Git executable\n")
    git_record = {
        "absolute_path_sha256": _digest(str(git_executable.resolve()).casefold()),
        "sha256": _digest(git_executable.read_bytes()),
        "size_bytes": git_executable.stat().st_size,
    }
    source_git_record = {
        "sha256": git_record["sha256"],
        "size_bytes": git_record["size_bytes"],
    }
    runner_args = ["preflight"]
    runner_args.extend(
        pair
        for option in sorted(launcher._REQUIRED_OPTIONS)
        for pair in (option, str(repository) if option != "--source-commit" else "2" * 40)
    )
    calls: list[str] = []
    options = {
        option: value for option, value in zip(runner_args[1::2], runner_args[2::2], strict=True)
    }
    options["--repository-root"] = str(repository)
    monkeypatch.setattr(launcher, "_extract_options", lambda args: options)
    monkeypatch.setattr(
        launcher,
        "_verify_bound_inputs",
        lambda *args, **kwargs: (
            calls.append("verify")
            or ({}, {"paths": [], "git_executable": source_git_record}, Path("runner"))
        ),
    )
    fake_calibration = SimpleNamespace(
        _parse_runtime_manifest=lambda data: {"git_executable": git_record},
        _verify_runtime=lambda *args, **kwargs: (
            base,
            {"packages": packages},
            {"packages": "."},
            Path(sys.executable),
            git_executable,
        ),
        _verify_empty_scratch=lambda path: None,
        _assert_scratch_tree_has_no_reparse=lambda path: None,
    )
    monkeypatch.setattr(launcher, "_load_calibration_launcher", lambda *args: fake_calibration)
    monkeypatch.setattr(launcher, "_stage_a_bootstrap", lambda module: "raise SystemExit(0)")
    child_modes: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append("subprocess")
        child_modes.append(command[16])
        assert command[1:4] == ["-I", "-S", "-B"]
        assert command[9] == launcher._authenticated_stdin_loader(b"raise SystemExit(0)")
        assert kwargs["input"] == b"raise SystemExit(0)"
        if command[16] == "prepare-inputs":
            assert "HF_HUB_OFFLINE" not in kwargs["env"]
        else:
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = launcher.launch(
        [
            "--base-runtime-root",
            str(base),
            "--git-executable",
            str(git_executable),
            "--package-root",
            f"packages={packages}",
            "--runtime-manifest",
            str(runtime_path),
            "--",
            *runner_args,
        ]
    )
    assert result == 0
    assert calls == ["verify", "subprocess", "verify", "subprocess", "verify"]
    assert child_modes == ["prepare-inputs", "preflight"]
