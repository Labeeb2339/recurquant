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


def _identity(
    bindings: dict[str, str],
    *,
    capture_receipt_sha256: str | None = None,
    identity_input_sha256: str | None = None,
    authorization_sha256: str | None = None,
) -> bytes:
    receipt_sha256 = _digest("fixture-stage-a-capture-receipt")
    if capture_receipt_sha256 is not None:
        receipt_sha256 = capture_receipt_sha256
    evidence = {
        "schema_version": 6,
        "identity_schema": "recurquant.experiment013.identity-frozen.v6",
        "status": "frozen",
        "phase": "stage_a",
        "identity_only": True,
        "promotion_required": False,
        "promotion": {
            "candidate_file_sha256": _digest("candidate-file"),
            "candidate_canonical_evidence_sha256": _digest("candidate-evidence"),
            "explicit": True,
            launcher.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: receipt_sha256,
        },
        "source_manifest_sha256": identity_input_sha256 or _digest("identity-input"),
        launcher.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: receipt_sha256,
        "execution_bindings": bindings,
        "calibration_binding": {
            "calibration_authorization_file_sha256": (
                authorization_sha256 or _digest("calibration-authorization")
            )
        },
    }
    return launcher._canonical_json_bytes(
        {
            "canonical_evidence_sha256": _digest(launcher._canonical_json_bytes(evidence)),
            "evidence": evidence,
        }
    )


def _capture_receipt(
    bindings: dict[str, str],
    *,
    binding_bytes: bytes,
    capture_source_sha256: str,
    identity_input_sha256: str | None = None,
    authorization_sha256: str | None = None,
) -> bytes:
    return launcher._canonical_json_bytes(
        {
            "artifact_kind": launcher.STAGE_A_CAPTURE_PROVENANCE_KIND,
            "calibration_authorization_file_sha256": (
                authorization_sha256 or _digest("calibration-authorization")
            ),
            "calibration_binding_file_sha256": _digest(binding_bytes),
            "capture_source": {
                "path": "scripts/capture_static_q468_identity_input.py",
                "sha256": capture_source_sha256,
            },
            "capture_version": 6,
            "critical_module_origins": [],
            "excluded_runtime_modules": ["pkg_resources", "setuptools"],
            "execution_bindings": dict(bindings),
            "identity_input_file_sha256": identity_input_sha256 or _digest("identity-input"),
            "phase": "stage_a",
            "publication_contract": launcher.STAGE_A_CAPTURE_PUBLICATION_CONTRACT,
            "runner_revision": launcher.STAGE_A_CAPTURE_RUNNER_REVISION,
            "schema_version": 1,
            "source_commit": "2" * 40,
            "status": launcher.STAGE_A_CAPTURE_PROVENANCE_STATUS,
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
            "scripts/capture_static_q468_identity_input.py",
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
        "--stage-a-capture-provenance-receipt": paths["receipt"],
        "--expected-stage-a-capture-provenance-receipt-sha256": _digest(
            paths["receipt"].read_bytes()
        ),
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
    _write(
        repository / "scripts" / "capture_static_q468_identity_input.py",
        b"# authenticated capture fixture\n",
    )
    source_paths = []
    for relative in sorted(
        {
            launcher.RUNNER_SOURCE_PATH,
            launcher.LAUNCHER_SOURCE_PATH,
            launcher.CALIBRATION_LAUNCHER_SOURCE_PATH,
            "scripts/capture_static_q468_identity_input.py",
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
    capture_source_sha256 = next(
        entry["raw_sha256"]
        for entry in source_paths
        if entry["path"] == "scripts/capture_static_q468_identity_input.py"
    )
    receipt_bytes = _capture_receipt(
        bindings,
        binding_bytes=binding_path.read_bytes(),
        capture_source_sha256=capture_source_sha256,
    )
    receipt_path = _write(artifacts / "stage-a-capture-provenance.json", receipt_bytes)
    identity_path = _write(
        artifacts / "identity.json",
        _identity(bindings, capture_receipt_sha256=_digest(receipt_bytes)),
    )
    paths = {
        "binding": binding_path,
        "identity": identity_path,
        "model": model_path,
        "parquet": parquet_path,
        "receipt": receipt_path,
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
        "cache_root": repository,
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


def test_identity_parser_accepts_only_promoted_stage_a_v6() -> None:
    bindings = {name: _digest(name) for name in launcher._BOUND_ARTIFACT_OPTIONS}
    assert launcher._parse_identity(_identity(bindings)) == {
        name: bindings[name] for name in sorted(bindings)
    }

    root = launcher._strict_json(_identity(bindings), context="identity")
    root["evidence"]["phase"] = "calibration"
    root["canonical_evidence_sha256"] = _digest(launcher._canonical_json_bytes(root["evidence"]))
    with pytest.raises(launcher.SealedStageALaunchError, match="promoted Stage-A"):
        launcher._parse_identity(launcher._canonical_json_bytes(root))

    legacy = launcher._strict_json(_identity(bindings), context="legacy identity")
    legacy["evidence"]["schema_version"] = 5
    legacy["evidence"]["identity_schema"] = "recurquant.experiment013.identity-frozen.v5"
    legacy["canonical_evidence_sha256"] = _digest(
        launcher._canonical_json_bytes(legacy["evidence"])
    )
    with pytest.raises(launcher.SealedStageALaunchError, match="resolver-v6"):
        launcher._parse_identity(launcher._canonical_json_bytes(legacy))


def test_runner_cli_forbids_method_seed_threshold_and_policy_options(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in ("identity", "binding", "receipt", "runtime", "model", "parquet", "source")
    }
    paths["receipt"].write_bytes(b"fixture receipt")
    paths["root"] = tmp_path
    digests = {name: _digest(name) for name in ("runtime", "model", "parquet")}
    arguments = _runner_arguments(paths, digests)
    assert set(launcher._extract_options(arguments)) == launcher._REQUIRED_OPTIONS
    receipt_option_index = arguments.index("--stage-a-capture-provenance-receipt")
    without_receipt = [
        *arguments[:receipt_option_index],
        *arguments[receipt_option_index + 2 :],
    ]
    with pytest.raises(launcher.SealedStageALaunchError, match="omit required inputs"):
        launcher._extract_options(without_receipt)
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


def test_calibration_launcher_swap_cannot_execute_unauthenticated_second_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_path = tmp_path / launcher.CALIBRATION_LAUNCHER_SOURCE_PATH
    sentinel = tmp_path / "unauthenticated-launcher-executed.txt"
    authenticated_bytes = b'SEALED_BOOTSTRAP = "authenticated-old"\nMARKER = "old"\n'
    swapped_bytes = (
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('new code executed', encoding='utf-8')\n"
        'SEALED_BOOTSTRAP = "unauthenticated-new"\n'
    ).encode()
    _write(launcher_path, authenticated_bytes)
    source = {
        "paths": [
            {
                "path": launcher.CALIBRATION_LAUNCHER_SOURCE_PATH,
                "raw_sha256": _digest(authenticated_bytes),
            }
        ]
    }
    original_read_bytes = Path.read_bytes
    raced = False

    def racing_read_bytes(path: Path) -> bytes:
        nonlocal raced
        data = original_read_bytes(path)
        if path.resolve() == launcher_path.resolve() and not raced:
            raced = True
            launcher_path.write_bytes(swapped_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    with pytest.raises(
        launcher.SealedStageALaunchError, match="changed while it was authenticated"
    ):
        launcher._load_calibration_launcher(tmp_path, source)
    assert raced is True
    assert not sentinel.exists()
    assert launcher.CALIBRATION_LAUNCHER_MODULE_NAME not in sys.modules


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
    receipt = _capture_receipt(
        bindings,
        binding_bytes=files["binding"],
        capture_source_sha256=_digest(b"scripts/capture_static_q468_identity_input.py"),
    )
    paths["receipt"] = tmp_path / "stage-a-capture-provenance.json"
    paths["receipt"].write_bytes(receipt)
    paths["identity"] = tmp_path / "identity.json"
    paths["identity"].write_bytes(_identity(bindings, capture_receipt_sha256=_digest(receipt)))
    arguments = _runner_arguments(
        paths,
        {"runtime": _digest(runtime), "model": _digest(model), "parquet": _digest(parquet)},
    )
    options = launcher._extract_options(arguments)
    parsed, _source, runner, authenticated_runtime = launcher._verify_bound_inputs(
        options, runtime_manifest_path=paths["runtime"]
    )
    assert parsed == bindings
    assert runner.name == "screen_static_q468_stage_a.py"
    assert authenticated_runtime == runtime

    paths["model"].write_bytes(b"tampered")
    with pytest.raises(launcher.SealedStageALaunchError, match="identity binding mismatch"):
        launcher._verify_bound_inputs(options, runtime_manifest_path=paths["runtime"])

    paths["model"].unlink()
    paths["receipt"].write_bytes(b"tampered finalized receipt")
    with pytest.raises(
        launcher.SealedStageALaunchError,
        match="capture provenance receipt differs from authenticated identity custody",
    ):
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
    assert 'evidence.get("schema_version") != 6' in bootstrap
    assert 'evidence.get("phase") != "stage_a"' in bootstrap
    assert launcher.RUNNER_SOURCE_PATH in bootstrap
    assert "scripts/run_static_q468_calibration.py" not in bootstrap
    assert "_stage_a_options" in bootstrap
    assert "--stage-a-calibration-binding" in bootstrap
    assert "--stage-a-capture-provenance-receipt" in bootstrap
    assert "--expected-stage-a-capture-provenance-receipt-sha256" in bootstrap
    assert "--fisher-h1-smoke" not in bootstrap
    assert "--prior-fisher-h1-smoke-report" not in bootstrap
    assert "--prior-fisher-h1-smoke-launch-finalization" not in bootstrap
    assert "--expected-prior-fisher-h1-smoke-launch-finalization-sha256" not in bootstrap
    assert "full calibration" not in bootstrap
    assert 'promotion.get("explicit") is not True' in bootstrap
    assert "isolated != 1" in bootstrap
    assert "no_site != 1" in bootstrap
    assert "dont_write_bytecode != 1" in bootstrap
    assert "experiment-013-static-q468-calibration-runner-v16" in bootstrap
    assert "_hf_hub_cache = _cache_root" in bootstrap
    assert (
        "allowlisted-private-scratch-roots-plus-explicit-dataset-and-phase-hub-roots-v4"
        in bootstrap
    )
    assert "exact-environment-root-allowlist-regular-non-link-cleanup-v1" in bootstrap
    assert "def _assert_isolated_cache(cache, runtime_roots):" in bootstrap
    assert "cache_identity in {(item[1], item[2]) for item in runtime_chain}" in bootstrap
    assert "runtime_identity in {(item[1], item[2]) for item in cache_chain}" in bootstrap


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
            cwd=scratch,
            env=launcher._sealed_environment(
                scratch_directory=scratch,
                dataset_cache_root=fixture["cache_root"],
                offline=offline,
            ),
            capture_output=True,
            input=fixture["bootstrap"],
        )

    completed = run_bootstrap(tmp_path / "pycache-valid")
    assert completed.returncode == 37, completed.stderr
    assert fixture["sentinel"].read_text(encoding="utf-8") == "stage-a-boundary\n"

    fixture["sentinel"].unlink()
    receipt_path = Path(
        fixture["runner_arguments"][
            fixture["runner_arguments"].index("--stage-a-capture-provenance-receipt") + 1
        ]
    )
    receipt_bytes = receipt_path.read_bytes()
    model_bytes = fixture["model_path"].read_bytes()
    receipt_path.write_bytes(b"tampered finalized Stage-A receipt")
    fixture["model_path"].unlink()
    receipt_rejected = run_bootstrap(tmp_path / "pycache-receipt-tampered")
    assert receipt_rejected.returncode != 37
    assert b"capture provenance receipt differs from identity custody" in receipt_rejected.stderr
    assert not fixture["sentinel"].exists()
    receipt_path.write_bytes(receipt_bytes)
    fixture["model_path"].write_bytes(model_bytes)

    missing_offline = run_bootstrap(tmp_path / "pycache-missing-offline", offline=False)
    assert missing_offline.returncode != 37
    assert b"private cache contract" in missing_offline.stderr
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
    cache_root = tmp_path / "dataset-cache"
    cache_root.mkdir()
    for name in (
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.setenv(name, "must-not-cross-child-boundary")

    networked = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
        offline=False,
    )
    offline = launcher._sealed_environment(
        scratch_directory=scratch,
        dataset_cache_root=cache_root,
        offline=True,
    )

    assert not {"HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} & set(networked)
    assert {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items() <= offline.items()
    forbidden = {"GITHUB_TOKEN", "HF_TOKEN", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"}
    assert not forbidden & set(networked)
    assert not forbidden & set(offline)
    assert networked["HOME"] == networked["USERPROFILE"] == str(scratch / "private-home")
    assert networked["HF_HUB_CACHE"] == str(cache_root)
    assert networked["HUGGINGFACE_HUB_CACHE"] == str(cache_root)
    assert networked["HF_DATASETS_CACHE"] == str(cache_root / "datasets")
    assert networked["HF_HUB_DISABLE_UPDATE_CHECK"] == "1"
    assert networked["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert networked["HF_HUB_DISABLE_XET"] == "1"


def test_stage_a_external_hub_link_stays_outside_owned_scratch(tmp_path: Path) -> None:
    helpers = _load_calibration_launcher("stage_a_external_hub_link_test")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cache_root = tmp_path / "dataset-cache"
    blob = cache_root / "datasets--google-research-datasets--mbpp" / "blobs" / ("a" * 40)
    snapshot = cache_root / "datasets--google-research-datasets--mbpp" / "snapshots" / ("b" * 40)
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
        offline=True,
    )
    assert environment["HF_HUB_CACHE"] == str(cache_root)
    assert link.is_symlink()
    scratch_identity = helpers._temporary_directory_identity(
        scratch,
        context="Stage-A sealed scratch directory",
    )
    helpers._cleanup_owned_temporary_directory(
        scratch,
        expected_identity=scratch_identity,
        context="Stage-A sealed scratch directory",
    )

    assert not scratch.exists()
    assert link.is_symlink()
    assert link.read_bytes() == b"dataset card\n"


def test_stage_a_partial_temp_creation_cleans_first_owned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _load_calibration_launcher("stage_a_partial_cleanup_test")
    created: list[Path] = []

    def mkdtemp(*, prefix: str) -> str:
        if created:
            raise OSError("second Stage-A temporary-root creation failed")
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

    with pytest.raises(OSError, match="second Stage-A temporary-root creation failed"):
        launcher._run_sealed_child(
            calibration_launcher=calibration,
            bootstrap=b"raise SystemExit(0)",
            runtime_manifest_path=tmp_path / "runtime.json",
            runtime_manifest_bytes=b"runtime",
            runtime_manifest={},
            interpreter=Path(sys.executable),
            base_runtime_root=tmp_path,
            package_roots={},
            git_executable=tmp_path / "git.exe",
            source={},
            options={},
            runner_arguments=["preflight"],
            dataset_cache_root=tmp_path,
            offline=True,
        )

    assert created and all(not path.exists() for path in created)


def test_stage_a_nonzero_child_preserves_code_and_cleans_allowlisted_scratch_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    helpers = _load_calibration_launcher("stage_a_residue_cleanup_test")
    base = tmp_path / "base"
    packages = tmp_path / "packages"
    cache = tmp_path / "dataset-cache"
    for path in (base, packages, cache):
        path.mkdir()
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(b"runtime")
    created: list[Path] = []
    events: list[str] = []

    def mkdtemp(*, prefix: str) -> str:
        path = tmp_path / f"{prefix}{len(created)}"
        path.mkdir()
        created.append(path)
        return str(path)

    def run_child(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        _write(Path(command[13]) / "late.pyc", b"bytecode")
        _write(Path(command[15]) / "private-home" / "late.cache", b"cache")
        return SimpleNamespace(returncode=37)

    calibration = SimpleNamespace(
        _temporary_directory_identity=helpers._temporary_directory_identity,
        _verify_empty_pycache=helpers._verify_empty_pycache,
        _verify_empty_scratch=helpers._verify_empty_scratch,
        _verify_contained_scratch=helpers._verify_contained_scratch,
        _verified_dataset_cache_root=helpers._verified_dataset_cache_root,
        _non_link_directory_identity_chain=helpers._non_link_directory_identity_chain,
        _assert_owned_temporary_tree_has_no_reparse=(
            helpers._assert_owned_temporary_tree_has_no_reparse
        ),
        _cleanup_owned_temporary_directory=helpers._cleanup_owned_temporary_directory,
        _postcondition_error=helpers._postcondition_error,
        _surface_secondary_failures=helpers._surface_secondary_failures,
        _verify_runtime=lambda *args, **kwargs: events.append("runtime"),
    )
    monkeypatch.setattr(launcher.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(launcher.subprocess, "run", run_child)
    monkeypatch.setattr(
        launcher,
        "_verify_bound_inputs",
        lambda *args, **kwargs: (
            events.append("bound")
            or (
                {},
                {"git_executable": {"sha256": "a", "size_bytes": 1}},
                Path("runner"),
                b"runtime",
            )
        ),
    )

    result = launcher._run_sealed_child(
        calibration_launcher=calibration,
        bootstrap=b"raise SystemExit(0)",
        runtime_manifest_path=runtime_path,
        runtime_manifest_bytes=b"runtime",
        runtime_manifest={},
        interpreter=Path(sys.executable),
        base_runtime_root=base,
        package_roots={"packages": packages},
        git_executable=tmp_path / "git.exe",
        source={"git_executable": {"sha256": "a", "size_bytes": 1}},
        options={},
        runner_arguments=["preflight"],
        dataset_cache_root=cache,
        offline=True,
    )

    assert result == 37
    assert events == ["bound", "runtime"]
    assert created and all(not path.exists() for path in created)
    diagnostic = capsys.readouterr().err
    assert "preserving child return code 37" in diagnostic
    assert "Stage-A pycache postcondition" in diagnostic
    assert "Stage-A scratch containment postcondition" not in diagnostic


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
    receipt_path = Path(
        fixture["runner_arguments"][
            fixture["runner_arguments"].index("--stage-a-capture-provenance-receipt") + 1
        ]
    )
    receipt = launcher._strict_json(receipt_path.read_bytes(), context="capture receipt")
    receipt["execution_bindings"]["repository_source_manifest_file_sha256"] = _digest(
        fixture["source_path"].read_bytes()
    )
    receipt_path.write_bytes(launcher._canonical_json_bytes(receipt))
    receipt_sha256 = _digest(receipt_path.read_bytes())
    identity["evidence"][launcher.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD] = receipt_sha256
    identity["evidence"]["promotion"][launcher.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD] = (
        receipt_sha256
    )
    identity["canonical_evidence_sha256"] = _digest(
        launcher._canonical_json_bytes(identity["evidence"])
    )
    fixture["identity_path"].write_bytes(launcher._canonical_json_bytes(identity))
    expected_receipt_index = fixture["runner_arguments"].index(
        "--expected-stage-a-capture-provenance-receipt-sha256"
    )
    fixture["runner_arguments"][expected_receipt_index + 1] = receipt_sha256
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
        cwd=scratch,
        env=launcher._sealed_environment(
            scratch_directory=scratch,
            dataset_cache_root=fixture["cache_root"],
            offline=True,
        ),
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
    authenticated_runtime_bytes = b"authenticated runtime manifest"
    swapped_runtime_bytes = b"swapped runtime selecting an attacker interpreter"
    runtime_path.write_bytes(authenticated_runtime_bytes)
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
    verification_count = 0

    def verify_bound_inputs(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        nonlocal verification_count
        verification_count += 1
        calls.append("verify")
        if verification_count == 1:
            runtime_path.write_bytes(swapped_runtime_bytes)
        return (
            {},
            {"paths": [], "git_executable": source_git_record},
            Path("runner"),
            authenticated_runtime_bytes,
        )

    monkeypatch.setattr(launcher, "_verify_bound_inputs", verify_bound_inputs)
    calibration_helpers = _load_calibration_launcher("stage_a_launch_helper_test")
    parsed_runtime_bytes: list[bytes] = []

    def parse_runtime_manifest(data: bytes) -> dict[str, Any]:
        parsed_runtime_bytes.append(data)
        return {"git_executable": git_record}

    class FakeExecutableCustody:
        active = False

        def __enter__(self) -> FakeExecutableCustody:
            self.active = True
            return self

        def __exit__(self, *_args: Any) -> bool:
            self.active = False
            return False

        def verify(self) -> None:
            assert self.active
            return None

    fake_custody = FakeExecutableCustody()
    fake_calibration = SimpleNamespace(
        _acquire_executable_custody=lambda **_kwargs: fake_custody,
        _parse_runtime_manifest=parse_runtime_manifest,
        _verify_runtime=lambda *args, **kwargs: (
            base,
            {"packages": packages},
            {"packages": "."},
            Path(sys.executable),
            git_executable,
        ),
        _verified_dataset_cache_root=calibration_helpers._verified_dataset_cache_root,
        _non_link_directory_identity_chain=(calibration_helpers._non_link_directory_identity_chain),
        _temporary_directory_identity=calibration_helpers._temporary_directory_identity,
        _verify_empty_pycache=calibration_helpers._verify_empty_pycache,
        _verify_empty_scratch=calibration_helpers._verify_empty_scratch,
        _verify_contained_scratch=calibration_helpers._verify_contained_scratch,
        _assert_owned_temporary_tree_has_no_reparse=(
            calibration_helpers._assert_owned_temporary_tree_has_no_reparse
        ),
        _cleanup_owned_temporary_directory=(calibration_helpers._cleanup_owned_temporary_directory),
        _postcondition_error=calibration_helpers._postcondition_error,
        _surface_secondary_failures=calibration_helpers._surface_secondary_failures,
    )
    monkeypatch.setattr(launcher, "_load_calibration_launcher", lambda *args: fake_calibration)
    monkeypatch.setattr(launcher, "_stage_a_bootstrap", lambda module: "raise SystemExit(0)")
    child_modes: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert fake_custody.active
        calls.append("subprocess")
        child_modes.append(command[16])
        assert command[1:4] == ["-I", "-S", "-B"]
        assert command[9] == launcher._authenticated_stdin_loader(b"raise SystemExit(0)")
        assert kwargs["input"] == b"raise SystemExit(0)"
        assert kwargs["cwd"] == Path(command[15])
        if command[16] == "prepare-inputs":
            assert "HF_HUB_OFFLINE" not in kwargs["env"]
        else:
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["HF_HUB_CACHE"] == str(repository)
        assert kwargs["env"]["HUGGINGFACE_HUB_CACHE"] == str(repository)
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
    assert fake_custody.active is False
    assert calls == [
        "verify",
        "verify",
        "subprocess",
        "verify",
        "subprocess",
        "verify",
    ]
    assert child_modes == ["prepare-inputs", "preflight"]
    assert parsed_runtime_bytes == [authenticated_runtime_bytes, authenticated_runtime_bytes]
    assert runtime_path.read_bytes() == swapped_runtime_bytes
