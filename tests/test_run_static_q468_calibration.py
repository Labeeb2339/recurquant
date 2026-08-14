from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
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


def fisher_boundary_contract(
    token_ids: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    return identity_resolver.build_fisher_boundary_contract(token_ids)


def record(token_ids: tuple[int, ...] = (1, 2, 3)) -> dict[str, object]:
    prompt_stop = max(1, len(token_ids) - 1)
    return {
        "canonical_id": "item-1",
        "config": "default",
        "family": "mbpp",
        "formatted_content_sha256": "b" * 64,
        "fisher_boundary": fisher_boundary_contract(token_ids),
        "generator_receipt_sha256": None,
        "prompt_token_ids_sha256": token_digest(token_ids[:prompt_stop]),
        "ruler_category": None,
        "seed": None,
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
    ) -> None:
        self.frozen_identity = frozen_identity
        self.events = events
        self.stability_passed = stability_passed
        self.decode_error = decode_error
        self.expected_source_commit = expected_source_commit or current_head()
        self.captured: list[Any] = []

    def decode_identity(self, data: bytes) -> Any:
        self.events.append("decode_identity")
        assert data == b"identity"
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
            stage_a_binding=b"binding",
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
        assert expected == {"manifest": "expected", "source_commit": expected_source_commit}
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
    source_bytes = runner.canonical_json_bytes(
        {"manifest": "expected", "source_commit": expected_source_commit}
    )
    model_bytes = model_manifest_bytes(files)
    runtime_bytes = runtime_manifest_bytes()
    parquet_bytes = b'{"artifact_kind":"test-parquet-materializations"}\n'
    frozen = identity(
        selected_records,
        source_manifest_sha256=digest(source_bytes),
        runtime_manifest_sha256=digest(runtime_bytes),
        model_manifest_sha256=digest(model_bytes),
        parquet_manifest_sha256=digest(parquet_bytes),
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
            distributions=(("transformers", "5.14.1"),),
            distribution_count=1,
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
        frozen_identity_bytes=b"identity",
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
            "adapter": {"fisher_step_count": first_fisher_count},
            "authenticated_distribution_count": authenticated_runtime.distribution_count,
            "authenticated_file_count": authenticated_runtime.file_count,
            "cuda_available": True,
            "cuda_runtime": "test",
            "elapsed_seconds_hex": (0.0).hex(),
            "gpu": {"name": "test-gpu"},
            "packages": dict(authenticated_runtime.distributions),
            "platform": "test",
            "python": "test",
            "runtime_manifest_file_sha256": authenticated_runtime.manifest_file_sha256,
            "torch": "test",
        },
        fisher_h1_smoke_report_file_sha256=None,
    )
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=smoke_report,
        prior_fisher_h1_smoke_complete_bytes=runner.FISHER_SMOKE_COMPLETE_BYTES,
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
        runner.BINDING_FILENAME,
        runner.REPORT_FILENAME,
        runner.COMPLETE_FILENAME,
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
    assert report["evidence"]["status"] == "passed"
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
        "fisher_h1_smoke_report_file_sha256": digest(config.prior_fisher_h1_smoke_report_bytes)
    }


def test_fisher_h1_smoke_runs_first_frozen_sequence_and_publishes_only_receipt(
    tmp_path: Path,
) -> None:
    first = record()
    second = {**record(), "canonical_id": "item-2"}
    config, adapter, services, events = configured_run(
        tmp_path,
        records=[first, second],
    )
    config = replace(
        config,
        fisher_h1_smoke=True,
        prior_fisher_h1_smoke_report_bytes=None,
        prior_fisher_h1_smoke_complete_bytes=None,
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
        runner.FISHER_SMOKE_COMPLETE_FILENAME,
    }
    report = json.loads((config.output_dir / runner.FISHER_SMOKE_REPORT_FILENAME).read_text())
    assert report["evidence"]["status"] == "fisher_h1_smoke_passed"
    assert report["evidence"]["calibration"] == {
        "expected_fisher_step_count": 1,
        "observed_fisher_step_count": 1,
        "fisher_boundary_count": 1,
        "post_token_anchor_count": 3,
        "sequence_count": 1,
        "token_count": 3,
    }
    assert report["evidence"]["artifacts"] == {}
    assert report["evidence"]["prerequisites"] == {"fisher_h1_smoke_report_file_sha256": None}


def test_full_calibration_requires_authenticated_prior_fisher_smoke_before_data(
    tmp_path: Path,
) -> None:
    config, adapter, services, events = configured_run(tmp_path)
    config = replace(
        config,
        prior_fisher_h1_smoke_report_bytes=None,
        prior_fisher_h1_smoke_complete_bytes=None,
    )

    with pytest.raises(runner.CalibrationRunError, match="requires the prior Fisher H=1"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity"]
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
    boundary = fisher_boundary_contract()
    evidence_record = {
        "canonical_id": "item-1",
        "fisher_boundary": boundary,
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
        runner.COMPLETE_FILENAME,
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

    with pytest.raises(runner.CalibrationRunError, match="frozen identity binding"):
        runner.run_calibration(config, adapter, services=services)

    assert events == ["decode_identity"]


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

    with pytest.raises(runner.CalibrationRunError, match="runtime manifest bytes"):
        runner.run_calibration(config, adapter, services=services)

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


def test_stage_model_authenticates_before_touching_downloader_cache_or_output(
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
            hub_cache_root=cache,
            output_root=output,
            downloader=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert not cache.exists()
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
    }

    first = runner.verify_frozen_identity_contract(**arguments)
    second = runner.verify_frozen_identity_contract(**arguments)

    expected = {
        "artifact_kind": runner.FROZEN_IDENTITY_CONTRACT_KIND,
        "assignment_sha256": "f" * 64,
        "canonical_evidence_sha256": "e" * 64,
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
            else pytest.fail("unexpected stable read outside common authorization")
        ),
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
    )

    assert result.identity == source_authorization.identity
    assert result.identity_commit == "3" * 40
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
    }

    first = runner.verify_identity_bound_model_staging_authorization(**arguments)
    second = runner.verify_identity_bound_model_staging_authorization(**arguments)

    expected = {
        "artifact_kind": runner.MODEL_STAGING_AUTHORIZATION_KIND,
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
    output = tmp_path / "published-model"
    result = runner.stage_identity_bound_model(
        frozen_identity_path=tmp_path / "identity.json",
        expected_frozen_identity_sha256="d" * 64,
        identity_commit="3" * 40,
        repository_root=SCRIPT.parents[1],
        repository_source_manifest_path=tmp_path / "source.json",
        source_commit="1" * 40,
        model_file_manifest_path=tmp_path / "model-manifest.json",
        expected_model_file_manifest_sha256=authorization.model_manifest.file_sha256,
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
            hub_cache_root=tmp_path / "cache",
            output_root=output,
            downloader=download,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".published-model.staging-*"))


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
            hub_cache_root=tmp_path / "cache",
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
        "--hub-cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "output"),
        "--local-files-only",
    ]

    assert runner.main(arguments) == 0
    assert calls == [
        {
            "expected_frozen_identity_sha256": "1" * 64,
            "expected_model_file_manifest_sha256": "4" * 64,
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
        }
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--identity-commit", "2" * 40),
        ("--model-file-manifest", "model.json"),
        ("--expected-model-file-manifest-sha256", "4" * 64),
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
    ]

    assert runner.main(arguments) == 0
    assert capsys.readouterr().out == runner.canonical_json_bytes(receipt).decode("utf-8")
    assert calls == [
        {
            "expected_frozen_identity_sha256": "1" * 64,
            "expected_model_file_manifest_sha256": "4" * 64,
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
    imported: list[bool] = []
    monkeypatch.setattr(runner, "_load_adapter", lambda *args, **kwargs: imported.append(True))

    with pytest.raises(runner.CalibrationRunError, match="bootstrap binding"):
        runner._official_main(
            [
                "--frozen-identity",
                str(paths["identity"]),
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
                "--ruler-root",
                str(tmp_path / "unopened-ruler"),
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
