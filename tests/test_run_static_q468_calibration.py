from __future__ import annotations

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


def record(token_ids: tuple[int, ...] = (1, 2, 3)) -> dict[str, object]:
    prompt_stop = max(1, len(token_ids) - 1)
    return {
        "canonical_id": "item-1",
        "config": "default",
        "family": "mbpp",
        "formatted_content_sha256": "b" * 64,
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
        "schema_version": 4,
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


def runtime_manifest_bytes() -> bytes:
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
    document["evidence"]["schema_version"] = 4.0
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
    ) -> None:
        self.frozen_identity = frozen_identity
        self.events = events
        self.stability_passed = stability_passed
        self.decode_error = decode_error
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
        assert source_commit == current_head()
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
            split_half=b"split",
            static_k27030=b"k27030",
            static_k29334=b"k29334",
            static_q48=b"q48",
            stage_a_binding=b"binding",
            stability=stability,
            calibration_scores_sha256="3" * 64,
            sequence_score_manifest_sha256="4" * 64,
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
        geometry = FakeBackend.geometry
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


def source_verifier(events: list[str], *, fail_on_call: int | None = None) -> Any:
    calls = 0

    def verify(expected: Mapping[str, object], root: Path) -> Any:
        nonlocal calls
        calls += 1
        events.append("verify_source")
        assert expected == {"manifest": "expected", "source_commit": current_head()}
        assert root == SCRIPT.parents[1]
        if fail_on_call == calls:
            raise runner.CalibrationRunError("source drift")
        return {"manifest": "expected", "source_commit": current_head()}, "5" * 64

    return verify


def configured_run(
    tmp_path: Path,
    *,
    records: Sequence[dict[str, object]] | None = None,
    stability_passed: bool = True,
    source_fail_on_call: int | None = None,
    decode_error: BaseException | None = None,
) -> tuple[Any, Any, Any, Any]:
    selected_records = list(records or [record()])
    files = {"config.json": b"{}", "model.safetensors": b"safe-test-placeholder"}
    source_bytes = runner.canonical_json_bytes(
        {"manifest": "expected", "source_commit": current_head()}
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
        verify_repository_source=source_verifier(events, fail_on_call=source_fail_on_call),
        validate_adapter=lambda _adapter: events.append("validate_adapter"),
        distortion_function=fake_distortions,
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
        expected_source_commit=current_head(),
        expected_model_file_manifest_sha256=digest(model_bytes),
        expected_parquet_materialization_manifest_sha256=digest(parquet_bytes),
        expected_runtime_manifest_sha256=digest(runtime_bytes),
        output_dir=tmp_path / "output",
        require_cuda=False,
    )
    return config, adapter, services, events


def test_success_authenticates_every_boundary_and_publishes_complete_set(tmp_path: Path) -> None:
    config, adapter, services, events = configured_run(tmp_path)

    result = runner.run_calibration(config, adapter, services=services)

    assert result["status"] == "passed"
    assert set(path.name for path in config.output_dir.iterdir()) == {
        runner.SCORE_FILENAME,
        runner.SPLIT_FILENAME,
        runner.K27030_FILENAME,
        runner.K29334_FILENAME,
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
        "anchor_count": 3,
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


def test_identity_view_consumes_strict_schema_v4_execution_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {
        "calibration_runtime_manifest_file_sha256": "1" * 64,
        "model_file_manifest_file_sha256": "2" * 64,
        "parquet_materialization_manifest_file_sha256": "9" * 64,
        "repository_source_manifest_file_sha256": "3" * 64,
    }
    evidence = {
        "execution_bindings": bindings,
        "model_contracts": {"primary": {"id": "example/model", "revision": "4" * 40}},
        "source_manifest_sha256": "5" * 64,
        "tokenizer": {"transformers_version": "5.14.1"},
    }
    payload = runner.canonical_json_bytes({"evidence": evidence})

    class Decoded:
        file_sha256 = digest(payload)
        canonical_evidence_sha256 = "6" * 64
        records = ()
        assignment = ()
        assignment_sha256 = "7" * 64
        tokenizer_manifest_sha256 = "8" * 64
        execution_bindings = bindings

    class Resolver:
        @staticmethod
        def deserialize_frozen_calibration_identity_artifact(data: bytes) -> Decoded:
            assert data == payload
            return Decoded()

    monkeypatch.setattr(runner, "_load_identity_resolver", lambda _root: Resolver())
    decoded = runner._identity_view(payload, tmp_path)

    assert decoded.repository_source_manifest_file_sha256 == "3" * 64
    assert decoded.runtime_manifest_file_sha256 == "1" * 64
    assert decoded.model_file_manifest_file_sha256 == "2" * 64
    assert decoded.parquet_materialization_manifest_file_sha256 == "9" * 64


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


def test_source_manifest_commit_must_equal_reported_head(tmp_path: Path) -> None:
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
        return (
            replace(authenticated, machine_name="drifted")
            if calls == 2
            else authenticated
        )

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

    assert runner.validate_materialized_sequence(item, candidate, calibration_api=api) == token_ids


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
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
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
    captured = runner.capture_sequence_causally(
        adapter,
        object(),
        item,
        tokens,
        geometry=geometry,
        calibration_api=api,
        require_cuda=False,
        distortion_function=fake_distortions,
    )

    assert captured.anchor_positions == runner.frozen_anchor_positions(17)
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


def test_official_capture_rejects_cpu_queries_even_when_cuda_might_exist() -> None:
    geometry = FakeBackend.geometry
    tokens = (1,)
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

    with pytest.raises(runner.CalibrationRunError, match="must be FP32"):
        runner.capture_sequence_causally(
            Bf16Adapter({}, []),
            object(),
            record((1,)),
            (1,),
            geometry=geometry,
            calibration_api=api,
            require_cuda=False,
            distortion_function=fake_distortions,
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


def test_model_manifest_rejects_noncanonical_duplicate_and_traversal() -> None:
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    valid = model_manifest_bytes(files)
    parsed = runner.parse_model_file_manifest(valid)
    assert parsed.file_sha256 == digest(valid)
    assert [item.name for item in parsed.files] == ["config.json", "model.safetensors"]

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

    assert [item.name for item in parsed.files] == ["config.json", "model.safetensors"]
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
            del args, kwargs
            calls.append("model_info")
            return Info()

        @staticmethod
        def list_repo_tree(*args: object, **kwargs: object) -> list[dict[str, object]]:
            del args, kwargs
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
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    root = tmp_path / "model"
    write_model_root(root, {**files, "unbound.txt": b"extra"})
    manifest = runner.parse_model_file_manifest(model_manifest_bytes(files))

    with pytest.raises(runner.CalibrationRunError, match="file set differs"):
        runner.authenticate_local_model_files(root, manifest, calibration_api=api)


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
    record_text = (
        "demo/__init__.py,,\n"
        "demo-1.0.dist-info/RECORD,,\n"
        "../../Scripts/demo.exe,,\n"
    )
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
        def capture_experiment013_source_manifest(root: Path) -> dict[str, object]:
            assert root == repository_root
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
        ) -> dict[str, object]:
            assert manifest is source_manifest
            assert root == repository_root
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
                "--repository-root",
                str(repository_root),
                "--output",
                str(source_output),
            ]
        )

    runtime_args = [
        "capture-runtime-manifest",
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
    ) -> bytes:
        assert (model_id, revision, transformers_version) == (
            "example/model",
            "4" * 40,
            "5.14.1",
        )
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


def test_source_manifest_output_location_allows_only_external_or_ignored_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")

    outside = tmp_path / "outside.json"
    ignored = repository / "artifacts" / "source.json"
    assert runner._assert_source_manifest_output_location(repository, outside) == outside.resolve()
    assert runner._assert_source_manifest_output_location(repository, ignored) == ignored.resolve()

    with pytest.raises(runner.CalibrationRunError, match="must be ignored"):
        runner._assert_source_manifest_output_location(repository, repository / "source.json")
    with pytest.raises(runner.CalibrationRunError, match="repository metadata"):
        runner._assert_source_manifest_output_location(repository, repository / ".git" / "new.json")


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
        [str(SCRIPT.parents[1] / ".venv" / "Scripts" / "python.exe"), "-c", code],
        cwd=SCRIPT.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
