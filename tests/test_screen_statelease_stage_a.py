from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recurquant.statelease_evaluation import (
    EQUAL_BYTE_NO_REPLAY_METHODS,
    FIXED_REPLAY_METHODS,
    FROZEN_STATELEASE_RESIDENT_BYTES,
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    evaluate_statelease_stage_a_gate,
)
from scripts import screen_statelease_stage_a as stage_a


def _unused(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("unauthorized hook was called")


def _record_transition_identity(
    receipt: object,
    _ledger: stage_a.AccessLedger,
) -> object:
    return receipt


def _hooks(
    *,
    authenticate=_unused,
    authenticate_readiness=_unused,
    load_config=_unused,
    reserve_attempt=_unused,
    load_exact_task=_unused,
    tokenize_task=_unused,
    load_weights=_unused,
    evaluate=_unused,
    record_access_transition=_record_transition_identity,
    record_evaluation_returned=_record_transition_identity,
    finalize=_unused,
    record_failure=_unused,
) -> stage_a.AccessHooks:
    return stage_a.AccessHooks(
        authenticate=authenticate,
        authenticate_readiness=authenticate_readiness,
        load_config=load_config,
        reserve_attempt=reserve_attempt,
        load_exact_task=load_exact_task,
        tokenize_task=tokenize_task,
        load_weights=load_weights,
        evaluate=evaluate,
        record_access_transition=record_access_transition,
        record_evaluation_returned=record_evaluation_returned,
        finalize=finalize,
        record_failure=record_failure,
    )


def test_ordered_access_authenticates_before_data_and_tokens_before_weights() -> None:
    events: list[str] = []

    def step(name: str, result: object):
        def callback(*_args: object) -> object:
            events.append(name)
            return result

        return callback

    hooks = _hooks(
        authenticate=step("authenticate", "auth"),
        authenticate_readiness=step("readiness", "ready"),
        load_config=step("config", "config"),
        reserve_attempt=step("reserve", "attempt"),
        load_exact_task=step("data", "row"),
        tokenize_task=step("tokenize", "tokens"),
        load_weights=step("weights", "model"),
        evaluate=step("evaluate", "result"),
        finalize=step("finalize", {"passed": True}),
        record_failure=step("failure", None),
    )
    assert stage_a.run_ordered_access(hooks) == {"passed": True}
    assert events == [
        "authenticate",
        "readiness",
        "config",
        "reserve",
        "data",
        "tokenize",
        "weights",
        "evaluate",
        "finalize",
    ]


def test_authentication_failure_prevents_config_data_reservation_and_weights() -> None:
    calls: list[str] = []

    def reject() -> object:
        calls.append("authenticate")
        raise stage_a.StageAAuthenticationError("tampered Stage 0")

    with pytest.raises(stage_a.StageAAuthenticationError, match="tampered Stage 0"):
        stage_a.run_ordered_access(_hooks(authenticate=reject))
    assert calls == ["authenticate"]


def test_config_identity_failure_prevents_one_run_reservation_and_data() -> None:
    calls: list[str] = []

    def reject_config(_authenticated: object) -> object:
        calls.append("config")
        raise stage_a.StageAAuthenticationError("geometry drift")

    with pytest.raises(stage_a.StageAAuthenticationError, match="geometry drift"):
        stage_a.run_ordered_access(
            _hooks(
                authenticate=lambda: calls.append("authenticate") or "auth",
                authenticate_readiness=lambda _auth: calls.append("readiness") or "ready",
                load_config=reject_config,
            )
        )
    assert calls == ["authenticate", "readiness", "config"]


def test_readiness_failure_prevents_config_reservation_data_and_weights() -> None:
    events: list[str] = []

    def reject_readiness(_authenticated: object) -> object:
        events.append("readiness")
        raise stage_a.StageAAuthenticationError("dependency drift")

    with pytest.raises(stage_a.StageAAuthenticationError, match="dependency drift"):
        stage_a.run_ordered_access(
            _hooks(
                authenticate=lambda: events.append("authenticate") or "auth",
                authenticate_readiness=reject_readiness,
            )
        )
    assert events == ["authenticate", "readiness"]


def test_token_authentication_failure_happens_after_reservation_but_before_weights() -> None:
    events: list[str] = []

    def reject_tokens(*_args: object) -> object:
        events.append("tokenize")
        raise stage_a.StageAAuthenticationError("token manifest drift")

    def record_failure(
        _attempt: object,
        error: BaseException,
        _ledger: stage_a.AccessLedger,
    ) -> None:
        events.append(f"failed:{type(error).__name__}")

    hooks = _hooks(
        authenticate=lambda: events.append("authenticate") or "auth",
        authenticate_readiness=lambda _auth: events.append("readiness") or "ready",
        load_config=lambda _auth: events.append("config") or "config",
        reserve_attempt=lambda _auth, _config, _ready: events.append("reserve") or "attempt",
        load_exact_task=lambda _auth: events.append("data") or "row",
        tokenize_task=reject_tokens,
        record_failure=record_failure,
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="token manifest drift"):
        stage_a.run_ordered_access(hooks)
    assert events == [
        "authenticate",
        "readiness",
        "config",
        "reserve",
        "data",
        "tokenize",
        "failed:StageAAuthenticationError",
    ]


def test_access_ledger_authenticates_exact_429_forward_passes_after_evaluation() -> None:
    entered = stage_a.AccessLedger(
        phase="evaluation_entered",
        task_load_entered=True,
        task_row_loaded=True,
        tokenizer_entered=True,
        tokenizer_loaded=True,
        model_weights_entered=True,
        model_weights_loaded=True,
        evaluation_entered=True,
    ).snapshot()
    assert entered["evaluation_returned"] is None
    assert entered["forward_passes"] is None
    assert entered["forward_passes_minimum"] == 0
    assert entered["quality_result_computed"] is None

    returned = stage_a._expected_access_snapshot("evaluation_returned")
    assert stage_a.FROZEN_STAGE_A_FORWARD_PASSES == 429
    assert returned["forward_passes"] == 429
    assert returned["forward_passes_minimum"] == 429
    assert returned["quality_result_computed"] is True


@pytest.mark.parametrize(
    ("failing_hook", "expected_phase"),
    [
        ("load_exact_task", "task_load_entered"),
        ("tokenize_task", "tokenizer_entered"),
        ("load_weights", "model_weights_entered"),
        ("evaluate", "evaluation_entered"),
        ("finalize", "finalization_entered"),
    ],
)
def test_every_access_hook_failure_reports_the_conservative_monotonic_ledger(
    failing_hook: str,
    expected_phase: str,
) -> None:
    transitions: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def checkpoint(receipt: object, ledger: stage_a.AccessLedger) -> object:
        transitions.append(copy.deepcopy(ledger.snapshot()))
        return receipt

    def operation(name: str, result: object):
        def callback(*_args: object) -> object:
            if name == failing_hook:
                raise RuntimeError(f"synthetic {name} interruption")
            return result

        return callback

    hooks = _hooks(
        authenticate=lambda: "auth",
        authenticate_readiness=lambda _auth: "ready",
        load_config=lambda _auth: "config",
        reserve_attempt=lambda *_args: "attempt",
        load_exact_task=operation("load_exact_task", "row"),
        tokenize_task=operation("tokenize_task", "tokens"),
        load_weights=operation("load_weights", "model"),
        evaluate=operation("evaluate", "result"),
        record_access_transition=checkpoint,
        record_evaluation_returned=checkpoint,
        finalize=operation("finalize", {"passed": True}),
        record_failure=lambda _receipt, _error, ledger: failures.append(
            copy.deepcopy(ledger.snapshot())
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        stage_a.run_ordered_access(hooks)

    assert failures == [stage_a._expected_access_snapshot(expected_phase)]
    phase_indices = [
        stage_a.ACCESS_PHASE_ORDER.index(str(snapshot["phase"])) for snapshot in transitions
    ]
    assert phase_indices == sorted(set(phase_indices))
    if expected_phase == "evaluation_entered":
        assert failures[0]["forward_passes"] is None
        assert failures[0]["forward_passes_minimum"] == 0
    if expected_phase == "finalization_entered":
        assert failures[0]["forward_passes"] == 429
        assert failures[0]["forward_passes_minimum"] == 429


def test_runtime_dependency_manifest_is_exact_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(
        stage_a.importlib,
        "import_module",
        lambda package: imported.append(package) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        stage_a.importlib.metadata,
        "version",
        lambda package: stage_a.EXPECTED_RUNTIME_PACKAGES[package],
    )
    versions = stage_a._runtime_dependency_versions()
    assert versions == stage_a.EXPECTED_RUNTIME_PACKAGES
    assert imported == list(stage_a.RUNTIME_PACKAGE_IMPORTS.values())
    manifest = (json.dumps(versions, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert hashlib.sha256(manifest).hexdigest() == (
        stage_a.EXPECTED_RUNTIME_PACKAGE_MANIFEST_SHA256
    )

    def missing(package: str) -> str:
        if package == "datasets":
            raise stage_a.importlib.metadata.PackageNotFoundError(package)
        return stage_a.EXPECTED_RUNTIME_PACKAGES[package]

    monkeypatch.setattr(stage_a.importlib.metadata, "version", missing)
    with pytest.raises(stage_a.StageAAuthenticationError, match="missing: datasets"):
        stage_a._runtime_dependency_versions()

    monkeypatch.setattr(
        stage_a.importlib.metadata,
        "version",
        lambda package: (
            "0.0.0" if package == "fsspec" else stage_a.EXPECTED_RUNTIME_PACKAGES[package]
        ),
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="fsspec"):
        stage_a._runtime_dependency_versions()


def _patch_readiness_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_dataset: object = lambda: None,
) -> None:
    monkeypatch.setattr(
        stage_a,
        "_runtime_dependency_versions",
        lambda: dict(stage_a.EXPECTED_RUNTIME_PACKAGES),
    )
    monkeypatch.setattr(
        stage_a.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(load_dataset=load_dataset) if name == "datasets" else _unused()
        ),
    )
    monkeypatch.setattr(stage_a.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stage_a.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(stage_a.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        stage_a,
        "_authenticate_cached_dataset_resources",
        lambda _module: {"resources": {}, "local_cache_only": True},
    )
    monkeypatch.setattr(
        stage_a,
        "_authenticate_cached_model_resources",
        lambda: {"resources": {}, "local_cache_only": True},
    )


def test_runtime_readiness_receipt_is_canonical_and_preseal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_readiness_prerequisites(monkeypatch)
    readiness = stage_a.authenticate_runtime_readiness(device_name="cuda")
    assert readiness.receipt["packages"] == stage_a.EXPECTED_RUNTIME_PACKAGES
    assert readiness.receipt["package_manifest_sha256"] == (
        stage_a.EXPECTED_RUNTIME_PACKAGE_MANIFEST_SHA256
    )
    assert readiness.receipt["datasets_api"]["load_dataset_called"] is False
    assert readiness.receipt["authenticated_before_one_run_seal"] is True
    assert readiness.canonical_sha256 == stage_a._canonical_sha256(readiness.receipt)


def test_missing_datasets_api_fails_before_reserve_or_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_readiness_prerequisites(monkeypatch, load_dataset=None)
    events: list[str] = []

    def readiness(_authenticated: object) -> object:
        events.append("readiness")
        return stage_a.authenticate_runtime_readiness(device_name="cuda")

    with pytest.raises(stage_a.StageAAuthenticationError, match="non-callable"):
        stage_a.run_ordered_access(
            _hooks(
                authenticate=lambda: events.append("authenticate") or "auth",
                authenticate_readiness=readiness,
            )
        )
    assert events == ["authenticate", "readiness"]


def test_cuda_and_bf16_fail_before_cache_reserve_or_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_readiness_prerequisites(monkeypatch)
    monkeypatch.setattr(stage_a.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        stage_a,
        "_authenticate_cached_dataset_resources",
        _unused,
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="CUDA device"):
        stage_a.authenticate_runtime_readiness(device_name="cuda")

    monkeypatch.setattr(stage_a.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stage_a.torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(stage_a.StageAAuthenticationError, match="BF16 support"):
        stage_a.authenticate_runtime_readiness(device_name="cuda")


def test_weight_cache_failure_prevents_reservation_and_task_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_readiness_prerequisites(monkeypatch)
    monkeypatch.setattr(
        stage_a,
        "_authenticate_cached_model_resources",
        lambda: (_ for _ in ()).throw(stage_a.StageAAuthenticationError("weight cache drift")),
    )
    events: list[str] = []

    def readiness(_authenticated: object) -> object:
        events.append("readiness")
        return stage_a.authenticate_runtime_readiness(device_name="cuda")

    with pytest.raises(stage_a.StageAAuthenticationError, match="weight cache drift"):
        stage_a.run_ordered_access(
            _hooks(
                authenticate=lambda: events.append("authenticate") or "auth",
                authenticate_readiness=readiness,
            )
        )
    assert events == ["authenticate", "readiness"]


def test_model_cache_resources_are_hashed_without_loading_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / stage_a.MODEL_REVISION
    snapshot.mkdir()
    resource = snapshot / stage_a.MODEL_WEIGHT_RESOURCE_FILENAME
    resource.write_bytes(b"frozen-weight-bytes")
    index = snapshot / stage_a.MODEL_WEIGHT_INDEX_FILENAME
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": stage_a.MODEL_WEIGHT_INDEX_TENSOR_BYTES},
                "weight_map": {"synthetic.weight": resource.name},
            }
        ),
        encoding="utf-8",
    )
    expected = {
        stage_a.MODEL_WEIGHT_INDEX_FILENAME: {
            "size_bytes": index.stat().st_size,
            "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
        stage_a.MODEL_WEIGHT_RESOURCE_FILENAME: {
            "size_bytes": resource.stat().st_size,
            "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        },
    }
    monkeypatch.setattr(stage_a, "MODEL_CACHE_RESOURCES", expected)
    monkeypatch.setattr(
        stage_a,
        "_resolve_cached_model_resource",
        lambda filename: snapshot / filename,
    )
    receipt = stage_a._authenticate_cached_model_resources()
    row = receipt["resources"][stage_a.MODEL_WEIGHT_RESOURCE_FILENAME]
    assert row["bytes_hashed_without_parsing_or_tensor_loading"] is True
    assert receipt["tensors_loaded"] is False

    resource.write_bytes(b"x" * resource.stat().st_size)
    with pytest.raises(stage_a.StageAAuthenticationError, match="SHA-256 drifted"):
        stage_a._authenticate_cached_model_resources()


@pytest.mark.parametrize(
    "alternate",
    [
        "model.safetensors",
        "pytorch_model.bin",
        "synthetic.safetensors.index.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "ADAPTER_CONFIG.JSON",
        "added_tokens.json",
        "SPECIAL_TOKENS_MAP.JSON",
        "tokenizer.model",
        "chat_templates",
    ],
)
def test_model_snapshot_rejects_every_unpinned_loader_resource(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    alternate: str,
) -> None:
    snapshot = tmp_path / stage_a.MODEL_REVISION
    snapshot.mkdir()
    shard = snapshot / stage_a.MODEL_WEIGHT_RESOURCE_FILENAME
    shard.write_bytes(b"frozen-weight-bytes")
    index = snapshot / stage_a.MODEL_WEIGHT_INDEX_FILENAME
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": stage_a.MODEL_WEIGHT_INDEX_TENSOR_BYTES},
                "weight_map": {"synthetic.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stage_a,
        "MODEL_CACHE_RESOURCES",
        {
            index.name: {
                "size_bytes": index.stat().st_size,
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            },
            shard.name: {
                "size_bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            },
        },
    )
    monkeypatch.setattr(
        stage_a,
        "_resolve_cached_model_resource",
        lambda filename: snapshot / filename,
    )
    alternate_path = snapshot / alternate
    if alternate == "chat_templates":
        alternate_path.mkdir()
    else:
        alternate_path.write_bytes(b"unpinned")
    with pytest.raises(
        stage_a.StageAAuthenticationError,
        match="weight file set drifted|tokenizer-affecting",
    ):
        stage_a._authenticate_cached_model_resources()


def test_model_weight_loader_uses_only_the_authenticated_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transformers

    snapshot = tmp_path / stage_a.MODEL_REVISION
    snapshot.mkdir()
    resource_manifest = "a" * 64
    receipt = {
        "snapshot_resource_id": f"{stage_a.MODEL_ID}@{stage_a.MODEL_REVISION}",
        "resource_manifest_sha256": resource_manifest,
        "alternate_or_unpinned_weight_files_absent": True,
    }
    authentication_calls: list[Path] = []

    def authenticate() -> tuple[Path, dict[str, object]]:
        authentication_calls.append(snapshot)
        return snapshot, copy.deepcopy(receipt)

    geometry = {"frozen": True}
    calls: list[tuple[object, dict[str, object]]] = []

    class SyntheticModel:
        config = object()

        def to(self, _device: torch.device) -> SyntheticModel:
            return self

        def eval(self) -> None:
            return None

    class SyntheticFactory:
        @classmethod
        def from_pretrained(cls, source: object, **kwargs: object) -> SyntheticModel:
            calls.append((source, kwargs))
            return SyntheticModel()

    monkeypatch.setattr(stage_a, "_authenticate_model_snapshot", authenticate)
    monkeypatch.setattr(stage_a, "_select_cuda_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(stage_a, "_validate_model_geometry", lambda _config: geometry)
    monkeypatch.setattr(transformers, "Qwen3_5ForCausalLM", SyntheticFactory)
    configuration = stage_a.ModelConfiguration(
        config=object(),
        identity={
            "geometry": geometry,
            "authenticated_snapshot_resource_id": receipt["snapshot_resource_id"],
            "model_cache_resource_manifest_sha256": resource_manifest,
        },
    )
    stage_a.load_model_weights(
        configuration,
        SimpleNamespace(),
        device_name="cuda",
        local_files_only=True,
    )
    assert authentication_calls == [snapshot, snapshot]
    assert len(calls) == 1
    source, kwargs = calls[0]
    assert source == str(snapshot)
    assert kwargs["revision"] == stage_a.MODEL_REVISION
    assert kwargs["local_files_only"] is True
    assert kwargs["use_safetensors"] is True
    with pytest.raises(stage_a.StageAAuthenticationError, match="local-files-only"):
        stage_a.load_model_weights(
            configuration,
            SimpleNamespace(),
            device_name="cuda",
            local_files_only=False,
        )


def test_dataset_cache_resources_are_hashed_without_decoding_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = tmp_path / "revision"
    revision.mkdir()
    info = revision / "dataset_info.json"
    arrow = revision / "mbpp-train.arrow"
    info.write_bytes(b"metadata")
    arrow.write_bytes(b"arrow-bytes")
    resources = {
        info.name: {
            "size_bytes": info.stat().st_size,
            "sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
        },
        arrow.name: {
            "size_bytes": arrow.stat().st_size,
            "sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
        },
    }
    monkeypatch.setattr(stage_a, "DATASET_CACHE_RESOURCES", resources)
    monkeypatch.setattr(
        stage_a,
        "_resolve_cached_dataset_revision",
        lambda _module: revision,
    )
    receipt = stage_a._authenticate_cached_dataset_resources(object())
    assert receipt["dataset_rows_decoded_or_iterated"] is False
    assert all(
        row["bytes_hashed_without_decoding_or_iteration"] is True
        for row in receipt["resources"].values()
    )

    arrow.write_bytes(b"arrow-drift")
    with pytest.raises(stage_a.StageAAuthenticationError, match="SHA-256 drifted"):
        stage_a._authenticate_cached_dataset_resources(object())


def test_local_arrow_loader_never_calls_load_dataset_and_requires_unique_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyarrow

    arrow_path = tmp_path / "mbpp-train.arrow"
    table = pyarrow.table(
        {
            "task_id": [665, 666, 667],
            "text": ["not selected", "target", "not selected"],
            "code": ["a", "b", "c"],
        }
    )
    with (
        arrow_path.open("wb") as sink,
        pyarrow.ipc.new_stream(sink, table.schema) as writer,
    ):
        writer.write_table(table)

    real_import = stage_a.importlib.import_module

    def guarded_import(name: str) -> object:
        if name == "datasets":
            raise AssertionError("datasets.load_dataset path must not be used")
        return real_import(name)

    monkeypatch.setattr(stage_a.importlib, "import_module", guarded_import)
    row = stage_a._select_exact_task_from_arrow(arrow_path, task_id=666)
    assert row["task_id"] == 666
    assert row["text"] == "target"
    with pytest.raises(stage_a.StageAAuthenticationError, match="non-unique task 999"):
        stage_a._select_exact_task_from_arrow(arrow_path, task_id=999)

    duplicate_path = tmp_path / "duplicate.arrow"
    duplicate = pyarrow.table(
        {
            "task_id": [666, 666],
            "text": ["first", "second"],
            "code": ["a", "b"],
        }
    )
    with (
        duplicate_path.open("wb") as sink,
        pyarrow.ipc.new_stream(sink, duplicate.schema) as writer,
    ):
        writer.write_table(duplicate)
    with pytest.raises(stage_a.StageAAuthenticationError, match="non-unique task 666"):
        stage_a._select_exact_task_from_arrow(duplicate_path, task_id=666)


def test_authenticated_arrow_resource_is_local_and_hash_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = tmp_path / "revision"
    revision.mkdir()
    arrow = revision / "mbpp-train.arrow"
    arrow.write_bytes(b"local-arrow")
    expected = {
        "size_bytes": arrow.stat().st_size,
        "sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
    }
    resources = copy.deepcopy(stage_a.DATASET_CACHE_RESOURCES)
    resources["mbpp-train.arrow"] = expected
    load_dataset_calls: list[str] = []
    datasets_module = SimpleNamespace(
        load_dataset=lambda *_args, **_kwargs: load_dataset_calls.append("called")
    )
    monkeypatch.setattr(stage_a, "DATASET_CACHE_RESOURCES", resources)
    monkeypatch.setattr(
        stage_a.importlib,
        "import_module",
        lambda name: datasets_module if name == "datasets" else _unused(),
    )
    monkeypatch.setattr(
        stage_a,
        "_resolve_cached_dataset_revision",
        lambda module: revision if module is datasets_module else _unused(),
    )
    assert stage_a._authenticated_mbpp_train_arrow() == arrow
    assert load_dataset_calls == []

    arrow.write_bytes(b"local-drift")
    with pytest.raises(stage_a.StageAAuthenticationError, match="resource drifted"):
        stage_a._authenticated_mbpp_train_arrow()


def _anchor_artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "recurquant_rht_cqer32_stage_a_screen",
        "canonical_evidence_sha256": stage_a.EXPERIMENT009_STAGE_A_CANONICAL_SHA256,
        "evidence": {
            "screening_only": True,
            "stage_a_gate": {"passed": True},
            "dataset": {
                "manifest": copy.deepcopy(stage_a.EXPECTED_DATASET_MANIFEST),
                "manifest_sha256": stage_a.EXPERIMENT009_MANIFEST_SHA256,
                "token_manifest": copy.deepcopy(stage_a.EXPECTED_TOKEN_MANIFEST),
                "identity": copy.deepcopy(stage_a.EXPECTED_TASK_IDENTITY),
            },
            "model": {
                "id": stage_a.MODEL_ID,
                "revision": stage_a.MODEL_REVISION,
                "dtype": str(stage_a.MODEL_DTYPE),
            },
            "selector_artifacts": {
                "authenticated": True,
                "selector_file_sha256": stage_a.SELECTOR_FILE_SHA256,
                "loss_selector_file_sha256": stage_a.LOSS_SELECTOR_FILE_SHA256,
                "selector_canonical_evidence_sha256": (stage_a.SELECTOR_CANONICAL_SHA256),
                "loss_selector_canonical_evidence_sha256": (stage_a.LOSS_SELECTOR_CANONICAL_SHA256),
                "quota_sum": 1976,
            },
        },
    }


def test_experiment010_administrative_null_and_raw_receipt_are_semantic_anchors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (repo_root / stage_a.EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    semantics = stage_a._validate_experiment010_administrative_null(artifact)
    assert semantics["scientific_result_available"] is False
    assert semantics["original_seal_commit"] == stage_a.EXPERIMENT010_SEAL_COMMIT

    tampered = copy.deepcopy(artifact)
    tampered["evidence"]["scientific_result_available"] = True
    with pytest.raises(stage_a.StageAAuthenticationError, match="classification drifted"):
        stage_a._validate_experiment010_administrative_null(tampered)

    receipt = json.loads(
        (repo_root / stage_a.EXPERIMENT010_ATTEMPT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    stage_a._validate_experiment010_failed_receipt(receipt)
    receipt["quality_aggregate_exposed"] = True
    with pytest.raises(stage_a.StageAAuthenticationError, match="semantics drifted"):
        stage_a._validate_experiment010_failed_receipt(receipt)


def test_experiment010_result_must_remain_absent(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in (
        stage_a.EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH,
        stage_a.EXPERIMENT010_ATTEMPT_RELATIVE_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo_root / relative).read_bytes())
    old_result = tmp_path / stage_a.EXPERIMENT010_OUTPUT_RELATIVE_PATH
    old_result.parent.mkdir(parents=True, exist_ok=True)
    old_result.write_text("forbidden", encoding="utf-8")
    with pytest.raises(stage_a.StageAAuthenticationError, match="result must remain absent"):
        stage_a.authenticate_experiment010_administrative_null(tmp_path)


def test_old_marker_is_required_while_new_marker_prevents_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def valid_history(_repo: Path, *arguments: str) -> str:
        if arguments[:2] == ("rev-list", "--all"):
            return f"{stage_a.EXPERIMENT010_H0_COMMIT}\n{stage_a.EXPERIMENT010_SEAL_COMMIT}"
        if arguments == (
            "show",
            "-s",
            "--format=%P",
            stage_a.EXPERIMENT010_SEAL_COMMIT,
        ):
            return stage_a.EXPERIMENT010_H0_COMMIT
        if arguments in {
            (
                "show",
                "-s",
                "--format=%T",
                stage_a.EXPERIMENT010_SEAL_COMMIT,
            ),
            ("show", "-s", "--format=%T", stage_a.EXPERIMENT010_H0_COMMIT),
        }:
            return stage_a.EXPERIMENT010_SEAL_TREE
        if arguments == (
            "show",
            "-s",
            "--format=%B",
            stage_a.EXPERIMENT010_SEAL_COMMIT,
        ):
            return stage_a.EXPERIMENT010_ONE_RUN_MARKER
        if arguments[:2] == ("log", "--all"):
            return stage_a.EXPERIMENT010_ONE_RUN_MARKER
        raise AssertionError(arguments)

    monkeypatch.setattr(stage_a, "_git", valid_history)
    stage_a._authenticate_experiment010_git_history(tmp_path)
    stage_a._assert_no_prior_stage_a_seal(tmp_path)

    def missing_old_marker(repo: Path, *arguments: str) -> str:
        value = valid_history(repo, *arguments)
        if arguments == (
            "show",
            "-s",
            "--format=%B",
            stage_a.EXPERIMENT010_SEAL_COMMIT,
        ):
            return "missing"
        return value

    monkeypatch.setattr(stage_a, "_git", missing_old_marker)
    with pytest.raises(stage_a.StageAAuthenticationError, match="marker is missing"):
        stage_a._authenticate_experiment010_git_history(tmp_path)

    def new_marker_present(repo: Path, *arguments: str) -> str:
        value = valid_history(repo, *arguments)
        if arguments[:2] == ("log", "--all"):
            return f"{value}\n{stage_a.ONE_RUN_MARKER}"
        return value

    monkeypatch.setattr(stage_a, "_git", new_marker_present)
    with pytest.raises(stage_a.StageAAuthenticationError, match="already present"):
        stage_a._assert_no_prior_stage_a_seal(tmp_path)


def test_anchor_validation_copies_only_authenticated_task_666_manifests() -> None:
    result = stage_a._validate_anchor_payload(_anchor_artifact())
    assert result["dataset_manifest"] == stage_a.EXPECTED_DATASET_MANIFEST
    assert result["token_manifest"] == stage_a.EXPECTED_TOKEN_MANIFEST
    assert result["task_identity"] == stage_a.EXPECTED_TASK_IDENTITY


def test_anchor_validation_rejects_task_or_token_manifest_drift() -> None:
    artifact = _anchor_artifact()
    artifact["evidence"]["dataset"]["token_manifest"][0]["task_id"] = 667
    with pytest.raises(stage_a.StageAAuthenticationError, match="token manifest drifted"):
        stage_a._validate_anchor_payload(artifact)


def test_stage0_incomplete_report_fails_before_any_quality_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = SimpleNamespace(
        verify_production_stage0=lambda *_args, **_kwargs: {
            "status": "verifier_self_test_pass",
            "experiment_stage0_complete": False,
            "quality_data_accessed": False,
            "protected_mbpp_window_accessed": False,
        }
    )
    monkeypatch.setattr(stage_a, "_script_module", lambda _name: verifier)
    with pytest.raises(stage_a.StageAAuthenticationError, match="complete synthetic-only"):
        stage_a.authenticate_stage0(tmp_path / "stage0.pt", None)


def test_stage0_artifact_head_must_equal_current_repository_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = SimpleNamespace(
        verify_production_stage0=lambda *_args, **_kwargs: {
            "status": "production_stage0_pass",
            "experiment_stage0_complete": True,
            "quality_data_accessed": False,
            "protected_mbpp_window_accessed": False,
            "repository_commit": "old-head",
        }
    )
    monkeypatch.setattr(stage_a, "_script_module", lambda _name: verifier)
    with pytest.raises(stage_a.StageAAuthenticationError, match="does not equal"):
        stage_a.authenticate_stage0(
            tmp_path / "stage0.pt",
            None,
            expected_repo_head="current-head",
        )


def test_one_run_receipt_is_exclusive_and_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    first = b'{"attempt":1}'
    stage_a._exclusive_write(path, first)
    with pytest.raises(stage_a.StageAAuthenticationError, match="refusing to overwrite"):
        stage_a._exclusive_write(path, b'{"attempt":2}')
    assert path.read_bytes() == first


def test_prepared_receipt_promotion_rejects_on_disk_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt.json"
    prepared = {
        "schema": "recurquant.experiment011.stage-a-attempt.v1",
        "status": "prepared_before_head_cas",
    }
    path.write_bytes(stage_a.canonical_json_bytes(prepared))
    path.write_bytes(b'{"status":"tampered"}\n')
    with pytest.raises(stage_a.StageAAuthenticationError, match="bytes drifted"):
        stage_a._promote_prepared_attempt_receipt(path, prepared)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "tampered"


def test_reservation_reauthenticates_freshness_before_creating_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight = SimpleNamespace(repository_start={"commit": "0" * 40})
    configuration = stage_a.ModelConfiguration(config=object(), identity={})
    receipt = {"accelerator": {"requested_device": "cuda"}}
    readiness = stage_a.RuntimeReadiness(
        receipt=receipt,
        canonical_sha256=stage_a._canonical_sha256(receipt),
    )
    events: list[str] = []

    def reject(*_args: object) -> object:
        events.append("freshness")
        raise stage_a.StageAAuthenticationError("Stage-0 drift")

    monkeypatch.setattr(stage_a, "_assert_preseal_freshness", reject)
    monkeypatch.setattr(stage_a, "_commit_tree", _unused)
    with pytest.raises(stage_a.StageAAuthenticationError, match="Stage-0 drift"):
        stage_a.reserve_one_run(preflight, configuration, readiness)
    assert events == ["freshness"]
    assert not list(tmp_path.iterdir())


def test_model_configuration_authentication_is_mandatorily_local_only() -> None:
    with pytest.raises(stage_a.StageAAuthenticationError, match="local-files-only"):
        stage_a.load_and_authenticate_config(
            SimpleNamespace(),
            local_files_only=False,
        )


def _init_synthetic_git_repository(path: Path) -> tuple[str, object]:
    path.mkdir()

    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    git("init")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")
    return git("rev-parse", "HEAD"), git


def test_all_git_subprocesses_scrub_hostile_routing_object_and_config_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    h0, _git = _init_synthetic_git_repository(repo)
    decoy = tmp_path / "decoy"
    _decoy_head, decoy_git = _init_synthetic_git_repository(decoy)
    decoy_git_dir = decoy_git("rev-parse", "--absolute-git-dir")
    decoy_objects = Path(decoy_git_dir) / "objects"
    hostile = {
        "GIT_DIR": decoy_git_dir,
        "git_work_tree": str(decoy),
        "GIT_INDEX_FILE": str(decoy / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(decoy_objects),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(decoy_objects),
        "GIT_COMMON_DIR": decoy_git_dir,
        "GIT_NAMESPACE": "hostile",
        "GIT_REPLACE_REF_BASE": "refs/hostile/",
        "GIT_EXEC_PATH": str(tmp_path / "hostile-exec-path"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": str(decoy),
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)

    state = stage_a._repository_state(repo)
    assert state["commit"] == h0
    assert state["git_identity"]["top_level_matches_repo_root"] is True
    environment = stage_a._sanitized_git_environment()
    safe_git_keys = {
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_CONFIG_KEY_1",
        "GIT_CONFIG_VALUE_1",
        "GIT_CONFIG_KEY_2",
        "GIT_CONFIG_VALUE_2",
        "GIT_CONFIG_KEY_3",
        "GIT_CONFIG_VALUE_3",
    }
    assert {key for key in environment if key.upper().startswith("GIT_")} == safe_git_keys
    assert environment["GIT_CONFIG_VALUE_0"] == stage_a.os.devnull
    assert environment["GIT_CONFIG_VALUE_1"] == "false"
    assert environment["GIT_CONFIG_VALUE_2"] == "false"
    assert environment["GIT_CONFIG_VALUE_3"] == "false"


def test_git_identity_rejects_alternates_replacements_grafts_and_shallow_history(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    h0, git = _init_synthetic_git_repository(repo)
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    objects = git_dir / "objects"
    decoy_objects = tmp_path / "decoy-objects"
    decoy_objects.mkdir()

    alternates = objects / "info" / "alternates"
    alternates.write_text(str(decoy_objects), encoding="utf-8")
    with pytest.raises(stage_a.StageAAuthenticationError, match="alternate object"):
        stage_a._assert_git_repository_identity(repo)
    alternates.unlink()

    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "second")
    second = git("rev-parse", "HEAD")
    git("replace", h0, second)
    with pytest.raises(stage_a.StageAAuthenticationError, match="replacement refs"):
        stage_a._assert_git_repository_identity(repo)
    git("replace", "-d", h0)

    grafts = git_dir / "info" / "grafts"
    grafts.write_text(f"{second} {h0}\n", encoding="utf-8")
    with pytest.raises(stage_a.StageAAuthenticationError, match="grafts"):
        stage_a._assert_git_repository_identity(repo)
    grafts.unlink()

    shallow = git_dir / "shallow"
    shallow.write_text(f"{second}\n", encoding="utf-8")
    with pytest.raises(stage_a.StageAAuthenticationError, match="shallow"):
        stage_a._assert_git_repository_identity(repo)


def test_source_head_authentication_hashes_raw_bytes_without_clean_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tracked.txt").write_bytes(b"changed worktree bytes\n")
    head_blob = "a" * 40
    calls: list[tuple[str, ...]] = []

    def synthetic_git(_repo: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments[:2] == ("ls-tree", "HEAD"):
            return f"100644 blob {head_blob}\ttracked.txt"
        if arguments[:2] == ("hash-object", "--no-filters"):
            return "b" * 40
        if arguments and arguments[0] == "hash-object":
            return head_blob
        raise AssertionError(arguments)

    monkeypatch.setattr(stage_a, "SOURCE_FILES", ("tracked.txt",))
    monkeypatch.setattr(stage_a, "_git", synthetic_git)
    with pytest.raises(stage_a.StageAAuthenticationError, match="differ from HEAD"):
        stage_a._assert_sources_match_head(tmp_path)
    assert ("hash-object", "--no-filters", "--", "tracked.txt") in calls


def test_empty_commit_seal_is_durable_after_local_receipt_deletion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    git("init")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (repo / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")
    h0 = git("rev-parse", "HEAD")
    git("config", "--unset", "user.name")
    git("config", "--unset", "user.email")
    attempt_path = tmp_path / "attempt.json"
    preflight = SimpleNamespace(
        repo_root=repo,
        repository_start={
            "commit": h0,
            "git_identity": stage_a._assert_git_repository_identity(repo),
        },
        source_hashes_start={"tracked.txt": hashlib.sha256(b"frozen\n").hexdigest()},
        stage0={
            "artifact_file_sha256": "4" * 64,
            "sidecar_file_sha256": "5" * 64,
        },
        experiment010_admin_null={
            "scientific_result_available": False,
            "canonical_provenance_sha256": "6" * 64,
        },
        attempt_path=attempt_path,
    )
    configuration = stage_a.ModelConfiguration(
        config=object(),
        identity={"id": stage_a.MODEL_ID, "revision": stage_a.MODEL_REVISION},
    )
    monkeypatch.setattr(stage_a, "_assert_sources_match_head", lambda _repo: None)
    monkeypatch.setattr(
        stage_a,
        "_source_hashes",
        lambda _repo: dict(preflight.source_hashes_start),
    )
    monkeypatch.setattr(
        stage_a,
        "_assert_preseal_freshness",
        lambda *_args: {key: True for key in stage_a.PRESEAL_FRESHNESS_KEYS},
    )
    readiness_receipt = {
        "schema": "recurquant.experiment011.runtime-readiness.v1",
        "accelerator": {"requested_device": "cuda"},
    }
    readiness = stage_a.RuntimeReadiness(
        receipt=readiness_receipt,
        canonical_sha256=stage_a._canonical_sha256(readiness_receipt),
    )
    reservation = stage_a.reserve_one_run(preflight, configuration, readiness)
    assert git("rev-parse", "HEAD") == reservation.seal_commit
    assert git("rev-parse", f"{reservation.seal_commit}^") == h0
    assert git("rev-parse", f"{reservation.seal_commit}^{{tree}}") == git(
        "rev-parse", f"{h0}^{{tree}}"
    )
    assert git("show", "-s", "--format=%an <%ae>", reservation.seal_commit) == (
        "Synthetic Test <synthetic@example.invalid>"
    )
    assert reservation.receipt["schema"] == "recurquant.experiment011.stage-a-attempt.v1"
    assert reservation.receipt["runtime_readiness"] == stage_a._readiness_bundle(readiness)
    assert reservation.receipt["experiment010_administrative_null"] == (
        preflight.experiment010_admin_null
    )
    seal_message = git("show", "-s", "--format=%B", reservation.seal_commit)
    assert stage_a.ONE_RUN_MARKER in seal_message
    assert readiness.canonical_sha256 in seal_message
    attempt_path.unlink()
    with pytest.raises(stage_a.StageAAuthenticationError, match="receipt is missing"):
        stage_a._validate_one_run_seal(
            preflight,
            reservation,
            require_receipt=True,
        )
    git("reset", "--hard", h0)
    assert git("rev-parse", "HEAD") == h0
    with pytest.raises(stage_a.StageAAuthenticationError, match="already present"):
        stage_a._assert_no_prior_stage_a_seal(repo)


def _synthetic_two_phase_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    Path,
    object,
    stage_a.ModelConfiguration,
    stage_a.RuntimeReadiness,
    object,
]:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    git("init")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (repo / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")
    h0 = git("rev-parse", "HEAD")
    git_identity = stage_a._assert_git_repository_identity(repo)
    preflight = SimpleNamespace(
        repo_root=repo,
        repository_start={"commit": h0, "git_identity": git_identity},
        source_hashes_start={"tracked.txt": hashlib.sha256(b"frozen\n").hexdigest()},
        stage0={
            "artifact_file_sha256": "4" * 64,
            "sidecar_file_sha256": "5" * 64,
        },
        experiment010_admin_null={
            "scientific_result_available": False,
            "canonical_provenance_sha256": "6" * 64,
        },
        attempt_path=tmp_path / "attempt.json",
    )
    configuration = stage_a.ModelConfiguration(
        config=object(),
        identity={"id": stage_a.MODEL_ID, "revision": stage_a.MODEL_REVISION},
    )
    readiness_receipt = {
        "schema": "recurquant.experiment011.runtime-readiness.v1",
        "accelerator": {"requested_device": "cuda"},
    }
    readiness = stage_a.RuntimeReadiness(
        receipt=readiness_receipt,
        canonical_sha256=stage_a._canonical_sha256(readiness_receipt),
    )
    monkeypatch.setattr(stage_a, "_assert_sources_match_head", lambda _repo: None)
    monkeypatch.setattr(
        stage_a,
        "_source_hashes",
        lambda _repo: dict(preflight.source_hashes_start),
    )
    monkeypatch.setattr(
        stage_a,
        "_assert_preseal_freshness",
        lambda *_args: {key: True for key in stage_a.PRESEAL_FRESHNESS_KEYS},
    )
    return repo, preflight, configuration, readiness, git


def test_one_run_cas_disables_reference_transaction_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, preflight, configuration, readiness, _git = _synthetic_two_phase_reservation(
        monkeypatch,
        tmp_path,
    )
    git_dir = Path(_git("rev-parse", "--absolute-git-dir"))
    marker = tmp_path / "reference-hook-ran.txt"
    hook = git_dir / "hooks" / "reference-transaction"
    hook.write_text(
        f"#!/bin/sh\nprintf invoked > '{marker.as_posix()}'\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    _git("status", "--porcelain", "--untracked-files=all")
    stage_a.reserve_one_run(preflight, configuration, readiness)
    assert not marker.exists()


def test_post_cas_promotion_crash_leaves_prepared_consumed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, preflight, configuration, readiness, git = _synthetic_two_phase_reservation(
        monkeypatch,
        tmp_path,
    )
    h0 = git("rev-parse", "HEAD")

    def crash_after_cas(path: Path, prepared: object) -> object:
        del path, prepared
        assert git("rev-parse", "HEAD") != h0
        raise KeyboardInterrupt("synthetic post-CAS interruption")

    monkeypatch.setattr(
        stage_a,
        "_promote_prepared_attempt_receipt",
        crash_after_cas,
    )
    with pytest.raises(KeyboardInterrupt, match="post-CAS interruption"):
        stage_a.reserve_one_run(preflight, configuration, readiness)

    seal = git("rev-parse", "HEAD")
    assert seal != h0
    seal_message = git("show", "-s", "--format=%B", seal)
    assert stage_a.ONE_RUN_MARKER in seal_message
    seal_payload = json.loads(seal_message.split("\n\n", 2)[2])
    assert seal_payload["attempt_number"] == 1
    assert seal_payload["completed_task_ids"] == []
    assert seal_payload["quality_data_accessed"] is False
    assert seal_payload["task_row_loaded"] is False
    assert seal_payload["tokenizer_loaded"] is False
    assert seal_payload["model_weights_loaded"] is False
    assert seal_payload["forward_passes"] == 0
    assert seal_payload["quality_aggregate_exposed"] is False
    assert seal_payload["rerun_automatically_authorized"] is False
    assert seal_payload["preseal_freshness"] == {
        key: True for key in stage_a.PRESEAL_FRESHNESS_KEYS
    }
    assert seal_payload["attempt_path"] == stage_a.ATTEMPT_RELATIVE_PATH
    assert seal_payload["output_path"] == stage_a.OUTPUT_RELATIVE_PATH
    assert seal_payload["claim_boundary"] == stage_a.CLAIM_BOUNDARY
    receipt = json.loads(preflight.attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "prepared_before_head_cas"
    assert receipt["one_run_seal_commit"] == seal
    assert receipt["completed_task_ids"] == []
    assert receipt["quality_data_accessed"] is False
    assert receipt["task_row_loaded"] is False
    assert receipt["model_weights_loaded"] is False
    assert receipt["quality_aggregate_exposed"] is False
    with pytest.raises(stage_a.StageAAuthenticationError, match="HEAD changed"):
        stage_a.reserve_one_run(preflight, configuration, readiness)
    assert git("rev-parse", "HEAD") == seal
    assert repo.is_dir()


def test_pre_cas_prepared_receipt_tampering_prevents_head_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _repo, preflight, configuration, readiness, git = _synthetic_two_phase_reservation(
        monkeypatch,
        tmp_path,
    )
    h0 = git("rev-parse", "HEAD")
    real_exclusive_write = stage_a._exclusive_write

    def write_then_tamper(path: Path, payload: bytes) -> None:
        real_exclusive_write(path, payload)
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["quality_data_accessed"] = True
        path.write_bytes(stage_a.canonical_json_bytes(tampered))

    monkeypatch.setattr(stage_a, "_exclusive_write", write_then_tamper)
    with pytest.raises(
        stage_a.StageAAuthenticationError,
        match="prepared one-run receipt bytes drifted",
    ):
        stage_a.reserve_one_run(preflight, configuration, readiness)
    assert git("rev-parse", "HEAD") == h0
    receipt = json.loads(preflight.attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "prepared_before_head_cas"
    assert receipt["quality_data_accessed"] is True


def test_cas_failure_leaves_prepared_fail_closed_receipt_without_sealing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _repo, preflight, configuration, readiness, git = _synthetic_two_phase_reservation(
        monkeypatch,
        tmp_path,
    )
    h0 = git("rev-parse", "HEAD")
    real_run = subprocess.run

    def fail_cas(arguments: object, *args: object, **kwargs: object) -> object:
        if isinstance(arguments, list) and arguments[:3] == ["git", "update-ref", "HEAD"]:
            return subprocess.CompletedProcess(
                arguments,
                1,
                stdout="",
                stderr="synthetic compare-and-swap failure",
            )
        return real_run(arguments, *args, **kwargs)

    monkeypatch.setattr(stage_a.subprocess, "run", fail_cas)
    with pytest.raises(stage_a.StageAAuthenticationError, match="compare-and-swap failed"):
        stage_a.reserve_one_run(preflight, configuration, readiness)
    assert git("rev-parse", "HEAD") == h0
    receipt = json.loads(preflight.attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "prepared_before_head_cas"
    assert receipt["quality_data_accessed"] is False
    assert receipt["rerun_automatically_authorized"] is False
    with pytest.raises(stage_a.StageAAuthenticationError, match="refusing to overwrite"):
        stage_a.reserve_one_run(preflight, configuration, readiness)
    assert git("rev-parse", "HEAD") == h0


def test_completed_artifact_publication_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    stage_a._atomic_publish_new(path, b'{"result":1}')
    with pytest.raises(stage_a.StageAAuthenticationError, match="refusing to overwrite"):
        stage_a._atomic_publish_new(path, b'{"result":2}')
    assert path.read_bytes() == b'{"result":1}'
    assert not list(tmp_path.glob("*.tmp"))


def test_record_access_transition_persists_exact_adjacent_phase_and_rejects_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_ledger = stage_a._expected_access_snapshot("reserved_before_task_entry")
    receipt = {
        "status": "reserved_before_quality_data_or_model_weights",
        "access_ledger": initial_ledger,
        "completed_task_ids": [],
        "quality_data_accessed": False,
        "task_row_loaded": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "forward_passes": 0,
        "forward_passes_minimum": 0,
        "quality_result_computed": False,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
    }
    attempt_path = tmp_path / "attempt.json"
    payload = stage_a.canonical_json_bytes(receipt)
    attempt_path.write_bytes(payload)
    preflight = SimpleNamespace(attempt_path=attempt_path, repo_root=tmp_path)
    reservation = stage_a.AttemptReservation(
        receipt=receipt,
        receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
        h0_commit="0" * 40,
        seal_commit="1" * 40,
        tree="2" * 40,
        seal_message_sha256="3" * 64,
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )

    entered = stage_a.AccessLedger(
        phase="task_load_entered",
        task_load_entered=True,
    )
    updated = stage_a.record_access_transition(preflight, reservation, entered)
    persisted = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert persisted["access_ledger"] == stage_a._expected_access_snapshot("task_load_entered")
    assert persisted["task_row_loaded"] is None
    assert updated.receipt_file_sha256 == hashlib.sha256(attempt_path.read_bytes()).hexdigest()

    with pytest.raises(
        stage_a.StageAAuthenticationError,
        match="regressed or skipped",
    ):
        stage_a.record_access_transition(
            preflight,
            updated,
            stage_a.AccessLedger(),
        )


def test_durable_evaluation_entry_is_unknown_if_return_checkpoint_never_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_ledger = stage_a._expected_access_snapshot("reserved_before_task_entry")
    receipt = {
        "status": "reserved_before_quality_data_or_model_weights",
        "access_ledger": initial_ledger,
        "completed_task_ids": [],
        "quality_data_accessed": False,
        "task_row_loaded": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "forward_passes": 0,
        "forward_passes_minimum": 0,
        "quality_result_computed": False,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
    }
    attempt_path = tmp_path / "attempt.json"
    payload = stage_a.canonical_json_bytes(receipt)
    attempt_path.write_bytes(payload)
    preflight = SimpleNamespace(attempt_path=attempt_path, repo_root=tmp_path)
    reservation = stage_a.AttemptReservation(
        receipt=receipt,
        receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
        h0_commit="0" * 40,
        seal_commit="1" * 40,
        tree="2" * 40,
        seal_message_sha256="3" * 64,
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )

    ledger = stage_a.AccessLedger()
    transitions = (
        ("task_load_entered", "task_load_entered"),
        ("task_row_loaded", "task_row_loaded"),
        ("tokenizer_entered", "tokenizer_entered"),
        ("tokenizer_loaded", "tokenizer_loaded"),
        ("model_weights_entered", "model_weights_entered"),
        ("model_weights_loaded", "model_weights_loaded"),
        ("evaluation_entered", "evaluation_entered"),
    )
    for phase, field in transitions:
        ledger.phase = phase
        setattr(ledger, field, True)
        reservation = stage_a.record_access_transition(
            preflight,
            reservation,
            ledger,
        )

    durable = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert durable["status"] == "running_with_monotonic_access_ledger"
    assert durable["access_ledger"]["evaluation_entered"] is True
    assert durable["access_ledger"]["evaluation_returned"] is None
    assert durable["evaluation_returned"] is None
    assert durable["quality_result_computed"] is None
    assert durable["forward_passes"] is None
    assert durable["forward_passes_minimum"] == 0
    assert durable["completed_task_ids"] == []


def _completion_artifact(
    *,
    passed: bool,
) -> tuple[dict[str, object], bytes, str, str, dict[str, object]]:
    gate = {
        "passed": passed,
        "checks": {
            "synthetic": {
                "passed": passed,
                "evidence": {"delta_nll": 0.123456},
                "error": None,
            }
        },
    }
    evidence = {
        "artifact_kind": stage_a.ARTIFACT_KIND,
        "stage_a_gate": gate,
    }
    canonical_hash = stage_a._canonical_sha256(evidence)
    artifact = {
        "schema_version": stage_a.ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": stage_a.ARTIFACT_KIND,
        "canonical_evidence_sha256": canonical_hash,
        "evidence": evidence,
    }
    payload = stage_a.canonical_json_bytes(artifact)
    return (
        artifact,
        payload,
        hashlib.sha256(payload).hexdigest(),
        canonical_hash,
        gate,
    )


def _completion_reservation(receipt: dict[str, object]) -> stage_a.AttemptReservation:
    payload = stage_a.canonical_json_bytes(receipt)
    return stage_a.AttemptReservation(
        receipt=receipt,
        receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
        h0_commit="0" * 40,
        seal_commit="1" * 40,
        tree="2" * 40,
        seal_message_sha256="3" * 64,
    )


def test_prepublication_completion_receipt_binds_but_does_not_expose_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _artifact, _payload, file_hash, canonical_hash, gate = _completion_artifact(passed=True)
    active = {
        "status": "evaluation_returned_before_artifact_finalization",
        "access_ledger": stage_a._expected_access_snapshot("finalization_entered"),
        "quality_aggregate_exposed": False,
    }
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_bytes(stage_a.canonical_json_bytes(active))
    preflight = SimpleNamespace(
        attempt_path=attempt_path,
        output_path=tmp_path / "result.json",
        repo_root=tmp_path,
    )
    reservation = _completion_reservation(active)
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )

    prepared = stage_a._prepare_result_completion_receipt(
        preflight,
        reservation,
        file_hash=file_hash,
        canonical_hash=canonical_hash,
        gate=gate,
    )
    assert prepared["status"] == stage_a.RESULT_PREPARED_STATUS
    assert prepared["quality_aggregate_exposed"] is False
    assert "stage_a_gate" not in prepared
    assert "stage_a_gate_passed" not in prepared
    assert prepared["stage_a_gate_sha256"] == stage_a._canonical_sha256(gate)
    assert "0.123456" not in attempt_path.read_text(encoding="utf-8")


def test_postpublication_promotion_crash_preserves_authenticated_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _artifact, payload, file_hash, canonical_hash, gate = _completion_artifact(passed=True)
    active = {
        "status": "evaluation_returned_before_artifact_finalization",
        "access_ledger": stage_a._expected_access_snapshot("finalization_entered"),
        "quality_aggregate_exposed": False,
    }
    preflight = SimpleNamespace(
        attempt_path=tmp_path / "attempt.json",
        output_path=tmp_path / "result.json",
        repo_root=tmp_path,
    )
    preflight.attempt_path.write_bytes(stage_a.canonical_json_bytes(active))
    reservation = _completion_reservation(active)
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )
    prepared = stage_a._prepare_result_completion_receipt(
        preflight,
        reservation,
        file_hash=file_hash,
        canonical_hash=canonical_hash,
        gate=gate,
    )
    stage_a._atomic_publish_new(preflight.output_path, payload)

    stage_a.record_attempt_failure(
        preflight,
        reservation,
        KeyboardInterrupt("synthetic receipt-promotion crash"),
        stage_a.AccessLedger(
            phase="finalization_entered",
            task_load_entered=True,
            task_row_loaded=True,
            tokenizer_entered=True,
            tokenizer_loaded=True,
            model_weights_entered=True,
            model_weights_loaded=True,
            evaluation_entered=True,
            evaluation_returned=True,
            finalization_entered=True,
        ),
    )
    receipt = json.loads(preflight.attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == stage_a.RESULT_PROMOTION_INTERRUPTED_STATUS
    assert receipt["result_available"] is True
    assert receipt["output_published"] is True
    assert receipt["quality_aggregate_exposed"] is True
    assert receipt["stage_a_gate_passed"] is True
    assert receipt["stage_a_gate_sha256"] == prepared["stage_a_gate_sha256"]
    assert receipt["rerun_automatically_authorized"] is False


def test_result_publication_failure_preserves_computed_result_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _artifact, _payload, file_hash, canonical_hash, gate = _completion_artifact(passed=False)
    active = {
        "status": "evaluation_returned_before_artifact_finalization",
        "access_ledger": stage_a._expected_access_snapshot("finalization_entered"),
        "quality_aggregate_exposed": False,
    }
    preflight = SimpleNamespace(
        attempt_path=tmp_path / "attempt.json",
        output_path=tmp_path / "result.json",
        repo_root=tmp_path,
    )
    preflight.attempt_path.write_bytes(stage_a.canonical_json_bytes(active))
    reservation = _completion_reservation(active)
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )
    prepared = stage_a._prepare_result_completion_receipt(
        preflight,
        reservation,
        file_hash=file_hash,
        canonical_hash=canonical_hash,
        gate=gate,
    )

    stage_a.record_attempt_failure(
        preflight,
        reservation,
        OSError(r"synthetic C:\Users\private publication failure"),
    )
    receipt = json.loads(preflight.attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == stage_a.RESULT_PUBLICATION_FAILED_STATUS
    assert receipt["quality_result_computed"] is True
    assert receipt["completed_task_ids"] == [stage_a.TASK_ID]
    assert receipt["result_available"] is False
    assert receipt["output_published"] is False
    assert receipt["quality_aggregate_exposed"] is False
    assert "stage_a_gate_passed" not in receipt
    assert receipt["stage_a_gate_sha256"] == prepared["stage_a_gate_sha256"]
    assert receipt["rerun_automatically_authorized"] is False


def test_finalize_runs_second_integrity_check_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    preflight = stage_a.StageAPreflight(
        repo_root=tmp_path,
        repository_start={"commit": "0" * 40},
        source_hashes_start={},
        identity_clarification={},
        experiment010_admin_null={},
        anchor={},
        selector_identity={},
        plan=object(),
        stage0={},
        stage0_artifact=tmp_path / "stage0.pt",
        stage0_sha256=tmp_path / "stage0.pt.sha256",
        output_path=tmp_path / "result.json",
        attempt_path=tmp_path / "attempt.json",
    )
    tokenized = stage_a.TokenizedTask(
        row={"task_id": stage_a.TASK_ID},
        prompt_ids=torch.zeros((1, 1), dtype=torch.long),
        code_ids=torch.zeros((1, 1), dtype=torch.long),
        token_manifest=[],
    )
    configuration = stage_a.ModelConfiguration(config=object(), identity={})
    reservation = stage_a.AttemptReservation(
        receipt={"status": "reserved"},
        receipt_file_sha256="1" * 64,
        h0_commit="0" * 40,
        seal_commit="2" * 40,
        tree="3" * 40,
        seal_message_sha256="4" * 64,
    )
    integrity = {
        "repository_end": {"commit": reservation.seal_commit},
        "source_hashes_end": {},
        "one_run_seal": {
            "seal_commit": reservation.seal_commit,
            "tree": reservation.tree,
        },
    }

    def check_integrity(*_args: object) -> dict[str, object]:
        events.append("integrity")
        return copy.deepcopy(integrity)

    monkeypatch.setattr(stage_a, "_assert_end_integrity", check_integrity)
    monkeypatch.setattr(
        stage_a,
        "_build_artifact",
        lambda *_args: (
            events.append("build")
            or {
                "schema_version": 1,
                "artifact_kind": stage_a.ARTIFACT_KIND,
                "canonical_evidence_sha256": "5" * 64,
                "evidence": {},
            },
            {"passed": False},
        ),
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_public_artifact",
        lambda *_args, **_kwargs: events.append("privacy"),
    )
    monkeypatch.setattr(
        stage_a,
        "_prepare_completed_artifact",
        lambda *_args: events.append("prepare") or (b"{}", "6" * 64, "5" * 64),
    )

    def publish(path: Path, payload: bytes) -> None:
        assert events.count("integrity") == 2
        events.append("publish")
        path.write_bytes(payload)

    monkeypatch.setattr(stage_a, "_atomic_publish_new", publish)
    monkeypatch.setattr(
        stage_a,
        "_prepare_result_completion_receipt",
        lambda *_args, **_kwargs: (
            events.append("prepare-receipt") or {"status": stage_a.RESULT_PREPARED_STATUS}
        ),
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_published_output_against_receipt",
        lambda *_args: (
            events.append("validate-output") or {"valid": True, "stage_a_gate_passed": False}
        ),
    )
    monkeypatch.setattr(
        stage_a,
        "_promote_result_completion_receipt",
        lambda *_args: events.append("promote-receipt") or {},
    )
    stage_a.finalize_stage_a(
        {},
        object(),
        tokenized,
        preflight,
        configuration,
        reservation,
    )
    assert events == [
        "integrity",
        "build",
        "privacy",
        "prepare",
        "integrity",
        "prepare-receipt",
        "publish",
        "validate-output",
        "promote-receipt",
    ]


def test_end_integrity_reauthenticates_unchanged_runtime_and_admin_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readiness_receipt = {
        "schema": "recurquant.experiment011.runtime-readiness.v1",
        "accelerator": {"requested_device": "cuda"},
    }
    readiness = stage_a.RuntimeReadiness(
        receipt=readiness_receipt,
        canonical_sha256=stage_a._canonical_sha256(readiness_receipt),
    )
    admin = {
        "scientific_result_available": False,
        "canonical_provenance_sha256": "a" * 64,
    }
    stage0_artifact = tmp_path / "stage0.pt"
    stage0_sidecar = tmp_path / "stage0.pt.sha256"
    git_identity = {"object_directory_authenticated": True}
    preflight = SimpleNamespace(
        repo_root=tmp_path,
        repository_start={"git_identity": git_identity},
        source_hashes_start={},
        experiment010_admin_null=admin,
        stage0={
            "artifact_file_sha256": "b" * 64,
            "sidecar_file_sha256": "c" * 64,
        },
        stage0_artifact=stage0_artifact,
        stage0_sha256=stage0_sidecar,
    )
    reservation = stage_a.AttemptReservation(
        receipt={
            "runtime_readiness": stage_a._readiness_bundle(readiness),
            "preseal_freshness": {
                "repository_reauthenticated": True,
                "source_head_blobs_reauthenticated": True,
                "new_marker_absent": True,
                "stage0_reauthenticated": True,
                "experiment010_administrative_null_reauthenticated": True,
                "runtime_readiness_reauthenticated": True,
                "configuration_reauthenticated_local_only": True,
            },
        },
        receipt_file_sha256="d" * 64,
        h0_commit="0" * 40,
        seal_commit="1" * 40,
        tree="2" * 40,
        seal_message_sha256="e" * 64,
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"seal_commit": reservation.seal_commit},
    )
    monkeypatch.setattr(
        stage_a,
        "_repository_state",
        lambda _root: {
            "worktree_clean": True,
            "commit": reservation.seal_commit,
            "git_identity": git_identity,
        },
    )
    monkeypatch.setattr(stage_a, "_source_hashes", lambda _root: {})

    def file_hash(path: Path) -> str:
        if path == stage0_artifact:
            return "b" * 64
        if path == stage0_sidecar:
            return "c" * 64
        if path.name == Path(stage_a.EXPERIMENT009_STAGE_A_RELATIVE_PATH).name:
            return stage_a.EXPERIMENT009_STAGE_A_FILE_SHA256
        if path.name == Path(stage_a.IDENTITY_NOTE_RELATIVE_PATH).name:
            return stage_a.IDENTITY_NOTE_FILE_SHA256
        if path.name == Path(stage_a.PROTOCOL_NOTE_RELATIVE_PATH).name:
            return stage_a.PROTOCOL_NOTE_FILE_SHA256
        if path.name == Path(stage_a.SELECTOR_RELATIVE_PATH).name:
            return stage_a.SELECTOR_FILE_SHA256
        if path.name == Path(stage_a.LOSS_SELECTOR_RELATIVE_PATH).name:
            return stage_a.LOSS_SELECTOR_FILE_SHA256
        raise AssertionError(path)

    monkeypatch.setattr(stage_a, "_file_sha256", file_hash)
    monkeypatch.setattr(
        stage_a,
        "authenticate_experiment010_administrative_null",
        lambda _root: copy.deepcopy(admin),
    )
    monkeypatch.setattr(
        stage_a,
        "authenticate_runtime_readiness",
        lambda *, device_name: readiness,
    )
    integrity = stage_a._assert_end_integrity(preflight, reservation)
    assert integrity["runtime_readiness_reauthenticated"] is True
    assert integrity["experiment010_administrative_null_reauthenticated"] is True

    changed_receipt = copy.deepcopy(readiness_receipt)
    changed_receipt["accelerator"]["resolved_device"] = "drift"
    changed = stage_a.RuntimeReadiness(
        receipt=changed_receipt,
        canonical_sha256=stage_a._canonical_sha256(changed_receipt),
    )
    monkeypatch.setattr(
        stage_a,
        "authenticate_runtime_readiness",
        lambda *, device_name: changed,
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="readiness changed"):
        stage_a._assert_end_integrity(preflight, reservation)


def test_failure_receipt_withholds_aggregate_and_forbids_automatic_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt = {"schema": "test", "status": "reserved"}
    payload = stage_a.canonical_json_bytes(attempt)
    stage_a._exclusive_write(attempt_path, payload)
    preflight = SimpleNamespace(
        attempt_path=attempt_path,
        output_path=tmp_path / "result.json",
        repo_root=tmp_path,
    )
    attempt["access_ledger"] = stage_a._expected_access_snapshot("reserved_before_task_entry")
    attempt_path.write_bytes(stage_a.canonical_json_bytes(attempt))
    payload = attempt_path.read_bytes()
    reservation = stage_a.AttemptReservation(
        receipt=attempt,
        receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
        h0_commit="0" * 40,
        seal_commit="1" * 40,
        tree="2" * 40,
        seal_message_sha256="3" * 64,
    )
    monkeypatch.setattr(
        stage_a,
        "_validate_one_run_seal",
        lambda *_args, **_kwargs: {"empty_tree_commit": True},
    )
    private_detail = (
        r"C:\Users\Labeeb\private\model.safetensors "
        "sk-proj-synthetic-secret-do-not-publish"
    )
    stage_a.record_attempt_failure(
        preflight,
        reservation,
        RuntimeError(private_detail),
    )
    receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    rendered = attempt_path.read_text(encoding="utf-8")
    assert receipt["status"] == "failed_without_authenticated_stage_a_result"
    assert receipt["quality_aggregate_exposed"] is False
    assert receipt["rerun_automatically_authorized"] is False
    assert private_detail not in rendered
    assert r"C:\Users\Labeeb" not in rendered
    assert "sk-proj-" not in rendered
    assert (
        receipt["failure_detail_sha256"]
        == hashlib.sha256(private_detail.encode("utf-8")).hexdigest()
    )
    stage_a._validate_public_artifact(receipt, repo_root=tmp_path)


def test_failure_recording_chains_receipt_authentication_error(
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("{malformed", encoding="utf-8")
    preflight = SimpleNamespace(
        attempt_path=attempt_path,
        output_path=tmp_path / "result.json",
        repo_root=tmp_path,
    )
    reservation = stage_a.AttemptReservation(
        receipt={},
        receipt_file_sha256="0" * 64,
        h0_commit="1" * 40,
        seal_commit="2" * 40,
        tree="3" * 40,
        seal_message_sha256="4" * 64,
    )

    with pytest.raises(
        stage_a.StageAAuthenticationError,
        match="cannot authenticate the current attempt receipt",
    ) as caught:
        stage_a.record_attempt_failure(
            preflight,
            reservation,
            RuntimeError("synthetic experiment failure"),
        )

    assert isinstance(caught.value.__cause__, stage_a.StageAAuthenticationError)
    assert str(caught.value.__cause__).startswith("cannot read authenticated JSON")


def test_prefill_identity_requires_exact_raw_fp32_state() -> None:
    state = torch.zeros((1, 16, 128, 128), dtype=torch.float32)
    reference = {layer: state for layer in stage_a.LINEAR_LAYER_INDICES}
    candidate = {layer: state.clone() for layer in stage_a.LINEAR_LAYER_INDICES}
    receipt = stage_a._assert_shared_prefill(
        reference,
        candidate,
        method=STATELEASE_METHOD,
    )
    assert receipt["identical_raw_fp32_prefill_state"] is True
    with pytest.raises(RuntimeError, match="identical FP32 prefill"):
        changed = dict(candidate)
        changed[0] = changed[0].clone()
        changed[0][0, 0, 0, 0] = 1.0
        stage_a._assert_shared_prefill(
            reference,
            changed,
            method=STATELEASE_METHOD,
        )
    with pytest.raises(RuntimeError, match="raw FP32 geometry"):
        wrong_dtype = dict(candidate)
        wrong_dtype[0] = wrong_dtype[0].to(torch.float64)
        stage_a._assert_shared_prefill(
            reference,
            wrong_dtype,
            method=STATELEASE_METHOD,
        )


def test_prefill_capture_records_raw_state_before_cache_quantization() -> None:
    class FakeCache:
        def update_recurrent_state(
            self,
            recurrent_states: torch.Tensor,
            layer_idx: int,
            state_idx: int = 0,
            **_kwargs: object,
        ) -> torch.Tensor:
            assert state_idx == 0
            assert layer_idx == 4
            return recurrent_states + 1

    cache = FakeCache()
    raw = torch.tensor([1.0], dtype=torch.float32)
    with stage_a._capture_prefill_writes(cache) as captured:
        returned = cache.update_recurrent_state(raw, 4)
    assert torch.equal(captured[4], raw)
    assert torch.equal(returned, raw + 1)
    assert torch.equal(cache.update_recurrent_state(raw, 4), raw + 1)


def test_nonrecurrent_prefill_covers_full_attention_kv_and_linear_convolution() -> None:
    layers: list[object] = []
    for layer_index in range(24):
        if layer_index in stage_a.LINEAR_LAYER_INDICES:
            layers.append(
                SimpleNamespace(
                    is_conv_states_initialized={0: True},
                    conv_states={0: torch.ones((1, 4, 3), dtype=torch.bfloat16)},
                )
            )
        else:
            layers.append(
                SimpleNamespace(
                    keys=torch.ones(
                        (1, 2, stage_a.PROMPT_TOKENS, 4),
                        dtype=torch.bfloat16,
                    ),
                    values=torch.ones(
                        (1, 2, stage_a.PROMPT_TOKENS, 4),
                        dtype=torch.bfloat16,
                    ),
                )
            )
    snapshot = stage_a._snapshot_nonrecurrent_prefill(SimpleNamespace(layers=layers))
    assert len(snapshot) == 30
    receipt = stage_a._assert_shared_nonrecurrent_prefill(
        snapshot,
        {name: tensor.clone() for name, tensor in snapshot.items()},
        method=STATELEASE_METHOD,
    )
    assert receipt["identical_full_attention_kv_and_convolution_prefill_state"] is True

    changed = {name: tensor.clone() for name, tensor in snapshot.items()}
    changed["layer.3.keys"][0, 0, 0, 0] = 0
    with pytest.raises(RuntimeError, match="non-recurrent prefill cache differs"):
        stage_a._assert_shared_nonrecurrent_prefill(
            snapshot,
            changed,
            method=STATELEASE_METHOD,
        )


def test_model_geometry_is_exact_and_fails_closed_on_layer_drift() -> None:
    layer_types = [
        "linear_attention" if index in stage_a.LINEAR_LAYER_INDICES else "full_attention"
        for index in range(24)
    ]
    config = SimpleNamespace(
        num_hidden_layers=24,
        layer_types=layer_types,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    assert stage_a._validate_model_geometry(config) == stage_a.EXPECTED_GEOMETRY
    config.layer_types[3] = "linear_attention"
    with pytest.raises(stage_a.StageAAuthenticationError, match="geometry drifted"):
        stage_a._validate_model_geometry(config)


def test_token_id_hashes_reject_same_count_content_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = torch.arange(stage_a.PROMPT_TOKENS, dtype=torch.long).unsqueeze(0)
    code = torch.arange(stage_a.CODE_TOKENS, dtype=torch.long).unsqueeze(0)
    monkeypatch.setattr(
        stage_a,
        "PROMPT_TOKEN_IDS_SHA256",
        stage_a._canonical_token_ids_sha256(prompt),
    )
    monkeypatch.setattr(
        stage_a,
        "CODE_TOKEN_IDS_SHA256",
        stage_a._canonical_token_ids_sha256(code),
    )
    receipt = stage_a._validate_token_id_hashes(prompt, code)
    assert receipt["token_id_hash_serialization"] == stage_a.TOKEN_ID_HASH_SERIALIZATION

    changed = code.clone()
    changed[0, -1] += 1
    with pytest.raises(stage_a.StageAAuthenticationError, match="code token-ID hash"):
        stage_a._validate_token_id_hashes(prompt, changed)


def test_aligned_accumulator_records_each_token_and_rejects_nonfinite_logits() -> None:
    accumulator = stage_a._AlignedAccumulator.empty()
    reference = torch.tensor([[[2.0, 0.0]]])
    candidate = torch.tensor([[[1.5, 0.5]]])
    target = torch.tensor([[0]], dtype=torch.long)
    accumulator.append(
        token_index=0,
        reference_logits=reference,
        candidate_logits=candidate,
        input_token=torch.tensor([[1]], dtype=torch.long),
        target=target,
    )
    summary = accumulator.summary()
    assert summary["token_count"] == 1
    assert summary["all_logits_finite"] is True
    assert accumulator.records[0]["write_index"] == 0
    assert accumulator.records[0]["input_token_id"] == 1
    assert accumulator.records[0]["target_token_id"] == 0

    with pytest.raises(RuntimeError, match="non-finite"):
        accumulator.append(
            token_index=1,
            reference_logits=reference,
            candidate_logits=torch.tensor([[[float("nan"), 0.0]]]),
            input_token=torch.tensor([[0]], dtype=torch.long),
            target=target,
        )


def _passing_gate_inputs() -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, float]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    methods = (
        RHT_CQER_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    )
    deltas = {method: 0.25 for method in methods}
    deltas.update(
        {
            STATELEASE_METHOD: 0.14,
            "fixed_cc1": 0.20,
            "fixed_cc2": 0.16,
            "fixed_cc4": 0.15,
            "fixed_cc5": 0.16,
            "fixed_cut4_in5": 0.15,
        }
    )
    metrics = {
        method: {
            "candidate_nll": 2.0 + deltas[method],
            "reference_nll": 2.0,
            "delta_nll": deltas[method],
            "top1_agreement": 0.99,
            "token_count": 38,
            "all_logits_finite": True,
        }
        for method in methods
    }
    trajectory = {
        method: {
            "trajectory_nmse_auc": 0.2,
            "scored_write_count": stage_a.ALIGNED_TOKENS,
            "layer_value_count": (stage_a.ALIGNED_TOKENS * len(stage_a.LINEAR_LAYER_INDICES)),
        }
        for method in methods
    }
    trajectory[STATELEASE_METHOD]["trajectory_nmse_auc"] = 0.05
    trajectory["fixed_cc1"]["trajectory_nmse_auc"] = 0.10
    storage = {
        "resident_bytes_including_statelease": FROZEN_STATELEASE_RESIDENT_BYTES,
        "persistent_fp32_state_mirror": False,
    }
    diagnostics = [
        {
            "layer_index": layer,
            "state_updates": 1 + stage_a.ALIGNED_TOKENS,
            "tokens_observed": stage_a.PROMPT_TOKENS + stage_a.ALIGNED_TOKENS,
            "boundary4_count": 1,
            "boundary5_count": 1,
            "tie_count": 1,
            "invalid_boundary_count": 0,
        }
        for layer in stage_a.LINEAR_LAYER_INDICES
    ]
    evidence = []
    for forward_index in range(1 + stage_a.ALIGNED_TOKENS):
        for layer in stage_a.LINEAR_LAYER_INDICES:
            boundary = 4 if forward_index == 1 else (5 if forward_index == 2 else None)
            evidence.append(
                {
                    "update_index": len(evidence),
                    "layer_index": layer,
                    "state_index": 0,
                    "token_count": stage_a.PROMPT_TOKENS if forward_index == 0 else 1,
                    "boundary": boundary,
                    "tie": forward_index == 2,
                }
            )
    return metrics, trajectory, storage, diagnostics, evidence


def _complete_stage_a_result_fixture() -> dict[str, object]:
    def per_token(*, candidate_offset: float = 0.0) -> list[dict[str, object]]:
        return [
            {
                "write_index": index,
                "input_token_id": index,
                "target_token_id": index + 1,
                "kl": 0.0 if candidate_offset == 0.0 else 0.01,
                "reference_nll": 1.0,
                "candidate_nll": 1.0 + candidate_offset,
                "top1_agreement": True,
                "delta_nll": candidate_offset,
                "all_logits_finite": True,
            }
            for index in range(stage_a.ALIGNED_TOKENS)
        ]

    per_layer_nmse = {str(layer): 0.1 for layer in stage_a.LINEAR_LAYER_INDICES}
    layer_macro_nmse = sum(per_layer_nmse.values()) / len(per_layer_nmse)
    trajectory_rows = [
        {
            "write_index": index,
            "per_layer_nmse": dict(per_layer_nmse),
            "layer_macro_nmse": layer_macro_nmse,
        }
        for index in range(stage_a.ALIGNED_TOKENS)
    ]
    _metrics, _trajectory, _storage, diagnostics, evidence = _passing_gate_inputs()
    reference_rows = per_token()
    reference_summary, _ = stage_a._recompute_aligned_summary(
        method=stage_a.FP32_METHOD,
        rows=reference_rows,
        reference_rows=None,
        expected_code_ids=None,
        reference_method=True,
    )
    reference_trajectory = {
        "trajectory_nmse_auc": 0.0,
        "scored_write_count": stage_a.ALIGNED_TOKENS,
        "layer_value_count": (stage_a.ALIGNED_TOKENS * len(stage_a.LINEAR_LAYER_INDICES)),
    }
    reference = {
        "aligned_metrics": reference_summary,
        "per_token": reference_rows,
        "trajectory": reference_trajectory,
    }
    candidates: dict[str, dict[str, object]] = {}
    for method in stage_a.QUALITY_METHODS:
        rows = per_token(candidate_offset=0.25)
        summary, _ = stage_a._recompute_aligned_summary(
            method=method,
            rows=rows,
            reference_rows=reference_rows,
            expected_code_ids=None,
            reference_method=False,
        )
        trajectory_summary = stage_a._recompute_trajectory_summary(
            method=method,
            rows=trajectory_rows,
        )
        candidates[method] = {
            "aligned_metrics": summary,
            "per_token_aligned": rows,
            "trajectory": trajectory_summary,
            "trajectory_per_write": copy.deepcopy(trajectory_rows),
            "diagnostics": copy.deepcopy(diagnostics) if method == STATELEASE_METHOD else [],
            "update_evidence": copy.deepcopy(evidence) if method == STATELEASE_METHOD else [],
        }
    return {
        "reference": reference,
        "candidates": candidates,
        "device": "cuda",
    }


def test_result_completeness_reconciliation_rejects_missing_or_misordered_evidence() -> None:
    complete = _complete_stage_a_result_fixture()
    receipt = stage_a._validate_stage_a_result_completeness(complete)
    assert receipt["authenticated_forward_passes"] == 429
    assert receipt["all_expected_records_present_and_ordered"] is True

    missing_token = copy.deepcopy(complete)
    missing_token["candidates"][STATELEASE_METHOD]["per_token_aligned"].pop()  # type: ignore[index]
    with pytest.raises(RuntimeError, match="per-token evidence"):
        stage_a._validate_stage_a_result_completeness(missing_token)

    missing_layer = copy.deepcopy(complete)
    del missing_layer["candidates"][STATELEASE_METHOD]["trajectory_per_write"][0][  # type: ignore[index]
        "per_layer_nmse"
    ][str(stage_a.LINEAR_LAYER_INDICES[0])]
    with pytest.raises(RuntimeError, match="trajectory evidence"):
        stage_a._validate_stage_a_result_completeness(missing_layer)

    misordered_update = copy.deepcopy(complete)
    misordered_update["candidates"][STATELEASE_METHOD]["update_evidence"][1][  # type: ignore[index]
        "update_index"
    ] = 999
    with pytest.raises(RuntimeError, match="misordered"):
        stage_a._validate_stage_a_result_completeness(misordered_update)


@pytest.mark.parametrize(
    "field",
    [
        "token_count",
        "mean_kl",
        "cvar95_kl",
        "max_kl",
        "top1_agreement",
        "reference_nll",
        "candidate_nll",
        "delta_nll",
        "all_logits_finite",
    ],
)
def test_every_stored_aligned_scalar_must_match_recomputed_raw_evidence(
    field: str,
) -> None:
    result = _complete_stage_a_result_fixture()
    summary = result["candidates"][STATELEASE_METHOD]["aligned_metrics"]  # type: ignore[index]
    value = summary[field]
    if isinstance(value, bool):
        summary[field] = not value
    elif isinstance(value, int):
        summary[field] = value + 1
    else:
        summary[field] = float(value) + 0.125
    with pytest.raises(RuntimeError, match=f"aligned scalar {field}"):
        stage_a._validate_stage_a_result_completeness(result)


@pytest.mark.parametrize(
    "field",
    [
        "trajectory_nmse_auc",
        "scored_write_count",
        "layer_value_count",
    ],
)
def test_every_stored_trajectory_scalar_must_match_per_layer_write_evidence(
    field: str,
) -> None:
    result = _complete_stage_a_result_fixture()
    summary = result["candidates"][STATELEASE_METHOD]["trajectory"]  # type: ignore[index]
    value = summary[field]
    summary[field] = value + 1 if isinstance(value, int) else float(value) + 0.125
    with pytest.raises(RuntimeError, match=f"trajectory scalar {field}"):
        stage_a._validate_stage_a_result_completeness(result)


def test_raw_per_token_scalar_and_reference_alignment_tampering_is_rejected() -> None:
    raw_kl = _complete_stage_a_result_fixture()
    raw_kl["candidates"][STATELEASE_METHOD]["per_token_aligned"][0]["kl"] += 0.5  # type: ignore[index]
    with pytest.raises(RuntimeError, match="stored aligned scalar"):
        stage_a._validate_stage_a_result_completeness(raw_kl)

    raw_delta = _complete_stage_a_result_fixture()
    raw_delta["candidates"][STATELEASE_METHOD]["per_token_aligned"][0][  # type: ignore[index]
        "delta_nll"
    ] += 0.5
    with pytest.raises(RuntimeError, match="delta_nll does not match"):
        stage_a._validate_stage_a_result_completeness(raw_delta)

    raw_nll = _complete_stage_a_result_fixture()
    nll_row = raw_nll["candidates"][STATELEASE_METHOD]["per_token_aligned"][0]  # type: ignore[index]
    nll_row["reference_nll"] = 1.1
    nll_row["candidate_nll"] = 1.35
    nll_row["delta_nll"] = 0.25
    with pytest.raises(RuntimeError, match="reference alignment"):
        stage_a._validate_stage_a_result_completeness(raw_nll)

    raw_top1 = _complete_stage_a_result_fixture()
    raw_top1["candidates"][STATELEASE_METHOD]["per_token_aligned"][0][  # type: ignore[index]
        "top1_agreement"
    ] = False
    with pytest.raises(RuntimeError, match="stored aligned scalar"):
        stage_a._validate_stage_a_result_completeness(raw_top1)

    raw_finite = _complete_stage_a_result_fixture()
    raw_finite["candidates"][STATELEASE_METHOD]["per_token_aligned"][0][  # type: ignore[index]
        "all_logits_finite"
    ] = False
    with pytest.raises(RuntimeError, match="boolean evidence"):
        stage_a._validate_stage_a_result_completeness(raw_finite)


def test_raw_token_ids_and_trajectory_layer_macro_are_authenticated() -> None:
    code_ids = torch.arange(stage_a.CODE_TOKENS, dtype=torch.long).reshape(1, -1)
    complete = _complete_stage_a_result_fixture()
    stage_a._validate_stage_a_result_completeness(complete, code_ids=code_ids)

    token_tamper = copy.deepcopy(complete)
    token_tamper["candidates"][STATELEASE_METHOD]["per_token_aligned"][0][  # type: ignore[index]
        "input_token_id"
    ] = 999
    with pytest.raises(RuntimeError, match="token identity drifted"):
        stage_a._validate_stage_a_result_completeness(
            token_tamper,
            code_ids=code_ids,
        )

    layer_tamper = copy.deepcopy(complete)
    layer_tamper["candidates"][STATELEASE_METHOD]["trajectory_per_write"][0][  # type: ignore[index]
        "per_layer_nmse"
    ][str(stage_a.LINEAR_LAYER_INDICES[0])] = 0.2
    with pytest.raises(RuntimeError, match="trajectory macro"):
        stage_a._validate_stage_a_result_completeness(layer_tamper)

    macro_tamper = copy.deepcopy(complete)
    macro_tamper["candidates"][STATELEASE_METHOD]["trajectory_per_write"][0][  # type: ignore[index]
        "layer_macro_nmse"
    ] = 0.2
    with pytest.raises(RuntimeError, match="trajectory macro"):
        stage_a._validate_stage_a_result_completeness(macro_tamper)


def test_gate_consumes_only_recomputed_metrics_and_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recomputed_metrics = {
        method: {"source": f"recomputed-metric-{method}"} for method in stage_a.ALL_METHODS
    }
    recomputed_trajectory = {
        method: {"source": f"recomputed-trajectory-{method}"} for method in stage_a.ALL_METHODS
    }
    statelease = {
        "storage": {"stored": "storage"},
        "diagnostics": [{"stored": "diagnostic"}],
        "update_evidence": [{"stored": "update"}],
    }
    candidates = {
        method: (
            statelease
            if method == STATELEASE_METHOD
            else {
                "aligned_metrics": {"source": "untrusted-stored"},
                "trajectory": {"source": "untrusted-stored"},
            }
        )
        for method in stage_a.QUALITY_METHODS
    }
    captured: dict[str, object] = {}

    def gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"passed": False}

    monkeypatch.setattr(stage_a, "evaluate_statelease_stage_a_gate", gate)
    result = stage_a._evaluate_gate_from_recomputed_evidence(
        candidates=candidates,
        recomputed_metrics=recomputed_metrics,
        recomputed_trajectory=recomputed_trajectory,
        stage0_complete=True,
        artifact_integrity=True,
    )
    assert result == {"passed": False}
    assert captured["aligned_metrics"] == {
        method: recomputed_metrics[method] for method in stage_a.QUALITY_METHODS
    }
    assert captured["trajectory_nmse_auc"] == {
        method: recomputed_trajectory[method] for method in stage_a.QUALITY_METHODS
    }
    assert captured["statelease_storage"] is statelease["storage"]


def _serialized_storage_candidates() -> dict[str, dict[str, object]]:
    def schemas(
        names: set[str] | frozenset[str],
        total_bytes: int,
        *,
        allow_empty: bool,
    ) -> dict[str, object]:
        ordered_names = sorted(names)
        count = len(ordered_names)
        minimum = 0 if allow_empty else count - 1
        rows: dict[str, object] = {
            name: {
                "dtype": "torch.uint8",
                "shape": [0 if allow_empty else 1],
                "logical_bytes": 0 if allow_empty else 1,
                "storage_bytes": 0 if allow_empty else 1,
            }
            for name in ordered_names
        }
        final_bytes = total_bytes - minimum
        rows[ordered_names[0]] = {
            "dtype": "torch.uint8",
            "shape": [final_bytes],
            "logical_bytes": final_bytes,
            "storage_bytes": final_bytes,
        }
        return rows

    candidates: dict[str, dict[str, object]] = {}
    for method in stage_a.QUALITY_METHODS:
        candidate_bytes = (
            2_711_552 if method == RHT_CQER_METHOD else FROZEN_STATELEASE_RESIDENT_BYTES
        )
        candidate_names = stage_a._expected_candidate_tensor_names(method)
        candidate_count = len(candidate_names)
        shared_bytes = 300
        common = {
            "persistent_fp32_state_mirror": False,
            "persistent_raw_state_bytes": 0,
            "candidate_persistent_tensor_count": candidate_count,
            "candidate_persistent_storage_bytes": candidate_bytes,
            "shared_persistent_tensor_count": (stage_a.EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT),
            "shared_persistent_storage_bytes": shared_bytes,
            "candidate_tensor_schema": schemas(
                candidate_names,
                candidate_bytes,
                allow_empty=True,
            ),
            "shared_tensor_schema": schemas(
                stage_a.EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES,
                shared_bytes,
                allow_empty=False,
            ),
            "runtime_reachable_tensor_storage_closure_passed": True,
            "runtime_storage_contract_passed": True,
        }
        if method == RHT_CQER_METHOD:
            storage = {
                **common,
                "resident_bytes": 2_564_096,
                "resident_bytes_including_selector": 2_711_552,
                "high_precision_groups": 1_976,
            }
        elif method == STATELEASE_METHOD:
            storage = {
                **common,
                "resident_bytes_including_statelease": FROZEN_STATELEASE_RESIDENT_BYTES,
            }
        elif method in FIXED_REPLAY_METHODS:
            storage = {
                **common,
                "resident_bytes_including_statelease": FROZEN_STATELEASE_RESIDENT_BYTES,
                "logical_resident_capacity_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                "capacity_fully_allocated": True,
                "off_budget": False,
            }
        else:
            storage = {
                **common,
                "resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                "expected_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
                "checkpoint_present": True,
            }
        candidates[method] = {"storage": storage}
    return candidates


def test_storage_contract_authenticates_every_runtime_comparator() -> None:
    candidates = _serialized_storage_candidates()
    receipts = stage_a._validate_candidate_storage_results(candidates)
    assert set(receipts) == set(stage_a.QUALITY_METHODS)

    fixed = FIXED_REPLAY_METHODS[0]
    candidates[fixed]["storage"]["logical_resident_capacity_bytes"] -= 1
    with pytest.raises(RuntimeError, match="serialized storage contract drifted"):
        stage_a._validate_candidate_storage_results(candidates)


def test_historical_rht_storage_requires_shared_schema_but_not_transaction_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        storage_summary=lambda: {
            "resident_bytes": 2_564_096,
            "resident_bytes_including_selector": 2_711_552,
            "high_precision_groups": 1_976,
        }
    )
    with pytest.raises(RuntimeError, match="lacks the FP32 shared-storage schema"):
        stage_a._validated_storage_summary(RHT_CQER_METHOD, cache)

    monkeypatch.setattr(
        stage_a,
        "_audit_persistent_raw_state",
        lambda *_args, **_kwargs: {
            "persistent_fp32_state_mirror": False,
            "persistent_raw_state_bytes": 0,
            "runtime_reachable_tensor_storage_closure_passed": True,
        },
    )
    summary = stage_a._validated_storage_summary(
        RHT_CQER_METHOD,
        cache,
        reference_shared_schema={},
    )
    assert summary["runtime_storage_contract_passed"] is True
    assert summary["runtime_reachable_tensor_storage_closure_passed"] is True


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.uint8,
    ],
)
@pytest.mark.parametrize("placement", ["global", "mapping", "split"])
def test_reachable_storage_closure_rejects_hidden_tensors_of_every_dtype(
    dtype: torch.dtype,
    placement: str,
) -> None:
    shared = torch.ones(2, dtype=torch.bfloat16)
    candidate = torch.ones(3, dtype=torch.uint8)
    cache = SimpleNamespace(shared=shared, candidate=candidate)
    one_value = torch.zeros(1, dtype=dtype)
    if placement == "global":
        cache.hidden_raw_state = one_value
    elif placement == "mapping":
        cache.hidden = {"raw": one_value}
    else:
        cache.hidden_chunks = [one_value]
    with pytest.raises(RuntimeError, match="undeclared persistent tensor storage"):
        stage_a._assert_reachable_tensor_storage_closure(
            method=STATELEASE_METHOD,
            reachable=stage_a._reachable_tensor_paths(cache),
            shared={"shared": shared},
            candidate={"candidate": candidate},
        )


def test_reachable_storage_closure_rejects_empty_view_with_backing_storage() -> None:
    shared = torch.ones(2, dtype=torch.bfloat16)
    candidate = torch.ones(3, dtype=torch.uint8)
    backing = torch.ones(4, dtype=torch.float32)
    cache = SimpleNamespace(
        shared=shared,
        candidate=candidate,
        hidden_empty=backing[:0],
    )
    with pytest.raises(RuntimeError, match="empty tensor views"):
        stage_a._assert_reachable_tensor_storage_closure(
            method=STATELEASE_METHOD,
            reachable=stage_a._reachable_tensor_paths(cache),
            shared={"shared": shared},
            candidate={"candidate": candidate},
        )


def test_reachable_storage_closure_accepts_declared_empty_zero_storage_component() -> None:
    shared = torch.ones(2, dtype=torch.bfloat16)
    candidate = torch.ones(3, dtype=torch.uint8)
    empty_component = torch.empty(0, dtype=torch.uint8)
    cache = SimpleNamespace(
        shared=shared,
        candidate=candidate,
        empty_component=empty_component,
    )
    stage_a._assert_reachable_tensor_storage_closure(
        method="synthetic_equal_byte",
        reachable=stage_a._reachable_tensor_paths(cache),
        shared={"shared": shared},
        candidate={
            "candidate": candidate,
            "empty_component": empty_component,
        },
    )


def test_real_exact_geometry_caches_pass_synthetic_storage_closure_smoke() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from recurquant.packed_cache import (
        RightRhtQueryEmaMixedPackedRecurrentStateCache,
    )
    from recurquant.qwen35 import EXPERIMENT010_STATELEASE_LAYER_QUOTAS
    from recurquant.row_policy import ExactBudgetRowPlan, RowLocation
    from recurquant.statelease_cache import StateLeaseRecurrentStateCache

    rows = tuple(
        RowLocation(
            layer_index=layer_index,
            head_index=flat_index // 128,
            row_index=flat_index % 128,
        )
        for layer_index, quota in EXPERIMENT010_STATELEASE_LAYER_QUOTAS.items()
        for flat_index in range(quota)
    )
    plan = ExactBudgetRowPlan(
        low_bits=4,
        high_bits=8,
        group_size=128,
        scale_bits=16,
        total_groups=36_864,
        mask_bytes=4_608,
        promotion_increment_bytes=64,
        target_resident_bytes=2_564_096,
        resident_bytes=2_564_096,
        high_precision_rows=rows,
        score_shapes=tuple(
            (layer_index, 16, 128) for layer_index in EXPERIMENT010_STATELEASE_LAYER_QUOTAS
        ),
    )
    linear = set(stage_a.LINEAR_LAYER_INDICES)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=[
            "linear_attention" if index in linear else "full_attention" for index in range(24)
        ],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [3, 3, 2],
        },
    )

    def initialize_shared(cache: object) -> None:
        for layer_index, layer in enumerate(cache.layers):
            if layer_index in linear:
                layer.conv_states[0] = torch.zeros((1, 1, 2), dtype=torch.bfloat16)
                layer.is_conv_states_initialized[0] = True
            else:
                layer.keys = torch.zeros((1, 1, 2, 1), dtype=torch.bfloat16)
                layer.values = torch.zeros((1, 1, 2, 1), dtype=torch.bfloat16)
                layer.is_initialized = True

    state = torch.zeros((1, 16, 128, 128), dtype=torch.float32)
    query = torch.zeros((1, 1, 16, 128), dtype=torch.float32)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    log_decay = torch.zeros((1, 1, 16), dtype=torch.float32)
    beta = torch.zeros_like(log_decay)

    statelease = StateLeaseRecurrentStateCache(config, plan=plan)
    initialize_shared(statelease)
    for layer_index in stage_a.LINEAR_LAYER_INDICES:
        statelease.stage_statelease_observation(
            layer_index,
            query,
            key,
            value,
            log_decay,
            beta,
            None,
            state,
        )
        statelease.update_recurrent_state(state, layer_idx=layer_index)
    statelease_summary = stage_a._validated_storage_summary(
        STATELEASE_METHOD,
        statelease,
        reference_shared_schema=stage_a._shared_cache_schema(statelease),
    )
    assert (
        statelease_summary["candidate_persistent_storage_bytes"] == FROZEN_STATELEASE_RESIDENT_BYTES
    )
    assert stage_a._validate_serialized_tensor_schemas(
        method=STATELEASE_METHOD,
        storage=statelease_summary,
    )

    historical = RightRhtQueryEmaMixedPackedRecurrentStateCache(config, plan=plan)
    initialize_shared(historical)
    for layer_index in stage_a.LINEAR_LAYER_INDICES:
        historical.stage_query_observation(layer_index, query)
        historical.update_recurrent_state(state, layer_idx=layer_index)
    historical_summary = stage_a._validated_storage_summary(
        RHT_CQER_METHOD,
        historical,
        reference_shared_schema=stage_a._shared_cache_schema(historical),
    )
    assert historical_summary["candidate_persistent_storage_bytes"] == 2_711_552
    assert stage_a._validate_serialized_tensor_schemas(
        method=RHT_CQER_METHOD,
        storage=historical_summary,
    )


def test_stage_a_gate_includes_historical_anchor_and_fails_closed() -> None:
    metrics, trajectory, storage, diagnostics, evidence = _passing_gate_inputs()
    gate = evaluate_statelease_stage_a_gate(
        aligned_metrics=metrics,
        trajectory_nmse_auc=trajectory,
        statelease_storage=storage,
        statelease_diagnostics=diagnostics,
        statelease_update_evidence=evidence,
        stage0_complete=True,
        artifact_integrity=True,
    )
    assert gate["passed"] is True
    assert gate["method_sets"]["historical_anchor"] == RHT_CQER_METHOD

    metrics["fixed_cc1"]["candidate_nll"] = 2.0
    metrics["fixed_cc1"]["delta_nll"] = 0.0
    failed = evaluate_statelease_stage_a_gate(
        aligned_metrics=metrics,
        trajectory_nmse_auc=trajectory,
        statelease_storage=storage,
        statelease_diagnostics=diagnostics,
        statelease_update_evidence=evidence,
        stage0_complete=True,
        artifact_integrity=True,
    )
    assert failed["passed"] is False
    assert failed["checks"]["cc1_excess_nll_reduction_at_least_10_percent"]["passed"] is False


def test_stage_a_source_set_covers_stage0_and_historical_comparator_sources() -> None:
    from scripts import capture_statelease_stage0 as stage0_capture

    assert set(stage0_capture.SOURCE_IDENTITY_PATHS) <= set(stage_a.SOURCE_FILES)
    assert "scripts/screen_rht_cqer.py" in stage_a.SOURCE_FILES
    assert stage_a.EXPERIMENT010_ATTEMPT_RELATIVE_PATH in stage_a.SOURCE_FILES
    assert stage_a.EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH in stage_a.SOURCE_FILES
    assert stage_a.EXPERIMENT010_ADMIN_NULL_NOTE_RELATIVE_PATH in stage_a.SOURCE_FILES
    assert "research/EXPERIMENT_010_STAGE_A_IDENTITY.md" in stage_a.SOURCE_FILES
    assert "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md" in stage_a.SOURCE_FILES
    assert "research/EXPERIMENT_011_STAGE_A_IDENTITY.md" in stage_a.SOURCE_FILES
    assert "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md" in stage_a.SOURCE_FILES


def test_loaded_local_module_outside_source_set_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = types.ModuleType("recurquant.synthetic_omitted")
    module.__file__ = str(tmp_path / "src" / "recurquant" / "synthetic_omitted.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(
        stage_a.StageAAuthenticationError,
        match="absent from the frozen source set",
    ):
        stage_a._assert_loaded_local_modules_declared(tmp_path)


def test_stage_a_method_matrix_and_claim_boundary_are_explicit() -> None:
    assert (
        RHT_CQER_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    ) == stage_a.QUALITY_METHODS
    assert "cannot support a public improvement" in stage_a.CLAIM_BOUNDARY
    assert "breakthrough claim" in stage_a.CLAIM_BOUNDARY
    assert "Experiment 011" in stage_a.CLAIM_BOUNDARY


def test_cli_has_no_output_override_and_requires_explicit_mode() -> None:
    artifact = Path("stage0.pt")
    args = stage_a.parse_args(
        [
            "--stage0-artifact",
            str(artifact),
            "--local-files-only",
            "--preflight-only",
        ]
    )
    assert args.preflight_only is True
    assert args.local_files_only is True
    assert not hasattr(args, "output")
    with pytest.raises(SystemExit):
        stage_a.parse_args(
            [
                "--stage0-artifact",
                str(artifact),
                "--local-files-only",
            ]
        )
    with pytest.raises(SystemExit):
        stage_a.parse_args(
            [
                "--stage0-artifact",
                str(artifact),
                "--preflight-only",
            ]
        )


def test_experiment011_paths_marker_and_schema_are_unique() -> None:
    assert "experiment011" in stage_a.OUTPUT_RELATIVE_PATH
    assert "experiment011" in stage_a.ATTEMPT_RELATIVE_PATH
    assert stage_a.OUTPUT_RELATIVE_PATH != stage_a.EXPERIMENT010_OUTPUT_RELATIVE_PATH
    assert stage_a.ATTEMPT_RELATIVE_PATH != stage_a.EXPERIMENT010_ATTEMPT_RELATIVE_PATH
    assert stage_a.ONE_RUN_MARKER != stage_a.EXPERIMENT010_ONE_RUN_MARKER
    assert stage_a.ARTIFACT_KIND.startswith("recurquant_experiment011_")


def test_preflight_performs_readiness_without_task_tokenizer_weights_or_seal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight = SimpleNamespace(
        repository_start={"commit": "0" * 40},
        experiment010_admin_null={"scientific_result_available": False},
    )
    readiness_receipt = {
        "schema": "recurquant.experiment011.runtime-readiness.v1",
        "accelerator": {"requested_device": "cuda"},
    }
    readiness = stage_a.RuntimeReadiness(
        receipt=readiness_receipt,
        canonical_sha256=stage_a._canonical_sha256(readiness_receipt),
    )
    configuration = stage_a.ModelConfiguration(
        config=object(),
        identity={"id": stage_a.MODEL_ID},
    )
    events: list[str] = []
    monkeypatch.setattr(
        stage_a,
        "authenticate_static_inputs",
        lambda _args: events.append("static") or preflight,
    )
    monkeypatch.setattr(
        stage_a,
        "authenticate_runtime_readiness",
        lambda *, device_name: events.append("readiness") or readiness,
    )
    monkeypatch.setattr(
        stage_a,
        "load_and_authenticate_config",
        lambda *_args, **_kwargs: events.append("config") or configuration,
    )
    monkeypatch.setattr(stage_a, "reserve_one_run", _unused)
    monkeypatch.setattr(stage_a, "load_exact_authenticated_task", _unused)
    monkeypatch.setattr(stage_a, "tokenize_authenticated_task", _unused)
    monkeypatch.setattr(stage_a, "load_model_weights", _unused)
    assert (
        stage_a.main(
            [
                "--stage0-artifact",
                "stage0.pt",
                "--device",
                "cuda",
                "--local-files-only",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert events == ["static", "readiness", "config"]
    report = json.loads(capsys.readouterr().out)
    assert report["task_row_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["model_weights_loaded"] is False
    assert report["one_run_reserved"] is False


def test_recorded_command_redacts_local_stage0_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_a.sys,
        "argv",
        [
            r"C:\Users\ExampleUser\private\screen_statelease_stage_a.py",
            "--stage0-artifact",
            r"C:\private\experiment010.pt",
            r"--stage0-sha256=C:\private\experiment010.pt.sha256",
            "--preflight-only",
        ],
    )
    command = stage_a._sanitized_command()
    rendered = " ".join(command)
    assert r"C:\private" not in rendered
    assert r"C:\Users\ExampleUser" not in rendered
    assert command[1] == "scripts/screen_statelease_stage_a.py"
    assert "experiment010.pt" in command
    assert "--stage0-sha256=experiment010.pt.sha256" in command


def test_public_artifact_guard_rejects_local_paths_and_secret_like_values(
    tmp_path: Path,
) -> None:
    stage_a._validate_public_artifact(
        {"path": "research/EXPERIMENT_010_STAGE_A_IDENTITY.md"},
        repo_root=tmp_path,
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="absolute local path"):
        stage_a._validate_public_artifact(
            {"path": r"C:\Users\ExampleUser\private\artifact.pt"},
            repo_root=tmp_path,
        )
    for path in ("/tmp/private.pt", "file:///workspace/private.pt"):
        with pytest.raises(
            stage_a.StageAAuthenticationError,
            match="absolute local path",
        ):
            stage_a._validate_public_artifact({"path": path}, repo_root=tmp_path)
    with pytest.raises(stage_a.StageAAuthenticationError, match="secret-like"):
        stage_a._validate_public_artifact(
            {"value": "sk-proj-synthetic-do-not-share"},
            repo_root=tmp_path,
        )
    with pytest.raises(stage_a.StageAAuthenticationError, match="sensitive field"):
        stage_a._validate_public_artifact(
            {"api_key": "redacted"},
            repo_root=tmp_path,
        )
