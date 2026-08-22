from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "screen_static_q468_stage_a.py"
SPEC = importlib.util.spec_from_file_location("screen_static_q468_stage_a_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stage_a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage_a
SPEC.loader.exec_module(stage_a)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity_bytes(
    execution_bindings: dict[str, str] | None = None,
    *,
    capture_receipt_sha256: str | None = None,
    identity_input_sha256: str | None = None,
    authorization_sha256: str | None = None,
) -> bytes:
    execution = (
        {name: _digest(name) for name in sorted(stage_a.EXECUTION_BINDING_FIELDS)}
        if execution_bindings is None
        else dict(execution_bindings)
    )
    calibration = {name: _digest(name) for name in sorted(stage_a.BINDING_FIELDS)}
    calibration["calibration_authorization_file_sha256"] = authorization_sha256 or _digest(
        "calibration-authorization"
    )
    receipt_sha256 = capture_receipt_sha256 or _digest("fixture-stage-a-capture-receipt")
    records = []
    for family in ("pg19", "ruler", "humaneval_plus"):
        for rank in range(4):
            prompt = 4_096 if family == "pg19" else 7
            records.append(
                {
                    "family": family,
                    "selection_rank": rank,
                    "token_span": {
                        "prefill_start": 0,
                        "prefill_stop": prompt,
                        "scored_start": prompt,
                        "scored_stop": prompt + 2,
                        "cache_exposed_start": prompt + 1,
                        "cache_exposed_stop": prompt + 2,
                    },
                }
            )
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
            stage_a.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: receipt_sha256,
        },
        "source_manifest_sha256": identity_input_sha256 or _digest("identity-input"),
        stage_a.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: receipt_sha256,
        "execution_bindings": execution,
        "calibration_binding": calibration,
        "records": records,
    }
    return stage_a.canonical_json_bytes(
        {
            "canonical_evidence_sha256": stage_a.sha256_bytes(
                stage_a.canonical_json_bytes(evidence)
            ),
            "evidence": evidence,
        }
    )


def _capture_receipt_bytes(
    execution_bindings: dict[str, str],
    *,
    binding_bytes: bytes,
    source_commit: str = "1" * 40,
    capture_source_sha256: str | None = None,
    identity_input_sha256: str | None = None,
    authorization_sha256: str | None = None,
) -> bytes:
    return stage_a.canonical_json_bytes(
        {
            "artifact_kind": stage_a.STAGE_A_CAPTURE_PROVENANCE_KIND,
            "calibration_authorization_file_sha256": (
                authorization_sha256 or _digest("calibration-authorization")
            ),
            "calibration_binding_file_sha256": stage_a.sha256_bytes(binding_bytes),
            "capture_source": {
                "path": stage_a.CAPTURE_SOURCE_PATH,
                "sha256": capture_source_sha256 or _digest("capture-source"),
            },
            "capture_version": 6,
            "critical_module_origins": [],
            "excluded_runtime_modules": ["pkg_resources", "setuptools"],
            "execution_bindings": dict(execution_bindings),
            "identity_input_file_sha256": identity_input_sha256 or _digest("identity-input"),
            "phase": "stage_a",
            "publication_contract": stage_a.STAGE_A_CAPTURE_PUBLICATION_CONTRACT,
            "runner_revision": stage_a.STAGE_A_CAPTURE_RUNNER_REVISION,
            "schema_version": 1,
            "source_commit": source_commit,
            "status": stage_a.STAGE_A_CAPTURE_PROVENANCE_STATUS,
        }
    )


def _bootstrap() -> Any:
    return stage_a.bootstrap_stage_a_identity(_identity_bytes())


def _runtime_record() -> dict[str, object]:
    return {
        "base_runtime_file_count": 3,
        "distribution_count": 1,
        "distributions": [["torch", "2.13.0+cu130"]],
        "file_count": 5,
        "git_executable_absolute_path_sha256": _digest("git-absolute-path"),
        "git_executable_sha256": _digest("git-executable"),
        "git_executable_size_bytes": 123_456,
        "interpreter_sha256": _digest("interpreter"),
        "machine_name": "AMD64",
        "manifest_file_sha256": _bootstrap().execution_bindings[
            "calibration_runtime_manifest_file_sha256"
        ],
        "package_root_count": 1,
        "python_cache_tag": "cpython-311",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
    }


def _runtime_namespace() -> Any:
    record = _runtime_record()
    return SimpleNamespace(
        **{
            **record,
            "distributions": tuple(tuple(item) for item in record["distributions"]),
        }
    )


def test_bootstrap_requires_promoted_v6_and_exact_forward_formula() -> None:
    decoded = _bootstrap()
    assert decoded.expected_forward_count == 9 * 12 * 2
    assert set(decoded.execution_bindings) == stage_a.EXECUTION_BINDING_FIELDS
    assert set(decoded.calibration_binding) == stage_a.BINDING_FIELDS

    root = stage_a._strict_json(_identity_bytes(), context="test identity")
    root["evidence"]["phase"] = "calibration"
    root["canonical_evidence_sha256"] = stage_a.sha256_bytes(
        stage_a.canonical_json_bytes(root["evidence"])
    )
    with pytest.raises(stage_a.StageAError, match="promoted resolver-v6"):
        stage_a.bootstrap_stage_a_identity(stage_a.canonical_json_bytes(root))


def test_bootstrap_accepts_exact_finalized_stage_a_capture_receipt_and_rejects_cross_chain() -> (
    None
):
    binding_bytes = b"production-shaped Stage-A calibration binding\n"
    provisional_identity = _bootstrap()
    receipt_bytes = _capture_receipt_bytes(
        dict(provisional_identity.execution_bindings),
        binding_bytes=binding_bytes,
    )
    identity = stage_a.bootstrap_stage_a_identity(
        _identity_bytes(
            dict(provisional_identity.execution_bindings),
            capture_receipt_sha256=stage_a.sha256_bytes(receipt_bytes),
        )
    )
    decoded = stage_a.bootstrap_stage_a_capture_provenance_receipt(
        receipt_bytes,
        expected_file_sha256=stage_a.sha256_bytes(receipt_bytes),
        calibration_binding_artifact=binding_bytes,
        identity=identity,
        expected_source_commit="1" * 40,
    )
    assert decoded["capture_version"] == 6
    assert decoded["runner_revision"] == stage_a.STAGE_A_CAPTURE_RUNNER_REVISION

    mutations = (
        ("capture_version", 5, "finalized envelope"),
        ("identity_input_file_sha256", _digest("other-input"), "different identity input"),
        ("calibration_binding_file_sha256", _digest("other-binding"), "different calibration"),
    )
    for field, value, message in mutations:
        root = stage_a._strict_json(receipt_bytes, context="test capture receipt")
        root[field] = value
        mutated = stage_a.canonical_json_bytes(root)
        rebound_identity = stage_a.bootstrap_stage_a_identity(
            _identity_bytes(
                dict(provisional_identity.execution_bindings),
                capture_receipt_sha256=stage_a.sha256_bytes(mutated),
            )
        )
        with pytest.raises(stage_a.StageAError, match=message):
            stage_a.bootstrap_stage_a_capture_provenance_receipt(
                mutated,
                expected_file_sha256=stage_a.sha256_bytes(mutated),
                calibration_binding_artifact=binding_bytes,
                identity=rebound_identity,
                expected_source_commit="1" * 40,
            )


def test_receipt_mismatch_fails_before_execution_source_provider_or_model_access(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    binding_bytes = b"Stage-A calibration binding\n"
    provisional_identity = _bootstrap()
    receipt_bytes = _capture_receipt_bytes(
        dict(provisional_identity.execution_bindings),
        binding_bytes=binding_bytes,
    )
    identity_bytes = _identity_bytes(
        dict(provisional_identity.execution_bindings),
        capture_receipt_sha256=stage_a.sha256_bytes(receipt_bytes),
    )
    identity_path = tmp_path / "identity.json"
    binding_path = tmp_path / "binding.json"
    receipt_path = tmp_path / "receipt.json"
    reads: list[Path] = []
    payloads = {
        identity_path: identity_bytes,
        binding_path: binding_bytes,
        receipt_path: receipt_bytes,
    }
    config = SimpleNamespace(
        frozen_identity_path=identity_path,
        calibration_binding_path=binding_path,
        stage_a_capture_provenance_receipt_path=receipt_path,
        expected_stage_a_capture_provenance_receipt_sha256=_digest("wrong-explicit-receipt"),
        source_commit="1" * 40,
    )

    def stable_bytes(path: Path, *, context: str) -> bytes:
        del context
        reads.append(path)
        return payloads[path]

    monkeypatch.setattr(stage_a, "_assert_output_paths_isolated", lambda _config: None)
    monkeypatch.setattr(stage_a, "_stable_file_bytes", stable_bytes)
    monkeypatch.setattr(
        stage_a,
        "_read_execution_artifacts",
        lambda _config: pytest.fail("execution artifacts must not be read"),
    )
    monkeypatch.setattr(
        stage_a,
        "_load_exact_module",
        lambda *_args, **_kwargs: pytest.fail("source/provider modules must not be loaded"),
    )
    with pytest.raises(stage_a.StageAError, match="differs from authenticated custody"):
        stage_a.authenticate_production(config, require_input_bundle=False)
    assert reads == [identity_path, binding_path, receipt_path]


def test_legacy_v5_identity_fails_before_binding_receipt_or_provider_access(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    legacy = stage_a._strict_json(_identity_bytes(), context="legacy Stage-A identity")
    legacy["evidence"]["schema_version"] = 5
    legacy["evidence"]["identity_schema"] = "recurquant.experiment013.identity-frozen.v5"
    legacy["canonical_evidence_sha256"] = stage_a.sha256_bytes(
        stage_a.canonical_json_bytes(legacy["evidence"])
    )
    legacy_bytes = stage_a.canonical_json_bytes(legacy)
    identity_path = tmp_path / "legacy-identity.json"
    reads: list[Path] = []

    def stable_bytes(path: Path, *, context: str) -> bytes:
        del context
        reads.append(path)
        if path != identity_path:
            pytest.fail("binding or receipt path must not be accessed for legacy v5")
        return legacy_bytes

    monkeypatch.setattr(stage_a, "_assert_output_paths_isolated", lambda _config: None)
    monkeypatch.setattr(stage_a, "_stable_file_bytes", stable_bytes)
    monkeypatch.setattr(
        stage_a,
        "_load_exact_module",
        lambda *_args, **_kwargs: pytest.fail("provider modules must not be loaded"),
    )
    with pytest.raises(stage_a.StageAError, match="promoted resolver-v6"):
        stage_a.authenticate_production(
            SimpleNamespace(frozen_identity_path=identity_path),
            require_input_bundle=False,
        )
    assert reads == [identity_path]


def test_exact_module_loader_executes_authenticated_bytes_not_swapped_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    name = "_recurquant_stage_a_exact_swap_regression"
    relative = "scripts/exact_swap_regression.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    sentinel = tmp_path / "unauthenticated-module-executed.txt"
    authenticated_bytes = b'VALUE = "authenticated-old"\n'
    swapped_bytes = (
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('new code executed', encoding='utf-8')\n"
        'VALUE = "unauthenticated-new"\n'
    ).encode()
    path.write_bytes(authenticated_bytes)
    entries = {relative: {"raw_sha256": stage_a.sha256_bytes(authenticated_bytes)}}

    def authenticated_read_then_swap(read_path: Path, *, context: str) -> bytes:
        del context
        assert read_path == path.resolve()
        data = read_path.read_bytes()
        read_path.write_bytes(swapped_bytes)
        return data

    monkeypatch.setattr(stage_a, "_stable_file_bytes", authenticated_read_then_swap)
    try:
        module = stage_a._load_exact_module(
            name,
            relative,
            repository_root=tmp_path,
            entries=entries,
        )
        assert module.VALUE == "authenticated-old"
        assert path.read_bytes() == swapped_bytes
        assert not sentinel.exists()
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda root: root["evidence"]["records"].reverse(),
        lambda root: root["evidence"]["records"][0]["token_span"].__setitem__(
            "cache_exposed_start", 4_096
        ),
        lambda root: root["evidence"]["calibration_binding"].pop(
            "static_k29334_policy_file_sha256"
        ),
    ),
)
def test_bootstrap_rejects_reorder_span_and_binding_inventory(mutation: Any) -> None:
    root = stage_a._strict_json(_identity_bytes(), context="test identity")
    mutation(root)
    root["canonical_evidence_sha256"] = stage_a.sha256_bytes(
        stage_a.canonical_json_bytes(root["evidence"])
    )
    with pytest.raises(stage_a.StageAError):
        stage_a.bootstrap_stage_a_identity(stage_a.canonical_json_bytes(root))


class _Sequence:
    def __init__(self, family: str, rank: int) -> None:
        prompt = (10, 11, 12)
        target = (1, 2)
        self.prompt_token_ids = prompt
        self.target_token_ids = target
        self.identity_record = {
            "family": family,
            "canonical_id": f"{family}-{rank}",
            "selection_rank": rank,
            "identity_record_sha256": _digest(f"{family}-{rank}"),
            "token_span": {
                "prefill_start": 0,
                "prefill_stop": len(prompt),
                "scored_start": len(prompt),
                "scored_stop": len(prompt) + len(target),
                "cache_exposed_start": len(prompt) + 1,
                "cache_exposed_stop": len(prompt) + len(target),
            },
        }


class _Engine:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.forward_calls = 0
        self.load_count = 0
        self.close_count = 0

    def load_model(self, authenticated_model_files: object) -> object:
        self.events.append("load")
        self.load_count += 1
        return authenticated_model_files

    def close_model(self, model: object) -> None:
        self.events.append("close")
        self.close_count += 1

    def begin_method(self, model: object, method: Any, sequence: object) -> object:
        self.events.append(f"begin:{method.method_id}")
        return {"method": method.method_id}

    @staticmethod
    def _observation(position: int, target: int, method_id: str) -> Any:
        logits = torch.tensor([1.2, 0.4, -0.8], dtype=torch.float32)
        logp = torch.log_softmax(logits, dim=-1)
        return stage_a.ForwardObservation(
            position=position,
            target_token_id=target,
            comparison_logits=logits,
            target_nll=-float(logp[target].item()),
            top1_token_id=0,
            local_codec_sse=0.01,
            trajectory_nmse=0.02,
            latency_ns=10,
            peak_allocated_bytes=20,
            peak_reserved_bytes=30,
            resident_bytes=stage_a.EXPECTED_RECURRENT_RESIDENT_BYTES[method_id],
            transient_bytes=50,
        )

    def prefill(
        self,
        session: object,
        *,
        prompt_token_ids: tuple[int, ...],
        first_target_token_id: int,
        position: int,
    ) -> Any:
        del prompt_token_ids
        self.forward_calls += 1
        return self._observation(position, first_target_token_id, session["method"])

    def step(
        self,
        session: object,
        *,
        input_token_id: int,
        target_token_id: int,
        position: int,
    ) -> Any:
        del input_token_id
        self.forward_calls += 1
        return self._observation(position, target_token_id, session["method"])

    def end_method(self, session: object) -> dict[str, int]:
        return {
            "resident_bytes": stage_a.EXPECTED_RECURRENT_RESIDENT_BYTES[session["method"]],
            "raw_state_workspace_peak_bytes": 20,
            "query_workspace_peak_bytes": 30,
        }

    def runtime_snapshot(self, model: object) -> dict[str, object]:
        del model
        return {
            "attention_implementation": "eager",
            "capability": [12, 0],
            "cuda_runtime": "13.0",
            "device_index": 0,
            "model_class": "Qwen3_5ForCausalLM",
            "model_config_class": "Qwen3_5TextConfig",
            "model_parameter_dtype": "torch.bfloat16",
            "name": "Test GPU",
            "torch_version": "2.13.0+cu130",
            "total_memory_bytes": 8_000_000_000,
        }


def _authenticated() -> Any:
    bootstrap = _bootstrap()
    methods = tuple(
        stage_a.StageAMethodSpec(
            method,
            None,
            (
                None
                if method in {stage_a.FP32_METHOD, stage_a.DYNAMIC_K27030_METHOD}
                else _digest(f"policy-{method}")
            ),
            "test",
        )
        for method in stage_a.METHOD_ORDER
    )
    return stage_a.AuthenticatedStageA(
        bootstrap_identity=bootstrap,
        identity=object(),
        binding=SimpleNamespace(file_sha256=_digest("binding")),
        capture_provenance_receipt=SimpleNamespace(
            file_sha256=bootstrap.stage_a_capture_provenance_receipt_file_sha256
        ),
        capture_provenance_receipt_bytes=b"fixture finalized Stage-A capture receipt\n",
        dependency_bytes=MappingProxyType({}),
        execution_artifact_bytes=MappingProxyType({}),
        source_manifest=MappingProxyType({}),
        source_manifest_file_sha256=_digest("source"),
        source_commit="1" * 40,
        input_bundle=object(),
        input_bundle_manifest_file_sha256=_digest("input-bundle"),
        model_manifest=object(),
        authenticated_model_files=object(),
        runtime_manifest=object(),
        authenticated_runtime=_runtime_namespace(),
        resolver=SimpleNamespace(),
        capture=SimpleNamespace(),
        calibration_runner=SimpleNamespace(),
        source_module=SimpleNamespace(),
        methods=methods,
    )


def _smoke_report() -> dict[str, object]:
    return {
        "profile": stage_a.PRESEAL_ENGINE_SMOKE_PROFILE,
        "passed": True,
        "stage_a_content_accessed": False,
        "input_profile": "fixed_public_synthetic_4096_plus_128_v3",
        "prompt_token_count": stage_a.PRESEAL_ENGINE_SMOKE_PROMPT_TOKEN_COUNT,
        "continuation_token_count": stage_a.PRESEAL_ENGINE_SMOKE_TARGET_TOKEN_COUNT,
        "model_load_count": 1,
        "method_order": list(stage_a.METHOD_ORDER),
        "forward_count": len(stage_a.METHOD_ORDER) * len(stage_a.PRESEAL_ENGINE_SMOKE_TARGET),
        "method_receipts": [
            {
                "method_id": method_id,
                "forward_count": len(stage_a.PRESEAL_ENGINE_SMOKE_TARGET),
                "logical_recurrent_resident_bytes": (
                    stage_a.EXPECTED_RECURRENT_RESIDENT_BYTES[method_id]
                ),
                "equal_byte_observer_required": method_id != stage_a.FP32_METHOD,
            }
            for method_id in stage_a.METHOD_ORDER
        ],
        "device": _Engine().runtime_snapshot(object()),
    }


def _materialization() -> Any:
    sequences = tuple(
        _Sequence(family, rank)
        for family in ("pg19", "ruler", "humaneval_plus")
        for rank in range(4)
    )
    return SimpleNamespace(
        sequences=sequences,
        frozen_identity_file_sha256=_bootstrap().file_sha256,
        calibration_binding_file_sha256=_digest("binding"),
        capture_input_sha256=_digest("capture"),
        token_sequence_manifest_sha256=_digest("tokens"),
        tokenizer_manifest_sha256=_digest("tokenizer"),
    )


def test_fixed_grid_counts_every_prefill_and_transition_without_token_ids() -> None:
    engine = _Engine()
    result = stage_a.evaluate_materialized_stage_a(
        _authenticated(), _materialization(), engine, object()
    )
    assert result.forward_count == 9 * 12 * 2
    assert engine.forward_calls == result.forward_count
    assert len(result.raw_rows) == 9 * 12
    assert len(result.gate_rows) == 9 * 12
    assert [row["method_id"] for row in result.raw_rows[:9]] == list(stage_a.METHOD_ORDER)
    assert all(
        "input_token_id" not in row and "target_token_id" not in row for row in result.raw_rows
    )
    assert not hasattr(stage_a, "_token_hash")
    assert all(
        "input_token_ids_sha256" not in row and "target_token_ids_sha256" not in row
        for row in result.raw_rows
    )
    assert all(len(row["authenticated_transition_sha256"]) == 64 for row in result.raw_rows)
    assert all(all(row["finite_checks"].values()) for row in result.raw_rows)


def test_comparison_logits_are_compact_fp32_and_kl_matches_reviewed_equation() -> None:
    reference = torch.tensor([2.0, -0.5, 0.25, 1.0], dtype=torch.float32)
    candidate = torch.tensor([1.75, -0.1, 0.5, 0.8], dtype=torch.float32)
    reference_logp = torch.log_softmax(reference, dim=-1)
    candidate_logp = torch.log_softmax(candidate, dim=-1)
    expected = float((reference_logp.exp() * (reference_logp - candidate_logp)).sum(dim=-1).item())
    assert stage_a._kl(reference, candidate) == expected
    observation = stage_a.ForwardObservation(
        position=3,
        target_token_id=2,
        comparison_logits=reference,
        target_nll=-float(reference_logp[2].item()),
        top1_token_id=0,
    )
    validated = stage_a._finite_observation(
        observation,
        expected_position=3,
        expected_target=2,
    )
    assert validated.comparison_logits is reference
    assert reference.device.type == "cpu" and reference.dtype == torch.float32
    assert reference.is_contiguous()
    assert ".tolist()" not in MODULE_PATH.read_text(encoding="utf-8")


def test_comparison_logits_reject_python_tuple_and_non_fp32_tensor() -> None:
    base = _Engine._observation(3, 1, stage_a.FP32_METHOD)
    for invalid in ((1.0, 0.0), torch.tensor([1.0, 0.0], dtype=torch.float64)):
        with pytest.raises(stage_a.StageAError, match="contiguous CPU float32"):
            stage_a._finite_observation(
                stage_a.dataclasses.replace(base, comparison_logits=invalid),
                expected_position=3,
                expected_target=1,
            )


def test_evaluator_rejects_method_reordering_before_any_forward() -> None:
    authenticated = _authenticated()
    reordered = (authenticated.methods[1], authenticated.methods[0], *authenticated.methods[2:])
    authenticated = stage_a.dataclasses.replace(authenticated, methods=reordered)
    engine = _Engine()
    with pytest.raises(stage_a.StageAError, match="reordered"):
        stage_a.evaluate_materialized_stage_a(authenticated, _materialization(), engine, object())
    assert engine.forward_calls == 0


def test_evaluator_rejects_materialized_span_tamper() -> None:
    materialization = _materialization()
    materialization.sequences[0].identity_record["token_span"]["cache_exposed_start"] = 3
    engine = _Engine()
    with pytest.raises(stage_a.StageAError, match="token span"):
        stage_a.evaluate_materialized_stage_a(_authenticated(), materialization, engine, object())


def test_run_orders_auth_seal_materialization_single_load_and_reauthentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    authenticated = _authenticated()
    materialization = _materialization()
    reservation = stage_a.AttemptReservation(MappingProxyType({}), "1" * 40, "2" * 40, "3" * 40)

    class OrderedEngine(_Engine):
        def load_model(self, authenticated_model_files: object) -> object:
            events.append("load")
            return super().load_model(authenticated_model_files)

        def close_model(self, model: object) -> None:
            events.append("close")
            super().close_model(model)

        def begin_method(self, model: object, method: Any, sequence: object) -> object:
            if "evaluate" not in events:
                events.append("evaluate")
            return super().begin_method(model, method, sequence)

    engine = OrderedEngine()

    def authenticate(_config: Any) -> Any:
        events.append("authenticate")
        return authenticated

    def reserve(_config: Any, _authenticated: Any) -> Any:
        events.append("reserve")
        return reservation

    def smoke(_authenticated: Any) -> Any:
        events.append("smoke")
        return _smoke_report()

    def reauthenticate(
        _config: Any,
        _authenticated: Any,
        _reservation: Any,
    ) -> None:
        events.append("reauthenticate")

    def persist(_config: Any, current: Any, updates: Any) -> Any:
        events.append(f"receipt:{updates['status']}")
        return stage_a.dataclasses.replace(
            current,
            receipt=MappingProxyType({**dict(current.receipt), **dict(updates)}),
        )

    def materialize(_config: Any, _authenticated: Any) -> Any:
        events.append("materialize")
        return materialization

    def build(
        _authenticated: Any,
        _materialization: Any,
        _evaluation: Any,
        _reservation: Any,
    ) -> bytes:
        events.append("build")
        return stage_a.canonical_json_bytes({"evidence": {"stage_a_passed": True}})

    def publish(_config: Any, _reservation: Any, _payload: bytes, summary: Any) -> Any:
        events.append("publish")
        return {"status": "published", **dict(summary)}

    monkeypatch.setattr(stage_a, "build_execution_artifact", build)
    services = stage_a.StageAServices(
        authenticate=authenticate,
        reauthenticate=reauthenticate,
        preseal_smoke=smoke,
        reserve=reserve,
        materialize=materialize,
        engine=engine,
        persist_receipt=persist,
        publish=publish,
        record_failure=lambda *_args: events.append("failure"),
    )
    result = stage_a.run_stage_a(SimpleNamespace(), services)
    assert result == {"status": "published", "stage_a_passed": True}
    assert engine.load_count == 1 and engine.close_count == 1
    assert events == [
        "authenticate",
        "smoke",
        "reauthenticate",
        "reserve",
        "receipt:preseal_engine_smoke_bound_before_materialization",
        "reauthenticate",
        "receipt:stage_a_materialization_entered",
        "materialize",
        "receipt:stage_a_content_materialized_before_model_load",
        "reauthenticate",
        "load",
        "reauthenticate",
        "receipt:model_loaded_once_before_evaluation",
        "evaluate",
        "close",
        "receipt:evaluation_returned_before_result_build",
        "reauthenticate",
        "build",
        "reauthenticate",
        "publish",
    ]


def test_loaded_model_is_closed_if_post_load_receipt_persistence_fails() -> None:
    authenticated = _authenticated()
    materialization = _materialization()
    reservation = stage_a.AttemptReservation(
        MappingProxyType({}),
        "1" * 40,
        "2" * 40,
        "3" * 40,
    )
    engine = _Engine()
    failures: list[str] = []

    def persist(_config: Any, current: Any, updates: Any) -> Any:
        if updates["status"] == "model_loaded_once_before_evaluation":
            raise OSError("injected receipt failure after model load")
        return stage_a.dataclasses.replace(
            current,
            receipt=MappingProxyType({**dict(current.receipt), **dict(updates)}),
        )

    services = stage_a.StageAServices(
        authenticate=lambda _config: authenticated,
        reauthenticate=lambda *_args: None,
        preseal_smoke=lambda _authenticated: _smoke_report(),
        reserve=lambda *_args: reservation,
        materialize=lambda *_args: materialization,
        engine=engine,
        persist_receipt=persist,
        publish=lambda *_args: pytest.fail("publication must not run"),
        record_failure=lambda _config, _reservation, _error, phase: failures.append(phase),
    )

    with pytest.raises(OSError, match="after model load"):
        stage_a.run_stage_a(SimpleNamespace(), services)

    assert engine.load_count == 1
    assert engine.close_count == 1
    assert failures == ["post_model_load_receipt"]


def test_run_rejects_device_runtime_drift_after_evaluation() -> None:
    authenticated = _authenticated()
    materialization = _materialization()
    reservation = stage_a.AttemptReservation(
        MappingProxyType({}),
        "1" * 40,
        "2" * 40,
        "3" * 40,
    )

    class DriftingEngine(_Engine):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_count = 0

        def runtime_snapshot(self, model: object) -> dict[str, object]:
            result = super().runtime_snapshot(model)
            self.snapshot_count += 1
            if self.snapshot_count == 2:
                result["name"] = "Drifted GPU"
            return result

    engine = DriftingEngine()
    failures: list[str] = []

    def persist(_config: Any, current: Any, updates: Any) -> Any:
        return stage_a.dataclasses.replace(
            current,
            receipt=MappingProxyType({**dict(current.receipt), **dict(updates)}),
        )

    services = stage_a.StageAServices(
        authenticate=lambda _config: authenticated,
        reauthenticate=lambda *_args: None,
        preseal_smoke=lambda _authenticated: _smoke_report(),
        reserve=lambda *_args: reservation,
        materialize=lambda *_args: materialization,
        engine=engine,
        persist_receipt=persist,
        publish=lambda *_args: pytest.fail("publication must not run"),
        record_failure=lambda _config, _reservation, _error, phase: failures.append(phase),
    )

    with pytest.raises(stage_a.StageAError, match="between model load and evaluation end"):
        stage_a.run_stage_a(SimpleNamespace(), services)

    assert engine.snapshot_count == 2
    assert engine.close_count == 1
    assert failures == ["post_evaluation_device_reauthentication"]


def test_reconstruction_has_no_q48_or_uniform_supplied_dependency(monkeypatch: Any) -> None:
    dependencies = {name: name.encode() for name in stage_a.BINDING_DEPENDENCY_NAMES}
    dependencies["extra_q48_policy"] = b"forbidden"
    with pytest.raises(stage_a.StageAError, match="exact authorized"):
        stage_a.reconstruct_stage_a_methods(
            dependency_bytes=dependencies,
            frozen_stage_a_identity=object(),
            source_commit="1" * 40,
        )

    aggregate = SimpleNamespace(
        d4=object(),
        d6=object(),
        d8=object(),
        sequence_score_manifest_sha256=_digest("scores"),
    )
    scores = SimpleNamespace(
        calibration_identity_sha256=_digest("calibration-identity"),
        calibration_scores_sha256=_digest("calibration-scores"),
        geometry=SimpleNamespace(total_rows=4),
        aggregate=aggregate,
    )

    class Policy:
        def __init__(self, method_id: str) -> None:
            self.method_id = method_id
            self.source_commit = "1" * 40
            self.policy_sha256 = _digest(method_id)

    static = SimpleNamespace(
        FROZEN_STATIC_Q48_PROMOTIONS=14_739,
        static_q48_distortion_sha256=lambda *args, **kwargs: _digest("q48-scores"),
        build_static_rht_q468_policy=lambda *args, method_id, **kwargs: Policy(method_id),
        build_static_rht_q48_policy=lambda *args, method_id, **kwargs: Policy(method_id),
        deserialize_static_rht_q468_policy=lambda payload: Policy(payload.decode()),
        serialize_static_rht_q468_policy=lambda policy: policy.method_id.encode(),
        serialize_static_rht_q48_policy=lambda policy: policy.method_id.encode(),
    )
    calibration = SimpleNamespace(deserialize_calibration_score_artifact=lambda payload: scores)
    original_import = stage_a.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "recurquant.static_q468":
            return static
        if name == "recurquant.static_q468_calibration":
            return calibration
        return original_import(name)

    monkeypatch.setattr(stage_a.importlib, "import_module", fake_import)
    clean = {name: name.encode() for name in stage_a.BINDING_DEPENDENCY_NAMES}
    clean["static_k27030_policy_artifact"] = stage_a.STATIC_K27030_METHOD.encode()
    clean["static_k29334_policy_artifact"] = stage_a.PRIMARY_K29334_METHOD.encode()
    clean["static_mse_k29334_policy_artifact"] = stage_a.MSE_K29334_METHOD.encode()
    clean["static_fisher_k29334_policy_artifact"] = stage_a.FISHER_K29334_METHOD.encode()
    identity = SimpleNamespace(
        calibration_binding={"calibration_identity_file_sha256": _digest("calibration-identity")},
        tokenizer_manifest_sha256=_digest("tokenizer"),
    )
    methods = stage_a.reconstruct_stage_a_methods(
        dependency_bytes=clean,
        frozen_stage_a_identity=identity,
        source_commit="1" * 40,
    )
    origins = {method.method_id: method.origin for method in methods}
    assert tuple(method.method_id for method in methods) == stage_a.METHOD_ORDER
    assert origins[stage_a.Q48_METHOD] == "reconstructed_candidate_scores_p14739"
    assert origins[stage_a.UNIFORM_Q4_METHOD] == "reconstructed_candidate_scores_k0"
    assert origins[stage_a.UNIFORM_Q8_METHOD] == "reconstructed_candidate_scores_k8"


def test_authorization_execution_mismatch_fails_before_model_file_touch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    h0 = "1" * 40
    identity_path = tmp_path / "identity.json"
    binding_path = tmp_path / "binding.json"
    receipt_path = tmp_path / "capture-receipt.json"
    identity_bytes = b"identity"
    binding_bytes = b"binding"
    receipt_bytes = b"capture receipt"
    execution = {
        "repository_source_manifest_file_sha256": b"source",
        "calibration_runtime_manifest_file_sha256": b"runtime",
        "model_file_manifest_file_sha256": b"model",
        "parquet_materialization_manifest_file_sha256": b"parquet",
    }
    execution_hashes = {name: stage_a.sha256_bytes(payload) for name, payload in execution.items()}
    wrong_authorized = dict(execution_hashes)
    wrong_authorized["model_file_manifest_file_sha256"] = _digest("wrong-model-chain")
    bootstrap = SimpleNamespace(
        execution_bindings=MappingProxyType(execution_hashes),
        calibration_binding=MappingProxyType({}),
        file_sha256=_digest("identity"),
        identity_input_file_sha256=_digest("identity-input"),
        stage_a_capture_provenance_receipt_file_sha256=stage_a.sha256_bytes(receipt_bytes),
    )
    config = SimpleNamespace(
        frozen_identity_path=identity_path,
        calibration_binding_path=binding_path,
        stage_a_capture_provenance_receipt_path=receipt_path,
        expected_stage_a_capture_provenance_receipt_sha256=stage_a.sha256_bytes(receipt_bytes),
        expected_runtime_manifest_sha256=execution_hashes[
            "calibration_runtime_manifest_file_sha256"
        ],
        expected_model_file_manifest_sha256=execution_hashes["model_file_manifest_file_sha256"],
        expected_parquet_materialization_manifest_sha256=execution_hashes[
            "parquet_materialization_manifest_file_sha256"
        ],
        source_commit=h0,
        repository_root=tmp_path,
        git_executable_path=tmp_path / "git.exe",
    )
    source_bootstrap = {"document": {}, "source_commit": h0}
    source_module = SimpleNamespace(
        validate_experiment013_source_manifest=lambda _document: {"source_commit": h0},
        verify_experiment013_source_manifest=lambda normalized, *_args, **_kwargs: normalized,
    )
    model_touched = False

    class FakeRunner:
        @staticmethod
        def authenticate_local_model_files(*_args: object, **_kwargs: object) -> None:
            nonlocal model_touched
            model_touched = True
            raise AssertionError("model files must not be touched")

    resolver = SimpleNamespace(
        deserialize_stage_a_calibration_binding_artifact=lambda _data: SimpleNamespace(
            execution_bindings=wrong_authorized,
            source_commit=h0,
        ),
        deserialize_stage_a_capture_provenance_receipt=lambda *_args, **_kwargs: object(),
    )

    monkeypatch.setattr(stage_a, "_assert_output_paths_isolated", lambda _config: None)
    monkeypatch.setattr(stage_a, "bootstrap_stage_a_identity", lambda _data: bootstrap)
    monkeypatch.setattr(
        stage_a,
        "bootstrap_stage_a_capture_provenance_receipt",
        lambda *_args, **_kwargs: {"capture_source": {"sha256": _digest("capture-source")}},
    )
    monkeypatch.setattr(stage_a, "_read_execution_artifacts", lambda _config: execution)
    monkeypatch.setattr(stage_a, "_bootstrap_source_manifest", lambda _data: source_bootstrap)
    monkeypatch.setattr(stage_a, "_verify_source_bytes", lambda *_args: None)
    monkeypatch.setattr(
        stage_a,
        "_source_entries",
        lambda _source: {stage_a.CAPTURE_SOURCE_PATH: {"raw_sha256": _digest("capture-source")}},
    )
    monkeypatch.setattr(stage_a, "_install_source_namespace", lambda _root: None)
    monkeypatch.setattr(stage_a, "_assert_tracked_identity_bytes", lambda *_args: None)

    def stable_bytes(path: Path, *, context: str) -> bytes:
        del context
        if path == identity_path:
            return identity_bytes
        if path == binding_path:
            return binding_bytes
        assert path == receipt_path
        return receipt_bytes

    monkeypatch.setattr(stage_a, "_stable_file_bytes", stable_bytes)

    def load_module(_name: str, relative_path: str, **_kwargs: object) -> object:
        if relative_path == stage_a.SOURCE_MODULE_PATH:
            return source_module
        if relative_path == stage_a.CALIBRATION_RUNNER_SOURCE_PATH:
            return FakeRunner()
        if relative_path == stage_a.RESOLVER_SOURCE_PATH:
            return resolver
        raise AssertionError(f"unexpected module load: {relative_path}")

    monkeypatch.setattr(stage_a, "_load_exact_module", load_module)
    with pytest.raises(stage_a.StageAError, match="different execution artifacts"):
        stage_a.authenticate_production(config, require_input_bundle=False)

    assert model_touched is False


def _run(root: Path, *args: str, input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_bytes,
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode().strip()


def _git_config(tmp_path: Path) -> tuple[Any, bytes]:
    _run(tmp_path, "init")
    _run(tmp_path, "config", "user.email", "stage-a@example.invalid")
    _run(tmp_path, "config", "user.name", "Stage A Test")
    _run(tmp_path, "config", "core.autocrlf", "false")
    identity_bytes = _identity_bytes()
    identity = tmp_path / "identity.json"
    identity.write_bytes(identity_bytes)
    (tmp_path / ".gitignore").write_text("out/\n", encoding="utf-8")
    _run(tmp_path, "add", "identity.json", ".gitignore")
    _run(tmp_path, "commit", "-m", "identity authorization")
    commit = _run(tmp_path, "rev-parse", "HEAD")
    config = stage_a.StageAConfig(
        frozen_identity_path=identity,
        calibration_binding_path=tmp_path / "binding.json",
        stage_a_capture_provenance_receipt_path=tmp_path / "capture-receipt.json",
        repository_source_manifest_path=tmp_path / "source.json",
        runtime_manifest_path=tmp_path / "runtime.json",
        model_file_manifest_path=tmp_path / "model.json",
        parquet_materialization_manifest_path=tmp_path / "parquet.json",
        model_root=tmp_path / "model-root",
        cache_root=tmp_path,
        ruler_root=tmp_path,
        input_bundle_root=tmp_path / "input-bundle",
        repository_root=tmp_path,
        source_commit=commit,
        identity_commit=commit,
        output_dir=tmp_path / "out",
        expected_runtime_manifest_sha256=_digest("runtime"),
        expected_model_file_manifest_sha256=_digest("model"),
        expected_parquet_materialization_manifest_sha256=_digest("parquet"),
        expected_stage_a_capture_provenance_receipt_sha256=(
            _bootstrap().stage_a_capture_provenance_receipt_file_sha256
        ),
    )
    return config, identity_bytes


def test_one_run_receipt_precedes_empty_diff_commit_cas_and_recovery(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = _authenticated()
    authenticated = stage_a.dataclasses.replace(
        authenticated,
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    assert _run(tmp_path, "rev-parse", "HEAD") == reservation.seal_commit
    assert _run(tmp_path, "show", "-s", "--format=%P", reservation.seal_commit) == (
        reservation.h1_commit
    )
    assert _run(tmp_path, "show", "-s", "--format=%T", reservation.seal_commit) == (
        reservation.tree
    )
    assert reservation.receipt["status"] == "reserved_before_stage_a_content_access"
    assert reservation.receipt["automatic_retry_authorized"] is False
    lock_path = stage_a._identity_attempt_lock_path(
        tmp_path,
        authenticated.bootstrap_identity.file_sha256,
        git_executable_path=config.git_executable_path,
    )
    assert lock_path.is_file()
    assert (
        hashlib.sha256(lock_path.read_bytes()).hexdigest()
        == reservation.receipt["identity_scoped_attempt_lock_file_sha256"]
    )
    with pytest.raises(stage_a.StageAError, match="already exists"):
        stage_a.reserve_one_run(config, authenticated)

    monkeypatch.setattr(
        stage_a,
        "_authenticate_recovery_boundary",
        lambda _config, _receipt, **_kwargs: (
            authenticated.bootstrap_identity,
            authenticated.binding.file_sha256,
            reservation.seal_commit,
        ),
    )
    recovered = stage_a.recover_interrupted(config)
    assert recovered == {
        "status": "consumed_attempt_interrupted_no_result",
        "result_available": False,
        "automatic_retry_authorized": False,
    }


def test_one_run_is_not_reopened_by_fresh_output_or_head_reset(tmp_path: Path) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)

    seal_as_h1 = stage_a.dataclasses.replace(
        config,
        identity_commit=reservation.seal_commit,
        output_dir=config.output_dir / "retry-from-seal",
    )
    with pytest.raises(stage_a.StageAError, match="one-run seal in Git history"):
        stage_a.reserve_one_run(seal_as_h1, authenticated)

    _run(tmp_path, "update-ref", "HEAD", reservation.h1_commit)
    reset_to_original_h1 = stage_a.dataclasses.replace(
        config,
        output_dir=config.output_dir / "retry-after-reset",
    )
    with pytest.raises(stage_a.StageAError, match="one-run seal in Git history"):
        stage_a.reserve_one_run(reset_to_original_h1, authenticated)


def test_post_reservation_reauthentication_rejects_same_tree_seal_substitution(
    tmp_path: Path,
) -> None:
    config, identity_bytes = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    alternate = _run(
        tmp_path,
        "commit-tree",
        reservation.tree,
        "-p",
        reservation.h1_commit,
        input_bytes=b"unrelated empty child\n",
    )
    _run(tmp_path, "update-ref", "HEAD", alternate, reservation.seal_commit)

    with pytest.raises(stage_a.StageAError, match="reserved one-run seal"):
        stage_a._assert_tracked_identity_bytes_after_seal(
            config,
            identity_bytes,
            authenticated,
            reservation,
        )


def test_reservation_rejects_linked_or_reparse_output_parent(tmp_path: Path) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    actual_output = tmp_path / "actual-output"
    actual_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        os.symlink(actual_output, linked_output, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")
    linked_config = stage_a.dataclasses.replace(config, output_dir=linked_output)

    with pytest.raises(stage_a.StageAError, match="link or reparse"):
        stage_a.reserve_one_run(linked_config, authenticated)

    assert not (actual_output / stage_a.ATTEMPT_FILENAME).exists()


def test_pre_cas_recovery_branch_remains_reachable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    original = stage_a._git_process

    def fail_cas(
        git_executable_path: Path | None,
        root: Path,
        *arguments: str,
        input_bytes: bytes | None = None,
    ) -> Any:
        if arguments and arguments[0] == "update-ref":
            return subprocess.CompletedProcess(
                ["git", *arguments],
                1,
                b"",
                b"injected CAS failure",
            )
        return original(
            git_executable_path,
            root,
            *arguments,
            input_bytes=input_bytes,
        )

    monkeypatch.setattr(stage_a, "_git_process", fail_cas)
    with pytest.raises(stage_a.StageAError, match="compare-and-swap failed"):
        stage_a.reserve_one_run(config, authenticated)
    monkeypatch.setattr(stage_a, "_git_process", original)
    receipt = stage_a._strict_json(config.attempt_path.read_bytes(), context="test receipt")
    seal = receipt["one_run_seal_commit"]
    observed: dict[str, object] = {}

    def authenticate_boundary(_config: Any, _receipt: Any, **kwargs: Any) -> Any:
        observed.update(kwargs)
        return authenticated.bootstrap_identity, authenticated.binding.file_sha256, seal

    monkeypatch.setattr(stage_a, "_authenticate_recovery_boundary", authenticate_boundary)
    recovered = stage_a.recover_interrupted(config)
    assert observed == {"allow_pre_cas_head": True}
    assert recovered["status"] == "pre_cas_attempt_receipt_present_no_automatic_retry"
    assert recovered["result_available"] is False
    receipt_before = config.attempt_path.read_bytes()
    assert stage_a.recover_interrupted(config) == recovered
    assert config.attempt_path.read_bytes() == receipt_before


def test_lock_only_crash_is_administratively_recovered_without_retry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    original_write = stage_a._exclusive_write
    injected = False

    def fail_attempt_write(path: Path, payload: bytes) -> None:
        nonlocal injected
        if Path(path) == config.attempt_path and not injected:
            injected = True
            raise OSError("injected attempt-receipt write failure")
        original_write(path, payload)

    monkeypatch.setattr(stage_a, "_exclusive_write", fail_attempt_write)
    with pytest.raises(OSError, match="attempt-receipt"):
        stage_a.reserve_one_run(config, authenticated)
    monkeypatch.setattr(stage_a, "_exclusive_write", original_write)
    assert not config.attempt_path.exists()
    assert _run(tmp_path, "rev-parse", "HEAD") == config.identity_commit
    lock_path = stage_a._identity_attempt_lock_path(
        tmp_path,
        authenticated.bootstrap_identity.file_sha256,
        git_executable_path=config.git_executable_path,
    )
    lock = stage_a._strict_json(lock_path.read_bytes(), context="test attempt lock")
    observed: list[bool] = []

    def authenticate_boundary(_config: Any, _receipt: Any, **kwargs: Any) -> Any:
        observed.append(kwargs.get("allow_pre_cas_head") is True)
        return (
            authenticated.bootstrap_identity,
            authenticated.binding.file_sha256,
            lock["one_run_seal_commit"],
        )

    monkeypatch.setattr(stage_a, "_authenticate_recovery_boundary", authenticate_boundary)
    recovered = stage_a.recover_interrupted(config)
    assert recovered["status"] == "pre_cas_attempt_receipt_present_no_automatic_retry"
    assert recovered["automatic_retry_authorized"] is False
    assert observed == [True, True]
    receipt = stage_a._strict_json(config.attempt_path.read_bytes(), context="recovered receipt")
    assert "lock_only_recovered_at_utc" in receipt


def test_recovery_rejects_missing_completed_result_and_preserves_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    monkeypatch.setattr(
        stage_a,
        "_authenticate_recovery_boundary",
        lambda _config, _receipt, **_kwargs: (
            authenticated.bootstrap_identity,
            authenticated.binding.file_sha256,
            reservation.seal_commit,
        ),
    )
    completed = stage_a.persist_receipt(
        config,
        reservation,
        {
            "status": "completed_with_authenticated_stage_a_result",
            "result_available": True,
        },
    )
    with pytest.raises(stage_a.StageAError, match="missing its published result"):
        stage_a.recover_interrupted(config)

    stage_a.persist_receipt(
        config,
        completed,
        {
            "status": "consumed_attempt_failed_no_automatic_retry",
            "result_available": False,
        },
    )
    before = config.attempt_path.read_bytes()
    recovered = stage_a.recover_interrupted(config)
    assert recovered["status"] == "consumed_attempt_failed_no_automatic_retry"
    assert config.attempt_path.read_bytes() == before


def test_real_recovery_boundary_authenticates_final_lock_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = stage_a.importlib.import_module("recurquant.experiment013_source")
    git_executable = stage_a._authenticated_git_executable(None)
    _run(tmp_path, "init")
    _run(tmp_path, "config", "user.email", "stage-a@example.invalid")
    _run(tmp_path, "config", "user.name", "Stage A Test")
    _run(tmp_path, "config", "core.autocrlf", "false")
    for relative in source.EXPERIMENT013_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"authenticated fixture for {relative}\n".encode())
    (tmp_path / ".gitignore").write_text("evidence/\nout/\n", encoding="utf-8")
    _run(tmp_path, "add", ".")
    _run(tmp_path, "commit", "-m", "frozen Experiment 013 source")
    h0 = _run(tmp_path, "rev-parse", "HEAD")

    source_manifest = source.capture_experiment013_source_manifest(
        tmp_path,
        git_executable=git_executable,
    )
    source_bytes = source.canonical_experiment013_source_manifest_bytes(source_manifest)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    artifact_bytes = {
        "repository_source_manifest_file_sha256": source_bytes,
        "calibration_runtime_manifest_file_sha256": b"runtime manifest fixture\n",
        "model_file_manifest_file_sha256": b"model manifest fixture\n",
        "parquet_materialization_manifest_file_sha256": b"parquet manifest fixture\n",
    }
    artifact_paths = {
        "repository_source_manifest_file_sha256": evidence_dir / "source.json",
        "calibration_runtime_manifest_file_sha256": evidence_dir / "runtime.json",
        "model_file_manifest_file_sha256": evidence_dir / "model.json",
        "parquet_materialization_manifest_file_sha256": evidence_dir / "parquet.json",
    }
    for name, payload in artifact_bytes.items():
        artifact_paths[name].write_bytes(payload)
    binding_bytes = b"calibration binding fixture\n"
    binding_path = evidence_dir / "binding.json"
    binding_path.write_bytes(binding_bytes)
    execution_bindings = {
        name: stage_a.sha256_bytes(payload) for name, payload in artifact_bytes.items()
    }
    capture_source_sha256 = stage_a.sha256_bytes(
        (tmp_path / stage_a.CAPTURE_SOURCE_PATH).read_bytes()
    )
    capture_receipt_bytes = _capture_receipt_bytes(
        execution_bindings,
        binding_bytes=binding_bytes,
        source_commit=h0,
        capture_source_sha256=capture_source_sha256,
    )
    capture_receipt_path = evidence_dir / "stage-a-capture-provenance.json"
    capture_receipt_path.write_bytes(capture_receipt_bytes)
    identity_bytes = _identity_bytes(
        execution_bindings,
        capture_receipt_sha256=stage_a.sha256_bytes(capture_receipt_bytes),
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(identity_bytes)
    _run(tmp_path, "add", "identity.json")
    _run(tmp_path, "commit", "-m", "authorize Stage-A identity")
    h1 = _run(tmp_path, "rev-parse", "HEAD")

    config = stage_a.StageAConfig(
        frozen_identity_path=identity_path,
        calibration_binding_path=binding_path,
        stage_a_capture_provenance_receipt_path=capture_receipt_path,
        repository_source_manifest_path=artifact_paths["repository_source_manifest_file_sha256"],
        runtime_manifest_path=artifact_paths["calibration_runtime_manifest_file_sha256"],
        model_file_manifest_path=artifact_paths["model_file_manifest_file_sha256"],
        parquet_materialization_manifest_path=artifact_paths[
            "parquet_materialization_manifest_file_sha256"
        ],
        model_root=tmp_path / "model-root",
        cache_root=tmp_path / "cache-root",
        ruler_root=tmp_path / "ruler-root",
        input_bundle_root=tmp_path / "input-bundle",
        repository_root=tmp_path,
        source_commit=h0,
        identity_commit=h1,
        output_dir=tmp_path / "out",
        expected_runtime_manifest_sha256=execution_bindings[
            "calibration_runtime_manifest_file_sha256"
        ],
        expected_model_file_manifest_sha256=execution_bindings["model_file_manifest_file_sha256"],
        expected_parquet_materialization_manifest_sha256=execution_bindings[
            "parquet_materialization_manifest_file_sha256"
        ],
        expected_stage_a_capture_provenance_receipt_sha256=stage_a.sha256_bytes(
            capture_receipt_bytes
        ),
        git_executable_path=git_executable,
    )
    bootstrap = stage_a.bootstrap_stage_a_identity(identity_bytes)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        bootstrap_identity=bootstrap,
        binding=SimpleNamespace(file_sha256=stage_a.sha256_bytes(binding_bytes)),
        capture_provenance_receipt=SimpleNamespace(
            file_sha256=stage_a.sha256_bytes(capture_receipt_bytes)
        ),
        capture_provenance_receipt_bytes=capture_receipt_bytes,
        execution_artifact_bytes=MappingProxyType(artifact_bytes),
        source_manifest=MappingProxyType(source_manifest),
        source_manifest_file_sha256=stage_a.sha256_bytes(source_bytes),
        source_commit=h0,
        input_bundle_manifest_file_sha256=_digest("real-input-bundle"),
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    lock_path = stage_a._identity_attempt_lock_path(
        tmp_path,
        bootstrap.file_sha256,
        git_executable_path=git_executable,
    )
    lock = stage_a._strict_json(lock_path.read_bytes(), context="real recovery lock")
    assert lock["schema"] == stage_a.IDENTITY_ATTEMPT_LOCK_SCHEMA
    assert reservation.receipt["schema"] == stage_a.ATTEMPT_SCHEMA
    assert reservation.receipt["stage_a_input_bundle_manifest_file_sha256"] == _digest(
        "real-input-bundle"
    )

    recovered = stage_a.recover_interrupted(config)
    receipt_after_first_recovery = config.attempt_path.read_bytes()
    recovered_again = stage_a.recover_interrupted(config)

    assert recovered == {
        "status": "consumed_attempt_interrupted_no_result",
        "result_available": False,
        "automatic_retry_authorized": False,
    }
    assert recovered_again == recovered
    assert config.attempt_path.read_bytes() == receipt_after_first_recovery
    assert _run(tmp_path, "rev-parse", "HEAD") == reservation.seal_commit


def _test_execution_artifact(
    reservation: Any,
    authenticated: Any,
) -> bytes:
    materialization = _materialization()
    evaluation = stage_a.evaluate_materialized_stage_a(
        authenticated,
        materialization,
        _Engine(),
        object(),
    )
    gate = stage_a.importlib.import_module("recurquant.experiment013_stage_a")
    gate_bytes = gate.build_stage_a_evidence_artifact(
        evaluation.examples,
        evaluation.gate_rows,
        stage_a_identity_file_sha256=authenticated.bootstrap_identity.file_sha256,
        stage_a_calibration_binding_file_sha256=authenticated.binding.file_sha256,
    )
    gate_artifact = stage_a._strict_json(gate_bytes, context="test gate")
    verified_gate = gate.deserialize_stage_a_evidence_artifact(gate_bytes)
    smoke = _smoke_report()
    smoke_sha256 = stage_a.sha256_bytes(stage_a.canonical_json_bytes(smoke))
    runtime_record = stage_a._authenticated_runtime_record(
        authenticated.authenticated_runtime,
        expected_manifest_file_sha256=authenticated.bootstrap_identity.execution_bindings[
            "calibration_runtime_manifest_file_sha256"
        ],
    )
    evidence = {
        "artifact_revision": stage_a.RUNNER_REVISION,
        "claim_boundary": stage_a.CLAIM_BOUNDARY,
        "dependencies": {
            "stage_a_identity_file_sha256": authenticated.bootstrap_identity.file_sha256,
            "stage_a_calibration_binding_file_sha256": authenticated.binding.file_sha256,
            stage_a.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
                authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
            ),
            "repository_source_manifest_file_sha256": authenticated.source_manifest_file_sha256,
            "stage_a_input_bundle_manifest_file_sha256": (
                authenticated.input_bundle_manifest_file_sha256
            ),
            "execution_bindings": dict(authenticated.bootstrap_identity.execution_bindings),
            "method_specs": stage_a._method_spec_receipts(authenticated.methods),
        },
        "execution_contract": stage_a._expected_execution_contract(evaluation.forward_count),
        "materialization": stage_a._materialization_receipt(materialization),
        "method_runtime": [dict(row) for row in evaluation.method_runtime],
        "one_run": {
            "attempt_schema": stage_a.ATTEMPT_SCHEMA,
            "automatic_retry_authorized": False,
            "h0_source_commit": authenticated.source_commit,
            "h1_identity_commit": reservation.h1_commit,
            "identity_scoped_attempt_lock_file_sha256": reservation.receipt[
                "identity_scoped_attempt_lock_file_sha256"
            ],
            "one_run_marker": stage_a.ONE_RUN_MARKER,
            "one_run_seal_commit": reservation.seal_commit,
            "one_run_seal_message_sha256": reservation.receipt["one_run_seal_message_sha256"],
            "one_run_seal_tree": reservation.tree,
            "preseal_engine_smoke_sha256": smoke_sha256,
            stage_a.STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: (
                authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
            ),
            "stage_a_input_bundle_manifest_file_sha256": (
                authenticated.input_bundle_manifest_file_sha256
            ),
        },
        "preseal_engine_smoke": smoke,
        "raw_token_evidence": [dict(row) for row in evaluation.raw_rows],
        "runtime": {
            "authenticated_runtime": dict(runtime_record),
            "device": _Engine().runtime_snapshot(object()),
        },
        "stage_a_gate_artifact": gate_artifact,
        "stage_a_gate_file_sha256": verified_gate.file_sha256,
        "stage_a_passed": verified_gate.passed,
    }
    return stage_a.canonical_json_bytes(
        {
            "artifact_kind": stage_a.EXECUTION_ARTIFACT_KIND,
            "schema_version": stage_a.EXECUTION_ARTIFACT_SCHEMA,
            "canonical_evidence_sha256": stage_a.sha256_bytes(
                stage_a.canonical_json_bytes(evidence)
            ),
            "evidence": evidence,
        }
    )


def _verification_kwargs(reservation: Any, authenticated: Any) -> dict[str, Any]:
    return {
        "expected_identity_file_sha256": authenticated.bootstrap_identity.file_sha256,
        "expected_calibration_binding_file_sha256": authenticated.binding.file_sha256,
        "expected_stage_a_capture_provenance_receipt_file_sha256": (
            authenticated.bootstrap_identity.stage_a_capture_provenance_receipt_file_sha256
        ),
        "expected_h1_commit": reservation.h1_commit,
        "expected_seal_commit": reservation.seal_commit,
        "expected_source_commit": authenticated.source_commit,
        "expected_source_manifest_file_sha256": authenticated.source_manifest_file_sha256,
        "expected_input_bundle_manifest_file_sha256": (
            authenticated.input_bundle_manifest_file_sha256
        ),
        "expected_execution_bindings": authenticated.bootstrap_identity.execution_bindings,
        "expected_method_specs": stage_a._method_spec_receipts(authenticated.methods),
        "expected_materialization": stage_a._materialization_receipt(_materialization()),
        "expected_seal_tree": reservation.tree,
        "expected_seal_message_sha256": reservation.receipt["one_run_seal_message_sha256"],
        "expected_attempt_lock_file_sha256": reservation.receipt[
            "identity_scoped_attempt_lock_file_sha256"
        ],
        "expected_authenticated_runtime": stage_a._authenticated_runtime_record(
            authenticated.authenticated_runtime,
            expected_manifest_file_sha256=authenticated.bootstrap_identity.execution_bindings[
                "calibration_runtime_manifest_file_sha256"
            ],
        ),
        "expected_device_runtime": _Engine().runtime_snapshot(object()),
        "expected_forward_count": authenticated.bootstrap_identity.expected_forward_count,
    }


def test_execution_artifact_verifier_binds_seal_runtime_and_redaction(tmp_path: Path) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    payload = _test_execution_artifact(reservation, authenticated)
    verified = stage_a.verify_execution_artifact(
        payload,
        **_verification_kwargs(reservation, authenticated),
    )
    assert verified["schema_version"] == 4

    tampered = stage_a._strict_json(payload, context="test result")
    tampered["evidence"]["raw_token_evidence"][0]["target_token_ids_sha256"] = _digest(
        "reversible-token"
    )
    tampered["canonical_evidence_sha256"] = stage_a.sha256_bytes(
        stage_a.canonical_json_bytes(tampered["evidence"])
    )
    with pytest.raises(stage_a.StageAError, match="low-entropy"):
        stage_a.verify_execution_artifact(
            stage_a.canonical_json_bytes(tampered),
            **_verification_kwargs(reservation, authenticated),
        )


def test_execution_artifact_verifier_rejects_forged_outer_diagnostics(
    tmp_path: Path,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    payload = _test_execution_artifact(reservation, authenticated)
    mutations = (
        (
            lambda root: root["evidence"]["raw_token_evidence"][0].__setitem__(
                "decode_model_forward_latency_ns", -1
            ),
            "nonnegative",
        ),
        (
            lambda root: root["evidence"]["method_runtime"][0].__setitem__(
                "policy_origin", "forged"
            ),
            "policy identity",
        ),
        (
            lambda root: root["evidence"]["method_runtime"][0]["storage"].__setitem__(
                "forged", "accepted"
            ),
            "frozen schema",
        ),
        (
            lambda root: root["evidence"]["materialization"].__setitem__(
                "capture_input_sha256", _digest("forged-capture")
            ),
            "materialization receipt drifted",
        ),
        (
            lambda root: root["evidence"]["runtime"]["authenticated_runtime"].__setitem__(
                "machine_name", "FORGED_MACHINE"
            ),
            "authenticated runtime evidence drifted",
        ),
        (
            lambda root: root["evidence"]["runtime"]["device"].__setitem__("name", "FORGED_GPU"),
            "device runtime evidence drifted",
        ),
    )
    for mutate, message in mutations:
        forged = stage_a._strict_json(payload, context="forged result")
        mutate(forged)
        forged["canonical_evidence_sha256"] = stage_a.sha256_bytes(
            stage_a.canonical_json_bytes(forged["evidence"])
        )
        with pytest.raises(stage_a.StageAError, match=message):
            stage_a.verify_execution_artifact(
                stage_a.canonical_json_bytes(forged),
                **_verification_kwargs(reservation, authenticated),
            )


def test_execution_artifact_builder_emits_self_verifying_bundle(tmp_path: Path) -> None:
    config, _identity = _git_config(tmp_path)
    base = _authenticated()
    authenticated = stage_a.dataclasses.replace(
        base,
        source_commit=config.source_commit,
        authenticated_runtime=_runtime_namespace(),
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    materialization = _materialization()
    evaluation = stage_a.evaluate_materialized_stage_a(
        authenticated,
        materialization,
        _Engine(),
        object(),
    )
    smoke = _smoke_report()
    reservation = stage_a.persist_receipt(
        config,
        reservation,
        {
            "status": "preseal_engine_smoke_bound_before_materialization",
            "preseal_engine_smoke": smoke,
            "preseal_engine_smoke_sha256": stage_a.sha256_bytes(
                stage_a.canonical_json_bytes(smoke)
            ),
            "post_load_authenticated_runtime": _runtime_record(),
            "post_load_device_runtime": _Engine().runtime_snapshot(object()),
        },
    )

    payload = stage_a.build_execution_artifact(
        authenticated,
        materialization,
        evaluation,
        reservation,
    )
    result = stage_a._strict_json(payload, context="built result")
    assert result["evidence"]["one_run"]["one_run_seal_commit"] == reservation.seal_commit
    assert result["evidence"]["runtime"]["device"]["name"] == "Test GPU"


@pytest.mark.parametrize("completed_receipt_before_crash", (False, True))
def test_recovery_finishes_missing_completion_marker_without_reevaluation(
    tmp_path: Path,
    completed_receipt_before_crash: bool,
    monkeypatch: Any,
) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    payload = _test_execution_artifact(reservation, authenticated)
    canonical_hash = stage_a._strict_json(payload, context="test result")[
        "canonical_evidence_sha256"
    ]
    smoke = _smoke_report()
    updates: dict[str, Any] = {
        "status": "result_prepared_before_atomic_publication",
        "result_available": False,
        "capture_input_sha256": _materialization().capture_input_sha256,
        "token_sequence_manifest_sha256": _materialization().token_sequence_manifest_sha256,
        "tokenizer_manifest_sha256": _materialization().tokenizer_manifest_sha256,
        "preseal_engine_smoke": smoke,
        "preseal_engine_smoke_sha256": stage_a.sha256_bytes(stage_a.canonical_json_bytes(smoke)),
        "post_load_authenticated_runtime": _runtime_record(),
        "post_load_device_runtime": _Engine().runtime_snapshot(object()),
        "result_file_sha256": stage_a.sha256_bytes(payload),
        "result_canonical_evidence_sha256": canonical_hash,
    }
    if completed_receipt_before_crash:
        updates.update(
            {
                "status": "completed_with_authenticated_stage_a_result",
                "result_available": True,
            }
        )
    stage_a.persist_receipt(config, reservation, updates)
    stage_a._atomic_publish_new(config.output_path, payload)
    assert not config.complete_path.exists()
    monkeypatch.setattr(
        stage_a,
        "_authenticate_recovery_boundary",
        lambda _config, _receipt, **_kwargs: (
            authenticated.bootstrap_identity,
            authenticated.binding.file_sha256,
            reservation.seal_commit,
        ),
    )

    recovered = stage_a.recover_interrupted(config)

    assert recovered["result_available"] is True
    assert recovered["completion_marker_available"] is True
    marker = stage_a._strict_json(config.complete_path.read_bytes(), context="test marker")
    assert marker["result_file_sha256"] == stage_a.sha256_bytes(payload)
    assert marker["attempt_file_sha256"] == stage_a.sha256_bytes(config.attempt_path.read_bytes())
    attempt_before = config.attempt_path.read_bytes()
    marker_before = config.complete_path.read_bytes()
    assert stage_a.recover_interrupted(config) == recovered
    assert config.attempt_path.read_bytes() == attempt_before
    assert config.complete_path.read_bytes() == marker_before


def test_failure_recording_rebases_stale_in_memory_receipt(tmp_path: Path) -> None:
    config, _identity = _git_config(tmp_path)
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    reservation = stage_a.reserve_one_run(config, authenticated)
    stage_a.persist_receipt(
        config,
        reservation,
        {"status": "result_prepared_before_atomic_publication"},
    )

    stage_a.record_failure(config, reservation, RuntimeError("injected"), "publication")

    receipt = stage_a._strict_json(config.attempt_path.read_bytes(), context="test receipt")
    assert receipt["status"] == "consumed_attempt_failed_no_automatic_retry"
    assert receipt["failure_phase"] == "publication"


def test_atomic_publication_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    stage_a._atomic_publish_new(destination, b"first")
    with pytest.raises(stage_a.StageAError, match="overwrite"):
        stage_a._atomic_publish_new(destination, b"second")
    assert destination.read_bytes() == b"first"


class _CausalCache:
    def __init__(self, length: int = 0) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def storage_summary(self) -> dict[str, int]:
        return {"resident_bytes": 0}


class _CausalModel:
    def __init__(
        self,
        cache: _CausalCache,
        *,
        advance: bool = True,
        replace_cache: bool = False,
    ) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.cache = cache
        self.advance = advance
        self.replace_cache = replace_cache
        self.call: dict[str, Any] | None = None

    def parameters(self) -> Any:
        yield self.parameter

    def __call__(self, **kwargs: Any) -> Any:
        self.call = kwargs
        if self.advance:
            self.cache.length += int(kwargs["input_ids"].shape[1])
        returned = _CausalCache(self.cache.length) if self.replace_cache else self.cache
        return SimpleNamespace(
            logits=torch.zeros((1, 1, 8), dtype=torch.float32),
            past_key_values=returned,
        )


def _disable_cuda_measurements(monkeypatch: Any) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)


def test_device_runtime_rejects_fallback_model_dtype() -> None:
    runtime = _Engine().runtime_snapshot(object())
    runtime["model_parameter_dtype"] = "torch.float32"
    with pytest.raises(stage_a.StageAError, match="BF16 dtype contract"):
        stage_a._validated_device_runtime(runtime)


def test_loaded_model_contract_rejects_fallback_parameter_dtype() -> None:
    class Config:
        def __init__(self) -> None:
            self._attn_implementation = "eager"
            self._attn_implementation_internal = "eager"

    class Model:
        def __init__(self, dtype: torch.dtype) -> None:
            self.config = Config()
            self.parameter = torch.nn.Parameter(torch.zeros((), dtype=dtype))

        def parameters(self) -> Any:
            yield self.parameter

    transformers = SimpleNamespace(
        Qwen3_5ForCausalLM=Model,
        Qwen3_5TextConfig=Config,
    )
    device = torch.device("cpu")
    stage_a.TorchStageAEngine._assert_loaded_model_contract(
        Model(torch.bfloat16),
        torch=torch,
        transformers=transformers,
        device=device,
    )
    with pytest.raises(stage_a.StageAError, match="is not BF16"):
        stage_a.TorchStageAEngine._assert_loaded_model_contract(
            Model(torch.float32),
            torch=torch,
            transformers=transformers,
            device=device,
        )
    sdpa_model = Model(torch.bfloat16)
    sdpa_model.config._attn_implementation = "sdpa"
    sdpa_model.config._attn_implementation_internal = "sdpa"
    with pytest.raises(stage_a.StageAError, match="retain eager attention"):
        stage_a.TorchStageAEngine._assert_loaded_model_contract(
            sdpa_model,
            torch=torch,
            transformers=transformers,
            device=device,
        )


def _dynamic_recurrent_cache(dtype: torch.dtype, *, complete: bool) -> object:
    static = stage_a.importlib.import_module("recurquant.static_q468")
    geometry = static.FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    shape = (1, geometry.heads, geometry.key_rows, geometry.value_width)
    expected = set(static.FROZEN_RECURRENT_LAYER_INDICES)
    layer_count = max(expected) + 1
    layers: list[object] = []
    for layer_index in range(layer_count):
        if layer_index in expected and (complete or layer_index == min(expected)):
            layers.append(
                SimpleNamespace(
                    recurrent_states=[torch.zeros(shape, dtype=dtype)],
                    is_recurrent_states_initialized=[True],
                )
            )
        else:
            layers.append(SimpleNamespace(recurrent_states=None))
    return SimpleNamespace(layers=layers)


def _packed_recurrent_cache(dtype: torch.dtype, *, device: torch.device) -> object:
    static = stage_a.importlib.import_module("recurquant.static_q468")
    geometry = static.FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    shape = (1, geometry.heads, geometry.key_rows, geometry.value_width)
    states = {
        layer_index: torch.zeros(shape, dtype=dtype, device=device)
        for layer_index in static.FROZEN_RECURRENT_LAYER_INDICES
    }
    return SimpleNamespace(
        checkpoint=SimpleNamespace(materialize=lambda: states),
    )


def test_fp32_reference_rejects_bf16_live_cache_state() -> None:
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    engine._model = SimpleNamespace(parameters=lambda: iter([parameter]))
    with pytest.raises(stage_a.StageAError, match="non-FP32 recurrent state"):
        engine._recurrent_states(
            _dynamic_recurrent_cache(torch.bfloat16, complete=False),
            packed=False,
        )


def test_fp32_reference_derives_exact_live_byte_ledger() -> None:
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    engine._model = SimpleNamespace(parameters=lambda: iter([parameter]))
    states = engine._recurrent_states(
        _dynamic_recurrent_cache(torch.float32, complete=True),
        packed=False,
    )
    observed_bytes = sum(tensor.numel() * tensor.element_size() for tensor in states.values())
    assert observed_bytes == stage_a.EXPECTED_RECURRENT_RESIDENT_BYTES[stage_a.FP32_METHOD]


def test_packed_reference_rejects_bf16_live_materialization() -> None:
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    engine._model = SimpleNamespace(parameters=lambda: iter([parameter]))
    with pytest.raises(stage_a.StageAError, match="non-FP32 recurrent state"):
        engine._recurrent_states(
            _packed_recurrent_cache(torch.bfloat16, device=torch.device("cpu")),
            packed=True,
        )


def test_packed_reference_rejects_wrong_live_device() -> None:
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch
    parameter = torch.nn.Parameter(
        torch.empty((), dtype=torch.bfloat16, device=torch.device("meta"))
    )
    engine._model = SimpleNamespace(parameters=lambda: iter([parameter]))
    with pytest.raises(stage_a.StageAError, match="model CUDA device"):
        engine._recurrent_states(
            _packed_recurrent_cache(torch.float32, device=torch.device("cpu")),
            packed=True,
        )


def test_production_engine_binds_explicit_positions_and_cache_advancement(
    monkeypatch: Any,
) -> None:
    _disable_cuda_measurements(monkeypatch)
    cache = _CausalCache()
    model = _CausalModel(cache)
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch

    observation = engine._forward(
        {
            "model": model,
            "cache": cache,
            "method": stage_a.StageAMethodSpec(
                stage_a.FP32_METHOD,
                None,
                None,
                "test",
            ),
        },
        (4, 5, 6),
        target_token_id=2,
        position=2,
        scored=False,
    )

    assert observation.position == 2
    assert cache.length == 3
    assert model.call is not None
    assert model.call["past_key_values"] is cache
    assert model.call["position_ids"].tolist() == [[0, 1, 2]]
    assert model.call["cache_position"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("initial_length", "advance", "replace_cache", "message"),
    (
        (1, True, False, "length drifted before"),
        (0, True, True, "different Stage-A cache"),
        (0, False, False, "did not advance"),
    ),
)
def test_production_engine_rejects_noncausal_cache_behavior(
    monkeypatch: Any,
    initial_length: int,
    advance: bool,
    replace_cache: bool,
    message: str,
) -> None:
    _disable_cuda_measurements(monkeypatch)
    cache = _CausalCache(initial_length)
    model = _CausalModel(cache, advance=advance, replace_cache=replace_cache)
    engine = stage_a.TorchStageAEngine()
    engine._torch = torch

    with pytest.raises(stage_a.StageAError, match=message):
        engine._forward(
            {
                "model": model,
                "cache": cache,
                "method": stage_a.StageAMethodSpec(
                    stage_a.FP32_METHOD,
                    None,
                    None,
                    "test",
                ),
            },
            (4, 5, 6),
            target_token_id=2,
            position=2,
            scored=False,
        )


def test_production_engine_installs_and_removes_equal_byte_observer(
    monkeypatch: Any,
) -> None:
    identity = _digest("observer-lifecycle")
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device="cuda:0")]),
    )
    cache = SimpleNamespace(storage_summary=lambda: {"resident_bytes": 123})
    events: list[str] = []

    class Observer:
        def __init__(self, observed_model: object, *, caches: list[object]) -> None:
            assert observed_model is model
            assert caches == [cache]

        def __enter__(self) -> Any:
            events.append("install")
            return self

        def remove(self) -> None:
            events.append("remove")

    cache_module = SimpleNamespace(
        create_qwen35_static_rht_cache=lambda *args, **kwargs: cache,
    )
    observer_module = SimpleNamespace(Qwen35EqualByteObserver=Observer)
    original_import = stage_a.importlib.import_module

    def fake_import(name: str) -> object:
        if name == "recurquant.static_q468_cache":
            return cache_module
        if name == "recurquant.statelease_equal_byte_cache":
            return observer_module
        return original_import(name)

    monkeypatch.setattr(stage_a.importlib, "import_module", fake_import)
    engine = stage_a.TorchStageAEngine()
    engine._model = model
    engine._torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda _device: None,
            empty_cache=lambda: None,
        )
    )
    engine._reference_states[identity] = [{}, {}]
    policy = SimpleNamespace(policy_sha256=_digest("policy"))
    method = stage_a.StageAMethodSpec(
        stage_a.MSE_K29334_METHOD,
        policy,
        policy.policy_sha256,
        "test",
    )
    sequence = SimpleNamespace(
        identity_record_sha256=identity,
        target_token_ids=(1, 2, 3),
    )

    session = engine.begin_method(model, method, sequence)
    session["step_index"] = 2
    assert engine.end_method(session) == {"resident_bytes": 123}
    assert events == ["install", "remove"]

    events.clear()
    session = engine.begin_method(model, method, sequence)
    with pytest.raises(stage_a.StageAError, match="did not complete"):
        engine.end_method(session)
    assert events == ["install", "remove"]


def test_primary_method_completion_releases_fp32_trajectory() -> None:
    engine = stage_a.TorchStageAEngine()
    identity = _digest("release-reference")
    engine._reference_states[identity] = [{}]
    session = {
        "cache": SimpleNamespace(storage_summary=lambda: {"resident_bytes": 1}),
        "method": stage_a.StageAMethodSpec(
            stage_a.PRIMARY_K29334_METHOD,
            None,
            _digest("primary-policy"),
            "test",
        ),
        "identity_record_sha256": identity,
        "expected_steps": 1,
        "step_index": 1,
    }
    assert engine.end_method(session) == {"resident_bytes": 1}
    assert identity not in engine._reference_states


def test_end_method_releases_completed_cache_before_next_method_boundary() -> None:
    class Cache:
        @staticmethod
        def storage_summary() -> dict[str, int]:
            return {"resident_bytes": 1}

    engine = stage_a.TorchStageAEngine()
    cache = Cache()
    cache_reference = weakref.ref(cache)
    session = {
        "model": object(),
        "cache": cache,
        "observer": None,
        "method": stage_a.StageAMethodSpec(
            stage_a.MSE_K29334_METHOD,
            None,
            _digest("release-cache-policy"),
            "test",
        ),
        "identity_record_sha256": _digest("release-cache-identity"),
        "expected_steps": 1,
        "step_index": 1,
    }
    assert engine.end_method(session) == {"resident_bytes": 1}
    assert session == {}
    del cache
    assert cache_reference() is None


@pytest.mark.parametrize("protected_kind", ("bundle", "model", "base", "package"))
@pytest.mark.parametrize("protected_inside_output", (False, True))
def test_authenticated_root_output_overlap_is_rejected_without_consuming_attempt(
    tmp_path: Path,
    protected_kind: str,
    protected_inside_output: bool,
) -> None:
    config, _identity = _git_config(tmp_path)
    protected_root = tmp_path / f"{protected_kind}-protected-root"
    if protected_inside_output:
        protected_root = config.output_dir / f"{protected_kind}-protected-root"
    else:
        config = stage_a.dataclasses.replace(config, output_dir=protected_root / "output")
    if protected_kind == "bundle":
        config = stage_a.dataclasses.replace(config, input_bundle_root=protected_root)
    elif protected_kind == "model":
        config = stage_a.dataclasses.replace(config, model_root=protected_root)
    elif protected_kind == "base":
        config = stage_a.dataclasses.replace(config, base_runtime_root=protected_root)
    else:
        config = stage_a.dataclasses.replace(
            config,
            package_roots=MappingProxyType({"fixture": protected_root}),
        )
    authenticated = stage_a.dataclasses.replace(
        _authenticated(),
        source_commit=config.source_commit,
    )
    head_before = _run(tmp_path, "rev-parse", "HEAD")
    lock_path = stage_a._identity_attempt_lock_path(
        config.repository_root,
        authenticated.bootstrap_identity.file_sha256,
        git_executable_path=config.git_executable_path,
    )

    with pytest.raises(stage_a.StageAError, match="must not overlap"):
        stage_a.reserve_one_run(config, authenticated)

    assert _run(tmp_path, "rev-parse", "HEAD") == head_before
    assert not lock_path.exists()
    assert not config.attempt_path.exists()
    assert not config.output_path.exists()
    assert not config.complete_path.exists()
