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


def _hooks(
    *,
    authenticate=_unused,
    load_config=_unused,
    reserve_attempt=_unused,
    load_exact_task=_unused,
    tokenize_task=_unused,
    load_weights=_unused,
    evaluate=_unused,
    finalize=_unused,
    record_failure=_unused,
) -> stage_a.AccessHooks:
    return stage_a.AccessHooks(
        authenticate=authenticate,
        load_config=load_config,
        reserve_attempt=reserve_attempt,
        load_exact_task=load_exact_task,
        tokenize_task=tokenize_task,
        load_weights=load_weights,
        evaluate=evaluate,
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
                load_config=reject_config,
            )
        )
    assert calls == ["authenticate", "config"]


def test_token_authentication_failure_happens_after_reservation_but_before_weights() -> None:
    events: list[str] = []

    def reject_tokens(*_args: object) -> object:
        events.append("tokenize")
        raise stage_a.StageAAuthenticationError("token manifest drift")

    def record_failure(_attempt: object, error: BaseException) -> None:
        events.append(f"failed:{type(error).__name__}")

    hooks = _hooks(
        authenticate=lambda: events.append("authenticate") or "auth",
        load_config=lambda _auth: events.append("config") or "config",
        reserve_attempt=lambda _auth, _config: events.append("reserve") or "attempt",
        load_exact_task=lambda _auth: events.append("data") or "row",
        tokenize_task=reject_tokens,
        record_failure=record_failure,
    )
    with pytest.raises(stage_a.StageAAuthenticationError, match="token manifest drift"):
        stage_a.run_ordered_access(hooks)
    assert events == [
        "authenticate",
        "config",
        "reserve",
        "data",
        "tokenize",
        "failed:StageAAuthenticationError",
    ]


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
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (repo / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")
    h0 = git("rev-parse", "HEAD")
    attempt_path = tmp_path / "attempt.json"
    preflight = SimpleNamespace(
        repo_root=repo,
        repository_start={"commit": h0},
        source_hashes_start={"tracked.txt": hashlib.sha256(b"frozen\n").hexdigest()},
        stage0={
            "artifact_file_sha256": "4" * 64,
            "sidecar_file_sha256": "5" * 64,
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
    reservation = stage_a.reserve_one_run(preflight, configuration)
    assert git("rev-parse", "HEAD") == reservation.seal_commit
    assert git("rev-parse", f"{reservation.seal_commit}^") == h0
    assert git("rev-parse", f"{reservation.seal_commit}^{{tree}}") == git(
        "rev-parse", f"{h0}^{{tree}}"
    )
    assert git("show", "-s", "--format=%an <%ae>", reservation.seal_commit) == (
        "Synthetic Test <synthetic@example.invalid>"
    )
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


def test_completed_artifact_publication_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    stage_a._atomic_publish_new(path, b'{"result":1}')
    with pytest.raises(stage_a.StageAAuthenticationError, match="refusing to overwrite"):
        stage_a._atomic_publish_new(path, b'{"result":2}')
    assert path.read_bytes() == b'{"result":1}'
    assert not list(tmp_path.glob("*.tmp"))


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
        "_atomic_replace_owned",
        lambda *_args: events.append("receipt"),
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
        "publish",
        "receipt",
    ]


def test_failure_receipt_withholds_aggregate_and_forbids_automatic_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt = {"schema": "test", "status": "reserved"}
    payload = stage_a.canonical_json_bytes(attempt)
    stage_a._exclusive_write(attempt_path, payload)
    preflight = SimpleNamespace(attempt_path=attempt_path)
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
    stage_a.record_attempt_failure(
        preflight,
        reservation,
        RuntimeError("synthetic failure"),
    )
    receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_without_authenticated_stage_a_result"
    assert receipt["quality_aggregate_exposed"] is False
    assert receipt["rerun_automatically_authorized"] is False


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
    trajectory = {method: {"trajectory_nmse_auc": 0.2} for method in methods}
    trajectory[STATELEASE_METHOD] = {"trajectory_nmse_auc": 0.05}
    trajectory["fixed_cc1"] = {"trajectory_nmse_auc": 0.10}
    storage = {
        "resident_bytes_including_statelease": FROZEN_STATELEASE_RESIDENT_BYTES,
        "persistent_fp32_state_mirror": False,
    }
    diagnostics = [
        {
            "boundary4_count": 1 if layer == 0 else 0,
            "boundary5_count": 1 if layer == 0 else 0,
            "tie_count": 1 if layer == 0 else 0,
            "invalid_boundary_count": 0,
        }
        for layer in range(18)
    ]
    evidence = [
        {"boundary": 4, "tie": False},
        {"boundary": 5, "tie": True},
    ]
    return metrics, trajectory, storage, diagnostics, evidence


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


def test_cli_has_no_output_override_and_requires_explicit_mode() -> None:
    artifact = Path("stage0.pt")
    args = stage_a.parse_args(["--stage0-artifact", str(artifact), "--preflight-only"])
    assert args.preflight_only is True
    assert not hasattr(args, "output")
    with pytest.raises(SystemExit):
        stage_a.parse_args(["--stage0-artifact", str(artifact)])


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
