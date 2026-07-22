from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from recurquant.cli import build_parser
from recurquant.confirmation import ConfirmationSpec, verify_mbpp_confirmation
from recurquant.evaluation import (
    TokenFidelity,
    fidelity_summary,
    paired_bootstrap_mean_improvement,
)
from recurquant.evidence import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class _Fixture:
    artifact: Path
    manifest: Path
    checkpoint: Path
    artifact_sha256: str
    artifact_evidence_sha256: str
    spec: ConfirmationSpec


def test_installed_cli_exposes_confirmation_verifier() -> None:
    args = build_parser().parse_args(
        [
            "verify-confirmation",
            "confirmation.json",
            "manifest.json",
            "--checkpoint",
            "checkpoint.json",
        ]
    )

    assert args.command == "verify-confirmation"
    assert args.artifact == Path("confirmation.json")
    assert args.prepared_manifest == Path("manifest.json")
    assert args.checkpoint == Path("checkpoint.json")


def _compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def _candidate_artifact(
    policy: dict[str, Any],
    *,
    task_ids: tuple[int, ...],
    code_counts: list[int],
    task_delta: list[float],
    mean_kl: float,
    top1_pair: tuple[bool, bool],
    resident_bytes: int,
    reference_bytes: int,
    transient_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    kl: list[float] = []
    reference_nll: list[float] = []
    candidate_nll: list[float] = []
    top1: list[bool] = []
    per_task: list[dict[str, Any]] = []
    for task_id, count, delta in zip(task_ids, code_counts, task_delta, strict=True):
        task_kl = [mean_kl] * count
        task_reference = [1.0] * count
        task_candidate = [1.0 + delta] * count
        task_top1 = list(top1_pair[:count])
        fidelity = TokenFidelity(
            kl=torch.tensor(task_kl, dtype=torch.float32),
            reference_nll=torch.tensor(task_reference, dtype=torch.float32),
            candidate_nll=torch.tensor(task_candidate, dtype=torch.float32),
            top1_agreement=torch.tensor(task_top1, dtype=torch.bool),
        )
        per_task.append(
            {
                "task_id": task_id,
                "code_tokens": count,
                **fidelity_summary(fidelity),
            }
        )
        kl.extend(task_kl)
        reference_nll.extend(task_reference)
        candidate_nll.extend(task_candidate)
        top1.extend(task_top1)

    global_values = {
        "kl": kl,
        "reference_nll": reference_nll,
        "candidate_nll": candidate_nll,
        "top1": top1,
    }
    token_fidelity = TokenFidelity(
        kl=torch.tensor(kl, dtype=torch.float32),
        reference_nll=torch.tensor(reference_nll, dtype=torch.float32),
        candidate_nll=torch.tensor(candidate_nll, dtype=torch.float32),
        top1_agreement=torch.tensor(top1, dtype=torch.bool),
    )
    task_macro = {
        "task_count": len(per_task),
        "reference_nll": fmean(float(row["reference_nll"]) for row in per_task),
        "candidate_nll": fmean(float(row["candidate_nll"]) for row in per_task),
        "delta_nll": fmean(float(row["delta_nll"]) for row in per_task),
        "mean_kl": fmean(float(row["mean_kl"]) for row in per_task),
        "top1_agreement": fmean(float(row["top1_agreement"]) for row in per_task),
    }
    quartiles = {
        f"Q{index + 1}": {
            "task_count": 1,
            "minimum_code_tokens": row["code_tokens"],
            "maximum_code_tokens": row["code_tokens"],
            "macro_delta_nll": row["delta_nll"],
            "macro_mean_kl": row["mean_kl"],
            "macro_top1_agreement": row["top1_agreement"],
        }
        for index, row in enumerate(per_task)
    }
    storage = {
        "resident_bytes": resident_bytes,
        "full_precision_equivalent_bytes": reference_bytes,
        "resident_compression_ratio": reference_bytes / resident_bytes,
        "largest_transient_state_bytes": transient_bytes,
        "physical_reduction_realized": True,
        "expected_resident_bytes": resident_bytes,
        "exact_byte_gate": True,
    }
    candidate = {
        "policy": policy,
        "storage": storage,
        "token_weighted": fidelity_summary(token_fidelity),
        "task_macro": task_macro,
        "task_delta_nll_distribution": _quantiles(
            [float(row["delta_nll"]) for row in per_task]
        ),
        "by_code_length_quartile": quartiles,
        "per_task": per_task,
    }
    checkpoint_storage = {
        key: storage[key]
        for key in (
            "resident_bytes",
            "full_precision_equivalent_bytes",
            "resident_compression_ratio",
            "largest_transient_state_bytes",
            "physical_reduction_realized",
        )
    }
    return candidate, global_values, checkpoint_storage


def _write_fixture(tmp_path: Path, *, weak_uniform_interval: bool = False) -> _Fixture:
    seed = 2339
    model_id = "example/model"
    model_revision = "1" * 40
    source_commit = "2" * 40
    calibration_sha256 = "3" * 64
    task_ids = (11, 12, 13, 14)
    code_counts = [2, 2, 2, 2]
    token_manifest = [
        {
            "task_id": task_id,
            "prompt_tokens": 5,
            "code_tokens": count,
            "total_tokens": 5 + count,
            "prompt_token_ids_sha256": f"{task_id:064x}",
            "code_token_ids_sha256": f"{task_id + 100:064x}",
        }
        for task_id, count in zip(task_ids, code_counts, strict=True)
    ]
    dataset_manifest = {
        "config": "full",
        "dataset_id": "example/mbpp",
        "formatter_version": "recurquant.mbpp-prompt-code.v1",
        "phase": "confirmation",
        "revision": "4" * 40,
        "row_count": len(task_ids),
        "rows": [
            {"sha256": f"{task_id + 200:064x}", "task_id": task_id}
            for task_id in task_ids
        ],
        "schema": "recurquant.mbpp-manifest.v1",
        "selection_namespace": None,
        "source_split": "test",
    }
    dataset_sha256 = _compact_sha256(dataset_manifest)
    token_sha256 = hashlib.sha256(canonical_json_bytes(token_manifest)).hexdigest()
    candidate_plan = [
        {
            "name": "uniform_int4_nearest",
            "default_bits": 4,
            "upgrade_layer": None,
            "rounding": "nearest",
            "seed": seed,
        },
        {
            "name": "uniform_int8_nearest",
            "default_bits": 8,
            "upgrade_layer": None,
            "rounding": "nearest",
            "seed": seed,
        },
        {
            "name": "read_risk_l0_nearest",
            "default_bits": 4,
            "upgrade_layer": 0,
            "rounding": "nearest",
            "seed": seed,
        },
        {
            "name": "random_l18_nearest",
            "default_bits": 4,
            "upgrade_layer": 18,
            "rounding": "nearest",
            "seed": seed,
        },
        {
            "name": "random_l4_nearest",
            "default_bits": 4,
            "upgrade_layer": 4,
            "rounding": "nearest",
            "seed": seed,
        },
        {
            "name": "random_l13_nearest",
            "default_bits": 4,
            "upgrade_layer": 13,
            "rounding": "nearest",
            "seed": seed,
        },
    ]
    manifest_evidence = {
        "claim_scope": {
            "phase": "confirmation",
            "protocol_eligible": True,
            "outcomes_computed": False,
            "confirmation_touched": True,
        },
        "source": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "dataset_manifest": dataset_manifest,
            "dataset_manifest_sha256": dataset_sha256,
            "token_manifest": token_manifest,
            "token_manifest_sha256": token_sha256,
            "calibration_evidence_sha256": calibration_sha256,
            "repository_commit": "5" * 40,
        },
        "environment": {"tracked_worktree_clean": True},
        "schedule": {"row_count": len(task_ids), "candidate_plan": candidate_plan},
    }
    manifest_evidence_sha256 = hashlib.sha256(
        canonical_json_bytes(manifest_evidence)
    ).hexdigest()
    manifest_document = {
        "schema_version": 1,
        "artifact_kind": "recurquant_mbpp_prepared_manifest",
        "created_at_utc": "2026-07-22T00:00:00+00:00",
        "canonical_evidence_sha256": manifest_evidence_sha256,
        "evidence": manifest_evidence,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_raw = canonical_json_bytes(manifest_document)
    manifest_path.write_bytes(manifest_raw)
    manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()

    reference_bytes = 1_000
    uniform4_bytes = 100
    mixed_bytes = 110
    uniform8_bytes = 200
    transient_bytes = 50
    primary_delta = [0.0, 0.0, 0.0, 2.8] if weak_uniform_interval else [0.5] * 4
    candidate_inputs = {
        "uniform_int4_nearest": ([1.0] * 4, 1.0, (True, False), uniform4_bytes),
        "uniform_int8_nearest": ([0.1] * 4, 0.1, (True, True), uniform8_bytes),
        "read_risk_l0_nearest": (primary_delta, 0.25, (True, True), mixed_bytes),
        "random_l18_nearest": (
            [value + 1.0 for value in primary_delta],
            0.8,
            (True, False),
            mixed_bytes,
        ),
        "random_l4_nearest": (
            [value + 1.0 for value in primary_delta],
            0.8,
            (True, False),
            mixed_bytes,
        ),
        "random_l13_nearest": (
            [value + 1.0 for value in primary_delta],
            0.8,
            (True, False),
            mixed_bytes,
        ),
    }
    candidates: dict[str, Any] = {}
    global_values: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    storage_by_candidate: dict[str, Any] = {}
    for policy in candidate_plan:
        name = policy["name"]
        delta, kl, top1_pair, resident_bytes = candidate_inputs[name]
        candidate, values, storage = _candidate_artifact(
            policy,
            task_ids=task_ids,
            code_counts=code_counts,
            task_delta=delta,
            mean_kl=kl,
            top1_pair=top1_pair,
            resident_bytes=resident_bytes,
            reference_bytes=reference_bytes,
            transient_bytes=transient_bytes,
        )
        candidates[name] = candidate
        global_values[name] = values
        per_task[name] = candidate["per_task"]
        storage_by_candidate[name] = storage

    uniform_delta = [
        float(row["delta_nll"])
        for row in candidates["uniform_int4_nearest"]["per_task"]
    ]
    primary_delta_recorded = [
        float(row["delta_nll"])
        for row in candidates["read_risk_l0_nearest"]["per_task"]
    ]
    random_mean = [
        fmean(
            float(candidates[name]["per_task"][index]["delta_nll"])
            for name in ("random_l18_nearest", "random_l4_nearest", "random_l13_nearest")
        )
        for index in range(len(task_ids))
    ]
    uniform_bootstrap = paired_bootstrap_mean_improvement(
        uniform_delta, primary_delta_recorded, samples=10_000, seed=seed
    )
    random_bootstrap = paired_bootstrap_mean_improvement(
        random_mean, primary_delta_recorded, samples=10_000, seed=seed
    )
    macro_uniform = fmean(uniform_delta)
    macro_primary = fmean(primary_delta_recorded)
    relative_reduction = (macro_uniform - macro_primary) / macro_uniform
    contrasts = {
        "primary_vs_uniform_int4": {
            "uniform_macro_delta_nll": macro_uniform,
            "primary_macro_delta_nll": macro_primary,
            "relative_reduction": relative_reduction,
            "paired_bootstrap": uniform_bootstrap,
        },
        "primary_vs_mean_random_equal_byte": {"paired_bootstrap": random_bootstrap},
    }
    legacy_gates = {
        "all_values_finite": True,
        "exact_resident_bytes": True,
        "primary_macro_delta_nll_reduction_at_least_15_percent": (
            relative_reduction >= 0.15
        ),
        "equal_byte_bootstrap_interval_above_zero": (
            random_bootstrap["confidence_interval"][0] > 0
        ),
        "primary_mean_token_kl_lower_than_uniform_int4": True,
        "primary_cvar95_token_kl_lower_than_uniform_int4": True,
        "primary_top1_not_lower_than_uniform_int4": True,
    }

    signature_evidence = {
        "phase": "confirmation",
        "model_id": model_id,
        "model_revision": model_revision,
        "repository_commit": source_commit,
        "calibration_evidence_sha256": calibration_sha256,
        "prepared_manifest_evidence_sha256": manifest_evidence_sha256,
        "token_manifest_sha256": token_sha256,
        "group_size": 128,
        "candidate_plan": candidate_plan,
    }
    run_signature = hashlib.sha256(canonical_json_bytes(signature_evidence)).hexdigest()
    checkpoint_state = {
        "completed_task_ids": list(task_ids),
        "global_values": global_values,
        "per_task": per_task,
        "storage_by_candidate": storage_by_candidate,
        "reference_state_bytes": reference_bytes,
        "elapsed_wall_seconds": 12.5,
    }
    checkpoint_document = {
        "schema_version": 1,
        "run_signature_sha256": run_signature,
        "state_sha256": hashlib.sha256(canonical_json_bytes(checkpoint_state)).hexdigest(),
        "state": checkpoint_state,
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_raw = canonical_json_bytes(checkpoint_document)
    checkpoint_path.write_bytes(checkpoint_raw)
    checkpoint_file_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()

    artifact_evidence = {
        "claim_scope": {
            "phase": "confirmation",
            "protocol_eligible": True,
            "teacher_forced_fidelity_only": True,
            "generated_code_executed": False,
            "speed_claim_allowed": False,
            "whole_model_memory_claim_allowed": False,
            "confirmation_touched": True,
        },
        "source": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "dataset_manifest": dataset_manifest,
            "dataset_manifest_sha256": dataset_sha256,
            "token_manifest": token_manifest,
            "token_manifest_sha256": token_sha256,
            "calibration_artifact_path": "calibration.json",
            "calibration_evidence_sha256": calibration_sha256,
            "repository_commit": source_commit,
            "prepared_manifest_evidence_sha256": manifest_evidence_sha256,
        },
        "environment": {"tracked_worktree_clean": True},
        "schedule": {
            "seed": seed,
            "phase": "confirmation",
            "row_count": len(task_ids),
            "group_size": 128,
            "candidate_order": [policy["name"] for policy in candidate_plan],
            "scored_first_code_token_from_prefill": True,
            "candidate_generated_tokens_fed_back": False,
            "elapsed_wall_seconds_not_a_latency_benchmark": 12.6,
            "run_signature_sha256": run_signature,
            "checkpoint_path": "checkpoint.json",
            "resumed_task_count": 0,
            "final_checkpoint_sha256": checkpoint_file_sha256,
        },
        "validity": {
            "configured_gdn_layer_indices": [0],
            "reference_recurrent_state_bytes": reference_bytes,
            "packed_qdq_preflight": {
                "absolute_tolerance": 1e-6,
                "maximum_absolute_difference_by_candidate": {
                    policy["name"]: 0.0 for policy in candidate_plan
                },
                "passed": True,
                "task_id": 945,
            },
        },
        "candidates": candidates,
        "contrasts": contrasts,
        "continuation_decision": {
            "gates": legacy_gates,
            "all_gates_pass": all(legacy_gates.values()),
            "confirmation_permitted": False,
        },
    }
    artifact_evidence_sha256 = hashlib.sha256(
        canonical_json_bytes(artifact_evidence)
    ).hexdigest()
    artifact_document = {
        "schema_version": 1,
        "artifact_kind": "recurquant_mbpp_teacher_forced_evaluation",
        "created_at_utc": "2026-07-22T00:00:01+00:00",
        "canonical_evidence_sha256": artifact_evidence_sha256,
        "evidence": artifact_evidence,
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_raw = canonical_json_bytes(artifact_document)
    artifact_path.write_bytes(artifact_raw)
    artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()

    spec = ConfirmationSpec(
        source_commit=source_commit,
        prepared_manifest_file_sha256=manifest_file_sha256,
        prepared_manifest_evidence_sha256=manifest_evidence_sha256,
        dataset_manifest_sha256=dataset_sha256,
        token_manifest_sha256=token_sha256,
        calibration_evidence_sha256=calibration_sha256,
        model_id=model_id,
        model_revision=model_revision,
        task_ids=task_ids,
        prompt_token_count=20,
        code_token_count=8,
        combined_token_count=28,
        gdn_layer_indices=(0,),
        reference_state_bytes=reference_bytes,
        uniform_int4_bytes=uniform4_bytes,
        mixed_int4_int8_bytes=mixed_bytes,
        uniform_int8_bytes=uniform8_bytes,
        largest_transient_state_bytes=transient_bytes,
        preflight_task_id=945,
    )
    return _Fixture(
        artifact=artifact_path,
        manifest=manifest_path,
        checkpoint=checkpoint_path,
        artifact_sha256=artifact_sha256,
        artifact_evidence_sha256=artifact_evidence_sha256,
        spec=spec,
    )


def test_verifies_artifact_and_manifest_without_large_checkpoint(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        expected_artifact_sha256=fixture.artifact_sha256.upper(),
        expected_artifact_evidence_sha256=fixture.artifact_evidence_sha256.upper(),
        spec=fixture.spec,
    )

    assert report["valid"] is True, report["errors"]
    assert report["artifact_manifest_verified"] is True
    assert report["checkpoint_verified"] is False
    assert report["outcome_verified"] is True
    assert report["outcome"]["quality_hypothesis_pass"] is True
    assert report["outcome"]["verification_basis"] == "externally_anchored_artifact"
    assert report["result"] == "pass"
    assert report["warnings"]


def test_unanchored_and_one_hash_artifact_modes_do_not_claim_an_outcome(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    unsigned = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        spec=fixture.spec,
    )
    one_hash = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        expected_artifact_sha256=fixture.artifact_sha256,
        spec=fixture.spec,
    )

    for report in (unsigned, one_hash):
        assert report["valid"] is True, report["errors"]
        assert report["artifact_manifest_verified"] is True
        assert report["outcome_verified"] is False
        assert report["outcome"]["quality_hypothesis_pass"] is None
        assert report["outcome"]["quality_gates"] is None
        assert report["outcome"]["verification_basis"] == "unanchored_artifact"
        assert report["result"] == "unverified"


def test_verifies_checkpoint_hash_state_signature_and_raw_arrays(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        checkpoint_path=fixture.checkpoint,
        spec=fixture.spec,
    )

    assert report["valid"] is True, report["errors"]
    assert report["checkpoint_verified"] is True
    assert report["outcome_verified"] is True
    assert report["outcome"]["verification_basis"] == "checkpoint_raw_arrays"
    assert report["outcome"]["quality_hypothesis_pass"] is True
    assert report["hashes"]["checkpoint_file_sha256"]
    assert report["hashes"]["checkpoint_state_sha256"]
    assert report["warnings"] == []


def test_uniform_interval_is_part_of_quality_decision_but_not_legacy_record(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, weak_uniform_interval=True)

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        expected_artifact_sha256=fixture.artifact_sha256,
        expected_artifact_evidence_sha256=fixture.artifact_evidence_sha256,
        spec=fixture.spec,
    )

    assert report["valid"] is True, report["errors"]
    assert report["outcome"]["quality_gates"][
        "primary_macro_delta_nll_reduction_at_least_15_percent"
    ] is True
    assert report["outcome"]["quality_gates"][
        "primary_vs_uniform_bootstrap_interval_above_zero"
    ] is False
    assert report["outcome"]["quality_hypothesis_pass"] is False
    assert report["outcome_verified"] is True
    assert report["result"] == "fail"


def test_checkpoint_raw_values_control_near_boundary_gate(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    artifact = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    checkpoint = json.loads(fixture.checkpoint.read_text(encoding="utf-8"))
    primary_name = "read_risk_l0_nearest"
    uniform_name = "uniform_int4_nearest"
    uniform_boundary = 0.5
    recorded_below_boundary = 0.49999999
    checkpoint_above_boundary = 0.50000003
    checkpoint_float32_summary = float(
        torch.tensor([checkpoint_above_boundary], dtype=torch.float32).mean().item()
    )
    assert checkpoint_float32_summary > uniform_boundary > recorded_below_boundary
    assert checkpoint_float32_summary - recorded_below_boundary < 1e-7

    for name, recorded_value in (
        (uniform_name, uniform_boundary),
        (primary_name, recorded_below_boundary),
    ):
        candidate = artifact["evidence"]["candidates"][name]
        for key in ("mean_kl", "cvar95_kl", "max_kl"):
            candidate["token_weighted"][key] = recorded_value
        candidate["task_macro"]["mean_kl"] = recorded_value
        for row in candidate["per_task"]:
            for key in ("mean_kl", "cvar95_kl", "max_kl"):
                row[key] = recorded_value
        for quartile in candidate["by_code_length_quartile"].values():
            quartile["macro_mean_kl"] = recorded_value
        for row in checkpoint["state"]["per_task"][name]:
            for key in ("mean_kl", "cvar95_kl", "max_kl"):
                row[key] = recorded_value
    checkpoint["state"]["global_values"][uniform_name]["kl"] = [
        uniform_boundary
    ] * fixture.spec.code_token_count
    checkpoint["state"]["global_values"][primary_name]["kl"] = [
        checkpoint_above_boundary
    ] * fixture.spec.code_token_count
    checkpoint["state_sha256"] = hashlib.sha256(
        canonical_json_bytes(checkpoint["state"])
    ).hexdigest()
    checkpoint_raw = canonical_json_bytes(checkpoint)
    fixture.checkpoint.write_bytes(checkpoint_raw)
    artifact["evidence"]["schedule"]["final_checkpoint_sha256"] = hashlib.sha256(
        checkpoint_raw
    ).hexdigest()
    artifact["canonical_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(artifact["evidence"])
    ).hexdigest()
    fixture.artifact.write_bytes(canonical_json_bytes(artifact))

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        checkpoint_path=fixture.checkpoint,
        spec=fixture.spec,
    )

    assert report["valid"] is False
    assert report["checkpoint_verified"] is False
    assert report["outcome_verified"] is False
    assert report["result"] == "invalid"
    assert report["outcome"]["verification_basis"] == "checkpoint_raw_arrays"
    assert report["outcome"]["quality_gates"][
        "primary_mean_token_kl_lower_than_uniform_int4"
    ] is False
    assert report["outcome"]["quality_hypothesis_pass"] is False
    assert any("recorded continuation gates" in error for error in report["errors"])


def test_huge_integer_metric_is_rejected_without_overflowing(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    artifact = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    artifact["evidence"]["candidates"]["read_risk_l0_nearest"]["token_weighted"][
        "mean_kl"
    ] = 10**400
    artifact["canonical_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(artifact["evidence"])
    ).hexdigest()
    fixture.artifact.write_bytes(canonical_json_bytes(artifact))

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        spec=fixture.spec,
    )

    assert report["valid"] is False
    assert report["outcome_verified"] is False
    assert report["result"] == "invalid"
    assert any(
        "not finite numeric data" in error or "could not complete" in error
        for error in report["errors"]
    )


def test_rejects_non_finite_json_even_when_json_parser_accepts_overflow(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    raw = fixture.artifact.read_text(encoding="utf-8")
    fixture.artifact.write_text(
        raw.replace('"mean_kl": 0.25', '"mean_kl": 1e400', 1),
        encoding="utf-8",
    )

    report = verify_mbpp_confirmation(
        fixture.artifact,
        fixture.manifest,
        spec=fixture.spec,
    )

    assert report["valid"] is False
    assert any("non-finite" in error for error in report["errors"])


def test_script_returns_nonzero_json_report_for_tampered_checkpoint(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    checkpoint = json.loads(fixture.checkpoint.read_text(encoding="utf-8"))
    checkpoint["state"]["completed_task_ids"][-1] = 999
    fixture.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "verify_mbpp_confirmation.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(fixture.artifact),
            str(fixture.manifest),
            "--checkpoint",
            str(fixture.checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["valid"] is False
    assert report["checkpoint_verified"] is False
    assert any("checkpoint" in error for error in report["errors"])
