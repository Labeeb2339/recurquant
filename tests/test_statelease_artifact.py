from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from recurquant.cli import main
from recurquant.evidence import canonical_json_bytes
from recurquant.statelease_artifact import (
    EXPERIMENT012_CANONICAL_EVIDENCE_SHA256,
    EXPERIMENT012_FILE_SHA256,
    verify_experiment012_statelease_stage_a,
)
from recurquant.statelease_evaluation import FROZEN_STAGE_A_RECURRENT_LAYER_INDICES

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "evidence" / "experiment012-statelease-stage-a-666.json"


@pytest.fixture(scope="module")
def artifact_document() -> dict[str, Any]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _copy_document(value: dict[str, Any]) -> dict[str, Any]:
    # JSON round-tripping avoids shared nested references between tamper tests.
    result = json.loads(json.dumps(value, allow_nan=False))
    assert isinstance(result, dict)
    return result


def _write_rehashed(
    tmp_path: Path,
    document: dict[str, Any],
) -> tuple[Path, str, str]:
    canonical = hashlib.sha256(canonical_json_bytes(document["evidence"])).hexdigest()
    document["canonical_evidence_sha256"] = canonical
    payload = canonical_json_bytes(document)
    path = tmp_path / "experiment012.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), canonical


def _verify_rehashed(tmp_path: Path, document: dict[str, Any]) -> dict[str, Any]:
    path, file_sha256, canonical_sha256 = _write_rehashed(tmp_path, document)
    return verify_experiment012_statelease_stage_a(
        path,
        expected_file_sha256=file_sha256,
        expected_canonical_evidence_sha256=canonical_sha256,
    )


def test_committed_experiment012_artifact_recomputes_every_layer() -> None:
    report = verify_experiment012_statelease_stage_a(ARTIFACT)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["file_sha256"] == EXPERIMENT012_FILE_SHA256
    assert (
        report["computed_canonical_evidence_sha256"]
        == EXPERIMENT012_CANONICAL_EVIDENCE_SHA256
    )
    assert all(report["checks"].values())
    recomputed = report["recomputed"]
    assert recomputed["aligned_metrics"]["statelease_h5"]["delta_nll"] == pytest.approx(
        0.023349463939666748
    )
    assert recomputed["trajectory_nmse"]["statelease_h5"][
        "trajectory_nmse_auc"
    ] == pytest.approx(0.019805083952766495)
    assert recomputed["stage_a_gate"]["passed"] is True
    assert len(recomputed["stage_a_gate"]["checks"]) == 8


def test_cli_verifies_the_pinned_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify-statelease-stage-a", str(ARTIFACT)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["file_sha256"] == EXPERIMENT012_FILE_SHA256
    assert report["checks"]["recorded_gate_matches"] is True


def test_frozen_file_hash_rejects_otherwise_rehashed_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    document["evidence"]["claim_boundary"] += " altered"
    path, _file_sha256, _canonical = _write_rehashed(tmp_path, document)

    report = verify_experiment012_statelease_stage_a(path)

    assert report["valid"] is False
    assert report["checks"]["file_hash"] is False
    assert any("file_hash: expected" in error for error in report["errors"])


def test_recomputed_per_token_metrics_reject_stored_aggregate_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    document["evidence"]["metrics_aligned"]["statelease_h5"]["delta_nll"] += 0.01

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["aligned_metrics"] is False
    assert any("delta_nll differs" in error for error in report["errors"])


def test_recomputed_per_token_metrics_reject_raw_row_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    row = document["evidence"]["per_token_aligned"]["statelease_h5"][0]
    row["candidate_nll"] += 0.01
    row["delta_nll"] = row["candidate_nll"] - row["reference_nll"]

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["aligned_metrics"] is False
    assert any("candidate_nll differs" in error for error in report["errors"])


def test_recomputed_trajectory_rejects_stored_summary_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    document["evidence"]["trajectory_nmse"]["statelease_h5"][
        "trajectory_nmse_auc"
    ] += 0.01

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["trajectory"] is False
    assert any("trajectory_nmse_auc differs" in error for error in report["errors"])


def test_recomputed_trajectory_rejects_raw_layer_write_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    row = document["evidence"]["trajectory_nmse_per_layer_write"]["statelease_h5"][0]
    row["per_layer_nmse"]["0"] += 0.01
    row["layer_macro_nmse"] = sum(
        row["per_layer_nmse"][str(layer)]
        for layer in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
    ) / len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["trajectory"] is False
    assert any("trajectory_nmse_auc differs" in error for error in report["errors"])


def test_storage_receipt_is_rebuilt_from_serialized_tensor_schemas(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    document["evidence"]["storage_contracts"]["statelease_h5"]["resident_bytes"] -= 1

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["storage_contracts"] is False
    assert any(
        "storage_contracts.statelease_h5.resident_bytes" in error
        for error in report["errors"]
    )


def test_frozen_tensor_schema_rejects_raw_dtype_tampering(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    schema = document["evidence"]["storage"]["expanded_rht_q4_q8"][
        "candidate_tensor_schema"
    ]
    schema["checkpoint.layer_0.q4_payload"]["dtype"] = "torch.int8"

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["storage_contracts"] is False
    assert any("dtype/shape schema differs" in error for error in report["errors"])


def test_every_recorded_gate_check_is_compared_with_recomputed_evidence(
    tmp_path: Path,
    artifact_document: dict[str, Any],
) -> None:
    document = _copy_document(artifact_document)
    check = document["evidence"]["stage_a_gate"]["checks"][
        "cc1_excess_nll_reduction_at_least_10_percent"
    ]
    check["passed"] = False
    check["evidence"]["relative_reduction"] = 0.0

    report = _verify_rehashed(tmp_path, document)

    assert report["valid"] is False
    assert report["checks"]["gate_recomputed"] is True
    assert report["checks"]["recorded_gate_matches"] is False
    assert report["recomputed"]["stage_a_gate"]["checks"][
        "cc1_excess_nll_reduction_at_least_10_percent"
    ]["passed"] is True
    assert any(
        "cc1_excess_nll_reduction_at_least_10_percent.evidence.relative_reduction"
        in error
        for error in report["errors"]
    )
    assert any(
        "cc1_excess_nll_reduction_at_least_10_percent.passed" in error
        for error in report["errors"]
    )


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    raw = ARTIFACT.read_bytes()
    marker = b'{\n  "artifact_kind"'
    assert marker in raw
    tampered = raw.replace(marker, b'{\n  "schema_version": 1,\n  "artifact_kind"', 1)
    path = tmp_path / "duplicate-key.json"
    path.write_bytes(tampered)

    report = verify_experiment012_statelease_stage_a(
        path,
        expected_file_sha256=hashlib.sha256(tampered).hexdigest(),
    )

    assert report["valid"] is False
    assert report["checks"]["strict_json"] is False
    assert any("duplicate JSON object key" in error for error in report["errors"])


def test_malformed_gate_inputs_return_invalid_report_without_cli_traceback(
    tmp_path: Path,
    artifact_document: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = _copy_document(artifact_document)
    document["evidence"]["diagnostics"] = "malformed"
    path, file_sha256, canonical_sha256 = _write_rehashed(tmp_path, document)

    report = verify_experiment012_statelease_stage_a(
        path,
        expected_file_sha256=file_sha256,
        expected_canonical_evidence_sha256=canonical_sha256,
    )

    assert report["valid"] is False
    assert report["checks"]["task_and_token_contract"] is False
    assert report["checks"]["gate_recomputed"] is False
    assert main(["verify-statelease-stage-a", str(path)]) == 1
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["valid"] is False
