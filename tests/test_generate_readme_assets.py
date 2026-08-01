from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts import generate_readme_assets as assets


def _synthetic_stage_b() -> tuple[dict[str, object], dict[str, object]]:
    task_ids = list(range(700, 732))
    improvements = [0.1] * 27 + [-0.05] * 5
    per_task: dict[str, list[dict[str, object]]] = {}
    macro = {
        assets.STAGE_B_METHODS[0]: 0.42,
        assets.STAGE_B_METHODS[1]: 0.45,
        assets.STAGE_B_CQER_METHOD: 0.3,
        assets.STAGE_B_RHT_METHOD: 0.2234375,
    }
    for method in assets.STAGE_B_METHODS:
        if method == assets.STAGE_B_CQER_METHOD:
            values = [0.3] * 32
        elif method == assets.STAGE_B_RHT_METHOD:
            values = [0.3 - improvement for improvement in improvements]
        else:
            values = [macro[method]] * 32
        per_task[method] = [
            {"task_id": task_id, "delta_nll": value}
            for task_id, value in zip(task_ids, values, strict=True)
        ]

    mean = math.fsum(improvements) / len(improvements)
    evidence: dict[str, object] = {
        "artifact_kind": assets.STAGE_B_ARTIFACT_KIND,
        "methods": list(assets.STAGE_B_METHODS),
        "repository": {
            "commit": assets.STAGE_B_RESULT_COMMIT,
            "stable_commit": True,
        },
        "stage_b_integrity": {"passed": True},
        "stage_b_gate": {
            "passed": True,
            "advancement_checks": {
                "at_least_20_task_level_excess_nll_wins": {
                    "rht_wins": 27,
                    "ties": 0,
                }
            },
        },
        "dataset": {"identity": {"ordered_task_ids": task_ids}},
        "aggregates": {method: {"macro_delta_nll": value} for method, value in macro.items()},
        "per_task": per_task,
        "state_error": {
            "aggregates": {
                assets.STAGE_B_CQER_METHOD: {
                    "aggregate_state_sse": 36_000.0,
                },
                assets.STAGE_B_RHT_METHOD: {
                    "aggregate_state_sse": 15_000.0,
                },
            }
        },
        "paired_bootstrap_cqer_minus_rht_aligned_delta_nll": {
            "bootstrap_samples": 10_000,
            "confidence": 0.95,
            "confidence_interval": [0.04, 0.11],
            "mean_improvement": mean,
            "paired_examples": 32,
            "seed": 2339,
        },
        "claim_boundary": "Frozen development evidence only.",
    }
    verification: dict[str, object] = {
        "passed": True,
        "integrity_passed": True,
        "advancement_passed": True,
        "task_count": 32,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2339,
        "advancement_check_count": 8,
        "artifact_sha256": assets.STAGE_B_ARTIFACT_SHA256,
        "canonical_evidence_sha256": assets.STAGE_B_CANONICAL_SHA256,
        "canonical_round_trip": True,
    }
    return evidence, verification


def _metadata(svg: str) -> dict[str, object]:
    root = ET.fromstring(svg)
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None
    assert metadata.text is not None
    return json.loads(metadata.text)


def test_confirmation_pareto_uses_authenticated_storage_and_fidelity() -> None:
    data = assets._confirmation_data()
    svg = assets._confirmation_pareto_svg(data)
    assets._validate_svg(
        svg,
        assets.ASSETS / "mbpp-confirmation-pareto.svg",
    )

    assert svg == assets._confirmation_pareto_svg(data)
    record = _metadata(svg)
    assert record["chart"] == "mbpp-confirmation-storage-fidelity-frontier"
    assert record["canonical_evidence_sha256"] == (
        "2a652df92f99fa81f785244d966829e909d31f200e5a1520b76e6b46fb45d3e0"
    )
    assert record["task_count"] == 500
    assert record["token_count"] == 30_244
    assert record["speed_claim_allowed"] is False
    assert record["whole_model_memory_claim_allowed"] is False
    assert record["evaluated_nearest_policy_count"] == 7
    assert record["unique_nearest_coordinate_count"] == 6
    assert record["frontier_coordinate_count"] == 3

    points = record["points"]
    assert isinstance(points, list)
    assert [
        (
            point["label"],
            point["resident_bytes"],
            point["task_macro_excess_nll"],
        )
        for point in points
    ] == [
        ("Uniform INT4", 2_433_024, 2.949742543697357),
        ("Frozen v0.2 mixed", 2_564_096, 0.8037128749489785),
        ("Uniform INT8", 4_792_320, 0.017209371507167816),
    ]
    assert all(
        point["pareto_nondominated_among_plotted_quantized_layouts"] is True for point in points
    )
    assert record["fp32_reference"] == {
        "resident_bytes": 18_874_368,
        "task_macro_excess_nll": 0.0,
        "excess_nll_zero_by_definition": True,
        "off_plot": True,
    }
    assert "breakthrough" not in svg.lower()
    assert "state-of-the-art" not in svg.lower()


def test_statelease_stage_a_chart_includes_the_stronger_no_replay_comparators() -> None:
    data = assets._statelease_stage_a_data()
    svg = assets._statelease_stage_a_svg(data)
    assets._validate_svg(
        svg,
        assets.ASSETS / "experiment012-stage-a-excess-nll.svg",
    )

    assert svg == assets._statelease_stage_a_svg(data)
    record = _metadata(svg)
    assert record["chart"] == "experiment012-statelease-stage-a-excess-nll"
    assert record["source_artifact_sha256"] == assets.STATELEASE_STAGE_A_ARTIFACT_SHA256
    assert (
        record["canonical_evidence_sha256"]
        == assets.STATELEASE_STAGE_A_CANONICAL_SHA256
    )
    assert record["task_id"] == 666
    assert record["token_count"] == 38
    assert record["gate_check_count"] == 8
    assert record["gate_passed"] is True

    values = record["values"]
    assert isinstance(values, list)
    by_method = {row["method"]: row for row in values}
    assert set(by_method) == set(assets.STATELEASE_STAGE_A_METHODS)
    assert by_method["rht_q4_q6_q8"]["delta_nll"] < by_method["statelease_h5"][
        "delta_nll"
    ]
    assert by_method["expanded_rht_q4_q8"]["delta_nll"] < by_method[
        "statelease_h5"
    ]["delta_nll"]
    assert "did not beat" in str(record["claim_boundary"])


def test_stage_b_assets_use_strict_loader_and_embed_authenticated_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, verification = _synthetic_stage_b()
    calls: list[Path] = []
    source = tmp_path / "stage-b-result.json"
    source.write_bytes(b"strict loader owns these bytes")

    def load(path: Path) -> tuple[dict[str, object], dict[str, object]]:
        calls.append(path)
        return evidence, verification

    monkeypatch.setattr(assets, "STAGE_B_SOURCE", source)
    monkeypatch.setattr(assets, "_load_validated_stage_b_result", load)
    data = assets._stage_b_data()

    assert calls == [source]
    assert data["wins"] == 27
    assert data["ties"] == 0
    assert data["losses"] == 5
    assert [
        point["task_id"]
        for point in data["paired"]  # type: ignore[index]
    ] == list(range(700, 732))

    overview = assets._stage_b_overview_svg(data)
    paired = assets._stage_b_paired_svg(data)
    assets._validate_svg(
        overview,
        assets.ASSETS / "experiment009-stage-b-overview.svg",
    )
    assets._validate_svg(
        paired,
        assets.ASSETS / "experiment009-stage-b-paired.svg",
    )
    assert overview == assets._stage_b_overview_svg(data)
    assert paired == assets._stage_b_paired_svg(data)

    for record in (_metadata(overview), _metadata(paired)):
        assert record["artifact_kind"] == assets.STAGE_B_ARTIFACT_KIND
        assert record["source_artifact_sha256"] == assets.STAGE_B_ARTIFACT_SHA256
        assert record["canonical_evidence_sha256"] == assets.STAGE_B_CANONICAL_SHA256
        assert record["original_result_commit"] == assets.STAGE_B_RESULT_COMMIT
        assert record["verifier_fix_commit"] == assets.STAGE_B_VERIFIER_FIX_COMMIT
        assert record["integrity_passed"] is True
        assert record["advancement_gate_passed"] is True


def test_stage_b_data_rejects_verification_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, verification = _synthetic_stage_b()
    verification["artifact_sha256"] = "0" * 64
    source = tmp_path / "stage-b-result.json"
    source.write_bytes(b"strict loader owns these bytes")
    monkeypatch.setattr(assets, "STAGE_B_SOURCE", source)
    monkeypatch.setattr(
        assets,
        "_load_validated_stage_b_result",
        lambda _path: (evidence, verification),
    )

    with pytest.raises(
        ValueError,
        match="verification record drifted",
    ):
        assets._stage_b_data()


def test_check_detects_tampered_stage_b_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "assets" / "experiment009-stage-b-overview.svg"
    output.parent.mkdir()
    expected = '<svg xmlns="http://www.w3.org/2000/svg"><title>authenticated</title></svg>\n'
    output.write_text(expected, encoding="utf-8", newline="\n")
    source = tmp_path / "stage-b-result.json"
    source.write_bytes(b"present")
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "STAGE_B_SOURCE", source)
    monkeypatch.setattr(assets, "_outputs", lambda: {output: expected})
    monkeypatch.setattr(sys, "argv", ["generate_readme_assets.py", "--check"])

    assert assets.main() == 0

    output.write_text(expected.replace("authenticated", "tampered"), encoding="utf-8")
    assert assets.main() == 1


def _write_clean_clone_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    evidence, verification = _synthetic_stage_b()
    generation_source = tmp_path / "generation-source.json"
    generation_source.write_bytes(b"strict loader owns these bytes")
    monkeypatch.setattr(assets, "STAGE_B_SOURCE", generation_source)
    monkeypatch.setattr(
        assets,
        "_load_validated_stage_b_result",
        lambda _path: (evidence, verification),
    )
    data = assets._stage_b_data()
    rendered = {
        "assets/experiment009-stage-b-overview.svg": (
            assets._stage_b_overview_svg(data),
            "experiment009-stage-b-overview",
        ),
        "assets/experiment009-stage-b-paired.svg": (
            assets._stage_b_paired_svg(data),
            "experiment009-stage-b-paired-cqer-minus-rht",
        ),
    }

    graph_receipts: dict[str, dict[str, object]] = {}
    for relative_path, (content, chart) in rendered.items():
        raw = content.encode("utf-8")
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        graph_receipts[relative_path] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "chart": chart,
        }

    manifest: dict[str, object] = {
        "artifact_kind": "recurquant_experiment009_stage_b_release_manifest",
        "schema_version": 1,
        "result": {
            "artifact_kind": assets.STAGE_B_ARTIFACT_KIND,
            "raw_filename": "experiment009-rht-cqer-stage-b-result-cdc603b.json",
            "raw_sha256": assets.STAGE_B_ARTIFACT_SHA256,
            "canonical_evidence_sha256": assets.STAGE_B_CANONICAL_SHA256,
            "evaluation_commit": assets.STAGE_B_RESULT_COMMIT,
            "verifier_fix_commit": assets.STAGE_B_VERIFIER_FIX_COMMIT,
        },
        "verification": {
            "strict_loader_passed": True,
            "canonical_round_trip": True,
            "integrity_passed": True,
            "advancement_passed": True,
            "advancement_check_count": 8,
            "task_count": 32,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 2339,
        },
        "graph_assets": {
            "source_artifact_sha256": assets.STAGE_B_ARTIFACT_SHA256,
            "canonical_evidence_sha256": assets.STAGE_B_CANONICAL_SHA256,
            "original_result_commit": assets.STAGE_B_RESULT_COMMIT,
            "verifier_fix_commit": assets.STAGE_B_VERIFIER_FIX_COMMIT,
            "artifact_kind": assets.STAGE_B_ARTIFACT_KIND,
            "integrity_passed": True,
            "advancement_gate_passed": True,
            "assets": graph_receipts,
        },
    }
    manifest_path = tmp_path / "evidence" / "experiment009-rht-cqer-stage-b-result-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    legacy = tmp_path / "assets" / "legacy.svg"
    legacy_content = '<svg xmlns="http://www.w3.org/2000/svg"><title>legacy</title></svg>\n'
    legacy.write_text(legacy_content, encoding="utf-8", newline="\n")
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "STAGE_B_MANIFEST", manifest_path)
    monkeypatch.setattr(
        assets,
        "STAGE_B_SOURCE",
        tmp_path / "artifacts" / "experiment009-rht-cqer-stage-b-result-cdc603b.json",
    )
    monkeypatch.setattr(assets, "_legacy_outputs", lambda: {legacy: legacy_content})
    monkeypatch.setattr(sys, "argv", ["generate_readme_assets.py", "--check"])
    return (
        tmp_path / "assets" / "experiment009-stage-b-overview.svg",
        manifest_path,
        manifest,
    )


def test_clean_clone_check_uses_manifest_and_rejects_tampered_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overview, _manifest_path, _manifest = _write_clean_clone_fixture(
        monkeypatch,
        tmp_path,
    )

    assert assets.main() == 0

    overview.write_text(
        overview.read_text(encoding="utf-8").replace(
            "Stage B development",
            "Stage B tampered",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    assert assets.main() == 1


def test_clean_clone_check_rejects_tampered_metadata_even_with_new_byte_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overview, manifest_path, manifest = _write_clean_clone_fixture(
        monkeypatch,
        tmp_path,
    )
    raw = overview.read_bytes().replace(
        assets.STAGE_B_RESULT_COMMIT.encode(),
        ("0" * 40).encode(),
        1,
    )
    overview.write_bytes(raw)
    graph_assets = manifest["graph_assets"]
    assert isinstance(graph_assets, dict)
    receipts = graph_assets["assets"]
    assert isinstance(receipts, dict)
    overview_receipt = receipts["assets/experiment009-stage-b-overview.svg"]
    assert isinstance(overview_receipt, dict)
    overview_receipt["bytes"] = len(raw)
    overview_receipt["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    assert assets.main() == 1


def test_clean_clone_check_rejects_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_clean_clone_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        assets,
        "STAGE_B_MANIFEST",
        tmp_path / "evidence" / "missing-manifest.json",
    )

    assert assets.main() == 1


def test_clean_clone_check_rejects_missing_graph_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overview, _manifest_path, _manifest = _write_clean_clone_fixture(
        monkeypatch,
        tmp_path,
    )
    overview.unlink()

    assert assets.main() == 1


def test_generation_without_raw_stage_b_source_fails_with_download_instruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(
        assets,
        "STAGE_B_SOURCE",
        tmp_path / "artifacts" / "experiment009-rht-cqer-stage-b-result-cdc603b.json",
    )
    monkeypatch.setattr(sys, "argv", ["generate_readme_assets.py"])

    assert assets.main() == 2
    error = capsys.readouterr().err
    assert assets.STAGE_B_RELEASE_DOWNLOAD in error
    assert assets.STAGE_B_ARTIFACT_SHA256 in error
    assert "extract the JSON to" in error
