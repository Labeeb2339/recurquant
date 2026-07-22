from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recurquant.cli import main
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact


def _write_artifact(path: Path, evidence: dict[str, object]) -> tuple[str, str]:
    canonical_sha256 = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    payload = {
        "artifact_kind": "test",
        "canonical_evidence_sha256": canonical_sha256,
        "evidence": evidence,
        "schema_version": 1,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), canonical_sha256


def test_verifier_accepts_exact_file_and_canonical_hashes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    file_sha256, canonical_sha256 = _write_artifact(path, {"metric": 1.25})

    report = verify_evidence_artifact(
        path,
        expected_file_sha256=file_sha256.upper(),
        expected_canonical_evidence_sha256=canonical_sha256,
    )

    assert report == {
        "artifact_path": str(path),
        "computed_canonical_evidence_sha256": canonical_sha256,
        "errors": [],
        "file_sha256": file_sha256,
        "recorded_canonical_evidence_sha256": canonical_sha256,
        "valid": True,
    }


def test_verifier_rejects_tampered_evidence(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    _write_artifact(path, {"metric": 1.25})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["evidence"]["metric"] = 9.0
    path.write_text(json.dumps(document), encoding="utf-8")

    report = verify_evidence_artifact(path)

    assert report["valid"] is False
    assert any("canonical evidence SHA256 mismatch" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        (b"{\"evidence\": {\"metric\": NaN}}", "strict UTF-8 JSON"),
        (b"[]", "artifact root must be a JSON object"),
        (b"{}", "artifact must contain an evidence object"),
    ],
)
def test_verifier_rejects_invalid_artifacts(
    tmp_path: Path,
    contents: bytes,
    expected_error: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)

    report = verify_evidence_artifact(path)

    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


def test_verify_artifact_cli_returns_status_and_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "artifact.json"
    file_sha256, _ = _write_artifact(path, {"metric": 1.25})

    assert main(["verify-artifact", str(path), "--expect-file-sha256", file_sha256]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["valid"] is True

    assert main(["verify-artifact", str(path), "--expect-file-sha256", "0" * 64]) == 1
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["valid"] is False
    assert any("file SHA256 mismatch" in error for error in rejected["errors"])
