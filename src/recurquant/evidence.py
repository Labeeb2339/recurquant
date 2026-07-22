"""Verification helpers for RecurQuant JSON evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value with the canonical format used by research artifacts."""

    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in string.hexdigits for character in value)


def verify_evidence_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_canonical_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify an artifact's file bytes and canonical ``evidence`` payload.

    The function always returns a JSON-serializable report. ``valid`` is false
    when the document is malformed, its recorded canonical hash is wrong, or an
    optional expected hash does not match.
    """

    artifact_path = Path(path)
    errors: list[str] = []
    try:
        raw = artifact_path.read_bytes()
    except OSError as exc:
        return {
            "artifact_path": str(artifact_path),
            "computed_canonical_evidence_sha256": None,
            "errors": [f"could not read artifact: {exc}"],
            "file_sha256": None,
            "recorded_canonical_evidence_sha256": None,
            "valid": False,
        }

    file_sha256 = hashlib.sha256(raw).hexdigest()
    expected_file = expected_file_sha256.lower() if expected_file_sha256 else None
    expected_canonical = (
        expected_canonical_evidence_sha256.lower()
        if expected_canonical_evidence_sha256
        else None
    )
    if expected_file is not None and not _valid_sha256(expected_file):
        errors.append("expected file SHA256 must contain exactly 64 hexadecimal characters")
    elif expected_file is not None and file_sha256 != expected_file:
        errors.append(
            f"file SHA256 mismatch: expected {expected_file}, computed {file_sha256}"
        )
    if expected_canonical is not None and not _valid_sha256(expected_canonical):
        errors.append(
            "expected canonical evidence SHA256 must contain exactly 64 hexadecimal characters"
        )

    try:
        document = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_finite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"artifact is not strict UTF-8 JSON: {exc}")
        return {
            "artifact_path": str(artifact_path),
            "computed_canonical_evidence_sha256": None,
            "errors": errors,
            "file_sha256": file_sha256,
            "recorded_canonical_evidence_sha256": None,
            "valid": False,
        }

    if not isinstance(document, dict):
        errors.append("artifact root must be a JSON object")
        evidence = None
        recorded_canonical = None
    else:
        evidence = document.get("evidence")
        recorded_canonical = document.get("canonical_evidence_sha256")

    computed_canonical: str | None = None
    if not isinstance(evidence, dict):
        errors.append("artifact must contain an evidence object")
    else:
        computed_canonical = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()

    if not isinstance(recorded_canonical, str) or not _valid_sha256(recorded_canonical):
        errors.append(
            "artifact canonical_evidence_sha256 must contain exactly 64 hexadecimal characters"
        )
        recorded_canonical_value = (
            recorded_canonical if isinstance(recorded_canonical, str) else None
        )
    else:
        recorded_canonical_value = recorded_canonical.lower()
        if computed_canonical is not None and recorded_canonical_value != computed_canonical:
            errors.append(
                "canonical evidence SHA256 mismatch: "
                f"recorded {recorded_canonical_value}, computed {computed_canonical}"
            )

    if (
        expected_canonical is not None
        and _valid_sha256(expected_canonical)
        and computed_canonical is not None
        and computed_canonical != expected_canonical
    ):
        errors.append(
            "expected canonical evidence SHA256 mismatch: "
            f"expected {expected_canonical}, computed {computed_canonical}"
        )

    return {
        "artifact_path": str(artifact_path),
        "computed_canonical_evidence_sha256": computed_canonical,
        "errors": errors,
        "file_sha256": file_sha256,
        "recorded_canonical_evidence_sha256": recorded_canonical_value,
        "valid": not errors,
    }
