"""Strict offline verification for the frozen RecurQuant v0.2 confirmation run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from .evaluation import TokenFidelity, fidelity_summary, paired_bootstrap_mean_improvement
from .evidence import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class ConfirmationSpec:
    """Immutable inputs that identify one preregistered confirmation run."""

    source_commit: str
    prepared_manifest_file_sha256: str
    prepared_manifest_evidence_sha256: str
    dataset_manifest_sha256: str
    token_manifest_sha256: str
    calibration_evidence_sha256: str
    model_id: str
    model_revision: str
    task_ids: tuple[int, ...]
    prompt_token_count: int
    code_token_count: int
    combined_token_count: int
    gdn_layer_indices: tuple[int, ...]
    reference_state_bytes: int
    uniform_int4_bytes: int
    mixed_int4_int8_bytes: int
    uniform_int8_bytes: int
    largest_transient_state_bytes: int
    preflight_task_id: int
    seed: int = 2339
    group_size: int = 128
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    preflight_tolerance: float = 1e-6


FROZEN_V02_CONFIRMATION = ConfirmationSpec(
    source_commit="6bd5bed2b61e192526ba8fdbec8232801cbea843",
    prepared_manifest_file_sha256=(
        "c6a7d0db6ef7577a66ac19fbbc0be166279488f6a6be432b364bd9eb6833f7b0"
    ),
    prepared_manifest_evidence_sha256=(
        "21a6d18c6a0887b1499d156a3d610d4bfafdd59d3557713485b62038e263b96a"
    ),
    dataset_manifest_sha256=(
        "060aaff7117dc47af6c01253a912f34b6956241c336bbc7216e73bca8624d2d4"
    ),
    token_manifest_sha256=(
        "199a8836489af9bd0af3fec027e85d57df356bd9919492b24015de51d143f525"
    ),
    calibration_evidence_sha256=(
        "7aa8227dd0b19bb7494963c0b590c8ec53cee29d3b696ccd4087c71a5ac461ee"
    ),
    model_id="Qwen/Qwen3.5-0.8B-Base",
    model_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    task_ids=tuple(range(11, 511)),
    prompt_token_count=68_904,
    code_token_count=30_244,
    combined_token_count=99_148,
    gdn_layer_indices=(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22),
    reference_state_bytes=18_874_368,
    uniform_int4_bytes=2_433_024,
    mixed_int4_int8_bytes=2_564_096,
    uniform_int8_bytes=4_792_320,
    largest_transient_state_bytes=1_048_576,
    preflight_task_id=945,
)


class _DuplicateKeyError(ValueError):
    pass


@dataclass(slots=True)
class _LoadedJSON:
    document: Any
    file_sha256: str


@dataclass(slots=True)
class _Audit:
    errors: list[str]

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        if not _deep_equal(actual, expected):
            self.errors.append(f"{label} does not match the frozen value")


def _deep_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _deep_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _deep_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)


def _pairs_to_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _check_finite_tree(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_tree(child, path=f"{path}[{index}]")


def _load_strict_json(path: str | Path, *, label: str, audit: _Audit) -> _LoadedJSON | None:
    resolved = Path(path)
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        audit.errors.append(f"could not read {label}: {exc}")
        return None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_to_object,
            parse_constant=_reject_non_finite_constant,
        )
        _check_finite_tree(document)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        audit.errors.append(f"{label} is not strict finite UTF-8 JSON: {exc}")
        return None
    return _LoadedJSON(document=document, file_sha256=digest)


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_compact_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, *, label: str, audit: _Audit) -> dict[str, Any]:
    if not isinstance(value, dict):
        audit.errors.append(f"{label} must be an object")
        return {}
    return value


def _list(value: Any, *, label: str, audit: _Audit) -> list[Any]:
    if not isinstance(value, list):
        audit.errors.append(f"{label} must be an array")
        return []
    return value


def _verify_envelope(
    loaded: _LoadedJSON,
    *,
    label: str,
    artifact_kind: str,
    audit: _Audit,
) -> dict[str, Any]:
    document = _mapping(loaded.document, label=f"{label} root", audit=audit)
    audit.equal(document.get("schema_version"), 1, f"{label} schema_version")
    audit.equal(document.get("artifact_kind"), artifact_kind, f"{label} artifact_kind")
    evidence = _mapping(document.get("evidence"), label=f"{label} evidence", audit=audit)
    computed = _sha256_canonical(evidence)
    recorded = document.get("canonical_evidence_sha256")
    audit.check(_valid_sha256(recorded), f"{label} canonical evidence SHA256 is invalid")
    audit.equal(recorded, computed, f"{label} canonical evidence SHA256")
    return evidence


def _expected_candidate_bytes(policy: Mapping[str, Any], spec: ConfirmationSpec) -> int:
    default_bits = policy.get("default_bits")
    upgrade_layer = policy.get("upgrade_layer")
    if default_bits == 8 and upgrade_layer is None:
        return spec.uniform_int8_bytes
    if default_bits == 4 and upgrade_layer is None:
        return spec.uniform_int4_bytes
    if default_bits == 4 and isinstance(upgrade_layer, int) and not isinstance(
        upgrade_layer, bool
    ):
        return spec.mixed_int4_int8_bytes
    raise ValueError(f"unsupported frozen candidate layout: {dict(policy)!r}")


def _quartile_membership(token_manifest: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    ordered = sorted(
        ((int(row["code_tokens"]), int(row["task_id"])) for row in token_manifest),
        key=lambda item: (item[0], item[1]),
    )
    quotient, remainder = divmod(len(ordered), 4)
    membership: dict[int, str] = {}
    offset = 0
    for index in range(4):
        size = quotient + (1 if index < remainder else 0)
        for _, task_id in ordered[offset : offset + size]:
            membership[task_id] = f"Q{index + 1}"
        offset += size
    return membership


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def _candidate_task_aggregates(
    rows: Sequence[Mapping[str, Any]],
    membership: Mapping[int, str],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    task_macro = {
        "task_count": len(rows),
        "reference_nll": fmean(float(row["reference_nll"]) for row in rows),
        "candidate_nll": fmean(float(row["candidate_nll"]) for row in rows),
        "delta_nll": fmean(float(row["delta_nll"]) for row in rows),
        "mean_kl": fmean(float(row["mean_kl"]) for row in rows),
        "top1_agreement": fmean(float(row["top1_agreement"]) for row in rows),
    }
    distribution = _quantiles([float(row["delta_nll"]) for row in rows])
    by_quartile: dict[str, Any] = {}
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        selected = [row for row in rows if membership[int(row["task_id"])] == quartile]
        if not selected:
            by_quartile[quartile] = {
                "task_count": 0,
                "minimum_code_tokens": None,
                "maximum_code_tokens": None,
                "macro_delta_nll": None,
                "macro_mean_kl": None,
                "macro_top1_agreement": None,
            }
            continue
        by_quartile[quartile] = {
            "task_count": len(selected),
            "minimum_code_tokens": min(int(row["code_tokens"]) for row in selected),
            "maximum_code_tokens": max(int(row["code_tokens"]) for row in selected),
            "macro_delta_nll": fmean(float(row["delta_nll"]) for row in selected),
            "macro_mean_kl": fmean(float(row["mean_kl"]) for row in selected),
            "macro_top1_agreement": fmean(
                float(row["top1_agreement"]) for row in selected
            ),
        }
    return task_macro, distribution, by_quartile


def _approximately_equal(left: Any, right: Any, *, tolerance: float = 1e-7) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return math.isclose(
                float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
            )
        except (OverflowError, ValueError):
            return False
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _approximately_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _approximately_equal(a, b, tolerance=tolerance)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _check_approximately_equal(
    audit: _Audit,
    actual: Any,
    expected: Any,
    label: str,
    *,
    tolerance: float = 1e-7,
) -> None:
    if not _approximately_equal(actual, expected, tolerance=tolerance):
        audit.errors.append(f"{label} does not match values recomputed from task data")


def _verify_manifest_and_source(
    manifest_evidence: dict[str, Any],
    artifact_evidence: dict[str, Any],
    *,
    spec: ConfirmationSpec,
    audit: _Audit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_claim = _mapping(
        manifest_evidence.get("claim_scope"), label="prepared manifest claim_scope", audit=audit
    )
    expected_manifest_claim = {
        "phase": "confirmation",
        "protocol_eligible": True,
        "outcomes_computed": False,
        "confirmation_touched": True,
    }
    audit.equal(manifest_claim, expected_manifest_claim, "prepared manifest claim_scope")

    manifest_source = _mapping(
        manifest_evidence.get("source"), label="prepared manifest source", audit=audit
    )
    audit.equal(manifest_source.get("model_id"), spec.model_id, "prepared manifest model_id")
    audit.equal(
        manifest_source.get("model_revision"),
        spec.model_revision,
        "prepared manifest model_revision",
    )
    audit.equal(
        manifest_source.get("tokenizer_revision"),
        spec.model_revision,
        "prepared manifest tokenizer_revision",
    )
    audit.equal(
        manifest_source.get("dataset_manifest_sha256"),
        spec.dataset_manifest_sha256,
        "prepared manifest dataset manifest SHA256",
    )
    audit.equal(
        manifest_source.get("token_manifest_sha256"),
        spec.token_manifest_sha256,
        "prepared manifest token manifest SHA256",
    )
    audit.equal(
        manifest_source.get("calibration_evidence_sha256"),
        spec.calibration_evidence_sha256,
        "prepared manifest calibration evidence SHA256",
    )

    dataset_manifest = _mapping(
        manifest_source.get("dataset_manifest"), label="prepared dataset manifest", audit=audit
    )
    token_manifest = _list(
        manifest_source.get("token_manifest"), label="prepared token manifest", audit=audit
    )
    audit.equal(
        _sha256_compact_json(dataset_manifest),
        spec.dataset_manifest_sha256,
        "recomputed dataset manifest SHA256",
    )
    audit.equal(
        _sha256_canonical(token_manifest),
        spec.token_manifest_sha256,
        "recomputed token manifest SHA256",
    )

    dataset_rows = _list(
        dataset_manifest.get("rows"), label="prepared dataset manifest rows", audit=audit
    )
    task_ids = list(spec.task_ids)
    audit.equal(
        [row.get("task_id") for row in dataset_rows if isinstance(row, dict)],
        task_ids,
        "prepared dataset task IDs",
    )
    audit.equal(
        [row.get("task_id") for row in token_manifest if isinstance(row, dict)],
        task_ids,
        "prepared token task IDs",
    )
    audit.equal(dataset_manifest.get("phase"), "confirmation", "dataset manifest phase")
    audit.equal(dataset_manifest.get("source_split"), "test", "dataset manifest source split")
    audit.equal(dataset_manifest.get("row_count"), len(task_ids), "dataset manifest row_count")

    prompt_count = sum(
        int(row.get("prompt_tokens", -1)) for row in token_manifest if isinstance(row, dict)
    )
    code_count = sum(
        int(row.get("code_tokens", -1)) for row in token_manifest if isinstance(row, dict)
    )
    combined_count = sum(
        int(row.get("total_tokens", -1)) for row in token_manifest if isinstance(row, dict)
    )
    audit.equal(prompt_count, spec.prompt_token_count, "prepared prompt token count")
    audit.equal(code_count, spec.code_token_count, "prepared code token count")
    audit.equal(combined_count, spec.combined_token_count, "prepared combined token count")
    for index, row_value in enumerate(token_manifest):
        row = _mapping(row_value, label=f"prepared token row {index}", audit=audit)
        audit.check(
            row.get("total_tokens") == row.get("prompt_tokens", -1) + row.get("code_tokens", -1),
            f"prepared token row {index} total_tokens is inconsistent",
        )
        for digest_key in ("prompt_token_ids_sha256", "code_token_ids_sha256"):
            audit.check(
                _valid_sha256(row.get(digest_key)),
                f"prepared token row {index} {digest_key} is invalid",
            )

    artifact_claim = _mapping(
        artifact_evidence.get("claim_scope"), label="artifact claim_scope", audit=audit
    )
    expected_artifact_claim = {
        "phase": "confirmation",
        "protocol_eligible": True,
        "teacher_forced_fidelity_only": True,
        "generated_code_executed": False,
        "speed_claim_allowed": False,
        "whole_model_memory_claim_allowed": False,
        "confirmation_touched": True,
    }
    audit.equal(artifact_claim, expected_artifact_claim, "artifact claim_scope")
    artifact_source = _mapping(
        artifact_evidence.get("source"), label="artifact source", audit=audit
    )
    audit.equal(artifact_source.get("model_id"), spec.model_id, "artifact model_id")
    audit.equal(
        artifact_source.get("model_revision"), spec.model_revision, "artifact model_revision"
    )
    audit.equal(
        artifact_source.get("tokenizer_revision"),
        spec.model_revision,
        "artifact tokenizer_revision",
    )
    audit.equal(
        artifact_source.get("repository_commit"), spec.source_commit, "artifact source commit"
    )
    audit.equal(
        artifact_source.get("dataset_manifest_sha256"),
        spec.dataset_manifest_sha256,
        "artifact dataset manifest SHA256",
    )
    audit.equal(
        artifact_source.get("token_manifest_sha256"),
        spec.token_manifest_sha256,
        "artifact token manifest SHA256",
    )
    audit.equal(
        artifact_source.get("calibration_evidence_sha256"),
        spec.calibration_evidence_sha256,
        "artifact calibration evidence SHA256",
    )
    audit.equal(
        artifact_source.get("prepared_manifest_evidence_sha256"),
        spec.prepared_manifest_evidence_sha256,
        "artifact prepared manifest evidence SHA256",
    )
    audit.equal(
        artifact_source.get("dataset_manifest"),
        dataset_manifest,
        "artifact/prepared dataset manifest deep equality",
    )
    audit.equal(
        artifact_source.get("token_manifest"),
        token_manifest,
        "artifact/prepared token manifest deep equality",
    )
    return dataset_rows, [dict(row) for row in token_manifest if isinstance(row, dict)]


def _verify_schedule_and_validity(
    artifact_evidence: dict[str, Any],
    *,
    candidate_names: list[str],
    spec: ConfirmationSpec,
    audit: _Audit,
) -> None:
    environment = _mapping(
        artifact_evidence.get("environment"), label="artifact environment", audit=audit
    )
    audit.equal(
        environment.get("tracked_worktree_clean"), True, "artifact tracked_worktree_clean"
    )
    schedule = _mapping(artifact_evidence.get("schedule"), label="artifact schedule", audit=audit)
    audit.equal(schedule.get("seed"), spec.seed, "artifact schedule seed")
    audit.equal(schedule.get("phase"), "confirmation", "artifact schedule phase")
    audit.equal(schedule.get("row_count"), len(spec.task_ids), "artifact schedule row_count")
    audit.equal(schedule.get("group_size"), spec.group_size, "artifact schedule group_size")
    audit.equal(
        schedule.get("candidate_order"), candidate_names, "artifact schedule candidate order"
    )
    audit.equal(
        schedule.get("scored_first_code_token_from_prefill"),
        True,
        "artifact first-token scoring flag",
    )
    audit.equal(
        schedule.get("candidate_generated_tokens_fed_back"),
        False,
        "artifact candidate feedback flag",
    )
    audit.check(
        _is_finite_number(schedule.get("elapsed_wall_seconds_not_a_latency_benchmark"))
        and float(schedule["elapsed_wall_seconds_not_a_latency_benchmark"]) >= 0,
        "artifact elapsed wall time must be a finite non-negative number",
    )
    resumed = schedule.get("resumed_task_count")
    audit.check(
        isinstance(resumed, int)
        and not isinstance(resumed, bool)
        and 0 <= resumed <= len(spec.task_ids),
        "artifact resumed_task_count is outside the frozen task range",
    )
    audit.check(
        _valid_sha256(schedule.get("run_signature_sha256")),
        "artifact run signature SHA256 is invalid",
    )
    audit.check(
        _valid_sha256(schedule.get("final_checkpoint_sha256")),
        "artifact final checkpoint SHA256 is invalid",
    )

    validity = _mapping(artifact_evidence.get("validity"), label="artifact validity", audit=audit)
    audit.equal(
        validity.get("configured_gdn_layer_indices"),
        list(spec.gdn_layer_indices),
        "configured Gated DeltaNet layer indices",
    )
    audit.equal(
        validity.get("reference_recurrent_state_bytes"),
        spec.reference_state_bytes,
        "reference recurrent-state bytes",
    )
    preflight = _mapping(
        validity.get("packed_qdq_preflight"), label="packed/QDQ preflight", audit=audit
    )
    audit.equal(preflight.get("passed"), True, "packed/QDQ preflight passed flag")
    audit.equal(
        preflight.get("absolute_tolerance"),
        spec.preflight_tolerance,
        "packed/QDQ preflight tolerance",
    )
    audit.equal(preflight.get("task_id"), spec.preflight_task_id, "packed/QDQ preflight task")
    maximums = _mapping(
        preflight.get("maximum_absolute_difference_by_candidate"),
        label="packed/QDQ preflight candidate differences",
        audit=audit,
    )
    audit.equal(
        set(maximums), set(candidate_names), "packed/QDQ preflight candidate set"
    )
    for name, value in maximums.items():
        audit.check(
            _is_finite_number(value) and 0 <= float(value) <= spec.preflight_tolerance,
            f"packed/QDQ preflight difference for {name} exceeds the frozen tolerance",
        )


_PER_TASK_KEYS = {
    "task_id",
    "code_tokens",
    "token_count",
    "mean_kl",
    "cvar95_kl",
    "max_kl",
    "top1_agreement",
    "reference_nll",
    "candidate_nll",
    "delta_nll",
}
_TOKEN_WEIGHTED_KEYS = {
    "token_count",
    "mean_kl",
    "cvar95_kl",
    "max_kl",
    "top1_agreement",
    "reference_nll",
    "candidate_nll",
    "delta_nll",
}


def _verify_candidates(
    artifact_evidence: dict[str, Any],
    *,
    candidate_plan: list[dict[str, Any]],
    token_manifest: list[dict[str, Any]],
    spec: ConfirmationSpec,
    audit: _Audit,
) -> dict[str, list[dict[str, Any]]]:
    candidates = _mapping(
        artifact_evidence.get("candidates"), label="artifact candidates", audit=audit
    )
    candidate_names = [str(policy.get("name")) for policy in candidate_plan]
    audit.equal(set(candidates), set(candidate_names), "artifact candidate set")
    membership = _quartile_membership(token_manifest)
    expected_task_ids = list(spec.task_ids)
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    reference_token_nll: Any = None

    for policy in candidate_plan:
        name = str(policy["name"])
        candidate = _mapping(candidates.get(name), label=f"candidate {name}", audit=audit)
        audit.equal(candidate.get("policy"), policy, f"candidate {name} policy")
        expected_bytes = _expected_candidate_bytes(policy, spec)
        storage = _mapping(candidate.get("storage"), label=f"candidate {name} storage", audit=audit)
        expected_storage_keys = {
            "resident_bytes",
            "full_precision_equivalent_bytes",
            "resident_compression_ratio",
            "largest_transient_state_bytes",
            "physical_reduction_realized",
            "expected_resident_bytes",
            "exact_byte_gate",
        }
        audit.equal(set(storage), expected_storage_keys, f"candidate {name} storage fields")
        audit.equal(
            storage.get("resident_bytes"), expected_bytes, f"candidate {name} resident bytes"
        )
        audit.equal(
            storage.get("expected_resident_bytes"),
            expected_bytes,
            f"candidate {name} expected resident bytes",
        )
        audit.equal(
            storage.get("full_precision_equivalent_bytes"),
            spec.reference_state_bytes,
            f"candidate {name} full-precision equivalent bytes",
        )
        audit.equal(
            storage.get("largest_transient_state_bytes"),
            spec.largest_transient_state_bytes,
            f"candidate {name} largest transient state bytes",
        )
        audit.equal(
            storage.get("physical_reduction_realized"),
            True,
            f"candidate {name} physical reduction flag",
        )
        audit.equal(storage.get("exact_byte_gate"), True, f"candidate {name} exact-byte gate")
        audit.check(
            _is_finite_number(storage.get("resident_compression_ratio"))
            and math.isclose(
                float(storage["resident_compression_ratio"]),
                spec.reference_state_bytes / expected_bytes,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"candidate {name} compression ratio is inconsistent with resident bytes",
        )

        task_rows_value = _list(
            candidate.get("per_task"), label=f"candidate {name} per_task", audit=audit
        )
        task_rows = [dict(row) for row in task_rows_value if isinstance(row, dict)]
        audit.equal(len(task_rows), len(spec.task_ids), f"candidate {name} task count")
        audit.equal(
            [row.get("task_id") for row in task_rows],
            expected_task_ids,
            f"candidate {name} task IDs",
        )
        for index, (row, token_row) in enumerate(
            zip(task_rows, token_manifest, strict=False)
        ):
            audit.equal(set(row), _PER_TASK_KEYS, f"candidate {name} task {index} fields")
            audit.equal(
                row.get("code_tokens"),
                token_row.get("code_tokens"),
                f"candidate {name} task {index} code token count",
            )
            audit.equal(
                row.get("token_count"),
                token_row.get("code_tokens"),
                f"candidate {name} task {index} scored token count",
            )
            for metric in _PER_TASK_KEYS - {"task_id", "code_tokens", "token_count"}:
                audit.check(
                    _is_finite_number(row.get(metric)),
                    f"candidate {name} task {index} {metric} is not finite numeric data",
                )
            if all(
                _is_finite_number(row.get(key))
                for key in ("candidate_nll", "reference_nll", "delta_nll")
            ):
                audit.check(
                    math.isclose(
                        float(row["candidate_nll"]) - float(row["reference_nll"]),
                        float(row["delta_nll"]),
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ),
                    f"candidate {name} task {index} delta_nll is inconsistent",
                )
        rows_by_candidate[name] = task_rows

        token_weighted = _mapping(
            candidate.get("token_weighted"),
            label=f"candidate {name} token_weighted",
            audit=audit,
        )
        audit.equal(
            set(token_weighted), _TOKEN_WEIGHTED_KEYS, f"candidate {name} token metrics fields"
        )
        audit.equal(
            token_weighted.get("token_count"),
            spec.code_token_count,
            f"candidate {name} scored token count",
        )
        for metric in _TOKEN_WEIGHTED_KEYS - {"token_count"}:
            audit.check(
                _is_finite_number(token_weighted.get(metric)),
                f"candidate {name} token metric {metric} is not finite numeric data",
            )
        if reference_token_nll is None:
            reference_token_nll = token_weighted.get("reference_nll")
        else:
            audit.equal(
                token_weighted.get("reference_nll"),
                reference_token_nll,
                f"candidate {name} token reference NLL",
            )
        if all(
            _is_finite_number(token_weighted.get(key))
            for key in ("candidate_nll", "reference_nll", "delta_nll")
        ):
            audit.check(
                math.isclose(
                    float(token_weighted["candidate_nll"])
                    - float(token_weighted["reference_nll"]),
                    float(token_weighted["delta_nll"]),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"candidate {name} token-weighted delta_nll is inconsistent",
            )

        if task_rows:
            expected_macro, expected_distribution, expected_quartiles = (
                _candidate_task_aggregates(task_rows, membership)
            )
            _check_approximately_equal(
                audit,
                candidate.get("task_macro"),
                expected_macro,
                f"candidate {name} task macro",
            )
            _check_approximately_equal(
                audit,
                candidate.get("task_delta_nll_distribution"),
                expected_distribution,
                f"candidate {name} task delta distribution",
            )
            _check_approximately_equal(
                audit,
                candidate.get("by_code_length_quartile"),
                expected_quartiles,
                f"candidate {name} length quartiles",
            )
    return rows_by_candidate


def _recompute_contrasts_and_outcome(
    artifact_evidence: dict[str, Any],
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    token_metrics_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
    all_values_finite_gate: bool | None = None,
    exact_resident_bytes_gate: bool | None = None,
    spec: ConfirmationSpec,
    audit: _Audit,
) -> dict[str, Any] | None:
    required = {
        "uniform_int4_nearest",
        "read_risk_l0_nearest",
        "random_l18_nearest",
        "random_l4_nearest",
        "random_l13_nearest",
    }
    if not required.issubset(rows_by_candidate):
        audit.errors.append("artifact is missing candidates required for frozen contrasts")
        return None
    if any(len(rows_by_candidate[name]) != len(spec.task_ids) for name in required):
        audit.errors.append("frozen contrasts cannot be recomputed from incomplete task rows")
        return None

    uniform_delta = [
        float(row["delta_nll"]) for row in rows_by_candidate["uniform_int4_nearest"]
    ]
    primary_delta = [
        float(row["delta_nll"]) for row in rows_by_candidate["read_risk_l0_nearest"]
    ]
    random_names = ("random_l18_nearest", "random_l4_nearest", "random_l13_nearest")
    random_mean_delta = [
        fmean(float(rows_by_candidate[name][index]["delta_nll"]) for name in random_names)
        for index in range(len(spec.task_ids))
    ]
    macro_uniform = fmean(uniform_delta)
    macro_primary = fmean(primary_delta)
    relative_reduction = (
        (macro_uniform - macro_primary) / macro_uniform if macro_uniform > 0 else None
    )
    uniform_bootstrap = paired_bootstrap_mean_improvement(
        uniform_delta,
        primary_delta,
        samples=spec.bootstrap_samples,
        confidence=spec.confidence,
        seed=spec.seed,
    )
    random_bootstrap = paired_bootstrap_mean_improvement(
        random_mean_delta,
        primary_delta,
        samples=spec.bootstrap_samples,
        confidence=spec.confidence,
        seed=spec.seed,
    )
    expected_contrasts = {
        "primary_vs_uniform_int4": {
            "uniform_macro_delta_nll": macro_uniform,
            "primary_macro_delta_nll": macro_primary,
            "relative_reduction": relative_reduction,
            "paired_bootstrap": uniform_bootstrap,
        },
        "primary_vs_mean_random_equal_byte": {"paired_bootstrap": random_bootstrap},
    }
    _check_approximately_equal(
        audit,
        artifact_evidence.get("contrasts"),
        expected_contrasts,
        "artifact contrasts",
        tolerance=1e-12,
    )

    candidates = _mapping(
        artifact_evidence.get("candidates"), label="artifact candidates", audit=audit
    )
    if token_metrics_by_candidate is None:
        token_metrics_by_candidate = {
            name: _mapping(
                _mapping(candidate, label=f"candidate {name}", audit=audit).get(
                    "token_weighted"
                ),
                label=f"candidate {name} token metrics",
                audit=audit,
            )
            for name, candidate in candidates.items()
        }
    primary_token = _mapping(
        token_metrics_by_candidate.get("read_risk_l0_nearest"),
        label="primary token metrics",
        audit=audit,
    )
    uniform_token = _mapping(
        token_metrics_by_candidate.get("uniform_int4_nearest"),
        label="uniform INT4 token metrics",
        audit=audit,
    )
    if all_values_finite_gate is None:
        all_values_finite_gate = True
    if exact_resident_bytes_gate is None:
        exact_resident_bytes_gate = all(
            _mapping(candidate, label=f"candidate {name}", audit=audit)
            .get("storage", {})
            .get("exact_byte_gate")
            is True
            for name, candidate in candidates.items()
        )
    legacy_gates = {
        "all_values_finite": all_values_finite_gate,
        "exact_resident_bytes": exact_resident_bytes_gate,
        "primary_macro_delta_nll_reduction_at_least_15_percent": (
            relative_reduction is not None and relative_reduction >= 0.15
        ),
        "equal_byte_bootstrap_interval_above_zero": (
            random_bootstrap["confidence_interval"][0] > 0
        ),
        "primary_mean_token_kl_lower_than_uniform_int4": (
            float(primary_token["mean_kl"]) < float(uniform_token["mean_kl"])
        ),
        "primary_cvar95_token_kl_lower_than_uniform_int4": (
            float(primary_token["cvar95_kl"]) < float(uniform_token["cvar95_kl"])
        ),
        "primary_top1_not_lower_than_uniform_int4": (
            float(primary_token["top1_agreement"])
            >= float(uniform_token["top1_agreement"])
        ),
    }
    recorded = _mapping(
        artifact_evidence.get("continuation_decision"),
        label="artifact continuation decision",
        audit=audit,
    )
    audit.equal(recorded.get("gates"), legacy_gates, "recorded continuation gates")
    audit.equal(
        recorded.get("all_gates_pass"),
        all(legacy_gates.values()),
        "recorded all_gates_pass",
    )
    audit.equal(
        recorded.get("confirmation_permitted"),
        False,
        "confirmation-phase confirmation_permitted",
    )

    quality_gates = {
        **legacy_gates,
        "primary_vs_uniform_bootstrap_interval_above_zero": (
            uniform_bootstrap["confidence_interval"][0] > 0
        ),
    }
    return {
        "quality_gates": quality_gates,
        "quality_hypothesis_pass": all(quality_gates.values()),
        "primary_vs_uniform_int4": expected_contrasts["primary_vs_uniform_int4"],
        "primary_vs_mean_random_equal_byte": expected_contrasts[
            "primary_vs_mean_random_equal_byte"
        ],
    }


def _verify_checkpoint(
    checkpoint_loaded: _LoadedJSON,
    artifact_evidence: dict[str, Any],
    manifest_evidence: dict[str, Any],
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    token_manifest: Sequence[Mapping[str, Any]],
    *,
    spec: ConfirmationSpec,
    audit: _Audit,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    bool,
    bool,
]:
    checkpoint = _mapping(checkpoint_loaded.document, label="checkpoint root", audit=audit)
    audit.equal(
        set(checkpoint),
        {"schema_version", "run_signature_sha256", "state_sha256", "state"},
        "checkpoint root fields",
    )
    audit.equal(checkpoint.get("schema_version"), 1, "checkpoint schema_version")
    audit.check(
        _valid_sha256(checkpoint.get("run_signature_sha256")),
        "checkpoint run signature SHA256 is invalid",
    )
    state = _mapping(checkpoint.get("state"), label="checkpoint state", audit=audit)
    audit.equal(
        set(state),
        {
            "completed_task_ids",
            "global_values",
            "per_task",
            "storage_by_candidate",
            "reference_state_bytes",
            "elapsed_wall_seconds",
        },
        "checkpoint state fields",
    )
    computed_state_sha256 = _sha256_canonical(state)
    audit.check(
        _valid_sha256(checkpoint.get("state_sha256")), "checkpoint state SHA256 is invalid"
    )
    audit.equal(
        checkpoint.get("state_sha256"), computed_state_sha256, "checkpoint canonical state SHA256"
    )

    schedule = _mapping(artifact_evidence.get("schedule"), label="artifact schedule", audit=audit)
    audit.equal(
        checkpoint_loaded.file_sha256,
        schedule.get("final_checkpoint_sha256"),
        "checkpoint file SHA256",
    )
    audit.equal(
        checkpoint.get("run_signature_sha256"),
        schedule.get("run_signature_sha256"),
        "checkpoint/artifact run signature",
    )

    manifest_source = _mapping(
        manifest_evidence.get("source"), label="prepared manifest source", audit=audit
    )
    artifact_source = _mapping(
        artifact_evidence.get("source"), label="artifact source", audit=audit
    )
    candidate_plan = _list(
        _mapping(
            manifest_evidence.get("schedule"), label="prepared manifest schedule", audit=audit
        ).get("candidate_plan"),
        label="prepared candidate plan",
        audit=audit,
    )
    signature_evidence = {
        "phase": "confirmation",
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "repository_commit": spec.source_commit,
        "calibration_evidence_sha256": spec.calibration_evidence_sha256,
        "prepared_manifest_evidence_sha256": spec.prepared_manifest_evidence_sha256,
        "token_manifest_sha256": spec.token_manifest_sha256,
        "group_size": spec.group_size,
        "candidate_plan": candidate_plan,
    }
    del manifest_source, artifact_source
    audit.equal(
        _sha256_canonical(signature_evidence),
        schedule.get("run_signature_sha256"),
        "recomputed frozen run signature",
    )

    audit.equal(
        state.get("completed_task_ids"), list(spec.task_ids), "checkpoint completed task IDs"
    )
    audit.equal(
        state.get("reference_state_bytes"),
        spec.reference_state_bytes,
        "checkpoint reference recurrent-state bytes",
    )
    audit.check(
        _is_finite_number(state.get("elapsed_wall_seconds"))
        and float(state["elapsed_wall_seconds"]) >= 0,
        "checkpoint elapsed wall time must be finite and non-negative",
    )

    plan_names = [str(policy["name"]) for policy in candidate_plan]
    checkpoint_tasks = _mapping(state.get("per_task"), label="checkpoint per_task", audit=audit)
    checkpoint_storage = _mapping(
        state.get("storage_by_candidate"), label="checkpoint storage", audit=audit
    )
    global_values = _mapping(
        state.get("global_values"), label="checkpoint global_values", audit=audit
    )
    audit.equal(set(checkpoint_tasks), set(plan_names), "checkpoint per-task candidate set")
    audit.equal(set(checkpoint_storage), set(plan_names), "checkpoint storage candidate set")
    audit.equal(set(global_values), set(plan_names), "checkpoint global-values candidate set")

    reference_values: list[Any] | None = None
    code_counts = [int(row["code_tokens"]) for row in token_manifest]
    raw_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    raw_token_metrics_by_candidate: dict[str, dict[str, Any]] = {}
    raw_values_finite = True
    checkpoint_exact_bytes = True
    for policy in candidate_plan:
        name = str(policy["name"])
        artifact_rows = list(rows_by_candidate.get(name, []))
        audit.equal(
            checkpoint_tasks.get(name), artifact_rows, f"checkpoint/artifact task rows for {name}"
        )
        artifact_storage = _mapping(
            _mapping(
                _mapping(
                    artifact_evidence.get("candidates"), label="artifact candidates", audit=audit
                ).get(name),
                label=f"candidate {name}",
                audit=audit,
            ).get("storage"),
            label=f"candidate {name} storage",
            audit=audit,
        )
        base_storage = {
            key: artifact_storage.get(key)
            for key in (
                "resident_bytes",
                "full_precision_equivalent_bytes",
                "resident_compression_ratio",
                "largest_transient_state_bytes",
                "physical_reduction_realized",
            )
        }
        audit.equal(
            checkpoint_storage.get(name), base_storage, f"checkpoint/artifact storage for {name}"
        )
        checkpoint_storage_value = checkpoint_storage.get(name)
        expected_bytes = _expected_candidate_bytes(policy, spec)
        storage_is_exact = (
            isinstance(checkpoint_storage_value, dict)
            and checkpoint_storage_value.get("resident_bytes") == expected_bytes
            and checkpoint_storage_value.get("full_precision_equivalent_bytes")
            == spec.reference_state_bytes
            and checkpoint_storage_value.get("largest_transient_state_bytes")
            == spec.largest_transient_state_bytes
            and checkpoint_storage_value.get("physical_reduction_realized") is True
        )
        checkpoint_exact_bytes = checkpoint_exact_bytes and storage_is_exact

        values = _mapping(
            global_values.get(name), label=f"checkpoint global values for {name}", audit=audit
        )
        audit.equal(
            set(values),
            {"kl", "reference_nll", "candidate_nll", "top1"},
            f"checkpoint global metric fields for {name}",
        )
        candidate_values_valid = True
        for metric in ("kl", "reference_nll", "candidate_nll", "top1"):
            metric_values = _list(
                values.get(metric), label=f"checkpoint {name} {metric}", audit=audit
            )
            audit.equal(
                len(metric_values), spec.code_token_count, f"checkpoint {name} {metric} length"
            )
            if metric == "top1":
                metric_is_valid = all(isinstance(value, bool) for value in metric_values)
                audit.check(
                    metric_is_valid,
                    f"checkpoint {name} top1 values must all be booleans",
                )
            else:
                metric_is_valid = all(_is_finite_number(value) for value in metric_values)
                audit.check(
                    metric_is_valid,
                    f"checkpoint {name} {metric} contains non-finite or non-numeric data",
                )
            raw_values_finite = raw_values_finite and metric_is_valid
            candidate_values_valid = candidate_values_valid and metric_is_valid
        if reference_values is None:
            reference_values = list(values.get("reference_nll", []))
        else:
            audit.equal(
                values.get("reference_nll"),
                reference_values,
                f"checkpoint shared reference NLL for {name}",
            )

        if all(
            isinstance(values.get(key), list)
            and len(values[key]) == spec.code_token_count
            for key in ("kl", "reference_nll", "candidate_nll", "top1")
        ) and candidate_values_valid:
            fidelity = TokenFidelity(
                kl=torch.tensor(values["kl"], dtype=torch.float32),
                reference_nll=torch.tensor(values["reference_nll"], dtype=torch.float32),
                candidate_nll=torch.tensor(values["candidate_nll"], dtype=torch.float32),
                top1_agreement=torch.tensor(values["top1"], dtype=torch.bool),
            )
            expected_token = fidelity_summary(fidelity)
            raw_token_metrics_by_candidate[name] = dict(expected_token)
            actual_token = _mapping(
                _mapping(
                    _mapping(
                        artifact_evidence.get("candidates"),
                        label="artifact candidates",
                        audit=audit,
                    ).get(name),
                    label=f"candidate {name}",
                    audit=audit,
                ).get("token_weighted"),
                label=f"candidate {name} token metrics",
                audit=audit,
            )
            _check_approximately_equal(
                audit,
                actual_token,
                expected_token,
                f"candidate {name} token metrics",
            )
            offset = 0
            reconstructed_rows: list[dict[str, Any]] = []
            for task_id, code_count in zip(spec.task_ids, code_counts, strict=True):
                task_fidelity = TokenFidelity(
                    kl=fidelity.kl[offset : offset + code_count],
                    reference_nll=fidelity.reference_nll[offset : offset + code_count],
                    candidate_nll=fidelity.candidate_nll[offset : offset + code_count],
                    top1_agreement=fidelity.top1_agreement[offset : offset + code_count],
                )
                reconstructed_rows.append(
                    {
                        "task_id": task_id,
                        "code_tokens": code_count,
                        **fidelity_summary(task_fidelity),
                    }
                )
                offset += code_count
            _check_approximately_equal(
                audit,
                artifact_rows,
                reconstructed_rows,
                f"candidate {name} per-task metrics",
            )
            raw_rows_by_candidate[name] = reconstructed_rows

    return (
        raw_rows_by_candidate,
        raw_token_metrics_by_candidate,
        raw_values_finite,
        checkpoint_exact_bytes,
    )


def verify_mbpp_confirmation(
    artifact_path: str | Path,
    prepared_manifest_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    expected_artifact_sha256: str | None = None,
    expected_artifact_evidence_sha256: str | None = None,
    spec: ConfirmationSpec = FROZEN_V02_CONFIRMATION,
) -> dict[str, Any]:
    """Verify the frozen confirmation artifact without model or network access.

    Supplying ``checkpoint_path`` additionally verifies the raw token arrays,
    canonical checkpoint state, final file hash, and frozen run signature. In
    artifact-only mode, both external artifact hashes are required before a
    quality outcome is trusted. A valid negative confirmation result remains
    valid evidence; only malformed or inconsistent inputs are invalid.
    """

    base_audit = _Audit(errors=[])
    artifact_loaded = _load_strict_json(
        artifact_path, label="confirmation artifact", audit=base_audit
    )
    manifest_loaded = _load_strict_json(
        prepared_manifest_path, label="prepared manifest", audit=base_audit
    )
    hashes: dict[str, Any] = {
        "artifact_file_sha256": (
            artifact_loaded.file_sha256 if artifact_loaded is not None else None
        ),
        "artifact_canonical_evidence_sha256": None,
        "prepared_manifest_file_sha256": (
            manifest_loaded.file_sha256 if manifest_loaded is not None else None
        ),
        "prepared_manifest_canonical_evidence_sha256": None,
        "checkpoint_file_sha256": None,
        "checkpoint_state_sha256": None,
    }
    expected_file_hash = (
        expected_artifact_sha256.lower()
        if isinstance(expected_artifact_sha256, str)
        else expected_artifact_sha256
    )
    expected_evidence_hash = (
        expected_artifact_evidence_sha256.lower()
        if isinstance(expected_artifact_evidence_sha256, str)
        else expected_artifact_evidence_sha256
    )
    both_artifact_hashes_supplied = (
        expected_artifact_sha256 is not None
        and expected_artifact_evidence_sha256 is not None
    )
    artifact_outcome: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    artifact_evidence: dict[str, Any] = {}
    manifest_evidence: dict[str, Any] = {}
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    token_manifest: list[dict[str, Any]] = []

    if artifact_loaded is not None and manifest_loaded is not None:
        try:
            artifact_evidence = _verify_envelope(
                artifact_loaded,
                label="confirmation artifact",
                artifact_kind="recurquant_mbpp_teacher_forced_evaluation",
                audit=base_audit,
            )
            manifest_evidence = _verify_envelope(
                manifest_loaded,
                label="prepared manifest",
                artifact_kind="recurquant_mbpp_prepared_manifest",
                audit=base_audit,
            )
            hashes["artifact_canonical_evidence_sha256"] = _sha256_canonical(
                artifact_evidence
            )
            hashes["prepared_manifest_canonical_evidence_sha256"] = _sha256_canonical(
                manifest_evidence
            )
            base_audit.equal(
                manifest_loaded.file_sha256,
                spec.prepared_manifest_file_sha256,
                "prepared manifest file SHA256",
            )
            base_audit.equal(
                hashes["prepared_manifest_canonical_evidence_sha256"],
                spec.prepared_manifest_evidence_sha256,
                "prepared manifest frozen evidence SHA256",
            )
            if expected_artifact_sha256 is not None:
                base_audit.check(
                    _valid_sha256(expected_file_hash),
                    "expected artifact file SHA256 is invalid",
                )
                base_audit.equal(
                    artifact_loaded.file_sha256,
                    expected_file_hash,
                    "expected artifact file SHA256",
                )
            if expected_artifact_evidence_sha256 is not None:
                base_audit.check(
                    _valid_sha256(expected_evidence_hash),
                    "expected artifact evidence SHA256 is invalid",
                )
                base_audit.equal(
                    hashes["artifact_canonical_evidence_sha256"],
                    expected_evidence_hash,
                    "expected artifact evidence SHA256",
                )

            _, token_manifest = _verify_manifest_and_source(
                manifest_evidence,
                artifact_evidence,
                spec=spec,
                audit=base_audit,
            )
            manifest_schedule = _mapping(
                manifest_evidence.get("schedule"),
                label="prepared manifest schedule",
                audit=base_audit,
            )
            candidate_plan_value = _list(
                manifest_schedule.get("candidate_plan"),
                label="prepared candidate plan",
                audit=base_audit,
            )
            candidate_plan = [
                dict(policy) for policy in candidate_plan_value if isinstance(policy, dict)
            ]
            base_audit.equal(
                manifest_schedule.get("row_count"),
                len(spec.task_ids),
                "prepared manifest schedule row_count",
            )
            candidate_names = [str(policy.get("name")) for policy in candidate_plan]
            base_audit.check(
                len(candidate_names) == len(set(candidate_names)) and all(candidate_names),
                "prepared candidate names must be unique and non-empty",
            )
            _verify_schedule_and_validity(
                artifact_evidence,
                candidate_names=candidate_names,
                spec=spec,
                audit=base_audit,
            )
            rows_by_candidate = _verify_candidates(
                artifact_evidence,
                candidate_plan=candidate_plan,
                token_manifest=token_manifest,
                spec=spec,
                audit=base_audit,
            )
            if checkpoint_path is None:
                artifact_outcome = _recompute_contrasts_and_outcome(
                    artifact_evidence,
                    rows_by_candidate,
                    spec=spec,
                    audit=base_audit,
                )
        except (
            KeyError,
            IndexError,
            OverflowError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            base_audit.errors.append(f"could not complete artifact verification: {exc}")

    checkpoint_audit = _Audit(errors=[])
    checkpoint_loaded: _LoadedJSON | None = None
    if checkpoint_path is not None:
        checkpoint_loaded = _load_strict_json(
            checkpoint_path, label="final checkpoint", audit=checkpoint_audit
        )
        if checkpoint_loaded is not None:
            hashes["checkpoint_file_sha256"] = checkpoint_loaded.file_sha256
            checkpoint_document = checkpoint_loaded.document
            if isinstance(checkpoint_document, dict) and isinstance(
                checkpoint_document.get("state"), dict
            ):
                hashes["checkpoint_state_sha256"] = _sha256_canonical(
                    checkpoint_document["state"]
                )
            if artifact_evidence and manifest_evidence and token_manifest:
                try:
                    (
                        raw_rows_by_candidate,
                        raw_token_metrics_by_candidate,
                        raw_values_finite,
                        checkpoint_exact_bytes,
                    ) = _verify_checkpoint(
                        checkpoint_loaded,
                        artifact_evidence,
                        manifest_evidence,
                        rows_by_candidate,
                        token_manifest,
                        spec=spec,
                        audit=checkpoint_audit,
                    )
                    outcome = _recompute_contrasts_and_outcome(
                        artifact_evidence,
                        raw_rows_by_candidate,
                        token_metrics_by_candidate=raw_token_metrics_by_candidate,
                        all_values_finite_gate=raw_values_finite,
                        exact_resident_bytes_gate=checkpoint_exact_bytes,
                        spec=spec,
                        audit=checkpoint_audit,
                    )
                except (
                    KeyError,
                    IndexError,
                    OverflowError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    checkpoint_audit.errors.append(
                        f"could not complete checkpoint verification: {exc}"
                    )
            else:
                checkpoint_audit.errors.append(
                    "checkpoint cross-checks require a valid artifact and prepared manifest "
                    "structure"
                )

    artifact_hashes_matched = (
        both_artifact_hashes_supplied
        and artifact_loaded is not None
        and expected_file_hash == artifact_loaded.file_sha256
        and expected_evidence_hash == hashes["artifact_canonical_evidence_sha256"]
    )
    errors = [*base_audit.errors, *checkpoint_audit.errors]
    artifact_manifest_verified = not base_audit.errors
    checkpoint_verified = (
        checkpoint_path is not None
        and checkpoint_loaded is not None
        and not base_audit.errors
        and not checkpoint_audit.errors
    )
    valid = not errors
    if checkpoint_path is None:
        if artifact_hashes_matched and artifact_manifest_verified and artifact_outcome is not None:
            outcome = {
                **artifact_outcome,
                "outcome_verified": True,
                "verification_basis": "externally_anchored_artifact",
            }
        else:
            outcome = {
                "quality_gates": None,
                "quality_hypothesis_pass": None,
                "primary_vs_uniform_int4": None,
                "primary_vs_mean_random_equal_byte": None,
                "outcome_verified": False,
                "verification_basis": "unanchored_artifact",
            }
    elif outcome is not None:
        outcome = {
            **outcome,
            "outcome_verified": checkpoint_verified,
            "verification_basis": "checkpoint_raw_arrays",
        }
    outcome_verified = bool(outcome is not None and outcome.get("outcome_verified") is True)
    quality_pass = outcome.get("quality_hypothesis_pass") if outcome_verified else None
    if not valid:
        result = "invalid"
    elif not outcome_verified:
        result = "unverified"
    elif quality_pass is True:
        result = "pass"
    elif quality_pass is False:
        result = "fail"
    else:
        result = "unverified"
    warnings: list[str] = []
    if checkpoint_path is None:
        warnings.append(
            "checkpoint not supplied; raw token arrays, canonical state, and final checkpoint "
            "file hash were not independently verified"
        )
        if not both_artifact_hashes_supplied:
            warnings.append(
                "artifact-only quality outcomes require both externally published artifact "
                "file and canonical evidence SHA256 values"
            )
    return {
        "artifact_path": str(Path(artifact_path)),
        "prepared_manifest_path": str(Path(prepared_manifest_path)),
        "checkpoint_path": str(Path(checkpoint_path)) if checkpoint_path is not None else None,
        "artifact_manifest_verified": artifact_manifest_verified,
        "checkpoint_verified": checkpoint_verified,
        "outcome_verified": outcome_verified,
        "errors": errors,
        "hashes": hashes,
        "outcome": outcome,
        "result": result,
        "valid": valid,
        "warnings": warnings,
    }
