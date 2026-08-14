"""Deterministic Stage-A evidence reduction and gates for Experiment 013.

The module consumes already-authenticated, per-transition measurements.  It is
deliberately unaware of models, datasets, caches, and devices: those belong to
the sealed evaluator.  Its job is to reject incomplete or reordered inputs,
reduce the fixed twelve-example/nine-method screen, run the prespecified paired
bootstrap, and produce a canonical self-verifying JSON artifact.

Stage A is a falsification screen.  Passing these gates is not confirmation,
selector-superiority evidence, a deployment result, or a breakthrough claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import torch

from .evidence import canonical_json_bytes
from .metrics import tail_mean
from .static_q468 import (
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    STATIC_Q468_UNIFORM_Q4_METHOD,
    STATIC_Q468_UNIFORM_Q8_METHOD,
)
from .static_q468_cache import DYNAMIC_Q468_BASELINE_METHOD

StageAFamily: TypeAlias = Literal["pg19", "ruler", "humaneval_plus"]

STAGE_A_ARTIFACT_KIND: Final = "recurquant_experiment013_stage_a_falsification"
STAGE_A_SCHEMA_VERSION: Final = 1
STAGE_A_ARTIFACT_REVISION: Final = "experiment-013-stage-a-falsification-v1"
STAGE_A_PROFILE: Final = "experiment-013-qwen35-0.8b-stage-a-frozen-v1"

FP32_METHOD: Final = "fp32_reference"
UNIFORM_RHT_Q4_METHOD: Final = STATIC_Q468_UNIFORM_Q4_METHOD
UNIFORM_RHT_Q8_METHOD: Final = STATIC_Q468_UNIFORM_Q8_METHOD

STAGE_A_FAMILY_ORDER: Final[tuple[StageAFamily, ...]] = (
    "pg19",
    "ruler",
    "humaneval_plus",
)
STAGE_A_METHOD_ORDER: Final = (
    FP32_METHOD,
    UNIFORM_RHT_Q4_METHOD,
    UNIFORM_RHT_Q8_METHOD,
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    DYNAMIC_Q468_BASELINE_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
)
DECISION_COMPARATOR_ORDER: Final = (
    DYNAMIC_Q468_BASELINE_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
)

STAGE_A_EXAMPLES_PER_FAMILY: Final = 4
STAGE_A_EXAMPLE_COUNT: Final = 12
BOOTSTRAP_SEED: Final = 2_339
BOOTSTRAP_SAMPLES: Final = 10_000
BOOTSTRAP_UPPER_QUANTILE: Final = 0.95
BOOTSTRAP_RNG_ALGORITHM: Final = "numpy-pcg64-integers-v1"
BOOTSTRAP_QUANTILE_ALGORITHM: Final = (
    "nearest-rank: ascending stable sort; zero-based index ceil(q*N)-1; no interpolation"
)
MAX_ONE_SIDED_95_UPPER_EXCESS_NLL: Final = 0.010
MAX_FAMILY_POINT_EXCESS_NLL: Final = 0.015
MAX_FAMILY_MACRO_TOP1_TRAIL: Final = 0.005
TAIL_KL_FRACTION: Final = 0.05

CLAIM_BOUNDARY: Final = (
    "Stage A is a falsification screen only. Passage is not confirmation, selector "
    "superiority evidence, deployment evidence, state of the art, or a breakthrough claim."
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OUTER_FIELDS = frozenset(
    {"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"}
)
_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_kind",
        "artifact_revision",
        "claim_boundary",
        "decision",
        "dependencies",
        "example_summaries",
        "examples",
        "method_roles",
        "method_summaries",
        "metric_contract",
        "profile",
        "schema_version",
        "token_rows",
    }
)
_DEPENDENCY_FIELDS = frozenset(
    {"stage_a_calibration_binding_file_sha256", "stage_a_identity_file_sha256"}
)
_EXAMPLE_FIELDS = frozenset(
    {
        "canonical_id",
        "continuation_token_count",
        "family",
        "identity_record_sha256",
        "selection_rank",
    }
)
_TOKEN_ROW_FIELDS = frozenset(
    {
        "canonical_id",
        "family",
        "identity_record_sha256",
        "kl",
        "method_id",
        "method_nll",
        "reference_nll",
        "selection_rank",
        "top1_agreement",
        "transition_index",
    }
)
_METRIC_FIELDS = (
    "excess_nll",
    "reference_nll",
    "method_nll",
    "mean_kl",
    "cvar95_kl",
    "max_kl",
    "top1_agreement",
)


@dataclass(frozen=True, slots=True)
class StageAExample:
    """One frozen Stage-A example and its cache-exposed metric span."""

    family: StageAFamily
    canonical_id: str
    selection_rank: int
    continuation_token_count: int
    identity_record_sha256: str

    @property
    def transition_count(self) -> int:
        return self.continuation_token_count - 1


@dataclass(frozen=True, slots=True)
class StageATokenRow:
    """One method's metrics for one identity-bound cache-exposed transition."""

    family: StageAFamily
    canonical_id: str
    selection_rank: int
    identity_record_sha256: str
    method_id: str
    transition_index: int
    reference_nll: float
    method_nll: float
    kl: float
    top1_agreement: bool


@dataclass(frozen=True, slots=True)
class StageAEvidenceArtifact:
    """Strictly decoded Stage-A evidence with both content commitments."""

    evidence: Mapping[str, Any]
    canonical_evidence_sha256: str
    file_sha256: str
    serialized_bytes: bytes
    _verified_passed: bool

    @property
    def passed(self) -> bool:
        return self._verified_passed


def _deep_freeze_json(value: Any) -> Any:
    """Return an immutable view of a JSON value tree.

    The verifier must not hand callers a mutable object whose later mutation
    can change the meaning of an already authenticated artifact.
    """

    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze_json(nested) for nested in value)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields drifted (missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )


def _require_canonical_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must use NFC Unicode normalization")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return 0.0 if result == 0.0 else result


def _normalized_examples(examples: Sequence[StageAExample]) -> tuple[StageAExample, ...]:
    if isinstance(examples, (str, bytes, bytearray)) or not isinstance(examples, Sequence):
        raise TypeError("examples must be a sequence")
    if len(examples) != STAGE_A_EXAMPLE_COUNT:
        raise ValueError("Stage A requires exactly twelve examples")

    normalized: list[StageAExample] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(examples):
        if not isinstance(raw, StageAExample):
            raise TypeError(f"examples[{index}] must be a StageAExample")
        expected_family = STAGE_A_FAMILY_ORDER[index // STAGE_A_EXAMPLES_PER_FAMILY]
        expected_rank = index % STAGE_A_EXAMPLES_PER_FAMILY
        if raw.family != expected_family or raw.selection_rank != expected_rank:
            raise ValueError(
                "Stage-A examples must be ordered by frozen family order and selection rank"
            )
        canonical_id = _require_canonical_id(raw.canonical_id, name=f"examples[{index}].id")
        continuation = _require_int(
            raw.continuation_token_count,
            name=f"examples[{index}].continuation_token_count",
            minimum=2,
        )
        identity_hash = _require_sha256(
            raw.identity_record_sha256,
            name=f"examples[{index}].identity_record_sha256",
        )
        identity = (raw.family, canonical_id)
        if identity in identities:
            raise ValueError(f"duplicate Stage-A example identity: {identity}")
        identities.add(identity)
        normalized.append(
            StageAExample(
                family=raw.family,
                canonical_id=canonical_id,
                selection_rank=expected_rank,
                continuation_token_count=continuation,
                identity_record_sha256=identity_hash,
            )
        )
    return tuple(normalized)


def _normalize_row(row: StageATokenRow, *, index: int) -> StageATokenRow:
    if not isinstance(row, StageATokenRow):
        raise TypeError(f"token_rows[{index}] must be a StageATokenRow")
    family = cast(StageAFamily, row.family)
    if family not in STAGE_A_FAMILY_ORDER:
        raise ValueError(f"token_rows[{index}].family is unknown")
    canonical_id = _require_canonical_id(row.canonical_id, name=f"token_rows[{index}].id")
    rank = _require_int(row.selection_rank, name=f"token_rows[{index}].selection_rank")
    identity_hash = _require_sha256(
        row.identity_record_sha256,
        name=f"token_rows[{index}].identity_record_sha256",
    )
    if row.method_id not in STAGE_A_METHOD_ORDER:
        raise ValueError(f"token_rows[{index}].method_id is not in the frozen method order")
    transition = _require_int(
        row.transition_index,
        name=f"token_rows[{index}].transition_index",
    )
    reference_nll = _require_finite(
        row.reference_nll,
        name=f"token_rows[{index}].reference_nll",
        nonnegative=True,
    )
    method_nll = _require_finite(
        row.method_nll,
        name=f"token_rows[{index}].method_nll",
        nonnegative=True,
    )
    kl = _require_finite(
        row.kl,
        name=f"token_rows[{index}].kl",
        nonnegative=True,
    )
    if not isinstance(row.top1_agreement, bool):
        raise TypeError(f"token_rows[{index}].top1_agreement must be a bool")
    return StageATokenRow(
        family=family,
        canonical_id=canonical_id,
        selection_rank=rank,
        identity_record_sha256=identity_hash,
        method_id=row.method_id,
        transition_index=transition,
        reference_nll=reference_nll,
        method_nll=method_nll,
        kl=kl,
        top1_agreement=row.top1_agreement,
    )


def _normalized_rows(
    examples: tuple[StageAExample, ...], token_rows: Sequence[StageATokenRow]
) -> tuple[StageATokenRow, ...]:
    if isinstance(token_rows, (str, bytes, bytearray)) or not isinstance(token_rows, Sequence):
        raise TypeError("token_rows must be a sequence")
    expected_count = sum(example.transition_count for example in examples) * len(
        STAGE_A_METHOD_ORDER
    )
    if len(token_rows) != expected_count:
        raise ValueError(
            f"Stage-A token-row count differs from the complete grid: "
            f"expected {expected_count}, observed {len(token_rows)}"
        )

    normalized = tuple(_normalize_row(row, index=index) for index, row in enumerate(token_rows))
    cursor = 0
    references: dict[tuple[str, str, int], float] = {}
    for example in examples:
        for method_id in STAGE_A_METHOD_ORDER:
            for transition_index in range(example.transition_count):
                row = normalized[cursor]
                expected_identity = (
                    example.family,
                    example.canonical_id,
                    example.selection_rank,
                    example.identity_record_sha256,
                    method_id,
                    transition_index,
                )
                observed_identity = (
                    row.family,
                    row.canonical_id,
                    row.selection_rank,
                    row.identity_record_sha256,
                    row.method_id,
                    row.transition_index,
                )
                if observed_identity != expected_identity:
                    raise ValueError(
                        f"missing, duplicate, or reordered Stage-A token row at index {cursor}"
                    )
                reference_key = (example.family, example.canonical_id, transition_index)
                recorded_reference = references.setdefault(reference_key, row.reference_nll)
                if row.reference_nll != recorded_reference:
                    raise ValueError(
                        "reference NLL differs across methods for one cache-exposed transition"
                    )
                if method_id == FP32_METHOD:
                    if row.method_nll != row.reference_nll:
                        raise ValueError("FP32 method NLL must exactly equal reference NLL")
                    if not row.top1_agreement:
                        raise ValueError("FP32 top-1 agreement must be true")
                    if row.kl != 0.0:
                        raise ValueError("FP32 self-KL must be exactly zero")
                cursor += 1
    return normalized


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty metric sequence")
    result = math.fsum(values) / len(values)
    return 0.0 if result == 0.0 else result


def _cvar95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty KL sequence")
    # Match the reviewed fidelity_summary contract exactly: values are first
    # rounded to FP32, top-k is selected there, and the tail mean is accumulated
    # in FP32 by Torch.
    tensor = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("KL values must remain finite after FP32 conversion")
    return float(tail_mean(tensor, fraction=TAIL_KL_FRACTION).item())


def _metric_summary(rows: Sequence[StageATokenRow]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty token-row sequence")
    reference_nll = [row.reference_nll for row in rows]
    method_nll = [row.method_nll for row in rows]
    excess = [row.method_nll - row.reference_nll for row in rows]
    kl = [row.kl for row in rows]
    top1 = [1.0 if row.top1_agreement else 0.0 for row in rows]
    return {
        "token_count": len(rows),
        "top1_agreement_count": sum(row.top1_agreement for row in rows),
        "excess_nll": _mean(excess),
        "reference_nll": _mean(reference_nll),
        "method_nll": _mean(method_nll),
        "mean_kl": _mean(kl),
        "cvar95_kl": _cvar95(kl),
        "max_kl": max(kl),
        "top1_agreement": _mean(top1),
    }


def _mean_summaries(summaries: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    if not summaries:
        raise ValueError("cannot macro-average empty summaries")
    return {
        field: _mean([float(summary[field]) for summary in summaries]) for field in _METRIC_FIELDS
    }


def _build_summaries(
    examples: tuple[StageAExample, ...], rows: tuple[StageATokenRow, ...]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_key: dict[tuple[str, str, str], list[StageATokenRow]] = {}
    for row in rows:
        by_key.setdefault((row.family, row.canonical_id, row.method_id), []).append(row)

    per_example: list[dict[str, object]] = []
    for example in examples:
        for method_id in STAGE_A_METHOD_ORDER:
            selected = by_key[(example.family, example.canonical_id, method_id)]
            per_example.append(
                {
                    "family": example.family,
                    "canonical_id": example.canonical_id,
                    "selection_rank": example.selection_rank,
                    "identity_record_sha256": example.identity_record_sha256,
                    "method_id": method_id,
                    "metrics": _metric_summary(selected),
                }
            )

    per_example_lookup = {
        (str(item["family"]), str(item["canonical_id"]), str(item["method_id"])): cast(
            Mapping[str, float | int], item["metrics"]
        )
        for item in per_example
    }
    method_summaries: list[dict[str, object]] = []
    for method_id in STAGE_A_METHOD_ORDER:
        family_summaries: list[dict[str, object]] = []
        for family in STAGE_A_FAMILY_ORDER:
            family_examples = [example for example in examples if example.family == family]
            example_metrics = [
                per_example_lookup[(family, example.canonical_id, method_id)]
                for example in family_examples
            ]
            token_subset = [
                row for row in rows if row.family == family and row.method_id == method_id
            ]
            family_summaries.append(
                {
                    "family": family,
                    "example_count": len(family_examples),
                    "token_count": len(token_subset),
                    "example_macro": _mean_summaries(example_metrics),
                    "token_micro": _metric_summary(token_subset),
                }
            )
        family_macro = _mean_summaries(
            [cast(Mapping[str, float | int], item["example_macro"]) for item in family_summaries]
        )
        method_summaries.append(
            {
                "method_id": method_id,
                "by_family": family_summaries,
                "family_macro": family_macro,
                "token_micro_diagnostic": _metric_summary(
                    [row for row in rows if row.method_id == method_id]
                ),
            }
        )
    return per_example, method_summaries


def _nearest_rank_quantile(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile values must be a non-empty finite vector")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = np.sort(values, kind="stable")
    index = max(0, math.ceil(probability * ordered.size) - 1)
    return float(ordered[index])


def stratified_bootstrap_upper_bound(
    differences_by_family: Mapping[str, Sequence[float]],
) -> dict[str, float | int | str]:
    """Return the frozen family-stratified one-sided 95% upper bound."""

    if tuple(differences_by_family) != STAGE_A_FAMILY_ORDER:
        raise ValueError("bootstrap families must use the exact frozen order")
    arrays: list[np.ndarray] = []
    for family in STAGE_A_FAMILY_ORDER:
        raw_values = differences_by_family[family]
        if len(raw_values) != STAGE_A_EXAMPLES_PER_FAMILY:
            raise ValueError("each Stage-A bootstrap family must contain exactly four examples")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (STAGE_A_EXAMPLES_PER_FAMILY,) or not np.isfinite(values).all():
            raise ValueError("Stage-A bootstrap differences must be finite scalar values")
        arrays.append(values)

    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled_family_means: list[np.ndarray] = []
    for values in arrays:
        indices = generator.integers(
            0,
            STAGE_A_EXAMPLES_PER_FAMILY,
            size=(BOOTSTRAP_SAMPLES, STAGE_A_EXAMPLES_PER_FAMILY),
        )
        sampled_family_means.append(values[indices].mean(axis=1, dtype=np.float64))
    sampled_macro = np.stack(sampled_family_means, axis=0).mean(axis=0, dtype=np.float64)
    family_point_means = [_mean(list(values)) for values in arrays]
    return {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "rng_algorithm": BOOTSTRAP_RNG_ALGORITHM,
        "quantile_probability": BOOTSTRAP_UPPER_QUANTILE,
        "quantile_algorithm": BOOTSTRAP_QUANTILE_ALGORITHM,
        "family_equal_point_estimate": _mean(family_point_means),
        "one_sided_95_upper_bound": _nearest_rank_quantile(sampled_macro, BOOTSTRAP_UPPER_QUANTILE),
    }


def _summary_lookup(
    method_summaries: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {str(summary["method_id"]): summary for summary in method_summaries}


def _family_lookup(summary: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = cast(Sequence[Mapping[str, object]], summary["by_family"])
    return {str(item["family"]): item for item in raw}


def _build_decision(
    per_example: Sequence[Mapping[str, object]],
    method_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    example_metric_lookup = {
        (str(item["family"]), str(item["canonical_id"]), str(item["method_id"])): cast(
            Mapping[str, float | int], item["metrics"]
        )
        for item in per_example
    }
    summaries = _summary_lookup(method_summaries)
    candidate_summary = summaries[STATIC_Q468_PRIMARY_METHOD]
    candidate_families = _family_lookup(candidate_summary)

    comparator_decisions: list[dict[str, object]] = []

    def family_macro_top1(method_id: str) -> Fraction:
        family_values: list[Fraction] = []
        for family in STAGE_A_FAMILY_ORDER:
            example_values: list[Fraction] = []
            for item in per_example:
                if item["family"] != family or item["method_id"] != method_id:
                    continue
                metrics = cast(Mapping[str, float | int], item["metrics"])
                example_values.append(
                    Fraction(
                        int(metrics["top1_agreement_count"]),
                        int(metrics["token_count"]),
                    )
                )
            family_values.append(sum(example_values, start=Fraction()) / len(example_values))
        return sum(family_values, start=Fraction()) / len(family_values)

    candidate_top1_fraction = family_macro_top1(STATIC_Q468_PRIMARY_METHOD)
    for comparator_method in DECISION_COMPARATOR_ORDER:
        differences_by_family: dict[str, list[float]] = {}
        family_points: list[dict[str, object]] = []
        for family in STAGE_A_FAMILY_ORDER:
            candidate_rows = [
                item
                for item in per_example
                if item["family"] == family and item["method_id"] == STATIC_Q468_PRIMARY_METHOD
            ]
            differences = [
                float(
                    example_metric_lookup[
                        (family, str(candidate_row["canonical_id"]), STATIC_Q468_PRIMARY_METHOD)
                    ]["excess_nll"]
                )
                - float(
                    example_metric_lookup[
                        (family, str(candidate_row["canonical_id"]), comparator_method)
                    ]["excess_nll"]
                )
                for candidate_row in candidate_rows
            ]
            differences_by_family[family] = differences
            family_points.append(
                {
                    "family": family,
                    "candidate_minus_comparator_excess_nll": _mean(differences),
                }
            )
        bootstrap = stratified_bootstrap_upper_bound(differences_by_family)
        candidate_top1 = float(candidate_top1_fraction)
        comparator_top1_fraction = family_macro_top1(comparator_method)
        comparator_top1 = float(comparator_top1_fraction)
        top1_trail_fraction = comparator_top1_fraction - candidate_top1_fraction
        top1_trail = float(top1_trail_fraction)
        checks = {
            "one_sided_95_upper_bound_at_most_0_010": (
                float(bootstrap["one_sided_95_upper_bound"]) <= MAX_ONE_SIDED_95_UPPER_EXCESS_NLL
            ),
            "every_family_point_at_most_0_015": all(
                float(item["candidate_minus_comparator_excess_nll"]) <= MAX_FAMILY_POINT_EXCESS_NLL
                for item in family_points
            ),
            "family_macro_top1_trail_at_most_0_005": (top1_trail_fraction <= Fraction(5, 1_000)),
        }
        comparator_decisions.append(
            {
                "comparator_method": comparator_method,
                "difference_direction": "candidate_excess_nll_minus_comparator_excess_nll",
                "family_points": family_points,
                "paired_family_stratified_bootstrap": bootstrap,
                "candidate_family_macro_top1_agreement": candidate_top1,
                "comparator_family_macro_top1_agreement": comparator_top1,
                "candidate_top1_trail": top1_trail,
                "candidate_top1_trail_exact_fraction": (
                    f"{top1_trail_fraction.numerator}/{top1_trail_fraction.denominator}"
                ),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    q48_families = _family_lookup(summaries[STATIC_Q48_COMPARATOR_METHOD])
    q48_points: list[dict[str, object]] = []
    for family in STAGE_A_FAMILY_ORDER:
        candidate_excess = float(
            cast(Mapping[str, float], candidate_families[family]["example_macro"])["excess_nll"]
        )
        q48_excess = float(
            cast(Mapping[str, float], q48_families[family]["example_macro"])["excess_nll"]
        )
        q48_points.append(
            {
                "family": family,
                "candidate_excess_nll": candidate_excess,
                "q48_excess_nll": q48_excess,
                "q48_minus_candidate_excess_nll": q48_excess - candidate_excess,
                "candidate_strictly_lower": candidate_excess < q48_excess,
            }
        )
    q48_passed = all(item["candidate_strictly_lower"] is True for item in q48_points)
    return {
        "candidate_method": STATIC_Q468_PRIMARY_METHOD,
        "noninferiority_comparators": comparator_decisions,
        "q48_every_family_strict_superiority": {
            "comparator_method": STATIC_Q48_COMPARATOR_METHOD,
            "family_points": q48_points,
            "equality_fails": True,
            "passed": q48_passed,
        },
        "excluded_from_decision": {
            STATIC_Q468_ABLATION_METHOD: "selection-step-matched diagnostic only",
            UNIFORM_RHT_Q4_METHOD: "descriptive anchor only",
            UNIFORM_RHT_Q8_METHOD: "descriptive anchor only",
            FP32_METHOD: "matched reference trajectory",
        },
        "stage_a_passed": all(item["passed"] is True for item in comparator_decisions)
        and q48_passed,
    }


def _example_payload(example: StageAExample) -> dict[str, object]:
    return {
        "family": example.family,
        "canonical_id": example.canonical_id,
        "selection_rank": example.selection_rank,
        "continuation_token_count": example.continuation_token_count,
        "identity_record_sha256": example.identity_record_sha256,
    }


def _row_payload(row: StageATokenRow) -> dict[str, object]:
    return {
        "family": row.family,
        "canonical_id": row.canonical_id,
        "selection_rank": row.selection_rank,
        "identity_record_sha256": row.identity_record_sha256,
        "method_id": row.method_id,
        "transition_index": row.transition_index,
        "reference_nll": row.reference_nll,
        "method_nll": row.method_nll,
        "kl": row.kl,
        "top1_agreement": row.top1_agreement,
    }


def build_stage_a_evidence_artifact(
    examples: Sequence[StageAExample],
    token_rows: Sequence[StageATokenRow],
    *,
    stage_a_identity_file_sha256: str,
    stage_a_calibration_binding_file_sha256: str,
) -> bytes:
    """Build the canonical Stage-A evidence artifact from the complete row grid."""

    identity_hash = _require_sha256(
        stage_a_identity_file_sha256,
        name="stage_a_identity_file_sha256",
    )
    binding_hash = _require_sha256(
        stage_a_calibration_binding_file_sha256,
        name="stage_a_calibration_binding_file_sha256",
    )
    normalized_examples = _normalized_examples(examples)
    normalized_rows = _normalized_rows(normalized_examples, token_rows)
    per_example, method_summaries = _build_summaries(normalized_examples, normalized_rows)
    evidence: dict[str, object] = {
        "artifact_kind": STAGE_A_ARTIFACT_KIND,
        "artifact_revision": STAGE_A_ARTIFACT_REVISION,
        "schema_version": STAGE_A_SCHEMA_VERSION,
        "profile": STAGE_A_PROFILE,
        "claim_boundary": CLAIM_BOUNDARY,
        "dependencies": {
            "stage_a_identity_file_sha256": identity_hash,
            "stage_a_calibration_binding_file_sha256": binding_hash,
        },
        "metric_contract": {
            "family_order": list(STAGE_A_FAMILY_ORDER),
            "examples_per_family": STAGE_A_EXAMPLES_PER_FAMILY,
            "method_order": list(STAGE_A_METHOD_ORDER),
            "cache_exposed_transitions": "m-1 for an identity-bound continuation of m>=2 tokens",
            "excess_nll": "mean(method_nll-reference_nll) over cache-exposed transitions",
            "example_macro": "equal-weight arithmetic mean of examples within one family",
            "family_macro": (
                "equal-weight arithmetic mean of PG19, RULER, and HumanEval+ family means"
            ),
            "token_micro": "all cache-exposed transitions equally weighted; diagnostic only",
            "cvar95_kl": (
                "FP32 mean of largest ceil(0.05*N) finite token KL values, "
                "matching fidelity_summary"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_rng_algorithm": BOOTSTRAP_RNG_ALGORITHM,
            "bootstrap_quantile_algorithm": BOOTSTRAP_QUANTILE_ALGORITHM,
        },
        "method_roles": {
            "fixed_order": list(STAGE_A_METHOD_ORDER),
            "candidate": STATIC_Q468_PRIMARY_METHOD,
            "decision_comparators": list(DECISION_COMPARATOR_ORDER),
            "strict_family_superiority_comparator": STATIC_Q48_COMPARATOR_METHOD,
            "diagnostic_only": [STATIC_Q468_ABLATION_METHOD],
            "descriptive_only": [UNIFORM_RHT_Q4_METHOD, UNIFORM_RHT_Q8_METHOD],
            "reference": FP32_METHOD,
        },
        "examples": [_example_payload(example) for example in normalized_examples],
        "token_rows": [_row_payload(row) for row in normalized_rows],
        "example_summaries": per_example,
        "method_summaries": method_summaries,
        "decision": _build_decision(per_example, method_summaries),
    }
    canonical_hash = _sha256_bytes(canonical_json_bytes(evidence))
    return canonical_json_bytes(
        {
            "artifact_kind": STAGE_A_ARTIFACT_KIND,
            "schema_version": STAGE_A_SCHEMA_VERSION,
            "evidence": evidence,
            "canonical_evidence_sha256": canonical_hash,
        }
    )


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Stage-A artifact contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage-A artifact must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Stage-A artifact root must be a JSON object")
    return value


def _example_from_payload(value: object, *, index: int) -> StageAExample:
    if not isinstance(value, Mapping):
        raise ValueError(f"examples[{index}] must be an object")
    _require_exact_fields(value, _EXAMPLE_FIELDS, name=f"examples[{index}]")
    return StageAExample(
        family=cast(StageAFamily, value["family"]),
        canonical_id=cast(str, value["canonical_id"]),
        selection_rank=cast(int, value["selection_rank"]),
        continuation_token_count=cast(int, value["continuation_token_count"]),
        identity_record_sha256=cast(str, value["identity_record_sha256"]),
    )


def _row_from_payload(value: object, *, index: int) -> StageATokenRow:
    if not isinstance(value, Mapping):
        raise ValueError(f"token_rows[{index}] must be an object")
    _require_exact_fields(value, _TOKEN_ROW_FIELDS, name=f"token_rows[{index}]")
    return StageATokenRow(
        family=cast(StageAFamily, value["family"]),
        canonical_id=cast(str, value["canonical_id"]),
        selection_rank=cast(int, value["selection_rank"]),
        identity_record_sha256=cast(str, value["identity_record_sha256"]),
        method_id=cast(str, value["method_id"]),
        transition_index=cast(int, value["transition_index"]),
        reference_nll=cast(float, value["reference_nll"]),
        method_nll=cast(float, value["method_nll"]),
        kl=cast(float, value["kl"]),
        top1_agreement=cast(bool, value["top1_agreement"]),
    )


def deserialize_stage_a_evidence_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
    expected_canonical_evidence_sha256: str | None = None,
    expected_stage_a_identity_file_sha256: str | None = None,
    expected_stage_a_calibration_binding_file_sha256: str | None = None,
) -> StageAEvidenceArtifact:
    """Strictly verify by reconstructing every summary, bootstrap, and gate."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A artifact data must be bytes")
    file_hash = _sha256_bytes(data)
    if expected_file_sha256 is not None and file_hash != _require_sha256(
        expected_file_sha256, name="expected_file_sha256"
    ):
        raise ValueError("Stage-A artifact file SHA-256 differs from the expected digest")
    root = _strict_json(data)
    _require_exact_fields(root, _OUTER_FIELDS, name="Stage-A artifact")
    if (
        root["artifact_kind"] != STAGE_A_ARTIFACT_KIND
        or root["schema_version"] != STAGE_A_SCHEMA_VERSION
    ):
        raise ValueError("Stage-A artifact kind or schema version drifted")
    evidence = root["evidence"]
    if not isinstance(evidence, Mapping):
        raise ValueError("Stage-A evidence must be an object")
    _require_exact_fields(evidence, _EVIDENCE_FIELDS, name="Stage-A evidence")
    if (
        evidence["artifact_kind"] != STAGE_A_ARTIFACT_KIND
        or evidence["schema_version"] != STAGE_A_SCHEMA_VERSION
        or evidence["artifact_revision"] != STAGE_A_ARTIFACT_REVISION
        or evidence["profile"] != STAGE_A_PROFILE
        or evidence["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise ValueError("Stage-A evidence identity drifted")
    recorded_canonical = _require_sha256(
        root["canonical_evidence_sha256"], name="canonical_evidence_sha256"
    )
    computed_canonical = _sha256_bytes(canonical_json_bytes(dict(evidence)))
    if recorded_canonical != computed_canonical:
        raise ValueError("Stage-A canonical evidence SHA-256 differs from its contents")
    if expected_canonical_evidence_sha256 is not None and computed_canonical != _require_sha256(
        expected_canonical_evidence_sha256,
        name="expected_canonical_evidence_sha256",
    ):
        raise ValueError("Stage-A canonical evidence SHA-256 differs from the expected digest")

    dependencies = evidence["dependencies"]
    if not isinstance(dependencies, Mapping):
        raise ValueError("Stage-A dependencies must be an object")
    _require_exact_fields(dependencies, _DEPENDENCY_FIELDS, name="Stage-A dependencies")
    identity_hash = _require_sha256(
        dependencies["stage_a_identity_file_sha256"],
        name="stage_a_identity_file_sha256",
    )
    binding_hash = _require_sha256(
        dependencies["stage_a_calibration_binding_file_sha256"],
        name="stage_a_calibration_binding_file_sha256",
    )
    if expected_stage_a_identity_file_sha256 is not None and identity_hash != _require_sha256(
        expected_stage_a_identity_file_sha256,
        name="expected_stage_a_identity_file_sha256",
    ):
        raise ValueError("Stage-A identity dependency differs from the expected digest")
    if (
        expected_stage_a_calibration_binding_file_sha256 is not None
        and binding_hash
        != _require_sha256(
            expected_stage_a_calibration_binding_file_sha256,
            name="expected_stage_a_calibration_binding_file_sha256",
        )
    ):
        raise ValueError("Stage-A calibration binding differs from the expected digest")

    raw_examples = evidence["examples"]
    raw_rows = evidence["token_rows"]
    if not isinstance(raw_examples, list) or not isinstance(raw_rows, list):
        raise ValueError("Stage-A examples and token_rows must be JSON arrays")
    examples = tuple(
        _example_from_payload(value, index=index) for index, value in enumerate(raw_examples)
    )
    rows = tuple(_row_from_payload(value, index=index) for index, value in enumerate(raw_rows))
    rebuilt = build_stage_a_evidence_artifact(
        examples,
        rows,
        stage_a_identity_file_sha256=identity_hash,
        stage_a_calibration_binding_file_sha256=binding_hash,
    )
    if rebuilt != data:
        raise ValueError("Stage-A artifact differs from its deterministic reconstruction")
    decision = evidence["decision"]
    if not isinstance(decision, Mapping) or not isinstance(decision.get("stage_a_passed"), bool):
        raise ValueError("Stage-A decision must contain a boolean stage_a_passed result")
    return StageAEvidenceArtifact(
        evidence=cast(Mapping[str, Any], _deep_freeze_json(dict(evidence))),
        canonical_evidence_sha256=computed_canonical,
        file_sha256=file_hash,
        serialized_bytes=bytes(data),
        _verified_passed=cast(bool, decision["stage_a_passed"]),
    )


def verify_stage_a_evidence_artifact(
    data: bytes,
    **expected: str | None,
) -> StageAEvidenceArtifact:
    """Alias for the strict reconstructing deserializer."""

    return deserialize_stage_a_evidence_artifact(data, **expected)


__all__ = [
    "BOOTSTRAP_QUANTILE_ALGORITHM",
    "BOOTSTRAP_RNG_ALGORITHM",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_UPPER_QUANTILE",
    "CLAIM_BOUNDARY",
    "DECISION_COMPARATOR_ORDER",
    "FP32_METHOD",
    "MAX_FAMILY_MACRO_TOP1_TRAIL",
    "MAX_FAMILY_POINT_EXCESS_NLL",
    "MAX_ONE_SIDED_95_UPPER_EXCESS_NLL",
    "STAGE_A_ARTIFACT_KIND",
    "STAGE_A_ARTIFACT_REVISION",
    "STAGE_A_EXAMPLE_COUNT",
    "STAGE_A_EXAMPLES_PER_FAMILY",
    "STAGE_A_FAMILY_ORDER",
    "STAGE_A_METHOD_ORDER",
    "STAGE_A_PROFILE",
    "STAGE_A_SCHEMA_VERSION",
    "TAIL_KL_FRACTION",
    "UNIFORM_RHT_Q4_METHOD",
    "UNIFORM_RHT_Q8_METHOD",
    "StageAEvidenceArtifact",
    "StageAExample",
    "StageAFamily",
    "StageATokenRow",
    "build_stage_a_evidence_artifact",
    "deserialize_stage_a_evidence_artifact",
    "stratified_bootstrap_upper_bound",
    "verify_stage_a_evidence_artifact",
]
