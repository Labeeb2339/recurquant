from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import pytest

from recurquant.evidence import canonical_json_bytes
from recurquant.experiment013_stage_a import (
    BOOTSTRAP_QUANTILE_ALGORITHM,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DECISION_COMPARATOR_ORDER,
    FP32_METHOD,
    STAGE_A_FAMILY_ORDER,
    STAGE_A_METHOD_ORDER,
    UNIFORM_RHT_Q4_METHOD,
    UNIFORM_RHT_Q8_METHOD,
    StageAExample,
    StageATokenRow,
    build_stage_a_evidence_artifact,
    deserialize_stage_a_evidence_artifact,
    stratified_bootstrap_upper_bound,
)
from recurquant.static_q468 import (
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
)
from recurquant.static_q468_cache import DYNAMIC_Q468_BASELINE_METHOD

IDENTITY_FILE_SHA256 = "a" * 64
BINDING_FILE_SHA256 = "b" * 64

Excess = Callable[[StageAExample, str, int], float]
Agreement = Callable[[StageAExample, str, int], bool]


def _examples(*, transitions: int | Callable[[str, int], int] = 1) -> tuple[StageAExample, ...]:
    result: list[StageAExample] = []
    for family in STAGE_A_FAMILY_ORDER:
        for rank in range(4):
            count = transitions(family, rank) if callable(transitions) else transitions
            identity_hash = hashlib.sha256(f"{family}:{rank}".encode()).hexdigest()
            result.append(
                StageAExample(
                    family=family,
                    canonical_id=f"{family}/{rank}",
                    selection_rank=rank,
                    continuation_token_count=count + 1,
                    identity_record_sha256=identity_hash,
                )
            )
    return tuple(result)


def _default_excess(example: StageAExample, method: str, transition: int) -> float:
    del example, transition
    return {
        FP32_METHOD: 0.0,
        UNIFORM_RHT_Q4_METHOD: 0.20,
        UNIFORM_RHT_Q8_METHOD: 0.10,
        STATIC_Q48_COMPARATOR_METHOD: 0.08,
        STATIC_Q468_ABLATION_METHOD: 100.0,
        DYNAMIC_Q468_BASELINE_METHOD: 0.015,
        STATIC_Q468_MSE_METHOD: 0.014,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD: 0.013,
        STATIC_Q468_PRIMARY_METHOD: 0.020,
    }[method]


def _rows(
    examples: tuple[StageAExample, ...],
    *,
    excess: Excess = _default_excess,
    agreement: Agreement | None = None,
) -> tuple[StageATokenRow, ...]:
    result: list[StageATokenRow] = []
    for example in examples:
        for method in STAGE_A_METHOD_ORDER:
            for transition in range(example.transition_count):
                delta = excess(example, method, transition)
                is_fp32 = method == FP32_METHOD
                result.append(
                    StageATokenRow(
                        family=example.family,
                        canonical_id=example.canonical_id,
                        selection_rank=example.selection_rank,
                        identity_record_sha256=example.identity_record_sha256,
                        method_id=method,
                        transition_index=transition,
                        reference_nll=0.0,
                        method_nll=delta,
                        kl=0.0 if is_fp32 else max(delta, 0.0) + 0.001,
                        top1_agreement=(
                            True
                            if is_fp32 or agreement is None
                            else agreement(example, method, transition)
                        ),
                    )
                )
    return tuple(result)


def _build(
    examples: tuple[StageAExample, ...],
    rows: tuple[StageATokenRow, ...],
) -> bytes:
    return build_stage_a_evidence_artifact(
        examples,
        rows,
        stage_a_identity_file_sha256=IDENTITY_FILE_SHA256,
        stage_a_calibration_binding_file_sha256=BINDING_FILE_SHA256,
    )


def _decision(raw: bytes) -> dict[str, object]:
    artifact = deserialize_stage_a_evidence_artifact(raw)
    return dict(artifact.evidence["decision"])


def _comparator(decision: dict[str, object], method: str) -> dict[str, object]:
    comparisons = decision["noninferiority_comparators"]
    assert isinstance(comparisons, Sequence)
    return dict(next(item for item in comparisons if item["comparator_method"] == method))


def test_frozen_method_order_and_m_equals_two_yields_one_transition() -> None:
    assert STAGE_A_METHOD_ORDER == (
        "fp32_reference",
        "rht_q468_uniform_q4",
        "rht_q468_uniform_q8",
        "rht_q48_static_p14739",
        "rht_q468_static_k27030",
        "rht_q468_dynamic_k27030",
        "rht_q468_static_mse_k29334",
        "rht_q468_static_diag_empirical_fisher_h1_k29334",
        "rht_q468_static_k29334",
    )
    examples = _examples(transitions=1)
    raw = _build(examples, _rows(examples))
    artifact = deserialize_stage_a_evidence_artifact(
        raw,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_canonical_evidence_sha256=json.loads(raw)["canonical_evidence_sha256"],
        expected_stage_a_identity_file_sha256=IDENTITY_FILE_SHA256,
        expected_stage_a_calibration_binding_file_sha256=BINDING_FILE_SHA256,
    )

    assert artifact.passed is True
    assert len(artifact.evidence["token_rows"]) == 12 * 9
    assert len(artifact.evidence["example_summaries"]) == 12 * 9
    summaries = artifact.evidence["method_summaries"]
    assert [summary["method_id"] for summary in summaries] == list(STAGE_A_METHOD_ORDER)
    assert all(
        family["token_count"] == 4 for summary in summaries for family in summary["by_family"]
    )


def test_per_example_excess_top1_and_existing_cvar95_definition_are_published() -> None:
    examples = _examples(transitions=21)
    rows = list(_rows(examples))
    first = examples[0]
    for index, row in enumerate(rows):
        if row.canonical_id == first.canonical_id and row.method_id == STATIC_Q468_PRIMARY_METHOD:
            rows[index] = replace(row, kl=float(row.transition_index))
    artifact = deserialize_stage_a_evidence_artifact(_build(examples, tuple(rows)))
    summary = next(
        item
        for item in artifact.evidence["example_summaries"]
        if item["canonical_id"] == first.canonical_id
        and item["method_id"] == STATIC_Q468_PRIMARY_METHOD
    )

    assert summary["metrics"]["excess_nll"] == pytest.approx(0.020)
    assert summary["metrics"]["top1_agreement"] == 1.0
    # ceil(0.05 * 21) = 2, so CVaR95 is the mean of token KL 20 and 19.
    assert summary["metrics"]["cvar95_kl"] == pytest.approx(19.5)


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "reordered"])
def test_token_grid_fails_closed_on_missing_duplicate_or_reordered_rows(
    malformation: str,
) -> None:
    examples = _examples()
    rows = list(_rows(examples))
    if malformation == "missing":
        rows.pop()
    elif malformation == "duplicate":
        rows[-1] = rows[-2]
    else:
        rows[1], rows[2] = rows[2], rows[1]

    with pytest.raises(ValueError, match="row count|missing, duplicate, or reordered"):
        _build(examples, tuple(rows))


def test_example_grid_rejects_duplicate_and_reordered_identities() -> None:
    examples = list(_examples())
    examples[0], examples[1] = examples[1], examples[0]
    with pytest.raises(ValueError, match="ordered by frozen family order"):
        _build(tuple(examples), _rows(_examples()))

    examples = list(_examples())
    examples[1] = StageAExample(
        family=examples[1].family,
        canonical_id=examples[0].canonical_id,
        selection_rank=examples[1].selection_rank,
        continuation_token_count=examples[1].continuation_token_count,
        identity_record_sha256=examples[1].identity_record_sha256,
    )
    with pytest.raises(ValueError, match="duplicate Stage-A example"):
        _build(tuple(examples), _rows(_examples()))


def test_reconstructing_verifier_rejects_tampered_rows_and_derived_metrics() -> None:
    examples = _examples()
    raw = _build(examples, _rows(examples))
    document = json.loads(raw)
    document["evidence"]["token_rows"][-1]["method_nll"] += 0.25
    document["canonical_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(document["evidence"])
    ).hexdigest()
    tampered = canonical_json_bytes(document)

    with pytest.raises(ValueError, match="deterministic reconstruction"):
        deserialize_stage_a_evidence_artifact(tampered)

    changed_rows = list(_rows(examples))
    changed_rows[-1] = replace(
        changed_rows[-1],
        method_nll=changed_rows[-1].method_nll + 0.25,
    )
    changed = _build(examples, tuple(changed_rows))
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        deserialize_stage_a_evidence_artifact(
            changed,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_duplicate_json_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        deserialize_stage_a_evidence_artifact(b'{"artifact_kind":"x","artifact_kind":"y"}')


def test_verified_evidence_tree_is_immutable_and_passed_is_stable() -> None:
    examples = _examples()
    raw = _build(examples, _rows(examples))
    artifact = deserialize_stage_a_evidence_artifact(raw)
    decision = artifact.evidence["decision"]
    assert isinstance(decision, Mapping)
    comparisons = decision["noninferiority_comparators"]
    assert isinstance(comparisons, tuple)

    with pytest.raises(TypeError):
        decision["stage_a_passed"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        comparisons[0]["passed"] = False  # type: ignore[index]

    assert artifact.passed is True
    assert artifact.serialized_bytes == raw
    assert artifact.file_sha256 == hashlib.sha256(raw).hexdigest()


def test_negative_kl_is_rejected() -> None:
    examples = _examples()
    rows = list(_rows(examples))
    rows[-1] = replace(rows[-1], kl=-1e-6)

    with pytest.raises(ValueError, match="kl must be nonnegative"):
        _build(examples, tuple(rows))


def test_cvar95_uses_reviewed_fp32_tail_mean_semantics_exactly() -> None:
    examples = _examples(transitions=21)
    rows = list(_rows(examples))
    first = examples[0]
    values = [16_777_216.0, 1.0, *([0.0] * 19)]
    for index, row in enumerate(rows):
        if row.canonical_id == first.canonical_id and row.method_id == UNIFORM_RHT_Q4_METHOD:
            rows[index] = replace(row, kl=values[row.transition_index])

    artifact = deserialize_stage_a_evidence_artifact(_build(examples, tuple(rows)))
    summary = next(
        item
        for item in artifact.evidence["example_summaries"]
        if item["canonical_id"] == first.canonical_id and item["method_id"] == UNIFORM_RHT_Q4_METHOD
    )

    # FP64 arithmetic would produce 8,388,608.5. Torch's reviewed FP32
    # fidelity path rounds that midpoint to 8,388,608.0.
    assert summary["metrics"]["cvar95_kl"] == 8_388_608.0


def test_cvar95_rejects_values_that_overflow_during_fp32_conversion() -> None:
    examples = _examples()
    rows = list(_rows(examples))
    rows[-1] = replace(rows[-1], kl=1e100)

    with pytest.raises(ValueError, match="remain finite after FP32 conversion"):
        _build(examples, tuple(rows))


def test_bootstrap_is_deterministic_family_equal_and_nearest_rank() -> None:
    differences = {
        "pg19": [1.0, 1.0, 1.0, 1.0],
        "ruler": [2.0, 2.0, 2.0, 2.0],
        "humaneval_plus": [6.0, 6.0, 6.0, 6.0],
    }
    first = stratified_bootstrap_upper_bound(differences)
    second = stratified_bootstrap_upper_bound(differences)

    assert first == second
    assert first["bootstrap_samples"] == BOOTSTRAP_SAMPLES == 10_000
    assert first["seed"] == BOOTSTRAP_SEED == 2_339
    assert first["family_equal_point_estimate"] == pytest.approx(3.0)
    assert first["one_sided_95_upper_bound"] == pytest.approx(3.0)
    assert first["quantile_algorithm"] == BOOTSTRAP_QUANTILE_ALGORITHM
    assert "ceil(q*N)-1" in BOOTSTRAP_QUANTILE_ALGORITHM


def test_token_micro_is_diagnostic_and_family_macro_weights_examples_equally() -> None:
    examples = _examples(
        transitions=lambda family, rank: 10 if (family, rank) == ("pg19", 0) else 1
    )

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del transition
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return 0.08 if (example.family, example.selection_rank) == ("pg19", 0) else 0.0
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return 0.20
        if method in DECISION_COMPARATOR_ORDER:
            return 0.10
        return 0.30

    artifact = deserialize_stage_a_evidence_artifact(
        _build(examples, _rows(examples, excess=excess))
    )
    summaries = {summary["method_id"]: summary for summary in artifact.evidence["method_summaries"]}
    primary = summaries[STATIC_Q468_PRIMARY_METHOD]

    # One of four PG19 examples is 0.08, so PG19 macro is 0.02.  The other
    # family means are zero, making the family-equal macro 0.02/3.
    assert primary["family_macro"]["excess_nll"] == pytest.approx(0.02 / 3.0)
    assert primary["token_micro_diagnostic"]["excess_nll"] > 0.02


def test_bootstrap_upper_conjunct_is_required() -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del transition
        candidate = 0.020
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return candidate
        if method == DYNAMIC_Q468_BASELINE_METHOD:
            difference = 0.015 if example.selection_rank < 3 else -0.015
            return candidate - difference
        if method in (
            STATIC_Q468_MSE_METHOD,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        ):
            return candidate + 0.020
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return candidate + 0.050
        return 0.10

    decision = _decision(_build(examples, _rows(examples, excess=excess)))
    comparison = _comparator(decision, DYNAMIC_Q468_BASELINE_METHOD)
    assert comparison["checks"] == {
        "one_sided_95_upper_bound_at_most_0_010": False,
        "every_family_point_at_most_0_015": True,
        "family_macro_top1_trail_at_most_0_005": True,
    }
    assert decision["stage_a_passed"] is False


def test_every_family_point_conjunct_is_required() -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del transition
        candidate = 0.020
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return candidate
        if method == DYNAMIC_Q468_BASELINE_METHOD:
            difference = 0.0151 if example.family == "pg19" else -0.020
            return candidate - difference
        if method in (
            STATIC_Q468_MSE_METHOD,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        ):
            return candidate + 0.020
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return candidate + 0.050
        return 0.10

    decision = _decision(_build(examples, _rows(examples, excess=excess)))
    comparison = _comparator(decision, DYNAMIC_Q468_BASELINE_METHOD)
    assert comparison["checks"] == {
        "one_sided_95_upper_bound_at_most_0_010": True,
        "every_family_point_at_most_0_015": False,
        "family_macro_top1_trail_at_most_0_005": True,
    }
    assert decision["stage_a_passed"] is False


def test_family_macro_top1_conjunct_is_required() -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del example, transition
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return 0.020
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return 0.080
        if method in DECISION_COMPARATOR_ORDER:
            return 0.040
        return 0.10

    def agreement(example: StageAExample, method: str, transition: int) -> bool:
        del example, transition
        if method == DYNAMIC_Q468_BASELINE_METHOD:
            return True
        return method not in (
            STATIC_Q468_PRIMARY_METHOD,
            STATIC_Q468_MSE_METHOD,
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        )

    decision = _decision(_build(examples, _rows(examples, excess=excess, agreement=agreement)))
    comparison = _comparator(decision, DYNAMIC_Q468_BASELINE_METHOD)
    assert comparison["checks"] == {
        "one_sided_95_upper_bound_at_most_0_010": True,
        "every_family_point_at_most_0_015": True,
        "family_macro_top1_trail_at_most_0_005": False,
    }
    assert decision["stage_a_passed"] is False


@pytest.mark.parametrize(
    "comparator",
    [STATIC_Q468_MSE_METHOD, STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD],
)
def test_each_frozen_static_selector_comparator_can_fail_the_screen(comparator: str) -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del example, transition
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return 0.020
        if method == comparator:
            return 0.0
        if method in DECISION_COMPARATOR_ORDER:
            return 0.040
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return 0.080
        return 0.10

    decision = _decision(_build(examples, _rows(examples, excess=excess)))
    assert _comparator(decision, comparator)["passed"] is False
    assert decision["stage_a_passed"] is False


def test_all_nonlinearity_boundaries_pass_at_equality() -> None:
    examples = _examples(transitions=50)
    difference_by_family = {"pg19": 0.015, "ruler": 0.010, "humaneval_plus": 0.005}

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del transition
        candidate = 0.020
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return candidate
        if method in DECISION_COMPARATOR_ORDER:
            return candidate - difference_by_family[example.family]
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return candidate + 0.050
        return 0.10

    def agreement(example: StageAExample, method: str, transition: int) -> bool:
        if method == STATIC_Q468_PRIMARY_METHOD:
            # One miss in one of four 50-transition examples per family gives
            # an exact family-macro trail of 1/(4*50) = 0.005.
            return example.selection_rank != 0 or transition != 0
        return True

    decision = _decision(_build(examples, _rows(examples, excess=excess, agreement=agreement)))
    for comparator in DECISION_COMPARATOR_ORDER:
        comparison = _comparator(decision, comparator)
        assert comparison["paired_family_stratified_bootstrap"][
            "one_sided_95_upper_bound"
        ] == pytest.approx(0.010)
        assert max(
            point["candidate_minus_comparator_excess_nll"] for point in comparison["family_points"]
        ) == pytest.approx(0.015)
        assert comparison["candidate_top1_trail"] == pytest.approx(0.005)
        assert comparison["checks"] == {
            "one_sided_95_upper_bound_at_most_0_010": True,
            "every_family_point_at_most_0_015": True,
            "family_macro_top1_trail_at_most_0_005": True,
        }
    assert decision["stage_a_passed"] is True


def test_q48_requires_strictly_lower_excess_nll_in_every_family() -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        del transition
        candidate = 0.020
        if method == FP32_METHOD:
            return 0.0
        if method == STATIC_Q468_PRIMARY_METHOD:
            return candidate
        if method == STATIC_Q48_COMPARATOR_METHOD:
            return candidate if example.family == "ruler" else candidate + 0.010
        if method in DECISION_COMPARATOR_ORDER:
            return candidate + 0.020
        return 0.10

    decision = _decision(_build(examples, _rows(examples, excess=excess)))
    q48 = decision["q48_every_family_strict_superiority"]
    assert q48["passed"] is False
    ruler = next(point for point in q48["family_points"] if point["family"] == "ruler")
    assert ruler["q48_minus_candidate_excess_nll"] == 0.0
    assert ruler["candidate_strictly_lower"] is False
    assert decision["stage_a_passed"] is False


def test_static_k27030_is_diagnostic_and_cannot_decide_passage() -> None:
    examples = _examples()

    def excess(example: StageAExample, method: str, transition: int) -> float:
        value = _default_excess(example, method, transition)
        return 10_000.0 if method == STATIC_Q468_ABLATION_METHOD else value

    decision = _decision(_build(examples, _rows(examples, excess=excess)))

    assert decision["stage_a_passed"] is True
    assert decision["excluded_from_decision"][STATIC_Q468_ABLATION_METHOD] == (
        "selection-step-matched diagnostic only"
    )
