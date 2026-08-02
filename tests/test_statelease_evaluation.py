from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from recurquant.statelease_evaluation import (
    EQUAL_BYTE_NO_REPLAY_METHODS,
    FIXED_REPLAY_METHODS,
    FROZEN_STAGE_A_ALIGNED_TOKENS,
    FROZEN_STAGE_A_FORWARD_COUNT,
    FROZEN_STAGE_A_PROMPT_TOKENS,
    FROZEN_STAGE_A_RECURRENT_LAYER_INDICES,
    FROZEN_STAGE_A_TOKENS_OBSERVED,
    FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES,
    FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS,
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    TrajectoryNmseAccumulator,
    evaluate_statelease_stage_a_gate,
    reference_aligned_trajectory_nmse,
)


def _metrics(
    *,
    statelease_delta: float = 0.80,
    cc1_delta: float = 1.00,
    strongest_delta: float = 0.79,
    statelease_top1: float = 0.90,
    best_top1: float = 0.91,
) -> dict[str, dict[str, float | int | bool]]:
    reference_nll = 0.5
    methods = (
        RHT_CQER_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    )
    deltas = {method: 1.20 for method in methods}
    deltas[STATELEASE_METHOD] = statelease_delta
    deltas["fixed_cc1"] = cc1_delta
    deltas["fixed_cc2"] = strongest_delta
    top1 = {method: 0.88 for method in methods}
    top1[STATELEASE_METHOD] = statelease_top1
    top1["fixed_cc4"] = best_top1
    return {
        method: {
            "reference_nll": reference_nll,
            "candidate_nll": reference_nll + deltas[method],
            "delta_nll": deltas[method],
            "top1_agreement": top1[method],
            "token_count": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "all_logits_finite": True,
        }
        for method in methods
    }


def _trajectory(
    *,
    statelease: float = 0.8,
    cc1: float = 1.0,
) -> dict[str, dict[str, float | int]]:
    methods = (
        RHT_CQER_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    )
    result = {
        method: {
            "trajectory_nmse_auc": 1.2,
            "scored_write_count": FROZEN_STAGE_A_ALIGNED_TOKENS,
            "layer_value_count": FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES,
        }
        for method in methods
    }
    result[STATELEASE_METHOD]["trajectory_nmse_auc"] = statelease
    result["fixed_cc1"]["trajectory_nmse_auc"] = cc1
    return result


def _diagnostics() -> list[dict[str, int]]:
    return [
        {
            "layer_index": layer_index,
            "state_updates": FROZEN_STAGE_A_FORWARD_COUNT,
            "tokens_observed": FROZEN_STAGE_A_TOKENS_OBSERVED,
            "boundary4_count": 2,
            "boundary5_count": 6,
            "tie_count": 1,
            "invalid_boundary_count": 0,
        }
        for layer_index in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
    ]


def _update_evidence() -> list[dict[str, int | bool | None]]:
    result: list[dict[str, int | bool | None]] = []
    boundary_by_decode_write = {
        5: 4,
        9: 5,
        14: 4,
        18: 5,
        23: 5,
        28: 5,
        33: 5,
        38: 5,
    }
    for forward_index in range(FROZEN_STAGE_A_FORWARD_COUNT):
        boundary = boundary_by_decode_write.get(forward_index)
        for layer_index in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES:
            result.append(
                {
                    "update_index": len(result),
                    "layer_index": layer_index,
                    "state_index": 0,
                    "token_count": (FROZEN_STAGE_A_PROMPT_TOKENS if forward_index == 0 else 1),
                    "boundary": boundary,
                    "tie": forward_index == 9,
                }
            )
    assert len(result) == FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS
    return result


def _gate(
    *,
    metrics: Mapping[str, Mapping[str, object]] | None = None,
    trajectory: Mapping[str, object] | None = None,
    diagnostics: list[dict[str, int]] | None = None,
    update_evidence: list[dict[str, int | bool | None]] | None = None,
    storage: Mapping[str, object] | None = None,
    storage_bytes: int = 3_454_664,
    stage0_complete: bool = True,
    artifact_integrity: bool = True,
) -> dict[str, object]:
    return evaluate_statelease_stage_a_gate(
        aligned_metrics=_metrics() if metrics is None else metrics,
        trajectory_nmse_auc=_trajectory() if trajectory is None else trajectory,
        statelease_storage=(
            {
                "resident_bytes_including_statelease": storage_bytes,
                "persistent_fp32_state_mirror": False,
            }
            if storage is None
            else storage
        ),
        statelease_diagnostics=_diagnostics() if diagnostics is None else diagnostics,
        statelease_update_evidence=(
            _update_evidence() if update_evidence is None else update_evidence
        ),
        stage0_complete=stage0_complete,
        artifact_integrity=artifact_integrity,
    )


def test_reference_aligned_trajectory_nmse_uses_fp64_and_matched_layers() -> None:
    reference = {
        2: torch.tensor([[[[1.0, 2.0]]]], dtype=torch.bfloat16),
        5: torch.tensor([[[[0.0, 4.0]]]], dtype=torch.float32),
    }
    candidate = {
        2: torch.tensor([[[[2.0, 2.0]]]], dtype=torch.bfloat16),
        5: torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32),
    }

    actual = reference_aligned_trajectory_nmse(reference, candidate)

    assert actual[2] == pytest.approx(1.0 / (5.0 + 1e-12))
    assert actual[5] == pytest.approx(4.0 / (16.0 + 1e-12))


@pytest.mark.parametrize(
    ("reference", "candidate", "match"),
    [
        ({0: torch.ones(2)}, {1: torch.ones(2)}, "do not match"),
        ({0: torch.ones(2)}, {0: torch.ones(3)}, "shape mismatch"),
        ({0: torch.tensor([float("nan")])}, {0: torch.ones(1)}, "must be finite"),
    ],
)
def test_reference_aligned_trajectory_nmse_fails_closed(
    reference: dict[int, torch.Tensor],
    candidate: dict[int, torch.Tensor],
    match: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        reference_aligned_trajectory_nmse(reference, candidate)


def test_trajectory_accumulator_is_layer_and_write_macro() -> None:
    accumulator = TrajectoryNmseAccumulator()
    accumulator.append({0: 1.0, 2: 3.0})
    accumulator.append({0: 5.0, 2: 7.0})

    assert accumulator.summary() == {
        "trajectory_nmse_auc": 4.0,
        "scored_write_count": 2,
        "layer_value_count": 4,
    }


def test_stage_a_gate_passes_only_when_every_frozen_condition_passes() -> None:
    gate = _gate()

    assert gate["passed"] is True
    assert all(
        check["passed"] is True
        for check in gate["checks"].values()  # type: ignore[union-attr]
    )
    strongest = gate["checks"][  # type: ignore[index]
        "no_more_than_5_percent_worse_than_strongest_fixed"
    ]["evidence"]
    assert strongest["strongest_fixed_method"] == "fixed_cc2"


@pytest.mark.parametrize(
    "storage",
    [
        {"resident_bytes_including_statelease": 3_454_664},
        {
            "resident_bytes_including_statelease": 3_454_664,
            "persistent_fp32_state_mirror": None,
        },
        {
            "resident_bytes_including_statelease": 3_454_664,
            "persistent_fp32_state_mirror": True,
        },
        {
            "resident_bytes_including_statelease": 3_454_664,
            "persistent_fp32_state_mirror": 0,
        },
    ],
    ids=["missing", "null", "true", "integer-zero"],
)
def test_stage_a_gate_requires_explicit_boolean_false_for_no_fp32_mirror(
    storage: Mapping[str, object],
) -> None:
    gate = _gate(storage=storage)

    check = gate["checks"]["exact_statelease_allocation"]  # type: ignore[index]
    assert gate["passed"] is False
    assert check["passed"] is False
    assert "must be explicitly false" in check["error"]


@pytest.mark.parametrize(
    ("name", "kwargs", "check"),
    [
        (
            "stage0",
            {"stage0_complete": False},
            "stage0_and_artifact_integrity",
        ),
        (
            "integrity",
            {"artifact_integrity": False},
            "stage0_and_artifact_integrity",
        ),
        (
            "bytes",
            {"storage_bytes": 3_454_663},
            "exact_statelease_allocation",
        ),
        (
            "cc1 improvement",
            {"metrics": _metrics(statelease_delta=0.91)},
            "cc1_excess_nll_reduction_at_least_10_percent",
        ),
        (
            "strongest fixed",
            {"metrics": _metrics(statelease_delta=0.84, strongest_delta=0.79)},
            "no_more_than_5_percent_worse_than_strongest_fixed",
        ),
        (
            "trajectory",
            {"trajectory": _trajectory(statelease=1.0, cc1=1.0)},
            "trajectory_nmse_auc_lower_than_cc1",
        ),
        (
            "top1",
            {"metrics": _metrics(statelease_top1=0.89, best_top1=0.91)},
            "top1_trail_at_most_0_01",
        ),
    ],
)
def test_stage_a_gate_records_each_independent_failure(
    name: str,
    kwargs: dict[str, object],
    check: str,
) -> None:
    del name
    gate = _gate(**kwargs)  # type: ignore[arg-type]

    assert gate["passed"] is False
    assert gate["checks"][check]["passed"] is False  # type: ignore[index]
    assert gate["checks"][check]["error"]  # type: ignore[index]


def test_stage_a_gate_fails_relative_checks_when_baseline_is_nonpositive() -> None:
    metrics = _metrics()
    for method in FIXED_REPLAY_METHODS:
        metrics[method]["candidate_nll"] = 0.4
        metrics[method]["delta_nll"] = -0.1

    gate = _gate(metrics=metrics)

    assert gate["passed"] is False
    assert (
        gate["checks"]["cc1_excess_nll_reduction_at_least_10_percent"]["passed"]  # type: ignore[index]
        is False
    )
    assert (
        gate["checks"]["no_more_than_5_percent_worse_than_strongest_fixed"]["passed"]  # type: ignore[index]
        is False
    )


def test_stage_a_gate_rejects_missing_or_extra_methods() -> None:
    metrics = _metrics()
    del metrics["fixed_cc5"]
    metrics["invented"] = next(iter(metrics.values())).copy()

    gate = _gate(metrics=metrics)

    check = gate["checks"]["stage0_and_artifact_integrity"]  # type: ignore[index]
    assert check["passed"] is False
    assert "frozen set" in check["error"]


def test_stage_a_gate_rejects_ties_not_assigned_to_c5() -> None:
    evidence = _update_evidence()
    tie_record = next(row for row in evidence if row["tie"] is True)
    tie_record["boundary"] = 4

    gate = _gate(update_evidence=evidence)

    check = gate["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert check["passed"] is False
    assert "tie" in check["error"]


def test_stage_a_gate_rejects_boundary_evidence_diagnostic_mismatch() -> None:
    diagnostics = _diagnostics()
    diagnostics[0]["boundary4_count"] += 1

    gate = _gate(diagnostics=diagnostics)

    check = gate["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert check["passed"] is False
    assert "does not match" in check["error"]


def test_stage_a_gate_rejects_inconsistent_delta_nll_and_nonfinite_flag() -> None:
    metrics = _metrics()
    metrics["fixed_cc4"]["delta_nll"] = 999.0
    gate = _gate(metrics=metrics)
    assert gate["passed"] is False
    assert (
        gate["checks"]["stage0_and_artifact_integrity"]["passed"] is False  # type: ignore[index]
    )

    metrics = _metrics()
    metrics["fixed_cc4"]["all_logits_finite"] = False
    gate = _gate(metrics=metrics)
    assert gate["passed"] is False
    assert gate["checks"]["all_primary_values_finite"]["passed"] is False  # type: ignore[index]


def test_stage_a_gate_rejects_truncated_aligned_metric_coverage() -> None:
    metrics = _metrics()
    for row in metrics.values():
        row["token_count"] = FROZEN_STAGE_A_ALIGNED_TOKENS - 1

    gate = _gate(metrics=metrics)

    check = gate["checks"]["stage0_and_artifact_integrity"]  # type: ignore[index]
    assert check["passed"] is False
    assert f"must equal the frozen Stage-A count {FROZEN_STAGE_A_ALIGNED_TOKENS}" in check["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("scalar", "must be a mapping"),
        ("writes", "scored_write_count must equal 38"),
        ("layers", "layer_value_count must equal 684"),
    ],
)
def test_stage_a_gate_rejects_incomplete_trajectory_coverage(
    mutation: str,
    expected_error: str,
) -> None:
    trajectory: dict[str, object] = _trajectory()
    if mutation == "scalar":
        trajectory[STATELEASE_METHOD] = 0.8
    elif mutation == "writes":
        trajectory[STATELEASE_METHOD]["scored_write_count"] = (  # type: ignore[index]
            FROZEN_STAGE_A_ALIGNED_TOKENS - 1
        )
    else:
        trajectory[STATELEASE_METHOD]["layer_value_count"] = (  # type: ignore[index]
            FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES - 1
        )

    gate = _gate(trajectory=trajectory)

    check = gate["checks"]["stage0_and_artifact_integrity"]  # type: ignore[index]
    assert check["passed"] is False
    assert expected_error in check["error"]


def test_stage_a_gate_rejects_truncated_or_duplicate_diagnostics() -> None:
    truncated = _gate(diagnostics=_diagnostics()[:-1])
    truncated_check = truncated["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert truncated_check["passed"] is False
    assert "exactly 18 recurrent layers" in truncated_check["error"]

    duplicated = _diagnostics()
    duplicated[-1]["layer_index"] = duplicated[0]["layer_index"]
    duplicate_gate = _gate(diagnostics=duplicated)
    duplicate_check = duplicate_gate["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert duplicate_check["passed"] is False
    assert "duplicate layer_index" in duplicate_check["error"]


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("state_updates", FROZEN_STAGE_A_FORWARD_COUNT - 1, "state_updates must equal 39"),
        ("tokens_observed", FROZEN_STAGE_A_TOKENS_OBSERVED - 1, "tokens_observed must equal 107"),
    ],
)
def test_stage_a_gate_rejects_incomplete_per_layer_diagnostic_counts(
    field: str,
    value: int,
    expected_error: str,
) -> None:
    diagnostics = _diagnostics()
    diagnostics[0][field] = value

    gate = _gate(diagnostics=diagnostics)

    check = gate["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert check["passed"] is False
    assert expected_error in check["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("truncated", "exactly 702 layer-write records"),
        ("index", "update_index must equal"),
        ("layer", "layer_index must equal"),
        ("state", "state_index must equal 0"),
        ("prefill_tokens", "token_count must equal 69"),
        ("decode_tokens", "token_count must equal 1"),
        ("prefill_boundary", "prefill evidence must not report a boundary"),
    ],
)
def test_stage_a_gate_rejects_incomplete_or_misaligned_update_evidence(
    mutation: str,
    expected_error: str,
) -> None:
    evidence = _update_evidence()
    if mutation == "truncated":
        evidence.pop()
    elif mutation == "index":
        evidence[1]["update_index"] = 999
    elif mutation == "layer":
        evidence[1]["layer_index"] = FROZEN_STAGE_A_RECURRENT_LAYER_INDICES[-1]
    elif mutation == "state":
        evidence[1]["state_index"] = 1
    elif mutation == "prefill_tokens":
        evidence[1]["token_count"] = 1
    elif mutation == "decode_tokens":
        evidence[len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)]["token_count"] = (
            FROZEN_STAGE_A_PROMPT_TOKENS
        )
    else:
        evidence[1]["boundary"] = 4

    gate = _gate(update_evidence=evidence)

    check = gate["checks"]["only_c4_c5_and_ties_to_c5"]  # type: ignore[index]
    assert check["passed"] is False
    assert expected_error in check["error"]
