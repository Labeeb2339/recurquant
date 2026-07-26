from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import screen_rht_cqer as screen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE_A_EVIDENCE = (
    REPOSITORY_ROOT
    / "evidence"
    / "experiment009-rht-cqer-stage-a-666-5be8d48.json"
)


def _metric(
    *,
    tokens: int,
    delta_nll: float,
    mean_kl: float,
    top1: float,
) -> dict[str, float | int | bool]:
    return {
        "token_count": tokens,
        "mean_kl": mean_kl,
        "cvar95_kl": mean_kl * 2,
        "max_kl": mean_kl * 3,
        "top1_agreement": top1,
        "reference_nll": 2.0,
        "candidate_nll": 2.0 + delta_nll,
        "delta_nll": delta_nll,
        "all_logits_finite": True,
    }


def _diagnostics(method: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for layer, quota in screen.FROZEN_LAYER_QUOTAS.items():
        records.append(
            {
                "layer_index": layer,
                "quota": quota,
                "current_selected_count": quota,
                "state_updates": screen.EXPECTED_STATE_WRITES,
                "observations_staged": screen.EXPECTED_STATE_WRITES,
                "observations_committed": screen.EXPECTED_STATE_WRITES,
                "tokens_observed": screen.EXPECTED_QUERY_TOKENS,
                "pending_observation": False,
                "confirmation_two": False,
                "selection_method": (
                    "query_ema32_weighted_aligned_mse_reduction"
                    if method == screen.CQER_METHOD
                    else screen.RHT_METHOD
                ),
                **(
                    {
                        "state_codec": "right_rht_sha256_signs_v1",
                        "state_codec_seed": screen.SEED,
                        "state_codec_axis": "value",
                        "state_codec_normalization": "orthonormal",
                        "state_codec_persistent_tensor_bytes": 0,
                    }
                    if method == screen.RHT_METHOD
                    else {}
                ),
                "current_mask_sha256": (
                    "a" * 64 if method == screen.CQER_METHOD else "b" * 64
                ),
            }
        )
    return records


def _state_error(*, sse: float, method: str) -> dict[str, object]:
    record_count = (
        screen.EXPECTED_STATE_WRITES * len(screen.FROZEN_LINEAR_LAYERS)
    )
    element_count = 16 * 128 * 128
    mse = sse / (record_count * element_count)
    mask_hash = "a" * 64 if method == screen.CQER_METHOD else "b" * 64
    selection_method = (
        "query_ema32_weighted_aligned_mse_reduction"
        if method == screen.CQER_METHOD
        else screen.RHT_METHOD
    )
    records = [
        {
            "update_index": write * len(screen.FROZEN_LINEAR_LAYERS) + offset,
            "layer_index": layer,
            "state_index": 0,
            "shape": [1, 16, 128, 128],
            "mean_squared_error": mse,
            "max_absolute_error": 0.1,
            "relative_l2_error": 0.1,
            "source_dtype": "torch.float32",
            "selection_method": selection_method,
            "total_groups": 2048,
            "high_precision_groups": screen.FROZEN_LAYER_QUOTAS[layer],
            "high_precision_mask_sha256": mask_hash,
        }
        for write in range(screen.EXPECTED_STATE_WRITES)
        for offset, layer in enumerate(screen.FROZEN_LINEAR_LAYERS)
    ]
    return screen.aggregate_state_error_evidence(records)


def _passing_gate_inputs() -> dict[str, object]:
    aligned = {
        screen.CQER_METHOD: _metric(
            tokens=screen.ALIGNED_TOKENS,
            delta_nll=1.0,
            mean_kl=0.5,
            top1=0.75,
        ),
        screen.RHT_METHOD: _metric(
            tokens=screen.ALIGNED_TOKENS,
            delta_nll=0.8,
            mean_kl=0.4,
            top1=0.76,
        ),
    }
    full = {
        screen.CQER_METHOD: _metric(
            tokens=screen.CODE_TOKENS,
            delta_nll=0.9,
            mean_kl=0.45,
            top1=0.76,
        ),
        screen.RHT_METHOD: _metric(
            tokens=screen.CODE_TOKENS,
            delta_nll=0.7,
            mean_kl=0.35,
            top1=0.77,
        ),
    }
    storage_record = {
        "payload_bytes": screen.TARGET_PAYLOAD_BYTES,
        "scale_bytes": screen.TARGET_SCALE_BYTES,
        "mask_bytes": screen.TARGET_MASK_BYTES,
        "resident_bytes": screen.TARGET_PACKED_STATE_BYTES,
        "selector_auxiliary_bytes": screen.TARGET_SELECTOR_BYTES,
        "resident_bytes_including_selector": screen.TARGET_TOTAL_BYTES,
        "high_precision_groups": screen.TARGET_PROMOTED_ROWS,
    }
    return {
        "aligned_metrics": aligned,
        "full_code_metrics": full,
        "storage": {
            screen.CQER_METHOD: dict(storage_record),
            screen.RHT_METHOD: dict(storage_record),
        },
        "selector_diagnostics": {
            method: _diagnostics(method) for method in screen.METHODS
        },
        "state_errors": {
            screen.CQER_METHOD: _state_error(
                sse=100.0,
                method=screen.CQER_METHOD,
            ),
            screen.RHT_METHOD: _state_error(
                sse=40.0,
                method=screen.RHT_METHOD,
            ),
        },
        "unit_evidence": {
            "inverse_relative_l2": 2e-7,
            "physical_pack_matches_transformed_qdq": True,
            "physical_pack_max_abs_difference": 0.0,
            "sign_schedule_sha256": screen.EXPECTED_SIGN_SCHEDULE_SHA256,
        },
        "integrity": {
            "repository_clean_at_start": True,
            "repository_clean_at_end": True,
            "repository_commit_stable": True,
            "source_hashes_stable": True,
            "identity_authenticated_before_model_weights": True,
            "protected_window_8_16_accessed": False,
        },
    }


def test_stage_a_gate_passes_only_the_complete_frozen_conjunction() -> None:
    gate = screen.evaluate_stage_a_gate(**_passing_gate_inputs())

    assert gate["passed"] is True
    assert all(check["passed"] is True for check in gate["checks"].values())
    assert (
        gate["checks"]["state_sse_relative_reduction"]["relative_reduction"]
        == pytest.approx(0.6)
    )
    assert (
        gate["checks"]["aligned_excess_nll_relative_reduction"][
            "relative_reduction"
        ]
        == pytest.approx(0.2)
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda values: values["integrity"].__setitem__(
                "repository_clean_at_end", False
            ),
            "clean_stable_repository",
        ),
        (
            lambda values: values["storage"][screen.RHT_METHOD].__setitem__(
                "resident_bytes", screen.TARGET_PACKED_STATE_BYTES - 1
            ),
            "exact_equal_storage",
        ),
        (
            lambda values: values["state_errors"][screen.RHT_METHOD].__setitem__(
                "aggregate_state_sse", 60.0
            ),
            "state_sse_relative_reduction",
        ),
        (
            lambda values: values["aligned_metrics"][screen.RHT_METHOD].__setitem__(
                "delta_nll", 0.95
            ),
            "aligned_excess_nll_relative_reduction",
        ),
        (
            lambda values: values["unit_evidence"].__setitem__(
                "inverse_relative_l2", screen.MAX_RHT_INVERSE_RELATIVE_L2
            ),
            "independent_rht_numeric_evidence",
        ),
    ],
)
def test_stage_a_gate_fails_closed_on_frozen_condition_drift(
    mutation,
    failed_check: str,
) -> None:
    values = deepcopy(_passing_gate_inputs())
    mutation(values)

    gate = screen.evaluate_stage_a_gate(**values)

    assert gate["passed"] is False
    assert gate["checks"][failed_check]["passed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_method", "unexpected_selector"),
        ("state_codec_seed", screen.SEED + 1),
        ("state_codec_axis", "key"),
        ("state_codec_persistent_tensor_bytes", 1),
    ],
)
def test_stage_a_gate_rejects_rht_method_or_codec_drift(
    field: str,
    value: object,
) -> None:
    inputs = deepcopy(_passing_gate_inputs())
    inputs["selector_diagnostics"][screen.RHT_METHOD][0][field] = value

    gate = screen.evaluate_stage_a_gate(**inputs)

    assert gate["passed"] is False
    assert (
        gate["checks"]["finite_metrics_exact_quotas_and_handshakes"]["passed"]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values["aligned_metrics"][screen.RHT_METHOD].__setitem__(
            "candidate_nll", 999.0
        ),
        lambda values: values["aligned_metrics"][screen.RHT_METHOD].__setitem__(
            "reference_nll", 1.5
        ),
        lambda values: values["aligned_metrics"][screen.RHT_METHOD].__setitem__(
            "mean_kl", -0.1
        ),
        lambda values: values["aligned_metrics"][screen.RHT_METHOD].__setitem__(
            "top1_agreement", 1.2
        ),
        lambda values: values["state_errors"][screen.RHT_METHOD]["records"][
            0
        ].__setitem__("mean_squared_error", 1.0),
        lambda values: values["state_errors"][screen.RHT_METHOD]["records"][
            -1
        ].__setitem__("high_precision_mask_sha256", "c" * 64),
    ],
)
def test_stage_a_gate_recomputes_metrics_state_totals_and_mask_links(
    mutation,
) -> None:
    inputs = deepcopy(_passing_gate_inputs())
    mutation(inputs)

    gate = screen.evaluate_stage_a_gate(**inputs)

    assert gate["passed"] is False
    assert (
        gate["checks"]["finite_metrics_exact_quotas_and_handshakes"]["passed"]
        is False
    )


def test_committed_stage_a_artifact_passes_the_hardened_gate() -> None:
    document = json.loads(STAGE_A_EVIDENCE.read_text(encoding="utf-8"))
    evidence = document["evidence"]
    repository = evidence["repository"]
    integrity = {
        "repository_clean_at_start": repository["start"]["worktree_clean"],
        "repository_clean_at_end": repository["end"]["worktree_clean"],
        "repository_commit_stable": repository["stable_commit"],
        "source_hashes_stable": evidence["source_files"]["stable"],
        "identity_authenticated_before_model_weights": evidence["dataset"][
            "identity_authenticated_before_model_weights"
        ],
        "protected_window_8_16_accessed": False,
    }

    gate = screen.evaluate_stage_a_gate(
        aligned_metrics=evidence["metrics_aligned"],
        full_code_metrics=evidence["metrics_full_code_secondary"],
        storage=evidence["storage"]["candidates"],
        selector_diagnostics=evidence["selector_diagnostics"],
        state_errors=evidence["state_error"],
        unit_evidence=evidence["unit_evidence"],
        integrity=integrity,
    )

    assert gate["passed"] is True
    assert all(check["passed"] is True for check in gate["checks"].values())


def test_stage_a_identity_accepts_only_frozen_task_hash_and_token_counts() -> None:
    identity = screen.validate_stage_a_identity(
        task_id=screen.TASK_ID,
        row_sha256=screen.TASK_ROW_SHA256,
        prompt_tokens=screen.PROMPT_TOKENS,
        code_tokens=screen.CODE_TOKENS,
        aligned_scored_tokens=screen.ALIGNED_TOKENS,
    )

    assert identity["authenticated_before_model_weights"] is True
    assert identity["aligned_scored_tokens"] == identity["code_tokens"] - 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", 667, "locked"),
        ("row_sha256", "0" * 64, "row SHA"),
        ("prompt_tokens", 68, "token identity"),
        ("code_tokens", 40, "token identity"),
        ("aligned_scored_tokens", 37, "token identity"),
    ],
)
def test_stage_a_identity_rejects_every_frozen_identity_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "task_id": screen.TASK_ID,
        "row_sha256": screen.TASK_ROW_SHA256,
        "prompt_tokens": screen.PROMPT_TOKENS,
        "code_tokens": screen.CODE_TOKENS,
        "aligned_scored_tokens": screen.ALIGNED_TOKENS,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        screen.validate_stage_a_identity(**values)


def test_state_error_evidence_aggregates_mse_to_sse_by_layer_and_write() -> None:
    records = [
        {
            "update_index": 0,
            "layer_index": 0,
            "state_index": 0,
            "shape": [1, 1, 2, 2],
            "mean_squared_error": 0.25,
            "max_absolute_error": 1.0,
            "relative_l2_error": 0.1,
        },
        {
            "update_index": 1,
            "layer_index": 1,
            "state_index": 0,
            "shape": [1, 1, 2, 3],
            "mean_squared_error": 0.5,
            "max_absolute_error": 1.5,
            "relative_l2_error": 0.2,
        },
    ]

    aggregate = screen.aggregate_state_error_evidence(records)

    assert aggregate["record_count"] == 2
    assert aggregate["element_count"] == 10
    assert aggregate["aggregate_state_sse"] == pytest.approx(4.0)
    assert aggregate["aggregate_state_mse"] == pytest.approx(0.4)
    assert aggregate["per_layer"]["0"]["state_sse"] == pytest.approx(1.0)
    assert aggregate["per_layer"]["1"]["state_sse"] == pytest.approx(3.0)
    assert aggregate["per_write"]["0"]["record_count"] == 2
    assert [record["write_ordinal"] for record in aggregate["records"]] == [0, 0]


def test_state_error_evidence_rejects_duplicate_or_nonfinite_records() -> None:
    record = {
        "update_index": 0,
        "layer_index": 0,
        "state_index": 0,
        "shape": [1, 2],
        "mean_squared_error": 0.1,
        "max_absolute_error": 0.2,
        "relative_l2_error": 0.3,
    }
    with pytest.raises(ValueError, match="duplicate"):
        screen.aggregate_state_error_evidence([record, dict(record)])
    with pytest.raises(ValueError, match="finite"):
        screen.aggregate_state_error_evidence(
            [{**record, "mean_squared_error": float("nan")}]
        )
