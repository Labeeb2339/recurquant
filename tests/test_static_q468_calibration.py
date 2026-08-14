from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import recurquant.static_q468_calibration as calibration
from recurquant.evidence import canonical_json_bytes
from recurquant.multibit_policy import allocate_exact_multibit_codes
from recurquant.quantization import QuantizationSpec, quantize_dequantize
from recurquant.rht import RHT_SEED, right_rht_encode
from recurquant.static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    StaticRhtQ468Geometry,
    build_static_rht_q468_policy,
)
from recurquant.static_q468_calibration import (
    CALIBRATION_SCORE_ARTIFACT_KIND,
    CALIBRATION_SCORE_ARTIFACT_PROFILE,
    CALIBRATION_SCORE_ARTIFACT_REVISION,
    COMPARATOR_SCORE_ARTIFACT_KIND,
    COMPARATOR_SCORE_ARTIFACT_PROFILE,
    COMPARATOR_SCORE_ARTIFACT_REVISION,
    FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
    FROZEN_SOURCE_TENSOR_CONTRACT,
    FROZEN_UNWEIGHTED_MSE_PROFILE,
    GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND,
    GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE,
    GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION,
    AnchorDistortionBatch,
    CalibrationAggregate,
    CalibrationSequenceScores,
    ComparatorAggregate,
    ComparatorSequenceScores,
    FrozenComparatorEndpointBatch,
    UnweightedEndpointBatch,
    aggregate_calibration_scores,
    aggregate_comparator_scores,
    allocate_frozen_static_q468_code_maps,
    allocate_static_q468_code_map,
    allocate_unweighted_endpoint_policy,
    balanced_sha_rank_halves,
    build_calibration_score_artifact,
    build_frozen_calibration_score_artifact,
    build_frozen_comparator_score_artifact,
    build_frozen_split_half_stability_artifact,
    calibration_sequence_rank_sha256,
    compute_rht_diagonal_empirical_fisher_h1_endpoints,
    compute_rht_unweighted_mse_endpoints,
    deserialize_calibration_score_artifact,
    deserialize_comparator_score_artifact,
    deserialize_frozen_split_half_stability_artifact,
    evaluate_policy_stability,
    fisher_h1_boundary_positions,
    fit_split_half_policy,
    frozen_anchor_positions,
    identity_record_sha256,
    per_layer_mean_bitwidth_shifts,
    q8_set_jaccard,
    reduce_anchor_distortions,
    reduce_frozen_anchor_distortions,
    reduce_frozen_comparator_endpoints,
    reduce_unweighted_endpoint_anchors,
    static_q468_code_map_sha256,
    verify_calibration_score_artifact,
    verify_comparator_score_artifact,
    verify_frozen_split_half_stability_artifact,
)

TINY_GEOMETRY = StaticRhtQ468Geometry(
    layer_indices=(0, 2),
    heads=1,
    key_rows=2,
    value_width=4,
    target_resident_bytes=64,
)
FAKE_IDENTITY_SHA256 = hashlib.sha256(b"experiment-013-identity").hexdigest()
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "calibration_test_capture_static_q468_identity_input",
    REPOSITORY_ROOT / "scripts" / "capture_static_q468_identity_input.py",
)
assert CAPTURE_SPEC is not None and CAPTURE_SPEC.loader is not None
capture = importlib.util.module_from_spec(CAPTURE_SPEC)
sys.modules[CAPTURE_SPEC.name] = capture
CAPTURE_SPEC.loader.exec_module(capture)


def _batch(
    *,
    family: str,
    config: str,
    canonical_id: str,
    values: torch.Tensor,
    ruler_category: str | None = None,
    seed: int | None = None,
    configured_length: int | None = None,
    token_count: int = 1,
    energy: torch.Tensor | None = None,
) -> AnchorDistortionBatch:
    anchors = frozen_anchor_positions(token_count)
    row = values.to(torch.float32).reshape(1, -1)
    repeated = row.repeat(len(anchors), 1)
    selected_energy = torch.ones_like(repeated) if energy is None else energy.to(torch.float32)
    return AnchorDistortionBatch(
        family=family,  # type: ignore[arg-type]
        config=config,
        ruler_category=ruler_category,  # type: ignore[arg-type]
        canonical_id=canonical_id,
        seed=seed,
        configured_length=configured_length,
        token_count=token_count,
        anchor_positions=anchors,
        query_energy=selected_energy,
        q4_mse=repeated,
        q6_mse=repeated / 2,
        q8_mse=repeated / 4,
    )


def _score(
    family: str,
    config: str,
    canonical_id: str,
    scalar: float,
    *,
    ruler_category: str | None = None,
    seed: int | None = None,
    configured_length: int | None = None,
) -> CalibrationSequenceScores:
    return reduce_anchor_distortions(
        _batch(
            family=family,
            config=config,
            canonical_id=canonical_id,
            values=scalar * torch.arange(1, 5, dtype=torch.float64),
            ruler_category=ruler_category,
            seed=seed,
            configured_length=configured_length,
        )
    )


def _calibration_sequences(*, identical_pairs: bool = False) -> list[CalibrationSequenceScores]:
    pairs: list[tuple[str, str | None, tuple[str, str], tuple[float, float]]] = [
        ("mbpp", None, ("default", "default"), (3.0, 3.0 if identical_pairs else 9.0)),
        ("pg19", None, ("default", "default"), (12.0, 12.0 if identical_pairs else 18.0)),
        (
            "ruler",
            "retrieval",
            ("niah_single_1", "niah_multikey_1"),
            (3.0, 3.0 if identical_pairs else 9.0),
        ),
        (
            "ruler",
            "multi_hop_tracing",
            ("vt", "vt"),
            (6.0, 6.0 if identical_pairs else 12.0),
        ),
        (
            "ruler",
            "aggregation",
            ("cwe", "fwe"),
            (9.0, 9.0 if identical_pairs else 15.0),
        ),
        (
            "ruler",
            "question_answering",
            ("qa_1", "qa_2"),
            (12.0, 12.0 if identical_pairs else 18.0),
        ),
    ]
    result: list[CalibrationSequenceScores] = []
    for group_index, (family, category, configs, values) in enumerate(pairs):
        for member_index, value in enumerate(values):
            config = configs[member_index]
            result.append(
                _score(
                    family,
                    config,
                    f"{family}-{config}-{group_index}-{member_index}",
                    value,
                    ruler_category=category,
                    seed=100 + member_index if family == "ruler" else None,
                    configured_length=(2_048, 4_096)[member_index] if family == "ruler" else None,
                )
            )
    return result


def _tiny_aggregate() -> CalibrationAggregate:
    return aggregate_calibration_scores(_calibration_sequences())


def _synthetic_frozen_aggregate() -> CalibrationAggregate:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    return CalibrationAggregate(
        d4=torch.full((rows,), 4.0, dtype=torch.float64),
        d6=torch.full((rows,), 2.0, dtype=torch.float64),
        d8=torch.full((rows,), 1.0, dtype=torch.float64),
        family_sequence_counts=(("mbpp", 128), ("pg19", 16), ("ruler", 16)),
        ruler_category_sequence_counts=(
            ("retrieval", 4),
            ("multi_hop_tracing", 4),
            ("aggregation", 4),
            ("question_answering", 4),
        ),
        sequence_score_manifest_sha256="d" * 64,
        source_contract=FROZEN_SOURCE_TENSOR_CONTRACT,
        identity_record_manifest_sha256="e" * 64,
    )


def _synthetic_frozen_half_aggregate(identity_manifest: str) -> CalibrationAggregate:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    row_axis = torch.arange(rows, dtype=torch.float64)
    return CalibrationAggregate(
        d4=4.0 + row_axis / rows,
        d6=2.0 + row_axis / (2 * rows),
        d8=1.0 + row_axis / (4 * rows),
        family_sequence_counts=(("mbpp", 64), ("pg19", 8), ("ruler", 8)),
        ruler_category_sequence_counts=(
            ("retrieval", 2),
            ("multi_hop_tracing", 2),
            ("aggregation", 2),
            ("question_answering", 2),
        ),
        sequence_score_manifest_sha256="c" * 64,
        source_contract=FROZEN_SOURCE_TENSOR_CONTRACT,
        identity_record_manifest_sha256=identity_manifest,
    )


def _rehashed_document(document: dict[str, object]) -> bytes:
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    document["canonical_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    return canonical_json_bytes(document)


def _relabel_generic_document_as_official(raw: bytes) -> dict[str, object]:
    document = json.loads(raw)
    document["artifact_kind"] = CALIBRATION_SCORE_ARTIFACT_KIND
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    evidence["artifact_profile"] = CALIBRATION_SCORE_ARTIFACT_PROFILE
    evidence["artifact_revision"] = CALIBRATION_SCORE_ARTIFACT_REVISION
    evidence["identity_record_manifest_sha256"] = "e" * 64
    return document


def _comparator_sequence(
    selector_profile: str,
    family: str,
    config: str,
    canonical_id: str,
    scalar: float,
    *,
    ruler_category: str | None = None,
    seed: int | None = None,
    configured_length: int | None = None,
) -> ComparatorSequenceScores:
    token_count = 3
    endpoint_positions = (
        frozen_anchor_positions(token_count)
        if selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE
        else fisher_h1_boundary_positions(token_count)
    )
    token_hash = calibration.sequence_token_ids_sha256((11, 12, 13))
    token_span = (
        ("prefill_start", 0),
        ("prefill_stop", 1),
        ("scored_start", 1),
        ("scored_stop", 3),
        ("cache_exposed_start", 3),
        ("cache_exposed_stop", 3),
    )
    identity_anchor_hash = calibration.identity_anchor_manifest_sha256(
        canonical_id=canonical_id,
        sequence_length=token_count,
        sequence_token_ids_sha256_value=token_hash,
        token_span=token_span,
    )
    identity_record_hash = hashlib.sha256(f"record:{canonical_id}".encode()).hexdigest()
    fisher_boundary_hash = hashlib.sha256(f"fisher:{canonical_id}".encode()).hexdigest()
    position_payload = calibration._comparator_position_payload(
        selector_profile=selector_profile,
        token_count=token_count,
        endpoint_positions=endpoint_positions,
        sequence_token_ids_sha256_value=token_hash,
        identity_anchor_manifest_sha256_value=identity_anchor_hash,
        identity_record_sha256_value=identity_record_hash,
        fisher_boundary_sha256=fisher_boundary_hash,
    )
    position_hash = calibration._domain_json_sha256(
        calibration._COMPARATOR_POSITION_HASH_DOMAIN,
        position_payload,
    )
    endpoint_hash = hashlib.sha256(
        f"endpoint:{selector_profile}:{canonical_id}".encode()
    ).hexdigest()
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    d4 = torch.full((rows,), scalar, dtype=torch.float64)
    d6 = torch.full((rows,), scalar / 2, dtype=torch.float64)
    d8 = torch.full((rows,), scalar / 4, dtype=torch.float64)
    sequence_hash = calibration._comparator_sequence_score_sha256(
        selector_profile=selector_profile,
        position_manifest_sha256=position_hash,
        endpoint_inputs_sha256=endpoint_hash,
        identity_record_sha256_value=identity_record_hash,
        d4=d4,
        d6=d6,
        d8=d8,
    )
    return ComparatorSequenceScores(
        selector_profile=selector_profile,
        family=family,
        config=config,
        ruler_category=ruler_category,
        canonical_id=canonical_id,
        seed=seed,
        configured_length=configured_length,
        token_count=token_count,
        endpoint_positions=endpoint_positions,
        position_manifest_sha256=position_hash,
        endpoint_inputs_sha256=endpoint_hash,
        sequence_scores_sha256=sequence_hash,
        d4=d4,
        d6=d6,
        d8=d8,
        source_shape=(
            len(endpoint_positions),
            *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape,
        ),
        sequence_token_ids_sha256=token_hash,
        token_span=token_span,
        identity_anchor_manifest_sha256=identity_anchor_hash,
        identity_record_sha256=identity_record_hash,
        fisher_boundary_sha256=fisher_boundary_hash,
        target_nlls_sha256=(
            None
            if selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE
            else hashlib.sha256(f"nll:{canonical_id}".encode()).hexdigest()
        ),
    )


def _comparator_sequences(selector_profile: str) -> list[ComparatorSequenceScores]:
    specifications = [
        ("mbpp", None, "default", 3.0),
        ("mbpp", None, "default", 9.0),
        ("pg19", None, "default", 12.0),
        ("pg19", None, "default", 18.0),
        ("ruler", "retrieval", "niah", 3.0),
        ("ruler", "retrieval", "niah", 9.0),
        ("ruler", "multi_hop_tracing", "vt", 6.0),
        ("ruler", "multi_hop_tracing", "vt", 12.0),
        ("ruler", "aggregation", "cwe", 9.0),
        ("ruler", "aggregation", "fwe", 15.0),
        ("ruler", "question_answering", "qa", 12.0),
        ("ruler", "question_answering", "qa", 18.0),
    ]
    return [
        _comparator_sequence(
            selector_profile,
            family,
            config,
            f"{family}-{category}-{index}",
            scalar,
            ruler_category=category,
            seed=index if family == "ruler" else None,
            configured_length=2_048 if family == "ruler" else None,
        )
        for index, (family, category, config, scalar) in enumerate(specifications)
    ]


def _synthetic_comparator_aggregate(
    selector_profile: str,
    *,
    identity_manifest_sha256: str = "e" * 64,
) -> ComparatorAggregate:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    row_axis = torch.arange(rows, dtype=torch.float64) / rows
    profile_offset = 0.0 if selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE else 0.125
    provisional = ComparatorAggregate(
        selector_profile=selector_profile,
        d4=4.0 + profile_offset + row_axis,
        d6=2.0 + profile_offset + row_axis / 2,
        d8=1.0 + profile_offset + row_axis / 4,
        family_sequence_counts=(("mbpp", 128), ("pg19", 16), ("ruler", 16)),
        ruler_category_sequence_counts=tuple(
            (category, 4) for category in calibration.RULER_CATEGORY_ORDER
        ),
        position_manifest_sha256=hashlib.sha256(
            f"positions:{selector_profile}".encode()
        ).hexdigest(),
        sequence_score_manifest_sha256=hashlib.sha256(
            f"sequences:{selector_profile}".encode()
        ).hexdigest(),
        identity_record_manifest_sha256=identity_manifest_sha256,
        aggregate_scores_sha256="0" * 64,
    )
    return replace(
        provisional,
        aggregate_scores_sha256=calibration._comparator_aggregate_score_sha256(provisional),
    )


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [
        (1, (0,)),
        (5, (0, 1, 2, 3, 4)),
        (16, tuple(range(16))),
        (17, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16)),
        (
            2_304,
            (
                143,
                287,
                431,
                575,
                719,
                863,
                1007,
                1151,
                1295,
                1439,
                1583,
                1727,
                1871,
                2015,
                2159,
                2303,
            ),
        ),
    ],
)
def test_frozen_anchor_equation(token_count: int, expected: tuple[int, ...]) -> None:
    assert frozen_anchor_positions(token_count) == expected


def test_fisher_h1_boundary_positions_reserve_causal_input_and_target() -> None:
    assert fisher_h1_boundary_positions(3) == (0,)
    assert fisher_h1_boundary_positions(7) == (0, 1, 2, 3, 4)
    assert fisher_h1_boundary_positions(18) == tuple(range(16))
    assert fisher_h1_boundary_positions(19) == frozen_anchor_positions(17)
    with pytest.raises(ValueError, match="at least three tokens"):
        fisher_h1_boundary_positions(2)


def _tiny_endpoint_state() -> torch.Tensor:
    return torch.tensor(
        [
            -1.17,
            -0.83,
            -0.21,
            0.47,
            0.18,
            0.71,
            1.09,
            1.63,
            -0.94,
            -0.37,
            0.26,
            0.88,
            0.33,
            0.69,
            1.31,
            1.91,
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 2, 4)


def _tiny_endpoint_specs() -> tuple[QuantizationSpec, ...]:
    return tuple(
        QuantizationSpec(
            bits=bits,
            group_size=TINY_GEOMETRY.value_width,
            scale_bits=16,
            flatten_last_dims=1,
            rounding="nearest",
            seed=RHT_SEED,
        )
        for bits in (4, 6, 8)
    )


def test_unweighted_mse_endpoint_math_matches_exact_rht_q468_codec() -> None:
    state = _tiny_endpoint_state()
    actual = compute_rht_unweighted_mse_endpoints(state, geometry=TINY_GEOMETRY)
    expected: list[list[torch.Tensor]] = [[], [], []]
    fp32_reductions: list[list[torch.Tensor]] = [[], [], []]
    for local_index, layer_index in enumerate(TINY_GEOMETRY.layer_indices):
        encoded = right_rht_encode(
            state[local_index].unsqueeze(0),
            layer_index=layer_index,
            expected_heads=TINY_GEOMETRY.heads,
        )
        for destination, fp32_destination, specification in zip(
            expected,
            fp32_reductions,
            _tiny_endpoint_specs(),
            strict=True,
        ):
            restored = quantize_dequantize(encoded, specification).tensor
            error = restored - encoded
            destination.append(error.to(torch.float64).square().mean(dim=-1).squeeze(0))
            fp32_destination.append(error.square().mean(dim=-1).squeeze(0).to(torch.float64))
    for observed, rows in zip(actual, expected, strict=True):
        reference = torch.stack(rows)
        assert observed.device.type == "cpu"
        assert observed.dtype == torch.float64
        assert torch.all(observed >= 0).item()
        torch.testing.assert_close(observed, reference, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(observed, torch.stack(legacy_rows))
        for observed, legacy_rows in zip(actual, fp32_reductions, strict=True)
    )


def test_diagonal_empirical_fisher_h1_math_transforms_state_and_gradient() -> None:
    state = _tiny_endpoint_state()
    gradient = torch.tensor(
        [
            0.11,
            -0.29,
            0.53,
            -0.71,
            0.07,
            0.19,
            -0.41,
            0.89,
            -0.13,
            0.31,
            -0.61,
            0.79,
            0.17,
            -0.23,
            0.47,
            -0.97,
        ],
        dtype=torch.float32,
    ).reshape_as(state)
    actual = compute_rht_diagonal_empirical_fisher_h1_endpoints(
        state,
        gradient,
        geometry=TINY_GEOMETRY,
    )
    expected: list[list[torch.Tensor]] = [[], [], []]
    fp32_reductions: list[list[torch.Tensor]] = [[], [], []]
    for local_index, layer_index in enumerate(TINY_GEOMETRY.layer_indices):
        encoded_state = right_rht_encode(
            state[local_index].unsqueeze(0),
            layer_index=layer_index,
            expected_heads=TINY_GEOMETRY.heads,
        )
        encoded_gradient = right_rht_encode(
            gradient[local_index].unsqueeze(0),
            layer_index=layer_index,
            expected_heads=TINY_GEOMETRY.heads,
        )
        for destination, fp32_destination, specification in zip(
            expected,
            fp32_reductions,
            _tiny_endpoint_specs(),
            strict=True,
        ):
            restored = quantize_dequantize(encoded_state, specification).tensor
            error = restored - encoded_state
            destination.append(
                (
                    0.5
                    * (
                        encoded_gradient.to(torch.float64).square()
                        * error.to(torch.float64).square()
                    ).sum(dim=-1)
                ).squeeze(0)
            )
            fp32_destination.append(
                (0.5 * (encoded_gradient.square() * error.square()).sum(dim=-1))
                .squeeze(0)
                .to(torch.float64)
            )
    for observed, rows in zip(actual, expected, strict=True):
        reference = torch.stack(rows)
        assert observed.device.type == "cpu"
        assert observed.dtype == torch.float64
        assert torch.all(observed >= 0).item()
        torch.testing.assert_close(observed, reference, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(observed, torch.stack(legacy_rows))
        for observed, legacy_rows in zip(actual, fp32_reductions, strict=True)
    )


def test_unweighted_endpoint_reduction_and_exact_policy_ignore_query_proxies() -> None:
    positions = frozen_anchor_positions(3)
    q4 = torch.tensor(
        [
            [[4.0, 3.0, 2.0, 1.0]],
            [[8.0, 6.0, 4.0, 2.0]],
            [[12.0, 9.0, 6.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    batch = UnweightedEndpointBatch(
        selector_profile=FROZEN_UNWEIGHTED_MSE_PROFILE,
        token_count=3,
        anchor_positions=positions,
        q4_scores=q4,
        q6_scores=q4 / 2,
        q8_scores=q4 / 4,
    )

    reduced = reduce_unweighted_endpoint_anchors(batch)
    torch.testing.assert_close(reduced[0], q4.mean(dim=0).reshape(-1))
    torch.testing.assert_close(reduced[1], (q4 / 2).mean(dim=0).reshape(-1))
    torch.testing.assert_close(reduced[2], (q4 / 4).mean(dim=0).reshape(-1))
    first = allocate_unweighted_endpoint_policy(batch, marginal_steps=3)
    second = allocate_unweighted_endpoint_policy(batch, marginal_steps=3)
    assert torch.equal(first, second)
    assert first.dtype == torch.uint8
    assert int(first.to(torch.int64).sum().item()) == 3

    fisher_batch = UnweightedEndpointBatch(
        selector_profile=FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        token_count=5,
        anchor_positions=fisher_h1_boundary_positions(5),
        q4_scores=q4,
        q6_scores=q4 / 2,
        q8_scores=q4 / 4,
    )
    assert reduce_unweighted_endpoint_anchors(fisher_batch)[0].shape == (4,)


def _identity_v5_record(
    *,
    canonical_id: str,
    sequence_token_ids: tuple[int, ...],
) -> dict[str, object]:
    tokenizer_manifest_sha256 = "a" * 64
    captured_record = capture._base_record(
        phase="calibration",
        family="mbpp",
        canonical_id=canonical_id,
        config="full",
        seed=None,
        configured_length=None,
        ruler_category=None,
        generator_receipt_sha256=None,
        source_payload={"task_id": canonical_id},
        formatted_payload={"prompt": "fixture"},
        prompt_ids=sequence_token_ids,
        target_ids=(),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
    )
    captured_record = capture._assign_sha_ranks([captured_record])[0]
    return capture.resolver._normalize_record(
        captured_record,
        index=0,
        phase="calibration",
        tokenizer_hash=tokenizer_manifest_sha256,
    )


def test_frozen_comparator_reduction_binds_v5_positions_inputs_scores_and_nll_hash() -> None:
    token_ids = (17, 18, 19)
    record = _identity_v5_record(canonical_id="comparator-1", sequence_token_ids=token_ids)
    trailing = FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape
    mse_positions = frozen_anchor_positions(len(token_ids))
    mse_values = torch.arange(
        1,
        len(mse_positions) * math.prod(trailing) + 1,
        dtype=torch.float64,
    ).reshape(len(mse_positions), *trailing)
    mse = reduce_frozen_comparator_endpoints(
        FrozenComparatorEndpointBatch(
            selector_profile=FROZEN_UNWEIGHTED_MSE_PROFILE,
            family="mbpp",
            config="full",
            ruler_category=None,
            canonical_id="comparator-1",
            seed=None,
            configured_length=None,
            token_count=len(token_ids),
            endpoint_positions=mse_positions,
            q4_scores=mse_values,
            q6_scores=mse_values / 2,
            q8_scores=mse_values / 4,
            sequence_token_ids=token_ids,
            identity_record=record,
        )
    )
    permuted_mse = reduce_frozen_comparator_endpoints(
        FrozenComparatorEndpointBatch(
            selector_profile=FROZEN_UNWEIGHTED_MSE_PROFILE,
            family="mbpp",
            config="full",
            ruler_category=None,
            canonical_id="comparator-1",
            seed=None,
            configured_length=None,
            token_count=len(token_ids),
            endpoint_positions=mse_positions,
            q4_scores=mse_values.flip(0),
            q6_scores=(mse_values / 2).flip(0),
            q8_scores=(mse_values / 4).flip(0),
            sequence_token_ids=token_ids,
            identity_record=record,
        )
    )
    fisher_positions = fisher_h1_boundary_positions(len(token_ids))
    fisher_values = mse_values[:1] / 10
    fisher = reduce_frozen_comparator_endpoints(
        FrozenComparatorEndpointBatch(
            selector_profile=FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            family="mbpp",
            config="full",
            ruler_category=None,
            canonical_id="comparator-1",
            seed=None,
            configured_length=None,
            token_count=len(token_ids),
            endpoint_positions=fisher_positions,
            q4_scores=fisher_values,
            q6_scores=fisher_values / 2,
            q8_scores=fisher_values / 4,
            sequence_token_ids=token_ids,
            identity_record=record,
            target_nlls=torch.tensor([1.25], dtype=torch.float64),
        )
    )

    torch.testing.assert_close(mse.d4, mse_values.mean(dim=0).reshape(-1))
    torch.testing.assert_close(mse.d4, permuted_mse.d4)
    torch.testing.assert_close(fisher.d4, fisher_values.reshape(-1))
    assert mse.endpoint_positions == (0, 1, 2)
    assert fisher.endpoint_positions == (0,)
    assert mse.position_manifest_sha256 != fisher.position_manifest_sha256
    assert mse.endpoint_inputs_sha256 != fisher.endpoint_inputs_sha256
    assert mse.sequence_scores_sha256 != fisher.sequence_scores_sha256
    assert mse.position_manifest_sha256 == permuted_mse.position_manifest_sha256
    assert mse.endpoint_inputs_sha256 != permuted_mse.endpoint_inputs_sha256
    assert mse.sequence_scores_sha256 != permuted_mse.sequence_scores_sha256
    assert mse.target_nlls_sha256 is None
    assert (
        fisher.target_nlls_sha256
        == hashlib.sha256(
            calibration._COMPARATOR_TARGET_NLL_HASH_DOMAIN
            + calibration._tensor_bytes(torch.tensor([1.25], dtype=torch.float64))
        ).hexdigest()
    )
    assert "target_nlls" not in fisher.manifest_record()
    assert fisher.manifest_record()["target_nlls_sha256"] == fisher.target_nlls_sha256


def test_frozen_comparator_reduction_rejects_v4_substitution_offbyone_and_nonfinite() -> None:
    token_ids = (31, 32, 33)
    record = _identity_v5_record(canonical_id="comparator-2", sequence_token_ids=token_ids)
    positions = fisher_h1_boundary_positions(len(token_ids))
    shape = (len(positions), *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape)
    values = torch.ones(shape, dtype=torch.float64)
    baseline = FrozenComparatorEndpointBatch(
        selector_profile=FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        family="mbpp",
        config="full",
        ruler_category=None,
        canonical_id="comparator-2",
        seed=None,
        configured_length=None,
        token_count=len(token_ids),
        endpoint_positions=positions,
        q4_scores=values,
        q6_scores=values / 2,
        q8_scores=values / 4,
        sequence_token_ids=token_ids,
        identity_record=record,
        target_nlls=torch.tensor([1.25], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="requires target_nlls"):
        reduce_frozen_comparator_endpoints(replace(baseline, target_nlls=None))
    with pytest.raises(ValueError, match=r"A\(T\)/B\(T\)"):
        reduce_frozen_comparator_endpoints(replace(baseline, endpoint_positions=(1,)))
    with pytest.raises(ValueError, match="token-ID SHA-256"):
        reduce_frozen_comparator_endpoints(replace(baseline, sequence_token_ids=(31, 99, 33)))
    legacy = dict(record)
    legacy.pop("fisher_boundary")
    with pytest.raises(ValueError, match="missing=.*fisher_boundary"):
        reduce_frozen_comparator_endpoints(replace(baseline, identity_record=legacy))
    for invalid in (math.nan, math.inf):
        nonfinite = values.clone()
        nonfinite[0, 0, 0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            reduce_frozen_comparator_endpoints(replace(baseline, q4_scores=nonfinite))


def test_fisher_sequence_artifact_requires_exact_target_nll_receipt() -> None:
    fisher = _comparator_sequence(
        FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        "pg19",
        "default",
        "fisher-without-nll-receipt",
        1.0,
    )

    with pytest.raises(ValueError, match="requires a target NLL receipt"):
        calibration.aggregate_comparator_scores([replace(fisher, target_nlls_sha256=None)])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_frozen_anchor_equation_rejects_empty_or_non_integer_counts(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        frozen_anchor_positions(bad)  # type: ignore[arg-type]


def test_anchor_reduction_is_energy_weighted_cpu_fp64_and_hash_bound() -> None:
    token_count = 4
    energy = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    )
    batch = _batch(
        family="mbpp",
        config="default",
        canonical_id="task-1",
        values=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        token_count=token_count,
        energy=energy,
    )

    result = reduce_anchor_distortions(batch)

    expected_q4 = (energy.to(torch.float64) * batch.q4_mse.to(torch.float64)).mean(dim=0)
    assert result.d4.dtype == torch.float64
    assert result.d4.device.type == "cpu"
    assert torch.equal(result.d4, expected_q4)
    assert torch.equal(result.d6, expected_q4 / 2)
    assert torch.equal(result.d8, expected_q4 / 4)
    assert len(result.anchor_manifest_sha256) == 64
    assert len(result.anchor_inputs_sha256) == 64
    assert len(result.sequence_scores_sha256) == 64
    assert reduce_anchor_distortions(batch).sequence_scores_sha256 == result.sequence_scores_sha256


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("anchor_positions", (0,), "anchor_positions"),
        ("query_energy", torch.tensor([[math.nan] * 4] * 4), "finite"),
        ("q4_mse", torch.tensor([[-1.0] * 4] * 4), "non-negative"),
        ("q6_mse", torch.ones(4, 5), "match shape"),
    ],
)
def test_anchor_reduction_rejects_malformed_or_nonfinite_inputs(
    field: str,
    replacement: object,
    message: str,
) -> None:
    original = _batch(
        family="mbpp",
        config="default",
        canonical_id="task-bad",
        values=torch.ones(4),
        token_count=4,
    )
    values = {name: getattr(original, name) for name in AnchorDistortionBatch.__dataclass_fields__}
    values[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        reduce_anchor_distortions(AnchorDistortionBatch(**values))


@pytest.mark.parametrize(
    ("family", "category", "message"),
    [
        ("ruler", None, "ruler_category must be one of"),
        ("ruler", "reasoning", "ruler_category must be one of"),
        ("mbpp", "retrieval", "must be None"),
        ("pg19", "question_answering", "must be None"),
    ],
)
def test_ruler_category_is_required_only_for_ruler(
    family: str,
    category: str | None,
    message: str,
) -> None:
    batch = _batch(
        family=family,
        config="exact-config",
        canonical_id=f"{family}-category-contract",
        values=torch.ones(4),
        ruler_category=category,
    )
    with pytest.raises(ValueError, match=message):
        reduce_anchor_distortions(batch)


@pytest.mark.parametrize(
    ("family", "category", "configured_length", "message"),
    [
        ("ruler", "retrieval", None, "configured_length must be an integer"),
        ("ruler", "retrieval", 0, "configured_length must be positive"),
        ("mbpp", None, 2_048, "configured_length must be None"),
    ],
)
def test_configured_length_is_separate_and_ruler_only(
    family: str,
    category: str | None,
    configured_length: int | None,
    message: str,
) -> None:
    batch = _batch(
        family=family,
        config="exact-config",
        canonical_id=f"{family}-length-contract",
        values=torch.ones(4),
        ruler_category=category,
        configured_length=configured_length,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        reduce_anchor_distortions(batch)


def test_ruler_configured_length_can_differ_from_actual_anchor_length() -> None:
    valid = reduce_anchor_distortions(
        _batch(
            family="ruler",
            config="niah_single_1",
            canonical_id="shorter-generated-sequence",
            values=torch.ones(4),
            ruler_category="retrieval",
            configured_length=4_096,
            token_count=3,
        )
    )
    assert valid.configured_length == 4_096

    with pytest.raises(ValueError, match="cannot exceed"):
        reduce_anchor_distortions(
            _batch(
                family="ruler",
                config="niah_single_1",
                canonical_id="ruler-too-long",
                values=torch.ones(4),
                ruler_category="retrieval",
                configured_length=4,
                token_count=5,
            )
        )
    assert valid.token_count == 3
    assert valid.anchor_positions == (0, 1, 2)


def test_metadata_requires_nfc_text_and_nonnegative_seed() -> None:
    decomposed_id = unicodedata.normalize("NFD", "café")
    assert decomposed_id != "café"
    with pytest.raises(ValueError, match="NFC"):
        reduce_anchor_distortions(
            _batch(
                family="mbpp",
                config="full",
                canonical_id=decomposed_id,
                values=torch.ones(4),
            )
        )
    with pytest.raises(ValueError, match="non-negative"):
        reduce_anchor_distortions(
            _batch(
                family="ruler",
                config="vt",
                canonical_id="vt-negative-seed",
                values=torch.ones(4),
                ruler_category="multi_hop_tracing",
                seed=-1,
                configured_length=2_048,
            )
        )


def test_source_shape_and_dtype_are_bound_into_sequence_hashes() -> None:
    def make(dtype: torch.dtype, shape: tuple[int, ...]) -> AnchorDistortionBatch:
        values = torch.ones(shape, dtype=dtype)
        return AnchorDistortionBatch(
            family="mbpp",
            config="full",
            ruler_category=None,
            canonical_id="source-contract",
            seed=None,
            configured_length=None,
            token_count=1,
            anchor_positions=(0,),
            query_energy=values,
            q4_mse=values,
            q6_mse=values / 2,
            q8_mse=values / 4,
        )

    fp32 = reduce_anchor_distortions(make(torch.float32, (1, 4)))
    fp64 = reduce_anchor_distortions(make(torch.float64, (1, 4)))
    reshaped = reduce_anchor_distortions(make(torch.float32, (1, 1, 4)))

    assert torch.equal(fp32.d4, fp64.d4)
    assert torch.equal(fp32.d4, reshaped.d4)
    assert fp32.sequence_scores_sha256 != fp64.sequence_scores_sha256
    assert fp32.anchor_inputs_sha256 != fp64.anchor_inputs_sha256
    assert fp32.sequence_scores_sha256 != reshaped.sequence_scores_sha256
    assert fp32.source_contract != fp64.source_contract
    assert fp32.source_contract != reshaped.source_contract


def test_frozen_reduction_requires_exact_layer_head_key_row_cpu_fp64_shape() -> None:
    shape = (3, *FROZEN_SOURCE_TENSOR_CONTRACT.trailing_shape)
    values = torch.ones(shape, dtype=torch.float64)
    sequence_token_ids = (17, 18, 19)
    tokenizer_manifest_sha256 = "a" * 64
    captured_record = capture._base_record(
        phase="calibration",
        family="mbpp",
        canonical_id="601",
        config="full",
        seed=None,
        configured_length=None,
        ruler_category=None,
        generator_receipt_sha256=None,
        source_payload={"task_id": 601},
        formatted_payload={"prompt": "fixture"},
        prompt_ids=sequence_token_ids,
        target_ids=(),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
    )
    captured_record = capture._assign_sha_ranks([captured_record])[0]
    identity_record = capture.resolver._normalize_record(
        captured_record,
        index=0,
        phase="calibration",
        tokenizer_hash=tokenizer_manifest_sha256,
    )
    batch = AnchorDistortionBatch(
        family="mbpp",
        config="full",
        ruler_category=None,
        canonical_id="601",
        seed=None,
        configured_length=None,
        token_count=3,
        anchor_positions=(0, 1, 2),
        query_energy=values,
        q4_mse=values,
        q6_mse=values / 2,
        q8_mse=values / 4,
        sequence_token_ids=sequence_token_ids,
        identity_record=identity_record,
    )

    result = reduce_frozen_anchor_distortions(batch)
    assert result.source_shape == shape
    assert result.source_contract == FROZEN_SOURCE_TENSOR_CONTRACT
    assert result.row_count == FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    assert result.sequence_token_ids_sha256 == captured_record["sequence_token_ids_sha256"]
    assert result.identity_anchor_manifest_sha256 == captured_record["anchor_manifest_sha256"]
    assert result.identity_record_sha256 == captured_record["identity_record_sha256"]

    def with_identity_record(record: dict[str, object]) -> AnchorDistortionBatch:
        fields = {name: getattr(batch, name) for name in AnchorDistortionBatch.__dataclass_fields__}
        fields["identity_record"] = record
        return AnchorDistortionBatch(**fields)

    legacy_record = dict(identity_record)
    legacy_record.pop("fisher_boundary")
    with pytest.raises(ValueError, match="missing=.*fisher_boundary"):
        reduce_frozen_anchor_distortions(with_identity_record(legacy_record))

    tampered_record = json.loads(json.dumps(identity_record))
    tampered_boundary = tampered_record["fisher_boundary"]
    assert isinstance(tampered_boundary, dict)
    tampered_boundary["input_token_ids_sha256"] = "0" * 64
    tampered_boundary["fisher_boundary_sha256"] = capture.resolver.fisher_boundary_sha256(
        tampered_boundary
    )
    original_record_hash = tampered_record["identity_record_sha256"]
    tampered_record["identity_record_sha256"] = identity_record_sha256(tampered_record)
    assert tampered_record["identity_record_sha256"] != original_record_hash
    with pytest.raises(ValueError, match="input token-ID hash"):
        reduce_frozen_anchor_distortions(with_identity_record(tampered_record))

    with pytest.raises(ValueError, match="frozen source shape"):
        reduce_frozen_anchor_distortions(
            _batch(
                family="mbpp",
                config="full",
                canonical_id="wrong-frozen-shape",
                values=torch.ones(4),
            )
        )

    wrong_dtype_values = values.to(torch.float32)
    wrong_dtype = AnchorDistortionBatch(
        family=batch.family,
        config=batch.config,
        ruler_category=batch.ruler_category,
        canonical_id="wrong-frozen-dtype",
        seed=batch.seed,
        configured_length=batch.configured_length,
        token_count=batch.token_count,
        anchor_positions=batch.anchor_positions,
        query_energy=wrong_dtype_values,
        q4_mse=wrong_dtype_values,
        q6_mse=wrong_dtype_values,
        q8_mse=wrong_dtype_values,
    )
    with pytest.raises(TypeError, match="CPU torch.float64"):
        reduce_frozen_anchor_distortions(wrong_dtype)


def test_family_aggregation_uses_equal_broad_and_ruler_category_weights() -> None:
    sequences = _calibration_sequences()
    aggregate = aggregate_calibration_scores(sequences)
    base = torch.arange(1, 5, dtype=torch.float64)

    # MBPP mean=6, PG19 mean=15, RULER category macro=(6+9+12+15)/4=10.5.
    expected_q4 = ((6.0 + 15.0 + 10.5) / 3) * base
    assert torch.equal(aggregate.d4, expected_q4)
    assert torch.equal(aggregate.d6, expected_q4 / 2)
    assert torch.equal(aggregate.d8, expected_q4 / 4)
    assert aggregate.family_sequence_counts == (("mbpp", 2), ("pg19", 2), ("ruler", 8))
    assert aggregate.ruler_category_sequence_counts == (
        ("retrieval", 2),
        ("multi_hop_tracing", 2),
        ("aggregation", 2),
        ("question_answering", 2),
    )
    assert {
        sequence.config for sequence in sequences if sequence.ruler_category == "retrieval"
    } == {"niah_single_1", "niah_multikey_1"}


def test_comparator_aggregation_reuses_exact_family_macro_and_domain_hashes() -> None:
    sequences = _comparator_sequences(FROZEN_UNWEIGHTED_MSE_PROFILE)
    aggregate = aggregate_comparator_scores(sequences)
    expected = torch.full(
        (FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows,),
        10.5,
        dtype=torch.float64,
    )

    torch.testing.assert_close(aggregate.d4, expected)
    torch.testing.assert_close(aggregate.d6, expected / 2)
    torch.testing.assert_close(aggregate.d8, expected / 4)
    assert aggregate.family_sequence_counts == (("mbpp", 2), ("pg19", 2), ("ruler", 8))
    assert aggregate.ruler_category_sequence_counts == tuple(
        (category, 2) for category in calibration.RULER_CATEGORY_ORDER
    )
    reverse = aggregate_comparator_scores(list(reversed(sequences)))
    assert torch.equal(aggregate.d4, reverse.d4)
    assert aggregate.position_manifest_sha256 == reverse.position_manifest_sha256
    assert aggregate.sequence_score_manifest_sha256 == reverse.sequence_score_manifest_sha256
    assert aggregate.aggregate_scores_sha256 == reverse.aggregate_scores_sha256

    substituted = list(sequences)
    original = substituted[0]
    replacement = replace(original, endpoint_inputs_sha256="9" * 64)
    replacement = replace(
        replacement,
        sequence_scores_sha256=calibration._comparator_sequence_score_sha256(
            selector_profile=replacement.selector_profile,
            position_manifest_sha256=replacement.position_manifest_sha256,
            endpoint_inputs_sha256=replacement.endpoint_inputs_sha256,
            identity_record_sha256_value=replacement.identity_record_sha256,
            d4=replacement.d4,
            d6=replacement.d6,
            d8=replacement.d8,
        ),
    )
    substituted[0] = replacement
    changed = aggregate_comparator_scores(substituted)
    assert changed.sequence_score_manifest_sha256 != aggregate.sequence_score_manifest_sha256
    assert changed.aggregate_scores_sha256 != aggregate.aggregate_scores_sha256
    assert changed.position_manifest_sha256 == aggregate.position_manifest_sha256

    drifted_position = replace(original, endpoint_positions=(1, 2, 3))
    with pytest.raises(ValueError, match="positions drifted"):
        aggregate_comparator_scores([drifted_position, *sequences[1:]])


def test_aggregation_and_split_are_invariant_to_input_order() -> None:
    sequences = _calibration_sequences()
    forward = aggregate_calibration_scores(sequences)
    reverse = aggregate_calibration_scores(list(reversed(sequences)))
    assert torch.equal(forward.d4, reverse.d4)
    assert torch.equal(forward.d6, reverse.d6)
    assert torch.equal(forward.d8, reverse.d8)
    assert forward.sequence_score_manifest_sha256 == reverse.sequence_score_manifest_sha256

    split_forward = balanced_sha_rank_halves(sequences)
    split_reverse = balanced_sha_rank_halves(list(reversed(sequences)))
    assert split_forward.assignment_sha256 == split_reverse.assignment_sha256
    assert [item.canonical_dict() for item in split_forward.assignments] == [
        item.canonical_dict() for item in split_reverse.assignments
    ]
    assert {item.group for item in split_forward.assignments} == {
        "mbpp",
        "pg19",
        "ruler:retrieval",
        "ruler:multi_hop_tracing",
        "ruler:aggregation",
        "ruler:question_answering",
    }
    assert all(
        {item.half for item in split_forward.assignments if item.group == group} == {"a", "b"}
        for group in {item.group for item in split_forward.assignments}
    )
    resolver_bytes = (
        json.dumps(
            [item.canonical_dict() for item in split_forward.assignments],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert split_forward.assignment_sha256 == hashlib.sha256(resolver_bytes).hexdigest()
    retrieval_assignments = [
        item for item in split_forward.assignments if item.group == "ruler:retrieval"
    ]
    assert {item.config for item in retrieval_assignments} == {
        "niah_single_1",
        "niah_multikey_1",
    }
    assert {item.configured_length for item in retrieval_assignments} == {2_048, 4_096}
    assert all(
        item.sequence_length == 1 and item.seed is not None for item in retrieval_assignments
    )


def test_split_rank_hash_matches_the_domain_separated_identity_equation() -> None:
    sequence = _score(
        "ruler",
        "vt",
        "ruler-vt-2048-12339",
        1.0,
        ruler_category="multi_hop_tracing",
        seed=12_339,
        configured_length=2_048,
    )
    identity = "\0".join(
        (
            sequence.family,
            str(sequence.ruler_category),
            sequence.config,
            sequence.canonical_id,
            str(sequence.seed),
            str(sequence.configured_length),
            str(sequence.token_count),
        )
    )
    expected = hashlib.sha256(
        b"recurquant.experiment013.calibration-split.v1\0" + identity.encode()
    ).hexdigest()
    assert calibration_sequence_rank_sha256(sequence) == expected


def test_ruler_category_is_bound_into_anchor_score_and_split_hashes() -> None:
    retrieval = _score(
        "ruler",
        "shared-exact-config",
        "shared-id",
        1.0,
        ruler_category="retrieval",
        seed=12_339,
        configured_length=2_048,
    )
    aggregation = _score(
        "ruler",
        "shared-exact-config",
        "shared-id",
        1.0,
        ruler_category="aggregation",
        seed=12_339,
        configured_length=2_048,
    )
    other_configured_length = _score(
        "ruler",
        "shared-exact-config",
        "shared-id",
        1.0,
        ruler_category="retrieval",
        seed=12_339,
        configured_length=4_096,
    )

    assert retrieval.anchor_manifest_sha256 != aggregation.anchor_manifest_sha256
    assert retrieval.anchor_inputs_sha256 != aggregation.anchor_inputs_sha256
    assert retrieval.sequence_scores_sha256 != aggregation.sequence_scores_sha256
    assert calibration_sequence_rank_sha256(retrieval) != calibration_sequence_rank_sha256(
        aggregation
    )
    assert retrieval.sequence_scores_sha256 != other_configured_length.sequence_scores_sha256
    assert calibration_sequence_rank_sha256(retrieval) != calibration_sequence_rank_sha256(
        other_configured_length
    )


def test_split_half_fit_allocates_independently_and_passes_identical_halves() -> None:
    fit = fit_split_half_policy(
        _calibration_sequences(identical_pairs=True),
        layer_indices=TINY_GEOMETRY.layer_indices,
        rows_per_layer=TINY_GEOMETRY.rows_per_layer,
        marginal_steps=4,
    )

    assert torch.equal(fit.half_a_codes, fit.half_b_codes)
    assert int(fit.half_a_codes.to(torch.int64).sum().item()) == 4
    assert fit.stability.passed is True
    assert fit.stability.spearman_average_ties == pytest.approx(1.0)
    assert fit.stability.q8_jaccard == 1.0
    assert fit.stability.max_layer_mean_bitwidth_shift == 0.0


def test_frozen_split_half_artifact_recomputes_k29334_and_rejects_tampering() -> None:
    identity_file_sha256 = "1" * 64
    canonical_identity_sha256 = "2" * 64
    resolver_assignment_sha256 = "3" * 64
    raw = build_frozen_split_half_stability_artifact(
        _synthetic_frozen_half_aggregate("a" * 64),
        _synthetic_frozen_half_aggregate("b" * 64),
        identity_file_sha256=identity_file_sha256,
        canonical_identity_sha256=canonical_identity_sha256,
        resolver_assignment_sha256=resolver_assignment_sha256,
        full_sequence_score_manifest_sha256="c" * 64,
        full_calibration_scores_sha256="d" * 64,
    )

    decoded = deserialize_frozen_split_half_stability_artifact(
        raw,
        expected_identity_file_sha256=identity_file_sha256,
        expected_canonical_identity_sha256=canonical_identity_sha256,
        expected_resolver_assignment_sha256=resolver_assignment_sha256,
    )
    assert decoded.stability.passed is True
    assert decoded.stability.spearman_average_ties == pytest.approx(1.0)
    assert torch.equal(decoded.half_a_codes, decoded.half_b_codes)
    assert int(decoded.half_a_codes.to(torch.int64).sum().item()) == (
        FROZEN_STATIC_Q468_PRIMARY_STEPS
    )

    threshold_tamper = json.loads(raw)
    threshold_tamper["evidence"]["thresholds"]["minimum_spearman_average_ties"] = (0.69).hex()
    report = verify_frozen_split_half_stability_artifact(_rehashed_document(threshold_tamper))
    assert report["valid"] is False
    assert "thresholds drifted" in report["errors"][0]

    metric_tamper = json.loads(raw)
    metric_tamper["evidence"]["metrics"]["q8_jaccard"] = (0.5).hex()
    report = verify_frozen_split_half_stability_artifact(_rehashed_document(metric_tamper))
    assert report["valid"] is False
    assert "metrics or checks drifted" in report["errors"][0]

    map_tamper = json.loads(raw)
    encoded_codes = map_tamper["evidence"]["halves"][0]["code_map"]["codes_base64"]
    code_bytes = bytearray(base64.b64decode(encoded_codes))
    code_bytes[0] = (code_bytes[0] + 1) % 3
    map_tamper["evidence"]["halves"][0]["code_map"]["codes_base64"] = base64.b64encode(
        code_bytes
    ).decode("ascii")
    report = verify_frozen_split_half_stability_artifact(_rehashed_document(map_tamper))
    assert report["valid"] is False
    assert "differs from exact allocation" in report["errors"][0]

    identity_tamper = json.loads(raw)
    identity_tamper["evidence"]["identity"]["identity_file_sha256"] = "9" * 64
    report = verify_frozen_split_half_stability_artifact(
        _rehashed_document(identity_tamper),
        expected_identity_file_sha256=identity_file_sha256,
        expected_canonical_identity_sha256=canonical_identity_sha256,
        expected_resolver_assignment_sha256=resolver_assignment_sha256,
    )
    assert report["valid"] is False
    assert "differs from expected identity" in report["errors"][0]


def test_fast_allocation_matches_exhaustive_oracle_for_every_tiny_budget() -> None:
    aggregate = CalibrationAggregate(
        d4=torch.tensor([10.0, 8.0, 4.0, 2.0], dtype=torch.float64),
        d6=torch.tensor([4.0, 5.0, 3.0, 1.5], dtype=torch.float64),
        d8=torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float64),
        family_sequence_counts=(("mbpp", 2), ("pg19", 2), ("ruler", 8)),
        ruler_category_sequence_counts=(
            ("retrieval", 2),
            ("multi_hop_tracing", 2),
            ("aggregation", 2),
            ("question_answering", 2),
        ),
        sequence_score_manifest_sha256="a" * 64,
        source_contract=_tiny_aggregate().source_contract,
    )
    for marginal_steps in range(2 * aggregate.row_count + 1):
        actual = allocate_static_q468_code_map(
            aggregate,
            marginal_steps=marginal_steps,
        )
        expected = allocate_exact_multibit_codes(
            aggregate.d4.reshape(1, -1),
            aggregate.d6.reshape(1, -1),
            aggregate.d8.reshape(1, -1),
            marginal_steps=marginal_steps,
        ).reshape(-1)
        assert torch.equal(actual, expected)


def test_stability_metrics_define_ties_empty_q8_and_bitwidth_shift() -> None:
    tied = evaluate_policy_stability(
        torch.tensor([0, 0, 2], dtype=torch.uint8),
        torch.tensor([0, 2, 2], dtype=torch.uint8),
        layer_indices=(0,),
        rows_per_layer=3,
    )
    assert tied.spearman_average_ties == pytest.approx(0.5)
    assert tied.q8_jaccard == pytest.approx(0.5)
    assert tied.layer_mean_bitwidth_shifts == ((0, pytest.approx(4 / 3)),)
    assert tied.passed is False

    constant = evaluate_policy_stability(
        torch.zeros(4, dtype=torch.uint8),
        torch.zeros(4, dtype=torch.uint8),
        layer_indices=(0, 1),
        rows_per_layer=2,
        expected_marginal_steps=0,
    )
    assert constant.spearman_average_ties is None
    assert constant.q8_jaccard == 1.0
    assert constant.passed is False

    shifts = per_layer_mean_bitwidth_shifts(
        torch.tensor([2, 0, 1, 1], dtype=torch.uint8),
        torch.tensor([1, 0, 1, 1], dtype=torch.uint8),
        layer_indices=(0, 1),
        rows_per_layer=2,
    )
    assert shifts == ((0, 1.0), (1, 0.0))
    assert (
        q8_set_jaccard(torch.zeros(2, dtype=torch.uint8), torch.zeros(2, dtype=torch.uint8)) == 1.0
    )


def test_stability_gate_enforces_exact_budget_and_rejects_bad_codes() -> None:
    with pytest.raises(ValueError, match="half B"):
        evaluate_policy_stability(
            torch.tensor([2, 1, 0, 1], dtype=torch.uint8),
            torch.tensor([2, 1, 0, 0], dtype=torch.uint8),
            layer_indices=(0, 1),
            rows_per_layer=2,
            expected_marginal_steps=4,
        )
    with pytest.raises(ValueError, match="0, 1, and 2"):
        evaluate_policy_stability(
            torch.tensor([3, 0], dtype=torch.uint8),
            torch.tensor([2, 1], dtype=torch.uint8),
            layer_indices=(0,),
            rows_per_layer=2,
        )


def test_canonical_score_artifact_round_trips_and_matches_policy_hashes() -> None:
    aggregate = _tiny_aggregate()
    first = build_calibration_score_artifact(
        aggregate,
        geometry=TINY_GEOMETRY,
        calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        marginal_steps=[4, 2],
    )
    second = build_calibration_score_artifact(
        aggregate,
        geometry=TINY_GEOMETRY,
        calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        marginal_steps=[2, 4],
    )
    assert first == second
    document = json.loads(first)
    evidence = document["evidence"]
    assert document["artifact_kind"] == GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND
    assert evidence["artifact_profile"] == GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE
    assert evidence["artifact_revision"] == GENERIC_CALIBRATION_SCORE_ARTIFACT_REVISION
    assert "ruler_category_sequence_counts" in evidence
    assert "ruler_subfamily_sequence_counts" not in evidence
    assert evidence["source_tensor_contract"] == aggregate.source_contract.canonical_dict()

    decoded = deserialize_calibration_score_artifact(
        first,
        expected_file_sha256=hashlib.sha256(first).hexdigest(),
    )
    report = verify_calibration_score_artifact(first)
    assert report["valid"] is True
    assert report["errors"] == []
    assert decoded.geometry == TINY_GEOMETRY
    assert decoded.artifact_kind == GENERIC_CALIBRATION_SCORE_ARTIFACT_KIND
    assert decoded.artifact_profile == GENERIC_CALIBRATION_SCORE_ARTIFACT_PROFILE
    assert decoded.aggregate.sequence_score_manifest_sha256 == (
        aggregate.sequence_score_manifest_sha256
    )
    assert [steps for steps, _codes, _digest in decoded.allocations] == [2, 4]

    for steps, codes, digest in decoded.allocations:
        policy = build_static_rht_q468_policy(
            aggregate.d4,
            aggregate.d6,
            aggregate.d8,
            geometry=TINY_GEOMETRY,
            marginal_steps=steps,
            calibration_manifest_sha256=aggregate.sequence_score_manifest_sha256,
            identity_artifact_sha256=FAKE_IDENTITY_SHA256,
            tokenizer_manifest_sha256="b" * 64,
            source_commit="c" * 40,
        )
        assert digest == policy.code_map_sha256
        assert digest == static_q468_code_map_sha256(
            codes,
            geometry=TINY_GEOMETRY,
            marginal_steps=steps,
        )
        assert decoded.calibration_scores_sha256 == policy.calibration_scores_sha256


def test_official_frozen_artifact_round_trips_only_with_exact_profile() -> None:
    raw = build_frozen_calibration_score_artifact(
        _synthetic_frozen_aggregate(),
        calibration_identity_sha256=FAKE_IDENTITY_SHA256,
    )

    decoded = deserialize_calibration_score_artifact(raw)
    document = json.loads(raw)

    assert decoded.artifact_kind == CALIBRATION_SCORE_ARTIFACT_KIND
    assert decoded.artifact_profile == CALIBRATION_SCORE_ARTIFACT_PROFILE
    assert decoded.artifact_revision == CALIBRATION_SCORE_ARTIFACT_REVISION
    assert decoded.geometry == FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    assert decoded.aggregate.source_contract == FROZEN_SOURCE_TENSOR_CONTRACT
    assert [steps for steps, _codes, _digest in decoded.allocations] == [
        FROZEN_STATIC_Q468_ABLATION_STEPS,
        FROZEN_STATIC_Q468_PRIMARY_STEPS,
    ]
    assert document["artifact_kind"] == CALIBRATION_SCORE_ARTIFACT_KIND
    assert verify_calibration_score_artifact(raw)["valid"] is True


def test_comparator_score_artifact_round_trips_both_exact_k29334_profiles() -> None:
    identity_sha256 = "1" * 64
    raw = build_frozen_comparator_score_artifact(
        _synthetic_comparator_aggregate(FROZEN_UNWEIGHTED_MSE_PROFILE),
        _synthetic_comparator_aggregate(FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE),
        calibration_identity_sha256=identity_sha256,
    )
    decoded = deserialize_comparator_score_artifact(
        raw,
        expected_calibration_identity_sha256=identity_sha256,
    )
    document = json.loads(raw)

    assert document["artifact_kind"] == COMPARATOR_SCORE_ARTIFACT_KIND
    assert document["schema_version"] == 1
    assert document["evidence"]["artifact_profile"] == COMPARATOR_SCORE_ARTIFACT_PROFILE
    assert document["evidence"]["artifact_revision"] == COMPARATOR_SCORE_ARTIFACT_REVISION
    assert "target_nll" not in raw.decode("utf-8")
    assert "token_ids" not in raw.decode("utf-8")
    assert list(decoded.selectors) == [
        FROZEN_UNWEIGHTED_MSE_PROFILE,
        FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
    ]
    assert decoded.calibration_identity_sha256 == identity_sha256
    assert decoded.file_sha256 == hashlib.sha256(raw).hexdigest()
    for method, selector in decoded.selectors.items():
        assert selector.method_id == method
        assert selector.marginal_steps == FROZEN_STATIC_Q468_PRIMARY_STEPS
        assert selector.precision_codes.dtype == torch.uint8
        assert selector.precision_codes.device.type == "cpu"
        assert int(selector.precision_codes.to(torch.int64).sum().item()) == (
            FROZEN_STATIC_Q468_PRIMARY_STEPS
        )
        assert selector.calibration_scores_sha256 == (selector.aggregate.aggregate_scores_sha256)
        assert selector.position_manifest_sha256 == (selector.aggregate.position_manifest_sha256)
    assert (
        verify_comparator_score_artifact(
            raw,
            expected_calibration_identity_sha256=identity_sha256,
        )["valid"]
        is True
    )


def test_comparator_score_artifact_rejects_profile_hash_array_and_allocation_tampering() -> None:
    raw = build_frozen_comparator_score_artifact(
        _synthetic_comparator_aggregate(FROZEN_UNWEIGHTED_MSE_PROFILE),
        _synthetic_comparator_aggregate(FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE),
        calibration_identity_sha256="1" * 64,
    )

    missing = json.loads(raw)
    missing["evidence"]["selectors"].pop()
    with pytest.raises(ValueError, match="exactly two selectors"):
        deserialize_comparator_score_artifact(_rehashed_document(missing))

    extra = json.loads(raw)
    extra["evidence"]["selectors"].append(extra["evidence"]["selectors"][0])
    with pytest.raises(ValueError, match="exactly two selectors"):
        deserialize_comparator_score_artifact(_rehashed_document(extra))

    reordered = json.loads(raw)
    reordered["evidence"]["selectors"].reverse()
    with pytest.raises(ValueError, match="exactly MSE then"):
        deserialize_comparator_score_artifact(_rehashed_document(reordered))

    score_tamper = json.loads(raw)
    encoded = score_tamper["evidence"]["selectors"][0]["scores"]["data_base64"]
    score_bytes = bytearray(base64.b64decode(encoded))
    score_bytes[0] ^= 1
    score_tamper["evidence"]["selectors"][0]["scores"]["data_base64"] = base64.b64encode(
        score_bytes
    ).decode("ascii")
    with pytest.raises(ValueError, match="aggregate-score SHA-256 drifted"):
        deserialize_comparator_score_artifact(_rehashed_document(score_tamper))

    nonfinite = json.loads(raw)
    encoded = nonfinite["evidence"]["selectors"][0]["scores"]["data_base64"]
    score_bytes = bytearray(base64.b64decode(encoded))
    score_bytes[:8] = torch.tensor(float("nan"), dtype=torch.float64).numpy().tobytes()
    nonfinite["evidence"]["selectors"][0]["scores"]["data_base64"] = base64.b64encode(
        score_bytes
    ).decode("ascii")
    with pytest.raises(ValueError, match="finite and non-negative"):
        deserialize_comparator_score_artifact(_rehashed_document(nonfinite))

    position_tamper = json.loads(raw)
    position_tamper["evidence"]["selectors"][0]["position_manifest_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="aggregate-score SHA-256 drifted"):
        deserialize_comparator_score_artifact(_rehashed_document(position_tamper))

    allocation_tamper = json.loads(raw)
    allocation_tamper["evidence"]["selectors"][0]["allocation"]["code_map_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="code-map SHA-256 drifted"):
        deserialize_comparator_score_artifact(_rehashed_document(allocation_tamper))

    with pytest.raises(ValueError, match="differs from expected identity"):
        deserialize_comparator_score_artifact(
            raw,
            expected_calibration_identity_sha256="2" * 64,
        )


def test_generic_artifact_cannot_be_relabelled_as_official_geometry_counts_or_budget() -> None:
    raw = build_calibration_score_artifact(
        _tiny_aggregate(),
        geometry=TINY_GEOMETRY,
        calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        marginal_steps=[2, 4],
    )

    tiny_geometry = _relabel_generic_document_as_official(raw)
    report = verify_calibration_score_artifact(_rehashed_document(tiny_geometry))
    assert report["valid"] is False
    assert "exact frozen geometry" in report["errors"][0]

    bad_counts = _relabel_generic_document_as_official(raw)
    evidence = bad_counts["evidence"]
    assert isinstance(evidence, dict)
    evidence["geometry"] = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.canonical_dict()
    evidence["geometry_sha256"] = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.geometry_sha256
    report = verify_calibration_score_artifact(_rehashed_document(bad_counts))
    assert report["valid"] is False
    assert "MBPP=128, PG19=16, RULER=16" in report["errors"][0]

    bad_budget = _relabel_generic_document_as_official(raw)
    evidence = bad_budget["evidence"]
    assert isinstance(evidence, dict)
    evidence["geometry"] = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.canonical_dict()
    evidence["geometry_sha256"] = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.geometry_sha256
    evidence["family_sequence_counts"] = [
        {"count": 128, "family": "mbpp"},
        {"count": 16, "family": "pg19"},
        {"count": 16, "family": "ruler"},
    ]
    evidence["ruler_category_sequence_counts"] = [
        {"category": category, "count": 4}
        for category in (
            "retrieval",
            "multi_hop_tracing",
            "aggregation",
            "question_answering",
        )
    ]
    evidence["source_tensor_contract"] = FROZEN_SOURCE_TENSOR_CONTRACT.canonical_dict()
    report = verify_calibration_score_artifact(_rehashed_document(bad_budget))
    assert report["valid"] is False
    assert "exactly K27030 and K29334" in report["errors"][0]


def test_artifact_validation_rejects_noncanonical_malformed_and_tampered_data() -> None:
    raw = build_calibration_score_artifact(
        _tiny_aggregate(),
        geometry=TINY_GEOMETRY,
        calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        marginal_steps=[2, 4],
    )

    assert verify_calibration_score_artifact(raw + b" ")["valid"] is False
    duplicate = raw.replace(
        b'{\n  "artifact_kind"',
        b'{\n  "schema_version": 1,\n  "artifact_kind"',
        1,
    )
    duplicate_report = verify_calibration_score_artifact(duplicate)
    assert duplicate_report["valid"] is False
    assert "duplicate JSON object key" in duplicate_report["errors"][0]

    document = json.loads(raw)
    evidence = document["evidence"]
    score_record = evidence["scores"]
    score_bytes = bytearray(base64.b64decode(score_record["data_base64"]))
    score_bytes[0] ^= 1
    score_record["data_base64"] = base64.b64encode(score_bytes).decode()
    score_tamper = verify_calibration_score_artifact(_rehashed_document(document))
    assert score_tamper["valid"] is False
    assert "calibration score SHA-256 mismatch" in score_tamper["errors"][0]

    document = json.loads(raw)
    evidence = document["evidence"]
    evidence["allocations"][0]["code_map_sha256"] = "0" * 64
    code_tamper = verify_calibration_score_artifact(_rehashed_document(document))
    assert code_tamper["valid"] is False
    assert "code-map SHA-256" in code_tamper["errors"][0]

    document = json.loads(raw)
    evidence = document["evidence"]
    score_record = evidence["scores"]
    score_bytes = bytearray(base64.b64decode(score_record["data_base64"]))
    score_bytes[:8] = torch.tensor(float("nan"), dtype=torch.float64).numpy().tobytes()
    score_record["data_base64"] = base64.b64encode(score_bytes).decode()
    nonfinite = verify_calibration_score_artifact(_rehashed_document(document))
    assert nonfinite["valid"] is False
    assert "finite and non-negative" in nonfinite["errors"][0]

    wrong_file = verify_calibration_score_artifact(raw, expected_file_sha256="0" * 64)
    assert wrong_file["valid"] is False
    assert "file SHA-256 mismatch" in wrong_file["errors"][0]


def test_artifact_builder_rejects_nonfinite_aggregate_and_bad_identity_hash() -> None:
    aggregate = _tiny_aggregate()
    broken = CalibrationAggregate(
        d4=aggregate.d4.clone(),
        d6=aggregate.d6.clone(),
        d8=aggregate.d8.clone(),
        family_sequence_counts=aggregate.family_sequence_counts,
        ruler_category_sequence_counts=aggregate.ruler_category_sequence_counts,
        sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
        source_contract=aggregate.source_contract,
    )
    broken.d4[0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        build_calibration_score_artifact(
            broken,
            geometry=TINY_GEOMETRY,
            calibration_identity_sha256=FAKE_IDENTITY_SHA256,
            marginal_steps=[2],
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_calibration_score_artifact(
            aggregate,
            geometry=TINY_GEOMETRY,
            calibration_identity_sha256="not-a-hash",
            marginal_steps=[2],
        )


def test_artifact_builder_rejects_inconsistent_and_nonfrozen_sequence_counts() -> None:
    aggregate = _tiny_aggregate()
    inconsistent = CalibrationAggregate(
        d4=aggregate.d4,
        d6=aggregate.d6,
        d8=aggregate.d8,
        family_sequence_counts=(("mbpp", 2), ("pg19", 2), ("ruler", 7)),
        ruler_category_sequence_counts=aggregate.ruler_category_sequence_counts,
        sequence_score_manifest_sha256=aggregate.sequence_score_manifest_sha256,
        source_contract=aggregate.source_contract,
    )
    with pytest.raises(ValueError, match="must equal"):
        build_calibration_score_artifact(
            inconsistent,
            geometry=TINY_GEOMETRY,
            calibration_identity_sha256=FAKE_IDENTITY_SHA256,
            marginal_steps=[2],
        )
    with pytest.raises(ValueError, match="MBPP=128"):
        build_frozen_calibration_score_artifact(
            aggregate,
            calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        )

    frozen = _synthetic_frozen_aggregate()
    wrong_source = CalibrationAggregate(
        d4=frozen.d4,
        d6=frozen.d6,
        d8=frozen.d8,
        family_sequence_counts=frozen.family_sequence_counts,
        ruler_category_sequence_counts=frozen.ruler_category_sequence_counts,
        sequence_score_manifest_sha256=frozen.sequence_score_manifest_sha256,
        source_contract=aggregate.source_contract,
    )
    with pytest.raises(ValueError, match="exact Experiment 013 source contract"):
        build_frozen_calibration_score_artifact(
            wrong_source,
            calibration_identity_sha256=FAKE_IDENTITY_SHA256,
        )


def test_real_geometry_allocates_both_frozen_exact_k_maps_without_large_data() -> None:
    geometry = FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    rows = geometry.total_rows
    row_axis = torch.arange(rows, dtype=torch.float64)
    aggregate = CalibrationAggregate(
        d4=4.0 + row_axis / rows,
        d6=2.0 + row_axis / (2 * rows),
        d8=1.0 + row_axis / (4 * rows),
        family_sequence_counts=(("mbpp", 128), ("pg19", 16), ("ruler", 16)),
        ruler_category_sequence_counts=(
            ("retrieval", 4),
            ("multi_hop_tracing", 4),
            ("aggregation", 4),
            ("question_answering", 4),
        ),
        sequence_score_manifest_sha256="d" * 64,
        source_contract=FROZEN_SOURCE_TENSOR_CONTRACT,
    )

    maps = allocate_frozen_static_q468_code_maps(aggregate)

    assert set(maps) == {
        FROZEN_STATIC_Q468_ABLATION_STEPS,
        FROZEN_STATIC_Q468_PRIMARY_STEPS,
    }
    for steps, codes in maps.items():
        assert codes.shape == (rows,)
        assert codes.dtype == torch.uint8
        assert int(codes.to(torch.int64).sum().item()) == steps
        assert sum(int((codes == code).sum().item()) for code in range(3)) == rows
