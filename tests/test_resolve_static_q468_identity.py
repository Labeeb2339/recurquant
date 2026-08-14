from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from recurquant import static_q468
from recurquant import static_q468_calibration as calibration

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "resolve_static_q468_identity.py"
SPEC = importlib.util.spec_from_file_location("resolve_static_q468_identity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)

REVISIONS = {
    "mbpp": resolver.MBPP_REVISION,
    "pg19": resolver.PG19_REVISION,
    "ruler": resolver.RULER_REVISION,
    "humaneval_plus": resolver.HUMANEVAL_PLUS_REVISION,
}
FIXTURE_BINDING_ARTIFACT = b"verified-stage-a-binding-fixture"
FIXTURE_BINDING = {
    "calibration_identity_file_sha256": resolver.sha256_bytes(b"calibration-file"),
    "calibration_score_artifact_file_sha256": resolver.sha256_bytes(b"calibration-scores"),
    "comparator_score_artifact_file_sha256": resolver.sha256_bytes(b"comparator-scores"),
    "split_half_stability_artifact_file_sha256": resolver.sha256_bytes(b"split-half"),
    "static_fisher_k29334_policy_file_sha256": resolver.sha256_bytes(b"fisher-k29334-policy"),
    "static_k27030_policy_file_sha256": resolver.sha256_bytes(b"k27030-policy"),
    "static_k29334_policy_file_sha256": resolver.sha256_bytes(b"k29334-policy"),
    "static_mse_k29334_policy_file_sha256": resolver.sha256_bytes(b"mse-k29334-policy"),
}
FIXTURE_EXECUTION_BINDINGS = {
    "repository_source_manifest_file_sha256": resolver.sha256_bytes(b"source-manifest"),
    "calibration_runtime_manifest_file_sha256": resolver.sha256_bytes(b"runtime-manifest"),
    "model_file_manifest_file_sha256": resolver.sha256_bytes(b"model-manifest"),
    "parquet_materialization_manifest_file_sha256": (
        resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    ),
}


def _hash(label: str) -> str:
    return resolver.sha256_bytes(label.encode())


def _exact_code_vector(steps: int, *, from_end: bool = False) -> torch.Tensor:
    rows = static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    codes = torch.zeros(rows, dtype=torch.uint8)
    if from_end:
        codes[-steps:] = 1
    else:
        codes[:steps] = 1
    return codes


def _fake_policy(
    *,
    method_id: str,
    marginal_steps: int,
    codes: torch.Tensor,
    identity_sha256: str,
    tokenizer_manifest_sha256: str,
    calibration_manifest_sha256: str,
    calibration_scores_sha256: str,
    source_commit: str,
) -> SimpleNamespace:
    geometry = static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    code_map_sha256 = calibration.static_q468_code_map_sha256(
        codes,
        geometry=geometry,
        marginal_steps=marginal_steps,
    )
    return SimpleNamespace(
        method_id=method_id,
        marginal_steps=marginal_steps,
        geometry=geometry,
        model_id=static_q468.PRIMARY_MODEL_ID,
        model_revision=static_q468.PRIMARY_MODEL_REVISION,
        tokenizer_id=static_q468.PRIMARY_TOKENIZER_ID,
        tokenizer_revision=static_q468.PRIMARY_TOKENIZER_REVISION,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        transformers_version=static_q468.FROZEN_TRANSFORMERS_VERSION,
        identity_artifact_sha256=identity_sha256,
        source_commit=source_commit,
        calibration_manifest_sha256=calibration_manifest_sha256,
        calibration_scores_sha256=calibration_scores_sha256,
        code_map_sha256=code_map_sha256,
        precision_codes=lambda: codes.reshape(
            geometry.layers,
            geometry.heads,
            geometry.key_rows,
        ).clone(),
    )


@contextmanager
def _binding_v3_fixture() -> Iterator[SimpleNamespace]:
    geometry = static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY
    identity_bytes = b"frozen-identity-v5"
    score_bytes = b"candidate-calibration-scores"
    split_bytes = b"split-half-stability"
    comparator_bytes = b"combined-comparator-scores"
    policy_bytes = {
        static_q468.STATIC_Q468_ABLATION_METHOD: b"static-k27030-policy",
        static_q468.STATIC_Q468_PRIMARY_METHOD: b"static-k29334-policy",
        static_q468.STATIC_Q468_MSE_METHOD: b"static-mse-k29334-policy",
        static_q468.STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD: (b"static-fisher-k29334-policy"),
    }
    dependencies = {
        "frozen_identity_artifact": identity_bytes,
        "calibration_score_artifact": score_bytes,
        "split_half_stability_artifact": split_bytes,
        "static_k27030_policy_artifact": policy_bytes[static_q468.STATIC_Q468_ABLATION_METHOD],
        "static_k29334_policy_artifact": policy_bytes[static_q468.STATIC_Q468_PRIMARY_METHOD],
        "comparator_score_artifact": comparator_bytes,
        "static_fisher_k29334_policy_artifact": policy_bytes[
            static_q468.STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD
        ],
        "static_mse_k29334_policy_artifact": policy_bytes[static_q468.STATIC_Q468_MSE_METHOD],
    }
    identity_sha256 = resolver.sha256_bytes(identity_bytes)
    tokenizer_manifest_sha256 = _hash("binding-tokenizer-manifest")
    identity_record_manifest_sha256 = _hash("complete-identity-v5-record-manifest")
    source_commit_h0 = "a" * 40
    candidate_sequence_manifest_sha256 = _hash("candidate-sequence-manifest")
    candidate_score_sha256 = _hash("candidate-raw-distortion-scores")
    k27030_codes = _exact_code_vector(static_q468.FROZEN_STATIC_Q468_ABLATION_STEPS)
    k29334_codes = _exact_code_vector(static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS)

    identity = SimpleNamespace(
        file_sha256=identity_sha256,
        canonical_evidence_sha256=_hash("identity-canonical-evidence"),
        records=({},),
        assignment=(),
        assignment_sha256=_hash("identity-assignment"),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
    )
    candidate_scores = SimpleNamespace(
        artifact_kind=calibration.CALIBRATION_SCORE_ARTIFACT_KIND,
        file_sha256=resolver.sha256_bytes(score_bytes),
        calibration_identity_sha256=identity_sha256,
        calibration_scores_sha256=candidate_score_sha256,
        aggregate=SimpleNamespace(
            identity_record_manifest_sha256=identity_record_manifest_sha256,
            sequence_score_manifest_sha256=candidate_sequence_manifest_sha256,
        ),
        allocations=(
            (
                static_q468.FROZEN_STATIC_Q468_ABLATION_STEPS,
                k27030_codes,
                calibration.static_q468_code_map_sha256(
                    k27030_codes,
                    geometry=geometry,
                    marginal_steps=static_q468.FROZEN_STATIC_Q468_ABLATION_STEPS,
                ),
            ),
            (
                static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
                k29334_codes,
                calibration.static_q468_code_map_sha256(
                    k29334_codes,
                    geometry=geometry,
                    marginal_steps=static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
                ),
            ),
        ),
    )
    split = SimpleNamespace(
        file_sha256=resolver.sha256_bytes(split_bytes),
        identity_file_sha256=identity_sha256,
        canonical_identity_sha256=identity.canonical_evidence_sha256,
        resolver_assignment_sha256=identity.assignment_sha256,
        full_sequence_score_manifest_sha256=candidate_sequence_manifest_sha256,
        full_calibration_scores_sha256=candidate_score_sha256,
        half_a_aggregate=SimpleNamespace(
            identity_record_manifest_sha256=_hash("half-a-record-manifest")
        ),
        half_b_aggregate=SimpleNamespace(
            identity_record_manifest_sha256=_hash("half-b-record-manifest")
        ),
    )

    row = torch.arange(geometry.total_rows, dtype=torch.float64)
    selector_specs = (
        (
            calibration.FROZEN_UNWEIGHTED_MSE_PROFILE,
            (
                ((row + 1) % 997) / 997,
                ((row + 5) % 991) / 991,
                ((row + 9) % 983) / 983,
            ),
            False,
        ),
        (
            calibration.FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            (
                ((row + 13) % 977) / 977,
                ((row + 17) % 971) / 971,
                ((row + 21) % 967) / 967,
            ),
            True,
        ),
    )
    selectors: dict[str, SimpleNamespace] = {}
    policies: dict[str, SimpleNamespace] = {
        static_q468.STATIC_Q468_ABLATION_METHOD: _fake_policy(
            method_id=static_q468.STATIC_Q468_ABLATION_METHOD,
            marginal_steps=static_q468.FROZEN_STATIC_Q468_ABLATION_STEPS,
            codes=k27030_codes,
            identity_sha256=identity_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            calibration_manifest_sha256=candidate_sequence_manifest_sha256,
            calibration_scores_sha256=candidate_score_sha256,
            source_commit=source_commit_h0,
        ),
        static_q468.STATIC_Q468_PRIMARY_METHOD: _fake_policy(
            method_id=static_q468.STATIC_Q468_PRIMARY_METHOD,
            marginal_steps=static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
            codes=k29334_codes,
            identity_sha256=identity_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            calibration_manifest_sha256=candidate_sequence_manifest_sha256,
            calibration_scores_sha256=candidate_score_sha256,
            source_commit=source_commit_h0,
        ),
    }
    for method_id, scores, reverse_codes in selector_specs:
        codes = _exact_code_vector(
            static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
            from_end=reverse_codes,
        )
        aggregate_score_sha256 = _hash(f"{method_id}-aggregate-evidence")
        aggregate = SimpleNamespace(
            identity_record_manifest_sha256=identity_record_manifest_sha256,
            sequence_score_manifest_sha256=_hash(f"{method_id}-sequence-manifest"),
            aggregate_scores_sha256=aggregate_score_sha256,
            scores=lambda values=scores: values,
        )
        selector = SimpleNamespace(
            method_id=method_id,
            aggregate=aggregate,
            calibration_scores_sha256=aggregate_score_sha256,
            marginal_steps=static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
            precision_codes=codes,
            code_map_sha256=calibration.static_q468_code_map_sha256(
                codes,
                geometry=geometry,
                marginal_steps=static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
            ),
        )
        selectors[method_id] = selector
        policies[method_id] = _fake_policy(
            method_id=method_id,
            marginal_steps=static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS,
            codes=codes,
            identity_sha256=identity_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            calibration_manifest_sha256=aggregate.sequence_score_manifest_sha256,
            calibration_scores_sha256=static_q468.static_q468_distortion_sha256(
                *scores,
                geometry=geometry,
            ),
            source_commit=source_commit_h0,
        )
    comparator_scores = SimpleNamespace(
        selectors=selectors,
        calibration_identity_sha256=identity_sha256,
        file_sha256=resolver.sha256_bytes(comparator_bytes),
    )
    policies_by_bytes = {policy_bytes[method_id]: policy for method_id, policy in policies.items()}

    def deserialize_policy(data: bytes) -> SimpleNamespace:
        try:
            return policies_by_bytes[data]
        except KeyError as error:
            raise ValueError("unknown policy fixture bytes") from error

    def rebuild_policy(*_scores: torch.Tensor, method_id: str, **_kwargs: object) -> object:
        return policies[method_id]

    def serialize_policy(policy: SimpleNamespace) -> bytes:
        return policy_bytes[policy.method_id]

    state = SimpleNamespace(
        dependencies=dependencies,
        identity=identity,
        candidate_scores=candidate_scores,
        split=split,
        comparator_scores=comparator_scores,
        policies=policies,
        policy_bytes=policy_bytes,
        identity_record_manifest_sha256=identity_record_manifest_sha256,
    )
    with (
        patch.object(
            resolver,
            "deserialize_frozen_calibration_identity_artifact",
            return_value=identity,
        ),
        patch.object(
            resolver,
            "_identity_half_record_manifests",
            return_value={
                "a": split.half_a_aggregate.identity_record_manifest_sha256,
                "b": split.half_b_aggregate.identity_record_manifest_sha256,
            },
        ),
        patch.object(
            calibration,
            "calibration_identity_record_manifest_sha256",
            return_value=identity_record_manifest_sha256,
        ),
        patch.object(
            calibration,
            "deserialize_calibration_score_artifact",
            return_value=candidate_scores,
        ),
        patch.object(
            calibration,
            "deserialize_frozen_split_half_stability_artifact",
            return_value=split,
        ),
        patch.object(
            calibration,
            "deserialize_comparator_score_artifact",
            return_value=comparator_scores,
        ),
        patch.object(
            static_q468,
            "deserialize_static_rht_q468_policy",
            side_effect=deserialize_policy,
        ),
        patch.object(
            static_q468,
            "build_static_rht_q468_policy",
            side_effect=rebuild_policy,
        ),
        patch.object(
            static_q468,
            "serialize_static_rht_q468_policy",
            side_effect=serialize_policy,
        ),
    ):
        yield state


def _reauthenticated_binding_bytes(document: dict[str, Any]) -> bytes:
    document["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(document["evidence"])
    )
    return resolver.canonical_json_bytes(document)


def _datasets() -> list[dict[str, Any]]:
    return [
        {
            "key": "mbpp",
            "dataset_id": resolver.MBPP_DATASET_ID,
            "config": resolver.MBPP_CONFIG,
            "revision": REVISIONS["mbpp"],
            "split": "train",
            "canonical_id_field": "task_id",
            "canonical_id_manifest_sha256": resolver.mbpp_calibration_identity()[1],
            "formatter_id": resolver.FROZEN_FORMATTER_IDS["mbpp"],
            "formatter_sha256": resolver.FROZEN_STATIC_FORMATTER_SHA256["mbpp"],
        },
        {
            "key": "pg19",
            "dataset_id": resolver.PG19_DATASET_ID,
            "config": "default",
            "revision": REVISIONS["pg19"],
            "split": "validation",
            "canonical_id_field": "url",
            "canonical_id_manifest_sha256": _hash("pg19-id-manifest"),
            "formatter_id": resolver.FROZEN_FORMATTER_IDS["pg19"],
            "formatter_sha256": resolver.FROZEN_STATIC_FORMATTER_SHA256["pg19"],
        },
        {
            "key": "ruler",
            "dataset_id": resolver.RULER_SOURCE_ID,
            "config": "official-generator",
            "revision": REVISIONS["ruler"],
            "split": "generated",
            "canonical_id_field": "configuration_id",
            "canonical_id_manifest_sha256": _hash("ruler-id-manifest"),
            "formatter_id": resolver.FROZEN_FORMATTER_IDS["ruler"],
            "formatter_sha256": _hash("ruler-formatter"),
        },
        {
            "key": "humaneval_plus",
            "dataset_id": resolver.HUMANEVAL_PLUS_DATASET_ID,
            "config": "default",
            "revision": REVISIONS["humaneval_plus"],
            "split": "test",
            "canonical_id_field": "task_id",
            "canonical_id_manifest_sha256": _hash("humaneval-id-manifest"),
            "formatter_id": resolver.FROZEN_FORMATTER_IDS["humaneval_plus"],
            "formatter_sha256": resolver.FROZEN_STATIC_FORMATTER_SHA256["humaneval_plus"],
        },
    ]


def _tokenizer() -> dict[str, Any]:
    return {
        "source_id": resolver.PRIMARY_MODEL_ID,
        "revision": resolver.PRIMARY_MODEL_REVISION,
        "class": "Qwen2Tokenizer",
        "transformers_version": resolver.TRANSFORMERS_VERSION,
        "files": [
            {"name": "tokenizer.json", "sha256": _hash("tokenizer"), "size_bytes": 100},
            {
                "name": "tokenizer_config.json",
                "sha256": _hash("tokenizer-config"),
                "size_bytes": 20,
            },
        ],
    }


def _tokenizer_manifest_hash() -> str:
    files = sorted(_tokenizer()["files"], key=lambda item: item["name"])
    return resolver.sha256_bytes(resolver.canonical_json_bytes(files))


def _record(
    *,
    family: str,
    canonical_id: str,
    config: str,
    rank: int,
    seed: int | None,
    sequence_length: int,
    prefill_stop: int,
    scored_stop: int,
    configured_length: int | None = None,
    ruler_category: str | None = None,
) -> dict[str, Any]:
    namespace = {
        "pg19": resolver.PG19_VALIDATION_NAMESPACE,
        "ruler": resolver.RULER_STAGE_A_SELECTION_NAMESPACE,
        "humaneval_plus": resolver.HUMANEVAL_AB_NAMESPACE,
    }[family]
    label = f"{family}-{canonical_id}-{config}-{seed}-{sequence_length}"
    sequence_hash = _hash(f"sequence-tokens-{label}")
    sequence_token_ids = tuple(range(sequence_length))
    token_span = {
        "prefill_start": 0,
        "prefill_stop": prefill_stop,
        "scored_start": prefill_stop,
        "scored_stop": scored_stop,
        "cache_exposed_start": prefill_stop + 1,
        "cache_exposed_stop": scored_stop,
    }
    record = {
        "family": family,
        "canonical_id": canonical_id,
        "config": config,
        "selection_rank": rank,
        "selection_sha256": resolver.selection_sha256(namespace, canonical_id),
        "seed": seed,
        "configured_length": configured_length,
        "sequence_length": sequence_length,
        "ruler_category": ruler_category,
        "generator_receipt_sha256": (
            _hash(f"generator-receipt-{label}") if family == "ruler" else None
        ),
        "source_content_sha256": _hash(f"source-{label}"),
        "formatted_content_sha256": _hash(f"formatted-{label}"),
        "prompt_token_ids_sha256": _hash(f"prompt-tokens-{label}"),
        "target_token_ids_sha256": _hash(f"target-tokens-{label}"),
        "sequence_token_ids_sha256": sequence_hash,
        "tokenizer_manifest_sha256": _tokenizer_manifest_hash(),
        "token_span": token_span,
        "anchor_manifest_sha256": resolver.identity_anchor_manifest_sha256(
            canonical_id=canonical_id,
            sequence_length=sequence_length,
            sequence_token_ids_sha256_value=sequence_hash,
            token_span=token_span,
        ),
        "fisher_boundary": resolver.build_fisher_boundary_contract(sequence_token_ids),
    }
    record["identity_record_sha256"] = resolver.identity_record_sha256(record)
    return record


def _refresh_record_lineage(record: dict[str, Any]) -> None:
    record["anchor_manifest_sha256"] = resolver.identity_anchor_manifest_sha256(
        canonical_id=record["canonical_id"],
        sequence_length=record["sequence_length"],
        sequence_token_ids_sha256_value=record["sequence_token_ids_sha256"],
        token_span=record["token_span"],
    )
    record["fisher_boundary"] = resolver.build_fisher_boundary_contract(
        tuple(range(record["sequence_length"]))
    )
    record["identity_record_sha256"] = resolver.identity_record_sha256(record)


def _stage_a_source() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for rank in range(4):
        records.append(
            _record(
                family="pg19",
                canonical_id=f"http://www.gutenberg.org/ebooks/{10_000 + rank}",
                config="default",
                rank=rank,
                seed=None,
                sequence_length=4_224,
                prefill_stop=4_096,
                scored_stop=4_224,
            )
        )
    ruler_rows = (
        ("retrieval", "niah_multiquery"),
        ("multi_hop_tracing", "vt"),
        ("aggregation", "fwe"),
        ("question_answering", "qa_1"),
    )
    for rank, (category, config) in enumerate(ruler_rows):
        records.append(
            _record(
                family="ruler",
                canonical_id=resolver.ruler_canonical_id(
                    category=category,
                    config=config,
                    configured_length=4_096,
                    seed=2_339,
                ),
                config=config,
                rank=rank,
                seed=2_339,
                sequence_length=4_096,
                prefill_stop=4_092,
                scored_stop=4_096,
                configured_length=4_096,
                ruler_category=category,
            )
        )
    for rank in range(4):
        records.append(
            _record(
                family="humaneval_plus",
                canonical_id=f"HumanEval/{rank}",
                config="default",
                rank=rank,
                seed=None,
                sequence_length=160 + rank,
                prefill_stop=64,
                scored_stop=160 + rank,
            )
        )
    for selected_family in ("pg19", "ruler", "humaneval_plus"):
        ranked = sorted(
            (row for row in records if row["family"] == selected_family),
            key=lambda row: (row["selection_sha256"], row["canonical_id"]),
        )
        for rank, row in enumerate(ranked):
            row["selection_rank"] = rank
            _refresh_record_lineage(row)
    return {
        "schema": resolver.INPUT_SCHEMA,
        "phase": "stage_a",
        "datasets": _datasets(),
        "tokenizer": _tokenizer(),
        "records": list(reversed(records)),
        "execution_bindings": dict(FIXTURE_EXECUTION_BINDINGS),
        "model_weights_loaded": False,
        "calibration_binding": dict(FIXTURE_BINDING),
    }


def _build_candidate(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("phase") != "stage_a":
        return resolver.build_candidate(source, expected_revisions=REVISIONS)
    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        return resolver.build_candidate(
            source,
            expected_revisions=REVISIONS,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resolver.canonical_json_bytes(value))


def test_stage_a_candidate_is_deterministic_and_complete() -> None:
    source = _stage_a_source()
    first = _build_candidate(source)
    second = _build_candidate(copy.deepcopy(source))

    assert first == second
    resolver.validate_candidate_artifact(first)
    evidence = first["evidence"]
    assert evidence["record_count"] == 12
    assert evidence["model_contracts"]["weights_loaded"] is False
    assert evidence["protected_identity"] == {
        "stage_b_read": False,
        "stage_c_read": False,
        "ordinary_tests_may_read_protected_content": False,
    }
    assert [row["family"] for row in evidence["records"]] == [
        *(["pg19"] * 4),
        *(["ruler"] * 4),
        *(["humaneval_plus"] * 4),
    ]
    assert evidence["tokenizer"]["file_manifest_sha256"] == _tokenizer_manifest_hash()
    assert evidence["execution_bindings"] == FIXTURE_EXECUTION_BINDINGS
    assert evidence["content_manifest_sha256"] == resolver.sha256_bytes(
        resolver.canonical_json_bytes(evidence["records"])
    )


def test_fisher_boundary_roundtrip_binds_h1_positions_and_ordered_token_hashes() -> None:
    token_ids = tuple(100 + index for index in range(19))

    boundary = resolver.build_fisher_boundary_contract(token_ids)
    normalized = resolver._normalize_fisher_boundary(
        copy.deepcopy(boundary),
        sequence_length=len(token_ids),
        context="fixture.fisher_boundary",
    )

    expected_boundaries = list(resolver.anchor_positions(len(token_ids) - 2))
    expected_inputs = [position + 1 for position in expected_boundaries]
    expected_targets = [position + 2 for position in expected_boundaries]
    assert normalized == boundary
    assert boundary["schema"] == resolver.FISHER_BOUNDARY_SCHEMA
    assert boundary["horizon"] == 1
    assert boundary["boundary_positions"] == expected_boundaries
    assert boundary["input_positions"] == expected_inputs
    assert boundary["target_positions"] == expected_targets
    assert boundary["input_token_ids_sha256"] == resolver._fisher_boundary_token_ids_sha256(
        [token_ids[position] for position in expected_inputs], role="input"
    )
    assert boundary["target_token_ids_sha256"] == resolver._fisher_boundary_token_ids_sha256(
        [token_ids[position] for position in expected_targets], role="target"
    )
    assert boundary["fisher_boundary_sha256"] == resolver.fisher_boundary_sha256(boundary)
    assert "input_token_ids" not in boundary
    assert "target_token_ids" not in boundary


def test_fisher_boundary_self_hash_and_record_hash_tampering_fail_closed() -> None:
    source = _stage_a_source()
    row = source["records"][0]
    row["fisher_boundary"]["fisher_boundary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash drifted"):
        _build_candidate(source)

    source = _stage_a_source()
    row = source["records"][0]
    row["fisher_boundary"]["input_token_ids_sha256"] = "0" * 64
    row["fisher_boundary"]["fisher_boundary_sha256"] = resolver.fisher_boundary_sha256(
        row["fisher_boundary"]
    )
    with pytest.raises(ValueError, match="identity record SHA-256 drifted"):
        _build_candidate(source)


def test_fisher_boundary_off_by_one_is_rejected_after_rehashing_both_layers() -> None:
    source = _stage_a_source()
    row = source["records"][0]
    row["fisher_boundary"]["boundary_positions"][0] += 1
    row["fisher_boundary"]["fisher_boundary_sha256"] = resolver.fisher_boundary_sha256(
        row["fisher_boundary"]
    )
    row["identity_record_sha256"] = resolver.identity_record_sha256(row)

    with pytest.raises(ValueError, match=r"B\(T\)=anchor_positions\(T-2\)"):
        _build_candidate(source)


def test_fisher_boundary_rejects_boolean_horizon_and_malformed_hash() -> None:
    source = _stage_a_source()
    row = source["records"][0]
    row["fisher_boundary"]["horizon"] = True
    row["fisher_boundary"]["fisher_boundary_sha256"] = resolver.fisher_boundary_sha256(
        row["fisher_boundary"]
    )
    row["identity_record_sha256"] = resolver.identity_record_sha256(row)
    with pytest.raises(ValueError, match="horizon must be an integer"):
        _build_candidate(source)

    source = _stage_a_source()
    row = source["records"][0]
    row["fisher_boundary"]["target_token_ids_sha256"] = "not-a-sha256"
    row["fisher_boundary"]["fisher_boundary_sha256"] = resolver.fisher_boundary_sha256(
        row["fisher_boundary"]
    )
    row["identity_record_sha256"] = resolver.identity_record_sha256(row)
    with pytest.raises(ValueError, match="target_token_ids_sha256 must be a lowercase SHA-256"):
        _build_candidate(source)


def test_fisher_boundary_rejects_sequences_shorter_than_three_tokens() -> None:
    with pytest.raises(ValueError, match="at least three tokens"):
        resolver.build_fisher_boundary_contract((1, 2))

    boundary = resolver.build_fisher_boundary_contract((1, 2, 3))
    with pytest.raises(ValueError, match="at least three tokens"):
        resolver._normalize_fisher_boundary(
            boundary,
            sequence_length=2,
            context="fixture.fisher_boundary",
        )


def test_v4_input_candidate_and_frozen_schemas_are_rejected() -> None:
    source = _stage_a_source()
    source["schema"] = "recurquant.experiment013.identity-input.v4"
    with pytest.raises(ValueError, match="identity input schema drifted"):
        _build_candidate(source)

    candidate = _build_candidate(_stage_a_source())
    candidate["evidence"]["identity_schema"] = "recurquant.experiment013.identity-candidate.v4"
    candidate["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(candidate["evidence"])
    )
    with pytest.raises(ValueError, match="candidate identity_schema drifted"):
        resolver.validate_candidate_artifact(candidate)

    candidate = _build_candidate(_stage_a_source())
    candidate_hash = resolver.sha256_bytes(resolver.canonical_json_bytes(candidate))
    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )
    frozen["evidence"]["identity_schema"] = "recurquant.experiment013.identity-frozen.v4"
    frozen["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(frozen["evidence"])
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        pytest.raises(ValueError, match="frozen Stage-A identity contract drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(frozen),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda evidence: evidence.update({"identity_only": 1}), "identity_only drifted"),
        (
            lambda evidence: evidence.update({"promotion_required": 1}),
            "promotion_required drifted",
        ),
        (
            lambda evidence: evidence["protected_identity"].update({"stage_b_read": 0}),
            "protected identity boundary drifted",
        ),
    ],
)
def test_candidate_rejects_boolean_integer_aliases(mutate: Any, message: str) -> None:
    candidate = _build_candidate(_stage_a_source())
    mutate(candidate["evidence"])
    candidate["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(candidate["evidence"])
    )

    with pytest.raises(ValueError, match=message):
        resolver.validate_candidate_artifact(candidate)


def test_stage_a_candidate_requires_and_matches_a_verified_binding_artifact() -> None:
    source = _stage_a_source()
    with pytest.raises(ValueError, match="requires a verified calibration binding"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)

    source["calibration_binding"]["static_k29334_policy_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the verified artifact"):
        _build_candidate(source)


def test_stage_a_binding_v3_round_trips_exact_eight_embedded_dependencies() -> None:
    with _binding_v3_fixture() as fixture:
        artifact = resolver.build_stage_a_calibration_binding_artifact(**fixture.dependencies)
        decoded = resolver.deserialize_stage_a_calibration_binding_artifact(artifact)

    document = json.loads(artifact)
    expected_dependencies = {
        "calibration_score_artifact",
        "comparator_score_artifact",
        "frozen_identity_artifact",
        "split_half_stability_artifact",
        "static_fisher_k29334_policy_artifact",
        "static_k27030_policy_artifact",
        "static_k29334_policy_artifact",
        "static_mse_k29334_policy_artifact",
    }
    assert resolver.STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION == 3
    assert resolver.STAGE_A_BINDING_ARTIFACT_REVISION.endswith("-v3")
    assert document["schema_version"] == 3
    assert document["evidence"]["artifact_revision"].endswith("-v3")
    assert set(document["evidence"]["dependencies_base64"]) == expected_dependencies
    assert set(document["evidence"]["dependency_file_sha256"]) == expected_dependencies
    assert set(decoded.binding) == resolver.CALIBRATION_BINDING_FIELDS
    assert decoded.binding["comparator_score_artifact_file_sha256"] == resolver.sha256_bytes(
        fixture.dependencies["comparator_score_artifact"]
    )
    assert decoded.binding["static_fisher_k29334_policy_file_sha256"] == (
        resolver.sha256_bytes(fixture.dependencies["static_fisher_k29334_policy_artifact"])
    )
    assert decoded.binding["static_mse_k29334_policy_file_sha256"] == resolver.sha256_bytes(
        fixture.dependencies["static_mse_k29334_policy_artifact"]
    )
    with pytest.raises(TypeError):
        decoded.binding["calibration_identity_file_sha256"] = "0" * 64
    with pytest.raises(TypeError):
        decoded.dependency_file_sha256["frozen_identity_artifact"] = "0" * 64
    with pytest.raises(AttributeError):
        decoded.file_sha256 = "0" * 64


def test_stage_a_binding_rejects_v2_missing_extra_and_one_byte_dependency_tamper() -> None:
    with _binding_v3_fixture() as fixture:
        artifact = resolver.build_stage_a_calibration_binding_artifact(**fixture.dependencies)

        legacy = json.loads(artifact)
        legacy["schema_version"] = 2
        legacy["evidence"]["artifact_revision"] = "experiment-013-stage-a-calibration-binding-v2"
        with pytest.raises(ValueError, match="kind or schema drifted"):
            resolver.deserialize_stage_a_calibration_binding_artifact(
                _reauthenticated_binding_bytes(legacy)
            )

        missing = json.loads(artifact)
        del missing["evidence"]["dependencies_base64"]["comparator_score_artifact"]
        with pytest.raises(ValueError, match="dependencies fields drifted"):
            resolver.deserialize_stage_a_calibration_binding_artifact(
                _reauthenticated_binding_bytes(missing)
            )

        extra = json.loads(artifact)
        extra["evidence"]["dependencies_base64"]["ninth_dependency"] = resolver._canonical_b64(
            b"forbidden", context="fixture"
        )
        with pytest.raises(ValueError, match="dependencies fields drifted"):
            resolver.deserialize_stage_a_calibration_binding_artifact(
                _reauthenticated_binding_bytes(extra)
            )

        tampered = json.loads(artifact)
        original = resolver._decode_canonical_b64(
            tampered["evidence"]["dependencies_base64"]["comparator_score_artifact"],
            context="fixture",
        )
        changed = bytes([original[0] ^ 1]) + original[1:]
        tampered["evidence"]["dependencies_base64"]["comparator_score_artifact"] = (
            resolver._canonical_b64(changed, context="fixture")
        )
        with pytest.raises(ValueError, match="dependency bytes differ"):
            resolver.deserialize_stage_a_calibration_binding_artifact(
                _reauthenticated_binding_bytes(tampered)
            )


def test_stage_a_binding_rejects_cross_profile_policy_swap() -> None:
    with _binding_v3_fixture() as fixture:
        dependencies = dict(fixture.dependencies)
        dependencies["static_fisher_k29334_policy_artifact"] = fixture.dependencies[
            "static_mse_k29334_policy_artifact"
        ]
        dependencies["static_mse_k29334_policy_artifact"] = fixture.dependencies[
            "static_fisher_k29334_policy_artifact"
        ]
        with pytest.raises(ValueError, match="does not satisfy its frozen K29334 geometry"):
            resolver.build_stage_a_calibration_binding_artifact(**dependencies)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda fixture, policy: setattr(policy, "model_id", "Qwen/wrong-model"),
            "frozen model contract drifted",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "tokenizer_manifest_sha256",
                _hash("wrong-tokenizer-manifest"),
            ),
            "frozen identity binding drifted",
        ),
        (
            lambda fixture, policy: setattr(policy, "source_commit", "b" * 40),
            "source commit differs from H0",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "marginal_steps",
                static_q468.FROZEN_STATIC_Q468_PRIMARY_STEPS - 1,
            ),
            "frozen K29334 geometry",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "geometry",
                replace(
                    static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
                    target_resident_bytes=(static_q468.FROZEN_STATELEASE_RESIDENT_BYTES + 8),
                ),
            ),
            "frozen K29334 geometry",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "identity_artifact_sha256",
                _hash("wrong-frozen-identity"),
            ),
            "frozen identity binding drifted",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "calibration_manifest_sha256",
                _hash("wrong-comparator-sequence-manifest"),
            ),
            "decoded selector scores",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "calibration_scores_sha256",
                _hash("wrong-raw-distortion-hash"),
            ),
            "raw distortion hash",
        ),
        (
            lambda fixture, policy: setattr(
                policy,
                "code_map_sha256",
                _hash("wrong-comparator-code-map"),
            ),
            "code map differs from its exact allocation",
        ),
    ],
)
def test_stage_a_binding_rederives_every_comparator_policy_contract(
    mutation: Any,
    message: str,
) -> None:
    with _binding_v3_fixture() as fixture:
        policy = fixture.policies[static_q468.STATIC_Q468_MSE_METHOD]
        mutation(fixture, policy)
        with pytest.raises(ValueError, match=message):
            resolver.build_stage_a_calibration_binding_artifact(**fixture.dependencies)


def test_stage_a_binding_rejects_incomplete_comparator_identity_manifest() -> None:
    with _binding_v3_fixture() as fixture:
        selector = fixture.comparator_scores.selectors[
            calibration.FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE
        ]
        selector.aggregate.identity_record_manifest_sha256 = _hash(
            "incomplete-comparator-identity-record-manifest"
        )
        with pytest.raises(ValueError, match="complete frozen identity"):
            resolver.build_stage_a_calibration_binding_artifact(**fixture.dependencies)


def test_raw_content_and_unknown_fields_fail_closed() -> None:
    source = _stage_a_source()
    source["records"][0]["prompt"] = "raw protected text"

    with pytest.raises(ValueError, match="fields drifted"):
        _build_candidate(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source["tokenizer"].update({"revision": "f" * 40}),
            "tokenizer revision",
        ),
        (
            lambda source: source["records"][0].update({"tokenizer_manifest_sha256": "0" * 64}),
            "tokenizer manifest binding",
        ),
        (
            lambda source: source["records"][0].update({"selection_sha256": "0" * 64}),
            "selection SHA-256",
        ),
        (
            lambda source: source["datasets"][1].update({"canonical_id_field": "book_id"}),
            "pg19 canonical ID field",
        ),
        (
            lambda source: source.update({"model_weights_loaded": True}),
            "before model weights",
        ),
        (
            lambda source: source["execution_bindings"].update(
                {"model_file_manifest_file_sha256": "not-a-sha256"}
            ),
            "model_file_manifest_file_sha256",
        ),
        (
            lambda source: source["execution_bindings"].update(
                {"parquet_materialization_manifest_file_sha256": "0" * 64}
            ),
            "Parquet materialization manifest file SHA-256 drifted",
        ),
        (
            lambda source: source["records"][0]["token_span"].update({"scored_start": 4_095}),
            "contiguous",
        ),
        (
            lambda source: source["records"][0]["token_span"].update(
                {"cache_exposed_start": source["records"][0]["token_span"]["scored_start"]}
            ),
            "exclude the first continuation token",
        ),
    ],
)
def test_identity_contract_drift_fails_closed(mutation: Any, message: str) -> None:
    source = _stage_a_source()
    mutation(source)

    with pytest.raises(ValueError, match=message):
        _build_candidate(source)


def test_dataset_revision_must_match_explicit_cli_contract() -> None:
    source = _stage_a_source()
    source["datasets"][1]["revision"] = "9" * 40

    with pytest.raises(ValueError, match="does not match the CLI contract"):
        _build_candidate(source)


@pytest.mark.parametrize(
    ("phase", "pg19_split"),
    [("calibration", "train"), ("stage_a", "validation")],
)
def test_dataset_contracts_bind_phase_splits_and_exact_formatters(
    phase: str,
    pg19_split: str,
) -> None:
    datasets = _datasets()
    next(item for item in datasets if item["key"] == "pg19")["split"] = pg19_split
    normalized = resolver._validate_dataset_contracts(
        datasets,
        expected_revisions=REVISIONS,
        phase=phase,
    )
    assert {item["key"]: item["split"] for item in normalized} == (
        resolver.FROZEN_DATASET_SPLITS[phase]
    )
    assert {item["key"]: item["formatter_id"] for item in normalized} == (
        resolver.FROZEN_FORMATTER_IDS
    )

    next(item for item in datasets if item["key"] == "pg19")["split"] = (
        "validation" if phase == "calibration" else "train"
    )
    with pytest.raises(ValueError, match=rf"{phase} pg19 dataset split must be"):
        resolver._validate_dataset_contracts(
            datasets,
            expected_revisions=REVISIONS,
            phase=phase,
        )

    datasets = _datasets()
    next(item for item in datasets if item["key"] == "pg19")["split"] = pg19_split
    next(item for item in datasets if item["key"] == "ruler")["formatter_id"] = (
        "recurquant.ruler-attacker-controlled.v1"
    )
    with pytest.raises(ValueError, match="ruler formatter ID must be"):
        resolver._validate_dataset_contracts(
            datasets,
            expected_revisions=REVISIONS,
            phase=phase,
        )

    datasets = _datasets()
    next(item for item in datasets if item["key"] == "pg19")["split"] = pg19_split
    next(item for item in datasets if item["key"] == "pg19")["formatter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pg19 formatter SHA-256 drifted"):
        resolver._validate_dataset_contracts(
            datasets,
            expected_revisions=REVISIONS,
            phase=phase,
        )


@pytest.mark.parametrize(
    ("family", "bad_id", "message"),
    [
        ("pg19", "https://www.gutenberg.org/ebooks/10000", "exact http"),
        ("pg19", "http://www.gutenberg.org/ebooks/010000", "exact http"),
        ("pg19", "http://www.gutenberg.org/ebooks/10000?raw=prompt", "exact http"),
        ("humaneval_plus", "HumanEvalPlus/0", "HumanEval/0..163"),
        ("humaneval_plus", "HumanEval/01", "HumanEval/0..163"),
        ("humaneval_plus", "HumanEval/164", "HumanEval/0..163"),
    ],
)
def test_canonical_dataset_identifier_shapes_fail_closed(
    family: str,
    bad_id: str,
    message: str,
) -> None:
    source = _stage_a_source()
    row = next(item for item in source["records"] if item["family"] == family)
    row["canonical_id"] = bad_id

    with pytest.raises(ValueError, match=message):
        _build_candidate(source)


@pytest.mark.parametrize(
    ("bad_id", "message"),
    [
        ("http://www.gutenberg.org/ebooks/10000\nraw-prompt", "control character"),
        ("x" * (resolver.MAX_METADATA_STRING_LENGTH + 1), "metadata length limit"),
        ("def solve(): return 'raw prompt'", "whitespace or raw content"),
    ],
)
def test_metadata_strings_reject_controls_overlong_values_and_raw_content(
    bad_id: str,
    message: str,
) -> None:
    source = _stage_a_source()
    row = next(item for item in source["records"] if item["family"] == "pg19")
    row["canonical_id"] = bad_id

    with pytest.raises(ValueError, match=message):
        _build_candidate(source)


def test_ruler_category_config_and_actual_length_are_independently_bound() -> None:
    source = _stage_a_source()
    ruler = next(row for row in source["records"] if row["family"] == "ruler")
    ruler["ruler_category"] = "aggregation"
    with pytest.raises(ValueError, match="config/category binding"):
        _build_candidate(source)

    source = _stage_a_source()
    ruler = next(row for row in source["records"] if row["family"] == "ruler")
    ruler["configured_length"] = 4_095
    ruler["canonical_id"] = resolver.ruler_canonical_id(
        category=ruler["ruler_category"],
        config=ruler["config"],
        configured_length=ruler["configured_length"],
        seed=ruler["seed"],
    )
    ruler["selection_sha256"] = resolver.selection_sha256(
        resolver.RULER_STAGE_A_SELECTION_NAMESPACE,
        ruler["canonical_id"],
    )
    _refresh_record_lineage(ruler)
    with pytest.raises(ValueError, match="exceeds the RULER configured length"):
        _build_candidate(source)

    source = _stage_a_source()
    pg19 = next(row for row in source["records"] if row["family"] == "pg19")
    pg19["ruler_category"] = "retrieval"
    with pytest.raises(ValueError, match="non-RULER rows"):
        _build_candidate(source)


def test_ruler_canonical_id_is_derived_from_the_complete_generation_tuple() -> None:
    source = _stage_a_source()
    ruler = next(row for row in source["records"] if row["family"] == "ruler")
    ruler["canonical_id"] = "forged-ruler-id"
    ruler["selection_sha256"] = resolver.selection_sha256(
        resolver.RULER_STAGE_A_SELECTION_NAMESPACE,
        ruler["canonical_id"],
    )
    _refresh_record_lineage(ruler)

    with pytest.raises(ValueError, match="RULER canonical ID drifted"):
        _build_candidate(source)


@pytest.mark.parametrize("family", ["pg19", "humaneval_plus"])
def test_non_ruler_record_configs_are_exact(family: str) -> None:
    source = _stage_a_source()
    row = next(item for item in source["records"] if item["family"] == family)
    row["config"] = "forged-config"
    _refresh_record_lineage(row)

    with pytest.raises(ValueError, match=rf"{family} config must be 'default'"):
        _build_candidate(source)


def test_duplicate_stage_a_canonical_ids_fail_after_complete_rehash() -> None:
    source = _stage_a_source()
    pg19 = [row for row in source["records"] if row["family"] == "pg19"]
    pg19[1]["canonical_id"] = pg19[0]["canonical_id"]
    pg19[1]["selection_sha256"] = resolver.selection_sha256(
        resolver.PG19_VALIDATION_NAMESPACE,
        pg19[1]["canonical_id"],
    )
    ranked = sorted(
        pg19,
        key=lambda row: (row["selection_sha256"], row["canonical_id"]),
    )
    for rank, row in enumerate(ranked):
        row["selection_rank"] = rank
        _refresh_record_lineage(row)

    with pytest.raises(ValueError, match="duplicate canonical selection keys"):
        _build_candidate(source)


def test_sha_rank_order_rejects_duplicate_keys_for_calibration_and_stage_a() -> None:
    rows = (
        {"selection_sha256": "a" * 64, "canonical_id": "same", "selection_rank": 0},
        {"selection_sha256": "a" * 64, "canonical_id": "same", "selection_rank": 1},
    )
    for context in ("calibration PG19", "Stage-A PG19"):
        with pytest.raises(ValueError, match="duplicate canonical selection keys"):
            resolver._validate_sha_rank_order(rows, context=context)


def test_stage_a_requires_two_continuation_tokens_for_one_cache_prediction() -> None:
    source = _stage_a_source()
    row = source["records"][0]
    stop = row["token_span"]["scored_stop"]
    row["token_span"].update(
        {
            "prefill_stop": stop - 1,
            "scored_start": stop - 1,
            "cache_exposed_start": stop,
            "cache_exposed_stop": stop,
        }
    )

    with pytest.raises(ValueError, match="continuation must contain at least two"):
        _build_candidate(source)


def test_calibration_cache_exposure_is_empty_at_continuation_stop() -> None:
    source = _stage_a_source()
    row = copy.deepcopy(next(item for item in source["records"] if item["family"] == "pg19"))
    row["selection_sha256"] = resolver.selection_sha256(
        resolver.PG19_TRAIN_NAMESPACE, row["canonical_id"]
    )
    stop = row["token_span"]["scored_stop"]
    row["token_span"].update({"cache_exposed_start": stop, "cache_exposed_stop": stop})
    _refresh_record_lineage(row)

    normalized = resolver._normalize_record(
        row,
        index=0,
        phase="calibration",
        tokenizer_hash=_tokenizer_manifest_hash(),
    )
    assert normalized["token_span"]["cache_exposed_start"] == stop
    assert normalized["token_span"]["cache_exposed_stop"] == stop

    row["token_span"]["cache_exposed_start"] = stop - 1
    with pytest.raises(ValueError, match="calibration cache-exposed prediction span"):
        resolver._normalize_record(
            row,
            index=0,
            phase="calibration",
            tokenizer_hash=_tokenizer_manifest_hash(),
        )


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _stage_a_source())
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)

    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        result = resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(source_path),
                "--calibration-binding",
                str(binding_path),
                "--dry-run",
                "--mbpp-revision",
                REVISIONS["mbpp"],
                "--pg19-revision",
                REVISIONS["pg19"],
                "--ruler-revision",
                REVISIONS["ruler"],
                "--humaneval-plus-revision",
                REVISIONS["humaneval_plus"],
            ]
        )

    assert result == 0
    assert len(capsys.readouterr().out.strip()) == 64
    assert sorted(path.name for path in tmp_path.iterdir()) == ["binding.json", "source.json"]


def test_candidate_requires_quarantine_then_exact_hash_promotion(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _stage_a_source())
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)
    candidate_path = tmp_path / ".quarantine" / "stage-a-candidate.json"
    base_args = [
        "--phase",
        "stage_a",
        "--input",
        str(source_path),
        "--calibration-binding",
        str(binding_path),
        "--mbpp-revision",
        REVISIONS["mbpp"],
        "--pg19-revision",
        REVISIONS["pg19"],
        "--ruler-revision",
        REVISIONS["ruler"],
        "--humaneval-plus-revision",
        REVISIONS["humaneval_plus"],
    ]

    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        assert resolver.main([*base_args, "--output", str(candidate_path)]) == 0
        candidate_hash = resolver.sha256_bytes(candidate_path.read_bytes())
        frozen_path = tmp_path / "frozen" / "stage-a-identity.json"
        assert (
            resolver.main(
                [
                    "--phase",
                    "stage_a",
                    "--input",
                    str(candidate_path),
                    "--output",
                    str(frozen_path),
                    "--promote",
                    "--calibration-binding",
                    str(binding_path),
                    "--expected-candidate-sha256",
                    candidate_hash,
                ]
            )
            == 0
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["evidence"]["status"] == "frozen"
    assert frozen["evidence"]["promotion"]["candidate_file_sha256"] == candidate_hash
    assert frozen["evidence"]["model_contracts"]["weights_loaded"] is False


def test_stage_a_promotion_requires_the_verified_binding_artifact() -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate_hash = resolver.sha256_bytes(resolver.canonical_json_bytes(candidate))

    with pytest.raises(ValueError, match="promotion requires a verified calibration binding"):
        resolver.promote_candidate(candidate, candidate_file_sha256=candidate_hash)

    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )

    assert frozen["evidence"]["status"] == "frozen"


def test_frozen_stage_a_decoder_reauthenticates_promotion_records_and_binding() -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate_hash = resolver.sha256_bytes(resolver.canonical_json_bytes(candidate))
    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING))
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        return_value=verified,
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )
        frozen_bytes = resolver.canonical_json_bytes(frozen)
        decoded = resolver.deserialize_frozen_stage_a_identity_artifact(
            frozen_bytes,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )

    assert decoded.file_sha256 == resolver.sha256_bytes(frozen_bytes)
    assert len(decoded.records) == 12
    assert decoded.calibration_binding == FIXTURE_BINDING
    assert decoded.execution_bindings == FIXTURE_EXECUTION_BINDINGS
    assert (
        decoded.parquet_materialization_manifest_file_sha256
        == resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    )
    original_canonical_id = decoded.records[0]["canonical_id"]
    original_boundaries = tuple(decoded.records[0]["fisher_boundary"]["boundary_positions"])
    with pytest.raises(TypeError):
        decoded.records[0]["canonical_id"] = "HumanEval/99"
    with pytest.raises(TypeError, match="immutable"):
        decoded.records[0]["fisher_boundary"]["boundary_positions"].append(999)
    with pytest.raises(TypeError):
        decoded.execution_bindings["repository_source_manifest_file_sha256"] = "0" * 64
    with pytest.raises(AttributeError):
        decoded.records = ()
    assert decoded.records[0]["canonical_id"] == original_canonical_id
    assert tuple(decoded.records[0]["fisher_boundary"]["boundary_positions"]) == original_boundaries

    tampered = copy.deepcopy(frozen)
    tampered["evidence"]["records"][0]["source_content_sha256"] = "0" * 64
    tampered["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"])
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        pytest.raises(ValueError, match="identity record SHA-256 drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(tampered),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )
    wrong_binding = dict(FIXTURE_BINDING)
    wrong_binding["static_k29334_policy_file_sha256"] = "0" * 64
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=SimpleNamespace(binding=wrong_binding),
        ),
        pytest.raises(ValueError, match="differs from the verified calibration binding"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            frozen_bytes,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )

    tampered = copy.deepcopy(frozen)
    tampered["evidence"]["promotion"]["candidate_file_sha256"] = "0" * 64
    tampered["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"])
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        pytest.raises(ValueError, match="candidate file SHA-256 drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(tampered),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )


def test_calibration_identity_dto_recursively_freezes_verified_data() -> None:
    record = {"nested": {"positions": [1, 2, 3]}}
    assignment = {"identity": ["pg19", "http://www.gutenberg.org/ebooks/10000"]}
    dto = resolver.FrozenCalibrationIdentityArtifact(
        file_sha256="1" * 64,
        canonical_evidence_sha256="2" * 64,
        records=(record,),
        assignment=(assignment,),
        assignment_sha256="3" * 64,
        tokenizer_manifest_sha256="4" * 64,
        parquet_materialization_manifest_file_sha256=(
            resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
        ),
        execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS),
    )

    record["nested"]["positions"].append(4)
    assignment["identity"].append("attacker")
    with pytest.raises(TypeError, match="immutable"):
        dto.records[0]["nested"]["positions"].append(5)
    with pytest.raises(TypeError):
        dto.assignment[0]["identity"][0] = "ruler"
    with pytest.raises(TypeError):
        dto.execution_bindings["repository_source_manifest_file_sha256"] = "0" * 64
    with pytest.raises(AttributeError):
        dto.assignment = ()
    assert dto.records[0]["nested"]["positions"] == [1, 2, 3]
    assert dto.assignment[0]["identity"] == [
        "pg19",
        "http://www.gutenberg.org/ebooks/10000",
    ]
    assert json.loads(resolver.canonical_json_bytes(dto.records[0])) == {
        "nested": {"positions": [1, 2, 3]}
    }


def test_handcrafted_incomplete_candidate_cannot_be_promoted() -> None:
    evidence = {
        "identity_schema": resolver.CANDIDATE_SCHEMA,
        "status": "candidate",
        "phase": "stage_a",
        "identity_only": True,
        "claim_boundary": resolver.CLAIM_BOUNDARY,
        "promotion_required": True,
        "model_contracts": {"weights_loaded": False},
        "records": [],
        "record_count": 0,
        "content_manifest_sha256": resolver.sha256_bytes(resolver.canonical_json_bytes([])),
        "protected_identity": {
            "stage_b_read": False,
            "stage_c_read": False,
            "ordinary_tests_may_read_protected_content": False,
        },
    }
    candidate = {
        "canonical_evidence_sha256": resolver.sha256_bytes(resolver.canonical_json_bytes(evidence)),
        "evidence": evidence,
    }

    with pytest.raises(ValueError, match="candidate evidence fields drifted"):
        resolver.promote_candidate(
            candidate,
            candidate_file_sha256=resolver.sha256_bytes(resolver.canonical_json_bytes(candidate)),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )


def test_noncanonical_candidate_bytes_cannot_be_promoted(tmp_path: Path) -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate_path = tmp_path / ".quarantine" / "candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)

    with pytest.raises(ValueError, match="not canonical resolver JSON"):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(candidate_path),
                "--output",
                str(tmp_path / "frozen" / "identity.json"),
                "--promote",
                "--calibration-binding",
                str(binding_path),
                "--expected-candidate-sha256",
                resolver.sha256_bytes(candidate_path.read_bytes()),
            ]
        )


def test_atomic_identity_publish_cannot_overwrite_an_existing_or_racing_file(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        resolver.atomic_write(existing, b"replacement")
    assert existing.read_bytes() == b"existing"

    racing = tmp_path / "racing.json"

    def create_racing_destination(_source: object, destination: object) -> None:
        Path(destination).write_bytes(b"racer")
        raise FileExistsError

    with (
        patch.object(resolver.os, "link", side_effect=create_racing_destination),
        pytest.raises(
            FileExistsError,
            match="refusing to overwrite",
        ),
    ):
        resolver.atomic_write(racing, b"candidate")
    assert racing.read_bytes() == b"racer"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_candidate_wrong_hash_cannot_be_promoted(tmp_path: Path) -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate_path = tmp_path / ".quarantine" / "candidate.json"
    _write_json(candidate_path, candidate)
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)

    with pytest.raises(ValueError, match="does not match explicit promotion hash"):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(candidate_path),
                "--output",
                str(tmp_path / "frozen" / "identity.json"),
                "--promote",
                "--calibration-binding",
                str(binding_path),
                "--expected-candidate-sha256",
                "0" * 64,
            ]
        )


@pytest.mark.parametrize("phase", ["stage_b", "stage_c"])
def test_protected_phase_is_rejected_before_input_read(tmp_path: Path, phase: str) -> None:
    nonexistent = tmp_path / "protected-content-that-must-not-be-read.json"

    with pytest.raises(PermissionError, match="before reading --input"):
        resolver.main(["--phase", phase, "--input", str(nonexistent), "--dry-run"])


def test_candidate_tampering_is_detected() -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate["evidence"]["records"][0]["source_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="canonical evidence SHA-256"):
        resolver.validate_candidate_artifact(candidate)
