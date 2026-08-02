from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "resolve_static_q468_identity.py"
SPEC = importlib.util.spec_from_file_location("resolve_static_q468_identity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)

REVISIONS = {
    "mbpp": resolver.MBPP_REVISION,
    "pg19": resolver.PG19_REVISION,
    "ruler": resolver.RULER_REVISION,
    "humaneval_plus": resolver.HUMANEVAL_PLUS_REVISION,
}


def _hash(label: str) -> str:
    return resolver.sha256_bytes(label.encode())


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
            "formatter_id": "recurquant.mbpp.v0.2",
            "formatter_sha256": _hash("mbpp-formatter"),
        },
        {
            "key": "pg19",
            "dataset_id": resolver.PG19_DATASET_ID,
            "config": "default",
            "revision": REVISIONS["pg19"],
            "split": "validation",
            "canonical_id_field": "url",
            "canonical_id_manifest_sha256": _hash("pg19-id-manifest"),
            "formatter_id": "recurquant.pg19.contiguous.v1",
            "formatter_sha256": _hash("pg19-formatter"),
        },
        {
            "key": "ruler",
            "dataset_id": resolver.RULER_SOURCE_ID,
            "config": "official-generator",
            "revision": REVISIONS["ruler"],
            "split": "generated",
            "canonical_id_field": "configuration_id",
            "canonical_id_manifest_sha256": _hash("ruler-id-manifest"),
            "formatter_id": "recurquant.ruler.official.v1",
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
            "formatter_id": "recurquant.humaneval-plus.v1",
            "formatter_sha256": _hash("humaneval-formatter"),
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
    return {
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
        "tokenizer_manifest_sha256": _tokenizer_manifest_hash(),
        "token_span": {
            "prefill_start": 0,
            "prefill_stop": prefill_stop,
            "scored_start": prefill_stop,
            "scored_stop": scored_stop,
        },
        "anchor_manifest_sha256": _hash(f"anchors-{label}"),
    }


def _stage_a_source() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for rank in range(4):
        records.append(
            _record(
                family="pg19",
                canonical_id=f"book-{rank}",
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
                canonical_id=f"{config}-4096-2339",
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
    return {
        "schema": resolver.INPUT_SCHEMA,
        "phase": "stage_a",
        "datasets": _datasets(),
        "tokenizer": _tokenizer(),
        "records": list(reversed(records)),
        "model_weights_loaded": False,
        "calibration_binding": {
            "identity_file_sha256": _hash("calibration-file"),
            "canonical_identity_sha256": _hash("calibration-identity"),
            "static_k29334_code_map_sha256": _hash("k29334"),
            "static_k27030_code_map_sha256": _hash("k27030"),
            "split_half_policy_manifest_sha256": _hash("split-half-policies"),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resolver.canonical_json_bytes(value))


def test_stage_a_candidate_is_deterministic_and_complete() -> None:
    source = _stage_a_source()
    first = resolver.build_candidate(source, expected_revisions=REVISIONS)
    second = resolver.build_candidate(copy.deepcopy(source), expected_revisions=REVISIONS)

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
    assert evidence["content_manifest_sha256"] == resolver.sha256_bytes(
        resolver.canonical_json_bytes(evidence["records"])
    )


def test_raw_content_and_unknown_fields_fail_closed() -> None:
    source = _stage_a_source()
    source["records"][0]["prompt"] = "raw protected text"

    with pytest.raises(ValueError, match="fields drifted"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source["tokenizer"].update({"revision": "f" * 40}),
            "tokenizer revision",
        ),
        (
            lambda source: source["records"][0].update(
                {"tokenizer_manifest_sha256": "0" * 64}
            ),
            "tokenizer manifest binding",
        ),
        (
            lambda source: source["records"][0].update({"selection_sha256": "0" * 64}),
            "selection SHA-256",
        ),
        (
            lambda source: source["datasets"][1].update(
                {"canonical_id_field": "book_id"}
            ),
            "pg19 canonical ID field",
        ),
        (
            lambda source: source.update({"model_weights_loaded": True}),
            "before model weights",
        ),
        (
            lambda source: source["records"][0]["token_span"].update(
                {"scored_start": 4_095}
            ),
            "contiguous",
        ),
    ],
)
def test_identity_contract_drift_fails_closed(mutation: Any, message: str) -> None:
    source = _stage_a_source()
    mutation(source)

    with pytest.raises(ValueError, match=message):
        resolver.build_candidate(source, expected_revisions=REVISIONS)


def test_dataset_revision_must_match_explicit_cli_contract() -> None:
    source = _stage_a_source()
    source["datasets"][1]["revision"] = "9" * 40

    with pytest.raises(ValueError, match="does not match the CLI contract"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)


def test_ruler_category_config_and_actual_length_are_independently_bound() -> None:
    source = _stage_a_source()
    ruler = next(row for row in source["records"] if row["family"] == "ruler")
    ruler["ruler_category"] = "aggregation"
    with pytest.raises(ValueError, match="config/category binding"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)

    source = _stage_a_source()
    ruler = next(row for row in source["records"] if row["family"] == "ruler")
    ruler["configured_length"] = 4_095
    with pytest.raises(ValueError, match="exceeds the RULER configured length"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)

    source = _stage_a_source()
    pg19 = next(row for row in source["records"] if row["family"] == "pg19")
    pg19["ruler_category"] = "retrieval"
    with pytest.raises(ValueError, match="non-RULER rows"):
        resolver.build_candidate(source, expected_revisions=REVISIONS)


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _stage_a_source())

    result = resolver.main(
        [
            "--phase",
            "stage_a",
            "--input",
            str(source_path),
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
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source.json"]


def test_candidate_requires_quarantine_then_exact_hash_promotion(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _stage_a_source())
    candidate_path = tmp_path / ".quarantine" / "stage-a-candidate.json"
    base_args = [
        "--phase",
        "stage_a",
        "--input",
        str(source_path),
        "--mbpp-revision",
        REVISIONS["mbpp"],
        "--pg19-revision",
        REVISIONS["pg19"],
        "--ruler-revision",
        REVISIONS["ruler"],
        "--humaneval-plus-revision",
        REVISIONS["humaneval_plus"],
    ]

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


def test_candidate_wrong_hash_cannot_be_promoted(tmp_path: Path) -> None:
    candidate = resolver.build_candidate(_stage_a_source(), expected_revisions=REVISIONS)
    candidate_path = tmp_path / ".quarantine" / "candidate.json"
    _write_json(candidate_path, candidate)

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
    candidate = resolver.build_candidate(_stage_a_source(), expected_revisions=REVISIONS)
    candidate["evidence"]["records"][0]["source_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="canonical evidence SHA-256"):
        resolver.validate_candidate_artifact(candidate)
