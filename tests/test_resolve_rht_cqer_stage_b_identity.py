from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recurquant.evidence import canonical_json_bytes
from recurquant.public_data import (
    MBPP_CALIBRATION_SIZE,
    mbpp_calibration_key,
    mbpp_manifest,
    mbpp_manifest_content_sha256,
    select_mbpp_calibration,
)

resolver = importlib.import_module("scripts.resolve_rht_cqer_stage_b_identity")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolver_runtime_modules() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    frozen_paths = set(resolver.STAGE_B_SOURCE_FILES)
    for name, module in sys.modules.items():
        if not (
            name == "recurquant"
            or name.startswith("recurquant.")
            or name == resolver.__name__
        ):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            continue
        if relative in frozen_paths:
            modules[name] = module
    return modules


def test_all_imported_repository_modules_are_source_frozen() -> None:
    imported = resolver.validate_all_imported_repository_modules_frozen(
        REPOSITORY_ROOT,
        modules=_resolver_runtime_modules(),
    )

    assert imported["recurquant"] == "src/recurquant/__init__.py"
    assert set(imported.values()) <= set(resolver.STAGE_B_SOURCE_FILES)


def test_unfrozen_imported_repository_module_is_rejected() -> None:
    unfrozen = SimpleNamespace(
        __file__=str(REPOSITORY_ROOT / "src" / "recurquant" / "cli.py")
    )

    with pytest.raises(ValueError, match="outside the Stage-B source freeze"):
        resolver.validate_all_imported_repository_modules_frozen(
            REPOSITORY_ROOT,
            modules={"recurquant.unfrozen_test": unfrozen},
        )


def _row(task_id: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "text": f"Return {task_id}.",
        "code": f"def answer_{task_id}():\n    return {task_id}\n",
        "test_list": [f"assert answer_{task_id}() == {task_id}"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }


class _TrackedRow(Mapping[str, Any]):
    def __init__(self, row: Mapping[str, Any], accesses: list[tuple[int, str]]) -> None:
        self._row = row
        self._accesses = accesses

    def __getitem__(self, key: str) -> Any:
        self._accesses.append((int(self._row["task_id"]), key))
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def get(self, key: str, default: Any = None) -> Any:
        self._accesses.append((int(self._row["task_id"]), key))
        return self._row.get(key, default)


class Qwen2Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, list[int]]:
        assert return_attention_mask is False
        assert return_token_type_ids is False
        payload = list(text.encode("utf-8"))
        return {"input_ids": ([1] if add_special_tokens else []) + payload}


def _compact_row_plan() -> dict[str, Any]:
    locations = [
        {
            "layer_index": layer,
            "head_index": flat_index // 128,
            "row_index": flat_index % 128,
        }
        for layer in resolver.FROZEN_LINEAR_LAYERS
        for flat_index in range(resolver.FROZEN_LAYER_QUOTAS[layer])
    ]
    plan: dict[str, Any] = {
        "schema": resolver.ROW_PLAN_SCHEMA,
        "method": resolver.ROW_PLAN_METHOD,
        "selector_binding": {
            "selector_file_sha256": resolver.SELECTOR_FILE_SHA256,
            "selector_canonical_evidence_sha256": (
                resolver.SELECTOR_CANONICAL_EVIDENCE_SHA256
            ),
            "loss_selector_file_sha256": resolver.LOSS_SELECTOR_FILE_SHA256,
            "loss_selector_canonical_evidence_sha256": (
                resolver.LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
            ),
        },
        "model": {
            "id": resolver.MODEL_ID,
            "revision": resolver.MODEL_REVISION,
        },
        "quantization": {
            "low_bits": 4,
            "high_bits": 8,
            "group_size": 128,
            "scale_bits": 16,
        },
        "accounting": {
            "total_groups": 36_864,
            "mask_bytes": 4_608,
            "promotion_increment_bytes": 64,
            "target_resident_bytes": 2_564_096,
            "resident_bytes": 2_564_096,
            "promoted_group_count": 1_976,
        },
        "score_shapes": [
            {"layer_index": layer, "heads": 16, "rows": 128}
            for layer in resolver.FROZEN_LINEAR_LAYERS
        ],
        "layer_quotas": {
            str(layer): quota
            for layer, quota in resolver.FROZEN_LAYER_QUOTAS.items()
        },
        "high_precision_rows": locations,
    }
    plan["canonical_plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(plan)
    ).hexdigest()
    return plan


def _loader_with_tracking(
    rows: list[dict[str, Any]],
    accesses: list[tuple[int, str]],
) -> Any:
    def loader(*args: Any, **kwargs: Any) -> tuple[_TrackedRow, ...]:
        assert args[:2] == (
            resolver.MBPP_DATASET_ID,
            resolver.MBPP_CONFIG,
        )
        assert kwargs == {
            "revision": resolver.MBPP_REVISION,
            "split": "train",
            "streaming": True,
        }
        return tuple(_TrackedRow(row, accesses) for row in rows)

    return loader


def _identity_evidence(
    rows: tuple[Mapping[str, Any], ...],
    records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    content_manifest = mbpp_manifest(rows, phase="calibration")
    content_hash = mbpp_manifest_content_sha256(content_manifest)
    source_hashes = {
        path: hashlib.sha256(path.encode()).hexdigest()
        for path in resolver.SOURCE_FILES
    }
    commit = "a" * 40
    return {
        "schema_version": 1,
        "artifact_kind": resolver.ARTIFACT_KIND,
        "identity_schema": resolver.IDENTITY_SCHEMA,
        "identity_only": True,
        "claim_boundary": resolver.CLAIM_BOUNDARY,
        "created_at_utc": "2026-07-26T00:00:00+00:00",
        "authorization": {
            "stage_a_artifact_kind": resolver.STAGE_A_ARTIFACT_KIND,
            "stage_a_file_sha256": resolver.STAGE_A_FILE_SHA256,
            "stage_a_canonical_evidence_sha256": (
                resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256
            ),
            "stage_a_implementation_commit": resolver.STAGE_A_IMPLEMENTATION_COMMIT,
            "stage_a_result_commit": resolver.STAGE_A_RESULT_COMMIT,
            "stage_a_gate_passed": True,
            "verified_before_dataset_access": True,
        },
        "row_plan": _compact_row_plan(),
        "model_contract": {
            "id": resolver.MODEL_ID,
            "revision": resolver.MODEL_REVISION,
            "weights_loaded": False,
        },
        "tokenizer_contract": {
            "source_id": resolver.MODEL_ID,
            "revision": resolver.MODEL_REVISION,
            "class": resolver.TOKENIZER_CLASS,
            "transformers_version": resolver.TRANSFORMERS_VERSION,
            "trust_remote_code": False,
            "prompt_add_special_tokens": True,
            "code_add_special_tokens": False,
            "formatter_version": resolver.MBPP_FORMATTER_VERSION,
            "token_id_hash_serialization": resolver.TOKEN_ID_HASH_SERIALIZATION,
            "text_hash_encoding": resolver.TEXT_HASH_ENCODING,
        },
        "dataset": {
            "id": resolver.MBPP_DATASET_ID,
            "config": resolver.MBPP_CONFIG,
            "revision": resolver.MBPP_REVISION,
            "phase": "calibration",
            "source_split": "train",
            "selection_namespace": resolver.MBPP_SELECTION_NAMESPACE,
            "formatter_version": resolver.MBPP_FORMATTER_VERSION,
            "selection_mode": "task_id_ranking_then_exact_task_id_stream",
            "selection_window": {
                "offset": resolver.STAGE_B_OFFSET,
                "limit": resolver.STAGE_B_LIMIT,
                "stop_exclusive": resolver.STAGE_B_STOP,
            },
            "protected_window": {
                "offset": resolver.PROTECTED_WINDOW[0],
                "stop_exclusive": resolver.PROTECTED_WINDOW[1],
                "content_retained_canonicalized_or_tokenized": False,
            },
            "ordered_task_ids": [int(row["task_id"]) for row in rows],
            "manifest": content_manifest,
            "content_manifest_sha256": content_hash,
            "token_manifest_sha256": resolver.token_manifest_sha256(records),
            "ordered_identity_sha256": resolver.ordered_identity_sha256(
                content_manifest_sha256=content_hash,
                task_records=records,
            ),
            "tasks": list(records),
            "totals": {
                "source_train_rows_seen_by_task_id_only": MBPP_CALIBRATION_SIZE,
                "retained_rows": resolver.STAGE_B_LIMIT,
                "prompt_tokens": sum(record["prompt_tokens"] for record in records),
                "code_tokens": sum(record["code_tokens"] for record in records),
                "aligned_scored_tokens": sum(
                    record["aligned_scored_tokens"] for record in records
                ),
                "full_code_scored_tokens": sum(
                    record["full_code_scored_tokens"] for record in records
                ),
            },
            "data_access": resolver.StageBDataAccessAudit(
                ranking_transport_records_yielded=MBPP_CALIBRATION_SIZE,
                ranking_task_id_fields_inspected=MBPP_CALIBRATION_SIZE,
                target_transport_records_yielded=MBPP_CALIBRATION_SIZE,
                target_task_id_fields_inspected=MBPP_CALIBRATION_SIZE,
                target_rows_retained_and_canonicalized=resolver.STAGE_B_LIMIT,
            ).as_dict(
                selected_task_ids=[int(row["task_id"]) for row in rows],
            ),
        },
        "integrity": {
            "stage_a_authenticated_before_dataset_access": True,
            "runtime_environment_authenticated_before_dataset_access": True,
            "selector_artifacts_authenticated_before_dataset_access": True,
            "repository_clean_at_start": True,
            "repository_clean_at_end": True,
            "repository_commit_stable": True,
            "source_hashes_stable": True,
            "task_id_only_ranking_pass": True,
            "only_stage_b_content_retained_canonicalized_and_tokenized": True,
            "imported_modules_resolved_to_authenticated_repository": True,
            "protected_window_8_16_content_retained_canonicalized_or_tokenized": False,
            "model_weights_loaded": False,
            "model_forward_pass_run": False,
            "logits_or_quality_metrics_observed": False,
            "output_path_external_or_git_ignored": True,
        },
        "repository": {
            "commit": commit,
            "start": {"commit": commit, "worktree_clean": True, "status": []},
            "end": {"commit": commit, "worktree_clean": True, "status": []},
            "stable_commit": True,
        },
        "source_files": {
            "paths": list(resolver.SOURCE_FILES),
            "sha256_start": source_hashes,
            "sha256_end": source_hashes,
            "stable": True,
            "imported_modules": dict(resolver.IMPORTED_MODULE_PATHS),
        },
        "environment": {
            "schema": resolver.RUNTIME_ENVIRONMENT_SCHEMA,
            "stage_a_binding": {
                "artifact_kind": resolver.STAGE_A_ARTIFACT_KIND,
                "file_sha256": resolver.STAGE_A_FILE_SHA256,
                "canonical_evidence_sha256": (
                    resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256
                ),
            },
            "python": {
                "major": resolver.STAGE_A_PYTHON_VERSION[0],
                "minor": resolver.STAGE_A_PYTHON_VERSION[1],
                "micro": resolver.STAGE_A_PYTHON_VERSION[2],
                "version": ".".join(
                    str(component)
                    for component in resolver.STAGE_A_PYTHON_VERSION
                ),
            },
            "packages": dict(resolver.STAGE_A_PACKAGE_VERSIONS),
            "cuda": dict(resolver.STAGE_A_CUDA_CONTRACT),
            "runtime_matches_stage_a": True,
            "local_files_only": True,
        },
    }


@pytest.fixture
def resolved_identity() -> tuple[
    list[dict[str, Any]],
    tuple[Mapping[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    source_rows = [_row(task_id) for task_id in range(1000, 1000 + MBPP_CALIBRATION_SIZE)]
    ranked = sorted(
        source_rows,
        key=lambda row: (mbpp_calibration_key(row["task_id"]), row["task_id"]),
    )
    rows = tuple(ranked[resolver.STAGE_B_OFFSET : resolver.STAGE_B_STOP])
    _, records = resolver.tokenize_stage_b_rows(Qwen2Tokenizer(), rows)
    evidence = _identity_evidence(rows, records)
    return source_rows, rows, records, evidence


def test_resolver_reads_only_task_id_outside_stage_b_window() -> None:
    source_rows = [_row(task_id) for task_id in range(1000, 1000 + MBPP_CALIBRATION_SIZE)]
    ranked = sorted(
        source_rows,
        key=lambda row: (mbpp_calibration_key(row["task_id"]), row["task_id"]),
    )
    protected_ids = {
        row["task_id"]
        for row in ranked[resolver.PROTECTED_WINDOW[0] : resolver.PROTECTED_WINDOW[1]]
    }
    expected_ids = tuple(
        row["task_id"] for row in ranked[resolver.STAGE_B_OFFSET : resolver.STAGE_B_STOP]
    )
    accesses: list[tuple[int, str]] = []

    resolved = resolver.resolve_stage_b_rows(
        load_dataset_fn=_loader_with_tracking(source_rows, accesses)
    )

    assert resolved.ordered_task_ids == expected_ids
    assert tuple(int(row["task_id"]) for row in resolved.rows) == expected_ids
    assert (
        resolved.access_audit.ranking_transport_records_yielded
        == MBPP_CALIBRATION_SIZE
    )
    assert all(key == "task_id" for task_id, key in accesses if task_id in protected_ids)
    content_access_ids = {task_id for task_id, key in accesses if key != "task_id"}
    assert content_access_ids == set(expected_ids)
    _, token_records = resolver.tokenize_stage_b_rows(
        Qwen2Tokenizer(),
        resolved.rows,
    )
    assert {record["task_id"] for record in token_records}.isdisjoint(protected_ids)
    audit = resolved.access_audit.as_dict(selected_task_ids=resolved.ordered_task_ids)
    for name in ("selected", "retained", "canonicalized", "formatted", "tokenized"):
        assert set(audit["application_task_id_sets"][name]).isdisjoint(protected_ids)
        assert audit["protected_window_intersection"][name] is False
    assert audit["application_task_id_sets"]["passed_to_model"] == []
    assert audit["application_task_id_sets"]["evaluated"] == []


def test_resolver_rejects_duplicate_or_undersized_source() -> None:
    rows = [_row(task_id) for task_id in range(MBPP_CALIBRATION_SIZE)]
    duplicate_rows = [*rows, _row(0)]
    with pytest.raises(ValueError, match="duplicate MBPP task_id"):
        resolver.resolve_stage_b_task_ids(
            load_dataset_fn=_loader_with_tracking(duplicate_rows, [])
        )
    with pytest.raises(ValueError, match="too small"):
        resolver.resolve_stage_b_task_ids(
            load_dataset_fn=_loader_with_tracking(rows[:-1], [])
        )


def test_id_only_ranking_matches_full_selector_on_shuffled_large_source() -> None:
    source_rows = [_row(task_id) for task_id in range(2000, 2500)]
    random.Random(2339).shuffle(source_rows)
    expected = select_mbpp_calibration(
        source_rows,
        size=MBPP_CALIBRATION_SIZE,
    )[resolver.STAGE_B_OFFSET : resolver.STAGE_B_STOP]

    actual, source_count = resolver.resolve_stage_b_task_ids(
        load_dataset_fn=_loader_with_tracking(source_rows, [])
    )

    assert source_count == 500
    assert actual == tuple(int(row["task_id"]) for row in expected)


def test_token_records_bind_row_text_and_exact_token_ids(
    resolved_identity: tuple[
        list[dict[str, Any]],
        tuple[Mapping[str, Any], ...],
        tuple[dict[str, Any], ...],
        dict[str, Any],
    ],
) -> None:
    _, rows, records, _ = resolved_identity
    encoded, repeated = resolver.tokenize_stage_b_rows(Qwen2Tokenizer(), rows)

    assert repeated == records
    assert len(encoded) == resolver.STAGE_B_LIMIT
    assert [record["rank"] for record in records] == list(
        range(resolver.STAGE_B_OFFSET, resolver.STAGE_B_STOP)
    )
    assert all(
        record["aligned_scored_tokens"] == record["code_tokens"] - 1
        for record in records
    )
    assert all(record["full_code_scored_tokens"] == record["code_tokens"] for record in records)
    assert resolver.token_manifest_sha256(records) == resolver.token_manifest_sha256(repeated)


def test_identity_validator_and_runtime_authentication_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    resolved_identity: tuple[
        list[dict[str, Any]],
        tuple[Mapping[str, Any], ...],
        tuple[dict[str, Any], ...],
        dict[str, Any],
    ],
) -> None:
    source_rows, rows, _, evidence = resolved_identity
    resolver.validate_stage_b_identity_evidence(evidence)
    monkeypatch.setattr(
        resolver.importlib.metadata,
        "version",
        lambda name: resolver.TRANSFORMERS_VERSION,
    )
    accesses: list[tuple[int, str]] = []
    loaded = resolver.load_authenticated_stage_b_rows(
        evidence,
        load_dataset_fn=_loader_with_tracking(source_rows, accesses),
    )
    authentication = resolver.authenticate_stage_b_runtime_identity(
        evidence,
        loaded,
        Qwen2Tokenizer(),
    )
    assert len(authentication.tasks) == resolver.STAGE_B_LIMIT
    assert tuple(task.record for task in authentication.tasks) == tuple(
        evidence["dataset"]["tasks"]
    )
    assert loaded.ordered_task_ids == tuple(evidence["dataset"]["ordered_task_ids"])
    assert authentication.access_audit == loaded.access_audit
    assert authentication.ordered_identity_sha256 == evidence["dataset"][
        "ordered_identity_sha256"
    ]


def test_runtime_rejects_coherent_wrong_rank_identity_before_content_access() -> None:
    source_rows = [
        _row(task_id) for task_id in range(1000, 1000 + MBPP_CALIBRATION_SIZE)
    ]
    ranked = sorted(
        source_rows,
        key=lambda row: (mbpp_calibration_key(row["task_id"]), row["task_id"]),
    )
    wrong_rows = tuple(ranked[: resolver.STAGE_B_LIMIT])
    _, wrong_records = resolver.tokenize_stage_b_rows(
        Qwen2Tokenizer(),
        wrong_rows,
    )
    coherent_wrong_evidence = _identity_evidence(wrong_rows, wrong_records)
    accesses: list[tuple[int, str]] = []

    with pytest.raises(ValueError, match=r"ranked window \[32, 64\)"):
        resolver.load_authenticated_stage_b_rows(
            coherent_wrong_evidence,
            load_dataset_fn=_loader_with_tracking(source_rows, accesses),
        )

    assert accesses
    assert all(key == "task_id" for _, key in accesses)


def test_imported_module_paths_must_match_authenticated_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = _resolver_runtime_modules()
    assert resolver.validate_imported_module_paths(
        REPOSITORY_ROOT,
        modules=modules,
    ) == dict(resolver.IMPORTED_MODULE_PATHS)
    evidence_module = resolver.sys.modules["recurquant.evidence"]
    monkeypatch.setattr(
        evidence_module,
        "__file__",
        str(tmp_path / "untrusted" / "evidence.py"),
    )

    with pytest.raises(ValueError, match="did not resolve"):
        resolver.validate_imported_module_paths(
            REPOSITORY_ROOT,
            modules=modules,
        )


@pytest.mark.skipif(
    not (
        REPOSITORY_ROOT / resolver.SELECTOR_ARTIFACT_RELATIVE_PATH
    ).is_file()
    or not (
        REPOSITORY_ROOT / resolver.LOSS_SELECTOR_ARTIFACT_RELATIVE_PATH
    ).is_file(),
    reason="original frozen selector artifacts are not present in this clone",
)
def test_exact_selector_pair_builds_a_complete_public_compact_plan() -> None:
    plan = resolver.build_compact_row_plan(
        REPOSITORY_ROOT / resolver.SELECTOR_ARTIFACT_RELATIVE_PATH,
        REPOSITORY_ROOT / resolver.LOSS_SELECTOR_ARTIFACT_RELATIVE_PATH,
    )

    resolver.validate_compact_row_plan(plan)
    assert len(plan["high_precision_rows"]) == 1_976
    assert plan["accounting"]["resident_bytes"] == 2_564_096
    assert plan["selector_binding"]["selector_file_sha256"] == (
        resolver.SELECTOR_FILE_SHA256
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["dataset"]["selection_window"].update({"offset": 8}),
            "ranked window",
        ),
        (
            lambda evidence: evidence["integrity"].update(
                {"protected_window_8_16_content_retained_canonicalized_or_tokenized": True}
            ),
            "forbidden",
        ),
        (
            lambda evidence: evidence["model_contract"].update({"weights_loaded": True}),
            "model contract",
        ),
        (
            lambda evidence: evidence.update(
                {
                    "claim_boundary": (
                        "This is not performance evidence. It proves a breakthrough."
                    )
                }
            ),
            "claim boundary",
        ),
        (
            lambda evidence: evidence["row_plan"]["high_precision_rows"][0].update(
                {"row_index": 127}
            ),
            "row.plan",
        ),
        (
            lambda evidence: evidence["dataset"]["tasks"][0].update({"prompt_tokens": 1}),
            "token manifest hash",
        ),
        (
            lambda evidence: evidence["dataset"]["manifest"]["rows"][0].update(
                {"sha256": "0" * 64}
            ),
            "content manifest hash",
        ),
        (
            lambda evidence: evidence["environment"]["python"].update({"micro": 14}),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"]["packages"].update(
                {"numpy": "2.4.5"}
            ),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"]["packages"].pop("safetensors"),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"]["cuda"].update(
                {"runtime_version": "12.7"}
            ),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"]["cuda"].update(
                {"available": False, "device_type": "cpu"}
            ),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"]["stage_a_binding"].update(
                {"file_sha256": "0" * 64}
            ),
            "runtime environment contract",
        ),
        (
            lambda evidence: evidence["environment"].update(
                {"runtime_matches_stage_a": False}
            ),
            "runtime environment contract",
        ),
    ],
)
def test_identity_validator_fails_closed_on_contract_drift(
    resolved_identity: tuple[
        list[dict[str, Any]],
        tuple[Mapping[str, Any], ...],
        tuple[dict[str, Any], ...],
        dict[str, Any],
    ],
    mutation: Any,
    message: str,
) -> None:
    evidence = copy.deepcopy(resolved_identity[3])
    mutation(evidence)
    with pytest.raises(ValueError, match=message):
        resolver.validate_stage_b_identity_evidence(evidence)


def test_artifact_loader_rejects_outer_or_internal_tampering(
    tmp_path: Path,
    resolved_identity: tuple[
        list[dict[str, Any]],
        tuple[Mapping[str, Any], ...],
        tuple[dict[str, Any], ...],
        dict[str, Any],
    ],
) -> None:
    evidence = resolved_identity[3]
    artifact = {
        "canonical_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest(),
        "evidence": evidence,
    }
    path = tmp_path / "identity.json"
    path.write_bytes(canonical_json_bytes(artifact))
    loaded, file_hash = resolver.load_stage_b_identity_artifact(path)
    assert loaded == evidence
    assert file_hash == hashlib.sha256(path.read_bytes()).hexdigest()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["evidence"]["dataset"]["tasks"][0]["task_id"] += 1
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="canonical verification"):
        resolver.load_stage_b_identity_artifact(path)


def test_stage_a_artifact_is_the_exact_authenticated_pass() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    stage_a_path = repository_root / resolver.STAGE_A_ARTIFACT_RELATIVE_PATH
    evidence, file_hash = resolver.authenticate_stage_a_artifact(stage_a_path)

    assert file_hash == resolver.STAGE_A_FILE_SHA256
    assert evidence["stage_a_gate"]["passed"] is True
    assert (
        evidence["dataset"]["protected_window_8_16_loaded_tokenized_or_evaluated"]
        is False
    )


def test_runtime_environment_is_derived_from_authenticated_stage_a() -> None:
    stage_a_path = REPOSITORY_ROOT / resolver.STAGE_A_ARTIFACT_RELATIVE_PATH
    stage_a_evidence, _ = resolver.authenticate_stage_a_artifact(stage_a_path)
    derived = resolver.derive_stage_a_runtime_environment(stage_a_evidence)

    assert derived == {
        "python": {
            "major": 3,
            "minor": 11,
            "micro": 15,
            "version": "3.11.15",
        },
        "packages": {
            "datasets": "4.8.5",
            "numpy": "2.4.6",
            "safetensors": "0.8.0",
            "torch": "2.11.0+cu128",
            "transformers": "5.14.1",
        },
        "cuda": {
            "available": True,
            "runtime_version": "12.8",
            "device_type": "cuda",
        },
    }
    contract = resolver.authenticate_runtime_environment(
        stage_a_evidence,
        local_files_only=True,
        runtime_environment=copy.deepcopy(derived),
    )
    assert contract["runtime_matches_stage_a"] is True
    assert contract["stage_a_binding"]["file_sha256"] == resolver.STAGE_A_FILE_SHA256
    assert "gpu" not in contract
    assert "platform" not in contract
    assert "executable" not in contract


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime["python"].update({"micro": 14}),
        lambda runtime: runtime["packages"].update({"datasets": "4.8.4"}),
        lambda runtime: runtime["packages"].update({"numpy": "2.4.5"}),
        lambda runtime: runtime["packages"].pop("safetensors"),
        lambda runtime: runtime["packages"].update({"torch": "2.11.0"}),
        lambda runtime: runtime["packages"].update({"transformers": "5.14.0"}),
        lambda runtime: runtime["cuda"].update({"runtime_version": "12.7"}),
        lambda runtime: runtime["cuda"].update(
            {"available": False, "device_type": "cpu"}
        ),
        lambda runtime: runtime["cuda"].update({"device_type": "cpu"}),
        lambda runtime: runtime.update({"unexpected_path": "C:/private/python.exe"}),
    ],
)
def test_runtime_environment_authentication_rejects_any_drift(
    mutation: Any,
) -> None:
    stage_a_path = REPOSITORY_ROOT / resolver.STAGE_A_ARTIFACT_RELATIVE_PATH
    stage_a_evidence, _ = resolver.authenticate_stage_a_artifact(stage_a_path)
    runtime = resolver.derive_stage_a_runtime_environment(stage_a_evidence)
    mutation(runtime)

    with pytest.raises(ValueError, match="does not exactly match"):
        resolver.authenticate_runtime_environment(
            stage_a_evidence,
            local_files_only=True,
            runtime_environment=runtime,
        )


def test_runtime_environment_rejects_stage_a_contract_drift() -> None:
    stage_a_path = REPOSITORY_ROOT / resolver.STAGE_A_ARTIFACT_RELATIVE_PATH
    stage_a_evidence, _ = resolver.authenticate_stage_a_artifact(stage_a_path)
    runtime = resolver.derive_stage_a_runtime_environment(stage_a_evidence)
    drifted_stage_a = copy.deepcopy(stage_a_evidence)
    drifted_stage_a["environment"]["packages"]["numpy"] = "2.4.5"

    with pytest.raises(ValueError, match="authenticated Stage-A runtime"):
        resolver.authenticate_runtime_environment(
            drifted_stage_a,
            local_files_only=True,
            runtime_environment=runtime,
        )


def test_output_path_must_be_external_or_git_ignored(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    resolver.validate_output_path(tmp_path / "identity.json", repository_root)
    resolver.validate_output_path(
        repository_root / "artifacts" / "identity.json",
        repository_root,
    )
    with pytest.raises(ValueError, match="must be Git-ignored"):
        resolver.validate_output_path(
            repository_root / "research" / "identity.json",
            repository_root,
        )
