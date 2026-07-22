from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from scripts import pilot_evaluate_hrr as evaluator
from scripts import pilot_hrr_rows, pilot_loss_sensitivity_rows
from scripts import pilot_validate_storage_boundary as storage_validator
from tests.test_transformers_cache import tiny_config


def _quantizers() -> dict[str, object]:
    common = {
        "group_size": 128,
        "scale_bits": 16,
        "flatten_last_dims": 1,
        "rounding": "nearest",
        "seed": 2339,
        "epsilon": 1e-12,
    }
    return {
        "axis_contract": "one independent group per recurrent [head, key-row]",
        "int4": {"bits": 4, **common},
        "int8": {"bits": 8, **common},
    }


def _selector(kind: str) -> dict[str, object]:
    manifest = {
        "schema": "recurquant.mbpp-manifest.v1",
        "dataset_id": "google-research-datasets/mbpp",
        "config": "full",
        "revision": "dataset-revision",
        "phase": "calibration",
        "source_split": "train",
        "selection_namespace": "rq-v0.2",
        "formatter_version": "recurquant.mbpp-prompt-code.v1",
        "row_count": 2,
        "rows": [
            {"task_id": 10, "sha256": "row-10"},
            {"task_id": 20, "sha256": "row-20"},
        ],
    }
    affected_field = (
        "captured_decode_tokens" if kind == evaluator.HRR_ARTIFACT_KIND else "scored_transitions"
    )
    method: dict[str, object] = {}
    if kind == evaluator.HRR_ARTIFACT_KIND:
        method["normalization_epsilon"] = 1e-6
    return {
        "artifact_kind": kind,
        "seed": 2339,
        "model": {
            "id": "model",
            "revision": "model-revision",
            "dtype": "torch.bfloat16",
            "linear_attention_layers": [0, 2],
        },
        "dataset": {
            "manifest": manifest,
            "manifest_sha256": evaluator.mbpp_manifest_content_sha256(manifest),
            "tasks": [
                {
                    "task_id": 20,
                    "prompt_tokens": 7,
                    "code_tokens": 4,
                    affected_field: 3,
                },
                {
                    "task_id": 10,
                    "prompt_tokens": 9,
                    "code_tokens": 3,
                    affected_field: 2,
                },
            ],
        },
        "method": method,
        "quantizers": _quantizers(),
        "byte_budget": {
            "target_resident_bytes": 1000,
            "group_size": 128,
            "scale_bits": 16,
            "precision_mask_bits_per_group": 1,
        },
    }


@pytest.mark.parametrize(
    "git_state",
    (
        pilot_hrr_rows.git_state,
        pilot_loss_sensitivity_rows.git_state,
        storage_validator._git_state,
    ),
)
def test_artifact_generators_emit_evaluator_repository_contract(git_state) -> None:
    repository = git_state()

    assert {"commit", "worktree_clean", "status"} <= repository.keys()
    assert repository["worktree_clean"] is (repository["status"] == [])


@pytest.mark.parametrize(
    ("generator", "kind", "affected_field"),
    (
        (pilot_hrr_rows, evaluator.HRR_ARTIFACT_KIND, "captured_decode_tokens"),
        (
            pilot_loss_sensitivity_rows,
            evaluator.LOSS_ARTIFACT_KIND,
            "scored_transitions",
        ),
    ),
)
def test_generator_manifest_hash_contract_is_accepted_and_content_authenticated(
    generator,
    kind: str,
    affected_field: str,
) -> None:
    rows = (
        {
            "task_id": 10,
            "text": "Return ten.",
            "code": "def answer():\n    return 10",
            "test_list": ["assert answer() == 10"],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 20,
            "text": "Return twenty.",
            "code": "def answer():\n    return 20",
            "test_list": ["assert answer() == 20"],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
    )
    manifest = generator.mbpp_manifest(rows, phase="calibration")
    recorded_hash = generator.mbpp_manifest_sha256(rows, phase="calibration")
    selector = _selector(kind)
    selector["dataset"] = {
        "manifest": manifest,
        "manifest_sha256": recorded_hash,
        "tasks": [
            {
                "task_id": row["task_id"],
                "prompt_tokens": 8,
                "code_tokens": 4,
                affected_field: 3,
            }
            for row in rows
        ],
    }

    evaluator.validate_selector_contract(selector)
    assert recorded_hash == evaluator.mbpp_manifest_content_sha256(manifest)

    selector["dataset"]["manifest"]["rows"][0]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="manifest hash"):
        evaluator.validate_selector_contract(selector)


def _frozen_selector(kind: str, *, commit: str = "frozen-commit") -> dict[str, object]:
    selector = _selector(kind)
    affected_field = (
        "captured_decode_tokens" if kind == evaluator.HRR_ARTIFACT_KIND else "scored_transitions"
    )
    ordered_ids = list(range(100, 108))
    tasks = [
        {
            "task_id": task_id,
            "prompt_tokens": 10 + index,
            "code_tokens": 4 + index,
            affected_field: 3 + index,
        }
        for index, task_id in enumerate(ordered_ids)
    ]
    manifest = {
        **selector["dataset"]["manifest"],
        "row_count": 8,
        "rows": [
            {"task_id": task_id, "sha256": f"row-{task_id}"} for task_id in sorted(ordered_ids)
        ],
    }
    selector["dataset"] = {
        "manifest": manifest,
        "manifest_sha256": evaluator.mbpp_manifest_content_sha256(manifest),
        "tasks": tasks,
    }
    selector["repository"] = {
        "commit": commit,
        "worktree_clean": True,
        "status": [],
    }
    if kind == evaluator.HRR_ARTIFACT_KIND:
        selector["method"]["horizon"] = 32
    return selector


def _passing_gate_inputs() -> dict[str, object]:
    primary = evaluator.ADAPTIVE_TARGET_FISHER
    adaptive_h1 = evaluator.ADAPTIVE_H1
    aggregates = {
        primary: {
            "macro_delta_nll": 0.60,
            "macro_top1_agreement": 0.900,
            "macro_cvar95_kl": 0.20,
        },
        adaptive_h1: {
            "macro_delta_nll": 0.90,
            "macro_top1_agreement": 0.895,
            "macro_cvar95_kl": 0.22,
        },
        "hrr_h1": {
            "macro_delta_nll": 1.00,
            "macro_top1_agreement": 0.905,
            "macro_cvar95_kl": 0.15,
        },
        "row_mse": {
            "macro_delta_nll": 1.20,
            "macro_top1_agreement": 0.89,
            "macro_cvar95_kl": 0.30,
        },
        "uniform_int4": {
            "macro_delta_nll": 2.00,
            "macro_top1_agreement": 0.80,
            "macro_cvar95_kl": 0.50,
        },
    }
    per_task = {
        primary: [
            {"task_id": 1, "delta_nll": 0.4},
            {"task_id": 2, "delta_nll": 0.8},
        ],
        adaptive_h1: [
            {"task_id": 1, "delta_nll": 0.8},
            {"task_id": 2, "delta_nll": 1.0},
        ],
        "hrr_h1": [
            {"task_id": 1, "delta_nll": 0.5},
            {"task_id": 2, "delta_nll": 1.5},
        ],
    }
    contrasts = {
        adaptive_h1: {"confidence_interval": [0.05, 0.50]},
        "hrr_h1": {"confidence_interval": [0.10, 0.70]},
    }
    storage = {
        primary: {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES},
        adaptive_h1: {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES},
        "hrr_h1": {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES},
        "row_mse": {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES},
        "uniform_int4": {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES - 1},
    }
    for index, name in enumerate(evaluator.FROZEN_STATIC_COMPARATORS, start=1):
        aggregates.setdefault(
            name,
            {
                "macro_delta_nll": 1.20 + index / 100,
                "macro_top1_agreement": 0.89,
                "macro_cvar95_kl": 0.30,
            },
        )
        storage.setdefault(name, {"resident_bytes": evaluator.TARGET_RESIDENT_BYTES})
    return {
        "aggregates": aggregates,
        "per_task": per_task,
        "contrasts": contrasts,
        "storage": storage,
        "primary_name": primary,
    }


def _storage_boundary_evidence() -> dict[str, object]:
    repository = {"commit": "frozen-commit", "worktree_clean": True, "status": []}
    source_hashes = {"scripts/pilot_validate_storage_boundary.py": "abc123"}
    return {
        "artifact_kind": evaluator.STORAGE_BOUNDARY_ARTIFACT_KIND,
        "model": {
            "id": "model",
            "revision": "model-revision",
            "dtype": "torch.float32",
        },
        "derivative_gate": {
            "passed": True,
            "failures": [],
            "thresholds": {
                "model_dtype": "torch.float32",
                "baseline_repeat_absolute_tolerance": 1e-7,
                "derivative_informative_floor": 1e-8,
                "near_zero_absolute_tolerance": 2e-7,
                "minimum_informative_rows": 3,
                "minimum_sign_agreement": 0.95,
                "maximum_median_relative_error": 0.10,
                "minimum_converged_row_fraction": 0.75,
            },
            "observed": {
                "rows": 4,
                "informative_rows": 4,
                "maximum_baseline_repeat_absolute_error": 1e-8,
                "sign_agreement": 1.0,
                "median_relative_error": 0.05,
                "converged_row_fraction": 1.0,
                "near_zero_checks": 0,
                "near_zero_checks_passed": 0,
            },
        },
        "implementation": {
            "source_hashes_start": source_hashes,
            "source_hashes_end": source_hashes,
            "unchanged_during_run": True,
        },
        "repository": {"start": repository, "end": repository},
    }


def test_scores_from_artifact_verifies_hash_and_preserves_float64() -> None:
    arrays = {"0": [[1.0000000000000002, -2.5]]}
    evidence = {
        "scores": {
            "candidate": {
                "arrays": arrays,
                "canonical_arrays_sha256": evaluator.sha256_bytes(
                    evaluator.canonical_json_bytes(arrays)
                ),
            }
        }
    }

    scores = evaluator.scores_from_artifact(evidence, "candidate")

    assert scores[0].dtype == torch.float64
    assert scores[0][0, 0].item() == 1.0000000000000002

    evidence["scores"]["candidate"]["arrays"]["0"][0][0] = 9.0
    with pytest.raises(ValueError, match="array hash"):
        evaluator.scores_from_artifact(evidence, "candidate")


def test_compatible_selectors_compare_full_evidence_contract() -> None:
    hrr = _selector(evaluator.HRR_ARTIFACT_KIND)
    loss = _selector(evaluator.LOSS_ARTIFACT_KIND)

    evaluator.validate_compatible_selector(hrr, loss)

    incompatible = deepcopy(loss)
    incompatible["quantizers"]["int4"]["epsilon"] = 1e-9
    incompatible["quantizers"]["int8"]["epsilon"] = 1e-9
    with pytest.raises(ValueError, match="quantizer contracts"):
        evaluator.validate_compatible_selector(hrr, incompatible)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["model"].update(dtype="torch.float32"), "dtype"),
        (lambda value: value["model"].update(linear_attention_layers=[0, 3]), "layers"),
        (
            lambda value: value["dataset"]["tasks"][0].update(
                prompt_tokens=8,
            ),
            "token manifests",
        ),
    ],
)
def test_compatible_selectors_reject_provenance_mismatch(mutation, message: str) -> None:
    hrr = _selector(evaluator.HRR_ARTIFACT_KIND)
    loss = _selector(evaluator.LOSS_ARTIFACT_KIND)
    mutation(loss)

    with pytest.raises(ValueError, match=message):
        evaluator.validate_compatible_selector(hrr, loss)


def test_selector_requires_explicit_quantizer_metadata() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    del selector["quantizers"]

    with pytest.raises(ValueError, match="selector quantizers"):
        evaluator.validate_selector_contract(selector)


def test_calibration_prefix_verifies_authenticated_row_content() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    full_manifest = selector["dataset"]["manifest"]
    actual_manifest = {
        **{key: value for key, value in full_manifest.items() if key not in ("row_count", "rows")},
        "row_count": 1,
        "rows": [{"task_id": 20, "sha256": "row-20"}],
    }

    evaluator.validate_calibration_prefix(
        selector,
        actual_manifest=actual_manifest,
        expected_task_ids=[20],
    )

    actual_manifest["rows"][0]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="content manifest"):
        evaluator.validate_calibration_prefix(
            selector,
            actual_manifest=actual_manifest,
            expected_task_ids=[20],
        )


def test_calibration_window_slices_disjoint_ranked_holdout() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    ranked_rows = tuple({"task_id": task_id} for task_id in (10, 20, 30, 40))

    selected = evaluator.select_calibration_window(
        ranked_rows,
        offset=2,
        limit=2,
        selectors=[selector],
    )

    assert [row["task_id"] for row in selected] == [30, 40]


def test_calibration_window_offset_zero_retains_selector_prefix() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    ranked_rows = tuple({"task_id": task_id} for task_id in (20, 10, 30))

    selected = evaluator.select_calibration_window(
        ranked_rows,
        offset=0,
        limit=2,
        selectors=[selector],
    )

    assert [row["task_id"] for row in selected] == [20, 10]


def test_calibration_holdout_refuses_any_selector_overlap() -> None:
    hrr = _selector(evaluator.HRR_ARTIFACT_KIND)
    loss = _selector(evaluator.LOSS_ARTIFACT_KIND)
    ranked_rows = tuple({"task_id": task_id} for task_id in (99, 20, 30))

    with pytest.raises(ValueError, match="overlap selector artifact tasks: 20"):
        evaluator.select_calibration_window(
            ranked_rows,
            offset=1,
            limit=2,
            selectors=[hrr, loss],
        )


def test_calibration_holdout_refuses_out_of_range_window() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    ranked_rows = tuple({"task_id": task_id} for task_id in (10, 20, 30))

    with pytest.raises(ValueError, match=r"window \[2:4\] exceeds 3"):
        evaluator.select_calibration_window(
            ranked_rows,
            offset=2,
            limit=2,
            selectors=[selector],
        )


def test_legacy_frozen_holdout_is_locked_after_experiment008_freeze() -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]

    with pytest.raises(ValueError, match="protected by Experiment 008"):
        evaluator.validate_frozen_holdout_request(
            offset=8,
            limit=8,
            bootstrap_samples=10_000,
            selectors=selectors,
            loss_selector_present=True,
            storage_boundary_present=True,
        )


def test_experiment006_rank_fusion_holdout_remains_fail_closed() -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]

    with pytest.raises(ValueError, match="rank-fusion holdout remains closed"):
        evaluator.validate_frozen_holdout_request(
            offset=8,
            limit=8,
            bootstrap_samples=10_000,
            selectors=selectors,
            loss_selector_present=True,
            storage_boundary_present=True,
            rank_fusion_enabled=True,
        )


def test_experiment007_query_ema_holdout_remains_fail_closed() -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]

    with pytest.raises(ValueError, match="CQER-32 holdout remains closed"):
        evaluator.validate_frozen_holdout_request(
            offset=8,
            limit=8,
            bootstrap_samples=10_000,
            selectors=selectors,
            loss_selector_present=True,
            storage_boundary_present=True,
            query_ema_enabled=True,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"offset": 1},
        {"limit": 7},
        {"bootstrap_samples": 9999},
    ],
)
def test_experiment007_development_request_refuses_partial_or_drifted_runs(
    changes: dict[str, int],
) -> None:
    arguments = {
        "enabled": True,
        "offset": 0,
        "limit": 8,
        "bootstrap_samples": 10_000,
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match="offset 0, exactly 8 tasks"):
        evaluator.validate_cqer_development_request(**arguments)

    evaluator.validate_cqer_development_request(
        enabled=False,
        offset=99,
        limit=1,
        bootstrap_samples=1,
    )


def test_experiment007_requires_exact_eight_task_selector_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]
    monkeypatch.setattr(
        evaluator,
        "CQER_DEVELOPMENT_TASK_IDS",
        tuple(range(100, 108)),
    )
    monkeypatch.setattr(
        evaluator,
        "CQER_FROZEN_SELECTOR_CANONICAL_SHA256S",
        tuple(
            evaluator.sha256_bytes(evaluator.canonical_json_bytes(selector))
            for selector in selectors
        ),
    )
    evaluator.validate_cqer_selector_artifacts(enabled=True, selectors=selectors)

    selectors[1]["dataset"]["tasks"].append(
        {
            "task_id": 999,
            "prompt_tokens": 1,
            "code_tokens": 2,
            "scored_transitions": 1,
        }
    )
    with pytest.raises(ValueError, match="eight-task selector artifacts"):
        evaluator.validate_cqer_selector_artifacts(enabled=True, selectors=selectors)


def test_experiment007_rejects_selector_content_with_drifted_canonical_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]
    monkeypatch.setattr(
        evaluator,
        "CQER_DEVELOPMENT_TASK_IDS",
        tuple(range(100, 108)),
    )
    monkeypatch.setattr(
        evaluator,
        "CQER_FROZEN_SELECTOR_CANONICAL_SHA256S",
        tuple(
            evaluator.sha256_bytes(evaluator.canonical_json_bytes(selector))
            for selector in selectors
        ),
    )
    selectors[1]["byte_budget"]["target_resident_bytes"] = 999

    with pytest.raises(ValueError, match="canonical hashes"):
        evaluator.validate_cqer_selector_artifacts(enabled=True, selectors=selectors)


def test_experiment007_pins_ordered_task_ids_and_layer_quotas() -> None:
    evaluator.validate_cqer_development_task_ids(
        enabled=True,
        task_ids=evaluator.CQER_DEVELOPMENT_TASK_IDS,
    )
    evaluator.validate_cqer_layer_quotas(
        enabled=True,
        quotas=dict(evaluator.CQER_FROZEN_LAYER_QUOTAS),
    )

    with pytest.raises(ValueError, match="frozen ordered prefix"):
        evaluator.validate_cqer_development_task_ids(
            enabled=True,
            task_ids=tuple(reversed(evaluator.CQER_DEVELOPMENT_TASK_IDS)),
        )
    drifted = dict(evaluator.CQER_FROZEN_LAYER_QUOTAS)
    drifted[0] -= 1
    drifted[1] += 1
    with pytest.raises(ValueError, match="frozen target-Fisher allocation"):
        evaluator.validate_cqer_layer_quotas(enabled=True, quotas=drifted)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"offset": 9}, "exact ranked window"),
        ({"limit": 7}, "exact ranked window"),
        ({"bootstrap_samples": 9999}, "exactly 10000"),
        ({"loss_selector_present": False}, "requires both HRR and loss"),
        ({"storage_boundary_present": False}, "storage-boundary artifact"),
    ],
)
def test_frozen_holdout_request_refuses_protocol_drift(changes, message: str) -> None:
    arguments = {
        "offset": 8,
        "limit": 8,
        "bootstrap_samples": 10_000,
        "selectors": [
            _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
            _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
        ],
        "loss_selector_present": True,
        "storage_boundary_present": True,
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        evaluator.validate_frozen_holdout_request(**arguments)


def test_frozen_holdout_request_refuses_non_eight_task_selector() -> None:
    hrr = _frozen_selector(evaluator.HRR_ARTIFACT_KIND)
    loss = _frozen_selector(evaluator.LOSS_ARTIFACT_KIND)
    loss["dataset"]["tasks"].pop()

    with pytest.raises(ValueError, match="exactly 8 tasks"):
        evaluator.validate_frozen_holdout_request(
            offset=8,
            limit=8,
            bootstrap_samples=10_000,
            selectors=[hrr, loss],
            loss_selector_present=True,
            storage_boundary_present=True,
        )


def test_frozen_holdout_request_refuses_non_h32_selector() -> None:
    hrr = _frozen_selector(evaluator.HRR_ARTIFACT_KIND)
    loss = _frozen_selector(evaluator.LOSS_ARTIFACT_KIND)
    hrr["method"]["horizon"] = 64

    with pytest.raises(ValueError, match="requires HRR horizon 32"):
        evaluator.validate_frozen_holdout_request(
            offset=8,
            limit=8,
            bootstrap_samples=10_000,
            selectors=[hrr, loss],
            loss_selector_present=True,
            storage_boundary_present=True,
        )


def test_authenticate_selector_prefix_checks_order_and_row_content() -> None:
    selector = _frozen_selector(evaluator.HRR_ARTIFACT_KIND)
    manifest = selector["dataset"]["manifest"]
    ordered_ids = [record["task_id"] for record in selector["dataset"]["tasks"]]

    evaluator.authenticate_selector_prefix(
        selector,
        ranked_prefix_manifest=manifest,
        ranked_prefix_task_ids=ordered_ids,
    )

    with pytest.raises(ValueError, match="ordered task IDs"):
        evaluator.authenticate_selector_prefix(
            selector,
            ranked_prefix_manifest=manifest,
            ranked_prefix_task_ids=list(reversed(ordered_ids)),
        )

    tampered = deepcopy(manifest)
    tampered["rows"][0]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="content manifest"):
        evaluator.authenticate_selector_prefix(
            selector,
            ranked_prefix_manifest=tampered,
            ranked_prefix_task_ids=ordered_ids,
        )


def test_frozen_selector_token_manifest_matches_every_prefix_task() -> None:
    selector = _frozen_selector(evaluator.HRR_ARTIFACT_KIND)
    records = [
        {
            "task_id": record["task_id"],
            "prompt_tokens": record["prompt_tokens"],
            "code_tokens": record["code_tokens"],
            "aligned_scored_tokens": record["captured_decode_tokens"],
            "full_code_scored_tokens": record["code_tokens"],
        }
        for record in selector["dataset"]["tasks"]
    ]

    evaluator.validate_actual_token_manifest(selector, records)

    records[-1]["code_tokens"] += 1
    with pytest.raises(ValueError, match="formatter/token manifest"):
        evaluator.validate_actual_token_manifest(selector, records)


def test_heldout_repository_requires_clean_matching_selector_commit() -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]
    repository = {"commit": "frozen-commit", "worktree_clean": True, "status": []}

    evaluator.validate_heldout_repository_start(repository, selectors)

    dirty = {**repository, "worktree_clean": False, "status": [" M source.py"]}
    with pytest.raises(ValueError, match="clean worktree"):
        evaluator.validate_heldout_repository_start(dirty, selectors)

    selectors[1]["repository"]["commit"] = "different"
    with pytest.raises(ValueError, match="commit does not match"):
        evaluator.validate_heldout_repository_start(repository, selectors)


def test_heldout_repository_end_refuses_commit_or_source_drift() -> None:
    clean = {"commit": "abc", "worktree_clean": True, "status": []}
    hashes = {"source.py": "hash"}

    evaluator.validate_heldout_repository_end(
        start_repository=clean,
        end_repository=clean,
        start_source_hashes=hashes,
        end_source_hashes=hashes,
    )

    with pytest.raises(RuntimeError, match="source files changed"):
        evaluator.validate_heldout_repository_end(
            start_repository=clean,
            end_repository=clean,
            start_source_hashes=hashes,
            end_source_hashes={"source.py": "changed"},
        )


def test_storage_boundary_prerequisite_recomputes_numeric_gate_and_provenance() -> None:
    selector = _frozen_selector(evaluator.HRR_ARTIFACT_KIND)
    evidence = _storage_boundary_evidence()

    evaluator.validate_storage_boundary_prerequisite(
        evidence,
        expected_commit="frozen-commit",
        expected_model=selector["model"],
    )

    failed = deepcopy(evidence)
    failed["derivative_gate"]["observed"]["sign_agreement"] = 0.94
    with pytest.raises(ValueError, match="numeric observations"):
        evaluator.validate_storage_boundary_prerequisite(
            failed,
            expected_commit="frozen-commit",
            expected_model=selector["model"],
        )

    wrong_commit = deepcopy(evidence)
    wrong_commit["repository"]["end"]["commit"] = "different"
    with pytest.raises(ValueError, match="commit does not match"):
        evaluator.validate_storage_boundary_prerequisite(
            wrong_commit,
            expected_commit="frozen-commit",
            expected_model=selector["model"],
        )


def test_evaluator_source_freeze_covers_metric_implementation() -> None:
    assert "src/recurquant/metrics.py" in evaluator.EVALUATOR_SOURCE_FILES


def test_source_file_hashes_are_path_stable_and_detect_changes(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"first")
    first = evaluator.source_file_hashes(tmp_path, ("source.py",))
    source.write_bytes(b"second")
    second = evaluator.source_file_hashes(tmp_path, ("source.py",))

    assert set(first) == {"source.py"}
    assert first != second


def test_actual_token_manifest_must_match_selected_prefix() -> None:
    selector = _selector(evaluator.HRR_ARTIFACT_KIND)
    records = [
        {
            "task_id": 20,
            "prompt_tokens": 7,
            "code_tokens": 4,
            "aligned_scored_tokens": 3,
            "full_code_scored_tokens": 4,
        }
    ]

    evaluator.validate_actual_token_manifest(selector, records)

    records[0]["prompt_tokens"] = 8
    with pytest.raises(ValueError, match="formatter/token manifest"):
        evaluator.validate_actual_token_manifest(selector, records)


def test_one_task_contrast_is_labeled_descriptive_without_interval() -> None:
    result = evaluator.paired_contrast([2.0], [1.25], samples=100, seed=17)

    assert result["mean_improvement"] == pytest.approx(0.75)
    assert result["confidence_interval"] is None
    assert result["bootstrap_samples"] == 0
    assert "descriptive only" in result["note"]


def test_multi_task_contrast_keeps_bootstrap_interval() -> None:
    result = evaluator.paired_contrast([2.0, 3.0], [1.0, 2.0], samples=100, seed=17)

    assert result["paired_examples"] == 2
    assert result["confidence_interval"] == pytest.approx([1.0, 1.0])


def test_frozen_holdout_gate_passes_all_prespecified_thresholds() -> None:
    gate = evaluator.evaluate_frozen_holdout_gate(**_passing_gate_inputs())

    assert gate["passed"] is True
    assert gate["strongest_equal_byte_static"] == "hrr_h1"
    assert all(check["passed"] is True for check in gate["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda inputs: inputs["aggregates"][evaluator.ADAPTIVE_TARGET_FISHER].update(
                macro_delta_nll=0.81
            ),
            "relative_nll_reduction_vs_strongest_static",
        ),
        (
            lambda inputs: inputs["aggregates"][evaluator.ADAPTIVE_TARGET_FISHER].update(
                macro_top1_agreement=0.894
            ),
            "top1_disadvantage_margin_vs_strongest_static",
        ),
        (
            lambda inputs: inputs["aggregates"][evaluator.ADAPTIVE_TARGET_FISHER].update(
                macro_cvar95_kl=0.251
            ),
            "cvar95_disadvantage_margin_vs_strongest_static",
        ),
        (
            lambda inputs: inputs["per_task"][evaluator.ADAPTIVE_TARGET_FISHER][0].update(
                delta_nll=1.6
            ),
            "maximum_per_task_nll_disadvantage_vs_strongest_static",
        ),
        (
            lambda inputs: inputs["contrasts"]["hrr_h1"].update(confidence_interval=[0.0, 0.5]),
            "paired_lower_ci_vs_strongest_static",
        ),
        (
            lambda inputs: inputs["contrasts"][evaluator.ADAPTIVE_H1].update(
                confidence_interval=[0.0, 0.5]
            ),
            "paired_lower_ci_vs_adaptive_h1",
        ),
        (
            lambda inputs: inputs["storage"][evaluator.ADAPTIVE_TARGET_FISHER].update(
                resident_bytes=evaluator.TARGET_RESIDENT_BYTES - 1
            ),
            "exact_resident_bytes",
        ),
    ],
)
def test_frozen_holdout_gate_reports_each_threshold_failure(mutation, failed_check: str) -> None:
    inputs = _passing_gate_inputs()
    mutation(inputs)

    gate = evaluator.evaluate_frozen_holdout_gate(**inputs)

    assert gate["passed"] is False
    assert gate["checks"][failed_check]["passed"] is False


def _passing_cqer_development_inputs() -> dict[str, object]:
    methods = {
        evaluator.QUERY_EMA_PRIMARY: {
            "macro_delta_nll": 0.40,
            "macro_mean_kl": 0.40,
            "macro_cvar95_kl": 1.00,
            "macro_top1_agreement": 0.80,
            "task_count": 1,
            "token_count": 10,
        },
        evaluator.ADAPTIVE_TARGET_FISHER: {
            "macro_delta_nll": 0.50,
            "macro_mean_kl": 0.50,
            "macro_cvar95_kl": 0.95,
            "macro_top1_agreement": 0.795,
            "task_count": 1,
            "token_count": 10,
        },
        "target_directional_fisher_difference_int4": {
            "macro_delta_nll": 0.60,
            "macro_mean_kl": 0.60,
            "macro_cvar95_kl": 1.10,
            "macro_top1_agreement": 0.805,
            "task_count": 1,
            "token_count": 10,
        },
    }
    per_task = {
        name: [
            {
                "task_id": task_id,
                "delta_nll": values["macro_delta_nll"],
                "mean_kl": values["macro_mean_kl"],
                "cvar95_kl": values["macro_cvar95_kl"],
                "top1_agreement": values["macro_top1_agreement"],
                "token_count": 10,
                "candidate_nll": 1.0,
                "reference_nll": 0.5,
                "max_kl": 1.5,
                "all_logits_finite": True,
            }
            for task_id in range(8)
        ]
        for name, values in methods.items()
    }
    storage = {
        name: {
            "resident_bytes": 100,
            "high_precision_groups": 2,
        }
        for name in methods
    }
    storage[evaluator.QUERY_EMA_PRIMARY].update(
        selector_auxiliary_bytes=8,
        resident_bytes_including_selector=108,
    )
    diagnostics = {
        evaluator.QUERY_EMA_PRIMARY: [
            {
                "task_id": task_id,
                "layers": [
                    {
                        "layer_index": 0,
                        "quota": 2,
                        "current_selected_count": 2,
                        "observations_committed": 3,
                        "state_updates": 3,
                        "selector_auxiliary_bytes": 8,
                        "current_mask_sha256": "a" * 64,
                        "pending_observation": False,
                        "last_cutoff_score_margin": 0.01,
                    }
                ],
            }
            for task_id in range(8)
        ]
    }
    return {
        "aggregates": methods,
        "per_task": per_task,
        "per_task_full_code": deepcopy(per_task),
        "storage": storage,
        "query_ema_diagnostics": diagnostics,
        "expected_quotas": {0: 2},
        "expected_packed_bytes": 100,
        "expected_selector_auxiliary_bytes": 8,
    }


def test_cqer_development_gate_passes_only_the_frozen_conjunction() -> None:
    inputs = _passing_cqer_development_inputs()

    gate = evaluator.evaluate_cqer_development_gate(**inputs)

    assert gate["passed"] is True
    assert all(check["passed"] is True for check in gate["checks"].values())

    inputs = _passing_cqer_development_inputs()
    inputs["aggregates"][evaluator.QUERY_EMA_PRIMARY]["macro_delta_nll"] = 0.49
    gate = evaluator.evaluate_cqer_development_gate(**inputs)
    assert gate["passed"] is False
    assert gate["checks"]["relative_nll_reduction_vs_plain_adaptive"]["passed"] is False
    assert gate["checks"]["relative_nll_reduction_vs_strongest_static"]["passed"] is False


def test_cqer_development_gate_rejects_stale_stage_consume_state() -> None:
    inputs = _passing_cqer_development_inputs()
    layer = inputs["query_ema_diagnostics"][evaluator.QUERY_EMA_PRIMARY][0]["layers"][0]
    layer["pending_observation"] = True

    gate = evaluator.evaluate_cqer_development_gate(**inputs)

    assert gate["passed"] is False
    assert gate["checks"]["exact_stage_consume_handshake"]["passed"] is False

    inputs = _passing_cqer_development_inputs()
    inputs["per_task_full_code"][evaluator.QUERY_EMA_PRIMARY][0][
        "all_logits_finite"
    ] = False
    gate = evaluator.evaluate_cqer_development_gate(**inputs)
    assert gate["passed"] is False
    assert gate["checks"]["all_values_finite"]["passed"] is False


def test_evaluator_runs_cqer_observer_through_prefill_and_decode() -> None:
    torch.manual_seed(443)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    plan = evaluator.select_rows_exact_budget(
        scores,
        target_resident_bytes=118,
        group_size=8,
    )
    prompt_ids = torch.randint(0, config.vocab_size, (1, 4))
    code_ids = torch.randint(0, config.vocab_size, (1, 3))

    with torch.inference_mode():
        summaries, _, storage, diagnostics, reference_bytes = evaluator.evaluate_task(
            model,
            prompt_ids=prompt_ids,
            code_ids=code_ids,
            plans={},
            adaptive_plans={},
            query_ema_plans={evaluator.QUERY_EMA_PRIMARY: plan},
        )

    query_summary = summaries[evaluator.QUERY_EMA_PRIMARY]
    query_storage = storage[evaluator.QUERY_EMA_PRIMARY]
    layer = diagnostics[evaluator.QUERY_EMA_PRIMARY][0]
    assert query_summary["token_count"] == 2
    assert query_storage["resident_bytes"] == plan.resident_bytes
    assert query_storage["selector_auxiliary_bytes"] == 64
    assert layer["observations_committed"] == 3
    assert layer["state_updates"] == 3
    assert layer["tokens_observed"] == 6
    assert layer["pending_observation"] is False
    assert reference_bytes > 0


def test_claim_and_exit_code_follow_actual_primary_and_gate() -> None:
    static_claim = evaluator.primary_claim_text("hrr_h32")
    adaptive_claim = evaluator.primary_claim_text(evaluator.ADAPTIVE_TARGET_FISHER)
    fusion_claim = evaluator.primary_claim_text(evaluator.RANK_FUSION_PRIMARY)
    query_ema_claim = evaluator.primary_claim_text(evaluator.QUERY_EMA_PRIMARY)

    assert "static HRR" in static_claim
    assert "no loss-selector" in static_claim
    assert "target-directional-Fisher" in adaptive_claim
    assert "equal-weight ordinal rank fusion" in fusion_claim
    assert "normalized-query-energy EMA" in query_ema_claim
    assert "32-token half-life" in query_ema_claim
    assert evaluator.QUERY_EMA_HALF_LIFE == 32
    assert evaluator.RANK_FUSION_METHODS == (
        ("rank_fusion_l025_target_fisher_adaptive_mse", 0.25),
        ("rank_fusion_l050_target_fisher_adaptive_mse", 0.50),
        ("rank_fusion_l075_target_fisher_adaptive_mse", 0.75),
    )
    assert evaluator.diagnostic_exit_code(heldout_calibration=True, gate_passed=False) == 2
    assert evaluator.diagnostic_exit_code(heldout_calibration=True, gate_passed=True) == 0
    assert evaluator.diagnostic_exit_code(heldout_calibration=False, gate_passed=False) == 0


def _cora_metric(
    *,
    delta_nll: float,
    top1: float,
    cvar95: float,
) -> dict[str, float | int]:
    return {
        "macro_delta_nll": delta_nll,
        "macro_mean_kl": delta_nll,
        "macro_cvar95_kl": cvar95,
        "macro_top1_agreement": top1,
        "task_count": evaluator.E008_DEVELOPMENT_LIMIT,
        "token_count": 2 * evaluator.E008_DEVELOPMENT_LIMIT,
    }


def _cora_task_rows(
    summary: dict[str, float | int],
) -> list[dict[str, float | int | bool]]:
    return [
        {
            "task_id": task_id,
            "delta_nll": summary["macro_delta_nll"],
            "mean_kl": summary["macro_mean_kl"],
            "cvar95_kl": summary["macro_cvar95_kl"],
            "top1_agreement": summary["macro_top1_agreement"],
            "token_count": 2,
            "candidate_nll": 1.0,
            "reference_nll": 0.5,
            "max_kl": 1.5,
            "all_logits_finite": True,
        }
        for task_id in evaluator.E008_DEVELOPMENT_TASK_IDS
    ]


def _query_selector_layer(*, confirmation_two: bool) -> dict[str, object]:
    return {
        "layer_index": 0,
        "quota": 2,
        "confirmation_two": confirmation_two,
        "ema_decay": 2.0 ** (-1.0 / evaluator.QUERY_EMA_HALF_LIFE),
        "l2norm_eps": 1e-6,
        "state_updates": 3,
        "observations_staged": 3,
        "observations_committed": 3,
        "tokens_observed": 6,
        "last_query_token_count": 1,
        "last_cutoff_score_margin": 0.01,
        "last_mask_overlap": 1,
        "last_mask_churn": 2,
        "current_selected_count": 2,
        "current_mask_sha256": "a" * 64,
        "raw_mask_sha256": "b" * 64,
        "committed_mask_sha256": "a" * 64,
        "pending_observation": False,
        "selector_auxiliary_bytes": 9 if confirmation_two else 8,
    }


def _cora_selector_layer(
    *,
    confirmation_two: bool,
    committed_xor: int,
) -> dict[str, object]:
    transitions = 2
    quota = 2
    denominator = 2 * quota * transitions
    raw_xor = 4
    admissions = committed_xor // 2
    dwell = quota * transitions - admissions
    return {
        "layer_index": 0,
        "selection_method": (
            evaluator.CORA_C2_PRIMARY if confirmation_two else evaluator.CORA_RAW
        ),
        "confirmation_two": confirmation_two,
        "quota": quota,
        "current_selected_count": quota,
        "raw_mask_sha256": "b" * 64,
        "committed_mask_sha256": "c" * 64,
        "observations_staged": 3,
        "observations_committed": 3,
        "tokens_observed": 6,
        "last_token_count": 1,
        "mask_transition_count": transitions,
        "raw_xor_churn_total": raw_xor,
        "committed_xor_churn_total": committed_xor,
        "raw_normalized_churn": raw_xor / denominator,
        "committed_normalized_churn": committed_xor / denominator,
        "last_raw_mask_overlap": 1,
        "last_committed_mask_overlap": 1,
        "last_dwell_count": 1,
        "last_admission_count": 1,
        "dwell_total": dwell,
        "admissions_total": admissions,
        "last_raw_cutoff_score": 0.1,
        "last_raw_score_gap": 0.01,
        "last_raw_normalized_gap": 0.1,
        "last_committed_cutoff_score": 0.1,
        "last_committed_score_gap": 0.01,
        "last_committed_normalized_gap": 0.1,
        "observability_trace": 1.0,
        "observability_min": 0.1,
        "observability_max": 0.9,
        "observability_dtype": "torch.float32",
        "l2norm_eps": 1e-6,
        "state_updates": 3,
        "pending_observation": False,
        "observability_diagonal_bytes": 8,
        # Raw CORA also keeps its previous raw mask to report cumulative churn.
        "previous_raw_mask_bytes": 1,
        "selector_auxiliary_bytes": 9,
    }


def _selector_task_diagnostics(layer: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"task_id": task_id, "layers": [deepcopy(layer)]}
        for task_id in evaluator.E008_DEVELOPMENT_TASK_IDS
    ]


def _passing_cora_c2_development_inputs() -> dict[str, object]:
    static = "target_directional_fisher_difference_int4"
    aggregates = {
        static: _cora_metric(delta_nll=1.00, top1=0.800, cvar95=1.00),
        evaluator.ADAPTIVE_TARGET_FISHER: _cora_metric(
            delta_nll=0.90,
            top1=0.810,
            cvar95=0.95,
        ),
        evaluator.QUERY_EMA_PRIMARY: _cora_metric(
            delta_nll=0.80,
            top1=0.800,
            cvar95=1.00,
        ),
        evaluator.CORA_RAW: _cora_metric(
            delta_nll=0.72,
            top1=0.804,
            cvar95=1.02,
        ),
        evaluator.QUERY_EMA_C2: _cora_metric(
            delta_nll=0.78,
            top1=0.801,
            cvar95=1.00,
        ),
        evaluator.CORA_C2_PRIMARY: _cora_metric(
            delta_nll=0.70,
            top1=0.805,
            cvar95=1.04,
        ),
    }
    per_task = {name: _cora_task_rows(values) for name, values in aggregates.items()}
    per_task_full_code = deepcopy(per_task)
    for task_rows in per_task_full_code.values():
        for row in task_rows:
            row["token_count"] = 3
    aggregates_full_code = deepcopy(aggregates)
    for summary in aggregates_full_code.values():
        summary["token_count"] = 3 * evaluator.E008_DEVELOPMENT_LIMIT
    packed_bytes = 100
    storage = {
        name: {
            "resident_bytes": packed_bytes,
            "high_precision_groups": 2,
        }
        for name in aggregates
    }
    storage[evaluator.QUERY_EMA_PRIMARY].update(
        selector_auxiliary_bytes=8,
        resident_bytes_including_selector=108,
    )
    storage[evaluator.CORA_RAW].update(
        selector_auxiliary_bytes=9,
        resident_bytes_including_selector=109,
    )
    for name in (evaluator.QUERY_EMA_C2, evaluator.CORA_C2_PRIMARY):
        storage[name].update(
            selector_auxiliary_bytes=9,
            resident_bytes_including_selector=109,
        )
    selector_diagnostics = {
        evaluator.QUERY_EMA_PRIMARY: _selector_task_diagnostics(
            _query_selector_layer(confirmation_two=False)
        ),
        evaluator.QUERY_EMA_C2: _selector_task_diagnostics(
            _query_selector_layer(confirmation_two=True)
        ),
        evaluator.CORA_RAW: _selector_task_diagnostics(
            _cora_selector_layer(confirmation_two=False, committed_xor=4)
        ),
        evaluator.CORA_C2_PRIMARY: _selector_task_diagnostics(
            _cora_selector_layer(confirmation_two=True, committed_xor=2)
        ),
    }
    return {
        "aggregates": aggregates,
        "aggregates_full_code": aggregates_full_code,
        "per_task": per_task,
        "per_task_full_code": per_task_full_code,
        "storage": storage,
        "selector_diagnostics": selector_diagnostics,
        "contrasts": {
            static: {"confidence_interval": [0.10, 0.50]},
            evaluator.QUERY_EMA_PRIMARY: {"confidence_interval": [0.02, 0.20]},
        },
        "token_manifest": [
            {
                "task_id": task_id,
                "prompt_tokens": 4,
                "code_tokens": 3,
                "aligned_scored_tokens": 2,
                "full_code_scored_tokens": 3,
            }
            for task_id in evaluator.E008_DEVELOPMENT_TASK_IDS
        ],
        "expected_quotas": {0: 2},
        "expected_packed_bytes": packed_bytes,
        "expected_query_auxiliary_bytes": 8,
        "expected_cora_auxiliary_bytes": 9,
    }


def _set_all_cora_c2_committed_churn(values: dict[str, object], churn: int) -> None:
    for task in values["selector_diagnostics"][evaluator.CORA_C2_PRIMARY]:
        layer = task["layers"][0]
        layer["committed_xor_churn_total"] = churn
        layer["committed_normalized_churn"] = churn / 8
        layer["admissions_total"] = churn // 2
        layer["dwell_total"] = 4 - churn // 2


@pytest.mark.parametrize(
    "changes",
    (
        {"offset": 8},
        {"offset": 15},
        {"offset": 17},
        {"limit": 15},
        {"bootstrap_samples": 9_999},
    ),
)
def test_experiment008_request_pins_development_and_protects_eight_to_sixteen(
    changes: dict[str, int],
) -> None:
    arguments = {
        "enabled": True,
        "offset": evaluator.E008_DEVELOPMENT_OFFSET,
        "limit": evaluator.E008_DEVELOPMENT_LIMIT,
        "bootstrap_samples": evaluator.FROZEN_BOOTSTRAP_SAMPLES,
    }
    evaluator.validate_cora_development_request(**arguments)
    arguments.update(changes)

    with pytest.raises(ValueError, match=r"\[16, 32\).*\[8, 16\) remains closed"):
        evaluator.validate_cora_development_request(**arguments)


def test_experiment008_disabled_request_is_a_noop() -> None:
    evaluator.validate_cora_development_request(
        enabled=False,
        offset=8,
        limit=1,
        bootstrap_samples=1,
    )


def test_experiment008_requires_a_clean_committed_repository_before_data_access() -> None:
    clean = {"commit": "a" * 40, "worktree_clean": True, "status": []}
    evaluator.validate_cora_development_repository_start(clean)

    with pytest.raises(ValueError, match="clean committed worktree"):
        evaluator.validate_cora_development_repository_start(
            {"commit": "a" * 40, "worktree_clean": False, "status": [" M source.py"]}
        )
    with pytest.raises(ValueError, match="40-character Git commit"):
        evaluator.validate_cora_development_repository_start(
            {"commit": "short", "worktree_clean": True, "status": []}
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("task_ids", tuple(reversed(evaluator.E008_DEVELOPMENT_TASK_IDS)), "task IDs"),
        ("content_manifest_sha256", "0" * 64, "content manifest"),
        ("token_manifest_sha256", "0" * 64, "token manifest"),
    ),
)
def test_experiment008_identity_rejects_every_frozen_identity_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = {
        "enabled": True,
        "task_ids": evaluator.E008_DEVELOPMENT_TASK_IDS,
        "content_manifest_sha256": evaluator.E008_DEVELOPMENT_CONTENT_MANIFEST_SHA256,
        "token_manifest_sha256": evaluator.E008_DEVELOPMENT_TOKEN_MANIFEST_SHA256,
    }
    evaluator.validate_cora_development_identity(**arguments)
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        evaluator.validate_cora_development_identity(**arguments)


def test_experiment008_authenticates_exact_selector_pair_and_quotas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]
    monkeypatch.setattr(evaluator, "CQER_DEVELOPMENT_TASK_IDS", tuple(range(100, 108)))
    monkeypatch.setattr(
        evaluator,
        "CQER_FROZEN_SELECTOR_CANONICAL_SHA256S",
        tuple(
            evaluator.sha256_bytes(evaluator.canonical_json_bytes(selector))
            for selector in selectors
        ),
    )

    evaluator.validate_cora_selector_artifacts(enabled=True, selectors=selectors)
    evaluator.validate_cora_layer_quotas(
        enabled=True,
        quotas=dict(evaluator.CQER_FROZEN_LAYER_QUOTAS),
    )

    drifted_selector = deepcopy(selectors)
    drifted_selector[1]["byte_budget"]["target_resident_bytes"] = 999
    with pytest.raises(ValueError, match="canonical hashes"):
        evaluator.validate_cora_selector_artifacts(
            enabled=True,
            selectors=drifted_selector,
        )

    drifted_quotas = dict(evaluator.CQER_FROZEN_LAYER_QUOTAS)
    drifted_quotas[0] -= 1
    drifted_quotas[1] += 1
    with pytest.raises(ValueError, match="target-Fisher allocation"):
        evaluator.validate_cora_layer_quotas(enabled=True, quotas=drifted_quotas)


def test_experiment008_cora_path_does_not_authorize_the_protected_window() -> None:
    with pytest.raises(ValueError, match=r"\[8, 16\) remains closed"):
        evaluator.validate_cora_development_request(
            enabled=True,
            offset=8,
            limit=8,
            bootstrap_samples=evaluator.FROZEN_BOOTSTRAP_SAMPLES,
        )


@pytest.mark.parametrize(
    ("offset", "limit", "bootstrap_samples"),
    (
        (8, 8, 10_000),
        (16, 15, 10_000),
        (16, 16, 9_999),
    ),
)
def test_generic_holdout_validator_only_bypasses_for_exact_e008_development_request(
    offset: int,
    limit: int,
    bootstrap_samples: int,
) -> None:
    evaluator.validate_frozen_holdout_request(
        offset=16,
        limit=16,
        bootstrap_samples=10_000,
        selectors=[],
        loss_selector_present=False,
        storage_boundary_present=False,
        cora_c2_enabled=True,
    )

    with pytest.raises(ValueError, match="Experiment 008 CORA-C2"):
        evaluator.validate_frozen_holdout_request(
            offset=offset,
            limit=limit,
            bootstrap_samples=bootstrap_samples,
            selectors=[],
            loss_selector_present=False,
            storage_boundary_present=False,
            cora_c2_enabled=True,
        )


def test_cora_c2_development_gate_passes_the_complete_frozen_conjunction() -> None:
    gate = evaluator.evaluate_cora_c2_development_gate(
        **_passing_cora_c2_development_inputs()
    )

    assert gate["schema"] == "recurquant.experiment008-cora-c2-development-gate.v1"
    assert gate["primary"] == evaluator.CORA_C2_PRIMARY
    assert gate["passed"] is True
    assert set(gate["checks"]) == {
        "exact_per_layer_quotas",
        "exact_packed_and_selector_bytes",
        "exact_stage_consume_handshake",
        "all_values_finite",
        "lower_nll_than_primary_comparators",
        "relative_nll_reduction_vs_static",
        "relative_nll_reduction_vs_adaptive",
        "relative_nll_reduction_vs_cqer",
        "paired_lower_ci_vs_static",
        "paired_lower_ci_vs_cqer",
        "top1_margin_vs_static_adaptive",
        "top1_not_lower_than_cqer",
        "cvar95_margin_vs_static_adaptive",
        "raw_cora_relative_nll_reduction_vs_cqer",
        "c2_churn_reduction_vs_raw",
        "c2_top1_not_lower_than_raw",
        "c2_nll_worsening_vs_raw",
    }
    assert all(check["passed"] is True for check in gate["checks"].values())


def test_cora_c2_gate_fails_closed_when_staging_evidence_is_missing() -> None:
    inputs = _passing_cora_c2_development_inputs()
    layer = inputs["selector_diagnostics"][evaluator.CORA_C2_PRIMARY][0]["layers"][0]
    del layer["observations_staged"]

    gate = evaluator.evaluate_cora_c2_development_gate(**inputs)

    assert gate["passed"] is False
    assert gate["checks"]["exact_stage_consume_handshake"]["passed"] is False


def test_cora_c2_gate_rejects_method_or_metric_token_drift() -> None:
    inputs = _passing_cora_c2_development_inputs()
    inputs["aggregates"]["posthoc_method"] = deepcopy(
        inputs["aggregates"][evaluator.CORA_C2_PRIMARY]
    )
    with pytest.raises(ValueError, match="frozen six-method set"):
        evaluator.evaluate_cora_c2_development_gate(**inputs)

    inputs = _passing_cora_c2_development_inputs()
    inputs["per_task"][evaluator.CORA_C2_PRIMARY][0]["token_count"] = 1
    with pytest.raises(ValueError, match="aligned token counts"):
        evaluator.evaluate_cora_c2_development_gate(**inputs)


def test_cora_c2_gate_rejects_dropped_transition_tokens() -> None:
    inputs = _passing_cora_c2_development_inputs()
    layer = inputs["selector_diagnostics"][evaluator.CORA_C2_PRIMARY][0]["layers"][0]
    layer["tokens_observed"] = 5

    gate = evaluator.evaluate_cora_c2_development_gate(**inputs)

    assert gate["passed"] is False
    assert gate["checks"]["exact_stage_consume_handshake"]["passed"] is False


def test_evaluator_runs_all_e008_dynamic_methods_with_disjoint_observers() -> None:
    torch.manual_seed(449)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    plan = evaluator.select_rows_exact_budget(
        scores,
        target_resident_bytes=118,
        group_size=8,
    )
    prompt_ids = torch.randint(0, config.vocab_size, (1, 4))
    code_ids = torch.randint(0, config.vocab_size, (1, 3))
    dynamic_methods = {
        evaluator.QUERY_EMA_PRIMARY,
        evaluator.QUERY_EMA_C2,
        evaluator.CORA_RAW,
        evaluator.CORA_C2_PRIMARY,
    }

    with torch.inference_mode():
        summaries, _, storage, diagnostics, reference_bytes = evaluator.evaluate_task(
            model,
            prompt_ids=prompt_ids,
            code_ids=code_ids,
            plans={},
            adaptive_plans={},
            query_ema_plans={evaluator.QUERY_EMA_PRIMARY: plan},
            query_ema_c2_plans={evaluator.QUERY_EMA_C2: plan},
            cora_specs={
                evaluator.CORA_RAW: (plan, False),
                evaluator.CORA_C2_PRIMARY: (plan, True),
            },
            include_default_baselines=False,
        )

    assert set(summaries) == dynamic_methods
    assert set(diagnostics) == dynamic_methods
    assert all(summaries[name]["token_count"] == 2 for name in dynamic_methods)
    assert all(storage[name]["resident_bytes"] == plan.resident_bytes for name in dynamic_methods)
    assert storage[evaluator.QUERY_EMA_PRIMARY]["selector_auxiliary_bytes"] == 64
    assert storage[evaluator.QUERY_EMA_C2]["selector_auxiliary_bytes"] == 66
    assert storage[evaluator.CORA_RAW]["selector_auxiliary_bytes"] == 66
    assert storage[evaluator.CORA_C2_PRIMARY]["selector_auxiliary_bytes"] == 66

    for name in dynamic_methods:
        assert len(diagnostics[name]) == 1
        layer = diagnostics[name][0]
        assert layer["state_updates"] == 3
        assert layer["observations_staged"] == 3
        assert layer["observations_committed"] == 3
        assert layer["tokens_observed"] == 6
        assert layer["current_selected_count"] == len(plan.groups_for_layer(0))
        assert layer["pending_observation"] is False

    assert diagnostics[evaluator.QUERY_EMA_PRIMARY][0]["confirmation_two"] is False
    assert diagnostics[evaluator.QUERY_EMA_C2][0]["confirmation_two"] is True
    assert diagnostics[evaluator.CORA_RAW][0]["selection_method"] == evaluator.CORA_RAW
    assert (
        diagnostics[evaluator.CORA_C2_PRIMARY][0]["selection_method"]
        == evaluator.CORA_C2_PRIMARY
    )
    assert reference_bytes > 0


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda values: values["selector_diagnostics"][evaluator.CORA_C2_PRIMARY][0][
                "layers"
            ][0].update(current_selected_count=1),
            "exact_per_layer_quotas",
        ),
        (
            lambda values: values["storage"][evaluator.CORA_C2_PRIMARY].update(
                resident_bytes_including_selector=110
            ),
            "exact_packed_and_selector_bytes",
        ),
        (
            lambda values: values["selector_diagnostics"][evaluator.CORA_C2_PRIMARY][0][
                "layers"
            ][0].update(observations_committed=2),
            "exact_stage_consume_handshake",
        ),
        (
            lambda values: values["per_task_full_code"][evaluator.CORA_C2_PRIMARY][0].update(
                all_logits_finite=False
            ),
            "all_values_finite",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_C2_PRIMARY].update(
                macro_delta_nll=0.80
            ),
            "lower_nll_than_primary_comparators",
        ),
        (
            lambda values: values["aggregates"][
                "target_directional_fisher_difference_int4"
            ].update(macro_delta_nll=0.874),
            "relative_nll_reduction_vs_static",
        ),
        (
            lambda values: values["aggregates"][evaluator.ADAPTIVE_TARGET_FISHER].update(
                macro_delta_nll=0.735
            ),
            "relative_nll_reduction_vs_adaptive",
        ),
        (
            lambda values: values["aggregates"][evaluator.QUERY_EMA_PRIMARY].update(
                macro_delta_nll=0.735
            ),
            "relative_nll_reduction_vs_cqer",
        ),
        (
            lambda values: values["contrasts"][
                "target_directional_fisher_difference_int4"
            ].update(confidence_interval=[0.0, 0.5]),
            "paired_lower_ci_vs_static",
        ),
        (
            lambda values: values["contrasts"][evaluator.QUERY_EMA_PRIMARY].update(
                confidence_interval=[0.0, 0.2]
            ),
            "paired_lower_ci_vs_cqer",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_C2_PRIMARY].update(
                macro_top1_agreement=0.799
            ),
            "top1_margin_vs_static_adaptive",
        ),
        (
            lambda values: values["aggregates"][evaluator.QUERY_EMA_PRIMARY].update(
                macro_top1_agreement=0.806
            ),
            "top1_not_lower_than_cqer",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_C2_PRIMARY].update(
                macro_cvar95_kl=1.051
            ),
            "cvar95_margin_vs_static_adaptive",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_RAW].update(
                macro_delta_nll=0.777
            ),
            "raw_cora_relative_nll_reduction_vs_cqer",
        ),
        (
            lambda values: _set_all_cora_c2_committed_churn(values, 4),
            "c2_churn_reduction_vs_raw",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_RAW].update(
                macro_top1_agreement=0.806
            ),
            "c2_top1_not_lower_than_raw",
        ),
        (
            lambda values: values["aggregates"][evaluator.CORA_RAW].update(
                macro_delta_nll=0.69
            ),
            "c2_nll_worsening_vs_raw",
        ),
    ),
)
def test_cora_c2_development_gate_fails_each_frozen_condition(
    mutation,
    failed_check: str,
) -> None:
    inputs = _passing_cora_c2_development_inputs()
    mutation(inputs)

    gate = evaluator.evaluate_cora_c2_development_gate(**inputs)

    assert gate["passed"] is False
    assert gate["checks"][failed_check]["passed"] is False
