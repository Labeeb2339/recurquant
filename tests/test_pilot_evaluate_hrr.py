from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts import pilot_evaluate_hrr as evaluator
from scripts import pilot_hrr_rows, pilot_loss_sensitivity_rows
from scripts import pilot_validate_storage_boundary as storage_validator


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
            "manifest_sha256": evaluator.sha256_bytes(evaluator.canonical_json_bytes(manifest)),
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
        "manifest_sha256": evaluator.sha256_bytes(evaluator.canonical_json_bytes(manifest)),
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


def test_frozen_holdout_request_requires_exact_window_and_two_eight_task_selectors() -> None:
    selectors = [
        _frozen_selector(evaluator.HRR_ARTIFACT_KIND),
        _frozen_selector(evaluator.LOSS_ARTIFACT_KIND),
    ]

    evaluator.validate_frozen_holdout_request(
        offset=8,
        limit=8,
        bootstrap_samples=10_000,
        selectors=selectors,
        loss_selector_present=True,
        storage_boundary_present=True,
    )


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


def test_claim_and_exit_code_follow_actual_primary_and_gate() -> None:
    static_claim = evaluator.primary_claim_text("hrr_h32")
    adaptive_claim = evaluator.primary_claim_text(evaluator.ADAPTIVE_TARGET_FISHER)

    assert "static HRR" in static_claim
    assert "no loss-selector" in static_claim
    assert "target-directional-Fisher" in adaptive_claim
    assert evaluator.diagnostic_exit_code(heldout_calibration=True, gate_passed=False) == 2
    assert evaluator.diagnostic_exit_code(heldout_calibration=True, gate_passed=True) == 0
    assert evaluator.diagnostic_exit_code(heldout_calibration=False, gate_passed=False) == 0
