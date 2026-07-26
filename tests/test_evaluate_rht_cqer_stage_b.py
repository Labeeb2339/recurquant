from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from recurquant.evaluation import paired_bootstrap_mean_improvement
from scripts import evaluate_rht_cqer_stage_b as stage_b


def _metric(
    *,
    task_id: int,
    token_count: int,
    delta_nll: float,
    mean_kl: float,
    cvar95_kl: float,
    top1_agreement: float,
) -> dict[str, float | int | bool]:
    return {
        "task_id": task_id,
        "token_count": token_count,
        "mean_kl": mean_kl,
        "cvar95_kl": cvar95_kl,
        "max_kl": cvar95_kl + 0.25,
        "top1_agreement": top1_agreement,
        "reference_nll": 1.0,
        "candidate_nll": 1.0 + delta_nll,
        "delta_nll": delta_nll,
        "all_logits_finite": True,
    }


def _task_records() -> list[dict[str, int | str]]:
    digest = "a" * 64
    return [
        {
            "rank": stage_b.STAGE_B_OFFSET + index,
            "task_id": 10_000 + ((index * 17) % stage_b.STAGE_B_LIMIT),
            "row_sha256": f"{index + 1:064x}",
            "prompt_tokens": 3,
            "code_tokens": 4,
            "aligned_scored_tokens": 3,
            "full_code_scored_tokens": 4,
            "prompt_text_sha256": digest,
            "code_text_sha256": digest,
            "prompt_token_ids_sha256": digest,
            "code_token_ids_sha256": digest,
        }
        for index in range(stage_b.STAGE_B_LIMIT)
    ]


def _frozen_row_plan_inputs() -> tuple[dict[str, object], dict[str, object]]:
    resolver = stage_b.identity_resolver
    rows = [
        {
            "layer_index": layer,
            "head_index": flat_index // 128,
            "row_index": flat_index % 128,
        }
        for layer in stage_b.FROZEN_LINEAR_LAYERS
        for flat_index in range(stage_b.FROZEN_LAYER_QUOTAS[layer])
    ]
    selector_binding = {
        "selector_file_sha256": resolver.SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": (
            resolver.SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
        "loss_selector_file_sha256": resolver.LOSS_SELECTOR_FILE_SHA256,
        "loss_selector_canonical_evidence_sha256": (
            resolver.LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
    }
    row_plan: dict[str, object] = {
        "schema": resolver.ROW_PLAN_SCHEMA,
        "method": stage_b.TARGET_FISHER_SCORE,
        "selector_binding": selector_binding,
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
            "mask_bytes": stage_b.TARGET_MASK_BYTES,
            "promotion_increment_bytes": 64,
            "target_resident_bytes": stage_b.TARGET_PACKED_STATE_BYTES,
            "resident_bytes": stage_b.TARGET_PACKED_STATE_BYTES,
            "promoted_group_count": stage_b.TARGET_PROMOTED_ROWS,
        },
        "score_shapes": [
            {"layer_index": layer, "heads": 16, "rows": 128}
            for layer in stage_b.FROZEN_LINEAR_LAYERS
        ],
        "layer_quotas": {
            str(layer): quota
            for layer, quota in stage_b.FROZEN_LAYER_QUOTAS.items()
        },
        "high_precision_rows": rows,
    }
    row_plan["canonical_plan_sha256"] = stage_b._sha256_bytes(  # noqa: SLF001
        stage_b.canonical_json_bytes(row_plan)
    )
    identity = {
        "row_plan": row_plan,
        "model_contract": {
            "id": resolver.MODEL_ID,
            "revision": resolver.MODEL_REVISION,
            "weights_loaded": False,
        },
    }
    stage_a = {"selector_artifacts": dict(selector_binding)}
    return identity, stage_a


def _rehash_row_plan(identity: dict[str, object]) -> None:
    row_plan = dict(identity["row_plan"])  # type: ignore[arg-type]
    row_plan.pop("canonical_plan_sha256")
    row_plan["canonical_plan_sha256"] = stage_b._sha256_bytes(  # noqa: SLF001
        stage_b.canonical_json_bytes(row_plan)
    )
    identity["row_plan"] = row_plan


def _per_task_metrics(
    task_records: list[dict[str, int | str]],
    *,
    token_field: str = "aligned_scored_tokens",
) -> dict[str, list[dict[str, float | int | bool]]]:
    values = {
        stage_b.STATIC_METHOD: (0.95, 0.9, 1.8, 0.88),
        stage_b.ADAPTIVE_METHOD: (0.90, 0.8, 1.7, 0.89),
        stage_b.CQER_METHOD: (1.00, 1.0, 2.0, 0.900),
        stage_b.RHT_METHOD: (0.70, 0.5, 1.5, 0.897),
    }
    return {
        method: [
            _metric(
                task_id=int(task["task_id"]),
                token_count=int(task[token_field]),
                delta_nll=method_values[0],
                mean_kl=method_values[1],
                cvar95_kl=method_values[2],
                top1_agreement=method_values[3],
            )
            for task in task_records
        ]
        for method, method_values in values.items()
    }


def _aggregates(
    per_task: dict[str, list[dict[str, float | int | bool]]],
) -> dict[str, dict[str, float | int]]:
    return stage_b._expected_aggregates(per_task)  # noqa: SLF001


def _bootstrap(
    per_task: dict[str, list[dict[str, float | int | bool]]],
) -> dict[str, object]:
    return paired_bootstrap_mean_improvement(
        [float(row["delta_nll"]) for row in per_task[stage_b.CQER_METHOD]],
        [float(row["delta_nll"]) for row in per_task[stage_b.RHT_METHOD]],
        samples=stage_b.BOOTSTRAP_SAMPLES,
        seed=stage_b.SEED,
    )


def _passing_gate_inputs() -> dict[str, object]:
    tasks = _task_records()
    per_task = _per_task_metrics(tasks)
    selector_diagnostics = _selector_diagnostics(tasks)
    per_task_state_errors = _per_task_state_errors(tasks)
    return {
        "aggregates": _aggregates(per_task),
        "per_task": per_task,
        "paired_bootstrap": _bootstrap(per_task),
        "aggregate_state_error": stage_b.aggregate_state_errors(
            per_task_state_errors
        ),
        "per_task_state_errors": per_task_state_errors,
        "selector_diagnostics": selector_diagnostics,
        "task_records": tasks,
        "integrity_gate": {"passed": True},
    }


def _storage() -> dict[str, dict[str, int]]:
    common = {
        "payload_bytes": stage_b.TARGET_PAYLOAD_BYTES,
        "scale_bytes": stage_b.TARGET_SCALE_BYTES,
        "mask_bytes": stage_b.TARGET_MASK_BYTES,
        "resident_bytes": stage_b.TARGET_PACKED_STATE_BYTES,
        "high_precision_groups": stage_b.TARGET_PROMOTED_ROWS,
    }
    storage = {method: dict(common) for method in stage_b.METHODS}
    for method in stage_b.QUERY_METHODS:
        storage[method].update(
            {
                "selector_auxiliary_bytes": stage_b.TARGET_SELECTOR_BYTES,
                "resident_bytes_including_selector": stage_b.TARGET_TOTAL_BYTES,
            }
        )
    return storage


def _selector_diagnostics(
    tasks: list[dict[str, int | str]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for method in stage_b.QUERY_METHODS:
        task_rows: list[dict[str, object]] = []
        for task in tasks:
            layers: list[dict[str, object]] = []
            for layer in stage_b.FROZEN_LINEAR_LAYERS:
                record: dict[str, object] = {
                    "layer_index": layer,
                    "quota": stage_b.FROZEN_LAYER_QUOTAS[layer],
                    "current_selected_count": stage_b.FROZEN_LAYER_QUOTAS[layer],
                    "state_updates": 4,
                    "observations_staged": 4,
                    "observations_committed": 4,
                    "tokens_observed": 6,
                    "pending_observation": False,
                    "confirmation_two": False,
                    "selection_method": (
                        "query_ema32_weighted_aligned_mse_reduction"
                        if method == stage_b.CQER_METHOD
                        else stage_b.RHT_METHOD
                    ),
                    "current_mask_sha256": "b" * 64,
                }
                if method == stage_b.RHT_METHOD:
                    record.update(
                        {
                            "state_codec": "right_rht_sha256_signs_v1",
                            "state_codec_seed": stage_b.SEED,
                            "state_codec_axis": "value",
                            "state_codec_normalization": "orthonormal",
                            "state_codec_persistent_tensor_bytes": 0,
                        }
                    )
                layers.append(record)
            task_rows.append({"task_id": int(task["task_id"]), "layers": layers})
        result[method] = task_rows
    return result


def _state_error_summary(*, method: str, sse_per_record: float) -> dict[str, object]:
    writes = 4
    layers = stage_b.FROZEN_LINEAR_LAYERS
    elements_per_record = 16 * 128 * 128
    record_count = writes * len(layers)
    element_count = record_count * elements_per_record
    selection_method = (
        "query_ema32_weighted_aligned_mse_reduction"
        if method == stage_b.CQER_METHOD
        else stage_b.RHT_METHOD
    )
    records: list[dict[str, object]] = []
    for write in range(writes):
        for layer in layers:
            quota = stage_b.FROZEN_LAYER_QUOTAS[layer]
            payload = 2_048 * 64 + quota * 64
            scale = 2_048 * 2
            mask = 2_048 // 8
            records.append(
                {
                    "update_index": len(records),
                    "layer_index": layer,
                    "state_index": 0,
                    "low_bits": 4,
                    "high_bits": 8,
                    "group_size": 128,
                    "scale_bits": 16,
                    "rounding": "nearest",
                    "source_dtype": "torch.float32",
                    "shape": [1, 16, 128, 128],
                    "total_groups": 2_048,
                    "high_precision_groups": quota,
                    "selection_method": selection_method,
                    "high_precision_mask_sha256": "b" * 64,
                    "baseline_bytes": elements_per_record * 4,
                    "payload_bytes": payload,
                    "scale_bytes": scale,
                    "mask_bytes": mask,
                    "resident_bytes": payload + scale + mask,
                    "relative_l2_error": 0.1,
                    "mean_squared_error": sse_per_record / elements_per_record,
                    "max_absolute_error": 0.2,
                    "element_count": elements_per_record,
                    "state_sse": sse_per_record,
                    "write_ordinal": write,
                }
            )
    coverage = [
        {
            "write_ordinal": int(record["write_ordinal"]),
            "layer_index": int(record["layer_index"]),
            "state_index": 0,
            "shape": [1, 16, 128, 128],
        }
        for record in records
    ]
    total_sse = sse_per_record * record_count
    return {
        "record_count": record_count,
        "element_count": element_count,
        "aggregate_state_sse": total_sse,
        "aggregate_state_mse": total_sse / element_count,
        "coverage": coverage,
        "per_layer": {
            str(layer): {
                "record_count": writes,
                "element_count": writes * elements_per_record,
                "state_sse": sum(sse_per_record for _ in range(writes)),
            }
            for layer in layers
        },
        "per_write": {
            str(write): {
                "record_count": len(layers),
                "element_count": len(layers) * elements_per_record,
                "state_sse": sum(sse_per_record for _ in layers),
            }
            for write in range(writes)
        },
        "records": records,
    }


def _per_task_state_errors(
    tasks: list[dict[str, int | str]],
) -> dict[str, list[dict[str, object]]]:
    return {
        method: [
            {
                "task_id": int(task["task_id"]),
                "state_error": _state_error_summary(
                    method=method,
                    sse_per_record=(
                        1.0 if method == stage_b.CQER_METHOD else 0.4
                    ),
                ),
            }
            for task in tasks
        ]
        for method in stage_b.QUERY_METHODS
    }


def _reference_aligned_summary(*, sse_per_layer: float) -> dict[str, object]:
    expected_shape = [1, 16, 128, 128]
    elements = 16 * 128 * 128
    writes: list[dict[str, object]] = []
    for write in range(4):
        layers = [
            {
                "layer_index": layer,
                "state_index": 0,
                "shape": expected_shape,
                "element_count": elements,
                "state_sse": sse_per_layer,
                "state_mse": sse_per_layer / elements,
                "relative_l2_error": 0.1,
                "max_absolute_error": 0.2,
            }
            for layer in stage_b.FROZEN_LINEAR_LAYERS
        ]
        writes.append(
            {
                "write_ordinal": write,
                "record_count": len(layers),
                "element_count": len(layers) * elements,
                "state_sse": sum(sse_per_layer for _ in layers),
                "layers": layers,
            }
        )
    total_elements = sum(int(write["element_count"]) for write in writes)
    total_sse = sum(float(write["state_sse"]) for write in writes)
    return {
        "metric": "candidate_materialized_state_minus_matched_fp32_state",
        "write_count": len(writes),
        "record_count": len(writes) * len(stage_b.FROZEN_LINEAR_LAYERS),
        "element_count": total_elements,
        "aggregate_state_sse": total_sse,
        "aggregate_state_mse": total_sse / total_elements,
        "writes": writes,
    }


def _per_task_reference_aligned_state_errors(
    tasks: list[dict[str, int | str]],
) -> dict[str, list[dict[str, object]]]:
    return {
        method: [
            {
                "task_id": int(task["task_id"]),
                "state_error": _reference_aligned_summary(
                    sse_per_layer=(
                        2.0 if method == stage_b.CQER_METHOD else 1.0
                    )
                ),
            }
            for task in tasks
        ]
        for method in stage_b.QUERY_METHODS
    }


def _unit_evidence() -> dict[str, object]:
    return {
        "production_self_check": {
            "inverse_relative_l2": 1e-7,
            "inverse_relative_l2_threshold": stage_b.MAX_RHT_INVERSE_RELATIVE_L2,
            "physical_pack_matches_transformed_qdq": True,
            "physical_pack_max_abs_difference": 0.0,
            "physical_pack_storage_bytes": (
                stage_b.PRODUCTION_SELF_CHECK_STORAGE_BYTES
            ),
            "sign_schedule_sha256": stage_b.EXPECTED_SIGN_SCHEDULE_SHA256,
            "expected_sign_schedule_sha256": (
                stage_b.EXPECTED_SIGN_SCHEDULE_SHA256
            ),
            "seed": stage_b.SEED,
            "device": "cpu",
            "dtype": "torch.float32",
        },
        "independent_dense_reference": {
            "reference": stage_b.INDEPENDENT_REFERENCE_DESCRIPTION,
            "passed": True,
            "signs_exact": True,
            "encode_max_abs_difference": 1e-7,
            "encode_max_abs_threshold": (
                stage_b.INDEPENDENT_ENCODE_MAX_ABS_THRESHOLD
            ),
            "physical_pack_max_abs_difference": 1e-7,
            "physical_pack_max_abs_threshold": (
                stage_b.INDEPENDENT_PACK_MAX_ABS_THRESHOLD
            ),
        },
    }


def _integrity_flags() -> dict[str, bool]:
    return {
        "repository_clean_at_start": True,
        "repository_clean_at_end": True,
        "repository_commit_stable": True,
        "source_hashes_stable": True,
        "stage_a_artifact_committed_authenticated_and_passed": True,
        "identity_artifact_committed_and_authenticated": True,
        "identity_authenticated_before_model_weights": True,
        "identity_row_plan_authenticated": True,
        "identity_source_freeze_matches_current_bytes": True,
        "imported_modules_resolved_to_authenticated_repository": True,
        "runtime_environment_authenticated_before_dataset_access": True,
        (
            "protected_window_8_16_content_selected_retained_canonicalized_"
            "formatted_tokenized_passed_to_model_or_evaluated"
        ): False,
    }


def _token_traces(
    tasks: list[dict[str, int | str]],
    *,
    token_field: str,
) -> dict[str, list[dict[str, object]]]:
    method_values = {
        stage_b.STATIC_METHOD: (0.95, 0.9),
        stage_b.ADAPTIVE_METHOD: (0.90, 0.8),
        stage_b.CQER_METHOD: (1.00, 1.0),
        stage_b.RHT_METHOD: (0.70, 0.5),
    }
    result: dict[str, list[dict[str, object]]] = {}
    for method, (delta, kl) in method_values.items():
        rows: list[dict[str, object]] = []
        for task in tasks:
            count = int(task[token_field])
            payload = {
                "token_count": count,
                "kl": [kl] * count,
                "reference_nll": [1.0] * count,
                "candidate_nll": [1.0 + delta] * count,
                "top1_agreement": [True] * count,
                "outputs_finite": [True] * count,
            }
            rows.append(
                {
                    "task_id": int(task["task_id"]),
                    **payload,
                    "canonical_primitives_sha256": stage_b._sha256_bytes(  # noqa: SLF001
                        stage_b.canonical_json_bytes(payload)
                    ),
                }
            )
        result[method] = rows
    return result


def _summaries_from_traces(
    traces: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, float | int | bool]]]:
    return {
        method: [
            {
                "task_id": int(trace["task_id"]),
                **stage_b._summary_from_trace(trace),  # noqa: SLF001
            }
            for trace in method_traces
        ]
        for method, method_traces in traces.items()
    }


def _passing_integrity_inputs() -> dict[str, object]:
    tasks = _task_records()
    traces = _token_traces(tasks, token_field="aligned_scored_tokens")
    full_traces = _token_traces(tasks, token_field="full_code_scored_tokens")
    per_task = _summaries_from_traces(traces)
    full = _summaries_from_traces(full_traces)
    return {
        "per_task": per_task,
        "per_task_full_code": full,
        "per_task_token_traces": traces,
        "per_task_full_code_token_traces": full_traces,
        "aggregates": _aggregates(per_task),
        "aggregates_full_code": _aggregates(full),
        "storage": _storage(),
        "selector_diagnostics": _selector_diagnostics(tasks),
        "per_task_state_errors": _per_task_state_errors(tasks),
        "per_task_reference_aligned_state_errors": (
            _per_task_reference_aligned_state_errors(tasks)
        ),
        "task_records": tasks,
        "unit_evidence": _unit_evidence(),
        "integrity": _integrity_flags(),
    }


def _result_identity_dataset(
    tasks: list[dict[str, int | str]],
) -> dict[str, object]:
    resolver = stage_b.identity_resolver
    ordered_ids = [int(task["task_id"]) for task in tasks]
    manifest = {
        "schema": resolver.MBPP_MANIFEST_SCHEMA,
        "dataset_id": resolver.MBPP_DATASET_ID,
        "config": resolver.MBPP_CONFIG,
        "revision": resolver.MBPP_REVISION,
        "phase": "calibration",
        "source_split": resolver.mbpp_source_split("calibration"),
        "selection_namespace": resolver.MBPP_SELECTION_NAMESPACE,
        "formatter_version": resolver.MBPP_FORMATTER_VERSION,
        "row_count": stage_b.STAGE_B_LIMIT,
        "rows": sorted(
            (
                {"task_id": int(task["task_id"]), "sha256": task["row_sha256"]}
                for task in tasks
            ),
            key=lambda row: row["task_id"],
        ),
    }
    content_hash = stage_b.mbpp_manifest_content_sha256(manifest)
    token_hash = resolver.token_manifest_sha256(tasks)
    ordered_hash = resolver.ordered_identity_sha256(
        content_manifest_sha256=content_hash,
        task_records=tasks,
    )
    source_rows = resolver.MBPP_CALIBRATION_SIZE
    target_rows_seen = stage_b.STAGE_B_STOP
    application_keys = (
        "selected",
        "retained",
        "canonicalized",
        "formatted",
        "tokenized",
        "passed_to_model",
        "evaluated",
    )
    data_access = {
        "transport_limitation": stage_b.DATA_ACCESS_TRANSPORT_LIMITATION,
        "ranking_pass": {
            "transport_records_yielded": source_rows,
            "task_id_fields_inspected": source_rows,
            "non_task_id_fields_read_by_recurquant": 0,
            "row_mappings_retained": 0,
        },
        "target_load_pass": {
            "transport_records_yielded": target_rows_seen,
            "task_id_fields_inspected": target_rows_seen,
            "non_target_content_fields_read_by_recurquant": 0,
            "target_rows_retained_and_canonicalized": stage_b.STAGE_B_LIMIT,
        },
        "application_task_id_sets": {
            "selected": ordered_ids,
            "retained": ordered_ids,
            "canonicalized": ordered_ids,
            "formatted": ordered_ids,
            "tokenized": ordered_ids,
            "passed_to_model": [],
            "evaluated": [],
        },
        "protected_window_intersection": {
            key: False for key in application_keys
        },
    }
    return {
        "id": resolver.MBPP_DATASET_ID,
        "config": resolver.MBPP_CONFIG,
        "revision": resolver.MBPP_REVISION,
        "phase": "calibration",
        "source_split": resolver.mbpp_source_split("calibration"),
        "selection_namespace": resolver.MBPP_SELECTION_NAMESPACE,
        "formatter_version": resolver.MBPP_FORMATTER_VERSION,
        "selection_mode": "task_id_ranking_then_exact_task_id_stream",
        "selection_window": {
            "offset": stage_b.STAGE_B_OFFSET,
            "limit": stage_b.STAGE_B_LIMIT,
            "stop_exclusive": stage_b.STAGE_B_STOP,
        },
        "protected_window": {
            "offset": stage_b.PROTECTED_WINDOW[0],
            "stop_exclusive": stage_b.PROTECTED_WINDOW[1],
            "content_retained_canonicalized_or_tokenized": False,
        },
        "ordered_task_ids": ordered_ids,
        "manifest": manifest,
        "content_manifest_sha256": content_hash,
        "token_manifest_sha256": token_hash,
        "ordered_identity_sha256": ordered_hash,
        "tasks": tasks,
        "totals": {
            "source_train_rows_seen_by_task_id_only": source_rows,
            "retained_rows": stage_b.STAGE_B_LIMIT,
            "prompt_tokens": sum(int(task["prompt_tokens"]) for task in tasks),
            "code_tokens": sum(int(task["code_tokens"]) for task in tasks),
            "aligned_scored_tokens": sum(
                int(task["aligned_scored_tokens"]) for task in tasks
            ),
            "full_code_scored_tokens": sum(
                int(task["full_code_scored_tokens"]) for task in tasks
            ),
        },
        "data_access": data_access,
    }


def _result_runtime_environment() -> dict[str, object]:
    resolver = stage_b.identity_resolver
    version = ".".join(str(item) for item in resolver.STAGE_A_PYTHON_VERSION)
    return {
        "schema": resolver.RUNTIME_ENVIRONMENT_SCHEMA,
        "stage_a_binding": {
            "artifact_kind": stage_b.STAGE_A_ARTIFACT_KIND,
            "file_sha256": resolver.STAGE_A_FILE_SHA256,
            "canonical_evidence_sha256": resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256,
        },
        "python": {
            "major": resolver.STAGE_A_PYTHON_VERSION[0],
            "minor": resolver.STAGE_A_PYTHON_VERSION[1],
            "micro": resolver.STAGE_A_PYTHON_VERSION[2],
            "version": version,
        },
        "packages": dict(resolver.STAGE_A_PACKAGE_VERSIONS),
        "cuda": dict(resolver.STAGE_A_CUDA_CONTRACT),
        "runtime_matches_stage_a": True,
        "local_files_only": False,
    }


def _stage_b_result_document() -> dict[str, object]:
    inputs = _passing_integrity_inputs()
    per_task = inputs["per_task"]
    state_aggregates = stage_b.aggregate_state_errors(
        inputs["per_task_state_errors"]
    )
    bootstrap = _bootstrap(per_task)
    integrity = stage_b.evaluate_stage_b_integrity(**inputs)
    gate = stage_b.evaluate_stage_b_gate(
        aggregates=inputs["aggregates"],
        per_task=per_task,
        paired_bootstrap=bootstrap,
        aggregate_state_error=state_aggregates,
        per_task_state_errors=inputs["per_task_state_errors"],
        selector_diagnostics=inputs["selector_diagnostics"],
        task_records=inputs["task_records"],
        integrity_gate=integrity,
    )
    tasks = inputs["task_records"]
    ordered_ids = [int(task["task_id"]) for task in tasks]
    identity_dataset = _result_identity_dataset(tasks)
    identity_access = identity_dataset["data_access"]
    application_keys = (
        "selected",
        "retained",
        "canonicalized",
        "formatted",
        "tokenized",
        "passed_to_model",
        "evaluated",
    )
    source_hashes = {path: "c" * 64 for path in stage_b.SOURCE_FILES}
    runtime_environment = _result_runtime_environment()
    python_version = runtime_environment["python"]["version"]
    evidence: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": stage_b.ARTIFACT_KIND,
        "diagnostic_only": True,
        "claim_boundary": stage_b.RESULT_CLAIM_BOUNDARY,
        "created_at_utc": "2026-07-26T00:00:00+00:00",
        "protocol": {
            "name": "Experiment 009 Stage B",
            "ranked_window": [stage_b.STAGE_B_OFFSET, stage_b.STAGE_B_STOP],
            "protected_ranked_window": list(stage_b.PROTECTED_WINDOW),
            "protected_window_application_content_intersection": False,
            "methods_locked": True,
            "thresholds_locked": True,
            "bootstrap_samples": stage_b.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": stage_b.SEED,
        },
        "prerequisite_artifacts": {
            "stage_a": {
                "path": stage_b.identity_resolver.STAGE_A_ARTIFACT_RELATIVE_PATH,
                "sha256": stage_b.identity_resolver.STAGE_A_FILE_SHA256,
                "git_blob_sha256": stage_b.identity_resolver.STAGE_A_FILE_SHA256,
                "artifact_sha256": stage_b.identity_resolver.STAGE_A_FILE_SHA256,
                "canonical_evidence_sha256": (
                    stage_b.identity_resolver.STAGE_A_CANONICAL_EVIDENCE_SHA256
                ),
                "implementation_commit": (
                    stage_b.identity_resolver.STAGE_A_IMPLEMENTATION_COMMIT
                ),
                "gate_recomputed_and_passed": "true",
                "artifact_kind": stage_b.STAGE_A_ARTIFACT_KIND,
                "historical_path_privacy_limitation": (
                    stage_b.STAGE_A_PATH_PRIVACY_LIMITATION
                ),
            },
            "stage_b_identity": {
                "path": "evidence/experiment009-stage-b-identity.json",
                "sha256": "d" * 64,
                "git_blob_sha256": "d" * 64,
                "artifact_sha256": "d" * 64,
                "canonical_evidence_sha256": "e" * 64,
                "resolver_commit": "f" * 40,
                "artifact_kind": stage_b.IDENTITY_ARTIFACT_KIND,
            },
            "committed_row_plan": {
                "passed": True,
                "canonical_plan_sha256": "1" * 64,
                "method": stage_b.TARGET_FISHER_SCORE,
                "promoted_group_count": stage_b.TARGET_PROMOTED_ROWS,
                "resident_bytes": stage_b.TARGET_PACKED_STATE_BYTES,
                "selector_binding": {
                    "selector_file_sha256": (
                        stage_b.identity_resolver.SELECTOR_FILE_SHA256
                    ),
                    "selector_canonical_evidence_sha256": (
                        stage_b.identity_resolver.SELECTOR_CANONICAL_EVIDENCE_SHA256
                    ),
                    "loss_selector_file_sha256": (
                        stage_b.identity_resolver.LOSS_SELECTOR_FILE_SHA256
                    ),
                    "loss_selector_canonical_evidence_sha256": (
                        stage_b.identity_resolver.LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
                    ),
                },
            },
            "identity_source_freeze": {
                "passed": True,
                "path_count": len(stage_b.SOURCE_FILES),
                "paths": list(stage_b.SOURCE_FILES),
                "sha256": source_hashes,
            },
        },
        "model": {
            "id": stage_b.identity_resolver.MODEL_ID,
            "revision": stage_b.identity_resolver.MODEL_REVISION,
            "dtype": "torch.bfloat16",
            "device": "cuda",
        },
        "methods": list(stage_b.METHODS),
        "dataset": {
            "phase": "calibration",
            "selection_mode": (
                "committed_exact_task_ids_from_ranked_window_32_64"
            ),
            "identity": identity_dataset,
            "identity_validation": {
                "passed": True,
                "authenticated_before_model_weights": True,
                "ordered_task_ids": ordered_ids,
                "content_manifest_sha256": (
                    identity_dataset["content_manifest_sha256"]
                ),
                "token_manifest_sha256": identity_dataset["token_manifest_sha256"],
                "ordered_identity_sha256": identity_dataset["ordered_identity_sha256"],
            },
            "data_access": {
                "transport_limitation": stage_b.DATA_ACCESS_TRANSPORT_LIMITATION,
                "identity_resolution": copy.deepcopy(identity_access),
                "evaluator_target_load": copy.deepcopy(
                    identity_access["target_load_pass"]
                ),
                "evaluator_application_task_id_sets": {
                    key: ordered_ids for key in application_keys
                },
                "protected_window_intersection": {
                    key: False for key in application_keys
                },
                "non_target_source_records": {
                    "recurquant_fields_inspected": ["task_id"],
                    (
                        "content_retained_canonicalized_formatted_tokenized_"
                        "or_evaluated"
                    ): False,
                },
            },
            "identity_authenticated_before_model_weights": True,
            stage_b.PROTECTED_EVALUATION_FIELD: False,
        },
        "metric_contract": {
            "primary": (
                "task-macro aligned excess next-token NLL versus FP32 state"
            ),
            "aligned_excludes": "prompt-to-first-code-token prediction",
            "secondary": "task-macro full-code metrics",
            "paired_bootstrap_samples": stage_b.BOOTSTRAP_SAMPLES,
            "paired_bootstrap_seed": stage_b.SEED,
        },
        "storage": {
            "fp32_reference_recurrent_state_bytes": 10_000_000,
            "candidates": inputs["storage"],
        },
        "aggregates": inputs["aggregates"],
        "aggregates_full_code_secondary": inputs["aggregates_full_code"],
        "per_task": per_task,
        "per_task_full_code_secondary": inputs["per_task_full_code"],
        "per_task_token_primitives": inputs["per_task_token_traces"],
        "per_task_full_code_token_primitives": (
            inputs["per_task_full_code_token_traces"]
        ),
        "paired_bootstrap_cqer_minus_rht_aligned_delta_nll": bootstrap,
        "selector_diagnostics": inputs["selector_diagnostics"],
        "state_error": {
            "primary_gate_metric": stage_b.PRIMARY_STATE_ERROR_METRIC,
            "aggregates": state_aggregates,
            "per_task": inputs["per_task_state_errors"],
            "reference_aligned_secondary": {
                "metric": stage_b.REFERENCE_ALIGNED_STATE_ERROR_METRIC,
                "per_task": inputs["per_task_reference_aligned_state_errors"]
            },
        },
        "unit_evidence": inputs["unit_evidence"],
        "integrity_inputs": inputs["integrity"],
        "stage_b_integrity": integrity,
        "stage_b_gate": gate,
        "runtime_environment": runtime_environment,
        "environment": {
            "python": f"{python_version} test-build",
            "platform": "Windows-test",
            "packages": dict(stage_b.identity_resolver.STAGE_A_PACKAGE_VERSIONS),
            "cuda_available": True,
            "cuda_runtime": (
                stage_b.identity_resolver.STAGE_A_CUDA_CONTRACT[
                    "runtime_version"
                ]
            ),
            "gpu": "Test CUDA GPU",
        },
        "repository": {
            "commit": "2" * 40,
            "start": {
                "commit": "2" * 40,
                "worktree_clean": True,
                "status": [],
            },
            "end": {
                "commit": "2" * 40,
                "worktree_clean": True,
                "status": [],
            },
            "stable_commit": True,
        },
        "source_files": {
            "paths": list(stage_b.SOURCE_FILES),
            "sha256_start": source_hashes,
            "sha256_end": dict(source_hashes),
            "stable": True,
            "imported_module_paths": dict(stage_b.RESULT_IMPORTED_MODULE_PATHS),
        },
        "command_template": [
            "python",
            "scripts/evaluate_rht_cqer_stage_b.py",
            "--stage-a-artifact",
            "<committed-stage-a-artifact>",
            "--identity-artifact",
            "<committed-stage-b-identity-artifact>",
            "--output",
            "<ignored-or-external-output>",
            "--device",
            "auto",
        ],
    }
    return {
        "schema_version": 1,
        "artifact_kind": stage_b.ARTIFACT_KIND,
        "canonical_evidence_sha256": stage_b._sha256_bytes(  # noqa: SLF001
            stage_b.canonical_json_bytes(evidence)
        ),
        "evidence": evidence,
    }


def _write_result_document(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(stage_b.canonical_json_bytes(document))


def _rehash_result_document(document: dict[str, object]) -> None:
    document["canonical_evidence_sha256"] = stage_b._sha256_bytes(  # noqa: SLF001
        stage_b.canonical_json_bytes(document["evidence"])
    )


def _recompute_result_outcomes(document: dict[str, object]) -> None:
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    dataset = evidence["dataset"]
    assert isinstance(dataset, dict)
    identity_dataset = dataset["identity"]
    assert isinstance(identity_dataset, dict)
    state_error = evidence["state_error"]
    assert isinstance(state_error, dict)
    reference_aligned = state_error["reference_aligned_secondary"]
    assert isinstance(reference_aligned, dict)
    storage = evidence["storage"]
    assert isinstance(storage, dict)
    integrity = stage_b.evaluate_stage_b_integrity(
        per_task=evidence["per_task"],
        per_task_full_code=evidence["per_task_full_code_secondary"],
        per_task_token_traces=evidence["per_task_token_primitives"],
        per_task_full_code_token_traces=(
            evidence["per_task_full_code_token_primitives"]
        ),
        aggregates=evidence["aggregates"],
        aggregates_full_code=evidence["aggregates_full_code_secondary"],
        storage=storage["candidates"],
        selector_diagnostics=evidence["selector_diagnostics"],
        per_task_state_errors=state_error["per_task"],
        per_task_reference_aligned_state_errors=reference_aligned["per_task"],
        task_records=identity_dataset["tasks"],
        unit_evidence=evidence["unit_evidence"],
        integrity=evidence["integrity_inputs"],
    )
    evidence["stage_b_integrity"] = integrity
    evidence["stage_b_gate"] = stage_b.evaluate_stage_b_gate(
        aggregates=evidence["aggregates"],
        per_task=evidence["per_task"],
        paired_bootstrap=(
            evidence["paired_bootstrap_cqer_minus_rht_aligned_delta_nll"]
        ),
        aggregate_state_error=state_error["aggregates"],
        per_task_state_errors=state_error["per_task"],
        selector_diagnostics=evidence["selector_diagnostics"],
        task_records=identity_dataset["tasks"],
        integrity_gate=integrity,
    )
    _rehash_result_document(document)


def test_stage_b_gate_passes_exactly_eight_frozen_checks() -> None:
    gate = stage_b.evaluate_stage_b_gate(**_passing_gate_inputs())

    assert gate["passed"] is True
    assert gate["integrity_passed"] is True
    assert len(gate["advancement_checks"]) == 8
    assert all(check["passed"] is True for check in gate["advancement_checks"].values())


@pytest.mark.parametrize(
    "check_name",
    [
        "macro_excess_nll_relative_reduction",
        "paired_95pct_lower_bound_above_zero",
        "at_least_20_task_level_excess_nll_wins",
        "lower_macro_mean_kl",
        "macro_cvar95_kl_not_higher",
        "macro_top1_disadvantage_at_most_0_005",
        "maximum_task_excess_nll_disadvantage_at_most_0_25",
        "aggregate_state_sse_relative_reduction",
    ],
)
def test_each_stage_b_advancement_condition_fails_closed(check_name: str) -> None:
    inputs = copy.deepcopy(_passing_gate_inputs())
    per_task = inputs["per_task"]

    if check_name == "macro_excess_nll_relative_reduction":
        for row in per_task[stage_b.RHT_METHOD]:
            row.update({"candidate_nll": 1.85, "delta_nll": 0.85})
        inputs["aggregates"] = _aggregates(per_task)
        inputs["paired_bootstrap"] = _bootstrap(per_task)
    elif check_name == "paired_95pct_lower_bound_above_zero":
        inputs["paired_bootstrap"]["confidence_interval"] = [-0.01, 0.2]
    elif check_name == "at_least_20_task_level_excess_nll_wins":
        for index, row in enumerate(per_task[stage_b.RHT_METHOD]):
            delta = 0.5 if index < 19 else 1.1
            row.update({"candidate_nll": 1.0 + delta, "delta_nll": delta})
        inputs["aggregates"] = _aggregates(per_task)
        inputs["paired_bootstrap"] = _bootstrap(per_task)
    elif check_name == "lower_macro_mean_kl":
        for row in per_task[stage_b.RHT_METHOD]:
            row["mean_kl"] = 1.0
        inputs["aggregates"] = _aggregates(per_task)
    elif check_name == "macro_cvar95_kl_not_higher":
        for row in per_task[stage_b.RHT_METHOD]:
            row.update({"cvar95_kl": 2.1, "max_kl": 2.35})
        inputs["aggregates"] = _aggregates(per_task)
    elif check_name == "macro_top1_disadvantage_at_most_0_005":
        for row in per_task[stage_b.RHT_METHOD]:
            row["top1_agreement"] = 0.894
        inputs["aggregates"] = _aggregates(per_task)
    elif check_name == "maximum_task_excess_nll_disadvantage_at_most_0_25":
        per_task[stage_b.RHT_METHOD][0].update(
            {"candidate_nll": 2.3, "delta_nll": 1.3}
        )
        inputs["aggregates"] = _aggregates(per_task)
        inputs["paired_bootstrap"] = _bootstrap(per_task)
    elif check_name == "aggregate_state_sse_relative_reduction":
        for task in inputs["per_task_state_errors"][stage_b.RHT_METHOD]:
            task["state_error"] = _state_error_summary(
                method=stage_b.RHT_METHOD,
                sse_per_record=0.51,
            )
        inputs["aggregate_state_error"] = stage_b.aggregate_state_errors(
            inputs["per_task_state_errors"]
        )

    gate = stage_b.evaluate_stage_b_gate(**inputs)

    assert gate["passed"] is False
    assert gate["advancement_checks"][check_name]["passed"] is False


def test_stage_b_gate_cannot_pass_when_integrity_fails() -> None:
    inputs = _passing_gate_inputs()
    inputs["integrity_gate"] = {"passed": False}

    gate = stage_b.evaluate_stage_b_gate(**inputs)

    assert gate["passed"] is False
    assert gate["integrity_passed"] is False
    assert all(check["passed"] is True for check in gate["advancement_checks"].values())


def test_stage_b_gate_rejects_detached_macro_aggregate() -> None:
    inputs = _passing_gate_inputs()
    for row in inputs["per_task"][stage_b.RHT_METHOD]:
        row.update({"candidate_nll": 1.99, "delta_nll": 0.99})
    inputs["paired_bootstrap"] = _bootstrap(inputs["per_task"])

    with pytest.raises(ValueError, match="aggregate differs"):
        stage_b.evaluate_stage_b_gate(**inputs)


def test_stage_b_gate_rejects_detached_state_aggregate() -> None:
    inputs = _passing_gate_inputs()
    inputs["aggregate_state_error"][stage_b.RHT_METHOD][
        "aggregate_state_sse"
    ] = 0.0

    with pytest.raises(ValueError, match="differs from raw state-error"):
        stage_b.evaluate_stage_b_gate(**inputs)


def test_stage_b_result_loader_recomputes_the_exact_canonical_result(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    path = tmp_path / "stage-b.json"
    _write_result_document(path, document)

    evidence, verification = stage_b.load_and_validate_stage_b_result_artifact(
        path
    )

    assert evidence == document["evidence"]
    assert verification["passed"] is True
    assert verification["integrity_passed"] is True
    assert verification["advancement_passed"] is True
    assert verification["canonical_round_trip"] is True
    assert verification["advancement_check_count"] == 8
    identity = evidence["dataset"]["identity"]
    task_ids = [record["task_id"] for record in identity["tasks"]]
    manifest_ids = [row["task_id"] for row in identity["manifest"]["rows"]]
    assert task_ids != manifest_ids
    assert manifest_ids == sorted(task_ids)


def _rehash_result_identity_dataset(identity: dict[str, object]) -> None:
    manifest = identity["manifest"]
    tasks = identity["tasks"]
    assert isinstance(manifest, dict)
    assert isinstance(tasks, list)
    content_hash = stage_b.mbpp_manifest_content_sha256(manifest)
    identity["content_manifest_sha256"] = content_hash
    identity["token_manifest_sha256"] = (
        stage_b.identity_resolver.token_manifest_sha256(tasks)
    )
    identity["ordered_identity_sha256"] = (
        stage_b.identity_resolver.ordered_identity_sha256(
            content_manifest_sha256=content_hash,
            task_records=tasks,
        )
    )


def test_stage_b_result_identity_rejects_swapped_manifest_hashes() -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows[0]["sha256"], rows[1]["sha256"] = rows[1]["sha256"], rows[0]["sha256"]
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest rows do not match tasks"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_stage_b_result_identity_rejects_manifest_row_set_drift(
    mutation: str,
) -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    else:
        rows.append({"task_id": 99_999, "sha256": "f" * 64})
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest rows|manifest task IDs"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_unique_unknown_manifest_task_id() -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows[-1]["task_id"] = 99_999
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest rows do not match tasks"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


@pytest.mark.parametrize("field_drift", ["missing", "extra"])
def test_stage_b_result_identity_rejects_manifest_row_field_drift(
    field_drift: str,
) -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    if field_drift == "missing":
        del rows[0]["sha256"]
    else:
        rows[0]["unexpected"] = "field"
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest row 0 fields drifted"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_invalid_manifest_sha256() -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows[0]["sha256"] = "not-a-sha256"
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest row 0.*SHA-256"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_boolean_manifest_task_id() -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows[0]["task_id"] = True
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest row 0 task_id"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_noncanonical_manifest_order() -> None:
    identity = _result_identity_dataset(_task_records())
    manifest = identity["manifest"]
    assert isinstance(manifest, dict)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    rows[0], rows[1] = rows[1], rows[0]
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="canonical task-ID order"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_rehashed_task_reorder() -> None:
    identity = _result_identity_dataset(_task_records())
    tasks = identity["tasks"]
    assert isinstance(tasks, list)
    tasks[0], tasks[1] = tasks[1], tasks[0]
    identity["ordered_task_ids"] = [task["task_id"] for task in tasks]
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="ordered window"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_identity_rejects_task_row_hash_tamper() -> None:
    identity = _result_identity_dataset(_task_records())
    tasks = identity["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["row_sha256"] = "f" * 64
    _rehash_result_identity_dataset(identity)

    with pytest.raises(ValueError, match="manifest rows do not match tasks"):
        stage_b._validate_result_identity_dataset(identity)  # noqa: SLF001


def test_stage_b_result_loader_authenticates_an_exact_negative_result(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    evidence = document["evidence"]
    traces = evidence["per_task_token_primitives"]
    for trace in traces[stage_b.RHT_METHOD]:
        token_count = int(trace["token_count"])
        trace["top1_agreement"] = [False] * token_count
        payload = {
            key: trace[key]
            for key in (
                "token_count",
                "kl",
                "reference_nll",
                "candidate_nll",
                "top1_agreement",
                "outputs_finite",
            )
        }
        trace["canonical_primitives_sha256"] = stage_b._sha256_bytes(  # noqa: SLF001
            stage_b.canonical_json_bytes(payload)
        )
    evidence["per_task"] = _summaries_from_traces(traces)
    evidence["aggregates"] = _aggregates(evidence["per_task"])
    evidence["paired_bootstrap_cqer_minus_rht_aligned_delta_nll"] = _bootstrap(
        evidence["per_task"]
    )
    _recompute_result_outcomes(document)
    path = tmp_path / "stage-b-negative.json"
    _write_result_document(path, document)

    loaded, verification = stage_b.load_and_validate_stage_b_result_artifact(path)

    assert verification["passed"] is True
    assert verification["integrity_passed"] is True
    assert verification["advancement_passed"] is False
    assert (
        loaded["stage_b_gate"]["advancement_checks"][
            "macro_top1_disadvantage_at_most_0_005"
        ]["passed"]
        is False
    )


def test_stage_b_result_loader_rejects_contradictory_unit_passed_flag(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    document["evidence"]["unit_evidence"]["independent_dense_reference"][
        "passed"
    ] = False
    _recompute_result_outcomes(document)
    path = tmp_path / "contradictory-unit-pass.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="passed flag contradicts"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_incomplete_unit_evidence_schema(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    del document["evidence"]["unit_evidence"]["independent_dense_reference"][
        "reference"
    ]
    _recompute_result_outcomes(document)
    path = tmp_path / "incomplete-unit-schema.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="reference fields drifted"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_missing_production_provenance(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    del document["evidence"]["protocol"]
    _rehash_result_document(document)
    path = tmp_path / "missing-provenance.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="evidence fields drifted"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_local_prerequisite_path(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    document["evidence"]["prerequisite_artifacts"]["stage_b_identity"][
        "path"
    ] = "/private/identity.json"
    _rehash_result_document(document)
    path = tmp_path / "local-path.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="repository-relative path"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_state_metric_label_drift(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    document["evidence"]["state_error"]["primary_gate_metric"] = "generic MSE"
    _rehash_result_document(document)
    path = tmp_path / "metric-label.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="state-error metric label drifted"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_rehashed_aggregate_tamper(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    document["evidence"]["aggregates"][stage_b.RHT_METHOD][
        "macro_delta_nll"
    ] = 0.01
    _rehash_result_document(document)
    path = tmp_path / "aggregate-tamper.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="integrity"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_result_loader_rejects_rehashed_state_aggregate_tamper(
    tmp_path: Path,
) -> None:
    document = _stage_b_result_document()
    document["evidence"]["state_error"]["aggregates"][stage_b.RHT_METHOD][
        "aggregate_state_sse"
    ] = 0.0
    _rehash_result_document(document)
    path = tmp_path / "state-tamper.json"
    _write_result_document(path, document)

    with pytest.raises(ValueError, match="differs from raw state-error"):
        stage_b.load_and_validate_stage_b_result_artifact(path)


def test_stage_b_integrity_accepts_complete_exact_contract() -> None:
    integrity = stage_b.evaluate_stage_b_integrity(**_passing_integrity_inputs())

    assert integrity["passed"] is True
    assert all(check["passed"] is True for check in integrity["checks"].values())


@pytest.mark.parametrize(
    ("section", "mutation", "failed_check"),
    [
        (
            "integrity",
            lambda values: values.update(
                {
                    (
                        "protected_window_8_16_content_selected_retained_"
                        "canonicalized_formatted_tokenized_passed_to_model_or_evaluated"
                    ): True
                }
            ),
            "committed_clean_stable_provenance",
        ),
        (
            "storage",
            lambda values: values[stage_b.RHT_METHOD].update({"resident_bytes": 1}),
            "exact_physical_storage",
        ),
        (
            "selector_diagnostics",
            lambda values: values[stage_b.RHT_METHOD][0]["layers"][0].update(
                {"state_codec_seed": 1}
            ),
            "exact_quotas_and_query_handshakes",
        ),
        (
            "per_task_state_errors",
            lambda values: values[stage_b.RHT_METHOD][0]["state_error"].update(
                {"coverage": []}
            ),
            "complete_matched_state_error_coverage",
        ),
        (
            "per_task",
            lambda values: values[stage_b.RHT_METHOD][0].update(
                {"candidate_nll": 99.0}
            ),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task",
            lambda values: values[stage_b.RHT_METHOD][0].update({"mean_kl": -0.1}),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task",
            lambda values: values[stage_b.RHT_METHOD][0].update(
                {"mean_kl": 2.0, "cvar95_kl": 1.0}
            ),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task",
            lambda values: values[stage_b.RHT_METHOD][0].update(
                {"top1_agreement": 1.1}
            ),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task",
            lambda values: values[stage_b.RHT_METHOD][0].update(
                {"reference_nll": 1.1, "candidate_nll": 1.8}
            ),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task_token_traces",
            lambda values: values[stage_b.RHT_METHOD][0].update(
                {"canonical_primitives_sha256": "0" * 64}
            ),
            "finite_exact_task_macro_metrics",
        ),
        (
            "per_task_state_errors",
            lambda values: values[stage_b.RHT_METHOD][0]["state_error"]["records"][
                0
            ].update({"mean_squared_error": 0.5}),
            "complete_matched_state_error_coverage",
        ),
        (
            "per_task_state_errors",
            lambda values: values[stage_b.RHT_METHOD][0]["state_error"]["records"][
                0
            ].update({"resident_bytes": 1}),
            "complete_matched_state_error_coverage",
        ),
        (
            "per_task_state_errors",
            lambda values: values[stage_b.RHT_METHOD][0]["state_error"]["records"][
                -1
            ].update({"high_precision_mask_sha256": "0" * 64}),
            "complete_matched_state_error_coverage",
        ),
        (
            "per_task_reference_aligned_state_errors",
            lambda values: values[stage_b.RHT_METHOD][0]["state_error"].update(
                {"aggregate_state_sse": 999.0}
            ),
            "authenticated_reference_aligned_state_secondary",
        ),
        (
            "unit_evidence",
            lambda values: values["production_self_check"].update(
                {"inverse_relative_l2": 1e-3}
            ),
            "deterministic_rht_codec_self_check",
        ),
    ],
)
def test_stage_b_integrity_fails_closed_on_contract_drift(
    section: str,
    mutation: object,
    failed_check: str,
) -> None:
    inputs = copy.deepcopy(_passing_integrity_inputs())
    mutation(inputs[section])

    integrity = stage_b.evaluate_stage_b_integrity(**inputs)

    assert integrity["passed"] is False
    assert integrity["checks"][failed_check]["passed"] is False


def test_stage_b_identity_tasks_accept_only_exact_ordered_window() -> None:
    tasks = _task_records()
    identity = {
        "dataset": {
            "tasks": tasks,
            "ordered_task_ids": [record["task_id"] for record in tasks],
            "selection_window": {
                "offset": stage_b.STAGE_B_OFFSET,
                "limit": stage_b.STAGE_B_LIMIT,
                "stop_exclusive": stage_b.STAGE_B_STOP,
            },
            "protected_window": {
                "offset": stage_b.PROTECTED_WINDOW[0],
                "stop_exclusive": stage_b.PROTECTED_WINDOW[1],
                "content_retained_canonicalized_or_tokenized": False,
            },
        }
    }

    assert stage_b._identity_task_records(identity) == tuple(tasks)  # noqa: SLF001

    drifted = copy.deepcopy(identity)
    drifted["dataset"]["tasks"][0]["rank"] = 31
    with pytest.raises(ValueError, match=r"\[32, 64\)"):
        stage_b._identity_task_records(drifted)  # noqa: SLF001


def test_stage_b_cli_exposes_no_quality_or_window_tuning_flags() -> None:
    required = [
        "--stage-a-artifact",
        "stage-a.json",
        "--identity-artifact",
        "identity.json",
        "--output",
        "result.json",
    ]
    parsed = stage_b.parse_args(required)
    assert parsed.device == "auto"

    with pytest.raises(SystemExit):
        stage_b.parse_args([*required, "--bootstrap-samples", "10"])


def test_identity_source_freeze_survives_canonical_json_round_trip() -> None:
    source_hashes = {
        path: stage_b._sha256_bytes(path.encode("utf-8"))  # noqa: SLF001
        for path in stage_b.SOURCE_FILES
    }
    identity = {
        "source_files": {
            "paths": list(stage_b.SOURCE_FILES),
            "sha256_start": source_hashes,
            "sha256_end": source_hashes,
            "stable": True,
        }
    }
    round_tripped = json.loads(stage_b.canonical_json_bytes(identity))

    authentication = stage_b.authenticate_identity_source_freeze(
        round_tripped,
        current_source_hashes=source_hashes,
    )

    assert stage_b.SOURCE_FILES is stage_b.identity_resolver.STAGE_B_SOURCE_FILES
    assert authentication["passed"] is True
    assert authentication["path_count"] == len(stage_b.SOURCE_FILES)


def test_identity_source_freeze_rejects_current_byte_drift() -> None:
    source_hashes = {
        path: stage_b._sha256_bytes(path.encode("utf-8"))  # noqa: SLF001
        for path in stage_b.SOURCE_FILES
    }
    identity = {
        "source_files": {
            "paths": list(stage_b.SOURCE_FILES),
            "sha256_start": source_hashes,
            "sha256_end": source_hashes,
            "stable": True,
        }
    }
    current = dict(source_hashes)
    current[stage_b.SOURCE_FILES[0]] = "0" * 64

    with pytest.raises(ValueError, match="differ from the identity freeze"):
        stage_b.authenticate_identity_source_freeze(
            identity,
            current_source_hashes=current,
        )


def test_runtime_environment_must_equal_the_committed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = {
        "schema": "recurquant.experiment009-stage-b-runtime-environment.v1",
        "runtime_matches_stage_a": True,
        "local_files_only": True,
    }
    calls: list[tuple[object, bool]] = []

    def fake_authenticate(
        stage_a: object,
        *,
        local_files_only: bool,
    ) -> dict[str, object]:
        calls.append((stage_a, local_files_only))
        return dict(authenticated)

    monkeypatch.setattr(
        stage_b.identity_resolver,
        "authenticate_runtime_environment",
        fake_authenticate,
    )
    stage_a = {"artifact_kind": stage_b.STAGE_A_ARTIFACT_KIND}
    identity = {"environment": dict(authenticated)}

    assert stage_b.authenticate_stage_b_runtime_environment(
        stage_a,
        identity,
        local_files_only=True,
    ) == authenticated
    assert calls == [(stage_a, True)]

    identity["environment"]["local_files_only"] = False
    with pytest.raises(ValueError, match="differs from the Stage-B identity"):
        stage_b.authenticate_stage_b_runtime_environment(
            stage_a,
            identity,
            local_files_only=True,
        )


def test_identity_row_plan_reconstructs_the_only_frozen_plan() -> None:
    identity, stage_a = _frozen_row_plan_inputs()

    plan, authentication = stage_b.plan_from_identity(identity, stage_a=stage_a)

    assert plan.promoted_group_count == stage_b.TARGET_PROMOTED_ROWS
    assert plan.resident_bytes == stage_b.TARGET_PACKED_STATE_BYTES
    assert {
        layer: len(plan.groups_for_layer(layer))
        for layer in stage_b.FROZEN_LINEAR_LAYERS
    } == stage_b.FROZEN_LAYER_QUOTAS
    assert authentication["passed"] is True


def test_identity_row_plan_rejects_canonical_hash_drift() -> None:
    identity, stage_a = _frozen_row_plan_inputs()
    identity["row_plan"]["canonical_plan_sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="canonical hash"):
        stage_b.plan_from_identity(identity, stage_a=stage_a)


def test_identity_row_plan_rejects_duplicate_promoted_row() -> None:
    identity, stage_a = _frozen_row_plan_inputs()
    row_plan = identity["row_plan"]  # type: ignore[assignment]
    rows = row_plan["high_precision_rows"]  # type: ignore[index]
    rows[1] = copy.deepcopy(rows[0])  # type: ignore[index]
    _rehash_row_plan(identity)

    with pytest.raises(ValueError, match="duplicated"):
        stage_b.plan_from_identity(identity, stage_a=stage_a)


def test_identity_row_plan_rejects_selector_binding_drift() -> None:
    identity, stage_a = _frozen_row_plan_inputs()
    row_plan = identity["row_plan"]  # type: ignore[assignment]
    selector_binding = row_plan["selector_binding"]  # type: ignore[index]
    selector_binding["selector_file_sha256"] = "0" * 64  # type: ignore[index]
    _rehash_row_plan(identity)

    with pytest.raises(ValueError, match="exact Stage-A selectors"):
        stage_b.plan_from_identity(identity, stage_a=stage_a)


def test_committed_stage_a_artifact_recomputes_as_a_passing_prerequisite() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    path = (
        repository_root
        / "evidence"
        / "experiment009-rht-cqer-stage-a-666-5be8d48.json"
    )

    evidence, authentication = stage_b.authenticate_stage_a_prerequisite(
        path,
        repository_root=repository_root,
    )

    assert evidence["stage_a_gate"]["passed"] is True
    assert authentication["gate_recomputed_and_passed"] == "true"


def test_independent_dense_rht_reference_passes() -> None:
    evidence = stage_b.compute_independent_dense_rht_evidence()

    assert evidence["passed"] is True
    assert evidence["signs_exact"] is True
    assert evidence["encode_max_abs_difference"] <= 2e-6
    assert evidence["physical_pack_max_abs_difference"] <= 6e-6


def test_imported_evaluator_modules_resolve_to_repository_sources() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    controlled_modules = {
        "stage_b": stage_b,
        "identity_resolver": stage_b.identity_resolver,
        "pilot": stage_b.pilot,
        "screen": stage_b.screen,
        "recurquant.cache": stage_b.importlib.import_module("recurquant.cache"),
        "recurquant.evaluation": stage_b.importlib.import_module(
            "recurquant.evaluation"
        ),
        "recurquant.packed_cache": stage_b.importlib.import_module(
            "recurquant.packed_cache"
        ),
        "recurquant.qwen35": stage_b.importlib.import_module("recurquant.qwen35"),
        "recurquant.rht": stage_b.importlib.import_module("recurquant.rht"),
    }

    paths = stage_b.authenticate_imported_module_paths(
        repository_root,
        modules=controlled_modules,
    )

    assert paths["stage_b_identity_resolver"] == (
        "scripts/resolve_rht_cqer_stage_b_identity.py"
    )
    assert paths["recurquant_qwen35"] == "src/recurquant/qwen35.py"


def test_stage_b_artifact_source_avoids_local_path_and_raw_command_fields() -> None:
    source = Path(stage_b.__file__).read_text(encoding="utf-8")

    assert '"path_supplied"' not in source
    assert '"selector_path"' not in source
    assert '"loss_selector_path"' not in source
    assert '"command": [sys.executable, *sys.argv]' not in source
    assert '"command_template"' in source
