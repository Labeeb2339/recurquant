from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from recurquant import experiment013_source, static_q468
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
FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT = b"finalized-stage-a-capture-receipt-fixture"
FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256 = resolver.sha256_bytes(
    FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT
)
FIXTURE_BINDING = {
    "calibration_authorization_file_sha256": resolver.sha256_bytes(b"calibration-authorization"),
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
    identity_bytes = resolver.canonical_json_bytes(
        {"evidence": {"source_manifest_sha256": _hash("identity-input-manifest")}}
    )
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

    authorization_record = {
        "fisher_boundary": {
            "boundary_positions": list(resolver.anchor_positions(15)),
        },
        "identity_record_sha256": _hash("authorization-identity-record"),
        "prompt_token_ids_sha256": _hash("authorization-prompt-tokens"),
        "sequence_length": 17,
        "sequence_token_ids_sha256": _hash("authorization-sequence-tokens"),
        "target_token_ids_sha256": _hash("authorization-target-tokens"),
    }
    identity = SimpleNamespace(
        file_sha256=identity_sha256,
        canonical_evidence_sha256=_hash("identity-canonical-evidence"),
        records=(authorization_record,),
        assignment=(),
        assignment_sha256=_hash("identity-assignment"),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS),
    )
    candidate_row = torch.arange(geometry.total_rows, dtype=torch.float64)
    candidate_d4 = ((candidate_row + 2) % 1_009) / 1_009
    candidate_d8 = ((candidate_row + 11) % 1_013) / 1_013
    candidate_scores = SimpleNamespace(
        artifact_kind=calibration.CALIBRATION_SCORE_ARTIFACT_KIND,
        file_sha256=resolver.sha256_bytes(score_bytes),
        calibration_identity_sha256=identity_sha256,
        calibration_scores_sha256=candidate_score_sha256,
        aggregate=SimpleNamespace(
            d4=candidate_d4,
            d8=candidate_d8,
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
        stability=SimpleNamespace(
            checks=(
                ("spearman_at_least_0_70", True),
                ("q8_jaccard_at_least_0_50", True),
                ("every_layer_mean_bitwidth_shift_at_most_0_25", True),
            ),
            layer_mean_bitwidth_shifts=((0, 0.125),),
            passed=True,
            q8_jaccard=0.75,
            spearman_average_ties=0.875,
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
        source_commit_h0=source_commit_h0,
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


def _authorization_source_manifest(source_commit: str) -> bytes:
    payload: dict[str, Any] = {
        "schema": experiment013_source.EXPERIMENT013_SOURCE_MANIFEST_SCHEMA,
        "profile": experiment013_source.EXPERIMENT013_SOURCE_MANIFEST_PROFILE,
        "object_format": "sha1",
        "source_commit": source_commit,
        "git_executable": {"sha256": _hash("git-executable"), "size_bytes": 100},
        "repository_binding": {
            "schema": experiment013_source.EXPERIMENT013_REPOSITORY_BINDING_SCHEMA,
            "worktree_layout": "primary",
            **{name: True for name in experiment013_source._TRUE_BINDING_FIELDS},
        },
        "paths": [
            {
                "git_blob_oid": _hash(f"git-blob-{path}")[:40],
                "index_blob_oid": _hash(f"git-blob-{path}")[:40],
                "mode": "100644",
                "path": path,
                "raw_sha256": (
                    _hash("capture-source")
                    if path == resolver.CALIBRATION_CAPTURE_SOURCE_PATH
                    else _hash(f"source-bytes-{path}")
                ),
                "worktree_blob_oid": _hash(f"git-blob-{path}")[:40],
            }
            for path in experiment013_source.EXPERIMENT013_SOURCE_PATHS
        ],
    }
    payload["canonical_manifest_sha256"] = (
        experiment013_source.canonical_experiment013_source_manifest_sha256(payload)
    )
    return experiment013_source.canonical_experiment013_source_manifest_bytes(payload)


def _authorization_runtime_manifest() -> tuple[bytes, list[dict[str, object]]]:
    module_to_distribution = dict(resolver.CALIBRATION_CAPTURE_CRITICAL_MODULE_DISTRIBUTIONS)
    root_name = "calibration-packages"
    import_path = "Lib/site-packages"
    distribution_modules = {
        distribution: module for module, distribution in module_to_distribution.items()
    }
    distribution_modules["torch"] = "torch"
    runtime_files = [
        {
            "path": f"{import_path}/{module}/__init__.py",
            "sha256": _hash(f"runtime-file-{module}"),
            "size_bytes": 100 + index,
        }
        for index, module in enumerate(sorted(distribution_modules.values()))
    ]
    runtime_files.sort(key=lambda item: str(item["path"]))
    distributions = []
    for distribution in sorted(distribution_modules):
        module = distribution_modules[distribution]
        distributions.append(
            {
                "files": [f"{import_path}/{module}/__init__.py"],
                "name": distribution,
                "package_root": root_name,
                "version": (
                    resolver.CALIBRATION_CANONICAL_TORCH_DISTRIBUTION_VERSION
                    if distribution == "torch"
                    else resolver.TRANSFORMERS_VERSION
                    if distribution == "transformers"
                    else "1.0.0"
                ),
            }
        )
    document = {
        "artifact_kind": resolver.CALIBRATION_RUNTIME_MANIFEST_KIND,
        "base_runtime_root": "base-runtime",
        "base_sys_path": ["."],
        "distributions": distributions,
        "git_executable": {
            "absolute_path_sha256": _hash("runtime-git-path"),
            "sha256": _hash("runtime-git"),
            "size_bytes": 200,
        },
        "interpreter": {
            "relative_path": "python.exe",
            "root": "base-runtime",
            "sha256": _hash("runtime-python"),
            "size_bytes": 300,
        },
        "launch_policy": dict(resolver.CALIBRATION_SEALED_LAUNCH_POLICY),
        "machine": {
            "architecture": "64bit",
            "byteorder": "little",
            "machine": "AMD64",
            "pointer_bits": 64,
            "system": "Windows",
        },
        "package_roots": [{"import_path": import_path, "name": root_name}],
        "python": {
            "abi_flags": "",
            "cache_tag": "cpython-313",
            "implementation": "CPython",
            "version": "3.13.7",
        },
        "runtime_trees": [
            {
                "files": [
                    {
                        "path": "python.exe",
                        "sha256": _hash("runtime-python"),
                        "size_bytes": 300,
                    }
                ],
                "kind": "base-runtime",
                "name": "base-runtime",
            },
            {"files": runtime_files, "kind": "packages", "name": root_name},
        ],
        "schema_version": resolver.CALIBRATION_RUNTIME_MANIFEST_SCHEMA_VERSION,
    }
    by_path = {str(item["path"]): item for item in runtime_files}
    by_distribution = {str(item["name"]): item for item in distributions}
    origins = []
    for module in sorted(module_to_distribution):
        distribution = module_to_distribution[module]
        path = f"{import_path}/{module}/__init__.py"
        file = by_path[path]
        dist = by_distribution[distribution]
        origins.append(
            {
                "distribution": distribution,
                "module": module,
                "package_root": root_name,
                "relative_path": path,
                "sha256": file["sha256"],
                "size_bytes": file["size_bytes"],
                "version": dist["version"],
            }
        )
    return resolver.canonical_json_bytes(document), origins


def _authorization_model_manifest() -> bytes:
    files = [
        {
            "git_blob_oid": "1" * 40,
            "lfs_sha256": None,
            "lfs_size_bytes": None,
            "name": "config.json",
            "sha256": None,
            "size_bytes": 100,
        },
        {
            "git_blob_oid": "2" * 40,
            "lfs_sha256": _hash("model-weight"),
            "lfs_size_bytes": 1_000,
            "name": "model-00001-of-00001.safetensors",
            "sha256": _hash("model-weight"),
            "size_bytes": 1_000,
        },
        {
            "git_blob_oid": "3" * 40,
            "lfs_sha256": None,
            "lfs_size_bytes": None,
            "name": "model.safetensors.index.json",
            "sha256": None,
            "size_bytes": 200,
        },
    ]
    tree_payload = [
        {
            "git_blob_oid": item["git_blob_oid"],
            "lfs_sha256": item["lfs_sha256"],
            "lfs_size_bytes": item["lfs_size_bytes"],
            "name": item["name"],
        }
        for item in files
    ]
    return resolver.canonical_json_bytes(
        {
            "artifact_kind": resolver.CALIBRATION_MODEL_FILE_MANIFEST_KIND,
            "files": files,
            "hub_tree_manifest_sha256": resolver.sha256_bytes(
                resolver.canonical_json_bytes(tree_payload)
            ),
            "metadata_derivation": resolver.CALIBRATION_MODEL_FILE_MANIFEST_DERIVATION,
            "model_id": static_q468.PRIMARY_MODEL_ID,
            "revision": static_q468.PRIMARY_MODEL_REVISION,
            "schema_version": resolver.CALIBRATION_MODEL_FILE_MANIFEST_SCHEMA_VERSION,
            "selection_profile": resolver.CALIBRATION_MODEL_FILE_SELECTION_PROFILE,
            "transformers_version": resolver.TRANSFORMERS_VERSION,
        }
    )


def _runner_report_bytes(
    *,
    identity: SimpleNamespace,
    identity_input_manifest_sha256: str,
    repository_source_manifest: bytes,
    runtime_manifest: bytes,
    model_manifest: bytes,
    artifacts: dict[str, str],
    capture_receipt_sha256: str,
    smoke_report_sha256: str | None,
    status: str,
) -> bytes:
    smoke = status == "fisher_h1_smoke_passed"
    source = resolver._deserialize_repository_source_manifest(repository_source_manifest)
    runtime = resolver._deserialize_calibration_runtime_manifest(runtime_manifest)
    model = resolver._deserialize_model_file_manifest(model_manifest)
    counts = resolver._calibration_count_receipt(identity.records, smoke=smoke)
    gpu = {
        "capability": [9, 0],
        "device_index": 0,
        "name": "Fixture GPU",
        "peak_allocated_bytes": 200 if smoke else 400,
        "peak_reserved_bytes": 300 if smoke else 500,
    }
    adapter = {
        "adapter_revision": resolver.CALIBRATION_CANONICAL_ADAPTER_REVISION,
        "capture_input_sha256": identity_input_manifest_sha256,
        "device": "cuda:0",
        "fisher_step_count": counts["expected_fisher_step_count"],
        "kernel_backend": resolver.CALIBRATION_CANONICAL_ADAPTER_KERNEL_BACKEND,
        "materialization_attempted": True,
        "materialized_sequence_count": len(identity.records),
        "model_dtype": resolver.CALIBRATION_CANONICAL_ADAPTER_MODEL_DTYPE,
        "model_id": model["model_id"],
        "model_loaded": True,
        "model_loading_diagnostic_counts": {
            name: 0 for name in sorted(resolver.CALIBRATION_CANONICAL_ADAPTER_LOADING_DIAGNOSTICS)
        },
        "model_revision": model["revision"],
        "query_shape": list(resolver.CALIBRATION_CANONICAL_ADAPTER_QUERY_SHAPE),
        "recurrent_layer_indices": list(
            resolver.CALIBRATION_CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES
        ),
        "state_shape": list(resolver.CALIBRATION_CANONICAL_ADAPTER_STATE_SHAPE),
        "token_sequence_manifest_sha256": resolver._frozen_token_sequence_manifest_sha256(
            identity.records
        ),
        "transformers_version": model["transformers_version"],
    }
    evidence = {
        "artifacts": artifacts,
        "calibration": counts,
        "identity": {
            "canonical_evidence_sha256": identity.canonical_evidence_sha256,
            "execution_bindings": dict(identity.execution_bindings),
            "file_sha256": identity.file_sha256,
            "identity_input_manifest_sha256": identity_input_manifest_sha256,
            "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
        },
        "model_files": {
            "file_count": model["file_count"],
            "hub_tree_manifest_sha256": model["hub_tree_manifest_sha256"],
            "manifest_file_sha256": model["file_sha256"],
            "model_id": model["model_id"],
            "revision": model["revision"],
            "transformers_version": model["transformers_version"],
        },
        "prerequisites": {
            "capture_provenance_receipt_file_sha256": capture_receipt_sha256,
            "fisher_h1_smoke_report_file_sha256": smoke_report_sha256,
        },
        "query_energy_ema": dict(resolver.CALIBRATION_QUERY_ENERGY_EMA),
        "repository": {
            "source_commit": source["source_commit"],
            "source_manifest_file_sha256": source["file_sha256"],
            "source_manifest_sha256": source["canonical_manifest_sha256"],
        },
        "runner_revision": resolver.CALIBRATION_RUNNER_REVISION,
        "runtime": {
            "adapter": adapter,
            "authenticated_distribution_count": runtime["distribution_count"],
            "authenticated_file_count": runtime["file_count"],
            "cuda_available": True,
            "cuda_runtime": resolver.CALIBRATION_CANONICAL_CUDA_RUNTIME_VERSION,
            "elapsed_seconds_hex": (1.0 if smoke else 2.0).hex(),
            "gpu": gpu,
            "packages": runtime["packages"],
            "platform": "Windows-11-fixture",
            "python": runtime["python_version"],
            "runtime_manifest_file_sha256": runtime["file_sha256"],
            "torch": resolver.CALIBRATION_CANONICAL_TORCH_RUNTIME_VERSION,
        },
        "stability": (
            {"checks": [], "evaluated": False, "passed": None, "scope": "smoke_only"}
            if smoke
            else resolver._runner_stability_receipt(identity.split.stability)
        ),
        "status": status,
    }
    return resolver.canonical_json_bytes(
        {
            "artifact_kind": resolver.CALIBRATION_RUN_REPORT_KIND,
            "canonical_evidence_sha256": resolver.sha256_bytes(
                resolver.canonical_json_bytes(evidence)
            ),
            "evidence": evidence,
            "schema_version": resolver.CALIBRATION_RUN_REPORT_SCHEMA_VERSION,
        }
    )


@contextmanager
def _authorization_fixture(fixture: SimpleNamespace) -> Iterator[SimpleNamespace]:
    identity_input_manifest_sha256 = _hash("identity-input-manifest")
    source_manifest = _authorization_source_manifest(fixture.source_commit_h0)
    runtime_manifest, origins = _authorization_runtime_manifest()
    model_manifest = _authorization_model_manifest()
    fixture.identity.execution_bindings = {
        "calibration_runtime_manifest_file_sha256": resolver.sha256_bytes(runtime_manifest),
        "model_file_manifest_file_sha256": resolver.sha256_bytes(model_manifest),
        "parquet_materialization_manifest_file_sha256": (
            resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
        ),
        "repository_source_manifest_file_sha256": resolver.sha256_bytes(source_manifest),
    }
    fixture.identity.split = fixture.split
    core_bytes = resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)
    q48 = static_q468.build_static_rht_q48_policy(
        fixture.candidate_scores.aggregate.d4,
        fixture.candidate_scores.aggregate.d8,
        geometry=static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        promoted_rows=static_q468.FROZEN_STATIC_Q48_PROMOTIONS,
        calibration_manifest_sha256=(
            fixture.candidate_scores.aggregate.sequence_score_manifest_sha256
        ),
        identity_artifact_sha256=fixture.identity.file_sha256,
        tokenizer_manifest_sha256=fixture.identity.tokenizer_manifest_sha256,
        source_commit=fixture.source_commit_h0,
        method_id=static_q468.STATIC_Q48_COMPARATOR_METHOD,
    )
    q48_bytes = static_q468.serialize_static_rht_q48_policy(q48)
    receipt = resolver.canonical_json_bytes(
        {
            "artifact_kind": resolver.CALIBRATION_CAPTURE_PROVENANCE_KIND,
            "capture_source": {
                "path": "scripts/capture_static_q468_identity_input.py",
                "sha256": _hash("capture-source"),
            },
            "capture_version": resolver.CALIBRATION_CAPTURE_VERSION,
            "critical_module_origins": origins,
            "excluded_runtime_modules": list(resolver.CALIBRATION_CAPTURE_EXCLUDED_RUNTIME_MODULES),
            "execution_bindings": dict(fixture.identity.execution_bindings),
            "identity_input_file_sha256": identity_input_manifest_sha256,
            "phase": "calibration",
            "publication_contract": resolver.CALIBRATION_CAPTURE_PUBLICATION_CONTRACT,
            "runner_revision": resolver.CALIBRATION_RUNNER_REVISION,
            "schema_version": resolver.CALIBRATION_CAPTURE_PROVENANCE_SCHEMA_VERSION,
            "source_commit": fixture.source_commit_h0,
            "status": resolver.CALIBRATION_CAPTURE_PROVENANCE_STATUS,
        }
    )
    smoke = _runner_report_bytes(
        identity=fixture.identity,
        identity_input_manifest_sha256=identity_input_manifest_sha256,
        repository_source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        artifacts={},
        capture_receipt_sha256=resolver.sha256_bytes(receipt),
        smoke_report_sha256=None,
        status="fisher_h1_smoke_passed",
    )
    output_hashes = {
        filename: resolver.sha256_bytes(
            core_bytes
            if role == "calibration_core_binding_artifact"
            else q48_bytes
            if role == "static_q48_policy_artifact"
            else fixture.dependencies[role]
        )
        for role, filename in resolver.CALIBRATION_OUTPUT_FILENAMES.items()
    }
    full_report = _runner_report_bytes(
        identity=fixture.identity,
        identity_input_manifest_sha256=identity_input_manifest_sha256,
        repository_source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        artifacts=output_hashes,
        capture_receipt_sha256=resolver.sha256_bytes(receipt),
        smoke_report_sha256=resolver.sha256_bytes(smoke),
        status="passed",
    )
    kwargs = {
        "calibration_run_report": full_report,
        "calibration_complete_marker": resolver.CALIBRATION_COMPLETE_BYTES,
        "capture_provenance_receipt": receipt,
        "fisher_h1_smoke_report": smoke,
        "fisher_h1_smoke_complete_marker": resolver.FISHER_H1_SMOKE_COMPLETE_BYTES,
        "calibration_core_binding_artifact": core_bytes,
        "calibration_runtime_manifest": runtime_manifest,
        "model_file_manifest": model_manifest,
        "repository_source_manifest": source_manifest,
        "static_q48_policy_artifact": q48_bytes,
    }
    artifact = resolver.build_stage_a_calibration_authorization_artifact(**kwargs)
    yield SimpleNamespace(
        artifact=artifact,
        kwargs=kwargs,
        output_hashes=output_hashes,
        receipt=receipt,
        smoke=smoke,
        full_report=full_report,
        core=core_bytes,
        model_manifest=model_manifest,
        runtime_manifest=runtime_manifest,
        source_manifest=source_manifest,
    )


def _rechain_authorization_receipt(
    authorization: SimpleNamespace, receipt: bytes
) -> dict[str, bytes]:
    smoke = json.loads(authorization.smoke)
    smoke["evidence"]["prerequisites"]["capture_provenance_receipt_file_sha256"] = (
        resolver.sha256_bytes(receipt)
    )
    smoke_bytes = _reauthenticated_binding_bytes(smoke)
    full = json.loads(authorization.full_report)
    full["evidence"]["prerequisites"] = {
        "capture_provenance_receipt_file_sha256": resolver.sha256_bytes(receipt),
        "fisher_h1_smoke_report_file_sha256": resolver.sha256_bytes(smoke_bytes),
    }
    kwargs = dict(authorization.kwargs)
    kwargs.update(
        {
            "calibration_run_report": _reauthenticated_binding_bytes(full),
            "capture_provenance_receipt": receipt,
            "fisher_h1_smoke_report": smoke_bytes,
        }
    )
    return kwargs


def _finalized_stage_a_capture_receipt_fixture(
    fixture: SimpleNamespace,
    authorization: SimpleNamespace,
) -> SimpleNamespace:
    binding_bytes = resolver.build_stage_a_calibration_binding_artifact(
        calibration_authorization_artifact=authorization.artifact
    )
    binding = resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)
    runtime_manifest, origins = _authorization_runtime_manifest()
    assert runtime_manifest == authorization.runtime_manifest
    identity_input_file_sha256 = _hash("finalized-stage-a-identity-input")
    document = {
        "artifact_kind": resolver.STAGE_A_CAPTURE_PROVENANCE_KIND,
        "calibration_authorization_file_sha256": binding.authorization_file_sha256,
        "calibration_binding_file_sha256": resolver.sha256_bytes(binding_bytes),
        "capture_source": {
            "path": resolver.CALIBRATION_CAPTURE_SOURCE_PATH,
            "sha256": _hash("capture-source"),
        },
        "capture_version": resolver.CALIBRATION_CAPTURE_VERSION,
        "critical_module_origins": origins,
        "excluded_runtime_modules": list(resolver.CALIBRATION_CAPTURE_EXCLUDED_RUNTIME_MODULES),
        "execution_bindings": dict(binding.execution_bindings),
        "identity_input_file_sha256": identity_input_file_sha256,
        "phase": "stage_a",
        "publication_contract": resolver.STAGE_A_CAPTURE_PUBLICATION_CONTRACT,
        "runner_revision": resolver.CALIBRATION_RUNNER_REVISION,
        "schema_version": resolver.STAGE_A_CAPTURE_PROVENANCE_SCHEMA_VERSION,
        "source_commit": fixture.source_commit_h0,
        "status": resolver.STAGE_A_CAPTURE_PROVENANCE_STATUS,
    }
    receipt = resolver.canonical_json_bytes(document)
    return SimpleNamespace(
        binding=binding,
        binding_bytes=binding_bytes,
        document=document,
        identity_input_file_sha256=identity_input_file_sha256,
        receipt=receipt,
        receipt_sha256=resolver.sha256_bytes(receipt),
    )


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
    for rank, (category, config, configured_length, seed) in enumerate(
        resolver.RULER_STAGE_A_SCHEDULE
    ):
        records.append(
            _record(
                family="ruler",
                canonical_id=resolver.ruler_canonical_id(
                    category=category,
                    config=config,
                    configured_length=configured_length,
                    seed=seed,
                ),
                config=config,
                rank=rank,
                seed=seed,
                sequence_length=4_096,
                prefill_stop=4_092,
                scored_stop=4_096,
                configured_length=configured_length,
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
    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    verified_capture = SimpleNamespace(
        file_sha256=FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
        identity_input_file_sha256=resolver.sha256_bytes(resolver.canonical_json_bytes(source)),
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
    ):
        return resolver.build_candidate(
            source,
            expected_revisions=REVISIONS,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resolver.canonical_json_bytes(value))


def _fixture_verified_stage_a_capture(source: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        file_sha256=FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
        identity_input_file_sha256=resolver.sha256_bytes(resolver.canonical_json_bytes(source)),
    )


def _fixture_stage_a_capture_cli_args(receipt_path: Path) -> list[str]:
    receipt_path.write_bytes(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT)
    return [
        "--stage-a-capture-provenance-receipt",
        str(receipt_path),
        "--expected-stage-a-capture-provenance-receipt-sha256",
        FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
    ]


def _validate_stage_a_input_before_receipt(
    source: Mapping[str, Any],
    *,
    verified_binding: SimpleNamespace | None = None,
) -> None:
    binding = verified_binding or SimpleNamespace(
        binding=dict(FIXTURE_BINDING),
        execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS),
    )
    expected_binding_hash = resolver.sha256_bytes(FIXTURE_BINDING_ARTIFACT)

    def deserialize_binding(data: bytes, **kwargs: object) -> SimpleNamespace:
        assert data == FIXTURE_BINDING_ARTIFACT
        assert kwargs == {"expected_file_sha256": expected_binding_hash}
        return binding

    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            side_effect=deserialize_binding,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            side_effect=AssertionError("pre-finalization validation read a finalized receipt"),
        ),
    ):
        resolver.validate_stage_a_identity_input_for_capture(
            source,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            expected_calibration_binding_file_sha256=expected_binding_hash,
        )


def test_stage_a_pre_finalization_validator_accepts_production_shaped_input_without_receipt() -> (
    None
):
    _validate_stage_a_input_before_receipt(_stage_a_source())


def test_build_candidate_still_requires_finalized_receipt_after_input_validation() -> None:
    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING),
        execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS),
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        pytest.raises(ValueError, match="requires a finalized capture provenance receipt"),
    ):
        resolver.build_candidate(
            _stage_a_source(),
            expected_revisions=REVISIONS,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source.__setitem__("datasets", []),
            "datasets must contain exactly four contracts",
        ),
        (
            lambda source: source.__setitem__("tokenizer", {}),
            "tokenizer fields drifted",
        ),
        (
            lambda source: source.__setitem__("records", []),
            "Stage A must contain exactly four",
        ),
        (
            lambda source: source.__setitem__("unexpected", "field"),
            "identity input fields drifted",
        ),
        (
            lambda source: source["records"][0].pop("target_token_ids_sha256"),
            r"records\[0\] fields drifted",
        ),
        (
            lambda source: source["records"][0].__setitem__("selection_sha256", "0" * 64),
            "selection SHA-256 drifted",
        ),
    ],
)
def test_stage_a_pre_finalization_validator_rejects_incomplete_or_forged_input(
    mutation: Any,
    message: str,
) -> None:
    source = _stage_a_source()
    mutation(source)

    with pytest.raises(ValueError, match=message):
        _validate_stage_a_input_before_receipt(source)


def test_stage_a_pre_finalization_validator_rejects_cross_chain_bindings() -> None:
    wrong_execution = dict(FIXTURE_EXECUTION_BINDINGS)
    wrong_execution["model_file_manifest_file_sha256"] = _hash("wrong-model-chain")
    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING),
        execution_bindings=wrong_execution,
    )

    with pytest.raises(ValueError, match="differ from calibration authorization"):
        _validate_stage_a_input_before_receipt(
            _stage_a_source(),
            verified_binding=verified,
        )


def test_stage_a_pre_finalization_validator_rejects_calibration_binding_drift() -> None:
    source = _stage_a_source()
    source["calibration_binding"]["static_k29334_policy_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="differs from the verified artifact"):
        _validate_stage_a_input_before_receipt(source)


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
    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    verified_capture = SimpleNamespace(
        file_sha256=FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
        identity_input_file_sha256=candidate["evidence"]["source_manifest_sha256"],
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
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
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
        pytest.raises(ValueError, match="frozen Stage-A identity contract drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(frozen),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
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


def test_stage_a_candidate_rejects_cross_chain_execution_bindings() -> None:
    source = _stage_a_source()
    wrong_execution = dict(FIXTURE_EXECUTION_BINDINGS)
    wrong_execution["model_file_manifest_file_sha256"] = _hash("wrong-model-chain")
    verified = SimpleNamespace(binding=dict(FIXTURE_BINDING), execution_bindings=wrong_execution)
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=SimpleNamespace(
                file_sha256=FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        ),
        pytest.raises(ValueError, match="differ from calibration authorization"),
    ):
        resolver.build_candidate(
            source,
            expected_revisions=REVISIONS,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        )


def test_stage_a_core_binding_v3_round_trips_exact_eight_embedded_dependencies() -> None:
    with _binding_v3_fixture() as fixture:
        artifact = resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)
        decoded = resolver.deserialize_stage_a_calibration_core_binding_artifact(artifact)

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
    assert resolver.STAGE_A_CORE_BINDING_ARTIFACT_SCHEMA_VERSION == 3
    assert resolver.STAGE_A_CORE_BINDING_ARTIFACT_REVISION.endswith("-v3")
    assert document["schema_version"] == 3
    assert document["evidence"]["artifact_revision"].endswith("-v3")
    assert set(document["evidence"]["dependencies_base64"]) == expected_dependencies
    assert set(document["evidence"]["dependency_file_sha256"]) == expected_dependencies
    assert set(decoded.binding) == resolver.CALIBRATION_BINDING_FIELDS - {
        "calibration_authorization_file_sha256"
    }
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


def test_stage_a_core_binding_rejects_v2_missing_extra_and_dependency_tamper() -> None:
    with _binding_v3_fixture() as fixture:
        artifact = resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)

        legacy = json.loads(artifact)
        legacy["schema_version"] = 2
        legacy["evidence"]["artifact_revision"] = "experiment-013-stage-a-calibration-binding-v2"
        with pytest.raises(ValueError, match="kind or schema drifted"):
            resolver.deserialize_stage_a_calibration_core_binding_artifact(
                _reauthenticated_binding_bytes(legacy)
            )

        missing = json.loads(artifact)
        del missing["evidence"]["dependencies_base64"]["comparator_score_artifact"]
        with pytest.raises(ValueError, match="dependencies fields drifted"):
            resolver.deserialize_stage_a_calibration_core_binding_artifact(
                _reauthenticated_binding_bytes(missing)
            )

        extra = json.loads(artifact)
        extra["evidence"]["dependencies_base64"]["ninth_dependency"] = resolver._canonical_b64(
            b"forbidden", context="fixture"
        )
        with pytest.raises(ValueError, match="dependencies fields drifted"):
            resolver.deserialize_stage_a_calibration_core_binding_artifact(
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
            resolver.deserialize_stage_a_calibration_core_binding_artifact(
                _reauthenticated_binding_bytes(tampered)
            )


def test_post_calibration_authorization_and_v4_binding_round_trip() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        verified_authorization = resolver.deserialize_stage_a_calibration_authorization_artifact(
            authorization.artifact
        )
        binding_bytes = resolver.build_stage_a_calibration_binding_artifact(
            calibration_authorization_artifact=authorization.artifact
        )
        verified_binding = resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)

    document = json.loads(binding_bytes)
    assert resolver.STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION == 4
    assert document["schema_version"] == 4
    assert document["evidence"]["artifact_revision"].endswith("-v4")
    assert set(document["evidence"]["dependencies_base64"]) == {
        "calibration_authorization_artifact"
    }
    assert verified_authorization.authorized_output_file_sha256 == authorization.output_hashes
    assert verified_authorization.source_commit == fixture.source_commit_h0
    assert set(verified_binding.binding) == resolver.CALIBRATION_BINDING_FIELDS
    assert verified_binding.binding["calibration_authorization_file_sha256"] == (
        resolver.sha256_bytes(authorization.artifact)
    )
    assert dict(verified_binding.calibration_dependencies) == fixture.dependencies


def test_finalized_stage_a_capture_provenance_round_trips_exact_flat_contract() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        capture = _finalized_stage_a_capture_receipt_fixture(fixture, authorization)
        verified = resolver.deserialize_stage_a_capture_provenance_receipt(
            capture.receipt,
            expected_file_sha256=capture.receipt_sha256,
            calibration_binding_artifact=capture.binding_bytes,
            expected_identity_input_file_sha256=capture.identity_input_file_sha256,
        )

    assert set(capture.document) == {
        "artifact_kind",
        "calibration_authorization_file_sha256",
        "calibration_binding_file_sha256",
        "capture_source",
        "capture_version",
        "critical_module_origins",
        "excluded_runtime_modules",
        "execution_bindings",
        "identity_input_file_sha256",
        "phase",
        "publication_contract",
        "runner_revision",
        "schema_version",
        "source_commit",
        "status",
    }
    assert capture.document["capture_version"] == 6
    assert capture.document["runner_revision"].endswith("-v12")
    assert verified.file_sha256 == capture.receipt_sha256
    assert verified.identity_input_file_sha256 == capture.identity_input_file_sha256
    assert verified.calibration_binding_file_sha256 == resolver.sha256_bytes(capture.binding_bytes)
    assert verified.calibration_authorization_file_sha256 == (
        capture.binding.authorization_file_sha256
    )
    assert dict(verified.execution_bindings) == dict(capture.binding.execution_bindings)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_field", "fields drifted"),
        ("extra_field", "fields drifted"),
        ("status", "finalized identity drifted"),
        ("publication", "finalized identity drifted"),
        ("source_commit", "differs from authorized H0"),
        ("identity_input", "binds a different identity input"),
        ("binding", "binds a different calibration binding"),
        ("authorization", "binds a different embedded authorization"),
        ("execution", "execution bindings drifted"),
        ("capture_path", "source path drifted"),
        ("capture_sha", "source differs from the repository manifest"),
        ("origins", "critical module inventory is not exact"),
        ("excluded", "excluded module inventory drifted"),
    ],
)
def test_finalized_stage_a_capture_provenance_rejects_rehashed_forgery(
    mutation: str,
    message: str,
) -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        capture = _finalized_stage_a_capture_receipt_fixture(fixture, authorization)
        document = copy.deepcopy(capture.document)
        if mutation == "missing_field":
            document.pop("status")
        elif mutation == "extra_field":
            document["unexpected"] = False
        elif mutation == "status":
            document["status"] = "captured_but_not_finalized"
        elif mutation == "publication":
            document["publication_contract"] = "overwrite-permitted"
        elif mutation == "source_commit":
            document["source_commit"] = "0" * 40
        elif mutation == "identity_input":
            document["identity_input_file_sha256"] = "0" * 64
        elif mutation == "binding":
            document["calibration_binding_file_sha256"] = "0" * 64
        elif mutation == "authorization":
            document["calibration_authorization_file_sha256"] = "0" * 64
        elif mutation == "execution":
            document["execution_bindings"]["model_file_manifest_file_sha256"] = "0" * 64
        elif mutation == "capture_path":
            document["capture_source"]["path"] = "scripts/forged_capture.py"
        elif mutation == "capture_sha":
            document["capture_source"]["sha256"] = "0" * 64
        elif mutation == "origins":
            document["critical_module_origins"] = document["critical_module_origins"][:-1]
        elif mutation == "excluded":
            document["excluded_runtime_modules"] = []
        else:  # pragma: no cover - parameter completeness guard
            raise AssertionError(mutation)
        receipt = resolver.canonical_json_bytes(document)
        with pytest.raises(ValueError, match=message):
            resolver.deserialize_stage_a_capture_provenance_receipt(
                receipt,
                expected_file_sha256=resolver.sha256_bytes(receipt),
                calibration_binding_artifact=capture.binding_bytes,
                expected_identity_input_file_sha256=capture.identity_input_file_sha256,
            )


def test_stage_a_capture_receipt_explicit_sha_precedes_binding_and_json_access() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        capture = _finalized_stage_a_capture_receipt_fixture(fixture, authorization)
        with (
            patch.object(
                resolver,
                "deserialize_stage_a_calibration_binding_artifact",
                side_effect=AssertionError("binding must not be touched after receipt SHA failure"),
            ),
            pytest.raises(ValueError, match="differs from its explicit SHA-256"),
        ):
            resolver.deserialize_stage_a_capture_provenance_receipt(
                b"not-json-and-wrong-hash",
                expected_file_sha256="0" * 64,
                calibration_binding_artifact=capture.binding_bytes,
            )


def test_stage_a_capture_receipt_rejects_noncanonical_bytes() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        capture = _finalized_stage_a_capture_receipt_fixture(fixture, authorization)
        receipt = json.dumps(capture.document, indent=2).encode("utf-8")
        with pytest.raises(ValueError, match="not canonical JSON"):
            resolver.deserialize_stage_a_capture_provenance_receipt(
                receipt,
                expected_file_sha256=resolver.sha256_bytes(receipt),
                calibration_binding_artifact=capture.binding_bytes,
                expected_identity_input_file_sha256=capture.identity_input_file_sha256,
            )


def test_post_calibration_authorization_rejects_chain_and_marker_tamper() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        wrong_marker = dict(authorization.kwargs)
        wrong_marker["calibration_complete_marker"] = b"not-complete\n"
        with pytest.raises(ValueError, match="completion marker drifted"):
            resolver.build_stage_a_calibration_authorization_artifact(**wrong_marker)

        wrong_report = json.loads(authorization.full_report)
        wrong_report["evidence"]["artifacts"]["calibration-scores.json"] = "0" * 64
        wrong_report["canonical_evidence_sha256"] = resolver.sha256_bytes(
            resolver.canonical_json_bytes(wrong_report["evidence"])
        )
        report_kwargs = dict(authorization.kwargs)
        report_kwargs["calibration_run_report"] = resolver.canonical_json_bytes(wrong_report)
        with pytest.raises(ValueError, match="artifact inventory drifted"):
            resolver.build_stage_a_calibration_authorization_artifact(**report_kwargs)

        wrong_receipt = json.loads(authorization.receipt)
        wrong_receipt["source_commit"] = "b" * 40
        receipt_kwargs = dict(authorization.kwargs)
        receipt_kwargs["capture_provenance_receipt"] = resolver.canonical_json_bytes(wrong_receipt)
        with pytest.raises(ValueError, match="prerequisite binding drifted"):
            resolver.build_stage_a_calibration_authorization_artifact(**receipt_kwargs)


def test_authorization_distinguishes_repository_digest_from_identity_input_hash() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        report = json.loads(authorization.full_report)["evidence"]
        assert (
            report["repository"]["source_manifest_sha256"]
            != (report["identity"]["identity_input_manifest_sha256"])
        )
        verified = resolver.deserialize_stage_a_calibration_authorization_artifact(
            authorization.artifact
        )
    assert verified.source_commit == fixture.source_commit_h0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["evidence"]["calibration"].update({"sequence_count": 999}),
            "full calibration counters drifted",
        ),
        (
            lambda report: report["evidence"].update({"query_energy_ema": {"forged": True}}),
            "full query-energy contract drifted",
        ),
        (
            lambda report: report["evidence"]["model_files"].update({"file_count": 999}),
            "full model receipt drifted",
        ),
        (
            lambda report: report["evidence"]["runtime"]["adapter"].update(
                {"query_shape": [1, 1, 1, 1]}
            ),
            "full runtime receipt adapter identity drifted",
        ),
        (
            lambda report: report["evidence"]["runtime"].update({"cuda_runtime": "forged"}),
            "full runtime receipt identity drifted",
        ),
    ],
)
def test_authorization_rejects_rehashed_impossible_runner_receipts(
    mutation: Any, message: str
) -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        report = json.loads(authorization.full_report)
        mutation(report)
        kwargs = dict(authorization.kwargs)
        kwargs["calibration_run_report"] = _reauthenticated_binding_bytes(report)
        with pytest.raises(ValueError, match=message):
            resolver.build_stage_a_calibration_authorization_artifact(**kwargs)


def test_authorization_rejects_full_smoke_runtime_identity_parity_drift() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        smoke = json.loads(authorization.smoke)
        smoke["evidence"]["runtime"]["platform"] = "forged-but-nonempty-platform"
        smoke_bytes = _reauthenticated_binding_bytes(smoke)
        full = json.loads(authorization.full_report)
        full["evidence"]["prerequisites"]["fisher_h1_smoke_report_file_sha256"] = (
            resolver.sha256_bytes(smoke_bytes)
        )
        kwargs = dict(authorization.kwargs)
        kwargs["fisher_h1_smoke_report"] = smoke_bytes
        kwargs["calibration_run_report"] = _reauthenticated_binding_bytes(full)
        with pytest.raises(ValueError, match="full and smoke runtime identity receipts differ"):
            resolver.build_stage_a_calibration_authorization_artifact(**kwargs)


@pytest.mark.parametrize("mutation", ["capture_source", "origins", "excluded"])
def test_authorization_rejects_rehashed_capture_provenance_forgery(mutation: str) -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        receipt = json.loads(authorization.receipt)
        if mutation == "capture_source":
            receipt["capture_source"]["sha256"] = _hash("forged-capture-source")
            message = "source differs from the repository manifest"
        elif mutation == "origins":
            receipt["critical_module_origins"] = [{}]
            message = "critical module inventory is not exact"
        else:
            receipt["excluded_runtime_modules"] = []
            message = "excluded module inventory drifted"
        kwargs = _rechain_authorization_receipt(
            authorization, resolver.canonical_json_bytes(receipt)
        )
        with pytest.raises(ValueError, match=message):
            resolver.build_stage_a_calibration_authorization_artifact(**kwargs)


def test_authorization_rejects_q48_h0_and_allocation_drift() -> None:
    with _binding_v3_fixture() as fixture, _authorization_fixture(fixture) as authorization:
        common = {
            "geometry": static_q468.FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
            "promoted_rows": static_q468.FROZEN_STATIC_Q48_PROMOTIONS,
            "calibration_manifest_sha256": (
                fixture.candidate_scores.aggregate.sequence_score_manifest_sha256
            ),
            "identity_artifact_sha256": fixture.identity.file_sha256,
            "tokenizer_manifest_sha256": fixture.identity.tokenizer_manifest_sha256,
            "method_id": static_q468.STATIC_Q48_COMPARATOR_METHOD,
        }
        wrong_h0 = static_q468.build_static_rht_q48_policy(
            fixture.candidate_scores.aggregate.d4,
            fixture.candidate_scores.aggregate.d8,
            source_commit="b" * 40,
            **common,
        )
        kwargs = dict(authorization.kwargs)
        kwargs["static_q48_policy_artifact"] = static_q468.serialize_static_rht_q48_policy(wrong_h0)
        with pytest.raises(ValueError, match="Q48 policy differs"):
            resolver.build_stage_a_calibration_authorization_artifact(**kwargs)

        wrong_allocation = static_q468.build_static_rht_q48_policy(
            fixture.candidate_scores.aggregate.d4.flip(0),
            fixture.candidate_scores.aggregate.d8,
            source_commit=fixture.source_commit_h0,
            **common,
        )
        kwargs["static_q48_policy_artifact"] = static_q468.serialize_static_rht_q48_policy(
            wrong_allocation
        )
        with pytest.raises(ValueError, match="deterministic P14739 reconstruction"):
            resolver.build_stage_a_calibration_authorization_artifact(**kwargs)


def test_final_stage_a_binding_rejects_core_binding_without_authorization() -> None:
    with _binding_v3_fixture() as fixture:
        core = resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)
    with pytest.raises(ValueError, match="kind or schema drifted"):
        resolver.deserialize_stage_a_calibration_binding_artifact(core)


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
            resolver.build_stage_a_calibration_core_binding_artifact(**dependencies)


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
            resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)


def test_stage_a_binding_rejects_incomplete_comparator_identity_manifest() -> None:
    with _binding_v3_fixture() as fixture:
        selector = fixture.comparator_scores.selectors[
            calibration.FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE
        ]
        selector.aggregate.identity_record_manifest_sha256 = _hash(
            "incomplete-comparator-identity-record-manifest"
        )
        with pytest.raises(ValueError, match="complete frozen identity"):
            resolver.build_stage_a_calibration_core_binding_artifact(**fixture.dependencies)


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
    source = _stage_a_source()
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)
    receipt_path = tmp_path / "stage-a-capture-provenance.json"

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=_fixture_verified_stage_a_capture(source),
        ),
    ):
        result = resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(source_path),
                "--calibration-binding",
                str(binding_path),
                *_fixture_stage_a_capture_cli_args(receipt_path),
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
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "binding.json",
        "source.json",
        "stage-a-capture-provenance.json",
    ]


def test_candidate_requires_quarantine_then_exact_hash_promotion(tmp_path: Path) -> None:
    source = _stage_a_source()
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)
    receipt_path = tmp_path / "stage-a-capture-provenance.json"
    candidate_path = tmp_path / ".quarantine" / "stage-a-candidate.json"
    base_args = [
        "--phase",
        "stage_a",
        "--input",
        str(source_path),
        "--calibration-binding",
        str(binding_path),
        *_fixture_stage_a_capture_cli_args(receipt_path),
        "--mbpp-revision",
        REVISIONS["mbpp"],
        "--pg19-revision",
        REVISIONS["pg19"],
        "--ruler-revision",
        REVISIONS["ruler"],
        "--humaneval-plus-revision",
        REVISIONS["humaneval_plus"],
    ]

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=_fixture_verified_stage_a_capture(source),
        ),
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
                    *_fixture_stage_a_capture_cli_args(receipt_path),
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

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    verified_capture = _fixture_verified_stage_a_capture(_stage_a_source())
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        )

    assert frozen["evidence"]["status"] == "frozen"


def test_frozen_stage_a_decoder_reauthenticates_promotion_records_and_binding() -> None:
    candidate = _build_candidate(_stage_a_source())
    candidate_hash = resolver.sha256_bytes(resolver.canonical_json_bytes(candidate))
    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    verified_capture = _fixture_verified_stage_a_capture(_stage_a_source())
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
    ):
        frozen = resolver.promote_candidate(
            candidate,
            candidate_file_sha256=candidate_hash,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        )
        frozen_bytes = resolver.canonical_json_bytes(frozen)
        decoded = resolver.deserialize_frozen_stage_a_identity_artifact(
            frozen_bytes,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
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
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
        pytest.raises(ValueError, match="identity record SHA-256 drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(tampered),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
        )
    wrong_binding = dict(FIXTURE_BINDING)
    wrong_binding["static_k29334_policy_file_sha256"] = "0" * 64
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=SimpleNamespace(
                binding=wrong_binding,
                execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS),
            ),
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
        pytest.raises(ValueError, match="differs from the verified calibration binding"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            frozen_bytes,
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
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
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=verified_capture,
        ),
        pytest.raises(ValueError, match="candidate file SHA-256 drifted"),
    ):
        resolver.deserialize_frozen_stage_a_identity_artifact(
            resolver.canonical_json_bytes(tampered),
            calibration_binding_artifact=FIXTURE_BINDING_ARTIFACT,
            stage_a_capture_provenance_receipt=(FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
            expected_stage_a_capture_provenance_receipt_sha256=(
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
            ),
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
    receipt_path = tmp_path / "stage-a-capture-provenance.json"

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=_fixture_verified_stage_a_capture(_stage_a_source()),
        ),
        pytest.raises(ValueError, match="not canonical resolver JSON"),
    ):
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
                *_fixture_stage_a_capture_cli_args(receipt_path),
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
    receipt_path = tmp_path / "stage-a-capture-provenance.json"

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            return_value=_fixture_verified_stage_a_capture(_stage_a_source()),
        ),
        pytest.raises(ValueError, match="does not match explicit promotion hash"),
    ):
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
                *_fixture_stage_a_capture_cli_args(receipt_path),
                "--expected-candidate-sha256",
                "0" * 64,
            ]
        )


def test_stage_a_cli_verifies_binding_before_reading_identity_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "protected-stage-a-input.json"
    input_path.write_bytes(b"must-not-be-read")
    binding_path = tmp_path / "invalid-binding.json"
    binding_path.write_bytes(b"not-a-binding")
    receipt_path = tmp_path / "capture-provenance.json"
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == input_path.resolve():
            raise AssertionError("Stage-A input was read before binding verification")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(ValueError, match="Stage-A calibration binding must be strict JSON"):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(input_path),
                "--calibration-binding",
                str(binding_path),
                "--stage-a-capture-provenance-receipt",
                str(receipt_path),
                "--expected-stage-a-capture-provenance-receipt-sha256",
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
                "--dry-run",
                "--mbpp-revision",
                resolver.MBPP_REVISION,
                "--pg19-revision",
                resolver.PG19_REVISION,
                "--ruler-revision",
                resolver.RULER_REVISION,
                "--humaneval-plus-revision",
                resolver.HUMANEVAL_PLUS_REVISION,
            ]
        )


def test_stage_a_promotion_cli_verifies_binding_before_reading_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = tmp_path / "protected-stage-a-candidate.json"
    candidate_path.write_bytes(b"must-not-be-read")
    binding_path = tmp_path / "invalid-binding.json"
    binding_path.write_bytes(b"not-a-binding")
    receipt_path = tmp_path / "capture-provenance.json"
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == candidate_path.resolve():
            raise AssertionError("Stage-A candidate was read before binding verification")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(ValueError, match="Stage-A calibration binding must be strict JSON"):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(candidate_path),
                "--output",
                str(tmp_path / "frozen-stage-a-identity.json"),
                "--promote",
                "--calibration-binding",
                str(binding_path),
                "--stage-a-capture-provenance-receipt",
                str(receipt_path),
                "--expected-stage-a-capture-provenance-receipt-sha256",
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
                "--expected-candidate-sha256",
                "0" * 64,
            ]
        )


def test_stage_a_cli_verifies_capture_receipt_before_reading_identity_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "protected-stage-a-input.json"
    input_path.write_bytes(b"must-not-be-read")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)
    receipt_path = tmp_path / "invalid-capture-provenance.json"
    receipt_path.write_bytes(b"not-a-receipt")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == input_path.resolve():
            raise AssertionError("Stage-A input was read before receipt verification")
        return original_read_bytes(path)

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            side_effect=ValueError("invalid finalized Stage-A capture provenance"),
        ),
        pytest.raises(ValueError, match="invalid finalized Stage-A capture provenance"),
    ):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(input_path),
                "--calibration-binding",
                str(binding_path),
                "--stage-a-capture-provenance-receipt",
                str(receipt_path),
                "--expected-stage-a-capture-provenance-receipt-sha256",
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
                "--dry-run",
                "--mbpp-revision",
                resolver.MBPP_REVISION,
                "--pg19-revision",
                resolver.PG19_REVISION,
                "--ruler-revision",
                resolver.RULER_REVISION,
                "--humaneval-plus-revision",
                resolver.HUMANEVAL_PLUS_REVISION,
            ]
        )


def test_stage_a_promotion_cli_verifies_capture_receipt_before_reading_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = tmp_path / "protected-stage-a-candidate.json"
    candidate_path.write_bytes(b"must-not-be-read")
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(FIXTURE_BINDING_ARTIFACT)
    receipt_path = tmp_path / "invalid-capture-provenance.json"
    receipt_path.write_bytes(b"not-a-receipt")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == candidate_path.resolve():
            raise AssertionError("Stage-A candidate was read before receipt verification")
        return original_read_bytes(path)

    verified = SimpleNamespace(
        binding=dict(FIXTURE_BINDING), execution_bindings=dict(FIXTURE_EXECUTION_BINDINGS)
    )
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with (
        patch.object(
            resolver,
            "deserialize_stage_a_calibration_binding_artifact",
            return_value=verified,
        ),
        patch.object(
            resolver,
            "deserialize_stage_a_capture_provenance_receipt",
            side_effect=ValueError("invalid finalized Stage-A capture provenance"),
        ),
        pytest.raises(ValueError, match="invalid finalized Stage-A capture provenance"),
    ):
        resolver.main(
            [
                "--phase",
                "stage_a",
                "--input",
                str(candidate_path),
                "--output",
                str(tmp_path / "frozen-stage-a-identity.json"),
                "--promote",
                "--calibration-binding",
                str(binding_path),
                "--stage-a-capture-provenance-receipt",
                str(receipt_path),
                "--expected-stage-a-capture-provenance-receipt-sha256",
                FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
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
