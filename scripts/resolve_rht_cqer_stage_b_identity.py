#!/usr/bin/env python3
"""Resolve the frozen Experiment 009 Stage-B identity without model weights.

The only content retained, canonicalized, or tokenized by this program belongs
to ranked MBPP calibration window ``[32, 64)``.  The first streaming pass reads
only ``task_id`` so the frozen ranking can be reconstructed.  A second
task-ID-only streaming pass retains exactly the 32 authorized rows.

This module deliberately does not load model weights, run a forward pass, or
compute a quality metric.  Its authenticated output is an identity amendment,
not experimental evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Final

from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.public_data import (
    MBPP_CALIBRATION_SIZE,
    MBPP_CONFIG,
    MBPP_DATASET_ID,
    MBPP_FORMATTER_VERSION,
    MBPP_MANIFEST_SCHEMA,
    MBPP_REVISION,
    MBPP_SELECTION_NAMESPACE,
    canonical_mbpp_row,
    format_mbpp_example,
    mbpp_calibration_key,
    mbpp_manifest,
    mbpp_manifest_content_sha256,
    mbpp_row_sha256,
    mbpp_source_split,
)

ARTIFACT_KIND: Final = "recurquant_rht_cqer32_stage_b_identity"
IDENTITY_SCHEMA: Final = "recurquant.experiment009-stage-b-identity.v1"
TOKEN_MANIFEST_SCHEMA: Final = "recurquant.experiment009-stage-b-token-manifest.v1"
ORDERED_IDENTITY_SCHEMA: Final = "recurquant.experiment009-stage-b-ordered-identity.v1"
CLAIM_BOUNDARY: Final = (
    "This artifact fixes the Experiment 009 Stage-B data and tokenizer "
    "identity before model weights or quality metrics are opened. It is "
    "not performance, novelty, state-of-the-art, or breakthrough evidence."
)

STAGE_B_OFFSET: Final = 32
STAGE_B_LIMIT: Final = 32
STAGE_B_STOP: Final = STAGE_B_OFFSET + STAGE_B_LIMIT
PROTECTED_WINDOW: Final = (8, 16)

MODEL_ID: Final = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION: Final = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
TOKENIZER_CLASS: Final = "Qwen2Tokenizer"
TRANSFORMERS_VERSION: Final = "5.14.1"
TOKEN_ID_HASH_SERIALIZATION: Final = "recurquant.canonical-json.v1"
TEXT_HASH_ENCODING: Final = "utf-8"
RUNTIME_ENVIRONMENT_SCHEMA: Final = (
    "recurquant.experiment009-stage-b-runtime-environment.v1"
)
STAGE_A_PYTHON_VERSION: Final = (3, 11, 15)
STAGE_A_PACKAGE_VERSIONS: Final = {
    "datasets": "4.8.5",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "torch": "2.11.0+cu128",
    "transformers": TRANSFORMERS_VERSION,
}
STAGE_A_CUDA_CONTRACT: Final = {
    "available": True,
    "runtime_version": "12.8",
    "device_type": "cuda",
}

STAGE_A_ARTIFACT_RELATIVE_PATH: Final = (
    "evidence/experiment009-rht-cqer-stage-a-666-5be8d48.json"
)
STAGE_A_ARTIFACT_KIND: Final = "recurquant_rht_cqer32_stage_a_screen"
STAGE_A_FILE_SHA256: Final = (
    "98a432843dc438f2d5fde34f8704f154ebc3ee12c93ba7c469369acfedfb15b5"
)
STAGE_A_CANONICAL_EVIDENCE_SHA256: Final = (
    "9e03a1e8cefb5801406a47a2e5e365686afb0a05e10e099a989cee616b505ed1"
)
STAGE_A_IMPLEMENTATION_COMMIT: Final = (
    "5be8d48369d94081e55aa389c25f63c303c7b0dd"
)
STAGE_A_RESULT_COMMIT: Final = "1cbc20f6c493e79f771d047fc63e96a7464eacf4"
ROW_PLAN_SCHEMA: Final = "recurquant.experiment009-stage-b-row-plan.v1"
ROW_PLAN_METHOD: Final = "target_directional_fisher_difference_int4"
SELECTOR_ARTIFACT_RELATIVE_PATH: Final = (
    "artifacts/experiment006-hrr-selector-8task-c2ad68b.json"
)
LOSS_SELECTOR_ARTIFACT_RELATIVE_PATH: Final = (
    "artifacts/experiment006-loss-selector-8task-c2ad68b.json"
)
SELECTOR_ARTIFACT_KIND: Final = "recurquant_hrr_calibration_diagnostic"
LOSS_SELECTOR_ARTIFACT_KIND: Final = (
    "recurquant_loss_sensitivity_calibration_diagnostic"
)
SELECTOR_FILE_SHA256: Final = (
    "d0c4267095ee3f5068627b189a1fd9f58cb02f6e25672d9b89dd0990e5b09330"
)
SELECTOR_CANONICAL_EVIDENCE_SHA256: Final = (
    "7970961fd88b522998189ad64f26b333aed9c88ff5f653de5449fd9e01d8cbc8"
)
LOSS_SELECTOR_FILE_SHA256: Final = (
    "95c16656edb32efbc985f2fea59e229634dd558f4f4bf04819b8efc37783a1d6"
)
LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256: Final = (
    "bff4e33253990b8115e1f35e74516c4975c2fe4aac5066475afe968eb8a64609"
)
FROZEN_LINEAR_LAYERS: Final = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
)
FROZEN_LAYER_QUOTAS: Final = {
    0: 355,
    1: 380,
    2: 269,
    4: 179,
    5: 185,
    6: 105,
    8: 80,
    9: 43,
    10: 84,
    12: 30,
    13: 62,
    14: 54,
    16: 45,
    17: 27,
    18: 7,
    20: 9,
    21: 7,
    22: 55,
}

STAGE_B_SOURCE_FILES: Final = (
    "research/EXPERIMENT_009_RHT_CQER_PROTOCOL.md",
    "research/EXPERIMENT_009_DATA_ACCESS_CLARIFICATION.md",
    "research/EXPERIMENT_009_STAGE_A_AUDIT.md",
    "research/EXPERIMENT_009_STAGE_A_RESULT.md",
    "scripts/evaluate_rht_cqer_stage_b.py",
    "scripts/resolve_rht_cqer_stage_b_identity.py",
    "scripts/screen_rht_cqer.py",
    "scripts/pilot_evaluate_hrr.py",
    "src/recurquant/__init__.py",
    "src/recurquant/cache.py",
    "src/recurquant/evaluation.py",
    "src/recurquant/evidence.py",
    "src/recurquant/finite_difference.py",
    "src/recurquant/fisher_sensitivity.py",
    "src/recurquant/horizon.py",
    "src/recurquant/horizon_calibration.py",
    "src/recurquant/intervention.py",
    "src/recurquant/metrics.py",
    "src/recurquant/mixed_quantization.py",
    "src/recurquant/model_fisher.py",
    "src/recurquant/multibit_policy.py",
    "src/recurquant/multibit_quantization.py",
    "src/recurquant/packed_cache.py",
    "src/recurquant/public_data.py",
    "src/recurquant/quantization.py",
    "src/recurquant/query_energy.py",
    "src/recurquant/qwen35.py",
    "src/recurquant/rht.py",
    "src/recurquant/row_policy.py",
    "src/recurquant/transition_observer.py",
    "tests/test_evaluate_rht_cqer_stage_b.py",
    "tests/test_resolve_rht_cqer_stage_b_identity.py",
    "tests/test_rht.py",
    "tests/test_rht_independent_reference.py",
    "tests/test_rht_mixed_quantization.py",
    "tests/test_right_rht_query_ema_cache.py",
    "tests/test_screen_rht_cqer.py",
)
SOURCE_FILES: Final = STAGE_B_SOURCE_FILES
IMPORTED_MODULE_PATHS: Final = {
    "recurquant": "src/recurquant/__init__.py",
    "recurquant.evidence": "src/recurquant/evidence.py",
    "recurquant.public_data": "src/recurquant/public_data.py",
}

LoadDataset = Callable[..., Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class EncodedIdentityTask:
    """One authenticated Stage-B row and its exact tokenizer outputs."""

    row: Mapping[str, Any]
    prompt_token_ids: tuple[int, ...]
    code_token_ids: tuple[int, ...]
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StageBDataAccessAudit:
    """Application-level access counters separated from transport behavior."""

    ranking_transport_records_yielded: int
    ranking_task_id_fields_inspected: int
    target_transport_records_yielded: int
    target_task_id_fields_inspected: int
    target_rows_retained_and_canonicalized: int

    def as_dict(self, *, selected_task_ids: Sequence[int]) -> dict[str, Any]:
        selected = [int(task_id) for task_id in selected_task_ids]
        return {
            "transport_limitation": (
                "The Hugging Face streaming transport may deserialize complete "
                "source records before yielding mappings. These counters describe "
                "fields read and rows retained by RecurQuant application code."
            ),
            "ranking_pass": {
                "transport_records_yielded": self.ranking_transport_records_yielded,
                "task_id_fields_inspected": self.ranking_task_id_fields_inspected,
                "non_task_id_fields_read_by_recurquant": 0,
                "row_mappings_retained": 0,
            },
            "target_load_pass": {
                "transport_records_yielded": self.target_transport_records_yielded,
                "task_id_fields_inspected": self.target_task_id_fields_inspected,
                "non_target_content_fields_read_by_recurquant": 0,
                "target_rows_retained_and_canonicalized": (
                    self.target_rows_retained_and_canonicalized
                ),
            },
            "application_task_id_sets": {
                "selected": selected,
                "retained": selected,
                "canonicalized": selected,
                "formatted": selected,
                "tokenized": selected,
                "passed_to_model": [],
                "evaluated": [],
            },
            "protected_window_intersection": {
                "selected": False,
                "retained": False,
                "canonicalized": False,
                "formatted": False,
                "tokenized": False,
                "passed_to_model": False,
                "evaluated": False,
            },
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedStageBRows:
    """Exact Stage-B rows plus application-level stream-access evidence."""

    rows: tuple[Mapping[str, Any], ...]
    ordered_task_ids: tuple[int, ...]
    access_audit: StageBDataAccessAudit


@dataclass(frozen=True, slots=True)
class RuntimeIdentityAuthentication:
    """Tokenized runtime identity and the hashes authenticated before weights."""

    tasks: tuple[EncodedIdentityTask, ...]
    ordered_task_ids: tuple[int, ...]
    content_manifest_sha256: str
    token_manifest_sha256: str
    ordered_identity_sha256: str
    access_audit: StageBDataAccessAudit | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve only the frozen Experiment 009 ranked MBPP [32, 64) "
            "development identity. No model weights or quality metrics are opened."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
        and value == value.lower()
    )


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _require_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def _python_triplet_from_stage_a(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError("Stage-A Python identity must be a string")
    version_token = value.split(maxsplit=1)[0]
    components = version_token.split(".")
    if (
        len(components) != 3
        or any(not component.isdigit() for component in components)
    ):
        raise ValueError("Stage-A Python identity lacks an exact major.minor.micro")
    triplet = tuple(int(component) for component in components)
    if ".".join(str(component) for component in triplet) != version_token:
        raise ValueError("Stage-A Python identity is not canonical")
    return triplet[0], triplet[1], triplet[2]


def derive_stage_a_runtime_environment(
    stage_a_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the frozen Stage-B runtime from authenticated Stage-A evidence."""

    environment = _require_mapping(
        stage_a_evidence.get("environment"),
        context="Stage-A environment",
    )
    if set(environment) != {
        "cuda_available",
        "cuda_runtime",
        "gpu",
        "packages",
        "platform",
        "python",
    }:
        raise ValueError("Stage-A environment fields drifted")
    packages = _require_mapping(
        environment.get("packages"),
        context="Stage-A environment packages",
    )
    if set(packages) != set(STAGE_A_PACKAGE_VERSIONS) or any(
        not isinstance(value, str) or not value for value in packages.values()
    ):
        raise ValueError("Stage-A environment package identity drifted")
    model = _require_mapping(stage_a_evidence.get("model"), context="Stage-A model")
    python_version = _python_triplet_from_stage_a(environment.get("python"))
    cuda_available = environment.get("cuda_available")
    cuda_runtime = environment.get("cuda_runtime")
    device_type = model.get("device")
    if (
        not isinstance(cuda_available, bool)
        or not isinstance(cuda_runtime, str)
        or not cuda_runtime
        or not isinstance(device_type, str)
        or not device_type
        or not isinstance(environment.get("gpu"), str)
        or not environment.get("gpu")
        or not isinstance(environment.get("platform"), str)
        or not environment.get("platform")
    ):
        raise ValueError("Stage-A accelerator environment identity is incomplete")
    return {
        "python": {
            "major": python_version[0],
            "minor": python_version[1],
            "micro": python_version[2],
            "version": ".".join(str(component) for component in python_version),
        },
        "packages": dict(packages),
        "cuda": {
            "available": cuda_available,
            "runtime_version": cuda_runtime,
            "device_type": device_type,
        },
    }


def _frozen_stage_a_runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "major": STAGE_A_PYTHON_VERSION[0],
            "minor": STAGE_A_PYTHON_VERSION[1],
            "micro": STAGE_A_PYTHON_VERSION[2],
            "version": ".".join(
                str(component) for component in STAGE_A_PYTHON_VERSION
            ),
        },
        "packages": dict(STAGE_A_PACKAGE_VERSIONS),
        "cuda": dict(STAGE_A_CUDA_CONTRACT),
    }


def inspect_runtime_environment() -> dict[str, Any]:
    """Inspect only version and accelerator metadata; do not open model weights."""

    packages: dict[str, str] = {}
    for name in STAGE_A_PACKAGE_VERSIONS:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"required runtime package is missing: {name}") from error

    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_runtime = torch.version.cuda
    if cuda_runtime is not None and not isinstance(cuda_runtime, str):
        raise ValueError("runtime CUDA version must be a string or null")
    version = sys.version_info
    return {
        "python": {
            "major": version.major,
            "minor": version.minor,
            "micro": version.micro,
            "version": f"{version.major}.{version.minor}.{version.micro}",
        },
        "packages": packages,
        "cuda": {
            "available": cuda_available,
            "runtime_version": cuda_runtime,
            "device_type": "cuda" if cuda_available else "cpu",
        },
    }


def authenticate_runtime_environment(
    stage_a_evidence: Mapping[str, Any],
    *,
    local_files_only: bool,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require an exact Stage-A runtime match before Stage-B dataset access."""

    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only must be boolean")
    derived = derive_stage_a_runtime_environment(stage_a_evidence)
    frozen = _frozen_stage_a_runtime_environment()
    if derived != frozen:
        raise ValueError("authenticated Stage-A runtime contract drifted")
    observed = (
        dict(runtime_environment)
        if runtime_environment is not None
        else inspect_runtime_environment()
    )
    if observed != derived:
        mismatches = [
            field
            for field in ("python", "packages", "cuda")
            if observed.get(field) != derived[field]
        ]
        extra = sorted(set(observed) - set(derived))
        if extra:
            mismatches.append("unexpected_fields")
        raise ValueError(
            "Stage-B runtime does not exactly match authenticated Stage A: "
            + ", ".join(mismatches)
        )
    return {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        "stage_a_binding": {
            "artifact_kind": STAGE_A_ARTIFACT_KIND,
            "file_sha256": STAGE_A_FILE_SHA256,
            "canonical_evidence_sha256": STAGE_A_CANONICAL_EVIDENCE_SHA256,
        },
        **derived,
        "runtime_matches_stage_a": True,
        "local_files_only": local_files_only,
    }


def _load_exact_evidence_artifact(
    path: Path,
    *,
    expected_kind: str,
    expected_file_sha256: str,
    expected_canonical_evidence_sha256: str,
) -> dict[str, Any]:
    verification = verify_evidence_artifact(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_canonical_evidence_sha256=expected_canonical_evidence_sha256,
    )
    if verification["valid"] is not True:
        raise ValueError(
            f"{expected_kind} artifact failed authentication: "
            + "; ".join(verification["errors"])
        )
    document = json.loads(path.read_bytes().decode("utf-8"))
    evidence = _require_mapping(document.get("evidence"), context=expected_kind)
    if evidence.get("artifact_kind") != expected_kind:
        raise ValueError(f"{expected_kind} artifact kind drifted")
    model = _require_mapping(evidence.get("model"), context=f"{expected_kind} model")
    if (
        model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("dtype") != "torch.bfloat16"
    ):
        raise ValueError(f"{expected_kind} model contract drifted")
    return dict(evidence)


def _normalize_high_precision_rows(
    value: object,
    *,
    context: str,
) -> tuple[dict[str, int], ...]:
    rows = _require_list(value, context=context)
    normalized: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, raw in enumerate(rows):
        record = _require_mapping(raw, context=f"{context} row {index}")
        if set(record) != {"layer_index", "head_index", "row_index"}:
            raise ValueError(f"{context} row {index} fields drifted")
        layer = _strict_int(
            record.get("layer_index"),
            context=f"{context} row {index} layer",
        )
        head = _strict_int(
            record.get("head_index"),
            context=f"{context} row {index} head",
        )
        row = _strict_int(
            record.get("row_index"),
            context=f"{context} row {index} row",
        )
        identity = (layer, head, row)
        if (
            layer not in FROZEN_LAYER_QUOTAS
            or not 0 <= head < 16
            or not 0 <= row < 128
            or identity in seen
        ):
            raise ValueError(f"{context} row {index} identity is invalid or repeated")
        seen.add(identity)
        normalized.append(
            {
                "layer_index": layer,
                "head_index": head,
                "row_index": row,
            }
        )
    if tuple(
        (record["layer_index"], record["head_index"], record["row_index"])
        for record in normalized
    ) != tuple(
        sorted(
            (
                record["layer_index"],
                record["head_index"],
                record["row_index"],
            )
            for record in normalized
        )
    ):
        raise ValueError(f"{context} rows are not in canonical order")
    return tuple(normalized)


def validate_compact_row_plan(plan: Mapping[str, Any]) -> None:
    """Authenticate the complete public row plan embedded in the identity."""

    expected_fields = {
        "schema",
        "method",
        "selector_binding",
        "model",
        "quantization",
        "accounting",
        "score_shapes",
        "layer_quotas",
        "high_precision_rows",
        "canonical_plan_sha256",
    }
    if set(plan) != expected_fields:
        raise ValueError("Stage-B compact row-plan fields drifted")
    if plan.get("schema") != ROW_PLAN_SCHEMA or plan.get("method") != ROW_PLAN_METHOD:
        raise ValueError("Stage-B compact row-plan identity drifted")
    if dict(
        _require_mapping(plan.get("selector_binding"), context="selector binding")
    ) != {
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": (
            SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "loss_selector_canonical_evidence_sha256": (
            LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
    }:
        raise ValueError("Stage-B selector binding drifted")
    if dict(_require_mapping(plan.get("model"), context="row-plan model")) != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("Stage-B row-plan model drifted")
    if dict(
        _require_mapping(plan.get("quantization"), context="row-plan quantization")
    ) != {
        "low_bits": 4,
        "high_bits": 8,
        "group_size": 128,
        "scale_bits": 16,
    }:
        raise ValueError("Stage-B row-plan quantization drifted")
    if dict(_require_mapping(plan.get("accounting"), context="row-plan accounting")) != {
        "total_groups": 36_864,
        "mask_bytes": 4_608,
        "promotion_increment_bytes": 64,
        "target_resident_bytes": 2_564_096,
        "resident_bytes": 2_564_096,
        "promoted_group_count": 1_976,
    }:
        raise ValueError("Stage-B row-plan accounting drifted")
    expected_shapes = [
        {"layer_index": layer, "heads": 16, "rows": 128}
        for layer in FROZEN_LINEAR_LAYERS
    ]
    if plan.get("score_shapes") != expected_shapes:
        raise ValueError("Stage-B row-plan score shapes drifted")
    if dict(_require_mapping(plan.get("layer_quotas"), context="layer quotas")) != {
        str(layer): quota for layer, quota in FROZEN_LAYER_QUOTAS.items()
    }:
        raise ValueError("Stage-B row-plan layer quotas drifted")
    rows = _normalize_high_precision_rows(
        plan.get("high_precision_rows"),
        context="Stage-B row plan",
    )
    if len(rows) != 1_976:
        raise ValueError("Stage-B row plan must contain exactly 1,976 promotions")
    observed_quotas = {
        layer: sum(record["layer_index"] == layer for record in rows)
        for layer in FROZEN_LINEAR_LAYERS
    }
    if observed_quotas != FROZEN_LAYER_QUOTAS:
        raise ValueError("Stage-B row-plan promotion counts drifted")
    recorded_hash = plan.get("canonical_plan_sha256")
    if not _valid_sha256(recorded_hash):
        raise ValueError("Stage-B row-plan canonical hash is invalid")
    hash_payload = dict(plan)
    del hash_payload["canonical_plan_sha256"]
    if _sha256_bytes(canonical_json_bytes(hash_payload)) != recorded_hash:
        raise ValueError("Stage-B row-plan canonical hash does not match its contents")


def build_compact_row_plan(
    selector_path: Path,
    loss_selector_path: Path,
) -> dict[str, Any]:
    """Authenticate frozen selectors and extract their complete public row plan."""

    selector = _load_exact_evidence_artifact(
        selector_path,
        expected_kind=SELECTOR_ARTIFACT_KIND,
        expected_file_sha256=SELECTOR_FILE_SHA256,
        expected_canonical_evidence_sha256=SELECTOR_CANONICAL_EVIDENCE_SHA256,
    )
    loss_selector = _load_exact_evidence_artifact(
        loss_selector_path,
        expected_kind=LOSS_SELECTOR_ARTIFACT_KIND,
        expected_file_sha256=LOSS_SELECTOR_FILE_SHA256,
        expected_canonical_evidence_sha256=LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256,
    )
    selector_dataset = _require_mapping(
        selector.get("dataset"),
        context="selector dataset",
    )
    loss_dataset = _require_mapping(
        loss_selector.get("dataset"),
        context="loss-selector dataset",
    )
    if (
        selector_dataset.get("manifest") != loss_dataset.get("manifest")
        or selector_dataset.get("manifest_sha256")
        != loss_dataset.get("manifest_sha256")
    ):
        raise ValueError("frozen selector dataset identities differ")
    plan_record = _require_mapping(
        _require_mapping(loss_selector.get("plans"), context="loss-selector plans").get(
            ROW_PLAN_METHOD
        ),
        context="target-Fisher plan",
    )
    if set(plan_record) != {"evidence", "locations", "promotions_by_layer"}:
        raise ValueError("target-Fisher plan record fields drifted")
    plan_evidence = _require_mapping(
        plan_record.get("evidence"),
        context="target-Fisher plan evidence",
    )
    rows = _normalize_high_precision_rows(
        plan_record.get("locations"),
        context="target-Fisher locations",
    )
    evidence_rows = _normalize_high_precision_rows(
        plan_evidence.get("high_precision_rows"),
        context="target-Fisher evidence rows",
    )
    if rows != evidence_rows:
        raise ValueError("target-Fisher plan locations disagree with its evidence")
    expected_evidence_without_rows = {
        "low_bits": 4,
        "high_bits": 8,
        "group_size": 128,
        "scale_bits": 16,
        "total_groups": 36_864,
        "mask_bytes": 4_608,
        "promotion_increment_bytes": 64,
        "target_resident_bytes": 2_564_096,
        "resident_bytes": 2_564_096,
        "promoted_group_count": 1_976,
        "score_shapes": [
            [layer, 16, 128] for layer in FROZEN_LINEAR_LAYERS
        ],
    }
    if {
        key: plan_evidence.get(key) for key in expected_evidence_without_rows
    } != expected_evidence_without_rows:
        raise ValueError("target-Fisher recorded plan accounting drifted")
    observed_promotions = {
        str(layer): sum(record["layer_index"] == layer for record in rows)
        for layer in FROZEN_LINEAR_LAYERS
    }
    if observed_promotions != {
        str(layer): quota for layer, quota in FROZEN_LAYER_QUOTAS.items()
    }:
        raise ValueError("target-Fisher recorded layer quotas drifted")
    promotions_by_layer = _require_mapping(
        plan_record.get("promotions_by_layer"),
        context="target-Fisher promotions by layer",
    )
    if {
        str(key): _strict_int(value, context="promotions_by_layer value")
        for key, value in promotions_by_layer.items()
    } != observed_promotions:
        raise ValueError("target-Fisher promotion summary drifted")
    plan: dict[str, Any] = {
        "schema": ROW_PLAN_SCHEMA,
        "method": ROW_PLAN_METHOD,
        "selector_binding": {
            "selector_file_sha256": SELECTOR_FILE_SHA256,
            "selector_canonical_evidence_sha256": (
                SELECTOR_CANONICAL_EVIDENCE_SHA256
            ),
            "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
            "loss_selector_canonical_evidence_sha256": (
                LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
            ),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
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
            for layer in FROZEN_LINEAR_LAYERS
        ],
        "layer_quotas": {
            str(layer): quota for layer, quota in FROZEN_LAYER_QUOTAS.items()
        },
        "high_precision_rows": list(rows),
    }
    plan["canonical_plan_sha256"] = _sha256_bytes(canonical_json_bytes(plan))
    validate_compact_row_plan(plan)
    return plan


def _load_dataset_lazily(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stage-B identity resolution requires the optional evaluation "
            "dependencies; install recurquant[eval]"
        ) from error
    return load_dataset(*args, **kwargs)


def _resolve_ranked_id_windows(
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> tuple[tuple[int, ...], frozenset[int], int]:
    """Resolve target/protected ranks while reading only source task IDs."""

    loader = load_dataset_fn or _load_dataset_lazily
    loaded = loader(
        MBPP_DATASET_ID,
        MBPP_CONFIG,
        revision=MBPP_REVISION,
        split=mbpp_source_split("calibration"),
        streaming=True,
    )
    task_ids: list[int] = []
    seen: set[int] = set()
    for source_index, raw_row in enumerate(loaded):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"MBPP source row {source_index} must be a mapping")
        task_id = _strict_int(
            raw_row.get("task_id"),
            context=f"MBPP source row {source_index} task_id",
        )
        if task_id in seen:
            raise ValueError(f"duplicate MBPP task_id {task_id} in source split")
        seen.add(task_id)
        task_ids.append(task_id)

    if len(task_ids) < MBPP_CALIBRATION_SIZE:
        raise ValueError(
            "MBPP train split is too small for the frozen calibration population: "
            f"{len(task_ids)} < {MBPP_CALIBRATION_SIZE}"
        )
    ranked = sorted(task_ids, key=lambda task_id: (mbpp_calibration_key(task_id), task_id))
    selected = tuple(ranked[STAGE_B_OFFSET:STAGE_B_STOP])
    protected = frozenset(ranked[PROTECTED_WINDOW[0] : PROTECTED_WINDOW[1]])
    if len(selected) != STAGE_B_LIMIT or len(set(selected)) != STAGE_B_LIMIT:
        raise RuntimeError("Stage-B ranking did not resolve exactly 32 unique task IDs")
    if set(selected) & protected:
        raise RuntimeError("Stage-B target IDs overlap the protected ranked window")
    return selected, protected, len(task_ids)


def resolve_stage_b_task_ids(
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> tuple[tuple[int, ...], int]:
    """Resolve ranks 32-63 while reading only task IDs from the source stream."""

    selected, _protected, source_row_count = _resolve_ranked_id_windows(
        load_dataset_fn=load_dataset_fn,
    )
    return selected, source_row_count


def _load_exact_rows_with_audit(
    *,
    task_ids: Sequence[int],
    protected_task_ids: frozenset[int],
    load_dataset_fn: LoadDataset | None,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Canonicalize targets only; non-target mappings are read for task_id only."""

    ordered_ids = tuple(_strict_int(task_id, context="target task_id") for task_id in task_ids)
    if (
        len(ordered_ids) != STAGE_B_LIMIT
        or len(set(ordered_ids)) != STAGE_B_LIMIT
    ):
        raise ValueError("target task IDs must contain exactly 32 unique integers")
    if set(ordered_ids) & protected_task_ids:
        raise ValueError("target task IDs overlap the protected ranked window")

    loader = load_dataset_fn or _load_dataset_lazily
    loaded = loader(
        MBPP_DATASET_ID,
        MBPP_CONFIG,
        revision=MBPP_REVISION,
        split=mbpp_source_split("calibration"),
        streaming=True,
    )
    targets = set(ordered_ids)
    selected: dict[int, Mapping[str, Any]] = {}
    transport_rows = 0
    for source_index, raw_row in enumerate(loaded):
        transport_rows += 1
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"MBPP source row {source_index} must be a mapping")
        task_id = _strict_int(
            raw_row.get("task_id"),
            context=f"MBPP source row {source_index} task_id",
        )
        if task_id not in targets:
            continue
        if task_id in protected_task_ids:
            raise RuntimeError("protected ranked row reached target canonicalization")
        if task_id in selected:
            raise ValueError(f"duplicate requested MBPP task_id {task_id} in source split")
        selected[task_id] = canonical_mbpp_row(raw_row)
        if len(selected) == len(targets):
            break

    missing = [task_id for task_id in ordered_ids if task_id not in selected]
    if missing:
        rendered = ", ".join(str(task_id) for task_id in missing)
        raise ValueError(f"requested MBPP task IDs are missing from source split: {rendered}")
    return tuple(selected[task_id] for task_id in ordered_ids), transport_rows


def resolve_stage_b_rows(
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> AuthenticatedStageBRows:
    """Retain exactly the rows in the frozen ranked calibration window."""

    task_ids, protected_ids, source_row_count = _resolve_ranked_id_windows(
        load_dataset_fn=load_dataset_fn,
    )
    rows, target_transport_rows = _load_exact_rows_with_audit(
        task_ids=task_ids,
        protected_task_ids=protected_ids,
        load_dataset_fn=load_dataset_fn,
    )
    actual_ids = tuple(_strict_int(row.get("task_id"), context="selected task_id") for row in rows)
    if actual_ids != task_ids:
        raise RuntimeError("task-ID loader changed the frozen Stage-B order")
    return AuthenticatedStageBRows(
        rows=tuple(rows),
        ordered_task_ids=task_ids,
        access_audit=StageBDataAccessAudit(
            ranking_transport_records_yielded=source_row_count,
            ranking_task_id_fields_inspected=source_row_count,
            target_transport_records_yielded=target_transport_rows,
            target_task_id_fields_inspected=target_transport_rows,
            target_rows_retained_and_canonicalized=len(rows),
        ),
    )


def _normalize_token_ids(value: object, *, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence of token IDs")
    normalized = tuple(_strict_int(token, context=f"{context} token") for token in value)
    if not normalized:
        raise ValueError(f"{context} must not be empty")
    if any(token < 0 for token in normalized):
        raise ValueError(f"{context} must contain non-negative token IDs")
    return normalized


def _tokenize_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer output must contain input_ids")
    return _normalize_token_ids(encoded["input_ids"], context="tokenizer input_ids")


def tokenize_stage_b_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[EncodedIdentityTask, ...], tuple[dict[str, Any], ...]]:
    """Tokenize and fingerprint only the exact ordered Stage-B rows."""

    if len(rows) != STAGE_B_LIMIT:
        raise ValueError(f"Stage-B identity requires exactly {STAGE_B_LIMIT} rows")

    encoded_tasks: list[EncodedIdentityTask] = []
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        rank = STAGE_B_OFFSET + index
        task_id = _strict_int(row.get("task_id"), context=f"rank {rank} task_id")
        if task_id in seen:
            raise ValueError(f"Stage-B rows repeat task_id {task_id}")
        seen.add(task_id)

        formatted = format_mbpp_example(row)
        prompt_ids = _tokenize_text(
            tokenizer,
            formatted.prompt,
            add_special_tokens=True,
        )
        code_ids = _tokenize_text(
            tokenizer,
            formatted.code,
            add_special_tokens=False,
        )
        if len(code_ids) < 2:
            raise ValueError(f"MBPP task {task_id} has fewer than two code tokens")

        record = {
            "rank": rank,
            "task_id": task_id,
            "row_sha256": mbpp_row_sha256(row),
            "prompt_tokens": len(prompt_ids),
            "code_tokens": len(code_ids),
            "aligned_scored_tokens": len(code_ids) - 1,
            "full_code_scored_tokens": len(code_ids),
            "prompt_text_sha256": _sha256_bytes(formatted.prompt.encode("utf-8")),
            "code_text_sha256": _sha256_bytes(formatted.code.encode("utf-8")),
            "prompt_token_ids_sha256": _sha256_bytes(canonical_json_bytes(prompt_ids)),
            "code_token_ids_sha256": _sha256_bytes(canonical_json_bytes(code_ids)),
        }
        records.append(record)
        encoded_tasks.append(
            EncodedIdentityTask(
                row=row,
                prompt_token_ids=prompt_ids,
                code_token_ids=code_ids,
                record=record,
            )
        )
    return tuple(encoded_tasks), tuple(records)


def token_manifest_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact ordered task/token records under a named schema."""

    payload = {
        "schema": TOKEN_MANIFEST_SCHEMA,
        "tasks": list(records),
    }
    return _sha256_bytes(canonical_json_bytes(payload))


def ordered_identity_sha256(
    *,
    content_manifest_sha256: str,
    task_records: Sequence[Mapping[str, Any]],
) -> str:
    """Bind data, model, tokenizer, row order, and token identity together."""

    payload = {
        "schema": ORDERED_IDENTITY_SCHEMA,
        "dataset": {
            "id": MBPP_DATASET_ID,
            "config": MBPP_CONFIG,
            "revision": MBPP_REVISION,
            "phase": "calibration",
            "source_split": mbpp_source_split("calibration"),
            "selection_namespace": MBPP_SELECTION_NAMESPACE,
            "ranked_window": [STAGE_B_OFFSET, STAGE_B_STOP],
            "content_manifest_sha256": content_manifest_sha256,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "tokenizer": {
            "source_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": TOKENIZER_CLASS,
            "transformers_version": TRANSFORMERS_VERSION,
            "prompt_add_special_tokens": True,
            "code_add_special_tokens": False,
            "formatter_version": MBPP_FORMATTER_VERSION,
            "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
            "text_hash_encoding": TEXT_HASH_ENCODING,
        },
        "tasks": list(task_records),
    }
    return _sha256_bytes(canonical_json_bytes(payload))


def _validate_task_records(records: object) -> tuple[Mapping[str, Any], ...]:
    raw_records = _require_list(records, context="dataset tasks")
    if len(raw_records) != STAGE_B_LIMIT:
        raise ValueError(f"identity must contain exactly {STAGE_B_LIMIT} task records")

    normalized: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    hash_fields = (
        "row_sha256",
        "prompt_text_sha256",
        "code_text_sha256",
        "prompt_token_ids_sha256",
        "code_token_ids_sha256",
    )
    required_fields = {
        "rank",
        "task_id",
        "row_sha256",
        "prompt_tokens",
        "code_tokens",
        "aligned_scored_tokens",
        "full_code_scored_tokens",
        "prompt_text_sha256",
        "code_text_sha256",
        "prompt_token_ids_sha256",
        "code_token_ids_sha256",
    }
    for index, raw_record in enumerate(raw_records):
        record = _require_mapping(raw_record, context=f"dataset task {index}")
        if set(record) != required_fields:
            raise ValueError(f"dataset task {index} fields do not match the frozen schema")
        rank = _strict_int(record.get("rank"), context=f"dataset task {index} rank")
        task_id = _strict_int(record.get("task_id"), context=f"dataset task {index} task_id")
        prompt_tokens = _strict_int(
            record.get("prompt_tokens"),
            context=f"dataset task {index} prompt_tokens",
        )
        code_tokens = _strict_int(
            record.get("code_tokens"),
            context=f"dataset task {index} code_tokens",
        )
        aligned_tokens = _strict_int(
            record.get("aligned_scored_tokens"),
            context=f"dataset task {index} aligned_scored_tokens",
        )
        full_tokens = _strict_int(
            record.get("full_code_scored_tokens"),
            context=f"dataset task {index} full_code_scored_tokens",
        )
        if rank != STAGE_B_OFFSET + index:
            raise ValueError("Stage-B rank order drifted")
        if task_id in seen:
            raise ValueError(f"identity repeats task_id {task_id}")
        if prompt_tokens < 1 or code_tokens < 2:
            raise ValueError(f"task {task_id} has invalid token counts")
        if aligned_tokens != code_tokens - 1 or full_tokens != code_tokens:
            raise ValueError(f"task {task_id} scoring counts are misaligned")
        for field in hash_fields:
            if not _valid_sha256(record.get(field)):
                raise ValueError(f"task {task_id} {field} is not a canonical SHA-256")
        seen.add(task_id)
        normalized.append(record)
    return tuple(normalized)


def validate_stage_b_identity_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed on any internal or frozen-contract drift in an identity."""

    expected_evidence_fields = {
        "schema_version",
        "artifact_kind",
        "identity_schema",
        "identity_only",
        "claim_boundary",
        "created_at_utc",
        "authorization",
        "row_plan",
        "model_contract",
        "tokenizer_contract",
        "dataset",
        "integrity",
        "repository",
        "source_files",
        "environment",
    }
    if set(evidence) != expected_evidence_fields:
        raise ValueError("Stage-B identity top-level fields drifted")
    if evidence.get("schema_version") != 1:
        raise ValueError("Stage-B identity schema_version must be 1")
    if evidence.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("unexpected Stage-B identity artifact kind")
    if evidence.get("identity_schema") != IDENTITY_SCHEMA:
        raise ValueError("unexpected Stage-B identity schema")
    if evidence.get("identity_only") is not True:
        raise ValueError("Stage-B artifact must be identity-only")
    if evidence.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("Stage-B claim boundary drifted")
    created_at = evidence.get("created_at_utc")
    if not isinstance(created_at, str):
        raise ValueError("Stage-B creation time must be an ISO-8601 string")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError("Stage-B creation time is not valid ISO-8601") from error
    if parsed_created_at.tzinfo is None:
        raise ValueError("Stage-B creation time must include a timezone")

    authorization = _require_mapping(
        evidence.get("authorization"),
        context="authorization",
    )
    authorization_expected = {
        "stage_a_artifact_kind": STAGE_A_ARTIFACT_KIND,
        "stage_a_file_sha256": STAGE_A_FILE_SHA256,
        "stage_a_canonical_evidence_sha256": STAGE_A_CANONICAL_EVIDENCE_SHA256,
        "stage_a_implementation_commit": STAGE_A_IMPLEMENTATION_COMMIT,
        "stage_a_result_commit": STAGE_A_RESULT_COMMIT,
        "stage_a_gate_passed": True,
        "verified_before_dataset_access": True,
    }
    if dict(authorization) != authorization_expected:
        raise ValueError("Stage-A authorization contract drifted")

    row_plan = _require_mapping(evidence.get("row_plan"), context="row plan")
    validate_compact_row_plan(row_plan)

    model = _require_mapping(evidence.get("model_contract"), context="model contract")
    if dict(model) != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_loaded": False,
    }:
        raise ValueError("Stage-B model contract drifted")

    tokenizer = _require_mapping(
        evidence.get("tokenizer_contract"),
        context="tokenizer contract",
    )
    if dict(tokenizer) != {
        "source_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "class": TOKENIZER_CLASS,
        "transformers_version": TRANSFORMERS_VERSION,
        "trust_remote_code": False,
        "prompt_add_special_tokens": True,
        "code_add_special_tokens": False,
        "formatter_version": MBPP_FORMATTER_VERSION,
        "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
        "text_hash_encoding": TEXT_HASH_ENCODING,
    }:
        raise ValueError("Stage-B tokenizer contract drifted")

    dataset = _require_mapping(evidence.get("dataset"), context="dataset")
    expected_dataset_fields = {
        "id",
        "config",
        "revision",
        "phase",
        "source_split",
        "selection_namespace",
        "formatter_version",
        "selection_mode",
        "selection_window",
        "protected_window",
        "ordered_task_ids",
        "manifest",
        "content_manifest_sha256",
        "token_manifest_sha256",
        "ordered_identity_sha256",
        "tasks",
        "totals",
        "data_access",
    }
    if set(dataset) != expected_dataset_fields:
        raise ValueError("Stage-B dataset fields drifted")
    if dataset.get("phase") != "calibration":
        raise ValueError("Stage-B identity must use calibration phase")
    expected_dataset_contract = {
        "id": MBPP_DATASET_ID,
        "config": MBPP_CONFIG,
        "revision": MBPP_REVISION,
        "source_split": mbpp_source_split("calibration"),
        "selection_namespace": MBPP_SELECTION_NAMESPACE,
        "formatter_version": MBPP_FORMATTER_VERSION,
    }
    for field, expected in expected_dataset_contract.items():
        if dataset.get(field) != expected:
            raise ValueError(f"Stage-B dataset {field} drifted")
    if dataset.get("selection_mode") != "task_id_ranking_then_exact_task_id_stream":
        raise ValueError("Stage-B selection mode drifted")
    if dataset.get("selection_window") != {
        "offset": STAGE_B_OFFSET,
        "limit": STAGE_B_LIMIT,
        "stop_exclusive": STAGE_B_STOP,
    }:
        raise ValueError("Stage-B ranked window drifted")
    if dataset.get("protected_window") != {
        "offset": PROTECTED_WINDOW[0],
        "stop_exclusive": PROTECTED_WINDOW[1],
        "content_retained_canonicalized_or_tokenized": False,
    }:
        raise ValueError("protected-window contract drifted")

    records = _validate_task_records(dataset.get("tasks"))
    ordered_ids = [_strict_int(value, context="ordered task_id") for value in _require_list(
        dataset.get("ordered_task_ids"),
        context="ordered task IDs",
    )]
    record_ids = [int(record["task_id"]) for record in records]
    if ordered_ids != record_ids:
        raise ValueError("ordered task IDs do not match task records")

    manifest = _require_mapping(dataset.get("manifest"), context="content manifest")
    expected_manifest_fields = {
        "schema",
        "dataset_id",
        "config",
        "revision",
        "phase",
        "source_split",
        "selection_namespace",
        "formatter_version",
        "row_count",
        "rows",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("content manifest fields drifted")
    for field, expected in {
        "schema": MBPP_MANIFEST_SCHEMA,
        "dataset_id": MBPP_DATASET_ID,
        "config": MBPP_CONFIG,
        "revision": MBPP_REVISION,
        "phase": "calibration",
        "source_split": mbpp_source_split("calibration"),
        "selection_namespace": MBPP_SELECTION_NAMESPACE,
        "formatter_version": MBPP_FORMATTER_VERSION,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"content manifest {field} drifted")
    content_hash = dataset.get("content_manifest_sha256")
    if not _valid_sha256(content_hash):
        raise ValueError("content manifest SHA-256 is invalid")
    if mbpp_manifest_content_sha256(manifest) != content_hash:
        raise ValueError("content manifest hash does not match embedded content")
    if manifest.get("row_count") != STAGE_B_LIMIT:
        raise ValueError("content manifest must contain exactly 32 rows")
    manifest_rows = _require_list(manifest.get("rows"), context="content manifest rows")
    record_hashes = {
        int(record["task_id"]): str(record["row_sha256"]) for record in records
    }
    normalized_manifest_rows = tuple(
        _require_mapping(item, context="content manifest row") for item in manifest_rows
    )
    if any(set(record) != {"task_id", "sha256"} for record in normalized_manifest_rows):
        raise ValueError("content manifest row fields drifted")
    manifest_hashes = {
        _strict_int(record.get("task_id"), context="manifest task_id"): record.get("sha256")
        for record in normalized_manifest_rows
    }
    if manifest_hashes != record_hashes:
        raise ValueError("content manifest rows do not match ordered row identities")

    expected_token_hash = token_manifest_sha256(records)
    if dataset.get("token_manifest_sha256") != expected_token_hash:
        raise ValueError("token manifest hash drifted")
    expected_ordered_hash = ordered_identity_sha256(
        content_manifest_sha256=str(content_hash),
        task_records=records,
    )
    if dataset.get("ordered_identity_sha256") != expected_ordered_hash:
        raise ValueError("ordered identity hash drifted")

    totals = _require_mapping(dataset.get("totals"), context="dataset totals")
    expected_totals = {
        "source_train_rows_seen_by_task_id_only": _strict_int(
            totals.get("source_train_rows_seen_by_task_id_only"),
            context="source row count",
        ),
        "retained_rows": STAGE_B_LIMIT,
        "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records),
        "code_tokens": sum(int(record["code_tokens"]) for record in records),
        "aligned_scored_tokens": sum(
            int(record["aligned_scored_tokens"]) for record in records
        ),
        "full_code_scored_tokens": sum(
            int(record["full_code_scored_tokens"]) for record in records
        ),
    }
    if expected_totals["source_train_rows_seen_by_task_id_only"] < MBPP_CALIBRATION_SIZE:
        raise ValueError("source row count is too small for frozen MBPP ranking")
    if dict(totals) != expected_totals:
        raise ValueError("Stage-B aggregate token totals drifted")

    data_access = _require_mapping(dataset.get("data_access"), context="data access")
    limitation = data_access.get("transport_limitation")
    if (
        not isinstance(limitation, str)
        or "may deserialize complete source records" not in limitation
        or "RecurQuant application code" not in limitation
    ):
        raise ValueError("data-access transport limitation is missing")
    ranking_pass = _require_mapping(
        data_access.get("ranking_pass"),
        context="ranking-pass access",
    )
    ranking_yielded = _strict_int(
        ranking_pass.get("transport_records_yielded"),
        context="ranking transport records",
    )
    if dict(ranking_pass) != {
        "transport_records_yielded": ranking_yielded,
        "task_id_fields_inspected": ranking_yielded,
        "non_task_id_fields_read_by_recurquant": 0,
        "row_mappings_retained": 0,
    }:
        raise ValueError("ranking-pass access counters drifted")
    if ranking_yielded != expected_totals["source_train_rows_seen_by_task_id_only"]:
        raise ValueError("ranking-pass access count does not match dataset totals")
    target_pass = _require_mapping(
        data_access.get("target_load_pass"),
        context="target-load access",
    )
    target_yielded = _strict_int(
        target_pass.get("transport_records_yielded"),
        context="target-load transport records",
    )
    if target_yielded < STAGE_B_LIMIT or dict(target_pass) != {
        "transport_records_yielded": target_yielded,
        "task_id_fields_inspected": target_yielded,
        "non_target_content_fields_read_by_recurquant": 0,
        "target_rows_retained_and_canonicalized": STAGE_B_LIMIT,
    }:
        raise ValueError("target-load access counters drifted")
    application_sets = _require_mapping(
        data_access.get("application_task_id_sets"),
        context="application task-ID sets",
    )
    expected_application_sets = {
        "selected": ordered_ids,
        "retained": ordered_ids,
        "canonicalized": ordered_ids,
        "formatted": ordered_ids,
        "tokenized": ordered_ids,
        "passed_to_model": [],
        "evaluated": [],
    }
    if dict(application_sets) != expected_application_sets:
        raise ValueError("application task-ID access sets drifted")
    protected_intersection = _require_mapping(
        data_access.get("protected_window_intersection"),
        context="protected-window intersection",
    )
    if set(protected_intersection) != set(expected_application_sets) or any(
        value is not False for value in protected_intersection.values()
    ):
        raise ValueError("protected ranked window intersects an application access set")

    integrity = _require_mapping(evidence.get("integrity"), context="integrity")
    required_true = (
        "stage_a_authenticated_before_dataset_access",
        "runtime_environment_authenticated_before_dataset_access",
        "selector_artifacts_authenticated_before_dataset_access",
        "repository_clean_at_start",
        "repository_clean_at_end",
        "repository_commit_stable",
        "source_hashes_stable",
        "task_id_only_ranking_pass",
        "only_stage_b_content_retained_canonicalized_and_tokenized",
        "imported_modules_resolved_to_authenticated_repository",
        "output_path_external_or_git_ignored",
    )
    if any(integrity.get(field) is not True for field in required_true):
        raise ValueError("Stage-B integrity contract contains a failed check")
    required_false = (
        "protected_window_8_16_content_retained_canonicalized_or_tokenized",
        "model_weights_loaded",
        "model_forward_pass_run",
        "logits_or_quality_metrics_observed",
    )
    if any(integrity.get(field) is not False for field in required_false):
        raise ValueError("Stage-B identity observed forbidden model/data state")
    if set(integrity) != set(required_true) | set(required_false):
        raise ValueError("Stage-B integrity fields drifted")

    repository = _require_mapping(evidence.get("repository"), context="repository")
    if set(repository) != {"commit", "start", "end", "stable_commit"}:
        raise ValueError("repository fields drifted")
    start = _require_mapping(repository.get("start"), context="repository start")
    end = _require_mapping(repository.get("end"), context="repository end")
    if set(start) != {"commit", "worktree_clean", "status"} or set(end) != {
        "commit",
        "worktree_clean",
        "status",
    }:
        raise ValueError("repository snapshot fields drifted")
    commit = repository.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or commit != commit.lower()
        or not all(character in string.hexdigits for character in commit)
    ):
        raise ValueError("repository commit must be a full lowercase Git SHA")
    if (
        start.get("commit") != commit
        or end.get("commit") != commit
        or start.get("worktree_clean") is not True
        or end.get("worktree_clean") is not True
        or start.get("status") != []
        or end.get("status") != []
        or repository.get("stable_commit") is not True
    ):
        raise ValueError("repository identity was not clean and stable")

    sources = _require_mapping(evidence.get("source_files"), context="source files")
    if set(sources) != {
        "paths",
        "sha256_start",
        "sha256_end",
        "stable",
        "imported_modules",
    }:
        raise ValueError("source-file fields drifted")
    if sources.get("paths") != list(SOURCE_FILES) or sources.get("stable") is not True:
        raise ValueError("source-file path contract drifted")
    start_hashes = _require_mapping(sources.get("sha256_start"), context="source hashes start")
    end_hashes = _require_mapping(sources.get("sha256_end"), context="source hashes end")
    if dict(start_hashes) != dict(end_hashes) or set(start_hashes) != set(SOURCE_FILES):
        raise ValueError("source-file hashes were not complete and stable")
    if any(not _valid_sha256(value) for value in start_hashes.values()):
        raise ValueError("source-file map contains an invalid SHA-256")
    imported_modules = _require_mapping(
        sources.get("imported_modules"),
        context="imported modules",
    )
    if dict(imported_modules) != IMPORTED_MODULE_PATHS:
        raise ValueError("imported-module source paths drifted")

    environment = _require_mapping(evidence.get("environment"), context="environment")
    if not isinstance(environment.get("local_files_only"), bool):
        raise ValueError("local_files_only must be boolean")
    expected_environment = {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        "stage_a_binding": {
            "artifact_kind": STAGE_A_ARTIFACT_KIND,
            "file_sha256": STAGE_A_FILE_SHA256,
            "canonical_evidence_sha256": STAGE_A_CANONICAL_EVIDENCE_SHA256,
        },
        **_frozen_stage_a_runtime_environment(),
        "runtime_matches_stage_a": True,
        "local_files_only": environment["local_files_only"],
    }
    if dict(environment) != expected_environment:
        raise ValueError("Stage-B runtime environment contract drifted")


def load_stage_b_identity_artifact(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and authenticate an Experiment 009 Stage-B identity artifact."""

    artifact_path = Path(path)
    verification = verify_evidence_artifact(artifact_path)
    if verification["valid"] is not True:
        raise ValueError(
            "Stage-B identity artifact failed canonical verification: "
            + "; ".join(verification["errors"])
        )
    raw = artifact_path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage-B identity artifact must be strict UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or set(document) != {"canonical_evidence_sha256", "evidence"}
        or not isinstance(document.get("evidence"), dict)
    ):
        raise ValueError("Stage-B identity artifact must contain an evidence object")
    evidence = document["evidence"]
    validate_stage_b_identity_evidence(evidence)
    return evidence, _sha256_bytes(raw)


def load_authenticated_stage_b_rows(
    evidence: Mapping[str, Any],
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> AuthenticatedStageBRows:
    """Independently re-rank IDs, then load only authenticated Stage-B content."""

    validate_stage_b_identity_evidence(evidence)
    dataset = _require_mapping(evidence.get("dataset"), context="dataset")
    task_ids = tuple(
        _strict_int(value, context="ordered task_id")
        for value in _require_list(dataset.get("ordered_task_ids"), context="ordered task IDs")
    )
    expected_task_ids, protected_task_ids, source_row_count = (
        _resolve_ranked_id_windows(load_dataset_fn=load_dataset_fn)
    )
    if task_ids != expected_task_ids:
        raise ValueError(
            "identity task IDs are not the independently resolved ranked window [32, 64)"
        )
    rows, transport_rows = _load_exact_rows_with_audit(
        task_ids=task_ids,
        protected_task_ids=protected_task_ids,
        load_dataset_fn=load_dataset_fn,
    )
    actual_manifest = mbpp_manifest(rows, phase="calibration")
    if actual_manifest != dataset.get("manifest"):
        raise ValueError("runtime Stage-B row content does not match the identity manifest")
    if mbpp_manifest_content_sha256(actual_manifest) != dataset.get(
        "content_manifest_sha256"
    ):
        raise ValueError("runtime Stage-B content manifest hash drifted")
    actual_ids = tuple(int(row["task_id"]) for row in rows)
    if actual_ids != task_ids:
        raise ValueError("runtime Stage-B row order drifted")
    return AuthenticatedStageBRows(
        rows=tuple(rows),
        ordered_task_ids=task_ids,
        access_audit=StageBDataAccessAudit(
            ranking_transport_records_yielded=source_row_count,
            ranking_task_id_fields_inspected=source_row_count,
            target_transport_records_yielded=transport_rows,
            target_task_id_fields_inspected=transport_rows,
            target_rows_retained_and_canonicalized=len(rows),
        ),
    )


def authenticate_stage_b_runtime_identity(
    evidence: Mapping[str, Any],
    rows: AuthenticatedStageBRows | Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> RuntimeIdentityAuthentication:
    """Re-tokenize and authenticate the complete identity before model loading."""

    validate_stage_b_identity_evidence(evidence)
    if tokenizer.__class__.__name__ != TOKENIZER_CLASS:
        raise ValueError(
            f"tokenizer class drifted: {tokenizer.__class__.__name__} != {TOKENIZER_CLASS}"
        )
    try:
        actual_transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("transformers is required to authenticate token identity") from error
    if actual_transformers_version != TRANSFORMERS_VERSION:
        raise ValueError(
            "transformers version drifted: "
            f"{actual_transformers_version} != {TRANSFORMERS_VERSION}"
        )

    dataset = _require_mapping(evidence.get("dataset"), context="dataset")
    authenticated_rows = rows if isinstance(rows, AuthenticatedStageBRows) else None
    row_sequence = authenticated_rows.rows if authenticated_rows is not None else tuple(rows)
    actual_manifest = mbpp_manifest(row_sequence, phase="calibration")
    if actual_manifest != dataset.get("manifest"):
        raise ValueError("runtime Stage-B content manifest drifted before tokenization")
    encoded, records = tokenize_stage_b_rows(tokenizer, row_sequence)
    if list(records) != dataset.get("tasks"):
        raise ValueError("runtime Stage-B token records drifted")
    if token_manifest_sha256(records) != dataset.get("token_manifest_sha256"):
        raise ValueError("runtime Stage-B token manifest hash drifted")
    if ordered_identity_sha256(
        content_manifest_sha256=str(dataset["content_manifest_sha256"]),
        task_records=records,
    ) != dataset.get("ordered_identity_sha256"):
        raise ValueError("runtime Stage-B ordered identity hash drifted")
    return RuntimeIdentityAuthentication(
        tasks=encoded,
        ordered_task_ids=tuple(int(record["task_id"]) for record in records),
        content_manifest_sha256=str(dataset["content_manifest_sha256"]),
        token_manifest_sha256=str(dataset["token_manifest_sha256"]),
        ordered_identity_sha256=str(dataset["ordered_identity_sha256"]),
        access_audit=(
            authenticated_rows.access_audit if authenticated_rows is not None else None
        ),
    )


def authenticate_stage_a_artifact(path: str | Path) -> tuple[dict[str, Any], str]:
    """Require the exact committed Stage-A pass before any Stage-B data access."""

    artifact_path = Path(path)
    verification = verify_evidence_artifact(
        artifact_path,
        expected_file_sha256=STAGE_A_FILE_SHA256,
        expected_canonical_evidence_sha256=STAGE_A_CANONICAL_EVIDENCE_SHA256,
    )
    if verification["valid"] is not True:
        raise ValueError(
            "Stage-A authorization artifact failed authentication: "
            + "; ".join(verification["errors"])
        )
    raw = artifact_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    evidence = _require_mapping(document.get("evidence"), context="Stage-A evidence")
    if evidence.get("artifact_kind") != STAGE_A_ARTIFACT_KIND:
        raise ValueError("Stage-A artifact kind drifted")
    gate = _require_mapping(evidence.get("stage_a_gate"), context="Stage-A gate")
    checks = _require_mapping(gate.get("checks"), context="Stage-A gate checks")
    if (
        gate.get("schema") != "recurquant.experiment009-stage-a-gate.v1"
        or gate.get("passed") is not True
        or not checks
        or any(
            _require_mapping(check, context="Stage-A gate check").get("passed") is not True
            for check in checks.values()
        )
    ):
        raise ValueError("Stage-A gate did not pass every frozen check")
    model = _require_mapping(evidence.get("model"), context="Stage-A model")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("Stage-A model identity drifted")
    selectors = _require_mapping(
        evidence.get("selector_artifacts"),
        context="Stage-A selector artifacts",
    )
    expected_selector_hashes = {
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": (
            SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "loss_selector_canonical_evidence_sha256": (
            LOSS_SELECTOR_CANONICAL_EVIDENCE_SHA256
        ),
    }
    if any(
        selectors.get(field) != expected
        for field, expected in expected_selector_hashes.items()
    ):
        raise ValueError("Stage-A selector artifact hashes drifted")
    repository = _require_mapping(evidence.get("repository"), context="Stage-A repository")
    if (
        repository.get("commit") != STAGE_A_IMPLEMENTATION_COMMIT
        or repository.get("stable_commit") is not True
    ):
        raise ValueError("Stage-A implementation commit drifted")
    stage_a_dataset = _require_mapping(
        evidence.get("dataset"),
        context="Stage-A dataset",
    )
    identity_gate = _require_mapping(
        checks.get("frozen_task_identity_before_model_weights"),
        context="Stage-A identity gate",
    )
    if (
        stage_a_dataset.get("identity_authenticated_before_model_weights") is not True
        or stage_a_dataset.get(
            "protected_window_8_16_loaded_tokenized_or_evaluated"
        )
        is not False
        or identity_gate.get("identity_authenticated_before_model_weights") is not True
        or identity_gate.get("protected_window_8_16_accessed") is not False
    ):
        raise ValueError("Stage-A data-access integrity drifted")
    return dict(evidence), _sha256_bytes(raw)


def git_state(repository_root: Path) -> dict[str, Any]:
    """Return a strict full-SHA and porcelain worktree snapshot."""

    commit_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip()
    status = status_result.stdout.splitlines()
    if (
        len(commit) != 40
        or commit != commit.lower()
        or not all(character in string.hexdigits for character in commit)
    ):
        raise RuntimeError("Git did not return a full lowercase commit SHA")
    return {
        "commit": commit,
        "worktree_clean": not status,
        "status": status,
    }


def validate_repository_start(repository_root: Path, repository: Mapping[str, Any]) -> None:
    """Require clean committed code containing the authenticated Stage-A result."""

    if repository.get("worktree_clean") is not True or repository.get("status") != []:
        raise ValueError("Stage-B identity resolution requires a clean worktree")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", STAGE_A_RESULT_COMMIT, "HEAD"),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("current HEAD does not contain the authenticated Stage-A result")


def validate_output_path(output: Path, repository_root: Path) -> None:
    """Require external or Git-ignored output so publication cannot dirty the run."""

    resolved_output = output.resolve()
    resolved_root = repository_root.resolve()
    try:
        relative = resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ("git", "check-ignore", "--quiet", "--", str(relative)),
        cwd=resolved_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "Stage-B identity output inside the repository must be Git-ignored"
        )


def source_file_hashes(repository_root: Path) -> dict[str, str]:
    """Hash the exact implementation, protocol, and tests used by the resolver."""

    hashes: dict[str, str] = {}
    resolved_root = repository_root.resolve()
    for relative in SOURCE_FILES:
        path = (resolved_root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"source path escapes repository root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {relative}")
        hashes[relative] = _sha256_bytes(path.read_bytes())
    return hashes


def validate_imported_module_paths(
    repository_root: Path,
    *,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Bind imported RecurQuant modules to the source files being authenticated."""

    active_modules = sys.modules if modules is None else modules
    validate_all_imported_repository_modules_frozen(
        repository_root,
        modules=active_modules,
    )
    resolved_root = repository_root.resolve()
    authenticated: dict[str, str] = {}
    for module_name, relative_path in IMPORTED_MODULE_PATHS.items():
        module = active_modules.get(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise ValueError(f"imported module {module_name} has no source file")
        actual_path = Path(raw_path).resolve()
        expected_path = (resolved_root / relative_path).resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"imported module {module_name} did not resolve to authenticated "
                f"repository source {relative_path}"
            )
        authenticated[module_name] = relative_path
    return authenticated


def validate_all_imported_repository_modules_frozen(
    repository_root: Path,
    *,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Reject any imported project script or package module outside the source freeze."""

    resolved_root = repository_root.resolve()
    frozen = set(STAGE_B_SOURCE_FILES)
    imported: dict[str, str] = {}
    active_modules = sys.modules if modules is None else modules
    for module_name, module in sorted(active_modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        actual_path = Path(raw_path).resolve()
        try:
            relative_path = actual_path.relative_to(resolved_root).as_posix()
        except ValueError:
            continue
        if not relative_path.startswith(("scripts/", "src/recurquant/")):
            continue
        if not actual_path.is_file():
            raise ValueError(
                f"imported repository module {module_name} has no source file: "
                f"{relative_path}"
            )
        if relative_path not in frozen:
            raise ValueError(
                f"imported repository module {module_name} is outside the "
                f"Stage-B source freeze: {relative_path}"
            )
        imported[module_name] = relative_path
    return imported


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    repository_start = git_state(repository_root)
    validate_repository_start(repository_root, repository_start)
    validate_output_path(args.output, repository_root)
    source_hashes_start = source_file_hashes(repository_root)
    imported_module_paths = validate_imported_module_paths(repository_root)

    stage_a_path = repository_root / STAGE_A_ARTIFACT_RELATIVE_PATH
    stage_a_evidence, stage_a_file_sha = authenticate_stage_a_artifact(stage_a_path)
    runtime_environment = authenticate_runtime_environment(
        stage_a_evidence,
        local_files_only=bool(args.local_files_only),
    )
    row_plan = build_compact_row_plan(
        repository_root / SELECTOR_ARTIFACT_RELATIVE_PATH,
        repository_root / LOSS_SELECTOR_ARTIFACT_RELATIVE_PATH,
    )

    resolved_rows = resolve_stage_b_rows()
    rows = resolved_rows.rows
    task_ids = resolved_rows.ordered_task_ids
    source_row_count = resolved_rows.access_audit.ranking_transport_records_yielded
    content_manifest = mbpp_manifest(rows, phase="calibration")
    content_hash = mbpp_manifest_content_sha256(content_manifest)

    actual_transformers_version = importlib.metadata.version("transformers")
    if actual_transformers_version != TRANSFORMERS_VERSION:
        raise ValueError(
            "transformers version drifted before tokenization: "
            f"{actual_transformers_version} != {TRANSFORMERS_VERSION}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.__class__.__name__ != TOKENIZER_CLASS:
        raise ValueError(
            f"tokenizer class drifted: {tokenizer.__class__.__name__} != {TOKENIZER_CLASS}"
        )
    _encoded_tasks, task_records = tokenize_stage_b_rows(tokenizer, rows)
    token_hash = token_manifest_sha256(task_records)
    ordered_hash = ordered_identity_sha256(
        content_manifest_sha256=content_hash,
        task_records=task_records,
    )

    repository_end = git_state(repository_root)
    source_hashes_end = source_file_hashes(repository_root)
    if repository_end != repository_start:
        raise RuntimeError("repository state changed during Stage-B identity resolution")
    if source_hashes_end != source_hashes_start:
        raise RuntimeError("source files changed during Stage-B identity resolution")

    totals = {
        "source_train_rows_seen_by_task_id_only": source_row_count,
        "retained_rows": len(rows),
        "prompt_tokens": sum(int(record["prompt_tokens"]) for record in task_records),
        "code_tokens": sum(int(record["code_tokens"]) for record in task_records),
        "aligned_scored_tokens": sum(
            int(record["aligned_scored_tokens"]) for record in task_records
        ),
        "full_code_scored_tokens": sum(
            int(record["full_code_scored_tokens"]) for record in task_records
        ),
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": IDENTITY_SCHEMA,
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authorization": {
            "stage_a_artifact_kind": STAGE_A_ARTIFACT_KIND,
            "stage_a_file_sha256": stage_a_file_sha,
            "stage_a_canonical_evidence_sha256": STAGE_A_CANONICAL_EVIDENCE_SHA256,
            "stage_a_implementation_commit": STAGE_A_IMPLEMENTATION_COMMIT,
            "stage_a_result_commit": STAGE_A_RESULT_COMMIT,
            "stage_a_gate_passed": True,
            "verified_before_dataset_access": True,
        },
        "row_plan": row_plan,
        "model_contract": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_loaded": False,
        },
        "tokenizer_contract": {
            "source_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": tokenizer.__class__.__name__,
            "transformers_version": actual_transformers_version,
            "trust_remote_code": False,
            "prompt_add_special_tokens": True,
            "code_add_special_tokens": False,
            "formatter_version": MBPP_FORMATTER_VERSION,
            "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
            "text_hash_encoding": TEXT_HASH_ENCODING,
        },
        "dataset": {
            "id": MBPP_DATASET_ID,
            "config": MBPP_CONFIG,
            "revision": MBPP_REVISION,
            "phase": "calibration",
            "source_split": mbpp_source_split("calibration"),
            "selection_namespace": MBPP_SELECTION_NAMESPACE,
            "formatter_version": MBPP_FORMATTER_VERSION,
            "selection_mode": "task_id_ranking_then_exact_task_id_stream",
            "selection_window": {
                "offset": STAGE_B_OFFSET,
                "limit": STAGE_B_LIMIT,
                "stop_exclusive": STAGE_B_STOP,
            },
            "protected_window": {
                "offset": PROTECTED_WINDOW[0],
                "stop_exclusive": PROTECTED_WINDOW[1],
                "content_retained_canonicalized_or_tokenized": False,
            },
            "ordered_task_ids": list(task_ids),
            "manifest": content_manifest,
            "content_manifest_sha256": content_hash,
            "token_manifest_sha256": token_hash,
            "ordered_identity_sha256": ordered_hash,
            "tasks": list(task_records),
            "totals": totals,
            "data_access": resolved_rows.access_audit.as_dict(
                selected_task_ids=task_ids,
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
            "commit": repository_end["commit"],
            "start": repository_start,
            "end": repository_end,
            "stable_commit": True,
        },
        "source_files": {
            "paths": list(SOURCE_FILES),
            "sha256_start": source_hashes_start,
            "sha256_end": source_hashes_end,
            "stable": True,
            "imported_modules": imported_module_paths,
        },
        "environment": runtime_environment,
    }
    validate_stage_b_identity_evidence(evidence)

    artifact = {
        "canonical_evidence_sha256": _sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    _atomic_write(args.output, canonical_json_bytes(artifact))
    verification = verify_evidence_artifact(args.output)
    if verification["valid"] is not True:
        raise RuntimeError(
            "written Stage-B identity artifact failed verification: "
            + "; ".join(verification["errors"])
        )

    post_write_repository = git_state(repository_root)
    post_write_source_hashes = source_file_hashes(repository_root)
    if post_write_repository != repository_start:
        raise RuntimeError("repository state changed while writing Stage-B identity")
    if post_write_source_hashes != source_hashes_start:
        raise RuntimeError("source files changed while writing Stage-B identity")
    loaded_evidence, file_sha = load_stage_b_identity_artifact(args.output)
    if loaded_evidence != evidence:
        raise RuntimeError("written Stage-B identity did not round-trip exactly")

    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "file_sha256": file_sha,
                "canonical_evidence_sha256": artifact["canonical_evidence_sha256"],
                "content_manifest_sha256": content_hash,
                "token_manifest_sha256": token_hash,
                "ordered_identity_sha256": ordered_hash,
                "ordered_task_ids": list(task_ids),
                "prompt_tokens": totals["prompt_tokens"],
                "code_tokens": totals["code_tokens"],
                "aligned_scored_tokens": totals["aligned_scored_tokens"],
                "protected_window_8_16_content_selected_retained_canonicalized_"
                "formatted_tokenized_passed_to_model_or_evaluated": False,
                "model_weights_loaded": False,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
