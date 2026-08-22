#!/usr/bin/env python3
"""Build a fail-closed Experiment 013 identity without loading model weights.

The resolver consumes an offline metadata manifest. Raw prompts, targets,
books, generated RULER examples, and token-ID arrays are forbidden: the input
contains only canonical identities, immutable revisions, spans, sizes, and
SHA-256 commitments produced by a separately audited extractor.

Resolution writes a deterministic *candidate* under a quarantine directory.
Freezing is a separate, explicit promotion bound to the candidate file hash.
This version deliberately refuses Stage B and Stage C before reading an input
file; protected identities require a later protocol amendment.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import string
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

INPUT_SCHEMA: Final = "recurquant.experiment013.identity-input.v5"
CANDIDATE_SCHEMA: Final = "recurquant.experiment013.identity-candidate.v5"
FROZEN_SCHEMA: Final = "recurquant.experiment013.identity-frozen.v5"
STAGE_A_CANDIDATE_SCHEMA: Final = "recurquant.experiment013.identity-candidate.v6"
STAGE_A_FROZEN_SCHEMA: Final = "recurquant.experiment013.identity-frozen.v6"
STAGE_A_IDENTITY_SCHEMA_VERSION: Final = 6
ARTIFACT_KIND: Final = "recurquant_static_rht_q468_identity"
# Procedure version.  The identity field sets remain the published v5 contract.
RESOLVER_VERSION: Final = 6
PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256: Final = (
    "ee5628e50e5d3516fd79077542d355fd915455ac0e53128d372f4177ad63d39c"
)

PRIMARY_MODEL_ID: Final = "Qwen/Qwen3.5-0.8B-Base"
PRIMARY_MODEL_REVISION: Final = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
CONDITIONAL_MODEL_ID: Final = "Qwen/Qwen3.5-2B-Base"
CONDITIONAL_MODEL_REVISION: Final = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TRANSFORMERS_VERSION: Final = "5.14.1"

MBPP_DATASET_ID: Final = "google-research-datasets/mbpp"
MBPP_CONFIG: Final = "full"
MBPP_REVISION: Final = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_SELECTION_NAMESPACE: Final = "rq-v0.2"
PG19_DATASET_ID: Final = "emozilla/pg19"
PG19_REVISION: Final = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"
RULER_SOURCE_ID: Final = "NVIDIA/RULER"
RULER_REVISION: Final = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
HUMANEVAL_PLUS_DATASET_ID: Final = "evalplus/humanevalplus"
HUMANEVAL_PLUS_REVISION: Final = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
EVALPLUS_SOURCE_ID: Final = "evalplus/evalplus"
EVALPLUS_SOURCE_REVISION: Final = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
DATASET_KEYS: Final = ("mbpp", "pg19", "ruler", "humaneval_plus")
FAMILY_ORDER: Final = {key: index for index, key in enumerate(DATASET_KEYS)}
FROZEN_DATASET_REVISIONS: Final = {
    "mbpp": MBPP_REVISION,
    "pg19": PG19_REVISION,
    "ruler": RULER_REVISION,
    "humaneval_plus": HUMANEVAL_PLUS_REVISION,
}
FROZEN_CANONICAL_ID_FIELDS: Final = {
    "mbpp": "task_id",
    "pg19": "url",
    "ruler": "configuration_id",
    "humaneval_plus": "task_id",
}
FROZEN_DATASET_CONFIGS: Final = {
    "mbpp": MBPP_CONFIG,
    "pg19": "default",
    "ruler": "official-generator",
    "humaneval_plus": "default",
}
FROZEN_DATASET_SPLITS: Final = {
    "calibration": {
        "mbpp": "train",
        "pg19": "train",
        "ruler": "generated",
        "humaneval_plus": "test",
    },
    "stage_a": {
        "mbpp": "train",
        "pg19": "validation",
        "ruler": "generated",
        "humaneval_plus": "test",
    },
}
FROZEN_FORMATTER_IDS: Final = {
    "mbpp": "recurquant.mbpp-prompt-code.v1",
    "pg19": "recurquant.pg19-token-slice.v1",
    "ruler": "recurquant.ruler-official-generated-record.v1",
    "humaneval_plus": "recurquant.humaneval-plus-prompt-solution.v1",
}
FROZEN_STATIC_FORMATTER_SHA256: Final = {
    "mbpp": "882e20ec9f5cbcb7e6f1310cbf46d19153721beac41ecc7ee308c39be17532ff",
    "pg19": "faea2480bf85adcd34339cae88a0e9b631b705eb547020572db74402bf525730",
    "humaneval_plus": "12204389715ddc210e5a8b1b291f4fbcaf2b64fde94c22699873de80d682204c",
}
MAX_METADATA_STRING_LENGTH: Final = 512
PG19_CANONICAL_URL_RE: Final = re.compile(
    r"http://www\.gutenberg\.org/ebooks/(?P<ebook_id>[1-9][0-9]{0,7})\Z"
)
HUMANEVAL_PLUS_TASK_ID_RE: Final = re.compile(r"HumanEval/(?P<task_number>0|[1-9][0-9]{0,2})\Z")
HUMANEVAL_PLUS_TASK_COUNT: Final = 164
TOKENIZER_CLASS_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}\Z")
TOKENIZER_FILE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

PG19_TRAIN_NAMESPACE: Final = "recurquant.experiment013.pg19.train.v1\0"
PG19_VALIDATION_NAMESPACE: Final = "recurquant.experiment013.pg19.validation.v1\0"
PG19_TEST_NAMESPACE: Final = "recurquant.experiment013.pg19.test.v1\0"
HUMANEVAL_AB_NAMESPACE: Final = "recurquant.experiment013.humaneval-plus.stage-a-b.v1\0"
HUMANEVAL_C_NAMESPACE: Final = "recurquant.experiment013.humaneval-plus.stage-c.v1\0"
CALIBRATION_SPLIT_NAMESPACE: Final = "recurquant.experiment013.calibration-split.v1\0"
IDENTITY_RECORD_NAMESPACE: Final = "recurquant.experiment013.identity-record.v1\0"
FISHER_BOUNDARY_SCHEMA: Final = "recurquant.experiment013.fisher-boundary.v1"
FISHER_BOUNDARY_NAMESPACE: Final = "recurquant.experiment013.fisher-boundary.v1\0"
FISHER_BOUNDARY_TOKEN_NAMESPACE: Final = (
    "recurquant.experiment013.fisher-boundary-token-sequence.v1\0"
)
FISHER_BOUNDARY_HORIZON: Final = 1
RULER_CALIBRATION_SELECTION_NAMESPACE: Final = (
    "recurquant.experiment013.ruler.calibration-sequence.v1\0"
)
RULER_STAGE_A_SELECTION_NAMESPACE: Final = "recurquant.experiment013.ruler.stage-a-sequence.v1\0"
RULER_SEQUENCE_NAMESPACE: Final = "recurquant.experiment013.ruler.sequence.v1"
RULER_CATEGORIES: Final = (
    "retrieval",
    "multi_hop_tracing",
    "aggregation",
    "question_answering",
)
RULER_CONFIG_CATEGORY: Final = {
    "niah_single_1": "retrieval",
    "niah_single_2": "retrieval",
    "niah_single_3": "retrieval",
    "niah_multikey_1": "retrieval",
    "niah_multikey_2": "retrieval",
    "niah_multikey_3": "retrieval",
    "niah_multivalue": "retrieval",
    "niah_multiquery": "retrieval",
    "vt": "multi_hop_tracing",
    "cwe": "aggregation",
    "fwe": "aggregation",
    "qa_1": "question_answering",
    "qa_2": "question_answering",
}
RULER_CALIBRATION_SCHEDULE: Final = (
    ("retrieval", "niah_multiquery", 2_048, 12_339),
    ("retrieval", "niah_multikey_2", 2_048, 12_340),
    ("retrieval", "niah_single_1", 4_096, 12_339),
    ("retrieval", "niah_multivalue", 4_096, 12_340),
    ("multi_hop_tracing", "vt", 2_048, 12_339),
    ("multi_hop_tracing", "vt", 2_048, 12_340),
    ("multi_hop_tracing", "vt", 4_096, 12_339),
    ("multi_hop_tracing", "vt", 4_096, 12_340),
    ("aggregation", "fwe", 2_048, 12_339),
    ("aggregation", "cwe", 2_048, 12_340),
    ("aggregation", "fwe", 4_096, 12_339),
    ("aggregation", "cwe", 4_096, 12_340),
    ("question_answering", "qa_1", 2_048, 12_339),
    ("question_answering", "qa_2", 2_048, 12_340),
    ("question_answering", "qa_1", 4_096, 12_339),
    ("question_answering", "qa_2", 4_096, 12_340),
)
RULER_STAGE_A_SCHEDULE: Final = (
    ("retrieval", "niah_multiquery", 4_096, 2_343),
    ("multi_hop_tracing", "vt", 4_096, 2_343),
    ("aggregation", "fwe", 4_096, 2_343),
    ("question_answering", "qa_1", 4_096, 2_343),
)

CLAIM_BOUNDARY: Final = (
    "This artifact freezes Experiment 013 data, tokenizer, source, runtime, and "
    "metadata-only model-file identity. "
    "It is not quality, latency, novelty, state-of-the-art, deployment, or "
    "breakthrough evidence."
)
PROTECTED_STAGES: Final = frozenset({"stage_b", "stage_c"})
ALLOWED_PHASES: Final = frozenset({"calibration", "stage_a"})
HEX_LENGTHS: Final = frozenset({40, 64})

DATASET_FIELDS: Final = frozenset(
    {
        "key",
        "dataset_id",
        "config",
        "revision",
        "split",
        "canonical_id_field",
        "canonical_id_manifest_sha256",
        "formatter_id",
        "formatter_sha256",
    }
)
TOKENIZER_FIELDS: Final = frozenset(
    {"source_id", "revision", "class", "transformers_version", "files"}
)
TOKENIZER_FILE_FIELDS: Final = frozenset({"name", "sha256", "size_bytes"})
RECORD_FIELDS: Final = frozenset(
    {
        "family",
        "canonical_id",
        "config",
        "selection_rank",
        "selection_sha256",
        "seed",
        "configured_length",
        "sequence_length",
        "ruler_category",
        "generator_receipt_sha256",
        "source_content_sha256",
        "formatted_content_sha256",
        "prompt_token_ids_sha256",
        "target_token_ids_sha256",
        "sequence_token_ids_sha256",
        "tokenizer_manifest_sha256",
        "token_span",
        "anchor_manifest_sha256",
        "fisher_boundary",
        "identity_record_sha256",
    }
)
IDENTITY_RECORD_PAYLOAD_FIELDS: Final = RECORD_FIELDS - {"identity_record_sha256"}
TOKEN_SPAN_FIELDS: Final = frozenset(
    {
        "prefill_start",
        "prefill_stop",
        "scored_start",
        "scored_stop",
        "cache_exposed_start",
        "cache_exposed_stop",
    }
)
FISHER_BOUNDARY_FIELDS: Final = frozenset(
    {
        "schema",
        "horizon",
        "boundary_positions",
        "input_positions",
        "target_positions",
        "input_token_ids_sha256",
        "target_token_ids_sha256",
        "fisher_boundary_sha256",
    }
)
FISHER_BOUNDARY_PAYLOAD_FIELDS: Final = FISHER_BOUNDARY_FIELDS - {"fisher_boundary_sha256"}
CALIBRATION_BINDING_FIELDS: Final = frozenset(
    {
        "calibration_authorization_file_sha256",
        "calibration_identity_file_sha256",
        "calibration_score_artifact_file_sha256",
        "comparator_score_artifact_file_sha256",
        "split_half_stability_artifact_file_sha256",
        "static_fisher_k29334_policy_file_sha256",
        "static_k27030_policy_file_sha256",
        "static_k29334_policy_file_sha256",
        "static_mse_k29334_policy_file_sha256",
    }
)
EXECUTION_BINDING_FIELDS: Final = frozenset(
    {
        "repository_source_manifest_file_sha256",
        "calibration_runtime_manifest_file_sha256",
        "model_file_manifest_file_sha256",
        "parquet_materialization_manifest_file_sha256",
    }
)
FROZEN_EVIDENCE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "identity_schema",
        "resolver_version",
        "status",
        "phase",
        "identity_only",
        "claim_boundary",
        "source_manifest_sha256",
        "execution_bindings",
        "model_contracts",
        "datasets",
        "upstream_tool_contracts",
        "tokenizer",
        "records",
        "record_count",
        "content_manifest_sha256",
        "selection",
        "calibration_split_half",
        "calibration_binding",
        "protected_identity",
        "promotion_required",
        "promotion",
    }
)
CANDIDATE_EVIDENCE_FIELDS: Final = FROZEN_EVIDENCE_FIELDS - {"promotion"}
STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD: Final = "stage_a_capture_provenance_receipt_file_sha256"
STAGE_A_FROZEN_EVIDENCE_FIELDS: Final = FROZEN_EVIDENCE_FIELDS | {
    STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD
}
STAGE_A_CANDIDATE_EVIDENCE_FIELDS: Final = STAGE_A_FROZEN_EVIDENCE_FIELDS - {"promotion"}
FROZEN_RECORD_FIELDS: Final = RECORD_FIELDS | {
    "anchor_positions",
    "anchor_positions_sha256",
}
STAGE_A_BINDING_ARTIFACT_KIND: Final = "recurquant_experiment013_stage_a_calibration_binding"
STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION: Final = 4
STAGE_A_BINDING_ARTIFACT_REVISION: Final = "experiment-013-stage-a-calibration-binding-v4"
STAGE_A_CORE_BINDING_ARTIFACT_KIND: Final = (
    "recurquant_experiment013_stage_a_calibration_core_binding"
)
STAGE_A_CORE_BINDING_ARTIFACT_SCHEMA_VERSION: Final = 3
STAGE_A_CORE_BINDING_ARTIFACT_REVISION: Final = "experiment-013-stage-a-calibration-core-binding-v3"
STAGE_A_CALIBRATION_AUTHORIZATION_ARTIFACT_KIND: Final = (
    "recurquant_experiment013_stage_a_calibration_authorization"
)
STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA_VERSION: Final = 1
STAGE_A_CALIBRATION_AUTHORIZATION_REVISION: Final = (
    "experiment-013-stage-a-calibration-authorization-v1"
)
STAGE_A_CALIBRATION_AUTHORIZATION_STATUS: Final = "authorized_for_stage_a"
CALIBRATION_RUN_REPORT_KIND: Final = "recurquant_experiment013_calibration_run"
CALIBRATION_RUN_REPORT_SCHEMA_VERSION: Final = 3
CALIBRATION_RUNNER_REVISION: Final = "experiment-013-static-q468-calibration-runner-v10"
CALIBRATION_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_calibration_identity_capture_provenance"
)
CALIBRATION_CAPTURE_PROVENANCE_SCHEMA_VERSION: Final = 2
CALIBRATION_CAPTURE_VERSION: Final = 6
CALIBRATION_CAPTURE_PROVENANCE_STATUS: Final = (
    "captured_under_authenticated_runtime_and_launcher_finalized"
)
CALIBRATION_CAPTURE_PUBLICATION_CONTRACT: Final = (
    "sealed-host-no-overwrite-after-postconditions-and-owned-root-cleanup-v1"
)
STAGE_A_CAPTURE_PROVENANCE_KIND: Final = (
    "recurquant_experiment013_stage_a_identity_capture_provenance"
)
STAGE_A_CAPTURE_PROVENANCE_SCHEMA_VERSION: Final = 1
STAGE_A_CAPTURE_PROVENANCE_STATUS: Final = (
    "captured_under_authenticated_runtime_and_launcher_finalized"
)
STAGE_A_CAPTURE_PUBLICATION_CONTRACT: Final = CALIBRATION_CAPTURE_PUBLICATION_CONTRACT
CALIBRATION_COMPLETE_BYTES: Final = b"recurquant-experiment013-calibration-complete-v1\n"
FISHER_H1_SMOKE_COMPLETE_BYTES: Final = b"recurquant-experiment013-fisher-h1-smoke-complete-v1\n"
CALIBRATION_OUTPUT_FILENAMES: Final = MappingProxyType(
    {
        "calibration_score_artifact": "calibration-scores.json",
        "comparator_score_artifact": "comparator-scores.json",
        "split_half_stability_artifact": "split-half-stability.json",
        "static_k27030_policy_artifact": "static-k27030-policy.json",
        "static_k29334_policy_artifact": "static-k29334-policy.json",
        "static_mse_k29334_policy_artifact": "static-mse-k29334-policy.json",
        "static_fisher_k29334_policy_artifact": ("static-diagonal-fisher-h1-k29334-policy.json"),
        "static_q48_policy_artifact": "static-q48-p14739-policy.json",
        "calibration_core_binding_artifact": "stage-a-calibration-core-binding.json",
    }
)
CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES: Final = frozenset(
    {
        "calibration_complete_marker",
        "calibration_core_binding_artifact",
        "calibration_run_report",
        "calibration_runtime_manifest",
        "capture_provenance_receipt",
        "fisher_h1_smoke_complete_marker",
        "fisher_h1_smoke_report",
        "model_file_manifest",
        "repository_source_manifest",
        "static_q48_policy_artifact",
    }
)

CALIBRATION_RUNTIME_MANIFEST_KIND: Final = "recurquant_experiment013_calibration_runtime_manifest"
CALIBRATION_RUNTIME_MANIFEST_SCHEMA_VERSION: Final = 5
CALIBRATION_MODEL_FILE_MANIFEST_KIND: Final = "recurquant_experiment013_model_file_manifest"
CALIBRATION_MODEL_FILE_MANIFEST_SCHEMA_VERSION: Final = 1
CALIBRATION_MODEL_FILE_MANIFEST_DERIVATION: Final = "huggingface-hub-pinned-tree-lfs-v1"
CALIBRATION_MODEL_FILE_SELECTION_PROFILE: Final = "qwen35-config-index-safetensors-v1"
CALIBRATION_CAPTURE_SOURCE_PATH: Final = "scripts/capture_static_q468_identity_input.py"
CALIBRATION_CAPTURE_CRITICAL_MODULE_DISTRIBUTIONS: Final = MappingProxyType(
    {
        "datasets": "datasets",
        "fsspec": "fsspec",
        "huggingface_hub": "huggingface-hub",
        "numpy": "numpy",
        "pyarrow": "pyarrow",
        "tokenizers": "tokenizers",
        "transformers": "transformers",
    }
)
CALIBRATION_CAPTURE_EXCLUDED_RUNTIME_MODULES: Final = ("pkg_resources", "setuptools")
CALIBRATION_QUERY_ENERGY_EMA: Final = MappingProxyType(
    {
        "decay_hex": (2.0 ** (-1.0 / 32.0)).hex(),
        "epsilon_hex": (1.0e-6).hex(),
        "prior": "uniform_1_over_key_rows",
    }
)
CALIBRATION_CANONICAL_ADAPTER_REVISION: Final = "experiment-013-qwen35-live-adapter-v2"
CALIBRATION_CANONICAL_ADAPTER_KERNEL_BACKEND: Final = "transformers_pure_torch_gated_delta_rule"
CALIBRATION_CANONICAL_ADAPTER_MODEL_DTYPE: Final = "bfloat16"
CALIBRATION_CANONICAL_TORCH_DISTRIBUTION_VERSION: Final = "2.13.0+cu130"
CALIBRATION_CANONICAL_TORCH_RUNTIME_VERSION: Final = "2.13.0+cu130"
CALIBRATION_CANONICAL_CUDA_RUNTIME_VERSION: Final = "13.0"
CALIBRATION_CANONICAL_ADAPTER_QUERY_SHAPE: Final = (1, 1, 16, 128)
CALIBRATION_CANONICAL_ADAPTER_STATE_SHAPE: Final = (1, 16, 128, 128)
CALIBRATION_CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES: Final = (
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
CALIBRATION_CANONICAL_ADAPTER_LOADING_DIAGNOSTICS: Final = frozenset(
    {"error_msgs", "mismatched_keys", "missing_keys", "unexpected_keys"}
)
CALIBRATION_SEALED_LAUNCH_POLICY: Final = MappingProxyType(
    {
        "bootstrap_mode": "stdlib-only-exact-runner-and-capture-v2",
        "cache_confinement_mode": "private-scratch-plus-explicit-dataset-root-v1",
        "child_cwd_mode": "authenticated-launcher-owned-scratch-v1",
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "package_path_mode": "authenticated-record-only-roots-v1",
        "pycache_mode": "new-verified-empty-prefix-v1",
        "safe_path": True,
        "site_loaded": False,
        "sys_path_mode": "staged-base-then-authenticated-packages-v1",
        "utf8_mode": 1,
        "virtualenv_hook_loaded": False,
    }
)
CALIBRATION_CORE_DEPENDENCY_NAMES: Final = frozenset(
    (
        set(CALIBRATION_OUTPUT_FILENAMES)
        - {"static_q48_policy_artifact", "calibration_core_binding_artifact"}
    )
    | {"frozen_identity_artifact"}
)


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository's deterministic JSON representation."""

    return (
        json.dumps(
            _deep_thaw(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in string.hexdigits for character in value)
    )


def require_sha256(value: object, *, context: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def require_exact_revision(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in HEX_LENGTHS
        or value != value.lower()
        or not all(character in string.hexdigits for character in value)
    ):
        raise ValueError(f"{context} must be an immutable lowercase 40- or 64-hex revision")
    return value


def require_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def require_string(
    value: object,
    *,
    context: str,
    allow_empty: bool = False,
    maximum_length: int = MAX_METADATA_STRING_LENGTH,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or value != value.strip():
        raise ValueError(f"{context} must be stripped NFC text")
    if len(value) > maximum_length:
        raise ValueError(f"{context} exceeds the metadata length limit of {maximum_length}")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{context} contains a forbidden control character")
    if any(character.isspace() for character in value):
        raise ValueError(f"{context} cannot contain whitespace or raw content")
    return value


def _validate_pg19_canonical_url(value: str, *, context: str) -> None:
    """Require the exact URL shape emitted by the pinned PG19 extractor."""

    match = PG19_CANONICAL_URL_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"{context} must be an exact http://www.gutenberg.org/ebooks/<positive-id> URL"
        )


def _validate_humaneval_plus_task_id(value: str, *, context: str) -> None:
    """Require one canonical HumanEval+ task ID from the pinned 164-row split."""

    match = HUMANEVAL_PLUS_TASK_ID_RE.fullmatch(value)
    if match is None or int(match.group("task_number")) >= HUMANEVAL_PLUS_TASK_COUNT:
        raise ValueError(f"{context} must be a canonical HumanEval/0..163 task ID")


class _FrozenSequence(tuple[Any, ...]):
    """Tuple storage with structural equality against ordinary JSON arrays."""

    def __new__(cls, values: Sequence[Any]) -> _FrozenSequence:
        return super().__new__(cls, values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = tuple.__hash__

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("verified resolver sequences are immutable")

    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _deep_freeze(value: Any) -> Any:
    """Recursively detach and freeze JSON-like verified data."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _FrozenSequence(tuple(_deep_freeze(item) for item in value))
    return value


def _deep_thaw(value: Any) -> Any:
    """Convert immutable DTO views back to ordinary canonical-JSON containers."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_deep_thaw(item) for item in value]
    return value


def require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def require_sequence(value: object, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    return value


def require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} fields drifted; missing={missing}, extra={extra}")


def sequence_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash the exact ordered token-ID sequence using resolver canonical JSON."""

    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(token_ids, Sequence):
        raise ValueError("sequence token IDs must be an integer sequence")
    normalized = [
        require_int(token_id, context=f"sequence token IDs[{index}]")
        for index, token_id in enumerate(token_ids)
    ]
    return sha256_bytes(canonical_json_bytes(normalized))


def _fisher_boundary_token_ids_sha256(token_ids: Sequence[int], *, role: str) -> str:
    """Hash one ordered Fisher token sequence under an exact role binding."""

    if role not in {"input", "target"}:
        raise ValueError("Fisher boundary token role must be input or target")
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(token_ids, Sequence):
        raise ValueError("Fisher boundary token IDs must be an integer sequence")
    normalized = [
        require_int(token_id, context=f"Fisher boundary {role} token IDs[{index}]")
        for index, token_id in enumerate(token_ids)
    ]
    payload = {"role": role, "token_ids": normalized}
    return sha256_bytes(
        FISHER_BOUNDARY_TOKEN_NAMESPACE.encode("utf-8") + canonical_json_bytes(payload)
    )


def fisher_boundary_sha256(boundary: Mapping[str, Any]) -> str:
    """Hash the public Fisher boundary payload under its dedicated domain."""

    missing = FISHER_BOUNDARY_PAYLOAD_FIELDS - set(boundary)
    if missing:
        raise ValueError(f"Fisher boundary payload is missing fields: {sorted(missing)}")
    payload = {name: boundary[name] for name in sorted(FISHER_BOUNDARY_PAYLOAD_FIELDS)}
    return sha256_bytes(FISHER_BOUNDARY_NAMESPACE.encode("utf-8") + canonical_json_bytes(payload))


def build_fisher_boundary_contract(token_ids: Sequence[int]) -> dict[str, Any]:
    """Bind the frozen H=1 Fisher input/target pairs without exposing token IDs."""

    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(token_ids, Sequence):
        raise ValueError("Fisher boundary token IDs must be an integer sequence")
    normalized = [
        require_int(token_id, context=f"Fisher boundary sequence token IDs[{index}]")
        for index, token_id in enumerate(token_ids)
    ]
    if len(normalized) < 3:
        raise ValueError("Fisher boundary requires a sequence length of at least three tokens")
    boundary_positions = list(anchor_positions(len(normalized) - 2))
    input_positions = [boundary + FISHER_BOUNDARY_HORIZON for boundary in boundary_positions]
    target_positions = [position + 1 for position in input_positions]
    payload: dict[str, Any] = {
        "schema": FISHER_BOUNDARY_SCHEMA,
        "horizon": FISHER_BOUNDARY_HORIZON,
        "boundary_positions": boundary_positions,
        "input_positions": input_positions,
        "target_positions": target_positions,
        "input_token_ids_sha256": _fisher_boundary_token_ids_sha256(
            [normalized[position] for position in input_positions],
            role="input",
        ),
        "target_token_ids_sha256": _fisher_boundary_token_ids_sha256(
            [normalized[position] for position in target_positions],
            role="target",
        ),
    }
    payload["fisher_boundary_sha256"] = fisher_boundary_sha256(payload)
    return payload


def _normalize_fisher_boundary(
    value: object, *, sequence_length: int, context: str
) -> dict[str, Any]:
    """Strictly validate a redacted Fisher boundary against B(T), H, and itself."""

    boundary = require_mapping(value, context=context)
    require_exact_fields(boundary, FISHER_BOUNDARY_FIELDS, context=context)
    if boundary["schema"] != FISHER_BOUNDARY_SCHEMA:
        raise ValueError(f"{context} schema drifted")
    horizon = require_int(boundary["horizon"], context=f"{context}.horizon", minimum=1)
    if horizon != FISHER_BOUNDARY_HORIZON:
        raise ValueError(f"{context} horizon must equal H=1")
    if sequence_length < 3:
        raise ValueError(f"{context} requires a sequence length of at least three tokens")

    expected_boundary_positions = list(anchor_positions(sequence_length - 2))
    expected_input_positions = [position + horizon for position in expected_boundary_positions]
    expected_target_positions = [position + 1 for position in expected_input_positions]

    def positions(name: str) -> list[int]:
        raw_positions = require_sequence(boundary[name], context=f"{context}.{name}")
        return [
            require_int(position, context=f"{context}.{name}[{index}]")
            for index, position in enumerate(raw_positions)
        ]

    normalized_boundary_positions = positions("boundary_positions")
    normalized_input_positions = positions("input_positions")
    normalized_target_positions = positions("target_positions")
    if normalized_boundary_positions != expected_boundary_positions:
        raise ValueError(f"{context} boundary positions differ from B(T)=anchor_positions(T-2)")
    if normalized_input_positions != expected_input_positions:
        raise ValueError(f"{context} input positions differ from x[b+1]")
    if normalized_target_positions != expected_target_positions:
        raise ValueError(f"{context} target positions differ from x[b+2]")

    normalized = {
        "schema": FISHER_BOUNDARY_SCHEMA,
        "horizon": horizon,
        "boundary_positions": normalized_boundary_positions,
        "input_positions": normalized_input_positions,
        "target_positions": normalized_target_positions,
        "input_token_ids_sha256": require_sha256(
            boundary["input_token_ids_sha256"],
            context=f"{context}.input_token_ids_sha256",
        ),
        "target_token_ids_sha256": require_sha256(
            boundary["target_token_ids_sha256"],
            context=f"{context}.target_token_ids_sha256",
        ),
        "fisher_boundary_sha256": require_sha256(
            boundary["fisher_boundary_sha256"],
            context=f"{context}.fisher_boundary_sha256",
        ),
    }
    if normalized["fisher_boundary_sha256"] != fisher_boundary_sha256(normalized):
        raise ValueError(f"{context} self-hash drifted")
    return normalized


def identity_anchor_manifest_sha256(
    *,
    canonical_id: str,
    sequence_length: int,
    sequence_token_ids_sha256_value: str,
    token_span: Mapping[str, Any],
) -> str:
    """Recompute the exact capture anchor-manifest commitment."""

    normalized_id = require_string(canonical_id, context="anchor canonical_id")
    length = require_int(sequence_length, context="anchor sequence_length", minimum=1)
    token_hash = require_sha256(
        sequence_token_ids_sha256_value,
        context="anchor sequence_token_ids_sha256",
    )
    require_exact_fields(token_span, TOKEN_SPAN_FIELDS, context="anchor token_span")
    normalized_span = {
        name: require_int(token_span[name], context=f"anchor token_span.{name}")
        for name in sorted(TOKEN_SPAN_FIELDS)
    }
    manifest = {
        "canonical_id": normalized_id,
        "positions": list(anchor_positions(length)),
        "sequence_token_ids_sha256": token_hash,
        "token_span": normalized_span,
    }
    return sha256_bytes(canonical_json_bytes(manifest))


def identity_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash the exact capture record payload, excluding its self-hash."""

    missing = IDENTITY_RECORD_PAYLOAD_FIELDS - set(record)
    if missing:
        raise ValueError(f"identity record payload is missing fields: {sorted(missing)}")
    payload = {name: record[name] for name in sorted(IDENTITY_RECORD_PAYLOAD_FIELDS)}
    return sha256_bytes(IDENTITY_RECORD_NAMESPACE.encode("utf-8") + canonical_json_bytes(payload))


def selection_sha256(namespace: str, canonical_id: str) -> str:
    return sha256_bytes(namespace.encode("utf-8") + canonical_id.encode("utf-8"))


def ruler_canonical_id(*, category: str, config: str, configured_length: int, seed: int) -> str:
    """Return the sole canonical identity for one frozen RULER generation tuple."""

    return (
        f"{RULER_SEQUENCE_NAMESPACE}:{RULER_REVISION}:{category}:{config}:"
        f"length={configured_length}:seed={seed}:sample=0"
    )


def mbpp_selection_sha256(canonical_id: str) -> str:
    return sha256_bytes(f"{MBPP_SELECTION_NAMESPACE}|{canonical_id}".encode())


def calibration_split_key(record: Mapping[str, Any]) -> str:
    identity = "\0".join(
        (
            str(record["family"]),
            str(record["ruler_category"]),
            str(record["config"]),
            str(record["canonical_id"]),
            str(record["seed"]),
            str(record["configured_length"]),
            str(record["sequence_length"]),
        )
    )
    return selection_sha256(CALIBRATION_SPLIT_NAMESPACE, identity)


def anchor_positions(token_count: int) -> tuple[int, ...]:
    """Return the frozen unique post-token anchor positions."""

    if token_count < 1:
        raise ValueError("calibration sequence token count must be positive")
    if token_count < 16:
        return tuple(range(token_count))
    positions = tuple((j + 1) * token_count // 16 - 1 for j in range(16))
    if len(positions) != len(set(positions)):
        raise RuntimeError("frozen anchor equation produced duplicate positions")
    return positions


def mbpp_calibration_identity() -> tuple[tuple[str, ...], str]:
    """Return the existing v0.2 ID-only calibration selection and pool hash."""

    population = tuple(str(task_id) for task_id in range(601, 975))
    ranked = sorted(population, key=lambda task_id: (mbpp_selection_sha256(task_id), int(task_id)))
    selected = tuple(ranked[:128])
    population_hash = sha256_bytes(canonical_json_bytes(population))
    return selected, population_hash


def _validate_sha_rank_order(rows: Sequence[Mapping[str, Any]], *, context: str) -> None:
    selection_keys = [(str(row["selection_sha256"]), str(row["canonical_id"])) for row in rows]
    if len(selection_keys) != len(set(selection_keys)):
        raise ValueError(f"{context} contains duplicate canonical selection keys")
    ranked = sorted(
        rows,
        key=lambda row: (str(row["selection_sha256"]), str(row["canonical_id"])),
    )
    if [int(row["selection_rank"]) for row in ranked] != list(range(len(rows))):
        raise ValueError(f"{context} selection ranks do not match SHA-256 order")


def _json_without_duplicate_keys(raw: bytes, *, context: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain one JSON object")
    return value


def _validate_dataset_contracts(
    value: object,
    *,
    expected_revisions: Mapping[str, str],
    phase: str,
) -> tuple[dict[str, Any], ...]:
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"dataset contract phase is unsupported: {phase!r}")
    entries = require_sequence(value, context="datasets")
    if len(entries) != len(DATASET_KEYS):
        raise ValueError("datasets must contain exactly four contracts")
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(entries):
        item = require_mapping(raw, context=f"datasets[{index}]")
        require_exact_fields(item, DATASET_FIELDS, context=f"datasets[{index}]")
        key = require_string(item["key"], context=f"datasets[{index}].key")
        if key not in DATASET_KEYS or key in normalized:
            raise ValueError(f"dataset key is unknown or duplicated: {key}")
        revision = require_exact_revision(item["revision"], context=f"datasets[{index}].revision")
        if revision != expected_revisions[key]:
            raise ValueError(f"{key} dataset revision does not match the CLI contract")
        contract = {
            "key": key,
            "dataset_id": require_string(
                item["dataset_id"], context=f"datasets[{index}].dataset_id"
            ),
            "config": require_string(
                item["config"], context=f"datasets[{index}].config", allow_empty=True
            ),
            "revision": revision,
            "split": require_string(item["split"], context=f"datasets[{index}].split"),
            "canonical_id_field": require_string(
                item["canonical_id_field"],
                context=f"datasets[{index}].canonical_id_field",
            ),
            "canonical_id_manifest_sha256": require_sha256(
                item["canonical_id_manifest_sha256"],
                context=f"datasets[{index}].canonical_id_manifest_sha256",
            ),
            "formatter_id": require_string(
                item["formatter_id"], context=f"datasets[{index}].formatter_id"
            ),
            "formatter_sha256": require_sha256(
                item["formatter_sha256"],
                context=f"datasets[{index}].formatter_sha256",
            ),
        }
        if contract["canonical_id_field"] != FROZEN_CANONICAL_ID_FIELDS[key]:
            raise ValueError(
                f"{key} canonical ID field must be {FROZEN_CANONICAL_ID_FIELDS[key]!r}"
            )
        if contract["config"] != FROZEN_DATASET_CONFIGS[key]:
            raise ValueError(f"{key} dataset config must be {FROZEN_DATASET_CONFIGS[key]!r}")
        if contract["split"] != FROZEN_DATASET_SPLITS[phase][key]:
            raise ValueError(
                f"{phase} {key} dataset split must be {FROZEN_DATASET_SPLITS[phase][key]!r}"
            )
        if contract["formatter_id"] != FROZEN_FORMATTER_IDS[key]:
            raise ValueError(f"{key} formatter ID must be {FROZEN_FORMATTER_IDS[key]!r}")
        expected_formatter_sha256 = FROZEN_STATIC_FORMATTER_SHA256.get(key)
        if (
            expected_formatter_sha256 is not None
            and contract["formatter_sha256"] != expected_formatter_sha256
        ):
            raise ValueError(f"{key} formatter SHA-256 drifted from the frozen specification")
        if key == "mbpp" and (
            contract["dataset_id"] != MBPP_DATASET_ID or contract["config"] != MBPP_CONFIG
        ):
            raise ValueError("MBPP identity must match the frozen v0.2 source")
        if key == "mbpp":
            _selected_ids, expected_population_hash = mbpp_calibration_identity()
            if contract["canonical_id_manifest_sha256"] != expected_population_hash:
                raise ValueError("MBPP train ID manifest does not match the frozen v0.2 pool")
        expected_source_id = {
            "mbpp": MBPP_DATASET_ID,
            "pg19": PG19_DATASET_ID,
            "ruler": RULER_SOURCE_ID,
            "humaneval_plus": HUMANEVAL_PLUS_DATASET_ID,
        }[key]
        if contract["dataset_id"] != expected_source_id:
            raise ValueError(f"{key} source ID does not match the frozen upstream identity")
        normalized[key] = contract
    return tuple(normalized[key] for key in DATASET_KEYS)


def _validate_tokenizer(value: object) -> dict[str, Any]:
    tokenizer = require_mapping(value, context="tokenizer")
    require_exact_fields(tokenizer, TOKENIZER_FIELDS, context="tokenizer")
    if tokenizer["source_id"] != PRIMARY_MODEL_ID:
        raise ValueError("tokenizer source must equal the primary model ID")
    if tokenizer["revision"] != PRIMARY_MODEL_REVISION:
        raise ValueError("tokenizer revision must equal the pinned primary revision")
    if tokenizer["transformers_version"] != TRANSFORMERS_VERSION:
        raise ValueError("Transformers version drifted")
    tokenizer_class = require_string(tokenizer["class"], context="tokenizer.class")
    if TOKENIZER_CLASS_RE.fullmatch(tokenizer_class) is None:
        raise ValueError("tokenizer.class must be a bounded Python identifier")
    raw_files = require_sequence(tokenizer["files"], context="tokenizer.files")
    if not raw_files:
        raise ValueError("tokenizer.files cannot be empty")
    files: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_files):
        item = require_mapping(raw, context=f"tokenizer.files[{index}]")
        require_exact_fields(item, TOKENIZER_FILE_FIELDS, context=f"tokenizer.files[{index}]")
        name = require_string(item["name"], context=f"tokenizer.files[{index}].name")
        if (
            Path(name).name != name
            or TOKENIZER_FILE_NAME_RE.fullmatch(name) is None
            or name in names
        ):
            raise ValueError("tokenizer file names must be unique basenames")
        names.add(name)
        files.append(
            {
                "name": name,
                "sha256": require_sha256(
                    item["sha256"], context=f"tokenizer.files[{index}].sha256"
                ),
                "size_bytes": require_int(
                    item["size_bytes"],
                    context=f"tokenizer.files[{index}].size_bytes",
                    minimum=1,
                ),
            }
        )
    files.sort(key=lambda item: item["name"])
    file_manifest_hash = sha256_bytes(canonical_json_bytes(files))
    return {
        "source_id": PRIMARY_MODEL_ID,
        "revision": PRIMARY_MODEL_REVISION,
        "class": tokenizer_class,
        "transformers_version": TRANSFORMERS_VERSION,
        "files": files,
        "file_manifest_sha256": file_manifest_hash,
    }


def _selection_namespace(phase: str, family: str) -> str | None:
    if phase == "calibration":
        if family == "mbpp":
            return None
        if family == "pg19":
            return PG19_TRAIN_NAMESPACE
        if family == "ruler":
            return RULER_CALIBRATION_SELECTION_NAMESPACE
        return CALIBRATION_SPLIT_NAMESPACE
    if phase == "stage_a":
        if family == "pg19":
            return PG19_VALIDATION_NAMESPACE
        if family == "humaneval_plus":
            return HUMANEVAL_AB_NAMESPACE
        if family == "ruler":
            return RULER_STAGE_A_SELECTION_NAMESPACE
        return CALIBRATION_SPLIT_NAMESPACE
    raise ValueError(f"unsupported phase: {phase}")


def _normalize_record(
    raw: object, *, index: int, phase: str, tokenizer_hash: str
) -> dict[str, Any]:
    item = require_mapping(raw, context=f"records[{index}]")
    require_exact_fields(item, RECORD_FIELDS, context=f"records[{index}]")
    family = require_string(item["family"], context=f"records[{index}].family")
    if family not in DATASET_KEYS:
        raise ValueError(f"records[{index}].family is unknown")
    canonical_id = require_string(item["canonical_id"], context=f"records[{index}].canonical_id")
    config = require_string(item["config"], context=f"records[{index}].config", allow_empty=True)
    if family == "pg19":
        _validate_pg19_canonical_url(
            canonical_id,
            context=f"records[{index}].canonical_id",
        )
    elif family == "humaneval_plus":
        _validate_humaneval_plus_task_id(
            canonical_id,
            context=f"records[{index}].canonical_id",
        )
    rank = require_int(item["selection_rank"], context=f"records[{index}].selection_rank")
    seed_value = item["seed"]
    seed = None if seed_value is None else require_int(seed_value, context=f"records[{index}].seed")
    configured_value = item["configured_length"]
    configured_length = (
        None
        if configured_value is None
        else require_int(
            configured_value,
            context=f"records[{index}].configured_length",
            minimum=1,
        )
    )
    category_value = item["ruler_category"]
    ruler_category = (
        None
        if category_value is None
        else require_string(
            category_value,
            context=f"records[{index}].ruler_category",
        )
    )
    generator_receipt_value = item["generator_receipt_sha256"]
    generator_receipt_sha256 = (
        None
        if generator_receipt_value is None
        else require_sha256(
            generator_receipt_value,
            context=f"records[{index}].generator_receipt_sha256",
        )
    )
    if family == "ruler":
        expected_category = RULER_CONFIG_CATEGORY.get(config)
        if expected_category is None or ruler_category != expected_category:
            raise ValueError(f"records[{index}] RULER config/category binding drifted")
        if configured_length is None or generator_receipt_sha256 is None or seed is None:
            raise ValueError(
                f"records[{index}] RULER configured length, seed, and generator receipt "
                "are required"
            )
        expected_canonical_id = ruler_canonical_id(
            category=ruler_category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        if canonical_id != expected_canonical_id:
            raise ValueError(f"records[{index}] RULER canonical ID drifted")
    elif any(
        value is not None for value in (configured_length, ruler_category, generator_receipt_sha256)
    ):
        raise ValueError(f"records[{index}] non-RULER rows cannot carry RULER-only fields")
    elif config != FROZEN_DATASET_CONFIGS[family]:
        raise ValueError(
            f"records[{index}] {family} config must be {FROZEN_DATASET_CONFIGS[family]!r}"
        )
    sequence_length = require_int(
        item["sequence_length"],
        context=f"records[{index}].sequence_length",
        minimum=1,
    )
    namespace = _selection_namespace(phase, family)
    expected_selection = (
        mbpp_selection_sha256(canonical_id)
        if family == "mbpp"
        else selection_sha256(str(namespace), canonical_id)
    )
    if item["selection_sha256"] != expected_selection:
        raise ValueError(f"records[{index}] selection SHA-256 drifted")
    if item["tokenizer_manifest_sha256"] != tokenizer_hash:
        raise ValueError(f"records[{index}] tokenizer manifest binding drifted")
    span = require_mapping(item["token_span"], context=f"records[{index}].token_span")
    require_exact_fields(span, TOKEN_SPAN_FIELDS, context=f"records[{index}].token_span")
    normalized_span = {
        name: require_int(span[name], context=f"records[{index}].token_span.{name}")
        for name in (
            "prefill_start",
            "prefill_stop",
            "scored_start",
            "scored_stop",
            "cache_exposed_start",
            "cache_exposed_stop",
        )
    }
    if (
        normalized_span["prefill_start"] != 0
        or normalized_span["prefill_stop"] != normalized_span["scored_start"]
        or normalized_span["prefill_stop"] < 1
        or normalized_span["scored_stop"] < normalized_span["scored_start"]
        or normalized_span["scored_stop"] != sequence_length
    ):
        raise ValueError(f"records[{index}] token span is not contiguous and canonical")
    if phase == "calibration":
        if (
            normalized_span["cache_exposed_start"] != normalized_span["scored_stop"]
            or normalized_span["cache_exposed_stop"] != normalized_span["scored_stop"]
        ):
            raise ValueError(
                f"records[{index}] calibration cache-exposed prediction span "
                "must be empty at continuation stop"
            )
    else:
        continuation_tokens = normalized_span["scored_stop"] - normalized_span["scored_start"]
        if continuation_tokens < 2:
            raise ValueError(
                f"records[{index}] Stage-A continuation must contain at least two "
                "tokens to expose one cache prediction"
            )
        if (
            normalized_span["cache_exposed_start"] != normalized_span["scored_start"] + 1
            or normalized_span["cache_exposed_stop"] != normalized_span["scored_stop"]
        ):
            raise ValueError(
                f"records[{index}] Stage-A cache-exposed prediction span must exclude "
                "the first continuation token"
            )
    if configured_length is not None and sequence_length > configured_length:
        raise ValueError(f"records[{index}] actual sequence exceeds the RULER configured length")
    positions = anchor_positions(sequence_length)
    fisher_boundary = _normalize_fisher_boundary(
        item["fisher_boundary"],
        sequence_length=sequence_length,
        context=f"records[{index}].fisher_boundary",
    )
    sequence_hash = require_sha256(
        item["sequence_token_ids_sha256"],
        context=f"records[{index}].sequence_token_ids_sha256",
    )
    recorded_anchor_hash = require_sha256(
        item["anchor_manifest_sha256"],
        context=f"records[{index}].anchor_manifest_sha256",
    )
    computed_anchor_hash = identity_anchor_manifest_sha256(
        canonical_id=canonical_id,
        sequence_length=sequence_length,
        sequence_token_ids_sha256_value=sequence_hash,
        token_span=normalized_span,
    )
    if recorded_anchor_hash != computed_anchor_hash:
        raise ValueError(f"records[{index}] anchor manifest SHA-256 drifted")
    normalized = {
        "family": family,
        "canonical_id": canonical_id,
        "config": config,
        "selection_rank": rank,
        "selection_sha256": expected_selection,
        "seed": seed,
        "configured_length": configured_length,
        "sequence_length": sequence_length,
        "ruler_category": ruler_category,
        "generator_receipt_sha256": generator_receipt_sha256,
        "source_content_sha256": require_sha256(
            item["source_content_sha256"],
            context=f"records[{index}].source_content_sha256",
        ),
        "formatted_content_sha256": require_sha256(
            item["formatted_content_sha256"],
            context=f"records[{index}].formatted_content_sha256",
        ),
        "prompt_token_ids_sha256": require_sha256(
            item["prompt_token_ids_sha256"],
            context=f"records[{index}].prompt_token_ids_sha256",
        ),
        "target_token_ids_sha256": require_sha256(
            item["target_token_ids_sha256"],
            context=f"records[{index}].target_token_ids_sha256",
        ),
        "sequence_token_ids_sha256": sequence_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "token_span": normalized_span,
        "anchor_manifest_sha256": recorded_anchor_hash,
        "fisher_boundary": fisher_boundary,
        "identity_record_sha256": require_sha256(
            item["identity_record_sha256"],
            context=f"records[{index}].identity_record_sha256",
        ),
        "anchor_positions": list(positions),
        "anchor_positions_sha256": sha256_bytes(canonical_json_bytes(positions)),
    }
    computed_record_hash = identity_record_sha256(normalized)
    if normalized["identity_record_sha256"] != computed_record_hash:
        raise ValueError(f"records[{index}] identity record SHA-256 drifted")
    return normalized


def _record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        FAMILY_ORDER[str(record["family"])],
        int(record["selection_rank"]),
        str(record["selection_sha256"]),
        str(record["canonical_id"]),
        "" if record["ruler_category"] is None else str(record["ruler_category"]),
        str(record["config"]),
        -1 if record["seed"] is None else int(record["seed"]),
        -1 if record["configured_length"] is None else int(record["configured_length"]),
        int(record["sequence_length"]),
    )


def _validate_calibration_records(records: Sequence[Mapping[str, Any]]) -> None:
    grouped = {
        family: [record for record in records if record["family"] == family]
        for family in DATASET_KEYS
    }
    expected_counts = {"mbpp": 128, "pg19": 16, "ruler": 16, "humaneval_plus": 0}
    if {family: len(rows) for family, rows in grouped.items()} != expected_counts:
        raise ValueError("calibration record counts must be MBPP=128, PG19=16, RULER=16")
    expected_mbpp_ids, _population_hash = mbpp_calibration_identity()
    actual_mbpp = sorted(grouped["mbpp"], key=lambda row: int(row["selection_rank"]))
    if tuple(str(row["canonical_id"]) for row in actual_mbpp) != expected_mbpp_ids:
        raise ValueError("MBPP rows do not match the frozen v0.2 calibration identity")
    if sorted(int(row["selection_rank"]) for row in grouped["pg19"]) != list(range(16)):
        raise ValueError("calibration PG19 ranks must be exactly 0..15")
    _validate_sha_rank_order(grouped["pg19"], context="calibration PG19")
    for row in grouped["pg19"]:
        span = row["token_span"]
        if row["seed"] is not None or row["sequence_length"] != 2_304:
            raise ValueError("calibration PG19 must use one 2,304-token sequence per book")
        if span != {
            "prefill_start": 0,
            "prefill_stop": 2_304,
            "scored_start": 2_304,
            "scored_stop": 2_304,
            "cache_exposed_start": 2_304,
            "cache_exposed_stop": 2_304,
        }:
            raise ValueError("calibration PG19 span must cover exactly 2,304 tokens")
    if sorted(int(row["selection_rank"]) for row in grouped["ruler"]) != list(range(16)):
        raise ValueError("calibration RULER ranks must be exactly 0..15")
    _validate_sha_rank_order(grouped["ruler"], context="calibration RULER")
    actual_ruler_schedule = {
        (
            str(row["ruler_category"]),
            str(row["config"]),
            int(row["configured_length"]),
            int(row["seed"]),
        )
        for row in grouped["ruler"]
    }
    if actual_ruler_schedule != set(RULER_CALIBRATION_SCHEDULE):
        raise ValueError("calibration RULER rows differ from the frozen 16-sequence schedule")
    if {
        category: sum(row["ruler_category"] == category for row in grouped["ruler"])
        for category in RULER_CATEGORIES
    } != {category: 4 for category in RULER_CATEGORIES}:
        raise ValueError("calibration RULER must contain four sequences per category")
    for row in grouped["ruler"]:
        span = row["token_span"]
        if span != {
            "prefill_start": 0,
            "prefill_stop": row["sequence_length"],
            "scored_start": row["sequence_length"],
            "scored_stop": row["sequence_length"],
            "cache_exposed_start": row["sequence_length"],
            "cache_exposed_stop": row["sequence_length"],
        }:
            raise ValueError("calibration RULER must anchor the actual prompt tokens only")
    for row in grouped["mbpp"]:
        if row["seed"] is not None:
            raise ValueError("MBPP calibration records cannot have a generator seed")
    identities = [(row["family"], row["canonical_id"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("calibration canonical identities are not unique")


def _validate_stage_a_records(records: Sequence[Mapping[str, Any]]) -> None:
    grouped = {
        family: [record for record in records if record["family"] == family]
        for family in DATASET_KEYS
    }
    expected_counts = {"mbpp": 0, "pg19": 4, "ruler": 4, "humaneval_plus": 4}
    if {family: len(rows) for family, rows in grouped.items()} != expected_counts:
        raise ValueError("Stage A must contain exactly four PG19, RULER, and HumanEval+ rows")
    for family in ("pg19", "humaneval_plus"):
        if sorted(int(row["selection_rank"]) for row in grouped[family]) != list(range(4)):
            raise ValueError(f"Stage-A {family} ranks must be exactly 0..3")
        _validate_sha_rank_order(grouped[family], context=f"Stage-A {family}")
    for row in grouped["pg19"]:
        span = row["token_span"]
        if row["seed"] is not None or row["sequence_length"] != 4_224:
            raise ValueError("Stage-A PG19 must use 4,096 prefill plus 128 continuation tokens")
        if span != {
            "prefill_start": 0,
            "prefill_stop": 4_096,
            "scored_start": 4_096,
            "scored_stop": 4_224,
            "cache_exposed_start": 4_097,
            "cache_exposed_stop": 4_224,
        }:
            raise ValueError(
                "Stage-A PG19 must bind 128 continuation tokens and exactly "
                "127 cache-exposed predictions"
            )
    if sorted(int(row["selection_rank"]) for row in grouped["ruler"]) != list(range(4)):
        raise ValueError("Stage-A RULER ranks must be exactly 0..3")
    _validate_sha_rank_order(grouped["ruler"], context="Stage-A RULER")
    actual_ruler_schedule = {
        (
            str(row["ruler_category"]),
            str(row["config"]),
            int(row["configured_length"]),
            int(row["seed"]),
        )
        for row in grouped["ruler"]
    }
    if actual_ruler_schedule != set(RULER_STAGE_A_SCHEDULE):
        raise ValueError("Stage-A RULER rows differ from the frozen category representatives")
    for row in grouped["ruler"]:
        span = row["token_span"]
        continuation = span["scored_stop"] - span["scored_start"]
        exposed = span["cache_exposed_stop"] - span["cache_exposed_start"]
        if continuation < 2 or exposed != continuation - 1:
            raise ValueError(
                "Stage-A RULER continuation must expose target_length - 1 cache predictions"
            )
    for row in grouped["humaneval_plus"]:
        span = row["token_span"]
        continuation = span["scored_stop"] - span["scored_start"]
        exposed = span["cache_exposed_stop"] - span["cache_exposed_start"]
        if row["seed"] is not None or not 2 <= continuation <= 128 or exposed != continuation - 1:
            raise ValueError(
                "Stage-A HumanEval+ must bind 2..128 continuation tokens and expose "
                "target_length - 1 cache predictions"
            )
        if row["sequence_length"] != span["scored_stop"]:
            raise ValueError("Stage-A HumanEval+ sequence length must equal span stop")
    identities = [(row["family"], row["canonical_id"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage-A canonical identities are not unique")


def _split_half_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    groups.append(("mbpp", [row for row in records if row["family"] == "mbpp"]))
    groups.append(("pg19", [row for row in records if row["family"] == "pg19"]))
    ruler_rows = [row for row in records if row["family"] == "ruler"]
    for category in RULER_CATEGORIES:
        groups.append(
            (
                f"ruler:{category}",
                [row for row in ruler_rows if row["ruler_category"] == category],
            )
        )
    for group, rows in groups:
        ranked = sorted(rows, key=lambda row: (calibration_split_key(row), _record_sort_key(row)))
        for rank, row in enumerate(ranked):
            assignments.append(
                {
                    "group": group,
                    "canonical_id": row["canonical_id"],
                    "ruler_category": row["ruler_category"],
                    "config": row["config"],
                    "configured_length": row["configured_length"],
                    "sequence_length": row["sequence_length"],
                    "seed": row["seed"],
                    "rank": rank,
                    "half": "a" if rank % 2 == 0 else "b",
                    "rank_sha256": calibration_split_key(row),
                }
            )
    assignments.sort(key=lambda item: (item["group"], item["rank"]))
    return {
        "namespace": CALIBRATION_SPLIT_NAMESPACE,
        "assignment": assignments,
        "assignment_sha256": sha256_bytes(canonical_json_bytes(assignments)),
    }


def _validate_calibration_binding(value: object) -> dict[str, str]:
    binding = require_mapping(value, context="calibration_binding")
    require_exact_fields(binding, CALIBRATION_BINDING_FIELDS, context="calibration_binding")
    return {
        key: require_sha256(binding[key], context=f"calibration_binding.{key}")
        for key in sorted(CALIBRATION_BINDING_FIELDS)
    }


def _validate_execution_bindings(value: object) -> dict[str, str]:
    bindings = require_mapping(value, context="execution_bindings")
    require_exact_fields(
        bindings,
        EXECUTION_BINDING_FIELDS,
        context="execution_bindings",
    )
    return {
        key: require_sha256(bindings[key], context=f"execution_bindings.{key}")
        for key in sorted(EXECUTION_BINDING_FIELDS)
    }


@dataclass(frozen=True, slots=True)
class _ValidatedIdentityInput:
    phase: str
    source_hash: str
    datasets: tuple[dict[str, Any], ...]
    tokenizer: dict[str, Any]
    execution_bindings: dict[str, str]
    records: tuple[dict[str, Any], ...]
    split_half: dict[str, Any] | None
    calibration_binding: dict[str, str] | None


def _validate_identity_input_source(
    source: Mapping[str, Any],
    *,
    expected_revisions: Mapping[str, str],
    calibration_binding_artifact: bytes | None,
    expected_calibration_binding_file_sha256: str | None = None,
) -> _ValidatedIdentityInput:
    """Validate one raw identity input without relying on finalized custody."""

    identity_input = require_mapping(source, context="identity input")
    phase = identity_input.get("phase")
    expected_fields = {
        "schema",
        "phase",
        "datasets",
        "tokenizer",
        "records",
        "execution_bindings",
        "model_weights_loaded",
    }
    if phase == "stage_a":
        expected_fields.add("calibration_binding")
    require_exact_fields(identity_input, frozenset(expected_fields), context="identity input")
    if identity_input["schema"] != INPUT_SCHEMA:
        raise ValueError("identity input schema drifted")
    if phase not in ALLOWED_PHASES:
        if phase in PROTECTED_STAGES:
            raise PermissionError(
                f"{phase} is protected and unavailable in resolver procedure v{RESOLVER_VERSION}"
            )
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if identity_input["model_weights_loaded"] is not False:
        raise ValueError("identity resolution must occur before model weights")

    verified_binding: StageACalibrationBindingArtifact | None = None
    if phase == "stage_a":
        if not isinstance(calibration_binding_artifact, bytes):
            raise ValueError("Stage A requires a verified calibration binding artifact")
        verified_binding = deserialize_stage_a_calibration_binding_artifact(
            calibration_binding_artifact,
            expected_file_sha256=expected_calibration_binding_file_sha256,
        )
    elif (
        calibration_binding_artifact is not None
        or expected_calibration_binding_file_sha256 is not None
    ):
        raise ValueError("calibration resolution forbids Stage-A authorization artifacts")

    if set(expected_revisions) != set(DATASET_KEYS):
        raise ValueError("all four dataset revisions are mandatory")
    revisions = {
        key: require_exact_revision(value, context=f"CLI {key} revision")
        for key, value in expected_revisions.items()
    }
    if revisions != FROZEN_DATASET_REVISIONS:
        raise ValueError("CLI dataset revisions do not match the frozen upstream commits")
    datasets = _validate_dataset_contracts(
        identity_input["datasets"],
        expected_revisions=revisions,
        phase=str(phase),
    )
    tokenizer = _validate_tokenizer(identity_input["tokenizer"])
    execution_bindings = _validate_execution_bindings(identity_input["execution_bindings"])
    if (
        execution_bindings["parquet_materialization_manifest_file_sha256"]
        != PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Parquet materialization manifest file SHA-256 drifted")
    if verified_binding is not None and execution_bindings != dict(
        verified_binding.execution_bindings
    ):
        raise ValueError("Stage-A execution bindings differ from calibration authorization")

    raw_records = require_sequence(identity_input["records"], context="records")
    records = [
        _normalize_record(
            raw,
            index=index,
            phase=str(phase),
            tokenizer_hash=tokenizer["file_manifest_sha256"],
        )
        for index, raw in enumerate(raw_records)
    ]
    records.sort(key=_record_sort_key)
    if phase == "calibration":
        _validate_calibration_records(records)
        split_half = _split_half_manifest(records)
        calibration_binding = None
    else:
        assert verified_binding is not None
        _validate_stage_a_records(records)
        split_half = None
        calibration_binding = _validate_calibration_binding(identity_input["calibration_binding"])
        if calibration_binding != dict(verified_binding.binding):
            raise ValueError("Stage-A input calibration binding differs from the verified artifact")

    return _ValidatedIdentityInput(
        phase=str(phase),
        source_hash=sha256_bytes(canonical_json_bytes(identity_input)),
        datasets=tuple(datasets),
        tokenizer=tokenizer,
        execution_bindings=execution_bindings,
        records=tuple(records),
        split_half=split_half,
        calibration_binding=calibration_binding,
    )


def validate_stage_a_identity_input_for_capture(
    source: Mapping[str, Any],
    *,
    calibration_binding_artifact: bytes,
    expected_calibration_binding_file_sha256: str,
) -> None:
    """Authenticate and validate raw Stage-A input before receipt finalization.

    This gate intentionally has no capture-receipt parameter: a receipt does not
    exist until the sealed runner has validated and published the identity input.
    """

    validated = _validate_identity_input_source(
        source,
        expected_revisions=FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=calibration_binding_artifact,
        expected_calibration_binding_file_sha256=require_sha256(
            expected_calibration_binding_file_sha256,
            context="expected Stage-A binding file SHA-256",
        ),
    )
    if validated.phase != "stage_a":
        raise ValueError("pre-finalization validation requires a Stage-A identity input")


def build_candidate(
    source: Mapping[str, Any],
    *,
    expected_revisions: Mapping[str, str],
    calibration_binding_artifact: bytes | None = None,
    stage_a_capture_provenance_receipt: bytes | None = None,
    expected_stage_a_capture_provenance_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate metadata and return a deterministic candidate artifact."""

    validated = _validate_identity_input_source(
        source,
        expected_revisions=expected_revisions,
        calibration_binding_artifact=calibration_binding_artifact,
    )
    phase = validated.phase
    stage_a_capture_provenance_receipt_file_sha256: str | None = None
    if phase == "stage_a":
        assert isinstance(calibration_binding_artifact, bytes)
        if not isinstance(stage_a_capture_provenance_receipt, bytes):
            raise ValueError("Stage A requires a finalized capture provenance receipt")
        if expected_stage_a_capture_provenance_receipt_sha256 is None:
            raise ValueError("Stage A requires an explicit capture provenance receipt SHA-256")
        verified_capture = deserialize_stage_a_capture_provenance_receipt(
            stage_a_capture_provenance_receipt,
            expected_file_sha256=expected_stage_a_capture_provenance_receipt_sha256,
            calibration_binding_artifact=calibration_binding_artifact,
            expected_identity_input_file_sha256=validated.source_hash,
        )
        stage_a_capture_provenance_receipt_file_sha256 = verified_capture.file_sha256
    elif (
        stage_a_capture_provenance_receipt is not None
        or expected_stage_a_capture_provenance_receipt_sha256 is not None
    ):
        raise ValueError("calibration resolution forbids Stage-A authorization artifacts")
    records = list(validated.records)
    content_manifest_hash = sha256_bytes(canonical_json_bytes(records))
    evidence: dict[str, Any] = {
        "schema_version": STAGE_A_IDENTITY_SCHEMA_VERSION if phase == "stage_a" else 5,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": STAGE_A_CANDIDATE_SCHEMA if phase == "stage_a" else CANDIDATE_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "candidate",
        "phase": phase,
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_manifest_sha256": validated.source_hash,
        "execution_bindings": validated.execution_bindings,
        "model_contracts": {
            "primary": {"id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
            "conditional_scale_check": {
                "id": CONDITIONAL_MODEL_ID,
                "revision": CONDITIONAL_MODEL_REVISION,
                "cold_start_peak_hbm_limit_bytes": 8_053_063_680,
            },
            "weights_loaded": False,
        },
        "datasets": list(validated.datasets),
        "upstream_tool_contracts": {
            "ruler_generator": {
                "id": RULER_SOURCE_ID,
                "revision": RULER_REVISION,
            },
            "evalplus_formatter": {
                "id": EVALPLUS_SOURCE_ID,
                "revision": EVALPLUS_SOURCE_REVISION,
            },
        },
        "tokenizer": validated.tokenizer,
        "records": records,
        "record_count": len(records),
        "content_manifest_sha256": content_manifest_hash,
        "selection": {
            "pg19_train_namespace": PG19_TRAIN_NAMESPACE,
            "pg19_validation_namespace": PG19_VALIDATION_NAMESPACE,
            "pg19_test_namespace": PG19_TEST_NAMESPACE,
            "humaneval_plus_stage_a_b_namespace": HUMANEVAL_AB_NAMESPACE,
            "humaneval_plus_stage_c_namespace": HUMANEVAL_C_NAMESPACE,
        },
        "calibration_split_half": validated.split_half,
        "calibration_binding": validated.calibration_binding,
        "protected_identity": {
            "stage_b_read": False,
            "stage_c_read": False,
            "ordinary_tests_may_read_protected_content": False,
        },
        "promotion_required": True,
    }
    if phase == "stage_a":
        assert stage_a_capture_provenance_receipt_file_sha256 is not None
        evidence[STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD] = (
            stage_a_capture_provenance_receipt_file_sha256
        )
    artifact = {
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    validate_candidate_artifact(artifact)
    return artifact


def validate_candidate_artifact(artifact: Mapping[str, Any]) -> None:
    if set(artifact) != {"canonical_evidence_sha256", "evidence"}:
        raise ValueError("candidate wrapper fields drifted")
    evidence = require_mapping(artifact.get("evidence"), context="candidate evidence")
    phase = evidence.get("phase")
    require_exact_fields(
        evidence,
        (STAGE_A_CANDIDATE_EVIDENCE_FIELDS if phase == "stage_a" else CANDIDATE_EVIDENCE_FIELDS),
        context="candidate evidence",
    )
    if artifact.get("canonical_evidence_sha256") != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("candidate canonical evidence SHA-256 drifted")
    exact_scalars = {
        "schema_version": STAGE_A_IDENTITY_SCHEMA_VERSION if phase == "stage_a" else 5,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": STAGE_A_CANDIDATE_SCHEMA if phase == "stage_a" else CANDIDATE_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "candidate",
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "promotion_required": True,
    }
    for name, expected in exact_scalars.items():
        if (
            isinstance(expected, (bool, int)) and type(evidence[name]) is not type(expected)
        ) or evidence[name] != expected:
            raise ValueError(f"candidate {name} drifted")
    if phase not in ALLOWED_PHASES:
        raise ValueError("candidate phase drifted")
    if phase == "stage_a":
        require_sha256(
            evidence[STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD],
            context="Stage-A capture provenance receipt file SHA-256",
        )
    require_sha256(
        evidence["source_manifest_sha256"],
        context="candidate source manifest SHA-256",
    )
    execution_bindings = _validate_execution_bindings(evidence["execution_bindings"])
    if dict(evidence["execution_bindings"]) != execution_bindings:
        raise ValueError("candidate execution bindings are not canonical")
    if (
        execution_bindings["parquet_materialization_manifest_file_sha256"]
        != PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    ):
        raise ValueError("candidate Parquet materialization manifest file SHA-256 drifted")

    models = require_mapping(evidence["model_contracts"], context="model contracts")
    require_exact_fields(
        models,
        frozenset({"primary", "conditional_scale_check", "weights_loaded"}),
        context="model contracts",
    )
    primary = require_mapping(models["primary"], context="primary model contract")
    require_exact_fields(
        primary,
        frozenset({"id", "revision"}),
        context="primary model contract",
    )
    conditional = require_mapping(
        models["conditional_scale_check"],
        context="conditional model contract",
    )
    require_exact_fields(
        conditional,
        frozenset({"id", "revision", "cold_start_peak_hbm_limit_bytes"}),
        context="conditional model contract",
    )
    if dict(primary) != {"id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION}:
        raise ValueError("candidate primary model contract drifted")
    if dict(conditional) != {
        "id": CONDITIONAL_MODEL_ID,
        "revision": CONDITIONAL_MODEL_REVISION,
        "cold_start_peak_hbm_limit_bytes": 8_053_063_680,
    }:
        raise ValueError("candidate conditional model contract drifted")
    if models["weights_loaded"] is not False:
        raise ValueError("candidate claims model weights were loaded")

    datasets = _validate_dataset_contracts(
        evidence["datasets"],
        expected_revisions=FROZEN_DATASET_REVISIONS,
        phase=str(phase),
    )
    if list(datasets) != evidence["datasets"]:
        raise ValueError("candidate dataset contracts are not canonical")
    tokenizer = require_mapping(evidence["tokenizer"], context="candidate tokenizer")
    require_exact_fields(
        tokenizer,
        TOKENIZER_FIELDS | {"file_manifest_sha256"},
        context="candidate tokenizer",
    )
    normalized_tokenizer = _validate_tokenizer({name: tokenizer[name] for name in TOKENIZER_FIELDS})
    if dict(tokenizer) != normalized_tokenizer:
        raise ValueError("candidate tokenizer contract or file manifest drifted")
    tokenizer_manifest_sha256 = normalized_tokenizer["file_manifest_sha256"]

    raw_records = require_sequence(evidence["records"], context="candidate records")
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        record = require_mapping(raw_record, context=f"candidate records[{index}]")
        require_exact_fields(
            record,
            FROZEN_RECORD_FIELDS,
            context=f"candidate records[{index}]",
        )
        normalized = _normalize_record(
            {name: record[name] for name in RECORD_FIELDS},
            index=index,
            phase=str(phase),
            tokenizer_hash=tokenizer_manifest_sha256,
        )
        if dict(record) != normalized:
            raise ValueError(f"candidate records[{index}] is not canonical")
        records.append(normalized)
    if records != sorted(records, key=_record_sort_key):
        raise ValueError("candidate records are not in canonical resolver order")
    if phase == "calibration":
        _validate_calibration_records(records)
        expected_record_count = 160
    else:
        _validate_stage_a_records(records)
        expected_record_count = 12
    if evidence["record_count"] != len(records) or len(records) != expected_record_count:
        raise ValueError("candidate record count drifted")
    if evidence["content_manifest_sha256"] != sha256_bytes(canonical_json_bytes(records)):
        raise ValueError("candidate content manifest SHA-256 drifted")

    expected_selection = {
        "pg19_train_namespace": PG19_TRAIN_NAMESPACE,
        "pg19_validation_namespace": PG19_VALIDATION_NAMESPACE,
        "pg19_test_namespace": PG19_TEST_NAMESPACE,
        "humaneval_plus_stage_a_b_namespace": HUMANEVAL_AB_NAMESPACE,
        "humaneval_plus_stage_c_namespace": HUMANEVAL_C_NAMESPACE,
    }
    if evidence["selection"] != expected_selection:
        raise ValueError("candidate selection namespaces drifted")
    expected_upstream = {
        "ruler_generator": {"id": RULER_SOURCE_ID, "revision": RULER_REVISION},
        "evalplus_formatter": {
            "id": EVALPLUS_SOURCE_ID,
            "revision": EVALPLUS_SOURCE_REVISION,
        },
    }
    if evidence["upstream_tool_contracts"] != expected_upstream:
        raise ValueError("candidate upstream tool contracts drifted")
    if phase == "calibration":
        if evidence["calibration_binding"] is not None:
            raise ValueError("calibration candidate cannot carry a Stage-A binding")
        if evidence["calibration_split_half"] != _split_half_manifest(records):
            raise ValueError("candidate calibration split assignments drifted")
    else:
        if evidence["calibration_split_half"] is not None:
            raise ValueError("Stage-A candidate cannot carry a calibration split")
        binding = _validate_calibration_binding(evidence["calibration_binding"])
        if dict(evidence["calibration_binding"]) != binding:
            raise ValueError("candidate calibration binding is not canonical")

    protected = require_mapping(evidence["protected_identity"], context="protected identity")
    protected_fields = frozenset(
        {
            "stage_b_read",
            "stage_c_read",
            "ordinary_tests_may_read_protected_content",
        }
    )
    require_exact_fields(protected, protected_fields, context="protected identity")
    if any(protected[field] is not False for field in protected_fields):
        raise ValueError("protected identity boundary drifted")


def promote_candidate(
    candidate: Mapping[str, Any],
    *,
    candidate_file_sha256: str,
    calibration_binding_artifact: bytes | None = None,
    stage_a_capture_provenance_receipt: bytes | None = None,
    expected_stage_a_capture_provenance_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic frozen identity from an authenticated candidate."""

    validate_candidate_artifact(candidate)
    expected_candidate_file_sha256 = require_sha256(
        candidate_file_sha256,
        context="candidate file SHA-256",
    )
    if expected_candidate_file_sha256 != sha256_bytes(canonical_json_bytes(candidate)):
        raise ValueError("candidate file bytes are not canonical resolver JSON")
    candidate_phase = candidate["evidence"]["phase"]
    if candidate_phase == "stage_a":
        if not isinstance(calibration_binding_artifact, bytes):
            raise ValueError("Stage-A promotion requires a verified calibration binding artifact")
        verified_binding_artifact = deserialize_stage_a_calibration_binding_artifact(
            calibration_binding_artifact
        )
        if candidate["evidence"]["calibration_binding"] != verified_binding_artifact.binding:
            raise ValueError("Stage-A candidate differs from the verified calibration binding")
        if candidate["evidence"]["execution_bindings"] != dict(
            verified_binding_artifact.execution_bindings
        ):
            raise ValueError(
                "Stage-A candidate execution bindings differ from calibration authorization"
            )
        if not isinstance(stage_a_capture_provenance_receipt, bytes):
            raise ValueError("Stage-A promotion requires finalized capture provenance")
        if expected_stage_a_capture_provenance_receipt_sha256 is None:
            raise ValueError("Stage-A promotion requires an explicit provenance receipt SHA-256")
        verified_capture = deserialize_stage_a_capture_provenance_receipt(
            stage_a_capture_provenance_receipt,
            expected_file_sha256=expected_stage_a_capture_provenance_receipt_sha256,
            calibration_binding_artifact=calibration_binding_artifact,
            expected_identity_input_file_sha256=candidate["evidence"]["source_manifest_sha256"],
        )
        if candidate["evidence"][STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD] != (
            verified_capture.file_sha256
        ):
            raise ValueError("Stage-A candidate binds a different capture provenance receipt")
    elif (
        calibration_binding_artifact is not None
        or stage_a_capture_provenance_receipt is not None
        or expected_stage_a_capture_provenance_receipt_sha256 is not None
    ):
        raise ValueError("calibration promotion forbids Stage-A authorization artifacts")
    evidence = deepcopy(dict(candidate["evidence"]))
    evidence["identity_schema"] = (
        STAGE_A_FROZEN_SCHEMA if candidate_phase == "stage_a" else FROZEN_SCHEMA
    )
    evidence["status"] = "frozen"
    evidence["promotion_required"] = False
    evidence["promotion"] = {
        "candidate_file_sha256": expected_candidate_file_sha256,
        "candidate_canonical_evidence_sha256": candidate["canonical_evidence_sha256"],
        "explicit": True,
    }
    if candidate_phase == "stage_a":
        evidence["promotion"][STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD] = evidence[
            STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD
        ]
    return {
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }


@dataclass(frozen=True, slots=True)
class FrozenCalibrationIdentityArtifact:
    """Strictly verified frozen calibration identity and its binding commitments."""

    file_sha256: str
    canonical_evidence_sha256: str
    records: tuple[Mapping[str, Any], ...]
    assignment: tuple[Mapping[str, Any], ...]
    assignment_sha256: str
    tokenizer_manifest_sha256: str
    parquet_materialization_manifest_file_sha256: str
    execution_bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(_deep_freeze(record) for record in self.records),
        )
        object.__setattr__(
            self,
            "assignment",
            tuple(_deep_freeze(item) for item in self.assignment),
        )
        object.__setattr__(self, "execution_bindings", _deep_freeze(self.execution_bindings))


@dataclass(frozen=True, slots=True)
class FrozenStageAIdentityArtifact:
    """Strictly verified frozen Stage-A identity and authorized calibration binding."""

    file_sha256: str
    canonical_evidence_sha256: str
    records: tuple[Mapping[str, Any], ...]
    tokenizer_manifest_sha256: str
    calibration_binding: Mapping[str, str]
    parquet_materialization_manifest_file_sha256: str
    execution_bindings: Mapping[str, str]
    stage_a_capture_provenance_receipt_file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(_deep_freeze(record) for record in self.records),
        )
        object.__setattr__(self, "calibration_binding", _deep_freeze(self.calibration_binding))
        object.__setattr__(self, "execution_bindings", _deep_freeze(self.execution_bindings))


@dataclass(frozen=True, slots=True)
class StageACaptureProvenanceReceipt:
    """Strictly verified finalized custody receipt for one Stage-A identity input."""

    file_sha256: str
    identity_input_file_sha256: str
    calibration_binding_file_sha256: str
    calibration_authorization_file_sha256: str
    source_commit: str
    execution_bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_bindings", _deep_freeze(self.execution_bindings))


@dataclass(frozen=True, slots=True)
class StageACalibrationCoreBindingArtifact:
    """Verified pre-authorization binding emitted by the full calibration run."""

    binding: Mapping[str, str]
    dependency_file_sha256: Mapping[str, str]
    calibration_dependencies: Mapping[str, bytes]
    canonical_evidence_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _deep_freeze(self.binding))
        object.__setattr__(
            self,
            "dependency_file_sha256",
            _deep_freeze(self.dependency_file_sha256),
        )
        object.__setattr__(
            self,
            "calibration_dependencies",
            _deep_freeze(self.calibration_dependencies),
        )


@dataclass(frozen=True, slots=True)
class StageACalibrationBindingArtifact:
    """Verified post-calibration Stage-A binding and authenticated calibration bytes."""

    binding: Mapping[str, str]
    dependency_file_sha256: Mapping[str, str]
    calibration_dependencies: Mapping[str, bytes]
    authorization_dependencies: Mapping[str, bytes]
    authorization_file_sha256: str
    execution_bindings: Mapping[str, str]
    source_commit: str
    canonical_evidence_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _deep_freeze(self.binding))
        object.__setattr__(
            self,
            "dependency_file_sha256",
            _deep_freeze(self.dependency_file_sha256),
        )
        object.__setattr__(
            self,
            "calibration_dependencies",
            _deep_freeze(self.calibration_dependencies),
        )
        object.__setattr__(
            self,
            "authorization_dependencies",
            _deep_freeze(self.authorization_dependencies),
        )
        object.__setattr__(self, "execution_bindings", _deep_freeze(self.execution_bindings))


@dataclass(frozen=True, slots=True)
class StageACalibrationAuthorizationArtifact:
    """Verified authorization over one finalized runner-v10 calibration chain."""

    binding: Mapping[str, str]
    calibration_dependencies: Mapping[str, bytes]
    authorization_dependencies: Mapping[str, bytes]
    authorized_output_file_sha256: Mapping[str, str]
    execution_bindings: Mapping[str, str]
    source_commit: str
    identity_input_manifest_sha256: str
    canonical_evidence_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _deep_freeze(self.binding))
        object.__setattr__(
            self,
            "calibration_dependencies",
            _deep_freeze(self.calibration_dependencies),
        )
        object.__setattr__(
            self,
            "authorization_dependencies",
            _deep_freeze(self.authorization_dependencies),
        )
        object.__setattr__(
            self,
            "authorized_output_file_sha256",
            _deep_freeze(self.authorized_output_file_sha256),
        )
        object.__setattr__(self, "execution_bindings", _deep_freeze(self.execution_bindings))


def deserialize_frozen_calibration_identity_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> FrozenCalibrationIdentityArtifact:
    """Decode and independently recompute the complete frozen calibration identity."""

    if not isinstance(data, bytes):
        raise TypeError("frozen identity artifact must be bytes")
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None:
        expected = require_sha256(
            expected_file_sha256,
            context="expected frozen identity file SHA-256",
        )
        if file_sha256 != expected:
            raise ValueError("frozen identity file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="frozen identity artifact")
    require_exact_fields(
        root,
        frozenset({"canonical_evidence_sha256", "evidence"}),
        context="frozen identity wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("frozen identity bytes are not canonical resolver JSON")
    evidence = require_mapping(root["evidence"], context="frozen identity evidence")
    require_exact_fields(
        evidence,
        FROZEN_EVIDENCE_FIELDS,
        context="frozen identity evidence",
    )
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="frozen identity canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("frozen identity canonical evidence SHA-256 drifted")
    exact_scalars = {
        "schema_version": 5,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": FROZEN_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "frozen",
        "phase": "calibration",
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "calibration_binding": None,
        "promotion_required": False,
    }
    for name, expected in exact_scalars.items():
        if (
            isinstance(expected, (bool, int)) and type(evidence[name]) is not type(expected)
        ) or evidence[name] != expected:
            raise ValueError(f"frozen identity {name} drifted")
    require_sha256(
        evidence["source_manifest_sha256"],
        context="frozen identity source manifest SHA-256",
    )
    execution_bindings = _validate_execution_bindings(evidence["execution_bindings"])
    if dict(evidence["execution_bindings"]) != execution_bindings:
        raise ValueError("frozen execution bindings are not canonical")
    parquet_manifest_sha256 = execution_bindings["parquet_materialization_manifest_file_sha256"]
    if parquet_manifest_sha256 != PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256:
        raise ValueError("frozen Parquet materialization manifest file SHA-256 drifted")

    models = require_mapping(evidence["model_contracts"], context="model contracts")
    require_exact_fields(
        models,
        frozenset({"primary", "conditional_scale_check", "weights_loaded"}),
        context="model contracts",
    )
    primary = require_mapping(models["primary"], context="primary model contract")
    require_exact_fields(
        primary,
        frozenset({"id", "revision"}),
        context="primary model contract",
    )
    conditional = require_mapping(
        models["conditional_scale_check"],
        context="conditional model contract",
    )
    require_exact_fields(
        conditional,
        frozenset({"id", "revision", "cold_start_peak_hbm_limit_bytes"}),
        context="conditional model contract",
    )
    if primary != {"id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION}:
        raise ValueError("primary model contract drifted")
    if conditional != {
        "id": CONDITIONAL_MODEL_ID,
        "revision": CONDITIONAL_MODEL_REVISION,
        "cold_start_peak_hbm_limit_bytes": 8_053_063_680,
    }:
        raise ValueError("conditional model contract drifted")
    if models["weights_loaded"] is not False:
        raise ValueError("frozen calibration identity claims model weights were loaded")

    datasets = _validate_dataset_contracts(
        evidence["datasets"],
        expected_revisions=FROZEN_DATASET_REVISIONS,
        phase="calibration",
    )
    if list(datasets) != evidence["datasets"]:
        raise ValueError("frozen dataset contracts are not canonical")
    tokenizer = require_mapping(evidence["tokenizer"], context="frozen tokenizer")
    require_exact_fields(
        tokenizer,
        TOKENIZER_FIELDS | {"file_manifest_sha256"},
        context="frozen tokenizer",
    )
    tokenizer_without_manifest = {name: tokenizer[name] for name in TOKENIZER_FIELDS}
    normalized_tokenizer = _validate_tokenizer(tokenizer_without_manifest)
    if dict(tokenizer) != normalized_tokenizer:
        raise ValueError("frozen tokenizer contract or file manifest drifted")
    tokenizer_manifest_sha256 = normalized_tokenizer["file_manifest_sha256"]

    raw_records = require_sequence(evidence["records"], context="frozen records")
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        record = require_mapping(raw_record, context=f"frozen records[{index}]")
        require_exact_fields(
            record,
            FROZEN_RECORD_FIELDS,
            context=f"frozen records[{index}]",
        )
        capture_record = {name: record[name] for name in RECORD_FIELDS}
        normalized = _normalize_record(
            capture_record,
            index=index,
            phase="calibration",
            tokenizer_hash=tokenizer_manifest_sha256,
        )
        if dict(record) != normalized:
            raise ValueError(f"frozen records[{index}] is not canonical")
        records.append(normalized)
    if records != sorted(records, key=_record_sort_key):
        raise ValueError("frozen calibration records are not in canonical resolver order")
    _validate_calibration_records(records)
    if evidence["record_count"] != len(records) or len(records) != 160:
        raise ValueError("frozen calibration record count drifted")
    content_manifest_sha256 = require_sha256(
        evidence["content_manifest_sha256"],
        context="frozen content manifest SHA-256",
    )
    if content_manifest_sha256 != sha256_bytes(canonical_json_bytes(records)):
        raise ValueError("frozen content manifest SHA-256 drifted")

    expected_selection = {
        "pg19_train_namespace": PG19_TRAIN_NAMESPACE,
        "pg19_validation_namespace": PG19_VALIDATION_NAMESPACE,
        "pg19_test_namespace": PG19_TEST_NAMESPACE,
        "humaneval_plus_stage_a_b_namespace": HUMANEVAL_AB_NAMESPACE,
        "humaneval_plus_stage_c_namespace": HUMANEVAL_C_NAMESPACE,
    }
    if evidence["selection"] != expected_selection:
        raise ValueError("frozen selection namespaces drifted")
    expected_upstream = {
        "ruler_generator": {"id": RULER_SOURCE_ID, "revision": RULER_REVISION},
        "evalplus_formatter": {
            "id": EVALPLUS_SOURCE_ID,
            "revision": EVALPLUS_SOURCE_REVISION,
        },
    }
    if evidence["upstream_tool_contracts"] != expected_upstream:
        raise ValueError("frozen upstream tool contracts drifted")
    protected = require_mapping(
        evidence["protected_identity"],
        context="frozen protected identity",
    )
    protected_fields = frozenset(
        {
            "stage_b_read",
            "stage_c_read",
            "ordinary_tests_may_read_protected_content",
        }
    )
    require_exact_fields(
        protected,
        protected_fields,
        context="frozen protected identity",
    )
    if any(protected[field] is not False for field in protected_fields):
        raise ValueError("frozen protected identity boundary drifted")

    split = require_mapping(
        evidence["calibration_split_half"],
        context="frozen calibration split",
    )
    expected_split = _split_half_manifest(records)
    if dict(split) != expected_split:
        raise ValueError("frozen calibration split assignments drifted")

    promotion = require_mapping(evidence["promotion"], context="frozen promotion")
    require_exact_fields(
        promotion,
        frozenset(
            {
                "candidate_file_sha256",
                "candidate_canonical_evidence_sha256",
                "explicit",
            }
        ),
        context="frozen promotion",
    )
    if promotion["explicit"] is not True:
        raise ValueError("frozen identity was not explicitly promoted")
    candidate_evidence = deepcopy(dict(evidence))
    candidate_evidence.pop("promotion")
    candidate_evidence["identity_schema"] = CANDIDATE_SCHEMA
    candidate_evidence["status"] = "candidate"
    candidate_evidence["promotion_required"] = True
    candidate_canonical_sha256 = sha256_bytes(canonical_json_bytes(candidate_evidence))
    if (
        require_sha256(
            promotion["candidate_canonical_evidence_sha256"],
            context="promoted candidate canonical evidence SHA-256",
        )
        != candidate_canonical_sha256
    ):
        raise ValueError("promoted candidate canonical evidence SHA-256 drifted")
    candidate_document = {
        "canonical_evidence_sha256": candidate_canonical_sha256,
        "evidence": candidate_evidence,
    }
    candidate_file_sha256 = sha256_bytes(canonical_json_bytes(candidate_document))
    if (
        require_sha256(
            promotion["candidate_file_sha256"],
            context="promoted candidate file SHA-256",
        )
        != candidate_file_sha256
    ):
        raise ValueError("promoted candidate file SHA-256 drifted")

    assignments = tuple(
        dict(item)
        for item in require_sequence(
            expected_split["assignment"],
            context="frozen split assignment",
        )
    )
    return FrozenCalibrationIdentityArtifact(
        file_sha256=file_sha256,
        canonical_evidence_sha256=canonical_evidence_sha256,
        records=tuple(records),
        assignment=assignments,
        assignment_sha256=str(expected_split["assignment_sha256"]),
        tokenizer_manifest_sha256=str(tokenizer_manifest_sha256),
        parquet_materialization_manifest_file_sha256=parquet_manifest_sha256,
        execution_bindings=execution_bindings,
    )


def deserialize_frozen_stage_a_identity_artifact(
    data: bytes,
    *,
    calibration_binding_artifact: bytes,
    stage_a_capture_provenance_receipt: bytes,
    expected_stage_a_capture_provenance_receipt_sha256: str,
    expected_file_sha256: str | None = None,
) -> FrozenStageAIdentityArtifact:
    """Decode Stage A and reauthenticate capture, promotion, and calibration custody."""

    if not isinstance(data, bytes):
        raise TypeError("frozen Stage-A identity artifact must be bytes")
    if not isinstance(calibration_binding_artifact, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    if not isinstance(stage_a_capture_provenance_receipt, bytes):
        raise TypeError("Stage-A capture provenance receipt must be bytes")
    verified_binding_artifact = deserialize_stage_a_calibration_binding_artifact(
        calibration_binding_artifact
    )
    verified_capture = deserialize_stage_a_capture_provenance_receipt(
        stage_a_capture_provenance_receipt,
        expected_file_sha256=expected_stage_a_capture_provenance_receipt_sha256,
        calibration_binding_artifact=calibration_binding_artifact,
    )
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None:
        expected = require_sha256(
            expected_file_sha256,
            context="expected frozen Stage-A identity file SHA-256",
        )
        if file_sha256 != expected:
            raise ValueError("frozen Stage-A identity file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="frozen Stage-A identity artifact")
    require_exact_fields(
        root,
        frozenset({"canonical_evidence_sha256", "evidence"}),
        context="frozen Stage-A identity wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("frozen Stage-A identity bytes are not canonical resolver JSON")
    evidence = require_mapping(root["evidence"], context="frozen Stage-A identity evidence")
    require_exact_fields(
        evidence,
        STAGE_A_FROZEN_EVIDENCE_FIELDS,
        context="frozen Stage-A identity evidence",
    )
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="frozen Stage-A canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("frozen Stage-A canonical evidence SHA-256 drifted")
    if (
        type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != STAGE_A_IDENTITY_SCHEMA_VERSION
        or evidence["identity_schema"] != STAGE_A_FROZEN_SCHEMA
        or evidence["status"] != "frozen"
        or evidence["phase"] != "stage_a"
        or evidence["promotion_required"] is not False
    ):
        raise ValueError("frozen Stage-A identity contract drifted")

    promotion = require_mapping(evidence["promotion"], context="frozen Stage-A promotion")
    require_exact_fields(
        promotion,
        frozenset(
            {
                "candidate_file_sha256",
                "candidate_canonical_evidence_sha256",
                "explicit",
                STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD,
            }
        ),
        context="frozen Stage-A promotion",
    )
    if promotion["explicit"] is not True:
        raise ValueError("frozen Stage-A identity was not explicitly promoted")
    if (
        require_sha256(
            evidence[STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD],
            context="frozen Stage-A capture provenance receipt SHA-256",
        )
        != verified_capture.file_sha256
        or require_sha256(
            promotion[STAGE_A_CAPTURE_PROVENANCE_EVIDENCE_FIELD],
            context="promoted Stage-A capture provenance receipt SHA-256",
        )
        != verified_capture.file_sha256
    ):
        raise ValueError("frozen Stage-A identity binds a different capture provenance receipt")
    candidate_evidence = deepcopy(dict(evidence))
    candidate_evidence.pop("promotion")
    candidate_evidence["identity_schema"] = STAGE_A_CANDIDATE_SCHEMA
    candidate_evidence["status"] = "candidate"
    candidate_evidence["promotion_required"] = True
    candidate_canonical_sha256 = sha256_bytes(canonical_json_bytes(candidate_evidence))
    candidate_document = {
        "canonical_evidence_sha256": candidate_canonical_sha256,
        "evidence": candidate_evidence,
    }
    validate_candidate_artifact(candidate_document)
    if candidate_evidence["source_manifest_sha256"] != verified_capture.identity_input_file_sha256:
        raise ValueError("frozen Stage-A identity differs from its captured identity input")
    if (
        require_sha256(
            promotion["candidate_canonical_evidence_sha256"],
            context="promoted Stage-A candidate canonical evidence SHA-256",
        )
        != candidate_canonical_sha256
    ):
        raise ValueError("promoted Stage-A candidate canonical evidence SHA-256 drifted")
    candidate_file_sha256 = sha256_bytes(canonical_json_bytes(candidate_document))
    if (
        require_sha256(
            promotion["candidate_file_sha256"],
            context="promoted Stage-A candidate file SHA-256",
        )
        != candidate_file_sha256
    ):
        raise ValueError("promoted Stage-A candidate file SHA-256 drifted")

    verified_binding = verified_binding_artifact.binding
    if candidate_evidence["calibration_binding"] != verified_binding:
        raise ValueError("frozen Stage-A identity differs from the verified calibration binding")
    records = tuple(dict(record) for record in candidate_evidence["records"])
    tokenizer_manifest_sha256 = require_sha256(
        candidate_evidence["tokenizer"]["file_manifest_sha256"],
        context="frozen Stage-A tokenizer manifest SHA-256",
    )
    execution_bindings = _validate_execution_bindings(candidate_evidence["execution_bindings"])
    if execution_bindings != dict(verified_binding_artifact.execution_bindings):
        raise ValueError("frozen Stage-A execution bindings differ from calibration authorization")
    parquet_manifest_sha256 = execution_bindings["parquet_materialization_manifest_file_sha256"]
    if parquet_manifest_sha256 != PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256:
        raise ValueError("frozen Stage-A Parquet materialization manifest file SHA-256 drifted")
    return FrozenStageAIdentityArtifact(
        file_sha256=file_sha256,
        canonical_evidence_sha256=canonical_evidence_sha256,
        records=records,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        calibration_binding=dict(verified_binding),
        parquet_materialization_manifest_file_sha256=parquet_manifest_sha256,
        execution_bindings=execution_bindings,
        stage_a_capture_provenance_receipt_file_sha256=verified_capture.file_sha256,
    )


def _canonical_b64(data: bytes, *, context: str) -> str:
    if not isinstance(data, bytes):
        raise TypeError(f"{context} must be bytes")
    return base64.b64encode(data).decode("ascii")


def _decode_canonical_b64(value: object, *, context: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{context} is invalid base64") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{context} is not canonical base64")
    return decoded


def _identity_half_record_manifests(
    identity: FrozenCalibrationIdentityArtifact,
) -> dict[str, str]:
    from recurquant.static_q468_calibration import (
        calibration_identity_record_manifest_sha256,
    )

    records_by_identity = {
        (
            record["family"],
            record["ruler_category"],
            record["config"],
            record["canonical_id"],
            record["seed"],
            record["configured_length"],
            record["sequence_length"],
        ): record
        for record in identity.records
    }
    halves: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    seen: set[tuple[object, ...]] = set()
    for index, assignment in enumerate(identity.assignment):
        group = assignment["group"]
        if not isinstance(group, str):
            raise ValueError(f"identity assignment {index} group is invalid")
        family = "ruler" if group.startswith("ruler:") else group
        key = (
            family,
            assignment["ruler_category"],
            assignment["config"],
            assignment["canonical_id"],
            assignment["seed"],
            assignment["configured_length"],
            assignment["sequence_length"],
        )
        if key not in records_by_identity or key in seen:
            raise ValueError(
                "resolver split assignment does not bijectively cover identity records"
            )
        half = assignment["half"]
        if half not in halves:
            raise ValueError("resolver split assignment half drifted")
        seen.add(key)
        halves[str(half)].append(records_by_identity[key])
    if seen != set(records_by_identity):
        raise ValueError("resolver split assignments do not cover all calibration records")
    return {
        half: calibration_identity_record_manifest_sha256(records)
        for half, records in halves.items()
    }


def _derive_stage_a_calibration_binding(
    *,
    frozen_identity_artifact: bytes,
    calibration_score_artifact: bytes,
    split_half_stability_artifact: bytes,
    static_k27030_policy_artifact: bytes,
    static_k29334_policy_artifact: bytes,
    comparator_score_artifact: bytes,
    static_fisher_k29334_policy_artifact: bytes,
    static_mse_k29334_policy_artifact: bytes,
) -> tuple[dict[str, str], dict[str, str]]:
    from recurquant.static_q468 import (
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        FROZEN_STATELEASE_RESIDENT_BYTES,
        FROZEN_STATIC_Q468_ABLATION_STEPS,
        FROZEN_STATIC_Q468_PRIMARY_STEPS,
        FROZEN_TRANSFORMERS_VERSION,
        PRIMARY_MODEL_ID,
        PRIMARY_MODEL_REVISION,
        PRIMARY_TOKENIZER_ID,
        PRIMARY_TOKENIZER_REVISION,
        STATIC_Q468_ABLATION_METHOD,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        STATIC_Q468_MSE_METHOD,
        STATIC_Q468_PRIMARY_METHOD,
        build_static_rht_q468_policy,
        deserialize_static_rht_q468_policy,
        serialize_static_rht_q468_policy,
        static_q468_byte_ledger,
        static_q468_distortion_sha256,
    )
    from recurquant.static_q468_calibration import (
        CALIBRATION_SCORE_ARTIFACT_KIND,
        FROZEN_COMPARATOR_PROFILE_ORDER,
        FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        FROZEN_UNWEIGHTED_MSE_PROFILE,
        calibration_identity_record_manifest_sha256,
        deserialize_calibration_score_artifact,
        deserialize_comparator_score_artifact,
        deserialize_frozen_split_half_stability_artifact,
        static_q468_code_map_sha256,
    )

    identity = deserialize_frozen_calibration_identity_artifact(frozen_identity_artifact)
    if identity.file_sha256 != sha256_bytes(frozen_identity_artifact):
        raise ValueError("frozen identity decoder returned the wrong dependency file hash")
    scores = deserialize_calibration_score_artifact(calibration_score_artifact)
    if scores.file_sha256 != sha256_bytes(calibration_score_artifact):
        raise ValueError("score decoder returned the wrong dependency file hash")
    if scores.artifact_kind != CALIBRATION_SCORE_ARTIFACT_KIND:
        raise ValueError("Stage-A binding requires the official frozen score artifact")
    expected_identity_manifest = calibration_identity_record_manifest_sha256(identity.records)
    if scores.calibration_identity_sha256 != identity.file_sha256:
        raise ValueError("score artifact is not bound to the frozen identity file")
    if scores.aggregate.identity_record_manifest_sha256 != expected_identity_manifest:
        raise ValueError("score identity-record manifest differs from the frozen identity")
    comparators = deserialize_comparator_score_artifact(
        comparator_score_artifact,
        expected_calibration_identity_sha256=identity.file_sha256,
    )
    if (
        comparators.file_sha256 != sha256_bytes(comparator_score_artifact)
        or comparators.calibration_identity_sha256 != identity.file_sha256
        or tuple(comparators.selectors) != FROZEN_COMPARATOR_PROFILE_ORDER
    ):
        raise ValueError("comparator score artifact must contain exactly the two official profiles")
    for method_id in FROZEN_COMPARATOR_PROFILE_ORDER:
        if (
            comparators.selectors[method_id].aggregate.identity_record_manifest_sha256
            != expected_identity_manifest
        ):
            raise ValueError(
                f"comparator {method_id} identity-record manifest differs from the "
                "complete frozen identity"
            )
    split = deserialize_frozen_split_half_stability_artifact(
        split_half_stability_artifact,
        expected_identity_file_sha256=identity.file_sha256,
        expected_canonical_identity_sha256=identity.canonical_evidence_sha256,
        expected_resolver_assignment_sha256=identity.assignment_sha256,
    )
    if split.file_sha256 != sha256_bytes(split_half_stability_artifact):
        raise ValueError("split-half decoder returned the wrong dependency file hash")
    if (
        split.identity_file_sha256 != identity.file_sha256
        or split.canonical_identity_sha256 != identity.canonical_evidence_sha256
        or split.resolver_assignment_sha256 != identity.assignment_sha256
    ):
        raise ValueError("split-half artifact identity binding drifted")
    if (
        split.full_sequence_score_manifest_sha256 != scores.aggregate.sequence_score_manifest_sha256
        or split.full_calibration_scores_sha256 != scores.calibration_scores_sha256
    ):
        raise ValueError("split-half artifact differs from the official full score artifact")
    half_manifests = _identity_half_record_manifests(identity)
    if (
        split.half_a_aggregate.identity_record_manifest_sha256 != half_manifests["a"]
        or split.half_b_aggregate.identity_record_manifest_sha256 != half_manifests["b"]
    ):
        raise ValueError("split-half identity-record manifests differ from resolver assignment")

    policy27030 = deserialize_static_rht_q468_policy(static_k27030_policy_artifact)
    policy29334 = deserialize_static_rht_q468_policy(static_k29334_policy_artifact)
    expected_policy_contracts = (
        (
            policy27030,
            STATIC_Q468_ABLATION_METHOD,
            FROZEN_STATIC_Q468_ABLATION_STEPS,
        ),
        (
            policy29334,
            STATIC_Q468_PRIMARY_METHOD,
            FROZEN_STATIC_Q468_PRIMARY_STEPS,
        ),
    )
    allocations = {steps: (codes, digest) for steps, codes, digest in scores.allocations}
    torch = __import__("torch")
    for policy, method_id, steps in expected_policy_contracts:
        if (
            policy.method_id != method_id
            or policy.marginal_steps != steps
            or policy.geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY
        ):
            raise ValueError(f"policy {method_id} does not satisfy its reserved exact-K contract")
        if policy.identity_artifact_sha256 != identity.file_sha256:
            raise ValueError(f"policy {method_id} is not bound to the frozen identity file")
        if policy.tokenizer_manifest_sha256 != identity.tokenizer_manifest_sha256:
            raise ValueError(f"policy {method_id} tokenizer manifest differs from identity")
        if (
            policy.model_id != PRIMARY_MODEL_ID
            or policy.model_revision != PRIMARY_MODEL_REVISION
            or policy.tokenizer_id != PRIMARY_TOKENIZER_ID
            or policy.tokenizer_revision != PRIMARY_TOKENIZER_REVISION
            or policy.transformers_version != FROZEN_TRANSFORMERS_VERSION
        ):
            raise ValueError(f"policy {method_id} frozen model contract drifted")
        if (
            policy.calibration_manifest_sha256 != scores.aggregate.sequence_score_manifest_sha256
            or policy.calibration_scores_sha256 != scores.calibration_scores_sha256
        ):
            raise ValueError(f"policy {method_id} differs from official calibration scores")
        if steps not in allocations:
            raise ValueError(f"official score artifact is missing exact K{steps}")
        allocation_codes, allocation_hash = allocations[steps]
        if policy.code_map_sha256 != allocation_hash or not torch.equal(
            policy.precision_codes().reshape(-1).to("cpu"),
            allocation_codes.reshape(-1).to("cpu"),
        ):
            raise ValueError(f"policy {method_id} code map differs from exact allocation")
    if policy27030.source_commit != policy29334.source_commit:
        raise ValueError("K27030 and K29334 policies must share one source commit")
    source_commit_h0 = policy29334.source_commit

    comparator_policies = (
        (
            deserialize_static_rht_q468_policy(static_mse_k29334_policy_artifact),
            STATIC_Q468_MSE_METHOD,
            FROZEN_UNWEIGHTED_MSE_PROFILE,
        ),
        (
            deserialize_static_rht_q468_policy(static_fisher_k29334_policy_artifact),
            STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
            FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        ),
    )
    for policy, method_id, selector_profile in comparator_policies:
        if method_id != selector_profile:
            raise RuntimeError("frozen comparator method/profile constants disagree")
        selector = comparators.selectors[selector_profile]
        if (
            policy.method_id != method_id
            or policy.marginal_steps != FROZEN_STATIC_Q468_PRIMARY_STEPS
            or policy.geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY
        ):
            raise ValueError(
                f"comparator policy {method_id} does not satisfy its frozen K29334 geometry"
            )
        if (
            policy.model_id != PRIMARY_MODEL_ID
            or policy.model_revision != PRIMARY_MODEL_REVISION
            or policy.tokenizer_id != PRIMARY_TOKENIZER_ID
            or policy.tokenizer_revision != PRIMARY_TOKENIZER_REVISION
            or policy.transformers_version != FROZEN_TRANSFORMERS_VERSION
        ):
            raise ValueError(f"comparator policy {method_id} frozen model contract drifted")
        if (
            policy.identity_artifact_sha256 != identity.file_sha256
            or policy.tokenizer_manifest_sha256 != identity.tokenizer_manifest_sha256
        ):
            raise ValueError(f"comparator policy {method_id} frozen identity binding drifted")
        if policy.source_commit != source_commit_h0:
            raise ValueError(f"comparator policy {method_id} source commit differs from H0")
        if (
            policy.calibration_manifest_sha256 != selector.aggregate.sequence_score_manifest_sha256
            or selector.calibration_scores_sha256 != selector.aggregate.aggregate_scores_sha256
        ):
            raise ValueError(
                f"comparator policy {method_id} differs from its decoded selector scores"
            )
        expected_policy_score_sha256 = static_q468_distortion_sha256(
            *selector.aggregate.scores(),
            geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        )
        if policy.calibration_scores_sha256 != expected_policy_score_sha256:
            raise ValueError(
                f"comparator policy {method_id} raw distortion hash differs from its "
                "decoded selector arrays"
            )
        expected_policy = build_static_rht_q468_policy(
            *selector.aggregate.scores(),
            geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
            marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
            calibration_manifest_sha256=(selector.aggregate.sequence_score_manifest_sha256),
            identity_artifact_sha256=identity.file_sha256,
            tokenizer_manifest_sha256=identity.tokenizer_manifest_sha256,
            source_commit=source_commit_h0,
            calibration_scores_sha256=expected_policy_score_sha256,
            method_id=method_id,
        )
        if serialize_static_rht_q468_policy(expected_policy) != (
            static_mse_k29334_policy_artifact
            if method_id == STATIC_Q468_MSE_METHOD
            else static_fisher_k29334_policy_artifact
        ):
            raise ValueError(
                f"comparator policy {method_id} bytes differ from deterministic reconstruction"
            )
        selector_codes = selector.precision_codes.reshape(-1).to("cpu")
        expected_code_map_sha256 = static_q468_code_map_sha256(
            selector_codes,
            geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
            marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
        )
        if (
            selector.method_id != method_id
            or selector.marginal_steps != FROZEN_STATIC_Q468_PRIMARY_STEPS
            or selector.code_map_sha256 != expected_code_map_sha256
            or policy.code_map_sha256 != expected_code_map_sha256
            or not torch.equal(
                policy.precision_codes().reshape(-1).to("cpu"),
                selector_codes,
            )
        ):
            raise ValueError(
                f"comparator policy {method_id} code map differs from its exact allocation"
            )
        ledger = static_q468_byte_ledger(
            policy.geometry,
            policy.marginal_steps,
            method_id=policy.method_id,
        )
        if (
            ledger.method_id != method_id
            or ledger.selected_units != FROZEN_STATIC_Q468_PRIMARY_STEPS
            or ledger.resident_bytes != FROZEN_STATELEASE_RESIDENT_BYTES
            or ledger.target_resident_bytes != FROZEN_STATELEASE_RESIDENT_BYTES
            or ledger.budget_delta_bytes != 0
            or ledger.exact_budget_eligible is not True
        ):
            raise ValueError(
                f"comparator policy {method_id} does not realize the exact "
                f"{FROZEN_STATELEASE_RESIDENT_BYTES}-byte ledger"
            )

    binding = {
        "calibration_identity_file_sha256": identity.file_sha256,
        "calibration_score_artifact_file_sha256": scores.file_sha256,
        "comparator_score_artifact_file_sha256": comparators.file_sha256,
        "split_half_stability_artifact_file_sha256": split.file_sha256,
        "static_fisher_k29334_policy_file_sha256": sha256_bytes(
            static_fisher_k29334_policy_artifact
        ),
        "static_k27030_policy_file_sha256": sha256_bytes(static_k27030_policy_artifact),
        "static_k29334_policy_file_sha256": sha256_bytes(static_k29334_policy_artifact),
        "static_mse_k29334_policy_file_sha256": sha256_bytes(static_mse_k29334_policy_artifact),
    }
    dependency_hashes = {
        "calibration_score_artifact": scores.file_sha256,
        "comparator_score_artifact": comparators.file_sha256,
        "frozen_identity_artifact": identity.file_sha256,
        "split_half_stability_artifact": split.file_sha256,
        "static_fisher_k29334_policy_artifact": sha256_bytes(static_fisher_k29334_policy_artifact),
        "static_k27030_policy_artifact": sha256_bytes(static_k27030_policy_artifact),
        "static_k29334_policy_artifact": sha256_bytes(static_k29334_policy_artifact),
        "static_mse_k29334_policy_artifact": sha256_bytes(static_mse_k29334_policy_artifact),
    }
    return binding, dependency_hashes


def build_stage_a_calibration_core_binding_artifact(
    *,
    frozen_identity_artifact: bytes,
    calibration_score_artifact: bytes,
    split_half_stability_artifact: bytes,
    static_k27030_policy_artifact: bytes,
    static_k29334_policy_artifact: bytes,
    comparator_score_artifact: bytes,
    static_fisher_k29334_policy_artifact: bytes,
    static_mse_k29334_policy_artifact: bytes,
) -> bytes:
    """Build the pre-authorization core binding from verified calibration outputs."""

    dependencies = {
        "calibration_score_artifact": calibration_score_artifact,
        "comparator_score_artifact": comparator_score_artifact,
        "frozen_identity_artifact": frozen_identity_artifact,
        "split_half_stability_artifact": split_half_stability_artifact,
        "static_fisher_k29334_policy_artifact": static_fisher_k29334_policy_artifact,
        "static_k27030_policy_artifact": static_k27030_policy_artifact,
        "static_k29334_policy_artifact": static_k29334_policy_artifact,
        "static_mse_k29334_policy_artifact": static_mse_k29334_policy_artifact,
    }
    binding, dependency_hashes = _derive_stage_a_calibration_binding(
        frozen_identity_artifact=frozen_identity_artifact,
        calibration_score_artifact=calibration_score_artifact,
        split_half_stability_artifact=split_half_stability_artifact,
        static_k27030_policy_artifact=static_k27030_policy_artifact,
        static_k29334_policy_artifact=static_k29334_policy_artifact,
        comparator_score_artifact=comparator_score_artifact,
        static_fisher_k29334_policy_artifact=static_fisher_k29334_policy_artifact,
        static_mse_k29334_policy_artifact=static_mse_k29334_policy_artifact,
    )
    evidence = {
        "artifact_revision": STAGE_A_CORE_BINDING_ARTIFACT_REVISION,
        "binding": binding,
        "dependencies_base64": {
            name: _canonical_b64(value, context=name)
            for name, value in sorted(dependencies.items())
        },
        "dependency_file_sha256": dependency_hashes,
    }
    document = {
        "artifact_kind": STAGE_A_CORE_BINDING_ARTIFACT_KIND,
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
        "schema_version": STAGE_A_CORE_BINDING_ARTIFACT_SCHEMA_VERSION,
    }
    return canonical_json_bytes(document)


def deserialize_stage_a_calibration_core_binding_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> StageACalibrationCoreBindingArtifact:
    """Strictly reverify every dependency of a pre-authorization core binding."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A calibration core binding artifact must be bytes")
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None and file_sha256 != require_sha256(
        expected_file_sha256,
        context="expected Stage-A core binding file SHA-256",
    ):
        raise ValueError("Stage-A calibration core binding file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="Stage-A calibration core binding")
    require_exact_fields(
        root,
        frozenset(
            {
                "artifact_kind",
                "canonical_evidence_sha256",
                "evidence",
                "schema_version",
            }
        ),
        context="Stage-A calibration core binding wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("Stage-A calibration core binding bytes are not canonical JSON")
    if (
        root["artifact_kind"] != STAGE_A_CORE_BINDING_ARTIFACT_KIND
        or root["schema_version"] != STAGE_A_CORE_BINDING_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("Stage-A calibration core binding kind or schema drifted")
    evidence = require_mapping(root["evidence"], context="Stage-A core binding evidence")
    require_exact_fields(
        evidence,
        frozenset(
            {
                "artifact_revision",
                "binding",
                "dependencies_base64",
                "dependency_file_sha256",
            }
        ),
        context="Stage-A core binding evidence",
    )
    if evidence["artifact_revision"] != STAGE_A_CORE_BINDING_ARTIFACT_REVISION:
        raise ValueError("Stage-A calibration core binding revision drifted")
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="Stage-A core binding canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("Stage-A core binding canonical evidence SHA-256 drifted")
    encoded_dependencies = require_mapping(
        evidence["dependencies_base64"],
        context="Stage-A core binding dependencies",
    )
    dependency_names = CALIBRATION_CORE_DEPENDENCY_NAMES
    require_exact_fields(
        encoded_dependencies,
        dependency_names,
        context="Stage-A core binding dependencies",
    )
    dependencies = {
        name: _decode_canonical_b64(
            encoded_dependencies[name],
            context=f"Stage-A dependency {name}",
        )
        for name in sorted(dependency_names)
    }
    recorded_dependency_hashes = require_mapping(
        evidence["dependency_file_sha256"],
        context="Stage-A core binding dependency hashes",
    )
    require_exact_fields(
        recorded_dependency_hashes,
        dependency_names,
        context="Stage-A core binding dependency hashes",
    )
    normalized_dependency_hashes = {
        name: require_sha256(
            recorded_dependency_hashes[name],
            context=f"Stage-A dependency {name} file SHA-256",
        )
        for name in sorted(dependency_names)
    }
    embedded_dependency_hashes = {
        name: sha256_bytes(dependencies[name]) for name in sorted(dependency_names)
    }
    if normalized_dependency_hashes != embedded_dependency_hashes:
        raise ValueError("Stage-A calibration dependency bytes differ from their file hashes")
    binding, dependency_hashes = _derive_stage_a_calibration_binding(
        frozen_identity_artifact=dependencies["frozen_identity_artifact"],
        calibration_score_artifact=dependencies["calibration_score_artifact"],
        split_half_stability_artifact=dependencies["split_half_stability_artifact"],
        static_k27030_policy_artifact=dependencies["static_k27030_policy_artifact"],
        static_k29334_policy_artifact=dependencies["static_k29334_policy_artifact"],
        comparator_score_artifact=dependencies["comparator_score_artifact"],
        static_fisher_k29334_policy_artifact=(dependencies["static_fisher_k29334_policy_artifact"]),
        static_mse_k29334_policy_artifact=dependencies["static_mse_k29334_policy_artifact"],
    )
    if evidence["binding"] != binding:
        raise ValueError("Stage-A calibration binding fields drifted")
    if normalized_dependency_hashes != dependency_hashes:
        raise ValueError("Stage-A calibration dependency hashes drifted")
    return StageACalibrationCoreBindingArtifact(
        binding=binding,
        dependency_file_sha256=dependency_hashes,
        calibration_dependencies=dependencies,
        canonical_evidence_sha256=canonical_evidence_sha256,
        file_sha256=file_sha256,
    )


def _deserialize_runner_v9_report(
    data: bytes,
    *,
    context: str,
    expected_status: str,
) -> Mapping[str, Any]:
    root = _json_without_duplicate_keys(data, context=context)
    require_exact_fields(
        root,
        frozenset(
            {
                "artifact_kind",
                "canonical_evidence_sha256",
                "evidence",
                "schema_version",
            }
        ),
        context=f"{context} wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError(f"{context} bytes are not canonical JSON")
    if (
        root["artifact_kind"] != CALIBRATION_RUN_REPORT_KIND
        or root["schema_version"] != CALIBRATION_RUN_REPORT_SCHEMA_VERSION
    ):
        raise ValueError(f"{context} kind or schema drifted")
    evidence = require_mapping(root["evidence"], context=f"{context} evidence")
    require_exact_fields(
        evidence,
        frozenset(
            {
                "artifacts",
                "calibration",
                "identity",
                "model_files",
                "prerequisites",
                "query_energy_ema",
                "repository",
                "runner_revision",
                "runtime",
                "stability",
                "status",
            }
        ),
        context=f"{context} evidence",
    )
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context=f"{context} canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError(f"{context} canonical evidence SHA-256 drifted")
    if (
        evidence["runner_revision"] != CALIBRATION_RUNNER_REVISION
        or evidence["status"] != expected_status
    ):
        raise ValueError(f"{context} runner revision or status drifted")
    return evidence


def _canonical_relative_posix_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty canonical path")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{context} must be a single-line POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{context} must be a canonical relative POSIX path")
    return value


def _deserialize_repository_source_manifest(data: bytes) -> dict[str, Any]:
    from recurquant.experiment013_source import (
        canonical_experiment013_source_manifest_bytes,
        validate_experiment013_source_manifest,
    )

    root = _json_without_duplicate_keys(data, context="authorization repository source manifest")
    normalized = validate_experiment013_source_manifest(root)
    if canonical_experiment013_source_manifest_bytes(normalized) != data:
        raise ValueError("authorization repository source manifest bytes are not canonical")
    paths = require_sequence(
        normalized["paths"], context="authorization repository source-manifest paths"
    )
    return {
        "canonical_manifest_sha256": require_sha256(
            normalized["canonical_manifest_sha256"],
            context="authorization repository canonical manifest SHA-256",
        ),
        "file_sha256": sha256_bytes(data),
        "paths": {
            str(require_mapping(item, context="authorization source path")["path"]): dict(item)
            for item in paths
        },
        "source_commit": require_exact_revision(
            normalized["source_commit"], context="authorization source-manifest H0"
        ),
    }


def _runtime_root_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value) is None:
        raise ValueError(f"{context} is not a canonical runtime-root name")
    return value


def _normalized_distribution_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} is not a canonical distribution name")
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None or normalized != value:
        raise ValueError(f"{context} is not a normalized distribution name")
    return normalized


def _deserialize_calibration_runtime_manifest(data: bytes) -> dict[str, Any]:
    root = _json_without_duplicate_keys(data, context="authorization calibration runtime manifest")
    require_exact_fields(
        root,
        frozenset(
            {
                "artifact_kind",
                "base_runtime_root",
                "base_sys_path",
                "distributions",
                "git_executable",
                "interpreter",
                "launch_policy",
                "machine",
                "package_roots",
                "python",
                "runtime_trees",
                "schema_version",
            }
        ),
        context="authorization calibration runtime manifest",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("authorization calibration runtime manifest bytes are not canonical")
    if (
        root["artifact_kind"] != CALIBRATION_RUNTIME_MANIFEST_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != CALIBRATION_RUNTIME_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("authorization calibration runtime manifest kind or schema drifted")
    launch_policy = require_mapping(
        root["launch_policy"], context="authorization runtime launch policy"
    )
    if dict(launch_policy) != dict(CALIBRATION_SEALED_LAUNCH_POLICY):
        raise ValueError("authorization calibration runtime launch policy drifted")

    git = require_mapping(root["git_executable"], context="authorization runtime Git receipt")
    require_exact_fields(
        git,
        frozenset({"absolute_path_sha256", "sha256", "size_bytes"}),
        context="authorization runtime Git receipt",
    )
    require_sha256(git["absolute_path_sha256"], context="runtime Git path SHA-256")
    require_sha256(git["sha256"], context="runtime Git executable SHA-256")
    require_int(git["size_bytes"], context="runtime Git executable size", minimum=1)

    python = require_mapping(root["python"], context="authorization runtime Python receipt")
    require_exact_fields(
        python,
        frozenset({"abi_flags", "cache_tag", "implementation", "version"}),
        context="authorization runtime Python receipt",
    )
    if not isinstance(python["abi_flags"], str):
        raise ValueError("authorization runtime Python ABI flags are invalid")
    for name in ("cache_tag", "implementation", "version"):
        if (
            not isinstance(python[name], str)
            or not python[name]
            or python[name] != python[name].strip()
        ):
            raise ValueError(f"authorization runtime Python {name} is invalid")

    machine = require_mapping(root["machine"], context="authorization runtime machine receipt")
    require_exact_fields(
        machine,
        frozenset({"architecture", "byteorder", "machine", "pointer_bits", "system"}),
        context="authorization runtime machine receipt",
    )
    for name in ("architecture", "machine", "system"):
        if (
            not isinstance(machine[name], str)
            or not machine[name]
            or machine[name] != machine[name].strip()
        ):
            raise ValueError(f"authorization runtime machine {name} is invalid")
    if machine["byteorder"] not in {"little", "big"}:
        raise ValueError("authorization runtime machine byte order is invalid")
    require_int(machine["pointer_bits"], context="runtime machine pointer bits", minimum=1)

    base_root = _runtime_root_name(
        root["base_runtime_root"], context="authorization base runtime root"
    )
    if base_root != "base-runtime":
        raise ValueError("authorization base runtime root drifted")
    base_sys_path = require_sequence(
        root["base_sys_path"], context="authorization runtime base sys.path"
    )
    normalized_base_sys_path: list[str] = []
    for item in base_sys_path:
        normalized_base_sys_path.append(
            "."
            if item == "."
            else _canonical_relative_posix_path(item, context="runtime base sys.path entry")
        )
    if not normalized_base_sys_path or len(
        {item.casefold() for item in normalized_base_sys_path}
    ) != len(normalized_base_sys_path):
        raise ValueError("authorization runtime base sys.path is not non-empty and unique")

    raw_roots = require_sequence(
        root["package_roots"], context="authorization runtime package roots"
    )
    package_import_paths: dict[str, str] = {}
    for index, raw in enumerate(raw_roots):
        item = require_mapping(raw, context=f"authorization package_roots[{index}]")
        require_exact_fields(
            item,
            frozenset({"import_path", "name"}),
            context=f"authorization package_roots[{index}]",
        )
        name = _runtime_root_name(item["name"], context="authorization package root name")
        import_path = _canonical_relative_posix_path(
            item["import_path"], context=f"authorization package root {name} import path"
        )
        if name in package_import_paths or name == base_root:
            raise ValueError("authorization runtime package root inventory is not unique")
        package_import_paths[name] = import_path
    if not package_import_paths or list(package_import_paths) != sorted(package_import_paths):
        raise ValueError("authorization runtime package roots must be non-empty and sorted")

    raw_trees = require_sequence(root["runtime_trees"], context="authorization runtime trees")
    expected_tree_names = [base_root, *package_import_paths]
    trees: dict[str, dict[str, tuple[str, int]]] = {}
    tree_kinds: dict[str, str] = {}
    for tree_index, raw_tree in enumerate(raw_trees):
        tree = require_mapping(raw_tree, context=f"authorization runtime_trees[{tree_index}]")
        require_exact_fields(
            tree,
            frozenset({"files", "kind", "name"}),
            context=f"authorization runtime_trees[{tree_index}]",
        )
        name = _runtime_root_name(tree["name"], context="authorization runtime tree name")
        if tree["kind"] not in {"base-runtime", "packages"}:
            raise ValueError("authorization runtime tree kind is invalid")
        files: dict[str, tuple[str, int]] = {}
        raw_files = require_sequence(
            tree["files"], context=f"authorization runtime tree {name} files"
        )
        for file_index, raw_file in enumerate(raw_files):
            item = require_mapping(
                raw_file, context=f"authorization runtime tree {name} files[{file_index}]"
            )
            require_exact_fields(
                item,
                frozenset({"path", "sha256", "size_bytes"}),
                context=f"authorization runtime tree {name} files[{file_index}]",
            )
            path = _canonical_relative_posix_path(
                item["path"], context=f"authorization runtime tree {name} path"
            )
            if path.casefold() in {existing.casefold() for existing in files}:
                raise ValueError("authorization runtime tree paths are not unique")
            files[path] = (
                require_sha256(item["sha256"], context=f"runtime tree {name} file SHA-256"),
                require_int(item["size_bytes"], context=f"runtime tree {name} file size"),
            )
        if not files or list(files) != sorted(files):
            raise ValueError("authorization runtime tree files must be non-empty and sorted")
        trees[name] = files
        tree_kinds[name] = str(tree["kind"])
    if list(trees) != expected_tree_names:
        raise ValueError("authorization runtime tree order or inventory drifted")
    if tree_kinds[base_root] != "base-runtime" or any(
        tree_kinds[name] != "packages" for name in package_import_paths
    ):
        raise ValueError("authorization runtime tree kinds drifted")

    interpreter = require_mapping(root["interpreter"], context="authorization runtime interpreter")
    require_exact_fields(
        interpreter,
        frozenset({"relative_path", "root", "sha256", "size_bytes"}),
        context="authorization runtime interpreter",
    )
    interpreter_path = _canonical_relative_posix_path(
        interpreter["relative_path"], context="authorization runtime interpreter path"
    )
    interpreter_record = (
        require_sha256(interpreter["sha256"], context="runtime interpreter SHA-256"),
        require_int(interpreter["size_bytes"], context="runtime interpreter size", minimum=1),
    )
    if interpreter["root"] != base_root or trees[base_root].get(interpreter_path) != (
        interpreter_record
    ):
        raise ValueError("authorization runtime interpreter differs from the base tree")

    raw_distributions = require_sequence(
        root["distributions"], context="authorization runtime distributions"
    )
    distributions: dict[str, dict[str, Any]] = {}
    owned: dict[str, set[str]] = {name: set() for name in package_import_paths}
    for index, raw in enumerate(raw_distributions):
        item = require_mapping(raw, context=f"authorization runtime distributions[{index}]")
        require_exact_fields(
            item,
            frozenset({"files", "name", "package_root", "version"}),
            context=f"authorization runtime distributions[{index}]",
        )
        name = _normalized_distribution_name(
            item["name"], context=f"authorization runtime distributions[{index}].name"
        )
        version = item["version"]
        if not isinstance(version, str) or not version or version != version.strip():
            raise ValueError(f"authorization runtime distribution {name} version is invalid")
        package_root = _runtime_root_name(
            item["package_root"], context=f"authorization runtime distribution {name} root"
        )
        if package_root not in owned:
            raise ValueError(f"authorization runtime distribution {name} has an unknown root")
        raw_files = require_sequence(
            item["files"], context=f"authorization runtime distribution {name} files"
        )
        files = [
            _canonical_relative_posix_path(
                path, context=f"authorization runtime distribution {name} file"
            )
            for path in raw_files
        ]
        if (
            not files
            or files != sorted(files)
            or len({path.casefold() for path in files}) != len(files)
            or owned[package_root].intersection(files)
        ):
            raise ValueError(f"authorization runtime distribution {name} file inventory is invalid")
        owned[package_root].update(files)
        if name in distributions:
            raise ValueError("authorization runtime distributions are not unique")
        distributions[name] = {
            "files": tuple(files),
            "package_root": package_root,
            "version": version,
        }
    if not distributions or list(distributions) != sorted(distributions):
        raise ValueError("authorization runtime distributions must be non-empty and sorted")
    for name in package_import_paths:
        if owned[name] != set(trees[name]):
            raise ValueError(
                f"authorization runtime package tree {name} differs from RECORD ownership"
            )
    return {
        "distribution_count": len(distributions),
        "distributions": distributions,
        "file_count": sum(len(files) for files in trees.values()),
        "file_sha256": sha256_bytes(data),
        "package_import_paths": package_import_paths,
        "packages": {name: item["version"] for name, item in distributions.items()},
        "python_version": python["version"],
        "trees": trees,
    }


def _deserialize_model_file_manifest(data: bytes) -> dict[str, Any]:
    root = _json_without_duplicate_keys(data, context="authorization model file manifest")
    require_exact_fields(
        root,
        frozenset(
            {
                "artifact_kind",
                "files",
                "hub_tree_manifest_sha256",
                "metadata_derivation",
                "model_id",
                "revision",
                "schema_version",
                "selection_profile",
                "transformers_version",
            }
        ),
        context="authorization model file manifest",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("authorization model file manifest bytes are not canonical")
    if (
        root["artifact_kind"] != CALIBRATION_MODEL_FILE_MANIFEST_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != CALIBRATION_MODEL_FILE_MANIFEST_SCHEMA_VERSION
        or root["metadata_derivation"] != CALIBRATION_MODEL_FILE_MANIFEST_DERIVATION
        or root["selection_profile"] != CALIBRATION_MODEL_FILE_SELECTION_PROFILE
    ):
        raise ValueError("authorization model file manifest contract drifted")
    model_id = require_string(root["model_id"], context="authorization model ID")
    revision = require_exact_revision(root["revision"], context="authorization model revision")
    if len(revision) != 40:
        raise ValueError("authorization model revision must be a SHA-1")
    transformers_version = root["transformers_version"]
    if (
        not isinstance(transformers_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", transformers_version) is None
    ):
        raise ValueError("authorization Transformers version must be exact semver")
    raw_files = require_sequence(root["files"], context="authorization model files")
    names: list[str] = []
    tree_payload: list[dict[str, object]] = []
    for index, raw in enumerate(raw_files):
        item = require_mapping(raw, context=f"authorization model files[{index}]")
        require_exact_fields(
            item,
            frozenset(
                {"git_blob_oid", "lfs_sha256", "lfs_size_bytes", "name", "sha256", "size_bytes"}
            ),
            context=f"authorization model files[{index}]",
        )
        name = _canonical_relative_posix_path(
            item["name"], context=f"authorization model files[{index}].name"
        )
        if "/" in name or (
            name not in {"config.json", "model.safetensors.index.json"}
            and re.fullmatch(
                r"(?:model(?:-[0-9]+-of-[0-9]+)?|model\.safetensors-[0-9]+-of-[0-9]+)"
                r"\.safetensors",
                name,
            )
            is None
        ):
            raise ValueError("authorization model file falls outside the selection profile")
        require_int(item["size_bytes"], context=f"authorization model file {name} size", minimum=1)
        git_blob_oid = require_exact_revision(
            item["git_blob_oid"], context=f"authorization model file {name} Git blob"
        )
        if len(git_blob_oid) != 40:
            raise ValueError("authorization model Git blob must be SHA-1")
        is_weight = name.endswith(".safetensors")
        if is_weight:
            lfs_sha256 = require_sha256(
                item["lfs_sha256"], context=f"authorization model file {name} LFS SHA-256"
            )
            file_sha256 = require_sha256(
                item["sha256"], context=f"authorization model file {name} SHA-256"
            )
            lfs_size = require_int(
                item["lfs_size_bytes"],
                context=f"authorization model file {name} LFS size",
                minimum=1,
            )
            if lfs_sha256 != file_sha256 or lfs_size != item["size_bytes"]:
                raise ValueError("authorization model LFS identity drifted")
        elif any(item[name] is not None for name in ("lfs_sha256", "lfs_size_bytes", "sha256")):
            raise ValueError("authorization ordinary model metadata must use null LFS fields")
        names.append(name)
        tree_payload.append(
            {
                "git_blob_oid": git_blob_oid,
                "lfs_sha256": item["lfs_sha256"],
                "lfs_size_bytes": item["lfs_size_bytes"],
                "name": name,
            }
        )
    if (
        not names
        or names != sorted(names)
        or len({name.casefold() for name in names}) != len(names)
        or "config.json" not in names
        or "model.safetensors.index.json" not in names
        or not any(name.endswith(".safetensors") for name in names)
    ):
        raise ValueError("authorization model file inventory is incomplete or non-canonical")
    hub_tree_sha256 = require_sha256(
        root["hub_tree_manifest_sha256"], context="authorization model Hub tree SHA-256"
    )
    if hub_tree_sha256 != sha256_bytes(canonical_json_bytes(tree_payload)):
        raise ValueError("authorization model Hub tree metadata hash drifted")
    return {
        "file_count": len(names),
        "file_sha256": sha256_bytes(data),
        "hub_tree_manifest_sha256": hub_tree_sha256,
        "model_id": model_id,
        "revision": revision,
        "transformers_version": transformers_version,
    }


def _validate_capture_provenance_source_runtime(
    root: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
) -> None:
    """Bind the capture module and critical imports to authenticated manifests."""

    capture_source = require_mapping(
        root["capture_source"], context="capture provenance source record"
    )
    require_exact_fields(
        capture_source,
        frozenset({"path", "sha256"}),
        context="capture provenance source record",
    )
    if capture_source["path"] != CALIBRATION_CAPTURE_SOURCE_PATH:
        raise ValueError("capture provenance source path drifted")
    source_paths = require_mapping(
        source_manifest["paths"], context="authorization repository source paths"
    )
    source_entry = require_mapping(
        source_paths.get(CALIBRATION_CAPTURE_SOURCE_PATH),
        context="capture source manifest entry",
    )
    if require_sha256(
        capture_source["sha256"], context="capture provenance source SHA-256"
    ) != require_sha256(
        source_entry.get("raw_sha256"), context="capture source-manifest entry SHA-256"
    ):
        raise ValueError("capture provenance source differs from the repository manifest")
    origins = require_sequence(
        root["critical_module_origins"], context="capture provenance critical module origins"
    )
    expected_modules = sorted(CALIBRATION_CAPTURE_CRITICAL_MODULE_DISTRIBUTIONS)
    if [item.get("module") if isinstance(item, Mapping) else None for item in origins] != (
        expected_modules
    ):
        raise ValueError("capture provenance critical module inventory is not exact and sorted")
    distributions = require_mapping(
        runtime_manifest["distributions"], context="authorization runtime distributions"
    )
    package_import_paths = require_mapping(
        runtime_manifest["package_import_paths"], context="authorization package import paths"
    )
    trees = require_mapping(runtime_manifest["trees"], context="authorization runtime trees")
    for raw_origin in origins:
        origin = require_mapping(raw_origin, context="capture provenance critical module origin")
        require_exact_fields(
            origin,
            frozenset(
                {
                    "distribution",
                    "module",
                    "package_root",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                    "version",
                }
            ),
            context="capture provenance critical module origin",
        )
        module = str(origin["module"])
        distribution_name = CALIBRATION_CAPTURE_CRITICAL_MODULE_DISTRIBUTIONS[module]
        if origin["distribution"] != distribution_name:
            raise ValueError(f"capture provenance distribution mapping drifted: {module}")
        distribution = require_mapping(
            distributions.get(distribution_name),
            context=f"capture provenance runtime distribution {distribution_name}",
        )
        package_root = _runtime_root_name(
            origin["package_root"], context=f"capture provenance {module} package root"
        )
        relative_path = _canonical_relative_posix_path(
            origin["relative_path"], context=f"capture provenance {module} relative path"
        )
        if package_root != distribution.get("package_root") or origin[
            "version"
        ] != distribution.get("version"):
            raise ValueError(f"capture provenance distribution identity drifted: {module}")
        import_path = package_import_paths.get(package_root)
        if not isinstance(import_path, str):
            raise ValueError(f"capture provenance package root is not importable: {module}")
        try:
            module_relative = PurePosixPath(relative_path).relative_to(PurePosixPath(import_path))
        except ValueError as error:
            raise ValueError(
                f"capture provenance module is outside its import root: {module}"
            ) from error
        if not module_relative.parts or module_relative.parts[0] != module:
            raise ValueError(f"capture provenance module path is shadowed: {module}")
        distribution_files = distribution.get("files")
        if not isinstance(distribution_files, Sequence) or relative_path not in distribution_files:
            raise ValueError(f"capture provenance module lacks RECORD ownership: {module}")
        tree = require_mapping(
            trees.get(package_root), context=f"capture provenance runtime tree {package_root}"
        )
        expected_file = tree.get(relative_path)
        actual_file = (
            require_sha256(origin["sha256"], context=f"capture provenance {module} file SHA-256"),
            require_int(origin["size_bytes"], context=f"capture provenance {module} file size"),
        )
        if expected_file != actual_file:
            raise ValueError(f"capture provenance module differs from runtime tree: {module}")
    excluded = require_sequence(
        root["excluded_runtime_modules"], context="capture provenance excluded modules"
    )
    if tuple(excluded) != CALIBRATION_CAPTURE_EXCLUDED_RUNTIME_MODULES:
        raise ValueError("capture provenance excluded module inventory drifted")


def _deserialize_capture_provenance_receipt(
    data: bytes,
    *,
    source_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = _json_without_duplicate_keys(data, context="calibration capture provenance receipt")
    require_exact_fields(
        root,
        frozenset(
            {
                "artifact_kind",
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
        ),
        context="calibration capture provenance receipt",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("calibration capture provenance receipt is not canonical JSON")
    if (
        root["artifact_kind"] != CALIBRATION_CAPTURE_PROVENANCE_KIND
        or root["schema_version"] != CALIBRATION_CAPTURE_PROVENANCE_SCHEMA_VERSION
        or root["capture_version"] != CALIBRATION_CAPTURE_VERSION
        or root["phase"] != "calibration"
        or root["publication_contract"] != CALIBRATION_CAPTURE_PUBLICATION_CONTRACT
        or root["runner_revision"] != CALIBRATION_RUNNER_REVISION
        or root["status"] != CALIBRATION_CAPTURE_PROVENANCE_STATUS
    ):
        raise ValueError("calibration capture provenance identity drifted")
    require_exact_revision(root["source_commit"], context="capture provenance source commit")
    if len(str(root["source_commit"])) != 40:
        raise ValueError("capture provenance source commit must be a SHA-1 H0")
    require_sha256(
        root["identity_input_file_sha256"],
        context="capture provenance identity input SHA-256",
    )
    _validate_execution_bindings(root["execution_bindings"])
    _validate_capture_provenance_source_runtime(
        root,
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
    )
    return root


def _exact_json_value(value: object, expected: object, *, context: str) -> None:
    try:
        actual_bytes = canonical_json_bytes(value)
        expected_bytes = canonical_json_bytes(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not canonical JSON data") from error
    if actual_bytes != expected_bytes:
        raise ValueError(f"{context} drifted")


def _canonical_nonnegative_float_hex(value: object, *, context: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a canonical float hex string")
    try:
        decoded = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{context} must be a canonical float hex string") from error
    if not math.isfinite(decoded) or decoded < 0.0 or decoded.hex() != value:
        raise ValueError(f"{context} must be a finite canonical non-negative float hex")
    return decoded


def _runner_stability_receipt(stability: object) -> dict[str, object]:
    checks = getattr(stability, "checks", None)
    shifts = getattr(stability, "layer_mean_bitwidth_shifts", None)
    passed = getattr(stability, "passed", None)
    spearman = getattr(stability, "spearman_average_ties", None)
    jaccard = getattr(stability, "q8_jaccard", None)
    if (
        not isinstance(checks, Sequence)
        or not isinstance(shifts, Sequence)
        or not isinstance(passed, bool)
        or not isinstance(jaccard, float)
        or (spearman is not None and not isinstance(spearman, float))
    ):
        raise ValueError("verified split-half stability result is incomplete")
    return {
        "checks": [{"name": str(name), "passed": bool(ok)} for name, ok in checks],
        "layer_mean_bitwidth_shifts": [
            {"layer_index": int(layer), "shift_hex": float(shift).hex()} for layer, shift in shifts
        ],
        "passed": passed,
        "q8_jaccard_hex": jaccard.hex(),
        "spearman_average_ties_hex": None if spearman is None else spearman.hex(),
    }


def _calibration_count_receipt(
    records: Sequence[Mapping[str, Any]], *, smoke: bool
) -> dict[str, int]:
    selected = records[:1] if smoke else records
    if not selected:
        raise ValueError("authorization frozen identity has no calibration records")
    token_count = 0
    post_token_anchor_count = 0
    fisher_boundary_count = 0
    for index, record in enumerate(selected):
        sequence_length = require_int(
            record.get("sequence_length"),
            context=f"authorization calibration record {index} sequence length",
            minimum=1,
        )
        boundary = require_mapping(
            record.get("fisher_boundary"),
            context=f"authorization calibration record {index} Fisher boundary",
        )
        positions = require_sequence(
            boundary.get("boundary_positions"),
            context=f"authorization calibration record {index} Fisher positions",
        )
        if not positions:
            raise ValueError("authorization calibration record has no Fisher H=1 positions")
        token_count += sequence_length
        post_token_anchor_count += len(anchor_positions(sequence_length))
        fisher_boundary_count += len(positions)
    return {
        "expected_fisher_step_count": fisher_boundary_count,
        "fisher_boundary_count": fisher_boundary_count,
        "observed_fisher_step_count": fisher_boundary_count,
        "post_token_anchor_count": post_token_anchor_count,
        "sequence_count": len(selected),
        "token_count": token_count,
    }


def _frozen_token_sequence_manifest_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    commitments: list[dict[str, object]] = []
    for index, record in enumerate(records):
        commitments.append(
            {
                "identity_record_sha256": require_sha256(
                    record.get("identity_record_sha256"),
                    context=f"authorization calibration record {index} identity SHA-256",
                ),
                "prompt_token_ids_sha256": require_sha256(
                    record.get("prompt_token_ids_sha256"),
                    context=f"authorization calibration record {index} prompt-token SHA-256",
                ),
                "sequence_length": require_int(
                    record.get("sequence_length"),
                    context=f"authorization calibration record {index} sequence length",
                    minimum=1,
                ),
                "sequence_token_ids_sha256": require_sha256(
                    record.get("sequence_token_ids_sha256"),
                    context=f"authorization calibration record {index} sequence-token SHA-256",
                ),
                "target_token_ids_sha256": require_sha256(
                    record.get("target_token_ids_sha256"),
                    context=f"authorization calibration record {index} target-token SHA-256",
                ),
            }
        )
    return sha256_bytes(canonical_json_bytes(commitments))


def _validate_runner_runtime_receipt(
    value: object,
    *,
    context: str,
    runtime_manifest: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    identity_input_manifest_sha256: str,
    records: Sequence[Mapping[str, Any]],
    fisher_step_count: int,
) -> Mapping[str, Any]:
    runtime = require_mapping(value, context=context)
    require_exact_fields(
        runtime,
        frozenset(
            {
                "adapter",
                "authenticated_distribution_count",
                "authenticated_file_count",
                "cuda_available",
                "cuda_runtime",
                "elapsed_seconds_hex",
                "gpu",
                "packages",
                "platform",
                "python",
                "runtime_manifest_file_sha256",
                "torch",
            }
        ),
        context=context,
    )
    expected_packages = runtime_manifest["packages"]
    if (
        type(runtime["authenticated_distribution_count"]) is not int
        or runtime["authenticated_distribution_count"] != runtime_manifest["distribution_count"]
        or type(runtime["authenticated_file_count"]) is not int
        or runtime["authenticated_file_count"] != runtime_manifest["file_count"]
        or runtime["runtime_manifest_file_sha256"] != runtime_manifest["file_sha256"]
        or runtime["packages"] != expected_packages
        or runtime["python"] != runtime_manifest["python_version"]
        or not isinstance(expected_packages, Mapping)
        or expected_packages.get("torch") != CALIBRATION_CANONICAL_TORCH_DISTRIBUTION_VERSION
        or runtime["torch"] != CALIBRATION_CANONICAL_TORCH_RUNTIME_VERSION
        or runtime["cuda_available"] is not True
        or runtime["cuda_runtime"] != CALIBRATION_CANONICAL_CUDA_RUNTIME_VERSION
        or not isinstance(runtime["platform"], str)
        or not runtime["platform"]
    ):
        raise ValueError(f"{context} identity drifted")
    _canonical_nonnegative_float_hex(
        runtime["elapsed_seconds_hex"], context=f"{context} elapsed seconds"
    )

    gpu = require_mapping(runtime["gpu"], context=f"{context} GPU")
    require_exact_fields(
        gpu,
        frozenset(
            {"capability", "device_index", "name", "peak_allocated_bytes", "peak_reserved_bytes"}
        ),
        context=f"{context} GPU",
    )
    device_index = require_int(gpu["device_index"], context=f"{context} GPU device index")
    capability = require_sequence(gpu["capability"], context=f"{context} GPU capability")
    if (
        len(capability) != 2
        or type(capability[0]) is not int
        or capability[0] <= 0
        or type(capability[1]) is not int
        or capability[1] < 0
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
    ):
        raise ValueError(f"{context} GPU identity drifted")
    peak_allocated = require_int(
        gpu["peak_allocated_bytes"], context=f"{context} GPU peak allocated bytes"
    )
    peak_reserved = require_int(
        gpu["peak_reserved_bytes"], context=f"{context} GPU peak reserved bytes"
    )
    if peak_reserved < peak_allocated:
        raise ValueError(f"{context} GPU peak counters are inconsistent")

    adapter = require_mapping(runtime["adapter"], context=f"{context} adapter")
    require_exact_fields(
        adapter,
        frozenset(
            {
                "adapter_revision",
                "capture_input_sha256",
                "device",
                "fisher_step_count",
                "kernel_backend",
                "materialization_attempted",
                "materialized_sequence_count",
                "model_dtype",
                "model_id",
                "model_loaded",
                "model_loading_diagnostic_counts",
                "model_revision",
                "query_shape",
                "recurrent_layer_indices",
                "state_shape",
                "token_sequence_manifest_sha256",
                "transformers_version",
            }
        ),
        context=f"{context} adapter",
    )
    diagnostics = require_mapping(
        adapter["model_loading_diagnostic_counts"], context=f"{context} model diagnostics"
    )
    require_exact_fields(
        diagnostics,
        CALIBRATION_CANONICAL_ADAPTER_LOADING_DIAGNOSTICS,
        context=f"{context} model diagnostics",
    )
    if any(type(value) is not int or value != 0 for value in diagnostics.values()):
        raise ValueError(f"{context} model diagnostics are not empty")
    if (
        adapter["adapter_revision"] != CALIBRATION_CANONICAL_ADAPTER_REVISION
        or adapter["kernel_backend"] != CALIBRATION_CANONICAL_ADAPTER_KERNEL_BACKEND
        or adapter["model_dtype"] != CALIBRATION_CANONICAL_ADAPTER_MODEL_DTYPE
        or adapter["model_id"] != model_manifest["model_id"]
        or adapter["model_revision"] != model_manifest["revision"]
        or adapter["transformers_version"] != model_manifest["transformers_version"]
        or adapter["device"] != f"cuda:{device_index}"
        or type(adapter["fisher_step_count"]) is not int
        or adapter["fisher_step_count"] != fisher_step_count
        or adapter["materialization_attempted"] is not True
        or type(adapter["materialized_sequence_count"]) is not int
        or adapter["materialized_sequence_count"] != len(records)
        or adapter["model_loaded"] is not True
        or adapter["query_shape"] != list(CALIBRATION_CANONICAL_ADAPTER_QUERY_SHAPE)
        or adapter["recurrent_layer_indices"]
        != list(CALIBRATION_CANONICAL_ADAPTER_RECURRENT_LAYER_INDICES)
        or adapter["state_shape"] != list(CALIBRATION_CANONICAL_ADAPTER_STATE_SHAPE)
        or adapter["capture_input_sha256"] != identity_input_manifest_sha256
        or adapter["token_sequence_manifest_sha256"]
        != _frozen_token_sequence_manifest_sha256(records)
    ):
        raise ValueError(f"{context} adapter identity drifted")
    return runtime


def _validate_runner_v9_receipts(
    *,
    full_report: Mapping[str, Any],
    smoke_report: Mapping[str, Any],
    identity: FrozenCalibrationIdentityArtifact,
    identity_input_manifest_sha256: str,
    execution_bindings: Mapping[str, str],
    repository_source_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    expected_full_stability: Mapping[str, Any],
) -> None:
    expected_identity = {
        "canonical_evidence_sha256": identity.canonical_evidence_sha256,
        "execution_bindings": dict(execution_bindings),
        "file_sha256": identity.file_sha256,
        "identity_input_manifest_sha256": identity_input_manifest_sha256,
        "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
    }
    expected_repository = {
        "source_commit": repository_source_manifest["source_commit"],
        "source_manifest_file_sha256": repository_source_manifest["file_sha256"],
        "source_manifest_sha256": repository_source_manifest["canonical_manifest_sha256"],
    }
    expected_model = {
        "file_count": model_manifest["file_count"],
        "hub_tree_manifest_sha256": model_manifest["hub_tree_manifest_sha256"],
        "manifest_file_sha256": model_manifest["file_sha256"],
        "model_id": model_manifest["model_id"],
        "revision": model_manifest["revision"],
        "transformers_version": model_manifest["transformers_version"],
    }
    full_counts = _calibration_count_receipt(identity.records, smoke=False)
    smoke_counts = _calibration_count_receipt(identity.records, smoke=True)
    _exact_json_value(full_report["identity"], expected_identity, context="full identity receipt")
    _exact_json_value(smoke_report["identity"], expected_identity, context="smoke identity receipt")
    _exact_json_value(
        full_report["repository"], expected_repository, context="full repository receipt"
    )
    _exact_json_value(
        smoke_report["repository"], expected_repository, context="smoke repository receipt"
    )
    _exact_json_value(full_report["model_files"], expected_model, context="full model receipt")
    _exact_json_value(smoke_report["model_files"], expected_model, context="smoke model receipt")
    _exact_json_value(
        full_report["query_energy_ema"],
        CALIBRATION_QUERY_ENERGY_EMA,
        context="full query-energy contract",
    )
    _exact_json_value(
        smoke_report["query_energy_ema"],
        CALIBRATION_QUERY_ENERGY_EMA,
        context="smoke query-energy contract",
    )
    _exact_json_value(full_report["calibration"], full_counts, context="full calibration counters")
    _exact_json_value(
        smoke_report["calibration"], smoke_counts, context="smoke calibration counters"
    )
    _exact_json_value(
        full_report["stability"], expected_full_stability, context="full stability receipt"
    )
    _exact_json_value(
        smoke_report["stability"],
        {"checks": [], "evaluated": False, "passed": None, "scope": "smoke_only"},
        context="smoke stability receipt",
    )
    full_runtime = _validate_runner_runtime_receipt(
        full_report["runtime"],
        context="full runtime receipt",
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        identity_input_manifest_sha256=identity_input_manifest_sha256,
        records=identity.records,
        fisher_step_count=full_counts["expected_fisher_step_count"],
    )
    smoke_runtime = _validate_runner_runtime_receipt(
        smoke_report["runtime"],
        context="smoke runtime receipt",
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        identity_input_manifest_sha256=identity_input_manifest_sha256,
        records=identity.records,
        fisher_step_count=smoke_counts["expected_fisher_step_count"],
    )
    parity_fields = {
        "authenticated_distribution_count",
        "authenticated_file_count",
        "cuda_available",
        "cuda_runtime",
        "packages",
        "platform",
        "python",
        "runtime_manifest_file_sha256",
        "torch",
    }
    if any(full_runtime[name] != smoke_runtime[name] for name in parity_fields):
        raise ValueError("full and smoke runtime identity receipts differ")
    full_gpu = require_mapping(full_runtime["gpu"], context="full runtime GPU")
    smoke_gpu = require_mapping(smoke_runtime["gpu"], context="smoke runtime GPU")
    if any(full_gpu[name] != smoke_gpu[name] for name in ("capability", "device_index", "name")):
        raise ValueError("full and smoke GPU identity receipts differ")
    full_adapter = dict(require_mapping(full_runtime["adapter"], context="full adapter"))
    smoke_adapter = dict(require_mapping(smoke_runtime["adapter"], context="smoke adapter"))
    full_adapter.pop("fisher_step_count")
    smoke_adapter.pop("fisher_step_count")
    if full_adapter != smoke_adapter:
        raise ValueError("full and smoke adapter identity receipts differ")


def _derive_stage_a_calibration_authorization(
    dependencies: Mapping[str, bytes],
) -> tuple[
    dict[str, str],
    dict[str, bytes],
    dict[str, str],
    dict[str, object],
]:
    require_exact_fields(
        dependencies,
        CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES,
        context="calibration authorization dependencies",
    )
    if dependencies["calibration_complete_marker"] != CALIBRATION_COMPLETE_BYTES:
        raise ValueError("calibration completion marker drifted")
    if dependencies["fisher_h1_smoke_complete_marker"] != FISHER_H1_SMOKE_COMPLETE_BYTES:
        raise ValueError("Fisher H=1 smoke completion marker drifted")

    core = deserialize_stage_a_calibration_core_binding_artifact(
        dependencies["calibration_core_binding_artifact"]
    )
    calibration_dependencies = dict(core.calibration_dependencies)
    identity_bytes = calibration_dependencies["frozen_identity_artifact"]
    identity = deserialize_frozen_calibration_identity_artifact(identity_bytes)
    identity_root = _json_without_duplicate_keys(
        identity_bytes, context="authorization frozen calibration identity"
    )
    identity_evidence = require_mapping(
        identity_root["evidence"], context="authorization frozen identity evidence"
    )
    identity_input_manifest_sha256 = require_sha256(
        identity_evidence["source_manifest_sha256"],
        context="authorization identity input manifest SHA-256",
    )
    execution_bindings = dict(identity.execution_bindings)
    if set(execution_bindings) != EXECUTION_BINDING_FIELDS:
        raise ValueError("authorization frozen identity execution bindings drifted")

    repository_source_manifest = _deserialize_repository_source_manifest(
        dependencies["repository_source_manifest"]
    )
    runtime_manifest = _deserialize_calibration_runtime_manifest(
        dependencies["calibration_runtime_manifest"]
    )
    model_manifest = _deserialize_model_file_manifest(dependencies["model_file_manifest"])
    embedded_execution_hashes = {
        "calibration_runtime_manifest_file_sha256": runtime_manifest["file_sha256"],
        "model_file_manifest_file_sha256": model_manifest["file_sha256"],
        "repository_source_manifest_file_sha256": repository_source_manifest["file_sha256"],
    }
    if any(
        execution_bindings[name] != digest for name, digest in embedded_execution_hashes.items()
    ):
        raise ValueError("authorization execution manifests differ from the frozen identity")

    from recurquant.static_q468 import (
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        FROZEN_STATIC_Q48_PROMOTIONS,
        PRIMARY_MODEL_ID,
        PRIMARY_MODEL_REVISION,
        PRIMARY_TOKENIZER_ID,
        PRIMARY_TOKENIZER_REVISION,
        STATIC_Q48_COMPARATOR_METHOD,
        build_static_rht_q48_policy,
        deserialize_static_rht_q48_policy,
        deserialize_static_rht_q468_policy,
        serialize_static_rht_q48_policy,
    )
    from recurquant.static_q468_calibration import (
        deserialize_calibration_score_artifact,
        deserialize_frozen_split_half_stability_artifact,
    )

    if (
        model_manifest["model_id"] != PRIMARY_MODEL_ID
        or model_manifest["revision"] != PRIMARY_MODEL_REVISION
        or model_manifest["transformers_version"] != TRANSFORMERS_VERSION
    ):
        raise ValueError("authorization model manifest differs from the frozen model contract")

    h0_policy = deserialize_static_rht_q468_policy(
        calibration_dependencies["static_k29334_policy_artifact"]
    )
    source_commit = require_exact_revision(
        h0_policy.source_commit, context="verified K29334 policy H0"
    )
    if len(source_commit) != 40:
        raise ValueError("verified K29334 policy H0 must be a SHA-1")
    if repository_source_manifest["source_commit"] != source_commit:
        raise ValueError("repository source-manifest H0 differs from the verified core policies")
    q48 = deserialize_static_rht_q48_policy(dependencies["static_q48_policy_artifact"])
    if (
        q48.identity_artifact_sha256 != identity.file_sha256
        or q48.tokenizer_manifest_sha256 != identity.tokenizer_manifest_sha256
        or q48.model_id != PRIMARY_MODEL_ID
        or q48.model_revision != PRIMARY_MODEL_REVISION
        or q48.tokenizer_id != PRIMARY_TOKENIZER_ID
        or q48.tokenizer_revision != PRIMARY_TOKENIZER_REVISION
        or q48.source_commit != source_commit
    ):
        raise ValueError("Q48 policy differs from the frozen calibration identity")
    scores = deserialize_calibration_score_artifact(
        calibration_dependencies["calibration_score_artifact"]
    )
    expected_q48 = build_static_rht_q48_policy(
        scores.aggregate.d4,
        scores.aggregate.d8,
        geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        promoted_rows=FROZEN_STATIC_Q48_PROMOTIONS,
        calibration_manifest_sha256=scores.aggregate.sequence_score_manifest_sha256,
        identity_artifact_sha256=identity.file_sha256,
        tokenizer_manifest_sha256=identity.tokenizer_manifest_sha256,
        source_commit=source_commit,
        method_id=STATIC_Q48_COMPARATOR_METHOD,
    )
    if serialize_static_rht_q48_policy(expected_q48) != dependencies["static_q48_policy_artifact"]:
        raise ValueError("Q48 policy bytes differ from deterministic P14739 reconstruction")

    authorized_output_file_sha256 = {
        filename: sha256_bytes(
            dependencies[role]
            if role in {"static_q48_policy_artifact", "calibration_core_binding_artifact"}
            else calibration_dependencies[role]
        )
        for role, filename in sorted(CALIBRATION_OUTPUT_FILENAMES.items())
    }
    full_report = _deserialize_runner_v9_report(
        dependencies["calibration_run_report"],
        context="full calibration run report",
        expected_status="passed",
    )
    artifacts = require_mapping(
        full_report["artifacts"], context="full calibration artifact inventory"
    )
    if dict(artifacts) != authorized_output_file_sha256:
        raise ValueError("full calibration report artifact inventory drifted")

    capture_receipt_sha256 = sha256_bytes(dependencies["capture_provenance_receipt"])
    smoke_report_sha256 = sha256_bytes(dependencies["fisher_h1_smoke_report"])
    prerequisites = require_mapping(
        full_report["prerequisites"], context="full calibration prerequisites"
    )
    if prerequisites != {
        "capture_provenance_receipt_file_sha256": capture_receipt_sha256,
        "fisher_h1_smoke_report_file_sha256": smoke_report_sha256,
    }:
        raise ValueError("full calibration prerequisite binding drifted")

    smoke_report = _deserialize_runner_v9_report(
        dependencies["fisher_h1_smoke_report"],
        context="Fisher H=1 smoke report",
        expected_status="fisher_h1_smoke_passed",
    )
    if smoke_report["artifacts"] != {}:
        raise ValueError("Fisher H=1 smoke report unexpectedly authorizes outputs")
    if smoke_report["prerequisites"] != {
        "capture_provenance_receipt_file_sha256": capture_receipt_sha256,
        "fisher_h1_smoke_report_file_sha256": None,
    }:
        raise ValueError("Fisher H=1 smoke prerequisite binding drifted")
    split = deserialize_frozen_split_half_stability_artifact(
        calibration_dependencies["split_half_stability_artifact"]
    )
    _validate_runner_v9_receipts(
        full_report=full_report,
        smoke_report=smoke_report,
        identity=identity,
        identity_input_manifest_sha256=identity_input_manifest_sha256,
        execution_bindings=execution_bindings,
        repository_source_manifest=repository_source_manifest,
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        expected_full_stability=_runner_stability_receipt(split.stability),
    )

    capture = _deserialize_capture_provenance_receipt(
        dependencies["capture_provenance_receipt"],
        source_manifest=repository_source_manifest,
        runtime_manifest=runtime_manifest,
    )
    if (
        capture["source_commit"] != source_commit
        or capture["identity_input_file_sha256"] != identity_input_manifest_sha256
        or capture["execution_bindings"] != execution_bindings
    ):
        raise ValueError("capture provenance receipt differs from H0/identity bindings")

    bindings: dict[str, object] = {
        "calibration_core_binding_file_sha256": core.file_sha256,
        "calibration_run_report_file_sha256": sha256_bytes(dependencies["calibration_run_report"]),
        "capture_provenance_receipt_file_sha256": capture_receipt_sha256,
        "execution_bindings": execution_bindings,
        "fisher_h1_smoke_report_file_sha256": smoke_report_sha256,
        "frozen_calibration_identity_file_sha256": identity.file_sha256,
        "identity_input_manifest_sha256": identity_input_manifest_sha256,
        "source_commit": source_commit,
        "static_q48_policy_file_sha256": sha256_bytes(dependencies["static_q48_policy_artifact"]),
    }
    return (
        dict(core.binding),
        calibration_dependencies,
        authorized_output_file_sha256,
        bindings,
    )


def build_stage_a_calibration_authorization_artifact(
    *,
    calibration_run_report: bytes,
    calibration_complete_marker: bytes,
    capture_provenance_receipt: bytes,
    fisher_h1_smoke_report: bytes,
    fisher_h1_smoke_complete_marker: bytes,
    calibration_core_binding_artifact: bytes,
    calibration_runtime_manifest: bytes,
    model_file_manifest: bytes,
    repository_source_manifest: bytes,
    static_q48_policy_artifact: bytes,
) -> bytes:
    """Authorize Stage A only after the complete runner-v10 chain is finalized."""

    dependencies = {
        "calibration_complete_marker": calibration_complete_marker,
        "calibration_core_binding_artifact": calibration_core_binding_artifact,
        "calibration_run_report": calibration_run_report,
        "calibration_runtime_manifest": calibration_runtime_manifest,
        "capture_provenance_receipt": capture_provenance_receipt,
        "fisher_h1_smoke_complete_marker": fisher_h1_smoke_complete_marker,
        "fisher_h1_smoke_report": fisher_h1_smoke_report,
        "model_file_manifest": model_file_manifest,
        "repository_source_manifest": repository_source_manifest,
        "static_q48_policy_artifact": static_q48_policy_artifact,
    }
    _binding, _calibration_dependencies, output_hashes, bindings = (
        _derive_stage_a_calibration_authorization(dependencies)
    )
    dependency_hashes = {name: sha256_bytes(value) for name, value in sorted(dependencies.items())}
    evidence = {
        "artifact_revision": STAGE_A_CALIBRATION_AUTHORIZATION_REVISION,
        "authorized_output_file_sha256": output_hashes,
        "bindings": bindings,
        "dependencies_base64": {
            name: _canonical_b64(value, context=name)
            for name, value in sorted(dependencies.items())
        },
        "dependency_file_sha256": dependency_hashes,
        "status": STAGE_A_CALIBRATION_AUTHORIZATION_STATUS,
    }
    document = {
        "artifact_kind": STAGE_A_CALIBRATION_AUTHORIZATION_ARTIFACT_KIND,
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
        "schema_version": STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA_VERSION,
    }
    return canonical_json_bytes(document)


def deserialize_stage_a_calibration_authorization_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> StageACalibrationAuthorizationArtifact:
    """Strictly rederive a post-calibration Stage-A authorization artifact."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A calibration authorization artifact must be bytes")
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None and file_sha256 != require_sha256(
        expected_file_sha256,
        context="expected calibration authorization file SHA-256",
    ):
        raise ValueError("Stage-A calibration authorization file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="Stage-A calibration authorization")
    require_exact_fields(
        root,
        frozenset({"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"}),
        context="Stage-A calibration authorization wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("Stage-A calibration authorization is not canonical JSON")
    if (
        root["artifact_kind"] != STAGE_A_CALIBRATION_AUTHORIZATION_ARTIFACT_KIND
        or root["schema_version"] != STAGE_A_CALIBRATION_AUTHORIZATION_SCHEMA_VERSION
    ):
        raise ValueError("Stage-A calibration authorization kind or schema drifted")
    evidence = require_mapping(root["evidence"], context="calibration authorization evidence")
    require_exact_fields(
        evidence,
        frozenset(
            {
                "artifact_revision",
                "authorized_output_file_sha256",
                "bindings",
                "dependencies_base64",
                "dependency_file_sha256",
                "status",
            }
        ),
        context="calibration authorization evidence",
    )
    if (
        evidence["artifact_revision"] != STAGE_A_CALIBRATION_AUTHORIZATION_REVISION
        or evidence["status"] != STAGE_A_CALIBRATION_AUTHORIZATION_STATUS
    ):
        raise ValueError("Stage-A calibration authorization revision or status drifted")
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="calibration authorization canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("calibration authorization canonical evidence SHA-256 drifted")
    encoded = require_mapping(
        evidence["dependencies_base64"], context="calibration authorization dependencies"
    )
    hashes = require_mapping(
        evidence["dependency_file_sha256"], context="calibration authorization hashes"
    )
    require_exact_fields(
        encoded,
        CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES,
        context="calibration authorization dependencies",
    )
    require_exact_fields(
        hashes,
        CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES,
        context="calibration authorization hashes",
    )
    dependencies = {
        name: _decode_canonical_b64(
            encoded[name], context=f"calibration authorization dependency {name}"
        )
        for name in sorted(CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES)
    }
    normalized_hashes = {
        name: require_sha256(
            hashes[name], context=f"calibration authorization dependency {name} SHA-256"
        )
        for name in sorted(CALIBRATION_AUTHORIZATION_DEPENDENCY_NAMES)
    }
    if normalized_hashes != {
        name: sha256_bytes(value) for name, value in sorted(dependencies.items())
    }:
        raise ValueError("calibration authorization dependency bytes differ from their hashes")
    binding, calibration_dependencies, output_hashes, bindings = (
        _derive_stage_a_calibration_authorization(dependencies)
    )
    if evidence["authorized_output_file_sha256"] != output_hashes:
        raise ValueError("calibration authorization output inventory drifted")
    if evidence["bindings"] != bindings:
        raise ValueError("calibration authorization custody bindings drifted")
    execution_bindings = require_mapping(
        bindings["execution_bindings"], context="calibration authorization execution bindings"
    )
    return StageACalibrationAuthorizationArtifact(
        binding=binding,
        calibration_dependencies=calibration_dependencies,
        authorization_dependencies=dependencies,
        authorized_output_file_sha256=output_hashes,
        execution_bindings=dict(execution_bindings),
        source_commit=str(bindings["source_commit"]),
        identity_input_manifest_sha256=str(bindings["identity_input_manifest_sha256"]),
        canonical_evidence_sha256=canonical_evidence_sha256,
        file_sha256=file_sha256,
    )


def build_stage_a_calibration_binding_artifact(
    *,
    calibration_authorization_artifact: bytes,
) -> bytes:
    """Build the only Stage-A-eligible binding from a verified authorization."""

    authorization = deserialize_stage_a_calibration_authorization_artifact(
        calibration_authorization_artifact
    )
    binding = dict(authorization.binding)
    binding["calibration_authorization_file_sha256"] = authorization.file_sha256
    dependency_hashes = {
        "calibration_authorization_artifact": authorization.file_sha256,
    }
    evidence = {
        "artifact_revision": STAGE_A_BINDING_ARTIFACT_REVISION,
        "binding": binding,
        "dependencies_base64": {
            "calibration_authorization_artifact": _canonical_b64(
                calibration_authorization_artifact,
                context="calibration_authorization_artifact",
            )
        },
        "dependency_file_sha256": dependency_hashes,
    }
    document = {
        "artifact_kind": STAGE_A_BINDING_ARTIFACT_KIND,
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
        "schema_version": STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION,
    }
    return canonical_json_bytes(document)


def deserialize_stage_a_calibration_binding_artifact(
    data: bytes,
    *,
    expected_file_sha256: str | None = None,
) -> StageACalibrationBindingArtifact:
    """Require and reverify the embedded post-calibration authorization chain."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None and file_sha256 != require_sha256(
        expected_file_sha256, context="expected Stage-A binding file SHA-256"
    ):
        raise ValueError("Stage-A calibration binding file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="Stage-A calibration binding")
    require_exact_fields(
        root,
        frozenset({"artifact_kind", "canonical_evidence_sha256", "evidence", "schema_version"}),
        context="Stage-A calibration binding wrapper",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("Stage-A calibration binding bytes are not canonical JSON")
    if (
        root["artifact_kind"] != STAGE_A_BINDING_ARTIFACT_KIND
        or root["schema_version"] != STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("Stage-A calibration binding kind or schema drifted")
    evidence = require_mapping(root["evidence"], context="Stage-A binding evidence")
    require_exact_fields(
        evidence,
        frozenset(
            {"artifact_revision", "binding", "dependencies_base64", "dependency_file_sha256"}
        ),
        context="Stage-A binding evidence",
    )
    if evidence["artifact_revision"] != STAGE_A_BINDING_ARTIFACT_REVISION:
        raise ValueError("Stage-A calibration binding revision drifted")
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"], context="Stage-A binding canonical evidence SHA-256"
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("Stage-A binding canonical evidence SHA-256 drifted")
    dependency_name = "calibration_authorization_artifact"
    encoded = require_mapping(evidence["dependencies_base64"], context="Stage-A dependencies")
    hashes = require_mapping(evidence["dependency_file_sha256"], context="Stage-A hashes")
    expected_names = frozenset({dependency_name})
    require_exact_fields(encoded, expected_names, context="Stage-A dependencies")
    require_exact_fields(hashes, expected_names, context="Stage-A hashes")
    authorization_bytes = _decode_canonical_b64(
        encoded[dependency_name], context="Stage-A calibration authorization dependency"
    )
    authorization_hash = require_sha256(
        hashes[dependency_name], context="Stage-A calibration authorization SHA-256"
    )
    if sha256_bytes(authorization_bytes) != authorization_hash:
        raise ValueError("Stage-A calibration authorization bytes differ from their hash")
    authorization = deserialize_stage_a_calibration_authorization_artifact(
        authorization_bytes,
        expected_file_sha256=authorization_hash,
    )
    binding = dict(authorization.binding)
    binding["calibration_authorization_file_sha256"] = authorization_hash
    if evidence["binding"] != binding:
        raise ValueError("Stage-A calibration binding fields drifted")
    return StageACalibrationBindingArtifact(
        binding=binding,
        dependency_file_sha256={dependency_name: authorization_hash},
        calibration_dependencies=dict(authorization.calibration_dependencies),
        authorization_dependencies=dict(authorization.authorization_dependencies),
        authorization_file_sha256=authorization_hash,
        execution_bindings=dict(authorization.execution_bindings),
        source_commit=authorization.source_commit,
        canonical_evidence_sha256=canonical_evidence_sha256,
        file_sha256=file_sha256,
    )


def deserialize_stage_a_capture_provenance_receipt(
    data: bytes,
    *,
    expected_file_sha256: str,
    calibration_binding_artifact: bytes,
    expected_identity_input_file_sha256: str | None = None,
) -> StageACaptureProvenanceReceipt:
    """Authenticate one finalized Stage-A capture before downstream data access."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A capture provenance receipt must be bytes")
    if not isinstance(calibration_binding_artifact, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    expected_receipt_sha256 = require_sha256(
        expected_file_sha256,
        context="expected Stage-A capture provenance receipt SHA-256",
    )
    file_sha256 = sha256_bytes(data)
    if file_sha256 != expected_receipt_sha256:
        raise ValueError("Stage-A capture provenance receipt differs from its explicit SHA-256")

    binding = deserialize_stage_a_calibration_binding_artifact(calibration_binding_artifact)
    root = _json_without_duplicate_keys(data, context="Stage-A capture provenance receipt")
    require_exact_fields(
        root,
        frozenset(
            {
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
        ),
        context="Stage-A capture provenance receipt",
    )
    if canonical_json_bytes(root) != data:
        raise ValueError("Stage-A capture provenance receipt is not canonical JSON")
    if (
        root["artifact_kind"] != STAGE_A_CAPTURE_PROVENANCE_KIND
        or type(root["schema_version"]) is not int
        or root["schema_version"] != STAGE_A_CAPTURE_PROVENANCE_SCHEMA_VERSION
        or type(root["capture_version"]) is not int
        or root["capture_version"] != CALIBRATION_CAPTURE_VERSION
        or root["runner_revision"] != CALIBRATION_RUNNER_REVISION
        or root["phase"] != "stage_a"
        or root["status"] != STAGE_A_CAPTURE_PROVENANCE_STATUS
        or root["publication_contract"] != STAGE_A_CAPTURE_PUBLICATION_CONTRACT
    ):
        raise ValueError("Stage-A capture provenance finalized identity drifted")

    source_commit = require_exact_revision(
        root["source_commit"], context="Stage-A capture provenance source commit"
    )
    if len(source_commit) != 40 or source_commit != binding.source_commit:
        raise ValueError("Stage-A capture provenance source commit differs from authorized H0")
    identity_input_file_sha256 = require_sha256(
        root["identity_input_file_sha256"],
        context="Stage-A capture provenance identity input SHA-256",
    )
    if expected_identity_input_file_sha256 is not None and identity_input_file_sha256 != (
        require_sha256(
            expected_identity_input_file_sha256,
            context="expected Stage-A identity input SHA-256",
        )
    ):
        raise ValueError("Stage-A capture provenance binds a different identity input")
    calibration_binding_file_sha256 = require_sha256(
        root["calibration_binding_file_sha256"],
        context="Stage-A capture provenance calibration binding SHA-256",
    )
    if calibration_binding_file_sha256 != binding.file_sha256:
        raise ValueError("Stage-A capture provenance binds a different calibration binding")
    calibration_authorization_file_sha256 = require_sha256(
        root["calibration_authorization_file_sha256"],
        context="Stage-A capture provenance calibration authorization SHA-256",
    )
    if calibration_authorization_file_sha256 != binding.authorization_file_sha256:
        raise ValueError("Stage-A capture provenance binds a different embedded authorization")
    execution_bindings = _validate_execution_bindings(root["execution_bindings"])
    if dict(root["execution_bindings"]) != execution_bindings or execution_bindings != dict(
        binding.execution_bindings
    ):
        raise ValueError("Stage-A capture provenance execution bindings drifted")

    authorization_dependencies = require_mapping(
        binding.authorization_dependencies,
        context="Stage-A calibration authorization dependencies",
    )
    source_manifest = _deserialize_repository_source_manifest(
        authorization_dependencies["repository_source_manifest"]
    )
    runtime_manifest = _deserialize_calibration_runtime_manifest(
        authorization_dependencies["calibration_runtime_manifest"]
    )
    if source_manifest["source_commit"] != source_commit:
        raise ValueError("Stage-A capture provenance source manifest differs from H0")
    _validate_capture_provenance_source_runtime(
        root,
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
    )
    return StageACaptureProvenanceReceipt(
        file_sha256=file_sha256,
        identity_input_file_sha256=identity_input_file_sha256,
        calibration_binding_file_sha256=calibration_binding_file_sha256,
        calibration_authorization_file_sha256=calibration_authorization_file_sha256,
        source_commit=source_commit,
        execution_bindings=execution_bindings,
    )


def validate_quarantine_output(path: Path) -> None:
    resolved = path.resolve()
    if not any("quarantine" in part.lower() for part in resolved.parts[:-1]):
        raise ValueError("candidate output must be inside a quarantine directory")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {resolved}")


def validate_promotion_output(path: Path) -> None:
    resolved = path.resolve()
    if any("quarantine" in part.lower() for part in resolved.parts[:-1]):
        raise ValueError("frozen identity output must be outside quarantine")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing identity: {resolved}")


def atomic_write(path: Path, payload: bytes) -> None:
    """Atomically publish one new file without a check-then-replace race."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing identity file: {resolved}"
            ) from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve or explicitly promote an Experiment 013 identity."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibration", "stage_a", "stage_b", "stage_c"),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--calibration-binding", type=Path)
    parser.add_argument("--stage-a-capture-provenance-receipt", type=Path)
    parser.add_argument("--expected-stage-a-capture-provenance-receipt-sha256")
    parser.add_argument("--mbpp-revision")
    parser.add_argument("--pg19-revision")
    parser.add_argument("--ruler-revision")
    parser.add_argument("--humaneval-plus-revision")
    return parser.parse_args(argv)


def _reject_protected_before_input(phase: str) -> None:
    if phase in PROTECTED_STAGES:
        raise PermissionError(
            f"{phase} is protected; resolver v{RESOLVER_VERSION} refuses it before reading --input"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _reject_protected_before_input(args.phase)
    if args.phase != "stage_a" and (
        args.stage_a_capture_provenance_receipt is not None
        or args.expected_stage_a_capture_provenance_receipt_sha256 is not None
    ):
        raise ValueError("Stage-A capture provenance options are valid only for Stage A")
    if args.promote:
        if args.dry_run or args.output is None:
            raise ValueError("promotion requires --output and forbids --dry-run")
        calibration_binding_artifact: bytes | None = None
        stage_a_capture_provenance_receipt: bytes | None = None
        expected_stage_a_capture_provenance_receipt_sha256: str | None = None
        if args.phase == "stage_a":
            if args.calibration_binding is None:
                raise ValueError("Stage-A promotion requires --calibration-binding")
            if args.stage_a_capture_provenance_receipt is None:
                raise ValueError("Stage-A promotion requires --stage-a-capture-provenance-receipt")
            if args.expected_stage_a_capture_provenance_receipt_sha256 is None:
                raise ValueError(
                    "Stage-A promotion requires an explicit capture provenance receipt SHA-256"
                )
            expected_stage_a_capture_provenance_receipt_sha256 = require_sha256(
                args.expected_stage_a_capture_provenance_receipt_sha256,
                context="--expected-stage-a-capture-provenance-receipt-sha256",
            )
            calibration_binding_artifact = args.calibration_binding.read_bytes()
            deserialize_stage_a_calibration_binding_artifact(calibration_binding_artifact)
            stage_a_capture_provenance_receipt = (
                args.stage_a_capture_provenance_receipt.read_bytes()
            )
            deserialize_stage_a_capture_provenance_receipt(
                stage_a_capture_provenance_receipt,
                expected_file_sha256=expected_stage_a_capture_provenance_receipt_sha256,
                calibration_binding_artifact=calibration_binding_artifact,
            )
        elif args.calibration_binding is not None:
            raise ValueError("calibration promotion forbids --calibration-binding")
        expected_hash = require_sha256(
            args.expected_candidate_sha256,
            context="--expected-candidate-sha256",
        )
        raw = args.input.read_bytes()
        actual_hash = sha256_bytes(raw)
        if actual_hash != expected_hash:
            raise ValueError("candidate file SHA-256 does not match explicit promotion hash")
        candidate = _json_without_duplicate_keys(raw, context="candidate artifact")
        if raw != canonical_json_bytes(candidate):
            raise ValueError("candidate file bytes are not canonical resolver JSON")
        if candidate.get("evidence", {}).get("phase") != args.phase:
            raise ValueError("candidate phase does not match --phase")
        frozen = promote_candidate(
            candidate,
            candidate_file_sha256=actual_hash,
            calibration_binding_artifact=calibration_binding_artifact,
            stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
            expected_stage_a_capture_provenance_receipt_sha256=(
                expected_stage_a_capture_provenance_receipt_sha256
            ),
        )
        validate_promotion_output(args.output)
        atomic_write(args.output, canonical_json_bytes(frozen))
        print(sha256_bytes(canonical_json_bytes(frozen)))
        return 0

    if args.expected_candidate_sha256 is not None:
        raise ValueError("--expected-candidate-sha256 is valid only with --promote")
    revisions = {
        "mbpp": args.mbpp_revision,
        "pg19": args.pg19_revision,
        "ruler": args.ruler_revision,
        "humaneval_plus": args.humaneval_plus_revision,
    }
    if any(value is None for value in revisions.values()):
        raise ValueError("all four dataset revision arguments are mandatory")
    calibration_binding_artifact: bytes | None = None
    stage_a_capture_provenance_receipt: bytes | None = None
    expected_stage_a_capture_provenance_receipt_sha256: str | None = None
    verified_stage_a_capture: StageACaptureProvenanceReceipt | None = None
    if args.phase == "stage_a":
        if args.calibration_binding is None:
            raise ValueError("Stage A requires --calibration-binding")
        if args.stage_a_capture_provenance_receipt is None:
            raise ValueError("Stage A requires --stage-a-capture-provenance-receipt")
        if args.expected_stage_a_capture_provenance_receipt_sha256 is None:
            raise ValueError("Stage A requires an explicit capture provenance receipt SHA-256")
        expected_stage_a_capture_provenance_receipt_sha256 = require_sha256(
            args.expected_stage_a_capture_provenance_receipt_sha256,
            context="--expected-stage-a-capture-provenance-receipt-sha256",
        )
        calibration_binding_artifact = args.calibration_binding.read_bytes()
        deserialize_stage_a_calibration_binding_artifact(calibration_binding_artifact)
        stage_a_capture_provenance_receipt = args.stage_a_capture_provenance_receipt.read_bytes()
        verified_stage_a_capture = deserialize_stage_a_capture_provenance_receipt(
            stage_a_capture_provenance_receipt,
            expected_file_sha256=expected_stage_a_capture_provenance_receipt_sha256,
            calibration_binding_artifact=calibration_binding_artifact,
        )
    elif args.calibration_binding is not None:
        raise ValueError("--calibration-binding is valid only for Stage A")
    raw_input = args.input.read_bytes()
    if verified_stage_a_capture is not None and sha256_bytes(raw_input) != (
        verified_stage_a_capture.identity_input_file_sha256
    ):
        raise ValueError("Stage-A identity input differs from finalized capture provenance")
    source = _json_without_duplicate_keys(raw_input, context="identity input")
    if source.get("phase") != args.phase:
        raise ValueError("input phase does not match --phase")
    candidate = build_candidate(
        source,
        expected_revisions=revisions,  # type: ignore[arg-type]
        calibration_binding_artifact=calibration_binding_artifact,
        stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
        expected_stage_a_capture_provenance_receipt_sha256=(
            expected_stage_a_capture_provenance_receipt_sha256
        ),
    )
    payload = canonical_json_bytes(candidate)
    digest = sha256_bytes(payload)
    if args.dry_run:
        if args.output is not None:
            raise ValueError("--dry-run forbids --output")
        print(digest)
        return 0
    if args.output is None:
        raise ValueError("candidate resolution requires --output or --dry-run")
    validate_quarantine_output(args.output)
    atomic_write(args.output, payload)
    print(digest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, PermissionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
