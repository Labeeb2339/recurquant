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
import os
import string
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

INPUT_SCHEMA: Final = "recurquant.experiment013.identity-input.v4"
CANDIDATE_SCHEMA: Final = "recurquant.experiment013.identity-candidate.v4"
FROZEN_SCHEMA: Final = "recurquant.experiment013.identity-frozen.v4"
ARTIFACT_KIND: Final = "recurquant_static_rht_q468_identity"
RESOLVER_VERSION: Final = 4
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

PG19_TRAIN_NAMESPACE: Final = "recurquant.experiment013.pg19.train.v1\0"
PG19_VALIDATION_NAMESPACE: Final = "recurquant.experiment013.pg19.validation.v1\0"
PG19_TEST_NAMESPACE: Final = "recurquant.experiment013.pg19.test.v1\0"
HUMANEVAL_AB_NAMESPACE: Final = "recurquant.experiment013.humaneval-plus.stage-a-b.v1\0"
HUMANEVAL_C_NAMESPACE: Final = "recurquant.experiment013.humaneval-plus.stage-c.v1\0"
CALIBRATION_SPLIT_NAMESPACE: Final = "recurquant.experiment013.calibration-split.v1\0"
IDENTITY_RECORD_NAMESPACE: Final = "recurquant.experiment013.identity-record.v1\0"
RULER_CALIBRATION_SELECTION_NAMESPACE: Final = (
    "recurquant.experiment013.ruler.calibration-sequence.v1\0"
)
RULER_STAGE_A_SELECTION_NAMESPACE: Final = "recurquant.experiment013.ruler.stage-a-sequence.v1\0"
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
    ("retrieval", "niah_multiquery", 4_096, 2_339),
    ("multi_hop_tracing", "vt", 4_096, 2_339),
    ("aggregation", "fwe", 4_096, 2_339),
    ("question_answering", "qa_1", 4_096, 2_339),
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
CALIBRATION_BINDING_FIELDS: Final = frozenset(
    {
        "calibration_identity_file_sha256",
        "calibration_score_artifact_file_sha256",
        "split_half_stability_artifact_file_sha256",
        "static_k27030_policy_file_sha256",
        "static_k29334_policy_file_sha256",
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
FROZEN_RECORD_FIELDS: Final = RECORD_FIELDS | {
    "anchor_positions",
    "anchor_positions_sha256",
}
STAGE_A_BINDING_ARTIFACT_KIND: Final = "recurquant_experiment013_stage_a_calibration_binding"
STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION: Final = 2
STAGE_A_BINDING_ARTIFACT_REVISION: Final = "experiment-013-stage-a-calibration-binding-v2"


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository's deterministic JSON representation."""

    return (
        json.dumps(
            value,
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


def require_string(value: object, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or value != value.strip():
        raise ValueError(f"{context} must be stripped NFC text")
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
    value: object, *, expected_revisions: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
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
    raw_files = require_sequence(tokenizer["files"], context="tokenizer.files")
    if not raw_files:
        raise ValueError("tokenizer.files cannot be empty")
    files: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_files):
        item = require_mapping(raw, context=f"tokenizer.files[{index}]")
        require_exact_fields(item, TOKENIZER_FILE_FIELDS, context=f"tokenizer.files[{index}]")
        name = require_string(item["name"], context=f"tokenizer.files[{index}].name")
        if Path(name).name != name or name in names:
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
        if configured_length is None or generator_receipt_sha256 is None:
            raise ValueError(
                f"records[{index}] RULER configured length and generator receipt are required"
            )
    elif any(
        value is not None for value in (configured_length, ruler_category, generator_receipt_sha256)
    ):
        raise ValueError(f"records[{index}] non-RULER rows cannot carry RULER-only fields")
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
    identities = [
        (
            row["family"],
            row["canonical_id"],
            row["ruler_category"],
            row["config"],
            row["configured_length"],
            row["seed"],
        )
        for row in records
    ]
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
    identities = [
        (
            row["family"],
            row["canonical_id"],
            row["ruler_category"],
            row["config"],
        )
        for row in records
    ]
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


def build_candidate(
    source: Mapping[str, Any],
    *,
    expected_revisions: Mapping[str, str],
    calibration_binding_artifact: bytes | None = None,
) -> dict[str, Any]:
    """Validate metadata and return a deterministic candidate artifact."""

    phase = source.get("phase")
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
    require_exact_fields(source, frozenset(expected_fields), context="identity input")
    if source["schema"] != INPUT_SCHEMA:
        raise ValueError("identity input schema drifted")
    if phase not in ALLOWED_PHASES:
        if phase in PROTECTED_STAGES:
            raise PermissionError(f"{phase} is protected and unavailable in resolver v4")
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if source["model_weights_loaded"] is not False:
        raise ValueError("identity resolution must occur before model weights")
    expected_calibration_binding: dict[str, str] | None = None
    if phase == "stage_a":
        if not isinstance(calibration_binding_artifact, bytes):
            raise ValueError("Stage A requires a verified calibration binding artifact")
        expected_calibration_binding = dict(
            deserialize_stage_a_calibration_binding_artifact(calibration_binding_artifact).binding
        )
    elif calibration_binding_artifact is not None:
        raise ValueError("calibration resolution forbids a Stage-A binding artifact")
    if set(expected_revisions) != set(DATASET_KEYS):
        raise ValueError("all four dataset revisions are mandatory")
    revisions = {
        key: require_exact_revision(value, context=f"CLI {key} revision")
        for key, value in expected_revisions.items()
    }
    if revisions != FROZEN_DATASET_REVISIONS:
        raise ValueError("CLI dataset revisions do not match the frozen upstream commits")
    datasets = _validate_dataset_contracts(source["datasets"], expected_revisions=revisions)
    tokenizer = _validate_tokenizer(source["tokenizer"])
    execution_bindings = _validate_execution_bindings(source["execution_bindings"])
    parquet_materialization_manifest_file_sha256 = execution_bindings[
        "parquet_materialization_manifest_file_sha256"
    ]
    if (
        parquet_materialization_manifest_file_sha256
        != PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Parquet materialization manifest file SHA-256 drifted")
    raw_records = require_sequence(source["records"], context="records")
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
        _validate_stage_a_records(records)
        split_half = None
        calibration_binding = _validate_calibration_binding(source["calibration_binding"])
        if calibration_binding != expected_calibration_binding:
            raise ValueError("Stage-A input calibration binding differs from the verified artifact")

    content_manifest_hash = sha256_bytes(canonical_json_bytes(records))
    source_hash = sha256_bytes(canonical_json_bytes(source))
    evidence: dict[str, Any] = {
        "schema_version": 4,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": CANDIDATE_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "candidate",
        "phase": phase,
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_manifest_sha256": source_hash,
        "execution_bindings": execution_bindings,
        "model_contracts": {
            "primary": {"id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
            "conditional_scale_check": {
                "id": CONDITIONAL_MODEL_ID,
                "revision": CONDITIONAL_MODEL_REVISION,
                "cold_start_peak_hbm_limit_bytes": 8_053_063_680,
            },
            "weights_loaded": False,
        },
        "datasets": list(datasets),
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
        "tokenizer": tokenizer,
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
        "calibration_split_half": split_half,
        "calibration_binding": calibration_binding,
        "protected_identity": {
            "stage_b_read": False,
            "stage_c_read": False,
            "ordinary_tests_may_read_protected_content": False,
        },
        "promotion_required": True,
    }
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
    require_exact_fields(
        evidence,
        CANDIDATE_EVIDENCE_FIELDS,
        context="candidate evidence",
    )
    if artifact.get("canonical_evidence_sha256") != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("candidate canonical evidence SHA-256 drifted")
    phase = evidence["phase"]
    exact_scalars = {
        "schema_version": 4,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": CANDIDATE_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "candidate",
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "promotion_required": True,
    }
    for name, expected in exact_scalars.items():
        if (
            isinstance(expected, (bool, int))
            and type(evidence[name]) is not type(expected)
        ) or evidence[name] != expected:
            raise ValueError(f"candidate {name} drifted")
    if phase not in ALLOWED_PHASES:
        raise ValueError("candidate phase drifted")
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
        verified_binding = deserialize_stage_a_calibration_binding_artifact(
            calibration_binding_artifact
        ).binding
        if candidate["evidence"]["calibration_binding"] != verified_binding:
            raise ValueError("Stage-A candidate differs from the verified calibration binding")
    elif calibration_binding_artifact is not None:
        raise ValueError("calibration promotion forbids a Stage-A binding artifact")
    evidence = deepcopy(dict(candidate["evidence"]))
    evidence["identity_schema"] = FROZEN_SCHEMA
    evidence["status"] = "frozen"
    evidence["promotion_required"] = False
    evidence["promotion"] = {
        "candidate_file_sha256": expected_candidate_file_sha256,
        "candidate_canonical_evidence_sha256": candidate["canonical_evidence_sha256"],
        "explicit": True,
    }
    return {
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }


class FrozenCalibrationIdentityArtifact:
    """Strictly verified frozen calibration identity and its binding commitments."""

    __slots__ = (
        "file_sha256",
        "canonical_evidence_sha256",
        "records",
        "assignment",
        "assignment_sha256",
        "tokenizer_manifest_sha256",
        "parquet_materialization_manifest_file_sha256",
        "execution_bindings",
    )

    def __init__(
        self,
        *,
        file_sha256: str,
        canonical_evidence_sha256: str,
        records: tuple[dict[str, Any], ...],
        assignment: tuple[dict[str, Any], ...],
        assignment_sha256: str,
        tokenizer_manifest_sha256: str,
        parquet_materialization_manifest_file_sha256: str,
        execution_bindings: dict[str, str],
    ) -> None:
        self.file_sha256 = file_sha256
        self.canonical_evidence_sha256 = canonical_evidence_sha256
        self.records = records
        self.assignment = assignment
        self.assignment_sha256 = assignment_sha256
        self.tokenizer_manifest_sha256 = tokenizer_manifest_sha256
        self.parquet_materialization_manifest_file_sha256 = (
            parquet_materialization_manifest_file_sha256
        )
        self.execution_bindings = execution_bindings


class FrozenStageAIdentityArtifact:
    """Strictly verified frozen Stage-A identity and five-file calibration binding."""

    __slots__ = (
        "file_sha256",
        "canonical_evidence_sha256",
        "records",
        "tokenizer_manifest_sha256",
        "calibration_binding",
        "parquet_materialization_manifest_file_sha256",
        "execution_bindings",
    )

    def __init__(
        self,
        *,
        file_sha256: str,
        canonical_evidence_sha256: str,
        records: tuple[dict[str, Any], ...],
        tokenizer_manifest_sha256: str,
        calibration_binding: dict[str, str],
        parquet_materialization_manifest_file_sha256: str,
        execution_bindings: dict[str, str],
    ) -> None:
        self.file_sha256 = file_sha256
        self.canonical_evidence_sha256 = canonical_evidence_sha256
        self.records = records
        self.tokenizer_manifest_sha256 = tokenizer_manifest_sha256
        self.calibration_binding = calibration_binding
        self.parquet_materialization_manifest_file_sha256 = (
            parquet_materialization_manifest_file_sha256
        )
        self.execution_bindings = execution_bindings


class StageACalibrationBindingArtifact:
    """Verified five-field Stage-A binding and authenticated dependency hashes."""

    __slots__ = (
        "binding",
        "dependency_file_sha256",
        "canonical_evidence_sha256",
        "file_sha256",
    )

    def __init__(
        self,
        *,
        binding: dict[str, str],
        dependency_file_sha256: dict[str, str],
        canonical_evidence_sha256: str,
        file_sha256: str,
    ) -> None:
        self.binding = binding
        self.dependency_file_sha256 = dependency_file_sha256
        self.canonical_evidence_sha256 = canonical_evidence_sha256
        self.file_sha256 = file_sha256


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
        "schema_version": 4,
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
            isinstance(expected, (bool, int))
            and type(evidence[name]) is not type(expected)
        ) or evidence[name] != expected:
            raise ValueError(f"frozen identity {name} drifted")
    require_sha256(
        evidence["source_manifest_sha256"],
        context="frozen identity source manifest SHA-256",
    )
    execution_bindings = _validate_execution_bindings(evidence["execution_bindings"])
    if dict(evidence["execution_bindings"]) != execution_bindings:
        raise ValueError("frozen execution bindings are not canonical")
    parquet_manifest_sha256 = execution_bindings[
        "parquet_materialization_manifest_file_sha256"
    ]
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
    expected_file_sha256: str | None = None,
) -> FrozenStageAIdentityArtifact:
    """Decode Stage A and reauthenticate both its promotion and calibration chain."""

    if not isinstance(data, bytes):
        raise TypeError("frozen Stage-A identity artifact must be bytes")
    if not isinstance(calibration_binding_artifact, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
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
        FROZEN_EVIDENCE_FIELDS,
        context="frozen Stage-A identity evidence",
    )
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="frozen Stage-A canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("frozen Stage-A canonical evidence SHA-256 drifted")
    if (
        evidence["identity_schema"] != FROZEN_SCHEMA
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
            }
        ),
        context="frozen Stage-A promotion",
    )
    if promotion["explicit"] is not True:
        raise ValueError("frozen Stage-A identity was not explicitly promoted")
    candidate_evidence = deepcopy(dict(evidence))
    candidate_evidence.pop("promotion")
    candidate_evidence["identity_schema"] = CANDIDATE_SCHEMA
    candidate_evidence["status"] = "candidate"
    candidate_evidence["promotion_required"] = True
    candidate_canonical_sha256 = sha256_bytes(canonical_json_bytes(candidate_evidence))
    candidate_document = {
        "canonical_evidence_sha256": candidate_canonical_sha256,
        "evidence": candidate_evidence,
    }
    validate_candidate_artifact(candidate_document)
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

    verified_binding = deserialize_stage_a_calibration_binding_artifact(
        calibration_binding_artifact
    ).binding
    if candidate_evidence["calibration_binding"] != verified_binding:
        raise ValueError("frozen Stage-A identity differs from the verified calibration binding")
    records = tuple(dict(record) for record in candidate_evidence["records"])
    tokenizer_manifest_sha256 = require_sha256(
        candidate_evidence["tokenizer"]["file_manifest_sha256"],
        context="frozen Stage-A tokenizer manifest SHA-256",
    )
    execution_bindings = _validate_execution_bindings(candidate_evidence["execution_bindings"])
    parquet_manifest_sha256 = execution_bindings[
        "parquet_materialization_manifest_file_sha256"
    ]
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
) -> tuple[dict[str, str], dict[str, str]]:
    from recurquant.static_q468 import (
        FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        FROZEN_STATIC_Q468_ABLATION_STEPS,
        FROZEN_STATIC_Q468_PRIMARY_STEPS,
        STATIC_Q468_ABLATION_METHOD,
        STATIC_Q468_PRIMARY_METHOD,
        deserialize_static_rht_q468_policy,
    )
    from recurquant.static_q468_calibration import (
        CALIBRATION_SCORE_ARTIFACT_KIND,
        calibration_identity_record_manifest_sha256,
        deserialize_calibration_score_artifact,
        deserialize_frozen_split_half_stability_artifact,
    )

    identity = deserialize_frozen_calibration_identity_artifact(frozen_identity_artifact)
    scores = deserialize_calibration_score_artifact(calibration_score_artifact)
    if scores.artifact_kind != CALIBRATION_SCORE_ARTIFACT_KIND:
        raise ValueError("Stage-A binding requires the official frozen score artifact")
    expected_identity_manifest = calibration_identity_record_manifest_sha256(identity.records)
    if scores.calibration_identity_sha256 != identity.file_sha256:
        raise ValueError("score artifact is not bound to the frozen identity file")
    if scores.aggregate.identity_record_manifest_sha256 != expected_identity_manifest:
        raise ValueError("score identity-record manifest differs from the frozen identity")
    split = deserialize_frozen_split_half_stability_artifact(
        split_half_stability_artifact,
        expected_identity_file_sha256=identity.file_sha256,
        expected_canonical_identity_sha256=identity.canonical_evidence_sha256,
        expected_resolver_assignment_sha256=identity.assignment_sha256,
    )
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
            policy.calibration_manifest_sha256 != scores.aggregate.sequence_score_manifest_sha256
            or policy.calibration_scores_sha256 != scores.calibration_scores_sha256
        ):
            raise ValueError(f"policy {method_id} differs from official calibration scores")
        if steps not in allocations:
            raise ValueError(f"official score artifact is missing exact K{steps}")
        allocation_codes, allocation_hash = allocations[steps]
        if policy.code_map_sha256 != allocation_hash or not __import__("torch").equal(
            policy.precision_codes().reshape(-1).to("cpu"),
            allocation_codes,
        ):
            raise ValueError(f"policy {method_id} code map differs from exact allocation")
    if policy27030.source_commit != policy29334.source_commit:
        raise ValueError("K27030 and K29334 policies must share one source commit")

    binding = {
        "calibration_identity_file_sha256": identity.file_sha256,
        "calibration_score_artifact_file_sha256": scores.file_sha256,
        "split_half_stability_artifact_file_sha256": split.file_sha256,
        "static_k27030_policy_file_sha256": sha256_bytes(static_k27030_policy_artifact),
        "static_k29334_policy_file_sha256": sha256_bytes(static_k29334_policy_artifact),
    }
    dependency_hashes = {
        "calibration_score_artifact": scores.file_sha256,
        "frozen_identity_artifact": identity.file_sha256,
        "split_half_stability_artifact": split.file_sha256,
        "static_k27030_policy_artifact": sha256_bytes(static_k27030_policy_artifact),
        "static_k29334_policy_artifact": sha256_bytes(static_k29334_policy_artifact),
    }
    return binding, dependency_hashes


def build_stage_a_calibration_binding_artifact(
    *,
    frozen_identity_artifact: bytes,
    calibration_score_artifact: bytes,
    split_half_stability_artifact: bytes,
    static_k27030_policy_artifact: bytes,
    static_k29334_policy_artifact: bytes,
) -> bytes:
    """Build the five-field Stage-A binding only from fully verified dependencies."""

    dependencies = {
        "calibration_score_artifact": calibration_score_artifact,
        "frozen_identity_artifact": frozen_identity_artifact,
        "split_half_stability_artifact": split_half_stability_artifact,
        "static_k27030_policy_artifact": static_k27030_policy_artifact,
        "static_k29334_policy_artifact": static_k29334_policy_artifact,
    }
    binding, dependency_hashes = _derive_stage_a_calibration_binding(
        frozen_identity_artifact=frozen_identity_artifact,
        calibration_score_artifact=calibration_score_artifact,
        split_half_stability_artifact=split_half_stability_artifact,
        static_k27030_policy_artifact=static_k27030_policy_artifact,
        static_k29334_policy_artifact=static_k29334_policy_artifact,
    )
    evidence = {
        "artifact_revision": STAGE_A_BINDING_ARTIFACT_REVISION,
        "binding": binding,
        "dependencies_base64": {
            name: _canonical_b64(value, context=name)
            for name, value in sorted(dependencies.items())
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
    """Strictly reverify every embedded dependency of a Stage-A binding artifact."""

    if not isinstance(data, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    file_sha256 = sha256_bytes(data)
    if expected_file_sha256 is not None and file_sha256 != require_sha256(
        expected_file_sha256,
        context="expected Stage-A binding file SHA-256",
    ):
        raise ValueError("Stage-A calibration binding file SHA-256 mismatch")
    root = _json_without_duplicate_keys(data, context="Stage-A calibration binding")
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
            {
                "artifact_revision",
                "binding",
                "dependencies_base64",
                "dependency_file_sha256",
            }
        ),
        context="Stage-A binding evidence",
    )
    if evidence["artifact_revision"] != STAGE_A_BINDING_ARTIFACT_REVISION:
        raise ValueError("Stage-A calibration binding revision drifted")
    canonical_evidence_sha256 = require_sha256(
        root["canonical_evidence_sha256"],
        context="Stage-A binding canonical evidence SHA-256",
    )
    if canonical_evidence_sha256 != sha256_bytes(canonical_json_bytes(evidence)):
        raise ValueError("Stage-A binding canonical evidence SHA-256 drifted")
    encoded_dependencies = require_mapping(
        evidence["dependencies_base64"],
        context="Stage-A binding dependencies",
    )
    dependency_names = frozenset(
        {
            "calibration_score_artifact",
            "frozen_identity_artifact",
            "split_half_stability_artifact",
            "static_k27030_policy_artifact",
            "static_k29334_policy_artifact",
        }
    )
    require_exact_fields(
        encoded_dependencies,
        dependency_names,
        context="Stage-A binding dependencies",
    )
    dependencies = {
        name: _decode_canonical_b64(
            encoded_dependencies[name],
            context=f"Stage-A dependency {name}",
        )
        for name in sorted(dependency_names)
    }
    binding, dependency_hashes = _derive_stage_a_calibration_binding(
        frozen_identity_artifact=dependencies["frozen_identity_artifact"],
        calibration_score_artifact=dependencies["calibration_score_artifact"],
        split_half_stability_artifact=dependencies["split_half_stability_artifact"],
        static_k27030_policy_artifact=dependencies["static_k27030_policy_artifact"],
        static_k29334_policy_artifact=dependencies["static_k29334_policy_artifact"],
    )
    if evidence["binding"] != binding:
        raise ValueError("Stage-A calibration binding fields drifted")
    if evidence["dependency_file_sha256"] != dependency_hashes:
        raise ValueError("Stage-A calibration dependency hashes drifted")
    return StageACalibrationBindingArtifact(
        binding=binding,
        dependency_file_sha256=dependency_hashes,
        canonical_evidence_sha256=canonical_evidence_sha256,
        file_sha256=file_sha256,
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
    parser.add_argument("--mbpp-revision")
    parser.add_argument("--pg19-revision")
    parser.add_argument("--ruler-revision")
    parser.add_argument("--humaneval-plus-revision")
    return parser.parse_args(argv)


def _reject_protected_before_input(phase: str) -> None:
    if phase in PROTECTED_STAGES:
        raise PermissionError(
            f"{phase} is protected; resolver v3 refuses it before reading --input"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _reject_protected_before_input(args.phase)
    if args.promote:
        if args.dry_run or args.output is None:
            raise ValueError("promotion requires --output and forbids --dry-run")
        calibration_binding_artifact: bytes | None = None
        if args.phase == "stage_a":
            if args.calibration_binding is None:
                raise ValueError("Stage-A promotion requires --calibration-binding")
            calibration_binding_artifact = args.calibration_binding.read_bytes()
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
    source = _json_without_duplicate_keys(args.input.read_bytes(), context="identity input")
    if source.get("phase") != args.phase:
        raise ValueError("input phase does not match --phase")
    calibration_binding_artifact: bytes | None = None
    if args.phase == "stage_a":
        if args.calibration_binding is None:
            raise ValueError("Stage A requires --calibration-binding")
        calibration_binding_artifact = args.calibration_binding.read_bytes()
    elif args.calibration_binding is not None:
        raise ValueError("--calibration-binding is valid only for Stage A")
    candidate = build_candidate(
        source,
        expected_revisions=revisions,  # type: ignore[arg-type]
        calibration_binding_artifact=calibration_binding_artifact,
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
