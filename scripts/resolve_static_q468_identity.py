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

INPUT_SCHEMA: Final = "recurquant.experiment013.identity-input.v1"
CANDIDATE_SCHEMA: Final = "recurquant.experiment013.identity-candidate.v1"
FROZEN_SCHEMA: Final = "recurquant.experiment013.identity-frozen.v1"
ARTIFACT_KIND: Final = "recurquant_static_rht_q468_identity"
RESOLVER_VERSION: Final = 1

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
PG19_VALIDATION_NAMESPACE: Final = (
    "recurquant.experiment013.pg19.validation.v1\0"
)
PG19_TEST_NAMESPACE: Final = "recurquant.experiment013.pg19.test.v1\0"
HUMANEVAL_AB_NAMESPACE: Final = (
    "recurquant.experiment013.humaneval-plus.stage-a-b.v1\0"
)
HUMANEVAL_C_NAMESPACE: Final = (
    "recurquant.experiment013.humaneval-plus.stage-c.v1\0"
)
CALIBRATION_SPLIT_NAMESPACE: Final = (
    "recurquant.experiment013.calibration-split.v1\0"
)

CLAIM_BOUNDARY: Final = (
    "This artifact freezes Experiment 013 data and tokenizer identity only. "
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
        "sequence_length",
        "source_content_sha256",
        "formatted_content_sha256",
        "prompt_token_ids_sha256",
        "target_token_ids_sha256",
        "tokenizer_manifest_sha256",
        "token_span",
        "anchor_manifest_sha256",
    }
)
TOKEN_SPAN_FIELDS: Final = frozenset(
    {"prefill_start", "prefill_stop", "scored_start", "scored_stop"}
)
CALIBRATION_BINDING_FIELDS: Final = frozenset(
    {
        "identity_file_sha256",
        "canonical_identity_sha256",
        "static_k29334_code_map_sha256",
        "static_k27030_code_map_sha256",
        "split_half_policy_manifest_sha256",
    }
)


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
        raise ValueError(
            f"{context} must be an immutable lowercase 40- or 64-hex revision"
        )
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


def selection_sha256(namespace: str, canonical_id: str) -> str:
    return sha256_bytes(namespace.encode("utf-8") + canonical_id.encode("utf-8"))


def mbpp_selection_sha256(canonical_id: str) -> str:
    return sha256_bytes(f"{MBPP_SELECTION_NAMESPACE}|{canonical_id}".encode())


def calibration_split_key(record: Mapping[str, Any]) -> str:
    identity = "\0".join(
        (
            str(record["family"]),
            str(record["config"]),
            str(record["canonical_id"]),
            str(record["seed"]),
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


def _validate_sha_rank_order(
    rows: Sequence[Mapping[str, Any]], *, context: str
) -> None:
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
        revision = require_exact_revision(
            item["revision"], context=f"datasets[{index}].revision"
        )
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
            "split": require_string(
                item["split"], context=f"datasets[{index}].split"
            ),
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
                f"{key} canonical ID field must be "
                f"{FROZEN_CANONICAL_ID_FIELDS[key]!r}"
            )
        if key == "mbpp" and (
            contract["dataset_id"] != MBPP_DATASET_ID
            or contract["config"] != MBPP_CONFIG
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
        require_exact_fields(
            item, TOKENIZER_FILE_FIELDS, context=f"tokenizer.files[{index}]"
        )
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
        return CALIBRATION_SPLIT_NAMESPACE
    if phase == "stage_a":
        if family == "pg19":
            return PG19_VALIDATION_NAMESPACE
        if family == "humaneval_plus":
            return HUMANEVAL_AB_NAMESPACE
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
    canonical_id = require_string(
        item["canonical_id"], context=f"records[{index}].canonical_id"
    )
    config = require_string(
        item["config"], context=f"records[{index}].config", allow_empty=True
    )
    rank = require_int(item["selection_rank"], context=f"records[{index}].selection_rank")
    seed_value = item["seed"]
    seed = None if seed_value is None else require_int(
        seed_value, context=f"records[{index}].seed"
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
        for name in ("prefill_start", "prefill_stop", "scored_start", "scored_stop")
    }
    if (
        normalized_span["prefill_start"] != 0
        or normalized_span["prefill_stop"] != normalized_span["scored_start"]
        or normalized_span["prefill_stop"] < 1
        or normalized_span["scored_stop"] < normalized_span["scored_start"]
    ):
        raise ValueError(f"records[{index}] token span is not contiguous and canonical")
    positions = anchor_positions(sequence_length)
    return {
        "family": family,
        "canonical_id": canonical_id,
        "config": config,
        "selection_rank": rank,
        "selection_sha256": expected_selection,
        "seed": seed,
        "sequence_length": sequence_length,
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
        "tokenizer_manifest_sha256": tokenizer_hash,
        "token_span": normalized_span,
        "anchor_manifest_sha256": require_sha256(
            item["anchor_manifest_sha256"],
            context=f"records[{index}].anchor_manifest_sha256",
        ),
        "anchor_positions": list(positions),
        "anchor_positions_sha256": sha256_bytes(canonical_json_bytes(positions)),
    }


def _record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        FAMILY_ORDER[str(record["family"])],
        int(record["selection_rank"]),
        str(record["selection_sha256"]),
        str(record["canonical_id"]),
        str(record["config"]),
        -1 if record["seed"] is None else int(record["seed"]),
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
    actual_mbpp = sorted(
        grouped["mbpp"], key=lambda row: int(row["selection_rank"])
    )
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
        }:
            raise ValueError("calibration PG19 span must cover exactly 2,304 tokens")
    ruler_tuples = {
        (str(row["config"]), int(row["sequence_length"]), int(row["seed"]))
        for row in grouped["ruler"]
    }
    if len({row["config"] for row in grouped["ruler"]}) != 4 or len(ruler_tuples) != 16:
        raise ValueError("calibration RULER must contain four unique official families")
    expected_pairs = {(length, seed) for length in (2_048, 4_096) for seed in (12_339, 12_340)}
    for config in {str(row["config"]) for row in grouped["ruler"]}:
        actual_pairs = {
            (int(row["sequence_length"]), int(row["seed"]))
            for row in grouped["ruler"]
            if row["config"] == config
        }
        if actual_pairs != expected_pairs:
            raise ValueError(f"RULER family {config!r} does not have the frozen grid")
    for row in grouped["mbpp"]:
        if row["seed"] is not None:
            raise ValueError("MBPP calibration records cannot have a generator seed")
    identities = [(row["family"], row["canonical_id"], row["config"]) for row in records]
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
            raise ValueError("Stage-A PG19 must use 4,096 prefill plus 128 scored tokens")
        if span != {
            "prefill_start": 0,
            "prefill_stop": 4_096,
            "scored_start": 4_096,
            "scored_stop": 4_224,
        }:
            raise ValueError("Stage-A PG19 token span drifted")
    if len({row["config"] for row in grouped["ruler"]}) != 4:
        raise ValueError("Stage-A RULER must contain four distinct official families")
    for row in grouped["ruler"]:
        span = row["token_span"]
        if row["seed"] != 2_339 or row["sequence_length"] != 4_096:
            raise ValueError("Stage-A RULER must use length 4,096 and seed 2,339")
        if span["scored_stop"] <= span["scored_start"]:
            raise ValueError("Stage-A RULER answer span cannot be empty")
    for row in grouped["humaneval_plus"]:
        span = row["token_span"]
        scored = span["scored_stop"] - span["scored_start"]
        if row["seed"] is not None or not 1 <= scored <= 128:
            raise ValueError("Stage-A HumanEval+ must score 1..128 solution tokens")
        if row["sequence_length"] != span["scored_stop"]:
            raise ValueError("Stage-A HumanEval+ sequence length must equal span stop")
    identities = [(row["family"], row["canonical_id"], row["config"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage-A canonical identities are not unique")


def _split_half_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    groups.append(("mbpp", [row for row in records if row["family"] == "mbpp"]))
    groups.append(("pg19", [row for row in records if row["family"] == "pg19"]))
    ruler_rows = [row for row in records if row["family"] == "ruler"]
    for config in sorted({str(row["config"]) for row in ruler_rows}):
        groups.append((f"ruler:{config}", [row for row in ruler_rows if row["config"] == config]))
    for group, rows in groups:
        ranked = sorted(rows, key=lambda row: (calibration_split_key(row), _record_sort_key(row)))
        for rank, row in enumerate(ranked):
            assignments.append(
                {
                    "group": group,
                    "canonical_id": row["canonical_id"],
                    "config": row["config"],
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
    require_exact_fields(
        binding, CALIBRATION_BINDING_FIELDS, context="calibration_binding"
    )
    return {
        key: require_sha256(binding[key], context=f"calibration_binding.{key}")
        for key in sorted(CALIBRATION_BINDING_FIELDS)
    }


def build_candidate(
    source: Mapping[str, Any], *, expected_revisions: Mapping[str, str]
) -> dict[str, Any]:
    """Validate metadata and return a deterministic candidate artifact."""

    phase = source.get("phase")
    expected_fields = {
        "schema",
        "phase",
        "datasets",
        "tokenizer",
        "records",
        "model_weights_loaded",
    }
    if phase == "stage_a":
        expected_fields.add("calibration_binding")
    require_exact_fields(source, frozenset(expected_fields), context="identity input")
    if source["schema"] != INPUT_SCHEMA:
        raise ValueError("identity input schema drifted")
    if phase not in ALLOWED_PHASES:
        if phase in PROTECTED_STAGES:
            raise PermissionError(f"{phase} is protected and unavailable in resolver v1")
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if source["model_weights_loaded"] is not False:
        raise ValueError("identity resolution must occur before model weights")
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

    content_manifest_hash = sha256_bytes(canonical_json_bytes(records))
    source_hash = sha256_bytes(canonical_json_bytes(source))
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "identity_schema": CANDIDATE_SCHEMA,
        "resolver_version": RESOLVER_VERSION,
        "status": "candidate",
        "phase": phase,
        "identity_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_manifest_sha256": source_hash,
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
    if artifact.get("canonical_evidence_sha256") != sha256_bytes(
        canonical_json_bytes(evidence)
    ):
        raise ValueError("candidate canonical evidence SHA-256 drifted")
    if (
        evidence.get("identity_schema") != CANDIDATE_SCHEMA
        or evidence.get("status") != "candidate"
        or evidence.get("phase") not in ALLOWED_PHASES
        or evidence.get("identity_only") is not True
        or evidence.get("claim_boundary") != CLAIM_BOUNDARY
        or evidence.get("promotion_required") is not True
    ):
        raise ValueError("candidate identity contract drifted")
    models = require_mapping(evidence.get("model_contracts"), context="model contracts")
    if models.get("weights_loaded") is not False:
        raise ValueError("candidate claims model weights were loaded")
    records = require_sequence(evidence.get("records"), context="candidate records")
    if evidence.get("record_count") != len(records):
        raise ValueError("candidate record count drifted")
    if evidence.get("content_manifest_sha256") != sha256_bytes(
        canonical_json_bytes(records)
    ):
        raise ValueError("candidate content manifest SHA-256 drifted")
    protected = require_mapping(
        evidence.get("protected_identity"), context="protected identity"
    )
    if protected != {
        "stage_b_read": False,
        "stage_c_read": False,
        "ordinary_tests_may_read_protected_content": False,
    }:
        raise ValueError("protected identity boundary drifted")


def promote_candidate(
    candidate: Mapping[str, Any], *, candidate_file_sha256: str
) -> dict[str, Any]:
    """Create a deterministic frozen identity from an authenticated candidate."""

    validate_candidate_artifact(candidate)
    require_sha256(candidate_file_sha256, context="candidate file SHA-256")
    evidence = deepcopy(dict(candidate["evidence"]))
    evidence["identity_schema"] = FROZEN_SCHEMA
    evidence["status"] = "frozen"
    evidence["promotion_required"] = False
    evidence["promotion"] = {
        "candidate_file_sha256": candidate_file_sha256,
        "candidate_canonical_evidence_sha256": candidate[
            "canonical_evidence_sha256"
        ],
        "explicit": True,
    }
    return {
        "canonical_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
    parser.add_argument("--mbpp-revision")
    parser.add_argument("--pg19-revision")
    parser.add_argument("--ruler-revision")
    parser.add_argument("--humaneval-plus-revision")
    return parser.parse_args(argv)


def _reject_protected_before_input(phase: str) -> None:
    if phase in PROTECTED_STAGES:
        raise PermissionError(
            f"{phase} is protected; resolver v1 refuses it before reading --input"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _reject_protected_before_input(args.phase)
    if args.promote:
        if args.dry_run or args.output is None:
            raise ValueError("promotion requires --output and forbids --dry-run")
        expected_hash = require_sha256(
            args.expected_candidate_sha256,
            context="--expected-candidate-sha256",
        )
        raw = args.input.read_bytes()
        actual_hash = sha256_bytes(raw)
        if actual_hash != expected_hash:
            raise ValueError("candidate file SHA-256 does not match explicit promotion hash")
        candidate = _json_without_duplicate_keys(raw, context="candidate artifact")
        if candidate.get("evidence", {}).get("phase") != args.phase:
            raise ValueError("candidate phase does not match --phase")
        frozen = promote_candidate(candidate, candidate_file_sha256=actual_hash)
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
    candidate = build_candidate(source, expected_revisions=revisions)  # type: ignore[arg-type]
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
