#!/usr/bin/env python3
"""Capture resolver-compatible Experiment 013 identity inputs without model weights.

Only the public calibration and Stage-A identity surfaces are implemented.  Stage B
and Stage C are rejected before a source object, local receipt, dataset, tokenizer,
or output path is touched.  The live source reads pinned dataset projections and
selected public rows, downloads tokenizer assets by an explicit allow-list, and
consumes separately prepared RULER generator receipts.  It never imports a model
class or requests a weight file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import string
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_PATH = REPOSITORY_ROOT / "scripts" / "resolve_static_q468_identity.py"
_RESOLVER_MODULE_NAME = "recurquant_experiment013_identity_resolver"
_RESOLVER_SPEC = importlib.util.spec_from_file_location(_RESOLVER_MODULE_NAME, RESOLVER_PATH)
if _RESOLVER_SPEC is None or _RESOLVER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load the Experiment 013 identity resolver")
_existing_resolver = sys.modules.get(_RESOLVER_MODULE_NAME)
if _existing_resolver is not None:
    existing_path = getattr(_existing_resolver, "__file__", None)
    if existing_path is None or Path(existing_path).resolve() != RESOLVER_PATH.resolve():
        raise RuntimeError("resolver module name is already bound to another file")
    resolver = _existing_resolver
else:
    resolver = importlib.util.module_from_spec(_RESOLVER_SPEC)
    sys.modules[_RESOLVER_MODULE_NAME] = resolver
    try:
        _RESOLVER_SPEC.loader.exec_module(resolver)
    except BaseException:
        sys.modules.pop(_RESOLVER_MODULE_NAME, None)
        raise

CAPTURE_VERSION: Final = 2
RULER_RECEIPT_SCHEMA: Final = "recurquant.experiment013.ruler-receipt.v1"
RULER_SEQUENCE_NAMESPACE: Final = "recurquant.experiment013.ruler.sequence.v1"
PG19_TRAIN_SEGMENT_NAMESPACE: Final = "recurquant.experiment013.pg19.segment.v1\0"
PG19_VALIDATION_SEGMENT_NAMESPACE: Final = "recurquant.experiment013.pg19.validation-segment.v1\0"

SOURCE_HEAD_ORDER: Final = (
    "primary_model",
    "mbpp",
    "pg19",
    "ruler",
    "humaneval_plus",
    "evalplus",
)
EXPECTED_SOURCE_HEADS: Final = {
    "primary_model": resolver.PRIMARY_MODEL_REVISION,
    "mbpp": resolver.MBPP_REVISION,
    "pg19": resolver.PG19_REVISION,
    "ruler": resolver.RULER_REVISION,
    "humaneval_plus": resolver.HUMANEVAL_PLUS_REVISION,
    "evalplus": resolver.EVALPLUS_SOURCE_REVISION,
}

RULER_CONFIGS_BY_CATEGORY: Final = {
    "retrieval": (
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
    ),
    "multi_hop_tracing": ("vt",),
    "aggregation": ("cwe", "fwe"),
    "question_answering": ("qa_1", "qa_2"),
}
RULER_ALL_CONFIGS: Final = tuple(
    config
    for category in resolver.RULER_CATEGORIES
    for config in RULER_CONFIGS_BY_CATEGORY[category]
)
RULER_REQUIRED_OUTPUT_COUNTS: Final = {
    "niah_single_1": 1,
    "niah_single_2": 1,
    "niah_single_3": 1,
    "niah_multikey_1": 1,
    "niah_multikey_2": 1,
    "niah_multikey_3": 1,
    "niah_multivalue": 4,
    "niah_multiquery": 4,
    "vt": 5,
    "cwe": 10,
    "fwe": 3,
}
RULER_REQUIRED_OUTPUT_SEPARATOR: Final = ", "

# Git object IDs from the recursive tree at RULER_REVISION.  Verifying Git's
# blob hash catches an unexpected raw response before it enters the formatter
# identity.  Auxiliary corpora and package resources remain receipt-bound.
RULER_GENERATOR_GIT_BLOBS: Final = {
    "scripts/synthetic.yaml": "29cfa5f60b49a7fa53f8dccbbd4f0c7c9e7834fa",
    "scripts/data/prepare.py": "4d106c46b7faa1deb1540b9e04a8bb2b71c01b4b",
    "scripts/data/template.py": "9bbf7b91382ddb20815315cccad92e50bd95bf7e",
    "scripts/data/tokenizer.py": "5a2ddb504ce26da5b43c0f196629e152fca1460b",
    "scripts/data/manifest_utils.py": "63153f8579e05cbde77006558e09f1990238bd8b",
    "scripts/data/synthetic/constants.py": "e1a880a1d41c953e55236966b7eb7e84174e00cc",
    "scripts/data/synthetic/niah.py": "729eddc260ef5a9aa0473557cd249abca232764a",
    "scripts/data/synthetic/variable_tracking.py": ("bc5dab381f38e810e5050340d8dae29ae1cfc82a"),
    "scripts/data/synthetic/common_words_extraction.py": (
        "af07a9bd76fbbb96910c61b790f1c8e8e944a901"
    ),
    "scripts/data/synthetic/freq_words_extraction.py": ("77ddcd383d698378d2049278657e2b3aad84e3e1"),
    "scripts/data/synthetic/qa.py": "d71cf0355026ab9265dc7f4de14cb04159c62230",
}

TOKENIZER_ASSET_NAMES: Final = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
FORBIDDEN_MODEL_FILE_RE: Final = re.compile(
    r"(?:^|/)(?:model(?:[-.]|$)|pytorch_model|tf_model|flax_model|adapter_model)"
    r"|\.(?:safetensors|bin|pt|pth|ckpt|onnx|gguf)$",
    flags=re.IGNORECASE,
)

MBPP_FORMATTER_SPEC: Final = {
    "id": "recurquant.mbpp-prompt-code.v1",
    "prompt": (
        "You are an expert Python programmer, and here is your task: {text}\\n"
        "Your code should pass these tests:\\n\\n{tests}\\n[BEGIN]\\n"
    ),
    "prompt_add_special_tokens": True,
    "target": "code",
    "target_add_special_tokens": False,
    "normalization": "CRLF_and_CR_to_LF",
}
PG19_FORMATTER_SPEC: Final = {
    "id": "recurquant.pg19-token-slice.v1",
    "canonical_id": "exact UTF-8 url field",
    "add_special_tokens": False,
    "calibration": {
        "tokens": 2_304,
        "namespace": PG19_TRAIN_SEGMENT_NAMESPACE,
    },
    "stage_a": {
        "tokens": 4_224,
        "prefill": 4_096,
        "continuation_tokens": 128,
        "cache_exposed_predictions": 127,
        "namespace": PG19_VALIDATION_SEGMENT_NAMESPACE,
    },
}
HUMANEVAL_FORMATTER_SPEC: Final = {
    "id": "recurquant.humaneval-plus-prompt-solution.v1",
    "canonical_id": "exact task_id field",
    "prompt_field": "prompt",
    "prompt_add_special_tokens": True,
    "target_field": "canonical_solution",
    "target_add_special_tokens": False,
    "target_token_cap": 128,
    "minimum_target_tokens": 2,
    "cache_exposed_predictions": "target_length - 1",
}

RULER_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "source_id",
        "revision",
        "category",
        "config",
        "configured_length",
        "seed",
        "sample_index",
        "generator_reported_length",
        "input",
        "answer_prefix",
        "outputs",
        "auxiliary_files",
    }
)
AUXILIARY_FILE_FIELDS: Final = frozenset({"name", "sha256", "size_bytes"})


def ruler_receipt_filename(*, category: str, config: str, configured_length: int, seed: int) -> str:
    """Return the frozen filename for one separately generated RULER receipt."""

    return f"{category}__{config}__l{configured_length}__s{seed}.json"


def required_ruler_receipts() -> tuple[dict[str, Any], ...]:
    """Return the 16 calibration and four Stage-A receipt identities."""

    result: list[dict[str, Any]] = []
    for phase, schedule in (
        ("calibration", resolver.RULER_CALIBRATION_SCHEDULE),
        ("stage_a", resolver.RULER_STAGE_A_SCHEDULE),
    ):
        for category, config, configured_length, seed in schedule:
            result.append(
                {
                    "phase": phase,
                    "category": category,
                    "config": config,
                    "configured_length": configured_length,
                    "seed": seed,
                    "sample_index": 0,
                    "filename": ruler_receipt_filename(
                        category=category,
                        config=config,
                        configured_length=configured_length,
                        seed=seed,
                    ),
                }
            )
    if len(result) != 20 or len({item["filename"] for item in result}) != 20:
        raise RuntimeError("frozen RULER receipt inventory is not 20 unique files")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProjectionRow:
    canonical_id: str
    offset: int


@dataclass(frozen=True, slots=True)
class TokenizerMaterial:
    tokenizer: Any
    tokenizer_class: str
    transformers_version: str
    files: Mapping[str, bytes]
    model_weights_loaded: bool = False


class CaptureSource(Protocol):
    """Narrow, mockable source surface used by the deterministic capturer."""

    def source_heads(self) -> Mapping[str, str]: ...

    def tokenizer_material(self) -> TokenizerMaterial: ...

    def mbpp_train_rows(self) -> Sequence[Mapping[str, Any]]: ...

    def pg19_projection(self, split: str) -> Sequence[ProjectionRow]: ...

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> Mapping[str, Any]: ...

    def ruler_generator_files(self) -> Mapping[str, bytes]: ...

    def ruler_receipt(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> Mapping[str, Any]: ...

    def humaneval_projection(self) -> Sequence[ProjectionRow]: ...

    def humaneval_row(self, *, offset: int, expected_task_id: str) -> Mapping[str, Any]: ...


def canonical_json_bytes(value: object) -> bytes:
    return resolver.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode()
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _strict_json(raw: bytes, *, context: str) -> dict[str, Any]:
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


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields drifted; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_string(value: object, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a {'string' if allow_empty else 'non-empty string'}")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{context} must be NFC text")
    return value


def _require_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _json_safe(value: object, *, context: str) -> object:
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not canonical-JSON serializable") from error
    return value


def _normalize_lf(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise ValueError("tokenizer input must be text")
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise ValueError("tokenizer.encode must return an integer sequence")
    result: list[int] = []
    for index, token_id in enumerate(encoded):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"tokenizer token {index} is not a non-negative integer")
        result.append(token_id)
    return tuple(result)


def _token_hash(token_ids: Sequence[int]) -> str:
    return resolver.sequence_token_ids_sha256(token_ids)


def _anchor_manifest_hash(
    *, canonical_id: str, sequence_ids: Sequence[int], token_span: Mapping[str, int]
) -> str:
    return resolver.identity_anchor_manifest_sha256(
        canonical_id=canonical_id,
        sequence_length=len(sequence_ids),
        sequence_token_ids_sha256_value=_token_hash(sequence_ids),
        token_span=token_span,
    )


def _base_record(
    *,
    phase: str,
    family: str,
    canonical_id: str,
    config: str,
    seed: int | None,
    configured_length: int | None,
    ruler_category: str | None,
    generator_receipt_sha256: str | None,
    source_payload: object,
    formatted_payload: object,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    tokenizer_manifest_sha256: str,
) -> dict[str, Any]:
    sequence_ids = tuple(prompt_ids) + tuple(target_ids)
    if not sequence_ids:
        raise ValueError(f"{family} record {canonical_id!r} produced no tokens")
    if phase == "stage_a" and len(target_ids) < 2:
        raise ValueError(
            f"Stage-A {family} continuation must contain at least two tokens "
            "to expose one cache prediction"
        )
    scored_stop = len(sequence_ids)
    cache_exposed_start = scored_stop if phase == "calibration" else len(prompt_ids) + 1
    token_span = {
        "prefill_start": 0,
        "prefill_stop": len(prompt_ids),
        "scored_start": len(prompt_ids),
        "scored_stop": scored_stop,
        "cache_exposed_start": cache_exposed_start,
        "cache_exposed_stop": scored_stop,
    }
    namespace = {
        ("calibration", "mbpp"): None,
        ("calibration", "pg19"): resolver.PG19_TRAIN_NAMESPACE,
        ("calibration", "ruler"): resolver.RULER_CALIBRATION_SELECTION_NAMESPACE,
        ("stage_a", "pg19"): resolver.PG19_VALIDATION_NAMESPACE,
        ("stage_a", "ruler"): resolver.RULER_STAGE_A_SELECTION_NAMESPACE,
        ("stage_a", "humaneval_plus"): resolver.HUMANEVAL_AB_NAMESPACE,
    }.get((phase, family))
    if family == "mbpp":
        selection_hash = resolver.mbpp_selection_sha256(canonical_id)
    elif namespace is not None:
        selection_hash = resolver.selection_sha256(namespace, canonical_id)
    else:
        raise ValueError(f"no selection namespace for {phase}/{family}")
    return {
        "family": family,
        "canonical_id": canonical_id,
        "config": config,
        "selection_rank": -1,
        "selection_sha256": selection_hash,
        "seed": seed,
        "configured_length": configured_length,
        "sequence_length": len(sequence_ids),
        "ruler_category": ruler_category,
        "generator_receipt_sha256": generator_receipt_sha256,
        "source_content_sha256": sha256_bytes(canonical_json_bytes(source_payload)),
        "formatted_content_sha256": sha256_bytes(canonical_json_bytes(formatted_payload)),
        "prompt_token_ids_sha256": _token_hash(prompt_ids),
        "target_token_ids_sha256": _token_hash(target_ids),
        "sequence_token_ids_sha256": _token_hash(sequence_ids),
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "token_span": token_span,
        "anchor_manifest_sha256": _anchor_manifest_hash(
            canonical_id=canonical_id,
            sequence_ids=sequence_ids,
            token_span=token_span,
        ),
    }


def _assign_sha_ranks(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (row["selection_sha256"], row["canonical_id"]),
    )
    for rank, row in enumerate(ordered):
        row["selection_rank"] = rank
        row["identity_record_sha256"] = resolver.identity_record_sha256(row)
    return ordered


def _tokenizer_contract(material: TokenizerMaterial) -> tuple[dict[str, Any], str]:
    if material.model_weights_loaded is not False:
        raise ValueError("tokenizer capture reports that model weights were loaded")
    if material.transformers_version != resolver.TRANSFORMERS_VERSION:
        raise ValueError("Transformers version drifted during tokenizer capture")
    if not material.files:
        raise ValueError("tokenizer capture returned no files")
    files: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_name, content in material.files.items():
        name = _require_string(raw_name, context="tokenizer file name")
        if Path(name).name != name or name in names:
            raise ValueError("tokenizer file names must be unique basenames")
        if FORBIDDEN_MODEL_FILE_RE.search(name):
            raise ValueError(f"model weight-like file is forbidden: {name}")
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"tokenizer file {name!r} must contain bytes")
        names.add(name)
        files.append({"name": name, "sha256": sha256_bytes(content), "size_bytes": len(content)})
    files.sort(key=lambda item: item["name"])
    manifest_hash = sha256_bytes(canonical_json_bytes(files))
    return (
        {
            "source_id": resolver.PRIMARY_MODEL_ID,
            "revision": resolver.PRIMARY_MODEL_REVISION,
            "class": _require_string(material.tokenizer_class, context="tokenizer class"),
            "transformers_version": resolver.TRANSFORMERS_VERSION,
            "files": files,
        },
        manifest_hash,
    )


def _validate_heads(heads: Mapping[str, str], *, context: str) -> dict[str, str]:
    if set(heads) != set(SOURCE_HEAD_ORDER):
        raise ValueError(f"{context} source-head fields drifted")
    normalized = {key: str(heads[key]) for key in SOURCE_HEAD_ORDER}
    if normalized != EXPECTED_SOURCE_HEADS:
        drift = {
            key: {"expected": EXPECTED_SOURCE_HEADS[key], "actual": normalized[key]}
            for key in SOURCE_HEAD_ORDER
            if normalized[key] != EXPECTED_SOURCE_HEADS[key]
        }
        raise ValueError(f"{context} source HEAD is not the pinned revision: {drift}")
    return normalized


def _canonical_mbpp_row(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _require_int(row.get("task_id"), context="MBPP task_id", minimum=1)
    text = _normalize_lf(_require_string(row.get("text"), context="MBPP text"))
    code = _normalize_lf(_require_string(row.get("code"), context="MBPP code"))
    tests = row.get("test_list")
    challenges = row.get("challenge_test_list")
    if isinstance(tests, (str, bytes)) or not isinstance(tests, Sequence):
        raise ValueError("MBPP test_list must be a string sequence")
    if isinstance(challenges, (str, bytes)) or not isinstance(challenges, Sequence):
        raise ValueError("MBPP challenge_test_list must be a string sequence")

    def strings(values: Sequence[Any], name: str) -> list[str]:
        result: list[str] = []
        for value in values:
            result.append(_normalize_lf(_require_string(value, context=name)))
        return result

    return {
        "task_id": task_id,
        "text": text,
        "code": code,
        "test_list": strings(tests, "MBPP test_list item"),
        "test_setup_code": _normalize_lf(
            _require_string(
                row.get("test_setup_code"),
                context="MBPP test_setup_code",
                allow_empty=True,
            )
        ),
        "challenge_test_list": strings(challenges, "MBPP challenge item"),
    }


def _capture_mbpp(
    source: CaptureSource,
    *,
    phase: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    _selected_ids, population_hash = resolver.mbpp_calibration_identity()
    if phase == "stage_a":
        return [], population_hash
    raw_rows = source.mbpp_train_rows()
    canonical_rows = [_canonical_mbpp_row(row) for row in raw_rows]
    by_id: dict[int, dict[str, Any]] = {}
    for row in canonical_rows:
        task_id = int(row["task_id"])
        if task_id in by_id:
            raise ValueError(f"duplicate MBPP task_id {task_id}")
        by_id[task_id] = row
    expected_population = set(range(601, 975))
    if set(by_id) != expected_population:
        raise ValueError("MBPP train population must be exactly task IDs 601..974")
    selected_ids, _ = resolver.mbpp_calibration_identity()
    records: list[dict[str, Any]] = []
    for task_id_text in selected_ids:
        row = by_id[int(task_id_text)]
        tests = "\n".join(row["test_list"])
        prompt = (
            f"You are an expert Python programmer, and here is your task: {row['text']}\n"
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
        )
        code = str(row["code"])
        prompt_ids = _encode(tokenizer, prompt, add_special_tokens=True)
        target_ids = _encode(tokenizer, code, add_special_tokens=False)
        if not prompt_ids or not target_ids:
            raise ValueError(f"MBPP task {task_id_text} produced an empty token side")
        records.append(
            _base_record(
                phase=phase,
                family="mbpp",
                canonical_id=task_id_text,
                config=resolver.MBPP_CONFIG,
                seed=None,
                configured_length=None,
                ruler_category=None,
                generator_receipt_sha256=None,
                source_payload=row,
                formatted_payload={"prompt": prompt, "target": code},
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            )
        )
    return _assign_sha_ranks(records), population_hash


def _validate_projection(
    rows: Sequence[ProjectionRow], *, context: str, exact_count: int | None = None
) -> tuple[ProjectionRow, ...]:
    if exact_count is not None and len(rows) != exact_count:
        raise ValueError(f"{context} must contain exactly {exact_count} identities")
    if not rows:
        raise ValueError(f"{context} cannot be empty")
    seen_ids: set[str] = set()
    seen_offsets: set[int] = set()
    normalized: list[ProjectionRow] = []
    for index, item in enumerate(rows):
        if not isinstance(item, ProjectionRow):
            raise ValueError(f"{context}[{index}] is not a ProjectionRow")
        canonical_id = _require_string(
            item.canonical_id, context=f"{context}[{index}].canonical_id"
        )
        offset = _require_int(item.offset, context=f"{context}[{index}].offset")
        if canonical_id in seen_ids or offset in seen_offsets:
            raise ValueError(f"{context} contains a duplicate identity or offset")
        seen_ids.add(canonical_id)
        seen_offsets.add(offset)
        normalized.append(ProjectionRow(canonical_id, offset))
    if sorted(seen_offsets) != list(range(len(normalized))):
        raise ValueError(f"{context} offsets must be contiguous from zero")
    return tuple(sorted(normalized, key=lambda item: item.offset))


def _segment_start(*, namespace: str, canonical_id: str, token_count: int, width: int) -> int:
    if token_count < width:
        raise ValueError("segment source is shorter than the requested width")
    digest = hashlib.sha256(namespace.encode() + canonical_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (token_count - width + 1)


def _capture_pg19(
    source: CaptureSource,
    *,
    phase: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str, str]:
    split = "train" if phase == "calibration" else "validation"
    required = 16 if phase == "calibration" else 4
    width = 2_304 if phase == "calibration" else 4_224
    segment_namespace = (
        PG19_TRAIN_SEGMENT_NAMESPACE
        if phase == "calibration"
        else PG19_VALIDATION_SEGMENT_NAMESPACE
    )
    selection_namespace = (
        resolver.PG19_TRAIN_NAMESPACE
        if phase == "calibration"
        else resolver.PG19_VALIDATION_NAMESPACE
    )
    projection = _validate_projection(
        source.pg19_projection(split),
        context=f"PG19 {split} URL projection",
        exact_count=13_684 if split == "train" else 50,
    )
    identity_manifest_hash = sha256_bytes(
        canonical_json_bytes([item.canonical_id for item in projection])
    )
    ranked = sorted(
        projection,
        key=lambda item: (
            resolver.selection_sha256(selection_namespace, item.canonical_id),
            item.canonical_id,
        ),
    )
    accepted: list[dict[str, Any]] = []
    for item in ranked:
        raw = source.pg19_row(split, offset=item.offset, expected_url=item.canonical_id)
        if set(raw) < {"url", "text"}:
            raise ValueError("PG19 selected row is missing url or text")
        url = _require_string(raw["url"], context="PG19 row url")
        if url != item.canonical_id:
            raise ValueError("PG19 selected row URL does not match its projection")
        text = _require_string(raw["text"], context="PG19 row text", allow_empty=True)
        full_ids = _encode(tokenizer, text, add_special_tokens=False)
        if len(full_ids) < width:
            continue
        start = _segment_start(
            namespace=segment_namespace,
            canonical_id=url,
            token_count=len(full_ids),
            width=width,
        )
        selected_ids = full_ids[start : start + width]
        if phase == "calibration":
            prompt_ids = selected_ids
            target_ids: tuple[int, ...] = ()
        else:
            prompt_ids = selected_ids[:4_096]
            target_ids = selected_ids[4_096:]
        source_payload = _json_safe(dict(raw), context="PG19 selected row")
        formatted_payload = {
            "url": url,
            "source_token_count": len(full_ids),
            "source_token_start": start,
            "source_token_stop": start + width,
            "selected_token_ids": list(selected_ids),
        }
        accepted.append(
            _base_record(
                phase=phase,
                family="pg19",
                canonical_id=url,
                config="default",
                seed=None,
                configured_length=None,
                ruler_category=None,
                generator_receipt_sha256=None,
                source_payload=source_payload,
                formatted_payload=formatted_payload,
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            )
        )
        if len(accepted) == required:
            break
    if len(accepted) != required:
        raise ValueError(
            f"PG19 {split} has only {len(accepted)} eligible rows; {required} required"
        )
    return _assign_sha_ranks(accepted), identity_manifest_hash, split


def _top_level_yaml_keys(raw: bytes) -> tuple[str, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("RULER synthetic.yaml must be UTF-8") from error
    keys = []
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_]+):", line)
        if match:
            keys.append(match.group(1))
    return tuple(keys)


def _ruler_generator_manifest(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    if set(files) != set(RULER_GENERATOR_GIT_BLOBS):
        raise ValueError("RULER generator file set drifted from the pinned capture contract")
    manifest: list[dict[str, Any]] = []
    for path in sorted(files):
        content = files[path]
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"RULER generator file {path!r} is empty")
        actual_blob = _git_blob_sha1(content)
        if actual_blob != RULER_GENERATOR_GIT_BLOBS[path]:
            raise ValueError(f"RULER generator Git blob drifted for {path}")
        manifest.append(
            {
                "path": path,
                "git_blob_sha1": actual_blob,
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        )
    config_keys = _top_level_yaml_keys(files["scripts/synthetic.yaml"])
    if set(config_keys) != set(RULER_ALL_CONFIGS) or len(config_keys) != len(RULER_ALL_CONFIGS):
        raise ValueError("pinned RULER synthetic config inventory drifted")
    return manifest


def _normalize_auxiliary_files(value: object, *, context: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    if not value:
        raise ValueError(f"{context} cannot be empty; external resources must be bound")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}[{index}] must be an object")
        _require_exact_fields(raw, AUXILIARY_FILE_FIELDS, context=f"{context}[{index}]")
        name = _require_string(raw["name"], context=f"{context}[{index}].name")
        if name in names:
            raise ValueError(f"{context} contains duplicate file name {name!r}")
        names.add(name)
        result.append(
            {
                "name": name,
                "sha256": _require_sha256(raw["sha256"], context=f"{context}[{index}].sha256"),
                "size_bytes": _require_int(
                    raw["size_bytes"],
                    context=f"{context}[{index}].size_bytes",
                    minimum=1,
                ),
            }
        )
    return sorted(result, key=lambda item: item["name"])


def _ruler_canonical_id(*, category: str, config: str, configured_length: int, seed: int) -> str:
    return (
        f"{RULER_SEQUENCE_NAMESPACE}:"
        f"{resolver.RULER_REVISION}:{category}:{config}:"
        f"length={configured_length}:seed={seed}:sample=0"
    )


def _normalize_ruler_receipt(
    value: Mapping[str, Any],
    *,
    category: str,
    config: str,
    configured_length: int,
    seed: int,
) -> dict[str, Any]:
    _require_exact_fields(value, RULER_RECEIPT_FIELDS, context="RULER receipt")
    expected = {
        "schema": RULER_RECEIPT_SCHEMA,
        "source_id": resolver.RULER_SOURCE_ID,
        "revision": resolver.RULER_REVISION,
        "category": category,
        "config": config,
        "configured_length": configured_length,
        "seed": seed,
        "sample_index": 0,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"RULER receipt {field} drifted")
    generator_length = _require_int(
        value["generator_reported_length"],
        context="RULER receipt generator_reported_length",
        minimum=1,
    )
    if generator_length > configured_length:
        raise ValueError("RULER generator-reported length exceeds configured length")
    input_text = _require_string(value["input"], context="RULER receipt input")
    answer_prefix = _require_string(
        value["answer_prefix"],
        context="RULER receipt answer_prefix",
        allow_empty=True,
    )
    outputs = value["outputs"]
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence) or not outputs:
        raise ValueError("RULER receipt outputs must be a non-empty string array")
    normalized_outputs = [_require_string(item, context="RULER receipt output") for item in outputs]
    required_count = RULER_REQUIRED_OUTPUT_COUNTS.get(config)
    if required_count is not None:
        if len(normalized_outputs) != required_count:
            raise ValueError(
                f"RULER receipt {config} must contain exactly {required_count} required outputs"
            )
        if len(set(normalized_outputs)) != len(normalized_outputs):
            raise ValueError(f"RULER receipt {config} required outputs must be unique")
    auxiliary_files = _normalize_auxiliary_files(
        value["auxiliary_files"], context="RULER receipt auxiliary_files"
    )
    return {
        **expected,
        "generator_reported_length": generator_length,
        "input": input_text,
        "answer_prefix": answer_prefix,
        "outputs": normalized_outputs,
        "auxiliary_files": auxiliary_files,
    }


def _ruler_stage_a_target(*, category: str, config: str, outputs: Sequence[str]) -> tuple[str, str]:
    """Freeze a teacher-forced target without changing RULER reference semantics."""

    if category == "question_answering":
        if config not in {"qa_1", "qa_2"}:
            raise ValueError("RULER QA target received a non-QA configuration")
        return outputs[0], "first_pinned_alternative_reference_v1"
    if config not in RULER_REQUIRED_OUTPUT_COUNTS:
        raise ValueError("RULER required-output target received an unknown configuration")
    return (
        RULER_REQUIRED_OUTPUT_SEPARATOR.join(outputs),
        "all_required_outputs_comma_space_v1",
    )


def _capture_ruler(
    source: CaptureSource,
    *,
    phase: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str, str]:
    generator_manifest = _ruler_generator_manifest(source.ruler_generator_files())
    schedule = (
        resolver.RULER_CALIBRATION_SCHEDULE
        if phase == "calibration"
        else resolver.RULER_STAGE_A_SCHEDULE
    )
    records: list[dict[str, Any]] = []
    auxiliary_manifests: list[dict[str, Any]] = []
    selected_identities: list[dict[str, Any]] = []
    for category, config, configured_length, seed in schedule:
        receipt = _normalize_ruler_receipt(
            source.ruler_receipt(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            ),
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        receipt_hash = sha256_bytes(canonical_json_bytes(receipt))
        prompt = receipt["input"] + receipt["answer_prefix"]
        prompt_ids = _encode(tokenizer, prompt, add_special_tokens=False)
        if phase == "calibration":
            target_text = ""
            target_semantics = None
            target_ids = ()
        else:
            target_text, target_semantics = _ruler_stage_a_target(
                category=category,
                config=config,
                outputs=receipt["outputs"],
            )
            target_ids = _encode(tokenizer, target_text, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("RULER receipt produced an empty prompt")
        if phase == "stage_a" and not target_ids:
            raise ValueError("Stage-A RULER receipt produced an empty answer")
        if len(prompt_ids) + len(target_ids) > configured_length:
            raise ValueError("RULER actual sequence exceeds configured length")
        if receipt["generator_reported_length"] < len(prompt_ids):
            raise ValueError("RULER generator receipt is shorter than the tokenized prompt")
        canonical_id = _ruler_canonical_id(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        records.append(
            _base_record(
                phase=phase,
                family="ruler",
                canonical_id=canonical_id,
                config=config,
                seed=seed,
                configured_length=configured_length,
                ruler_category=category,
                generator_receipt_sha256=receipt_hash,
                source_payload=receipt,
                formatted_payload={
                    "prompt": prompt,
                    "target": target_text,
                    "target_semantics": target_semantics,
                    "official_output_indices": (
                        None
                        if phase == "calibration"
                        else (
                            [0]
                            if category == "question_answering"
                            else list(range(len(receipt["outputs"])))
                        )
                    ),
                },
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            )
        )
        auxiliary_manifests.append(
            {
                "canonical_id": canonical_id,
                "files": receipt["auxiliary_files"],
            }
        )
        selected_identities.append(
            {
                "canonical_id": canonical_id,
                "category": category,
                "config": config,
                "configured_length": configured_length,
                "seed": seed,
            }
        )
    inventory = {
        "revision": resolver.RULER_REVISION,
        "configs_by_category": RULER_CONFIGS_BY_CATEGORY,
        "calibration_schedule": resolver.RULER_CALIBRATION_SCHEDULE,
        "stage_a_schedule": resolver.RULER_STAGE_A_SCHEDULE,
        "selected": selected_identities,
    }
    identity_manifest_hash = sha256_bytes(canonical_json_bytes(inventory))
    formatter = {
        "id": "recurquant.ruler-teacher-forced-target.v2",
        "capture_version": CAPTURE_VERSION,
        "prompt": "input + answer_prefix",
        "prompt_add_special_tokens": False,
        "stage_a_target": {
            "required_reference_tasks": {
                "categories": ["retrieval", "multi_hop_tracing", "aggregation"],
                "serialization": (
                    "comma-space join of every official required output in source order"
                ),
                "separator": RULER_REQUIRED_OUTPUT_SEPARATOR,
            },
            "alternative_reference_tasks": {
                "category": "question_answering",
                "serialization": "first pinned official alternative in source order",
            },
        },
        "target_add_special_tokens": False,
        "minimum_stage_a_target_tokens": 2,
        "cache_exposed_predictions": "target_length - 1",
        "generator_files": generator_manifest,
        "auxiliary_receipts": auxiliary_manifests,
    }
    formatter_hash = sha256_bytes(canonical_json_bytes(formatter))
    return _assign_sha_ranks(records), identity_manifest_hash, formatter_hash


def _capture_humaneval(
    source: CaptureSource,
    *,
    phase: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    projection = _validate_projection(
        source.humaneval_projection(),
        context="HumanEval+ task_id projection",
        exact_count=164,
    )
    identity_manifest_hash = sha256_bytes(
        canonical_json_bytes([item.canonical_id for item in projection])
    )
    if phase == "calibration":
        return [], identity_manifest_hash
    ranked = sorted(
        projection,
        key=lambda item: (
            resolver.selection_sha256(resolver.HUMANEVAL_AB_NAMESPACE, item.canonical_id),
            item.canonical_id,
        ),
    )
    records: list[dict[str, Any]] = []
    for item in ranked[:4]:
        row = source.humaneval_row(offset=item.offset, expected_task_id=item.canonical_id)
        task_id = _require_string(row.get("task_id"), context="HumanEval+ task_id")
        if task_id != item.canonical_id:
            raise ValueError("HumanEval+ row does not match its task_id projection")
        prompt = _require_string(row.get("prompt"), context="HumanEval+ prompt")
        solution = _require_string(
            row.get("canonical_solution"), context="HumanEval+ canonical_solution"
        )
        prompt_ids = _encode(tokenizer, prompt, add_special_tokens=True)
        target_ids = _encode(tokenizer, solution, add_special_tokens=False)[:128]
        if not prompt_ids or not target_ids:
            raise ValueError(f"HumanEval+ task {task_id!r} produced an empty token side")
        records.append(
            _base_record(
                phase=phase,
                family="humaneval_plus",
                canonical_id=task_id,
                config="default",
                seed=None,
                configured_length=None,
                ruler_category=None,
                generator_receipt_sha256=None,
                source_payload=_json_safe(dict(row), context="HumanEval+ row"),
                formatted_payload={"prompt": prompt, "target": solution},
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            )
        )
    return _assign_sha_ranks(records), identity_manifest_hash


def _dataset_contracts(
    *,
    mbpp_manifest_hash: str,
    pg19_manifest_hash: str,
    pg19_split: str,
    ruler_manifest_hash: str,
    ruler_formatter_hash: str,
    humaneval_manifest_hash: str,
) -> list[dict[str, Any]]:
    return [
        {
            "key": "mbpp",
            "dataset_id": resolver.MBPP_DATASET_ID,
            "config": resolver.MBPP_CONFIG,
            "revision": resolver.MBPP_REVISION,
            "split": "train",
            "canonical_id_field": "task_id",
            "canonical_id_manifest_sha256": mbpp_manifest_hash,
            "formatter_id": MBPP_FORMATTER_SPEC["id"],
            "formatter_sha256": sha256_bytes(canonical_json_bytes(MBPP_FORMATTER_SPEC)),
        },
        {
            "key": "pg19",
            "dataset_id": resolver.PG19_DATASET_ID,
            "config": "default",
            "revision": resolver.PG19_REVISION,
            "split": pg19_split,
            "canonical_id_field": "url",
            "canonical_id_manifest_sha256": pg19_manifest_hash,
            "formatter_id": PG19_FORMATTER_SPEC["id"],
            "formatter_sha256": sha256_bytes(canonical_json_bytes(PG19_FORMATTER_SPEC)),
        },
        {
            "key": "ruler",
            "dataset_id": resolver.RULER_SOURCE_ID,
            "config": "official-generator",
            "revision": resolver.RULER_REVISION,
            "split": "generated",
            "canonical_id_field": "configuration_id",
            "canonical_id_manifest_sha256": ruler_manifest_hash,
            "formatter_id": "recurquant.ruler-official-generated-record.v1",
            "formatter_sha256": ruler_formatter_hash,
        },
        {
            "key": "humaneval_plus",
            "dataset_id": resolver.HUMANEVAL_PLUS_DATASET_ID,
            "config": "default",
            "revision": resolver.HUMANEVAL_PLUS_REVISION,
            "split": "test",
            "canonical_id_field": "task_id",
            "canonical_id_manifest_sha256": humaneval_manifest_hash,
            "formatter_id": HUMANEVAL_FORMATTER_SPEC["id"],
            "formatter_sha256": sha256_bytes(canonical_json_bytes(HUMANEVAL_FORMATTER_SPEC)),
        },
    ]


def _normalize_calibration_binding(value: object) -> dict[str, str]:
    if not isinstance(value, bytes):
        raise ValueError("Stage-A calibration binding must be a verified artifact byte string")
    verified = resolver.deserialize_stage_a_calibration_binding_artifact(value)
    return dict(verified.binding)


def capture_identity_input(
    *,
    phase: str,
    source: CaptureSource,
    calibration_binding: bytes | None = None,
) -> dict[str, Any]:
    """Capture one deterministic calibration or Stage-A resolver input."""

    if phase in resolver.PROTECTED_STAGES:
        raise PermissionError(f"{phase} is protected; capture v2 refuses it before source access")
    if phase not in resolver.ALLOWED_PHASES:
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if phase == "stage_a" and calibration_binding is None:
        raise ValueError("Stage A requires a frozen calibration binding")
    if phase == "calibration" and calibration_binding is not None:
        raise ValueError("calibration capture forbids a Stage-A binding")

    before = _validate_heads(source.source_heads(), context="pre-capture")
    material = source.tokenizer_material()
    tokenizer_contract, tokenizer_manifest_hash = _tokenizer_contract(material)
    mbpp_records, mbpp_manifest_hash = _capture_mbpp(
        source,
        phase=phase,
        tokenizer=material.tokenizer,
        tokenizer_manifest_sha256=tokenizer_manifest_hash,
    )
    pg19_records, pg19_manifest_hash, pg19_split = _capture_pg19(
        source,
        phase=phase,
        tokenizer=material.tokenizer,
        tokenizer_manifest_sha256=tokenizer_manifest_hash,
    )
    ruler_records, ruler_manifest_hash, ruler_formatter_hash = _capture_ruler(
        source,
        phase=phase,
        tokenizer=material.tokenizer,
        tokenizer_manifest_sha256=tokenizer_manifest_hash,
    )
    humaneval_records, humaneval_manifest_hash = _capture_humaneval(
        source,
        phase=phase,
        tokenizer=material.tokenizer,
        tokenizer_manifest_sha256=tokenizer_manifest_hash,
    )
    after = _validate_heads(source.source_heads(), context="post-capture")
    if after != before:  # Defensive even though both were pinned.
        raise ValueError("source HEAD changed during capture")

    result: dict[str, Any] = {
        "schema": resolver.INPUT_SCHEMA,
        "phase": phase,
        "datasets": _dataset_contracts(
            mbpp_manifest_hash=mbpp_manifest_hash,
            pg19_manifest_hash=pg19_manifest_hash,
            pg19_split=pg19_split,
            ruler_manifest_hash=ruler_manifest_hash,
            ruler_formatter_hash=ruler_formatter_hash,
            humaneval_manifest_hash=humaneval_manifest_hash,
        ),
        "tokenizer": tokenizer_contract,
        "records": [
            *mbpp_records,
            *pg19_records,
            *ruler_records,
            *humaneval_records,
        ],
        "model_weights_loaded": False,
    }
    if phase == "stage_a":
        result["calibration_binding"] = _normalize_calibration_binding(calibration_binding)
    expected_revisions = dict(resolver.FROZEN_DATASET_REVISIONS)
    resolver.build_candidate(
        result,
        expected_revisions=expected_revisions,
        calibration_binding_artifact=calibration_binding,
    )
    return result


def atomic_write_no_overwrite(path: Path, payload: bytes) -> None:
    """Atomically publish *payload* while refusing an existing destination."""

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
            raise FileExistsError(f"refusing to overwrite existing capture: {resolved}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


class LiveCaptureSource:
    """Pinned, read-only network source; RULER generated rows come from receipts."""

    def __init__(self, *, cache_dir: Path, ruler_receipt_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.ruler_receipt_dir = ruler_receipt_dir.resolve()

    @staticmethod
    def _github_head(repo_id: str) -> str:
        repo_request = urllib.request.Request(
            f"https://api.github.com/repos/{repo_id}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "RecurQuant-Experiment-013-identity-capture",
            },
        )
        try:
            with urllib.request.urlopen(repo_request, timeout=30) as response:
                metadata = json.load(response)
            branch = _require_string(
                metadata.get("default_branch"), context=f"{repo_id} default branch"
            )
            commit_request = urllib.request.Request(
                f"https://api.github.com/repos/{repo_id}/commits/"
                f"{urllib.parse.quote(branch, safe='')}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "RecurQuant-Experiment-013-identity-capture",
                },
            )
            with urllib.request.urlopen(commit_request, timeout=30) as response:
                commit = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot resolve GitHub HEAD for {repo_id}") from error
        return _require_string(commit.get("sha"), context=f"{repo_id} HEAD")

    def source_heads(self) -> Mapping[str, str]:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("live capture requires huggingface-hub") from error
        api = HfApi()
        return {
            "primary_model": str(api.model_info(resolver.PRIMARY_MODEL_ID).sha),
            "mbpp": str(api.dataset_info(resolver.MBPP_DATASET_ID).sha),
            "pg19": str(api.dataset_info(resolver.PG19_DATASET_ID).sha),
            "ruler": self._github_head(resolver.RULER_SOURCE_ID),
            "humaneval_plus": str(api.dataset_info(resolver.HUMANEVAL_PLUS_DATASET_ID).sha),
            "evalplus": self._github_head(resolver.EVALPLUS_SOURCE_ID),
        }

    def tokenizer_material(self) -> TokenizerMaterial:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("live capture requires huggingface-hub and Transformers") from error
        api = HfApi()
        available = set(
            api.list_repo_files(resolver.PRIMARY_MODEL_ID, revision=resolver.PRIMARY_MODEL_REVISION)
        )
        selected = [name for name in TOKENIZER_ASSET_NAMES if name in available]
        if "tokenizer.json" not in selected or "tokenizer_config.json" not in selected:
            raise ValueError("pinned model revision lacks mandatory tokenizer assets")
        if any(FORBIDDEN_MODEL_FILE_RE.search(name) for name in selected):
            raise ValueError("tokenizer allow-list unexpectedly includes a model file")
        paths = [
            Path(
                hf_hub_download(
                    repo_id=resolver.PRIMARY_MODEL_ID,
                    filename=name,
                    revision=resolver.PRIMARY_MODEL_REVISION,
                    cache_dir=self.cache_dir,
                )
            )
            for name in selected
        ]
        files = {path.name: path.read_bytes() for path in paths}
        if set(files) != set(selected):
            raise ValueError("downloaded tokenizer file inventory drifted")
        # A shared Hub snapshot can contain files fetched by an earlier run.
        # Load only from a new directory populated with the exact authenticated
        # allow-list so no unbound sibling can affect tokenizer construction.
        with tempfile.TemporaryDirectory(prefix="recurquant-exp013-tokenizer-") as temporary:
            isolated = Path(temporary)
            for name, data in files.items():
                (isolated / name).write_bytes(data)
            tokenizer = AutoTokenizer.from_pretrained(
                isolated,
                local_files_only=True,
                trust_remote_code=False,
            )
            isolated_inventory = {
                path.relative_to(isolated).as_posix()
                for path in isolated.rglob("*")
                if path.is_file()
            }
            if isolated_inventory != set(selected):
                raise ValueError("tokenizer construction changed the isolated file inventory")
        return TokenizerMaterial(
            tokenizer=tokenizer,
            tokenizer_class=tokenizer.__class__.__name__,
            transformers_version=importlib.metadata.version("transformers"),
            files=files,
            model_weights_loaded=False,
        )

    def mbpp_train_rows(self) -> Sequence[Mapping[str, Any]]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("live capture requires the datasets extra") from error
        rows = load_dataset(
            resolver.MBPP_DATASET_ID,
            resolver.MBPP_CONFIG,
            revision=resolver.MBPP_REVISION,
            split="train",
            streaming=True,
        )
        return tuple(dict(row) for row in rows)

    @staticmethod
    def _active_parquet_aliases(
        *, dataset_id: str, revision: str, config: str, split: str
    ) -> tuple[str, ...]:
        """Resolve only Dataset Viewer's active parquet conversion at *revision*.

        Dataset repositories can retain stale or sibling parquet files that are
        not members of the active builder split.  The `/parquet` manifest is the
        authority for the active conversion; its `x-revision` must bind it to the
        pinned source commit before any alias is opened.
        """

        query = urllib.parse.urlencode({"dataset": dataset_id})
        request = urllib.request.Request(
            f"https://datasets-server.huggingface.co/parquet?{query}",
            headers={"User-Agent": "RecurQuant-Experiment-013-identity-capture"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                source_revision = response.headers.get("x-revision")
                payload = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cannot resolve active parquet manifest for {dataset_id}"
            ) from error
        if source_revision != revision:
            raise ValueError("Dataset Viewer parquet x-revision is not pinned")
        raw_entries = payload.get("parquet_files")
        if not isinstance(raw_entries, list):
            raise ValueError("Dataset Viewer parquet manifest is malformed")
        aliases: list[str] = []
        names: set[str] = set()
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError("Dataset Viewer parquet entry is malformed")
            if (
                raw.get("dataset") != dataset_id
                or raw.get("config") != config
                or raw.get("split") != split
            ):
                continue
            filename = _require_string(raw.get("filename"), context="active parquet filename")
            if Path(filename).name != filename or filename in names:
                raise ValueError("active parquet filenames must be unique basenames")
            _require_int(raw.get("size"), context="active parquet size", minimum=1)
            url = _require_string(raw.get("url"), context="active parquet URL")
            expected_prefix = (
                f"https://huggingface.co/datasets/{dataset_id}/resolve/refs%2Fconvert%2Fparquet/"
            )
            if not url.startswith(expected_prefix):
                raise ValueError("active parquet URL does not match its alias identity")
            relative = url.removeprefix(expected_prefix)
            parts = relative.split("/")
            if (
                len(parts) != 3
                or parts[0] != config
                or parts[1] not in {split, f"partial-{split}"}
                or parts[2] != filename
            ):
                raise ValueError("active parquet URL does not match its alias identity")
            names.add(filename)
            aliases.append(f"datasets/{dataset_id}@~parquet/{relative}")
        if not aliases:
            raise ValueError(f"no active {dataset_id} {config}/{split} parquet files")
        return tuple(sorted(aliases))

    @staticmethod
    def _parquet_projection(
        *, dataset_id: str, revision: str, config: str, split: str, column: str
    ) -> tuple[ProjectionRow, ...]:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfFileSystem
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("ID-only projection requires pyarrow and huggingface-hub") from error
        aliases = LiveCaptureSource._active_parquet_aliases(
            dataset_id=dataset_id,
            revision=revision,
            config=config,
            split=split,
        )

        def read_column(alias: str) -> list[Any]:
            fs = HfFileSystem()
            with fs.open(alias, "rb") as handle:
                table = pq.ParquetFile(handle).read(columns=[column])
            return table.column(column).to_pylist()

        worker_count = min(4, len(aliases))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            shards = tuple(executor.map(read_column, aliases))
        result: list[ProjectionRow] = []
        offset = 0
        for values in shards:
            for value in values:
                result.append(
                    ProjectionRow(
                        _require_string(value, context=f"{dataset_id} {column}"),
                        offset,
                    )
                )
                offset += 1
        return tuple(result)

    @staticmethod
    def _viewer_row(
        *,
        dataset_id: str,
        config: str,
        split: str,
        offset: int,
        expected_revision: str,
    ) -> Mapping[str, Any]:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 1,
            }
        )
        request = urllib.request.Request(
            f"https://datasets-server.huggingface.co/rows?{query}",
            headers={"User-Agent": "RecurQuant-Experiment-013-identity-capture"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                revision = response.headers.get("x-revision")
                payload = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Dataset Viewer row request failed for {dataset_id}") from error
        if revision != expected_revision:
            raise ValueError("Dataset Viewer x-revision does not match the pinned commit")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("Dataset Viewer did not return exactly one row")
        item = rows[0]
        if not isinstance(item, Mapping) or item.get("row_idx") != offset:
            raise ValueError("Dataset Viewer row offset drifted")
        truncated = item.get("truncated_cells")
        if truncated not in (None, []):
            raise ValueError("Dataset Viewer truncated a selected source row")
        row = item.get("row")
        if not isinstance(row, Mapping):
            raise ValueError("Dataset Viewer selected row is malformed")
        return dict(row)

    def pg19_projection(self, split: str) -> Sequence[ProjectionRow]:
        return self._parquet_projection(
            dataset_id=resolver.PG19_DATASET_ID,
            revision=resolver.PG19_REVISION,
            config="default",
            split=split,
            column="url",
        )

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> Mapping[str, Any]:
        row = self._viewer_row(
            dataset_id=resolver.PG19_DATASET_ID,
            config="default",
            split=split,
            offset=offset,
            expected_revision=resolver.PG19_REVISION,
        )
        if row.get("url") != expected_url:
            raise ValueError("Dataset Viewer PG19 URL does not match the pinned projection")
        return row

    def ruler_generator_files(self) -> Mapping[str, bytes]:
        base = (
            "https://raw.githubusercontent.com/"
            f"{resolver.RULER_SOURCE_ID}/{resolver.RULER_REVISION}/"
        )
        files: dict[str, bytes] = {}
        for path in RULER_GENERATOR_GIT_BLOBS:
            request = urllib.request.Request(
                base + path,
                headers={"User-Agent": "RecurQuant-Experiment-013-identity-capture"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    files[path] = response.read()
            except (OSError, urllib.error.HTTPError) as error:
                raise RuntimeError(f"cannot fetch pinned RULER source file {path}") from error
        return files

    @staticmethod
    def _receipt_filename(*, category: str, config: str, configured_length: int, seed: int) -> str:
        return ruler_receipt_filename(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )

    def ruler_receipt(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> Mapping[str, Any]:
        path = self.ruler_receipt_dir / self._receipt_filename(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing audited RULER receipt; generation is intentionally separate: {path}"
            )
        return _strict_json(path.read_bytes(), context=f"RULER receipt {path.name}")

    def humaneval_projection(self) -> Sequence[ProjectionRow]:
        return self._parquet_projection(
            dataset_id=resolver.HUMANEVAL_PLUS_DATASET_ID,
            revision=resolver.HUMANEVAL_PLUS_REVISION,
            config="default",
            split="test",
            column="task_id",
        )

    def humaneval_row(self, *, offset: int, expected_task_id: str) -> Mapping[str, Any]:
        row = self._viewer_row(
            dataset_id=resolver.HUMANEVAL_PLUS_DATASET_ID,
            config="default",
            split="test",
            offset=offset,
            expected_revision=resolver.HUMANEVAL_PLUS_REVISION,
        )
        if row.get("task_id") != expected_task_id:
            raise ValueError("Dataset Viewer HumanEval+ ID does not match projection")
        return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a calibration or Stage-A Experiment 013 identity input. "
            "No model weights are requested or loaded."
        )
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibration", "stage_a", "stage_b", "stage_c"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/exp013-identity"))
    parser.add_argument("--ruler-receipt-dir", type=Path)
    parser.add_argument("--calibration-binding", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase in resolver.PROTECTED_STAGES:
        raise PermissionError(
            f"{args.phase} is protected; capture v2 refuses it before file or source access"
        )
    if args.ruler_receipt_dir is None:
        raise ValueError("--ruler-receipt-dir is required")
    if args.dry_run and args.output is not None:
        raise ValueError("--dry-run forbids --output")
    if not args.dry_run and args.output is None:
        raise ValueError("capture requires --output or --dry-run")
    binding: bytes | None = None
    if args.phase == "stage_a":
        if args.calibration_binding is None:
            raise ValueError("Stage A requires --calibration-binding")
        binding = args.calibration_binding.read_bytes()
    elif args.calibration_binding is not None:
        raise ValueError("--calibration-binding is valid only for Stage A")
    source = LiveCaptureSource(
        cache_dir=args.cache_dir,
        ruler_receipt_dir=args.ruler_receipt_dir,
    )
    captured = capture_identity_input(
        phase=args.phase,
        source=source,
        calibration_binding=binding,
    )
    payload = canonical_json_bytes(captured)
    digest = sha256_bytes(payload)
    if args.dry_run:
        print(digest)
        return 0
    atomic_write_no_overwrite(args.output, payload)
    print(digest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
