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
import base64
import binascii
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Final, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_PATH = REPOSITORY_ROOT / "scripts" / "resolve_static_q468_identity.py"
CALIBRATION_RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "run_static_q468_calibration.py"
PARQUET_MATERIALIZATION_MANIFEST_PATH = (
    REPOSITORY_ROOT / "research" / "experiment013-parquet-materializations.json"
)
_RESOLVER_MODULE_NAME = "recurquant_experiment013_identity_resolver"
_CALIBRATION_RUNNER_MODULE_NAME = "_recurquant_experiment013_calibration_runner_for_capture"
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

# Procedure version.  The resolver-compatible identity field sets remain v5.
CAPTURE_VERSION: Final = 9
# Frozen legacy field inside the RULER formatter-v2 fingerprint.  This value
# records the capture epoch in which the unchanged prompt/target serialization
# was frozen; it is not the live capture procedure version.  Keeping these
# version domains separate prevents custody-only capture changes from silently
# changing the scientific formatter commitment.
RULER_FORMATTER_FROZEN_CAPTURE_VERSION: Final = 6
RUNTIME_AUTHENTICATION_CONTEXT_FIELDS: Final = frozenset(
    {
        "base_runtime_root",
        "git_executable",
        "staged_interpreter",
        "package_runtime_roots",
        "package_import_paths",
    }
)
_RUNTIME_ROOT_NAME_RE: Final = re.compile(r"[a-z][a-z0-9-]{0,63}")
RULER_RECEIPT_SCHEMA: Final = "recurquant.experiment013.ruler-receipt.v1"
RULER_GENERATION_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-generation-manifest.v2"
RULER_RUNTIME_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-runtime-manifest.v3"
RULER_LAUNCHER_REVISION: Final = "experiment-013-ruler-argv-launcher-v7"
RULER_RUNTIME_PYTHON_VERSION: Final = "3.11.15"
RULER_SEALED_STARTUP_POLICY: Final = {
    "dont_write_bytecode": 1,
    "no_site": 1,
    "package_path_mode": "staged-record-only-site-packages-v1",
    "pycache_mode": "verified-empty-prefix-no-write-v1",
    "site_loaded": False,
    "utf8_mode": 1,
    "virtualenv_hook_loaded": False,
}
RULER_EXCLUDED_VIRTUALENV_STARTUP_FILES: Final = {
    "_virtualenv.pth": (
        18,
        "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    ),
    "_virtualenv.py": (
        5_246,
        "cfb3db86aaa53bb62b5ff764970bec2d71c9228590a0ebec57f6ec926cc0bf1a",
    ),
}
RULER_LAUNCHER_PATH: Final = (
    REPOSITORY_ROOT / "scripts" / ("generate_static_q468_ruler_receipts.py")
)
RULER_REQUIREMENTS_PATH: Final = REPOSITORY_ROOT / "requirements" / ("experiment013-ruler.txt")
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
RULER_GENERATOR_TOKENS: Final = {
    "niah_multiquery": 128,
    "niah_multikey_2": 128,
    "niah_single_1": 128,
    "niah_multivalue": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}
RULER_NIAH_CONFIGS: Final = frozenset(
    {"niah_multiquery", "niah_multikey_2", "niah_single_1", "niah_multivalue"}
)
_STAGE_A_MATERIALIZATION_AUTHENTICATION_SEAL: Final = object()


@dataclass(frozen=True, slots=True)
class RulerTaskInvariant:
    """Frozen semantic checks replayed independently of the receipt launcher."""

    input_marker: str
    input_prefix: str
    input_suffix: str
    answer_prefix_marker: str
    answer_prefix_suffix: str
    expected_output_count: int | None
    unique_outputs: bool
    output_pattern: str | None
    outputs_must_appear: bool


RULER_TASK_INVARIANTS: Final = {
    "niah_multiquery": RulerTaskInvariant(
        input_marker="What are all the special magic numbers",
        input_prefix="Some special magic numbers are hidden within the following text.",
        input_suffix="mentioned in the provided text?",
        answer_prefix_marker=" The special magic numbers",
        answer_prefix_suffix="mentioned in the provided text are",
        expected_output_count=4,
        unique_outputs=True,
        output_pattern=r"[0-9]{7}",
        outputs_must_appear=True,
    ),
    "niah_multikey_2": RulerTaskInvariant(
        input_marker="What is the special magic number",
        input_prefix="A special magic number is hidden within the following text.",
        input_suffix="mentioned in the provided text?",
        answer_prefix_marker=" The special magic number",
        answer_prefix_suffix="mentioned in the provided text is",
        expected_output_count=1,
        unique_outputs=True,
        output_pattern=r"[0-9]{7}",
        outputs_must_appear=True,
    ),
    "niah_single_1": RulerTaskInvariant(
        input_marker="What is the special magic number",
        input_prefix="A special magic number is hidden within the following text.",
        input_suffix="mentioned in the provided text?",
        answer_prefix_marker=" The special magic number",
        answer_prefix_suffix="mentioned in the provided text is",
        expected_output_count=1,
        unique_outputs=True,
        output_pattern=r"[0-9]{7}",
        outputs_must_appear=True,
    ),
    "niah_multivalue": RulerTaskInvariant(
        input_marker="What are all the special magic numbers",
        input_prefix="Some special magic numbers are hidden within the following text.",
        input_suffix="mentioned in the provided text?",
        answer_prefix_marker=" The special magic numbers",
        answer_prefix_suffix="mentioned in the provided text are",
        expected_output_count=4,
        unique_outputs=True,
        output_pattern=r"[0-9]{7}",
        outputs_must_appear=True,
    ),
    "vt": RulerTaskInvariant(
        input_marker="Question: Find all variables",
        input_prefix="Memorize and track the chain(s) of variable assignment",
        input_suffix="in the text above.",
        answer_prefix_marker=(
            " Answer: According to the chain(s) of variable assignment in the text above,"
        ),
        answer_prefix_suffix=", they are: ",
        expected_output_count=5,
        unique_outputs=True,
        output_pattern=r"[A-Z]{5}",
        outputs_must_appear=True,
    ),
    "cwe": RulerTaskInvariant(
        input_marker="Question: What are the 10 most common words",
        input_prefix="Below is a numbered list of words.",
        input_suffix="in the above list?",
        answer_prefix_marker=" Answer: The top 10 words that appear most often in the list are:",
        answer_prefix_suffix="in the list are:",
        expected_output_count=10,
        unique_outputs=True,
        output_pattern=None,
        outputs_must_appear=True,
    ),
    "fwe": RulerTaskInvariant(
        input_marker="What are the three most frequently appeared words",
        input_prefix="Read the following coded text and track the frequency of each coded word.",
        input_suffix="in the above coded text?",
        answer_prefix_marker=" Answer: According to the coded text above,",
        answer_prefix_suffix="the three most frequently appeared words are:",
        expected_output_count=3,
        unique_outputs=True,
        output_pattern=r"[a-z]{6}",
        outputs_must_appear=True,
    ),
    "qa_1": RulerTaskInvariant(
        input_marker="The following are given documents.",
        input_prefix="Answer the question based on the given documents.",
        input_suffix="?",
        answer_prefix_marker=" Answer:",
        answer_prefix_suffix=" Answer:",
        expected_output_count=None,
        unique_outputs=False,
        output_pattern=None,
        outputs_must_appear=False,
    ),
    "qa_2": RulerTaskInvariant(
        input_marker="The following are given documents.",
        input_prefix="Answer the question based on the given documents.",
        input_suffix="?",
        answer_prefix_marker=" Answer:",
        answer_prefix_suffix=" Answer:",
        expected_output_count=None,
        unique_outputs=False,
        output_pattern=None,
        outputs_must_appear=False,
    ),
}
RULER_FORBIDDEN_RUNTIME_MODULES: Final = (
    "accelerate",
    "bitsandbytes",
    "flax",
    "jax",
    "onnx",
    "onnxruntime",
    "tensorflow",
    "torch",
)
RULER_EXPECTED_CORPORA: Final = {
    "PaulGrahamEssays.json": (
        3_108_621,
        "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd",
    ),
    "english_words.json": (
        8_564_991,
        "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    ),
    "hotpotqa.json": (
        61_065_698,
        "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    ),
    "squad.json": (
        4_370_528,
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
}
RULER_EXPECTED_PACKAGE_RESOURCES: Final = {
    "nltk/punkt/english.pickle": (
        433_305,
        "dda37972ae88998a6fd3e3ec002697a6bd362b32d050fda7d7ca5276873092aa",
    ),
    "nltk/punkt/PY3/english.pickle": (
        406_697,
        "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c",
    ),
    "wonderwords/adjectivelist.txt": (
        7_403,
        "66814d46b7e292c83e839d12fe2fa083c8f66779b14b294e1be4e1342c5d4131",
    ),
    "wonderwords/nounlist.txt": (
        54_752,
        "8d88b3ebc2e2969ed92ebe49cdb0c8a87c2ad3b27493d7afdff608f2e51c5bc9",
    ),
    "wonderwords/verblist.txt": (
        6_967,
        "9fbf5e4e69b8869aebd88e6d545312ff9f878a747327ede5f6dbbba5c27b08ad",
    ),
}
RULER_EXPECTED_TOKENIZER_ASSETS: Final = {
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "tokenizer.json": (
        12_807_196,
        "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    ),
    "tokenizer_config.json": (
        16_712,
        "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
}

# These are hashes of the complete canonical command-manifest bytes for every
# required receipt.  They freeze the entire portable argv, including templates
# and task-specific flags, without trusting claims embedded in the receipt set.
RULER_COMMAND_MANIFEST_SHA256_BY_FILENAME: Final = {
    "aggregation__cwe__l2048__s12340.json": (
        "2e51bae29377e1cc5d4f92d0b8479951e788e7fb27fd2f8fcc177c9e3cc2811a"
    ),
    "aggregation__cwe__l4096__s12340.json": (
        "a63f87abadca40d66a682584ee831122d3c837ecc44a1ddb57a9e0b676744aa7"
    ),
    "aggregation__fwe__l2048__s12339.json": (
        "ec652c7d26916a1e988c02a88c9d856ff05ce926583e1a2fce953b919b3c7ad2"
    ),
    "aggregation__fwe__l4096__s12339.json": (
        "49f19c6045484e1c584657fb57412f1fbd1945d6cfcca6504a666b344765a0d0"
    ),
    "aggregation__fwe__l4096__s2344.json": (
        "39aa60129ed248a6b722cf83c51b6a1a1784982824f167727fe2657a96612568"
    ),
    "multi_hop_tracing__vt__l2048__s12339.json": (
        "9d253ced1f09e593fc1cffff0f035604685f54acb0f4dc0080d5f920f9a65993"
    ),
    "multi_hop_tracing__vt__l2048__s12340.json": (
        "25e0258140d82c80ae6d466c8ba3c22dc98aae29c6b8b87bbbf1817c1c4aa561"
    ),
    "multi_hop_tracing__vt__l4096__s12339.json": (
        "570153e676619042f72a71c2ed2eb27829fcef292cebce7734d3523b10581271"
    ),
    "multi_hop_tracing__vt__l4096__s12340.json": (
        "93ee5ec2be6fe5ac4bde20d82f32a1e9f46ec4008aab2b4d9c035b813255dae5"
    ),
    "multi_hop_tracing__vt__l4096__s2344.json": (
        "320e4832d9b346087e4094262a5ec18606422badca4d201658a0e2f94bd1cae5"
    ),
    "question_answering__qa_1__l2048__s12339.json": (
        "12cc42446211811efe8e525e3fa8f68306338b7cb15901e4bc6c037563505bea"
    ),
    "question_answering__qa_1__l4096__s12339.json": (
        "ff9623b65bd1fa8285c9078bcc861a7ceefe9ca6b6bfe470dae6c2cc08a6de83"
    ),
    "question_answering__qa_1__l4096__s2344.json": (
        "f058521c0d729e6596f3cbda4f308d1c29c4d015a89ac5e783548af4f6cea183"
    ),
    "question_answering__qa_2__l2048__s12340.json": (
        "402b92a2c76288d37b819683a1e5e83e8a5bd95c60143d958e0b027e74975997"
    ),
    "question_answering__qa_2__l4096__s12340.json": (
        "e841f60a9858d16da1490cee99c64db20435c8d992f27a71d196a1601b478e2a"
    ),
    "retrieval__niah_multikey_2__l2048__s12340.json": (
        "58a474ae63c52c5bf897de988ef952e85a87326096a5db2374a577e44c53aeed"
    ),
    "retrieval__niah_multiquery__l2048__s12339.json": (
        "61b77e3475cdf72fed65e7d28d97151dcbb0a9b728cf40756e634371be99009d"
    ),
    "retrieval__niah_multiquery__l4096__s2344.json": (
        "e45f31c8e697f0468836ba3635ac4d3c2116c2ac05d6866544d28626a8bedbb7"
    ),
    "retrieval__niah_multivalue__l4096__s12340.json": (
        "d791dcf45a923a7d8aafc5cf2d9dda3b039878288556307bf75023a338c0b6f8"
    ),
    "retrieval__niah_single_1__l4096__s12339.json": (
        "d08571d33bde611a4e8b7394ce0b6ec92742f6b4d15a6a3f044974ea99d42949"
    ),
}

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


@dataclass(frozen=True, slots=True)
class _RuntimeAuthenticationContext:
    base_runtime_root: Path
    git_executable: Path
    staged_interpreter: Path
    package_runtime_roots: Mapping[str, Path]
    package_import_paths: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _AuthenticatedExecutionBindings:
    bindings: Mapping[str, str]
    source_manifest: Mapping[str, object]
    runtime_manifest: Any
    model_manifest: Any
    runner: Any
    runtime_context: _RuntimeAuthenticationContext


@dataclass(frozen=True, slots=True)
class _DecodedExecutionBindingArtifacts:
    bindings: Mapping[str, str]
    source_manifest: Mapping[str, object]
    runtime_manifest: Any
    model_manifest: Any
    source_module: Any
    parquet_module: Any


@dataclass(frozen=True, slots=True)
class _DecodedRepositorySourceArtifact:
    manifest_file_sha256: str
    source_manifest: Mapping[str, object]
    source_module: Any


@dataclass(frozen=True, slots=True)
class VerifiedRulerBundle:
    receipts: Mapping[str, dict[str, Any]]
    generator_manifest: tuple[dict[str, Any], ...]
    generation_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class MaterializedCalibrationSequence:
    """One exact calibration identity record with its formatter-produced tokens.

    Canonical record bytes are kept privately and decoded to a fresh mapping on
    every access.  The object therefore cannot leak source payloads or let an
    adapter mutate the captured identity record in place.
    """

    _identity_record_bytes: bytes = dataclass_field(repr=False)
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        record = _strict_json(
            self._identity_record_bytes,
            context="materialized calibration identity record",
        )
        if canonical_json_bytes(record) != self._identity_record_bytes:
            raise ValueError("materialized calibration identity record is not canonical")
        expected_fields = set(resolver.IDENTITY_RECORD_PAYLOAD_FIELDS) | {"identity_record_sha256"}
        if set(record) != expected_fields:
            raise ValueError("materialized calibration identity record fields drifted")
        if not isinstance(self.prompt_token_ids, tuple) or not isinstance(
            self.target_token_ids, tuple
        ):
            raise TypeError("materialized token IDs must be tuples")
        for side, token_ids in (
            ("prompt", self.prompt_token_ids),
            ("target", self.target_token_ids),
        ):
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in token_ids
            ):
                raise ValueError(f"materialized {side} token IDs must be non-negative integers")
        sequence_ids = self.sequence_token_ids
        span = record.get("token_span")
        if not isinstance(span, Mapping):
            raise ValueError("materialized identity record token span is malformed")
        if (
            record.get("identity_record_sha256") != resolver.identity_record_sha256(record)
            or record.get("prompt_token_ids_sha256") != _token_hash(self.prompt_token_ids)
            or record.get("target_token_ids_sha256") != _token_hash(self.target_token_ids)
            or record.get("sequence_token_ids_sha256") != _token_hash(sequence_ids)
            or record.get("fisher_boundary")
            != resolver.build_fisher_boundary_contract(sequence_ids)
            or record.get("sequence_length") != len(sequence_ids)
            or span.get("prefill_start") != 0
            or span.get("prefill_stop") != len(self.prompt_token_ids)
            or span.get("scored_start") != len(self.prompt_token_ids)
            or span.get("scored_stop") != len(sequence_ids)
            or span.get("cache_exposed_start") != len(sequence_ids)
            or span.get("cache_exposed_stop") != len(sequence_ids)
        ):
            raise ValueError("materialized calibration tokens differ from their identity record")
        forbidden = {
            "answer_prefix",
            "code",
            "formatted_payload",
            "input",
            "outputs",
            "source_payload",
            "text",
        }
        if forbidden & set(record):
            raise ValueError("materialized identity record contains forbidden raw content")

    @property
    def identity_record(self) -> dict[str, Any]:
        """Return a fresh exact copy of the canonical identity record."""

        return _strict_json(
            self._identity_record_bytes,
            context="materialized calibration identity record",
        )

    @property
    def identity_record_sha256(self) -> str:
        return str(self.identity_record["identity_record_sha256"])

    @property
    def sequence_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.target_token_ids


@dataclass(frozen=True, slots=True)
class CalibrationIdentityMaterialization:
    """Content-redacted token materialization consumed by a future live adapter."""

    sequences: tuple[MaterializedCalibrationSequence, ...]
    tokenizer_manifest_sha256: str
    capture_input_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequences, tuple) or len(self.sequences) != 160:
            raise ValueError("calibration materialization must contain exactly 160 sequences")
        tokenizer_hash = _require_sha256(
            self.tokenizer_manifest_sha256,
            context="calibration materialization tokenizer manifest",
        )
        _require_sha256(
            self.capture_input_sha256,
            context="calibration materialization capture input",
        )
        records = [sequence.identity_record for sequence in self.sequences]
        identities = [sequence.identity_record_sha256 for sequence in self.sequences]
        if len(set(identities)) != len(identities):
            raise ValueError("calibration materialization contains duplicate identities")
        if any(record["tokenizer_manifest_sha256"] != tokenizer_hash for record in records):
            raise ValueError("calibration materialization tokenizer commitments differ")
        if records != sorted(records, key=resolver._record_sort_key):
            raise ValueError("calibration materialization records are not in canonical order")

    @property
    def identity_records(self) -> tuple[dict[str, Any], ...]:
        """Return fresh record copies in canonical capture order."""

        return tuple(sequence.identity_record for sequence in self.sequences)

    @property
    def by_identity_record_sha256(
        self,
    ) -> Mapping[str, MaterializedCalibrationSequence]:
        """Return an immutable digest lookup independent of traversal order."""

        return MappingProxyType(
            {sequence.identity_record_sha256: sequence for sequence in self.sequences}
        )

    def lookup(self, identity_record_sha256: str) -> MaterializedCalibrationSequence:
        digest = _require_sha256(
            identity_record_sha256,
            context="calibration materialization identity lookup",
        )
        try:
            return self.by_identity_record_sha256[digest]
        except KeyError as error:
            raise KeyError(f"unknown calibration identity record: {digest}") from error

    @property
    def token_sequence_manifest_sha256(self) -> str:
        commitments = [
            {
                "identity_record_sha256": sequence.identity_record_sha256,
                "prompt_token_ids_sha256": _token_hash(sequence.prompt_token_ids),
                "target_token_ids_sha256": _token_hash(sequence.target_token_ids),
                "sequence_token_ids_sha256": _token_hash(sequence.sequence_token_ids),
                "sequence_length": len(sequence.sequence_token_ids),
            }
            for sequence in self.sequences
        ]
        return sha256_bytes(canonical_json_bytes(commitments))


@dataclass(frozen=True, slots=True)
class MaterializedStageASequence:
    """One authenticated Stage-A identity record and its exact token sequence.

    The record is the content-redacted capture projection authenticated by a
    promoted resolver-v9 Stage-A artifact.  Raw source rows, formatted prompts,
    targets, and receipt bodies are never retained on this object.
    """

    _identity_record_bytes: bytes = dataclass_field(repr=False)
    prompt_token_ids: tuple[int, ...] = dataclass_field(repr=False)
    target_token_ids: tuple[int, ...] = dataclass_field(repr=False)
    _authentication_seal: object = dataclass_field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authentication_seal is not _STAGE_A_MATERIALIZATION_AUTHENTICATION_SEAL:
            raise ValueError(
                "Stage-A sequences may be created only by authenticated v6 materialization"
            )
        record = _strict_json(
            self._identity_record_bytes,
            context="materialized Stage-A identity record",
        )
        if canonical_json_bytes(record) != self._identity_record_bytes:
            raise ValueError("materialized Stage-A identity record is not canonical")
        expected_fields = set(resolver.IDENTITY_RECORD_PAYLOAD_FIELDS) | {"identity_record_sha256"}
        if set(record) != expected_fields:
            raise ValueError("materialized Stage-A identity record fields drifted")
        if not isinstance(self.prompt_token_ids, tuple) or not isinstance(
            self.target_token_ids, tuple
        ):
            raise TypeError("materialized Stage-A token IDs must be tuples")
        for side, token_ids in (
            ("prompt", self.prompt_token_ids),
            ("target", self.target_token_ids),
        ):
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in token_ids
            ):
                raise ValueError(
                    f"materialized Stage-A {side} token IDs must be non-negative integers"
                )
        if not self.prompt_token_ids:
            raise ValueError("materialized Stage-A prompt cannot be empty")
        if len(self.target_token_ids) < 2:
            raise ValueError("materialized Stage-A target must contain at least two tokens")

        for name in (
            "source_content_sha256",
            "formatted_content_sha256",
            "prompt_token_ids_sha256",
            "target_token_ids_sha256",
            "sequence_token_ids_sha256",
            "tokenizer_manifest_sha256",
            "anchor_manifest_sha256",
            "identity_record_sha256",
        ):
            _require_sha256(record.get(name), context=f"materialized Stage-A {name}")
        receipt_hash = record.get("generator_receipt_sha256")
        if receipt_hash is not None:
            _require_sha256(
                receipt_hash,
                context="materialized Stage-A generator receipt",
            )

        sequence_ids = self.sequence_token_ids
        span = record.get("token_span")
        if not isinstance(span, Mapping) or set(span) != resolver.TOKEN_SPAN_FIELDS:
            raise ValueError("materialized Stage-A identity token span is malformed")
        expected_span = {
            "prefill_start": 0,
            "prefill_stop": len(self.prompt_token_ids),
            "scored_start": len(self.prompt_token_ids),
            "scored_stop": len(sequence_ids),
            "cache_exposed_start": len(self.prompt_token_ids) + 1,
            "cache_exposed_stop": len(sequence_ids),
        }
        if dict(span) != expected_span:
            raise ValueError("materialized Stage-A full token span differs from its tokens")
        continuation_count = expected_span["scored_stop"] - expected_span["scored_start"]
        exposed_count = expected_span["cache_exposed_stop"] - expected_span["cache_exposed_start"]
        if continuation_count != len(self.target_token_ids) or exposed_count != (
            continuation_count - 1
        ):
            raise ValueError("materialized Stage-A cache-exposed transition count drifted")

        family = record.get("family")
        if family == "pg19":
            if len(self.prompt_token_ids) != 4_096 or len(self.target_token_ids) != 128:
                raise ValueError("materialized Stage-A PG19 must contain 4096+128 tokens")
            if exposed_count != 127:
                raise ValueError("materialized Stage-A PG19 must expose 127 transitions")
        elif family == "ruler":
            if record.get("configured_length") != 4_096:
                raise ValueError("materialized Stage-A RULER length must be configured at 4096")
        elif family == "humaneval_plus":
            if len(self.target_token_ids) > 128:
                raise ValueError("materialized Stage-A HumanEval+ target exceeds 128 tokens")
        else:
            raise ValueError("materialized Stage-A identity family is not in the frozen inventory")

        if (
            record.get("identity_record_sha256") != resolver.identity_record_sha256(record)
            or record.get("prompt_token_ids_sha256") != _token_hash(self.prompt_token_ids)
            or record.get("target_token_ids_sha256") != _token_hash(self.target_token_ids)
            or record.get("sequence_token_ids_sha256") != _token_hash(sequence_ids)
            or record.get("sequence_length") != len(sequence_ids)
            or record.get("anchor_manifest_sha256")
            != _anchor_manifest_hash(
                canonical_id=str(record.get("canonical_id")),
                sequence_ids=sequence_ids,
                token_span=expected_span,
            )
            or record.get("fisher_boundary")
            != resolver.build_fisher_boundary_contract(sequence_ids)
        ):
            raise ValueError("materialized Stage-A tokens differ from their identity record")

        forbidden = {
            "answer_prefix",
            "canonical_solution",
            "code",
            "formatted_payload",
            "input",
            "outputs",
            "prompt",
            "source_payload",
            "text",
        }
        if forbidden & set(record):
            raise ValueError("materialized Stage-A identity record contains forbidden raw content")

    @property
    def identity_record(self) -> dict[str, Any]:
        """Return a fresh content-redacted copy of the authenticated record."""

        return _strict_json(
            self._identity_record_bytes,
            context="materialized Stage-A identity record",
        )

    @property
    def identity_record_sha256(self) -> str:
        return str(self.identity_record["identity_record_sha256"])

    @property
    def sequence_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.target_token_ids

    @property
    def cache_exposed_transition_count(self) -> int:
        return len(self.target_token_ids) - 1


@dataclass(frozen=True, slots=True)
class StageAIdentityMaterialization:
    """Twelve content-redacted sequences authenticated by a frozen v6 identity."""

    sequences: tuple[MaterializedStageASequence, ...]
    tokenizer_manifest_sha256: str
    capture_input_sha256: str
    frozen_identity_file_sha256: str
    frozen_identity_canonical_evidence_sha256: str
    calibration_binding_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequences, tuple) or len(self.sequences) != 12:
            raise ValueError("Stage-A materialization must contain exactly 12 sequences")
        tokenizer_hash = _require_sha256(
            self.tokenizer_manifest_sha256,
            context="Stage-A materialization tokenizer manifest",
        )
        for name, value in (
            ("capture input", self.capture_input_sha256),
            ("frozen identity file", self.frozen_identity_file_sha256),
            (
                "frozen identity canonical evidence",
                self.frozen_identity_canonical_evidence_sha256,
            ),
            ("calibration binding file", self.calibration_binding_file_sha256),
        ):
            _require_sha256(value, context=f"Stage-A materialization {name}")

        records = [sequence.identity_record for sequence in self.sequences]
        expected_order = [
            (family, rank) for family in ("pg19", "ruler", "humaneval_plus") for rank in range(4)
        ]
        identities = [sequence.identity_record_sha256 for sequence in self.sequences]
        if len(set(identities)) != len(identities):
            raise ValueError("Stage-A materialization contains duplicate identities")
        actual_order = [
            (str(record["family"]), int(record["selection_rank"])) for record in records
        ]
        if actual_order != expected_order:
            raise ValueError("Stage-A materialization is not ordered by family then rank")
        if any(record["tokenizer_manifest_sha256"] != tokenizer_hash for record in records):
            raise ValueError("Stage-A materialization tokenizer commitments differ")
        if records != sorted(records, key=resolver._record_sort_key):
            raise ValueError("Stage-A materialization records are not in canonical order")

    @property
    def identity_records(self) -> tuple[dict[str, Any], ...]:
        """Return fresh content-redacted record copies in frozen identity order."""

        return tuple(sequence.identity_record for sequence in self.sequences)

    @property
    def by_identity_record_sha256(self) -> Mapping[str, MaterializedStageASequence]:
        """Return an immutable digest lookup for the twelve frozen records."""

        return MappingProxyType(
            {sequence.identity_record_sha256: sequence for sequence in self.sequences}
        )

    def lookup(self, identity_record_sha256: str) -> MaterializedStageASequence:
        digest = _require_sha256(
            identity_record_sha256,
            context="Stage-A materialization identity lookup",
        )
        try:
            return self.by_identity_record_sha256[digest]
        except KeyError as error:
            raise KeyError(f"unknown Stage-A identity record: {digest}") from error

    @property
    def token_sequence_manifest_sha256(self) -> str:
        commitments = [
            {
                "identity_record_sha256": sequence.identity_record_sha256,
                "prompt_token_ids_sha256": _token_hash(sequence.prompt_token_ids),
                "target_token_ids_sha256": _token_hash(sequence.target_token_ids),
                "sequence_token_ids_sha256": _token_hash(sequence.sequence_token_ids),
                "sequence_length": len(sequence.sequence_token_ids),
                "cache_exposed_transition_count": sequence.cache_exposed_transition_count,
            }
            for sequence in self.sequences
        ]
        return sha256_bytes(canonical_json_bytes(commitments))


TokenCaptureSink = dict[tuple[str, str], tuple[tuple[int, ...], tuple[int, ...]]]


class CaptureSource(Protocol):
    """Narrow, mockable source surface used by the deterministic capturer."""

    def source_heads(self) -> Mapping[str, str]: ...

    def tokenizer_material(self) -> TokenizerMaterial: ...

    def mbpp_train_rows(self) -> Sequence[Mapping[str, Any]]: ...

    def pg19_projection(self, split: str) -> Sequence[ProjectionRow]: ...

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> Mapping[str, Any]: ...

    def ruler_generator_files(self) -> Mapping[str, bytes]: ...

    def ruler_generation_manifest_bytes(self) -> bytes: ...

    def ruler_receipt_bytes(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> bytes: ...

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
    token_sink: TokenCaptureSink | None = None,
) -> dict[str, Any]:
    prompt_token_ids = tuple(prompt_ids)
    target_token_ids = tuple(target_ids)
    sequence_ids = prompt_token_ids + target_token_ids
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
    record = {
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
        "fisher_boundary": resolver.build_fisher_boundary_contract(sequence_ids),
    }
    if token_sink is not None:
        key = (family, canonical_id)
        if key in token_sink:
            raise ValueError(f"duplicate materialized token identity: {family}/{canonical_id}")
        token_sink[key] = (prompt_token_ids, target_token_ids)
    return record


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
    token_sink: TokenCaptureSink | None = None,
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
                token_sink=token_sink,
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
    token_sink: TokenCaptureSink | None = None,
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
                token_sink=token_sink,
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


def _ruler_runtime_packages() -> dict[str, str]:
    raw = RULER_REQUIREMENTS_PATH.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("RULER runtime requirements must be UTF-8") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError("RULER runtime requirements must use exact == pins")
        name, version = line.split("==")
        if not name or not re.fullmatch(r"[0-9]+(?:\.[0-9A-Za-z]+)+", version):
            raise ValueError("RULER runtime requirement is malformed")
        if name in result:
            raise ValueError("RULER runtime requirements contain a duplicate package")
        result[name] = version
    if not result:
        raise ValueError("RULER runtime requirements cannot be empty")
    return result


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_bound_file(
    value: object,
    *,
    context: str,
    expected_name: str | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_fields(value, AUXILIARY_FILE_FIELDS, context=context)
    name = _require_string(value["name"], context=f"{context}.name")
    if expected_name is not None and name != expected_name:
        raise ValueError(f"{context} name drifted")
    if (
        "\\" in name
        or "\0" in name
        or "\n" in name
        or "\r" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"{context} name must be a canonical relative POSIX path")
    return {
        "name": name,
        "sha256": _require_sha256(value["sha256"], context=f"{context}.sha256"),
        "size_bytes": _require_int(
            value["size_bytes"],
            context=f"{context}.size_bytes",
            minimum=0 if allow_empty else 1,
        ),
    }


def _normalize_ruler_runtime_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("RULER runtime manifest must be an object")
    expected_fields = frozenset(
        {
            "schema",
            "python",
            "implementation",
            "cache_tag",
            "platform",
            "machine",
            "flags",
            "startup_policy",
            "excluded_startup_files",
            "source_python",
            "source_pyvenv_config",
            "python_runtime_files",
            "executable",
            "packages",
            "installed_distributions",
            "distribution_file_inventory",
            "forbidden_modules",
        }
    )
    _require_exact_fields(value, expected_fields, context="RULER runtime manifest")
    if value["schema"] != RULER_RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("RULER runtime manifest schema drifted")
    if value["python"] != RULER_RUNTIME_PYTHON_VERSION or value["implementation"] != "cpython":
        raise ValueError("RULER runtime Python identity drifted")
    for field in ("cache_tag", "platform", "machine"):
        _require_string(value[field], context=f"RULER runtime {field}")
    expected_flags = {"ignore_environment": 1, "isolated": 1, "no_user_site": 1}
    flags = value["flags"]
    if (
        not isinstance(flags, Mapping)
        or set(flags) != set(expected_flags)
        or any(type(flags[field]) is not int or flags[field] != 1 for field in expected_flags)
    ):
        raise ValueError("RULER runtime isolation flags drifted")
    startup_policy = value["startup_policy"]
    if (
        startup_policy != RULER_SEALED_STARTUP_POLICY
        or not isinstance(startup_policy, Mapping)
        or any(
            type(startup_policy[field]) is not int or startup_policy[field] != 1
            for field in ("dont_write_bytecode", "no_site", "utf8_mode")
        )
        or any(
            type(startup_policy[field]) is not bool or startup_policy[field] is not False
            for field in ("site_loaded", "virtualenv_hook_loaded")
        )
    ):
        raise ValueError("RULER runtime sealed-startup policy drifted")
    excluded_startup_files = [
        {
            "name": name,
            "sha256": digest,
            "size_bytes": size,
        }
        for name, (size, digest) in sorted(RULER_EXCLUDED_VIRTUALENV_STARTUP_FILES.items())
    ]
    if value["excluded_startup_files"] != excluded_startup_files:
        raise ValueError("RULER excluded startup-file identity drifted")
    source_python = _normalize_bound_file(
        value["source_python"],
        context="RULER source Python launcher",
        expected_name="source/python.exe",
    )
    source_pyvenv_config = _normalize_bound_file(
        value["source_pyvenv_config"],
        context="RULER source pyvenv config",
        expected_name="source/pyvenv.cfg",
    )
    raw_python_runtime_files = value["python_runtime_files"]
    if isinstance(raw_python_runtime_files, (str, bytes)) or not isinstance(
        raw_python_runtime_files, Sequence
    ):
        raise ValueError("RULER Python runtime file inventory must be an array")
    python_runtime_files = [
        _normalize_bound_file(
            item,
            context="RULER Python runtime file",
            allow_empty=True,
        )
        for item in raw_python_runtime_files
    ]
    runtime_names = [str(item["name"]) for item in python_runtime_files]
    if (
        runtime_names != sorted(runtime_names)
        or len(runtime_names) != len(set(runtime_names))
        or not {"python.exe", "python3.dll", "python311.dll"} <= set(runtime_names)
    ):
        raise ValueError("RULER Python runtime file inventory drifted")
    executable = _normalize_bound_file(
        value["executable"], context="RULER runtime executable", expected_name="python.exe"
    )
    packages = _ruler_runtime_packages()
    if value["packages"] != packages:
        raise ValueError("RULER runtime package versions drifted")
    installed = {_canonical_distribution_name(name): version for name, version in packages.items()}
    if value["installed_distributions"] != installed:
        raise ValueError("RULER runtime installed-distribution set drifted")
    if value["forbidden_modules"] != {name: False for name in RULER_FORBIDDEN_RUNTIME_MODULES}:
        raise ValueError("RULER runtime contains a forbidden model framework")
    inventory = value["distribution_file_inventory"]
    if not isinstance(inventory, Mapping) or set(inventory) != set(packages):
        raise ValueError("RULER runtime distribution-file inventory drifted")
    normalized_inventory: dict[str, Any] = {}
    for package_name in packages:
        raw_distribution = inventory[package_name]
        if not isinstance(raw_distribution, Mapping):
            raise ValueError("RULER package-code inventory must be an object")
        _require_exact_fields(
            raw_distribution,
            frozenset({"canonical_name", "version", "record_sha256", "record_size_bytes", "files"}),
            context=f"RULER package-code inventory {package_name}",
        )
        if (
            raw_distribution["canonical_name"] != _canonical_distribution_name(package_name)
            or raw_distribution["version"] != packages[package_name]
        ):
            raise ValueError(f"RULER package-code identity drifted for {package_name}")
        record_sha256 = _require_sha256(
            raw_distribution["record_sha256"],
            context=f"RULER {package_name} RECORD SHA-256",
        )
        record_size = _require_int(
            raw_distribution["record_size_bytes"],
            context=f"RULER {package_name} RECORD size",
            minimum=1,
        )
        raw_files = raw_distribution["files"]
        if isinstance(raw_files, (str, bytes)) or not isinstance(raw_files, Sequence):
            raise ValueError(f"RULER {package_name} file inventory must be an array")
        files = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise ValueError(f"RULER {package_name} installed file must be an object")
            _require_exact_fields(
                raw_file,
                frozenset({"path", "sha256", "size_bytes"}),
                context=f"RULER {package_name} installed file",
            )
            path = _require_string(raw_file["path"], context=f"RULER {package_name} installed path")
            if (
                "\\" in path
                or "\0" in path
                or "\n" in path
                or "\r" in path
                or path.startswith("/")
                or re.match(r"^[A-Za-z]:", path)
            ):
                raise ValueError(f"RULER {package_name} installed path is not portable")
            files.append(
                {
                    "name": path,
                    "sha256": _require_sha256(
                        raw_file["sha256"],
                        context=f"RULER {package_name} installed SHA-256",
                    ),
                    "size_bytes": _require_int(
                        raw_file["size_bytes"],
                        context=f"RULER {package_name} installed size",
                        minimum=0,
                    ),
                }
            )
        paths = [item["name"] for item in files]
        if paths != sorted(paths) or len(paths) != len(set(paths)) or not paths:
            raise ValueError(f"RULER {package_name} installed paths drifted")
        records = [item for item in files if item["name"].endswith(".dist-info/RECORD")]
        if len(records) != 1 or (
            records[0]["sha256"] != record_sha256 or records[0]["size_bytes"] != record_size
        ):
            raise ValueError(f"RULER {package_name} RECORD binding drifted")
        normalized_inventory[package_name] = {
            "canonical_name": raw_distribution["canonical_name"],
            "version": raw_distribution["version"],
            "record_sha256": record_sha256,
            "record_size_bytes": record_size,
            "files": [
                {"path": item["name"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
                for item in files
            ],
        }
    return {
        "schema": RULER_RUNTIME_MANIFEST_SCHEMA,
        "python": value["python"],
        "implementation": value["implementation"],
        "cache_tag": value["cache_tag"],
        "platform": value["platform"],
        "machine": value["machine"],
        "flags": expected_flags,
        "startup_policy": dict(RULER_SEALED_STARTUP_POLICY),
        "excluded_startup_files": excluded_startup_files,
        "source_python": source_python,
        "source_pyvenv_config": source_pyvenv_config,
        "python_runtime_files": python_runtime_files,
        "executable": executable,
        "packages": packages,
        "installed_distributions": installed,
        "distribution_file_inventory": normalized_inventory,
        "forbidden_modules": dict(value["forbidden_modules"]),
    }


def _expected_ruler_static_inputs(
    *,
    source_manifest: Sequence[Mapping[str, Any]],
    runtime_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, (size, digest) in RULER_EXPECTED_CORPORA.items():
        entries.append({"name": f"corpora/{name}", "sha256": digest, "size_bytes": size})
    for name, (size, digest) in RULER_EXPECTED_PACKAGE_RESOURCES.items():
        entries.append({"name": f"packages/{name}", "sha256": digest, "size_bytes": size})
    for name, (size, digest) in RULER_EXPECTED_TOKENIZER_ASSETS.items():
        entries.append({"name": f"tokenizer/{name}", "sha256": digest, "size_bytes": size})
    requirements = RULER_REQUIREMENTS_PATH.read_bytes()
    launcher = RULER_LAUNCHER_PATH.read_bytes()
    entries.extend(
        (
            {
                "name": "runtime/package-manifest.json",
                "sha256": sha256_bytes(canonical_json_bytes(runtime_manifest)),
                "size_bytes": len(canonical_json_bytes(runtime_manifest)),
            },
            {
                "name": "runtime/requirements.txt",
                "sha256": sha256_bytes(requirements),
                "size_bytes": len(requirements),
            },
            {
                "name": "launcher/generate_static_q468_ruler_receipts.py",
                "sha256": sha256_bytes(launcher),
                "size_bytes": len(launcher),
            },
            {
                "name": "ruler/source-manifest.json",
                "sha256": sha256_bytes(canonical_json_bytes(list(source_manifest))),
                "size_bytes": len(canonical_json_bytes(list(source_manifest))),
            },
        )
    )
    return sorted(entries, key=lambda item: item["name"])


def _replay_ruler_task_invariants(
    *,
    config: str,
    input_text: str,
    answer_prefix: str,
    outputs: Sequence[str],
) -> None:
    """Replay the frozen task semantics without trusting launcher-derived claims."""

    try:
        invariant = RULER_TASK_INVARIANTS[config]
    except KeyError as error:
        raise ValueError("RULER task invariant is unavailable") from error
    if (
        not input_text.startswith(invariant.input_prefix)
        or invariant.input_marker not in input_text
        or not input_text.endswith(invariant.input_suffix)
    ):
        raise ValueError(f"RULER {config} input task markers drifted")
    if not answer_prefix.startswith(invariant.answer_prefix_marker) or not answer_prefix.endswith(
        invariant.answer_prefix_suffix
    ):
        raise ValueError(f"RULER {config} answer-prefix boundaries drifted")
    if (
        invariant.expected_output_count is not None
        and len(outputs) != invariant.expected_output_count
    ):
        raise ValueError(
            f"RULER receipt {config} must contain exactly "
            f"{invariant.expected_output_count} required outputs"
        )
    if invariant.unique_outputs and len(set(outputs)) != len(outputs):
        raise ValueError(f"RULER {config} outputs must be unique")
    if invariant.output_pattern is not None and any(
        re.fullmatch(invariant.output_pattern, output) is None for output in outputs
    ):
        raise ValueError(f"RULER {config} output format drifted")
    if invariant.outputs_must_appear and any(output not in input_text for output in outputs):
        raise ValueError(f"RULER {config} required answer is absent from its input")


def _verify_ruler_raw_row(
    raw_data: bytes,
    *,
    receipt: Mapping[str, Any],
    tokenizer: Any,
) -> None:
    lines = raw_data.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("RULER raw validation must contain exactly one JSON row")
    row = _strict_json(lines[0], context="RULER raw validation row")
    config = str(receipt["config"])
    niah = config in RULER_NIAH_CONFIGS
    expected_fields = {
        "index",
        "input",
        "outputs",
        "length",
        "length_w_model_temp",
        "answer_prefix",
    } | ({"token_position_answer"} if niah else set())
    if set(row) != expected_fields:
        raise ValueError("RULER raw validation fields drifted")
    row_length = _require_int(row["length"], context="RULER raw validation length", minimum=1)
    row_length_with_template = _require_int(
        row["length_w_model_temp"],
        context="RULER raw validation length_w_model_temp",
        minimum=1,
    )
    row_index = _require_int(row["index"], context="RULER raw validation index")
    if (
        row["input"] != receipt["input"]
        or row["outputs"] != receipt["outputs"]
        or row["answer_prefix"] != receipt["answer_prefix"]
        or row_length != receipt["generator_reported_length"]
        or row_length_with_template != row_length
    ):
        raise ValueError("RULER receipt does not reproduce its raw generator row")
    input_text = _require_string(row["input"], context="RULER raw validation input")
    answer_prefix = _require_string(
        row["answer_prefix"],
        context="RULER raw validation answer_prefix",
        allow_empty=True,
    )
    raw_outputs = row["outputs"]
    if isinstance(raw_outputs, (str, bytes)) or not isinstance(raw_outputs, Sequence):
        raise ValueError("RULER raw validation outputs must be an array")
    outputs = tuple(
        _require_string(value, context="RULER raw validation output") for value in raw_outputs
    )
    if not outputs:
        raise ValueError("RULER raw validation outputs cannot be empty")
    _replay_ruler_task_invariants(
        config=config,
        input_text=input_text,
        answer_prefix=answer_prefix,
        outputs=outputs,
    )
    expected_length = (
        len(
            _encode(
                tokenizer,
                input_text + answer_prefix,
                add_special_tokens=False,
            )
        )
        + RULER_GENERATOR_TOKENS[config]
    )
    if row_length != expected_length:
        raise ValueError("RULER raw length disagrees with independent tokenization")
    if niah:
        first_output = outputs[0]
        index = input_text.find(first_output)
        if index < 0 or row_index != index:
            raise ValueError("RULER NIAH raw answer position drifted")
        expected_position = len(_encode(tokenizer, input_text[:index], add_special_tokens=False))
        token_position = _require_int(
            row["token_position_answer"],
            context="RULER raw validation token_position_answer",
        )
        if token_position != expected_position:
            raise ValueError("RULER NIAH raw token position drifted")
    elif row_index != 0:
        raise ValueError("RULER non-NIAH raw index must equal zero")


def _verify_complete_ruler_bundle(
    source: CaptureSource,
    *,
    phase: str,
    tokenizer_material: TokenizerMaterial,
    expected_generation_manifest_file_sha256: str | None = None,
) -> VerifiedRulerBundle:
    if phase not in resolver.ALLOWED_PHASES:
        raise ValueError(f"unsupported RULER verification phase: {phase!r}")
    raw_manifest = source.ruler_generation_manifest_bytes()
    generation_manifest_sha256 = sha256_bytes(raw_manifest)
    if expected_generation_manifest_file_sha256 is not None and (
        generation_manifest_sha256
        != _require_sha256(
            expected_generation_manifest_file_sha256,
            context="expected RULER generation manifest file SHA-256",
        )
    ):
        raise ValueError("RULER generation manifest differs from authenticated custody")
    manifest = _strict_json(raw_manifest, context="RULER generation manifest")
    if canonical_json_bytes(manifest) != raw_manifest:
        raise ValueError("RULER generation manifest is not canonical JSON")
    _require_exact_fields(
        manifest,
        frozenset(
            {
                "schema",
                "launcher_revision",
                "launcher_source",
                "ruler_revision",
                "source_manifest",
                "source_manifest_sha256",
                "runtime_manifest",
                "runtime_manifest_sha256",
                "static_inputs",
                "receipt_count",
                "receipts",
            }
        ),
        context="RULER generation manifest",
    )
    receipt_count = _require_int(
        manifest["receipt_count"],
        context="RULER generation manifest receipt_count",
        minimum=1,
    )
    if (
        manifest["schema"] != RULER_GENERATION_MANIFEST_SCHEMA
        or manifest["launcher_revision"] != RULER_LAUNCHER_REVISION
        or manifest["ruler_revision"] != resolver.RULER_REVISION
        or receipt_count != 20
    ):
        raise ValueError("RULER generation manifest identity drifted")
    generator_manifest = _ruler_generator_manifest(source.ruler_generator_files())
    if manifest["source_manifest"] != generator_manifest or manifest[
        "source_manifest_sha256"
    ] != sha256_bytes(canonical_json_bytes(generator_manifest)):
        raise ValueError("RULER generation source manifest drifted")
    runtime_manifest = _normalize_ruler_runtime_manifest(manifest["runtime_manifest"])
    if manifest["runtime_manifest"] != runtime_manifest or manifest[
        "runtime_manifest_sha256"
    ] != sha256_bytes(canonical_json_bytes(runtime_manifest)):
        raise ValueError("RULER runtime manifest binding drifted")
    expected_static = _expected_ruler_static_inputs(
        source_manifest=generator_manifest,
        runtime_manifest=runtime_manifest,
    )
    if manifest["static_inputs"] != expected_static:
        raise ValueError("RULER static-input inventory drifted")
    launcher_entry = _normalize_bound_file(
        manifest["launcher_source"],
        context="RULER launcher source",
        expected_name="launcher/generate_static_q468_ruler_receipts.py",
    )
    launcher_expected = next(
        item for item in expected_static if item["name"] == launcher_entry["name"]
    )
    if launcher_entry != launcher_expected:
        raise ValueError("RULER launcher source bytes drifted")
    material_files = {
        name: (len(data), sha256_bytes(data)) for name, data in tokenizer_material.files.items()
    }
    if material_files != RULER_EXPECTED_TOKENIZER_ASSETS:
        raise ValueError("capture tokenizer assets differ from the RULER tokenizer contract")
    required = required_ruler_receipts()
    raw_results = manifest["receipts"]
    if not isinstance(raw_results, list) or len(raw_results) != len(required):
        raise ValueError("RULER generation manifest must contain all 20 receipt results")
    if set(RULER_COMMAND_MANIFEST_SHA256_BY_FILENAME) != {
        str(item["filename"]) for item in required
    }:
        raise RuntimeError("frozen RULER command-manifest hash inventory is incomplete")
    verified: dict[str, dict[str, Any]] = {}
    static_by_name = {item["name"]: item for item in expected_static}
    result_fields = frozenset(
        {
            "category",
            "command_manifest",
            "command_manifest_file",
            "config",
            "configured_length",
            "filename",
            "generator_reported_length",
            "phase",
            "raw_validation_base64",
            "raw_validation_file",
            "seed",
            "sha256",
            "size_bytes",
        }
    )
    for expected, result in zip(required, raw_results, strict=True):
        if not isinstance(result, Mapping):
            raise ValueError("RULER receipt result must be an object")
        _require_exact_fields(result, result_fields, context="RULER receipt result")
        for field, minimum in (
            ("configured_length", 1),
            ("seed", 0),
            ("size_bytes", 1),
        ):
            _require_int(
                result[field],
                context=f"RULER receipt result {field}",
                minimum=minimum,
            )
        for field in (
            "category",
            "config",
            "configured_length",
            "filename",
            "phase",
            "seed",
        ):
            if result[field] != expected[field]:
                raise ValueError(f"RULER receipt result {field} drifted")
        filename = str(expected["filename"])
        _require_sha256(result["sha256"], context=f"RULER receipt result {filename} SHA-256")
        if expected["phase"] != phase:
            # The complete generation-manifest hash commits to all 20 results, but a
            # phase process must not open or semantically validate the other phase's
            # receipt bodies. Matching bytes are authenticated and decoded in their
            # own identity-capture phase; the offline evaluator reauthenticates them
            # again after its one-run reservation.
            continue
        _require_int(
            result["generator_reported_length"],
            context="RULER receipt result generator_reported_length",
            minimum=1,
        )
        receipt_bytes = source.ruler_receipt_bytes(
            category=str(expected["category"]),
            config=str(expected["config"]),
            configured_length=int(expected["configured_length"]),
            seed=int(expected["seed"]),
        )
        if result["sha256"] != sha256_bytes(receipt_bytes) or result["size_bytes"] != len(
            receipt_bytes
        ):
            raise ValueError(f"RULER receipt file identity drifted: {filename}")
        receipt_value = _strict_json(receipt_bytes, context=f"RULER receipt {filename}")
        if canonical_json_bytes(receipt_value) != receipt_bytes:
            raise ValueError(f"RULER receipt is not canonical: {filename}")
        receipt = _normalize_ruler_receipt(
            receipt_value,
            category=str(expected["category"]),
            config=str(expected["config"]),
            configured_length=int(expected["configured_length"]),
            seed=int(expected["seed"]),
        )
        if receipt != receipt_value:
            raise ValueError(f"RULER receipt normalization drifted: {filename}")
        if result["generator_reported_length"] != receipt["generator_reported_length"]:
            raise ValueError(f"RULER receipt file identity drifted: {filename}")
        command = result["command_manifest"]
        if not isinstance(command, Mapping):
            raise ValueError("RULER command manifest must be an object")
        _require_exact_fields(
            command,
            frozenset(
                {
                    "launcher_revision",
                    "launcher_source_sha256",
                    "ruler_revision",
                    "config",
                    "configured_length",
                    "seed",
                    "argv",
                    "shell",
                }
            ),
            context="RULER command manifest",
        )
        _require_int(
            command["configured_length"],
            context="RULER command configured_length",
            minimum=1,
        )
        _require_int(command["seed"], context="RULER command seed")
        if (
            command["launcher_revision"] != RULER_LAUNCHER_REVISION
            or command["launcher_source_sha256"] != launcher_entry["sha256"]
            or command["ruler_revision"] != resolver.RULER_REVISION
            or command["config"] != expected["config"]
            or command["configured_length"] != expected["configured_length"]
            or command["seed"] != expected["seed"]
            or command["shell"] is not False
        ):
            raise ValueError(f"RULER command identity drifted: {filename}")
        command_bytes = canonical_json_bytes(command)
        if sha256_bytes(command_bytes) != RULER_COMMAND_MANIFEST_SHA256_BY_FILENAME[filename]:
            raise ValueError(f"RULER command argv drifted: {filename}")
        command_file = _normalize_bound_file(
            result["command_manifest_file"],
            context="RULER command-manifest file",
            expected_name="generator/command-manifest.json",
        )
        if command_file != {
            "name": "generator/command-manifest.json",
            "sha256": sha256_bytes(command_bytes),
            "size_bytes": len(command_bytes),
        }:
            raise ValueError(f"RULER command-manifest bytes drifted: {filename}")
        encoded_raw = _require_string(
            result["raw_validation_base64"], context="RULER raw validation base64"
        )
        try:
            raw_data = base64.b64decode(encoded_raw, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("RULER raw validation is not canonical base64") from error
        if base64.b64encode(raw_data).decode("ascii") != encoded_raw:
            raise ValueError("RULER raw validation base64 encoding drifted")
        raw_file = _normalize_bound_file(
            result["raw_validation_file"],
            context="RULER raw-validation file",
            expected_name="generator/raw-validation.jsonl",
        )
        if raw_file != {
            "name": "generator/raw-validation.jsonl",
            "sha256": sha256_bytes(raw_data),
            "size_bytes": len(raw_data),
        }:
            raise ValueError(f"RULER raw-validation bytes drifted: {filename}")
        _verify_ruler_raw_row(raw_data, receipt=receipt, tokenizer=tokenizer_material.tokenizer)
        auxiliary_by_name = {item["name"]: item for item in receipt["auxiliary_files"]}
        expected_auxiliary = {
            **static_by_name,
            command_file["name"]: command_file,
            raw_file["name"]: raw_file,
        }
        if auxiliary_by_name != expected_auxiliary:
            raise ValueError(f"RULER receipt auxiliary inventory drifted: {filename}")
        verified[filename] = receipt
    return VerifiedRulerBundle(
        receipts=verified,
        generator_manifest=tuple(generator_manifest),
        generation_manifest_sha256=generation_manifest_sha256,
    )


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
    return resolver.ruler_canonical_id(
        category=category,
        config=config,
        configured_length=configured_length,
        seed=seed,
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
    for field, minimum in (
        ("configured_length", 1),
        ("seed", 0),
        ("sample_index", 0),
    ):
        _require_int(value[field], context=f"RULER receipt {field}", minimum=minimum)
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
    _replay_ruler_task_invariants(
        config=config,
        input_text=input_text,
        answer_prefix=answer_prefix,
        outputs=normalized_outputs,
    )
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
    *,
    phase: str,
    tokenizer: Any,
    tokenizer_manifest_sha256: str,
    bundle: VerifiedRulerBundle,
    token_sink: TokenCaptureSink | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    generator_manifest = list(bundle.generator_manifest)
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
            bundle.receipts[
                ruler_receipt_filename(
                    category=category,
                    config=config,
                    configured_length=configured_length,
                    seed=seed,
                )
            ],
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
                token_sink=token_sink,
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
        "capture_version": RULER_FORMATTER_FROZEN_CAPTURE_VERSION,
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
        "complete_generation_manifest_sha256": bundle.generation_manifest_sha256,
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
    token_sink: TokenCaptureSink | None = None,
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
                token_sink=token_sink,
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


def _normalize_runtime_authentication_context(
    value: object,
) -> _RuntimeAuthenticationContext:
    if not isinstance(value, Mapping):
        raise ValueError("runtime_authentication_context must be a mapping")
    _require_exact_fields(
        value,
        RUNTIME_AUTHENTICATION_CONTEXT_FIELDS,
        context="runtime_authentication_context",
    )
    base_runtime_root = value["base_runtime_root"]
    git_executable = value["git_executable"]
    staged_interpreter = value["staged_interpreter"]
    if (
        not isinstance(base_runtime_root, Path)
        or not isinstance(git_executable, Path)
        or not isinstance(staged_interpreter, Path)
    ):
        raise ValueError(
            "runtime authentication roots, Git executable, and interpreter must be Path values"
        )
    raw_roots = value["package_runtime_roots"]
    raw_import_paths = value["package_import_paths"]
    if not isinstance(raw_roots, Mapping) or not raw_roots:
        raise ValueError("package_runtime_roots must be a non-empty mapping")
    if not isinstance(raw_import_paths, Mapping) or not raw_import_paths:
        raise ValueError("package_import_paths must be a non-empty mapping")
    roots: dict[str, Path] = {}
    for raw_name, raw_path in raw_roots.items():
        if not isinstance(raw_name, str) or _RUNTIME_ROOT_NAME_RE.fullmatch(raw_name) is None:
            raise ValueError("package runtime root name is not canonical")
        if not isinstance(raw_path, Path):
            raise ValueError("package runtime root must be a Path")
        roots[raw_name] = raw_path
    import_paths: dict[str, str] = {}
    for raw_name, raw_path in raw_import_paths.items():
        if not isinstance(raw_name, str) or _RUNTIME_ROOT_NAME_RE.fullmatch(raw_name) is None:
            raise ValueError("package import-path name is not canonical")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise ValueError("package import path must be canonical relative POSIX text")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_path
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("package import path must be canonical relative POSIX text")
        import_paths[raw_name] = raw_path
    if set(roots) != set(import_paths):
        raise ValueError("package runtime-root and import-path names differ")
    return _RuntimeAuthenticationContext(
        base_runtime_root=base_runtime_root,
        git_executable=git_executable,
        staged_interpreter=staged_interpreter,
        package_runtime_roots=MappingProxyType(dict(sorted(roots.items()))),
        package_import_paths=MappingProxyType(dict(sorted(import_paths.items()))),
    )


def _runner_source_manifest_entry(
    source_manifest: Mapping[str, object],
) -> Mapping[str, object]:
    raw_paths = source_manifest.get("paths")
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        raise RuntimeError("authenticated source manifest paths are missing")
    relative = "scripts/run_static_q468_calibration.py"
    matches = [
        item for item in raw_paths if isinstance(item, Mapping) and item.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError("calibration runner is absent from the authenticated source manifest")
    entry = matches[0]
    resolver.require_sha256(
        entry.get("raw_sha256"),
        context="authenticated calibration runner source SHA-256",
    )
    return entry


def _load_calibration_runner_module(
    source_manifest: Mapping[str, object],
) -> Any:
    """Execute only exact runner bytes authenticated by the source artifact."""

    if _CALIBRATION_RUNNER_MODULE_NAME in sys.modules:
        raise RuntimeError("refusing a preloaded Experiment 013 calibration runner")
    entry = _runner_source_manifest_entry(source_manifest)
    runner_bytes = _bundle_stable_descendant_bytes(
        REPOSITORY_ROOT,
        "scripts/run_static_q468_calibration.py",
        context="calibration runner source",
    )
    runner_sha256 = sha256_bytes(runner_bytes)
    if runner_sha256 != entry["raw_sha256"]:
        raise RuntimeError("calibration runner source differs from the authenticated manifest")
    try:
        code = compile(
            runner_bytes,
            str(CALIBRATION_RUNNER_PATH),
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError) as error:
        raise RuntimeError("authenticated calibration runner source cannot be compiled") from error
    module = ModuleType(_CALIBRATION_RUNNER_MODULE_NAME)
    module.__file__ = str(CALIBRATION_RUNNER_PATH)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__recurquant_authenticated_source_sha256__"] = runner_sha256
    module.__dict__["__recurquant_authenticated_source_size_bytes__"] = len(runner_bytes)
    sys.modules[_CALIBRATION_RUNNER_MODULE_NAME] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)
        raise
    return module


def _decode_repository_source_artifact(
    artifacts: Mapping[str, bytes],
) -> _DecodedRepositorySourceArtifact:
    """Decode the canonical source artifact without executing runner code."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        resolver.EXECUTION_BINDING_FIELDS
    ):
        raise ValueError("all four verified execution-binding artifacts are required")
    for field, data in artifacts.items():
        if not isinstance(data, bytes) or not data:
            raise ValueError(f"execution-binding artifact {field} must be non-empty bytes")
    source_bytes = artifacts["repository_source_manifest_file_sha256"]
    try:
        from recurquant import experiment013_source
    except ImportError as error:  # pragma: no cover - installation guard
        raise RuntimeError("Experiment 013 source validator is unavailable") from error
    source_value = _strict_json(source_bytes, context="repository source manifest")
    normalized_source = experiment013_source.validate_experiment013_source_manifest(source_value)
    if (
        experiment013_source.canonical_experiment013_source_manifest_bytes(normalized_source)
        != source_bytes
    ):
        raise ValueError("repository source manifest is not canonical JSON")
    return _DecodedRepositorySourceArtifact(
        manifest_file_sha256=sha256_bytes(source_bytes),
        source_manifest=MappingProxyType(dict(normalized_source)),
        source_module=experiment013_source,
    )


def _decode_execution_binding_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    runner: Any,
    source_artifact: _DecodedRepositorySourceArtifact | None = None,
) -> _DecodedExecutionBindingArtifacts:
    """Strictly decode each execution artifact before binding its exact bytes."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        resolver.EXECUTION_BINDING_FIELDS
    ):
        raise ValueError("all four verified execution-binding artifacts are required")
    for field, data in artifacts.items():
        if not isinstance(data, bytes) or not data:
            raise ValueError(f"execution-binding artifact {field} must be non-empty bytes")
    source_bytes = artifacts["repository_source_manifest_file_sha256"]
    runtime_bytes = artifacts["calibration_runtime_manifest_file_sha256"]
    model_bytes = artifacts["model_file_manifest_file_sha256"]
    parquet_bytes = artifacts["parquet_materialization_manifest_file_sha256"]
    try:
        from recurquant import experiment013_parquet
    except ImportError as error:  # pragma: no cover - installation guard
        raise RuntimeError("Experiment 013 source/Parquet validators are unavailable") from error
    decoded_source = source_artifact or _decode_repository_source_artifact(artifacts)
    if decoded_source.manifest_file_sha256 != sha256_bytes(source_bytes):
        raise ValueError("repository source manifest changed during artifact decoding")
    runtime_manifest = runner.parse_calibration_runtime_manifest(runtime_bytes)
    model_manifest = runner.parse_model_file_manifest(model_bytes)
    if runtime_manifest.file_sha256 != sha256_bytes(runtime_bytes):
        raise ValueError("calibration runtime manifest file identity drifted")
    if (
        model_manifest.file_sha256 != sha256_bytes(model_bytes)
        or model_manifest.model_id != resolver.PRIMARY_MODEL_ID
        or model_manifest.revision != resolver.PRIMARY_MODEL_REVISION
        or model_manifest.transformers_version != resolver.TRANSFORMERS_VERSION
    ):
        raise ValueError("model file manifest does not match the frozen primary model")
    parquet_path = Path(experiment013_parquet.EXPERIMENT013_PARQUET_MANIFEST_PATH)
    if (
        parquet_path.resolve(strict=True)
        != PARQUET_MATERIALIZATION_MANIFEST_PATH.resolve(strict=True)
        or experiment013_parquet.EXPERIMENT013_PARQUET_MANIFEST_SHA256
        != resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
        or sha256_bytes(parquet_bytes) != resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
        or parquet_bytes != parquet_path.read_bytes()
    ):
        raise ValueError("Parquet materialization manifest file identity drifted")
    experiment013_parquet.load_experiment013_parquet_manifest(parquet_path)
    bindings = {
        field: sha256_bytes(artifacts[field]) for field in sorted(resolver.EXECUTION_BINDING_FIELDS)
    }
    return _DecodedExecutionBindingArtifacts(
        bindings=MappingProxyType(bindings),
        source_manifest=decoded_source.source_manifest,
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        source_module=decoded_source.source_module,
        parquet_module=experiment013_parquet,
    )


def _validate_execution_binding_artifacts(
    artifacts: Mapping[str, bytes],
) -> dict[str, str]:
    """Strictly decode all four artifacts without retaining an imported runner."""

    source_artifact = _decode_repository_source_artifact(artifacts)
    runner = _load_calibration_runner_module(source_artifact.source_manifest)
    try:
        decoded = _decode_execution_binding_artifacts(
            artifacts,
            runner=runner,
            source_artifact=source_artifact,
        )
        return dict(decoded.bindings)
    finally:
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)


def _verify_loaded_runner_source(
    runner: Any,
    source_manifest: Mapping[str, object],
) -> None:
    entry = _runner_source_manifest_entry(source_manifest)
    raw_file = getattr(runner, "__file__", None)
    if not isinstance(raw_file, (str, os.PathLike)):
        raise RuntimeError("calibration runner is absent from the authenticated source manifest")
    declared = Path(raw_file)
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("calibration runner source is unavailable") from error
    live_bytes = _bundle_stable_descendant_bytes(
        REPOSITORY_ROOT,
        "scripts/run_static_q468_calibration.py",
        context="loaded calibration runner source",
    )
    if (
        declared.is_symlink()
        or resolved != CALIBRATION_RUNNER_PATH.resolve(strict=True)
        or getattr(runner, "__recurquant_authenticated_source_sha256__", None)
        != entry["raw_sha256"]
        or getattr(runner, "__recurquant_authenticated_source_size_bytes__", None)
        != len(live_bytes)
        or sha256_bytes(live_bytes) != entry["raw_sha256"]
    ):
        raise RuntimeError("loaded calibration runner source bytes drifted")


def _validate_runtime_context_for_manifest(
    context: _RuntimeAuthenticationContext,
    runtime_manifest: Any,
) -> None:
    expected_import_paths = {item.name: item.import_path for item in runtime_manifest.package_roots}
    if dict(context.package_import_paths) != expected_import_paths:
        raise ValueError("runtime package import paths differ from the frozen manifest")
    if set(context.package_runtime_roots) != set(expected_import_paths):
        raise ValueError("runtime package root names differ from the frozen manifest")
    for name, raw_root in context.package_runtime_roots.items():
        try:
            root = raw_root.resolve(strict=True)
            import_root = (root / PurePosixPath(context.package_import_paths[name])).resolve(
                strict=True
            )
            import_root.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError("runtime package import path escapes or is unavailable") from error
        if not root.is_dir() or not import_root.is_dir():
            raise ValueError("runtime package root and import path must be directories")


def _authenticate_execution_binding_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    runtime_context: _RuntimeAuthenticationContext,
    model_file_manifest_attestation: bytes | None = None,
    previous: _AuthenticatedExecutionBindings | None = None,
) -> _AuthenticatedExecutionBindings:
    """Reauthenticate source, runtime, model metadata, and Parquet at point of use."""

    source_artifact = _decode_repository_source_artifact(artifacts)
    created_runner = previous is None
    if created_runner:
        runner = _load_calibration_runner_module(source_artifact.source_manifest)
    else:
        runner = previous.runner
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is not runner:
            raise RuntimeError("authenticated calibration runner module binding drifted")
        if runtime_context != previous.runtime_context:
            raise ValueError("runtime authentication context changed during capture")
    try:
        decoded = _decode_execution_binding_artifacts(
            artifacts,
            runner=runner,
            source_artifact=source_artifact,
        )
        if previous is not None and dict(decoded.bindings) != dict(previous.bindings):
            raise ValueError("execution-binding artifacts changed during capture")
        verified_source = decoded.source_module.verify_experiment013_source_manifest(
            decoded.source_manifest,
            repo_root=REPOSITORY_ROOT,
            git_executable=runtime_context.git_executable,
        )
        if verified_source != dict(decoded.source_manifest):
            raise RuntimeError("repository source verifier returned a different manifest")
        decoded.source_module.verify_loaded_experiment013_recurquant_modules(
            decoded.source_manifest,
            REPOSITORY_ROOT,
            (
                "recurquant.experiment013_source",
                "recurquant.experiment013_parquet",
            ),
        )
        _verify_loaded_runner_source(runner, decoded.source_manifest)
        _validate_runtime_context_for_manifest(runtime_context, decoded.runtime_manifest)
        authenticated_runtime = runner.authenticate_calibration_runtime(
            decoded.runtime_manifest,
            base_runtime_root=runtime_context.base_runtime_root,
            package_roots=runtime_context.package_runtime_roots,
            interpreter_path=runtime_context.staged_interpreter,
            git_executable_path=runtime_context.git_executable,
        )
        if (
            authenticated_runtime.manifest_file_sha256
            != decoded.bindings["calibration_runtime_manifest_file_sha256"]
        ):
            raise RuntimeError("runtime authenticator returned a different manifest identity")
        if model_file_manifest_attestation is None:
            live_model_manifest = runner.capture_model_file_manifest_from_hub(
                resolver.PRIMARY_MODEL_ID,
                resolver.PRIMARY_MODEL_REVISION,
                transformers_version=resolver.TRANSFORMERS_VERSION,
                token=False,
            )
        else:
            if not isinstance(model_file_manifest_attestation, bytes):
                raise TypeError("model file manifest attestation must be bytes")
            live_model_manifest = model_file_manifest_attestation
        if live_model_manifest != artifacts["model_file_manifest_file_sha256"]:
            raise ValueError("pinned model Hub metadata differs from the frozen manifest")
        return _AuthenticatedExecutionBindings(
            bindings=decoded.bindings,
            source_manifest=decoded.source_manifest,
            runtime_manifest=decoded.runtime_manifest,
            model_manifest=decoded.model_manifest,
            runner=runner,
            runtime_context=runtime_context,
        )
    except BaseException:
        if created_runner and sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)
        raise


def _capture_identity_input_with_tokens(
    *,
    phase: str,
    source: CaptureSource,
    calibration_binding: bytes | None = None,
    execution_binding_artifacts: Mapping[str, bytes] | None = None,
    runtime_authentication_context: Mapping[str, object] | None = None,
    expected_ruler_generation_manifest_file_sha256: str | None = None,
    collect_tokens: bool,
) -> tuple[dict[str, Any], TokenCaptureSink]:
    """Run the sole capture flow, optionally retaining formatter token IDs."""

    if phase in resolver.PROTECTED_STAGES:
        raise PermissionError(
            f"{phase} is protected; capture v{CAPTURE_VERSION} refuses it before source access"
        )
    if phase not in resolver.ALLOWED_PHASES:
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if phase == "stage_a" and calibration_binding is None:
        raise ValueError("Stage A requires a frozen calibration binding")
    if phase == "calibration" and calibration_binding is not None:
        raise ValueError("calibration capture forbids a Stage-A binding")
    normalized_calibration_binding: dict[str, str] | None = None
    authorized_execution_bindings: dict[str, str] | None = None
    if phase == "stage_a":
        if not isinstance(calibration_binding, bytes):
            raise ValueError("Stage-A calibration binding must be a verified artifact byte string")
        verified_calibration_binding = resolver.deserialize_stage_a_calibration_binding_artifact(
            calibration_binding
        )
        normalized_calibration_binding = dict(verified_calibration_binding.binding)
        authorized_execution_bindings = dict(verified_calibration_binding.execution_bindings)
        binding_manifest_sha256 = verified_calibration_binding.ruler_generation_manifest_file_sha256
        if (
            expected_ruler_generation_manifest_file_sha256 is not None
            and _require_sha256(
                expected_ruler_generation_manifest_file_sha256,
                context="expected RULER generation manifest file SHA-256",
            )
            != binding_manifest_sha256
        ):
            raise ValueError(
                "Stage-A caller and calibration binding name different RULER manifests"
            )
        expected_ruler_generation_manifest_file_sha256 = binding_manifest_sha256
    elif expected_ruler_generation_manifest_file_sha256 is not None:
        expected_ruler_generation_manifest_file_sha256 = _require_sha256(
            expected_ruler_generation_manifest_file_sha256,
            context="expected RULER generation manifest file SHA-256",
        )

    if runtime_authentication_context is None:
        runtime_provider = getattr(source, "runtime_authentication_context", None)
        if runtime_provider is None or not callable(runtime_provider):
            raise ValueError("capture requires an explicit sealed runtime authentication context")
        runtime_authentication_context = runtime_provider()
    runtime_context = _normalize_runtime_authentication_context(runtime_authentication_context)
    if execution_binding_artifacts is None:
        fixture_provider = getattr(source, "execution_binding_artifacts", None)
        if fixture_provider is None or not callable(fixture_provider):
            raise ValueError("capture requires all four verified execution-binding artifacts")
        execution_binding_artifacts = fixture_provider()
    model_attestation_provider = getattr(source, "model_file_manifest_attestation", None)
    model_attestation = (
        model_attestation_provider()
        if model_attestation_provider is not None and callable(model_attestation_provider)
        else None
    )
    authentication_kwargs: dict[str, bytes] = {}
    if model_attestation is not None:
        authentication_kwargs["model_file_manifest_attestation"] = model_attestation
    authentication = _authenticate_execution_binding_artifacts(
        execution_binding_artifacts,
        runtime_context=runtime_context,
        **authentication_kwargs,
    )
    if (
        authorized_execution_bindings is not None
        and dict(authentication.bindings) != authorized_execution_bindings
    ):
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is authentication.runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)
        raise ValueError("Stage-A execution artifacts differ from the calibration authorization")
    try:
        before = _validate_heads(source.source_heads(), context="pre-capture")
        material = source.tokenizer_material()
        tokenizer_contract, tokenizer_manifest_hash = _tokenizer_contract(material)
        ruler_bundle = _verify_complete_ruler_bundle(
            source,
            phase=phase,
            tokenizer_material=material,
            expected_generation_manifest_file_sha256=(
                expected_ruler_generation_manifest_file_sha256
            ),
        )
        token_sink: TokenCaptureSink | None = {} if collect_tokens else None
        mbpp_records, mbpp_manifest_hash = _capture_mbpp(
            source,
            phase=phase,
            tokenizer=material.tokenizer,
            tokenizer_manifest_sha256=tokenizer_manifest_hash,
            token_sink=token_sink,
        )
        pg19_records, pg19_manifest_hash, pg19_split = _capture_pg19(
            source,
            phase=phase,
            tokenizer=material.tokenizer,
            tokenizer_manifest_sha256=tokenizer_manifest_hash,
            token_sink=token_sink,
        )
        ruler_records, ruler_manifest_hash, ruler_formatter_hash = _capture_ruler(
            phase=phase,
            tokenizer=material.tokenizer,
            tokenizer_manifest_sha256=tokenizer_manifest_hash,
            bundle=ruler_bundle,
            token_sink=token_sink,
        )
        humaneval_records, humaneval_manifest_hash = _capture_humaneval(
            source,
            phase=phase,
            tokenizer=material.tokenizer,
            tokenizer_manifest_sha256=tokenizer_manifest_hash,
            token_sink=token_sink,
        )
        after = _validate_heads(source.source_heads(), context="post-capture")
        if after != before:  # Defensive even though both were pinned.
            raise ValueError("source HEAD changed during capture")
        _authenticate_execution_binding_artifacts(
            execution_binding_artifacts,
            runtime_context=runtime_context,
            previous=authentication,
            **authentication_kwargs,
        )

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
            "execution_bindings": dict(authentication.bindings),
            "records": [
                *mbpp_records,
                *pg19_records,
                *ruler_records,
                *humaneval_records,
            ],
            "model_weights_loaded": False,
        }
        if phase == "stage_a":
            assert normalized_calibration_binding is not None
            result["calibration_binding"] = normalized_calibration_binding
        if phase == "calibration":
            resolver.build_candidate(
                result,
                expected_revisions=dict(resolver.FROZEN_DATASET_REVISIONS),
            )
        return result, {} if token_sink is None else token_sink
    finally:
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is authentication.runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)


def capture_identity_input(
    *,
    phase: str,
    source: CaptureSource,
    calibration_binding: bytes | None = None,
    execution_binding_artifacts: Mapping[str, bytes] | None = None,
    runtime_authentication_context: Mapping[str, object] | None = None,
    expected_ruler_generation_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Capture one deterministic calibration or Stage-A resolver input."""

    result, _token_sink = _capture_identity_input_with_tokens(
        phase=phase,
        source=source,
        calibration_binding=calibration_binding,
        execution_binding_artifacts=execution_binding_artifacts,
        runtime_authentication_context=runtime_authentication_context,
        expected_ruler_generation_manifest_file_sha256=(
            expected_ruler_generation_manifest_file_sha256
        ),
        collect_tokens=False,
    )
    return result


def materialize_calibration_identity_sequences(
    *,
    source: CaptureSource,
    execution_binding_artifacts: Mapping[str, bytes] | None = None,
    runtime_authentication_context: Mapping[str, object] | None = None,
) -> CalibrationIdentityMaterialization:
    """Materialize exact calibration token IDs through the canonical capture path.

    This read-only API is the sole supported bridge for a future live adapter.
    It returns no dataset text, formatted prompts, tokenizer object, or RULER
    receipt body.  Every public identity record is a fresh copy reconstructed
    from canonical bytes and is addressable by its frozen record digest.
    """

    result, token_sink = _capture_identity_input_with_tokens(
        phase="calibration",
        source=source,
        execution_binding_artifacts=execution_binding_artifacts,
        runtime_authentication_context=runtime_authentication_context,
        collect_tokens=True,
    )
    sequences: list[MaterializedCalibrationSequence] = []
    remaining = dict(token_sink)
    for record in result["records"]:
        key = (str(record["family"]), str(record["canonical_id"]))
        try:
            prompt_ids, target_ids = remaining.pop(key)
        except KeyError as error:
            raise RuntimeError(f"missing materialized tokens for {key[0]}/{key[1]}") from error
        sequences.append(
            MaterializedCalibrationSequence(
                _identity_record_bytes=canonical_json_bytes(record),
                prompt_token_ids=prompt_ids,
                target_token_ids=target_ids,
            )
        )
    if remaining:
        raise RuntimeError("materialized token sink contains identities absent from capture")
    tokenizer_manifest_sha256 = sha256_bytes(canonical_json_bytes(result["tokenizer"]["files"]))
    return CalibrationIdentityMaterialization(
        sequences=tuple(sequences),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        capture_input_sha256=sha256_bytes(canonical_json_bytes(result)),
    )


def materialize_stage_a_identity_sequences(
    *,
    source: CaptureSource,
    frozen_stage_a_identity_artifact: bytes,
    calibration_binding_artifact: bytes,
    stage_a_capture_provenance_receipt: bytes,
    expected_stage_a_capture_provenance_receipt_sha256: str,
    expected_frozen_stage_a_identity_file_sha256: str | None = None,
    execution_binding_artifacts: Mapping[str, bytes] | None = None,
    runtime_authentication_context: Mapping[str, object] | None = None,
) -> StageAIdentityMaterialization:
    """Authenticate and materialize the exact twelve frozen Stage-A sequences.

    The promoted resolver-v9 artifact and its complete calibration binding are
    authenticated before any data source is touched.  The canonical capture is
    then replayed once with token retention, and every content-redacted record
    must equal the corresponding authenticated frozen record byte for byte.
    Candidate-only, unpromoted, altered, missing, duplicate, or reordered
    identity inputs therefore cannot enter Stage-A evaluation through this API.
    """

    if not isinstance(frozen_stage_a_identity_artifact, bytes):
        raise TypeError("frozen Stage-A identity artifact must be bytes")
    if not isinstance(calibration_binding_artifact, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    if (
        CAPTURE_VERSION != 9
        or resolver.RESOLVER_VERSION != 9
        or resolver.INPUT_SCHEMA != "recurquant.experiment013.identity-input.v5"
        or resolver.STAGE_A_FROZEN_SCHEMA != "recurquant.experiment013.identity-frozen.v6"
    ):
        raise RuntimeError("Stage-A materialization requires the resolver-v9 identity contract")

    frozen = resolver.deserialize_frozen_stage_a_identity_artifact(
        frozen_stage_a_identity_artifact,
        calibration_binding_artifact=calibration_binding_artifact,
        stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
        expected_stage_a_capture_provenance_receipt_sha256=(
            expected_stage_a_capture_provenance_receipt_sha256
        ),
        expected_file_sha256=expected_frozen_stage_a_identity_file_sha256,
    )
    result, token_sink = _capture_identity_input_with_tokens(
        phase="stage_a",
        source=source,
        calibration_binding=calibration_binding_artifact,
        execution_binding_artifacts=execution_binding_artifacts,
        runtime_authentication_context=runtime_authentication_context,
        collect_tokens=True,
    )
    if result.get("schema") != resolver.INPUT_SCHEMA or result.get("phase") != "stage_a":
        raise RuntimeError("Stage-A capture did not return the resolver-v5 input contract")

    replayed_candidate = resolver.build_candidate(
        result,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=calibration_binding_artifact,
        stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
        expected_stage_a_capture_provenance_receipt_sha256=(
            expected_stage_a_capture_provenance_receipt_sha256
        ),
    )
    frozen_document = _strict_json(
        frozen_stage_a_identity_artifact,
        context="authenticated frozen Stage-A identity artifact",
    )
    frozen_promotion = frozen_document["evidence"]["promotion"]
    if (
        sha256_bytes(canonical_json_bytes(replayed_candidate))
        != frozen_promotion["candidate_file_sha256"]
    ):
        raise ValueError("Stage-A capture lineage differs from the authenticated identity")

    frozen_records = tuple(
        {name: record[name] for name in resolver.RECORD_FIELDS} for record in frozen.records
    )
    captured_records = tuple(result["records"])
    if canonical_json_bytes(captured_records) != canonical_json_bytes(frozen_records):
        raise ValueError("materialized Stage-A records differ from the authenticated identity")
    if len(captured_records) != 12:
        raise ValueError("authenticated Stage-A identity must contain exactly 12 records")

    sequences: list[MaterializedStageASequence] = []
    remaining = dict(token_sink)
    for record in frozen_records:
        key = (str(record["family"]), str(record["canonical_id"]))
        try:
            prompt_ids, target_ids = remaining.pop(key)
        except KeyError as error:
            raise RuntimeError(
                f"missing materialized Stage-A tokens for {key[0]}/{key[1]}"
            ) from error
        sequences.append(
            MaterializedStageASequence(
                _identity_record_bytes=canonical_json_bytes(record),
                prompt_token_ids=prompt_ids,
                target_token_ids=target_ids,
                _authentication_seal=_STAGE_A_MATERIALIZATION_AUTHENTICATION_SEAL,
            )
        )
    if remaining:
        raise RuntimeError("Stage-A token sink contains identities absent from frozen capture")

    tokenizer_manifest_sha256 = sha256_bytes(canonical_json_bytes(result["tokenizer"]["files"]))
    if tokenizer_manifest_sha256 != frozen.tokenizer_manifest_sha256:
        raise ValueError("Stage-A tokenizer differs from the authenticated identity")
    return StageAIdentityMaterialization(
        sequences=tuple(sequences),
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        capture_input_sha256=sha256_bytes(canonical_json_bytes(result)),
        frozen_identity_file_sha256=frozen.file_sha256,
        frozen_identity_canonical_evidence_sha256=frozen.canonical_evidence_sha256,
        calibration_binding_file_sha256=sha256_bytes(calibration_binding_artifact),
    )


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


def _verify_live_ruler_receipt_inventory(root: Path) -> Mapping[str, Path]:
    """Return the exact complete non-redirected live receipt-file inventory."""

    unresolved = Path(os.path.abspath(root))

    def is_redirected(path: Path) -> bool:
        try:
            status = path.lstat()
        except OSError as error:
            raise ValueError(f"cannot authenticate RULER receipt path: {path}") from error
        return path.is_symlink() or bool(getattr(status, "st_file_attributes", 0) & 0x400)

    if is_redirected(unresolved) or not unresolved.is_dir():
        raise ValueError("RULER receipt root must be a regular non-redirected directory")
    entries = sorted(unresolved.iterdir(), key=lambda path: path.name)
    names = [path.name for path in entries]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("RULER receipt directory contains case-colliding names")
    observed = set(names)
    expected = {"generation-manifest.json"} | {
        str(item["filename"]) for item in required_ruler_receipts()
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"RULER receipt directory inventory drifted: missing={missing}, unexpected={unexpected}"
        )
    result: dict[str, Path] = {}
    for path in entries:
        if is_redirected(path) or not path.is_file():
            raise ValueError(
                f"RULER receipt entry must be a regular non-redirected file: {path.name}"
            )
        result[path.name] = path
    return MappingProxyType(result)


class LiveCaptureSource:
    """Pinned, read-only network source; RULER generated rows come from receipts."""

    def __init__(self, *, cache_dir: Path, ruler_receipt_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.ruler_receipt_dir = Path(os.path.abspath(ruler_receipt_dir))

    @staticmethod
    def _github_revision(repo_id: str, revision: str) -> str:
        commit_request = urllib.request.Request(
            f"https://api.github.com/repos/{repo_id}/commits/"
            f"{urllib.parse.quote(revision, safe='')}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "RecurQuant-Experiment-013-identity-capture",
            },
        )
        try:
            with urllib.request.urlopen(commit_request, timeout=30) as response:
                commit = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot resolve pinned GitHub revision for {repo_id}") from error
        resolved = _require_string(commit.get("sha"), context=f"{repo_id} pinned revision")
        if resolved != revision:
            raise ValueError(f"GitHub returned a different object for pinned {repo_id} revision")
        return resolved

    def source_heads(self) -> Mapping[str, str]:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("live capture requires huggingface-hub") from error
        api = HfApi(token=False, endpoint="https://huggingface.co")
        return {
            "primary_model": str(
                api.model_info(
                    resolver.PRIMARY_MODEL_ID,
                    revision=resolver.PRIMARY_MODEL_REVISION,
                    token=False,
                ).sha
            ),
            "mbpp": str(
                api.dataset_info(
                    resolver.MBPP_DATASET_ID,
                    revision=resolver.MBPP_REVISION,
                    token=False,
                ).sha
            ),
            "pg19": str(
                api.dataset_info(
                    resolver.PG19_DATASET_ID,
                    revision=resolver.PG19_REVISION,
                    token=False,
                ).sha
            ),
            "ruler": self._github_revision(resolver.RULER_SOURCE_ID, resolver.RULER_REVISION),
            "humaneval_plus": str(
                api.dataset_info(
                    resolver.HUMANEVAL_PLUS_DATASET_ID,
                    revision=resolver.HUMANEVAL_PLUS_REVISION,
                    token=False,
                ).sha
            ),
            "evalplus": self._github_revision(
                resolver.EVALPLUS_SOURCE_ID, resolver.EVALPLUS_SOURCE_REVISION
            ),
        }

    def tokenizer_material(self) -> TokenizerMaterial:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("live capture requires huggingface-hub and Transformers") from error
        api = HfApi(token=False, endpoint="https://huggingface.co")
        available = set(
            api.list_repo_files(
                resolver.PRIMARY_MODEL_ID,
                revision=resolver.PRIMARY_MODEL_REVISION,
                token=False,
            )
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
                    token=False,
                    endpoint="https://huggingface.co",
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
                token=False,
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
            token=False,
        )
        return tuple(dict(row) for row in rows)

    def pg19_projection(self, split: str) -> Sequence[ProjectionRow]:
        from recurquant import experiment013_parquet

        expected_count = 13_684 if split == "train" else 50
        projection = experiment013_parquet.project_experiment013_parquet_columns(
            "pg19",
            split,
            columns=("url",),
            expected_count=expected_count,
        )
        return tuple(
            ProjectionRow(
                _require_string(row.values[0], context=f"PG19 {split} url"),
                row.global_offset,
            )
            for row in projection.rows
        )

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> Mapping[str, Any]:
        from recurquant import experiment013_parquet

        selected = experiment013_parquet.read_experiment013_parquet_row(
            "pg19",
            split,
            offset,
            columns=("url", "text"),
        )
        row = dict(selected.values)
        if row.get("url") != expected_url:
            raise ValueError("immutable PG19 row URL does not match the pinned projection")
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

    def ruler_generation_manifest_bytes(self) -> bytes:
        inventory = _verify_live_ruler_receipt_inventory(self.ruler_receipt_dir)
        return inventory["generation-manifest.json"].read_bytes()

    def ruler_receipt_bytes(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> bytes:
        filename = self._receipt_filename(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        inventory = _verify_live_ruler_receipt_inventory(self.ruler_receipt_dir)
        try:
            path = inventory[filename]
        except KeyError as error:
            raise FileNotFoundError(
                f"requested RULER receipt is outside the exact frozen inventory: {filename}"
            ) from error
        return path.read_bytes()

    def ruler_receipt(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> Mapping[str, Any]:
        """Compatibility helper for callers inspecting one receipt directly."""

        raw = self.ruler_receipt_bytes(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        return _strict_json(raw, context="RULER receipt")

    def humaneval_projection(self) -> Sequence[ProjectionRow]:
        from recurquant import experiment013_parquet

        projection = experiment013_parquet.project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("task_id",),
            expected_count=164,
        )
        return tuple(
            ProjectionRow(
                _require_string(row.values[0], context="HumanEval+ task_id"),
                row.global_offset,
            )
            for row in projection.rows
        )

    def humaneval_row(self, *, offset: int, expected_task_id: str) -> Mapping[str, Any]:
        from recurquant import experiment013_parquet

        selected = experiment013_parquet.read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            offset,
            columns=("task_id", "prompt", "canonical_solution"),
        )
        row = dict(selected.values)
        if row.get("task_id") != expected_task_id:
            raise ValueError("immutable HumanEval+ task ID does not match projection")
        return row


STAGE_A_INPUT_BUNDLE_SCHEMA: Final = "recurquant.experiment013.stage-a-input-bundle.v1"
STAGE_A_INPUT_BUNDLE_FILENAME: Final = "stage-a-input-bundle.json"
STAGE_A_INPUT_BUNDLE_PROFILE: Final = "opaque-byte-copy-no-semantic-decode-v1"
_STAGE_A_INPUT_BUNDLE_AUTHENTICATION_SEAL: Final = object()
_STAGE_A_OFFLINE_ENVIRONMENT: Final = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_STAGE_A_FORBIDDEN_CREDENTIAL_ENVIRONMENT: Final = frozenset(
    {
        "GITHUB_TOKEN",
        "HF_API_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
_STAGE_A_INPUT_BUNDLE_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "phase",
        "capture_version",
        "staging_profile",
        "frozen_identity_file_sha256",
        "calibration_binding_file_sha256",
        "execution_bindings",
        "source_heads",
        "model_hub_manifest_file_sha256",
        "parquet_hub_snapshots",
        "objects",
    }
)
_STAGE_A_INPUT_BUNDLE_OBJECT_FIELDS: Final = frozenset(
    {
        "role",
        "source_id",
        "revision",
        "logical_path",
        "relative_path",
        "sha256",
        "size_bytes",
        "git_blob_oid",
        "lfs_sha256",
    }
)
_STAGE_A_INPUT_BUNDLE_ROLES: Final = frozenset(
    {
        "model_hub_manifest",
        "tokenizer",
        "parquet",
        "ruler_generator",
        "ruler_generation_manifest",
        "ruler_receipt",
    }
)
_WINDOWS_REPARSE_POINT: Final = 0x400


def _bundle_is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as error:
        raise ValueError("cannot authenticate Stage-A bundle path status") from error
    return path.is_symlink() or bool(
        int(getattr(status, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _bundle_safe_relative_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty canonical relative path")
    if any(character in value for character in ("\\", "\0", "\n", "\r", ":")):
        raise ValueError(f"{context} is unsafe")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{context} is not a canonical relative path")
    return value


def _bundle_safe_directory(path: Path, *, create: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError("Stage-A input bundle directory must be absolute")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            if not create:
                raise FileNotFoundError(f"Stage-A input bundle directory is absent: {current}")
            current.mkdir()
        if _bundle_is_link_or_reparse(current) or not current.is_dir():
            raise ValueError("Stage-A input bundle path traverses a link or non-directory")
    return absolute


def _bundle_stable_file_bytes(path: Path, *, context: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    if _bundle_is_link_or_reparse(absolute) or not absolute.is_file():
        raise ValueError(f"{context} must be a regular non-link file")
    before = absolute.stat()
    payload = absolute.read_bytes()
    after = absolute.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ValueError(f"{context} changed while it was read")
    return payload


def _bundle_stable_descendant_bytes(
    root: Path,
    relative_path: str,
    *,
    context: str,
) -> bytes:
    """Read one bundle file while authenticating every lexical path component."""

    safe_relative = _bundle_safe_relative_path(relative_path, context=f"{context} path")
    absolute_root = _bundle_safe_directory(root, create=False)
    components = PurePosixPath(safe_relative).parts
    current = absolute_root
    for index, component in enumerate(components):
        current /= component
        if _bundle_is_link_or_reparse(current):
            raise ValueError(f"{context} path traverses a link or reparse point")
        if index < len(components) - 1:
            if not current.is_dir():
                raise ValueError(f"{context} parent is not a directory")
        elif not current.is_file():
            raise ValueError(f"{context} is not a regular file")
    payload = _bundle_stable_file_bytes(current, context=context)
    # Rewalk after the read so a swapped parent cannot silently survive the
    # point-of-use authentication boundary.
    repeated = absolute_root
    for index, component in enumerate(components):
        repeated /= component
        if _bundle_is_link_or_reparse(repeated):
            raise ValueError(f"{context} path changed to a link or reparse point")
        if index < len(components) - 1:
            if not repeated.is_dir():
                raise ValueError(f"{context} parent changed during authentication")
        elif not repeated.is_file():
            raise ValueError(f"{context} changed during authentication")
    return payload


def _bundle_deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _bundle_deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_bundle_deep_freeze(item) for item in value)
    return value


def _require_stage_a_offline_environment() -> None:
    inherited = {name.upper(): value for name, value in os.environ.items()}
    if any(
        inherited.get(name) != expected for name, expected in _STAGE_A_OFFLINE_ENVIRONMENT.items()
    ):
        raise RuntimeError("offline Stage-A capture requires all frozen offline-mode flags")
    present_credentials = sorted(_STAGE_A_FORBIDDEN_CREDENTIAL_ENVIRONMENT & set(inherited))
    if present_credentials:
        raise RuntimeError(
            "offline Stage-A capture environment contains forbidden credential variables: "
            + ", ".join(present_credentials)
        )


def _bundle_directory_entries(path: Path, *, context: str) -> dict[str, Path]:
    if _bundle_is_link_or_reparse(path) or not path.is_dir():
        raise ValueError(f"{context} must be a non-link directory")
    try:
        entries = tuple(path.iterdir())
    except OSError as error:
        raise ValueError(f"cannot enumerate {context}") from error
    names = [entry.name for entry in entries]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f"{context} contains case-colliding names")
    return {entry.name: entry for entry in entries}


def _bundle_authenticate_filesystem_inventory(root: Path, *, digests: set[str]) -> None:
    """Reject anything outside the exact manifest-plus-content-addressed tree."""

    absolute_root = _bundle_safe_directory(root, create=False)
    root_entries = _bundle_directory_entries(absolute_root, context="Stage-A input bundle root")
    if set(root_entries) != {STAGE_A_INPUT_BUNDLE_FILENAME, "objects"}:
        raise ValueError("Stage-A input bundle root filesystem inventory drifted")
    manifest_path = root_entries[STAGE_A_INPUT_BUNDLE_FILENAME]
    if _bundle_is_link_or_reparse(manifest_path) or not manifest_path.is_file():
        raise ValueError("Stage-A input bundle manifest must be a regular non-link file")
    object_root = root_entries["objects"]
    object_entries = _bundle_directory_entries(
        object_root,
        context="Stage-A input bundle object directory",
    )
    if set(object_entries) != digests:
        raise ValueError("Stage-A input bundle object filesystem inventory drifted")
    for digest, path in object_entries.items():
        _require_sha256(digest, context="Stage-A input bundle object filename")
        if _bundle_is_link_or_reparse(path) or not path.is_file():
            raise ValueError("Stage-A input bundle object must be a regular non-link file")


def _bundle_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _bundle_cache_payload_bytes(cache_dir: Path, returned_path: object, *, context: str) -> bytes:
    if not isinstance(returned_path, (str, os.PathLike)):
        raise ValueError(f"{context} downloader returned no filesystem path")
    cache = _bundle_safe_directory(cache_dir, create=True).resolve(strict=True)
    returned = Path(os.path.abspath(returned_path))
    if not _bundle_path_within(returned, cache):
        raise ValueError(f"{context} downloader returned a path outside the explicit cache")
    try:
        resolved = returned.resolve(strict=True)
        resolved.relative_to(cache)
    except (OSError, ValueError) as error:
        raise ValueError(f"{context} cache pointer escapes the explicit cache") from error
    candidate = cache
    for component in resolved.relative_to(cache).parts:
        candidate /= component
        if _bundle_is_link_or_reparse(candidate):
            raise ValueError(f"{context} resolved cache path traverses a link or reparse point")
    return _bundle_stable_file_bytes(resolved, context=context)


def _bundle_object_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["role"]), str(record["logical_path"])


def _bundle_lfs_pointer_bytes(*, sha256: str, size_bytes: int) -> bytes:
    digest = _require_sha256(sha256, context="Stage-A bundle LFS pointer SHA-256")
    size = _require_int(size_bytes, context="Stage-A bundle LFS pointer size", minimum=1)
    return (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n"
    ).encode("ascii")


def _bundle_expected_object_record(
    *,
    role: str,
    source_id: str,
    revision: str,
    logical_path: str,
    payload: bytes,
    git_blob_oid: str | None = None,
    lfs_sha256: str | None = None,
) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    return {
        "role": role,
        "source_id": source_id,
        "revision": revision,
        "logical_path": logical_path,
        "relative_path": f"objects/{digest}",
        "sha256": digest,
        "size_bytes": len(payload),
        "git_blob_oid": git_blob_oid,
        "lfs_sha256": lfs_sha256,
    }


def _bundle_add_object(
    staging_root: Path,
    records: list[dict[str, Any]],
    *,
    role: str,
    source_id: str,
    revision: str,
    logical_path: str,
    payload: bytes,
    git_blob_oid: str | None = None,
    lfs_sha256: str | None = None,
) -> None:
    if role not in _STAGE_A_INPUT_BUNDLE_ROLES:
        raise ValueError("Stage-A input bundle object role is unsupported")
    _bundle_safe_relative_path(logical_path, context=f"{role} logical path")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{role} object {logical_path!r} must contain bytes")
    digest = sha256_bytes(payload)
    if lfs_sha256 is not None and digest != _require_sha256(
        lfs_sha256, context=f"{role} {logical_path} LFS SHA-256"
    ):
        raise ValueError(f"{role} object {logical_path!r} differs from its LFS identity")
    if git_blob_oid is not None:
        if not isinstance(git_blob_oid, str) or re.fullmatch(r"[0-9a-f]{40}", git_blob_oid) is None:
            raise ValueError(f"{role} object {logical_path!r} has an invalid Git blob OID")
        git_payload = (
            _bundle_lfs_pointer_bytes(sha256=digest, size_bytes=len(payload))
            if lfs_sha256 is not None
            else payload
        )
        if _git_blob_sha1(git_payload) != git_blob_oid:
            raise ValueError(f"{role} object {logical_path!r} differs from its Git blob")
    relative_path = f"objects/{digest}"
    destination = staging_root / PurePosixPath(relative_path)
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if _bundle_stable_file_bytes(destination, context="deduplicated bundle object") != payload:
            raise ValueError("content-addressed Stage-A bundle object collided")
    else:
        destination.write_bytes(payload)
    record = {
        "role": role,
        "source_id": source_id,
        "revision": revision,
        "logical_path": logical_path,
        "relative_path": relative_path,
        "sha256": digest,
        "size_bytes": len(payload),
        "git_blob_oid": git_blob_oid,
        "lfs_sha256": lfs_sha256,
    }
    if _bundle_object_key(record) in {_bundle_object_key(item) for item in records}:
        raise ValueError(f"duplicate Stage-A bundle object identity: {role}/{logical_path}")
    records.append(record)


def _bundle_public_github_revision(
    repo_id: str,
    revision: str,
    *,
    opener: Any,
) -> str:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo_id}/commits/{urllib.parse.quote(revision, safe='')}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "RecurQuant-Experiment-013-stage-a-stager",
        },
    )
    try:
        with opener(request, timeout=30) as response:
            value = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot resolve public GitHub revision for {repo_id}") from error
    resolved = _require_string(value.get("sha"), context=f"{repo_id} public revision")
    if resolved != revision:
        raise ValueError(f"GitHub returned a different object for pinned {repo_id} revision")
    return resolved


def _bundle_public_source_heads(*, api: Any, opener: Any) -> dict[str, str]:
    return _validate_heads(
        {
            "primary_model": str(
                api.model_info(
                    resolver.PRIMARY_MODEL_ID,
                    revision=resolver.PRIMARY_MODEL_REVISION,
                    token=False,
                ).sha
            ),
            "mbpp": str(
                api.dataset_info(
                    resolver.MBPP_DATASET_ID,
                    revision=resolver.MBPP_REVISION,
                    token=False,
                ).sha
            ),
            "pg19": str(
                api.dataset_info(
                    resolver.PG19_DATASET_ID,
                    revision=resolver.PG19_REVISION,
                    token=False,
                ).sha
            ),
            "ruler": _bundle_public_github_revision(
                resolver.RULER_SOURCE_ID,
                resolver.RULER_REVISION,
                opener=opener,
            ),
            "humaneval_plus": str(
                api.dataset_info(
                    resolver.HUMANEVAL_PLUS_DATASET_ID,
                    revision=resolver.HUMANEVAL_PLUS_REVISION,
                    token=False,
                ).sha
            ),
            "evalplus": _bundle_public_github_revision(
                resolver.EVALPLUS_SOURCE_ID,
                resolver.EVALPLUS_SOURCE_REVISION,
                opener=opener,
            ),
        },
        context="opaque Stage-A staging",
    )


def _bundle_tokenizer_files(frozen_identity_bytes: bytes) -> dict[str, dict[str, Any]]:
    document = _strict_json(frozen_identity_bytes, context="frozen Stage-A identity")
    evidence = document.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("frozen Stage-A identity evidence is unavailable")
    tokenizer = evidence.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("frozen Stage-A tokenizer contract is unavailable")
    raw_files = tokenizer.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("frozen Stage-A tokenizer files are unavailable")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping) or set(raw) != {"name", "sha256", "size_bytes"}:
            raise ValueError(f"frozen Stage-A tokenizer files[{index}] is malformed")
        name = _bundle_safe_relative_path(raw["name"], context="tokenizer file name")
        if PurePosixPath(name).name != name or name in result:
            raise ValueError("frozen Stage-A tokenizer file inventory is unsafe or duplicated")
        digest = _require_sha256(raw["sha256"], context=f"tokenizer {name} SHA-256")
        size = _require_int(raw["size_bytes"], context=f"tokenizer {name} size", minimum=1)
        result[name] = {"name": name, "sha256": digest, "size_bytes": size}
    expected = {
        name: {"name": name, "size_bytes": size, "sha256": digest}
        for name, (size, digest) in RULER_EXPECTED_TOKENIZER_ASSETS.items()
    }
    if result != expected:
        raise ValueError("frozen Stage-A tokenizer inventory differs from the exact four-file set")
    return result


def _bundle_expected_parquet_files() -> tuple[tuple[Any, Any], ...]:
    from recurquant import experiment013_parquet

    manifest = experiment013_parquet.load_experiment013_parquet_manifest(
        PARQUET_MATERIALIZATION_MANIFEST_PATH
    )
    selected: list[tuple[Any, Any]] = []
    for dataset_key, logical_split in (("pg19", "validation"), ("humaneval_plus", "test")):
        dataset = manifest.dataset(dataset_key)
        files = tuple(file for file in dataset.files if file.logical_split == logical_split)
        if len(files) != 1:
            raise RuntimeError(
                "Stage-A Parquet inventory must contain exactly two single-file splits"
            )
        selected.append((dataset, files[0]))
    return tuple(selected)


def _bundle_authoritative_ruler_receipt_sha256(
    *,
    verified_binding: object,
    frozen_stage_a_identity: object,
) -> Mapping[str, str]:
    """Derive all 20 opaque receipt hashes from already authenticated identities."""

    calibration_dependencies = getattr(verified_binding, "calibration_dependencies", None)
    if not isinstance(calibration_dependencies, Mapping):
        raise ValueError("Stage-A calibration binding lacks authenticated dependencies")
    calibration_identity_bytes = calibration_dependencies.get("frozen_identity_artifact")
    if not isinstance(calibration_identity_bytes, bytes):
        raise ValueError("Stage-A calibration binding lacks its frozen calibration identity")
    calibration_identity = resolver.deserialize_frozen_calibration_identity_artifact(
        calibration_identity_bytes
    )
    calibration_records = getattr(calibration_identity, "records", None)
    stage_a_records = getattr(frozen_stage_a_identity, "records", None)
    if not isinstance(calibration_records, tuple) or not isinstance(stage_a_records, tuple):
        raise ValueError("authenticated RULER identity records are unavailable")

    required = {str(item["filename"]): item for item in required_ruler_receipts()}
    if len(required) != 20:
        raise RuntimeError("frozen RULER receipt schedule is not exact")
    hashes: dict[str, str] = {}
    for phase, records in (("calibration", calibration_records), ("stage_a", stage_a_records)):
        for raw in records:
            if not isinstance(raw, Mapping) or raw.get("family") != "ruler":
                continue
            category = _require_string(
                raw.get("ruler_category"), context=f"{phase} RULER receipt category"
            )
            config = _require_string(raw.get("config"), context=f"{phase} RULER receipt config")
            configured_length = _require_int(
                raw.get("configured_length"),
                context=f"{phase} RULER configured length",
                minimum=1,
            )
            seed = _require_int(raw.get("seed"), context=f"{phase} RULER receipt seed", minimum=0)
            filename = ruler_receipt_filename(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )
            schedule = required.get(filename)
            if schedule is None or schedule["phase"] != phase or filename in hashes:
                raise ValueError("authenticated RULER receipt identity inventory drifted")
            hashes[filename] = _require_sha256(
                raw.get("generator_receipt_sha256"),
                context=f"authenticated RULER receipt {filename} SHA-256",
            )
    if set(hashes) != set(required):
        raise ValueError("authenticated identities do not commit to all 20 RULER receipts")
    return MappingProxyType(dict(sorted(hashes.items())))


def _bundle_read_authoritative_ruler_receipts(
    *,
    expected_sha256: Mapping[str, str],
    reader: Callable[[str], bytes],
    context: str,
) -> Mapping[str, bytes]:
    """Read each opaque receipt once and authenticate its bytes without decoding it."""

    required_names = tuple(str(item["filename"]) for item in required_ruler_receipts())
    if set(expected_sha256) != set(required_names) or len(required_names) != 20:
        raise ValueError("authoritative RULER receipt hash inventory is incomplete")
    payloads: dict[str, bytes] = {}
    for filename in required_names:
        payload = reader(filename)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"{context} RULER receipt bytes are unavailable: {filename}")
        if sha256_bytes(payload) != expected_sha256[filename]:
            raise ValueError(
                f"{context} RULER receipt differs from authenticated identity: {filename}"
            )
        payloads[filename] = bytes(payload)
    return MappingProxyType(payloads)


@dataclass(frozen=True, slots=True)
class AuthenticatedStageAInputBundle:
    root: Path
    manifest: Mapping[str, Any]
    manifest_file_sha256: str
    objects: Mapping[tuple[str, str], Mapping[str, Any]]
    _authentication_seal: object = dataclass_field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authentication_seal is not _STAGE_A_INPUT_BUNDLE_AUTHENTICATION_SEAL:
            raise ValueError(
                "Stage-A input bundles may be created only by the authenticated loader"
            )

    def object_bytes(self, role: str, logical_path: str) -> bytes:
        try:
            record = self.objects[(role, logical_path)]
        except KeyError as error:
            raise KeyError(f"unknown Stage-A input bundle object: {role}/{logical_path}") from error
        payload = _bundle_stable_descendant_bytes(
            self.root,
            str(record["relative_path"]),
            context=f"Stage-A bundle {role}/{logical_path}",
        )
        if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
            raise ValueError(f"Stage-A bundle object changed: {role}/{logical_path}")
        return payload


def authenticate_stage_a_input_bundle(
    bundle_root: Path,
    *,
    frozen_stage_a_identity_artifact: bytes,
    calibration_binding_artifact: bytes,
    stage_a_capture_provenance_receipt: bytes,
    expected_stage_a_capture_provenance_receipt_sha256: str,
    execution_binding_artifacts: Mapping[str, bytes],
) -> AuthenticatedStageAInputBundle:
    """Authenticate an opaque Stage-A byte bundle without decoding protected rows."""

    if not isinstance(frozen_stage_a_identity_artifact, bytes):
        raise TypeError("frozen Stage-A identity artifact must be bytes")
    if not isinstance(calibration_binding_artifact, bytes):
        raise TypeError("Stage-A calibration binding artifact must be bytes")
    verified_binding = resolver.deserialize_stage_a_calibration_binding_artifact(
        calibration_binding_artifact
    )
    frozen = resolver.deserialize_frozen_stage_a_identity_artifact(
        frozen_stage_a_identity_artifact,
        calibration_binding_artifact=calibration_binding_artifact,
        stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
        expected_stage_a_capture_provenance_receipt_sha256=(
            expected_stage_a_capture_provenance_receipt_sha256
        ),
    )
    authoritative_ruler_receipt_sha256 = _bundle_authoritative_ruler_receipt_sha256(
        verified_binding=verified_binding,
        frozen_stage_a_identity=frozen,
    )
    expected_bindings = _validate_execution_binding_artifacts(execution_binding_artifacts)
    if dict(frozen.execution_bindings) != expected_bindings:
        raise ValueError("Stage-A input bundle execution bindings differ from the frozen identity")
    root = _bundle_safe_directory(bundle_root, create=False)
    raw_manifest = _bundle_stable_descendant_bytes(
        root,
        STAGE_A_INPUT_BUNDLE_FILENAME,
        context="Stage-A input bundle manifest",
    )
    manifest = _strict_json(raw_manifest, context="Stage-A input bundle manifest")
    if canonical_json_bytes(manifest) != raw_manifest:
        raise ValueError("Stage-A input bundle manifest is not canonical JSON")
    _require_exact_fields(
        manifest,
        _STAGE_A_INPUT_BUNDLE_MANIFEST_FIELDS,
        context="Stage-A input bundle manifest",
    )
    if (
        manifest["schema"] != STAGE_A_INPUT_BUNDLE_SCHEMA
        or manifest["phase"] != "stage_a"
        or manifest["capture_version"] != CAPTURE_VERSION
        or manifest["staging_profile"] != STAGE_A_INPUT_BUNDLE_PROFILE
        or manifest["frozen_identity_file_sha256"] != frozen.file_sha256
        or manifest["calibration_binding_file_sha256"] != sha256_bytes(calibration_binding_artifact)
        or manifest["execution_bindings"] != expected_bindings
        or manifest["source_heads"] != EXPECTED_SOURCE_HEADS
        or manifest["model_hub_manifest_file_sha256"]
        != expected_bindings["model_file_manifest_file_sha256"]
    ):
        raise ValueError("Stage-A input bundle identity or staging contract drifted")
    snapshots = manifest["parquet_hub_snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != 2:
        raise ValueError("Stage-A input bundle must bind two Parquet snapshots")
    raw_objects = manifest["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Stage-A input bundle contains no objects")
    objects: dict[tuple[str, str], Mapping[str, Any]] = {}
    normalized_records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Stage-A input bundle objects[{index}] must be an object")
        _require_exact_fields(
            raw,
            _STAGE_A_INPUT_BUNDLE_OBJECT_FIELDS,
            context=f"Stage-A input bundle objects[{index}]",
        )
        role = _require_string(raw["role"], context=f"bundle objects[{index}].role")
        if role not in _STAGE_A_INPUT_BUNDLE_ROLES:
            raise ValueError("Stage-A input bundle contains an unknown object role")
        source_id = _require_string(raw["source_id"], context=f"bundle objects[{index}].source_id")
        revision = _require_string(raw["revision"], context=f"bundle objects[{index}].revision")
        logical_path = _bundle_safe_relative_path(
            raw["logical_path"], context=f"bundle objects[{index}].logical_path"
        )
        relative_path = _bundle_safe_relative_path(
            raw["relative_path"], context=f"bundle objects[{index}].relative_path"
        )
        digest = _require_sha256(raw["sha256"], context=f"bundle objects[{index}].sha256")
        size = _require_int(raw["size_bytes"], context=f"bundle objects[{index}].size", minimum=1)
        if relative_path != f"objects/{digest}":
            raise ValueError("Stage-A input bundle object is not content-addressed")
        git_blob_oid = raw["git_blob_oid"]
        if git_blob_oid is not None and (
            not isinstance(git_blob_oid, str) or re.fullmatch(r"[0-9a-f]{40}", git_blob_oid) is None
        ):
            raise ValueError("Stage-A input bundle object Git blob OID is invalid")
        lfs_sha256 = raw["lfs_sha256"]
        if lfs_sha256 is not None:
            _require_sha256(lfs_sha256, context="Stage-A input bundle object LFS SHA-256")
        record = {
            "role": role,
            "source_id": source_id,
            "revision": revision,
            "logical_path": logical_path,
            "relative_path": relative_path,
            "sha256": digest,
            "size_bytes": size,
            "git_blob_oid": git_blob_oid,
            "lfs_sha256": lfs_sha256,
        }
        key = _bundle_object_key(record)
        if key in objects:
            raise ValueError("Stage-A input bundle object identities are duplicated")
        objects[key] = MappingProxyType(record)
        normalized_records.append(record)
    if normalized_records != sorted(
        normalized_records,
        key=lambda item: (item["role"], item["source_id"], item["logical_path"]),
    ):
        raise ValueError("Stage-A input bundle object inventory is not canonical")
    object_digests = {str(record["sha256"]) for record in normalized_records}
    _bundle_authenticate_filesystem_inventory(root, digests=object_digests)
    bundle = AuthenticatedStageAInputBundle(
        root=root,
        manifest=_bundle_deep_freeze(manifest),
        manifest_file_sha256=sha256_bytes(raw_manifest),
        objects=_bundle_deep_freeze(objects),
        _authentication_seal=_STAGE_A_INPUT_BUNDLE_AUTHENTICATION_SEAL,
    )

    expected_records: dict[tuple[str, str], dict[str, Any]] = {}
    generation_manifest = bundle.object_bytes(
        "ruler_generation_manifest",
        "generation-manifest.json",
    )
    if sha256_bytes(generation_manifest) != (
        verified_binding.ruler_generation_manifest_file_sha256
    ):
        raise ValueError("Stage-A input bundle RULER manifest differs from calibration custody")
    expected_records[("ruler_generation_manifest", "generation-manifest.json")] = (
        _bundle_expected_object_record(
            role="ruler_generation_manifest",
            source_id=resolver.RULER_SOURCE_ID,
            revision=resolver.RULER_REVISION,
            logical_path="generation-manifest.json",
            payload=generation_manifest,
        )
    )
    model_bytes = bundle.object_bytes("model_hub_manifest", "model-file-manifest.json")
    if model_bytes != execution_binding_artifacts["model_file_manifest_file_sha256"]:
        raise ValueError("Stage-A bundle model Hub attestation differs from the frozen manifest")
    expected_records[("model_hub_manifest", "model-file-manifest.json")] = (
        _bundle_expected_object_record(
            role="model_hub_manifest",
            source_id=resolver.PRIMARY_MODEL_ID,
            revision=resolver.PRIMARY_MODEL_REVISION,
            logical_path="model-file-manifest.json",
            payload=model_bytes,
        )
    )
    tokenizer_files = _bundle_tokenizer_files(frozen_stage_a_identity_artifact)
    for name, expected in tokenizer_files.items():
        payload = bundle.object_bytes("tokenizer", name)
        if len(payload) != expected["size_bytes"] or sha256_bytes(payload) != expected["sha256"]:
            raise ValueError(f"Stage-A bundle tokenizer object drifted: {name}")
        expected_records[("tokenizer", name)] = _bundle_expected_object_record(
            role="tokenizer",
            source_id=resolver.PRIMARY_MODEL_ID,
            revision=resolver.PRIMARY_MODEL_REVISION,
            logical_path=name,
            payload=payload,
        )
    for path, git_blob_oid in RULER_GENERATOR_GIT_BLOBS.items():
        payload = bundle.object_bytes("ruler_generator", path)
        if _git_blob_sha1(payload) != git_blob_oid:
            raise ValueError(f"Stage-A bundle RULER generator object drifted: {path}")
        expected_records[("ruler_generator", path)] = _bundle_expected_object_record(
            role="ruler_generator",
            source_id=resolver.RULER_SOURCE_ID,
            revision=resolver.RULER_REVISION,
            logical_path=path,
            payload=payload,
            git_blob_oid=git_blob_oid,
        )
    ruler_receipt_payloads = _bundle_read_authoritative_ruler_receipts(
        expected_sha256=authoritative_ruler_receipt_sha256,
        reader=lambda filename: bundle.object_bytes("ruler_receipt", filename),
        context="Stage-A input bundle",
    )
    for item in required_ruler_receipts():
        filename = str(item["filename"])
        payload = ruler_receipt_payloads[filename]
        expected_records[("ruler_receipt", filename)] = _bundle_expected_object_record(
            role="ruler_receipt",
            source_id=resolver.RULER_SOURCE_ID,
            revision=resolver.RULER_REVISION,
            logical_path=filename,
            payload=payload,
        )
    expected_snapshots: list[dict[str, Any]] = []
    for dataset, file in _bundle_expected_parquet_files():
        logical = f"{dataset.key}/{file.logical_split}/{file.immutable_path}"
        payload = bundle.object_bytes("parquet", logical)
        if (
            len(payload) != file.size_bytes
            or sha256_bytes(payload) != file.lfs_sha256
            or _git_blob_sha1(
                _bundle_lfs_pointer_bytes(
                    sha256=file.lfs_sha256,
                    size_bytes=file.size_bytes,
                )
            )
            != file.git_blob_oid
        ):
            raise ValueError(f"Stage-A bundle Parquet object drifted: {logical}")
        expected_records[("parquet", logical)] = _bundle_expected_object_record(
            role="parquet",
            source_id=dataset.dataset_id,
            revision=dataset.conversion_revision,
            logical_path=logical,
            payload=payload,
            git_blob_oid=file.git_blob_oid,
            lfs_sha256=file.lfs_sha256,
        )
        expected_snapshots.append(
            {
                "dataset_key": dataset.key,
                "dataset_id": dataset.dataset_id,
                "source_revision": dataset.source_revision,
                "conversion_revision": dataset.conversion_revision,
                "files": [
                    {
                        "path": file.immutable_path,
                        "git_blob_oid": file.git_blob_oid,
                        "lfs_sha256": file.lfs_sha256,
                        "size_bytes": file.size_bytes,
                    }
                ],
            }
        )
    if snapshots != expected_snapshots:
        raise ValueError("Stage-A input bundle Parquet Hub snapshots drifted")
    if {key: dict(record) for key, record in objects.items()} != expected_records:
        raise ValueError("Stage-A input bundle object records differ from the frozen semantics")
    return bundle


def stage_stage_a_input_bundle(
    *,
    bundle_root: Path,
    cache_dir: Path,
    ruler_receipt_dir: Path,
    frozen_stage_a_identity_artifact: bytes,
    calibration_binding_artifact: bytes,
    stage_a_capture_provenance_receipt: bytes,
    expected_stage_a_capture_provenance_receipt_sha256: str,
    execution_binding_artifacts: Mapping[str, bytes],
    runtime_authentication_context: Mapping[str, object],
) -> AuthenticatedStageAInputBundle:
    """Stage exact public bytes without decoding any protected Stage-A content."""

    destination = Path(os.path.abspath(bundle_root))
    repository = REPOSITORY_ROOT.resolve(strict=True)
    if _bundle_path_within(destination, repository):
        raise ValueError("Stage-A input bundle must be outside the repository")
    parent = _bundle_safe_directory(destination.parent, create=True)
    cache = _bundle_safe_directory(cache_dir, create=True)
    if _bundle_path_within(destination, cache) or _bundle_path_within(cache, destination):
        raise ValueError("Stage-A input bundle and shared Hub cache must not be nested")
    ruler_root = _bundle_safe_directory(ruler_receipt_dir, create=False)
    verified_binding = resolver.deserialize_stage_a_calibration_binding_artifact(
        calibration_binding_artifact
    )
    frozen = resolver.deserialize_frozen_stage_a_identity_artifact(
        frozen_stage_a_identity_artifact,
        calibration_binding_artifact=calibration_binding_artifact,
        stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
        expected_stage_a_capture_provenance_receipt_sha256=(
            expected_stage_a_capture_provenance_receipt_sha256
        ),
    )
    authoritative_ruler_receipt_sha256 = _bundle_authoritative_ruler_receipt_sha256(
        verified_binding=verified_binding,
        frozen_stage_a_identity=frozen,
    )
    expected_execution_bindings = _validate_execution_binding_artifacts(execution_binding_artifacts)
    if dict(frozen.execution_bindings) != expected_execution_bindings:
        raise ValueError("opaque stager inputs differ from the frozen Stage-A identity")
    ruler_inventory = _verify_live_ruler_receipt_inventory(ruler_root)
    generation_manifest = _bundle_stable_file_bytes(
        ruler_inventory["generation-manifest.json"],
        context="RULER generation manifest for opaque staging",
    )
    if sha256_bytes(generation_manifest) != (
        verified_binding.ruler_generation_manifest_file_sha256
    ):
        raise ValueError("opaque stager RULER manifest differs from calibration custody")

    def read_live_ruler_receipt(filename: str) -> bytes:
        return _bundle_stable_file_bytes(
            ruler_inventory[filename],
            context=f"RULER receipt {filename} for opaque staging",
        )

    ruler_receipt_payloads = _bundle_read_authoritative_ruler_receipts(
        expected_sha256=authoritative_ruler_receipt_sha256,
        reader=read_live_ruler_receipt,
        context="opaque stager",
    )
    tokenizer_files = _bundle_tokenizer_files(frozen_stage_a_identity_artifact)
    runtime_context = _normalize_runtime_authentication_context(runtime_authentication_context)
    authentication = _authenticate_execution_binding_artifacts(
        execution_binding_artifacts,
        runtime_context=runtime_context,
    )
    try:
        if os.path.lexists(destination):
            return authenticate_stage_a_input_bundle(
                destination,
                frozen_stage_a_identity_artifact=frozen_stage_a_identity_artifact,
                calibration_binding_artifact=calibration_binding_artifact,
                stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
                expected_stage_a_capture_provenance_receipt_sha256=(
                    expected_stage_a_capture_provenance_receipt_sha256
                ),
                execution_binding_artifacts=execution_binding_artifacts,
            )
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("opaque Stage-A staging requires huggingface-hub") from error
        api = HfApi(endpoint="https://huggingface.co", token=False)
        opener = urllib.request.urlopen
        before = _bundle_public_source_heads(api=api, opener=opener)
        staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
        records: list[dict[str, Any]] = []
        try:
            _bundle_add_object(
                staging_root,
                records,
                role="model_hub_manifest",
                source_id=resolver.PRIMARY_MODEL_ID,
                revision=resolver.PRIMARY_MODEL_REVISION,
                logical_path="model-file-manifest.json",
                payload=execution_binding_artifacts["model_file_manifest_file_sha256"],
            )
            for name, expected in sorted(tokenizer_files.items()):
                returned = hf_hub_download(
                    repo_id=resolver.PRIMARY_MODEL_ID,
                    filename=name,
                    repo_type="model",
                    revision=resolver.PRIMARY_MODEL_REVISION,
                    cache_dir=cache,
                    token=False,
                    endpoint="https://huggingface.co",
                )
                payload = _bundle_cache_payload_bytes(cache, returned, context=f"tokenizer {name}")
                if (
                    len(payload) != expected["size_bytes"]
                    or sha256_bytes(payload) != expected["sha256"]
                ):
                    raise ValueError(f"downloaded tokenizer object drifted: {name}")
                _bundle_add_object(
                    staging_root,
                    records,
                    role="tokenizer",
                    source_id=resolver.PRIMARY_MODEL_ID,
                    revision=resolver.PRIMARY_MODEL_REVISION,
                    logical_path=name,
                    payload=payload,
                )

            parquet_snapshots: list[dict[str, Any]] = []
            from recurquant import experiment013_parquet

            metadata = experiment013_parquet.HuggingFaceHubMetadataBackend(token=False)
            for dataset, file in _bundle_expected_parquet_files():
                if (
                    metadata.resolve_dataset_revision(
                        repo_id=dataset.dataset_id,
                        revision=dataset.source_revision,
                    )
                    != dataset.source_revision
                ):
                    raise ValueError("Stage-A Parquet source revision drifted during staging")
                snapshot = metadata.snapshot_parquet_files(
                    repo_id=dataset.dataset_id,
                    revision=dataset.conversion_revision,
                    paths=(file.immutable_path,),
                )
                observed = snapshot.files[0] if len(snapshot.files) == 1 else None
                if (
                    snapshot.commit_hash != dataset.conversion_revision
                    or observed is None
                    or observed.path != file.immutable_path
                    or observed.git_blob_oid != file.git_blob_oid
                    or observed.lfs_sha256 != file.lfs_sha256
                    or observed.size_bytes != file.size_bytes
                    or observed.lfs_size_bytes != file.lfs_size_bytes
                    or observed.etag != file.lfs_sha256
                ):
                    raise ValueError("Stage-A Parquet Hub metadata drifted during staging")
                returned = hf_hub_download(
                    repo_id=dataset.dataset_id,
                    filename=file.immutable_path,
                    repo_type="dataset",
                    revision=dataset.conversion_revision,
                    cache_dir=cache,
                    token=False,
                    endpoint="https://huggingface.co",
                )
                payload = _bundle_cache_payload_bytes(
                    cache,
                    returned,
                    context=f"Parquet {dataset.key}/{file.logical_split}",
                )
                if len(payload) != file.size_bytes or sha256_bytes(payload) != file.lfs_sha256:
                    raise ValueError("downloaded Stage-A Parquet object differs from frozen LFS")
                logical = f"{dataset.key}/{file.logical_split}/{file.immutable_path}"
                _bundle_add_object(
                    staging_root,
                    records,
                    role="parquet",
                    source_id=dataset.dataset_id,
                    revision=dataset.conversion_revision,
                    logical_path=logical,
                    payload=payload,
                    git_blob_oid=file.git_blob_oid,
                    lfs_sha256=file.lfs_sha256,
                )
                parquet_snapshots.append(
                    {
                        "dataset_key": dataset.key,
                        "dataset_id": dataset.dataset_id,
                        "source_revision": dataset.source_revision,
                        "conversion_revision": dataset.conversion_revision,
                        "files": [
                            {
                                "path": file.immutable_path,
                                "git_blob_oid": file.git_blob_oid,
                                "lfs_sha256": file.lfs_sha256,
                                "size_bytes": file.size_bytes,
                            }
                        ],
                    }
                )

            raw_base = (
                "https://raw.githubusercontent.com/"
                f"{resolver.RULER_SOURCE_ID}/{resolver.RULER_REVISION}/"
            )
            for path, git_blob_oid in sorted(RULER_GENERATOR_GIT_BLOBS.items()):
                request = urllib.request.Request(
                    raw_base + path,
                    headers={"User-Agent": "RecurQuant-Experiment-013-stage-a-stager"},
                )
                try:
                    with opener(request, timeout=30) as response:
                        payload = response.read()
                except (OSError, urllib.error.HTTPError) as error:
                    raise RuntimeError(f"cannot stage pinned RULER source file {path}") from error
                _bundle_add_object(
                    staging_root,
                    records,
                    role="ruler_generator",
                    source_id=resolver.RULER_SOURCE_ID,
                    revision=resolver.RULER_REVISION,
                    logical_path=path,
                    payload=payload,
                    git_blob_oid=git_blob_oid,
                )
            _bundle_add_object(
                staging_root,
                records,
                role="ruler_generation_manifest",
                source_id=resolver.RULER_SOURCE_ID,
                revision=resolver.RULER_REVISION,
                logical_path="generation-manifest.json",
                payload=generation_manifest,
            )
            for item in required_ruler_receipts():
                filename = str(item["filename"])
                payload = ruler_receipt_payloads[filename]
                _bundle_add_object(
                    staging_root,
                    records,
                    role="ruler_receipt",
                    source_id=resolver.RULER_SOURCE_ID,
                    revision=resolver.RULER_REVISION,
                    logical_path=filename,
                    payload=payload,
                )
            after = _bundle_public_source_heads(api=api, opener=opener)
            if after != before:
                raise ValueError("public source heads changed during opaque Stage-A staging")
            records.sort(key=lambda item: (item["role"], item["source_id"], item["logical_path"]))
            manifest = {
                "schema": STAGE_A_INPUT_BUNDLE_SCHEMA,
                "phase": "stage_a",
                "capture_version": CAPTURE_VERSION,
                "staging_profile": STAGE_A_INPUT_BUNDLE_PROFILE,
                "frozen_identity_file_sha256": frozen.file_sha256,
                "calibration_binding_file_sha256": sha256_bytes(calibration_binding_artifact),
                "execution_bindings": dict(frozen.execution_bindings),
                "source_heads": before,
                "model_hub_manifest_file_sha256": sha256_bytes(
                    execution_binding_artifacts["model_file_manifest_file_sha256"]
                ),
                "parquet_hub_snapshots": parquet_snapshots,
                "objects": records,
            }
            (staging_root / STAGE_A_INPUT_BUNDLE_FILENAME).write_bytes(
                canonical_json_bytes(manifest)
            )
            authenticate_stage_a_input_bundle(
                staging_root,
                frozen_stage_a_identity_artifact=frozen_stage_a_identity_artifact,
                calibration_binding_artifact=calibration_binding_artifact,
                stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
                expected_stage_a_capture_provenance_receipt_sha256=(
                    expected_stage_a_capture_provenance_receipt_sha256
                ),
                execution_binding_artifacts=execution_binding_artifacts,
            )
            lock_path = parent / f".{destination.name}.publish.lock"
            lock_payload = canonical_json_bytes(
                {
                    "bundle_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
                    "owner_nonce": os.urandom(32).hex(),
                    "staging_directory": staging_root.name,
                }
            )
            lock_owned = False
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                lock_owned = True
            except FileExistsError as error:
                raise FileExistsError(
                    "Stage-A input bundle publication is already owned"
                ) from error
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(lock_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.rename(staging_root, destination)
            finally:
                if lock_owned and os.path.lexists(lock_path):
                    observed_lock = _bundle_stable_file_bytes(
                        lock_path,
                        context="Stage-A input bundle publication lock",
                    )
                    if observed_lock != lock_payload:
                        raise ValueError("Stage-A input bundle publication lock ownership changed")
                    lock_path.unlink()
                    if os.path.lexists(lock_path):
                        raise ValueError(
                            "Stage-A input bundle publication lock survived owned cleanup"
                        )
        except BaseException:
            # The uniquely-owned temporary tree is intentionally retained for
            # forensic inspection. It is never treated as an authenticated bundle.
            raise
        return authenticate_stage_a_input_bundle(
            destination,
            frozen_stage_a_identity_artifact=frozen_stage_a_identity_artifact,
            calibration_binding_artifact=calibration_binding_artifact,
            stage_a_capture_provenance_receipt=stage_a_capture_provenance_receipt,
            expected_stage_a_capture_provenance_receipt_sha256=(
                expected_stage_a_capture_provenance_receipt_sha256
            ),
            execution_binding_artifacts=execution_binding_artifacts,
        )
    finally:
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is authentication.runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)


class _StagedParquetHubBackend:
    def __init__(self, bundle: AuthenticatedStageAInputBundle) -> None:
        from recurquant import experiment013_parquet

        self._module = experiment013_parquet
        self._snapshots = {
            str(item["dataset_id"]): item for item in bundle.manifest["parquet_hub_snapshots"]
        }

    def resolve_dataset_revision(self, *, repo_id: str, revision: str) -> str:
        snapshot = self._snapshots.get(repo_id)
        if snapshot is None or snapshot["source_revision"] != revision:
            raise ValueError("offline Parquet source revision is absent or different")
        return revision

    def snapshot_parquet_files(
        self,
        *,
        repo_id: str,
        revision: str,
        paths: tuple[str, ...],
    ) -> Any:
        snapshot = self._snapshots.get(repo_id)
        if snapshot is None or snapshot["conversion_revision"] != revision:
            raise ValueError("offline Parquet conversion revision is absent or different")
        files_by_path = {str(item["path"]): item for item in snapshot["files"]}
        files = []
        for path in paths:
            item = files_by_path.get(path)
            if item is None:
                raise ValueError("offline Parquet snapshot omitted a requested path")
            files.append(
                self._module.HubFileMetadata(
                    path=path,
                    commit_hash=revision,
                    size_bytes=int(item["size_bytes"]),
                    git_blob_oid=str(item["git_blob_oid"]),
                    lfs_sha256=str(item["lfs_sha256"]),
                    lfs_size_bytes=int(item["size_bytes"]),
                    etag=str(item["lfs_sha256"]),
                )
            )
        return self._module.HubDatasetMetadata(commit_hash=revision, files=tuple(files))


class _StagedParquetBackend:
    def __init__(self, bundle: AuthenticatedStageAInputBundle) -> None:
        self._bundle = bundle
        self._uris: dict[str, str] = {}
        for snapshot in bundle.manifest["parquet_hub_snapshots"]:
            dataset_key = str(snapshot["dataset_key"])
            dataset_id = str(snapshot["dataset_id"])
            revision = str(snapshot["conversion_revision"])
            for file in snapshot["files"]:
                path = str(file["path"])
                split = "validation" if dataset_key == "pg19" else "test"
                logical = f"{dataset_key}/{split}/{path}"
                self._uris[f"hf://datasets/{dataset_id}@{revision}/{path}"] = logical

    def _parquet_file(self, uri: str) -> Any:
        from io import BytesIO

        import pyarrow.parquet as parquet

        try:
            logical_path = self._uris[uri]
        except KeyError as error:
            raise ValueError("offline Parquet URI is outside the staged inventory") from error
        payload = self._bundle.object_bytes("parquet", logical_path)
        return parquet.ParquetFile(BytesIO(payload))

    def inspect(self, uri: str) -> Any:
        from recurquant import experiment013_parquet

        parquet_file = self._parquet_file(uri)
        metadata = parquet_file.metadata
        return experiment013_parquet.ParquetFileLayout(
            row_group_rows=tuple(
                metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
            ),
            columns=tuple(parquet_file.schema_arrow.names),
        )

    def read_row(
        self,
        uri: str,
        *,
        row_group_index: int,
        row_index_in_group: int,
        columns: tuple[str, ...],
    ) -> Mapping[str, object]:
        parquet_file = self._parquet_file(uri)
        table = parquet_file.read_row_group(row_group_index, columns=list(columns))
        rows = table.slice(row_index_in_group, 1).to_pylist()
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError("offline Parquet backend did not return exactly one row")
        return rows[0]

    def read_row_group_projection(
        self,
        uri: str,
        *,
        row_group_index: int,
        columns: tuple[str, ...],
    ) -> Sequence[Mapping[str, object]]:
        return (
            self._parquet_file(uri)
            .read_row_group(
                row_group_index,
                columns=list(columns),
            )
            .to_pylist()
        )


class StagedCaptureSource:
    """Strictly local CaptureSource backed by one authenticated opaque bundle."""

    def __init__(self, bundle: AuthenticatedStageAInputBundle) -> None:
        if not isinstance(bundle, AuthenticatedStageAInputBundle):
            raise TypeError("StagedCaptureSource requires an authenticated Stage-A input bundle")
        _require_stage_a_offline_environment()
        self.bundle = bundle
        self._hub = _StagedParquetHubBackend(bundle)
        self._parquet = _StagedParquetBackend(bundle)

    def model_file_manifest_attestation(self) -> bytes:
        return self.bundle.object_bytes("model_hub_manifest", "model-file-manifest.json")

    def source_heads(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.bundle.manifest["source_heads"]))

    def tokenizer_material(self) -> TokenizerMaterial:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("offline Stage-A capture requires Transformers") from error
        records = {
            logical_path: record
            for (role, logical_path), record in self.bundle.objects.items()
            if role == "tokenizer"
        }
        files = {name: self.bundle.object_bytes("tokenizer", name) for name in sorted(records)}
        with tempfile.TemporaryDirectory(prefix="recurquant-exp013-staged-tokenizer-") as temporary:
            isolated = Path(temporary)
            for name, payload in files.items():
                (isolated / name).write_bytes(payload)
            tokenizer = AutoTokenizer.from_pretrained(
                isolated,
                local_files_only=True,
                trust_remote_code=False,
                token=False,
            )
            inventory = {
                path.relative_to(isolated).as_posix()
                for path in isolated.rglob("*")
                if path.is_file()
            }
            if inventory != set(files):
                raise ValueError("offline tokenizer construction changed the staged inventory")
        return TokenizerMaterial(
            tokenizer=tokenizer,
            tokenizer_class=tokenizer.__class__.__name__,
            transformers_version=importlib.metadata.version("transformers"),
            files=MappingProxyType(files),
            model_weights_loaded=False,
        )

    def mbpp_train_rows(self) -> Sequence[Mapping[str, Any]]:
        raise RuntimeError("Stage-A offline source forbids MBPP payload access")

    def pg19_projection(self, split: str) -> Sequence[ProjectionRow]:
        if split != "validation":
            raise ValueError("Stage-A offline source permits only PG19 validation")
        from recurquant import experiment013_parquet

        projection = experiment013_parquet.project_experiment013_parquet_columns(
            "pg19",
            split,
            columns=("url",),
            expected_count=50,
            hub_backend=self._hub,
            parquet_backend=self._parquet,
        )
        return tuple(
            ProjectionRow(
                _require_string(row.values[0], context="PG19 validation url"),
                row.global_offset,
            )
            for row in projection.rows
        )

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> Mapping[str, Any]:
        if split != "validation":
            raise ValueError("Stage-A offline source permits only PG19 validation")
        from recurquant import experiment013_parquet

        selected = experiment013_parquet.read_experiment013_parquet_row(
            "pg19",
            split,
            offset,
            columns=("url", "text"),
            hub_backend=self._hub,
            parquet_backend=self._parquet,
        )
        row = dict(selected.values)
        if row.get("url") != expected_url:
            raise ValueError("offline PG19 row URL differs from its projection")
        return row

    def ruler_generator_files(self) -> Mapping[str, bytes]:
        return MappingProxyType(
            {
                path: self.bundle.object_bytes("ruler_generator", path)
                for path in sorted(RULER_GENERATOR_GIT_BLOBS)
            }
        )

    def ruler_generation_manifest_bytes(self) -> bytes:
        return self.bundle.object_bytes("ruler_generation_manifest", "generation-manifest.json")

    def ruler_receipt_bytes(
        self,
        *,
        category: str,
        config: str,
        configured_length: int,
        seed: int,
    ) -> bytes:
        filename = ruler_receipt_filename(
            category=category,
            config=config,
            configured_length=configured_length,
            seed=seed,
        )
        return self.bundle.object_bytes("ruler_receipt", filename)

    def humaneval_projection(self) -> Sequence[ProjectionRow]:
        from recurquant import experiment013_parquet

        projection = experiment013_parquet.project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("task_id",),
            expected_count=164,
            hub_backend=self._hub,
            parquet_backend=self._parquet,
        )
        return tuple(
            ProjectionRow(
                _require_string(row.values[0], context="HumanEval+ task_id"),
                row.global_offset,
            )
            for row in projection.rows
        )

    def humaneval_row(self, *, offset: int, expected_task_id: str) -> Mapping[str, Any]:
        from recurquant import experiment013_parquet

        selected = experiment013_parquet.read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            offset,
            columns=("task_id", "prompt", "canonical_solution"),
            hub_backend=self._hub,
            parquet_backend=self._parquet,
        )
        row = dict(selected.values)
        if row.get("task_id") != expected_task_id:
            raise ValueError("offline HumanEval+ row differs from its projection")
        return row


def _parse_named_cli_values(
    values: Sequence[str],
    *,
    context: str,
    paths: bool,
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for raw in values:
        if not isinstance(raw, str) or "=" not in raw:
            raise ValueError(f"{context} must use NAME=VALUE")
        name, rendered = raw.split("=", 1)
        if _RUNTIME_ROOT_NAME_RE.fullmatch(name) is None or not rendered:
            raise ValueError(f"{context} contains a non-canonical name or empty value")
        if name in parsed:
            raise ValueError(f"{context} contains a duplicate name")
        parsed[name] = Path(rendered) if paths else rendered
    if not parsed:
        raise ValueError(f"at least one {context} is required")
    return parsed


def _runtime_context_from_cli(args: argparse.Namespace) -> _RuntimeAuthenticationContext:
    if (
        args.base_runtime_root is None
        or args.git_executable is None
        or args.staged_interpreter is None
    ):
        raise ValueError(
            "capture requires --base-runtime-root, --git-executable, and --staged-interpreter"
        )
    roots = _parse_named_cli_values(
        args.package_root,
        context="--package-root",
        paths=True,
    )
    import_paths = _parse_named_cli_values(
        args.package_import_path,
        context="--package-import-path",
        paths=False,
    )
    return _normalize_runtime_authentication_context(
        {
            "base_runtime_root": args.base_runtime_root,
            "git_executable": args.git_executable,
            "staged_interpreter": args.staged_interpreter,
            "package_runtime_roots": roots,
            "package_import_paths": import_paths,
        }
    )


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
    parser.add_argument("--repository-source-manifest", type=Path)
    parser.add_argument("--calibration-runtime-manifest", type=Path)
    parser.add_argument("--model-file-manifest", type=Path)
    parser.add_argument("--parquet-materialization-manifest", type=Path)
    parser.add_argument("--base-runtime-root", type=Path)
    parser.add_argument("--git-executable", type=Path)
    parser.add_argument("--staged-interpreter", type=Path)
    parser.add_argument("--package-root", action="append", default=[])
    parser.add_argument("--package-import-path", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase in resolver.PROTECTED_STAGES:
        raise PermissionError(
            f"{args.phase} is protected; capture v{CAPTURE_VERSION} refuses it before file or "
            "source access"
        )
    if args.phase == "stage_a":
        raise PermissionError(
            "Stage-A capture is sealed-launcher-only; direct CLI refuses it before "
            "binding, path, source, or provider access"
        )
    if args.ruler_receipt_dir is None:
        raise ValueError("--ruler-receipt-dir is required")
    if args.dry_run and args.output is not None:
        raise ValueError("--dry-run forbids --output")
    if not args.dry_run and args.output is None:
        raise ValueError("capture requires --output or --dry-run")
    runtime_context = _runtime_context_from_cli(args)
    binding: bytes | None = None
    if args.phase == "stage_a":
        if args.calibration_binding is None:
            raise ValueError("Stage A requires --calibration-binding")
        binding = args.calibration_binding.read_bytes()
    elif args.calibration_binding is not None:
        raise ValueError("--calibration-binding is valid only for Stage A")
    binding_paths = {
        "repository_source_manifest_file_sha256": args.repository_source_manifest,
        "calibration_runtime_manifest_file_sha256": args.calibration_runtime_manifest,
        "model_file_manifest_file_sha256": args.model_file_manifest,
        "parquet_materialization_manifest_file_sha256": (args.parquet_materialization_manifest),
    }
    if any(path is None for path in binding_paths.values()):
        raise ValueError(
            "capture requires --repository-source-manifest, "
            "--calibration-runtime-manifest, --model-file-manifest, and "
            "--parquet-materialization-manifest"
        )
    execution_binding_artifacts = {
        field: path.read_bytes() for field, path in binding_paths.items() if path is not None
    }
    source = LiveCaptureSource(
        cache_dir=args.cache_dir,
        ruler_receipt_dir=args.ruler_receipt_dir,
    )
    captured = capture_identity_input(
        phase=args.phase,
        source=source,
        calibration_binding=binding,
        execution_binding_artifacts=execution_binding_artifacts,
        runtime_authentication_context={
            "base_runtime_root": runtime_context.base_runtime_root,
            "git_executable": runtime_context.git_executable,
            "staged_interpreter": runtime_context.staged_interpreter,
            "package_runtime_roots": runtime_context.package_runtime_roots,
            "package_import_paths": runtime_context.package_import_paths,
        },
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
