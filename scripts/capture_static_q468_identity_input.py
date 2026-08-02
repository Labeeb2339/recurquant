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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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

CAPTURE_VERSION: Final = 4
RUNTIME_AUTHENTICATION_CONTEXT_FIELDS: Final = frozenset(
    {
        "base_runtime_root",
        "staged_interpreter",
        "package_runtime_roots",
        "package_import_paths",
    }
)
_RUNTIME_ROOT_NAME_RE: Final = re.compile(r"[a-z][a-z0-9-]{0,63}")
RULER_RECEIPT_SCHEMA: Final = "recurquant.experiment013.ruler-receipt.v1"
RULER_GENERATION_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-generation-manifest.v2"
RULER_RUNTIME_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-runtime-manifest.v3"
RULER_LAUNCHER_REVISION: Final = "experiment-013-ruler-argv-launcher-v6"
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
        "de342cdbdfd2876e9a63d1201f00ebf1ab539ba030311dd4d6b8c9e7539cc38b"
    ),
    "aggregation__cwe__l4096__s12340.json": (
        "dba911e2ccd64251688a34c23974c14b38f9b6f56cee1dabff92b772e4da9684"
    ),
    "aggregation__fwe__l2048__s12339.json": (
        "60396cfcf9a5528fd6c2f9be4bf8727adde8f3794fae478962c6535abb950eb3"
    ),
    "aggregation__fwe__l4096__s12339.json": (
        "b33f0f12630b8b94db46fd2c7ba9cfcef3097d6161e28d6fec6d332548e3d9af"
    ),
    "aggregation__fwe__l4096__s2339.json": (
        "747d0a4b3af13a91a0aed560023bd5da11546e06eb6471ee9d964612f3cf64cc"
    ),
    "multi_hop_tracing__vt__l2048__s12339.json": (
        "edd4e2e387bd48bd7fe26211d9bd6fa56e23d187b33dba2ceb522d211b853888"
    ),
    "multi_hop_tracing__vt__l2048__s12340.json": (
        "ab2cc8b7b60fa459ea9b592ad890e37b39991bbe61163426daa39152970f1c92"
    ),
    "multi_hop_tracing__vt__l4096__s12339.json": (
        "15f9b2c1e1d79ebb466e771febddc0640ee4e3b21894591df70f0f2b7b750bec"
    ),
    "multi_hop_tracing__vt__l4096__s12340.json": (
        "612759085fdf165a665281e43bcec060d86e6e0b900e8f3ab25bc9d4d3aa5eac"
    ),
    "multi_hop_tracing__vt__l4096__s2339.json": (
        "5a5ebdf3ab054a825bc9e18176a8513f28964a555536e5a2138dd71a4a53a042"
    ),
    "question_answering__qa_1__l2048__s12339.json": (
        "574119d0a8fe4d8ec4518c2665a5ecbc9dc0f0d97c040c7e07cfc7c7a3b6e59e"
    ),
    "question_answering__qa_1__l4096__s12339.json": (
        "c9da9b5c0a2b4a8e691a4df84ca1f23bcf8d4427bc6b29e940c065715e31b3a2"
    ),
    "question_answering__qa_1__l4096__s2339.json": (
        "fd5935bc5ba0da67015995fb08df1288d33b81cc0b4f29b0fbb5b40f8e43a690"
    ),
    "question_answering__qa_2__l2048__s12340.json": (
        "16f9804bdd9705a068dafa3746936501f8d01120930f180a81511e02eba50e3b"
    ),
    "question_answering__qa_2__l4096__s12340.json": (
        "abe15545a61b50fc06ad77d9262cffdbfe8480f84ff468e30cf038678736a591"
    ),
    "retrieval__niah_multikey_2__l2048__s12340.json": (
        "f12039ecaf89bbf7fe7180255ad36aa304148d84f2fdae551648e04c5d25c428"
    ),
    "retrieval__niah_multiquery__l2048__s12339.json": (
        "099c511108e1b8d4431d03b530b772298e510b6dbefb6d9ff2f51b72b059641d"
    ),
    "retrieval__niah_multiquery__l4096__s2339.json": (
        "f36ead6ac85202ae799598dc2d9086442686aed2cc918951cc46166ad3780918"
    ),
    "retrieval__niah_multivalue__l4096__s12340.json": (
        "03cd0aa9b7e2d16d23ab157a6138cfbf344fd884863cb32723f31131de8ecde2"
    ),
    "retrieval__niah_single_1__l4096__s12339.json": (
        "4edd45298317225a7f6cd3ae22c93f05b33d47902b62d2dea7425bd7d86d19c2"
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
    if (
        not answer_prefix.startswith(invariant.answer_prefix_marker)
        or not answer_prefix.endswith(invariant.answer_prefix_suffix)
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
    row_length = _require_int(
        row["length"], context="RULER raw validation length", minimum=1
    )
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
        expected_position = len(
            _encode(tokenizer, input_text[:index], add_special_tokens=False)
        )
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
    tokenizer_material: TokenizerMaterial,
) -> VerifiedRulerBundle:
    raw_manifest = source.ruler_generation_manifest_bytes()
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
            ("generator_reported_length", 1),
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
        receipt_bytes = source.ruler_receipt_bytes(
            category=str(expected["category"]),
            config=str(expected["config"]),
            configured_length=int(expected["configured_length"]),
            seed=int(expected["seed"]),
        )
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
        if (
            result["sha256"] != sha256_bytes(receipt_bytes)
            or result["size_bytes"] != len(receipt_bytes)
            or result["generator_reported_length"] != receipt["generator_reported_length"]
        ):
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
        generation_manifest_sha256=sha256_bytes(raw_manifest),
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
    staged_interpreter = value["staged_interpreter"]
    if not isinstance(base_runtime_root, Path) or not isinstance(staged_interpreter, Path):
        raise ValueError("runtime authentication roots and interpreter must be Path values")
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
        staged_interpreter=staged_interpreter,
        package_runtime_roots=MappingProxyType(dict(sorted(roots.items()))),
        package_import_paths=MappingProxyType(dict(sorted(import_paths.items()))),
    )


def _load_calibration_runner_module() -> Any:
    if _CALIBRATION_RUNNER_MODULE_NAME in sys.modules:
        raise RuntimeError("refusing a preloaded Experiment 013 calibration runner")
    spec = importlib.util.spec_from_file_location(
        _CALIBRATION_RUNNER_MODULE_NAME,
        CALIBRATION_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load the Experiment 013 calibration artifact validators")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CALIBRATION_RUNNER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)
        raise
    return module


def _decode_execution_binding_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    runner: Any,
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
        from recurquant import experiment013_parquet, experiment013_source
    except ImportError as error:  # pragma: no cover - installation guard
        raise RuntimeError("Experiment 013 source/Parquet validators are unavailable") from error
    source_value = _strict_json(source_bytes, context="repository source manifest")
    normalized_source = experiment013_source.validate_experiment013_source_manifest(source_value)
    if (
        experiment013_source.canonical_experiment013_source_manifest_bytes(normalized_source)
        != source_bytes
    ):
        raise ValueError("repository source manifest is not canonical JSON")
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
        or sha256_bytes(parquet_bytes)
        != resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
        or parquet_bytes != parquet_path.read_bytes()
    ):
        raise ValueError("Parquet materialization manifest file identity drifted")
    experiment013_parquet.load_experiment013_parquet_manifest(parquet_path)
    bindings = {
        field: sha256_bytes(artifacts[field]) for field in sorted(resolver.EXECUTION_BINDING_FIELDS)
    }
    return _DecodedExecutionBindingArtifacts(
        bindings=MappingProxyType(bindings),
        source_manifest=MappingProxyType(dict(normalized_source)),
        runtime_manifest=runtime_manifest,
        model_manifest=model_manifest,
        source_module=experiment013_source,
        parquet_module=experiment013_parquet,
    )


def _validate_execution_binding_artifacts(
    artifacts: Mapping[str, bytes],
) -> dict[str, str]:
    """Strictly decode all four artifacts without retaining an imported runner."""

    runner = _load_calibration_runner_module()
    try:
        decoded = _decode_execution_binding_artifacts(artifacts, runner=runner)
        return dict(decoded.bindings)
    finally:
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is runner:
            sys.modules.pop(_CALIBRATION_RUNNER_MODULE_NAME, None)


def _verify_loaded_runner_source(
    runner: Any,
    source_manifest: Mapping[str, object],
) -> None:
    entries = {
        str(item["path"]): item
        for item in source_manifest["paths"]  # type: ignore[index]
    }
    relative = "scripts/run_static_q468_calibration.py"
    entry = entries.get(relative)
    raw_file = getattr(runner, "__file__", None)
    if not isinstance(entry, Mapping) or not isinstance(raw_file, (str, os.PathLike)):
        raise RuntimeError("calibration runner is absent from the authenticated source manifest")
    declared = Path(raw_file)
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("calibration runner source is unavailable") from error
    if (
        declared.is_symlink()
        or resolved != CALIBRATION_RUNNER_PATH.resolve(strict=True)
        or sha256_bytes(resolved.read_bytes()) != entry["raw_sha256"]
    ):
        raise RuntimeError("loaded calibration runner source bytes drifted")


def _validate_runtime_context_for_manifest(
    context: _RuntimeAuthenticationContext,
    runtime_manifest: Any,
) -> None:
    expected_import_paths = {
        item.name: item.import_path for item in runtime_manifest.package_roots
    }
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
    previous: _AuthenticatedExecutionBindings | None = None,
) -> _AuthenticatedExecutionBindings:
    """Reauthenticate source, runtime, model metadata, and Parquet at point of use."""

    created_runner = previous is None
    if created_runner:
        runner = _load_calibration_runner_module()
    else:
        runner = previous.runner
        if sys.modules.get(_CALIBRATION_RUNNER_MODULE_NAME) is not runner:
            raise RuntimeError("authenticated calibration runner module binding drifted")
        if runtime_context != previous.runtime_context:
            raise ValueError("runtime authentication context changed during capture")
    try:
        decoded = _decode_execution_binding_artifacts(artifacts, runner=runner)
        if previous is not None and dict(decoded.bindings) != dict(previous.bindings):
            raise ValueError("execution-binding artifacts changed during capture")
        verified_source = decoded.source_module.verify_experiment013_source_manifest(
            decoded.source_manifest,
            repo_root=REPOSITORY_ROOT,
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
        )
        if (
            authenticated_runtime.manifest_file_sha256
            != decoded.bindings["calibration_runtime_manifest_file_sha256"]
        ):
            raise RuntimeError("runtime authenticator returned a different manifest identity")
        live_model_manifest = runner.capture_model_file_manifest_from_hub(
            resolver.PRIMARY_MODEL_ID,
            resolver.PRIMARY_MODEL_REVISION,
            transformers_version=resolver.TRANSFORMERS_VERSION,
        )
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
    collect_tokens: bool,
) -> tuple[dict[str, Any], TokenCaptureSink]:
    """Run the sole capture flow, optionally retaining formatter token IDs."""

    if phase in resolver.PROTECTED_STAGES:
        raise PermissionError(f"{phase} is protected; capture v4 refuses it before source access")
    if phase not in resolver.ALLOWED_PHASES:
        raise ValueError(f"unsupported identity phase: {phase!r}")
    if phase == "stage_a" and calibration_binding is None:
        raise ValueError("Stage A requires a frozen calibration binding")
    if phase == "calibration" and calibration_binding is not None:
        raise ValueError("calibration capture forbids a Stage-A binding")

    if runtime_authentication_context is None:
        runtime_provider = getattr(source, "runtime_authentication_context", None)
        if runtime_provider is None or not callable(runtime_provider):
            raise ValueError("capture requires an explicit sealed runtime authentication context")
        runtime_authentication_context = runtime_provider()
    runtime_context = _normalize_runtime_authentication_context(
        runtime_authentication_context
    )
    if execution_binding_artifacts is None:
        fixture_provider = getattr(source, "execution_binding_artifacts", None)
        if fixture_provider is None or not callable(fixture_provider):
            raise ValueError("capture requires all four verified execution-binding artifacts")
        execution_binding_artifacts = fixture_provider()
    authentication = _authenticate_execution_binding_artifacts(
        execution_binding_artifacts,
        runtime_context=runtime_context,
    )
    try:
        before = _validate_heads(source.source_heads(), context="pre-capture")
        material = source.tokenizer_material()
        tokenizer_contract, tokenizer_manifest_hash = _tokenizer_contract(material)
        ruler_bundle = _verify_complete_ruler_bundle(source, tokenizer_material=material)
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
            result["calibration_binding"] = _normalize_calibration_binding(calibration_binding)
        expected_revisions = dict(resolver.FROZEN_DATASET_REVISIONS)
        resolver.build_candidate(
            result,
            expected_revisions=expected_revisions,
            calibration_binding_artifact=calibration_binding,
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
) -> dict[str, Any]:
    """Capture one deterministic calibration or Stage-A resolver input."""

    result, _token_sink = _capture_identity_input_with_tokens(
        phase=phase,
        source=source,
        calibration_binding=calibration_binding,
        execution_binding_artifacts=execution_binding_artifacts,
        runtime_authentication_context=runtime_authentication_context,
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
        api = HfApi()
        return {
            "primary_model": str(
                api.model_info(
                    resolver.PRIMARY_MODEL_ID, revision=resolver.PRIMARY_MODEL_REVISION
                ).sha
            ),
            "mbpp": str(
                api.dataset_info(resolver.MBPP_DATASET_ID, revision=resolver.MBPP_REVISION).sha
            ),
            "pg19": str(
                api.dataset_info(resolver.PG19_DATASET_ID, revision=resolver.PG19_REVISION).sha
            ),
            "ruler": self._github_revision(resolver.RULER_SOURCE_ID, resolver.RULER_REVISION),
            "humaneval_plus": str(
                api.dataset_info(
                    resolver.HUMANEVAL_PLUS_DATASET_ID,
                    revision=resolver.HUMANEVAL_PLUS_REVISION,
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
        path = self.ruler_receipt_dir / "generation-manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing complete RULER generation manifest: {path}")
        return path.read_bytes()

    def ruler_receipt_bytes(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> bytes:
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
    if args.base_runtime_root is None or args.staged_interpreter is None:
        raise ValueError("capture requires --base-runtime-root and --staged-interpreter")
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
    parser.add_argument("--staged-interpreter", type=Path)
    parser.add_argument("--package-root", action="append", default=[])
    parser.add_argument("--package-import-path", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase in resolver.PROTECTED_STAGES:
        raise PermissionError(
            f"{args.phase} is protected; capture v4 refuses it before file or source access"
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
        "parquet_materialization_manifest_file_sha256": (
            args.parquet_materialization_manifest
        ),
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
