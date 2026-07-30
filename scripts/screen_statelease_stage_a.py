#!/usr/bin/env python3
"""Run the one permitted Experiment 011 StateLease Stage-A falsification screen.

The command is deliberately difficult to run accidentally.  It authenticates
the Experiment 010 administrative-null provenance, committed Experiment 009
task-666 evidence, both frozen selector artifacts, an independently verified
production Stage-0 artifact, the exact repository source set, the complete
runtime/dependency readiness receipt, and the pinned model configuration
before reserving the single Stage-A attempt.  Only then may it read the
already-open task 666, tokenize it, or load model weights.

Stage A cannot support a public improvement, novelty, deployment, speed,
state-of-the-art, or breakthrough claim, even when every frozen gate passes.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from recurquant.cache import iter_recurrent_states
from recurquant.evaluation import TokenFidelity, fidelity_summary, token_fidelity
from recurquant.evidence import canonical_json_bytes, verify_evidence_artifact
from recurquant.statelease_evaluation import (
    EQUAL_BYTE_NO_REPLAY_METHODS,
    FIXED_REPLAY_METHODS,
    FROZEN_STATELEASE_RESIDENT_BYTES,
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    TrajectoryNmseAccumulator,
    evaluate_statelease_stage_a_gate,
    reference_aligned_trajectory_nmse,
)

SEED = 2339
ARTIFACT_KIND = "recurquant_experiment011_statelease_stage_a_falsification"
ARTIFACT_SCHEMA_VERSION = 1

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MODEL_DTYPE = torch.bfloat16
TASK_ID = 666
TASK_ROW_SHA256 = "b4f5989005c921c3ab94ab52c8115e79f99a22390bc1d6e6235d36fd02687fb9"
PROMPT_TEXT_SHA256 = "b6f0f93b9d15b96ac42bbabbdb349a09d2d24e57667d47cafe900c1ea91fd64b"
CODE_TEXT_SHA256 = "d2701e79ccd968c9e5af78474af16256f3bbf39cdfedbec2199ac92e1a4f397e"
PROMPT_TOKEN_IDS_SHA256 = "729215c4c99cdf96b13ad73f6ac7b537ddf9e882409b77e479d609aee046bffa"
CODE_TOKEN_IDS_SHA256 = "a920370c4892513c8a5cdb9f88a33fd95d4c90201af39fdb7d517f3ad42a9d9a"
TOKEN_ID_HASH_SERIALIZATION = "sha256(recurquant.evidence.canonical_json_bytes(list[int]))"
PROMPT_TOKENS = 69
CODE_TOKENS = 39
ALIGNED_TOKENS = 38

ORIGINAL_RHT_METHOD = RHT_CQER_METHOD
FP32_METHOD = "fp32_reference"
QUALITY_METHODS = (
    ORIGINAL_RHT_METHOD,
    STATELEASE_METHOD,
    *FIXED_REPLAY_METHODS,
    *EQUAL_BYTE_NO_REPLAY_METHODS,
)
ALL_METHODS = (FP32_METHOD, *QUALITY_METHODS)
FROZEN_STAGE_A_FORWARD_PASSES = 429
if len(ALL_METHODS) * (1 + ALIGNED_TOKENS) != FROZEN_STAGE_A_FORWARD_PASSES:
    raise RuntimeError("frozen Stage-A forward-pass count drifted from the method matrix")

LINEAR_LAYER_INDICES = (
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
EXPECTED_GEOMETRY = {
    "num_hidden_layers": 24,
    "recurrent_layers": 18,
    "linear_layer_indices": list(LINEAR_LAYER_INDICES),
    "value_heads": 16,
    "key_rows": 128,
    "value_width": 128,
    "batch_size": 1,
    "recurrent_state_dtype": "torch.float32",
}

EXPERIMENT009_STAGE_A_RELATIVE_PATH = "evidence/experiment009-rht-cqer-stage-a-666-5be8d48.json"
EXPERIMENT009_STAGE_A_FILE_SHA256 = (
    "98a432843dc438f2d5fde34f8704f154ebc3ee12c93ba7c469369acfedfb15b5"
)
EXPERIMENT009_STAGE_A_CANONICAL_SHA256 = (
    "9e03a1e8cefb5801406a47a2e5e365686afb0a05e10e099a989cee616b505ed1"
)
EXPERIMENT009_MANIFEST_SHA256 = "3c21e7d6534bcdeb5b6af7fdf80560529639e27b77ffaa4f086491a7a3a50ea4"
EXPERIMENT009_MANIFEST_CANONICAL_SHA256 = (
    "f9d6715f9f280d5de83e72ff815987e7aff2be406ab501d99776b3f119434873"
)
EXPERIMENT009_TOKEN_MANIFEST_SHA256 = (
    "79732fe8d64bab58dae58f64f97426945a24f317ce4c6d25511fab82d0679643"
)
EXPERIMENT009_IDENTITY_SHA256 = "135a8fe824c8fd640f0738f23dc165b6fa17b7795c9832273a742d6bee8a850f"

SELECTOR_RELATIVE_PATH = "artifacts/experiment006-hrr-selector-8task-c2ad68b.json"
LOSS_SELECTOR_RELATIVE_PATH = "artifacts/experiment006-loss-selector-8task-c2ad68b.json"
SELECTOR_FILE_SHA256 = "d0c4267095ee3f5068627b189a1fd9f58cb02f6e25672d9b89dd0990e5b09330"
LOSS_SELECTOR_FILE_SHA256 = "95c16656edb32efbc985f2fea59e229634dd558f4f4bf04819b8efc37783a1d6"
SELECTOR_CANONICAL_SHA256 = "7970961fd88b522998189ad64f26b333aed9c88ff5f653de5449fd9e01d8cbc8"
LOSS_SELECTOR_CANONICAL_SHA256 = "bff4e33253990b8115e1f35e74516c4975c2fe4aac5066475afe968eb8a64609"

EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH = (
    "evidence/experiment010-statelease-stage-a-administrative-null.json"
)
EXPERIMENT010_ADMIN_NULL_FILE_SHA256 = (
    "2baa25005d4220f99ea784d21bce1c869311987b7ecc56cb9338f76c14b36d12"
)
EXPERIMENT010_ADMIN_NULL_CANONICAL_SHA256 = (
    "c5f779ed4fd5a48284e212dfaead9146cbd2bb0b53404a5628fd49bc74ee31f3"
)
EXPERIMENT010_ADMIN_NULL_NOTE_RELATIVE_PATH = (
    "research/EXPERIMENT_010_STAGE_A_ADMINISTRATIVE_NULL.md"
)
EXPERIMENT010_ATTEMPT_RELATIVE_PATH = "artifacts/experiment010-statelease-stage-a-666.attempt.json"
EXPERIMENT010_ATTEMPT_FILE_SHA256 = (
    "f53cbb53f043180d40e472cacda64397014b8a60ec065fabcb5c0738d53adc15"
)
EXPERIMENT010_OUTPUT_RELATIVE_PATH = "artifacts/experiment010-statelease-stage-a-666.json"
EXPERIMENT010_H0_COMMIT = "0e3dbcec2cb9cca1cdb062ec2491954ae052d7b9"
EXPERIMENT010_SEAL_COMMIT = "c0ef99c924121b981d7bbda8ba4b9b76d3b14f51"
EXPERIMENT010_SEAL_TREE = "e271ba8f11bdf588c361e6ffc797ec795671e7f8"
EXPERIMENT010_ONE_RUN_MARKER = "RecurQuant-One-Run: experiment010-stage-a-task666-v1"

OUTPUT_RELATIVE_PATH = "artifacts/experiment011-statelease-stage-a-666.json"
ATTEMPT_RELATIVE_PATH = "artifacts/experiment011-statelease-stage-a-666.attempt.json"
IDENTITY_NOTE_RELATIVE_PATH = "research/EXPERIMENT_011_STAGE_A_IDENTITY.md"
IDENTITY_NOTE_FILE_SHA256 = "9a1a855df14ba96e05bc948d016d1f360dadcdb5a510a15f02b87f26e4390536"
PROTOCOL_NOTE_RELATIVE_PATH = "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md"
PROTOCOL_NOTE_FILE_SHA256 = "29ad6a7d6c6eec243191a0d444a748219ed2ed12ab42f48e01af7316c8ab2737"
ONE_RUN_MARKER = "RecurQuant-One-Run: experiment011-stage-a-task666-v1"
ONE_RUN_LIMITATION = (
    "The local Git commit plus reflog is tamper-evident for normal repository "
    "operations, not cryptographically non-bypassable against deliberate ref and "
    "reflog destruction; external append-only anchoring is outside this evaluator."
)
POSTSEAL_RECEIPT_LIMITATION = (
    "The complete prepared receipt is exclusively written and file-fsynced before "
    "the Git compare-and-swap. If post-CAS status promotion is interrupted, the "
    "prepared receipt plus HEAD at the proposed seal is sufficient consumed evidence. "
    "A filesystem or hardware failure that loses already-fsynced data remains outside "
    "the evaluator's guarantees; it never resets or automatically retries."
)

EXPECTED_RUNTIME_PACKAGES = {
    "datasets": "4.8.5",
    "fsspec": "2026.2.0",
    "huggingface-hub": "1.26.0",
    "numpy": "2.4.6",
    "pyarrow": "25.0.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cu128",
    "transformers": "5.14.1",
}
RUNTIME_PACKAGE_IMPORTS = {
    "datasets": "datasets",
    "fsspec": "fsspec",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
}
EXPECTED_RUNTIME_PACKAGE_MANIFEST_SHA256 = (
    "2466ad25043894fcd1604c97c373e5d5680061fdb7637f861b83d5c9465c31fe"
)
MODEL_WEIGHT_RESOURCE_FILENAME = "model.safetensors-00001-of-00001.safetensors"
MODEL_WEIGHT_RESOURCE_SIZE = 1_746_942_600
MODEL_WEIGHT_RESOURCE_SHA256 = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
MODEL_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"
MODEL_WEIGHT_INDEX_TENSOR_BYTES = 1_746_882_752
ALLOWED_MODEL_WEIGHT_FILENAMES = frozenset(
    {
        MODEL_WEIGHT_INDEX_FILENAME,
        MODEL_WEIGHT_RESOURCE_FILENAME,
    }
)
ALTERNATE_MODEL_WEIGHT_FILENAMES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "flax_model.msgpack",
        "model.ckpt.index",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "tf_model.h5",
    }
)
ALTERNATE_TOKENIZER_RESOURCE_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_templates",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.model",
    }
)
MODEL_CACHE_RESOURCES = {
    "config.json": {
        "size_bytes": 2_907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    MODEL_WEIGHT_INDEX_FILENAME: {
        "size_bytes": 50_900,
        "sha256": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    },
    MODEL_WEIGHT_RESOURCE_FILENAME: {
        "size_bytes": MODEL_WEIGHT_RESOURCE_SIZE,
        "sha256": MODEL_WEIGHT_RESOURCE_SHA256,
    },
    "tokenizer_config.json": {
        "size_bytes": 16_712,
        "sha256": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
    },
    "tokenizer.json": {
        "size_bytes": 12_807_196,
        "sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    },
    "merges.txt": {
        "size_bytes": 3_353_259,
        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    },
    "vocab.json": {
        "size_bytes": 6_722_759,
        "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    },
}
DATASET_CACHE_REVISION_RELATIVE_ID = (
    "google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c"
)
DATASET_CACHE_RESOURCES = {
    "dataset_info.json": {
        "size_bytes": 1_069,
        "sha256": "141cbe58ff5cb6fe53772f36a41520c1f7f3adda9f773848e11fa7a5bd40123c",
    },
    "mbpp-train.arrow": {
        "size_bytes": 178_448,
        "sha256": "dbd85255cf0fad7b11f3b39233045a0ab1799c4fe51846ec57946e0abe59ed70",
    },
}

EXPECTED_DATASET_MANIFEST = {
    "config": "full",
    "dataset_id": "google-research-datasets/mbpp",
    "formatter_version": "recurquant.mbpp-prompt-code.v1",
    "phase": "calibration",
    "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "row_count": 1,
    "rows": [{"sha256": TASK_ROW_SHA256, "task_id": TASK_ID}],
    "schema": "recurquant.mbpp-manifest.v1",
    "selection_namespace": "rq-v0.2",
    "source_split": "train",
}
EXPECTED_TOKEN_MANIFEST = [
    {
        "aligned_scored_tokens": ALIGNED_TOKENS,
        "code_tokens": CODE_TOKENS,
        "full_code_scored_tokens": CODE_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "task_id": TASK_ID,
    }
]
EXPECTED_TASK_IDENTITY = {
    "aligned_scored_tokens": ALIGNED_TOKENS,
    "authenticated_before_model_weights": True,
    "code_tokens": CODE_TOKENS,
    "prompt_tokens": PROMPT_TOKENS,
    "row_sha256": TASK_ROW_SHA256,
    "task_id": TASK_ID,
}

SOURCE_FILES = (
    "pyproject.toml",
    EXPERIMENT010_ATTEMPT_RELATIVE_PATH,
    EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH,
    "scripts/capture_statelease_stage0.py",
    "scripts/screen_rht_cqer.py",
    "scripts/screen_statelease_stage_a.py",
    "scripts/verify_statelease_stage0.py",
    IDENTITY_NOTE_RELATIVE_PATH,
    PROTOCOL_NOTE_RELATIVE_PATH,
    "research/EXPERIMENT_010_STAGE_A_IDENTITY.md",
    "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
    EXPERIMENT010_ADMIN_NULL_NOTE_RELATIVE_PATH,
    "src/recurquant/__init__.py",
    "src/recurquant/cache.py",
    "src/recurquant/cli.py",
    "src/recurquant/confirmation.py",
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
    "src/recurquant/policies.py",
    "src/recurquant/public_data.py",
    "src/recurquant/quantization.py",
    "src/recurquant/query_energy.py",
    "src/recurquant/qwen35.py",
    "src/recurquant/qwen35_quickstart.py",
    "src/recurquant/rht.py",
    "src/recurquant/row_policy.py",
    "src/recurquant/signals.py",
    "src/recurquant/statelease.py",
    "src/recurquant/statelease_baselines.py",
    "src/recurquant/statelease_cache.py",
    "src/recurquant/statelease_equal_byte_baselines.py",
    "src/recurquant/statelease_equal_byte_cache.py",
    "src/recurquant/statelease_evaluation.py",
    "src/recurquant/statelease_observer.py",
    "src/recurquant/storage_boundary_validation.py",
    "src/recurquant/transformers_cache.py",
    "src/recurquant/transition_observer.py",
    "src/recurquant/triton_state.py",
    "tests/test_capture_statelease_stage0.py",
    "tests/test_mixed_quantization.py",
    "tests/test_multibit_policy.py",
    "tests/test_multibit_quantization.py",
    "tests/test_quantization.py",
    "tests/test_qwen35_factory.py",
    "tests/test_rht.py",
    "tests/test_right_rht_query_ema_cache.py",
    "tests/test_row_policy.py",
    "tests/test_statelease.py",
    "tests/test_statelease_baselines.py",
    "tests/test_statelease_cache.py",
    "tests/test_statelease_equal_byte_baselines.py",
    "tests/test_statelease_equal_byte_cache.py",
    "tests/test_statelease_evaluation.py",
    "tests/test_statelease_observer.py",
    "tests/test_screen_statelease_stage_a.py",
    "tests/test_verify_statelease_stage0.py",
)

CLAIM_BOUNDARY = (
    "Experiment 011 Stage A is one-task, already-open falsification evidence. "
    "Passing only authorizes the separately frozen development-identity step. "
    "It cannot support a public improvement, novelty, deployment, speed, "
    "state-of-the-art, or breakthrough claim."
)
PRESEAL_FRESHNESS_KEYS = frozenset(
    {
        "repository_reauthenticated",
        "source_head_blobs_reauthenticated",
        "new_marker_absent",
        "stage0_reauthenticated",
        "experiment010_administrative_null_reauthenticated",
        "runtime_readiness_reauthenticated",
        "configuration_reauthenticated_local_only",
    }
)
RESULT_PREPARED_STATUS = "result_prepared_before_output_publish"
RESULT_COMPLETED_STATUS = "completed_with_authenticated_stage_a_result"
RESULT_PROMOTION_INTERRUPTED_STATUS = "completed_result_published_receipt_promotion_interrupted"
RESULT_PUBLICATION_FAILED_STATUS = "result_prepared_output_publication_failed"


class StageAAuthenticationError(RuntimeError):
    """An input, source, identity, or one-run condition failed closed."""


@dataclass(frozen=True, slots=True)
class StageAPreflight:
    repo_root: Path
    repository_start: dict[str, object]
    source_hashes_start: dict[str, str]
    identity_clarification: dict[str, object]
    experiment010_admin_null: dict[str, object]
    anchor: dict[str, object]
    selector_identity: dict[str, object]
    plan: object
    stage0: dict[str, object]
    stage0_artifact: Path
    stage0_sha256: Path
    output_path: Path
    attempt_path: Path


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    config: object
    identity: dict[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeReadiness:
    receipt: dict[str, object]
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizedTask:
    row: Mapping[str, object]
    prompt_ids: torch.Tensor
    code_ids: torch.Tensor
    token_manifest: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    receipt: dict[str, object]
    receipt_file_sha256: str
    h0_commit: str
    seal_commit: str
    tree: str
    seal_message_sha256: str


@dataclass(slots=True)
class AccessLedger:
    phase: str = "reserved_before_task_entry"
    task_load_entered: bool = False
    task_row_loaded: bool = False
    tokenizer_entered: bool = False
    tokenizer_loaded: bool = False
    model_weights_entered: bool = False
    model_weights_loaded: bool = False
    evaluation_entered: bool = False
    evaluation_returned: bool = False
    finalization_entered: bool = False

    def snapshot(self) -> dict[str, object]:
        def conservative(completed: bool, entered: bool) -> bool | None:
            if completed:
                return True
            if entered:
                return None
            return False

        return {
            "phase": self.phase,
            "task_load_entered": self.task_load_entered,
            "task_row_loaded": conservative(
                self.task_row_loaded,
                self.task_load_entered,
            ),
            "tokenizer_entered": self.tokenizer_entered,
            "tokenizer_loaded": conservative(
                self.tokenizer_loaded,
                self.tokenizer_entered,
            ),
            "model_weights_entered": self.model_weights_entered,
            "model_weights_loaded": conservative(
                self.model_weights_loaded,
                self.model_weights_entered,
            ),
            "evaluation_entered": self.evaluation_entered,
            "evaluation_returned": conservative(
                self.evaluation_returned,
                self.evaluation_entered,
            ),
            "forward_passes": (
                FROZEN_STAGE_A_FORWARD_PASSES
                if self.evaluation_returned
                else (None if self.evaluation_entered else 0)
            ),
            "forward_passes_minimum": (
                FROZEN_STAGE_A_FORWARD_PASSES if self.evaluation_returned else 0
            ),
            "quality_result_computed": conservative(
                self.evaluation_returned,
                self.evaluation_entered,
            ),
            "finalization_entered": self.finalization_entered,
        }


ACCESS_PHASE_ORDER = (
    "reserved_before_task_entry",
    "task_load_entered",
    "task_row_loaded",
    "tokenizer_entered",
    "tokenizer_loaded",
    "model_weights_entered",
    "model_weights_loaded",
    "evaluation_entered",
    "evaluation_returned",
    "finalization_entered",
)


def _expected_access_snapshot(phase: str) -> dict[str, object]:
    try:
        phase_index = ACCESS_PHASE_ORDER.index(phase)
    except ValueError as error:
        raise StageAAuthenticationError(
            f"unknown Stage-A access-ledger phase: {phase!r}"
        ) from error
    ledger = AccessLedger(phase=phase)
    ledger.task_load_entered = phase_index >= 1
    ledger.task_row_loaded = phase_index >= 2
    ledger.tokenizer_entered = phase_index >= 3
    ledger.tokenizer_loaded = phase_index >= 4
    ledger.model_weights_entered = phase_index >= 5
    ledger.model_weights_loaded = phase_index >= 6
    ledger.evaluation_entered = phase_index >= 7
    ledger.evaluation_returned = phase_index >= 8
    ledger.finalization_entered = phase_index >= 9
    return ledger.snapshot()


def _validated_access_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise StageAAuthenticationError("attempt receipt lacks its access ledger")
    phase = value.get("phase")
    if not isinstance(phase, str):
        raise StageAAuthenticationError("attempt access-ledger phase is malformed")
    expected = _expected_access_snapshot(phase)
    if dict(value) != expected:
        raise StageAAuthenticationError(
            "attempt access ledger is not the exact frozen snapshot for its phase"
        )
    return expected


def _assert_adjacent_access_transition(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    *,
    allow_equal: bool,
) -> None:
    previous_phase = str(previous["phase"])
    current_phase = str(current["phase"])
    previous_index = ACCESS_PHASE_ORDER.index(previous_phase)
    current_index = ACCESS_PHASE_ORDER.index(current_phase)
    allowed = {previous_index + 1}
    if allow_equal:
        allowed.add(previous_index)
    if current_index not in allowed:
        raise StageAAuthenticationError("attempt access ledger regressed or skipped a frozen phase")


@dataclass(frozen=True, slots=True)
class AccessHooks:
    """Dependency-injected ordering contract used by production and tests."""

    authenticate: Callable[[], object]
    authenticate_readiness: Callable[[object], object]
    load_config: Callable[[object], object]
    reserve_attempt: Callable[[object, object, object], object]
    load_exact_task: Callable[[object], object]
    tokenize_task: Callable[[object, object], object]
    load_weights: Callable[[object, object], object]
    evaluate: Callable[[object, object, object, object], object]
    record_access_transition: Callable[[object, AccessLedger], object]
    record_evaluation_returned: Callable[[object, AccessLedger], object]
    finalize: Callable[[object, object, object, object, object, object], object]
    record_failure: Callable[[object, BaseException, AccessLedger], None]


def run_ordered_access(hooks: AccessHooks) -> object:
    """Enforce complete readiness before sealing/data and tokens before weights."""

    authenticated = hooks.authenticate()
    readiness = hooks.authenticate_readiness(authenticated)
    configuration = hooks.load_config(authenticated)
    attempt = hooks.reserve_attempt(authenticated, configuration, readiness)
    ledger = AccessLedger()
    try:
        ledger.phase = "task_load_entered"
        ledger.task_load_entered = True
        attempt = hooks.record_access_transition(attempt, ledger)
        row = hooks.load_exact_task(authenticated)
        ledger.phase = "task_row_loaded"
        ledger.task_row_loaded = True
        attempt = hooks.record_access_transition(attempt, ledger)
        ledger.tokenizer_entered = True
        ledger.phase = "tokenizer_entered"
        attempt = hooks.record_access_transition(attempt, ledger)
        tokenized = hooks.tokenize_task(authenticated, row)
        ledger.tokenizer_loaded = True
        ledger.phase = "tokenizer_loaded"
        attempt = hooks.record_access_transition(attempt, ledger)
        ledger.model_weights_entered = True
        ledger.phase = "model_weights_entered"
        attempt = hooks.record_access_transition(attempt, ledger)
        model = hooks.load_weights(configuration, authenticated)
        ledger.model_weights_loaded = True
        ledger.phase = "model_weights_loaded"
        attempt = hooks.record_access_transition(attempt, ledger)
        ledger.evaluation_entered = True
        ledger.phase = "evaluation_entered"
        attempt = hooks.record_access_transition(attempt, ledger)
        result = hooks.evaluate(model, tokenized, authenticated, configuration)
        ledger.evaluation_returned = True
        ledger.phase = "evaluation_returned"
        attempt = hooks.record_evaluation_returned(attempt, ledger)
        ledger.finalization_entered = True
        ledger.phase = "finalization_entered"
        attempt = hooks.record_access_transition(attempt, ledger)
        return hooks.finalize(
            result,
            model,
            tokenized,
            authenticated,
            configuration,
            attempt,
        )
    except BaseException as error:
        hooks.record_failure(attempt, error, ledger)
        raise


def _script_module(name: str) -> Any:
    try:
        return importlib.import_module(f"scripts.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_mapping(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise StageAAuthenticationError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageAAuthenticationError(
            f"cannot read authenticated JSON {path.name}: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StageAAuthenticationError(f"{path.name} must contain a JSON object")
    return value


def _sanitized_git_environment(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a Git environment with no inherited repository/object routing."""

    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_COUNT"] = "4"
    environment["GIT_CONFIG_KEY_0"] = "core.hooksPath"
    environment["GIT_CONFIG_VALUE_0"] = os.devnull
    environment["GIT_CONFIG_KEY_1"] = "core.fsmonitor"
    environment["GIT_CONFIG_VALUE_1"] = "false"
    environment["GIT_CONFIG_KEY_2"] = "core.untrackedCache"
    environment["GIT_CONFIG_VALUE_2"] = "false"
    environment["GIT_CONFIG_KEY_3"] = "core.autocrlf"
    environment["GIT_CONFIG_VALUE_3"] = "false"
    if extra is not None:
        allowed = {
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
        }
        if any(key.upper() not in allowed for key in extra):
            raise StageAAuthenticationError("unsupported explicit Git environment override")
        environment.update({key.upper(): value for key, value in extra.items()})
    return environment


def _run_git_process(
    repo_root: Path,
    *arguments: str,
    input_text: str | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
        env=_sanitized_git_environment(environment_overrides),
    )


def _git(repo_root: Path, *arguments: str) -> str:
    process = _run_git_process(repo_root, *arguments)
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"git {' '.join(arguments)} failed: {message}")
    return process.stdout.strip()


def _git_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise StageAAuthenticationError("resolved Git repository path is unavailable") from error


def _assert_git_repository_identity(repo_root: Path) -> dict[str, object]:
    """Authenticate the exact worktree, Git directory, and object view."""

    try:
        expected_root = repo_root.resolve(strict=True)
    except OSError as error:
        raise StageAAuthenticationError("Stage-A repository root is unavailable") from error
    top_level = _git_path(repo_root, _git(repo_root, "rev-parse", "--show-toplevel"))
    git_dir = _git_path(repo_root, _git(repo_root, "rev-parse", "--absolute-git-dir"))
    common_dir = _git_path(repo_root, _git(repo_root, "rev-parse", "--git-common-dir"))
    object_dir = _git_path(repo_root, _git(repo_root, "rev-parse", "--git-path", "objects"))
    index_path_raw = _git(repo_root, "rev-parse", "--git-path", "index")
    index_path = Path(index_path_raw)
    if not index_path.is_absolute():
        index_path = repo_root / index_path
    index_path = index_path.resolve(strict=False)

    if top_level != expected_root:
        raise StageAAuthenticationError("Git top-level does not match the Stage-A repository root")
    if _git(repo_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise StageAAuthenticationError(
            "Stage A is not executing inside the authenticated worktree"
        )
    if _git(repo_root, "rev-parse", "--is-bare-repository") != "false":
        raise StageAAuthenticationError("Stage A refuses a bare Git repository")
    if _git(repo_root, "rev-parse", "--show-object-format") != "sha1":
        raise StageAAuthenticationError("Stage A requires the frozen SHA-1 Git object format")
    if _git(repo_root, "rev-parse", "--is-shallow-repository") != "false":
        raise StageAAuthenticationError("Stage A refuses a shallow Git history")
    if not git_dir.is_dir() or not common_dir.is_dir():
        raise StageAAuthenticationError("resolved Git directory identity is malformed")
    dot_git = expected_root / ".git"
    if dot_git.is_dir():
        expected_git_dir = dot_git.resolve(strict=True)
        if git_dir != expected_git_dir or common_dir != expected_git_dir:
            raise StageAAuthenticationError(
                "Git directory is not the authenticated repository's .git store"
            )
        worktree_layout = "primary"
    elif dot_git.is_file():
        try:
            pointer = dot_git.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise StageAAuthenticationError(
                "cannot authenticate the linked-worktree Git pointer"
            ) from error
        if not pointer.startswith("gitdir: "):
            raise StageAAuthenticationError("linked-worktree Git pointer is malformed")
        pointer_path = Path(pointer.removeprefix("gitdir: "))
        if not pointer_path.is_absolute():
            pointer_path = expected_root / pointer_path
        try:
            pointer_git_dir = pointer_path.resolve(strict=True)
        except OSError as error:
            raise StageAAuthenticationError(
                "linked-worktree Git directory is unavailable"
            ) from error
        if (
            pointer_git_dir != git_dir
            or git_dir.parent.name != "worktrees"
            or git_dir.parent.parent != common_dir
        ):
            raise StageAAuthenticationError(
                "linked-worktree Git directory/common-store identity drifted"
            )
        back_pointer = git_dir / "gitdir"
        try:
            linked_dot_git = Path(back_pointer.read_text(encoding="utf-8").strip()).resolve(
                strict=True
            )
        except (OSError, UnicodeError) as error:
            raise StageAAuthenticationError(
                "linked-worktree reverse Git pointer is unavailable"
            ) from error
        if linked_dot_git != dot_git.resolve(strict=True):
            raise StageAAuthenticationError("linked-worktree reverse Git pointer drifted")
        worktree_layout = "linked"
    else:
        raise StageAAuthenticationError("repository .git identity is unavailable")
    expected_object_dir = (common_dir / "objects").resolve(strict=True)
    if object_dir != expected_object_dir or not object_dir.is_dir():
        raise StageAAuthenticationError(
            "Git object directory is not the authenticated common store"
        )
    expected_index = (git_dir / "index").resolve(strict=False)
    if index_path != expected_index:
        raise StageAAuthenticationError("Git index path is redirected from the authenticated store")

    alternates = object_dir / "info" / "alternates"
    http_alternates = object_dir / "info" / "http-alternates"
    for alternate_file in (alternates, http_alternates):
        if not alternate_file.exists():
            continue
        try:
            alternate_bytes = alternate_file.read_bytes()
        except OSError as error:
            raise StageAAuthenticationError("cannot authenticate Git object alternates") from error
        if alternate_bytes.strip():
            raise StageAAuthenticationError("Git alternate object stores are forbidden")
    shallow = common_dir / "shallow"
    if shallow.exists():
        try:
            shallow_bytes = shallow.read_bytes()
        except OSError as error:
            raise StageAAuthenticationError("cannot authenticate Git shallow history") from error
        if shallow_bytes.strip():
            raise StageAAuthenticationError("Git shallow history metadata is forbidden")
    if _git(repo_root, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise StageAAuthenticationError("Git replacement refs are forbidden")
    grafts = common_dir / "info" / "grafts"
    if grafts.exists():
        try:
            graft_bytes = grafts.read_bytes()
        except OSError as error:
            raise StageAAuthenticationError("cannot authenticate Git history grafts") from error
        if graft_bytes.strip():
            raise StageAAuthenticationError("Git history grafts are forbidden")

    def identity_digest(path: Path) -> str:
        normalized = os.path.normcase(str(path))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return {
        "top_level_matches_repo_root": True,
        "inside_worktree": True,
        "bare_repository": False,
        "git_directory_authenticated": True,
        "common_git_directory_authenticated": True,
        "index_path_authenticated": True,
        "object_directory_authenticated": True,
        "alternate_object_stores_absent": True,
        "shallow_history_absent": True,
        "replacement_refs_absent": True,
        "history_grafts_absent": True,
        "object_format": "sha1",
        "inherited_git_environment_scrubbed": True,
        "system_and_global_git_config_disabled": True,
        "worktree_layout": worktree_layout,
        "top_level_identity_sha256": identity_digest(top_level),
        "git_directory_identity_sha256": identity_digest(git_dir),
        "common_git_directory_identity_sha256": identity_digest(common_dir),
        "index_path_identity_sha256": identity_digest(index_path),
        "object_directory_identity_sha256": identity_digest(object_dir),
    }


def _repository_state(repo_root: Path) -> dict[str, object]:
    git_identity = _assert_git_repository_identity(repo_root)
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    branch = _git(repo_root, "branch", "--show-current")
    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": not status,
        "porcelain": status.splitlines(),
        "git_identity": git_identity,
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise StageAAuthenticationError(f"Stage-A source set is incomplete: {missing}")
    return {relative: _file_sha256(repo_root / relative) for relative in SOURCE_FILES}


def _assert_loaded_local_modules_declared(repo_root: Path) -> tuple[str, ...]:
    root = repo_root.resolve()
    declared = set(SOURCE_FILES)
    loaded: set[str] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if relative.startswith("src/recurquant/") or relative in {
            "scripts/capture_statelease_stage0.py",
            "scripts/screen_rht_cqer.py",
            "scripts/screen_statelease_stage_a.py",
            "scripts/verify_statelease_stage0.py",
        }:
            loaded.add(relative)
    omitted = sorted(loaded - declared)
    if omitted:
        raise StageAAuthenticationError(
            f"loaded local modules are absent from the frozen source set: {omitted}"
        )
    return tuple(sorted(loaded))


def _assert_sources_match_head(repo_root: Path) -> None:
    for relative in SOURCE_FILES:
        tree_entry = _git(repo_root, "ls-tree", "HEAD", "--", relative)
        parts = tree_entry.split(maxsplit=3)
        if len(parts) != 4 or parts[0] not in {"100644", "100755"} or parts[1] != "blob":
            raise StageAAuthenticationError(
                f"Stage-A source is not a tracked regular file at HEAD: {relative}"
            )
        head_blob = parts[2]
        worktree_blob = _git(
            repo_root,
            "hash-object",
            "--no-filters",
            "--",
            relative,
        )
        if worktree_blob != head_blob:
            raise StageAAuthenticationError(
                f"Stage-A source bytes differ from HEAD despite Git status: {relative}"
            )


def _assert_no_prior_stage_a_seal(repo_root: Path) -> None:
    history = _git(
        repo_root,
        "log",
        "--all",
        "--reflog",
        "--format=%B%x00",
    )
    if ONE_RUN_MARKER in history:
        raise StageAAuthenticationError(
            "the durable one-run seal is already present in repository history"
        )


def _validate_experiment010_administrative_null(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    if set(artifact) != {
        "artifact_kind",
        "canonical_evidence_sha256",
        "evidence",
    }:
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null artifact schema drifted"
        )
    if (
        artifact.get("artifact_kind")
        != "recurquant_experiment010_statelease_stage_a_administrative_null"
        or artifact.get("canonical_evidence_sha256") != EXPERIMENT010_ADMIN_NULL_CANONICAL_SHA256
    ):
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null artifact identity drifted"
        )
    evidence = artifact.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StageAAuthenticationError("Experiment 010 administrative-null evidence is missing")
    if (
        evidence.get("classification") != "infrastructure_failure_before_evaluation"
        or evidence.get("scientific_result_available") is not False
    ):
        raise StageAAuthenticationError("Experiment 010 administrative-null classification drifted")

    access = evidence.get("access_boundary")
    expected_false_access = (
        "aggregate_exposed",
        "candidate_metrics_computed",
        "dataset_load_dataset_called",
        "logits_computed",
        "model_weights_loaded",
        "protected_mbpp_window_accessed",
        "quality_result_exposed",
        "task_row_loaded",
        "tokenizer_loaded",
    )
    if (
        not isinstance(access, Mapping)
        or access.get("task_id") != TASK_ID
        or access.get("completed_task_ids", []) != []
        or access.get("forward_passes") != 0
        or any(access.get(key) is not False for key in expected_false_access)
    ):
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null access boundary drifted"
        )

    original = evidence.get("original_attempt")
    expected_original = {
        "attempt_number": 1,
        "automatic_rerun_authorized": False,
        "h0_commit": EXPERIMENT010_H0_COMMIT,
        "h0_tree": EXPERIMENT010_SEAL_TREE,
        "one_run_marker": EXPERIMENT010_ONE_RUN_MARKER,
        "raw_receipt_file_sha256": EXPERIMENT010_ATTEMPT_FILE_SHA256,
        "receipt_schema": "recurquant.experiment010.stage-a-attempt.v1",
        "receipt_status": "failed_without_authenticated_stage_a_result",
        "result_artifact_created": False,
        "seal_commit": EXPERIMENT010_SEAL_COMMIT,
        "seal_parent": EXPERIMENT010_H0_COMMIT,
        "seal_tree": EXPERIMENT010_SEAL_TREE,
        "seal_tree_matches_h0": True,
    }
    if not isinstance(original, Mapping) or any(
        original.get(key) != value for key, value in expected_original.items()
    ):
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null attempt provenance drifted"
        )

    recovery = evidence.get("recovery_disposition")
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("experiment010_resume_authorized") is not False
        or recovery.get("failed_seal_and_receipt_preserved") is not True
        or recovery.get("next_identity") != "Experiment 011"
        or recovery.get("scientific_method_or_gate_changed") is not False
    ):
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null recovery disposition drifted"
        )
    return {
        "classification": "infrastructure_failure_before_evaluation",
        "scientific_result_available": False,
        "quality_data_accessed": False,
        "original_h0_commit": EXPERIMENT010_H0_COMMIT,
        "original_seal_commit": EXPERIMENT010_SEAL_COMMIT,
        "original_seal_tree": EXPERIMENT010_SEAL_TREE,
        "original_one_run_marker": EXPERIMENT010_ONE_RUN_MARKER,
        "experiment010_resume_authorized": False,
        "next_experiment": "Experiment 011",
    }


def _validate_experiment010_failed_receipt(receipt: Mapping[str, object]) -> None:
    expected = {
        "schema": "recurquant.experiment010.stage-a-attempt.v1",
        "status": "failed_without_authenticated_stage_a_result",
        "attempt_number": 1,
        "task_id": TASK_ID,
        "output_path": EXPERIMENT010_OUTPUT_RELATIVE_PATH,
        "h0_repository_commit": EXPERIMENT010_H0_COMMIT,
        "one_run_seal_commit": EXPERIMENT010_SEAL_COMMIT,
        "one_run_seal_tree": EXPERIMENT010_SEAL_TREE,
        "one_run_marker": EXPERIMENT010_ONE_RUN_MARKER,
        "completed_task_ids": [],
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        "failure_type": "RuntimeError",
        "failure_message": (
            "MBPP loading requires the optional evaluation dependencies; install recurquant[eval]"
        ),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise StageAAuthenticationError("Experiment 010 failed-attempt receipt semantics drifted")


def _authenticate_experiment010_git_history(repo_root: Path) -> None:
    reachable = set(_git(repo_root, "rev-list", "--all", "--reflog").splitlines())
    if EXPERIMENT010_H0_COMMIT not in reachable or EXPERIMENT010_SEAL_COMMIT not in reachable:
        raise StageAAuthenticationError(
            "Experiment 010 H0 and one-run seal are not preserved in repository history"
        )
    if (
        _git(repo_root, "show", "-s", "--format=%P", EXPERIMENT010_SEAL_COMMIT)
        != EXPERIMENT010_H0_COMMIT
        or _git(repo_root, "show", "-s", "--format=%T", EXPERIMENT010_SEAL_COMMIT)
        != EXPERIMENT010_SEAL_TREE
        or _git(repo_root, "show", "-s", "--format=%T", EXPERIMENT010_H0_COMMIT)
        != EXPERIMENT010_SEAL_TREE
    ):
        raise StageAAuthenticationError("Experiment 010 preserved H0/seal ancestry or tree drifted")
    old_message = _git(
        repo_root,
        "show",
        "-s",
        "--format=%B",
        EXPERIMENT010_SEAL_COMMIT,
    )
    if EXPERIMENT010_ONE_RUN_MARKER not in old_message:
        raise StageAAuthenticationError("Experiment 010 preserved one-run marker is missing")


def authenticate_experiment010_administrative_null(
    repo_root: Path,
) -> dict[str, object]:
    admin_path = repo_root / EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH
    if not admin_path.is_file() or _file_sha256(admin_path) != EXPERIMENT010_ADMIN_NULL_FILE_SHA256:
        raise StageAAuthenticationError(
            "committed Experiment 010 administrative-null file hash drifted"
        )
    verification = verify_evidence_artifact(admin_path)
    if (
        verification.get("valid") is not True
        or verification.get("file_sha256") != EXPERIMENT010_ADMIN_NULL_FILE_SHA256
        or verification.get("computed_canonical_evidence_sha256")
        != EXPERIMENT010_ADMIN_NULL_CANONICAL_SHA256
    ):
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null canonical verification failed"
        )
    semantics = _validate_experiment010_administrative_null(_json_mapping(admin_path))

    receipt_path = repo_root / EXPERIMENT010_ATTEMPT_RELATIVE_PATH
    if (
        not receipt_path.is_file()
        or _file_sha256(receipt_path) != EXPERIMENT010_ATTEMPT_FILE_SHA256
    ):
        raise StageAAuthenticationError("Experiment 010 raw failed-attempt receipt hash drifted")
    _validate_experiment010_failed_receipt(_json_mapping(receipt_path))
    if (repo_root / EXPERIMENT010_OUTPUT_RELATIVE_PATH).exists():
        raise StageAAuthenticationError(
            "Experiment 010 result must remain absent after its administrative null"
        )

    _authenticate_experiment010_git_history(repo_root)

    provenance = {
        **semantics,
        "administrative_null_path": EXPERIMENT010_ADMIN_NULL_RELATIVE_PATH,
        "administrative_null_file_sha256": EXPERIMENT010_ADMIN_NULL_FILE_SHA256,
        "administrative_null_canonical_evidence_sha256": (
            EXPERIMENT010_ADMIN_NULL_CANONICAL_SHA256
        ),
        "raw_failed_receipt_path": EXPERIMENT010_ATTEMPT_RELATIVE_PATH,
        "raw_failed_receipt_file_sha256": EXPERIMENT010_ATTEMPT_FILE_SHA256,
        "experiment010_result_path": EXPERIMENT010_OUTPUT_RELATIVE_PATH,
        "experiment010_result_absent": True,
        "preserved_git_history_authenticated": True,
    }
    provenance["canonical_provenance_sha256"] = _canonical_sha256(provenance)
    return provenance


def _assert_repository_start(repo_root: Path) -> tuple[dict[str, object], dict[str, str]]:
    state = _repository_state(repo_root)
    if state["worktree_clean"] is not True:
        raise StageAAuthenticationError(
            "Stage A requires a clean tracked and untracked repository before any access"
        )
    _assert_sources_match_head(repo_root)
    _assert_no_prior_stage_a_seal(repo_root)
    return state, _source_hashes(repo_root)


def _assert_ignored_exact_output(repo_root: Path) -> tuple[Path, Path]:
    output = (repo_root / OUTPUT_RELATIVE_PATH).resolve()
    attempt = (repo_root / ATTEMPT_RELATIVE_PATH).resolve()
    expected_output = (repo_root.resolve() / OUTPUT_RELATIVE_PATH).resolve()
    expected_attempt = (repo_root.resolve() / ATTEMPT_RELATIVE_PATH).resolve()
    if output != expected_output or attempt != expected_attempt:
        raise StageAAuthenticationError("Stage-A output identity is not exact")
    for path in (output, attempt):
        ignored = _run_git_process(repo_root, "check-ignore", "--quiet", str(path))
        if ignored.returncode != 0:
            raise StageAAuthenticationError(
                f"one-run artifact path is not Git-ignored: {path.name}"
            )
        if path.exists():
            raise StageAAuthenticationError(
                f"one-run artifact already exists; refusing another Stage-A run: {path.name}"
            )
    return output, attempt


def _validate_anchor_payload(artifact: Mapping[str, object]) -> dict[str, object]:
    if set(artifact) != {
        "schema_version",
        "artifact_kind",
        "canonical_evidence_sha256",
        "evidence",
    }:
        raise StageAAuthenticationError("Experiment 009 Stage-A artifact schema drifted")
    if artifact["artifact_kind"] != "recurquant_rht_cqer32_stage_a_screen":
        raise StageAAuthenticationError("Experiment 009 Stage-A artifact kind drifted")
    if artifact["canonical_evidence_sha256"] != EXPERIMENT009_STAGE_A_CANONICAL_SHA256:
        raise StageAAuthenticationError("Experiment 009 canonical evidence digest drifted")
    evidence = artifact["evidence"]
    if not isinstance(evidence, Mapping):
        raise StageAAuthenticationError("Experiment 009 evidence must be an object")
    if evidence.get("screening_only") is not True:
        raise StageAAuthenticationError("Experiment 009 artifact is not a screening artifact")
    gate = evidence.get("stage_a_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise StageAAuthenticationError("Experiment 009 authenticated Stage-A gate did not pass")

    dataset = evidence.get("dataset")
    if not isinstance(dataset, Mapping):
        raise StageAAuthenticationError("Experiment 009 dataset evidence is missing")
    manifest = dataset.get("manifest")
    token_manifest = dataset.get("token_manifest")
    identity = dataset.get("identity")
    if manifest != EXPECTED_DATASET_MANIFEST:
        raise StageAAuthenticationError("Experiment 009 task-666 manifest drifted")
    if token_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError("Experiment 009 task-666 token manifest drifted")
    if identity != EXPECTED_TASK_IDENTITY:
        raise StageAAuthenticationError("Experiment 009 task-666 identity drifted")
    if dataset.get("manifest_sha256") != EXPERIMENT009_MANIFEST_SHA256:
        raise StageAAuthenticationError("Experiment 009 manifest receipt drifted")
    if _canonical_sha256(manifest) != EXPERIMENT009_MANIFEST_CANONICAL_SHA256:
        raise StageAAuthenticationError("Experiment 009 manifest bytes failed authentication")
    if _canonical_sha256(token_manifest) != EXPERIMENT009_TOKEN_MANIFEST_SHA256:
        raise StageAAuthenticationError("Experiment 009 token manifest failed authentication")
    if _canonical_sha256(identity) != EXPERIMENT009_IDENTITY_SHA256:
        raise StageAAuthenticationError("Experiment 009 task identity failed authentication")

    model = evidence.get("model")
    if not isinstance(model, Mapping) or {
        "id": model.get("id"),
        "revision": model.get("revision"),
        "dtype": model.get("dtype"),
    } != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": str(MODEL_DTYPE),
    }:
        raise StageAAuthenticationError("Experiment 009 model identity drifted")
    selectors = evidence.get("selector_artifacts")
    if not isinstance(selectors, Mapping):
        raise StageAAuthenticationError("Experiment 009 selector receipts are missing")
    expected_selector_receipts = {
        "authenticated": True,
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "selector_canonical_evidence_sha256": SELECTOR_CANONICAL_SHA256,
        "loss_selector_canonical_evidence_sha256": LOSS_SELECTOR_CANONICAL_SHA256,
        "quota_sum": 1976,
    }
    for key, expected in expected_selector_receipts.items():
        if selectors.get(key) != expected:
            raise StageAAuthenticationError(f"Experiment 009 selector receipt drifted: {key}")
    return {
        "artifact": dict(artifact),
        "evidence": dict(evidence),
        "dataset_manifest": dict(manifest),
        "token_manifest": [dict(item) for item in token_manifest],
        "task_identity": dict(identity),
        "selector_receipts": dict(selectors),
    }


def authenticate_experiment009_anchor(repo_root: Path) -> dict[str, object]:
    path = repo_root / EXPERIMENT009_STAGE_A_RELATIVE_PATH
    if _file_sha256(path) != EXPERIMENT009_STAGE_A_FILE_SHA256:
        raise StageAAuthenticationError("committed Experiment 009 Stage-A file hash drifted")
    verification = verify_evidence_artifact(path)
    if verification.get("valid") is not True:
        raise StageAAuthenticationError(
            "committed Experiment 009 Stage-A artifact failed canonical verification"
        )
    if (
        verification.get("file_sha256") != EXPERIMENT009_STAGE_A_FILE_SHA256
        or verification.get("computed_canonical_evidence_sha256")
        != EXPERIMENT009_STAGE_A_CANONICAL_SHA256
    ):
        raise StageAAuthenticationError("Experiment 009 verification receipt drifted")
    result = _validate_anchor_payload(_json_mapping(path))
    result["path"] = EXPERIMENT009_STAGE_A_RELATIVE_PATH
    result["file_sha256"] = EXPERIMENT009_STAGE_A_FILE_SHA256
    result["canonical_evidence_sha256"] = EXPERIMENT009_STAGE_A_CANONICAL_SHA256
    return result


def authenticate_stage_a_identity_clarification(
    repo_root: Path,
) -> dict[str, object]:
    path = repo_root / IDENTITY_NOTE_RELATIVE_PATH
    protocol_path = repo_root / PROTOCOL_NOTE_RELATIVE_PATH
    if not path.is_file() or _file_sha256(path) != IDENTITY_NOTE_FILE_SHA256:
        raise StageAAuthenticationError(
            "Experiment 011 Stage-A identity clarification hash drifted"
        )
    if not protocol_path.is_file() or _file_sha256(protocol_path) != PROTOCOL_NOTE_FILE_SHA256:
        raise StageAAuthenticationError("Experiment 011 StateLease protocol hash drifted")
    return {
        "path": IDENTITY_NOTE_RELATIVE_PATH,
        "file_sha256": IDENTITY_NOTE_FILE_SHA256,
        "protocol_path": PROTOCOL_NOTE_RELATIVE_PATH,
        "protocol_file_sha256": PROTOCOL_NOTE_FILE_SHA256,
        "task_id": TASK_ID,
        "row_sha256": TASK_ROW_SHA256,
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "code_text_sha256": CODE_TEXT_SHA256,
        "prompt_token_ids_sha256": PROMPT_TOKEN_IDS_SHA256,
        "code_token_ids_sha256": CODE_TOKEN_IDS_SHA256,
        "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
        "identity_only": True,
        "model_weights_or_quality_accessed": False,
    }


def authenticate_selectors(
    repo_root: Path,
    anchor: Mapping[str, object],
) -> tuple[dict[str, object], object]:
    selector_path = repo_root / SELECTOR_RELATIVE_PATH
    loss_selector_path = repo_root / LOSS_SELECTOR_RELATIVE_PATH
    actual_hashes = {
        "selector_file_sha256": _file_sha256(selector_path),
        "loss_selector_file_sha256": _file_sha256(loss_selector_path),
    }
    expected_hashes = {
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise StageAAuthenticationError("frozen selector artifact file hashes drifted")
    screen009 = _script_module("screen_rht_cqer")
    try:
        _selector, _loss_selector, hashes, plan = screen009._authenticate_selectors(  # noqa: SLF001
            selector_path,
            loss_selector_path,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise StageAAuthenticationError(
            f"selector semantic authentication failed: {error}"
        ) from error
    expected = {
        **expected_hashes,
        "selector_canonical_evidence_sha256": SELECTOR_CANONICAL_SHA256,
        "loss_selector_canonical_evidence_sha256": LOSS_SELECTOR_CANONICAL_SHA256,
    }
    if hashes != expected:
        raise StageAAuthenticationError("selector canonical identities drifted")
    receipts = anchor.get("selector_receipts")
    if not isinstance(receipts, Mapping) or any(
        receipts.get(key) != value for key, value in expected.items()
    ):
        raise StageAAuthenticationError(
            "selector identities do not match the committed task-666 anchor"
        )
    identity = {
        **expected,
        "selector_path": SELECTOR_RELATIVE_PATH,
        "loss_selector_path": LOSS_SELECTOR_RELATIVE_PATH,
        "target_fisher_score": receipts.get("target_fisher_score"),
        "target_fisher_layer_quotas": receipts.get("target_fisher_layer_quotas"),
        "quota_sum": receipts.get("quota_sum"),
        "authenticated": True,
    }
    return identity, plan


def authenticate_stage0(
    artifact_path: Path,
    sha256_path: Path | None,
    *,
    expected_repo_head: str | None = None,
) -> tuple[dict[str, object], Path, Path]:
    artifact = artifact_path.resolve()
    sidecar = (
        artifact.with_suffix(artifact.suffix + ".sha256")
        if sha256_path is None
        else sha256_path.resolve()
    )
    verifier = _script_module("verify_statelease_stage0")
    try:
        report = verifier.verify_production_stage0(
            artifact,
            sha256_path=sidecar,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise StageAAuthenticationError(
            f"production Stage 0 failed independent verification: {error}"
        ) from error
    required = {
        "status": "production_stage0_pass",
        "experiment_stage0_complete": True,
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
    }
    if not isinstance(report, dict) or any(
        report.get(key) != value for key, value in required.items()
    ):
        raise StageAAuthenticationError(
            "independent verifier did not authenticate a complete synthetic-only Stage 0"
        )
    if expected_repo_head is not None:
        artifact_head = report.get("repository_commit")
        if artifact_head != expected_repo_head:
            raise StageAAuthenticationError(
                "production Stage-0 repository HEAD does not equal the current Stage-A HEAD"
            )
    report = dict(report)
    report["artifact"] = artifact.name
    report["sidecar"] = sidecar.name
    report["sidecar_file_sha256"] = _file_sha256(sidecar)
    return report, artifact, sidecar


def authenticate_static_inputs(args: argparse.Namespace) -> StageAPreflight:
    repo_root = Path(__file__).resolve().parents[1]
    repository_start, source_hashes_start = _assert_repository_start(repo_root)
    output_path, attempt_path = _assert_ignored_exact_output(repo_root)
    identity_clarification = authenticate_stage_a_identity_clarification(repo_root)
    experiment010_admin_null = authenticate_experiment010_administrative_null(repo_root)
    anchor = authenticate_experiment009_anchor(repo_root)
    selector_identity, plan = authenticate_selectors(repo_root, anchor)
    stage0, stage0_artifact, stage0_sha256 = authenticate_stage0(
        args.stage0_artifact,
        args.stage0_sha256,
        expected_repo_head=str(repository_start["commit"]),
    )
    _assert_loaded_local_modules_declared(repo_root)
    return StageAPreflight(
        repo_root=repo_root,
        repository_start=repository_start,
        source_hashes_start=source_hashes_start,
        identity_clarification=identity_clarification,
        experiment010_admin_null=experiment010_admin_null,
        anchor=anchor,
        selector_identity=selector_identity,
        plan=plan,
        stage0=stage0,
        stage0_artifact=stage0_artifact,
        stage0_sha256=stage0_sha256,
        output_path=output_path,
        attempt_path=attempt_path,
    )


def _validate_model_geometry(config: object) -> dict[str, object]:
    getter = getattr(config, "get_text_config", None)
    text_config = getter(decoder=True) if callable(getter) else config
    layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        raise StageAAuthenticationError("pinned Qwen configuration lacks layer_types")
    linear_indices = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    ]
    geometry = {
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "recurrent_layers": len(linear_indices),
        "linear_layer_indices": linear_indices,
        "value_heads": getattr(text_config, "linear_num_value_heads", None),
        "key_rows": getattr(text_config, "linear_key_head_dim", None),
        "value_width": getattr(text_config, "linear_value_head_dim", None),
        "batch_size": 1,
        "recurrent_state_dtype": "torch.float32",
    }
    if geometry != EXPECTED_GEOMETRY:
        raise StageAAuthenticationError(f"pinned Qwen recurrent geometry drifted: {geometry}")
    return geometry


def _runtime_dependency_versions() -> dict[str, str]:
    actual: dict[str, str] = {}
    for package, expected in EXPECTED_RUNTIME_PACKAGES.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise StageAAuthenticationError(
                f"required runtime dependency is missing: {package}"
            ) from error
        if version != expected:
            raise StageAAuthenticationError(
                f"runtime dependency version drifted for {package}: "
                f"expected {expected}, received {version}"
            )
        try:
            importlib.import_module(RUNTIME_PACKAGE_IMPORTS[package])
        except (ImportError, RuntimeError) as error:
            raise StageAAuthenticationError(
                f"required runtime dependency could not be imported: {package}"
            ) from error
        actual[package] = version
    manifest_bytes = (json.dumps(actual, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != (EXPECTED_RUNTIME_PACKAGE_MANIFEST_SHA256):
        raise StageAAuthenticationError("runtime dependency compact-manifest SHA-256 drifted")
    return actual


def _resolve_cached_model_resource(filename: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        MODEL_ID,
        filename,
        revision=MODEL_REVISION,
    )
    if not isinstance(cached, str):
        raise StageAAuthenticationError(
            f"pinned model resource is absent from the local cache: {filename}"
        )
    return Path(cached)


def _is_recognized_model_weight_filename(filename: str) -> bool:
    normalized = filename.casefold()
    allowed = {item.casefold() for item in ALLOWED_MODEL_WEIGHT_FILENAMES}
    alternates = {item.casefold() for item in ALTERNATE_MODEL_WEIGHT_FILENAMES}
    return (
        normalized in allowed
        or normalized in alternates
        or normalized.endswith((".safetensors", ".bin", ".h5", ".msgpack"))
        or normalized.endswith((".safetensors.index.json", ".bin.index.json"))
        or normalized.startswith("model.ckpt")
    )


def _is_unpinned_tokenizer_resource(filename: str) -> bool:
    normalized = filename.casefold()
    alternates = {item.casefold() for item in ALTERNATE_TOKENIZER_RESOURCE_FILENAMES}
    return (
        normalized in alternates
        or normalized.endswith(".model")
        or normalized.startswith("chat_template")
    )


def _authenticate_model_snapshot() -> tuple[Path, dict[str, object]]:
    resources: dict[str, dict[str, object]] = {}
    snapshot: Path | None = None
    for filename, expected in MODEL_CACHE_RESOURCES.items():
        cached = _resolve_cached_model_resource(filename).absolute()
        if cached.name != filename or not cached.is_file():
            raise StageAAuthenticationError(
                f"pinned model cache identity is missing or malformed: {filename}"
            )
        if snapshot is None:
            snapshot = cached.parent
        elif cached.parent != snapshot:
            raise StageAAuthenticationError(
                "pinned model resources do not share one authenticated snapshot"
            )
        try:
            size = cached.stat().st_size
        except OSError as error:
            raise StageAAuthenticationError(
                f"cannot stat pinned model cache resource: {filename}"
            ) from error
        if size != expected["size_bytes"]:
            raise StageAAuthenticationError(f"pinned model cache resource size drifted: {filename}")
        digest = _file_sha256(cached)
        if digest != expected["sha256"]:
            raise StageAAuthenticationError(
                f"pinned model cache resource SHA-256 drifted: {filename}"
            )
        resources[filename] = {
            "resource_id": (f"{MODEL_ID}@{MODEL_REVISION}/{filename}"),
            "size_bytes": size,
            "sha256": digest,
            "bytes_hashed_without_parsing_or_tensor_loading": True,
        }
    if snapshot is None or not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
        raise StageAAuthenticationError("pinned model snapshot revision identity drifted")

    recognized = {
        path.name for path in snapshot.iterdir() if _is_recognized_model_weight_filename(path.name)
    }
    if recognized != set(ALLOWED_MODEL_WEIGHT_FILENAMES):
        extras = sorted(recognized - set(ALLOWED_MODEL_WEIGHT_FILENAMES))
        missing = sorted(set(ALLOWED_MODEL_WEIGHT_FILENAMES) - recognized)
        raise StageAAuthenticationError(
            "pinned model snapshot weight file set drifted "
            f"(missing={missing}, alternate_or_extra={extras})"
        )
    unpinned_tokenizer_resources = sorted(
        path.name for path in snapshot.iterdir() if _is_unpinned_tokenizer_resource(path.name)
    )
    if unpinned_tokenizer_resources:
        raise StageAAuthenticationError(
            "pinned model snapshot contains unpinned tokenizer-affecting resources"
        )

    index_path = snapshot / MODEL_WEIGHT_INDEX_FILENAME
    index = _json_mapping(index_path)
    if set(index) != {"metadata", "weight_map"}:
        raise StageAAuthenticationError("pinned model weight index schema drifted")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("total_size") != MODEL_WEIGHT_INDEX_TENSOR_BYTES
        or not isinstance(weight_map, Mapping)
        or not weight_map
        or any(
            not isinstance(parameter, str) or shard != MODEL_WEIGHT_RESOURCE_FILENAME
            for parameter, shard in weight_map.items()
        )
    ):
        raise StageAAuthenticationError("pinned model weight index shard binding drifted")

    resource_manifest_sha256 = _canonical_sha256(resources)
    receipt = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_resource_id": f"{MODEL_ID}@{MODEL_REVISION}",
        "resources": resources,
        "resource_manifest_sha256": resource_manifest_sha256,
        "weight_index_filename": MODEL_WEIGHT_INDEX_FILENAME,
        "weight_index_tensor_bytes": MODEL_WEIGHT_INDEX_TENSOR_BYTES,
        "weight_shards": [MODEL_WEIGHT_RESOURCE_FILENAME],
        "recognized_weight_files": sorted(recognized),
        "alternate_or_unpinned_weight_files_absent": True,
        "alternate_or_unpinned_tokenizer_resources_absent": True,
        "loader_must_use_authenticated_snapshot_directory": True,
        "local_cache_only": True,
        "config_or_tokenizer_parsed": False,
        "tensors_loaded": False,
    }
    return snapshot, receipt


def _authenticate_cached_model_resources() -> dict[str, object]:
    _snapshot, receipt = _authenticate_model_snapshot()
    return receipt


def _resolve_cached_dataset_revision(datasets_module: object) -> Path:
    config = getattr(datasets_module, "config", None)
    cache_root = getattr(config, "HF_DATASETS_CACHE", None)
    if not isinstance(cache_root, (str, os.PathLike)):
        raise StageAAuthenticationError(
            "datasets cache root is unavailable from the authenticated runtime"
        )
    return Path(cache_root) / Path(DATASET_CACHE_REVISION_RELATIVE_ID)


def _authenticate_cached_dataset_resources(
    datasets_module: object,
) -> dict[str, object]:
    revision = _resolve_cached_dataset_revision(datasets_module)
    if not revision.is_dir():
        raise StageAAuthenticationError(
            "pinned MBPP revision is absent from the local datasets cache"
        )
    resources: dict[str, dict[str, object]] = {}
    for filename, expected in DATASET_CACHE_RESOURCES.items():
        path = revision / filename
        if not path.is_file():
            raise StageAAuthenticationError(f"pinned MBPP cache resource is missing: {filename}")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise StageAAuthenticationError(
                f"cannot stat pinned MBPP cache resource: {filename}"
            ) from error
        if size != expected["size_bytes"]:
            raise StageAAuthenticationError(f"pinned MBPP cache resource size drifted: {filename}")
        digest = _file_sha256(path)
        if digest != expected["sha256"]:
            raise StageAAuthenticationError(
                f"pinned MBPP cache resource SHA-256 drifted: {filename}"
            )
        resources[filename] = {
            "resource_id": f"{DATASET_CACHE_REVISION_RELATIVE_ID}/{filename}",
            "size_bytes": size,
            "sha256": digest,
            "bytes_hashed_without_decoding_or_iteration": True,
        }
    return {
        "dataset_id": EXPECTED_DATASET_MANIFEST["dataset_id"],
        "revision": EXPECTED_DATASET_MANIFEST["revision"],
        "config": EXPECTED_DATASET_MANIFEST["config"],
        "cache_revision_resource_id": DATASET_CACHE_REVISION_RELATIVE_ID,
        "resources": resources,
        "local_cache_only": True,
        "dataset_rows_decoded_or_iterated": False,
    }


def authenticate_runtime_readiness(*, device_name: str) -> RuntimeReadiness:
    if device_name not in {"auto", "cuda"}:
        raise StageAAuthenticationError("Experiment 011 supports only the pinned CUDA BF16 path")
    packages = _runtime_dependency_versions()
    try:
        datasets_module = importlib.import_module("datasets")
    except (ImportError, RuntimeError) as error:
        raise StageAAuthenticationError(
            "the exact datasets runtime could not be imported"
        ) from error
    if not callable(getattr(datasets_module, "load_dataset", None)):
        raise StageAAuthenticationError("datasets.load_dataset is unavailable or non-callable")
    if not torch.cuda.is_available():
        raise StageAAuthenticationError(
            "the frozen Experiment 011 Stage-A CUDA device is unavailable"
        )
    visible_device_count = int(torch.cuda.device_count())
    if visible_device_count < 1:
        raise StageAAuthenticationError(
            "the frozen Experiment 011 Stage-A runtime has no visible CUDA device"
        )
    bf16_check = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(bf16_check) or bf16_check() is not True:
        raise StageAAuthenticationError(
            "the frozen Experiment 011 Stage-A CUDA device lacks BF16 support"
        )
    dataset_resources = _authenticate_cached_dataset_resources(datasets_module)
    model_resources = _authenticate_cached_model_resources()
    receipt: dict[str, object] = {
        "schema": "recurquant.experiment011.runtime-readiness.v1",
        "packages": packages,
        "package_manifest_sha256": EXPECTED_RUNTIME_PACKAGE_MANIFEST_SHA256,
        "package_imports": dict(RUNTIME_PACKAGE_IMPORTS),
        "datasets_api": {
            "module_imported": True,
            "load_dataset_callable": True,
            "load_dataset_called": False,
        },
        "accelerator": {
            "requested_device": device_name,
            "resolved_device": "cuda",
            "cuda_available": True,
            "visible_device_count": visible_device_count,
            "bf16_supported": True,
            "model_dtype": str(MODEL_DTYPE),
            "torch_cuda_runtime": torch.version.cuda,
        },
        "model_cache_resources": model_resources,
        "dataset_cache_resources": dataset_resources,
        "authenticated_before_one_run_seal": True,
        "authenticated_before_task_tokenizer_or_model_weights": True,
    }
    return RuntimeReadiness(
        receipt=receipt,
        canonical_sha256=_canonical_sha256(receipt),
    )


def _readiness_bundle(readiness: RuntimeReadiness) -> dict[str, object]:
    if _canonical_sha256(readiness.receipt) != readiness.canonical_sha256:
        raise StageAAuthenticationError("runtime readiness receipt canonical hash drifted")
    return {
        "receipt": readiness.receipt,
        "canonical_sha256": readiness.canonical_sha256,
    }


def _runtime_readiness_from_bundle(value: object) -> RuntimeReadiness:
    if not isinstance(value, Mapping):
        raise StageAAuthenticationError("runtime readiness bundle is malformed")
    receipt = value.get("receipt")
    canonical_sha256 = value.get("canonical_sha256")
    if not isinstance(receipt, Mapping) or not isinstance(canonical_sha256, str):
        raise StageAAuthenticationError("runtime readiness receipt is malformed")
    readiness = RuntimeReadiness(
        receipt=dict(receipt),
        canonical_sha256=canonical_sha256,
    )
    _readiness_bundle(readiness)
    return readiness


def load_and_authenticate_config(
    preflight: StageAPreflight,
    *,
    local_files_only: bool,
) -> ModelConfiguration:
    del preflight
    if local_files_only is not True:
        raise StageAAuthenticationError(
            "Experiment 011 configuration authentication requires --local-files-only"
        )
    from transformers import AutoConfig

    snapshot, cache_receipt = _authenticate_model_snapshot()
    config = AutoConfig.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    geometry = _validate_model_geometry(config)
    _assert_loaded_local_modules_declared(Path(__file__).resolve().parents[1])
    return ModelConfiguration(
        config=config,
        identity={
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": str(MODEL_DTYPE),
            "attn_implementation": "eager",
            "geometry": geometry,
            "authenticated_snapshot_resource_id": cache_receipt["snapshot_resource_id"],
            "model_cache_resource_manifest_sha256": cache_receipt["resource_manifest_sha256"],
            "configuration_loaded_from_authenticated_snapshot": True,
            "configuration_authenticated_before_task_or_model_weights": True,
        },
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise StageAAuthenticationError(
            f"refusing to overwrite one-run artifact {path.name}"
        ) from error


def _atomic_replace_owned(path: Path, payload: bytes) -> None:
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


def _commit_tree(
    repo_root: Path,
    *,
    tree: str,
    parent: str,
    message: str,
) -> str:
    _assert_git_repository_identity(repo_root)
    parent_payload = _git(repo_root, "cat-file", "commit", parent)
    author_headers = [
        line.removeprefix("author ")
        for line in parent_payload.splitlines()
        if line.startswith("author ")
    ]
    if len(author_headers) != 1:
        raise StageAAuthenticationError(
            "parent commit does not contain one authenticated author identity"
        )
    author = author_headers[0]
    email_start = author.rfind(" <")
    email_end = author.find("> ", email_start + 2)
    if email_start <= 0 or email_end <= email_start + 2:
        raise StageAAuthenticationError("parent commit author identity is malformed")
    name = author[:email_start]
    email = author[email_start + 2 : email_end]
    if not name or not email or "\n" in name or "\n" in email:
        raise StageAAuthenticationError(
            "authenticated parent author identity is required for the one-run seal"
        )
    identity = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }
    process = _run_git_process(
        repo_root,
        "commit-tree",
        tree,
        "-p",
        parent,
        input_text=f"{message}\n",
        environment_overrides=identity,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"cannot create one-run seal commit: {detail}")
    commit = process.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise StageAAuthenticationError("one-run seal commit identity is malformed")
    _assert_git_repository_identity(repo_root)
    return commit


def _seal_message(
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
    readiness: RuntimeReadiness,
    preseal_freshness: Mapping[str, object],
) -> str:
    initial_access_ledger = _expected_access_snapshot("reserved_before_task_entry")
    payload = {
        "schema": "recurquant.experiment011.stage-a-one-run-seal.v1",
        "marker": ONE_RUN_MARKER,
        "h0_commit": preflight.repository_start["commit"],
        "source_set_sha256": _canonical_sha256(preflight.source_hashes_start),
        "identity_note_file_sha256": IDENTITY_NOTE_FILE_SHA256,
        "protocol_note_file_sha256": PROTOCOL_NOTE_FILE_SHA256,
        "experiment010_administrative_null": preflight.experiment010_admin_null,
        "experiment009_file_sha256": EXPERIMENT009_STAGE_A_FILE_SHA256,
        "stage0_artifact_file_sha256": preflight.stage0.get("artifact_file_sha256"),
        "stage0_sidecar_file_sha256": preflight.stage0.get("sidecar_file_sha256"),
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "runtime_readiness": _readiness_bundle(readiness),
        "preseal_freshness": dict(preseal_freshness),
        "model": configuration.identity,
        "task_id": TASK_ID,
        "attempt_number": 1,
        "attempt_path": ATTEMPT_RELATIVE_PATH,
        "output_path": OUTPUT_RELATIVE_PATH,
        "claim_boundary": CLAIM_BOUNDARY,
        "postseal_receipt_limitation": POSTSEAL_RECEIPT_LIMITATION,
        "completed_task_ids": [],
        "quality_data_accessed": False,
        "task_row_loaded": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "evaluation_entered": False,
        "evaluation_returned": False,
        "forward_passes": 0,
        "forward_passes_minimum": 0,
        "quality_result_computed": False,
        "access_ledger": initial_access_ledger,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        "quality_data_accessed_before_seal": False,
        "model_weights_loaded_before_seal": False,
    }
    payload_bytes = canonical_json_bytes(payload)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    return (
        "chore: reserve Experiment 011 Stage-A one-run\n\n"
        f"{ONE_RUN_MARKER}\n"
        f"RecurQuant-One-Run-Payload-SHA256: {payload_hash}\n\n"
        f"{payload_bytes.decode('utf-8')}"
    ).rstrip("\r\n")


def _validated_preseal_freshness(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != PRESEAL_FRESHNESS_KEYS
        or any(item is not True for item in value.values())
    ):
        raise StageAAuthenticationError("pre-seal freshness receipt is incomplete or malformed")
    return dict(value)


def _validate_one_run_seal(
    preflight: StageAPreflight,
    reservation: AttemptReservation,
    *,
    require_receipt: bool,
) -> dict[str, object]:
    state = _repository_state(preflight.repo_root)
    start_git_identity = preflight.repository_start.get("git_identity")
    if not isinstance(start_git_identity, Mapping) or state.get("git_identity") != dict(
        start_git_identity
    ):
        raise StageAAuthenticationError("Git repository/object identity changed after H0")
    if state["worktree_clean"] is not True:
        raise StageAAuthenticationError("repository became dirty after one-run sealing")
    if state["commit"] != reservation.seal_commit:
        raise StageAAuthenticationError("repository no longer points at the one-run seal")
    if _git(preflight.repo_root, "show", "-s", "--format=%P", reservation.seal_commit) != (
        reservation.h0_commit
    ):
        raise StageAAuthenticationError("one-run seal parent drifted")
    actual_tree = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%T",
        reservation.seal_commit,
    )
    h0_tree = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%T",
        reservation.h0_commit,
    )
    if actual_tree != reservation.tree or h0_tree != reservation.tree:
        raise StageAAuthenticationError("one-run seal is not an empty-tree commit")
    message = _git(
        preflight.repo_root,
        "show",
        "-s",
        "--format=%B",
        reservation.seal_commit,
    )
    model_identity = reservation.receipt.get("model")
    if not isinstance(model_identity, Mapping):
        raise StageAAuthenticationError("one-run receipt model identity is malformed")
    expected_receipt_identity = {
        "schema": "recurquant.experiment011.stage-a-attempt.v1",
        "attempt_number": 1,
        "task_id": TASK_ID,
        "output_path": OUTPUT_RELATIVE_PATH,
        "attempt_path": ATTEMPT_RELATIVE_PATH,
        "h0_repository_commit": reservation.h0_commit,
        "one_run_seal_commit": reservation.seal_commit,
        "one_run_seal_tree": reservation.tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": reservation.seal_message_sha256,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if any(
        reservation.receipt.get(key) != value for key, value in expected_receipt_identity.items()
    ):
        raise StageAAuthenticationError(
            "promoted one-run receipt identity or access boundary drifted"
        )
    status = reservation.receipt.get("status")
    if status == "reserved_before_quality_data_or_model_weights":
        access_ledger = _validated_access_snapshot(reservation.receipt.get("access_ledger"))
        expected_access = {
            "completed_task_ids": [],
            "quality_data_accessed": False,
            "task_row_loaded": False,
            "tokenizer_loaded": False,
            "model_weights_loaded": False,
            "evaluation_entered": False,
            "evaluation_returned": False,
            "forward_passes": 0,
            "forward_passes_minimum": 0,
            "quality_result_computed": False,
        }
        if access_ledger != _expected_access_snapshot("reserved_before_task_entry") or any(
            reservation.receipt.get(key) != value for key, value in expected_access.items()
        ):
            raise StageAAuthenticationError("reserved receipt zero-access boundary drifted")
    elif status in {
        "running_with_monotonic_access_ledger",
        "evaluation_returned_before_artifact_finalization",
    }:
        ledger = _validated_access_snapshot(reservation.receipt.get("access_ledger"))
        evaluation_returned = ledger.get("evaluation_returned") is True
        expected_access = {
            "completed_task_ids": [TASK_ID] if evaluation_returned else [],
            "quality_data_accessed": ledger.get("task_load_entered") is True,
            "task_row_loaded": ledger.get("task_row_loaded"),
            "tokenizer_loaded": ledger.get("tokenizer_loaded"),
            "model_weights_loaded": ledger.get("model_weights_loaded"),
            "evaluation_entered": ledger.get("evaluation_entered"),
            "evaluation_returned": ledger.get("evaluation_returned"),
            "forward_passes": ledger.get("forward_passes"),
            "forward_passes_minimum": ledger.get("forward_passes_minimum"),
            "quality_result_computed": ledger.get("quality_result_computed"),
        }
        if any(reservation.receipt.get(key) != value for key, value in expected_access.items()):
            raise StageAAuthenticationError("running receipt access ledger reconciliation failed")
        if (status == "evaluation_returned_before_artifact_finalization") != evaluation_returned:
            raise StageAAuthenticationError("running receipt phase/status reconciliation failed")
    else:
        raise StageAAuthenticationError("one-run receipt status is not valid for active execution")
    if (
        reservation.receipt.get("experiment010_administrative_null")
        != preflight.experiment010_admin_null
    ):
        raise StageAAuthenticationError(
            "one-run receipt Experiment 010 administrative-null provenance drifted"
        )
    readiness = _runtime_readiness_from_bundle(reservation.receipt.get("runtime_readiness"))
    preseal_freshness = _validated_preseal_freshness(reservation.receipt.get("preseal_freshness"))
    expected_message = _seal_message(
        preflight,
        ModelConfiguration(config=None, identity=dict(model_identity)),
        readiness,
        preseal_freshness,
    )
    if (
        message != expected_message
        or ONE_RUN_MARKER not in message
        or hashlib.sha256(message.encode("utf-8")).hexdigest() != reservation.seal_message_sha256
    ):
        raise StageAAuthenticationError("one-run seal message drifted")
    _assert_sources_match_head(preflight.repo_root)
    if _source_hashes(preflight.repo_root) != preflight.source_hashes_start:
        raise StageAAuthenticationError("source hashes changed after one-run sealing")
    if require_receipt:
        try:
            payload = preflight.attempt_path.read_bytes()
        except OSError as error:
            raise StageAAuthenticationError("reserved attempt receipt is missing") from error
        if hashlib.sha256(payload).hexdigest() != reservation.receipt_file_sha256:
            raise StageAAuthenticationError("reserved attempt receipt bytes drifted")
        if payload != canonical_json_bytes(reservation.receipt):
            raise StageAAuthenticationError(
                "reserved attempt receipt object differs from its persisted bytes"
            )
    return {
        "h0_commit": reservation.h0_commit,
        "seal_commit": reservation.seal_commit,
        "tree": reservation.tree,
        "seal_message_sha256": reservation.seal_message_sha256,
        "empty_tree_commit": True,
        "durable_one_run_marker": ONE_RUN_MARKER,
        "local_seal_limitation": ONE_RUN_LIMITATION,
    }


def _assert_preseal_freshness(
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
    readiness: RuntimeReadiness,
) -> dict[str, object]:
    h0_commit = str(preflight.repository_start["commit"])
    state = _repository_state(preflight.repo_root)
    start_git_identity = preflight.repository_start.get("git_identity")
    if state["worktree_clean"] is not True or state["commit"] != h0_commit:
        raise StageAAuthenticationError("repository identity changed before Experiment 011 sealing")
    if not isinstance(start_git_identity, Mapping) or state.get("git_identity") != dict(
        start_git_identity
    ):
        raise StageAAuthenticationError("Git repository/object identity changed before sealing")
    _assert_sources_match_head(preflight.repo_root)
    _assert_no_prior_stage_a_seal(preflight.repo_root)
    if _source_hashes(preflight.repo_root) != preflight.source_hashes_start:
        raise StageAAuthenticationError(
            "authenticated source hashes changed before Experiment 011 sealing"
        )

    stage0_now, artifact_now, sidecar_now = authenticate_stage0(
        preflight.stage0_artifact,
        preflight.stage0_sha256,
        expected_repo_head=h0_commit,
    )
    if (
        stage0_now != preflight.stage0
        or artifact_now != preflight.stage0_artifact
        or sidecar_now != preflight.stage0_sha256
    ):
        raise StageAAuthenticationError(
            "production Stage-0 identity changed before Experiment 011 sealing"
        )
    admin_now = authenticate_experiment010_administrative_null(preflight.repo_root)
    if admin_now != preflight.experiment010_admin_null:
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null provenance changed before sealing"
        )

    accelerator = readiness.receipt.get("accelerator")
    if not isinstance(accelerator, Mapping):
        raise StageAAuthenticationError("runtime readiness accelerator receipt is malformed")
    requested_device = accelerator.get("requested_device")
    if not isinstance(requested_device, str):
        raise StageAAuthenticationError("runtime readiness requested device is malformed")
    readiness_now = authenticate_runtime_readiness(device_name=requested_device)
    if _readiness_bundle(readiness_now) != _readiness_bundle(readiness):
        raise StageAAuthenticationError("runtime dependency/cache readiness changed before sealing")
    configuration_now = load_and_authenticate_config(
        preflight,
        local_files_only=True,
    )
    if configuration_now.identity != configuration.identity:
        raise StageAAuthenticationError("pinned local model configuration changed before sealing")

    final_state = _repository_state(preflight.repo_root)
    if final_state["worktree_clean"] is not True or final_state["commit"] != h0_commit:
        raise StageAAuthenticationError(
            "repository identity changed during final pre-seal authentication"
        )
    if final_state.get("git_identity") != dict(start_git_identity):
        raise StageAAuthenticationError(
            "Git repository/object identity changed during final pre-seal authentication"
        )
    _assert_sources_match_head(preflight.repo_root)
    if _source_hashes(preflight.repo_root) != preflight.source_hashes_start:
        raise StageAAuthenticationError("source bytes changed during final pre-seal authentication")
    _assert_no_prior_stage_a_seal(preflight.repo_root)
    return {
        "repository_reauthenticated": True,
        "source_head_blobs_reauthenticated": True,
        "new_marker_absent": True,
        "stage0_reauthenticated": True,
        "experiment010_administrative_null_reauthenticated": True,
        "runtime_readiness_reauthenticated": True,
        "configuration_reauthenticated_local_only": True,
    }


def _promote_prepared_attempt_receipt(
    path: Path,
    prepared: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    expected_prepared_bytes = canonical_json_bytes(prepared)
    try:
        actual_prepared_bytes = path.read_bytes()
    except OSError as error:
        raise StageAAuthenticationError(
            "prepared one-run receipt is missing before status promotion"
        ) from error
    if actual_prepared_bytes != expected_prepared_bytes:
        raise StageAAuthenticationError(
            "prepared one-run receipt bytes drifted before status promotion"
        )
    promoted = {
        **prepared,
        "status": "reserved_before_quality_data_or_model_weights",
    }
    payload = canonical_json_bytes(promoted)
    _atomic_replace_owned(path, payload)
    return promoted, payload


def _authenticate_prepared_attempt_receipt(
    path: Path,
    prepared: Mapping[str, object],
    *,
    h0_commit: str,
    seal_commit: str,
    tree: str,
    seal_message_sha256: str,
) -> str:
    initial_access_ledger = _expected_access_snapshot("reserved_before_task_entry")
    expected_identity = {
        "schema": "recurquant.experiment011.stage-a-attempt.v1",
        "status": "prepared_before_head_cas",
        "attempt_number": 1,
        "task_id": TASK_ID,
        "output_path": OUTPUT_RELATIVE_PATH,
        "attempt_path": ATTEMPT_RELATIVE_PATH,
        "h0_repository_commit": h0_commit,
        "one_run_seal_commit": seal_commit,
        "one_run_seal_tree": tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": seal_message_sha256,
        "completed_task_ids": [],
        "quality_data_accessed": False,
        "task_row_loaded": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "evaluation_entered": False,
        "evaluation_returned": False,
        "forward_passes": 0,
        "forward_passes_minimum": 0,
        "quality_result_computed": False,
        "access_ledger": initial_access_ledger,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if any(prepared.get(key) != value for key, value in expected_identity.items()):
        raise StageAAuthenticationError(
            "prepared one-run receipt identity or zero-result boundary is malformed"
        )
    _runtime_readiness_from_bundle(prepared.get("runtime_readiness"))
    _validated_preseal_freshness(prepared.get("preseal_freshness"))
    expected_bytes = canonical_json_bytes(prepared)
    try:
        actual_bytes = path.read_bytes()
    except OSError as error:
        raise StageAAuthenticationError(
            "prepared one-run receipt is missing before HEAD compare-and-swap"
        ) from error
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    if actual_bytes != expected_bytes or hashlib.sha256(actual_bytes).hexdigest() != expected_hash:
        raise StageAAuthenticationError(
            "prepared one-run receipt bytes drifted before HEAD compare-and-swap"
        )
    parsed = _json_mapping(path)
    if parsed != dict(prepared) or canonical_json_bytes(parsed) != actual_bytes:
        raise StageAAuthenticationError(
            "prepared one-run receipt canonical JSON authentication failed"
        )
    return expected_hash


def reserve_one_run(
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
    readiness: RuntimeReadiness,
) -> AttemptReservation:
    h0_commit = str(preflight.repository_start["commit"])
    preseal_freshness = _assert_preseal_freshness(
        preflight,
        configuration,
        readiness,
    )
    preseal_freshness = _validated_preseal_freshness(preseal_freshness)
    if _git(preflight.repo_root, "rev-parse", "HEAD") != h0_commit:
        raise StageAAuthenticationError("repository HEAD changed before one-run sealing")
    tree = _git(preflight.repo_root, "show", "-s", "--format=%T", h0_commit)
    message = _seal_message(
        preflight,
        configuration,
        readiness,
        preseal_freshness,
    )
    seal_commit = _commit_tree(
        preflight.repo_root,
        tree=tree,
        parent=h0_commit,
        message=message,
    )
    seal_message_sha256 = hashlib.sha256(message.encode("utf-8")).hexdigest()
    initial_access_ledger = _expected_access_snapshot("reserved_before_task_entry")
    prepared = {
        "schema": "recurquant.experiment011.stage-a-attempt.v1",
        "status": "prepared_before_head_cas",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attempt_number": 1,
        "task_id": TASK_ID,
        "output_path": OUTPUT_RELATIVE_PATH,
        "attempt_path": ATTEMPT_RELATIVE_PATH,
        "h0_repository_commit": h0_commit,
        "one_run_seal_commit": seal_commit,
        "one_run_seal_tree": tree,
        "one_run_marker": ONE_RUN_MARKER,
        "one_run_seal_message_sha256": seal_message_sha256,
        "one_run_seal_limitation": ONE_RUN_LIMITATION,
        "postseal_receipt_limitation": POSTSEAL_RECEIPT_LIMITATION,
        "source_hashes": preflight.source_hashes_start,
        "experiment010_administrative_null": preflight.experiment010_admin_null,
        "experiment009_file_sha256": EXPERIMENT009_STAGE_A_FILE_SHA256,
        "stage0_artifact_file_sha256": preflight.stage0.get("artifact_file_sha256"),
        "stage0_sidecar_file_sha256": preflight.stage0.get("sidecar_file_sha256"),
        "selector_file_sha256": SELECTOR_FILE_SHA256,
        "loss_selector_file_sha256": LOSS_SELECTOR_FILE_SHA256,
        "runtime_readiness": _readiness_bundle(readiness),
        "preseal_freshness": preseal_freshness,
        "model": configuration.identity,
        "completed_task_ids": [],
        "quality_data_accessed": False,
        "task_row_loaded": False,
        "tokenizer_loaded": False,
        "model_weights_loaded": False,
        "evaluation_entered": False,
        "evaluation_returned": False,
        "forward_passes": 0,
        "forward_passes_minimum": 0,
        "quality_result_computed": False,
        "access_ledger": initial_access_ledger,
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    prepared_bytes = canonical_json_bytes(prepared)
    _exclusive_write(preflight.attempt_path, prepared_bytes)
    _authenticate_prepared_attempt_receipt(
        preflight.attempt_path,
        prepared,
        h0_commit=h0_commit,
        seal_commit=seal_commit,
        tree=tree,
        seal_message_sha256=seal_message_sha256,
    )
    _assert_git_repository_identity(preflight.repo_root)
    process = _run_git_process(
        preflight.repo_root,
        "update-ref",
        "HEAD",
        seal_commit,
        h0_commit,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise StageAAuthenticationError(f"one-run HEAD compare-and-swap failed: {detail}")
    _assert_git_repository_identity(preflight.repo_root)
    if _git(preflight.repo_root, "rev-parse", "HEAD") != seal_commit:
        raise StageAAuthenticationError("one-run HEAD compare-and-swap did not reach the seal")
    receipt, receipt_bytes = _promote_prepared_attempt_receipt(
        preflight.attempt_path,
        prepared,
    )
    reservation = AttemptReservation(
        receipt=receipt,
        receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        h0_commit=h0_commit,
        seal_commit=seal_commit,
        tree=tree,
        seal_message_sha256=seal_message_sha256,
    )
    _validate_one_run_seal(preflight, reservation, require_receipt=True)
    return reservation


def _privacy_safe_failure(error: BaseException) -> dict[str, object]:
    raw = str(error)
    raw_sha256 = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    raw_type = type(error).__name__
    safe_type = (
        raw_type
        if raw_type
        and len(raw_type) <= 80
        and all(character.isalnum() or character == "_" for character in raw_type)
        else "Exception"
    )
    if isinstance(error, StageAAuthenticationError):
        category = "authentication_or_integrity"
    elif isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        category = "memory_or_accelerator_capacity"
    elif isinstance(error, OSError):
        category = "filesystem_or_io"
    elif isinstance(error, (KeyboardInterrupt, SystemExit)):
        category = "interruption"
    else:
        category = "runtime"
    return {
        "failure_type": safe_type,
        "failure_category": category,
        "failure_summary": f"{category} failure; raw detail withheld",
        "failure_detail_sha256": raw_sha256,
        "failure_detail_recorded": False,
    }


def record_access_transition(
    preflight: StageAPreflight,
    receipt: object,
    ledger: AccessLedger,
) -> AttemptReservation:
    if not isinstance(receipt, AttemptReservation):
        raise TypeError("Stage-A attempt receipt is invalid")
    _validate_one_run_seal(preflight, receipt, require_receipt=True)
    previous_snapshot = _validated_access_snapshot(receipt.receipt.get("access_ledger"))
    snapshot = _validated_access_snapshot(ledger.snapshot())
    _assert_adjacent_access_transition(
        previous_snapshot,
        snapshot,
        allow_equal=False,
    )
    evaluation_returned = snapshot.get("evaluation_returned") is True
    updated = {
        **receipt.receipt,
        "status": (
            "evaluation_returned_before_artifact_finalization"
            if evaluation_returned
            else "running_with_monotonic_access_ledger"
        ),
        "access_ledger": snapshot,
        "quality_data_accessed": snapshot.get("task_load_entered") is True,
        "task_row_loaded": snapshot.get("task_row_loaded"),
        "tokenizer_loaded": snapshot.get("tokenizer_loaded"),
        "model_weights_loaded": snapshot.get("model_weights_loaded"),
        "evaluation_entered": snapshot.get("evaluation_entered"),
        "evaluation_returned": snapshot.get("evaluation_returned"),
        "forward_passes": snapshot.get("forward_passes"),
        "forward_passes_minimum": snapshot.get("forward_passes_minimum"),
        "quality_result_computed": snapshot.get("quality_result_computed"),
        "completed_task_ids": [TASK_ID] if evaluation_returned else [],
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
    }
    _validate_public_artifact(updated, repo_root=preflight.repo_root)
    payload = canonical_json_bytes(updated)
    _atomic_replace_owned(preflight.attempt_path, payload)
    if preflight.attempt_path.read_bytes() != payload:
        raise StageAAuthenticationError("monotonic access-ledger receipt persistence drifted")
    return dataclasses.replace(
        receipt,
        receipt=updated,
        receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def record_evaluation_returned(
    preflight: StageAPreflight,
    receipt: object,
    ledger: AccessLedger,
) -> AttemptReservation:
    if not isinstance(receipt, AttemptReservation):
        raise TypeError("Stage-A attempt receipt is invalid")
    snapshot = ledger.snapshot()
    if (
        snapshot.get("evaluation_returned") is not True
        or snapshot.get("task_row_loaded") is not True
        or snapshot.get("tokenizer_loaded") is not True
        or snapshot.get("model_weights_loaded") is not True
    ):
        raise StageAAuthenticationError("evaluation-returned access ledger is incomplete")
    return record_access_transition(preflight, receipt, ledger)


def record_attempt_failure(
    preflight: StageAPreflight,
    receipt: object,
    error: BaseException,
    ledger: AccessLedger | None = None,
) -> None:
    if not isinstance(receipt, AttemptReservation):
        return
    try:
        disk_payload = preflight.attempt_path.read_bytes()
        disk_receipt = _json_mapping(preflight.attempt_path)
    except (OSError, StageAAuthenticationError) as receipt_error:
        raise StageAAuthenticationError(
            "cannot authenticate the current attempt receipt while recording failure"
        ) from receipt_error
    status = disk_receipt.get("status")
    safe_failure = _privacy_safe_failure(error)

    completion_statuses = {
        RESULT_PREPARED_STATUS,
        RESULT_PUBLICATION_FAILED_STATUS,
    }
    if status in completion_statuses:
        if preflight.output_path.exists():
            publication = _validate_published_output_against_receipt(
                preflight.output_path,
                disk_receipt,
            )
            preserved = {
                **disk_receipt,
                "status": RESULT_PROMOTION_INTERRUPTED_STATUS,
                "receipt_promotion_interrupted": True,
                "result_available": True,
                "output_published": True,
                "quality_aggregate_exposed": True,
                "stage_a_gate_passed": publication["stage_a_gate_passed"],
                "rerun_automatically_authorized": False,
                **safe_failure,
            }
        else:
            preserved = {
                **disk_receipt,
                "status": RESULT_PUBLICATION_FAILED_STATUS,
                "result_available": False,
                "output_published": False,
                "quality_aggregate_exposed": False,
                "rerun_automatically_authorized": False,
                **safe_failure,
            }
        _validate_public_artifact(preserved, repo_root=preflight.repo_root)
        _atomic_replace_owned(
            preflight.attempt_path,
            canonical_json_bytes(preserved),
        )
        return
    if status in {
        RESULT_COMPLETED_STATUS,
        RESULT_PROMOTION_INTERRUPTED_STATUS,
    }:
        return
    if preflight.output_path.exists():
        raise StageAAuthenticationError(
            "existing Stage-A output has no authenticated completion binding; "
            "refusing to downgrade it to a no-result failure"
        )

    current = dataclasses.replace(
        receipt,
        receipt=disk_receipt,
        receipt_file_sha256=hashlib.sha256(disk_payload).hexdigest(),
    )
    _validate_one_run_seal(preflight, current, require_receipt=True)
    disk_snapshot = _validated_access_snapshot(disk_receipt.get("access_ledger"))
    if ledger is None:
        snapshot = disk_snapshot
    else:
        snapshot = _validated_access_snapshot(ledger.snapshot())
        _assert_adjacent_access_transition(
            disk_snapshot,
            snapshot,
            allow_equal=True,
        )
    evaluation_returned = snapshot.get("evaluation_returned") is True
    failed = {
        **disk_receipt,
        "status": (
            "failed_after_evaluation_without_authenticated_artifact"
            if evaluation_returned
            else "failed_without_authenticated_stage_a_result"
        ),
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "access_ledger": snapshot,
        "quality_data_accessed": snapshot.get("task_load_entered") is True,
        "task_row_loaded": snapshot.get("task_row_loaded"),
        "tokenizer_loaded": snapshot.get("tokenizer_loaded"),
        "model_weights_loaded": snapshot.get("model_weights_loaded"),
        "evaluation_entered": snapshot.get("evaluation_entered"),
        "evaluation_returned": snapshot.get("evaluation_returned"),
        "forward_passes": snapshot.get("forward_passes"),
        "forward_passes_minimum": snapshot.get("forward_passes_minimum"),
        "quality_result_computed": snapshot.get("quality_result_computed"),
        "completed_task_ids": [TASK_ID] if evaluation_returned else [],
        "quality_aggregate_exposed": False,
        "rerun_automatically_authorized": False,
        **safe_failure,
    }
    _validate_public_artifact(failed, repo_root=preflight.repo_root)
    _atomic_replace_owned(preflight.attempt_path, canonical_json_bytes(failed))


def _authenticated_mbpp_train_arrow() -> Path:
    datasets_module = importlib.import_module("datasets")
    revision = _resolve_cached_dataset_revision(datasets_module)
    path = revision / "mbpp-train.arrow"
    expected = DATASET_CACHE_RESOURCES["mbpp-train.arrow"]
    if not path.is_file():
        raise StageAAuthenticationError(
            "authenticated local MBPP training Arrow resource is missing"
        )
    try:
        size = path.stat().st_size
    except OSError as error:
        raise StageAAuthenticationError(
            "cannot stat authenticated local MBPP training Arrow resource"
        ) from error
    if size != expected["size_bytes"] or _file_sha256(path) != expected["sha256"]:
        raise StageAAuthenticationError("authenticated local MBPP training Arrow resource drifted")
    return path


def _select_exact_task_from_arrow(
    arrow_path: Path,
    *,
    task_id: int,
) -> Mapping[str, object]:
    try:
        pyarrow = importlib.import_module("pyarrow")
        pyarrow_compute = importlib.import_module("pyarrow.compute")
        pyarrow_ipc = importlib.import_module("pyarrow.ipc")
        selected_batches: list[object] = []
        selected_rows = 0
        with pyarrow.memory_map(str(arrow_path), "r") as source:
            reader = pyarrow_ipc.open_stream(source)
            task_id_index = reader.schema.get_field_index("task_id")
            if task_id_index < 0:
                raise StageAAuthenticationError(
                    "authenticated local MBPP Arrow schema lacks task_id"
                )
            for batch in reader:
                task_ids = batch.column(task_id_index)
                mask = pyarrow_compute.equal(
                    task_ids,
                    pyarrow.scalar(task_id, type=task_ids.type),
                )
                filtered = batch.filter(mask)
                if filtered.num_rows:
                    selected_batches.append(filtered)
                    selected_rows += int(filtered.num_rows)
        if selected_rows != 1:
            raise StageAAuthenticationError(
                f"authenticated local MBPP Arrow selection returned non-unique task {task_id}"
            )
        table = pyarrow.Table.from_batches(selected_batches)
        rows = table.to_pylist()
    except StageAAuthenticationError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise StageAAuthenticationError(
            "cannot scan the authenticated local MBPP Arrow resource"
        ) from error
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or int(rows[0].get("task_id", -1)) != task_id
    ):
        raise StageAAuthenticationError("authenticated local MBPP Arrow row identity is malformed")
    return dict(rows[0])


def load_exact_authenticated_task(preflight: StageAPreflight) -> Mapping[str, object]:
    del preflight
    from recurquant.public_data import mbpp_row_sha256

    verifier = _script_module("verify_statelease_stage0")
    verifier.guard_protected_mbpp_window(
        stage="stagea",
        task_ids=(str(TASK_ID),),
        contains_quality_data=True,
    )
    arrow_path = _authenticated_mbpp_train_arrow()
    row = _select_exact_task_from_arrow(arrow_path, task_id=TASK_ID)
    if mbpp_row_sha256(row) != TASK_ROW_SHA256:
        raise StageAAuthenticationError("already-open task-666 row content drifted")
    return row


def tokenize_authenticated_task(
    preflight: StageAPreflight,
    row: Mapping[str, object],
    *,
    local_files_only: bool,
) -> TokenizedTask:
    if local_files_only is not True:
        raise StageAAuthenticationError(
            "Experiment 011 tokenizer loading requires --local-files-only"
        )
    from transformers import AutoTokenizer

    from recurquant.public_data import format_mbpp_example

    formatted = format_mbpp_example(row)
    if hashlib.sha256(formatted.prompt.encode("utf-8")).hexdigest() != PROMPT_TEXT_SHA256:
        raise StageAAuthenticationError("task-666 formatted prompt text drifted")
    if hashlib.sha256(formatted.code.encode("utf-8")).hexdigest() != CODE_TEXT_SHA256:
        raise StageAAuthenticationError("task-666 formatted code text drifted")
    snapshot, cache_receipt = _authenticate_model_snapshot()
    if (
        cache_receipt.get("snapshot_resource_id") != f"{MODEL_ID}@{MODEL_REVISION}"
        or cache_receipt.get("alternate_or_unpinned_weight_files_absent") is not True
    ):
        raise StageAAuthenticationError("tokenizer snapshot authentication drifted")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.__class__.__name__ != "Qwen2Tokenizer":
        raise StageAAuthenticationError("pinned tokenizer class drifted")
    prompt_ids = tokenizer(
        formatted.prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    code_ids = tokenizer(
        formatted.code,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    token_manifest = [
        {
            "task_id": TASK_ID,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "code_tokens": int(code_ids.shape[1]),
            "aligned_scored_tokens": int(code_ids.shape[1]) - 1,
            "full_code_scored_tokens": int(code_ids.shape[1]),
        }
    ]
    if token_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError(
            "tokenized task 666 does not match the authenticated token manifest"
        )
    copied_manifest = preflight.anchor.get("token_manifest")
    if copied_manifest != EXPECTED_TOKEN_MANIFEST:
        raise StageAAuthenticationError(
            "runtime token identity differs from the copied Experiment 009 manifest"
        )
    if int(row.get("task_id", -1)) != TASK_ID:
        raise StageAAuthenticationError("encoded row identity drifted")
    _validate_token_id_hashes(prompt_ids, code_ids)
    return TokenizedTask(
        row=row,
        prompt_ids=prompt_ids,
        code_ids=code_ids,
        token_manifest=[dict(item) for item in token_manifest],
    )


def _canonical_token_ids_sha256(value: torch.Tensor) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.long
        or value.ndim != 2
        or value.shape[0] != 1
    ):
        raise StageAAuthenticationError("token IDs must be one batch of torch.long IDs")
    ids = [int(item) for item in value.detach().to("cpu").reshape(-1).tolist()]
    return hashlib.sha256(canonical_json_bytes(ids)).hexdigest()


def _validate_token_id_hashes(
    prompt_ids: torch.Tensor,
    code_ids: torch.Tensor,
) -> dict[str, object]:
    if tuple(prompt_ids.shape) != (1, PROMPT_TOKENS):
        raise StageAAuthenticationError("prompt token shape drifted")
    if tuple(code_ids.shape) != (1, CODE_TOKENS):
        raise StageAAuthenticationError("code token shape drifted")
    prompt_hash = _canonical_token_ids_sha256(prompt_ids)
    code_hash = _canonical_token_ids_sha256(code_ids)
    if prompt_hash != PROMPT_TOKEN_IDS_SHA256:
        raise StageAAuthenticationError("prompt token-ID hash drifted")
    if code_hash != CODE_TOKEN_IDS_SHA256:
        raise StageAAuthenticationError("code token-ID hash drifted")
    return {
        "prompt_token_ids_sha256": prompt_hash,
        "code_token_ids_sha256": code_hash,
        "token_id_hash_serialization": TOKEN_ID_HASH_SERIALIZATION,
        "prompt_shape": list(prompt_ids.shape),
        "code_shape": list(code_ids.shape),
    }


def _select_cuda_device(requested: str) -> torch.device:
    if requested not in {"auto", "cuda"}:
        raise StageAAuthenticationError("Stage A supports only the pinned CUDA BF16 path")
    if not torch.cuda.is_available():
        raise StageAAuthenticationError("the frozen Stage-A CUDA device is unavailable")
    return torch.device("cuda")


def load_model_weights(
    configuration: ModelConfiguration,
    preflight: StageAPreflight,
    *,
    device_name: str,
    local_files_only: bool,
) -> tuple[torch.nn.Module, torch.device]:
    del preflight
    from transformers import Qwen3_5ForCausalLM

    if local_files_only is not True:
        raise StageAAuthenticationError("Experiment 011 model loading requires --local-files-only")
    snapshot, cache_receipt = _authenticate_model_snapshot()
    if (
        configuration.identity.get("authenticated_snapshot_resource_id")
        != cache_receipt.get("snapshot_resource_id")
        or configuration.identity.get("model_cache_resource_manifest_sha256")
        != cache_receipt.get("resource_manifest_sha256")
        or cache_receipt.get("alternate_or_unpinned_weight_files_absent") is not True
    ):
        raise StageAAuthenticationError(
            "authenticated model snapshot differs from the pre-weight configuration"
        )
    device = _select_cuda_device(device_name)
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=local_files_only,
        trust_remote_code=False,
    ).to(device)
    snapshot_after, receipt_after = _authenticate_model_snapshot()
    if snapshot_after != snapshot or receipt_after != cache_receipt:
        raise StageAAuthenticationError("authenticated model snapshot changed during weight load")
    model.eval()
    loaded_geometry = _validate_model_geometry(model.config)
    if loaded_geometry != configuration.identity["geometry"]:
        raise StageAAuthenticationError(
            "loaded model geometry differs from the pre-weight configuration"
        )
    return model, device


@dataclass(slots=True)
class _AlignedAccumulator:
    kl: list[torch.Tensor]
    reference_nll: list[torch.Tensor]
    candidate_nll: list[torch.Tensor]
    top1: list[torch.Tensor]
    finite: list[bool]
    records: list[dict[str, object]]

    @classmethod
    def empty(cls) -> _AlignedAccumulator:
        return cls([], [], [], [], [], [])

    def append(
        self,
        *,
        token_index: int,
        reference_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        input_token: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        reference = reference_logits.detach().to("cpu")
        candidate = candidate_logits.detach().to("cpu")
        input_cpu = input_token.detach().to("cpu")
        target_cpu = target.detach().to("cpu")
        logits_finite = bool(
            torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item()
        )
        values = token_fidelity(reference, candidate, target_cpu).to_cpu()
        scalars = {
            "kl": float(values.kl.reshape(-1)[0].item()),
            "reference_nll": float(values.reference_nll.reshape(-1)[0].item()),
            "candidate_nll": float(values.candidate_nll.reshape(-1)[0].item()),
            "top1_agreement": bool(values.top1_agreement.reshape(-1)[0].item()),
        }
        if not logits_finite or not all(
            math.isfinite(float(value)) for key, value in scalars.items() if key != "top1_agreement"
        ):
            raise RuntimeError(f"aligned token {token_index} produced a non-finite value")
        self.kl.append(values.kl.reshape(-1))
        self.reference_nll.append(values.reference_nll.reshape(-1))
        self.candidate_nll.append(values.candidate_nll.reshape(-1))
        self.top1.append(values.top1_agreement.reshape(-1))
        self.finite.append(logits_finite)
        self.records.append(
            {
                "write_index": token_index,
                "input_token_id": int(input_cpu.reshape(-1)[0].item()),
                "target_token_id": int(target_cpu.reshape(-1)[0].item()),
                **scalars,
                "delta_nll": scalars["candidate_nll"] - scalars["reference_nll"],
                "all_logits_finite": logits_finite,
            }
        )

    def summary(self) -> dict[str, float | int | bool]:
        if not self.records:
            raise RuntimeError("aligned accumulator is empty")
        summary = fidelity_summary(
            TokenFidelity(
                kl=torch.cat(self.kl),
                reference_nll=torch.cat(self.reference_nll),
                candidate_nll=torch.cat(self.candidate_nll),
                top1_agreement=torch.cat(self.top1),
            )
        )
        return {**summary, "all_logits_finite": all(self.finite)}


def _snapshot_states_cpu(cache: object) -> dict[int, torch.Tensor]:
    checkpoint = getattr(cache, "checkpoint", None)
    layout = getattr(cache, "layout", None)
    materialize = getattr(checkpoint, "materialize", None)
    if checkpoint is not None and layout is not None and callable(materialize):
        states = materialize()
        if isinstance(states, Mapping):
            return {
                int(layer): tensor.detach().to("cpu").clone() for layer, tensor in states.items()
            }
    result: dict[int, torch.Tensor] = {}
    for state in iter_recurrent_states(cache):
        if state.state_index != 0:
            raise RuntimeError("Stage A requires one recurrent state per layer")
        if state.layer_index in result:
            raise RuntimeError("Stage A observed a duplicate recurrent layer")
        result[state.layer_index] = state.tensor.detach().to("cpu").clone()
    if tuple(sorted(result)) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"recurrent-state snapshot layers drifted: {sorted(result)}")
    return result


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().to("cpu")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_snapshot_digest(states: Mapping[int, torch.Tensor]) -> str:
    return _canonical_sha256(
        {
            str(layer): {
                "dtype": str(states[layer].dtype),
                "shape": list(states[layer].shape),
                "sha256": _tensor_digest(states[layer]),
            }
            for layer in sorted(states)
        }
    )


@contextlib.contextmanager
def _capture_prefill_writes(cache: object) -> Iterator[dict[int, torch.Tensor]]:
    original = getattr(cache, "update_recurrent_state", None)
    if not callable(original):
        raise TypeError("candidate cache does not expose update_recurrent_state")
    captured: dict[int, torch.Tensor] = {}

    def wrapped(
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        if state_idx != 0:
            raise RuntimeError("Stage A supports only recurrent state index zero")
        layer_index = int(layer_idx)
        if layer_index in captured:
            raise RuntimeError(f"prefill wrote recurrent layer {layer_index} twice")
        captured[layer_index] = recurrent_states.detach().to("cpu").clone()
        return original(
            recurrent_states,
            layer_idx,
            state_idx=state_idx,
            **kwargs,
        )

    cache.update_recurrent_state = wrapped  # type: ignore[attr-defined]
    try:
        yield captured
    finally:
        cache.update_recurrent_state = original  # type: ignore[attr-defined]


def _assert_shared_prefill(
    reference: Mapping[int, torch.Tensor],
    candidate: Mapping[int, torch.Tensor],
    *,
    method: str,
) -> dict[str, object]:
    expected_layers = set(LINEAR_LAYER_INDICES)
    if set(reference) != expected_layers or set(candidate) != expected_layers:
        raise RuntimeError(f"{method} prefill layer set differs from FP32")
    expected_shape = (
        1,
        int(EXPECTED_GEOMETRY["value_heads"]),
        int(EXPECTED_GEOMETRY["key_rows"]),
        int(EXPECTED_GEOMETRY["value_width"]),
    )
    for layer_index in sorted(reference):
        reference_tensor = reference[layer_index]
        candidate_tensor = candidate[layer_index]
        if (
            not isinstance(reference_tensor, torch.Tensor)
            or not isinstance(candidate_tensor, torch.Tensor)
            or reference_tensor.dtype != torch.float32
            or candidate_tensor.dtype != torch.float32
            or tuple(reference_tensor.shape) != expected_shape
            or tuple(candidate_tensor.shape) != expected_shape
        ):
            raise RuntimeError(
                f"{method} prefill layer {layer_index} is not the exact raw FP32 geometry"
            )
        if not torch.equal(reference_tensor, candidate_tensor):
            maximum = float((reference_tensor - candidate_tensor).abs().max().item())
            raise RuntimeError(
                f"{method} did not pack the identical FP32 prefill state "
                f"at layer {layer_index}; max_abs={maximum}"
            )
    return {
        "identical_raw_fp32_prefill_state": True,
        "layer_count": len(reference),
        "reference_digest": _state_snapshot_digest(reference),
        "candidate_digest": _state_snapshot_digest(candidate),
    }


def _snapshot_nonrecurrent_prefill(cache: object) -> dict[str, torch.Tensor]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or len(layers) != EXPECTED_GEOMETRY["num_hidden_layers"]:
        raise RuntimeError("prefill cache does not expose the frozen 24-layer geometry")
    result: dict[str, torch.Tensor] = {}
    linear = set(LINEAR_LAYER_INDICES)
    for layer_index, layer in enumerate(layers):
        if layer_index in linear:
            initialized = getattr(layer, "is_conv_states_initialized", None)
            conv_states = getattr(layer, "conv_states", None)
            try:
                is_initialized = bool(initialized[0])
                tensor = conv_states[0]
            except (IndexError, KeyError, TypeError, AttributeError) as error:
                raise RuntimeError(
                    f"linear layer {layer_index} lacks an auditable convolution cache"
                ) from error
            if (
                not is_initialized
                or not isinstance(tensor, torch.Tensor)
                or tensor.shape[0] != 1
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise RuntimeError(
                    f"linear layer {layer_index} convolution prefill cache is invalid"
                )
            result[f"layer.{layer_index}.conv.0"] = tensor.detach().to("cpu").clone()
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape[0] != 1
                or tensor.shape[-2] != PROMPT_TOKENS
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise RuntimeError(
                    f"full-attention layer {layer_index} {name} prefill cache is invalid"
                )
            result[f"layer.{layer_index}.{name}"] = tensor.detach().to("cpu").clone()
    expected_count = len(LINEAR_LAYER_INDICES) + 2 * (
        int(EXPECTED_GEOMETRY["num_hidden_layers"]) - len(LINEAR_LAYER_INDICES)
    )
    if len(result) != expected_count:
        raise RuntimeError("non-recurrent prefill cache tensor count drifted")
    return result


def _assert_shared_nonrecurrent_prefill(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    method: str,
) -> dict[str, object]:
    if set(reference) != set(candidate):
        raise RuntimeError(f"{method} non-recurrent prefill cache keys differ from FP32")
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left, right)
        ):
            raise RuntimeError(f"{method} non-recurrent prefill cache differs from FP32 at {name}")
    digest = _state_snapshot_digest(
        {index: reference[name] for index, name in enumerate(sorted(reference))}
    )
    return {
        "identical_full_attention_kv_and_convolution_prefill_state": True,
        "nonrecurrent_tensor_count": len(reference),
        "nonrecurrent_reference_digest": digest,
        "nonrecurrent_candidate_digest": _state_snapshot_digest(
            {index: candidate[name] for index, name in enumerate(sorted(candidate))}
        ),
    }


def _validate_prefill_cache_contract(
    method: str,
    cache: object,
) -> dict[str, object]:
    if method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        diagnostics = _diagnostics(cache)
        if len(diagnostics) != len(LINEAR_LAYER_INDICES):
            raise RuntimeError(f"{method} prefill diagnostics omitted recurrent layers")
        for row in diagnostics:
            if (
                row.get("state_updates") != 1
                or row.get("tokens_observed") != PROMPT_TOKENS
                or row.get("replay_valid_count") != 0
            ):
                raise RuntimeError(
                    f"{method} did not start decode from one packed prefill "
                    "checkpoint and an empty replay buffer"
                )
        records = _evidence_records(cache)
        if len(records) != len(LINEAR_LAYER_INDICES):
            raise RuntimeError(f"{method} prefill evidence count drifted")
        if any(
            record.get("token_count") != PROMPT_TOKENS
            or record.get("boundary") is not None
            or record.get("replay_valid_count") != 0
            or "prefill" not in str(record.get("action"))
            for record in records
        ):
            raise RuntimeError(f"{method} prefill checkpoint receipt drifted")
        return {
            "packed_prefill_checkpoint": True,
            "empty_replay_buffer": True,
            "prefill_evidence_records": len(records),
        }
    if method in EQUAL_BYTE_NO_REPLAY_METHODS:
        storage = _storage_summary(cache)
        records = _evidence_records(cache)
        if (
            storage.get("checkpoint_present") is not True
            or storage.get("update_count") != 1
            or storage.get("successful_tokens") != PROMPT_TOKENS
            or len(records) != 1
            or records[0].get("previous_checkpoint_present") is not False
            or records[0].get("token_count") != PROMPT_TOKENS
        ):
            raise RuntimeError(f"{method} did not install exactly one prefill checkpoint")
        return {
            "packed_prefill_checkpoint": True,
            "no_replay_buffer": True,
            "prefill_evidence_records": 1,
        }
    storage = _storage_summary(cache)
    if (
        storage.get("resident_bytes") != 2_564_096
        or storage.get("resident_bytes_including_selector") != 2_711_552
        or storage.get("high_precision_groups") != 1_976
    ):
        raise RuntimeError("historical RHT-CQER prefill storage identity drifted")
    return {
        "packed_prefill_checkpoint": True,
        "historical_anchor": True,
        "resident_bytes_including_selector": 2_711_552,
    }


def _evidence_records(cache: object) -> list[dict[str, object]]:
    records = getattr(cache, "update_evidence", ())
    result: list[dict[str, object]] = []
    for index, record in enumerate(records):
        converter = getattr(record, "evidence_dict", None)
        value = converter() if callable(converter) else record
        if not isinstance(value, Mapping):
            raise RuntimeError(f"cache evidence record {index} is not a mapping")
        result.append(_jsonable(dict(value)))
    return result


def _storage_summary(cache: object) -> dict[str, object]:
    callback = getattr(cache, "storage_summary", None)
    if not callable(callback):
        raise RuntimeError("candidate cache does not expose storage_summary")
    value = callback()
    if not isinstance(value, Mapping):
        raise RuntimeError("candidate storage summary is not a mapping")
    return _jsonable(dict(value))


TensorStorageKey = tuple[str, int | None, int, int]


def _expected_shared_tensor_names() -> frozenset[str]:
    linear = set(LINEAR_LAYER_INDICES)
    names = {f"layer.{layer_index}.conv.0" for layer_index in LINEAR_LAYER_INDICES}
    for layer_index in range(24):
        if layer_index not in linear:
            names.add(f"layer.{layer_index}.keys")
            names.add(f"layer.{layer_index}.values")
    return frozenset(names)


def _expected_candidate_tensor_names(method: str) -> frozenset[str]:
    if method == ORIGINAL_RHT_METHOD:
        per_layer = (
            "checkpoint.low_payload",
            "checkpoint.high_payload",
            "checkpoint.scales",
            "checkpoint.precision_mask",
            "query_energy_ema",
        )
        return frozenset(
            f"layer.{layer_index}.{field}"
            for layer_index in LINEAR_LAYER_INDICES
            for field in per_layer
        )
    if method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        per_layer = (
            "checkpoint.low_payload",
            "checkpoint.high_payload",
            "checkpoint.scales",
            "checkpoint.precision_mask",
            "query_energy_ema",
            "normalized_key_buffer",
            "update_buffer",
            "log_decay_buffer",
            "valid_count",
        )
        return frozenset(
            f"layer.{layer_index}.{field}"
            for layer_index in LINEAR_LAYER_INDICES
            for field in per_layer
        )
    equal_byte_fields = {
        "expanded_rht_q4_q8": (
            "q4_payload",
            "q8_payload",
            "scales",
            "precision_mask",
        ),
        "rht_q4_q6_q8": (
            "q4_payload",
            "q6_payload",
            "q8_payload",
            "scales",
            "precision_codes",
        ),
        "rht_residual_q4": (
            "base_q4_payload",
            "base_scales",
            "residual_q4_payload",
            "residual_scales",
            "lease_mask",
        ),
    }
    fields = equal_byte_fields.get(method)
    if fields is None:
        raise ValueError(f"{method} has no expected candidate tensor schema")
    names = {
        f"checkpoint.layer_{position}.{field}"
        for position in range(len(LINEAR_LAYER_INDICES))
        for field in fields
    }
    names.update(
        {
            "checkpoint.query_energy_ema",
            "checkpoint.reserved_padding",
        }
    )
    return frozenset(names)


EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES = _expected_shared_tensor_names()
EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT = len(EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES)
EXPECTED_CANDIDATE_PERSISTENT_TENSOR_COUNTS = {
    method: len(_expected_candidate_tensor_names(method)) for method in QUALITY_METHODS
}


def _tensor_storage_key(tensor: torch.Tensor) -> TensorStorageKey:
    if tensor.numel() == 0:
        raise RuntimeError("persistent tensor storage cannot be empty")
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        int(storage.nbytes()),
    )


def _tensor_schema(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": int(tensor.untyped_storage().nbytes()),
    }


def _shared_cache_tensors(cache: object) -> dict[str, torch.Tensor]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or len(layers) != 24:
        raise RuntimeError("cache does not expose the frozen 24-layer tensor layout")
    shared: dict[str, torch.Tensor] = {}
    linear = set(LINEAR_LAYER_INDICES)
    for layer_index, layer in enumerate(layers):
        if layer_index in linear:
            initialized = getattr(layer, "is_conv_states_initialized", None)
            states = getattr(layer, "conv_states", None)
            try:
                tensor = states[0]
                ready = bool(initialized[0])
            except (IndexError, KeyError, TypeError, AttributeError) as error:
                raise RuntimeError(
                    f"linear layer {layer_index} lacks shared convolution storage"
                ) from error
            if not ready or not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"linear layer {layer_index} convolution storage is not initialized"
                )
            shared[f"layer.{layer_index}.conv.0"] = tensor
            continue
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"full-attention layer {layer_index} lacks shared {name} storage"
                )
            shared[f"layer.{layer_index}.{name}"] = tensor
        sliding = getattr(layer, "_sliding_window_tensor", None)
        if isinstance(sliding, torch.Tensor):
            shared[f"layer.{layer_index}.sliding_window"] = sliding
    keys = [_tensor_storage_key(tensor) for tensor in shared.values()]
    if len(keys) != len(set(keys)):
        raise RuntimeError("shared cache tensors unexpectedly alias persistent storage")
    return shared


def _shared_cache_schema(cache: object) -> dict[str, dict[str, object]]:
    return {
        name: _tensor_schema(tensor)
        for name, tensor in sorted(_shared_cache_tensors(cache).items())
    }


def _assert_shared_cache_schema(
    cache: object,
    reference_schema: Mapping[str, object],
    *,
    method: str,
) -> dict[str, dict[str, object]]:
    actual = _shared_cache_schema(cache)
    if actual != reference_schema:
        raise RuntimeError(
            f"{method} post-decode shared KV/convolution storage schema differs from FP32"
        )
    return actual


def _statelease_candidate_tensors(
    cache: object,
    *,
    method: str,
) -> dict[str, torch.Tensor]:
    from recurquant.mixed_quantization import PackedMixedQuantizedTensor
    from recurquant.statelease_baselines import (
        FixedCC1RecurrentStateCache,
        FixedCC2RecurrentStateCache,
        FixedCC4RecurrentStateCache,
        FixedCC5RecurrentStateCache,
        FixedCut4In5RecurrentStateCache,
        FixedReplayLinearAttentionLayer,
    )
    from recurquant.statelease_cache import (
        StateLeaseLinearAttentionLayer,
        StateLeaseRecurrentStateCache,
        _StateLeaseStateView,
    )

    expected_cache_types = {
        STATELEASE_METHOD: StateLeaseRecurrentStateCache,
        "fixed_cc1": FixedCC1RecurrentStateCache,
        "fixed_cc2": FixedCC2RecurrentStateCache,
        "fixed_cc4": FixedCC4RecurrentStateCache,
        "fixed_cc5": FixedCC5RecurrentStateCache,
        "fixed_cut4_in5": FixedCut4In5RecurrentStateCache,
    }
    expected_cache = expected_cache_types[method]
    if type(cache) is not expected_cache:
        raise RuntimeError(
            f"{method} cache type drifted: {type(cache).__module__}.{type(cache).__qualname__}"
        )
    layers = tuple(cache.statelease_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"{method} recurrent layer identities drifted")
    expected_layer = (
        StateLeaseLinearAttentionLayer
        if method == STATELEASE_METHOD
        else FixedReplayLinearAttentionLayer
    )
    result: dict[str, torch.Tensor] = {}
    for layer_index, layer in layers:
        if type(layer) is not expected_layer or type(layer.recurrent_states) is not (
            _StateLeaseStateView
        ):
            raise RuntimeError(f"{method} layer {layer_index} type or state view drifted")
        packed = layer.packed_checkpoint
        if type(packed) is not PackedMixedQuantizedTensor:
            raise RuntimeError(f"{method} layer {layer_index} checkpoint type drifted")
        assert isinstance(packed, PackedMixedQuantizedTensor)
        if tuple(packed.original_shape) != (1, 16, 128, 128):
            raise RuntimeError(f"{method} layer {layer_index} checkpoint geometry drifted")
        packed_fields = {
            "low_payload": packed.low_payload,
            "high_payload": packed.high_payload,
            "scales": packed.scales,
            "precision_mask": packed.precision_mask,
        }
        expected_dtypes = {
            "low_payload": torch.uint8,
            "high_payload": torch.int8,
            "scales": torch.float16,
            "precision_mask": torch.uint8,
        }
        for name, tensor in packed_fields.items():
            if tensor.dtype != expected_dtypes[name]:
                raise RuntimeError(f"{method} layer {layer_index} checkpoint {name} dtype drifted")
            result[f"layer.{layer_index}.checkpoint.{name}"] = tensor
        auxiliaries = {
            "query_energy_ema": layer.query_energy_ema,
            "normalized_key_buffer": layer.normalized_key_buffer,
            "update_buffer": layer.update_buffer,
            "log_decay_buffer": layer.log_decay_buffer,
            "valid_count": layer.valid_count,
        }
        expected_auxiliary = {
            "query_energy_ema": (torch.float32, (16, 128)),
            "normalized_key_buffer": (torch.bfloat16, (5, 16, 128)),
            "update_buffer": (torch.bfloat16, (5, 16, 128)),
            "log_decay_buffer": (torch.float32, (5, 16)),
            "valid_count": (torch.int32, (1,)),
        }
        for name, value in auxiliaries.items():
            expected_dtype, expected_shape = expected_auxiliary[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != expected_dtype
                or tuple(value.shape) != expected_shape
            ):
                raise RuntimeError(f"{method} layer {layer_index} persistent {name} schema drifted")
            result[f"layer.{layer_index}.{name}"] = value
    return result


def _equal_byte_candidate_tensors(
    cache: object,
    *,
    method: str,
) -> dict[str, torch.Tensor]:
    from recurquant.statelease_equal_byte_baselines import EqualByteCheckpoint
    from recurquant.statelease_equal_byte_cache import (
        EqualByteLinearAttentionLayer,
        EqualByteQwen35Cache,
        _EqualByteStateView,
    )

    if type(cache) is not EqualByteQwen35Cache or getattr(cache, "codec", None) != method:
        raise RuntimeError(f"{method} equal-byte cache type or codec drifted")
    checkpoint = getattr(cache, "checkpoint", None)
    if type(checkpoint) is not EqualByteCheckpoint:
        raise RuntimeError(f"{method} equal-byte checkpoint type drifted")
    assert isinstance(checkpoint, EqualByteCheckpoint)
    checkpoint.validate()
    layers = tuple(cache.equal_byte_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError(f"{method} equal-byte layer identities drifted")
    for layer_index, layer in layers:
        if (
            type(layer) is not EqualByteLinearAttentionLayer
            or type(layer.recurrent_states) is not _EqualByteStateView
        ):
            raise RuntimeError(f"{method} layer {layer_index} type or state view drifted")
    persistent = checkpoint.persistent_tensors()
    names = [name for name, _tensor in persistent]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{method} checkpoint tensor names are not unique")
    return {f"checkpoint.{name}": tensor for name, tensor in persistent}


def _rht_candidate_tensors(cache: object) -> dict[str, torch.Tensor]:
    from recurquant.mixed_quantization import PackedMixedQuantizedTensor
    from recurquant.packed_cache import (
        RightRhtQueryEmaMixedPackedLinearAttentionLayer,
        RightRhtQueryEmaMixedPackedRecurrentStateCache,
        _MixedPackedStateView,
    )

    if type(cache) is not RightRhtQueryEmaMixedPackedRecurrentStateCache:
        raise RuntimeError("historical RHT-CQER cache type drifted")
    layers = tuple(cache.mixed_packed_layers())
    if tuple(index for index, _layer in layers) != LINEAR_LAYER_INDICES:
        raise RuntimeError("historical RHT-CQER recurrent layer identities drifted")
    result: dict[str, torch.Tensor] = {}
    for layer_index, layer in layers:
        if (
            type(layer) is not RightRhtQueryEmaMixedPackedLinearAttentionLayer
            or type(layer.recurrent_states) is not _MixedPackedStateView
        ):
            raise RuntimeError(
                f"historical RHT-CQER layer {layer_index} type or state view drifted"
            )
        packed = layer.packed_states.get(0)
        if type(packed) is not PackedMixedQuantizedTensor:
            raise RuntimeError(f"historical RHT-CQER layer {layer_index} checkpoint type drifted")
        assert isinstance(packed, PackedMixedQuantizedTensor)
        if tuple(packed.original_shape) != (1, 16, 128, 128):
            raise RuntimeError(
                f"historical RHT-CQER layer {layer_index} checkpoint geometry drifted"
            )
        for name, expected_dtype in {
            "low_payload": torch.uint8,
            "high_payload": torch.int8,
            "scales": torch.float16,
            "precision_mask": torch.uint8,
        }.items():
            tensor = getattr(packed, name)
            if tensor.dtype != expected_dtype:
                raise RuntimeError(f"historical RHT-CQER layer {layer_index} {name} dtype drifted")
            result[f"layer.{layer_index}.checkpoint.{name}"] = tensor
        query_ema = layer.query_energy_ema
        if (
            not isinstance(query_ema, torch.Tensor)
            or query_ema.dtype != torch.float32
            or tuple(query_ema.shape) != (16, 128)
        ):
            raise RuntimeError(f"historical RHT-CQER layer {layer_index} query EMA drifted")
        if layer.previous_raw_mask_packed is not None:
            raise RuntimeError("historical RHT-CQER unexpectedly retained a raw mask")
        result[f"layer.{layer_index}.query_energy_ema"] = query_ema
    return result


def _candidate_persistent_tensors(
    method: str,
    cache: object,
) -> dict[str, torch.Tensor]:
    if method == ORIGINAL_RHT_METHOD:
        result = _rht_candidate_tensors(cache)
        expected_resident = 2_711_552
    elif method in {STATELEASE_METHOD, *FIXED_REPLAY_METHODS}:
        result = _statelease_candidate_tensors(cache, method=method)
        expected_resident = FROZEN_STATELEASE_RESIDENT_BYTES
    elif method in EQUAL_BYTE_NO_REPLAY_METHODS:
        result = _equal_byte_candidate_tensors(cache, method=method)
        expected_resident = FROZEN_STATELEASE_RESIDENT_BYTES
    else:
        raise ValueError(f"{method} has no strict candidate-storage schema")
    keys = [_tensor_storage_key(tensor) for tensor in result.values() if tensor.numel() != 0]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{method} candidate persistent tensors alias storage")
    for name, tensor in result.items():
        logical = tensor.numel() * tensor.element_size()
        storage_bytes = int(tensor.untyped_storage().nbytes())
        if tensor.numel() == 0:
            if storage_bytes != 0:
                raise RuntimeError(
                    f"{method} empty candidate tensor {name} retains backing storage"
                )
            continue
        if not tensor.is_contiguous() or tensor.storage_offset() != 0 or storage_bytes != logical:
            raise RuntimeError(f"{method} candidate tensor {name} does not own exact storage")
    resident = sum(key[3] for key in keys)
    if resident != expected_resident:
        raise RuntimeError(
            f"{method} explicit candidate tensors own {resident} bytes; expected "
            f"{expected_resident}"
        )
    return result


def _reachable_tensor_paths(cache: object) -> list[tuple[str, torch.Tensor]]:
    from recurquant.packed_cache import _MixedPackedStateView
    from recurquant.statelease_cache import _StateLeaseStateView
    from recurquant.statelease_equal_byte_cache import _EqualByteStateView

    state_view_types = (
        _MixedPackedStateView,
        _StateLeaseStateView,
        _EqualByteStateView,
    )
    tensors: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    visited_nodes = 0

    def visit(value: object, path: str) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > 100_000:
            raise RuntimeError("persistent cache traversal exceeded its closed bound")
        if isinstance(value, torch.Tensor):
            tensors.append((path, value))
            return
        if value is None or isinstance(
            value,
            (str, bytes, bytearray, int, float, bool, Path, torch.device, torch.dtype),
        ):
            return
        if isinstance(value, (types.ModuleType, type)):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) in state_view_types:
            attributes = getattr(value, "__dict__", {})
            for name, item in attributes.items():
                if name not in {"_owner", "owner"}:
                    visit(item, f"{path}.{name}")
            return
        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                visit(key, f"{path}.key[{index}]")
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            ordered = sorted(
                value,
                key=lambda item: repr(type(item)) + repr(item),
            )
            for index, item in enumerate(ordered):
                visit(item, f"{path}.set[{index}]")
            return
        if isinstance(value, types.MethodType):
            visit(value.__self__, f"{path}.__self__")
            visit(value.__func__, f"{path}.__func__")
            return
        if isinstance(value, types.FunctionType):
            visit(value.__defaults__, f"{path}.__defaults__")
            visit(value.__kwdefaults__, f"{path}.__kwdefaults__")
            if value.__closure__ is not None:
                for index, cell in enumerate(value.__closure__):
                    try:
                        contents = cell.cell_contents
                    except ValueError:
                        continue
                    visit(contents, f"{path}.__closure__[{index}]")
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                visit(item, f"{path}.{name}")
        slot_names: set[str] = set()
        for owner in type(value).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            slot_names.update(name for name in slots if name not in {"__dict__", "__weakref__"})
        for name in sorted(slot_names):
            if not isinstance(attributes, dict) or name not in attributes:
                try:
                    item = getattr(value, name)
                except AttributeError:
                    continue
                visit(item, f"{path}.{name}")

    visit(cache, "cache")
    return tensors


def _assert_reachable_tensor_storage_closure(
    *,
    method: str,
    reachable: Sequence[tuple[str, torch.Tensor]],
    shared: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> None:
    shared_keys = {_tensor_storage_key(tensor) for tensor in shared.values()}
    candidate_keys = {
        _tensor_storage_key(tensor) for tensor in candidate.values() if tensor.numel() != 0
    }
    if shared_keys & candidate_keys:
        raise RuntimeError(f"{method} candidate storage aliases shared KV/convolution storage")
    allowed = shared_keys | candidate_keys
    hidden_empty = [
        path
        for path, tensor in reachable
        if tensor.numel() == 0 and int(tensor.untyped_storage().nbytes()) != 0
    ]
    if hidden_empty:
        raise RuntimeError(
            f"{method} retains empty tensor views over undeclared storage: {sorted(hidden_empty)}"
        )
    paths_by_storage: dict[TensorStorageKey, list[str]] = {}
    unknown: dict[str, dict[str, object]] = {}
    for path, tensor in reachable:
        if tensor.numel() == 0:
            continue
        key = _tensor_storage_key(tensor)
        paths_by_storage.setdefault(key, []).append(path)
        if key not in allowed:
            unknown[path] = _tensor_schema(tensor)
    if unknown:
        details = [
            f"{path} ({schema['dtype']}, {schema['shape']})"
            for path, schema in sorted(unknown.items())
        ]
        raise RuntimeError(f"{method} retains undeclared persistent tensor storage: {details}")
    missing = allowed - set(paths_by_storage)
    if missing:
        raise RuntimeError(f"{method} explicit persistent tensor storage is unreachable")
    aliases = {key: sorted(paths) for key, paths in paths_by_storage.items() if len(paths) != 1}
    if aliases:
        raise RuntimeError(f"{method} persistent tensor storage has unexpected aliases")


def _audit_persistent_raw_state(
    cache: object,
    *,
    method: str,
    reference_shared_schema: Mapping[str, object],
) -> dict[str, object]:
    shared = _shared_cache_tensors(cache)
    shared_schema = _assert_shared_cache_schema(
        cache,
        reference_shared_schema,
        method=method,
    )
    candidate = _candidate_persistent_tensors(method, cache)
    shared_keys = {_tensor_storage_key(tensor) for tensor in shared.values()}
    candidate_keys = {
        _tensor_storage_key(tensor) for tensor in candidate.values() if tensor.numel() != 0
    }
    reachable = _reachable_tensor_paths(cache)
    _assert_reachable_tensor_storage_closure(
        method=method,
        reachable=reachable,
        shared=shared,
        candidate=candidate,
    )
    return {
        "persistent_fp32_state_mirror": False,
        "persistent_raw_state_bytes": 0,
        "candidate_persistent_tensor_count": len(candidate),
        "candidate_persistent_storage_bytes": sum(key[3] for key in candidate_keys),
        "shared_persistent_tensor_count": len(shared),
        "shared_persistent_storage_bytes": sum(key[3] for key in shared_keys),
        "candidate_tensor_schema": {
            name: _tensor_schema(tensor) for name, tensor in sorted(candidate.items())
        },
        "shared_tensor_schema": shared_schema,
        "runtime_reachable_tensor_storage_closure_passed": True,
    }


def _validated_storage_summary(
    method: str,
    cache: object,
    reference_shared_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary = _storage_summary(cache)
    transactional = method in {
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    }
    if transactional and summary.get("forward_transaction_active") is not False:
        raise RuntimeError(f"{method} ended with an active cache transaction")
    if method == ORIGINAL_RHT_METHOD:
        if (
            summary.get("resident_bytes") != 2_564_096
            or summary.get("resident_bytes_including_selector") != 2_711_552
            or summary.get("high_precision_groups") != 1_976
        ):
            raise RuntimeError("historical RHT-CQER storage identity drifted")
        if reference_shared_schema is None:
            raise RuntimeError("historical RHT-CQER lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
    elif method == STATELEASE_METHOD:
        if summary.get("resident_bytes_including_statelease") != (FROZEN_STATELEASE_RESIDENT_BYTES):
            raise RuntimeError("StateLease did not occupy the exact frozen byte budget")
        if reference_shared_schema is None:
            raise RuntimeError("StateLease lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    elif method in FIXED_REPLAY_METHODS:
        expected = FROZEN_STATELEASE_RESIDENT_BYTES
        if (
            summary.get("resident_bytes_including_statelease") != expected
            or summary.get("logical_resident_capacity_bytes") != expected
            or summary.get("capacity_fully_allocated") is not True
            or summary.get("off_budget") is not False
        ):
            raise RuntimeError(f"{method} is not an exact-byte resident comparator")
        if reference_shared_schema is None:
            raise RuntimeError(f"{method} lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    elif method in EQUAL_BYTE_NO_REPLAY_METHODS:
        expected = FROZEN_STATELEASE_RESIDENT_BYTES
        if (
            summary.get("resident_bytes") != expected
            or summary.get("expected_resident_bytes") != expected
            or summary.get("persistent_raw_state_bytes") != 0
            or summary.get("checkpoint_present") is not True
        ):
            raise RuntimeError(f"{method} is not an exact-byte resident comparator")
        if reference_shared_schema is None:
            raise RuntimeError(f"{method} lacks the FP32 shared-storage schema")
        summary.update(
            _audit_persistent_raw_state(
                cache,
                method=method,
                reference_shared_schema=reference_shared_schema,
            )
        )
        summary["stage0_independently_verified_no_hidden_fp32_state_mirror"] = True
    else:
        raise ValueError(f"unknown Stage-A storage method {method!r}")
    summary["runtime_storage_contract_passed"] = True
    return summary


def _diagnostics(cache: object) -> list[dict[str, object]]:
    callback = getattr(cache, "statelease_diagnostics", None)
    if callable(callback):
        value = callback()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RuntimeError("StateLease diagnostics are not a sequence")
        return [_jsonable(dict(item)) for item in value]
    query_callback = getattr(cache, "query_ema_diagnostics", None)
    if callable(query_callback):
        value = query_callback()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_jsonable(dict(item)) for item in value]
    generic = {
        name: getattr(cache, name)
        for name in (
            "update_count",
            "successful_tokens",
            "raw_state_workspace_peak_bytes",
            "query_workspace_peak_bytes",
        )
        if hasattr(cache, name)
    }
    return [_jsonable(generic)] if generic else []


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("large tensors cannot be embedded in Stage-A JSON")
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Stage-A evidence cannot contain non-finite floats")
        return value
    raise TypeError(f"Stage-A evidence cannot serialize {type(value).__name__}")


def _cuda_measurement_start(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_before": int(torch.cuda.memory_allocated(device)),
        "reserved_before": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_measurement_end(
    device: torch.device,
    start: Mapping[str, int],
) -> dict[str, int]:
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        **dict(start),
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "incremental_peak_allocated": max(
            0,
            peak_allocated - int(start["allocated_before"]),
        ),
        "incremental_peak_reserved": max(
            0,
            peak_reserved - int(start["reserved_before"]),
        ),
    }


def _reference_run(
    model: torch.nn.Module,
    tokenized: TokenizedTask,
    device: torch.device,
) -> dict[str, object]:
    from transformers import DynamicCache

    prompt = tokenized.prompt_ids.to(device)
    code = tokenized.code_ids.to(device)
    cache = DynamicCache(config=model.config)
    measurement = _cuda_measurement_start(device)
    logits: list[torch.Tensor] = []
    states: list[dict[int, torch.Tensor]] = []
    reference_accumulator = _AlignedAccumulator.empty()
    with torch.inference_mode():
        model(
            prompt,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_states = _snapshot_states_cpu(cache)
        nonrecurrent_prefill = _snapshot_nonrecurrent_prefill(cache)
        for token_index in range(ALIGNED_TOKENS):
            input_token = code[:, token_index : token_index + 1]
            target = code[:, token_index + 1 : token_index + 2]
            output = model(
                input_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cpu_logits = output.logits.detach().to("cpu").clone()
            logits.append(cpu_logits)
            states.append(_snapshot_states_cpu(cache))
            reference_accumulator.append(
                token_index=token_index,
                reference_logits=cpu_logits,
                candidate_logits=cpu_logits,
                input_token=input_token,
                target=target,
            )
    cuda = _cuda_measurement_end(device, measurement)
    shared_storage_schema = _shared_cache_schema(cache)
    reference_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in prefill_states.values()
    )
    if reference_bytes != 18_874_368:
        raise RuntimeError(f"FP32 recurrent-state byte identity drifted: {reference_bytes}")
    return {
        "prefill_states": prefill_states,
        "nonrecurrent_prefill": nonrecurrent_prefill,
        "aligned_logits": logits,
        "aligned_states": states,
        "aligned_metrics": reference_accumulator.summary(),
        "per_token": reference_accumulator.records,
        "trajectory": {
            "trajectory_nmse_auc": 0.0,
            "scored_write_count": ALIGNED_TOKENS,
            "layer_value_count": ALIGNED_TOKENS * len(LINEAR_LAYER_INDICES),
        },
        "recurrent_state_bytes": reference_bytes,
        "cuda": cuda,
        "shared_storage_schema": shared_storage_schema,
    }


def _candidate_factory(
    method: str,
    model: torch.nn.Module,
    plan: object,
) -> tuple[object, object]:
    from recurquant.query_energy import Qwen35QueryEnergyObserver
    from recurquant.qwen35 import (
        create_qwen35_experiment010_fixed_replay_cache,
        create_qwen35_experiment010_statelease_cache,
        create_qwen35_right_rht_query_ema_exact_budget_cache,
    )
    from recurquant.statelease_equal_byte_baselines import (
        EXPANDED_RHT_Q4_Q8,
        RHT_Q4_Q6_Q8,
        RHT_RESIDUAL_Q4,
    )
    from recurquant.statelease_equal_byte_cache import (
        Qwen35EqualByteObserver,
        create_qwen35_equal_byte_cache,
    )
    from recurquant.statelease_observer import Qwen35StateLeaseObserver

    if method == ORIGINAL_RHT_METHOD:
        cache = create_qwen35_right_rht_query_ema_exact_budget_cache(
            model,
            plan=plan,
            record_evidence=True,
        )
        return cache, Qwen35QueryEnergyObserver(model, caches=[cache])
    if method == STATELEASE_METHOD:
        cache = create_qwen35_experiment010_statelease_cache(
            model,
            plan=plan,
            record_evidence=True,
        )
        return cache, Qwen35StateLeaseObserver(model, caches=[cache])
    if method in FIXED_REPLAY_METHODS:
        cache = create_qwen35_experiment010_fixed_replay_cache(
            model,
            plan=plan,
            mode=method,
            record_evidence=True,
        )
        return cache, Qwen35StateLeaseObserver(model, caches=[cache])
    codec_by_method = {
        "expanded_rht_q4_q8": EXPANDED_RHT_Q4_Q8,
        "rht_q4_q6_q8": RHT_Q4_Q6_Q8,
        "rht_residual_q4": RHT_RESIDUAL_Q4,
    }
    try:
        codec = codec_by_method[method]
    except KeyError as error:
        raise ValueError(f"unknown Stage-A method {method!r}") from error
    cache = create_qwen35_equal_byte_cache(
        model,
        codec=codec,
        record_evidence=True,
    )
    return cache, Qwen35EqualByteObserver(model, caches=[cache])


def _run_candidate(
    method: str,
    model: torch.nn.Module,
    tokenized: TokenizedTask,
    reference: Mapping[str, object],
    plan: object,
    device: torch.device,
) -> dict[str, object]:
    gc.collect()
    measurement = _cuda_measurement_start(device)
    cache, observer = _candidate_factory(method, model, plan)
    prompt = tokenized.prompt_ids.to(device)
    code = tokenized.code_ids.to(device)
    accumulator = _AlignedAccumulator.empty()
    trajectory = TrajectoryNmseAccumulator()
    trajectory_records: list[dict[str, object]] = []
    reference_prefill = reference["prefill_states"]
    reference_logits = reference["aligned_logits"]
    reference_states = reference["aligned_states"]
    reference_nonrecurrent = reference["nonrecurrent_prefill"]
    if not isinstance(reference_prefill, Mapping):
        raise RuntimeError("FP32 prefill reference is invalid")
    if not isinstance(reference_logits, list) or not isinstance(reference_states, list):
        raise RuntimeError("FP32 aligned reference is invalid")

    with torch.inference_mode(), observer:
        with _capture_prefill_writes(cache) as raw_prefill:
            model(
                prompt,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        prefill_receipt = _assert_shared_prefill(
            reference_prefill,
            raw_prefill,
            method=method,
        )
        if not isinstance(reference_nonrecurrent, Mapping):
            raise RuntimeError("FP32 non-recurrent prefill reference is invalid")
        candidate_nonrecurrent = _snapshot_nonrecurrent_prefill(cache)
        prefill_receipt.update(
            _assert_shared_nonrecurrent_prefill(
                reference_nonrecurrent,
                candidate_nonrecurrent,
                method=method,
            )
        )
        prefill_receipt.update(_validate_prefill_cache_contract(method, cache))
        for token_index in range(ALIGNED_TOKENS):
            target = code[:, token_index + 1 : token_index + 2]
            output = model(
                code[:, token_index : token_index + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            accumulator.append(
                token_index=token_index,
                reference_logits=reference_logits[token_index],
                candidate_logits=output.logits,
                input_token=code[:, token_index : token_index + 1],
                target=target,
            )
            candidate_states = _snapshot_states_cpu(cache)
            layer_nmse = reference_aligned_trajectory_nmse(
                reference_states[token_index],
                candidate_states,
            )
            trajectory.append(layer_nmse)
            trajectory_records.append(
                {
                    "write_index": token_index,
                    "per_layer_nmse": {
                        str(layer): value for layer, value in sorted(layer_nmse.items())
                    },
                    "layer_macro_nmse": sum(layer_nmse.values()) / len(layer_nmse),
                }
            )

    result = {
        "aligned_metrics": accumulator.summary(),
        "per_token_aligned": accumulator.records,
        "trajectory": trajectory.summary(),
        "trajectory_per_write": trajectory_records,
        "storage": _validated_storage_summary(
            method,
            cache,
            reference.get("shared_storage_schema")
            if isinstance(reference.get("shared_storage_schema"), Mapping)
            else None,
        ),
        "update_evidence": _evidence_records(cache),
        "diagnostics": _diagnostics(cache),
        "prefill": prefill_receipt,
        "cuda": _cuda_measurement_end(device, measurement),
    }
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_frozen_stage_a(
    loaded_model: tuple[torch.nn.Module, torch.device],
    tokenized: TokenizedTask,
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
) -> dict[str, object]:
    del configuration
    model, device = loaded_model
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    reference = _reference_run(model, tokenized, device)
    candidates: dict[str, dict[str, object]] = {}
    for method in QUALITY_METHODS:
        candidates[method] = _run_candidate(
            method,
            model,
            tokenized,
            reference,
            preflight.plan,
            device,
        )
    return {
        "reference": reference,
        "candidates": candidates,
        "device": str(device),
    }


def _assert_end_integrity(
    preflight: StageAPreflight,
    reservation: AttemptReservation,
) -> dict[str, object]:
    seal = _validate_one_run_seal(
        preflight,
        reservation,
        require_receipt=True,
    )
    repository_end = _repository_state(preflight.repo_root)
    source_hashes_end = _source_hashes(preflight.repo_root)
    if repository_end["worktree_clean"] is not True:
        raise StageAAuthenticationError("repository became dirty during Stage A")
    if repository_end["commit"] != reservation.seal_commit:
        raise StageAAuthenticationError("repository left the durable one-run seal")
    start_git_identity = preflight.repository_start.get("git_identity")
    if not isinstance(start_git_identity, Mapping) or repository_end.get("git_identity") != dict(
        start_git_identity
    ):
        raise StageAAuthenticationError("Git repository/object identity changed during Stage A")
    if source_hashes_end != preflight.source_hashes_start:
        raise StageAAuthenticationError("authenticated source hashes changed during Stage A")
    if _file_sha256(preflight.repo_root / EXPERIMENT009_STAGE_A_RELATIVE_PATH) != (
        EXPERIMENT009_STAGE_A_FILE_SHA256
    ):
        raise StageAAuthenticationError("Experiment 009 anchor changed during Stage A")
    if _file_sha256(preflight.repo_root / IDENTITY_NOTE_RELATIVE_PATH) != (
        IDENTITY_NOTE_FILE_SHA256
    ):
        raise StageAAuthenticationError("Stage-A identity clarification changed during Stage A")
    if _file_sha256(preflight.repo_root / PROTOCOL_NOTE_RELATIVE_PATH) != (
        PROTOCOL_NOTE_FILE_SHA256
    ):
        raise StageAAuthenticationError("Experiment 011 StateLease protocol changed during Stage A")
    selector_hashes = {
        "selector": _file_sha256(preflight.repo_root / SELECTOR_RELATIVE_PATH),
        "loss_selector": _file_sha256(preflight.repo_root / LOSS_SELECTOR_RELATIVE_PATH),
    }
    if selector_hashes != {
        "selector": SELECTOR_FILE_SHA256,
        "loss_selector": LOSS_SELECTOR_FILE_SHA256,
    }:
        raise StageAAuthenticationError("selector artifacts changed during Stage A")
    if _file_sha256(preflight.stage0_artifact) != preflight.stage0.get(
        "artifact_file_sha256"
    ) or _file_sha256(preflight.stage0_sha256) != preflight.stage0.get("sidecar_file_sha256"):
        raise StageAAuthenticationError("production Stage-0 identity changed during Stage A")
    experiment010_admin_null_end = authenticate_experiment010_administrative_null(
        preflight.repo_root
    )
    if experiment010_admin_null_end != preflight.experiment010_admin_null:
        raise StageAAuthenticationError(
            "Experiment 010 administrative-null provenance changed during Stage A"
        )
    readiness_start = _runtime_readiness_from_bundle(reservation.receipt.get("runtime_readiness"))
    preseal_freshness = _validated_preseal_freshness(reservation.receipt.get("preseal_freshness"))
    accelerator = readiness_start.receipt.get("accelerator")
    if not isinstance(accelerator, Mapping):
        raise StageAAuthenticationError("runtime readiness accelerator receipt is malformed")
    requested_device = accelerator.get("requested_device")
    if not isinstance(requested_device, str):
        raise StageAAuthenticationError("runtime readiness requested device is malformed")
    readiness_end = authenticate_runtime_readiness(device_name=requested_device)
    if _readiness_bundle(readiness_end) != _readiness_bundle(readiness_start):
        raise StageAAuthenticationError("runtime dependency/cache readiness changed during Stage A")
    return {
        "repository_end": repository_end,
        "source_hashes_end": source_hashes_end,
        "one_run_seal": seal,
        "stage0_file_hashes_reauthenticated": True,
        "stage0_independent_verification_completed_at_h0": True,
        "selectors_reauthenticated": True,
        "anchor_reauthenticated": True,
        "identity_clarification_reauthenticated": True,
        "experiment010_administrative_null": experiment010_admin_null_end,
        "experiment010_administrative_null_reauthenticated": True,
        "runtime_readiness": _readiness_bundle(readiness_end),
        "runtime_readiness_reauthenticated": True,
        "preseal_freshness": dict(preseal_freshness),
        "artifact_integrity": True,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in EXPECTED_RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_serialized_tensor_schemas(
    *,
    method: str,
    storage: Mapping[str, object],
) -> bool:
    expected_candidate_names = _expected_candidate_tensor_names(method)
    expected_candidate_count = len(expected_candidate_names)
    candidate_schema = storage.get("candidate_tensor_schema")
    shared_schema = storage.get("shared_tensor_schema")
    if (
        not isinstance(candidate_schema, Mapping)
        or set(candidate_schema) != expected_candidate_names
        or len(candidate_schema) != expected_candidate_count
        or storage.get("candidate_persistent_tensor_count") != expected_candidate_count
        or not isinstance(shared_schema, Mapping)
        or set(shared_schema) != EXPECTED_SHARED_PERSISTENT_TENSOR_NAMES
        or len(shared_schema) != EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT
        or storage.get("shared_persistent_tensor_count") != EXPECTED_SHARED_PERSISTENT_TENSOR_COUNT
    ):
        return False

    def valid_rows(
        rows: Mapping[object, object],
        *,
        allow_empty: bool,
        exact_storage: bool,
    ) -> tuple[bool, int]:
        element_sizes = {
            "torch.uint8": 1,
            "torch.int8": 1,
            "torch.float16": 2,
            "torch.bfloat16": 2,
            "torch.float32": 4,
            "torch.int32": 4,
            "torch.int64": 8,
        }
        storage_total = 0
        for name, raw_schema in rows.items():
            if not isinstance(name, str) or not isinstance(raw_schema, Mapping):
                return False, 0
            if set(raw_schema) != {
                "dtype",
                "shape",
                "logical_bytes",
                "storage_bytes",
            }:
                return False, 0
            dtype = raw_schema.get("dtype")
            shape = raw_schema.get("shape")
            logical_bytes = raw_schema.get("logical_bytes")
            storage_bytes = raw_schema.get("storage_bytes")
            if (
                not isinstance(dtype, str)
                or dtype not in element_sizes
                or not isinstance(shape, list)
                or any(
                    isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape
                )
                or isinstance(logical_bytes, bool)
                or not isinstance(logical_bytes, int)
                or logical_bytes < 0
                or isinstance(storage_bytes, bool)
                or not isinstance(storage_bytes, int)
                or storage_bytes < logical_bytes
                or (not allow_empty and logical_bytes == 0)
                or (exact_storage and storage_bytes != logical_bytes)
            ):
                return False, 0
            logical_elements = math.prod(shape)
            if logical_bytes != logical_elements * element_sizes[dtype]:
                return False, 0
            storage_total += storage_bytes
        return True, storage_total

    candidate_valid, candidate_total = valid_rows(
        candidate_schema,
        allow_empty=True,
        exact_storage=True,
    )
    shared_valid, shared_total = valid_rows(
        shared_schema,
        allow_empty=False,
        exact_storage=False,
    )
    return (
        candidate_valid
        and shared_valid
        and storage.get("candidate_persistent_storage_bytes") == candidate_total
        and storage.get("shared_persistent_storage_bytes") == shared_total
    )


def _validate_candidate_storage_results(
    candidates: Mapping[str, object],
) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for method in QUALITY_METHODS:
        candidate = candidates.get(method)
        if not isinstance(candidate, Mapping):
            raise RuntimeError(f"{method} candidate result is missing")
        storage = candidate.get("storage")
        if not isinstance(storage, Mapping):
            raise RuntimeError(f"{method} storage result is missing")
        if storage.get("runtime_storage_contract_passed") is not True:
            raise RuntimeError(f"{method} lacks a passed runtime storage contract")
        expected_candidate_bytes = (
            2_711_552 if method == ORIGINAL_RHT_METHOD else FROZEN_STATELEASE_RESIDENT_BYTES
        )
        common_passed = (
            storage.get("runtime_reachable_tensor_storage_closure_passed") is True
            and storage.get("candidate_persistent_storage_bytes") == expected_candidate_bytes
            and storage.get("persistent_raw_state_bytes") == 0
            and storage.get("persistent_fp32_state_mirror") is False
            and _validate_serialized_tensor_schemas(method=method, storage=storage)
        )
        if method == ORIGINAL_RHT_METHOD:
            passed = (
                common_passed
                and storage.get("resident_bytes") == 2_564_096
                and storage.get("resident_bytes_including_selector") == 2_711_552
                and storage.get("high_precision_groups") == 1_976
            )
        elif method == STATELEASE_METHOD:
            passed = (
                common_passed
                and storage.get("resident_bytes_including_statelease")
                == FROZEN_STATELEASE_RESIDENT_BYTES
            )
        elif method in FIXED_REPLAY_METHODS:
            passed = (
                common_passed
                and storage.get("resident_bytes_including_statelease")
                == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("logical_resident_capacity_bytes")
                == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("capacity_fully_allocated") is True
                and storage.get("off_budget") is False
            )
        else:
            passed = (
                common_passed
                and storage.get("resident_bytes") == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("expected_resident_bytes") == FROZEN_STATELEASE_RESIDENT_BYTES
                and storage.get("checkpoint_present") is True
            )
        if not passed:
            raise RuntimeError(f"{method} serialized storage contract drifted")
        receipts[method] = {
            "passed": True,
            "resident_bytes": (
                storage.get("resident_bytes_including_selector")
                if method == ORIGINAL_RHT_METHOD
                else storage.get(
                    "resident_bytes_including_statelease",
                    storage.get("resident_bytes"),
                )
            ),
        }
    return receipts


def _sanitized_command() -> list[str]:
    result = [Path(sys.executable).name, "scripts/screen_statelease_stage_a.py"]
    redact_next_path = False
    for argument in sys.argv[1:]:
        if redact_next_path:
            result.append(Path(argument).name)
            redact_next_path = False
            continue
        matched_inline_path = False
        for option in ("--stage0-artifact", "--stage0-sha256"):
            prefix = f"{option}="
            if argument.startswith(prefix):
                result.append(f"{prefix}{Path(argument[len(prefix) :]).name}")
                matched_inline_path = True
                break
        if matched_inline_path:
            continue
        result.append(argument)
        if argument in {"--stage0-artifact", "--stage0-sha256"}:
            redact_next_path = True
    return result


def _validate_public_artifact(value: object, *, repo_root: Path) -> None:
    forbidden_roots = {
        str(repo_root.resolve()).casefold(),
        str(Path.home().resolve()).casefold(),
    }
    sensitive_key_fragments = (
        "api_key",
        "access_token",
        "auth_token",
        "password",
        "private_key",
        "client_secret",
        "cookie",
    )
    secret_value_prefixes = (
        "sk-",
        "sk_proj-",
        "ghp_",
        "github_pat_",
        "hf_",
        "bearer ",
    )

    def walk(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized_key = str(key).casefold()
                if any(fragment in normalized_key for fragment in sensitive_key_fragments):
                    raise StageAAuthenticationError(
                        f"public artifact contains a sensitive field at {path}.{key}"
                    )
                walk(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if not isinstance(item, str):
            return
        normalized = item.strip().casefold()
        if (
            normalized.startswith(("\\\\", "/", "file://"))
            or (
                len(normalized) >= 3
                and normalized[0].isalpha()
                and normalized[1] == ":"
                and normalized[2] in {"\\", "/"}
            )
            or any(root and root in normalized for root in forbidden_roots)
        ):
            raise StageAAuthenticationError(
                f"public artifact contains an absolute local path at {path}"
            )
        if any(normalized.startswith(prefix) for prefix in secret_value_prefixes):
            raise StageAAuthenticationError(
                f"public artifact contains a secret-like value at {path}"
            )

    walk(value, "$")


def _strict_record_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} is not a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{context} is non-finite")
    return result


def _assert_exact_summary(
    *,
    method: str,
    kind: str,
    stored: object,
    recomputed: Mapping[str, object],
) -> None:
    if not isinstance(stored, Mapping) or set(stored) != set(recomputed):
        raise RuntimeError(f"{method} stored {kind} summary schema drifted")
    for key, expected in recomputed.items():
        actual = stored.get(key)
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
            )
        else:
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and float(actual) == float(expected)
            )
        if not matches:
            raise RuntimeError(
                f"{method} stored {kind} scalar {key} does not exactly match raw evidence"
            )


def _recompute_aligned_summary(
    *,
    method: str,
    rows: object,
    reference_rows: Sequence[Mapping[str, object]] | None,
    expected_code_ids: Sequence[int] | None,
    reference_method: bool,
) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != ALIGNED_TOKENS
    ):
        raise RuntimeError(
            f"{method} per-token evidence does not contain exactly {ALIGNED_TOKENS} rows"
        )
    expected_keys = {
        "write_index",
        "input_token_id",
        "target_token_id",
        "kl",
        "reference_nll",
        "candidate_nll",
        "top1_agreement",
        "delta_nll",
        "all_logits_finite",
    }
    normalized: list[Mapping[str, object]] = []
    kl_values: list[float] = []
    reference_nll_values: list[float] = []
    candidate_nll_values: list[float] = []
    top1_values: list[bool] = []
    finite_values: list[bool] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_keys:
            raise RuntimeError(f"{method} per-token evidence schema drifted at write {index}")
        write_index = raw_row.get("write_index")
        input_token_id = raw_row.get("input_token_id")
        target_token_id = raw_row.get("target_token_id")
        if (
            isinstance(write_index, bool)
            or not isinstance(write_index, int)
            or write_index != index
            or isinstance(input_token_id, bool)
            or not isinstance(input_token_id, int)
            or isinstance(target_token_id, bool)
            or not isinstance(target_token_id, int)
        ):
            raise RuntimeError(f"{method} per-token evidence is not ordered at write {index}")
        if expected_code_ids is not None and (
            input_token_id != expected_code_ids[index]
            or target_token_id != expected_code_ids[index + 1]
        ):
            raise RuntimeError(f"{method} per-token token identity drifted at write {index}")

        kl = _strict_record_float(raw_row.get("kl"), context=f"{method}[{index}].kl")
        reference_nll = _strict_record_float(
            raw_row.get("reference_nll"),
            context=f"{method}[{index}].reference_nll",
        )
        candidate_nll = _strict_record_float(
            raw_row.get("candidate_nll"),
            context=f"{method}[{index}].candidate_nll",
        )
        delta_nll = _strict_record_float(
            raw_row.get("delta_nll"),
            context=f"{method}[{index}].delta_nll",
        )
        top1 = raw_row.get("top1_agreement")
        finite = raw_row.get("all_logits_finite")
        if not isinstance(top1, bool) or not isinstance(finite, bool) or not finite:
            raise RuntimeError(f"{method} per-token boolean evidence drifted at write {index}")
        if delta_nll != candidate_nll - reference_nll:
            raise RuntimeError(
                f"{method} per-token delta_nll does not match raw NLLs at write {index}"
            )
        if reference_rows is not None:
            reference_row = reference_rows[index]
            if (
                input_token_id != reference_row.get("input_token_id")
                or target_token_id != reference_row.get("target_token_id")
                or reference_nll != reference_row.get("reference_nll")
            ):
                raise RuntimeError(
                    f"{method} per-token reference alignment drifted at write {index}"
                )
        if reference_method and (
            kl != 0.0 or candidate_nll != reference_nll or delta_nll != 0.0 or top1 is not True
        ):
            raise RuntimeError(f"{method} self-reference evidence drifted at write {index}")

        normalized.append(raw_row)
        kl_values.append(kl)
        reference_nll_values.append(reference_nll)
        candidate_nll_values.append(candidate_nll)
        top1_values.append(top1)
        finite_values.append(finite)

    summary = fidelity_summary(
        TokenFidelity(
            kl=torch.tensor(kl_values, dtype=torch.float32),
            reference_nll=torch.tensor(reference_nll_values, dtype=torch.float32),
            candidate_nll=torch.tensor(candidate_nll_values, dtype=torch.float32),
            top1_agreement=torch.tensor(top1_values, dtype=torch.bool),
        )
    )
    return {**summary, "all_logits_finite": all(finite_values)}, normalized


def _recompute_trajectory_summary(
    *,
    method: str,
    rows: object,
) -> dict[str, object]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != ALIGNED_TOKENS
    ):
        raise RuntimeError(
            f"{method} trajectory evidence does not contain exactly {ALIGNED_TOKENS} writes"
        )
    expected_layer_keys = {str(layer) for layer in LINEAR_LAYER_INDICES}
    accumulator = TrajectoryNmseAccumulator()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "write_index",
            "per_layer_nmse",
            "layer_macro_nmse",
        }:
            raise RuntimeError(f"{method} trajectory evidence schema drifted at write {index}")
        per_layer = raw_row.get("per_layer_nmse")
        if (
            raw_row.get("write_index") != index
            or not isinstance(per_layer, Mapping)
            or set(per_layer) != expected_layer_keys
        ):
            raise RuntimeError(f"{method} trajectory evidence is incomplete at write {index}")
        layer_values = {
            layer: _strict_record_float(
                per_layer[str(layer)],
                context=f"{method}[{index}].trajectory[{layer}]",
            )
            for layer in LINEAR_LAYER_INDICES
        }
        if any(value < 0 for value in layer_values.values()):
            raise RuntimeError(f"{method} trajectory evidence is negative at write {index}")
        macro = _strict_record_float(
            raw_row.get("layer_macro_nmse"),
            context=f"{method}[{index}].layer_macro_nmse",
        )
        if macro != sum(layer_values.values()) / len(layer_values):
            raise RuntimeError(
                f"{method} trajectory macro does not match per-layer evidence at write {index}"
            )
        accumulator.append(layer_values)
    return dict(accumulator.summary())


def _recompute_and_reconcile_stage_a_summaries(
    result: Mapping[str, object],
    *,
    code_ids: torch.Tensor | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    candidates = result.get("candidates")
    reference = result.get("reference")
    if not isinstance(candidates, Mapping) or set(candidates) != set(QUALITY_METHODS):
        raise RuntimeError("Stage-A candidate result set is incomplete")
    if not isinstance(reference, Mapping):
        raise RuntimeError("Stage-A reference result is missing")
    expected_code_ids: list[int] | None = None
    if code_ids is not None:
        if code_ids.dtype != torch.long or tuple(code_ids.shape) != (1, CODE_TOKENS):
            raise RuntimeError("authenticated code-token identity is malformed")
        expected_code_ids = [
            int(value) for value in code_ids.detach().to("cpu").reshape(-1).tolist()
        ]

    reference_summary, reference_rows = _recompute_aligned_summary(
        method=FP32_METHOD,
        rows=reference.get("per_token"),
        reference_rows=None,
        expected_code_ids=expected_code_ids,
        reference_method=True,
    )
    _assert_exact_summary(
        method=FP32_METHOD,
        kind="aligned",
        stored=reference.get("aligned_metrics"),
        recomputed=reference_summary,
    )
    reference_trajectory = {
        "trajectory_nmse_auc": 0.0,
        "scored_write_count": ALIGNED_TOKENS,
        "layer_value_count": ALIGNED_TOKENS * len(LINEAR_LAYER_INDICES),
    }
    _assert_exact_summary(
        method=FP32_METHOD,
        kind="trajectory",
        stored=reference.get("trajectory"),
        recomputed=reference_trajectory,
    )

    aligned = {FP32_METHOD: reference_summary}
    trajectories = {FP32_METHOD: reference_trajectory}
    for method in QUALITY_METHODS:
        method_result = candidates[method]
        if not isinstance(method_result, Mapping):
            raise RuntimeError(f"{method} Stage-A result is malformed")
        summary, _rows = _recompute_aligned_summary(
            method=method,
            rows=method_result.get("per_token_aligned"),
            reference_rows=reference_rows,
            expected_code_ids=expected_code_ids,
            reference_method=False,
        )
        _assert_exact_summary(
            method=method,
            kind="aligned",
            stored=method_result.get("aligned_metrics"),
            recomputed=summary,
        )
        trajectory = _recompute_trajectory_summary(
            method=method,
            rows=method_result.get("trajectory_per_write"),
        )
        _assert_exact_summary(
            method=method,
            kind="trajectory",
            stored=method_result.get("trajectory"),
            recomputed=trajectory,
        )
        aligned[method] = summary
        trajectories[method] = trajectory
    return aligned, trajectories


def _validate_stage_a_result_completeness(
    result: Mapping[str, object],
    *,
    code_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    candidates = result.get("candidates")
    reference = result.get("reference")
    if not isinstance(candidates, Mapping) or set(candidates) != set(QUALITY_METHODS):
        raise RuntimeError("Stage-A candidate result set is incomplete")
    if not isinstance(reference, Mapping):
        raise RuntimeError("Stage-A reference result is missing")

    _recompute_and_reconcile_stage_a_summaries(result, code_ids=code_ids)

    statelease = candidates[STATELEASE_METHOD]
    if not isinstance(statelease, Mapping):
        raise RuntimeError("StateLease result is malformed")
    diagnostics = statelease.get("diagnostics")
    if (
        not isinstance(diagnostics, Sequence)
        or isinstance(diagnostics, (str, bytes))
        or len(diagnostics) != len(LINEAR_LAYER_INDICES)
    ):
        raise RuntimeError("StateLease diagnostics do not cover the frozen layer set")
    if {row.get("layer_index") for row in diagnostics if isinstance(row, Mapping)} != set(
        LINEAR_LAYER_INDICES
    ):
        raise RuntimeError("StateLease diagnostic layer identity is incomplete")

    update_evidence = statelease.get("update_evidence")
    expected_evidence_count = (1 + ALIGNED_TOKENS) * len(LINEAR_LAYER_INDICES)
    if (
        not isinstance(update_evidence, Sequence)
        or isinstance(update_evidence, (str, bytes))
        or len(update_evidence) != expected_evidence_count
    ):
        raise RuntimeError("StateLease update evidence does not cover every frozen layer-write")
    for index, row in enumerate(update_evidence):
        expected_layer = LINEAR_LAYER_INDICES[index % len(LINEAR_LAYER_INDICES)]
        forward_index = index // len(LINEAR_LAYER_INDICES)
        expected_token_count = PROMPT_TOKENS if forward_index == 0 else 1
        if (
            not isinstance(row, Mapping)
            or row.get("update_index") != index
            or row.get("layer_index") != expected_layer
            or row.get("state_index") != 0
            or row.get("token_count") != expected_token_count
        ):
            raise RuntimeError(
                f"StateLease update evidence is incomplete or misordered at row {index}"
            )

    return {
        "method_count": len(ALL_METHODS),
        "authenticated_forward_passes": FROZEN_STAGE_A_FORWARD_PASSES,
        "per_token_rows_per_method": ALIGNED_TOKENS,
        "trajectory_writes_per_method": ALIGNED_TOKENS,
        "trajectory_layers_per_write": len(LINEAR_LAYER_INDICES),
        "statelease_update_evidence_records": expected_evidence_count,
        "aligned_summaries_recomputed_from_per_token_evidence": True,
        "aligned_aggregation_semantics": "recurquant.evaluation.fidelity_summary_fp32",
        "trajectory_summaries_recomputed_from_per_layer_write_evidence": True,
        "trajectory_aggregation_semantics": (
            "recurquant.statelease_evaluation.TrajectoryNmseAccumulator_fp64_neumaier"
        ),
        "stored_aggregates_exactly_reconciled": True,
        "all_expected_records_present_and_ordered": True,
    }


def _evaluate_gate_from_recomputed_evidence(
    *,
    candidates: Mapping[str, object],
    recomputed_metrics: Mapping[str, Mapping[str, object]],
    recomputed_trajectory: Mapping[str, Mapping[str, object]],
    stage0_complete: bool,
    artifact_integrity: bool,
) -> dict[str, object]:
    gate_methods = (
        ORIGINAL_RHT_METHOD,
        STATELEASE_METHOD,
        *FIXED_REPLAY_METHODS,
        *EQUAL_BYTE_NO_REPLAY_METHODS,
    )
    if set(recomputed_metrics) != set(ALL_METHODS) or set(recomputed_trajectory) != set(
        ALL_METHODS
    ):
        raise RuntimeError("recomputed Stage-A gate evidence method set drifted")
    statelease = candidates.get(STATELEASE_METHOD)
    if not isinstance(statelease, Mapping):
        raise RuntimeError("StateLease result is malformed")
    return evaluate_statelease_stage_a_gate(
        aligned_metrics={method: recomputed_metrics[method] for method in gate_methods},
        trajectory_nmse_auc={method: recomputed_trajectory[method] for method in gate_methods},
        statelease_storage=statelease["storage"],
        statelease_diagnostics=statelease["diagnostics"],
        statelease_update_evidence=statelease["update_evidence"],
        stage0_complete=stage0_complete,
        artifact_integrity=artifact_integrity,
    )


def _build_artifact(
    result: Mapping[str, object],
    tokenized: TokenizedTask,
    preflight: StageAPreflight,
    configuration: ModelConfiguration,
    integrity: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    completeness = _validate_stage_a_result_completeness(
        result,
        code_ids=tokenized.code_ids,
    )
    recomputed_metrics, recomputed_trajectory = _recompute_and_reconcile_stage_a_summaries(
        result,
        code_ids=tokenized.code_ids,
    )
    candidates = result.get("candidates")
    reference = result.get("reference")
    if not isinstance(candidates, Mapping) or set(candidates) != set(QUALITY_METHODS):
        raise RuntimeError("Stage-A candidate result set is incomplete")
    if not isinstance(reference, Mapping):
        raise RuntimeError("Stage-A reference result is missing")
    storage_contracts = _validate_candidate_storage_results(candidates)

    gate = _evaluate_gate_from_recomputed_evidence(
        candidates=candidates,
        recomputed_metrics=recomputed_metrics,
        recomputed_trajectory=recomputed_trajectory,
        stage0_complete=preflight.stage0.get("experiment_stage0_complete") is True,
        artifact_integrity=integrity.get("artifact_integrity") is True,
    )

    per_token = {
        FP32_METHOD: reference["per_token"],
        **{method: candidates[method]["per_token_aligned"] for method in QUALITY_METHODS},
    }
    aligned_metrics = recomputed_metrics
    trajectory = recomputed_trajectory
    trajectory_per_write = {
        FP32_METHOD: [
            {
                "write_index": index,
                "per_layer_nmse": {str(layer): 0.0 for layer in LINEAR_LAYER_INDICES},
                "layer_macro_nmse": 0.0,
            }
            for index in range(ALIGNED_TOKENS)
        ],
        **{method: candidates[method]["trajectory_per_write"] for method in QUALITY_METHODS},
    }

    evidence: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "screening_only": True,
        "one_authenticated_quality_run": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol": {
            "name": "Experiment 011 Stage A",
            "method": "StateLease-H5",
            "task_locked": True,
            "thresholds_locked": True,
            "output_path_locked": OUTPUT_RELATIVE_PATH,
            "protected_ranked_window": [8, 16],
            "protected_window_accessed": False,
        },
        "input_authentication": {
            "stage_a_identity_clarification": preflight.identity_clarification,
            "experiment010_administrative_null": integrity["experiment010_administrative_null"],
            "runtime_readiness": integrity["runtime_readiness"],
            "preseal_freshness": integrity["preseal_freshness"],
            "experiment009_stage_a": {
                "path": preflight.anchor["path"],
                "file_sha256": preflight.anchor["file_sha256"],
                "canonical_evidence_sha256": preflight.anchor["canonical_evidence_sha256"],
            },
            "selectors": preflight.selector_identity,
            "production_stage0": preflight.stage0,
            "all_passed_before_one_run_seal_quality_data_or_model_weights": True,
        },
        "model": {
            **configuration.identity,
            "device": result["device"],
        },
        "dataset": {
            "phase": "calibration",
            "selection_mode": "exact_already_open_task_id",
            "identity": EXPECTED_TASK_IDENTITY,
            "manifest_copied_from_authenticated_experiment009": preflight.anchor[
                "dataset_manifest"
            ],
            "manifest_sha256": EXPERIMENT009_MANIFEST_SHA256,
            "token_manifest_copied_from_authenticated_experiment009": preflight.anchor[
                "token_manifest"
            ],
            "runtime_token_manifest": tokenized.token_manifest,
            "runtime_text_sha256": {
                "prompt": PROMPT_TEXT_SHA256,
                "code": CODE_TEXT_SHA256,
            },
            "runtime_token_ids": _validate_token_id_hashes(
                tokenized.prompt_ids,
                tokenized.code_ids,
            ),
            "protected_window_8_16_loaded_tokenized_or_evaluated": False,
        },
        "metric_contract": {
            "aligned_primary": "one-token decode writes after the shared FP32 prefill",
            "aligned_token_count": ALIGNED_TOKENS,
            "statelease_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
            "authenticated_forward_passes": FROZEN_STAGE_A_FORWARD_PASSES,
            "reference_aligned_trajectory": (
                "per-layer FP64 NMSE against the matched FP32 recurrent state at every write"
            ),
        },
        "result_completeness": completeness,
        "methods": list(ALL_METHODS),
        "metrics_aligned": aligned_metrics,
        "per_token_aligned": per_token,
        "trajectory_nmse": trajectory,
        "trajectory_nmse_per_layer_write": trajectory_per_write,
        "storage": {
            FP32_METHOD: {
                "recurrent_state_bytes": reference["recurrent_state_bytes"],
            },
            **{method: candidates[method]["storage"] for method in QUALITY_METHODS},
        },
        "storage_contracts": storage_contracts,
        "prefill_identity": {method: candidates[method]["prefill"] for method in QUALITY_METHODS},
        "update_evidence": {
            method: candidates[method]["update_evidence"] for method in QUALITY_METHODS
        },
        "diagnostics": {method: candidates[method]["diagnostics"] for method in QUALITY_METHODS},
        "cuda_memory": {
            FP32_METHOD: reference["cuda"],
            **{method: candidates[method]["cuda"] for method in QUALITY_METHODS},
            "scope": (
                "PyTorch allocator peaks for the correctness-first Python path; "
                "not a deployment or end-to-end process-memory claim"
            ),
            "model_load_peak_included": False,
            "tokenization_and_dataset_peak_included": False,
            "end_to_end_process_peak_available": False,
            "temporary_workspace_interpretation": (
                "method-scoped PyTorch CUDA allocator peaks and cache-reported "
                "workspace peaks only; untracked native or process-wide peaks are unavailable"
            ),
        },
        "stage_a_gate": gate,
        "repository": {
            "h0_commit": preflight.repository_start["commit"],
            "one_run_seal_commit": integrity["one_run_seal"]["seal_commit"],
            "one_run_seal_tree": integrity["one_run_seal"]["tree"],
            "start": preflight.repository_start,
            "end": integrity["repository_end"],
            "commit_transition_authorized": True,
            "empty_tree_one_run_seal": True,
            "source_tree_stable": True,
        },
        "source_files": {
            "paths": list(SOURCE_FILES),
            "loaded_local_modules": list(
                _assert_loaded_local_modules_declared(preflight.repo_root)
            ),
            "sha256_start": preflight.source_hashes_start,
            "sha256_end": integrity["source_hashes_end"],
            "stable": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(result["device"])),
        },
        "command": _sanitized_command(),
    }
    json_evidence = _jsonable(evidence)
    if not isinstance(json_evidence, dict):
        raise TypeError("Stage-A evidence did not normalize to a JSON object")
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "canonical_evidence_sha256": _canonical_sha256(json_evidence),
        "evidence": json_evidence,
    }
    return artifact, gate


def _atomic_publish_new(path: Path, payload: bytes) -> None:
    """Atomically publish a new file without replacing an existing result."""

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
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise StageAAuthenticationError(
                f"refusing to overwrite one-run artifact {path.name}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_completed_artifact(
    output_parent: Path,
    artifact: Mapping[str, object],
) -> tuple[bytes, str, str]:
    payload = canonical_json_bytes(artifact)
    output_parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".experiment011-stage-a-verify.",
        suffix=".json",
        dir=output_parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        verification = verify_evidence_artifact(temporary)
        if verification.get("valid") is not True:
            raise RuntimeError("temporary Stage-A artifact failed canonical verification")
        file_hash = hashlib.sha256(payload).hexdigest()
        canonical_hash = str(artifact["canonical_evidence_sha256"])
        if (
            verification.get("file_sha256") != file_hash
            or verification.get("computed_canonical_evidence_sha256") != canonical_hash
        ):
            raise RuntimeError("temporary Stage-A verification receipt drifted")
        return payload, file_hash, canonical_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_result_completion_receipt(
    preflight: StageAPreflight,
    attempt: AttemptReservation,
    *,
    file_hash: str,
    canonical_hash: str,
    gate: Mapping[str, object],
) -> dict[str, object]:
    _validate_one_run_seal(preflight, attempt, require_receipt=True)
    json_gate = _jsonable(gate)
    if not isinstance(json_gate, dict) or not isinstance(json_gate.get("passed"), bool):
        raise StageAAuthenticationError("Stage-A gate is malformed before completion preparation")
    prepared = {
        **attempt.receipt,
        "status": RESULT_PREPARED_STATUS,
        "completion_prepared_at_utc": datetime.now(UTC).isoformat(),
        "completed_task_ids": [TASK_ID],
        "quality_result_computed": True,
        "result_available": False,
        "output_published": False,
        "quality_aggregate_exposed": False,
        "intended_output_path": OUTPUT_RELATIVE_PATH,
        "output_file_sha256": file_hash,
        "output_canonical_evidence_sha256": canonical_hash,
        "stage_a_gate_sha256": _canonical_sha256(json_gate),
        "rerun_automatically_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _validate_public_artifact(prepared, repo_root=preflight.repo_root)
    payload = canonical_json_bytes(prepared)
    _atomic_replace_owned(preflight.attempt_path, payload)
    if preflight.attempt_path.read_bytes() != payload:
        raise StageAAuthenticationError("prepared completion receipt persistence drifted")
    return prepared


def _validate_published_output_against_receipt(
    output_path: Path,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    expected_file_hash = receipt.get("output_file_sha256")
    expected_canonical_hash = receipt.get("output_canonical_evidence_sha256")
    if not isinstance(expected_file_hash, str) or not isinstance(
        expected_canonical_hash,
        str,
    ):
        raise StageAAuthenticationError("completion receipt lacks authenticated output hashes")
    try:
        payload = output_path.read_bytes()
    except OSError as error:
        raise StageAAuthenticationError(
            "prepared Stage-A output is absent after publication"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_file_hash:
        raise StageAAuthenticationError("published Stage-A output file hash drifted")
    verification = verify_evidence_artifact(output_path)
    if (
        verification.get("valid") is not True
        or verification.get("file_sha256") != expected_file_hash
        or verification.get("computed_canonical_evidence_sha256") != expected_canonical_hash
    ):
        raise StageAAuthenticationError("published Stage-A output canonical verification failed")
    artifact = _json_mapping(output_path)
    if (
        artifact.get("artifact_kind") != ARTIFACT_KIND
        or artifact.get("canonical_evidence_sha256") != expected_canonical_hash
    ):
        raise StageAAuthenticationError("published Stage-A output identity drifted")
    evidence = artifact.get("evidence")
    gate = evidence.get("stage_a_gate") if isinstance(evidence, Mapping) else None
    expected_gate_hash = receipt.get("stage_a_gate_sha256")
    if (
        not isinstance(gate, Mapping)
        or not isinstance(gate.get("passed"), bool)
        or not isinstance(expected_gate_hash, str)
        or _canonical_sha256(gate) != expected_gate_hash
    ):
        raise StageAAuthenticationError(
            "published Stage-A gate does not match its non-revealing receipt binding"
        )
    return {
        "artifact_file_sha256": expected_file_hash,
        "canonical_evidence_sha256": expected_canonical_hash,
        "stage_a_gate_passed": gate["passed"],
        "stage_a_gate_sha256": expected_gate_hash,
        "valid": True,
    }


def _promote_result_completion_receipt(
    preflight: StageAPreflight,
    prepared: Mapping[str, object],
) -> dict[str, object]:
    expected_prepared = canonical_json_bytes(prepared)
    try:
        actual_prepared = preflight.attempt_path.read_bytes()
    except OSError as error:
        raise StageAAuthenticationError(
            "prepared completion receipt is missing before promotion"
        ) from error
    if actual_prepared != expected_prepared:
        raise StageAAuthenticationError(
            "prepared completion receipt bytes drifted before promotion"
        )
    publication = _validate_published_output_against_receipt(
        preflight.output_path,
        prepared,
    )
    completed = {
        **prepared,
        "status": RESULT_COMPLETED_STATUS,
        "result_available": True,
        "output_published": True,
        "quality_aggregate_exposed": True,
        "stage_a_gate_passed": publication["stage_a_gate_passed"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _validate_public_artifact(completed, repo_root=preflight.repo_root)
    payload = canonical_json_bytes(completed)
    _atomic_replace_owned(preflight.attempt_path, payload)
    if preflight.attempt_path.read_bytes() != payload:
        raise StageAAuthenticationError("completed result receipt promotion drifted")
    return completed


def finalize_stage_a(
    result: object,
    loaded_model: object,
    tokenized: object,
    authenticated: object,
    configuration: object,
    attempt: object,
) -> dict[str, object]:
    del loaded_model
    if not isinstance(result, Mapping):
        raise TypeError("Stage-A evaluation result must be a mapping")
    if not isinstance(tokenized, TokenizedTask):
        raise TypeError("Stage-A token identity is invalid")
    if not isinstance(authenticated, StageAPreflight):
        raise TypeError("Stage-A authentication bundle is invalid")
    if not isinstance(configuration, ModelConfiguration):
        raise TypeError("Stage-A model configuration is invalid")
    if not isinstance(attempt, AttemptReservation):
        raise TypeError("Stage-A attempt receipt is invalid")
    integrity = _assert_end_integrity(authenticated, attempt)
    artifact, gate = _build_artifact(
        result,
        tokenized,
        authenticated,
        configuration,
        integrity,
    )
    _validate_public_artifact(artifact, repo_root=authenticated.repo_root)
    payload, file_hash, canonical_hash = _prepare_completed_artifact(
        authenticated.output_path.parent,
        artifact,
    )
    final_integrity = _assert_end_integrity(authenticated, attempt)
    if (
        final_integrity["repository_end"] != integrity["repository_end"]
        or final_integrity["source_hashes_end"] != integrity["source_hashes_end"]
    ):
        raise StageAAuthenticationError(
            "repository identity changed while preparing the Stage-A artifact"
        )
    prepared_completion = _prepare_result_completion_receipt(
        authenticated,
        attempt,
        file_hash=file_hash,
        canonical_hash=canonical_hash,
        gate=gate,
    )
    _atomic_publish_new(authenticated.output_path, payload)
    _validate_published_output_against_receipt(
        authenticated.output_path,
        prepared_completion,
    )
    _promote_result_completion_receipt(authenticated, prepared_completion)
    return {
        "output": OUTPUT_RELATIVE_PATH,
        "artifact_file_sha256": file_hash,
        "canonical_evidence_sha256": canonical_hash,
        "stage_a_gate": gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def execute_stage_a(args: argparse.Namespace) -> dict[str, object]:
    holder: dict[str, StageAPreflight] = {}

    def authenticate() -> StageAPreflight:
        value = authenticate_static_inputs(args)
        holder["preflight"] = value
        return value

    hooks = AccessHooks(
        authenticate=authenticate,
        authenticate_readiness=lambda _preflight: authenticate_runtime_readiness(
            device_name=args.device,
        ),
        load_config=lambda preflight: load_and_authenticate_config(
            preflight,
            local_files_only=args.local_files_only,
        ),
        reserve_attempt=lambda preflight, configuration, readiness: reserve_one_run(
            preflight,
            configuration,
            readiness,
        ),
        load_exact_task=load_exact_authenticated_task,
        tokenize_task=lambda preflight, row: tokenize_authenticated_task(
            preflight,
            row,
            local_files_only=args.local_files_only,
        ),
        load_weights=lambda configuration, preflight: load_model_weights(
            configuration,
            preflight,
            device_name=args.device,
            local_files_only=args.local_files_only,
        ),
        evaluate=evaluate_frozen_stage_a,
        record_access_transition=lambda receipt, ledger: record_access_transition(
            holder["preflight"],
            receipt,
            ledger,
        ),
        record_evaluation_returned=lambda receipt, ledger: record_evaluation_returned(
            holder["preflight"],
            receipt,
            ledger,
        ),
        finalize=finalize_stage_a,
        record_failure=lambda receipt, error, ledger: record_attempt_failure(
            holder["preflight"],
            receipt,
            error,
            ledger,
        ),
    )
    result = run_ordered_access(hooks)
    if not isinstance(result, dict):
        raise RuntimeError("Stage-A execution returned an invalid report")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage0-artifact",
        type=Path,
        required=True,
        help="Authenticated production Stage-0 torch artifact.",
    )
    parser.add_argument(
        "--stage0-sha256",
        type=Path,
        help="Optional explicit Stage-0 SHA-256 sidecar.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda"),
        default="auto",
        help="Frozen Stage A requires CUDA BF16; auto resolves only to CUDA.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        required=True,
        help="Required fail-closed mode; all model, tokenizer, and dataset resources stay local.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Authenticate artifacts, sources, and model config without data or weights.",
    )
    mode.add_argument(
        "--execute-frozen-stage-a",
        action="store_true",
        help="Irreversibly reserve and execute the one permitted Stage-A quality run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        preflight = authenticate_static_inputs(args)
        readiness = authenticate_runtime_readiness(device_name=args.device)
        configuration = load_and_authenticate_config(
            preflight,
            local_files_only=args.local_files_only,
        )
        print(
            json.dumps(
                {
                    "status": "stage_a_preflight_pass",
                    "quality_data_accessed": False,
                    "model_weights_loaded": False,
                    "one_run_reserved": False,
                    "task_row_loaded": False,
                    "tokenizer_loaded": False,
                    "repository_commit": preflight.repository_start["commit"],
                    "runtime_readiness": _readiness_bundle(readiness),
                    "experiment010_administrative_null": (preflight.experiment010_admin_null),
                    "model": configuration.identity,
                    "output": OUTPUT_RELATIVE_PATH,
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = execute_stage_a(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stage_a_gate"]["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
