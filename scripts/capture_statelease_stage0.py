#!/usr/bin/env python
"""Capture authenticated production receipts for Experiment 012 Stage 0.

The producer is intentionally allowed to import RecurQuant.  It runs only
deterministic synthetic recurrent-state transitions, serializes a closed
tensor schema, and writes both a canonical payload digest and a SHA-256
sidecar for the serialized file.  The independent verifier in
``verify_statelease_stage0.py`` is responsible for deciding whether these
receipts complete Stage 0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from recurquant.qwen35 import (
    EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256,
    EXPERIMENT010_STATELEASE_LAYER_QUOTAS,
    create_qwen35_experiment010_fixed_replay_cache,
    create_qwen35_experiment010_statelease_cache,
    create_qwen35_right_rht_query_ema_exact_budget_cache,
    create_qwen35_statelease_cache,
    experiment010_statelease_effective_plan_sha256,
)
from recurquant.row_policy import (
    ExactBudgetRowPlan,
    RowLocation,
    select_rows_exact_budget,
)
from recurquant.statelease import replay_gated_delta_state
from recurquant.statelease_cache import (
    STATELEASE_SELECTION_METHOD,
    StateLeaseLinearAttentionLayer,
    StateLeaseRecurrentStateCache,
)
from recurquant.statelease_equal_byte_baselines import (
    EXPANDED_RHT_Q4_Q8,
    FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    RHT_Q4_Q6_Q8,
    RHT_RESIDUAL_Q4,
    pack_expanded_rht_q4_q8,
    pack_rht_q4_q6_q8,
    pack_rht_residual_q4,
)
from recurquant.statelease_observer import Qwen35StateLeaseObserver

EXPERIMENT_ID = "experiment012"
SCHEMA_NAME = f"recurquant.{EXPERIMENT_ID}.stage0.production.v1"
SCHEMA_VERSION = 1
FROZEN_SEED = 2339
PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256 = (
    "110c81fe9d47e73de06e01ce3435377709beebda81eb0858418bfbf173bb16df"
)
RUNTIME_PACKAGE_DISTRIBUTIONS = (
    "datasets",
    "fsspec",
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_REPOSITORY_BINDING_SCHEMA = "recurquant.git-repository-binding.v1"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts" / f"{EXPERIMENT_ID}_stage0_production.pt"
LINEAR_LAYER_INDICES = tuple(EXPERIMENT010_STATELEASE_LAYER_QUOTAS)
FROZEN_STATELEASE_RESIDENT_BYTES = 3_454_664
RAW_STATE_ELEMENTS_PER_LAYER = 16 * 128 * 128
RAW_STATE_ELEMENTS_ALL_LAYERS = len(LINEAR_LAYER_INDICES) * RAW_STATE_ELEMENTS_PER_LAYER
EXPERIMENT012_SOURCE_PROVENANCE_PATHS = (
    "research/EXPERIMENT_012_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_012_STAGE_A_IDENTITY.md",
    "research/EXPERIMENT_011_STAGE_A_ADMINISTRATIVE_NULL.md",
    "evidence/experiment011-statelease-stage-a-administrative-null.json",
    "artifacts/experiment011-statelease-stage-a-666.attempt.json",
    "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_011_STAGE_A_IDENTITY.md",
    "research/EXPERIMENT_010_STAGE_A_ADMINISTRATIVE_NULL.md",
    "evidence/experiment010-statelease-stage-a-administrative-null.json",
    "artifacts/experiment010-statelease-stage-a-666.attempt.json",
    "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
    "research/EXPERIMENT_010_STAGE_A_IDENTITY.md",
)
SOURCE_IDENTITY_PATHS = (
    "pyproject.toml",
    "scripts/capture_statelease_stage0.py",
    "scripts/screen_statelease_stage_a.py",
    "scripts/verify_statelease_stage0.py",
    *EXPERIMENT012_SOURCE_PROVENANCE_PATHS,
    "src/recurquant/__init__.py",
    "src/recurquant/cache.py",
    "src/recurquant/cli.py",
    "src/recurquant/confirmation.py",
    "src/recurquant/evaluation.py",
    "src/recurquant/evidence.py",
    "src/recurquant/experiment013_calibration_api.py",
    "src/recurquant/experiment013_parquet.py",
    "src/recurquant/experiment013_qwen35_adapter.py",
    "src/recurquant/experiment013_source.py",
    "src/recurquant/experiment013_stage_a.py",
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
    "src/recurquant/static_q468.py",
    "src/recurquant/static_q468_cache.py",
    "src/recurquant/static_q468_calibration.py",
    "src/recurquant/statelease.py",
    "src/recurquant/statelease_artifact.py",
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
    "tests/test_statelease_artifact.py",
    "tests/test_statelease_baselines.py",
    "tests/test_statelease_cache.py",
    "tests/test_statelease_equal_byte_baselines.py",
    "tests/test_statelease_equal_byte_cache.py",
    "tests/test_statelease_evaluation.py",
    "tests/test_statelease_observer.py",
    "tests/test_screen_statelease_stage_a.py",
    "tests/test_verify_statelease_stage0.py",
)
REQUIRED_LOADED_RECURQUANT_MODULE_PATHS = (
    ("recurquant", "src/recurquant/__init__.py"),
    ("recurquant.evidence", "src/recurquant/evidence.py"),
    ("recurquant.finite_difference", "src/recurquant/finite_difference.py"),
    ("recurquant.fisher_sensitivity", "src/recurquant/fisher_sensitivity.py"),
    ("recurquant.horizon", "src/recurquant/horizon.py"),
    ("recurquant.horizon_calibration", "src/recurquant/horizon_calibration.py"),
    ("recurquant.intervention", "src/recurquant/intervention.py"),
    ("recurquant.mixed_quantization", "src/recurquant/mixed_quantization.py"),
    ("recurquant.model_fisher", "src/recurquant/model_fisher.py"),
    ("recurquant.multibit_policy", "src/recurquant/multibit_policy.py"),
    ("recurquant.multibit_quantization", "src/recurquant/multibit_quantization.py"),
    ("recurquant.packed_cache", "src/recurquant/packed_cache.py"),
    ("recurquant.quantization", "src/recurquant/quantization.py"),
    ("recurquant.query_energy", "src/recurquant/query_energy.py"),
    ("recurquant.qwen35", "src/recurquant/qwen35.py"),
    ("recurquant.rht", "src/recurquant/rht.py"),
    ("recurquant.row_policy", "src/recurquant/row_policy.py"),
    ("recurquant.statelease", "src/recurquant/statelease.py"),
    ("recurquant.statelease_baselines", "src/recurquant/statelease_baselines.py"),
    ("recurquant.statelease_cache", "src/recurquant/statelease_cache.py"),
    (
        "recurquant.statelease_equal_byte_baselines",
        "src/recurquant/statelease_equal_byte_baselines.py",
    ),
    (
        "recurquant.statelease_equal_byte_cache",
        "src/recurquant/statelease_equal_byte_cache.py",
    ),
    ("recurquant.statelease_evaluation", "src/recurquant/statelease_evaluation.py"),
    ("recurquant.statelease_observer", "src/recurquant/statelease_observer.py"),
    ("recurquant.transition_observer", "src/recurquant/transition_observer.py"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_payload_sha256(value: object) -> str:
    """Hash the closed schema independently of torch serialization details."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        item_type = type(item)
        if item is None:
            digest.update(b"n")
        elif item_type is bool:
            digest.update(b"b1" if item else b"b0")
        elif item_type is int:
            digest.update(b"i")
            digest.update(str(item).encode("ascii"))
            digest.update(b"\0")
        elif item_type is float:
            if not math.isfinite(item):
                raise ValueError("canonical payload cannot contain non-finite scalars")
            digest.update(b"f")
            digest.update(item.hex().encode("ascii"))
            digest.update(b"\0")
        elif item_type is str:
            encoded = item.encode("utf-8")
            digest.update(b"s")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        elif item_type is torch.Tensor:
            digest.update(b"t")
            digest.update(_tensor_sha256(item).encode("ascii"))
        elif item_type is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("closed-schema mapping keys must be plain strings")
            digest.update(b"d")
            digest.update(len(item).to_bytes(8, "little"))
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif item_type is list or item_type is tuple:
            digest.update(b"l" if item_type is list else b"t")
            digest.update(len(item).to_bytes(8, "little"))
            for child in item:
                visit(child)
        else:
            raise TypeError(f"closed schema cannot hash {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone(memory_format=torch.contiguous_format)


def _frozen_config() -> Qwen3_5TextConfig:
    layer_types = [
        "linear_attention" if index in EXPERIMENT010_STATELEASE_LAYER_QUOTAS else "full_attention"
        for index in range(24)
    ]
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        layer_types=layer_types,
    )


def _frozen_plan() -> ExactBudgetRowPlan:
    rows = tuple(
        RowLocation(
            layer_index=layer_index,
            head_index=flat_index // 128,
            row_index=flat_index % 128,
        )
        for layer_index, quota in EXPERIMENT010_STATELEASE_LAYER_QUOTAS.items()
        for flat_index in range(quota)
    )
    plan = ExactBudgetRowPlan(
        low_bits=4,
        high_bits=8,
        group_size=128,
        scale_bits=16,
        total_groups=36_864,
        mask_bytes=4_608,
        promotion_increment_bytes=64,
        target_resident_bytes=2_564_096,
        resident_bytes=2_564_096,
        high_precision_rows=rows,
        score_shapes=tuple(
            (layer_index, 16, 128) for layer_index in EXPERIMENT010_STATELEASE_LAYER_QUOTAS
        ),
    )
    if (
        experiment010_statelease_effective_plan_sha256(plan)
        != EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256
    ):
        raise RuntimeError(
            "constructed Stage-0 plan does not match the frozen storage-and-quota summary"
        )
    return plan


def _transition_signals(
    initial_state: torch.Tensor | None,
    *,
    seed: int,
    key_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn((1, 1, 16, 128), generator=generator)
    key = torch.randn((1, 1, 16, 128), generator=generator).to(key_dtype)
    value = torch.randn((1, 1, 16, 128), generator=generator)
    log_decay = -0.15 * torch.rand((1, 1, 16), generator=generator)
    beta = torch.rand((1, 1, 16), generator=generator)
    source = (
        torch.zeros((1, 16, 128, 128), dtype=torch.float32)
        if initial_state is None
        else initial_state
    )
    final_state = replay_gated_delta_state(
        source,
        key,
        value,
        log_decay,
        beta,
    )
    return {
        "query": query,
        "consumed_key": key,
        "value": value,
        "log_decay": log_decay,
        "beta": beta,
        "initial_state": source,
        "successful_final_state": final_state,
    }


def _empty_record() -> dict[str, object]:
    return {
        "present": False,
        "normalized_key": torch.empty(0, dtype=torch.float32),
        "update": torch.empty(0, dtype=torch.float32),
        "log_decay": torch.empty(0, dtype=torch.float32),
    }


def _pending_record(layer: StateLeaseLinearAttentionLayer) -> dict[str, object]:
    pending = layer._pending_statelease_observation
    if pending is None:
        raise RuntimeError("production layer did not retain its staged observation")
    if pending.normalized_key is None:
        return _empty_record()
    assert pending.update is not None
    assert pending.log_decay is not None
    return {
        "present": True,
        "normalized_key": _clone_cpu(pending.normalized_key),
        "update": _clone_cpu(pending.update),
        "log_decay": _clone_cpu(pending.log_decay),
    }


def _layer_buffers(layer: StateLeaseLinearAttentionLayer) -> dict[str, torch.Tensor]:
    if (
        layer.normalized_key_buffer is None
        or layer.update_buffer is None
        or layer.log_decay_buffer is None
        or layer.valid_count is None
    ):
        raise RuntimeError("production layer has no allocated replay storage")
    return {
        "normalized_keys": _clone_cpu(layer.normalized_key_buffer),
        "updates": _clone_cpu(layer.update_buffer),
        "log_decays": _clone_cpu(layer.log_decay_buffer),
        "valid_count": _clone_cpu(layer.valid_count),
    }


def _capture_statelease_trace(
    cache: StateLeaseRecurrentStateCache,
) -> list[dict[str, object]]:
    layer = cache.layers[0]
    if not isinstance(layer, StateLeaseLinearAttentionLayer):
        raise RuntimeError("frozen layer zero is not a StateLease layer")
    trace: list[dict[str, object]] = []
    current: torch.Tensor | None = None
    for step_index in range(6):
        signals = _transition_signals(
            current,
            seed=FROZEN_SEED + 100 + step_index,
            key_dtype=torch.bfloat16,
        )
        initial_for_production = None if step_index == 0 else current
        final_state = signals["successful_final_state"]
        cache.stage_statelease_observation(
            0,
            signals["query"],
            signals["consumed_key"],
            signals["value"],
            signals["log_decay"],
            signals["beta"],
            initial_for_production,
            final_state,
        )
        record = _pending_record(layer)
        materialized = cache.update_recurrent_state(final_state, layer_idx=0)
        evidence = layer.last_update_evidence
        if evidence is None or layer.query_energy_ema is None:
            raise RuntimeError("production StateLease update has no evidence or query EMA")
        trace.append(
            {
                "identity": f"statelease_trace_step_{step_index}",
                "step_index": step_index,
                "signals": {name: _clone_cpu(value) for name, value in signals.items()},
                "production_record": record,
                "production_query_ema": _clone_cpu(layer.query_energy_ema),
                "production_materialized_state": _clone_cpu(materialized),
                "production_buffers": _layer_buffers(layer),
                "production_evidence": evidence.evidence_dict(),
            }
        )
        current = materialized
    if trace[-1]["production_evidence"]["action"] not in {"boundary_4", "boundary_5"}:
        raise RuntimeError("synthetic trace did not exercise the full H5 boundary")
    return trace


def _observer_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [3, 3, 2],
        },
    )


def _observer_plan() -> ExactBudgetRowPlan:
    total_groups = 16
    low_group_bytes = 6
    target = total_groups * low_group_bytes + 2 + 5 * 4
    return select_rows_exact_budget(
        {0: torch.arange(total_groups, dtype=torch.float32).reshape(2, 8)},
        target_resident_bytes=target,
        group_size=8,
    )


def _capture_successful_kernel_receipt() -> dict[str, object]:
    """Run the real Qwen observer around successful chunk/recurrent kernels."""

    config = _observer_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        model = Qwen3_5ForCausalLM._from_config(
            config,
            attn_implementation="eager",
        ).eval()
    cache = create_qwen35_statelease_cache(
        config,
        plan=_observer_plan(),
        record_evidence=True,
    )
    staged: list[dict[str, object]] = []
    original_stage = cache.stage_statelease_observation

    def capture_stage(
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        final_state: torch.Tensor,
    ) -> None:
        staged.append(
            {
                "layer_index": layer_index,
                "query": _clone_cpu(query),
                "consumed_key": _clone_cpu(key),
                "value": _clone_cpu(value),
                "log_decay": _clone_cpu(log_decay),
                "beta": _clone_cpu(beta),
                "initial_state_present": initial_state is not None,
                "initial_state": (
                    torch.empty(0, dtype=torch.float32)
                    if initial_state is None
                    else _clone_cpu(initial_state)
                ),
                "successful_final_state": _clone_cpu(final_state),
            }
        )
        original_stage(
            layer_index,
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            final_state,
        )

    cache.stage_statelease_observation = capture_stage  # type: ignore[method-assign]
    input_ids = (
        torch.tensor([[7, 11, 13]], dtype=torch.long),
        torch.tensor([[17]], dtype=torch.long),
    )
    receipts: list[dict[str, object]] = []
    with torch.inference_mode(), Qwen35StateLeaseObserver(model, caches=[cache]):
        for call_index, tokens in enumerate(input_ids):
            before = len(staged)
            output = model(
                tokens,
                past_key_values=cache,
                use_cache=True,
            )
            if len(staged) != before + 1:
                raise RuntimeError("observer did not capture exactly one successful kernel")
            layer = cache.layers[0]
            if not isinstance(layer, StateLeaseLinearAttentionLayer):
                raise RuntimeError("observer receipt cache has incompatible layer zero")
            materialized = layer.materialize_recurrent_state()
            evidence = layer.last_update_evidence
            if materialized is None or evidence is None or layer.query_energy_ema is None:
                raise RuntimeError("successful kernel receipt was not committed")
            receipt = dict(staged[-1])
            receipt.update(
                {
                    "call_index": call_index,
                    "input_ids": _clone_cpu(tokens),
                    "logits_shape": list(output.logits.shape),
                    "logits_finite": bool(torch.isfinite(output.logits).all().item()),
                    "production_query_ema": _clone_cpu(layer.query_energy_ema),
                    "production_materialized_state": _clone_cpu(materialized),
                    "production_buffers": _layer_buffers(layer),
                    "production_evidence": evidence.evidence_dict(),
                }
            )
            receipts.append(receipt)
    return {
        "observer": "Qwen35StateLeaseObserver",
        "model": "randomly_initialized_tiny_Qwen3_5ForCausalLM",
        "pretrained_checkpoint_loaded": False,
        "synthetic_token_ids_only": True,
        "successful_model_forwards": 2,
        "successful_kernel_calls": 2,
        "receipts": receipts,
    }


def _layer_persistent_tensors(
    layer_index: int,
    layer: StateLeaseLinearAttentionLayer,
) -> tuple[tuple[str, torch.Tensor], ...]:
    packed = layer.packed_checkpoint
    if (
        packed is None
        or layer.query_energy_ema is None
        or layer.normalized_key_buffer is None
        or layer.update_buffer is None
        or layer.log_decay_buffer is None
        or layer.valid_count is None
    ):
        raise RuntimeError(f"StateLease layer {layer_index} is not fully initialized")
    return (
        (f"layer_{layer_index}.checkpoint.low_payload", packed.low_payload),
        (f"layer_{layer_index}.checkpoint.high_payload", packed.high_payload),
        (f"layer_{layer_index}.checkpoint.scales", packed.scales),
        (f"layer_{layer_index}.checkpoint.precision_mask", packed.precision_mask),
        (f"layer_{layer_index}.query_energy_ema", layer.query_energy_ema),
        (f"layer_{layer_index}.normalized_key_buffer", layer.normalized_key_buffer),
        (f"layer_{layer_index}.update_buffer", layer.update_buffer),
        (f"layer_{layer_index}.log_decay_buffer", layer.log_decay_buffer),
        (f"layer_{layer_index}.valid_count", layer.valid_count),
    )


def _stable_graph_item_key(value: object) -> tuple[str, str, str]:
    value_type = type(value)
    try:
        rendered = repr(value)
    except Exception:
        rendered = "<unrepresentable>"
    return value_type.__module__, value_type.__qualname__, rendered


def _graph_key_segment(value: object, index: int) -> str:
    if isinstance(value, (str, int)):
        return repr(value)
    return f"<{type(value).__module__}.{type(value).__qualname__}:{index}>"


def _object_slots(value: object) -> tuple[str, ...]:
    names: set[str] = set()
    for value_type in type(value).__mro__:
        declared = value_type.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if isinstance(name, str) and name not in {"__dict__", "__weakref__"}:
                names.add(name)
    return tuple(sorted(names))


def _discover_object_graph_tensor_views(
    root: object,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Walk stored object/container fields without invoking semantic cache views."""

    seen_objects: set[int] = set()
    discovered: list[tuple[str, torch.Tensor]] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            discovered.append((path, value))
            return
        if value is None or isinstance(
            value,
            (str, bytes, int, float, bool, complex, torch.dtype, torch.device),
        ):
            return
        if isinstance(value, type) or inspect.ismodule(value) or inspect.isroutine(value):
            return
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)

        if isinstance(value, dict):
            ordered_keys = sorted(value, key=_stable_graph_item_key)
            for index, key in enumerate(ordered_keys):
                segment = _graph_key_segment(key, index)
                if not isinstance(key, (str, bytes, int, float, bool)):
                    visit(key, f"{path}.<key:{segment}>")
                visit(value[key], f"{path}[{segment}]")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            for index, child in enumerate(sorted(value, key=_stable_graph_item_key)):
                visit(child, f"{path}.<set:{index}>")
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return

        traversed_names: set[str] = set()
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, child in sorted(attributes.items()):
                traversed_names.add(name)
                visit(child, f"{path}.{name}")
        for name in _object_slots(value):
            if name in traversed_names or not hasattr(value, name):
                continue
            visit(getattr(value, name), f"{path}.{name}")
        if not traversed_names and not _object_slots(value) and isinstance(value, Mapping):
            ordered_keys = sorted(value, key=_stable_graph_item_key)
            for index, key in enumerate(ordered_keys):
                segment = _graph_key_segment(key, index)
                visit(value[key], f"{path}[{segment}]")

    visit(root, "cache")
    return tuple(sorted(discovered, key=lambda item: item[0]))


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    storage = tensor.untyped_storage()
    storage_handle = int(getattr(storage, "_cdata", storage.data_ptr()))
    return tensor.device.type, tensor.device.index, storage_handle


def _candidate_graph_path(name: str) -> str:
    checkpoint = re.fullmatch(
        r"layer_(\d+)\.checkpoint\.(low_payload|high_payload|scales|precision_mask)",
        name,
    )
    if checkpoint is not None:
        return f"cache.layers[{int(checkpoint.group(1))}].packed_states[0].{checkpoint.group(2)}"
    direct = re.fullmatch(
        r"layer_(\d+)\.(query_energy_ema|normalized_key_buffer|update_buffer|"
        r"log_decay_buffer|valid_count)",
        name,
    )
    if direct is None:
        raise RuntimeError(f"unknown StateLease persistent tensor name: {name}")
    return f"cache.layers[{int(direct.group(1))}].{direct.group(2)}"


def _tensor_view_metadata(path: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "path": path,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
        "storage_offset": tensor.storage_offset(),
        "sha256": _tensor_sha256(tensor),
    }


def _is_shared_kv_path(path: str) -> bool:
    match = re.match(r"cache\.layers\[(\d+)\]\.(keys|values)(?:$|[\[.])", path)
    return match is not None and int(match.group(1)) not in LINEAR_LAYER_INDICES


def _is_shared_conv_path(path: str) -> bool:
    return re.match(r"cache\.layers\[(\d+)\]\.conv_states(?:$|[\[.])", path) is not None


def _build_whole_cache_storage_inventory(
    cache: object,
    captured: Sequence[tuple[str, torch.Tensor]],
) -> list[dict[str, object]]:
    allowed_by_storage: dict[tuple[str, int | None, int], str] = {}
    for name, tensor in captured:
        identity = _storage_identity(tensor)
        if identity in allowed_by_storage:
            raise RuntimeError(
                f"allowed StateLease storage aliases {name} and {allowed_by_storage[identity]}"
            )
        allowed_by_storage[identity] = name

    grouped: dict[
        tuple[str, int | None, int],
        list[tuple[str, torch.Tensor]],
    ] = {}
    for path, tensor in _discover_object_graph_tensor_views(cache):
        grouped.setdefault(_storage_identity(tensor), []).append((path, tensor))

    pending: list[dict[str, object]] = []
    for identity, views in grouped.items():
        ordered_views = sorted(views, key=lambda item: item[0])
        allowed_name = allowed_by_storage.get(identity)
        paths = [path for path, _ in ordered_views]
        if allowed_name is not None:
            classification = "statelease_candidate"
        elif all(_is_shared_kv_path(path) for path in paths):
            classification = "shared_kv"
        elif all(_is_shared_conv_path(path) for path in paths):
            classification = "shared_conv"
        else:
            classification = "unexplained"
        storage_nbytes = ordered_views[0][1].untyped_storage().nbytes()
        if any(tensor.untyped_storage().nbytes() != storage_nbytes for _, tensor in ordered_views):
            raise RuntimeError("storage aliases disagree on physical byte size")
        pending.append(
            {
                "classification": classification,
                "allowed_tensor_name": allowed_name,
                "storage_nbytes": storage_nbytes,
                "views": [_tensor_view_metadata(path, tensor) for path, tensor in ordered_views],
            }
        )

    pending.sort(key=lambda entry: entry["views"][0]["path"])
    return [{"storage_index": index, **entry} for index, entry in enumerate(pending)]


def _expected_whole_cache_storage_inventory(
    captured: Sequence[tuple[str, torch.Tensor]],
) -> list[dict[str, object]]:
    pending = []
    for name, tensor in captured:
        pending.append(
            {
                "classification": "statelease_candidate",
                "allowed_tensor_name": name,
                "storage_nbytes": tensor.untyped_storage().nbytes(),
                "views": [_tensor_view_metadata(_candidate_graph_path(name), tensor)],
            }
        )
    pending.sort(key=lambda entry: entry["views"][0]["path"])
    return [{"storage_index": index, **entry} for index, entry in enumerate(pending)]


def _summarize_whole_cache_inventory(
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    bytes_by_class = {
        "statelease_candidate": 0,
        "shared_kv": 0,
        "shared_conv": 0,
        "unexplained": 0,
    }
    unexplained_elements = 0
    for entry in inventory:
        classification = str(entry["classification"])
        if classification not in bytes_by_class:
            raise RuntimeError(f"unknown whole-cache storage class: {classification}")
        bytes_by_class[classification] += int(entry["storage_nbytes"])
        if classification == "unexplained":
            views = entry["views"]
            if not isinstance(views, list):
                raise RuntimeError("whole-cache inventory views must be a list")
            unexplained_elements += max(int(view["numel"]) for view in views)
    return {
        "inventory": list(inventory),
        "candidate_storage_count": sum(
            entry["classification"] == "statelease_candidate" for entry in inventory
        ),
        "candidate_unique_storage_bytes": bytes_by_class["statelease_candidate"],
        "shared_kv_unique_storage_bytes": bytes_by_class["shared_kv"],
        "shared_conv_unique_storage_bytes": bytes_by_class["shared_conv"],
        "unexplained_unique_storage_bytes": bytes_by_class["unexplained"],
        "total_unique_storage_bytes": sum(bytes_by_class.values()),
        "unexplained_tensor_elements": unexplained_elements,
        "raw_state_elements_per_layer": RAW_STATE_ELEMENTS_PER_LAYER,
        "raw_state_elements_all_layers": RAW_STATE_ELEMENTS_ALL_LAYERS,
        "unexplained_raw_state_equivalent_layer_floor": (
            unexplained_elements // RAW_STATE_ELEMENTS_PER_LAYER
        ),
        "all_persistent_storage_classified": bytes_by_class["unexplained"] == 0,
        "no_unexplained_persistent_storage": bytes_by_class["unexplained"] == 0,
        "storage_deduplicated": True,
    }


def _audit_whole_cache_persistent_tensors(
    cache: object,
    captured: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, object]:
    actual_inventory = _build_whole_cache_storage_inventory(cache, captured)
    actual = _summarize_whole_cache_inventory(actual_inventory)
    if actual["unexplained_unique_storage_bytes"] != 0:
        unexplained_paths = [
            view["path"]
            for entry in actual_inventory
            if entry["classification"] == "unexplained"
            for view in entry["views"]
        ]
        raise RuntimeError(
            "whole-cache all-dtype audit found unexplained persistent tensor storage: "
            f"{unexplained_paths}"
        )
    if (
        actual["shared_kv_unique_storage_bytes"] != 0
        or actual["shared_conv_unique_storage_bytes"] != 0
    ):
        raise RuntimeError(
            "synthetic Stage-0 cache unexpectedly owns shared KV or convolution storage"
        )
    if actual["candidate_unique_storage_bytes"] != FROZEN_STATELEASE_RESIDENT_BYTES:
        raise RuntimeError(
            "whole-cache candidate storage owns "
            f"{actual['candidate_unique_storage_bytes']} bytes; "
            f"expected {FROZEN_STATELEASE_RESIDENT_BYTES}"
        )
    expected_inventory = _expected_whole_cache_storage_inventory(captured)
    if actual_inventory != expected_inventory:
        raise RuntimeError(
            "whole-cache persistent tensor inventory differs from the exact "
            "StateLease storage schema"
        )
    return actual


def _tensor_enumeration(
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "nbytes": tensor.numel() * tensor.element_size(),
            "sha256": _tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors, key=lambda item: item[0])
    ]


def _initialize_remaining_layers(cache: StateLeaseRecurrentStateCache) -> None:
    for layer_index, layer in cache.statelease_layers():
        if layer.packed_checkpoint is not None:
            continue
        signals = _transition_signals(
            None,
            seed=FROZEN_SEED + 1000 + layer_index,
            key_dtype=torch.float32,
        )
        cache.stage_statelease_observation(
            layer_index,
            signals["query"],
            signals["consumed_key"],
            signals["value"],
            signals["log_decay"],
            signals["beta"],
            None,
            signals["successful_final_state"],
        )
        cache.update_recurrent_state(
            signals["successful_final_state"],
            layer_idx=layer_index,
        )


def _capture_resident_snapshot(
    cache: StateLeaseRecurrentStateCache,
) -> tuple[dict[str, object], dict[int, torch.Tensor], torch.Tensor]:
    _initialize_remaining_layers(cache)
    fields: dict[str, list[torch.Tensor]] = {
        "checkpoint_low_payloads": [],
        "checkpoint_high_payloads": [],
        "checkpoint_scales": [],
        "checkpoint_masks": [],
        "query_emas": [],
        "normalized_key_buffers": [],
        "update_buffers": [],
        "log_decay_buffers": [],
        "valid_counts": [],
    }
    captured: list[tuple[str, torch.Tensor]] = []
    source_states: dict[int, torch.Tensor] = {}
    query_emas: list[torch.Tensor] = []
    for layer_index, layer in cache.statelease_layers():
        packed = layer.packed_checkpoint
        materialized = layer.materialize_recurrent_state()
        if packed is None or materialized is None or layer.query_energy_ema is None:
            raise RuntimeError(f"StateLease layer {layer_index} is incomplete")
        fields["checkpoint_low_payloads"].append(_clone_cpu(packed.low_payload))
        fields["checkpoint_high_payloads"].append(_clone_cpu(packed.high_payload))
        fields["checkpoint_scales"].append(_clone_cpu(packed.scales))
        fields["checkpoint_masks"].append(_clone_cpu(packed.precision_mask))
        fields["query_emas"].append(_clone_cpu(layer.query_energy_ema))
        fields["normalized_key_buffers"].append(_clone_cpu(layer.normalized_key_buffer))
        fields["update_buffers"].append(_clone_cpu(layer.update_buffer))
        fields["log_decay_buffers"].append(_clone_cpu(layer.log_decay_buffer))
        fields["valid_counts"].append(_clone_cpu(layer.valid_count))
        captured.extend(_layer_persistent_tensors(layer_index, layer))
        source_states[layer_index] = _clone_cpu(materialized)
        query_emas.append(_clone_cpu(layer.query_energy_ema))
    captured_enumeration = _tensor_enumeration(captured)
    whole_cache_storage_audit = _audit_whole_cache_persistent_tensors(cache, captured)
    snapshot: dict[str, object] = {
        **fields,
        "persistent_enumeration": captured_enumeration,
        "whole_cache_storage_audit": whole_cache_storage_audit,
        "no_hidden_persistent_state_mirror": True,
        "storage_summary": cache.storage_summary(),
    }
    return snapshot, source_states, torch.stack(query_emas).contiguous()


def _layer_state_digest(layer_index: int, layer: StateLeaseLinearAttentionLayer) -> str:
    payload = {
        name: _clone_cpu(tensor) for name, tensor in _layer_persistent_tensors(layer_index, layer)
    }
    payload["diagnostics"] = layer.statelease_diagnostics()
    return canonical_payload_sha256(payload)


def _capture_rollback_receipt(
    cache: StateLeaseRecurrentStateCache,
) -> dict[str, object]:
    layer = cache.layers[0]
    if not isinstance(layer, StateLeaseLinearAttentionLayer):
        raise RuntimeError("layer zero is not StateLease")
    current = layer.materialize_recurrent_state()
    if current is None:
        raise RuntimeError("rollback receipt needs an initialized state")
    before_digest = _layer_state_digest(0, layer)
    before_update_index = cache._update_index
    before_evidence_count = len(cache.update_evidence)
    signals = _transition_signals(current, seed=FROZEN_SEED + 5000)
    transaction = cache.begin_statelease_forward_transaction()
    cache.stage_statelease_observation(
        0,
        signals["query"],
        signals["consumed_key"],
        signals["value"],
        signals["log_decay"],
        signals["beta"],
        current,
        signals["successful_final_state"],
    )
    cache.update_recurrent_state(signals["successful_final_state"], layer_idx=0)
    mutated_digest = _layer_state_digest(0, layer)
    cache.rollback_statelease_forward_transaction(transaction)
    restored_layer = cache.layers[0]
    assert isinstance(restored_layer, StateLeaseLinearAttentionLayer)
    after_digest = _layer_state_digest(0, restored_layer)
    if before_digest == mutated_digest or before_digest != after_digest:
        raise RuntimeError("production rollback receipt failed")
    return {
        "before_sha256": before_digest,
        "mutated_sha256": mutated_digest,
        "after_sha256": after_digest,
        "before_update_index": before_update_index,
        "after_update_index": cache._update_index,
        "before_evidence_count": before_evidence_count,
        "after_evidence_count": len(cache.update_evidence),
        "transaction_active_after_rollback": (cache._active_statelease_transaction is not None),
    }


def _capture_cc1_compatibility(
    config: Qwen3_5TextConfig,
    plan: ExactBudgetRowPlan,
) -> dict[str, object]:
    fixed = create_qwen35_experiment010_fixed_replay_cache(
        config,
        plan=plan,
        mode="fixed_cc1",
        record_evidence=True,
    )
    anchor = create_qwen35_right_rht_query_ema_exact_budget_cache(
        config,
        plan=plan,
        record_evidence=True,
    )
    fixed_layer = fixed.layers[0]
    anchor_layer = anchor.layers[0]
    if not isinstance(fixed_layer, StateLeaseLinearAttentionLayer):
        raise RuntimeError("fixed CC1 layer zero is incompatible")
    fixed_trajectory: list[torch.Tensor] = []
    anchor_trajectory: list[torch.Tensor] = []
    fixed_masks: list[torch.Tensor] = []
    anchor_masks: list[torch.Tensor] = []
    transitions: list[dict[str, torch.Tensor]] = []
    current: torch.Tensor | None = None
    for step_index in range(4):
        signals = _transition_signals(
            current,
            seed=FROZEN_SEED + 6000 + step_index,
            key_dtype=torch.float32,
        )
        initial = None if step_index == 0 else current
        final_state = signals["successful_final_state"]
        fixed.stage_statelease_observation(
            0,
            signals["query"],
            signals["consumed_key"],
            signals["value"],
            signals["log_decay"],
            signals["beta"],
            initial,
            final_state,
        )
        anchor.stage_query_observation(0, signals["query"])
        fixed_state = fixed.update_recurrent_state(final_state, layer_idx=0)
        anchor_state = anchor.update_recurrent_state(final_state, layer_idx=0)
        fixed_packed = fixed_layer.packed_checkpoint
        anchor_packed = anchor_layer.packed_states[0]
        if fixed_packed is None or anchor_packed is None:
            raise RuntimeError("CC1 compatibility path did not create packed checkpoints")
        fixed_trajectory.append(_clone_cpu(fixed_state))
        anchor_trajectory.append(_clone_cpu(anchor_state))
        fixed_masks.append(_clone_cpu(fixed_packed.precision_mask))
        anchor_masks.append(_clone_cpu(anchor_packed.precision_mask))
        transitions.append({name: _clone_cpu(value) for name, value in signals.items()})
        if not torch.equal(fixed_state, anchor_state):
            maximum = float((fixed_state - anchor_state).abs().max().item())
            raise RuntimeError(f"fixed CC1 differs from RHT-CQER; max_abs={maximum:.9g}")
        current = fixed_state
    fixed_tensor = torch.stack(fixed_trajectory)
    anchor_tensor = torch.stack(anchor_trajectory)
    reference = torch.stack([transition["successful_final_state"] for transition in transitions])

    def metrics(candidate: torch.Tensor) -> dict[str, float]:
        error = candidate.to(torch.float64) - reference.to(torch.float64)
        denominator = torch.linalg.vector_norm(reference.to(torch.float64)).clamp_min(1e-12)
        return {
            "synthetic_state_mse": float(error.square().mean().item()),
            "synthetic_state_relative_l2": float(
                (torch.linalg.vector_norm(error) / denominator).item()
            ),
        }

    row_plan = {
        "effective_plan_sha256": EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256,
        "layer_0_quota": EXPERIMENT010_STATELEASE_LAYER_QUOTAS[0],
    }
    return {
        "transitions": transitions,
        "fixed_cc1_trajectory": fixed_tensor,
        "rht_cqer_trajectory": anchor_tensor,
        "fixed_cc1_masks": fixed_masks,
        "rht_cqer_masks": anchor_masks,
        "fixed_cc1_metrics": metrics(fixed_tensor),
        "rht_cqer_metrics": metrics(anchor_tensor),
        "row_plan": row_plan,
        "hashes": {
            "fixed_trajectory": _tensor_sha256(fixed_tensor),
            "rht_cqer_trajectory": _tensor_sha256(anchor_tensor),
            "fixed_masks": canonical_payload_sha256(fixed_masks),
            "rht_cqer_masks": canonical_payload_sha256(anchor_masks),
            "row_plan": canonical_payload_sha256(row_plan),
        },
    }


def _capture_equal_byte_comparators(
    source_states: Mapping[int, torch.Tensor],
    query_ema: torch.Tensor,
) -> dict[str, object]:
    packers = (
        (EXPANDED_RHT_Q4_Q8, pack_expanded_rht_q4_q8),
        (RHT_Q4_Q6_Q8, pack_rht_q4_q6_q8),
        (RHT_RESIDUAL_Q4, pack_rht_residual_q4),
    )
    snapshots: dict[str, object] = {}
    for codec, packer in packers:
        checkpoint = packer(
            source_states,
            query_ema,
            layout=FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
        )
        tensors = tuple(
            (name, _clone_cpu(tensor)) for name, tensor in checkpoint.persistent_tensors()
        )
        snapshots[codec] = {
            "tensors": {name: tensor for name, tensor in tensors},
            "persistent_enumeration": _tensor_enumeration(tensors),
            "evidence": checkpoint.evidence.evidence_dict(),
            "no_replay": True,
            "no_hidden_persistent_state_mirror": True,
        }
    return {
        "source_states": {
            str(layer_index): _clone_cpu(source_states[layer_index])
            for layer_index in LINEAR_LAYER_INDICES
        },
        "query_energy_ema": _clone_cpu(query_ema),
        "snapshots": snapshots,
    }


def _sanitized_capture_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _capture_git(
    *arguments: str,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.useReplaceRefs=false",
            "-c",
            f"core.attributesFile={os.devnull}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *arguments,
        ],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
        input=input_text,
        env=_sanitized_capture_git_environment(),
    )


def _resolved_git_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _private_path_sha256(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve())).replace("\\", "/")
    return _sha256_bytes(normalized.encode("utf-8"))


def _capture_local_config_sha256() -> str:
    process = _capture_git("config", "--local", "--no-includes", "--null", "--list")
    entries: list[dict[str, str]] = []
    for raw in process.stdout.split("\0"):
        if not raw:
            continue
        key, separator, value = raw.partition("\n")
        if not separator or not key:
            raise RuntimeError("local Git config contains a malformed entry")
        normalized_key = key.lower()
        entries.append({"key": normalized_key, "value": value})
        forbidden = (
            normalized_key.startswith(("include.", "includeif.", "filter."))
            or normalized_key
            in {
                "core.alternaterefscommand",
                "core.alternaterefsprefixes",
                "core.attributesfile",
                "core.fsmonitor",
                "core.hookspath",
                "core.sparsecheckout",
                "core.sparsecheckoutcone",
                "core.untrackedcache",
                "core.worktree",
                "extensions.partialclone",
                "extensions.worktreeconfig",
                "index.sparse",
            }
            or (
                normalized_key.startswith("remote.")
                and normalized_key.endswith((".promisor", ".partialclonefilter"))
            )
        )
        if forbidden:
            raise RuntimeError(f"unsafe local Git config key is present: {normalized_key}")
        if normalized_key == "core.usereplacerefs" and value.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            raise RuntimeError("local Git config enables replacement objects")
    values_by_key: dict[str, list[str]] = {}
    for entry in entries:
        values_by_key.setdefault(entry["key"], []).append(entry["value"])
    if values_by_key.get("core.repositoryformatversion") != ["0"]:
        raise RuntimeError("Git repository format version is not exactly zero")
    if values_by_key.get("core.bare") != ["false"]:
        raise RuntimeError("Git repository must explicitly be non-bare")
    return canonical_payload_sha256(entries)


def _assert_capture_index_has_no_hidden_flags() -> None:
    """Reject index flags that can hide tracked worktree changes from status."""

    try:
        process = _capture_git("ls-files", "--cached", "-v", "-z")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot authenticate Git index visibility flags") from error
    records = [record for record in process.stdout.split("\0") if record]
    malformed = [record for record in records if len(record) < 3 or record[1] != " "]
    if malformed:
        raise RuntimeError("Git index visibility output is malformed")
    unsafe_tags = sorted({record[0] for record in records if record[0] != "H"})
    if unsafe_tags:
        raise RuntimeError(
            f"Git index contains hidden or non-canonical tracked entries (tags={unsafe_tags})"
        )


def _capture_repository_binding() -> dict[str, object]:
    try:
        top_level = _resolved_git_path(_capture_git("rev-parse", "--show-toplevel").stdout.strip())
        git_dir = _resolved_git_path(_capture_git("rev-parse", "--absolute-git-dir").stdout.strip())
        common_dir = _resolved_git_path(
            _capture_git("rev-parse", "--git-common-dir").stdout.strip()
        )
        index_path = _resolved_git_path(
            _capture_git("rev-parse", "--git-path", "index").stdout.strip()
        )
        object_dir = _resolved_git_path(
            _capture_git("rev-parse", "--git-path", "objects").stdout.strip()
        )
        object_format = _capture_git(
            "rev-parse",
            "--show-object-format",
        ).stdout.strip()
        inside_worktree = _capture_git(
            "rev-parse",
            "--is-inside-work-tree",
        ).stdout.strip()
        bare = _capture_git("rev-parse", "--is-bare-repository").stdout.strip()
        shallow = _capture_git(
            "rev-parse",
            "--is-shallow-repository",
        ).stdout.strip()
        shallow_path = _resolved_git_path(
            _capture_git("rev-parse", "--git-path", "shallow").stdout.strip()
        )
        replace_refs = _capture_git(
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace/",
        ).stdout.splitlines()
        local_config_sha256 = _capture_local_config_sha256()
        _assert_capture_index_has_no_hidden_flags()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot authenticate the Git repository/object view") from error

    expected_top_level = REPO_ROOT.resolve()
    if top_level != expected_top_level or inside_worktree != "true" or bare != "false":
        raise RuntimeError("Git top-level/worktree identity differs from the expected repository")
    if object_format != "sha1":
        raise RuntimeError("Stage-0 source identity requires the exact SHA-1 Git object format")
    if index_path != git_dir / "index":
        raise RuntimeError("Git index is not the exact worktree index")
    if object_dir != common_dir / "objects" or not object_dir.is_dir():
        raise RuntimeError("Git object directory is not the exact common object store")

    dot_git = expected_top_level / ".git"
    if dot_git.is_dir():
        if dot_git.is_symlink():
            raise RuntimeError("main-worktree .git directory must not be a symlink")
        git_dir_kind = "main_worktree"
        if git_dir != dot_git.resolve() or common_dir != git_dir:
            raise RuntimeError("main-worktree Git directory/common directory identity differs")
    elif dot_git.is_file():
        if dot_git.is_symlink():
            raise RuntimeError("linked-worktree .git marker must not be a symlink")
        marker = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not marker.startswith(prefix):
            raise RuntimeError("linked-worktree .git marker is malformed")
        declared_git_dir = Path(marker[len(prefix) :])
        if not declared_git_dir.is_absolute():
            declared_git_dir = expected_top_level / declared_git_dir
        if declared_git_dir.resolve() != git_dir:
            raise RuntimeError("linked-worktree .git marker redirects to a different Git directory")
        if git_dir.parent != common_dir / "worktrees":
            raise RuntimeError(
                "linked-worktree Git directory is outside the common worktree registry"
            )
        reverse_pointer = git_dir / "gitdir"
        if not reverse_pointer.is_file() or reverse_pointer.is_symlink():
            raise RuntimeError("linked-worktree Git directory has no canonical reverse pointer")
        declared_dot_git = Path(reverse_pointer.read_text(encoding="utf-8").strip())
        if not declared_dot_git.is_absolute():
            declared_dot_git = git_dir / declared_dot_git
        if declared_dot_git.resolve() != dot_git.resolve():
            raise RuntimeError(
                "linked-worktree Git directory reverse pointer targets a different worktree"
            )
        git_dir_kind = "linked_worktree"
    else:
        raise RuntimeError("repository has neither a canonical .git directory nor marker")

    unsafe_object_files = (
        object_dir / "info" / "alternates",
        object_dir / "info" / "http-alternates",
        git_dir / "info" / "grafts",
        common_dir / "info" / "grafts",
        shallow_path,
    )
    if any(path.exists() for path in unsafe_object_files):
        raise RuntimeError("Git alternates, grafts, or shallow object view is not permitted")
    if shallow != "false":
        raise RuntimeError("shallow Git history is not permitted")
    if replace_refs:
        raise RuntimeError("Git replacement refs are not permitted")

    return {
        "schema": GIT_REPOSITORY_BINDING_SCHEMA,
        "top_level_path_sha256": _private_path_sha256(top_level),
        "worktree_path_sha256": _private_path_sha256(expected_top_level),
        "git_dir_path_sha256": _private_path_sha256(git_dir),
        "common_dir_path_sha256": _private_path_sha256(common_dir),
        "index_path_sha256": _private_path_sha256(index_path),
        "object_dir_path_sha256": _private_path_sha256(object_dir),
        "git_dir_kind": git_dir_kind,
        "object_format": object_format,
        "inside_worktree": True,
        "bare": False,
        "shallow": False,
        "alternates_absent": True,
        "grafts_absent": True,
        "replace_refs_absent": True,
        "unsafe_local_config_absent": True,
        "hidden_index_flags_absent": True,
        "local_config_sha256": local_config_sha256,
        "replacement_objects_disabled": True,
        "system_and_global_config_disabled": True,
        "fsmonitor_and_untracked_cache_disabled": True,
        "hooks_disabled": True,
        "worktree_gitdir_binding_verified": True,
        "raw_source_hash_mode": "git_hash_object_no_filters_stdin_paths",
    }


def _git_blob_hashes_for_authenticated_sources() -> tuple[dict[str, str], dict[str, str]]:
    try:
        tree_process = _capture_git(
            "ls-tree",
            "-r",
            "--full-tree",
            "HEAD",
            "--",
            *SOURCE_IDENTITY_PATHS,
        )
        worktree_process = _capture_git(
            "hash-object",
            "--no-filters",
            "--stdin-paths",
            input_text="\n".join(SOURCE_IDENTITY_PATHS),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot compare authenticated source bytes with repository HEAD"
        ) from error
    head_blobs: dict[str, str] = {}
    for line in tree_process.stdout.splitlines():
        metadata, separator, relative = line.partition("\t")
        parts = metadata.split()
        if (
            separator != "\t"
            or len(parts) != 3
            or parts[0] not in {"100644", "100755"}
            or parts[1] != "blob"
            or relative not in SOURCE_IDENTITY_PATHS
        ):
            raise RuntimeError("authenticated source has an invalid repository tree entry")
        head_blobs[relative] = parts[2]
    if set(head_blobs) != set(SOURCE_IDENTITY_PATHS):
        missing = sorted(set(SOURCE_IDENTITY_PATHS) - set(head_blobs))
        raise RuntimeError(f"authenticated source is not a regular tracked HEAD blob: {missing}")
    worktree_lines = worktree_process.stdout.splitlines()
    if len(worktree_lines) != len(SOURCE_IDENTITY_PATHS):
        raise RuntimeError("Git did not hash every authenticated worktree source")
    worktree_blobs = dict(zip(SOURCE_IDENTITY_PATHS, worktree_lines, strict=True))
    malformed = [
        f"{kind}:{relative}"
        for kind, hashes in (("head", head_blobs), ("worktree", worktree_blobs))
        for relative, digest in hashes.items()
        if len(digest) != 40 or any(character not in "0123456789abcdef" for character in digest)
    ]
    if malformed:
        raise RuntimeError(f"authenticated source Git blob identity is malformed: {malformed}")
    return head_blobs, worktree_blobs


def _repository_source_snapshot() -> dict[str, object]:
    repository_binding = _capture_repository_binding()
    missing = [
        relative
        for relative in SOURCE_IDENTITY_PATHS
        if not (REPO_ROOT / relative).is_file() or (REPO_ROOT / relative).is_symlink()
    ]
    if missing:
        raise RuntimeError(f"authenticated Stage-0 source set is incomplete: {missing}")
    hashes = {relative: _file_sha256(REPO_ROOT / relative) for relative in SOURCE_IDENTITY_PATHS}
    head_blobs, worktree_blobs = _git_blob_hashes_for_authenticated_sources()
    try:
        head = _capture_git("rev-parse", "HEAD").stdout.strip()
        status = _capture_git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot authenticate repository source identity") from error
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("repository HEAD is not a lowercase SHA-1 commit identity")
    return {
        "repo_head": head,
        "repository_binding": repository_binding,
        "source_hashes": hashes,
        "source_set_sha256": canonical_payload_sha256(hashes),
        "head_blob_hashes": head_blobs,
        "worktree_blob_hashes": worktree_blobs,
        "sources_match_head": head_blobs == worktree_blobs,
        "worktree_clean": status == "",
    }


def _loaded_recurquant_module_paths() -> dict[str, str]:
    """Bind every loaded RecurQuant module name to its exact in-repository file."""

    required = dict(REQUIRED_LOADED_RECURQUANT_MODULE_PATHS)
    repo_root = REPO_ROOT.resolve()
    observed: dict[str, str] = {}
    for module_name, module in sorted(tuple(sys.modules.items())):
        if module_name != "recurquant" and not module_name.startswith("recurquant."):
            continue
        path_value = getattr(module, "__file__", None)
        if not isinstance(path_value, (str, os.PathLike)):
            raise RuntimeError(
                f"loaded RecurQuant module has no regular source file: {module_name}"
            )
        declared_path = Path(path_value)
        if declared_path.is_symlink():
            raise RuntimeError(f"loaded RecurQuant module source is a symlink: {module_name}")
        try:
            resolved_path = declared_path.resolve(strict=True)
            relative = resolved_path.relative_to(repo_root).as_posix()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"loaded RecurQuant module is outside the authenticated repository: {module_name}"
            ) from error
        if not resolved_path.is_file():
            raise RuntimeError(
                f"loaded RecurQuant module source is not a regular file: {module_name}"
            )
        observed[module_name] = relative
    missing = sorted(set(required) - set(observed))
    mismatched = sorted(
        module_name
        for module_name in set(required) & set(observed)
        if observed[module_name] != required[module_name]
    )
    unauthenticated = sorted(
        module_name
        for module_name, relative in observed.items()
        if relative not in SOURCE_IDENTITY_PATHS
        or relative
        not in {
            f"src/{module_name.replace('.', '/')}.py",
            f"src/{module_name.replace('.', '/')}/__init__.py",
        }
    )
    if missing or mismatched or unauthenticated:
        raise RuntimeError(
            "loaded RecurQuant module closure differs from the authenticated local closure "
            f"(missing={missing}, mismatched={mismatched}, "
            f"unauthenticated={unauthenticated})"
        )
    return observed


def _loaded_local_source_paths() -> tuple[str, ...]:
    module_paths = _loaded_recurquant_module_paths()
    producer = Path(__file__)
    expected_producer = (REPO_ROOT / "scripts" / "capture_statelease_stage0.py").resolve()
    if producer.is_symlink() or producer.resolve() != expected_producer:
        raise RuntimeError("Stage-0 producer source is not the authenticated local file")
    return tuple(sorted({"scripts/capture_statelease_stage0.py", *module_paths.values()}))


def _finalize_source_identity(
    capture_start: Mapping[str, object],
    capture_end: Mapping[str, object],
) -> dict[str, object]:
    if capture_start != capture_end:
        raise RuntimeError(
            "repository HEAD, worktree, or authenticated source changed during capture"
        )
    if capture_start["worktree_clean"] is not True:
        raise RuntimeError(
            "the complete repository worktree must be committed and clean before capture"
        )
    if capture_start["sources_match_head"] is not True:
        raise RuntimeError(
            "authenticated source bytes must exactly equal their regular-file blobs at HEAD"
        )
    loaded_module_paths = _loaded_recurquant_module_paths()
    loaded_paths = _loaded_local_source_paths()
    undeclared = sorted(set(loaded_paths) - set(SOURCE_IDENTITY_PATHS))
    if undeclared:
        raise RuntimeError(f"loaded local production source is unauthenticated: {undeclared}")
    start = {
        "repo_head": capture_start["repo_head"],
        "repository_binding": dict(capture_start["repository_binding"]),
        "source_hashes": dict(capture_start["source_hashes"]),
        "source_set_sha256": capture_start["source_set_sha256"],
        "head_blob_hashes": dict(capture_start["head_blob_hashes"]),
        "worktree_blob_hashes": dict(capture_start["worktree_blob_hashes"]),
        "sources_match_head": capture_start["sources_match_head"],
        "worktree_clean": capture_start["worktree_clean"],
    }
    end = {
        "repo_head": capture_end["repo_head"],
        "repository_binding": dict(capture_end["repository_binding"]),
        "source_hashes": dict(capture_end["source_hashes"]),
        "source_set_sha256": capture_end["source_set_sha256"],
        "head_blob_hashes": dict(capture_end["head_blob_hashes"]),
        "worktree_blob_hashes": dict(capture_end["worktree_blob_hashes"]),
        "sources_match_head": capture_end["sources_match_head"],
        "worktree_clean": capture_end["worktree_clean"],
    }
    return {
        "repo_head": start["repo_head"],
        "repository_binding": dict(start["repository_binding"]),
        "source_hashes": dict(start["source_hashes"]),
        "source_set_sha256": start["source_set_sha256"],
        "head_blob_hashes": dict(start["head_blob_hashes"]),
        "worktree_blob_hashes": dict(start["worktree_blob_hashes"]),
        "sources_match_head": True,
        "worktree_clean": True,
        "capture_start": start,
        "capture_end": end,
        "capture_start_equals_end": start == end,
        "loaded_local_source_paths": list(loaded_paths),
        "loaded_recurquant_module_paths": loaded_module_paths,
    }


def _runtime_package_manifest() -> tuple[dict[str, str], str]:
    packages = {
        distribution: str(importlib.metadata.version(distribution))
        for distribution in RUNTIME_PACKAGE_DISTRIBUTIONS
    }
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")) + "\n"
    manifest_sha256 = _sha256_bytes(payload.encode("utf-8"))
    if manifest_sha256 != PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256:
        raise RuntimeError(
            "the Experiment 012 runtime package manifest differs from the frozen identity"
        )
    return packages, manifest_sha256


def _runtime_identity() -> dict[str, object]:
    packages, package_manifest_sha256 = _runtime_package_manifest()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "python_environment": Path(sys.prefix).name,
        "datasets_version": packages["datasets"],
        "fsspec_version": packages["fsspec"],
        "huggingface_hub_version": packages["huggingface-hub"],
        "numpy_version": packages["numpy"],
        "pyarrow_version": packages["pyarrow"],
        "safetensors_version": packages["safetensors"],
        "tokenizers_version": packages["tokenizers"],
        "torch_version": packages["torch"],
        "transformers_version": packages["transformers"],
        "package_manifest_sha256": package_manifest_sha256,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "kernel_receipt_device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
    }


def build_production_artifact() -> dict[str, object]:
    """Run all production-only synthetic Stage-0 captures."""

    capture_start = _repository_source_snapshot()
    if capture_start["worktree_clean"] is not True:
        raise RuntimeError(
            "the complete repository worktree must be committed and clean before capture"
        )
    runtime_identity = _runtime_identity()
    config = _frozen_config()
    plan = _frozen_plan()
    cache = create_qwen35_experiment010_statelease_cache(
        config,
        plan=plan,
        record_evidence=True,
    )
    successful_kernel = _capture_successful_kernel_receipt()
    trace = _capture_statelease_trace(cache)
    rollback = _capture_rollback_receipt(cache)
    resident, source_states, query_ema = _capture_resident_snapshot(cache)
    storage_before_reset = cache.storage_summary()
    cache.reset()
    reset_snapshot, _, _ = _capture_resident_snapshot(cache)
    reset = {
        "resident_bytes_before_reset": storage_before_reset["resident_bytes_including_statelease"],
        "resident_bytes_after_reset": cache.resident_bytes_including_statelease(),
        "update_index_after_reset": cache._update_index,
        "evidence_count_after_reset": len(cache.update_evidence),
        "all_has_previous_state_flags_cleared": all(
            not layer.has_previous_state[0] for _, layer in cache.statelease_layers()
        ),
        "all_pending_observations_cleared": all(
            layer._pending_statelease_observation is None for _, layer in cache.statelease_layers()
        ),
        "post_reset_snapshot": reset_snapshot,
    }
    compatibility = _capture_cc1_compatibility(config, plan)
    equal_byte = _capture_equal_byte_comparators(source_states, query_ema)
    capture_end = _repository_source_snapshot()
    ending_runtime_identity = _runtime_identity()
    source_identity = _finalize_source_identity(capture_start, capture_end)
    if runtime_identity != ending_runtime_identity:
        raise RuntimeError("runtime identity changed during capture")
    identity_hashes = {
        "method": _sha256_bytes(STATELEASE_SELECTION_METHOD.encode("utf-8")),
        "plan": EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256,
        "source_set": source_identity["source_set_sha256"],
        "synthetic_manifest": canonical_payload_sha256([step["identity"] for step in trace]),
    }
    ordered_identities = [step["identity"] for step in trace]
    completed = {
        identity: canonical_payload_sha256(trace[index])
        for index, identity in enumerate(ordered_identities[:3])
    }
    artifact: dict[str, object] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "declarations": {
            "stage": "stage0",
            "synthetic_only": True,
            "quality_data_accessed": False,
            "protected_mbpp_window_accessed": False,
            "pretrained_checkpoint_loaded": False,
            "random_model_parameters_initialized": True,
            "tokenizer_loaded": False,
        },
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "method_identity": {
            "method": "StateLease-H5",
            "selection_method": STATELEASE_SELECTION_METHOD,
            "effective_plan_sha256": EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256,
            "seed": FROZEN_SEED,
            "replay_capacity": 5,
            "boundary_rule": "strictly_lower_c4_else_c5",
            "key_normalization": "consumed_dtype_then_fp32",
            "checkpoint_codec": "right_rht_sha256_signs_v1_q4_q8",
            "linear_layer_indices": list(LINEAR_LAYER_INDICES),
            "layer_quotas": {
                str(index): quota for index, quota in EXPERIMENT010_STATELEASE_LAYER_QUOTAS.items()
            },
        },
        "production_trace": trace,
        "successful_kernel_receipt": successful_kernel,
        "resident_snapshot": resident,
        "lifecycle": {
            "rollback": rollback,
            "reset": reset,
            "resume": {
                "prior_identity_hashes": identity_hashes,
                "resumed_identity_hashes": dict(identity_hashes),
                "expected_identities": ordered_identities,
                "completed_record_hashes": completed,
                "resumed_completed_record_hashes": dict(completed),
                "resumed_remaining_identities": ordered_identities[3:],
            },
        },
        "cc1_compatibility": compatibility,
        "equal_byte_comparators": equal_byte,
    }
    artifact["canonical_payload_sha256"] = canonical_payload_sha256(artifact)
    return artifact


def write_artifact(artifact: Mapping[str, object], path: Path) -> dict[str, object]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary = path.with_name(f".{path.name}.tmp")
    existing = [candidate.name for candidate in (path, sidecar, temporary) if candidate.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite an existing Stage-0 artifact output: {sorted(existing)}"
        )
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        ignored = _capture_git(
            "check-ignore",
            "--quiet",
            "--",
            relative.as_posix(),
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(
                "Stage-0 production artifacts inside the repository must use "
                "a Git-ignored destination"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), temporary)
    os.link(temporary, path)
    temporary.unlink()
    file_digest = _file_sha256(path)
    with sidecar.open("xb") as handle:
        handle.write(file_digest.encode("ascii") + b"  " + path.name.encode("ascii") + b"\n")
    return {
        "artifact": _public_output_label(path),
        "sidecar": _public_output_label(sidecar),
        "file_sha256": file_digest,
        "canonical_payload_sha256": artifact["canonical_payload_sha256"],
        "bytes": path.stat().st_size,
        "quality_data_accessed": False,
        "protected_mbpp_window_accessed": False,
    }


def _public_output_label(path: Path) -> str:
    """Return a stable report label without exposing an absolute local path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="destination .pt artifact",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact rather than indented JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = write_artifact(build_production_artifact(), args.output)
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
