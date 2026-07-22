"""Validated public factory for the supported Qwen3.5 cache integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import transformers
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .packed_cache import (
    AdaptiveMixedPackedRecurrentStateCache,
    CoraMixedPackedRecurrentStateCache,
    MixedPackedRecurrentStateCache,
    PackedRecurrentStateCache,
    QueryEmaMixedPackedRecurrentStateCache,
    RankFusedMixedPackedRecurrentStateCache,
)
from .quantization import QuantizationSpec, RoundingMode
from .row_policy import ExactBudgetRowPlan

if TYPE_CHECKING:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    Qwen35Source = Qwen3_5ForCausalLM | Qwen3_5TextConfig
else:
    Qwen35Source = object

_SUPPORTED_TRANSFORMERS_REQUIREMENT = "transformers==5.14.1"
_SUPPORTED_TRANSFORMERS_SPEC = SpecifierSet("==5.14.1")
_SUPPORTED_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})


def _validate_transformers_compatibility() -> Version:
    installed = transformers.__version__
    try:
        parsed = Version(installed)
    except InvalidVersion:
        raise RuntimeError(
            f"RecurQuant could not interpret Transformers version {installed!r}. "
            "Install a stable tested release with: "
            f"pip install '{_SUPPORTED_TRANSFORMERS_REQUIREMENT}'."
        ) from None
    if (
        parsed.is_prerelease
        or parsed.is_devrelease
        or parsed not in _SUPPORTED_TRANSFORMERS_SPEC
    ):
        raise RuntimeError(
            "RecurQuant's Qwen3.5 cache is tested only with "
            f"{_SUPPORTED_TRANSFORMERS_REQUIREMENT}; "
            f"found transformers=={installed}. Install the tested release before creating "
            "the cache."
        )
    return parsed


def _load_qwen_classes() -> tuple[type[Any], type[Any]]:
    try:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError as exc:
        raise RuntimeError(
            "The installed Transformers build does not provide the tested Qwen3.5 "
            "classes. Install the supported build with: "
            f"pip install '{_SUPPORTED_TRANSFORMERS_REQUIREMENT}'."
        ) from exc
    return Qwen3_5ForCausalLM, Qwen3_5TextConfig


def _mapped_device_label(device: object) -> str:
    if isinstance(device, int) and not isinstance(device, bool):
        return f"cuda:{device}"
    return str(device).lower()


def _validate_model_runtime(model: torch.nn.Module, config: object) -> None:
    if model.training:
        raise ValueError(
            "The packed Qwen3.5 cache is inference-only. Call model.eval() before "
            "create_qwen35_packed_cache(model, ...), and run every prefill/decode "
            "forward inside torch.inference_mode() or torch.no_grad(); eval mode alone "
            "does not disable autograd."
        )

    attention_implementation = getattr(config, "_attn_implementation", None)
    if attention_implementation != "eager":
        raise ValueError(
            "This Qwen3.5 model is not configured for the tested eager attention path "
            f"(found {attention_implementation!r}). Reload it with "
            "attn_implementation='eager' and then create the cache."
        )

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping) and device_map:
        mapped_devices = {_mapped_device_label(device) for device in device_map.values()}
        if mapped_devices & {"disk", "meta"}:
            raise ValueError(
                "RecurQuant does not support disk-offloaded or meta-device Qwen3.5 "
                "models. Reload the complete model on one CPU or one CUDA device "
                "without disk offload."
            )
        if len(mapped_devices) > 1:
            raise ValueError(
                "RecurQuant does not support sharded or multi-device Qwen3.5 models. "
                "Reload the complete model on one CPU or one CUDA device without "
                "device_map='auto'."
            )

    actual_devices = {parameter.device for parameter in model.parameters()}
    actual_devices.update(buffer.device for buffer in model.buffers())
    if any(device.type == "meta" for device in actual_devices):
        raise ValueError(
            "RecurQuant cannot use a Qwen3.5 model containing meta tensors. Materialize "
            "all weights on one CPU or one CUDA device before creating the cache."
        )
    if len(actual_devices) != 1:
        rendered_devices = sorted(str(device) for device in actual_devices) or ["none"]
        raise ValueError(
            "RecurQuant requires the complete Qwen3.5 model on exactly one device; "
            f"found {rendered_devices}. Move all parameters and buffers to one CPU or "
            "one CUDA device before creating the cache."
        )


def _validated_text_config(model_or_config: Qwen35Source) -> Any:
    qwen_model_class, qwen_config_class = _load_qwen_classes()
    if isinstance(model_or_config, qwen_model_class):
        config = model_or_config.config
        _validate_model_runtime(model_or_config, config)
    elif isinstance(model_or_config, qwen_config_class):
        config = model_or_config
    else:
        actual = type(model_or_config).__name__
        raise TypeError(
            "create_qwen35_packed_cache expects a transformers.Qwen3_5ForCausalLM "
            f"model or Qwen3_5TextConfig, got {actual}. Load a text-only Qwen3.5 causal "
            "language model and pass the evaluated model or its config. Passing only "
            "a config validates structure, not the model's runtime placement or "
            "attention backend."
        )

    layer_types = getattr(config, "layer_types", None)
    number_of_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(layer_types, (list, tuple)) or not isinstance(number_of_layers, int):
        raise ValueError(
            "The Qwen3.5 config must define layer_types and num_hidden_layers. Reload the "
            "model with the tested Transformers version instead of modifying its config."
        )
    if len(layer_types) != number_of_layers:
        raise ValueError(
            "The Qwen3.5 config is inconsistent: len(layer_types) must equal "
            f"num_hidden_layers ({len(layer_types)} != {number_of_layers}). Reload the "
            "original model config."
        )

    unknown_layer_types = set(layer_types) - _SUPPORTED_LAYER_TYPES
    if unknown_layer_types:
        raise ValueError(
            "The Qwen3.5 config contains unsupported layer types "
            f"{sorted(unknown_layer_types)}. RecurQuant currently supports only "
            "linear_attention and full_attention layers."
        )
    if "linear_attention" not in layer_types:
        raise ValueError(
            "This Qwen3.5 config has no linear_attention layers, so it has no supported "
            "recurrent state to pack. Use a hybrid Qwen3.5 text checkpoint such as "
            "Qwen/Qwen3.5-0.8B-Base."
        )
    return config


def create_qwen35_packed_cache(
    model_or_config: Qwen35Source,
    *,
    bits: int = 4,
    group_size: int = 128,
    scale_bits: int = 16,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    layer_specs: Mapping[int, QuantizationSpec] | None = None,
    record_evidence: bool = False,
) -> PackedRecurrentStateCache:
    """Create a validated packed recurrent-state cache for Qwen3.5 inference.

    This factory covers text-only ``Qwen3_5ForCausalLM`` models on the tested stable
    Transformers release. Passing a model additionally validates evaluation
    mode, eager attention, and single-device materialization. Passing only a config
    validates its layer structure; it cannot validate the eventual model runtime.
    Every model forward must run under ``torch.inference_mode()`` or
    ``torch.no_grad()``. ``bits`` may be 4 or 8. Layer-specific specs use model-layer
    indices and are checked against the model's linear-attention layers.

    The returned cache physically packs persistent recurrent states. Whether this
    uses fewer resident bytes depends on tensor shape, grouping, payload width, and
    scale overhead; inspect ``storage_summary()`` and its
    ``physical_reduction_realized`` field after an update. The current PyTorch path
    materializes one state while that layer executes and makes no latency or
    whole-model peak-memory claim.
    """

    _validate_transformers_compatibility()
    config = _validated_text_config(model_or_config)
    if bits not in (4, 8):
        raise ValueError(
            f"The packed cache supports bits=4 or bits=8, got bits={bits}. "
            "Use quantize_dequantize for numerical experiments at other bit widths."
        )

    spec = QuantizationSpec(
        bits=bits,
        group_size=group_size,
        scale_bits=scale_bits,
        rounding=rounding,
        seed=seed,
    )
    return PackedRecurrentStateCache(
        config,
        spec=spec,
        layer_specs=layer_specs,
        record_evidence=record_evidence,
    )


def _validated_exact_budget_config(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
) -> object:
    _validate_transformers_compatibility()
    config = _validated_text_config(model_or_config)
    geometry_fields = {
        "linear_num_value_heads": getattr(config, "linear_num_value_heads", None),
        "linear_key_head_dim": getattr(config, "linear_key_head_dim", None),
        "linear_value_head_dim": getattr(config, "linear_value_head_dim", None),
    }
    invalid_geometry = {
        name: value
        for name, value in geometry_fields.items()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0
    }
    if invalid_geometry:
        raise ValueError(
            f"The Qwen3.5 config has invalid recurrent-state geometry: {invalid_geometry}"
        )

    expected_heads = geometry_fields["linear_num_value_heads"]
    expected_rows = geometry_fields["linear_key_head_dim"]
    expected_group_size = geometry_fields["linear_value_head_dim"]
    if plan.group_size != expected_group_size:
        raise ValueError(
            "row plan group_size must equal Qwen3.5 linear_value_head_dim: "
            f"{plan.group_size} != {expected_group_size}"
        )
    incompatible_shapes = [
        (layer_index, head_count, row_count)
        for layer_index, head_count, row_count in plan.score_shapes
        if head_count != expected_heads or row_count != expected_rows
    ]
    if incompatible_shapes:
        raise ValueError(
            "row plan score geometry must match Qwen3.5 recurrent states "
            f"[heads={expected_heads}, rows={expected_rows}]; incompatible entries: "
            f"{incompatible_shapes}"
        )
    return config


def create_qwen35_exact_budget_cache(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
) -> MixedPackedRecurrentStateCache:
    """Create a validated row-level mixed INT4/INT8 Qwen3.5 cache.

    The plan must cover every linear-attention model-layer index exactly. Its
    ``[heads, rows]`` score geometry must equal Qwen3.5's recurrent state geometry,
    and its group size must equal ``linear_value_head_dim`` so every state row owns
    one independently packed group. No layer, geometry, or byte-budget adaptation
    is performed implicitly.
    """

    return MixedPackedRecurrentStateCache(
        _validated_exact_budget_config(model_or_config, plan=plan),
        plan=plan,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
    )


def create_qwen35_adaptive_exact_budget_cache(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
) -> AdaptiveMixedPackedRecurrentStateCache:
    """Create the batch-one adaptive-row mixed-cache prototype for Qwen3.5.

    The exact plan still fixes each layer's INT8 promotion count and resident
    byte count. On every state write, row identities are selected by instantaneous
    aligned INT4-versus-INT8 quantization-error reduction. This experimental
    factory does not establish a quality, latency, or novelty result.
    """

    return AdaptiveMixedPackedRecurrentStateCache(
        _validated_exact_budget_config(model_or_config, plan=plan),
        plan=plan,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
    )


def create_qwen35_query_ema_exact_budget_cache(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
    confirmation_two: bool = False,
) -> QueryEmaMixedPackedRecurrentStateCache:
    """Create the frozen half-life-32 query-EMA mixed cache for Qwen3.5.

    The exact plan continues to fix each layer's promotion quota and packed-state
    bytes. Before every recurrent-state update, a matching post-convolution query
    must be supplied through ``stage_query_observation`` (normally by the Qwen3.5
    query observer). Missing or invalid observations fail closed. Persistent FP32
    EMA metadata is reported separately from packed recurrent-state storage.
    """

    return QueryEmaMixedPackedRecurrentStateCache(
        _validated_exact_budget_config(model_or_config, plan=plan),
        plan=plan,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
        confirmation_two=confirmation_two,
    )


def create_qwen35_cora_exact_budget_cache(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
    confirmation_two: bool = True,
) -> CoraMixedPackedRecurrentStateCache:
    """Create the frozen causal-observability Qwen3.5 mixed cache.

    The exact row plan fixes per-layer promotion quotas and packed-state bytes.
    A transition observer must stage the successful Gated DeltaNet kernel's
    post-convolution query, key, log-decay, and beta tensors before every state
    write. ``confirmation_two=False`` selects the frozen raw-CORA ablation.
    """

    return CoraMixedPackedRecurrentStateCache(
        _validated_exact_budget_config(model_or_config, plan=plan),
        plan=plan,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
        confirmation_two=confirmation_two,
    )


def create_qwen35_rank_fused_exact_budget_cache(
    model_or_config: Qwen35Source,
    *,
    plan: ExactBudgetRowPlan,
    static_scores_by_layer: Mapping[int, torch.Tensor],
    static_rank_weight: float,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
) -> RankFusedMixedPackedRecurrentStateCache:
    """Create an exact-byte static/dynamic rank-fusion cache for Qwen3.5.

    The row plan fixes each layer's promotion quota. Within that quota, one
    global weight fuses calibrated static-score ranks with causal per-write ranks
    of aligned INT4-to-INT8 MSE reduction. Static score tensors must cover the
    plan's layers exactly and reside on the same device as recurrent states.
    """

    return RankFusedMixedPackedRecurrentStateCache(
        _validated_exact_budget_config(model_or_config, plan=plan),
        plan=plan,
        static_scores_by_layer=static_scores_by_layer,
        static_rank_weight=static_rank_weight,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
    )


def create_qwen35_v02_mixed_cache(
    model_or_config: Qwen35Source,
    *,
    record_evidence: bool = False,
) -> PackedRecurrentStateCache:
    """Create the frozen v0.2 mixed-precision Qwen3.5 cache.

    The policy stores model layer 0 at INT8 and every other supported recurrent
    layer at INT4, with group size 128, FP16 scales, nearest rounding, and seed
    2339. It is the fixed development policy reported by the repository, not an
    automatic selector for other checkpoints.
    """

    return create_qwen35_packed_cache(
        model_or_config,
        bits=4,
        group_size=128,
        scale_bits=16,
        rounding="nearest",
        seed=2339,
        layer_specs={
            0: QuantizationSpec(
                bits=8,
                group_size=128,
                scale_bits=16,
                rounding="nearest",
                seed=2339,
            )
        },
        record_evidence=record_evidence,
    )
