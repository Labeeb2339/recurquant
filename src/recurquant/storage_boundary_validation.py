"""Independent one-row validation at a recurrent-state storage boundary.

This module checks one local claim made by the loss-sensitivity selector: for a
single recurrent-state row, ``-g dot (Q8 - Q4)`` should approximate the target
token NLL reduction obtained by replacing that stored row's physical INT4
endpoint with its physical INT8 endpoint.  It deliberately does not select a
policy or measure model quality.

The validation starts from a raw, warm ``transformers.DynamicCache``.  Every
initialized recurrent state is physically packed and dequantized as INT4.  The
selected row then follows the controlled path

``Q4(raw) + alpha * (Q8(raw) - Q4(raw))``.

Only ``alpha=0`` and ``alpha=1`` are deployable quantized endpoints.  Interior
and negative values exist solely to make an independent central-difference
check possible at the exact state consumed by the next model token.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .quantization import QuantizationSpec, quantize_pack


@dataclass(frozen=True, order=True, slots=True)
class StorageRowLocation:
    """One batch-one Qwen3.5 recurrent-state row."""

    layer_index: int
    head_index: int
    row_index: int

    def __post_init__(self) -> None:
        values = (self.layer_index, self.head_index, self.row_index)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("storage-row indices must be non-negative integers")


@dataclass(frozen=True, slots=True)
class DirectionalBenefitComparison:
    """Scalar derivative and endpoint comparison along one row direction."""

    epsilon: float
    loss_at_zero: float
    repeated_loss_at_zero: float
    loss_at_minus_epsilon: float
    loss_at_plus_epsilon: float
    loss_at_int8_endpoint: float
    autograd_directional_derivative: float
    central_directional_derivative: float
    predicted_benefit_autograd: float
    predicted_benefit_central: float
    measured_endpoint_benefit: float
    derivative_absolute_error: float
    endpoint_taylor_residual: float
    baseline_repeat_absolute_error: float
    derivative_sign_agreement: bool | None
    endpoint_sign_agreement: bool | None


@dataclass(frozen=True, slots=True)
class StorageBoundaryRowValidation:
    """Detached result of one real-model storage-boundary validation."""

    location: StorageRowLocation
    comparison: DirectionalBenefitComparison
    recurrent_layer_indices: tuple[int, ...]
    raw_row: torch.Tensor = field(repr=False)
    int4_row: torch.Tensor = field(repr=False)
    int8_row: torch.Tensor = field(repr=False)
    direction: torch.Tensor = field(repr=False)
    gradient_row: torch.Tensor = field(repr=False)


def _finite_scalar(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _directional_dot_float64(
    gradient_row: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """Accumulate a saved FP32 row dot product reproducibly on CPU in FP64."""

    if gradient_row.shape != direction.shape:
        raise ValueError("gradient row and direction must have the same shape")
    if gradient_row.dtype != torch.float32 or direction.dtype != torch.float32:
        raise TypeError("gradient row and direction must use float32 before FP64 accumulation")
    gradient64 = gradient_row.detach().to(device="cpu", dtype=torch.float64)
    direction64 = direction.detach().to(device="cpu", dtype=torch.float64)
    value = float((gradient64 * direction64).sum(dtype=torch.float64).item())
    if not math.isfinite(value):
        raise RuntimeError("directional dot product became non-finite")
    return value


def _sign_agreement(left: float, right: float, *, floor: float) -> bool | None:
    if abs(left) <= floor and abs(right) <= floor:
        return None
    if abs(left) <= floor or abs(right) <= floor:
        return False
    return (left > 0) == (right > 0)


def compare_directional_benefits(
    *,
    epsilon: float,
    loss_at_zero: float,
    repeated_loss_at_zero: float,
    loss_at_minus_epsilon: float,
    loss_at_plus_epsilon: float,
    loss_at_int8_endpoint: float,
    autograd_directional_derivative: float,
    sign_floor: float = 1e-12,
) -> DirectionalBenefitComparison:
    """Compare autograd, central difference, and the exact INT8 endpoint.

    Derivatives have the sign of increasing NLL.  Benefits use the selector's
    opposite sign: positive means that moving toward the INT8 row lowers NLL.
    A sign comparison is ``None`` only when both quantities are numerically
    indistinguishable from zero under ``sign_floor``.
    """

    epsilon_value = _finite_scalar("epsilon", epsilon)
    floor_value = _finite_scalar("sign_floor", sign_floor)
    if epsilon_value <= 0 or epsilon_value > 1:
        raise ValueError("epsilon must be in (0, 1]")
    if floor_value < 0:
        raise ValueError("sign_floor must be non-negative")

    zero = _finite_scalar("loss_at_zero", loss_at_zero)
    repeated_zero = _finite_scalar("repeated_loss_at_zero", repeated_loss_at_zero)
    minus = _finite_scalar("loss_at_minus_epsilon", loss_at_minus_epsilon)
    plus = _finite_scalar("loss_at_plus_epsilon", loss_at_plus_epsilon)
    endpoint = _finite_scalar("loss_at_int8_endpoint", loss_at_int8_endpoint)
    autograd = _finite_scalar(
        "autograd_directional_derivative", autograd_directional_derivative
    )
    central = (plus - minus) / (2.0 * epsilon_value)
    predicted_autograd = -autograd
    predicted_central = -central
    measured_endpoint = zero - endpoint

    return DirectionalBenefitComparison(
        epsilon=epsilon_value,
        loss_at_zero=zero,
        repeated_loss_at_zero=repeated_zero,
        loss_at_minus_epsilon=minus,
        loss_at_plus_epsilon=plus,
        loss_at_int8_endpoint=endpoint,
        autograd_directional_derivative=autograd,
        central_directional_derivative=central,
        predicted_benefit_autograd=predicted_autograd,
        predicted_benefit_central=predicted_central,
        measured_endpoint_benefit=measured_endpoint,
        derivative_absolute_error=abs(autograd - central),
        endpoint_taylor_residual=measured_endpoint - predicted_autograd,
        baseline_repeat_absolute_error=abs(zero - repeated_zero),
        derivative_sign_agreement=_sign_agreement(
            autograd,
            central,
            floor=floor_value,
        ),
        endpoint_sign_agreement=_sign_agreement(
            predicted_autograd,
            measured_endpoint,
            floor=floor_value,
        ),
    )


def interpolate_storage_row(
    int4_state: torch.Tensor,
    direction: torch.Tensor,
    *,
    location: StorageRowLocation,
    alpha: float,
) -> torch.Tensor:
    """Return a detached state with exactly one row moved along ``direction``."""

    alpha_value = _finite_scalar("alpha", alpha)
    if int4_state.ndim != 4 or int4_state.shape[0] != 1:
        raise ValueError("int4_state must have shape [1, heads, rows, value_dim]")
    if not int4_state.is_floating_point() or int4_state.device.type == "meta":
        raise TypeError("int4_state must be a materialized floating-point tensor")
    if direction.shape != int4_state.shape[-1:]:
        raise ValueError("direction must have shape [value_dim]")
    if direction.dtype != int4_state.dtype or direction.device != int4_state.device:
        raise ValueError("direction must match int4_state dtype and device")
    if location.head_index >= int4_state.shape[1]:
        raise ValueError("head_index is outside the recurrent-state geometry")
    if location.row_index >= int4_state.shape[2]:
        raise ValueError("row_index is outside the recurrent-state geometry")
    if not torch.isfinite(int4_state).all().item() or not torch.isfinite(direction).all().item():
        raise ValueError("int4_state and direction must contain only finite values")

    moved = int4_state.detach().clone()
    moved[0, location.head_index, location.row_index] = (
        moved[0, location.head_index, location.row_index] + alpha_value * direction
    )
    return moved


def _validate_quantization_specs(
    int4_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
) -> None:
    if int4_spec.bits != 4 or int8_spec.bits != 8:
        raise ValueError("storage-boundary validation requires INT4 and INT8 specs")
    matched = ("group_size", "scale_bits", "flatten_last_dims", "rounding", "seed", "epsilon")
    mismatched = [name for name in matched if getattr(int4_spec, name) != getattr(int8_spec, name)]
    if mismatched:
        raise ValueError(f"INT4 and INT8 specs may differ only in bits: {mismatched}")
    if int4_spec.flatten_last_dims != 1:
        raise ValueError("flatten_last_dims must be 1 so each state row owns its quantizer group")
    if int4_spec.rounding != "nearest":
        raise ValueError("the deterministic validation path currently requires nearest rounding")


def _recurrent_state(cache: DynamicCache, layer_index: int) -> torch.Tensor:
    if layer_index >= len(cache.layers):
        raise ValueError(f"cache has no layer {layer_index}")
    layer = cache.layers[layer_index]
    states = getattr(layer, "recurrent_states", None)
    initialized = getattr(layer, "is_recurrent_states_initialized", None)
    previous = getattr(layer, "has_previous_state", None)
    if not isinstance(states, dict) or not isinstance(initialized, dict):
        raise ValueError(f"cache layer {layer_index} is not a recurrent-state layer")
    if not initialized.get(0, False) or not isinstance(states.get(0), torch.Tensor):
        raise ValueError(f"cache layer {layer_index} has no initialized recurrent state")
    if not isinstance(previous, dict) or not previous.get(0, False):
        raise ValueError(f"cache layer {layer_index} has no previous recurrent state")
    state = states[0]
    if state.dtype != torch.float32:
        raise TypeError(f"cache layer {layer_index} recurrent state must use float32")
    if state.ndim != 4 or state.shape[0] != 1:
        raise ValueError(
            f"cache layer {layer_index} recurrent state must have batch-one rank-4 geometry"
        )
    if not torch.isfinite(state).all().item():
        raise ValueError(f"cache layer {layer_index} recurrent state is non-finite")
    return state


def recurrent_layer_indices(cache: DynamicCache) -> tuple[int, ...]:
    """Return initialized recurrent-state layers from a warm DynamicCache."""

    if not isinstance(cache, DynamicCache):
        raise TypeError("cache must be a transformers.DynamicCache")
    if getattr(cache, "offloading", False):
        raise ValueError("offloaded DynamicCache validation is not supported")
    indices: list[int] = []
    for layer_index, layer in enumerate(cache.layers):
        states = getattr(layer, "recurrent_states", None)
        initialized = getattr(layer, "is_recurrent_states_initialized", None)
        if isinstance(states, dict) and isinstance(initialized, dict) and initialized.get(0, False):
            _recurrent_state(cache, layer_index)
            indices.append(layer_index)
    if not indices:
        raise ValueError("cache has no initialized recurrent states; run a no-grad prefill first")
    return tuple(indices)


def store_uniform_int4_boundary(
    cache: DynamicCache,
    *,
    int4_spec: QuantizationSpec,
) -> tuple[int, ...]:
    """Physically pack/dequantize every warm recurrent state as INT4 in place."""

    if int4_spec.bits != 4 or int4_spec.flatten_last_dims != 1:
        raise ValueError("int4_spec must use 4 bits and flatten_last_dims=1")
    indices = recurrent_layer_indices(cache)
    for layer_index in indices:
        state = _recurrent_state(cache, layer_index)
        if int4_spec.group_size != state.shape[-1]:
            raise ValueError(
                "int4 group_size must equal recurrent-state value_dim at layer "
                f"{layer_index} ({int4_spec.group_size} != {state.shape[-1]})"
            )
        packed = quantize_pack(state, int4_spec)
        cache.layers[layer_index].recurrent_states[0] = packed.dequantize()
    return indices


def advance_uniform_int4_trajectory(
    model: torch.nn.Module,
    raw_cache: DynamicCache,
    previous_input_ids: torch.Tensor,
    *,
    int4_spec: QuantizationSpec,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Advance prior tokens and stop at the next raw pre-storage boundary.

    ``raw_cache`` must initially contain the raw prefill update.  Before every
    prior token, all recurrent states are stored through physical INT4 packing.
    The last prior token's update remains raw, ready for the validation function
    to construct both Q4 and Q8 endpoints.
    """

    if model.training:
        raise ValueError("model must be in evaluation mode")
    if previous_input_ids.ndim != 2 or previous_input_ids.shape[0] != 1:
        raise ValueError("previous_input_ids must have shape [1, tokens]")
    if previous_input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("previous_input_ids must use an integer dtype")
    kwargs = dict(forward_kwargs or {})
    conflicts = {"input_ids", "past_key_values", "use_cache", "labels"} & set(kwargs)
    if conflicts:
        raise ValueError(f"forward_kwargs contains managed arguments: {sorted(conflicts)}")

    for token_index in range(previous_input_ids.shape[1]):
        store_uniform_int4_boundary(raw_cache, int4_spec=int4_spec)
        with torch.no_grad():
            model(
                input_ids=previous_input_ids[:, token_index : token_index + 1],
                past_key_values=raw_cache,
                use_cache=True,
                **kwargs,
            )


def _target_nll(
    model: torch.nn.Module,
    cache: DynamicCache,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    output = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        **forward_kwargs,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("model output must contain rank-3 logits")
    return F.cross_entropy(logits[:, -1, :].to(torch.float32), targets)


def _loss_at_alpha(
    model: torch.nn.Module,
    base_cache: DynamicCache,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    location: StorageRowLocation,
    direction: torch.Tensor,
    alpha: float,
    forward_kwargs: Mapping[str, Any],
) -> float:
    cache = copy.deepcopy(base_cache)
    state = _recurrent_state(cache, location.layer_index)
    cache.layers[location.layer_index].recurrent_states[0] = interpolate_storage_row(
        state,
        direction,
        location=location,
        alpha=alpha,
    )
    with torch.no_grad():
        loss = _target_nll(model, cache, input_ids, targets, forward_kwargs)
    value = float(loss.item())
    if not math.isfinite(value):
        raise RuntimeError("target NLL became non-finite")
    return value


def _autograd_directional_derivative(
    model: torch.nn.Module,
    base_cache: DynamicCache,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    location: StorageRowLocation,
    direction: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
) -> tuple[float, float, torch.Tensor]:
    cache = copy.deepcopy(base_cache)
    state = _recurrent_state(cache, location.layer_index)
    leaf = state.detach().clone().requires_grad_(True)
    cache.layers[location.layer_index].recurrent_states[0] = leaf

    recurrent_layers = frozenset(recurrent_layer_indices(cache))
    had_instance_update = "update_recurrent_state" in cache.__dict__
    previous_instance_update = cache.__dict__.get("update_recurrent_state")

    def differentiable_update(
        working_cache: DynamicCache,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **update_kwargs: Any,
    ) -> torch.Tensor:
        del update_kwargs
        if layer_idx not in recurrent_layers:
            raise ValueError(
                f"unexpected recurrent-state update for unprepared layer {layer_idx}"
            )
        if state_idx != 0:
            raise ValueError("Qwen3.5 storage-boundary validation supports state_idx=0 only")
        layer = working_cache.layers[layer_idx]
        states = getattr(layer, "recurrent_states", None)
        initialized = getattr(layer, "is_recurrent_states_initialized", None)
        if not isinstance(states, dict) or not isinstance(initialized, dict):
            raise RuntimeError(f"cache layer {layer_idx} lost recurrent-state metadata")
        if not initialized.get(state_idx, False):
            raise RuntimeError(f"cache layer {layer_idx} was not warm at the measured boundary")

        # Transformers normally copy_()s every updated GDN state into its warm
        # cache tensor to preserve a static address. During this differentiable
        # diagnostic, later GDN updates can depend on the selected leaf. Their
        # CopyBackwards destinations would overwrite tensors saved by the
        # recurrent kernels before autograd runs. Assignment preserves the exact
        # value stored for the next token without mutating any current-token
        # input in place. Only ``leaf`` remains a differentiation target.
        layer.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("model parameters must not have pre-existing gradients")
    original_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    cache.update_recurrent_state = MethodType(differentiable_update, cache)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            loss = _target_nll(model, cache, input_ids, targets, forward_kwargs)
            (gradient,) = torch.autograd.grad(loss, leaf, retain_graph=False, create_graph=False)
        gradient_row = gradient[0, location.head_index, location.row_index].detach()
        directional = _directional_dot_float64(gradient_row, direction)
        loss_value = float(loss.detach().item())
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)
        if had_instance_update:
            cache.update_recurrent_state = previous_instance_update
        else:
            del cache.__dict__["update_recurrent_state"]

    if not math.isfinite(loss_value) or not math.isfinite(directional):
        raise RuntimeError("autograd storage-boundary result became non-finite")
    if not torch.isfinite(gradient_row).all().item():
        raise RuntimeError("storage-boundary row gradient became non-finite")
    return loss_value, directional, gradient_row.cpu()


def validate_qwen_storage_boundary_row(
    model: torch.nn.Module,
    raw_cache: DynamicCache,
    input_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    location: StorageRowLocation,
    int4_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
    epsilon: float = 0.5,
    sign_floor: float = 1e-12,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> StorageBoundaryRowValidation:
    """Validate one Qwen row with autograd, central difference, and endpoints.

    ``raw_cache`` is not mutated.  It must describe the raw recurrent update at
    the chosen token boundary, after any earlier tokens have followed the
    repeated-INT4 trajectory.  The function performs five bounded one-token
    forwards: one autograd baseline and no-grad evaluations at alpha 0,
    ``-epsilon``, ``+epsilon``, and 1.
    """

    _validate_quantization_specs(int4_spec, int8_spec)
    if model.training:
        raise ValueError("model must be in evaluation mode")
    if input_ids.ndim != 2 or input_ids.shape != (1, 1):
        raise ValueError("input_ids must have shape [1, 1]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must use an integer dtype")
    targets = target_token_ids
    if targets.ndim == 2 and targets.shape == (1, 1):
        targets = targets[:, 0]
    if targets.ndim != 1 or targets.shape[0] != 1:
        raise ValueError("target_token_ids must have shape [1] or [1, 1]")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_token_ids must use an integer dtype")
    if targets.device != input_ids.device:
        raise ValueError("input_ids and target_token_ids must be on the same device")
    kwargs = dict(forward_kwargs or {})
    conflicts = {"input_ids", "past_key_values", "use_cache", "labels"} & set(kwargs)
    if conflicts:
        raise ValueError(f"forward_kwargs contains managed arguments: {sorted(conflicts)}")

    raw_state = _recurrent_state(raw_cache, location.layer_index)
    if raw_state.device != input_ids.device:
        raise ValueError("input_ids and recurrent-state cache must be on the same device")
    if int4_spec.group_size != raw_state.shape[-1]:
        raise ValueError(
            "quantizer group_size must equal recurrent-state value_dim "
            f"({int4_spec.group_size} != {raw_state.shape[-1]})"
        )
    if location.head_index >= raw_state.shape[1] or location.row_index >= raw_state.shape[2]:
        raise ValueError("selected row is outside the recurrent-state geometry")

    int4_state = quantize_pack(raw_state, int4_spec).dequantize()
    int8_state = quantize_pack(raw_state, int8_spec).dequantize()
    direction = (
        int8_state[0, location.head_index, location.row_index]
        - int4_state[0, location.head_index, location.row_index]
    ).detach()
    if not torch.isfinite(direction).all().item():
        raise RuntimeError("INT4-to-INT8 row direction became non-finite")

    base_cache = copy.deepcopy(raw_cache)
    layers = store_uniform_int4_boundary(base_cache, int4_spec=int4_spec)
    base_selected = _recurrent_state(base_cache, location.layer_index)
    if not torch.equal(base_selected, int4_state):
        raise RuntimeError("selected INT4 storage endpoint was not reproduced exactly")

    loss_zero, autograd_derivative, gradient_row = _autograd_directional_derivative(
        model,
        base_cache,
        input_ids,
        targets,
        location=location,
        direction=direction,
        forward_kwargs=kwargs,
    )
    repeated_zero = _loss_at_alpha(
        model,
        base_cache,
        input_ids,
        targets,
        location=location,
        direction=direction,
        alpha=0.0,
        forward_kwargs=kwargs,
    )
    minus = _loss_at_alpha(
        model,
        base_cache,
        input_ids,
        targets,
        location=location,
        direction=direction,
        alpha=-float(epsilon),
        forward_kwargs=kwargs,
    )
    plus = _loss_at_alpha(
        model,
        base_cache,
        input_ids,
        targets,
        location=location,
        direction=direction,
        alpha=float(epsilon),
        forward_kwargs=kwargs,
    )
    endpoint = _loss_at_alpha(
        model,
        base_cache,
        input_ids,
        targets,
        location=location,
        direction=direction,
        alpha=1.0,
        forward_kwargs=kwargs,
    )
    comparison = compare_directional_benefits(
        epsilon=epsilon,
        loss_at_zero=loss_zero,
        repeated_loss_at_zero=repeated_zero,
        loss_at_minus_epsilon=minus,
        loss_at_plus_epsilon=plus,
        loss_at_int8_endpoint=endpoint,
        autograd_directional_derivative=autograd_derivative,
        sign_floor=sign_floor,
    )

    return StorageBoundaryRowValidation(
        location=location,
        comparison=comparison,
        recurrent_layer_indices=layers,
        raw_row=raw_state[0, location.head_index, location.row_index].detach().cpu(),
        int4_row=int4_state[0, location.head_index, location.row_index].detach().cpu(),
        int8_row=int8_state[0, location.head_index, location.row_index].detach().cpu(),
        direction=direction.cpu(),
        gradient_row=gradient_row,
    )
