"""One-step loss sensitivity for Qwen3.5 recurrent-state rows.

This module measures how the next-token loss responds to recurrent states in
an ordinary Transformers ``DynamicCache``. It supports either the untouched
FP32 trajectory or a repeated-QDQ INT4 trajectory. Both are deliberately
bounded calibration primitives: one teacher-forced token, one reverse-mode
traversal, and no retained graph after the step.

The returned scores estimate the benefit of replacing an INT4 row by its INT8
version. They are calibration signals, not evidence of downstream accuracy,
memory, or latency improvements.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .quantization import (
    QuantizationSpec,
    quantize_dequantize,
    scheduled_quantization_spec,
)


@dataclass(frozen=True, slots=True)
class RowPromotionSensitivityScores:
    """Signed INT4-to-INT8 sensitivity scores for one recurrent-state layer.

    Every tensor has shape ``[heads, key_dim]`` and is the batch mean of
    per-example row scores. ``taylor_benefit`` is the primary first-order
    estimate of loss reduction, so positive values favor promotion to INT8.
    The Fisher fields are signed differences, not guaranteed improvements.
    ``delta_direction_fisher_magnitude`` preserves a non-negative local
    perturbation diagnostic but must not be interpreted as promotion benefit.
    """

    layer_index: int
    taylor_benefit: torch.Tensor
    directional_fisher_difference: torch.Tensor
    diagonal_fisher_difference: torch.Tensor
    delta_direction_fisher_magnitude: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.taylor_benefit.shape


@dataclass(frozen=True, slots=True)
class FisherStepResult:
    """Detached result of one teacher-forced sensitivity step."""

    mean_nll: float
    batch_size: int
    trajectory: Literal["fp32", "int4"]
    layers: tuple[RowPromotionSensitivityScores, ...]

    def layer(self, layer_index: int) -> RowPromotionSensitivityScores:
        for scores in self.layers:
            if scores.layer_index == layer_index:
                return scores
        raise KeyError(f"step has no layer {layer_index}")


@dataclass(frozen=True, slots=True)
class TaskMacroSensitivitySummary:
    """Equal-task mean of token-mean promotion-sensitivity scores."""

    layer_index: int
    tasks: int
    steps: int
    mean_nll: float
    trajectory: Literal["fp32", "int4"]
    taylor_benefit: torch.Tensor
    directional_fisher_difference: torch.Tensor
    diagonal_fisher_difference: torch.Tensor
    delta_direction_fisher_magnitude: torch.Tensor


def _validate_specs(int4_spec: QuantizationSpec, int8_spec: QuantizationSpec) -> None:
    if int4_spec.bits != 4:
        raise ValueError("int4_spec must use bits=4")
    if int8_spec.bits != 8:
        raise ValueError("int8_spec must use bits=8")
    if int4_spec.flatten_last_dims != 1 or int8_spec.flatten_last_dims != 1:
        raise ValueError(
            "INT4 and INT8 specs must use flatten_last_dims=1 so each state row "
            "is quantized independently"
        )
    matched_fields = (
        "group_size",
        "scale_bits",
        "flatten_last_dims",
        "rounding",
        "seed",
        "epsilon",
    )
    mismatched = [
        name for name in matched_fields if getattr(int4_spec, name) != getattr(int8_spec, name)
    ]
    if mismatched:
        raise ValueError(
            f"INT4 and INT8 specs may differ only in bit width; mismatched fields: {mismatched}"
        )


def row_promotion_scores_from_errors(
    gradient: torch.Tensor,
    int4_error: torch.Tensor,
    int8_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute signed promotion scores from aligned state errors and a loss gradient.

    Inputs must have shape ``[batch, heads, key_dim, value_dim]``. With
    ``delta = Q8(S) - Q4(S) = int8_error - int4_error``, the primary score is
    ``-sum(gradient * delta)``. This estimates ``L(Q4(S)) - L(Q8(S))`` by a
    first-order Taylor expansion around the trajectory where ``gradient`` was
    measured. Positive values therefore favor INT8 under that local model.

    The two signed curvature diagnostics are
    ``(g dot e4)**2 - (g dot e8)**2`` and
    ``sum(g**2 * (e4**2 - e8**2))``. The final returned tensor is the old
    non-negative ``(g dot (e4-e8))**2`` perturbation magnitude; it is retained
    only as a diagnostic and is not an estimated benefit.
    """

    tensors = {
        "gradient": gradient,
        "int4_error": int4_error,
        "int8_error": int8_error,
    }
    if gradient.ndim != 4:
        raise ValueError("gradient must have shape [batch, heads, key_dim, value_dim]")
    if int4_error.shape != gradient.shape or int8_error.shape != gradient.shape:
        raise ValueError("gradient, int4_error, and int8_error must have the same shape")
    if not all(tensor.is_floating_point() for tensor in tensors.values()):
        raise TypeError("gradient, int4_error, and int8_error must be floating-point")
    if any(tensor.dtype != torch.float32 for tensor in tensors.values()):
        raise TypeError("gradient, int4_error, and int8_error must use torch.float32")
    if len({tensor.device for tensor in tensors.values()}) != 1:
        raise ValueError("gradient, int4_error, and int8_error must be on the same device")
    if gradient.device.type == "meta":
        raise ValueError("gradient and errors must be materialized")
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("gradient and errors must contain only finite values")

    grad = gradient.detach()
    error4 = int4_error.detach()
    error8 = int8_error.detach()
    delta = error8 - error4
    taylor_per_example = -(grad * delta).sum(dim=-1)
    projection4 = (grad * error4).sum(dim=-1)
    projection8 = (grad * error8).sum(dim=-1)
    directional_difference_per_example = projection4.square() - projection8.square()
    diagonal_difference_per_example = (grad.square() * (error4.square() - error8.square())).sum(
        dim=-1
    )
    delta_direction_magnitude_per_example = (grad * (error4 - error8)).sum(dim=-1).square()

    outputs = (
        taylor_per_example.mean(dim=0),
        directional_difference_per_example.mean(dim=0),
        diagonal_difference_per_example.mean(dim=0),
        delta_direction_magnitude_per_example.mean(dim=0),
    )
    if not all(torch.isfinite(value).all().item() for value in outputs):
        raise RuntimeError("promotion sensitivity scores became non-finite")
    return (
        outputs[0].detach().cpu(),
        outputs[1].detach().cpu(),
        outputs[2].detach().cpu(),
        outputs[3].detach().cpu(),
    )


def row_promotion_sensitivity_scores(
    state: torch.Tensor,
    gradient: torch.Tensor,
    *,
    int4_spec: QuantizationSpec,
    int8_spec: QuantizationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an FP32 state and compute its local row-promotion scores.

    ``state`` and ``gradient`` must have shape
    ``[batch, heads, key_dim, value_dim]``. The quantizer group size must equal
    ``value_dim``; this makes each physical quantization group exactly one
    recurrent-state row. Each returned score tensor has shape
    ``[heads, key_dim]``. The gradient is interpreted at ``state``; callers
    measuring gradients on a quantized trajectory should instead call
    :func:`row_promotion_scores_from_errors` with errors relative to the raw
    pre-quantization update.
    """

    _validate_specs(int4_spec, int8_spec)
    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, key_dim, value_dim]")
    if gradient.shape != state.shape:
        raise ValueError("gradient must have the same shape as state")
    if not state.is_floating_point() or not gradient.is_floating_point():
        raise TypeError("state and gradient must use floating-point dtypes")
    if state.device != gradient.device:
        raise ValueError("state and gradient must be on the same device")
    if state.device.type == "meta":
        raise ValueError("state and gradient must be materialized")
    if state.dtype != torch.float32 or gradient.dtype != torch.float32:
        raise TypeError("state and gradient must use torch.float32")
    if int4_spec.group_size != state.shape[-1]:
        raise ValueError(
            "quantizer group_size must equal recurrent-state value_dim so each "
            f"row owns one group ({int4_spec.group_size} != {state.shape[-1]})"
        )
    if not torch.isfinite(state).all().item() or not torch.isfinite(gradient).all().item():
        raise ValueError("state and gradient must contain only finite values")

    source = state.detach()
    int4 = quantize_dequantize(source, int4_spec).tensor.to(torch.float32)
    int8 = quantize_dequantize(source, int8_spec).tensor.to(torch.float32)
    error4 = int4 - source
    error8 = int8 - source
    return row_promotion_scores_from_errors(gradient, error4, error8)


def _cache_state(cache: DynamicCache, layer_index: int) -> torch.Tensor:
    if layer_index >= len(cache.layers):
        raise ValueError(f"cache has no layer {layer_index}")
    layer = cache.layers[layer_index]
    recurrent_states = getattr(layer, "recurrent_states", None)
    initialized = getattr(layer, "is_recurrent_states_initialized", None)
    previous = getattr(layer, "has_previous_state", None)
    if not isinstance(recurrent_states, dict) or not isinstance(initialized, dict):
        raise ValueError(f"cache layer {layer_index} is not a linear-attention cache layer")
    if not initialized.get(0, False) or not isinstance(recurrent_states.get(0), torch.Tensor):
        raise ValueError(
            f"cache layer {layer_index} has no recurrent state; run a no-grad prefill first"
        )
    if not isinstance(previous, dict) or not previous.get(0, False):
        raise ValueError(
            f"cache layer {layer_index} has no previous state; run a no-grad prefill first"
        )
    state = recurrent_states[0]
    if state.dtype != torch.float32:
        raise TypeError(
            f"cache layer {layer_index} recurrent state must use torch.float32, got {state.dtype}"
        )
    if state.ndim != 4:
        raise ValueError(
            f"cache layer {layer_index} recurrent state must have shape "
            "[batch, heads, key_dim, value_dim]"
        )
    if state.device.type == "meta" or not torch.isfinite(state).all().item():
        raise ValueError(
            f"cache layer {layer_index} recurrent state must be materialized and finite"
        )
    return state


def _detach_cache_tensors(cache: DynamicCache) -> None:
    """Detach tensors that a differentiable decode may have stored in the cache."""

    for layer in cache.layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, name, value.detach())
        for name in ("conv_states", "recurrent_states"):
            values = getattr(layer, name, None)
            if isinstance(values, dict):
                for state_index, value in tuple(values.items()):
                    if isinstance(value, torch.Tensor):
                        values[state_index] = value.detach()


_CACHE_SNAPSHOT_ATTRIBUTES = (
    "keys",
    "values",
    "conv_states",
    "recurrent_states",
    "is_conv_states_initialized",
    "is_recurrent_states_initialized",
    "has_previous_state",
    "conv_kernel_size",
    "is_initialized",
    "device",
    "dtype",
    "record_past",
)


def _snapshot_cache(cache: DynamicCache) -> list[dict[str, tuple[bool, object]]]:
    """Capture enough warm DynamicCache state to roll back one failed token."""

    snapshots: list[dict[str, tuple[bool, object]]] = []
    for layer in cache.layers:
        snapshot: dict[str, tuple[bool, object]] = {}
        for name in _CACHE_SNAPSHOT_ATTRIBUTES:
            if not hasattr(layer, name):
                snapshot[name] = (False, None)
                continue
            value = getattr(layer, name)
            if name in ("conv_states", "recurrent_states") and isinstance(value, dict):
                copied = {
                    state_index: state.detach().clone()
                    if isinstance(state, torch.Tensor)
                    else state
                    for state_index, state in value.items()
                }
            elif name in ("keys", "values") and isinstance(value, torch.Tensor):
                # Dynamic attention updates assign a concatenated tensor rather
                # than mutating the old one, so retaining the detached object is
                # enough and avoids copying an arbitrarily long KV history.
                copied = value.detach()
            elif isinstance(value, dict):
                copied = dict(value)
            else:
                copied = value
            snapshot[name] = (True, copied)
        snapshots.append(snapshot)
    return snapshots


def _restore_cache(
    cache: DynamicCache,
    snapshots: Sequence[dict[str, tuple[bool, object]]],
) -> None:
    if len(cache.layers) != len(snapshots):
        raise RuntimeError("cannot roll back a DynamicCache whose layer count changed")
    for layer, snapshot in zip(cache.layers, snapshots, strict=True):
        for name, (existed, value) in snapshot.items():
            if existed:
                setattr(layer, name, value)
            elif hasattr(layer, name):
                delattr(layer, name)


class GDNOneStepFisherCalibrator:
    """Run bounded next-token sensitivity calibration on a warm Qwen3.5 cache.

    The cache must be an ordinary, non-offloaded ``DynamicCache`` populated by
    a no-grad prefill. For each call to :meth:`step`, selected recurrent states
    are detached and reintroduced as FP32 leaves. Model parameters are frozen
    only for that call and their original ``requires_grad`` flags are restored.
    Updated cache tensors are detached in ``finally`` so the next token keeps
    the numerical trajectory without retaining the autograd graph. The
    default ``fp32`` trajectory evaluates quantization errors around the FP32
    state. Use :class:`GDNInt4TrajectoryFisherCalibrator` when the gradient
    itself must be measured on a repeated-QDQ all-INT4 trajectory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache: DynamicCache,
        *,
        layer_indices: Collection[int],
        int4_spec: QuantizationSpec | None = None,
        int8_spec: QuantizationSpec | None = None,
        trajectory: Literal["fp32", "int4"] = "fp32",
    ) -> None:
        if not isinstance(cache, DynamicCache):
            raise TypeError("cache must be an ordinary transformers.DynamicCache")
        if getattr(cache, "offloading", False):
            raise ValueError("offloaded DynamicCache calibration is not supported")
        if trajectory not in ("fp32", "int4"):
            raise ValueError("trajectory must be 'fp32' or 'int4'")
        selected = tuple(sorted(set(layer_indices)))
        if not selected:
            raise ValueError("layer_indices must select at least one GDN layer")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected
        ):
            raise ValueError("layer_indices must contain non-negative integers")

        resolved_int4 = int4_spec or QuantizationSpec(
            bits=4,
            group_size=128,
            flatten_last_dims=1,
        )
        resolved_int8 = int8_spec or QuantizationSpec(
            bits=8,
            group_size=128,
            flatten_last_dims=1,
        )
        _validate_specs(resolved_int4, resolved_int8)

        modules = {
            int(module.layer_idx): module
            for module in model.modules()
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
            and isinstance(getattr(module, "layer_idx", None), int)
        }
        missing = set(selected) - set(modules)
        if missing:
            raise ValueError(
                f"model does not contain selected Qwen3.5 GDN layers {sorted(missing)}"
            )
        expected_state_shapes: dict[int, tuple[int, int, int]] = {}
        for layer_index in selected:
            module = modules[layer_index]
            geometry = (
                getattr(module, "num_v_heads", None),
                getattr(module, "head_k_dim", None),
                getattr(module, "head_v_dim", None),
            )
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in geometry
            ):
                raise ValueError(f"Qwen3.5 GDN layer {layer_index} has invalid state geometry")
            expected_state_shapes[layer_index] = geometry

        self.model = model
        self.cache = cache
        self.layer_indices = selected
        self.int4_spec = resolved_int4
        self.int8_spec = resolved_int8
        self._expected_state_shapes = expected_state_shapes
        self.trajectory = trajectory
        self._int4_error_bases: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer_update_counts = dict.fromkeys(selected, 0)
        self._active = False
        if trajectory == "int4":
            boundary_states: dict[int, torch.Tensor] = {}
            for layer_index in selected:
                state = _cache_state(cache, layer_index)
                self._validate_state_geometry(layer_index, state)
                boundary_states[layer_index] = state
            for layer_index, state in boundary_states.items():
                self._store_int4_boundary(layer_index, state)

    def _validate_state_geometry(self, layer_index: int, state: torch.Tensor) -> None:
        if tuple(state.shape[1:]) != self._expected_state_shapes[layer_index]:
            raise ValueError(
                f"cache layer {layer_index} state shape {tuple(state.shape[1:])} does not "
                f"match Qwen3.5 GDN geometry {self._expected_state_shapes[layer_index]}"
            )
        if self.int4_spec.group_size != state.shape[-1]:
            raise ValueError(
                "quantizer group_size must equal recurrent-state value_dim at "
                f"layer {layer_index} ({self.int4_spec.group_size} != {state.shape[-1]})"
            )

    def _store_int4_boundary(self, layer_index: int, raw_state: torch.Tensor) -> None:
        """QDQ one raw update, store Q4, and retain the aligned promotion errors."""

        update_index = self._layer_update_counts[layer_index]
        int4_spec = scheduled_quantization_spec(
            self.int4_spec,
            layer_index=layer_index,
            layer_update_index=update_index,
        )
        int8_spec = scheduled_quantization_spec(
            self.int8_spec,
            layer_index=layer_index,
            layer_update_index=update_index,
        )
        source = raw_state.detach().to(torch.float32)
        int4 = quantize_dequantize(source, int4_spec).tensor.to(torch.float32).detach()
        int8 = quantize_dequantize(source, int8_spec).tensor.to(torch.float32).detach()
        self.cache.layers[layer_index].recurrent_states[0] = int4
        self._int4_error_bases[layer_index] = (
            (int4 - source).detach(),
            (int8 - source).detach(),
        )
        self._layer_update_counts[layer_index] = update_index + 1

    def step(
        self,
        input_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        *,
        forward_kwargs: Mapping[str, Any] | None = None,
    ) -> FisherStepResult:
        """Score one teacher-forced token and advance the selected trajectory."""

        if self._active:
            raise RuntimeError("calibrator.step is not reentrant")
        if self.model.training:
            raise ValueError("model must be in evaluation mode")
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("input_ids must have shape [batch, 1]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        targets = target_token_ids
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 1 or targets.shape[0] != input_ids.shape[0]:
            raise ValueError("target_token_ids must have shape [batch] or [batch, 1]")
        if targets.dtype not in (torch.int32, torch.int64):
            raise TypeError("target_token_ids must use an integer dtype")
        if targets.device != input_ids.device:
            raise ValueError("input_ids and target_token_ids must be on the same device")

        kwargs = dict(forward_kwargs or {})
        conflicts = {"input_ids", "past_key_values", "use_cache", "labels"} & set(kwargs)
        if conflicts:
            raise ValueError(f"forward_kwargs contains managed arguments: {sorted(conflicts)}")

        states = {index: _cache_state(self.cache, index) for index in self.layer_indices}
        batch_sizes = {state.shape[0] for state in states.values()}
        if batch_sizes != {input_ids.shape[0]}:
            raise ValueError(
                "input batch must match every selected recurrent-state batch; "
                f"input={input_ids.shape[0]}, states={sorted(batch_sizes)}"
            )
        for layer_index, state in states.items():
            self._validate_state_geometry(layer_index, state)
            if state.device != input_ids.device:
                raise ValueError(
                    f"input_ids and cache layer {layer_index} state must be on the same device"
                )
            if self.trajectory == "int4":
                error4, error8 = self._int4_error_bases[layer_index]
                if error4.shape != state.shape or error8.shape != state.shape:
                    raise RuntimeError(
                        f"INT4 trajectory error basis at layer {layer_index} is misaligned"
                    )

        parameters = tuple(self.model.parameters())
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError("model parameters must not have pre-existing gradients")
        original_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        original_update_attribute = self.cache.__dict__.get("update_recurrent_state")
        had_update_attribute = "update_recurrent_state" in self.cache.__dict__
        original_update = self.cache.update_recurrent_state
        cache_snapshot = _snapshot_cache(self.cache)
        error_bases_snapshot = dict(self._int4_error_bases)
        update_counts_snapshot = dict(self._layer_update_counts)
        leaves: dict[int, torch.Tensor] = {}
        raw_updates: dict[int, torch.Tensor] = {}
        step_succeeded = False
        self._active = True

        def differentiable_update(
            cache: DynamicCache,
            recurrent_states: torch.Tensor,
            layer_idx: int,
            state_idx: int = 0,
            **update_kwargs: Any,
        ) -> torch.Tensor:
            if layer_idx not in leaves:
                return original_update(
                    recurrent_states,
                    layer_idx,
                    state_idx=state_idx,
                    **update_kwargs,
                )
            if state_idx != 0:
                raise ValueError("Qwen3.5 Fisher calibration supports recurrent state_idx=0 only")
            layer = cache.layers[layer_idx]
            layer.recurrent_states[state_idx] = recurrent_states
            layer.is_recurrent_states_initialized[state_idx] = True
            raw_updates[layer_idx] = recurrent_states
            return recurrent_states

        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            for layer_index, state in states.items():
                leaf = state.detach().to(torch.float32).requires_grad_(True)
                self.cache.layers[layer_index].recurrent_states[0] = leaf
                leaves[layer_index] = leaf
            self.cache.update_recurrent_state = MethodType(  # type: ignore[method-assign]
                differentiable_update,
                self.cache,
            )

            with torch.enable_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                    **kwargs,
                )
                logits = getattr(outputs, "logits", None)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                    raise TypeError(
                        "model output must expose logits with shape [batch, sequence, vocab]"
                    )
                if logits.shape[:2] != input_ids.shape:
                    raise ValueError("model logits must align with the single-token input")
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("model logits became non-finite")
                per_example_nll = F.cross_entropy(
                    logits[:, -1, :].to(torch.float32),
                    targets.to(torch.long),
                    reduction="none",
                )
                if not torch.isfinite(per_example_nll).all().item():
                    raise RuntimeError("target-token NLL became non-finite")
                gradients = torch.autograd.grad(
                    # Sum keeps each independent batch item's gradient unscaled;
                    # row scores and reported NLL are averaged across the batch.
                    per_example_nll.sum(),
                    tuple(leaves[index] for index in self.layer_indices),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            layer_scores: list[RowPromotionSensitivityScores] = []
            for layer_index, gradient in zip(self.layer_indices, gradients, strict=True):
                if gradient.dtype != torch.float32:
                    gradient = gradient.to(torch.float32)
                if self.trajectory == "fp32":
                    scores = row_promotion_sensitivity_scores(
                        leaves[layer_index],
                        gradient,
                        int4_spec=self.int4_spec,
                        int8_spec=self.int8_spec,
                    )
                else:
                    error4, error8 = self._int4_error_bases[layer_index]
                    scores = row_promotion_scores_from_errors(gradient, error4, error8)
                layer_scores.append(
                    RowPromotionSensitivityScores(
                        layer_index=layer_index,
                        taylor_benefit=scores[0],
                        directional_fisher_difference=scores[1],
                        diagonal_fisher_difference=scores[2],
                        delta_direction_fisher_magnitude=scores[3],
                    )
                )
                updated = self.cache.layers[layer_index].recurrent_states[0]
                if updated is leaves[layer_index]:
                    raise RuntimeError(
                        f"cache layer {layer_index} was not advanced by the model forward"
                    )
                if updated.dtype != torch.float32 or not torch.isfinite(updated).all().item():
                    raise RuntimeError(
                        f"cache layer {layer_index} produced a non-finite or "
                        "non-FP32 recurrent state"
                    )

            if any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("model parameter gradients were populated during calibration")
            result = FisherStepResult(
                mean_nll=float(per_example_nll.detach().mean().item()),
                batch_size=input_ids.shape[0],
                trajectory=self.trajectory,
                layers=tuple(layer_scores),
            )
            step_succeeded = True
            return result
        finally:
            try:
                try:
                    if step_succeeded:
                        try:
                            if self.trajectory == "int4":
                                for layer_index in self.layer_indices:
                                    self._store_int4_boundary(
                                        layer_index,
                                        raw_updates[layer_index],
                                    )
                        except BaseException:
                            _restore_cache(self.cache, cache_snapshot)
                            self._int4_error_bases = error_bases_snapshot
                            self._layer_update_counts = update_counts_snapshot
                            raise
                    else:
                        _restore_cache(self.cache, cache_snapshot)
                        self._int4_error_bases = error_bases_snapshot
                        self._layer_update_counts = update_counts_snapshot
                finally:
                    _detach_cache_tensors(self.cache)
            finally:
                if had_update_attribute:
                    self.cache.__dict__["update_recurrent_state"] = original_update_attribute
                else:
                    self.cache.__dict__.pop("update_recurrent_state", None)
                for parameter, requires_grad in zip(
                    parameters,
                    original_requires_grad,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)
                self._active = False


class GDNInt4TrajectoryFisherCalibrator(GDNOneStepFisherCalibrator):
    """Measure loss gradients on a repeated-QDQ all-INT4 GDN trajectory.

    Construction is the explicit prefill boundary: every raw GDN recurrent
    state in the warm ordinary ``DynamicCache`` is quantized, Q4 is stored for
    the next token, and ``Q8(raw)-Q4(raw)`` is retained as the next promotion
    direction. After each measured token, the raw updated state is captured,
    Q4 is stored again, and the next aligned error basis replaces the old one.
    All Qwen3.5 GDN layers must participate so this mode represents a uniform
    all-INT4 recurrent-state trajectory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache: DynamicCache,
        *,
        int4_spec: QuantizationSpec | None = None,
        int8_spec: QuantizationSpec | None = None,
    ) -> None:
        layer_indices = {
            int(module.layer_idx)
            for module in model.modules()
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
            and isinstance(getattr(module, "layer_idx", None), int)
        }
        if not layer_indices:
            raise ValueError("model has no Qwen3.5 GDN layers")
        super().__init__(
            model,
            cache,
            layer_indices=layer_indices,
            int4_spec=int4_spec,
            int8_spec=int8_spec,
            trajectory="int4",
        )


_SCORE_FIELDS = (
    "taylor_benefit",
    "directional_fisher_difference",
    "diagonal_fisher_difference",
    "delta_direction_fisher_magnitude",
)


@dataclass(slots=True)
class TaskMacroSensitivityAccumulator:
    """Average token scores within each task, then give every task equal weight."""

    _tasks: int = 0
    _steps: int = 0
    _trajectory: Literal["fp32", "int4"] | None = None
    _layer_shapes: dict[int, tuple[int, ...]] | None = None
    _score_sums: dict[str, dict[int, torch.Tensor]] = field(
        default_factory=lambda: {name: {} for name in _SCORE_FIELDS}
    )
    _nll_sum: float = 0.0

    @property
    def tasks(self) -> int:
        return self._tasks

    @property
    def steps(self) -> int:
        return self._steps

    def add_task(self, steps: Sequence[FisherStepResult]) -> None:
        """Consume all token steps for one task without retaining token traces."""

        if not steps:
            raise ValueError("task must contain at least one sensitivity step")
        trajectory = steps[0].trajectory
        if trajectory not in ("fp32", "int4"):
            raise ValueError("step trajectory must be 'fp32' or 'int4'")
        if self._trajectory is not None and trajectory != self._trajectory:
            raise ValueError("trajectory changed across tasks")
        first_layers = {scores.layer_index: scores for scores in steps[0].layers}
        if len(first_layers) != len(steps[0].layers) or not first_layers:
            raise ValueError("each step must contain a non-empty unique layer set")
        shapes = {
            layer_index: tuple(scores.taylor_benefit.shape)
            for layer_index, scores in first_layers.items()
        }
        if self._layer_shapes is not None and shapes != self._layer_shapes:
            raise ValueError("task layer set or score shapes changed")

        task_sums = {
            name: {
                layer_index: torch.zeros(shape, dtype=torch.float64)
                for layer_index, shape in shapes.items()
            }
            for name in _SCORE_FIELDS
        }
        task_nll = 0.0
        for step in steps:
            if step.trajectory != trajectory:
                raise ValueError("trajectory changed within task")
            current = {scores.layer_index: scores for scores in step.layers}
            if len(current) != len(step.layers) or not current:
                raise ValueError("each step must contain a non-empty unique layer set")
            current_shapes = {
                layer_index: tuple(scores.taylor_benefit.shape)
                for layer_index, scores in current.items()
            }
            if current_shapes != shapes:
                raise ValueError("layer set or score shapes changed within task")
            if not torch.isfinite(torch.tensor(step.mean_nll)).item():
                raise ValueError("step mean_nll must be finite")
            task_nll += step.mean_nll
            for layer_index, scores in current.items():
                values = {name: getattr(scores, name) for name in _SCORE_FIELDS}
                if any(tuple(value.shape) != shapes[layer_index] for value in values.values()):
                    raise ValueError("all promotion score shapes must match")
                if any(not value.is_floating_point() for value in values.values()):
                    raise TypeError("promotion scores must be floating-point")
                if any(value.device.type == "meta" for value in values.values()):
                    raise ValueError("promotion scores must be materialized")
                for name, value in values.items():
                    detached = value.detach().cpu().to(torch.float64)
                    if not torch.isfinite(detached).all().item():
                        raise ValueError("promotion scores must be finite")
                    task_sums[name][layer_index] += detached

        divisor = float(len(steps))
        for name in _SCORE_FIELDS:
            for layer_index in shapes:
                task_sums[name][layer_index] /= divisor
                if layer_index not in self._score_sums[name]:
                    self._score_sums[name][layer_index] = torch.zeros_like(
                        task_sums[name][layer_index]
                    )
                self._score_sums[name][layer_index] += task_sums[name][layer_index]
        if self._layer_shapes is None:
            self._layer_shapes = shapes
        if self._trajectory is None:
            self._trajectory = trajectory
        self._nll_sum += task_nll / divisor
        self._tasks += 1
        self._steps += len(steps)

    def summary(self, layer_index: int) -> TaskMacroSensitivitySummary:
        if self._tasks == 0 or layer_index not in self._score_sums["taylor_benefit"]:
            raise ValueError(f"no accumulated sensitivity scores for layer {layer_index}")
        assert self._trajectory is not None
        return TaskMacroSensitivitySummary(
            layer_index=layer_index,
            tasks=self._tasks,
            steps=self._steps,
            mean_nll=self._nll_sum / self._tasks,
            trajectory=self._trajectory,
            taylor_benefit=self._score_sums["taylor_benefit"][layer_index] / self._tasks,
            directional_fisher_difference=(
                self._score_sums["directional_fisher_difference"][layer_index] / self._tasks
            ),
            diagonal_fisher_difference=(
                self._score_sums["diagonal_fisher_difference"][layer_index] / self._tasks
            ),
            delta_direction_fisher_magnitude=(
                self._score_sums["delta_direction_fisher_magnitude"][layer_index] / self._tasks
            ),
        )


# Public names reflect that signed Taylor benefit is the primary output. Keep
# the original experimental class names as aliases for local artifact replay.
SensitivityStepResult = FisherStepResult
GDNOneStepSensitivityCalibrator = GDNOneStepFisherCalibrator
GDNInt4TrajectorySensitivityCalibrator = GDNInt4TrajectoryFisherCalibrator
