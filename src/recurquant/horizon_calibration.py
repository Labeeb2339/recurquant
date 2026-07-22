"""Bounded calibration capture for finite-horizon Gated DeltaNet scoring.

The recorder observes selected Qwen3.5 single-token recurrent calls without
changing their arguments or return values. It keeps only normalized q/k,
log-decay, beta, and per-row quantization-error energies for the current task.
Full recurrent-state error matrices are temporary and never retained.

These APIs are experimental calibration primitives. They do not establish an
accuracy, memory, or latency improvement without a separate held-out test.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import torch

from .horizon import HorizonReadRisk, finite_horizon_row_read_risk_from_energies
from .quantization import QuantizationSpec, quantize_dequantize


@dataclass(frozen=True, slots=True)
class GDNCalibrationTrace:
    """One bounded task trace for one Gated DeltaNet layer.

    All tensors are FP32. Vector tensors have shape
    ``[time, batch, heads, key_dim]`` and scalar tensors have shape
    ``[time, batch, heads]``. The q/k vectors are already normalized exactly as
    the tested Qwen3.5 PyTorch kernel normalizes them. At index ``t``, the row
    energies describe the recurrent state immediately before the same token's
    q/k/g/beta transition.
    """

    layer_index: int
    queries: torch.Tensor
    keys: torch.Tensor
    log_decays: torch.Tensor
    betas: torch.Tensor
    int4_row_error_energies: torch.Tensor
    int8_row_error_energies: torch.Tensor
    dropped_calls: int = 0
    missing_initial_state_calls: int = 0

    @property
    def tokens(self) -> int:
        return int(self.queries.shape[0])

    @property
    def retained_bytes(self) -> int:
        tensors = (
            self.queries,
            self.keys,
            self.log_decays,
            self.betas,
            self.int4_row_error_energies,
            self.int8_row_error_energies,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def complete(self) -> bool:
        return self.dropped_calls == 0 and self.missing_initial_state_calls == 0


@dataclass(frozen=True, slots=True)
class BitwidthHorizonReadRisk:
    """Aligned INT4 and INT8 finite-horizon row scores for one layer/task."""

    layer_index: int
    int4: HorizonReadRisk
    int8: HorizonReadRisk

    @property
    def int4_minus_int8(self) -> torch.Tensor:
        """Return the modeled read-risk difference, without clamping its sign."""

        return self.int4.scores - self.int8.scores


@dataclass(frozen=True, slots=True)
class TaskMacroHorizonSummary:
    """Task-macro mean row scores retained by the streaming accumulator."""

    layer_index: int
    horizon: int
    tasks: int
    int4_scores: torch.Tensor
    int8_scores: torch.Tensor

    @property
    def int4_minus_int8(self) -> torch.Tensor:
        return self.int4_scores - self.int8_scores


def _validate_row_spec(spec: QuantizationSpec, *, bits: int, name: str) -> None:
    if spec.bits != bits:
        raise ValueError(f"{name} must use bits={bits}")
    if spec.flatten_last_dims != 1:
        raise ValueError(
            f"{name} must use flatten_last_dims=1 so recurrent-state rows are "
            "quantized independently"
        )


def row_quantization_error_energies(
    initial_state: torch.Tensor,
    spec: QuantizationSpec,
) -> torch.Tensor:
    """Return FP32 squared quantization-error norms for each state row.

    ``initial_state`` must have shape ``[batch, heads, key_dim, value_dim]``.
    The source is explicitly converted to FP32 before quantization so the
    energy measures integer quantization rather than a lower-precision input
    cast. Only the returned ``[batch, heads, key_dim]`` tensor is persistent.
    """

    if initial_state.ndim != 4:
        raise ValueError("initial_state must have shape [batch, heads, key_dim, value_dim]")
    if not initial_state.is_floating_point():
        raise TypeError("initial_state must use a floating-point dtype")
    if initial_state.device.type == "meta":
        raise ValueError("initial_state must be materialized, not on the meta device")
    if not torch.isfinite(initial_state).all().item():
        raise ValueError("initial_state must contain only finite values")
    if spec.flatten_last_dims != 1:
        raise ValueError(
            "spec must use flatten_last_dims=1 so recurrent-state rows are quantized independently"
        )

    source = initial_state.detach().to(torch.float32)
    quantized = quantize_dequantize(source, spec).tensor.to(torch.float32)
    return (quantized - source).square().sum(dim=-1)


def score_gdn_calibration_trace(
    trace: GDNCalibrationTrace,
    *,
    horizon: int = 32,
    epsilon: float = 1e-6,
) -> BitwidthHorizonReadRisk:
    """Score an already-normalized calibration trace at INT4 and INT8."""

    if not trace.complete:
        raise ValueError(
            "cannot score an incomplete calibration trace: "
            f"dropped_calls={trace.dropped_calls}, "
            f"missing_initial_state_calls={trace.missing_initial_state_calls}"
        )

    int4 = finite_horizon_row_read_risk_from_energies(
        trace.int4_row_error_energies,
        trace.queries,
        trace.keys,
        trace.log_decays,
        trace.betas,
        horizon=horizon,
        normalize_qk=False,
        epsilon=epsilon,
    )
    int8 = finite_horizon_row_read_risk_from_energies(
        trace.int8_row_error_energies,
        trace.queries,
        trace.keys,
        trace.log_decays,
        trace.betas,
        horizon=horizon,
        normalize_qk=False,
        epsilon=epsilon,
    )
    return BitwidthHorizonReadRisk(layer_index=trace.layer_index, int4=int4, int8=int8)


@dataclass(slots=True)
class _LayerTraceBuffer:
    queries: list[torch.Tensor] = field(default_factory=list)
    keys: list[torch.Tensor] = field(default_factory=list)
    log_decays: list[torch.Tensor] = field(default_factory=list)
    betas: list[torch.Tensor] = field(default_factory=list)
    int4_energies: list[torch.Tensor] = field(default_factory=list)
    int8_energies: list[torch.Tensor] = field(default_factory=list)
    expected_vector_shape: tuple[int, ...] | None = None
    dropped_calls: int = 0
    missing_initial_state_calls: int = 0

    @property
    def tokens(self) -> int:
        return len(self.queries)


def _argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
    position: int,
) -> object | None:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


def _normalize_qk(value: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    working = value.detach()
    normalized = working * torch.rsqrt(working.square().sum(dim=-1, keepdim=True) + epsilon)
    return normalized.to(torch.float32)


def _validate_single_token_call(
    query: object,
    key: object,
    log_decay: object,
    beta: object,
    initial_state: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    named = {
        "query": query,
        "key": key,
        "g": log_decay,
        "beta": beta,
        "initial_state": initial_state,
    }
    if not all(isinstance(value, torch.Tensor) for value in named.values()):
        missing = [name for name, value in named.items() if not isinstance(value, torch.Tensor)]
        raise TypeError(f"Gated DeltaNet call must expose tensor {', '.join(missing)}")

    query_tensor = query
    key_tensor = key
    decay_tensor = log_decay
    beta_tensor = beta
    state_tensor = initial_state
    if query_tensor.ndim != 4 or key_tensor.ndim != 4:
        raise ValueError("query and key must have shape [batch, 1, heads, key_dim]")
    if query_tensor.shape != key_tensor.shape or query_tensor.shape[1] != 1:
        raise ValueError(
            "recorder accepts only aligned single-token query/key tensors with "
            "shape [batch, 1, heads, key_dim]"
        )
    expected_scalars = query_tensor.shape[:3]
    if decay_tensor.shape != expected_scalars or beta_tensor.shape != expected_scalars:
        raise ValueError("g and beta must have shape [batch, 1, heads]")
    expected_state = (
        query_tensor.shape[0],
        query_tensor.shape[2],
        query_tensor.shape[3],
    )
    if state_tensor.ndim != 4 or state_tensor.shape[:3] != expected_state:
        raise ValueError(
            "initial_state must have shape [batch, heads, key_dim, value_dim] "
            "aligned with query/key"
        )
    tensors = (query_tensor, key_tensor, decay_tensor, beta_tensor, state_tensor)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("all captured Gated DeltaNet inputs must use floating-point dtypes")
    if state_tensor.dtype != torch.float32:
        raise TypeError(
            "initial_state must use torch.float32 so calibration starts from the "
            "full-precision recurrent-state reference"
        )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all captured Gated DeltaNet inputs must be on the same device")
    if query_tensor.device.type == "meta":
        raise ValueError("captured Gated DeltaNet inputs must be materialized")
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("all captured Gated DeltaNet inputs must be finite")
    return query_tensor, key_tensor, decay_tensor, beta_tensor, state_tensor


class GDNHorizonCalibrationRecorder:
    """Capture bounded single-token traces from selected Qwen3.5 GDN layers.

    The caller must name at least one layer and choose a positive per-layer token
    bound. Once a layer reaches that bound, further calls pass through unchanged
    and are counted as dropped. ``drain_traces`` transfers the current task's
    compact traces and clears recorder storage before the next task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_indices: Collection[int],
        max_tokens_per_layer: int,
        int4_spec: QuantizationSpec | None = None,
        int8_spec: QuantizationSpec | None = None,
        storage_device: torch.device | str = "cpu",
        epsilon: float = 1e-6,
    ) -> None:
        selected = tuple(sorted(set(layer_indices)))
        if not selected:
            raise ValueError("layer_indices must select at least one layer")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected
        ):
            raise ValueError("layer_indices must contain non-negative integers")
        if (
            isinstance(max_tokens_per_layer, bool)
            or not isinstance(max_tokens_per_layer, int)
            or max_tokens_per_layer <= 0
        ):
            raise ValueError("max_tokens_per_layer must be a positive integer")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

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
        _validate_row_spec(resolved_int4, bits=4, name="int4_spec")
        _validate_row_spec(resolved_int8, bits=8, name="int8_spec")
        resolved_storage = torch.device(storage_device)
        if resolved_storage.type == "meta":
            raise ValueError("storage_device must retain materialized tensors")

        self.model = model
        self.layer_indices = selected
        self.max_tokens_per_layer = max_tokens_per_layer
        self.int4_spec = resolved_int4
        self.int8_spec = resolved_int8
        self.storage_device = resolved_storage
        self.epsilon = epsilon
        self.enabled = True
        self._buffers = {index: _LayerTraceBuffer() for index in selected}
        self._installed: list[tuple[object, str, object]] = []

    @property
    def captured_tokens(self) -> dict[int, int]:
        return {index: buffer.tokens for index, buffer in self._buffers.items()}

    @property
    def dropped_calls(self) -> dict[int, int]:
        return {index: buffer.dropped_calls for index, buffer in self._buffers.items()}

    @property
    def missing_initial_state_calls(self) -> dict[int, int]:
        return {
            index: buffer.missing_initial_state_calls for index, buffer in self._buffers.items()
        }

    def _selected_modules(self) -> dict[int, torch.nn.Module]:
        selected: dict[int, torch.nn.Module] = {}
        for module in self.model.modules():
            if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
                continue
            layer_index = getattr(module, "layer_idx", None)
            if layer_index not in self._buffers:
                continue
            if layer_index in selected:
                raise ValueError(f"model contains duplicate GDN layer index {layer_index}")
            selected[layer_index] = module
        missing = set(self.layer_indices) - set(selected)
        if missing:
            raise ValueError(f"model does not contain selected GDN layers {sorted(missing)}")
        return selected

    def _store(self, value: torch.Tensor) -> torch.Tensor:
        return (
            value.detach().to(device=self.storage_device, dtype=torch.float32).clone().contiguous()
        )

    def _capture_call(
        self,
        *,
        layer_index: int,
        query: object,
        key: object,
        log_decay: object,
        beta: object,
        initial_state: object,
        use_qk_l2norm_in_kernel: object,
    ) -> None:
        buffer = self._buffers[layer_index]
        if initial_state is None:
            buffer.missing_initial_state_calls += 1
            return
        if buffer.tokens >= self.max_tokens_per_layer:
            buffer.dropped_calls += 1
            return
        if use_qk_l2norm_in_kernel is not True:
            raise ValueError(
                "calibration requires use_qk_l2norm_in_kernel=True to match the "
                "tested Qwen3.5 recurrence"
            )
        query_tensor, key_tensor, decay_tensor, beta_tensor, state_tensor = (
            _validate_single_token_call(query, key, log_decay, beta, initial_state)
        )
        vector_shape = tuple(query_tensor[:, 0].shape)
        if buffer.expected_vector_shape is None:
            buffer.expected_vector_shape = vector_shape
        elif buffer.expected_vector_shape != vector_shape:
            raise ValueError(
                "captured batch/head/key dimensions changed within one task; "
                "drain_traces() before recording the next task"
            )

        normalized_query = _normalize_qk(query_tensor[:, 0], epsilon=self.epsilon)
        normalized_key = _normalize_qk(key_tensor[:, 0], epsilon=self.epsilon)
        state_float = state_tensor.detach().to(torch.float32)
        int4_energy = row_quantization_error_energies(state_float, self.int4_spec)
        int8_energy = row_quantization_error_energies(state_float, self.int8_spec)

        buffer.queries.append(self._store(normalized_query))
        buffer.keys.append(self._store(normalized_key))
        buffer.log_decays.append(self._store(decay_tensor[:, 0]))
        buffer.betas.append(self._store(beta_tensor[:, 0]))
        buffer.int4_energies.append(self._store(int4_energy))
        buffer.int8_energies.append(self._store(int8_energy))

    def _make_wrapper(self, module: torch.nn.Module, original: Any):
        layer_index = int(module.layer_idx)

        def wrapped(*args: object, **kwargs: object):
            if self.enabled:
                self._capture_call(
                    layer_index=layer_index,
                    query=_argument(args, kwargs, "query", 0),
                    key=_argument(args, kwargs, "key", 1),
                    log_decay=_argument(args, kwargs, "g", 3),
                    beta=_argument(args, kwargs, "beta", 4),
                    initial_state=_argument(args, kwargs, "initial_state", 5),
                    use_qk_l2norm_in_kernel=_argument(
                        args,
                        kwargs,
                        "use_qk_l2norm_in_kernel",
                        7,
                    ),
                )
            return original(*args, **kwargs)

        return wrapped

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("recorder is already installed")
        modules = self._selected_modules()
        try:
            for layer_index in self.layer_indices:
                module = modules[layer_index]
                attribute = "recurrent_gated_delta_rule"
                original = getattr(module, attribute)
                setattr(module, attribute, self._make_wrapper(module, original))
                self._installed.append((module, attribute, original))
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        while self._installed:
            module, attribute, original = self._installed.pop()
            setattr(module, attribute, original)

    def clear(self) -> None:
        self._buffers = {index: _LayerTraceBuffer() for index in self.layer_indices}

    def drain_traces(
        self,
        *,
        require_complete: bool = True,
    ) -> dict[int, GDNCalibrationTrace]:
        """Return current task traces and clear storage after validation.

        The default rejects missing or truncated calls instead of silently
        turning a token bound into a shorter calibration example. Set
        ``require_complete=False`` only for recorder diagnostics; incomplete
        traces are marked and cannot be scored.
        """

        if require_complete:
            incomplete = {
                layer_index: {
                    "captured_tokens": buffer.tokens,
                    "dropped_calls": buffer.dropped_calls,
                    "missing_initial_state_calls": buffer.missing_initial_state_calls,
                }
                for layer_index, buffer in self._buffers.items()
                if buffer.tokens == 0 or buffer.dropped_calls or buffer.missing_initial_state_calls
            }
            if incomplete:
                raise RuntimeError(f"calibration task trace is incomplete: {incomplete}")

        traces: dict[int, GDNCalibrationTrace] = {}
        for layer_index, buffer in self._buffers.items():
            if not buffer.queries:
                continue
            traces[layer_index] = GDNCalibrationTrace(
                layer_index=layer_index,
                queries=torch.stack(buffer.queries),
                keys=torch.stack(buffer.keys),
                log_decays=torch.stack(buffer.log_decays),
                betas=torch.stack(buffer.betas),
                int4_row_error_energies=torch.stack(buffer.int4_energies),
                int8_row_error_energies=torch.stack(buffer.int8_energies),
                dropped_calls=buffer.dropped_calls,
                missing_initial_state_calls=buffer.missing_initial_state_calls,
            )
        self.clear()
        return traces

    def __enter__(self) -> GDNHorizonCalibrationRecorder:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.remove()


class TaskMacroHorizonAccumulator:
    """Aggregate task-level layer scores without retaining task traces."""

    def __init__(self, *, horizon: int = 32, epsilon: float = 1e-6) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.horizon = horizon
        self.epsilon = epsilon
        self._int4_sums: dict[int, torch.Tensor] = {}
        self._int8_sums: dict[int, torch.Tensor] = {}
        self._task_counts: dict[int, int] = {}
        self._expected_layer_indices: frozenset[int] | None = None

    def add_task(self, traces: Mapping[int, GDNCalibrationTrace]) -> None:
        """Score one task and retain only one score matrix per bitwidth/layer."""

        if not traces:
            raise ValueError("traces must contain at least one layer")
        layer_indices = frozenset(traces)
        if self._expected_layer_indices is None:
            self._expected_layer_indices = layer_indices
        elif layer_indices != self._expected_layer_indices:
            raise ValueError(
                "trace layer set changed across tasks: "
                f"expected={sorted(self._expected_layer_indices)}, "
                f"actual={sorted(layer_indices)}"
            )
        for layer_index, trace in traces.items():
            if layer_index != trace.layer_index:
                raise ValueError("trace mapping key must match trace.layer_index")
            result = score_gdn_calibration_trace(
                trace,
                horizon=self.horizon,
                epsilon=self.epsilon,
            )
            int4_score = result.int4.scores.detach().to(device="cpu", dtype=torch.float64)
            int8_score = result.int8.scores.detach().to(device="cpu", dtype=torch.float64)
            if layer_index in self._int4_sums:
                if self._int4_sums[layer_index].shape != int4_score.shape:
                    raise ValueError(f"layer {layer_index} score shape changed across tasks")
                self._int4_sums[layer_index] += int4_score
                self._int8_sums[layer_index] += int8_score
            else:
                self._int4_sums[layer_index] = int4_score.clone()
                self._int8_sums[layer_index] = int8_score.clone()
                self._task_counts[layer_index] = 0
            self._task_counts[layer_index] += 1

    def summary(self, layer_index: int) -> TaskMacroHorizonSummary:
        if layer_index not in self._task_counts:
            raise KeyError(f"layer {layer_index} has no accumulated tasks")
        tasks = self._task_counts[layer_index]
        return TaskMacroHorizonSummary(
            layer_index=layer_index,
            horizon=self.horizon,
            tasks=tasks,
            int4_scores=(self._int4_sums[layer_index] / tasks).clone(),
            int8_scores=(self._int8_sums[layer_index] / tasks).clone(),
        )

    def summaries(self) -> dict[int, TaskMacroHorizonSummary]:
        return {layer_index: self.summary(layer_index) for layer_index in self._task_counts}
