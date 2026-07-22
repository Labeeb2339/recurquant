"""Duck-typed access to Hugging Face linear-attention cache states."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass

import torch

from .quantization import QuantizationResult, QuantizationSpec, quantize_dequantize


@dataclass(frozen=True, slots=True)
class RecurrentStateRef:
    layer_index: int
    state_index: int
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class LayerQuantizationEvidence:
    layer_index: int
    state_index: int
    shape: tuple[int, ...]
    dtype: str
    bits: int
    group_size: int
    baseline_bytes: int
    estimated_bytes: int
    compression_ratio: float
    mean_squared_error: float
    relative_l2_error: float
    max_absolute_error: float

    @classmethod
    def from_result(cls, state: RecurrentStateRef, result: QuantizationResult):
        return cls(
            layer_index=state.layer_index,
            state_index=state.state_index,
            shape=tuple(state.tensor.shape),
            dtype=str(state.tensor.dtype),
            bits=result.spec.bits,
            group_size=result.spec.group_size,
            baseline_bytes=result.baseline_bytes,
            estimated_bytes=result.estimated_bytes,
            compression_ratio=result.compression_ratio,
            mean_squared_error=result.mean_squared_error,
            relative_l2_error=result.relative_l2_error,
            max_absolute_error=result.max_absolute_error,
        )

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


SpecSelector = Callable[[RecurrentStateRef], QuantizationSpec]


def _indexed_values(container: object) -> Iterator[tuple[int, object]]:
    if isinstance(container, Mapping):
        for key, value in container.items():
            yield int(key), value
    elif isinstance(container, (list, tuple)):
        yield from enumerate(container)
    else:
        raise TypeError("recurrent_states must be a mapping, list, or tuple")


def iter_recurrent_states(cache: object) -> Iterator[RecurrentStateRef]:
    """Yield initialized recurrent tensors from a Transformers-style cache."""

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("cache does not expose a layers attribute")

    for layer_index, layer in enumerate(layers):
        states = getattr(layer, "recurrent_states", None)
        if states is None:
            continue
        initialized = getattr(layer, "is_recurrent_states_initialized", None)
        for state_index, tensor in _indexed_values(states):
            if tensor is None:
                continue
            if initialized is not None:
                try:
                    is_initialized = initialized[state_index]
                except (KeyError, IndexError, TypeError):
                    is_initialized = True
                if not is_initialized:
                    continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"layer {layer_index} state {state_index} is not a torch.Tensor"
                )
            yield RecurrentStateRef(layer_index, state_index, tensor)


def snapshot_recurrent_states(cache: object) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (state.layer_index, state.state_index): state.tensor.detach().clone()
        for state in iter_recurrent_states(cache)
    }


def quantize_cache_in_place(
    cache: object,
    spec: QuantizationSpec | SpecSelector,
) -> list[LayerQuantizationEvidence]:
    """Emulate quantized persistent storage by round-tripping each cache state."""

    reports: list[LayerQuantizationEvidence] = []
    with torch.no_grad():
        for state in iter_recurrent_states(cache):
            selected = spec(state) if callable(spec) else spec
            result = quantize_dequantize(state.tensor, selected)
            state.tensor.copy_(result.tensor)
            reports.append(LayerQuantizationEvidence.from_result(state, result))
    return reports


def state_update_ratios(
    before: Mapping[tuple[int, int], torch.Tensor],
    after_cache: object,
    *,
    epsilon: float = 1e-12,
) -> dict[tuple[int, int], float]:
    """Measure ||S_t - S_(t-1)||_2 / max(||S_(t-1)||_2, epsilon)."""

    ratios: dict[tuple[int, int], float] = {}
    for state in iter_recurrent_states(after_cache):
        key = (state.layer_index, state.state_index)
        previous = before.get(key)
        if previous is None:
            continue
        if previous.shape != state.tensor.shape:
            raise ValueError(f"state shape changed for layer/state {key}")
        current = state.tensor.detach().to(torch.float32)
        previous_float = previous.to(device=current.device, dtype=torch.float32)
        numerator = torch.linalg.vector_norm(current - previous_float)
        denominator = torch.linalg.vector_norm(previous_float).clamp_min(epsilon)
        ratios[key] = float((numerator / denominator).item())
    return ratios
