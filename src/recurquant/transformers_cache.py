"""Transformers cache adapter that inserts QDQ at the persistent-state boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionLayer

from .quantization import (
    QuantizationSpec,
    quantize_dequantize,
    scheduled_quantization_spec,
)


@dataclass(frozen=True, slots=True)
class CacheUpdateEvidence:
    update_index: int
    layer_index: int
    state_index: int
    shape: tuple[int, ...]
    source_dtype: str
    bits: int
    group_size: int
    rounding: str
    baseline_bytes: int
    estimated_bytes: int
    relative_l2_error: float
    mean_squared_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


class RecurrentStateQDQCache(DynamicCache):
    """QDQ recurrent states before Transformers copies them into its cache.

    The stored cache tensor remains floating point. This class emulates the
    numerical effect of packed storage but does not realize memory savings.
    """

    def __init__(
        self,
        config: object,
        *,
        spec: QuantizationSpec,
        layer_specs: Mapping[int, QuantizationSpec] | None = None,
        enabled_layers: Iterable[int] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.spec = spec
        self.layer_specs = dict(layer_specs or {})
        self.enabled_layers = None if enabled_layers is None else frozenset(enabled_layers)
        self.update_evidence: list[CacheUpdateEvidence] = []
        self._update_index = 0
        self._layer_update_counts: dict[int, int] = {}

        linear_layer_indices = {
            layer_index
            for layer_index, layer in enumerate(self.layers)
            if isinstance(layer, LinearAttentionLayer)
        }
        if any(isinstance(index, bool) or not isinstance(index, int) for index in self.layer_specs):
            raise TypeError("layer_specs keys must be integer model-layer indices")
        invalid_overrides = set(self.layer_specs) - linear_layer_indices
        if invalid_overrides:
            raise ValueError(
                "layer_specs contains non-linear or unknown layer indices: "
                f"{sorted(invalid_overrides)}"
            )
        if self.enabled_layers is not None:
            invalid_enabled = set(self.enabled_layers) - linear_layer_indices
            if invalid_enabled:
                raise ValueError(
                    "enabled_layers contains non-linear or unknown layer indices: "
                    f"{sorted(invalid_enabled)}"
                )

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        enabled = self.enabled_layers is None or layer_idx in self.enabled_layers
        stored = recurrent_states
        if enabled:
            selected_spec = self.layer_specs.get(layer_idx, self.spec)
            layer_update_index = self._layer_update_counts.get(layer_idx, 0)
            selected_spec = scheduled_quantization_spec(
                selected_spec,
                layer_index=layer_idx,
                layer_update_index=layer_update_index,
            )
            result = quantize_dequantize(recurrent_states, selected_spec)
            stored = result.tensor
            self.update_evidence.append(
                CacheUpdateEvidence(
                    update_index=self._update_index,
                    layer_index=layer_idx,
                    state_index=state_idx,
                    shape=tuple(recurrent_states.shape),
                    source_dtype=str(recurrent_states.dtype),
                    bits=selected_spec.bits,
                    group_size=selected_spec.group_size,
                    rounding=selected_spec.rounding,
                    baseline_bytes=result.baseline_bytes,
                    estimated_bytes=result.estimated_bytes,
                    relative_l2_error=result.relative_l2_error,
                    mean_squared_error=result.mean_squared_error,
                    max_absolute_error=result.max_absolute_error,
                )
            )
            self._layer_update_counts[layer_idx] = layer_update_index + 1
        self._update_index += 1
        return super().update_recurrent_state(stored, layer_idx, state_idx, **kwargs)
