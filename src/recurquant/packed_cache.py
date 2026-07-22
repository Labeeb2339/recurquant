"""Physically packed persistent-state cache for Transformers linear attention."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import asdict, dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionLayer

from .quantization import (
    PackedQuantizedTensor,
    QuantizationSpec,
    quantize_pack,
    scheduled_quantization_spec,
)


class _PackedStateView(MutableMapping[int, torch.Tensor | None]):
    """Compatibility view that materializes a packed state only when read."""

    def __init__(self, owner: PackedLinearAttentionLayer) -> None:
        self._owner = owner

    def __getitem__(self, key: int) -> torch.Tensor | None:
        packed = self._owner.packed_states[key]
        return None if packed is None else packed.dequantize()

    def __setitem__(self, key: int, value: torch.Tensor | None) -> None:
        if value is None:
            self._owner.packed_states[key] = None
            self._owner.is_recurrent_states_initialized[key] = False
        else:
            self._owner._store(value, key)

    def __delitem__(self, key: int) -> None:
        self.__setitem__(key, None)

    def __iter__(self) -> Iterator[int]:
        return iter(self._owner.packed_states)

    def __len__(self) -> int:
        return len(self._owner.packed_states)


@dataclass(frozen=True, slots=True)
class PackedCacheUpdateEvidence:
    update_index: int
    layer_index: int
    state_index: int
    bits: int
    group_size: int
    scale_bits: int
    rounding: str
    source_dtype: str
    shape: tuple[int, ...]
    baseline_bytes: int
    estimated_bytes: int
    resident_bytes: int
    relative_l2_error: float
    mean_squared_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


UpdateCallback = Callable[
    [int, int, torch.Tensor, PackedQuantizedTensor, torch.Tensor],
    None,
]


class PackedLinearAttentionLayer(LinearAttentionLayer):
    """Linear-attention cache layer with integer recurrent-state residency."""

    is_compileable = False

    def __init__(
        self,
        *,
        spec: QuantizationSpec,
        layer_index: int,
        number_of_states: int = 1,
        on_update: UpdateCallback | None = None,
    ) -> None:
        super().__init__(number_of_states=number_of_states)
        if spec.bits not in (4, 8):
            raise ValueError("packed cache layers currently support only INT4 and INT8")
        self.spec = spec
        self.layer_index = layer_index
        self.on_update = on_update
        self.packed_states: dict[int, PackedQuantizedTensor | None] = dict.fromkeys(
            range(number_of_states)
        )
        self.recurrent_states = _PackedStateView(self)
        self._update_count = 0

    def _selected_spec(self) -> QuantizationSpec:
        return scheduled_quantization_spec(
            self.spec,
            layer_index=self.layer_index,
            layer_update_index=self._update_count,
        )

    def _store(self, recurrent_states: torch.Tensor, state_idx: int) -> torch.Tensor:
        if torch.is_grad_enabled() and recurrent_states.requires_grad:
            raise RuntimeError(
                "Packed recurrent states are inference-only and cannot accept an "
                "autograd-tracked tensor. Wrap every prefill and decode forward in "
                "torch.inference_mode() or torch.no_grad(); model.eval() alone does "
                "not disable autograd."
            )
        selected_spec = self._selected_spec()
        packed = quantize_pack(recurrent_states, selected_spec)
        self.packed_states[state_idx] = packed
        self.is_recurrent_states_initialized[state_idx] = True
        if self.device is None:
            self.dtype = recurrent_states.dtype
            self.device = recurrent_states.device
        materialized = packed.dequantize()
        if self.on_update is not None:
            self.on_update(
                self.layer_index,
                state_idx,
                recurrent_states,
                packed,
                materialized,
            )
        self._update_count += 1
        return materialized

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        return self._store(recurrent_states, state_idx)

    def resident_recurrent_state_bytes(self) -> int:
        return sum(
            packed.storage_bytes for packed in self.packed_states.values() if packed is not None
        )

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        return sum(
            packed.elements * torch.empty((), dtype=packed.original_dtype).element_size()
            for packed in self.packed_states.values()
            if packed is not None
        )

    def largest_materialized_recurrent_state_bytes(self) -> int:
        """Return bytes in the largest single dequantized recurrent state.

        This is an exact tensor-size count, not a device allocator peak or a
        measurement of the layer's total temporary workspace.
        """

        return max(
            (
                packed.elements * torch.empty((), dtype=packed.original_dtype).element_size()
                for packed in self.packed_states.values()
                if packed is not None
            ),
            default=0,
        )

    def reset(self) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx].zero_()
            packed = self.packed_states[state_idx]
            if packed is not None:
                packed.payload.zero_()
                packed.scales.zero_()
            self.has_previous_state[state_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx] = self.conv_states[state_idx].index_select(
                    0, beam_idx.to(self.device)
                )
            packed = self.packed_states[state_idx]
            if packed is not None:
                reordered = packed.dequantize().index_select(0, beam_idx.to(packed.payload.device))
                self.packed_states[state_idx] = quantize_pack(reordered, packed.spec)

    def offload(self) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx] = self.conv_states[state_idx].to(
                    "cpu", non_blocking=True
                )
            packed = self.packed_states[state_idx]
            if packed is not None:
                self.packed_states[state_idx] = packed.to("cpu")

    def prefetch(self) -> None:
        for state_idx in range(self.number_of_states):
            if (
                self.is_conv_states_initialized[state_idx]
                and self.conv_states[state_idx].device != self.device
            ):
                self.conv_states[state_idx] = self.conv_states[state_idx].to(
                    self.device, non_blocking=True
                )
            packed = self.packed_states[state_idx]
            if packed is not None and packed.payload.device != self.device:
                self.packed_states[state_idx] = packed.to(self.device)


class PackedRecurrentStateCache(DynamicCache):
    """Drop-in Qwen3.5 cache that keeps Gated DeltaNet states physically packed.

    Whether physical packing reduces resident bytes depends on the state shape and
    quantization spec; inspect ``storage_summary()`` after a cache update. The
    current PyTorch implementation materializes one recurrent state while its layer
    executes. It makes no speed or whole-model peak-memory claim and is not
    compatible with ``torch.compile``.
    """

    def __init__(
        self,
        config: object,
        *,
        spec: QuantizationSpec,
        layer_specs: Mapping[int, QuantizationSpec] | None = None,
        record_evidence: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.spec = spec
        self.layer_specs = dict(layer_specs or {})
        self.record_evidence = record_evidence
        self.update_evidence: list[PackedCacheUpdateEvidence] = []
        self._update_index = 0

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

        replaced = 0
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, LinearAttentionLayer):
                selected_spec = self.layer_specs.get(layer_index, self.spec)
                self.layers[layer_index] = PackedLinearAttentionLayer(
                    spec=selected_spec,
                    layer_index=layer_index,
                    number_of_states=layer.number_of_states,
                    on_update=self._record_update if record_evidence else None,
                )
                replaced += 1
        if replaced == 0:
            raise TypeError("config did not create any Transformers linear-attention layers")

    def _record_update(
        self,
        layer_index: int,
        state_index: int,
        source: torch.Tensor,
        packed: PackedQuantizedTensor,
        materialized: torch.Tensor,
    ) -> None:
        error = materialized.to(torch.float32) - source.detach().to(torch.float32)
        source_norm = torch.linalg.vector_norm(source.detach().to(torch.float32))
        relative_l2 = torch.linalg.vector_norm(error) / source_norm.clamp_min(1e-12)
        self.update_evidence.append(
            PackedCacheUpdateEvidence(
                update_index=self._update_index,
                layer_index=layer_index,
                state_index=state_index,
                bits=packed.spec.bits,
                group_size=packed.spec.group_size,
                scale_bits=packed.spec.scale_bits,
                rounding=packed.spec.rounding,
                source_dtype=str(source.dtype),
                shape=tuple(source.shape),
                baseline_bytes=source.numel() * source.element_size(),
                estimated_bytes=packed.storage_bytes,
                resident_bytes=packed.storage_bytes,
                relative_l2_error=float(relative_l2.item()),
                mean_squared_error=float(error.square().mean().item()),
                max_absolute_error=float(error.abs().max().item()),
            )
        )
        self._update_index += 1

    def packed_layers(self) -> Iterator[tuple[int, PackedLinearAttentionLayer]]:
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, PackedLinearAttentionLayer):
                yield layer_index, layer

    def resident_recurrent_state_bytes(self) -> int:
        return sum(layer.resident_recurrent_state_bytes() for _, layer in self.packed_layers())

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        return sum(
            layer.full_precision_equivalent_recurrent_state_bytes()
            for _, layer in self.packed_layers()
        )

    def largest_materialized_recurrent_state_bytes(self) -> int:
        """Return bytes in the largest single dequantized recurrent state.

        This does not measure allocator peak memory or the full layer workspace.
        """

        return max(
            (
                layer.largest_materialized_recurrent_state_bytes()
                for _, layer in self.packed_layers()
            ),
            default=0,
        )

    def storage_summary(self) -> dict[str, int | float | bool]:
        resident = self.resident_recurrent_state_bytes()
        full_precision_equivalent = self.full_precision_equivalent_recurrent_state_bytes()
        return {
            "resident_bytes": resident,
            "full_precision_equivalent_bytes": full_precision_equivalent,
            "largest_materialized_state_bytes": (
                self.largest_materialized_recurrent_state_bytes()
            ),
            "resident_compression_ratio": (
                full_precision_equivalent / resident if resident else 0.0
            ),
            "physical_reduction_realized": (
                resident > 0 and resident < full_precision_equivalent
            ),
        }
