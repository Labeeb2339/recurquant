"""Physically packed persistent-state cache for Transformers linear attention."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import asdict, dataclass
from numbers import Real

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionLayer

from .mixed_quantization import PackedMixedQuantizedTensor, quantize_pack_mixed
from .quantization import (
    PackedQuantizedTensor,
    QuantizationSpec,
    RoundingMode,
    quantize_dequantize,
    quantize_pack,
    scheduled_quantization_spec,
)
from .row_policy import ExactBudgetRowPlan


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


def _reorder_packed_batch(
    packed: PackedQuantizedTensor,
    beam_idx: torch.LongTensor,
) -> PackedQuantizedTensor:
    """Select packed batch entries without another quantization round trip."""

    if len(packed.original_shape) <= packed.spec.flatten_last_dims:
        raise RuntimeError(
            "Cannot reorder a packed state when its quantization groups include the "
            "batch dimension. Use flatten_last_dims smaller than the state rank."
        )

    batch_size = packed.original_shape[0]
    if packed.rows % batch_size:
        raise RuntimeError("packed row metadata is inconsistent with its batch dimension")
    rows_per_batch = packed.rows // batch_size
    selected = beam_idx.to(packed.payload.device)
    selected_batch_size = selected.numel()

    scales = packed.scales.reshape(
        batch_size,
        rows_per_batch,
        packed.groups_per_row,
    )
    reordered_scales = scales.index_select(0, selected).reshape(
        selected_batch_size * rows_per_batch,
        packed.groups_per_row,
    )

    codes_per_batch = rows_per_batch * packed.padded_size
    if packed.spec.bits == 8:
        codes = packed.payload.reshape(batch_size, codes_per_batch)
        reordered_payload = codes.index_select(0, selected).reshape(-1)
    elif packed.spec.bits == 4:
        payload = packed.payload.reshape(-1)
        low = torch.bitwise_and(payload, 0x0F)
        high = torch.bitwise_right_shift(payload, 4)
        nibbles = torch.empty(
            payload.numel() * 2,
            dtype=torch.uint8,
            device=payload.device,
        )
        nibbles[0::2] = low
        nibbles[1::2] = high
        codes = nibbles[: packed.rows * packed.padded_size].reshape(
            batch_size,
            codes_per_batch,
        )
        reordered_codes = codes.index_select(0, selected).reshape(-1)
        if reordered_codes.numel() % 2:
            reordered_codes = torch.cat(
                [
                    reordered_codes,
                    torch.zeros(1, dtype=torch.uint8, device=reordered_codes.device),
                ]
            )
        reordered_payload = torch.bitwise_or(
            reordered_codes[0::2],
            torch.bitwise_left_shift(reordered_codes[1::2], 4),
        )
    else:
        raise RuntimeError(f"Packed reordering does not support INT{packed.spec.bits}")

    return PackedQuantizedTensor(
        payload=reordered_payload.contiguous(),
        scales=reordered_scales.contiguous(),
        spec=packed.spec,
        original_shape=(selected_batch_size, *packed.original_shape[1:]),
        original_dtype=packed.original_dtype,
        flattened_size=packed.flattened_size,
        padded_size=packed.padded_size,
        rows=selected_batch_size * rows_per_batch,
        groups_per_row=packed.groups_per_row,
    )


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
                self.packed_states[state_idx] = _reorder_packed_batch(packed, beam_idx)

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


def _validate_exact_budget_plan(
    plan: ExactBudgetRowPlan,
    linear_layer_indices: set[int],
) -> dict[int, tuple[int, int]]:
    """Validate every byte and geometry invariant needed by the mixed cache."""

    if plan.low_bits != 4 or plan.high_bits != 8:
        raise ValueError("mixed row-plan caches require low_bits=4 and high_bits=8")
    if plan.group_size <= 0:
        raise ValueError("row plan group_size must be positive")
    if plan.scale_bits not in (16, 32):
        raise ValueError("row plan scale_bits must be 16 or 32")

    shapes: dict[int, tuple[int, int]] = {}
    for layer_index, head_count, row_count in plan.score_shapes:
        if layer_index in shapes:
            raise ValueError(f"row plan contains duplicate geometry for layer {layer_index}")
        if layer_index < 0 or head_count <= 0 or row_count <= 0:
            raise ValueError(
                "row plan layer indices must be nonnegative and dimensions must be positive"
            )
        shapes[layer_index] = (head_count, row_count)

    plan_layers = set(shapes)
    missing = sorted(linear_layer_indices - plan_layers)
    extra = sorted(plan_layers - linear_layer_indices)
    if missing or extra:
        raise ValueError(
            "row plan layers must exactly match linear-attention layers; "
            f"missing={missing}, extra={extra}"
        )

    total_groups = sum(heads * rows for heads, rows in shapes.values())
    if plan.total_groups != total_groups:
        raise ValueError(
            "row plan total_groups is inconsistent with score_shapes: "
            f"{plan.total_groups} != {total_groups}"
        )
    expected_mask_bytes = math.ceil(total_groups / 8)
    if plan.mask_bytes != expected_mask_bytes:
        raise ValueError(
            "row plan mask_bytes is inconsistent with total_groups: "
            f"{plan.mask_bytes} != {expected_mask_bytes}"
        )

    selected = set(plan.high_precision_rows)
    if len(selected) != plan.promoted_group_count:
        raise ValueError("row plan contains duplicate high-precision row locations")
    for location in selected:
        geometry = shapes.get(location.layer_index)
        if geometry is None:
            raise ValueError(f"row plan promotion references unknown layer {location.layer_index}")
        head_count, row_count = geometry
        if not (0 <= location.head_index < head_count):
            raise ValueError(f"row plan promotion has out-of-range head index at {location}")
        if not (0 <= location.row_index < row_count):
            raise ValueError(f"row plan promotion has out-of-range row index at {location}")

    low_payload_bytes = math.ceil(plan.low_bits * plan.group_size / 8)
    high_payload_bytes = math.ceil(plan.high_bits * plan.group_size / 8)
    promotion_increment = high_payload_bytes - low_payload_bytes
    if plan.promotion_increment_bytes != promotion_increment:
        raise ValueError(
            "row plan promotion_increment_bytes is inconsistent with its bit widths and "
            f"group size: {plan.promotion_increment_bytes} != {promotion_increment}"
        )
    minimum_bytes = total_groups * (low_payload_bytes + plan.scale_bits // 8) + (
        expected_mask_bytes
    )
    expected_resident = minimum_bytes + plan.promoted_group_count * promotion_increment
    if plan.resident_bytes != expected_resident or plan.target_resident_bytes != expected_resident:
        raise ValueError(
            "row plan resident-byte fields are inconsistent with its physical layout: "
            f"expected {expected_resident}"
        )

    # Each layer owns a separately addressable mask tensor. When a layer is not
    # byte-aligned, independently packing those masks would exceed the selector's
    # globally packed mask model, so reject instead of silently changing the budget.
    per_layer_mask_bytes = sum(math.ceil(heads * rows / 8) for heads, rows in shapes.values())
    if per_layer_mask_bytes != expected_mask_bytes:
        raise ValueError(
            "row plan mask accounting is incompatible with separate per-layer masks; "
            f"plan counts {expected_mask_bytes} bytes but layers require "
            f"{per_layer_mask_bytes}"
        )
    return shapes


class _MixedPackedStateView(MutableMapping[int, torch.Tensor | None]):
    """Compatibility view that materializes one mixed packed state when read."""

    def __init__(self, owner: MixedPackedLinearAttentionLayer) -> None:
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
class MixedPackedCacheUpdateEvidence:
    update_index: int
    layer_index: int
    state_index: int
    low_bits: int
    high_bits: int
    group_size: int
    scale_bits: int
    rounding: str
    source_dtype: str
    shape: tuple[int, ...]
    total_groups: int
    high_precision_groups: int
    selection_method: str
    high_precision_mask_sha256: str
    baseline_bytes: int
    payload_bytes: int
    scale_bytes: int
    mask_bytes: int
    resident_bytes: int
    relative_l2_error: float
    mean_squared_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


MixedUpdateCallback = Callable[
    [int, int, torch.Tensor, PackedMixedQuantizedTensor, torch.Tensor],
    None,
]


class MixedPackedLinearAttentionLayer(LinearAttentionLayer):
    """Linear-attention cache layer with row-selected INT4/INT8 residency."""

    is_compileable = False

    def __init__(
        self,
        *,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
        layer_index: int,
        expected_heads: int,
        expected_rows: int,
        high_precision_group_indices: tuple[int, ...],
        number_of_states: int = 1,
        on_update: MixedUpdateCallback | None = None,
    ) -> None:
        super().__init__(number_of_states=number_of_states)
        if low_spec.bits != 4 or high_spec.bits != 8:
            raise ValueError("mixed packed layers require low INT4 and high INT8 specs")
        if any(
            getattr(low_spec, field) != getattr(high_spec, field)
            for field in (
                "group_size",
                "scale_bits",
                "flatten_last_dims",
                "rounding",
                "seed",
                "epsilon",
            )
        ):
            raise ValueError("mixed packed layer specs must differ only in bits")
        if low_spec.flatten_last_dims != 2:
            raise ValueError("mixed row-plan cache requires flatten_last_dims=2")
        expected_groups = expected_heads * expected_rows
        if len(set(high_precision_group_indices)) != len(high_precision_group_indices):
            raise ValueError("high-precision group indices must be unique")
        if any(index < 0 or index >= expected_groups for index in high_precision_group_indices):
            raise ValueError("high-precision group index is outside the layer geometry")

        self.low_spec = low_spec
        self.high_spec = high_spec
        self.layer_index = layer_index
        self.expected_heads = expected_heads
        self.expected_rows = expected_rows
        self.high_precision_group_indices = tuple(sorted(high_precision_group_indices))
        self.on_update = on_update
        self.packed_states: dict[int, PackedMixedQuantizedTensor | None] = dict.fromkeys(
            range(number_of_states)
        )
        self.recurrent_states = _MixedPackedStateView(self)
        self._update_count = 0

    def _selected_specs(self) -> tuple[QuantizationSpec, QuantizationSpec]:
        return (
            scheduled_quantization_spec(
                self.low_spec,
                layer_index=self.layer_index,
                layer_update_index=self._update_count,
            ),
            scheduled_quantization_spec(
                self.high_spec,
                layer_index=self.layer_index,
                layer_update_index=self._update_count,
            ),
        )

    def _precision_mask(
        self,
        recurrent_states: torch.Tensor,
        *,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
    ) -> torch.Tensor:
        del low_spec, high_spec
        if recurrent_states.ndim != 4:
            raise ValueError(
                f"layer {self.layer_index} recurrent state must have rank 4; "
                f"got shape {tuple(recurrent_states.shape)}"
            )
        expected_shape = (
            recurrent_states.shape[0],
            self.expected_heads,
            self.expected_rows,
            self.low_spec.group_size,
        )
        if tuple(recurrent_states.shape) != expected_shape:
            rendered = (
                f"[batch, {self.expected_heads}, {self.expected_rows}, {self.low_spec.group_size}]"
            )
            raise ValueError(
                f"layer {self.layer_index} recurrent state must have shape {rendered}; "
                f"got {tuple(recurrent_states.shape)}"
            )
        base = torch.zeros(
            self.expected_heads * self.expected_rows,
            dtype=torch.bool,
            device=recurrent_states.device,
        )
        if self.high_precision_group_indices:
            selected = torch.tensor(
                self.high_precision_group_indices,
                dtype=torch.long,
                device=recurrent_states.device,
            )
            base[selected] = True
        return base.repeat(recurrent_states.shape[0]).reshape(
            recurrent_states.shape[0] * self.expected_heads,
            self.expected_rows,
        )

    def _store(self, recurrent_states: torch.Tensor, state_idx: int) -> torch.Tensor:
        if torch.is_grad_enabled() and recurrent_states.requires_grad:
            raise RuntimeError(
                "Mixed packed recurrent states are inference-only and cannot accept an "
                "autograd-tracked tensor. Wrap every prefill and decode forward in "
                "torch.inference_mode() or torch.no_grad(); model.eval() alone does "
                "not disable autograd."
            )
        low_spec, high_spec = self._selected_specs()
        mask = self._precision_mask(
            recurrent_states,
            low_spec=low_spec,
            high_spec=high_spec,
        )
        packed = quantize_pack_mixed(
            recurrent_states,
            mask,
            low_spec=low_spec,
            high_spec=high_spec,
        )
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

    def resident_payload_bytes(self) -> int:
        return sum(
            packed.payload_bytes for packed in self.packed_states.values() if packed is not None
        )

    def resident_scale_bytes(self) -> int:
        return sum(
            packed.scale_bytes for packed in self.packed_states.values() if packed is not None
        )

    def resident_mask_bytes(self) -> int:
        return sum(
            packed.mask_bytes for packed in self.packed_states.values() if packed is not None
        )

    def resident_recurrent_state_bytes(self) -> int:
        return (
            self.resident_payload_bytes()
            + self.resident_scale_bytes()
            + self.resident_mask_bytes()
        )

    def high_precision_group_count(self) -> int:
        return sum(
            packed.high_precision_groups
            for packed in self.packed_states.values()
            if packed is not None
        )

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        return sum(
            packed.elements * torch.empty((), dtype=packed.original_dtype).element_size()
            for packed in self.packed_states.values()
            if packed is not None
        )

    def largest_materialized_recurrent_state_bytes(self) -> int:
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
                packed.low_payload.zero_()
                packed.high_payload.zero_()
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
                self.packed_states[state_idx] = packed.reorder_batch(beam_idx)

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
            if packed is not None and packed.low_payload.device != self.device:
                self.packed_states[state_idx] = packed.to(self.device)


class AdaptiveMixedPackedLinearAttentionLayer(MixedPackedLinearAttentionLayer):
    """Batch-one prototype selecting the fixed INT8 quota on each state update.

    The plan determines only this layer's promotion count. The promoted rows are
    recomputed from the incoming state using the per-row reduction in aligned
    quantize-dequantize MSE from INT4 to INT8. Equal scores keep flattened row
    order, making ties deterministic without changing the byte budget.
    """

    def _precision_mask(
        self,
        recurrent_states: torch.Tensor,
        *,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
    ) -> torch.Tensor:
        # Reuse the static path's complete rank/geometry validation before
        # applying the prototype's deliberately narrow batch-one contract.
        super()._precision_mask(
            recurrent_states,
            low_spec=low_spec,
            high_spec=high_spec,
        )
        if recurrent_states.shape[0] != 1:
            raise ValueError(
                "adaptive mixed cache selection currently requires batch size 1; "
                f"got {recurrent_states.shape[0]}"
            )

        total_groups = self.expected_heads * self.expected_rows
        quota = len(self.high_precision_group_indices)
        mask = torch.zeros(
            total_groups,
            dtype=torch.bool,
            device=recurrent_states.device,
        )
        if quota == 0:
            return mask.reshape(self.expected_heads, self.expected_rows)
        if quota == total_groups:
            return torch.ones_like(mask).reshape(self.expected_heads, self.expected_rows)

        # quantize_dequantize detaches its input. Keep the scope explicit so this
        # selector cannot retain an autograd graph if it is called independently.
        with torch.no_grad():
            source = recurrent_states.detach().to(torch.float32)
            low = quantize_dequantize(recurrent_states, low_spec).tensor.to(torch.float32)
            high = quantize_dequantize(recurrent_states, high_spec).tensor.to(torch.float32)
            low_mse = (low - source).square().mean(dim=-1).reshape(-1)
            high_mse = (high - source).square().mean(dim=-1).reshape(-1)
            benefit = low_mse - high_mse

            # Stable descending sort resolves exact ties by the pre-existing
            # flattened [head, row] order (lower index first).
            ranked = torch.argsort(benefit, descending=True, stable=True)
            mask[ranked[:quota]] = True
        return mask.reshape(self.expected_heads, self.expected_rows)


def _validate_static_rank_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("static_rank_weight must be a real number")
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("static_rank_weight must be finite and in [0, 1]")
    return weight


def _descending_rank_positions(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic zero-best ordinal rank positions."""

    flat = values.reshape(-1)
    order = torch.argsort(flat, descending=True, stable=True)
    ranks = torch.empty(flat.numel(), dtype=torch.int64, device=flat.device)
    ranks[order] = torch.arange(flat.numel(), dtype=torch.int64, device=flat.device)
    return ranks


class RankFusedMixedPackedLinearAttentionLayer(AdaptiveMixedPackedLinearAttentionLayer):
    """Fuse calibrated static ranks with causal per-write MSE-benefit ranks."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.static_rank_weight: float | None = None
        self.static_rank_positions: torch.Tensor | None = None

    def configure_rank_fusion(
        self,
        static_scores: torch.Tensor,
        *,
        static_rank_weight: float,
    ) -> None:
        expected_shape = (self.expected_heads, self.expected_rows)
        if tuple(static_scores.shape) != expected_shape:
            raise ValueError(
                f"layer {self.layer_index} static scores must have shape {expected_shape}; "
                f"got {tuple(static_scores.shape)}"
            )
        if not static_scores.is_floating_point():
            raise TypeError(f"layer {self.layer_index} static scores must be floating point")
        if static_scores.device.type == "meta":
            raise ValueError(f"layer {self.layer_index} static scores must be materialized")
        if not torch.isfinite(static_scores).all().item():
            raise ValueError(f"layer {self.layer_index} static scores must be finite")
        self.static_rank_weight = _validate_static_rank_weight(static_rank_weight)
        with torch.no_grad():
            self.static_rank_positions = _descending_rank_positions(
                static_scores.detach().clone()
            )

    def _precision_mask(
        self,
        recurrent_states: torch.Tensor,
        *,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
    ) -> torch.Tensor:
        # Reuse the complete geometry checks without applying the static mask.
        MixedPackedLinearAttentionLayer._precision_mask(
            self,
            recurrent_states,
            low_spec=low_spec,
            high_spec=high_spec,
        )
        if recurrent_states.shape[0] != 1:
            raise ValueError(
                "rank-fused mixed cache selection currently requires batch size 1; "
                f"got {recurrent_states.shape[0]}"
            )
        if self.static_rank_weight is None or self.static_rank_positions is None:
            raise RuntimeError(f"layer {self.layer_index} rank fusion was not configured")
        if self.static_rank_positions.device != recurrent_states.device:
            raise ValueError(
                f"layer {self.layer_index} static scores and recurrent state must use "
                "the same device"
            )

        total_groups = self.expected_heads * self.expected_rows
        quota = len(self.high_precision_group_indices)
        mask = torch.zeros(
            total_groups,
            dtype=torch.bool,
            device=recurrent_states.device,
        )
        if quota == 0:
            return mask.reshape(self.expected_heads, self.expected_rows)
        if quota == total_groups:
            return torch.ones_like(mask).reshape(self.expected_heads, self.expected_rows)

        with torch.no_grad():
            source = recurrent_states.detach().to(torch.float32)
            low = quantize_dequantize(recurrent_states, low_spec).tensor.to(torch.float32)
            high = quantize_dequantize(recurrent_states, high_spec).tensor.to(torch.float32)
            dynamic_benefit = (
                (low - source).square().mean(dim=-1)
                - (high - source).square().mean(dim=-1)
            ).reshape(-1)
            dynamic_rank = _descending_rank_positions(dynamic_benefit)
            quarter_units = self.static_rank_weight * 4.0
            if quarter_units.is_integer():
                static_units = int(quarter_units)
                fused_cost = (
                    static_units * self.static_rank_positions
                    + (4 - static_units) * dynamic_rank
                )
            else:
                fused_cost = self.static_rank_weight * self.static_rank_positions + (
                    1.0 - self.static_rank_weight
                ) * dynamic_rank
            ranked = torch.argsort(fused_cost, descending=False, stable=True)
            mask[ranked[:quota]] = True
        return mask.reshape(self.expected_heads, self.expected_rows)


class MixedPackedRecurrentStateCache(DynamicCache):
    """Drop-in cache driven by an exact-byte row-level INT4/INT8 plan."""

    _mixed_layer_class = MixedPackedLinearAttentionLayer
    selection_method = "static_plan"

    def __init__(
        self,
        config: object,
        *,
        plan: ExactBudgetRowPlan,
        rounding: RoundingMode = "nearest",
        seed: int = 2339,
        record_evidence: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.plan = plan
        self.record_evidence = record_evidence
        self.update_evidence: list[MixedPackedCacheUpdateEvidence] = []
        self._update_index = 0

        linear_layer_indices = {
            layer_index
            for layer_index, layer in enumerate(self.layers)
            if isinstance(layer, LinearAttentionLayer)
        }
        shapes = _validate_exact_budget_plan(plan, linear_layer_indices)
        low_spec = QuantizationSpec(
            bits=plan.low_bits,
            group_size=plan.group_size,
            scale_bits=plan.scale_bits,
            rounding=rounding,
            seed=seed,
        )
        high_spec = QuantizationSpec(
            bits=plan.high_bits,
            group_size=plan.group_size,
            scale_bits=plan.scale_bits,
            rounding=rounding,
            seed=seed,
        )

        replaced = 0
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, LinearAttentionLayer):
                expected_heads, expected_rows = shapes[layer_index]
                self.layers[layer_index] = self._mixed_layer_class(
                    low_spec=low_spec,
                    high_spec=high_spec,
                    layer_index=layer_index,
                    expected_heads=expected_heads,
                    expected_rows=expected_rows,
                    high_precision_group_indices=plan.groups_for_layer(layer_index),
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
        packed: PackedMixedQuantizedTensor,
        materialized: torch.Tensor,
    ) -> None:
        error = materialized.to(torch.float32) - source.detach().to(torch.float32)
        source_norm = torch.linalg.vector_norm(source.detach().to(torch.float32))
        relative_l2 = torch.linalg.vector_norm(error) / source_norm.clamp_min(1e-12)
        self.update_evidence.append(
            MixedPackedCacheUpdateEvidence(
                update_index=self._update_index,
                layer_index=layer_index,
                state_index=state_index,
                low_bits=packed.low_spec.bits,
                high_bits=packed.high_spec.bits,
                group_size=packed.low_spec.group_size,
                scale_bits=packed.low_spec.scale_bits,
                rounding=packed.low_spec.rounding,
                source_dtype=str(source.dtype),
                shape=tuple(source.shape),
                total_groups=packed.total_groups,
                high_precision_groups=packed.high_precision_groups,
                selection_method=self.selection_method,
                high_precision_mask_sha256=hashlib.sha256(
                    bytes(packed.precision_mask.detach().cpu().contiguous().tolist())
                ).hexdigest(),
                baseline_bytes=source.numel() * source.element_size(),
                payload_bytes=packed.payload_bytes,
                scale_bytes=packed.scale_bytes,
                mask_bytes=packed.mask_bytes,
                resident_bytes=packed.storage_bytes,
                relative_l2_error=float(relative_l2.item()),
                mean_squared_error=float(error.square().mean().item()),
                max_absolute_error=float(error.abs().max().item()),
            )
        )
        self._update_index += 1

    def mixed_packed_layers(self) -> Iterator[tuple[int, MixedPackedLinearAttentionLayer]]:
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, MixedPackedLinearAttentionLayer):
                yield layer_index, layer

    def resident_payload_bytes(self) -> int:
        return sum(layer.resident_payload_bytes() for _, layer in self.mixed_packed_layers())

    def resident_scale_bytes(self) -> int:
        return sum(layer.resident_scale_bytes() for _, layer in self.mixed_packed_layers())

    def resident_mask_bytes(self) -> int:
        return sum(layer.resident_mask_bytes() for _, layer in self.mixed_packed_layers())

    def resident_recurrent_state_bytes(self) -> int:
        return (
            self.resident_payload_bytes()
            + self.resident_scale_bytes()
            + self.resident_mask_bytes()
        )

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        return sum(
            layer.full_precision_equivalent_recurrent_state_bytes()
            for _, layer in self.mixed_packed_layers()
        )

    def largest_materialized_recurrent_state_bytes(self) -> int:
        return max(
            (
                layer.largest_materialized_recurrent_state_bytes()
                for _, layer in self.mixed_packed_layers()
            ),
            default=0,
        )

    def high_precision_group_count(self) -> int:
        return sum(layer.high_precision_group_count() for _, layer in self.mixed_packed_layers())

    def storage_summary(self) -> dict[str, int | float | bool]:
        payload = self.resident_payload_bytes()
        scales = self.resident_scale_bytes()
        mask = self.resident_mask_bytes()
        resident = payload + scales + mask
        full_precision_equivalent = self.full_precision_equivalent_recurrent_state_bytes()
        return {
            "payload_bytes": payload,
            "scale_bytes": scales,
            "mask_bytes": mask,
            "resident_bytes": resident,
            "high_precision_groups": self.high_precision_group_count(),
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


class AdaptiveMixedPackedRecurrentStateCache(MixedPackedRecurrentStateCache):
    """Exact-byte adaptive-row prototype for batch-one inference diagnostics.

    Per-layer promotion quotas come from the supplied exact-budget plan, so the
    resident representation has the same payload, scale, and mask byte counts as
    its static counterpart. Row identities may change at every storage update.
    This class makes no quality, speed, or novelty claim.
    """

    _mixed_layer_class = AdaptiveMixedPackedLinearAttentionLayer
    selection_method = "instantaneous_aligned_mse_reduction"


class RankFusedMixedPackedRecurrentStateCache(MixedPackedRecurrentStateCache):
    """Exact-byte cache fusing static selector and instantaneous MSE ranks."""

    _mixed_layer_class = RankFusedMixedPackedLinearAttentionLayer
    selection_method = "quota_preserving_static_dynamic_rank_fusion"

    def __init__(
        self,
        config: object,
        *,
        plan: ExactBudgetRowPlan,
        static_scores_by_layer: Mapping[int, torch.Tensor],
        static_rank_weight: float,
        rounding: RoundingMode = "nearest",
        seed: int = 2339,
        record_evidence: bool = False,
    ) -> None:
        weight = _validate_static_rank_weight(static_rank_weight)
        if not isinstance(static_scores_by_layer, Mapping) or not static_scores_by_layer:
            raise ValueError("static_scores_by_layer must be a non-empty mapping")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in static_scores_by_layer
        ):
            raise ValueError("static score layer indices must be non-negative integers")

        expected_shapes = {
            layer_index: (head_count, row_count)
            for layer_index, head_count, row_count in plan.score_shapes
        }
        actual_layers = set(static_scores_by_layer)
        expected_layers = set(expected_shapes)
        if actual_layers != expected_layers:
            missing = sorted(expected_layers - actual_layers)
            extra = sorted(actual_layers - expected_layers)
            raise ValueError(
                "static score layers must exactly match the row plan; "
                f"missing={missing}, extra={extra}"
            )

        validated_scores: dict[int, torch.Tensor] = {}
        score_devices: set[torch.device] = set()
        for layer_index in sorted(expected_layers):
            scores = static_scores_by_layer[layer_index]
            if not isinstance(scores, torch.Tensor):
                raise TypeError(f"layer {layer_index} static scores must be a tensor")
            expected_shape = expected_shapes[layer_index]
            if tuple(scores.shape) != expected_shape:
                raise ValueError(
                    f"layer {layer_index} static scores must have shape {expected_shape}; "
                    f"got {tuple(scores.shape)}"
                )
            if not scores.is_floating_point():
                raise TypeError(f"layer {layer_index} static scores must be floating point")
            if scores.device.type == "meta":
                raise ValueError(f"layer {layer_index} static scores must be materialized")
            if not torch.isfinite(scores).all().item():
                raise ValueError(f"layer {layer_index} static scores must be finite")
            validated_scores[layer_index] = scores.detach().clone()
            score_devices.add(scores.device)
        if len(score_devices) != 1:
            raise ValueError("all static score tensors must use the same device")

        self.static_rank_weight = weight
        self.static_scores_by_layer = validated_scores
        super().__init__(
            config,
            plan=plan,
            rounding=rounding,
            seed=seed,
            record_evidence=record_evidence,
        )
        for layer_index, layer in self.mixed_packed_layers():
            if not isinstance(layer, RankFusedMixedPackedLinearAttentionLayer):
                raise RuntimeError(f"layer {layer_index} is not rank-fusion capable")
            layer.configure_rank_fusion(
                validated_scores[layer_index],
                static_rank_weight=weight,
            )
