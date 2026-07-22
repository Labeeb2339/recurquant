"""Exact-byte row-level INT4/INT8 policy selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, order=True, slots=True)
class RowLocation:
    layer_index: int
    head_index: int
    row_index: int


@dataclass(frozen=True, slots=True)
class ExactBudgetRowPlan:
    """A deterministic row promotion plan with complete storage accounting."""

    low_bits: int
    high_bits: int
    group_size: int
    scale_bits: int
    total_groups: int
    mask_bytes: int
    promotion_increment_bytes: int
    target_resident_bytes: int
    resident_bytes: int
    high_precision_rows: tuple[RowLocation, ...]
    score_shapes: tuple[tuple[int, int, int], ...]

    @property
    def promoted_group_count(self) -> int:
        return len(self.high_precision_rows)

    def groups_for_layer(self, layer_index: int) -> tuple[int, ...]:
        """Return batch-independent flat ``head * rows + row`` group indices."""

        matching_shapes = [shape for shape in self.score_shapes if shape[0] == layer_index]
        if not matching_shapes:
            raise KeyError(f"layer {layer_index} is absent from this row plan")
        _, _, row_count = matching_shapes[0]
        return tuple(
            location.head_index * row_count + location.row_index
            for location in self.high_precision_rows
            if location.layer_index == layer_index
        )

    def evidence_dict(self) -> dict[str, object]:
        evidence = asdict(self)
        evidence["promoted_group_count"] = self.promoted_group_count
        return evidence


def _payload_bytes(*, bits: int, group_size: int) -> int:
    return math.ceil(bits * group_size / 8)


def select_rows_exact_budget(
    scores_by_layer: Mapping[int, torch.Tensor],
    *,
    target_resident_bytes: int,
    low_bits: int = 4,
    high_bits: int = 8,
    group_size: int = 128,
    scale_bits: int = 16,
) -> ExactBudgetRowPlan:
    """Promote the highest-scoring rows under an exact physical-byte budget.

    Every score tensor must have shape ``[heads, rows]``. The storage model uses
    one group per row, one FP scale per group, and one packed precision-mask bit
    per group. A target that cannot be represented exactly is rejected rather
    than silently rounded down.
    """

    if not scores_by_layer:
        raise ValueError("scores_by_layer must not be empty")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in scores_by_layer
    ):
        raise ValueError("layer indices must be nonnegative integers")
    if low_bits < 2 or high_bits <= low_bits:
        raise ValueError("high_bits must exceed low_bits, and low_bits must be at least 2")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if scale_bits not in (16, 32):
        raise ValueError("scale_bits must be 16 or 32")
    if target_resident_bytes <= 0:
        raise ValueError("target_resident_bytes must be positive")

    candidates: list[tuple[float, RowLocation]] = []
    score_shapes: list[tuple[int, int, int]] = []
    for layer_index in sorted(scores_by_layer):
        scores = scores_by_layer[layer_index]
        if scores.ndim != 2 or scores.numel() == 0:
            raise ValueError(f"layer {layer_index} scores must have shape [heads, rows]")
        if not scores.is_floating_point():
            raise TypeError(f"layer {layer_index} scores must use a floating-point dtype")
        if not torch.isfinite(scores).all().item():
            raise ValueError(f"layer {layer_index} scores must be finite")
        head_count, row_count = scores.shape
        score_shapes.append((layer_index, head_count, row_count))
        values = scores.detach().to(device="cpu", dtype=torch.float64)
        for head_index in range(head_count):
            for row_index in range(row_count):
                candidates.append(
                    (
                        float(values[head_index, row_index].item()),
                        RowLocation(layer_index, head_index, row_index),
                    )
                )

    total_groups = len(candidates)
    scale_bytes = scale_bits // 8
    low_group_bytes = _payload_bytes(bits=low_bits, group_size=group_size) + scale_bytes
    promotion_increment = _payload_bytes(bits=high_bits, group_size=group_size) - (
        _payload_bytes(bits=low_bits, group_size=group_size)
    )
    if promotion_increment <= 0:
        raise ValueError(
            "packed low_bits and high_bits must have a positive promotion byte increment; "
            "increase group_size or choose bit widths with distinct payload sizes"
        )
    mask_bytes = math.ceil(total_groups / 8)
    minimum_bytes = total_groups * low_group_bytes + mask_bytes
    maximum_bytes = minimum_bytes + total_groups * promotion_increment
    if not minimum_bytes <= target_resident_bytes <= maximum_bytes:
        raise ValueError(
            "target_resident_bytes is outside the representable range "
            f"[{minimum_bytes}, {maximum_bytes}]"
        )
    extra_bytes = target_resident_bytes - minimum_bytes
    if extra_bytes % promotion_increment:
        lower = minimum_bytes + (extra_bytes // promotion_increment) * promotion_increment
        upper = lower + promotion_increment
        raise ValueError(
            "target_resident_bytes is not exactly representable; nearest valid targets are "
            f"{lower} and {upper}"
        )
    promotion_count = extra_bytes // promotion_increment

    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
    selected = tuple(sorted(location for _, location in ordered[:promotion_count]))
    resident_bytes = minimum_bytes + len(selected) * promotion_increment
    return ExactBudgetRowPlan(
        low_bits=low_bits,
        high_bits=high_bits,
        group_size=group_size,
        scale_bits=scale_bits,
        total_groups=total_groups,
        mask_bytes=mask_bytes,
        promotion_increment_bytes=promotion_increment,
        target_resident_bytes=target_resident_bytes,
        resident_bytes=resident_bytes,
        high_precision_rows=selected,
        score_shapes=tuple(score_shapes),
    )
