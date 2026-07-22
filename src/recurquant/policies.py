"""Deterministic precision-plan helpers for diagnostic experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LayerPrecisionPlan:
    default_bits: int
    high_bits: int
    high_precision_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.default_bits < 2:
            raise ValueError("default_bits must be at least 2")
        if self.high_bits <= self.default_bits:
            raise ValueError("high_bits must exceed default_bits")
        if len(set(self.high_precision_layers)) != len(self.high_precision_layers):
            raise ValueError("high_precision_layers must be unique")
        if any(layer < 0 for layer in self.high_precision_layers):
            raise ValueError("layer indices must be nonnegative")

    def average_payload_bits(self, *, layer_count: int) -> float:
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if len(self.high_precision_layers) > layer_count:
            raise ValueError("plan selects more layers than layer_count")
        low_count = layer_count - len(self.high_precision_layers)
        return (
            low_count * self.default_bits + len(self.high_precision_layers) * self.high_bits
        ) / layer_count

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


def select_high_precision_layers(
    scores: Mapping[int, float],
    *,
    count: int,
) -> tuple[int, ...]:
    """Select highest scores, breaking exact ties toward the lower layer index."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(scores):
        raise ValueError("count cannot exceed the number of scores")
    if any(layer < 0 for layer in scores):
        raise ValueError("layer indices must be nonnegative")
    ordered = sorted(scores, key=lambda layer: (-scores[layer], layer))
    return tuple(sorted(ordered[:count]))
