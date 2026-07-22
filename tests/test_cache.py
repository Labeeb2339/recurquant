from __future__ import annotations

import torch

from recurquant.cache import (
    iter_recurrent_states,
    quantize_cache_in_place,
    snapshot_recurrent_states,
    state_update_ratios,
)
from recurquant.quantization import QuantizationSpec


class FakeAttentionLayer:
    pass


class FakeLinearLayer:
    def __init__(self, tensor: torch.Tensor, *, initialized: bool = True) -> None:
        self.recurrent_states = {0: tensor}
        self.is_recurrent_states_initialized = {0: initialized}


class FakeCache:
    def __init__(self, layers: list[object]) -> None:
        self.layers = layers


def test_only_initialized_recurrent_states_are_exposed() -> None:
    first = torch.randn(1, 2, 4, 4)
    ignored = torch.randn(1, 2, 4, 4)
    cache = FakeCache(
        [FakeLinearLayer(first), FakeAttentionLayer(), FakeLinearLayer(ignored, initialized=False)]
    )

    states = list(iter_recurrent_states(cache))

    assert len(states) == 1
    assert states[0].layer_index == 0
    assert states[0].state_index == 0
    assert states[0].tensor is first


def test_cache_round_trip_is_in_place_and_reports_layer() -> None:
    state = torch.linspace(-2, 2, 32).reshape(1, 2, 4, 4)
    original_identity = id(state)
    cache = FakeCache([FakeAttentionLayer(), FakeLinearLayer(state)])

    reports = quantize_cache_in_place(cache, QuantizationSpec(bits=4, group_size=8))

    assert id(state) == original_identity
    assert len(reports) == 1
    assert reports[0].layer_index == 1
    assert reports[0].bits == 4
    assert reports[0].estimated_bytes < reports[0].baseline_bytes


def test_state_update_ratio_uses_matching_layer_and_state() -> None:
    state = torch.ones(1, 1, 2, 2)
    cache = FakeCache([FakeLinearLayer(state)])
    before = snapshot_recurrent_states(cache)
    state.mul_(2)

    ratios = state_update_ratios(before, cache)

    assert ratios[(0, 0)] == 1.0
