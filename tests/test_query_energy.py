from __future__ import annotations

from typing import Any

import pytest
import torch

from recurquant.query_energy import Qwen35QueryEnergyObserver


class _QueryAwareCache:
    def __init__(self, *, has_previous_state: bool = False) -> None:
        self.previous = has_previous_state
        self.pending: dict[int, torch.Tensor] = {}
        self.staged: list[tuple[int, torch.Tensor, float]] = []
        self.updates: list[tuple[int, torch.Tensor]] = []
        self.discards: list[int] = []
        self.fail_stage = False
        self.fail_update = False

    def has_previous_state(self, layer_index: int) -> bool:
        return self.previous

    def stage_query_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        *,
        l2norm_eps: float,
    ) -> None:
        if self.fail_stage:
            raise RuntimeError("stage failed")
        snapshot = query.detach().clone()
        self.pending[layer_index] = snapshot
        self.staged.append((layer_index, snapshot, l2norm_eps))

    def discard_pending_query_observation(self, layer_index: int) -> None:
        self.pending.pop(layer_index, None)
        self.discards.append(layer_index)

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.fail_update:
            raise RuntimeError("update failed")
        if layer_idx not in self.pending:
            raise RuntimeError("query was not staged before recurrent update")
        self.pending.pop(layer_idx)
        self.updates.append((layer_idx, state.detach().clone()))
        self.previous = True
        return state


class _ReferenceCache:
    def __init__(self, *, has_previous_state: bool = False) -> None:
        self.previous = has_previous_state
        self.updates = 0

    def has_previous_state(self, layer_index: int) -> bool:
        return self.previous

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        self.previous = True
        self.updates += 1
        return state


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kernel_calls: list[str] = []
        self.queries: list[torch.Tensor] = []
        self.chunk_gated_delta_rule = self._chunk
        self.recurrent_gated_delta_rule = self._recurrent

    def _chunk(self, query: torch.Tensor, *args: object, **kwargs: object):
        self.kernel_calls.append("chunk")
        self.queries.append(query.detach().clone())
        return query, query.sum(dim=1)

    def _recurrent(self, query: torch.Tensor, *args: object, **kwargs: object):
        self.kernel_calls.append("recurrent")
        self.queries.append(query.detach().clone())
        return query, query.sum(dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del attention_mask
        # Stand in for Qwen3.5's causal convolution, reshape, and key-head repeat.
        query = (hidden_states + 7.0).repeat_interleave(2, dim=2)
        previous = cache_params is not None and cache_params.has_previous_state(  # type: ignore[attr-defined]
            self.layer_idx
        )
        if previous and query.shape[1] == 1:
            output, final_state = self.recurrent_gated_delta_rule(query)
        else:
            output, final_state = self.chunk_gated_delta_rule(
                query,
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(  # type: ignore[attr-defined]
                final_state,
                self.layer_idx,
            )
        return output


class _TinyModel(torch.nn.Module):
    def __init__(self, layers: int = 1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Qwen3_5GatedDeltaNet(index) for index in range(layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        output = hidden_states
        for layer in self.layers:
            output = layer(output, cache_params=cache_params, **kwargs)
        return output


def _input(sequence_length: int) -> torch.Tensor:
    return torch.arange(sequence_length * 2, dtype=torch.float32).reshape(
        1,
        sequence_length,
        1,
        2,
    )


def test_observes_post_convolution_repeated_query_for_all_forward_modes() -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()

    with Qwen35QueryEnergyObserver(model, caches=[cache], l2norm_eps=2e-5):
        model(_input(3), cache_params=cache)  # uncached prefill
        model(_input(2), cache_params=cache)  # cached multi-token chunk
        model(_input(1), cache_params=cache)  # cached one-token decode

    assert model.layers[0].kernel_calls == ["chunk", "chunk", "recurrent"]
    assert len(cache.staged) == 3
    assert len(cache.updates) == 3
    assert not cache.pending
    for (layer_index, staged, eps), kernel_query in zip(
        cache.staged,
        model.layers[0].queries,
        strict=True,
    ):
        assert layer_index == 0
        assert torch.equal(staged, kernel_query)
        assert staged.shape[2] == 2
        assert eps == 2e-5


def test_dispatches_strictly_by_identity_across_two_caches() -> None:
    model = _TinyModel()
    first = _QueryAwareCache()
    second = _QueryAwareCache()

    with Qwen35QueryEnergyObserver(model, caches=[first, second]):
        model(_input(2), cache_params=first)
        model(_input(2), cache_params=second)

    assert len(first.staged) == 1
    assert len(second.staged) == 1
    assert first.staged[0][1].data_ptr() != second.staged[0][1].data_ptr()


def test_reference_cache_and_direct_kernel_calls_pass_through_unchanged() -> None:
    model = _TinyModel()
    observed = _QueryAwareCache()
    reference = _ReferenceCache()
    kernel = model.layers[0].chunk_gated_delta_rule

    with Qwen35QueryEnergyObserver(model, caches=[observed]):
        output = model(_input(2), cache_params=reference)
        direct_output = model.layers[0].chunk_gated_delta_rule(_input(1))

    assert output.shape == (1, 2, 2, 2)
    assert isinstance(direct_output, tuple)
    assert reference.updates == 1
    assert not observed.staged
    assert not observed.discards
    assert model.layers[0].chunk_gated_delta_rule is kernel


def test_restores_forward_and_both_kernels_after_normal_and_exceptional_exit() -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()
    module = model.layers[0]
    originals = (
        module.forward,
        module.chunk_gated_delta_rule,
        module.recurrent_gated_delta_rule,
    )

    with Qwen35QueryEnergyObserver(model, caches=[cache]):
        assert module.forward is not originals[0]
        assert module.chunk_gated_delta_rule is not originals[1]
        assert module.recurrent_gated_delta_rule is not originals[2]

    assert module.forward.__func__ is originals[0].__func__
    assert "forward" not in module.__dict__
    assert module.chunk_gated_delta_rule is originals[1]
    assert module.recurrent_gated_delta_rule is originals[2]

    with (
        pytest.raises(RuntimeError, match="body failed"),
        Qwen35QueryEnergyObserver(model, caches=[cache]),
    ):
        raise RuntimeError("body failed")

    assert module.forward.__func__ is originals[0].__func__
    assert "forward" not in module.__dict__
    assert module.chunk_gated_delta_rule is originals[1]
    assert module.recurrent_gated_delta_rule is originals[2]


@pytest.mark.parametrize("failure", ["stage", "update"])
def test_forward_failure_discards_unconsumed_observation(failure: str) -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()
    if failure == "stage":
        cache.fail_stage = True
    else:
        cache.fail_update = True

    with (
        Qwen35QueryEnergyObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match=f"{failure} failed"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert cache.discards == [0]


def test_packed_cu_seqlens_is_rejected_before_kernel_or_cache_mutation() -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()

    with (
        Qwen35QueryEnergyObserver(model, caches=[cache]),
        pytest.raises(ValueError, match="does not yet support packed cu_seqlens"),
    ):
        model(
            _input(2),
            cache_params=cache,
            cu_seq_lens_q=torch.tensor([0, 2]),
        )

    assert not model.layers[0].kernel_calls
    assert not cache.staged
    assert not cache.updates


def test_duplicate_observers_and_duplicate_cache_entries_are_rejected() -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()

    with pytest.raises(ValueError, match="same object"):
        Qwen35QueryEnergyObserver(model, caches=[cache, cache])

    first = Qwen35QueryEnergyObserver(model, caches=[cache])
    second = Qwen35QueryEnergyObserver(model, caches=[cache])
    with first, pytest.raises(RuntimeError, match="already installed"):
        second.install()


@pytest.mark.parametrize("cache_argument", [None, _ReferenceCache()])
def test_unselected_or_missing_cache_bypasses_observer(cache_argument: object | None) -> None:
    model = _TinyModel()
    observed = _QueryAwareCache()

    with Qwen35QueryEnergyObserver(model, caches=[observed]):
        model(_input(2), cache_params=cache_argument)

    assert not observed.staged
    assert not observed.discards


@pytest.mark.parametrize(
    ("caches", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ([object()], TypeError, "stage_query_observation"),
    ],
)
def test_constructor_fails_closed_on_invalid_cache_contract(
    caches: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        Qwen35QueryEnergyObserver(_TinyModel(), caches=caches)


def test_install_fails_atomically_when_a_kernel_attribute_is_invalid() -> None:
    model = _TinyModel(layers=2)
    cache = _QueryAwareCache()
    first = model.layers[0]
    first_originals = (
        first.forward,
        first.chunk_gated_delta_rule,
        first.recurrent_gated_delta_rule,
    )
    model.layers[1].recurrent_gated_delta_rule = None  # type: ignore[assignment]

    observer = Qwen35QueryEnergyObserver(model, caches=[cache])
    with pytest.raises(TypeError, match="must be callable"):
        observer.install()

    assert first.forward.__func__ is first_originals[0].__func__
    assert "forward" not in first.__dict__
    assert first.chunk_gated_delta_rule is first_originals[1]
    assert first.recurrent_gated_delta_rule is first_originals[2]
    assert not hasattr(model, "_recurquant_query_energy_observer")
    assert all(
        not hasattr(module, "_recurquant_query_energy_observer")
        for module in model.layers
    )


def test_selected_cache_with_non_tensor_kernel_query_fails_closed() -> None:
    model = _TinyModel()
    cache = _QueryAwareCache()
    module = model.layers[0]

    def malformed_forward(
        hidden_states: torch.Tensor,
        cache_params: object | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        module.chunk_gated_delta_rule("not-a-tensor")
        raise AssertionError("unreachable")

    module.forward = malformed_forward  # type: ignore[method-assign]
    module.chunk_gated_delta_rule = lambda query: (query, query)
    with (
        Qwen35QueryEnergyObserver(model, caches=[cache]),
        pytest.raises(TypeError, match="did not expose its query tensor"),
    ):
        module(_input(2), cache_params=cache)

    assert not cache.pending
    assert cache.discards == [0]
