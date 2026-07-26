from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.quantization import QuantizationSpec
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.transformers_cache import RecurrentStateQDQCache
from recurquant.transition_observer import Qwen35TransitionObserver
from tests.test_transformers_cache import tiny_config


class _TransitionCache:
    def __init__(
        self,
        *,
        has_previous_state: bool = False,
        require_query: bool = False,
    ) -> None:
        self.previous = has_previous_state
        self.require_query = require_query
        self.pending: dict[int, tuple[torch.Tensor, ...]] = {}
        self.query_pending: dict[int, torch.Tensor] = {}
        self.staged: list[tuple[int, tuple[torch.Tensor, ...]]] = []
        self.query_staged: list[tuple[int, torch.Tensor]] = []
        self.updates: list[tuple[int, torch.Tensor]] = []
        self.discards: list[int] = []
        self.query_discards: list[int] = []
        self.events: list[str] = []
        self.fail_stage = False
        self.fail_update = False

    def has_previous_state(self, layer_index: int) -> bool:
        del layer_index
        return self.previous

    def stage_transition_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if self.fail_stage:
            raise RuntimeError("stage failed")
        snapshot = tuple(
            tensor.detach().clone() for tensor in (query, key, log_decay, beta)
        )
        self.pending[layer_index] = snapshot
        self.staged.append((layer_index, snapshot))
        self.events.append("transition-stage")

    def discard_pending_transition_observations(self, layer_index: int) -> None:
        self.pending.pop(layer_index, None)
        self.discards.append(layer_index)

    def stage_query_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        *,
        l2norm_eps: float,
    ) -> None:
        assert l2norm_eps > 0
        snapshot = query.detach().clone()
        self.query_pending[layer_index] = snapshot
        self.query_staged.append((layer_index, snapshot))
        self.events.append("query-stage")

    def discard_pending_query_observation(self, layer_index: int) -> None:
        self.query_pending.pop(layer_index, None)
        self.query_discards.append(layer_index)

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        self.events.append("cache-write")
        if self.fail_update:
            raise RuntimeError("update failed")
        if layer_idx not in self.pending:
            raise RuntimeError("transition was not staged before recurrent update")
        if self.require_query and layer_idx not in self.query_pending:
            raise RuntimeError("query was not staged before recurrent update")
        self.pending.pop(layer_idx)
        self.query_pending.pop(layer_idx, None)
        self.updates.append((layer_idx, state.detach().clone()))
        self.previous = True
        return state


class _ReferenceCache:
    def __init__(self) -> None:
        self.previous = False
        self.updates = 0

    def has_previous_state(self, layer_index: int) -> bool:
        del layer_index
        return self.previous

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del layer_idx
        self.previous = True
        self.updates += 1
        return state


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kernel_calls: list[str] = []
        self.kernel_inputs: list[tuple[torch.Tensor, ...]] = []
        self.events: list[str] = []
        self.fail_kernel = False
        self.bypass_kernel_wrapper = False
        self.skip_cache_write = False
        self.call_kernel_twice = False
        self.chunk_gated_delta_rule = self._chunk
        self.recurrent_gated_delta_rule = self._recurrent

    def _kernel(
        self,
        kind: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        self.events.append("kernel-start")
        if self.fail_kernel:
            raise RuntimeError("kernel failed")
        self.kernel_calls.append(kind)
        snapshot = tuple(
            tensor.detach().clone() for tensor in (query, key, g, beta)
        )
        self.kernel_inputs.append(snapshot)
        self.events.append("kernel-done")
        return value, query.sum(dim=1)

    def _chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._kernel("chunk", query, key, value, g=g, beta=beta, **kwargs)

    def _recurrent(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._kernel(
            "recurrent",
            query,
            key,
            value,
            g=g,
            beta=beta,
            **kwargs,
        )

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
        key = (hidden_states - 3.0).repeat_interleave(2, dim=2)
        value = hidden_states.repeat_interleave(2, dim=2)
        log_decay = -torch.arange(
            1,
            query.shape[1] + 1,
            dtype=query.dtype,
            device=query.device,
        ).reshape(1, -1, 1).expand(query.shape[0], -1, query.shape[2])
        beta = torch.linspace(
            0.1,
            0.9,
            query.shape[1],
            dtype=query.dtype,
            device=query.device,
        ).reshape(1, -1, 1).expand(query.shape[0], -1, query.shape[2])
        previous = cache_params is not None and cache_params.has_previous_state(  # type: ignore[attr-defined]
            self.layer_idx
        )
        kernel = (
            self.recurrent_gated_delta_rule
            if previous and query.shape[1] == 1
            else self.chunk_gated_delta_rule
        )
        if self.bypass_kernel_wrapper:
            kernel = self._recurrent if previous and query.shape[1] == 1 else self._chunk
        output, final_state = kernel(
            query,
            key,
            value,
            g=log_decay,
            beta=beta,
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )
        if self.call_kernel_twice:
            kernel(query, key, value, g=log_decay, beta=beta)
        if cache_params is not None and not self.skip_cache_write:
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


def test_observes_exact_transition_inputs_for_prefill_chunk_and_decode() -> None:
    model = _TinyModel()
    cache = _TransitionCache()
    cache.events = model.layers[0].events

    with Qwen35TransitionObserver(model, caches=[cache]):
        model(_input(3), cache_params=cache)
        model(_input(2), cache_params=cache)
        model(_input(1), cache_params=cache)

    assert model.layers[0].kernel_calls == ["chunk", "chunk", "recurrent"]
    assert len(cache.staged) == 3
    assert len(cache.updates) == 3
    assert not cache.pending
    for (layer_index, staged), kernel_inputs in zip(
        cache.staged,
        model.layers[0].kernel_inputs,
        strict=True,
    ):
        assert layer_index == 0
        assert len(staged) == 4
        for observed, expected in zip(staged, kernel_inputs, strict=True):
            assert torch.equal(observed, expected)
        query, key, log_decay, beta = staged
        assert query.shape == key.shape
        assert query.shape[2] == 2
        assert log_decay.shape == beta.shape == query.shape[:3]

    assert cache.events == [
        "kernel-start",
        "kernel-done",
        "transition-stage",
        "cache-write",
        "kernel-start",
        "kernel-done",
        "transition-stage",
        "cache-write",
        "kernel-start",
        "kernel-done",
        "transition-stage",
        "cache-write",
    ]


def test_dispatches_strictly_by_identity_and_leaves_reference_cache_alone() -> None:
    model = _TinyModel()
    first = _TransitionCache()
    second = _TransitionCache()
    reference = _ReferenceCache()

    with Qwen35TransitionObserver(model, caches=[first, second]):
        model(_input(2), cache_params=first)
        model(_input(2), cache_params=second)
        model(_input(2), cache_params=reference)

    assert len(first.staged) == len(second.staged) == 1
    assert first.staged[0][1][0].data_ptr() != second.staged[0][1][0].data_ptr()
    assert reference.updates == 1


@pytest.mark.parametrize("failure", ["kernel", "stage", "update"])
def test_every_forward_failure_clears_pending_observation(failure: str) -> None:
    model = _TinyModel()
    cache = _TransitionCache()
    module = model.layers[0]
    original_forward = module.forward
    original_chunk = module.chunk_gated_delta_rule
    original_recurrent = module.recurrent_gated_delta_rule
    if failure == "kernel":
        model.layers[0].fail_kernel = True
    elif failure == "stage":
        cache.fail_stage = True
    else:
        cache.fail_update = True

    with (
        Qwen35TransitionObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match=f"{failure} failed"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert cache.discards == [0]
    assert module.forward.__func__ is original_forward.__func__
    assert "forward" not in module.__dict__
    assert module.chunk_gated_delta_rule is original_chunk
    assert module.recurrent_gated_delta_rule is original_recurrent


@pytest.mark.parametrize("packed_name", ["cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens"])
def test_rejects_packed_cu_seqlens_before_kernel_or_cache_mutation(
    packed_name: str,
) -> None:
    model = _TinyModel()
    cache = _TransitionCache()

    with (
        Qwen35TransitionObserver(model, caches=[cache]),
        pytest.raises(ValueError, match="does not support packed cu_seqlens"),
    ):
        model(
            _input(2),
            cache_params=cache,
            **{packed_name: torch.tensor([0, 2])},
        )

    assert not model.layers[0].kernel_calls
    assert not cache.staged
    assert not cache.updates


def test_fails_closed_if_forward_bypasses_wrapped_kernel() -> None:
    model = _TinyModel()
    cache = _TransitionCache()
    model.layers[0].bypass_kernel_wrapper = True
    model.layers[0].skip_cache_write = True

    with (
        Qwen35TransitionObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="exactly one observed state-kernel call"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert not cache.staged
    assert cache.discards == [0]


def test_rejects_a_second_kernel_call_without_staging_twice() -> None:
    model = _TinyModel()
    cache = _TransitionCache()
    model.layers[0].call_kernel_twice = True

    with (
        Qwen35TransitionObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="more than one state kernel"),
    ):
        model(_input(2), cache_params=cache)

    assert len(cache.staged) == 1
    assert not cache.pending
    assert cache.discards == [0]


@pytest.mark.parametrize("transition_outermost", [False, True])
def test_nests_with_query_energy_observer_in_either_order(
    transition_outermost: bool,
) -> None:
    model = _TinyModel()
    cache = _TransitionCache(require_query=True)
    module = model.layers[0]
    original_forward = module.forward
    original_chunk = module.chunk_gated_delta_rule
    original_recurrent = module.recurrent_gated_delta_rule
    transition = Qwen35TransitionObserver(model, caches=[cache])
    query = Qwen35QueryEnergyObserver(model, caches=[cache])
    observers = (query, transition) if transition_outermost else (transition, query)

    with ExitStack() as stack:
        for observer in observers:
            stack.enter_context(observer)
        model(_input(2), cache_params=cache)

    assert len(cache.staged) == len(cache.query_staged) == len(cache.updates) == 1
    assert not cache.pending
    assert not cache.query_pending
    assert module.forward.__func__ is original_forward.__func__
    assert "forward" not in module.__dict__
    assert module.chunk_gated_delta_rule is original_chunk
    assert module.recurrent_gated_delta_rule is original_recurrent


def test_nested_observers_both_discard_when_cache_write_fails() -> None:
    model = _TinyModel()
    cache = _TransitionCache(require_query=True)
    cache.fail_update = True

    with (
        Qwen35TransitionObserver(model, caches=[cache]),
        Qwen35QueryEnergyObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="update failed"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert not cache.query_pending
    assert cache.discards == [0]
    assert cache.query_discards == [0]


def test_installation_is_atomic_and_duplicate_installation_is_rejected() -> None:
    model = _TinyModel(layers=2)
    cache = _TransitionCache()
    first = model.layers[0]
    originals = (
        first.forward,
        first.chunk_gated_delta_rule,
        first.recurrent_gated_delta_rule,
    )
    model.layers[1].recurrent_gated_delta_rule = None  # type: ignore[assignment]

    observer = Qwen35TransitionObserver(model, caches=[cache])
    with pytest.raises(TypeError, match="must be callable"):
        observer.install()

    assert first.forward.__func__ is originals[0].__func__
    assert "forward" not in first.__dict__
    assert first.chunk_gated_delta_rule is originals[1]
    assert first.recurrent_gated_delta_rule is originals[2]
    assert not hasattr(model, "_recurquant_transition_observer")

    valid_model = _TinyModel()
    first_observer = Qwen35TransitionObserver(valid_model, caches=[cache])
    second_observer = Qwen35TransitionObserver(valid_model, caches=[cache])
    with first_observer, pytest.raises(RuntimeError, match="already installed"):
        second_observer.install()


@pytest.mark.parametrize(
    ("caches", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ([object()], TypeError, "stage_transition_observation"),
    ],
)
def test_constructor_rejects_invalid_cache_contract(
    caches: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        Qwen35TransitionObserver(_TinyModel(), caches=caches)


def test_constructor_rejects_the_same_cache_identity_twice() -> None:
    cache = _TransitionCache()

    with pytest.raises(ValueError, match="same object"):
        Qwen35TransitionObserver(_TinyModel(), caches=[cache, cache])


class _ObservedQDQCache(RecurrentStateQDQCache):
    def __init__(self, config: Any) -> None:
        super().__init__(config, spec=QuantizationSpec(bits=4, group_size=8))
        self.transition_observations: list[tuple[int, tuple[torch.Tensor, ...]]] = []
        self.pending_transitions: dict[int, tuple[torch.Tensor, ...]] = {}

    def stage_transition_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        snapshot = tuple(
            tensor.detach().clone() for tensor in (query, key, log_decay, beta)
        )
        self.pending_transitions[layer_index] = snapshot
        self.transition_observations.append((layer_index, snapshot))

    def discard_pending_transition_observations(self, layer_index: int) -> None:
        self.pending_transitions.pop(layer_index, None)

    def update_recurrent_state(
        self,
        recurrent_state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if layer_idx not in self.pending_transitions:
            raise RuntimeError("transition was not staged before recurrent update")
        self.pending_transitions.pop(layer_idx)
        return super().update_recurrent_state(recurrent_state, layer_idx)


def test_real_tiny_qwen_runs_prefill_cached_chunk_and_decode() -> None:
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).to(device).eval()
    cache = _ObservedQDQCache(config)

    with torch.inference_mode(), Qwen35TransitionObserver(model, caches=[cache]):
        prefill = model(
            torch.randint(0, config.vocab_size, (1, 5), device=device),
            past_key_values=cache,
        )
        chunk = model(
            torch.randint(0, config.vocab_size, (1, 2), device=device),
            past_key_values=cache,
        )
        decode = model(
            torch.randint(0, config.vocab_size, (1, 1), device=device),
            past_key_values=cache,
        )

    assert prefill.logits.shape == (1, 5, config.vocab_size)
    assert chunk.logits.shape == (1, 2, config.vocab_size)
    assert decode.logits.shape == (1, 1, config.vocab_size)
    assert len(cache.transition_observations) == 3
    assert not cache.pending_transitions
    assert [item[1][0].shape[1] for item in cache.transition_observations] == [5, 2, 1]
    for layer_index, (query, key, log_decay, beta) in cache.transition_observations:
        assert layer_index == 0
        assert query.shape == key.shape == (
            1,
            query.shape[1],
            config.linear_num_value_heads,
            config.linear_key_head_dim,
        )
        assert log_decay.shape == beta.shape == query.shape[:3]
        assert query.device == key.device == log_decay.device == beta.device == device
