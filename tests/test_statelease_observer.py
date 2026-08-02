from __future__ import annotations

from contextlib import ExitStack
from itertools import permutations
from typing import Any

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.quantization import QuantizationSpec
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.statelease_observer import Qwen35StateLeaseObserver
from recurquant.transformers_cache import RecurrentStateQDQCache
from recurquant.transition_observer import Qwen35TransitionObserver
from tests.test_transformers_cache import tiny_config

_StateLeaseObservation = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]


class _StateLeaseCache:
    def __init__(
        self,
        *,
        initial_state: torch.Tensor | None = None,
        require_companion_observers: bool = False,
    ) -> None:
        self.current_state = initial_state
        self.require_companion_observers = require_companion_observers
        self.pending: dict[int, _StateLeaseObservation] = {}
        self.query_pending: dict[int, torch.Tensor] = {}
        self.transition_pending: dict[int, tuple[torch.Tensor, ...]] = {}
        self.staged: list[tuple[int, _StateLeaseObservation]] = []
        self.query_staged: list[tuple[int, torch.Tensor]] = []
        self.transition_staged: list[tuple[int, tuple[torch.Tensor, ...]]] = []
        self.updates: list[tuple[int, torch.Tensor]] = []
        self.discards: list[int] = []
        self.query_discards: list[int] = []
        self.transition_discards: list[int] = []
        self.events: list[str] = []
        self.fail_stage = False
        self.fail_update = False
        self._active_transaction: object | None = None

    def begin_statelease_forward_transaction(self) -> object:
        if self._active_transaction is not None:
            raise RuntimeError("transaction already active")
        transaction = (
            self.current_state,
            dict(self.pending),
            dict(self.query_pending),
            dict(self.transition_pending),
            len(self.updates),
        )
        self._active_transaction = transaction
        return transaction

    def commit_statelease_forward_transaction(self, transaction: object) -> None:
        if transaction is not self._active_transaction:
            raise RuntimeError("transaction is not active")
        self._active_transaction = None

    def rollback_statelease_forward_transaction(self, transaction: object) -> None:
        if transaction is not self._active_transaction:
            raise RuntimeError("transaction is not active")
        current, pending, query_pending, transition_pending, update_count = transaction  # type: ignore[misc]
        self.current_state = current
        self.pending = dict(pending)
        self.query_pending = dict(query_pending)
        self.transition_pending = dict(transition_pending)
        del self.updates[update_count:]
        self._active_transaction = None

    def has_pending_statelease_observation(self, layer_index: int) -> bool:
        return layer_index in self.pending

    def has_previous_state(self, layer_index: int) -> bool:
        del layer_index
        return self.current_state is not None

    def stage_statelease_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        final_state: torch.Tensor,
    ) -> None:
        if self.fail_stage:
            raise RuntimeError("stage failed")
        observation = (
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            final_state,
        )
        self.pending[layer_index] = observation
        self.staged.append((layer_index, observation))
        self.events.append("statelease-stage")

    def discard_pending_statelease_observation(self, layer_index: int) -> None:
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
        self.query_pending[layer_index] = query
        self.query_staged.append((layer_index, query))

    def discard_pending_query_observation(self, layer_index: int) -> None:
        self.query_pending.pop(layer_index, None)
        self.query_discards.append(layer_index)

    def stage_transition_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        observation = (query, key, log_decay, beta)
        self.transition_pending[layer_index] = observation
        self.transition_staged.append((layer_index, observation))

    def discard_pending_transition_observations(self, layer_index: int) -> None:
        self.transition_pending.pop(layer_index, None)
        self.transition_discards.append(layer_index)

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        self.events.append("cache-write")
        if self.fail_update:
            raise RuntimeError("update failed")
        if layer_idx not in self.pending:
            raise RuntimeError("StateLease observation was not staged before cache write")
        if self.require_companion_observers:
            if layer_idx not in self.query_pending:
                raise RuntimeError("query observation was not staged before cache write")
            if layer_idx not in self.transition_pending:
                raise RuntimeError("transition observation was not staged before cache write")
        self.pending.pop(layer_idx)
        self.query_pending.pop(layer_idx, None)
        self.transition_pending.pop(layer_idx, None)
        self.current_state = state
        self.updates.append((layer_idx, state))
        return state


class _ReferenceCache:
    def __init__(self) -> None:
        self.current_state: torch.Tensor | None = None
        self.updates = 0

    def has_previous_state(self, layer_index: int) -> bool:
        del layer_index
        return self.current_state is not None

    def update_recurrent_state(
        self,
        state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del layer_idx
        self.current_state = state
        self.updates += 1
        return state


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kernel_calls: list[str] = []
        self.kernel_inputs: list[_StateLeaseObservation] = []
        self.events: list[str] = []
        self.fail_kernel = False
        self.bypass_kernel_wrapper = False
        self.skip_cache_write = False
        self.call_kernel_twice = False
        self.positional_kernel_arguments = False
        self.malformed_kernel_output: str | None = None
        self.chunk_gated_delta_rule = self._chunk
        self.recurrent_gated_delta_rule = self._recurrent

    def _kernel(
        self,
        kind: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        **kwargs: object,
    ) -> object:
        del kwargs
        self.events.append("kernel-start")
        if self.fail_kernel:
            raise RuntimeError("kernel failed")
        update = torch.einsum("bthk,bthv->bhkv", key, value)
        final_state = update if initial_state is None else initial_state + update
        self.kernel_calls.append(kind)
        self.kernel_inputs.append((query, key, value, g, beta, initial_state, final_state))
        self.events.append("kernel-done")
        if self.malformed_kernel_output == "not-tuple":
            return value
        if self.malformed_kernel_output == "one-item":
            return (value,)
        if self.malformed_kernel_output == "non-tensor-state":
            return value, object()
        return value, final_state

    def _chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> object:
        del chunk_size, output_final_state, use_qk_l2norm_in_kernel
        return self._kernel(
            "chunk",
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            **kwargs,
        )

    def _recurrent(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> object:
        del output_final_state, use_qk_l2norm_in_kernel
        return self._kernel(
            "recurrent",
            query,
            key,
            value,
            g,
            beta,
            initial_state,
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
        query = hidden_states + 7.0
        key = hidden_states - 3.0
        value = hidden_states + 1.0
        log_decay = (
            -torch.arange(
                1,
                query.shape[1] + 1,
                dtype=query.dtype,
                device=query.device,
            )
            .reshape(1, -1, 1)
            .expand(query.shape[:3])
        )
        beta = (
            torch.linspace(
                0.1,
                0.9,
                query.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            .reshape(1, -1, 1)
            .expand(query.shape[:3])
        )
        previous = cache_params is not None and cache_params.has_previous_state(  # type: ignore[attr-defined]
            self.layer_idx
        )
        initial_state = (
            cache_params.current_state  # type: ignore[attr-defined]
            if previous
            else None
        )
        kernel = (
            self.recurrent_gated_delta_rule
            if previous and query.shape[1] == 1
            else self.chunk_gated_delta_rule
        )
        if self.bypass_kernel_wrapper:
            kernel = self._recurrent if previous and query.shape[1] == 1 else self._chunk
        if self.positional_kernel_arguments:
            if previous and query.shape[1] == 1:
                result = kernel(
                    query,
                    key,
                    value,
                    log_decay,
                    beta,
                    initial_state,
                    True,
                    True,
                    cu_seqlens=kwargs.get("cu_seq_lens_q"),
                )
            else:
                result = kernel(
                    query,
                    key,
                    value,
                    log_decay,
                    beta,
                    64,
                    initial_state,
                    True,
                    True,
                    cu_seqlens=kwargs.get("cu_seq_lens_q"),
                )
        else:
            result = kernel(
                query,
                key,
                value,
                g=log_decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
        if self.call_kernel_twice:
            kernel(
                query,
                key,
                value,
                g=log_decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        output, final_state = result  # type: ignore[misc]
        if cache_params is not None and not self.skip_cache_write:
            cache_params.update_recurrent_state(  # type: ignore[attr-defined]
                final_state,
                self.layer_idx,
            )
        return output


class _TinyModel(torch.nn.Module):
    def __init__(self, layers: int = 1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3_5GatedDeltaNet(index) for index in range(layers)])

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


def test_observes_exact_prefill_chunk_and_decode_transition_and_states() -> None:
    model = _TinyModel()
    cache = _StateLeaseCache()
    cache.events = model.layers[0].events

    with Qwen35StateLeaseObserver(model, caches=[cache]):
        model(_input(3), cache_params=cache)
        model(_input(2), cache_params=cache)
        model(_input(1), cache_params=cache)

    assert model.layers[0].kernel_calls == ["chunk", "chunk", "recurrent"]
    assert len(cache.staged) == len(cache.updates) == 3
    assert not cache.pending
    for call_index, ((layer_index, staged), kernel_inputs) in enumerate(
        zip(cache.staged, model.layers[0].kernel_inputs, strict=True)
    ):
        assert layer_index == 0
        assert len(staged) == 7
        for observed, expected in zip(staged, kernel_inputs, strict=True):
            assert observed is expected
        assert staged[-1] is cache.updates[call_index][1]
        if call_index == 0:
            assert staged[-2] is None
        else:
            assert staged[-2] is cache.updates[call_index - 1][1]

    assert cache.events == [
        "kernel-start",
        "kernel-done",
        "statelease-stage",
        "cache-write",
        "kernel-start",
        "kernel-done",
        "statelease-stage",
        "cache-write",
        "kernel-start",
        "kernel-done",
        "statelease-stage",
        "cache-write",
    ]


def test_extracts_initial_state_from_positional_kernel_argument_six() -> None:
    initial_state = torch.randn(
        1,
        1,
        2,
        2,
        generator=torch.Generator().manual_seed(9),
    )
    model = _TinyModel()
    model.layers[0].positional_kernel_arguments = True
    cache = _StateLeaseCache(initial_state=initial_state)

    with Qwen35StateLeaseObserver(model, caches=[cache]):
        model(_input(2), cache_params=cache)

    assert cache.staged[0][1][-2] is initial_state


def test_dispatches_strictly_by_identity_and_leaves_reference_cache_alone() -> None:
    model = _TinyModel()
    first = _StateLeaseCache()
    second = _StateLeaseCache()
    reference = _ReferenceCache()

    with Qwen35StateLeaseObserver(model, caches=[first, second]):
        model(_input(2), cache_params=first)
        model(_input(2), cache_params=second)
        model(_input(2), cache_params=reference)

    assert len(first.staged) == len(second.staged) == 1
    assert reference.updates == 1


@pytest.mark.parametrize("failure", ["kernel", "stage", "update"])
def test_every_forward_failure_clears_pending_observation(failure: str) -> None:
    model = _TinyModel()
    cache = _StateLeaseCache()
    module = model.layers[0]
    original_forward = module.forward
    original_chunk = module.chunk_gated_delta_rule
    original_recurrent = module.recurrent_gated_delta_rule
    if failure == "kernel":
        module.fail_kernel = True
    elif failure == "stage":
        cache.fail_stage = True
    else:
        cache.fail_update = True

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match=f"{failure} failed"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert cache.discards == [0]
    assert module.forward.__func__ is original_forward.__func__
    assert "forward" not in module.__dict__
    assert module.chunk_gated_delta_rule is original_chunk
    assert module.recurrent_gated_delta_rule is original_recurrent


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        ("not-tuple", "exactly"),
        ("one-item", "exactly"),
        ("non-tensor-state", "tensor final state"),
    ],
)
def test_malformed_successful_kernel_output_fails_closed(
    malformed: str,
    message: str,
) -> None:
    model = _TinyModel()
    cache = _StateLeaseCache()
    model.layers[0].malformed_kernel_output = malformed

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
        pytest.raises(TypeError, match=message),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert not cache.staged
    assert cache.discards == [0]


@pytest.mark.parametrize("packed_name", ["cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens"])
def test_rejects_packed_cu_seqlens_before_kernel_or_cache_mutation(
    packed_name: str,
) -> None:
    model = _TinyModel()
    cache = _StateLeaseCache()

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
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
    cache = _StateLeaseCache()
    model.layers[0].bypass_kernel_wrapper = True
    model.layers[0].skip_cache_write = True

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="exactly one observed state-kernel call"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert not cache.staged
    assert cache.discards == [0]


def test_rejects_a_second_kernel_call_without_staging_twice() -> None:
    model = _TinyModel()
    cache = _StateLeaseCache()
    model.layers[0].call_kernel_twice = True

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="more than one state kernel"),
    ):
        model(_input(2), cache_params=cache)

    assert len(cache.staged) == 1
    assert not cache.pending
    assert cache.discards == [0]


@pytest.mark.parametrize(
    "observer_order",
    list(permutations(("statelease", "query", "transition"))),
)
def test_nests_with_existing_observers_in_every_context_order(
    observer_order: tuple[str, ...],
) -> None:
    model = _TinyModel()
    cache = _StateLeaseCache(require_companion_observers=True)
    observers = {
        "statelease": Qwen35StateLeaseObserver(model, caches=[cache]),
        "query": Qwen35QueryEnergyObserver(model, caches=[cache]),
        "transition": Qwen35TransitionObserver(model, caches=[cache]),
    }

    with ExitStack() as stack:
        for name in observer_order:
            stack.enter_context(observers[name])
        model(_input(2), cache_params=cache)

    assert len(cache.staged) == 1
    assert len(cache.query_staged) == 1
    assert len(cache.transition_staged) == 1
    assert len(cache.updates) == 1
    assert not cache.pending
    assert not cache.query_pending
    assert not cache.transition_pending


def test_nested_observers_all_discard_when_cache_write_fails() -> None:
    model = _TinyModel()
    cache = _StateLeaseCache(require_companion_observers=True)
    cache.fail_update = True

    with (
        Qwen35StateLeaseObserver(model, caches=[cache]),
        Qwen35QueryEnergyObserver(model, caches=[cache]),
        Qwen35TransitionObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="update failed"),
    ):
        model(_input(2), cache_params=cache)

    assert not cache.pending
    assert not cache.query_pending
    assert not cache.transition_pending
    assert cache.discards == [0]
    assert cache.query_discards == [0]
    assert cache.transition_discards == [0]


def test_installation_is_atomic_and_duplicate_installation_is_rejected() -> None:
    model = _TinyModel(layers=2)
    cache = _StateLeaseCache()
    first = model.layers[0]
    originals = (
        first.forward,
        first.chunk_gated_delta_rule,
        first.recurrent_gated_delta_rule,
    )
    model.layers[1].recurrent_gated_delta_rule = None  # type: ignore[assignment]

    observer = Qwen35StateLeaseObserver(model, caches=[cache])
    with pytest.raises(TypeError, match="must be callable"):
        observer.install()

    assert first.forward.__func__ is originals[0].__func__
    assert "forward" not in first.__dict__
    assert first.chunk_gated_delta_rule is originals[1]
    assert first.recurrent_gated_delta_rule is originals[2]
    assert not hasattr(model, "_recurquant_statelease_observer")

    valid_model = _TinyModel()
    first_observer = Qwen35StateLeaseObserver(valid_model, caches=[cache])
    second_observer = Qwen35StateLeaseObserver(valid_model, caches=[cache])
    with first_observer, pytest.raises(RuntimeError, match="already installed"):
        second_observer.install()


@pytest.mark.parametrize(
    ("caches", "error", "message"),
    [
        ([], ValueError, "at least one"),
        ([object()], TypeError, "stage_statelease_observation"),
    ],
)
def test_constructor_rejects_invalid_cache_contract(
    caches: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        Qwen35StateLeaseObserver(_TinyModel(), caches=caches)


def test_constructor_rejects_the_same_cache_identity_twice() -> None:
    cache = _StateLeaseCache()

    with pytest.raises(ValueError, match="same object"):
        Qwen35StateLeaseObserver(_TinyModel(), caches=[cache, cache])


class _ObservedQDQCache(RecurrentStateQDQCache):
    def __init__(self, config: Any) -> None:
        super().__init__(config, spec=QuantizationSpec(bits=4, group_size=8))
        self.observations: list[tuple[int, _StateLeaseObservation]] = []
        self.pending_statelease: dict[int, _StateLeaseObservation] = {}
        self._statelease_transaction: object | None = None

    def begin_statelease_forward_transaction(self) -> object:
        if self._statelease_transaction is not None:
            raise RuntimeError("transaction already active")
        transaction = object()
        self._statelease_transaction = transaction
        return transaction

    def commit_statelease_forward_transaction(self, transaction: object) -> None:
        if transaction is not self._statelease_transaction:
            raise RuntimeError("transaction is not active")
        self._statelease_transaction = None

    def rollback_statelease_forward_transaction(self, transaction: object) -> None:
        if transaction is not self._statelease_transaction:
            raise RuntimeError("transaction is not active")
        self.pending_statelease.clear()
        self._statelease_transaction = None

    def has_pending_statelease_observation(self, layer_index: int) -> bool:
        return layer_index in self.pending_statelease

    def stage_statelease_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        final_state: torch.Tensor,
    ) -> None:
        snapshot: _StateLeaseObservation = (
            query.detach().clone(),
            key.detach().clone(),
            value.detach().clone(),
            log_decay.detach().clone(),
            beta.detach().clone(),
            None if initial_state is None else initial_state.detach().clone(),
            final_state.detach().clone(),
        )
        self.pending_statelease[layer_index] = snapshot
        self.observations.append((layer_index, snapshot))

    def discard_pending_statelease_observation(self, layer_index: int) -> None:
        self.pending_statelease.pop(layer_index, None)

    def update_recurrent_state(
        self,
        recurrent_state: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if layer_idx not in self.pending_statelease:
            raise RuntimeError("StateLease observation was not staged before cache write")
        self.pending_statelease.pop(layer_idx)
        return super().update_recurrent_state(recurrent_state, layer_idx)


def test_real_tiny_qwen_runs_prefill_cached_chunk_and_decode() -> None:
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config = tiny_config()
    model = (
        Qwen3_5ForCausalLM._from_config(
            config,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    cache = _ObservedQDQCache(config)

    with torch.inference_mode(), Qwen35StateLeaseObserver(model, caches=[cache]):
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
    assert len(cache.observations) == 3
    assert not cache.pending_statelease
    assert [item[1][0].shape[1] for item in cache.observations] == [5, 2, 1]
    for call_index, (
        layer_index,
        (query, key, value, log_decay, beta, initial_state, final_state),
    ) in enumerate(cache.observations):
        token_count = query.shape[1]
        assert layer_index == 0
        assert (
            query.shape
            == key.shape
            == (
                1,
                token_count,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
            )
        )
        assert value.shape == (
            1,
            token_count,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )
        assert log_decay.shape == beta.shape == query.shape[:3]
        assert final_state.shape == (
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )
        assert (initial_state is None) is (call_index == 0)
        assert query.device == key.device == value.device == final_state.device == device
