from __future__ import annotations

import copy
from collections.abc import Iterator

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.statelease_equal_byte_baselines import (
    EXPANDED_RHT_Q4_Q8,
    FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    RHT_Q4_Q6_Q8,
    RHT_RESIDUAL_Q4,
    EqualByteLayout,
    update_causal_query_ema,
)
from recurquant.statelease_equal_byte_cache import (
    EqualByteLinearAttentionLayer,
    EqualByteQwen35Cache,
    Qwen35EqualByteObserver,
    create_qwen35_equal_byte_cache,
)
from tests.test_transformers_cache import tiny_config

CODECS = (EXPANDED_RHT_Q4_Q8, RHT_Q4_Q6_Q8, RHT_RESIDUAL_Q4)

# 16 rows, 128 FP32 state elements.  Every format owns exactly 180 bytes:
# expanded = 64 Q4 + 16 Q8 + 32 scales + 2 mask + 64 EMA + 2 pad;
# multibit = 64 Q4 + 16 marginal + 32 scales + 4 codes + 64 EMA;
# residual = 64 Q4 + 12 residual + 38 scales + 2 mask + 64 EMA.
ONE_LAYER_LAYOUT = EqualByteLayout(
    layer_indices=(0,),
    heads=2,
    key_rows=8,
    value_width=8,
    expanded_q8_promotions=4,
    multibit_marginal_steps=8,
    residual_q4_rows=3,
    expanded_padding_bytes=2,
    multibit_padding_bytes=0,
    residual_padding_bytes=0,
    expected_resident_bytes=180,
)

# 32 rows across two model layers.  Every format owns exactly 356 bytes.
TWO_LAYER_LAYOUT = EqualByteLayout(
    layer_indices=(0, 1),
    heads=2,
    key_rows=8,
    value_width=8,
    expanded_q8_promotions=8,
    multibit_marginal_steps=14,
    residual_q4_rows=5,
    expanded_padding_bytes=0,
    multibit_padding_bytes=0,
    residual_padding_bytes=2,
    expected_resident_bytes=356,
)


def _kernel_observation(
    *,
    tokens: int,
    state: torch.Tensor,
    query_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    query = torch.randn(1, tokens, 2, 8, dtype=query_dtype)
    key = torch.randn(1, tokens, 2, 8, dtype=query_dtype)
    value = torch.randn(1, tokens, 2, 8, dtype=query_dtype)
    log_decay = -torch.rand(1, tokens, 2)
    beta = torch.rand(1, tokens, 2, dtype=query_dtype)
    return query, key, value, log_decay, beta, state


def _direct_global_commit(
    cache: EqualByteQwen35Cache,
    *,
    tokens: int = 2,
    query_dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    transaction = cache.begin_statelease_forward_transaction()
    sources: dict[int, torch.Tensor] = {}
    try:
        for layer_index in cache.layout.layer_indices:
            layer = cache.layers[layer_index]
            assert isinstance(layer, EqualByteLinearAttentionLayer)
            initial = layer.recurrent_states[0]
            state = torch.randn(
                1,
                cache.layout.heads,
                cache.layout.key_rows,
                cache.layout.value_width,
                dtype=torch.float32,
            )
            query, key, value, log_decay, beta, final = _kernel_observation(
                tokens=tokens,
                state=state,
                query_dtype=query_dtype,
            )
            cache.stage_statelease_observation(
                layer_index,
                query,
                key,
                value,
                log_decay,
                beta,
                initial,
                final,
            )
            assert cache.has_pending_statelease_observation(layer_index)
            assert cache.update_recurrent_state(final, layer_index) is final
            sources[layer_index] = final
        cache.commit_statelease_forward_transaction(transaction)
    except BaseException:
        if transaction.active:
            cache.rollback_statelease_forward_transaction(transaction)
        raise
    return sources


def _checkpoint_tensors(
    cache: EqualByteQwen35Cache,
) -> tuple[tuple[str, torch.Tensor], ...]:
    assert cache.checkpoint is not None
    return tuple(
        (name, tensor.detach().clone()) for name, tensor in cache.checkpoint.persistent_tensors()
    )


def _assert_checkpoint_equal(
    left: EqualByteQwen35Cache,
    right: EqualByteQwen35Cache,
) -> None:
    left_tensors = _checkpoint_tensors(left)
    right_tensors = _checkpoint_tensors(right)
    assert [name for name, _ in left_tensors] == [name for name, _ in right_tensors]
    for (_, actual), (_, expected) in zip(left_tensors, right_tensors, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert left.checkpoint is not None
    assert right.checkpoint is not None
    assert left.checkpoint.evidence == right.checkpoint.evidence


@pytest.mark.parametrize("codec", CODECS)
def test_global_cache_packs_all_layers_atomically_for_each_codec(codec: str) -> None:
    torch.manual_seed(100)
    config = tiny_config(["linear_attention", "linear_attention", "full_attention"])
    cache = EqualByteQwen35Cache(
        config,
        codec=codec,  # type: ignore[arg-type]
        layout=TWO_LAYER_LAYOUT,
        record_evidence=True,
    )

    sources = _direct_global_commit(cache, tokens=3)

    assert cache.checkpoint is not None
    assert cache.checkpoint.codec == codec
    assert cache.checkpoint.resident_bytes == TWO_LAYER_LAYOUT.expected_resident_bytes
    assert cache.update_count == 1
    assert len(cache.update_evidence) == 1
    assert cache.last_evidence is cache.update_evidence[0]
    assert cache.last_evidence.layer_indices == TWO_LAYER_LAYOUT.layer_indices
    assert cache.last_evidence.raw_state_workspace_peak_bytes == (TWO_LAYER_LAYOUT.fp32_state_bytes)
    assert cache.persistent_raw_state_bytes() == 0
    assert cache.storage_summary()["raw_state_workspace_current_bytes"] == 0
    assert cache.storage_summary()["query_workspace_current_bytes"] == 0
    assert not cache._previous_states
    assert not cache._final_states
    assert not cache._queries
    assert not cache._pending_observations

    materialized = cache.checkpoint.materialize()
    assert set(materialized) == set(sources)
    for state in materialized.values():
        assert state.dtype == torch.float32
        assert state.shape == (1, 2, 8, 8)


def test_query_ema_uses_observed_input_dtype_values_and_is_causal() -> None:
    torch.manual_seed(101)
    cache = EqualByteQwen35Cache(
        tiny_config(),
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    transaction = cache.begin_statelease_forward_transaction()
    state = torch.randn(1, 2, 8, 8)
    query, key, value, log_decay, beta, final = _kernel_observation(
        tokens=4,
        state=state,
        query_dtype=torch.bfloat16,
    )
    expected = update_causal_query_ema(
        None,
        {0: query.detach().clone()},
        layout=ONE_LAYER_LAYOUT,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        None,
        final,
    )
    cache.update_recurrent_state(final, 0)
    cache.commit_statelease_forward_transaction(transaction)

    assert cache.checkpoint is not None
    torch.testing.assert_close(cache.checkpoint.query_energy_ema, expected)
    assert cache.last_evidence is not None
    assert cache.last_evidence.query_input_dtypes == ("torch.bfloat16",)
    assert cache.last_evidence.query_workspace_peak_bytes == query.numel() * 2


def test_partial_global_receipts_never_replace_previous_checkpoint() -> None:
    torch.manual_seed(105)
    cache = EqualByteQwen35Cache(
        tiny_config(["linear_attention", "linear_attention", "full_attention"]),
        codec=EXPANDED_RHT_Q4_Q8,
        layout=TWO_LAYER_LAYOUT,
    )
    _direct_global_commit(cache)
    previous = cache.checkpoint
    assert previous is not None

    transaction = cache.begin_statelease_forward_transaction()
    layer = cache.layers[0]
    assert isinstance(layer, EqualByteLinearAttentionLayer)
    initial = layer.recurrent_states[0]
    state = torch.randn(1, 2, 8, 8)
    query, key, value, log_decay, beta, final = _kernel_observation(
        tokens=1,
        state=state,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        initial,
        final,
    )
    cache.update_recurrent_state(final, 0)

    assert cache.checkpoint is previous
    assert cache.update_count == 1
    cache.rollback_statelease_forward_transaction(transaction)
    assert cache.checkpoint is previous
    assert cache.update_count == 1
    assert not cache._final_states
    assert not cache._queries


@pytest.mark.parametrize("codec", CODECS)
def test_real_tiny_qwen_prefill_and_decode_for_each_codec(codec: str) -> None:
    torch.manual_seed(102)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_equal_byte_cache(
        model,
        codec=codec,  # type: ignore[arg-type]
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode(), Qwen35EqualByteObserver(model, caches=[cache]):
        prefill = model(prompt, past_key_values=cache, use_cache=True)
        decode = model(next_token, past_key_values=cache, use_cache=True)

    assert prefill.logits.shape == (1, 4, config.vocab_size)
    assert decode.logits.shape == (1, 1, config.vocab_size)
    assert cache.update_count == 2
    assert cache.successful_tokens == 5
    assert len(cache.update_evidence) == 2
    assert cache.checkpoint is not None
    assert cache.checkpoint.resident_bytes == ONE_LAYER_LAYOUT.expected_resident_bytes
    assert cache.persistent_raw_state_bytes() == 0
    assert not cache.storage_summary()["forward_transaction_active"]
    assert cache.layers[0].has_previous_state[0]


def _initialized_tensor_values(cache: EqualByteQwen35Cache) -> Iterator[torch.Tensor]:
    for layer in cache.layers:
        attributes = getattr(layer, "__dict__", {})
        for name in ("conv_states",):
            values = attributes.get(name, {})
            if isinstance(values, dict):
                for value in values.values():
                    if isinstance(value, torch.Tensor):
                        yield value
        for name in ("keys", "values"):
            value = attributes.get(name)
            if isinstance(value, torch.Tensor):
                yield value


def test_lm_head_failure_rolls_back_every_cache_layer_and_retry_matches_clean() -> None:
    torch.manual_seed(103)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    clean_model = copy.deepcopy(model).eval()
    cache = create_qwen35_equal_byte_cache(
        model,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    clean_cache = create_qwen35_equal_byte_cache(
        clean_model,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            model(prompt, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean_model(prompt, past_key_values=clean_cache, use_cache=True)

    checkpoint_before = cache.checkpoint
    evidence_before = tuple(cache.update_evidence)
    update_count_before = cache.update_count
    tensors_before = tuple(value.detach().clone() for value in _initialized_tensor_values(cache))

    def fail_after_lm_head(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs, output
        raise RuntimeError("injected later-layer failure")

    handle = model.lm_head.register_forward_hook(fail_after_lm_head)
    try:
        with (
            torch.inference_mode(),
            Qwen35EqualByteObserver(model, caches=[cache]),
            pytest.raises(RuntimeError, match="injected later-layer failure"),
        ):
            model(next_token, past_key_values=cache, use_cache=True)
    finally:
        handle.remove()

    assert cache.checkpoint is checkpoint_before
    assert tuple(cache.update_evidence) == evidence_before
    assert cache.update_count == update_count_before
    assert not cache._pending_observations
    assert not cache._previous_states
    assert not cache._final_states
    assert not cache._queries
    tensors_after = tuple(_initialized_tensor_values(cache))
    assert len(tensors_after) == len(tensors_before)
    for actual, expected in zip(tensors_after, tensors_before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            retry = model(next_token, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean = clean_model(
                next_token,
                past_key_values=clean_cache,
                use_cache=True,
            )

    torch.testing.assert_close(retry.logits, clean.logits, rtol=0, atol=0)
    _assert_checkpoint_equal(cache, clean_cache)


def test_global_pack_failure_rolls_back_every_cache_layer_and_retry_matches_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(106)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    clean_model = copy.deepcopy(model).eval()
    cache = create_qwen35_equal_byte_cache(
        model,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    clean_cache = create_qwen35_equal_byte_cache(
        clean_model,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            model(prompt, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean_model(prompt, past_key_values=clean_cache, use_cache=True)

    checkpoint_before = cache.checkpoint
    evidence_before = tuple(cache.update_evidence)
    update_count_before = cache.update_count
    tensors_before = tuple(value.detach().clone() for value in _initialized_tensor_values(cache))

    def fail_global_pack(
        states: dict[int, torch.Tensor],
        query_ema: torch.Tensor,
    ) -> None:
        del states, query_ema
        raise RuntimeError("injected global pack failure")

    monkeypatch.setattr(cache, "_pack_candidate", fail_global_pack)
    with (
        torch.inference_mode(),
        Qwen35EqualByteObserver(model, caches=[cache]),
        pytest.raises(RuntimeError, match="injected global pack failure"),
    ):
        model(next_token, past_key_values=cache, use_cache=True)

    assert cache.checkpoint is checkpoint_before
    assert tuple(cache.update_evidence) == evidence_before
    assert cache.update_count == update_count_before
    assert not cache._pending_observations
    assert not cache._previous_states
    assert not cache._final_states
    assert not cache._queries
    tensors_after = tuple(_initialized_tensor_values(cache))
    assert len(tensors_after) == len(tensors_before)
    for actual, expected in zip(tensors_after, tensors_before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    monkeypatch.undo()
    with torch.inference_mode():
        with Qwen35EqualByteObserver(model, caches=[cache]):
            retry = model(next_token, past_key_values=cache, use_cache=True)
        with Qwen35EqualByteObserver(clean_model, caches=[clean_cache]):
            clean = clean_model(
                next_token,
                past_key_values=clean_cache,
                use_cache=True,
            )

    torch.testing.assert_close(retry.logits, clean.logits, rtol=0, atol=0)
    _assert_checkpoint_equal(cache, clean_cache)


def test_malformed_receipts_and_incomplete_forward_fail_closed() -> None:
    cache = EqualByteQwen35Cache(
        tiny_config(),
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
    )
    state = torch.randn(1, 2, 8, 8)

    with pytest.raises(RuntimeError, match="active root-model"):
        cache.update_recurrent_state(state, 0)

    transaction = cache.begin_statelease_forward_transaction()
    with pytest.raises(RuntimeError, match="no staged kernel receipt"):
        cache.update_recurrent_state(state, 0)
    cache.rollback_statelease_forward_transaction(transaction)

    transaction = cache.begin_statelease_forward_transaction()
    query, key, value, log_decay, beta, _ = _kernel_observation(
        tokens=1,
        state=state,
    )
    with pytest.raises(TypeError, match="must use torch.float32"):
        cache.stage_statelease_observation(
            0,
            query,
            key,
            value,
            log_decay,
            beta,
            None,
            state.to(torch.bfloat16),
        )
    cache.rollback_statelease_forward_transaction(transaction)

    transaction = cache.begin_statelease_forward_transaction()
    with pytest.raises(RuntimeError, match="did not produce exactly one"):
        cache.commit_statelease_forward_transaction(transaction)
    cache.rollback_statelease_forward_transaction(transaction)


def test_receipt_requires_exact_kernel_final_state_identity() -> None:
    cache = EqualByteQwen35Cache(
        tiny_config(),
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
    )
    transaction = cache.begin_statelease_forward_transaction()
    state = torch.randn(1, 2, 8, 8)
    query, key, value, log_decay, beta, final = _kernel_observation(
        tokens=1,
        state=state,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        None,
        final,
    )

    with pytest.raises(RuntimeError, match="final-state identity"):
        cache.update_recurrent_state(final.clone(), 0)
    cache.rollback_statelease_forward_transaction(transaction)


def test_lifecycle_storage_identity_reorder_offload_prefetch_and_reset() -> None:
    torch.manual_seed(104)
    cache = EqualByteQwen35Cache(
        tiny_config(),
        codec=RHT_RESIDUAL_Q4,
        layout=ONE_LAYER_LAYOUT,
        record_evidence=True,
    )
    _direct_global_commit(cache)

    summary = cache.storage_summary()
    assert summary["resident_bytes"] == ONE_LAYER_LAYOUT.expected_resident_bytes
    assert summary["persistent_raw_state_bytes"] == 0
    persistent = cache.persistent_recurrent_tensors()
    assert persistent
    assert all(
        tensor.dtype != torch.float32 or name == "query_energy_ema" for name, tensor in persistent
    )

    cache.reorder_cache(torch.tensor([0], dtype=torch.long))
    with pytest.raises(ValueError, match="batch-one"):
        cache.reorder_cache(torch.tensor([0, 0], dtype=torch.long))
    with pytest.raises(ValueError, match="cannot repeat"):
        cache.batch_repeat_interleave(2)
    with pytest.raises(RuntimeError, match="speculative past recording"):
        cache.activate_past_recording()
    with pytest.raises(RuntimeError, match="cannot crop"):
        cache.crop(-1)
    layer = cache.layers[0]
    assert isinstance(layer, EqualByteLinearAttentionLayer)
    with pytest.raises(RuntimeError, match="speculative past recording"):
        layer.activate_past_recording()
    with pytest.raises(RuntimeError, match="cannot crop"):
        layer.crop(-1)

    cache.offload_all()
    assert cache.checkpoint is not None
    assert all(tensor.device.type == "cpu" for _, tensor in cache.checkpoint.persistent_tensors())
    cache.prefetch_all()
    assert cache.checkpoint is not None
    assert cache.checkpoint.resident_bytes == ONE_LAYER_LAYOUT.expected_resident_bytes

    cache.reset()
    assert cache.checkpoint is None
    assert cache.update_count == 0
    assert not cache.update_evidence
    assert not cache.layers[0].has_previous_state[0]
    assert cache.persistent_raw_state_bytes() == 0


def test_frozen_qwen_accounting_constructs_without_allocating_raw_state() -> None:
    layer_types = [
        (
            "linear_attention"
            if index in FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.layer_indices
            else "full_attention"
        )
        for index in range(23)
    ]
    config = tiny_config(layer_types)
    config.linear_num_value_heads = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.heads
    config.linear_key_head_dim = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.key_rows
    config.linear_value_head_dim = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.value_width
    cache = EqualByteQwen35Cache(
        config,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    )

    assert cache.checkpoint is None
    assert cache.persistent_recurrent_tensors() == ()
    assert cache.persistent_raw_state_bytes() == 0
    assert cache.storage_summary()["expected_resident_bytes"] == 3_454_664
    assert tuple(index for index, _ in cache.equal_byte_layers()) == (
        FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.layer_indices
    )


def test_observer_rejects_non_equal_byte_cache_identity() -> None:
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()

    with pytest.raises(TypeError, match="only EqualByteQwen35Cache"):
        Qwen35EqualByteObserver(model, caches=[object()])

    cache = create_qwen35_equal_byte_cache(
        model,
        codec=EXPANDED_RHT_Q4_Q8,
        layout=ONE_LAYER_LAYOUT,
    )
    with pytest.raises(TypeError, match="outer Qwen3_5ForCausalLM"):
        Qwen35EqualByteObserver(model.model, caches=[cache])
