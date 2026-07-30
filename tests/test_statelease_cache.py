from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

import recurquant.statelease_cache as statelease_cache_module
from recurquant.qwen35 import (
    EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256,
    EXPERIMENT010_STATELEASE_LAYER_QUOTAS,
    create_qwen35_experiment010_statelease_cache,
    create_qwen35_statelease_cache,
    experiment010_statelease_effective_plan_sha256,
)
from recurquant.row_policy import (
    ExactBudgetRowPlan,
    RowLocation,
    select_rows_exact_budget,
)
from recurquant.statelease import (
    replay_gated_delta_state,
    replay_gated_delta_updates,
)
from recurquant.statelease_cache import (
    StateLeaseLinearAttentionLayer,
    StateLeaseRecurrentStateCache,
)
from recurquant.statelease_observer import Qwen35StateLeaseObserver
from tests.test_transformers_cache import tiny_config


def _target_for_promotions(promotions: int) -> int:
    total_groups = 16
    group_size = 8
    low_group_bytes = math.ceil(4 * group_size / 8) + 2
    promotion_increment = math.ceil(8 * group_size / 8) - math.ceil(4 * group_size / 8)
    return (
        total_groups * low_group_bytes
        + math.ceil(total_groups / 8)
        + promotions * promotion_increment
    )


def _plan(promotions: int = 5) -> ExactBudgetRowPlan:
    return select_rows_exact_budget(
        {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)},
        target_resident_bytes=_target_for_promotions(promotions),
        group_size=8,
    )


def _experiment010_plan() -> ExactBudgetRowPlan:
    rows = tuple(
        RowLocation(
            layer_index=layer_index,
            head_index=flat_index // 128,
            row_index=flat_index % 128,
        )
        for layer_index, quota in EXPERIMENT010_STATELEASE_LAYER_QUOTAS.items()
        for flat_index in range(quota)
    )
    return ExactBudgetRowPlan(
        low_bits=4,
        high_bits=8,
        group_size=128,
        scale_bits=16,
        total_groups=36_864,
        mask_bytes=4_608,
        promotion_increment_bytes=64,
        target_resident_bytes=2_564_096,
        resident_bytes=2_564_096,
        high_precision_rows=rows,
        score_shapes=tuple(
            (layer_index, 16, 128) for layer_index in EXPERIMENT010_STATELEASE_LAYER_QUOTAS
        ),
    )


def _cache(
    *,
    promotions: int = 5,
    record_evidence: bool = False,
) -> StateLeaseRecurrentStateCache:
    return StateLeaseRecurrentStateCache(
        tiny_config(),
        plan=_plan(promotions),
        record_evidence=record_evidence,
    )


def _layer(cache: StateLeaseRecurrentStateCache) -> StateLeaseLinearAttentionLayer:
    layer = cache.layers[0]
    assert isinstance(layer, StateLeaseLinearAttentionLayer)
    return layer


def _signals(
    *,
    initial_state: torch.Tensor | None,
    tokens: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn((1, tokens, 2, 8), generator=generator)
    key = torch.randn((1, tokens, 2, 8), generator=generator)
    value = torch.randn((1, tokens, 2, 8), generator=generator)
    log_decay = -0.2 * torch.rand((1, tokens, 2), generator=generator)
    beta = torch.rand((1, tokens, 2), generator=generator)
    source = (
        torch.zeros((1, 2, 8, 8), dtype=torch.float32) if initial_state is None else initial_state
    )
    final_state = replay_gated_delta_state(
        source,
        key,
        value,
        log_decay,
        beta,
    )
    return query, key, value, log_decay, beta, final_state


def _commit(
    cache: StateLeaseRecurrentStateCache,
    *,
    tokens: int = 1,
    seed: int,
    initial_state: torch.Tensor | None | object = ...,
) -> torch.Tensor:
    layer = _layer(cache)
    if initial_state is ...:
        initial_state = layer.recurrent_states[0]
    assert initial_state is None or isinstance(initial_state, torch.Tensor)
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=initial_state,
        tokens=tokens,
        seed=seed,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        final_state,
    )
    return cache.update_recurrent_state(final_state, layer_idx=0)


def _prefill(cache: StateLeaseRecurrentStateCache, *, seed: int = 101) -> torch.Tensor:
    return _commit(cache, tokens=3, seed=seed, initial_state=None)


def _fill_four_records(cache: StateLeaseRecurrentStateCache) -> None:
    for offset in range(4):
        _commit(cache, seed=200 + offset)
        assert _layer(cache)._buffer_valid_count() == offset + 1


def _risk_sequence(values: tuple[float, float]):
    iterator: Iterator[float] = iter(values)

    def risk(
        reference: torch.Tensor,
        approximation: torch.Tensor,
        query_energy: torch.Tensor,
    ) -> torch.Tensor:
        del approximation, query_energy
        return torch.tensor(next(iterator), device=reference.device)

    return risk


def test_prefill_builds_one_rht_cqer_checkpoint_and_zero_age_lease() -> None:
    cache = _cache(promotions=5, record_evidence=True)

    restored = _prefill(cache)

    layer = _layer(cache)
    packed = layer.packed_checkpoint
    assert packed is not None
    assert packed.right_rht_layer_index == 0
    assert packed.right_rht_expected_heads == 2
    assert packed.high_precision_groups == 5
    assert layer._buffer_valid_count() == 0
    assert layer.query_energy_ema is not None
    assert layer.query_energy_ema.dtype == torch.float32
    assert layer.normalized_key_buffer is not None
    assert layer.normalized_key_buffer.dtype == torch.bfloat16
    assert layer.update_buffer is not None
    assert layer.update_buffer.dtype == torch.bfloat16
    assert layer.log_decay_buffer is not None
    assert layer.log_decay_buffer.dtype == torch.float32
    assert layer.valid_count is not None
    assert layer.valid_count.dtype == torch.int32
    assert torch.equal(restored, layer.recurrent_states[0])
    assert cache.update_evidence[0].action == "checkpoint_prefill"


def test_first_four_decode_steps_append_without_repacking() -> None:
    cache = _cache()
    _prefill(cache)
    layer = _layer(cache)
    checkpoint = layer.packed_checkpoint

    for offset in range(4):
        _commit(cache, seed=300 + offset)
        assert layer.packed_checkpoint is checkpoint
        assert layer._buffer_valid_count() == offset + 1
        assert layer.last_action == "replay_append"

    diagnostics = layer.statelease_diagnostics()
    assert diagnostics["checkpoint_count"] == 1
    assert diagnostics["replay_valid_count"] == 4
    assert diagnostics["state_updates"] == 5


def test_full_buffer_cut5_compares_same_raw_s5_and_clears_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    _prefill(cache)
    _fill_four_records(cache)
    references: list[torch.Tensor] = []
    query_energies: list[torch.Tensor] = []
    values = iter((0.2, 0.1))
    full_events: list[tuple[int, int]] = []
    original_append = StateLeaseLinearAttentionLayer._append_candidate_record

    def capture_append(**kwargs: object) -> None:
        original_append(**kwargs)
        valid_count = kwargs["valid_count"]
        assert isinstance(valid_count, torch.Tensor)
        full_events.append((int(kwargs["slot"]), int(valid_count.item())))

    def capture_risk(
        reference: torch.Tensor,
        approximation: torch.Tensor,
        query_energy: torch.Tensor,
    ) -> torch.Tensor:
        del approximation
        references.append(reference)
        query_energies.append(query_energy)
        return torch.tensor(next(values), device=reference.device)

    monkeypatch.setattr(
        statelease_cache_module,
        "query_weighted_row_mse",
        capture_risk,
    )
    monkeypatch.setattr(
        StateLeaseLinearAttentionLayer,
        "_append_candidate_record",
        staticmethod(capture_append),
    )
    layer = _layer(cache)
    initial = layer.recurrent_states[0]
    assert initial is not None
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=initial,
        tokens=1,
        seed=399,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        initial,
        final_state,
    )
    pending = layer._pending_statelease_observation
    assert pending is not None
    raw_candidate_ema = pending.candidate_ema.clone()
    cache.update_recurrent_state(final_state, layer_idx=0)

    assert layer.last_boundary == 5
    assert layer.last_action == "boundary_5"
    assert layer._buffer_valid_count() == 0
    assert layer.boundary5_count == 1
    assert full_events == [(4, 5)]
    assert references[0] is references[1]
    assert query_energies[0] is query_energies[1]
    assert layer.query_energy_ema is not None
    assert torch.equal(layer.query_energy_ema, raw_candidate_ema)
    assert torch.allclose(
        query_energies[0].sum(dim=-1),
        torch.ones(2),
        rtol=0,
        atol=2e-6,
    )


def test_exact_risk_tie_stably_chooses_cut5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    _prefill(cache)
    _fill_four_records(cache)
    monkeypatch.setattr(
        statelease_cache_module,
        "query_weighted_row_mse",
        _risk_sequence((0.125, 0.125)),
    )

    _commit(cache, seed=401)

    layer = _layer(cache)
    assert layer.last_boundary == 5
    assert layer._buffer_valid_count() == 0
    assert layer.tie_count == 1
    assert layer.statelease_diagnostics()["tie_count"] == 1


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros((2, 8), dtype=torch.float32),
        torch.full((2, 8), float("nan"), dtype=torch.float32),
    ],
)
def test_risk_energy_normalization_fails_closed_on_invalid_mass(
    invalid: torch.Tensor,
) -> None:
    with pytest.raises(RuntimeError, match="invalid per-head mass"):
        StateLeaseLinearAttentionLayer._normalized_risk_energy(invalid)


def test_cut4_replaces_checkpoint_at_s4_and_retains_token5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    _prefill(cache)
    _fill_four_records(cache)
    previous_checkpoint = _layer(cache).packed_checkpoint
    monkeypatch.setattr(
        statelease_cache_module,
        "query_weighted_row_mse",
        _risk_sequence((0.05, 0.2)),
    )

    restored = _commit(cache, seed=403)

    layer = _layer(cache)
    assert layer.last_boundary == 4
    assert layer.last_action == "boundary_4"
    assert layer.packed_checkpoint is not previous_checkpoint
    assert layer._buffer_valid_count() == 1
    assert layer.boundary4_count == 1
    assert torch.equal(restored, layer.recurrent_states[0])


def test_recurrent_view_is_checkpoint_plus_exact_stored_update_replay() -> None:
    cache = _cache()
    _prefill(cache)
    for offset in range(3):
        _commit(cache, seed=500 + offset)
    layer = _layer(cache)
    packed = layer.packed_checkpoint
    assert packed is not None
    assert layer.normalized_key_buffer is not None
    assert layer.update_buffer is not None
    assert layer.log_decay_buffer is not None
    count = layer._buffer_valid_count()

    expected = replay_gated_delta_updates(
        packed.dequantize(),
        layer.normalized_key_buffer[:count].unsqueeze(0),
        layer.update_buffer[:count].unsqueeze(0),
        layer.log_decay_buffer[:count].unsqueeze(0),
    )

    assert torch.equal(layer.recurrent_states[0], expected)


def test_capacity_accounting_charges_all_five_slots_not_occupancy() -> None:
    plan = _plan(5)
    cache = StateLeaseRecurrentStateCache(tiny_config(), plan=plan)
    _prefill(cache)

    summary = cache.storage_summary()
    expected_query_ema = 2 * 8 * 4
    expected_key = 5 * 2 * 8 * 2
    expected_update = 5 * 2 * 8 * 2
    expected_decay = 5 * 2 * 4
    expected_count = 4
    expected_replay = expected_key + expected_update + expected_decay + expected_count
    expected_total = plan.resident_bytes + expected_query_ema + expected_replay
    assert summary["checkpoint_bytes"] == plan.resident_bytes
    assert summary["query_ema_bytes"] == expected_query_ema
    assert summary["replay_capacity_bytes"] == expected_replay
    assert summary["replay_occupied_bytes"] == expected_count
    assert summary["resident_bytes_including_statelease"] == expected_total
    assert summary["effective_bits_per_state_element"] == pytest.approx(expected_total * 8 / 128)

    _commit(cache, seed=601)
    occupied_record = 2 * 8 * 2 + 2 * 8 * 2 + 2 * 4
    after = cache.storage_summary()
    assert after["replay_capacity_bytes"] == expected_replay
    assert after["replay_occupied_bytes"] == expected_count + occupied_record
    assert after["resident_bytes_including_statelease"] == expected_total


def test_multitoken_cached_chunk_forces_checkpoint_and_clears_records() -> None:
    cache = _cache()
    _prefill(cache)
    _commit(cache, seed=701)
    _commit(cache, seed=702)
    previous = _layer(cache).packed_checkpoint

    _commit(cache, tokens=2, seed=703)

    layer = _layer(cache)
    assert layer.packed_checkpoint is not previous
    assert layer._buffer_valid_count() == 0
    assert layer.last_action == "checkpoint_chunk"
    assert layer.last_reason == "multi_token"


def test_stage_rejects_wrong_transition_and_invalid_batch() -> None:
    cache = _cache()
    _prefill(cache)
    layer = _layer(cache)
    initial = layer.recurrent_states[0]
    assert initial is not None
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=initial,
        tokens=1,
        seed=801,
    )
    corrupted = final_state.clone()
    corrupted[0, 0, 0, 0] += 0.1

    with pytest.raises(ValueError, match="does not reproduce"):
        cache.stage_statelease_observation(
            0,
            query,
            key,
            value,
            log_decay,
            beta,
            initial,
            corrupted,
        )
    assert layer._pending_statelease_observation is None

    with pytest.raises(ValueError, match="query must have shape"):
        cache.stage_statelease_observation(
            0,
            query.expand(2, -1, -1, -1),
            key,
            value,
            log_decay,
            beta,
            initial,
            final_state,
        )
    assert layer._pending_statelease_observation is None


def test_failed_second_candidate_pack_rolls_back_every_persistent_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(record_evidence=True)
    _prefill(cache)
    _fill_four_records(cache)
    layer = _layer(cache)
    previous_checkpoint = layer.packed_checkpoint
    assert layer.query_energy_ema is not None
    assert layer.normalized_key_buffer is not None
    assert layer.update_buffer is not None
    assert layer.log_decay_buffer is not None
    assert layer.valid_count is not None
    previous_ema = layer.query_energy_ema.clone()
    previous_key = layer.normalized_key_buffer.clone()
    previous_update = layer.update_buffer.clone()
    previous_decay = layer.log_decay_buffer.clone()
    previous_count = layer.valid_count.clone()
    previous_diagnostics = layer.statelease_diagnostics()
    previous_evidence = list(cache.update_evidence)
    original = StateLeaseLinearAttentionLayer._pack_checkpoint
    calls = 0

    def fail_second_pack(
        self: StateLeaseLinearAttentionLayer,
        *args: object,
        **kwargs: object,
    ):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected cut5 pack failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        StateLeaseLinearAttentionLayer,
        "_pack_checkpoint",
        fail_second_pack,
    )
    initial = layer.recurrent_states[0]
    assert initial is not None
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=initial,
        tokens=1,
        seed=899,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        initial,
        final_state,
    )

    with pytest.raises(RuntimeError, match="injected cut5"):
        cache.update_recurrent_state(final_state, layer_idx=0)

    assert calls == 2
    assert layer.packed_checkpoint is previous_checkpoint
    assert torch.equal(layer.query_energy_ema, previous_ema)
    assert torch.equal(layer.normalized_key_buffer, previous_key)
    assert torch.equal(layer.update_buffer, previous_update)
    assert torch.equal(layer.log_decay_buffer, previous_decay)
    assert torch.equal(layer.valid_count, previous_count)
    assert layer.statelease_diagnostics() == previous_diagnostics
    assert cache.update_evidence == previous_evidence
    assert layer._pending_statelease_observation is None


def test_reset_transfer_and_batch_one_reorder_are_fail_closed() -> None:
    cache = _cache(record_evidence=True)
    _prefill(cache)
    _commit(cache, seed=901)
    layer = _layer(cache)
    checkpoint = layer.packed_checkpoint

    with pytest.raises(ValueError, match="batch-one"):
        cache.reorder_cache(torch.tensor([0, 0], dtype=torch.long))
    assert layer.packed_checkpoint is checkpoint
    cache.reorder_cache(torch.tensor([0], dtype=torch.long))

    initial = layer.recurrent_states[0]
    assert initial is not None
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=initial,
        tokens=1,
        seed=902,
    )
    cache.stage_statelease_observation(
        0,
        query,
        key,
        value,
        log_decay,
        beta,
        initial,
        final_state,
    )
    with pytest.raises(RuntimeError, match="cannot offload with a pending"):
        layer.offload()
    assert layer._pending_statelease_observation is None

    layer.offload()
    assert layer.packed_checkpoint is not None
    assert layer.packed_checkpoint.low_payload.device.type == "cpu"
    layer.prefetch()

    cache.reset()
    assert layer._buffer_valid_count() == 0
    assert layer.query_energy_ema is not None
    assert torch.equal(
        layer.query_energy_ema,
        torch.full((2, 8), 1.0 / 8, dtype=torch.float32),
    )
    diagnostics = layer.statelease_diagnostics()
    assert diagnostics["state_updates"] == 0
    assert diagnostics["checkpoint_count"] == 0
    assert cache.update_evidence == []


def test_real_tiny_qwen_runs_prefill_cached_chunk_and_decode() -> None:
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_statelease_cache(
        model,
        plan=_plan(5),
        record_evidence=True,
    )

    with torch.inference_mode(), Qwen35StateLeaseObserver(model, caches=[cache]):
        prefill = model(
            torch.randint(0, config.vocab_size, (1, 5)),
            past_key_values=cache,
        )
        chunk = model(
            torch.randint(0, config.vocab_size, (1, 2)),
            past_key_values=cache,
        )
        decode = model(
            torch.randint(0, config.vocab_size, (1, 1)),
            past_key_values=cache,
        )

    assert prefill.logits.shape == (1, 5, config.vocab_size)
    assert chunk.logits.shape == (1, 2, config.vocab_size)
    assert decode.logits.shape == (1, 1, config.vocab_size)
    diagnostics = _layer(cache).statelease_diagnostics()
    assert diagnostics["state_updates"] == 3
    assert diagnostics["checkpoint_count"] == 2
    assert diagnostics["replay_valid_count"] == 1
    assert diagnostics["pending_observation"] is False
    assert len(cache.update_evidence) == 3


def test_frozen_recurrent_state_path_rejects_non_fp32_state() -> None:
    cache = _cache()
    query, key, value, log_decay, beta, final_state = _signals(
        initial_state=None,
        tokens=2,
        seed=1001,
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
            final_state.to(torch.bfloat16),
        )


def test_recurrent_only_clear_rejects_live_convolution_history() -> None:
    cache = _cache()
    _prefill(cache)
    layer = _layer(cache)
    checkpoint = layer.packed_checkpoint
    layer.conv_states[0] = torch.randn(1, 4, 2)
    layer.is_conv_states_initialized[0] = True
    layer.has_previous_state[0] = True

    with pytest.raises(RuntimeError, match="complete layer atomically"):
        layer.recurrent_states[0] = None

    assert layer.packed_checkpoint is checkpoint
    assert layer.has_previous_state[0]


def test_persistent_storage_validator_rejects_views_and_aliases() -> None:
    backing = torch.zeros(16)
    view = backing[:8]
    with pytest.raises(RuntimeError, match="owns"):
        statelease_cache_module._validate_owned_persistent_tensors(  # type: ignore[attr-defined]
            (("view", view),)
        )
    first = torch.zeros(8)
    with pytest.raises(RuntimeError, match="aliases"):
        statelease_cache_module._validate_owned_persistent_tensors(  # type: ignore[attr-defined]
            (("first", first), ("second", first))
        )


def test_outer_model_transaction_rolls_back_every_layer_and_retry_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1011)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    failed_cache = create_qwen35_statelease_cache(
        model,
        plan=_plan(5),
        record_evidence=True,
    )
    clean_cache = create_qwen35_statelease_cache(
        model,
        plan=_plan(5),
        record_evidence=True,
    )
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    token = torch.randint(0, config.vocab_size, (1, 1))

    with (
        torch.inference_mode(),
        Qwen35StateLeaseObserver(
            model,
            caches=[failed_cache, clean_cache],
        ),
    ):
        model(prompt, past_key_values=failed_cache)
        model(prompt, past_key_values=clean_cache)
        failed_linear = _layer(failed_cache)
        failed_attention = failed_cache.layers[1]
        conv_reference = failed_linear.conv_states[0]
        assert isinstance(conv_reference, torch.Tensor)
        conv_value = conv_reference.clone()
        recurrent_value = failed_linear.materialize_recurrent_state()
        assert recurrent_value is not None
        recurrent_value = recurrent_value.clone()
        checkpoint_reference = failed_linear.packed_checkpoint
        attention_key_reference = failed_attention.keys
        attention_value_reference = failed_attention.values
        diagnostics = failed_linear.statelease_diagnostics()
        evidence = list(failed_cache.update_evidence)

        with monkeypatch.context() as patch:

            def fail_after_cache_updates(*args: object, **kwargs: object) -> object:
                del args, kwargs
                raise RuntimeError("injected post-cache failure")

            patch.setattr(model.lm_head, "forward", fail_after_cache_updates)
            with pytest.raises(RuntimeError, match="post-cache"):
                model(token, past_key_values=failed_cache)

        assert failed_linear.conv_states[0] is conv_reference
        assert torch.equal(failed_linear.conv_states[0], conv_value)
        assert failed_linear.packed_checkpoint is checkpoint_reference
        torch.testing.assert_close(
            failed_linear.materialize_recurrent_state(),
            recurrent_value,
            rtol=0,
            atol=0,
        )
        assert failed_attention.keys is attention_key_reference
        assert failed_attention.values is attention_value_reference
        assert failed_linear.statelease_diagnostics() == diagnostics
        assert failed_cache.update_evidence == evidence

        retried = model(token, past_key_values=failed_cache)
        clean = model(token, past_key_values=clean_cache)

    torch.testing.assert_close(retried.logits, clean.logits, rtol=0, atol=0)


def test_cache_write_failure_rolls_back_convolution_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1013)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    failed_cache = create_qwen35_statelease_cache(model, plan=_plan(5))
    clean_cache = create_qwen35_statelease_cache(model, plan=_plan(5))
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    original_update = failed_cache.update_recurrent_state

    with (
        torch.inference_mode(),
        Qwen35StateLeaseObserver(
            model,
            caches=[failed_cache, clean_cache],
        ),
    ):

        def fail_update(*args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            raise RuntimeError("injected cache write failure")

        with monkeypatch.context() as patch:
            patch.setattr(failed_cache, "update_recurrent_state", fail_update)
            with pytest.raises(RuntimeError, match="cache write"):
                model(prompt, past_key_values=failed_cache)

        failed_layer = _layer(failed_cache)
        assert failed_layer.conv_states[0] is None
        assert not failed_layer.is_conv_states_initialized[0]
        assert not failed_layer.has_previous_state[0]
        assert failed_layer.packed_checkpoint is None

        monkeypatch.setattr(
            failed_cache,
            "update_recurrent_state",
            original_update,
        )
        retried = model(prompt, past_key_values=failed_cache)
        clean = model(prompt, past_key_values=clean_cache)

    torch.testing.assert_close(retried.logits, clean.logits, rtol=0, atol=0)


def test_experiment010_factory_is_identity_locked() -> None:
    plan = _experiment010_plan()
    layer_types = [
        (
            "linear_attention"
            if layer_index in EXPERIMENT010_STATELEASE_LAYER_QUOTAS
            else "full_attention"
        )
        for layer_index in range(23)
    ]
    config = tiny_config(layer_types)
    config.linear_num_value_heads = 16
    config.linear_num_key_heads = 16
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128

    assert (
        experiment010_statelease_effective_plan_sha256(plan)
        == EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256
    )
    cache = create_qwen35_experiment010_statelease_cache(
        config,
        plan=plan,
    )
    assert cache.selection_method == statelease_cache_module.STATELEASE_SELECTION_METHOD
    assert cache.experiment_identity_sha256 == EXPERIMENT010_STATELEASE_EFFECTIVE_PLAN_SHA256
    assert all(
        layer.selection_method == statelease_cache_module.STATELEASE_SELECTION_METHOD
        for _, layer in cache.statelease_layers()
    )

    tampered = replace(plan, resident_bytes=plan.resident_bytes + 1)
    with pytest.raises(ValueError, match="storage identity"):
        create_qwen35_experiment010_statelease_cache(
            config,
            plan=tampered,
        )

    generic = create_qwen35_statelease_cache(
        tiny_config(),
        plan=_plan(),
    )
    assert generic.selection_method != statelease_cache_module.STATELEASE_SELECTION_METHOD
    assert generic.experiment_identity_sha256 is None
