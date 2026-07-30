from __future__ import annotations

import math
from contextlib import ExitStack
from itertools import islice, permutations

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.qwen35 import create_qwen35_query_ema_exact_budget_cache
from recurquant.row_policy import (
    ExactBudgetRowPlan,
    RowLocation,
    select_rows_exact_budget,
)
from recurquant.statelease import replay_gated_delta_state
from recurquant.statelease_baselines import (
    FixedCC1RecurrentStateCache,
    FixedCC2RecurrentStateCache,
    FixedCC4RecurrentStateCache,
    FixedCC5RecurrentStateCache,
    FixedCC8RecurrentStateCache,
    FixedCut4In5RecurrentStateCache,
    FixedReplayLinearAttentionLayer,
    FixedReplayRecurrentStateCache,
    create_fixed_replay_cache,
    fixed_cc1,
    fixed_cc2,
    fixed_cc4,
    fixed_cc5,
    fixed_cc8,
    fixed_cut4_in5,
    fixed_replay_policy,
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


def _cache(
    mode: str,
    *,
    record_evidence: bool = False,
) -> FixedReplayRecurrentStateCache:
    return create_fixed_replay_cache(
        tiny_config(),
        plan=_plan(),
        mode=mode,
        record_evidence=record_evidence,
    )


def _layer(cache: FixedReplayRecurrentStateCache) -> FixedReplayLinearAttentionLayer:
    layer = cache.layers[0]
    assert isinstance(layer, FixedReplayLinearAttentionLayer)
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


def _stage(
    cache: FixedReplayRecurrentStateCache,
    *,
    tokens: int,
    seed: int,
    initial_state: torch.Tensor | None | object = ...,
) -> torch.Tensor:
    layer = _layer(cache)
    if initial_state is ...:
        initial_state = layer.materialize_recurrent_state()
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
    return final_state


def _commit(
    cache: FixedReplayRecurrentStateCache,
    *,
    seed: int,
    tokens: int = 1,
    initial_state: torch.Tensor | None | object = ...,
) -> torch.Tensor:
    final_state = _stage(
        cache,
        tokens=tokens,
        seed=seed,
        initial_state=initial_state,
    )
    return cache.update_recurrent_state(final_state, layer_idx=0)


def _prefill(cache: FixedReplayRecurrentStateCache) -> torch.Tensor:
    return _commit(cache, seed=101, tokens=3, initial_state=None)


@pytest.mark.parametrize(
    ("factory", "expected_type", "mode"),
    [
        (fixed_cc1, FixedCC1RecurrentStateCache, "fixed_cc1"),
        (fixed_cc2, FixedCC2RecurrentStateCache, "fixed_cc2"),
        (fixed_cc4, FixedCC4RecurrentStateCache, "fixed_cc4"),
        (fixed_cc5, FixedCC5RecurrentStateCache, "fixed_cc5"),
        (fixed_cut4_in5, FixedCut4In5RecurrentStateCache, "fixed_cut4_in5"),
        (fixed_cc8, FixedCC8RecurrentStateCache, "fixed_cc8"),
    ],
)
def test_public_named_factories_are_exact_and_alias_free(
    factory: object,
    expected_type: type[FixedReplayRecurrentStateCache],
    mode: str,
) -> None:
    cache = factory(tiny_config(), plan=_plan())  # type: ignore[operator]

    assert isinstance(cache, expected_type)
    assert cache.policy.mode == mode
    assert _layer(cache).policy is cache.policy
    assert fixed_replay_policy(mode) is cache.policy


def test_factory_rejects_unknown_modes_and_non_string_modes() -> None:
    with pytest.raises(ValueError, match="unknown fixed replay mode"):
        create_fixed_replay_cache(tiny_config(), plan=_plan(), mode="cc4")
    with pytest.raises(TypeError, match="must be a string"):
        create_fixed_replay_cache(tiny_config(), plan=_plan(), mode=4)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "period"),
    [
        ("fixed_cc1", 1),
        ("fixed_cc2", 2),
        ("fixed_cc4", 4),
        ("fixed_cc5", 5),
        ("fixed_cc8", 8),
    ],
)
def test_fixed_cc_schedules_checkpoint_at_exact_strides(
    mode: str,
    period: int,
) -> None:
    cache = _cache(mode, record_evidence=True)
    _prefill(cache)
    layer = _layer(cache)
    events: list[int] = []

    for token_index in range(1, 2 * period + 1):
        before = layer.scheduled_checkpoint_count
        _commit(cache, seed=200 + token_index)
        if layer.scheduled_checkpoint_count != before:
            events.append(token_index)
        assert layer._buffer_valid_count() == token_index % period

    assert events == [period, 2 * period]
    assert layer.checkpoint_count == 3
    assert layer.forced_checkpoint_count == 1
    assert layer.scheduled_checkpoint_count == 2
    assert layer.replay_append_count == 2 * (period - 1)
    assert layer.retained_tail_count == 0
    assert cache.update_evidence[-1].action == f"{mode}_checkpoint"
    assert cache.update_evidence[-1].boundary == period


def test_fixed_cut4_in5_retains_exact_rounded_fifth_record_and_strides_by_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache("fixed_cut4_in5", record_evidence=True)
    _prefill(cache)
    layer = _layer(cache)
    for token_index in range(1, 5):
        _commit(cache, seed=300 + token_index)
        assert layer._buffer_valid_count() == token_index

    raw_s4 = layer.materialize_recurrent_state()
    assert raw_s4 is not None
    final_state = _stage(cache, tokens=1, seed=305)
    pending = layer._pending_statelease_observation
    assert pending is not None
    assert pending.normalized_key is not None
    assert pending.update is not None
    assert pending.log_decay is not None
    expected_key = pending.normalized_key.to(torch.bfloat16)
    expected_update = pending.update.to(torch.bfloat16)
    expected_decay = pending.log_decay.to(torch.float32)

    original_pack = layer._pack_checkpoint
    packed_sources: list[torch.Tensor] = []

    def capture_pack(source: torch.Tensor, **kwargs: object):
        packed_sources.append(source.detach().clone())
        return original_pack(source, **kwargs)

    monkeypatch.setattr(layer, "_pack_checkpoint", capture_pack)
    restored = cache.update_recurrent_state(final_state, layer_idx=0)

    assert len(packed_sources) == 1
    assert torch.equal(packed_sources[0], raw_s4)
    assert layer._buffer_valid_count() == 1
    assert layer.normalized_key_buffer is not None
    assert layer.update_buffer is not None
    assert layer.log_decay_buffer is not None
    assert torch.equal(layer.normalized_key_buffer[0], expected_key)
    assert torch.equal(layer.update_buffer[0], expected_update)
    assert torch.equal(layer.log_decay_buffer[0], expected_decay)
    assert torch.equal(restored, layer.materialize_recurrent_state())
    assert layer.last_action == "fixed_cut4_in5_boundary_4"
    assert layer.last_boundary == 4
    assert layer.retained_tail_count == 1

    second_event: int | None = None
    for token_index in range(6, 10):
        before = layer.scheduled_checkpoint_count
        _commit(cache, seed=300 + token_index)
        if layer.scheduled_checkpoint_count != before:
            second_event = token_index
    assert second_event == 9
    assert layer._buffer_valid_count() == 1
    assert layer.retained_tail_count == 2


@pytest.mark.parametrize(
    ("mode", "capacity"),
    [
        ("fixed_cc1", 5),
        ("fixed_cc2", 5),
        ("fixed_cc4", 5),
        ("fixed_cc5", 5),
        ("fixed_cut4_in5", 5),
        ("fixed_cc8", 8),
    ],
)
def test_storage_charges_capacity_not_occupancy(mode: str, capacity: int) -> None:
    cache = _cache(mode)
    _prefill(cache)
    layer = _layer(cache)
    bytes_per_record = 2 * 8 * 2 + 2 * 8 * 2 + 2 * 4
    expected_replay = capacity * bytes_per_record + 4
    expected_ema = 2 * 8 * 4
    expected_total = _plan().resident_bytes + expected_ema + expected_replay
    summary = cache.storage_summary()

    assert layer.replay_capacity == capacity
    assert layer._buffer_valid_count() == 0
    assert layer.replay_occupied_bytes() == 4
    assert layer.replay_capacity_bytes() == expected_replay
    assert layer.logical_replay_capacity_bytes() == expected_replay
    assert summary["logical_replay_capacity_bytes"] == expected_replay
    assert summary["replay_capacity_bytes"] == expected_replay
    assert summary["logical_resident_capacity_bytes"] == expected_total
    assert summary["resident_bytes_including_statelease"] == expected_total
    assert summary["capacity_fully_allocated"] is True
    assert summary["off_budget"] is (mode == "fixed_cc8")


def _production_plan() -> ExactBudgetRowPlan:
    shapes = tuple((layer_index, 16, 128) for layer_index in range(18))
    locations = (
        RowLocation(layer_index, head_index, row_index)
        for layer_index in range(18)
        for head_index in range(16)
        for row_index in range(128)
    )
    promoted = tuple(islice(locations, 1_976))
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
        high_precision_rows=promoted,
        score_shapes=shapes,
    )


def _production_geometry_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        layer_types=["linear_attention"] * 18,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [32, 32, 32, 32],
        },
    )


def test_production_geometry_reports_frozen_equal_and_off_budget_totals() -> None:
    config = _production_geometry_config()
    plan = _production_plan()

    equal = fixed_cc1(config, plan=plan).storage_summary()
    off_budget = fixed_cc8(config, plan=plan).storage_summary()

    assert equal["logical_checkpoint_bytes"] == 2_564_096
    assert equal["logical_query_ema_bytes"] == 147_456
    assert equal["logical_replay_capacity_bytes"] == 743_112
    assert equal["logical_resident_capacity_bytes"] == 3_454_664
    assert equal["equal_allocation"] is True
    assert equal["off_budget"] is False
    assert off_budget["logical_replay_capacity_bytes"] == 1_188_936
    assert off_budget["logical_resident_capacity_bytes"] == 3_900_488
    assert off_budget["equal_allocation"] is False
    assert off_budget["off_budget"] is True


def test_multitoken_cached_chunk_forces_checkpoint_and_clears_replay() -> None:
    cache = _cache("fixed_cc5")
    _prefill(cache)
    layer = _layer(cache)
    for token_index in range(3):
        _commit(cache, seed=400 + token_index)
    assert layer._buffer_valid_count() == 3
    previous_checkpoint = layer.packed_checkpoint

    _commit(cache, seed=410, tokens=2)

    assert layer._buffer_valid_count() == 0
    assert layer.packed_checkpoint is not previous_checkpoint
    assert layer.last_action == "fixed_cc5_checkpoint_chunk"
    assert layer.last_reason == "multi_token"
    assert layer.forced_checkpoint_count == 2
    assert layer.scheduled_checkpoint_count == 0


def test_scheduled_pack_failure_rolls_back_all_persistent_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache("fixed_cc2", record_evidence=True)
    _prefill(cache)
    _commit(cache, seed=501)
    layer = _layer(cache)
    packed = layer.packed_checkpoint
    assert packed is not None
    assert layer.normalized_key_buffer is not None
    assert layer.update_buffer is not None
    assert layer.log_decay_buffer is not None
    assert layer.valid_count is not None
    snapshot = {
        "packed": packed,
        "ema": layer.query_energy_ema.clone(),  # type: ignore[union-attr]
        "key": layer.normalized_key_buffer.clone(),
        "update": layer.update_buffer.clone(),
        "decay": layer.log_decay_buffer.clone(),
        "count": layer.valid_count.clone(),
        "updates": layer._update_count,
        "scheduled": layer.scheduled_checkpoint_count,
        "evidence": tuple(cache.update_evidence),
    }
    final_state = _stage(cache, tokens=1, seed=502)

    def fail_pack(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("injected pack failure")

    monkeypatch.setattr(layer, "_pack_checkpoint", fail_pack)
    with pytest.raises(RuntimeError, match="injected pack failure"):
        cache.update_recurrent_state(final_state, layer_idx=0)

    assert layer._pending_statelease_observation is None
    assert layer.packed_checkpoint is snapshot["packed"]
    assert torch.equal(layer.query_energy_ema, snapshot["ema"])
    assert torch.equal(layer.normalized_key_buffer, snapshot["key"])
    assert torch.equal(layer.update_buffer, snapshot["update"])
    assert torch.equal(layer.log_decay_buffer, snapshot["decay"])
    assert torch.equal(layer.valid_count, snapshot["count"])
    assert layer._update_count == snapshot["updates"]
    assert layer.scheduled_checkpoint_count == snapshot["scheduled"]
    assert tuple(cache.update_evidence) == snapshot["evidence"]


def test_transfer_reset_identity_reorder_and_pending_fail_closed() -> None:
    cache = _cache("fixed_cc4", record_evidence=True)
    _prefill(cache)
    _commit(cache, seed=601)
    layer = _layer(cache)

    cache.reorder_cache(torch.tensor([0], dtype=torch.long))
    with pytest.raises(ValueError, match="identity"):
        cache.reorder_cache(torch.tensor([1], dtype=torch.long))

    final_state = _stage(cache, tokens=1, seed=602)
    del final_state
    with pytest.raises(ValueError, match="identity"):
        cache.reorder_cache(torch.tensor([0, 0], dtype=torch.long))
    assert layer._pending_statelease_observation is None

    final_state = _stage(cache, tokens=1, seed=603)
    del final_state
    with pytest.raises(RuntimeError, match="pending"):
        layer.offload()
    assert layer._pending_statelease_observation is None

    layer.offload()
    assert layer.packed_checkpoint is not None
    assert layer.packed_checkpoint.low_payload.device.type == "cpu"
    layer.prefetch()

    cache.reset()
    diagnostics = layer.statelease_diagnostics()
    assert diagnostics["state_updates"] == 0
    assert diagnostics["scheduled_checkpoint_count"] == 0
    assert diagnostics["forced_checkpoint_count"] == 0
    assert diagnostics["replay_append_count"] == 0
    assert diagnostics["retained_tail_count"] == 0
    assert diagnostics["replay_valid_count"] == 0
    assert cache.update_evidence == []


def test_persistent_tensors_contain_no_hidden_fp32_state_mirror() -> None:
    cache = _cache("fixed_cc5")
    _prefill(cache)
    _commit(cache, seed=701)
    layer = _layer(cache)
    expected_state_shape = (1, 2, 8, 8)

    owned_tensors = [value for value in vars(layer).values() if isinstance(value, torch.Tensor)]
    assert not any(
        tuple(tensor.shape) == expected_state_shape and tensor.dtype == torch.float32
        for tensor in owned_tensors
    )
    assert layer.normalized_key_buffer is not None
    assert layer.normalized_key_buffer.dtype == torch.bfloat16
    assert layer.update_buffer is not None
    assert layer.update_buffer.dtype == torch.bfloat16
    assert layer.log_decay_buffer is not None
    assert layer.log_decay_buffer.dtype == torch.float32
    assert layer.query_energy_ema is not None
    assert layer.query_energy_ema.dtype == torch.float32
    assert layer.valid_count is not None
    assert layer.valid_count.dtype == torch.int32


def test_real_tiny_qwen_observer_handshake_runs_fixed_schedule() -> None:
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    cache = fixed_cc2(config, plan=_plan(), record_evidence=True)

    with torch.inference_mode(), Qwen35StateLeaseObserver(model, caches=[cache]):
        prefill = model(
            torch.randint(0, config.vocab_size, (1, 5)),
            past_key_values=cache,
        )
        first = model(
            torch.randint(0, config.vocab_size, (1, 1)),
            past_key_values=cache,
        )
        second = model(
            torch.randint(0, config.vocab_size, (1, 1)),
            past_key_values=cache,
        )
        chunk = model(
            torch.randint(0, config.vocab_size, (1, 2)),
            past_key_values=cache,
        )

    assert prefill.logits.shape == (1, 5, config.vocab_size)
    assert first.logits.shape == (1, 1, config.vocab_size)
    assert second.logits.shape == (1, 1, config.vocab_size)
    assert chunk.logits.shape == (1, 2, config.vocab_size)
    diagnostics = _layer(cache).statelease_diagnostics()
    assert diagnostics["state_updates"] == 4
    assert diagnostics["scheduled_checkpoint_count"] == 1
    assert diagnostics["forced_checkpoint_count"] == 2
    assert diagnostics["replay_valid_count"] == 0
    assert diagnostics["pending_observation"] is False
    assert [item.action for item in cache.update_evidence] == [
        "fixed_cc2_checkpoint_prefill",
        "fixed_cc2_replay_append",
        "fixed_cc2_checkpoint",
        "fixed_cc2_checkpoint_chunk",
    ]


@pytest.mark.parametrize(
    "observer_order",
    list(permutations(("fixed_replay", "query_ema"))),
)
def test_real_tiny_qwen_observer_composes_with_query_ema_observer(
    observer_order: tuple[str, ...],
) -> None:
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    ).eval()
    fixed_cache = fixed_cc4(config, plan=_plan(), record_evidence=True)
    query_cache = create_qwen35_query_ema_exact_budget_cache(
        config,
        plan=_plan(),
        record_evidence=True,
    )
    observers = {
        "fixed_replay": Qwen35StateLeaseObserver(model, caches=[fixed_cache]),
        "query_ema": Qwen35QueryEnergyObserver(model, caches=[query_cache]),
    }

    with torch.inference_mode(), ExitStack() as stack:
        for name in observer_order:
            stack.enter_context(observers[name])
        fixed_output = model(
            torch.randint(0, config.vocab_size, (1, 3)),
            past_key_values=fixed_cache,
        )
        query_output = model(
            torch.randint(0, config.vocab_size, (1, 3)),
            past_key_values=query_cache,
        )

    assert fixed_output.logits.shape == (1, 3, config.vocab_size)
    assert query_output.logits.shape == (1, 3, config.vocab_size)
    assert [item.action for item in fixed_cache.update_evidence] == ["fixed_cc4_checkpoint_prefill"]
    assert len(query_cache.update_evidence) == 1
    assert _layer(fixed_cache)._pending_statelease_observation is None
