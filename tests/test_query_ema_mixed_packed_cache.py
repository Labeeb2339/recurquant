from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant import (
    QueryEmaMixedPackedLinearAttentionLayer,
    QueryEmaMixedPackedRecurrentStateCache,
    create_qwen35_query_ema_exact_budget_cache,
)
from recurquant.mixed_quantization import PackedMixedQuantizedTensor
from recurquant.query_energy import Qwen35QueryEnergyObserver
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget
from tests.test_transformers_cache import tiny_config


def _target_for_promotions(promotions: int) -> int:
    total_groups = 16
    group_size = 8
    low_group_bytes = math.ceil(4 * group_size / 8) + 2
    high_increment = math.ceil(8 * group_size / 8) - math.ceil(4 * group_size / 8)
    return total_groups * low_group_bytes + math.ceil(total_groups / 8) + (
        promotions * high_increment
    )


def _plan(promotions: int) -> ExactBudgetRowPlan:
    scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    return select_rows_exact_budget(
        scores,
        target_resident_bytes=_target_for_promotions(promotions),
        group_size=8,
    )


def _cache(
    promotions: int = 1,
    *,
    record_evidence: bool = False,
    confirmation_two: bool = False,
) -> QueryEmaMixedPackedRecurrentStateCache:
    return create_qwen35_query_ema_exact_budget_cache(
        tiny_config(),
        plan=_plan(promotions),
        record_evidence=record_evidence,
        confirmation_two=confirmation_two,
    )


def _layer(
    cache: QueryEmaMixedPackedRecurrentStateCache,
) -> QueryEmaMixedPackedLinearAttentionLayer:
    layer = cache.layers[0]
    assert isinstance(layer, QueryEmaMixedPackedLinearAttentionLayer)
    return layer


def _query(*active_rows: int, tokens: int = 1, scale: float = 1.0) -> torch.Tensor:
    query = torch.zeros((1, tokens, 2, 8), dtype=torch.float32)
    for token in range(tokens):
        for row in active_rows:
            query.reshape(1, tokens, 16)[0, token, row] = scale
    return query


def _difficult_state() -> torch.Tensor:
    pattern = torch.tensor([1.0, 0.51, -0.37, 0.23, 0.14, -0.08, 0.03, -0.01])
    state = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    state.reshape(16, 8)[0] = pattern * 10.0
    state.reshape(16, 8)[1] = pattern
    return state


def _packed_mask(cache: QueryEmaMixedPackedRecurrentStateCache) -> torch.Tensor:
    packed = _layer(cache).packed_states[0]
    assert packed is not None
    return packed.high_precision_mask().reshape(-1)


def test_query_energy_can_override_an_irrelevant_high_mse_row() -> None:
    cache = _cache(promotions=1)
    cache.stage_query_observation(0, _query(1, tokens=256))

    cache.update_recurrent_state(_difficult_state(), layer_idx=0)

    assert _packed_mask(cache).nonzero().reshape(-1).tolist() == [1]


def test_first_update_starts_from_frozen_uniform_ema() -> None:
    cache = _cache(promotions=1)
    cache.stage_query_observation(0, _query(0))
    cache.update_recurrent_state(_difficult_state(), layer_idx=0)
    layer = _layer(cache)
    assert layer.query_energy_ema is not None

    expected = torch.full((2, 8), layer.query_ema_decay / 8, dtype=torch.float32)
    expected[0, 0] += (1.0 - layer.query_ema_decay) / (1.0 + layer.query_l2norm_eps)
    assert torch.allclose(layer.query_energy_ema, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("promotions", [0, 5, 16])
def test_fixed_quota_packed_bytes_and_selector_auxiliary_bytes(promotions: int) -> None:
    plan = _plan(promotions)
    cache = create_qwen35_query_ema_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )
    cache.stage_query_observation(0, _query(0, 3, tokens=2))

    cache.update_recurrent_state(torch.zeros((1, 2, 8, 8)), layer_idx=0)

    summary = cache.storage_summary()
    assert _packed_mask(cache).sum().item() == promotions
    assert summary["resident_bytes"] == plan.resident_bytes
    assert summary["selector_auxiliary_bytes"] == 2 * 8 * 4
    assert summary["resident_bytes_including_selector"] == plan.resident_bytes + 64
    assert cache.update_evidence[0].selection_method == (
        "query_ema32_weighted_aligned_mse_reduction"
    )


def test_chunk_ema_matches_sequential_one_token_updates() -> None:
    generator = torch.Generator().manual_seed(401)
    queries = torch.randn((1, 7, 2, 8), generator=generator)
    state = torch.randn((1, 2, 8, 8), generator=generator)
    chunked = _cache(promotions=3)
    sequential = _cache(promotions=3)

    chunked.stage_query_observation(0, queries)
    chunked.update_recurrent_state(state, layer_idx=0)
    for token in range(queries.shape[1]):
        sequential.stage_query_observation(0, queries[:, token : token + 1])
        sequential.update_recurrent_state(state, layer_idx=0)

    chunked_ema = _layer(chunked).query_energy_ema
    sequential_ema = _layer(sequential).query_energy_ema
    assert chunked_ema is not None
    assert sequential_ema is not None
    layer = _layer(chunked)
    assert torch.allclose(
        chunked_ema,
        sequential_ema,
        atol=layer.query_ema_chunk_atol,
        rtol=layer.query_ema_chunk_rtol,
    )


def test_query_scaling_preserves_row_selection() -> None:
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(409))
    first = _cache(promotions=4)
    second = _cache(promotions=4)
    query = torch.randn((1, 3, 2, 8), generator=torch.Generator().manual_seed(419))

    first.stage_query_observation(0, query)
    second.stage_query_observation(0, query * 32.0)
    first.update_recurrent_state(state, layer_idx=0)
    second.update_recurrent_state(state, layer_idx=0)

    assert torch.equal(_packed_mask(first), _packed_mask(second))


def test_query_ema_confirmation_two_requires_two_consecutive_raw_hits() -> None:
    cache = _cache(promotions=1, confirmation_two=True)
    for row, expected in zip((0, 1, 1), (0, 0, 1), strict=True):
        cache.stage_query_observation(0, _query(row, tokens=256))
        cache.update_recurrent_state(_difficult_state(), layer_idx=0)
        assert _packed_mask(cache).nonzero().flatten().tolist() == [expected]

    diagnostics = cache.query_ema_diagnostics()[0]
    assert _packed_mask(cache).nonzero().flatten().tolist() == [1]
    assert diagnostics["confirmation_two"] is True
    assert diagnostics["observations_staged"] == 3
    assert diagnostics["observations_committed"] == 3
    assert diagnostics["mask_transition_count"] == 2
    assert diagnostics["raw_xor_churn_total"] == 2
    assert diagnostics["committed_xor_churn_total"] == 2
    assert diagnostics["admissions_total"] == 1
    assert diagnostics["dwell_total"] == 1
    assert diagnostics["previous_raw_mask_bytes"] == 2
    assert diagnostics["selector_auxiliary_bytes"] == 66
    assert isinstance(diagnostics["raw_mask_sha256"], str)
    assert isinstance(diagnostics["committed_mask_sha256"], str)

    cache.reset()
    reset = cache.query_ema_diagnostics()[0]
    assert reset["observations_staged"] == 0
    assert reset["observations_committed"] == 0
    assert reset["mask_transition_count"] == 0
    assert reset["raw_xor_churn_total"] == 0
    assert reset["committed_xor_churn_total"] == 0
    assert reset["previous_raw_mask_bytes"] == 0
    assert reset["last_raw_cutoff_score"] is None
    assert reset["last_committed_normalized_gap"] is None


def test_missing_and_duplicate_query_observations_fail_closed() -> None:
    cache = _cache(promotions=2)
    state = torch.zeros((1, 2, 8, 8))

    with pytest.raises(RuntimeError, match="no staged query observation"):
        cache.update_recurrent_state(state, layer_idx=0)
    assert _layer(cache).packed_states[0] is None

    cache.stage_query_observation(0, _query(0))
    with pytest.raises(RuntimeError, match="duplicate query observation"):
        cache.stage_query_observation(0, _query(1))
    with pytest.raises(RuntimeError, match="no staged query observation"):
        cache.update_recurrent_state(state, layer_idx=0)
    assert _layer(cache).query_energy_ema is None


@pytest.mark.parametrize(
    ("query", "error_type", "message"),
    [
        (torch.zeros((2, 1, 2, 8)), ValueError, "must have shape"),
        (torch.zeros((1, 0, 2, 8)), ValueError, "must have shape"),
        (torch.zeros((1, 1, 1, 8)), ValueError, "must have shape"),
        (torch.zeros((1, 1, 2, 7)), ValueError, "must have shape"),
        (torch.zeros((1, 1, 2, 8), dtype=torch.int64), TypeError, "floating point"),
        (torch.full((1, 1, 2, 8), float("nan")), ValueError, "must be finite"),
        (torch.zeros((1, 1, 2, 8), device="meta"), ValueError, "materialized"),
    ],
)
def test_invalid_query_observations_are_rejected(
    query: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    cache = _cache()

    with pytest.raises(error_type, match=message):
        cache.stage_query_observation(0, query)

    assert _layer(cache)._pending_query_observation is None


def test_query_observation_must_share_the_persistent_ema_device() -> None:
    layer = _layer(_cache())
    layer.query_energy_ema = torch.empty((2, 8), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="query observation and query EMA"):
        layer.stage_query_observation(_query(0))

    assert layer._pending_query_observation is None


@pytest.mark.parametrize("epsilon", [0.0, 1e-5, float("nan"), float("inf")])
def test_frozen_normalization_epsilon_cannot_be_tuned(epsilon: float) -> None:
    with pytest.raises(ValueError, match="frozen at 1e-6"):
        _cache().stage_query_observation(0, _query(0), l2norm_eps=epsilon)


def test_stale_observation_and_failed_pack_do_not_commit_ema() -> None:
    stale = _cache()
    stale.stage_query_observation(0, _query(0))
    stale_layer = _layer(stale)
    stale_layer._update_count += 1
    with pytest.raises(RuntimeError, match="stale query observation"):
        stale.update_recurrent_state(torch.zeros((1, 2, 8, 8)), layer_idx=0)
    assert stale_layer.query_energy_ema is None
    assert stale_layer._pending_query_observation is None

    tracked = _cache()
    tracked.stage_query_observation(0, _query(0))
    with pytest.raises(RuntimeError, match="inference-only"):
        tracked.update_recurrent_state(
            torch.zeros((1, 2, 8, 8), requires_grad=True),
            layer_idx=0,
        )
    assert _layer(tracked).query_energy_ema is None
    assert _layer(tracked)._pending_query_observation is None


def test_reset_offload_prefetch_and_diagnostics_handle_fp32_ema() -> None:
    cache = _cache(promotions=2)
    cache.stage_query_observation(0, _query(0, 1, tokens=3))
    cache.update_recurrent_state(_difficult_state(), layer_idx=0)
    layer = _layer(cache)
    assert layer.query_energy_ema is not None
    assert layer.query_energy_ema.dtype == torch.float32
    diagnostics = cache.query_ema_diagnostics()[0]
    assert diagnostics["observations_committed"] == 1
    assert diagnostics["tokens_observed"] == 3
    assert diagnostics["selector_auxiliary_bytes"] == 64

    layer.offload()
    assert layer.query_energy_ema.device.type == "cpu"
    layer.prefetch()
    assert layer.query_energy_ema.device == torch.device("cpu")

    cache.reset()
    assert torch.equal(
        layer.query_energy_ema,
        torch.full((2, 8), 1.0 / 8, dtype=torch.float32),
    )
    assert cache.query_ema_diagnostics()[0]["observations_committed"] == 0
    assert cache.query_ema_diagnostics()[0]["state_updates"] == 0

    cache.stage_query_observation(0, _query(2))
    cache.update_recurrent_state(_difficult_state(), layer_idx=0)
    after_reset = cache.query_ema_diagnostics()[0]
    assert after_reset["observations_committed"] == 1
    assert after_reset["state_updates"] == 1
    assert after_reset["last_mask_overlap"] is None
    assert after_reset["last_mask_churn"] is None


def test_pending_observation_is_discarded_if_transfer_is_attempted() -> None:
    layer = _layer(_cache())
    layer.stage_query_observation(_query(0))

    with pytest.raises(RuntimeError, match="cannot offload with a pending"):
        layer.offload()

    assert layer._pending_query_observation is None
    layer.discard_pending_query_observation()


@pytest.mark.parametrize("confirmation_two", [False, True])
def test_failed_materialization_rolls_back_packed_state_ema_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    confirmation_two: bool,
) -> None:
    cache = _cache(promotions=2, confirmation_two=confirmation_two)
    layer = _layer(cache)
    cache.stage_query_observation(0, _query(0, 1, tokens=2))
    cache.update_recurrent_state(_difficult_state(), layer_idx=0)
    previous_packed = layer.packed_states[0]
    assert previous_packed is not None
    assert layer.query_energy_ema is not None
    previous_ema = layer.query_energy_ema.clone()
    previous_diagnostics = layer.query_ema_diagnostics()

    cache.stage_query_observation(0, _query(3))

    def fail_dequantize(self: PackedMixedQuantizedTensor) -> torch.Tensor:
        raise RuntimeError("injected dequantize failure")

    monkeypatch.setattr(PackedMixedQuantizedTensor, "dequantize", fail_dequantize)
    with pytest.raises(RuntimeError, match="injected dequantize failure"):
        cache.update_recurrent_state(_difficult_state() * 2, layer_idx=0)

    assert layer.packed_states[0] is previous_packed
    assert torch.equal(layer.query_energy_ema, previous_ema)
    assert layer.query_ema_diagnostics() == previous_diagnostics
    assert layer._pending_query_observation is None


def test_failed_postvalidation_rolls_back_without_appending_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(promotions=2, record_evidence=True)
    layer = _layer(cache)
    cache.stage_query_observation(0, _query(0, 1, tokens=2))
    cache.update_recurrent_state(_difficult_state(), layer_idx=0)
    previous_packed = layer.packed_states[0]
    assert previous_packed is not None
    assert layer.query_energy_ema is not None
    previous_ema = layer.query_energy_ema.clone()
    previous_diagnostics = layer.query_ema_diagnostics()
    previous_evidence = list(cache.update_evidence)
    previous_update_index = cache._update_index

    cache.stage_query_observation(0, _query(3))
    original_high_precision_mask = PackedMixedQuantizedTensor.high_precision_mask
    mask_calls = 0

    def fail_postvalidation_mask(
        self: PackedMixedQuantizedTensor,
    ) -> torch.Tensor:
        nonlocal mask_calls
        mask_calls += 1
        if mask_calls == 3:
            raise RuntimeError("injected postvalidation failure")
        return original_high_precision_mask(self)

    monkeypatch.setattr(
        PackedMixedQuantizedTensor,
        "high_precision_mask",
        fail_postvalidation_mask,
    )
    with pytest.raises(RuntimeError, match="injected postvalidation failure"):
        cache.update_recurrent_state(_difficult_state() * 2, layer_idx=0)

    assert mask_calls == 3
    assert layer.packed_states[0] is previous_packed
    assert torch.equal(layer.query_energy_ema, previous_ema)
    assert layer.query_ema_diagnostics() == previous_diagnostics
    assert cache.update_evidence == previous_evidence
    assert cache._update_index == previous_update_index
    assert layer._pending_query_observation is None


def test_invalid_state_index_discards_pending_observation() -> None:
    layer = _layer(_cache())
    layer.stage_query_observation(_query(0))

    with pytest.raises(IndexError, match="state_idx 99"):
        layer.update_recurrent_state(_difficult_state(), state_idx=99)

    assert layer._pending_query_observation is None


def test_real_tiny_qwen_observer_runs_prefill_cached_chunk_and_decode() -> None:
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_query_ema_exact_budget_cache(
        model,
        plan=_plan(5),
        record_evidence=True,
    )

    with torch.inference_mode(), Qwen35QueryEnergyObserver(model, caches=[cache]):
        prefill = model(
            torch.randint(0, model.config.vocab_size, (1, 5)),
            past_key_values=cache,
        )
        chunk = model(
            torch.randint(0, model.config.vocab_size, (1, 2)),
            past_key_values=cache,
        )
        decode = model(
            torch.randint(0, model.config.vocab_size, (1, 1)),
            past_key_values=cache,
        )

    assert prefill.logits.shape == (1, 5, model.config.vocab_size)
    assert chunk.logits.shape == (1, 2, model.config.vocab_size)
    assert decode.logits.shape == (1, 1, model.config.vocab_size)
    diagnostics = cache.query_ema_diagnostics()[0]
    assert diagnostics["observations_committed"] == 3
    assert diagnostics["state_updates"] == diagnostics["observations_committed"]
    assert diagnostics["tokens_observed"] == 8
    assert diagnostics["pending_observation"] is False
    assert diagnostics["current_selected_count"] == 5
    assert isinstance(diagnostics["current_mask_sha256"], str)
    assert len(cache.update_evidence) == 3
