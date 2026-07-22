from __future__ import annotations

import math
from types import MethodType

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.mixed_quantization import PackedMixedQuantizedTensor
from recurquant.packed_cache import (
    CoraMixedPackedLinearAttentionLayer,
    CoraMixedPackedRecurrentStateCache,
)
from recurquant.qwen35 import create_qwen35_cora_exact_budget_cache
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget
from recurquant.transition_observer import Qwen35TransitionObserver
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
    promotions: int = 2,
    *,
    confirmation_two: bool = True,
    record_evidence: bool = False,
) -> CoraMixedPackedRecurrentStateCache:
    return create_qwen35_cora_exact_budget_cache(
        tiny_config(),
        plan=_plan(promotions),
        confirmation_two=confirmation_two,
        record_evidence=record_evidence,
    )


def _layer(cache: CoraMixedPackedRecurrentStateCache) -> CoraMixedPackedLinearAttentionLayer:
    layer = cache.layers[0]
    assert isinstance(layer, CoraMixedPackedLinearAttentionLayer)
    return layer


def _transition(
    active_row: int | None = None,
    *,
    tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros((1, tokens, 2, 8), dtype=torch.float32)
    if active_row is not None:
        query.reshape(1, tokens, 16)[:, :, active_row] = 1.0
    key = torch.zeros_like(query)
    log_decay = torch.full((1, tokens, 2), -100.0, dtype=torch.float32)
    beta = torch.zeros((1, tokens, 2), dtype=torch.float32)
    return query, key, log_decay, beta


def _state() -> torch.Tensor:
    pattern = torch.tensor([1.0, 0.51, -0.37, 0.23, 0.14, -0.08, 0.03, -0.01])
    return pattern.repeat(16).reshape(1, 2, 8, 8)


def _committed_mask(cache: CoraMixedPackedRecurrentStateCache) -> torch.Tensor:
    packed = _layer(cache).packed_states[0]
    assert packed is not None
    return packed.high_precision_mask().reshape(-1)


def _raw_mask(cache: CoraMixedPackedRecurrentStateCache) -> torch.Tensor:
    layer = _layer(cache)
    packed = layer.previous_raw_mask_packed
    assert packed is not None
    shifts = torch.arange(8, dtype=torch.int16)
    expanded = torch.bitwise_right_shift(packed.to(torch.int16).unsqueeze(1), shifts)
    return torch.bitwise_and(expanded, 1).reshape(-1)[:16].to(torch.bool)


def _stage(
    cache: CoraMixedPackedRecurrentStateCache,
    transition: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    cache.stage_transition_observation(0, *transition)


def test_diagonal_recurrence_matches_explicit_transition_matrix() -> None:
    cache = _cache(confirmation_two=False)
    layer = _layer(cache)
    generator = torch.Generator().manual_seed(811)
    query = torch.randn((1, 1, 2, 8), generator=generator)
    key = torch.randn((1, 1, 2, 8), generator=generator)
    log_decay = torch.tensor([[[-0.3, -0.7]]], dtype=torch.float32)
    beta = torch.tensor([[[0.2, 0.8]]], dtype=torch.float32)

    _stage(cache, (query, key, log_decay, beta))
    pending = layer._pending_transition_observation
    assert pending is not None

    p0 = torch.full((2, 8), 1.0 / 16, dtype=torch.float64)
    exact = torch.empty_like(p0)
    for head in range(2):
        q = query[0, 0, head].to(torch.float64)
        k = key[0, 0, head].to(torch.float64)
        q_hat = q / torch.sqrt(q.square().sum() + 1e-6)
        k_hat = k / torch.sqrt(k.square().sum() + 1e-6)
        transition = torch.exp(log_decay[0, 0, head].to(torch.float64)) * (
            torch.eye(8, dtype=torch.float64)
            - beta[0, 0, head].to(torch.float64) * torch.outer(k_hat, k_hat)
        )
        gram = transition.T @ torch.diag(p0[head]) @ transition
        exact[head] = q_hat.square() / 8 + torch.diag(gram)
    exact /= exact.sum()

    assert pending.candidate_diagonal.dtype == torch.float32
    assert torch.allclose(
        pending.candidate_diagonal.to(torch.float64),
        exact,
        atol=2e-8,
        rtol=2e-7,
    )


def test_chronological_chunk_matches_one_token_state_writes() -> None:
    generator = torch.Generator().manual_seed(821)
    query = torch.randn((1, 7, 2, 8), generator=generator)
    key = torch.randn((1, 7, 2, 8), generator=generator)
    log_decay = -torch.rand((1, 7, 2), generator=generator)
    beta = torch.rand((1, 7, 2), generator=generator)
    state = torch.randn((1, 2, 8, 8), generator=generator)
    chunked = _cache(4, confirmation_two=False)
    stepped = _cache(4, confirmation_two=False)

    _stage(chunked, (query, key, log_decay, beta))
    chunked.update_recurrent_state(state, layer_idx=0)
    for token in range(7):
        selection = slice(token, token + 1)
        _stage(
            stepped,
            (
                query[:, selection],
                key[:, selection],
                log_decay[:, selection],
                beta[:, selection],
            ),
        )
        stepped.update_recurrent_state(state, layer_idx=0)

    chunked_p = _layer(chunked).observability_diagonal
    stepped_p = _layer(stepped).observability_diagonal
    assert chunked_p is not None and stepped_p is not None
    assert torch.allclose(chunked_p, stepped_p, atol=4e-8, rtol=4e-7)


@pytest.mark.parametrize("promotions", [0, 5, 16])
def test_exact_quota_and_selector_storage(promotions: int) -> None:
    plan = _plan(promotions)
    cache = _cache(promotions, record_evidence=True)
    _stage(cache, _transition(0, tokens=2))
    cache.update_recurrent_state(_state(), layer_idx=0)

    diagnostics = cache.selection_diagnostics()[0]
    summary = cache.storage_summary()
    assert _committed_mask(cache).sum().item() == promotions
    assert diagnostics["quota"] == promotions
    assert diagnostics["current_selected_count"] == promotions
    assert diagnostics["observability_diagonal_bytes"] == 64
    assert diagnostics["previous_raw_mask_bytes"] == 2
    assert diagnostics["selector_auxiliary_bytes"] == 66
    assert summary["resident_bytes"] == plan.resident_bytes
    assert summary["selector_auxiliary_bytes"] == 66
    assert summary["resident_bytes_including_selector"] == plan.resident_bytes + 66
    assert cache.update_evidence[0].selection_method == (
        "causal_observability_confirm2_mse_target_fisher_quota"
    )


def test_confirmation_two_requires_two_consecutive_raw_hits() -> None:
    cache = _cache(1, confirmation_two=True)
    state = _state()

    _stage(cache, _transition(0))
    cache.update_recurrent_state(state, layer_idx=0)
    assert _raw_mask(cache).nonzero().flatten().tolist() == [0]
    assert _committed_mask(cache).nonzero().flatten().tolist() == [0]

    _stage(cache, _transition(1))
    cache.update_recurrent_state(state, layer_idx=0)
    assert _raw_mask(cache).nonzero().flatten().tolist() == [1]
    assert _committed_mask(cache).nonzero().flatten().tolist() == [0]

    _stage(cache, _transition(1))
    cache.update_recurrent_state(state, layer_idx=0)
    assert _raw_mask(cache).nonzero().flatten().tolist() == [1]
    assert _committed_mask(cache).nonzero().flatten().tolist() == [1]
    diagnostics = cache.selection_diagnostics()[0]
    assert diagnostics["mask_transition_count"] == 2
    assert diagnostics["raw_xor_churn_total"] == 2
    assert diagnostics["committed_xor_churn_total"] == 2
    assert diagnostics["admissions_total"] == 1
    assert diagnostics["dwell_total"] == 1
    assert diagnostics["raw_normalized_churn"] == pytest.approx(0.5)
    assert diagnostics["committed_normalized_churn"] == pytest.approx(0.5)


def test_raw_cora_switches_without_confirmation_delay() -> None:
    cache = _cache(1, confirmation_two=False)
    _stage(cache, _transition(0))
    cache.update_recurrent_state(_state(), layer_idx=0)
    _stage(cache, _transition(1))
    cache.update_recurrent_state(_state(), layer_idx=0)
    assert _committed_mask(cache).nonzero().flatten().tolist() == [1]


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        ("query_batch", ValueError, "shape"),
        ("key_width", ValueError, "shape"),
        ("gate_width", ValueError, "shape"),
        ("integer", TypeError, "floating point"),
        ("nonfinite", ValueError, "finite"),
        ("empty", ValueError, "at least one token"),
    ],
)
def test_invalid_transition_observations_fail_closed(
    replacement: str,
    error_type: type[Exception],
    message: str,
) -> None:
    cache = _cache()
    query, key, log_decay, beta = _transition(0)
    if replacement == "query_batch":
        query = query.repeat(2, 1, 1, 1)
    elif replacement == "key_width":
        key = key[..., :7]
    elif replacement == "gate_width":
        beta = beta[..., :1]
    elif replacement == "integer":
        key = key.to(torch.int64)
    elif replacement == "nonfinite":
        log_decay[0, 0, 0] = float("nan")
    elif replacement == "empty":
        query = query[:, :0]
        key = key[:, :0]
        log_decay = log_decay[:, :0]
        beta = beta[:, :0]

    with pytest.raises(error_type, match=message):
        _stage(cache, (query, key, log_decay, beta))
    assert _layer(cache)._pending_transition_observation is None


def test_missing_duplicate_stale_and_invalid_index_fail_closed() -> None:
    cache = _cache()
    layer = _layer(cache)
    with pytest.raises(RuntimeError, match="no staged transition"):
        cache.update_recurrent_state(_state(), layer_idx=0)

    _stage(cache, _transition(0))
    with pytest.raises(RuntimeError, match="duplicate transition"):
        _stage(cache, _transition(1))
    assert layer._pending_transition_observation is None

    _stage(cache, _transition(0))
    layer._update_count += 1
    with pytest.raises(RuntimeError, match="stale transition"):
        cache.update_recurrent_state(_state(), layer_idx=0)
    assert layer._pending_transition_observation is None

    layer._update_count = 0
    _stage(cache, _transition(0))
    with pytest.raises(IndexError, match="state_idx 9"):
        layer.update_recurrent_state(_state(), state_idx=9)
    assert layer._pending_transition_observation is None


def test_failed_pack_rolls_back_every_selector_field_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(2, record_evidence=True)
    layer = _layer(cache)
    _stage(cache, _transition(0, tokens=2))
    cache.update_recurrent_state(_state(), layer_idx=0)
    previous_packed = layer.packed_states[0]
    previous_diagonal = layer.observability_diagonal
    previous_raw = layer.previous_raw_mask_packed
    assert previous_packed is not None
    assert previous_diagonal is not None
    assert previous_raw is not None
    previous_diagnostics = layer.observability_diagnostics()
    previous_evidence = list(cache.update_evidence)
    previous_evidence_index = cache._update_index

    _stage(cache, _transition(1))

    def fail_dequantize(self: PackedMixedQuantizedTensor) -> torch.Tensor:
        raise RuntimeError("injected CORA materialization failure")

    monkeypatch.setattr(PackedMixedQuantizedTensor, "dequantize", fail_dequantize)
    with pytest.raises(RuntimeError, match="injected CORA materialization failure"):
        cache.update_recurrent_state(_state() * 2, layer_idx=0)

    assert layer.packed_states[0] is previous_packed
    assert layer.observability_diagonal is previous_diagonal
    assert layer.previous_raw_mask_packed is previous_raw
    assert layer.observability_diagnostics() == previous_diagnostics
    assert cache.update_evidence == previous_evidence
    assert cache._update_index == previous_evidence_index
    assert layer._pending_transition_observation is None


def test_failed_previous_mask_read_clears_candidate_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(2, record_evidence=True)
    layer = _layer(cache)
    _stage(cache, _transition(0))
    cache.update_recurrent_state(_state(), layer_idx=0)
    previous_packed = layer.packed_states[0]
    previous_diagonal = layer.observability_diagonal
    previous_raw = layer.previous_raw_mask_packed
    previous_diagnostics = layer.observability_diagnostics()
    previous_evidence = list(cache.update_evidence)
    _stage(cache, _transition(1))

    def fail_mask(self: PackedMixedQuantizedTensor) -> torch.Tensor:
        raise RuntimeError("injected previous-mask failure")

    monkeypatch.setattr(PackedMixedQuantizedTensor, "high_precision_mask", fail_mask)
    with pytest.raises(RuntimeError, match="injected previous-mask failure"):
        cache.update_recurrent_state(_state() * 2, layer_idx=0)

    assert layer.packed_states[0] is previous_packed
    assert layer.observability_diagonal is previous_diagonal
    assert layer.previous_raw_mask_packed is previous_raw
    assert layer.observability_diagnostics() == previous_diagnostics
    assert cache.update_evidence == previous_evidence
    assert layer._pending_transition_observation is None


def test_callback_failure_rolls_back_appended_evidence_and_every_state() -> None:
    cache = _cache(2, record_evidence=True)
    layer = _layer(cache)
    _stage(cache, _transition(0))
    cache.update_recurrent_state(_state(), layer_idx=0)
    previous_packed = layer.packed_states[0]
    previous_diagonal = layer.observability_diagonal
    previous_raw = layer.previous_raw_mask_packed
    previous_diagnostics = layer.observability_diagnostics()
    previous_evidence = list(cache.update_evidence)
    previous_evidence_index = cache._update_index
    _stage(cache, _transition(1))

    def append_then_fail(
        owner: CoraMixedPackedRecurrentStateCache,
        layer_index: int,
        state_index: int,
        source: torch.Tensor,
        packed: PackedMixedQuantizedTensor,
        materialized: torch.Tensor,
    ) -> None:
        del layer_index, state_index, source, packed, materialized
        owner.update_evidence.append(previous_evidence[0])
        owner._update_index += 1
        raise RuntimeError("injected evidence callback failure")

    layer.on_update = MethodType(append_then_fail, cache)
    with pytest.raises(RuntimeError, match="injected evidence callback failure"):
        cache.update_recurrent_state(_state() * 2, layer_idx=0)

    assert layer.packed_states[0] is previous_packed
    assert layer.observability_diagonal is previous_diagonal
    assert layer.previous_raw_mask_packed is previous_raw
    assert layer.observability_diagnostics() == previous_diagnostics
    assert cache.update_evidence == previous_evidence
    assert cache._update_index == previous_evidence_index
    assert layer._pending_transition_observation is None


def test_reset_offload_prefetch_and_pending_transfer() -> None:
    cache = _cache(2)
    layer = _layer(cache)
    _stage(cache, _transition(0, tokens=3))
    cache.update_recurrent_state(_state(), layer_idx=0)
    layer.offload()
    assert layer.observability_diagonal is not None
    assert layer.observability_diagonal.device.type == "cpu"
    assert layer.previous_raw_mask_packed is not None
    layer.prefetch()

    cache.reset()
    assert torch.equal(
        layer.observability_diagonal,
        torch.full((2, 8), 1.0 / 16, dtype=torch.float32),
    )
    assert layer.previous_raw_mask_packed is None
    assert cache.selection_diagnostics()[0]["state_updates"] == 0

    _stage(cache, _transition(0))
    with pytest.raises(RuntimeError, match="cannot offload with a pending"):
        layer.offload()
    assert layer._pending_transition_observation is None


def test_real_tiny_qwen_observer_runs_cora_prefill_chunk_and_decode() -> None:
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_cora_exact_budget_cache(
        model,
        plan=_plan(5),
        record_evidence=True,
    )

    with torch.inference_mode(), Qwen35TransitionObserver(model, caches=[cache]):
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
    diagnostics = cache.selection_diagnostics()[0]
    assert diagnostics["observations_staged"] == 3
    assert diagnostics["observations_committed"] == 3
    assert diagnostics["state_updates"] == 3
    assert diagnostics["tokens_observed"] == 8
    assert diagnostics["mask_transition_count"] == 2
    assert diagnostics["pending_observation"] is False
    assert diagnostics["current_selected_count"] == 5
    assert len(cache.update_evidence) == 3
