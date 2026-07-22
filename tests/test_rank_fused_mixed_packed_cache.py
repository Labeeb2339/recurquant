from __future__ import annotations

import hashlib
import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant import (
    RankFusedMixedPackedRecurrentStateCache,
    create_qwen35_rank_fused_exact_budget_cache,
)
from recurquant.packed_cache import (
    AdaptiveMixedPackedLinearAttentionLayer,
    RankFusedMixedPackedLinearAttentionLayer,
)
from recurquant.qwen35 import create_qwen35_adaptive_exact_budget_cache
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget
from tests.test_transformers_cache import tiny_config


def _target_for_promotions(
    scores: dict[int, torch.Tensor],
    *,
    promotions: int,
    group_size: int = 8,
    scale_bits: int = 16,
) -> int:
    total_groups = sum(score.numel() for score in scores.values())
    low_group_bytes = math.ceil(4 * group_size / 8) + scale_bits // 8
    increment = math.ceil(8 * group_size / 8) - math.ceil(4 * group_size / 8)
    return total_groups * low_group_bytes + math.ceil(total_groups / 8) + promotions * increment


def _plan(
    *,
    promotions: int,
    scores: dict[int, torch.Tensor] | None = None,
) -> ExactBudgetRowPlan:
    if scores is None:
        scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    return select_rows_exact_budget(
        scores,
        target_resident_bytes=_target_for_promotions(scores, promotions=promotions),
        group_size=8,
    )


def _rank_fused_mask(
    cache: RankFusedMixedPackedRecurrentStateCache,
    layer_index: int = 0,
) -> torch.Tensor:
    layer = cache.layers[layer_index]
    assert isinstance(layer, RankFusedMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    return packed.high_precision_mask().reshape(-1)


def _adaptive_mask(cache: object, layer_index: int = 0) -> torch.Tensor:
    layer = cache.layers[layer_index]  # type: ignore[attr-defined]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    return packed.high_precision_mask().reshape(-1)


def test_zero_weight_exactly_matches_existing_adaptive_selector_and_storage() -> None:
    static_scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    plan = _plan(promotions=5, scores=static_scores)
    adaptive = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=plan)
    rank_fused = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.0,
    )
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(307))

    adaptive_result = adaptive.update_recurrent_state(state, layer_idx=0)
    rank_fused_result = rank_fused.update_recurrent_state(state, layer_idx=0)

    assert torch.equal(_rank_fused_mask(rank_fused), _adaptive_mask(adaptive))
    assert torch.equal(rank_fused_result, adaptive_result)
    adaptive_packed = adaptive.layers[0].packed_states[0]  # type: ignore[attr-defined]
    rank_fused_packed = rank_fused.layers[0].packed_states[0]  # type: ignore[attr-defined]
    assert adaptive_packed is not None
    assert rank_fused_packed is not None
    assert torch.equal(rank_fused_packed.low_payload, adaptive_packed.low_payload)
    assert torch.equal(rank_fused_packed.high_payload, adaptive_packed.high_payload)
    assert torch.equal(rank_fused_packed.scales, adaptive_packed.scales)
    assert torch.equal(rank_fused_packed.precision_mask, adaptive_packed.precision_mask)
    assert rank_fused.storage_summary() == adaptive.storage_summary()


def test_one_weight_exactly_selects_static_plan_rows() -> None:
    static_scores = {
        0: torch.tensor(
            [1.0, 8.0, 3.0, 14.0, 0.0, 5.0, 12.0, 4.0, 11.0, 2.0, 15.0, 7.0,
             9.0, 13.0, 6.0, 10.0]
        ).reshape(2, 8)
    }
    plan = _plan(promotions=5, scores=static_scores)
    cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=1.0,
    )
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(311))

    cache.update_recurrent_state(state, layer_idx=0)

    expected = torch.zeros(16, dtype=torch.bool)
    expected[list(plan.groups_for_layer(0))] = True
    assert torch.equal(_rank_fused_mask(cache), expected)


def test_equal_static_and_dynamic_scores_use_stable_flattened_order() -> None:
    static_scores = {0: torch.zeros((2, 8), dtype=torch.float32)}
    plan = _plan(promotions=3, scores=static_scores)
    first = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )
    second = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )
    state = torch.zeros((1, 2, 8, 8))

    first.update_recurrent_state(state, layer_idx=0)
    second.update_recurrent_state(state, layer_idx=0)

    expected = torch.zeros(16, dtype=torch.bool)
    expected[:3] = True
    assert torch.equal(_rank_fused_mask(first), expected)
    assert torch.equal(_rank_fused_mask(second), expected)


@pytest.mark.parametrize("promotions", [0, 16])
def test_zero_and_full_quotas_keep_exact_counts_and_bytes(promotions: int) -> None:
    static_scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    plan = _plan(promotions=promotions, scores=static_scores)
    cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )

    cache.update_recurrent_state(
        torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(309)),
        layer_idx=0,
    )

    assert _rank_fused_mask(cache).sum().item() == promotions
    assert cache.storage_summary()["high_precision_groups"] == promotions
    assert cache.storage_summary()["resident_bytes"] == plan.resident_bytes


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_static_rank_weight_values_are_rejected(weight: float) -> None:
    scores = {0: torch.zeros((2, 8))}

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        create_qwen35_rank_fused_exact_budget_cache(
            tiny_config(),
            plan=_plan(promotions=3, scores=scores),
            static_scores_by_layer=scores,
            static_rank_weight=weight,
        )


@pytest.mark.parametrize("weight", [True, "0.5"])
def test_non_real_static_rank_weights_are_rejected(weight: object) -> None:
    scores = {0: torch.zeros((2, 8))}

    with pytest.raises(TypeError, match="must be a real number"):
        create_qwen35_rank_fused_exact_budget_cache(
            tiny_config(),
            plan=_plan(promotions=3, scores=scores),
            static_scores_by_layer=scores,
            static_rank_weight=weight,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("static_scores", "error_type", "message"),
    [
        ({1: torch.zeros((2, 8))}, ValueError, "exactly match the row plan"),
        ({0: torch.zeros(16)}, ValueError, "must have shape"),
        ({0: torch.zeros((2, 8), dtype=torch.int64)}, TypeError, "floating point"),
        ({0: torch.full((2, 8), float("nan"))}, ValueError, "must be finite"),
        ({0: torch.zeros((2, 8), device="meta")}, ValueError, "must be materialized"),
        ({0: object()}, TypeError, "must be a tensor"),
    ],
)
def test_invalid_static_score_maps_are_rejected(
    static_scores: dict[int, object],
    error_type: type[Exception],
    message: str,
) -> None:
    plan_scores = {0: torch.zeros((2, 8))}

    with pytest.raises(error_type, match=message):
        create_qwen35_rank_fused_exact_budget_cache(
            tiny_config(),
            plan=_plan(promotions=3, scores=plan_scores),
            static_scores_by_layer=static_scores,  # type: ignore[arg-type]
            static_rank_weight=0.5,
        )


def test_static_scores_must_share_the_recurrent_state_device() -> None:
    static_scores = {0: torch.zeros((2, 8))}
    cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=_plan(promotions=3, scores=static_scores),
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )

    with pytest.raises(ValueError, match="same device"):
        cache.update_recurrent_state(torch.zeros((1, 2, 8, 8), device="meta"), layer_idx=0)


def test_each_layer_preserves_quota_exact_bytes_and_distinct_evidence_method() -> None:
    static_scores = {
        0: torch.tensor([100.0, 99.0, 98.0] + [-100.0] * 13).reshape(2, 8),
        1: torch.tensor([97.0, 96.0, 95.0, 94.0] + [-200.0] * 12).reshape(2, 8),
    }
    plan = _plan(promotions=7, scores=static_scores)
    cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(["linear_attention", "linear_attention"]),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.25,
        record_evidence=True,
    )
    generator = torch.Generator().manual_seed(313)

    cache.update_recurrent_state(torch.randn((1, 2, 8, 8), generator=generator), layer_idx=0)
    cache.update_recurrent_state(torch.randn((1, 2, 8, 8), generator=generator), layer_idx=1)

    assert [_rank_fused_mask(cache, index).sum().item() for index in (0, 1)] == [3, 4]
    summary = cache.storage_summary()
    assert summary["high_precision_groups"] == plan.promoted_group_count == 7
    assert summary["resident_bytes"] == plan.resident_bytes
    assert len(cache.update_evidence) == 2
    assert all(
        evidence.selection_method == "quota_preserving_static_dynamic_rank_fusion"
        for evidence in cache.update_evidence
    )
    for evidence, layer_index in zip(cache.update_evidence, (0, 1), strict=True):
        packed = cache.layers[layer_index].packed_states[0]  # type: ignore[attr-defined]
        assert packed is not None
        expected_hash = hashlib.sha256(
            bytes(packed.precision_mask.detach().cpu().tolist())
        ).hexdigest()
        assert evidence.high_precision_mask_sha256 == expected_hash


def test_batch_one_and_inference_only_contracts_are_preserved() -> None:
    static_scores = {0: torch.zeros((2, 8))}
    plan = _plan(promotions=3, scores=static_scores)
    batch_cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )

    with pytest.raises(ValueError, match="requires batch size 1; got 2"):
        batch_cache.update_recurrent_state(torch.zeros((2, 2, 8, 8)), layer_idx=0)

    grad_cache = create_qwen35_rank_fused_exact_budget_cache(
        tiny_config(),
        plan=plan,
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )
    state = torch.randn((1, 2, 8, 8), requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        grad_cache.update_recurrent_state(state, layer_idx=0)
    with torch.no_grad():
        restored = grad_cache.update_recurrent_state(state, layer_idx=0)
    assert not restored.requires_grad


def test_rank_fused_factory_runs_tiny_qwen_prefill_and_decode() -> None:
    torch.manual_seed(317)
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()
    static_scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    cache = create_qwen35_rank_fused_exact_budget_cache(
        model,
        plan=_plan(promotions=5, scores=static_scores),
        static_scores_by_layer=static_scores,
        static_rank_weight=0.5,
    )

    with torch.inference_mode():
        model(torch.randint(0, model.config.vocab_size, (1, 5)), past_key_values=cache)
        output = model(torch.randint(0, model.config.vocab_size, (1, 1)), past_key_values=cache)

    assert output.logits.shape == (1, 1, model.config.vocab_size)
    assert _rank_fused_mask(cache).sum().item() == 5
