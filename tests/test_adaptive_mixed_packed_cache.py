from __future__ import annotations

import hashlib
import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.packed_cache import (
    AdaptiveMixedPackedLinearAttentionLayer,
    AdaptiveMixedPackedRecurrentStateCache,
)
from recurquant.quantization import QuantizationSpec, quantize_dequantize, quantize_pack
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


def _plan(*, promotions: int, scores: dict[int, torch.Tensor] | None = None) -> ExactBudgetRowPlan:
    if scores is None:
        scores = {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)}
    return select_rows_exact_budget(
        scores,
        target_resident_bytes=_target_for_promotions(scores, promotions=promotions),
        group_size=8,
    )


def _packed_mask(
    cache: AdaptiveMixedPackedRecurrentStateCache,
    layer_index: int = 0,
) -> torch.Tensor:
    layer = cache.layers[layer_index]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    return packed.high_precision_mask().reshape(-1)


def _state_with_difficult_group(group_index: int) -> torch.Tensor:
    state = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    state.reshape(16, 8)[group_index] = torch.tensor(
        [1.0, 0.51, -0.37, 0.23, 0.14, -0.08, 0.03, -0.01]
    )
    return state


def test_adaptive_mask_matches_aligned_per_row_mse_reduction() -> None:
    plan = _plan(promotions=5)
    cache = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=plan)
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(211))
    low_spec = QuantizationSpec(bits=4, group_size=8)
    high_spec = QuantizationSpec(bits=8, group_size=8)
    source = state.to(torch.float32)
    low = quantize_dequantize(state, low_spec).tensor.to(torch.float32)
    high = quantize_dequantize(state, high_spec).tensor.to(torch.float32)
    benefit = (
        (low - source).square().mean(dim=-1) - (high - source).square().mean(dim=-1)
    ).reshape(-1)
    expected_indices = torch.argsort(benefit, descending=True, stable=True)[:5]

    cache.update_recurrent_state(state, layer_idx=0)

    actual = _packed_mask(cache)
    assert actual.sum().item() == 5
    assert torch.equal(actual.nonzero().reshape(-1), expected_indices.sort().values)


def test_adaptive_stored_state_equals_standalone_physical_endpoints_by_computed_mask() -> None:
    plan = _plan(promotions=5)
    cache = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=plan)
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(219))
    layer = cache.layers[0]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)

    restored = cache.update_recurrent_state(state, layer_idx=0)

    packed = layer.packed_states[0]
    assert packed is not None
    computed_mask = packed.high_precision_mask().reshape(1, 2, 8, 1)
    standalone_int4 = quantize_pack(state, layer.low_spec).dequantize()
    standalone_int8 = quantize_pack(state, layer.high_spec).dequantize()
    physical_benefit = (
        (standalone_int4 - state).square().mean(dim=-1)
        - (standalone_int8 - state).square().mean(dim=-1)
    ).reshape(-1)
    selected = torch.argsort(physical_benefit, descending=True, stable=True)[:5]
    independent_mask = torch.zeros(16, dtype=torch.bool)
    independent_mask[selected] = True
    independent_mask = independent_mask.reshape(1, 2, 8, 1)
    expected = torch.where(independent_mask, standalone_int8, standalone_int4)

    assert computed_mask.sum().item() == 5
    assert torch.equal(computed_mask, independent_mask)
    assert torch.equal(restored, expected)
    assert torch.equal(packed.dequantize(), expected)
    assert torch.equal(layer.recurrent_states[0], expected)


def test_equal_scores_use_stable_flattened_row_order_and_repeat_exactly() -> None:
    plan = _plan(promotions=3)
    first = create_qwen35_adaptive_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )
    second = create_qwen35_adaptive_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )
    state = torch.zeros((1, 2, 8, 8))

    first.update_recurrent_state(state, layer_idx=0)
    second.update_recurrent_state(state, layer_idx=0)

    expected = torch.zeros(16, dtype=torch.bool)
    expected[:3] = True
    assert torch.equal(_packed_mask(first), expected)
    assert torch.equal(_packed_mask(second), expected)
    assert first.update_evidence[0].high_precision_mask_sha256 == (
        second.update_evidence[0].high_precision_mask_sha256
    )


@pytest.mark.parametrize("promotions", [0, 16])
def test_zero_and_full_layer_quotas_preserve_exact_counts_and_bytes(promotions: int) -> None:
    plan = _plan(promotions=promotions)
    cache = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=plan)
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(223))

    cache.update_recurrent_state(state, layer_idx=0)

    assert _packed_mask(cache).sum().item() == promotions
    assert cache.storage_summary()["high_precision_groups"] == promotions
    assert cache.storage_summary()["resident_bytes"] == plan.resident_bytes


def test_repeated_updates_can_change_rows_without_changing_quota_or_bytes() -> None:
    plan = _plan(promotions=1)
    cache = create_qwen35_adaptive_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )

    cache.update_recurrent_state(_state_with_difficult_group(2), layer_idx=0)
    first_mask = _packed_mask(cache).clone()
    first_bytes = cache.storage_summary()["resident_bytes"]
    cache.update_recurrent_state(_state_with_difficult_group(11), layer_idx=0)
    second_mask = _packed_mask(cache).clone()

    assert first_mask.nonzero().reshape(-1).tolist() == [2]
    assert second_mask.nonzero().reshape(-1).tolist() == [11]
    assert not torch.equal(first_mask, second_mask)
    assert first_bytes == cache.storage_summary()["resident_bytes"] == plan.resident_bytes
    assert [item.high_precision_groups for item in cache.update_evidence] == [1, 1]
    assert (
        cache.update_evidence[0].high_precision_mask_sha256
        != cache.update_evidence[1].high_precision_mask_sha256
    )


def test_each_layer_keeps_its_plan_quota_and_global_resident_byte_count() -> None:
    scores = {
        0: torch.tensor([100.0, 99.0, 98.0] + [-100.0] * 13).reshape(2, 8),
        1: torch.tensor([97.0, 96.0, 95.0, 94.0] + [-200.0] * 12).reshape(2, 8),
    }
    plan = _plan(promotions=7, scores=scores)
    assert [len(plan.groups_for_layer(index)) for index in (0, 1)] == [3, 4]
    cache = create_qwen35_adaptive_exact_budget_cache(
        tiny_config(["linear_attention", "linear_attention"]),
        plan=plan,
    )
    generator = torch.Generator().manual_seed(227)

    cache.update_recurrent_state(torch.randn((1, 2, 8, 8), generator=generator), layer_idx=0)
    cache.update_recurrent_state(torch.randn((1, 2, 8, 8), generator=generator), layer_idx=1)

    assert [_packed_mask(cache, index).sum().item() for index in (0, 1)] == [3, 4]
    summary = cache.storage_summary()
    assert summary["high_precision_groups"] == plan.promoted_group_count
    assert summary["resident_bytes"] == plan.resident_bytes


def test_batch_size_greater_than_one_is_rejected_before_storage() -> None:
    cache = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=_plan(promotions=3))

    with pytest.raises(ValueError, match="requires batch size 1; got 2"):
        cache.update_recurrent_state(torch.zeros((2, 2, 8, 8)), layer_idx=0)

    layer = cache.layers[0]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)
    assert layer.packed_states[0] is None


def test_adaptive_storage_is_inference_only_and_retains_no_autograd_graph() -> None:
    cache = create_qwen35_adaptive_exact_budget_cache(tiny_config(), plan=_plan(promotions=3))
    state = torch.randn((1, 2, 8, 8), requires_grad=True)

    with pytest.raises(RuntimeError, match="inference-only"):
        cache.update_recurrent_state(state, layer_idx=0)

    with torch.no_grad():
        restored = cache.update_recurrent_state(state, layer_idx=0)

    layer = cache.layers[0]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    assert not restored.requires_grad
    assert all(
        not tensor.requires_grad
        for tensor in (
            packed.low_payload,
            packed.high_payload,
            packed.scales,
            packed.precision_mask,
        )
    )


def test_evidence_records_selection_method_current_mask_hash_and_count() -> None:
    plan = _plan(promotions=4)
    cache = create_qwen35_adaptive_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )
    cache.update_recurrent_state(
        torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(229)),
        layer_idx=0,
    )
    layer = cache.layers[0]
    assert isinstance(layer, AdaptiveMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    expected_hash = hashlib.sha256(bytes(packed.precision_mask.detach().cpu().tolist())).hexdigest()

    evidence = cache.update_evidence[0]
    assert evidence.selection_method == "instantaneous_aligned_mse_reduction"
    assert evidence.high_precision_groups == 4
    assert evidence.high_precision_mask_sha256 == expected_hash
    assert evidence.resident_bytes == plan.resident_bytes
    assert evidence.evidence_dict()["high_precision_mask_sha256"] == expected_hash


def test_adaptive_factory_runs_tiny_qwen_prefill_and_decode() -> None:
    torch.manual_seed(233)
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_adaptive_exact_budget_cache(model, plan=_plan(promotions=5))

    with torch.inference_mode():
        model(torch.randint(0, model.config.vocab_size, (1, 5)), past_key_values=cache)
        output = model(torch.randint(0, model.config.vocab_size, (1, 1)), past_key_values=cache)

    assert output.logits.shape == (1, 1, model.config.vocab_size)
    assert _packed_mask(cache).sum().item() == 5
