from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.packed_cache import (
    MixedPackedLinearAttentionLayer,
    MixedPackedRecurrentStateCache,
)
from recurquant.quantization import QuantizationSpec
from recurquant.qwen35 import create_qwen35_exact_budget_cache
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget
from recurquant.transformers_cache import RecurrentStateQDQCache
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
    return total_groups * low_group_bytes + math.ceil(total_groups / 8) + (promotions * increment)


def _tiny_plan(
    *,
    layer_indices: tuple[int, ...] = (0,),
    promotions: int = 5,
    group_size: int = 8,
    heads: int = 2,
    rows: int = 8,
) -> ExactBudgetRowPlan:
    scores = {
        layer_index: torch.arange(heads * rows, dtype=torch.float32).reshape(heads, rows)
        + offset * 100
        for offset, layer_index in enumerate(layer_indices)
    }
    return select_rows_exact_budget(
        scores,
        target_resident_bytes=_target_for_promotions(
            scores,
            promotions=promotions,
            group_size=group_size,
        ),
        group_size=group_size,
    )


def _tiny_model(layer_types: list[str] | None = None) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(layer_types),
        attn_implementation="eager",
    ).eval()


def test_exact_budget_factory_runs_prefill_decode_and_reports_physical_bytes() -> None:
    torch.manual_seed(131)
    model = _tiny_model()
    plan = _tiny_plan(promotions=5)
    cache = create_qwen35_exact_budget_cache(
        model,
        plan=plan,
        record_evidence=True,
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 6))
    next_token = torch.randint(0, model.config.vocab_size, (1, 1))

    with torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True)
        output = model(next_token, past_key_values=cache, use_cache=True)

    assert output.logits.shape == (1, 1, model.config.vocab_size)
    assert isinstance(cache, MixedPackedRecurrentStateCache)
    layer = cache.layers[0]
    assert isinstance(layer, MixedPackedLinearAttentionLayer)
    assert layer.recurrent_states[0].shape == (1, 2, 8, 8)
    assert cache.storage_summary() == {
        "payload_bytes": 84,
        "scale_bytes": 32,
        "mask_bytes": 2,
        "resident_bytes": 118,
        "high_precision_groups": 5,
        "full_precision_equivalent_bytes": 512,
        "largest_materialized_state_bytes": 512,
        "resident_compression_ratio": 512 / 118,
        "physical_reduction_realized": True,
    }
    assert len(cache.update_evidence) == 2
    assert cache.update_evidence[-1].high_precision_groups == 5
    assert cache.update_evidence[-1].resident_bytes == 118


def test_all_int4_row_plan_matches_qdq_cache_state_and_logits_exactly() -> None:
    torch.manual_seed(137)
    model = _tiny_model()
    config = model.config
    plan = _tiny_plan(promotions=0)
    qdq_cache = RecurrentStateQDQCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=8),
    )
    mixed_cache = create_qwen35_exact_budget_cache(model, plan=plan)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    continuation = torch.randint(0, config.vocab_size, (1, 2))

    with torch.inference_mode():
        model(prompt, past_key_values=qdq_cache, use_cache=True)
        qdq_output = model(continuation, past_key_values=qdq_cache, use_cache=True)
        model(prompt, past_key_values=mixed_cache, use_cache=True)
        mixed_output = model(continuation, past_key_values=mixed_cache, use_cache=True)

    assert torch.equal(mixed_output.logits, qdq_output.logits)
    assert torch.equal(
        mixed_cache.layers[0].recurrent_states[0],
        qdq_cache.layers[0].recurrent_states[0],
    )


def test_batch_independent_policy_repeats_exactly_for_each_batch_item() -> None:
    plan = _tiny_plan(promotions=7)
    cache = create_qwen35_exact_budget_cache(tiny_config(), plan=plan)
    state = torch.randn((2, 2, 8, 8), generator=torch.Generator().manual_seed(139))

    cache.update_recurrent_state(state, layer_idx=0)

    layer = cache.layers[0]
    assert isinstance(layer, MixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    unpacked = packed.high_precision_mask().reshape(2, 16)
    assert torch.equal(unpacked[0], unpacked[1])
    assert unpacked[0].sum().item() == 7
    assert cache.storage_summary()["resident_bytes"] == 2 * plan.resident_bytes
    assert cache.storage_summary()["high_precision_groups"] == 14


def test_exact_budget_cache_supports_beam_generation_and_lossless_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(149)
    model = _tiny_model()
    plan = _tiny_plan(promotions=5)
    cache = create_qwen35_exact_budget_cache(model, plan=plan)
    prompt = torch.tensor([[1, 2, 3, 4]])
    reorder_indices: list[torch.Tensor] = []
    original_reorder = MixedPackedLinearAttentionLayer.reorder_cache

    def record_reorder(
        layer: MixedPackedLinearAttentionLayer,
        beam_idx: torch.LongTensor,
    ) -> None:
        reorder_indices.append(beam_idx.detach().cpu().clone())
        original_reorder(layer, beam_idx)

    monkeypatch.setattr(MixedPackedLinearAttentionLayer, "reorder_cache", record_reorder)
    with torch.inference_mode():
        generated = model.generate(
            prompt,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=3,
            do_sample=False,
            num_beams=3,
            pad_token_id=0,
        )

    assert generated.shape == (1, 7)
    assert reorder_indices
    assert all(indices.shape == (3,) for indices in reorder_indices)
    assert cache.storage_summary()["resident_bytes"] == 3 * plan.resident_bytes
    assert cache.storage_summary()["high_precision_groups"] == 15


def test_factory_rejects_missing_and_extra_plan_layers() -> None:
    missing_plan = _tiny_plan(layer_indices=(0,), promotions=0)
    with pytest.raises(ValueError, match=r"missing=\[1\].*extra=\[\]"):
        create_qwen35_exact_budget_cache(
            tiny_config(["linear_attention", "linear_attention", "full_attention"]),
            plan=missing_plan,
        )

    extra_plan = _tiny_plan(layer_indices=(0, 1), promotions=0)
    with pytest.raises(ValueError, match=r"missing=\[\].*extra=\[1\]"):
        create_qwen35_exact_budget_cache(tiny_config(), plan=extra_plan)


def test_factory_rejects_incompatible_plan_and_runtime_geometry() -> None:
    incompatible_scores = {0: torch.ones((1, 16))}
    incompatible_shape = select_rows_exact_budget(
        incompatible_scores,
        target_resident_bytes=_target_for_promotions(
            incompatible_scores,
            promotions=0,
        ),
        group_size=8,
    )
    with pytest.raises(ValueError, match="score geometry"):
        create_qwen35_exact_budget_cache(tiny_config(), plan=incompatible_shape)

    wrong_group_size = _tiny_plan(promotions=0, group_size=4)
    with pytest.raises(ValueError, match="linear_value_head_dim"):
        create_qwen35_exact_budget_cache(tiny_config(), plan=wrong_group_size)

    cache = create_qwen35_exact_budget_cache(tiny_config(), plan=_tiny_plan())
    with pytest.raises(ValueError, match=r"\[batch, 2, 8, 8\]"):
        cache.update_recurrent_state(torch.ones((1, 2, 4, 16)), layer_idx=0)


def test_factory_rejects_global_mask_budget_that_grows_when_split_by_layer() -> None:
    config = tiny_config(["linear_attention", "linear_attention"])
    config.linear_num_value_heads = 1
    config.linear_key_head_dim = 4
    config.linear_value_head_dim = 8
    scores = {0: torch.ones((1, 4)), 1: torch.ones((1, 4))}
    plan = select_rows_exact_budget(
        scores,
        target_resident_bytes=_target_for_promotions(scores, promotions=0),
        group_size=8,
    )

    with pytest.raises(ValueError, match="separate per-layer masks"):
        create_qwen35_exact_budget_cache(config, plan=plan)


def test_qwen_eighteen_recurrent_layer_layout_realizes_exact_target() -> None:
    layer_types = [
        layer_type
        for _ in range(6)
        for layer_type in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
    ]
    linear_indices = tuple(
        index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    )
    config = tiny_config(layer_types)
    config.linear_num_value_heads = 16
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128
    scores = {
        layer_index: torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128) + order * 10_000
        for order, layer_index in enumerate(linear_indices)
    }
    plan = select_rows_exact_budget(scores, target_resident_bytes=2_564_096)
    cache = create_qwen35_exact_budget_cache(config, plan=plan)
    state = torch.zeros((1, 16, 128, 128))

    for layer_index in linear_indices:
        cache.update_recurrent_state(state, layer_idx=layer_index)

    assert len(linear_indices) == 18
    assert plan.promoted_group_count == 1_976
    assert cache.storage_summary() == {
        "payload_bytes": 2_485_760,
        "scale_bytes": 73_728,
        "mask_bytes": 4_608,
        "resident_bytes": 2_564_096,
        "high_precision_groups": 1_976,
        "full_precision_equivalent_bytes": 18_874_368,
        "largest_materialized_state_bytes": 1_048_576,
        "resident_compression_ratio": 18_874_368 / 2_564_096,
        "physical_reduction_realized": True,
    }
