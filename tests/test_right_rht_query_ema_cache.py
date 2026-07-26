from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant import (
    Qwen35QueryEnergyObserver,
    RightRhtQueryEmaMixedPackedLinearAttentionLayer,
    RightRhtQueryEmaMixedPackedRecurrentStateCache,
    create_qwen35_right_rht_query_ema_exact_budget_cache,
)
from recurquant.quantization import QuantizationSpec
from recurquant.row_policy import ExactBudgetRowPlan, select_rows_exact_budget
from tests.test_transformers_cache import tiny_config


def _target_for_promotions(promotions: int) -> int:
    total_groups = 16
    group_size = 8
    low_group_bytes = math.ceil(4 * group_size / 8) + 2
    high_increment = math.ceil(8 * group_size / 8) - math.ceil(4 * group_size / 8)
    return (
        total_groups * low_group_bytes
        + math.ceil(total_groups / 8)
        + promotions * high_increment
    )


def _plan(promotions: int = 3) -> ExactBudgetRowPlan:
    return select_rows_exact_budget(
        {0: torch.arange(16, dtype=torch.float32).reshape(2, 8)},
        target_resident_bytes=_target_for_promotions(promotions),
        group_size=8,
    )


def _query() -> torch.Tensor:
    query = torch.zeros((1, 2, 2, 8), dtype=torch.float32)
    query[:, :, :, 3] = 1
    return query


def test_right_rht_query_cache_realizes_same_exact_byte_plan() -> None:
    plan = _plan()
    cache = create_qwen35_right_rht_query_ema_exact_budget_cache(
        tiny_config(),
        plan=plan,
        record_evidence=True,
    )
    cache.stage_query_observation(0, _query())

    cache.update_recurrent_state(
        torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(121)),
        layer_idx=0,
    )

    assert isinstance(cache, RightRhtQueryEmaMixedPackedRecurrentStateCache)
    layer = cache.layers[0]
    assert isinstance(layer, RightRhtQueryEmaMixedPackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    assert packed.right_rht_layer_index == 0
    assert packed.right_rht_expected_heads == 2
    summary = cache.storage_summary()
    assert summary["resident_bytes"] == plan.resident_bytes
    assert summary["selector_auxiliary_bytes"] == 2 * 8 * 4
    assert summary["resident_bytes_including_selector"] == plan.resident_bytes + 64
    assert cache.high_precision_group_count() == plan.promoted_group_count

    diagnostics = cache.query_ema_diagnostics()
    assert diagnostics[0]["selection_method"] == (
        "right_rht_query_ema32_weighted_mse_target_fisher_quota"
    )
    assert diagnostics[0]["state_codec"] == "right_rht_sha256_signs_v1"
    assert diagnostics[0]["state_codec_seed"] == 2339
    assert diagnostics[0]["state_codec_persistent_tensor_bytes"] == 0


def test_right_rht_query_cache_rejects_confirmation_two() -> None:
    with pytest.raises(ValueError, match="does not support Confirmation-2"):
        RightRhtQueryEmaMixedPackedRecurrentStateCache(
            tiny_config(),
            plan=_plan(),
            confirmation_two=True,
        )


def test_right_rht_selector_benefit_matches_decoded_endpoint_error() -> None:
    cache = create_qwen35_right_rht_query_ema_exact_budget_cache(
        tiny_config(),
        plan=_plan(),
    )
    layer = cache.layers[0]
    assert isinstance(layer, RightRhtQueryEmaMixedPackedLinearAttentionLayer)
    state = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(127))
    low_spec = QuantizationSpec(bits=4, group_size=8, flatten_last_dims=2)
    high_spec = QuantizationSpec(bits=8, group_size=8, flatten_last_dims=2)

    transformed_benefit = layer._aligned_mse_benefit(
        state,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    low = layer._quantized_endpoint(state, low_spec)
    high = layer._quantized_endpoint(state, high_spec)
    decoded_benefit = (low - state).square().mean(dim=-1) - (
        high - state
    ).square().mean(dim=-1)

    torch.testing.assert_close(transformed_benefit, decoded_benefit, rtol=2e-5, atol=2e-7)


def test_right_rht_query_cache_runs_tiny_qwen_prefill_and_decode() -> None:
    torch.manual_seed(131)
    model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()
    cache = create_qwen35_right_rht_query_ema_exact_budget_cache(
        model,
        plan=_plan(),
    )
    prompt = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode(), Qwen35QueryEnergyObserver(model, caches=[cache]):
        prefill = model(prompt, past_key_values=cache, use_cache=True)
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode = model(next_token, past_key_values=cache, use_cache=True)

    assert decode.logits.shape == (1, 1, model.config.vocab_size)
    diagnostics = cache.query_ema_diagnostics()
    assert diagnostics[0]["state_updates"] == 2
    assert diagnostics[0]["observations_committed"] == 2
    assert diagnostics[0]["tokens_observed"] == 5
