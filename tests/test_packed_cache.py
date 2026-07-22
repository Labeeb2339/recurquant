from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from recurquant.packed_cache import PackedLinearAttentionLayer, PackedRecurrentStateCache
from recurquant.quantization import QuantizationSpec
from recurquant.transformers_cache import RecurrentStateQDQCache
from tests.test_transformers_cache import tiny_config


def test_packed_cache_runs_prefill_and_decode_with_integer_residency() -> None:
    torch.manual_seed(19)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    cache = PackedRecurrentStateCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16),
        record_evidence=True,
    )
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True)
        output = model(next_token, past_key_values=cache, use_cache=True)

    layer = cache.layers[0]
    assert isinstance(layer, PackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    assert packed.payload.dtype == torch.uint8
    assert packed.scales.dtype == torch.float16
    assert layer.recurrent_states[0].shape == (1, 2, 8, 8)
    assert output.logits.shape == (1, 1, 128)
    assert len(cache.update_evidence) == 2

    summary = cache.storage_summary()
    assert summary["resident_bytes"] == 80
    assert summary["full_precision_equivalent_bytes"] == 512
    assert summary["largest_materialized_state_bytes"] == 512
    assert summary["resident_compression_ratio"] == 6.4
    assert summary["physical_reduction_realized"] is True


def test_packed_cache_does_not_collect_unbounded_evidence_by_default() -> None:
    config = tiny_config()
    cache = PackedRecurrentStateCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16),
    )
    state = torch.linspace(-1, 1, 128).reshape(1, 2, 8, 8)

    cache.update_recurrent_state(state, layer_idx=0)

    assert cache.update_evidence == []
    assert cache.layers[0].is_compileable is False


def test_packed_cache_reports_when_group_padding_outweighs_packing() -> None:
    cache = PackedRecurrentStateCache(
        tiny_config(),
        spec=QuantizationSpec(bits=4, group_size=1024),
    )

    cache.update_recurrent_state(torch.ones(1, 2, 8, 8), layer_idx=0)

    assert cache.storage_summary() == {
        "resident_bytes": 1028,
        "full_precision_equivalent_bytes": 512,
        "largest_materialized_state_bytes": 512,
        "resident_compression_ratio": 512 / 1028,
        "physical_reduction_realized": False,
    }


def test_packed_cache_rejects_autograd_tracked_state_but_allows_no_grad() -> None:
    cache = PackedRecurrentStateCache(
        tiny_config(),
        spec=QuantizationSpec(bits=4, group_size=16),
    )
    state = torch.linspace(-1, 1, 128, requires_grad=True).reshape(1, 2, 8, 8)

    with pytest.raises(RuntimeError, match=r"torch\.inference_mode\(\).*torch\.no_grad\(\)"):
        cache.update_recurrent_state(state, layer_idx=0)

    with torch.no_grad():
        materialized = cache.update_recurrent_state(state, layer_idx=0)
    assert materialized.requires_grad is False


def test_packed_cache_matches_qdq_cache_numerically() -> None:
    torch.manual_seed(23)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    spec = QuantizationSpec(bits=4, group_size=16)
    qdq_cache = RecurrentStateQDQCache(config, spec=spec)
    packed_cache = PackedRecurrentStateCache(config, spec=spec)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode():
        model(prompt, past_key_values=qdq_cache, use_cache=True)
        qdq_output = model(next_token, past_key_values=qdq_cache, use_cache=True)
        model(prompt, past_key_values=packed_cache, use_cache=True)
        packed_output = model(next_token, past_key_values=packed_cache, use_cache=True)

    assert torch.equal(qdq_output.logits, packed_output.logits)
    assert torch.equal(
        qdq_cache.layers[0].recurrent_states[0],
        packed_cache.layers[0].recurrent_states[0],
    )


def test_stochastic_packed_cache_matches_qdq_across_multiple_linear_layers() -> None:
    torch.manual_seed(29)
    config = tiny_config(
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    )
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    spec = QuantizationSpec(bits=4, group_size=16, rounding="stochastic", seed=31)
    qdq_cache = RecurrentStateQDQCache(config, spec=spec)
    packed_cache = PackedRecurrentStateCache(config, spec=spec)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    continuation = torch.randint(0, config.vocab_size, (1, 2))

    with torch.inference_mode():
        model(prompt, past_key_values=qdq_cache, use_cache=True)
        qdq_output = model(continuation, past_key_values=qdq_cache, use_cache=True)
        model(prompt, past_key_values=packed_cache, use_cache=True)
        packed_output = model(continuation, past_key_values=packed_cache, use_cache=True)

    assert torch.equal(qdq_output.logits, packed_output.logits)
    for layer_index in range(3):
        assert torch.equal(
            qdq_cache.layers[layer_index].recurrent_states[0],
            packed_cache.layers[layer_index].recurrent_states[0],
        )


def test_layer_override_changes_physical_payload_width() -> None:
    config = tiny_config()
    cache = PackedRecurrentStateCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16),
        layer_specs={0: QuantizationSpec(bits=8, group_size=16)},
    )
    state = torch.linspace(-1, 1, 128).reshape(1, 2, 8, 8)

    cache.update_recurrent_state(state, layer_idx=0)

    layer = cache.layers[0]
    assert isinstance(layer, PackedLinearAttentionLayer)
    packed = layer.packed_states[0]
    assert packed is not None
    assert packed.payload.dtype == torch.int8
    assert packed.storage_bytes == 144


def test_packed_cache_rejects_non_linear_and_unknown_overrides() -> None:
    config = tiny_config()

    for layer_index in (1, 99):
        with pytest.raises(ValueError, match="non-linear or unknown"):
            PackedRecurrentStateCache(
                config,
                spec=QuantizationSpec(bits=4, group_size=16),
                layer_specs={layer_index: QuantizationSpec(bits=8, group_size=16)},
            )
