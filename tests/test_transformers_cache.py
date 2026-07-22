from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from recurquant.quantization import QuantizationSpec
from recurquant.transformers_cache import RecurrentStateQDQCache


def tiny_config(layer_types: list[str] | None = None) -> Qwen3_5TextConfig:
    if layer_types is None:
        layer_types = ["linear_attention", "full_attention"]
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=layer_types,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [3, 3, 2],
        },
    )


def test_qdq_cache_runs_prefill_and_one_token_decode() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    cache = RecurrentStateQDQCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16),
    )
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True)
        output = model(next_token, past_key_values=cache, use_cache=True)

    assert cache.layers[0].recurrent_states[0].shape == (1, 2, 8, 8)
    assert cache.layers[0].conv_states[0].shape == (1, 48, 2)
    assert output.logits.shape == (1, 1, 128)
    assert len(cache.update_evidence) == 2
    assert all(item.layer_index == 0 for item in cache.update_evidence)


def test_stochastic_cache_uses_a_distinct_reproducible_stream_per_update() -> None:
    config = tiny_config()
    cache = RecurrentStateQDQCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16, rounding="stochastic", seed=11),
    )
    state = torch.linspace(-1, 1, 128).reshape(1, 2, 8, 8)

    first = cache.update_recurrent_state(state, layer_idx=0).clone()
    second = cache.update_recurrent_state(state, layer_idx=0).clone()

    assert not torch.equal(first, second)

    repeated_cache = RecurrentStateQDQCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16, rounding="stochastic", seed=11),
    )
    repeated_first = repeated_cache.update_recurrent_state(state, layer_idx=0).clone()
    repeated_second = repeated_cache.update_recurrent_state(state, layer_idx=0).clone()
    assert torch.equal(first, repeated_first)
    assert torch.equal(second, repeated_second)


def test_layer_specific_spec_overrides_default() -> None:
    config = tiny_config()
    cache = RecurrentStateQDQCache(
        config,
        spec=QuantizationSpec(bits=4, group_size=16),
        layer_specs={0: QuantizationSpec(bits=8, group_size=16)},
    )
    state = torch.linspace(-1, 1, 128).reshape(1, 2, 8, 8)

    cache.update_recurrent_state(state, layer_idx=0)

    assert cache.update_evidence[0].bits == 8


def test_qdq_cache_rejects_non_linear_or_unknown_configuration() -> None:
    config = tiny_config()
    spec = QuantizationSpec(bits=4, group_size=16)

    with pytest.raises(ValueError, match="non-linear or unknown"):
        RecurrentStateQDQCache(config, spec=spec, layer_specs={1: spec})
    with pytest.raises(ValueError, match="non-linear or unknown"):
        RecurrentStateQDQCache(config, spec=spec, enabled_layers=[99])
