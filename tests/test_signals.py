from __future__ import annotations

import torch
from transformers import DynamicCache, Qwen3_5ForCausalLM

from recurquant.quantization import QuantizationSpec
from recurquant.signals import GatedDeltaSignalRecorder
from tests.test_transformers_cache import tiny_config


def test_signal_recorder_captures_prefill_and_decode_without_mutating_shape() -> None:
    torch.manual_seed(13)
    config = tiny_config()
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    cache = DynamicCache(config=config)
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    recorder = GatedDeltaSignalRecorder(
        model,
        probe_spec=QuantizationSpec(bits=4, group_size=16),
    )
    with recorder, torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True)
        output = model(next_token, past_key_values=cache, use_cache=True)

    assert output.logits.shape == (1, 1, 128)
    assert len(recorder.records) == 2
    prefill, decode = recorder.records
    assert prefill.sequence_length == 6
    assert not prefill.had_initial_state
    assert prefill.state_update_relative_l2 is None
    assert decode.sequence_length == 1
    assert decode.had_initial_state
    assert decode.state_update_relative_l2 is not None
    assert decode.state_update_relative_l2 >= 0
    assert decode.committed_residual_rms is not None
    assert decode.probe_state_relative_l2 is not None
    assert decode.probe_read_error_rms is not None
    assert decode.probe_read_relative_l2 is not None
    assert decode.probe_state_relative_l2 >= 0
    assert decode.probe_read_error_rms >= 0
    assert decode.probe_read_relative_l2 >= 0
    assert 0 <= decode.beta_min <= decode.beta_max <= 1
    assert 0 <= decode.retention_min <= decode.retention_max <= 1


def test_signal_recorder_restores_kernel_functions() -> None:
    model = Qwen3_5ForCausalLM._from_config(tiny_config(), attn_implementation="eager").eval()
    gdn = model.model.layers[0].linear_attn
    original_chunk = gdn.chunk_gated_delta_rule
    original_recurrent = gdn.recurrent_gated_delta_rule

    with GatedDeltaSignalRecorder(model):
        assert gdn.chunk_gated_delta_rule is not original_chunk
        assert gdn.recurrent_gated_delta_rule is not original_recurrent

    assert gdn.chunk_gated_delta_rule is original_chunk
    assert gdn.recurrent_gated_delta_rule is original_recurrent
