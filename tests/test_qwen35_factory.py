from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, Qwen3_5ForCausalLM

import recurquant.qwen35 as qwen35_module
from examples.qwen35_quickstart import _model_dtype
from recurquant import PackedRecurrentStateCache, create_qwen35_packed_cache
from recurquant.packed_cache import PackedLinearAttentionLayer
from tests.test_transformers_cache import tiny_config


def _tiny_model(layer_types: list[str] | None = None) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(layer_types),
        attn_implementation="eager",
    ).eval()


def test_public_factory_runs_prefill_and_incremental_decode_on_cpu() -> None:
    torch.manual_seed(41)
    model = _tiny_model()
    cache = create_qwen35_packed_cache(model, bits=4, group_size=16)
    prompt = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        prefill = model(prompt, past_key_values=cache, use_cache=True)
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode = model(next_token, past_key_values=cache, use_cache=True)

    assert isinstance(cache, PackedRecurrentStateCache)
    assert decode.logits.shape == (1, 1, model.config.vocab_size)
    assert cache.storage_summary() == {
        "resident_bytes": 80,
        "full_precision_equivalent_bytes": 512,
        "largest_materialized_state_bytes": 512,
        "resident_compression_ratio": 6.4,
        "physical_reduction_realized": True,
    }


def test_public_factory_runs_greedy_generation_on_cpu() -> None:
    torch.manual_seed(43)
    model = _tiny_model()
    cache = create_qwen35_packed_cache(model.config, bits=4, group_size=16)
    prompt = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        generated = model.generate(
            prompt,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=0,
        )

    assert generated.shape == (1, 7)
    assert cache.storage_summary()["physical_reduction_realized"] is True


def test_public_factory_runs_beam_generation_and_reorders_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(47)
    model = _tiny_model()
    cache = create_qwen35_packed_cache(model, bits=4, group_size=16)
    prompt = torch.tensor([[1, 2, 3, 4]])
    reorder_indices: list[torch.Tensor] = []
    original_reorder = PackedLinearAttentionLayer.reorder_cache

    def record_reorder(
        layer: PackedLinearAttentionLayer,
        beam_idx: torch.LongTensor,
    ) -> None:
        reorder_indices.append(beam_idx.detach().cpu().clone())
        original_reorder(layer, beam_idx)

    monkeypatch.setattr(PackedLinearAttentionLayer, "reorder_cache", record_reorder)
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
    assert cache.storage_summary()["resident_bytes"] == 240


def test_public_factory_rejects_unsupported_inputs_with_actions() -> None:
    with pytest.raises(TypeError, match="text-only Qwen3.5"):
        create_qwen35_packed_cache(GPT2Config())  # type: ignore[arg-type]

    training_model = Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    )
    with pytest.raises(ValueError, match=r"Call model\.eval\(\).*torch\.inference_mode\(\)"):
        create_qwen35_packed_cache(training_model)

    with pytest.raises(ValueError, match="no linear_attention layers"):
        create_qwen35_packed_cache(tiny_config(["full_attention"]))

    with pytest.raises(ValueError, match="bits=4 or bits=8"):
        create_qwen35_packed_cache(tiny_config(), bits=3)


def test_config_only_factory_validates_structure_not_runtime() -> None:
    config = tiny_config()
    config._attn_implementation = "sdpa"

    cache = create_qwen35_packed_cache(config, bits=4, group_size=16)

    assert isinstance(cache, PackedRecurrentStateCache)


def test_model_factory_rejects_untested_attention_backend() -> None:
    model = _tiny_model()
    model.config._attn_implementation = "sdpa"

    with pytest.raises(ValueError, match="attn_implementation='eager'"):
        create_qwen35_packed_cache(model)


def test_model_factory_rejects_sharded_device_map() -> None:
    model = _tiny_model()
    model.hf_device_map = {
        "model.layers.0": "cpu",
        "model.layers.1": "cuda:0",
    }

    with pytest.raises(ValueError, match="sharded or multi-device"):
        create_qwen35_packed_cache(model)


def test_model_factory_rejects_meta_parameters() -> None:
    model = _tiny_model().to("meta")

    with pytest.raises(ValueError, match="containing meta tensors"):
        create_qwen35_packed_cache(model)


def test_eval_model_forward_requires_disabled_autograd() -> None:
    model = _tiny_model()
    cache = create_qwen35_packed_cache(model, bits=4, group_size=16)

    with pytest.raises(RuntimeError, match=r"model\.eval\(\) alone does not disable autograd"):
        model(torch.tensor([[1, 2, 3, 4]]), past_key_values=cache, use_cache=True)


@pytest.mark.parametrize("version", ["5.15.0", "5.14.2rc1", "5.14.2.dev1"])
def test_public_factory_rejects_untested_or_prerelease_transformers(
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen35_module.transformers, "__version__", version)
    with pytest.raises(RuntimeError, match=r"transformers>=5\.14\.1,<5\.15"):
        create_qwen35_packed_cache(tiny_config())


def test_public_factory_rejects_invalid_version_before_loading_qwen_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen35_module.transformers, "__version__", "not-a-version")

    def classes_must_not_load() -> tuple[type[object], type[object]]:
        pytest.fail("Qwen classes loaded before version validation")

    monkeypatch.setattr(qwen35_module, "_load_qwen_classes", classes_must_not_load)
    with pytest.raises(RuntimeError, match="could not interpret Transformers version"):
        create_qwen35_packed_cache(tiny_config())


def test_quickstart_dtype_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _model_dtype(torch.device("cpu")) is torch.float32

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _model_dtype(torch.device("cuda")) is torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.warns(RuntimeWarning, match="not been validated for FP16"):
        assert _model_dtype(torch.device("cuda")) is torch.float16
