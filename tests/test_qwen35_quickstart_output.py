from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import recurquant.qwen35_quickstart as quickstart


class _FakeEncoding(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> _FakeEncoding:
        del device
        return self


class _FakeTokenizer:
    eos_token_id = 2

    def __call__(self, prompt: str, *, return_tensors: str) -> _FakeEncoding:
        assert prompt == quickstart.DEFAULT_PROMPT
        assert return_tensors == "pt"
        return _FakeEncoding(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert token_ids.tolist() == [2]
        assert skip_special_tokens is True
        return "synthetic completion"


class _FakeModel:
    def __init__(self) -> None:
        self.training = True

    def to(self, device: torch.device) -> _FakeModel:
        assert device == torch.device("cpu")
        return self

    def eval(self) -> _FakeModel:
        self.training = False
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        assert kwargs["use_cache"] is True
        assert kwargs["logits_to_keep"] == 1
        logits = torch.tensor([[[0.0, 0.0, 1.0]]])
        return SimpleNamespace(logits=logits)


class _FakeCache:
    summary = {
        "resident_bytes": 24,
        "full_precision_equivalent_bytes": 192,
        "largest_materialized_state_bytes": 96,
        "resident_compression_ratio": 8.0,
        "physical_reduction_realized": True,
    }

    def storage_summary(self) -> dict[str, int | float | bool]:
        return dict(self.summary)


@pytest.fixture
def fake_workflow(monkeypatch: pytest.MonkeyPatch) -> _FakeCache:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    cache = _FakeCache()

    def load_tokenizer(*args: object, **kwargs: object) -> _FakeTokenizer:
        assert args == (quickstart.MODEL_ID,)
        assert kwargs["revision"] == quickstart.MODEL_REVISION
        assert kwargs["local_files_only"] is True
        return tokenizer

    def load_model(*args: object, **kwargs: object) -> _FakeModel:
        assert args == (quickstart.MODEL_ID,)
        assert kwargs["revision"] == quickstart.MODEL_REVISION
        assert kwargs["local_files_only"] is True
        return model

    def create_cache(loaded_model: torch.nn.Module, policy: str) -> _FakeCache:
        assert loaded_model is model
        assert policy == quickstart.MIXED_POLICY
        return cache

    monkeypatch.setattr(quickstart.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(quickstart.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(quickstart, "_create_cache", create_cache)
    return cache


def test_json_mode_prints_one_complete_document(
    fake_workflow: _FakeCache,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = quickstart.main(
        ["--device", "cpu", "--max-new-tokens", "1", "--local-files-only", "--json"]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "generated_text": "synthetic completion",
        "model_id": quickstart.MODEL_ID,
        "model_revision": quickstart.MODEL_REVISION,
        "policy": quickstart.MIXED_POLICY,
        "storage_summary": fake_workflow.summary,
    }


def test_human_mode_preserves_existing_output(
    fake_workflow: _FakeCache,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = quickstart.main(
        ["--device", "cpu", "--max-new-tokens", "1", "--local-files-only"]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == (
        "policy=mixed-v02\n"
        "synthetic completion\n"
        "resident_recurrent_state_bytes=24\n"
        "full_precision_equivalent_recurrent_state_bytes=192\n"
        "largest_materialized_recurrent_state_bytes=96\n"
        "resident_compression_ratio=8.000x\n"
        "physical_reduction_realized=True\n"
    )
