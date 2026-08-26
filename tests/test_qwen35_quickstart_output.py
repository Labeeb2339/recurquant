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


class _FakeStateLeaseCache:
    summary = {
        "checkpoint_bytes": 2_564_096,
        "full_precision_equivalent_bytes": 18_874_368,
        "largest_materialized_state_bytes": 1_048_576,
        "physical_reduction_realized_including_statelease": True,
        "resident_bytes": 2_564_096,
        "resident_bytes_including_statelease": 3_454_664,
        "resident_compression_ratio_including_statelease": 5.463445,
    }

    def storage_summary(self) -> dict[str, int | float | bool]:
        return dict(self.summary)

    def statelease_diagnostics(self) -> list[dict[str, int]]:
        return [
            {
                "boundary4_count": 2,
                "boundary5_count": 1,
                "checkpoint_count": 3,
                "observations_committed": 4,
                "tie_count": 0,
            }
        ]


class _ObservedContext:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> None:
        self.entered = True

    def __exit__(self, *args: object) -> None:
        self.exited = True


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


def test_statelease_policy_uses_observer_and_reports_complete_resident_bytes(
    fake_workflow: _FakeCache,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_workflow
    cache = _FakeStateLeaseCache()
    observed = _ObservedContext()

    monkeypatch.setattr(
        quickstart,
        "_create_cache",
        lambda _model, policy: cache
        if policy == quickstart.STATELEASE_H5_POLICY
        else pytest.fail("unexpected policy"),
    )
    monkeypatch.setattr(
        quickstart,
        "_forward_context",
        lambda _model, selected_cache, policy: observed
        if selected_cache is cache and policy == quickstart.STATELEASE_H5_POLICY
        else pytest.fail("StateLease observer context was not selected"),
    )

    result = quickstart.main(
        [
            "--device",
            "cpu",
            "--max-new-tokens",
            "1",
            "--local-files-only",
            "--policy",
            quickstart.STATELEASE_H5_POLICY,
        ]
    )

    assert result == 0
    assert observed.entered is True
    assert observed.exited is True
    assert capsys.readouterr().out == (
        "policy=statelease-h5\n"
        "synthetic completion\n"
        "resident_recurrent_state_and_statelease_bytes=3454664\n"
        "packed_checkpoint_bytes=2564096\n"
        "statelease_exact_row_plan_sha256="
        f"{quickstart.EXPERIMENT012_STATELEASE_H5_EXACT_ROW_PLAN_SHA256}\n"
        "statelease_boundary4_count=2\n"
        "statelease_boundary5_count=1\n"
        "evidence_scope=interactive_smoke_only\n"
        "full_precision_equivalent_recurrent_state_bytes=18874368\n"
        "largest_materialized_recurrent_state_bytes=1048576\n"
        "resident_compression_ratio=5.463x\n"
        "physical_reduction_realized=True\n"
    )


def test_statelease_json_reports_identity_diagnostics_and_use_boundary(
    fake_workflow: _FakeCache,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_workflow
    cache = _FakeStateLeaseCache()
    observed = _ObservedContext()
    monkeypatch.setattr(quickstart, "_create_cache", lambda _model, _policy: cache)
    monkeypatch.setattr(
        quickstart,
        "_forward_context",
        lambda _model, _cache, _policy: observed,
    )

    result = quickstart.main(
        [
            "--device",
            "cpu",
            "--max-new-tokens",
            "1",
            "--local-files-only",
            "--policy",
            quickstart.STATELEASE_H5_POLICY,
            "--json",
        ]
    )

    assert result == 0
    assert observed.entered is True
    assert observed.exited is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["statelease_exact_row_plan_sha256"] == (
        quickstart.EXPERIMENT012_STATELEASE_H5_EXACT_ROW_PLAN_SHA256
    )
    assert payload["statelease_diagnostics"] == {
        "boundary4_count": 2,
        "boundary5_count": 1,
        "checkpoint_count": 3,
        "layers": 1,
        "observations_committed": 4,
        "tie_count": 0,
    }
    assert payload["use_boundary"] == (
        "interactive smoke only; not new Experiment 012 or Stage-B evidence"
    )
    assert payload["storage_summary"] == cache.summary


def test_statelease_policy_rejects_resident_byte_drift(
    fake_workflow: _FakeCache,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_workflow
    cache = _FakeStateLeaseCache()
    cache.summary = {**cache.summary, "resident_bytes_including_statelease": 3_454_663}
    observed = _ObservedContext()
    monkeypatch.setattr(quickstart, "_create_cache", lambda _model, _policy: cache)
    monkeypatch.setattr(
        quickstart,
        "_forward_context",
        lambda _model, _cache, _policy: observed,
    )

    with pytest.raises(RuntimeError, match="resident bytes do not match"):
        quickstart.main(
            [
                "--device",
                "cpu",
                "--max-new-tokens",
                "1",
                "--local-files-only",
                "--policy",
                quickstart.STATELEASE_H5_POLICY,
            ]
        )

    assert observed.entered is True
    assert observed.exited is True
    assert capsys.readouterr().out == ""
