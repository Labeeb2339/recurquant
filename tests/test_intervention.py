from __future__ import annotations

import math

import pytest
import torch
from transformers import Qwen3_5ForCausalLM

import recurquant.intervention as intervention_module
from recurquant.intervention import (
    evaluate_physical_row_promotions,
    target_nll_values,
)
from recurquant.row_policy import RowLocation
from tests.test_transformers_cache import tiny_config


def _tiny_model(layer_types: list[str] | None = None) -> Qwen3_5ForCausalLM:
    torch.manual_seed(211)
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(layer_types),
        attn_implementation="eager",
    ).eval()


def test_physical_row_promotions_use_fixed_budget_and_repeated_cache_updates() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]])
    continuation = torch.tensor([[4, 5]])
    candidates = (RowLocation(0, 1, 7), RowLocation(0, 0, 0))

    result = evaluate_physical_row_promotions(
        model,
        prompt_ids=prompt,
        continuation_ids=continuation,
        candidate_rows=candidates,
    )

    # 16 rows, each with a four-byte INT4 payload and two-byte scale, plus
    # a two-byte packed mask. One promoted eight-element row adds four bytes.
    assert result.baseline_plan_bytes == 98
    assert result.intervention_plan_bytes == 102
    assert result.promotion_increment_bytes == 4
    assert result.baseline.resident_bytes == 98
    assert result.baseline.high_precision_groups == 0
    assert result.baseline.token_count == 2
    assert result.baseline.cache_update_count == 2
    assert tuple(measurement.location for measurement in result.measurements) == (
        RowLocation(0, 0, 0),
        RowLocation(0, 1, 7),
    )
    for measurement in result.measurements:
        assert measurement.run.resident_bytes == 102
        assert measurement.run.high_precision_groups == 1
        assert measurement.run.cache_update_count == 2
        assert math.isfinite(measurement.run.mean_metric)
        assert measurement.metric_delta == pytest.approx(
            measurement.run.mean_metric - result.baseline.mean_metric
        )
        assert measurement.improvement == pytest.approx(-measurement.metric_delta)
    assert sorted(result.ranked_measurements(), key=lambda item: -item.improvement) == list(
        result.ranked_measurements()
    )


def test_background_intervention_and_custom_metric_are_deterministic() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[9, 10]])
    continuation = torch.tensor([[11]])

    def target_log_probability(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -target_nll_values(logits, targets)

    arguments = {
        "prompt_ids": prompt,
        "continuation_ids": continuation,
        "candidate_rows": (RowLocation(0, 0, 1),),
        "background_rows": (RowLocation(0, 0, 0),),
        "metric": target_log_probability,
        "metric_name": "target_log_probability",
        "lower_is_better": False,
    }
    first = evaluate_physical_row_promotions(model, **arguments)
    second = evaluate_physical_row_promotions(model, **arguments)

    assert first == second
    assert first.baseline.high_precision_groups == 1
    assert first.measurements[0].run.high_precision_groups == 2
    assert first.baseline_plan_bytes == 102
    assert first.intervention_plan_bytes == 106
    assert first.measurements[0].improvement == pytest.approx(first.measurements[0].metric_delta)


def test_cache_update_count_covers_every_recurrent_layer_and_forward() -> None:
    model = _tiny_model(
        ["linear_attention", "linear_attention", "full_attention"]
    )

    result = evaluate_physical_row_promotions(
        model,
        prompt_ids=torch.tensor([[1, 2]]),
        continuation_ids=torch.tensor([[3, 4, 5]]),
        candidate_rows=(RowLocation(0, 0, 0),),
    )

    # One prompt forward plus two continuation transitions, each updating one
    # recurrent state in both linear-attention layers.
    assert result.baseline.cache_update_count == 6
    assert result.measurements[0].run.cache_update_count == 6


def test_intervention_rejects_unbounded_invalid_or_inexact_candidates() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2]])
    continuation = torch.tensor([[3]])
    location = RowLocation(0, 0, 0)

    common = {
        "model": model,
        "prompt_ids": prompt,
        "continuation_ids": continuation,
    }
    with pytest.raises(ValueError, match="must not contain duplicates"):
        evaluate_physical_row_promotions(
            **common,
            candidate_rows=(location, location),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        evaluate_physical_row_promotions(
            **common,
            candidate_rows=(location,),
            background_rows=(location,),
        )
    with pytest.raises(ValueError, match="exceeding max_candidates"):
        evaluate_physical_row_promotions(
            **common,
            candidate_rows=(location, RowLocation(0, 0, 1)),
            max_candidates=1,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_physical_row_promotions(
            **common,
            candidate_rows=(RowLocation(0, 2, 0),),
        )
    with pytest.raises(ValueError, match="must exactly encode"):
        evaluate_physical_row_promotions(
            **common,
            candidate_rows=(location,),
            intervention_resident_bytes=101,
        )


def test_intervention_validates_metric_output_per_target() -> None:
    model = _tiny_model()

    def scalar_metric(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return logits.mean()

    with pytest.raises(ValueError, match="one value per target"):
        evaluate_physical_row_promotions(
            model,
            prompt_ids=torch.tensor([[1, 2]]),
            continuation_ids=torch.tensor([[3]]),
            candidate_rows=(RowLocation(0, 0, 0),),
            metric=scalar_metric,
        )


def test_intervention_rejects_an_incomplete_cache_update_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_factory = intervention_module.create_qwen35_exact_budget_cache

    class DropFirstEvidence(list[object]):
        def __init__(self) -> None:
            super().__init__()
            self._dropped = False

        def append(self, value: object) -> None:
            if not self._dropped:
                self._dropped = True
                return
            super().append(value)

    def create_cache_with_incomplete_evidence(*args: object, **kwargs: object) -> object:
        cache = original_factory(*args, **kwargs)  # type: ignore[arg-type]
        cache.update_evidence = DropFirstEvidence()  # type: ignore[assignment]
        return cache

    monkeypatch.setattr(
        intervention_module,
        "create_qwen35_exact_budget_cache",
        create_cache_with_incomplete_evidence,
    )

    with pytest.raises(RuntimeError, match="update trace did not cover"):
        evaluate_physical_row_promotions(
            _tiny_model(),
            prompt_ids=torch.tensor([[1, 2]]),
            continuation_ids=torch.tensor([[3, 4]]),
            candidate_rows=(RowLocation(0, 0, 0),),
        )
