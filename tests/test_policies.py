from __future__ import annotations

import pytest

from recurquant.policies import LayerPrecisionPlan, select_high_precision_layers


def test_select_high_precision_layers_is_deterministic_on_ties() -> None:
    scores = {4: 0.2, 0: 0.9, 2: 0.9, 6: 0.1}
    assert select_high_precision_layers(scores, count=2) == (0, 2)


def test_layer_precision_plan_reports_average_payload_bits() -> None:
    plan = LayerPrecisionPlan(default_bits=4, high_bits=8, high_precision_layers=(0,))
    assert plan.average_payload_bits(layer_count=18) == pytest.approx(4.2222222222)


def test_invalid_layer_precision_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceed"):
        LayerPrecisionPlan(default_bits=4, high_bits=4, high_precision_layers=(0,))
