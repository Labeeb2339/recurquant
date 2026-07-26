from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache

from recurquant.storage_boundary_validation import (
    StorageRowLocation,
)
from recurquant.storage_boundary_validation import (
    _directional_dot_float64 as _module_directional_dot_float64,
)
from tests.test_transformers_cache import tiny_config


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "pilot_validate_storage_boundary.py"
    spec = importlib.util.spec_from_file_location("pilot_validate_storage_boundary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_grid_and_strata_resolve_without_scores() -> None:
    script = _load_script()
    recurrent_layers = tuple(range(0, 36, 2))

    resolved = script._resolve_strata(recurrent_layers, heads=16, rows=128)

    assert script.EPSILONS == (0.25, 0.125, 0.0625, 0.03125)
    assert resolved == (
        ("early_low", StorageRowLocation(0, 0, 0)),
        ("early_mid", StorageRowLocation(10, 5, 42)),
        ("late_mid", StorageRowLocation(22, 10, 85)),
        ("late_high", StorageRowLocation(34, 15, 127)),
    )


def test_fixed_strata_refuse_incompatible_geometry() -> None:
    script = _load_script()

    with pytest.raises(ValueError, match="recurrent-layer geometry"):
        script._resolve_strata(tuple(range(17)), heads=16, rows=128)
    with pytest.raises(ValueError, match="recurrent-state geometry"):
        script._resolve_strata(tuple(range(18)), heads=8, rows=128)


def test_cache_fingerprint_covers_recurrent_history_flags() -> None:
    script = _load_script()
    cache = DynamicCache(config=tiny_config())
    before = script._cache_fingerprint(cache)

    cache.layers[0].has_previous_state[0] = True

    assert script._cache_fingerprint(cache) != before


def test_saved_directional_dot_uses_matching_float64_accumulation() -> None:
    script = _load_script()
    gradient = torch.tensor([100_000_000.0, 1.0, -100_000_000.0], dtype=torch.float32)
    direction = torch.ones_like(gradient)

    assert float((gradient * direction).sum().item()) == 0.0
    assert _module_directional_dot_float64(gradient, direction) == 1.0
    assert script._directional_dot_float64(gradient, direction) == 1.0


def _gate_rows(script, *, central_scale: float = 1.0, repeat_error: float = 0.0):
    rows = []
    for row_index, (label, *_coordinates) in enumerate(script.ROW_STRATA, start=1):
        autograd = row_index * 1e-4
        rows.append(
            {
                "stratum": label,
                "epsilon_results": [
                    {
                        "epsilon": epsilon,
                        "autograd_directional_derivative": autograd,
                        "central_directional_derivative": autograd * central_scale,
                        "baseline_repeat_absolute_error": repeat_error,
                    }
                    for epsilon in script.EPSILONS
                ],
            }
        )
    return rows


def test_frozen_derivative_gate_passes_consistent_fp32_grid() -> None:
    script = _load_script()

    gate = script._evaluate_derivative_gate(_gate_rows(script, central_scale=1.02))

    assert gate["passed"] is True
    assert gate["failures"] == []
    assert gate["observed"]["informative_rows"] == 4
    assert gate["observed"]["sign_agreement"] == 1.0
    assert gate["observed"]["median_relative_error"] == pytest.approx(0.02)


def test_frozen_derivative_gate_rejects_sign_and_repeat_failures() -> None:
    script = _load_script()

    gate = script._evaluate_derivative_gate(
        _gate_rows(script, central_scale=-1.0, repeat_error=1e-4)
    )

    assert gate["passed"] is False
    assert any("sign agreement" in failure for failure in gate["failures"])
    assert any("alpha=0" in failure for failure in gate["failures"])


def test_frozen_derivative_gate_rejects_too_many_near_zero_rows() -> None:
    script = _load_script()
    rows = _gate_rows(script)
    for row in rows[:2]:
        for result in row["epsilon_results"]:
            result["autograd_directional_derivative"] = 0.0
            result["central_directional_derivative"] = 0.0

    gate = script._evaluate_derivative_gate(rows)

    assert gate["passed"] is False
    assert gate["observed"]["informative_rows"] == 2
    assert any("informative rows" in failure for failure in gate["failures"])


def test_frozen_derivative_gate_all_near_zero_is_json_safe_failure() -> None:
    script = _load_script()
    rows = _gate_rows(script)
    for row in rows:
        for result in row["epsilon_results"]:
            result["autograd_directional_derivative"] = script.DERIVATIVE_INFORMATIVE_FLOOR
            result["central_directional_derivative"] = 0.0

    gate = script._evaluate_derivative_gate(rows)

    assert gate["passed"] is False
    assert gate["observed"]["informative_rows"] == 0
    assert gate["observed"]["near_zero_rows"] == 4
    assert gate["observed"]["sign_agreement"] is None
    assert gate["observed"]["median_relative_error"] is None
    assert gate["observed"]["converged_row_fraction"] is None
    json.dumps(gate, allow_nan=False)


def test_frozen_derivative_gate_rejects_nonfinite_inputs_and_wrong_strata() -> None:
    script = _load_script()
    rows = _gate_rows(script)
    rows[0]["epsilon_results"][0]["central_directional_derivative"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        script._evaluate_derivative_gate(rows)

    rows = _gate_rows(script)
    rows[0]["stratum"] = "substituted_after_results"
    with pytest.raises(ValueError, match="source order"):
        script._evaluate_derivative_gate(rows)
