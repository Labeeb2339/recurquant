from __future__ import annotations

import pytest
import torch

from recurquant.quantization import QuantizationSpec, quantize_dequantize


def test_zero_state_round_trips_exactly() -> None:
    state = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    result = quantize_dequantize(state, QuantizationSpec(bits=4, group_size=8))

    assert torch.equal(result.tensor, state)
    assert result.mean_squared_error == 0.0
    assert result.max_absolute_error == 0.0


def test_storage_estimate_includes_group_scales() -> None:
    state = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    result = quantize_dequantize(state, QuantizationSpec(bits=4, group_size=8))

    assert result.elements == 32
    assert result.groups == 4
    assert result.payload_bits == 128
    assert result.scale_bits == 64
    assert result.estimated_bytes == 24
    assert result.baseline_bytes == 128


def test_more_bits_reduce_error_on_same_groups() -> None:
    state = torch.linspace(-3.1, 2.7, 257, dtype=torch.float32).reshape(1, 1, 1, 257)
    int4 = quantize_dequantize(state, QuantizationSpec(bits=4, group_size=64))
    int8 = quantize_dequantize(state, QuantizationSpec(bits=8, group_size=64))

    assert int8.mean_squared_error < int4.mean_squared_error
    assert int8.max_absolute_error < int4.max_absolute_error


def test_stochastic_rounding_is_seeded() -> None:
    state = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    spec = QuantizationSpec(bits=4, group_size=16, rounding="stochastic", seed=17)

    first = quantize_dequantize(state, spec)
    second = quantize_dequantize(state, spec)

    assert torch.equal(first.tensor, second.tensor)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bits": 1}, "bits"),
        ({"bits": 4, "group_size": 0}, "group_size"),
        ({"bits": 4, "scale_bits": 0}, "scale_bits"),
        ({"bits": 4, "flatten_last_dims": 0}, "flatten_last_dims"),
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QuantizationSpec(**kwargs)


def test_integer_tensor_is_rejected() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        quantize_dequantize(torch.ones(4, dtype=torch.int64), QuantizationSpec(bits=4))
