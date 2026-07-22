from __future__ import annotations

import pytest
import torch

from recurquant.quantization import QuantizationSpec, quantize_dequantize, quantize_pack


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


def test_fp16_scale_storage_changes_the_numerical_round_trip() -> None:
    state = torch.tensor([0.0, 0.017, 0.061, 0.1234567], dtype=torch.float32)

    fp16_scales = quantize_dequantize(
        state,
        QuantizationSpec(bits=4, group_size=4, scale_bits=16, flatten_last_dims=1),
    )
    fp32_scales = quantize_dequantize(
        state,
        QuantizationSpec(bits=4, group_size=4, scale_bits=32, flatten_last_dims=1),
    )

    assert not torch.equal(fp16_scales.tensor, fp32_scales.tensor)
    assert fp16_scales.scale_bits == 16
    assert fp32_scales.scale_bits == 32


def test_fp16_scale_storage_stays_finite_for_tiny_nonzero_values() -> None:
    state = torch.full((8,), 1e-10, dtype=torch.float32)
    spec = QuantizationSpec(
        bits=4,
        group_size=8,
        scale_bits=16,
        flatten_last_dims=1,
        epsilon=1e-12,
    )

    qdq = quantize_dequantize(state, spec)
    packed = quantize_pack(state, spec)

    assert torch.isfinite(qdq.tensor).all()
    assert torch.isfinite(packed.dequantize()).all()
    assert torch.equal(packed.dequantize(), qdq.tensor)


@pytest.mark.parametrize("bits", [4, 8])
def test_physical_pack_matches_qdq_and_exact_storage(bits: int) -> None:
    state = torch.linspace(-2.3, 1.7, 30, dtype=torch.float32).reshape(1, 2, 3, 5)
    spec = QuantizationSpec(bits=bits, group_size=8)

    qdq = quantize_dequantize(state, spec)
    packed = quantize_pack(state, spec)

    assert torch.equal(packed.dequantize(), qdq.tensor)
    assert packed.storage_bytes == qdq.estimated_bytes
    assert packed.payload.dtype == (torch.uint8 if bits == 4 else torch.int8)


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
        ({"bits": 4, "scale_bits": 8}, "scale_bits"),
        ({"bits": 4, "flatten_last_dims": 0}, "flatten_last_dims"),
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QuantizationSpec(**kwargs)


def test_integer_tensor_is_rejected() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        quantize_dequantize(torch.ones(4, dtype=torch.int64), QuantizationSpec(bits=4))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_tensor_is_rejected_consistently(invalid: float) -> None:
    state = torch.tensor([1.0, invalid])
    spec = QuantizationSpec(bits=4, group_size=2, flatten_last_dims=1)

    with pytest.raises(ValueError, match="finite"):
        quantize_dequantize(state, spec)
    with pytest.raises(ValueError, match="finite"):
        quantize_pack(state, spec)


def test_physical_pack_rejects_unimplemented_bit_width() -> None:
    with pytest.raises(ValueError, match="INT4 and INT8"):
        quantize_pack(torch.ones(8), QuantizationSpec(bits=6))
