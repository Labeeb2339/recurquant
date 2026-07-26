from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from recurquant.mixed_quantization import quantize_pack_mixed
from recurquant.quantization import QuantizationSpec
from recurquant.rht import right_rht_encode, right_rht_signs

LOW = QuantizationSpec(bits=4, group_size=128, flatten_last_dims=2)
HIGH = QuantizationSpec(bits=8, group_size=128, flatten_last_dims=2)
DOMAIN = b"recurquant.right-rht.signs.v1\0"
SEED = 2339


def _reference_signs(layer: int, heads: int, width: int) -> np.ndarray:
    result = np.empty((1, heads, 1, width), dtype=np.float32)
    for head in range(heads):
        values: list[float] = []
        counter = 0
        while len(values) < width:
            message = b"".join(
                (
                    DOMAIN,
                    SEED.to_bytes(8, "little"),
                    layer.to_bytes(8, "little"),
                    head.to_bytes(8, "little"),
                    width.to_bytes(8, "little"),
                    counter.to_bytes(8, "little"),
                )
            )
            for byte in hashlib.sha256(message).digest():
                for bit in range(8):
                    values.append(1.0 if byte & (1 << bit) else -1.0)
                    if len(values) == width:
                        break
                if len(values) == width:
                    break
            counter += 1
        result[0, head, 0] = np.asarray(values, dtype=np.float32)
    return result


def _dense_hadamard(width: int) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < width:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _reference_encode(state: np.ndarray, *, layer: int) -> np.ndarray:
    width = state.shape[-1]
    signed = state.astype(np.float32) * _reference_signs(
        layer,
        state.shape[1],
        width,
    )
    return np.matmul(signed, _dense_hadamard(width)) / math.sqrt(width)


def _reference_decode(encoded: np.ndarray, *, layer: int) -> np.ndarray:
    width = encoded.shape[-1]
    decoded = np.matmul(encoded.astype(np.float32), _dense_hadamard(width))
    decoded /= math.sqrt(width)
    return decoded * _reference_signs(layer, encoded.shape[1], width)


def _reference_mixed_qdq(
    state: np.ndarray,
    mask: np.ndarray,
    *,
    layer: int,
) -> np.ndarray:
    encoded = _reference_encode(state, layer=layer)
    rows = encoded.reshape(state.shape[0] * state.shape[1], -1)
    groups = rows.reshape(-1, 128)
    flat_mask = mask.reshape(-1)
    qmax = np.where(flat_mask, 127.0, 7.0).reshape(-1, 1)
    ideal_scales = np.max(np.abs(groups), axis=1, keepdims=True) / qmax
    ideal_scales = np.where(ideal_scales > 1e-8, ideal_scales, 1.0)
    ideal_scales = np.clip(
        ideal_scales,
        2.0**-24,
        np.finfo(np.float16).max,
    )
    scales = ideal_scales.astype(np.float16).astype(np.float32)
    codes = np.rint(groups / scales)
    codes = np.maximum(np.minimum(codes, qmax), -qmax)
    quantized = (codes * scales).reshape(encoded.shape)
    return _reference_decode(quantized, layer=layer)


def test_production_rht_matches_independent_dense_reference() -> None:
    state = torch.randn(
        (1, 2, 3, 128),
        generator=torch.Generator().manual_seed(811),
        dtype=torch.float32,
    )
    reference_signs = _reference_signs(layer=7, heads=2, width=128)
    reference_encoded = _reference_encode(state.numpy(), layer=7)

    production_signs = right_rht_signs(
        layer_index=7,
        expected_heads=2,
        width=128,
        device="cpu",
    )
    production_encoded = right_rht_encode(
        state,
        layer_index=7,
        expected_heads=2,
    )

    np.testing.assert_array_equal(production_signs.numpy(), reference_signs)
    np.testing.assert_allclose(
        production_encoded.numpy(),
        reference_encoded,
        rtol=2e-6,
        atol=2e-6,
    )


def test_physical_rht_pack_matches_independent_dense_quantizer() -> None:
    state = torch.randn(
        (1, 2, 4, 128),
        generator=torch.Generator().manual_seed(823),
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[False, True, False, True], [True, False, False, True]],
        dtype=torch.bool,
    )
    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=LOW,
        high_spec=HIGH,
        right_rht_layer_index=9,
        right_rht_expected_heads=2,
    )

    reference = _reference_mixed_qdq(
        state.numpy(),
        mask.numpy(),
        layer=9,
    )

    np.testing.assert_allclose(
        packed.dequantize().numpy(),
        reference,
        rtol=5e-6,
        atol=6e-6,
    )
