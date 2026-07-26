from __future__ import annotations

import hashlib

import pytest
import torch

from recurquant.rht import (
    fwht_unnormalized,
    right_rht_decode,
    right_rht_encode,
    right_rht_signs,
)


def _q4_absmax(tensor: torch.Tensor) -> torch.Tensor:
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(tensor / scale).clamp(-7, 7) * scale


def test_unnormalized_fwht_has_expected_small_example_and_dtype_restore() -> None:
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)

    transformed = fwht_unnormalized(vector, output_dtype=torch.float32)

    assert transformed.dtype == torch.float32
    assert torch.equal(transformed, torch.tensor([10.0, -2.0, -4.0, 0.0]))
    assert fwht_unnormalized(vector).dtype == torch.float16


def test_unnormalized_fwht_rejects_scalar_and_unsupported_output_dtype() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        fwht_unnormalized(torch.tensor(1.0))
    with pytest.raises(TypeError, match="output_dtype"):
        fwht_unnormalized(torch.ones(8), output_dtype=torch.float64)


def test_encode_decode_inverse_has_tight_fp32_error_bound() -> None:
    generator = torch.Generator().manual_seed(11)
    state = torch.randn((2, 3, 5, 128), generator=generator)

    encoded = right_rht_encode(state, layer_index=7, expected_heads=3)
    restored = right_rht_decode(encoded, layer_index=7, expected_heads=3)
    relative_l2 = torch.linalg.vector_norm(restored - state) / torch.linalg.vector_norm(state)

    assert float(relative_l2.item()) < 3e-7
    torch.testing.assert_close(restored, state, rtol=2e-6, atol=1e-6)


def test_sign_schedule_is_deterministic_portable_and_rng_independent() -> None:
    torch.manual_seed(1)
    first = right_rht_signs(
        layer_index=4,
        expected_heads=3,
        width=128,
        device="cpu",
    )
    torch.manual_seed(999_999)
    second = right_rht_signs(
        layer_index=4,
        expected_heads=3,
        width=128,
        device="cpu",
    )
    digest = hashlib.sha256(first.to(torch.int8).numpy().tobytes()).hexdigest()

    assert torch.equal(first, second)
    assert digest == "3cc14ffaf1ad8de3d77a1d277cb027c3dee9360429c0746232780acc55d42f55"


def test_layers_and_heads_receive_different_sign_vectors() -> None:
    first_layer = right_rht_signs(
        layer_index=2,
        expected_heads=4,
        width=128,
        device="cpu",
    )
    second_layer = right_rht_signs(
        layer_index=3,
        expected_heads=4,
        width=128,
        device="cpu",
    )

    assert all(
        not torch.equal(first_layer[0, left, 0], first_layer[0, right, 0])
        for left in range(4)
        for right in range(left + 1, 4)
    )
    assert not torch.equal(first_layer, second_layer)


def test_normalized_encode_preserves_every_row_norm() -> None:
    generator = torch.Generator().manual_seed(23)
    state = torch.randn((2, 4, 7, 128), generator=generator)

    encoded = right_rht_encode(state, layer_index=5, expected_heads=4)

    torch.testing.assert_close(
        encoded.square().sum(dim=-1),
        state.square().sum(dim=-1),
        rtol=2e-6,
        atol=2e-5,
    )


def test_decode_preserves_quantization_error_energy_per_row() -> None:
    generator = torch.Generator().manual_seed(41)
    state = torch.randn((2, 3, 6, 128), generator=generator)
    encoded = right_rht_encode(state, layer_index=9, expected_heads=3)
    quantized = _q4_absmax(encoded)
    restored = right_rht_decode(quantized, layer_index=9, expected_heads=3)

    source_error = (restored - state).square().sum(dim=-1)
    transformed_error = (quantized - encoded).square().sum(dim=-1)

    torch.testing.assert_close(source_error, transformed_error, rtol=8e-6, atol=2e-5)


def test_zero_state_is_exact_and_batch_rows_broadcast_independently() -> None:
    state = torch.zeros((3, 2, 5, 64), dtype=torch.float32)

    encoded = right_rht_encode(state, layer_index=1, expected_heads=2)
    restored = right_rht_decode(encoded, layer_index=1, expected_heads=2)

    assert torch.equal(encoded, state)
    assert torch.equal(restored, state)


def test_hadamard_aligned_vector_maps_to_one_coordinate() -> None:
    spike = torch.zeros((1, 1, 1, 128), dtype=torch.float32)
    spike[..., 37] = 3.25
    aligned = right_rht_decode(spike, layer_index=6, expected_heads=1)

    encoded = right_rht_encode(aligned, layer_index=6, expected_heads=1)

    torch.testing.assert_close(encoded, spike, rtol=2e-6, atol=1e-6)


def test_exact_grid_adversary_documents_that_rotation_is_not_always_better() -> None:
    values = torch.arange(128, dtype=torch.float32).remainder(15).sub(7)
    state = values.reshape(1, 1, 1, 128)
    baseline = _q4_absmax(state)
    encoded = right_rht_encode(state, layer_index=0, expected_heads=1)
    restored = right_rht_decode(
        _q4_absmax(encoded),
        layer_index=0,
        expected_heads=1,
    )

    assert torch.equal(baseline, state)
    assert float((restored - state).square().sum().item()) > 0.0


@pytest.mark.parametrize(
    ("state", "expected_heads", "message"),
    [
        (torch.ones((1, 2, 128)), 2, "shape"),
        (torch.ones((1, 2, 3, 96)), 2, "power of two"),
        (torch.ones((1, 2, 3, 128)), 3, "head dimension"),
        (torch.ones((1, 2, 3, 128), dtype=torch.int64), 2, "float16"),
        (torch.full((1, 2, 3, 128), float("nan")), 2, "finite"),
        (torch.full((1, 2, 3, 128), float("inf")), 2, "finite"),
        (torch.empty((0, 2, 3, 128)), 2, "empty"),
    ],
)
def test_encode_fails_closed_on_invalid_state(
    state: torch.Tensor,
    expected_heads: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        right_rht_encode(
            state,
            layer_index=0,
            expected_heads=expected_heads,
        )


@pytest.mark.parametrize(
    ("layer_index", "expected_heads", "error", "message"),
    [
        (-1, 2, ValueError, "non-negative"),
        (True, 2, TypeError, "integer"),
        (0, 0, ValueError, "positive"),
        (0, True, TypeError, "integer"),
    ],
)
def test_encode_rejects_invalid_identity(
    layer_index: int,
    expected_heads: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        right_rht_encode(
            torch.ones((1, 2, 3, 128)),
            layer_index=layer_index,
            expected_heads=expected_heads,
        )


def test_decode_can_restore_requested_recurrent_state_dtype() -> None:
    state = torch.randn((1, 2, 3, 128), dtype=torch.float32).to(torch.bfloat16)

    encoded = right_rht_encode(
        state,
        layer_index=8,
        expected_heads=2,
        output_dtype=torch.float32,
    )
    restored = right_rht_decode(
        encoded,
        layer_index=8,
        expected_heads=2,
        output_dtype=torch.bfloat16,
    )

    assert encoded.dtype == torch.float32
    assert restored.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_sign_schedule_and_inverse_bound() -> None:
    generator = torch.Generator().manual_seed(71)
    state_cpu = torch.randn((2, 3, 4, 128), generator=generator)
    state_cuda = state_cpu.cuda()

    cpu_signs = right_rht_signs(
        layer_index=11,
        expected_heads=3,
        width=128,
        device="cpu",
    )
    cuda_signs = right_rht_signs(
        layer_index=11,
        expected_heads=3,
        width=128,
        device="cuda",
    )
    encoded = right_rht_encode(state_cuda, layer_index=11, expected_heads=3)
    restored = right_rht_decode(encoded, layer_index=11, expected_heads=3)
    relative_l2 = torch.linalg.vector_norm(restored - state_cuda) / torch.linalg.vector_norm(
        state_cuda
    )

    assert torch.equal(cpu_signs, cuda_signs.cpu())
    assert float(relative_l2.item()) < 3e-7
