from __future__ import annotations

import pytest
import torch

from recurquant.mixed_quantization import quantize_pack_mixed
from recurquant.quantization import QuantizationSpec, quantize_dequantize
from recurquant.rht import right_rht_decode, right_rht_encode

LOW = QuantizationSpec(bits=4, group_size=128, flatten_last_dims=2)
HIGH = QuantizationSpec(bits=8, group_size=128, flatten_last_dims=2)


def _independent_rht_qdq(
    state: torch.Tensor,
    mask: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    encoded = right_rht_encode(
        state,
        layer_index=layer_index,
        expected_heads=state.shape[1],
    )
    low = quantize_dequantize(encoded, LOW).tensor
    high = quantize_dequantize(encoded, HIGH).tensor
    selected = torch.where(mask.reshape(1, state.shape[1], state.shape[2], 1), high, low)
    return right_rht_decode(
        selected,
        layer_index=layer_index,
        expected_heads=state.shape[1],
        output_dtype=state.dtype,
    )


def test_rht_mixed_pack_matches_independent_transformed_qdq() -> None:
    generator = torch.Generator().manual_seed(91)
    state = torch.randn((1, 3, 7, 128), generator=generator, dtype=torch.float32)
    mask = torch.zeros((3, 7), dtype=torch.bool)
    mask[0, 2] = True
    mask[2, 4:] = True

    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=LOW,
        high_spec=HIGH,
        right_rht_layer_index=5,
        right_rht_expected_heads=3,
    )

    expected = _independent_rht_qdq(state, mask, layer_index=5)
    torch.testing.assert_close(packed.dequantize(), expected, rtol=0, atol=0)
    assert packed.high_precision_groups == int(mask.sum().item())
    assert packed.storage_bytes == (
        packed.payload_bytes + packed.scale_bytes + packed.mask_bytes
    )


def test_rht_changes_error_without_changing_physical_byte_contract() -> None:
    state = torch.randn((1, 2, 8, 128), generator=torch.Generator().manual_seed(101))
    mask = torch.rand((2, 8), generator=torch.Generator().manual_seed(102)) > 0.6

    baseline = quantize_pack_mixed(
        state,
        mask,
        low_spec=LOW,
        high_spec=HIGH,
    )
    transformed = quantize_pack_mixed(
        state,
        mask,
        low_spec=LOW,
        high_spec=HIGH,
        right_rht_layer_index=4,
        right_rht_expected_heads=2,
    )

    assert transformed.storage_bytes == baseline.storage_bytes
    assert transformed.payload_bytes == baseline.payload_bytes
    assert transformed.scale_bytes == baseline.scale_bytes
    assert transformed.mask_bytes == baseline.mask_bytes
    assert not torch.equal(transformed.dequantize(), baseline.dequantize())


def test_rht_metadata_survives_transfer_and_batch_reorder() -> None:
    state = torch.randn((3, 2, 4, 128), generator=torch.Generator().manual_seed(111))
    mask = torch.zeros((state.shape[0] * state.shape[1], state.shape[2]), dtype=torch.bool)
    mask[::2, 1] = True
    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=LOW,
        high_spec=HIGH,
        right_rht_layer_index=8,
        right_rht_expected_heads=2,
    )
    expected = packed.dequantize()

    moved = packed.to("cpu")
    assert moved.right_rht_layer_index == 8
    assert moved.right_rht_expected_heads == 2
    torch.testing.assert_close(moved.dequantize(), expected, rtol=0, atol=0)

    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    reordered = packed.reorder_batch(indices)
    assert reordered.right_rht_layer_index == 8
    assert reordered.right_rht_expected_heads == 2
    torch.testing.assert_close(
        reordered.dequantize(),
        expected.index_select(0, indices),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("layer_index", "expected_heads"),
    [
        (0, None),
        (None, 2),
    ],
)
def test_rht_configuration_must_be_complete(
    layer_index: int | None,
    expected_heads: int | None,
) -> None:
    state = torch.ones((1, 2, 3, 128))
    mask = torch.zeros((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="configured together"):
        quantize_pack_mixed(
            state,
            mask,
            low_spec=LOW,
            high_spec=HIGH,
            right_rht_layer_index=layer_index,
            right_rht_expected_heads=expected_heads,
        )


def test_rht_pack_rejects_wrong_head_geometry() -> None:
    state = torch.ones((1, 2, 3, 128))
    mask = torch.zeros((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="head dimension"):
        quantize_pack_mixed(
            state,
            mask,
            low_spec=LOW,
            high_spec=HIGH,
            right_rht_layer_index=0,
            right_rht_expected_heads=3,
        )


def test_rht_reorder_rejects_empty_batch() -> None:
    state = torch.ones((1, 2, 3, 128))
    packed = quantize_pack_mixed(
        state,
        torch.zeros((2, 3), dtype=torch.bool),
        low_spec=LOW,
        high_spec=HIGH,
        right_rht_layer_index=0,
        right_rht_expected_heads=2,
    )

    with pytest.raises(ValueError, match="empty batch"):
        packed.reorder_batch(torch.empty(0, dtype=torch.long))
