from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from recurquant.mixed_quantization import quantize_pack_mixed
from recurquant.quantization import QuantizationSpec, quantize_dequantize


def _per_group_qdq_reference(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    low_spec: QuantizationSpec,
    high_spec: QuantizationSpec,
) -> torch.Tensor:
    flattened_size = math.prod(tensor.shape[-low_spec.flatten_last_dims :])
    rows = tensor.detach().to(torch.float32).reshape(-1, flattened_size)
    groups_per_row = math.ceil(flattened_size / low_spec.group_size)
    padded_size = groups_per_row * low_spec.group_size
    if padded_size != flattened_size:
        rows = torch.nn.functional.pad(rows, (0, padded_size - flattened_size))
    groups = rows.reshape(-1, low_spec.group_size)
    flat_mask = mask.reshape(-1)

    restored_groups = []
    for index, group in enumerate(groups):
        selected = high_spec if flat_mask[index] else low_spec
        restored_groups.append(
            quantize_dequantize(
                group,
                replace(selected, flatten_last_dims=1),
            ).tensor
        )
    restored = torch.stack(restored_groups).reshape(-1, padded_size)[:, :flattened_size]
    return restored.reshape(tensor.shape).to(tensor.dtype)


def _specs(**kwargs: object) -> tuple[QuantizationSpec, QuantizationSpec]:
    return QuantizationSpec(bits=4, **kwargs), QuantizationSpec(bits=8, **kwargs)


def test_mixed_pack_matches_independent_per_group_qdq() -> None:
    state = torch.linspace(-2.7, 3.1, 60, dtype=torch.float32).reshape(2, 2, 3, 5)
    low_spec, high_spec = _specs(group_size=4)
    mask = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, True],
            [False, False, True, True],
            [True, True, False, False],
        ]
    )

    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    reference = _per_group_qdq_reference(state, mask, low_spec, high_spec)

    assert torch.equal(packed.dequantize(), reference)
    assert torch.equal(packed.high_precision_mask(), mask)
    assert packed.low_payload.dtype == torch.uint8
    assert packed.high_payload.dtype == torch.int8
    assert packed.scales.dtype == torch.float16
    assert packed.low_precision_groups == 8
    assert packed.high_precision_groups == 8


def test_qwen35_row_layout_hits_v02_exact_byte_target() -> None:
    total_groups = 18 * 16 * 128
    promoted_groups = 1_976
    state = torch.zeros((1, 18 * 16, 128, 128), dtype=torch.float32)
    mask = torch.zeros(total_groups, dtype=torch.bool)
    mask[:promoted_groups] = True
    low_spec, high_spec = _specs(group_size=128)

    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )

    assert packed.total_groups == 36_864
    assert packed.high_precision_groups == 1_976
    assert packed.low_precision_groups == 34_888
    assert packed.payload_bytes == 2_485_760
    assert packed.scale_bytes == 73_728
    assert packed.mask_bytes == 4_608
    assert packed.storage_bytes == 2_564_096


def test_odd_group_size_and_ninth_mask_bit_have_exact_storage() -> None:
    state = torch.linspace(-1.3, 2.1, 25)
    mask = torch.tensor([True, False, True, False, False, False, False, False, True])
    low_spec, high_spec = _specs(group_size=3, flatten_last_dims=1)

    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    reference = _per_group_qdq_reference(state, mask, low_spec, high_spec)

    assert packed.precision_mask.tolist() == [0b00000101, 0b00000001]
    assert packed.low_payload.shape == (6, 2)
    assert packed.high_payload.shape == (3, 3)
    assert packed.payload_bytes == 21
    assert packed.scale_bytes == 18
    assert packed.mask_bytes == 2
    assert packed.storage_bytes == 41
    assert torch.equal(packed.dequantize(), reference)


def test_stochastic_mixed_pack_is_seeded_and_deterministic() -> None:
    state = torch.linspace(-1.0, 1.0, 77)
    mask = torch.tensor([False, True, False, True, True, False, True])
    low_spec, high_spec = _specs(
        group_size=12,
        flatten_last_dims=1,
        rounding="stochastic",
        seed=71,
    )

    first = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    second = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )

    assert torch.equal(first.low_payload, second.low_payload)
    assert torch.equal(first.high_payload, second.high_payload)
    assert torch.equal(first.scales, second.scales)
    assert torch.equal(first.dequantize(), second.dequantize())


def test_batch_reorder_reuses_integer_codes_and_supports_duplicates() -> None:
    generator = torch.Generator().manual_seed(103)
    state = torch.randn((3, 2, 3, 5), generator=generator).to(torch.bfloat16)
    low_spec, high_spec = _specs(group_size=4)
    mask = (torch.arange(24) % 3 == 1).reshape(6, 4)
    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )
    before = packed.dequantize()
    before_codes = packed._integer_groups().reshape(3, 8, 4)
    beam_idx = torch.tensor([2, 2, 0], dtype=torch.long)

    reordered = packed.reorder_batch(beam_idx)

    assert torch.equal(reordered.dequantize(), before.index_select(0, beam_idx))
    assert torch.equal(
        reordered._integer_groups().reshape(3, 8, 4),
        before_codes.index_select(0, beam_idx),
    )
    assert reordered.original_shape == state.shape


@pytest.mark.parametrize("high_value", [False, True])
def test_all_groups_can_use_one_payload_width(high_value: bool) -> None:
    state = torch.linspace(-2.0, 2.0, 16)
    mask = torch.full((2,), high_value, dtype=torch.bool)
    low_spec, high_spec = _specs(group_size=8, flatten_last_dims=1)

    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )

    assert packed.high_precision_groups == (2 if high_value else 0)
    assert packed.low_precision_groups == (0 if high_value else 2)
    assert torch.equal(
        packed.dequantize(),
        _per_group_qdq_reference(state, mask, low_spec, high_spec),
    )


@pytest.mark.parametrize(
    ("low_spec", "high_spec", "message"),
    [
        (QuantizationSpec(bits=6), QuantizationSpec(bits=8), "low INT4"),
        (QuantizationSpec(bits=4), QuantizationSpec(bits=6), "high INT8"),
        (
            QuantizationSpec(bits=4, group_size=8),
            QuantizationSpec(bits=8, group_size=16),
            "group_size",
        ),
        (
            QuantizationSpec(bits=4, scale_bits=16),
            QuantizationSpec(bits=8, scale_bits=32),
            "scale_bits",
        ),
        (
            QuantizationSpec(bits=4, seed=1),
            QuantizationSpec(bits=8, seed=2),
            "seed",
        ),
    ],
)
def test_invalid_mixed_specs_are_rejected(
    low_spec: QuantizationSpec,
    high_spec: QuantizationSpec,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quantize_pack_mixed(
            torch.ones(128),
            torch.zeros(1, dtype=torch.bool),
            low_spec=low_spec,
            high_spec=high_spec,
        )


def test_invalid_masks_are_rejected() -> None:
    state = torch.ones((2, 8))
    low_spec, high_spec = _specs(group_size=4, flatten_last_dims=1)

    with pytest.raises(TypeError, match="torch.bool"):
        quantize_pack_mixed(
            state,
            torch.zeros((2, 2), dtype=torch.int8),
            low_spec=low_spec,
            high_spec=high_spec,
        )
    with pytest.raises(ValueError, match="must have shape"):
        quantize_pack_mixed(
            state,
            torch.zeros((1, 4), dtype=torch.bool),
            low_spec=low_spec,
            high_spec=high_spec,
        )


def test_invalid_reorder_indices_are_rejected() -> None:
    low_spec, high_spec = _specs(group_size=4, flatten_last_dims=1)
    packed = quantize_pack_mixed(
        torch.ones((2, 8)),
        torch.zeros((2, 2), dtype=torch.bool),
        low_spec=low_spec,
        high_spec=high_spec,
    )

    with pytest.raises(TypeError, match="int32 or int64"):
        packed.reorder_batch(torch.tensor([0.0, 1.0]))
    with pytest.raises(IndexError, match="out-of-range"):
        packed.reorder_batch(torch.tensor([2], dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_to_cuda_preserves_storage_and_reorder_exactly() -> None:
    state = torch.linspace(-2.0, 2.0, 48).reshape(3, 2, 8)
    mask = (torch.arange(12) % 2 == 0).reshape(6, 2)
    low_spec, high_spec = _specs(group_size=4, flatten_last_dims=1)
    packed = quantize_pack_mixed(
        state,
        mask,
        low_spec=low_spec,
        high_spec=high_spec,
    )

    on_cuda = packed.to("cuda")
    beam_idx = torch.tensor([2, 0, 2], dtype=torch.long, device="cuda")
    reordered = on_cuda.reorder_batch(beam_idx)

    assert on_cuda.low_payload.device.type == "cuda"
    assert on_cuda.high_payload.device.type == "cuda"
    assert on_cuda.scales.device.type == "cuda"
    assert on_cuda.precision_mask.device.type == "cuda"
    assert on_cuda.storage_bytes == packed.storage_bytes
    assert torch.equal(on_cuda.dequantize().cpu(), packed.dequantize())
    assert torch.equal(
        reordered.dequantize().cpu(),
        packed.dequantize().index_select(0, beam_idx.cpu()),
    )
