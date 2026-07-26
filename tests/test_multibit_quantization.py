from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from recurquant.multibit_quantization import (
    INT6_PRECISION_CODE,
    _pack_int6_groups,
    _pack_precision_codes,
    _unpack_int6_groups,
    _unpack_precision_codes,
    quantize_pack_multibit,
)
from recurquant.quantization import QuantizationSpec


def _specs(
    *,
    group_size: int = 128,
    scale_bits: int = 16,
    flatten_last_dims: int = 2,
    rounding: str = "nearest",
    seed: int = 2339,
) -> tuple[QuantizationSpec, QuantizationSpec, QuantizationSpec]:
    common = {
        "group_size": group_size,
        "scale_bits": scale_bits,
        "flatten_last_dims": flatten_last_dims,
        "rounding": rounding,
        "seed": seed,
    }
    return (
        QuantizationSpec(bits=4, **common),
        QuantizationSpec(bits=6, **common),
        QuantizationSpec(bits=8, **common),
    )


def _independent_nearest_reference(
    tensor: torch.Tensor,
    precision_codes: torch.Tensor,
    spec: QuantizationSpec,
) -> torch.Tensor:
    working = tensor.detach().to(torch.float32)
    flattened_size = math.prod(working.shape[-spec.flatten_last_dims :])
    rows = working.reshape(-1, flattened_size)
    groups_per_row = math.ceil(flattened_size / spec.group_size)
    padded_size = groups_per_row * spec.group_size
    if padded_size != flattened_size:
        rows = torch.nn.functional.pad(rows, (0, padded_size - flattened_size))
    groups = rows.reshape(-1, spec.group_size)

    qmax = torch.tensor((7.0, 31.0, 127.0))[precision_codes.reshape(-1).long()]
    absmax = groups.abs().amax(dim=1)
    ideal_scales = torch.where(
        absmax > spec.epsilon,
        absmax / qmax,
        torch.ones_like(absmax),
    )
    scale_dtype = torch.float16 if spec.scale_bits == 16 else torch.float32
    if scale_dtype == torch.float16:
        ideal_scales = ideal_scales.clamp(
            min=2.0**-24,
            max=torch.finfo(torch.float16).max,
        )
    scales = ideal_scales.to(scale_dtype).to(torch.float32)
    integer_codes = torch.round(groups / scales.unsqueeze(1))
    integer_codes = torch.minimum(
        torch.maximum(integer_codes, -qmax.unsqueeze(1)),
        qmax.unsqueeze(1),
    )
    restored = (integer_codes * scales.unsqueeze(1)).reshape(-1, padded_size)
    restored = restored[:, :flattened_size]
    return restored.reshape(tensor.shape).to(tensor.dtype)


def _packed_validation_fixture():
    state = torch.linspace(-1.0, 1.0, 12).reshape(1, 12)
    precision = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    return quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )


def test_int6_exhaustive_symmetric_code_roundtrip() -> None:
    values = torch.arange(-31, 32, dtype=torch.int16)
    codes = values.unsqueeze(1).repeat(1, 4)

    payload = _pack_int6_groups(codes)
    restored = _unpack_int6_groups(payload, group_size=4)

    assert payload.shape == (63, 3)
    assert payload.dtype == torch.uint8
    assert torch.equal(restored, codes)


def test_int6_uses_little_endian_four_code_three_byte_layout() -> None:
    codes = torch.tensor(
        [
            [0, 1, 2, 3],
            [-1, -2, -3, -31],
        ],
        dtype=torch.int16,
    )
    unsigned = torch.bitwise_and(codes.to(torch.int64), 0x3F)
    words = (
        unsigned[:, 0]
        | (unsigned[:, 1] << 6)
        | (unsigned[:, 2] << 12)
        | (unsigned[:, 3] << 18)
    )
    independent_payload = torch.stack(
        (
            torch.bitwise_and(words, 0xFF),
            torch.bitwise_and(words >> 8, 0xFF),
            torch.bitwise_and(words >> 16, 0xFF),
        ),
        dim=1,
    ).to(torch.uint8)

    assert torch.equal(_pack_int6_groups(codes), independent_payload)


def test_group_size_128_int6_payload_is_exactly_96_bytes() -> None:
    codes = torch.arange(128, dtype=torch.int16).remainder(63) - 31

    payload = _pack_int6_groups(codes.reshape(1, 128))

    assert payload.shape == (1, 96)
    assert payload.numel() * payload.element_size() == 96
    assert torch.equal(_unpack_int6_groups(payload, 128), codes.reshape(1, 128))


def test_precision_codes_have_canonical_two_bit_encoding() -> None:
    precision = torch.tensor([0, 1, 2, 0, 2, 1, 0, 2], dtype=torch.uint8)

    first = _pack_precision_codes(precision)
    second = _pack_precision_codes(precision.clone())

    assert first.tolist() == [0x24, 0x86]
    assert torch.equal(first, second)
    assert torch.equal(_unpack_precision_codes(first, 8), precision)


def test_mixed_qdq_matches_independent_reference_exactly() -> None:
    state = torch.tensor(
        [
            [
                [0.0, -0.23, 0.91, -1.7, 3.2],
                [0.07, -0.51, 0.34, 1.2, -2.8],
                [0.03, -0.11, 0.57, -0.8, 2.1],
            ],
            [
                [-0.4, 0.0, 0.2, 0.8, -1.4],
                [2.7, -1.3, 0.33, -0.09, 0.72],
                [-3.1, 1.8, -0.63, 0.14, 0.01],
            ],
        ],
        dtype=torch.float32,
    )
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    precision = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 2],
            [1, 0],
            [2, 1],
        ],
        dtype=torch.uint8,
    )

    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )
    reference = _independent_nearest_reference(state, precision, int4_spec)

    assert torch.equal(packed.dequantize(), reference)
    assert torch.equal(packed.precision_codes(), precision)
    assert packed.int4_payload.dtype == torch.uint8
    assert packed.int6_payload.dtype == torch.uint8
    assert packed.int8_payload.dtype == torch.int8
    assert packed.scales.dtype == torch.float16
    assert (packed.int4_groups, packed.int6_groups, packed.int8_groups) == (4, 4, 4)


def test_qwen_layout_hits_old_target_with_corrected_two_bit_metadata() -> None:
    total_groups = 18 * 16 * 128
    int6_groups = 3_808
    state = torch.zeros((1, 18 * 16, 128, 128), dtype=torch.float32)
    precision = torch.zeros(total_groups, dtype=torch.uint8)
    precision[:int6_groups] = INT6_PRECISION_CODE
    int4_spec, int6_spec, int8_spec = _specs(group_size=128)

    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )

    assert packed.total_groups == 36_864
    assert packed.int4_groups == 33_056
    assert packed.int6_groups == 3_808
    assert packed.int8_groups == 0
    assert packed.payload_bytes == 2_481_152
    assert packed.scale_bytes == 73_728
    assert packed.precision_code_bytes == 9_216
    assert packed.storage_bytes == 2_564_096


def test_exact_storage_with_all_three_widths_and_fp32_scales() -> None:
    state = torch.linspace(-2.0, 2.0, 48)
    precision = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.uint8,
    )
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        scale_bits=32,
        flatten_last_dims=1,
    )

    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )

    assert packed.int4_payload.shape == (4, 2)
    assert packed.int6_payload.shape == (4, 3)
    assert packed.int8_payload.shape == (4, 4)
    assert packed.payload_bytes == 36
    assert packed.scale_bytes == 48
    assert packed.precision_code_bytes == 3
    assert packed.storage_bytes == 87
    assert packed.scales.dtype == torch.float32


def test_stochastic_quantization_is_seeded_and_deterministic() -> None:
    state = torch.linspace(-1.13, 0.91, 400)
    precision = torch.arange(100, dtype=torch.uint8).remainder(3)
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
        rounding="stochastic",
        seed=71,
    )

    first = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )
    second = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )
    different_seed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=replace(int4_spec, seed=72),
        int6_spec=replace(int6_spec, seed=72),
        int8_spec=replace(int8_spec, seed=72),
    )

    assert torch.equal(first.int4_payload, second.int4_payload)
    assert torch.equal(first.int6_payload, second.int6_payload)
    assert torch.equal(first.int8_payload, second.int8_payload)
    assert torch.equal(first.scales, second.scales)
    assert torch.equal(first.packed_precision_codes, second.packed_precision_codes)
    assert not torch.equal(first.dequantize(), different_seed.dequantize())


def test_batch_reorder_preserves_integer_codes_scales_and_duplicates() -> None:
    generator = torch.Generator().manual_seed(103)
    state = torch.randn((3, 2, 3, 5), generator=generator).to(torch.bfloat16)
    precision = torch.arange(24, dtype=torch.uint8).remainder(3).reshape(6, 4)
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=2,
    )
    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )
    before = packed.dequantize()
    before_codes = packed._integer_groups().reshape(3, 8, 4)
    before_precision = packed.precision_codes().reshape(3, 8)
    beam_idx = torch.tensor([2, 2, 0], dtype=torch.long)

    reordered = packed.reorder_batch(beam_idx)

    assert torch.equal(reordered.dequantize(), before.index_select(0, beam_idx))
    assert torch.equal(
        reordered._integer_groups().reshape(3, 8, 4),
        before_codes.index_select(0, beam_idx),
    )
    assert torch.equal(
        reordered.precision_codes().reshape(3, 8),
        before_precision.index_select(0, beam_idx),
    )
    assert reordered.original_shape == state.shape


def test_to_cpu_preserves_every_resident_tensor_and_storage() -> None:
    state = torch.linspace(-2.0, 2.0, 48).reshape(3, 2, 8)
    precision = torch.arange(12, dtype=torch.uint8).remainder(3).reshape(6, 2)
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )

    transferred = packed.to(torch.device("cpu"))

    assert transferred.int4_payload.device.type == "cpu"
    assert transferred.int6_payload.device.type == "cpu"
    assert transferred.int8_payload.device.type == "cpu"
    assert transferred.scales.device.type == "cpu"
    assert transferred.packed_precision_codes.device.type == "cpu"
    assert transferred.storage_bytes == packed.storage_bytes
    assert torch.equal(transferred.dequantize(), packed.dequantize())


def test_empty_batch_reorder_remains_a_valid_packed_object() -> None:
    packed = _packed_validation_fixture()

    reordered = packed.reorder_batch(torch.empty(0, dtype=torch.long))

    assert reordered.original_shape == (0, 12)
    assert reordered.rows == 0
    assert reordered.total_groups == 0
    assert reordered.storage_bytes == 0
    assert reordered.dequantize().shape == (0, 12)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"flattened_size": 13}, "flattened_size"),
        ({"padded_size": 16}, "padded_size"),
        ({"rows": 2}, "rows"),
        ({"groups_per_row": 4}, "groups_per_row"),
        ({"original_shape": (1, 16)}, "flattened_size"),
        ({"original_shape": [1, 12]}, "original_shape must be a tuple"),
        ({"original_shape": (1, 0)}, "flattened.*must be positive"),
        ({"original_dtype": torch.int32}, "floating-point dtype"),
    ],
)
def test_direct_construction_rejects_inconsistent_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    packed = _packed_validation_fixture()

    with pytest.raises((TypeError, ValueError), match=message):
        replace(packed, **changes)


def test_direct_construction_rejects_invalid_specs() -> None:
    packed = _packed_validation_fixture()

    with pytest.raises(TypeError, match="int4_spec.*QuantizationSpec"):
        replace(packed, int4_spec=object())
    with pytest.raises(ValueError, match="mismatched fields: group_size"):
        replace(packed, int6_spec=replace(packed.int6_spec, group_size=8))


def test_direct_construction_rejects_invalid_scales() -> None:
    packed = _packed_validation_fixture()

    with pytest.raises(TypeError, match="one-dimensional torch.float16"):
        replace(packed, scales=packed.scales.to(torch.float32))
    with pytest.raises(ValueError, match="one value per group"):
        replace(packed, scales=packed.scales[:-1].clone())

    nonfinite = packed.scales.clone()
    nonfinite[0] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        replace(packed, scales=nonfinite)

    nonpositive = packed.scales.clone()
    nonpositive[0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        replace(packed, scales=nonpositive)

    backing = torch.empty(
        packed.scales.numel() * 2,
        dtype=packed.scales.dtype,
    )
    noncontiguous = backing[::2]
    noncontiguous.copy_(packed.scales)
    with pytest.raises(ValueError, match="scales must be contiguous"):
        replace(packed, scales=noncontiguous)


def test_direct_construction_rejects_invalid_precision_stream() -> None:
    packed = _packed_validation_fixture()

    with pytest.raises(TypeError, match="packed precision codes.*torch.uint8"):
        replace(
            packed,
            packed_precision_codes=packed.packed_precision_codes.to(torch.int16),
        )
    with pytest.raises(ValueError, match="must contain 1 bytes"):
        replace(
            packed,
            packed_precision_codes=packed.packed_precision_codes[:0].clone(),
        )

    reserved = packed.packed_precision_codes.clone()
    reserved[0] = torch.bitwise_or(
        torch.bitwise_and(reserved[0], 0xFC),
        torch.tensor(3, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="reserved precision code 3"):
        replace(packed, packed_precision_codes=reserved)

    noncanonical_padding = packed.packed_precision_codes.clone()
    noncanonical_padding[0] = torch.bitwise_or(
        noncanonical_padding[0],
        torch.tensor(0x40, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="padding bits must be zero"):
        replace(packed, packed_precision_codes=noncanonical_padding)


@pytest.mark.parametrize(
    ("field", "dtype", "message"),
    [
        ("int4_payload", torch.int8, "int4_payload.*torch.uint8"),
        ("int6_payload", torch.int8, "int6_payload.*torch.uint8"),
        ("int8_payload", torch.uint8, "int8_payload.*torch.int8"),
    ],
)
def test_direct_construction_rejects_payload_dtypes(
    field: str,
    dtype: torch.dtype,
    message: str,
) -> None:
    packed = _packed_validation_fixture()

    with pytest.raises(TypeError, match=message):
        replace(packed, **{field: getattr(packed, field).to(dtype)})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("int4_payload", "int4_payload must have shape"),
        ("int6_payload", "int6_payload must have shape"),
        ("int8_payload", "int8_payload must have shape"),
    ],
)
def test_direct_construction_rejects_payload_counts(
    field: str,
    message: str,
) -> None:
    packed = _packed_validation_fixture()
    payload = getattr(packed, field)

    with pytest.raises(ValueError, match=message):
        replace(packed, **{field: payload[:0].clone()})


def test_direct_construction_rejects_payload_row_widths_and_devices() -> None:
    packed = _packed_validation_fixture()

    with pytest.raises(ValueError, match="int6_payload must have shape"):
        replace(packed, int6_payload=packed.int6_payload[:, :-1].clone())
    with pytest.raises(ValueError, match="all resident tensors must share one device"):
        replace(packed, int8_payload=packed.int8_payload.to("meta"))


def test_direct_construction_rejects_reserved_symmetric_payload_codes() -> None:
    packed = _packed_validation_fixture()

    invalid_int4 = packed.int4_payload.clone()
    invalid_int4[0, 0] = int(invalid_int4[0, 0].item()) & 0xF0 | 0x08
    with pytest.raises(ValueError, match="int4_payload.*-8"):
        replace(packed, int4_payload=invalid_int4)

    invalid_int6 = packed.int6_payload.clone()
    invalid_int6[0, 0] = int(invalid_int6[0, 0].item()) & 0xC0 | 0x20
    with pytest.raises(ValueError, match="INT6 payload.*-32"):
        replace(packed, int6_payload=invalid_int6)

    invalid_int8 = packed.int8_payload.clone()
    invalid_int8[0, 0] = -128
    with pytest.raises(ValueError, match="int8_payload.*-128"):
        replace(packed, int8_payload=invalid_int8)


@pytest.mark.parametrize(
    ("spec_index", "replacement", "message"),
    [
        (0, QuantizationSpec(bits=5), "int4_spec"),
        (1, QuantizationSpec(bits=5), "int6_spec"),
        (2, QuantizationSpec(bits=7), "int8_spec"),
        (1, QuantizationSpec(bits=6, group_size=64), "group_size"),
        (2, QuantizationSpec(bits=8, scale_bits=32), "scale_bits"),
    ],
)
def test_invalid_specs_are_rejected(
    spec_index: int,
    replacement: QuantizationSpec,
    message: str,
) -> None:
    specs = list(_specs())
    specs[spec_index] = replacement

    with pytest.raises(ValueError, match=message):
        quantize_pack_multibit(
            torch.ones(128),
            torch.zeros(1, dtype=torch.uint8),
            int4_spec=specs[0],
            int6_spec=specs[1],
            int8_spec=specs[2],
        )


def test_group_size_must_make_every_width_byte_representable() -> None:
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=2,
        flatten_last_dims=1,
    )

    with pytest.raises(ValueError, match="whole-byte payloads.*INT6"):
        quantize_pack_multibit(
            torch.ones(4),
            torch.zeros(2, dtype=torch.uint8),
            int4_spec=int4_spec,
            int6_spec=int6_spec,
            int8_spec=int8_spec,
        )


def test_invalid_precision_codes_are_rejected() -> None:
    state = torch.ones((2, 8))
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )

    with pytest.raises(TypeError, match="torch.uint8"):
        quantize_pack_multibit(
            state,
            torch.zeros((2, 2), dtype=torch.int64),
            int4_spec=int4_spec,
            int6_spec=int6_spec,
            int8_spec=int8_spec,
        )
    with pytest.raises(ValueError, match="0=INT4"):
        quantize_pack_multibit(
            state,
            torch.tensor([[0, 3], [1, 2]], dtype=torch.uint8),
            int4_spec=int4_spec,
            int6_spec=int6_spec,
            int8_spec=int8_spec,
        )
    with pytest.raises(ValueError, match="must have shape"):
        quantize_pack_multibit(
            state,
            torch.zeros((1, 4), dtype=torch.uint8),
            int4_spec=int4_spec,
            int6_spec=int6_spec,
            int8_spec=int8_spec,
        )
    with pytest.raises(ValueError, match="reserved precision code 3"):
        _unpack_precision_codes(torch.tensor([0b00000011], dtype=torch.uint8), 1)


def test_nonfloating_nonfinite_and_empty_sources_are_rejected() -> None:
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    kwargs = {
        "int4_spec": int4_spec,
        "int6_spec": int6_spec,
        "int8_spec": int8_spec,
    }

    with pytest.raises(TypeError, match="floating-point"):
        quantize_pack_multibit(
            torch.ones(4, dtype=torch.int32),
            torch.zeros(1, dtype=torch.uint8),
            **kwargs,
        )
    with pytest.raises(ValueError, match="finite"):
        quantize_pack_multibit(
            torch.tensor([1.0, float("nan"), 2.0, 3.0]),
            torch.zeros(1, dtype=torch.uint8),
            **kwargs,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        quantize_pack_multibit(
            torch.empty(0),
            torch.empty(0, dtype=torch.uint8),
            **kwargs,
        )


def test_invalid_int6_codes_and_payload_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[-31, 31\]"):
        _pack_int6_groups(torch.tensor([[-32, 0, 0, 0]], dtype=torch.int16))
    with pytest.raises(ValueError, match="whole number of bytes"):
        _pack_int6_groups(torch.zeros((1, 2), dtype=torch.int16))
    with pytest.raises(TypeError, match="two-dimensional"):
        _unpack_int6_groups(torch.zeros(3, dtype=torch.uint8), 4)
    with pytest.raises(ValueError, match="inconsistent"):
        _unpack_int6_groups(torch.zeros((1, 4), dtype=torch.uint8), 4)


def test_invalid_reorder_indices_are_rejected() -> None:
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    packed = quantize_pack_multibit(
        torch.ones((2, 8)),
        torch.zeros((2, 2), dtype=torch.uint8),
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )

    with pytest.raises(TypeError, match="int32 or int64"):
        packed.reorder_batch(torch.tensor([0.0, 1.0]))
    with pytest.raises(IndexError, match="out-of-range"):
        packed.reorder_batch(torch.tensor([2], dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_to_cuda_preserves_storage_dequantization_and_reorder() -> None:
    state = torch.linspace(-2.0, 2.0, 48).reshape(3, 2, 8)
    precision = torch.arange(12, dtype=torch.uint8).remainder(3).reshape(6, 2)
    int4_spec, int6_spec, int8_spec = _specs(
        group_size=4,
        flatten_last_dims=1,
    )
    packed = quantize_pack_multibit(
        state,
        precision,
        int4_spec=int4_spec,
        int6_spec=int6_spec,
        int8_spec=int8_spec,
    )

    on_cuda = packed.to("cuda")
    beam_idx = torch.tensor([2, 0, 2], dtype=torch.long, device="cuda")
    reordered = on_cuda.reorder_batch(beam_idx)

    assert on_cuda.storage_bytes == packed.storage_bytes
    assert torch.equal(on_cuda.dequantize().cpu(), packed.dequantize())
    assert torch.equal(
        reordered.dequantize().cpu(),
        packed.dequantize().index_select(0, beam_idx.cpu()),
    )


def test_public_package_exports_multibit_primitives() -> None:
    import recurquant

    assert recurquant.quantize_pack_multibit is quantize_pack_multibit
    assert recurquant.INT4_PRECISION_CODE == 0
    assert recurquant.INT6_PRECISION_CODE == 1
    assert recurquant.INT8_PRECISION_CODE == 2
    assert callable(recurquant.allocate_exact_multibit_codes)
    assert callable(recurquant.frozen_qwen35_multibit_step_budgets)
