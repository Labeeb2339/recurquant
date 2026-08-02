from __future__ import annotations

import math

import pytest
import torch

from recurquant.quantization import QuantizationSpec, quantize_pack
from recurquant.triton_state import (
    TritonFp32Workspace,
    TritonGatedDeltaWorkspace,
    TritonPackedState,
    allocate_fp32_workspace,
    allocate_gated_delta_workspace,
    gated_delta_step,
    pack_triton_state,
    prepare_fp32_step,
    prepare_gated_delta_step,
    prepare_qwen35_decode_inputs,
    triton_is_available,
    unpack_triton_state,
    validate_packed_state,
)

CUDA_REQUIRED = pytest.mark.skipif(
    not triton_is_available(),
    reason="native Triton CUDA runtime is not available",
)


def _inputs(
    *,
    batch_size: int = 2,
    heads: int = 2,
    key_dim: int = 5,
    value_dim: int = 7,
    seed: int = 2339,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        (batch_size, heads, key_dim, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    query = torch.randn(
        (batch_size, heads, key_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    key = torch.randn(
        (batch_size, heads, key_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    query = torch.nn.functional.normalize(query, dim=-1).contiguous()
    key = torch.nn.functional.normalize(key, dim=-1).contiguous()
    value = torch.randn(
        (batch_size, heads, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    g = (
        -0.01
        - 0.2
        * torch.rand(
            (batch_size, heads),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous()
    beta = torch.rand(
        (batch_size, heads),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    return state, query, key, value, g, beta


def _reference_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decayed = state * torch.exp(g)[..., None, None]
    remembered = (decayed * key[..., :, None]).sum(dim=-2)
    update = beta[..., None] * (value - remembered)
    updated = decayed + key[..., :, None] * update[..., None, :]
    output = (updated * query[..., :, None]).sum(dim=-2)
    return updated, output, update


def test_int4_odd_group_size_is_rejected_explicitly_before_device_access() -> None:
    with pytest.raises(ValueError, match="even group_size"):
        pack_triton_state(
            torch.zeros((1, 1, 2, 3), dtype=torch.float32),
            bits=4,
            group_size=3,
        )


@CUDA_REQUIRED
@pytest.mark.parametrize(("bits", "group_size"), [(4, 8), (8, 7)])
def test_pack_uses_uint8_and_matches_canonical_layout(bits: int, group_size: int) -> None:
    state, *_ = _inputs(batch_size=1, heads=2, key_dim=3, value_dim=5)

    packed = pack_triton_state(state, bits=bits, group_size=group_size)
    canonical = quantize_pack(
        state,
        QuantizationSpec(bits=bits, group_size=group_size),
    )

    assert packed.payload.dtype == torch.uint8
    assert torch.equal(packed.payload, canonical.payload.view(torch.uint8))
    assert torch.equal(packed.scales, canonical.scales)
    assert torch.equal(unpack_triton_state(packed), canonical.dequantize())
    expected_codes = packed.rows * packed.padded_size
    expected_payload = expected_codes if bits == 8 else expected_codes // 2
    assert packed.payload.numel() == expected_payload
    assert packed.storage_bytes == canonical.storage_bytes


@CUDA_REQUIRED
@pytest.mark.parametrize(("bits", "group_size"), [(4, 8), (8, 7)])
def test_packed_step_matches_pytorch_and_canonical_repack(
    bits: int,
    group_size: int,
) -> None:
    source_state, query, key, value, g, beta = _inputs()
    packed = pack_triton_state(source_state, bits=bits, group_size=group_size)
    decoded = unpack_triton_state(packed)
    expected_state, expected_output, expected_update = _reference_step(
        decoded,
        query,
        key,
        value,
        g,
        beta,
    )

    result = gated_delta_step(packed, query, key, value, g, beta)
    expected_packed = quantize_pack(
        expected_state,
        QuantizationSpec(bits=bits, group_size=group_size),
    )

    assert result.state.payload.dtype == torch.uint8
    assert result.state.payload.data_ptr() != packed.payload.data_ptr()
    assert result.state.scales.data_ptr() != packed.scales.data_ptr()
    assert torch.equal(result.state.payload, expected_packed.payload.view(torch.uint8))
    assert torch.equal(result.state.scales, expected_packed.scales)
    torch.testing.assert_close(result.update, expected_update, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(result.output, expected_output, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        unpack_triton_state(result.state),
        expected_packed.dequantize(),
        rtol=0,
        atol=0,
    )


@CUDA_REQUIRED
def test_q4_tail_group_is_zero_padded_without_changing_valid_state() -> None:
    source_state, query, key, value, g, beta = _inputs(
        batch_size=1,
        heads=1,
        key_dim=3,
        value_dim=5,
    )
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    assert packed.flattened_size == 15
    assert packed.padded_size == 16

    result = gated_delta_step(packed, query, key, value, g, beta)

    assert result.state.payload.numel() == 8
    validate_packed_state(result.state)
    assert unpack_triton_state(result.state).shape == source_state.shape


@CUDA_REQUIRED
def test_prepared_step_is_deterministic_and_reuses_fixed_buffers() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    prepared = prepare_gated_delta_step(packed, query, key, value, g, beta)

    first = prepared.run()
    torch.cuda.synchronize()
    first_payload = first.state.payload.clone()
    first_scales = first.state.scales.clone()
    first_output = first.output.clone()
    first_update = first.update.clone()
    second = prepared.run()
    torch.cuda.synchronize()

    assert second.state.payload.data_ptr() == first.state.payload.data_ptr()
    assert second.state.scales.data_ptr() == first.state.scales.data_ptr()
    assert torch.equal(second.state.payload, first_payload)
    assert torch.equal(second.state.scales, first_scales)
    assert torch.equal(second.output, first_output)
    assert torch.equal(second.update, first_update)


@CUDA_REQUIRED
def test_step_accepts_singleton_scalar_axes_and_mixed_input_dtypes() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=8, group_size=7)
    query_bf16 = query.to(torch.bfloat16)
    key_bf16 = key.to(torch.bfloat16)
    value_bf16 = value.to(torch.bfloat16)

    result = gated_delta_step(
        packed,
        query_bf16,
        key_bf16,
        value_bf16,
        g.unsqueeze(-1),
        beta.unsqueeze(-1),
    )
    expected_state, expected_output, _ = _reference_step(
        unpack_triton_state(packed),
        query_bf16.float(),
        key_bf16.float(),
        value_bf16.float(),
        g,
        beta,
    )
    expected_packed = quantize_pack(
        expected_state,
        QuantizationSpec(bits=8, group_size=7),
    )

    assert torch.equal(result.state.payload, expected_packed.payload.view(torch.uint8))
    torch.testing.assert_close(result.output, expected_output, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
def test_preparation_fails_closed_on_bad_shapes_values_and_aliasing() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=4, group_size=8)

    with pytest.raises(ValueError, match="query shape"):
        prepare_gated_delta_step(packed, query[..., :-1].contiguous(), key, value, g, beta)
    with pytest.raises(ValueError, match=r"beta must lie in \[0, 1\]"):
        prepare_gated_delta_step(packed, query, key, value, g, beta + 2)
    with pytest.raises(ValueError, match="g must be non-positive"):
        prepare_gated_delta_step(packed, query, key, value, g.abs(), beta)
    tracked_query = query.detach().requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        prepare_gated_delta_step(packed, tracked_query, key, value, g, beta)
    bad_query = query.clone()
    bad_query[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="query must contain only finite"):
        prepare_gated_delta_step(packed, bad_query, key, value, g, beta)

    aliased = TritonGatedDeltaWorkspace(
        payload=packed.payload,
        scales=torch.empty_like(packed.scales),
        output=torch.empty_like(value, dtype=torch.float32),
        update=torch.empty_like(value, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_gated_delta_step(
            packed,
            query,
            key,
            value,
            g,
            beta,
            workspace=aliased,
        )


@CUDA_REQUIRED
def test_state_rejects_partially_overlapping_payload_and_scales() -> None:
    source_state, *_ = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    scale_bytes = packed.scales.numel() * packed.scales.element_size()
    scale_offset = packed.payload.numel() - 2
    backing = torch.empty(
        scale_offset + scale_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    payload = backing[: packed.payload.numel()]
    scales = backing[scale_offset:].view(torch.float16).reshape(packed.scales.shape)
    payload.copy_(packed.payload)
    scales.copy_(packed.scales)
    overlapping = TritonPackedState(
        payload=payload,
        scales=scales,
        shape=packed.shape,
        bits=packed.bits,
        group_size=packed.group_size,
    )

    with pytest.raises(ValueError, match="state.payload and state.scales must not overlap"):
        validate_packed_state(overlapping)


@CUDA_REQUIRED
def test_workspace_rejects_partially_overlapping_state_payload() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    backing = torch.empty(
        packed.payload.numel() + 1,
        dtype=torch.uint8,
        device="cuda",
    )
    input_payload = backing[: packed.payload.numel()]
    input_payload.copy_(packed.payload)
    overlapping_state = TritonPackedState(
        payload=input_payload,
        scales=packed.scales,
        shape=packed.shape,
        bits=packed.bits,
        group_size=packed.group_size,
    )
    workspace = allocate_gated_delta_workspace(packed)
    overlapping_workspace = TritonGatedDeltaWorkspace(
        payload=backing[1:],
        scales=workspace.scales,
        output=workspace.output,
        update=workspace.update,
    )

    with pytest.raises(ValueError, match="workspace.payload and state.payload must not overlap"):
        prepare_gated_delta_step(
            overlapping_state,
            query,
            key,
            value,
            g,
            beta,
            workspace=overlapping_workspace,
        )


@CUDA_REQUIRED
def test_workspace_rejects_partial_internal_overlap() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    workspace = allocate_gated_delta_workspace(packed)
    vector_elements = workspace.output.numel()
    backing = torch.empty(vector_elements + 1, dtype=torch.float32, device="cuda")
    overlapping_workspace = TritonGatedDeltaWorkspace(
        payload=workspace.payload,
        scales=workspace.scales,
        output=backing[:vector_elements].reshape(workspace.output.shape),
        update=backing[1:].reshape(workspace.update.shape),
    )

    with pytest.raises(
        ValueError,
        match="workspace.output and workspace.update must not overlap",
    ):
        prepare_gated_delta_step(
            packed,
            query,
            key,
            value,
            g,
            beta,
            workspace=overlapping_workspace,
        )


@CUDA_REQUIRED
def test_workspace_rejects_partial_overlap_with_recurrence_input() -> None:
    source_state, query, key, value, g, beta = _inputs(
        batch_size=1,
        key_dim=5,
        value_dim=5,
    )
    packed = pack_triton_state(source_state, bits=4, group_size=8)
    workspace = allocate_gated_delta_workspace(packed)
    elements = query.numel()
    backing = torch.empty(elements + 1, dtype=torch.float32, device="cuda")
    overlapping_query = backing[:elements].reshape(query.shape)
    overlapping_query.copy_(query)
    overlapping_workspace = TritonGatedDeltaWorkspace(
        payload=workspace.payload,
        scales=workspace.scales,
        output=backing[1:].reshape(workspace.output.shape),
        update=workspace.update,
    )

    with pytest.raises(ValueError, match="workspace.output and query must not overlap"):
        prepare_gated_delta_step(
            packed,
            overlapping_query,
            key,
            value,
            g,
            beta,
            workspace=overlapping_workspace,
        )


@CUDA_REQUIRED
def test_fp32_workspace_rejects_partial_overlap_with_input_state() -> None:
    source_state, query, key, value, g, beta = _inputs(batch_size=1)
    elements = source_state.numel()
    backing = torch.empty(elements + 1, dtype=torch.float32, device="cuda")
    overlapping_state = backing[:elements].reshape(source_state.shape)
    overlapping_state.copy_(source_state)
    workspace = allocate_fp32_workspace(source_state)
    overlapping_workspace = TritonFp32Workspace(
        state=backing[1:].reshape(source_state.shape),
        output=workspace.output,
        update=workspace.update,
    )

    with pytest.raises(ValueError, match="workspace.state and state must not overlap"):
        prepare_fp32_step(
            overlapping_state,
            query,
            key,
            value,
            g,
            beta,
            workspace=overlapping_workspace,
        )


@CUDA_REQUIRED
def test_payload_rejects_codes_outside_symmetric_quantizer_contract() -> None:
    source_state, *_ = _inputs(batch_size=1)
    q8 = pack_triton_state(source_state, bits=8, group_size=7)
    q8.payload[0] = 0x80
    with pytest.raises(ValueError, match="-128"):
        validate_packed_state(q8)

    q4 = pack_triton_state(source_state, bits=4, group_size=8)
    q4.payload[0] = 0x08
    with pytest.raises(ValueError, match="-8"):
        validate_packed_state(q4)


@CUDA_REQUIRED
def test_same_schedule_fp32_kernel_matches_pytorch_reference() -> None:
    source_state, query, key, value, g, beta = _inputs()
    expected_state, expected_output, expected_update = _reference_step(
        source_state,
        query,
        key,
        value,
        g,
        beta,
    )

    actual_state, actual_output, actual_update = prepare_fp32_step(
        source_state,
        query,
        key,
        value,
        g,
        beta,
        group_size=8,
    ).run()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_update, expected_update, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_qwen35_adapter_matches_pinned_preprocessing_and_output_contract(
    input_dtype: torch.dtype,
) -> None:
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    generator = torch.Generator(device="cuda").manual_seed(811)
    batch_size, sequence_length, heads, key_dim, value_dim = 1, 1, 2, 5, 7
    raw_query = torch.randn(
        (batch_size, sequence_length, heads, key_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(input_dtype)
    raw_key = torch.randn(
        raw_query.shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(input_dtype)
    # Exercise the pinned epsilon rather than only well-conditioned random rows.
    raw_query[0, 0, 0] = torch.tensor(
        [1.0e-4, -2.0e-4, 3.0e-4, -4.0e-4, 5.0e-4],
        device="cuda",
        dtype=input_dtype,
    )
    raw_key[0, 0, 0] = torch.tensor(
        [-5.0e-4, 4.0e-4, -3.0e-4, 2.0e-4, -1.0e-4],
        device="cuda",
        dtype=input_dtype,
    )
    raw_value = torch.randn(
        (batch_size, sequence_length, heads, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(input_dtype)
    g = (
        -0.01
        - 0.2
        * torch.rand(
            (batch_size, sequence_length, heads),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous()
    beta = torch.rand(
        (batch_size, sequence_length, heads),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(input_dtype)

    adapted = prepare_qwen35_decode_inputs(raw_query, raw_key, raw_value, g, beta)
    expected_query = modeling.l2norm(raw_query, dim=-1, eps=1.0e-6)
    expected_key = modeling.l2norm(raw_key, dim=-1, eps=1.0e-6)
    expected_query = (
        expected_query.transpose(1, 2).contiguous().to(torch.float32) * (1.0 / math.sqrt(key_dim))
    )[:, :, 0].contiguous()
    expected_key = expected_key.transpose(1, 2).contiguous().to(torch.float32)[:, :, 0].contiguous()
    expected_value = raw_value.transpose(1, 2).contiguous().to(torch.float32)[:, :, 0].contiguous()

    assert torch.equal(adapted.query, expected_query)
    assert torch.equal(adapted.key, expected_key)
    assert torch.equal(adapted.value, expected_value)
    assert adapted.output_dtype == input_dtype

    initial_state = torch.randn(
        (batch_size, heads, key_dim, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    packed = pack_triton_state(initial_state, bits=8, group_size=7)
    decoded = unpack_triton_state(packed)
    expected_output, _ = modeling.torch_recurrent_gated_delta_rule(
        raw_query,
        raw_key,
        raw_value,
        g,
        beta,
        decoded,
        True,
        True,
    )
    actual = gated_delta_step(
        packed,
        adapted.query,
        adapted.key,
        adapted.value,
        adapted.g,
        adapted.beta,
    )
    restored = adapted.restore_output(actual.output)

    assert tuple(restored.shape) == tuple(expected_output.shape)
    assert restored.dtype == input_dtype
    torch.testing.assert_close(restored, expected_output, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize("bits", [4, 8])
def test_qwen_sized_dimensions_remain_close_across_many_groups(bits: int) -> None:
    source_state, query, key, value, g, beta = _inputs(
        batch_size=2,
        heads=4,
        key_dim=128,
        value_dim=128,
        seed=2340,
    )
    packed = pack_triton_state(source_state, bits=bits, group_size=128)
    decoded = unpack_triton_state(packed)
    expected_state, expected_output, _ = _reference_step(
        decoded,
        query,
        key,
        value,
        g,
        beta,
    )
    expected_packed = quantize_pack(
        expected_state,
        QuantizationSpec(bits=bits, group_size=128),
    )

    result = gated_delta_step(packed, query, key, value, g, beta)
    torch.cuda.synchronize()
    actual_state = unpack_triton_state(result.state)
    expected_dequantized = expected_packed.dequantize()
    relative_l2 = torch.linalg.vector_norm(
        actual_state - expected_dequantized
    ) / torch.linalg.vector_norm(expected_dequantized).clamp_min(1e-12)

    assert float(relative_l2.item()) <= 1e-4
    torch.testing.assert_close(result.output, expected_output, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize("bits", [4, 8])
def test_multiple_decode_steps_ping_pong_between_packed_buffers(bits: int) -> None:
    source_state, *_ = _inputs(
        batch_size=1,
        heads=2,
        key_dim=5,
        value_dim=7,
        seed=71,
    )
    actual = pack_triton_state(source_state, bits=bits, group_size=8)
    expected = unpack_triton_state(actual)
    workspaces = [
        allocate_gated_delta_workspace(actual),
        allocate_gated_delta_workspace(actual),
    ]
    generator = torch.Generator(device="cuda").manual_seed(72)

    for step in range(6):
        query = torch.nn.functional.normalize(
            torch.randn(
                (1, 2, 5),
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            ),
            dim=-1,
        ).contiguous()
        key = torch.nn.functional.normalize(
            torch.randn(
                (1, 2, 5),
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            ),
            dim=-1,
        ).contiguous()
        value = torch.randn(
            (1, 2, 7),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        g = (-0.1 * torch.rand((1, 2), generator=generator, device="cuda")).contiguous()
        beta = torch.rand((1, 2), generator=generator, device="cuda").contiguous()
        expected_raw, expected_output, _ = _reference_step(
            expected,
            query,
            key,
            value,
            g,
            beta,
        )
        expected_packed = quantize_pack(
            expected_raw,
            QuantizationSpec(bits=bits, group_size=8),
        )
        expected = expected_packed.dequantize()

        result = gated_delta_step(
            actual,
            query,
            key,
            value,
            g,
            beta,
            workspace=workspaces[step % 2],
        )
        torch.cuda.synchronize()
        actual = result.state

        torch.testing.assert_close(
            unpack_triton_state(actual),
            expected,
            rtol=1e-4,
            atol=3e-3,
        )
        torch.testing.assert_close(result.output, expected_output, rtol=2e-5, atol=2e-5)
