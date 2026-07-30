"""Native Triton decode step for uniformly packed Gated DeltaNet state.

The implementation keeps the persistent ``[batch, heads, key, value]`` state
as a physical UINT8 payload with FP16 group scales.  A decode step uses two
Triton kernels:

1. stream the packed state to compute the remembered value, update vector, and
   output; and
2. apply the rank-one update and immediately re-quantize into another packed
   payload.

No full floating-point state is materialized.  The small ``[B, H, V]`` update
and output tensors are the only FP32 intermediates.  The layout matches
``quantize_pack`` with ``flatten_last_dims=2``; INT8 codes are exposed as
UINT8 two's-complement bytes so both supported formats have one public payload
dtype.

This is an optional CUDA path.  Importing the module does not require Triton,
but preparing or launching a step does.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .quantization import PackedQuantizedTensor, QuantizationSpec, quantize_pack

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised by the CPU-only test environment
    triton = None
    tl = None
    libdevice = None


_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_KEY_DIM = 256
_MAX_VALUE_DIM = 256
_MAX_GROUP_SIZE = 256
_QWEN35_L2NORM_EPS = 1.0e-6


@dataclass(frozen=True, slots=True)
class TritonPackedState:
    """Uniformly quantized state with a UINT8 two's-complement payload."""

    payload: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int, int, int]
    bits: int
    group_size: int

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def heads(self) -> int:
        return self.shape[1]

    @property
    def key_dim(self) -> int:
        return self.shape[2]

    @property
    def value_dim(self) -> int:
        return self.shape[3]

    @property
    def rows(self) -> int:
        return self.batch_size * self.heads

    @property
    def flattened_size(self) -> int:
        return self.key_dim * self.value_dim

    @property
    def groups_per_row(self) -> int:
        return math.ceil(self.flattened_size / self.group_size)

    @property
    def padded_size(self) -> int:
        return self.groups_per_row * self.group_size

    @property
    def storage_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size() + (
            self.scales.numel() * self.scales.element_size()
        )


@dataclass(frozen=True, slots=True)
class TritonGatedDeltaStepResult:
    """Packed next state plus the FP32 decode output and update vector.

    Every tensor borrows storage from the workspace used for the launch. A later
    launch with that workspace overwrites this result. Clone any tensor that
    must outlive the next launch.
    """

    state: TritonPackedState
    output: torch.Tensor
    update: torch.Tensor


@dataclass(slots=True)
class TritonGatedDeltaWorkspace:
    """Reusable output buffers for a packed decode step.

    A workspace supports only one in-flight launch. Do not use one workspace
    concurrently, from multiple CUDA streams, or while a previous result backed
    by it still needs to remain unchanged.
    """

    payload: torch.Tensor
    scales: torch.Tensor
    output: torch.Tensor
    update: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.payload, self.scales, self.output, self.update)
        )


@dataclass(slots=True)
class TritonFp32Workspace:
    """Reusable output buffers for the same-schedule FP32 reference.

    A workspace supports only one in-flight launch and one CUDA stream. Returned
    tensors borrow this storage and are overwritten by the next launch.
    """

    state: torch.Tensor
    output: torch.Tensor
    update: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.state, self.output, self.update)
        )


@dataclass(frozen=True, slots=True)
class Qwen35TritonDecodeInputs:
    """Exact single-token adapter from pinned Qwen3.5 inputs to this kernel.

    Qwen3.5 normalizes query and key in their input dtype with epsilon ``1e-6``,
    converts recurrence inputs to FP32, and scales only the query by
    ``key_dim**-0.5``. The packed kernel deliberately keeps that model-specific
    preprocessing outside its generic recurrence API.

    This adapter does not integrate a Qwen cache, RHT mixed precision,
    StateLease replay, or its c4/c5 controller.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    output_dtype: torch.dtype
    output_shape: tuple[int, int, int, int]

    def restore_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the pinned Qwen ``[B, 1, H, V]`` layout and input dtype."""

        batch_size, _, heads, value_dim = self.output_shape
        _require_cuda_contiguous(
            output,
            name="output",
            device=self.query.device,
            dtypes=(torch.float32,),
        )
        expected_shape = (batch_size, heads, value_dim)
        if tuple(output.shape) != expected_shape:
            raise ValueError(f"output shape must be {expected_shape}, got {tuple(output.shape)}")
        # Match the pinned implementation's transpose/contiguous/cast ordering.
        return output.unsqueeze(2).transpose(1, 2).contiguous().to(self.output_dtype)


def triton_is_available() -> bool:
    """Return whether both Triton and CUDA are available."""

    return triton is not None and torch.cuda.is_available()


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError(
            "Triton is not installed; install a CUDA-compatible Triton build "
            "(the tested Windows stack is Triton 3.6 with PyTorch 2.11/CUDA 12.8)"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


def _require_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("packed state shape must be [batch, heads, key_dim, value_dim]")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise TypeError("packed state dimensions must be integers")
    if any(size <= 0 for size in shape):
        raise ValueError("packed state dimensions must be positive")
    batch_size, heads, key_dim, value_dim = shape
    if key_dim > _MAX_KEY_DIM:
        raise ValueError(f"key_dim must be <= {_MAX_KEY_DIM} for this Triton kernel")
    if value_dim > _MAX_VALUE_DIM:
        raise ValueError(f"value_dim must be <= {_MAX_VALUE_DIM} for this Triton kernel")
    return batch_size, heads, key_dim, value_dim


def _require_quantization_contract(bits: int, group_size: int) -> None:
    if isinstance(bits, bool) or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if not 1 <= group_size <= _MAX_GROUP_SIZE:
        raise ValueError(f"group_size must be between 1 and {_MAX_GROUP_SIZE}")
    if bits == 4 and group_size % 2:
        raise ValueError(
            "INT4 requires an even group_size so independently launched groups "
            "never race while writing a shared nibble byte"
        )


def _expected_payload_elements(state: TritonPackedState) -> int:
    codes = state.rows * state.padded_size
    if state.bits == 8:
        return codes
    if codes % 2:
        raise ValueError("INT4 padded code count must be even")
    return codes // 2


def _require_cuda_contiguous(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device | None = None,
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> torch.device:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if dtypes is not None and tensor.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must use one of ({expected}), got {tensor.dtype}")
    return tensor.device


def _storage_byte_interval(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """Return the exact occupied byte interval for a validated contiguous tensor."""

    start = tensor.data_ptr()
    return tensor.device, start, start + tensor.numel() * tensor.element_size()


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two contiguous tensors occupy any common storage byte."""

    left_device, left_start, left_end = _storage_byte_interval(left)
    right_device, right_start, right_end = _storage_byte_interval(right)
    return left_device == right_device and left_start < right_end and right_start < left_end


def _require_no_overlap(
    left: torch.Tensor,
    *,
    left_name: str,
    right: torch.Tensor,
    right_name: str,
) -> None:
    if _tensors_overlap(left, right):
        raise ValueError(f"{left_name} and {right_name} must not overlap")


def _require_pairwise_no_overlap(
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> None:
    named = tuple(tensors)
    for index, (left_name, left) in enumerate(named):
        for right_name, right in named[index + 1 :]:
            _require_no_overlap(
                left,
                left_name=left_name,
                right=right,
                right_name=right_name,
            )


def _require_finite(tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in tensors:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values")


def _require_inference_only(tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    if not torch.is_grad_enabled():
        return
    tracked = [name for name, tensor in tensors if tensor.requires_grad]
    if tracked:
        raise RuntimeError(
            "Triton recurrent-state kernels are inference-only; disable autograd "
            f"before preparing tensors that require gradients: {', '.join(tracked)}"
        )


def prepare_qwen35_decode_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Qwen35TritonDecodeInputs:
    """Prepare one pinned Qwen3.5 decode token without changing its arithmetic.

    Raw Qwen tensors use ``[B, 1, H, D]`` for query/key/value and ``[B, 1, H]``
    for log-decay and beta. Query, key, and value must share an input dtype.
    Normalization occurs before the FP32 conversion, matching Transformers
    5.14.1 and the FLA contract. The returned output adapter restores the
    original query dtype.
    """

    named = (
        ("query", query),
        ("key", key),
        ("value", value),
        ("g", g),
        ("beta", beta),
    )
    device: torch.device | None = None
    for name, tensor in named:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
            expected = ", ".join(str(dtype) for dtype in _SUPPORTED_INPUT_DTYPES)
            raise TypeError(f"{name} must use one of ({expected}), got {tensor.dtype}")

    if query.ndim != 4:
        raise ValueError("query must have shape [batch, 1, heads, key_dim]")
    batch_size, sequence_length, heads, key_dim = query.shape
    if min(batch_size, sequence_length, heads, key_dim) <= 0:
        raise ValueError("Qwen3.5 decode dimensions must be positive")
    if sequence_length != 1:
        raise ValueError("the Qwen3.5 Triton adapter supports one decode token")
    if tuple(key.shape) != tuple(query.shape):
        raise ValueError(f"key shape must be {tuple(query.shape)}, got {tuple(key.shape)}")
    if value.ndim != 4 or tuple(value.shape[:3]) != (batch_size, 1, heads):
        raise ValueError(
            f"value must have shape {(batch_size, 1, heads, 'value_dim')}, got {tuple(value.shape)}"
        )
    value_dim = value.shape[-1]
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    scalar_shape = (batch_size, 1, heads)
    if tuple(g.shape) != scalar_shape:
        raise ValueError(f"g shape must be {scalar_shape}, got {tuple(g.shape)}")
    if tuple(beta.shape) != scalar_shape:
        raise ValueError(f"beta shape must be {scalar_shape}, got {tuple(beta.shape)}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("Qwen3.5 query, key, and value must use the same input dtype")

    _require_inference_only(named)
    _require_finite(named)
    if bool(((beta < 0) | (beta > 1)).any().item()):
        raise ValueError("beta must lie in [0, 1]")
    if bool((g > 0).any().item()):
        raise ValueError("g must be non-positive for the Qwen3.5 decay contract")

    with torch.no_grad():
        raw_query = query.detach()
        raw_key = key.detach()
        normalized_query = raw_query * torch.rsqrt(
            (raw_query * raw_query).sum(dim=-1, keepdim=True) + _QWEN35_L2NORM_EPS
        )
        normalized_key = raw_key * torch.rsqrt(
            (raw_key * raw_key).sum(dim=-1, keepdim=True) + _QWEN35_L2NORM_EPS
        )
        query32 = normalized_query.transpose(1, 2).contiguous().to(torch.float32)
        key32 = normalized_key.transpose(1, 2).contiguous().to(torch.float32)
        value32 = value.detach().transpose(1, 2).contiguous().to(torch.float32)
        beta32 = beta.detach().transpose(1, 2).contiguous().to(torch.float32)
        g32 = g.detach().transpose(1, 2).contiguous().to(torch.float32)

        prepared = Qwen35TritonDecodeInputs(
            query=(query32[:, :, 0] * (1.0 / math.sqrt(key_dim))).contiguous(),
            key=key32[:, :, 0].contiguous(),
            value=value32[:, :, 0].contiguous(),
            g=g32[:, :, 0].contiguous(),
            beta=beta32[:, :, 0].contiguous(),
            output_dtype=query.dtype,
            output_shape=(batch_size, 1, heads, value_dim),
        )
    _require_finite(
        (
            ("prepared query", prepared.query),
            ("prepared key", prepared.key),
            ("prepared value", prepared.value),
            ("prepared g", prepared.g),
            ("prepared beta", prepared.beta),
        )
    )
    return prepared


def validate_packed_state(state: TritonPackedState, *, check_values: bool = True) -> None:
    """Fail closed on layout, dtype, device, scale, and code-range drift."""

    if not isinstance(state, TritonPackedState):
        raise TypeError("state must be a TritonPackedState")
    _require_shape(state.shape)
    _require_quantization_contract(state.bits, state.group_size)
    device = _require_cuda_contiguous(
        state.payload,
        name="state.payload",
        dtypes=(torch.uint8,),
    )
    _require_cuda_contiguous(
        state.scales,
        name="state.scales",
        device=device,
        dtypes=(torch.float16,),
    )
    if state.payload.ndim != 1:
        raise ValueError("state.payload must be one-dimensional")
    _require_no_overlap(
        state.payload,
        left_name="state.payload",
        right=state.scales,
        right_name="state.scales",
    )
    if state.scales.shape != (state.rows, state.groups_per_row):
        raise ValueError(
            "state.scales shape must be "
            f"{(state.rows, state.groups_per_row)}, got {tuple(state.scales.shape)}"
        )
    expected_payload = _expected_payload_elements(state)
    if state.payload.numel() != expected_payload:
        raise ValueError(
            f"state.payload has {state.payload.numel()} bytes, expected {expected_payload}"
        )
    if not check_values:
        return
    if not bool(torch.isfinite(state.scales).all().item()):
        raise ValueError("state.scales must contain only finite values")
    if not bool((state.scales > 0).all().item()):
        raise ValueError("state.scales must be strictly positive")
    if state.bits == 8:
        if bool((state.payload == 0x80).any().item()):
            raise ValueError("INT8 payload contains the unsupported -128 code")
    else:
        low = torch.bitwise_and(state.payload, 0x0F)
        high = torch.bitwise_right_shift(state.payload, 4)
        if bool(((low == 0x08) | (high == 0x08)).any().item()):
            raise ValueError("INT4 payload contains the unsupported -8 code")


def pack_triton_state(
    state: torch.Tensor,
    *,
    bits: int,
    group_size: int = 128,
) -> TritonPackedState:
    """Pack one FP32 state using the canonical nearest/FP16-scale layout."""

    _require_shape(tuple(state.shape))
    _require_quantization_contract(bits, group_size)
    if state.dtype != torch.float32:
        raise TypeError("state must use torch.float32")
    if state.device.type != "cuda":
        raise ValueError("state must be on CUDA")
    if not state.is_contiguous():
        raise ValueError("state must be contiguous")
    _require_inference_only((("state", state),))
    packed = quantize_pack(
        state,
        QuantizationSpec(
            bits=bits,
            group_size=group_size,
            scale_bits=16,
            flatten_last_dims=2,
            rounding="nearest",
        ),
    )
    payload = packed.payload.view(torch.uint8)
    result = TritonPackedState(
        payload=payload.contiguous(),
        scales=packed.scales.contiguous(),
        shape=tuple(state.shape),
        bits=bits,
        group_size=group_size,
    )
    validate_packed_state(result)
    return result


def unpack_triton_state(state: TritonPackedState) -> torch.Tensor:
    """Materialize FP32 state for tests and offline diagnostics only."""

    validate_packed_state(state)
    spec = QuantizationSpec(
        bits=state.bits,
        group_size=state.group_size,
        scale_bits=16,
        flatten_last_dims=2,
        rounding="nearest",
    )
    payload = state.payload if state.bits == 4 else state.payload.view(torch.int8)
    canonical = PackedQuantizedTensor(
        payload=payload,
        scales=state.scales,
        spec=spec,
        original_shape=state.shape,
        original_dtype=torch.float32,
        flattened_size=state.flattened_size,
        padded_size=state.padded_size,
        rows=state.rows,
        groups_per_row=state.groups_per_row,
    )
    return canonical.dequantize()


def _canonical_step_shapes(
    state: TritonPackedState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    check_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_packed_state(state, check_values=check_values)
    batch_size, heads, key_dim, value_dim = state.shape
    device = state.payload.device
    expected_vectors = {
        "query": (query, (batch_size, heads, key_dim)),
        "key": (key, (batch_size, heads, key_dim)),
        "value": (value, (batch_size, heads, value_dim)),
    }
    for name, (tensor, expected_shape) in expected_vectors.items():
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=device,
            dtypes=_SUPPORTED_INPUT_DTYPES,
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}")

    def canonical_scalar(tensor: torch.Tensor, name: str) -> torch.Tensor:
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=device,
            dtypes=_SUPPORTED_INPUT_DTYPES,
        )
        if tuple(tensor.shape) == (batch_size, heads):
            return tensor
        if tuple(tensor.shape) == (batch_size, heads, 1):
            return tensor.reshape(batch_size, heads)
        raise ValueError(
            f"{name} shape must be {(batch_size, heads)} or "
            f"{(batch_size, heads, 1)}, got {tuple(tensor.shape)}"
        )

    canonical_g = canonical_scalar(g, "g")
    canonical_beta = canonical_scalar(beta, "beta")
    _require_inference_only(
        (
            ("query", query),
            ("key", key),
            ("value", value),
            ("g", canonical_g),
            ("beta", canonical_beta),
        )
    )
    if check_values:
        _require_finite(
            (
                ("query", query),
                ("key", key),
                ("value", value),
                ("g", canonical_g),
                ("beta", canonical_beta),
            )
        )
        if bool(((canonical_beta < 0) | (canonical_beta > 1)).any().item()):
            raise ValueError("beta must lie in [0, 1]")
        if bool((canonical_g > 0).any().item()):
            raise ValueError("g must be non-positive for the Qwen3.5 decay contract")
    return canonical_g, canonical_beta


def allocate_gated_delta_workspace(state: TritonPackedState) -> TritonGatedDeltaWorkspace:
    """Allocate fixed output buffers without materializing FP32 state."""

    validate_packed_state(state)
    vector_shape = (state.batch_size, state.heads, state.value_dim)
    return TritonGatedDeltaWorkspace(
        payload=torch.empty_like(state.payload),
        scales=torch.empty_like(state.scales),
        output=torch.empty(vector_shape, dtype=torch.float32, device=state.payload.device),
        update=torch.empty(vector_shape, dtype=torch.float32, device=state.payload.device),
    )


def _validate_workspace(
    state: TritonPackedState,
    workspace: TritonGatedDeltaWorkspace,
) -> None:
    if not isinstance(workspace, TritonGatedDeltaWorkspace):
        raise TypeError("workspace must be a TritonGatedDeltaWorkspace")
    expected = {
        "workspace.payload": (workspace.payload, state.payload.shape, torch.uint8),
        "workspace.scales": (workspace.scales, state.scales.shape, torch.float16),
        "workspace.output": (
            workspace.output,
            (state.batch_size, state.heads, state.value_dim),
            torch.float32,
        ),
        "workspace.update": (
            workspace.update,
            (state.batch_size, state.heads, state.value_dim),
            torch.float32,
        ),
    }
    for name, (tensor, shape, dtype) in expected.items():
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=state.payload.device,
            dtypes=(dtype,),
        )
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} shape must be {tuple(shape)}, got {tuple(tensor.shape)}")
    workspace_tensors = (
        ("workspace.payload", workspace.payload),
        ("workspace.scales", workspace.scales),
        ("workspace.output", workspace.output),
        ("workspace.update", workspace.update),
    )
    _require_pairwise_no_overlap(workspace_tensors)
    for workspace_name, workspace_tensor in workspace_tensors:
        for state_name, state_tensor in (
            ("state.payload", state.payload),
            ("state.scales", state.scales),
        ):
            _require_no_overlap(
                workspace_tensor,
                left_name=workspace_name,
                right=state_tensor,
                right_name=state_name,
            )


if triton is not None:

    @triton.jit
    def _load_signed_code(payload_ptr, code_offsets, mask, bits: tl.constexpr):
        if bits == 8:
            raw = tl.load(payload_ptr + code_offsets, mask=mask, other=0).to(tl.int32)
            return tl.where(raw >= 128, raw - 256, raw)
        raw = tl.load(payload_ptr + code_offsets // 2, mask=mask, other=0).to(tl.int32)
        shift = (code_offsets & 1) * 4
        nibble = (raw >> shift) & 15
        return tl.where(nibble >= 8, nibble - 16, nibble)

    @triton.jit
    def _packed_read_kernel(
        payload_ptr,
        scales_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        g_ptr,
        beta_ptr,
        update_ptr,
        output_ptr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        groups_per_row: tl.constexpr,
        padded_size: tl.constexpr,
        group_size: tl.constexpr,
        bits: tl.constexpr,
        block_key: tl.constexpr,
        block_value: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        key_offsets = tl.arange(0, block_key)
        value_offsets = value_block * block_value + tl.arange(0, block_value)
        key_mask = key_offsets < key_dim
        value_mask = value_offsets < value_dim

        flat_offsets = key_offsets[:, None] * value_dim + value_offsets[None, :]
        state_mask = key_mask[:, None] & value_mask[None, :]
        code_offsets = row * padded_size + flat_offsets
        codes = _load_signed_code(payload_ptr, code_offsets, state_mask, bits)
        scale_offsets = row * groups_per_row + flat_offsets // group_size
        scales = tl.load(scales_ptr + scale_offsets, mask=state_mask, other=0.0).to(tl.float32)
        decay = tl.exp(tl.load(g_ptr + row).to(tl.float32))
        decayed_state = codes.to(tl.float32) * scales * decay

        vector_base = row * key_dim
        key_vector = tl.load(key_ptr + vector_base + key_offsets, mask=key_mask, other=0.0).to(
            tl.float32
        )
        query_vector = tl.load(
            query_ptr + vector_base + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        remembered = tl.sum(decayed_state * key_vector[:, None], axis=0)
        query_read = tl.sum(decayed_state * query_vector[:, None], axis=0)
        key_query = tl.sum(key_vector * query_vector, axis=0)

        output_base = row * value_dim
        source_value = tl.load(
            value_ptr + output_base + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        beta_value = tl.load(beta_ptr + row).to(tl.float32)
        update = beta_value * (source_value - remembered)
        output = query_read + key_query * update
        tl.store(update_ptr + output_base + value_offsets, update, mask=value_mask)
        tl.store(output_ptr + output_base + value_offsets, output, mask=value_mask)

    @triton.jit
    def _packed_update_q8_kernel(
        payload_ptr,
        scales_ptr,
        key_ptr,
        g_ptr,
        update_ptr,
        output_payload_ptr,
        output_scales_ptr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        groups_per_row: tl.constexpr,
        flattened_size: tl.constexpr,
        padded_size: tl.constexpr,
        group_size: tl.constexpr,
        block_group: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        local_offsets = tl.arange(0, block_group)
        flat_offsets = group * group_size + local_offsets
        valid = (local_offsets < group_size) & (flat_offsets < flattened_size)
        code_offsets = row * padded_size + flat_offsets
        codes = _load_signed_code(payload_ptr, code_offsets, valid, 8)
        old_scale = tl.load(scales_ptr + row * groups_per_row + group).to(tl.float32)

        key_offsets = flat_offsets // value_dim
        value_offsets = flat_offsets - key_offsets * value_dim
        key_values = tl.load(
            key_ptr + row * key_dim + key_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        updates = tl.load(
            update_ptr + row * value_dim + value_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        decay = tl.exp(tl.load(g_ptr + row).to(tl.float32))
        updated = codes.to(tl.float32) * old_scale * decay + key_values * updates
        updated = tl.where(valid, updated, 0.0)

        absmax = tl.max(tl.abs(updated), axis=0)
        ideal_scale = tl.where(absmax > 1.0e-12, absmax / 127.0, 1.0)
        ideal_scale = tl.minimum(tl.maximum(ideal_scale, 2.0**-24), 65504.0)
        stored_scale = ideal_scale.to(tl.float16)
        working_scale = stored_scale.to(tl.float32)
        quantized = libdevice.rint(updated / working_scale)
        quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0).to(tl.int32)
        unsigned = tl.where(quantized < 0, quantized + 256, quantized).to(tl.uint8)

        tl.store(output_payload_ptr + code_offsets, unsigned, mask=local_offsets < group_size)
        tl.store(output_scales_ptr + row * groups_per_row + group, stored_scale)

    @triton.jit
    def _packed_update_q4_kernel(
        payload_ptr,
        scales_ptr,
        key_ptr,
        g_ptr,
        update_ptr,
        output_payload_ptr,
        output_scales_ptr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        groups_per_row: tl.constexpr,
        flattened_size: tl.constexpr,
        padded_size: tl.constexpr,
        group_size: tl.constexpr,
        block_bytes: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        byte_lanes = tl.arange(0, block_bytes)
        first_local = byte_lanes * 2
        second_local = first_local + 1
        group_start = group * group_size
        first_flat = group_start + first_local
        second_flat = group_start + second_local
        first_valid = (first_local < group_size) & (first_flat < flattened_size)
        second_valid = (second_local < group_size) & (second_flat < flattened_size)
        first_code_offset = row * padded_size + first_flat
        second_code_offset = row * padded_size + second_flat
        first_code = _load_signed_code(payload_ptr, first_code_offset, first_valid, 4)
        second_code = _load_signed_code(payload_ptr, second_code_offset, second_valid, 4)
        old_scale = tl.load(scales_ptr + row * groups_per_row + group).to(tl.float32)
        decay = tl.exp(tl.load(g_ptr + row).to(tl.float32))

        first_key_offset = first_flat // value_dim
        first_value_offset = first_flat - first_key_offset * value_dim
        second_key_offset = second_flat // value_dim
        second_value_offset = second_flat - second_key_offset * value_dim
        first_key = tl.load(
            key_ptr + row * key_dim + first_key_offset,
            mask=first_valid,
            other=0.0,
        ).to(tl.float32)
        second_key = tl.load(
            key_ptr + row * key_dim + second_key_offset,
            mask=second_valid,
            other=0.0,
        ).to(tl.float32)
        first_update = tl.load(
            update_ptr + row * value_dim + first_value_offset,
            mask=first_valid,
            other=0.0,
        ).to(tl.float32)
        second_update = tl.load(
            update_ptr + row * value_dim + second_value_offset,
            mask=second_valid,
            other=0.0,
        ).to(tl.float32)
        first_updated = first_code.to(tl.float32) * old_scale * decay + first_key * first_update
        second_updated = second_code.to(tl.float32) * old_scale * decay + second_key * second_update
        first_updated = tl.where(first_valid, first_updated, 0.0)
        second_updated = tl.where(second_valid, second_updated, 0.0)

        absmax = tl.maximum(
            tl.max(tl.abs(first_updated), axis=0),
            tl.max(tl.abs(second_updated), axis=0),
        )
        ideal_scale = tl.where(absmax > 1.0e-12, absmax / 7.0, 1.0)
        ideal_scale = tl.minimum(tl.maximum(ideal_scale, 2.0**-24), 65504.0)
        stored_scale = ideal_scale.to(tl.float16)
        working_scale = stored_scale.to(tl.float32)
        first_quantized = libdevice.rint(first_updated / working_scale)
        second_quantized = libdevice.rint(second_updated / working_scale)
        first_quantized = tl.minimum(tl.maximum(first_quantized, -7.0), 7.0).to(tl.int32)
        second_quantized = tl.minimum(tl.maximum(second_quantized, -7.0), 7.0).to(tl.int32)
        low = first_quantized & 15
        high = second_quantized & 15
        packed_byte = (low | (high << 4)).to(tl.uint8)

        output_byte_base = (row * padded_size + group_start) // 2
        tl.store(
            output_payload_ptr + output_byte_base + byte_lanes,
            packed_byte,
            mask=first_local < group_size,
        )
        tl.store(output_scales_ptr + row * groups_per_row + group, stored_scale)

    @triton.jit
    def _fp32_read_kernel(
        state_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        g_ptr,
        beta_ptr,
        update_ptr,
        output_ptr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        block_key: tl.constexpr,
        block_value: tl.constexpr,
    ):
        row = tl.program_id(0)
        value_block = tl.program_id(1)
        key_offsets = tl.arange(0, block_key)
        value_offsets = value_block * block_value + tl.arange(0, block_value)
        key_mask = key_offsets < key_dim
        value_mask = value_offsets < value_dim
        state_offsets = (
            row * key_dim * value_dim + key_offsets[:, None] * value_dim + value_offsets[None, :]
        )
        state_mask = key_mask[:, None] & value_mask[None, :]
        decay = tl.exp(tl.load(g_ptr + row).to(tl.float32))
        decayed_state = (
            tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(tl.float32) * decay
        )
        vector_base = row * key_dim
        key_vector = tl.load(key_ptr + vector_base + key_offsets, mask=key_mask, other=0.0).to(
            tl.float32
        )
        query_vector = tl.load(
            query_ptr + vector_base + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        remembered = tl.sum(decayed_state * key_vector[:, None], axis=0)
        query_read = tl.sum(decayed_state * query_vector[:, None], axis=0)
        key_query = tl.sum(key_vector * query_vector, axis=0)
        output_base = row * value_dim
        source_value = tl.load(
            value_ptr + output_base + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        beta_value = tl.load(beta_ptr + row).to(tl.float32)
        update = beta_value * (source_value - remembered)
        output = query_read + key_query * update
        tl.store(update_ptr + output_base + value_offsets, update, mask=value_mask)
        tl.store(output_ptr + output_base + value_offsets, output, mask=value_mask)

    @triton.jit
    def _fp32_update_kernel(
        state_ptr,
        key_ptr,
        g_ptr,
        update_ptr,
        output_state_ptr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        flattened_size: tl.constexpr,
        group_size: tl.constexpr,
        block_group: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        local_offsets = tl.arange(0, block_group)
        flat_offsets = group * group_size + local_offsets
        valid = (local_offsets < group_size) & (flat_offsets < flattened_size)
        key_offsets = flat_offsets // value_dim
        value_offsets = flat_offsets - key_offsets * value_dim
        state_offsets = row * flattened_size + flat_offsets
        decay = tl.exp(tl.load(g_ptr + row).to(tl.float32))
        old_state = tl.load(state_ptr + state_offsets, mask=valid, other=0.0).to(tl.float32)
        key_values = tl.load(
            key_ptr + row * key_dim + key_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        updates = tl.load(
            update_ptr + row * value_dim + value_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_state_ptr + state_offsets,
            old_state * decay + key_values * updates,
            mask=valid,
        )


def _launch_packed_step(
    state: TritonPackedState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    workspace: TritonGatedDeltaWorkspace,
) -> None:
    if triton is None:  # pragma: no cover - guarded by preparation
        raise RuntimeError("Triton is not installed")
    block_key = triton.next_power_of_2(state.key_dim)
    block_value = min(32, triton.next_power_of_2(state.value_dim))
    read_grid = (state.rows, triton.cdiv(state.value_dim, block_value))
    _packed_read_kernel[read_grid](
        state.payload,
        state.scales,
        query,
        key,
        value,
        g,
        beta,
        workspace.update,
        workspace.output,
        key_dim=state.key_dim,
        value_dim=state.value_dim,
        groups_per_row=state.groups_per_row,
        padded_size=state.padded_size,
        group_size=state.group_size,
        bits=state.bits,
        block_key=block_key,
        block_value=block_value,
        num_warps=4,
    )
    update_grid = (state.rows, state.groups_per_row)
    if state.bits == 8:
        _packed_update_q8_kernel[update_grid](
            state.payload,
            state.scales,
            key,
            g,
            workspace.update,
            workspace.payload,
            workspace.scales,
            key_dim=state.key_dim,
            value_dim=state.value_dim,
            groups_per_row=state.groups_per_row,
            flattened_size=state.flattened_size,
            padded_size=state.padded_size,
            group_size=state.group_size,
            block_group=triton.next_power_of_2(state.group_size),
            num_warps=4,
        )
    else:
        _packed_update_q4_kernel[update_grid](
            state.payload,
            state.scales,
            key,
            g,
            workspace.update,
            workspace.payload,
            workspace.scales,
            key_dim=state.key_dim,
            value_dim=state.value_dim,
            groups_per_row=state.groups_per_row,
            flattened_size=state.flattened_size,
            padded_size=state.padded_size,
            group_size=state.group_size,
            block_bytes=triton.next_power_of_2(state.group_size // 2),
            num_warps=4,
        )


@dataclass(slots=True)
class PreparedTritonGatedDeltaStep:
    """Validated, allocation-free launch object for repeated diagnostic timing.

    Tensor metadata and values are validated at construction.  Callers must not
    mutate the prepared input tensors between ``run`` calls. Results borrow the
    workspace and are overwritten by the next call. A prepared step and its
    workspace support one in-flight call on one CUDA stream; concurrent or
    cross-stream reuse is unsupported.
    """

    state: TritonPackedState
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    workspace: TritonGatedDeltaWorkspace

    def run(self) -> TritonGatedDeltaStepResult:
        _launch_packed_step(
            self.state,
            self.query,
            self.key,
            self.value,
            self.g,
            self.beta,
            self.workspace,
        )
        next_state = TritonPackedState(
            payload=self.workspace.payload,
            scales=self.workspace.scales,
            shape=self.state.shape,
            bits=self.state.bits,
            group_size=self.state.group_size,
        )
        return TritonGatedDeltaStepResult(
            state=next_state,
            output=self.workspace.output,
            update=self.workspace.update,
        )


def prepare_gated_delta_step(
    state: TritonPackedState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    workspace: TritonGatedDeltaWorkspace | None = None,
) -> PreparedTritonGatedDeltaStep:
    """Validate and prepare one packed, single-token decode step.

    ``query`` and ``key`` must already be recurrence-ready (including Qwen3.5
    L2 normalization and query scaling when those model conventions are wanted).
    """

    _require_triton_cuda()
    canonical_g, canonical_beta = _canonical_step_shapes(
        state,
        query,
        key,
        value,
        g,
        beta,
        check_values=True,
    )
    selected_workspace = workspace or allocate_gated_delta_workspace(state)
    _validate_workspace(state, selected_workspace)
    for workspace_name, workspace_tensor in (
        ("workspace.payload", selected_workspace.payload),
        ("workspace.scales", selected_workspace.scales),
        ("workspace.output", selected_workspace.output),
        ("workspace.update", selected_workspace.update),
    ):
        for input_name, input_tensor in (
            ("query", query),
            ("key", key),
            ("value", value),
            ("g", canonical_g),
            ("beta", canonical_beta),
        ):
            _require_no_overlap(
                workspace_tensor,
                left_name=workspace_name,
                right=input_tensor,
                right_name=input_name,
            )
    return PreparedTritonGatedDeltaStep(
        state=state,
        query=query,
        key=key,
        value=value,
        g=canonical_g,
        beta=canonical_beta,
        workspace=selected_workspace,
    )


def gated_delta_step(
    state: TritonPackedState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    workspace: TritonGatedDeltaWorkspace | None = None,
) -> TritonGatedDeltaStepResult:
    """Run one validated packed-state decode step."""

    return prepare_gated_delta_step(
        state,
        query,
        key,
        value,
        g,
        beta,
        workspace=workspace,
    ).run()


def _validate_fp32_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda()
    if state.dtype != torch.float32:
        raise TypeError("FP32 reference state must use torch.float32")
    if state.ndim != 4:
        raise ValueError("FP32 reference state must have shape [batch, heads, key, value]")
    batch_size, heads, key_dim, value_dim = _require_shape(tuple(state.shape))
    _require_cuda_contiguous(
        state,
        name="state",
        dtypes=(torch.float32,),
    )
    expected = {
        "query": (query, (batch_size, heads, key_dim)),
        "key": (key, (batch_size, heads, key_dim)),
        "value": (value, (batch_size, heads, value_dim)),
    }
    for name, (tensor, shape) in expected.items():
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=state.device,
            dtypes=_SUPPORTED_INPUT_DTYPES,
        )
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")

    def scalar(tensor: torch.Tensor, name: str) -> torch.Tensor:
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=state.device,
            dtypes=_SUPPORTED_INPUT_DTYPES,
        )
        if tuple(tensor.shape) == (batch_size, heads):
            return tensor
        if tuple(tensor.shape) == (batch_size, heads, 1):
            return tensor.reshape(batch_size, heads)
        raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")

    canonical_g = scalar(g, "g")
    canonical_beta = scalar(beta, "beta")
    _require_inference_only(
        (
            ("state", state),
            ("query", query),
            ("key", key),
            ("value", value),
            ("g", canonical_g),
            ("beta", canonical_beta),
        )
    )
    _require_finite(
        (
            ("state", state),
            ("query", query),
            ("key", key),
            ("value", value),
            ("g", canonical_g),
            ("beta", canonical_beta),
        )
    )
    if bool(((canonical_beta < 0) | (canonical_beta > 1)).any().item()):
        raise ValueError("beta must lie in [0, 1]")
    if bool((canonical_g > 0).any().item()):
        raise ValueError("g must be non-positive for the Qwen3.5 decay contract")
    return canonical_g, canonical_beta


def allocate_fp32_workspace(state: torch.Tensor) -> TritonFp32Workspace:
    """Allocate the same-schedule FP32 baseline outputs."""

    if state.dtype != torch.float32 or state.ndim != 4 or state.device.type != "cuda":
        raise ValueError("state must be a CUDA FP32 tensor with four dimensions")
    vector_shape = (state.shape[0], state.shape[1], state.shape[3])
    return TritonFp32Workspace(
        state=torch.empty_like(state),
        output=torch.empty(vector_shape, dtype=torch.float32, device=state.device),
        update=torch.empty(vector_shape, dtype=torch.float32, device=state.device),
    )


def _launch_fp32_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    workspace: TritonFp32Workspace,
    *,
    group_size: int,
) -> None:
    if triton is None:  # pragma: no cover - guarded by preparation
        raise RuntimeError("Triton is not installed")
    batch_size, heads, key_dim, value_dim = state.shape
    rows = batch_size * heads
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    _fp32_read_kernel[(rows, triton.cdiv(value_dim, block_value))](
        state,
        query,
        key,
        value,
        g,
        beta,
        workspace.update,
        workspace.output,
        key_dim=key_dim,
        value_dim=value_dim,
        block_key=block_key,
        block_value=block_value,
        num_warps=4,
    )
    flattened_size = key_dim * value_dim
    groups_per_row = math.ceil(flattened_size / group_size)
    _fp32_update_kernel[(rows, groups_per_row)](
        state,
        key,
        g,
        workspace.update,
        workspace.state,
        key_dim=key_dim,
        value_dim=value_dim,
        flattened_size=flattened_size,
        group_size=group_size,
        block_group=triton.next_power_of_2(group_size),
        num_warps=4,
    )


@dataclass(slots=True)
class PreparedTritonFp32Step:
    """Validated same-schedule FP32 reference launch.

    Results borrow the workspace and are overwritten by the next call. One
    prepared step/workspace may have only one in-flight launch on one stream.
    """

    state: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    workspace: TritonFp32Workspace
    group_size: int

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _launch_fp32_step(
            self.state,
            self.query,
            self.key,
            self.value,
            self.g,
            self.beta,
            self.workspace,
            group_size=self.group_size,
        )
        return self.workspace.state, self.workspace.output, self.workspace.update


def prepare_fp32_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    group_size: int = 128,
    workspace: TritonFp32Workspace | None = None,
) -> PreparedTritonFp32Step:
    """Prepare a same-schedule FP32 reference for local kernel comparisons."""

    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if not 1 <= group_size <= _MAX_GROUP_SIZE:
        raise ValueError(f"group_size must be between 1 and {_MAX_GROUP_SIZE}")
    canonical_g, canonical_beta = _validate_fp32_step(
        state,
        query,
        key,
        value,
        g,
        beta,
    )
    selected = workspace or allocate_fp32_workspace(state)
    if not isinstance(selected, TritonFp32Workspace):
        raise TypeError("workspace must be a TritonFp32Workspace")
    expected = {
        "workspace.state": (selected.state, state.shape),
        "workspace.output": (
            selected.output,
            (state.shape[0], state.shape[1], state.shape[3]),
        ),
        "workspace.update": (
            selected.update,
            (state.shape[0], state.shape[1], state.shape[3]),
        ),
    }
    for name, (tensor, shape) in expected.items():
        _require_cuda_contiguous(
            tensor,
            name=name,
            device=state.device,
            dtypes=(torch.float32,),
        )
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} shape must be {tuple(shape)}, got {tuple(tensor.shape)}")
    workspace_tensors = (
        ("workspace.state", selected.state),
        ("workspace.output", selected.output),
        ("workspace.update", selected.update),
    )
    _require_pairwise_no_overlap(workspace_tensors)
    for workspace_name, workspace_tensor in workspace_tensors:
        _require_no_overlap(
            workspace_tensor,
            left_name=workspace_name,
            right=state,
            right_name="state",
        )
        for input_name, input_tensor in (
            ("query", query),
            ("key", key),
            ("value", value),
            ("g", canonical_g),
            ("beta", canonical_beta),
        ):
            _require_no_overlap(
                workspace_tensor,
                left_name=workspace_name,
                right=input_tensor,
                right_name=input_name,
            )
    return PreparedTritonFp32Step(
        state=state,
        query=query,
        key=key,
        value=value,
        g=canonical_g,
        beta=canonical_beta,
        workspace=selected,
        group_size=group_size,
    )
