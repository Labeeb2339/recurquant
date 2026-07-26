"""Deterministic right-side randomized Hadamard transforms for recurrent states.

The codec is stateless.  Its sign vectors are derived from the frozen seed,
model-layer index, head index, and value width, so packed objects do not need
to retain a sign tensor or RNG state.  All arithmetic uses an FP32 workspace.
"""

from __future__ import annotations

import hashlib
import math

import torch

RHT_SEED = 2339
_SIGN_DOMAIN = b"recurquant.right-rht.signs.v1\0"
_SUPPORTED_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))


def _validate_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive_integer(value: object, *, name: str) -> int:
    validated = _validate_nonnegative_integer(value, name=name)
    if validated == 0:
        raise ValueError(f"{name} must be positive")
    return validated


def _validate_output_dtype(dtype: torch.dtype) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        rendered = ", ".join(str(value) for value in sorted(_SUPPORTED_DTYPES, key=str))
        raise TypeError(f"output_dtype must be one of {rendered}")


def _validate_floating_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use float16, bfloat16, or float32; got {tensor.dtype}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_power_of_two_width(width: int) -> None:
    if width <= 0 or width & (width - 1):
        raise ValueError(f"last dimension must be a positive power of two; got {width}")


def _validate_state(
    state: torch.Tensor,
    *,
    expected_heads: object,
    name: str,
) -> tuple[int, int]:
    _validate_floating_tensor(state, name=name)
    heads = _validate_positive_integer(expected_heads, name="expected_heads")
    if state.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, heads, rows, value]; got rank {state.ndim}"
        )
    if any(dimension <= 0 for dimension in state.shape):
        raise ValueError(f"{name} dimensions must all be positive")
    if state.shape[1] != heads:
        raise ValueError(
            f"{name} head dimension does not match expected_heads: {state.shape[1]} != {heads}"
        )
    width = state.shape[-1]
    _validate_power_of_two_width(width)
    return heads, width


def _fwht_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Return an unnormalized last-axis FWHT from an already validated tensor."""

    width = tensor.shape[-1]
    result = tensor.to(torch.float32).reshape(-1, width)
    stride = 1
    while stride < width:
        blocks = result.reshape(-1, width // (2 * stride), 2 * stride)
        left = blocks[..., :stride]
        right = blocks[..., stride:]
        result = torch.cat((left + right, left - right), dim=-1).reshape(-1, width)
        stride *= 2
    return result.reshape(tensor.shape)


def fwht_unnormalized(
    tensor: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply an unnormalized Walsh-Hadamard transform over the last dimension.

    The last dimension must be a positive power of two.  Computation is always
    performed in FP32.  By default the result is converted back to the input
    dtype; callers may request another supported output dtype.
    """

    _validate_floating_tensor(tensor, name="tensor")
    if tensor.ndim == 0:
        raise ValueError("tensor must have a last dimension")
    _validate_power_of_two_width(tensor.shape[-1])
    selected_dtype = tensor.dtype if output_dtype is None else output_dtype
    _validate_output_dtype(selected_dtype)
    return _fwht_fp32(tensor).to(selected_dtype)


def _portable_sign_values(*, layer_index: int, head_index: int, width: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < width:
        message = b"".join(
            (
                _SIGN_DOMAIN,
                RHT_SEED.to_bytes(8, "little", signed=False),
                layer_index.to_bytes(8, "little", signed=False),
                head_index.to_bytes(8, "little", signed=False),
                width.to_bytes(8, "little", signed=False),
                counter.to_bytes(8, "little", signed=False),
            )
        )
        digest = hashlib.sha256(message).digest()
        for byte in digest:
            for bit_index in range(8):
                values.append(1.0 if byte & (1 << bit_index) else -1.0)
                if len(values) == width:
                    return values
        counter += 1
    return values


def right_rht_signs(
    *,
    layer_index: int,
    expected_heads: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize the frozen transient sign schedule as ``[1, heads, 1, width]``."""

    layer = _validate_nonnegative_integer(layer_index, name="layer_index")
    heads = _validate_positive_integer(expected_heads, name="expected_heads")
    validated_width = _validate_positive_integer(width, name="width")
    _validate_power_of_two_width(validated_width)
    values = [
        _portable_sign_values(
            layer_index=layer,
            head_index=head_index,
            width=validated_width,
        )
        for head_index in range(heads)
    ]
    return torch.tensor(
        values,
        dtype=torch.float32,
        device=device,
    ).reshape(1, heads, 1, validated_width)


def right_rht_encode(
    state: torch.Tensor,
    *,
    layer_index: int,
    expected_heads: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode ``[batch, heads, rows, value]`` state rows with normalized RHT.

    For an unnormalized Hadamard matrix ``H`` and deterministic sign diagonal
    ``D``, this computes ``state @ D @ H / sqrt(value)`` independently for each
    layer and head.
    """

    layer = _validate_nonnegative_integer(layer_index, name="layer_index")
    heads, width = _validate_state(
        state,
        expected_heads=expected_heads,
        name="state",
    )
    _validate_output_dtype(output_dtype)
    signs = right_rht_signs(
        layer_index=layer,
        expected_heads=heads,
        width=width,
        device=state.device,
    )
    encoded = _fwht_fp32(state.to(torch.float32) * signs) / math.sqrt(width)
    return encoded.to(output_dtype)


def right_rht_decode(
    encoded: torch.Tensor,
    *,
    layer_index: int,
    expected_heads: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode a normalized right-RHT tensor into the original value basis."""

    layer = _validate_nonnegative_integer(layer_index, name="layer_index")
    heads, width = _validate_state(
        encoded,
        expected_heads=expected_heads,
        name="encoded",
    )
    _validate_output_dtype(output_dtype)
    signs = right_rht_signs(
        layer_index=layer,
        expected_heads=heads,
        width=width,
        device=encoded.device,
    )
    decoded = (_fwht_fp32(encoded) / math.sqrt(width)) * signs
    return decoded.to(output_dtype)
