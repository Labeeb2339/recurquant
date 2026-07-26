"""Correctness-first exact-budget policy for row-wise INT4/INT6/INT8.

The allocator in this module is deliberately a CPU reference implementation.
For ``N`` rows and an exact marginal-step budget ``K``, it uses ``O(NK)`` time,
``O(NK)`` bytes for backtracking choices, and ``O(K)`` floating-point working
memory. It makes no runtime or deployment claim.

Each output code consumes zero, one, or two 2-bit marginal steps:

``0 -> INT4``, ``1 -> INT6``, and ``2 -> INT8``.

The dynamic program chooses among all three complete row states. It therefore
remains globally optimal when a row's INT6-to-INT8 benefit exceeds its
INT4-to-INT6 benefit, where sorting individual marginal gains would violate the
required precedence.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import torch

FROZEN_QWEN35_INT8_ROW_QUOTAS = MappingProxyType(
    {
        0: 355,
        1: 380,
        2: 269,
        4: 179,
        5: 185,
        6: 105,
        8: 80,
        9: 43,
        10: 84,
        12: 30,
        13: 62,
        14: 54,
        16: 45,
        17: 27,
        18: 7,
        20: 9,
        21: 7,
        22: 55,
    }
)

QWEN35_ROWS_PER_LINEAR_LAYER = 16 * 128
QWEN35_VALUE_DIM = 128
QWEN35_SCALE_BYTES_PER_ROW = 2
QWEN35_OLD_PRECISION_BITS_PER_ROW = 1
QWEN35_MULTIBIT_PRECISION_BITS_PER_ROW = 2
QWEN35_MARGINAL_STEP_BYTES = 32
QWEN35_FROZEN_TOTAL_MARGINAL_STEPS = 3_808
QWEN35_FROZEN_TOTAL_STATE_BYTES = 2_564_096


def _validate_distortion_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.numel() == 0:
        raise ValueError(f"{name} must have non-empty shape [heads, rows]")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    shape = (tensor.shape[0], tensor.shape[1])
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"D4, D6, and D8 must have identical shapes; {name} has {shape}, "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    if (tensor < 0).any().item():
        raise ValueError(f"{name} must contain only nonnegative values")
    return shape


def allocate_exact_multibit_codes(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Minimize exact total distortion under a 0/1/2 step budget.

    ``d4``, ``d6``, and ``d8`` contain the complete weighted distortion for
    representing each row at INT4, INT6, and INT8. All tensors must have the
    same non-empty ``[heads, rows]`` shape and contain finite, nonnegative
    floating-point values.

    The returned CPU ``torch.uint8`` tensor has the input shape, uses codes
    ``0``/``1``/``2`` for INT4/INT6/INT8, and has an exact code sum equal to
    ``marginal_steps``. Among solutions with exactly equal CPU-FP64 objective,
    the lexicographically greatest flattened code vector is chosen: lower
    flattened row indices receive higher precision first.

    This is a correctness/reference path. Its complexity is ``O(NK)`` time,
    ``O(NK)`` choice bytes, and ``O(K)`` FP64 workspace for ``N`` rows and
    budget ``K``.
    """

    shape = _validate_distortion_tensor(d4, name="D4", expected_shape=None)
    _validate_distortion_tensor(d6, name="D6", expected_shape=shape)
    _validate_distortion_tensor(d8, name="D8", expected_shape=shape)
    if isinstance(marginal_steps, bool) or not isinstance(marginal_steps, int):
        raise TypeError("marginal_steps must be an integer")

    total_rows = math.prod(shape)
    if not 0 <= marginal_steps <= 2 * total_rows:
        raise ValueError(
            "marginal_steps must be in the exact representable range "
            f"[0, {2 * total_rows}]"
        )

    if marginal_steps == 0:
        return torch.zeros(shape, dtype=torch.uint8)
    if marginal_steps == 2 * total_rows:
        return torch.full(shape, 2, dtype=torch.uint8)

    flat_distortions = np.stack(
        [
            tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
            for tensor in (d4, d6, d8)
        ],
        axis=1,
    )

    # Removing a row-wise constant leaves every allocation objective difference
    # unchanged. Scaling the remaining nonnegative costs prevents overflow while
    # retaining the exact discrete optimization problem in CPU FP64.
    flat_distortions -= flat_distortions.min(axis=1, keepdims=True)
    maximum = float(flat_distortions.max())
    if maximum > 0.0:
        flat_distortions /= maximum

    unreachable = np.inf
    next_cost = np.full(marginal_steps + 1, unreachable, dtype=np.float64)
    next_cost[0] = 0.0
    choices = np.full(
        (total_rows, marginal_steps + 1),
        np.iinfo(np.uint8).max,
        dtype=np.uint8,
    )

    # Build suffix optima. Iterating rows backwards makes an equal-cost choice
    # of the larger current code exactly equivalent to lexicographic priority
    # for lower flattened row indices.
    for row_index in range(total_rows - 1, -1, -1):
        suffix_rows = total_rows - row_index - 1
        next_maximum_steps = min(marginal_steps, 2 * suffix_rows)
        current_cost = np.full(marginal_steps + 1, unreachable, dtype=np.float64)
        current_choice = choices[row_index]

        for code in (0, 1, 2):
            available = min(next_maximum_steps, marginal_steps - code)
            if available < 0:
                continue
            destination = slice(code, code + available + 1)
            candidate = (
                next_cost[: available + 1] + flat_distortions[row_index, code]
            )
            incumbent = current_cost[destination]
            incumbent_code = current_choice[destination]
            update = np.isfinite(candidate) & (
                (candidate < incumbent)
                | ((candidate == incumbent) & (code > incumbent_code))
            )
            incumbent[update] = candidate[update]
            incumbent_code[update] = code

        next_cost = current_cost

    if not np.isfinite(next_cost[marginal_steps]):
        raise RuntimeError("exact multibit dynamic program did not reach the requested budget")

    flat_codes = np.empty(total_rows, dtype=np.uint8)
    remaining = marginal_steps
    invalid_choice = np.iinfo(np.uint8).max
    for row_index in range(total_rows):
        code = int(choices[row_index, remaining])
        if code == invalid_choice:
            raise RuntimeError("exact multibit dynamic-program backtracking is incomplete")
        flat_codes[row_index] = code
        remaining -= code
    if remaining != 0:
        raise RuntimeError("exact multibit dynamic-program backtracking changed the budget")

    return torch.from_numpy(flat_codes.reshape(shape).copy())


def frozen_qwen35_multibit_step_budgets() -> dict[int, int]:
    """Return the exact per-layer 2-bit-step budgets matching frozen Q4/Q8 bytes.

    A Qwen3.5-0.8B Gated DeltaNet layer contains 2,048 row groups. Replacing its
    one-bit Q4/Q8 mask with a two-bit Q4/Q6/Q8 code costs 256 additional bytes,
    equal to eight 32-byte marginal steps. Therefore a layer with frozen INT8
    row quota ``q`` receives ``K = 2*q - 8`` steps.

    The helper validates every old/new per-layer byte count, the 3,808-step
    global sum, and the 2,564,096-byte global state total before returning.
    """

    budgets = {
        layer_index: 2 * quota - 8
        for layer_index, quota in FROZEN_QWEN35_INT8_ROW_QUOTAS.items()
    }
    if any(not 0 <= steps <= 2 * QWEN35_ROWS_PER_LINEAR_LAYER for steps in budgets.values()):
        raise RuntimeError("frozen Qwen3.5 multibit step budget is outside [0, 2N]")
    if sum(budgets.values()) != QWEN35_FROZEN_TOTAL_MARGINAL_STEPS:
        raise RuntimeError("frozen Qwen3.5 multibit step budgets do not sum to 3,808")

    q4_payload_bytes = math.ceil(4 * QWEN35_VALUE_DIM / 8)
    q8_payload_bytes = math.ceil(8 * QWEN35_VALUE_DIM / 8)
    old_mask_bytes = math.ceil(
        QWEN35_ROWS_PER_LINEAR_LAYER * QWEN35_OLD_PRECISION_BITS_PER_ROW / 8
    )
    multibit_mask_bytes = math.ceil(
        QWEN35_ROWS_PER_LINEAR_LAYER * QWEN35_MULTIBIT_PRECISION_BITS_PER_ROW / 8
    )
    base_payload_and_scales = QWEN35_ROWS_PER_LINEAR_LAYER * (
        q4_payload_bytes + QWEN35_SCALE_BYTES_PER_ROW
    )

    old_total = 0
    multibit_total = 0
    for layer_index, quota in FROZEN_QWEN35_INT8_ROW_QUOTAS.items():
        old_bytes = (
            base_payload_and_scales
            + old_mask_bytes
            + quota * (q8_payload_bytes - q4_payload_bytes)
        )
        multibit_bytes = (
            base_payload_and_scales
            + multibit_mask_bytes
            + budgets[layer_index] * QWEN35_MARGINAL_STEP_BYTES
        )
        if old_bytes != multibit_bytes:
            raise RuntimeError(
                f"frozen Qwen3.5 layer {layer_index} byte parity failed: "
                f"{old_bytes} != {multibit_bytes}"
            )
        old_total += old_bytes
        multibit_total += multibit_bytes

    if old_total != QWEN35_FROZEN_TOTAL_STATE_BYTES:
        raise RuntimeError("frozen Qwen3.5 Q4/Q8 state bytes do not equal 2,564,096")
    if multibit_total != QWEN35_FROZEN_TOTAL_STATE_BYTES:
        raise RuntimeError("frozen Qwen3.5 multibit state bytes do not equal 2,564,096")
    return budgets
