"""Exact-budget policies for row-wise INT4/INT6/INT8.

``allocate_exact_multibit_codes`` is the deliberately slow dynamic-programming
oracle.  ``allocate_exact_multibit_codes_fast`` solves the same complete-state
problem in ``O(N log N)`` time and ``O(N)`` memory while preserving the
oracle's exact lexicographic tie rule.

Each output code consumes zero, one, or two 2-bit marginal steps:

``0 -> INT4``, ``1 -> INT6``, and ``2 -> INT8``.

Both allocators choose among all three complete row states.  They therefore
remain globally optimal when a row's INT6-to-INT8 benefit exceeds its
INT4-to-INT6 benefit, where naively sorting individual marginal gains would
violate the required precedence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def _exact_flat_distortions(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    """Return exact integer units for the CPU-FP64 distortion values.

    Every finite binary64 value is an integer times a power of two.  Converting
    all values to one shared binary quantum makes objective addition and
    equality independent of row order, gain subtraction, or prefix-sum order.
    A row-wise constant and a final shared power-of-two factor are removed to
    keep the integers compact; neither operation changes an allocation.
    """

    shape = _validate_distortion_tensor(d4, name="D4", expected_shape=None)
    _validate_distortion_tensor(d6, name="D6", expected_shape=shape)
    _validate_distortion_tensor(d8, name="D8", expected_shape=shape)
    flat_values = np.stack(
        [
            tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
            for tensor in (d4, d6, d8)
        ],
        axis=1,
    )
    maximum_denominator_exponent = 0
    for value in flat_values.reshape(-1):
        _, denominator = float(value).as_integer_ratio()
        maximum_denominator_exponent = max(
            maximum_denominator_exponent,
            denominator.bit_length() - 1,
        )

    exact_rows: list[tuple[int, int, int]] = []
    common_trailing_zeros: int | None = None
    for row in flat_values:
        scaled: list[int] = []
        for value in row:
            numerator, denominator = float(value).as_integer_ratio()
            denominator_exponent = denominator.bit_length() - 1
            scaled.append(numerator << (maximum_denominator_exponent - denominator_exponent))
        row_minimum = min(scaled)
        reduced = tuple(value - row_minimum for value in scaled)
        exact_rows.append(reduced)
        for value in reduced:
            if value:
                trailing_zeros = (value & -value).bit_length() - 1
                common_trailing_zeros = (
                    trailing_zeros
                    if common_trailing_zeros is None
                    else min(common_trailing_zeros, trailing_zeros)
                )

    if common_trailing_zeros:
        exact_rows = [tuple(value >> common_trailing_zeros for value in row) for row in exact_rows]
    return shape, exact_rows


def _validate_marginal_steps(marginal_steps: object, *, total_rows: int) -> int:
    if isinstance(marginal_steps, bool) or not isinstance(marginal_steps, int):
        raise TypeError("marginal_steps must be an integer")
    if not 0 <= marginal_steps <= 2 * total_rows:
        raise ValueError(
            f"marginal_steps must be in the exact representable range [0, {2 * total_rows}]"
        )
    return marginal_steps


class _RangeMinimum:
    """Small iterative RMQ used only for exact lexicographic comparisons."""

    def __init__(self, values: np.ndarray, *, sentinel: int) -> None:
        self.length = int(values.size)
        size = 1
        while size < self.length:
            size *= 2
        self.size = size
        self.sentinel = sentinel
        tree = np.full(2 * size, sentinel, dtype=np.int64)
        if self.length:
            tree[size : size + self.length] = values.astype(np.int64, copy=False)
        for index in range(size - 1, 0, -1):
            tree[index] = min(tree[2 * index], tree[2 * index + 1])
        self.tree = tree

    def query(self, start: int, stop: int) -> int:
        if not 0 <= start <= stop <= self.length:
            raise RuntimeError("internal range-minimum query is outside its domain")
        if start == stop:
            return self.sentinel
        left = start + self.size
        right = stop + self.size
        result = self.sentinel
        while left < right:
            if left & 1:
                result = min(result, int(self.tree[left]))
                left += 1
            if right & 1:
                right -= 1
                result = min(result, int(self.tree[right]))
            left //= 2
            right //= 2
        return result


@dataclass(frozen=True, slots=True)
class _FastAllocationCandidate:
    convex_steps: int
    bundle_count: int
    singleton_rank: int | None
    gain: int


@dataclass(frozen=True, slots=True)
class _FastAllocationStructure:
    total_rows: int
    convex_mask: np.ndarray
    convex_first_rank: np.ndarray
    convex_second_rank: np.ndarray
    nonconvex_rank: np.ndarray
    bundle_rows: np.ndarray
    convex_row_rmq: _RangeMinimum
    bundle_row_rmq: _RangeMinimum


def _candidate_bundle_limit(candidate: _FastAllocationCandidate) -> int:
    singleton = candidate.singleton_rank
    return candidate.bundle_count + int(
        singleton is not None and singleton < candidate.bundle_count
    )


def _candidate_nonconvex_code(
    candidate: _FastAllocationCandidate,
    rank: int,
) -> int:
    singleton = candidate.singleton_rank
    if rank < _candidate_bundle_limit(candidate) and rank != singleton:
        return 2
    if rank == singleton:
        return 1
    return 0


def _candidate_code_at_row(
    candidate: _FastAllocationCandidate,
    row: int,
    structure: _FastAllocationStructure,
) -> int:
    if structure.convex_mask[row]:
        return int(structure.convex_first_rank[row] < candidate.convex_steps) + int(
            structure.convex_second_rank[row] < candidate.convex_steps
        )
    return _candidate_nonconvex_code(candidate, int(structure.nonconvex_rank[row]))


def _candidate_lexicographically_greater(
    left: _FastAllocationCandidate,
    right: _FastAllocationCandidate,
    structure: _FastAllocationStructure,
) -> bool:
    """Compare two implicit complete code vectors without materializing either."""

    sentinel = structure.total_rows
    earliest = sentinel
    if left.convex_steps != right.convex_steps:
        earliest = min(
            earliest,
            structure.convex_row_rmq.query(
                min(left.convex_steps, right.convex_steps),
                max(left.convex_steps, right.convex_steps),
            ),
        )

    nonconvex_rows = structure.bundle_rows.size
    if nonconvex_rows:
        boundaries = {
            0,
            nonconvex_rows,
            _candidate_bundle_limit(left),
            _candidate_bundle_limit(right),
        }
        for singleton in (left.singleton_rank, right.singleton_rank):
            if singleton is not None:
                boundaries.add(singleton)
                boundaries.add(singleton + 1)
        ordered = sorted(boundary for boundary in boundaries if 0 <= boundary <= nonconvex_rows)
        for start, stop in zip(ordered, ordered[1:], strict=False):
            if start == stop:
                continue
            if _candidate_nonconvex_code(
                left,
                start,
            ) != _candidate_nonconvex_code(right, start):
                earliest = min(
                    earliest,
                    structure.bundle_row_rmq.query(start, stop),
                )

    if earliest == sentinel:
        return False
    left_code = _candidate_code_at_row(left, earliest, structure)
    right_code = _candidate_code_at_row(right, earliest, structure)
    if left_code == right_code:
        raise RuntimeError("implicit lexicographic comparison failed to find a differing row")
    return left_code > right_code


def _better_fast_candidate(
    candidate: _FastAllocationCandidate,
    incumbent: _FastAllocationCandidate | None,
    structure: _FastAllocationStructure,
) -> _FastAllocationCandidate:
    if incumbent is None or candidate.gain > incumbent.gain:
        return candidate
    if candidate.gain == incumbent.gain and _candidate_lexicographically_greater(
        candidate,
        incumbent,
        structure,
    ):
        return candidate
    return incumbent


def allocate_exact_multibit_codes_fast(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    """Return the exact complete-state optimum in ``O(N log N)`` time.

    Let ``a = D4 - D6`` and ``b = D6 - D8`` be the two precision-step gains.
    A row with ``a >= b`` is convex: its two gains can be sorted with the
    first step ordered before the second on an exact tie.  A row with ``a < b``
    is nonconvex and its code-2 choice is treated as one two-step bundle.

    The structural lemma behind the scan is strict.  If an allocation had two
    nonconvex rows ``i`` and ``j`` at code 1, replacing their two singletons by
    code pair ``(2, 0)`` changes gain by ``b_i - a_j`` and replacing them by
    ``(0, 2)`` changes gain by ``b_j - a_i``.  If neither change were positive,
    then

    ``a_i < b_i <= a_j < b_j <= a_i``,

    an impossibility.  Therefore an optimum cannot contain two nonconvex
    singletons.  The algorithm scans the number of complete nonconvex bundles,
    handles zero or one singleton with exact prefix/suffix maxima, and fills
    the remaining budget from sorted convex increments.

    Equal objective values preserve the oracle's lexicographically greatest
    flattened code vector.  Stable gain ordering handles ties within each
    component, while an RMQ-backed comparator resolves ties between different
    bundle counts without constructing ``O(N)`` candidate vectors.
    """

    shape, flat_distortions = _exact_flat_distortions(d4, d6, d8)
    total_rows = math.prod(shape)
    steps = _validate_marginal_steps(marginal_steps, total_rows=total_rows)
    if steps == 0:
        return torch.zeros(shape, dtype=torch.uint8)
    if steps == 2 * total_rows:
        return torch.full(shape, 2, dtype=torch.uint8)

    first_gain = [row[0] - row[1] for row in flat_distortions]
    second_gain = [row[1] - row[2] for row in flat_distortions]
    convex_mask = np.fromiter(
        (first_gain[row] >= second_gain[row] for row in range(total_rows)),
        dtype=np.bool_,
        count=total_rows,
    )
    convex_rows = np.flatnonzero(convex_mask).astype(np.int64, copy=False)
    nonconvex_rows = np.flatnonzero(~convex_mask).astype(np.int64, copy=False)

    # Convex rows contribute two precedence-safe unit increments.  Equal gains
    # prefer the earlier original row, then its first step before its second.
    convex_count = convex_rows.size
    increments = [(first_gain[int(row)], int(row), 0) for row in convex_rows]
    increments.extend((second_gain[int(row)], int(row), 1) for row in convex_rows)
    increments.sort(key=lambda item: (-item[0], item[1], item[2]))
    ordered_increment_rows = np.fromiter(
        (row for _, row, _ in increments),
        dtype=np.int64,
        count=len(increments),
    )
    convex_prefix = [0]
    for gain, _, _ in increments:
        convex_prefix.append(convex_prefix[-1] + gain)
    convex_first_rank = np.full(total_rows, -1, dtype=np.int64)
    convex_second_rank = np.full(total_rows, -1, dtype=np.int64)
    for rank, (_, row, precision_step) in enumerate(increments):
        if precision_step == 0:
            convex_first_rank[row] = rank
        else:
            convex_second_rank[row] = rank

    # Nonconvex rows contribute code-2 bundles.  Equal bundle gains prefer the
    # earlier original row because its code 2 is lexicographically greater.
    bundles = sorted(
        (
            (
                first_gain[int(row)] + second_gain[int(row)],
                int(row),
                first_gain[int(row)],
            )
            for row in nonconvex_rows
        ),
        key=lambda item: (-item[0], item[1]),
    )
    bundle_rows = np.fromiter(
        (row for _, row, _ in bundles),
        dtype=np.int64,
        count=len(bundles),
    )
    bundle_gains = [gain for gain, _, _ in bundles]
    bundle_first_gains = [gain for _, _, gain in bundles]
    bundle_prefix = [0]
    for gain in bundle_gains:
        bundle_prefix.append(bundle_prefix[-1] + gain)
    nonconvex_rank = np.full(total_rows, -1, dtype=np.int64)
    nonconvex_rank[bundle_rows] = np.arange(bundle_rows.size, dtype=np.int64)

    structure = _FastAllocationStructure(
        total_rows=total_rows,
        convex_mask=convex_mask,
        convex_first_rank=convex_first_rank,
        convex_second_rank=convex_second_rank,
        nonconvex_rank=nonconvex_rank,
        bundle_rows=bundle_rows,
        convex_row_rmq=_RangeMinimum(
            ordered_increment_rows,
            sentinel=total_rows,
        ),
        bundle_row_rmq=_RangeMinimum(bundle_rows, sentinel=total_rows),
    )

    # For a singleton inside the first m bundle ranks, the replacement set is
    # the first m+1 bundles except that singleton.  Its objective adjustment is
    # a-(a+b)=-b.  Equal adjustments prefer the later original singleton row,
    # leaving code 2 on the earlier row.  A singleton outside the first m
    # bundles contributes a directly; equal gains prefer the earlier row.
    bundle_count = bundle_rows.size
    prefix_singleton = np.full(bundle_count + 1, -1, dtype=np.int64)
    best_prefix = -1
    prefix_adjustment = [
        first - complete for first, complete in zip(bundle_first_gains, bundle_gains, strict=True)
    ]
    for stop in range(1, bundle_count + 1):
        rank = stop - 1
        if (
            best_prefix < 0
            or prefix_adjustment[rank] > prefix_adjustment[best_prefix]
            or (
                prefix_adjustment[rank] == prefix_adjustment[best_prefix]
                and bundle_rows[rank] > bundle_rows[best_prefix]
            )
        ):
            best_prefix = rank
        prefix_singleton[stop] = best_prefix

    suffix_singleton = np.full(bundle_count + 1, -1, dtype=np.int64)
    best_suffix = -1
    for start in range(bundle_count - 1, -1, -1):
        if (
            best_suffix < 0
            or bundle_first_gains[start] > bundle_first_gains[best_suffix]
            or (
                bundle_first_gains[start] == bundle_first_gains[best_suffix]
                and bundle_rows[start] < bundle_rows[best_suffix]
            )
        ):
            best_suffix = start
        suffix_singleton[start] = best_suffix

    best: _FastAllocationCandidate | None = None
    convex_increments = len(increments)

    # No nonconvex singleton.
    minimum_bundles = max(0, math.ceil((steps - convex_increments) / 2))
    maximum_bundles = min(bundle_count, steps // 2)
    for selected_bundles in range(minimum_bundles, maximum_bundles + 1):
        selected_convex = steps - 2 * selected_bundles
        candidate = _FastAllocationCandidate(
            convex_steps=selected_convex,
            bundle_count=selected_bundles,
            singleton_rank=None,
            gain=convex_prefix[selected_convex] + bundle_prefix[selected_bundles],
        )
        best = _better_fast_candidate(candidate, best, structure)

    # Exactly one nonconvex singleton.
    if bundle_count:
        minimum_bundles = max(
            0,
            math.ceil((steps - 1 - convex_increments) / 2),
        )
        maximum_bundles = min(bundle_count - 1, (steps - 1) // 2)
        for selected_bundles in range(minimum_bundles, maximum_bundles + 1):
            selected_convex = steps - 2 * selected_bundles - 1
            if selected_bundles:
                rank = int(prefix_singleton[selected_bundles])
                prefix_candidate = _FastAllocationCandidate(
                    convex_steps=selected_convex,
                    bundle_count=selected_bundles,
                    singleton_rank=rank,
                    gain=(
                        convex_prefix[selected_convex]
                        + bundle_prefix[selected_bundles + 1]
                        + prefix_adjustment[rank]
                    ),
                )
                best = _better_fast_candidate(prefix_candidate, best, structure)

            rank = int(suffix_singleton[selected_bundles])
            if rank >= 0:
                suffix_candidate = _FastAllocationCandidate(
                    convex_steps=selected_convex,
                    bundle_count=selected_bundles,
                    singleton_rank=rank,
                    gain=(
                        convex_prefix[selected_convex]
                        + bundle_prefix[selected_bundles]
                        + bundle_first_gains[rank]
                    ),
                )
                best = _better_fast_candidate(suffix_candidate, best, structure)

    if best is None:
        raise RuntimeError("fast exact multibit allocator found no feasible allocation")

    flat_codes = np.zeros(total_rows, dtype=np.uint8)
    if convex_count:
        flat_codes[convex_rows] = (convex_first_rank[convex_rows] < best.convex_steps).astype(
            np.uint8
        ) + (convex_second_rank[convex_rows] < best.convex_steps).astype(np.uint8)
    limit = _candidate_bundle_limit(best)
    if limit:
        flat_codes[bundle_rows[:limit]] = 2
    if best.singleton_rank is not None:
        flat_codes[bundle_rows[best.singleton_rank]] = 1
    if int(flat_codes.astype(np.int64).sum()) != steps:
        raise RuntimeError("fast exact multibit allocator changed the requested step budget")
    return torch.from_numpy(flat_codes.reshape(shape).copy())


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
    flattened row indices receive higher precision first. CPU-converted FP64
    distortions are represented in exact shared binary integer units, so this
    objective and its equality rule do not depend on summation order.

    This is a correctness/reference path. Its complexity is ``O(NK)`` time,
    ``O(NK)`` choice bytes, and ``O(K)`` arbitrary-precision integer workspace
    for ``N`` rows and budget ``K``.
    """

    shape, flat_distortions = _exact_flat_distortions(d4, d6, d8)
    total_rows = math.prod(shape)
    marginal_steps = _validate_marginal_steps(
        marginal_steps,
        total_rows=total_rows,
    )

    if marginal_steps == 0:
        return torch.zeros(shape, dtype=torch.uint8)
    if marginal_steps == 2 * total_rows:
        return torch.full(shape, 2, dtype=torch.uint8)

    next_cost: list[int | None] = [None] * (marginal_steps + 1)
    next_cost[0] = 0
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
        current_cost: list[int | None] = [None] * (marginal_steps + 1)
        current_choice = choices[row_index]

        for code in (0, 1, 2):
            available = min(next_maximum_steps, marginal_steps - code)
            if available < 0:
                continue
            row_cost = flat_distortions[row_index][code]
            for suffix_steps in range(available + 1):
                suffix_cost = next_cost[suffix_steps]
                if suffix_cost is None:
                    continue
                total_steps = suffix_steps + code
                candidate = suffix_cost + row_cost
                incumbent = current_cost[total_steps]
                if (
                    incumbent is None
                    or candidate < incumbent
                    or (candidate == incumbent and code > int(current_choice[total_steps]))
                ):
                    current_cost[total_steps] = candidate
                    current_choice[total_steps] = code

        next_cost = current_cost

    if next_cost[marginal_steps] is None:
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
        layer_index: 2 * quota - 8 for layer_index, quota in FROZEN_QWEN35_INT8_ROW_QUOTAS.items()
    }
    if any(not 0 <= steps <= 2 * QWEN35_ROWS_PER_LINEAR_LAYER for steps in budgets.values()):
        raise RuntimeError("frozen Qwen3.5 multibit step budget is outside [0, 2N]")
    if sum(budgets.values()) != QWEN35_FROZEN_TOTAL_MARGINAL_STEPS:
        raise RuntimeError("frozen Qwen3.5 multibit step budgets do not sum to 3,808")

    q4_payload_bytes = math.ceil(4 * QWEN35_VALUE_DIM / 8)
    q8_payload_bytes = math.ceil(8 * QWEN35_VALUE_DIM / 8)
    old_mask_bytes = math.ceil(QWEN35_ROWS_PER_LINEAR_LAYER * QWEN35_OLD_PRECISION_BITS_PER_ROW / 8)
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
            base_payload_and_scales + old_mask_bytes + quota * (q8_payload_bytes - q4_payload_bytes)
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
