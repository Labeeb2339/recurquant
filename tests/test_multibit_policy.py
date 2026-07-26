from __future__ import annotations

import itertools

import pytest
import torch

from recurquant.multibit_policy import (
    FROZEN_QWEN35_INT8_ROW_QUOTAS,
    QWEN35_FROZEN_TOTAL_MARGINAL_STEPS,
    allocate_exact_multibit_codes,
    frozen_qwen35_multibit_step_budgets,
)


def _objective(
    codes: torch.Tensor,
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
) -> float:
    stacked = torch.stack((d4, d6, d8), dim=-1).reshape(-1, 3).to(torch.float64)
    selected = stacked.gather(1, codes.reshape(-1, 1).to(torch.long))
    return float(selected.sum().item())


def _brute_force_codes(
    d4: torch.Tensor,
    d6: torch.Tensor,
    d8: torch.Tensor,
    *,
    marginal_steps: int,
) -> torch.Tensor:
    shape = d4.shape
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for codes in itertools.product((0, 1, 2), repeat=d4.numel()):
        if sum(codes) != marginal_steps:
            continue
        tensor = torch.tensor(codes, dtype=torch.uint8).reshape(shape)
        candidates.append((_objective(tensor, d4, d6, d8), codes))
    _, best = min(candidates, key=lambda item: (item[0], tuple(-code for code in item[1])))
    return torch.tensor(best, dtype=torch.uint8).reshape(shape)


def test_exact_allocator_handles_increasing_second_marginal_globally() -> None:
    # Row 0 gains only 1 unit at 4->6 but 100 at 6->8. An available-marginal
    # greedy allocator takes row 1's 60-unit first step and misses the optimum.
    d4 = torch.tensor([[101.0, 60.0]])
    d6 = torch.tensor([[100.0, 0.0]])
    d8 = torch.tensor([[0.0, 0.0]])

    codes = allocate_exact_multibit_codes(d4, d6, d8, marginal_steps=2)

    assert torch.equal(codes, torch.tensor([[2, 0]], dtype=torch.uint8))
    assert codes.sum().item() == 2
    assert _objective(codes, d4, d6, d8) == 60.0


@pytest.mark.parametrize("marginal_steps", range(9))
def test_exact_allocator_matches_exhaustive_search(marginal_steps: int) -> None:
    d4 = torch.tensor([[8.0, 2.0], [5.0, 9.0]], dtype=torch.float64)
    d6 = torch.tensor([[7.0, 3.0], [1.0, 4.0]], dtype=torch.float64)
    d8 = torch.tensor([[0.0, 1.0], [2.0, 6.0]], dtype=torch.float64)

    actual = allocate_exact_multibit_codes(
        d4,
        d6,
        d8,
        marginal_steps=marginal_steps,
    )
    expected = _brute_force_codes(
        d4,
        d6,
        d8,
        marginal_steps=marginal_steps,
    )

    assert torch.equal(actual, expected)
    assert actual.dtype == torch.uint8
    assert actual.device.type == "cpu"
    assert tuple(actual.shape) == (2, 2)
    assert actual.sum().item() == marginal_steps


def test_exact_ties_give_higher_codes_to_earlier_flattened_rows() -> None:
    equal = torch.ones((2, 2))

    codes = allocate_exact_multibit_codes(
        equal,
        equal,
        equal,
        marginal_steps=3,
    )

    assert torch.equal(codes, torch.tensor([[2, 1], [0, 0]], dtype=torch.uint8))


def test_endpoint_budgets_return_all_int4_or_all_int8() -> None:
    d4 = torch.ones((2, 3))
    d6 = torch.ones((2, 3))
    d8 = torch.ones((2, 3))

    low = allocate_exact_multibit_codes(d4, d6, d8, marginal_steps=0)
    high = allocate_exact_multibit_codes(d4, d6, d8, marginal_steps=12)

    assert torch.equal(low, torch.zeros((2, 3), dtype=torch.uint8))
    assert torch.equal(high, torch.full((2, 3), 2, dtype=torch.uint8))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ((torch.tensor([[float("nan")]]), torch.ones((1, 1)), torch.ones((1, 1))), "finite"),
        ((-torch.ones((1, 1)), torch.ones((1, 1)), torch.ones((1, 1))), "nonnegative"),
        ((torch.ones(1), torch.ones(1), torch.ones(1)), r"shape \[heads, rows\]"),
        (
            (torch.ones((1, 2)), torch.ones((2, 1)), torch.ones((1, 2))),
            "identical shapes",
        ),
    ],
)
def test_invalid_distortion_values_and_shapes_are_rejected(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        allocate_exact_multibit_codes(*values, marginal_steps=0)


def test_integer_distortions_and_invalid_step_budgets_are_rejected() -> None:
    integer = torch.ones((1, 2), dtype=torch.int64)
    floating = torch.ones((1, 2))

    with pytest.raises(TypeError, match="floating-point"):
        allocate_exact_multibit_codes(integer, floating, floating, marginal_steps=1)
    with pytest.raises(TypeError, match="integer"):
        allocate_exact_multibit_codes(floating, floating, floating, marginal_steps=True)
    with pytest.raises(ValueError, match="representable range"):
        allocate_exact_multibit_codes(floating, floating, floating, marginal_steps=5)


def test_frozen_qwen35_step_budgets_match_old_bytes_exactly() -> None:
    budgets = frozen_qwen35_multibit_step_budgets()

    assert budgets == {
        0: 702,
        1: 752,
        2: 530,
        4: 350,
        5: 362,
        6: 202,
        8: 152,
        9: 78,
        10: 160,
        12: 52,
        13: 116,
        14: 100,
        16: 82,
        17: 46,
        18: 6,
        20: 10,
        21: 6,
        22: 102,
    }
    assert sum(FROZEN_QWEN35_INT8_ROW_QUOTAS.values()) == 1_976
    assert sum(budgets.values()) == QWEN35_FROZEN_TOTAL_MARGINAL_STEPS == 3_808

    rows = 16 * 128
    old_base_per_layer = rows * (64 + 2) + 256
    multibit_base_per_layer = rows * (64 + 2) + 512
    for layer_index, old_quota in FROZEN_QWEN35_INT8_ROW_QUOTAS.items():
        assert old_base_per_layer + 64 * old_quota == (
            multibit_base_per_layer + 32 * budgets[layer_index]
        )
