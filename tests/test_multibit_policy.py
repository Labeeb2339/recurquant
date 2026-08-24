from __future__ import annotations

import itertools
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

import recurquant.static_q468_artifact_contract as artifact_contract
from recurquant.multibit_policy import (
    FROZEN_QWEN35_INT8_ROW_QUOTAS,
    QWEN35_FROZEN_TOTAL_MARGINAL_STEPS,
    allocate_exact_multibit_codes,
    allocate_exact_multibit_codes_fast,
    frozen_qwen35_multibit_step_budgets,
)

ALLOCATORS = (allocate_exact_multibit_codes, allocate_exact_multibit_codes_fast)


def test_artifact_contract_imports_in_isolated_process_without_torch() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "recurquant"
        / "static_q468_artifact_contract.py"
    )
    script = r"""
import builtins
import importlib.util
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.partition(".")[0]
    if root in {"recurquant", "torch"}:
        raise AssertionError(f"forbidden import attempted: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location(
    "static_q468_artifact_contract_isolated",
    sys.argv[1],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("recurquant") for name in sys.modules)
assert module.FROZEN_STATIC_Q468_PRIMARY_STEPS == 29_334
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(module_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    flattened = torch.stack((d4, d6, d8), dim=-1).reshape(-1, 3).to(torch.float64)
    exact = [tuple(Fraction.from_float(float(value)) for value in row) for row in flattened]
    candidates: list[tuple[Fraction, tuple[int, ...]]] = []
    for codes in itertools.product((0, 1, 2), repeat=d4.numel()):
        if sum(codes) != marginal_steps:
            continue
        objective = sum(
            (exact[row][code] for row, code in enumerate(codes)),
            start=Fraction(),
        )
        candidates.append((objective, codes))
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
    fast = allocate_exact_multibit_codes_fast(
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
    assert torch.equal(fast, expected)
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


@pytest.mark.parametrize("seed", range(32))
def test_fast_exact_allocator_matches_dynamic_program_for_every_small_budget(
    seed: int,
) -> None:
    rows = 1 + seed % 7
    generator = torch.Generator().manual_seed(2339 + seed)
    if seed % 2:
        distortions = [
            torch.randint(
                0,
                19,
                (1, rows),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.float64)
            for _ in range(3)
        ]
    else:
        distortions = [
            torch.rand((1, rows), generator=generator, dtype=torch.float64) for _ in range(3)
        ]

    for marginal_steps in range(2 * rows + 1):
        expected = allocate_exact_multibit_codes(
            *distortions,
            marginal_steps=marginal_steps,
        )
        actual = allocate_exact_multibit_codes_fast(
            *distortions,
            marginal_steps=marginal_steps,
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "distortions",
    [
        (
            torch.tensor([[5.0, 5.0]], dtype=torch.float64),
            torch.tensor([[4.0, 8.0]], dtype=torch.float64),
            torch.tensor([[3.0, 7.0]], dtype=torch.float64),
        ),
        (
            torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float64),
            torch.tensor([[1.0, 2.0, 1.0, 2.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        ),
        (
            torch.ones((1, 6), dtype=torch.float64),
            torch.ones((1, 6), dtype=torch.float64),
            torch.ones((1, 6), dtype=torch.float64),
        ),
        (
            torch.tensor(
                [[2.0**900, 2.0**-900, 3.0, 2.0**500]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[2.0**899, 0.0, 3.0, 2.0**499]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[0.0, 2.0**-901, 3.0, 0.0]],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_fast_allocator_matches_exact_oracles_on_adversarial_nonconvex_and_ties(
    distortions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    rows = distortions[0].numel()
    for marginal_steps in range(2 * rows + 1):
        dynamic = allocate_exact_multibit_codes(
            *distortions,
            marginal_steps=marginal_steps,
        )
        fast = allocate_exact_multibit_codes_fast(
            *distortions,
            marginal_steps=marginal_steps,
        )
        brute = _brute_force_codes(
            *distortions,
            marginal_steps=marginal_steps,
        )
        assert torch.equal(fast, dynamic)
        assert torch.equal(fast, brute)


@pytest.mark.parametrize(
    "distortions",
    [
        (
            torch.tensor([[101.0, 60.0]], dtype=torch.float64),
            torch.tensor([[100.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        ),
        (
            torch.ones((2, 2), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
        ),
        (
            torch.tensor([[5.0, 5.0, 2.0, 2.0]], dtype=torch.float64),
            torch.tensor([[4.0, 8.0, 1.0, 2.0]], dtype=torch.float64),
            torch.tensor([[3.0, 7.0, 0.0, 0.0]], dtype=torch.float64),
        ),
    ],
    ids=("increasing-second-marginal", "exact-ties", "mixed-nonconvex"),
)
def test_torch_free_artifact_contract_allocator_matches_exact_oracles_for_every_budget(
    distortions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    numpy_distortions = tuple(value.numpy() for value in distortions)
    rows = distortions[0].numel()

    for marginal_steps in range(2 * rows + 1):
        dynamic = allocate_exact_multibit_codes(
            *distortions,
            marginal_steps=marginal_steps,
        )
        fast = allocate_exact_multibit_codes_fast(
            *distortions,
            marginal_steps=marginal_steps,
        )
        brute = _brute_force_codes(
            *distortions,
            marginal_steps=marginal_steps,
        )
        contract_codes = artifact_contract.allocate_exact_multibit_codes(
            *numpy_distortions,
            marginal_steps=marginal_steps,
        )

        assert np.array_equal(contract_codes, dynamic.numpy())
        assert np.array_equal(contract_codes, fast.numpy())
        assert np.array_equal(contract_codes, brute.numpy())
        assert contract_codes.dtype == np.uint8
        assert contract_codes.flags.c_contiguous
        assert int(contract_codes.astype(np.int64).sum()) == marginal_steps


def test_exact_optimum_has_at_most_one_nonconvex_singleton() -> None:
    generator = torch.Generator().manual_seed(9917)
    for _ in range(64):
        distortions = [
            torch.randint(
                0,
                13,
                (1, 6),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.float64)
            for _ in range(3)
        ]
        first_gain = distortions[0] - distortions[1]
        second_gain = distortions[1] - distortions[2]
        nonconvex = first_gain < second_gain
        for marginal_steps in range(13):
            codes = allocate_exact_multibit_codes_fast(
                *distortions,
                marginal_steps=marginal_steps,
            )
            assert int(((codes == 1) & nonconvex).sum().item()) <= 1


def test_fast_allocator_runs_the_frozen_complete_state_budget() -> None:
    total_rows = 36_864
    row = torch.arange(total_rows, dtype=torch.float64)
    d4 = (((17 * row + 13) % 1009) / 1009).reshape(18, 2048)
    d6 = (((29 * row + 7) % 1013) / 1013).reshape(18, 2048)
    d8 = (((43 * row + 3) % 1019) / 1019).reshape(18, 2048)

    codes = allocate_exact_multibit_codes_fast(
        d4,
        d6,
        d8,
        marginal_steps=27_030,
    )

    assert codes.dtype == torch.uint8
    assert codes.device.type == "cpu"
    assert tuple(codes.shape) == (18, 2048)
    assert int(codes.to(torch.int64).sum().item()) == 27_030


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_endpoint_budgets_return_all_int4_or_all_int8(allocator) -> None:
    d4 = torch.ones((2, 3))
    d6 = torch.ones((2, 3))
    d8 = torch.ones((2, 3))

    low = allocator(d4, d6, d8, marginal_steps=0)
    high = allocator(d4, d6, d8, marginal_steps=12)

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
@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_invalid_distortion_values_and_shapes_are_rejected(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    match: str,
    allocator,
) -> None:
    with pytest.raises(ValueError, match=match):
        allocator(*values, marginal_steps=0)


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_integer_distortions_and_invalid_step_budgets_are_rejected(allocator) -> None:
    integer = torch.ones((1, 2), dtype=torch.int64)
    floating = torch.ones((1, 2))

    with pytest.raises(TypeError, match="floating-point"):
        allocator(integer, floating, floating, marginal_steps=1)
    with pytest.raises(TypeError, match="integer"):
        allocator(floating, floating, floating, marginal_steps=True)
    with pytest.raises(ValueError, match="representable range"):
        allocator(floating, floating, floating, marginal_steps=5)


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
