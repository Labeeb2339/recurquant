from __future__ import annotations

import pytest
import torch

from recurquant.finite_difference import check_directional_derivative


def test_quadratic_directional_derivative_matches_central_difference() -> None:
    state = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)
    direction = torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64)

    result = check_directional_derivative(
        lambda value: value.square().sum(),
        state,
        direction,
    )

    expected = float((2 * state * direction).sum().item())
    assert result.autograd_derivative == pytest.approx(expected)
    assert result.gradient_l2_norm == pytest.approx(float(torch.linalg.vector_norm(2 * state)))
    assert result.direction_l2_norm == pytest.approx(float(torch.linalg.vector_norm(direction)))
    assert all(point.central_derivative == pytest.approx(expected) for point in result.points)
    assert result.median_relative_error < 1e-12
    assert result.nonzero_sign_agreement_rate == 1.0


def test_nonlinear_central_difference_error_shrinks_with_epsilon() -> None:
    state = torch.tensor([0.7, -0.4], dtype=torch.float64)
    direction = torch.tensor([0.3, 0.2], dtype=torch.float64)

    result = check_directional_derivative(
        lambda value: value.pow(3).sum(),
        state,
        direction,
        epsilons=(0.5, 0.25, 0.125, 0.0625),
    )

    errors = [point.absolute_error for point in result.points]
    assert errors == sorted(errors, reverse=True)
    assert result.points[-1].relative_error < result.points[0].relative_error
    assert result.nonzero_sign_agreement_rate == 1.0


def test_near_zero_derivative_preserves_error_without_forcing_a_sign() -> None:
    state = torch.zeros(3, dtype=torch.float32)
    direction = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)

    result = check_directional_derivative(
        lambda value: value.square().sum(),
        state,
        direction,
        epsilons=(0.25, 0.125),
        near_zero_floor=1e-6,
    )

    assert result.autograd_derivative == 0.0
    assert all(point.near_zero for point in result.points)
    assert all(point.sign_agreement is None for point in result.points)
    assert result.nonzero_sign_agreement_rate is None


@pytest.mark.parametrize(
    ("loss_fn", "expected_autograd", "expected_central"),
    (
        (lambda value: value[0].pow(3) - value[0], -1.0, 0.0),
        (lambda value: value[0].pow(3), 0.0, 1.0),
    ),
)
def test_exactly_zero_derivative_does_not_manufacture_sign_agreement(
    loss_fn: object,
    expected_autograd: float,
    expected_central: float,
) -> None:
    result = check_directional_derivative(
        loss_fn,  # type: ignore[arg-type]
        torch.zeros(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        epsilons=(1.0,),
        near_zero_floor=1e-8,
    )

    assert result.autograd_derivative == expected_autograd
    assert result.points[0].central_derivative == expected_central
    assert not result.points[0].near_zero
    assert result.points[0].sign_agreement is False
    assert result.nonzero_sign_agreement_rate == 0.0


def test_directional_derivative_validation_rejects_bad_inputs_and_losses() -> None:
    state = torch.ones(2, dtype=torch.float32)
    direction = torch.ones(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="same shape"):
        check_directional_derivative(lambda value: value.sum(), state, direction[:1])
    with pytest.raises(ValueError, match="positive finite"):
        check_directional_derivative(
            lambda value: value.sum(),
            state,
            direction,
            epsilons=(0.1, 0.0),
        )
    with pytest.raises(ValueError, match="one floating-point scalar"):
        check_directional_derivative(lambda value: value, state, direction)
    with pytest.raises(ValueError, match="finite"):
        check_directional_derivative(
            lambda value: value.sum() * torch.tensor(float("nan")),
            state,
            direction,
        )
