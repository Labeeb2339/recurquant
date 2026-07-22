"""Local finite-difference checks for recurrent-state promotion directions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

import torch

ScalarLoss = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class FiniteDifferencePoint:
    """One central-difference comparison at a fixed interpolation scale."""

    epsilon: float
    plus_loss: float
    minus_loss: float
    central_derivative: float
    absolute_error: float
    relative_error: float
    near_zero: bool
    sign_agreement: bool | None


@dataclass(frozen=True, slots=True)
class DirectionalDerivativeCheck:
    """Autograd derivative and its central finite-difference convergence trace."""

    base_loss: float
    autograd_derivative: float
    gradient_l2_norm: float
    direction_l2_norm: float
    near_zero_floor: float
    points: tuple[FiniteDifferencePoint, ...]

    @property
    def median_relative_error(self) -> float:
        return float(median(point.relative_error for point in self.points))

    @property
    def nonzero_sign_agreement_rate(self) -> float | None:
        agreements = [
            point.sign_agreement for point in self.points if point.sign_agreement is not None
        ]
        if not agreements:
            return None
        return sum(agreements) / len(agreements)


def _scalar_loss(loss_fn: ScalarLoss, state: torch.Tensor, *, name: str) -> torch.Tensor:
    loss = loss_fn(state)
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} loss_fn output must be a torch.Tensor")
    if loss.numel() != 1 or not loss.is_floating_point():
        raise ValueError(f"{name} loss_fn output must be one floating-point scalar")
    if not torch.isfinite(loss).item():
        raise ValueError(f"{name} loss_fn output must be finite")
    return loss.reshape(())


def check_directional_derivative(
    loss_fn: ScalarLoss,
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilons: Sequence[float] = (0.25, 0.125, 0.0625, 0.03125),
    near_zero_floor: float = 1e-8,
) -> DirectionalDerivativeCheck:
    """Compare ``grad(loss) dot direction`` with central differences.

    This checks the local differentiable loss supplied by ``loss_fn``. It does
    not validate a hard quantizer, a repeated-QDQ suffix, or a complete packed
    policy unless the caller explicitly builds those semantics into the loss
    function. Near-zero comparisons retain their absolute/relative errors and
    use ``sign_agreement=None`` instead of manufacturing a pass.
    """

    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if not isinstance(state, torch.Tensor) or not isinstance(direction, torch.Tensor):
        raise TypeError("state and direction must be torch.Tensor values")
    if state.shape != direction.shape:
        raise ValueError("state and direction must have the same shape")
    if state.numel() == 0:
        raise ValueError("state and direction must not be empty")
    if not state.is_floating_point() or not direction.is_floating_point():
        raise TypeError("state and direction must use floating-point dtypes")
    if state.dtype != direction.dtype or state.device != direction.device:
        raise ValueError("state and direction must share dtype and device")
    if state.device.type == "meta":
        raise ValueError("state and direction must be materialized")
    if not torch.isfinite(state).all().item() or not torch.isfinite(direction).all().item():
        raise ValueError("state and direction must contain only finite values")
    if not math.isfinite(near_zero_floor) or near_zero_floor <= 0:
        raise ValueError("near_zero_floor must be a positive finite number")

    rendered_epsilons = tuple(float(value) for value in epsilons)
    if not rendered_epsilons:
        raise ValueError("epsilons must contain at least one value")
    if any(not math.isfinite(value) or value <= 0 for value in rendered_epsilons):
        raise ValueError("epsilons must contain only positive finite values")
    if len(set(rendered_epsilons)) != len(rendered_epsilons):
        raise ValueError("epsilons must not contain duplicates")

    leaf = state.detach().clone().requires_grad_(True)
    base = _scalar_loss(loss_fn, leaf, name="base")
    (gradient,) = torch.autograd.grad(
        base,
        leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    if not torch.isfinite(gradient).all().item():
        raise RuntimeError("loss gradient became non-finite")
    autograd_value = float((gradient * direction).sum().detach().item())

    points: list[FiniteDifferencePoint] = []
    with torch.no_grad():
        for epsilon in rendered_epsilons:
            plus = _scalar_loss(
                loss_fn,
                state.detach() + epsilon * direction.detach(),
                name="plus",
            )
            minus = _scalar_loss(
                loss_fn,
                state.detach() - epsilon * direction.detach(),
                name="minus",
            )
            plus_value = float(plus.item())
            minus_value = float(minus.item())
            central = (plus_value - minus_value) / (2.0 * epsilon)
            absolute_error = abs(central - autograd_value)
            scale = max(abs(central), abs(autograd_value), near_zero_floor)
            near_zero = max(abs(central), abs(autograd_value)) <= near_zero_floor
            if near_zero:
                sign_agreement = None
            elif central == 0.0 or autograd_value == 0.0:
                sign_agreement = False
            else:
                sign_agreement = math.copysign(1.0, central) == math.copysign(
                    1.0, autograd_value
                )
            points.append(
                FiniteDifferencePoint(
                    epsilon=epsilon,
                    plus_loss=plus_value,
                    minus_loss=minus_value,
                    central_derivative=central,
                    absolute_error=absolute_error,
                    relative_error=absolute_error / scale,
                    near_zero=near_zero,
                    sign_agreement=sign_agreement,
                )
            )

    return DirectionalDerivativeCheck(
        base_loss=float(base.detach().item()),
        autograd_derivative=autograd_value,
        gradient_l2_norm=float(torch.linalg.vector_norm(gradient).item()),
        direction_l2_norm=float(torch.linalg.vector_norm(direction).item()),
        near_zero_floor=near_zero_floor,
        points=tuple(points),
    )
