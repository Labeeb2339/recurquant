from __future__ import annotations

import pytest
import torch

from recurquant.horizon import (
    finite_horizon_row_read_risk,
    finite_horizon_row_read_risk_from_energies,
)


def _brute_force_scores(
    errors: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    horizon: int,
) -> torch.Tensor:
    time_steps, batch_size, heads, key_dim, value_dim = errors.shape
    result = torch.zeros((time_steps, batch_size, heads, key_dim), dtype=torch.float32)
    scale = key_dim**-0.5
    for start in range(time_steps):
        stop = min(time_steps, start + horizon)
        for batch in range(batch_size):
            for head in range(heads):
                for source_row in range(key_dim):
                    propagated = torch.zeros((key_dim, value_dim), dtype=torch.float32)
                    propagated[source_row] = errors[start, batch, head, source_row]
                    energy = 0.0
                    for step in range(start, stop):
                        key = keys[step, batch, head]
                        projected = (key.unsqueeze(-1) * propagated).sum(dim=0)
                        propagated = log_decays[step, batch, head].exp() * (
                            propagated - betas[step, batch, head] * key.unsqueeze(-1) * projected
                        )
                        read = (queries[step, batch, head] * scale) @ propagated
                        energy += float(read.square().sum().item())
                    result[start, batch, head, source_row] = energy / (stop - start)
    return result


def test_horizon_scores_match_brute_force_isolated_row_propagation() -> None:
    generator = torch.Generator().manual_seed(2339)
    errors = torch.randn((4, 1, 2, 3, 2), generator=generator) * 0.05
    queries = torch.nn.functional.normalize(torch.randn((4, 1, 2, 3), generator=generator), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn((4, 1, 2, 3), generator=generator), dim=-1)
    log_decays = -torch.rand((4, 1, 2), generator=generator) * 0.3
    betas = torch.rand((4, 1, 2), generator=generator)

    result = finite_horizon_row_read_risk(
        errors,
        queries,
        keys,
        log_decays,
        betas,
        horizon=3,
        normalize_qk=False,
    )
    expected = _brute_force_scores(
        errors,
        queries,
        keys,
        log_decays,
        betas,
        horizon=3,
    )

    assert result.horizon == 3
    assert torch.allclose(result.per_state_scores, expected, atol=1e-7, rtol=1e-5)
    assert torch.allclose(result.scores, expected.mean(dim=(0, 1)), atol=1e-7, rtol=1e-5)
    assert result.future_read_counts.tolist() == [3, 3, 2, 1]


def test_zero_state_error_has_zero_horizon_risk() -> None:
    errors = torch.zeros((2, 1, 1, 3, 4))
    queries = torch.ones((2, 1, 1, 3))
    keys = torch.ones((2, 1, 1, 3))
    log_decays = torch.zeros((2, 1, 1))
    betas = torch.full((2, 1, 1), 0.5)

    result = finite_horizon_row_read_risk(errors, queries, keys, log_decays, betas)

    assert torch.count_nonzero(result.scores).item() == 0


def test_row_energy_api_matches_full_error_api_exactly() -> None:
    generator = torch.Generator().manual_seed(72)
    errors = torch.randn((5, 2, 3, 4, 7), generator=generator) * 0.03
    queries = torch.randn((5, 2, 3, 4), generator=generator)
    keys = torch.randn((5, 2, 3, 4), generator=generator)
    log_decays = -torch.rand((5, 2, 3), generator=generator)
    betas = torch.rand((5, 2, 3), generator=generator)

    full = finite_horizon_row_read_risk(
        errors,
        queries,
        keys,
        log_decays,
        betas,
        horizon=4,
    )
    energies = errors.to(torch.float32).square().sum(dim=-1)
    reduced = finite_horizon_row_read_risk_from_energies(
        energies,
        queries,
        keys,
        log_decays,
        betas,
        horizon=4,
    )

    assert full.horizon == reduced.horizon
    assert torch.equal(full.future_read_counts, reduced.future_read_counts)
    assert torch.equal(full.per_state_scores, reduced.per_state_scores)
    assert torch.equal(full.scores, reduced.scores)


def test_row_energy_api_rejects_negative_energy() -> None:
    energies = torch.zeros((1, 1, 1, 2))
    energies[..., 0] = -1
    vectors = torch.ones((1, 1, 1, 2))
    scalars = torch.zeros((1, 1, 1))

    with pytest.raises(ValueError, match="non-negative"):
        finite_horizon_row_read_risk_from_energies(
            energies,
            vectors,
            vectors,
            scalars,
            scalars,
        )


def test_row_energy_api_rejects_shape_and_dtype_mismatch() -> None:
    energies = torch.zeros((2, 1, 1, 3))
    vectors = torch.ones((2, 1, 1, 3))
    scalars = torch.zeros((2, 1, 1))

    with pytest.raises(ValueError, match="query/key dimensions"):
        finite_horizon_row_read_risk_from_energies(
            energies,
            vectors[..., :2],
            vectors,
            scalars,
            scalars,
        )
    with pytest.raises(TypeError, match="floating-point"):
        finite_horizon_row_read_risk_from_energies(
            energies.to(torch.int64),
            vectors,
            vectors,
            scalars,
            scalars,
        )


@pytest.mark.parametrize("horizon", [0, -1, True])
def test_invalid_horizon_is_rejected(horizon: int) -> None:
    errors = torch.zeros((1, 1, 1, 2, 2))
    vectors = torch.ones((1, 1, 1, 2))
    scalars = torch.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="positive integer"):
        finite_horizon_row_read_risk(
            errors,
            vectors,
            vectors,
            scalars,
            scalars,
            horizon=horizon,
        )
