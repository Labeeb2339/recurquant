"""Finite-horizon sensitivity for Gated DeltaNet recurrent-state rows.

The score follows the linearized error dynamics of the recurrent update while
holding the full-precision q/k/g/beta trajectory fixed. It is a calibration
primitive, not evidence that a selected policy improves language-model quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HorizonReadRisk:
    """Per-row finite-horizon read-risk scores and their aligned components."""

    horizon: int
    scores: torch.Tensor
    per_state_scores: torch.Tensor
    future_read_counts: torch.Tensor


def _l2_normalize(value: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + epsilon)


def _validate_energy_inputs(
    row_error_energies: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    horizon: int,
    epsilon: float,
) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if row_error_energies.ndim != 4:
        raise ValueError("row_error_energies must have shape [time, batch, heads, key_dim]")
    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError("queries and keys must have shape [time, batch, heads, key_dim]")
    if log_decays.ndim != 3 or betas.ndim != 3:
        raise ValueError("log_decays and betas must have shape [time, batch, heads]")
    if queries.shape != row_error_energies.shape or keys.shape != row_error_energies.shape:
        raise ValueError(
            "query/key dimensions must match row_error_energies time, batch, heads, and key_dim"
        )
    expected_scalar_shape = row_error_energies.shape[:3]
    if log_decays.shape != expected_scalar_shape or betas.shape != expected_scalar_shape:
        raise ValueError(
            "log_decay/beta dimensions must match row_error_energies time, batch, and heads"
        )
    if row_error_energies.shape[0] == 0:
        raise ValueError("the trace must contain at least one token")
    tensors = (row_error_energies, queries, keys, log_decays, betas)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("all horizon inputs must use floating-point dtypes")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all horizon inputs must be on the same device")
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("all horizon inputs must be finite")
    if (row_error_energies < 0).any().item():
        raise ValueError("row_error_energies must be non-negative")


def _validate_full_error_inputs(state_errors: torch.Tensor) -> None:
    if state_errors.ndim != 5:
        raise ValueError("state_errors must have shape [time, batch, heads, key_dim, value_dim]")
    if not state_errors.is_floating_point():
        raise TypeError("all horizon inputs must use floating-point dtypes")


def _right_apply_transition(
    row_vector: torch.Tensor,
    *,
    key: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply ``g * (I - beta * k k^T)`` to a row vector from the right."""

    projected = (row_vector * key).sum(dim=-1, keepdim=True)
    corrected = row_vector - beta.unsqueeze(-1) * projected * key
    return log_decay.exp().unsqueeze(-1) * corrected


def finite_horizon_row_read_risk_from_energies(
    row_error_energies: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    horizon: int = 32,
    normalize_qk: bool = True,
    epsilon: float = 1e-6,
) -> HorizonReadRisk:
    """Score per-row error energies over future Gated DeltaNet reads.

    ``row_error_energies[t, b, h, r]`` is the squared L2 norm of row ``r`` in
    the recurrent-state quantization error just before token ``t``. This is the
    sufficient error statistic for isolated-row scoring: it avoids retaining a
    ``value_dim``-wide error matrix for every token while producing the same
    result as :func:`finite_horizon_row_read_risk`.

    For each future read, the function propagates the query backwards through
    the analytic Gated DeltaNet error transition

    ``E' = exp(g) * (I - beta * k k^T) * E``.

    The returned ``scores`` have shape ``[heads, key_dim]`` and average the
    isolated squared read contribution over trace positions and batch items.
    Cross-row cancellation is intentionally excluded so rows can be ranked.
    """

    _validate_energy_inputs(
        row_error_energies,
        queries,
        keys,
        log_decays,
        betas,
        horizon=horizon,
        epsilon=epsilon,
    )
    working_energies = row_error_energies.to(torch.float32)
    working_queries = queries.to(torch.float32)
    working_keys = keys.to(torch.float32)
    working_decays = log_decays.to(torch.float32)
    working_betas = betas.to(torch.float32)
    if normalize_qk:
        working_queries = _l2_normalize(working_queries, epsilon=epsilon)
        working_keys = _l2_normalize(working_keys, epsilon=epsilon)

    time_steps, batch_size, heads, key_dim = working_energies.shape
    query_scale = key_dim**-0.5
    influence = torch.zeros(
        (time_steps, batch_size, heads, key_dim),
        dtype=torch.float32,
        device=working_energies.device,
    )
    future_read_counts = torch.zeros(
        time_steps,
        dtype=torch.int64,
        device=working_energies.device,
    )

    for read_index in range(time_steps):
        propagated_query = working_queries[read_index] * query_scale
        earliest_start = max(0, read_index - horizon + 1)
        for start_index in range(read_index, earliest_start - 1, -1):
            propagated_query = _right_apply_transition(
                propagated_query,
                key=working_keys[start_index],
                log_decay=working_decays[start_index],
                beta=working_betas[start_index],
            )
            influence[start_index] += propagated_query.square()
            future_read_counts[start_index] += 1

    normalized_influence = influence / future_read_counts.clamp_min(1).view(-1, 1, 1, 1)
    per_state_scores = normalized_influence * working_energies
    scores = per_state_scores.mean(dim=(0, 1))
    return HorizonReadRisk(
        horizon=min(horizon, time_steps),
        scores=scores,
        per_state_scores=per_state_scores,
        future_read_counts=future_read_counts,
    )


def finite_horizon_row_read_risk(
    state_errors: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    log_decays: torch.Tensor,
    betas: torch.Tensor,
    *,
    horizon: int = 32,
    normalize_qk: bool = True,
    epsilon: float = 1e-6,
) -> HorizonReadRisk:
    """Score isolated recurrent-state row errors over future GDN reads.

    ``state_errors[t]`` is the quantization error in the recurrent state just
    before token ``t``. For each future read, the function propagates the query
    backwards through the analytic Gated DeltaNet error transition

    ``E' = exp(g) * (I - beta * k k^T) * E``.

    This avoids expanding a separate full error matrix for every source row.
    The returned ``scores`` have shape ``[heads, key_dim]`` and average the
    isolated squared read contribution over trace positions and batch items.
    Cross-row cancellation is intentionally excluded so rows can be ranked.
    """

    _validate_full_error_inputs(state_errors)
    row_error_energies = state_errors.to(torch.float32).square().sum(dim=-1)
    return finite_horizon_row_read_risk_from_energies(
        row_error_energies,
        queries,
        keys,
        log_decays,
        betas,
        horizon=horizon,
        normalize_qk=normalize_qk,
        epsilon=epsilon,
    )
