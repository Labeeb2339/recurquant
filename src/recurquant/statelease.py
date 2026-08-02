"""Reference primitives for recurrence-aware Gated DeltaNet checkpoint leasing.

StateLease keeps a quantized recurrent-state checkpoint and reconstructs the
current state by replaying a bounded sequence of Gated DeltaNet transitions.
This module contains only the architecture-level recurrence, its conservative
error propagation rule, and the causal checkpoint decision. Cache integration
and physical storage accounting live in :mod:`recurquant.packed_cache`.

The implementation is correctness-first. It materializes FP32 workspaces and
makes no fused-kernel or latency claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

STATELEASE_L2NORM_EPS = 1e-6


def _validate_transition_tensor(
    value: torch.Tensor,
    *,
    name: str,
    rank: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_l2norm_eps(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("l2norm_eps must be a positive finite real number")
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("l2norm_eps must be a positive finite real number")
    return epsilon


def normalize_gated_delta_key(
    key: torch.Tensor,
    *,
    l2norm_eps: float = STATELEASE_L2NORM_EPS,
) -> torch.Tensor:
    """Normalize a Qwen3.5 key exactly as the pinned state kernel consumes it.

    Transformers 5.14.1 performs L2 normalization in the kernel input dtype
    and only then converts the normalized key to FP32 for the recurrence.
    That ordering is observable for BF16 and FP16 inputs, so casting the raw
    key before normalization defines a different transition.
    """

    epsilon = _validate_l2norm_eps(l2norm_eps)
    _validate_transition_tensor(key, name="key", rank=4)
    with torch.no_grad():
        source = key.detach()
        normalized = source * torch.rsqrt((source * source).sum(dim=-1, keepdim=True) + epsilon)
        normalized32 = normalized.to(torch.float32)
        if not torch.isfinite(normalized32).all().item():
            raise RuntimeError("Gated DeltaNet key normalization produced non-finite values")
    return normalized32


def replay_gated_delta_state(
    initial_state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    l2norm_eps: float = STATELEASE_L2NORM_EPS,
) -> torch.Tensor:
    """Replay chronological Qwen3.5 Gated DeltaNet transitions in FP32.

    ``initial_state`` has shape ``[batch, heads, key_rows, value_width]``.
    ``key`` and ``value`` have shape ``[batch, tokens, heads, width]`` while
    ``log_decay`` and ``beta`` have shape ``[batch, tokens, heads]``.

    Qwen3.5 normalizes keys inside its state kernel. The reference replay
    mirrors that operation and the exact recurrent update used by the pinned
    Transformers implementation. The result is returned in the initial
    state's dtype.
    """

    epsilon = _validate_l2norm_eps(l2norm_eps)
    _validate_transition_tensor(initial_state, name="initial_state", rank=4)
    _validate_transition_tensor(key, name="key", rank=4)
    _validate_transition_tensor(value, name="value", rank=4)
    _validate_transition_tensor(log_decay, name="log_decay", rank=3)
    _validate_transition_tensor(beta, name="beta", rank=3)

    batch, heads, key_rows, value_width = initial_state.shape
    token_count = key.shape[1]
    expected_key = (batch, token_count, heads, key_rows)
    expected_value = (batch, token_count, heads, value_width)
    expected_gate = (batch, token_count, heads)
    if tuple(key.shape) != expected_key:
        raise ValueError(f"key must have shape {expected_key}; got {tuple(key.shape)}")
    if tuple(value.shape) != expected_value:
        raise ValueError(f"value must have shape {expected_value}; got {tuple(value.shape)}")
    if tuple(log_decay.shape) != expected_gate:
        raise ValueError(f"log_decay must have shape {expected_gate}; got {tuple(log_decay.shape)}")
    if tuple(beta.shape) != expected_gate:
        raise ValueError(f"beta must have shape {expected_gate}; got {tuple(beta.shape)}")
    if token_count <= 0:
        raise ValueError("transition sequence must contain at least one token")
    devices = {
        initial_state.device,
        key.device,
        value.device,
        log_decay.device,
        beta.device,
    }
    if len(devices) != 1:
        raise ValueError("initial state and transition tensors must share one device")
    if (log_decay > 0).any().item():
        raise ValueError("Qwen3.5 log_decay must be non-positive")
    if ((beta < 0) | (beta > 1)).any().item():
        raise ValueError("Qwen3.5 beta must lie in [0, 1]")

    output_dtype = initial_state.dtype
    with torch.no_grad():
        state = initial_state.detach().to(torch.float32)
        key32 = normalize_gated_delta_key(key, l2norm_eps=epsilon)
        value32 = value.detach().to(torch.float32)
        decay32 = log_decay.detach().to(torch.float32)
        beta32 = beta.detach().to(torch.float32)

        for token_index in range(token_count):
            key_token = key32[:, token_index]
            value_token = value32[:, token_index]
            decay_token = decay32[:, token_index].exp().unsqueeze(-1).unsqueeze(-1)
            beta_token = beta32[:, token_index].unsqueeze(-1)

            state = state * decay_token
            remembered = (state * key_token.unsqueeze(-1)).sum(dim=-2)
            delta = (value_token - remembered) * beta_token
            state = state + key_token.unsqueeze(-1) * delta.unsqueeze(-2)

        if not torch.isfinite(state).all().item():
            raise RuntimeError("Gated DeltaNet replay produced non-finite state values")
    return state.to(output_dtype)


def replay_gated_delta_updates(
    initial_state: torch.Tensor,
    normalized_key: torch.Tensor,
    update: torch.Tensor,
    log_decay: torch.Tensor,
) -> torch.Tensor:
    """Replay stored ``(u, k, g)`` records from a checkpoint in FP32.

    The tensors use the same batch/token/head layout as
    :func:`replay_gated_delta_state`. ``normalized_key`` must already contain
    the key used by the successful kernel and ``update`` is the value-axis
    correction ``u`` from ``S_t = exp(g_t) S_{t-1} + k_t u_t^T``.

    StateLease deliberately stores the already-normalized key because
    normalizing a rounded buffer value again would define a different
    recurrence.
    """

    _validate_transition_tensor(initial_state, name="initial_state", rank=4)
    _validate_transition_tensor(
        normalized_key,
        name="normalized_key",
        rank=4,
    )
    _validate_transition_tensor(update, name="update", rank=4)
    _validate_transition_tensor(log_decay, name="log_decay", rank=3)

    batch, heads, key_rows, value_width = initial_state.shape
    token_count = normalized_key.shape[1]
    expected_key = (batch, token_count, heads, key_rows)
    expected_update = (batch, token_count, heads, value_width)
    expected_gate = (batch, token_count, heads)
    if tuple(normalized_key.shape) != expected_key:
        raise ValueError(
            f"normalized_key must have shape {expected_key}; got {tuple(normalized_key.shape)}"
        )
    if tuple(update.shape) != expected_update:
        raise ValueError(f"update must have shape {expected_update}; got {tuple(update.shape)}")
    if tuple(log_decay.shape) != expected_gate:
        raise ValueError(f"log_decay must have shape {expected_gate}; got {tuple(log_decay.shape)}")
    if token_count <= 0:
        raise ValueError("stored update sequence must contain at least one token")
    devices = {
        initial_state.device,
        normalized_key.device,
        update.device,
        log_decay.device,
    }
    if len(devices) != 1:
        raise ValueError("checkpoint and stored updates must share one device")
    if (log_decay > 0).any().item():
        raise ValueError("Qwen3.5 log_decay must be non-positive")

    output_dtype = initial_state.dtype
    with torch.no_grad():
        state = initial_state.detach().to(torch.float32)
        key32 = normalized_key.detach().to(torch.float32)
        update32 = update.detach().to(torch.float32)
        decay32 = log_decay.detach().to(torch.float32)
        for token_index in range(token_count):
            state = state * decay32[:, token_index].exp().unsqueeze(-1).unsqueeze(-1) + key32[
                :, token_index
            ].unsqueeze(-1) * update32[:, token_index].unsqueeze(-2)
        if not torch.isfinite(state).all().item():
            raise RuntimeError("stored Gated DeltaNet replay produced non-finite values")
    return state.to(output_dtype)


def propagate_frobenius_error_bound(
    previous_bound: torch.Tensor,
    log_decay: torch.Tensor,
) -> torch.Tensor:
    """Propagate a per-head Frobenius error bound through exact replay.

    For normalized key ``k`` and ``beta`` in ``[0, 1]``, the Qwen3.5 state
    transition matrix is ``exp(g) * (I - beta * k k^T)``. Its spectral norm is
    at most ``exp(g)``. Therefore a per-head state-error Frobenius bound
    contracts by ``exp(sum(g))`` over a chronological transition sequence.

    ``previous_bound`` has shape ``[batch, heads]`` and ``log_decay`` has shape
    ``[batch, tokens, heads]``.
    """

    _validate_transition_tensor(previous_bound, name="previous_bound", rank=2)
    _validate_transition_tensor(log_decay, name="log_decay", rank=3)
    if previous_bound.shape[0] != log_decay.shape[0]:
        raise ValueError("previous_bound and log_decay batch dimensions must match")
    if previous_bound.shape[1] != log_decay.shape[2]:
        raise ValueError("previous_bound and log_decay head dimensions must match")
    if log_decay.shape[1] <= 0:
        raise ValueError("log_decay must contain at least one token")
    if previous_bound.device != log_decay.device:
        raise ValueError("previous_bound and log_decay must share one device")
    if (previous_bound < 0).any().item():
        raise ValueError("previous_bound must be nonnegative")
    if (log_decay > 0).any().item():
        raise ValueError("Qwen3.5 log_decay must be non-positive")

    with torch.no_grad():
        multiplier = log_decay.detach().to(torch.float64).sum(dim=1).exp()
        propagated = previous_bound.detach().to(torch.float64) * multiplier
        if not torch.isfinite(propagated).all().item():
            raise RuntimeError("error-bound propagation produced non-finite values")
    return propagated.to(torch.float32)


def per_head_frobenius_error(
    reference: torch.Tensor,
    approximation: torch.Tensor,
) -> torch.Tensor:
    """Return per-batch, per-head FP32 Frobenius reconstruction error."""

    _validate_transition_tensor(reference, name="reference", rank=4)
    _validate_transition_tensor(approximation, name="approximation", rank=4)
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have identical shapes")
    if reference.device != approximation.device:
        raise ValueError("reference and approximation must share one device")
    with torch.no_grad():
        error = approximation.detach().to(torch.float32) - reference.detach().to(torch.float32)
        result = torch.linalg.vector_norm(error, dim=(-2, -1))
        if not torch.isfinite(result).all().item():
            raise RuntimeError("reconstruction error produced non-finite values")
    return result


def query_weighted_row_mse(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    query_energy: torch.Tensor,
) -> torch.Tensor:
    """Return the causal query-weighted row reconstruction risk.

    ``reference`` and ``approximation`` have shape
    ``[1, heads, key_rows, value_width]``. ``query_energy`` is the matching
    ``[heads, key_rows]`` FP32 causal EMA and must sum to one across key rows
    for each head. The result is the mean over heads of the EMA-weighted
    value-axis row MSE.
    """

    _validate_transition_tensor(reference, name="reference", rank=4)
    _validate_transition_tensor(approximation, name="approximation", rank=4)
    _validate_transition_tensor(query_energy, name="query_energy", rank=2)
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have identical shapes")
    if reference.shape[0] != 1:
        raise ValueError("query-weighted row MSE currently requires batch size 1")
    expected_energy_shape = (reference.shape[1], reference.shape[2])
    if tuple(query_energy.shape) != expected_energy_shape:
        raise ValueError(
            f"query_energy must have shape {expected_energy_shape}; got {tuple(query_energy.shape)}"
        )
    devices = {reference.device, approximation.device, query_energy.device}
    if len(devices) != 1:
        raise ValueError("reference, approximation, and query_energy must share one device")
    if (query_energy < 0).any().item():
        raise ValueError("query_energy must be nonnegative")
    row_sums = query_energy.detach().to(torch.float64).sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=2e-5,
        atol=2e-6,
    ):
        raise ValueError("query_energy must sum to one across rows for each head")

    with torch.no_grad():
        row_mse = (
            (approximation.detach().to(torch.float32) - reference.detach().to(torch.float32))
            .square()
            .mean(dim=-1)
            .squeeze(0)
        )
        risk = (row_mse * query_energy.detach().to(torch.float32)).sum(dim=-1).mean()
        if not torch.isfinite(risk).item() or risk.item() < 0.0:
            raise RuntimeError("query-weighted row MSE produced an invalid risk")
    return risk


@dataclass(frozen=True, slots=True)
class StateLeaseBoundaryDecision:
    """Stable choice between the frozen four- and five-token handoffs."""

    boundary: int
    cut4_risk: float
    cut5_risk: float
    tie: bool


def select_statelease_boundary(
    cut4_risk: torch.Tensor,
    cut5_risk: torch.Tensor,
) -> StateLeaseBoundaryDecision:
    """Choose the lower-risk full-buffer handoff; an exact tie chooses cut 5."""

    _validate_transition_tensor(cut4_risk, name="cut4_risk", rank=0)
    _validate_transition_tensor(cut5_risk, name="cut5_risk", rank=0)
    if cut4_risk.device != cut5_risk.device:
        raise ValueError("cut4_risk and cut5_risk must share one device")
    if cut4_risk.item() < 0.0 or cut5_risk.item() < 0.0:
        raise ValueError("handoff risks must be nonnegative")
    cut4 = float(cut4_risk.detach().to(torch.float64).item())
    cut5 = float(cut5_risk.detach().to(torch.float64).item())
    tie = cut4 == cut5
    return StateLeaseBoundaryDecision(
        boundary=4 if cut4 < cut5 else 5,
        cut4_risk=cut4,
        cut5_risk=cut5,
        tie=tie,
    )
