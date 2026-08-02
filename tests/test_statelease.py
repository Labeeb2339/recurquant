from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    l2norm as qwen35_l2norm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

from recurquant.statelease import (
    normalize_gated_delta_key,
    per_head_frobenius_error,
    propagate_frobenius_error_bound,
    query_weighted_row_mse,
    replay_gated_delta_state,
    replay_gated_delta_updates,
    select_statelease_boundary,
)


def _transition(
    *,
    batch: int = 1,
    tokens: int = 3,
    heads: int = 2,
    key_rows: int = 4,
    value_width: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2339)
    key = torch.randn(
        batch,
        tokens,
        heads,
        key_rows,
        generator=generator,
    )
    value = torch.randn(
        batch,
        tokens,
        heads,
        value_width,
        generator=generator,
    )
    log_decay = -torch.rand(batch, tokens, heads, generator=generator)
    beta = torch.rand(batch, tokens, heads, generator=generator)
    return key, value, log_decay, beta


def test_replay_matches_direct_qwen_recurrence() -> None:
    key, value, log_decay, beta = _transition()
    generator = torch.Generator().manual_seed(7)
    initial = torch.randn(1, 2, 4, 5, generator=generator)

    actual = replay_gated_delta_state(
        initial,
        key,
        value,
        log_decay,
        beta,
    )

    expected = initial.to(torch.float32)
    normalized_key = key.to(torch.float32)
    normalized_key = normalized_key * torch.rsqrt(
        normalized_key.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    for token_index in range(key.shape[1]):
        key_token = normalized_key[:, token_index]
        expected = expected * log_decay[:, token_index].exp()[:, :, None, None]
        remembered = (expected * key_token.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, token_index] - remembered) * beta[:, token_index].unsqueeze(-1)
        expected = expected + key_token.unsqueeze(-1) * delta.unsqueeze(-2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_key_normalization_and_replay_match_pinned_qwen_dtype_order(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(1, 2, 2, 8, generator=generator).to(dtype)
    key = torch.randn(1, 2, 2, 8, generator=generator).to(dtype)
    value = torch.randn(1, 2, 2, 8, generator=generator).to(dtype)
    log_decay = (-0.2 * torch.rand(1, 2, 2, generator=generator)).to(dtype)
    beta = torch.rand(1, 2, 2, generator=generator).to(dtype)
    initial = torch.randn(1, 2, 8, 8, generator=generator)

    expected_key = qwen35_l2norm(key, dim=-1, eps=1e-6).to(torch.float32)
    actual_key = normalize_gated_delta_key(key)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)

    _, expected_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        log_decay,
        beta,
        initial,
        True,
        True,
    )
    actual_state = replay_gated_delta_state(
        initial,
        key,
        value,
        log_decay,
        beta,
    )
    assert expected_state is not None
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-7)


def test_replay_chunk_equivalence() -> None:
    key, value, log_decay, beta = _transition(tokens=4)
    initial = torch.randn(1, 2, 4, 5, generator=torch.Generator().manual_seed(11))

    together = replay_gated_delta_state(
        initial,
        key,
        value,
        log_decay,
        beta,
    )
    first = replay_gated_delta_state(
        initial,
        key[:, :2],
        value[:, :2],
        log_decay[:, :2],
        beta[:, :2],
    )
    split = replay_gated_delta_state(
        first,
        key[:, 2:],
        value[:, 2:],
        log_decay[:, 2:],
        beta[:, 2:],
    )
    torch.testing.assert_close(together, split, rtol=1e-6, atol=2e-7)


def test_stored_update_replay_matches_raw_transition_replay() -> None:
    key, value, log_decay, beta = _transition(tokens=4)
    initial = torch.randn(1, 2, 4, 5, generator=torch.Generator().manual_seed(17))
    normalized_key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    state = initial.to(torch.float32)
    updates: list[torch.Tensor] = []
    for token_index in range(key.shape[1]):
        key_token = normalized_key[:, token_index]
        state = state * log_decay[:, token_index].exp()[:, :, None, None]
        remembered = (state * key_token.unsqueeze(-1)).sum(dim=-2)
        update = (value[:, token_index] - remembered) * beta[:, token_index].unsqueeze(-1)
        updates.append(update)
        state = state + key_token.unsqueeze(-1) * update.unsqueeze(-2)

    replayed = replay_gated_delta_updates(
        initial,
        normalized_key,
        torch.stack(updates, dim=1),
        log_decay,
    )
    raw = replay_gated_delta_state(
        initial,
        key,
        value,
        log_decay,
        beta,
    )
    torch.testing.assert_close(replayed, raw, rtol=0, atol=0)


def test_frobenius_bound_covers_actual_error() -> None:
    key, value, log_decay, beta = _transition(tokens=5)
    exact = torch.randn(1, 2, 4, 5, generator=torch.Generator().manual_seed(19))
    perturbation = 0.03 * torch.randn(
        1,
        2,
        4,
        5,
        generator=torch.Generator().manual_seed(23),
    )
    approximate = exact + perturbation

    exact_result = replay_gated_delta_state(
        exact,
        key,
        value,
        log_decay,
        beta,
    )
    approximate_result = replay_gated_delta_state(
        approximate,
        key,
        value,
        log_decay,
        beta,
    )
    initial_bound = per_head_frobenius_error(exact, approximate)
    propagated = propagate_frobenius_error_bound(initial_bound, log_decay)
    observed = per_head_frobenius_error(exact_result, approximate_result)

    assert torch.all(observed <= propagated + 2e-6)


def test_query_weighted_row_mse_uses_causal_row_weights() -> None:
    reference = torch.zeros(1, 2, 2, 3)
    approximation = reference.clone()
    approximation[0, 0, 0] = 2.0
    approximation[0, 0, 1] = 1.0
    query_energy = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    risk = query_weighted_row_mse(reference, approximation, query_energy)
    assert risk.item() == pytest.approx((0.75 * 4.0 + 0.25 * 1.0) / 2.0)


@pytest.mark.parametrize(
    ("cut4", "cut5", "boundary", "tie"),
    [
        (0.1, 0.2, 4, False),
        (0.2, 0.1, 5, False),
        (0.1, 0.1, 5, True),
    ],
)
def test_statelease_boundary_is_stable(
    cut4: float,
    cut5: float,
    boundary: int,
    tie: bool,
) -> None:
    decision = select_statelease_boundary(
        torch.tensor(cut4),
        torch.tensor(cut5),
    )
    assert decision.boundary == boundary
    assert decision.tie is tie


def test_statelease_rejects_invalid_qwen_gates() -> None:
    key, value, log_decay, beta = _transition(tokens=1)
    initial = torch.zeros(1, 2, 4, 5)
    with pytest.raises(ValueError, match="non-positive"):
        replay_gated_delta_state(
            initial,
            key,
            value,
            torch.ones_like(log_decay),
            beta,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replay_gated_delta_state(
            initial,
            key,
            value,
            log_decay,
            torch.full_like(beta, 1.1),
        )
