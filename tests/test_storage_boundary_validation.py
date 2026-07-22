from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache, Qwen3_5ForCausalLM

from recurquant.quantization import QuantizationSpec
from recurquant.storage_boundary_validation import (
    StorageRowLocation,
    advance_uniform_int4_trajectory,
    compare_directional_benefits,
    interpolate_storage_row,
    validate_qwen_storage_boundary_row,
)
from tests.test_transformers_cache import tiny_config


def _spec(bits: int) -> QuantizationSpec:
    return QuantizationSpec(bits=bits, group_size=8, flatten_last_dims=1)


def _tiny_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(2339)
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()


def _warm_cache(model: Qwen3_5ForCausalLM, prompt: torch.Tensor) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True)
    return cache


def test_directional_comparison_preserves_loss_and_benefit_signs() -> None:
    # L(alpha) = 3 + 2 alpha + 0.5 alpha^2.
    epsilon = 0.25
    def loss(alpha: float) -> float:
        return 3.0 + 2.0 * alpha + 0.5 * alpha**2

    result = compare_directional_benefits(
        epsilon=epsilon,
        loss_at_zero=loss(0),
        repeated_loss_at_zero=loss(0),
        loss_at_minus_epsilon=loss(-epsilon),
        loss_at_plus_epsilon=loss(epsilon),
        loss_at_int8_endpoint=loss(1),
        autograd_directional_derivative=2.0,
    )

    assert result.central_directional_derivative == pytest.approx(2.0)
    assert result.predicted_benefit_autograd == pytest.approx(-2.0)
    assert result.predicted_benefit_central == pytest.approx(-2.0)
    assert result.measured_endpoint_benefit == pytest.approx(-2.5)
    assert result.derivative_sign_agreement is True
    assert result.endpoint_sign_agreement is True
    assert result.endpoint_taylor_residual == pytest.approx(-0.5)


def test_directional_comparison_handles_exact_zero_without_false_sign() -> None:
    result = compare_directional_benefits(
        epsilon=0.5,
        loss_at_zero=1.0,
        repeated_loss_at_zero=1.0,
        loss_at_minus_epsilon=1.0,
        loss_at_plus_epsilon=1.0,
        loss_at_int8_endpoint=1.0,
        autograd_directional_derivative=0.0,
    )

    assert result.derivative_sign_agreement is None
    assert result.endpoint_sign_agreement is None


def test_interpolation_changes_only_the_selected_row() -> None:
    state = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    direction = torch.tensor([0.25, -0.5, 0.75, -1.0])
    location = StorageRowLocation(layer_index=7, head_index=1, row_index=2)

    moved = interpolate_storage_row(state, direction, location=location, alpha=0.5)

    difference = moved - state
    assert torch.equal(difference[0, 1, 2], direction * 0.5)
    difference[0, 1, 2] = 0
    assert torch.count_nonzero(difference).item() == 0
    assert torch.equal(state, torch.arange(24, dtype=torch.float32).reshape_as(state))


def test_real_tiny_qwen_storage_boundary_matches_central_difference() -> None:
    model = _tiny_model()
    cache = _warm_cache(model, torch.tensor([[1, 2, 3, 4]]))
    raw_before = cache.layers[0].recurrent_states[0].clone()
    location = StorageRowLocation(layer_index=0, head_index=0, row_index=0)

    result = validate_qwen_storage_boundary_row(
        model,
        cache,
        torch.tensor([[5]]),
        torch.tensor([[6]]),
        location=location,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
        epsilon=0.25,
        sign_floor=1e-9,
        forward_kwargs={"logits_to_keep": 1},
    )

    assert torch.equal(cache.layers[0].recurrent_states[0], raw_before)
    assert result.recurrent_layer_indices == (0,)
    assert torch.equal(result.direction, result.int8_row - result.int4_row)
    assert result.comparison.baseline_repeat_absolute_error < 1e-7
    assert result.comparison.autograd_directional_derivative == pytest.approx(
        result.comparison.central_directional_derivative,
        rel=3e-2,
        abs=2e-6,
    )
    assert result.comparison.predicted_benefit_autograd == pytest.approx(
        result.comparison.measured_endpoint_benefit,
        rel=0.15,
        abs=5e-6,
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_advancing_prior_token_stops_on_raw_update() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[3, 4, 5]])
    advanced = _warm_cache(model, prompt)

    advance_uniform_int4_trajectory(
        model,
        advanced,
        torch.tensor([[6]]),
        int4_spec=_spec(4),
        forward_kwargs={"logits_to_keep": 1},
    )

    reference = _warm_cache(model, prompt)
    # Quantizing before the prior token changes the update, while leaving the
    # post-token state raw instead of immediately QDQing it again.
    assert not torch.equal(
        advanced.layers[0].recurrent_states[0],
        reference.layers[0].recurrent_states[0],
    )
    assert advanced.layers[0].has_previous_state[0]


@pytest.mark.parametrize("bad_epsilon", [0.0, -0.1, 1.1, float("nan")])
def test_directional_comparison_rejects_invalid_epsilon(bad_epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        compare_directional_benefits(
            epsilon=bad_epsilon,
            loss_at_zero=1,
            repeated_loss_at_zero=1,
            loss_at_minus_epsilon=1,
            loss_at_plus_epsilon=1,
            loss_at_int8_endpoint=1,
            autograd_directional_derivative=0,
        )
