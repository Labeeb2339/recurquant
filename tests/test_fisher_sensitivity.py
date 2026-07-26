from __future__ import annotations

from types import MethodType
from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache, Qwen3_5ForCausalLM

from recurquant.fisher_sensitivity import (
    FisherStepResult,
    GDNInt4TrajectoryFisherCalibrator,
    GDNOneStepFisherCalibrator,
    RowPromotionSensitivityScores,
    TaskMacroSensitivityAccumulator,
    row_promotion_scores_from_errors,
    row_promotion_sensitivity_scores,
)
from recurquant.quantization import QuantizationSpec, RoundingMode, quantize_dequantize
from recurquant.transformers_cache import RecurrentStateQDQCache
from tests.test_transformers_cache import tiny_config


def _spec(bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bits=bits,
        group_size=8,
        flatten_last_dims=1,
    )


def _tiny_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(2339)
    return Qwen3_5ForCausalLM._from_config(
        tiny_config(),
        attn_implementation="eager",
    ).eval()


def _warm_cache(
    model: Qwen3_5ForCausalLM,
    prompt: torch.Tensor,
) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True)
    return cache


def _all_cached_tensors(cache: DynamicCache) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for layer in cache.layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        for name in ("conv_states", "recurrent_states"):
            values = getattr(layer, name, None)
            if isinstance(values, dict):
                tensors.extend(
                    value for value in values.values() if isinstance(value, torch.Tensor)
                )
    return tensors


def test_row_promotion_scores_match_signed_taylor_and_fisher_formulas() -> None:
    state = torch.tensor(
        [
            [
                [
                    [0.04, -0.13, 0.21, -0.37, 0.55, -0.72, 0.81, -0.94],
                    [-0.02, 0.11, -0.29, 0.41, -0.58, 0.69, -0.83, 0.97],
                ]
            ],
            [
                [
                    [-0.09, 0.17, -0.24, 0.38, -0.49, 0.63, -0.79, 0.91],
                    [0.01, -0.15, 0.27, -0.44, 0.57, -0.68, 0.86, -0.99],
                ]
            ],
        ],
        dtype=torch.float32,
    )
    gradient = torch.linspace(-0.4, 0.7, state.numel(), dtype=torch.float32).reshape_as(state)
    int4_spec = _spec(4)
    int8_spec = _spec(8)
    error4 = quantize_dequantize(state, int4_spec).tensor - state
    error8 = quantize_dequantize(state, int8_spec).tensor - state
    delta = error8 - error4
    expected_taylor = -(gradient * delta).sum(dim=-1).mean(dim=0)
    projected4 = (gradient * error4).sum(dim=-1)
    projected8 = (gradient * error8).sum(dim=-1)
    expected_directional_difference = (projected4.square() - projected8.square()).mean(dim=0)
    expected_diagonal_difference = (
        (gradient.square() * (error4.square() - error8.square())).sum(dim=-1).mean(dim=0)
    )
    expected_delta_magnitude = (gradient * (error4 - error8)).sum(dim=-1).square().mean(dim=0)

    taylor, directional_difference, diagonal_difference, delta_magnitude = (
        row_promotion_sensitivity_scores(
            state,
            gradient,
            int4_spec=int4_spec,
            int8_spec=int8_spec,
        )
    )

    assert taylor.shape == (1, 2)
    assert directional_difference.shape == (1, 2)
    assert diagonal_difference.shape == (1, 2)
    assert delta_magnitude.shape == (1, 2)
    assert torch.equal(taylor, expected_taylor)
    assert torch.equal(directional_difference, expected_directional_difference)
    assert torch.equal(diagonal_difference, expected_diagonal_difference)
    assert torch.equal(delta_magnitude, expected_delta_magnitude)
    assert (delta_magnitude >= 0).all()

    from_errors = row_promotion_scores_from_errors(
        gradient,
        error4,
        error8,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            from_errors,
            (
                taylor,
                directional_difference,
                diagonal_difference,
                delta_magnitude,
            ),
            strict=True,
        )
    )


def test_signed_taylor_benefit_uses_promotion_loss_reduction_sign() -> None:
    gradient = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float32)
    error4 = torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float32)
    error8 = torch.tensor([[[[0.1, 0.05]]]], dtype=torch.float32)

    taylor, _, diagonal, _ = row_promotion_scores_from_errors(
        gradient,
        error4,
        error8,
    )

    delta = error8 - error4
    assert taylor.item() == pytest.approx(-(gradient * delta).sum().item())
    assert taylor.item() > 0
    # Signed diagnostics are intentionally not clamped.
    assert diagonal.item() == pytest.approx(
        (gradient.square() * (error4.square() - error8.square())).sum().item()
    )


def _expected_scores(
    state: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return row_promotion_sensitivity_scores(
        state,
        gradient,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )


def test_full_model_step_uses_pre_token_state_and_preserves_continuity() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    measured_cache = _warm_cache(model, prompt)
    reference_cache = _warm_cache(model, prompt)
    before = measured_cache.layers[0].recurrent_states[0].clone()
    attention = model.model.layers[0].linear_attn
    original_kernel = attention.recurrent_gated_delta_rule
    captured_states: list[torch.Tensor] = []
    captured_gradients: list[torch.Tensor] = []

    def capture_kernel(*args: object, **kwargs: object):
        state = kwargs["initial_state"]
        assert isinstance(state, torch.Tensor)
        assert state.is_leaf
        assert state.requires_grad
        assert state.dtype == torch.float32
        captured_states.append(state.detach().clone())
        state.register_hook(lambda gradient: captured_gradients.append(gradient.detach().clone()))
        return original_kernel(*args, **kwargs)

    attention.recurrent_gated_delta_rule = capture_kernel
    flags_before = tuple(parameter.requires_grad for parameter in model.parameters())
    next_input = torch.tensor([[5]])
    target = torch.tensor([6])
    calibrator = GDNOneStepFisherCalibrator(
        model,
        measured_cache,
        layer_indices=[0],
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    try:
        result = calibrator.step(next_input, target)
    finally:
        attention.recurrent_gated_delta_rule = original_kernel

    with torch.no_grad():
        reference_output = model(
            next_input,
            past_key_values=reference_cache,
            use_cache=True,
        )
    expected_nll = F.cross_entropy(reference_output.logits[:, -1, :].float(), target)
    expected = _expected_scores(captured_states[0], captured_gradients[0])

    assert len(captured_states) == 1
    assert len(captured_gradients) == 1
    assert torch.equal(captured_states[0], before)
    assert result.mean_nll == pytest.approx(expected_nll.item(), abs=1e-6)
    assert result.trajectory == "fp32"
    assert torch.equal(result.layer(0).taylor_benefit, expected[0])
    assert torch.equal(result.layer(0).directional_fisher_difference, expected[1])
    assert torch.equal(result.layer(0).diagonal_fisher_difference, expected[2])
    assert torch.equal(result.layer(0).delta_direction_fisher_magnitude, expected[3])
    assert torch.allclose(
        measured_cache.layers[0].recurrent_states[0],
        reference_cache.layers[0].recurrent_states[0],
    )
    assert not torch.equal(measured_cache.layers[0].recurrent_states[0], before)
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == flags_before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert "update_recurrent_state" not in measured_cache.__dict__
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in _all_cached_tensors(measured_cache)
    )


def test_one_reverse_pass_scores_every_selected_gdn_state() -> None:
    config = tiny_config(["linear_attention", "linear_attention", "full_attention"])
    torch.manual_seed(71)
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    cache = _warm_cache(model, torch.tensor([[1, 3, 5]]))
    before = {
        layer_index: cache.layers[layer_index].recurrent_states[0].clone() for layer_index in (0, 1)
    }
    calibrator = GDNOneStepFisherCalibrator(
        model,
        cache,
        layer_indices=[1, 0],
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )

    result = calibrator.step(torch.tensor([[7]]), torch.tensor([9]))

    assert tuple(scores.layer_index for scores in result.layers) == (0, 1)
    assert all(scores.shape == (2, 8) for scores in result.layers)
    assert all(
        not torch.equal(cache.layers[layer_index].recurrent_states[0], before[layer_index])
        for layer_index in (0, 1)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_repeated_int4_trajectory_matches_qdq_cache_and_uses_aligned_delta(
    rounding: RoundingMode,
) -> None:
    model = _tiny_model()
    prompt = torch.tensor([[9, 10, 11, 12]])
    measured_cache = _warm_cache(model, prompt)
    raw_boundary = measured_cache.layers[0].recurrent_states[0].clone()
    int4_spec = QuantizationSpec(
        bits=4,
        group_size=8,
        flatten_last_dims=1,
        rounding=rounding,
    )
    int8_spec = QuantizationSpec(
        bits=8,
        group_size=8,
        flatten_last_dims=1,
        rounding=rounding,
    )
    reference_cache = RecurrentStateQDQCache(
        model.config,
        spec=int4_spec,
        record_evidence=False,
    )
    with torch.no_grad():
        model(prompt, past_key_values=reference_cache, use_cache=True)

    calibrator = GDNInt4TrajectoryFisherCalibrator(
        model,
        measured_cache,
        int4_spec=int4_spec,
        int8_spec=int8_spec,
    )
    assert torch.equal(
        measured_cache.layers[0].recurrent_states[0],
        reference_cache.layers[0].recurrent_states[0],
    )

    attention = model.model.layers[0].linear_attn
    original_kernel = attention.recurrent_gated_delta_rule
    captured_states: list[torch.Tensor] = []
    captured_gradients: list[torch.Tensor] = []

    def capture_kernel(*args: object, **kwargs: object):
        state = kwargs["initial_state"]
        assert isinstance(state, torch.Tensor)
        assert state.is_leaf and state.requires_grad
        captured_states.append(state.detach().clone())
        state.register_hook(lambda gradient: captured_gradients.append(gradient.detach().clone()))
        return original_kernel(*args, **kwargs)

    attention.recurrent_gated_delta_rule = capture_kernel
    token = torch.tensor([[13]])
    target = torch.tensor([14])
    try:
        result = calibrator.step(token, target)
    finally:
        attention.recurrent_gated_delta_rule = original_kernel
    with torch.no_grad():
        reference_output = model(token, past_key_values=reference_cache, use_cache=True)

    q4 = quantize_dequantize(raw_boundary, int4_spec).tensor.to(torch.float32)
    q8 = quantize_dequantize(raw_boundary, int8_spec).tensor.to(torch.float32)
    expected = row_promotion_scores_from_errors(
        captured_gradients[0],
        q4 - raw_boundary,
        q8 - raw_boundary,
    )

    assert result.trajectory == "int4"
    assert torch.equal(captured_states[0], q4)
    assert result.mean_nll == pytest.approx(
        F.cross_entropy(reference_output.logits[:, -1, :].float(), target).item(),
        abs=1e-6,
    )
    assert torch.equal(result.layer(0).taylor_benefit, expected[0])
    assert torch.equal(result.layer(0).directional_fisher_difference, expected[1])
    assert torch.equal(result.layer(0).diagonal_fisher_difference, expected[2])
    assert torch.equal(result.layer(0).delta_direction_fisher_magnitude, expected[3])
    assert torch.equal(
        measured_cache.layers[0].recurrent_states[0],
        reference_cache.layers[0].recurrent_states[0],
    )

    second_token = torch.tensor([[15]])
    second_target = torch.tensor([16])
    second = calibrator.step(second_token, second_target)
    with torch.no_grad():
        second_reference = model(
            second_token,
            past_key_values=reference_cache,
            use_cache=True,
        )
    assert second.mean_nll == pytest.approx(
        F.cross_entropy(second_reference.logits[:, -1, :].float(), second_target).item(),
        abs=1e-6,
    )
    assert torch.equal(
        measured_cache.layers[0].recurrent_states[0],
        reference_cache.layers[0].recurrent_states[0],
    )
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in _all_cached_tensors(measured_cache)
    )


def test_partial_multilayer_failure_rolls_back_int4_trajectory_atomically() -> None:
    config = tiny_config(["linear_attention", "linear_attention", "full_attention"])
    torch.manual_seed(91)
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager").eval()
    prompt = torch.tensor([[3, 6, 9]])
    measured_cache = _warm_cache(model, prompt)
    reference_cache = RecurrentStateQDQCache(
        config,
        spec=_spec(4),
        record_evidence=False,
    )
    with torch.no_grad():
        model(prompt, past_key_values=reference_cache, use_cache=True)
    calibrator = GDNInt4TrajectoryFisherCalibrator(
        model,
        measured_cache,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    before = [tensor.clone() for tensor in _all_cached_tensors(measured_cache)]
    second_attention = model.model.layers[1].linear_attn
    original_kernel = second_attention.recurrent_gated_delta_rule

    def fail_after_first_layer(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("failure after first GDN update")

    second_attention.recurrent_gated_delta_rule = fail_after_first_layer
    try:
        with pytest.raises(RuntimeError, match="after first GDN update"):
            calibrator.step(torch.tensor([[12]]), torch.tensor([15]))
    finally:
        second_attention.recurrent_gated_delta_rule = original_kernel

    after = _all_cached_tensors(measured_cache)
    assert len(after) == len(before)
    assert all(
        torch.equal(current, original) for current, original in zip(after, before, strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert "update_recurrent_state" not in measured_cache.__dict__

    token = torch.tensor([[12]])
    target = torch.tensor([15])
    measured = calibrator.step(token, target)
    with torch.no_grad():
        reference = model(token, past_key_values=reference_cache, use_cache=True)
    assert measured.mean_nll == pytest.approx(
        F.cross_entropy(reference.logits[:, -1, :].float(), target).item(),
        abs=1e-6,
    )
    for layer_index in (0, 1):
        assert torch.equal(
            measured_cache.layers[layer_index].recurrent_states[0],
            reference_cache.layers[layer_index].recurrent_states[0],
        )

    second_input = torch.tensor([[7]])
    calibrator.step(second_input, torch.tensor([8]))
    with torch.no_grad():
        model(second_input, past_key_values=reference_cache, use_cache=True)
    assert torch.allclose(
        measured_cache.layers[0].recurrent_states[0],
        reference_cache.layers[0].recurrent_states[0],
    )
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in _all_cached_tensors(measured_cache)
    )


def test_step_restores_cache_method_parameters_and_graph_state_after_failure() -> None:
    model = _tiny_model()
    cache = _warm_cache(model, torch.tensor([[1, 2, 3]]))
    state_before = cache.layers[0].recurrent_states[0].clone()
    parameters = tuple(model.parameters())
    parameters[0].requires_grad_(False)
    flags_before = tuple(parameter.requires_grad for parameter in parameters)
    original_forward = model.forward

    def broken_forward(_model: torch.nn.Module, **kwargs: object):
        del kwargs
        raise RuntimeError("synthetic forward failure")

    model.forward = MethodType(broken_forward, model)
    calibrator = GDNOneStepFisherCalibrator(
        model,
        cache,
        layer_indices=[0],
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            calibrator.step(torch.tensor([[4]]), torch.tensor([5]))
    finally:
        model.forward = original_forward

    assert tuple(parameter.requires_grad for parameter in parameters) == flags_before
    assert all(parameter.grad is None for parameter in parameters)
    assert "update_recurrent_state" not in cache.__dict__
    assert torch.equal(cache.layers[0].recurrent_states[0], state_before)
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None for tensor in _all_cached_tensors(cache)
    )


def test_step_rejects_preexisting_parameter_gradients_without_mutating_cache() -> None:
    model = _tiny_model()
    cache = _warm_cache(model, torch.tensor([[2, 4, 6]]))
    state_before = cache.layers[0].recurrent_states[0].clone()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    calibrator = GDNOneStepFisherCalibrator(
        model,
        cache,
        layer_indices=[0],
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )

    with pytest.raises(ValueError, match="pre-existing gradients"):
        calibrator.step(torch.tensor([[8]]), torch.tensor([10]))

    assert torch.equal(cache.layers[0].recurrent_states[0], state_before)
    assert "update_recurrent_state" not in cache.__dict__


def _step(
    value: float,
    *,
    nll: float = 1.0,
    trajectory: Literal["fp32", "int4"] = "fp32",
) -> FisherStepResult:
    scores = torch.full((2, 3), value, dtype=torch.float32)
    return FisherStepResult(
        mean_nll=nll,
        batch_size=1,
        trajectory=trajectory,
        layers=(
            RowPromotionSensitivityScores(
                layer_index=0,
                taylor_benefit=scores,
                directional_fisher_difference=scores * 2,
                diagonal_fisher_difference=scores * 3,
                delta_direction_fisher_magnitude=scores * 4,
            ),
        ),
    )


def test_task_macro_accumulator_weights_tasks_not_token_count() -> None:
    accumulator = TaskMacroSensitivityAccumulator()
    accumulator.add_task([_step(0, nll=2)])
    accumulator.add_task([_step(6, nll=4), _step(9, nll=8), _step(12, nll=12)])

    summary = accumulator.summary(0)

    # Task means are 0 and 9, so equal-task macro is 4.5. A token mean would be 6.75.
    assert summary.tasks == 2
    assert summary.steps == 4
    assert summary.mean_nll == pytest.approx(5.0)
    assert summary.trajectory == "fp32"
    assert torch.equal(summary.taylor_benefit, torch.full((2, 3), 4.5, dtype=torch.float64))
    assert torch.equal(
        summary.directional_fisher_difference,
        torch.full((2, 3), 9.0, dtype=torch.float64),
    )
    assert torch.equal(
        summary.diagonal_fisher_difference,
        torch.full((2, 3), 13.5, dtype=torch.float64),
    )
    assert torch.equal(
        summary.delta_direction_fisher_magnitude,
        torch.full((2, 3), 18.0, dtype=torch.float64),
    )


def test_validation_rejects_non_row_specs_shapes_and_unready_cache() -> None:
    state = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    gradient = torch.zeros_like(state)
    with pytest.raises(ValueError, match="group_size"):
        row_promotion_sensitivity_scores(
            state,
            gradient,
            int4_spec=QuantizationSpec(bits=4, group_size=4, flatten_last_dims=1),
            int8_spec=QuantizationSpec(bits=8, group_size=4, flatten_last_dims=1),
        )
    with pytest.raises(ValueError, match="differ only in bit width"):
        row_promotion_sensitivity_scores(
            state,
            gradient,
            int4_spec=_spec(4),
            int8_spec=QuantizationSpec(bits=8, group_size=8, scale_bits=32, flatten_last_dims=1),
        )
    with pytest.raises(ValueError, match="same shape"):
        row_promotion_sensitivity_scores(
            state,
            gradient[..., :7],
            int4_spec=_spec(4),
            int8_spec=_spec(8),
        )
    nonfinite = gradient.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        row_promotion_sensitivity_scores(
            state,
            nonfinite,
            int4_spec=_spec(4),
            int8_spec=_spec(8),
        )

    model = _tiny_model()
    empty_cache = DynamicCache(config=model.config)
    calibrator = GDNOneStepFisherCalibrator(
        model,
        empty_cache,
        layer_indices=[0],
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    with pytest.raises(ValueError, match="run a no-grad prefill"):
        calibrator.step(torch.tensor([[1]]), torch.tensor([2]))


def test_task_macro_rejects_shape_changes_and_nonfinite_values() -> None:
    accumulator = TaskMacroSensitivityAccumulator()
    accumulator.add_task([_step(1)])
    wrong_shape = FisherStepResult(
        mean_nll=1.0,
        batch_size=1,
        trajectory="fp32",
        layers=(
            RowPromotionSensitivityScores(
                layer_index=0,
                taylor_benefit=torch.ones(2, 2),
                directional_fisher_difference=torch.ones(2, 2),
                diagonal_fisher_difference=torch.ones(2, 2),
                delta_direction_fisher_magnitude=torch.ones(2, 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match="layer set or score shapes changed"):
        accumulator.add_task([wrong_shape])
    nonfinite = _step(1, nll=float("nan"))
    with pytest.raises(ValueError, match="mean_nll"):
        accumulator.add_task([nonfinite])
