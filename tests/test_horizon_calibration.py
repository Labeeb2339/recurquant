from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from recurquant.horizon_calibration import (
    GDNHorizonCalibrationRecorder,
    TaskMacroHorizonAccumulator,
    row_quantization_error_energies,
    score_gdn_calibration_trace,
)
from recurquant.quantization import QuantizationSpec, quantize_dequantize


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.calls = 0

        def recurrent_kernel(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            use_qk_l2norm_in_kernel: bool,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            del key, g, beta, use_qk_l2norm_in_kernel
            self.calls += 1
            output = value.clone()
            final = initial_state + 1 if output_final_state else None
            return output, final

        self.recurrent_gated_delta_rule: Callable[..., object] = recurrent_kernel


class FakeModel(torch.nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Qwen3_5GatedDeltaNet(layer_idx=index) for index in range(layers)]
        )


def _inputs(
    *,
    batch: int = 1,
    sequence: int = 1,
    heads: int = 2,
    key_dim: int = 4,
    value_dim: int = 4,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2339)
    query = torch.randn((batch, sequence, heads, key_dim), generator=generator)
    key = torch.randn((batch, sequence, heads, key_dim), generator=generator)
    value = torch.randn((batch, sequence, heads, value_dim), generator=generator)
    log_decay = -torch.rand((batch, sequence, heads), generator=generator) * 0.2
    beta = torch.rand((batch, sequence, heads), generator=generator)
    state = torch.randn((batch, heads, key_dim, value_dim), generator=generator) * 0.3
    return query, key, value, log_decay, beta, state


def _call(
    module: Qwen3_5GatedDeltaNet,
    inputs: tuple[torch.Tensor, ...],
    *,
    normalized: bool = True,
) -> object:
    query, key, value, log_decay, beta, state = inputs
    return module.recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=log_decay,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=normalized,
    )


def _spec(bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bits=bits,
        group_size=4,
        flatten_last_dims=1,
    )


def test_recorder_is_non_mutating_and_restores_selected_hook() -> None:
    model = FakeModel()
    selected = model.layers[1]
    untouched = model.layers[0]
    original_selected = selected.recurrent_gated_delta_rule
    original_untouched = untouched.recurrent_gated_delta_rule
    inputs = _inputs()
    snapshots = tuple(tensor.clone() for tensor in inputs)

    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=[1],
        max_tokens_per_layer=3,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    with recorder:
        assert selected.recurrent_gated_delta_rule is not original_selected
        assert untouched.recurrent_gated_delta_rule is original_untouched
        output, final_state = _call(selected, inputs)

    assert selected.recurrent_gated_delta_rule is original_selected
    assert untouched.recurrent_gated_delta_rule is original_untouched
    assert selected.calls == 1
    assert output.shape == inputs[0].shape
    assert torch.equal(final_state, inputs[-1] + 1)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        assert torch.equal(value, snapshot)


def test_context_restores_hook_after_exception() -> None:
    model = FakeModel(layers=1)
    module = model.layers[0]
    original = module.recurrent_gated_delta_rule
    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=[0],
        max_tokens_per_layer=1,
    )

    with pytest.raises(RuntimeError, match="stop"), recorder:
        raise RuntimeError("stop")

    assert module.recurrent_gated_delta_rule is original


def test_recorder_captures_normalized_compact_trace_and_exact_row_energies() -> None:
    model = FakeModel(layers=1)
    module = model.layers[0]
    inputs = _inputs(value_dim=5)
    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=[0],
        max_tokens_per_layer=2,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )

    with recorder:
        _call(module, inputs)
        _call(module, inputs)
        _call(module, inputs)

    assert recorder.captured_tokens == {0: 2}
    assert recorder.dropped_calls == {0: 1}
    with pytest.raises(RuntimeError, match="trace is incomplete"):
        recorder.drain_traces()
    traces = recorder.drain_traces(require_complete=False)
    trace = traces[0]
    query, key, _, log_decay, beta, state = inputs
    expected_query = query[:, 0].to(torch.float32)
    expected_query *= torch.rsqrt(expected_query.square().sum(dim=-1, keepdim=True) + 1e-6)
    expected_key = key[:, 0].to(torch.float32)
    expected_key *= torch.rsqrt(expected_key.square().sum(dim=-1, keepdim=True) + 1e-6)
    expected_int4 = row_quantization_error_energies(state, _spec(4))
    expected_int8 = row_quantization_error_energies(state, _spec(8))

    assert trace.tokens == 2
    assert trace.dropped_calls == 1
    assert not trace.complete
    assert trace.queries.shape == (2, 1, 2, 4)
    assert trace.int4_row_error_energies.shape == (2, 1, 2, 4)
    assert trace.int8_row_error_energies.shape == (2, 1, 2, 4)
    assert trace.log_decays.shape == (2, 1, 2)
    assert trace.betas.shape == (2, 1, 2)
    assert all(
        tensor.dtype == torch.float32 and tensor.device.type == "cpu"
        for tensor in (
            trace.queries,
            trace.keys,
            trace.log_decays,
            trace.betas,
            trace.int4_row_error_energies,
            trace.int8_row_error_energies,
        )
    )
    assert torch.allclose(trace.queries[0], expected_query)
    assert torch.allclose(trace.keys[0], expected_key)
    assert torch.equal(trace.log_decays[0], log_decay[:, 0])
    assert torch.equal(trace.betas[0], beta[:, 0])
    assert torch.equal(trace.int4_row_error_energies[0], expected_int4)
    assert torch.equal(trace.int8_row_error_energies[0], expected_int8)
    assert recorder.captured_tokens == {0: 0}
    assert recorder.dropped_calls == {0: 0}

    with pytest.raises(ValueError, match="incomplete calibration trace"):
        score_gdn_calibration_trace(trace)


@pytest.mark.parametrize("bits", [4, 8])
def test_row_energy_uses_fp32_source_and_matches_direct_qdq(bits: int) -> None:
    state = torch.tensor(
        [[[[0.03125, -0.1171875, 0.203125, 0.4453125, -0.6875]]]],
        dtype=torch.bfloat16,
    )
    spec = _spec(bits)
    source = state.to(torch.float32)
    qdq = quantize_dequantize(source, spec).tensor.to(torch.float32)
    expected = (qdq - source).square().sum(dim=-1)

    actual = row_quantization_error_energies(state, spec)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_capture_validation_rejects_shape_dtype_device_and_kernel_mode() -> None:
    model = FakeModel(layers=1)
    module = model.layers[0]
    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=[0],
        max_tokens_per_layer=2,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    good = _inputs()

    with recorder:
        with pytest.raises(ValueError, match="single-token"):
            _call(module, _inputs(sequence=2))

        invalid_dtype = list(good)
        invalid_dtype[0] = invalid_dtype[0].to(torch.int64)
        with pytest.raises(TypeError, match="floating-point"):
            _call(module, tuple(invalid_dtype))

        lower_precision_state = list(good)
        lower_precision_state[-1] = lower_precision_state[-1].to(torch.bfloat16)
        with pytest.raises(TypeError, match="initial_state must use torch.float32"):
            _call(module, tuple(lower_precision_state))

        mixed_device = list(good)
        mixed_device[-1] = torch.empty(good[-1].shape, device="meta")
        with pytest.raises(ValueError, match="same device"):
            _call(module, tuple(mixed_device))

        with pytest.raises(ValueError, match="l2norm"):
            _call(module, good, normalized=False)


def test_recorder_requires_existing_bounded_layers_and_row_specs() -> None:
    model = FakeModel(layers=1)
    with pytest.raises(ValueError, match="at least one layer"):
        GDNHorizonCalibrationRecorder(
            model,
            layer_indices=[],
            max_tokens_per_layer=1,
        )
    with (
        pytest.raises(ValueError, match="selected GDN layers"),
        GDNHorizonCalibrationRecorder(
            model,
            layer_indices=[7],
            max_tokens_per_layer=1,
        ),
    ):
        pass
    with pytest.raises(ValueError, match="flatten_last_dims=1"):
        GDNHorizonCalibrationRecorder(
            model,
            layer_indices=[0],
            max_tokens_per_layer=1,
            int4_spec=QuantizationSpec(bits=4, flatten_last_dims=2),
        )


def test_task_macro_accumulator_scores_then_discards_token_traces() -> None:
    model = FakeModel(layers=1)
    module = model.layers[0]
    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=[0],
        max_tokens_per_layer=3,
        int4_spec=_spec(4),
        int8_spec=_spec(8),
    )
    with recorder:
        for _ in range(3):
            _call(module, _inputs())
    traces = recorder.drain_traces()
    expected = score_gdn_calibration_trace(traces[0], horizon=2)
    accumulator = TaskMacroHorizonAccumulator(horizon=2)

    accumulator.add_task(traces)
    accumulator.add_task(traces)
    with pytest.raises(ValueError, match="layer set changed"):
        accumulator.add_task({1: traces[0]})
    summary = accumulator.summary(0)

    assert summary.tasks == 2
    assert summary.horizon == 2
    assert torch.allclose(summary.int4_scores, expected.int4.scores.to(torch.float64))
    assert torch.allclose(summary.int8_scores, expected.int8.scores.to(torch.float64))
    assert recorder.captured_tokens == {0: 0}
