"""Diagnostic benchmark for the uniform direct-state Triton prototype.

This measures an isolated single-token Gated DeltaNet recurrence, not complete
model generation.  The comparison uses a same-schedule two-kernel FP32 Triton
reference (packed read/update versus FP32 read/update), with allocations and
validation outside the timed region.  Results therefore support only a local
kernel comparison, never an end-to-end latency or memory claim.

The packed cases are uniform direct Q4 or Q8 states. They are not the RHT mixed
Q4/Q8 StateLease checkpoint, do not consume its replay buffer or run its c4/c5
controller, and do not constitute Experiment 010 Stage-D evidence. Repeats use
fixed tensor addresses and values in one process; they do not advance a
recurrent trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch
import triton

from recurquant.quantization import QuantizationSpec, quantize_pack
from recurquant.triton_state import (
    PreparedTritonFp32Step,
    PreparedTritonGatedDeltaStep,
    pack_triton_state,
    prepare_fp32_step,
    prepare_gated_delta_step,
    triton_is_available,
    unpack_triton_state,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must not repeat")
    return parsed


def _make_inputs(
    batch_size: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        (batch_size, heads, key_dim, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    query = torch.randn(
        (batch_size, heads, key_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    key = torch.randn(
        (batch_size, heads, key_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    # Match the recurrence-ready Qwen3.5 convention: normalized q/k and the
    # model's query scale.  The Triton API intentionally does not hide this.
    query = (
        query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1.0e-6) / math.sqrt(key_dim)
    ).contiguous()
    key = (key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1.0e-6)).contiguous()
    value = torch.randn(
        (batch_size, heads, value_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    g = (
        -0.01
        - 0.2
        * torch.rand(
            (batch_size, heads),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous()
    beta = torch.rand(
        (batch_size, heads),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    return state, query, key, value, g, beta


def _reference_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decayed = state * torch.exp(g)[..., None, None]
    remembered = (decayed * key[..., :, None]).sum(dim=-2)
    update = beta[..., None] * (value - remembered)
    updated = decayed + key[..., :, None] * update[..., None, :]
    output = (updated * query[..., :, None]).sum(dim=-2)
    return updated, output


def _time_kernels(
    launch: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, object], int]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    peak_increment = max(0, torch.cuda.max_memory_allocated() - allocated_before)
    timing = {
        "latency_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
        "latency_max_ms": max(samples),
        "latency_samples_ms": samples,
    }
    return timing, peak_increment


def _packed_correctness(
    prepared: PreparedTritonGatedDeltaStep,
) -> dict[str, object]:
    decoded = unpack_triton_state(prepared.state)
    expected_state, expected_output = _reference_step(
        decoded,
        prepared.query.float(),
        prepared.key.float(),
        prepared.value.float(),
        prepared.g.float(),
        prepared.beta.float(),
    )
    expected_packed = quantize_pack(
        expected_state,
        QuantizationSpec(
            bits=prepared.state.bits,
            group_size=prepared.state.group_size,
        ),
    )
    actual = prepared.run()
    torch.cuda.synchronize()
    expected_payload = expected_packed.payload.view(torch.uint8)
    payload_matches = actual.state.payload == expected_payload
    scale_matches = actual.state.scales == expected_packed.scales
    actual_state = unpack_triton_state(actual.state)
    expected_dequantized = expected_packed.dequantize()
    state_error = actual_state - expected_dequantized
    relative_state_l2 = float(
        (
            torch.linalg.vector_norm(state_error)
            / torch.linalg.vector_norm(expected_dequantized).clamp_min(1.0e-12)
        ).item()
    )
    output_max_error = float((actual.output - expected_output).abs().max().item())
    # Parallel reductions need not be bit-identical to PyTorch's reduction
    # order. A handful of values can therefore cross a quantizer boundary even
    # when the recurrence agrees numerically. Report byte identity separately
    # and gate on the dequantized-state error.
    numerically_valid = relative_state_l2 <= 1.0e-4 and output_max_error <= 2.0e-5
    return {
        "payload_exact": bool(payload_matches.all().item()),
        "payload_match_fraction": float(payload_matches.float().mean().item()),
        "scales_exact": bool(scale_matches.all().item()),
        "scale_match_fraction": float(scale_matches.float().mean().item()),
        "state_max_absolute_error": float(state_error.abs().max().item()),
        "state_mean_squared_error": float(state_error.square().mean().item()),
        "state_relative_l2_error": relative_state_l2,
        "output_max_absolute_error": output_max_error,
        "valid": numerically_valid,
    }


def _fp32_correctness(prepared: PreparedTritonFp32Step) -> dict[str, object]:
    expected_state, expected_output = _reference_step(
        prepared.state,
        prepared.query.float(),
        prepared.key.float(),
        prepared.value.float(),
        prepared.g.float(),
        prepared.beta.float(),
    )
    actual_state, actual_output, _ = prepared.run()
    torch.cuda.synchronize()
    state_max_error = float((actual_state - expected_state).abs().max().item())
    output_max_error = float((actual_output - expected_output).abs().max().item())
    return {
        "state_max_absolute_error": state_max_error,
        "output_max_absolute_error": output_max_error,
        "valid": state_max_error <= 2.0e-5 and output_max_error <= 2.0e-5,
    }


def _rate_report(
    timing: dict[str, object],
    *,
    batch_size: int,
    state_elements: int,
) -> dict[str, object]:
    latency_ms = float(timing["latency_ms"])
    return {
        **timing,
        "sequence_tokens_per_second": batch_size * 1000.0 / latency_ms,
        "state_elements_per_second": state_elements * 1000.0 / latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[1, 8, 16])
    parser.add_argument("--bits", type=_csv_ints, default=[4, 8])
    parser.add_argument("--heads", type=_positive_int, default=16)
    parser.add_argument("--key-dim", type=_positive_int, default=128)
    parser.add_argument("--value-dim", type=_positive_int, default=128)
    parser.add_argument("--group-size", type=_positive_int, default=128)
    parser.add_argument("--warmup", type=_positive_int, default=50)
    parser.add_argument("--iterations", type=_positive_int, default=200)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=2339)
    args = parser.parse_args()

    if any(bits not in (4, 8) for bits in args.bits):
        raise ValueError("--bits supports only 4 and 8")
    if 4 in args.bits and args.group_size % 2:
        raise ValueError("INT4 benchmarking requires an even --group-size")
    if not triton_is_available():
        raise RuntimeError("a native Triton CUDA runtime is required")

    cases: list[dict[str, object]] = []
    all_correct = True
    for case_index, batch_size in enumerate(args.batch_sizes):
        state, query, key, value, g, beta = _make_inputs(
            batch_size,
            args.heads,
            args.key_dim,
            args.value_dim,
            args.seed + case_index,
        )
        fp32 = prepare_fp32_step(
            state,
            query,
            key,
            value,
            g,
            beta,
            group_size=args.group_size,
        )
        fp32_correctness = _fp32_correctness(fp32)
        fp32_timing, fp32_peak_increment = _time_kernels(
            fp32.run,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        fp32_latency = float(fp32_timing["latency_ms"])
        state_elements = state.numel()
        fp32_report = {
            **_rate_report(
                fp32_timing,
                batch_size=batch_size,
                state_elements=state_elements,
            ),
            "counted_input_state_tensor_bytes": state.numel() * state.element_size(),
            "counted_output_workspace_tensor_bytes": fp32.workspace.storage_bytes,
            "counted_live_input_plus_output_workspace_tensor_bytes": (
                state.numel() * state.element_size() + fp32.workspace.storage_bytes
            ),
            "post_warmup_peak_allocator_increment_bytes": fp32_peak_increment,
            "correctness": fp32_correctness,
        }
        all_correct &= bool(fp32_correctness["valid"])

        packed_reports: list[dict[str, object]] = []
        for bits in args.bits:
            packed = pack_triton_state(
                state,
                bits=bits,
                group_size=args.group_size,
            )
            prepared = prepare_gated_delta_step(packed, query, key, value, g, beta)
            correctness = _packed_correctness(prepared)
            timing, peak_increment = _time_kernels(
                prepared.run,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            latency = float(timing["latency_ms"])
            all_correct &= bool(correctness["valid"])
            packed_reports.append(
                {
                    "bits": bits,
                    **_rate_report(
                        timing,
                        batch_size=batch_size,
                        state_elements=state_elements,
                    ),
                    "counted_input_state_tensor_bytes": packed.storage_bytes,
                    "counted_output_workspace_tensor_bytes": prepared.workspace.storage_bytes,
                    "counted_live_input_plus_output_workspace_tensor_bytes": packed.storage_bytes
                    + prepared.workspace.storage_bytes,
                    "post_warmup_peak_allocator_increment_bytes": peak_increment,
                    "diagnostic_fp32_kernel_latency_divided_by_uniform_packed_kernel_latency": (
                        fp32_latency / latency
                    ),
                    "correctness": correctness,
                }
            )

        cases.append(
            {
                "batch_size": batch_size,
                "state_shape": list(state.shape),
                "fp32_same_schedule": fp32_report,
                "packed": packed_reports,
            }
        )

    report = {
        "schema": "recurquant.uniform-triton-state-diagnostic.v1",
        "scope": "isolated single-token two-kernel uniform-state recurrence",
        "claim_scope": (
            "diagnostic local kernel comparison only; no speed, deployment, "
            "end-to-end latency, throughput, peak-memory, or StateLease claim"
        ),
        "statelease_scope": (
            "not integrated: no RHT mixed Q4/Q8 checkpoint, replay buffer, query EMA, "
            "or c4/c5 controller; not Experiment 010 Stage-D evidence"
        ),
        "comparison": (
            "same recurrence schedule and two Triton launches for FP32 and packed paths; "
            "validation and allocation excluded; fixed inputs are repeatedly launched "
            "without advancing a recurrent trajectory"
        ),
        "stage_d_evidence": False,
        "fresh_process_starts": 1,
        "timed_recurrent_trajectory_advanced": False,
        "memory_accounting_scope": (
            "counts input-state and output-workspace tensors plus post-warmup PyTorch "
            "allocator growth; excludes recurrence inputs, allocator reservation, "
            "runtime/kernel memory, and end-to-end model peak HBM"
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "group_size": args.group_size,
        "all_correct": all_correct,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
