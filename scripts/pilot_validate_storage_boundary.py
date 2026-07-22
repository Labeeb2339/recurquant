#!/usr/bin/env python3
"""Validate one-row Taylor directions at real Qwen3.5 storage boundaries.

This bounded diagnostic reads only the pinned MBPP calibration partition.  It
uses four geometry-stratified rows fixed in source before any benefit is
measured and the fixed central-difference epsilon grid {1/4, 1/8, 1/16, 1/32}.
It is an implementation check, not a policy search or quality benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.evidence import canonical_json_bytes
from recurquant.public_data import (
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
)
from recurquant.quantization import QuantizationSpec, quantize_pack
from recurquant.storage_boundary_validation import (
    StorageBoundaryRowValidation,
    StorageRowLocation,
    advance_uniform_int4_trajectory,
    validate_qwen_storage_boundary_row,
)

SEED: Final = 2339
MODEL_ID: Final = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION: Final = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
ARTIFACT_KIND: Final = "recurquant_storage_boundary_taylor_diagnostic"
TASK_PREFIX_LIMIT: Final = 1
TASK_INDEX: Final = 0
TRANSITION_INDEX: Final = 0
EPSILONS: Final = (1 / 4, 1 / 8, 1 / 16, 1 / 32)
BASELINE_REPEAT_ABSOLUTE_TOLERANCE: Final = 1e-7
DERIVATIVE_INFORMATIVE_FLOOR: Final = 1e-8
NEAR_ZERO_ABSOLUTE_TOLERANCE: Final = 2e-7
MINIMUM_INFORMATIVE_ROWS: Final = 3
MINIMUM_SIGN_AGREEMENT: Final = 0.95
MAXIMUM_MEDIAN_RELATIVE_ERROR: Final = 0.10
MINIMUM_CONVERGED_ROW_FRACTION: Final = 0.75

# (label, recurrent-layer ordinal, head index, key-row index).  These four
# coordinates span early/late layers and low/high head/row indices.  They are
# intentionally unrelated to any selector score or measured loss benefit.
ROW_STRATA: Final = (
    ("early_low", 0, 0, 0),
    ("early_mid", 5, 5, 42),
    ("late_mid", 11, 10, 85),
    ("late_high", 17, 15, 127),
)

CACHE_FINGERPRINT_CACHE_METADATA: Final = (
    "offloading",
    "layer_class_to_replicate",
)
CACHE_FINGERPRINT_LAYER_METADATA: Final = (
    "number_of_states",
    "is_conv_states_initialized",
    "is_recurrent_states_initialized",
    "has_previous_state",
    "conv_kernel_size",
    "is_initialized",
    "cumulative_length",
    "sliding_window",
    "max_cache_len",
    "device",
    "dtype",
    "record_past",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pinned calibration-only one-row storage-boundary derivative diagnostic; "
            "not a quality benchmark."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--phase",
        default="calibration",
        help="Must remain 'calibration'; development and confirmation are refused.",
    )
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "status": status,
        "worktree_clean": not status,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
    }


def _package_versions() -> dict[str, str]:
    names = ("datasets", "numpy", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


def _implementation_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    return (
        Path(__file__).resolve(),
        root / "src" / "recurquant" / "storage_boundary_validation.py",
        root / "src" / "recurquant" / "quantization.py",
        root / "src" / "recurquant" / "evidence.py",
        root / "src" / "recurquant" / "public_data.py",
    )


def _source_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    hashes: dict[str, str] = {}
    for path in paths:
        relative = path.resolve().relative_to(root).as_posix()
        hashes[relative] = _sha256_bytes(path.read_bytes())
    return hashes


def _tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return _sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _directional_dot_float64(
    gradient_row: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """Recompute the recorded FP32 row dot on CPU with FP64 accumulation."""

    if gradient_row.shape != direction.shape:
        raise ValueError("gradient row and direction must have the same shape")
    if gradient_row.dtype != torch.float32 or direction.dtype != torch.float32:
        raise TypeError("gradient row and direction must use float32 before FP64 accumulation")
    gradient64 = gradient_row.detach().to(device="cpu", dtype=torch.float64)
    direction64 = direction.detach().to(device="cpu", dtype=torch.float64)
    value = float((gradient64 * direction64).sum(dtype=torch.float64).item())
    if not math.isfinite(value):
        raise RuntimeError("directional dot product became non-finite")
    return value


def _tensor_record(tensor: torch.Tensor, *, include_values: bool) -> dict[str, object]:
    value = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError("artifact tensors must be non-empty and finite")
    values = value.tolist()
    canonical = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": values,
        }
    )
    record: dict[str, object] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "canonical_values_sha256": _sha256_bytes(canonical),
        "raw_bytes_sha256": _tensor_bytes_sha256(tensor),
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
    }
    if include_values:
        record["values_float64"] = values
    return record


def _cache_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {
            str(key): _cache_metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_cache_metadata_value(item) for item in value]
    raise TypeError(f"unsupported cache fingerprint metadata type: {type(value).__name__}")


def _cache_fingerprint(cache: DynamicCache) -> str:
    records: list[dict[str, object]] = []
    records.append(
        {
            "scope": "cache",
            "class": f"{cache.__class__.__module__}.{cache.__class__.__qualname__}",
            "layer_count": len(cache.layers),
            "metadata": {
                name: {
                    "present": hasattr(cache, name),
                    "value": (
                        _cache_metadata_value(getattr(cache, name))
                        if hasattr(cache, name)
                        else None
                    ),
                }
                for name in CACHE_FINGERPRINT_CACHE_METADATA
            },
        }
    )
    for layer_index, layer in enumerate(cache.layers):
        records.append(
            {
                "scope": "layer",
                "layer_index": layer_index,
                "class": f"{layer.__class__.__module__}.{layer.__class__.__qualname__}",
                "metadata": {
                    name: {
                        "present": hasattr(layer, name),
                        "value": (
                            _cache_metadata_value(getattr(layer, name))
                            if hasattr(layer, name)
                            else None
                        ),
                    }
                    for name in CACHE_FINGERPRINT_LAYER_METADATA
                },
            }
        )
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                records.append(
                    {
                        "scope": "tensor",
                        "layer_index": layer_index,
                        "attribute": name,
                        "state_index": None,
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "raw_bytes_sha256": _tensor_bytes_sha256(value),
                    }
                )
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if not isinstance(states, dict):
                continue
            for state_index, value in sorted(states.items()):
                if isinstance(value, torch.Tensor):
                    records.append(
                        {
                            "scope": "tensor",
                            "layer_index": layer_index,
                            "attribute": name,
                            "state_index": int(state_index),
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "raw_bytes_sha256": _tensor_bytes_sha256(value),
                        }
                    )
    return _sha256_bytes(canonical_json_bytes(records))


def _resolve_strata(
    recurrent_layers: tuple[int, ...],
    *,
    heads: int,
    rows: int,
) -> tuple[tuple[str, StorageRowLocation], ...]:
    resolved: list[tuple[str, StorageRowLocation]] = []
    for label, layer_ordinal, head_index, row_index in ROW_STRATA:
        if layer_ordinal >= len(recurrent_layers):
            raise ValueError(f"row stratum {label} exceeds recurrent-layer geometry")
        if head_index >= heads or row_index >= rows:
            raise ValueError(f"row stratum {label} exceeds recurrent-state geometry")
        resolved.append(
            (
                label,
                StorageRowLocation(
                    layer_index=recurrent_layers[layer_ordinal],
                    head_index=head_index,
                    row_index=row_index,
                ),
            )
        )
    return tuple(resolved)


def _physical_endpoint_record(
    raw_row: torch.Tensor,
    endpoint_row: torch.Tensor,
    spec: QuantizationSpec,
) -> dict[str, object]:
    shaped = raw_row.reshape(1, 1, 1, -1)
    packed = quantize_pack(shaped, spec)
    dequantized = packed.dequantize().reshape(-1).cpu()
    if not torch.equal(dequantized, endpoint_row):
        raise RuntimeError(f"INT{spec.bits} row endpoint does not match physical packing")
    return {
        "bits": spec.bits,
        "payload_bytes": packed.payload.numel() * packed.payload.element_size(),
        "scale_bytes": packed.scales.numel() * packed.scales.element_size(),
        "resident_bytes": packed.storage_bytes,
        "payload": _tensor_record(packed.payload, include_values=False),
        "scales": _tensor_record(packed.scales, include_values=True),
    }


def _row_tensors_record(result: StorageBoundaryRowValidation) -> dict[str, object]:
    if not torch.equal(result.direction, result.int8_row - result.int4_row):
        raise RuntimeError("recorded row direction does not equal INT8 minus INT4")
    return {
        "raw": _tensor_record(result.raw_row, include_values=True),
        "int4": _tensor_record(result.int4_row, include_values=True),
        "int8": _tensor_record(result.int8_row, include_values=True),
        "direction_int8_minus_int4": _tensor_record(result.direction, include_values=True),
        "loss_gradient_at_int4": _tensor_record(result.gradient_row, include_values=True),
    }


def _assert_same_row_result(
    reference: StorageBoundaryRowValidation,
    candidate: StorageBoundaryRowValidation,
) -> None:
    fields = ("raw_row", "int4_row", "int8_row", "direction", "gradient_row")
    if any(not torch.equal(getattr(reference, name), getattr(candidate, name)) for name in fields):
        raise RuntimeError("row tensors changed across the fixed epsilon grid")
    scalar_fields = (
        "loss_at_zero",
        "repeated_loss_at_zero",
        "loss_at_int8_endpoint",
        "autograd_directional_derivative",
        "predicted_benefit_autograd",
        "measured_endpoint_benefit",
    )
    if any(
        getattr(reference.comparison, name) != getattr(candidate.comparison, name)
        for name in scalar_fields
    ):
        raise RuntimeError("shared endpoint or autograd results changed across epsilons")


def _evaluate_derivative_gate(row_records: list[dict[str, object]]) -> dict[str, object]:
    """Apply the source-frozen FP32 derivative gate to detached row records."""

    if len(row_records) != len(ROW_STRATA):
        raise ValueError("derivative gate requires every frozen row stratum")
    expected_labels = tuple(label for label, *_ in ROW_STRATA)
    observed_labels = tuple(str(record.get("stratum")) for record in row_records)
    if observed_labels != expected_labels:
        raise ValueError("derivative gate requires frozen row strata in source order")
    informative_rows = 0
    near_zero_rows = 0
    sign_checks: list[bool] = []
    relative_errors: list[float] = []
    converged_rows: list[bool] = []
    baseline_repeat_errors: list[float] = []
    near_zero_checks: list[bool] = []
    failures: list[str] = []

    for row_record in row_records:
        label = str(row_record["stratum"])
        raw_results = row_record["epsilon_results"]
        if not isinstance(raw_results, list) or len(raw_results) != len(EPSILONS):
            raise ValueError(f"row {label} lacks the frozen epsilon grid")
        results = [dict(value) for value in raw_results]
        observed_epsilons = tuple(float(value["epsilon"]) for value in results)
        if any(not math.isfinite(value) for value in observed_epsilons):
            raise ValueError(f"row {label} epsilon grid must be finite")
        if observed_epsilons != EPSILONS:
            raise ValueError(f"row {label} epsilon grid changed")

        autograd_values = [
            float(value["autograd_directional_derivative"]) for value in results
        ]
        central_values = [
            float(value["central_directional_derivative"]) for value in results
        ]
        current_baseline_errors = [
            float(value["baseline_repeat_absolute_error"]) for value in results
        ]
        if any(
            not math.isfinite(value)
            for value in (*autograd_values, *central_values, *current_baseline_errors)
        ):
            raise ValueError(f"row {label} derivative-gate values must be finite")
        if any(value < 0 for value in current_baseline_errors):
            raise ValueError(f"row {label} baseline repeat errors must be non-negative")
        unique_autograd_values = set(autograd_values)
        if len(unique_autograd_values) != 1:
            raise ValueError(f"row {label} autograd derivative changed across epsilons")
        autograd = unique_autograd_values.pop()
        absolute_errors = [abs(central - autograd) for central in central_values]
        baseline_repeat_errors.extend(current_baseline_errors)

        if abs(autograd) <= DERIVATIVE_INFORMATIVE_FLOOR:
            near_zero_rows += 1
            checks = [error <= NEAR_ZERO_ABSOLUTE_TOLERANCE for error in absolute_errors]
            near_zero_checks.extend(checks)
            if not all(checks):
                failures.append(
                    f"{label}: near-zero absolute derivative error exceeded tolerance"
                )
            continue

        informative_rows += 1
        row_signs = [
            (central > 0) == (autograd > 0) and central != 0
            for central in central_values
        ]
        sign_checks.extend(row_signs)
        relative_errors.extend(error / abs(autograd) for error in absolute_errors)
        # The final three grid points are 1/8, 1/16, and 1/32. Numerical noise
        # need not make them strictly monotone, but the smallest step must not
        # be worse than the first of those three.
        converged_rows.append(absolute_errors[-1] <= absolute_errors[1])

    maximum_baseline_repeat_error = max(baseline_repeat_errors, default=math.inf)
    sign_agreement = sum(sign_checks) / len(sign_checks) if sign_checks else None
    median_relative_error = (
        statistics.median(relative_errors) if relative_errors else None
    )
    converged_fraction = (
        sum(converged_rows) / len(converged_rows) if converged_rows else None
    )
    if informative_rows < MINIMUM_INFORMATIVE_ROWS:
        failures.append(
            f"only {informative_rows} informative rows; require {MINIMUM_INFORMATIVE_ROWS}"
        )
    if maximum_baseline_repeat_error > BASELINE_REPEAT_ABSOLUTE_TOLERANCE:
        failures.append("repeated alpha=0 baseline exceeded absolute tolerance")
    if sign_agreement is None or sign_agreement < MINIMUM_SIGN_AGREEMENT:
        failures.append("central-difference sign agreement fell below threshold")
    if (
        median_relative_error is None
        or median_relative_error > MAXIMUM_MEDIAN_RELATIVE_ERROR
    ):
        failures.append("median central-difference relative error exceeded threshold")
    if converged_fraction is None or converged_fraction < MINIMUM_CONVERGED_ROW_FRACTION:
        failures.append("too few informative rows improved over the final epsilon range")

    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "model_dtype": "torch.float32",
            "baseline_repeat_absolute_tolerance": BASELINE_REPEAT_ABSOLUTE_TOLERANCE,
            "derivative_informative_floor": DERIVATIVE_INFORMATIVE_FLOOR,
            "near_zero_absolute_tolerance": NEAR_ZERO_ABSOLUTE_TOLERANCE,
            "minimum_informative_rows": MINIMUM_INFORMATIVE_ROWS,
            "minimum_sign_agreement": MINIMUM_SIGN_AGREEMENT,
            "maximum_median_relative_error": MAXIMUM_MEDIAN_RELATIVE_ERROR,
            "minimum_converged_row_fraction": MINIMUM_CONVERGED_ROW_FRACTION,
        },
        "observed": {
            "rows": len(row_records),
            "informative_rows": informative_rows,
            "near_zero_rows": near_zero_rows,
            "sign_checks": len(sign_checks),
            "sign_agreement": sign_agreement,
            "median_relative_error": median_relative_error,
            "converged_row_fraction": converged_fraction,
            "maximum_baseline_repeat_absolute_error": maximum_baseline_repeat_error,
            "near_zero_checks": len(near_zero_checks),
            "near_zero_checks_passed": sum(near_zero_checks),
        },
    }


def main() -> int:
    args = parse_args()
    if args.phase != "calibration":
        raise ValueError(
            "This implementation diagnostic refuses development and confirmation data; "
            "--phase must be calibration."
        )

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = _select_device(args.device)
    # The frozen gate uses FP32 weights and FP32 NLL so central differences do
    # not collapse merely because a BF16 matmul rounds away a small alpha move.
    dtype = torch.float32
    implementation_paths = _implementation_paths()
    source_hashes_start = _source_hashes(implementation_paths)
    repository_start = _git_state()

    rows = load_mbpp_rows("calibration", limit=TASK_PREFIX_LIMIT)
    if len(rows) != TASK_PREFIX_LIMIT:
        raise RuntimeError("pinned calibration loader returned an unexpected task count")
    row = rows[TASK_INDEX]
    formatted = format_mbpp_example(row)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    prompt_ids = tokenizer(
        formatted.prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    code_ids = tokenizer(
        formatted.code,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if code_ids.shape[1] < TRANSITION_INDEX + 2:
        raise RuntimeError("frozen MBPP task does not contain the required target transition")

    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    recurrent_layers = tuple(
        index
        for index, layer_type in enumerate(model.config.layer_types)
        if layer_type == "linear_attention"
    )
    heads = int(model.config.linear_num_value_heads)
    key_rows = int(model.config.linear_key_head_dim)
    value_dim = int(model.config.linear_value_head_dim)
    if (len(recurrent_layers), heads, key_rows, value_dim) != (18, 16, 128, 128):
        raise RuntimeError(
            "Pinned Qwen3.5-0.8B recurrent geometry changed; refusing to adapt fixed strata"
        )
    strata = _resolve_strata(recurrent_layers, heads=heads, rows=key_rows)

    int4_spec = QuantizationSpec(
        bits=4,
        group_size=value_dim,
        scale_bits=16,
        flatten_last_dims=1,
        rounding="nearest",
        seed=SEED,
    )
    int8_spec = QuantizationSpec(
        bits=8,
        group_size=value_dim,
        scale_bits=16,
        flatten_last_dims=1,
        rounding="nearest",
        seed=SEED,
    )

    raw_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(
            prompt_ids,
            past_key_values=raw_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    previous_ids = code_ids[:, :TRANSITION_INDEX]
    advance_uniform_int4_trajectory(
        model,
        raw_cache,
        previous_ids,
        int4_spec=int4_spec,
        forward_kwargs={"logits_to_keep": 1},
    )
    input_ids = code_ids[:, TRANSITION_INDEX : TRANSITION_INDEX + 1]
    target_ids = code_ids[:, TRANSITION_INDEX + 1 : TRANSITION_INDEX + 2]
    raw_cache_fingerprint = _cache_fingerprint(raw_cache)

    row_records: list[dict[str, object]] = []
    for row_number, (label, location) in enumerate(strata, start=1):
        reference: StorageBoundaryRowValidation | None = None
        epsilon_records: list[dict[str, object]] = []
        for epsilon_number, epsilon in enumerate(EPSILONS, start=1):
            before = _cache_fingerprint(raw_cache)
            if before != raw_cache_fingerprint:
                raise RuntimeError("raw boundary cache changed before validation")
            result = validate_qwen_storage_boundary_row(
                model,
                raw_cache,
                input_ids,
                target_ids,
                location=location,
                int4_spec=int4_spec,
                int8_spec=int8_spec,
                epsilon=epsilon,
                sign_floor=DERIVATIVE_INFORMATIVE_FLOOR,
                forward_kwargs={"logits_to_keep": 1},
            )
            after = _cache_fingerprint(raw_cache)
            if after != before:
                raise RuntimeError("validation mutated the raw boundary cache")
            if reference is None:
                reference = result
            else:
                _assert_same_row_result(reference, result)
            epsilon_records.append(asdict(result.comparison))
            print(
                f"row {row_number}/{len(strata)} {label} "
                f"epsilon {epsilon_number}/{len(EPSILONS)}={epsilon:.5f} "
                f"predicted={result.comparison.predicted_benefit_autograd:.9g} "
                f"central={result.comparison.predicted_benefit_central:.9g} "
                f"endpoint={result.comparison.measured_endpoint_benefit:.9g}",
                flush=True,
            )

        assert reference is not None
        dot_product = _directional_dot_float64(
            reference.gradient_row,
            reference.direction,
        )
        if not math.isclose(
            -dot_product,
            reference.comparison.predicted_benefit_autograd,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("stored gradient and direction do not reproduce Taylor benefit")
        row_records.append(
            {
                "stratum": label,
                "location": asdict(location),
                "selection_basis": "fixed geometry only; no selector scores or benefits read",
                "tensors": _row_tensors_record(reference),
                "physical_endpoints": {
                    "int4": _physical_endpoint_record(
                        reference.raw_row,
                        reference.int4_row,
                        int4_spec,
                    ),
                    "int8": _physical_endpoint_record(
                        reference.raw_row,
                        reference.int8_row,
                        int8_spec,
                    ),
                },
                "negative_gradient_dot_direction": -dot_product,
                "epsilon_results": epsilon_records,
            }
        )

    if _cache_fingerprint(raw_cache) != raw_cache_fingerprint:
        raise RuntimeError("raw boundary cache changed during the diagnostic")
    derivative_gate = _evaluate_derivative_gate(row_records)
    source_hashes_end = _source_hashes(implementation_paths)
    if source_hashes_end != source_hashes_start:
        raise RuntimeError("implementation source changed while the diagnostic was running")
    repository_end = _git_state()

    config_payload = model.config.to_dict()
    prompt_token_ids = prompt_ids.detach().cpu().reshape(-1).tolist()
    code_token_ids = code_ids.detach().cpu().reshape(-1).tolist()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "diagnostic_only": True,
        "claim_boundary": (
            "One pinned calibration task, one target transition, four rows fixed by geometry, "
            "and four finite-difference step sizes. This checks implementation sign and local "
            "derivative consistency only. It does not establish policy quality, held-out "
            "generalization, free generation, statistical significance, novelty, speed, "
            "memory improvement, or a breakthrough."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": model.__class__.__name__,
            "dtype": str(dtype),
            "device": str(device),
            "attention_implementation": "eager",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "config_canonical_sha256": _sha256_bytes(canonical_json_bytes(config_payload)),
            "recurrent_geometry": {
                "model_layer_indices": list(recurrent_layers),
                "value_heads": heads,
                "key_rows": key_rows,
                "value_dim": value_dim,
            },
        },
        "dataset": {
            "phase": "calibration",
            "development_or_confirmation_loaded": False,
            "manifest": mbpp_manifest(rows, phase="calibration"),
            "manifest_sha256": mbpp_manifest_sha256(rows, phase="calibration"),
            "task_prefix_limit": TASK_PREFIX_LIMIT,
            "task_index": TASK_INDEX,
            "task_id": int(row["task_id"]),
            "prompt_utf8_sha256": _sha256_bytes(formatted.prompt.encode("utf-8")),
            "code_utf8_sha256": _sha256_bytes(formatted.code.encode("utf-8")),
            "prompt_tokens": len(prompt_token_ids),
            "code_tokens": len(code_token_ids),
            "prompt_token_ids_canonical_sha256": _sha256_bytes(
                canonical_json_bytes(prompt_token_ids)
            ),
            "code_token_ids_canonical_sha256": _sha256_bytes(
                canonical_json_bytes(code_token_ids)
            ),
            "transition_index": TRANSITION_INDEX,
            "input_token_id": int(input_ids.item()),
            "target_token_id": int(target_ids.item()),
        },
        "method": {
            "quantity_checked": "-gradient dot (Q8(raw row) - Q4(raw row))",
            "loss": "teacher-forced next-token target NLL",
            "trajectory": (
                "uniform physical-INT4 pack/dequantize storage before every prior token; "
                "raw update retained at the measured boundary"
            ),
            "storage_path": "Q4(raw) + alpha * (Q8(raw) - Q4(raw)) for one row",
            "deployable_alphas": [0, 1],
            "negative_alpha_note": (
                "-epsilon is controlled extrapolation used only for central difference"
            ),
            "epsilons": list(EPSILONS),
            "forward_count": len(ROW_STRATA) * len(EPSILONS) * 5,
            "cache_isolation": {
                "procedure": (
                    "The raw cache fingerprint is checked before and after every "
                    "row-epsilon validation. Each alpha forward deep-copies an identical "
                    "physical-INT4 boundary cache."
                ),
                "fingerprint_scope": (
                    "Tensor shapes, dtypes, and bytes for keys, values, convolution states, "
                    "and recurrent states; cache/layer classes; layer count; and the listed "
                    "cache-history, initialization, geometry, dtype, and device metadata."
                ),
                "cache_metadata_attributes": list(CACHE_FINGERPRINT_CACHE_METADATA),
                "layer_metadata_attributes": list(CACHE_FINGERPRINT_LAYER_METADATA),
                "claim_limit": (
                    "This fingerprint does not claim to serialize arbitrary unlisted Python "
                    "attributes or external model state."
                ),
            },
            "row_selection": {
                "frozen_in_source": True,
                "uses_measured_benefit": False,
                "strata_source_coordinates": [
                    {
                        "label": label,
                        "recurrent_layer_ordinal": ordinal,
                        "head_index": head,
                        "row_index": key_row,
                    }
                    for label, ordinal, head, key_row in ROW_STRATA
                ],
            },
        },
        "quantizers": {
            "int4": asdict(int4_spec),
            "int8": asdict(int8_spec),
            "physical_endpoint_note": (
                "Q4 and Q8 endpoint rows are decoded from actual packed payloads and FP16 scales; "
                "intermediate alphas are floating-point interventions"
            ),
        },
        "raw_boundary_cache_fingerprint_sha256": raw_cache_fingerprint,
        "rows": row_records,
        "derivative_gate": derivative_gate,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "implementation": {
            "source_hashes_start": source_hashes_start,
            "source_hashes_end": source_hashes_end,
            "unchanged_during_run": source_hashes_start == source_hashes_end,
        },
        "repository": {
            "start": repository_start,
            "end": repository_end,
        },
        "command": [sys.executable, *sys.argv],
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "canonical_evidence_sha256": _sha256_bytes(canonical_json_bytes(evidence)),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": _sha256_bytes(payload),
                "canonical_evidence_sha256": artifact["canonical_evidence_sha256"],
                "task_id": int(row["task_id"]),
                "transition_index": TRANSITION_INDEX,
                "rows": len(row_records),
                "epsilons": list(EPSILONS),
                "passed": derivative_gate["passed"],
                "failures": derivative_gate["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if derivative_gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
