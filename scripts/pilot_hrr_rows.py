#!/usr/bin/env python3
"""Run a calibration-only diagnostic pilot for HRR row selection.

This script never opens MBPP development or confirmation data. Its output is a
debug artifact for validating trace timing, score construction, and exact-byte
row-plan generation before a protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.evidence import canonical_json_bytes
from recurquant.horizon_calibration import (
    GDNHorizonCalibrationRecorder,
    TaskMacroHorizonAccumulator,
)
from recurquant.public_data import (
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
)
from recurquant.quantization import QuantizationSpec
from recurquant.row_policy import ExactBudgetRowPlan, RowLocation, select_rows_exact_budget

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 2339
TARGET_RESIDENT_BYTES = 2_564_096
HORIZON_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration-only HRR row-selector diagnostic; not a benchmark run."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-code-tokens", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--target-resident-bytes", type=int, default=TARGET_RESIDENT_BYTES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_state() -> dict[str, object]:
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
    return {"commit": commit, "worktree_clean": not status, "status": status}


def _score_dict(
    summaries: Mapping[int, object],
) -> dict[int, torch.Tensor]:
    return {
        layer_index: summary.int4_minus_int8.to(torch.float32)
        for layer_index, summary in summaries.items()
    }


def _plan_counts(plan: ExactBudgetRowPlan) -> dict[str, int]:
    counts = {str(layer_index): 0 for layer_index, _, _ in plan.score_shapes}
    for location in plan.high_precision_rows:
        counts[str(location.layer_index)] += 1
    return counts


def _plan_locations(plan: ExactBudgetRowPlan) -> list[dict[str, int]]:
    return [
        {
            "layer_index": location.layer_index,
            "head_index": location.head_index,
            "row_index": location.row_index,
        }
        for location in plan.high_precision_rows
    ]


def _score_payload(scores: Mapping[int, torch.Tensor]) -> dict[str, object]:
    arrays = {
        str(layer_index): tensor.detach().to(torch.float32).cpu().tolist()
        for layer_index, tensor in sorted(scores.items())
    }
    encoded = canonical_json_bytes(arrays)
    flattened = torch.cat([scores[index].reshape(-1).cpu() for index in sorted(scores)])
    return {
        "arrays": arrays,
        "canonical_arrays_sha256": sha256_bytes(encoded),
        "minimum": float(flattened.min().item()),
        "maximum": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
        "negative_rows": int((flattened < 0).sum().item()),
    }


def _overlap(left: ExactBudgetRowPlan, right: ExactBudgetRowPlan) -> dict[str, float | int]:
    left_set = set(left.high_precision_rows)
    right_set = set(right.high_precision_rows)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
    }


def _top_rows(
    scores: Mapping[int, torch.Tensor],
    *,
    count: int = 20,
) -> list[dict[str, float | int]]:
    candidates: list[tuple[float, RowLocation]] = []
    for layer_index, layer_scores in scores.items():
        values = layer_scores.detach().to(device="cpu", dtype=torch.float64)
        for head_index in range(values.shape[0]):
            for row_index in range(values.shape[1]):
                candidates.append(
                    (
                        float(values[head_index, row_index].item()),
                        RowLocation(layer_index, head_index, row_index),
                    )
                )
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))[:count]
    return [
        {
            "score": score,
            "layer_index": location.layer_index,
            "head_index": location.head_index,
            "row_index": location.row_index,
        }
        for score, location in ordered
    ]


def _package_versions() -> dict[str, str]:
    names = ("datasets", "numpy", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit <= 16:
        raise ValueError("--limit must be between 1 and 16 for a diagnostic pilot")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.max_code_tokens < 2:
        raise ValueError("--max-code-tokens must be at least 2")

    torch.manual_seed(SEED)
    device = select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rows = load_mbpp_rows("calibration", limit=args.limit)
    dataset_manifest = mbpp_manifest(rows, phase="calibration")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    layer_indices = tuple(
        index
        for index, layer_type in enumerate(model.config.layer_types)
        if layer_type == "linear_attention"
    )
    if len(layer_indices) != 18:
        raise RuntimeError(f"Expected 18 GDN layers, found {len(layer_indices)}")

    int4_spec = QuantizationSpec(
        bits=4,
        group_size=args.group_size,
        scale_bits=16,
        flatten_last_dims=1,
        rounding="nearest",
        seed=SEED,
    )
    int8_spec = QuantizationSpec(
        bits=8,
        group_size=args.group_size,
        scale_bits=16,
        flatten_last_dims=1,
        rounding="nearest",
        seed=SEED,
    )
    recorder = GDNHorizonCalibrationRecorder(
        model,
        layer_indices=layer_indices,
        max_tokens_per_layer=args.max_code_tokens - 1,
        int4_spec=int4_spec,
        int8_spec=int8_spec,
        epsilon=HORIZON_EPSILON,
    )
    h1_accumulator = TaskMacroHorizonAccumulator(horizon=1, epsilon=HORIZON_EPSILON)
    hrr_accumulator = TaskMacroHorizonAccumulator(
        horizon=args.horizon,
        epsilon=HORIZON_EPSILON,
    )
    mse_sums: dict[int, torch.Tensor] = {}
    task_records: list[dict[str, int]] = []

    with torch.inference_mode(), recorder:
        for row_number, row in enumerate(rows, start=1):
            formatted = format_mbpp_example(row)
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
            code_tokens = int(code_ids.shape[1])
            if code_tokens < 2:
                raise RuntimeError(f"MBPP task {row['task_id']} has fewer than two code tokens")
            if code_tokens > args.max_code_tokens:
                raise RuntimeError(
                    f"MBPP task {row['task_id']} has {code_tokens} code tokens, exceeding "
                    f"--max-code-tokens={args.max_code_tokens}; no truncation is allowed"
                )

            cache = DynamicCache(config=model.config)
            recorder.enabled = False
            model(
                prompt_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            recorder.enabled = True
            for token_index in range(code_tokens - 1):
                model(
                    code_ids[:, token_index : token_index + 1],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            recorder.enabled = False

            traces = recorder.drain_traces()
            if set(traces) != set(layer_indices):
                raise RuntimeError("captured trace layers do not match the model GDN layers")
            h1_accumulator.add_task(traces)
            hrr_accumulator.add_task(traces)
            for layer_index, trace in traces.items():
                task_marginal_mse = (
                    trace.int4_row_error_energies - trace.int8_row_error_energies
                ).mean(dim=(0, 1), dtype=torch.float64)
                if layer_index in mse_sums:
                    mse_sums[layer_index] += task_marginal_mse
                else:
                    mse_sums[layer_index] = task_marginal_mse.clone()
            retained_trace_bytes = sum(trace.retained_bytes for trace in traces.values())
            task_records.append(
                {
                    "task_id": row["task_id"],
                    "prompt_tokens": int(prompt_ids.shape[1]),
                    "code_tokens": code_tokens,
                    "captured_decode_tokens": code_tokens - 1,
                    "retained_trace_bytes": retained_trace_bytes,
                }
            )
            print(
                f"[{row_number}/{len(rows)}] task={row['task_id']} "
                f"code_tokens={code_tokens} trace_bytes={retained_trace_bytes}",
                flush=True,
            )

    h1_scores = _score_dict(h1_accumulator.summaries())
    hrr_scores = _score_dict(hrr_accumulator.summaries())
    mse_scores = {layer_index: score / len(rows) for layer_index, score in mse_sums.items()}
    plans = {
        "hrr_h1": select_rows_exact_budget(
            h1_scores,
            target_resident_bytes=args.target_resident_bytes,
            group_size=args.group_size,
        ),
        f"hrr_h{args.horizon}": select_rows_exact_budget(
            hrr_scores,
            target_resident_bytes=args.target_resident_bytes,
            group_size=args.group_size,
        ),
        "row_mse": select_rows_exact_budget(
            mse_scores,
            target_resident_bytes=args.target_resident_bytes,
            group_size=args.group_size,
        ),
    }

    plan_payload = {
        name: {
            "evidence": plan.evidence_dict(),
            "promotions_by_layer": _plan_counts(plan),
            "locations": _plan_locations(plan),
        }
        for name, plan in plans.items()
    }
    score_payload = {
        "hrr_h1": _score_payload(h1_scores),
        f"hrr_h{args.horizon}": _score_payload(hrr_scores),
        "row_mse": _score_payload(mse_scores),
    }
    primary_name = f"hrr_h{args.horizon}"
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "recurquant_hrr_calibration_diagnostic",
        "diagnostic_only": True,
        "claim_boundary": (
            "Calibration-only selector diagnostic. It does not measure language-model "
            "quality, generalization, memory peaks, latency, novelty, or a breakthrough."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "dtype": str(dtype),
            "device": str(device),
            "linear_attention_layers": list(layer_indices),
        },
        "dataset": {
            "manifest": dataset_manifest,
            "manifest_sha256": mbpp_manifest_sha256(rows, phase="calibration"),
            "tasks": task_records,
        },
        "method": {
            "horizon": args.horizon,
            "normalization_epsilon": HORIZON_EPSILON,
            "score": "finite-horizon INT4 read risk minus INT8 read risk",
            "task_averaging": "mean writes within task, then equal-weight task mean",
            "normalization": "kernel-compatible q/k L2 normalization; query scale once",
            "state_timing": "FP32 recurrent state immediately before each decode token update",
        },
        "quantizers": {
            "axis_contract": "one independent group per recurrent [head, key-row]",
            "int4": asdict(int4_spec),
            "int8": asdict(int8_spec),
        },
        "byte_budget": {
            "target_resident_bytes": args.target_resident_bytes,
            "group_size": args.group_size,
            "scale_bits": 16,
            "precision_mask_bits_per_group": 1,
        },
        "scores": score_payload,
        "plans": plan_payload,
        "primary_top_rows": _top_rows(hrr_scores),
        "policy_overlap": {
            "primary_vs_h1": _overlap(plans[primary_name], plans["hrr_h1"]),
            "primary_vs_row_mse": _overlap(plans[primary_name], plans["row_mse"]),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "repository": git_state(),
        "command": [sys.executable, *sys.argv],
    }
    canonical_evidence = canonical_json_bytes(evidence)
    artifact = {
        "schema_version": 1,
        "artifact_kind": "recurquant_hrr_calibration_diagnostic",
        "canonical_evidence_sha256": sha256_bytes(canonical_evidence),
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": sha256_bytes(payload),
                "canonical_evidence_sha256": artifact["canonical_evidence_sha256"],
                "primary_plan": primary_name,
                "resident_bytes": plans[primary_name].resident_bytes,
                "promoted_rows": plans[primary_name].promoted_group_count,
                "promotions_by_layer": _plan_counts(plans[primary_name]),
                "overlap": evidence["policy_overlap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
