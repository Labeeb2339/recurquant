#!/usr/bin/env python3
"""Build exact-byte row plans from one-step loss sensitivity.

This script opens only the pinned MBPP calibration partition. It follows a
repeated-QDQ all-INT4 recurrent-state trajectory, differentiates the next-token
target loss once per transition, and task-macro averages the resulting row
scores. Its output is an implementation diagnostic, not held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.evidence import canonical_json_bytes
from recurquant.fisher_sensitivity import (
    GDNInt4TrajectorySensitivityCalibrator,
    TaskMacroSensitivityAccumulator,
)
from recurquant.public_data import (
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
)
from recurquant.quantization import QuantizationSpec
from recurquant.row_policy import ExactBudgetRowPlan, RowLocation, select_rows_exact_budget

SEED = 2339
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
TARGET_RESIDENT_BYTES = 2_564_096
ARTIFACT_KIND = "recurquant_loss_sensitivity_calibration_diagnostic"
PRIMARY_SCORE = "signed_taylor_next_int4"
SCORE_FIELDS = {
    PRIMARY_SCORE: "taylor_benefit",
    "target_directional_fisher_difference_int4": "directional_fisher_difference",
    "target_diagonal_fisher_difference_int4": "diagonal_fisher_difference",
    "delta_direction_magnitude_int4": "delta_direction_fisher_magnitude",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-only repeated-INT4 loss-sensitivity selector diagnostic; "
            "not a benchmark or confirmation run."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-code-tokens", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--target-resident-bytes",
        type=int,
        default=TARGET_RESIDENT_BYTES,
    )
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
    return {
        "commit": commit,
        "worktree_clean": not status,
        "status": status,
    }


def _package_versions() -> dict[str, str]:
    names = ("datasets", "numpy", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


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


def _score_payload(scores: dict[int, torch.Tensor]) -> dict[str, object]:
    arrays = {
        str(layer_index): tensor.detach().to(torch.float64).cpu().tolist()
        for layer_index, tensor in sorted(scores.items())
    }
    flattened = torch.cat(
        [
            scores[layer_index].detach().to(torch.float64).reshape(-1).cpu()
            for layer_index in sorted(scores)
        ]
    )
    encoded = canonical_json_bytes(arrays)
    return {
        "arrays": arrays,
        "canonical_arrays_sha256": sha256_bytes(encoded),
        "row_count": int(flattened.numel()),
        "minimum": float(flattened.min().item()),
        "maximum": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
        "positive_rows": int((flattened > 0).sum().item()),
        "zero_rows": int((flattened == 0).sum().item()),
        "negative_rows": int((flattened < 0).sum().item()),
    }


def _overlap(left: ExactBudgetRowPlan, right: ExactBudgetRowPlan) -> dict[str, float | int]:
    left_rows = set(left.high_precision_rows)
    right_rows = set(right.high_precision_rows)
    intersection = len(left_rows & right_rows)
    union = len(left_rows | right_rows)
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
    }


def _top_rows(
    scores: dict[int, torch.Tensor],
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


def _scores_from_accumulator(
    accumulator: TaskMacroSensitivityAccumulator,
    layer_indices: tuple[int, ...],
) -> dict[str, dict[int, torch.Tensor]]:
    by_name: dict[str, dict[int, torch.Tensor]] = {name: {} for name in SCORE_FIELDS}
    for layer_index in layer_indices:
        summary = accumulator.summary(layer_index)
        if summary.trajectory != "int4":
            raise RuntimeError("loss-sensitivity accumulator did not retain the INT4 trajectory")
        for public_name, field_name in SCORE_FIELDS.items():
            value = getattr(summary, field_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"sensitivity summary field {field_name} must be a tensor")
            by_name[public_name][layer_index] = value
    return by_name


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit <= 16:
        raise ValueError("--limit must be between 1 and 16 for a diagnostic pilot")
    if args.max_code_tokens < 2:
        raise ValueError("--max-code-tokens must be at least 2")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")

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
    if args.group_size != int(model.config.linear_value_head_dim):
        raise ValueError(
            "--group-size must equal model.config.linear_value_head_dim "
            f"({args.group_size} != {model.config.linear_value_head_dim})"
        )

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
    accumulator = TaskMacroSensitivityAccumulator()
    task_records: list[dict[str, float | int]] = []

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
        with torch.no_grad():
            model(
                prompt_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        calibrator = GDNInt4TrajectorySensitivityCalibrator(
            model,
            cache,
            int4_spec=int4_spec,
            int8_spec=int8_spec,
        )
        steps = []
        for token_index in range(code_tokens - 1):
            steps.append(
                calibrator.step(
                    code_ids[:, token_index : token_index + 1],
                    code_ids[:, token_index + 1 : token_index + 2],
                    forward_kwargs={"logits_to_keep": 1},
                )
            )
        if any(
            tuple(score.layer_index for score in step.layers) != layer_indices for step in steps
        ):
            raise RuntimeError("sensitivity step layers do not match the model GDN layers")
        accumulator.add_task(steps)
        mean_nll = fmean(step.mean_nll for step in steps)
        task_records.append(
            {
                "task_id": row["task_id"],
                "prompt_tokens": int(prompt_ids.shape[1]),
                "code_tokens": code_tokens,
                "scored_transitions": code_tokens - 1,
                "mean_target_nll_on_int4_trajectory": mean_nll,
            }
        )
        print(
            f"[{row_number}/{len(rows)}] task={row['task_id']} "
            f"code_tokens={code_tokens} transitions={code_tokens - 1} "
            f"mean_nll={mean_nll:.6f}",
            flush=True,
        )

    scores = _scores_from_accumulator(accumulator, layer_indices)
    plans = {
        name: select_rows_exact_budget(
            layer_scores,
            target_resident_bytes=args.target_resident_bytes,
            group_size=args.group_size,
        )
        for name, layer_scores in scores.items()
    }
    primary_plan = plans[PRIMARY_SCORE]
    for name, plan in plans.items():
        if plan.resident_bytes != args.target_resident_bytes:
            raise RuntimeError(f"{name} did not realize the exact target byte budget")
        if plan.promoted_group_count != primary_plan.promoted_group_count:
            raise RuntimeError(f"{name} did not use the common promotion count")

    score_payload = {name: _score_payload(layer_scores) for name, layer_scores in scores.items()}
    plan_payload = {
        name: {
            "evidence": plan.evidence_dict(),
            "promotions_by_layer": _plan_counts(plan),
            "locations": _plan_locations(plan),
        }
        for name, plan in plans.items()
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "diagnostic_only": True,
        "claim_boundary": (
            "Calibration-only, target-dependent, one-step selector diagnostic. It does "
            "not establish held-out quality, free-generation behavior, novelty, speed, "
            "peak memory, model-Fisher validity, or a breakthrough."
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
            "primary": PRIMARY_SCORE,
            "trajectory": "uniform INT4 QDQ after prefill and after every recurrent update",
            "loss": "teacher-forced next-token target NLL; one reverse pass per transition",
            "state_timing": "stored Q4 state before the next token update",
            "promotion_direction": "Q8(raw pre-storage update) minus Q4(raw pre-storage update)",
            "task_averaging": "mean transitions within task, then equal-weight task mean",
            "score_definitions": {
                PRIMARY_SCORE: "-gradient dot (Q8 - Q4); signed predicted next-token NLL reduction",
                "target_directional_fisher_difference_int4": (
                    "(gradient dot e4)^2 - (gradient dot e8)^2 using observed targets"
                ),
                "target_diagonal_fisher_difference_int4": (
                    "sum gradient^2 * (e4^2 - e8^2), without coordinate clamping"
                ),
                "delta_direction_magnitude_int4": (
                    "(gradient dot (e4 - e8))^2; non-benefit sensitivity diagnostic"
                ),
            },
            "label_free_model_fisher": (
                "not implemented in this artifact; it requires model-sampled pseudo-label probes"
            ),
            "suffix_gradient": "not implemented; this artifact scores only the next token",
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
            "promoted_rows": primary_plan.promoted_group_count,
        },
        "scores": score_payload,
        "plans": plan_payload,
        "primary_top_rows": _top_rows(scores[PRIMARY_SCORE]),
        "policy_overlap_with_primary": {
            name: _overlap(primary_plan, plan)
            for name, plan in plans.items()
            if name != PRIMARY_SCORE
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
        "artifact_kind": ARTIFACT_KIND,
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
                "primary_plan": PRIMARY_SCORE,
                "resident_bytes": primary_plan.resident_bytes,
                "promoted_rows": primary_plan.promoted_group_count,
                "promotions_by_layer": _plan_counts(primary_plan),
                "overlap": evidence["policy_overlap_with_primary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
