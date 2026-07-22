#!/usr/bin/env python3
"""Freeze public-MBPP layer selectors for the RecurQuant v0.2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.public_data import (
    MBPP_CALIBRATION_SIZE,
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
)
from recurquant.quantization import QuantizationSpec, quantize_dequantize
from recurquant.signals import GatedDeltaSignal, GatedDeltaSignalRecorder

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 2339
RANDOM_BASELINE_SEEDS = (1101, 2202, 3303)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_worktree_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def nvidia_driver_versions(device: torch.device) -> list[str]:
    if device.type != "cuda":
        return []
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate frozen one-layer INT8 selectors on pinned MBPP train rows."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mbpp-v02-calibration.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic-only prefix of the frozen 128-row calibration population.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=list(RANDOM_BASELINE_SEEDS),
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _mean_by_layer(
    values: dict[int, list[float]],
    layer_indices: list[int],
) -> dict[str, float]:
    missing = [layer_index for layer_index in layer_indices if not values[layer_index]]
    if missing:
        raise RuntimeError(f"No calibration observations for GDN layers: {missing}")
    return {
        str(layer_index): fmean(values[layer_index])
        for layer_index in layer_indices
    }


def _argmax_layer(means: dict[str, float]) -> int:
    return min(
        (int(layer_index) for layer_index in means),
        key=lambda layer_index: (-means[str(layer_index)], layer_index),
    )


def _random_baseline_layer(layer_indices: list[int], seed: int) -> int:
    digest = hashlib.sha256(f"rq-v0.2-random|{seed}".encode()).digest()
    return layer_indices[int.from_bytes(digest, "big") % len(layer_indices)]


def _consume_signals(
    records: list[GatedDeltaSignal],
    *,
    read_risk: dict[int, list[float]],
    state_error: dict[int, list[float]],
) -> None:
    if not records:
        raise RuntimeError("Signal recorder captured no Gated DeltaNet calls")
    for record in records:
        if record.probe_read_relative_l2 is None or record.probe_state_relative_l2 is None:
            raise RuntimeError(
                f"Layer {record.layer_index} did not expose a decode-time quantization probe"
            )
        read_risk[record.layer_index].append(record.probe_read_relative_l2)
        state_error[record.layer_index].append(record.probe_state_relative_l2**2)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("--random-seeds must be unique")

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

    layer_indices = [
        index
        for index, layer_type in enumerate(model.config.layer_types)
        if layer_type == "linear_attention"
    ]
    if len(layer_indices) != 18:
        raise RuntimeError(f"Expected 18 GDN layers, found {len(layer_indices)}")

    probe_spec = QuantizationSpec(
        bits=4,
        group_size=args.group_size,
        scale_bits=16,
        rounding="nearest",
        seed=SEED,
    )
    task_mean_read_risk: dict[int, list[float]] = defaultdict(list)
    task_mean_state_error: dict[int, list[float]] = defaultdict(list)
    task_token_counts: list[dict[str, int]] = []
    scored_tokens = 0

    recorder = GatedDeltaSignalRecorder(model, probe_spec=probe_spec)
    with torch.inference_mode(), recorder:
        for row_index, row in enumerate(rows, start=1):
            task_read_risk: dict[int, list[float]] = defaultdict(list)
            task_state_error: dict[int, list[float]] = defaultdict(list)
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
            if code_ids.shape[1] < 2:
                raise RuntimeError(f"MBPP task {row['task_id']} has fewer than two code tokens")

            cache = DynamicCache(config=model.config)
            recorder.enabled = False
            model(
                prompt_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            for state in iter_recurrent_states(cache):
                qdq = quantize_dequantize(state.tensor, probe_spec)
                task_state_error[state.layer_index].append(qdq.relative_l2_error**2)
            for token_index in range(code_ids.shape[1] - 1):
                recorder.records.clear()
                recorder.enabled = True
                model(
                    code_ids[:, token_index : token_index + 1],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                recorder.enabled = False
                _consume_signals(
                    recorder.records,
                    read_risk=task_read_risk,
                    state_error=task_state_error,
                )

            for layer_index in layer_indices:
                if not task_read_risk[layer_index] or not task_state_error[layer_index]:
                    raise RuntimeError(
                        f"Task {row['task_id']} has no observations for GDN layer {layer_index}"
                    )
                task_mean_read_risk[layer_index].append(fmean(task_read_risk[layer_index]))
                task_mean_state_error[layer_index].append(fmean(task_state_error[layer_index]))

            eligible = int(code_ids.shape[1] - 1)
            scored_tokens += eligible
            task_token_counts.append(
                {
                    "task_id": row["task_id"],
                    "prompt_tokens": int(prompt_ids.shape[1]),
                    "code_tokens": int(code_ids.shape[1]),
                    "scored_state_reads": eligible,
                }
            )
            print(
                f"[{row_index}/{len(rows)}] task={row['task_id']} "
                f"code_tokens={code_ids.shape[1]}",
                flush=True,
            )

    read_risk_means = _mean_by_layer(task_mean_read_risk, layer_indices)
    state_error_means = _mean_by_layer(task_mean_state_error, layer_indices)
    candidate_layers: dict[str, int] = {
        "read_risk_l0_nearest": 0,
        "mse_selected_nearest": _argmax_layer(state_error_means),
    }
    for random_seed in args.random_seeds:
        random_layer = _random_baseline_layer(layer_indices, random_seed)
        candidate_layers[f"random_l{random_layer}_nearest"] = random_layer

    evidence: dict[str, Any] = {
        "claim_scope": {
            "protocol_eligible": len(rows) == MBPP_CALIBRATION_SIZE and args.limit is None,
            "selector_only": True,
            "development_touched": False,
            "confirmation_touched": False,
            "research_result": False,
        },
        "source": {
            "model_id": args.model_id,
            "model_revision": args.revision,
            "tokenizer_revision": args.revision,
            "dataset_manifest": dataset_manifest,
            "dataset_manifest_sha256": mbpp_manifest_sha256(rows, phase="calibration"),
            "repository_commit": git_commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver_versions": nvidia_driver_versions(device),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "model_dtype": str(next(model.parameters()).dtype),
            "command": [Path(sys.executable).name, *sys.argv],
            "tracked_worktree_clean": tracked_worktree_is_clean(),
        },
        "calibration": {
            "seed": SEED,
            "group_size": args.group_size,
            "quantization": {
                "low_bits": 4,
                "high_bits": 8,
                "scale_bits": 16,
                "rounding": "nearest",
            },
            "row_count": len(rows),
            "scored_state_reads": scored_tokens,
            "task_token_counts": task_token_counts,
            "diagnostic_read_risk_mean_by_layer": read_risk_means,
            "diagnostic_read_risk_selected_layer": _argmax_layer(read_risk_means),
            "normalized_state_mse_mean_by_layer": state_error_means,
            "averaging": "equal writes within task, then equal tasks",
            "random_baseline_seeds": list(args.random_seeds),
            "candidate_layers": candidate_layers,
        },
    }
    evidence_hash = sha256_bytes(canonical_json_bytes(evidence))
    artifact = {
        "schema_version": 1,
        "artifact_kind": "recurquant_mbpp_layer_calibration",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "canonical_evidence_sha256": evidence_hash,
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
                "canonical_evidence_sha256": evidence_hash,
                "protocol_eligible": evidence["claim_scope"]["protocol_eligible"],
                "candidate_layers": candidate_layers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
