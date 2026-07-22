#!/usr/bin/env python3
"""Run a small paired FP32-state versus QDQ-state Qwen3.5 smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.metrics import (
    paired_logit_summary,
    spearman_correlation,
    token_kl_divergence,
    top1_agreement,
)
from recurquant.quantization import QuantizationSpec
from recurquant.signals import GatedDeltaSignalRecorder
from recurquant.transformers_cache import RecurrentStateQDQCache

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 2339
PROMPT_PROFILES = {
    "retrieval": (
        "RecurQuant diagnostic sequence. The access code is violet-cedar-27. "
        "A fixed recurrent state stores compressed key-value associations. "
        "Quantization may introduce small errors that accumulate over time. "
        "This sequence is repeated only to exercise cached teacher-forced decoding. "
        "RecurQuant diagnostic sequence. The access code is violet-cedar-27. "
        "What was the access code? It was violet-cedar-27."
    ),
    "code": (
        "A rotate-left function for an unsigned eight-bit integer masks the shift by seven. "
        "First compute left = (value << shift) & 255. Then compute right = value >> (8 - shift). "
        "Return (left | right) & 255. The invariant is that rotating by eight returns the input. "
        "For value 129 and shift one, left is 2, right is 1, and the result is 3. "
        "This deterministic example exercises symbols, numbers, and repeated dependencies."
    ),
    "multilingual": (
        "Catatan makmal menyimpan kod rujukan kenyalang-314. The experiment compares a full "
        "precision recurrent state dengan keadaan yang dikuantumkan. Setiap langkah mesti guna "
        "token yang sama supaya perbandingan adil. Apakah kod rujukan dalam catatan tadi? "
        "Kod rujukannya ialah kenyalang-314, dan jawapan ini mengulang bukti asal."
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen35-smoke.json"))
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rounding", choices=("nearest", "stochastic"), default="nearest")
    parser.add_argument("--sensitivity-sweep", action="store_true")
    parser.add_argument("--low-bits", type=int, default=4)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--upgrade-layers", type=int, nargs="+")
    parser.add_argument("--prefill-tokens", type=int, default=24)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--prompt-profile", choices=tuple(PROMPT_PROFILES), default="retrieval")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def validate_state_layout(cache: object) -> list[dict[str, object]]:
    states = list(iter_recurrent_states(cache))
    if len(states) != 18:
        raise RuntimeError(f"Expected 18 initialized recurrent states, found {len(states)}")
    layout: list[dict[str, object]] = []
    for state in states:
        expected = (1, 16, 128, 128)
        if tuple(state.tensor.shape) != expected:
            raise RuntimeError(
                f"Layer {state.layer_index} state shape {tuple(state.tensor.shape)} != {expected}"
            )
        layout.append(
            {
                "layer_index": state.layer_index,
                "state_index": state.state_index,
                "shape": list(state.tensor.shape),
                "dtype": str(state.tensor.dtype),
                "bytes": state.tensor.numel() * state.tensor.element_size(),
            }
        )
    return layout


def main() -> int:
    args = parse_args()
    if not args.sensitivity_sweep and len(set(args.bits)) != len(args.bits):
        raise ValueError("--bits values must be unique")
    if args.sensitivity_sweep and args.low_bits >= args.high_bits:
        raise ValueError("--low-bits must be lower than --high-bits")
    if args.sensitivity_sweep and args.upgrade_layers is not None:
        raise ValueError("--sensitivity-sweep and --upgrade-layers are mutually exclusive")
    if args.upgrade_layers is not None and args.low_bits >= args.high_bits:
        raise ValueError("--low-bits must be lower than --high-bits")
    if args.prefill_tokens < 2 or args.decode_tokens < 1:
        raise ValueError("prefill-tokens must be >=2 and decode-tokens must be >=1")

    torch.manual_seed(SEED)
    device = select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

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

    prompt_text = PROMPT_PROFILES[args.prompt_profile]
    encoded = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    required = args.prefill_tokens + args.decode_tokens
    if encoded.shape[1] < required:
        repeats = (required // encoded.shape[1]) + 1
        encoded = encoded.repeat(1, repeats)
    tokens = encoded[:, :required].to(device)
    prefill = tokens[:, : args.prefill_tokens]
    continuation = tokens[:, args.prefill_tokens :]

    base_spec_by_bits = {
        bits: QuantizationSpec(
            bits=bits,
            group_size=args.group_size,
            rounding=args.rounding,
            seed=SEED,
        )
        for bits in set(args.bits + [args.low_bits, args.high_bits])
    }
    layer_types = list(model.config.layer_types)
    linear_layer_indices = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    ]
    if len(linear_layer_indices) != 18:
        raise RuntimeError(
            f"Expected 18 configured linear-attention layers, found {len(linear_layer_indices)}"
        )

    candidate_definitions: dict[str, dict[str, object]] = {}
    if args.sensitivity_sweep:
        candidate_definitions[f"uniform_int{args.low_bits}"] = {
            "default_spec": base_spec_by_bits[args.low_bits],
            "layer_specs": {},
        }
        for layer_index in linear_layer_indices:
            candidate_definitions[
                f"upgrade_layer_{layer_index}_int{args.low_bits}_to_int{args.high_bits}"
            ] = {
                "default_spec": base_spec_by_bits[args.low_bits],
                "layer_specs": {layer_index: base_spec_by_bits[args.high_bits]},
            }
    elif args.upgrade_layers is not None:
        unknown_layers = sorted(set(args.upgrade_layers) - set(linear_layer_indices))
        if unknown_layers:
            raise ValueError(f"Upgrade layers are not GDN layers: {unknown_layers}")
        if len(set(args.upgrade_layers)) != len(args.upgrade_layers):
            raise ValueError("--upgrade-layers must be unique")
        candidate_definitions[f"uniform_int{args.low_bits}"] = {
            "default_spec": base_spec_by_bits[args.low_bits],
            "layer_specs": {},
        }
        layer_suffix = "_".join(str(layer) for layer in sorted(args.upgrade_layers))
        candidate_definitions[
            f"upgrade_layers_{layer_suffix}_int{args.low_bits}_to_int{args.high_bits}"
        ] = {
            "default_spec": base_spec_by_bits[args.low_bits],
            "layer_specs": {
                layer_index: base_spec_by_bits[args.high_bits]
                for layer_index in args.upgrade_layers
            },
        }
    else:
        for bits in args.bits:
            candidate_definitions[f"int{bits}"] = {
                "default_spec": base_spec_by_bits[bits],
                "layer_specs": {},
            }

    reference_cache = DynamicCache(config=model.config)
    candidates = {
        label: RecurrentStateQDQCache(
            model.config,
            spec=definition["default_spec"],
            layer_specs=definition["layer_specs"],
        )
        for label, definition in candidate_definitions.items()
    }

    per_candidate_logits: dict[str, list[torch.Tensor]] = {
        label: [] for label in candidate_definitions
    }
    reference_logits: list[torch.Tensor] = []
    signal_recorder = GatedDeltaSignalRecorder(
        model,
        probe_spec=(base_spec_by_bits[args.low_bits] if args.sensitivity_sweep else None),
    )
    with torch.inference_mode(), signal_recorder:
        signal_recorder.enabled = True
        model(
            prefill,
            past_key_values=reference_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        signal_recorder.enabled = False
        for cache in candidates.values():
            model(prefill, past_key_values=cache, use_cache=True, logits_to_keep=1)

        reference_layout = validate_state_layout(reference_cache)
        for cache in candidates.values():
            validate_state_layout(cache)

        for token_index in range(continuation.shape[1]):
            token = continuation[:, token_index : token_index + 1]
            signal_recorder.enabled = True
            reference_output = model(
                token,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            reference_logits.append(reference_output.logits.detach().to(torch.float32).cpu())
            signal_recorder.enabled = False
            for label, cache in candidates.items():
                output = model(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                per_candidate_logits[label].append(output.logits.detach().to(torch.float32).cpu())

    stacked_reference = torch.cat(reference_logits, dim=1)
    candidate_results: dict[str, object] = {}
    for label, logits in per_candidate_logits.items():
        stacked_candidate = torch.cat(logits, dim=1)
        kl = token_kl_divergence(stacked_reference, stacked_candidate).flatten()
        agreement = top1_agreement(stacked_reference, stacked_candidate).flatten()
        evidence = candidates[label].update_evidence
        first_by_layer: dict[int, object] = {}
        for row in evidence:
            first_by_layer.setdefault(row.layer_index, row)
        total_baseline = sum(row.baseline_bytes for row in first_by_layer.values())
        total_estimated = sum(row.estimated_bytes for row in first_by_layer.values())
        definition = candidate_definitions[label]
        default_spec = definition["default_spec"]
        layer_specs = definition["layer_specs"]
        candidate_results[label] = {
            "policy": {
                "default_spec": asdict(default_spec),
                "layer_overrides": {
                    str(layer_index): asdict(layer_spec)
                    for layer_index, layer_spec in layer_specs.items()
                },
            },
            "summary": paired_logit_summary(stacked_reference, stacked_candidate),
            "per_token_kl": [float(value) for value in kl.tolist()],
            "per_token_top1_agreement": [bool(value) for value in agreement.tolist()],
            "persistent_state_storage": {
                "baseline_bytes": total_baseline,
                "estimated_bytes": total_estimated,
                "compression_ratio": total_baseline / total_estimated,
                "includes_scale_overhead": True,
                "physical_reduction_realized": False,
            },
            "first_update_by_layer": [
                row.evidence_dict()
                for row in sorted(
                    first_by_layer.values(),
                    key=lambda item: item.layer_index,
                )
            ],
            "cache_update_count": len(evidence),
        }

    evidence: dict[str, object] = {
        "claim_scope": {
            "diagnostic_only": True,
            "memory_reduction_realized": False,
            "latency_claim_allowed": False,
            "research_result": False,
        },
        "source": {
            "model_id": args.model_id,
            "model_revision": args.revision,
            "tokenizer_revision": args.revision,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface-hub"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "model_dtype": str(next(model.parameters()).dtype),
        },
        "schedule": {
            "seed": SEED,
            "prefill_tokens": args.prefill_tokens,
            "teacher_forced_decode_tokens": args.decode_tokens,
            "candidate_order": list(candidate_definitions),
            "group_size": args.group_size,
            "rounding": args.rounding,
            "prompt_profile": args.prompt_profile,
            "requested_upgrade_layers": args.upgrade_layers,
            "token_ids_sha256": sha256_bytes(tokens.detach().cpu().numpy().tobytes()),
            "prompt_text_sha256": sha256_bytes(prompt_text.encode("utf-8")),
        },
        "reference_state_layout": reference_layout,
        "reference_gdn_signals": [
            signal.evidence_dict() for signal in signal_recorder.records
        ],
        "candidates": candidate_results,
    }
    if args.sensitivity_sweep:
        baseline_label = f"uniform_int{args.low_bits}"
        baseline_cvar = candidate_results[baseline_label]["summary"]["cvar95_kl"]
        upgrade_labels = [label for label in candidate_results if label != baseline_label]
        best_label = min(
            upgrade_labels,
            key=lambda label: candidate_results[label]["summary"]["cvar95_kl"],
        )
        best_cvar = candidate_results[best_label]["summary"]["cvar95_kl"]
        relative_improvement = (
            (baseline_cvar - best_cvar) / baseline_cvar if baseline_cvar > 0 else 0.0
        )
        evidence["sensitivity_sweep"] = {
            "baseline_label": baseline_label,
            "best_single_layer_upgrade": best_label,
            "baseline_cvar95_kl": baseline_cvar,
            "best_cvar95_kl": best_cvar,
            "relative_cvar95_improvement": relative_improvement,
            "average_payload_bits_per_state_element": (
                (len(linear_layer_indices) - 1) * args.low_bits + args.high_bits
            )
            / len(linear_layer_indices),
            "diagnostic_headroom_gate_15_percent": relative_improvement >= 0.15,
        }
        decode_signals = [
            signal
            for signal in signal_recorder.records
            if signal.had_initial_state and signal.sequence_length == 1
        ]
        signals_by_layer = {
            layer_index: [
                signal for signal in decode_signals if signal.layer_index == layer_index
            ]
            for layer_index in linear_layer_indices
        }
        signal_summary_by_layer: dict[str, dict[str, float]] = {}
        sensitivity_improvements: list[float] = []
        signal_vectors: dict[str, list[float]] = {
            "beta_mean": [],
            "forget_activity_mean": [],
            "state_update_relative_l2": [],
            "committed_residual_rms": [],
            "probe_state_relative_l2": [],
            "probe_read_error_rms": [],
            "probe_read_relative_l2": [],
        }
        for layer_index in linear_layer_indices:
            rows = signals_by_layer[layer_index]
            if not rows:
                raise RuntimeError(f"No decode signals captured for GDN layer {layer_index}")
            summary = {
                "beta_mean": fmean(row.beta_mean for row in rows),
                "forget_activity_mean": fmean(1.0 - row.retention_mean for row in rows),
                "state_update_relative_l2": fmean(
                    row.state_update_relative_l2
                    for row in rows
                    if row.state_update_relative_l2 is not None
                ),
                "committed_residual_rms": fmean(
                    row.committed_residual_rms
                    for row in rows
                    if row.committed_residual_rms is not None
                ),
                "probe_state_relative_l2": fmean(
                    row.probe_state_relative_l2
                    for row in rows
                    if row.probe_state_relative_l2 is not None
                ),
                "probe_read_error_rms": fmean(
                    row.probe_read_error_rms
                    for row in rows
                    if row.probe_read_error_rms is not None
                ),
                "probe_read_relative_l2": fmean(
                    row.probe_read_relative_l2
                    for row in rows
                    if row.probe_read_relative_l2 is not None
                ),
            }
            signal_summary_by_layer[str(layer_index)] = summary
            upgrade_label = (
                f"upgrade_layer_{layer_index}_int{args.low_bits}_to_int{args.high_bits}"
            )
            upgrade_cvar = candidate_results[upgrade_label]["summary"]["cvar95_kl"]
            sensitivity_improvements.append(baseline_cvar - upgrade_cvar)
            for signal_name in signal_vectors:
                signal_vectors[signal_name].append(summary[signal_name])

        signal_correlations: dict[str, float | None] = {}
        for signal_name, values in signal_vectors.items():
            try:
                signal_correlations[signal_name] = spearman_correlation(
                    values,
                    sensitivity_improvements,
                )
            except ValueError:
                signal_correlations[signal_name] = None
        evidence["sensitivity_sweep"]["signal_summary_by_layer"] = signal_summary_by_layer
        evidence["sensitivity_sweep"]["signal_spearman_vs_tail_kl_improvement"] = (
            signal_correlations
        )
    canonical_evidence_sha256 = sha256_bytes(canonical_json_bytes(evidence))
    artifact: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "qwen35_recurrent_state_qdq_smoke",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "canonical_evidence_sha256": canonical_evidence_sha256,
        "evidence": evidence,
    }
    final_payload = canonical_json_bytes(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(final_payload)

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": len(final_payload),
                "artifact_sha256": sha256_bytes(final_payload),
                "canonical_evidence_sha256": canonical_evidence_sha256,
                "candidates": {
                    label: result["summary"] for label, result in candidate_results.items()
                },
                "sensitivity_sweep": evidence.get("sensitivity_sweep"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
