"""Shared installed workflow for the pinned Qwen3.5 RecurQuant quickstart."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evidence import canonical_json_bytes, verify_evidence_artifact
from .qwen35 import (
    create_qwen35_packed_cache,
    create_qwen35_rank_fused_exact_budget_cache,
    create_qwen35_v02_mixed_cache,
)
from .row_policy import ExactBudgetRowPlan, select_rows_exact_budget

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DEFAULT_PROMPT = "Explain recurrent-state quantization in two sentences."
MIXED_POLICY = "mixed-v02"
UNIFORM_INT4_STRESS_POLICY = "uniform-int4-stress"
RANK_FUSED_POLICY = "rank-fused-target-fisher"
DEFAULT_SELECTOR_METHOD = "target_directional_fisher_difference_int4"
DEFAULT_RANK_WEIGHT = 0.5


def add_qwen35_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared Qwen3.5 quickstart arguments to ``parser``."""

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--policy",
        choices=(MIXED_POLICY, UNIFORM_INT4_STRESS_POLICY, RANK_FUSED_POLICY),
        default=MIXED_POLICY,
        help=(
            "mixed-v02 uses the frozen layer-0 INT8/rest INT4 policy; "
            "uniform-int4-stress retains uniform INT4 only as a stress baseline; "
            "rank-fused-target-fisher combines target-Fisher priors with dynamic "
            "INT4-to-INT8 MSE reduction"
        ),
    )
    parser.add_argument(
        "--selector-artifact",
        type=Path,
        help=(
            "For rank-fused policy only: path to a calibrated selector evidence JSON "
            "artifact containing score arrays and the matching plan."
        ),
    )
    parser.add_argument(
        "--selector-method",
        default=DEFAULT_SELECTOR_METHOD,
        help=(
            "Selector score name in --selector-artifact (for rank-fused policy only). "
            f"Default: {DEFAULT_SELECTOR_METHOD}"
        ),
    )
    parser.add_argument(
        "--static-rank-weight",
        type=float,
        default=DEFAULT_RANK_WEIGHT,
        help="Rank-fusion weight in [0, 1] between static and dynamic ranks.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already-cached copy of the pinned model and tokenizer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON document instead of human-readable output.",
    )
    return parser


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def _model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    warnings.warn(
        "CUDA BF16 is unavailable; falling back to FP16. RecurQuant's public "
        "full-model fidelity evidence has not been validated for FP16 weights.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.float16


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sha256_strict(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_rank_fused_selector_artifact(
    path: Path,
    *,
    selector_method: str,
) -> tuple[ExactBudgetRowPlan, dict[int, torch.Tensor]]:
    """Load selector scores and exact plan for rank-fused cache construction."""

    verification = verify_evidence_artifact(path)
    if not verification["valid"]:
        details = "; ".join(verification["errors"])
        raise ValueError(
            f"selector artifact failed evidence verification: {details or 'unknown error'}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selector artifact root must be an object")

    evidence = raw.get("evidence", raw)
    if not isinstance(evidence, dict):
        raise ValueError("selector artifact evidence payload must be an object")

    scores_root = _require_mapping(evidence.get("scores"), "artifact scores")
    method_record = _require_mapping(
        scores_root.get(selector_method),
        f"artifact scores for method {selector_method!r}",
    )
    raw_arrays = _require_mapping(
        method_record.get("arrays"),
        f"artifact method {selector_method!r} arrays",
    )
    recorded_arrays_sha256 = method_record.get("canonical_arrays_sha256")
    if isinstance(recorded_arrays_sha256, str):
        actual_arrays_sha256 = _sha256_strict(dict(raw_arrays))
        if recorded_arrays_sha256 != actual_arrays_sha256:
            raise ValueError(
                "selector array hash mismatch for "
                f"{selector_method}: recorded {recorded_arrays_sha256}, "
                f"computed {actual_arrays_sha256}"
            )

    static_scores: dict[int, torch.Tensor] = {}
    for raw_layer, values in raw_arrays.items():
        if not isinstance(raw_layer, str):
            raise ValueError("selector layer keys must be strings")
        try:
            layer_index = int(raw_layer)
        except ValueError as error:
            raise ValueError(
                f"selector layer key {raw_layer!r} is not an integer string"
            ) from error
        if layer_index < 0:
            raise ValueError(f"selector layer key {raw_layer!r} must be non-negative")
        if layer_index in static_scores:
            raise ValueError(f"selector layer key {layer_index} appears twice")

        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(
                f"selector score matrix for layer {layer_index} must be 2D"
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"selector score matrix for layer {layer_index} must be floating-point"
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"selector score matrix for layer {layer_index} must be finite"
            )
        static_scores[layer_index] = tensor

    if not static_scores:
        raise ValueError(f"no selector scores found for method {selector_method!r}")

    byte_budget = _require_mapping(evidence.get("byte_budget"), "artifact byte budget")
    required_budget_fields = (
        "low_bits",
        "high_bits",
        "group_size",
        "scale_bits",
        "target_resident_bytes",
    )
    for field in required_budget_fields:
        if field not in byte_budget:
            raise ValueError(f"artifact byte budget is missing {field!r}")

    plan = select_rows_exact_budget(
        static_scores,
        target_resident_bytes=int(byte_budget["target_resident_bytes"]),
        low_bits=int(byte_budget["low_bits"]),
        high_bits=int(byte_budget["high_bits"]),
        group_size=int(byte_budget["group_size"]),
        scale_bits=int(byte_budget["scale_bits"]),
    )

    plans = _require_mapping(evidence.get("plans"), "artifact plans")
    method_plan_record = _require_mapping(
        plans.get(selector_method),
        f"artifact plan for method {selector_method!r}",
    )
    raw_locations = method_plan_record.get("locations")
    if not isinstance(raw_locations, list):
        raise ValueError(
            f"artifact method {selector_method!r} locations must be a list"
        )

    declared_locations: list[tuple[int, int, int]] = []
    for raw_location in raw_locations:
        location = _require_mapping(
            raw_location,
            f"artifact method {selector_method!r} location item",
        )
        try:
            declared_locations.append(
                (
                    int(location["layer_index"]),
                    int(location["head_index"]),
                    int(location["row_index"]),
                )
            )
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(
                f"artifact method {selector_method!r} location is malformed"
            ) from error

    planned_locations = tuple(
        (location.layer_index, location.head_index, location.row_index)
        for location in sorted(plan.high_precision_rows)
    )
    if tuple(declared_locations) != planned_locations:
        if len(declared_locations) != len(planned_locations):
            raise ValueError(
                "selector artifact plan location count does not match the contract: "
                f"{len(planned_locations)} expected, {len(declared_locations)} declared"
            )
        raise ValueError(
            "selector artifact locations do not match computed plan for contract"
        )

    return plan, static_scores


def _coerce_rank_weight(value: float) -> float:
    if not isinstance(value, (float, int)):
        raise TypeError("static rank weight must be a real number")
    rank_weight = float(value)
    if not math.isfinite(rank_weight) or not (0.0 <= rank_weight <= 1.0):
        raise ValueError("static rank weight must be finite and in [0, 1]")
    return rank_weight


def _create_cache(
    model: torch.nn.Module,
    policy: str,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if policy == MIXED_POLICY:
        return create_qwen35_v02_mixed_cache(model)
    if policy == UNIFORM_INT4_STRESS_POLICY:
        warnings.warn(
            "Uniform INT4 is retained only as a stress baseline; the default mixed-v02 "
            "policy is the repository's frozen development policy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return create_qwen35_packed_cache(model, bits=4, group_size=128)
    if policy == RANK_FUSED_POLICY:
        if args.selector_artifact is None:
            raise ValueError(
                "--selector-artifact is required when --policy is "
                "rank-fused-target-fisher"
            )
        plan, static_scores = _load_rank_fused_selector_artifact(
            args.selector_artifact,
            selector_method=str(args.selector_method),
        )
        first_param = next(iter(model.parameters()), None)
        first_buffer = next(iter(model.buffers()), None)
        model_device = (
            first_param.device
            if first_param is not None
            else first_buffer.device
            if first_buffer is not None
            else torch.device("cpu")
        )
        return create_qwen35_rank_fused_exact_budget_cache(
            model,
            plan=plan,
            static_scores_by_layer={
                layer: scores.to(model_device) for layer, scores in static_scores.items()
            },
            static_rank_weight=_coerce_rank_weight(args.static_rank_weight),
        )
    raise ValueError(f"unknown Qwen3.5 quickstart policy: {policy!r}")


def run_qwen35_quickstart(args: argparse.Namespace) -> int:
    """Run the pinned model and print generated text plus recurrent-state bytes."""

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    device = _device(args.device)
    dtype = _model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    cache = _create_cache(model, args.policy, args)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    generated: list[torch.Tensor] = []

    with torch.inference_mode():
        output = model(
            **encoded,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        for step in range(args.max_new_tokens):
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            reached_eos = (
                tokenizer.eos_token_id is not None
                and bool((next_token == tokenizer.eos_token_id).all().item())
            )
            if reached_eos or step + 1 == args.max_new_tokens:
                break
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

    generated_ids = torch.cat(generated, dim=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    summary = cache.storage_summary()

    rank_weight = (
        _coerce_rank_weight(float(args.static_rank_weight))
        if args.policy == RANK_FUSED_POLICY
        else None
    )

    if args.json:
        print(
            json.dumps(
                {
                    "generated_text": generated_text,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "policy": args.policy,
                    "selector_method": (
                        str(args.selector_method)
                        if args.policy == RANK_FUSED_POLICY
                        else None
                    ),
                    "selector_artifact": (
                        str(args.selector_artifact)
                        if args.selector_artifact is not None
                        else None
                    ),
                    "static_rank_weight": rank_weight,
                    "storage_summary": summary,
                },
                sort_keys=True,
            )
        )
        return 0

    print(f"policy={args.policy}")
    if args.policy == RANK_FUSED_POLICY:
        print(f"selector_method={args.selector_method}")
        print(f"static_rank_weight={rank_weight}")
        print(f"selector_artifact={args.selector_artifact}")
    print(generated_text)
    print(f"resident_recurrent_state_bytes={summary['resident_bytes']}")
    print(
        "full_precision_equivalent_recurrent_state_bytes="
        f"{summary['full_precision_equivalent_bytes']}"
    )
    print(
        "largest_materialized_recurrent_state_bytes="
        f"{summary['largest_materialized_state_bytes']}"
    )
    print(f"resident_compression_ratio={summary['resident_compression_ratio']:.3f}x")
    print(f"physical_reduction_realized={summary['physical_reduction_realized']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_qwen35_arguments(parser)
    parsed = parser.parse_args(None if argv is None else list(argv))
    return run_qwen35_quickstart(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
