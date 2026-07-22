"""Shared installed workflow for the pinned Qwen3.5 RecurQuant quickstart."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen35 import create_qwen35_packed_cache, create_qwen35_v02_mixed_cache

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DEFAULT_PROMPT = "Explain recurrent-state quantization in two sentences."
MIXED_POLICY = "mixed-v02"
UNIFORM_INT4_STRESS_POLICY = "uniform-int4-stress"


def add_qwen35_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared Qwen3.5 quickstart arguments to ``parser``."""

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--policy",
        choices=(MIXED_POLICY, UNIFORM_INT4_STRESS_POLICY),
        default=MIXED_POLICY,
        help=(
            "mixed-v02 uses the frozen layer-0 INT8/rest INT4 policy; "
            "uniform-int4-stress retains uniform INT4 only as a stress baseline"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already-cached copy of the pinned model and tokenizer.",
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


def _create_cache(model: torch.nn.Module, policy: str):
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

    cache = _create_cache(model, args.policy)
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
    print(f"policy={args.policy}")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    summary = cache.storage_summary()
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
