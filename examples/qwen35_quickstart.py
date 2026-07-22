"""Run a pinned Qwen3.5 model with RecurQuant's packed recurrent-state cache."""

from __future__ import annotations

import argparse
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recurquant import create_qwen35_packed_cache

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="Explain recurrent-state quantization in two sentences.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


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


def main() -> None:
    args = _arguments()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    device = _device(args.device)
    dtype = _model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    cache = create_qwen35_packed_cache(model, bits=4, group_size=128)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    generated: list[torch.Tensor] = []

    with torch.inference_mode():
        output = model(**encoded, past_key_values=cache, use_cache=True)
        for step in range(args.max_new_tokens):
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            reached_eos = (
                tokenizer.eos_token_id is not None
                and bool((next_token == tokenizer.eos_token_id).all().item())
            )
            if reached_eos or step + 1 == args.max_new_tokens:
                break
            output = model(input_ids=next_token, past_key_values=cache, use_cache=True)

    generated_ids = torch.cat(generated, dim=1)
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


if __name__ == "__main__":
    main()
