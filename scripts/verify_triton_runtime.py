"""Verify the optional Windows/Linux Triton CUDA runtime with one tiny kernel."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    left,
    right,
    output,
    elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    mask = offsets < elements
    tl.store(
        output + offsets,
        tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask),
        mask=mask,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1024)
    args = parser.parse_args()
    if args.elements <= 0:
        raise ValueError("--elements must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    left = torch.randn(args.elements, device="cuda", dtype=torch.float32)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    block_size = triton.next_power_of_2(args.elements)
    _add_kernel[(1,)](
        left,
        right,
        output,
        elements=args.elements,
        block_size=block_size,
    )
    torch.cuda.synchronize()
    maximum_error = float((output - (left + right)).abs().max().item())
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "elements": args.elements,
        "maximum_absolute_error": maximum_error,
        "valid": maximum_error == 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
