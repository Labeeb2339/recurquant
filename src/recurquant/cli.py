"""Command-line entry points for RecurQuant."""

from __future__ import annotations

import argparse
import json

import torch

from .quantization import QuantizationSpec, quantize_dequantize
from .qwen35_quickstart import add_qwen35_arguments, run_qwen35_quickstart


def _demo(args: argparse.Namespace) -> int:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    state = torch.randn(
        (args.batch, args.heads, args.key_dim, args.value_dim),
        generator=generator,
        dtype=torch.float32,
    )
    result = quantize_dequantize(
        state,
        QuantizationSpec(
            bits=args.bits,
            group_size=args.group_size,
            rounding=args.rounding,
            seed=args.seed,
        ),
    )
    print(json.dumps(result.evidence_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recurquant",
        description="Persistent-state quantization research utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run a synthetic state round-trip.")
    demo.add_argument("--bits", type=int, default=4)
    demo.add_argument("--group-size", type=int, default=128)
    demo.add_argument("--rounding", choices=("nearest", "stochastic"), default="nearest")
    demo.add_argument("--seed", type=int, default=2339)
    demo.add_argument("--batch", type=int, default=1)
    demo.add_argument("--heads", type=int, default=2)
    demo.add_argument("--key-dim", type=int, default=16)
    demo.add_argument("--value-dim", type=int, default=16)
    demo.set_defaults(handler=_demo)

    qwen35 = subparsers.add_parser(
        "qwen35",
        help="Run the pinned Qwen3.5 model with the packed recurrent-state cache.",
    )
    add_qwen35_arguments(qwen35)
    qwen35.set_defaults(handler=run_qwen35_quickstart)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
