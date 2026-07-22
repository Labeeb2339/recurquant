"""Command-line entry points that do not require a model download."""

from __future__ import annotations

import argparse
import json

import torch

from .quantization import QuantizationSpec, quantize_dequantize


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
