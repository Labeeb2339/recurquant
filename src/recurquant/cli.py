"""Command-line entry points for RecurQuant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .confirmation import verify_mbpp_confirmation
from .evidence import verify_evidence_artifact
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


def _verify_artifact(args: argparse.Namespace) -> int:
    report = verify_evidence_artifact(
        args.artifact,
        expected_file_sha256=args.expect_file_sha256,
        expected_canonical_evidence_sha256=args.expect_canonical_evidence_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def _verify_confirmation(args: argparse.Namespace) -> int:
    report = verify_mbpp_confirmation(
        args.artifact,
        args.prepared_manifest,
        checkpoint_path=args.checkpoint,
        expected_artifact_sha256=args.expect_artifact_sha256,
        expected_artifact_evidence_sha256=args.expect_artifact_evidence_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["valid"] else 1


def _statelease_stage_b_identity(args: argparse.Namespace) -> int:
    """Run the frozen StateLease Stage-B identity adapter."""

    from scripts import resolve_statelease_stage_b_identity as identity_resolver

    argv = ["--output", str(args.output)]
    if bool(args.local_files_only):
        argv.append("--local-files-only")

    old_argv = sys.argv
    try:
        sys.argv = ["scripts/resolve_statelease_stage_b_identity.py", *argv]
        return identity_resolver.main()
    finally:
        sys.argv = old_argv


def _statelease_stage_b_evaluate(args: argparse.Namespace) -> int:
    """Run the frozen StateLease Stage-B development quality evaluator."""

    from scripts import evaluate_statelease_stage_b as evaluator

    argv = [
        "--stage-a-artifact",
        str(args.stage_a_artifact),
        "--identity-artifact",
        str(args.identity_artifact),
        "--output",
        str(args.output),
        "--device",
        str(args.device),
    ]
    if bool(args.local_files_only):
        argv.append("--local-files-only")
    return evaluator.main(argv)


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

    verify = subparsers.add_parser(
        "verify-artifact",
        help="Verify a JSON artifact's file and canonical evidence hashes.",
    )
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--expect-file-sha256")
    verify.add_argument("--expect-canonical-evidence-sha256")
    verify.set_defaults(handler=_verify_artifact)

    confirmation = subparsers.add_parser(
        "verify-confirmation",
        help="Verify the frozen v0.2 MBPP confirmation and its quality decision.",
    )
    confirmation.add_argument("artifact", type=Path)
    confirmation.add_argument("prepared_manifest", type=Path)
    confirmation.add_argument("--checkpoint", type=Path)
    confirmation.add_argument("--expect-artifact-sha256")
    confirmation.add_argument("--expect-artifact-evidence-sha256")
    confirmation.set_defaults(handler=_verify_confirmation)

    statelease_identity = subparsers.add_parser(
        "resolve-statelease-stage-b-identity",
        help="Resolve frozen StateLease Stage-B identities without loading model weights.",
    )
    statelease_identity.add_argument("--output", type=Path, required=True)
    statelease_identity.add_argument("--local-files-only", action="store_true")
    statelease_identity.set_defaults(handler=_statelease_stage_b_identity)

    statelease_evaluate = subparsers.add_parser(
        "evaluate-statelease-stage-b",
        help="Evaluate the frozen StateLease Stage-B development window.",
    )
    statelease_evaluate.add_argument("--stage-a-artifact", type=Path, required=True)
    statelease_evaluate.add_argument("--identity-artifact", type=Path, required=True)
    statelease_evaluate.add_argument("--output", type=Path, required=True)
    statelease_evaluate.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    statelease_evaluate.add_argument("--local-files-only", action="store_true")
    statelease_evaluate.set_defaults(handler=_statelease_stage_b_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
