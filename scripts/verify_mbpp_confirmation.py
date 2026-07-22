#!/usr/bin/env python3
"""Verify the frozen RecurQuant v0.2 MBPP confirmation evidence offline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from recurquant.confirmation import verify_mbpp_confirmation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a RecurQuant v0.2 confirmation artifact against its frozen prepared "
            "manifest, optionally including the final raw checkpoint. Artifact-only quality "
            "outcomes require both expected artifact hashes."
        )
    )
    parser.add_argument("artifact", type=Path, help="Final confirmation evidence JSON")
    parser.add_argument("prepared_manifest", type=Path, help="Frozen prepared-manifest JSON")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Final evaluator checkpoint for full raw-array and state verification",
    )
    parser.add_argument(
        "--expect-artifact-sha256",
        help="Optional published SHA256 of the complete artifact file",
    )
    parser.add_argument(
        "--expect-artifact-evidence-sha256",
        help="Optional published SHA256 of the artifact's canonical evidence object",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_mbpp_confirmation(
        args.artifact,
        args.prepared_manifest,
        checkpoint_path=args.checkpoint,
        expected_artifact_sha256=args.expect_artifact_sha256,
        expected_artifact_evidence_sha256=args.expect_artifact_evidence_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
