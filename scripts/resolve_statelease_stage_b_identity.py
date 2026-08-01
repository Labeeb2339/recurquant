#!/usr/bin/env python3
"""Resolve the frozen StateLease Stage-B identity without runtime model forward.

This module intentionally reuses the proven RHT-CQER Stage-B identity pipeline and
adapts only artifact naming and resolver bindings for StateLease workflows. It
keeps the same strict contract, verification checks, and replay-safe evidence
logic, while producing a distinct StateLease identity artifact kind.
"""

from __future__ import annotations

from pathlib import Path

from scripts import resolve_rht_cqer_stage_b_identity as _resolver


def _set_statelease_aliases() -> None:
    """Patch resolver constants to expose StateLease-specific naming."""

    aliases = {
        "ARTIFACT_KIND": "recurquant_statelease_stage_b_identity",
        "IDENTITY_SCHEMA": "recurquant.experiment010-statelease-stage-b-identity.v1",
        "TOKEN_MANIFEST_SCHEMA": "recurquant.experiment010-stage-b-token-manifest.v1",
        "ORDERED_IDENTITY_SCHEMA": (
            "recurquant.experiment010-stage-b-ordered-identity.v1"
        ),
        "CLAIM_BOUNDARY": (
            "This artifact defines the frozen StateLease Stage-B data and tokenizer "
            "identity before any model weights, model forward pass, or quality metric "
            "is opened. It is not evidence of novelty, breakthrough, speed, "
            "state-of-the-art, deployment, or production readiness."
        ),
        "ROW_PLAN_METHOD": "target_directional_fisher_difference_int4",
    }
    for name, value in aliases.items():
        setattr(_resolver, name, value)
        globals()[name] = value


def __getattr__(name: str):  # pragma: no cover - compatibility passthrough
    if name == "Path":
        return Path
    if name == "__all__":
        return [item for item in dir(_resolver) if not item.startswith("_")]
    try:
        return getattr(_resolver, name)
    except AttributeError as error:
        raise AttributeError(name) from error


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(dir(_resolver)) | set(globals()))


_set_statelease_aliases()


if __name__ == "__main__":
    raise SystemExit(_resolver.main())
