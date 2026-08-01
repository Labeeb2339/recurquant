#!/usr/bin/env python3
"""Evaluate the frozen StateLease Stage-B development window.

This module is an adapter over the validated RHT-CQER Stage-B evaluator.
Evaluation mechanics, integrity gates, and bootstrap policy stay identical while the
result and identity artifacts are projected into a StateLease Stage-B naming.
"""

from __future__ import annotations

from collections.abc import Sequence

from scripts import evaluate_rht_cqer_stage_b as _rht_evaluator
from scripts import resolve_statelease_stage_b_identity as identity_resolver


def _set_statelease_aliases() -> None:
    """Patch evaluator state so artifacts and references are labeled for StateLease."""

    aliases = {
        "ARTIFACT_KIND": "recurquant_statelease_stage_b_development",
        "IDENTITY_ARTIFACT_KIND": (
            "recurquant_statelease_stage_b_identity"
        ),
        "RESULT_CLAIM_BOUNDARY": (
            "This is the frozen 32-task StateLease development result for a "
            "known right-RHT codec and one pinned model on one MBPP window. It is "
            "not performance, novelty, state-of-the-art, speed, deployment, or "
            "breakthrough evidence."
        ),
        "PROTECTED_EVALUATION_FIELD": (
            "protected_window_8_16_content_selected_retained_canonicalized_"
            "formatted_tokenized_passed_to_model_or_evaluated"
        ),
    }
    for name, value in aliases.items():
        setattr(_rht_evaluator, name, value)
        globals()[name] = value

    # Rebind identity resolver reference used throughout gate/evidence checks.
    _rht_evaluator.identity_resolver = identity_resolver


def __getattr__(name: str):  # pragma: no cover - compatibility passthrough
    try:
        return getattr(_rht_evaluator, name)
    except AttributeError as error:
        raise AttributeError(name) from error


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(dir(_rht_evaluator)) | set(globals()))


_set_statelease_aliases()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wrapped Stage-B evaluator with StateLease artifact naming."""
    return _rht_evaluator.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
