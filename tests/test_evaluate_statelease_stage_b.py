from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest


def _modules() -> tuple[object, object]:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    evaluator = import_module("scripts.evaluate_statelease_stage_b")
    identity_resolver = import_module("scripts.resolve_statelease_stage_b_identity")
    return evaluator, identity_resolver


def test_statelease_stage_b_wrapper_switches_identity_and_artifact_names() -> None:
    evaluator, identity_resolver = _modules()
    assert evaluator.identity_resolver is identity_resolver
    assert evaluator.ARTIFACT_KIND == "recurquant_statelease_stage_b_development"
    assert (
        evaluator.IDENTITY_ARTIFACT_KIND
        == "recurquant_statelease_stage_b_identity"
    )
    assert (
        "StateLease Stage-B" in evaluator.RESULT_CLAIM_BOUNDARY
        or "StateLease development" in evaluator.RESULT_CLAIM_BOUNDARY
    )


def test_statelease_stage_b_parser_disallows_extra_tuning_flags() -> None:
    evaluator, _ = _modules()
    args = evaluator.parse_args(
        [
            "--stage-a-artifact",
            str(Path("artifacts") / "stage-a.json"),
            "--identity-artifact",
            str(Path("artifacts") / "identity.json"),
            "--output",
            str(Path("artifacts") / "result.json"),
        ]
    )
    assert args.device == "auto"

    with pytest.raises(SystemExit):
        evaluator.parse_args(
            [
                "--stage-a-artifact",
                "stage-a.json",
                "--identity-artifact",
                "identity.json",
                "--output",
                "result.json",
                "--local-output-only",
            ]
        )


def test_statelease_wrapper_delegates_main(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, _ = _modules()
    calls: list[dict[str, Any]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append({"argv": argv})
        return 7

    monkeypatch.setattr(evaluator._rht_evaluator, "main", fake_main)

    code = evaluator.main(
        [
            "--stage-a-artifact",
            "a.json",
            "--identity-artifact",
            "b.json",
            "--output",
            "c.json",
        ]
    )

    assert code == 7
    assert calls == [
        {
            "argv": [
                "--stage-a-artifact",
                "a.json",
                "--identity-artifact",
                "b.json",
                "--output",
                "c.json",
            ]
        }
    ]

