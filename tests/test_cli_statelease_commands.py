import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from recurquant.cli import build_parser


def _install_fake_script_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    main: Any,
) -> None:
    package_name = "scripts"
    import_module(package_name)

    fake_module = types.ModuleType(f"{package_name}.{name}")
    fake_module.main = main
    monkeypatch.setitem(sys.modules, f"{package_name}.{name}", fake_module)


def _resolve_identity_handler(parser, output_path: Path, *, local: bool = False) -> int:
    args = parser.parse_args(
        [
            "resolve-statelease-stage-b-identity",
            "--output",
            str(output_path),
            *(["--local-files-only"] if local else []),
        ]
    )
    return int(args.handler(args))


def _evaluate_handler(
    parser,
    output_path: Path,
    *,
    local: bool = False,
    device: str = "auto",
) -> int:
    args = parser.parse_args(
        [
            "evaluate-statelease-stage-b",
            "--stage-a-artifact",
            "stage-a.json",
            "--identity-artifact",
            "identity.json",
            "--output",
            str(output_path),
            "--device",
            device,
            *(["--local-files-only"] if local else []),
        ]
    )
    return int(args.handler(args))


def test_cli_registers_statelease_stage_b_commands() -> None:
    parser = build_parser()

    identity = parser.parse_args(
        [
            "resolve-statelease-stage-b-identity",
            "--output",
            "statelease-identity.json",
        ]
    )
    assert identity.command == "resolve-statelease-stage-b-identity"
    assert identity.output == Path("statelease-identity.json")
    assert identity.local_files_only is False

    evaluate = parser.parse_args(
        [
            "evaluate-statelease-stage-b",
            "--stage-a-artifact",
            "stage-a.json",
            "--identity-artifact",
            "identity.json",
            "--output",
            "statelease-result.json",
            "--device",
            "cpu",
            "--local-files-only",
        ]
    )
    assert evaluate.command == "evaluate-statelease-stage-b"
    assert evaluate.stage_a_artifact == Path("stage-a.json")
    assert evaluate.identity_artifact == Path("identity.json")
    assert evaluate.output == Path("statelease-result.json")
    assert evaluate.device == "cpu"
    assert evaluate.local_files_only is True


def test_cli_resolve_statelease_stage_b_identity_calls_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_parser()
    calls: list[tuple[str, ...]] = []

    def fake_main() -> int:
        calls.append(tuple(sys.argv))
        return 7

    _install_fake_script_module(
        monkeypatch, "resolve_statelease_stage_b_identity", main=fake_main
    )

    code = _resolve_identity_handler(parser, Path("identity.json"), local=True)

    assert code == 7
    assert calls == [
        (
            "scripts/resolve_statelease_stage_b_identity.py",
            "--output",
            "identity.json",
            "--local-files-only",
        )
    ]


def test_cli_evaluate_statelease_stage_b_calls_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = build_parser()
    received: list[tuple[tuple[str, ...] | None]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        received.append((tuple(argv) if argv is not None else None,))
        return 9

    _install_fake_script_module(
        monkeypatch, "evaluate_statelease_stage_b", main=fake_main
    )

    code = _evaluate_handler(
        parser,
        tmp_path / "statelease-result.json",
        local=True,
        device="cuda",
    )

    assert code == 9
    assert received == [
        (
            (
                "--stage-a-artifact",
                "stage-a.json",
                "--identity-artifact",
                "identity.json",
                "--output",
                str(tmp_path / "statelease-result.json"),
                "--device",
                "cuda",
                "--local-files-only",
            ),
        )
    ]
