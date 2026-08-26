from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_static_q468_ruler_public_canary.py"
)
SPEC = importlib.util.spec_from_file_location("verify_static_q468_ruler_public_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_allowlist(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    payloads = {
        receipt.filename: f"fixture public receipt {index:02d}\n".encode()
        for index, receipt in enumerate(canary.PUBLIC_RECEIPTS)
    }
    receipts = tuple(
        canary.PublicReceipt(
            receipt.filename,
            receipt.config,
            len(payloads[receipt.filename]),
            _sha256(payloads[receipt.filename]),
        )
        for receipt in canary.PUBLIC_RECEIPTS
    )
    monkeypatch.setattr(canary, "PUBLIC_RECEIPTS", receipts)
    monkeypatch.setattr(
        canary,
        "TRUSTED_PUBLIC_TOTAL_SIZE",
        sum(receipt.size_bytes for receipt in receipts),
    )
    monkeypatch.setattr(
        canary,
        "TRUSTED_PUBLIC_AGGREGATE_SHA256",
        canary._allowlist_aggregate(receipts),
    )
    return payloads


def _cli_arguments(tmp_path: Path) -> list[str]:
    return [
        "--ruler-root",
        str(tmp_path / "ruler-source"),
        "--git-executable",
        str(tmp_path / "git.exe"),
        "--python",
        sys.executable,
        "--tokenizer-dir",
        str(tmp_path / "tokenizer"),
        "--nltk-data",
        str(tmp_path / "nltk-data"),
        "--canary-raw-dir",
        str(tmp_path / "canary-raw"),
        "--canary-output-dir",
        str(tmp_path / "canary-output"),
        "--official-raw-dir",
        str(tmp_path / "official-raw"),
        "--official-output-dir",
        str(tmp_path / "official-output"),
        "--timeout-seconds",
        "7",
    ]


def _option_path(command: list[str], option: str) -> Path:
    return Path(command[command.index(option) + 1])


def _selected_receipts(command: list[str]) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command) if value == "--receipt"]


def _publish_fixture_canary(command: list[str], payloads: dict[str, bytes]) -> None:
    raw_root = _option_path(command, "--raw-dir")
    output_root = _option_path(command, "--output-dir")
    assert raw_root.is_dir() and not any(raw_root.iterdir())
    assert output_root.is_dir() and not any(output_root.iterdir())
    for receipt in canary.PUBLIC_RECEIPTS:
        (output_root / receipt.filename).write_bytes(payloads[receipt.filename])
        receipt_root = raw_root / receipt.filename.removesuffix(".json")
        (receipt_root / receipt.config).mkdir(parents=True)
        for relative in (
            "command-manifest.json",
            "runtime-manifest.json",
            "stdout.log",
            "stderr.log",
            f"{receipt.config}/validation.jsonl",
        ):
            (receipt_root / relative).write_bytes(b"public canary fixture\n")


def _success_result(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=canary._expected_generator_stdout(),
        stderr="",
    )


def test_trusted_public_allowlist_is_exact_and_has_no_protected_schedule() -> None:
    canary._validate_trusted_allowlist()

    expected_order = (
        "aggregation__cwe__l2048__s12340.json",
        "aggregation__cwe__l4096__s12340.json",
        "aggregation__fwe__l2048__s12339.json",
        "aggregation__fwe__l4096__s12339.json",
        "multi_hop_tracing__vt__l2048__s12339.json",
        "multi_hop_tracing__vt__l2048__s12340.json",
        "multi_hop_tracing__vt__l4096__s12339.json",
        "multi_hop_tracing__vt__l4096__s12340.json",
        "question_answering__qa_1__l2048__s12339.json",
        "question_answering__qa_1__l4096__s12339.json",
        "question_answering__qa_2__l2048__s12340.json",
        "question_answering__qa_2__l4096__s12340.json",
        "retrieval__niah_multikey_2__l2048__s12340.json",
        "retrieval__niah_multiquery__l2048__s12339.json",
        "retrieval__niah_multivalue__l4096__s12340.json",
        "retrieval__niah_single_1__l4096__s12339.json",
    )
    assert len(canary.PUBLIC_RECEIPTS) == 16
    assert tuple(receipt.filename for receipt in canary.PUBLIC_RECEIPTS) == expected_order
    assert expected_order == canary.PUBLIC_CANARY_ORDER
    assert sum(receipt.size_bytes for receipt in canary.PUBLIC_RECEIPTS) == 205_253
    assert canary._allowlist_aggregate(canary.PUBLIC_RECEIPTS) == (
        "275fc263510097d7a6e5cf55efe46f8f3835f81224a8dc792766ce821078d79b"
    )
    assert all(
        receipt.filename.endswith(("__s12339.json", "__s12340.json"))
        for receipt in canary.PUBLIC_RECEIPTS
    )
    assert not any(
        token in receipt.filename.casefold()
        for receipt in canary.PUBLIC_RECEIPTS
        for token in ("protected", "s2343", "s2344")
    )


def test_canonical_success_record_is_exact_at_os_pipe_boundary() -> None:
    success = {
        "schema": canary.SUCCESS_SCHEMA,
        "receipt_count": canary.TRUSTED_PUBLIC_RECEIPT_COUNT,
        "aggregate_sha256": canary.TRUSTED_PUBLIC_AGGREGATE_SHA256,
    }
    expected = canary._canonical_json_bytes(success)
    code = "\n".join(
        (
            "import importlib.util",
            "import pathlib",
            "import sys",
            f"path = pathlib.Path({str(SCRIPT)!r})",
            "spec = importlib.util.spec_from_file_location('canary_pipe_probe', path)",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            f"module._emit_canonical_stdout({success!r})",
        )
    )

    result = subprocess.run(
        [sys.executable, "-I", "-B", "-X", "utf8", "-c", code],
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout == expected
    assert len(result.stdout) == 174
    assert _sha256(result.stdout) == (
        "42d42af93655a9e9d0c41e151b7d086f85552b1ef6d4f7e0c52d8a8ba7254a5b"
    )
    assert result.stdout.endswith(b"}\n")
    assert not result.stdout.endswith(b"}\r\n")


def test_canonical_stdout_fails_closed_without_complete_binary_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canary, "sys", SimpleNamespace(stdout=object()))
    with pytest.raises(canary.PublicCanaryError, match="no binary buffer"):
        canary._emit_canonical_stdout({"status": "fixture"})

    class IncompleteBuffer:
        def write(self, data: bytes) -> int:
            return len(data) - 1

        def flush(self) -> None:
            return None

    monkeypatch.setattr(
        canary,
        "sys",
        SimpleNamespace(stdout=SimpleNamespace(buffer=IncompleteBuffer())),
    )
    with pytest.raises(canary.PublicCanaryError, match="write was incomplete"):
        canary._emit_canonical_stdout({"status": "fixture"})

    class BrokenBuffer:
        def write(self, _data: bytes) -> int:
            raise OSError("fixture failure")

        def flush(self) -> None:  # pragma: no cover - write fails first
            raise AssertionError("must not flush")

    monkeypatch.setattr(
        canary,
        "sys",
        SimpleNamespace(stdout=SimpleNamespace(buffer=BrokenBuffer())),
    )
    with pytest.raises(canary.PublicCanaryError, match="could not be written"):
        canary._emit_canonical_stdout({"status": "fixture"})

    class BrokenFlushBuffer:
        def write(self, data: bytes) -> int:
            return len(data)

        def flush(self) -> None:
            raise OSError("fixture flush failure")

    monkeypatch.setattr(
        canary,
        "sys",
        SimpleNamespace(stdout=SimpleNamespace(buffer=BrokenFlushBuffer())),
    )
    with pytest.raises(canary.PublicCanaryError, match="could not be written"):
        canary._emit_canonical_stdout({"status": "fixture"})


def test_success_invokes_one_exact_partial_schedule_and_emits_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        _publish_fixture_canary(command, payloads)
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    assert canary.main(_cli_arguments(tmp_path)) == 0

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert _selected_receipts(command) == [receipt.filename for receipt in canary.PUBLIC_RECEIPTS]
    assert command.count("--receipt") == 16
    assert str(tmp_path / "official-raw") not in command
    assert str(tmp_path / "official-output") not in command
    assert command[:6] == [
        sys.executable,
        "-I",
        "-B",
        "-X",
        "utf8",
        str(canary.GENERATOR_SCRIPT),
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "strict"
    assert kwargs["check"] is False
    assert "timeout" not in kwargs
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()

    stdout = capsys.readouterr().out
    assert str(tmp_path) not in stdout
    assert stdout == (
        f'{{"aggregate_sha256":"{canary.TRUSTED_PUBLIC_AGGREGATE_SHA256}",'
        '"receipt_count":16,'
        '"schema":"recurquant.experiment013.ruler-public-canary-success.v1"}\n'
    )
    assert json.loads(stdout) == {
        "aggregate_sha256": canary.TRUSTED_PUBLIC_AGGREGATE_SHA256,
        "receipt_count": 16,
        "schema": canary.SUCCESS_SCHEMA,
    }


def test_preexisting_canary_or_official_root_rejects_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture_allowlist(monkeypatch)
    (tmp_path / "canary-raw").mkdir()
    marker = tmp_path / "canary-raw" / "preserve.txt"
    marker.write_bytes(b"preserve")
    launched = False

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="canary raw root must be absent"):
        canary.main(_cli_arguments(tmp_path))

    assert not launched
    assert marker.read_bytes() == b"preserve"

    (tmp_path / "canary-raw").rename(tmp_path / "retired-canary-raw")
    (tmp_path / "official-output").mkdir()
    with pytest.raises(canary.PublicCanaryError, match="official output root must be absent"):
        canary.main(_cli_arguments(tmp_path))
    assert not launched


@pytest.mark.parametrize(
    ("returncode", "stdout_transform", "stderr", "message"),
    [
        (9, lambda value: value, "", "return exact success"),
        (0, lambda value: value + "unexpected\n", "", "stdout drifted"),
        (0, lambda value: value, "warning\n", "stderr was not empty"),
    ],
)
def test_outer_subprocess_contract_fails_closed_and_preserves_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    stdout_transform: Callable[[str], str],
    stderr: str,
    message: str,
) -> None:
    _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raw_root = _option_path(command, "--raw-dir")
        output_root = _option_path(command, "--output-dir")
        (raw_root / "preserve.txt").write_bytes(b"raw incident")
        (output_root / "preserve.txt").write_bytes(b"output incident")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout_transform(canary._expected_generator_stdout()),
            stderr=stderr,
        )

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match=message):
        canary.main(_cli_arguments(tmp_path))

    assert (tmp_path / "canary-raw" / "preserve.txt").read_bytes() == b"raw incident"
    assert (tmp_path / "canary-output" / "preserve.txt").read_bytes() == b"output incident"
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_launch_error_preserves_empty_reservations_and_blocks_exact_root_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture_allowlist(monkeypatch)
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raw_root = _option_path(command, "--raw-dir")
        output_root = _option_path(command, "--output-dir")
        assert raw_root.is_dir() and not any(raw_root.iterdir())
        assert output_root.is_dir() and not any(output_root.iterdir())
        raise OSError("fixture launch failure")

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="could not be launched"):
        canary.main(_cli_arguments(tmp_path))

    assert (tmp_path / "canary-raw").is_dir()
    assert not any((tmp_path / "canary-raw").iterdir())
    assert (tmp_path / "canary-output").is_dir()
    assert not any((tmp_path / "canary-output").iterdir())
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()
    with pytest.raises(canary.PublicCanaryError, match="canary raw root must be absent"):
        canary.main(_cli_arguments(tmp_path))
    assert calls == 1


def test_official_root_created_during_launch_is_detected_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        official = tmp_path / "official-raw"
        official.mkdir()
        (official / "incident.txt").write_bytes(b"preserve")
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="official raw root did not remain absent"):
        canary.main(_cli_arguments(tmp_path))

    assert (tmp_path / "official-raw" / "incident.txt").read_bytes() == b"preserve"
    assert (tmp_path / "canary-output").is_dir()


@pytest.mark.parametrize("location", ["output-manifest", "raw-protected"])
def test_nonpublic_inventory_names_fail_closed_and_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    location: str,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        if location == "output-manifest":
            (_option_path(command, "--output-dir") / "generation-manifest.json").write_bytes(
                b"forbidden\n"
            )
        else:
            protected = _option_path(command, "--raw-dir") / "protected__s2344"
            protected.mkdir()
            (protected / "preserve.txt").write_bytes(b"forbidden\n")
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="non-public name"):
        canary.main(_cli_arguments(tmp_path))

    assert (tmp_path / "canary-raw").is_dir()
    assert (tmp_path / "canary-output").is_dir()
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_missing_raw_sibling_fails_closed_and_preserves_complete_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        first = canary.PUBLIC_RECEIPTS[0]
        missing = (
            _option_path(command, "--raw-dir") / first.filename.removesuffix(".json") / "stderr.log"
        )
        missing.unlink()
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="raw sibling inventory drifted"):
        canary.main(_cli_arguments(tmp_path))

    assert len(list((tmp_path / "canary-output").iterdir())) == 16
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_public_receipt_byte_drift_fails_closed_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        _publish_fixture_canary(command, payloads)
        first = canary.PUBLIC_RECEIPTS[0]
        path = _option_path(command, "--output-dir") / first.filename
        path.write_bytes(b"x" * len(payloads[first.filename]))
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="bytes differ"):
        canary.main(_cli_arguments(tmp_path))

    assert calls == 1
    assert (tmp_path / "canary-output" / canary.PUBLIC_RECEIPTS[0].filename).exists()
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()
    with pytest.raises(canary.PublicCanaryError, match="canary raw root must be absent"):
        canary.main(_cli_arguments(tmp_path))
    assert calls == 1


def test_redirected_output_item_is_rejected_when_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        first = canary.PUBLIC_RECEIPTS[0]
        output_root = _option_path(command, "--output-dir")
        path = output_root / first.filename
        target = output_root / "target.txt"
        target.write_bytes(payloads[first.filename])
        path.unlink()
        try:
            os.symlink(target, path)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        return _success_result(command)

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="redirected item"):
        canary.main(_cli_arguments(tmp_path))

    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_generator_hash_gate_prevents_launch_and_creates_no_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture_allowlist(monkeypatch)
    monkeypatch.setattr(canary, "GENERATOR_SCRIPT_SHA256", "0" * 64)
    launched = False

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="generator script whole-file SHA256"):
        canary.main(_cli_arguments(tmp_path))

    assert not launched
    assert not any(
        (tmp_path / name).exists()
        for name in (
            "canary-raw",
            "canary-output",
            "official-raw",
            "official-output",
        )
    )


def test_canary_roots_must_not_overlap_supplied_input_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture_allowlist(monkeypatch)
    ruler_root = tmp_path / "ruler-source"
    ruler_root.mkdir()
    arguments = _cli_arguments(tmp_path)
    arguments[arguments.index("--canary-raw-dir") + 1] = str(ruler_root / "canary-raw")
    launched = False

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="must not overlap RULER source root"):
        canary.main(arguments)

    assert not launched
    assert not (ruler_root / "canary-raw").exists()


def test_raw_addition_after_first_hash_pass_fails_final_stability_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    original_hash = canary._hash_regular_file
    public_hashes = 0

    def mutating_hash(path: Path, *, context: str) -> tuple[int, str]:
        nonlocal public_hashes
        result = original_hash(path, context=context)
        if context == "public receipt":
            public_hashes += 1
            if public_hashes == 16:
                (tmp_path / "canary-raw" / "late-addition.txt").write_bytes(b"preserve\n")
        return result

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        return _success_result(command)

    monkeypatch.setattr(canary, "_hash_regular_file", mutating_hash)
    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="canary raw root changed"):
        canary.main(_cli_arguments(tmp_path))

    assert public_hashes == 16
    assert (tmp_path / "canary-raw" / "late-addition.txt").read_bytes() == b"preserve\n"
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_output_replacement_after_first_hash_pass_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    original_hash = canary._hash_regular_file
    public_hashes = 0
    replaced_path: Path | None = None

    def mutating_hash(path: Path, *, context: str) -> tuple[int, str]:
        nonlocal public_hashes, replaced_path
        result = original_hash(path, context=context)
        if context == "public receipt":
            public_hashes += 1
            if public_hashes == 16:
                replaced_path = tmp_path / "canary-output" / canary.PUBLIC_RECEIPTS[0].filename
                replacement = (
                    bytes([replaced_path.read_bytes()[0] ^ 1]) + replaced_path.read_bytes()[1:]
                )
                replaced_path.unlink()
                replaced_path.write_bytes(replacement)
        return result

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        return _success_result(command)

    monkeypatch.setattr(canary, "_hash_regular_file", mutating_hash)
    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(
        canary.PublicCanaryError,
        match="canary output root changed|public receipt hash mismatch",
    ):
        canary.main(_cli_arguments(tmp_path))

    assert public_hashes in (16, 17)
    assert replaced_path is not None
    assert replaced_path.read_bytes() != payloads[canary.PUBLIC_RECEIPTS[0].filename]
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_output_addition_after_second_hash_pass_fails_final_rescan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    original_hash = canary._hash_regular_file
    public_hashes = 0

    def mutating_hash(path: Path, *, context: str) -> tuple[int, str]:
        nonlocal public_hashes
        result = original_hash(path, context=context)
        if context == "public receipt":
            public_hashes += 1
            if public_hashes == 32:
                (tmp_path / "canary-output" / "late-addition.json").write_bytes(b"preserve\n")
        return result

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _publish_fixture_canary(command, payloads)
        return _success_result(command)

    monkeypatch.setattr(canary, "_hash_regular_file", mutating_hash)
    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="canary output root changed"):
        canary.main(_cli_arguments(tmp_path))

    assert public_hashes == 32
    assert (tmp_path / "canary-output" / "late-addition.json").read_bytes() == b"preserve\n"
    assert not (tmp_path / "official-raw").exists()
    assert not (tmp_path / "official-output").exists()


def test_primary_failure_and_both_official_violations_are_reported_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fixture_allowlist(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        for name in ("official-raw", "official-output"):
            root = tmp_path / name
            root.mkdir()
            (root / "incident.txt").write_bytes(name.encode())
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="")

    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError) as failure:
        canary.main(_cli_arguments(tmp_path))

    message = str(failure.value)
    assert "primary=generator subprocess did not return exact success" in message
    assert "official raw root" in message
    assert "official output root" in message
    assert (tmp_path / "official-raw" / "incident.txt").read_bytes() == b"official-raw"
    assert (tmp_path / "official-output" / "incident.txt").read_bytes() == b"official-output"
    assert capsys.readouterr().out == ""


def test_original_official_parent_identity_is_reauthenticated_before_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payloads = _fixture_allowlist(monkeypatch)
    original_snapshot = canary._directory_chain_snapshot
    attempt_finished = False

    def identity_drift(
        path: Path, *, context: str
    ) -> tuple[Path, tuple[tuple[int, int, int, int], ...]]:
        resolved, identities = original_snapshot(path, context=context)
        if attempt_finished and context == "official output root parent":
            changed = list(identities)
            device, inode, mode, attributes = changed[-1]
            changed[-1] = (device, inode + 1, mode, attributes)
            identities = tuple(changed)
        return resolved, identities

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempt_finished
        _publish_fixture_canary(command, payloads)
        attempt_finished = True
        return _success_result(command)

    monkeypatch.setattr(canary, "_directory_chain_snapshot", identity_drift)
    monkeypatch.setattr(canary.subprocess, "run", fake_run)

    with pytest.raises(canary.PublicCanaryError, match="parent identity changed"):
        canary.main(_cli_arguments(tmp_path))

    assert capsys.readouterr().out == ""
