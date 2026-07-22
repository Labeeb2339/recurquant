from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from recurquant.public_data import (
    MBPP_CONFIG,
    MBPP_CONFIRMATION_LOCK,
    MBPP_CONFIRMATION_TOKEN,
    MBPP_DATASET_ID,
    MBPP_REVISION,
    ConfirmationLockedError,
    MBPPPhase,
    build_mbpp_manifest,
    format_mbpp_example,
    format_mbpp_prompt_code,
    load_mbpp_phase,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
    mbpp_row_sha256,
    mbpp_source_split,
    select_mbpp_calibration,
)


def row(task_id: int, *, newline: str = "\n", code: str | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "text": f"Write task {task_id}.{newline}",
        "code": code if code is not None else f"def task_{task_id}():{newline}    return {task_id}",
        "test_list": [f"assert task_{task_id}() == {task_id}"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }


def test_pins_and_phase_split_mapping_are_frozen() -> None:
    assert MBPP_DATASET_ID == "google-research-datasets/mbpp"
    assert MBPP_CONFIG == "full"
    assert MBPP_REVISION == "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
    assert MBPP_CONFIRMATION_LOCK == MBPP_CONFIRMATION_TOKEN
    assert mbpp_source_split(MBPPPhase.CALIBRATION) == "train"
    assert mbpp_source_split("development") == "validation"
    assert mbpp_source_split("confirmation") == "test"


def test_calibration_selection_uses_lowest_frozen_task_hashes() -> None:
    candidates = [row(task_id) for task_id in range(1, 9)]
    expected_ids = [
        task_id
        for _, task_id in sorted(
            (
                hashlib.sha256(f"rq-v0.2|{task_id}".encode()).hexdigest(),
                task_id,
            )
            for task_id in range(1, 9)
        )[:3]
    ]

    selected = select_mbpp_calibration(reversed(candidates), size=3)

    assert [item["task_id"] for item in selected] == expected_ids


def test_calibration_selection_rejects_invalid_size_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_mbpp_calibration([row(1)], size=0)
    with pytest.raises(ValueError, match="exceeds"):
        select_mbpp_calibration([row(1)], size=2)
    with pytest.raises(ValueError, match="duplicate"):
        select_mbpp_calibration([row(1), row(1)], size=1)


def test_row_hash_is_canonical_across_newline_styles_and_mapping_order() -> None:
    unix = row(7, newline="\n")
    windows = dict(reversed(list(row(7, newline="\r\n").items())))

    assert mbpp_row_sha256(unix) == mbpp_row_sha256(windows)
    expected = "d5d1b650c6c244c8b0abd9526e578c08b8317b102f6e6c378cc21b919ed86bed"
    assert mbpp_row_sha256(unix) == expected
    assert mbpp_row_sha256(unix) != mbpp_row_sha256(row(7, code="def changed():\n    pass"))


def test_manifest_hash_is_order_independent_but_phase_bound() -> None:
    rows = [row(3), row(1), row(2)]

    first = mbpp_manifest_sha256(rows, phase="development")
    second = mbpp_manifest_sha256(reversed(rows), phase="development")
    confirmation = mbpp_manifest_sha256(rows, phase="confirmation")

    assert first == second
    assert first == "95ddcf8f5bc51b4445d9676bbf4c5644580f244d2de23b7abd5d12eb7820ee81"
    assert first != confirmation
    manifest = build_mbpp_manifest(rows, phase="development")
    assert mbpp_manifest(rows, phase="development") == manifest
    assert manifest["source_split"] == "validation"
    assert manifest["selection_namespace"] is None
    assert [item["task_id"] for item in manifest["rows"]] == [1, 2, 3]


def test_prompt_code_formatter_keeps_target_separate_and_normalizes_newlines() -> None:
    example = row(9, newline="\r\n", code="def answer():\r\n    return 9\r\n\r\n")
    example["text"] = "Write task 9."
    formatted = format_mbpp_prompt_code(example)

    assert formatted.prompt == (
        "You are an expert Python programmer, and here is your task: Write task 9.\n"
        "Your code should pass these tests:\n\nassert task_9() == 9\n[BEGIN]\n"
    )
    assert formatted.code == "def answer():\n    return 9\n\n"
    assert formatted.combined[formatted.code_start :] == formatted.code
    assert format_mbpp_example(row(9)).prompt.startswith("You are an expert Python programmer")


def test_loader_is_pinned_and_calibration_is_selected_after_lazy_call() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        calls.append((args, kwargs))
        return [row(task_id) for task_id in range(1, 7)]

    loaded = load_mbpp_phase(
        "calibration",
        calibration_size=2,
        load_dataset_fn=fake_loader,
    )

    assert len(loaded) == 2
    assert calls == [
        (
            (MBPP_DATASET_ID, MBPP_CONFIG),
            {"revision": MBPP_REVISION, "split": "train", "streaming": True},
        )
    ]


def test_development_maps_to_validation_without_confirmation_token() -> None:
    seen_split: list[str] = []

    def fake_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        seen_split.append(kwargs["split"])
        return [row(5)]

    assert load_mbpp_phase("development", load_dataset_fn=fake_loader)[0]["task_id"] == 5
    assert seen_split == ["validation"]


def test_script_api_applies_limit_after_frozen_calibration_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [row(task_id) for task_id in range(1, 150)]

    def fake_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        assert kwargs["split"] == "train"
        return candidates

    monkeypatch.setattr("recurquant.public_data._load_dataset_lazily", fake_loader)
    full = load_mbpp_rows("calibration")
    limited = load_mbpp_rows("calibration", limit=3)

    assert len(full) == 128
    assert limited == full[:3]
    with pytest.raises(ValueError, match="limit"):
        load_mbpp_rows("development", limit=0)


def test_confirmation_guard_blocks_before_loader_and_accepts_explicit_token() -> None:
    calls = 0

    def fake_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        assert kwargs["split"] == "test"
        return [row(601)]

    with pytest.raises(ConfirmationLockedError, match="confirmation is locked"):
        load_mbpp_phase("confirmation", load_dataset_fn=fake_loader)
    with pytest.raises(ConfirmationLockedError):
        load_mbpp_phase(
            "confirmation",
            confirmation_token="wrong-token",
            load_dataset_fn=fake_loader,
        )
    assert calls == 0

    loaded = load_mbpp_phase(
        "confirmation",
        confirmation_token=MBPP_CONFIRMATION_TOKEN,
        load_dataset_fn=fake_loader,
    )
    assert calls == 1
    assert loaded[0]["task_id"] == 601


def test_script_api_requires_explicit_confirmation_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        return [row(601)]

    monkeypatch.setattr("recurquant.public_data._load_dataset_lazily", fake_loader)
    with pytest.raises(ConfirmationLockedError):
        load_mbpp_rows("confirmation")
    assert calls == 0

    with pytest.raises(ValueError, match="limit is forbidden"):
        load_mbpp_rows(
            "confirmation",
            limit=1,
            confirmation_lock=MBPP_CONFIRMATION_LOCK,
        )
    assert calls == 0

    loaded = load_mbpp_rows(
        "confirmation",
        confirmation_lock=MBPP_CONFIRMATION_LOCK,
    )
    assert calls == 1
    assert loaded[0]["task_id"] == 601


def test_unknown_phase_and_invalid_rows_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown MBPP phase"):
        mbpp_source_split("preview")
    with pytest.raises(TypeError, match="task_id"):
        mbpp_row_sha256({**row(1), "task_id": "1"})
    with pytest.raises(TypeError, match="test_list"):
        mbpp_row_sha256({**row(1), "test_list": "assert True"})
