"""Pinned, contamination-aware utilities for the public MBPP evaluation data."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from typing import Any, Final, TypedDict

MBPP_DATASET_ID: Final = "google-research-datasets/mbpp"
MBPP_CONFIG: Final = "full"
MBPP_REVISION: Final = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_CALIBRATION_SIZE: Final = 128
MBPP_SELECTION_NAMESPACE: Final = "rq-v0.2"
MBPP_MANIFEST_SCHEMA: Final = "recurquant.mbpp-manifest.v1"
MBPP_FORMATTER_VERSION: Final = "recurquant.mbpp-prompt-code.v1"

# This is an acknowledgement token, not a secret. Requiring it makes opening the
# untouched public test split an explicit action instead of an accidental code path.
MBPP_CONFIRMATION_TOKEN: Final = "recurquant:unlock-mbpp-confirmation:rq-v0.2"
MBPP_CONFIRMATION_LOCK: Final = MBPP_CONFIRMATION_TOKEN


class MBPPPhase(StrEnum):
    """Evaluation phases and their permitted source splits."""

    CALIBRATION = "calibration"
    DEVELOPMENT = "development"
    CONFIRMATION = "confirmation"


MBPP_PHASE_SPLITS: Final[Mapping[MBPPPhase, str]] = {
    MBPPPhase.CALIBRATION: "train",
    MBPPPhase.DEVELOPMENT: "validation",
    MBPPPhase.CONFIRMATION: "test",
}


class MBPPRow(TypedDict):
    """Canonical fields exposed by the pinned MBPP ``full`` configuration."""

    task_id: int
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]


class MBPPManifestRow(TypedDict):
    task_id: int
    sha256: str


class MBPPManifest(TypedDict):
    schema: str
    dataset_id: str
    config: str
    revision: str
    phase: str
    source_split: str
    selection_namespace: str | None
    formatter_version: str
    row_count: int
    rows: list[MBPPManifestRow]


class ConfirmationLockedError(PermissionError):
    """Raised before the untouched confirmation split can be loaded."""


@dataclass(frozen=True, slots=True)
class MBPPPromptCode:
    """A stable prompt/target pair suitable for teacher-forced code scoring."""

    prompt: str
    code: str

    @property
    def combined(self) -> str:
        return self.prompt + self.code

    @property
    def code_start(self) -> int:
        """Character offset at which target code starts in :attr:`combined`."""

        return len(self.prompt)


LoadDataset = Callable[..., Iterable[Mapping[str, Any]]]


def mbpp_source_split(phase: MBPPPhase | str) -> str:
    """Return the official MBPP split assigned to an evaluation phase."""

    normalized = _coerce_phase(phase)
    return MBPP_PHASE_SPLITS[normalized]


def load_mbpp_rows(
    phase: MBPPPhase | str,
    *,
    limit: int | None = None,
    confirmation_lock: str | None = None,
) -> tuple[MBPPRow, ...]:
    """Load a frozen phase population through the simple script-facing API.

    ``limit`` only truncates calibration or development: calibration is always
    selected from the protocol's 128 ranked training rows before truncation.
    Confirmation requires ``MBPP_CONFIRMATION_LOCK`` and always loads the entire
    pinned test split, preventing partial inspection or cherry-picking.
    """

    normalized_phase = _coerce_phase(phase)
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if normalized_phase is MBPPPhase.CONFIRMATION and limit is not None:
        raise ValueError("limit is forbidden for the untouched MBPP confirmation phase")
    rows = load_mbpp_phase(
        normalized_phase,
        confirmation_token=confirmation_lock,
    )
    return rows if limit is None else rows[:limit]


def mbpp_calibration_key(row_or_task_id: Mapping[str, Any] | int) -> str:
    """Return the frozen SHA-256 ranking key for calibration selection."""

    task_id = _task_id(row_or_task_id)
    payload = f"{MBPP_SELECTION_NAMESPACE}|{task_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_mbpp_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    size: int = MBPP_CALIBRATION_SIZE,
) -> tuple[MBPPRow, ...]:
    """Select the rows with the lowest frozen calibration ranking hashes."""

    if size < 1:
        raise ValueError("calibration size must be at least 1")

    canonical_rows = tuple(canonical_mbpp_row(row) for row in rows)
    _ensure_unique_task_ids(canonical_rows)
    if size > len(canonical_rows):
        raise ValueError(
            f"calibration size {size} exceeds the available {len(canonical_rows)} rows"
        )

    ranked = sorted(
        canonical_rows,
        key=lambda row: (mbpp_calibration_key(row), row["task_id"]),
    )
    return tuple(ranked[:size])


def canonical_mbpp_row(row: Mapping[str, Any]) -> MBPPRow:
    """Validate and normalize one MBPP row for platform-stable hashing."""

    return {
        "task_id": _task_id(row),
        "text": _string_field(row, "text"),
        "code": _string_field(row, "code"),
        "test_list": _string_list_field(row, "test_list"),
        "test_setup_code": _string_field(row, "test_setup_code"),
        "challenge_test_list": _string_list_field(row, "challenge_test_list"),
    }


def mbpp_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash the canonical content of one MBPP row."""

    return hashlib.sha256(_canonical_json(canonical_mbpp_row(row))).hexdigest()


def build_mbpp_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    phase: MBPPPhase | str,
) -> MBPPManifest:
    """Build an order-independent manifest containing IDs and canonical row hashes."""

    normalized_phase = _coerce_phase(phase)
    canonical_rows = tuple(canonical_mbpp_row(row) for row in rows)
    _ensure_unique_task_ids(canonical_rows)
    manifest_rows: list[MBPPManifestRow] = [
        {"task_id": row["task_id"], "sha256": mbpp_row_sha256(row)}
        for row in sorted(canonical_rows, key=lambda item: item["task_id"])
    ]
    return {
        "schema": MBPP_MANIFEST_SCHEMA,
        "dataset_id": MBPP_DATASET_ID,
        "config": MBPP_CONFIG,
        "revision": MBPP_REVISION,
        "phase": normalized_phase.value,
        "source_split": mbpp_source_split(normalized_phase),
        "selection_namespace": (
            MBPP_SELECTION_NAMESPACE if normalized_phase is MBPPPhase.CALIBRATION else None
        ),
        "formatter_version": MBPP_FORMATTER_VERSION,
        "row_count": len(manifest_rows),
        "rows": manifest_rows,
    }


def mbpp_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    phase: MBPPPhase | str,
) -> MBPPManifest:
    """Return the canonical provenance manifest used by evaluation scripts."""

    return build_mbpp_manifest(rows, phase=phase)


def mbpp_manifest_sha256(
    rows: Iterable[Mapping[str, Any]],
    *,
    phase: MBPPPhase | str,
) -> str:
    """Hash a canonical, provenance-bearing MBPP phase manifest."""

    return hashlib.sha256(_canonical_json(build_mbpp_manifest(rows, phase=phase))).hexdigest()


def format_mbpp_prompt_code(row: Mapping[str, Any]) -> MBPPPromptCode:
    """Format a Base-model prompt and a separate code target without a chat template."""

    canonical = canonical_mbpp_row(row)
    task = canonical["text"]
    tests = "\n".join(canonical["test_list"])
    return MBPPPromptCode(
        prompt=(
            f"You are an expert Python programmer, and here is your task: {task}\n"
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
        ),
        code=canonical["code"],
    )


def format_mbpp_example(row: Mapping[str, Any]) -> MBPPPromptCode:
    """Return the stable prompt and separate reference-code target for one row."""

    return format_mbpp_prompt_code(row)


def load_mbpp_phase(
    phase: MBPPPhase | str,
    *,
    confirmation_token: str | None = None,
    calibration_size: int = MBPP_CALIBRATION_SIZE,
    load_dataset_fn: LoadDataset | None = None,
) -> tuple[MBPPRow, ...]:
    """Load one pinned MBPP phase, guarding confirmation before any data access.

    ``datasets`` is an optional dependency and is imported only when this function
    is called without an injected loader.
    """

    normalized_phase = _coerce_phase(phase)
    if normalized_phase is MBPPPhase.CONFIRMATION:
        _require_confirmation_token(confirmation_token)

    loader = load_dataset_fn or _load_dataset_lazily
    loaded = loader(
        MBPP_DATASET_ID,
        MBPP_CONFIG,
        revision=MBPP_REVISION,
        split=mbpp_source_split(normalized_phase),
    )
    if normalized_phase is MBPPPhase.CALIBRATION:
        return select_mbpp_calibration(loaded, size=calibration_size)

    canonical_rows = tuple(canonical_mbpp_row(row) for row in loaded)
    _ensure_unique_task_ids(canonical_rows)
    return canonical_rows


def _load_dataset_lazily(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MBPP loading requires the optional evaluation dependencies; "
            "install recurquant[eval]"
        ) from error
    return load_dataset(*args, **kwargs)


def _require_confirmation_token(token: str | None) -> None:
    if token is None or not hmac.compare_digest(token, MBPP_CONFIRMATION_TOKEN):
        raise ConfirmationLockedError(
            "MBPP confirmation is locked; pass the explicit MBPP_CONFIRMATION_TOKEN "
            "only after freezing the candidate and development manifest"
        )


def _coerce_phase(phase: MBPPPhase | str) -> MBPPPhase:
    try:
        return MBPPPhase(phase)
    except ValueError as error:
        choices = ", ".join(item.value for item in MBPPPhase)
        raise ValueError(f"unknown MBPP phase {phase!r}; expected one of: {choices}") from error


def _task_id(row_or_task_id: Mapping[str, Any] | int) -> int:
    raw = row_or_task_id.get("task_id") if isinstance(row_or_task_id, Mapping) else row_or_task_id
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise TypeError("MBPP task_id must be an integer")
    return int(raw)


def _string_field(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise TypeError(f"MBPP {name} must be a string")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _string_list_field(row: Mapping[str, Any], name: str) -> list[str]:
    value = row.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"MBPP {name} must be a sequence of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"MBPP {name} must contain only strings")
        normalized.append(item.replace("\r\n", "\n").replace("\r", "\n"))
    return normalized


def _ensure_unique_task_ids(rows: Iterable[MBPPRow]) -> None:
    seen: set[int] = set()
    for row in rows:
        task_id = row["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate MBPP task_id in manifest: {task_id}")
        seen.add(task_id)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
