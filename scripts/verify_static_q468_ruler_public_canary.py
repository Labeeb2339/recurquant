"""Run the Experiment 013 RULER generator against public calibration receipts only.

The canary is deliberately one-shot.  Its fresh raw and output roots are never
cleaned or reused, including after failure.  Official generation roots are
supplied only as absence sentinels and are never passed to the generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

GENERATOR_SCRIPT: Final = Path(os.path.abspath(__file__)).with_name(
    "generate_static_q468_ruler_receipts.py"
)
GENERATOR_SCRIPT_SHA256: Final = "3a8b2291db0162cf76e08f3a021e6c80551d4ab38ac3c0f0009f6290a46c2a63"
SUCCESS_SCHEMA: Final = "recurquant.experiment013.ruler-public-canary-success.v1"
GENERATOR_PROGRESS_SCHEMA: Final = "recurquant.experiment013.ruler-generation-progress.v1"
WINDOWS_REPARSE_POINT: Final = 0x400
TRUSTED_PUBLIC_RECEIPT_COUNT: Final = 16
TRUSTED_PUBLIC_TOTAL_SIZE: Final = 205_253
TRUSTED_PUBLIC_AGGREGATE_SHA256: Final = (
    "275fc263510097d7a6e5cf55efe46f8f3835f81224a8dc792766ce821078d79b"
)


class PublicCanaryError(RuntimeError):
    """Raised when the one-shot public canary fails closed."""


@dataclass(frozen=True, slots=True)
class PublicReceipt:
    filename: str
    config: str
    size_bytes: int
    sha256: str


# Whole-file identities authenticated from the preserved pre-G0 public receipt
# allowlist.  Protected Stage-A receipt identities are intentionally absent.
PUBLIC_RECEIPTS: Final = (
    PublicReceipt(
        "aggregation__cwe__l2048__s12340.json",
        "cwe",
        7_274,
        "1c23f946536f8b05ad81d768edf4b1a5cc74ce39a3e9048124593635eb45ad57",
    ),
    PublicReceipt(
        "aggregation__cwe__l4096__s12340.json",
        "cwe",
        10_574,
        "3a6698b2ea721d388d72f28e1c4d4646616d6952fe01ed8038c01dc53dee3a72",
    ),
    PublicReceipt(
        "aggregation__fwe__l2048__s12339.json",
        "fwe",
        8_634,
        "bb2518d8cd4ac6e41fa7aafc28a2951da762de64e8c2843d63b20f731b44ed69",
    ),
    PublicReceipt(
        "aggregation__fwe__l4096__s12339.json",
        "fwe",
        13_631,
        "7fe1d0a912ee13f2dc1aae6839a7f8e87a623d5df30a3408ac5c6059f88aed13",
    ),
    PublicReceipt(
        "multi_hop_tracing__vt__l2048__s12339.json",
        "vt",
        10_172,
        "b9a9b27500644e9aa2532304d577b85ab988c4359dd5edf7630fc8a8bee4b60e",
    ),
    PublicReceipt(
        "multi_hop_tracing__vt__l2048__s12340.json",
        "vt",
        10_172,
        "e8b50ab62299b2f7306b66c83a8c6f2b471e1198bc02aba090f1ab406da24110",
    ),
    PublicReceipt(
        "multi_hop_tracing__vt__l4096__s12339.json",
        "vt",
        17_635,
        "8ea9a21c91b1b5203697a120d21824bea8dc810455d228e58a7ae319dc72dd82",
    ),
    PublicReceipt(
        "multi_hop_tracing__vt__l4096__s12340.json",
        "vt",
        17_635,
        "5b6cd316a368db4420845994f9423aaeedc0fd75802aba9eafb7deb4f4eddc73",
    ),
    PublicReceipt(
        "question_answering__qa_1__l2048__s12339.json",
        "qa_1",
        9_159,
        "8d5f10bad274c66452a465f737da20099210de1d161c27a36c6331783da5f4f2",
    ),
    PublicReceipt(
        "question_answering__qa_1__l4096__s12339.json",
        "qa_1",
        21_074,
        "2dd908d09753092d8a358262f5073246919dd0069dfb0a60eca406aba764caa5",
    ),
    PublicReceipt(
        "question_answering__qa_2__l2048__s12340.json",
        "qa_2",
        10_515,
        "e58aa114cbdb8502b1dff17517a049edca17022eacfe90cd5afc0e2eda097c3e",
    ),
    PublicReceipt(
        "question_answering__qa_2__l4096__s12340.json",
        "qa_2",
        12_283,
        "fba640bf322cb8c1f6175cc1a73e6267b18618e8555d9601e202268427ebabe8",
    ),
    PublicReceipt(
        "retrieval__niah_multikey_2__l2048__s12340.json",
        "niah_multikey_2",
        8_407,
        "801e44bdb609773d985e8fd74f1d853f1afb30cf0a66d2cee0f7013ae8749c9c",
    ),
    PublicReceipt(
        "retrieval__niah_multiquery__l2048__s12339.json",
        "niah_multiquery",
        8_992,
        "0e8034964c18899e91a435a5737bfa29f7791fc41c29f42e4f2d9d75fa3a98ff",
    ),
    PublicReceipt(
        "retrieval__niah_multivalue__l4096__s12340.json",
        "niah_multivalue",
        21_691,
        "fc75f4bdae8929325589536065fb7c8f5bb1fd737001da616d339365aa94920a",
    ),
    PublicReceipt(
        "retrieval__niah_single_1__l4096__s12339.json",
        "niah_single_1",
        17_405,
        "bd8f1387bc9aca3118ced5d9287c08f72c670bfbe6ef88729d36d6940cc2a22f",
    ),
)
PUBLIC_CANARY_ORDER: Final = (
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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _allowlist_aggregate(receipts: Sequence[PublicReceipt]) -> str:
    payload = [
        {
            "filename": receipt.filename,
            "sha256": receipt.sha256,
            "size_bytes": receipt.size_bytes,
        }
        for receipt in receipts
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_trusted_allowlist() -> None:
    names = [receipt.filename for receipt in PUBLIC_RECEIPTS]
    if (
        len(PUBLIC_RECEIPTS) != TRUSTED_PUBLIC_RECEIPT_COUNT
        or tuple(names) != PUBLIC_CANARY_ORDER
        or len({name.casefold() for name in names}) != len(names)
    ):
        raise PublicCanaryError("trusted public receipt inventory drifted")
    for receipt in PUBLIC_RECEIPTS:
        name = receipt.filename
        if (
            Path(name).name != name
            or not name.endswith(".json")
            or not name.endswith(("__s12339.json", "__s12340.json"))
            or any(token in name.casefold() for token in ("protected", "s2343", "s2344"))
            or Path(receipt.config).name != receipt.config
            or type(receipt.size_bytes) is not int
            or receipt.size_bytes < 1
            or len(receipt.sha256) != 64
            or any(character not in "0123456789abcdef" for character in receipt.sha256)
        ):
            raise PublicCanaryError("trusted public receipt identity drifted")
    if sum(receipt.size_bytes for receipt in PUBLIC_RECEIPTS) != TRUSTED_PUBLIC_TOTAL_SIZE:
        raise PublicCanaryError("trusted public receipt size aggregate drifted")
    if _allowlist_aggregate(PUBLIC_RECEIPTS) != TRUSTED_PUBLIC_AGGREGATE_SHA256:
        raise PublicCanaryError("trusted public receipt aggregate drifted")


def _is_link_or_reparse_status(status: os.stat_result) -> bool:
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & WINDOWS_REPARSE_POINT
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        getattr(status, "st_file_attributes", 0),
    )


def _directory_chain_snapshot(
    path: Path, *, context: str
) -> tuple[Path, tuple[tuple[int, int, int, int], ...]]:
    if not path.is_absolute():
        raise PublicCanaryError(f"{context} must be absolute")
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    components = [current]
    for part in absolute.parts[1:]:
        current /= part
        components.append(current)

    def snapshot() -> tuple[tuple[int, int, int, int], ...]:
        identities: list[tuple[int, int, int, int]] = []
        for component in components:
            if not os.path.lexists(component):
                raise PublicCanaryError(f"{context} must already exist")
            try:
                status = component.lstat()
            except OSError as exc:
                raise PublicCanaryError(f"{context} is unavailable") from exc
            if _is_link_or_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
                raise PublicCanaryError(f"{context} traverses a redirected or non-directory item")
            identities.append(_directory_identity(status))
        return tuple(identities)

    before = snapshot()
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PublicCanaryError(f"{context} is unavailable") from exc
    if snapshot() != before:
        raise PublicCanaryError(f"{context} changed while it was authenticated")
    return resolved, before


def _require_existing_regular_directory(path: Path, *, context: str) -> Path:
    resolved, _identities = _directory_chain_snapshot(path, context=context)
    return resolved


@dataclass(frozen=True, slots=True)
class _PreparedAbsentRoot:
    path: Path
    parent_identities: tuple[tuple[int, int, int, int], ...]


def _normalize_absent_root(path: Path, *, context: str) -> _PreparedAbsentRoot:
    if not path.is_absolute():
        raise PublicCanaryError(f"{context} must be absolute")
    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."}:
        raise PublicCanaryError(f"{context} is not a valid fresh root")
    parent, parent_identities = _directory_chain_snapshot(
        absolute.parent, context=f"{context} parent"
    )
    candidate = parent / absolute.name
    if os.path.lexists(candidate):
        raise PublicCanaryError(f"{context} must be absent")
    return _PreparedAbsentRoot(path=candidate, parent_identities=parent_identities)


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _roots_overlap(first: Path, second: Path) -> bool:
    try:
        common = os.path.commonpath((str(first), str(second)))
    except ValueError:
        return False
    common_key = _path_key(Path(common))
    return common_key in {_path_key(first), _path_key(second)}


def _normalize_protected_input(path: Path, *, context: str) -> Path:
    if not path.is_absolute():
        raise PublicCanaryError(f"{context} must be absolute")
    try:
        return Path(os.path.abspath(path)).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PublicCanaryError(f"{context} is unavailable") from exc


def _prepare_absent_roots(
    paths: Mapping[str, Path], *, protected_paths: Mapping[str, Path]
) -> dict[str, _PreparedAbsentRoot]:
    normalized = {
        context: _normalize_absent_root(path, context=context) for context, path in paths.items()
    }
    items = list(normalized.items())
    for index, (first_context, first) in enumerate(items):
        for second_context, second in items[index + 1 :]:
            if _roots_overlap(first.path, second.path):
                raise PublicCanaryError(
                    f"{first_context} and {second_context} must be distinct non-nested roots"
                )
    protected = {
        context: _normalize_protected_input(path, context=context)
        for context, path in protected_paths.items()
    }
    for root_context, root in normalized.items():
        for protected_context, protected_path in protected.items():
            if _roots_overlap(root.path, protected_path):
                raise PublicCanaryError(f"{root_context} must not overlap {protected_context}")
    return normalized


def _require_root_still_absent(root: _PreparedAbsentRoot, *, context: str) -> None:
    parent, identities = _directory_chain_snapshot(root.path.parent, context=f"{context} parent")
    if identities != root.parent_identities:
        raise PublicCanaryError(f"{context} parent identity changed")
    candidate = parent / root.path.name
    if _path_key(candidate) != _path_key(root.path) or os.path.lexists(candidate):
        raise PublicCanaryError(f"{context} did not remain absent")


def _reserve_canary_roots(roots: Mapping[str, _PreparedAbsentRoot]) -> None:
    for context, root in roots.items():
        _require_root_still_absent(root, context=context)
    for context, root in roots.items():
        try:
            root.path.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise PublicCanaryError(f"{context} was claimed before reservation") from exc
        except OSError as exc:
            raise PublicCanaryError(f"{context} could not be reserved") from exc
        try:
            status = root.path.lstat()
            with os.scandir(root.path) as iterator:
                empty = next(iterator, None) is None
        except OSError as exc:
            raise PublicCanaryError(f"{context} reservation is unavailable") from exc
        if _is_link_or_reparse_status(status) or not stat.S_ISDIR(status.st_mode) or not empty:
            raise PublicCanaryError(f"{context} reservation is not an empty regular directory")


def _subprocess_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _generator_command(args: argparse.Namespace, *, raw_root: Path, output_root: Path) -> list[str]:
    command = [
        str(args.python),
        "-I",
        "-B",
        "-X",
        "utf8",
        str(GENERATOR_SCRIPT),
        "--ruler-root",
        str(args.ruler_root),
        "--git-executable",
        str(args.git_executable),
        "--python",
        str(args.python),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--nltk-data",
        str(args.nltk_data),
        "--raw-dir",
        str(raw_root),
        "--output-dir",
        str(output_root),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    for receipt in PUBLIC_RECEIPTS:
        command.extend(("--receipt", receipt.filename))
    return command


def _expected_generator_stdout() -> str:
    progress_lines = "".join(
        f"[{index}/{TRUSTED_PUBLIC_RECEIPT_COUNT}] generating {receipt.filename}\n"
        for index, receipt in enumerate(PUBLIC_RECEIPTS, start=1)
    )
    progress = {
        "schema": GENERATOR_PROGRESS_SCHEMA,
        "complete": False,
        "present_receipts": TRUSTED_PUBLIC_RECEIPT_COUNT,
        "required_receipts": 20,
    }
    return progress_lines + json.dumps(progress, indent=2, sort_keys=True) + "\n"


def _validate_generator_result(result: subprocess.CompletedProcess[str]) -> None:
    if type(result.returncode) is not int or result.returncode != 0:
        raise PublicCanaryError("generator subprocess did not return exact success")
    if not isinstance(result.stderr, str) or result.stderr != "":
        raise PublicCanaryError("generator subprocess stderr was not empty")
    if not isinstance(result.stdout, str) or result.stdout != _expected_generator_stdout():
        raise PublicCanaryError("generator subprocess stdout drifted")


@dataclass(frozen=True, slots=True)
class _TreeItem:
    kind: str
    path: Path
    identity: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _TreeSnapshot:
    root_identity: tuple[int, int, int, int, int, int, int]
    items: Mapping[str, _TreeItem]

    def signature(
        self,
    ) -> tuple[
        tuple[int, int, int, int, int, int, int],
        tuple[tuple[str, str, tuple[int, int, int, int, int, int, int]], ...],
    ]:
        return (
            self.root_identity,
            tuple((name, item.kind, item.identity) for name, item in sorted(self.items.items())),
        )


def _tree_item_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    stable_mtime_ns = 0 if stat.S_ISDIR(status.st_mode) else status.st_mtime_ns
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        stable_mtime_ns,
        status.st_ctime_ns,
        getattr(status, "st_file_attributes", 0),
    )


def _scan_regular_tree(root: Path, *, context: str) -> _TreeSnapshot:
    verified_root = _require_existing_regular_directory(root, context=context)
    if _path_key(verified_root) != _path_key(root):
        raise PublicCanaryError(f"{context} resolved away from its fresh root")
    try:
        root_status = verified_root.lstat()
    except OSError as exc:
        raise PublicCanaryError(f"{context} is unavailable") from exc
    if _is_link_or_reparse_status(root_status) or not stat.S_ISDIR(root_status.st_mode):
        raise PublicCanaryError(f"{context} is not a regular non-redirected directory")
    observed: dict[str, _TreeItem] = {}
    pending: list[tuple[Path, str]] = [(verified_root, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise PublicCanaryError(f"{context} is unavailable") from exc
        names = [entry.name for entry in entries]
        if len({name.casefold() for name in names}) != len(names):
            raise PublicCanaryError(f"{context} contains case-colliding names")
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PublicCanaryError(f"{context} contains an unavailable item") from exc
            if _is_link_or_reparse_status(status):
                raise PublicCanaryError(f"{context} contains a redirected item")
            path = Path(entry.path)
            if stat.S_ISDIR(status.st_mode):
                observed[relative] = _TreeItem("directory", path, _tree_item_identity(status))
                pending.append((path, relative))
            elif stat.S_ISREG(status.st_mode):
                observed[relative] = _TreeItem("file", path, _tree_item_identity(status))
            else:
                raise PublicCanaryError(f"{context} contains a non-regular item")
    return _TreeSnapshot(root_identity=_tree_item_identity(root_status), items=observed)


def _reject_nonpublic_names(items: Mapping[str, _TreeItem], *, context: str) -> None:
    for relative in items:
        for component in relative.split("/"):
            folded = component.casefold()
            if (
                folded == "generation-manifest.json"
                or "protected" in folded
                or "s2343" in folded
                or "s2344" in folded
                or "seed2343" in folded
                or "seed2344" in folded
            ):
                raise PublicCanaryError(f"{context} contains a non-public name")


def _expected_raw_inventory() -> dict[str, str]:
    expected: dict[str, str] = {}
    for receipt in PUBLIC_RECEIPTS:
        stem = receipt.filename.removesuffix(".json")
        expected[stem] = "directory"
        expected[f"{stem}/{receipt.config}"] = "directory"
        for relative in (
            "command-manifest.json",
            "runtime-manifest.json",
            "stdout.log",
            "stderr.log",
            f"{receipt.config}/validation.jsonl",
        ):
            expected[f"{stem}/{relative}"] = "file"
    return expected


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _hash_regular_file(path: Path, *, context: str) -> tuple[int, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise PublicCanaryError(f"{context} is unavailable") from exc
    if _is_link_or_reparse_status(before) or not stat.S_ISREG(before.st_mode):
        raise PublicCanaryError(f"{context} is not a regular non-redirected file")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise PublicCanaryError(f"{context} changed while it was opened")
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as exc:
        raise PublicCanaryError(f"{context} is unavailable") from exc
    identity = _file_identity(before)
    if (
        _file_identity(opened_before) != identity
        or _file_identity(opened_after) != identity
        or _file_identity(after) != identity
        or _is_link_or_reparse_status(after)
    ):
        raise PublicCanaryError(f"{context} changed while it was authenticated")
    return size, digest.hexdigest()


def _authenticate_generator_script() -> None:
    parent = _require_existing_regular_directory(
        GENERATOR_SCRIPT.parent, context="generator script parent"
    )
    candidate = parent / GENERATOR_SCRIPT.name
    if _path_key(candidate) != _path_key(GENERATOR_SCRIPT):
        raise PublicCanaryError("generator script resolved away from its fixed source path")
    _size, digest = _hash_regular_file(candidate, context="generator script")
    if digest != GENERATOR_SCRIPT_SHA256:
        raise PublicCanaryError("generator script whole-file SHA256 drifted")


def _authenticate_public_receipts(items: Mapping[str, _TreeItem]) -> None:
    for receipt in PUBLIC_RECEIPTS:
        size, digest = _hash_regular_file(items[receipt.filename].path, context="public receipt")
        if size != receipt.size_bytes or digest != receipt.sha256:
            raise PublicCanaryError("public receipt bytes differ from the trusted allowlist")


def _require_same_tree(expected: _TreeSnapshot, observed: _TreeSnapshot, *, context: str) -> None:
    if observed.signature() != expected.signature():
        raise PublicCanaryError(f"{context} changed during authentication")


def _validate_canary_inventory(*, raw_root: Path, output_root: Path) -> None:
    initial_output = _scan_regular_tree(output_root, context="canary output root")
    initial_raw = _scan_regular_tree(raw_root, context="canary raw root")
    _reject_nonpublic_names(initial_output.items, context="canary output root")
    _reject_nonpublic_names(initial_raw.items, context="canary raw root")

    expected_outputs = {receipt.filename: "file" for receipt in PUBLIC_RECEIPTS}
    observed_outputs = {name: item.kind for name, item in initial_output.items.items()}
    if observed_outputs != expected_outputs:
        raise PublicCanaryError("canary output inventory drifted")
    observed_raw = {name: item.kind for name, item in initial_raw.items.items()}
    if observed_raw != _expected_raw_inventory():
        raise PublicCanaryError("canary raw sibling inventory drifted")

    _authenticate_public_receipts(initial_output.items)
    middle_output = _scan_regular_tree(output_root, context="canary output root")
    middle_raw = _scan_regular_tree(raw_root, context="canary raw root")
    _require_same_tree(initial_output, middle_output, context="canary output root")
    _require_same_tree(initial_raw, middle_raw, context="canary raw root")

    _authenticate_public_receipts(middle_output.items)
    final_output = _scan_regular_tree(output_root, context="canary output root")
    final_raw = _scan_regular_tree(raw_root, context="canary raw root")
    _require_same_tree(middle_output, final_output, context="canary output root")
    _require_same_tree(middle_raw, final_raw, context="canary raw root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--canary-raw-dir", type=Path, required=True)
    parser.add_argument("--canary-output-dir", type=Path, required=True)
    parser.add_argument("--official-raw-dir", type=Path, required=True)
    parser.add_argument("--official-output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser


def _official_custody_violations(
    roots: Mapping[str, _PreparedAbsentRoot],
) -> tuple[str, ...]:
    violations: list[str] = []
    for context, root in roots.items():
        try:
            _require_root_still_absent(root, context=context)
        except PublicCanaryError as exc:
            violations.append(f"{context}: {exc}")
    return tuple(violations)


def _raise_attempt_failure(
    primary: BaseException | None, custody_violations: Sequence[str]
) -> None:
    if primary is None and not custody_violations:
        return
    custody_message = "; ".join(custody_violations)
    if primary is None:
        raise PublicCanaryError(f"official-root custody violation: {custody_message}")
    if custody_violations:
        primary_message = (
            str(primary)
            if isinstance(primary, PublicCanaryError)
            else f"{type(primary).__name__}: {primary}"
        )
        raise PublicCanaryError(
            "canary attempt failed and official-root custody also failed; "
            f"primary={primary_message}; custody={custody_message}"
        ) from primary
    raise primary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds < 1:
        raise PublicCanaryError("timeout must be positive")
    _validate_trusted_allowlist()
    _authenticate_generator_script()
    roots = _prepare_absent_roots(
        {
            "canary raw root": args.canary_raw_dir,
            "canary output root": args.canary_output_dir,
            "official raw root": args.official_raw_dir,
            "official output root": args.official_output_dir,
        },
        protected_paths={
            "generator repository": GENERATOR_SCRIPT.parent.parent,
            "RULER source root": args.ruler_root,
            "Git executable": args.git_executable,
            "Python executable": args.python,
            "tokenizer root": args.tokenizer_dir,
            "NLTK data root": args.nltk_data,
        },
    )
    canary_roots = {
        "canary raw root": roots["canary raw root"],
        "canary output root": roots["canary output root"],
    }
    canary_raw = canary_roots["canary raw root"].path
    canary_output = canary_roots["canary output root"].path
    official_roots = {
        "official raw root": roots["official raw root"],
        "official output root": roots["official output root"],
    }
    command = _generator_command(args, raw_root=canary_raw, output_root=canary_output)
    primary_failure: BaseException | None = None
    try:
        _reserve_canary_roots(canary_roots)
        try:
            result = subprocess.run(
                command,
                cwd=GENERATOR_SCRIPT.parent.parent,
                env=_subprocess_environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
            )
        except OSError as exc:
            raise PublicCanaryError("generator subprocess could not be launched") from exc
        _validate_generator_result(result)
        _validate_canary_inventory(raw_root=canary_raw, output_root=canary_output)
    except BaseException as exc:
        primary_failure = exc
    custody_violations = _official_custody_violations(official_roots)
    _raise_attempt_failure(primary_failure, custody_violations)

    success = {
        "schema": SUCCESS_SCHEMA,
        "receipt_count": TRUSTED_PUBLIC_RECEIPT_COUNT,
        "aggregate_sha256": TRUSTED_PUBLIC_AGGREGATE_SHA256,
    }
    print(_canonical_json_bytes(success).decode("utf-8"), end="", flush=True)
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except PublicCanaryError as exc:
        print(f"RULER public canary failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
