"""Fail-closed local-source identity for Experiment 013.

The manifest produced here is portable: it contains only repository-relative
paths and Git/content identities, including the canonical Git executable's
digest and size.  Absolute worktree, Git-directory, index, object-store, and
executable paths are authenticated locally but never serialized.

The verifier and its tests are part of the frozen inventory.  This does not
create a hash cycle: their committed bytes do not embed the resulting manifest
or its canonical hash.  No expected source-manifest digest is hard-coded here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXPERIMENT013_SOURCE_MANIFEST_SCHEMA = "recurquant.experiment013.source-manifest.v2"
EXPERIMENT013_SOURCE_MANIFEST_PROFILE = "experiment-013-static-q468-frozen-source-v2"
EXPERIMENT013_REPOSITORY_BINDING_SCHEMA = "recurquant.experiment013.repository-binding.v2"

# Keep this explicit.  Discovery by glob would silently change the experiment
# when an unrelated file was added or removed.
EXPERIMENT013_SOURCE_PATHS: tuple[str, ...] = tuple(
    sorted(
        {
            "pyproject.toml",
            "requirements/experiment013-calibration.txt",
            "requirements/experiment013-ruler.txt",
            "research/EXPERIMENT_013_STATIC_RHT_Q468_PROTOCOL.md",
            "research/experiment013-parquet-materializations.json",
            "scripts/capture_static_q468_identity_input.py",
            "scripts/generate_static_q468_ruler_receipts.py",
            "scripts/launch_static_q468_calibration.py",
            "scripts/launch_static_q468_stage_a.py",
            "scripts/resolve_static_q468_identity.py",
            "scripts/run_static_q468_calibration.py",
            "scripts/screen_static_q468_stage_a.py",
            "src/recurquant/cache.py",
            "src/recurquant/evidence.py",
            "src/recurquant/experiment013_calibration_api.py",
            "src/recurquant/experiment013_parquet.py",
            "src/recurquant/experiment013_qwen35_adapter.py",
            "src/recurquant/experiment013_source.py",
            "src/recurquant/experiment013_stage_a.py",
            "src/recurquant/metrics.py",
            "src/recurquant/mixed_quantization.py",
            "src/recurquant/multibit_policy.py",
            "src/recurquant/multibit_quantization.py",
            "src/recurquant/packed_cache.py",
            "src/recurquant/quantization.py",
            "src/recurquant/qwen35.py",
            "src/recurquant/rht.py",
            "src/recurquant/row_policy.py",
            "src/recurquant/statelease.py",
            "src/recurquant/statelease_baselines.py",
            "src/recurquant/statelease_cache.py",
            "src/recurquant/statelease_equal_byte_baselines.py",
            "src/recurquant/statelease_equal_byte_cache.py",
            "src/recurquant/statelease_observer.py",
            "src/recurquant/static_q468.py",
            "src/recurquant/static_q468_artifact_contract.py",
            "src/recurquant/static_q468_cache.py",
            "src/recurquant/static_q468_calibration.py",
            "tests/test_capture_static_q468_identity_input.py",
            "tests/test_experiment013_calibration_api.py",
            "tests/test_experiment013_parquet.py",
            "tests/test_experiment013_qwen35_adapter.py",
            "tests/test_experiment013_source.py",
            "tests/test_experiment013_stage_a.py",
            "tests/test_generate_static_q468_ruler_receipts.py",
            "tests/test_launch_static_q468_calibration.py",
            "tests/test_launch_static_q468_stage_a.py",
            "tests/test_multibit_policy.py",
            "tests/test_resolve_static_q468_identity.py",
            "tests/test_run_static_q468_calibration.py",
            "tests/test_screen_static_q468_stage_a.py",
            "tests/test_static_q468.py",
            "tests/test_static_q468_cache.py",
            "tests/test_static_q468_calibration.py",
        }
    )
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "profile",
        "object_format",
        "source_commit",
        "git_executable",
        "repository_binding",
        "paths",
        "canonical_manifest_sha256",
    }
)
_GIT_EXECUTABLE_FIELDS = frozenset({"sha256", "size_bytes"})
_PATH_FIELDS = frozenset(
    {
        "path",
        "mode",
        "git_blob_oid",
        "index_blob_oid",
        "worktree_blob_oid",
        "raw_sha256",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "schema",
        "worktree_layout",
        "top_level_matches_requested_root",
        "git_directory_bound",
        "common_object_store_bound",
        "index_bound_to_worktree_git_directory",
        "linked_worktree_pointers_verified",
        "hidden_index_flags_absent",
        "tracked_staged_changes_absent",
        "tracked_unstaged_changes_absent",
        "untracked_nonignored_paths_absent",
        "ignored_artifacts_permitted",
        "unsafe_local_config_absent",
        "shallow_history_absent",
        "alternate_object_stores_absent",
        "replacement_refs_absent",
        "history_grafts_absent",
        "inherited_git_environment_scrubbed",
        "system_and_global_git_config_disabled",
        "authenticated_git_executable_bound",
        "git_executable_regular_non_reparse",
        "path_lookup_not_used_after_git_resolution",
    }
)
_TRUE_BINDING_FIELDS = _BINDING_FIELDS - {"schema", "worktree_layout"}
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_GIT_MODES = frozenset({"100644", "100755"})
_WINDOWS_REPARSE_POINT = 0x400
_FORBIDDEN_LOCAL_CONFIG_KEYS = frozenset(
    {
        "core.alternaterefscommand",
        "core.alternaterefsprefixes",
        "core.attributesfile",
        "core.checkstat",
        "core.excludesfile",
        "core.fsmonitor",
        "core.hookspath",
        "core.ignorestat",
        "core.sparsecheckout",
        "core.sparsecheckoutcone",
        "core.splitindex",
        "core.trustctime",
        "core.untrackedcache",
        "core.worktree",
        "extensions.partialclone",
        "extensions.worktreeconfig",
        "index.sparse",
    }
)


class Experiment013SourceError(RuntimeError):
    """Raised when the Experiment 013 source identity cannot be authenticated."""


@dataclass(frozen=True)
class _RepositoryIdentity:
    root: Path
    git_dir: Path
    common_dir: Path
    index_path: Path
    object_dir: Path
    public_binding: dict[str, object]


@dataclass(frozen=True)
class GitExecutableIdentity:
    path: Path
    size_bytes: int
    sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Experiment013SourceError("source manifest is not canonical JSON data") from error


def canonical_experiment013_source_manifest_sha256(
    payload_without_hash: Mapping[str, object],
) -> str:
    """Hash a canonical manifest payload that does not contain its own hash."""

    if "canonical_manifest_sha256" in payload_without_hash:
        raise Experiment013SourceError(
            "canonical payload must not contain canonical_manifest_sha256"
        )
    return hashlib.sha256(_canonical_json_bytes(dict(payload_without_hash))).hexdigest()


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if any(not isinstance(field, str) for field in value):
        raise Experiment013SourceError(f"{name} field names must be strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Experiment013SourceError(f"{name} fields drifted (missing={missing}, extra={extra})")


def _sha1(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise Experiment013SourceError(f"{name} must be a lowercase SHA-1 object ID")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment013SourceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Experiment013SourceError(f"{name} must be a positive integer")
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError as error:
        raise Experiment013SourceError(f"required path is unavailable: {path}") from error
    return path.is_symlink() or bool(
        getattr(status, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _assert_no_link_or_reparse_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            raise Experiment013SourceError(
                "authenticated Git executable traverses a link or reparse point"
            )


def authenticate_git_executable(
    executable: str | os.PathLike[str] | None = None,
) -> GitExecutableIdentity:
    """Resolve and hash one canonical Git binary before any Git subprocess runs."""

    selected: str | os.PathLike[str]
    if executable is None:
        discovered = shutil.which("git")
        if discovered is None:
            raise Experiment013SourceError("Git executable is unavailable")
        selected = discovered
    else:
        selected = executable
    try:
        resolved = Path(selected).resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise Experiment013SourceError("Git executable path is unavailable") from error
    if not resolved.is_absolute():
        raise Experiment013SourceError("Git executable path is not absolute")
    if resolved.name.casefold() == "git.exe" and resolved.parent.name.casefold() == "cmd":
        implementation = resolved.parent.parent / "mingw64" / "bin" / "git.exe"
        try:
            resolved = implementation.resolve(strict=True)
        except OSError as error:
            raise Experiment013SourceError(
                "Git-for-Windows cmd shim has no canonical mingw64 executable"
            ) from error
    _assert_no_link_or_reparse_components(resolved)
    try:
        before = resolved.stat()
        data = resolved.read_bytes()
        after = resolved.stat()
    except OSError as error:
        raise Experiment013SourceError("Git executable cannot be authenticated") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_size <= 0
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or len(data) != after.st_size
        or _is_link_or_reparse(resolved)
    ):
        raise Experiment013SourceError("Git executable changed or is not a regular file")
    return GitExecutableIdentity(
        path=resolved,
        size_bytes=after.st_size,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _git_executable_record(identity: GitExecutableIdentity) -> dict[str, object]:
    return {"sha256": identity.sha256, "size_bytes": identity.size_bytes}


def _canonical_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Experiment013SourceError(f"{name} must be a non-empty canonical path")
    if "\\" in value or "\0" in value or "\n" in value or "\r" in value:
        raise Experiment013SourceError(f"{name} must use a single-line POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Experiment013SourceError(f"{name} must be repository-relative")
    if path.as_posix() != value:
        raise Experiment013SourceError(f"{name} is not a canonical POSIX path")
    return value


def validate_experiment013_source_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Strictly normalize a source manifest without consulting Git or the filesystem."""

    if not isinstance(manifest, Mapping):
        raise Experiment013SourceError("source manifest must be a mapping")
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, name="source manifest")
    if manifest["schema"] != EXPERIMENT013_SOURCE_MANIFEST_SCHEMA:
        raise Experiment013SourceError("source manifest schema drifted")
    if manifest["profile"] != EXPERIMENT013_SOURCE_MANIFEST_PROFILE:
        raise Experiment013SourceError("source manifest profile drifted")
    if manifest["object_format"] != "sha1":
        raise Experiment013SourceError("source manifest requires the frozen SHA-1 format")
    source_commit = _sha1(manifest["source_commit"], name="source_commit")

    raw_git_executable = manifest["git_executable"]
    if not isinstance(raw_git_executable, Mapping):
        raise Experiment013SourceError("git_executable must be a mapping")
    _exact_fields(
        raw_git_executable,
        _GIT_EXECUTABLE_FIELDS,
        name="git_executable",
    )
    git_executable = {
        "sha256": _sha256(
            raw_git_executable["sha256"],
            name="git_executable.sha256",
        ),
        "size_bytes": _positive_int(
            raw_git_executable["size_bytes"],
            name="git_executable.size_bytes",
        ),
    }

    raw_binding = manifest["repository_binding"]
    if not isinstance(raw_binding, Mapping):
        raise Experiment013SourceError("repository_binding must be a mapping")
    _exact_fields(raw_binding, _BINDING_FIELDS, name="repository_binding")
    if raw_binding["schema"] != EXPERIMENT013_REPOSITORY_BINDING_SCHEMA:
        raise Experiment013SourceError("repository binding schema drifted")
    if raw_binding["worktree_layout"] not in {"primary", "linked"}:
        raise Experiment013SourceError("repository worktree_layout is invalid")
    for field in _TRUE_BINDING_FIELDS:
        if raw_binding[field] is not True:
            raise Experiment013SourceError(f"repository binding check is not true: {field}")
    binding = {field: raw_binding[field] for field in sorted(_BINDING_FIELDS)}

    raw_paths = manifest["paths"]
    if isinstance(raw_paths, (str, bytes, bytearray)) or not isinstance(raw_paths, Sequence):
        raise Experiment013SourceError("source manifest paths must be a sequence")
    if len(raw_paths) != len(EXPERIMENT013_SOURCE_PATHS):
        raise Experiment013SourceError("source manifest path inventory size drifted")
    entries: list[dict[str, str]] = []
    for index, raw_entry in enumerate(raw_paths):
        if not isinstance(raw_entry, Mapping):
            raise Experiment013SourceError(f"paths[{index}] must be a mapping")
        _exact_fields(raw_entry, _PATH_FIELDS, name=f"paths[{index}]")
        path = _canonical_relative_path(raw_entry["path"], name=f"paths[{index}].path")
        if path != EXPERIMENT013_SOURCE_PATHS[index]:
            raise Experiment013SourceError(
                f"source path inventory drifted at index {index}: {path}"
            )
        mode = raw_entry["mode"]
        if not isinstance(mode, str) or mode not in _SAFE_GIT_MODES:
            raise Experiment013SourceError(f"paths[{index}].mode is not a regular-file mode")
        git_blob_oid = _sha1(raw_entry["git_blob_oid"], name=f"paths[{index}].git_blob_oid")
        index_blob_oid = _sha1(raw_entry["index_blob_oid"], name=f"paths[{index}].index_blob_oid")
        worktree_blob_oid = _sha1(
            raw_entry["worktree_blob_oid"], name=f"paths[{index}].worktree_blob_oid"
        )
        if len({git_blob_oid, index_blob_oid, worktree_blob_oid}) != 1:
            raise Experiment013SourceError(
                f"paths[{index}] Git/index/worktree blob identities differ"
            )
        entries.append(
            {
                "path": path,
                "mode": mode,
                "git_blob_oid": git_blob_oid,
                "index_blob_oid": index_blob_oid,
                "worktree_blob_oid": worktree_blob_oid,
                "raw_sha256": _sha256(raw_entry["raw_sha256"], name=f"paths[{index}].raw_sha256"),
            }
        )

    normalized_without_hash: dict[str, object] = {
        "schema": EXPERIMENT013_SOURCE_MANIFEST_SCHEMA,
        "profile": EXPERIMENT013_SOURCE_MANIFEST_PROFILE,
        "object_format": "sha1",
        "source_commit": source_commit,
        "git_executable": git_executable,
        "repository_binding": binding,
        "paths": entries,
    }
    recorded_hash = _sha256(manifest["canonical_manifest_sha256"], name="canonical_manifest_sha256")
    computed_hash = canonical_experiment013_source_manifest_sha256(normalized_without_hash)
    if recorded_hash != computed_hash:
        raise Experiment013SourceError("canonical source manifest SHA-256 drifted")
    return {**normalized_without_hash, "canonical_manifest_sha256": recorded_hash}


def _sanitized_git_environment() -> dict[str, str]:
    inherited = {key.upper(): (key, value) for key, value in os.environ.items()}
    environment = {
        inherited[name][0]: inherited[name][1]
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC")
        if name in inherited
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "core.untrackedCache",
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_CONFIG_KEY_2": "core.hooksPath",
            "GIT_CONFIG_VALUE_2": os.devnull,
            "GIT_AUTHOR_NAME": "RecurQuant Experiment 013",
            "GIT_AUTHOR_EMAIL": "experiment013@invalid",
            "GIT_COMMITTER_NAME": "RecurQuant Experiment 013",
            "GIT_COMMITTER_EMAIL": "experiment013@invalid",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _run_git(
    git: GitExecutableIdentity,
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [str(git.path), *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            input=input_bytes,
            env=_sanitized_git_environment(),
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Experiment013SourceError(f"git {arguments[0]} could not be executed") from error


def _git_bytes(
    git: GitExecutableIdentity,
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    process = _run_git(git, root, *arguments, input_bytes=input_bytes)
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise Experiment013SourceError(
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    return process.stdout


def _git_text(git: GitExecutableIdentity, root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(git, root, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise Experiment013SourceError("Git returned non-UTF-8 identity data") from error


def _resolved_git_path(root: Path, value: str, *, must_exist: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=must_exist)
    except OSError as error:
        raise Experiment013SourceError("authenticated Git path is unavailable") from error


def _path_has_symlink_component(root: Path, relative: str) -> bool:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _assert_safe_local_config(git: GitExecutableIdentity, root: Path) -> None:
    raw = _git_bytes(git, root, "config", "--local", "--no-includes", "--null", "--list")
    for record in (item for item in raw.split(b"\0") if item):
        key_bytes, separator, value_bytes = record.partition(b"\n")
        if not separator or not key_bytes:
            raise Experiment013SourceError("local Git config contains a malformed entry")
        try:
            key = key_bytes.decode("utf-8").lower()
            value = value_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Experiment013SourceError("local Git config is not UTF-8") from error
        forbidden = (
            key in _FORBIDDEN_LOCAL_CONFIG_KEYS
            or key.startswith(("include.", "includeif.", "filter."))
            or (key.startswith("remote.") and key.endswith((".promisor", ".partialclonefilter")))
        )
        if forbidden:
            raise Experiment013SourceError(f"unsafe local Git config key is present: {key}")
        if key == "core.usereplacerefs" and value.lower() not in {"0", "false", "no", "off"}:
            raise Experiment013SourceError("local Git config enables replacement objects")


def _assert_no_hidden_index_flags(git: GitExecutableIdentity, root: Path) -> None:
    raw = _git_bytes(git, root, "ls-files", "--cached", "-v", "-z")
    records = [record for record in raw.split(b"\0") if record]
    malformed = [record for record in records if len(record) < 3 or record[1:2] != b" "]
    if malformed:
        raise Experiment013SourceError("Git index visibility output is malformed")
    unsafe = sorted({chr(record[0]) for record in records if record[:1] != b"H"})
    if unsafe:
        raise Experiment013SourceError(
            "Git index contains skip-worktree, assume-unchanged, or non-stage-zero entries "
            f"(tags={unsafe})"
        )


def _assert_no_tracked_or_untracked_changes(git: GitExecutableIdentity, root: Path) -> None:
    checks = (
        ("tracked staged", ("diff", "--cached", "--no-ext-diff", "--quiet", "HEAD", "--")),
        ("tracked unstaged", ("diff", "--no-ext-diff", "--quiet", "--")),
    )
    for label, arguments in checks:
        process = _run_git(git, root, *arguments)
        if process.returncode == 1:
            raise Experiment013SourceError(f"repository has {label} changes")
        if process.returncode != 0:
            raise Experiment013SourceError(f"cannot authenticate {label} changes")
    status = _git_bytes(git, root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        raise Experiment013SourceError(
            "repository contains non-ignored untracked paths or tracked status drift"
        )


def _authenticate_repository(
    repo_root: str | Path,
    git: GitExecutableIdentity,
) -> _RepositoryIdentity:
    try:
        root = Path(repo_root).resolve(strict=True)
    except OSError as error:
        raise Experiment013SourceError("repository root is unavailable") from error
    if not root.is_dir():
        raise Experiment013SourceError("repository root must be a directory")

    top_level = _resolved_git_path(root, _git_text(git, root, "rev-parse", "--show-toplevel"))
    git_dir = _resolved_git_path(root, _git_text(git, root, "rev-parse", "--absolute-git-dir"))
    common_dir = _resolved_git_path(root, _git_text(git, root, "rev-parse", "--git-common-dir"))
    object_dir = _resolved_git_path(
        root, _git_text(git, root, "rev-parse", "--git-path", "objects")
    )
    index_path = _resolved_git_path(root, _git_text(git, root, "rev-parse", "--git-path", "index"))
    if top_level != root:
        raise Experiment013SourceError("Git top-level does not equal the requested repository root")
    if _git_text(git, root, "rev-parse", "--is-inside-work-tree") != "true":
        raise Experiment013SourceError("repository root is not inside a Git worktree")
    if _git_text(git, root, "rev-parse", "--is-bare-repository") != "false":
        raise Experiment013SourceError("bare Git repositories are forbidden")
    if _git_text(git, root, "rev-parse", "--show-object-format") != "sha1":
        raise Experiment013SourceError("Experiment 013 requires the SHA-1 Git object format")
    if _git_text(git, root, "rev-parse", "--is-shallow-repository") != "false":
        raise Experiment013SourceError("shallow Git history is forbidden")
    if not git_dir.is_dir() or not common_dir.is_dir() or not index_path.is_file():
        raise Experiment013SourceError("Git directory or index identity is malformed")
    if index_path != (git_dir / "index").resolve(strict=True):
        raise Experiment013SourceError("Git index is not bound to the worktree Git directory")
    if object_dir != (common_dir / "objects").resolve(strict=True) or not object_dir.is_dir():
        raise Experiment013SourceError("Git object directory is not the common object store")

    dot_git = root / ".git"
    if dot_git.is_symlink():
        raise Experiment013SourceError("repository .git identity cannot be a symlink")
    if dot_git.is_dir():
        if git_dir != dot_git.resolve(strict=True) or common_dir != git_dir:
            raise Experiment013SourceError("primary-worktree Git directory binding drifted")
        layout = "primary"
    elif dot_git.is_file():
        try:
            marker = dot_git.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise Experiment013SourceError("linked-worktree .git pointer is unreadable") from error
        if not marker.startswith("gitdir: "):
            raise Experiment013SourceError("linked-worktree .git pointer is malformed")
        marker_path = Path(marker.removeprefix("gitdir: "))
        if not marker_path.is_absolute():
            marker_path = root / marker_path
        try:
            marker_git_dir = marker_path.resolve(strict=True)
        except OSError as error:
            raise Experiment013SourceError(
                "linked-worktree Git directory is unavailable"
            ) from error
        if (
            marker_git_dir != git_dir
            or git_dir.parent.name != "worktrees"
            or git_dir.parent.parent != common_dir
        ):
            raise Experiment013SourceError("linked-worktree/common-directory binding drifted")
        reverse_pointer = git_dir / "gitdir"
        if not reverse_pointer.is_file() or reverse_pointer.is_symlink():
            raise Experiment013SourceError("linked-worktree reverse pointer is unavailable")
        try:
            reverse_path = Path(reverse_pointer.read_text(encoding="utf-8").strip())
            if not reverse_path.is_absolute():
                reverse_path = git_dir / reverse_path
            reverse_path = reverse_path.resolve(strict=True)
        except (OSError, UnicodeError) as error:
            raise Experiment013SourceError(
                "linked-worktree reverse pointer is malformed"
            ) from error
        if reverse_path != dot_git.resolve(strict=True):
            raise Experiment013SourceError("linked-worktree reverse pointer drifted")
        layout = "linked"
    else:
        raise Experiment013SourceError("repository has no canonical .git identity")

    for unsafe_path in (
        object_dir / "info" / "alternates",
        object_dir / "info" / "http-alternates",
        common_dir / "info" / "grafts",
        git_dir / "info" / "grafts",
        common_dir / "shallow",
    ):
        if unsafe_path.exists():
            raise Experiment013SourceError("Git alternates, grafts, or shallow metadata exists")
    if _git_text(git, root, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise Experiment013SourceError("Git replacement refs are forbidden")
    _assert_safe_local_config(git, root)
    _assert_no_hidden_index_flags(git, root)
    _assert_no_tracked_or_untracked_changes(git, root)

    binding: dict[str, object] = {
        "schema": EXPERIMENT013_REPOSITORY_BINDING_SCHEMA,
        "worktree_layout": layout,
        "top_level_matches_requested_root": True,
        "git_directory_bound": True,
        "common_object_store_bound": True,
        "index_bound_to_worktree_git_directory": True,
        "linked_worktree_pointers_verified": True,
        "hidden_index_flags_absent": True,
        "tracked_staged_changes_absent": True,
        "tracked_unstaged_changes_absent": True,
        "untracked_nonignored_paths_absent": True,
        "ignored_artifacts_permitted": True,
        "unsafe_local_config_absent": True,
        "shallow_history_absent": True,
        "alternate_object_stores_absent": True,
        "replacement_refs_absent": True,
        "history_grafts_absent": True,
        "inherited_git_environment_scrubbed": True,
        "system_and_global_git_config_disabled": True,
        "authenticated_git_executable_bound": True,
        "git_executable_regular_non_reparse": True,
        "path_lookup_not_used_after_git_resolution": True,
    }
    return _RepositoryIdentity(root, git_dir, common_dir, index_path, object_dir, binding)


def _tree_entries(
    git: GitExecutableIdentity,
    root: Path,
    commit: str,
) -> dict[str, tuple[str, str]]:
    raw = _git_bytes(
        git,
        root,
        "ls-tree",
        "-r",
        "--full-tree",
        "-z",
        commit,
        "--",
        *EXPERIMENT013_SOURCE_PATHS,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in (item for item in raw.split(b"\0") if item):
        metadata, separator, path_bytes = record.partition(b"\t")
        fields = metadata.split(b" ")
        try:
            path = path_bytes.decode("utf-8")
            mode = fields[0].decode("ascii")
            kind = fields[1].decode("ascii")
            oid = fields[2].decode("ascii")
        except (IndexError, UnicodeDecodeError) as error:
            raise Experiment013SourceError("Git tree entry is malformed") from error
        if separator != b"\t" or len(fields) != 3 or kind != "blob" or mode not in _SAFE_GIT_MODES:
            raise Experiment013SourceError("source inventory contains a non-regular Git tree entry")
        if path not in EXPERIMENT013_SOURCE_PATHS or path in entries:
            raise Experiment013SourceError("Git tree returned an unexpected source path")
        entries[path] = (mode, _sha1(oid, name=f"Git tree OID for {path}"))
    if set(entries) != set(EXPERIMENT013_SOURCE_PATHS):
        missing = sorted(set(EXPERIMENT013_SOURCE_PATHS) - set(entries))
        raise Experiment013SourceError(f"source inventory is not tracked by the commit: {missing}")
    return entries


def _index_entries(git: GitExecutableIdentity, root: Path) -> dict[str, tuple[str, str]]:
    raw = _git_bytes(
        git,
        root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        *EXPERIMENT013_SOURCE_PATHS,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in (item for item in raw.split(b"\0") if item):
        metadata, separator, path_bytes = record.partition(b"\t")
        fields = metadata.split(b" ")
        try:
            path = path_bytes.decode("utf-8")
            mode = fields[0].decode("ascii")
            oid = fields[1].decode("ascii")
            stage = fields[2].decode("ascii")
        except (IndexError, UnicodeDecodeError) as error:
            raise Experiment013SourceError("Git index entry is malformed") from error
        if (
            separator != b"\t"
            or len(fields) != 3
            or stage != "0"
            or mode not in _SAFE_GIT_MODES
            or path not in EXPERIMENT013_SOURCE_PATHS
            or path in entries
        ):
            raise Experiment013SourceError("source inventory has a non-canonical index entry")
        entries[path] = (mode, _sha1(oid, name=f"Git index OID for {path}"))
    if set(entries) != set(EXPERIMENT013_SOURCE_PATHS):
        missing = sorted(set(EXPERIMENT013_SOURCE_PATHS) - set(entries))
        raise Experiment013SourceError(f"source inventory is not in the Git index: {missing}")
    return entries


def _worktree_oids(git: GitExecutableIdentity, root: Path) -> dict[str, str]:
    payload = "".join(f"{path}\n" for path in EXPERIMENT013_SOURCE_PATHS).encode("utf-8")
    raw = _git_bytes(
        git,
        root,
        "hash-object",
        "--no-filters",
        "--stdin-paths",
        input_bytes=payload,
    )
    lines = raw.decode("ascii", errors="strict").splitlines()
    if len(lines) != len(EXPERIMENT013_SOURCE_PATHS):
        raise Experiment013SourceError("Git did not hash every source worktree path")
    return {
        path: _sha1(oid, name=f"worktree hash-object OID for {path}")
        for path, oid in zip(EXPERIMENT013_SOURCE_PATHS, lines, strict=True)
    }


def _raw_file_identities(root: Path) -> dict[str, tuple[str, str]]:
    identities: dict[str, tuple[str, str]] = {}
    for relative in EXPERIMENT013_SOURCE_PATHS:
        path = root / PurePosixPath(relative)
        if _path_has_symlink_component(root, relative) or not path.is_file():
            raise Experiment013SourceError(f"source path is not a regular local file: {relative}")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise Experiment013SourceError(f"source path is unreadable: {relative}") from error
        git_header = f"blob {len(raw)}\0".encode("ascii")
        identities[relative] = (
            hashlib.sha1(git_header + raw).hexdigest(),  # noqa: S324 - Git's frozen object ID.
            hashlib.sha256(raw).hexdigest(),
        )
    return identities


def _source_entries(
    git: GitExecutableIdentity,
    root: Path,
    commit: str,
) -> list[dict[str, str]]:
    tree = _tree_entries(git, root, commit)
    index = _index_entries(git, root)
    worktree = _worktree_oids(git, root)
    raw = _raw_file_identities(root)
    entries: list[dict[str, str]] = []
    for relative in EXPERIMENT013_SOURCE_PATHS:
        mode, git_oid = tree[relative]
        index_mode, index_oid = index[relative]
        worktree_oid = worktree[relative]
        raw_git_oid, raw_sha256 = raw[relative]
        if index_mode != mode or len({git_oid, index_oid, worktree_oid, raw_git_oid}) != 1:
            raise Experiment013SourceError(
                f"source bytes/blob/mode differ from commit and index: {relative}"
            )
        entries.append(
            {
                "path": relative,
                "mode": mode,
                "git_blob_oid": git_oid,
                "index_blob_oid": index_oid,
                "worktree_blob_oid": worktree_oid,
                "raw_sha256": raw_sha256,
            }
        )
    return entries


def _head(git: GitExecutableIdentity, root: Path) -> str:
    head = _sha1(_git_text(git, root, "rev-parse", "HEAD"), name="repository HEAD")
    if _git_text(git, root, "cat-file", "-t", head) != "commit":
        raise Experiment013SourceError("repository HEAD is not a commit")
    return head


def capture_experiment013_source_manifest(
    repo_root: str | Path,
    *,
    git_executable: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Capture the exact committed Experiment 013 source identity.

    Non-ignored untracked files and all tracked staged/unstaged changes are
    rejected.  Ignored artifacts do not affect capture.
    """

    git = authenticate_git_executable(git_executable)
    first_repository = _authenticate_repository(repo_root, git)
    first_head = _head(git, first_repository.root)
    first_entries = _source_entries(git, first_repository.root, first_head)

    second_repository = _authenticate_repository(first_repository.root, git)
    second_head = _head(git, second_repository.root)
    second_entries = _source_entries(git, second_repository.root, second_head)
    if (
        first_head != second_head
        or first_entries != second_entries
        or first_repository.public_binding != second_repository.public_binding
    ):
        raise Experiment013SourceError("repository source identity changed during capture")

    payload: dict[str, object] = {
        "schema": EXPERIMENT013_SOURCE_MANIFEST_SCHEMA,
        "profile": EXPERIMENT013_SOURCE_MANIFEST_PROFILE,
        "object_format": "sha1",
        "source_commit": first_head,
        "git_executable": _git_executable_record(git),
        "repository_binding": first_repository.public_binding,
        "paths": first_entries,
    }
    payload["canonical_manifest_sha256"] = canonical_experiment013_source_manifest_sha256(payload)
    return validate_experiment013_source_manifest(payload)


def _assert_ancestor(
    git: GitExecutableIdentity,
    root: Path,
    ancestor: str,
    descendant: str,
) -> None:
    object_type = _run_git(git, root, "cat-file", "-e", f"{ancestor}^{{commit}}")
    if object_type.returncode != 0:
        raise Experiment013SourceError("source commit is unavailable from the local object store")
    process = _run_git(git, root, "merge-base", "--is-ancestor", ancestor, descendant)
    if process.returncode == 1:
        raise Experiment013SourceError("source commit is not an ancestor of current HEAD")
    if process.returncode != 0:
        raise Experiment013SourceError("cannot authenticate source-commit ancestry")


def verify_experiment013_source_manifest(
    manifest: Mapping[str, object],
    repo_root: str | Path,
    *,
    git_executable: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Verify a frozen manifest against the clean source used at point-of-use.

    The source commit may be an ancestor of current ``HEAD``.  Every frozen
    path must nevertheless have byte-identical commit, index, and worktree
    identities, so descendant commits cannot alter experiment code silently.
    """

    normalized = validate_experiment013_source_manifest(manifest)
    source_commit = str(normalized["source_commit"])
    git = authenticate_git_executable(git_executable)
    if _git_executable_record(git) != normalized["git_executable"]:
        raise Experiment013SourceError(
            "authenticated Git executable differs from the frozen source manifest"
        )

    first_repository = _authenticate_repository(repo_root, git)
    first_head = _head(git, first_repository.root)
    _assert_ancestor(git, first_repository.root, source_commit, first_head)
    first_entries = _source_entries(git, first_repository.root, source_commit)
    if first_entries != normalized["paths"]:
        raise Experiment013SourceError("live Experiment 013 source differs from its manifest")

    second_repository = _authenticate_repository(first_repository.root, git)
    second_head = _head(git, second_repository.root)
    _assert_ancestor(git, second_repository.root, source_commit, second_head)
    second_entries = _source_entries(git, second_repository.root, source_commit)
    if first_head != second_head or first_entries != second_entries:
        raise Experiment013SourceError("repository source identity changed during verification")
    if second_entries != normalized["paths"]:
        raise Experiment013SourceError("live Experiment 013 source drifted during verification")
    return normalized


def canonical_experiment013_source_manifest_bytes(
    manifest: Mapping[str, object],
) -> bytes:
    """Validate and serialize a source manifest in its canonical file form."""

    return _canonical_json_bytes(validate_experiment013_source_manifest(manifest))


def verify_loaded_experiment013_recurquant_modules(
    manifest: Mapping[str, object],
    repo_root: str | Path,
    required_module_names: Sequence[str],
) -> dict[str, str]:
    """Authenticate selected loaded ``recurquant`` modules against a manifest.

    This helper is intentionally explicit rather than scanning every eagerly
    imported package module.  Callers supply the modules that participate in a
    point-of-use code path after verifying the manifest itself.
    """

    normalized = validate_experiment013_source_manifest(manifest)
    try:
        root = Path(repo_root).resolve(strict=True)
    except OSError as error:
        raise Experiment013SourceError("repository root is unavailable") from error
    entries = {str(item["path"]): item for item in normalized["paths"]}  # type: ignore[index]
    observed: dict[str, str] = {}
    for module_name in required_module_names:
        if (
            not isinstance(module_name, str)
            or not module_name.startswith("recurquant.")
            or module_name.endswith(".")
        ):
            raise Experiment013SourceError("required module name is not canonical")
        module = sys.modules.get(module_name)
        if module is None:
            raise Experiment013SourceError(
                f"required RecurQuant module is not loaded: {module_name}"
            )
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, (str, os.PathLike)):
            raise Experiment013SourceError(f"loaded module has no source file: {module_name}")
        declared = Path(raw_file)
        if declared.is_symlink():
            raise Experiment013SourceError(f"loaded module source is a symlink: {module_name}")
        try:
            resolved = declared.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise Experiment013SourceError(
                f"loaded module is outside the authenticated repository: {module_name}"
            ) from error
        expected = f"src/{module_name.replace('.', '/')}.py"
        if relative != expected or relative not in entries:
            raise Experiment013SourceError(
                f"loaded module is not declared at its canonical source path: {module_name}"
            )
        if _path_has_symlink_component(root, relative):
            raise Experiment013SourceError(f"loaded module path traverses a symlink: {module_name}")
        try:
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError as error:
            raise Experiment013SourceError(
                f"loaded module source is unreadable: {module_name}"
            ) from error
        if digest != entries[relative]["raw_sha256"]:  # type: ignore[index]
            raise Experiment013SourceError(f"loaded module source bytes drifted: {module_name}")
        observed[module_name] = relative
    if len(observed) != len(required_module_names):
        raise Experiment013SourceError("required module names contain duplicates")
    return observed


__all__ = [
    "EXPERIMENT013_REPOSITORY_BINDING_SCHEMA",
    "EXPERIMENT013_SOURCE_MANIFEST_PROFILE",
    "EXPERIMENT013_SOURCE_MANIFEST_SCHEMA",
    "EXPERIMENT013_SOURCE_PATHS",
    "Experiment013SourceError",
    "GitExecutableIdentity",
    "authenticate_git_executable",
    "canonical_experiment013_source_manifest_bytes",
    "canonical_experiment013_source_manifest_sha256",
    "capture_experiment013_source_manifest",
    "validate_experiment013_source_manifest",
    "verify_experiment013_source_manifest",
    "verify_loaded_experiment013_recurquant_modules",
]
