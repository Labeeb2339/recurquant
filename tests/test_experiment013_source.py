from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from recurquant.experiment013_source import (
    EXPERIMENT013_SOURCE_PATHS,
    Experiment013SourceError,
    canonical_experiment013_source_manifest_bytes,
    canonical_experiment013_source_manifest_sha256,
    capture_experiment013_source_manifest,
    validate_experiment013_source_manifest,
    verify_experiment013_source_manifest,
    verify_loaded_experiment013_recurquant_modules,
)


def _git(root: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    environment = os.environ.copy()
    if env:
        environment.update(env)
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode != 0:
        raise AssertionError(process.stderr or process.stdout)
    return process.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _init_repository(root: Path, *, marker: str = "fixture") -> Path:
    root.mkdir(parents=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Experiment 013 Test")
    _git(root, "config", "user.email", "experiment013@example.invalid")
    _write(root / ".gitattributes", "* text eol=lf\n")
    _write(root / ".gitignore", "artifacts/\n")
    for relative in EXPERIMENT013_SOURCE_PATHS:
        _write(root / relative, f"{marker}:{relative}\n")
    _git(root, "add", "--", ".")
    _git(root, "commit", "-m", "fixture source")
    return root


def _rehash(manifest: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(manifest)
    payload.pop("canonical_manifest_sha256")
    payload["canonical_manifest_sha256"] = canonical_experiment013_source_manifest_sha256(payload)
    return payload


def test_frozen_inventory_covers_all_experiment013_surfaces_without_hash_constants() -> None:
    required = {
        "research/EXPERIMENT_013_STATIC_RHT_Q468_PROTOCOL.md",
        "research/experiment013-parquet-materializations.json",
        "requirements/experiment013-calibration.txt",
        "scripts/capture_static_q468_identity_input.py",
        "scripts/generate_static_q468_ruler_receipts.py",
        "scripts/launch_static_q468_calibration.py",
        "scripts/resolve_static_q468_identity.py",
        "scripts/run_static_q468_calibration.py",
        "requirements/experiment013-ruler.txt",
        "src/recurquant/static_q468.py",
        "src/recurquant/static_q468_cache.py",
        "src/recurquant/static_q468_calibration.py",
        "src/recurquant/experiment013_calibration_api.py",
        "src/recurquant/experiment013_parquet.py",
        "src/recurquant/experiment013_qwen35_adapter.py",
        "src/recurquant/experiment013_source.py",
        "tests/test_capture_static_q468_identity_input.py",
        "tests/test_generate_static_q468_ruler_receipts.py",
        "tests/test_launch_static_q468_calibration.py",
        "tests/test_resolve_static_q468_identity.py",
        "tests/test_run_static_q468_calibration.py",
        "tests/test_static_q468.py",
        "tests/test_static_q468_cache.py",
        "tests/test_static_q468_calibration.py",
        "tests/test_experiment013_calibration_api.py",
        "tests/test_experiment013_parquet.py",
        "tests/test_experiment013_qwen35_adapter.py",
        "tests/test_experiment013_source.py",
    }

    assert required <= set(EXPERIMENT013_SOURCE_PATHS)
    assert tuple(sorted(EXPERIMENT013_SOURCE_PATHS)) == EXPERIMENT013_SOURCE_PATHS
    assert len(set(EXPERIMENT013_SOURCE_PATHS)) == len(EXPERIMENT013_SOURCE_PATHS)
    source = Path(__file__).resolve().parents[1] / "src" / "recurquant" / "experiment013_source.py"
    assert 'canonical_manifest_sha256 = "' not in source.read_text(encoding="utf-8")


def test_capture_is_portable_complete_and_allows_ignored_artifacts(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    _write(root / "artifacts" / "ignored-result.json", "{}\n")

    manifest = capture_experiment013_source_manifest(root)

    assert validate_experiment013_source_manifest(manifest) == manifest
    assert manifest["source_commit"] == _git(root, "rev-parse", "HEAD")
    assert [entry["path"] for entry in manifest["paths"]] == list(  # type: ignore[index]
        EXPERIMENT013_SOURCE_PATHS
    )
    for entry in manifest["paths"]:  # type: ignore[assignment]
        assert entry["git_blob_oid"] == entry["index_blob_oid"]
        assert entry["git_blob_oid"] == entry["worktree_blob_oid"]
        assert len(entry["raw_sha256"]) == 64
    serialized = json.dumps(manifest, sort_keys=True)
    assert str(root) not in serialized
    assert str(root.resolve()) not in serialized
    assert manifest["repository_binding"]["ignored_artifacts_permitted"] is True  # type: ignore[index]

    payload = canonical_experiment013_source_manifest_bytes(manifest)
    assert payload.endswith(b"\n")
    assert json.loads(payload) == manifest
    assert payload == canonical_experiment013_source_manifest_bytes(manifest)


@pytest.mark.parametrize("state", ["staged", "unstaged", "untracked"])
def test_capture_rejects_non_clean_nonignored_state(tmp_path: Path, state: str) -> None:
    root = _init_repository(tmp_path / "repository")
    target = root / EXPERIMENT013_SOURCE_PATHS[0]
    if state == "untracked":
        _write(root / "unexpected.txt", "not ignored\n")
    else:
        target.write_text("changed\n", encoding="utf-8", newline="\n")
        if state == "staged":
            _git(root, "add", "--", EXPERIMENT013_SOURCE_PATHS[0])

    with pytest.raises(Experiment013SourceError, match="changes|untracked|status"):
        capture_experiment013_source_manifest(root)


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_capture_rejects_hidden_index_flags(tmp_path: Path, flag: str) -> None:
    root = _init_repository(tmp_path / "repository")
    _git(root, "update-index", flag, "--", EXPERIMENT013_SOURCE_PATHS[0])

    with pytest.raises(Experiment013SourceError, match="skip-worktree|assume-unchanged"):
        capture_experiment013_source_manifest(root)


def test_capture_scrubs_inherited_git_index_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _init_repository(tmp_path / "repository")
    decoy = tmp_path / "decoy-index"
    decoy.write_bytes(b"not a Git index")
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "elsewhere"))

    manifest = capture_experiment013_source_manifest(root)

    assert manifest["source_commit"] == _git(root, "rev-parse", "HEAD")


def test_capture_rejects_unsafe_local_git_config(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    _git(root, "config", "core.fsmonitor", "true")

    with pytest.raises(Experiment013SourceError, match="unsafe local Git config"):
        capture_experiment013_source_manifest(root)


def test_validation_rejects_non_string_field_names(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    malformed = dict(manifest)
    malformed[1] = "not a field name"

    with pytest.raises(Experiment013SourceError, match="field names must be strings"):
        validate_experiment013_source_manifest(malformed)


def test_linked_worktree_gitdir_common_store_and_index_are_authenticated(tmp_path: Path) -> None:
    primary = _init_repository(tmp_path / "primary")
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-b", "linked-source", str(linked))

    manifest = capture_experiment013_source_manifest(linked)

    assert manifest["repository_binding"]["worktree_layout"] == "linked"  # type: ignore[index]
    assert verify_experiment013_source_manifest(manifest, linked) == manifest


def test_verify_accepts_unchanged_inventory_on_descendant_commit(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    _write(root / "notes.md", "unrelated tracked descendant\n")
    _git(root, "add", "--", "notes.md")
    _git(root, "commit", "-m", "unrelated descendant")

    verified = verify_experiment013_source_manifest(manifest, root)

    assert verified == manifest
    assert verified["source_commit"] != _git(root, "rev-parse", "HEAD")


def test_verify_rejects_changed_inventory_even_on_descendant_commit(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    target = EXPERIMENT013_SOURCE_PATHS[0]
    _write(root / target, "changed in descendant\n")
    _git(root, "add", "--", target)
    _git(root, "commit", "-m", "change frozen source")

    with pytest.raises(Experiment013SourceError, match="differ|drift"):
        verify_experiment013_source_manifest(manifest, root)


@pytest.mark.parametrize(
    "protected_path",
    ["src/recurquant/experiment013_source.py", "tests/test_experiment013_source.py"],
)
def test_verify_rejects_descendant_changes_to_verifier_or_its_tests(
    tmp_path: Path, protected_path: str
) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    _write(root / protected_path, "replaced verifier surface\n")
    _git(root, "add", "--", protected_path)
    _git(root, "commit", "-m", "replace source verifier surface")

    with pytest.raises(Experiment013SourceError, match="differ|drift"):
        verify_experiment013_source_manifest(manifest, root)


def test_verify_rejects_unavailable_or_unrelated_source_commit(tmp_path: Path) -> None:
    first = _init_repository(tmp_path / "first", marker="first")
    second = _init_repository(tmp_path / "second", marker="second")
    manifest = capture_experiment013_source_manifest(first)

    with pytest.raises(Experiment013SourceError, match="unavailable|ancestor"):
        verify_experiment013_source_manifest(manifest, second)


def test_validation_rejects_absolute_path_even_with_recomputed_manifest_hash(
    tmp_path: Path,
) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    manifest["paths"][0]["path"] = "C:/private/source.py"  # type: ignore[index]
    tampered = _rehash(manifest)

    with pytest.raises(Experiment013SourceError, match="repository-relative|inventory"):
        validate_experiment013_source_manifest(tampered)


def test_verify_rejects_rehashed_raw_content_identity_tampering(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    manifest["paths"][0]["raw_sha256"] = "0" * 64  # type: ignore[index]
    tampered = _rehash(manifest)
    assert validate_experiment013_source_manifest(tampered) == tampered

    with pytest.raises(Experiment013SourceError, match="differs"):
        verify_experiment013_source_manifest(tampered, root)


def test_validation_rejects_unknown_fields_and_canonical_hash_drift(tmp_path: Path) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    extra = {**manifest, "absolute_root": str(root)}
    with pytest.raises(Experiment013SourceError, match="fields drifted"):
        validate_experiment013_source_manifest(extra)

    drifted = copy.deepcopy(manifest)
    drifted["canonical_manifest_sha256"] = "0" * 64
    with pytest.raises(Experiment013SourceError, match="canonical source manifest"):
        validate_experiment013_source_manifest(drifted)


def test_loaded_module_helper_binds_canonical_local_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    module_name = "recurquant.static_q468"
    module = types.ModuleType(module_name)
    module.__file__ = str(root / "src" / "recurquant" / "static_q468.py")
    monkeypatch.setitem(sys.modules, module_name, module)

    observed = verify_loaded_experiment013_recurquant_modules(manifest, root, [module_name])

    assert observed == {module_name: "src/recurquant/static_q468.py"}


def test_loaded_module_helper_rejects_outside_or_undeclared_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _init_repository(tmp_path / "repository")
    manifest = capture_experiment013_source_manifest(root)
    outside = tmp_path / "outside.py"
    _write(outside, "pass\n")
    module_name = "recurquant.static_q468"
    module = types.ModuleType(module_name)
    module.__file__ = str(outside)
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(Experiment013SourceError, match="outside"):
        verify_loaded_experiment013_recurquant_modules(manifest, root, [module_name])
