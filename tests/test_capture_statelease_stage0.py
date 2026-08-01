from __future__ import annotations

import importlib.metadata
import copy
import hashlib
import json
import subprocess
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.capture_statelease_stage0 as capture_stage0
import scripts.verify_statelease_stage0 as verify_stage0
from scripts.capture_statelease_stage0 import (
    DEFAULT_ARTIFACT,
    EXPERIMENT012_SOURCE_PROVENANCE_PATHS,
    EXPERIMENT_ID,
    PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256,
    REPO_ROOT,
    SCHEMA_NAME,
    SOURCE_IDENTITY_PATHS,
    _audit_whole_cache_persistent_tensors,
    _build_whole_cache_storage_inventory,
    _finalize_source_identity,
    _runtime_identity,
    build_production_artifact,
    canonical_payload_sha256,
    write_artifact,
)
from scripts.verify_statelease_stage0 import (
    EFFECTIVE_PLAN_SHA256,
    PRODUCTION_SCHEMA,
    STATELEASE_BYTES,
    Stage0VerificationError,
    _expected_whole_cache_storage_audit,
    _verify_runtime_identity,
    _verify_whole_cache_storage_audit,
    verify_production_stage0,
)
from scripts.verify_statelease_stage0 import (
    EXPERIMENT012_SOURCE_PROVENANCE_PATHS as VERIFIER_EXPERIMENT012_SOURCE_PROVENANCE_PATHS,
)
from scripts.verify_statelease_stage0 import (
    EXPERIMENT_ID as VERIFIER_EXPERIMENT_ID,
)
from scripts.verify_statelease_stage0 import (
    SOURCE_IDENTITY_PATHS as VERIFIER_SOURCE_IDENTITY_PATHS,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resign_payload(payload: dict[str, object], path: Path) -> None:
    unhashed = {key: value for key, value in payload.items() if key != "canonical_payload_sha256"}
    payload["canonical_payload_sha256"] = canonical_payload_sha256(unhashed)
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".sha256").write_bytes(
        _file_sha256(path).encode("ascii") + b"  " + path.name.encode("ascii") + b"\n"
    )


@pytest.fixture(scope="module")
def authenticated_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    directory = tmp_path_factory.mktemp("statelease-stage0-production")
    path = directory / "production.pt"
    artifact = build_production_artifact()
    write_artifact(artifact, path)
    yield path


def test_canonical_payload_hash_is_mapping_order_independent_and_tensor_sensitive() -> None:
    first = {
        "alpha": 1,
        "tensor": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    reordered = {
        "tensor": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "alpha": 1,
    }
    changed = {
        "alpha": 1,
        "tensor": torch.tensor([1.0, 3.0], dtype=torch.float32),
    }

    assert canonical_payload_sha256(first) == canonical_payload_sha256(reordered)
    assert canonical_payload_sha256(first) != canonical_payload_sha256(changed)


def test_canonical_payload_hash_distinguishes_list_and_tuple_types() -> None:
    list_payload = {"sequence": [1, 2, 3]}
    tuple_payload = {"sequence": (1, 2, 3)}

    assert canonical_payload_sha256(list_payload) != canonical_payload_sha256(tuple_payload)
    assert verify_stage0.canonical_payload_sha256(list_payload) != (
        verify_stage0.canonical_payload_sha256(tuple_payload)
    )
    assert canonical_payload_sha256(list_payload) == (
        verify_stage0.canonical_payload_sha256(list_payload)
    )
    assert canonical_payload_sha256(tuple_payload) == (
        verify_stage0.canonical_payload_sha256(tuple_payload)
    )


def test_canonical_payload_rejects_schema_subclasses_in_both_implementations() -> None:
    class UnsafeString(str):
        pass

    class UnsafeDict(dict[str, object]):
        pass

    for payload, type_name in (
        ({"unsafe": UnsafeString("looks-like-a-string")}, "UnsafeString"),
        (UnsafeDict({"looks": "like-a-dict"}), "UnsafeDict"),
    ):
        with pytest.raises(TypeError, match=rf"closed schema cannot hash {type_name}"):
            canonical_payload_sha256(payload)
        with pytest.raises(Stage0VerificationError, match=rf"unsupported type {type_name}"):
            verify_stage0.canonical_payload_sha256(payload)


def test_source_identity_matches_verifier_and_covers_the_complete_package() -> None:
    package_sources = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src" / "recurquant").glob("*.py")
    }

    assert SOURCE_IDENTITY_PATHS == VERIFIER_SOURCE_IDENTITY_PATHS
    assert package_sources <= set(SOURCE_IDENTITY_PATHS)
    module_paths = capture_stage0._loaded_recurquant_module_paths()
    required = dict(capture_stage0.REQUIRED_LOADED_RECURQUANT_MODULE_PATHS)
    assert required == dict(verify_stage0.REQUIRED_LOADED_RECURQUANT_MODULE_PATHS)
    assert all(module_paths.get(name) == path for name, path in required.items())
    assert set(capture_stage0._loaded_local_source_paths()) <= set(SOURCE_IDENTITY_PATHS)
    assert "pyproject.toml" in SOURCE_IDENTITY_PATHS
    assert "tests/test_statelease_evaluation.py" in SOURCE_IDENTITY_PATHS


@pytest.mark.parametrize("tamper", ["external", "missing"])
def test_loaded_recurquant_module_closure_rejects_substitution_or_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    module_name = "recurquant.qwen35"
    module = capture_stage0.sys.modules[module_name]
    if tamper == "external":
        external = tmp_path / "qwen35.py"
        external.write_text("outside authenticated repository\n", encoding="utf-8")
        monkeypatch.setattr(module, "__file__", str(external))
    else:
        monkeypatch.delitem(capture_stage0.sys.modules, module_name)

    with pytest.raises(
        RuntimeError,
        match="outside the authenticated repository|closure differs",
    ):
        capture_stage0._loaded_recurquant_module_paths()


def test_experiment012_stage0_identity_and_provenance_are_exact() -> None:
    expected_provenance = (
        "research/EXPERIMENT_012_STATELEASE_PROTOCOL.md",
        "research/EXPERIMENT_012_STAGE_A_IDENTITY.md",
        "research/EXPERIMENT_011_STAGE_A_ADMINISTRATIVE_NULL.md",
        "evidence/experiment011-statelease-stage-a-administrative-null.json",
        "artifacts/experiment011-statelease-stage-a-666.attempt.json",
        "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md",
        "research/EXPERIMENT_011_STAGE_A_IDENTITY.md",
        "research/EXPERIMENT_010_STAGE_A_ADMINISTRATIVE_NULL.md",
        "evidence/experiment010-statelease-stage-a-administrative-null.json",
        "artifacts/experiment010-statelease-stage-a-666.attempt.json",
        "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
        "research/EXPERIMENT_010_STAGE_A_IDENTITY.md",
    )

    assert EXPERIMENT_ID == VERIFIER_EXPERIMENT_ID == "experiment012"
    assert SCHEMA_NAME == PRODUCTION_SCHEMA == "recurquant.experiment012.stage0.production.v1"
    assert DEFAULT_ARTIFACT == REPO_ROOT / "artifacts" / "experiment012_stage0_production.pt"
    assert expected_provenance == EXPERIMENT012_SOURCE_PROVENANCE_PATHS
    assert expected_provenance == VERIFIER_EXPERIMENT012_SOURCE_PROVENANCE_PATHS
    assert all(relative in SOURCE_IDENTITY_PATHS for relative in expected_provenance)
    assert all(relative in VERIFIER_SOURCE_IDENTITY_PATHS for relative in expected_provenance)


def test_whole_cache_inventory_deduplicates_storage_and_records_all_alias_paths() -> None:
    mirror = torch.zeros((1, 16, 128, 128), dtype=torch.float32)
    cache = SimpleNamespace(
        layers=[],
        first=mirror,
        nested={"second": [mirror]},
    )

    inventory = _build_whole_cache_storage_inventory(cache, ())

    assert len(inventory) == 1
    assert inventory[0]["classification"] == "unexplained"
    assert inventory[0]["storage_nbytes"] == mirror.untyped_storage().nbytes()
    assert [view["path"] for view in inventory[0]["views"]] == [
        "cache.first",
        "cache.nested['second'][0]",
    ]


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.int8],
)
def test_whole_cache_audit_rejects_unexplained_raw_storage_at_every_dtype(
    dtype: torch.dtype,
) -> None:
    cache = SimpleNamespace(
        layers=[],
        hidden_raw_state=torch.zeros((1, 16, 128, 128), dtype=dtype),
    )

    with pytest.raises(RuntimeError, match="whole-cache all-dtype audit found unexplained"):
        _audit_whole_cache_persistent_tensors(cache, ())


def test_whole_cache_audit_rejects_global_and_per_layer_split_raw_mirrors() -> None:
    global_cache = SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(24)],
        hidden_raw_state=torch.zeros((18, 1, 16, 128, 128), dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match="whole-cache all-dtype audit found unexplained"):
        _audit_whole_cache_persistent_tensors(global_cache, ())

    mapped_cache = SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(24)],
        raw_by_layer={
            layer_index: torch.zeros((1, 16, 128, 128), dtype=torch.float32)
            for layer_index in capture_stage0.LINEAR_LAYER_INDICES
        },
    )
    with pytest.raises(RuntimeError, match="whole-cache all-dtype audit found unexplained"):
        _audit_whole_cache_persistent_tensors(mapped_cache, ())

    split_layers = [SimpleNamespace() for _ in range(24)]
    for layer_index in capture_stage0.LINEAR_LAYER_INDICES:
        split_layers[layer_index].raw_left = [torch.zeros((1, 16, 64, 128), dtype=torch.float32)]
        split_layers[layer_index].raw_right = [torch.zeros((1, 16, 64, 128), dtype=torch.float32)]
    split_cache = SimpleNamespace(layers=split_layers)
    with pytest.raises(RuntimeError, match="whole-cache all-dtype audit found unexplained"):
        _audit_whole_cache_persistent_tensors(split_cache, ())


def test_whole_cache_audit_traverses_dataclasses_slots_and_sets() -> None:
    @dataclass(slots=True)
    class DataclassHolder:
        payload: object

    class SlotHolder:
        __slots__ = ("payload",)

        def __init__(self, payload: object) -> None:
            self.payload = payload

    hidden = torch.ones((32,), dtype=torch.int16)
    cache = SimpleNamespace(
        layers=[],
        dataclass_holder=DataclassHolder({"chunks": (hidden,)}),
        slot_set={SlotHolder(hidden)},
    )

    with pytest.raises(RuntimeError, match="whole-cache all-dtype audit found unexplained"):
        _audit_whole_cache_persistent_tensors(cache, ())


def test_independent_verifier_recomputes_whole_cache_inventory_not_declaration() -> None:
    tensors: dict[str, torch.Tensor] = {}
    for layer_index, high_rows in capture_stage0.EXPERIMENT010_STATELEASE_LAYER_QUOTAS.items():
        tensors.update(
            {
                f"layer_{layer_index}.checkpoint.low_payload": torch.zeros(
                    (2048 - high_rows, 64),
                    dtype=torch.uint8,
                ),
                f"layer_{layer_index}.checkpoint.high_payload": torch.zeros(
                    (high_rows, 128),
                    dtype=torch.int8,
                ),
                f"layer_{layer_index}.checkpoint.scales": torch.zeros(
                    (2048,),
                    dtype=torch.float16,
                ),
                f"layer_{layer_index}.checkpoint.precision_mask": torch.zeros(
                    (256,),
                    dtype=torch.uint8,
                ),
                f"layer_{layer_index}.query_energy_ema": torch.zeros(
                    (16, 128),
                    dtype=torch.float32,
                ),
                f"layer_{layer_index}.normalized_key_buffer": torch.zeros(
                    (5, 16, 128),
                    dtype=torch.bfloat16,
                ),
                f"layer_{layer_index}.update_buffer": torch.zeros(
                    (5, 16, 128),
                    dtype=torch.bfloat16,
                ),
                f"layer_{layer_index}.log_decay_buffer": torch.zeros(
                    (5, 16),
                    dtype=torch.float32,
                ),
                f"layer_{layer_index}.valid_count": torch.zeros(
                    (1,),
                    dtype=torch.int32,
                ),
            }
        )
    audit = _expected_whole_cache_storage_audit(tensors)

    assert audit["candidate_unique_storage_bytes"] == STATELEASE_BYTES
    assert _verify_whole_cache_storage_audit(audit, tensors) == audit
    tampered = copy.deepcopy(audit)
    tampered["inventory"][0]["views"][0]["path"] = "cache.hidden_chunks[0]"
    tampered["no_unexplained_persistent_storage"] = True
    with pytest.raises(Stage0VerificationError, match="does not reconcile"):
        _verify_whole_cache_storage_audit(tampered, tensors)


def test_repository_snapshot_checks_the_complete_untracked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_bytes(b"source\n")
    for arguments in (
        ["git", "init", "--quiet"],
        ["git", "add", "source.py"],
        [
            "git",
            "-c",
            "user.name=Stage0 Test",
            "-c",
            "user.email=stage0@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
    ):
        subprocess.run(arguments, cwd=repo, check=True, capture_output=True, text=True)
    (repo / "unrelated.py").write_text("untracked\n", encoding="utf-8")
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    snapshot = capture_stage0._repository_source_snapshot()

    assert snapshot["worktree_clean"] is False
    assert snapshot["sources_match_head"] is True
    assert snapshot["repository_binding"]["object_format"] == "sha1"
    assert snapshot["repository_binding"]["raw_source_hash_mode"] == (
        "git_hash_object_no_filters_stdin_paths"
    )


def test_source_identity_persists_independent_matching_start_and_end_snapshots() -> None:
    snapshot = {
        "repo_head": "a" * 40,
        "repository_binding": {"schema": "synthetic"},
        "source_hashes": {"source.py": "b" * 64},
        "source_set_sha256": "c" * 64,
        "head_blob_hashes": {"source.py": "e" * 40},
        "worktree_blob_hashes": {"source.py": "e" * 40},
        "sources_match_head": True,
        "worktree_clean": True,
    }

    identity = _finalize_source_identity(snapshot, dict(snapshot))

    assert identity["capture_start"] == snapshot
    assert identity["capture_end"] == snapshot
    assert identity["capture_start"] is not identity["capture_end"]
    assert identity["capture_start_equals_end"] is True
    changed = dict(snapshot)
    changed["repo_head"] = "d" * 40
    with pytest.raises(RuntimeError, match="changed during capture"):
        _finalize_source_identity(snapshot, changed)
    dirty = dict(snapshot)
    dirty["worktree_clean"] = False
    with pytest.raises(RuntimeError, match="complete repository worktree"):
        _finalize_source_identity(dirty, dirty)


def test_capture_refuses_a_dirty_complete_worktree_before_production_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture_stage0,
        "_repository_source_snapshot",
        lambda: {"worktree_clean": False},
    )
    with pytest.raises(RuntimeError, match="complete repository worktree"):
        build_production_artifact()


def test_capture_refuses_source_bytes_that_do_not_match_head_before_production_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture_stage0,
        "_repository_source_snapshot",
        lambda: {"worktree_clean": True, "sources_match_head": False},
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "runtime package manifest differs from the frozen identity"
            "|regular-file blobs at HEAD"
        ),
    ):
        build_production_artifact()


@pytest.mark.parametrize("index_flag", ["--skip-worktree", "--assume-unchanged"])
def test_repository_snapshot_rejects_hidden_index_flag_on_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    source.write_text("original\n", encoding="utf-8")
    for arguments in (
        ["git", "init", "--quiet"],
        ["git", "add", "source.py"],
        [
            "git",
            "-c",
            "user.name=Stage0 Test",
            "-c",
            "user.email=stage0@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
        ["git", "update-index", index_flag, "source.py"],
    ):
        subprocess.run(arguments, cwd=repo, check=True, capture_output=True, text=True)
    source.write_text("tampered\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""

    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="hidden or non-canonical tracked entries"):
        capture_stage0._repository_source_snapshot()
    with pytest.raises(
        Stage0VerificationError,
        match="hidden or non-canonical tracked entries",
    ):
        verify_stage0._current_repository_source_snapshot(repo)


@pytest.mark.parametrize("index_flag", ["--skip-worktree", "--assume-unchanged"])
def test_repository_snapshot_rejects_hidden_index_flag_outside_source_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    unrelated = repo / "unrelated.py"
    source.write_text("authenticated\n", encoding="utf-8")
    unrelated.write_text("original\n", encoding="utf-8")
    for arguments in (
        ["git", "init", "--quiet"],
        ["git", "add", "source.py", "unrelated.py"],
        [
            "git",
            "-c",
            "user.name=Stage0 Test",
            "-c",
            "user.email=stage0@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
        ["git", "update-index", index_flag, "unrelated.py"],
    ):
        subprocess.run(arguments, cwd=repo, check=True, capture_output=True, text=True)
    unrelated.write_text("tampered but hidden\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""

    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="hidden or non-canonical tracked entries"):
        capture_stage0._repository_source_snapshot()
    with pytest.raises(
        Stage0VerificationError,
        match="hidden or non-canonical tracked entries",
    ):
        verify_stage0._current_repository_source_snapshot(repo)


def test_capture_git_environment_scrubs_all_caller_git_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {
        "GIT_DIR": "attacker.git",
        "GIT_WORK_TREE": "attacker-tree",
        "GIT_COMMON_DIR": "attacker-common",
        "GIT_INDEX_FILE": "attacker-index",
        "GIT_OBJECT_DIRECTORY": "attacker-objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "attacker-alternates",
        "GIT_REPLACE_REF_BASE": "refs/evil/",
        "GIT_GRAFT_FILE": "attacker-grafts",
        "GIT_SHALLOW_FILE": "attacker-shallow",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": "attacker-tree",
        "GIT_CONFIG_PARAMETERS": "'core.bare=true'",
    }
    for key, value in injected.items():
        monkeypatch.setenv(key, value)

    environment = capture_stage0._sanitized_capture_git_environment()

    assert not set(injected).intersection(environment)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == capture_stage0.os.devnull
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_capture_git_forces_read_only_consistency_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["arguments"] = arguments
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(capture_stage0.subprocess, "run", fake_run)

    capture_stage0._capture_git("status", "--porcelain=v1")

    arguments = recorded["arguments"]
    assert isinstance(arguments, list)
    joined = "\0".join(arguments)
    assert "core.useReplaceRefs=false" in joined
    assert f"core.attributesFile={capture_stage0.os.devnull}" in joined
    assert "core.fsmonitor=false" in joined
    assert "core.untrackedCache=false" in joined
    assert f"core.hooksPath={capture_stage0.os.devnull}" in joined


def _initialize_source_repository(path: Path) -> Path:
    path.mkdir()
    source = path / "source.py"
    source.write_bytes(b"original\n")
    for arguments in (
        ["git", "init", "--quiet"],
        ["git", "add", "source.py"],
        [
            "git",
            "-c",
            "user.name=Stage0 Test",
            "-c",
            "user.email=stage0@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
    ):
        subprocess.run(arguments, cwd=path, check=True, capture_output=True, text=True)
    return source


def test_repository_snapshot_ignores_hostile_git_routing_and_config_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    attacker = tmp_path / "attacker"
    _initialize_source_repository(trusted)
    _initialize_source_repository(attacker).write_text("attacker\n", encoding="utf-8")
    trusted_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=trusted,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", trusted)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))
    monkeypatch.setenv("GIT_DIR", str(attacker / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(attacker))
    monkeypatch.setenv("GIT_INDEX_FILE", str(attacker / ".git" / "index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(attacker))

    snapshot = capture_stage0._repository_source_snapshot()

    assert snapshot["repo_head"] == trusted_head
    assert snapshot["sources_match_head"] is True
    assert (
        snapshot["repository_binding"]["top_level_path_sha256"]
        == (snapshot["repository_binding"]["worktree_path_sha256"])
    )


@pytest.mark.parametrize(
    "hazard",
    ["alternates", "http-alternates", "grafts", "replace", "shallow"],
)
def test_repository_binding_rejects_unsafe_object_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    repo = tmp_path / "repo"
    _initialize_source_repository(repo)
    git_dir = repo / ".git"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if hazard in {"alternates", "http-alternates"}:
        info = git_dir / "objects" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / hazard).write_text(str(git_dir / "objects"), encoding="utf-8")
    elif hazard == "grafts":
        info = git_dir / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "grafts").write_text(f"{head}\n", encoding="ascii")
    elif hazard == "replace":
        subprocess.run(
            ["git", "update-ref", f"refs/replace/{head}", head],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        (git_dir / "shallow").write_text(f"{head}\n", encoding="ascii")
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="alternates|grafts|shallow|replacement"):
        capture_stage0._repository_source_snapshot()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("filter.attacker.clean", "type"),
        ("core.fsmonitor", "true"),
        ("core.untrackedCache", "true"),
        ("core.hooksPath", "hooks"),
    ],
)
def test_repository_binding_rejects_unsafe_local_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    repo = tmp_path / "repo"
    _initialize_source_repository(repo)
    subprocess.run(
        ["git", "config", "--local", key, value],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="unsafe local Git config"):
        capture_stage0._repository_source_snapshot()


def test_linked_worktree_binding_rejects_mismatched_reverse_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    _initialize_source_repository(primary)
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(linked)],
        cwd=primary,
        check=True,
        capture_output=True,
        text=True,
    )
    git_marker = linked / ".git"
    git_dir = Path(git_marker.read_text(encoding="utf-8").removeprefix("gitdir: ").strip())
    (git_dir / "gitdir").write_text(str(primary / ".git"), encoding="utf-8")
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", linked)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="reverse pointer"):
        capture_stage0._repository_source_snapshot()


def test_raw_no_filter_hash_detects_attribute_hidden_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    source.write_bytes(b"original\n")
    (repo / ".gitattributes").write_text("source.py text eol=lf\n", encoding="utf-8")
    for arguments in (
        ["git", "init", "--quiet"],
        ["git", "add", "source.py", ".gitattributes"],
        [
            "git",
            "-c",
            "user.name=Stage0 Test",
            "-c",
            "user.email=stage0@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
    ):
        subprocess.run(arguments, cwd=repo, check=True, capture_output=True, text=True)
    source.write_bytes(b"original\r\n")
    filtered_hash = subprocess.run(
        ["git", "hash-object", "--path=source.py", "source.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_hash = subprocess.run(
        ["git", "rev-parse", "HEAD:source.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert filtered_hash == head_hash
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    monkeypatch.setattr(capture_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    snapshot = capture_stage0._repository_source_snapshot()

    assert snapshot["sources_match_head"] is False
    assert (
        snapshot["head_blob_hashes"]["source.py"] != (snapshot["worktree_blob_hashes"]["source.py"])
    )


def test_runtime_identity_is_exact_and_includes_pinned_package_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        runtime = _runtime_identity()
    except RuntimeError as error:
        if (
            "runtime package manifest differs from the frozen identity" not in str(error)
        ):
            raise
        legacy_packages = {
            distribution: str(importlib.metadata.version(distribution))
            for distribution in capture_stage0.RUNTIME_PACKAGE_DISTRIBUTIONS
        }

        def legacy_runtime_package_manifest() -> tuple[dict[str, str], str]:
            payload = json.dumps(legacy_packages, sort_keys=True, separators=(",", ":")) + "\n"
            return (
                dict(legacy_packages),
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            )

        monkeypatch.setattr(
            capture_stage0, "_runtime_package_manifest", legacy_runtime_package_manifest
        )
        monkeypatch.setattr(
            verify_stage0, "_runtime_package_manifest", legacy_runtime_package_manifest
        )
        runtime = _runtime_identity()

    assert verify_stage0._verify_runtime_identity(runtime) == runtime
    assert runtime["python_executable"] == Path(capture_stage0.sys.executable).name
    assert runtime["python_environment"] == Path(capture_stage0.sys.prefix).name
    assert ":\\" not in runtime["python_executable"]
    assert ":\\" not in runtime["python_environment"]
    distribution_fields = (
        ("datasets", "datasets_version"),
        ("fsspec", "fsspec_version"),
        ("huggingface-hub", "huggingface_hub_version"),
        ("numpy", "numpy_version"),
        ("pyarrow", "pyarrow_version"),
        ("safetensors", "safetensors_version"),
        ("tokenizers", "tokenizers_version"),
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
    )
    version_fields = tuple(field for _distribution, field in distribution_fields)
    assert all(type(runtime[field]) is str for field in version_fields)
    packages = {distribution: runtime[field] for distribution, field in distribution_fields}
    assert set(packages) == set(capture_stage0.RUNTIME_PACKAGE_DISTRIBUTIONS)
    canonical = json.dumps(packages, sort_keys=True, separators=(",", ":")) + "\n"
    assert runtime["package_manifest_sha256"] == PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        PINNED_RUNTIME_PACKAGE_MANIFEST_SHA256
    )
    for field in (*version_fields, "package_manifest_sha256"):
        changed = dict(runtime)
        changed[field] = "0.0-tampered"
        with pytest.raises(Stage0VerificationError, match="runtime identity differs"):
            _verify_runtime_identity(changed)


def test_full_production_capture_completes_only_after_independent_verification(
    authenticated_artifact: Path,
) -> None:
    report = verify_production_stage0(authenticated_artifact)

    assert report["status"] == "production_stage0_pass"
    assert report["experiment_stage0_complete"] is True
    assert report["quality_data_accessed"] is False
    assert report["protected_mbpp_window_accessed"] is False
    assert report["weights_only_load"] is True
    assert report["artifact"] == authenticated_artifact.name
    assert report["sidecar"] == f"{authenticated_artifact.name}.sha256"
    assert str(authenticated_artifact.resolve()) not in json.dumps(report)
    assert report["sidecar_file_sha256"] == _file_sha256(
        authenticated_artifact.with_suffix(authenticated_artifact.suffix + ".sha256")
    )
    assert (
        report["repository_commit"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert report["runtime_identity"] == _runtime_identity()
    assert report["storage"]["resident_bytes"] == 3_454_664
    assert report["whole_cache_storage_audit"]["candidate_unique_storage_bytes"] == 3_454_664
    assert report["whole_cache_storage_audit"]["unexplained_unique_storage_bytes"] == 0
    assert set(report["equal_byte_comparators"]) == {
        "expanded_rht_q4_q8",
        "rht_q4_q6_q8",
        "rht_residual_q4",
    }


def test_serialized_schema_loads_with_weights_only_and_pins_effective_plan_summary(
    authenticated_artifact: Path,
) -> None:
    artifact = torch.load(
        authenticated_artifact,
        map_location="cpu",
        weights_only=True,
    )

    assert artifact["schema"] == SCHEMA_NAME
    assert artifact["method_identity"]["effective_plan_sha256"] == EFFECTIVE_PLAN_SHA256
    assert artifact["declarations"]["synthetic_only"] is True
    assert artifact["declarations"]["quality_data_accessed"] is False


def test_writer_rejects_a_nonignored_repository_destination() -> None:
    with pytest.raises(ValueError, match="Git-ignored destination"):
        write_artifact(
            {"canonical_payload_sha256": "0" * 64},
            REPO_ROOT / "research" / "must-not-write-stage0.pt",
        )


def test_writer_accepts_an_ignored_repository_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    path = repo / "artifacts" / "stage0.pt"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True)
    (repo / ".gitignore").write_text("artifacts/**\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setattr(capture_stage0, "REPO_ROOT", repo)
    assert not path.exists()
    assert not sidecar.exists()
    receipt = write_artifact(
        {"canonical_payload_sha256": "0" * 64},
        path,
    )
    assert receipt["artifact"] == "artifacts/stage0.pt"
    assert receipt["sidecar"] == "artifacts/stage0.pt.sha256"
    assert path.is_file()
    assert sidecar.is_file()
    assert sidecar.read_bytes() == (str(receipt["file_sha256"]).encode("ascii") + b"  stage0.pt\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_artifact(
            {"canonical_payload_sha256": "0" * 64},
            path,
        )


def test_serialized_byte_tamper_fails_before_deserialization(
    authenticated_artifact: Path,
) -> None:
    sidecar = authenticated_artifact.with_suffix(authenticated_artifact.suffix + ".sha256")
    original = sidecar.read_bytes()
    sidecar.write_bytes(b"0" * 64 + b"  " + authenticated_artifact.name.encode("ascii") + b"\n")
    try:
        with pytest.raises(Stage0VerificationError, match="serialized artifact SHA-256"):
            verify_production_stage0(authenticated_artifact)
    finally:
        sidecar.write_bytes(original)


@pytest.mark.parametrize(
    "invalid_separator_or_terminator",
    [b"\t", b" ", b"  ", b"  \r"],
)
def test_sidecar_parser_rejects_noncanonical_whitespace_or_line_ending(
    authenticated_artifact: Path,
    invalid_separator_or_terminator: bytes,
) -> None:
    sidecar = authenticated_artifact.with_suffix(authenticated_artifact.suffix + ".sha256")
    original = sidecar.read_bytes()
    digest = _file_sha256(authenticated_artifact).encode("ascii")
    name = authenticated_artifact.name.encode("ascii")
    if invalid_separator_or_terminator == b"  \r":
        tampered = digest + b"  " + name + b"\r\n"
    elif invalid_separator_or_terminator == b"  ":
        tampered = digest + b"  " + name
    else:
        tampered = digest + invalid_separator_or_terminator + name + b"\n"
    sidecar.write_bytes(tampered)
    try:
        with pytest.raises(Stage0VerificationError, match="closed syntax"):
            verify_stage0.load_authenticated_production_artifact(authenticated_artifact)
    finally:
        sidecar.write_bytes(original)


def test_loaded_artifact_root_must_be_an_exact_plain_dict(
    authenticated_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = verify_stage0.torch.load

    def ordered_load(*args: object, **kwargs: object) -> OrderedDict[str, object]:
        loaded = original_load(*args, **kwargs)
        return OrderedDict(loaded)

    monkeypatch.setattr(verify_stage0.torch, "load", ordered_load)
    with pytest.raises(Stage0VerificationError, match="exact plain dict"):
        verify_stage0.load_authenticated_production_artifact(authenticated_artifact)


def test_weights_only_load_uses_the_exact_authenticated_immutable_buffer(
    authenticated_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = verify_stage0.torch.load
    original_artifact = authenticated_artifact.read_bytes()
    loaded_from_buffer = False

    def swap_path_during_load(source: object, *args: object, **kwargs: object) -> object:
        nonlocal loaded_from_buffer
        assert isinstance(source, verify_stage0.io.BytesIO)
        loaded_from_buffer = True
        authenticated_artifact.write_bytes(b"transient replacement")
        try:
            return original_load(source, *args, **kwargs)
        finally:
            authenticated_artifact.write_bytes(original_artifact)

    monkeypatch.setattr(verify_stage0.torch, "load", swap_path_during_load)
    loaded = verify_stage0.load_authenticated_production_artifact(authenticated_artifact)

    assert loaded_from_buffer is True
    assert loaded["schema"] == SCHEMA_NAME


def test_verifier_rejects_artifact_mutation_after_authenticated_load(
    authenticated_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_artifact = authenticated_artifact.read_bytes()
    original_sidecar = authenticated_artifact.with_suffix(
        authenticated_artifact.suffix + ".sha256"
    ).read_bytes()
    original_verify_identity = verify_stage0._verify_source_and_method_identity
    mutated = False

    def mutate_after_load(*args: object, **kwargs: object) -> None:
        nonlocal mutated
        if not mutated:
            with authenticated_artifact.open("ab") as handle:
                handle.write(b"concurrent mutation")
            mutated = True
        original_verify_identity(*args, **kwargs)

    monkeypatch.setattr(
        verify_stage0,
        "_verify_source_and_method_identity",
        mutate_after_load,
    )
    try:
        with pytest.raises(
            Stage0VerificationError,
            match=(
                "serialized artifact SHA-256 differs from its authenticated sidecar"
                "|changed during independent Stage-0 verification"
            ),
        ):
            verify_production_stage0(authenticated_artifact)
    finally:
        authenticated_artifact.write_bytes(original_artifact)
        authenticated_artifact.with_suffix(authenticated_artifact.suffix + ".sha256").write_bytes(
            original_sidecar
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("source", "source changed"),
        ("repository_commit", "HEAD differs"),
        ("capture_end", "capture start/end"),
        ("runtime", "runtime identity differs"),
        ("loaded_module", "module-name/source-path closure differs"),
        ("trace", "successful kernel final state|dense replay mismatch"),
        ("resident", "did not declare absence"),
        ("inventory", "does not reconcile"),
        ("comparator", "differs"),
        ("allocation", "differs|invalid"),
    ],
)
def test_semantically_resigned_tampering_still_fails_closed(
    authenticated_artifact: Path,
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    payload = torch.load(
        authenticated_artifact,
        map_location="cpu",
        weights_only=True,
    )
    if tamper == "source":
        source_identity = payload["source_identity"]
        for snapshot in (
            source_identity,
            source_identity["capture_start"],
            source_identity["capture_end"],
        ):
            snapshot["source_hashes"]["src/recurquant/statelease.py"] = "0" * 64
            snapshot["source_set_sha256"] = canonical_payload_sha256(snapshot["source_hashes"])
    elif tamper == "repository_commit":
        payload["source_identity"]["repo_head"] = "0" * 40
        payload["source_identity"]["capture_start"]["repo_head"] = "0" * 40
        payload["source_identity"]["capture_end"]["repo_head"] = "0" * 40
    elif tamper == "capture_end":
        payload["source_identity"]["capture_end"]["repo_head"] = "1" * 40
    elif tamper == "runtime":
        payload["runtime_identity"]["numpy_version"] = "0.0-tampered"
    elif tamper == "loaded_module":
        payload["source_identity"]["loaded_recurquant_module_paths"]["recurquant.qwen35"] = (
            "src/recurquant/statelease.py"
        )
    elif tamper == "trace":
        final = payload["production_trace"][2]["signals"]["successful_final_state"]
        final[0, 0, 0, 0] += 0.25
    elif tamper == "resident":
        payload["resident_snapshot"]["no_hidden_persistent_state_mirror"] = False
    elif tamper == "inventory":
        payload["resident_snapshot"]["whole_cache_storage_audit"]["inventory"][0]["views"][0][
            "path"
        ] = "cache.hidden_raw_state"
    elif tamper == "comparator":
        tensor = payload["equal_byte_comparators"]["snapshots"]["expanded_rht_q4_q8"]["tensors"][
            "layer_0.q4_payload"
        ]
        tensor[0, 0] ^= 1
    else:
        tensor = payload["equal_byte_comparators"]["snapshots"]["rht_q4_q6_q8"]["tensors"][
            "layer_0.precision_codes"
        ]
        tensor[0] ^= 1
    path = tmp_path / f"tampered-{tamper}.pt"
    _resign_payload(payload, path)

    with pytest.raises(Stage0VerificationError, match=message):
        verify_production_stage0(path)
