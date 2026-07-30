from __future__ import annotations

import copy
import hashlib
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.capture_statelease_stage0 as capture_stage0
import scripts.verify_statelease_stage0 as verify_stage0
from scripts.capture_statelease_stage0 import (
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
    STATELEASE_BYTES,
    Stage0VerificationError,
    _expected_whole_cache_storage_audit,
    _verify_runtime_identity,
    _verify_whole_cache_storage_audit,
    verify_production_stage0,
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
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{_file_sha256(path)}  {path.name}\n",
        encoding="ascii",
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


def test_source_identity_matches_verifier_and_covers_the_complete_package() -> None:
    package_sources = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src" / "recurquant").glob("*.py")
    }

    assert SOURCE_IDENTITY_PATHS == VERIFIER_SOURCE_IDENTITY_PATHS
    assert package_sources <= set(SOURCE_IDENTITY_PATHS)
    assert set(capture_stage0._loaded_local_source_paths()) <= set(SOURCE_IDENTITY_PATHS)
    assert "pyproject.toml" in SOURCE_IDENTITY_PATHS
    assert "research/EXPERIMENT_010_STAGE_A_IDENTITY.md" in SOURCE_IDENTITY_PATHS
    assert "tests/test_statelease_evaluation.py" in SOURCE_IDENTITY_PATHS


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_: object) -> SimpleNamespace:
        calls.append(arguments)
        if arguments[1] == "ls-tree":
            stdout = "".join(
                f"100644 blob {'c' * 40}\t{relative}\n" for relative in SOURCE_IDENTITY_PATHS
            )
        elif arguments[1] == "hash-object":
            stdout = "".join(f"{'c' * 40}\n" for _ in SOURCE_IDENTITY_PATHS)
        elif arguments[1] == "rev-parse":
            stdout = "a" * 40 + "\n"
        else:
            stdout = "?? unrelated.py\n"
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(capture_stage0, "_file_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(capture_stage0.subprocess, "run", fake_run)

    snapshot = capture_stage0._repository_source_snapshot()

    assert snapshot["worktree_clean"] is False
    assert snapshot["sources_match_head"] is True
    assert calls[3] == ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    assert "--" not in calls[3]
    assert calls[0][1:4] == ["ls-tree", "-r", "HEAD"]
    assert calls[1] == ["git", "hash-object", "--stdin-paths"]


def test_source_identity_persists_independent_matching_start_and_end_snapshots() -> None:
    snapshot = {
        "repo_head": "a" * 40,
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
    with pytest.raises(RuntimeError, match="regular-file blobs at HEAD"):
        build_production_artifact()


@pytest.mark.parametrize("index_flag", ["--skip-worktree", "--assume-unchanged"])
def test_source_blob_check_defeats_git_index_hidden_modification(
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

    producer_snapshot = capture_stage0._repository_source_snapshot()
    verifier_snapshot = verify_stage0._current_repository_source_snapshot(repo)

    assert producer_snapshot["worktree_clean"] is True
    assert verifier_snapshot["worktree_clean"] is True
    assert producer_snapshot["sources_match_head"] is False
    assert verifier_snapshot["sources_match_head"] is False
    assert producer_snapshot["head_blob_hashes"] != producer_snapshot["worktree_blob_hashes"]


def test_runtime_identity_is_exact_and_includes_numpy_and_python_environment() -> None:
    runtime = _runtime_identity()

    assert _verify_runtime_identity(runtime) == runtime
    assert runtime["python_executable"] == Path(capture_stage0.sys.executable).name
    assert runtime["python_environment"] == Path(capture_stage0.sys.prefix).name
    assert ":\\" not in runtime["python_executable"]
    assert ":\\" not in runtime["python_environment"]
    assert type(runtime["torch_version"]) is str
    assert type(runtime["transformers_version"]) is str
    assert isinstance(runtime["numpy_version"], str)
    changed = dict(runtime)
    changed["numpy_version"] = "0.0-tampered"
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


def test_serialized_schema_loads_with_weights_only_and_pins_plan(
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
    assert receipt["artifact"] == str(path.resolve())
    assert path.is_file()
    assert sidecar.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_artifact(
            {"canonical_payload_sha256": "0" * 64},
            path,
        )


def test_serialized_byte_tamper_fails_before_deserialization(
    authenticated_artifact: Path,
) -> None:
    sidecar = authenticated_artifact.with_suffix(authenticated_artifact.suffix + ".sha256")
    original = sidecar.read_text(encoding="ascii")
    sidecar.write_text(f"{'0' * 64}  {authenticated_artifact.name}\n", encoding="ascii")
    try:
        with pytest.raises(Stage0VerificationError, match="serialized artifact SHA-256"):
            verify_production_stage0(authenticated_artifact)
    finally:
        sidecar.write_text(original, encoding="ascii")


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("source", "source changed"),
        ("repository_commit", "HEAD differs"),
        ("capture_end", "capture start/end"),
        ("runtime", "runtime identity differs"),
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
