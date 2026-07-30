from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import scripts.verify_statelease_stage0 as verify_stage0
from scripts.verify_statelease_stage0 import (
    CHECKPOINT_BYTES,
    EXPANDED_Q48_PROMOTIONS,
    EXPERIMENT011_SOURCE_PROVENANCE_PATHS,
    EXPERIMENT_ID,
    FROZEN_SEED,
    KEY_NORM_EPS,
    MULTIBIT_MARGINAL_STEPS,
    PRODUCTION_SCHEMA,
    REPLAY_CAPACITY,
    RESIDUAL_Q4_ROWS,
    SOURCE_IDENTITY_PATHS,
    STATELEASE_BYTES,
    ReplayRecord,
    Stage0VerificationError,
    _independent_allocate_multibit_fast,
    allocate_exact_multibit,
    assert_independent_imports,
    assert_replay_matches,
    assert_rollback_preserved,
    audit_resident_snapshot,
    canonical_empty_resident_snapshot,
    compact_full_buffer,
    construct_and_choose_boundary,
    dense_replay,
    dense_transition_from_record,
    derive_successful_record,
    equal_byte_codec_contracts,
    frozen_storage_contract,
    guard_protected_mbpp_window,
    independent_rht_decode,
    independent_rht_encode,
    independent_rht_signs,
    normalize_consumed_key,
    pack_physical_q4_q6_q8,
    pack_physical_q4_q8,
    pack_physical_residual_q4,
    q4_q8_physical_benefit,
    replay_stored_buffers,
    resident_snapshot_digest,
    stable_descending_indices,
    store_records_bf16,
    verify_cc1_compatibility,
    verify_reset_snapshot,
    verify_resume_integrity,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_statelease_stage0.py"


def test_experiment011_verifier_identity_pins_complete_provenance() -> None:
    expected_provenance = (
        "research/EXPERIMENT_011_STATELEASE_PROTOCOL.md",
        "research/EXPERIMENT_011_STAGE_A_IDENTITY.md",
        "research/EXPERIMENT_010_STAGE_A_ADMINISTRATIVE_NULL.md",
        "evidence/experiment010-statelease-stage-a-administrative-null.json",
        "artifacts/experiment010-statelease-stage-a-666.attempt.json",
        "research/EXPERIMENT_010_STATELEASE_PROTOCOL.md",
        "research/EXPERIMENT_010_STAGE_A_IDENTITY.md",
    )

    assert EXPERIMENT_ID == "experiment011"
    assert PRODUCTION_SCHEMA == "recurquant.experiment011.stage0.production.v1"
    assert expected_provenance == EXPERIMENT011_SOURCE_PROVENANCE_PATHS
    assert all(relative in SOURCE_IDENTITY_PATHS for relative in expected_provenance)


def _initialize_verifier_source_repository(path: Path) -> Path:
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


def _synthetic_repository_binding() -> dict[str, object]:
    return {
        "schema": verify_stage0.GIT_REPOSITORY_BINDING_SCHEMA,
        "top_level_path_sha256": "1" * 64,
        "worktree_path_sha256": "1" * 64,
        "git_dir_path_sha256": "2" * 64,
        "common_dir_path_sha256": "2" * 64,
        "index_path_sha256": "3" * 64,
        "object_dir_path_sha256": "4" * 64,
        "git_dir_kind": "main_worktree",
        "object_format": "sha1",
        "inside_worktree": True,
        "bare": False,
        "shallow": False,
        "alternates_absent": True,
        "grafts_absent": True,
        "replace_refs_absent": True,
        "unsafe_local_config_absent": True,
        "hidden_index_flags_absent": True,
        "local_config_sha256": "5" * 64,
        "replacement_objects_disabled": True,
        "system_and_global_config_disabled": True,
        "fsmonitor_and_untracked_cache_disabled": True,
        "hooks_disabled": True,
        "worktree_gitdir_binding_verified": True,
        "raw_source_hash_mode": "git_hash_object_no_filters_stdin_paths",
    }


def test_verifier_git_environment_scrubs_all_caller_git_overrides(
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

    environment = verify_stage0._sanitized_verifier_git_environment()

    assert not set(injected).intersection(environment)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == verify_stage0.os.devnull
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_verifier_git_forces_read_only_consistency_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["arguments"] = arguments
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(verify_stage0.subprocess, "run", fake_run)

    verify_stage0._verifier_git(tmp_path, "status", "--porcelain=v1")

    arguments = recorded["arguments"]
    assert isinstance(arguments, list)
    joined = "\0".join(arguments)
    assert "core.useReplaceRefs=false" in joined
    assert f"core.attributesFile={verify_stage0.os.devnull}" in joined
    assert "core.fsmonitor=false" in joined
    assert "core.untrackedCache=false" in joined
    assert f"core.hooksPath={verify_stage0.os.devnull}" in joined


def test_current_repository_snapshot_ignores_hostile_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    attacker = tmp_path / "attacker"
    _initialize_verifier_source_repository(trusted)
    _initialize_verifier_source_repository(attacker).write_bytes(b"attacker\n")
    trusted_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=trusted,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))
    monkeypatch.setenv("GIT_DIR", str(attacker / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(attacker))
    monkeypatch.setenv("GIT_COMMON_DIR", str(attacker / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(attacker / ".git" / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(attacker / ".git" / "objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(attacker))

    snapshot = verify_stage0._current_repository_source_snapshot(trusted)

    assert snapshot["repo_head"] == trusted_head
    assert snapshot["sources_match_head"] is True
    assert (
        snapshot["repository_binding"]["top_level_path_sha256"]
        == (snapshot["repository_binding"]["worktree_path_sha256"])
    )
    assert str(trusted.resolve()) not in json.dumps(snapshot["repository_binding"])


@pytest.mark.parametrize(
    "hazard",
    ["alternates", "http-alternates", "grafts", "replace", "shallow"],
)
def test_current_repository_binding_rejects_unsafe_object_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    repo = tmp_path / "repo"
    _initialize_verifier_source_repository(repo)
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
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(
        Stage0VerificationError,
        match="alternates|grafts|shallow|replacement",
    ):
        verify_stage0._current_repository_source_snapshot(repo)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("filter.attacker.clean", "type"),
        ("core.fsmonitor", "true"),
        ("core.untrackedCache", "true"),
        ("core.hooksPath", "hooks"),
        ("include.path", "attacker.config"),
        ("extensions.partialClone", "origin"),
    ],
)
def test_current_repository_binding_rejects_unsafe_local_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    repo = tmp_path / "repo"
    _initialize_verifier_source_repository(repo)
    subprocess.run(
        ["git", "config", "--local", key, value],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(Stage0VerificationError, match="unsafe local Git config"):
        verify_stage0._current_repository_source_snapshot(repo)


def test_linked_worktree_binding_rejects_mismatched_reverse_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    _initialize_verifier_source_repository(primary)
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
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    with pytest.raises(Stage0VerificationError, match="reverse pointer"):
        verify_stage0._current_repository_source_snapshot(linked)


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
    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))

    snapshot = verify_stage0._current_repository_source_snapshot(repo)

    assert snapshot["sources_match_head"] is False
    assert (
        snapshot["head_blob_hashes"]["source.py"] != (snapshot["worktree_blob_hashes"]["source.py"])
    )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("schema", "recurquant.git-repository-binding.v0"),
        ("object_format", "sha256"),
        ("raw_source_hash_mode", "git_hash_object_filtered"),
        ("git_dir_kind", []),
        ("inside_worktree", False),
        ("bare", True),
        ("shallow", True),
        ("alternates_absent", False),
        ("grafts_absent", False),
        ("replace_refs_absent", False),
        ("unsafe_local_config_absent", False),
        ("hidden_index_flags_absent", False),
        ("replacement_objects_disabled", False),
        ("system_and_global_config_disabled", False),
        ("fsmonitor_and_untracked_cache_disabled", False),
        ("hooks_disabled", False),
        ("worktree_gitdir_binding_verified", False),
        ("local_config_sha256", "A" * 64),
        ("top_level_path_sha256", "9" * 64),
    ],
)
def test_recorded_repository_binding_rejects_any_weakened_attestation(
    field: str,
    tampered: object,
) -> None:
    binding = _synthetic_repository_binding()
    binding[field] = tampered

    with pytest.raises(Stage0VerificationError):
        verify_stage0._recorded_repository_binding(binding, name="binding")


def test_recorded_repository_and_source_bindings_use_closed_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _synthetic_repository_binding()
    assert verify_stage0._recorded_repository_binding(binding, name="binding") == binding
    missing_binding = dict(binding)
    missing_binding.pop("hooks_disabled")
    with pytest.raises(Stage0VerificationError, match="closed schema"):
        verify_stage0._recorded_repository_binding(missing_binding, name="binding")

    monkeypatch.setattr(verify_stage0, "SOURCE_IDENTITY_PATHS", ("source.py",))
    source_hashes = {"source.py": "a" * 64}
    snapshot = {
        "repo_head": "b" * 40,
        "repository_binding": binding,
        "source_hashes": source_hashes,
        "source_set_sha256": verify_stage0.canonical_payload_sha256(source_hashes),
        "head_blob_hashes": {"source.py": "c" * 40},
        "worktree_blob_hashes": {"source.py": "c" * 40},
        "sources_match_head": True,
        "worktree_clean": True,
    }
    assert verify_stage0._recorded_source_snapshot(snapshot, name="snapshot") == snapshot
    missing_source_binding = dict(snapshot)
    missing_source_binding.pop("repository_binding")
    with pytest.raises(Stage0VerificationError, match="closed schema"):
        verify_stage0._recorded_source_snapshot(
            missing_source_binding,
            name="snapshot",
        )


def _one_transition(
    dtype: torch.dtype,
    *,
    seed: int,
) -> tuple[torch.Tensor, object, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn((1, 2, 8, 8), generator=generator)
    key = torch.randn((1, 1, 2, 8), generator=generator).to(dtype)
    value = torch.randn((1, 1, 2, 8), generator=generator)
    decay = -0.2 * torch.rand((1, 1, 2), generator=generator)
    beta = torch.rand((1, 1, 2), generator=generator)
    normalized = normalize_consumed_key(key)
    decayed = initial * decay[:, 0].exp().unsqueeze(-1).unsqueeze(-1)
    remembered = (decayed * normalized[:, 0].unsqueeze(-1)).sum(dim=-2)
    update = (value[:, 0] - remembered) * beta[:, 0].unsqueeze(-1)
    final = decayed + normalized[:, 0].unsqueeze(-1) * update.unsqueeze(-2)
    record = derive_successful_record(
        initial_state=initial,
        consumed_key=key,
        value=value,
        log_decay=decay,
        beta=beta,
        successful_final_state=final,
    )
    return initial, record, final


def _five_records(
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = FROZEN_SEED,
):
    initial, first, state = _one_transition(dtype, seed=seed)
    records = [first]
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(4):
        key = torch.randn((1, 1, 2, 8), generator=generator).to(dtype)
        value = torch.randn((1, 1, 2, 8), generator=generator)
        decay = -0.2 * torch.rand((1, 1, 2), generator=generator)
        beta = torch.rand((1, 1, 2), generator=generator)
        normalized = normalize_consumed_key(key)[:, 0]
        decayed = state * decay[:, 0].exp().unsqueeze(-1).unsqueeze(-1)
        remembered = (decayed * normalized.unsqueeze(-1)).sum(dim=-2)
        update = (value[:, 0] - remembered) * beta[:, 0].unsqueeze(-1)
        final = decayed + normalized.unsqueeze(-1) * update.unsqueeze(-2)
        record = derive_successful_record(
            initial_state=state,
            consumed_key=key,
            value=value,
            log_decay=decay,
            beta=beta,
            successful_final_state=final,
        )
        records.append(record)
        state = final
    return initial, records, state


def test_verifier_source_has_no_package_under_test_import() -> None:
    assert_independent_imports(SCRIPT)
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == "recurquant" or name.startswith("recurquant.") for name in imported)


@pytest.mark.parametrize(
    "source",
    [
        "from scripts.capture_statelease_stage0 import build_production_artifact\n",
        "import capture_statelease_stage0\n",
        "from scripts import capture_statelease_stage0\n",
        "from . import capture_statelease_stage0\n",
    ],
)
def test_independence_guard_rejects_stage0_producer_imports(
    tmp_path: Path,
    source: str,
) -> None:
    candidate = tmp_path / "candidate_verifier.py"
    candidate.write_text(source, encoding="utf-8")

    with pytest.raises(Stage0VerificationError, match="Stage-0 producer under test"):
        assert_independent_imports(candidate)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_key_is_normalized_in_consumed_dtype_before_fp32_conversion(
    dtype: torch.dtype,
) -> None:
    source = torch.tensor(
        [[[[0.12347, -0.44991, 0.77123, -1.0027, 0.03129, 2.1881, -0.7711, 0.5]]]],
        dtype=torch.float32,
    )
    consumed = source.to(dtype)
    expected = consumed * torch.rsqrt((consumed * consumed).sum(-1, keepdim=True) + KEY_NORM_EPS)
    expected = expected.to(torch.float32)

    actual = normalize_consumed_key(consumed)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    if dtype != torch.float32:
        source32 = consumed.to(torch.float32)
        wrong_order = source32 * torch.rsqrt(source32.square().sum(-1, keepdim=True) + KEY_NORM_EPS)
        assert not torch.equal(actual, wrong_order)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_successful_transition_captures_normalized_key_and_post_correction_update(
    dtype: torch.dtype,
) -> None:
    initial, record, final = _one_transition(dtype, seed=101)

    reconstructed = dense_transition_from_record(initial, record)

    relative_l2, max_absolute = assert_replay_matches(final, reconstructed)
    assert relative_l2 <= 3e-6
    assert max_absolute <= 1e-5
    assert record.normalized_key.dtype == torch.float32
    assert record.update.dtype == torch.float32
    assert record.log_decay.dtype == torch.float32


def test_dense_replay_is_chronological_and_bf16_buffer_roundtrip_is_explicit() -> None:
    initial, records, final = _five_records(dtype=torch.float16, seed=201)

    replayed = dense_replay(initial, records)
    buffers = store_records_bf16(records)
    buffered = replay_stored_buffers(initial, buffers)
    manual_records = [
        replace(
            record,
            normalized_key=record.normalized_key.to(torch.bfloat16).to(torch.float32),
            update=record.update.to(torch.bfloat16).to(torch.float32),
        )
        for record in records
    ]

    assert_replay_matches(final, replayed)
    assert torch.equal(buffered, dense_replay(initial, manual_records))
    assert buffers.normalized_keys.dtype == torch.bfloat16
    assert buffers.updates.dtype == torch.bfloat16
    assert buffers.log_decays.dtype == torch.float32
    assert buffers.valid_count.dtype == torch.int32
    assert int(buffers.valid_count.item()) == 5
    reversed_state = dense_replay(initial, list(reversed(records)))
    assert not torch.allclose(replayed, reversed_state)


def test_dense_replay_fails_closed_outside_frozen_error_limits() -> None:
    reference = torch.ones((1, 1, 2, 2))
    damaged = reference.clone()
    damaged[0, 0, 0, 0] += 2e-5
    with pytest.raises(Stage0VerificationError, match="dense replay mismatch"):
        assert_replay_matches(reference, damaged)


def test_independent_rht_schedule_and_inverse() -> None:
    generator = torch.Generator().manual_seed(301)
    state = torch.randn((1, 3, 5, 128), generator=generator)
    signs = independent_rht_signs(
        layer_index=4,
        heads=3,
        width=128,
        device="cpu",
    )
    digest = signs.to(torch.int8).numpy().tobytes()

    encoded = independent_rht_encode(state, layer_index=4)
    restored = independent_rht_decode(encoded, layer_index=4)

    assert hashlib.sha256(digest).hexdigest() == (
        "3cc14ffaf1ad8de3d77a1d277cb027c3dee9360429c0746232780acc55d42f55"
    )
    torch.testing.assert_close(restored, state, rtol=2e-6, atol=1e-6)


def test_c4_and_c5_candidates_use_dense_construction_and_tie_selects_c5() -> None:
    s4 = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(401))
    record5 = ReplayRecord(
        normalized_key=torch.zeros((1, 2, 8), dtype=torch.float32),
        update=torch.zeros((1, 2, 8), dtype=torch.float32),
        log_decay=torch.zeros((1, 2), dtype=torch.float32),
    )
    raw_s5 = s4.clone()
    query_ema = torch.full((2, 8), 0.25, dtype=torch.float32)

    tied = construct_and_choose_boundary(
        s4=s4,
        raw_s5=raw_s5,
        record5=record5,
        query_ema=query_ema,
        pack_unpack=lambda state: state.clone(),
    )

    assert tied.tie is True
    assert tied.boundary == 5
    assert tied.cut4_risk == tied.cut5_risk == 0.0
    assert torch.equal(
        tied.cut4_candidate,
        dense_transition_from_record(
            s4,
            replace(
                record5,
                normalized_key=record5.normalized_key.to(torch.bfloat16).to(torch.float32),
                update=record5.update.to(torch.bfloat16).to(torch.float32),
            ),
        ),
    )


def test_lower_c4_risk_selects_c4_without_threshold() -> None:
    s4, record5, raw_s5 = _one_transition(torch.float32, seed=402)
    query_ema = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    calls = 0

    def asymmetric_codec(state: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            return state.clone()
        damaged = state.clone()
        damaged[..., 0] += 0.5
        return damaged

    audit = construct_and_choose_boundary(
        s4=s4,
        raw_s5=raw_s5,
        record5=record5,
        query_ema=query_ema,
        pack_unpack=asymmetric_codec,
    )

    assert calls == 2
    assert audit.cut4_risk < audit.cut5_risk
    assert audit.boundary == 4
    assert audit.tie is False


def test_boundary_candidates_run_through_independent_rht_physical_checkpoint() -> None:
    s4, record5, raw_s5 = _one_transition(torch.float16, seed=403)
    query_ema = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)

    def physical_checkpoint(state: torch.Tensor) -> torch.Tensor:
        encoded = independent_rht_encode(state, layer_index=0)
        rows = encoded.reshape(-1, encoded.shape[-1])
        benefit = q4_q8_physical_benefit(rows, query_ema.reshape(-1))
        selected = stable_descending_indices(benefit, 5)
        mask = torch.zeros(rows.shape[0], dtype=torch.bool)
        mask[selected] = True
        packed = pack_physical_q4_q8(rows, mask)
        restored = packed.dequantize().reshape_as(encoded)
        return independent_rht_decode(restored, layer_index=0)

    audit = construct_and_choose_boundary(
        s4=s4,
        raw_s5=raw_s5,
        record5=record5,
        query_ema=query_ema,
        pack_unpack=physical_checkpoint,
    )

    assert audit.boundary in (4, 5)
    assert audit.cut4_risk >= 0
    assert audit.cut5_risk >= 0
    assert torch.isfinite(audit.cut4_candidate).all()
    assert torch.isfinite(audit.cut5_candidate).all()


def test_full_buffer_compaction_retains_only_slot_four_or_clears_all() -> None:
    _, records, _ = _five_records(seed=501)
    full = store_records_bf16(records)

    cut4 = compact_full_buffer(full, boundary=4)
    cut5 = compact_full_buffer(full, boundary=5)

    assert int(cut4.valid_count.item()) == 1
    assert torch.equal(cut4.normalized_keys[0], full.normalized_keys[4])
    assert torch.equal(cut4.updates[0], full.updates[4])
    assert torch.equal(cut4.log_decays[0], full.log_decays[4])
    assert torch.count_nonzero(cut4.normalized_keys[1:]).item() == 0
    assert int(cut5.valid_count.item()) == 0
    assert torch.count_nonzero(cut5.normalized_keys).item() == 0
    assert torch.count_nonzero(cut5.updates).item() == 0
    assert torch.count_nonzero(cut5.log_decays).item() == 0
    assert int(full.valid_count.item()) == REPLAY_CAPACITY


def test_exact_full_storage_contract_and_owned_snapshot() -> None:
    contract = frozen_storage_contract()
    snapshot = canonical_empty_resident_snapshot()

    audited = audit_resident_snapshot(snapshot)

    assert contract == audited
    assert contract["checkpoint_bytes"] == CHECKPOINT_BYTES
    assert contract["resident_bytes"] == STATELEASE_BYTES
    assert contract["bits_per_state_element"] == pytest.approx(5.857109917534722)
    assert contract["fp32_compression_ratio"] == pytest.approx(5.463445355506431)
    assert resident_snapshot_digest(snapshot) == resident_snapshot_digest(snapshot)
    verify_reset_snapshot(snapshot)


@pytest.mark.parametrize(
    "tamper",
    [
        "extra_fp32",
        "extra_bf16",
        "split_fp16",
        "alias",
        "dtype",
        "count",
        "shape",
        "mask",
    ],
)
def test_resident_snapshot_tampering_fails_closed(tamper: str) -> None:
    snapshot = canonical_empty_resident_snapshot()
    if tamper == "extra_fp32":
        snapshot = replace(
            snapshot,
            extra_persistent_tensors=(
                ("persistent_fp32_state_mirror", torch.zeros((1, 16, 128, 128))),
            ),
        )
    elif tamper == "extra_bf16":
        snapshot = replace(
            snapshot,
            extra_persistent_tensors=(
                (
                    "persistent_bf16_state_mirror",
                    torch.zeros((1, 16, 128, 128), dtype=torch.bfloat16),
                ),
            ),
        )
    elif tamper == "split_fp16":
        snapshot = replace(
            snapshot,
            extra_persistent_tensors=(
                (
                    "persistent_state_left",
                    torch.zeros((1, 16, 64, 128), dtype=torch.float16),
                ),
                (
                    "persistent_state_right",
                    torch.zeros((1, 16, 64, 128), dtype=torch.float16),
                ),
            ),
        )
    elif tamper == "alias":
        snapshot = replace(snapshot, update_buffers=snapshot.normalized_key_buffers)
    elif tamper == "dtype":
        snapshot = replace(
            snapshot,
            valid_counts=(snapshot.valid_counts[0].to(torch.int64),),
        )
    elif tamper == "count":
        invalid = snapshot.valid_counts[0].clone()
        invalid[3] = 6
        snapshot = replace(snapshot, valid_counts=(invalid,))
    elif tamper == "shape":
        snapshot = replace(
            snapshot,
            checkpoint_low_payloads=(snapshot.checkpoint_low_payloads[0][:, :-1].contiguous(),),
        )
    else:
        snapshot = replace(
            snapshot,
            checkpoint_masks=(torch.zeros_like(snapshot.checkpoint_masks[0]),),
        )

    with pytest.raises((Stage0VerificationError, TypeError)):
        audit_resident_snapshot(snapshot)


def test_rollback_and_reset_checks_detect_mutation() -> None:
    before = canonical_empty_resident_snapshot()
    after = copy.deepcopy(before)
    assert_rollback_preserved(before, after)
    changed_key = after.normalized_key_buffers[0].clone()
    changed_key[0, 0, 0, 0] = 1
    changed = replace(after, normalized_key_buffers=(changed_key,))

    with pytest.raises(Stage0VerificationError, match="rollback changed"):
        assert_rollback_preserved(before, changed)
    with pytest.raises(Stage0VerificationError, match="nonzero normalized key"):
        verify_reset_snapshot(changed)


def test_q4_q8_physical_pools_ranking_ties_and_bytes() -> None:
    generator = torch.Generator().manual_seed(601)
    rows = torch.randn((9, 128), generator=generator)
    weights = torch.linspace(0.5, 1.5, 9)
    benefit = q4_q8_physical_benefit(rows, weights)
    selected = stable_descending_indices(benefit, 4)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[selected] = True

    packed = pack_physical_q4_q8(rows, mask)
    restored = packed.dequantize()

    assert packed.low_payload.shape == (5, 64)
    assert packed.high_payload.shape == (4, 128)
    assert packed.scales.shape == (9,)
    assert packed.precision_mask.shape == (2,)
    assert packed.storage_bytes == 5 * 64 + 4 * 128 + 9 * 2 + 2
    assert restored.shape == rows.shape
    assert torch.isfinite(restored).all()
    ties = stable_descending_indices(torch.ones(9), 4)
    assert torch.equal(ties, torch.tensor([0, 1, 2, 3]))
    with pytest.raises((Stage0VerificationError, TypeError)):
        replace(packed, high_payload=packed.high_payload.to(torch.float32)).dequantize()


def test_q4_q6_q8_exact_dp_physical_pools_and_invalid_code() -> None:
    d4 = torch.tensor([10.0, 10.0, 10.0, 10.0])
    d6 = torch.tensor([5.0, 5.0, 5.0, 5.0])
    d8 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    precision = allocate_exact_multibit(d4, d6, d8, marginal_steps=3)
    assert torch.equal(precision, torch.tensor([2, 1, 0, 0], dtype=torch.uint8))
    rows = torch.randn((4, 128), generator=torch.Generator().manual_seed(602))

    packed = pack_physical_q4_q6_q8(rows, precision)
    restored = packed.dequantize()

    assert packed.q4_payload.shape == (2, 64)
    assert packed.q6_payload.shape == (1, 96)
    assert packed.q8_payload.shape == (1, 128)
    assert packed.precision_codes.shape == (1,)
    assert packed.storage_bytes == 2 * 64 + 96 + 128 + 4 * 2 + 1
    assert restored.shape == rows.shape
    invalid_codes = packed.precision_codes.clone()
    invalid_codes[0] = invalid_codes[0] | (3 << 6)
    tampered = replace(packed, precision_codes=invalid_codes)
    with pytest.raises(Stage0VerificationError, match="invalid"):
        tampered.dequantize()


@pytest.mark.parametrize("only_code", [0, 1, 2])
def test_q4_q6_q8_empty_precision_pools_keep_canonical_shapes(
    only_code: int,
) -> None:
    rows = torch.randn((5, 128), generator=torch.Generator().manual_seed(660 + only_code))
    precision = torch.full((5,), only_code, dtype=torch.uint8)

    packed = pack_physical_q4_q6_q8(rows, precision)
    restored = packed.dequantize()

    assert packed.q4_payload.shape == ((5 if only_code == 0 else 0), 64)
    assert packed.q6_payload.shape == ((5 if only_code == 1 else 0), 96)
    assert packed.q8_payload.shape == ((5 if only_code == 2 else 0), 128)
    assert restored.shape == rows.shape
    assert torch.isfinite(restored).all()


def test_residual_q4_is_physical_and_reconstructs_transformed_residual() -> None:
    rows = torch.randn((7, 128), generator=torch.Generator().manual_seed(603))
    selected = torch.tensor([True, False, True, False, False, True, False])

    packed = pack_physical_residual_q4(rows, selected)
    restored = packed.dequantize()

    assert packed.base_payload.shape == (7, 64)
    assert packed.base_scales.shape == (7,)
    assert packed.residual_mask.shape == (1,)
    assert packed.residual_payload.shape == (3, 64)
    assert packed.residual_scales.shape == (3,)
    assert packed.storage_bytes == 7 * 64 + 7 * 2 + 1 + 3 * 64 + 3 * 2
    assert restored.shape == rows.shape
    assert torch.isfinite(restored).all()


def test_equal_total_byte_arithmetic_includes_explicit_padding() -> None:
    contracts = equal_byte_codec_contracts()

    assert contracts["expanded_q4_q8"]["allocation_units"] == EXPANDED_Q48_PROMOTIONS
    assert contracts["expanded_q4_q8"]["padding_bytes"] == 8
    assert contracts["q4_q6_q8"]["allocation_units"] == MULTIBIT_MARGINAL_STEPS
    assert contracts["q4_q6_q8"]["padding_bytes"] == 8
    assert contracts["residual_q4"]["allocation_units"] == RESIDUAL_Q4_ROWS
    assert contracts["residual_q4"]["padding_bytes"] == 26
    assert all(value["total_bytes"] == STATELEASE_BYTES for value in contracts.values())


def test_cc1_compatibility_requires_trajectory_metrics_plan_and_hashes() -> None:
    anchor = {
        "trajectory": torch.tensor([1.0, 2.0, 3.0]),
        "aligned_metrics": {"nll": 2.125, "top1": 0.75},
        "row_plan": {"layer_0": [1, 3, 5]},
        "hashes": {"plan": "abc", "trajectory": "def"},
    }
    candidate = {
        **anchor,
        "trajectory": anchor["trajectory"] + 1e-8,
        "aligned_metrics": {"nll": 2.125000001, "top1": 0.75},
    }
    verify_cc1_compatibility(candidate, anchor, rtol=1e-6, atol=1e-7)
    broken = dict(candidate)
    broken["row_plan"] = {"layer_0": [1, 3, 6]}
    with pytest.raises(Stage0VerificationError, match="row plans differ"):
        verify_cc1_compatibility(broken, anchor, rtol=1e-6, atol=1e-7)


def test_resume_integrity_accepts_exact_omission_and_rejects_tampering() -> None:
    kwargs = {
        "prior_identity_hashes": {"method": "a", "model": "b", "runtime": "c"},
        "resumed_identity_hashes": {"method": "a", "model": "b", "runtime": "c"},
        "expected_identities": ("one", "two", "three"),
        "completed_record_hashes": {"one": "h1"},
        "resumed_completed_record_hashes": {"one": "h1"},
        "resumed_remaining_identities": ("two", "three"),
    }
    verify_resume_integrity(**kwargs)
    tampered = dict(kwargs)
    tampered["resumed_identity_hashes"] = {"method": "changed", "model": "b", "runtime": "c"}
    with pytest.raises(Stage0VerificationError, match="resume changed"):
        verify_resume_integrity(**tampered)
    skipped = dict(kwargs)
    skipped["resumed_remaining_identities"] = ("three",)
    with pytest.raises(Stage0VerificationError, match="omit exactly"):
        verify_resume_integrity(**skipped)


def test_protected_window_guard_is_fail_closed() -> None:
    guard_protected_mbpp_window(stage="stage0")
    guard_protected_mbpp_window(stage="stagea", task_ids=("666",))
    guard_protected_mbpp_window(stage="stageb", ranked_indices=(0, 7, 16, 31))
    with pytest.raises(Stage0VerificationError, match=r"\[8, 16\)"):
        guard_protected_mbpp_window(stage="stageb", ranked_indices=(8,))
    with pytest.raises(Stage0VerificationError, match="only synthetic"):
        guard_protected_mbpp_window(stage="stage0", task_ids=("666",))
    with pytest.raises(Stage0VerificationError, match="restricted"):
        guard_protected_mbpp_window(stage="stagea", task_ids=("665",))


def test_cli_runs_without_quality_data_and_reports_exact_bytes() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--compact"],
        check=False,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parents[1],
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "verifier_self_test_pass"
    assert report["experiment_stage0_complete"] is False
    assert report["quality_data_accessed"] is False
    assert report["protected_mbpp_window_accessed"] is False
    assert report["storage"]["resident_bytes"] == STATELEASE_BYTES


@pytest.mark.parametrize("seed", range(8))
def test_independent_fast_multibit_allocator_matches_small_exact_oracle(
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(8000 + seed)
    distortions = torch.rand((3, 9), generator=generator)
    for marginal_steps in range(19):
        oracle = allocate_exact_multibit(
            distortions[0],
            distortions[1],
            distortions[2],
            marginal_steps=marginal_steps,
        )
        fast = _independent_allocate_multibit_fast(
            distortions[0],
            distortions[1],
            distortions[2],
            marginal_steps=marginal_steps,
        )
        assert torch.equal(fast, oracle)


def test_independent_fast_multibit_allocator_has_exact_binary_tie_rule() -> None:
    tied = torch.ones(6, dtype=torch.float64)
    for marginal_steps in range(13):
        actual = _independent_allocate_multibit_fast(
            tied,
            tied,
            tied,
            marginal_steps=marginal_steps,
        )
        expected = torch.zeros(6, dtype=torch.uint8)
        remaining = marginal_steps
        for row in range(6):
            code = min(2, remaining)
            expected[row] = code
            remaining -= code
        assert torch.equal(actual, expected)


def test_independent_fast_multibit_allocator_matches_adversarial_nonconvex_oracle() -> None:
    unit = 2.0**-40
    d4 = torch.tensor([9, 9, 12, 12, 7, 7], dtype=torch.float64) * unit
    d6 = torch.tensor([8, 8, 11, 4, 6, 2], dtype=torch.float64) * unit
    d8 = torch.tensor([0, 0, 10, 3, 1, 1], dtype=torch.float64) * unit
    for marginal_steps in range(13):
        oracle = allocate_exact_multibit(
            d4,
            d6,
            d8,
            marginal_steps=marginal_steps,
        )
        exact = _independent_allocate_multibit_fast(
            d4,
            d6,
            d8,
            marginal_steps=marginal_steps,
        )
        assert torch.equal(exact, oracle)


def test_cli_rejects_sha256_without_artifact() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--sha256", "unused.sha256", "--compact"],
        check=False,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parents[1],
    )
    assert completed.returncode != 0
    assert "--sha256 requires --artifact" in completed.stderr
