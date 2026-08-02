from __future__ import annotations

import base64
import copy
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from recurquant.static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    build_static_rht_q468_policy,
    serialize_static_rht_q468_policy,
)
from recurquant.static_q468_calibration import (
    FROZEN_SOURCE_TENSOR_CONTRACT,
    CalibrationAggregate,
    build_frozen_calibration_score_artifact,
    build_frozen_split_half_stability_artifact,
    calibration_identity_record_manifest_sha256,
    deserialize_calibration_score_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "capture_static_q468_identity_input.py"
SPEC = importlib.util.spec_from_file_location("capture_static_q468_identity_input", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)
resolver = capture.resolver
FIXTURE_BINDING_ARTIFACT = b"verified-fixture-binding-artifact"


def test_capture_script_imports_in_direct_cli_process() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Capture a calibration or Stage-A Experiment 013 identity input" in (
        completed.stdout.replace("\n", " ")
    )


def _hash(label: str) -> str:
    return capture.sha256_bytes(label.encode())


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return [*prefix, *(2 + (ord(character) % 251) for character in text)]


def _fake_generator_files() -> dict[str, bytes]:
    config_yaml = "".join(
        f"{config}:\n  task: fixture\n" for config in capture.RULER_ALL_CONFIGS
    ).encode()
    return {
        path: (
            config_yaml
            if path == "scripts/synthetic.yaml"
            else f"fixture source for {path}\n".encode()
        )
        for path in capture.RULER_GENERATOR_GIT_BLOBS
    }


@pytest.fixture(autouse=True)
def _bind_fixture_generator_blobs(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _fake_generator_files()
    monkeypatch.setattr(
        capture,
        "RULER_GENERATOR_GIT_BLOBS",
        {path: capture._git_blob_sha1(content) for path, content in files.items()},
    )
    strict_binding_decoder = resolver.deserialize_stage_a_calibration_binding_artifact

    def decode_binding(data: bytes) -> object:
        if data == FIXTURE_BINDING_ARTIFACT:
            return SimpleNamespace(
                binding={
                    key: _hash(f"binding-{key}")
                    for key in sorted(resolver.CALIBRATION_BINDING_FIELDS)
                }
            )
        return strict_binding_decoder(data)

    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        decode_binding,
    )


class FakeSource:
    def __init__(self) -> None:
        self.accesses: list[str] = []
        self.head_calls = 0
        self.drift_after_capture = False
        self.short_pg19_ids: set[str] = set()
        self.extra_tokenizer_files: dict[str, bytes] = {}
        self.generator_files = _fake_generator_files()
        self.receipt_mutator: Any = None

    def source_heads(self) -> dict[str, str]:
        self.accesses.append("source_heads")
        self.head_calls += 1
        heads = dict(capture.EXPECTED_SOURCE_HEADS)
        if self.drift_after_capture and self.head_calls > 1:
            heads["pg19"] = "f" * 40
        return heads

    def tokenizer_material(self) -> Any:
        self.accesses.append("tokenizer_material")
        return capture.TokenizerMaterial(
            tokenizer=FakeTokenizer(),
            tokenizer_class="FixtureTokenizer",
            transformers_version=resolver.TRANSFORMERS_VERSION,
            files={
                "tokenizer.json": b"fixture-tokenizer",
                "tokenizer_config.json": b"fixture-tokenizer-config",
                **self.extra_tokenizer_files,
            },
            model_weights_loaded=False,
        )

    def mbpp_train_rows(self) -> tuple[dict[str, Any], ...]:
        self.accesses.append("mbpp_train_rows")
        return tuple(
            {
                "task_id": task_id,
                "text": f"Return {task_id}.",
                "code": f"def answer():\n    return {task_id}\n",
                "test_list": [f"assert answer() == {task_id}"],
                "test_setup_code": "",
                "challenge_test_list": [],
            }
            for task_id in range(601, 975)
        )

    def pg19_projection(self, split: str) -> tuple[Any, ...]:
        self.accesses.append(f"pg19_projection:{split}")
        count = 13_684 if split == "train" else 50
        return tuple(
            capture.ProjectionRow(f"https://pg19.example/{split}/{offset}", offset)
            for offset in range(count)
        )

    def pg19_text(self, split: str, url: str) -> str:
        if url in self.short_pg19_ids:
            width = 2_300 if split == "train" else 4_200
        else:
            width = 2_420 if split == "train" else 4_340
        return chr(65 + (int(url.rsplit("/", 1)[1]) % 20)) * width

    def pg19_row(self, split: str, *, offset: int, expected_url: str) -> dict[str, Any]:
        self.accesses.append(f"pg19_row:{split}:{offset}")
        return {"url": expected_url, "text": self.pg19_text(split, expected_url)}

    def ruler_generator_files(self) -> dict[str, bytes]:
        self.accesses.append("ruler_generator_files")
        return dict(self.generator_files)

    def ruler_receipt(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> dict[str, Any]:
        self.accesses.append(f"ruler_receipt:{category}:{config}:{configured_length}:{seed}")
        prompt = f"RULER {category} {config} {configured_length} {seed}."
        output_count = capture.RULER_REQUIRED_OUTPUT_COUNTS.get(config, 2)
        receipt: dict[str, Any] = {
            "schema": capture.RULER_RECEIPT_SCHEMA,
            "source_id": resolver.RULER_SOURCE_ID,
            "revision": resolver.RULER_REVISION,
            "category": category,
            "config": config,
            "configured_length": configured_length,
            "seed": seed,
            "sample_index": 0,
            "generator_reported_length": len(prompt) + 32,
            "input": prompt,
            "answer_prefix": " Answer:",
            "outputs": [f"result-{config}-{seed}-{index}" for index in range(output_count)],
            "auxiliary_files": [
                {
                    "name": f"fixture/{config}.txt",
                    "sha256": _hash(f"aux-{config}"),
                    "size_bytes": 100 + len(config),
                }
            ],
        }
        if self.receipt_mutator is not None:
            self.receipt_mutator(receipt)
        return receipt

    def humaneval_projection(self) -> tuple[Any, ...]:
        self.accesses.append("humaneval_projection")
        return tuple(capture.ProjectionRow(f"HumanEval/{offset}", offset) for offset in range(164))

    def humaneval_row(self, *, offset: int, expected_task_id: str) -> dict[str, Any]:
        self.accesses.append(f"humaneval_row:{offset}")
        return {
            "task_id": expected_task_id,
            "prompt": f"def task_{offset}(x):\n",
            "canonical_solution": "    return x\n" * 20,
            "entry_point": f"task_{offset}",
            "test": f"assert task_{offset}(1) == 1",
        }


def _binding() -> bytes:
    return FIXTURE_BINDING_ARTIFACT


def _frozen_aggregate(
    *,
    half: bool,
    identity_manifest_sha256: str,
    sequence_manifest_sha256: str,
) -> CalibrationAggregate:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    row_axis = torch.arange(rows, dtype=torch.float64)
    broad_counts = (
        (("mbpp", 64), ("pg19", 8), ("ruler", 8))
        if half
        else (("mbpp", 128), ("pg19", 16), ("ruler", 16))
    )
    ruler_count = 2 if half else 4
    return CalibrationAggregate(
        d4=4.0 + row_axis / rows,
        d6=2.0 + row_axis / (2 * rows),
        d8=1.0 + row_axis / (4 * rows),
        family_sequence_counts=broad_counts,
        ruler_category_sequence_counts=tuple(
            (category, ruler_count) for category in resolver.RULER_CATEGORIES
        ),
        sequence_score_manifest_sha256=sequence_manifest_sha256,
        source_contract=FROZEN_SOURCE_TENSOR_CONTRACT,
        identity_record_manifest_sha256=identity_manifest_sha256,
    )


def test_calibration_capture_is_deterministic_and_resolver_compatible() -> None:
    first = capture.capture_identity_input(phase="calibration", source=FakeSource())
    second = capture.capture_identity_input(phase="calibration", source=FakeSource())

    assert capture.canonical_json_bytes(first) == capture.canonical_json_bytes(second)
    candidate = resolver.build_candidate(
        first, expected_revisions=resolver.FROZEN_DATASET_REVISIONS
    )
    assert candidate["evidence"]["record_count"] == 160
    counts = {
        family: sum(row["family"] == family for row in first["records"])
        for family in resolver.DATASET_KEYS
    }
    assert counts == {"mbpp": 128, "pg19": 16, "ruler": 16, "humaneval_plus": 0}
    assert first["model_weights_loaded"] is False
    assert all(
        row["ruler_category"] is None
        and row["configured_length"] is None
        and row["generator_receipt_sha256"] is None
        for row in first["records"]
        if row["family"] != "ruler"
    )
    assert all(
        row["token_span"]["cache_exposed_start"]
        == row["token_span"]["scored_stop"]
        == row["token_span"]["cache_exposed_stop"]
        for row in first["records"]
    )


def test_frozen_calibration_identity_decoder_recomputes_capture_lineage() -> None:
    captured = capture.capture_identity_input(phase="calibration", source=FakeSource())
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
    )
    candidate_bytes = resolver.canonical_json_bytes(candidate)
    frozen = resolver.promote_candidate(
        candidate,
        candidate_file_sha256=resolver.sha256_bytes(candidate_bytes),
    )
    frozen_bytes = resolver.canonical_json_bytes(frozen)

    decoded = resolver.deserialize_frozen_calibration_identity_artifact(frozen_bytes)

    assert decoded.file_sha256 == resolver.sha256_bytes(frozen_bytes)
    assert decoded.canonical_evidence_sha256 == frozen["canonical_evidence_sha256"]
    assert len(decoded.records) == 160
    assert len(decoded.assignment) == 160
    assert (
        decoded.assignment_sha256
        == frozen["evidence"]["calibration_split_half"]["assignment_sha256"]
    )
    assert all(
        row["identity_record_sha256"] == resolver.identity_record_sha256(row)
        for row in decoded.records
    )

    tampered = copy.deepcopy(frozen)
    tampered["evidence"]["records"][0]["anchor_positions"][0] += 1
    tampered["evidence"]["content_manifest_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"]["records"])
    )
    tampered["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"])
    )
    with pytest.raises(ValueError, match="not canonical"):
        resolver.deserialize_frozen_calibration_identity_artifact(
            resolver.canonical_json_bytes(tampered)
        )


def test_stage_a_binding_is_derived_from_identity_scores_split_and_policies() -> None:
    captured = capture.capture_identity_input(phase="calibration", source=FakeSource())
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
    )
    candidate_bytes = resolver.canonical_json_bytes(candidate)
    frozen = resolver.promote_candidate(
        candidate,
        candidate_file_sha256=resolver.sha256_bytes(candidate_bytes),
    )
    identity_bytes = resolver.canonical_json_bytes(frozen)
    identity = resolver.deserialize_frozen_calibration_identity_artifact(identity_bytes)
    full_identity_manifest = calibration_identity_record_manifest_sha256(identity.records)
    half_identity_manifests = resolver._identity_half_record_manifests(identity)
    full_sequence_manifest = "c" * 64
    full_aggregate = _frozen_aggregate(
        half=False,
        identity_manifest_sha256=full_identity_manifest,
        sequence_manifest_sha256=full_sequence_manifest,
    )
    score_bytes = build_frozen_calibration_score_artifact(
        full_aggregate,
        calibration_identity_sha256=identity.file_sha256,
    )
    score = deserialize_calibration_score_artifact(score_bytes)
    split_bytes = build_frozen_split_half_stability_artifact(
        _frozen_aggregate(
            half=True,
            identity_manifest_sha256=half_identity_manifests["a"],
            sequence_manifest_sha256="a" * 64,
        ),
        _frozen_aggregate(
            half=True,
            identity_manifest_sha256=half_identity_manifests["b"],
            sequence_manifest_sha256="b" * 64,
        ),
        identity_file_sha256=identity.file_sha256,
        canonical_identity_sha256=identity.canonical_evidence_sha256,
        resolver_assignment_sha256=identity.assignment_sha256,
        full_sequence_score_manifest_sha256=full_sequence_manifest,
        full_calibration_scores_sha256=score.calibration_scores_sha256,
    )
    policy_arguments = {
        "d4": full_aggregate.d4,
        "d6": full_aggregate.d6,
        "d8": full_aggregate.d8,
        "geometry": FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
        "calibration_manifest_sha256": full_sequence_manifest,
        "identity_artifact_sha256": identity.file_sha256,
        "tokenizer_manifest_sha256": identity.tokenizer_manifest_sha256,
        "source_commit": "f" * 40,
        "calibration_scores_sha256": score.calibration_scores_sha256,
    }
    policy27030_bytes = serialize_static_rht_q468_policy(
        build_static_rht_q468_policy(
            **policy_arguments,
            marginal_steps=FROZEN_STATIC_Q468_ABLATION_STEPS,
            method_id=STATIC_Q468_ABLATION_METHOD,
        )
    )
    policy29334_bytes = serialize_static_rht_q468_policy(
        build_static_rht_q468_policy(
            **policy_arguments,
            marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
            method_id=STATIC_Q468_PRIMARY_METHOD,
        )
    )
    binding_bytes = resolver.build_stage_a_calibration_binding_artifact(
        frozen_identity_artifact=identity_bytes,
        calibration_score_artifact=score_bytes,
        split_half_stability_artifact=split_bytes,
        static_k27030_policy_artifact=policy27030_bytes,
        static_k29334_policy_artifact=policy29334_bytes,
    )

    verified = resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)

    assert capture._normalize_calibration_binding(binding_bytes) == verified.binding
    assert verified.binding == {
        "calibration_identity_file_sha256": identity.file_sha256,
        "calibration_score_artifact_file_sha256": resolver.sha256_bytes(score_bytes),
        "split_half_stability_artifact_file_sha256": resolver.sha256_bytes(split_bytes),
        "static_k27030_policy_file_sha256": resolver.sha256_bytes(policy27030_bytes),
        "static_k29334_policy_file_sha256": resolver.sha256_bytes(policy29334_bytes),
    }

    tampered = json.loads(binding_bytes)
    encoded_policy = tampered["evidence"]["dependencies_base64"]["static_k29334_policy_artifact"]
    policy_payload = bytearray(base64.b64decode(encoded_policy))
    policy_payload[0] ^= 1
    tampered["evidence"]["dependencies_base64"]["static_k29334_policy_artifact"] = base64.b64encode(
        policy_payload
    ).decode("ascii")
    tampered["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"])
    )
    with pytest.raises(ValueError):
        resolver.deserialize_stage_a_calibration_binding_artifact(
            resolver.canonical_json_bytes(tampered)
        )


def test_stage_a_capture_uses_exact_schedules_and_token_caps() -> None:
    captured = capture.capture_identity_input(
        phase="stage_a", source=FakeSource(), calibration_binding=_binding()
    )
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=_binding(),
    )
    assert candidate["evidence"]["record_count"] == 12
    ruler_rows = [row for row in captured["records"] if row["family"] == "ruler"]
    assert {
        (
            row["ruler_category"],
            row["config"],
            row["configured_length"],
            row["seed"],
        )
        for row in ruler_rows
    } == set(resolver.RULER_STAGE_A_SCHEDULE)
    assert all(row["sequence_length"] < row["configured_length"] for row in ruler_rows)
    assert all(
        row["token_span"]["cache_exposed_stop"] - row["token_span"]["cache_exposed_start"]
        == row["token_span"]["scored_stop"] - row["token_span"]["scored_start"] - 1
        for row in ruler_rows
    )
    human_rows = [row for row in captured["records"] if row["family"] == "humaneval_plus"]
    assert len(human_rows) == 4
    assert all(
        row["token_span"]["scored_stop"] - row["token_span"]["scored_start"] == 128
        for row in human_rows
    )
    assert all(
        row["token_span"]["cache_exposed_stop"] - row["token_span"]["cache_exposed_start"] == 127
        for row in human_rows
    )


def test_ruler_stage_a_target_includes_all_required_outputs_and_selects_one_qa_alternative() -> (
    None
):
    required, required_semantics = capture._ruler_stage_a_target(
        category="retrieval",
        config="niah_multiquery",
        outputs=("11", "22", "33", "44"),
    )
    alternative, alternative_semantics = capture._ruler_stage_a_target(
        category="question_answering",
        config="qa_1",
        outputs=("first answer", "alternate answer"),
    )

    assert required == "11, 22, 33, 44"
    assert required_semantics == "all_required_outputs_comma_space_v1"
    assert alternative == "first answer"
    assert alternative_semantics == "first_pinned_alternative_reference_v1"


def test_ruler_receipt_required_output_cardinality_and_uniqueness_fail_closed() -> None:
    source = FakeSource()
    receipt = source.ruler_receipt(
        category="retrieval",
        config="niah_multiquery",
        configured_length=4_096,
        seed=2_339,
    )
    receipt["outputs"] = ["only-one"]
    with pytest.raises(ValueError, match="exactly 4 required outputs"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_339,
        )

    receipt["outputs"] = ["same"] * 4
    with pytest.raises(ValueError, match="must be unique"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_339,
        )


def test_pg19_ranks_all_ids_before_text_and_skips_ineligible_rows() -> None:
    source = FakeSource()
    projection = source.pg19_projection("train")
    source.accesses.clear()
    ranked = sorted(
        projection,
        key=lambda item: (
            resolver.selection_sha256(resolver.PG19_TRAIN_NAMESPACE, item.canonical_id),
            item.canonical_id,
        ),
    )
    source.short_pg19_ids.update(item.canonical_id for item in ranked[:2])
    captured = capture.capture_identity_input(phase="calibration", source=source)

    projection_index = source.accesses.index("pg19_projection:train")
    first_text_index = next(
        index for index, value in enumerate(source.accesses) if value.startswith("pg19_row:")
    )
    assert projection_index < first_text_index
    accessed_offsets = [
        int(value.rsplit(":", 1)[1])
        for value in source.accesses
        if value.startswith("pg19_row:train:")
    ]
    assert accessed_offsets == [item.offset for item in ranked[:18]]
    selected = [row for row in captured["records"] if row["family"] == "pg19"]
    assert {row["canonical_id"] for row in selected}.isdisjoint(source.short_pg19_ids)


def test_pg19_stage_a_uses_frozen_hashed_4224_token_slice() -> None:
    source = FakeSource()
    captured = capture.capture_identity_input(
        phase="stage_a", source=source, calibration_binding=_binding()
    )
    tokenizer = FakeTokenizer()
    rows = [row for row in captured["records"] if row["family"] == "pg19"]
    for row in rows:
        url = row["canonical_id"]
        full_ids = tokenizer.encode(source.pg19_text("validation", url), add_special_tokens=False)
        start = capture._segment_start(
            namespace=capture.PG19_VALIDATION_SEGMENT_NAMESPACE,
            canonical_id=url,
            token_count=len(full_ids),
            width=4_224,
        )
        selected = full_ids[start : start + 4_224]
        assert row["prompt_token_ids_sha256"] == capture._token_hash(selected[:4_096])
        assert row["target_token_ids_sha256"] == capture._token_hash(selected[4_096:])
        assert row["token_span"] == {
            "prefill_start": 0,
            "prefill_stop": 4_096,
            "scored_start": 4_096,
            "scored_stop": 4_224,
            "cache_exposed_start": 4_097,
            "cache_exposed_stop": 4_224,
        }


def test_stage_a_ruler_requires_two_continuation_tokens() -> None:
    source = FakeSource()
    source.receipt_mutator = lambda receipt: (
        receipt.update({"outputs": ["x"]}) if receipt["config"] == "qa_1" else None
    )

    with pytest.raises(ValueError, match="continuation must contain at least two tokens"):
        capture.capture_identity_input(
            phase="stage_a", source=source, calibration_binding=_binding()
        )


def test_stage_a_humaneval_requires_two_continuation_tokens() -> None:
    source = FakeSource()
    original = source.humaneval_row

    def one_token_solution(*, offset: int, expected_task_id: str) -> dict[str, Any]:
        row = original(offset=offset, expected_task_id=expected_task_id)
        row["canonical_solution"] = "x"
        return row

    source.humaneval_row = one_token_solution  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="continuation must contain at least two tokens"):
        capture.capture_identity_input(
            phase="stage_a", source=source, calibration_binding=_binding()
        )


@pytest.mark.parametrize("phase", ["stage_b", "stage_c"])
def test_protected_phases_fail_before_any_source_access(phase: str) -> None:
    source = FakeSource()

    with pytest.raises(PermissionError, match="before source access"):
        capture.capture_identity_input(phase=phase, source=source)

    assert source.accesses == []


def test_cli_rejects_protected_phase_before_paths_are_read(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(PermissionError, match="before file or source access"):
        capture.main(
            [
                "--phase",
                "stage_b",
                "--ruler-receipt-dir",
                str(tmp_path / "missing-receipts"),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_source_head_drift_fails_after_capture() -> None:
    source = FakeSource()
    source.drift_after_capture = True

    with pytest.raises(ValueError, match="post-capture source HEAD"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_tokenizer_weight_file_is_rejected_before_dataset_content() -> None:
    source = FakeSource()
    source.extra_tokenizer_files["model.safetensors"] = b"not-a-real-weight"

    with pytest.raises(ValueError, match="model weight-like file is forbidden"):
        capture.capture_identity_input(phase="calibration", source=source)

    assert not any(value == "mbpp_train_rows" for value in source.accesses)


def test_live_tokenizer_load_uses_only_authenticated_files_from_an_isolated_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "shared-snapshot"
    snapshot.mkdir()
    expected = {
        "tokenizer.json": b'{"version":"fixture"}',
        "tokenizer_config.json": b'{"tokenizer_class":"Fixture"}',
    }
    for name, data in expected.items():
        (snapshot / name).write_bytes(data)
    (snapshot / "tokenizer.model").write_bytes(b"unbound-stray-snapshot-file")
    observed: dict[str, object] = {}

    class FakeApi:
        @staticmethod
        def list_repo_files(_repo_id: str, *, revision: str) -> list[str]:
            assert revision == resolver.PRIMARY_MODEL_REVISION
            return [*expected, "tokenizer.model", "model.safetensors"]

    def fake_download(*, repo_id: str, filename: str, revision: str, cache_dir: Path) -> str:
        assert repo_id == resolver.PRIMARY_MODEL_ID
        assert revision == resolver.PRIMARY_MODEL_REVISION
        assert cache_dir == (tmp_path / "cache").resolve()
        return str(snapshot / filename)

    class IsolatedTokenizer:
        pass

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> IsolatedTokenizer:
            isolated = Path(path)
            observed["path"] = isolated
            observed["files"] = sorted(
                item.relative_to(isolated).as_posix()
                for item in isolated.rglob("*")
                if item.is_file()
            )
            observed["kwargs"] = kwargs
            return IsolatedTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    source = capture.LiveCaptureSource(
        cache_dir=tmp_path / "cache",
        ruler_receipt_dir=tmp_path / "receipts",
    )

    material = source.tokenizer_material()

    assert material.files == expected
    assert observed["files"] == sorted(expected)
    assert observed["path"] != snapshot
    assert observed["kwargs"] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert not Path(observed["path"]).exists()


def test_ruler_receipt_category_drift_fails_closed() -> None:
    source = FakeSource()
    source.receipt_mutator = lambda receipt: receipt.update({"category": "retrieval"})

    with pytest.raises(ValueError, match="receipt category drifted"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_ruler_generator_source_tamper_is_rejected() -> None:
    source = FakeSource()
    path = next(path for path in source.generator_files if path != "scripts/synthetic.yaml")
    source.generator_files[path] += b"tamper"

    with pytest.raises(ValueError, match="generator Git blob drifted"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_duplicate_projection_identity_fails_closed() -> None:
    source = FakeSource()
    original = source.pg19_projection

    def duplicated(split: str) -> tuple[Any, ...]:
        rows = list(original(split))
        rows[1] = capture.ProjectionRow(rows[0].canonical_id, rows[1].offset)
        return tuple(rows)

    source.pg19_projection = duplicated  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="duplicate identity or offset"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_atomic_write_is_canonical_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "identity-input.json"
    first = capture.canonical_json_bytes({"value": 1})
    second = capture.canonical_json_bytes({"value": 2})

    capture.atomic_write_no_overwrite(path, first)
    assert path.read_bytes() == first
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        capture.atomic_write_no_overwrite(path, second)
    assert path.read_bytes() == first
    assert not list(tmp_path.glob("*.tmp"))


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        capture._strict_json(b'{"a":1,"a":2}', context="fixture")


def test_active_parquet_manifest_excludes_stale_repo_tree_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "parquet_files": [
            {
                "dataset": resolver.PG19_DATASET_ID,
                "config": "default",
                "split": "train",
                "url": (
                    "https://huggingface.co/datasets/emozilla/pg19/resolve/"
                    "refs%2Fconvert%2Fparquet/default/partial-train/0001.parquet"
                ),
                "filename": "0001.parquet",
                "size": 200,
            },
            {
                "dataset": resolver.PG19_DATASET_ID,
                "config": "default",
                "split": "train",
                "url": (
                    "https://huggingface.co/datasets/emozilla/pg19/resolve/"
                    "refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet"
                ),
                "filename": "0000.parquet",
                "size": 100,
            },
        ],
        "repo_tree_files": [
            "data/train-00022-of-00023-stale-sibling.parquet",
        ],
    }

    class Response(io.BytesIO):
        headers = {"x-revision": resolver.PG19_REVISION}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(_request: object, timeout: int) -> Response:
        assert timeout == 60
        return Response(json.dumps(payload).encode())

    monkeypatch.setattr(capture.urllib.request, "urlopen", fake_urlopen)
    aliases = capture.LiveCaptureSource._active_parquet_aliases(
        dataset_id=resolver.PG19_DATASET_ID,
        revision=resolver.PG19_REVISION,
        config="default",
        split="train",
    )

    assert aliases == (
        "datasets/emozilla/pg19@~parquet/default/partial-train/0000.parquet",
        "datasets/emozilla/pg19@~parquet/default/partial-train/0001.parquet",
    )
    assert all("stale" not in alias for alias in aliases)


def test_calibration_binding_requires_verified_artifact_and_is_normalized() -> None:
    binding = _binding()
    captured = capture.capture_identity_input(
        phase="stage_a", source=FakeSource(), calibration_binding=binding
    )

    assert captured["calibration_binding"] == {
        key: _hash(f"binding-{key}") for key in sorted(resolver.CALIBRATION_BINDING_FIELDS)
    }
    with pytest.raises(ValueError, match="verified artifact byte string"):
        capture.capture_identity_input(
            phase="stage_a",
            source=FakeSource(),
            calibration_binding={
                key: _hash(f"binding-{key}") for key in resolver.CALIBRATION_BINDING_FIELDS
            },  # type: ignore[arg-type]
        )


def test_capture_output_contains_no_raw_model_or_weight_claim() -> None:
    captured = capture.capture_identity_input(phase="calibration", source=FakeSource())
    serialized = capture.canonical_json_bytes(copy.deepcopy(captured))

    assert b"model.safetensors" not in serialized
    assert captured["model_weights_loaded"] is False


def test_required_ruler_receipt_inventory_is_exact_and_unique() -> None:
    receipts = capture.required_ruler_receipts()

    assert len(receipts) == 20
    assert len({item["filename"] for item in receipts}) == 20
    assert sum(item["phase"] == "calibration" for item in receipts) == 16
    assert sum(item["phase"] == "stage_a" for item in receipts) == 4
    assert receipts[0]["filename"] == ("retrieval__niah_multiquery__l2048__s12339.json")
    assert receipts[-1]["filename"] == ("question_answering__qa_1__l4096__s2339.json")
