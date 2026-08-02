from __future__ import annotations

import base64
import copy
import importlib.util
import io
import json
import subprocess
import sys
import urllib.parse
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
FIXTURE_EXECUTION_ARTIFACTS = {
    "repository_source_manifest_file_sha256": b"fixture-source-manifest",
    "calibration_runtime_manifest_file_sha256": b"fixture-runtime-manifest",
    "model_file_manifest_file_sha256": b"fixture-model-manifest",
    "parquet_materialization_manifest_file_sha256": (
        REPOSITORY_ROOT / "research" / "experiment013-parquet-materializations.json"
    ).read_bytes(),
}
FIXTURE_RUNTIME_CONTEXT = {
    "base_runtime_root": REPOSITORY_ROOT / "fixture-base-runtime",
    "staged_interpreter": REPOSITORY_ROOT / "fixture-base-runtime" / "python.exe",
    "package_runtime_roots": {
        "fixture-packages": REPOSITORY_ROOT / "fixture-packages"
    },
    "package_import_paths": {"fixture-packages": "Lib/site-packages"},
}


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


def _fake_command(receipt: dict[str, Any]) -> dict[str, Any]:
    launcher = capture.RULER_LAUNCHER_PATH.read_bytes()
    return {
        "launcher_revision": capture.RULER_LAUNCHER_REVISION,
        "launcher_source_sha256": capture.sha256_bytes(launcher),
        "ruler_revision": resolver.RULER_REVISION,
        "config": receipt["config"],
        "configured_length": receipt["configured_length"],
        "seed": receipt["seed"],
        "argv": ["<RULER_PYTHON>", "fixture", receipt["filename"]],
        "shell": False,
    }


def _fake_ruler_content(config: str, seed: int) -> tuple[str, str, list[str]]:
    count = capture.RULER_REQUIRED_OUTPUT_COUNTS.get(config, 2)
    if config in capture.RULER_NIAH_CONFIGS:
        outputs = [f"{1_000_000 + seed + index:07d}" for index in range(count)]
        if config in {"niah_multiquery", "niah_multivalue"}:
            prompt = (
                "Some special magic numbers are hidden within the following text. "
                f"Memorize these values: {' '.join(outputs)}. "
                "What are all the special magic numbers for fixture-key mentioned in the "
                "provided text?"
            )
            answer_prefix = (
                " The special magic numbers for fixture-key mentioned in the provided text are"
            )
        else:
            prompt = (
                "A special magic number is hidden within the following text. "
                f"Memorize this value: {outputs[0]}. "
                "What is the special magic number for fixture-key mentioned in the provided text?"
            )
            answer_prefix = (
                " The special magic number for fixture-key mentioned in the provided text is"
            )
        return prompt, answer_prefix, outputs
    if config == "vt":
        outputs = [chr(65 + index) * 5 for index in range(count)]
        prompt = (
            "Memorize and track the chain(s) of variable assignment hidden in the following "
            f"text. {' '.join(outputs)}. Question: Find all variables that are assigned the "
            "value 12345 in the text above."
        )
        prefix = (
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "5 variables are assigned the value 12345, they are: "
        )
        return prompt, prefix, outputs
    if config == "cwe":
        outputs = [f"fixtureword{index}" for index in range(count)]
        prompt = (
            "Below is a numbered list of words. In these words, some appear more often than "
            f"others. {' '.join(outputs)}. Question: What are the 10 most common words in "
            "the above list?"
        )
        prefix = " Answer: The top 10 words that appear most often in the list are:"
        return prompt, prefix, outputs
    if config == "fwe":
        outputs = [chr(97 + index) * 6 for index in range(count)]
        prompt = (
            "Read the following coded text and track the frequency of each coded word. "
            f"{' '.join(outputs)}. What are the three most frequently appeared words in "
            "the above coded text?"
        )
        prefix = (
            " Answer: According to the coded text above, the three most frequently appeared "
            "words are:"
        )
        return prompt, prefix, outputs
    if config in {"qa_1", "qa_2"}:
        outputs = [f"fixture-answer-{seed}-{index}" for index in range(count)]
        prompt = (
            "Answer the question based on the given documents. Only give me the answer and do "
            "not output any other words.\n\nThe following are given documents.\n\n"
            "Fixture evidence.\n\nQuestion: Is the fixture valid?"
        )
        return prompt, " Answer:", outputs
    raise AssertionError(f"unhandled fake RULER config: {config}")


@pytest.fixture(autouse=True)
def _bind_fixture_generator_blobs(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _fake_generator_files()
    monkeypatch.setattr(
        capture,
        "RULER_GENERATOR_GIT_BLOBS",
        {path: capture._git_blob_sha1(content) for path, content in files.items()},
    )
    monkeypatch.setattr(
        capture,
        "RULER_EXPECTED_CORPORA",
        {"fixture-corpus.json": (17, _hash("fixture-corpus"))},
    )
    monkeypatch.setattr(
        capture,
        "RULER_EXPECTED_PACKAGE_RESOURCES",
        {"fixture/resource.txt": (19, _hash("fixture-resource"))},
    )
    tokenizer_files = {
        "tokenizer.json": b"fixture-tokenizer",
        "tokenizer_config.json": b"fixture-tokenizer-config",
    }
    monkeypatch.setattr(
        capture,
        "RULER_EXPECTED_TOKENIZER_ASSETS",
        {name: (len(data), capture.sha256_bytes(data)) for name, data in tokenizer_files.items()},
    )
    monkeypatch.setattr(
        capture,
        "RULER_COMMAND_MANIFEST_SHA256_BY_FILENAME",
        {
            item["filename"]: capture.sha256_bytes(
                capture.canonical_json_bytes(_fake_command(item))
            )
            for item in capture.required_ruler_receipts()
        },
    )
    strict_execution_decoder = capture._validate_execution_binding_artifacts
    strict_execution_authenticator = capture._authenticate_execution_binding_artifacts

    def decode_execution(artifacts: dict[str, bytes]) -> dict[str, str]:
        if artifacts == FIXTURE_EXECUTION_ARTIFACTS:
            return {
                field: capture.sha256_bytes(data)
                for field, data in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
            }
        return strict_execution_decoder(artifacts)

    monkeypatch.setattr(capture, "_validate_execution_binding_artifacts", decode_execution)

    def authenticate_execution(
        artifacts: dict[str, bytes],
        *,
        runtime_context: Any,
        previous: Any = None,
    ) -> Any:
        if artifacts == FIXTURE_EXECUTION_ARTIFACTS:
            normalized_context = capture._normalize_runtime_authentication_context(
                runtime_context
                if isinstance(runtime_context, dict)
                else {
                    "base_runtime_root": runtime_context.base_runtime_root,
                    "staged_interpreter": runtime_context.staged_interpreter,
                    "package_runtime_roots": runtime_context.package_runtime_roots,
                    "package_import_paths": runtime_context.package_import_paths,
                }
            )
            if previous is not None:
                assert normalized_context == previous.runtime_context
                return previous
            return capture._AuthenticatedExecutionBindings(
                bindings={
                    field: capture.sha256_bytes(data)
                    for field, data in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
                },
                source_manifest={},
                runtime_manifest=object(),
                model_manifest=object(),
                runner=object(),
                runtime_context=normalized_context,
            )
        return strict_execution_authenticator(
            artifacts,
            runtime_context=runtime_context,
            previous=previous,
        )

    monkeypatch.setattr(
        capture,
        "_authenticate_execution_binding_artifacts",
        authenticate_execution,
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
        self.manifest_mutator: Any = None

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

    def execution_binding_artifacts(self) -> dict[str, bytes]:
        return dict(FIXTURE_EXECUTION_ARTIFACTS)

    def runtime_authentication_context(self) -> dict[str, object]:
        return {
            "base_runtime_root": FIXTURE_RUNTIME_CONTEXT["base_runtime_root"],
            "staged_interpreter": FIXTURE_RUNTIME_CONTEXT["staged_interpreter"],
            "package_runtime_roots": dict(
                FIXTURE_RUNTIME_CONTEXT["package_runtime_roots"]  # type: ignore[arg-type]
            ),
            "package_import_paths": dict(
                FIXTURE_RUNTIME_CONTEXT["package_import_paths"]  # type: ignore[arg-type]
            ),
        }

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
        prompt, answer_prefix, outputs = _fake_ruler_content(config, seed)
        receipt: dict[str, Any] = {
            "schema": capture.RULER_RECEIPT_SCHEMA,
            "source_id": resolver.RULER_SOURCE_ID,
            "revision": resolver.RULER_REVISION,
            "category": category,
            "config": config,
            "configured_length": configured_length,
            "seed": seed,
            "sample_index": 0,
            "generator_reported_length": len(
                FakeTokenizer().encode(prompt + answer_prefix, add_special_tokens=False)
            )
            + capture.RULER_GENERATOR_TOKENS[config],
            "input": prompt,
            "answer_prefix": answer_prefix,
            "outputs": outputs,
            "auxiliary_files": [],
        }
        if self.receipt_mutator is not None:
            self.receipt_mutator(receipt)
        generator_manifest = capture._ruler_generator_manifest(self.generator_files)
        runtime_manifest = self._runtime_manifest()
        static_inputs = capture._expected_ruler_static_inputs(
            source_manifest=generator_manifest,
            runtime_manifest=runtime_manifest,
        )
        identity = next(
            item
            for item in capture.required_ruler_receipts()
            if item["category"] == category
            and item["config"] == config
            and item["configured_length"] == configured_length
            and item["seed"] == seed
        )
        command = _fake_command(identity)
        raw_data = self._raw_row(receipt)
        receipt["auxiliary_files"] = sorted(
            [
                *static_inputs,
                {
                    "name": "generator/command-manifest.json",
                    "sha256": capture.sha256_bytes(capture.canonical_json_bytes(command)),
                    "size_bytes": len(capture.canonical_json_bytes(command)),
                },
                {
                    "name": "generator/raw-validation.jsonl",
                    "sha256": capture.sha256_bytes(raw_data),
                    "size_bytes": len(raw_data),
                },
            ],
            key=lambda item: item["name"],
        )
        return receipt

    @staticmethod
    def _runtime_manifest() -> dict[str, Any]:
        packages = capture._ruler_runtime_packages()
        inventory: dict[str, Any] = {}
        for name, version in packages.items():
            record = f"{name}=={version}\n".encode()
            canonical = capture._canonical_distribution_name(name)
            inventory[name] = {
                "canonical_name": canonical,
                "version": version,
                "record_sha256": capture.sha256_bytes(record),
                "record_size_bytes": len(record),
                "files": [
                    {
                        "path": f"{canonical}-{version}.dist-info/RECORD",
                        "sha256": capture.sha256_bytes(record),
                        "size_bytes": len(record),
                    }
                ],
            }
        executable = b"fixture-python"
        return {
            "schema": capture.RULER_RUNTIME_MANIFEST_SCHEMA,
            "python": capture.RULER_RUNTIME_PYTHON_VERSION,
            "implementation": "cpython",
            "cache_tag": "cpython-311",
            "platform": "fixture-platform",
            "machine": "fixture-machine",
            "flags": {
                "ignore_environment": 1,
                "isolated": 1,
                "no_user_site": 1,
            },
            "startup_policy": dict(capture.RULER_SEALED_STARTUP_POLICY),
            "excluded_startup_files": [
                {
                    "name": name,
                    "sha256": digest,
                    "size_bytes": size,
                }
                for name, (size, digest) in sorted(
                    capture.RULER_EXCLUDED_VIRTUALENV_STARTUP_FILES.items()
                )
            ],
            "source_python": {
                "name": "source/python.exe",
                "sha256": capture.sha256_bytes(b"fixture-source-python"),
                "size_bytes": len(b"fixture-source-python"),
            },
            "source_pyvenv_config": {
                "name": "source/pyvenv.cfg",
                "sha256": capture.sha256_bytes(b"fixture-pyvenv-config"),
                "size_bytes": len(b"fixture-pyvenv-config"),
            },
            "python_runtime_files": [
                {
                    "name": name,
                    "sha256": capture.sha256_bytes(data),
                    "size_bytes": len(data),
                }
                for name, data in (
                    ("python.exe", b"fixture-runtime-python"),
                    ("python3.dll", b"fixture-python3-dll"),
                    ("python311.dll", b"fixture-python311-dll"),
                )
            ],
            "executable": {
                "name": "python.exe",
                "sha256": capture.sha256_bytes(executable),
                "size_bytes": len(executable),
            },
            "packages": packages,
            "installed_distributions": {
                capture._canonical_distribution_name(name): version
                for name, version in packages.items()
            },
            "distribution_file_inventory": inventory,
            "forbidden_modules": {name: False for name in capture.RULER_FORBIDDEN_RUNTIME_MODULES},
        }

    @staticmethod
    def _raw_row(receipt: dict[str, Any]) -> bytes:
        input_text = receipt["input"]
        first_output = receipt["outputs"][0]
        niah = receipt["config"] in capture.RULER_NIAH_CONFIGS
        row: dict[str, Any] = {
            "index": input_text.find(first_output) if niah else 0,
            "input": input_text,
            "outputs": receipt["outputs"],
            "length": receipt["generator_reported_length"],
            "length_w_model_temp": receipt["generator_reported_length"],
            "answer_prefix": receipt["answer_prefix"],
        }
        if niah:
            row["token_position_answer"] = len(
                FakeTokenizer().encode(
                    input_text[: input_text.find(first_output)], add_special_tokens=False
                )
            )
        return capture.canonical_json_bytes(row)

    def ruler_receipt_bytes(
        self, *, category: str, config: str, configured_length: int, seed: int
    ) -> bytes:
        return capture.canonical_json_bytes(
            self.ruler_receipt(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )
        )

    def ruler_generation_manifest_bytes(self) -> bytes:
        generator_manifest = capture._ruler_generator_manifest(self.generator_files)
        runtime_manifest = self._runtime_manifest()
        static_inputs = capture._expected_ruler_static_inputs(
            source_manifest=generator_manifest,
            runtime_manifest=runtime_manifest,
        )
        results = []
        for identity in capture.required_ruler_receipts():
            receipt_bytes = self.ruler_receipt_bytes(
                category=identity["category"],
                config=identity["config"],
                configured_length=identity["configured_length"],
                seed=identity["seed"],
            )
            receipt = json.loads(receipt_bytes)
            command = _fake_command(identity)
            command_bytes = capture.canonical_json_bytes(command)
            raw_data = self._raw_row(receipt)
            results.append(
                {
                    "category": identity["category"],
                    "command_manifest": command,
                    "command_manifest_file": {
                        "name": "generator/command-manifest.json",
                        "sha256": capture.sha256_bytes(command_bytes),
                        "size_bytes": len(command_bytes),
                    },
                    "config": identity["config"],
                    "configured_length": identity["configured_length"],
                    "filename": identity["filename"],
                    "generator_reported_length": receipt["generator_reported_length"],
                    "phase": identity["phase"],
                    "raw_validation_base64": base64.b64encode(raw_data).decode("ascii"),
                    "raw_validation_file": {
                        "name": "generator/raw-validation.jsonl",
                        "sha256": capture.sha256_bytes(raw_data),
                        "size_bytes": len(raw_data),
                    },
                    "seed": identity["seed"],
                    "sha256": capture.sha256_bytes(receipt_bytes),
                    "size_bytes": len(receipt_bytes),
                }
            )
        launcher = next(
            item
            for item in static_inputs
            if item["name"] == "launcher/generate_static_q468_ruler_receipts.py"
        )
        manifest = {
            "schema": capture.RULER_GENERATION_MANIFEST_SCHEMA,
            "launcher_revision": capture.RULER_LAUNCHER_REVISION,
            "launcher_source": launcher,
            "ruler_revision": resolver.RULER_REVISION,
            "source_manifest": generator_manifest,
            "source_manifest_sha256": capture.sha256_bytes(
                capture.canonical_json_bytes(generator_manifest)
            ),
            "runtime_manifest": runtime_manifest,
            "runtime_manifest_sha256": capture.sha256_bytes(
                capture.canonical_json_bytes(runtime_manifest)
            ),
            "static_inputs": static_inputs,
            "receipt_count": 20,
            "receipts": results,
        }
        if self.manifest_mutator is not None:
            self.manifest_mutator(manifest)
        return capture.canonical_json_bytes(manifest)

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


def _fixture_execution_authentication(
    runtime_context: Any,
    *,
    runner: object | None = None,
) -> Any:
    return capture._AuthenticatedExecutionBindings(
        bindings={
            field: capture.sha256_bytes(data)
            for field, data in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
        },
        source_manifest={},
        runtime_manifest=object(),
        model_manifest=object(),
        runner=object() if runner is None else runner,
        runtime_context=runtime_context,
    )


@pytest.mark.parametrize(
    "flags",
    [
        {"ignore_environment": 0, "isolated": 1, "no_user_site": 1},
        {"ignore_environment": 1, "isolated": 0, "no_user_site": 1},
        {"ignore_environment": 1, "isolated": 1, "no_user_site": 0},
        {"ignore_environment": True, "isolated": 1, "no_user_site": 1},
        {
            "ignore_environment": 1,
            "isolated": 1,
            "no_user_site": 1,
            "unexpected": 1,
        },
    ],
)
def test_ruler_runtime_manifest_rejects_nonisolated_or_extra_flags(
    flags: dict[str, int],
) -> None:
    manifest = FakeSource._runtime_manifest()
    manifest["flags"] = flags

    with pytest.raises(ValueError, match="isolation flags drifted"):
        capture._normalize_ruler_runtime_manifest(manifest)


def test_ruler_runtime_manifest_rejects_boolean_numeric_startup_attestation() -> None:
    manifest = FakeSource._runtime_manifest()
    manifest["startup_policy"]["no_site"] = True

    with pytest.raises(ValueError, match="sealed-startup policy drifted"):
        capture._normalize_ruler_runtime_manifest(manifest)


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


def test_execution_artifacts_are_authenticated_before_and_after_all_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource()

    def authenticate(
        _artifacts: Any,
        *,
        runtime_context: Any,
        previous: Any = None,
    ) -> Any:
        source.accesses.append(
            "execution_auth:post" if previous is not None else "execution_auth:pre"
        )
        if previous is not None:
            return previous
        return _fixture_execution_authentication(runtime_context)

    monkeypatch.setattr(capture, "_authenticate_execution_binding_artifacts", authenticate)
    capture.capture_identity_input(phase="calibration", source=source)

    assert source.accesses[0] == "execution_auth:pre"
    assert source.accesses[1] == "source_heads"
    assert source.accesses[-2:] == ["source_heads", "execution_auth:post"]


def test_pre_capture_execution_authentication_failure_touches_no_data_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource()

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("pinned model Hub metadata authentication failed")

    monkeypatch.setattr(capture, "_authenticate_execution_binding_artifacts", reject)

    with pytest.raises(ValueError, match="model Hub metadata"):
        capture.capture_identity_input(phase="calibration", source=source)
    assert source.accesses == []


def test_post_capture_execution_authentication_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource()

    def authenticate(
        _artifacts: Any,
        *,
        runtime_context: Any,
        previous: Any = None,
    ) -> Any:
        if previous is not None:
            raise ValueError("execution-binding artifacts changed during capture")
        return _fixture_execution_authentication(runtime_context)

    monkeypatch.setattr(capture, "_authenticate_execution_binding_artifacts", authenticate)

    with pytest.raises(ValueError, match="changed during capture"):
        capture.capture_identity_input(phase="calibration", source=source)
    assert "humaneval_projection" in source.accesses
    assert source.accesses[-1] == "source_heads"


def test_preloaded_calibration_runner_is_rejected() -> None:
    sentinel = object()
    sys.modules[capture._CALIBRATION_RUNNER_MODULE_NAME] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="preloaded"):
            capture._load_calibration_runner_module()
    finally:
        if sys.modules.get(capture._CALIBRATION_RUNNER_MODULE_NAME) is sentinel:
            sys.modules.pop(capture._CALIBRATION_RUNNER_MODULE_NAME, None)


def test_public_calibration_materialization_is_the_exact_capture_with_tokens() -> None:
    captured = capture.capture_identity_input(phase="calibration", source=FakeSource())
    shared_result, shared_tokens = capture._capture_identity_input_with_tokens(
        phase="calibration",
        source=FakeSource(),
        collect_tokens=True,
    )
    source = FakeSource()
    materialized = capture.materialize_calibration_identity_sequences(source=source)

    assert capture.canonical_json_bytes(shared_result) == capture.canonical_json_bytes(captured)
    assert len(shared_tokens) == 160
    assert materialized.capture_input_sha256 == capture.sha256_bytes(
        capture.canonical_json_bytes(captured)
    )
    assert capture.canonical_json_bytes(materialized.identity_records) == (
        capture.canonical_json_bytes(captured["records"])
    )
    assert len(materialized.sequences) == len(materialized.by_identity_record_sha256) == 160
    assert source.head_calls == 2
    assert source.accesses.count("tokenizer_material") == 1
    assert source.accesses.count("ruler_generator_files") == 1

    for sequence, expected_record in zip(materialized.sequences, captured["records"], strict=True):
        record = sequence.identity_record
        assert capture.canonical_json_bytes(record) == capture.canonical_json_bytes(expected_record)
        assert isinstance(sequence.prompt_token_ids, tuple)
        assert isinstance(sequence.target_token_ids, tuple)
        assert sequence.sequence_token_ids == (
            sequence.prompt_token_ids + sequence.target_token_ids
        )
        assert record["prompt_token_ids_sha256"] == capture._token_hash(sequence.prompt_token_ids)
        assert record["target_token_ids_sha256"] == capture._token_hash(sequence.target_token_ids)
        assert record["sequence_token_ids_sha256"] == capture._token_hash(
            sequence.sequence_token_ids
        )
        assert record["sequence_length"] == len(sequence.sequence_token_ids)
        assert record["token_span"] == {
            "prefill_start": 0,
            "prefill_stop": len(sequence.prompt_token_ids),
            "scored_start": len(sequence.prompt_token_ids),
            "scored_stop": len(sequence.sequence_token_ids),
            "cache_exposed_start": len(sequence.sequence_token_ids),
            "cache_exposed_stop": len(sequence.sequence_token_ids),
        }
        assert (bool(sequence.target_token_ids)) is (record["family"] == "mbpp")

    for sequence in reversed(materialized.sequences):
        assert materialized.lookup(sequence.identity_record_sha256) is sequence
    first_copy = materialized.sequences[0].identity_record
    first_copy["token_span"]["prefill_start"] = 99
    assert materialized.sequences[0].identity_record["token_span"]["prefill_start"] == 0
    assert len(materialized.token_sequence_manifest_sha256) == 64


def test_public_calibration_materialization_returns_no_raw_content() -> None:
    materialized = capture.materialize_calibration_identity_sequences(source=FakeSource())
    record_bytes = b"".join(
        capture.canonical_json_bytes(record) for record in materialized.identity_records
    )

    assert b"Return 601" not in record_bytes
    assert b"RULER retrieval" not in record_bytes
    assert b"answer_prefix" not in record_bytes
    assert b"auxiliary_files" not in record_bytes
    assert b"source_payload" not in record_bytes
    assert b"formatted_payload" not in record_bytes


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: setattr(source, "drift_after_capture", True),
            "post-capture source HEAD",
        ),
        (
            lambda source: source.extra_tokenizer_files.update({"model.safetensors": b"forbidden"}),
            "model weight-like file is forbidden",
        ),
        (
            lambda source: setattr(
                source,
                "manifest_mutator",
                lambda manifest: manifest["receipts"].pop(),
            ),
            "all 20 receipt results",
        ),
    ],
)
def test_public_calibration_materialization_preserves_capture_failures(
    mutate: Any, message: str
) -> None:
    source = FakeSource()
    mutate(source)

    with pytest.raises(ValueError, match=message):
        capture.materialize_calibration_identity_sequences(source=source)


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
    assert decoded.execution_bindings == {
        field: capture.sha256_bytes(data)
        for field, data in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
    }
    assert (
        decoded.parquet_materialization_manifest_file_sha256
        == resolver.PARQUET_MATERIALIZATION_MANIFEST_FILE_SHA256
    )
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


@pytest.mark.parametrize(
    ("config", "mutate", "message"),
    [
        (
            "niah_multiquery",
            lambda receipt: receipt.update(
                {
                    "input": receipt["input"].replace(
                        "What are all the special magic numbers",
                        "Which values",
                    )
                }
            ),
            "input task markers drifted",
        ),
        (
            "vt",
            lambda receipt: receipt.update({"answer_prefix": " Answer:"}),
            "answer-prefix boundaries drifted",
        ),
        (
            "fwe",
            lambda receipt: receipt.update({"outputs": ["INVALID", "bbbbbb", "cccccc"]}),
            "output format drifted",
        ),
        (
            "vt",
            lambda receipt: receipt.update(
                {"outputs": ["ZZZZZ", *receipt["outputs"][1:]]}
            ),
            "required answer is absent",
        ),
    ],
)
def test_ruler_receipt_replays_frozen_task_semantics(
    config: str,
    mutate: Any,
    message: str,
) -> None:
    category = next(
        category
        for category, configs in capture.RULER_CONFIGS_BY_CATEGORY.items()
        if config in configs
    )
    receipt = FakeSource().ruler_receipt(
        category=category,
        config=config,
        configured_length=4_096,
        seed=2_339,
    )
    mutate(receipt)

    with pytest.raises(ValueError, match=message):
        capture._normalize_ruler_receipt(
            receipt,
            category=category,
            config=config,
            configured_length=4_096,
            seed=2_339,
        )


def test_ruler_receipt_rejects_boolean_sample_index() -> None:
    receipt = FakeSource().ruler_receipt(
        category="retrieval",
        config="niah_multiquery",
        configured_length=4_096,
        seed=2_339,
    )
    receipt["sample_index"] = False

    with pytest.raises(ValueError, match="sample_index must be an integer"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_339,
        )


@pytest.mark.parametrize("field", ["index", "token_position_answer"])
def test_ruler_raw_row_rejects_boolean_numeric_fields(field: str) -> None:
    source = FakeSource()
    receipt = source.ruler_receipt(
        category="retrieval",
        config="niah_multiquery",
        configured_length=4_096,
        seed=2_339,
    )
    row = json.loads(source._raw_row(receipt))
    row[field] = False

    with pytest.raises(ValueError, match="must be an integer"):
        capture._verify_ruler_raw_row(
            capture.canonical_json_bytes(row),
            receipt=receipt,
            tokenizer=FakeTokenizer(),
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


def test_incomplete_ruler_generation_manifest_is_rejected() -> None:
    source = FakeSource()
    source.manifest_mutator = lambda manifest: manifest["receipts"].pop()

    with pytest.raises(ValueError, match="all 20 receipt results"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_ruler_command_manifest_argv_tamper_is_rejected() -> None:
    source = FakeSource()
    source.manifest_mutator = lambda manifest: manifest["receipts"][0]["command_manifest"][
        "argv"
    ].append("--unbound")

    with pytest.raises(ValueError, match="command argv drifted"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_ruler_receipt_rejects_arbitrary_auxiliary_claim() -> None:
    source = FakeSource()
    original = source.ruler_receipt_bytes

    def with_extra_auxiliary(**kwargs: Any) -> bytes:
        value = json.loads(original(**kwargs))
        value["auxiliary_files"].append(
            {"name": "unbound/claim.txt", "sha256": "7" * 64, "size_bytes": 1}
        )
        value["auxiliary_files"].sort(key=lambda item: item["name"])
        return capture.canonical_json_bytes(value)

    source.ruler_receipt_bytes = with_extra_auxiliary  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="auxiliary inventory drifted"):
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


def test_live_capture_source_contains_no_mutable_dataset_viewer_paths() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"/rows"' not in source
    assert '"/parquet"' not in source
    assert "@~parquet" not in source
    assert "project_experiment013_parquet_columns" in source
    assert "read_experiment013_parquet_row" in source


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


def test_execution_bindings_are_derived_from_verified_artifact_bytes() -> None:
    captured = capture.capture_identity_input(phase="calibration", source=FakeSource())

    assert captured["execution_bindings"] == {
        field: capture.sha256_bytes(data)
        for field, data in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
    }
    source = FakeSource()
    source.execution_binding_artifacts = lambda: {  # type: ignore[method-assign]
        **FIXTURE_EXECUTION_ARTIFACTS,
        "repository_source_manifest_file_sha256": b"not-json",
    }
    with pytest.raises(ValueError, match="repository source manifest"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_execution_artifact_decoders_run_before_file_hash_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recurquant import experiment013_source

    source_payload: dict[str, Any] = {
        "schema": experiment013_source.EXPERIMENT013_SOURCE_MANIFEST_SCHEMA,
        "profile": experiment013_source.EXPERIMENT013_SOURCE_MANIFEST_PROFILE,
        "object_format": "sha1",
        "source_commit": "a" * 40,
        "repository_binding": {
            "schema": experiment013_source.EXPERIMENT013_REPOSITORY_BINDING_SCHEMA,
            "worktree_layout": "primary",
            **{field: True for field in experiment013_source._TRUE_BINDING_FIELDS},
        },
        "paths": [
            {
                "path": path,
                "mode": "100644",
                "git_blob_oid": "b" * 40,
                "index_blob_oid": "b" * 40,
                "worktree_blob_oid": "b" * 40,
                "raw_sha256": "c" * 64,
            }
            for path in experiment013_source.EXPERIMENT013_SOURCE_PATHS
        ],
    }
    source_payload["canonical_manifest_sha256"] = (
        experiment013_source.canonical_experiment013_source_manifest_sha256(source_payload)
    )
    source_bytes = experiment013_source._canonical_json_bytes(source_payload)
    runtime_bytes = b"strict-runtime-manifest\n"
    model_bytes = b"strict-model-manifest\n"

    class FakeRunner:
        @staticmethod
        def parse_calibration_runtime_manifest(data: bytes) -> Any:
            if data != runtime_bytes:
                raise ValueError("calibration runtime manifest is invalid")
            return SimpleNamespace(file_sha256=capture.sha256_bytes(data))

        @staticmethod
        def parse_model_file_manifest(data: bytes) -> Any:
            if data != model_bytes:
                raise ValueError("model file manifest is invalid")
            return SimpleNamespace(
                file_sha256=capture.sha256_bytes(data),
                model_id=resolver.PRIMARY_MODEL_ID,
                revision=resolver.PRIMARY_MODEL_REVISION,
                transformers_version=resolver.TRANSFORMERS_VERSION,
            )

    monkeypatch.setattr(capture, "_load_calibration_runner_module", lambda: FakeRunner())
    artifacts = {
        "repository_source_manifest_file_sha256": source_bytes,
        "calibration_runtime_manifest_file_sha256": runtime_bytes,
        "model_file_manifest_file_sha256": model_bytes,
        "parquet_materialization_manifest_file_sha256": FIXTURE_EXECUTION_ARTIFACTS[
            "parquet_materialization_manifest_file_sha256"
        ],
    }

    assert capture._validate_execution_binding_artifacts(artifacts) == {
        field: capture.sha256_bytes(data) for field, data in sorted(artifacts.items())
    }
    with pytest.raises(ValueError, match="calibration runtime manifest"):
        capture._validate_execution_binding_artifacts(
            {**artifacts, "calibration_runtime_manifest_file_sha256": b"{}\n"}
        )
    with pytest.raises(ValueError, match="model file manifest"):
        capture._validate_execution_binding_artifacts(
            {**artifacts, "model_file_manifest_file_sha256": b"{}\n"}
        )


def test_point_of_use_authentication_rechecks_source_runtime_modules_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    package_root = tmp_path / "packages"
    (package_root / "Lib" / "site-packages").mkdir(parents=True)
    runtime_context = capture._normalize_runtime_authentication_context(
        {
            "base_runtime_root": tmp_path / "runtime",
            "staged_interpreter": tmp_path / "runtime" / "python.exe",
            "package_runtime_roots": {"packages": package_root},
            "package_import_paths": {"packages": "Lib/site-packages"},
        }
    )
    artifacts = {
        "repository_source_manifest_file_sha256": b"source\n",
        "calibration_runtime_manifest_file_sha256": b"runtime\n",
        "model_file_manifest_file_sha256": b"model\n",
        "parquet_materialization_manifest_file_sha256": b"parquet\n",
    }
    bindings = {
        field: capture.sha256_bytes(data) for field, data in sorted(artifacts.items())
    }
    source_manifest = {"paths": []}

    class SourceModule:
        @staticmethod
        def verify_experiment013_source_manifest(manifest: Any, *, repo_root: Path) -> Any:
            assert repo_root == REPOSITORY_ROOT
            events.append("source")
            return dict(manifest)

        @staticmethod
        def verify_loaded_experiment013_recurquant_modules(
            _manifest: Any,
            _root: Path,
            names: Any,
        ) -> None:
            assert names == (
                "recurquant.experiment013_source",
                "recurquant.experiment013_parquet",
            )
            events.append("loaded-source-parquet")

    runtime_manifest = SimpleNamespace(
        package_roots=(SimpleNamespace(name="packages", import_path="Lib/site-packages"),)
    )

    class Runner:
        @staticmethod
        def authenticate_calibration_runtime(
            _manifest: Any,
            *,
            base_runtime_root: Path,
            package_roots: Any,
            interpreter_path: Path,
        ) -> Any:
            assert base_runtime_root == runtime_context.base_runtime_root
            assert package_roots == runtime_context.package_runtime_roots
            assert interpreter_path == runtime_context.staged_interpreter
            events.append("runtime")
            return SimpleNamespace(
                manifest_file_sha256=bindings[
                    "calibration_runtime_manifest_file_sha256"
                ]
            )

        @staticmethod
        def capture_model_file_manifest_from_hub(
            model_id: str,
            revision: str,
            *,
            transformers_version: str,
        ) -> bytes:
            assert (model_id, revision, transformers_version) == (
                resolver.PRIMARY_MODEL_ID,
                resolver.PRIMARY_MODEL_REVISION,
                resolver.TRANSFORMERS_VERSION,
            )
            events.append("model")
            return artifacts["model_file_manifest_file_sha256"]

    runner = Runner()
    decoded = capture._DecodedExecutionBindingArtifacts(
        bindings=bindings,
        source_manifest=source_manifest,
        runtime_manifest=runtime_manifest,
        model_manifest=object(),
        source_module=SourceModule(),
        parquet_module=object(),
    )

    def load_runner() -> Any:
        events.append("load-runner")
        sys.modules[capture._CALIBRATION_RUNNER_MODULE_NAME] = runner  # type: ignore[assignment]
        return runner

    def decode(_artifacts: Any, *, runner: Any) -> Any:
        assert runner is not None
        events.append("decode")
        return decoded

    monkeypatch.setattr(capture, "_load_calibration_runner_module", load_runner)
    monkeypatch.setattr(capture, "_decode_execution_binding_artifacts", decode)
    monkeypatch.setattr(
        capture,
        "_verify_loaded_runner_source",
        lambda *_args: events.append("runner-source"),
    )
    try:
        first = capture._authenticate_execution_binding_artifacts(
            artifacts,
            runtime_context=runtime_context,
        )
        capture._authenticate_execution_binding_artifacts(
            artifacts,
            runtime_context=runtime_context,
            previous=first,
        )
    finally:
        if sys.modules.get(capture._CALIBRATION_RUNNER_MODULE_NAME) is runner:
            sys.modules.pop(capture._CALIBRATION_RUNNER_MODULE_NAME, None)

    assert events == [
        "load-runner",
        "decode",
        "source",
        "loaded-source-parquet",
        "runner-source",
        "runtime",
        "model",
        "decode",
        "source",
        "loaded-source-parquet",
        "runner-source",
        "runtime",
        "model",
    ]


def test_loaded_calibration_runner_source_is_bound_to_source_manifest() -> None:
    runner = SimpleNamespace(__file__=str(capture.CALIBRATION_RUNNER_PATH))
    manifest = {
        "paths": [
            {
                "path": "scripts/run_static_q468_calibration.py",
                "raw_sha256": capture.sha256_bytes(
                    capture.CALIBRATION_RUNNER_PATH.read_bytes()
                ),
            }
        ]
    }

    capture._verify_loaded_runner_source(runner, manifest)
    manifest["paths"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runner source bytes drifted"):
        capture._verify_loaded_runner_source(runner, manifest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda context: context.update({"extra": Path("unused")}),
        lambda context: context.update({"base_runtime_root": "not-a-path"}),
        lambda context: context.update({"package_runtime_roots": {"packages": False}}),
        lambda context: context.update(
            {"package_import_paths": {"packages": "../site-packages"}}
        ),
        lambda context: context.update(
            {"package_import_paths": {"packages": "Lib\\site-packages"}}
        ),
        lambda context: context.update(
            {"package_import_paths": {"different": "Lib/site-packages"}}
        ),
    ],
)
def test_runtime_authentication_context_rejects_noncanonical_values(mutate: Any) -> None:
    context = copy.deepcopy(FIXTURE_RUNTIME_CONTEXT)
    mutate(context)

    with pytest.raises(ValueError):
        capture._normalize_runtime_authentication_context(context)


def test_cli_runtime_context_rejects_duplicate_package_names() -> None:
    with pytest.raises(ValueError, match="duplicate name"):
        capture._parse_named_cli_values(
            ["packages=first", "packages=second"],
            context="--package-root",
            paths=True,
        )


def test_cli_requires_all_four_execution_artifact_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-source-manifest"):
        capture.main(
            [
                "--phase",
                "calibration",
                "--ruler-receipt-dir",
                str(tmp_path / "receipts"),
                "--dry-run",
                "--base-runtime-root",
                str(tmp_path / "runtime"),
                "--staged-interpreter",
                str(tmp_path / "runtime" / "python.exe"),
                "--package-root",
                f"packages={tmp_path / 'packages'}",
                "--package-import-path",
                "packages=Lib/site-packages",
            ]
        )


def test_live_source_probes_only_the_frozen_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed_hf: list[tuple[str, str, str]] = []
    observed_urls: list[str] = []

    class FakeApi:
        @staticmethod
        def model_info(repo_id: str, *, revision: str) -> Any:
            observed_hf.append(("model", repo_id, revision))
            return SimpleNamespace(sha=revision)

        @staticmethod
        def dataset_info(repo_id: str, *, revision: str) -> Any:
            observed_hf.append(("dataset", repo_id, revision))
            return SimpleNamespace(sha=revision)

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(request: Any, timeout: int) -> Response:
        assert timeout == 30
        observed_urls.append(request.full_url)
        revision = urllib.parse.unquote(request.full_url.rsplit("/", 1)[1])
        return Response(json.dumps({"sha": revision}).encode())

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setattr(capture.urllib.request, "urlopen", fake_urlopen)
    source = capture.LiveCaptureSource(
        cache_dir=tmp_path / "cache", ruler_receipt_dir=tmp_path / "receipts"
    )

    assert source.source_heads() == capture.EXPECTED_SOURCE_HEADS
    assert all(revision in capture.EXPECTED_SOURCE_HEADS.values() for _, _, revision in observed_hf)
    assert any(url.endswith(resolver.RULER_REVISION) for url in observed_urls)
    assert any(url.endswith(resolver.EVALPLUS_SOURCE_REVISION) for url in observed_urls)
    assert all("default_branch" not in url for url in observed_urls)


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
