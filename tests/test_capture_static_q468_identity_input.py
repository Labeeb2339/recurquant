from __future__ import annotations

import base64
import copy
import importlib.util
import io
import json
import subprocess
import sys
import urllib.parse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import recurquant.static_q468_calibration as calibration
from recurquant import experiment013_source
from recurquant.static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    FROZEN_STATIC_Q468_ABLATION_STEPS,
    FROZEN_STATIC_Q468_PRIMARY_STEPS,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    build_static_rht_q468_policy,
    serialize_static_rht_q468_policy,
)
from recurquant.static_q468_calibration import (
    FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
    FROZEN_SOURCE_TENSOR_CONTRACT,
    FROZEN_UNWEIGHTED_MSE_PROFILE,
    CalibrationAggregate,
    ComparatorAggregate,
    build_frozen_calibration_score_artifact,
    build_frozen_comparator_score_artifact,
    build_frozen_split_half_stability_artifact,
    calibration_identity_record_manifest_sha256,
    deserialize_calibration_score_artifact,
    deserialize_comparator_score_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "capture_static_q468_identity_input.py"
SPEC = importlib.util.spec_from_file_location("capture_static_q468_identity_input", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)
resolver = capture.resolver
FIXTURE_GIT_EXECUTABLE = experiment013_source.authenticate_git_executable().path
FIXTURE_BINDING_ARTIFACT = b"verified-fixture-binding-artifact"
FIXTURE_FROZEN_CALIBRATION_IDENTITY_ARTIFACT = b"verified-fixture-calibration-identity"
FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT = b"verified-fixture-capture-provenance"
FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256 = capture.sha256_bytes(
    FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT
)
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
    "git_executable": FIXTURE_GIT_EXECUTABLE,
    "package_runtime_roots": {"fixture-packages": REPOSITORY_ROOT / "fixture-packages"},
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


def _runner_source_manifest() -> dict[str, object]:
    runner_bytes = capture.CALIBRATION_RUNNER_PATH.read_bytes()
    return {
        "paths": [
            {
                "path": "scripts/run_static_q468_calibration.py",
                "raw_sha256": capture.sha256_bytes(runner_bytes),
            }
        ]
    }


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
    fixture_ruler_generation_manifest_file_sha256 = capture.sha256_bytes(
        FakeSource().ruler_generation_manifest_bytes()
    )
    fixture_binding_prefix = FIXTURE_BINDING_ARTIFACT + b":"

    def fixture_binding_manifest_sha256(data: bytes) -> str:
        if data == FIXTURE_BINDING_ARTIFACT:
            return fixture_ruler_generation_manifest_file_sha256
        if data.startswith(fixture_binding_prefix):
            return capture._require_sha256(
                data[len(fixture_binding_prefix) :].decode("ascii"),
                context="fixture RULER generation manifest file SHA-256",
            )
        raise ValueError("not a fixture Stage-A binding")

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
                    "git_executable": runtime_context.git_executable,
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
    strict_calibration_identity_decoder = resolver.deserialize_frozen_calibration_identity_artifact
    fixture_calibration_identity: object | None = None

    def decode_calibration_identity(
        data: bytes,
        *,
        expected_file_sha256: str | None = None,
    ) -> object:
        nonlocal fixture_calibration_identity
        if data != FIXTURE_FROZEN_CALIBRATION_IDENTITY_ARTIFACT:
            return strict_calibration_identity_decoder(
                data,
                expected_file_sha256=expected_file_sha256,
            )
        if expected_file_sha256 is not None and expected_file_sha256 != capture.sha256_bytes(data):
            raise ValueError("fixture calibration identity differs from its explicit SHA-256")
        if fixture_calibration_identity is None:
            source = FakeSource()
            fixture_calibration_identity = SimpleNamespace(
                records=tuple(
                    {
                        "family": "ruler",
                        "ruler_category": item["category"],
                        "config": item["config"],
                        "configured_length": item["configured_length"],
                        "seed": item["seed"],
                        "generator_receipt_sha256": capture.sha256_bytes(
                            source.ruler_receipt_bytes(
                                category=item["category"],
                                config=item["config"],
                                configured_length=item["configured_length"],
                                seed=item["seed"],
                            )
                        ),
                    }
                    for item in capture.required_ruler_receipts()
                    if item["phase"] == "calibration"
                )
            )
        return fixture_calibration_identity

    monkeypatch.setattr(
        resolver,
        "deserialize_frozen_calibration_identity_artifact",
        decode_calibration_identity,
    )

    def decode_binding(
        data: bytes,
        *,
        expected_file_sha256: str | None = None,
    ) -> object:
        if data == FIXTURE_BINDING_ARTIFACT or data.startswith(fixture_binding_prefix):
            if expected_file_sha256 is not None and expected_file_sha256 != capture.sha256_bytes(
                data
            ):
                raise ValueError("fixture Stage-A binding differs from its explicit SHA-256")
            return SimpleNamespace(
                binding={
                    key: _hash(f"binding-{key}")
                    for key in sorted(resolver.CALIBRATION_BINDING_FIELDS)
                },
                execution_bindings={
                    field: capture.sha256_bytes(payload)
                    for field, payload in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
                },
                calibration_dependencies={
                    "frozen_identity_artifact": FIXTURE_FROZEN_CALIBRATION_IDENTITY_ARTIFACT,
                },
                ruler_generation_manifest_file_sha256=(fixture_binding_manifest_sha256(data)),
            )
        return strict_binding_decoder(
            data,
            expected_file_sha256=expected_file_sha256,
        )

    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        decode_binding,
    )
    strict_capture_decoder = resolver.deserialize_stage_a_capture_provenance_receipt
    latest_identity_input_sha256: str | None = None

    def decode_capture_provenance(
        data: bytes,
        *,
        expected_file_sha256: str,
        calibration_binding_artifact: bytes,
        expected_identity_input_file_sha256: str | None = None,
    ) -> object:
        nonlocal latest_identity_input_sha256
        if data != FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT:
            return strict_capture_decoder(
                data,
                expected_file_sha256=expected_file_sha256,
                calibration_binding_artifact=calibration_binding_artifact,
                expected_identity_input_file_sha256=expected_identity_input_file_sha256,
            )
        assert expected_file_sha256 == FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
        assert calibration_binding_artifact == FIXTURE_BINDING_ARTIFACT or (
            calibration_binding_artifact.startswith(fixture_binding_prefix)
        )
        if expected_identity_input_file_sha256 is not None:
            latest_identity_input_sha256 = expected_identity_input_file_sha256
        assert latest_identity_input_sha256 is not None
        return SimpleNamespace(
            file_sha256=FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256,
            identity_input_file_sha256=latest_identity_input_sha256,
            ruler_generation_manifest_file_sha256=(
                fixture_binding_manifest_sha256(calibration_binding_artifact)
            ),
        )

    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_capture_provenance_receipt",
        decode_capture_provenance,
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
            "git_executable": FIXTURE_RUNTIME_CONTEXT["git_executable"],
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
        ebook_base = 100_000 if split == "train" else 200_000
        return tuple(
            capture.ProjectionRow(
                f"http://www.gutenberg.org/ebooks/{ebook_base + offset}",
                offset,
            )
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


def _binding(ruler_generation_manifest_file_sha256: str | None = None) -> bytes:
    if ruler_generation_manifest_file_sha256 is None:
        return FIXTURE_BINDING_ARTIFACT
    return (
        FIXTURE_BINDING_ARTIFACT
        + b":"
        + capture._require_sha256(
            ruler_generation_manifest_file_sha256,
            context="fixture RULER generation manifest file SHA-256",
        ).encode("ascii")
    )


def _binding_for_source(source: FakeSource) -> bytes:
    manifest_sha256 = capture.sha256_bytes(source.ruler_generation_manifest_bytes())
    source.accesses.clear()
    return _binding(manifest_sha256)


def _stage_a_capture_provenance_kwargs() -> dict[str, object]:
    return {
        "stage_a_capture_provenance_receipt": (FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT),
        "expected_stage_a_capture_provenance_receipt_sha256": (
            FIXTURE_STAGE_A_CAPTURE_PROVENANCE_RECEIPT_SHA256
        ),
    }


def _frozen_stage_a_identity(
    source: FakeSource | None = None,
    *,
    calibration_binding: bytes | None = None,
) -> bytes:
    binding = _binding() if calibration_binding is None else calibration_binding
    captured = capture.capture_identity_input(
        phase="stage_a",
        source=FakeSource() if source is None else source,
        calibration_binding=binding,
    )
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=binding,
        **_stage_a_capture_provenance_kwargs(),
    )
    candidate_bytes = resolver.canonical_json_bytes(candidate)
    frozen = resolver.promote_candidate(
        candidate,
        candidate_file_sha256=resolver.sha256_bytes(candidate_bytes),
        calibration_binding_artifact=binding,
        **_stage_a_capture_provenance_kwargs(),
    )
    return resolver.canonical_json_bytes(frozen)


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


def _frozen_comparator_aggregate(
    selector_profile: str,
    *,
    identity_manifest_sha256: str,
) -> ComparatorAggregate:
    rows = FROZEN_QWEN35_STATIC_Q468_GEOMETRY.total_rows
    row_axis = torch.arange(rows, dtype=torch.float64) / rows
    offset = 0.0 if selector_profile == FROZEN_UNWEIGHTED_MSE_PROFILE else 0.125
    provisional = ComparatorAggregate(
        selector_profile=selector_profile,  # type: ignore[arg-type]
        d4=4.0 + offset + row_axis,
        d6=2.0 + offset + row_axis / 2,
        d8=1.0 + offset + row_axis / 4,
        family_sequence_counts=(("mbpp", 128), ("pg19", 16), ("ruler", 16)),
        ruler_category_sequence_counts=tuple(
            (category, 4) for category in calibration.RULER_CATEGORY_ORDER
        ),
        position_manifest_sha256=_hash(f"positions:{selector_profile}"),
        sequence_score_manifest_sha256=_hash(f"sequences:{selector_profile}"),
        identity_record_manifest_sha256=identity_manifest_sha256,
        aggregate_scores_sha256="0" * 64,
    )
    return replace(
        provisional,
        aggregate_scores_sha256=calibration._comparator_aggregate_score_sha256(provisional),
    )


def test_calibration_capture_is_deterministic_and_resolver_compatible() -> None:
    first = capture.capture_identity_input(phase="calibration", source=FakeSource())
    second = capture.capture_identity_input(phase="calibration", source=FakeSource())

    assert capture.canonical_json_bytes(first) == capture.canonical_json_bytes(second)
    candidate = resolver.build_candidate(
        first, expected_revisions=resolver.FROZEN_DATASET_REVISIONS
    )
    assert candidate["evidence"]["record_count"] == 160
    assert capture.CAPTURE_VERSION == resolver.RESOLVER_VERSION == 9
    assert capture.RULER_FORMATTER_FROZEN_CAPTURE_VERSION == 6
    ruler_dataset = next(item for item in first["datasets"] if item["key"] == "ruler")
    assert ruler_dataset["formatter_sha256"] == (
        "50d896b551a28e63096adc51727a24e5723903be8dd8a32c221d7c6c6c42ff3f"
    )
    assert ruler_dataset["canonical_id_manifest_sha256"] == (
        "83cc661a8393c491d403c81b702b4f206abb64ee6080c368a5267c93edc45946"
    )
    assert first["schema"] == "recurquant.experiment013.identity-input.v5"
    assert candidate["evidence"]["identity_schema"] == (
        "recurquant.experiment013.identity-candidate.v5"
    )
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


def test_ruler_requirements_materialize_as_frozen_crlf_bytes() -> None:
    payload = capture.RULER_REQUIREMENTS_PATH.read_bytes()

    def git_output(*arguments: str) -> str:
        return subprocess.run(
            [str(FIXTURE_GIT_EXECUTABLE), "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    assert len(payload) == 838
    assert capture.sha256_bytes(payload) == (
        "0f058010181c8fa0e28ff1174a931197e1afef6a9a419b99505777dcf7e28804"
    )
    assert payload.count(b"\r\n") == 40
    without_crlf = payload.replace(b"\r\n", b"")
    assert b"\r" not in without_crlf
    assert b"\n" not in without_crlf
    relative = "requirements/experiment013-ruler.txt"
    raw_oid = git_output("hash-object", "--no-filters", "--", relative)
    assert raw_oid == "d2d30b8994bed276f0161ddfcce2eb4305fc881e"
    assert git_output("hash-object", "--", relative) == raw_oid
    assert git_output("rev-parse", f"HEAD:{relative}") == raw_oid
    assert git_output("rev-parse", f":{relative}") == raw_oid
    assert git_output("check-attr", "text", "whitespace", "--", relative).splitlines() == [
        f"{relative}: text: unset",
        f"{relative}: whitespace: cr-at-eol",
    ]


@pytest.mark.parametrize(
    ("protected_field", "protected_value", "stage_a_error"),
    [
        (
            "command_manifest",
            {"protected_stage_a": "must-remain-uninterpreted"},
            "RULER command manifest fields drifted",
        ),
        (
            "raw_validation_base64",
            "protected-stage-a-must-not-be-decoded",
            "RULER raw validation is not canonical base64",
        ),
    ],
)
def test_calibration_leaves_stage_a_ruler_embedded_values_uninterpreted(
    protected_field: str,
    protected_value: object,
    stage_a_error: str,
) -> None:
    fixture = FakeSource()
    manifest = json.loads(fixture.ruler_generation_manifest_bytes())
    stage_a_filenames = {
        item["filename"] for item in capture.required_ruler_receipts() if item["phase"] == "stage_a"
    }
    calibration_filenames = {
        item["filename"]
        for item in capture.required_ruler_receipts()
        if item["phase"] == "calibration"
    }
    for result in manifest["receipts"]:
        if result["phase"] == "stage_a":
            result[protected_field] = protected_value
    manifest_bytes = capture.canonical_json_bytes(manifest)

    class ManifestSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.receipt_reads: set[str] = set()

        def ruler_generation_manifest_bytes(self) -> bytes:
            return manifest_bytes

        def ruler_receipt_bytes(
            self, *, category: str, config: str, configured_length: int, seed: int
        ) -> bytes:
            filename = capture.ruler_receipt_filename(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )
            self.receipt_reads.add(filename)
            return super().ruler_receipt_bytes(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )

    class OpaqueStageASource(ManifestSource):
        def ruler_receipt_bytes(
            self, *, category: str, config: str, configured_length: int, seed: int
        ) -> bytes:
            filename = capture.ruler_receipt_filename(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )
            if filename in stage_a_filenames:
                raise AssertionError("calibration attempted to read a protected Stage-A receipt")
            return super().ruler_receipt_bytes(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )

    source = OpaqueStageASource()
    captured = capture.capture_identity_input(phase="calibration", source=source)

    assert captured["phase"] == "calibration"
    assert source.receipt_reads == calibration_filenames

    stage_a_source = ManifestSource()
    with pytest.raises(ValueError, match=stage_a_error):
        capture.capture_identity_input(
            phase="stage_a",
            source=stage_a_source,
            calibration_binding=_binding(capture.sha256_bytes(manifest_bytes)),
        )
    assert stage_a_source.receipt_reads
    assert stage_a_source.receipt_reads <= stage_a_filenames


def test_stage_a_capture_reads_only_stage_a_ruler_receipts() -> None:
    manifest_bytes = FakeSource().ruler_generation_manifest_bytes()

    class CountingSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.receipt_reads: set[str] = set()

        def ruler_generation_manifest_bytes(self) -> bytes:
            return manifest_bytes

        def ruler_receipt_bytes(
            self, *, category: str, config: str, configured_length: int, seed: int
        ) -> bytes:
            filename = capture.ruler_receipt_filename(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )
            self.receipt_reads.add(filename)
            return super().ruler_receipt_bytes(
                category=category,
                config=config,
                configured_length=configured_length,
                seed=seed,
            )

    expected = {
        item["filename"] for item in capture.required_ruler_receipts() if item["phase"] == "stage_a"
    }
    source = CountingSource()
    captured = capture.capture_identity_input(
        phase="stage_a",
        source=source,
        calibration_binding=_binding(),
    )

    assert captured["phase"] == "stage_a"
    assert source.receipt_reads == expected


def _stage_a_manifest_mismatch_source() -> Any:
    manifest = json.loads(FakeSource().ruler_generation_manifest_bytes())
    manifest["receipts"][0]["sha256"] = "0" * 64
    manifest_bytes = capture.canonical_json_bytes(manifest)

    class ManifestMismatchSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.receipt_body_reads = 0

        def ruler_generation_manifest_bytes(self) -> bytes:
            return manifest_bytes

        def ruler_receipt_bytes(
            self, *, category: str, config: str, configured_length: int, seed: int
        ) -> bytes:
            self.receipt_body_reads += 1
            raise AssertionError("manifest mismatch must precede protected receipt access")

    return ManifestMismatchSource()


def test_stage_a_capture_rejects_cross_manifest_before_receipt_body_access() -> None:
    source = _stage_a_manifest_mismatch_source()

    with pytest.raises(ValueError, match="generation manifest differs from authenticated custody"):
        capture.capture_identity_input(
            phase="stage_a",
            source=source,
            calibration_binding=_binding(),
        )

    assert source.receipt_body_reads == 0


def test_stage_a_materialization_rejects_cross_manifest_before_receipt_body_access() -> None:
    frozen_bytes = _frozen_stage_a_identity()
    source = _stage_a_manifest_mismatch_source()

    with pytest.raises(ValueError, match="generation manifest differs from authenticated custody"):
        capture.materialize_stage_a_identity_sequences(
            source=source,
            frozen_stage_a_identity_artifact=frozen_bytes,
            calibration_binding_artifact=_binding(),
            **_stage_a_capture_provenance_kwargs(),
        )

    assert source.receipt_body_reads == 0


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
            capture._load_calibration_runner_module(_runner_source_manifest())
    finally:
        if sys.modules.get(capture._CALIBRATION_RUNNER_MODULE_NAME) is sentinel:
            sys.modules.pop(capture._CALIBRATION_RUNNER_MODULE_NAME, None)


def test_calibration_runner_path_swap_cannot_execute_unauthenticated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    runner_path = repository / "scripts" / "run_static_q468_calibration.py"
    runner_path.parent.mkdir(parents=True)
    authenticated_bytes = b"AUTHENTICATED_SENTINEL = 'exact-buffer'\n"
    side_effect = tmp_path / "unauthenticated-side-effect.txt"
    malicious_bytes = (
        "from pathlib import Path\n"
        f"Path({str(side_effect)!r}).write_text('executed', encoding='utf-8')\n"
    ).encode()
    runner_path.write_bytes(authenticated_bytes)
    manifest = {
        "paths": [
            {
                "path": "scripts/run_static_q468_calibration.py",
                "raw_sha256": capture.sha256_bytes(authenticated_bytes),
            }
        ]
    }
    stable_read = capture._bundle_stable_descendant_bytes

    def read_then_swap(
        root: Path,
        relative_path: str,
        *,
        context: str,
    ) -> bytes:
        payload = stable_read(root, relative_path, context=context)
        runner_path.write_bytes(malicious_bytes)
        return payload

    monkeypatch.setattr(capture, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(capture, "CALIBRATION_RUNNER_PATH", runner_path)
    monkeypatch.setattr(capture, "_bundle_stable_descendant_bytes", read_then_swap)
    runner = capture._load_calibration_runner_module(manifest)
    try:
        assert runner.AUTHENTICATED_SENTINEL == "exact-buffer"
        assert not side_effect.exists()
    finally:
        if sys.modules.get(capture._CALIBRATION_RUNNER_MODULE_NAME) is runner:
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
        assert record["fisher_boundary"] == resolver.build_fisher_boundary_contract(
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
    assert b'"input_token_ids":' not in record_bytes
    assert b'"target_token_ids":' not in record_bytes


def test_materialized_sequence_recomputes_fisher_boundary_from_exact_tokens() -> None:
    sequence = capture.materialize_calibration_identity_sequences(source=FakeSource()).sequences[0]
    record = sequence.identity_record
    record["fisher_boundary"]["input_token_ids_sha256"] = "0" * 64
    record["fisher_boundary"]["fisher_boundary_sha256"] = resolver.fisher_boundary_sha256(
        record["fisher_boundary"]
    )
    record["identity_record_sha256"] = resolver.identity_record_sha256(record)

    with pytest.raises(ValueError, match="tokens differ from their identity record"):
        capture.MaterializedCalibrationSequence(
            _identity_record_bytes=capture.canonical_json_bytes(record),
            prompt_token_ids=sequence.prompt_token_ids,
            target_token_ids=sequence.target_token_ids,
        )


def test_capture_rejects_fisher_sequences_shorter_than_three_tokens() -> None:
    with pytest.raises(ValueError, match="at least three tokens"):
        capture._base_record(
            phase="calibration",
            family="pg19",
            canonical_id="short-sequence",
            config="default",
            seed=None,
            configured_length=None,
            ruler_category=None,
            generator_receipt_sha256=None,
            source_payload={"source": "fixture"},
            formatted_payload={"formatted": "fixture"},
            prompt_ids=(1, 2),
            target_ids=(),
            tokenizer_manifest_sha256="a" * 64,
        )


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


def test_stage_a_binding_is_derived_from_identity_scores_split_and_policies(
    tmp_path: Path,
) -> None:
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
    comparator_bytes = build_frozen_comparator_score_artifact(
        _frozen_comparator_aggregate(
            FROZEN_UNWEIGHTED_MSE_PROFILE,
            identity_manifest_sha256=full_identity_manifest,
        ),
        _frozen_comparator_aggregate(
            FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
            identity_manifest_sha256=full_identity_manifest,
        ),
        calibration_identity_sha256=identity.file_sha256,
    )
    comparators = deserialize_comparator_score_artifact(
        comparator_bytes,
        expected_calibration_identity_sha256=identity.file_sha256,
    )

    def comparator_policy_bytes(profile: str, method_id: str) -> bytes:
        selector = comparators.selectors[profile]
        return serialize_static_rht_q468_policy(
            build_static_rht_q468_policy(
                selector.aggregate.d4,
                selector.aggregate.d6,
                selector.aggregate.d8,
                geometry=FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
                marginal_steps=FROZEN_STATIC_Q468_PRIMARY_STEPS,
                calibration_manifest_sha256=(selector.aggregate.sequence_score_manifest_sha256),
                identity_artifact_sha256=identity.file_sha256,
                tokenizer_manifest_sha256=identity.tokenizer_manifest_sha256,
                source_commit="f" * 40,
                method_id=method_id,
            )
        )

    mse_policy_bytes = comparator_policy_bytes(
        FROZEN_UNWEIGHTED_MSE_PROFILE,
        STATIC_Q468_MSE_METHOD,
    )
    fisher_policy_bytes = comparator_policy_bytes(
        FROZEN_DIAGONAL_EMPIRICAL_FISHER_H1_PROFILE,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    )
    core_dependencies = {
        "frozen_identity_artifact": identity_bytes,
        "calibration_score_artifact": score_bytes,
        "split_half_stability_artifact": split_bytes,
        "static_k27030_policy_artifact": policy27030_bytes,
        "static_k29334_policy_artifact": policy29334_bytes,
        "comparator_score_artifact": comparator_bytes,
        "static_fisher_k29334_policy_artifact": fisher_policy_bytes,
        "static_mse_k29334_policy_artifact": mse_policy_bytes,
    }

    dependency_dir = tmp_path / "real-binding-dependencies"
    dependency_dir.mkdir()
    for name, payload in core_dependencies.items():
        (dependency_dir / f"{name}.bin").write_bytes(payload)
    isolated_script = r"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve(strict=True)
dependencies = Path(sys.argv[2]).resolve(strict=True)
runner_path = repository / "scripts" / "run_static_q468_calibration.py"
spec = importlib.util.spec_from_file_location(
    "experiment013_production_runner_isolation_regression",
    runner_path,
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

contract_path = repository / runner.STATIC_Q468_ARTIFACT_CONTRACT_SOURCE_PATH
resolver_path = repository / runner.IDENTITY_RESOLVER_SOURCE_PATH
entries = {
    runner.STATIC_Q468_ARTIFACT_CONTRACT_SOURCE_PATH: {
        "raw_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    },
    runner.IDENTITY_RESOLVER_SOURCE_PATH: {
        "raw_sha256": hashlib.sha256(resolver_path.read_bytes()).hexdigest(),
    },
}
isolation = runner._CalibrationIdentityImportIsolation()
primary_error = None
try:
    isolation.activate()
    namespace = runner._install_authenticated_recurquant_namespace(repository)
    contract = runner._load_exact_source_module(
        runner.STATIC_Q468_ARTIFACT_CONTRACT_MODULE,
        runner.STATIC_Q468_ARTIFACT_CONTRACT_SOURCE_PATH,
        repository_root=repository,
        entry=entries[runner.STATIC_Q468_ARTIFACT_CONTRACT_SOURCE_PATH],
    )
    namespace.static_q468_artifact_contract = contract
    resolver = runner._load_exact_source_module(
        runner.AUTHORIZATION_IDENTITY_RESOLVER_MODULE,
        runner.IDENTITY_RESOLVER_SOURCE_PATH,
        repository_root=repository,
        entry=entries[runner.IDENTITY_RESOLVER_SOURCE_PATH],
    )
    names = (
        "frozen_identity_artifact",
        "calibration_score_artifact",
        "split_half_stability_artifact",
        "static_k27030_policy_artifact",
        "static_k29334_policy_artifact",
        "comparator_score_artifact",
        "static_fisher_k29334_policy_artifact",
        "static_mse_k29334_policy_artifact",
    )
    payloads = {name: (dependencies / f"{name}.bin").read_bytes() for name in names}
    core = resolver.build_stage_a_calibration_core_binding_artifact(**payloads)
    decoded = resolver.deserialize_stage_a_calibration_core_binding_artifact(core)
    assert decoded.binding["calibration_identity_file_sha256"] == hashlib.sha256(
        payloads["frozen_identity_artifact"]
    ).hexdigest()

    def expect_rejected(action):
        try:
            action()
        except (TypeError, ValueError):
            return
        raise AssertionError("semantic or canonical tampering was accepted")

    score_document = json.loads(payloads["calibration_score_artifact"])
    score_document["evidence"]["allocations"][0]["code_counts_q4_q6_q8"][0] += 1
    score_document["canonical_evidence_sha256"] = hashlib.sha256(
        resolver.canonical_json_bytes(score_document["evidence"])
    ).hexdigest()
    semantic_score = resolver.canonical_json_bytes(score_document)
    expect_rejected(
        lambda: resolver.build_stage_a_calibration_core_binding_artifact(
            **{**payloads, "calibration_score_artifact": semantic_score}
        )
    )

    policy_document = json.loads(payloads["static_k29334_policy_artifact"])
    policy_document["content"]["pool_counts"][0] += 1
    policy_document["policy_sha256"] = hashlib.sha256(
        contract._policy_canonical_json(policy_document["content"])
    ).hexdigest()
    semantic_policy = contract._policy_canonical_json(policy_document) + b"\n"
    expect_rejected(
        lambda: resolver.build_stage_a_calibration_core_binding_artifact(
            **{**payloads, "static_k29334_policy_artifact": semantic_policy}
        )
    )

    core_document = json.loads(core)
    core_document["evidence"]["dependency_file_sha256"][
        "calibration_score_artifact"
    ] = "0" * 64
    core_document["canonical_evidence_sha256"] = hashlib.sha256(
        resolver.canonical_json_bytes(core_document["evidence"])
    ).hexdigest()
    expect_rejected(
        lambda: resolver.deserialize_stage_a_calibration_core_binding_artifact(
            resolver.canonical_json_bytes(core_document)
        )
    )
    expect_rejected(
        lambda: resolver.deserialize_stage_a_calibration_core_binding_artifact(core + b"\n")
    )

    isolation.assert_intact()
    assert not {
        "torch",
        "recurquant.static_q468",
        "recurquant.static_q468_calibration",
    } & set(sys.modules)
    assert isolation.blocker.attempts == []
    print(json.dumps({"status": "real_binding_verified_without_torch"}, sort_keys=True))
except BaseException:
    primary_error = sys.exception()
    raise
finally:
    try:
        isolation.restore(primary_error=primary_error)
    finally:
        for name in reversed(runner.AUTHORIZATION_EXACT_MODULE_NAMES):
            sys.modules.pop(name, None)
"""
    isolated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            isolated_script,
            str(REPOSITORY_ROOT),
            str(dependency_dir),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
    assert json.loads(isolated.stdout) == {"status": "real_binding_verified_without_torch"}

    core_bytes = resolver.build_stage_a_calibration_core_binding_artifact(**core_dependencies)
    core = resolver.deserialize_stage_a_calibration_core_binding_artifact(core_bytes)
    authorization_bytes = b"verified-post-calibration-authorization"
    ruler_generation_manifest_file_sha256 = _hash("fixture-ruler-generation-manifest")
    authorization = SimpleNamespace(
        binding=dict(core.binding),
        calibration_dependencies=core_dependencies,
        authorization_dependencies={},
        execution_bindings=dict(identity.execution_bindings),
        source_commit="f" * 40,
        ruler_generation_manifest_file_sha256=(ruler_generation_manifest_file_sha256),
        file_sha256=resolver.sha256_bytes(authorization_bytes),
    )
    with patch.object(
        resolver,
        "deserialize_stage_a_calibration_authorization_artifact",
        return_value=authorization,
    ):
        binding_bytes = resolver.build_stage_a_calibration_binding_artifact(
            calibration_authorization_artifact=authorization_bytes
        )
        verified = resolver.deserialize_stage_a_calibration_binding_artifact(binding_bytes)
        normalized = capture._normalize_calibration_binding(binding_bytes)

    assert normalized == verified.binding
    assert resolver.STAGE_A_BINDING_ARTIFACT_SCHEMA_VERSION == 5
    assert json.loads(binding_bytes)["schema_version"] == 5
    assert "ruler_generation_manifest_file_sha256" not in verified.binding
    assert verified.ruler_generation_manifest_file_sha256 == (ruler_generation_manifest_file_sha256)
    assert (
        json.loads(binding_bytes)["evidence"]["ruler_generation_manifest_file_sha256"]
        == ruler_generation_manifest_file_sha256
    )
    assert verified.binding == {
        "calibration_authorization_file_sha256": resolver.sha256_bytes(authorization_bytes),
        "calibration_identity_file_sha256": identity.file_sha256,
        "calibration_score_artifact_file_sha256": resolver.sha256_bytes(score_bytes),
        "split_half_stability_artifact_file_sha256": resolver.sha256_bytes(split_bytes),
        "static_k27030_policy_file_sha256": resolver.sha256_bytes(policy27030_bytes),
        "static_k29334_policy_file_sha256": resolver.sha256_bytes(policy29334_bytes),
        "comparator_score_artifact_file_sha256": resolver.sha256_bytes(comparator_bytes),
        "static_fisher_k29334_policy_file_sha256": resolver.sha256_bytes(fisher_policy_bytes),
        "static_mse_k29334_policy_file_sha256": resolver.sha256_bytes(mse_policy_bytes),
    }

    tampered = json.loads(binding_bytes)
    encoded_authorization = tampered["evidence"]["dependencies_base64"][
        "calibration_authorization_artifact"
    ]
    authorization_payload = bytearray(base64.b64decode(encoded_authorization))
    authorization_payload[0] ^= 1
    tampered["evidence"]["dependencies_base64"]["calibration_authorization_artifact"] = (
        base64.b64encode(authorization_payload).decode("ascii")
    )
    tampered["canonical_evidence_sha256"] = resolver.sha256_bytes(
        resolver.canonical_json_bytes(tampered["evidence"])
    )
    with pytest.raises(ValueError, match="authorization bytes differ"):
        resolver.deserialize_stage_a_calibration_binding_artifact(
            resolver.canonical_json_bytes(tampered)
        )

    # The pre-authorization core is intentionally not accepted by capture.
    with pytest.raises(ValueError, match="kind or schema drifted"):
        capture._normalize_calibration_binding(core_bytes)


def test_stage_a_capture_uses_exact_schedules_and_token_caps() -> None:
    captured = capture.capture_identity_input(
        phase="stage_a", source=FakeSource(), calibration_binding=_binding()
    )
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
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


def test_stage_a_materialization_authenticates_exact_inventory_tokens_and_spans() -> None:
    frozen_bytes = _frozen_stage_a_identity()
    frozen = resolver.deserialize_frozen_stage_a_identity_artifact(
        frozen_bytes,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
    )
    source = FakeSource()

    materialized = capture.materialize_stage_a_identity_sequences(
        source=source,
        frozen_stage_a_identity_artifact=frozen_bytes,
        calibration_binding_artifact=_binding(),
        expected_frozen_stage_a_identity_file_sha256=resolver.sha256_bytes(frozen_bytes),
        **_stage_a_capture_provenance_kwargs(),
    )

    assert materialized.frozen_identity_file_sha256 == frozen.file_sha256
    assert materialized.frozen_identity_canonical_evidence_sha256 == (
        frozen.canonical_evidence_sha256
    )
    assert materialized.calibration_binding_file_sha256 == resolver.sha256_bytes(_binding())
    assert len(materialized.sequences) == len(materialized.by_identity_record_sha256) == 12
    assert [
        (sequence.identity_record["family"], sequence.identity_record["selection_rank"])
        for sequence in materialized.sequences
    ] == [(family, rank) for family in ("pg19", "ruler", "humaneval_plus") for rank in range(4)]
    assert capture.canonical_json_bytes(materialized.identity_records) == (
        capture.canonical_json_bytes(
            tuple(
                {name: record[name] for name in resolver.RECORD_FIELDS} for record in frozen.records
            )
        )
    )

    for sequence in materialized.sequences:
        record = sequence.identity_record
        span = record["token_span"]
        assert record["identity_record_sha256"] == resolver.identity_record_sha256(record)
        assert record["prompt_token_ids_sha256"] == capture._token_hash(sequence.prompt_token_ids)
        assert record["target_token_ids_sha256"] == capture._token_hash(sequence.target_token_ids)
        assert record["sequence_token_ids_sha256"] == capture._token_hash(
            sequence.sequence_token_ids
        )
        assert all(
            len(record[field]) == 64
            for field in (
                "tokenizer_manifest_sha256",
                "source_content_sha256",
                "formatted_content_sha256",
            )
        )
        assert span == {
            "prefill_start": 0,
            "prefill_stop": len(sequence.prompt_token_ids),
            "scored_start": len(sequence.prompt_token_ids),
            "scored_stop": len(sequence.sequence_token_ids),
            "cache_exposed_start": len(sequence.prompt_token_ids) + 1,
            "cache_exposed_stop": len(sequence.sequence_token_ids),
        }
        assert len(sequence.target_token_ids) >= 2
        assert sequence.cache_exposed_transition_count == len(sequence.target_token_ids) - 1
        assert materialized.lookup(sequence.identity_record_sha256) is sequence

    pg19 = [
        sequence
        for sequence in materialized.sequences
        if sequence.identity_record["family"] == "pg19"
    ]
    assert all(len(sequence.prompt_token_ids) == 4_096 for sequence in pg19)
    assert all(len(sequence.target_token_ids) == 128 for sequence in pg19)
    assert all(sequence.cache_exposed_transition_count == 127 for sequence in pg19)

    ruler_inventory = {
        (
            sequence.identity_record["ruler_category"],
            sequence.identity_record["config"],
            sequence.identity_record["configured_length"],
            sequence.identity_record["seed"],
        )
        for sequence in materialized.sequences
        if sequence.identity_record["family"] == "ruler"
    }
    assert ruler_inventory == set(resolver.RULER_STAGE_A_SCHEDULE)
    for sequence in materialized.sequences:
        record = sequence.identity_record
        if record["family"] != "ruler":
            continue
        receipt = source.ruler_receipt(
            category=record["ruler_category"],
            config=record["config"],
            configured_length=record["configured_length"],
            seed=record["seed"],
        )
        target, _semantics = capture._ruler_stage_a_target(
            category=record["ruler_category"],
            config=record["config"],
            outputs=receipt["outputs"],
        )
        assert sequence.prompt_token_ids == tuple(
            FakeTokenizer().encode(
                receipt["input"] + receipt["answer_prefix"],
                add_special_tokens=False,
            )
        )
        assert sequence.target_token_ids == tuple(
            FakeTokenizer().encode(target, add_special_tokens=False)
        )

    humaneval_projection = {row.canonical_id: row.offset for row in source.humaneval_projection()}
    for sequence in materialized.sequences:
        record = sequence.identity_record
        if record["family"] != "humaneval_plus":
            continue
        row = source.humaneval_row(
            offset=humaneval_projection[record["canonical_id"]],
            expected_task_id=record["canonical_id"],
        )
        assert sequence.prompt_token_ids == tuple(
            FakeTokenizer().encode(row["prompt"], add_special_tokens=True)
        )
        assert sequence.target_token_ids == tuple(
            FakeTokenizer().encode(row["canonical_solution"], add_special_tokens=False)[:128]
        )
    assert all(
        len(sequence.target_token_ids) <= 128
        for sequence in materialized.sequences
        if sequence.identity_record["family"] == "humaneval_plus"
    )
    assert len(materialized.token_sequence_manifest_sha256) == 64
    assert source.head_calls == 2


def test_stage_a_materialization_accepts_exact_two_token_target() -> None:
    source = FakeSource()
    source.receipt_mutator = lambda receipt: (
        receipt.update({"outputs": ["xy"]})
        if receipt["config"] == "qa_1" and receipt["seed"] == 2_344
        else None
    )
    binding = _binding_for_source(source)
    frozen_bytes = _frozen_stage_a_identity(source, calibration_binding=binding)

    materialized = capture.materialize_stage_a_identity_sequences(
        source=source,
        frozen_stage_a_identity_artifact=frozen_bytes,
        calibration_binding_artifact=binding,
        **_stage_a_capture_provenance_kwargs(),
    )

    qa = next(
        sequence
        for sequence in materialized.sequences
        if sequence.identity_record["config"] == "qa_1"
    )
    assert len(qa.target_token_ids) == 2
    assert qa.cache_exposed_transition_count == 1
    assert qa.identity_record["token_span"]["cache_exposed_start"] == (
        qa.identity_record["token_span"]["scored_start"] + 1
    )


def test_stage_a_materialization_rejects_candidate_and_drift_before_use() -> None:
    captured = capture.capture_identity_input(
        phase="stage_a",
        source=FakeSource(),
        calibration_binding=_binding(),
    )
    candidate = resolver.build_candidate(
        captured,
        expected_revisions=resolver.FROZEN_DATASET_REVISIONS,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
    )
    source = FakeSource()
    with pytest.raises(ValueError, match="frozen Stage-A identity"):
        capture.materialize_stage_a_identity_sequences(
            source=source,
            frozen_stage_a_identity_artifact=resolver.canonical_json_bytes(candidate),
            calibration_binding_artifact=_binding(),
            **_stage_a_capture_provenance_kwargs(),
        )
    assert source.accesses == []

    frozen_bytes = _frozen_stage_a_identity()
    changed = FakeSource()
    original_humaneval_row = changed.humaneval_row

    def changed_humaneval_row(*, offset: int, expected_task_id: str) -> dict[str, Any]:
        row = original_humaneval_row(offset=offset, expected_task_id=expected_task_id)
        row["canonical_solution"] += "\n# authenticated-content-drift"
        return row

    changed.humaneval_row = changed_humaneval_row  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="authenticated identity"):
        capture.materialize_stage_a_identity_sequences(
            source=changed,
            frozen_stage_a_identity_artifact=frozen_bytes,
            calibration_binding_artifact=_binding(),
            **_stage_a_capture_provenance_kwargs(),
        )


def test_stage_a_materialization_rejects_missing_duplicate_reordered_and_tampered() -> None:
    frozen_bytes = _frozen_stage_a_identity()
    materialized = capture.materialize_stage_a_identity_sequences(
        source=FakeSource(),
        frozen_stage_a_identity_artifact=frozen_bytes,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
    )
    common = {
        "tokenizer_manifest_sha256": materialized.tokenizer_manifest_sha256,
        "capture_input_sha256": materialized.capture_input_sha256,
        "frozen_identity_file_sha256": materialized.frozen_identity_file_sha256,
        "frozen_identity_canonical_evidence_sha256": (
            materialized.frozen_identity_canonical_evidence_sha256
        ),
        "calibration_binding_file_sha256": materialized.calibration_binding_file_sha256,
    }

    with pytest.raises(ValueError, match="exactly 12"):
        capture.StageAIdentityMaterialization(
            sequences=materialized.sequences[:-1],
            **common,
        )
    with pytest.raises(ValueError, match="duplicate identities"):
        capture.StageAIdentityMaterialization(
            sequences=materialized.sequences[:-1] + (materialized.sequences[0],),
            **common,
        )
    with pytest.raises(ValueError, match="ordered by family then rank"):
        capture.StageAIdentityMaterialization(
            sequences=(materialized.sequences[1], materialized.sequences[0])
            + materialized.sequences[2:],
            **common,
        )

    sequence = materialized.sequences[0]
    tampered_record = sequence.identity_record
    tampered_record["target_token_ids_sha256"] = "0" * 64
    tampered_record["identity_record_sha256"] = resolver.identity_record_sha256(tampered_record)
    with pytest.raises(ValueError, match="only by authenticated v6 materialization"):
        capture.MaterializedStageASequence(
            _identity_record_bytes=capture.canonical_json_bytes(sequence.identity_record),
            prompt_token_ids=sequence.prompt_token_ids,
            target_token_ids=sequence.target_token_ids,
            _authentication_seal=object(),
        )
    with pytest.raises(ValueError, match="tokens differ from their identity record"):
        capture.MaterializedStageASequence(
            _identity_record_bytes=capture.canonical_json_bytes(tampered_record),
            prompt_token_ids=sequence.prompt_token_ids,
            target_token_ids=sequence.target_token_ids,
            _authentication_seal=capture._STAGE_A_MATERIALIZATION_AUTHENTICATION_SEAL,
        )


def test_stage_a_materialization_identity_records_are_content_redacted() -> None:
    frozen_bytes = _frozen_stage_a_identity()
    materialized = capture.materialize_stage_a_identity_sequences(
        source=FakeSource(),
        frozen_stage_a_identity_artifact=frozen_bytes,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
    )
    records = capture.canonical_json_bytes(materialized.identity_records)

    for forbidden in (
        b'"answer_prefix":',
        b'"canonical_solution":',
        b'"formatted_payload":',
        b'"input":',
        b'"outputs":',
        b'"prompt":',
        b'"source_payload":',
        b'"text":',
        b'"prompt_token_ids":',
        b'"target_token_ids":',
    ):
        assert forbidden not in records


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
        seed=2_344,
    )
    receipt["outputs"] = ["only-one"]
    with pytest.raises(ValueError, match="exactly 4 required outputs"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_344,
        )

    receipt["outputs"] = ["same"] * 4
    with pytest.raises(ValueError, match="must be unique"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_344,
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
            lambda receipt: receipt.update({"outputs": ["ZZZZZ", *receipt["outputs"][1:]]}),
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
        seed=2_344,
    )
    mutate(receipt)

    with pytest.raises(ValueError, match=message):
        capture._normalize_ruler_receipt(
            receipt,
            category=category,
            config=config,
            configured_length=4_096,
            seed=2_344,
        )


def test_ruler_receipt_rejects_boolean_sample_index() -> None:
    receipt = FakeSource().ruler_receipt(
        category="retrieval",
        config="niah_multiquery",
        configured_length=4_096,
        seed=2_344,
    )
    receipt["sample_index"] = False

    with pytest.raises(ValueError, match="sample_index must be an integer"):
        capture._normalize_ruler_receipt(
            receipt,
            category="retrieval",
            config="niah_multiquery",
            configured_length=4_096,
            seed=2_344,
        )


@pytest.mark.parametrize("field", ["index", "token_position_answer"])
def test_ruler_raw_row_rejects_boolean_numeric_fields(field: str) -> None:
    source = FakeSource()
    receipt = source.ruler_receipt(
        category="retrieval",
        config="niah_multiquery",
        configured_length=4_096,
        seed=2_344,
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
    binding = _binding_for_source(source)

    with pytest.raises(ValueError, match="continuation must contain at least two tokens"):
        capture.capture_identity_input(phase="stage_a", source=source, calibration_binding=binding)


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


def test_direct_stage_a_cli_fails_before_binding_path_source_or_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def fail_if_touched(name: str) -> Any:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"direct Stage-A CLI touched {name}")

        return fail

    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        fail_if_touched("binding decoder"),
    )
    monkeypatch.setattr(Path, "read_bytes", fail_if_touched("path bytes"))
    monkeypatch.setattr(
        capture,
        "_runtime_context_from_cli",
        fail_if_touched("runtime provider"),
    )
    monkeypatch.setattr(capture, "LiveCaptureSource", fail_if_touched("source constructor"))
    monkeypatch.setattr(
        capture,
        "capture_identity_input",
        fail_if_touched("capture provider"),
    )

    with pytest.raises(PermissionError, match="sealed-launcher-only.*before binding"):
        capture.main(["--phase", "stage_a"])

    assert touched == []


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
        def __init__(self, *, token: bool, endpoint: str) -> None:
            assert token is False
            assert endpoint == "https://huggingface.co"

        @staticmethod
        def list_repo_files(_repo_id: str, *, revision: str, token: bool) -> list[str]:
            assert revision == resolver.PRIMARY_MODEL_REVISION
            assert token is False
            return [*expected, "tokenizer.model", "model.safetensors"]

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: Path,
        token: bool,
        endpoint: str,
    ) -> str:
        assert repo_id == resolver.PRIMARY_MODEL_ID
        assert revision == resolver.PRIMARY_MODEL_REVISION
        assert cache_dir == (tmp_path / "cache").resolve()
        assert token is False
        assert endpoint == "https://huggingface.co"
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
        "token": False,
        "trust_remote_code": False,
    }
    assert not Path(observed["path"]).exists()


def test_ruler_receipt_category_drift_fails_closed() -> None:
    source = FakeSource()
    source.receipt_mutator = lambda receipt: receipt.update({"category": "retrieval"})

    with pytest.raises(ValueError, match="receipt category drifted"):
        capture.capture_identity_input(phase="calibration", source=source)


def test_ruler_receipt_hash_is_checked_before_json_decode() -> None:
    source = FakeSource()
    frozen_manifest = source.ruler_generation_manifest_bytes()
    source.ruler_generation_manifest_bytes = lambda: frozen_manifest  # type: ignore[method-assign]
    source.ruler_receipt_bytes = lambda **_kwargs: b"not-json"  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="receipt file identity drifted"):
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


def test_live_source_rejects_retired_stage_a_receipt_directory_extra(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    (receipt_dir / "generation-manifest.json").write_bytes(b"fixture")
    for item in capture.required_ruler_receipts():
        (receipt_dir / item["filename"]).write_bytes(b"fixture")
    retired = "retrieval__niah_multiquery__l4096__s2339.json"
    (receipt_dir / retired).write_bytes(b"retired")
    source = capture.LiveCaptureSource(
        cache_dir=tmp_path / "cache",
        ruler_receipt_dir=receipt_dir,
    )

    with pytest.raises(ValueError, match=r"inventory drifted: .*unexpected=.*s2339"):
        source.ruler_generation_manifest_bytes()


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


@pytest.mark.parametrize("binding", [b"legacy-binding", b"{}\n", b"tampered-binding"])
def test_invalid_stage_a_binding_fails_before_runtime_or_source_access(binding: bytes) -> None:
    source = FakeSource()
    source.runtime_authentication_context = lambda: pytest.fail(  # type: ignore[method-assign]
        "runtime provider must not be touched before binding verification"
    )

    with pytest.raises(ValueError):
        capture.capture_identity_input(
            phase="stage_a",
            source=source,
            calibration_binding=binding,
        )

    assert source.accesses == []


def test_stage_a_authorization_execution_mismatch_fails_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource()
    wrong = {
        field: capture.sha256_bytes(payload)
        for field, payload in sorted(FIXTURE_EXECUTION_ARTIFACTS.items())
    }
    wrong["model_file_manifest_file_sha256"] = _hash("wrong-authorized-model-manifest")
    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        lambda _data: SimpleNamespace(
            binding={
                key: _hash(f"binding-{key}") for key in sorted(resolver.CALIBRATION_BINDING_FIELDS)
            },
            execution_bindings=wrong,
            ruler_generation_manifest_file_sha256=_hash("fixture-ruler-generation-manifest"),
        ),
    )

    with pytest.raises(ValueError, match="differ from the calibration authorization"):
        capture.capture_identity_input(
            phase="stage_a",
            source=source,
            calibration_binding=b"valid-but-cross-chain-binding",
        )

    assert source.accesses == []


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
        "git_executable": {
            "sha256": capture.sha256_bytes(FIXTURE_GIT_EXECUTABLE.read_bytes()),
            "size_bytes": FIXTURE_GIT_EXECUTABLE.stat().st_size,
        },
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

    monkeypatch.setattr(
        capture,
        "_load_calibration_runner_module",
        lambda _source_manifest: FakeRunner(),
    )
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
            "git_executable": FIXTURE_GIT_EXECUTABLE,
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
    bindings = {field: capture.sha256_bytes(data) for field, data in sorted(artifacts.items())}
    source_manifest = _runner_source_manifest()

    class SourceModule:
        @staticmethod
        def verify_experiment013_source_manifest(
            manifest: Any,
            *,
            repo_root: Path,
            git_executable: Path,
        ) -> Any:
            assert repo_root == REPOSITORY_ROOT
            assert git_executable == FIXTURE_GIT_EXECUTABLE
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
            git_executable_path: Path,
        ) -> Any:
            assert base_runtime_root == runtime_context.base_runtime_root
            assert package_roots == runtime_context.package_runtime_roots
            assert interpreter_path == runtime_context.staged_interpreter
            assert git_executable_path == runtime_context.git_executable
            events.append("runtime")
            return SimpleNamespace(
                manifest_file_sha256=bindings["calibration_runtime_manifest_file_sha256"]
            )

        @staticmethod
        def capture_model_file_manifest_from_hub(
            model_id: str,
            revision: str,
            *,
            transformers_version: str,
            token: bool,
        ) -> bytes:
            assert (model_id, revision, transformers_version) == (
                resolver.PRIMARY_MODEL_ID,
                resolver.PRIMARY_MODEL_REVISION,
                resolver.TRANSFORMERS_VERSION,
            )
            assert token is False
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

    source_artifact = capture._DecodedRepositorySourceArtifact(
        manifest_file_sha256=bindings["repository_source_manifest_file_sha256"],
        source_manifest=source_manifest,
        source_module=SourceModule(),
    )

    def decode_source(_artifacts: Any) -> Any:
        events.append("decode-source")
        return source_artifact

    def load_runner(authenticated_source_manifest: Any) -> Any:
        assert authenticated_source_manifest == source_manifest
        events.append("load-runner")
        sys.modules[capture._CALIBRATION_RUNNER_MODULE_NAME] = runner  # type: ignore[assignment]
        return runner

    def decode(_artifacts: Any, *, runner: Any, source_artifact: Any) -> Any:
        assert runner is not None
        assert source_artifact is not None
        events.append("decode")
        return decoded

    monkeypatch.setattr(capture, "_decode_repository_source_artifact", decode_source)
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
        "decode-source",
        "load-runner",
        "decode",
        "source",
        "loaded-source-parquet",
        "runner-source",
        "runtime",
        "model",
        "decode-source",
        "decode",
        "source",
        "loaded-source-parquet",
        "runner-source",
        "runtime",
        "model",
    ]


def test_loaded_calibration_runner_source_is_bound_to_source_manifest() -> None:
    runner_bytes = capture.CALIBRATION_RUNNER_PATH.read_bytes()
    runner_sha256 = capture.sha256_bytes(runner_bytes)
    runner = SimpleNamespace(
        __file__=str(capture.CALIBRATION_RUNNER_PATH),
        __recurquant_authenticated_source_sha256__=runner_sha256,
        __recurquant_authenticated_source_size_bytes__=len(runner_bytes),
    )
    manifest = {
        "paths": [
            {
                "path": "scripts/run_static_q468_calibration.py",
                "raw_sha256": runner_sha256,
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
        lambda context: context.update({"package_import_paths": {"packages": "../site-packages"}}),
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
                "--git-executable",
                str(FIXTURE_GIT_EXECUTABLE),
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
        def __init__(self, *, token: bool, endpoint: str) -> None:
            assert token is False
            assert endpoint == "https://huggingface.co"

        @staticmethod
        def model_info(repo_id: str, *, revision: str, token: bool) -> Any:
            assert token is False
            observed_hf.append(("model", repo_id, revision))
            return SimpleNamespace(sha=revision)

        @staticmethod
        def dataset_info(repo_id: str, *, revision: str, token: bool) -> Any:
            assert token is False
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
    stage_a_receipts = [item for item in receipts if item["phase"] == "stage_a"]

    assert len(receipts) == 20
    assert len({item["filename"] for item in receipts}) == 20
    assert sum(item["phase"] == "calibration" for item in receipts) == 16
    assert len(stage_a_receipts) == 4
    assert {item["seed"] for item in stage_a_receipts} == {2344}
    assert all("__s2339.json" not in item["filename"] for item in stage_a_receipts)
    assert all("__s2343.json" not in item["filename"] for item in receipts)
    assert receipts[0]["filename"] == ("retrieval__niah_multiquery__l2048__s12339.json")
    assert receipts[-1]["filename"] == ("question_answering__qa_1__l4096__s2344.json")


def _parquet_bytes(columns: dict[str, list[str]]) -> bytes:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    sink = arrow.BufferOutputStream()
    parquet.write_table(arrow.table(columns), sink)
    return sink.getvalue().to_pybytes()


def _fixture_stage_a_input_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    from recurquant import experiment013_parquet

    frozen_bytes = _frozen_stage_a_identity()
    frozen = resolver.deserialize_frozen_stage_a_identity_artifact(
        frozen_bytes,
        calibration_binding_artifact=_binding(),
        **_stage_a_capture_provenance_kwargs(),
    )
    parquet_payloads = {
        "pg19": _parquet_bytes(
            {
                "url": [
                    f"http://www.gutenberg.org/ebooks/{200_000 + index}" for index in range(50)
                ],
                "text": [f"fixture PG19 text {index}" for index in range(50)],
            }
        ),
        "humaneval_plus": _parquet_bytes(
            {
                "task_id": [f"HumanEval/{index}" for index in range(164)],
                "prompt": [f"def task_{index}(x):\n" for index in range(164)],
                "canonical_solution": ["    return x\n" for _ in range(164)],
            }
        ),
    }
    selected: list[tuple[Any, Any]] = []
    datasets: dict[str, Any] = {}
    for dataset, file in capture._bundle_expected_parquet_files():
        payload = parquet_payloads[dataset.key]
        lfs_sha256 = capture.sha256_bytes(payload)
        size_bytes = len(payload)
        git_blob_oid = capture._git_blob_sha1(
            capture._bundle_lfs_pointer_bytes(
                sha256=lfs_sha256,
                size_bytes=size_bytes,
            )
        )
        fixture_file = replace(
            file,
            size_bytes=size_bytes,
            git_blob_oid=git_blob_oid,
            lfs_sha256=lfs_sha256,
            lfs_size_bytes=size_bytes,
        )
        fixture_dataset = replace(
            dataset,
            selected_splits=(file.logical_split,),
            files=(fixture_file,),
        )
        selected.append((fixture_dataset, fixture_file))
        datasets[dataset.key] = fixture_dataset
    frozen_selected = tuple(selected)
    monkeypatch.setattr(
        capture,
        "_bundle_expected_parquet_files",
        lambda: frozen_selected,
    )
    fixture_manifest = SimpleNamespace(dataset=lambda key: datasets[key])
    monkeypatch.setattr(
        experiment013_parquet,
        "load_experiment013_parquet_manifest",
        lambda *_args, **_kwargs: fixture_manifest,
    )

    bundle_root = tmp_path / "stage-a-input-bundle"
    bundle_root.mkdir()
    records: list[dict[str, Any]] = []
    ruler_source = FakeSource()
    capture._bundle_add_object(
        bundle_root,
        records,
        role="model_hub_manifest",
        source_id=resolver.PRIMARY_MODEL_ID,
        revision=resolver.PRIMARY_MODEL_REVISION,
        logical_path="model-file-manifest.json",
        payload=FIXTURE_EXECUTION_ARTIFACTS["model_file_manifest_file_sha256"],
    )
    tokenizer_payloads = {
        "tokenizer.json": b"fixture-tokenizer",
        "tokenizer_config.json": b"fixture-tokenizer-config",
    }
    assert set(tokenizer_payloads) == set(capture.RULER_EXPECTED_TOKENIZER_ASSETS)
    for name, payload in sorted(tokenizer_payloads.items()):
        capture._bundle_add_object(
            bundle_root,
            records,
            role="tokenizer",
            source_id=resolver.PRIMARY_MODEL_ID,
            revision=resolver.PRIMARY_MODEL_REVISION,
            logical_path=name,
            payload=payload,
        )
    for path, payload in sorted(_fake_generator_files().items()):
        capture._bundle_add_object(
            bundle_root,
            records,
            role="ruler_generator",
            source_id=resolver.RULER_SOURCE_ID,
            revision=resolver.RULER_REVISION,
            logical_path=path,
            payload=payload,
            git_blob_oid=capture.RULER_GENERATOR_GIT_BLOBS[path],
        )
    capture._bundle_add_object(
        bundle_root,
        records,
        role="ruler_generation_manifest",
        source_id=resolver.RULER_SOURCE_ID,
        revision=resolver.RULER_REVISION,
        logical_path="generation-manifest.json",
        payload=ruler_source.ruler_generation_manifest_bytes(),
    )
    for item in capture.required_ruler_receipts():
        filename = str(item["filename"])
        capture._bundle_add_object(
            bundle_root,
            records,
            role="ruler_receipt",
            source_id=resolver.RULER_SOURCE_ID,
            revision=resolver.RULER_REVISION,
            logical_path=filename,
            payload=ruler_source.ruler_receipt_bytes(
                category=item["category"],
                config=item["config"],
                configured_length=item["configured_length"],
                seed=item["seed"],
            ),
        )
    snapshots: list[dict[str, Any]] = []
    parquet_logicals: dict[str, str] = {}
    for dataset, file in frozen_selected:
        logical = f"{dataset.key}/{file.logical_split}/{file.immutable_path}"
        parquet_logicals[dataset.key] = logical
        capture._bundle_add_object(
            bundle_root,
            records,
            role="parquet",
            source_id=dataset.dataset_id,
            revision=dataset.conversion_revision,
            logical_path=logical,
            payload=parquet_payloads[dataset.key],
            git_blob_oid=file.git_blob_oid,
            lfs_sha256=file.lfs_sha256,
        )
        snapshots.append(
            {
                "dataset_key": dataset.key,
                "dataset_id": dataset.dataset_id,
                "source_revision": dataset.source_revision,
                "conversion_revision": dataset.conversion_revision,
                "files": [
                    {
                        "path": file.immutable_path,
                        "git_blob_oid": file.git_blob_oid,
                        "lfs_sha256": file.lfs_sha256,
                        "size_bytes": file.size_bytes,
                    }
                ],
            }
        )
    records.sort(key=lambda item: (item["role"], item["source_id"], item["logical_path"]))
    manifest = {
        "schema": capture.STAGE_A_INPUT_BUNDLE_SCHEMA,
        "phase": "stage_a",
        "capture_version": capture.CAPTURE_VERSION,
        "staging_profile": capture.STAGE_A_INPUT_BUNDLE_PROFILE,
        "frozen_identity_file_sha256": frozen.file_sha256,
        "calibration_binding_file_sha256": capture.sha256_bytes(_binding()),
        "execution_bindings": dict(frozen.execution_bindings),
        "source_heads": dict(capture.EXPECTED_SOURCE_HEADS),
        "model_hub_manifest_file_sha256": capture.sha256_bytes(
            FIXTURE_EXECUTION_ARTIFACTS["model_file_manifest_file_sha256"]
        ),
        "parquet_hub_snapshots": snapshots,
        "objects": records,
    }
    manifest_path = bundle_root / capture.STAGE_A_INPUT_BUNDLE_FILENAME
    manifest_path.write_bytes(capture.canonical_json_bytes(manifest))
    bundle = capture.authenticate_stage_a_input_bundle(
        bundle_root,
        frozen_stage_a_identity_artifact=frozen_bytes,
        calibration_binding_artifact=_binding(),
        execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
        **_stage_a_capture_provenance_kwargs(),
    )
    return SimpleNamespace(
        bundle=bundle,
        root=bundle_root,
        manifest_path=manifest_path,
        frozen_bytes=frozen_bytes,
        records=records,
        selected=frozen_selected,
        parquet_logicals=parquet_logicals,
    )


def _authenticate_fixture_bundle(fixture: SimpleNamespace) -> Any:
    return capture.authenticate_stage_a_input_bundle(
        fixture.root,
        frozen_stage_a_identity_artifact=fixture.frozen_bytes,
        calibration_binding_artifact=_binding(),
        execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
        **_stage_a_capture_provenance_kwargs(),
    )


def test_stage_a_bundle_rejects_cross_manifest_before_receipt_object_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    decode_binding = resolver.deserialize_stage_a_calibration_binding_artifact

    def decode_binding_with_different_manifest(data: bytes, **kwargs: object) -> object:
        verified = decode_binding(data, **kwargs)
        return SimpleNamespace(
            **{
                **vars(verified),
                "ruler_generation_manifest_file_sha256": "f" * 64,
            }
        )

    monkeypatch.setattr(
        resolver,
        "deserialize_stage_a_calibration_binding_artifact",
        decode_binding_with_different_manifest,
    )
    reads: list[str] = []
    object_bytes = capture.AuthenticatedStageAInputBundle.object_bytes

    def tracked_object_bytes(
        bundle: capture.AuthenticatedStageAInputBundle,
        role: str,
        logical_path: str,
    ) -> bytes:
        reads.append(role)
        return object_bytes(bundle, role, logical_path)

    monkeypatch.setattr(
        capture.AuthenticatedStageAInputBundle,
        "object_bytes",
        tracked_object_bytes,
    )

    with pytest.raises(ValueError, match="manifest differs from calibration custody"):
        _authenticate_fixture_bundle(fixture)

    assert reads == ["ruler_generation_manifest"]


def test_stage_a_stager_rejects_cross_manifest_before_receipt_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ruler_root = tmp_path / "ruler-receipts"
    ruler_root.mkdir()
    (ruler_root / "generation-manifest.json").write_bytes(b"different manifest\n")
    for item in capture.required_ruler_receipts():
        (ruler_root / str(item["filename"])).write_bytes(b"opaque receipt\n")
    reads: list[str] = []
    stable_file_bytes = capture._bundle_stable_file_bytes

    def tracked_stable_file_bytes(path: Path, *, context: str) -> bytes:
        reads.append(Path(path).name)
        return stable_file_bytes(path, context=context)

    monkeypatch.setattr(capture, "_bundle_stable_file_bytes", tracked_stable_file_bytes)

    with pytest.raises(ValueError, match="stager RULER manifest differs"):
        capture.stage_stage_a_input_bundle(
            bundle_root=tmp_path / "staged-bundle",
            cache_dir=tmp_path / "cache",
            ruler_receipt_dir=ruler_root,
            frozen_stage_a_identity_artifact=_frozen_stage_a_identity(),
            calibration_binding_artifact=_binding(),
            execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
            runtime_authentication_context=FIXTURE_RUNTIME_CONTEXT,
            **_stage_a_capture_provenance_kwargs(),
        )

    assert reads == ["generation-manifest.json"]


def test_stage_a_bundle_rejects_self_rechained_wrong_receipt_without_decoding_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest_path.read_bytes())
    record = next(item for item in manifest["objects"] if item["role"] == "ruler_receipt")
    old_relative = str(record["relative_path"])
    wrong_payload = b"opaque receipt bytes from a different generation manifest\n"
    wrong_sha256 = capture.sha256_bytes(wrong_payload)
    record["relative_path"] = f"objects/{wrong_sha256}"
    record["sha256"] = wrong_sha256
    record["size_bytes"] = len(wrong_payload)
    (fixture.root / record["relative_path"]).write_bytes(wrong_payload)
    if sum(item["relative_path"] == old_relative for item in manifest["objects"]) == 0:
        (fixture.root / old_relative).unlink()
    fixture.manifest_path.write_bytes(capture.canonical_json_bytes(manifest))

    strict_json = capture._strict_json

    def reject_wrong_receipt_decode(data: bytes, *, context: str) -> dict[str, Any]:
        if data == wrong_payload:
            raise AssertionError("wrong opaque RULER receipt was semantically decoded")
        return strict_json(data, context=context)

    monkeypatch.setattr(capture, "_strict_json", reject_wrong_receipt_decode)
    with pytest.raises(ValueError, match="receipt differs from authenticated identity"):
        _authenticate_fixture_bundle(fixture)


def test_stage_a_bundle_authenticates_receipts_without_semantic_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest_path.read_bytes())
    receipt_payloads = {
        (fixture.root / str(item["relative_path"])).read_bytes()
        for item in manifest["objects"]
        if item["role"] == "ruler_receipt"
    }
    assert len(receipt_payloads) == 20
    strict_json = capture._strict_json

    def reject_receipt_decode(data: bytes, *, context: str) -> dict[str, Any]:
        if data in receipt_payloads:
            raise AssertionError("opaque RULER receipt was semantically decoded")
        return strict_json(data, context=context)

    monkeypatch.setattr(capture, "_strict_json", reject_receipt_decode)
    authenticated = _authenticate_fixture_bundle(fixture)

    assert authenticated.manifest_file_sha256 == capture.sha256_bytes(
        fixture.manifest_path.read_bytes()
    )


def test_opaque_ruler_receipt_snapshot_reads_each_authenticated_body_once() -> None:
    source = FakeSource()
    payloads = {
        str(item["filename"]): source.ruler_receipt_bytes(
            category=item["category"],
            config=item["config"],
            configured_length=item["configured_length"],
            seed=item["seed"],
        )
        for item in capture.required_ruler_receipts()
    }
    expected = {name: capture.sha256_bytes(payload) for name, payload in payloads.items()}
    reads: list[str] = []

    def read_once(filename: str) -> bytes:
        reads.append(filename)
        payload = payloads[filename]
        payloads[filename] = b"changed after the authenticated read\n"
        return payload

    frozen = capture._bundle_read_authoritative_ruler_receipts(
        expected_sha256=expected,
        reader=read_once,
        context="synthetic opaque stager",
    )

    assert reads == [str(item["filename"]) for item in capture.required_ruler_receipts()]
    assert len(reads) == len(set(reads)) == 20
    assert all(capture.sha256_bytes(frozen[name]) == expected[name] for name in expected)


def test_stage_a_stager_rejects_wrong_receipt_before_network_or_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource()
    ruler_root = tmp_path / "ruler-receipts"
    ruler_root.mkdir()
    (ruler_root / "generation-manifest.json").write_bytes(source.ruler_generation_manifest_bytes())
    wrong_filename = str(capture.required_ruler_receipts()[0]["filename"])
    wrong_payload = b"opaque receipt bytes from a different generation manifest\n"
    for item in capture.required_ruler_receipts():
        filename = str(item["filename"])
        payload = (
            wrong_payload
            if filename == wrong_filename
            else source.ruler_receipt_bytes(
                category=item["category"],
                config=item["config"],
                configured_length=item["configured_length"],
                seed=item["seed"],
            )
        )
        (ruler_root / filename).write_bytes(payload)
    strict_json = capture._strict_json

    def reject_wrong_receipt_decode(data: bytes, *, context: str) -> dict[str, Any]:
        if data == wrong_payload:
            raise AssertionError("wrong opaque RULER receipt was semantically decoded")
        return strict_json(data, context=context)

    def reject_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("opaque stager reached network before receipt authentication")

    monkeypatch.setattr(capture, "_strict_json", reject_wrong_receipt_decode)
    monkeypatch.setattr(capture.urllib.request, "urlopen", reject_network)
    with pytest.raises(ValueError, match="receipt differs from authenticated identity"):
        capture.stage_stage_a_input_bundle(
            bundle_root=tmp_path / "staged-bundle",
            cache_dir=tmp_path / "cache",
            ruler_receipt_dir=ruler_root,
            frozen_stage_a_identity_artifact=_frozen_stage_a_identity(),
            calibration_binding_artifact=_binding(),
            execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
            runtime_authentication_context=FIXTURE_RUNTIME_CONTEXT,
            **_stage_a_capture_provenance_kwargs(),
        )


def test_stage_a_bundle_authenticates_both_lfs_pointer_and_payload_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)

    assert len(fixture.selected) == 2
    for dataset, file in fixture.selected:
        record = fixture.bundle.objects[("parquet", fixture.parquet_logicals[dataset.key])]
        assert record["sha256"] == file.lfs_sha256
        assert record["size_bytes"] == file.size_bytes
        assert record["git_blob_oid"] == capture._git_blob_sha1(
            capture._bundle_lfs_pointer_bytes(
                sha256=file.lfs_sha256,
                size_bytes=file.size_bytes,
            )
        )

    collision_root = tmp_path / "bad-lfs"
    collision_root.mkdir()
    payload = b"valid LFS payload bytes\n"
    with pytest.raises(ValueError, match="Git blob"):
        capture._bundle_add_object(
            collision_root,
            [],
            role="parquet",
            source_id="fixture/dataset",
            revision="a" * 40,
            logical_path="default/test/0000.parquet",
            payload=payload,
            git_blob_oid="0" * 40,
            lfs_sha256=capture.sha256_bytes(payload),
        )


def test_stage_a_bundle_rejects_object_tamper_and_extra_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    logical = fixture.parquet_logicals["pg19"]
    record = fixture.bundle.objects[("parquet", logical)]
    object_path = fixture.root / Path(str(record["relative_path"]))
    object_path.write_bytes(object_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed"):
        _authenticate_fixture_bundle(fixture)


@pytest.mark.parametrize("location", ("root", "objects"))
def test_stage_a_bundle_rejects_extra_filesystem_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    parent = fixture.root if location == "root" else fixture.root / "objects"
    (parent / "unexpected").write_bytes(b"unexpected\n")
    with pytest.raises(ValueError, match="filesystem inventory drifted"):
        _authenticate_fixture_bundle(fixture)


def test_stage_a_bundle_rejects_forged_semantic_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest_path.read_bytes())
    record = next(item for item in manifest["objects"] if item["role"] == "model_hub_manifest")
    record["revision"] = "f" * 40
    fixture.manifest_path.write_bytes(capture.canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="frozen semantics"):
        _authenticate_fixture_bundle(fixture)


def test_stage_a_bundle_rejects_linked_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    record = next(iter(fixture.bundle.objects.values()))
    object_path = fixture.root / Path(str(record["relative_path"]))
    outside = tmp_path / "outside-object"
    outside.write_bytes(object_path.read_bytes())
    object_path.unlink()
    try:
        object_path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    with pytest.raises(ValueError, match="link|reparse"):
        _authenticate_fixture_bundle(fixture)


def test_stage_a_bundle_path_status_errors_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot authenticate"):
        capture._bundle_is_link_or_reparse(tmp_path / "absent")


def test_existing_stage_a_bundle_still_requires_runtime_auth_and_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    ruler_root = tmp_path / "receipts"
    ruler_root.mkdir()

    with pytest.raises(ValueError):
        capture.stage_stage_a_input_bundle(
            bundle_root=fixture.root,
            cache_dir=tmp_path / "cache",
            ruler_receipt_dir=ruler_root,
            frozen_stage_a_identity_artifact=fixture.frozen_bytes,
            calibration_binding_artifact=_binding(),
            execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
            runtime_authentication_context={"invalid": "context"},
            **_stage_a_capture_provenance_kwargs(),
        )
    with pytest.raises(ValueError, match="must not be nested"):
        capture.stage_stage_a_input_bundle(
            bundle_root=fixture.root,
            cache_dir=tmp_path,
            ruler_receipt_dir=ruler_root,
            frozen_stage_a_identity_artifact=fixture.frozen_bytes,
            calibration_binding_artifact=_binding(),
            execution_binding_artifacts=FIXTURE_EXECUTION_ARTIFACTS,
            runtime_authentication_context=FIXTURE_RUNTIME_CONTEXT,
            **_stage_a_capture_provenance_kwargs(),
        )


def test_staged_capture_source_is_offline_and_reads_only_authenticated_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    for name in capture._STAGE_A_FORBIDDEN_CREDENTIAL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in capture._STAGE_A_OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    def reject_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("offline staged capture attempted network access")

    monkeypatch.setattr(capture.urllib.request, "urlopen", reject_network)
    source = capture.StagedCaptureSource(fixture.bundle)
    assert source.source_heads() == capture.EXPECTED_SOURCE_HEADS
    assert set(source.ruler_generator_files()) == set(capture.RULER_GENERATOR_GIT_BLOBS)
    assert len(source.pg19_projection("validation")) == 50
    pg19 = source.pg19_row(
        "validation",
        offset=0,
        expected_url="http://www.gutenberg.org/ebooks/200000",
    )
    assert pg19["text"] == "fixture PG19 text 0"
    assert len(source.humaneval_projection()) == 164
    humaneval = source.humaneval_row(offset=0, expected_task_id="HumanEval/0")
    assert humaneval["canonical_solution"] == "    return x\n"

    class FixtureAutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> FakeTokenizer:
            assert kwargs == {
                "local_files_only": True,
                "trust_remote_code": False,
                "token": False,
            }
            assert {item.name for item in path.iterdir()} == set(
                capture.RULER_EXPECTED_TOKENIZER_ASSETS
            )
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FixtureAutoTokenizer),
    )
    tokenizer = source.tokenizer_material()
    assert tokenizer.model_weights_loaded is False
    assert set(tokenizer.files) == set(capture.RULER_EXPECTED_TOKENIZER_ASSETS)


def test_staged_capture_source_rejects_missing_offline_flag_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_stage_a_input_bundle(tmp_path, monkeypatch)
    for name in capture._STAGE_A_FORBIDDEN_CREDENTIAL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in capture._STAGE_A_OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(RuntimeError, match="offline-mode flags"):
        capture.StagedCaptureSource(fixture.bundle)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "forbidden-fixture-token")
    with pytest.raises(RuntimeError, match="forbidden credential"):
        capture.StagedCaptureSource(fixture.bundle)
