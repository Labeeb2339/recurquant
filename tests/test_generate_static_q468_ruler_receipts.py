from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_static_q468_ruler_receipts.py"
SPEC = importlib.util.spec_from_file_location("generate_static_q468_ruler_receipts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ruler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ruler
SPEC.loader.exec_module(ruler)


class WhitespaceTokenizer:
    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()


def _valid_niah_row() -> dict[str, object]:
    answers = ["8028695", "4670027", "8310696", "7938875"]
    input_text = (
        "Some special magic numbers are hidden here. "
        + " ".join(f"One special value is {answer}." for answer in answers)
        + " "
        "What are all the special magic numbers in the provided text?"
    )
    prefix = " The special magic numbers are"
    return {
        "index": input_text.find(answers[0]),
        "input": input_text,
        "outputs": answers,
        "length": len(WhitespaceTokenizer.tokenize(input_text + prefix)) + 128,
        "length_w_model_temp": len(WhitespaceTokenizer.tokenize(input_text + prefix)) + 128,
        "answer_prefix": prefix,
        "token_position_answer": len(
            WhitespaceTokenizer.tokenize(input_text[: input_text.find(answers[0])])
        ),
    }


def test_task_specs_cover_the_exact_required_receipt_configs() -> None:
    capture = ruler._load_capture_module()
    required_configs = {str(item["config"]) for item in capture.required_ruler_receipts()}

    assert set(ruler.TASK_SPECS) == required_configs
    assert len(capture.required_ruler_receipts()) == 20
    assert {config: spec.expected_output_count for config, spec in ruler.TASK_SPECS.items()} == {
        config: capture.RULER_REQUIRED_OUTPUT_COUNTS.get(config) for config in required_configs
    }


def test_generator_argv_preserves_multiline_template_as_one_argument(tmp_path) -> None:
    receipt = {
        "config": "niah_multiquery",
        "configured_length": 2048,
        "seed": 12339,
    }

    actual, portable = ruler.generator_argv(
        python=tmp_path / "python.exe",
        ruler_root=tmp_path / "ruler",
        raw_root=tmp_path / "raw",
        receipt=receipt,
    )

    template_index = actual.index("--template") + 1
    assert actual[template_index] == ruler.NIAH_TEMPLATE
    assert "\n{context}\n" in actual[template_index]
    assert portable[0] == "<RULER_PYTHON>"
    assert portable[1] == "-s"
    assert portable[2] == "scripts/data/synthetic/niah.py"
    assert portable[4] == "<RAW_RECEIPT_DIR>"


def test_valid_niah_row_is_independently_recounted() -> None:
    normalized = ruler._normalize_output_row(
        _valid_niah_row(),
        config="niah_multiquery",
        configured_length=2048,
        tokenizer=WhitespaceTokenizer(),
    )

    assert normalized["outputs"] == ["8028695", "4670027", "8310696", "7938875"]
    assert normalized["generator_reported_length"] == _valid_niah_row()["length"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(input="prompt without the required question"), "task marker"),
        (lambda row: row.update(answer_prefix="."), "answer prefix"),
        (lambda row: row.update(index=-1), "not present"),
        (lambda row: row.update(length=int(row["length"]) + 1), "tokenization"),
        (lambda row: row.update(outputs=[]), "outputs"),
        (
            lambda row: row.update(outputs=["8028695", "4670027", "8310696"]),
            "output count",
        ),
        (
            lambda row: row.update(outputs=["8028695", "4670027", "8310696", "8310696"]),
            "unique",
        ),
        (
            lambda row: row.update(outputs=["8028695", "4670027", "8310696", "1234567"]),
            "absent",
        ),
    ],
)
def test_malformed_or_truncated_ruler_rows_fail_closed(mutation, message: str) -> None:
    row = _valid_niah_row()
    mutation(row)
    if row["length"] != row["length_w_model_temp"]:
        row["length_w_model_temp"] = row["length"]

    with pytest.raises(ValueError, match=message):
        ruler._normalize_output_row(
            row,
            config="niah_multiquery",
            configured_length=2048,
            tokenizer=WhitespaceTokenizer(),
        )


def test_original_windows_false_success_is_rejected() -> None:
    malformed = {
        "index": -1,
        "input": (
            "Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards"
        ),
        "outputs": ["8028695", "4670027", "8310696", "7938875"],
        "length": 155,
        "length_w_model_temp": 155,
        "answer_prefix": ".",
        "token_position_answer": 26,
    }

    with pytest.raises(ValueError, match="task marker"):
        ruler._normalize_output_row(
            malformed,
            config="niah_multiquery",
            configured_length=2048,
            tokenizer=WhitespaceTokenizer(),
        )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        ruler._strict_json(b'{"a":1,"a":2}', context="fixture")


def test_atomic_publish_refuses_to_replace_existing_artifact(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    ruler._atomic_publish_new(path, b"first\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ruler._atomic_publish_new(path, b"second\n")

    assert path.read_bytes() == b"first\n"
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []


def test_atomic_publish_same_is_idempotent_but_rejects_drift(tmp_path) -> None:
    path = tmp_path / "generation-manifest.json"
    ruler._atomic_publish_same(path, b"same\n")
    ruler._atomic_publish_same(path, b"same\n")

    with pytest.raises(FileExistsError, match="differs"):
        ruler._atomic_publish_same(path, b"different\n")


def test_subprocess_environment_removes_python_injection(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHOME", "untrusted-home")
    monkeypatch.setenv("PYTHONPATH", "untrusted-path")

    env = ruler._subprocess_env(TEST_MARKER="bound")

    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["TEST_MARKER"] == "bound"


def test_independent_tokenizer_uses_verified_isolated_python(monkeypatch, tmp_path) -> None:
    python = tmp_path / "python.exe"
    tokenizer_dir = tmp_path / "tokenizer"
    python.write_bytes(b"fixture")
    tokenizer_dir.mkdir()
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        observed["request"] = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, stdout='{"count":3}\n', stderr="")

    monkeypatch.setattr(ruler.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONHOME", "untrusted-home")
    tokenizer = ruler.IndependentTokenizer(python=python, tokenizer_dir=tokenizer_dir)

    assert tokenizer.count_tokens("one two three") == 3
    assert observed["argv"][:2] == [str(python.resolve()), "-I"]
    assert observed["request"] == {
        "text": "one two three",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
    }
    assert "PYTHONHOME" not in observed["env"]
    assert observed["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_source_config_correspondence_fails_on_argument_drift() -> None:
    source_task_for_script = {script: task for task, script in ruler.SOURCE_SCRIPT_BY_TASK.items()}
    configs = {}
    tasks = {}
    for config, spec in ruler.TASK_SPECS.items():
        task = source_task_for_script[spec.script]
        extras = ruler.LAUNCHER_ONLY_ARGUMENTS.get(task, {})
        configs[config] = {
            "task": task,
            "args": {name: value for name, value in spec.arguments if name not in extras},
        }
        tasks[task] = {
            "tokens_to_generate": spec.tokens_to_generate,
            "template": spec.template,
            "answer_prefix": "",
        }
    yaml_lines: list[str] = []
    for config, value in configs.items():
        yaml_lines.extend(
            [
                f"{config}:",
                f"  task: {value['task']}",
                "  args:",
                *[f"    {name}: {argument}" for name, argument in value["args"].items()],
            ]
        )
    synthetic_yaml = "\n".join(yaml_lines)
    constants_py = f"TASKS = {tasks!r}\n".encode()

    ruler._verify_task_specs_against_source(
        synthetic_yaml=synthetic_yaml.encode(), constants_py=constants_py
    )
    drifted = synthetic_yaml.replace("num_needle_q: 4", "num_needle_q: 3", 1)
    with pytest.raises(ValueError, match="launcher arguments differ"):
        ruler._verify_task_specs_against_source(
            synthetic_yaml=drifted.encode(), constants_py=constants_py
        )


def test_frozen_runtime_requirements_match_the_probe_contract() -> None:
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements" / "experiment013-ruler.txt"
    ).read_text(encoding="utf-8")
    pinned = {
        line.split("==", 1)[0]: line.split("==", 1)[1]
        for line in requirements.splitlines()
        if line and not line.startswith("#")
    }

    assert pinned == ruler.RUNTIME_PACKAGES
    assert "torch" not in {ruler._canonical_distribution_name(name) for name in pinned}
    assert "torch" in ruler.FORBIDDEN_RUNTIME_MODULES
