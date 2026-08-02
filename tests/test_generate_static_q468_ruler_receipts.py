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


def _minimal_recorded_package_root(root: Path, *, pth_payload: str = "") -> Path:
    root.mkdir(parents=True)
    extra_paths = ("fixture.py", "hostile.pth")
    (root / "fixture.py").write_text("VALUE = 'package-bound'\n", encoding="utf-8")
    (root / "hostile.pth").write_text(pth_payload, encoding="utf-8")
    first = True
    for name, version in ruler.RUNTIME_PACKAGES.items():
        stem = ruler._canonical_distribution_name(name).replace("-", "_")
        info_name = f"{stem}-{version}.dist-info"
        info = root / info_name
        info.mkdir()
        metadata_path = info / "METADATA"
        metadata_path.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
        record_paths = [f"{info_name}/METADATA", f"{info_name}/RECORD"]
        if first:
            record_paths.extend(extra_paths)
            first = False
        (info / "RECORD").write_text(
            "".join(f"{path},,\n" for path in record_paths),
            encoding="utf-8",
        )
    return root


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


def test_capture_freezes_every_exact_launcher_command_manifest() -> None:
    capture = ruler._load_capture_module()
    launcher_entry = ruler._launcher_source_entry()
    observed: dict[str, str] = {}
    for receipt in capture.required_ruler_receipts():
        _actual, portable = ruler.generator_argv(
            python=Path("python.exe"),
            package_root=Path("site-packages"),
            staged_root=Path("staged"),
            raw_root=Path("raw"),
            receipt=receipt,
        )
        command = ruler._command_manifest(
            receipt=receipt,
            portable_argv=portable,
            capture=capture,
            static_entries=[launcher_entry],
        )
        observed[receipt["filename"]] = ruler._sha256_bytes(ruler._canonical_json_bytes(command))

    assert observed == capture.RULER_COMMAND_MANIFEST_SHA256_BY_FILENAME


def test_generator_argv_preserves_multiline_template_as_one_argument(tmp_path) -> None:
    receipt = {
        "config": "niah_multiquery",
        "configured_length": 2048,
        "seed": 12339,
    }

    actual, portable = ruler.generator_argv(
        python=tmp_path / "python.exe",
        package_root=tmp_path / "site-packages",
        staged_root=tmp_path / "staged-ruler",
        raw_root=tmp_path / "raw",
        receipt=receipt,
    )

    template_index = actual.index("--template") + 1
    assert actual[template_index] == ruler.NIAH_TEMPLATE
    assert "\n{context}\n" in actual[template_index]
    assert portable[0] == "<RULER_PYTHON>"
    # ``-I`` ignores PYTHONDONTWRITEBYTECODE, so ``-B`` must be explicit to
    # keep imported bytecode out of the authenticated staged source tree.
    assert portable[1:9] == [
        "-I",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=<EMPTY_PYCACHE_PREFIX>",
        "-X",
        "utf8",
        "-c",
    ]
    assert portable[10] == "<RULER_SITE_PACKAGES>"
    assert portable[11] == "<EMPTY_PYCACHE_PREFIX>"
    assert portable[12] == "<STAGED_RULER_DATA_ROOT>"
    assert portable[13] == "synthetic/niah.py"
    assert portable[15] == "<RAW_RECEIPT_DIR>"


def test_isolated_bootstrap_imports_authenticated_sibling_module(tmp_path) -> None:
    root = tmp_path / "data"
    script_dir = root / "synthetic"
    script_dir.mkdir(parents=True)
    (script_dir / "constants.py").write_text("VALUE = 'bound'\n", encoding="utf-8")
    (script_dir / "task.py").write_text(
        "from constants import VALUE\nfrom fixture import VALUE as PACKAGE_VALUE\n"
        "print(VALUE + ':' + PACKAGE_VALUE)\n",
        encoding="utf-8",
    )
    marker = tmp_path / "pth-executed"
    package_root = _minimal_recorded_package_root(
        tmp_path / "site-packages",
        pth_payload=f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')\n",
    )
    adjacent_pycache = package_root / "__pycache__"
    adjacent_pycache.mkdir()
    (adjacent_pycache / f"fixture.{sys.implementation.cache_tag}.pyc").write_bytes(
        b"unbound-adjacent-bytecode"
    )
    pycache_prefix = tmp_path / "empty-pycache"
    pycache_prefix.mkdir()

    result = subprocess.run(
        ruler._sealed_python_argv(
            python=Path(sys.executable),
            package_root=package_root,
            pycache_prefix=pycache_prefix,
            code=ruler.ISOLATED_SOURCE_BOOTSTRAP,
            arguments=(str(root), "synthetic/task.py"),
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "bound:package-bound\n"
    assert not marker.exists()
    assert not any(pycache_prefix.iterdir())
    assert not any(root.rglob("*.pyc"))


def test_sealed_bootstrap_rejects_unrecorded_shadow_module(tmp_path: Path) -> None:
    package_root = _minimal_recorded_package_root(tmp_path / "site-packages")
    (package_root / "rogue.py").write_text("VALUE = 'unbound'\n", encoding="utf-8")
    pycache_prefix = tmp_path / "empty-pycache"
    pycache_prefix.mkdir()

    result = subprocess.run(
        ruler._sealed_python_argv(
            python=Path(sys.executable),
            package_root=package_root,
            pycache_prefix=pycache_prefix,
            code=ruler.SEALED_STARTUP_BOOTSTRAP,
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )

    assert result.returncode != 0
    assert "unrecorded importable code" in result.stderr


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


def test_retry_reclaims_only_owned_staging_and_complete_diagnostic_orphans(tmp_path) -> None:
    filename = "retrieval__niah_single_1__l4096__s12339.json"
    config = "niah_single_1"
    raw_root = tmp_path / "raw"
    output_dir = tmp_path / "output"
    raw_root.mkdir()
    output_dir.mkdir()
    receipt_key = ruler._sha256_bytes(filename.encode("utf-8"))[:12]
    staging = raw_root / f".rq-{receipt_key}.fixture.staging"
    staging.mkdir()
    (staging / "command-manifest.json").write_bytes(b"fixture")
    published = raw_root / filename.removesuffix(".json")
    (published / config).mkdir(parents=True)
    for relative in (
        "command-manifest.json",
        "runtime-manifest.json",
        "stdout.log",
        "stderr.log",
        f"{config}/validation.jsonl",
    ):
        (published / relative).write_bytes(b"fixture")

    recovered = ruler.recover_owned_receipt_orphans(
        filename=filename,
        config=config,
        raw_root=raw_root,
        output_dir=output_dir,
    )

    assert recovered == (staging.name, published.name)
    assert not staging.exists()
    assert not published.exists()


def test_retry_refuses_mutated_orphan_and_preserves_it(tmp_path) -> None:
    filename = "retrieval__niah_single_1__l4096__s12339.json"
    config = "niah_single_1"
    raw_root = tmp_path / "raw"
    output_dir = tmp_path / "output"
    raw_root.mkdir()
    output_dir.mkdir()
    receipt_key = ruler._sha256_bytes(filename.encode("utf-8"))[:12]
    staging = raw_root / f".rq-{receipt_key}.fixture.staging"
    staging.mkdir()
    (staging / "unowned.txt").write_bytes(b"preserve")

    with pytest.raises(ValueError, match="unexpected file"):
        ruler.recover_owned_receipt_orphans(
            filename=filename,
            config=config,
            raw_root=raw_root,
            output_dir=output_dir,
        )

    assert (staging / "unowned.txt").read_bytes() == b"preserve"


def test_retry_never_cleans_diagnostics_beside_a_published_receipt(tmp_path) -> None:
    filename = "retrieval__niah_single_1__l4096__s12339.json"
    config = "niah_single_1"
    raw_root = tmp_path / "raw"
    output_dir = tmp_path / "output"
    raw_root.mkdir()
    output_dir.mkdir()
    published = raw_root / filename.removesuffix(".json")
    published.mkdir()
    (output_dir / filename).write_bytes(b"published")

    assert (
        ruler.recover_owned_receipt_orphans(
            filename=filename,
            config=config,
            raw_root=raw_root,
            output_dir=output_dir,
        )
        == ()
    )
    assert published.is_dir()


def test_two_partial_invocations_then_full_set_publish_one_complete_manifest(
    monkeypatch, tmp_path
) -> None:
    required = [
        {
            "filename": f"receipt-{index:02d}.json",
            "phase": "calibration" if index < 16 else "stage_a",
            "category": "retrieval",
            "config": "niah_single_1",
            "configured_length": 2_048,
            "seed": index,
        }
        for index in range(20)
    ]
    output_dir = tmp_path / "receipts"
    output_dir.mkdir()
    verified: list[str] = []

    def fake_verify(*, path: Path, receipt, **_kwargs):
        verified.append(path.name)
        return {
            "category": receipt["category"],
            "command_manifest": {"fixture": path.name},
            "command_manifest_file": {
                "name": "generator/command-manifest.json",
                "sha256": "1" * 64,
                "size_bytes": 1,
            },
            "config": receipt["config"],
            "configured_length": receipt["configured_length"],
            "filename": path.name,
            "generator_reported_length": 100,
            "phase": receipt["phase"],
            "raw_validation_base64": "e30K",
            "raw_validation_file": {
                "name": "generator/raw-validation.jsonl",
                "sha256": "2" * 64,
                "size_bytes": 3,
            },
            "seed": receipt["seed"],
            "sha256": "3" * 64,
            "size_bytes": 4,
        }

    monkeypatch.setattr(ruler, "_load_existing_receipt_result", fake_verify)
    capture = type(
        "Capture",
        (),
        {"resolver": type("Resolver", (), {"RULER_REVISION": "a" * 40})()},
    )()
    kwargs = {
        "required": required,
        "output_dir": output_dir,
        "raw_root": tmp_path / "raw",
        "capture": capture,
        "python": tmp_path / "python.exe",
        "package_root": tmp_path / "site-packages",
        "staged_root": tmp_path / "staged",
        "tokenizer": object(),
        "static_entries": [
            {
                "name": "launcher/generate_static_q468_ruler_receipts.py",
                "sha256": "4" * 64,
                "size_bytes": 5,
            }
        ],
        "source_manifest": [],
        "runtime_manifest": {"schema": ruler.RUNTIME_MANIFEST_SCHEMA},
    }

    for item in required[:7]:
        (output_dir / item["filename"]).write_bytes(b"partial-one")
    assert ruler.finalize_generation_manifest_if_complete(**kwargs) is None
    assert not (output_dir / "generation-manifest.json").exists()
    for item in required[7:13]:
        (output_dir / item["filename"]).write_bytes(b"partial-two")
    assert ruler.finalize_generation_manifest_if_complete(**kwargs) is None
    assert not (output_dir / "generation-manifest.json").exists()
    for item in required[13:]:
        (output_dir / item["filename"]).write_bytes(b"complete")

    manifest = ruler.finalize_generation_manifest_if_complete(**kwargs)

    assert manifest is not None
    assert manifest["receipt_count"] == 20
    assert [item["filename"] for item in manifest["receipts"]] == [
        item["filename"] for item in required
    ]
    assert verified == [item["filename"] for item in required]
    assert list(output_dir.glob("*manifest.json")) == [output_dir / "generation-manifest.json"]


def test_subprocess_environment_removes_python_injection(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHOME", "untrusted-home")
    monkeypatch.setenv("PYTHONPATH", "untrusted-path")

    env = ruler._subprocess_env(TEST_MARKER="bound")

    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["TEST_MARKER"] == "bound"


def test_isolated_stage_contains_only_verified_blob_and_corpus_bytes(tmp_path) -> None:
    checkout = ruler.VerifiedRulerCheckout(
        source_manifest=(),
        source_files={"scripts/data/synthetic/task.py": b"print('verified')\n"},
    )
    static_inputs = ruler.VerifiedStaticInputs(
        entries=(), corpus_files={"fixture.json": b"{}\n"}, sealed_runtime_files={}
    )
    staged_root = tmp_path / "staged"

    manifest = ruler.stage_verified_ruler_source(
        staged_root, checkout=checkout, static_inputs=static_inputs
    )

    assert sorted(manifest) == [
        "scripts/data/synthetic/json/fixture.json",
        "scripts/data/synthetic/task.py",
    ]
    (staged_root / "unexpected.py").write_bytes(b"shadow")
    with pytest.raises(ValueError, match="inventory drifted"):
        ruler.verify_staged_ruler_source(staged_root, expected=manifest)


def test_independent_tokenizer_uses_verified_isolated_python(monkeypatch, tmp_path) -> None:
    python_runtime_root = tmp_path / "python-runtime"
    python_runtime_root.mkdir()
    python = python_runtime_root / "python.exe"
    package_runtime_root = tmp_path / "package-runtime"
    package_root = package_runtime_root / "Lib" / "site-packages"
    package_root.mkdir(parents=True)
    runtime_input_root = tmp_path / "runtime-inputs"
    tokenizer_dir = runtime_input_root / "tokenizer"
    python.write_bytes(b"fixture")
    tokenizer_dir.mkdir(parents=True)
    python_runtime_manifest = [ruler._tree_file_entry("python.exe", python.read_bytes())]
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        observed["encoding"] = kwargs["encoding"]
        observed["errors"] = kwargs["errors"]
        observed["timeout"] = kwargs["timeout"]
        observed["request"] = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, stdout='{"count":3}\n', stderr="")

    monkeypatch.setattr(ruler.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONHOME", "untrusted-home")
    tokenizer = ruler.IndependentTokenizer(
        python=python,
        tokenizer_dir=tokenizer_dir,
        package_root=package_root,
        package_tree_manifest=(),
        runtime_input_root=runtime_input_root,
        runtime_input_manifest={},
        python_runtime_root=python_runtime_root,
        python_runtime_manifest=python_runtime_manifest,
    )

    assert tokenizer.count_tokens("one two three \u0e04\u0e33\u0e16\u0e32\u0e21") == 3
    assert observed["argv"][:9] == [
        str(python.resolve()),
        "-I",
        "-S",
        "-B",
        "-X",
        observed["argv"][5],
        "-X",
        "utf8",
        "-c",
    ]
    assert str(observed["argv"][5]).startswith("pycache_prefix=")
    assert observed["request"] == {
        "text": "one two three \u0e04\u0e33\u0e16\u0e32\u0e21",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
    }
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "strict"
    assert observed["timeout"] == ruler.TOKENIZER_TIMEOUT_SECONDS
    assert observed["argv"][10] == str(package_root.resolve())
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


def test_produced_runtime_manifest_round_trips_through_capture_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python_runtime_root = tmp_path / "python-runtime"
    python_runtime_root.mkdir()
    python = python_runtime_root / "python.exe"
    python.write_bytes(b"fixture-ruler-python")
    for name, data in (
        ("python3.dll", b"fixture-python3-dll"),
        ("python311.dll", b"fixture-python311-dll"),
    ):
        (python_runtime_root / name).write_bytes(data)
    python_runtime_manifest = [
        ruler._tree_file_entry(path.name, path.read_bytes())
        for path in sorted(python_runtime_root.iterdir(), key=lambda path: path.name)
    ]
    package_runtime_root = tmp_path / "package-runtime"
    package_root = package_runtime_root / "Lib" / "site-packages"
    package_root.mkdir(parents=True)
    inventories: dict[str, object] = {}
    for name, version in ruler.RUNTIME_PACKAGES.items():
        canonical_name = ruler._canonical_distribution_name(name)
        record = f"{name}=={version}\n".encode()
        inventories[name] = {
            "canonical_name": canonical_name,
            "version": version,
            "record_sha256": ruler._sha256_bytes(record),
            "record_size_bytes": len(record),
            "files": [
                {
                    "path": f"{canonical_name}-{version}.dist-info/RECORD",
                    "sha256": ruler._sha256_bytes(record),
                    "size_bytes": len(record),
                }
            ],
        }
    payload = {
        "python": ruler.RUNTIME_PYTHON_VERSION,
        "implementation": "cpython",
        "cache_tag": "cpython-311",
        "executable": str(python.resolve()),
        "platform": "fixture-platform",
        "machine": "fixture-machine",
        "flags": {
            "ignore_environment": 1,
            "isolated": 1,
            "no_user_site": 1,
        },
        "startup_policy": dict(ruler.SEALED_STARTUP_POLICY),
        "packages": dict(ruler.RUNTIME_PACKAGES),
        "installed_distributions": {
            ruler._canonical_distribution_name(name): version
            for name, version in ruler.RUNTIME_PACKAGES.items()
        },
        "distribution_file_inventory": inventories,
        "forbidden_modules": {name: False for name in ruler.FORBIDDEN_RUNTIME_MODULES},
        "resources": {
            name: str(tmp_path / name.replace("/", "_"))
            for name in ruler.EXPECTED_PACKAGE_RESOURCES
        },
    }

    observed: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[:5] == [str(python.resolve()), "-I", "-S", "-B", "-X"]
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            stderr="",
        )

    monkeypatch.setattr(ruler.subprocess, "run", fake_run)
    excluded_startup_files = [
        {"name": name, "sha256": digest, "size_bytes": size}
        for name, (size, digest) in sorted(ruler.EXPECTED_EXCLUDED_VIRTUALENV_STARTUP_FILES.items())
    ]
    produced, _resource_paths = ruler.verify_runtime(
        python,
        tmp_path / "nltk-data",
        package_root=package_root,
        package_tree_manifest=(),
        excluded_startup_files=excluded_startup_files,
        python_runtime_root=python_runtime_root,
        python_runtime_manifest=python_runtime_manifest,
        source_python=ruler._tree_file_entry("source/python.exe", b"source-python"),
        source_pyvenv_config=ruler._tree_file_entry("source/pyvenv.cfg", b"source-pyvenv"),
    )
    capture = ruler._load_capture_module()

    assert capture.RULER_RUNTIME_MANIFEST_SCHEMA == ruler.RUNTIME_MANIFEST_SCHEMA
    assert capture.RULER_LAUNCHER_REVISION == ruler.LAUNCHER_REVISION
    assert capture._normalize_ruler_runtime_manifest(produced) == produced
    assert observed["timeout"] == ruler.RUNTIME_PROBE_TIMEOUT_SECONDS
