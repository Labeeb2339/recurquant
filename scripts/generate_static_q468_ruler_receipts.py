#!/usr/bin/env python3
"""Generate authenticated Experiment 013 RULER receipts without model weights.

NVIDIA RULER's pinned ``prepare.py`` builds a multiline shell string. On
Windows that string truncates the multiline task template while still exiting
zero and writing a malformed row. This launcher verifies the pinned upstream
files, then invokes the same pinned task scripts with an argument vector. It
independently checks the generated row, tokenizer count, runtime, corpora, and
package resources before atomically publishing a receipt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = REPOSITORY_ROOT / "scripts" / "capture_static_q468_identity_input.py"

LAUNCHER_REVISION: Final = "experiment-013-ruler-argv-launcher-v2"
RUNTIME_PYTHON_VERSION: Final = "3.11.15"
RUNTIME_PACKAGES: Final = {
    "PyYAML": "6.0.3",
    "annotated-doc": "0.0.5",
    "anyio": "4.14.2",
    "beautifulsoup4": "4.15.0",
    "certifi": "2026.7.22",
    "click": "8.4.2",
    "colorama": "0.4.6",
    "defusedxml": "0.7.1",
    "filelock": "3.32.2",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.5.2",
    "html2text": "2025.4.15",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.26.0",
    "idna": "3.18",
    "joblib": "1.5.3",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "nltk": "3.8.1",
    "numpy": "2.4.6",
    "packaging": "26.2",
    "pygments": "2.20.0",
    "regex": "2026.7.19",
    "rich": "15.0.0",
    "safetensors": "0.8.0",
    "scipy": "1.17.1",
    "shellingham": "1.5.4",
    "soupsieve": "2.9.1",
    "tenacity": "9.1.4",
    "tokenizers": "0.22.2",
    "tqdm": "4.70.0",
    "transformers": "5.14.1",
    "typer": "0.27.0",
    "typing-extensions": "4.16.0",
    "wonderwords": "3.0.1",
}

EXPECTED_CORPORA: Final = {
    "PaulGrahamEssays.json": (
        3_108_621,
        "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd",
    ),
    "english_words.json": (
        8_564_991,
        "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    ),
    "hotpotqa.json": (
        61_065_698,
        "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    ),
    "squad.json": (
        4_370_528,
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
}
EXPECTED_PACKAGE_RESOURCES: Final = {
    "nltk/punkt/PY3/english.pickle": (
        406_697,
        "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c",
    ),
    "wonderwords/adjectivelist.txt": (
        7_403,
        "66814d46b7e292c83e839d12fe2fa083c8f66779b14b294e1be4e1342c5d4131",
    ),
    "wonderwords/nounlist.txt": (
        54_752,
        "8d88b3ebc2e2969ed92ebe49cdb0c8a87c2ad3b27493d7afdff608f2e51c5bc9",
    ),
    "wonderwords/verblist.txt": (
        6_967,
        "9fbf5e4e69b8869aebd88e6d545312ff9f878a747327ede5f6dbbba5c27b08ad",
    ),
}
EXPECTED_TOKENIZER_ASSETS: Final = {
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "tokenizer.json": (
        12_807_196,
        "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    ),
    "tokenizer_config.json": (
        16_712,
        "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
}

NIAH_TEMPLATE: Final = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned "
    "in the provided text? The special magic {type_needle_v} for {query} mentioned "
    "in the provided text are"
)
VT_TEMPLATE: Final = (
    "Memorize and track the chain(s) of variable assignment hidden in the following "
    "text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} "
    "in the text above. Answer: According to the chain(s) of variable assignment in "
    "the text above, {num_v} variables are assigned the value {query}, they are: "
)
CWE_TEMPLATE: Final = (
    "Below is a numbered list of words. In these words, some appear more often than "
    "others. Memorize the ones that appear most often.\n{context}\nQuestion: What are "
    "the 10 most common words in the above list? Answer: The top 10 words that appear "
    "most often in the list are:"
)
FWE_TEMPLATE: Final = (
    "Read the following coded text and track the frequency of each coded word. Find the "
    "three most frequently appeared coded words. {context}\nQuestion: Do not provide any "
    "explanation. Please ignore the dots '....'. What are the three most frequently "
    "appeared words in the above coded text? Answer: According to the coded text above, "
    "the three most frequently appeared words are:"
)
QA_TEMPLATE: Final = (
    "Answer the question based on the given documents. Only give me the answer and do "
    "not output any other words.\n\nThe following are given documents.\n\n{context}\n\n"
    "Answer the question based on the given documents. Only give me the answer and do "
    "not output any other words.\n\nQuestion: {query} Answer:"
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    script: str
    tokens_to_generate: int
    template: str
    arguments: tuple[tuple[str, str], ...]
    input_marker: str
    answer_prefix_marker: str
    expected_output_count: int | None
    unique_outputs: bool
    output_pattern: str | None = None
    outputs_must_appear: bool = False
    niah: bool = False


TASK_SPECS: Final = {
    "niah_multiquery": TaskSpec(
        "niah.py",
        128,
        NIAH_TEMPLATE,
        (
            ("type_haystack", "essay"),
            ("type_needle_k", "words"),
            ("type_needle_v", "numbers"),
            ("num_needle_k", "1"),
            ("num_needle_v", "1"),
            ("num_needle_q", "4"),
        ),
        "What are all the special magic numbers",
        " The special magic numbers",
        4,
        True,
        r"[0-9]{7}",
        True,
        True,
    ),
    "niah_multikey_2": TaskSpec(
        "niah.py",
        128,
        NIAH_TEMPLATE,
        (
            ("type_haystack", "needle"),
            ("type_needle_k", "words"),
            ("type_needle_v", "numbers"),
            ("num_needle_k", "1"),
            ("num_needle_v", "1"),
            ("num_needle_q", "1"),
        ),
        "What is the special magic number",
        " The special magic number",
        1,
        True,
        r"[0-9]{7}",
        True,
        True,
    ),
    "niah_single_1": TaskSpec(
        "niah.py",
        128,
        NIAH_TEMPLATE,
        (
            ("type_haystack", "noise"),
            ("type_needle_k", "words"),
            ("type_needle_v", "numbers"),
            ("num_needle_k", "1"),
            ("num_needle_v", "1"),
            ("num_needle_q", "1"),
        ),
        "What is the special magic number",
        " The special magic number",
        1,
        True,
        r"[0-9]{7}",
        True,
        True,
    ),
    "niah_multivalue": TaskSpec(
        "niah.py",
        128,
        NIAH_TEMPLATE,
        (
            ("type_haystack", "essay"),
            ("type_needle_k", "words"),
            ("type_needle_v", "numbers"),
            ("num_needle_k", "1"),
            ("num_needle_v", "4"),
            ("num_needle_q", "1"),
        ),
        "What are all the special magic numbers",
        " The special magic numbers",
        4,
        True,
        r"[0-9]{7}",
        True,
        True,
    ),
    "vt": TaskSpec(
        "variable_tracking.py",
        30,
        VT_TEMPLATE,
        (("type_haystack", "noise"), ("num_chains", "1"), ("num_hops", "4")),
        "Question: Find all variables",
        " Answer:",
        5,
        True,
        r"[A-Z]{5}",
        True,
    ),
    "cwe": TaskSpec(
        "common_words_extraction.py",
        120,
        CWE_TEMPLATE,
        (("freq_cw", "30"), ("freq_ucw", "3"), ("num_cw", "10")),
        "Question: What are the 10 most common words",
        " Answer:",
        10,
        True,
        None,
        True,
    ),
    "fwe": TaskSpec(
        "freq_words_extraction.py",
        50,
        FWE_TEMPLATE,
        (("alpha", "2.0"),),
        "What are the three most frequently appeared words",
        " Answer:",
        3,
        True,
        r"[a-z]{6}",
        True,
    ),
    "qa_1": TaskSpec(
        "qa.py",
        32,
        QA_TEMPLATE,
        (("dataset", "squad"), ("pre_samples", "0")),
        "The following are given documents.",
        " Answer:",
        None,
        False,
    ),
    "qa_2": TaskSpec(
        "qa.py",
        32,
        QA_TEMPLATE,
        (("dataset", "hotpotqa"), ("pre_samples", "0")),
        "The following are given documents.",
        " Answer:",
        None,
        False,
    ),
}

SOURCE_SCRIPT_BY_TASK: Final = {
    "niah": "niah.py",
    "variable_tracking": "variable_tracking.py",
    "common_words_extraction": "common_words_extraction.py",
    "freq_words_extraction": "freq_words_extraction.py",
    "qa": "qa.py",
}
LAUNCHER_ONLY_ARGUMENTS: Final = {"qa": {"pre_samples": "0"}}
FORBIDDEN_RUNTIME_MODULES: Final = (
    "accelerate",
    "bitsandbytes",
    "flax",
    "jax",
    "onnx",
    "onnxruntime",
    "tensorflow",
    "torch",
)


def _load_capture_module() -> Any:
    name = "_recurquant_experiment013_ruler_capture"
    spec = importlib.util.spec_from_file_location(name, CAPTURE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load the Experiment 013 capture module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _subprocess_env(**updates: str) -> dict[str, str]:
    """Return a child environment without caller-controlled Python injection."""

    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    env.update(updates)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _file_entry(name: str, data: bytes) -> dict[str, object]:
    if not data:
        raise ValueError(f"bound file {name!r} is empty")
    return {"name": name, "sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _verified_file(path: Path, *, size: int, sha256: str, name: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"missing required {name}: {path}")
    data = path.read_bytes()
    if len(data) != size or _sha256_bytes(data) != sha256:
        raise ValueError(f"{name} differs from the frozen Experiment 013 bytes")
    return data


def _strict_json(data: bytes, *, context: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from error


def _git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode()
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _source_tasks(constants_data: bytes) -> Mapping[str, object]:
    try:
        tree = ast.parse(constants_data.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("pinned RULER constants.py is not valid UTF-8 Python") from error
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("pinned RULER constants.py must assign TASKS exactly once")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError) as error:
        raise ValueError("pinned RULER TASKS must be a literal mapping") from error
    if not isinstance(value, Mapping):
        raise ValueError("pinned RULER TASKS must be a mapping")
    return value


def _verify_task_specs_against_source(*, synthetic_yaml: bytes, constants_py: bytes) -> None:
    import yaml

    try:
        configs = yaml.safe_load(synthetic_yaml.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("pinned RULER synthetic.yaml is invalid") from error
    if not isinstance(configs, Mapping):
        raise ValueError("pinned RULER synthetic.yaml must be a mapping")
    tasks = _source_tasks(constants_py)
    for config, spec in TASK_SPECS.items():
        source_config = configs.get(config)
        if not isinstance(source_config, Mapping) or set(source_config) != {"task", "args"}:
            raise ValueError(f"pinned RULER config {config!r} has an unexpected shape")
        task = source_config["task"]
        arguments = source_config["args"]
        if not isinstance(task, str) or not isinstance(arguments, Mapping):
            raise ValueError(f"pinned RULER config {config!r} is malformed")
        if SOURCE_SCRIPT_BY_TASK.get(task) != spec.script:
            raise ValueError(f"launcher script differs from pinned config {config!r}")
        expected_arguments = {str(key): str(value) for key, value in arguments.items()}
        launcher_arguments = dict(spec.arguments)
        if len(launcher_arguments) != len(spec.arguments):
            raise ValueError(f"launcher arguments repeat a key for config {config!r}")
        extras = LAUNCHER_ONLY_ARGUMENTS.get(task, {})
        if launcher_arguments != expected_arguments | extras:
            raise ValueError(f"launcher arguments differ from pinned config {config!r}")
        source_task = tasks.get(task)
        if not isinstance(source_task, Mapping):
            raise ValueError(f"pinned RULER task {task!r} is missing")
        if source_task.get("tokens_to_generate") != spec.tokens_to_generate:
            raise ValueError(f"launcher token budget differs from pinned task {task!r}")
        template = source_task.get("template")
        answer_prefix = source_task.get("answer_prefix", "")
        if not isinstance(template, str) or not isinstance(answer_prefix, str):
            raise ValueError(f"pinned RULER task {task!r} has malformed templates")
        if template + answer_prefix != spec.template:
            raise ValueError(f"launcher template differs from pinned task {task!r}")


def _requirements_bytes() -> bytes:
    path = REPOSITORY_ROOT / "requirements" / "experiment013-ruler.txt"
    data = path.read_bytes()
    pinned: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.count("==") != 1:
            raise ValueError("RULER runtime requirements must use exact == pins")
        name, version = stripped.split("==", 1)
        if not name or not version or name in pinned:
            raise ValueError("RULER runtime requirements contain an invalid pin")
        pinned[name] = version
    if pinned != RUNTIME_PACKAGES:
        raise ValueError("RULER runtime requirements differ from the probe contract")
    return data


def _launcher_source_entry() -> dict[str, object]:
    return _file_entry(
        "launcher/generate_static_q468_ruler_receipts.py", Path(__file__).read_bytes()
    )


def verify_ruler_checkout(ruler_root: Path, capture: Any) -> list[dict[str, object]]:
    ruler_root = ruler_root.resolve()
    result = subprocess.run(
        ["git", "-C", str(ruler_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if head != capture.resolver.RULER_REVISION:
        raise ValueError("RULER checkout HEAD differs from the frozen revision")
    files: dict[str, bytes] = {}
    for relative, expected_blob in capture.RULER_GENERATOR_GIT_BLOBS.items():
        path = ruler_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned RULER source file: {relative}")
        # Git may legitimately smudge LF blobs to CRLF in a Windows checkout.
        # Hash the worktree path through its clean filters, then bind the exact
        # immutable object bytes used by the capture contract.
        worktree_hash = subprocess.run(
            [
                "git",
                "-C",
                str(ruler_root),
                "hash-object",
                f"--path={relative}",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if worktree_hash != expected_blob:
            raise ValueError(f"pinned RULER source file drifted: {relative}")
        data = subprocess.run(
            ["git", "-C", str(ruler_root), "cat-file", "blob", expected_blob],
            check=True,
            capture_output=True,
        ).stdout
        if _git_blob_sha1(data) != expected_blob:
            raise RuntimeError(f"Git returned corrupt blob bytes for {relative}")
        files[relative] = data
    _verify_task_specs_against_source(
        synthetic_yaml=files["scripts/synthetic.yaml"],
        constants_py=files["scripts/data/synthetic/constants.py"],
    )
    return capture._ruler_generator_manifest(files)


def verify_runtime(python: Path, nltk_data: Path) -> tuple[dict[str, object], dict[str, Path]]:
    code = """
import importlib.metadata as metadata
import importlib.util
import json
import pathlib
import platform
import re
import sys
import nltk
import wonderwords
names = __PACKAGE_NAMES__
forbidden = __FORBIDDEN_MODULES__
root = pathlib.Path(wonderwords.__file__).resolve().parent
canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()
payload = {
    'python': sys.version.split()[0],
    'implementation': sys.implementation.name,
    'cache_tag': sys.implementation.cache_tag,
    'executable': str(pathlib.Path(sys.executable).resolve()),
    'platform': platform.platform(),
    'flags': {
        'ignore_environment': sys.flags.ignore_environment,
        'isolated': sys.flags.isolated,
        'no_user_site': sys.flags.no_user_site,
    },
    'packages': {name: metadata.version(name) for name in names},
    'installed_distributions': {
        canonical(dist.metadata['Name']): dist.version for dist in metadata.distributions()
    },
    'forbidden_modules': {
        name: importlib.util.find_spec(name) is not None for name in forbidden
    },
    'resources': {
        'nltk/punkt/PY3/english.pickle': str(nltk.data.find('tokenizers/punkt/english.pickle')),
        'wonderwords/adjectivelist.txt': str(root / 'assets' / 'adjectivelist.txt'),
        'wonderwords/nounlist.txt': str(root / 'assets' / 'nounlist.txt'),
        'wonderwords/verblist.txt': str(root / 'assets' / 'verblist.txt'),
    },
}
print(json.dumps(payload, sort_keys=True, separators=(',', ':')))
""".replace("__PACKAGE_NAMES__", repr(sorted(RUNTIME_PACKAGES))).replace(
        "__FORBIDDEN_MODULES__", repr(FORBIDDEN_RUNTIME_MODULES)
    )
    python = python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"missing required RULER Python executable: {python}")
    env = _subprocess_env(
        NLTK_DATA=str(nltk_data.resolve()),
        TOKENIZERS_PARALLELISM="false",
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = _strict_json(result.stdout.encode(), context="RULER runtime probe")
    if not isinstance(payload, Mapping):
        raise ValueError("RULER runtime probe did not return an object")
    if payload.get("python") != RUNTIME_PYTHON_VERSION:
        raise ValueError("RULER Python patch version drifted")
    if payload.get("packages") != RUNTIME_PACKAGES:
        raise ValueError("RULER package versions drifted")
    expected_distributions = {
        _canonical_distribution_name(name): version for name, version in RUNTIME_PACKAGES.items()
    }
    if payload.get("installed_distributions") != expected_distributions:
        raise ValueError("RULER installed-distribution inventory drifted")
    expected_absence = {name: False for name in FORBIDDEN_RUNTIME_MODULES}
    if payload.get("forbidden_modules") != expected_absence:
        raise ValueError("RULER runtime contains a forbidden model framework")
    if payload.get("implementation") != "cpython":
        raise ValueError("RULER Python implementation drifted")
    flags = payload.get("flags")
    if flags != {"ignore_environment": 1, "isolated": 1, "no_user_site": 1}:
        raise ValueError("RULER runtime probe was not isolated")
    executable = payload.get("executable")
    if not isinstance(executable, str) or not Path(executable).samefile(python):
        raise ValueError("RULER runtime probe used a different Python executable")
    for field in ("cache_tag", "platform"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"RULER runtime probe omitted {field}")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != set(EXPECTED_PACKAGE_RESOURCES):
        raise ValueError("RULER package-resource inventory drifted")
    paths: dict[str, Path] = {}
    for name, raw_path in resources.items():
        if not isinstance(raw_path, str):
            raise ValueError("RULER runtime returned a non-string resource path")
        paths[str(name)] = Path(raw_path)
    executable_data = python.read_bytes()
    runtime_manifest = {
        "python": payload["python"],
        "implementation": payload["implementation"],
        "cache_tag": payload["cache_tag"],
        "platform": payload["platform"],
        "flags": payload["flags"],
        "executable": _file_entry("python.exe", executable_data),
        "packages": payload["packages"],
        "installed_distributions": payload["installed_distributions"],
        "forbidden_modules": payload["forbidden_modules"],
    }
    return runtime_manifest, paths


def verify_static_inputs(
    *,
    ruler_root: Path,
    tokenizer_dir: Path,
    resource_paths: Mapping[str, Path],
    runtime_manifest: Mapping[str, object],
    capture: Any,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    corpus_root = ruler_root / "scripts" / "data" / "synthetic" / "json"
    for name, (size, sha256) in sorted(EXPECTED_CORPORA.items()):
        data = _verified_file(
            corpus_root / name,
            size=size,
            sha256=sha256,
            name=f"RULER corpus {name}",
        )
        entries.append(_file_entry(f"corpora/{name}", data))
    for name, (size, sha256) in sorted(EXPECTED_PACKAGE_RESOURCES.items()):
        data = _verified_file(
            resource_paths[name],
            size=size,
            sha256=sha256,
            name=f"RULER package resource {name}",
        )
        entries.append(_file_entry(f"packages/{name}", data))

    tokenizer_dir = tokenizer_dir.resolve()
    if tokenizer_dir.name != capture.resolver.PRIMARY_MODEL_REVISION:
        raise ValueError("tokenizer snapshot directory does not name the frozen revision")
    files = {path.name: path for path in tokenizer_dir.iterdir() if path.is_file()}
    if set(files) != set(EXPECTED_TOKENIZER_ASSETS):
        raise ValueError("tokenizer-only asset inventory drifted")
    for path in tokenizer_dir.rglob("*"):
        if path.is_file() and capture.FORBIDDEN_MODEL_FILE_RE.search(
            path.relative_to(tokenizer_dir).as_posix()
        ):
            raise ValueError(f"model-weight-like file is forbidden in tokenizer snapshot: {path}")
    for name, (size, sha256) in sorted(EXPECTED_TOKENIZER_ASSETS.items()):
        data = _verified_file(
            files[name],
            size=size,
            sha256=sha256,
            name=f"tokenizer asset {name}",
        )
        entries.append(_file_entry(f"tokenizer/{name}", data))

    runtime_bytes = _canonical_json_bytes(runtime_manifest)
    entries.append(_file_entry("runtime/package-manifest.json", runtime_bytes))
    entries.append(_file_entry("runtime/requirements.txt", _requirements_bytes()))
    entries.append(_launcher_source_entry())

    return sorted(entries, key=lambda item: str(item["name"]))


class IndependentTokenizer:
    """Recompute token counts inside the verified tokenizer-only interpreter."""

    def __init__(self, *, python: Path, tokenizer_dir: Path) -> None:
        self._python = python.resolve()
        self._tokenizer_dir = tokenizer_dir.resolve()

    def count_tokens(self, text: str) -> int:
        code = """
import json
import sys
from transformers import AutoTokenizer
request = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained(
    request['tokenizer_dir'], local_files_only=True, trust_remote_code=False
)
print(json.dumps({'count': len(tokenizer.tokenize(request['text']))}))
"""
        request = {"text": text, "tokenizer_dir": str(self._tokenizer_dir)}
        result = subprocess.run(
            [str(self._python), "-I", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            env=_subprocess_env(
                HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                TRANSFORMERS_OFFLINE="1",
            ),
        )
        payload = _strict_json(result.stdout.encode(), context="independent tokenizer count")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"count"}
            or isinstance(payload["count"], bool)
            or not isinstance(payload["count"], int)
            or payload["count"] < 0
        ):
            raise ValueError("independent tokenizer returned an invalid token count")
        return int(payload["count"])


def _token_count(tokenizer: Any, text: str) -> int:
    count = getattr(tokenizer, "count_tokens", None)
    value = count(text) if callable(count) else len(tokenizer.tokenize(text))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("tokenizer returned an invalid token count")
    return value


def generator_argv(
    *,
    python: Path,
    ruler_root: Path,
    raw_root: Path,
    receipt: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    config = str(receipt["config"])
    try:
        spec = TASK_SPECS[config]
    except KeyError as error:
        raise ValueError(f"no frozen launcher specification for RULER config {config}") from error
    script = ruler_root / "scripts" / "data" / "synthetic" / spec.script
    actual = [
        str(python.resolve()),
        "-s",
        str(script.resolve()),
        "--save_dir",
        str(raw_root.resolve()),
        "--save_name",
        config,
        "--subset",
        "validation",
        "--tokenizer_path",
        "<TOKENIZER_DIR>",
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(receipt["configured_length"]),
        "--tokens_to_generate",
        str(spec.tokens_to_generate),
        "--num_samples",
        "1",
        "--random_seed",
        str(receipt["seed"]),
    ]
    for name, value in spec.arguments:
        actual.extend((f"--{name}", value))
    actual.extend(("--template", spec.template))
    portable = list(actual)
    portable[0] = "<RULER_PYTHON>"
    portable[2] = f"scripts/data/synthetic/{spec.script}"
    portable[4] = "<RAW_RECEIPT_DIR>"
    return actual, portable


def _entry_named(entries: Sequence[Mapping[str, object]], name: str) -> Mapping[str, object]:
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"bound file inventory must contain exactly one {name!r}")
    return matches[0]


def _command_manifest(
    *,
    receipt: Mapping[str, object],
    portable_argv: Sequence[str],
    capture: Any,
    static_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    launcher_entry = _entry_named(static_entries, "launcher/generate_static_q468_ruler_receipts.py")
    return {
        "launcher_revision": LAUNCHER_REVISION,
        "launcher_source_sha256": launcher_entry["sha256"],
        "ruler_revision": capture.resolver.RULER_REVISION,
        "config": receipt["config"],
        "configured_length": receipt["configured_length"],
        "seed": receipt["seed"],
        "argv": list(portable_argv),
        "shell": False,
    }


def _normalize_output_row(
    row: object,
    *,
    config: str,
    configured_length: int,
    tokenizer: Any,
) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("RULER generator row must be an object")
    spec = TASK_SPECS[config]
    required = {
        "index",
        "input",
        "outputs",
        "length",
        "length_w_model_temp",
        "answer_prefix",
    }
    expected = required | ({"token_position_answer"} if spec.niah else set())
    if set(row) != expected:
        raise ValueError("RULER generator row fields drifted")
    input_text = row["input"]
    prefix = row["answer_prefix"]
    outputs = row["outputs"]
    if not isinstance(input_text, str) or not input_text:
        raise ValueError("RULER generator input must be a non-empty string")
    if spec.input_marker not in input_text:
        raise ValueError("RULER generator input is missing the frozen task marker")
    if not isinstance(prefix, str) or not prefix.startswith(spec.answer_prefix_marker):
        raise ValueError("RULER generator answer prefix is truncated or drifted")
    if (
        isinstance(outputs, (str, bytes))
        or not isinstance(outputs, Sequence)
        or not outputs
        or any(not isinstance(value, str) or not value for value in outputs)
    ):
        raise ValueError("RULER generator outputs must be non-empty strings")
    normalized_outputs = list(outputs)
    if (
        spec.expected_output_count is not None
        and len(normalized_outputs) != spec.expected_output_count
    ):
        raise ValueError(f"RULER {config} output count must equal {spec.expected_output_count}")
    if spec.unique_outputs and len(set(normalized_outputs)) != len(normalized_outputs):
        raise ValueError(f"RULER {config} outputs must be unique")
    if spec.output_pattern is not None and any(
        re.fullmatch(spec.output_pattern, output) is None for output in normalized_outputs
    ):
        raise ValueError(f"RULER {config} output format drifted")
    if spec.outputs_must_appear and any(output not in input_text for output in normalized_outputs):
        raise ValueError(f"RULER {config} required answer is absent from its input")
    if isinstance(row["length"], bool) or not isinstance(row["length"], int):
        raise ValueError("RULER generator length must be an integer")
    if row["length_w_model_temp"] != row["length"]:
        raise ValueError("base-template RULER lengths disagree")
    if not 1 <= row["length"] <= configured_length:
        raise ValueError("RULER generator length exceeds its configured bound")
    independent_length = _token_count(tokenizer, input_text + prefix) + spec.tokens_to_generate
    if independent_length != row["length"]:
        raise ValueError("RULER generator length disagrees with independent tokenization")
    index = row["index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("RULER generator index must be an integer")
    if spec.niah:
        if index < 0 or input_text.find(normalized_outputs[0]) != index:
            raise ValueError("RULER NIAH answer is not present at the reported position")
        token_position = row["token_position_answer"]
        if isinstance(token_position, bool) or not isinstance(token_position, int):
            raise ValueError("RULER NIAH token position must be an integer")
        if token_position != _token_count(tokenizer, input_text[:index]):
            raise ValueError("RULER NIAH answer token position drifted")
    elif index != 0:
        raise ValueError("single-sample RULER generator index must equal zero")
    return {
        "input": input_text,
        "answer_prefix": prefix,
        "outputs": normalized_outputs,
        "generator_reported_length": int(row["length"]),
    }


def _atomic_publish_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite RULER artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_publish_same(path: Path, data: bytes) -> None:
    """Publish new bytes, or accept an already-published byte-identical file."""

    try:
        _atomic_publish_new(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise FileExistsError(f"existing RULER artifact differs: {path}") from None


def _load_existing_receipt_result(
    *,
    path: Path,
    receipt: Mapping[str, object],
    capture: Any,
    python: Path,
    ruler_root: Path,
    raw_root: Path,
    tokenizer: Any,
    static_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload = path.read_bytes()
    value = _strict_json(payload, context=f"existing RULER receipt {path.name}")
    if not isinstance(value, Mapping) or _canonical_json_bytes(value) != payload:
        raise ValueError(f"existing RULER receipt is not canonical: {path}")
    config = str(receipt["config"])
    capture._normalize_ruler_receipt(
        value,
        category=str(receipt["category"]),
        config=config,
        configured_length=int(receipt["configured_length"]),
        seed=int(receipt["seed"]),
    )
    normalized = _normalize_output_row(
        {
            "index": (
                str(value["input"]).find(value["outputs"][0]) if TASK_SPECS[config].niah else 0
            ),
            "input": value["input"],
            "outputs": value["outputs"],
            "length": value["generator_reported_length"],
            "length_w_model_temp": value["generator_reported_length"],
            "answer_prefix": value["answer_prefix"],
            **(
                {
                    "token_position_answer": _token_count(
                        tokenizer,
                        str(value["input"])[: str(value["input"]).find(value["outputs"][0])],
                    )
                }
                if TASK_SPECS[config].niah
                else {}
            ),
        },
        config=config,
        configured_length=int(receipt["configured_length"]),
        tokenizer=tokenizer,
    )
    auxiliary = value["auxiliary_files"]
    if not isinstance(auxiliary, Sequence):  # capture normalization already rejects this
        raise ValueError("existing RULER receipt auxiliary inventory is invalid")
    auxiliary_by_name = {str(entry["name"]): entry for entry in auxiliary}
    if len(auxiliary_by_name) != len(auxiliary):
        raise ValueError("existing RULER receipt repeats an auxiliary file")
    for expected in static_entries:
        if auxiliary_by_name.get(str(expected["name"])) != expected:
            raise ValueError("existing RULER receipt has a stale static-input binding")
    _, portable = generator_argv(
        python=python,
        ruler_root=ruler_root,
        raw_root=raw_root,
        receipt=receipt,
    )
    command_bytes = _canonical_json_bytes(
        _command_manifest(
            receipt=receipt,
            portable_argv=portable,
            capture=capture,
            static_entries=static_entries,
        )
    )
    expected_command = _file_entry("generator/command-manifest.json", command_bytes)
    if auxiliary_by_name.get("generator/command-manifest.json") != expected_command:
        raise ValueError("existing RULER receipt has a stale launcher binding")
    expected_names = {str(entry["name"]) for entry in static_entries} | {
        "generator/command-manifest.json",
        "generator/raw-validation.jsonl",
    }
    if set(auxiliary_by_name) != expected_names:
        raise ValueError("existing RULER receipt auxiliary inventory drifted")
    return {
        "filename": path.name,
        "phase": receipt["phase"],
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "generator_reported_length": normalized["generator_reported_length"],
    }


def generate_receipt(
    *,
    receipt: Mapping[str, object],
    capture: Any,
    python: Path,
    ruler_root: Path,
    tokenizer_dir: Path,
    nltk_data: Path,
    raw_root: Path,
    output_dir: Path,
    tokenizer: Any,
    static_entries: Sequence[Mapping[str, object]],
    runtime_manifest: Mapping[str, object],
    timeout_seconds: int,
) -> dict[str, object]:
    filename = str(receipt["filename"])
    output_path = output_dir / filename
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RULER receipt: {output_path}")
    raw_root.mkdir(parents=True, exist_ok=True)
    receipt_key = _sha256_bytes(filename.encode("utf-8"))[:12]
    receipt_raw_root = Path(
        tempfile.mkdtemp(prefix=f".rq-{receipt_key}.", suffix=".staging", dir=raw_root)
    )

    actual, portable = generator_argv(
        python=python,
        ruler_root=ruler_root,
        raw_root=receipt_raw_root,
        receipt=receipt,
    )
    tokenizer_index = actual.index("<TOKENIZER_DIR>")
    actual[tokenizer_index] = str(tokenizer_dir.resolve())
    command_manifest = _command_manifest(
        receipt=receipt,
        portable_argv=portable,
        capture=capture,
        static_entries=static_entries,
    )
    command_bytes = _canonical_json_bytes(command_manifest)
    runtime_bytes = _canonical_json_bytes(runtime_manifest)
    _atomic_publish_new(receipt_raw_root / "command-manifest.json", command_bytes)
    _atomic_publish_new(receipt_raw_root / "runtime-manifest.json", runtime_bytes)

    env = _subprocess_env(
        HF_HUB_OFFLINE="1",
        NLTK_DATA=str(nltk_data.resolve()),
        PYTHONHASHSEED="0",
        TOKENIZERS_PARALLELISM="false",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        actual,
        cwd=ruler_root / "scripts" / "data",
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    (receipt_raw_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (receipt_raw_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"RULER generator failed for {filename} with exit code {result.returncode}"
        )
    config = str(receipt["config"])
    row_path = receipt_raw_root / config / "validation.jsonl"
    if not row_path.is_file():
        raise RuntimeError(f"RULER generator returned zero without producing {row_path}")
    raw_data = row_path.read_bytes()
    lines = raw_data.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("RULER generator must produce exactly one non-empty JSONL row")
    row = _strict_json(lines[0], context=f"RULER output {filename}")
    normalized = _normalize_output_row(
        row,
        config=config,
        configured_length=int(receipt["configured_length"]),
        tokenizer=tokenizer,
    )
    dynamic_entries = [
        _file_entry("generator/command-manifest.json", command_bytes),
        _file_entry("generator/raw-validation.jsonl", raw_data),
    ]
    auxiliary = sorted(
        [dict(item) for item in static_entries] + dynamic_entries,
        key=lambda item: str(item["name"]),
    )
    receipt_data = {
        "schema": capture.RULER_RECEIPT_SCHEMA,
        "source_id": capture.resolver.RULER_SOURCE_ID,
        "revision": capture.resolver.RULER_REVISION,
        "category": receipt["category"],
        "config": config,
        "configured_length": receipt["configured_length"],
        "seed": receipt["seed"],
        "sample_index": 0,
        **normalized,
        "auxiliary_files": auxiliary,
    }
    capture._normalize_ruler_receipt(
        receipt_data,
        category=str(receipt["category"]),
        config=config,
        configured_length=int(receipt["configured_length"]),
        seed=int(receipt["seed"]),
    )
    payload = _canonical_json_bytes(receipt_data)
    _atomic_publish_new(output_path, payload)
    published_raw_root = raw_root / filename.removesuffix(".json")
    try:
        receipt_raw_root.rename(published_raw_root)
    except FileExistsError:
        # Raw logs are diagnostics, not publication inputs. Keep this staged
        # run rather than replacing an older diagnostic directory.
        print(
            f"warning: kept raw diagnostics at {receipt_raw_root} because "
            f"{published_raw_root} already exists",
            file=sys.stderr,
            flush=True,
        )
    return {
        "filename": filename,
        "phase": receipt["phase"],
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "generator_reported_length": normalized["generator_reported_length"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--receipt",
        action="append",
        default=[],
        help="Generate only an exact required receipt filename; may be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    capture = _load_capture_module()
    source_manifest = verify_ruler_checkout(args.ruler_root, capture)
    runtime_manifest, resource_paths = verify_runtime(args.python, args.nltk_data)
    static_entries = verify_static_inputs(
        ruler_root=args.ruler_root,
        tokenizer_dir=args.tokenizer_dir,
        resource_paths=resource_paths,
        runtime_manifest=runtime_manifest,
        capture=capture,
    )
    static_entries = sorted(
        [
            *static_entries,
            _file_entry("ruler/source-manifest.json", _canonical_json_bytes(source_manifest)),
        ],
        key=lambda item: str(item["name"]),
    )
    tokenizer = IndependentTokenizer(python=args.python, tokenizer_dir=args.tokenizer_dir)
    required = list(capture.required_ruler_receipts())
    by_filename = {str(item["filename"]): item for item in required}
    if len(by_filename) != len(required):
        raise ValueError("required RULER receipt filenames must be unique")
    identities = {
        (
            item["phase"],
            item["category"],
            item["config"],
            item["configured_length"],
            item["seed"],
        )
        for item in required
    }
    if len(identities) != len(required):
        raise ValueError("required RULER receipt identities must be unique")
    if args.receipt:
        if len(set(args.receipt)) != len(args.receipt):
            raise ValueError("--receipt values must be unique")
        unknown = sorted(set(args.receipt) - set(by_filename))
        if unknown:
            raise ValueError(f"unknown required RULER receipts: {unknown}")
        selected = [by_filename[name] for name in args.receipt]
    else:
        selected = required
    results = []
    for index, receipt in enumerate(selected, start=1):
        output_path = args.output_dir / str(receipt["filename"])
        if output_path.exists():
            print(
                f"[{index}/{len(selected)}] verifying existing {receipt['filename']}",
                flush=True,
            )
            result = _load_existing_receipt_result(
                path=output_path,
                receipt=receipt,
                capture=capture,
                python=args.python,
                ruler_root=args.ruler_root,
                raw_root=args.raw_dir,
                tokenizer=tokenizer,
                static_entries=static_entries,
            )
        else:
            print(f"[{index}/{len(selected)}] generating {receipt['filename']}", flush=True)
            result = generate_receipt(
                receipt=receipt,
                capture=capture,
                python=args.python,
                ruler_root=args.ruler_root,
                tokenizer_dir=args.tokenizer_dir,
                nltk_data=args.nltk_data,
                raw_root=args.raw_dir,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                static_entries=static_entries,
                runtime_manifest=runtime_manifest,
                timeout_seconds=args.timeout_seconds,
            )
        results.append(result)
    launcher_entry = _entry_named(static_entries, "launcher/generate_static_q468_ruler_receipts.py")
    manifest = {
        "schema": "recurquant.experiment013.ruler-generation-manifest.v1",
        "launcher_revision": LAUNCHER_REVISION,
        "launcher_source_sha256": launcher_entry["sha256"],
        "ruler_revision": capture.resolver.RULER_REVISION,
        "source_manifest_sha256": _sha256_bytes(_canonical_json_bytes(source_manifest)),
        "runtime_manifest_sha256": _sha256_bytes(_canonical_json_bytes(runtime_manifest)),
        "selected_receipts": results,
    }
    _atomic_publish_same(
        args.output_dir / "generation-manifest.json", _canonical_json_bytes(manifest)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
