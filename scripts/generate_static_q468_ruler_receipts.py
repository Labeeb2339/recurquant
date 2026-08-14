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
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = REPOSITORY_ROOT / "scripts" / "capture_static_q468_identity_input.py"

LAUNCHER_REVISION: Final = "experiment-013-ruler-argv-launcher-v6"
GENERATION_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-generation-manifest.v2"
RUNTIME_MANIFEST_SCHEMA: Final = "recurquant.experiment013.ruler-runtime-manifest.v3"
RUNTIME_PYTHON_VERSION: Final = "3.11.15"
RUNTIME_PROBE_TIMEOUT_SECONDS: Final = 300
TOKENIZER_TIMEOUT_SECONDS: Final = 180
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
RUNTIME_IMPORTABLE_SUFFIXES: Final = frozenset({".dll", ".pyd", ".py", ".pyw", ".pth"})
EXPECTED_EXCLUDED_VIRTUALENV_STARTUP_FILES: Final = {
    "_virtualenv.pth": (
        18,
        "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    ),
    "_virtualenv.py": (
        5_246,
        "cfb3db86aaa53bb62b5ff764970bec2d71c9228590a0ebec57f6ec926cc0bf1a",
    ),
}
SEALED_STARTUP_POLICY: Final = {
    "dont_write_bytecode": 1,
    "no_site": 1,
    "package_path_mode": "staged-record-only-site-packages-v1",
    "pycache_mode": "verified-empty-prefix-no-write-v1",
    "site_loaded": False,
    "utf8_mode": 1,
    "virtualenv_hook_loaded": False,
}
SEALED_STARTUP_BOOTSTRAP: Final = r"""
import importlib.metadata as _rq_metadata
import pathlib as _rq_pathlib
import re as _rq_re
import sys as _rq_sys

_rq_package_root_raw = _rq_pathlib.Path(_rq_sys.argv[1])
_rq_pycache_root_raw = _rq_pathlib.Path(_rq_sys.argv[2])
if not _rq_package_root_raw.is_absolute() or not _rq_pycache_root_raw.is_absolute():
    raise RuntimeError("sealed RULER startup paths must be absolute")
_rq_package_root = _rq_package_root_raw.resolve()
_rq_pycache_root = _rq_pycache_root_raw.resolve()
_rq_reparse = lambda path: path.is_symlink() or bool(
    getattr(path.stat(), "st_file_attributes", 0) & 0x400
)
if (
    not _rq_package_root.is_dir()
    or not _rq_pycache_root.is_dir()
    or _rq_reparse(_rq_package_root)
    or _rq_reparse(_rq_pycache_root)
    or any(_rq_pycache_root.iterdir())
):
    raise RuntimeError("sealed RULER startup paths are missing, redirected, or non-empty")
_rq_flags = {
    "ignore_environment": _rq_sys.flags.ignore_environment,
    "isolated": _rq_sys.flags.isolated,
    "no_user_site": _rq_sys.flags.no_user_site,
}
if _rq_flags != {"ignore_environment": 1, "isolated": 1, "no_user_site": 1}:
    raise RuntimeError("sealed RULER isolation flags drifted")
if (
    _rq_sys.flags.no_site != 1
    or _rq_sys.flags.dont_write_bytecode != 1
    or _rq_sys.flags.utf8_mode != 1
    or _rq_sys.pycache_prefix is None
    or _rq_pathlib.Path(_rq_sys.pycache_prefix).resolve() != _rq_pycache_root
    or "site" in _rq_sys.modules
    or "_virtualenv" in _rq_sys.modules
):
    raise RuntimeError("sealed RULER startup policy drifted")
_rq_canonical = lambda name: _rq_re.sub(r"[-_.]+", "-", name).lower()
_rq_expected_packages = __PACKAGE_VERSIONS__
_rq_distributions = list(_rq_metadata.distributions(path=[str(_rq_package_root)]))
_rq_by_name = {}
for _rq_dist in _rq_distributions:
    _rq_name = _rq_canonical(_rq_dist.metadata["Name"])
    if _rq_name in _rq_by_name:
        raise RuntimeError("sealed RULER package root contains a duplicate distribution")
    _rq_by_name[_rq_name] = _rq_dist
if {
    name: dist.version for name, dist in _rq_by_name.items()
} != {_rq_canonical(name): version for name, version in _rq_expected_packages.items()}:
    raise RuntimeError("sealed RULER package root distribution set drifted")
_rq_recorded_paths = set()
for _rq_dist in _rq_distributions:
    _rq_files = list(_rq_dist.files or ())
    if not _rq_files:
        raise RuntimeError("sealed RULER distribution has no RECORD inventory")
    for _rq_item in _rq_files:
        _rq_path = _rq_pathlib.Path(_rq_dist.locate_file(_rq_item))
        if not _rq_path.is_file() or _rq_reparse(_rq_path):
            raise RuntimeError("sealed RULER RECORD path is missing or redirected")
        _rq_recorded_paths.add(_rq_path.resolve())
for _rq_path in _rq_package_root.rglob("*"):
    if _rq_path.is_symlink() or (
        _rq_path.exists() and bool(getattr(_rq_path.stat(), "st_file_attributes", 0) & 0x400)
    ):
        raise RuntimeError("sealed RULER package root contains a redirected path")
    if (
        _rq_path.is_file()
        and _rq_path.suffix.lower() in {".dll", ".pyd", ".py", ".pyw", ".pth"}
        and _rq_path.resolve() not in _rq_recorded_paths
    ):
        raise RuntimeError("sealed RULER package root contains unrecorded importable code")
_recurquant_startup_flags = _rq_flags
_recurquant_startup_policy = {
    "dont_write_bytecode": _rq_sys.flags.dont_write_bytecode,
    "no_site": _rq_sys.flags.no_site,
    "package_path_mode": "staged-record-only-site-packages-v1",
    "pycache_mode": "verified-empty-prefix-no-write-v1",
    "site_loaded": "site" in _rq_sys.modules,
    "utf8_mode": _rq_sys.flags.utf8_mode,
    "virtualenv_hook_loaded": "_virtualenv" in _rq_sys.modules,
}
_rq_sys.path.insert(0, str(_rq_package_root))
""".strip().replace("__PACKAGE_VERSIONS__", repr(RUNTIME_PACKAGES))
ISOLATED_SOURCE_BOOTSTRAP: Final = (
    SEALED_STARTUP_BOOTSTRAP
    + "\n"
    + r"""
import runpy as _rq_runpy

_rq_root = _rq_pathlib.Path(_rq_sys.argv[3]).resolve()
_rq_relative = _rq_pathlib.PurePosixPath(_rq_sys.argv[4])
_rq_script = (_rq_root / _rq_relative).resolve()
_rq_script.relative_to(_rq_root)
_rq_sys.path[:0] = [str(_rq_script.parent), str(_rq_root)]
_rq_sys.argv = [str(_rq_script), *_rq_sys.argv[5:]]
_rq_runpy.run_path(str(_rq_script), run_name="__main__")
""".strip()
)

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
    "nltk/punkt/english.pickle": (
        433_305,
        "dda37972ae88998a6fd3e3ec002697a6bd362b32d050fda7d7ca5276873092aa",
    ),
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


@dataclass(frozen=True, slots=True)
class VerifiedRulerCheckout:
    source_manifest: tuple[dict[str, object], ...]
    source_files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class VerifiedStaticInputs:
    entries: tuple[dict[str, object], ...]
    corpus_files: Mapping[str, bytes]
    sealed_runtime_files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class VerifiedRuntimePackageTree:
    entries: tuple[dict[str, object], ...]
    source_files: Mapping[str, Path]
    excluded_startup_files: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class VerifiedPythonRuntime:
    entries: tuple[dict[str, object], ...]
    source_files: Mapping[str, Path]
    source_launcher: dict[str, object]
    source_pyvenv_config: dict[str, object]


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


def _is_exact_one_flag_mapping(value: object, fields: frozenset[str]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(fields)
        and all(type(value[field]) is int and value[field] == 1 for field in fields)
    )


def _subprocess_env(**updates: str) -> dict[str, str]:
    """Return a child environment without caller-controlled Python injection."""

    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    env.update(updates)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _git_env() -> dict[str, str]:
    """Return a Git environment without caller or machine configuration."""

    inherited = {key.upper(): (key, value) for key, value in os.environ.items()}
    env = {
        inherited[name][0]: inherited[name][1]
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC")
        if name in inherited
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_AUTHOR_NAME": "RecurQuant Experiment 013",
            "GIT_AUTHOR_EMAIL": "experiment013@invalid",
            "GIT_COMMITTER_NAME": "RecurQuant Experiment 013",
            "GIT_COMMITTER_EMAIL": "experiment013@invalid",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return env


def _authenticated_git_executable(path: Path | None) -> Path:
    selected: str | os.PathLike[str]
    if path is None:
        discovered = shutil.which("git")
        if discovered is None:
            raise ValueError("Git executable is unavailable")
        selected = discovered
    else:
        selected = path
    try:
        resolved = Path(selected).resolve(strict=True)
    except OSError as error:
        raise ValueError("Git executable is unavailable") from error
    if resolved.name.casefold() == "git.exe" and resolved.parent.name.casefold() == "cmd":
        try:
            resolved = (resolved.parent.parent / "mingw64" / "bin" / "git.exe").resolve(strict=True)
        except OSError as error:
            raise ValueError(
                "Git-for-Windows cmd shim has no canonical mingw64 executable"
            ) from error
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        if _is_reparse_point(current):
            raise ValueError("Git executable traverses a link or reparse point")
    if not resolved.is_file() or _is_reparse_point(resolved) or resolved.stat().st_size <= 0:
        raise ValueError("Git executable must be a non-empty regular non-link file")
    return resolved


def _file_entry(name: str, data: bytes) -> dict[str, object]:
    if not data:
        raise ValueError(f"bound file {name!r} is empty")
    return {"name": name, "sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _tree_file_entry(name: str, data: bytes) -> dict[str, object]:
    return {"name": name, "sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _is_reparse_point(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path.stat(), "st_file_attributes", 0) & 0x400)


def _runtime_layout(python: Path) -> tuple[Path, Path, Path]:
    unresolved_python = Path(os.path.abspath(python))
    if not unresolved_python.is_file() or _is_reparse_point(unresolved_python):
        raise ValueError("RULER Python executable is missing or redirected")
    resolved_python = unresolved_python.resolve()
    if (
        resolved_python.name.casefold() != "python.exe"
        or resolved_python.parent.name.casefold() != "scripts"
    ):
        raise ValueError("RULER Python executable is not in a Windows virtual environment")
    runtime_root = resolved_python.parent.parent
    package_root = runtime_root / "Lib" / "site-packages"
    for path in (runtime_root, runtime_root / "Lib", package_root):
        if not path.is_dir() or _is_reparse_point(path):
            raise ValueError("RULER virtual-environment package path is missing or redirected")
    return resolved_python, runtime_root.resolve(), package_root.resolve()


def verify_python_runtime_source(python: Path) -> VerifiedPythonRuntime:
    source_python, venv_root, _package_root = _runtime_layout(python)
    pyvenv_path = venv_root / "pyvenv.cfg"
    if not pyvenv_path.is_file() or _is_reparse_point(pyvenv_path):
        raise ValueError("RULER Python virtual environment lacks a bound pyvenv.cfg")
    pyvenv_data = pyvenv_path.read_bytes()
    try:
        config_lines = pyvenv_data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("RULER pyvenv.cfg must be UTF-8") from error
    config: dict[str, str] = {}
    for line in config_lines:
        if not line.strip():
            continue
        if line.count("=") != 1:
            raise ValueError("RULER pyvenv.cfg is malformed")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in config:
            raise ValueError("RULER pyvenv.cfg contains a duplicate or empty key")
        config[key] = value
    if config.get("version_info") != "3.11" or config.get("include-system-site-packages") != (
        "false"
    ):
        raise ValueError("RULER pyvenv.cfg version or site-package policy drifted")
    raw_home = config.get("home")
    if not raw_home:
        raise ValueError("RULER pyvenv.cfg omitted its base runtime")
    base_root = Path(raw_home)
    if not base_root.is_absolute() or not base_root.is_dir() or _is_reparse_point(base_root):
        raise ValueError("RULER base Python runtime is missing or redirected")
    base_root = base_root.resolve()

    source_files: dict[str, Path] = {}
    for root_name in ("Lib", "DLLs"):
        root = base_root / root_name
        if not root.is_dir() or _is_reparse_point(root):
            raise ValueError(f"RULER base Python runtime omitted {root_name}")
        for path in root.rglob("*"):
            if _is_reparse_point(path):
                raise ValueError("RULER base Python runtime contains a redirected path")
            if not path.is_file():
                continue
            relative = path.relative_to(base_root)
            if "site-packages" in {part.casefold() for part in relative.parts}:
                continue
            if "__pycache__" in relative.parts or path.suffix.casefold() == ".pyc":
                continue
            source_files[relative.as_posix()] = path.resolve()
    root_patterns = (
        re.compile(r"^python.*\.(?:dll|exe|zip)$", flags=re.IGNORECASE),
        re.compile(r"^vcruntime.*\.dll$", flags=re.IGNORECASE),
    )
    for path in base_root.iterdir():
        if _is_reparse_point(path):
            raise ValueError("RULER base Python root contains a redirected path")
        if path.is_file() and any(pattern.fullmatch(path.name) for pattern in root_patterns):
            source_files[path.name] = path.resolve()
    required_names = {"python.exe", "python3.dll", "python311.dll"}
    if not required_names <= set(source_files):
        raise ValueError("RULER base Python runtime omitted an executable or adjacent DLL")
    entries = tuple(
        _tree_file_entry(name, source_files[name].read_bytes()) for name in sorted(source_files)
    )
    return VerifiedPythonRuntime(
        entries=entries,
        source_files=dict(source_files),
        source_launcher=_tree_file_entry("source/python.exe", source_python.read_bytes()),
        source_pyvenv_config=_tree_file_entry("source/pyvenv.cfg", pyvenv_data),
    )


def verify_staged_python_runtime(root: Path, *, expected: Sequence[Mapping[str, object]]) -> None:
    root = root.resolve()
    observed: dict[str, Path] = {}
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("staged RULER Python runtime contains a redirected path")
        if path.is_file():
            observed[path.relative_to(root).as_posix()] = path
    expected_by_name = {str(item["name"]): item for item in expected}
    if set(observed) != set(expected_by_name):
        raise ValueError("staged RULER Python runtime inventory drifted")
    for name, path in observed.items():
        if _tree_file_entry(name, path.read_bytes()) != expected_by_name[name]:
            raise ValueError(f"staged RULER Python runtime bytes drifted: {name}")


def stage_verified_python_runtime(root: Path, *, verified: VerifiedPythonRuntime) -> Path:
    if root.exists():
        raise FileExistsError(f"refusing to replace staged RULER Python runtime: {root}")
    root.mkdir(parents=True)
    for name, source in verified.source_files.items():
        destination = root / Path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    verify_staged_python_runtime(root, expected=verified.entries)
    python = root / "python.exe"
    if not python.is_file():
        raise ValueError("staged RULER Python runtime omitted python.exe")
    return python


def verify_runtime_package_source(python: Path) -> VerifiedRuntimePackageTree:
    """Freeze a live venv into a RECORD-only, non-executable source inventory."""

    _python, runtime_root, package_root = _runtime_layout(python)
    distributions = list(importlib.metadata.distributions(path=[str(package_root)]))
    by_name: dict[str, importlib.metadata.Distribution] = {}
    for distribution in distributions:
        canonical_name = _canonical_distribution_name(distribution.metadata["Name"])
        if canonical_name in by_name:
            raise ValueError("RULER runtime contains a duplicate installed distribution")
        by_name[canonical_name] = distribution
    expected_versions = {
        _canonical_distribution_name(name): version for name, version in RUNTIME_PACKAGES.items()
    }
    if {name: distribution.version for name, distribution in by_name.items()} != expected_versions:
        raise ValueError("RULER runtime installed-distribution inventory drifted")

    entries_by_name: dict[str, dict[str, object]] = {}
    source_files: dict[str, Path] = {}
    recorded_paths: set[Path] = set()
    for canonical_name in sorted(by_name):
        distribution = by_name[canonical_name]
        files = list(distribution.files or ())
        records = [
            item for item in files if str(item).replace("\\", "/").endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise ValueError(f"RULER package {canonical_name} must have exactly one RECORD")
        for item in files:
            unresolved = Path(distribution.locate_file(item))
            if not unresolved.is_file() or _is_reparse_point(unresolved):
                raise ValueError(f"RULER package {canonical_name} has a missing or redirected file")
            resolved = unresolved.resolve()
            try:
                relative = resolved.relative_to(runtime_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"RULER package {canonical_name} RECORD escapes the virtual environment"
                ) from error
            data = resolved.read_bytes()
            entry = _tree_file_entry(relative, data)
            previous = entries_by_name.get(relative)
            if previous is not None and previous != entry:
                raise ValueError("RULER distributions disagree about a shared installed file")
            entries_by_name[relative] = entry
            source_files[relative] = resolved
            recorded_paths.add(resolved)

    excluded: dict[str, dict[str, object]] = {}
    for path in package_root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("RULER live package tree contains a redirected path")
        if not path.is_file() or path.suffix.lower() not in RUNTIME_IMPORTABLE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in recorded_paths:
            continue
        relative = path.relative_to(package_root).as_posix()
        try:
            expected_size, expected_sha256 = EXPECTED_EXCLUDED_VIRTUALENV_STARTUP_FILES[relative]
        except KeyError as error:
            raise ValueError(
                f"RULER live package tree contains unrecorded importable code: {relative}"
            ) from error
        data = path.read_bytes()
        entry = _tree_file_entry(relative, data)
        if entry["size_bytes"] != expected_size or entry["sha256"] != expected_sha256:
            raise ValueError(f"RULER excluded startup file drifted: {relative}")
        excluded[relative] = entry
    if set(excluded) != set(EXPECTED_EXCLUDED_VIRTUALENV_STARTUP_FILES):
        raise ValueError("RULER excluded virtualenv startup-file inventory drifted")
    entries = tuple(entries_by_name[name] for name in sorted(entries_by_name))
    return VerifiedRuntimePackageTree(
        entries=entries,
        source_files=dict(source_files),
        excluded_startup_files=tuple(excluded[name] for name in sorted(excluded)),
    )


def verify_staged_runtime_package_tree(
    runtime_root: Path, *, expected: Sequence[Mapping[str, object]]
) -> None:
    runtime_root = runtime_root.resolve()
    observed: dict[str, Path] = {}
    for path in runtime_root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("staged RULER runtime package tree contains a redirected path")
        if path.is_file():
            observed[path.relative_to(runtime_root).as_posix()] = path
    expected_by_name = {str(entry["name"]): entry for entry in expected}
    if set(observed) != set(expected_by_name):
        raise ValueError("staged RULER runtime package inventory drifted")
    for name, path in observed.items():
        if _tree_file_entry(name, path.read_bytes()) != expected_by_name[name]:
            raise ValueError(f"staged RULER runtime package bytes drifted: {name}")


def stage_verified_runtime_package_tree(
    runtime_root: Path, *, verified: VerifiedRuntimePackageTree
) -> Path:
    if runtime_root.exists():
        raise FileExistsError(f"refusing to replace staged RULER runtime: {runtime_root}")
    runtime_root.mkdir(parents=True)
    for name, source in verified.source_files.items():
        destination = runtime_root / Path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    verify_staged_runtime_package_tree(runtime_root, expected=verified.entries)
    package_root = runtime_root / "Lib" / "site-packages"
    if not package_root.is_dir():
        raise ValueError("staged RULER runtime omitted its site-packages directory")
    return package_root


def _verify_empty_pycache_prefix(path: Path) -> None:
    if not path.is_dir() or _is_reparse_point(path) or any(path.iterdir()):
        raise ValueError("sealed RULER pycache prefix is missing, redirected, or non-empty")


def _sealed_python_argv(
    *,
    python: Path,
    package_root: Path,
    pycache_prefix: Path,
    code: str,
    arguments: Sequence[str] = (),
) -> list[str]:
    return [
        str(python.resolve()),
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix.resolve()}",
        "-X",
        "utf8",
        "-c",
        code,
        str(package_root.resolve()),
        str(pycache_prefix.resolve()),
        *arguments,
    ]


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


def verify_ruler_checkout(
    ruler_root: Path,
    capture: Any,
    *,
    git_executable_path: Path | None = None,
) -> VerifiedRulerCheckout:
    ruler_root = ruler_root.resolve()
    git_executable = _authenticated_git_executable(git_executable_path)
    result = subprocess.run(
        [str(git_executable), "-C", str(ruler_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    head = result.stdout.strip()
    if head != capture.resolver.RULER_REVISION:
        raise ValueError("RULER checkout HEAD differs from the frozen revision")
    files: dict[str, bytes] = {}
    for relative, expected_blob in capture.RULER_GENERATOR_GIT_BLOBS.items():
        data = subprocess.run(
            [str(git_executable), "-C", str(ruler_root), "cat-file", "blob", expected_blob],
            check=True,
            capture_output=True,
            env=_git_env(),
        ).stdout
        if _git_blob_sha1(data) != expected_blob:
            raise RuntimeError(f"Git returned corrupt blob bytes for {relative}")
        files[relative] = data
    _verify_task_specs_against_source(
        synthetic_yaml=files["scripts/synthetic.yaml"],
        constants_py=files["scripts/data/synthetic/constants.py"],
    )
    return VerifiedRulerCheckout(
        source_manifest=tuple(capture._ruler_generator_manifest(files)),
        source_files=dict(files),
    )


def verify_runtime(
    python: Path,
    nltk_data: Path,
    *,
    package_root: Path,
    package_tree_manifest: Sequence[Mapping[str, object]],
    excluded_startup_files: Sequence[Mapping[str, object]],
    python_runtime_root: Path,
    python_runtime_manifest: Sequence[Mapping[str, object]],
    source_python: Mapping[str, object],
    source_pyvenv_config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, Path]]:
    code = """
import importlib.metadata as metadata
import importlib.util
import hashlib
import json
import pathlib
import platform
import re
import sys
names = __PACKAGE_NAMES__
forbidden = __FORBIDDEN_MODULES__
canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()
def distribution_inventory(name):
    dist = metadata.distribution(name)
    items = list(dist.files or ())
    records = [
        item
        for item in items
        if str(item).replace('\\\\', '/').endswith('.dist-info/RECORD')
    ]
    if len(records) != 1:
        raise RuntimeError(f'{name} must have exactly one installed RECORD inventory')
    record_path = pathlib.Path(dist.locate_file(records[0]))
    if not record_path.is_file():
        raise RuntimeError(f'{name} installed RECORD is unavailable')
    record_bytes = record_path.read_bytes()
    if not record_bytes:
        raise RuntimeError(f'{name} installed RECORD is empty')
    files = []
    seen = set()
    for item in items:
        relative = str(item).replace('\\\\', '/')
        if not relative or relative in seen:
            raise RuntimeError(f'{name} has a duplicate or empty installed path')
        seen.add(relative)
        path = pathlib.Path(dist.locate_file(item))
        if not path.is_file():
            raise RuntimeError(f'{name} installed file is unavailable: {relative}')
        data = path.read_bytes()
        files.append({
            'path': relative,
            'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data),
        })
    if not files:
        raise RuntimeError(f'{name} has no installed files')
    files.sort(key=lambda item: item['path'])
    return {
        'canonical_name': canonical(dist.metadata['Name']),
        'version': dist.version,
        'record_sha256': hashlib.sha256(record_bytes).hexdigest(),
        'record_size_bytes': len(record_bytes),
        'files': files,
    }
pre_inventory = {name: distribution_inventory(name) for name in names}
pre_installed = {
    canonical(dist.metadata['Name']): dist.version for dist in metadata.distributions()
}
import nltk
import wonderwords
root = pathlib.Path(wonderwords.__file__).resolve().parent
post_inventory = {name: distribution_inventory(name) for name in names}
post_installed = {
    canonical(dist.metadata['Name']): dist.version for dist in metadata.distributions()
}
if post_inventory != pre_inventory or post_installed != pre_installed:
    raise RuntimeError('RULER package code changed while imports were active')
payload = {
    'python': sys.version.split()[0],
    'implementation': sys.implementation.name,
    'cache_tag': sys.implementation.cache_tag,
    'executable': str(pathlib.Path(sys.executable).resolve()),
    'platform': platform.platform(),
    'machine': platform.machine(),
    'flags': {
        **_recurquant_startup_flags,
    },
    'startup_policy': _recurquant_startup_policy,
    'packages': {name: metadata.version(name) for name in names},
    'installed_distributions': pre_installed,
    'distribution_file_inventory': pre_inventory,
    'forbidden_modules': {
        name: importlib.util.find_spec(name) is not None for name in forbidden
    },
    'resources': {
        'nltk/punkt/english.pickle': next(
            str(pathlib.Path(data_root) / 'tokenizers' / 'punkt' / 'english.pickle')
            for data_root in nltk.data.path
            if (pathlib.Path(data_root) / 'tokenizers' / 'punkt' / 'english.pickle').is_file()
        ),
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
    runtime_root = package_root.resolve().parents[1]
    verify_staged_runtime_package_tree(runtime_root, expected=package_tree_manifest)
    verify_staged_python_runtime(python_runtime_root, expected=python_runtime_manifest)
    with tempfile.TemporaryDirectory(prefix="recurquant-exp013-ruler-pycache-") as temporary:
        pycache_prefix = Path(temporary)
        _verify_empty_pycache_prefix(pycache_prefix)
        try:
            result = subprocess.run(
                _sealed_python_argv(
                    python=python,
                    package_root=package_root,
                    pycache_prefix=pycache_prefix,
                    code=SEALED_STARTUP_BOOTSTRAP + "\n" + code,
                ),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                env=env,
                timeout=RUNTIME_PROBE_TIMEOUT_SECONDS,
            )
        finally:
            _verify_empty_pycache_prefix(pycache_prefix)
            verify_staged_runtime_package_tree(runtime_root, expected=package_tree_manifest)
            verify_staged_python_runtime(python_runtime_root, expected=python_runtime_manifest)
        if result.returncode != 0:
            diagnostic = result.stderr[-4_000:].strip()
            raise RuntimeError(
                "sealed RULER runtime probe failed"
                + (f": {diagnostic}" if diagnostic else " without stderr")
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
    file_inventory = payload.get("distribution_file_inventory")
    if not isinstance(file_inventory, Mapping) or set(file_inventory) != set(RUNTIME_PACKAGES):
        raise ValueError("RULER distribution-file inventory drifted")
    for package_name, version in RUNTIME_PACKAGES.items():
        raw_distribution = file_inventory[package_name]
        if not isinstance(raw_distribution, Mapping) or set(raw_distribution) != {
            "canonical_name",
            "version",
            "record_sha256",
            "record_size_bytes",
            "files",
        }:
            raise ValueError(f"RULER package-code inventory is malformed for {package_name}")
        if (
            raw_distribution["canonical_name"] != _canonical_distribution_name(package_name)
            or raw_distribution["version"] != version
        ):
            raise ValueError(f"RULER package-code identity drifted for {package_name}")
        record_sha256 = raw_distribution["record_sha256"]
        record_size = raw_distribution["record_size_bytes"]
        if (
            not isinstance(record_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_sha256) is None
            or isinstance(record_size, bool)
            or not isinstance(record_size, int)
            or record_size < 1
        ):
            raise ValueError(f"RULER package RECORD identity drifted for {package_name}")
        files_value = raw_distribution["files"]
        if isinstance(files_value, (str, bytes)) or not isinstance(files_value, Sequence):
            raise ValueError(f"RULER package file list is malformed for {package_name}")
        normalized_files: list[dict[str, object]] = []
        for item in files_value:
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size_bytes"}:
                raise ValueError(f"RULER package file entry is malformed for {package_name}")
            path_value = item["path"]
            size_value = item["size_bytes"]
            digest_value = item["sha256"]
            if (
                not isinstance(path_value, str)
                or not path_value
                or "\\" in path_value
                or "\0" in path_value
                or "\n" in path_value
                or "\r" in path_value
                or not isinstance(digest_value, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
                or isinstance(size_value, bool)
                or not isinstance(size_value, int)
                or size_value < 0
            ):
                raise ValueError(f"RULER package file identity drifted for {package_name}")
            normalized_files.append(dict(item))
        if (
            not normalized_files
            or normalized_files != sorted(normalized_files, key=lambda item: str(item["path"]))
            or len({str(item["path"]) for item in normalized_files}) != len(normalized_files)
        ):
            raise ValueError(f"RULER package file ordering drifted for {package_name}")
    expected_absence = {name: False for name in FORBIDDEN_RUNTIME_MODULES}
    if payload.get("forbidden_modules") != expected_absence:
        raise ValueError("RULER runtime contains a forbidden model framework")
    if payload.get("implementation") != "cpython":
        raise ValueError("RULER Python implementation drifted")
    flags = payload.get("flags")
    if not _is_exact_one_flag_mapping(
        flags, frozenset({"ignore_environment", "isolated", "no_user_site"})
    ):
        raise ValueError("RULER runtime probe was not isolated")
    startup_policy = payload.get("startup_policy")
    if (
        startup_policy != SEALED_STARTUP_POLICY
        or not isinstance(startup_policy, Mapping)
        or any(
            type(startup_policy[field]) is not int or startup_policy[field] != 1
            for field in ("dont_write_bytecode", "no_site", "utf8_mode")
        )
        or any(
            type(startup_policy[field]) is not bool or startup_policy[field] is not False
            for field in ("site_loaded", "virtualenv_hook_loaded")
        )
    ):
        raise ValueError("RULER runtime probe was not sealed")
    executable = payload.get("executable")
    if not isinstance(executable, str) or not Path(executable).samefile(python):
        raise ValueError("RULER runtime probe used a different Python executable")
    for field in ("cache_tag", "platform", "machine"):
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
    expected_excluded = [
        {"name": name, "sha256": digest, "size_bytes": size}
        for name, (size, digest) in sorted(EXPECTED_EXCLUDED_VIRTUALENV_STARTUP_FILES.items())
    ]
    if [dict(item) for item in excluded_startup_files] != expected_excluded:
        raise ValueError("RULER excluded startup-file binding drifted")
    executable_data = python.read_bytes()
    runtime_manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "python": payload["python"],
        "implementation": payload["implementation"],
        "cache_tag": payload["cache_tag"],
        "platform": payload["platform"],
        "machine": payload["machine"],
        "flags": payload["flags"],
        "startup_policy": payload["startup_policy"],
        "excluded_startup_files": [dict(item) for item in excluded_startup_files],
        "source_python": dict(source_python),
        "source_pyvenv_config": dict(source_pyvenv_config),
        "python_runtime_files": [dict(item) for item in python_runtime_manifest],
        "executable": _file_entry("python.exe", executable_data),
        "packages": payload["packages"],
        "installed_distributions": payload["installed_distributions"],
        "distribution_file_inventory": payload["distribution_file_inventory"],
        "forbidden_modules": payload["forbidden_modules"],
    }
    return runtime_manifest, paths


def verify_static_runtime_input_source(
    *, tokenizer_dir: Path, nltk_data: Path, capture: Any
) -> dict[str, bytes]:
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
    result: dict[str, bytes] = {}
    for name, (size, sha256) in sorted(EXPECTED_TOKENIZER_ASSETS.items()):
        result[f"tokenizer/{capture.resolver.PRIMARY_MODEL_REVISION}/{name}"] = _verified_file(
            files[name],
            size=size,
            sha256=sha256,
            name=f"tokenizer asset {name}",
        )
    for name in ("nltk/punkt/english.pickle", "nltk/punkt/PY3/english.pickle"):
        relative = name.removeprefix("nltk/punkt/")
        nltk_path = nltk_data.resolve() / "tokenizers" / "punkt" / Path(relative)
        size, sha256 = EXPECTED_PACKAGE_RESOURCES[name]
        result[f"nltk_data/tokenizers/punkt/{relative}"] = _verified_file(
            nltk_path,
            size=size,
            sha256=sha256,
            name=f"RULER NLTK punkt resource {relative}",
        )
    return result


def verify_static_inputs(
    *,
    ruler_root: Path,
    tokenizer_dir: Path,
    resource_paths: Mapping[str, Path],
    runtime_manifest: Mapping[str, object],
    capture: Any,
) -> VerifiedStaticInputs:
    entries: list[dict[str, object]] = []
    corpus_files: dict[str, bytes] = {}
    sealed_runtime_files: dict[str, bytes] = {}
    corpus_root = ruler_root / "scripts" / "data" / "synthetic" / "json"
    for name, (size, sha256) in sorted(EXPECTED_CORPORA.items()):
        data = _verified_file(
            corpus_root / name,
            size=size,
            sha256=sha256,
            name=f"RULER corpus {name}",
        )
        entries.append(_file_entry(f"corpora/{name}", data))
        corpus_files[name] = data
    for name, (size, sha256) in sorted(EXPECTED_PACKAGE_RESOURCES.items()):
        data = _verified_file(
            resource_paths[name],
            size=size,
            sha256=sha256,
            name=f"RULER package resource {name}",
        )
        entries.append(_file_entry(f"packages/{name}", data))
        if name.startswith("nltk/punkt/"):
            relative = name.removeprefix("nltk/punkt/")
            sealed_runtime_files[f"nltk_data/tokenizers/punkt/{relative}"] = data

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
        sealed_runtime_files[f"tokenizer/{capture.resolver.PRIMARY_MODEL_REVISION}/{name}"] = data

    runtime_bytes = _canonical_json_bytes(runtime_manifest)
    entries.append(_file_entry("runtime/package-manifest.json", runtime_bytes))
    entries.append(_file_entry("runtime/requirements.txt", _requirements_bytes()))
    entries.append(_launcher_source_entry())

    return VerifiedStaticInputs(
        entries=tuple(sorted(entries, key=lambda item: str(item["name"]))),
        corpus_files=corpus_files,
        sealed_runtime_files=dict(sealed_runtime_files),
    )


def stage_verified_ruler_source(
    root: Path,
    *,
    checkout: VerifiedRulerCheckout,
    static_inputs: VerifiedStaticInputs,
) -> dict[str, dict[str, object]]:
    """Materialize only authenticated source/corpus bytes in a new isolated tree."""

    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("isolated RULER source root must start empty")
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = dict(checkout.source_files)
    for name, data in static_inputs.corpus_files.items():
        relative = f"scripts/data/synthetic/json/{name}"
        if relative in files:
            raise ValueError("isolated RULER source inventory contains a duplicate path")
        files[relative] = data
    manifest: dict[str, dict[str, object]] = {}
    for relative, data in sorted(files.items()):
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        manifest[relative] = {
            "sha256": _sha256_bytes(data),
            "size_bytes": len(data),
        }
    verify_staged_ruler_source(root, expected=manifest)
    return manifest


def verify_staged_ruler_source(
    root: Path,
    *,
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    """Reject added, removed, aliased, or modified staged import inputs."""

    root = root.resolve()
    observed: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("isolated RULER source tree may not contain symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            observed[relative] = path
    observed_names = set(observed)
    expected_names = set(expected)
    if observed_names != expected_names:
        raise ValueError(
            "isolated RULER source inventory drifted; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for relative, path in observed.items():
        data = path.read_bytes()
        identity = expected[relative]
        if identity.get("sha256") != _sha256_bytes(data) or identity.get("size_bytes") != len(data):
            raise ValueError(f"isolated RULER source bytes drifted: {relative}")


def stage_verified_runtime_inputs(
    root: Path, *, files: Mapping[str, bytes]
) -> tuple[dict[str, dict[str, object]], Path, Path]:
    if root.exists():
        raise FileExistsError(f"refusing to replace staged RULER runtime inputs: {root}")
    root.mkdir(parents=True)
    manifest: dict[str, dict[str, object]] = {}
    for relative, data in sorted(files.items()):
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        manifest[relative] = _tree_file_entry(relative, data)
    verify_staged_runtime_inputs(root, expected=manifest)
    tokenizer_parent = root / "tokenizer"
    tokenizer_directories = (
        [path for path in tokenizer_parent.iterdir() if path.is_dir()]
        if tokenizer_parent.is_dir()
        else []
    )
    nltk_data = root / "nltk_data"
    if len(tokenizer_directories) != 1 or not nltk_data.is_dir():
        raise ValueError("staged RULER tokenizer or NLTK data is missing")
    return manifest, tokenizer_directories[0], nltk_data


def verify_staged_runtime_inputs(
    root: Path, *, expected: Mapping[str, Mapping[str, object]]
) -> None:
    root = root.resolve()
    observed: dict[str, Path] = {}
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("staged RULER runtime inputs contain a redirected path")
        if path.is_file():
            observed[path.relative_to(root).as_posix()] = path
    if set(observed) != set(expected):
        raise ValueError("staged RULER runtime-input inventory drifted")
    for name, path in observed.items():
        if _tree_file_entry(name, path.read_bytes()) != expected[name]:
            raise ValueError(f"staged RULER runtime-input bytes drifted: {name}")


class IndependentTokenizer:
    """Recompute token counts inside the verified tokenizer-only interpreter."""

    def __init__(
        self,
        *,
        python: Path,
        tokenizer_dir: Path,
        package_root: Path,
        package_tree_manifest: Sequence[Mapping[str, object]],
        runtime_input_root: Path,
        runtime_input_manifest: Mapping[str, Mapping[str, object]],
        python_runtime_root: Path,
        python_runtime_manifest: Sequence[Mapping[str, object]],
    ) -> None:
        self._python = python.resolve()
        self._tokenizer_dir = tokenizer_dir.resolve()
        self._package_root = package_root.resolve()
        self._runtime_root = self._package_root.parents[1]
        self._package_tree_manifest = tuple(dict(item) for item in package_tree_manifest)
        self._runtime_input_root = runtime_input_root.resolve()
        self._runtime_input_manifest = {
            str(name): dict(item) for name, item in runtime_input_manifest.items()
        }
        self._python_runtime_root = python_runtime_root.resolve()
        self._python_runtime_manifest = tuple(dict(item) for item in python_runtime_manifest)

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
        verify_staged_runtime_package_tree(self._runtime_root, expected=self._package_tree_manifest)
        verify_staged_runtime_inputs(
            self._runtime_input_root, expected=self._runtime_input_manifest
        )
        verify_staged_python_runtime(
            self._python_runtime_root, expected=self._python_runtime_manifest
        )
        with tempfile.TemporaryDirectory(
            prefix="recurquant-exp013-tokenizer-pycache-"
        ) as temporary:
            pycache_prefix = Path(temporary)
            _verify_empty_pycache_prefix(pycache_prefix)
            try:
                result = subprocess.run(
                    _sealed_python_argv(
                        python=self._python,
                        package_root=self._package_root,
                        pycache_prefix=pycache_prefix,
                        code=SEALED_STARTUP_BOOTSTRAP + "\n" + code,
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    input=json.dumps(request, ensure_ascii=False, allow_nan=False),
                    env=_subprocess_env(
                        HF_HUB_OFFLINE="1",
                        TOKENIZERS_PARALLELISM="false",
                        TRANSFORMERS_OFFLINE="1",
                    ),
                    timeout=TOKENIZER_TIMEOUT_SECONDS,
                )
            finally:
                _verify_empty_pycache_prefix(pycache_prefix)
                verify_staged_runtime_package_tree(
                    self._runtime_root, expected=self._package_tree_manifest
                )
                verify_staged_runtime_inputs(
                    self._runtime_input_root, expected=self._runtime_input_manifest
                )
                verify_staged_python_runtime(
                    self._python_runtime_root, expected=self._python_runtime_manifest
                )
            if result.returncode != 0:
                diagnostic = result.stderr[-4_000:].strip()
                raise RuntimeError(
                    "sealed independent tokenizer failed"
                    + (f": {diagnostic}" if diagnostic else " without stderr")
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
    package_root: Path,
    staged_root: Path,
    raw_root: Path,
    receipt: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    config = str(receipt["config"])
    try:
        spec = TASK_SPECS[config]
    except KeyError as error:
        raise ValueError(f"no frozen launcher specification for RULER config {config}") from error
    data_root = staged_root / "scripts" / "data"
    pycache_prefix = raw_root / ".sealed-pycache"
    script_relative = f"synthetic/{spec.script}"
    actual = _sealed_python_argv(
        python=python,
        package_root=package_root,
        pycache_prefix=pycache_prefix,
        code=ISOLATED_SOURCE_BOOTSTRAP,
        arguments=(
            str(data_root.resolve()),
            script_relative,
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
        ),
    )
    for name, value in spec.arguments:
        actual.extend((f"--{name}", value))
    actual.extend(("--template", spec.template))
    portable = list(actual)
    portable[0] = "<RULER_PYTHON>"
    portable[5] = "pycache_prefix=<EMPTY_PYCACHE_PREFIX>"
    portable[10] = "<RULER_SITE_PACKAGES>"
    portable[11] = "<EMPTY_PYCACHE_PREFIX>"
    portable[12] = "<STAGED_RULER_DATA_ROOT>"
    portable[13] = script_relative
    portable[15] = "<RAW_RECEIPT_DIR>"
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
    package_root: Path,
    staged_root: Path,
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
    diagnostic_root = raw_root / path.name.removesuffix(".json")
    command_path = diagnostic_root / "command-manifest.json"
    raw_path = diagnostic_root / config / "validation.jsonl"
    if not command_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(f"existing RULER receipt lacks raw verification inputs: {path}")
    raw_data = raw_path.read_bytes()
    raw_lines = raw_data.splitlines()
    if len(raw_lines) != 1 or not raw_lines[0]:
        raise ValueError("existing RULER raw validation must contain exactly one row")
    raw_row = _strict_json(raw_lines[0], context=f"existing raw RULER row {path.name}")
    normalized = _normalize_output_row(
        raw_row,
        config=config,
        configured_length=int(receipt["configured_length"]),
        tokenizer=tokenizer,
    )
    for field in ("input", "answer_prefix", "outputs", "generator_reported_length"):
        if normalized[field] != value[field]:
            raise ValueError(f"existing RULER receipt differs from raw generator row: {field}")
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
        package_root=package_root,
        staged_root=staged_root,
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
    if command_path.read_bytes() != command_bytes:
        raise ValueError("existing RULER diagnostic command manifest drifted")
    expected_command = _file_entry("generator/command-manifest.json", command_bytes)
    expected_raw = _file_entry("generator/raw-validation.jsonl", raw_data)
    if auxiliary_by_name.get("generator/command-manifest.json") != expected_command:
        raise ValueError("existing RULER receipt has a stale launcher binding")
    if auxiliary_by_name.get("generator/raw-validation.jsonl") != expected_raw:
        raise ValueError("existing RULER receipt has a stale raw-row binding")
    expected_names = {str(entry["name"]) for entry in static_entries} | {
        "generator/command-manifest.json",
        "generator/raw-validation.jsonl",
    }
    if set(auxiliary_by_name) != expected_names:
        raise ValueError("existing RULER receipt auxiliary inventory drifted")
    return {
        "category": receipt["category"],
        "command_manifest": _strict_json(command_bytes, context="command manifest"),
        "command_manifest_file": expected_command,
        "config": config,
        "configured_length": receipt["configured_length"],
        "filename": path.name,
        "generator_reported_length": normalized["generator_reported_length"],
        "phase": receipt["phase"],
        "raw_validation_base64": base64.b64encode(raw_data).decode("ascii"),
        "raw_validation_file": expected_raw,
        "seed": receipt["seed"],
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _verify_owned_orphan_tree(
    path: Path,
    *,
    raw_root: Path,
    config: str,
    require_complete: bool,
) -> None:
    raw_root = raw_root.resolve()
    if path.parent.resolve() != raw_root or not path.is_dir() or _is_reparse_point(path):
        raise ValueError("RULER orphan candidate is outside the requested raw root")
    allowed_directories = {"", config, ".sealed-pycache"}
    allowed_files = {
        "command-manifest.json",
        "runtime-manifest.json",
        "stdout.log",
        "stderr.log",
        f"{config}/validation.jsonl",
    }
    observed_files: set[str] = set()
    for item in path.rglob("*"):
        if _is_reparse_point(item):
            raise ValueError("RULER orphan candidate contains a redirected path")
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            if relative not in allowed_directories:
                raise ValueError("RULER orphan candidate contains an unexpected directory")
        elif item.is_file():
            if relative not in allowed_files:
                raise ValueError("RULER orphan candidate contains an unexpected file")
            observed_files.add(relative)
        else:
            raise ValueError("RULER orphan candidate contains an unsupported filesystem object")
    if require_complete and observed_files != allowed_files:
        raise ValueError("published RULER diagnostic orphan is incomplete")


def recover_owned_receipt_orphans(
    *,
    filename: str,
    config: str,
    raw_root: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    """Remove only exact generator-owned leftovers when no receipt was published."""

    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("RULER receipt filename is not a canonical basename")
    raw_root = Path(os.path.abspath(raw_root))
    output_dir = Path(os.path.abspath(output_dir))
    raw_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(raw_root) or _is_reparse_point(output_dir):
        raise ValueError("RULER raw or output root is redirected")
    output_path = output_dir / filename
    if output_path.parent.resolve() != output_dir.resolve():
        raise ValueError("RULER receipt output escapes the requested output root")
    if output_path.exists():
        return ()
    receipt_key = _sha256_bytes(filename.encode("utf-8"))[:12]
    staging_pattern = re.compile(rf"^\.rq-{re.escape(receipt_key)}\.[A-Za-z0-9_-]+\.staging$")
    candidates = sorted(
        (path for path in raw_root.iterdir() if staging_pattern.fullmatch(path.name) is not None),
        key=lambda path: path.name,
    )
    published = raw_root / filename.removesuffix(".json")
    if published.exists():
        candidates.append(published)
    recovered: list[str] = []
    for candidate in candidates:
        _verify_owned_orphan_tree(
            candidate,
            raw_root=raw_root,
            config=config,
            require_complete=candidate == published,
        )
    for candidate in candidates:
        shutil.rmtree(candidate)
        recovered.append(candidate.name)
    return tuple(recovered)


def generate_receipt(
    *,
    receipt: Mapping[str, object],
    capture: Any,
    python: Path,
    package_root: Path,
    package_tree_manifest: Sequence[Mapping[str, object]],
    python_runtime_root: Path,
    python_runtime_manifest: Sequence[Mapping[str, object]],
    runtime_input_root: Path,
    runtime_input_manifest: Mapping[str, Mapping[str, object]],
    staged_root: Path,
    staged_manifest: Mapping[str, Mapping[str, object]],
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
    recovered = recover_owned_receipt_orphans(
        filename=filename,
        config=str(receipt["config"]),
        raw_root=raw_root,
        output_dir=output_dir,
    )
    if recovered:
        print(
            f"recovered {len(recovered)} generator-owned orphan(s) for {filename}",
            flush=True,
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RULER receipt: {output_path}")
    published_raw_root = raw_root / filename.removesuffix(".json")
    if published_raw_root.exists():
        raise FileExistsError(
            f"refusing to replace existing RULER diagnostics: {published_raw_root}"
        )
    receipt_key = _sha256_bytes(filename.encode("utf-8"))[:12]
    receipt_raw_root = Path(
        tempfile.mkdtemp(prefix=f".rq-{receipt_key}.", suffix=".staging", dir=raw_root)
    )
    pycache_prefix = receipt_raw_root / ".sealed-pycache"
    pycache_prefix.mkdir()
    _verify_empty_pycache_prefix(pycache_prefix)

    actual, portable = generator_argv(
        python=python,
        package_root=package_root,
        staged_root=staged_root,
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
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        TOKENIZERS_PARALLELISM="false",
        TRANSFORMERS_OFFLINE="1",
    )
    runtime_root = package_root.resolve().parents[1]
    verify_staged_runtime_package_tree(runtime_root, expected=package_tree_manifest)
    verify_staged_python_runtime(python_runtime_root, expected=python_runtime_manifest)
    verify_staged_runtime_inputs(runtime_input_root, expected=runtime_input_manifest)
    try:
        result = subprocess.run(
            actual,
            cwd=staged_root / "scripts" / "data",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
            check=False,
        )
    finally:
        verify_staged_ruler_source(staged_root, expected=staged_manifest)
        verify_staged_runtime_package_tree(runtime_root, expected=package_tree_manifest)
        verify_staged_python_runtime(python_runtime_root, expected=python_runtime_manifest)
        verify_staged_runtime_inputs(runtime_input_root, expected=runtime_input_manifest)
        _verify_empty_pycache_prefix(pycache_prefix)
        pycache_prefix.rmdir()
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
    # Publish the diagnostics first.  A crash can therefore leave an obvious,
    # fail-closed orphan, but can never leave a receipt without the raw inputs
    # needed to reproduce it.
    receipt_raw_root.rename(published_raw_root)
    _atomic_publish_new(output_path, payload)
    command_entry = _file_entry("generator/command-manifest.json", command_bytes)
    raw_entry = _file_entry("generator/raw-validation.jsonl", raw_data)
    return {
        "category": receipt["category"],
        "command_manifest": command_manifest,
        "command_manifest_file": command_entry,
        "config": config,
        "configured_length": receipt["configured_length"],
        "filename": filename,
        "generator_reported_length": normalized["generator_reported_length"],
        "phase": receipt["phase"],
        "raw_validation_base64": base64.b64encode(raw_data).decode("ascii"),
        "raw_validation_file": raw_entry,
        "seed": receipt["seed"],
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def finalize_generation_manifest_if_complete(
    *,
    required: Sequence[Mapping[str, object]],
    output_dir: Path,
    raw_root: Path,
    capture: Any,
    python: Path,
    package_root: Path,
    staged_root: Path,
    tokenizer: Any,
    static_entries: Sequence[Mapping[str, object]],
    source_manifest: Sequence[Mapping[str, object]],
    runtime_manifest: Mapping[str, object],
) -> dict[str, object] | None:
    """Publish the sole manifest only after re-verifying the full 20-file set."""

    manifest_path = output_dir / "generation-manifest.json"
    missing = [
        str(item["filename"])
        for item in required
        if not (output_dir / str(item["filename"])).is_file()
    ]
    if missing:
        if manifest_path.exists():
            raise ValueError("complete RULER generation manifest exists beside an incomplete set")
        return None

    # Do not trust results retained from the generation loop: independently
    # reopen every receipt, raw row, and command binding in canonical order.
    results = [
        _load_existing_receipt_result(
            path=output_dir / str(receipt["filename"]),
            receipt=receipt,
            capture=capture,
            python=python,
            package_root=package_root,
            staged_root=staged_root,
            raw_root=raw_root,
            tokenizer=tokenizer,
            static_entries=static_entries,
        )
        for receipt in required
    ]
    launcher_entry = _entry_named(static_entries, "launcher/generate_static_q468_ruler_receipts.py")
    source_manifest_value = [dict(item) for item in source_manifest]
    manifest: dict[str, object] = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "launcher_revision": LAUNCHER_REVISION,
        "launcher_source": dict(launcher_entry),
        "ruler_revision": capture.resolver.RULER_REVISION,
        "source_manifest": source_manifest_value,
        "source_manifest_sha256": _sha256_bytes(_canonical_json_bytes(source_manifest_value)),
        "runtime_manifest": dict(runtime_manifest),
        "runtime_manifest_sha256": _sha256_bytes(_canonical_json_bytes(runtime_manifest)),
        "static_inputs": [dict(item) for item in static_entries],
        "receipt_count": len(results),
        "receipts": results,
    }
    if len(results) != 20:
        raise ValueError("complete RULER generation manifest must contain exactly 20 receipts")
    _atomic_publish_same(manifest_path, _canonical_json_bytes(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
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
    checkout = verify_ruler_checkout(
        args.ruler_root,
        capture,
        git_executable_path=args.git_executable,
    )
    verified_package_tree = verify_runtime_package_source(args.python)
    verified_python_runtime = verify_python_runtime_source(args.python)
    static_runtime_files = verify_static_runtime_input_source(
        tokenizer_dir=args.tokenizer_dir,
        nltk_data=args.nltk_data,
        capture=capture,
    )
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
    with tempfile.TemporaryDirectory(prefix="recurquant-exp013-ruler-sealed-") as temporary:
        temporary_root = Path(temporary)
        python_runtime_root = temporary_root / "python-runtime"
        staged_python = stage_verified_python_runtime(
            python_runtime_root, verified=verified_python_runtime
        )
        package_runtime_root = temporary_root / "package-runtime"
        package_root = stage_verified_runtime_package_tree(
            package_runtime_root, verified=verified_package_tree
        )
        runtime_input_root = temporary_root / "runtime-inputs"
        runtime_input_manifest, staged_tokenizer_dir, staged_nltk_data = (
            stage_verified_runtime_inputs(
                runtime_input_root,
                files=static_runtime_files,
            )
        )
        runtime_manifest, resource_paths = verify_runtime(
            staged_python,
            staged_nltk_data,
            package_root=package_root,
            package_tree_manifest=verified_package_tree.entries,
            excluded_startup_files=verified_package_tree.excluded_startup_files,
            python_runtime_root=python_runtime_root,
            python_runtime_manifest=verified_python_runtime.entries,
            source_python=verified_python_runtime.source_launcher,
            source_pyvenv_config=verified_python_runtime.source_pyvenv_config,
        )
        verified_static_inputs = verify_static_inputs(
            ruler_root=args.ruler_root,
            tokenizer_dir=staged_tokenizer_dir,
            resource_paths=resource_paths,
            runtime_manifest=runtime_manifest,
            capture=capture,
        )
        if verified_static_inputs.sealed_runtime_files != static_runtime_files:
            raise ValueError("staged RULER tokenizer or NLTK bytes drifted during verification")
        static_entries = sorted(
            [
                *verified_static_inputs.entries,
                _file_entry(
                    "ruler/source-manifest.json",
                    _canonical_json_bytes(list(checkout.source_manifest)),
                ),
            ],
            key=lambda item: str(item["name"]),
        )
        tokenizer = IndependentTokenizer(
            python=staged_python,
            tokenizer_dir=staged_tokenizer_dir,
            package_root=package_root,
            package_tree_manifest=verified_package_tree.entries,
            runtime_input_root=runtime_input_root,
            runtime_input_manifest=runtime_input_manifest,
            python_runtime_root=python_runtime_root,
            python_runtime_manifest=verified_python_runtime.entries,
        )
        staged_root = temporary_root / "ruler-source"
        staged_manifest = stage_verified_ruler_source(
            staged_root,
            checkout=checkout,
            static_inputs=verified_static_inputs,
        )
        for index, receipt in enumerate(selected, start=1):
            output_path = args.output_dir / str(receipt["filename"])
            if output_path.exists():
                print(
                    f"[{index}/{len(selected)}] verifying existing {receipt['filename']}",
                    flush=True,
                )
                _load_existing_receipt_result(
                    path=output_path,
                    receipt=receipt,
                    capture=capture,
                    python=staged_python,
                    package_root=package_root,
                    staged_root=staged_root,
                    raw_root=args.raw_dir,
                    tokenizer=tokenizer,
                    static_entries=static_entries,
                )
            else:
                print(f"[{index}/{len(selected)}] generating {receipt['filename']}", flush=True)
                generate_receipt(
                    receipt=receipt,
                    capture=capture,
                    python=staged_python,
                    package_root=package_root,
                    package_tree_manifest=verified_package_tree.entries,
                    python_runtime_root=python_runtime_root,
                    python_runtime_manifest=verified_python_runtime.entries,
                    runtime_input_root=runtime_input_root,
                    runtime_input_manifest=runtime_input_manifest,
                    staged_root=staged_root,
                    staged_manifest=staged_manifest,
                    tokenizer_dir=staged_tokenizer_dir,
                    nltk_data=staged_nltk_data,
                    raw_root=args.raw_dir,
                    output_dir=args.output_dir,
                    tokenizer=tokenizer,
                    static_entries=static_entries,
                    runtime_manifest=runtime_manifest,
                    timeout_seconds=args.timeout_seconds,
                )
        manifest = finalize_generation_manifest_if_complete(
            required=required,
            output_dir=args.output_dir,
            raw_root=args.raw_dir,
            capture=capture,
            python=staged_python,
            package_root=package_root,
            staged_root=staged_root,
            tokenizer=tokenizer,
            static_entries=static_entries,
            source_manifest=checkout.source_manifest,
            runtime_manifest=runtime_manifest,
        )
        verify_staged_python_runtime(python_runtime_root, expected=verified_python_runtime.entries)
        verify_staged_runtime_package_tree(
            package_runtime_root, expected=verified_package_tree.entries
        )
        verify_staged_runtime_inputs(runtime_input_root, expected=runtime_input_manifest)
    if manifest is None:
        present = sum((args.output_dir / str(item["filename"])).is_file() for item in required)
        progress = {
            "schema": "recurquant.experiment013.ruler-generation-progress.v1",
            "complete": False,
            "present_receipts": present,
            "required_receipts": len(required),
        }
        print(json.dumps(progress, indent=2, sort_keys=True), flush=True)
    else:
        manifest_path = args.output_dir / "generation-manifest.json"
        summary = {
            "schema": "recurquant.experiment013.ruler-generation-success.v1",
            "complete": True,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
            "receipt_count": manifest["receipt_count"],
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
