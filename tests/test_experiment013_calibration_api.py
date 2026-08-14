from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from recurquant.experiment013_calibration_api import (
    RUNTIME_AUTHENTICATION_CONTEXT_KEYS,
    AdapterConstructionContext,
    AuthenticatedModelFiles,
    AuthenticatedSequence,
    CalibrationAdapter,
    FisherStepObservation,
    ModelFileIdentity,
    StepObservation,
)

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "src" / "recurquant" / "experiment013_calibration_api.py"


def binding_artifacts() -> dict[str, bytes]:
    return {
        "calibration_runtime_manifest_bytes": b"runtime\n",
        "model_file_manifest_bytes": b"model\n",
        "parquet_materialization_manifest_bytes": b"parquet\n",
        "repository_source_manifest_bytes": b"source\n",
    }


def runtime_context() -> dict[str, object]:
    return {
        "base_runtime_root": ROOT / "runtime" / "base",
        "git_executable": ROOT / "tools" / "git.exe",
        "staged_interpreter": ROOT / "runtime" / "base" / "python.exe",
        "package_runtime_roots": {"calibration": ROOT / "runtime" / "packages"},
        "package_import_paths": {"calibration": "Lib/site-packages"},
    }


def test_context_copy_normalizes_exact_binding_bytes() -> None:
    source = binding_artifacts()
    context = AdapterConstructionContext(
        repository_root=ROOT,
        model_root=ROOT / "model",
        cache_root=ROOT / "cache",
        ruler_root=ROOT / "ruler",
        runtime_authentication_context=runtime_context(),
        execution_binding_artifacts=source,
    )

    source["model_file_manifest_bytes"] = b"changed"
    assert context.execution_binding_artifacts["model_file_manifest_bytes"] == b"model\n"
    with pytest.raises(TypeError):
        context.execution_binding_artifacts["new"] = b"no"  # type: ignore[index]

    incomplete = binding_artifacts()
    incomplete.pop("model_file_manifest_bytes")
    with pytest.raises(ValueError, match="keys differ"):
        AdapterConstructionContext(ROOT, ROOT, ROOT, ROOT, runtime_context(), incomplete)


def test_context_normalizes_and_freezes_runtime_authentication_paths() -> None:
    source = runtime_context()
    package_roots = source["package_runtime_roots"]
    package_import_paths = source["package_import_paths"]
    assert isinstance(package_roots, dict)
    assert isinstance(package_import_paths, dict)
    context = AdapterConstructionContext(
        ROOT,
        ROOT,
        ROOT,
        ROOT,
        source,
        binding_artifacts(),
    )
    source["git_executable"] = ROOT / "changed-git.exe"
    source["package_runtime_roots"] = {}
    package_roots["calibration"] = ROOT / "changed-packages"
    package_import_paths["calibration"] = "changed/site-packages"
    assert set(context.runtime_authentication_context) == RUNTIME_AUTHENTICATION_CONTEXT_KEYS
    assert context.runtime_authentication_context["base_runtime_root"] == (
        ROOT / "runtime" / "base"
    )
    assert context.runtime_authentication_context["git_executable"] == (ROOT / "tools" / "git.exe")
    assert context.runtime_authentication_context["package_runtime_roots"] == {
        "calibration": ROOT / "runtime" / "packages"
    }
    assert context.runtime_authentication_context["package_import_paths"] == {
        "calibration": "Lib/site-packages"
    }
    with pytest.raises(TypeError):
        context.runtime_authentication_context["new"] = ROOT  # type: ignore[index]
    with pytest.raises(TypeError):
        context.runtime_authentication_context["package_runtime_roots"]["new"] = ROOT  # type: ignore[index]
    with pytest.raises(TypeError):
        context.runtime_authentication_context["package_import_paths"]["new"] = "Lib"  # type: ignore[index]

    malformed = runtime_context()
    malformed["package_import_paths"] = {"calibration": "../site-packages"}
    with pytest.raises(ValueError, match="not canonical"):
        AdapterConstructionContext(ROOT, ROOT, ROOT, ROOT, malformed, binding_artifacts())

    relative_git = runtime_context()
    relative_git["git_executable"] = Path("git.exe")
    with pytest.raises(ValueError, match="git_executable must be an absolute normalized Path"):
        AdapterConstructionContext(ROOT, ROOT, ROOT, ROOT, relative_git, binding_artifacts())

    missing_git = runtime_context()
    missing_git.pop("git_executable")
    with pytest.raises(ValueError, match="keys differ"):
        AdapterConstructionContext(ROOT, ROOT, ROOT, ROOT, missing_git, binding_artifacts())


def test_adapter_facing_values_have_one_stable_importable_identity() -> None:
    sequence = AuthenticatedSequence((1, 2), "a" * 64, "b" * 64, None, "c" * 64)
    observation = StepObservation(0, 1, (0,), object(), None, (1,))
    fisher = FisherStepObservation(0, 1, 2, 2, 3, observation, object(), object(), 1.25)
    file_identity = ModelFileIdentity(
        "model.safetensors",
        1,
        "d" * 64,
        "e" * 40,
        "d" * 64,
        1,
    )
    authenticated = AuthenticatedModelFiles(
        ROOT,
        "example/model",
        "f" * 40,
        "5.14.1",
        (file_identity,),
        "1" * 64,
        "2" * 64,
    )

    assert sequence.token_ids == (1, 2)
    assert observation.successful_kernel_calls_per_layer == (1,)
    assert (fisher.boundary_position, fisher.input_position, fisher.target_position) == (0, 1, 2)
    assert fisher.step_observation is observation
    assert authenticated.files == (file_identity,)


def test_api_import_is_stdlib_only_even_when_loaded_under_external_name() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path
path = Path({str(API_PATH)!r})
spec = importlib.util.spec_from_file_location('external_experiment013_api', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'datasets' not in sys.modules
assert module.AuthenticatedSequence.__module__ == 'external_experiment013_api'
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_runtime_protocol_accepts_structural_adapter() -> None:
    class Adapter:
        def materialize_sequence(self, record: object) -> object:
            del record
            return object()

        def load_model(self, authenticated: object) -> object:
            del authenticated
            return object()

        def begin_sequence(self, model: object, record: object) -> None:
            del model, record

        def step_token(self, model: object, **kwargs: object) -> object:
            del model, kwargs
            return object()

        def step_token_with_fisher(self, model: object, **kwargs: object) -> object:
            del model, kwargs
            return object()

        def end_sequence(self, model: object, record: object) -> None:
            del model, record

        def close_model(self, model: object) -> None:
            del model

        def runtime_metadata(self) -> dict[str, object]:
            return {}

    assert isinstance(Adapter(), CalibrationAdapter)
