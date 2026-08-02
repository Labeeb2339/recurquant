"""Stdlib-only live-adapter contract for Experiment 013 calibration.

This module intentionally imports no tensor, model, dataset, or Hub library.
The runner authenticates these exact source bytes before importing the module.
Adapter factories receive paths as inert :class:`~pathlib.Path` values and must
perform no I/O during construction.  Dataset/tokenizer access starts only in
``materialize_sequence``; model-file access starts only in ``load_model``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, runtime_checkable

EXECUTION_BINDING_ARTIFACT_KEYS = frozenset(
    {
        "calibration_runtime_manifest_bytes",
        "model_file_manifest_bytes",
        "parquet_materialization_manifest_bytes",
        "repository_source_manifest_bytes",
    }
)
RUNTIME_AUTHENTICATION_CONTEXT_KEYS = frozenset(
    {
        "base_runtime_root",
        "package_import_paths",
        "package_runtime_roots",
        "staged_interpreter",
    }
)


def _absolute_inert_path(value: object, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{name} must be an absolute normalized Path")
    return Path(value)


def _canonical_package_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or value[-1] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in value)
    ):
        raise ValueError("runtime package-root names must be canonical lowercase names")
    return value


def _canonical_import_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"runtime import path for {name} is not canonical")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"runtime import path for {name} is not canonical")
    return value


def _normalize_runtime_authentication_context(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_AUTHENTICATION_CONTEXT_KEYS:
        raise ValueError("runtime_authentication_context keys differ from the frozen API")
    base_root = _absolute_inert_path(value["base_runtime_root"], name="base_runtime_root")
    interpreter = _absolute_inert_path(
        value["staged_interpreter"],
        name="staged_interpreter",
    )
    try:
        interpreter.relative_to(base_root)
    except ValueError as error:
        raise ValueError("staged_interpreter must be inside base_runtime_root") from error
    raw_roots = value["package_runtime_roots"]
    raw_imports = value["package_import_paths"]
    if not isinstance(raw_roots, Mapping) or not raw_roots:
        raise ValueError("package_runtime_roots must be a non-empty mapping")
    if not isinstance(raw_imports, Mapping) or set(raw_imports) != set(raw_roots):
        raise ValueError("package runtime-root and import-path names must match exactly")
    normalized_names = tuple(_canonical_package_name(raw_name) for raw_name in raw_roots)
    package_roots: dict[str, Path] = {}
    import_paths: dict[str, str] = {}
    for name in sorted(normalized_names):
        package_roots[name] = _absolute_inert_path(
            raw_roots[name],
            name=f"package_runtime_roots[{name!r}]",
        )
        import_paths[name] = _canonical_import_path(raw_imports[name], name=name)
    return MappingProxyType(
        {
            "base_runtime_root": base_root,
            "package_import_paths": MappingProxyType(import_paths),
            "package_runtime_roots": MappingProxyType(package_roots),
            "staged_interpreter": interpreter,
        }
    )


@dataclass(frozen=True, slots=True)
class AdapterConstructionContext:
    """Authenticated locations supplied to the reviewed adapter factory.

    Construction is a pure wiring step.  The factory must not stat, resolve,
    open, import from, or otherwise inspect any of these locations.
    """

    repository_root: Path
    model_root: Path
    cache_root: Path
    ruler_root: Path
    runtime_authentication_context: Mapping[str, object]
    execution_binding_artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_authentication_context",
            _normalize_runtime_authentication_context(self.runtime_authentication_context),
        )
        if set(self.execution_binding_artifacts) != EXECUTION_BINDING_ARTIFACT_KEYS:
            raise ValueError("execution_binding_artifacts keys differ from the frozen API")
        normalized: dict[str, bytes] = {}
        for key in sorted(EXECUTION_BINDING_ARTIFACT_KEYS):
            value = self.execution_binding_artifacts[key]
            if not isinstance(value, bytes):
                raise TypeError(f"execution_binding_artifacts[{key!r}] must be bytes")
            normalized[key] = bytes(value)
        object.__setattr__(self, "execution_binding_artifacts", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class AuthenticatedSequence:
    """Exact token sequence and content commitments returned by an adapter."""

    token_ids: tuple[int, ...]
    source_content_sha256: str
    formatted_content_sha256: str
    generator_receipt_sha256: str | None
    tokenizer_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class StepObservation:
    """One reviewed one-token recurrent-kernel observation.

    Tensor values use ``object`` so importing the contract cannot import a
    tensor runtime.  The authenticated runner performs the exact tensor checks.
    """

    position: int
    token_id: int
    layer_indices: tuple[int, ...]
    recurrence_query: object
    recurrent_state: object | None
    successful_kernel_calls_per_layer: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelFileIdentity:
    """One immutable file in the authenticated local model snapshot."""

    name: str
    size_bytes: int
    sha256: str | None
    git_blob_oid: str
    lfs_sha256: str | None
    lfs_size_bytes: int | None


@dataclass(frozen=True, slots=True)
class AuthenticatedModelFiles:
    """Point-of-use model-file contract passed to ``load_model``."""

    model_root: Path
    model_id: str
    revision: str
    transformers_version: str
    files: tuple[ModelFileIdentity, ...]
    hub_tree_manifest_sha256: str
    manifest_file_sha256: str


@runtime_checkable
class CalibrationAdapter(Protocol):
    """Reviewed bridge from authenticated data/model files to causal tensors."""

    def materialize_sequence(self, record: Mapping[str, object]) -> AuthenticatedSequence: ...

    def load_model(self, authenticated: AuthenticatedModelFiles) -> object: ...

    def begin_sequence(self, model: object, record: Mapping[str, object]) -> None: ...

    def step_token(
        self,
        model: object,
        *,
        token_id: int,
        position: int,
        capture_state: bool,
    ) -> StepObservation: ...

    def end_sequence(self, model: object, record: Mapping[str, object]) -> None: ...

    def close_model(self, model: object) -> None: ...

    def runtime_metadata(self) -> Mapping[str, object]: ...


@runtime_checkable
class CalibrationAdapterFactory(Protocol):
    """Pure constructor for the single reviewed live adapter."""

    def __call__(self, context: AdapterConstructionContext, /) -> CalibrationAdapter: ...


__all__ = [
    "AdapterConstructionContext",
    "AuthenticatedModelFiles",
    "AuthenticatedSequence",
    "CalibrationAdapter",
    "CalibrationAdapterFactory",
    "EXECUTION_BINDING_ARTIFACT_KEYS",
    "ModelFileIdentity",
    "RUNTIME_AUTHENTICATION_CONTEXT_KEYS",
    "StepObservation",
]
