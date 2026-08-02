"""Reviewed live Qwen3.5 adapter for Experiment 013 calibration.

The factory in this module is deliberately inert.  Dataset/tokenizer access is
deferred until the first sequence materialization, and model-file access is
deferred until :meth:`Experiment013Qwen35Adapter.load_model`.  The adapter uses
the canonical capture module for all formatting and observes the exact
post-convolution query and returned recurrent state at the pinned Transformers
Gated DeltaNet kernel boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import torch

from .experiment013_calibration_api import (
    AdapterConstructionContext,
    AuthenticatedModelFiles,
    AuthenticatedSequence,
    StepObservation,
)

ADAPTER_REVISION: Final = "experiment-013-qwen35-live-adapter-v1"
ADAPTER_SOURCE_PATH: Final = "src/recurquant/experiment013_qwen35_adapter.py"
CAPTURE_SOURCE_PATH: Final = "scripts/capture_static_q468_identity_input.py"
CAPTURE_MODULE_NAME: Final = "_recurquant_experiment013_capture_for_live_adapter"
SOURCE_MANIFEST_SCHEMA: Final = "recurquant.experiment013.source-manifest.v1"

MODEL_ID: Final = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION: Final = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
TRANSFORMERS_VERSION: Final = "5.14.1"
MODEL_DTYPE_NAME: Final = "bfloat16"

RECURRENT_LAYER_INDICES: Final = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
)
LAYER_TYPES: Final = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(24)
)
QUERY_SHAPE: Final = (1, 1, 16, 128)
STATE_SHAPE: Final = (1, 16, 128, 128)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_WEIGHT_FILE_RE: Final = re.compile(
    r"(?:model(?:-[0-9]+-of-[0-9]+)?|model\.safetensors-[0-9]+-of-[0-9]+)\.safetensors"
)
_OBSERVER_ATTRIBUTE: Final = "_recurquant_experiment013_qwen35_observer"
_FILE_ATTRIBUTE_REPARSE_POINT: Final = 0x0400

_GATED_DELTA_GEOMETRY: Final = {
    "hidden_size": 1024,
    "num_v_heads": 16,
    "num_k_heads": 16,
    "head_k_dim": 128,
    "head_v_dim": 128,
    "key_dim": 2048,
    "value_dim": 2048,
    "conv_kernel_size": 4,
    "conv_dim": 6144,
}

_LOADING_DIAGNOSTIC_TYPES: Final = {
    "missing_keys": set,
    "unexpected_keys": set,
    "mismatched_keys": set,
    "error_msgs": list,
}

_CAPTURE_BINDING_KEYS: Final = {
    "repository_source_manifest_file_sha256": "repository_source_manifest_bytes",
    "calibration_runtime_manifest_file_sha256": "calibration_runtime_manifest_bytes",
    "model_file_manifest_file_sha256": "model_file_manifest_bytes",
    "parquet_materialization_manifest_file_sha256": ("parquet_materialization_manifest_bytes"),
}


class Experiment013AdapterError(RuntimeError):
    """Raised when the reviewed adapter cannot prove its live observation."""


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment013AdapterError(f"{context} must be a lowercase SHA-256")
    return value


def _require_non_negative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Experiment013AdapterError(f"{context} must be a non-negative integer")
    return value


def _argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    name: str,
    position: int,
) -> object | None:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


def _capture_manifest_sha256(source_manifest_bytes: bytes) -> str:
    if not isinstance(source_manifest_bytes, bytes):
        raise Experiment013AdapterError("repository source manifest must be exact bytes")
    try:
        manifest = json.loads(source_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment013AdapterError("repository source manifest is not UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise Experiment013AdapterError("repository source manifest schema drifted")
    entries = manifest.get("paths")
    if not isinstance(entries, list):
        raise Experiment013AdapterError("repository source manifest paths are not a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == CAPTURE_SOURCE_PATH
    ]
    if len(matches) != 1:
        raise Experiment013AdapterError(
            "repository source manifest must bind exactly one capture source"
        )
    return _require_sha256(
        matches[0].get("raw_sha256"),
        context="repository source manifest capture source",
    )


def _assert_no_link_or_reparse(repository_root: Path, source_path: Path) -> None:
    root = Path(os.path.abspath(repository_root))
    path = Path(os.path.abspath(source_path))
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Experiment013AdapterError(
            "capture source resolves outside repository_root"
        ) from error
    candidates = [root]
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        candidates.append(current)
    for candidate in candidates:
        try:
            status = candidate.lstat()
        except OSError as error:
            raise Experiment013AdapterError("capture source path is unavailable") from error
        if stat.S_ISLNK(status.st_mode) or (
            getattr(status, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise Experiment013AdapterError("capture source path traverses a link or reparse point")


def _read_authenticated_capture_source(
    repository_root: Path,
    source_path: Path,
    expected_sha256: str,
) -> bytes:
    _assert_no_link_or_reparse(repository_root, source_path)
    try:
        before = source_path.stat(follow_symlinks=False)
        payload = source_path.read_bytes()
        after = source_path.stat(follow_symlinks=False)
    except OSError as error:
        raise Experiment013AdapterError("authenticated capture source is unreadable") from error
    _assert_no_link_or_reparse(repository_root, source_path)
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(after.st_mode):
        raise Experiment013AdapterError("authenticated capture source is not a regular file")
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise Experiment013AdapterError("capture source changed while it was read")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise Experiment013AdapterError(
            "capture source differs from the repository source manifest"
        )
    return payload


@dataclass(frozen=True, slots=True)
class _CaptureModuleBinding:
    module: ModuleType
    repository_root: Path
    source_path: Path
    raw_sha256: str


def _verify_capture_binding(binding: _CaptureModuleBinding) -> None:
    if not isinstance(binding, _CaptureModuleBinding):
        raise Experiment013AdapterError("capture module lacks an authenticated source binding")
    if binding.module.__name__ != CAPTURE_MODULE_NAME or getattr(
        binding.module, "__file__", None
    ) != str(binding.source_path):
        raise Experiment013AdapterError("capture module identity differs from its source binding")
    if CAPTURE_MODULE_NAME in sys.modules:
        raise Experiment013AdapterError(
            "capture module name was preloaded outside the exact loader"
        )
    _read_authenticated_capture_source(
        binding.repository_root,
        binding.source_path,
        binding.raw_sha256,
    )


def _load_capture_module(
    repository_root: Path,
    source_manifest_bytes: bytes,
) -> _CaptureModuleBinding:
    """Execute only the manifest-bound capture bytes and recheck them afterwards."""

    if CAPTURE_MODULE_NAME in sys.modules:
        raise Experiment013AdapterError("capture module name is already loaded")
    root = Path(os.path.abspath(repository_root))
    path = root / Path(CAPTURE_SOURCE_PATH)
    expected_sha256 = _capture_manifest_sha256(source_manifest_bytes)
    payload = _read_authenticated_capture_source(root, path, expected_sha256)
    if CAPTURE_MODULE_NAME in sys.modules:
        raise Experiment013AdapterError("capture module name is already loaded")
    module = ModuleType(CAPTURE_MODULE_NAME)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[CAPTURE_MODULE_NAME] = module
    try:
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if sys.modules.get(CAPTURE_MODULE_NAME) is not module:
            raise Experiment013AdapterError("capture module replaced its exact loader binding")
    finally:
        sys.modules.pop(CAPTURE_MODULE_NAME, None)
        _read_authenticated_capture_source(root, path, expected_sha256)
    return _CaptureModuleBinding(
        module=module,
        repository_root=root,
        source_path=path,
        raw_sha256=expected_sha256,
    )


@dataclass(frozen=True, slots=True)
class _TransformersRuntime:
    version: str
    qwen_config_class: type[Any]
    qwen_model_class: type[Any]
    qwen_gated_delta_net_class: type[Any]
    dynamic_cache_class: type[Any]
    torch_chunk_gated_delta_rule: Any
    torch_recurrent_gated_delta_rule: Any
    torch_causal_conv1d_update: Any


def _load_transformers_runtime() -> _TransformersRuntime:
    import transformers
    from transformers import DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        torch_causal_conv1d_update,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    return _TransformersRuntime(
        version=str(transformers.__version__),
        qwen_config_class=Qwen3_5TextConfig,
        qwen_model_class=Qwen3_5ForCausalLM,
        qwen_gated_delta_net_class=Qwen3_5GatedDeltaNet,
        dynamic_cache_class=DynamicCache,
        torch_chunk_gated_delta_rule=torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule=torch_recurrent_gated_delta_rule,
        torch_causal_conv1d_update=torch_causal_conv1d_update,
    )


def _validate_qwen_config(config: object) -> None:
    exact = {
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    for name, expected in exact.items():
        if getattr(config, name, None) != expected:
            raise Experiment013AdapterError(
                f"authenticated Qwen3.5 config {name} differs from {expected!r}"
            )
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or tuple(layer_types) != LAYER_TYPES:
        raise Experiment013AdapterError("authenticated Qwen3.5 layer schedule drifted")


def _qwen_modules(
    model: object,
    runtime: _TransformersRuntime,
) -> tuple[tuple[int, object], ...]:
    text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
        raise Experiment013AdapterError("loaded Qwen3.5 model does not expose indexed layers")
    if len(layers) != len(LAYER_TYPES):
        raise Experiment013AdapterError("loaded Qwen3.5 model does not expose exactly 24 layers")
    result: list[tuple[int, object]] = []
    for layer_index in RECURRENT_LAYER_INDICES:
        module = getattr(layers[layer_index], "linear_attn", None)
        if module is None or type(module) is not runtime.qwen_gated_delta_net_class:
            raise Experiment013AdapterError(
                f"Qwen3.5 layer {layer_index} is not the pinned Gated DeltaNet module"
            )
        if getattr(module, "layer_idx", None) != layer_index:
            raise Experiment013AdapterError("Qwen3.5 Gated DeltaNet layer index drifted")
        for name, expected in _GATED_DELTA_GEOMETRY.items():
            if getattr(module, name, None) != expected:
                raise Experiment013AdapterError(
                    f"Qwen3.5 Gated DeltaNet {name} geometry drifted at layer {layer_index}"
                )
        if getattr(module, "layer_type", None) != "linear_attention":
            raise Experiment013AdapterError("Qwen3.5 Gated DeltaNet layer type drifted")
        result.append((layer_index, module))
    return tuple(result)


def _freeze_torch_fallbacks(
    modules: Sequence[tuple[int, object]], runtime: _TransformersRuntime
) -> None:
    for _layer_index, module in modules:
        module.causal_conv1d_fn = None  # type: ignore[attr-defined]
        module.causal_conv1d_update = runtime.torch_causal_conv1d_update  # type: ignore[attr-defined]
        module.chunk_gated_delta_rule = runtime.torch_chunk_gated_delta_rule  # type: ignore[attr-defined]
        module.recurrent_gated_delta_rule = (  # type: ignore[attr-defined]
            runtime.torch_recurrent_gated_delta_rule
        )
        if (
            module.causal_conv1d_fn is not None  # type: ignore[attr-defined]
            or module.causal_conv1d_update  # type: ignore[attr-defined]
            is not runtime.torch_causal_conv1d_update
            or module.chunk_gated_delta_rule  # type: ignore[attr-defined]
            is not runtime.torch_chunk_gated_delta_rule
            or module.recurrent_gated_delta_rule  # type: ignore[attr-defined]
            is not runtime.torch_recurrent_gated_delta_rule
        ):
            raise Experiment013AdapterError("could not freeze the pure-Torch Qwen3.5 kernels")


def _model_devices(model: object) -> set[torch.device]:
    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise Experiment013AdapterError("loaded model does not expose parameters and buffers")
    devices = {parameter.device for parameter in parameters()}
    devices.update(buffer.device for buffer in buffers())
    return devices


@dataclass(slots=True)
class _KernelReceipt:
    layer_index: int
    kernel_name: str
    query: torch.Tensor
    final_state: torch.Tensor


@dataclass(slots=True)
class _StepCapture:
    cache: object
    position: int
    receipts: dict[int, _KernelReceipt]

    @property
    def expected_kernel(self) -> str:
        return "chunk_gated_delta_rule" if self.position == 0 else "recurrent_gated_delta_rule"


class _Qwen35StepObserver:
    """Token-scoped observer for the exact pinned Gated DeltaNet kernels."""

    _POSITIONS: Final = {
        "chunk_gated_delta_rule": {
            "initial_state": 6,
            "output_final_state": 7,
            "use_qk_l2norm_in_kernel": 8,
        },
        "recurrent_gated_delta_rule": {
            "initial_state": 5,
            "output_final_state": 6,
            "use_qk_l2norm_in_kernel": 7,
        },
    }

    def __init__(
        self,
        modules: Sequence[tuple[int, object]],
        *,
        query_device: torch.device,
        _allow_test_non_cuda: bool = False,
    ) -> None:
        normalized = tuple(modules)
        if tuple(index for index, _module in normalized) != RECURRENT_LAYER_INDICES:
            raise Experiment013AdapterError("observer modules differ from frozen recurrent layers")
        if not isinstance(query_device, torch.device):
            raise TypeError("query_device must be torch.device")
        if query_device.type != "cuda" and not _allow_test_non_cuda:
            raise Experiment013AdapterError("Qwen3.5 query device must be CUDA")
        self.modules = normalized
        self.query_device = query_device
        self._active: ContextVar[_StepCapture | None] = ContextVar(
            f"recurquant_experiment013_qwen35_step_{id(self)}",
            default=None,
        )
        self._installed: list[tuple[object, str, object]] = []

    @staticmethod
    def _cache_state(cache: object, layer_index: int) -> torch.Tensor | None:
        layers = getattr(cache, "layers", None)
        if layers is None or not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
            return None
        if layer_index >= len(layers):
            return None
        recurrent_states = getattr(layers[layer_index], "recurrent_states", None)
        if not isinstance(recurrent_states, Mapping):
            return None
        state = recurrent_states.get(0)
        return state if isinstance(state, torch.Tensor) else None

    def _make_wrapper(self, layer_index: int, kernel_name: str, original: Any):
        positions = self._POSITIONS[kernel_name]

        def wrapped(*args: object, **kwargs: object):
            capture = self._active.get()
            if capture is None:
                raise Experiment013AdapterError(
                    "Qwen3.5 state kernel ran outside an authenticated one-token step"
                )
            if kernel_name != capture.expected_kernel:
                raise Experiment013AdapterError(
                    f"position {capture.position} called unexpected {kernel_name}"
                )
            if layer_index in capture.receipts:
                raise Experiment013AdapterError(
                    f"Qwen3.5 layer {layer_index} called more than one state kernel"
                )
            query = _argument(args, kwargs, "query", 0)
            if (
                not isinstance(query, torch.Tensor)
                or tuple(query.shape) != QUERY_SHAPE
                or query.dtype != torch.bfloat16
                or query.device != self.query_device
            ):
                raise Experiment013AdapterError(
                    f"Qwen3.5 layer {layer_index} query must be BF16 {QUERY_SHAPE} "
                    f"on {self.query_device}"
                )
            initial_state = _argument(
                args,
                kwargs,
                "initial_state",
                positions["initial_state"],
            )
            if capture.position == 0:
                if initial_state is not None:
                    raise Experiment013AdapterError("the first token must have no recurrent state")
            else:
                cached = self._cache_state(capture.cache, layer_index)
                if not isinstance(initial_state, torch.Tensor) or initial_state is not cached:
                    raise Experiment013AdapterError(
                        "cached one-token recurrence did not use the exact persistent state"
                    )
            if (
                _argument(
                    args,
                    kwargs,
                    "output_final_state",
                    positions["output_final_state"],
                )
                is not True
                or _argument(
                    args,
                    kwargs,
                    "use_qk_l2norm_in_kernel",
                    positions["use_qk_l2norm_in_kernel"],
                )
                is not True
            ):
                raise Experiment013AdapterError(
                    "Qwen3.5 state kernel omitted the pinned final-state/L2-normalization flags"
                )

            output = original(*args, **kwargs)
            if not isinstance(output, tuple) or len(output) != 2:
                raise Experiment013AdapterError(
                    "Qwen3.5 state kernel did not return exactly (output, final_state)"
                )
            final_state = output[1]
            if (
                not isinstance(final_state, torch.Tensor)
                or tuple(final_state.shape) != STATE_SHAPE
                or final_state.dtype != torch.float32
                or final_state.device != query.device
            ):
                raise Experiment013AdapterError(
                    f"Qwen3.5 layer {layer_index} final state must be FP32 {STATE_SHAPE}"
                )
            # The receipt becomes visible only after the selected kernel returned.
            capture.receipts[layer_index] = _KernelReceipt(
                layer_index=layer_index,
                kernel_name=kernel_name,
                query=query,
                final_state=final_state,
            )
            return output

        return wrapped

    def install(self) -> None:
        if self._installed:
            raise Experiment013AdapterError("Qwen3.5 step observer is already installed")
        try:
            for layer_index, module in self.modules:
                if hasattr(module, _OBSERVER_ATTRIBUTE):
                    raise Experiment013AdapterError(
                        "another Experiment 013 observer is already installed"
                    )
                setattr(module, _OBSERVER_ATTRIBUTE, self)
                self._installed.append((module, _OBSERVER_ATTRIBUTE, _Missing))
                for kernel_name in self._POSITIONS:
                    original = getattr(module, kernel_name, None)
                    if not callable(original):
                        raise Experiment013AdapterError(
                            f"Qwen3.5 module has no callable {kernel_name}"
                        )
                    restore = (
                        original if kernel_name in getattr(module, "__dict__", {}) else _Missing
                    )
                    setattr(
                        module,
                        kernel_name,
                        self._make_wrapper(layer_index, kernel_name, original),
                    )
                    self._installed.append((module, kernel_name, restore))
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        while self._installed:
            target, name, original = self._installed.pop()
            if original is _Missing:
                if name in getattr(target, "__dict__", {}):
                    delattr(target, name)
            else:
                setattr(target, name, original)

    def activate(self, capture: _StepCapture) -> Token[_StepCapture | None]:
        if self._active.get() is not None:
            raise Experiment013AdapterError("nested Qwen3.5 token observation is forbidden")
        return self._active.set(capture)

    def deactivate(self, token: Token[_StepCapture | None]) -> None:
        self._active.reset(token)

    @property
    def is_idle(self) -> bool:
        return self._active.get() is None


class _MissingType:
    pass


_Missing = _MissingType()


@dataclass(slots=True)
class _SequenceState:
    cache: object
    identity_record_sha256: str
    token_count: int
    next_position: int = 0


class Experiment013Qwen35Adapter:
    """Canonical materialization and one-token Qwen3.5 observation adapter."""

    def __init__(self, context: AdapterConstructionContext) -> None:
        if not isinstance(context, AdapterConstructionContext):
            raise TypeError("context must be AdapterConstructionContext")
        # Do not resolve, stat, import, or open any path during construction.
        self._repository_root = Path(context.repository_root)
        self._model_root = Path(context.model_root)
        self._cache_root = Path(context.cache_root)
        self._ruler_root = Path(context.ruler_root)
        self._execution_binding_artifacts: Mapping[str, bytes] | None = dict(
            context.execution_binding_artifacts
        )
        self._runtime_authentication_context: Mapping[str, object] | None = (
            context.runtime_authentication_context
        )
        self._materialization_attempted = False
        self._materialized_sequences: dict[str, AuthenticatedSequence] | None = None
        self._capture_input_sha256: str | None = None
        self._token_sequence_manifest_sha256: str | None = None
        self._runtime: _TransformersRuntime | None = None
        self._model: object | None = None
        self._model_device: torch.device | None = None
        self._observer: _Qwen35StepObserver | None = None
        self._model_loading_diagnostic_counts: dict[str, int] | None = None
        self._sequence: _SequenceState | None = None
        self._sequence_failed = False

    def _prepare_materialization(self) -> None:
        if self._materialization_attempted:
            if self._materialized_sequences is None:
                raise Experiment013AdapterError("calibration materialization previously failed")
            return
        self._materialization_attempted = True
        bindings = self._execution_binding_artifacts
        try:
            if bindings is None:
                raise Experiment013AdapterError("execution-binding artifacts were already released")
            runtime_context = self._runtime_authentication_context
            if runtime_context is None:
                raise Experiment013AdapterError("runtime authentication context was released")
            source_manifest_bytes = bytes(bindings["repository_source_manifest_bytes"])
            capture_binding = _load_capture_module(
                self._repository_root,
                source_manifest_bytes,
            )
            capture = capture_binding.module
            source_class = getattr(capture, "LiveCaptureSource", None)
            materialize = getattr(capture, "materialize_calibration_identity_sequences", None)
            if not callable(source_class) or not callable(materialize):
                raise Experiment013AdapterError(
                    "authenticated capture module lacks the calibration materialization API"
                )
            source = source_class(
                cache_dir=self._cache_root,
                ruler_receipt_dir=self._ruler_root,
            )
            capture_bindings = {
                capture_key: bytes(bindings[context_key])
                for capture_key, context_key in _CAPTURE_BINDING_KEYS.items()
            }
            _verify_capture_binding(capture_binding)
            try:
                materialization = materialize(
                    source=source,
                    execution_binding_artifacts=capture_bindings,
                    runtime_authentication_context=runtime_context,
                )
            finally:
                _verify_capture_binding(capture_binding)
            sequences = getattr(materialization, "sequences", None)
            if not isinstance(sequences, tuple) or len(sequences) != 160:
                raise Experiment013AdapterError(
                    "canonical capture did not return exactly 160 calibration sequences"
                )
            tokenizer_hash = _require_sha256(
                getattr(materialization, "tokenizer_manifest_sha256", None),
                context="materialization tokenizer manifest",
            )
            prepared: dict[str, AuthenticatedSequence] = {}
            for sequence in sequences:
                record = getattr(sequence, "identity_record", None)
                token_ids = getattr(sequence, "sequence_token_ids", None)
                digest = _require_sha256(
                    getattr(sequence, "identity_record_sha256", None),
                    context="materialized identity record",
                )
                if (
                    not isinstance(record, Mapping)
                    or record.get("identity_record_sha256") != digest
                ):
                    raise Experiment013AdapterError("materialized identity record digest drifted")
                if not isinstance(token_ids, tuple):
                    raise Experiment013AdapterError(
                        "materialized sequence token IDs are not a tuple"
                    )
                if digest in prepared:
                    raise Experiment013AdapterError("materialization returned a duplicate identity")
                sequence_tokenizer_hash = _require_sha256(
                    record.get("tokenizer_manifest_sha256"),
                    context="materialized sequence tokenizer manifest",
                )
                if sequence_tokenizer_hash != tokenizer_hash:
                    raise Experiment013AdapterError("materialized tokenizer commitments differ")
                prepared[digest] = AuthenticatedSequence(
                    token_ids=tuple(token_ids),
                    source_content_sha256=_require_sha256(
                        record.get("source_content_sha256"),
                        context="materialized source content",
                    ),
                    formatted_content_sha256=_require_sha256(
                        record.get("formatted_content_sha256"),
                        context="materialized formatted content",
                    ),
                    generator_receipt_sha256=(
                        None
                        if record.get("generator_receipt_sha256") is None
                        else _require_sha256(
                            record.get("generator_receipt_sha256"),
                            context="materialized generator receipt",
                        )
                    ),
                    tokenizer_manifest_sha256=sequence_tokenizer_hash,
                )
            self._capture_input_sha256 = _require_sha256(
                getattr(materialization, "capture_input_sha256", None),
                context="materialization capture input",
            )
            self._token_sequence_manifest_sha256 = _require_sha256(
                getattr(materialization, "token_sequence_manifest_sha256", None),
                context="materialization token sequence manifest",
            )
            self._materialized_sequences = prepared
        finally:
            # These potentially large artifact bytes are needed only for the
            # canonical capture call.  Do not retain capture/source objects.
            self._execution_binding_artifacts = None
            self._runtime_authentication_context = None

    def materialize_sequence(self, record: Mapping[str, object]) -> AuthenticatedSequence:
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        self._prepare_materialization()
        digest = _require_sha256(record.get("identity_record_sha256"), context="identity record")
        assert self._materialized_sequences is not None
        try:
            result = self._materialized_sequences[digest]
        except KeyError as error:
            raise Experiment013AdapterError(
                f"canonical materialization has no identity record {digest}"
            ) from error
        exact = {
            "source_content_sha256": result.source_content_sha256,
            "formatted_content_sha256": result.formatted_content_sha256,
            "generator_receipt_sha256": result.generator_receipt_sha256,
            "tokenizer_manifest_sha256": result.tokenizer_manifest_sha256,
            "sequence_length": len(result.token_ids),
        }
        for name, expected in exact.items():
            if record.get(name) != expected:
                raise Experiment013AdapterError(
                    f"frozen identity {name} differs from canonical materialization"
                )
        return result

    def _validate_authenticated_model(self, authenticated: AuthenticatedModelFiles) -> Path:
        if not isinstance(authenticated, AuthenticatedModelFiles):
            raise TypeError("authenticated must be AuthenticatedModelFiles")
        if (
            authenticated.model_id != MODEL_ID
            or authenticated.revision != MODEL_REVISION
            or authenticated.transformers_version != TRANSFORMERS_VERSION
        ):
            raise Experiment013AdapterError(
                "authenticated model identity differs from the frozen model"
            )
        expected_root = self._model_root.resolve(strict=True)
        actual_root = authenticated.model_root.resolve(strict=True)
        if actual_root != expected_root or not actual_root.is_dir():
            raise Experiment013AdapterError("authenticated model root differs from adapter context")
        names = tuple(item.name for item in authenticated.files)
        if (
            names != tuple(sorted(names))
            or len(names) != len(set(names))
            or "config.json" not in names
            or "model.safetensors.index.json" not in names
            or not any(_WEIGHT_FILE_RE.fullmatch(name) for name in names)
            or any(
                name not in {"config.json", "model.safetensors.index.json"}
                and _WEIGHT_FILE_RE.fullmatch(name) is None
                for name in names
            )
        ):
            raise Experiment013AdapterError(
                "authenticated model file inventory is not the pinned profile"
            )
        return actual_root

    def load_model(self, authenticated: AuthenticatedModelFiles) -> object:
        if self._model is not None:
            raise Experiment013AdapterError("the live model is already loaded")
        model_root = self._validate_authenticated_model(authenticated)
        runtime = _load_transformers_runtime()
        if runtime.version != TRANSFORMERS_VERSION:
            raise Experiment013AdapterError("Transformers runtime differs from 5.14.1")
        if not torch.cuda.is_available():
            raise Experiment013AdapterError("official Experiment 013 model loading requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        config = runtime.qwen_config_class.from_pretrained(
            str(model_root),
            local_files_only=True,
            trust_remote_code=False,
        )
        _validate_qwen_config(config)
        loaded = runtime.qwen_model_class.from_pretrained(
            str(model_root),
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            weights_only=True,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if type(loaded) is not tuple or len(loaded) != 2:
            raise Experiment013AdapterError(
                "Transformers did not return exactly (model, loading_info)"
            )
        model, loading_info = loaded
        if type(loading_info) is not dict or set(loading_info) != set(_LOADING_DIAGNOSTIC_TYPES):
            raise Experiment013AdapterError("Transformers loading diagnostics schema drifted")
        diagnostic_counts: dict[str, int] = {}
        for name, expected_type in _LOADING_DIAGNOSTIC_TYPES.items():
            diagnostic = loading_info[name]
            if type(diagnostic) is not expected_type:
                raise Experiment013AdapterError(f"Transformers {name} diagnostic type drifted")
            diagnostic_counts[name] = len(diagnostic)
            if diagnostic:
                raise Experiment013AdapterError(
                    f"Transformers reported non-empty {name} while loading authenticated weights"
                )
        self._model_loading_diagnostic_counts = diagnostic_counts
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        _validate_qwen_config(getattr(model, "config", None))
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise Experiment013AdapterError("loaded Qwen3.5 model is not using eager attention")
        if getattr(model, "training", True):
            raise Experiment013AdapterError("loaded Qwen3.5 model remained in training mode")
        devices = _model_devices(model)
        parameters = tuple(model.parameters())
        if not parameters:
            raise Experiment013AdapterError("loaded Qwen3.5 model exposes no parameters")
        if (
            devices != {device}
            or any(parameter.requires_grad for parameter in parameters)
            or any(
                parameter.is_floating_point() and parameter.dtype != torch.bfloat16
                for parameter in parameters
            )
        ):
            raise Experiment013AdapterError(
                "loaded Qwen3.5 weights are not frozen BF16 on exactly one CUDA device"
            )
        modules = _qwen_modules(model, runtime)
        _freeze_torch_fallbacks(modules, runtime)
        observer = _Qwen35StepObserver(modules, query_device=device)
        observer.install()
        self._runtime = runtime
        self._model = model
        self._model_device = device
        self._observer = observer
        return model

    def _require_loaded_model(self, model: object) -> None:
        if model is not self._model or self._runtime is None or self._observer is None:
            raise Experiment013AdapterError("model is not the adapter's authenticated live model")

    def begin_sequence(self, model: object, record: Mapping[str, object]) -> None:
        self._require_loaded_model(model)
        if self._sequence is not None:
            raise Experiment013AdapterError("a calibration sequence is already active")
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        digest = _require_sha256(record.get("identity_record_sha256"), context="identity record")
        token_count = _require_non_negative_int(
            record.get("sequence_length"), context="sequence length"
        )
        if token_count == 0:
            raise Experiment013AdapterError("calibration sequence cannot be empty")
        assert self._runtime is not None
        cache = self._runtime.dynamic_cache_class(config=model.config)
        self._sequence = _SequenceState(
            cache=cache,
            identity_record_sha256=digest,
            token_count=token_count,
        )
        self._sequence_failed = False

    @staticmethod
    def _cache_length(cache: object) -> int:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            raise Experiment013AdapterError("DynamicCache does not expose get_seq_length")
        length = get_seq_length()
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise Experiment013AdapterError("DynamicCache returned an invalid sequence length")
        return length

    def step_token(
        self,
        model: object,
        *,
        token_id: int,
        position: int,
        capture_state: bool,
    ) -> StepObservation:
        self._require_loaded_model(model)
        sequence = self._sequence
        observer = self._observer
        device = self._model_device
        if sequence is None or observer is None or device is None or self._sequence_failed:
            raise Experiment013AdapterError("no healthy calibration sequence is active")
        token = _require_non_negative_int(token_id, context="token_id")
        current = _require_non_negative_int(position, context="position")
        if not isinstance(capture_state, bool):
            raise TypeError("capture_state must be bool")
        if current != sequence.next_position or current >= sequence.token_count:
            raise Experiment013AdapterError(
                "adapter token position is not the next causal position"
            )
        if getattr(model, "training", True):
            raise Experiment013AdapterError(
                "Qwen3.5 model entered training mode during calibration"
            )
        if self._cache_length(sequence.cache) != current:
            raise Experiment013AdapterError("DynamicCache length drifted before one-token forward")

        capture = _StepCapture(cache=sequence.cache, position=current, receipts={})
        context_token = observer.activate(capture)
        try:
            input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
            position_ids = torch.tensor([[current]], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=sequence.cache,
                    use_cache=True,
                )
            if getattr(output, "past_key_values", None) is not sequence.cache:
                raise Experiment013AdapterError("Qwen3.5 returned a different DynamicCache object")
            if self._cache_length(sequence.cache) != current + 1:
                raise Experiment013AdapterError("DynamicCache did not advance by exactly one token")
            if tuple(capture.receipts) != RECURRENT_LAYER_INDICES:
                raise Experiment013AdapterError(
                    "one-token forward did not produce one ordered receipt for every "
                    "recurrent layer"
                )
            queries: list[torch.Tensor] = []
            states: list[torch.Tensor] = []
            cache_matches: list[torch.Tensor] = []
            for layer_index in RECURRENT_LAYER_INDICES:
                receipt = capture.receipts[layer_index]
                cached = observer._cache_state(sequence.cache, layer_index)
                if (
                    cached is None
                    or tuple(cached.shape) != STATE_SHAPE
                    or cached.dtype != torch.float32
                    or cached.device != receipt.final_state.device
                ):
                    raise Experiment013AdapterError(
                        f"DynamicCache state geometry differs at layer {layer_index}"
                    )
                # Queue every exact comparison, then transfer only 18 scalar
                # receipts.  Calling torch.equal here would synchronize CUDA
                # separately for every layer and every calibration token.
                cache_matches.append(torch.eq(cached, receipt.final_state).all())
                queries.append(receipt.query[0, 0])
                if capture_state:
                    states.append(cached[0])
            match_values = torch.stack(cache_matches).detach().to(device="cpu").tolist()
            mismatched = [
                layer_index
                for layer_index, matches in zip(RECURRENT_LAYER_INDICES, match_values, strict=True)
                if not matches
            ]
            if mismatched:
                raise Experiment013AdapterError(
                    f"DynamicCache state differs from kernel output at layers {mismatched}"
                )
            recurrence_query = torch.stack(queries, dim=0).contiguous()
            recurrent_state = torch.stack(states, dim=0).contiguous() if capture_state else None
            sequence.next_position += 1
            return StepObservation(
                position=current,
                token_id=token,
                layer_indices=RECURRENT_LAYER_INDICES,
                recurrence_query=recurrence_query,
                recurrent_state=recurrent_state,
                successful_kernel_calls_per_layer=(1,) * len(RECURRENT_LAYER_INDICES),
            )
        except BaseException:
            self._sequence_failed = True
            raise
        finally:
            observer.deactivate(context_token)

    def end_sequence(self, model: object, record: Mapping[str, object]) -> None:
        self._require_loaded_model(model)
        del record
        # Cleanup must not mask a runner-side failure with a second exception.
        self._sequence = None
        self._sequence_failed = False

    def close_model(self, model: object) -> None:
        if self._model is None:
            return
        if model is not self._model:
            raise Experiment013AdapterError("refusing to close a different model object")
        self._sequence = None
        self._sequence_failed = False
        if self._observer is not None:
            self._observer.remove()
        self._observer = None
        self._runtime = None
        self._model_device = None
        self._model = None

    def runtime_metadata(self) -> Mapping[str, object]:
        return {
            "adapter_revision": ADAPTER_REVISION,
            "capture_input_sha256": self._capture_input_sha256,
            "device": None if self._model_device is None else str(self._model_device),
            "kernel_backend": "transformers_pure_torch_gated_delta_rule",
            "materialization_attempted": self._materialization_attempted,
            "materialized_sequence_count": (
                0 if self._materialized_sequences is None else len(self._materialized_sequences)
            ),
            "model_dtype": MODEL_DTYPE_NAME,
            "model_id": MODEL_ID,
            "model_loaded": self._model is not None,
            "model_loading_diagnostic_counts": (
                None
                if self._model_loading_diagnostic_counts is None
                else dict(self._model_loading_diagnostic_counts)
            ),
            "model_revision": MODEL_REVISION,
            "query_shape": list(QUERY_SHAPE),
            "recurrent_layer_indices": list(RECURRENT_LAYER_INDICES),
            "state_shape": list(STATE_SHAPE),
            "token_sequence_manifest_sha256": self._token_sequence_manifest_sha256,
            "transformers_version": TRANSFORMERS_VERSION,
        }

    @staticmethod
    def source_binding() -> Mapping[str, object]:
        """Return the adapter's own source bytes identity for independent checks."""

        payload = Path(__file__).read_bytes()
        return {
            "path": ADAPTER_SOURCE_PATH,
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
        }


def create_adapter(context: AdapterConstructionContext) -> Experiment013Qwen35Adapter:
    """Construct the canonical adapter without performing I/O or importing data/model code."""

    return Experiment013Qwen35Adapter(context)


__all__ = [
    "ADAPTER_REVISION",
    "Experiment013AdapterError",
    "Experiment013Qwen35Adapter",
    "create_adapter",
]
