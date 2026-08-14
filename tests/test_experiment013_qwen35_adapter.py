from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import types
import weakref
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from recurquant import experiment013_qwen35_adapter as adapter_module
from recurquant import experiment013_source as source_module
from recurquant.experiment013_calibration_api import (
    AdapterConstructionContext,
    AuthenticatedModelFiles,
    AuthenticatedSequence,
    CalibrationAdapter,
    ModelFileIdentity,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


def _context(tmp_path: Path) -> AdapterConstructionContext:
    return AdapterConstructionContext(
        repository_root=tmp_path / "repository-does-not-need-to-exist-at-construction",
        model_root=tmp_path / "model-does-not-need-to-exist-at-construction",
        cache_root=tmp_path / "cache-does-not-need-to-exist-at-construction",
        ruler_root=tmp_path / "ruler-does-not-need-to-exist-at-construction",
        runtime_authentication_context={
            "base_runtime_root": tmp_path / "runtime" / "base",
            "staged_interpreter": tmp_path / "runtime" / "base" / "python.exe",
            "package_runtime_roots": {"calibration": tmp_path / "runtime" / "packages"},
            "package_import_paths": {"calibration": "Lib/site-packages"},
        },
        execution_binding_artifacts={
            "repository_source_manifest_bytes": b"source-manifest\n",
            "calibration_runtime_manifest_bytes": b"runtime-manifest\n",
            "model_file_manifest_bytes": b"model-manifest\n",
            "parquet_materialization_manifest_bytes": b"parquet-manifest\n",
        },
    )


def _fake_capture_module(**attributes: object) -> types.ModuleType:
    module = types.ModuleType(adapter_module.CAPTURE_MODULE_NAME)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _fake_capture_binding(
    tmp_path: Path,
    module: types.ModuleType,
    *,
    payload: bytes = b"# authenticated test capture source\n",
) -> adapter_module._CaptureModuleBinding:
    root = _context(tmp_path).repository_root
    source_path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(payload)
    module.__file__ = str(source_path)
    return adapter_module._CaptureModuleBinding(
        module=module,
        repository_root=root,
        source_path=source_path,
        raw_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _source_manifest_bytes(payload: bytes) -> bytes:
    return json.dumps(
        {
            "schema": source_module.EXPERIMENT013_SOURCE_MANIFEST_SCHEMA,
            "paths": [
                {
                    "path": adapter_module.CAPTURE_SOURCE_PATH,
                    "raw_sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
    ).encode("utf-8")


def _git(root: Path, *arguments: str) -> None:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if process.returncode != 0:
        raise AssertionError(process.stderr or process.stdout)


def _canonical_v2_source_manifest_bytes(root: Path, payload: bytes) -> bytes:
    root.mkdir(parents=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Experiment 013 Adapter Test")
    _git(root, "config", "user.email", "experiment013-adapter@example.invalid")
    (root / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8", newline="\n")
    (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8", newline="\n")
    for relative in source_module.EXPERIMENT013_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            payload
            if relative == adapter_module.CAPTURE_SOURCE_PATH
            else f"adapter-v2-fixture:{relative}\n".encode()
        )
    _git(root, "add", "--", ".")
    _git(root, "commit", "-m", "adapter v2 source fixture")
    manifest = source_module.capture_experiment013_source_manifest(root)
    return source_module.canonical_experiment013_source_manifest_bytes(manifest)


def _identity_record(index: int, token_ids: tuple[int, ...]) -> dict[str, object]:
    return {
        "identity_record_sha256": _sha(index + 1),
        "source_content_sha256": _sha(1_000 + index),
        "formatted_content_sha256": _sha(2_000 + index),
        "generator_receipt_sha256": None,
        "tokenizer_manifest_sha256": _sha(3_000),
        "sequence_length": len(token_ids),
    }


class _FakeMaterializedSequence:
    def __init__(self, record: dict[str, object], token_ids: tuple[int, ...]) -> None:
        self.identity_record = dict(record)
        self.identity_record_sha256 = record["identity_record_sha256"]
        self.sequence_token_ids = token_ids


class _FakeMaterialization:
    def __init__(self) -> None:
        self.sequences = tuple(
            _FakeMaterializedSequence(
                _identity_record(index, (index, index + 1)),
                (index, index + 1),
            )
            for index in range(160)
        )
        self.tokenizer_manifest_sha256 = _sha(3_000)
        self.capture_input_sha256 = _sha(4_000)
        self.token_sequence_manifest_sha256 = _sha(5_000)
        self.private_source_text = "must-not-be-retained"


def test_factory_is_zero_io_and_uses_the_shared_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        adapter_module,
        "_load_capture_module",
        lambda *_args: pytest.fail("factory touched capture source"),
    )
    monkeypatch.setattr(
        adapter_module,
        "_load_transformers_runtime",
        lambda: pytest.fail("factory imported the model runtime"),
    )

    adapter = adapter_module.create_adapter(_context(tmp_path))

    assert isinstance(adapter, adapter_module.Experiment013Qwen35Adapter)
    assert isinstance(adapter, CalibrationAdapter)
    assert adapter.runtime_metadata()["materialization_attempted"] is False
    assert adapter.runtime_metadata()["model_loaded"] is False


def test_canonical_materialization_runs_once_and_retains_only_tokens_and_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, bytes]] = []
    runtime_calls: list[Mapping[str, object]] = []
    source_references: list[weakref.ReferenceType[object]] = []
    materialization_references: list[weakref.ReferenceType[object]] = []

    class FakeLiveCaptureSource:
        def __init__(self, *, cache_dir: Path, ruler_receipt_dir: Path) -> None:
            self.cache_dir = cache_dir
            self.ruler_receipt_dir = ruler_receipt_dir
            self.private_source_text = "must-not-be-retained"
            source_references.append(weakref.ref(self))

    def materialize(
        *,
        source: object,
        execution_binding_artifacts: dict[str, bytes],
        runtime_authentication_context: Mapping[str, object],
    ):
        assert isinstance(source, FakeLiveCaptureSource)
        calls.append(dict(execution_binding_artifacts))
        runtime_calls.append(runtime_authentication_context)
        result = _FakeMaterialization()
        materialization_references.append(weakref.ref(result))
        return result

    capture_module = _fake_capture_module(
        LiveCaptureSource=FakeLiveCaptureSource,
        materialize_calibration_identity_sequences=materialize,
    )
    capture_binding = _fake_capture_binding(tmp_path, capture_module)
    monkeypatch.setattr(
        adapter_module,
        "_load_capture_module",
        lambda _root, _manifest: capture_binding,
    )
    adapter = adapter_module.create_adapter(_context(tmp_path))

    first_record = _identity_record(0, (0, 1))
    first = adapter.materialize_sequence(first_record)
    second = adapter.materialize_sequence(_identity_record(1, (1, 2)))
    gc.collect()

    assert first == AuthenticatedSequence(
        token_ids=(0, 1),
        source_content_sha256=_sha(1_000),
        formatted_content_sha256=_sha(2_000),
        generator_receipt_sha256=None,
        tokenizer_manifest_sha256=_sha(3_000),
    )
    assert second.token_ids == (1, 2)
    assert calls == [
        {
            "repository_source_manifest_file_sha256": b"source-manifest\n",
            "calibration_runtime_manifest_file_sha256": b"runtime-manifest\n",
            "model_file_manifest_file_sha256": b"model-manifest\n",
            "parquet_materialization_manifest_file_sha256": b"parquet-manifest\n",
        }
    ]
    assert source_references[0]() is None
    assert materialization_references[0]() is None
    assert adapter._execution_binding_artifacts is None
    assert adapter._runtime_authentication_context is None
    assert runtime_calls[0]["staged_interpreter"] == (tmp_path / "runtime" / "base" / "python.exe")
    assert set(adapter._materialized_sequences or {}) == {_sha(index + 1) for index in range(160)}
    assert all(
        isinstance(value, AuthenticatedSequence)
        for value in (adapter._materialized_sequences or {}).values()
    )
    assert "must-not-be-retained" not in repr(adapter.__dict__)
    metadata = adapter.runtime_metadata()
    assert metadata["materialized_sequence_count"] == 160
    assert metadata["capture_input_sha256"] == _sha(4_000)
    assert metadata["token_sequence_manifest_sha256"] == _sha(5_000)


def test_materialization_rejects_frozen_commitment_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture_module = _fake_capture_module(
        LiveCaptureSource=lambda **_kwargs: object(),
        materialize_calibration_identity_sequences=lambda **_kwargs: _FakeMaterialization(),
    )
    capture_binding = _fake_capture_binding(tmp_path, capture_module)
    monkeypatch.setattr(
        adapter_module,
        "_load_capture_module",
        lambda _root, _manifest: capture_binding,
    )
    adapter = adapter_module.create_adapter(_context(tmp_path))
    record = _identity_record(0, (0, 1))
    record["formatted_content_sha256"] = _sha(99_999)

    with pytest.raises(adapter_module.Experiment013AdapterError, match="formatted_content"):
        adapter.materialize_sequence(record)


def test_capture_loader_executes_manifest_bound_bytes_without_leaving_a_module(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    path.parent.mkdir(parents=True)
    payload = b"AUTHENTICATED_VALUE = 17\n"
    path.write_bytes(payload)

    binding = adapter_module._load_capture_module(root, _source_manifest_bytes(payload))

    assert binding.module.AUTHENTICATED_VALUE == 17
    assert binding.raw_sha256 == hashlib.sha256(payload).hexdigest()
    assert adapter_module.CAPTURE_MODULE_NAME not in sys.modules
    adapter_module._verify_capture_binding(binding)


def test_capture_loader_accepts_canonical_v2_source_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    payload = b"AUTHENTICATED_VALUE = 23\n"
    manifest_bytes = _canonical_v2_source_manifest_bytes(root, payload)
    manifest = json.loads(manifest_bytes)

    assert manifest["schema"] == "recurquant.experiment013.source-manifest.v2"
    assert manifest["schema"] == source_module.EXPERIMENT013_SOURCE_MANIFEST_SCHEMA

    binding = adapter_module._load_capture_module(root, manifest_bytes)

    assert binding.module.AUTHENTICATED_VALUE == 23
    assert binding.raw_sha256 == hashlib.sha256(payload).hexdigest()
    assert adapter_module.CAPTURE_MODULE_NAME not in sys.modules
    adapter_module._verify_capture_binding(binding)


def test_capture_loader_rejects_retired_v1_source_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    path.parent.mkdir(parents=True)
    payload = b"AUTHENTICATED_VALUE = 17\n"
    path.write_bytes(payload)
    manifest = json.loads(_source_manifest_bytes(payload))
    manifest["schema"] = "recurquant.experiment013.source-manifest.v1"

    with pytest.raises(
        adapter_module.Experiment013AdapterError,
        match="repository source manifest schema drifted",
    ):
        adapter_module._load_capture_module(root, json.dumps(manifest).encode("utf-8"))


def test_capture_loader_rejects_every_preloaded_capture_module(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    path.parent.mkdir(parents=True)
    payload = b"AUTHENTICATED_VALUE = 17\n"
    path.write_bytes(payload)
    preloaded = types.ModuleType(adapter_module.CAPTURE_MODULE_NAME)
    preloaded.__file__ = str(path)
    sys.modules[adapter_module.CAPTURE_MODULE_NAME] = preloaded
    try:
        with pytest.raises(adapter_module.Experiment013AdapterError, match="already loaded"):
            adapter_module._load_capture_module(root, _source_manifest_bytes(payload))
    finally:
        sys.modules.pop(adapter_module.CAPTURE_MODULE_NAME, None)
    sys.modules[adapter_module.CAPTURE_MODULE_NAME] = None
    try:
        with pytest.raises(adapter_module.Experiment013AdapterError, match="already loaded"):
            adapter_module._load_capture_module(root, _source_manifest_bytes(payload))
    finally:
        sys.modules.pop(adapter_module.CAPTURE_MODULE_NAME, None)


def test_capture_loader_rechecks_source_after_execution(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    path.parent.mkdir(parents=True)
    payload = (
        b"from pathlib import Path\n"
        b"Path(__file__).write_bytes(b'tampered after authenticated execution')\n"
    )
    path.write_bytes(payload)

    with pytest.raises(
        adapter_module.Experiment013AdapterError,
        match="differs from the repository source manifest",
    ):
        adapter_module._load_capture_module(root, _source_manifest_bytes(payload))

    assert adapter_module.CAPTURE_MODULE_NAME not in sys.modules


def test_capture_loader_rejects_link_or_reparse_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    path = root / Path(adapter_module.CAPTURE_SOURCE_PATH)
    path.parent.mkdir(parents=True)
    payload = b"AUTHENTICATED_VALUE = 17\n"
    path.write_bytes(payload)
    monkeypatch.setattr(adapter_module.stat, "S_ISLNK", lambda _mode: True)

    with pytest.raises(adapter_module.Experiment013AdapterError, match="link or reparse"):
        adapter_module._load_capture_module(root, _source_manifest_bytes(payload))


def test_materialization_rechecks_capture_source_after_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_module = _fake_capture_module()
    capture_binding = _fake_capture_binding(tmp_path, capture_module)

    class FakeLiveCaptureSource:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def materialize(**_kwargs: object) -> _FakeMaterialization:
        capture_binding.source_path.write_bytes(b"changed during materialization\n")
        return _FakeMaterialization()

    capture_module.LiveCaptureSource = FakeLiveCaptureSource
    capture_module.materialize_calibration_identity_sequences = materialize
    monkeypatch.setattr(
        adapter_module,
        "_load_capture_module",
        lambda _root, _manifest: capture_binding,
    )
    adapter = adapter_module.create_adapter(_context(tmp_path))

    with pytest.raises(
        adapter_module.Experiment013AdapterError,
        match="differs from the repository source manifest",
    ):
        adapter.materialize_sequence(_identity_record(0, (0, 1)))

    assert adapter._materialized_sequences is None
    assert adapter._execution_binding_artifacts is None
    assert adapter._runtime_authentication_context is None


class _FakeCacheLayer:
    def __init__(self) -> None:
        self.recurrent_states: dict[int, torch.Tensor | None] = {0: None}
        self.is_recurrent_states_initialized: dict[int, bool] = {0: False}


class _FakeDynamicCache:
    corrupt_layer: int | None = None

    def __init__(self, *, config: object) -> None:
        del config
        self.layers = [_FakeCacheLayer() for _index in range(24)]
        self.sequence_length = 0

    def get_seq_length(self) -> int:
        return self.sequence_length

    def update_recurrent_state(self, state: torch.Tensor, layer_index: int) -> torch.Tensor:
        cached = self.layers[layer_index].recurrent_states[0]
        if cached is None:
            cached = torch.empty_like(state)
            self.layers[layer_index].recurrent_states[0] = cached
        cached.copy_(state)
        self.layers[layer_index].is_recurrent_states_initialized[0] = True
        if layer_index == self.corrupt_layer:
            cached.add_(1)
        return cached


class Qwen3_5GatedDeltaNet:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.layer_type = "linear_attention"
        for name, value in adapter_module._GATED_DELTA_GEOMETRY.items():
            setattr(self, name, value)
        self.calls: list[str] = []
        self.fail = False
        self.double_call = False
        self.bad_query = False
        self.bad_query_dtype = False
        self.bad_state = False
        self.mutate_source_state = False
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = lambda *args, **kwargs: None
        self.chunk_gated_delta_rule = self._chunk
        self.recurrent_gated_delta_rule = self._recurrent

    def _state(self, initial_state: torch.Tensor | None) -> torch.Tensor:
        shape = (1, 16, 128, 127) if self.bad_state else adapter_module.STATE_SHAPE
        if initial_state is None:
            return torch.full(
                shape,
                float(self.layer_idx + 1),
                dtype=torch.float32,
            )
        if self.mutate_source_state:
            initial_state.data.add_(0.5)
        return initial_state + 1.0

    def _chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key, value, g, beta, chunk_size, output_final_state, use_qk_l2norm_in_kernel, kwargs
        self.calls.append("chunk")
        if self.fail:
            raise RuntimeError("fake kernel failure")
        return query, self._state(initial_state)

    def _recurrent(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key, value, g, beta, output_final_state, use_qk_l2norm_in_kernel
        self.calls.append("recurrent")
        if self.fail:
            raise RuntimeError("fake kernel failure")
        return query, self._state(initial_state)


class _FakeDecoderLayer:
    def __init__(self, index: int) -> None:
        if index in adapter_module.RECURRENT_LAYER_INDICES:
            self.linear_attn = Qwen3_5GatedDeltaNet(index)


class _FakeTextModel:
    def __init__(self) -> None:
        self.layers = [_FakeDecoderLayer(index) for index in range(24)]
        self.skip_layer: int | None = None
        self.force_recurrent_at_zero = False
        self.clone_initial_state = False
        self.return_different_cache = False
        self.sequence_increment = 1

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: _FakeDynamicCache,
        use_cache: bool,
    ) -> object:
        assert tuple(input_ids.shape) == (1, 1)
        assert tuple(position_ids.shape) == (1, 1)
        assert use_cache is True
        assert torch.is_inference_mode_enabled()
        position = int(position_ids.item())
        for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
            if layer_index == self.skip_layer:
                continue
            module = self.layers[layer_index].linear_attn
            query_shape = (1, 1, 16, 127) if module.bad_query else adapter_module.QUERY_SHAPE
            query_dtype = torch.float32 if module.bad_query_dtype else torch.bfloat16
            query = torch.full(query_shape, float(layer_index + position), dtype=query_dtype)
            key = torch.zeros_like(query)
            value = torch.zeros(adapter_module.QUERY_SHAPE, dtype=query_dtype)
            g = torch.zeros((1, 1, 16), dtype=torch.float32)
            beta = torch.zeros((1, 1, 16), dtype=torch.float32)
            cached = past_key_values.layers[layer_index].recurrent_states[0]
            kernel_initial = (
                cached.clone()
                if self.clone_initial_state and isinstance(cached, torch.Tensor)
                else cached
            )
            if position == 0 and not self.force_recurrent_at_zero:
                output = module.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                if module.double_call:
                    module.chunk_gated_delta_rule(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=None,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
            else:
                output = module.recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=kernel_initial,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            past_key_values.update_recurrent_state(output[1], layer_index)
        past_key_values.sequence_length += self.sequence_increment
        returned_cache = object() if self.return_different_cache else past_key_values
        return types.SimpleNamespace(past_key_values=returned_cache)


class _FakeLiveModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace()
        self.model = _FakeTextModel()
        self.training = False
        self._parameter = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.fail_after_fisher_forward = False

    def parameters(self):
        return iter((self._parameter,))

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: _FakeDynamicCache,
        use_cache: bool,
        logits_to_keep: int,
    ) -> object:
        assert tuple(input_ids.shape) == (1, 1)
        assert tuple(position_ids.shape) == (1, 1)
        assert use_cache is True
        assert logits_to_keep == 1
        assert torch.is_grad_enabled()
        position = int(position_ids.item())
        state_scalars: list[torch.Tensor] = []
        for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
            module = self.model.layers[layer_index].linear_attn
            query = torch.full(
                adapter_module.QUERY_SHAPE,
                float(layer_index + position),
                dtype=torch.bfloat16,
            )
            cached = past_key_values.layers[layer_index].recurrent_states[0]
            assert isinstance(cached, torch.Tensor)
            output = module.recurrent_gated_delta_rule(
                query,
                torch.zeros_like(query),
                torch.zeros_like(query),
                g=torch.zeros((1, 1, 16), dtype=torch.float32),
                beta=torch.zeros((1, 1, 16), dtype=torch.float32),
                initial_state=cached,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            past_key_values.update_recurrent_state(output[1], layer_index)
            state_scalars.append(output[1][0, 0, 0, 0])
        past_key_values.sequence_length += 1
        if self.fail_after_fisher_forward:
            raise RuntimeError("fake differentiable forward failure")
        score = torch.stack(state_scalars).sum() * 1e-4
        vocabulary_axis = torch.arange(32, dtype=torch.float32)
        logits = (score * vocabulary_axis).reshape(1, 1, -1)
        return types.SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _bind_fake_model(
    adapter: adapter_module.Experiment013Qwen35Adapter,
) -> tuple[_FakeLiveModel, adapter_module._Qwen35StepObserver]:
    model = _FakeLiveModel()
    runtime = adapter_module._TransformersRuntime(
        version=adapter_module.TRANSFORMERS_VERSION,
        qwen_config_class=object,
        qwen_model_class=object,
        qwen_gated_delta_net_class=Qwen3_5GatedDeltaNet,
        dynamic_cache_class=_FakeDynamicCache,
        torch_chunk_gated_delta_rule=lambda *args, **kwargs: None,
        torch_recurrent_gated_delta_rule=lambda *args, **kwargs: None,
        torch_causal_conv1d_update=lambda *args, **kwargs: None,
    )
    modules = adapter_module._qwen_modules(model, runtime)
    observer = adapter_module._Qwen35StepObserver(
        modules,
        query_device=torch.device("cpu"),
        _allow_test_non_cuda=True,
    )
    observer.install()
    adapter._runtime = runtime
    adapter._model = model
    adapter._model_device = torch.device("cpu")
    adapter._observer = observer
    return model, observer


def test_qwen_modules_require_exact_authenticated_class_and_geometry(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, _observer = _bind_fake_model(adapter)
    assert adapter._runtime is not None
    authenticated_runtime = adapter._runtime
    adapter.close_model(model)

    class SameNamedSubclass(Qwen3_5GatedDeltaNet):
        pass

    impostor_model = _FakeLiveModel()
    impostor_model.model.layers[0].linear_attn = SameNamedSubclass(0)
    with pytest.raises(adapter_module.Experiment013AdapterError, match="pinned Gated DeltaNet"):
        adapter_module._qwen_modules(impostor_model, authenticated_runtime)

    geometry_model = _FakeLiveModel()
    geometry_model.model.layers[0].linear_attn.num_k_heads = 8
    runtime = adapter_module._TransformersRuntime(
        version=adapter_module.TRANSFORMERS_VERSION,
        qwen_config_class=object,
        qwen_model_class=object,
        qwen_gated_delta_net_class=Qwen3_5GatedDeltaNet,
        dynamic_cache_class=_FakeDynamicCache,
        torch_chunk_gated_delta_rule=lambda *args, **kwargs: None,
        torch_recurrent_gated_delta_rule=lambda *args, **kwargs: None,
        torch_causal_conv1d_update=lambda *args, **kwargs: None,
    )
    with pytest.raises(adapter_module.Experiment013AdapterError, match="num_k_heads geometry"):
        adapter_module._qwen_modules(geometry_model, runtime)


def test_production_observer_rejects_a_non_cuda_query_contract(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    modules = observer.modules
    adapter.close_model(model)

    with pytest.raises(adapter_module.Experiment013AdapterError, match="must be CUDA"):
        adapter_module._Qwen35StepObserver(modules, query_device=torch.device("cpu"))


def _sequence_record(length: int = 2) -> dict[str, object]:
    return {
        "identity_record_sha256": _sha(77),
        "sequence_length": length,
    }


def test_one_token_chunk_then_recurrent_and_anchor_only_state(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record()
    adapter.begin_sequence(model, record)

    first = adapter.step_token(model, token_id=11, position=0, capture_state=False)
    second = adapter.step_token(model, token_id=12, position=1, capture_state=True)

    assert first.recurrence_query.shape == (18, 16, 128)
    assert first.recurrent_state is None
    assert second.recurrence_query.shape == (18, 16, 128)
    assert second.recurrent_state is not None
    assert second.recurrent_state.shape == (18, 16, 128, 128)
    assert second.recurrent_state.dtype == torch.float32
    assert first.layer_indices == adapter_module.RECURRENT_LAYER_INDICES
    assert first.successful_kernel_calls_per_layer == (1,) * 18
    for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
        assert model.model.layers[layer_index].linear_attn.calls == ["chunk", "recurrent"]
    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_h1_fisher_step_is_causal_functional_and_detached(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record(length=4)
    adapter.begin_sequence(model, record)
    adapter.step_token(model, token_id=5, position=0, capture_state=False)

    result = adapter.step_token_with_fisher(
        model,
        token_id=6,
        position=1,
        target_token_id=7,
        capture_state=True,
    )

    assert (result.boundary_position, result.input_position, result.target_position) == (0, 1, 2)
    assert (result.input_token_id, result.target_token_id) == (6, 7)
    assert result.step_observation.position == 1
    assert result.step_observation.token_id == 6
    assert result.step_observation.recurrence_query.shape == (18, 16, 128)
    assert result.step_observation.recurrent_state.shape == (18, 16, 128, 128)
    assert result.source_recurrent_state.shape == (18, 16, 128, 128)
    assert result.source_state_gradient.shape == (18, 16, 128, 128)
    assert result.source_recurrent_state.dtype == torch.float32
    assert result.source_state_gradient.dtype == torch.float32
    assert torch.count_nonzero(result.source_state_gradient).item() == 18
    assert result.target_nll > 0.0
    score = (result.source_recurrent_state[:, 0, 0, 0] + 1.0).sum() * 1e-4
    vocabulary_axis = torch.arange(32, dtype=torch.float32)
    probabilities = torch.softmax(score * vocabulary_axis, dim=0)
    analytic_gradient = float(
        ((probabilities * vocabulary_axis).sum() - result.target_token_id).item() * 1e-4
    )
    observed_gradient = float(result.source_state_gradient[0, 0, 0, 0].item())
    assert observed_gradient == pytest.approx(analytic_gradient, rel=2e-6, abs=1e-9)

    score_fp64 = float(score.item())
    axis_fp64 = torch.arange(32, dtype=torch.float64)

    def perturbed_nll(delta: float) -> float:
        logits = (score_fp64 + delta * 1e-4) * axis_fp64
        return float((torch.logsumexp(logits, dim=0) - logits[7]).item())

    epsilon = 1e-2
    finite_difference = (perturbed_nll(epsilon) - perturbed_nll(-epsilon)) / (2 * epsilon)
    assert observed_gradient == pytest.approx(finite_difference, rel=2e-5, abs=1e-9)
    assert adapter.runtime_metadata()["fisher_step_count"] == 1
    assert "update_recurrent_state" not in adapter._sequence.cache.__dict__  # type: ignore[union-attr]
    assert model._parameter.requires_grad is False
    assert model._parameter.grad is None
    for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
        cached = adapter._sequence.cache.layers[layer_index].recurrent_states[0]  # type: ignore[union-attr]
        assert isinstance(cached, torch.Tensor)
        assert cached.requires_grad is False
        assert cached.grad_fn is None

    continued = adapter.step_token(model, token_id=7, position=2, capture_state=True)
    assert continued.position == 2
    assert continued.recurrent_state is not None
    assert torch.equal(
        continued.recurrent_state[:, 0, 0, 0],
        result.step_observation.recurrent_state[:, 0, 0, 0] + 1.0,
    )
    assert adapter._sequence is not None
    assert adapter._sequence.next_position == 3
    assert adapter._sequence.cache.get_seq_length() == 3
    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_h1_fisher_failure_rolls_back_cache_and_invalidates_sequence(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record(length=3)
    adapter.begin_sequence(model, record)
    adapter.step_token(model, token_id=5, position=0, capture_state=False)
    assert adapter._sequence is not None
    before = {
        layer_index: adapter._sequence.cache.layers[layer_index].recurrent_states[0].clone()
        for layer_index in adapter_module.RECURRENT_LAYER_INDICES
    }
    model.fail_after_fisher_forward = True

    with pytest.raises(RuntimeError, match="fake differentiable forward failure"):
        adapter.step_token_with_fisher(
            model,
            token_id=6,
            position=1,
            target_token_id=7,
            capture_state=False,
        )

    assert adapter._sequence_failed is True
    assert adapter._sequence.cache.get_seq_length() == 1
    assert "update_recurrent_state" not in adapter._sequence.cache.__dict__
    for layer_index, expected in before.items():
        restored = adapter._sequence.cache.layers[layer_index].recurrent_states[0]
        assert torch.equal(restored, expected)
        assert restored.requires_grad is False
        assert restored.grad_fn is None
    assert observer.is_idle
    assert model._parameter.grad is None
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_h1_fisher_rejects_in_place_source_state_mutation_and_rolls_back(
    tmp_path: Path,
) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record(length=3)
    adapter.begin_sequence(model, record)
    adapter.step_token(model, token_id=5, position=0, capture_state=False)
    assert adapter._sequence is not None
    before = {
        layer_index: adapter._sequence.cache.layers[layer_index].recurrent_states[0].clone()
        for layer_index in adapter_module.RECURRENT_LAYER_INDICES
    }
    for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
        model.model.layers[layer_index].linear_attn.mutate_source_state = True

    with pytest.raises(
        adapter_module.Experiment013AdapterError,
        match="mutated source state",
    ):
        adapter.step_token_with_fisher(
            model,
            token_id=6,
            position=1,
            target_token_id=7,
            capture_state=False,
        )

    assert adapter._sequence_failed is True
    assert adapter._sequence.cache.get_seq_length() == 1
    for layer_index, expected in before.items():
        restored = adapter._sequence.cache.layers[layer_index].recurrent_states[0]
        assert torch.equal(restored, expected)
        assert restored.requires_grad is False
        assert restored.grad_fn is None
    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_h1_fisher_detach_failure_rolls_back_without_advancing_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record(length=3)
    adapter.begin_sequence(model, record)
    adapter.step_token(model, token_id=5, position=0, capture_state=False)
    assert adapter._sequence is not None
    before = {
        layer_index: adapter._sequence.cache.layers[layer_index].recurrent_states[0].clone()
        for layer_index in adapter_module.RECURRENT_LAYER_INDICES
    }

    def fail_detach(_cache: object) -> None:
        raise RuntimeError("synthetic detach failure")

    monkeypatch.setattr(adapter_module, "_detach_cache_tensors", fail_detach)
    with pytest.raises(RuntimeError, match="synthetic detach failure"):
        adapter.step_token_with_fisher(
            model,
            token_id=6,
            position=1,
            target_token_id=7,
            capture_state=False,
        )

    assert adapter._sequence_failed is True
    assert adapter._sequence.next_position == 1
    assert adapter._sequence.cache.get_seq_length() == 1
    assert adapter.runtime_metadata()["fisher_step_count"] == 0
    for layer_index, expected in before.items():
        restored = adapter._sequence.cache.layers[layer_index].recurrent_states[0]
        assert torch.equal(restored, expected)
    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_h1_fisher_requires_warm_boundary_and_future_target(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, _observer = _bind_fake_model(adapter)
    warm_record = _sequence_record(length=3)
    adapter.begin_sequence(model, warm_record)
    with pytest.raises(adapter_module.Experiment013AdapterError, match="warm S_b"):
        adapter.step_token_with_fisher(
            model,
            token_id=5,
            position=0,
            target_token_id=6,
            capture_state=False,
        )
    adapter.end_sequence(model, warm_record)

    short_record = _sequence_record(length=2)
    adapter.begin_sequence(model, short_record)
    adapter.step_token(model, token_id=5, position=0, capture_state=False)
    with pytest.raises(adapter_module.Experiment013AdapterError, match=r"x_\(b\+2\) target"):
        adapter.step_token_with_fisher(
            model,
            token_id=6,
            position=1,
            target_token_id=7,
            capture_state=False,
        )
    adapter.end_sequence(model, short_record)
    adapter.close_model(model)


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "missing",
        "bad_query",
        "bad_query_dtype",
        "bad_state",
        "wrong_kernel",
    ],
)
def test_one_call_receipts_and_shapes_fail_closed(tmp_path: Path, failure: str) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    if failure == "duplicate":
        model.model.layers[0].linear_attn.double_call = True
    elif failure == "missing":
        model.model.skip_layer = 22
    elif failure == "bad_query":
        model.model.layers[0].linear_attn.bad_query = True
    elif failure == "bad_query_dtype":
        model.model.layers[0].linear_attn.bad_query_dtype = True
    elif failure == "bad_state":
        model.model.layers[0].linear_attn.bad_state = True
    else:
        model.model.force_recurrent_at_zero = True
    record = _sequence_record(length=1)
    adapter.begin_sequence(model, record)

    with pytest.raises(adapter_module.Experiment013AdapterError):
        adapter.step_token(model, token_id=1, position=0, capture_state=False)

    assert observer.is_idle
    with pytest.raises(adapter_module.Experiment013AdapterError, match="healthy"):
        adapter.step_token(model, token_id=1, position=0, capture_state=False)
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_recurrent_step_requires_the_exact_cached_state_object(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    record = _sequence_record()
    adapter.begin_sequence(model, record)
    adapter.step_token(model, token_id=1, position=0, capture_state=False)
    model.model.clone_initial_state = True

    with pytest.raises(adapter_module.Experiment013AdapterError, match="exact persistent state"):
        adapter.step_token(model, token_id=2, position=1, capture_state=False)

    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


@pytest.mark.parametrize("failure", ["cache_identity", "sequence_length"])
def test_dynamic_cache_identity_and_length_are_proven(tmp_path: Path, failure: str) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    if failure == "cache_identity":
        model.model.return_different_cache = True
        expected = "different DynamicCache"
    else:
        model.model.sequence_increment = 2
        expected = "exactly one token"
    record = _sequence_record(length=1)
    adapter.begin_sequence(model, record)

    with pytest.raises(adapter_module.Experiment013AdapterError, match=expected):
        adapter.step_token(model, token_id=1, position=0, capture_state=False)

    assert observer.is_idle
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_kernel_exception_resets_context_and_invalidates_sequence(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    model.model.layers[0].linear_attn.fail = True
    record = _sequence_record(length=1)
    adapter.begin_sequence(model, record)

    with pytest.raises(RuntimeError, match="fake kernel failure"):
        adapter.step_token(model, token_id=1, position=0, capture_state=False)

    assert observer.is_idle
    assert adapter._sequence_failed is True
    adapter.end_sequence(model, record)
    adapter.close_model(model)


def test_failed_kernel_never_appends_a_receipt(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    module = model.model.layers[0].linear_attn
    module.fail = True
    cache = _FakeDynamicCache(config=model.config)
    capture = adapter_module._StepCapture(cache=cache, position=0, receipts={})
    token = observer.activate(capture)
    query = torch.zeros(adapter_module.QUERY_SHAPE, dtype=torch.bfloat16)
    try:
        with pytest.raises(RuntimeError, match="fake kernel failure"):
            module.chunk_gated_delta_rule(
                query,
                query,
                query,
                g=torch.zeros((1, 1, 16)),
                beta=torch.zeros((1, 1, 16)),
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        assert capture.receipts == {}
    finally:
        observer.deactivate(token)
        adapter.close_model(model)


def test_post_forward_cache_equality_is_required(tmp_path: Path) -> None:
    adapter = adapter_module.create_adapter(_context(tmp_path))
    model, observer = _bind_fake_model(adapter)
    _FakeDynamicCache.corrupt_layer = 5
    record = _sequence_record(length=1)
    adapter.begin_sequence(model, record)
    try:
        with pytest.raises(adapter_module.Experiment013AdapterError, match="state differs"):
            adapter.step_token(model, token_id=1, position=0, capture_state=True)
        assert observer.is_idle
    finally:
        _FakeDynamicCache.corrupt_layer = None
        adapter.end_sequence(model, record)
        adapter.close_model(model)


class _FakeDeviceValue:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.requires_grad = True
        self.dtype = torch.bfloat16

    @staticmethod
    def is_floating_point() -> bool:
        return True


class _LoaderModel:
    def __init__(self, config: object, events: list[tuple[str, object]]) -> None:
        self.config = config
        self.model = types.SimpleNamespace(layers=[_FakeDecoderLayer(index) for index in range(24)])
        self.training = True
        self._parameter = _FakeDeviceValue()
        self._events = events

    def to(self, device: torch.device) -> _LoaderModel:
        self._events.append(("to", device))
        self._parameter.device = device
        return self

    def eval(self) -> _LoaderModel:
        self._events.append(("eval", None))
        self.training = False
        return self

    def requires_grad_(self, enabled: bool) -> _LoaderModel:
        self._events.append(("requires_grad", enabled))
        self._parameter.requires_grad = enabled
        return self

    def parameters(self):
        return iter((self._parameter,))

    def buffers(self):
        return iter(())


def _qwen_config() -> object:
    return types.SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=1024,
        num_hidden_layers=24,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=list(adapter_module.LAYER_TYPES),
        _attn_implementation="eager",
    )


def _model_identity(model_root: Path) -> AuthenticatedModelFiles:
    names = sorted(
        (
            "config.json",
            "model.safetensors-00001-of-00001.safetensors",
            "model.safetensors.index.json",
        )
    )
    files = tuple(
        ModelFileIdentity(
            name=name,
            size_bytes=1,
            sha256=_sha(index + 10) if name.endswith(".safetensors") else None,
            git_blob_oid="1" * 40,
            lfs_sha256=_sha(index + 10) if name.endswith(".safetensors") else None,
            lfs_size_bytes=1 if name.endswith(".safetensors") else None,
        )
        for index, name in enumerate(names)
    )
    return AuthenticatedModelFiles(
        model_root=model_root,
        model_id=adapter_module.MODEL_ID,
        revision=adapter_module.MODEL_REVISION,
        transformers_version=adapter_module.TRANSFORMERS_VERSION,
        files=files,
        hub_tree_manifest_sha256=_sha(88),
        manifest_file_sha256=_sha(89),
    )


def _loader_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_factory: object,
) -> tuple[
    adapter_module.Experiment013Qwen35Adapter,
    AuthenticatedModelFiles,
    list[tuple[str, object]],
]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    context = _context(tmp_path)
    context = AdapterConstructionContext(
        repository_root=context.repository_root,
        model_root=model_root,
        cache_root=context.cache_root,
        ruler_root=context.ruler_root,
        runtime_authentication_context=context.runtime_authentication_context,
        execution_binding_artifacts=context.execution_binding_artifacts,
    )
    events: list[tuple[str, object]] = []
    config = _qwen_config()

    class FakeConfigClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            events.append(("config", (path, kwargs)))
            return config

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            events.append(("model", (path, kwargs)))
            model = _LoaderModel(config, events)
            if not callable(result_factory):
                raise AssertionError("result_factory must be callable")
            return result_factory(model)

    runtime = adapter_module._TransformersRuntime(
        version=adapter_module.TRANSFORMERS_VERSION,
        qwen_config_class=FakeConfigClass,
        qwen_model_class=FakeModelClass,
        qwen_gated_delta_net_class=Qwen3_5GatedDeltaNet,
        dynamic_cache_class=_FakeDynamicCache,
        torch_chunk_gated_delta_rule=lambda *args, **kwargs: None,
        torch_recurrent_gated_delta_rule=lambda *args, **kwargs: None,
        torch_causal_conv1d_update=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(adapter_module, "_load_transformers_runtime", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return (
        adapter_module.create_adapter(context),
        _model_identity(model_root),
        events,
    )


@pytest.mark.parametrize(
    ("diagnostic_name", "diagnostic_value"),
    [
        ("missing_keys", {"model.layers.0.weight"}),
        ("unexpected_keys", {"unknown.weight"}),
        ("mismatched_keys", {("model.weight", (1,), (2,))}),
        ("error_msgs", ["checkpoint load failed"]),
    ],
)
def test_loader_rejects_each_non_empty_transformers_diagnostic_before_cuda_transfer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    diagnostic_name: str,
    diagnostic_value: object,
) -> None:
    def result(model: object) -> object:
        diagnostics: dict[str, object] = {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }
        diagnostics[diagnostic_name] = diagnostic_value
        return model, diagnostics

    adapter, identity, events = _loader_adapter(monkeypatch, tmp_path, result)

    with pytest.raises(adapter_module.Experiment013AdapterError, match=diagnostic_name):
        adapter.load_model(identity)

    assert [name for name, _value in events] == ["config", "model"]
    assert adapter.runtime_metadata()["model_loaded"] is False
    assert adapter.runtime_metadata()["model_loading_diagnostic_counts"] is None


@pytest.mark.parametrize(
    "result_factory",
    [
        lambda model: model,
        lambda model: [model, {}],
        lambda model: (model, {}, None),
    ],
)
def test_loader_requires_exact_transformers_model_loading_tuple(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_factory: object,
) -> None:
    adapter, identity, events = _loader_adapter(monkeypatch, tmp_path, result_factory)

    with pytest.raises(adapter_module.Experiment013AdapterError, match="exactly"):
        adapter.load_model(identity)

    assert [name for name, _value in events] == ["config", "model"]


@pytest.mark.parametrize(
    "loading_info",
    [
        {
            "missing_keys": [],
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        },
        {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
        },
        {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
            "conversion_errors": {},
        },
    ],
)
def test_loader_requires_exact_transformers_loading_diagnostic_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loading_info: dict[str, object],
) -> None:
    adapter, identity, events = _loader_adapter(
        monkeypatch,
        tmp_path,
        lambda model: (model, loading_info),
    )

    with pytest.raises(adapter_module.Experiment013AdapterError, match="diagnostic"):
        adapter.load_model(identity)

    assert [name for name, _value in events] == ["config", "model"]


def test_loader_is_local_bf16_safetensors_eager_and_freezes_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    context = _context(tmp_path)
    context = AdapterConstructionContext(
        repository_root=context.repository_root,
        model_root=model_root,
        cache_root=context.cache_root,
        ruler_root=context.ruler_root,
        runtime_authentication_context=context.runtime_authentication_context,
        execution_binding_artifacts=context.execution_binding_artifacts,
    )
    events: list[tuple[str, object]] = []
    config = _qwen_config()

    class FakeConfigClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            events.append(("config", (path, kwargs)))
            return config

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            events.append(("model", (path, kwargs)))
            return (
                _LoaderModel(config, events),
                {
                    "missing_keys": set(),
                    "unexpected_keys": set(),
                    "mismatched_keys": set(),
                    "error_msgs": [],
                },
            )

    def torch_chunk(*args: object, **kwargs: object):
        del args, kwargs

    def torch_recurrent(*args: object, **kwargs: object):
        del args, kwargs

    def torch_conv(*args: object, **kwargs: object):
        del args, kwargs

    runtime = adapter_module._TransformersRuntime(
        version=adapter_module.TRANSFORMERS_VERSION,
        qwen_config_class=FakeConfigClass,
        qwen_model_class=FakeModelClass,
        qwen_gated_delta_net_class=Qwen3_5GatedDeltaNet,
        dynamic_cache_class=_FakeDynamicCache,
        torch_chunk_gated_delta_rule=torch_chunk,
        torch_recurrent_gated_delta_rule=torch_recurrent,
        torch_causal_conv1d_update=torch_conv,
    )
    monkeypatch.setattr(adapter_module, "_load_transformers_runtime", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    adapter = adapter_module.create_adapter(context)

    model = adapter.load_model(_model_identity(model_root))

    assert [name for name, _value in events] == [
        "config",
        "model",
        "to",
        "eval",
        "requires_grad",
    ]
    config_call = events[0][1]
    assert isinstance(config_call, tuple)
    assert config_call[1] == {"local_files_only": True, "trust_remote_code": False}
    model_call = events[1][1]
    assert isinstance(model_call, tuple)
    assert model_call[1] == {
        "config": config,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "weights_only": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "output_loading_info": True,
    }
    for layer_index in adapter_module.RECURRENT_LAYER_INDICES:
        module = model.model.layers[layer_index].linear_attn
        assert module.causal_conv1d_fn is None
        assert module.causal_conv1d_update is torch_conv
        # The state functions are wrapped after the exact fallbacks are frozen.
        assert module.chunk_gated_delta_rule is not torch_chunk
        assert module.recurrent_gated_delta_rule is not torch_recurrent
    assert adapter.runtime_metadata()["model_loaded"] is True
    assert adapter.runtime_metadata()["device"] == "cuda:0"
    assert adapter.runtime_metadata()["kernel_backend"] == (
        "transformers_pure_torch_gated_delta_rule"
    )
    assert adapter.runtime_metadata()["model_loading_diagnostic_counts"] == {
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_keys": 0,
        "error_msgs": 0,
    }
    adapter.close_model(model)
    assert adapter.runtime_metadata()["model_loaded"] is False


def test_source_binding_hashes_the_adapter_own_bytes() -> None:
    binding = adapter_module.Experiment013Qwen35Adapter.source_binding()
    expected = hashlib.sha256(Path(adapter_module.__file__).read_bytes()).hexdigest()

    assert binding == {
        "path": "src/recurquant/experiment013_qwen35_adapter.py",
        "raw_sha256": expected,
    }
