"""Transactional Qwen3.5 runtime for StateLease equal-byte comparators.

The equal-byte comparators choose precision globally across every recurrent
layer.  A layer-local Transformers cache therefore cannot safely replace its
state as soon as that layer returns.  This module keeps the previous complete
packed checkpoint during one outer model forward, materializes it only as
measured transient workspace, stages every successful FP32 final state and
causal query, and atomically packs one new global checkpoint after the complete
model (including full-attention layers and the LM head) succeeds.

The runtime is intentionally correctness-first.  The public comparator packers
allocate additional transient selection workspaces; no fused-kernel, latency,
or allocator-peak claim is made here.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import transformers
from transformers import DynamicCache, Qwen3_5ForCausalLM
from transformers.cache_utils import LinearAttentionLayer

from .statelease_equal_byte_baselines import (
    EXPANDED_RHT_Q4_Q8,
    FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    RHT_Q4_Q6_Q8,
    RHT_RESIDUAL_Q4,
    EqualByteCheckpoint,
    EqualByteCodecEvidence,
    EqualByteCodecName,
    EqualByteLayout,
    pack_expanded_rht_q4_q8,
    pack_rht_q4_q6_q8,
    pack_rht_residual_q4,
    update_causal_query_ema,
)
from .statelease_observer import Qwen35StateLeaseObserver

PINNED_TRANSFORMERS_VERSION = "5.14.1"


def _require_pinned_transformers() -> None:
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "the equal-byte Qwen3.5 runtime is pinned to transformers "
            f"{PINNED_TRANSFORMERS_VERSION}; found {transformers.__version__}"
        )


def _storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _unique_storage_bytes(tensors: Iterator[torch.Tensor]) -> int:
    owners: set[tuple[str, int | None, int]] = set()
    total = 0
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        key = (
            tensor.device.type,
            tensor.device.index,
            tensor.untyped_storage().data_ptr(),
        )
        if key in owners:
            continue
        owners.add(key)
        total += _storage_bytes(tensor)
    return total


def _clone_attribute(value: object) -> object:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return value


@dataclass(frozen=True, slots=True)
class EqualByteCacheUpdateEvidence:
    """Evidence for one successful outer-forward checkpoint transaction."""

    update_index: int
    codec: str
    token_count: int
    layer_indices: tuple[int, ...]
    query_input_dtypes: tuple[str, ...]
    previous_checkpoint_present: bool
    raw_state_workspace_peak_bytes: int
    query_workspace_peak_bytes: int
    logical_fp32_state_bytes: int
    resident_bytes: int
    selection_sha256: str
    selected_units: int
    mean_squared_error: float
    relative_l2_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PendingObservation:
    query: torch.Tensor
    final_state: torch.Tensor
    token_count: int


@dataclass(frozen=True, slots=True)
class _MutableTensorSnapshot:
    tensor: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class _LayerSnapshot:
    attributes: dict[str, object]
    mutable_tensors: tuple[_MutableTensorSnapshot, ...]


@dataclass(slots=True)
class _EqualByteForwardTransaction:
    cache_identity: int
    layer_snapshots: tuple[_LayerSnapshot, ...]
    checkpoint: EqualByteCheckpoint | None
    update_evidence: tuple[EqualByteCacheUpdateEvidence, ...]
    update_count: int
    last_evidence: EqualByteCacheUpdateEvidence | None
    successful_tokens: int
    state_workspace_peak: int
    query_workspace_peak: int
    active: bool = True


class _EqualByteStateView(MutableMapping[int, torch.Tensor | None]):
    """Read-only Transformers mapping backed by forward-scoped materialization."""

    def __init__(self, owner: EqualByteQwen35Cache, layer_index: int) -> None:
        self._owner = owner
        self._layer_index = layer_index

    def __getitem__(self, key: int) -> torch.Tensor | None:
        if key != 0:
            raise IndexError(f"state_idx {key} is outside [0, 1)")
        return self._owner._forward_recurrent_state(self._layer_index)

    def __setitem__(self, key: int, value: torch.Tensor | None) -> None:
        del key, value
        raise RuntimeError(
            "equal-byte recurrent states are committed globally; use the cache "
            "update_recurrent_state receipt"
        )

    def __delitem__(self, key: int) -> None:
        del key
        raise RuntimeError("equal-byte recurrent states cannot be deleted layer-locally")

    def __iter__(self) -> Iterator[int]:
        return iter((0,))

    def __len__(self) -> int:
        return 1


class EqualByteLinearAttentionLayer(LinearAttentionLayer):
    """Linear-attention cache layer with no persistent raw recurrent tensor."""

    is_compileable = False

    def __init__(self, owner: EqualByteQwen35Cache, *, layer_index: int) -> None:
        super().__init__(number_of_states=1)
        self.owner = owner
        self.layer_index = layer_index
        self.recurrent_states = _EqualByteStateView(owner, layer_index)

    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
    ) -> None:
        if state_idx != 0:
            raise IndexError(f"state_idx {state_idx} is outside [0, 1)")
        if recurrent_states is not None:
            raise RuntimeError("equal-byte recurrent states cannot be initialized layer-locally")
        if conv_states is not None:
            super().lazy_initialization(
                conv_states=conv_states,
                state_idx=state_idx,
                conv_kernel_size=conv_kernel_size,
            )

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        return self.owner._accept_recurrent_state(
            recurrent_states,
            layer_idx=self.layer_index,
            state_idx=state_idx,
        )

    def reset(self) -> None:
        if self.is_conv_states_initialized[0]:
            conv_state = self.conv_states[0]
            assert isinstance(conv_state, torch.Tensor)
            conv_state.zero_()
        self.has_previous_state[0] = False
        self.is_recurrent_states_initialized[0] = False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if (
            not isinstance(beam_idx, torch.Tensor)
            or beam_idx.dtype != torch.long
            or beam_idx.ndim != 1
            or beam_idx.numel() != 1
            or int(beam_idx.item()) != 0
        ):
            raise ValueError(
                "equal-byte recurrent caches are batch-one and support only identity beam_idx=[0]"
            )
        if self.is_conv_states_initialized[0]:
            conv_state = self.conv_states[0]
            assert isinstance(conv_state, torch.Tensor)
            self.conv_states[0] = conv_state.index_select(
                0,
                beam_idx.to(conv_state.device),
            )

    def activate_past_recording(self) -> None:
        raise RuntimeError(
            "equal-byte recurrent caches do not support speculative past "
            "recording or recurrent-state rollback"
        )

    def crop(self, tokens_to_remove: int) -> None:
        del tokens_to_remove
        raise RuntimeError(
            "equal-byte recurrent caches cannot crop or roll back a globally "
            "packed recurrent checkpoint"
        )

    def offload(self) -> None:
        if self.is_conv_states_initialized[0]:
            conv_state = self.conv_states[0]
            assert isinstance(conv_state, torch.Tensor)
            self.conv_states[0] = conv_state.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        if self.device is None or not self.is_conv_states_initialized[0]:
            return
        conv_state = self.conv_states[0]
        assert isinstance(conv_state, torch.Tensor)
        target = torch.device(self.device)
        if conv_state.device != target:
            self.conv_states[0] = conv_state.to(target, non_blocking=True)


class EqualByteQwen35Cache(DynamicCache):
    """Batch-one, globally packed no-replay cache for pinned Qwen3.5.

    Every model call using this cache must execute inside
    :class:`Qwen35EqualByteObserver`.  Direct cache writes fail closed because
    only the root-model transaction can provide atomic rollback across
    convolution state, full-attention KV, recurrent state, and the LM head.
    """

    is_compileable = False

    def __init__(
        self,
        config: object,
        *,
        codec: EqualByteCodecName,
        layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
        record_evidence: bool = False,
    ) -> None:
        _require_pinned_transformers()
        if codec not in {
            EXPANDED_RHT_Q4_Q8,
            RHT_Q4_Q6_Q8,
            RHT_RESIDUAL_Q4,
        }:
            raise ValueError(f"unsupported equal-byte codec: {codec!r}")
        if not isinstance(layout, EqualByteLayout):
            raise TypeError("layout must be an EqualByteLayout")
        if not isinstance(record_evidence, bool):
            raise TypeError("record_evidence must be a bool")

        text_config = (
            config.get_text_config(decoder=True)
            if callable(getattr(config, "get_text_config", None))
            else config
        )
        if text_config.__class__.__name__ != "Qwen3_5TextConfig":
            raise TypeError("config must be a pinned Qwen3_5TextConfig or parent config")
        geometry = {
            "heads": getattr(text_config, "linear_num_value_heads", None),
            "key_rows": getattr(text_config, "linear_key_head_dim", None),
            "value_width": getattr(text_config, "linear_value_head_dim", None),
        }
        expected_geometry = {
            "heads": layout.heads,
            "key_rows": layout.key_rows,
            "value_width": layout.value_width,
        }
        if geometry != expected_geometry:
            raise ValueError(
                f"Qwen3.5 recurrent geometry {geometry} does not match layout {expected_geometry}"
            )

        super().__init__(config=config)
        recurrent_indices = tuple(
            index
            for index, layer in enumerate(self.layers)
            if isinstance(layer, LinearAttentionLayer)
        )
        if recurrent_indices != layout.layer_indices:
            raise ValueError(
                "layout.layer_indices must exactly match the Qwen3.5 recurrent "
                f"layers; got {layout.layer_indices}, expected {recurrent_indices}"
            )
        for index in recurrent_indices:
            if type(self.layers[index]) is not LinearAttentionLayer:
                raise TypeError(
                    "equal-byte runtime supports Qwen3.5's plain LinearAttentionLayer only"
                )
            self.layers[index] = EqualByteLinearAttentionLayer(
                self,
                layer_index=index,
            )

        self.codec = codec
        self.layout = layout
        self.record_evidence = record_evidence
        self.checkpoint: EqualByteCheckpoint | None = None
        self.update_evidence: list[EqualByteCacheUpdateEvidence] = []
        self.last_evidence: EqualByteCacheUpdateEvidence | None = None
        self.update_count = 0
        self.successful_tokens = 0
        self.raw_state_workspace_peak_bytes = 0
        self.query_workspace_peak_bytes = 0
        self._forward_state_workspace_peak_bytes = 0
        self._forward_query_workspace_peak_bytes = 0

        self._active_equal_byte_transaction: _EqualByteForwardTransaction | None = None
        self._previous_states: dict[int, torch.Tensor] = {}
        self._pending_observations: dict[int, _PendingObservation] = {}
        self._final_states: dict[int, torch.Tensor] = {}
        self._queries: dict[int, torch.Tensor] = {}
        self._receipt_order: list[int] = []

    def equal_byte_layers(
        self,
    ) -> Iterator[tuple[int, EqualByteLinearAttentionLayer]]:
        for layer_index in self.layout.layer_indices:
            layer = self.layers[layer_index]
            if not isinstance(layer, EqualByteLinearAttentionLayer):
                raise RuntimeError(f"cache layer {layer_index} lost its equal-byte layer identity")
            yield layer_index, layer

    @staticmethod
    def _snapshot_layer(layer: object) -> _LayerSnapshot:
        source = getattr(layer, "__dict__", None)
        if not isinstance(source, dict):
            raise TypeError("cache layers must expose mutable instance attributes")
        attributes = {name: _clone_attribute(value) for name, value in source.items()}
        mutable: list[_MutableTensorSnapshot] = []
        for name in ("conv_states", "recurrent_states"):
            value = attributes.get(name)
            candidates: tuple[object, ...]
            if isinstance(value, dict):
                candidates = tuple(value.values())
            elif isinstance(value, (list, tuple)):
                candidates = tuple(value)
            else:
                candidates = ()
            for candidate in candidates:
                if isinstance(candidate, torch.Tensor):
                    mutable.append(
                        _MutableTensorSnapshot(
                            tensor=candidate,
                            value=candidate.detach().clone(memory_format=torch.preserve_format),
                        )
                    )
        return _LayerSnapshot(attributes, tuple(mutable))

    def _require_active_transaction(
        self,
        transaction: object | None = None,
    ) -> _EqualByteForwardTransaction:
        active = self._active_equal_byte_transaction
        if active is None:
            raise RuntimeError(
                "equal-byte cache access requires an active root-model observer transaction"
            )
        if transaction is not None:
            if not isinstance(transaction, _EqualByteForwardTransaction):
                raise TypeError("transaction is not an equal-byte forward transaction")
            if transaction.cache_identity != id(self):
                raise ValueError("equal-byte transaction belongs to another cache")
            if active is not transaction:
                raise RuntimeError("equal-byte transaction is not active")
        if not active.active:
            raise RuntimeError("equal-byte transaction is not active")
        return active

    def _clear_forward_workspace(self) -> None:
        self._previous_states.clear()
        self._pending_observations.clear()
        self._final_states.clear()
        self._queries.clear()
        self._receipt_order.clear()

    def _current_state_workspace_bytes(self) -> int:
        return _unique_storage_bytes(
            iter((*self._previous_states.values(), *self._final_states.values()))
        )

    def _current_query_workspace_bytes(self) -> int:
        pending = (item.query for item in self._pending_observations.values())
        return _unique_storage_bytes(iter((*self._queries.values(), *pending)))

    def _observe_workspace_peak(self) -> None:
        state_bytes = self._current_state_workspace_bytes()
        query_bytes = self._current_query_workspace_bytes()
        self._forward_state_workspace_peak_bytes = max(
            self._forward_state_workspace_peak_bytes,
            state_bytes,
        )
        self._forward_query_workspace_peak_bytes = max(
            self._forward_query_workspace_peak_bytes,
            query_bytes,
        )
        self.raw_state_workspace_peak_bytes = max(
            self.raw_state_workspace_peak_bytes,
            state_bytes,
        )
        self.query_workspace_peak_bytes = max(
            self.query_workspace_peak_bytes,
            query_bytes,
        )

    def _validate_materialized_states(
        self,
        states: dict[int, torch.Tensor],
    ) -> None:
        if set(states) != set(self.layout.layer_indices):
            raise RuntimeError("checkpoint materialization omitted a recurrent layer")
        expected = (
            1,
            self.layout.heads,
            self.layout.key_rows,
            self.layout.value_width,
        )
        for layer_index, state in states.items():
            if state.dtype != torch.float32 or tuple(state.shape) != expected:
                raise RuntimeError(f"layer {layer_index} materialized state is not FP32 {expected}")
            if not torch.isfinite(state).all().item():
                raise RuntimeError(
                    f"layer {layer_index} materialized state contains non-finite values"
                )

    def begin_statelease_forward_transaction(self) -> _EqualByteForwardTransaction:
        """Begin one all-layer, all-cache-state model-forward transaction."""

        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("an equal-byte model-forward transaction is already active")
        if (
            self._pending_observations
            or self._final_states
            or self._queries
            or self._previous_states
            or self._receipt_order
        ):
            raise RuntimeError("equal-byte cache has stale transient forward workspace")

        previous_states = {} if self.checkpoint is None else self.checkpoint.materialize()
        if previous_states:
            self._validate_materialized_states(previous_states)
        transaction = _EqualByteForwardTransaction(
            cache_identity=id(self),
            layer_snapshots=tuple(self._snapshot_layer(layer) for layer in self.layers),
            checkpoint=self.checkpoint,
            update_evidence=tuple(self.update_evidence),
            update_count=self.update_count,
            last_evidence=self.last_evidence,
            successful_tokens=self.successful_tokens,
            state_workspace_peak=self.raw_state_workspace_peak_bytes,
            query_workspace_peak=self.query_workspace_peak_bytes,
        )
        self._active_equal_byte_transaction = transaction
        self._forward_state_workspace_peak_bytes = 0
        self._forward_query_workspace_peak_bytes = 0
        self._previous_states = previous_states
        self._observe_workspace_peak()
        return transaction

    def _forward_recurrent_state(self, layer_index: int) -> torch.Tensor | None:
        self._require_active_transaction()
        if self.checkpoint is None:
            return None
        try:
            return self._previous_states[layer_index]
        except KeyError as error:
            raise RuntimeError(
                f"layer {layer_index} recurrent state was read more than once or after its receipt"
            ) from error

    @staticmethod
    def _validate_kernel_tensor(
        value: object,
        *,
        name: str,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.device.type == "meta":
            raise ValueError(f"{name} must be materialized")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def stage_statelease_observation(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        final_state: torch.Tensor,
    ) -> None:
        """Stage one successful pinned Qwen3.5 kernel transition."""

        self._require_active_transaction()
        if layer_index not in self.layout.layer_indices:
            raise ValueError(f"layer {layer_index} is not a frozen recurrent layer")
        if layer_index in self._pending_observations or layer_index in self._final_states:
            self._pending_observations.pop(layer_index, None)
            raise RuntimeError(f"layer {layer_index} produced a duplicate equal-byte observation")

        if not isinstance(query, torch.Tensor) or query.ndim != 4:
            raise ValueError("query must have rank four")
        tokens = int(query.shape[1])
        if tokens <= 0:
            raise ValueError("query token count must be positive")
        qk_shape = (1, tokens, self.layout.heads, self.layout.key_rows)
        value_shape = (1, tokens, self.layout.heads, self.layout.value_width)
        gate_shape = (1, tokens, self.layout.heads)
        query = self._validate_kernel_tensor(query, name="query", shape=qk_shape)
        key = self._validate_kernel_tensor(key, name="key", shape=qk_shape)
        value = self._validate_kernel_tensor(value, name="value", shape=value_shape)
        log_decay = self._validate_kernel_tensor(
            log_decay,
            name="log_decay",
            shape=gate_shape,
        )
        beta = self._validate_kernel_tensor(beta, name="beta", shape=gate_shape)
        state_shape = (
            1,
            self.layout.heads,
            self.layout.key_rows,
            self.layout.value_width,
        )
        final_state = self._validate_kernel_tensor(
            final_state,
            name="final_state",
            shape=state_shape,
        )
        if final_state.dtype != torch.float32:
            raise TypeError("equal-byte recurrent final_state must use torch.float32")
        devices = {tensor.device for tensor in (query, key, value, log_decay, beta, final_state)}
        if len(devices) != 1:
            raise ValueError("all observed kernel tensors must share one device")
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value, log_decay, beta, final_state)
        ):
            raise RuntimeError("equal-byte recurrent caching is inference-only")

        expected_initial = self._previous_states.get(layer_index)
        if self.checkpoint is None:
            if initial_state is not None:
                raise RuntimeError(
                    "first equal-byte transition unexpectedly received an initial state"
                )
        elif initial_state is not expected_initial:
            raise RuntimeError(
                f"layer {layer_index} kernel did not consume the transaction's "
                "materialized checkpoint state"
            )

        query_snapshot = query.detach().clone(memory_format=torch.contiguous_format)
        self._pending_observations[layer_index] = _PendingObservation(
            query=query_snapshot,
            final_state=final_state,
            token_count=tokens,
        )
        self._observe_workspace_peak()

    def has_pending_statelease_observation(self, layer_index: int) -> bool:
        return layer_index in self._pending_observations

    def discard_pending_statelease_observation(self, layer_index: int) -> None:
        self._pending_observations.pop(layer_index, None)

    def _accept_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        *,
        layer_idx: int,
        state_idx: int,
    ) -> torch.Tensor:
        self._require_active_transaction()
        if state_idx != 0:
            raise IndexError(f"state_idx {state_idx} is outside [0, 1)")
        observation = self._pending_observations.get(layer_idx)
        if observation is None:
            raise RuntimeError(f"layer {layer_idx} recurrent write has no staged kernel receipt")
        if recurrent_states is not observation.final_state:
            self._pending_observations.pop(layer_idx, None)
            raise RuntimeError(
                f"layer {layer_idx} recurrent write does not match the staged "
                "kernel final-state identity"
            )
        expected_position = len(self._receipt_order)
        if (
            expected_position >= len(self.layout.layer_indices)
            or self.layout.layer_indices[expected_position] != layer_idx
        ):
            self._pending_observations.pop(layer_idx, None)
            raise RuntimeError(
                "equal-byte recurrent receipts did not follow frozen model-layer order"
            )

        self._pending_observations.pop(layer_idx)
        self._queries[layer_idx] = observation.query
        self._final_states[layer_idx] = recurrent_states.detach()
        self._previous_states.pop(layer_idx, None)
        self._receipt_order.append(layer_idx)
        self._observe_workspace_peak()
        return recurrent_states

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        if layer_idx not in self.layout.layer_indices:
            raise ValueError(f"layer {layer_idx} is not an equal-byte recurrent layer")
        layer = self.layers[layer_idx]
        if not isinstance(layer, EqualByteLinearAttentionLayer):
            raise RuntimeError(f"layer {layer_idx} has the wrong cache-layer type")
        return layer.update_recurrent_state(recurrent_states, state_idx=state_idx)

    def _pack_candidate(
        self,
        states: dict[int, torch.Tensor],
        query_ema: torch.Tensor,
    ) -> EqualByteCheckpoint:
        if self.codec == EXPANDED_RHT_Q4_Q8:
            return pack_expanded_rht_q4_q8(
                states,
                query_ema,
                layout=self.layout,
            )
        if self.codec == RHT_Q4_Q6_Q8:
            return pack_rht_q4_q6_q8(
                states,
                query_ema,
                layout=self.layout,
            )
        return pack_rht_residual_q4(
            states,
            query_ema,
            layout=self.layout,
        )

    def commit_statelease_forward_transaction(self, transaction: object) -> None:
        """Pack and atomically install one global checkpoint after model success."""

        active = self._require_active_transaction(transaction)
        if self._pending_observations:
            raise RuntimeError(
                "model forward returned with unconsumed equal-byte observations "
                f"on layers {sorted(self._pending_observations)}"
            )
        if tuple(self._receipt_order) != self.layout.layer_indices:
            raise RuntimeError(
                "model forward did not produce exactly one recurrent receipt for every frozen layer"
            )
        if set(self._final_states) != set(self.layout.layer_indices):
            raise RuntimeError("global recurrent-state staging is incomplete")
        if set(self._queries) != set(self.layout.layer_indices):
            raise RuntimeError("global causal-query staging is incomplete")
        token_counts = {int(query.shape[1]) for query in self._queries.values()}
        if len(token_counts) != 1:
            raise RuntimeError("recurrent layers observed inconsistent token counts")
        token_count = next(iter(token_counts))

        previous_ema = None if active.checkpoint is None else active.checkpoint.query_energy_ema
        candidate_ema = update_causal_query_ema(
            previous_ema,
            self._queries,
            layout=self.layout,
        )
        candidate = self._pack_candidate(self._final_states, candidate_ema)
        candidate.validate()
        if candidate.resident_bytes != self.layout.expected_resident_bytes:
            raise RuntimeError("candidate checkpoint violates the exact byte budget")
        codec_evidence: EqualByteCodecEvidence = candidate.evidence
        evidence = EqualByteCacheUpdateEvidence(
            update_index=self.update_count,
            codec=self.codec,
            token_count=token_count,
            layer_indices=tuple(self._receipt_order),
            query_input_dtypes=tuple(
                str(self._queries[index].dtype) for index in self.layout.layer_indices
            ),
            previous_checkpoint_present=active.checkpoint is not None,
            raw_state_workspace_peak_bytes=self._forward_state_workspace_peak_bytes,
            query_workspace_peak_bytes=self._forward_query_workspace_peak_bytes,
            logical_fp32_state_bytes=self.layout.fp32_state_bytes,
            resident_bytes=candidate.resident_bytes,
            selection_sha256=codec_evidence.selection_sha256,
            selected_units=codec_evidence.selected_units,
            mean_squared_error=codec_evidence.mean_squared_error,
            relative_l2_error=codec_evidence.relative_l2_error,
            max_absolute_error=codec_evidence.max_absolute_error,
        )

        # All fallible validation and packing is complete.  References are
        # replaced only here, so rollback can restore the old checkpoint.
        self.checkpoint = candidate
        self.update_count += 1
        self.successful_tokens += token_count
        self.last_evidence = evidence
        if self.record_evidence:
            self.update_evidence.append(evidence)
        for _, layer in self.equal_byte_layers():
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True

        self._clear_forward_workspace()
        active.active = False
        self._active_equal_byte_transaction = None

    def rollback_statelease_forward_transaction(self, transaction: object) -> None:
        """Restore convolution, attention KV, packed state, and all evidence."""

        active = self._require_active_transaction(transaction)
        if len(active.layer_snapshots) != len(self.layers):
            raise RuntimeError("cache geometry changed during the model forward")
        try:
            with torch.no_grad():
                for layer, snapshot in zip(
                    self.layers,
                    active.layer_snapshots,
                    strict=True,
                ):
                    for mutable in snapshot.mutable_tensors:
                        mutable.tensor.copy_(mutable.value)
                    attributes = getattr(layer, "__dict__", None)
                    if not isinstance(attributes, dict):
                        raise TypeError("cache layers must expose mutable instance attributes")
                    attributes.clear()
                    attributes.update(snapshot.attributes)
            self.checkpoint = active.checkpoint
            self.update_evidence = list(active.update_evidence)
            self.update_count = active.update_count
            self.last_evidence = active.last_evidence
            self.successful_tokens = active.successful_tokens
            self.raw_state_workspace_peak_bytes = active.state_workspace_peak
            self.query_workspace_peak_bytes = active.query_workspace_peak
        finally:
            self._clear_forward_workspace()
            self._forward_state_workspace_peak_bytes = 0
            self._forward_query_workspace_peak_bytes = 0
            active.active = False
            self._active_equal_byte_transaction = None

    def materialize_recurrent_state(self, layer_index: int) -> torch.Tensor | None:
        """Return a fresh diagnostic materialization without retaining it."""

        if layer_index not in self.layout.layer_indices:
            raise ValueError(f"layer {layer_index} is not a frozen recurrent layer")
        if self.checkpoint is None:
            return None
        return self.checkpoint.materialize()[layer_index]

    def persistent_recurrent_tensors(
        self,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        if self.checkpoint is None:
            return ()
        return self.checkpoint.persistent_tensors()

    def persistent_raw_state_bytes(self) -> int:
        """Return zero after verifying no layer owns a raw recurrent tensor."""

        for layer_index, layer in self.equal_byte_layers():
            if not isinstance(layer.recurrent_states, _EqualByteStateView):
                raise RuntimeError(
                    f"layer {layer_index} unexpectedly owns a recurrent-state mapping"
                )
        return 0

    def storage_summary(self) -> dict[str, int | float | bool | str]:
        resident = 0 if self.checkpoint is None else self.checkpoint.resident_bytes
        return {
            "codec": self.codec,
            "checkpoint_present": self.checkpoint is not None,
            "resident_bytes": resident,
            "expected_resident_bytes": self.layout.expected_resident_bytes,
            "persistent_raw_state_bytes": self.persistent_raw_state_bytes(),
            "full_precision_equivalent_bytes": self.layout.fp32_state_bytes,
            "effective_bits_per_state_element": (
                8.0 * resident / self.layout.state_elements if resident else 0.0
            ),
            "raw_state_workspace_current_bytes": self._current_state_workspace_bytes(),
            "raw_state_workspace_peak_bytes": self.raw_state_workspace_peak_bytes,
            "query_workspace_current_bytes": self._current_query_workspace_bytes(),
            "query_workspace_peak_bytes": self.query_workspace_peak_bytes,
            "forward_transaction_active": (self._active_equal_byte_transaction is not None),
            "update_count": self.update_count,
            "successful_tokens": self.successful_tokens,
        }

    def reset(self) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot reset during an equal-byte model forward")
        super().reset()
        self.checkpoint = None
        self.update_evidence.clear()
        self.last_evidence = None
        self.update_count = 0
        self.successful_tokens = 0
        self.raw_state_workspace_peak_bytes = 0
        self.query_workspace_peak_bytes = 0
        self._forward_state_workspace_peak_bytes = 0
        self._forward_query_workspace_peak_bytes = 0
        self._clear_forward_workspace()

    @staticmethod
    def _validate_identity_indices(indices: torch.Tensor, *, name: str) -> None:
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or indices.ndim != 1
            or indices.numel() != 1
            or int(indices.item()) != 0
        ):
            raise ValueError(f"equal-byte cache is batch-one and supports only identity {name}=[0]")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot reorder during an equal-byte model forward")
        self._validate_identity_indices(beam_idx, name="beam_idx")
        super().reorder_cache(beam_idx)

    def activate_past_recording(self) -> None:
        """Fail before speculative decoding can create an untracked checkpoint."""

        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot activate past recording during a model forward")
        raise RuntimeError(
            "equal-byte caches do not support speculative past recording; "
            "crop cannot restore the globally packed recurrent checkpoint"
        )

    def crop(self, max_length: int) -> None:
        """Fail closed because recurrent checkpoints have no token-history journal."""

        del max_length
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot crop during an equal-byte model forward")
        raise RuntimeError(
            "equal-byte caches cannot crop or roll back a globally packed "
            "recurrent checkpoint"
        )

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1:
            raise ValueError("equal-byte cache is batch-one and cannot repeat batches")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._validate_identity_indices(indices, name="indices")
        self.reorder_cache(indices)

    def offload(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot offload during an equal-byte model forward")
        if layer_idx in self.layout.layer_indices:
            if self.checkpoint is not None:
                self.checkpoint = self.checkpoint.to("cpu")
            layer = self.layers[layer_idx]
            assert isinstance(layer, EqualByteLinearAttentionLayer)
            layer.offload()
            return
        super().offload(layer_idx, only_non_sliding=only_non_sliding)

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot prefetch during an equal-byte model forward")
        if layer_idx in self.layout.layer_indices:
            layer = self.layers[layer_idx]
            assert isinstance(layer, EqualByteLinearAttentionLayer)
            layer.prefetch()
            if self.checkpoint is not None and layer.device is not None:
                self.checkpoint = self.checkpoint.to(torch.device(layer.device))
            return
        layer = self.layers[layer_idx]
        if not (only_non_sliding and getattr(layer, "is_sliding", False)):
            layer.prefetch()

    def offload_all(self) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot offload during an equal-byte model forward")
        if self.checkpoint is not None:
            self.checkpoint = self.checkpoint.to("cpu")
        for layer in self.layers:
            layer.offload()

    def prefetch_all(self) -> None:
        if self._active_equal_byte_transaction is not None:
            raise RuntimeError("cannot prefetch during an equal-byte model forward")
        target: torch.device | None = None
        for _, layer in self.equal_byte_layers():
            layer.prefetch()
            if layer.device is not None:
                candidate = torch.device(layer.device)
                if target is None:
                    target = candidate
                elif candidate != target:
                    raise RuntimeError(
                        "global equal-byte checkpoints require recurrent layers on one device"
                    )
        if self.checkpoint is not None and target is not None:
            self.checkpoint = self.checkpoint.to(target)
        for layer_index, layer in enumerate(self.layers):
            if layer_index not in self.layout.layer_indices:
                layer.prefetch()


class Qwen35EqualByteObserver(Qwen35StateLeaseObserver):
    """Pinned root-model observer for :class:`EqualByteQwen35Cache`.

    This deliberately reuses the audited identity-dispatch Qwen3.5 kernel
    instrumentation.  The equal-byte cache implements its transaction contract
    at the outer model boundary, so failures after any recurrent layer,
    full-attention layer, or LM-head operation are rolled back together.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        caches: list[object] | tuple[object, ...],
    ) -> None:
        _require_pinned_transformers()
        if not isinstance(model, Qwen3_5ForCausalLM):
            raise TypeError(
                "Qwen35EqualByteObserver must wrap the outer "
                "Qwen3_5ForCausalLM so LM-head failures remain transactional"
            )
        if any(not isinstance(cache, EqualByteQwen35Cache) for cache in caches):
            raise TypeError("Qwen35EqualByteObserver accepts only EqualByteQwen35Cache objects")
        super().__init__(model, caches=caches)


def create_qwen35_equal_byte_cache(
    model_or_config: object,
    *,
    codec: EqualByteCodecName,
    layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    record_evidence: bool = False,
) -> EqualByteQwen35Cache:
    """Create a pinned Qwen3.5 equal-byte cache from a model or config."""

    _require_pinned_transformers()
    config: Any
    if isinstance(model_or_config, torch.nn.Module):
        if model_or_config.__class__.__name__ != "Qwen3_5ForCausalLM":
            raise TypeError("model must be a Qwen3_5ForCausalLM")
        config = getattr(model_or_config, "config", None)
    else:
        config = model_or_config
    if config is None:
        raise TypeError("model_or_config must expose a Qwen3.5 config")
    return EqualByteQwen35Cache(
        config,
        codec=codec,
        layout=layout,
        record_evidence=record_evidence,
    )
