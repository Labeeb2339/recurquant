"""Correctness-first StateLease cache for Qwen3.5 Gated DeltaNet states.

StateLease keeps one physical RHT-CQER INT4/INT8 checkpoint and a bounded
five-record replay lease.  The replay records are the already-normalized key
``k``, update ``u``, and log decay ``g`` from the successful recurrent
transition.  Keys and updates are resident in BF16; log decays and the causal
query-energy EMA are resident in FP32.

At the fifth one-token decode transition, the cache compares only the frozen
four- and five-token boundaries:

* cut 4 independently packs raw ``S4`` and replays the retained fifth record;
* cut 5 independently packs raw ``S5`` and clears the replay lease.

Both risks are measured against the same raw ``S5`` with the same causal query
EMA.  An exact tie chooses cut 5.  This is a local representation decision; it
does not retroactively repair trajectory error inherited from older
quantization.

The implementation materializes FP32 selection and replay workspaces and makes
no fused-kernel, latency, or allocator-peak claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, MutableMapping
from dataclasses import asdict, dataclass, replace

import torch
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionLayer

from .mixed_quantization import PackedMixedQuantizedTensor, quantize_pack_mixed
from .packed_cache import (
    _QUERY_EMA_DECAY,
    _QUERY_L2NORM_EPS,
    _stable_top_mask,
    _validate_exact_budget_plan,
)
from .quantization import (
    QuantizationSpec,
    RoundingMode,
    quantize_dequantize,
    scheduled_quantization_spec,
)
from .rht import RHT_SEED, right_rht_encode
from .row_policy import ExactBudgetRowPlan
from .statelease import (
    STATELEASE_L2NORM_EPS,
    normalize_gated_delta_key,
    query_weighted_row_mse,
    replay_gated_delta_updates,
    select_statelease_boundary,
)

STATELEASE_REPLAY_CAPACITY = 5
STATELEASE_TRANSITION_ATOL = 2e-5
STATELEASE_TRANSITION_RTOL = 2e-5
STATELEASE_SELECTION_METHOD = "statelease_cut4_cut5_right_rht_query_ema32_weighted_mse_fisher_quota"
STATELEASE_GENERIC_SELECTION_METHOD = (
    "statelease_cut4_cut5_right_rht_query_ema32_weighted_mse_configurable_quota"
)


def _tensor_bytes(value: torch.Tensor | None) -> int:
    """Return allocated backing-storage bytes, never just logical view bytes."""

    return 0 if value is None else int(value.untyped_storage().nbytes())


def _validate_owned_persistent_tensors(
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Require compact, independently owned storage for persistent tensors."""

    owners: dict[int, str] = {}
    for name, tensor in tensors:
        logical_bytes = tensor.numel() * tensor.element_size()
        storage_bytes = _tensor_bytes(tensor)
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} persistent storage must be contiguous")
        if tensor.storage_offset() != 0:
            raise RuntimeError(f"{name} persistent storage must have zero storage offset")
        if storage_bytes != logical_bytes:
            raise RuntimeError(
                f"{name} persistent storage owns {storage_bytes} bytes but exposes "
                f"{logical_bytes} logical bytes"
            )
        if tensor.numel() == 0:
            continue
        pointer = tensor.untyped_storage().data_ptr()
        previous = owners.get(pointer)
        if previous is not None:
            raise RuntimeError(f"{name} persistent storage aliases {previous}")
        owners[pointer] = name


def _mask_sha256(packed: PackedMixedQuantizedTensor | None) -> str | None:
    if packed is None:
        return None
    return hashlib.sha256(
        bytes(packed.precision_mask.detach().cpu().contiguous().tolist())
    ).hexdigest()


def _validate_float_tensor(
    value: object,
    *,
    name: str,
    rank: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


@dataclass(frozen=True, slots=True)
class StateLeaseUpdateEvidence:
    """One successfully committed StateLease state write."""

    update_index: int
    layer_index: int
    state_index: int
    token_count: int
    action: str
    boundary: int | None
    tie: bool
    cut4_risk: float | None
    cut5_risk: float | None
    replay_valid_count: int
    replay_capacity: int
    low_bits: int
    high_bits: int
    group_size: int
    scale_bits: int
    rounding: str
    source_dtype: str
    shape: tuple[int, ...]
    high_precision_groups: int
    high_precision_mask_sha256: str
    checkpoint_bytes: int
    query_ema_bytes: int
    replay_capacity_bytes: int
    resident_bytes: int
    relative_l2_error: float
    mean_squared_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PendingStateLeaseObservation:
    update_index: int
    token_count: int
    candidate_ema: torch.Tensor
    initial_state: torch.Tensor | None
    final_state: torch.Tensor
    normalized_key: torch.Tensor | None
    update: torch.Tensor | None
    log_decay: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _MutableTensorSnapshot:
    tensor: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class _LayerForwardSnapshot:
    attributes: dict[str, object]
    mutable_tensors: tuple[_MutableTensorSnapshot, ...]


@dataclass(slots=True)
class _StateLeaseForwardTransaction:
    cache_identity: int
    layer_snapshots: tuple[_LayerForwardSnapshot, ...]
    update_evidence: tuple[StateLeaseUpdateEvidence, ...]
    update_index: int
    active: bool = True


class _StateLeaseStateView(MutableMapping[int, torch.Tensor | None]):
    """Transformers-compatible view that reconstructs a leased state on read."""

    def __init__(self, owner: StateLeaseLinearAttentionLayer) -> None:
        self._owner = owner

    def __getitem__(self, key: int) -> torch.Tensor | None:
        return self._owner.materialize_recurrent_state(key)

    def __setitem__(self, key: int, value: torch.Tensor | None) -> None:
        if key != 0:
            self._owner.discard_pending_statelease_observation()
            raise IndexError(f"layer {self._owner.layer_index} state_idx {key} is outside [0, 1)")
        if value is None:
            self._owner.clear_recurrent_state()
            return
        self._owner.update_recurrent_state(value, state_idx=key)

    def __delitem__(self, key: int) -> None:
        self.__setitem__(key, None)

    def __iter__(self) -> Iterator[int]:
        return iter((0,))

    def __len__(self) -> int:
        return 1


class StateLeaseLinearAttentionLayer(LinearAttentionLayer):
    """One batch-one StateLease layer with a fixed five-record replay capacity."""

    is_compileable = False
    selection_method = STATELEASE_GENERIC_SELECTION_METHOD
    state_codec = "right_rht_sha256_signs_v1"
    state_codec_seed = RHT_SEED
    replay_capacity = STATELEASE_REPLAY_CAPACITY
    query_ema_decay = _QUERY_EMA_DECAY
    query_l2norm_eps = _QUERY_L2NORM_EPS

    def __init__(
        self,
        *,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
        layer_index: int,
        expected_heads: int,
        expected_rows: int,
        high_precision_group_indices: tuple[int, ...],
        number_of_states: int = 1,
        selection_method: str = STATELEASE_GENERIC_SELECTION_METHOD,
    ) -> None:
        if number_of_states != 1:
            raise ValueError("StateLease currently requires exactly one recurrent state")
        super().__init__(number_of_states=number_of_states)
        if low_spec.bits != 4 or high_spec.bits != 8:
            raise ValueError("StateLease requires low INT4 and high INT8 specs")
        if any(
            getattr(low_spec, field) != getattr(high_spec, field)
            for field in (
                "group_size",
                "scale_bits",
                "flatten_last_dims",
                "rounding",
                "seed",
                "epsilon",
            )
        ):
            raise ValueError("StateLease specs must differ only in bit width")
        if low_spec.flatten_last_dims != 2:
            raise ValueError("StateLease requires flatten_last_dims=2")
        if expected_heads <= 0 or expected_rows <= 0:
            raise ValueError("StateLease layer geometry must be positive")
        if not isinstance(selection_method, str) or not selection_method:
            raise ValueError("selection_method must be a non-empty string")
        total_groups = expected_heads * expected_rows
        if len(set(high_precision_group_indices)) != len(high_precision_group_indices):
            raise ValueError("high-precision group indices must be unique")
        if any(
            group_index < 0 or group_index >= total_groups
            for group_index in high_precision_group_indices
        ):
            raise ValueError("high-precision group index is outside layer geometry")

        self.low_spec = low_spec
        self.high_spec = high_spec
        self.layer_index = layer_index
        self.expected_heads = expected_heads
        self.expected_rows = expected_rows
        self.high_precision_group_indices = tuple(sorted(high_precision_group_indices))
        self.selection_method = selection_method
        self.packed_states: dict[int, PackedMixedQuantizedTensor | None] = {0: None}
        self.recurrent_states = _StateLeaseStateView(self)

        self.query_energy_ema: torch.Tensor | None = None
        self.normalized_key_buffer: torch.Tensor | None = None
        self.update_buffer: torch.Tensor | None = None
        self.log_decay_buffer: torch.Tensor | None = None
        self.valid_count: torch.Tensor | None = None
        self._pending_statelease_observation: _PendingStateLeaseObservation | None = None
        self._update_count = 0
        self.observations_staged = 0
        self.observations_committed = 0
        self.tokens_observed = 0
        self.checkpoint_count = 0
        self.boundary4_count = 0
        self.boundary5_count = 0
        self.tie_count = 0
        self.last_action: str | None = None
        self.last_reason: str | None = None
        self.last_boundary: int | None = None
        self.last_cut4_risk: float | None = None
        self.last_cut5_risk: float | None = None
        self.last_update_evidence: StateLeaseUpdateEvidence | None = None

    @property
    def packed_checkpoint(self) -> PackedMixedQuantizedTensor | None:
        return self.packed_states[0]

    def _selected_specs(self) -> tuple[QuantizationSpec, QuantizationSpec]:
        return (
            scheduled_quantization_spec(
                self.low_spec,
                layer_index=self.layer_index,
                layer_update_index=self._update_count,
            ),
            scheduled_quantization_spec(
                self.high_spec,
                layer_index=self.layer_index,
                layer_update_index=self._update_count,
            ),
        )

    def _expected_state_shape(self) -> tuple[int, int, int, int]:
        return (
            1,
            self.expected_heads,
            self.expected_rows,
            self.low_spec.group_size,
        )

    def _validate_state(self, state: object, *, name: str) -> torch.Tensor:
        tensor = _validate_float_tensor(state, name=name, rank=4)
        expected = self._expected_state_shape()
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"layer {self.layer_index} {name} must have shape {expected}; "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"layer {self.layer_index} {name} must use torch.float32 because "
                "the pinned Qwen3.5 state kernel accumulates recurrent state in FP32"
            )
        return tensor

    def _buffer_valid_count(self) -> int:
        if self.valid_count is None:
            return 0
        if self.valid_count.dtype != torch.int32 or tuple(self.valid_count.shape) != (1,):
            raise RuntimeError("StateLease valid-count storage is malformed")
        count = int(self.valid_count.item())
        if not 0 <= count <= self.replay_capacity:
            raise RuntimeError("StateLease valid count is outside replay capacity")
        return count

    def _validate_buffer_storage(self, *, device: torch.device) -> None:
        tensors = (
            self.normalized_key_buffer,
            self.update_buffer,
            self.log_decay_buffer,
            self.valid_count,
        )
        if all(tensor is None for tensor in tensors):
            return
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("StateLease replay-buffer storage is incomplete")
        assert self.normalized_key_buffer is not None
        assert self.update_buffer is not None
        assert self.log_decay_buffer is not None
        assert self.valid_count is not None
        expected_key = (
            self.replay_capacity,
            self.expected_heads,
            self.expected_rows,
        )
        expected_update = (
            self.replay_capacity,
            self.expected_heads,
            self.low_spec.group_size,
        )
        expected_decay = (self.replay_capacity, self.expected_heads)
        if (
            tuple(self.normalized_key_buffer.shape) != expected_key
            or self.normalized_key_buffer.dtype != torch.bfloat16
            or tuple(self.update_buffer.shape) != expected_update
            or self.update_buffer.dtype != torch.bfloat16
            or tuple(self.log_decay_buffer.shape) != expected_decay
            or self.log_decay_buffer.dtype != torch.float32
            or tuple(self.valid_count.shape) != (1,)
            or self.valid_count.dtype != torch.int32
        ):
            raise RuntimeError("StateLease replay-buffer geometry or dtype is malformed")
        if any(tensor.device != device for tensor in tensors if tensor is not None):
            raise RuntimeError("StateLease replay buffers and state use different devices")
        _validate_owned_persistent_tensors(
            (
                ("normalized_key_buffer", self.normalized_key_buffer),
                ("update_buffer", self.update_buffer),
                ("log_decay_buffer", self.log_decay_buffer),
                ("valid_count", self.valid_count),
            )
        )
        self._buffer_valid_count()

    def _candidate_buffers(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_buffer_storage(device=device)
        if self.normalized_key_buffer is None:
            return (
                torch.zeros(
                    (
                        self.replay_capacity,
                        self.expected_heads,
                        self.expected_rows,
                    ),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.zeros(
                    (
                        self.replay_capacity,
                        self.expected_heads,
                        self.low_spec.group_size,
                    ),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.zeros(
                    (self.replay_capacity, self.expected_heads),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.zeros((1,), dtype=torch.int32, device=device),
            )
        assert self.update_buffer is not None
        assert self.log_decay_buffer is not None
        assert self.valid_count is not None
        return (
            self.normalized_key_buffer.clone(),
            self.update_buffer.clone(),
            self.log_decay_buffer.clone(),
            self.valid_count.clone(),
        )

    def _candidate_query_ema(self, query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            source = query.detach().to(torch.float32)
            squared = source.square()
            energy = squared / (squared.sum(dim=-1, keepdim=True) + self.query_l2norm_eps)
            energy = energy.squeeze(0)
            token_count = energy.shape[0]
            previous = self.query_energy_ema
            if previous is None:
                previous = torch.full(
                    (self.expected_heads, self.expected_rows),
                    1.0 / self.expected_rows,
                    dtype=torch.float32,
                    device=query.device,
                )
            exponents = torch.arange(
                token_count - 1,
                -1,
                -1,
                dtype=torch.float32,
                device=query.device,
            )
            weights = torch.pow(
                torch.tensor(
                    self.query_ema_decay,
                    dtype=torch.float32,
                    device=query.device,
                ),
                exponents,
            )
            candidate = (self.query_ema_decay**token_count) * previous + (
                1.0 - self.query_ema_decay
            ) * (energy * weights[:, None, None]).sum(dim=0)
            if not torch.isfinite(candidate).all().item():
                raise RuntimeError("StateLease query EMA produced non-finite values")
        return candidate

    @staticmethod
    def _normalized_risk_energy(query_ema: torch.Tensor) -> torch.Tensor:
        """Return the simplex view required by ``query_weighted_row_mse``.

        The persistent EMA deliberately remains bit-for-bit compatible with
        CQER's ``q^2 / (sum(q^2) + 1e-6)`` recurrence.  That epsilon makes its
        per-head sum slightly smaller than one, so risk evaluation uses one
        shared per-head normalized view for both frozen boundary candidates.
        """

        row_sums = query_ema.sum(dim=-1, keepdim=True)
        if (row_sums <= 0).any().item() or not torch.isfinite(row_sums).all().item():
            raise RuntimeError("StateLease query EMA has invalid per-head mass")
        normalized = query_ema / row_sums
        if not torch.isfinite(normalized).all().item():
            raise RuntimeError("StateLease normalized risk energy is non-finite")
        return normalized

    def _derive_single_token_record(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        final_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            normalized_key = normalize_gated_delta_key(
                key,
                l2norm_eps=STATELEASE_L2NORM_EPS,
            )
            key_token = normalized_key[:, 0]
            decay_token = log_decay.detach().to(torch.float32)[:, 0]
            decayed = initial_state.detach().to(torch.float32) * decay_token.exp().unsqueeze(
                -1
            ).unsqueeze(-1)
            remembered = (decayed * key_token.unsqueeze(-1)).sum(dim=-2)
            update = (value.detach().to(torch.float32)[:, 0] - remembered) * beta.detach().to(
                torch.float32
            )[:, 0].unsqueeze(-1)
            reconstructed = decayed + key_token.unsqueeze(-1) * update.unsqueeze(-2)
            reference = final_state.detach().to(torch.float32)
            if not torch.allclose(
                reconstructed,
                reference,
                rtol=STATELEASE_TRANSITION_RTOL,
                atol=STATELEASE_TRANSITION_ATOL,
            ):
                maximum = float((reconstructed - reference).abs().max().item())
                raise ValueError(
                    "single-token StateLease transition does not reproduce the "
                    f"successful kernel final state; max_abs_error={maximum:.8g}"
                )
        return (
            key_token.squeeze(0),
            update.squeeze(0),
            decay_token.squeeze(0),
        )

    def stage_statelease_observation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        final_state: torch.Tensor,
    ) -> None:
        """Stage exact kernel inputs for the next recurrent-state write."""

        if self._pending_statelease_observation is not None:
            self.discard_pending_statelease_observation()
            raise RuntimeError(
                f"layer {self.layer_index} received a duplicate StateLease "
                "observation; the pending observation was discarded"
            )
        query = _validate_float_tensor(query, name="query", rank=4)
        key = _validate_float_tensor(key, name="key", rank=4)
        value = _validate_float_tensor(value, name="value", rank=4)
        log_decay = _validate_float_tensor(log_decay, name="log_decay", rank=3)
        beta = _validate_float_tensor(beta, name="beta", rank=3)
        final_state = self._validate_state(final_state, name="final_state")
        token_count = query.shape[1]
        expected_query = (
            1,
            token_count,
            self.expected_heads,
            self.expected_rows,
        )
        expected_value = (
            1,
            token_count,
            self.expected_heads,
            self.low_spec.group_size,
        )
        expected_gate = (1, token_count, self.expected_heads)
        if token_count <= 0 or tuple(query.shape) != expected_query:
            raise ValueError(
                f"query must have shape [1, tokens, {self.expected_heads}, "
                f"{self.expected_rows}]; got {tuple(query.shape)}"
            )
        if tuple(key.shape) != expected_query:
            raise ValueError(f"key must have shape {expected_query}; got {tuple(key.shape)}")
        if tuple(value.shape) != expected_value:
            raise ValueError(f"value must have shape {expected_value}; got {tuple(value.shape)}")
        if tuple(log_decay.shape) != expected_gate:
            raise ValueError(
                f"log_decay must have shape {expected_gate}; got {tuple(log_decay.shape)}"
            )
        if tuple(beta.shape) != expected_gate:
            raise ValueError(f"beta must have shape {expected_gate}; got {tuple(beta.shape)}")
        devices = {
            query.device,
            key.device,
            value.device,
            log_decay.device,
            beta.device,
            final_state.device,
        }
        if initial_state is not None:
            initial_state = self._validate_state(
                initial_state,
                name="initial_state",
            )
            devices.add(initial_state.device)
        if len(devices) != 1:
            raise ValueError("StateLease observation tensors must share one device")
        if (log_decay > 0).any().item():
            raise ValueError("Qwen3.5 log_decay must be non-positive")
        if ((beta < 0) | (beta > 1)).any().item():
            raise ValueError("Qwen3.5 beta must lie in [0, 1]")
        if self.device is not None and torch.device(self.device) != query.device:
            raise ValueError(
                f"layer {self.layer_index} observation and recurrent state must use the same device"
            )
        if self.query_energy_ema is not None and self.query_energy_ema.device != query.device:
            raise ValueError("StateLease query EMA and observation use different devices")
        if initial_state is not None:
            current = self.materialize_recurrent_state(0)
            if current is None:
                raise ValueError("StateLease received an initial state before a checkpoint existed")
            if not torch.equal(
                current.detach().to(torch.float32),
                initial_state.detach().to(torch.float32),
            ):
                raise ValueError("StateLease initial state does not match the current leased state")

        candidate_ema = self._candidate_query_ema(query)
        normalized_key = update = stored_decay = None
        frozen_initial = None
        if initial_state is not None:
            frozen_initial = initial_state.detach().clone()
        if token_count == 1 and initial_state is not None:
            normalized_key, update, stored_decay = self._derive_single_token_record(
                key=key,
                value=value,
                log_decay=log_decay,
                beta=beta,
                initial_state=initial_state,
                final_state=final_state,
            )
        self._pending_statelease_observation = _PendingStateLeaseObservation(
            update_index=self._update_count,
            token_count=token_count,
            candidate_ema=candidate_ema,
            initial_state=frozen_initial,
            final_state=final_state,
            normalized_key=normalized_key,
            update=update,
            log_decay=stored_decay,
        )

    def discard_pending_statelease_observation(self) -> None:
        """Idempotently discard one staged kernel observation."""

        self._pending_statelease_observation = None

    def _pack_checkpoint(
        self,
        source: torch.Tensor,
        *,
        query_ema: torch.Tensor,
        low_spec: QuantizationSpec,
        high_spec: QuantizationSpec,
    ) -> tuple[PackedMixedQuantizedTensor, torch.Tensor]:
        encoded = right_rht_encode(
            source,
            layer_index=self.layer_index,
            expected_heads=self.expected_heads,
            output_dtype=torch.float32,
        )
        low = quantize_dequantize(encoded, low_spec).tensor.to(torch.float32)
        high = quantize_dequantize(encoded, high_spec).tensor.to(torch.float32)
        benefit = (low - encoded).square().mean(dim=-1) - (high - encoded).square().mean(dim=-1)
        benefit = benefit.reshape(self.expected_heads, self.expected_rows)
        scores = query_ema * benefit
        if not torch.isfinite(scores).all().item():
            raise RuntimeError("StateLease checkpoint scores must be finite")
        quota = len(self.high_precision_group_indices)
        mask = _stable_top_mask(scores, quota).reshape(
            self.expected_heads,
            self.expected_rows,
        )
        packed = quantize_pack_mixed(
            source,
            mask,
            low_spec=low_spec,
            high_spec=high_spec,
            right_rht_layer_index=self.layer_index,
            right_rht_expected_heads=self.expected_heads,
        )
        _validate_owned_persistent_tensors(
            (
                ("packed.low_payload", packed.low_payload),
                ("packed.high_payload", packed.high_payload),
                ("packed.scales", packed.scales),
                ("packed.precision_mask", packed.precision_mask),
            )
        )
        materialized = packed.dequantize()
        if (
            tuple(materialized.shape) != self._expected_state_shape()
            or not torch.isfinite(materialized).all().item()
        ):
            raise RuntimeError("StateLease checkpoint materialization is invalid")
        if packed.high_precision_groups != quota:
            raise RuntimeError("StateLease checkpoint did not preserve its exact row quota")
        return packed, materialized

    @staticmethod
    def _clear_candidate_buffers(
        key_buffer: torch.Tensor,
        update_buffer: torch.Tensor,
        decay_buffer: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> None:
        key_buffer.zero_()
        update_buffer.zero_()
        decay_buffer.zero_()
        valid_count.zero_()

    @staticmethod
    def _append_candidate_record(
        *,
        key_buffer: torch.Tensor,
        update_buffer: torch.Tensor,
        decay_buffer: torch.Tensor,
        valid_count: torch.Tensor,
        slot: int,
        normalized_key: torch.Tensor,
        update: torch.Tensor,
        log_decay: torch.Tensor,
    ) -> None:
        key_buffer[slot].copy_(normalized_key.to(torch.bfloat16))
        update_buffer[slot].copy_(update.to(torch.bfloat16))
        decay_buffer[slot].copy_(log_decay.to(torch.float32))
        valid_count.fill_(slot + 1)

    def _materialize_candidate(
        self,
        *,
        packed: PackedMixedQuantizedTensor,
        key_buffer: torch.Tensor,
        update_buffer: torch.Tensor,
        decay_buffer: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> torch.Tensor:
        checkpoint = packed.dequantize()
        count = int(valid_count.item())
        if count == 0:
            return checkpoint
        return replay_gated_delta_updates(
            checkpoint,
            key_buffer[:count].unsqueeze(0),
            update_buffer[:count].unsqueeze(0),
            decay_buffer[:count].unsqueeze(0),
        )

    def _build_evidence(
        self,
        *,
        final_state: torch.Tensor,
        materialized: torch.Tensor,
        packed: PackedMixedQuantizedTensor,
        pending: _PendingStateLeaseObservation,
        action: str,
        boundary: int | None,
        tie: bool,
        cut4_risk: float | None,
        cut5_risk: float | None,
        replay_valid_count: int,
        query_ema: torch.Tensor,
        key_buffer: torch.Tensor,
        update_buffer: torch.Tensor,
        decay_buffer: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> StateLeaseUpdateEvidence:
        _validate_owned_persistent_tensors(
            (
                ("packed.low_payload", packed.low_payload),
                ("packed.high_payload", packed.high_payload),
                ("packed.scales", packed.scales),
                ("packed.precision_mask", packed.precision_mask),
                ("query_energy_ema", query_ema),
                ("normalized_key_buffer", key_buffer),
                ("update_buffer", update_buffer),
                ("log_decay_buffer", decay_buffer),
                ("valid_count", valid_count),
            )
        )
        error = materialized.detach().to(torch.float32) - final_state.detach().to(torch.float32)
        source_norm = torch.linalg.vector_norm(final_state.detach().to(torch.float32))
        relative_l2 = torch.linalg.vector_norm(error) / source_norm.clamp_min(1e-12)
        replay_bytes = sum(
            _tensor_bytes(tensor)
            for tensor in (
                key_buffer,
                update_buffer,
                decay_buffer,
                valid_count,
            )
        )
        return StateLeaseUpdateEvidence(
            update_index=self._update_count,
            layer_index=self.layer_index,
            state_index=0,
            token_count=pending.token_count,
            action=action,
            boundary=boundary,
            tie=tie,
            cut4_risk=cut4_risk,
            cut5_risk=cut5_risk,
            replay_valid_count=replay_valid_count,
            replay_capacity=self.replay_capacity,
            low_bits=packed.low_spec.bits,
            high_bits=packed.high_spec.bits,
            group_size=packed.low_spec.group_size,
            scale_bits=packed.low_spec.scale_bits,
            rounding=packed.low_spec.rounding,
            source_dtype=str(final_state.dtype),
            shape=tuple(final_state.shape),
            high_precision_groups=packed.high_precision_groups,
            high_precision_mask_sha256=_mask_sha256(packed) or "",
            checkpoint_bytes=sum(
                _tensor_bytes(tensor)
                for tensor in (
                    packed.low_payload,
                    packed.high_payload,
                    packed.scales,
                    packed.precision_mask,
                )
            ),
            query_ema_bytes=_tensor_bytes(query_ema),
            replay_capacity_bytes=replay_bytes,
            resident_bytes=sum(
                _tensor_bytes(tensor)
                for tensor in (
                    packed.low_payload,
                    packed.high_payload,
                    packed.scales,
                    packed.precision_mask,
                )
            )
            + _tensor_bytes(query_ema)
            + replay_bytes,
            relative_l2_error=float(relative_l2.item()),
            mean_squared_error=float(error.square().mean().item()),
            max_absolute_error=float(error.abs().max().item()),
        )

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Commit exactly one staged observation and return its resident state."""

        del kwargs
        if state_idx != 0:
            self.discard_pending_statelease_observation()
            raise IndexError(f"layer {self.layer_index} state_idx {state_idx} is outside [0, 1)")
        pending = self._pending_statelease_observation
        if pending is None:
            raise RuntimeError(
                f"layer {self.layer_index} has no staged StateLease observation "
                f"for state update {self._update_count}"
            )
        try:
            recurrent_states = self._validate_state(
                recurrent_states,
                name="recurrent_states",
            )
            if torch.is_grad_enabled() and recurrent_states.requires_grad:
                raise RuntimeError(
                    "StateLease recurrent states are inference-only and cannot "
                    "accept an autograd-tracked tensor"
                )
            if recurrent_states is not pending.final_state:
                raise RuntimeError(
                    "StateLease recurrent-state write is not the staged kernel final-state tensor"
                )
            if pending.update_index != self._update_count:
                raise RuntimeError(
                    f"layer {self.layer_index} has a stale StateLease observation "
                    f"for update {pending.update_index}; expected {self._update_count}"
                )
            if pending.candidate_ema.device != recurrent_states.device:
                raise ValueError("StateLease query EMA and recurrent state use different devices")
            expected_ema = (self.expected_heads, self.expected_rows)
            if (
                tuple(pending.candidate_ema.shape) != expected_ema
                or pending.candidate_ema.dtype != torch.float32
                or not torch.isfinite(pending.candidate_ema).all().item()
            ):
                raise RuntimeError("staged StateLease query EMA is malformed")

            low_spec, high_spec = self._selected_specs()
            key_buffer, update_buffer, decay_buffer, valid_count = self._candidate_buffers(
                device=recurrent_states.device
            )
            current_count = self._buffer_valid_count()
            previous_packed = self.packed_states[0]
            action: str
            reason: str
            boundary: int | None = None
            tie = False
            cut4_value: float | None = None
            cut5_value: float | None = None

            force_checkpoint = pending.initial_state is None or pending.token_count != 1
            if force_checkpoint:
                chosen_packed, materialized = self._pack_checkpoint(
                    recurrent_states,
                    query_ema=pending.candidate_ema,
                    low_spec=low_spec,
                    high_spec=high_spec,
                )
                self._clear_candidate_buffers(
                    key_buffer,
                    update_buffer,
                    decay_buffer,
                    valid_count,
                )
                action = (
                    "checkpoint_prefill" if pending.initial_state is None else "checkpoint_chunk"
                )
                reason = "prefill_or_uncached" if pending.initial_state is None else "multi_token"
            else:
                if (
                    pending.normalized_key is None
                    or pending.update is None
                    or pending.log_decay is None
                ):
                    raise RuntimeError("single-token StateLease observation lacks a replay record")
                if previous_packed is None:
                    raise RuntimeError("StateLease cannot append replay without a checkpoint")
                if current_count < self.replay_capacity - 1:
                    self._append_candidate_record(
                        key_buffer=key_buffer,
                        update_buffer=update_buffer,
                        decay_buffer=decay_buffer,
                        valid_count=valid_count,
                        slot=current_count,
                        normalized_key=pending.normalized_key,
                        update=pending.update,
                        log_decay=pending.log_decay,
                    )
                    chosen_packed = previous_packed
                    materialized = self._materialize_candidate(
                        packed=chosen_packed,
                        key_buffer=key_buffer,
                        update_buffer=update_buffer,
                        decay_buffer=decay_buffer,
                        valid_count=valid_count,
                    )
                    action = "replay_append"
                    reason = "lease_not_full"
                elif current_count == self.replay_capacity - 1:
                    assert pending.initial_state is not None
                    # Exercise the physically charged H=5 capacity before
                    # scoring either handoff.  Cut 4 retains this exact rounded
                    # slot after the decision; cut 5 discards it.
                    self._append_candidate_record(
                        key_buffer=key_buffer,
                        update_buffer=update_buffer,
                        decay_buffer=decay_buffer,
                        valid_count=valid_count,
                        slot=current_count,
                        normalized_key=pending.normalized_key,
                        update=pending.update,
                        log_decay=pending.log_decay,
                    )
                    if int(valid_count.item()) != self.replay_capacity:
                        raise RuntimeError("StateLease full event did not occupy all five slots")
                    cut4_packed, cut4_checkpoint = self._pack_checkpoint(
                        pending.initial_state,
                        query_ema=pending.candidate_ema,
                        low_spec=low_spec,
                        high_spec=high_spec,
                    )
                    record_key = key_buffer[current_count].clone()
                    record_update = update_buffer[current_count].clone()
                    record_decay = decay_buffer[current_count].clone()
                    cut4_materialized = replay_gated_delta_updates(
                        cut4_checkpoint,
                        record_key.unsqueeze(0).unsqueeze(0),
                        record_update.unsqueeze(0).unsqueeze(0),
                        record_decay.unsqueeze(0).unsqueeze(0),
                    )
                    cut5_packed, cut5_materialized = self._pack_checkpoint(
                        recurrent_states,
                        query_ema=pending.candidate_ema,
                        low_spec=low_spec,
                        high_spec=high_spec,
                    )
                    risk_energy = self._normalized_risk_energy(pending.candidate_ema)
                    cut4_risk = query_weighted_row_mse(
                        recurrent_states,
                        cut4_materialized,
                        risk_energy,
                    )
                    cut5_risk = query_weighted_row_mse(
                        recurrent_states,
                        cut5_materialized,
                        risk_energy,
                    )
                    decision = select_statelease_boundary(cut4_risk, cut5_risk)
                    boundary = decision.boundary
                    tie = decision.tie
                    cut4_value = decision.cut4_risk
                    cut5_value = decision.cut5_risk
                    self._clear_candidate_buffers(
                        key_buffer,
                        update_buffer,
                        decay_buffer,
                        valid_count,
                    )
                    if boundary == 4:
                        chosen_packed = cut4_packed
                        self._append_candidate_record(
                            key_buffer=key_buffer,
                            update_buffer=update_buffer,
                            decay_buffer=decay_buffer,
                            valid_count=valid_count,
                            slot=0,
                            normalized_key=record_key,
                            update=record_update,
                            log_decay=record_decay,
                        )
                        materialized = cut4_materialized
                        action = "boundary_4"
                    else:
                        chosen_packed = cut5_packed
                        materialized = cut5_materialized
                        action = "boundary_5"
                    reason = "full_buffer_cut4_vs_cut5"
                else:
                    raise RuntimeError(
                        "StateLease reached an invalid replay age before the four-vs-five handoff"
                    )

            replay_valid_count = int(valid_count.item())
            if (
                tuple(materialized.shape) != self._expected_state_shape()
                or not torch.isfinite(materialized).all().item()
            ):
                raise RuntimeError("StateLease resident state candidate is invalid")
            evidence = self._build_evidence(
                final_state=recurrent_states,
                materialized=materialized,
                packed=chosen_packed,
                pending=pending,
                action=action,
                boundary=boundary,
                tie=tie,
                cut4_risk=cut4_value,
                cut5_risk=cut5_value,
                replay_valid_count=replay_valid_count,
                query_ema=pending.candidate_ema,
                key_buffer=key_buffer,
                update_buffer=update_buffer,
                decay_buffer=decay_buffer,
                valid_count=valid_count,
            )

            # All fallible work is complete.  Commit the candidate references as
            # one bounded transaction; no persistent tensor was mutated above.
            self.packed_states[0] = chosen_packed
            self.query_energy_ema = pending.candidate_ema
            self.normalized_key_buffer = key_buffer
            self.update_buffer = update_buffer
            self.log_decay_buffer = decay_buffer
            self.valid_count = valid_count
            self.is_recurrent_states_initialized[0] = True
            if self.device is None:
                self.device = recurrent_states.device
                self.dtype = recurrent_states.dtype
            self._update_count += 1
            self.observations_staged += 1
            self.observations_committed += 1
            self.tokens_observed += pending.token_count
            if action != "replay_append":
                self.checkpoint_count += 1
            if boundary == 4:
                self.boundary4_count += 1
            elif boundary == 5:
                self.boundary5_count += 1
            if tie:
                self.tie_count += 1
            self.last_action = action
            self.last_reason = reason
            self.last_boundary = boundary
            self.last_cut4_risk = cut4_value
            self.last_cut5_risk = cut5_value
            self.last_update_evidence = evidence
            self._pending_statelease_observation = None
            return materialized
        except Exception:
            self.discard_pending_statelease_observation()
            raise

    def materialize_recurrent_state(self, state_idx: int = 0) -> torch.Tensor | None:
        if state_idx != 0:
            raise IndexError(f"layer {self.layer_index} state_idx {state_idx} is outside [0, 1)")
        packed = self.packed_states[0]
        if packed is None:
            return None
        device = packed.low_payload.device
        self._validate_buffer_storage(device=device)
        checkpoint = packed.dequantize()
        count = self._buffer_valid_count()
        if count == 0:
            return checkpoint
        assert self.normalized_key_buffer is not None
        assert self.update_buffer is not None
        assert self.log_decay_buffer is not None
        return replay_gated_delta_updates(
            checkpoint,
            self.normalized_key_buffer[:count].unsqueeze(0),
            self.update_buffer[:count].unsqueeze(0),
            self.log_decay_buffer[:count].unsqueeze(0),
        )

    def clear_recurrent_state(self) -> None:
        self.discard_pending_statelease_observation()
        if self.is_conv_states_initialized[0] or self.has_previous_state[0]:
            raise RuntimeError(
                f"layer {self.layer_index} cannot clear only its recurrent state "
                "while convolution history is active; call reset() to clear the "
                "complete layer atomically"
            )
        self.packed_states[0] = None
        self.query_energy_ema = None
        self.normalized_key_buffer = None
        self.update_buffer = None
        self.log_decay_buffer = None
        self.valid_count = None
        self.is_recurrent_states_initialized[0] = False

    def resident_payload_bytes(self) -> int:
        packed = self.packed_states[0]
        return (
            0
            if packed is None
            else _tensor_bytes(packed.low_payload) + _tensor_bytes(packed.high_payload)
        )

    def resident_scale_bytes(self) -> int:
        packed = self.packed_states[0]
        return 0 if packed is None else _tensor_bytes(packed.scales)

    def resident_mask_bytes(self) -> int:
        packed = self.packed_states[0]
        return 0 if packed is None else _tensor_bytes(packed.precision_mask)

    def resident_checkpoint_bytes(self) -> int:
        return (
            self.resident_payload_bytes() + self.resident_scale_bytes() + self.resident_mask_bytes()
        )

    def resident_recurrent_state_bytes(self) -> int:
        return self.resident_checkpoint_bytes()

    def query_ema_bytes(self) -> int:
        return _tensor_bytes(self.query_energy_ema)

    def replay_capacity_bytes(self) -> int:
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                self.normalized_key_buffer,
                self.update_buffer,
                self.log_decay_buffer,
                self.valid_count,
            )
        )

    def replay_occupied_bytes(self) -> int:
        if self.valid_count is None:
            return 0
        count = self._buffer_valid_count()
        per_record = (
            self.expected_heads * self.expected_rows * 2
            + self.expected_heads * self.low_spec.group_size * 2
            + self.expected_heads * 4
        )
        return _tensor_bytes(self.valid_count) + count * per_record

    def resident_bytes_including_statelease(self) -> int:
        return (
            self.resident_checkpoint_bytes() + self.query_ema_bytes() + self.replay_capacity_bytes()
        )

    def high_precision_group_count(self) -> int:
        packed = self.packed_states[0]
        return 0 if packed is None else packed.high_precision_groups

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        packed = self.packed_states[0]
        if packed is None:
            return 0
        dtype_size = torch.empty((), dtype=packed.original_dtype).element_size()
        return packed.elements * dtype_size

    def largest_materialized_recurrent_state_bytes(self) -> int:
        return self.full_precision_equivalent_recurrent_state_bytes()

    def statelease_diagnostics(
        self,
    ) -> dict[str, int | float | bool | str | None]:
        packed = self.packed_states[0]
        total = self.resident_bytes_including_statelease()
        elements = 0 if packed is None else packed.elements
        return {
            "layer_index": self.layer_index,
            "selection_method": self.selection_method,
            "state_codec": self.state_codec,
            "state_codec_seed": self.state_codec_seed,
            "state_codec_axis": "value",
            "state_codec_normalization": "orthonormal",
            "state_updates": self._update_count,
            "observations_staged": self.observations_staged,
            "observations_committed": self.observations_committed,
            "tokens_observed": self.tokens_observed,
            "checkpoint_count": self.checkpoint_count,
            "boundary4_count": self.boundary4_count,
            "boundary5_count": self.boundary5_count,
            "tie_count": self.tie_count,
            "replay_valid_count": self._buffer_valid_count(),
            "replay_capacity": self.replay_capacity,
            "last_action": self.last_action,
            "last_reason": self.last_reason,
            "last_boundary": self.last_boundary,
            "last_cut4_risk": self.last_cut4_risk,
            "last_cut5_risk": self.last_cut5_risk,
            "pending_observation": self._pending_statelease_observation is not None,
            "current_selected_count": self.high_precision_group_count(),
            "current_mask_sha256": _mask_sha256(packed),
            "checkpoint_bytes": self.resident_checkpoint_bytes(),
            "query_ema_bytes": self.query_ema_bytes(),
            "replay_capacity_bytes": self.replay_capacity_bytes(),
            "replay_occupied_bytes": self.replay_occupied_bytes(),
            "resident_bytes_including_statelease": total,
            "effective_bits_per_state_element": (8.0 * total / elements if elements else 0.0),
        }

    def reset(self) -> None:
        self.discard_pending_statelease_observation()
        if self.is_conv_states_initialized[0]:
            self.conv_states[0].zero_()
        packed = self.packed_states[0]
        if packed is not None:
            packed.low_payload.zero_()
            packed.high_payload.zero_()
            packed.scales.zero_()
        if self.query_energy_ema is not None:
            self.query_energy_ema.fill_(1.0 / self.expected_rows)
        for tensor in (
            self.normalized_key_buffer,
            self.update_buffer,
            self.log_decay_buffer,
            self.valid_count,
        ):
            if tensor is not None:
                tensor.zero_()
        self.has_previous_state[0] = False
        self._update_count = 0
        self.observations_staged = 0
        self.observations_committed = 0
        self.tokens_observed = 0
        self.checkpoint_count = 0
        self.boundary4_count = 0
        self.boundary5_count = 0
        self.tie_count = 0
        self.last_action = None
        self.last_reason = None
        self.last_boundary = None
        self.last_cut4_risk = None
        self.last_cut5_risk = None
        self.last_update_evidence = None

    def _reject_pending_transfer(self, operation: str) -> None:
        if self._pending_statelease_observation is None:
            return
        self.discard_pending_statelease_observation()
        raise RuntimeError(
            f"layer {self.layer_index} cannot {operation} with a pending "
            "StateLease observation; it was discarded"
        )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._reject_pending_transfer("reorder")
        if (
            not isinstance(beam_idx, torch.Tensor)
            or beam_idx.dtype != torch.long
            or beam_idx.ndim != 1
            or beam_idx.numel() != 1
            or int(beam_idx.item()) != 0
        ):
            raise ValueError("StateLease is batch-one only and supports only identity beam_idx=[0]")

    def offload(self) -> None:
        self._reject_pending_transfer("offload")
        if self.is_conv_states_initialized[0]:
            self.conv_states[0] = self.conv_states[0].to("cpu", non_blocking=True)
        packed = self.packed_states[0]
        if packed is not None:
            self.packed_states[0] = packed.to("cpu")
        if self.query_energy_ema is not None:
            self.query_energy_ema = self.query_energy_ema.to("cpu")
        for attribute in (
            "normalized_key_buffer",
            "update_buffer",
            "log_decay_buffer",
            "valid_count",
        ):
            tensor = getattr(self, attribute)
            if tensor is not None:
                setattr(self, attribute, tensor.to("cpu"))

    def prefetch(self) -> None:
        self._reject_pending_transfer("prefetch")
        if self.device is None:
            return
        target = torch.device(self.device)
        if self.is_conv_states_initialized[0] and self.conv_states[0].device != target:
            self.conv_states[0] = self.conv_states[0].to(target, non_blocking=True)
        packed = self.packed_states[0]
        if packed is not None and packed.low_payload.device != target:
            self.packed_states[0] = packed.to(target)
        if self.query_energy_ema is not None and self.query_energy_ema.device != target:
            self.query_energy_ema = self.query_energy_ema.to(
                target,
                non_blocking=True,
            )
        for attribute in (
            "normalized_key_buffer",
            "update_buffer",
            "log_decay_buffer",
            "valid_count",
        ):
            tensor = getattr(self, attribute)
            if tensor is not None and tensor.device != target:
                setattr(
                    self,
                    attribute,
                    tensor.to(target, non_blocking=True),
                )


class StateLeaseRecurrentStateCache(DynamicCache):
    """Drop-in exact-budget StateLease cache for Qwen3.5 linear layers."""

    selection_method = STATELEASE_GENERIC_SELECTION_METHOD

    def __init__(
        self,
        config: object,
        *,
        plan: ExactBudgetRowPlan,
        rounding: RoundingMode = "nearest",
        seed: int = 2339,
        record_evidence: bool = False,
        selection_method: str = STATELEASE_GENERIC_SELECTION_METHOD,
        experiment_identity_sha256: str | None = None,
    ) -> None:
        if not isinstance(record_evidence, bool):
            raise TypeError("record_evidence must be a bool")
        if not isinstance(selection_method, str) or not selection_method:
            raise ValueError("selection_method must be a non-empty string")
        if experiment_identity_sha256 is not None and (
            not isinstance(experiment_identity_sha256, str)
            or len(experiment_identity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in experiment_identity_sha256)
        ):
            raise ValueError("experiment_identity_sha256 must be a lowercase SHA-256 digest")
        super().__init__(config=config)
        self.plan = plan
        self.record_evidence = record_evidence
        self.selection_method = selection_method
        self.experiment_identity_sha256 = experiment_identity_sha256
        self.update_evidence: list[StateLeaseUpdateEvidence] = []
        self._update_index = 0
        self._active_statelease_transaction: _StateLeaseForwardTransaction | None = None

        linear_layer_indices = {
            layer_index
            for layer_index, layer in enumerate(self.layers)
            if isinstance(layer, LinearAttentionLayer)
        }
        shapes = _validate_exact_budget_plan(plan, linear_layer_indices)
        low_spec = QuantizationSpec(
            bits=plan.low_bits,
            group_size=plan.group_size,
            scale_bits=plan.scale_bits,
            rounding=rounding,
            seed=seed,
        )
        high_spec = QuantizationSpec(
            bits=plan.high_bits,
            group_size=plan.group_size,
            scale_bits=plan.scale_bits,
            rounding=rounding,
            seed=seed,
        )
        replaced = 0
        for layer_index, layer in enumerate(self.layers):
            if not isinstance(layer, LinearAttentionLayer):
                continue
            if layer.number_of_states != 1:
                raise ValueError(
                    f"layer {layer_index} has {layer.number_of_states} recurrent "
                    "states; StateLease requires exactly one"
                )
            expected_heads, expected_rows = shapes[layer_index]
            self.layers[layer_index] = StateLeaseLinearAttentionLayer(
                low_spec=low_spec,
                high_spec=high_spec,
                layer_index=layer_index,
                expected_heads=expected_heads,
                expected_rows=expected_rows,
                high_precision_group_indices=plan.groups_for_layer(layer_index),
                number_of_states=layer.number_of_states,
                selection_method=selection_method,
            )
            replaced += 1
        if replaced == 0:
            raise TypeError("config did not create any linear-attention layers")

    def _statelease_layer(self, layer_index: int) -> StateLeaseLinearAttentionLayer:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(f"layer_index {layer_index} is outside this cache")
        layer = self.layers[layer_index]
        if not isinstance(layer, StateLeaseLinearAttentionLayer):
            raise ValueError(f"layer {layer_index} is not a StateLease linear layer")
        return layer

    def statelease_layers(
        self,
    ) -> Iterator[tuple[int, StateLeaseLinearAttentionLayer]]:
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, StateLeaseLinearAttentionLayer):
                yield layer_index, layer

    @staticmethod
    def _snapshot_forward_layer(layer: object) -> _LayerForwardSnapshot:
        attributes: dict[str, object] = {}
        mutable_tensors: list[_MutableTensorSnapshot] = []
        source_attributes = getattr(layer, "__dict__", None)
        if not isinstance(source_attributes, dict):
            raise TypeError("cache layers must expose mutable instance attributes")
        for name, value in source_attributes.items():
            if isinstance(value, dict):
                copied: object = dict(value)
            elif isinstance(value, list):
                copied = list(value)
            elif isinstance(value, set):
                copied = set(value)
            else:
                copied = value
            attributes[name] = copied

            # Pinned Transformers updates convolution and conventional
            # recurrent-state tensors in place. Attention K/V tensors and
            # StateLease checkpoint/buffer tensors are replaced by reference,
            # so restoring their shallow attribute snapshot is sufficient.
            if name not in {"conv_states", "recurrent_states"}:
                continue
            values: tuple[object, ...]
            if isinstance(copied, dict):
                values = tuple(copied.values())
            elif isinstance(copied, (list, tuple)):
                values = tuple(copied)
            else:
                values = (copied,)
            for candidate in values:
                if isinstance(candidate, torch.Tensor):
                    mutable_tensors.append(
                        _MutableTensorSnapshot(
                            tensor=candidate,
                            value=candidate.detach().clone(memory_format=torch.preserve_format),
                        )
                    )
        return _LayerForwardSnapshot(
            attributes=attributes,
            mutable_tensors=tuple(mutable_tensors),
        )

    def begin_statelease_forward_transaction(self) -> _StateLeaseForwardTransaction:
        """Snapshot all cache layers before one complete outer model forward.

        The snapshot journals references for append/replace-style cache fields
        and copies only tensors that pinned Transformers mutates in place. It
        therefore covers convolution state, prior full-attention K/V
        references, StateLease checkpoints and replay metadata, flags,
        diagnostics, and evidence without cloning the full resident KV cache.
        """

        if self._active_statelease_transaction is not None:
            raise RuntimeError("a StateLease model-forward transaction is already active")
        pending_layers = [
            layer_index
            for layer_index, layer in self.statelease_layers()
            if layer._pending_statelease_observation is not None
        ]
        if pending_layers:
            raise RuntimeError(
                "cannot begin a StateLease model-forward transaction with pending "
                f"observations on layers {pending_layers}"
            )
        transaction = _StateLeaseForwardTransaction(
            cache_identity=id(self),
            layer_snapshots=tuple(self._snapshot_forward_layer(layer) for layer in self.layers),
            update_evidence=tuple(self.update_evidence),
            update_index=self._update_index,
        )
        self._active_statelease_transaction = transaction
        return transaction

    def _require_active_transaction(
        self,
        transaction: object,
    ) -> _StateLeaseForwardTransaction:
        if not isinstance(transaction, _StateLeaseForwardTransaction):
            raise TypeError("transaction is not a StateLease forward transaction")
        if transaction.cache_identity != id(self):
            raise ValueError("StateLease transaction belongs to a different cache")
        if self._active_statelease_transaction is not transaction or not transaction.active:
            raise RuntimeError("StateLease transaction is not active")
        return transaction

    def commit_statelease_forward_transaction(self, transaction: object) -> None:
        """Commit one successful outer model forward after receipt checks."""

        active = self._require_active_transaction(transaction)
        pending_layers = [
            layer_index
            for layer_index, layer in self.statelease_layers()
            if layer._pending_statelease_observation is not None
        ]
        if pending_layers:
            raise RuntimeError(
                "StateLease model forward returned with unconsumed observations "
                f"on layers {pending_layers}"
            )
        active.active = False
        self._active_statelease_transaction = None

    def rollback_statelease_forward_transaction(self, transaction: object) -> None:
        """Restore every layer after any exception in the outer model forward."""

        active = self._require_active_transaction(transaction)
        if len(active.layer_snapshots) != len(self.layers):
            raise RuntimeError("StateLease cache geometry changed during a model forward")
        try:
            with torch.no_grad():
                for layer, snapshot in zip(
                    self.layers,
                    active.layer_snapshots,
                    strict=True,
                ):
                    for tensor_snapshot in snapshot.mutable_tensors:
                        tensor_snapshot.tensor.copy_(tensor_snapshot.value)
                    layer_attributes = getattr(layer, "__dict__", None)
                    if not isinstance(layer_attributes, dict):
                        raise TypeError("cache layers must expose mutable instance attributes")
                    layer_attributes.clear()
                    layer_attributes.update(snapshot.attributes)
            self.update_evidence = list(active.update_evidence)
            self._update_index = active.update_index
        finally:
            active.active = False
            self._active_statelease_transaction = None

    def has_pending_statelease_observation(self, layer_index: int) -> bool:
        """Return whether one successful kernel record awaits its cache write."""

        return self._statelease_layer(layer_index)._pending_statelease_observation is not None

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
        self._statelease_layer(layer_index).stage_statelease_observation(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            final_state,
        )

    def discard_pending_statelease_observation(self, layer_index: int) -> None:
        self._statelease_layer(layer_index).discard_pending_statelease_observation()

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        layer = self._statelease_layer(layer_idx)
        materialized = layer.update_recurrent_state(
            recurrent_states,
            state_idx,
            **kwargs,
        )
        if self.record_evidence:
            evidence = layer.last_update_evidence
            if evidence is None:
                raise RuntimeError("StateLease committed without update evidence")
            self.update_evidence.append(replace(evidence, update_index=self._update_index))
            self._update_index += 1
        return materialized

    def resident_payload_bytes(self) -> int:
        return sum(layer.resident_payload_bytes() for _, layer in self.statelease_layers())

    def resident_scale_bytes(self) -> int:
        return sum(layer.resident_scale_bytes() for _, layer in self.statelease_layers())

    def resident_mask_bytes(self) -> int:
        return sum(layer.resident_mask_bytes() for _, layer in self.statelease_layers())

    def resident_checkpoint_bytes(self) -> int:
        return sum(layer.resident_checkpoint_bytes() for _, layer in self.statelease_layers())

    def query_ema_bytes(self) -> int:
        return sum(layer.query_ema_bytes() for _, layer in self.statelease_layers())

    def replay_capacity_bytes(self) -> int:
        return sum(layer.replay_capacity_bytes() for _, layer in self.statelease_layers())

    def replay_occupied_bytes(self) -> int:
        return sum(layer.replay_occupied_bytes() for _, layer in self.statelease_layers())

    def resident_bytes_including_statelease(self) -> int:
        return sum(
            layer.resident_bytes_including_statelease() for _, layer in self.statelease_layers()
        )

    def full_precision_equivalent_recurrent_state_bytes(self) -> int:
        return sum(
            layer.full_precision_equivalent_recurrent_state_bytes()
            for _, layer in self.statelease_layers()
        )

    def largest_materialized_recurrent_state_bytes(self) -> int:
        return max(
            (
                layer.largest_materialized_recurrent_state_bytes()
                for _, layer in self.statelease_layers()
            ),
            default=0,
        )

    def high_precision_group_count(self) -> int:
        return sum(layer.high_precision_group_count() for _, layer in self.statelease_layers())

    def storage_summary(self) -> dict[str, int | float | bool | str | None]:
        checkpoint = self.resident_checkpoint_bytes()
        query_ema = self.query_ema_bytes()
        replay_capacity = self.replay_capacity_bytes()
        replay_occupied = self.replay_occupied_bytes()
        total = checkpoint + query_ema + replay_capacity
        full_precision = self.full_precision_equivalent_recurrent_state_bytes()
        elements = sum(
            0 if layer.packed_checkpoint is None else layer.packed_checkpoint.elements
            for _, layer in self.statelease_layers()
        )
        return {
            "selection_method": self.selection_method,
            "experiment_identity_sha256": self.experiment_identity_sha256,
            "payload_bytes": self.resident_payload_bytes(),
            "scale_bytes": self.resident_scale_bytes(),
            "mask_bytes": self.resident_mask_bytes(),
            "checkpoint_bytes": checkpoint,
            "resident_bytes": checkpoint,
            "query_ema_bytes": query_ema,
            "replay_capacity_bytes": replay_capacity,
            "replay_occupied_bytes": replay_occupied,
            "resident_bytes_including_statelease": total,
            "high_precision_groups": self.high_precision_group_count(),
            "full_precision_equivalent_bytes": full_precision,
            "largest_materialized_state_bytes": (self.largest_materialized_recurrent_state_bytes()),
            "effective_bits_per_state_element": (8.0 * total / elements if elements else 0.0),
            "resident_compression_ratio": (full_precision / checkpoint if checkpoint else 0.0),
            "resident_compression_ratio_including_statelease": (
                full_precision / total if total else 0.0
            ),
            "physical_reduction_realized": (checkpoint > 0 and checkpoint < full_precision),
            "physical_reduction_realized_including_statelease": (
                total > 0 and total < full_precision
            ),
            "forward_transaction_active": self._active_statelease_transaction is not None,
        }

    def statelease_diagnostics(
        self,
    ) -> list[dict[str, int | float | bool | str | None]]:
        return [layer.statelease_diagnostics() for _, layer in self.statelease_layers()]

    def reset(self) -> None:
        if self._active_statelease_transaction is not None:
            raise RuntimeError("cannot reset a StateLease cache during a model forward")
        super().reset()
        self.update_evidence.clear()
        self._update_index = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if (
            not isinstance(beam_idx, torch.Tensor)
            or beam_idx.dtype != torch.long
            or beam_idx.ndim != 1
            or beam_idx.numel() != 1
            or int(beam_idx.item()) != 0
        ):
            raise ValueError("StateLease is batch-one only and supports only identity beam_idx=[0]")
        super().reorder_cache(beam_idx)

    def offload(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        layer = self.layers[layer_idx]
        if isinstance(layer, StateLeaseLinearAttentionLayer):
            layer.offload()
            return
        super().offload(layer_idx, only_non_sliding=only_non_sliding)

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        layer = self.layers[layer_idx]
        if isinstance(layer, StateLeaseLinearAttentionLayer):
            layer.prefetch()
            return
        super().prefetch(layer_idx, only_non_sliding=only_non_sliding)
