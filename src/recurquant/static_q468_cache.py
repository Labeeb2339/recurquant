"""Transactional Qwen3.5 runtime for frozen static RHT policies.

The static policies choose every recurrent-row precision during calibration.
This runtime deliberately reuses the audited equal-byte cache's root-model
transaction, layer receipts, rollback, and batch-one lifecycle.  It changes
only the complete-state packer and the evidence needed for a static policy.

The implementation is correctness-first.  Every model forward materializes
the previous complete FP32 state as transient workspace and repacks the new
complete state after the LM head succeeds.  It is suitable for quality
experiments, but it is not the packed-native deployment path.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import torch

from .qwen35 import _validate_transformers_compatibility, _validated_text_config
from .statelease_equal_byte_baselines import (
    FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
    RHT_Q4_Q6_Q8,
    EqualByteLayout,
)
from .statelease_equal_byte_cache import EqualByteQwen35Cache, create_qwen35_equal_byte_cache
from .static_q468 import (
    FROZEN_QWEN35_STATIC_Q468_GEOMETRY,
    STATIC_Q48_COMPARATOR_METHOD,
    STATIC_Q468_ABLATION_METHOD,
    STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
    STATIC_Q468_MSE_METHOD,
    STATIC_Q468_PRIMARY_METHOD,
    STATIC_Q468_UNIFORM_Q4_METHOD,
    STATIC_Q468_UNIFORM_Q8_METHOD,
    StaticPackedRhtQ48State,
    StaticPackedRhtQ468State,
    StaticRhtByteLedger,
    StaticRhtQ48Policy,
    StaticRhtQ468Geometry,
    StaticRhtQ468Policy,
    deserialize_static_rht_q48_policy,
    deserialize_static_rht_q468_policy,
    pack_static_rht_q48,
    pack_static_rht_q468,
    serialize_static_rht_q48_policy,
    serialize_static_rht_q468_policy,
    static_q48_byte_ledger,
    static_q468_byte_ledger,
    verify_static_packed_rht_q48,
    verify_static_packed_rht_q468,
    verify_static_rht_q48_policy,
    verify_static_rht_q468_policy,
)

StaticRhtPolicy: TypeAlias = StaticRhtQ468Policy | StaticRhtQ48Policy
StaticPackedRhtState: TypeAlias = StaticPackedRhtQ468State | StaticPackedRhtQ48State
StaticPolicyKind: TypeAlias = Literal["q468", "q48"]

DYNAMIC_Q468_BASELINE_METHOD = "rht_q468_dynamic_k27030"
# Compatibility alias for callers using the pre-correction public name.  The
# method is a dynamic comparison baseline, not an oracle for held-out quality.
DYNAMIC_Q468_ORACLE_METHOD = DYNAMIC_Q468_BASELINE_METHOD
FROZEN_STATIC_RUNTIME_METHODS = frozenset(
    (
        STATIC_Q468_PRIMARY_METHOD,
        STATIC_Q468_ABLATION_METHOD,
        STATIC_Q468_MSE_METHOD,
        STATIC_Q468_DIAG_EMPIRICAL_FISHER_H1_METHOD,
        STATIC_Q468_UNIFORM_Q4_METHOD,
        STATIC_Q468_UNIFORM_Q8_METHOD,
        STATIC_Q48_COMPARATOR_METHOD,
    )
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _structural_geometry(geometry: StaticRhtQ468Geometry) -> tuple[object, ...]:
    return (
        geometry.layer_indices,
        geometry.heads,
        geometry.key_rows,
        geometry.value_width,
    )


def _layout_geometry(layout: EqualByteLayout) -> tuple[object, ...]:
    return (
        layout.layer_indices,
        layout.heads,
        layout.key_rows,
        layout.value_width,
    )


@dataclass(frozen=True, slots=True)
class StaticRhtRuntimeCheckpoint:
    """One complete static packed checkpoint and its authenticated identity."""

    state: StaticPackedRhtState
    expected_policy_sha256: str
    expected_method_id: str

    def __post_init__(self) -> None:
        self.validate()

    @property
    def policy(self) -> StaticRhtPolicy:
        return self.state.policy

    @property
    def ledger(self) -> StaticRhtByteLedger:
        return self.state.ledger

    @property
    def resident_bytes(self) -> int:
        return self.state.resident_bytes

    def persistent_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return self.state.persistent_tensors()

    def materialize(self) -> dict[int, torch.Tensor]:
        # Policy tensors are ordinary mutable torch storage even though the
        # dataclass is frozen.  Re-authenticate immediately before every decode
        # so an accidental in-place edit cannot silently change the reserved
        # method while retaining its old expected digest.
        self.validate()
        return self.state.materialize()

    def validate(self) -> None:
        _require_sha256(self.expected_policy_sha256, name="expected_policy_sha256")
        if self.policy.policy_sha256 != self.expected_policy_sha256:
            raise ValueError("static checkpoint policy SHA-256 does not match the expected digest")
        if self.policy.method_id != self.expected_method_id:
            raise ValueError("static checkpoint method identity drifted")
        if isinstance(self.state, StaticPackedRhtQ468State):
            verify_static_packed_rht_q468(self.state)
        elif isinstance(self.state, StaticPackedRhtQ48State):
            verify_static_packed_rht_q48(self.state)
        else:
            raise TypeError("state must be a supported static packed RHT state")
        if self.resident_bytes != self.ledger.resident_bytes:
            raise ValueError("static checkpoint resident storage differs from its byte ledger")

    def to(self, device: torch.device | str) -> StaticRhtRuntimeCheckpoint:
        return StaticRhtRuntimeCheckpoint(
            state=self.state.clone_to(device),
            expected_policy_sha256=self.expected_policy_sha256,
            expected_method_id=self.expected_method_id,
        )


@dataclass(frozen=True, slots=True)
class StaticRhtCacheUpdateEvidence:
    """Evidence for one successful static all-layer checkpoint transaction."""

    update_index: int
    method_id: str
    policy_sha256: str
    selection_sha256: str
    pool_offsets_sha256: str
    token_count: int
    layer_indices: tuple[int, ...]
    query_input_dtypes: tuple[str, ...]
    previous_checkpoint_present: bool
    retained_raw_state_workspace_peak_bytes: int
    retained_query_workspace_peak_bytes: int
    workspace_measurement_scope: str
    cuda_allocator_peak_measured: bool
    cuda_allocator_peak_bytes: int | None
    logical_fp32_state_bytes: int
    payload_bytes: int
    scale_bytes: int
    precision_code_bytes: int
    pool_offset_bytes: int
    alignment_bytes: int
    resident_tensor_storage_bytes: int
    target_resident_bytes: int
    budget_delta_bytes: int
    exact_budget_eligible: bool
    selected_units: int
    mean_squared_error: float
    relative_l2_error: float
    max_absolute_error: float

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


def _error_metrics(
    source: dict[int, torch.Tensor],
    materialized: dict[int, torch.Tensor],
    *,
    geometry: StaticRhtQ468Geometry,
) -> tuple[float, float, float]:
    if set(source) != set(geometry.layer_indices) or set(materialized) != set(
        geometry.layer_indices
    ):
        raise RuntimeError("static codec error measurement omitted a recurrent layer")
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_source = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    for layer_index in geometry.layer_indices:
        reference = source[layer_index].detach().to(device="cpu", dtype=torch.float64)
        restored = materialized[layer_index].detach().to(device="cpu", dtype=torch.float64)
        error = restored - reference
        squared_error += error.square().sum()
        squared_source += reference.square().sum()
        maximum = max(maximum, float(error.abs().max().item()))
    mse = float((squared_error / geometry.state_elements).item())
    relative = float((squared_error.sqrt() / squared_source.sqrt().clamp_min(1e-12)).item())
    return mse, relative, maximum


class StaticRhtQwen35Cache(EqualByteQwen35Cache):
    """Batch-one transactional cache for one immutable static RHT policy.

    The cache stores only canonical policy bytes and scalar/hash metadata before
    its first checkpoint.  It never retains a second tensor policy beside the
    policy owned by the installed checkpoint.
    """

    def __init__(
        self,
        config: object,
        *,
        policy: StaticRhtPolicy,
        expected_policy_sha256: str,
        layout: EqualByteLayout = FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
        record_evidence: bool = False,
    ) -> None:
        if not isinstance(layout, EqualByteLayout):
            raise TypeError("layout must be an EqualByteLayout")
        if not isinstance(record_evidence, bool):
            raise TypeError("record_evidence must be a bool")

        if isinstance(policy, StaticRhtQ468Policy):
            policy_kind: StaticPolicyKind = "q468"
            serialized = serialize_static_rht_q468_policy(policy)
            canonical = deserialize_static_rht_q468_policy(serialized)
            verify_static_rht_q468_policy(
                canonical,
                expected_policy_sha256=expected_policy_sha256,
            )
            selection_sha256 = canonical.code_map_sha256
            ledger = static_q468_byte_ledger(
                canonical.geometry,
                canonical.marginal_steps,
                method_id=canonical.method_id,
            )
        elif isinstance(policy, StaticRhtQ48Policy):
            policy_kind = "q48"
            serialized = serialize_static_rht_q48_policy(policy)
            canonical = deserialize_static_rht_q48_policy(serialized)
            verify_static_rht_q48_policy(
                canonical,
                expected_policy_sha256=expected_policy_sha256,
            )
            selection_sha256 = canonical.mask_sha256
            ledger = static_q48_byte_ledger(
                canonical.geometry,
                canonical.promoted_rows,
                method_id=canonical.method_id,
            )
        else:
            raise TypeError("policy must be a StaticRhtQ468Policy or StaticRhtQ48Policy")

        expected_policy_sha256 = _require_sha256(
            expected_policy_sha256,
            name="expected_policy_sha256",
        )
        if _structural_geometry(canonical.geometry) != _layout_geometry(layout):
            raise ValueError("static policy geometry does not match the Qwen3.5 cache layout")
        if canonical.geometry.target_resident_bytes != layout.expected_resident_bytes:
            raise ValueError("static policy target bytes do not match the cache layout target")

        # The superclass establishes the audited layer objects and transaction
        # state.  Its dynamic codec is never called by this subclass.
        super().__init__(
            config,
            codec=RHT_Q4_Q6_Q8,
            layout=layout,
            record_evidence=record_evidence,
        )

        # Retain immutable host bytes and scalars only.  In particular, do not
        # assign ``canonical`` or the caller's ``policy`` to this object.
        self.codec = canonical.method_id  # type: ignore[assignment]
        self.method_id = canonical.method_id
        self.policy_kind = policy_kind
        self.expected_policy_sha256 = expected_policy_sha256
        self.selection_sha256 = selection_sha256
        self.pool_offsets_sha256 = canonical.pool_offsets_sha256
        self._serialized_policy = serialized
        self._static_target_resident_bytes = canonical.geometry.target_resident_bytes
        self._static_expected_resident_bytes = ledger.resident_bytes
        self._static_budget_delta_bytes = ledger.budget_delta_bytes
        self._static_exact_budget_eligible = ledger.exact_budget_eligible
        self.checkpoint: StaticRhtRuntimeCheckpoint | None = None  # type: ignore[assignment]
        self.update_evidence: list[StaticRhtCacheUpdateEvidence] = []  # type: ignore[assignment]
        self.last_evidence: StaticRhtCacheUpdateEvidence | None = None  # type: ignore[assignment]

    def _load_policy(self) -> StaticRhtPolicy:
        if self.policy_kind == "q468":
            policy = deserialize_static_rht_q468_policy(self._serialized_policy)
            verify_static_rht_q468_policy(
                policy,
                expected_policy_sha256=self.expected_policy_sha256,
            )
        else:
            policy = deserialize_static_rht_q48_policy(self._serialized_policy)
            verify_static_rht_q48_policy(
                policy,
                expected_policy_sha256=self.expected_policy_sha256,
            )
        if policy.method_id != self.method_id:
            raise RuntimeError("serialized static policy method identity drifted")
        return policy

    def _pack_static_candidate(
        self,
        states: dict[int, torch.Tensor],
    ) -> StaticRhtRuntimeCheckpoint:
        policy = self._load_policy()
        if isinstance(policy, StaticRhtQ468Policy):
            state: StaticPackedRhtState = pack_static_rht_q468(states, policy)
        else:
            state = pack_static_rht_q48(states, policy)
        return StaticRhtRuntimeCheckpoint(
            state=state,
            expected_policy_sha256=self.expected_policy_sha256,
            expected_method_id=self.method_id,
        )

    def _pack_candidate(
        self,
        states: dict[int, torch.Tensor],
        query_ema: torch.Tensor,
    ) -> None:
        del states, query_ema
        raise RuntimeError(
            "static RHT caches must use the immutable-policy packer, not the dynamic codec"
        )

    def commit_statelease_forward_transaction(self, transaction: object) -> None:
        """Pack and install one static checkpoint only after complete model success."""

        active = self._require_active_transaction(transaction)
        if self._pending_observations:
            raise RuntimeError(
                "model forward returned with unconsumed static observations "
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

        candidate = self._pack_static_candidate(self._final_states)
        candidate.validate()
        ledger = candidate.ledger
        if candidate.resident_bytes != ledger.resident_bytes:
            raise RuntimeError("static candidate checkpoint violates its physical byte ledger")
        if candidate.resident_bytes != self._static_expected_resident_bytes:
            raise RuntimeError("static candidate checkpoint violates its frozen method bytes")
        if ledger.target_resident_bytes != self._static_target_resident_bytes:
            raise RuntimeError("static candidate target byte identity drifted")
        if ledger.budget_delta_bytes != self._static_budget_delta_bytes:
            raise RuntimeError("static candidate budget delta drifted")
        if ledger.exact_budget_eligible is not self._static_exact_budget_eligible:
            raise RuntimeError("static candidate exact-budget eligibility drifted")

        materialized = candidate.materialize()
        mse, relative, maximum = _error_metrics(
            self._final_states,
            materialized,
            geometry=candidate.policy.geometry,
        )
        layers = tuple(self.equal_byte_layers())
        if tuple(index for index, _layer in layers) != self.layout.layer_indices:
            raise RuntimeError("static cache recurrent layer identity drifted before commit")

        evidence = StaticRhtCacheUpdateEvidence(
            update_index=self.update_count,
            method_id=self.method_id,
            policy_sha256=self.expected_policy_sha256,
            selection_sha256=self.selection_sha256,
            pool_offsets_sha256=self.pool_offsets_sha256,
            token_count=token_count,
            layer_indices=tuple(self._receipt_order),
            query_input_dtypes=tuple(
                str(self._queries[index].dtype) for index in self.layout.layer_indices
            ),
            previous_checkpoint_present=active.checkpoint is not None,
            retained_raw_state_workspace_peak_bytes=self._forward_state_workspace_peak_bytes,
            retained_query_workspace_peak_bytes=self._forward_query_workspace_peak_bytes,
            workspace_measurement_scope="cache_retained_forward_tensors_only",
            cuda_allocator_peak_measured=False,
            cuda_allocator_peak_bytes=None,
            logical_fp32_state_bytes=self.layout.fp32_state_bytes,
            payload_bytes=ledger.payload_bytes,
            scale_bytes=ledger.scale_bytes,
            precision_code_bytes=ledger.precision_code_bytes,
            pool_offset_bytes=ledger.pool_offset_bytes,
            alignment_bytes=ledger.alignment_bytes,
            resident_tensor_storage_bytes=ledger.resident_bytes,
            target_resident_bytes=ledger.target_resident_bytes,
            budget_delta_bytes=ledger.budget_delta_bytes,
            exact_budget_eligible=ledger.exact_budget_eligible,
            selected_units=ledger.selected_units,
            mean_squared_error=mse,
            relative_l2_error=relative,
            max_absolute_error=maximum,
        )
        next_evidence = (
            [*self.update_evidence, evidence] if self.record_evidence else self.update_evidence
        )

        # Every fallible pack, materialization, metric, ledger, and layer check
        # is complete.  Replace references only at this final commit point.
        self.checkpoint = candidate
        self.update_count += 1
        self.successful_tokens += token_count
        self.last_evidence = evidence
        if self.record_evidence:
            self.update_evidence = next_evidence
        for _, layer in layers:
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True

        self._clear_forward_workspace()
        active.active = False
        self._active_equal_byte_transaction = None

    def storage_summary(self) -> dict[str, object]:
        summary: dict[str, object] = dict(super().storage_summary())
        resident = 0 if self.checkpoint is None else self.checkpoint.resident_bytes
        summary.update(
            {
                "codec": f"static_rht_{self.policy_kind}",
                "method_id": self.method_id,
                "policy_sha256": self.expected_policy_sha256,
                "selection_sha256": self.selection_sha256,
                "pool_offsets_sha256": self.pool_offsets_sha256,
                "resident_bytes": resident,
                "resident_tensor_storage_bytes": resident,
                "expected_resident_bytes": self._static_expected_resident_bytes,
                "target_resident_bytes": self._static_target_resident_bytes,
                "budget_delta_bytes": self._static_budget_delta_bytes,
                "exact_budget_eligible": self._static_exact_budget_eligible,
                "policy_tensor_storage_bytes_outside_checkpoint": 0,
                "serialized_policy_host_bytes": len(self._serialized_policy),
                "workspace_measurement_scope": "cache_retained_forward_tensors_only",
                "cuda_allocator_peak_measured": False,
                "cuda_allocator_peak_bytes": None,
            }
        )
        return summary


def create_qwen35_static_rht_cache(
    model_or_config: object,
    *,
    policy: StaticRhtPolicy,
    expected_policy_sha256: str,
    record_evidence: bool = False,
) -> StaticRhtQwen35Cache:
    """Create a policy-locked Experiment 013 static Qwen3.5 cache.

    The expected policy SHA must come from the separately authenticated and
    promoted experiment identity.  This factory validates model structure and,
    when given a model, evaluation mode, eager attention, and single-device
    placement.  It does not authenticate local model files or a Hub revision.
    """

    _validate_transformers_compatibility()
    config = _validated_text_config(model_or_config)
    if not isinstance(policy, (StaticRhtQ468Policy, StaticRhtQ48Policy)):
        raise TypeError("policy must be a StaticRhtQ468Policy or StaticRhtQ48Policy")
    if policy.method_id not in FROZEN_STATIC_RUNTIME_METHODS:
        raise ValueError("public Experiment 013 runtime accepts only the seven frozen methods")
    if policy.geometry != FROZEN_QWEN35_STATIC_Q468_GEOMETRY:
        raise ValueError("frozen Experiment 013 method requires the frozen Qwen3.5 geometry")
    return StaticRhtQwen35Cache(
        config,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        record_evidence=record_evidence,
    )


def create_qwen35_dynamic_q468_baseline_cache(
    model_or_config: object,
    *,
    record_evidence: bool = False,
) -> EqualByteQwen35Cache:
    """Create the named dynamic K27030 comparison baseline.

    This is the existing global RHT Q4/Q6/Q8 allocator with its frozen 27,030
    marginal steps and declared persistent query-energy EMA.  The factory adds
    an experiment method identity; it does not change the codec or claim an
    oracle for held-out model quality.
    """

    _validate_transformers_compatibility()
    config = _validated_text_config(model_or_config)
    if FROZEN_QWEN35_EQUAL_BYTE_LAYOUT.multibit_marginal_steps != 27_030:
        raise RuntimeError("dynamic Q468 baseline layout no longer has K27030")
    cache = create_qwen35_equal_byte_cache(
        config,
        codec=RHT_Q4_Q6_Q8,
        layout=FROZEN_QWEN35_EQUAL_BYTE_LAYOUT,
        record_evidence=record_evidence,
    )
    cache.method_id = DYNAMIC_Q468_BASELINE_METHOD  # type: ignore[attr-defined]
    return cache


def create_qwen35_dynamic_q468_oracle_cache(
    model_or_config: object,
    *,
    record_evidence: bool = False,
) -> EqualByteQwen35Cache:
    """Compatibility wrapper for the former dynamic-Q468 ``oracle`` name.

    New code should use :func:`create_qwen35_dynamic_q468_baseline_cache`.
    """

    return create_qwen35_dynamic_q468_baseline_cache(
        model_or_config,
        record_evidence=record_evidence,
    )
