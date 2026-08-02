"""Fixed checkpoint/replay comparators for Experiment 010.

The fixed policies deliberately reuse :mod:`recurquant.statelease_cache` for
observation validation, causal query-energy tracking, RHT-CQER checkpoint
packing, BF16/FP32 replay storage, materialization, and evidence accounting.
Only the checkpoint schedule differs:

``fixed_ccN``
    Checkpoint the current raw state after every ``N`` one-token decode
    records and clear the replay buffer.

``fixed_cut4_in5``
    Wait for five records, checkpoint the incoming ``S4`` state, and retain
    the exact rounded fifth record in slot zero.

The equal-allocation policies always reserve five replay slots.  The
Nemotron-style ``fixed_cc8`` reference reserves eight slots and is explicitly
off budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from .quantization import RoundingMode
from .row_policy import ExactBudgetRowPlan
from .statelease_cache import (
    StateLeaseLinearAttentionLayer,
    StateLeaseRecurrentStateCache,
)

FixedReplayMode: TypeAlias = Literal[
    "fixed_cc1",
    "fixed_cc2",
    "fixed_cc4",
    "fixed_cc5",
    "fixed_cut4_in5",
    "fixed_cc8",
]

EQUAL_ALLOCATION_REPLAY_CAPACITY = 5
OFF_BUDGET_CC8_REPLAY_CAPACITY = 8


@dataclass(frozen=True, slots=True)
class FixedReplayPolicy:
    """One immutable fixed replay schedule."""

    mode: FixedReplayMode
    checkpoint_period: int
    replay_capacity: int
    retain_last_record: bool
    equal_allocation: bool
    off_budget: bool


_POLICIES: dict[str, FixedReplayPolicy] = {
    "fixed_cc1": FixedReplayPolicy(
        mode="fixed_cc1",
        checkpoint_period=1,
        replay_capacity=EQUAL_ALLOCATION_REPLAY_CAPACITY,
        retain_last_record=False,
        equal_allocation=True,
        off_budget=False,
    ),
    "fixed_cc2": FixedReplayPolicy(
        mode="fixed_cc2",
        checkpoint_period=2,
        replay_capacity=EQUAL_ALLOCATION_REPLAY_CAPACITY,
        retain_last_record=False,
        equal_allocation=True,
        off_budget=False,
    ),
    "fixed_cc4": FixedReplayPolicy(
        mode="fixed_cc4",
        checkpoint_period=4,
        replay_capacity=EQUAL_ALLOCATION_REPLAY_CAPACITY,
        retain_last_record=False,
        equal_allocation=True,
        off_budget=False,
    ),
    "fixed_cc5": FixedReplayPolicy(
        mode="fixed_cc5",
        checkpoint_period=5,
        replay_capacity=EQUAL_ALLOCATION_REPLAY_CAPACITY,
        retain_last_record=False,
        equal_allocation=True,
        off_budget=False,
    ),
    "fixed_cut4_in5": FixedReplayPolicy(
        mode="fixed_cut4_in5",
        checkpoint_period=5,
        replay_capacity=EQUAL_ALLOCATION_REPLAY_CAPACITY,
        retain_last_record=True,
        equal_allocation=True,
        off_budget=False,
    ),
    "fixed_cc8": FixedReplayPolicy(
        mode="fixed_cc8",
        checkpoint_period=8,
        replay_capacity=OFF_BUDGET_CC8_REPLAY_CAPACITY,
        retain_last_record=False,
        equal_allocation=False,
        off_budget=True,
    ),
}


def fixed_replay_policy(mode: FixedReplayMode | str) -> FixedReplayPolicy:
    """Return the frozen schedule for ``mode`` and fail closed on aliases."""

    if not isinstance(mode, str):
        raise TypeError("fixed replay mode must be a string")
    try:
        return _POLICIES[mode]
    except KeyError as error:
        choices = ", ".join(_POLICIES)
        raise ValueError(
            f"unknown fixed replay mode {mode!r}; expected one of {choices}"
        ) from error


class FixedReplayLinearAttentionLayer(StateLeaseLinearAttentionLayer):
    """One fixed-schedule batch-one Gated DeltaNet replay layer."""

    def __init__(
        self,
        *,
        policy: FixedReplayPolicy,
        selection_method: str | None = None,
        **kwargs: object,
    ) -> None:
        if not isinstance(policy, FixedReplayPolicy):
            raise TypeError("policy must be a FixedReplayPolicy")
        super().__init__(**kwargs)
        self.policy = policy
        self.replay_capacity = policy.replay_capacity
        self.selection_method = selection_method or (
            f"{policy.mode}_right_rht_query_ema32_weighted_mse_configurable_quota"
        )
        self.scheduled_checkpoint_count = 0
        self.forced_checkpoint_count = 0
        self.replay_append_count = 0
        self.retained_tail_count = 0

    def logical_replay_capacity_bytes(self) -> int:
        """Return charged tensor bytes independent of current occupancy."""

        bytes_per_record = (
            self.expected_heads * self.expected_rows * 2
            + self.expected_heads * self.low_spec.group_size * 2
            + self.expected_heads * 4
        )
        return self.replay_capacity * bytes_per_record + 4

    def _validate_pending_commit(
        self,
        recurrent_states: torch.Tensor,
        *,
        state_idx: int,
    ) -> tuple[torch.Tensor, object]:
        if state_idx != 0:
            self.discard_pending_statelease_observation()
            raise IndexError(f"layer {self.layer_index} state_idx {state_idx} is outside [0, 1)")
        pending = self._pending_statelease_observation
        if pending is None:
            raise RuntimeError(
                f"layer {self.layer_index} has no staged StateLease observation "
                f"for state update {self._update_count}"
            )
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
        return recurrent_states, pending

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Commit one staged observation according to the frozen schedule."""

        del kwargs
        try:
            recurrent_states, pending_object = self._validate_pending_commit(
                recurrent_states,
                state_idx=state_idx,
            )
            pending = pending_object
            low_spec, high_spec = self._selected_specs()
            key_buffer, update_buffer, decay_buffer, valid_count = self._candidate_buffers(
                device=recurrent_states.device
            )
            current_count = self._buffer_valid_count()
            previous_packed = self.packed_states[0]

            action: str
            reason: str
            boundary: int | None = None
            did_checkpoint = False
            forced_checkpoint = False
            appended_record = False
            retained_tail = False

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
                suffix = "prefill" if pending.initial_state is None else "chunk"
                action = f"{self.policy.mode}_checkpoint_{suffix}"
                reason = "prefill_or_uncached" if pending.initial_state is None else "multi_token"
                did_checkpoint = True
                forced_checkpoint = True
            else:
                if (
                    pending.normalized_key is None
                    or pending.update is None
                    or pending.log_decay is None
                ):
                    raise RuntimeError("single-token StateLease observation lacks a replay record")
                if previous_packed is None:
                    raise RuntimeError("fixed replay cannot append without a checkpoint")

                checkpoint_now = current_count == self.policy.checkpoint_period - 1
                if current_count > self.policy.checkpoint_period - 1:
                    raise RuntimeError(
                        f"{self.policy.mode} reached invalid replay age {current_count}"
                    )

                if not checkpoint_now:
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
                    action = f"{self.policy.mode}_replay_append"
                    reason = "fixed_period_not_reached"
                    appended_record = True
                else:
                    # Materialize the chronological rounded record in the
                    # charged slot before either clearing it or retaining it.
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
                    if int(valid_count.item()) != self.policy.checkpoint_period:
                        raise RuntimeError(
                            f"{self.policy.mode} did not reach its checkpoint period"
                        )
                    if self.policy.retain_last_record:
                        if pending.initial_state is None:
                            raise RuntimeError("fixed_cut4_in5 requires an incoming S4 state")
                        retained_key = key_buffer[current_count].clone()
                        retained_update = update_buffer[current_count].clone()
                        retained_decay = decay_buffer[current_count].clone()
                        chosen_packed, _ = self._pack_checkpoint(
                            pending.initial_state,
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
                        self._append_candidate_record(
                            key_buffer=key_buffer,
                            update_buffer=update_buffer,
                            decay_buffer=decay_buffer,
                            valid_count=valid_count,
                            slot=0,
                            normalized_key=retained_key,
                            update=retained_update,
                            log_decay=retained_decay,
                        )
                        materialized = self._materialize_candidate(
                            packed=chosen_packed,
                            key_buffer=key_buffer,
                            update_buffer=update_buffer,
                            decay_buffer=decay_buffer,
                            valid_count=valid_count,
                        )
                        boundary = 4
                        action = "fixed_cut4_in5_boundary_4"
                        reason = "fixed_full_buffer_cut4"
                        retained_tail = True
                    else:
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
                        boundary = self.policy.checkpoint_period
                        action = f"{self.policy.mode}_checkpoint"
                        reason = "fixed_period_reached"
                    did_checkpoint = True

            replay_valid_count = int(valid_count.item())
            if (
                tuple(materialized.shape) != self._expected_state_shape()
                or not torch.isfinite(materialized).all().item()
            ):
                raise RuntimeError("fixed replay resident state candidate is invalid")
            evidence = self._build_evidence(
                final_state=recurrent_states,
                materialized=materialized,
                packed=chosen_packed,
                pending=pending,
                action=action,
                boundary=boundary,
                tie=False,
                cut4_risk=None,
                cut5_risk=None,
                replay_valid_count=replay_valid_count,
                query_ema=pending.candidate_ema,
                key_buffer=key_buffer,
                update_buffer=update_buffer,
                decay_buffer=decay_buffer,
                valid_count=valid_count,
            )

            # Candidate tensors are clones.  Commit references only after all
            # validation, packing, replay, and evidence construction succeed.
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
            if did_checkpoint:
                self.checkpoint_count += 1
            if forced_checkpoint:
                self.forced_checkpoint_count += 1
            elif did_checkpoint:
                self.scheduled_checkpoint_count += 1
            if appended_record:
                self.replay_append_count += 1
            if retained_tail:
                self.retained_tail_count += 1
            if boundary == 4:
                self.boundary4_count += 1
            elif boundary == 5:
                self.boundary5_count += 1
            self.last_action = action
            self.last_reason = reason
            self.last_boundary = boundary
            self.last_cut4_risk = None
            self.last_cut5_risk = None
            self.last_update_evidence = evidence
            self._pending_statelease_observation = None
            return materialized
        except Exception:
            self.discard_pending_statelease_observation()
            raise

    def statelease_diagnostics(
        self,
    ) -> dict[str, int | float | bool | str | None]:
        diagnostics = super().statelease_diagnostics()
        diagnostics.update(
            {
                "baseline_mode": self.policy.mode,
                "checkpoint_period": self.policy.checkpoint_period,
                "equal_allocation": self.policy.equal_allocation,
                "off_budget": self.policy.off_budget,
                "retains_last_record": self.policy.retain_last_record,
                "scheduled_checkpoint_count": self.scheduled_checkpoint_count,
                "forced_checkpoint_count": self.forced_checkpoint_count,
                "replay_append_count": self.replay_append_count,
                "retained_tail_count": self.retained_tail_count,
                "logical_replay_capacity_bytes": self.logical_replay_capacity_bytes(),
                "replay_capacity_allocated": (
                    self.replay_capacity_bytes() == self.logical_replay_capacity_bytes()
                ),
            }
        )
        return diagnostics

    def reset(self) -> None:
        super().reset()
        self.scheduled_checkpoint_count = 0
        self.forced_checkpoint_count = 0
        self.replay_append_count = 0
        self.retained_tail_count = 0


class FixedReplayRecurrentStateCache(StateLeaseRecurrentStateCache):
    """A Qwen3.5 cache using one of Experiment 010's fixed schedules."""

    def __init__(
        self,
        config: object,
        *,
        plan: ExactBudgetRowPlan,
        mode: FixedReplayMode | str,
        rounding: RoundingMode = "nearest",
        seed: int = 2339,
        record_evidence: bool = False,
        selection_method: str | None = None,
        experiment_identity_sha256: str | None = None,
    ) -> None:
        policy = fixed_replay_policy(mode)
        effective_selection_method = selection_method or (
            f"{policy.mode}_right_rht_query_ema32_weighted_mse_configurable_quota"
        )
        super().__init__(
            config,
            plan=plan,
            rounding=rounding,
            seed=seed,
            record_evidence=record_evidence,
            selection_method=effective_selection_method,
            experiment_identity_sha256=experiment_identity_sha256,
        )
        replacements = list(self.statelease_layers())
        if not replacements:
            raise TypeError("config did not create any linear-attention layers")
        for layer_index, layer in replacements:
            self.layers[layer_index] = FixedReplayLinearAttentionLayer(
                policy=policy,
                selection_method=effective_selection_method,
                low_spec=layer.low_spec,
                high_spec=layer.high_spec,
                layer_index=layer.layer_index,
                expected_heads=layer.expected_heads,
                expected_rows=layer.expected_rows,
                high_precision_group_indices=layer.high_precision_group_indices,
                number_of_states=layer.number_of_states,
            )
        self.policy = policy
        self.selection_method = effective_selection_method

    def fixed_replay_layers(
        self,
    ) -> tuple[tuple[int, FixedReplayLinearAttentionLayer], ...]:
        layers: list[tuple[int, FixedReplayLinearAttentionLayer]] = []
        for layer_index, layer in self.statelease_layers():
            if not isinstance(layer, FixedReplayLinearAttentionLayer):
                raise RuntimeError(f"layer {layer_index} is not a fixed replay layer")
            layers.append((layer_index, layer))
        return tuple(layers)

    def logical_replay_capacity_bytes(self) -> int:
        return sum(layer.logical_replay_capacity_bytes() for _, layer in self.fixed_replay_layers())

    def logical_query_ema_bytes(self) -> int:
        return sum(
            layer.expected_heads * layer.expected_rows * 4
            for _, layer in self.fixed_replay_layers()
        )

    def logical_resident_capacity_bytes(self) -> int:
        return (
            self.plan.resident_bytes
            + self.logical_query_ema_bytes()
            + self.logical_replay_capacity_bytes()
        )

    def storage_summary(self) -> dict[str, object]:
        summary: dict[str, object] = dict(super().storage_summary())
        summary.update(
            {
                "baseline_mode": self.policy.mode,
                "checkpoint_period": self.policy.checkpoint_period,
                "replay_capacity": self.policy.replay_capacity,
                "equal_allocation": self.policy.equal_allocation,
                "off_budget": self.policy.off_budget,
                "logical_checkpoint_bytes": self.plan.resident_bytes,
                "logical_query_ema_bytes": self.logical_query_ema_bytes(),
                "logical_replay_capacity_bytes": self.logical_replay_capacity_bytes(),
                "logical_resident_capacity_bytes": self.logical_resident_capacity_bytes(),
                "capacity_fully_allocated": (
                    self.replay_capacity_bytes() == self.logical_replay_capacity_bytes()
                    and self.query_ema_bytes() == self.logical_query_ema_bytes()
                    and self.resident_checkpoint_bytes() == self.plan.resident_bytes
                ),
            }
        )
        return summary

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        valid_identity = (
            isinstance(beam_idx, torch.Tensor)
            and beam_idx.dtype == torch.long
            and beam_idx.ndim == 1
            and beam_idx.numel() == 1
            and int(beam_idx.item()) == 0
        )
        if not valid_identity:
            for _, layer in self.fixed_replay_layers():
                layer.discard_pending_statelease_observation()
            raise ValueError(
                "fixed replay is batch-one only and supports only identity beam_idx=[0]"
            )
        super().reorder_cache(beam_idx)


class FixedCC1RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cc1", **kwargs)


class FixedCC2RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cc2", **kwargs)


class FixedCC4RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cc4", **kwargs)


class FixedCC5RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cc5", **kwargs)


class FixedCut4In5RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cut4_in5", **kwargs)


class FixedCC8RecurrentStateCache(FixedReplayRecurrentStateCache):
    def __init__(self, config: object, *, plan: ExactBudgetRowPlan, **kwargs: object) -> None:
        super().__init__(config, plan=plan, mode="fixed_cc8", **kwargs)


def create_fixed_replay_cache(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    mode: FixedReplayMode | str,
    rounding: RoundingMode = "nearest",
    seed: int = 2339,
    record_evidence: bool = False,
) -> FixedReplayRecurrentStateCache:
    """Create a configurable fixed replay research baseline by schedule mode."""

    constructors = {
        "fixed_cc1": FixedCC1RecurrentStateCache,
        "fixed_cc2": FixedCC2RecurrentStateCache,
        "fixed_cc4": FixedCC4RecurrentStateCache,
        "fixed_cc5": FixedCC5RecurrentStateCache,
        "fixed_cut4_in5": FixedCut4In5RecurrentStateCache,
        "fixed_cc8": FixedCC8RecurrentStateCache,
    }
    policy = fixed_replay_policy(mode)
    constructor = constructors[policy.mode]
    return constructor(
        config,
        plan=plan,
        rounding=rounding,
        seed=seed,
        record_evidence=record_evidence,
    )


def fixed_cc1(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCC1RecurrentStateCache:
    return FixedCC1RecurrentStateCache(config, plan=plan, **kwargs)


def fixed_cc2(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCC2RecurrentStateCache:
    return FixedCC2RecurrentStateCache(config, plan=plan, **kwargs)


def fixed_cc4(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCC4RecurrentStateCache:
    return FixedCC4RecurrentStateCache(config, plan=plan, **kwargs)


def fixed_cc5(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCC5RecurrentStateCache:
    return FixedCC5RecurrentStateCache(config, plan=plan, **kwargs)


def fixed_cut4_in5(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCut4In5RecurrentStateCache:
    return FixedCut4In5RecurrentStateCache(config, plan=plan, **kwargs)


def fixed_cc8(
    config: object,
    *,
    plan: ExactBudgetRowPlan,
    **kwargs: object,
) -> FixedCC8RecurrentStateCache:
    return FixedCC8RecurrentStateCache(config, plan=plan, **kwargs)


__all__ = [
    "EQUAL_ALLOCATION_REPLAY_CAPACITY",
    "OFF_BUDGET_CC8_REPLAY_CAPACITY",
    "FixedCC1RecurrentStateCache",
    "FixedCC2RecurrentStateCache",
    "FixedCC4RecurrentStateCache",
    "FixedCC5RecurrentStateCache",
    "FixedCC8RecurrentStateCache",
    "FixedCut4In5RecurrentStateCache",
    "FixedReplayLinearAttentionLayer",
    "FixedReplayMode",
    "FixedReplayPolicy",
    "FixedReplayRecurrentStateCache",
    "create_fixed_replay_cache",
    "fixed_cc1",
    "fixed_cc2",
    "fixed_cc4",
    "fixed_cc5",
    "fixed_cc8",
    "fixed_cut4_in5",
    "fixed_replay_policy",
]
