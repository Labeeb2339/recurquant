"""Fail-closed Experiment 011 StateLease evaluation primitives.

This module contains no dataset or model-loading code.  It provides the
reference-aligned recurrent trajectory metric and the frozen Stage-A gate so
their numerical and selection rules can be tested before the one permitted
quality run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import torch

STATELEASE_METHOD = "statelease_h5"
RHT_CQER_METHOD = "rht_cqer32"
FIXED_REPLAY_METHODS = (
    "fixed_cc1",
    "fixed_cc2",
    "fixed_cc4",
    "fixed_cc5",
    "fixed_cut4_in5",
)
EQUAL_BYTE_NO_REPLAY_METHODS = (
    "expanded_rht_q4_q8",
    "rht_q4_q6_q8",
    "rht_residual_q4",
)
STAGE_A_EQUAL_BYTE_METHODS = FIXED_REPLAY_METHODS + EQUAL_BYTE_NO_REPLAY_METHODS
STAGE_A_REQUIRED_METHODS = (
    RHT_CQER_METHOD,
    STATELEASE_METHOD,
    *STAGE_A_EQUAL_BYTE_METHODS,
)

FROZEN_STATELEASE_RESIDENT_BYTES = 3_454_664
FROZEN_STAGE_A_PROMPT_TOKENS = 69
FROZEN_STAGE_A_ALIGNED_TOKENS = 38
FROZEN_STAGE_A_RECURRENT_LAYER_INDICES = (
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
FROZEN_STAGE_A_FORWARD_COUNT = 1 + FROZEN_STAGE_A_ALIGNED_TOKENS
FROZEN_STAGE_A_TOKENS_OBSERVED = FROZEN_STAGE_A_PROMPT_TOKENS + FROZEN_STAGE_A_ALIGNED_TOKENS
FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES = FROZEN_STAGE_A_ALIGNED_TOKENS * len(
    FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
)
FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS = FROZEN_STAGE_A_FORWARD_COUNT * len(
    FROZEN_STAGE_A_RECURRENT_LAYER_INDICES
)
MINIMUM_CC1_EXCESS_NLL_REDUCTION = 0.10
MAXIMUM_STRONGEST_FIXED_RELATIVE_DISADVANTAGE = 0.05
MAXIMUM_TOP1_TRAIL = 0.01


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _state_mapping(
    states: Mapping[int, torch.Tensor],
    *,
    context: str,
) -> dict[int, torch.Tensor]:
    if not isinstance(states, Mapping) or not states:
        raise ValueError(f"{context} must be a non-empty layer mapping")
    normalized: dict[int, torch.Tensor] = {}
    for layer_index, tensor in states.items():
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError(f"{context} layer indices must be integers")
        if layer_index < 0:
            raise ValueError(f"{context} layer indices must be non-negative")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{context}[{layer_index}] must be a tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{context}[{layer_index}] must be floating point")
        if tensor.numel() == 0:
            raise ValueError(f"{context}[{layer_index}] must be non-empty")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{context}[{layer_index}] must be finite")
        normalized[layer_index] = tensor
    return normalized


def reference_aligned_trajectory_nmse(
    reference_states: Mapping[int, torch.Tensor],
    candidate_states: Mapping[int, torch.Tensor],
    *,
    denominator_epsilon: float = 1e-12,
) -> dict[int, float]:
    """Return FP64 trajectory NMSE for each recurrent layer at one write.

    The candidate is compared with the matched FP32 trajectory, not with its
    own pre-quantization state.  Inputs may use lower floating dtypes; all
    accumulation is performed in FP64.
    """

    epsilon = _finite_float(
        denominator_epsilon,
        context="denominator_epsilon",
    )
    if epsilon <= 0:
        raise ValueError("denominator_epsilon must be positive")
    reference = _state_mapping(reference_states, context="reference_states")
    candidate = _state_mapping(candidate_states, context="candidate_states")
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            "candidate recurrent layers do not match the reference: "
            f"missing={missing}, extra={extra}"
        )

    result: dict[int, float] = {}
    for layer_index in sorted(reference):
        reference_tensor = reference[layer_index]
        candidate_tensor = candidate[layer_index]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"layer {layer_index} shape mismatch: "
                f"{tuple(reference_tensor.shape)} != {tuple(candidate_tensor.shape)}"
            )
        if reference_tensor.device != candidate_tensor.device:
            raise ValueError(f"layer {layer_index} reference and candidate devices differ")
        reference_fp64 = reference_tensor.detach().to(torch.float64)
        error_fp64 = candidate_tensor.detach().to(torch.float64) - reference_fp64
        numerator = torch.sum(error_fp64.square(), dtype=torch.float64)
        denominator = torch.sum(reference_fp64.square(), dtype=torch.float64) + epsilon
        value = float((numerator / denominator).item())
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"layer {layer_index} trajectory NMSE is invalid")
        result[layer_index] = value
    return result


@dataclass(slots=True)
class TrajectoryNmseAccumulator:
    """Accumulate the protocol's layer-and-write macro in FP64."""

    _sum: float = 0.0
    _compensation: float = 0.0
    _write_count: int = 0
    _layer_value_count: int = 0

    def append(self, layer_nmse: Mapping[int, Real]) -> None:
        if not isinstance(layer_nmse, Mapping) or not layer_nmse:
            raise ValueError("layer_nmse must be a non-empty mapping")
        values: list[float] = []
        for layer_index in sorted(layer_nmse):
            if isinstance(layer_index, bool) or not isinstance(layer_index, int):
                raise TypeError("trajectory layer indices must be integers")
            value = _finite_float(
                layer_nmse[layer_index],
                context=f"trajectory layer {layer_index}",
            )
            if value < 0:
                raise ValueError("trajectory NMSE values must be non-negative")
            values.append(value)

        # Neumaier compensation makes the scalar accumulation insensitive to
        # common large/small-value cancellation without retaining every state.
        for value in values:
            provisional = self._sum + value
            if abs(self._sum) >= abs(value):
                self._compensation += (self._sum - provisional) + value
            else:
                self._compensation += (value - provisional) + self._sum
            self._sum = provisional
        self._write_count += 1
        self._layer_value_count += len(values)

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def layer_value_count(self) -> int:
        return self._layer_value_count

    def summary(self) -> dict[str, float | int]:
        if self._write_count == 0 or self._layer_value_count == 0:
            raise RuntimeError("trajectory accumulator contains no scored writes")
        total = self._sum + self._compensation
        mean = total / self._layer_value_count
        if not math.isfinite(mean) or mean < 0:
            raise RuntimeError("trajectory NMSE aggregate is invalid")
        return {
            "trajectory_nmse_auc": mean,
            "scored_write_count": self._write_count,
            "layer_value_count": self._layer_value_count,
        }


def _metric_rows(
    metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float | int | bool]]:
    required_methods = set(STAGE_A_REQUIRED_METHODS)
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    missing = sorted(required_methods - set(metrics))
    extra = sorted(set(metrics) - required_methods)
    if missing or extra:
        raise ValueError(
            f"Stage-A metric methods differ from the frozen set: missing={missing}, extra={extra}"
        )

    normalized: dict[str, dict[str, float | int | bool]] = {}
    for method in sorted(required_methods):
        row = metrics[method]
        if not isinstance(row, Mapping):
            raise TypeError(f"metrics[{method!r}] must be a mapping")
        candidate_nll = _finite_float(
            row.get("candidate_nll"),
            context=f"{method} candidate_nll",
        )
        reference_nll = _finite_float(
            row.get("reference_nll"),
            context=f"{method} reference_nll",
        )
        delta_nll = _finite_float(
            row.get("delta_nll"),
            context=f"{method} delta_nll",
        )
        top1 = _finite_float(
            row.get("top1_agreement"),
            context=f"{method} top1_agreement",
        )
        token_count = _strict_int(
            row.get("token_count"),
            context=f"{method} token_count",
        )
        finite_flag = row.get("all_logits_finite")
        if not isinstance(finite_flag, bool):
            raise TypeError(f"{method} all_logits_finite must be a bool")
        if candidate_nll < 0 or reference_nll < 0:
            raise ValueError(f"{method} NLL values must be non-negative")
        if not math.isclose(
            candidate_nll - reference_nll,
            delta_nll,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(f"{method} delta_nll is inconsistent with its NLL values")
        if not 0 <= top1 <= 1:
            raise ValueError(f"{method} top1_agreement must lie in [0, 1]")
        if token_count != FROZEN_STAGE_A_ALIGNED_TOKENS:
            raise ValueError(
                f"{method} token_count must equal the frozen Stage-A count "
                f"{FROZEN_STAGE_A_ALIGNED_TOKENS}; got {token_count}"
            )
        normalized[method] = {
            "candidate_nll": candidate_nll,
            "reference_nll": reference_nll,
            "delta_nll": delta_nll,
            "top1_agreement": top1,
            "token_count": token_count,
            "all_logits_finite": finite_flag,
        }

    token_counts = {int(row["token_count"]) for row in normalized.values()}
    reference_nlls = {float(row["reference_nll"]) for row in normalized.values()}
    if len(token_counts) != 1:
        raise ValueError("Stage-A methods do not score the same number of tokens")
    if len(reference_nlls) != 1:
        raise ValueError("Stage-A methods do not share an identical reference NLL")
    return normalized


def _trajectory_rows(
    trajectory: Mapping[str, object],
) -> dict[str, float]:
    required_methods = set(STAGE_A_REQUIRED_METHODS)
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be a mapping")
    missing = sorted(required_methods - set(trajectory))
    extra = sorted(set(trajectory) - required_methods)
    if missing or extra:
        raise ValueError(
            f"Stage-A trajectory methods differ from the frozen set: "
            f"missing={missing}, extra={extra}"
        )
    result: dict[str, float] = {}
    for method in sorted(required_methods):
        row = trajectory[method]
        if not isinstance(row, Mapping):
            raise TypeError(f"trajectory[{method!r}] must be a mapping")
        normalized = _finite_float(
            row.get("trajectory_nmse_auc"),
            context=f"{method} trajectory_nmse_auc",
        )
        if normalized < 0:
            raise ValueError("trajectory NMSE AUC must be non-negative")
        scored_write_count = _strict_int(
            row.get("scored_write_count"),
            context=f"{method} scored_write_count",
        )
        layer_value_count = _strict_int(
            row.get("layer_value_count"),
            context=f"{method} layer_value_count",
        )
        if scored_write_count != FROZEN_STAGE_A_ALIGNED_TOKENS:
            raise ValueError(
                f"{method} scored_write_count must equal "
                f"{FROZEN_STAGE_A_ALIGNED_TOKENS}; got {scored_write_count}"
            )
        if layer_value_count != FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES:
            raise ValueError(
                f"{method} layer_value_count must equal "
                f"{FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES}; got {layer_value_count}"
            )
        result[method] = normalized
    return result


def _gate_check(function: object) -> dict[str, object]:
    if not callable(function):
        raise TypeError("gate check must be callable")
    try:
        evidence = function()
        if not isinstance(evidence, Mapping):
            raise TypeError("gate check evidence must be a mapping")
        return {"passed": True, "evidence": dict(evidence), "error": None}
    except (TypeError, ValueError, RuntimeError) as error:
        return {
            "passed": False,
            "evidence": None,
            "error": f"{type(error).__name__}: {error}",
        }


def evaluate_statelease_stage_a_gate(
    *,
    aligned_metrics: Mapping[str, Mapping[str, object]],
    trajectory_nmse_auc: Mapping[str, object],
    statelease_storage: Mapping[str, object],
    statelease_diagnostics: Sequence[Mapping[str, object]],
    statelease_update_evidence: Sequence[Mapping[str, object]],
    stage0_complete: bool,
    artifact_integrity: bool,
) -> dict[str, object]:
    """Evaluate every frozen Experiment 011 Stage-A condition conjunctively."""

    metrics: dict[str, dict[str, float | int | bool]] | None = None
    trajectory: dict[str, float] | None = None

    def inputs_check() -> dict[str, object]:
        nonlocal metrics, trajectory
        if not isinstance(stage0_complete, bool):
            raise TypeError("stage0_complete must be a bool")
        if not isinstance(artifact_integrity, bool):
            raise TypeError("artifact_integrity must be a bool")
        if not stage0_complete:
            raise ValueError("authenticated production Stage 0 is incomplete")
        if not artifact_integrity:
            raise ValueError("Stage-A artifact integrity checks did not pass")
        metrics = _metric_rows(aligned_metrics)
        trajectory = _trajectory_rows(trajectory_nmse_auc)
        return {
            "stage0_complete": stage0_complete,
            "artifact_integrity": artifact_integrity,
            "method_count": len(metrics),
        }

    def storage_check() -> dict[str, object]:
        if not isinstance(statelease_storage, Mapping):
            raise TypeError("statelease_storage must be a mapping")
        allocated = _strict_int(
            statelease_storage.get("resident_bytes_including_statelease"),
            context="StateLease allocated resident bytes",
        )
        hidden_fp32 = statelease_storage.get("persistent_fp32_state_mirror")
        if allocated != FROZEN_STATELEASE_RESIDENT_BYTES:
            raise ValueError(
                f"StateLease allocated {allocated} bytes; expected "
                f"{FROZEN_STATELEASE_RESIDENT_BYTES}"
            )
        if hidden_fp32 is not False:
            raise ValueError("StateLease persistent_fp32_state_mirror must be explicitly false")
        return {
            "allocated_resident_bytes": allocated,
            "expected_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
            "persistent_fp32_state_mirror": False,
        }

    def boundary_check() -> dict[str, object]:
        if (
            not isinstance(statelease_diagnostics, Sequence)
            or isinstance(statelease_diagnostics, (str, bytes))
            or len(statelease_diagnostics) != len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)
        ):
            raise ValueError(
                "statelease_diagnostics must contain exactly "
                f"{len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)} recurrent layers"
            )
        diagnostic_boundary4 = diagnostic_boundary5 = diagnostic_ties = 0
        diagnostic_layers: set[int] = set()
        for index, row in enumerate(statelease_diagnostics):
            if not isinstance(row, Mapping):
                raise TypeError(f"statelease_diagnostics[{index}] must be a mapping")
            layer_index = _strict_int(
                row.get("layer_index"),
                context=f"diagnostics[{index}] layer_index",
            )
            if layer_index not in FROZEN_STAGE_A_RECURRENT_LAYER_INDICES:
                raise ValueError(
                    f"diagnostics[{index}] layer_index {layer_index} is not a frozen "
                    "Qwen3.5 recurrent layer"
                )
            if layer_index in diagnostic_layers:
                raise ValueError(
                    f"statelease_diagnostics contains duplicate layer_index {layer_index}"
                )
            diagnostic_layers.add(layer_index)
            state_updates = _strict_int(
                row.get("state_updates"),
                context=f"diagnostics[{index}] state_updates",
            )
            tokens_observed = _strict_int(
                row.get("tokens_observed"),
                context=f"diagnostics[{index}] tokens_observed",
            )
            if state_updates != FROZEN_STAGE_A_FORWARD_COUNT:
                raise ValueError(
                    f"diagnostics[{index}] state_updates must equal "
                    f"{FROZEN_STAGE_A_FORWARD_COUNT}; got {state_updates}"
                )
            if tokens_observed != FROZEN_STAGE_A_TOKENS_OBSERVED:
                raise ValueError(
                    f"diagnostics[{index}] tokens_observed must equal "
                    f"{FROZEN_STAGE_A_TOKENS_OBSERVED}; got {tokens_observed}"
                )
            b4 = _strict_int(
                row.get("boundary4_count"),
                context=f"diagnostics[{index}] boundary4_count",
            )
            b5 = _strict_int(
                row.get("boundary5_count"),
                context=f"diagnostics[{index}] boundary5_count",
            )
            tie_count = _strict_int(
                row.get("tie_count"),
                context=f"diagnostics[{index}] tie_count",
            )
            for name, value in (
                ("boundary4_count", b4),
                ("boundary5_count", b5),
                ("tie_count", tie_count),
            ):
                if value < 0:
                    raise ValueError(f"diagnostics[{index}] {name} is negative")
            invalid = row.get("invalid_boundary_count", 0)
            invalid_count = _strict_int(
                invalid,
                context=f"diagnostics[{index}] invalid_boundary_count",
            )
            if invalid_count != 0:
                raise ValueError("StateLease reports a boundary outside c4/c5")
            diagnostic_boundary4 += b4
            diagnostic_boundary5 += b5
            diagnostic_ties += tie_count

        if diagnostic_layers != set(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES):
            raise ValueError(
                "statelease_diagnostics does not contain the exact frozen recurrent-layer set"
            )

        if (
            not isinstance(statelease_update_evidence, Sequence)
            or isinstance(statelease_update_evidence, (str, bytes))
            or len(statelease_update_evidence) != FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS
        ):
            raise ValueError(
                "statelease_update_evidence must contain exactly "
                f"{FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS} layer-write records"
            )
        evidence_boundary4 = evidence_boundary5 = evidence_ties = 0
        for index, row in enumerate(statelease_update_evidence):
            if not isinstance(row, Mapping):
                raise TypeError(f"statelease_update_evidence[{index}] must be a mapping")
            update_index = _strict_int(
                row.get("update_index"),
                context=f"statelease_update_evidence[{index}] update_index",
            )
            if update_index != index:
                raise ValueError(
                    f"statelease_update_evidence[{index}] update_index must equal {index}; "
                    f"got {update_index}"
                )
            expected_layer = FROZEN_STAGE_A_RECURRENT_LAYER_INDICES[
                index % len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)
            ]
            layer_index = _strict_int(
                row.get("layer_index"),
                context=f"statelease_update_evidence[{index}] layer_index",
            )
            if layer_index != expected_layer:
                raise ValueError(
                    f"statelease_update_evidence[{index}] layer_index must equal "
                    f"{expected_layer}; got {layer_index}"
                )
            state_index = _strict_int(
                row.get("state_index"),
                context=f"statelease_update_evidence[{index}] state_index",
            )
            if state_index != 0:
                raise ValueError(f"statelease_update_evidence[{index}] state_index must equal 0")
            forward_index = index // len(FROZEN_STAGE_A_RECURRENT_LAYER_INDICES)
            expected_token_count = FROZEN_STAGE_A_PROMPT_TOKENS if forward_index == 0 else 1
            token_count = _strict_int(
                row.get("token_count"),
                context=f"statelease_update_evidence[{index}] token_count",
            )
            if token_count != expected_token_count:
                raise ValueError(
                    f"statelease_update_evidence[{index}] token_count must equal "
                    f"{expected_token_count}; got {token_count}"
                )
            boundary = row.get("boundary")
            tie = row.get("tie")
            if not isinstance(tie, bool):
                raise TypeError(f"statelease_update_evidence[{index}] tie must be a bool")
            if boundary is not None:
                boundary = _strict_int(
                    boundary,
                    context=f"statelease_update_evidence[{index}] boundary",
                )
                if boundary not in (4, 5):
                    raise ValueError("StateLease reports a checkpoint interval outside c4/c5")
                if boundary == 4:
                    evidence_boundary4 += 1
                else:
                    evidence_boundary5 += 1
            if forward_index == 0 and boundary is not None:
                raise ValueError("StateLease prefill evidence must not report a boundary")
            if tie:
                evidence_ties += 1
                if boundary != 5:
                    raise ValueError("a StateLease tie was not assigned to boundary c5")

        if (
            evidence_boundary4 != diagnostic_boundary4
            or evidence_boundary5 != diagnostic_boundary5
            or evidence_ties != diagnostic_ties
        ):
            raise ValueError("StateLease boundary evidence does not match per-layer diagnostics")
        if evidence_boundary4 + evidence_boundary5 <= 0:
            raise ValueError("Stage A observed no full-buffer StateLease decision")
        return {
            "boundary4_count": evidence_boundary4,
            "boundary5_count": evidence_boundary5,
            "tie_count": evidence_ties,
            "ties_assigned_to_c5": True,
            "boundary_records_authenticated": True,
        }

    def cc1_reduction_check() -> dict[str, object]:
        if metrics is None:
            raise RuntimeError("normalized metrics are unavailable")
        candidate = float(metrics[STATELEASE_METHOD]["delta_nll"])
        baseline = float(metrics["fixed_cc1"]["delta_nll"])
        if baseline <= 0:
            raise ValueError("fixed_cc1 excess NLL is non-positive; relative gate fails closed")
        reduction = (baseline - candidate) / baseline
        if reduction < MINIMUM_CC1_EXCESS_NLL_REDUCTION and not math.isclose(
            reduction,
            MINIMUM_CC1_EXCESS_NLL_REDUCTION,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("StateLease excess-NLL reduction versus fixed_cc1 is below 10%")
        return {
            "statelease_excess_nll": candidate,
            "fixed_cc1_excess_nll": baseline,
            "relative_reduction": reduction,
            "minimum_relative_reduction": MINIMUM_CC1_EXCESS_NLL_REDUCTION,
        }

    def strongest_fixed_check() -> dict[str, object]:
        if metrics is None:
            raise RuntimeError("normalized metrics are unavailable")
        strongest = min(
            FIXED_REPLAY_METHODS,
            key=lambda method: float(metrics[method]["delta_nll"]),
        )
        baseline = float(metrics[strongest]["delta_nll"])
        candidate = float(metrics[STATELEASE_METHOD]["delta_nll"])
        if baseline <= 0:
            raise ValueError(
                "strongest fixed replay excess NLL is non-positive; relative gate fails closed"
            )
        relative_disadvantage = (candidate - baseline) / baseline
        if (
            relative_disadvantage > MAXIMUM_STRONGEST_FIXED_RELATIVE_DISADVANTAGE
            and not math.isclose(
                relative_disadvantage,
                MAXIMUM_STRONGEST_FIXED_RELATIVE_DISADVANTAGE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "StateLease excess NLL is more than 5% worse than the strongest fixed replay"
            )
        return {
            "strongest_fixed_method": strongest,
            "statelease_excess_nll": candidate,
            "strongest_fixed_excess_nll": baseline,
            "relative_disadvantage": relative_disadvantage,
            "maximum_relative_disadvantage": (MAXIMUM_STRONGEST_FIXED_RELATIVE_DISADVANTAGE),
        }

    def trajectory_check() -> dict[str, object]:
        if trajectory is None:
            raise RuntimeError("normalized trajectory metrics are unavailable")
        candidate = trajectory[STATELEASE_METHOD]
        baseline = trajectory["fixed_cc1"]
        if not candidate < baseline:
            raise ValueError("StateLease trajectory NMSE AUC is not lower than fixed_cc1")
        return {
            "statelease_trajectory_nmse_auc": candidate,
            "fixed_cc1_trajectory_nmse_auc": baseline,
            "strictly_lower": True,
        }

    def top1_check() -> dict[str, object]:
        if metrics is None:
            raise RuntimeError("normalized metrics are unavailable")
        best_method = max(
            FIXED_REPLAY_METHODS,
            key=lambda method: float(metrics[method]["top1_agreement"]),
        )
        candidate = float(metrics[STATELEASE_METHOD]["top1_agreement"])
        best = float(metrics[best_method]["top1_agreement"])
        trail = best - candidate
        if trail > MAXIMUM_TOP1_TRAIL and not math.isclose(
            trail,
            MAXIMUM_TOP1_TRAIL,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("StateLease top-1 agreement trails the best fixed replay by over 0.01")
        return {
            "best_fixed_method": best_method,
            "statelease_top1_agreement": candidate,
            "best_fixed_top1_agreement": best,
            "trail": trail,
            "maximum_trail": MAXIMUM_TOP1_TRAIL,
        }

    def finiteness_check() -> dict[str, object]:
        if metrics is None or trajectory is None:
            raise RuntimeError("normalized Stage-A values are unavailable")
        nonfinite_flags = sorted(
            method for method, row in metrics.items() if row["all_logits_finite"] is not True
        )
        if nonfinite_flags:
            raise ValueError(f"methods reported non-finite logits: {nonfinite_flags}")
        return {
            "all_metric_scalars_finite": True,
            "all_trajectory_scalars_finite": True,
            "all_logits_finite": True,
        }

    checks = {
        "stage0_and_artifact_integrity": _gate_check(inputs_check),
        "exact_statelease_allocation": _gate_check(storage_check),
        "only_c4_c5_and_ties_to_c5": _gate_check(boundary_check),
        "cc1_excess_nll_reduction_at_least_10_percent": _gate_check(cc1_reduction_check),
        "no_more_than_5_percent_worse_than_strongest_fixed": _gate_check(strongest_fixed_check),
        "trajectory_nmse_auc_lower_than_cc1": _gate_check(trajectory_check),
        "top1_trail_at_most_0_01": _gate_check(top1_check),
        "all_primary_values_finite": _gate_check(finiteness_check),
    }
    return {
        "passed": all(check["passed"] is True for check in checks.values()),
        "checks": checks,
        "thresholds": {
            "statelease_resident_bytes": FROZEN_STATELEASE_RESIDENT_BYTES,
            "minimum_cc1_excess_nll_reduction": (MINIMUM_CC1_EXCESS_NLL_REDUCTION),
            "maximum_strongest_fixed_relative_disadvantage": (
                MAXIMUM_STRONGEST_FIXED_RELATIVE_DISADVANTAGE
            ),
            "maximum_top1_trail": MAXIMUM_TOP1_TRAIL,
        },
        "method_sets": {
            "historical_anchor": RHT_CQER_METHOD,
            "statelease": STATELEASE_METHOD,
            "fixed_replay": list(FIXED_REPLAY_METHODS),
            "equal_byte_no_replay": list(EQUAL_BYTE_NO_REPLAY_METHODS),
        },
    }


__all__ = [
    "EQUAL_BYTE_NO_REPLAY_METHODS",
    "FIXED_REPLAY_METHODS",
    "FROZEN_STAGE_A_ALIGNED_TOKENS",
    "FROZEN_STAGE_A_FORWARD_COUNT",
    "FROZEN_STAGE_A_PROMPT_TOKENS",
    "FROZEN_STAGE_A_RECURRENT_LAYER_INDICES",
    "FROZEN_STAGE_A_TOKENS_OBSERVED",
    "FROZEN_STAGE_A_TRAJECTORY_LAYER_VALUES",
    "FROZEN_STAGE_A_UPDATE_EVIDENCE_RECORDS",
    "FROZEN_STATELEASE_RESIDENT_BYTES",
    "RHT_CQER_METHOD",
    "STAGE_A_EQUAL_BYTE_METHODS",
    "STAGE_A_REQUIRED_METHODS",
    "STATELEASE_METHOD",
    "TrajectoryNmseAccumulator",
    "evaluate_statelease_stage_a_gate",
    "reference_aligned_trajectory_nmse",
]
