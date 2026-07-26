"""Bounded physical-cache interventions for validating row rankings.

This module deliberately implements an expensive analysis oracle, not a
deployable selector.  It reruns a fixed teacher-forced sequence once for an
all-INT4/background control and once per candidate row.  Every candidate run
uses a fresh :class:`MixedPackedRecurrentStateCache`, so the measured metric
includes the real repeated INT4/INT8 packing and dequantization path.

Candidate metrics use target tokens and can therefore overfit calibration
data.  The results are appropriate for auditing a scalable sensitivity proxy
on a small, predeclared candidate set; they are not an inference-time policy or
confirmation evidence by themselves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .qwen35 import create_qwen35_exact_budget_cache
from .row_policy import ExactBudgetRowPlan, RowLocation, select_rows_exact_budget

TokenMetric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_MAX_CANDIDATE_CAP = 256


@dataclass(frozen=True, slots=True)
class PhysicalMetricRun:
    """One fixed-sequence measurement from a physical mixed cache."""

    mean_metric: float
    token_count: int
    resident_bytes: int
    high_precision_groups: int
    cache_update_count: int


@dataclass(frozen=True, slots=True)
class RowPromotionMeasurement:
    """The measured marginal effect of promoting one candidate row."""

    location: RowLocation
    run: PhysicalMetricRun
    metric_delta: float
    improvement: float


@dataclass(frozen=True, slots=True)
class PhysicalRowPromotionOracleResult:
    """Results from a bounded, post-hoc physical intervention audit."""

    metric_name: str
    lower_is_better: bool
    baseline: PhysicalMetricRun
    measurements: tuple[RowPromotionMeasurement, ...]
    background_rows: tuple[RowLocation, ...]
    baseline_plan_bytes: int
    intervention_plan_bytes: int
    promotion_increment_bytes: int

    def ranked_measurements(self) -> tuple[RowPromotionMeasurement, ...]:
        """Return candidates from largest measured improvement to smallest."""

        return tuple(
            sorted(
                self.measurements,
                key=lambda measurement: (-measurement.improvement, measurement.location),
            )
        )


def target_nll_values(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return one teacher-forced negative log-likelihood value per target token."""

    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    return -log_probs.gather(-1, target_ids.to(torch.int64).unsqueeze(-1)).squeeze(-1)


def _validate_token_inputs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> None:
    for name, value in (("prompt_ids", prompt_ids), ("continuation_ids", continuation_ids)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"{name} must have non-empty shape [batch, tokens]")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use torch.int32 or torch.int64")
    if prompt_ids.shape[0] != continuation_ids.shape[0]:
        raise ValueError("prompt_ids and continuation_ids must have the same batch size")
    if prompt_ids.device != continuation_ids.device:
        raise ValueError("prompt_ids and continuation_ids must be on the same device")

    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.device != prompt_ids.device:
        raise ValueError(
            "token tensors must be on the model device: "
            f"tokens={prompt_ids.device}, model={parameter.device}"
        )

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("model.config.vocab_size must be a positive integer")
    for name, value in (("prompt_ids", prompt_ids), ("continuation_ids", continuation_ids)):
        if value.min().item() < 0 or value.max().item() >= vocab_size:
            raise ValueError(f"{name} contains a token outside [0, {vocab_size})")


def _qwen_row_geometry(model: torch.nn.Module) -> tuple[tuple[int, int, int], ...]:
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    head_count = getattr(config, "linear_num_value_heads", None)
    row_count = getattr(config, "linear_key_head_dim", None)
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError("model.config.layer_types must be a list or tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (head_count, row_count)
    ):
        raise ValueError("model config has invalid recurrent-state row geometry")
    geometry = tuple(
        (layer_index, head_count, row_count)
        for layer_index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    )
    if not geometry:
        raise ValueError("model config has no linear_attention layers")
    return geometry


def _validated_locations(
    locations: Sequence[RowLocation],
    *,
    name: str,
    shapes: tuple[tuple[int, int, int], ...],
) -> tuple[RowLocation, ...]:
    rendered = tuple(locations)
    if any(not isinstance(location, RowLocation) for location in rendered):
        raise TypeError(f"{name} must contain only RowLocation values")
    if len(set(rendered)) != len(rendered):
        raise ValueError(f"{name} must not contain duplicates")

    geometry = {layer_index: (heads, rows) for layer_index, heads, rows in shapes}
    for location in rendered:
        values = (location.layer_index, location.head_index, location.row_index)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{name} indices must be integers")
        layer_shape = geometry.get(location.layer_index)
        if layer_shape is None:
            raise ValueError(f"{name} references non-recurrent layer {location.layer_index}")
        heads, rows = layer_shape
        if not (0 <= location.head_index < heads and 0 <= location.row_index < rows):
            raise ValueError(f"{name} contains an out-of-range location: {location}")
    return tuple(sorted(rendered))


def _resident_bytes(
    *,
    total_groups: int,
    promoted_groups: int,
    group_size: int,
    scale_bits: int,
) -> tuple[int, int]:
    low_payload_bytes = math.ceil(4 * group_size / 8)
    high_payload_bytes = math.ceil(8 * group_size / 8)
    increment = high_payload_bytes - low_payload_bytes
    minimum = total_groups * (low_payload_bytes + scale_bits // 8) + math.ceil(total_groups / 8)
    return minimum + promoted_groups * increment, increment


def _plan_for_rows(
    *,
    shapes: tuple[tuple[int, int, int], ...],
    promoted_rows: tuple[RowLocation, ...],
    target_resident_bytes: int,
    group_size: int,
    scale_bits: int,
) -> ExactBudgetRowPlan:
    scores = {
        layer_index: torch.zeros((heads, rows), dtype=torch.float64)
        for layer_index, heads, rows in shapes
    }
    for location in promoted_rows:
        scores[location.layer_index][location.head_index, location.row_index] = 1.0
    plan = select_rows_exact_budget(
        scores,
        target_resident_bytes=target_resident_bytes,
        group_size=group_size,
        scale_bits=scale_bits,
    )
    if plan.high_precision_rows != tuple(sorted(promoted_rows)):
        raise RuntimeError("exact-byte selector did not realize the requested intervention rows")
    return plan


def _metric_values(
    metric: TokenMetric,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    values = metric(logits, target_ids)
    if not isinstance(values, torch.Tensor):
        raise TypeError("metric must return a torch.Tensor")
    if values.shape != target_ids.shape:
        raise ValueError(
            "metric must return one value per target with shape "
            f"{tuple(target_ids.shape)}, got {tuple(values.shape)}"
        )
    if not values.is_floating_point():
        raise TypeError("metric values must use a floating-point dtype")
    if not torch.isfinite(values).all().item():
        raise ValueError("metric returned a non-finite value")
    return values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)


def _run_plan(
    model: torch.nn.Module,
    *,
    plan: ExactBudgetRowPlan,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    metric: TokenMetric,
    seed: int,
) -> PhysicalMetricRun:
    cache = create_qwen35_exact_budget_cache(
        model,
        plan=plan,
        rounding="nearest",
        seed=seed,
        record_evidence=True,
    )
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        output = model(
            prompt_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        values.append(_metric_values(metric, output.logits, continuation_ids[:, :1]))

        for token_index in range(continuation_ids.shape[1] - 1):
            output = model(
                continuation_ids[:, token_index : token_index + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            target = continuation_ids[:, token_index + 1 : token_index + 2]
            values.append(_metric_values(metric, output.logits, target))

    summary = cache.storage_summary()
    batch_size = prompt_ids.shape[0]
    expected_resident = batch_size * plan.resident_bytes
    expected_high_precision = batch_size * plan.promoted_group_count
    resident = int(summary["resident_bytes"])
    high_precision = int(summary["high_precision_groups"])
    if resident != expected_resident or high_precision != expected_high_precision:
        raise RuntimeError(
            "physical cache did not realize the intervention plan: "
            f"resident={resident}/{expected_resident}, "
            f"high_precision_groups={high_precision}/{expected_high_precision}"
        )

    recurrent_layer_indices = tuple(
        layer_index for layer_index, _, _ in plan.score_shapes
    )
    forward_count = 1 + (continuation_ids.shape[1] - 1)
    expected_update_count = len(recurrent_layer_indices) * forward_count
    actual_update_count = len(cache.update_evidence)
    update_counts = {
        (layer_index, state_index): sum(
            evidence.layer_index == layer_index and evidence.state_index == state_index
            for evidence in cache.update_evidence
        )
        for layer_index in recurrent_layer_indices
        for state_index in (0,)
    }
    expected_update_counts = {
        (layer_index, 0): forward_count for layer_index in recurrent_layer_indices
    }
    if (
        actual_update_count != expected_update_count
        or update_counts != expected_update_counts
    ):
        raise RuntimeError(
            "physical cache update trace did not cover one recurrent state per layer "
            "for every model forward: "
            f"updates={actual_update_count}/{expected_update_count}, "
            f"per_layer_state={update_counts}/{expected_update_counts}"
        )

    combined = torch.cat(values)
    return PhysicalMetricRun(
        mean_metric=float(combined.mean().item()),
        token_count=combined.numel(),
        resident_bytes=resident,
        high_precision_groups=high_precision,
        cache_update_count=actual_update_count,
    )


def evaluate_physical_row_promotions(
    model: torch.nn.Module,
    *,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    candidate_rows: Sequence[RowLocation],
    background_rows: Sequence[RowLocation] = (),
    intervention_resident_bytes: int | None = None,
    scale_bits: int = 16,
    seed: int = 2339,
    max_candidates: int = 32,
    metric: TokenMetric = target_nll_values,
    metric_name: str = "target_nll",
    lower_is_better: bool = True,
) -> PhysicalRowPromotionOracleResult:
    """Measure marginal row promotions through the physical recurrent cache.

    A fresh cache first runs ``background_rows`` as the control.  Each candidate
    then runs in a separate cache with exactly ``background_rows + candidate``.
    Thus all candidate interventions have the same exact resident-byte budget,
    while the control is smaller by one INT4-to-INT8 payload increment.  The
    sequence, targets, nearest-rounding rule, and quantizer seed are held fixed.

    The work scales linearly with ``len(candidate_rows)`` and is intentionally
    capped.  Because the targets directly define the default NLL metric, use
    held-out data for claims and never describe this post-hoc audit as a
    deployable oracle.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    if not 1 <= max_candidates <= _MAX_CANDIDATE_CAP:
        raise ValueError(f"max_candidates must be between 1 and {_MAX_CANDIDATE_CAP}")
    if isinstance(scale_bits, bool) or scale_bits not in (16, 32):
        raise ValueError("scale_bits must be 16 or 32")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not callable(metric):
        raise TypeError("metric must be callable")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a non-empty string")
    if not isinstance(lower_is_better, bool):
        raise TypeError("lower_is_better must be a bool")

    _validate_token_inputs(model, prompt_ids, continuation_ids)
    shapes = _qwen_row_geometry(model)
    candidates = _validated_locations(candidate_rows, name="candidate_rows", shapes=shapes)
    background = _validated_locations(background_rows, name="background_rows", shapes=shapes)
    if not candidates:
        raise ValueError("candidate_rows must not be empty")
    if len(candidates) > max_candidates:
        raise ValueError(
            "candidate_rows has "
            f"{len(candidates)} entries, exceeding max_candidates={max_candidates}"
        )
    overlap = sorted(set(candidates) & set(background))
    if overlap:
        raise ValueError(f"candidate_rows must not overlap background_rows: {overlap}")

    group_size = getattr(model.config, "linear_value_head_dim", None)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("model config has invalid recurrent-state group size")
    total_groups = sum(heads * rows for _, heads, rows in shapes)
    if len(background) + 1 > total_groups:
        raise ValueError("background_rows leaves no row available for an intervention")

    baseline_bytes, promotion_increment = _resident_bytes(
        total_groups=total_groups,
        promoted_groups=len(background),
        group_size=group_size,
        scale_bits=scale_bits,
    )
    expected_intervention_bytes, _ = _resident_bytes(
        total_groups=total_groups,
        promoted_groups=len(background) + 1,
        group_size=group_size,
        scale_bits=scale_bits,
    )
    if intervention_resident_bytes is None:
        intervention_bytes = expected_intervention_bytes
    else:
        if isinstance(intervention_resident_bytes, bool) or not isinstance(
            intervention_resident_bytes, int
        ):
            raise TypeError("intervention_resident_bytes must be an integer")
        if intervention_resident_bytes != expected_intervention_bytes:
            raise ValueError(
                "intervention_resident_bytes must exactly encode background_rows plus one "
                f"promotion: expected {expected_intervention_bytes}, got "
                f"{intervention_resident_bytes}"
            )
        intervention_bytes = intervention_resident_bytes

    baseline_plan = _plan_for_rows(
        shapes=shapes,
        promoted_rows=background,
        target_resident_bytes=baseline_bytes,
        group_size=group_size,
        scale_bits=scale_bits,
    )
    baseline = _run_plan(
        model,
        plan=baseline_plan,
        prompt_ids=prompt_ids,
        continuation_ids=continuation_ids,
        metric=metric,
        seed=seed,
    )

    measurements: list[RowPromotionMeasurement] = []
    for candidate in candidates:
        promoted = tuple(sorted((*background, candidate)))
        plan = _plan_for_rows(
            shapes=shapes,
            promoted_rows=promoted,
            target_resident_bytes=intervention_bytes,
            group_size=group_size,
            scale_bits=scale_bits,
        )
        run = _run_plan(
            model,
            plan=plan,
            prompt_ids=prompt_ids,
            continuation_ids=continuation_ids,
            metric=metric,
            seed=seed,
        )
        metric_delta = run.mean_metric - baseline.mean_metric
        improvement = -metric_delta if lower_is_better else metric_delta
        measurements.append(
            RowPromotionMeasurement(
                location=candidate,
                run=run,
                metric_delta=metric_delta,
                improvement=improvement,
            )
        )

    return PhysicalRowPromotionOracleResult(
        metric_name=metric_name.strip(),
        lower_is_better=lower_is_better,
        baseline=baseline,
        measurements=tuple(measurements),
        background_rows=background,
        baseline_plan_bytes=baseline_bytes,
        intervention_plan_bytes=intervention_bytes,
        promotion_increment_bytes=promotion_increment,
    )
