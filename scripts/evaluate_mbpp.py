#!/usr/bin/env python3
"""Run the frozen RecurQuant v0.2 teacher-forced MBPP evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM

from recurquant.cache import iter_recurrent_states
from recurquant.evaluation import (
    TokenFidelity,
    fidelity_summary,
    paired_bootstrap_mean_improvement,
)
from recurquant.packed_cache import PackedRecurrentStateCache
from recurquant.public_data import (
    MBPP_CALIBRATION_SIZE,
    MBPP_CONFIRMATION_LOCK,
    MBPP_REVISION,
    MBPPPhase,
    format_mbpp_example,
    load_mbpp_rows,
    mbpp_manifest,
    mbpp_manifest_sha256,
)
from recurquant.quantization import QuantizationSpec
from recurquant.transformers_cache import RecurrentStateQDQCache

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 2339
GDN_LAYER_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
EXPECTED_BYTES = {
    "fp32_state": 18_874_368,
    "uniform_int4": 2_433_024,
    "mixed_int4_int8": 2_564_096,
    "uniform_int8": 4_792_320,
}


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    name: str
    default_bits: int
    upgrade_layer: int | None
    rounding: str = "nearest"
    seed: int = SEED

    @property
    def layout(self) -> str:
        if self.default_bits == 8:
            return "uniform_int8"
        if self.upgrade_layer is None:
            return "uniform_int4"
        return "mixed_int4_int8"


@dataclass(frozen=True, slots=True)
class EncodedTask:
    task_id: int
    prompt_ids: tuple[int, ...]
    code_ids: tuple[int, ...]
    prompt_sha256: str
    code_sha256: str

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.code_ids)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt_tokens": len(self.prompt_ids),
            "code_tokens": len(self.code_ids),
            "total_tokens": self.total_tokens,
            "prompt_token_ids_sha256": self.prompt_sha256,
            "code_token_ids_sha256": self.code_sha256,
        }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return sha256_bytes(
        json.dumps(token_ids, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate packed recurrent-state fidelity on pinned MBPP data."
    )
    parser.add_argument(
        "--phase",
        choices=tuple(phase.value for phase in MBPPPhase),
        default=MBPPPhase.CALIBRATION.value,
    )
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mbpp-v02-evaluation.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic-only prefix; forbidden for confirmation by the data loader.",
    )
    parser.add_argument("--confirmation-lock")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--skip-qdq-preflight", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-calibration",
        action="store_true",
        help="Allow a non-protocol calibration artifact for calibration-split smoke runs only.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_calibration_artifact(
    path: Path,
    *,
    phase: str,
    allow_diagnostic: bool,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_kind") != "recurquant_mbpp_layer_calibration":
        raise ValueError("--calibration-artifact has the wrong artifact_kind")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("calibration artifact does not contain evidence")
    expected_hash = sha256_bytes(canonical_json_bytes(evidence))
    if artifact.get("canonical_evidence_sha256") != expected_hash:
        raise ValueError("calibration artifact evidence hash does not match its contents")

    source = evidence.get("source", {})
    if source.get("model_id") != model_id or source.get("model_revision") != revision:
        raise ValueError("calibration artifact model pin does not match the evaluator")
    dataset_manifest = source.get("dataset_manifest", {})
    if dataset_manifest.get("revision") != MBPP_REVISION:
        raise ValueError("calibration artifact MBPP revision does not match the evaluator")

    protocol_eligible = evidence.get("claim_scope", {}).get("protocol_eligible") is True
    diagnostic_allowed = (
        allow_diagnostic and phase == MBPPPhase.CALIBRATION.value
    )
    if not protocol_eligible and not diagnostic_allowed:
        raise ValueError(
            "calibration artifact is diagnostic-only; run the full 128-task calibration "
            "before development"
        )
    return artifact


def candidate_definitions(calibration_artifact: dict[str, Any]) -> list[CandidateDefinition]:
    layers = calibration_artifact["evidence"]["calibration"]["candidate_layers"]
    required = {
        "read_risk_l0_nearest": 0,
        "random_l18_nearest": 18,
        "random_l4_nearest": 4,
        "random_l13_nearest": 13,
    }
    for name, expected_layer in required.items():
        if layers.get(name) != expected_layer:
            raise ValueError(f"calibration artifact changed frozen candidate {name}")
    mse_layer = layers.get("mse_selected_nearest")
    if isinstance(mse_layer, bool) or not isinstance(mse_layer, int):
        raise TypeError("mse_selected_nearest must be an integer layer index")
    if mse_layer not in GDN_LAYER_INDICES:
        raise ValueError("mse_selected_nearest is not a Gated DeltaNet layer")

    candidates = [
        CandidateDefinition("uniform_int4_nearest", 4, None),
        CandidateDefinition("uniform_int8_nearest", 8, None),
        CandidateDefinition("read_risk_l0_nearest", 4, 0),
        CandidateDefinition("mse_selected_nearest", 4, mse_layer),
        CandidateDefinition("random_l18_nearest", 4, 18),
        CandidateDefinition("random_l4_nearest", 4, 4),
        CandidateDefinition("random_l13_nearest", 4, 13),
    ]
    candidates.extend(
        CandidateDefinition(
            f"read_risk_l0_stochastic_seed_{seed}",
            4,
            0,
            rounding="stochastic",
            seed=seed,
        )
        for seed in (2339, 2340, 2341)
    )
    return candidates


def candidate_specs(
    candidate: CandidateDefinition,
    *,
    group_size: int,
) -> tuple[QuantizationSpec, dict[int, QuantizationSpec]]:
    default_spec = QuantizationSpec(
        bits=candidate.default_bits,
        group_size=group_size,
        scale_bits=16,
        rounding=candidate.rounding,
        seed=candidate.seed,
    )
    layer_specs: dict[int, QuantizationSpec] = {}
    if candidate.upgrade_layer is not None:
        layer_specs[candidate.upgrade_layer] = QuantizationSpec(
            bits=8,
            group_size=group_size,
            scale_bits=16,
            rounding=candidate.rounding,
            seed=candidate.seed,
        )
    return default_spec, layer_specs


def make_packed_cache(
    config: object,
    candidate: CandidateDefinition,
    *,
    group_size: int,
) -> PackedRecurrentStateCache:
    spec, layer_specs = candidate_specs(candidate, group_size=group_size)
    return PackedRecurrentStateCache(
        config,
        spec=spec,
        layer_specs=layer_specs,
        record_evidence=False,
    )


def make_qdq_cache(
    config: object,
    candidate: CandidateDefinition,
    *,
    group_size: int,
) -> RecurrentStateQDQCache:
    spec, layer_specs = candidate_specs(candidate, group_size=group_size)
    return RecurrentStateQDQCache(
        config,
        spec=spec,
        layer_specs=layer_specs,
        record_evidence=False,
    )


def encode_tasks(
    tokenizer: Any,
    rows: tuple[dict[str, Any], ...],
    *,
    max_position_embeddings: int,
) -> list[EncodedTask]:
    encoded: list[EncodedTask] = []
    for row in rows:
        formatted = format_mbpp_example(row)
        prompt_ids = tuple(
            tokenizer(formatted.prompt, add_special_tokens=True)["input_ids"]
        )
        code_ids = tuple(tokenizer(formatted.code, add_special_tokens=False)["input_ids"])
        if not prompt_ids:
            raise RuntimeError(f"MBPP task {row['task_id']} produced an empty prompt")
        if not code_ids:
            raise RuntimeError(f"MBPP task {row['task_id']} produced an empty code target")
        if len(prompt_ids) + len(code_ids) > max_position_embeddings:
            raise RuntimeError(
                f"MBPP task {row['task_id']} exceeds the pinned model context"
            )
        encoded.append(
            EncodedTask(
                task_id=row["task_id"],
                prompt_ids=prompt_ids,
                code_ids=code_ids,
                prompt_sha256=token_ids_sha256(prompt_ids),
                code_sha256=token_ids_sha256(code_ids),
            )
        )
    return encoded


def _tensor_ids(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long, device=device)


def _assert_logits_close(
    packed_logits: torch.Tensor,
    qdq_logits: torch.Tensor,
    *,
    candidate: str,
    step: int,
    tolerance: float,
) -> float:
    difference = (packed_logits.to(torch.float32) - qdq_logits.to(torch.float32)).abs().max()
    maximum = float(difference.item())
    if maximum > tolerance:
        raise RuntimeError(
            f"Packed/QDQ parity failed for {candidate} at step {step}: {maximum} > {tolerance}"
        )
    return maximum


def verify_qdq_parity(
    model: Qwen3_5ForCausalLM,
    task: EncodedTask,
    candidates: list[CandidateDefinition],
    *,
    group_size: int,
    device: torch.device,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    prompt = _tensor_ids(task.prompt_ids, device)
    code = _tensor_ids(task.code_ids, device)
    maximum_by_candidate: dict[str, float] = {}

    with torch.inference_mode():
        for candidate in candidates:
            packed_cache = make_packed_cache(
                model.config, candidate, group_size=group_size
            )
            qdq_cache = make_qdq_cache(model.config, candidate, group_size=group_size)
            packed_output = model(
                prompt,
                past_key_values=packed_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            qdq_output = model(
                prompt,
                past_key_values=qdq_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            maximum = _assert_logits_close(
                packed_output.logits,
                qdq_output.logits,
                candidate=candidate.name,
                step=0,
                tolerance=tolerance,
            )
            for token_index in range(code.shape[1] - 1):
                token = code[:, token_index : token_index + 1]
                packed_output = model(
                    token,
                    past_key_values=packed_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                qdq_output = model(
                    token,
                    past_key_values=qdq_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                maximum = max(
                    maximum,
                    _assert_logits_close(
                        packed_output.logits,
                        qdq_output.logits,
                        candidate=candidate.name,
                        step=token_index + 1,
                        tolerance=tolerance,
                    ),
                )

            packed_states = {
                (state.layer_index, state.state_index): state.tensor
                for state in iter_recurrent_states(packed_cache)
            }
            qdq_states = {
                (state.layer_index, state.state_index): state.tensor
                for state in iter_recurrent_states(qdq_cache)
            }
            if packed_states.keys() != qdq_states.keys():
                raise RuntimeError(f"Packed/QDQ state layout differs for {candidate.name}")
            for state_key in packed_states:
                state_difference = (
                    packed_states[state_key].to(torch.float32)
                    - qdq_states[state_key].to(torch.float32)
                ).abs().max()
                maximum = max(maximum, float(state_difference.item()))
            if maximum > tolerance:
                raise RuntimeError(
                    f"Packed/QDQ state parity failed for {candidate.name}: {maximum}"
                )
            maximum_by_candidate[candidate.name] = maximum

    return {
        "task_id": task.task_id,
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference_by_candidate": maximum_by_candidate,
        "passed": True,
    }


def _append_step_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: dict[str, torch.Tensor],
    target_id: torch.Tensor,
    accumulators: dict[str, dict[str, list[torch.Tensor]]],
) -> None:
    reference = reference_logits.to(torch.float32)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    reference_probs = reference_log_probs.exp()
    target = target_id.to(torch.int64).reshape(1, 1, 1)
    reference_nll = -reference_log_probs.gather(-1, target).squeeze(-1)
    reference_top1 = reference.argmax(dim=-1)

    for name, logits in candidate_logits.items():
        candidate = logits.to(torch.float32)
        candidate_log_probs = torch.log_softmax(candidate, dim=-1)
        kl = (
            reference_probs * (reference_log_probs - candidate_log_probs)
        ).sum(dim=-1)
        candidate_nll = -candidate_log_probs.gather(-1, target).squeeze(-1)
        accumulators[name]["kl"].append(kl.detach())
        accumulators[name]["reference_nll"].append(reference_nll.detach())
        accumulators[name]["candidate_nll"].append(candidate_nll.detach())
        accumulators[name]["top1"].append(
            (reference_top1 == candidate.argmax(dim=-1)).detach()
        )


def _task_fidelity(
    values: dict[str, list[torch.Tensor]],
) -> TokenFidelity:
    return TokenFidelity(
        kl=torch.cat(values["kl"], dim=1).cpu(),
        reference_nll=torch.cat(values["reference_nll"], dim=1).cpu(),
        candidate_nll=torch.cat(values["candidate_nll"], dim=1).cpu(),
        top1_agreement=torch.cat(values["top1"], dim=1).cpu(),
    )


def evaluate_task(
    model: Qwen3_5ForCausalLM,
    task: EncodedTask,
    candidates: list[CandidateDefinition],
    *,
    group_size: int,
    device: torch.device,
) -> tuple[dict[str, TokenFidelity], dict[str, dict[str, Any]], int]:
    prompt = _tensor_ids(task.prompt_ids, device)
    code = _tensor_ids(task.code_ids, device)
    reference_cache = DynamicCache(config=model.config)
    caches = {
        candidate.name: make_packed_cache(
            model.config, candidate, group_size=group_size
        )
        for candidate in candidates
    }
    accumulators: dict[str, dict[str, list[torch.Tensor]]] = {
        candidate.name: {
            "kl": [],
            "reference_nll": [],
            "candidate_nll": [],
            "top1": [],
        }
        for candidate in candidates
    }

    with torch.inference_mode():
        reference_output = model(
            prompt,
            past_key_values=reference_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        candidate_outputs = {
            name: model(
                prompt,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits
            for name, cache in caches.items()
        }
        _append_step_metrics(
            reference_output.logits,
            candidate_outputs,
            code[:, 0],
            accumulators,
        )

        for token_index in range(code.shape[1] - 1):
            token = code[:, token_index : token_index + 1]
            reference_output = model(
                token,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            candidate_outputs = {
                name: model(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits
                for name, cache in caches.items()
            }
            _append_step_metrics(
                reference_output.logits,
                candidate_outputs,
                code[:, token_index + 1],
                accumulators,
            )

    reference_bytes = sum(
        state.tensor.numel() * state.tensor.element_size()
        for state in iter_recurrent_states(reference_cache)
    )
    storage = {name: cache.storage_summary() for name, cache in caches.items()}
    return (
        {name: _task_fidelity(values) for name, values in accumulators.items()},
        storage,
        reference_bytes,
    )


def _length_quartile_membership(tasks: list[EncodedTask]) -> dict[int, str]:
    ordered = sorted(tasks, key=lambda task: (len(task.code_ids), task.task_id))
    chunks = np.array_split(np.asarray([task.task_id for task in ordered]), 4)
    membership: dict[int, str] = {}
    for index, chunk in enumerate(chunks, start=1):
        for task_id in chunk.tolist():
            membership[int(task_id)] = f"Q{index}"
    return membership


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def _expected_resident_bytes(candidate: CandidateDefinition) -> int:
    return EXPECTED_BYTES[candidate.layout]


def main() -> int:
    args = parse_args()
    phase = MBPPPhase(args.phase)
    if args.group_size != 128:
        raise ValueError("the frozen v0.2 protocol requires --group-size 128")
    if args.bootstrap_samples != 10_000:
        raise ValueError("the frozen v0.2 protocol requires 10,000 bootstrap samples")
    if phase is MBPPPhase.CONFIRMATION and args.confirmation_lock != MBPP_CONFIRMATION_LOCK:
        raise ValueError("confirmation requires the explicit frozen confirmation lock")

    calibration_artifact = load_calibration_artifact(
        args.calibration_artifact,
        phase=phase.value,
        allow_diagnostic=args.allow_diagnostic_calibration,
        model_id=args.model_id,
        revision=args.revision,
    )
    candidates = candidate_definitions(calibration_artifact)
    rows = load_mbpp_rows(
        phase,
        limit=args.limit,
        confirmation_lock=args.confirmation_lock,
    )
    dataset_manifest = mbpp_manifest(rows, phase=phase)

    torch.manual_seed(SEED)
    device = select_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    configured_gdn_layers = tuple(
        index
        for index, layer_type in enumerate(model.config.layer_types)
        if layer_type == "linear_attention"
    )
    if configured_gdn_layers != GDN_LAYER_INDICES:
        raise RuntimeError(
            f"Pinned GDN layout mismatch: {configured_gdn_layers} != {GDN_LAYER_INDICES}"
        )

    tasks = encode_tasks(
        tokenizer,
        rows,
        max_position_embeddings=model.config.max_position_embeddings,
    )
    token_manifest = [task.manifest_dict() for task in tasks]
    token_manifest_sha256 = sha256_bytes(canonical_json_bytes(token_manifest))

    parity: dict[str, Any] | None = None
    if not args.skip_qdq_preflight:
        calibration_row = load_mbpp_rows("calibration", limit=1)
        parity_task = encode_tasks(
            tokenizer,
            calibration_row,
            max_position_embeddings=model.config.max_position_embeddings,
        )[0]
        parity = verify_qdq_parity(
            model,
            parity_task,
            candidates,
            group_size=args.group_size,
            device=device,
        )
    elif phase is not MBPPPhase.CALIBRATION or args.limit is None:
        raise ValueError("protocol-eligible runs may not skip the packed/QDQ preflight")

    global_values: dict[str, dict[str, list[float | bool]]] = {
        candidate.name: {
            "kl": [],
            "reference_nll": [],
            "candidate_nll": [],
            "top1": [],
        }
        for candidate in candidates
    }
    per_task: dict[str, list[dict[str, Any]]] = {
        candidate.name: [] for candidate in candidates
    }
    storage_by_candidate: dict[str, dict[str, Any]] | None = None
    reference_state_bytes: int | None = None
    started = time.perf_counter()

    for task_index, task in enumerate(tasks, start=1):
        task_fidelity, storage, task_reference_bytes = evaluate_task(
            model,
            task,
            candidates,
            group_size=args.group_size,
            device=device,
        )
        if reference_state_bytes is None:
            reference_state_bytes = task_reference_bytes
        elif reference_state_bytes != task_reference_bytes:
            raise RuntimeError("reference recurrent-state bytes changed between tasks")
        if storage_by_candidate is None:
            storage_by_candidate = storage
        elif storage_by_candidate != storage:
            raise RuntimeError("packed recurrent-state storage changed between tasks")

        for candidate in candidates:
            fidelity = task_fidelity[candidate.name]
            summary = fidelity_summary(fidelity)
            per_task[candidate.name].append(
                {
                    "task_id": task.task_id,
                    "code_tokens": len(task.code_ids),
                    **summary,
                }
            )
            global_values[candidate.name]["kl"].extend(
                float(value) for value in fidelity.kl.flatten().tolist()
            )
            global_values[candidate.name]["reference_nll"].extend(
                float(value) for value in fidelity.reference_nll.flatten().tolist()
            )
            global_values[candidate.name]["candidate_nll"].extend(
                float(value) for value in fidelity.candidate_nll.flatten().tolist()
            )
            global_values[candidate.name]["top1"].extend(
                bool(value) for value in fidelity.top1_agreement.flatten().tolist()
            )
        print(
            f"[{task_index}/{len(tasks)}] task={task.task_id} "
            f"prompt={len(task.prompt_ids)} code={len(task.code_ids)}"
        )

    elapsed_seconds = time.perf_counter() - started
    assert storage_by_candidate is not None
    assert reference_state_bytes is not None
    if reference_state_bytes != EXPECTED_BYTES["fp32_state"]:
        raise RuntimeError(
            f"FP32-state bytes {reference_state_bytes} != {EXPECTED_BYTES['fp32_state']}"
        )

    length_membership = _length_quartile_membership(tasks)
    candidate_results: dict[str, Any] = {}
    for candidate in candidates:
        name = candidate.name
        values = global_values[name]
        fidelity = TokenFidelity(
            kl=torch.tensor(values["kl"], dtype=torch.float32),
            reference_nll=torch.tensor(values["reference_nll"], dtype=torch.float32),
            candidate_nll=torch.tensor(values["candidate_nll"], dtype=torch.float32),
            top1_agreement=torch.tensor(values["top1"], dtype=torch.bool),
        )
        token_weighted = fidelity_summary(fidelity)
        task_rows = per_task[name]
        task_delta = [float(row["delta_nll"]) for row in task_rows]
        task_reference_nll = [float(row["reference_nll"]) for row in task_rows]
        task_candidate_nll = [float(row["candidate_nll"]) for row in task_rows]
        by_quartile: dict[str, Any] = {}
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            quartile_rows = [
                row
                for row in task_rows
                if length_membership[int(row["task_id"])] == quartile
            ]
            if not quartile_rows:
                by_quartile[quartile] = {
                    "task_count": 0,
                    "minimum_code_tokens": None,
                    "maximum_code_tokens": None,
                    "macro_delta_nll": None,
                    "macro_mean_kl": None,
                    "macro_top1_agreement": None,
                }
                continue
            by_quartile[quartile] = {
                "task_count": len(quartile_rows),
                "minimum_code_tokens": min(int(row["code_tokens"]) for row in quartile_rows),
                "maximum_code_tokens": max(int(row["code_tokens"]) for row in quartile_rows),
                "macro_delta_nll": fmean(float(row["delta_nll"]) for row in quartile_rows),
                "macro_mean_kl": fmean(float(row["mean_kl"]) for row in quartile_rows),
                "macro_top1_agreement": fmean(
                    float(row["top1_agreement"]) for row in quartile_rows
                ),
            }

        expected_bytes = _expected_resident_bytes(candidate)
        actual_storage = storage_by_candidate[name]
        if actual_storage["resident_bytes"] != expected_bytes:
            raise RuntimeError(
                f"{name} resident bytes {actual_storage['resident_bytes']} != {expected_bytes}"
            )
        candidate_results[name] = {
            "policy": asdict(candidate),
            "storage": {
                **actual_storage,
                "expected_resident_bytes": expected_bytes,
                "exact_byte_gate": True,
            },
            "token_weighted": token_weighted,
            "task_macro": {
                "task_count": len(task_rows),
                "reference_nll": fmean(task_reference_nll),
                "candidate_nll": fmean(task_candidate_nll),
                "delta_nll": fmean(task_delta),
                "mean_kl": fmean(float(row["mean_kl"]) for row in task_rows),
                "top1_agreement": fmean(
                    float(row["top1_agreement"]) for row in task_rows
                ),
            },
            "task_delta_nll_distribution": _quantiles(task_delta),
            "by_code_length_quartile": by_quartile,
            "per_task": task_rows,
        }

    uniform_rows = per_task["uniform_int4_nearest"]
    primary_rows = per_task["read_risk_l0_nearest"]
    uniform_delta = [float(row["delta_nll"]) for row in uniform_rows]
    primary_delta = [float(row["delta_nll"]) for row in primary_rows]
    random_names = ("random_l18_nearest", "random_l4_nearest", "random_l13_nearest")
    random_mean_delta = [
        fmean(float(per_task[name][task_index]["delta_nll"]) for name in random_names)
        for task_index in range(len(tasks))
    ]
    macro_uniform = fmean(uniform_delta)
    macro_primary = fmean(primary_delta)
    relative_reduction = (
        (macro_uniform - macro_primary) / macro_uniform if macro_uniform > 0 else None
    )
    uniform_bootstrap = paired_bootstrap_mean_improvement(
        uniform_delta,
        primary_delta,
        samples=args.bootstrap_samples,
        seed=SEED,
    )
    equal_byte_bootstrap = paired_bootstrap_mean_improvement(
        random_mean_delta,
        primary_delta,
        samples=args.bootstrap_samples,
        seed=SEED,
    )
    primary_token = candidate_results["read_risk_l0_nearest"]["token_weighted"]
    uniform_token = candidate_results["uniform_int4_nearest"]["token_weighted"]
    finite_values = all(
        math.isfinite(float(value))
        for candidate_values in global_values.values()
        for metric_values in candidate_values.values()
        if metric_values and not isinstance(metric_values[0], bool)
        for value in metric_values
    )
    continuation_gates = {
        "all_values_finite": finite_values,
        "exact_resident_bytes": all(
            result["storage"]["exact_byte_gate"] for result in candidate_results.values()
        ),
        "primary_macro_delta_nll_reduction_at_least_15_percent": (
            relative_reduction is not None and relative_reduction >= 0.15
        ),
        "equal_byte_bootstrap_interval_above_zero": (
            equal_byte_bootstrap["confidence_interval"][0] > 0
        ),
        "primary_mean_token_kl_lower_than_uniform_int4": (
            primary_token["mean_kl"] < uniform_token["mean_kl"]
        ),
        "primary_cvar95_token_kl_lower_than_uniform_int4": (
            primary_token["cvar95_kl"] < uniform_token["cvar95_kl"]
        ),
        "primary_top1_not_lower_than_uniform_int4": (
            primary_token["top1_agreement"] >= uniform_token["top1_agreement"]
        ),
    }
    all_continuation_gates_pass = all(continuation_gates.values())
    expected_count = {
        MBPPPhase.CALIBRATION: MBPP_CALIBRATION_SIZE,
        MBPPPhase.DEVELOPMENT: 90,
        MBPPPhase.CONFIRMATION: 500,
    }[phase]
    protocol_eligible = (
        len(rows) == expected_count
        and args.limit is None
        and parity is not None
        and calibration_artifact["evidence"]["claim_scope"]["protocol_eligible"] is True
    )

    evidence: dict[str, Any] = {
        "claim_scope": {
            "phase": phase.value,
            "protocol_eligible": protocol_eligible,
            "teacher_forced_fidelity_only": True,
            "generated_code_executed": False,
            "speed_claim_allowed": False,
            "whole_model_memory_claim_allowed": False,
            "confirmation_touched": phase is MBPPPhase.CONFIRMATION,
        },
        "source": {
            "model_id": args.model_id,
            "model_revision": args.revision,
            "tokenizer_revision": args.revision,
            "dataset_manifest": dataset_manifest,
            "dataset_manifest_sha256": mbpp_manifest_sha256(rows, phase=phase),
            "token_manifest": token_manifest,
            "token_manifest_sha256": token_manifest_sha256,
            "calibration_artifact_path": str(args.calibration_artifact),
            "calibration_evidence_sha256": calibration_artifact[
                "canonical_evidence_sha256"
            ],
            "repository_commit": git_commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "model_dtype": str(next(model.parameters()).dtype),
            "command": [Path(sys.executable).name, *sys.argv],
            "tracked_worktree_clean": not subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "schedule": {
            "seed": SEED,
            "phase": phase.value,
            "row_count": len(rows),
            "group_size": args.group_size,
            "candidate_order": [candidate.name for candidate in candidates],
            "scored_first_code_token_from_prefill": True,
            "candidate_generated_tokens_fed_back": False,
            "elapsed_wall_seconds_not_a_latency_benchmark": elapsed_seconds,
        },
        "validity": {
            "configured_gdn_layer_indices": list(configured_gdn_layers),
            "reference_recurrent_state_bytes": reference_state_bytes,
            "packed_qdq_preflight": parity,
        },
        "candidates": candidate_results,
        "contrasts": {
            "primary_vs_uniform_int4": {
                "uniform_macro_delta_nll": macro_uniform,
                "primary_macro_delta_nll": macro_primary,
                "relative_reduction": relative_reduction,
                "paired_bootstrap": uniform_bootstrap,
            },
            "primary_vs_mean_random_equal_byte": {
                "paired_bootstrap": equal_byte_bootstrap,
            },
        },
        "continuation_decision": {
            "gates": continuation_gates,
            "all_gates_pass": all_continuation_gates_pass,
            "confirmation_permitted": (
                phase is MBPPPhase.DEVELOPMENT
                and protocol_eligible
                and all_continuation_gates_pass
            ),
        },
    }
    evidence_hash = sha256_bytes(canonical_json_bytes(evidence))
    artifact = {
        "schema_version": 1,
        "artifact_kind": "recurquant_mbpp_teacher_forced_evaluation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "canonical_evidence_sha256": evidence_hash,
        "evidence": evidence,
    }
    payload = canonical_json_bytes(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": sha256_bytes(payload),
                "canonical_evidence_sha256": evidence_hash,
                "protocol_eligible": protocol_eligible,
                "phase": phase.value,
                "primary_vs_uniform_int4": evidence["contrasts"][
                    "primary_vs_uniform_int4"
                ],
                "primary_vs_mean_random_equal_byte": evidence["contrasts"][
                    "primary_vs_mean_random_equal_byte"
                ],
                "continuation_decision": evidence["continuation_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
