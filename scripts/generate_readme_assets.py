"""Generate deterministic README charts from authenticated RecurQuant evidence.

The script deliberately parses the source records instead of duplicating their
numbers. Run it without arguments to update the SVG files, or with ``--check``
to fail when committed assets are stale.
"""

# SVG templates remain one element per line so generated diffs stay readable.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "EXPERIMENT_002_SCALE_CORRECTION.md"
DEVELOPMENT_SOURCE = ROOT / "evidence" / "mbpp-v02-development.json"
CONFIRMATION_SOURCE = ROOT / "evidence" / "mbpp-v02-confirmation.json"
STAGE_B_SOURCE = ROOT / "artifacts" / "experiment009-rht-cqer-stage-b-result-cdc603b.json"
STAGE_B_MANIFEST = ROOT / "evidence" / "experiment009-rht-cqer-stage-b-result-manifest.json"
ASSETS = ROOT / "assets"

STAGE_B_ARTIFACT_KIND = "recurquant_rht_cqer32_stage_b_development"
STAGE_B_ARTIFACT_SHA256 = "57b341d37871a52977b1ff89709864f3e6e0927154e5b2b9275b6f374953fe05"
STAGE_B_CANONICAL_SHA256 = "2b15c732e894510f0421a22fcca9435e035dd15c4d3b50e2fcb733c0d1df58a8"
STAGE_B_RESULT_COMMIT = "8168c469b252bc9e707e51feaeccc3f940f190bb"
STAGE_B_VERIFIER_FIX_COMMIT = "2075154e642c39a14432adcc8ec32da679b534d3"
STAGE_B_RELEASE_DOWNLOAD = (
    "https://github.com/Labeeb2339/recurquant/releases/download/"
    "experiment009-stage-b-cdc603b/"
    "experiment009-rht-cqer-stage-b-result-cdc603b.json.zip"
)
STAGE_B_GRAPH_ASSETS = {
    "assets/experiment009-stage-b-overview.svg": ("experiment009-stage-b-overview"),
    "assets/experiment009-stage-b-paired.svg": ("experiment009-stage-b-paired-cqer-minus-rht"),
}
STAGE_B_METHODS = (
    "target_directional_fisher_difference_int4",
    "adaptive_mse_target_directional_fisher_quota",
    "query_ema32_weighted_mse_target_fisher_quota",
    "right_rht_query_ema32_weighted_mse_target_fisher_quota",
)
STAGE_B_CQER_METHOD = "query_ema32_weighted_mse_target_fisher_quota"
STAGE_B_RHT_METHOD = "right_rht_query_ema32_weighted_mse_target_fisher_quota"


def _capture(source: str, pattern: str, label: str) -> tuple[str, ...]:
    match = re.search(pattern, source, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find {label} in {SOURCE.relative_to(ROOT)}")
    return match.groups()


def _table_row(source: str, label: str, columns: int) -> tuple[str, ...]:
    cells = r"\s*\|\s*".join([r"([^|]+?)"] * columns)
    pattern = rf"^\|\s*{re.escape(label)}\s*\|\s*{cells}\s*\|\s*$"
    return tuple(cell.strip() for cell in _capture(source, pattern, label))


def _evidence_hash(source: str, label: str) -> str:
    (digest,) = _capture(
        source,
        rf"^-\s*{re.escape(label)}:\s*`([0-9a-f]{{64}})`\s*$",
        label,
    )
    return digest


def _number(value: str) -> Decimal:
    return Decimal(value.replace(",", "").replace("%", "").split("x", 1)[0].strip())


def _source_data() -> dict[str, object]:
    raw = SOURCE.read_bytes()
    text = raw.decode("utf-8")

    diagnostics = []
    for source_label, display_label in (
        ("Retrieval", "Retrieval"),
        ("Code", "Code"),
        ("Multilingual correction replay", "Multilingual replay"),
    ):
        uniform, mixed, reduction = _table_row(text, source_label, 3)
        diagnostics.append(
            {
                "label": display_label,
                "uniform": uniform,
                "mixed": mixed,
                "reduction": reduction,
            }
        )

    storage = []
    for source_label, display_label, detail in (
        ("FP32 recurrent states", "FP32 states", "reference"),
        ("Uniform INT4 plus FP16 scales", "Uniform INT4", "FP16 scales"),
        (
            "Layer 0 INT8, rest INT4, plus FP16 scales",
            "Mixed: L0 INT8",
            "17 layers INT4 · FP16 scales",
        ),
    ):
        resident_bytes, ratio = _table_row(text, source_label, 2)
        storage.append(
            {
                "label": display_label,
                "detail": detail,
                "bytes": resident_bytes,
                "ratio": ratio,
            }
        )

    evidence_hashes = {
        "retrieval_sweep": _evidence_hash(text, "Corrected retrieval sweep evidence hash"),
        "code_sweep": _evidence_hash(text, "Corrected code sweep evidence hash"),
        "multilingual_qdq": _evidence_hash(text, "Corrected multilingual QDQ replay evidence hash"),
        "multilingual_packed": _evidence_hash(text, "Packed multilingual replay evidence hash"),
    }
    return {
        "document_sha256": hashlib.sha256(raw).hexdigest(),
        "diagnostics": diagnostics,
        "storage": storage,
        "evidence_hashes": evidence_hashes,
    }


def _development_data() -> dict[str, object]:
    raw = DEVELOPMENT_SOURCE.read_bytes()
    document = json.loads(raw)
    evidence = document["evidence"]
    claim_scope = evidence["claim_scope"]
    schedule = evidence["schedule"]
    candidates = evidence["candidates"]
    primary = candidates["read_risk_l0_nearest"]
    mse = candidates["mse_selected_nearest"]
    uniform = candidates["uniform_int4_nearest"]
    primary_contrast = evidence["contrasts"]["primary_vs_uniform_int4"]
    random_contrast = evidence["contrasts"]["primary_vs_mean_random_equal_byte"]

    if claim_scope["phase"] != "development":
        raise ValueError("MBPP README evidence must remain development-only")
    if claim_scope["confirmation_touched"]:
        raise ValueError("Refusing to label evidence as development after confirmation access")
    if not claim_scope["teacher_forced_fidelity_only"]:
        raise ValueError("README chart expects teacher-forced fidelity evidence")
    if claim_scope["generated_code_executed"]:
        raise ValueError("README chart must not imply generated-code execution")
    if claim_scope["speed_claim_allowed"]:
        raise ValueError("README chart must not imply a speed result")
    if claim_scope["whole_model_memory_claim_allowed"]:
        raise ValueError("README chart must not imply whole-model memory reduction")
    if primary["policy"]["upgrade_layer"] != 0:
        raise ValueError("README chart expects the frozen primary policy to upgrade layer 0")
    if mse["policy"]["upgrade_layer"] != primary["policy"]["upgrade_layer"]:
        raise ValueError("README caveat expects the MSE and primary selectors to coincide")
    if primary["task_macro"] != mse["task_macro"]:
        raise ValueError("Coincident layer selectors must have identical task-macro results")
    if not primary["storage"]["exact_byte_gate"]:
        raise ValueError("README storage claim requires the exact-byte gate")
    if primary["task_macro"]["task_count"] != schedule["row_count"]:
        raise ValueError("MBPP task counts disagree")
    if primary["token_weighted"]["token_count"] != uniform["token_weighted"]["token_count"]:
        raise ValueError("MBPP candidate token counts disagree")

    return {
        "document_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_evidence_sha256": document["canonical_evidence_sha256"],
        "phase": claim_scope["phase"],
        "confirmation_touched": claim_scope["confirmation_touched"],
        "teacher_forced_fidelity_only": claim_scope["teacher_forced_fidelity_only"],
        "generated_code_executed": claim_scope["generated_code_executed"],
        "speed_claim_allowed": claim_scope["speed_claim_allowed"],
        "whole_model_memory_claim_allowed": claim_scope["whole_model_memory_claim_allowed"],
        "task_count": schedule["row_count"],
        "token_count": primary["token_weighted"]["token_count"],
        "uniform_macro_delta_nll": uniform["task_macro"]["delta_nll"],
        "primary_macro_delta_nll": primary["task_macro"]["delta_nll"],
        "relative_reduction": primary_contrast["relative_reduction"],
        "primary_vs_uniform": primary_contrast["paired_bootstrap"],
        "primary_vs_mean_random_equal_byte": random_contrast["paired_bootstrap"],
        "primary_resident_bytes": primary["storage"]["resident_bytes"],
        "uniform_resident_bytes": uniform["storage"]["resident_bytes"],
        "full_precision_equivalent_bytes": primary["storage"]["full_precision_equivalent_bytes"],
        "primary_upgrade_layer": primary["policy"]["upgrade_layer"],
        "mse_upgrade_layer": mse["policy"]["upgrade_layer"],
    }


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not permitted: {value}")


def _confirmation_data() -> dict[str, object]:
    raw = CONFIRMATION_SOURCE.read_bytes()
    document = json.loads(raw, parse_constant=_reject_non_finite)
    evidence = document["evidence"]
    claim_scope = evidence["claim_scope"]
    schedule = evidence["schedule"]
    candidates = evidence["candidates"]
    primary = candidates["read_risk_l0_nearest"]
    mse = candidates["mse_selected_nearest"]
    uniform = candidates["uniform_int4_nearest"]
    primary_contrast = evidence["contrasts"]["primary_vs_uniform_int4"]
    random_contrast = evidence["contrasts"]["primary_vs_mean_random_equal_byte"]
    continuation = evidence["continuation_decision"]

    if document["artifact_kind"] != "recurquant_mbpp_teacher_forced_evaluation":
        raise ValueError("Unexpected MBPP confirmation artifact kind")
    if document["schema_version"] != 1:
        raise ValueError("Unexpected MBPP confirmation schema version")
    canonical = hashlib.sha256(
        (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    if document["canonical_evidence_sha256"] != canonical:
        raise ValueError("MBPP confirmation canonical evidence hash does not verify")

    if claim_scope["phase"] != "confirmation":
        raise ValueError("MBPP confirmation chart requires confirmation-phase evidence")
    if not claim_scope["confirmation_touched"]:
        raise ValueError("MBPP confirmation chart requires confirmation_touched=true")
    if not claim_scope["protocol_eligible"]:
        raise ValueError("MBPP confirmation chart requires protocol-eligible evidence")
    if not claim_scope["teacher_forced_fidelity_only"]:
        raise ValueError("MBPP confirmation chart expects teacher-forced fidelity evidence")
    if claim_scope["generated_code_executed"]:
        raise ValueError("MBPP confirmation chart must not imply generated-code execution")
    if claim_scope["speed_claim_allowed"]:
        raise ValueError("MBPP confirmation chart must not imply a speed result")
    if claim_scope["whole_model_memory_claim_allowed"]:
        raise ValueError("MBPP confirmation chart must not imply whole-model memory reduction")
    if not continuation["all_gates_pass"]:
        raise ValueError("MBPP confirmation chart requires all preregistered quality gates")

    expected_task_count = 500
    expected_token_count = 30_244
    if schedule["phase"] != "confirmation":
        raise ValueError("MBPP confirmation schedule phase disagrees with claim scope")
    if schedule["row_count"] != expected_task_count:
        raise ValueError("MBPP confirmation must contain exactly 500 held-out tasks")
    if evidence["source"]["dataset_manifest"]["row_count"] != expected_task_count:
        raise ValueError("MBPP confirmation dataset manifest task count disagrees")
    for label, candidate in (
        ("uniform INT4", uniform),
        ("read-risk mixed", primary),
        ("MSE mixed", mse),
    ):
        if candidate["task_macro"]["task_count"] != expected_task_count:
            raise ValueError(f"{label} task count disagrees with the confirmation run")
        if candidate["token_weighted"]["token_count"] != expected_token_count:
            raise ValueError(f"{label} token count disagrees with the confirmation run")

    if primary["policy"]["upgrade_layer"] != 0:
        raise ValueError("MBPP confirmation chart expects the frozen policy to upgrade layer 0")
    if mse["policy"]["upgrade_layer"] != primary["policy"]["upgrade_layer"]:
        raise ValueError("Read-risk and MSE selectors no longer choose the same layer")
    for result_key in ("task_macro", "token_weighted", "storage"):
        if mse[result_key] != primary[result_key]:
            raise ValueError(f"Coincident read-risk and MSE selectors disagree on {result_key}")

    expected_storage = {
        "uniform": 2_433_024,
        "primary": 2_564_096,
        "full_precision": 18_874_368,
    }
    for label, candidate, expected_bytes in (
        ("uniform INT4", uniform, expected_storage["uniform"]),
        ("mixed", primary, expected_storage["primary"]),
    ):
        storage = candidate["storage"]
        if not storage["exact_byte_gate"] or not storage["physical_reduction_realized"]:
            raise ValueError(f"{label} storage is not exact and physically realized")
        if storage["resident_bytes"] != expected_bytes:
            raise ValueError(f"{label} resident storage changed unexpectedly")
        if storage["expected_resident_bytes"] != storage["resident_bytes"]:
            raise ValueError(f"{label} measured and expected resident bytes disagree")
        if storage["full_precision_equivalent_bytes"] != expected_storage["full_precision"]:
            raise ValueError(f"{label} full-precision storage reference disagrees")
    if (
        evidence["validity"]["reference_recurrent_state_bytes"]
        != expected_storage["full_precision"]
    ):
        raise ValueError("Confirmation validity record has a different storage reference")

    uniform_macro = Decimal(str(uniform["task_macro"]["delta_nll"]))
    primary_macro = Decimal(str(primary["task_macro"]["delta_nll"]))
    relative_reduction = Decimal(str(primary_contrast["relative_reduction"]))
    computed_reduction = (uniform_macro - primary_macro) / uniform_macro
    if abs(relative_reduction - computed_reduction) > Decimal("1e-15"):
        raise ValueError("MBPP confirmation relative reduction does not match candidates")
    if f"{relative_reduction * Decimal(100):.2f}" != "72.75":
        raise ValueError("MBPP confirmation headline reduction changed unexpectedly")

    for label, contrast in (
        ("uniform", primary_contrast["paired_bootstrap"]),
        ("equal-byte random", random_contrast["paired_bootstrap"]),
    ):
        if contrast["paired_examples"] != expected_task_count:
            raise ValueError(f"{label} bootstrap task count disagrees with confirmation")
        if contrast["confidence"] != 0.95:
            raise ValueError(f"{label} confirmation interval must be a 95% interval")
        low, high = contrast["confidence_interval"]
        if not (0 < low <= contrast["mean_improvement"] <= high):
            raise ValueError(f"{label} paired interval does not support the chart claim")

    return {
        "document_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_evidence_sha256": canonical,
        "phase": claim_scope["phase"],
        "confirmation_touched": claim_scope["confirmation_touched"],
        "protocol_eligible": claim_scope["protocol_eligible"],
        "teacher_forced_fidelity_only": claim_scope["teacher_forced_fidelity_only"],
        "generated_code_executed": claim_scope["generated_code_executed"],
        "speed_claim_allowed": claim_scope["speed_claim_allowed"],
        "whole_model_memory_claim_allowed": claim_scope["whole_model_memory_claim_allowed"],
        "all_quality_gates_pass": continuation["all_gates_pass"],
        "task_count": expected_task_count,
        "token_count": expected_token_count,
        "uniform_macro_delta_nll": uniform["task_macro"]["delta_nll"],
        "primary_macro_delta_nll": primary["task_macro"]["delta_nll"],
        "relative_reduction": primary_contrast["relative_reduction"],
        "primary_vs_uniform": primary_contrast["paired_bootstrap"],
        "primary_vs_mean_random_equal_byte": random_contrast["paired_bootstrap"],
        "primary_resident_bytes": primary["storage"]["resident_bytes"],
        "uniform_resident_bytes": uniform["storage"]["resident_bytes"],
        "full_precision_equivalent_bytes": expected_storage["full_precision"],
        "primary_upgrade_layer": primary["policy"]["upgrade_layer"],
        "mse_upgrade_layer": mse["policy"]["upgrade_layer"],
    }


def _metadata(
    payload: dict[str, object],
    *,
    source_document: str = "research/EXPERIMENT_002_SCALE_CORRECTION.md",
) -> str:
    record = {
        "generated_by": "scripts/generate_readme_assets.py",
        "source_document": source_document,
        **payload,
    }
    encoded = json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return html.escape(encoded, quote=False)


def _storage_svg(data: dict[str, object]) -> str:
    rows = data["storage"]
    assert isinstance(rows, list)
    evidence_hashes = data["evidence_hashes"]
    assert isinstance(evidence_hashes, dict)

    width = Decimal(600)
    axis_max = Decimal(20_000_000)
    bar_x = Decimal(250)
    y_positions = (Decimal(95), Decimal(171), Decimal(247))
    row_markup = []
    colors = ("reference", "uniform", "mixed")
    for row, y, color in zip(rows, y_positions, colors, strict=True):
        assert isinstance(row, dict)
        resident = _number(str(row["bytes"]))
        bar_width = resident / axis_max * width
        ratio = str(row["ratio"])
        ratio_label = "1.000x reference" if color == "reference" else ratio
        row_markup.append(
            f'''  <g>
    <text class="label" x="232" y="{y + 13}" text-anchor="end">{html.escape(str(row["label"]))}</text>
    <text class="muted" x="232" y="{y + 32}" text-anchor="end">{html.escape(str(row["detail"]))} · {html.escape(ratio_label)}</text>
    <rect class="bar {color}" x="250" y="{y}" width="{bar_width:.2f}" height="30">
      <title>{html.escape(str(row["label"]))}: {resident:,} resident bytes; {html.escape(ratio_label)}</title>
    </rect>
    <text class="value" x="{bar_x + bar_width + 9:.2f}" y="{y + 20}">{resident:,}</text>
  </g>'''
        )

    ticks = []
    for million in range(0, 21, 5):
        x = bar_x + Decimal(million) / Decimal(20) * width
        ticks.append(
            f'''  <line class="grid" x1="{x}" y1="80" x2="{x}" y2="300" />
  <text class="tick" x="{x}" y="320" text-anchor="middle">{million}</text>'''
        )

    metadata = _metadata(
        {
            "chart": "resident-recurrent-state-bytes",
            "source_document_sha256": data["document_sha256"],
            "canonical_evidence_sha256": {
                "packed_multilingual_replay": evidence_hashes["multilingual_packed"]
            },
            "values": rows,
        }
    )
    fp32, uniform, mixed = rows
    storage_desc = (
        f"At batch one, FP32 recurrent states occupy {fp32['bytes']} bytes. "
        f"Uniform INT4 with FP16 scales occupies {uniform['bytes']} bytes, "
        f"{_number(str(uniform['ratio']))} times smaller. Keeping layer zero at INT8 "
        f"and the other 17 layers at INT4 occupies {mixed['bytes']} bytes, "
        f"{_number(str(mixed['ratio']))} times smaller. These are persistent state "
        "bytes only, not whole-model or peak memory."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="365" viewBox="0 0 960 365" role="img" aria-labelledby="storage-title storage-desc">
  <title id="storage-title">Resident recurrent-state storage on Qwen3.5-0.8B-Base</title>
  <desc id="storage-desc">{html.escape(storage_desc)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <style>
    svg {{ background-color: #ffffff; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #172033; }}
    .title {{ font-size: 22px; font-weight: 500; }}
    .subtitle, .muted, .tick, .note {{ fill: #556176; }}
    .subtitle {{ font-size: 14px; }}
    .label, .value {{ font-size: 14px; font-weight: 500; }}
    .muted, .tick, .note {{ font-size: 12px; }}
    .grid {{ stroke: #d5dae3; stroke-width: 1; }}
    .bar {{ shape-rendering: geometricPrecision; }}
    .reference {{ fill: #64748b; }}
    .uniform {{ fill: #2563eb; }}
    .mixed {{ fill: #d97706; }}
    @media (prefers-color-scheme: dark) {{
      svg {{ background-color: #0d1117; }}
      text {{ fill: #e6edf3; }}
      .subtitle, .muted, .tick, .note {{ fill: #a9b4c2; }}
      .grid {{ stroke: #3a4555; }}
      .reference {{ fill: #94a3b8; }}
      .uniform {{ fill: #60a5fa; }}
      .mixed {{ fill: #fbbf24; }}
    }}
  </style>
  <text class="title" x="28" y="30">Resident recurrent-state storage</text>
  <text class="subtitle" x="28" y="54">Qwen3.5-0.8B-Base · batch 1 · resident bytes (millions) · lower is better</text>
{chr(10).join(ticks)}
{chr(10).join(row_markup)}
  <text class="note" x="28" y="340">Persistent recurrent state only; excludes model weights and six standard KV caches.</text>
  <text class="note" x="28" y="357">The Python path materializes one layer state; no peak-memory or speed claim.</text>
</svg>
"""


def _diagnostic_svg(data: dict[str, object]) -> str:
    rows = data["diagnostics"]
    assert isinstance(rows, list)
    evidence_hashes = data["evidence_hashes"]
    assert isinstance(evidence_hashes, dict)

    plot_top = Decimal(96)
    plot_bottom = Decimal(350)
    plot_height = plot_bottom - plot_top
    axis_max = Decimal(8)
    centers = (Decimal(230), Decimal(480), Decimal(730))
    bar_width = Decimal(58)
    series_gap = Decimal(9)
    groups = []
    for row, center in zip(rows, centers, strict=True):
        assert isinstance(row, dict)
        uniform = _number(str(row["uniform"]))
        mixed = _number(str(row["mixed"]))
        uniform_height = uniform / axis_max * plot_height
        mixed_height = mixed / axis_max * plot_height
        uniform_x = center - bar_width - series_gap
        mixed_x = center + series_gap
        uniform_y = plot_bottom - uniform_height
        mixed_y = plot_bottom - mixed_height
        groups.append(
            f'''  <g>
    <rect class="bar uniform" x="{uniform_x}" y="{uniform_y:.2f}" width="{bar_width}" height="{uniform_height:.2f}">
      <title>{html.escape(str(row["label"]))}, uniform INT4: {uniform} CVaR95 KL</title>
    </rect>
    <text class="value" x="{uniform_x + bar_width / 2}" y="{uniform_y - 8:.2f}" text-anchor="middle">{uniform}</text>
    <rect class="bar mixed" x="{mixed_x}" y="{mixed_y:.2f}" width="{bar_width}" height="{mixed_height:.2f}">
      <title>{html.escape(str(row["label"]))}, mixed layer zero INT8: {mixed} CVaR95 KL</title>
    </rect>
    <text class="value" x="{mixed_x + bar_width / 2}" y="{mixed_y - 8:.2f}" text-anchor="middle">{mixed}</text>
    <text class="label" x="{center}" y="378" text-anchor="middle">{html.escape(str(row["label"]))}</text>
    <text class="reduction" x="{center}" y="399" text-anchor="middle">−{html.escape(str(row["reduction"]))} mixed vs uniform</text>
  </g>'''
        )

    grid = []
    for tick in range(0, 9, 2):
        y = plot_bottom - Decimal(tick) / axis_max * plot_height
        grid.append(
            f'''  <line class="grid" x1="82" y1="{y:.2f}" x2="900" y2="{y:.2f}" />
  <text class="tick" x="70" y="{y + 4:.2f}" text-anchor="end">{tick}</text>'''
        )

    metadata = _metadata(
        {
            "chart": "corrected-diagnostic-cvar95-kl",
            "source_document_sha256": data["document_sha256"],
            "canonical_evidence_sha256": {
                "retrieval_sweep": evidence_hashes["retrieval_sweep"],
                "code_sweep": evidence_hashes["code_sweep"],
                "multilingual_qdq": evidence_hashes["multilingual_qdq"],
                "multilingual_packed": evidence_hashes["multilingual_packed"],
            },
            "values": rows,
        }
    )
    comparisons = ", ".join(
        f"{row['uniform']} versus {row['mixed']} for {str(row['label']).lower()}" for row in rows
    )
    diagnostic_desc = (
        "On short diagnostic traces, uniform INT4 versus layer zero INT8 and the "
        f"remaining layers INT4 produced CVaR95 KL of {comparisons}. Lower is better. "
        "These diagnostics are not a public benchmark or breakthrough evidence."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="465" viewBox="0 0 960 465" role="img" aria-labelledby="kl-title kl-desc">
  <title id="kl-title">Corrected diagnostic tail KL by recurrent-state precision policy</title>
  <desc id="kl-desc">{html.escape(diagnostic_desc)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <style>
    svg {{ background-color: #ffffff; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #172033; }}
    .title {{ font-size: 22px; font-weight: 500; }}
    .subtitle, .tick, .note {{ fill: #556176; }}
    .subtitle {{ font-size: 14px; }}
    .label, .value {{ font-size: 13px; font-weight: 500; }}
    .tick, .note, .reduction, .legend {{ font-size: 12px; }}
    .reduction {{ fill: #6b4c15; }}
    .grid {{ stroke: #d5dae3; stroke-width: 1; }}
    .bar {{ shape-rendering: geometricPrecision; }}
    .uniform {{ fill: #2563eb; }}
    .mixed {{ fill: #d97706; }}
    @media (prefers-color-scheme: dark) {{
      svg {{ background-color: #0d1117; }}
      text {{ fill: #e6edf3; }}
      .subtitle, .tick, .note {{ fill: #a9b4c2; }}
      .reduction {{ fill: #f6c866; }}
      .grid {{ stroke: #3a4555; }}
      .uniform {{ fill: #60a5fa; }}
      .mixed {{ fill: #fbbf24; }}
    }}
  </style>
  <text class="title" x="28" y="30">Corrected diagnostic tail KL</text>
  <text class="subtitle" x="28" y="54">CVaR95 token KL · stored FP16 scales · lower is better</text>
  <rect class="uniform" x="650" y="69" width="16" height="12" />
  <text class="legend" x="674" y="79">Uniform INT4</text>
  <rect class="mixed" x="784" y="69" width="16" height="12" />
  <text class="legend" x="808" y="79">L0 INT8, rest INT4</text>
{chr(10).join(grid)}
{chr(10).join(groups)}
  <text class="subtitle" x="24" y="223" text-anchor="middle" transform="rotate(-90 24 223)">CVaR95 KL</text>
  <text class="note" x="28" y="435">Short paired diagnostics, not a public benchmark or breakthrough claim.</text>
  <text class="note" x="28" y="452">Multilingual is a correction replay; packed and QDQ token metrics matched exactly.</text>
</svg>
"""


def _development_svg(data: dict[str, object]) -> str:
    uniform = Decimal(str(data["uniform_macro_delta_nll"]))
    primary = Decimal(str(data["primary_macro_delta_nll"]))
    reduction = Decimal(str(data["relative_reduction"])) * Decimal(100)
    primary_contrast = data["primary_vs_uniform"]
    random_contrast = data["primary_vs_mean_random_equal_byte"]
    assert isinstance(primary_contrast, dict)
    assert isinstance(random_contrast, dict)

    task_count = int(data["task_count"])
    token_count = int(data["token_count"])
    for label, contrast in (
        ("uniform", primary_contrast),
        ("equal-byte random", random_contrast),
    ):
        if contrast["paired_examples"] != task_count:
            raise ValueError(f"{label} bootstrap task count disagrees with MBPP run")
        if contrast["confidence"] != 0.95:
            raise ValueError(f"{label} README interval must be a 95% interval")

    uniform_ci = [Decimal(str(value)) for value in primary_contrast["confidence_interval"]]
    random_ci = [Decimal(str(value)) for value in random_contrast["confidence_interval"]]
    uniform_mean = Decimal(str(primary_contrast["mean_improvement"]))
    random_mean = Decimal(str(random_contrast["mean_improvement"]))

    axis_max = Decimal("3.2")
    bar_x = Decimal(250)
    plot_width = Decimal(600)
    rows = (
        ("Uniform INT4", "all 18 recurrent layers INT4", uniform, "uniform", Decimal(110)),
        (
            "Mixed L0 INT8",
            "layer 0 INT8; other 17 layers INT4",
            primary,
            "mixed",
            Decimal(190),
        ),
    )
    row_markup = []
    for label, detail, value, color, y in rows:
        bar_width = value / axis_max * plot_width
        row_markup.append(
            f'''  <g>
    <text class="label" x="232" y="{y + 13}" text-anchor="end">{label}</text>
    <text class="muted" x="232" y="{y + 32}" text-anchor="end">{detail}</text>
    <rect class="bar {color}" x="{bar_x}" y="{y}" width="{bar_width:.2f}" height="34">
      <title>{label}: {value:.6f} task-macro excess NLL</title>
    </rect>
    <text class="value" x="{bar_x + bar_width + 9:.2f}" y="{y + 22}">{value:.4f}</text>
  </g>'''
        )

    ticks = []
    for tick in range(4):
        x = bar_x + Decimal(tick) / axis_max * plot_width
        ticks.append(
            f'''  <line class="grid" x1="{x:.2f}" y1="88" x2="{x:.2f}" y2="245" />
  <text class="tick" x="{x:.2f}" y="265" text-anchor="middle">{tick}</text>'''
        )

    metadata = _metadata(
        {
            "chart": "mbpp-development-task-macro-excess-nll",
            "source_document_sha256": data["document_sha256"],
            "canonical_evidence_sha256": data["canonical_evidence_sha256"],
            "phase": data["phase"],
            "confirmation_touched": data["confirmation_touched"],
            "teacher_forced_fidelity_only": data["teacher_forced_fidelity_only"],
            "generated_code_executed": data["generated_code_executed"],
            "speed_claim_allowed": data["speed_claim_allowed"],
            "whole_model_memory_claim_allowed": data["whole_model_memory_claim_allowed"],
            "task_count": task_count,
            "token_count": token_count,
            "uniform_macro_delta_nll": data["uniform_macro_delta_nll"],
            "primary_macro_delta_nll": data["primary_macro_delta_nll"],
            "relative_reduction": data["relative_reduction"],
            "primary_vs_uniform": primary_contrast,
            "primary_vs_mean_random_equal_byte": random_contrast,
            "primary_upgrade_layer": data["primary_upgrade_layer"],
            "mse_upgrade_layer": data["mse_upgrade_layer"],
        },
        source_document="evidence/mbpp-v02-development.json",
    )
    description = (
        f"On the MBPP validation development split, uniform INT4 had task-macro "
        f"excess negative log likelihood {uniform:.6f}, while keeping recurrent layer "
        f"zero at INT8 and the other 17 recurrent layers at INT4 had {primary:.6f}, "
        f"a {reduction:.2f} percent reduction. Across {task_count} paired tasks, the "
        f"mean improvement was {uniform_mean:.6f} nats per token with a 95 percent "
        f"bootstrap interval from {uniform_ci[0]:.6f} to {uniform_ci[1]:.6f}. This is "
        f"teacher-forced development evidence over {token_count} reference code tokens, "
        "not confirmation or generated-code correctness evidence."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="430" viewBox="0 0 960 430" role="img" aria-labelledby="mbpp-title mbpp-desc">
  <title id="mbpp-title">MBPP development task-macro excess negative log likelihood</title>
  <desc id="mbpp-desc">{html.escape(description)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <style>
    svg {{ background-color: #ffffff; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #172033; }}
    .title {{ font-size: 22px; font-weight: 500; }}
    .subtitle, .muted, .tick, .note {{ fill: #556176; }}
    .subtitle {{ font-size: 14px; }}
    .label, .value, .result {{ font-size: 14px; font-weight: 500; }}
    .muted, .tick, .note {{ font-size: 12px; }}
    .grid {{ stroke: #d5dae3; stroke-width: 1; }}
    .bar {{ shape-rendering: geometricPrecision; }}
    .uniform {{ fill: #2563eb; }}
    .mixed {{ fill: #d97706; }}
    @media (prefers-color-scheme: dark) {{
      svg {{ background-color: #0d1117; }}
      text {{ fill: #e6edf3; }}
      .subtitle, .muted, .tick, .note {{ fill: #a9b4c2; }}
      .grid {{ stroke: #3a4555; }}
      .uniform {{ fill: #60a5fa; }}
      .mixed {{ fill: #fbbf24; }}
    }}
  </style>
  <text class="title" x="28" y="30">MBPP development fidelity</text>
  <text class="subtitle" x="28" y="54">Task-macro excess NLL (nats/token) &#183; lower is better</text>
{chr(10).join(ticks)}
{chr(10).join(row_markup)}
  <text class="result" x="28" y="302">{reduction:.2f}% lower macro excess NLL</text>
  <text class="note" x="28" y="326">Paired mixed-vs-uniform improvement: {uniform_mean:.4f}; 95% bootstrap CI {uniform_ci[0]:.4f}&#8211;{uniform_ci[1]:.4f}.</text>
  <text class="note" x="28" y="350">Equal-byte mixed-vs-mean-random improvement: {random_mean:.4f}; 95% bootstrap CI {random_ci[0]:.4f}&#8211;{random_ci[1]:.4f}.</text>
  <text class="note" x="28" y="386">MBPP validation development split &#183; {task_count} tasks &#183; {token_count:,} teacher-forced reference code tokens.</text>
  <text class="note" x="28" y="410">Development evidence only; no confirmation, generated-code correctness, speed, or whole-model memory claim.</text>
</svg>
"""


def _confirmation_svg(data: dict[str, object]) -> str:
    uniform = Decimal(str(data["uniform_macro_delta_nll"]))
    primary = Decimal(str(data["primary_macro_delta_nll"]))
    reduction = Decimal(str(data["relative_reduction"])) * Decimal(100)
    primary_contrast = data["primary_vs_uniform"]
    random_contrast = data["primary_vs_mean_random_equal_byte"]
    assert isinstance(primary_contrast, dict)
    assert isinstance(random_contrast, dict)

    task_count = int(data["task_count"])
    token_count = int(data["token_count"])
    uniform_ci = [Decimal(str(value)) for value in primary_contrast["confidence_interval"]]
    random_ci = [Decimal(str(value)) for value in random_contrast["confidence_interval"]]
    uniform_mean = Decimal(str(primary_contrast["mean_improvement"]))
    random_mean = Decimal(str(random_contrast["mean_improvement"]))

    axis_max = Decimal("3.2")
    bar_x = Decimal(250)
    plot_width = Decimal(600)
    rows = (
        (
            "Uniform INT4",
            "all 18 recurrent layers INT4",
            uniform,
            "uniform",
            Decimal(110),
        ),
        (
            "Mixed L0 INT8",
            "layer 0 INT8; other 17 layers INT4",
            primary,
            "mixed",
            Decimal(190),
        ),
    )
    row_markup = []
    for label, detail, value, color, y in rows:
        bar_width = value / axis_max * plot_width
        row_markup.append(
            f'''  <g>
    <text class="label" x="232" y="{y + 13}" text-anchor="end">{label}</text>
    <text class="muted" x="232" y="{y + 32}" text-anchor="end">{detail}</text>
    <rect class="bar {color}" x="{bar_x}" y="{y}" width="{bar_width:.2f}" height="34">
      <title>{label}: {value:.6f} task-macro excess NLL</title>
    </rect>
    <text class="value" x="{bar_x + bar_width + 9:.2f}" y="{y + 22}">{value:.4f}</text>
  </g>'''
        )

    ticks = []
    for tick in range(4):
        x = bar_x + Decimal(tick) / axis_max * plot_width
        ticks.append(
            f'''  <line class="grid" x1="{x:.2f}" y1="88" x2="{x:.2f}" y2="245" />
  <text class="tick" x="{x:.2f}" y="265" text-anchor="middle">{tick}</text>'''
        )

    exact_storage = {
        "mixed_resident_bytes": data["primary_resident_bytes"],
        "uniform_int4_resident_bytes": data["uniform_resident_bytes"],
        "fp32_recurrent_reference_bytes": data["full_precision_equivalent_bytes"],
    }
    metadata = _metadata(
        {
            "chart": "mbpp-confirmation-task-macro-excess-nll",
            "source_document_sha256": data["document_sha256"],
            "canonical_evidence_sha256": data["canonical_evidence_sha256"],
            "phase": data["phase"],
            "confirmation_touched": data["confirmation_touched"],
            "protocol_eligible": data["protocol_eligible"],
            "teacher_forced_fidelity_only": data["teacher_forced_fidelity_only"],
            "generated_code_executed": data["generated_code_executed"],
            "speed_claim_allowed": data["speed_claim_allowed"],
            "whole_model_memory_claim_allowed": data["whole_model_memory_claim_allowed"],
            "all_quality_gates_pass": data["all_quality_gates_pass"],
            "task_count": task_count,
            "token_count": token_count,
            "uniform_macro_delta_nll": data["uniform_macro_delta_nll"],
            "primary_macro_delta_nll": data["primary_macro_delta_nll"],
            "relative_reduction": data["relative_reduction"],
            "primary_vs_uniform": primary_contrast,
            "primary_vs_mean_random_equal_byte": random_contrast,
            "exact_storage": exact_storage,
            "primary_upgrade_layer": data["primary_upgrade_layer"],
            "mse_upgrade_layer": data["mse_upgrade_layer"],
        },
        source_document="evidence/mbpp-v02-confirmation.json",
    )
    description = (
        f"On the held-out MBPP confirmation split, uniform INT4 had task-macro "
        f"excess negative log likelihood {uniform:.6f}, while keeping recurrent layer "
        f"zero at INT8 and the other 17 recurrent layers at INT4 had {primary:.6f}, "
        f"a {reduction:.2f} percent reduction. Across {task_count} paired tasks, the "
        f"mean improvement was {uniform_mean:.6f} nats per token with a 95 percent "
        f"bootstrap interval from {uniform_ci[0]:.6f} to {uniform_ci[1]:.6f}. This is "
        f"teacher-forced fidelity evidence over {token_count} reference code tokens. "
        "Generated code was not executed, and the evidence supports no speed or "
        "whole-model memory claim."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480" role="img" aria-labelledby="mbpp-confirmation-title mbpp-confirmation-desc">
  <title id="mbpp-confirmation-title">MBPP held-out confirmation task-macro excess negative log likelihood</title>
  <desc id="mbpp-confirmation-desc">{html.escape(description)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <style>
    svg {{ background-color: #ffffff; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #172033; }}
    .title {{ font-size: 22px; font-weight: 500; }}
    .subtitle, .muted, .tick, .note {{ fill: #556176; }}
    .subtitle {{ font-size: 14px; }}
    .label, .value, .result {{ font-size: 14px; font-weight: 500; }}
    .muted, .tick, .note {{ font-size: 12px; }}
    .grid {{ stroke: #d5dae3; stroke-width: 1; }}
    .bar {{ shape-rendering: geometricPrecision; }}
    .uniform {{ fill: #2563eb; }}
    .mixed {{ fill: #d97706; }}
    @media (prefers-color-scheme: dark) {{
      svg {{ background-color: #0d1117; }}
      text {{ fill: #e6edf3; }}
      .subtitle, .muted, .tick, .note {{ fill: #a9b4c2; }}
      .grid {{ stroke: #3a4555; }}
      .uniform {{ fill: #60a5fa; }}
      .mixed {{ fill: #fbbf24; }}
    }}
  </style>
  <text class="title" x="28" y="30">MBPP held-out confirmation fidelity</text>
  <text class="subtitle" x="28" y="54">Task-macro excess NLL (nats/token) &#183; teacher-forced &#183; lower is better</text>
{chr(10).join(ticks)}
{chr(10).join(row_markup)}
  <text class="result" x="28" y="302">{reduction:.2f}% lower macro excess NLL</text>
  <text class="note" x="28" y="328">Paired mixed-vs-uniform improvement: {uniform_mean:.4f}; 95% bootstrap CI {uniform_ci[0]:.4f}&#8211;{uniform_ci[1]:.4f}.</text>
  <text class="note" x="28" y="352">Equal-byte mixed-vs-mean-random improvement: {random_mean:.4f}; 95% bootstrap CI {random_ci[0]:.4f}&#8211;{random_ci[1]:.4f}.</text>
  <text class="note" x="28" y="382">Read-risk and MSE selectors both chose recurrent layer 0.</text>
  <text class="note" x="28" y="406">Exact resident recurrent-state bytes: mixed {int(data["primary_resident_bytes"]):,}; uniform INT4 {int(data["uniform_resident_bytes"]):,}; FP32 reference {int(data["full_precision_equivalent_bytes"]):,}.</text>
  <text class="note" x="28" y="436">Held-out MBPP confirmation &#183; {task_count} tasks &#183; {token_count:,} teacher-forced reference code tokens.</text>
  <text class="note" x="28" y="460">Generated code was not executed; no speed or whole-model memory claim.</text>
</svg>
"""


def _load_validated_stage_b_result(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load Stage B through its strict canonical and semantic verifier."""

    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from scripts.evaluate_rht_cqer_stage_b import (  # noqa: PLC0415
        load_and_validate_stage_b_result_artifact,
    )

    return load_and_validate_stage_b_result_artifact(path)


def _stage_b_data() -> dict[str, object]:
    if not STAGE_B_SOURCE.is_file():
        raise FileNotFoundError(
            "The authenticated Experiment 009 Stage-B raw result is required to "
            "generate its README assets. Download "
            f"{STAGE_B_RELEASE_DOWNLOAD}, verify SHA256 "
            f"{STAGE_B_ARTIFACT_SHA256}, and extract the JSON to "
            f"{STAGE_B_SOURCE.relative_to(ROOT)}. A clean clone can still run "
            "`python scripts/generate_readme_assets.py --check` against the "
            "tracked release manifest."
        )
    # This call is intentionally first: no chart field is read until the complete
    # canonical and semantic audit has succeeded.
    evidence, verification = _load_validated_stage_b_result(STAGE_B_SOURCE)

    expected_verification = {
        "passed": True,
        "integrity_passed": True,
        "advancement_passed": True,
        "task_count": 32,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2339,
        "advancement_check_count": 8,
        "artifact_sha256": STAGE_B_ARTIFACT_SHA256,
        "canonical_evidence_sha256": STAGE_B_CANONICAL_SHA256,
        "canonical_round_trip": True,
    }
    if verification != expected_verification:
        raise ValueError("Experiment 009 Stage-B verification record drifted")
    if evidence.get("artifact_kind") != STAGE_B_ARTIFACT_KIND:
        raise ValueError("Unexpected Experiment 009 Stage-B artifact kind")
    if evidence.get("methods") != list(STAGE_B_METHODS):
        raise ValueError("Experiment 009 Stage-B method order drifted")

    repository = evidence.get("repository")
    if not isinstance(repository, dict):
        raise ValueError("Experiment 009 Stage-B repository record is missing")
    if (
        repository.get("commit") != STAGE_B_RESULT_COMMIT
        or repository.get("stable_commit") is not True
    ):
        raise ValueError("Experiment 009 Stage-B result commit drifted")

    integrity = evidence.get("stage_b_integrity")
    gate = evidence.get("stage_b_gate")
    if not isinstance(integrity, dict) or integrity.get("passed") is not True:
        raise ValueError("Experiment 009 Stage-B integrity did not pass")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("Experiment 009 Stage-B advancement gate did not pass")

    dataset = evidence.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("identity"), dict):
        raise ValueError("Experiment 009 Stage-B identity is missing")
    ordered_task_ids = dataset["identity"].get("ordered_task_ids")
    if (
        not isinstance(ordered_task_ids, list)
        or len(ordered_task_ids) != 32
        or any(type(task_id) is not int for task_id in ordered_task_ids)
    ):
        raise ValueError("Experiment 009 Stage-B frozen task order drifted")

    aggregates = evidence.get("aggregates")
    per_task = evidence.get("per_task")
    if (
        not isinstance(aggregates, dict)
        or set(aggregates) != set(STAGE_B_METHODS)
        or not isinstance(per_task, dict)
        or set(per_task) != set(STAGE_B_METHODS)
    ):
        raise ValueError("Experiment 009 Stage-B metrics do not match frozen methods")

    macro_delta_nll: dict[str, float] = {}
    for method in STAGE_B_METHODS:
        aggregate = aggregates[method]
        rows = per_task[method]
        if not isinstance(aggregate, dict) or not isinstance(rows, list):
            raise ValueError(f"Experiment 009 Stage-B {method} metrics are malformed")
        if [row.get("task_id") for row in rows] != ordered_task_ids:
            raise ValueError(f"Experiment 009 Stage-B {method} task order drifted")
        value = aggregate.get("macro_delta_nll")
        if type(value) not in {float, int} or not math.isfinite(float(value)):
            raise ValueError(f"Experiment 009 Stage-B {method} macro NLL is invalid")
        macro_delta_nll[method] = float(value)

    state_error = evidence.get("state_error")
    if not isinstance(state_error, dict) or not isinstance(
        state_error.get("aggregates"),
        dict,
    ):
        raise ValueError("Experiment 009 Stage-B state-error aggregates are missing")
    state_aggregates = state_error["aggregates"]
    if set(state_aggregates) != {STAGE_B_CQER_METHOD, STAGE_B_RHT_METHOD}:
        raise ValueError("Experiment 009 Stage-B state-error method set drifted")
    state_sse: dict[str, float] = {}
    for method in (STAGE_B_CQER_METHOD, STAGE_B_RHT_METHOD):
        record = state_aggregates[method]
        if not isinstance(record, dict):
            raise ValueError(f"Experiment 009 Stage-B {method} state error is malformed")
        value = record.get("aggregate_state_sse")
        if type(value) not in {float, int} or not math.isfinite(float(value)):
            raise ValueError(f"Experiment 009 Stage-B {method} state SSE is invalid")
        state_sse[method] = float(value)

    cqer_rows = per_task[STAGE_B_CQER_METHOD]
    rht_rows = per_task[STAGE_B_RHT_METHOD]
    paired = []
    for task_id, cqer_row, rht_row in zip(
        ordered_task_ids,
        cqer_rows,
        rht_rows,
        strict=True,
    ):
        cqer_value = float(cqer_row["delta_nll"])
        rht_value = float(rht_row["delta_nll"])
        improvement = cqer_value - rht_value
        paired.append(
            {
                "task_id": task_id,
                "cqer_delta_nll": cqer_value,
                "rht_delta_nll": rht_value,
                "cqer_minus_rht_delta_nll": improvement,
                "outcome": ("win" if improvement > 0 else "loss" if improvement < 0 else "tie"),
            }
        )

    wins = sum(point["outcome"] == "win" for point in paired)
    ties = sum(point["outcome"] == "tie" for point in paired)
    losses = sum(point["outcome"] == "loss" for point in paired)
    bootstrap = evidence.get("paired_bootstrap_cqer_minus_rht_aligned_delta_nll")
    if not isinstance(bootstrap, dict):
        raise ValueError("Experiment 009 Stage-B paired bootstrap is missing")
    mean_improvement = math.fsum(
        float(point["cqer_minus_rht_delta_nll"]) for point in paired
    ) / len(paired)
    if (
        bootstrap.get("paired_examples") != 32
        or bootstrap.get("bootstrap_samples") != 10_000
        or bootstrap.get("seed") != 2339
        or bootstrap.get("confidence") != 0.95
        or not math.isclose(
            float(bootstrap.get("mean_improvement", math.nan)),
            mean_improvement,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("Experiment 009 Stage-B paired bootstrap drifted")
    confidence_interval = bootstrap.get("confidence_interval")
    if (
        not isinstance(confidence_interval, list)
        or len(confidence_interval) != 2
        or not all(
            type(value) in {float, int} and math.isfinite(float(value))
            for value in confidence_interval
        )
    ):
        raise ValueError("Experiment 009 Stage-B paired interval is invalid")

    win_check = gate.get("advancement_checks", {}).get(
        "at_least_20_task_level_excess_nll_wins",
        {},
    )
    if (
        win_check.get("rht_wins") != wins
        or win_check.get("ties") != ties
        or wins + ties + losses != 32
    ):
        raise ValueError("Experiment 009 Stage-B paired outcomes drifted")

    return {
        "artifact_kind": STAGE_B_ARTIFACT_KIND,
        "artifact_sha256": STAGE_B_ARTIFACT_SHA256,
        "canonical_evidence_sha256": STAGE_B_CANONICAL_SHA256,
        "result_commit": STAGE_B_RESULT_COMMIT,
        "verifier_fix_commit": STAGE_B_VERIFIER_FIX_COMMIT,
        "verification": verification,
        "integrity_passed": True,
        "gate_passed": True,
        "claim_boundary": evidence["claim_boundary"],
        "ordered_task_ids": ordered_task_ids,
        "macro_delta_nll": macro_delta_nll,
        "state_sse": state_sse,
        "paired": paired,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap": bootstrap,
    }


def _stage_b_metadata(
    data: dict[str, object],
    *,
    chart: str,
    values: object,
) -> str:
    return _metadata(
        {
            "chart": chart,
            "artifact_kind": data["artifact_kind"],
            "source_artifact_sha256": data["artifact_sha256"],
            "canonical_evidence_sha256": data["canonical_evidence_sha256"],
            "original_result_commit": data["result_commit"],
            "verifier_fix_commit": data["verifier_fix_commit"],
            "integrity_passed": data["integrity_passed"],
            "advancement_gate_passed": data["gate_passed"],
            "semantic_verification": data["verification"],
            "claim_boundary": data["claim_boundary"],
            "values": values,
        },
        source_document=("artifacts/experiment009-rht-cqer-stage-b-result-cdc603b.json"),
    )


def _stage_b_overview_svg(data: dict[str, object]) -> str:
    macro = data["macro_delta_nll"]
    state_sse = data["state_sse"]
    assert isinstance(macro, dict)
    assert isinstance(state_sse, dict)

    method_rows = (
        ("Directional Fisher INT4", STAGE_B_METHODS[0], "baseline"),
        ("Adaptive MSE quota", STAGE_B_METHODS[1], "baseline"),
        ("CQER-32", STAGE_B_CQER_METHOD, "cqer"),
        ("right-RHT + CQER-32", STAGE_B_RHT_METHOD, "rht"),
    )
    macro_axis_max = Decimal("0.5")
    macro_x = Decimal(215)
    macro_width = Decimal(330)
    macro_rows = []
    for index, (label, method, css_class) in enumerate(method_rows):
        value = Decimal(str(macro[method]))
        y = Decimal(132 + index * 66)
        width = value / macro_axis_max * macro_width
        macro_rows.append(
            f'''  <g>
    <text class="label" x="32" y="{y + 18}">{html.escape(label)}</text>
    <rect class="track" x="{macro_x}" y="{y}" width="{macro_width}" height="28" />
    <rect class="bar {css_class}" x="{macro_x}" y="{y}" width="{width:.2f}" height="28">
      <title>{html.escape(label)}: {value:.6f} task-macro aligned excess NLL</title>
    </rect>
    <text class="value" x="{macro_x + width + 8:.2f}" y="{y + 19}">{value:.4f}</text>
  </g>'''
        )

    cqer_sse = Decimal(str(state_sse[STAGE_B_CQER_METHOD]))
    rht_sse = Decimal(str(state_sse[STAGE_B_RHT_METHOD]))
    state_axis_max = Decimal("40000")
    state_x = Decimal(665)
    state_width = Decimal(240)
    state_rows = []
    for label, value, css_class, y in (
        ("CQER-32", cqer_sse, "cqer", Decimal(160)),
        ("right-RHT + CQER-32", rht_sse, "rht", Decimal(250)),
    ):
        width = value / state_axis_max * state_width
        state_rows.append(
            f'''  <g>
    <text class="label" x="{state_x}" y="{y - 10}">{html.escape(label)}</text>
    <rect class="track" x="{state_x}" y="{y}" width="{state_width}" height="30" />
    <rect class="bar {css_class}" x="{state_x}" y="{y}" width="{width:.2f}" height="30">
      <title>{html.escape(label)}: {value:.6f} aggregate local codec state SSE</title>
    </rect>
    <text class="value" x="{state_x + width + 8:.2f}" y="{y + 21}">{value:,.0f}</text>
  </g>'''
        )

    macro_reduction = (
        Decimal(1)
        - Decimal(str(macro[STAGE_B_RHT_METHOD])) / Decimal(str(macro[STAGE_B_CQER_METHOD]))
    ) * Decimal(100)
    state_reduction = (Decimal(1) - rht_sse / cqer_sse) * Decimal(100)
    values = {
        "macro_delta_nll": macro,
        "state_sse": state_sse,
        "cqer_to_rht_macro_delta_nll_reduction": float(macro_reduction / Decimal(100)),
        "cqer_to_rht_state_sse_reduction": float(state_reduction / Decimal(100)),
    }
    metadata = _stage_b_metadata(
        data,
        chart="experiment009-stage-b-overview",
        values=values,
    )
    description = (
        "Experiment 009 Stage B compares four frozen methods on 32 MBPP development "
        f"tasks. CQER-32 has task-macro aligned excess NLL "
        f"{Decimal(str(macro[STAGE_B_CQER_METHOD])):.6f}; composing it with a "
        f"right randomized Hadamard transform has {Decimal(str(macro[STAGE_B_RHT_METHOD])):.6f}, "
        f"{macro_reduction:.2f} percent lower. Aggregate local codec state SSE falls "
        f"from {cqer_sse:.6f} to {rht_sse:.6f}, {state_reduction:.2f} percent lower. "
        "All eight frozen development advancement gates passed. This is not "
        "confirmation, novelty, speed, deployment, or breakthrough evidence."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500" role="img" aria-labelledby="stage-b-overview-title stage-b-overview-desc">
  <title id="stage-b-overview-title">Experiment 009 Stage-B authenticated development overview</title>
  <desc id="stage-b-overview-desc">{html.escape(description)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <defs>
    <linearGradient id="stage-b-rht-gradient" x1="0" x2="1">
      <stop offset="0" stop-color="#42c8c5" />
      <stop offset="1" stop-color="#80ebe5" />
    </linearGradient>
  </defs>
  <style>
    svg {{ background: #071118; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #eef8f8; }}
    .title {{ font-size: 23px; font-weight: 500; }}
    .subtitle, .section, .note, .axis-note {{ fill: #91adb2; }}
    .subtitle {{ font-size: 13px; letter-spacing: 0.5px; }}
    .section {{ font-size: 12px; letter-spacing: 1.3px; text-transform: uppercase; }}
    .label, .value, .result {{ font-size: 13px; font-weight: 500; }}
    .note, .axis-note {{ font-size: 11px; }}
    .track {{ fill: #0c2028; stroke: #173943; stroke-width: 1; }}
    .bar {{ shape-rendering: geometricPrecision; }}
    .baseline {{ fill: #376774; }}
    .cqer {{ fill: #9b82e8; }}
    .rht {{ fill: url(#stage-b-rht-gradient); }}
    .divider, .matrix {{ stroke: #20515b; stroke-width: 1; }}
    .matrix {{ fill: none; opacity: 0.34; }}
  </style>
  <rect class="matrix" x="832" y="18" width="18" height="18" />
  <rect class="matrix" x="856" y="18" width="18" height="18" />
  <rect class="matrix" x="880" y="18" width="18" height="18" />
  <rect class="matrix" x="832" y="42" width="18" height="18" />
  <rect class="matrix" x="856" y="42" width="18" height="18" />
  <rect class="matrix" x="880" y="42" width="18" height="18" />
  <text class="title" x="28" y="34">Experiment 009 &#183; Stage B development</text>
  <text class="subtitle" x="28" y="58">AUTHENTICATED 32-TASK RESULT &#183; LOWER IS BETTER</text>
  <line class="divider" x1="28" y1="78" x2="932" y2="78" />
  <text class="section" x="28" y="104">Task-macro aligned excess NLL</text>
  <text class="axis-note" x="545" y="104" text-anchor="end">nats / token</text>
  <text class="section" x="632" y="104">Local codec state SSE</text>
  <text class="axis-note" x="905" y="104" text-anchor="end">aggregate</text>
  <line class="divider" x1="603" y1="96" x2="603" y2="386" />
{chr(10).join(macro_rows)}
{chr(10).join(state_rows)}
  <text class="result" x="632" y="342">{macro_reduction:.2f}% lower macro excess NLL</text>
  <text class="result" x="632" y="368">{state_reduction:.2f}% lower state SSE</text>
  <line class="divider" x1="28" y1="410" x2="932" y2="410" />
  <text class="note" x="28" y="438">Frozen MBPP development window &#183; 32 tasks &#183; 1,956 aligned reference tokens &#183; 8/8 advancement gates passed</text>
  <text class="note" x="28" y="466">Diagnostic development evidence only; no confirmation, novelty, speed, deployment, or breakthrough claim.</text>
</svg>
"""


def _stage_b_paired_svg(data: dict[str, object]) -> str:
    points = data["paired"]
    bootstrap = data["bootstrap"]
    assert isinstance(points, list)
    assert isinstance(bootstrap, dict)

    plot_x = Decimal(62)
    plot_width = Decimal(854)
    plot_top = Decimal(164)
    plot_height = Decimal(286)
    domain_min = Decimal("-0.1")
    domain_max = Decimal("0.6")

    def y_position(value: Decimal) -> Decimal:
        return plot_top + (domain_max - value) / (domain_max - domain_min) * plot_height

    zero_y = y_position(Decimal(0))
    step = plot_width / Decimal(len(points))
    bar_width = Decimal(15)
    bars = []
    labels = []
    for index, point in enumerate(points):
        assert isinstance(point, dict)
        value = Decimal(str(point["cqer_minus_rht_delta_nll"]))
        center = plot_x + (Decimal(index) + Decimal("0.5")) * step
        value_y = y_position(value)
        rect_y = min(value_y, zero_y)
        height = abs(zero_y - value_y)
        if height == 0:
            height = Decimal("1")
        outcome = str(point["outcome"])
        bars.append(
            f'''  <rect class="paired-bar {outcome}" x="{center - bar_width / 2:.2f}" y="{rect_y:.2f}" width="{bar_width}" height="{height:.2f}">
    <title>Task {point["task_id"]}: CQER minus right-RHT delta NLL {value:+.6f}; {outcome}</title>
  </rect>'''
        )
        labels.append(
            f"""  <text class="task" transform="translate({center:.2f} 469) rotate(58)">{point["task_id"]}</text>"""
        )

    grid = []
    for tick_tenths in range(-1, 7):
        value = Decimal(tick_tenths) / Decimal(10)
        y = y_position(value)
        css_class = "zero" if tick_tenths == 0 else "grid"
        grid.append(
            f'''  <line class="{css_class}" x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_width}" y2="{y:.2f}" />
  <text class="tick" x="52" y="{y + 4:.2f}" text-anchor="end">{value:+.1f}</text>'''
        )

    mean = Decimal(str(bootstrap["mean_improvement"]))
    ci_low, ci_high = (Decimal(str(value)) for value in bootstrap["confidence_interval"])
    mean_y = y_position(mean)
    ci_axis_x = Decimal(62)
    ci_axis_width = Decimal(300)
    ci_axis_max = Decimal("0.25")

    def ci_x(value: Decimal) -> Decimal:
        return ci_axis_x + value / ci_axis_max * ci_axis_width

    values = {
        "ordered_points": points,
        "wins": data["wins"],
        "ties": data["ties"],
        "losses": data["losses"],
        "paired_bootstrap": bootstrap,
    }
    metadata = _stage_b_metadata(
        data,
        chart="experiment009-stage-b-paired-cqer-minus-rht",
        values=values,
    )
    description = (
        "In frozen task order, the paired CQER-32 minus right-RHT plus CQER-32 "
        f"aligned excess NLL difference is positive on {data['wins']} tasks, zero "
        f"on {data['ties']}, and negative on {data['losses']}. Positive values favor "
        f"right-RHT. The mean improvement is {mean:.6f} nats per token; the paired "
        f"10,000-sample bootstrap 95 percent interval is {ci_low:.6f} to "
        f"{ci_high:.6f}. This is authenticated development evidence, not confirmation."
    )
    return f"""<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600" role="img" aria-labelledby="stage-b-paired-title stage-b-paired-desc">
  <title id="stage-b-paired-title">Experiment 009 Stage-B paired CQER minus right-RHT excess NLL by frozen task order</title>
  <desc id="stage-b-paired-desc">{html.escape(description)}</desc>
  <metadata id="recurquant-provenance">{metadata}</metadata>
  <defs>
    <linearGradient id="stage-b-win-gradient" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0" stop-color="#288f91" />
      <stop offset="1" stop-color="#79e8e2" />
    </linearGradient>
  </defs>
  <style>
    svg {{ background: #071118; }}
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #eef8f8; }}
    .title {{ font-size: 23px; font-weight: 500; }}
    .subtitle, .note, .tick, .task, .ci-label {{ fill: #91adb2; }}
    .subtitle {{ font-size: 13px; letter-spacing: 0.5px; }}
    .summary {{ font-size: 13px; font-weight: 500; }}
    .note, .tick, .task, .ci-label {{ font-size: 10px; }}
    .grid {{ stroke: #16343d; stroke-width: 1; }}
    .zero {{ stroke: #8fb5bb; stroke-width: 1.3; }}
    .mean {{ stroke: #77ddd8; stroke-width: 1.3; stroke-dasharray: 5 4; }}
    .win {{ fill: url(#stage-b-win-gradient); }}
    .loss {{ fill: #9b82e8; }}
    .tie {{ fill: #81949a; }}
    .ci-axis {{ stroke: #2b5962; stroke-width: 2; }}
    .ci-range {{ stroke: #70dfda; stroke-width: 5; stroke-linecap: round; }}
    .ci-cap {{ stroke: #70dfda; stroke-width: 1; }}
    .ci-mean {{ fill: #eef8f8; stroke: #071118; stroke-width: 2; }}
    .divider {{ stroke: #20515b; stroke-width: 1; }}
  </style>
  <text class="title" x="28" y="34">Paired task result &#183; CQER-32 minus right-RHT</text>
  <text class="subtitle" x="28" y="58">POSITIVE VALUES FAVOR RIGHT-RHT &#183; FROZEN TASK ORDER</text>
  <line class="divider" x1="28" y1="78" x2="932" y2="78" />
  <text class="ci-label" x="62" y="99">PAIRED MEAN &#183; 95% BOOTSTRAP CI</text>
  <line class="ci-axis" x1="{ci_axis_x}" y1="120" x2="{ci_axis_x + ci_axis_width}" y2="120" />
  <line class="ci-range" x1="{ci_x(ci_low):.2f}" y1="120" x2="{ci_x(ci_high):.2f}" y2="120" />
  <line class="ci-cap" x1="{ci_x(ci_low):.2f}" y1="112" x2="{ci_x(ci_low):.2f}" y2="128" />
  <line class="ci-cap" x1="{ci_x(ci_high):.2f}" y1="112" x2="{ci_x(ci_high):.2f}" y2="128" />
  <circle class="ci-mean" cx="{ci_x(mean):.2f}" cy="120" r="5" />
  <text class="summary" x="380" y="125">{mean:+.4f} mean &#183; [{ci_low:.4f}, {ci_high:.4f}] 95% CI</text>
  <text class="summary" x="704" y="125">{data["wins"]} wins &#183; {data["ties"]} ties &#183; {data["losses"]} losses</text>
  <text class="ci-label" x="62" y="151">CQER &#8722; RIGHT-RHT ALIGNED EXCESS NLL (NATS / TOKEN)</text>
{chr(10).join(grid)}
{chr(10).join(bars)}
  <line class="mean" x1="{plot_x}" y1="{mean_y:.2f}" x2="{plot_x + plot_width}" y2="{mean_y:.2f}" />
  <text class="ci-label" x="{plot_x + plot_width - 4}" y="{mean_y - 6:.2f}" text-anchor="end">mean {mean:+.4f}</text>
{chr(10).join(labels)}
  <text class="note" x="28" y="550">Task IDs follow the authenticated identity order &#183; 10,000 paired task bootstraps &#183; seed 2339 &#183; two-sided equal-tailed 95% CI</text>
  <text class="note" x="28" y="576">32-task MBPP development evidence only; no confirmation, novelty, speed, deployment, or breakthrough claim.</text>
</svg>
"""


def _stage_b_graph_metadata(content: str, *, path: Path) -> dict[str, object]:
    _validate_svg(content, path)
    root = ET.fromstring(content)
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    if metadata is None or metadata.text is None:
        raise ValueError(f"{path.name} has no RecurQuant provenance metadata")
    try:
        record = json.loads(metadata.text, parse_constant=_reject_non_finite)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} has invalid provenance JSON") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path.name} provenance must be a JSON object")
    return record


def _validate_stage_b_graph_assets_from_manifest() -> None:
    """Validate tracked Stage-B SVG receipts when the raw release asset is absent."""

    try:
        manifest_raw = STAGE_B_MANIFEST.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(
            f"tracked Stage-B release manifest is missing: {STAGE_B_MANIFEST.relative_to(ROOT)}"
        ) from error
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"),
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tracked Stage-B release manifest is not strict UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("tracked Stage-B release manifest must be a JSON object")

    result = manifest.get("result")
    verification = manifest.get("verification")
    graph_assets = manifest.get("graph_assets")
    if (
        manifest.get("artifact_kind") != "recurquant_experiment009_stage_b_release_manifest"
        or manifest.get("schema_version") != 1
        or not isinstance(result, dict)
        or not isinstance(verification, dict)
        or not isinstance(graph_assets, dict)
    ):
        raise ValueError("tracked Stage-B release manifest schema drifted")

    if (
        result.get("artifact_kind") != STAGE_B_ARTIFACT_KIND
        or result.get("raw_filename") != STAGE_B_SOURCE.name
        or result.get("raw_sha256") != STAGE_B_ARTIFACT_SHA256
        or result.get("canonical_evidence_sha256") != STAGE_B_CANONICAL_SHA256
        or result.get("evaluation_commit") != STAGE_B_RESULT_COMMIT
        or result.get("verifier_fix_commit") != STAGE_B_VERIFIER_FIX_COMMIT
    ):
        raise ValueError("tracked Stage-B result provenance drifted")
    if (
        verification.get("strict_loader_passed") is not True
        or verification.get("canonical_round_trip") is not True
        or verification.get("integrity_passed") is not True
        or verification.get("advancement_passed") is not True
        or verification.get("advancement_check_count") != 8
        or verification.get("task_count") != 32
        or verification.get("bootstrap_samples") != 10_000
        or verification.get("bootstrap_seed") != 2339
    ):
        raise ValueError("tracked Stage-B verification receipt drifted")

    expected_graph_provenance = {
        "source_artifact_sha256": STAGE_B_ARTIFACT_SHA256,
        "canonical_evidence_sha256": STAGE_B_CANONICAL_SHA256,
        "original_result_commit": STAGE_B_RESULT_COMMIT,
        "verifier_fix_commit": STAGE_B_VERIFIER_FIX_COMMIT,
        "artifact_kind": STAGE_B_ARTIFACT_KIND,
        "integrity_passed": True,
        "advancement_gate_passed": True,
    }
    if {
        key: graph_assets.get(key) for key in expected_graph_provenance
    } != expected_graph_provenance:
        raise ValueError("tracked Stage-B graph provenance drifted")

    asset_receipts = graph_assets.get("assets")
    if not isinstance(asset_receipts, dict) or set(asset_receipts) != set(STAGE_B_GRAPH_ASSETS):
        raise ValueError("tracked Stage-B graph asset set drifted")

    expected_semantic_verification = {
        "passed": True,
        "integrity_passed": True,
        "advancement_passed": True,
        "task_count": 32,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2339,
        "advancement_check_count": 8,
        "artifact_sha256": STAGE_B_ARTIFACT_SHA256,
        "canonical_evidence_sha256": STAGE_B_CANONICAL_SHA256,
        "canonical_round_trip": True,
    }
    for relative_path, chart in STAGE_B_GRAPH_ASSETS.items():
        receipt = asset_receipts[relative_path]
        if not isinstance(receipt, dict) or set(receipt) != {
            "bytes",
            "sha256",
            "chart",
        }:
            raise ValueError(f"tracked graph receipt schema drifted for {relative_path}")
        if receipt.get("chart") != chart:
            raise ValueError(f"tracked graph chart identity drifted for {relative_path}")
        expected_bytes = receipt.get("bytes")
        expected_sha256 = receipt.get("sha256")
        if (
            type(expected_bytes) is not int
            or expected_bytes <= 0
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError(f"tracked graph byte receipt is invalid for {relative_path}")

        path = ROOT / Path(relative_path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise ValueError(f"tracked Stage-B graph is missing: {relative_path}") from error
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(
                f"tracked Stage-B graph bytes drifted for {relative_path}: "
                f"expected {expected_bytes} bytes sha256={expected_sha256}, "
                f"got {len(raw)} bytes sha256={actual_sha256}"
            )
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{relative_path} is not UTF-8 SVG") from error
        metadata = _stage_b_graph_metadata(content, path=path)
        expected_metadata = {
            "generated_by": "scripts/generate_readme_assets.py",
            "source_document": ("artifacts/experiment009-rht-cqer-stage-b-result-cdc603b.json"),
            "chart": chart,
            **expected_graph_provenance,
        }
        if {key: metadata.get(key) for key in expected_metadata} != expected_metadata:
            raise ValueError(f"embedded graph provenance drifted for {relative_path}")
        if metadata.get("semantic_verification") != expected_semantic_verification:
            raise ValueError(f"embedded semantic verification drifted for {relative_path}")
        if "values" not in metadata or "claim_boundary" not in metadata:
            raise ValueError(f"embedded graph evidence is incomplete for {relative_path}")


def _legacy_outputs() -> dict[Path, str]:
    data = _source_data()
    development_data = _development_data()
    confirmation_data = _confirmation_data()
    return {
        ASSETS / "recurrent-state-storage.svg": _storage_svg(data),
        ASSETS / "diagnostic-tail-kl.svg": _diagnostic_svg(data),
        ASSETS / "mbpp-development-fidelity.svg": _development_svg(development_data),
        ASSETS / "mbpp-confirmation-fidelity.svg": _confirmation_svg(confirmation_data),
    }


def _outputs() -> dict[Path, str]:
    outputs = _legacy_outputs()
    stage_b_data = _stage_b_data()
    outputs.update(
        {
            ASSETS / "experiment009-stage-b-overview.svg": _stage_b_overview_svg(stage_b_data),
            ASSETS / "experiment009-stage-b-paired.svg": _stage_b_paired_svg(stage_b_data),
        }
    )
    return outputs


def _validate_svg(content: str, path: Path) -> None:
    try:
        ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Generated invalid SVG for {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated assets differ from the files on disk",
    )
    args = parser.parse_args()

    stage_b_source_available = STAGE_B_SOURCE.is_file()
    if not args.check and not stage_b_source_available:
        try:
            _stage_b_data()
        except FileNotFoundError as error:
            print(str(error), file=sys.stderr)
            return 2
        raise AssertionError("missing Stage-B source check did not fail")

    outputs = _outputs() if stage_b_source_available else _legacy_outputs()
    stale = []
    for path, content in outputs.items():
        _validate_svg(content, path)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(ROOT))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        print(f"wrote {path.relative_to(ROOT)} sha256={digest}")

    manifest_error = None
    if args.check and not stage_b_source_available:
        try:
            _validate_stage_b_graph_assets_from_manifest()
        except ValueError as error:
            manifest_error = str(error)

    if stale:
        for path in stale:
            print(f"stale: {path}", file=sys.stderr)
    if manifest_error is not None:
        print(f"invalid Stage-B graph receipt: {manifest_error}", file=sys.stderr)
    if stale or manifest_error is not None:
        return 1
    if args.check:
        if stage_b_source_available:
            print("README assets are up to date and valid XML.")
        else:
            print(
                "README assets are up to date; Stage-B SVG bytes and provenance "
                "match the tracked release manifest."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
