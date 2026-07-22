"""Generate deterministic README charts from committed RecurQuant evidence.

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
import re
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "EXPERIMENT_002_SCALE_CORRECTION.md"
DEVELOPMENT_SOURCE = ROOT / "evidence" / "mbpp-v02-development.json"
ASSETS = ROOT / "assets"


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
        "retrieval_sweep": _evidence_hash(
            text, "Corrected retrieval sweep evidence hash"
        ),
        "code_sweep": _evidence_hash(text, "Corrected code sweep evidence hash"),
        "multilingual_qdq": _evidence_hash(
            text, "Corrected multilingual QDQ replay evidence hash"
        ),
        "multilingual_packed": _evidence_hash(
            text, "Packed multilingual replay evidence hash"
        ),
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
        "whole_model_memory_claim_allowed": claim_scope[
            "whole_model_memory_claim_allowed"
        ],
        "task_count": schedule["row_count"],
        "token_count": primary["token_weighted"]["token_count"],
        "uniform_macro_delta_nll": uniform["task_macro"]["delta_nll"],
        "primary_macro_delta_nll": primary["task_macro"]["delta_nll"],
        "relative_reduction": primary_contrast["relative_reduction"],
        "primary_vs_uniform": primary_contrast["paired_bootstrap"],
        "primary_vs_mean_random_equal_byte": random_contrast["paired_bootstrap"],
        "primary_resident_bytes": primary["storage"]["resident_bytes"],
        "uniform_resident_bytes": uniform["storage"]["resident_bytes"],
        "full_precision_equivalent_bytes": primary["storage"][
            "full_precision_equivalent_bytes"
        ],
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
    return f'''<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
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
'''


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
        f"{row['uniform']} versus {row['mixed']} for {str(row['label']).lower()}"
        for row in rows
    )
    diagnostic_desc = (
        "On short diagnostic traces, uniform INT4 versus layer zero INT8 and the "
        f"remaining layers INT4 produced CVaR95 KL of {comparisons}. Lower is better. "
        "These diagnostics are not a public benchmark or breakthrough evidence."
    )
    return f'''<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
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
'''


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
            "whole_model_memory_claim_allowed": data[
                "whole_model_memory_claim_allowed"
            ],
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
    return f'''<!-- Generated by scripts/generate_readme_assets.py; do not edit by hand. -->
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
'''


def _outputs() -> dict[Path, str]:
    data = _source_data()
    development_data = _development_data()
    return {
        ASSETS / "recurrent-state-storage.svg": _storage_svg(data),
        ASSETS / "diagnostic-tail-kl.svg": _diagnostic_svg(data),
        ASSETS / "mbpp-development-fidelity.svg": _development_svg(development_data),
    }


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

    outputs = _outputs()
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

    if stale:
        for path in stale:
            print(f"stale: {path}", file=sys.stderr)
        return 1
    if args.check:
        print("README assets are up to date and valid XML.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
