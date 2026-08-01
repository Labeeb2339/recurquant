"""Generate deterministic README charts from the committed correction record.

The script deliberately parses the evidence tables instead of duplicating their
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
            "17 layers INT4 + FP16 scales",
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


def _metadata(payload: dict[str, object]) -> str:
    record = {
        "generated_by": "scripts/generate_readme_assets.py",
        "source_document": "research/EXPERIMENT_002_SCALE_CORRECTION.md",
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
    <text class="muted" x="232" y="{y + 32}" text-anchor="end">{html.escape(str(row["detail"]))} - {html.escape(ratio_label)}</text>
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
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 365" role="img" aria-labelledby="storage-title storage-desc">
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
  <text class="subtitle" x="28" y="54">Qwen3.5-0.8B-Base - batch 1 - resident bytes (millions) - lower is better</text>
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
    <text class="reduction" x="{center}" y="399" text-anchor="middle">- {html.escape(str(row["reduction"]))} mixed vs uniform</text>
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
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 465" role="img" aria-labelledby="kl-title kl-desc">
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
  <text class="subtitle" x="28" y="54">CVaR95 token KL - stored FP16 scales - lower is better</text>
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


def _outputs() -> dict[Path, str]:
    data = _source_data()
    return {
        ASSETS / "recurrent-state-storage.svg": _storage_svg(data),
        ASSETS / "diagnostic-tail-kl.svg": _diagnostic_svg(data),
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
