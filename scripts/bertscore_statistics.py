from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "bertscore_statistics"
EXPECTED_SCHEMA_VERSION = "caption_field_scores.v1"
METRIC_KEYS = (
    ("precision", "bertscore_precision"),
    ("recall", "bertscore_recall"),
    ("f1", "bertscore_f1"),
)
METRIC_COLORS = {
    "precision": "#2b6db0",
    "recall": "#db7842",
    "f1": "#499868",
}
COVERAGE_STATES = (
    "both_present",
    "both_absent",
    "can_missed",
    "can_extra",
    "unresolved",
)
COVERAGE_COLORS = {
    "both_present": "#499868",
    "both_absent": "#a9adb4",
    "can_missed": "#d9534f",
    "can_extra": "#e6a23c",
    "unresolved": "#7b61a8",
}

BERT_FIELD_SPECS = (
    ("image_observation.body_site", None),
    ("image_observation.visual_summary", None),
    ("image_observation.measurement_tool.description", None),
    ("image_observation.measurement_tool.estimated_size", None),
    ("wound_features.shape_pattern", None),
    ("wound_features.edges_margins", None),
    ("wound_features.wound_bed", None),
    ("wound_features.color", None),
    ("wound_features.texture", None),
    ("wound_features.fluid_exudate_bleeding", None),
    ("wound_features.periwound_skin", None),
    ("category_specific_check.observed_supporting_features", "ref_best"),
    ("category_specific_check.observed_supporting_features", "can_best"),
    ("category_specific_check.expected_but_not_observed", "ref_best"),
    ("category_specific_check.expected_but_not_observed", "can_best"),
)

COVERAGE_FIELD_SPECS = (
    "image_observation.measurement_tool.visible",
    "wound_features.shape_pattern",
    "wound_features.edges_margins",
    "wound_features.wound_bed",
    "wound_features.color",
    "wound_features.texture",
    "wound_features.fluid_exudate_bleeding",
    "wound_features.periwound_skin",
    "wound_features.foreign_material_debris",
    "wound_features.necrosis_eschar",
    "category_specific_check.observed_supporting_features",
    "category_specific_check.expected_but_not_observed",
    "category_specific_check.differential_visual_conflicts",
    "uncertainty.low_confidence_regions",
    "uncertainty.cannot_determine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one Markdown statistics report from one caption_field_scores.v1 "
            "ref/can JSON database. BERTScore is not recomputed or transformed."
        )
    )
    parser.add_argument("score_json", help="Field-level BERTScore JSON database.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Report output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Score JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read score JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("Score JSON must contain a top-level object.")
    if value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported schema_version: {value.get('schema_version')!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION!r}."
        )
    if not isinstance(value.get("blocks"), dict) or not value["blocks"]:
        raise SystemExit("Score JSON contains no filename blocks.")
    return value


def nested_value(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def category_for(block: dict[str, Any]) -> str:
    metadata = block.get("metadata", {})
    category = metadata.get("category") if isinstance(metadata, dict) else None
    return str(category or "unknown")


def grouped_blocks(database: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for filename, block in sorted(database["blocks"].items(), key=lambda item: item[0].casefold()):
        grouped[category_for(block)].append((filename, block))
    return dict(sorted(grouped.items(), key=lambda item: item[0].casefold()))


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def mean_metric_records(records: Iterable[dict[str, Any]]) -> dict[str, float] | None:
    records = list(records)
    if not records:
        return None
    means: dict[str, float] = {}
    for short_name, json_key in METRIC_KEYS:
        values = [finite_float(record.get(json_key)) for record in records]
        if any(value is None for value in values):
            return None
        means[short_name] = statistics.fmean(value for value in values if value is not None)
    return means


def image_bert_value(block: dict[str, Any], field_path: str, direction: str | None) -> dict[str, float] | None:
    result = nested_value(block.get("fields", {}), field_path)
    if not isinstance(result, dict) or result.get("status") != "scored":
        return None
    if direction is None:
        return mean_metric_records([result])
    source_key = "ref_best_scores" if direction == "ref_best" else "can_best_scores"
    source = result.get(source_key)
    if not isinstance(source, list):
        return None
    return mean_metric_records(record for record in source if isinstance(record, dict))


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else None,
    }


def aggregate_bertscore(
    grouped: dict[str, list[tuple[str, dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for category, items in grouped.items():
        rows = []
        for field_path, direction in BERT_FIELD_SPECS:
            image_values = [
                value
                for _, block in items
                if (value := image_bert_value(block, field_path, direction)) is not None
            ]
            row: dict[str, Any] = {
                "field_path": field_path,
                "direction": direction or "scalar",
            }
            for metric, _ in METRIC_KEYS:
                row[metric] = summarize_values([value[metric] for value in image_values])
            rows.append(row)
        output[category] = rows
    return output


def global_bertscore_ylim(aggregates: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    found = False
    for rows in aggregates.values():
        for row in rows:
            for metric, _ in METRIC_KEYS:
                summary = row[metric]
                if summary["mean"] is None:
                    continue
                found = True
                spread = summary["std"] or 0.0
                lower = min(lower, summary["mean"] - spread)
                upper = max(upper, summary["mean"] + spread)
    if not found:
        return (-0.1, 0.1)
    span = upper - lower
    padding = max(0.05, span * 0.08)
    return lower - padding, upper + padding


def coverage_outcome(result: Any) -> str:
    if not isinstance(result, dict) or result.get("status") == "unresolved":
        return "unresolved"
    metric = result.get("metric")
    status = result.get("status")
    quadrant = result.get("quadrant")
    if metric in {"dynamic_bertscore", "enum_quadrant"}:
        return {
            "both_observed": "both_present",
            "both_not_observed": "both_absent",
            "ref_only_observed": "can_missed",
            "can_only_observed": "can_extra",
        }.get(quadrant, "unresolved")
    if metric == "item_level_bertscore":
        return {
            "scored": "both_present",
            "both_empty": "both_absent",
            "can_empty": "can_missed",
            "ref_empty": "can_extra",
        }.get(status, "unresolved")
    if metric == "empty_nonempty":
        return {
            "both_nonempty": "both_present",
            "both_empty": "both_absent",
            "can_empty": "can_missed",
            "ref_empty": "can_extra",
        }.get(status, "unresolved")
    if metric == "binary":
        return {
            "both_true": "both_present",
            "both_false": "both_absent",
            "ref_only_true": "can_missed",
            "can_only_true": "can_extra",
        }.get(status, "unresolved")
    return "unresolved"


def aggregate_coverage(
    grouped: dict[str, list[tuple[str, dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for category, items in grouped.items():
        rows = []
        total = len(items)
        for field_path in COVERAGE_FIELD_SPECS:
            counts = Counter(
                coverage_outcome(nested_value(block.get("fields", {}), field_path))
                for _, block in items
            )
            if sum(counts.values()) != total:
                raise RuntimeError(f"Coverage count does not equal category total: {category}/{field_path}")
            rows.append(
                {
                    "field_path": field_path,
                    "total": total,
                    "counts": {state: counts[state] for state in COVERAGE_STATES},
                    "rates": {
                        state: (counts[state] / total if total else 0.0)
                        for state in COVERAGE_STATES
                    },
                }
            )
        output[category] = rows
    return output


def aggregate_wound_presence(
    grouped: dict[str, list[tuple[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows = []
    for category, items in grouped.items():
        counts = Counter()
        for _, block in items:
            result = nested_value(block.get("fields", {}), "image_observation.wound_presence")
            status = result.get("status") if isinstance(result, dict) else "unresolved"
            counts[status if status in {"matched", "mismatched", "unresolved"} else "unresolved"] += 1
        comparable = counts["matched"] + counts["mismatched"]
        rows.append(
            {
                "category": category,
                "total": len(items),
                "matched": counts["matched"],
                "mismatched": counts["mismatched"],
                "unresolved": counts["unresolved"],
                "exact_match_rate": counts["matched"] / comparable if comparable else None,
            }
        )
    return rows


def append_field_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def normalized_anomaly_records(
    blocks: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename, block in sorted(blocks.items(), key=lambda item: item[0].casefold()):
        for anomaly_index, anomaly in enumerate(block.get("anomalies", [])):
            if not isinstance(anomaly, dict) or anomaly.get("side") != side:
                continue
            anomaly_type = str(anomaly.get("type", "unknown"))
            original_path = str(anomaly.get("field_path", "unknown"))
            message = str(anomaly.get("message", ""))
            paths = [original_path]
            resolution = "original"
            if anomaly_type == "schema_required_error":
                matches = re.findall(r"['\"]([^'\"]+)['\"]\s+is a required property", message)
                if matches:
                    paths = [append_field_path(original_path, match) for match in matches]
                    resolution = "derived_required_property"
                else:
                    resolution = "unresolved_parent_path"
            elif anomaly_type == "schema_additionalProperties_error":
                matches = re.findall(r"['\"]([^'\"]+)['\"]", message)
                if matches:
                    paths = [append_field_path(original_path, match) for match in matches]
                    resolution = "derived_additional_property"
                else:
                    resolution = "unresolved_parent_path"
            for effective_path in paths:
                records.append(
                    {
                        "event_id": f"{filename}:{anomaly_index}",
                        "filename": filename,
                        "side": side,
                        "severity": "schema_error" if anomaly_type.startswith("schema_") else "contract_warning",
                        "type": anomaly_type,
                        "original_path": original_path,
                        "field_path": effective_path,
                        "path_resolution": resolution,
                        "message": message,
                        "raw_value": anomaly.get("raw_value"),
                    }
                )
    return records


def compliance_summary(blocks: dict[str, Any], can_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(blocks)
    strict_failed = {
        record["filename"] for record in can_records if record["severity"] == "schema_error"
    }
    extended_failed = {record["filename"] for record in can_records}
    return [
        {
            "level": "strict_schema",
            "total": total,
            "failed": len(strict_failed),
            "passed": total - len(strict_failed),
            "compliance_rate": (total - len(strict_failed)) / total if total else None,
        },
        {
            "level": "extended_contract",
            "total": total,
            "failed": len(extended_failed),
            "passed": total - len(extended_failed),
            "compliance_rate": (total - len(extended_failed)) / total if total else None,
        },
    ]


def field_error_rows(
    blocks: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    total = len(blocks)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["severity"], record["field_path"])].append(record)
    rows = []
    for (severity, field_path), items in grouped.items():
        affected = {item["filename"] for item in items}
        rows.append(
            {
                "severity": severity,
                "field_path": field_path,
                "affected_images": len(affected),
                "total_images": total,
                "error_rate": len(affected) / total if total else None,
                "anomaly_events": len({item["event_id"] for item in items}),
            }
        )
    return sorted(rows, key=lambda row: (-row["error_rate"], row["field_path"], row["severity"]))


def error_type_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_totals = Counter(record["field_path"] for record in records)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["field_path"], record["type"], record["severity"])].append(record)
    rows = []
    for (field_path, error_type, severity), items in grouped.items():
        rows.append(
            {
                "field_path": field_path,
                "severity": severity,
                "error_type": error_type,
                "affected_images": len({item["filename"] for item in items}),
                "event_count": len({item["event_id"] for item in items}),
                "within_field_rate": len(items) / field_totals[field_path],
            }
        )
    return sorted(rows, key=lambda row: (row["field_path"], -row["event_count"], row["error_type"]))


def report_stem(source: Path) -> str:
    suffix = "_caption_field_scores"
    stem = source.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}_statistics"


def report_paths(source: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = report_stem(source)
    return output_dir / f"{stem}.md", output_dir / f"{stem}_assets"


def chart_label(field_path: str, direction: str | None = None) -> str:
    label = field_path.split(".")[-1]
    if direction and direction != "scalar":
        return f"{label}\n{direction}"
    if field_path.startswith("image_observation.measurement_tool"):
        return f"measurement.{label}"
    return label


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_.-") or "unknown"


def bert_bar_label(summary: dict[str, Any]) -> str:
    mean = summary.get("mean")
    if mean is None:
        return "N/A"
    std = summary.get("std")
    bold_mean = rf"$\mathbf{{{mean:.2f}}}$"
    return f"{bold_mean} ± {std:.2f}" if std is not None else f"{bold_mean} ± N/A"


def coverage_segment_label(row: dict[str, Any], state: str) -> str:
    count = row["counts"][state]
    if count == 0:
        return ""
    return f"{count} ({row['rates'][state] * 100:.1f}%)"


def load_pyplot() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wound_caption_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def atomic_save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.png"
    try:
        fig.savefig(temporary, dpi=160, format="png", bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_bertscore_category(
    category: str,
    rows: list[dict[str, Any]],
    ylim: tuple[float, float],
    path: Path,
) -> None:
    plt = load_pyplot()
    labels = [chart_label(row["field_path"], row["direction"]) for row in rows]
    width = 0.25
    centers = list(range(len(rows)))
    fig_width = max(22.0, len(rows) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 10.0))
    offsets = {"precision": -width, "recall": 0.0, "f1": width}
    y_span = ylim[1] - ylim[0]
    for metric, _ in METRIC_KEYS:
        legend_used = False
        for center, row in zip(centers, rows):
            summary = row[metric]
            if summary["mean"] is None:
                continue
            kwargs: dict[str, Any] = {
                "color": METRIC_COLORS[metric],
                "width": width,
                "label": metric.upper() if not legend_used else None,
                "edgecolor": "#333333",
                "linewidth": 0.5,
            }
            if summary["n"] == 1:
                kwargs["hatch"] = "///"
            if summary["std"] is not None:
                kwargs.update({"yerr": summary["std"], "capsize": 3, "error_kw": {"elinewidth": 1}})
            bar_x = center + offsets[metric]
            ax.bar(bar_x, summary["mean"], **kwargs)
            label_offset = y_span * 0.012
            is_positive = summary["mean"] >= 0
            ax.text(
                bar_x,
                summary["mean"] + (label_offset if is_positive else -label_offset),
                bert_bar_label(summary),
                ha="center",
                va="bottom" if is_positive else "top",
                rotation=90,
                fontsize=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
                zorder=5,
            )
            baseline_offset = y_span * 0.012
            ax.text(
                bar_x,
                -baseline_offset if is_positive else baseline_offset,
                f"n={summary['n']}",
                ha="center",
                va="top" if is_positive else "bottom",
                rotation=90,
                fontsize=10,
                color="black",
                fontweight="bold",
                zorder=6,
            )
            legend_used = True

    na_y = 0.0 + y_span * 0.012
    for center, row in zip(centers, rows):
        n = row["f1"]["n"]
        if n == 0:
            ax.text(center, na_y, "N/A", ha="center", va="bottom", fontsize=12, rotation=90)

    ax.axhline(0.0, color="#202020", linewidth=1.2, zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Mean official baseline-rescaled BERTScore", fontsize=15)
    ax.set_title(f"BERTScore by Field — {category}", fontsize=20, pad=14)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper right", fontsize=13)
    fig.text(
        0.01,
        0.01,
        "Bars start at the official baseline (0). Error bars are ±1 sample SD; hatched bars have n=1.",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    atomic_save_figure(fig, path)
    plt.close(fig)


def plot_coverage_category(category: str, rows: list[dict[str, Any]], path: Path) -> None:
    plt = load_pyplot()
    centers = list(range(len(rows)))
    labels = [chart_label(row["field_path"]) for row in rows]
    fig_width = max(26.0, len(rows) * 1.75)
    fig, ax = plt.subplots(figsize=(fig_width, 10.0))
    bottoms = [0.0] * len(rows)
    for state in COVERAGE_STATES:
        values = [row["rates"][state] * 100 for row in rows]
        ax.bar(
            centers,
            values,
            bottom=bottoms,
            color=COVERAGE_COLORS[state],
            width=0.9,
            label=state,
            edgecolor="white",
            linewidth=0.4,
        )
        text_color = "#202020" if state in {"both_absent", "can_extra"} else "white"
        for center, bottom, value, row in zip(centers, bottoms, values, rows):
            label = coverage_segment_label(row, state)
            if not label:
                continue
            ax.text(
                center,
                bottom + value / 2,
                label,
                ha="center",
                va="center",
                fontsize=12,
                color=text_color,
                fontweight="bold",
                clip_on=False,
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylim(0, 100)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Percentage of images", fontsize=15)
    total = rows[0]["total"] if rows else 0
    ax.set_title(
        f"Coverage State Distribution — {category} (Total images: {total})",
        fontsize=20,
        pad=72,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=13)
    fig.tight_layout()
    atomic_save_figure(fig, path)
    plt.close(fig)


def markdown_escape(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return text


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(markdown_escape(value) for value in row) + " |" for row in rows)
    return lines


def fmt_float(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def fmt_metric(summary: dict[str, Any]) -> str:
    if summary["mean"] is None:
        return "N/A"
    if summary["std"] is None:
        return fmt_float(summary["mean"])
    return f"{summary['mean']:.4f} ± {summary['std']:.4f}"


def raw_preview(value: Any, limit: int = 100) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def relative_image_path(report_path: Path, image_path: Path) -> str:
    return os.path.relpath(image_path, report_path.parent).replace("\\", "/")


def report_metadata_rows(database: dict[str, Any], source: Path) -> list[list[Any]]:
    report = database.get("report_metadata", {})
    bertscore = report.get("bertscore", {}) if isinstance(report.get("bertscore"), dict) else {}
    first_block = next(iter(database["blocks"].values()))
    metadata = first_block.get("metadata", {}) if isinstance(first_block, dict) else {}
    return [
        ["Source score database", str(source.resolve())],
        ["Reference root", report.get("ref_root_name")],
        ["Candidate root", report.get("can_root_name")],
        ["Reference model", metadata.get("ref_model")],
        ["Candidate model", metadata.get("can_model")],
        ["Matched blocks", len(database["blocks"])],
        ["BERTScore model", bertscore.get("model")],
        ["BERTScore hash", bertscore.get("bertscore_hash")],
        ["Score representation", bertscore.get("score_representation")],
    ]


def build_markdown(
    database: dict[str, Any],
    source: Path,
    report_path: Path,
    asset_paths: dict[str, dict[str, Path]],
    grouped: dict[str, list[tuple[str, dict[str, Any]]]],
    bertscore: dict[str, list[dict[str, Any]]],
    coverage: dict[str, list[dict[str, Any]]],
    wound_presence: list[dict[str, Any]],
    can_records: list[dict[str, Any]],
    ref_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Field-Level BERTScore Statistics Report",
        "",
        "## 1. Report Metadata",
        "",
        *markdown_table(["Item", "Value"], report_metadata_rows(database, source)),
        "",
        "### Category Sample Counts",
        "",
        *markdown_table(
            ["Category", "Images"],
            ([category, len(items)] for category, items in grouped.items()),
        ),
        "",
        "> Scores are official baseline-rescaled BERTScore values. They may be below 0 or slightly above 1. "
        "Zero is the official baseline; this report does not clip, normalize, or recompute scores.",
        "",
        "## 2. Semantic Quality",
        "",
    ]
    for category in grouped:
        image = asset_paths[category]["bertscore"]
        lines.extend(
            [
                f"### {category} (Total images: {len(grouped[category])})",
                "",
                f"![{category} BERTScore]({relative_image_path(report_path, image)})",
                "",
            ]
        )

    lines.extend(["## 3. Coverage", ""])
    for category in grouped:
        image = asset_paths[category]["coverage"]
        lines.extend(
            [
                f"### {category} (Total images: {len(grouped[category])})",
                "",
                f"![{category} coverage]({relative_image_path(report_path, image)})",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Wound Presence Categorical Accuracy",
            "",
            *markdown_table(
                ["Category", "Total", "Matched", "Mismatched", "Unresolved", "Exact-match rate"],
                (
                    [
                        row["category"],
                        row["total"],
                        row["matched"],
                        row["mismatched"],
                        row["unresolved"],
                        fmt_rate(row["exact_match_rate"]),
                    ]
                    for row in wound_presence
                ),
            ),
            "",
            "Exact-match rate excludes unresolved images from its denominator.",
            "",
            "## 5. Candidate Schema and Contract Compliance",
            "",
            "### Severity Definitions",
            "",
            *markdown_table(
                ["Severity", "Meaning", "Compliance impact"],
                [
                    ["schema_error", "The candidate violates the declared JSON schema.", "Fails strict and extended compliance."],
                    ["contract_warning", "The schema is valid, but a prompt, enum-state, assignment, or cross-field contract is inconsistent.", "Fails extended compliance only."],
                ],
            ),
            "",
        ]
    )
    compliance = compliance_summary(database["blocks"], can_records)
    lines.extend(
        [
            *markdown_table(
                ["Level", "Passed", "Failed", "Total", "Compliance rate"],
                (
                    [row["level"], row["passed"], row["failed"], row["total"], fmt_rate(row["compliance_rate"])]
                    for row in compliance
                ),
            ),
            "",
            "`strict_schema` includes only `schema_*` errors. `extended_contract` includes schema errors and prompt/data consistency warnings.",
            "",
            "### Field Error Rates",
            "",
            *markdown_table(
                ["Severity", "Field", "Affected images", "Total", "Error rate", "Anomaly events"],
                (
                    [
                        row["severity"],
                        row["field_path"],
                        row["affected_images"],
                        row["total_images"],
                        fmt_rate(row["error_rate"]),
                        row["anomaly_events"],
                    ]
                    for row in field_error_rows(database["blocks"], can_records)
                ),
            ),
            "",
            "### Error Type Distribution",
            "",
            *markdown_table(
                ["Field", "Severity", "Error type", "Affected images", "Events", "Within-field share"],
                (
                    [
                        row["field_path"],
                        row["severity"],
                        row["error_type"],
                        row["affected_images"],
                        row["event_count"],
                        fmt_rate(row["within_field_rate"]),
                    ]
                    for row in error_type_rows(can_records)
                ),
            ),
            "",
            "### Candidate Anomaly Evidence",
            "",
        ]
    )
    if can_records:
        lines.extend(
            markdown_table(
                ["File", "Severity", "Field", "Type", "Path resolution", "Message", "Raw value"],
                (
                    [
                        row["filename"],
                        row["severity"],
                        row["field_path"],
                        row["type"],
                        row["path_resolution"],
                        row["message"],
                        raw_preview(row["raw_value"]),
                    ]
                    for row in can_records
                ),
            )
        )
    else:
        lines.append("No candidate anomalies.")

    lines.extend(["", "## 6. Reference Quality Warnings", ""])
    if ref_records:
        lines.extend(
            markdown_table(
                ["File", "Field", "Type", "Message", "Raw value"],
                (
                    [
                        row["filename"],
                        row["field_path"],
                        row["type"],
                        row["message"],
                        raw_preview(row["raw_value"]),
                    ]
                    for row in ref_records
                ),
            )
        )
    else:
        lines.append("No reference warnings.")
    lines.extend(["", f"Generated at `{now_iso()}`.", ""])
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def generate_report(source: Path, output_dir: Path) -> tuple[Path, Path]:
    database = read_database(source)
    grouped = grouped_blocks(database)
    bertscore = aggregate_bertscore(grouped)
    coverage = aggregate_coverage(grouped)
    wound_presence = aggregate_wound_presence(grouped)
    can_records = normalized_anomaly_records(database["blocks"], "can")
    ref_records = normalized_anomaly_records(database["blocks"], "ref")
    ylim = global_bertscore_ylim(bertscore)
    report_path, assets_dir = report_paths(source, output_dir)
    asset_paths: dict[str, dict[str, Path]] = {}
    for category in grouped:
        safe_category = safe_filename(category)
        bert_path = assets_dir / f"bertscore_{safe_category}.png"
        coverage_path = assets_dir / f"coverage_{safe_category}.png"
        plot_bertscore_category(category, bertscore[category], ylim, bert_path)
        plot_coverage_category(category, coverage[category], coverage_path)
        asset_paths[category] = {"bertscore": bert_path, "coverage": coverage_path}
    markdown = build_markdown(
        database,
        source,
        report_path,
        asset_paths,
        grouped,
        bertscore,
        coverage,
        wound_presence,
        can_records,
        ref_records,
    )
    atomic_write_text(report_path, markdown)
    return report_path, assets_dir


def main() -> None:
    args = parse_args()
    report_path, assets_dir = generate_report(Path(args.score_json), Path(args.output_dir))
    print(f"Saved {report_path}")
    print(f"Saved {assets_dir}")


if __name__ == "__main__":
    main()
