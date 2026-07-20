import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CAPTION_FIELDS = [
    "caption.image_observation.visual_summary",
    "caption.wound_features.shape_pattern",
    "caption.wound_features.edges_margins",
    "caption.wound_features.wound_bed",
    "caption.wound_features.color",
    "caption.wound_features.texture",
    "caption.wound_features.fluid_exudate_bleeding",
    "caption.wound_features.periwound_skin",
    "caption.category_specific_check.observed_supporting_features",
]
COMPARISONS = {
    "AC": {
        "group": "caption_quality",
        "name": "caption_source_score",
        "reference_key": "a_reference_english",
        "candidate_key": "c_candidate_english",
        "meaning": "local English caption vs SaaS/Gemini English reference",
    },
    "BD": {
        "group": "human_review_calibration",
        "name": "translated_review_score",
        "reference_key": "b_translated_reference",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs translated reference",
    },
}
DIAGNOSTIC_COMPARISONS = {
    "AB": {
        "group": "translation_faithfulness",
        "name": "reference_translation_faithfulness",
        "reference_key": "a_reference_english",
        "candidate_key": "b_translated_reference",
        "meaning": "translated reference vs original English reference",
    },
    "CD": {
        "group": "translation_faithfulness",
        "name": "candidate_translation_faithfulness",
        "reference_key": "c_candidate_english",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs original English candidate",
    },
    "AD": {
        "group": "human_review_calibration",
        "name": "crosslingual_candidate_to_reference",
        "reference_key": "a_reference_english",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs English reference",
    },
}
SCOPES = ("visual_summary_only", "full_caption_fields")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute grouped BERTScore comparisons from existing run outputs."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--mode", default="simple")
    parser.add_argument("--score-type", choices=["grouped"], default="grouped")
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="A source: reference English JSON directory.",
    )
    parser.add_argument(
        "--reference-translation-dir",
        required=True,
        help="B source: translated reference JSON directory.",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="C source: candidate English JSON directory.",
    )
    parser.add_argument(
        "--candidate-translation-dir",
        required=True,
        help="D source: translated candidate JSON directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/bertscore_reports",
        help="Independent output folder. Do not put BERTScore reports inside a single run folder.",
    )
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--include-diagnostic", action="store_true")
    parser.add_argument(
        "--scopes",
        default="visual_summary_only,full_caption_fields",
        help="Comma-separated scoring scopes: visual_summary_only, full_caption_fields.",
    )
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--model-type", default="bert-base-multilingual-cased")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.selected_scopes = parse_scopes(args.scopes)
    return args


def parse_scopes(value: str) -> list[str]:
    scopes = [scope.strip() for scope in value.split(",") if scope.strip()]
    invalid = [scope for scope in scopes if scope not in SCOPES]
    if invalid:
        raise SystemExit(
            "Invalid --scopes value: "
            + ", ".join(invalid)
            + f". Valid scopes: {', '.join(SCOPES)}"
        )
    if not scopes:
        raise SystemExit("--scopes must include at least one scope.")
    return scopes


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def caption_text(bundle: dict[str, Any]) -> str:
    caption = bundle.get("caption", {})
    image_observation = caption.get("image_observation", {})
    wound_features = caption.get("wound_features", {})
    category_check = caption.get("category_specific_check", {})
    parts = [
        image_observation.get("visual_summary", ""),
        wound_features.get("shape_pattern", ""),
        wound_features.get("edges_margins", ""),
        wound_features.get("wound_bed", ""),
        wound_features.get("color", ""),
        wound_features.get("texture", ""),
        wound_features.get("fluid_exudate_bleeding", ""),
        wound_features.get("periwound_skin", ""),
        " ".join(category_check.get("observed_supporting_features", [])),
    ]
    return " ".join(str(part) for part in parts if part).strip()


def visual_summary_text(bundle: dict[str, Any]) -> str:
    return str(
        bundle.get("caption", {})
        .get("image_observation", {})
        .get("visual_summary", "")
        or ""
    ).strip()


def text_for_scope(bundle: dict[str, Any], scope: str) -> str:
    if scope == "visual_summary_only":
        return visual_summary_text(bundle)
    if scope == "full_caption_fields":
        return caption_text(bundle)
    raise ValueError(f"Unknown BERTScore scope: {scope}")


def category_for(bundle: dict[str, Any], fallback_name: str) -> str:
    metadata = bundle.get("metadata", {})
    caption = bundle.get("caption", {})
    category_check = caption.get("category_specific_check", {})
    return (
        metadata.get("target_category")
        or metadata.get("raw_category")
        or category_check.get("target_category")
        or fallback_name.split(" (", 1)[0]
        or "unknown"
    )


def load_bundles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    bundles: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        try:
            bundles[json_path.stem] = read_json(json_path)
        except json.JSONDecodeError:
            continue
    return bundles


def json_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json_path.stem for json_path in path.glob("*.json")}


def require_complete_grouped_inputs(data_map: dict[str, dict[str, Any]]) -> None:
    missing = [
        f"{key}: {item['path']} (exists={item['exists']}, json_files={item['file_count']})"
        for key, item in data_map.items()
        if item["file_count"] == 0
    ]
    if not missing:
        return
    detail = "\n".join(f"- {item}" for item in missing)
    raise SystemExit(
        "Grouped BERTScore needs all four data sources (a/b/c/d) to score every group.\n"
        "Missing or empty sources:\n"
        f"{detail}\n\n"
        "Fix by passing explicit A/B/C/D directories. This script intentionally does "
        "not auto-search other run folders because that can mix unrelated experiments."
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_cosine_score(value: float) -> float:
    return min(1.0, max(0.0, (value + 1.0) / 2.0))


def load_bert_score():
    try:
        from bert_score import score as bertscore_score
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: bert_score. Install it with `pip install bert-score` "
            "or add `bert-score` to requirements.txt and install project dependencies."
        ) from exc
    return bertscore_score


def compute_bert_scores(
    candidates: list[str],
    references: list[str],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[float]]:
    bertscore_score = load_bert_score()
    kwargs: dict[str, Any] = {
        "lang": args.lang,
        "idf": args.idf,
        "rescale_with_baseline": args.rescale_with_baseline,
        "batch_size": args.batch_size,
        "verbose": False,
    }
    if args.model_type:
        kwargs["model_type"] = args.model_type
    if args.num_layers is not None:
        kwargs["num_layers"] = args.num_layers
    if args.device:
        kwargs["device"] = args.device
    if args.use_fast_tokenizer:
        kwargs["use_fast_tokenizer"] = True
    precision, recall, f1 = bertscore_score(candidates, references, **kwargs)
    return (
        [float(value) for value in precision],
        [float(value) for value in recall],
        [float(value) for value in f1],
    )


def summarize_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        category: {
            "matched_image_count": len(items),
            "bertscore_precision": mean([item["bertscore_precision"] for item in items]),
            "bertscore_recall": mean([item["bertscore_recall"] for item in items]),
            "bertscore_f1": mean([item["bertscore_f1"] for item in items]),
            "raw_bertscore_precision": mean([item["raw_bertscore_precision"] for item in items]),
            "raw_bertscore_recall": mean([item["raw_bertscore_recall"] for item in items]),
            "raw_bertscore_f1": mean([item["raw_bertscore_f1"] for item in items]),
        }
        for category, items in sorted(grouped.items())
    }


def calibration_flag(delta_f1: float | None) -> str:
    if delta_f1 is None:
        return "not_available"
    abs_delta = abs(delta_f1)
    if abs_delta >= 0.10:
        return "strong"
    if abs_delta >= 0.05:
        return "noticeable"
    return "none"


def score_scope(
    comparison_id: str,
    comparison: dict[str, str],
    scope: str,
    data: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_key = comparison["reference_key"]
    candidate_key = comparison["candidate_key"]
    reference_bundles = data.get(reference_key, {})
    candidate_bundles = data.get(candidate_key, {})
    common = sorted(set(reference_bundles) & set(candidate_bundles))
    if args.limit > 0:
        common = common[: args.limit]
    unmatched = sorted(set(reference_bundles) ^ set(candidate_bundles))
    result: dict[str, Any] = {
        "comparison": comparison_id,
        "scope": scope,
        "matched_image_count": len(common),
        "unmatched_image_count": len(unmatched),
        "unmatched_images": unmatched,
        "bertscore_precision": 0.0,
        "bertscore_recall": 0.0,
        "bertscore_f1": 0.0,
        "by_category": {},
        "by_image": [],
    }
    if not common:
        result["status"] = "skipped_no_matched_images"
        return result

    candidates = [text_for_scope(candidate_bundles[key], scope) for key in common]
    references = [text_for_scope(reference_bundles[key], scope) for key in common]
    valid_indexes = [
        index for index, (candidate, reference) in enumerate(zip(candidates, references)) if candidate and reference
    ]
    if not valid_indexes:
        result["status"] = "skipped_no_nonempty_caption_text"
        return result

    valid_candidates = [candidates[index] for index in valid_indexes]
    valid_references = [references[index] for index in valid_indexes]
    raw_precision, raw_recall, raw_f1 = compute_bert_scores(valid_candidates, valid_references, args)
    precision = [normalize_cosine_score(value) for value in raw_precision]
    recall = [normalize_cosine_score(value) for value in raw_recall]
    f1 = [normalize_cosine_score(value) for value in raw_f1]

    rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(valid_indexes):
        key = common[source_index]
        candidate_bundle = candidate_bundles[key]
        reference_bundle = reference_bundles[key]
        row = {
            "image": key,
            "category": category_for(candidate_bundle, key) or category_for(reference_bundle, key),
            "comparison": comparison_id,
            "scope": scope,
            "reference_key": reference_key,
            "candidate_key": candidate_key,
            "bertscore_precision": precision[output_index],
            "bertscore_recall": recall[output_index],
            "bertscore_f1": f1[output_index],
            "raw_bertscore_precision": raw_precision[output_index],
            "raw_bertscore_recall": raw_recall[output_index],
            "raw_bertscore_f1": raw_f1[output_index],
            "candidate_text": valid_candidates[output_index],
            "reference_text": valid_references[output_index],
        }
        rows.append(row)

    result.update(
        {
            "status": "scored",
            "scored_image_count": len(rows),
            "empty_text_pair_count": len(common) - len(rows),
            "bertscore_precision": mean([row["bertscore_precision"] for row in rows]),
            "bertscore_recall": mean([row["bertscore_recall"] for row in rows]),
            "bertscore_f1": mean([row["bertscore_f1"] for row in rows]),
            "raw_bertscore_precision": mean([row["raw_bertscore_precision"] for row in rows]),
            "raw_bertscore_recall": mean([row["raw_bertscore_recall"] for row in rows]),
            "raw_bertscore_f1": mean([row["raw_bertscore_f1"] for row in rows]),
            "by_category": summarize_by_category(rows),
            "by_image": rows,
        }
    )
    return result


def score_comparison(
    comparison_id: str,
    comparison: dict[str, str],
    data: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": comparison["name"],
        "group": comparison["group"],
        "meaning": comparison["meaning"],
        "reference_key": comparison["reference_key"],
        "candidate_key": comparison["candidate_key"],
        "scopes": {},
        "by_image_evidence": [],
    }

    for scope in args.selected_scopes:
        result["scopes"][scope] = score_scope(comparison_id, comparison, scope, data, args)

    visual_rows = {
        row["image"]: row
        for row in result["scopes"].get("visual_summary_only", {}).get("by_image", [])
    }
    full_rows = {
        row["image"]: row
        for row in result["scopes"].get("full_caption_fields", {}).get("by_image", [])
    }
    evidence_rows = []
    for image in sorted(set(visual_rows) | set(full_rows)):
        visual = visual_rows.get(image)
        full = full_rows.get(image)
        delta = None
        if visual and full:
            delta = full["bertscore_f1"] - visual["bertscore_f1"]
        for row in [visual, full]:
            if not row:
                continue
            evidence = dict(row)
            evidence["delta_f1"] = delta
            evidence["calibration_flag"] = calibration_flag(delta)
            evidence_rows.append(evidence)
    result["by_image_evidence"] = evidence_rows

    primary_scope = args.selected_scopes[0]
    primary_summary = result["scopes"][primary_scope]
    result["primary_scope"] = primary_scope
    result["status"] = primary_summary.get("status", "unknown")
    result["matched_image_count"] = primary_summary.get("matched_image_count", 0)
    result["scored_image_count"] = primary_summary.get("scored_image_count", 0)
    result["bertscore_precision"] = primary_summary.get("bertscore_precision", 0.0)
    result["bertscore_recall"] = primary_summary.get("bertscore_recall", 0.0)
    result["bertscore_f1"] = primary_summary.get("bertscore_f1", 0.0)
    result["raw_bertscore_precision"] = primary_summary.get("raw_bertscore_precision", 0.0)
    result["raw_bertscore_recall"] = primary_summary.get("raw_bertscore_recall", 0.0)
    result["raw_bertscore_f1"] = primary_summary.get("raw_bertscore_f1", 0.0)
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "unknown"


def path_label(path: Path) -> str:
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        return "_".join(parts[index + 1 :])
    return path.name


def default_report_id(args: argparse.Namespace) -> str:
    candidate_label = path_label(Path(args.candidate_dir))
    reference_label = path_label(Path(args.reference_dir))
    model_label = safe_id(args.model_type.replace("/", "_"))
    return safe_id(
        f"{Path(args.run_dir).name}_{args.provider}_{args.mode}_"
        f"A-{reference_label}_C-{candidate_label}_{model_label}"
    )


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_id = safe_id(args.report_id) if args.report_id else default_report_id(args)
    stem = f"{report_id}_{timestamp}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.md"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# BERTScore Evidence Report",
        "",
        "## 1. Run / Data Source",
        "",
        f"- Candidate run: `{report['run_dir']}`",
        f"- Main comparisons: `{', '.join(report['main_comparisons'])}`",
        f"- Diagnostic comparisons included: `{report['include_diagnostic']}`",
        "",
    ]
    data_rows = []
    for key, item in report["data_map"].items():
        data_rows.append(
            [
                key,
                item["path"],
                item["exists"],
                item["file_count"],
                item["filename_overlap_with_source"],
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "Path", "Exists", "JSON files", "Filename overlap"],
            data_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 2. BERTScore Parameters",
            "",
        ]
    )
    settings = report["settings"]
    lines.extend(
        markdown_table(
            ["Parameter", "Value"],
            [[key, settings.get(key)] for key in sorted(settings)],
        )
    )
    lines.extend(
        [
            "",
            "## 3. Summary",
            "",
        ]
    )
    summary_rows = []
    for comparison_id, item in report["summary"].items():
        for scope, scope_item in item.get("scopes", {}).items():
            summary_rows.append(
                [
                    comparison_id,
                    item["group"],
                    scope,
                    scope_item["status"],
                    scope_item["matched_image_count"],
                    scope_item.get("scored_image_count", 0),
                    fmt(scope_item["bertscore_precision"]),
                    fmt(scope_item["bertscore_recall"]),
                    fmt(scope_item["bertscore_f1"]),
                ]
            )
    lines.extend(
        markdown_table(
            ["Comparison", "Group", "Scope", "Status", "Matched", "Scored", "P", "R", "F1"],
            summary_rows,
        )
    )
    lines.extend(["", "## 4. Category Table", ""])
    for comparison_id, item in report["summary"].items():
        lines.extend([f"### {comparison_id} - {item['name']}", ""])
        for scope, scope_item in item.get("scopes", {}).items():
            lines.extend([f"#### {scope}", ""])
            category_rows = [
                [
                    category,
                    values["matched_image_count"],
                    fmt(values["bertscore_precision"]),
                    fmt(values["bertscore_recall"]),
                    fmt(values["bertscore_f1"]),
                ]
                for category, values in scope_item.get("by_category", {}).items()
            ]
            if category_rows:
                lines.extend(markdown_table(["Category", "Count", "P", "R", "F1"], category_rows))
            else:
                lines.append("No scored category rows.")
            lines.append("")
        lines.append("")

    lines.extend(
        [
            "## 5. Per-Image Evidence",
            "",
        ]
    )
    for comparison_id, item in report["summary"].items():
        lines.extend([f"### {comparison_id} - {item['name']}", ""])
        evidence = sorted(
            item.get("by_image_evidence", []),
            key=lambda row: (row["image"], row["scope"]),
        )
        if not evidence:
            lines.append("No scored evidence rows.")
            lines.append("")
            continue
        current_image = None
        for row in evidence:
            if row["image"] != current_image:
                current_image = row["image"]
                lines.extend([f"#### {current_image}", "", f"- Category: `{row['category']}`", ""])
            lines.extend(
                [
                    f"**{row['scope']}**",
                    "",
                    f"- Normalized P/R/F1: `{fmt(row['bertscore_precision'])}` / `{fmt(row['bertscore_recall'])}` / `{fmt(row['bertscore_f1'])}`",
                    f"- Raw P/R/F1: `{fmt(row['raw_bertscore_precision'])}` / `{fmt(row['raw_bertscore_recall'])}` / `{fmt(row['raw_bertscore_f1'])}`",
                    f"- Delta F1 full-minus-visual: `{fmt(row['delta_f1'])}`",
                    f"- Calibration flag: `{row['calibration_flag']}`",
                    "",
                    "Reference text:",
                    "",
                    "```text",
                    row["reference_text"],
                    "```",
                    "",
                    "Candidate text:",
                    "",
                    "```text",
                    row["candidate_text"],
                    "```",
                    "",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 6. Interpretation Notes",
            "",
            "- BERTScore is semantic similarity, not medical correctness.",
            "- Main P/R/F1 values are normalized with `(raw + 1) / 2` and clipped to `[0, 1]`.",
            "- Raw BERTScore values are preserved in JSON and shown in per-image Markdown evidence.",
            "- `AC` and `BD` are the main comparison groups for 620.",
            "- `visual_summary_only` checks the free visual description only.",
            "- `full_caption_fields` includes visual summary plus structured wound fields.",
            "- `--scopes` controls which scope is computed. Use one scope per command when you want independent reports.",
            "- `delta_f1 = full_caption_fields_f1 - visual_summary_only_f1` is available only when both scopes are included in the same report.",
            "- Mixed Chinese-English translation text is expected when English medical terms are intentionally preserved.",
            "- Do not compress AC and BD into one final score.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    reference_dir = Path(args.reference_dir)
    reference_translation_dir = Path(args.reference_translation_dir)
    candidate_dir = Path(args.candidate_dir)
    candidate_translation_dir = Path(args.candidate_translation_dir)
    data_sources = {
        "a_reference_english": {
            "path": reference_dir,
            "filename_overlap_with_source": len(json_stems(reference_dir)),
        },
        "b_translated_reference": {
            "path": reference_translation_dir,
            "filename_overlap_with_source": len(json_stems(reference_dir) & json_stems(reference_translation_dir)),
        },
        "c_candidate_english": {
            "path": candidate_dir,
            "filename_overlap_with_source": len(json_stems(candidate_dir)),
        },
        "d_translated_candidate": {
            "path": candidate_translation_dir,
            "filename_overlap_with_source": len(json_stems(candidate_dir) & json_stems(candidate_translation_dir)),
        },
    }
    data_paths = {key: value["path"] for key, value in data_sources.items()}
    data = {key: load_bundles(path) for key, path in data_paths.items()}
    data_map = {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(data[key]),
            "filename_overlap_with_source": data_sources[key]["filename_overlap_with_source"],
        }
        for key, path in data_paths.items()
    }
    require_complete_grouped_inputs(data_map)
    settings = {
        "lang": args.lang,
        "model_type": args.model_type,
        "num_layers": args.num_layers,
        "idf": args.idf,
        "rescale_with_baseline": args.rescale_with_baseline,
        "device": args.device or "auto",
        "batch_size": args.batch_size,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "mixed_language_expected": True,
        "score_normalization": "cosine_to_0_1",
        "score_normalization_formula": "normalized = clamp((raw + 1) / 2, 0, 1)",
        "score_normalization_note": "Main bertscore_* fields are normalized; raw_bertscore_* fields preserve the BERTScore library output.",
        "hash_code": None,
        "hash_code_note": "Not available from this wrapper; preserve all settings for reproducibility.",
        "caption_fields": CAPTION_FIELDS,
        "available_scopes": SCOPES,
        "selected_scopes": args.selected_scopes,
    }
    comparisons = dict(COMPARISONS)
    if args.include_diagnostic:
        comparisons.update(DIAGNOSTIC_COMPARISONS)
    summary = {
        comparison_id: score_comparison(comparison_id, comparison, data, args)
        for comparison_id, comparison in comparisons.items()
    }
    return {
        "created_at": now_iso(),
        "metric": "BERTScore",
        "score_type": "grouped_semantic_comparison",
        "status": "exploratory",
        "run_dir": str(run_dir),
        "provider": args.provider,
        "mode": args.mode,
        "limit": args.limit,
        "data_map": data_map,
        "comparison_groups": {
            "main": list(COMPARISONS),
            "diagnostic": list(DIAGNOSTIC_COMPARISONS),
        },
        "main_comparisons": list(COMPARISONS),
        "include_diagnostic": args.include_diagnostic,
        "settings": settings,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json, output_md = output_paths(args)
    write_json(output_json, report)
    write_markdown(output_md, report)
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")


if __name__ == "__main__":
    main()
