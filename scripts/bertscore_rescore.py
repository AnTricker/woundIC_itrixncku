import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version as package_version
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
        "lang": "en",
        "default_model": "roberta-large",
        "meaning": "local English caption vs SaaS/Gemini English reference",
    },
    "BD": {
        "group": "human_review_calibration",
        "name": "translated_review_score",
        "reference_key": "b_translated_reference",
        "candidate_key": "d_translated_candidate",
        "lang": "zh",
        "default_model": "bert-base-chinese",
        "meaning": "translated candidate vs translated reference",
    },
}
SCOPES = ("visual_summary_only", "full_caption_fields")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one official language-specific, baseline-rescaled BERTScore report "
            "from existing outputs."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--comparisons",
        required=True,
        help="One comparison per report: AC (English) or BD (Chinese translation).",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="A source: reference English JSON directory.",
    )
    parser.add_argument(
        "--reference-translation-dir",
        default=None,
        help="B source: translated reference JSON directory.",
    )
    parser.add_argument(
        "--candidate-dir",
        default=None,
        help="C source: candidate English JSON directory.",
    )
    parser.add_argument(
        "--candidate-translation-dir",
        default=None,
        help="D source: translated candidate JSON directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/bertscore_reports",
        help="Independent output folder. Do not put BERTScore reports inside a single run folder.",
    )
    parser.add_argument("--report-id", default=None)
    parser.add_argument(
        "--scopes",
        required=True,
        help="One scoring scope per report: visual_summary_only or full_caption_fields.",
    )
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.selected_comparisons = parse_comparisons(args.comparisons)
    args.selected_scopes = parse_scopes(args.scopes)
    return args


def parse_comparisons(value: str) -> list[str]:
    comparisons = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [item for item in comparisons if item not in COMPARISONS]
    if invalid:
        raise SystemExit(
            "Invalid --comparisons value: "
            + ", ".join(invalid)
            + f". Valid comparisons: {', '.join(COMPARISONS)}"
        )
    if len(comparisons) != 1:
        raise SystemExit("--comparisons must select exactly one comparison per report.")
    return comparisons


def parse_scopes(value: str) -> list[str]:
    scopes = [scope.strip() for scope in value.split(",") if scope.strip()]
    invalid = [scope for scope in scopes if scope not in SCOPES]
    if invalid:
        raise SystemExit(
            "Invalid --scopes value: "
            + ", ".join(invalid)
            + f". Valid scopes: {', '.join(SCOPES)}"
        )
    if len(scopes) != 1:
        raise SystemExit("--scopes must select exactly one scope per report.")
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


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def text_for_scope(bundle: dict[str, Any], scope: str) -> str:
    if scope == "visual_summary_only":
        return normalize_text(visual_summary_text(bundle))
    if scope == "full_caption_fields":
        return normalize_text(caption_text(bundle))
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


def require_complete_inputs(data_map: dict[str, dict[str, Any]]) -> None:
    missing = [
        f"{key}: {item['path']} (exists={item['exists']}, json_files={item['file_count']})"
        for key, item in data_map.items()
        if item["file_count"] == 0
    ]
    if not missing:
        return
    detail = "\n".join(f"- {item}" for item in missing)
    raise SystemExit(
        "BERTScore needs every data source required by the selected comparison.\n"
        "Missing or empty sources:\n"
        f"{detail}\n\n"
        "Fix by passing the explicit source directories. This script intentionally does "
        "not auto-search other run folders because that can mix unrelated experiments."
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_bert_score_runtime(
    lang: str,
) -> tuple[Any, str, int, Path, dict[str, str]]:
    try:
        import bert_score
        import torch
        import transformers
        from bert_score import score as bertscore_score
        from bert_score.utils import lang2model, model2layers
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: bert_score. Install it with `pip install bert-score` "
            "or add `bert-score` to requirements.txt and install project dependencies."
        ) from exc
    model_type = lang2model[lang]
    num_layers = model2layers[model_type]
    baseline_path = (
        Path(bert_score.__file__).resolve().parent
        / "rescale_baseline"
        / lang
        / f"{model_type}.tsv"
    )
    if not baseline_path.is_file():
        raise SystemExit(
            "Official BERTScore rescale baseline is missing for "
            f"lang={lang}, model={model_type}: {baseline_path}"
        )
    versions = {
        "bert_score_version": package_version("bert-score"),
        "transformers_version": str(getattr(transformers, "__version__", "unknown")),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
    }
    return bertscore_score, model_type, num_layers, baseline_path, versions


def compute_bert_scores(
    candidates: list[str],
    references: list[str],
    args: argparse.Namespace,
    lang: str,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    (
        bertscore_score,
        model_type,
        num_layers,
        baseline_path,
        versions,
    ) = load_bert_score_runtime(lang)
    kwargs: dict[str, Any] = {
        "lang": lang,
        "idf": args.idf,
        "rescale_with_baseline": True,
        "return_hash": True,
        "batch_size": args.batch_size,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    if args.use_fast_tokenizer:
        kwargs["use_fast_tokenizer"] = True
    (precision, recall, f1), hash_code = bertscore_score(
        candidates, references, **kwargs
    )
    return (
        [float(value) for value in precision],
        [float(value) for value in recall],
        [float(value) for value in f1],
        {
            "lang": lang,
            "model_type": model_type,
            "num_layers": num_layers,
            "baseline_source": "bert_score_package_default",
            "baseline_path": str(baseline_path),
            "bertscore_hash": str(hash_code),
            "score_representation": "official_baseline_rescaled",
            **versions,
        },
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
        }
        for category, items in sorted(grouped.items())
    }


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
    precision, recall, f1, bertscore_config = compute_bert_scores(
        valid_candidates,
        valid_references,
        args,
        comparison["lang"],
    )
    if bertscore_config["model_type"] != comparison["default_model"]:
        raise RuntimeError(
            f"Unexpected upstream default model for {comparison_id}: "
            f"expected {comparison['default_model']}, got {bertscore_config['model_type']}"
        )

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
            "bertscore_config": bertscore_config,
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
        "lang": comparison["lang"],
        "default_model": comparison["default_model"],
        "scopes": {},
        "by_image_evidence": [],
    }

    for scope in args.selected_scopes:
        result["scopes"][scope] = score_scope(comparison_id, comparison, scope, data, args)

    primary_scope = args.selected_scopes[0]
    primary_summary = result["scopes"][primary_scope]
    result["by_image_evidence"] = primary_summary.get("by_image", [])
    result["primary_scope"] = primary_scope
    result["status"] = primary_summary.get("status", "unknown")
    result["matched_image_count"] = primary_summary.get("matched_image_count", 0)
    result["scored_image_count"] = primary_summary.get("scored_image_count", 0)
    result["bertscore_precision"] = primary_summary.get("bertscore_precision", 0.0)
    result["bertscore_recall"] = primary_summary.get("bertscore_recall", 0.0)
    result["bertscore_f1"] = primary_summary.get("bertscore_f1", 0.0)
    result["bertscore_config"] = primary_summary.get("bertscore_config", {})
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


def default_report_id(args: argparse.Namespace) -> str:
    comparison_id = args.selected_comparisons[0]
    scope = args.selected_scopes[0]
    return safe_id(f"{Path(args.run_dir).name}_{comparison_id}_{scope}_official_rescaled")


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
        f"- Comparison: `{report['comparison']}`",
        f"- Scope: `{report['scope']}`",
        f"- Score representation: `{report['score_representation']}`",
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
                item["filename_overlap_with_pair"],
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
                    scope_item.get("bertscore_config", {}).get("lang", ""),
                    scope_item.get("bertscore_config", {}).get("model_type", ""),
                ]
            )
    lines.extend(
        markdown_table(
            [
                "Comparison", "Group", "Scope", "Status", "Matched", "Scored",
                "P", "R", "F1", "Language", "Model",
            ],
            summary_rows,
        )
    )
    lines.extend(["", "### Upstream Runtime Configuration", ""])
    for comparison_id, item in report["summary"].items():
        config = item.get("bertscore_config", {})
        if not config:
            lines.append(f"- `{comparison_id}` was not scored; no runtime hash is available.")
            continue
        lines.extend(
            markdown_table(
                ["Comparison", "Language", "Model", "Layer", "BERTScore hash", "Baseline"],
                [
                    [
                        comparison_id,
                        config.get("lang"),
                        config.get("model_type"),
                        config.get("num_layers"),
                        config.get("bertscore_hash"),
                        config.get("baseline_path"),
                    ]
                ],
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
                    f"- Official baseline-rescaled P/R/F1: `{fmt(row['bertscore_precision'])}` / `{fmt(row['bertscore_recall'])}` / `{fmt(row['bertscore_f1'])}`",
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
            "- P/R/F1 are rescaled by the official `bert-score` language/model/layer baseline.",
            "- No project-specific `(raw + 1) / 2` transformation is applied.",
            "- `AC` uses the upstream English default; `BD` uses the upstream Chinese default.",
            "- `visual_summary_only` checks the free visual description only.",
            "- `full_caption_fields` includes visual summary plus structured wound fields.",
            "- Each report contains exactly one comparison and one scope.",
            "- Do not compress AC and BD into one final score.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    comparison_id = args.selected_comparisons[0]
    comparison = COMPARISONS[comparison_id]
    source_args = {
        "a_reference_english": "reference_dir",
        "b_translated_reference": "reference_translation_dir",
        "c_candidate_english": "candidate_dir",
        "d_translated_candidate": "candidate_translation_dir",
    }
    required_keys = [comparison["reference_key"], comparison["candidate_key"]]
    missing_args = [
        f"--{source_args[key].replace('_', '-')}"
        for key in required_keys
        if not getattr(args, source_args[key])
    ]
    if missing_args:
        raise SystemExit(
            f"Comparison {comparison_id} requires: {', '.join(missing_args)}"
        )
    data_paths = {
        key: Path(getattr(args, source_args[key]))
        for key in required_keys
    }
    data = {key: load_bundles(path) for key, path in data_paths.items()}
    filename_overlap = len(
        json_stems(data_paths[comparison["reference_key"]])
        & json_stems(data_paths[comparison["candidate_key"]])
    )
    data_map = {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(data[key]),
            "filename_overlap_with_pair": filename_overlap,
        }
        for key, path in data_paths.items()
    }
    require_complete_inputs(data_map)
    settings = {
        "idf": args.idf,
        "rescale_with_baseline": True,
        "device": args.device or "auto",
        "batch_size": args.batch_size,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "score_representation": "official_baseline_rescaled",
        "caption_fields": CAPTION_FIELDS,
        "selected_scopes": args.selected_scopes,
    }
    summary = {comparison_id: score_comparison(comparison_id, comparison, data, args)}
    return {
        "report_schema_version": 2,
        "created_at": now_iso(),
        "metric": "BERTScore",
        "score_type": "official_language_baseline_rescaled",
        "score_representation": "official_baseline_rescaled",
        "status": "exploratory",
        "run_dir": str(run_dir),
        "limit": args.limit,
        "comparison": comparison_id,
        "scope": args.selected_scopes[0],
        "data_map": data_map,
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
