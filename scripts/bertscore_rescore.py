import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DIR = Path("runs/saas_simple_baseline")
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
    "a_vs_c": {
        "group": "caption_quality",
        "name": "caption_source_score",
        "reference_key": "a_reference_english",
        "candidate_key": "c_candidate_english",
        "meaning": "local English caption vs SaaS/Gemini English reference",
    },
    "a_vs_b": {
        "group": "translation_faithfulness",
        "name": "reference_translation_faithfulness",
        "reference_key": "a_reference_english",
        "candidate_key": "b_translated_reference",
        "meaning": "translated reference vs original English reference",
    },
    "c_vs_d": {
        "group": "translation_faithfulness",
        "name": "candidate_translation_faithfulness",
        "reference_key": "c_candidate_english",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs original English candidate",
    },
    "a_vs_d": {
        "group": "human_review_calibration",
        "name": "crosslingual_candidate_to_reference",
        "reference_key": "a_reference_english",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs English reference",
    },
    "b_vs_d": {
        "group": "human_review_calibration",
        "name": "translated_review_score",
        "reference_key": "b_translated_reference",
        "candidate_key": "d_translated_candidate",
        "meaning": "translated candidate vs translated reference",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute grouped BERTScore comparisons from existing run outputs."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--mode", default="simple")
    parser.add_argument("--score-type", choices=["grouped"], default="grouped")
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--translation-root", default=None)
    parser.add_argument("--reference-translation-dir", default=None)
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--candidate-translation-dir", default=None)
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--model-type", default="bert-base-multilingual-cased")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing a/b/c/d sources and score only available comparison groups.",
    )
    return parser.parse_args()


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


def runs_root_for(run_dir: Path) -> Path:
    return run_dir.parent if run_dir.parent.name == "runs" else Path("runs")


def candidate_translation_dirs(runs_root: Path, provider: str, mode: str) -> list[Path]:
    candidates = []
    if runs_root.exists():
        candidates.extend(sorted(runs_root.glob(f"*/output_translation/{provider}/{mode}")))
    return candidates


def normalize_model_name(model: str | None) -> str:
    return str(model or "").lower().replace(":", "_").replace("-", "_")


def first_model_in_dir(path: Path) -> str:
    for json_path in sorted(path.glob("*.json")):
        try:
            bundle = read_json(json_path)
        except json.JSONDecodeError:
            continue
        model = bundle.get("metadata", {}).get("model")
        if model:
            return str(model)
    return ""


def first_translation_source_model_in_dir(path: Path) -> str:
    for json_path in sorted(path.glob("*.json")):
        try:
            bundle = read_json(json_path)
        except json.JSONDecodeError:
            continue
        metadata = bundle.get("metadata", {})
        source_model = metadata.get("translation", {}).get("source_model") or metadata.get("model")
        if source_model:
            return str(source_model)
    return ""


def resolve_translation_dir(
    preferred: Path,
    source_dir: Path,
    runs_root: Path,
    provider: str,
    mode: str,
    source_model: str = "",
) -> tuple[Path, str, int]:
    source_stems = json_stems(source_dir)
    if preferred.exists():
        return preferred, "provided", len(source_stems & json_stems(preferred))

    best_path = preferred
    best_overlap = 0
    best_model_match = False
    expected_model = normalize_model_name(source_model)
    for candidate in candidate_translation_dirs(runs_root, provider, mode):
        overlap = len(source_stems & json_stems(candidate))
        candidate_model = normalize_model_name(first_translation_source_model_in_dir(candidate))
        model_match = bool(expected_model and candidate_model == expected_model)
        if (overlap, model_match) > (best_overlap, best_model_match):
            best_path = candidate
            best_overlap = overlap
            best_model_match = model_match

    if best_overlap > 0:
        source = "auto_discovered_by_filename_overlap"
        if best_model_match:
            source = "auto_discovered_by_filename_and_model_match"
        return best_path, source, best_overlap
    return preferred, "missing", 0


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
        "Fix by running `python scripts/translate.py --run-dir <run-dir> --model <model>` "
        "or pass explicit paths with `--reference-translation-dir` and "
        "`--candidate-translation-dir`. Use `--allow-partial` only if you intentionally "
        "want an incomplete report."
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        }
        for category, items in sorted(grouped.items())
    }


def score_comparison(
    comparison_id: str,
    comparison: dict[str, str],
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
        "name": comparison["name"],
        "group": comparison["group"],
        "meaning": comparison["meaning"],
        "reference_key": reference_key,
        "candidate_key": candidate_key,
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

    candidates = [caption_text(candidate_bundles[key]) for key in common]
    references = [caption_text(reference_bundles[key]) for key in common]
    valid_indexes = [
        index for index, (candidate, reference) in enumerate(zip(candidates, references)) if candidate and reference
    ]
    if not valid_indexes:
        result["status"] = "skipped_no_nonempty_caption_text"
        return result

    valid_candidates = [candidates[index] for index in valid_indexes]
    valid_references = [references[index] for index in valid_indexes]
    precision, recall, f1 = compute_bert_scores(valid_candidates, valid_references, args)

    rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(valid_indexes):
        key = common[source_index]
        candidate_bundle = candidate_bundles[key]
        reference_bundle = reference_bundles[key]
        row = {
            "image": key,
            "category": category_for(candidate_bundle, key) or category_for(reference_bundle, key),
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
            "by_category": summarize_by_category(rows),
            "by_image": rows,
        }
    )
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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Grouped Semantic Comparison BERTScore",
        "",
        "## 1. Run / Data Source",
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
                item["source"],
                item["filename_overlap_with_source"],
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "Path", "Exists", "JSON files", "Source", "Filename overlap"],
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
        summary_rows.append(
            [
                comparison_id,
                item["group"],
                item["name"],
                item["status"],
                item["matched_image_count"],
                item.get("scored_image_count", 0),
                fmt(item["bertscore_precision"]),
                fmt(item["bertscore_recall"]),
                fmt(item["bertscore_f1"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["Comparison", "Group", "Name", "Status", "Matched", "Scored", "P", "R", "F1"],
            summary_rows,
        )
    )
    lines.extend(["", "## 4. Category Table", ""])
    for comparison_id, item in report["summary"].items():
        lines.extend([f"### {comparison_id} - {item['name']}", ""])
        category_rows = [
            [
                category,
                values["matched_image_count"],
                fmt(values["bertscore_precision"]),
                fmt(values["bertscore_recall"]),
                fmt(values["bertscore_f1"]),
            ]
            for category, values in item.get("by_category", {}).items()
        ]
        if category_rows:
            lines.extend(markdown_table(["Category", "Count", "P", "R", "F1"], category_rows))
        else:
            lines.append("No scored category rows.")
        lines.append("")

    lines.extend(
        [
            "## 5. Lowest / Highest Examples",
            "",
        ]
    )
    for comparison_id, item in report["summary"].items():
        images = sorted(item.get("by_image", []), key=lambda row: row["bertscore_f1"])
        lines.extend([f"### {comparison_id} - {item['name']}", ""])
        example_rows = []
        for row in images[:3] + images[-3:]:
            example_rows.append(
                [
                    row["image"],
                    row["category"],
                    fmt(row["bertscore_precision"]),
                    fmt(row["bertscore_recall"]),
                    fmt(row["bertscore_f1"]),
                ]
            )
        if example_rows:
            lines.extend(markdown_table(["Image", "Category", "P", "R", "F1"], example_rows))
        else:
            lines.append("No scored examples.")
        lines.append("")

    lines.extend(
        [
            "## 6. Interpretation Notes",
            "",
            "- BERTScore is semantic similarity, not medical correctness.",
            "- Caption quality, translation faithfulness, and human-review calibration are separate comparison groups with equal report priority.",
            "- Mixed Chinese-English translation text is expected when English medical terms are intentionally preserved.",
            "- Do not compress the five comparison groups into one final score.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    runs_root = runs_root_for(run_dir)
    translation_root = Path(args.translation_root) if args.translation_root else run_dir / "output_translation"
    reference_dir = Path(args.reference_dir)
    candidate_dir = (
        Path(args.candidate_dir)
        if args.candidate_dir
        else run_dir / "outputs" / args.provider / args.mode
    )
    preferred_reference_translation_dir = (
        Path(args.reference_translation_dir)
        if args.reference_translation_dir
        else translation_root / "saas" / args.mode
    )
    preferred_candidate_translation_dir = (
        Path(args.candidate_translation_dir)
        if args.candidate_translation_dir
        else translation_root / args.provider / args.mode
    )
    reference_translation_dir, reference_translation_source, reference_translation_overlap = (
        resolve_translation_dir(
            preferred_reference_translation_dir,
            reference_dir,
            runs_root,
            "saas",
            args.mode,
            first_model_in_dir(reference_dir),
        )
    )
    candidate_translation_dir, candidate_translation_source, candidate_translation_overlap = (
        resolve_translation_dir(
            preferred_candidate_translation_dir,
            candidate_dir,
            runs_root,
            args.provider,
            args.mode,
            first_model_in_dir(candidate_dir),
        )
    )
    data_sources = {
        "a_reference_english": {
            "path": reference_dir,
            "source": "provided",
            "filename_overlap_with_source": len(json_stems(reference_dir)),
        },
        "b_translated_reference": {
            "path": reference_translation_dir,
            "source": reference_translation_source,
            "filename_overlap_with_source": reference_translation_overlap,
        },
        "c_candidate_english": {
            "path": candidate_dir,
            "source": "provided",
            "filename_overlap_with_source": len(json_stems(candidate_dir)),
        },
        "d_translated_candidate": {
            "path": candidate_translation_dir,
            "source": candidate_translation_source,
            "filename_overlap_with_source": candidate_translation_overlap,
        },
    }
    data_paths = {key: value["path"] for key, value in data_sources.items()}
    data = {key: load_bundles(path) for key, path in data_paths.items()}
    data_map = {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(data[key]),
            "source": data_sources[key]["source"],
            "filename_overlap_with_source": data_sources[key]["filename_overlap_with_source"],
        }
        for key, path in data_paths.items()
    }
    if not args.allow_partial:
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
        "hash_code": None,
        "hash_code_note": "Not available from this wrapper; preserve all settings for reproducibility.",
        "caption_fields": CAPTION_FIELDS,
    }
    summary = {
        comparison_id: score_comparison(comparison_id, comparison, data, args)
        for comparison_id, comparison in COMPARISONS.items()
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
            "caption_quality": ["a_vs_c"],
            "translation_faithfulness": ["a_vs_b", "c_vs_d"],
            "human_review_calibration": ["a_vs_d", "b_vs_d"],
        },
        "settings": settings,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    report = build_report(args)
    output_json = run_dir / "bertscore_comparisons.json"
    output_md = run_dir / "bertscore_comparisons.md"
    write_json(output_json, report)
    write_markdown(output_md, report)
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")


if __name__ == "__main__":
    main()
