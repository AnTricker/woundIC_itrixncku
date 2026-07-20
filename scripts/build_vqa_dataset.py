import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CHOICES = [
    "abrasions",
    "bruises",
    "burns",
    "cut",
    "ingrown_nail",
    "laceration",
    "stab_wound",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic visual-summary multiple-choice VQA dataset from run outputs."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--mode", default="simple")
    parser.add_argument("--language", choices=["en", "zh_tw"], default="zh_tw")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def category_for(bundle: dict[str, Any], fallback_name: str) -> str:
    metadata = bundle.get("metadata", {})
    category_check = bundle.get("caption", {}).get("category_specific_check", {})
    return (
        metadata.get("target_category")
        or metadata.get("raw_category")
        or category_check.get("target_category")
        or fallback_name.split(" (", 1)[0]
        or "unknown"
    )


def visual_summary(bundle: dict[str, Any]) -> str:
    return str(
        bundle.get("caption", {})
        .get("image_observation", {})
        .get("visual_summary", "")
        or ""
    ).strip()


def image_name(bundle: dict[str, Any], fallback: str) -> str:
    metadata = bundle.get("metadata", {})
    return str(metadata.get("image_name") or Path(str(metadata.get("image", fallback))).name)


def rotate_choices(answer: str, categories: list[str], count: int = 4) -> list[str]:
    choices = [answer]
    for category in categories:
        if category != answer and category not in choices:
            choices.append(category)
        if len(choices) >= count:
            break
    return choices


def question_for(language: str) -> str:
    if language == "en":
        return "Based on the visual description, which wound category is most likely?"
    return "根據視覺描述，最可能是哪一類傷口？"


def build_rows(run_dir: Path, provider: str, mode: str, language: str) -> list[dict[str, Any]]:
    output_dir = run_dir / "outputs" / provider / mode
    if not output_dir.exists():
        raise SystemExit(f"Output directory does not exist: {output_dir}")

    bundles: list[tuple[Path, dict[str, Any]]] = []
    categories = set(DEFAULT_CHOICES)
    for path in sorted(output_dir.glob("*.json")):
        try:
            bundle = read_json(path)
        except json.JSONDecodeError:
            continue
        bundles.append((path, bundle))
        categories.add(category_for(bundle, path.stem))

    category_list = sorted(categories)
    rows: list[dict[str, Any]] = []
    for path, bundle in bundles:
        answer = category_for(bundle, path.stem)
        evidence_text = visual_summary(bundle)
        if not evidence_text:
            continue
        rows.append(
            {
                "image": image_name(bundle, path.name),
                "image_stem": path.stem,
                "category": answer,
                "source_run": str(run_dir),
                "source_provider": provider,
                "source_mode": mode,
                "caption_scope": "visual_summary_only",
                "question": question_for(language),
                "choices": rotate_choices(answer, category_list),
                "answer": answer,
                "evidence_text": evidence_text,
                "language": language,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VQA Dataset Summary",
        "",
        f"- Row count: `{len(rows)}`",
        "- Generation: deterministic template, no LLM call.",
        "- Source caption field: `caption.image_observation.visual_summary`",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    for category, count in sorted(counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Example Rows", ""])
    for row in rows[:5]:
        lines.extend(
            [
                f"### {row['image']}",
                "",
                f"- Answer: `{row['answer']}`",
                f"- Choices: `{', '.join(row['choices'])}`",
                "",
                "Evidence:",
                "",
                "```text",
                row["evidence_text"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = build_rows(run_dir, args.provider, args.mode, args.language)
    stem = f"{run_dir.name}_{args.provider}_{args.mode}_visual_summary_vqa"
    output_dir = run_dir / "vqa_dataset"
    jsonl_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem}_summary.md"
    write_jsonl(jsonl_path, rows)
    write_json(output_dir / f"{stem}_summary.json", {"row_count": len(rows), "rows": rows})
    write_markdown(summary_path, rows)
    print(f"Saved {jsonl_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
