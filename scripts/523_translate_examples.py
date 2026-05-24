import argparse
import json
from pathlib import Path
from typing import Any

import ollama


FIELDS_TO_TRANSLATE = [
    ("image_observation", "visual_summary"),
    ("wound_features", "shape_pattern"),
    ("wound_features", "edges_margins"),
    ("wound_features", "wound_bed"),
    ("wound_features", "color"),
    ("wound_features", "texture"),
    ("wound_features", "fluid_exudate_bleeding"),
    ("wound_features", "periwound_skin"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_map(run_dir: Path, provider: str, mode: str) -> dict[str, Path]:
    output_dir = run_dir / "outputs" / provider / mode
    return {path.stem: path for path in sorted(output_dir.glob("*.json"))}


def caption_extract(bundle: dict[str, Any]) -> dict[str, Any]:
    caption = bundle.get("caption", {})
    extracted: dict[str, Any] = {}
    for section, field in FIELDS_TO_TRANSLATE:
        extracted[f"{section}.{field}"] = caption.get(section, {}).get(field, "")
    extracted["category_specific_check.observed_supporting_features"] = caption.get(
        "category_specific_check", {}
    ).get("observed_supporting_features", [])
    extracted["uncertainty.cannot_determine"] = caption.get("uncertainty", {}).get(
        "cannot_determine", []
    )
    return extracted


def translation_prompt(example: dict[str, Any]) -> str:
    return (
        "You are translating wound image-captioning review text for a Traditional Chinese "
        "research report.\n\n"
        "Rules:\n"
        "- Output valid JSON only.\n"
        "- Use Traditional Chinese.\n"
        "- Preserve important English professional terms in parentheses, for example "
        "紅斑（erythema）, 滲出液（exudate）, 壞死（necrosis）, 焦痂（eschar）.\n"
        "- Do not add new medical claims.\n"
        "- Translate the meaning faithfully for human review only.\n\n"
        "Input JSON:\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}\n\n"
        "Output JSON shape:\n"
        "{\n"
        '  "local_zh_tw": {},\n'
        '  "saas_zh_tw": {},\n'
        '  "translation_notes": []\n'
        "}\n"
    )


def parse_json_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def translate_example(model: str, example: dict[str, Any]) -> dict[str, Any]:
    try:
        response = ollama.chat(
            model=model,
            format="json",
            messages=[{"role": "user", "content": translation_prompt(example)}],
        )
    except Exception as exc:
        raise RuntimeError(
            "Ollama local model call failed. Start Ollama or choose an installed model. "
            f"Original error: {exc}"
        ) from exc
    return parse_json_text(response["message"]["content"])


def select_examples(run_dir: Path, limit: int) -> list[dict[str, Any]]:
    local = output_map(run_dir, "local", "simple")
    saas = output_map(run_dir, "saas", "simple")
    common = sorted(set(local) & set(saas))
    examples = []
    for key in common[:limit]:
        local_bundle = read_json(local[key])
        saas_bundle = read_json(saas[key])
        metadata = local_bundle.get("metadata", {})
        examples.append(
            {
                "image_id": key,
                "image": metadata.get("image"),
                "image_name": metadata.get("image_name"),
                "category": metadata.get("raw_category"),
                "local": caption_extract(local_bundle),
                "saas": caption_extract(saas_bundle),
            }
        )
    return examples


def build_markdown(records: list[dict[str, Any]], model: str) -> str:
    lines = [
        "# 523 Translation Examples",
        "",
        f"Translation model: `{model}`",
        "",
        "These translations are for human review only and are not scoring inputs.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['image_id']}",
                "",
                f"- Image: `{record['image']}`",
                f"- Category: `{record['category']}`",
                "",
                "### Local Translation",
                "",
                "```json",
                json.dumps(record["translation"].get("local_zh_tw", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "### SaaS Translation",
                "",
                "```json",
                json.dumps(record["translation"].get("saas_zh_tw", {}), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        notes = record["translation"].get("translation_notes", [])
        if notes:
            lines.extend(["### Notes", ""])
            lines.extend(f"- {note}" for note in notes)
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate selected smoke examples to zh-TW.")
    parser.add_argument("--run-dir", default="runs/510_smoke_v1")
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-md", default="docs/523translation_examples.md")
    parser.add_argument(
        "--output-json",
        default="runs/510_smoke_v1/translations/selected_examples_zh_tw.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    examples = select_examples(run_dir, args.limit)
    records = []
    for example in examples:
        print(f"Translating {example['image_id']}...")
        records.append({**example, "translation": translate_example(args.model, example)})

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model": args.model,
                "count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(records, args.model), encoding="utf-8")
    print(f"Saved {output_md}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
