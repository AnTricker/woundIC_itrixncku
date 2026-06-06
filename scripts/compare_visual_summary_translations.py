import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_RUN_510 = Path("runs/510_smoke_v1")
DEFAULT_RUN_523 = Path("runs/523_smoke_v2")
ANALYSIS_APPENDIX_MARKER = "\n## 分數偏低原因分析\n"
LABEL_510_LOCAL = "gemma4:e2b"
LABEL_523_LOCAL = "gemma4:26b(MoE)"
LABEL_SAAS_SIMPLE = "gemini-3.1flash-lite"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def simple_translation_dir(run_dir: Path, provider: str) -> Path:
    return run_dir / "output_translation" / provider / "simple"


def load_outputs(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing translated output directory: {directory}")
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        outputs[path.stem] = read_json(path)
    return outputs


def image_name(bundle: dict[str, Any], fallback_stem: str) -> str:
    metadata = bundle.get("metadata", {})
    return metadata.get("image_name") or f"{fallback_stem}.jpg"


def category_name(bundle: dict[str, Any], fallback_stem: str) -> str:
    metadata = bundle.get("metadata", {})
    return (
        metadata.get("raw_category")
        or metadata.get("target_category")
        or fallback_stem.split(" (", 1)[0]
    )


def visual_summary(bundle: dict[str, Any]) -> str:
    value = (
        bundle.get("caption", {})
        .get("image_observation", {})
        .get("visual_summary")
    )
    if value is None:
        return "missing visual_summary"
    if isinstance(value, str):
        return value.strip() or "empty visual_summary"
    return json.dumps(value, ensure_ascii=False)


def markdown_image_link(bundle: dict[str, Any], output_path: Path, fallback_stem: str) -> str:
    image_value = bundle.get("metadata", {}).get("image")
    image_path = Path(image_value) if image_value else Path("images") / f"{fallback_stem}.jpg"
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    relative_path = os.path.relpath(image_path, output_path.parent)
    encoded_path = "/".join(quote(part) for part in Path(relative_path).parts)
    return (
        f'<img src="{encoded_path}" alt="{image_name(bundle, fallback_stem)}" '
        'width="50%">'
    )


def build_report(
    *,
    run_510: Path,
    run_523: Path,
    output_path: Path,
    per_category: int,
    seed: int,
) -> str:
    local_510 = load_outputs(simple_translation_dir(run_510, "local"))
    local_523 = load_outputs(simple_translation_dir(run_523, "local"))
    saas_simple = load_outputs(simple_translation_dir(run_510, "saas"))

    common_stems = sorted(set(local_510) & set(local_523) & set(saas_simple))
    by_category: dict[str, list[str]] = defaultdict(list)
    for stem in common_stems:
        by_category[category_name(local_510[stem], stem)].append(stem)

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for category, stems in sorted(by_category.items()):
        selected[category] = sorted(rng.sample(stems, min(per_category, len(stems))))

    lines = [
        "# Visual Summary Translation Comparison",
        "",
        "Purpose:",
        "",
        "- Randomly select translated simple-prompt results by wound type.",
        "- Compare only the `caption.image_observation.visual_summary` field.",
        f"- Each selected image is listed in this order: {LABEL_510_LOCAL}, {LABEL_523_LOCAL}, {LABEL_SAAS_SIMPLE}.",
        "",
        "Sources:",
        "",
        f"- {LABEL_510_LOCAL}: `{simple_translation_dir(run_510, 'local')}`",
        f"- {LABEL_523_LOCAL}: `{simple_translation_dir(run_523, 'local')}`",
        f"- {LABEL_SAAS_SIMPLE}: `{simple_translation_dir(run_510, 'saas')}`",
        "",
        "Selection:",
        "",
        f"- Seed: `{seed}`",
        f"- Requested per type: `{per_category}`",
        f"- Complete matched image count: `{len(common_stems)}`",
        "",
    ]

    for category, stems in selected.items():
        lines.extend([f"## {category}", ""])
        if not stems:
            lines.extend(["No complete matched translations found.", ""])
            continue
        for stem in stems:
            lines.extend(
                [
                    f"### {image_name(local_510[stem], stem)}",
                    "",
                    markdown_image_link(local_510[stem], output_path, stem),
                    "",
                    f"- which image(filename): `{image_name(local_510[stem], stem)}`",
                    f"- {LABEL_510_LOCAL}: {visual_summary(local_510[stem])}",
                    f"- {LABEL_523_LOCAL}: {visual_summary(local_523[stem])}",
                    f"- {LABEL_SAAS_SIMPLE}: {visual_summary(saas_simple[stem])}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare translated visual_summary values across 510, 523, and SaaS simple outputs."
    )
    parser.add_argument("--run-510", default=str(DEFAULT_RUN_510))
    parser.add_argument("--run-523", default=str(DEFAULT_RUN_523))
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="runs/visual_summary_translation_comparison.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    runs_root = Path("runs").resolve()
    if not output_path.resolve().is_relative_to(runs_root):
        raise ValueError("--output must stay inside runs/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_appendix = ""
    if output_path.exists():
        existing_report = output_path.read_text(encoding="utf-8")
        marker_index = existing_report.find(ANALYSIS_APPENDIX_MARKER)
        if marker_index >= 0:
            existing_appendix = existing_report[marker_index:]
    report = build_report(
        run_510=Path(args.run_510),
        run_523=Path(args.run_523),
        output_path=output_path,
        per_category=args.per_type,
        seed=args.seed,
    )
    if existing_appendix:
        report = report.rstrip() + "\n" + existing_appendix
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
