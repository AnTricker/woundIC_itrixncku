import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROVIDERS = ("local", "saas")
MODES = ("simple", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze output schema fields for presence, fixed values, and rough text cost."
    )
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, child_prefix))
        return result
    if isinstance(value, list):
        return {prefix: " ".join(str(item) for item in value)}
    return {prefix: value}


def text_len(value: Any) -> int:
    if value is None:
        return 0
    return len(str(value))


def is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def load_outputs(run_dir: Path) -> list[tuple[str, str, Path, dict[str, Any]]]:
    outputs = []
    for provider in PROVIDERS:
        for mode in MODES:
            output_dir = run_dir / "outputs" / provider / mode
            if not output_dir.exists():
                continue
            for path in sorted(output_dir.glob("*.json")):
                try:
                    outputs.append((provider, mode, path, read_json(path)))
                except json.JSONDecodeError:
                    continue
    return outputs


def analyze(outputs: list[tuple[str, str, Path, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for provider, mode, path, bundle in outputs:
        key = f"{provider}/{mode}"
        flat_caption = flatten(bundle.get("caption", {}), "caption")
        flat_extra = {
            "auxiliary_vqa": bundle.get("auxiliary_vqa"),
            "self_check": bundle.get("self_check"),
            "schema_validation.pass": bundle.get("schema_validation", {}).get("pass"),
        }
        grouped[key].append({**flat_caption, **flat_extra})

    result: dict[str, Any] = {}
    for group, rows in sorted(grouped.items()):
        all_fields = sorted({field for row in rows for field in row})
        field_rows = []
        for field in all_fields:
            values = [row.get(field) for row in rows]
            present_values = [value for value in values if not is_empty(value)]
            value_counter = Counter(str(value) for value in present_values)
            most_common_value, most_common_count = (
                value_counter.most_common(1)[0] if value_counter else ("", 0)
            )
            field_rows.append(
                {
                    "field": field,
                    "output_count": len(rows),
                    "present_count": len(present_values),
                    "present_rate": len(present_values) / len(rows) if rows else 0.0,
                    "unique_value_count": len(value_counter),
                    "most_common_value": most_common_value,
                    "most_common_rate": most_common_count / len(present_values) if present_values else 0.0,
                    "avg_text_chars": sum(text_len(value) for value in present_values) / len(present_values)
                    if present_values
                    else 0.0,
                    "candidate_action": candidate_action(
                        len(present_values) / len(rows) if rows else 0.0,
                        len(value_counter),
                        most_common_count / len(present_values) if present_values else 0.0,
                        field,
                    ),
                }
            )
        result[group] = {
            "output_count": len(rows),
            "fields": field_rows,
        }
    return result


def candidate_action(
    present_rate: float,
    unique_count: int,
    most_common_rate: float,
    field: str,
) -> str:
    if field in {"caption.image_observation.visual_summary", "schema_validation.pass"}:
        return "keep"
    if present_rate < 0.20:
        return "consider optional/remove"
    if unique_count <= 2 and most_common_rate >= 0.80:
        return "possible fixed field; inspect"
    return "keep or review"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "/").replace("\n", " ")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Schema Field Analysis",
        "",
        "This report analyzes generated output fields only. It does not modify schema or prompts.",
        "",
    ]
    for group, item in data.items():
        lines.extend(
            [
                f"## {group}",
                "",
                f"- Output count: `{item['output_count']}`",
                "",
                "| Field | Present | Unique | Most common rate | Avg chars | Suggested action |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in item["fields"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(row["field"]),
                        f"{row['present_count']}/{row['output_count']} ({row['present_rate']:.1%})",
                        fmt(row["unique_value_count"]),
                        f"{row['most_common_rate']:.1%}",
                        fmt(row["avg_text_chars"]),
                        fmt(row["candidate_action"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    outputs = load_outputs(run_dir)
    data = {
        "run_dir": str(run_dir),
        "output_count": len(outputs),
        "groups": analyze(outputs),
    }
    output_dir = run_dir / "schema_analysis"
    json_path = output_dir / f"{run_dir.name}_schema_field_analysis.json"
    md_path = output_dir / f"{run_dir.name}_schema_field_analysis.md"
    write_json(json_path, data)
    write_markdown(md_path, data["groups"])
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
