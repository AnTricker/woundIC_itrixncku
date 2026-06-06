import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def backup_charts(charts_dir: Path) -> Path:
    backup_dir = charts_dir / "_backup_before_523_fix"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in charts_dir.glob("*.png"):
        target = backup_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
    return backup_dir


def save_bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    note: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(labels, values, color=["#2b6db0", "#db7842", "#499868", "#9a67ba"])
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}" if isinstance(value, float) and value < 10 else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if note:
        fig.text(0.01, 0.01, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite 510 smoke charts with clear labels.")
    parser.add_argument("--run-dir", default="runs/510_smoke_v1")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    charts_dir = run_dir / "charts"
    scores = read_json(run_dir / "scores.json")
    record = read_json(run_dir / "experiment_record.json")
    split = read_json(run_dir / "split_summary.json")
    backup_dir = backup_charts(charts_dir)

    cells = scores.get("cells", {})
    comparisons = scores.get("evaluation_comparisons", {})

    json_labels = []
    json_values = []
    for provider in ("local", "saas"):
        for mode in ("simple", "full"):
            json_labels.append(f"{provider} {mode}")
            json_values.append(cells.get(provider, {}).get(mode, {}).get("schema_valid_rate", 0.0))
    save_bar(
        charts_dir / "json_valid_rate.png",
        "JSON Schema Valid Rate by Provider / Prompt",
        json_labels,
        json_values,
        ylabel="Valid rate",
        ylim=(0, 1.1),
        note="SaaS full is 0 because no SaaS full smoke outputs exist yet.",
    )

    simple = comparisons.get("local_simple_vs_saas_simple", {})
    full = comparisons.get("local_full_vs_saas_full", {})
    save_bar(
        charts_dir / "score_comparison_table.png",
        "Official Same-Mode Score: Local vs SaaS",
        ["simple BLEU", "simple ROUGE-L", "simple METEOR", "simple CIDEr"],
        [
            simple.get("bleu4", 0.0),
            simple.get("rouge_l", 0.0),
            simple.get("meteor_lite", 0.0),
            simple.get("cider_lite", 0.0),
        ],
        ylabel="Score",
        ylim=(0, 0.5),
        note="Only simple is shown because SaaS full reference is missing.",
    )

    category_counts = split.get("smoke_category_counts", split.get("category_counts", {}))
    save_bar(
        charts_dir / "category_counts.png",
        "Smoke Image Count by Category",
        list(category_counts.keys()),
        [float(value) for value in category_counts.values()],
        ylabel="Image count",
    )

    save_bar(
        charts_dir / "matched_image_count.png",
        "Matched vs Unmatched Images for Official Scores",
        ["simple matched", "simple unmatched", "full matched", "full unmatched"],
        [
            simple.get("matched_image_count", 0),
            simple.get("unmatched_image_count", 0),
            full.get("matched_image_count", 0),
            full.get("unmatched_image_count", 0),
        ],
        ylabel="Image count",
        note="Full unmatched means missing SaaS full reference, not failed local full output.",
    )

    saas = record.get("generation", {}).get("saas", {})
    save_bar(
        charts_dir / "saas_api_errors.png",
        "Gemini API Request / Retry Status",
        ["requests", "api success", "retries", "rate limits", "quota errors"],
        [
            saas.get("request_count", 0),
            saas.get("api_success_count", 0),
            saas.get("retry_count", 0),
            saas.get("rate_limit_errors", 0),
            saas.get("quota_errors", 0),
        ],
        ylabel="Count",
    )

    save_bar(
        charts_dir / "same_mode_local_score_delta.png",
        "Prompt Delta Is Not Available Yet",
        ["BLEU", "ROUGE-L", "METEOR", "CIDEr"],
        [0.0, 0.0, 0.0, 0.0],
        ylabel="Full - simple delta",
        note="SaaS full reference is missing, so full-simple delta must not be interpreted.",
        ylim=(-0.1, 0.1),
    )

    by_category = scores.get("score_by_category", {}).get("simple", {})
    save_bar(
        charts_dir / "score_by_category.png",
        "CIDEr-lite by Category: local simple vs SaaS simple",
        list(by_category.keys()),
        [value.get("cider_lite", 0.0) for value in by_category.values()],
        ylabel="CIDEr-lite",
        ylim=(0, 0.35),
        note="Category scores are simple-mode only.",
    )

    generation = record.get("generation", {})
    latency_labels = []
    latency_values = []
    for provider in ("local", "saas"):
        by_mode = generation.get(provider, {}).get("by_mode", {})
        for mode, data in by_mode.items():
            count = data.get("output_count", 0)
            if count:
                latency_labels.append(f"{provider} {mode}")
                latency_values.append(data.get("total_duration_sec", 0.0) / count)
    save_bar(
        charts_dir / "latency_by_provider_mode.png",
        "Average Duration per Output by Provider / Prompt",
        latency_labels,
        latency_values,
        ylabel="Seconds per output",
        note="Local stats may include reruns; use as runtime context, not final benchmark.",
    )

    save_bar(
        charts_dir / "tokens_per_second_local.png",
        "Local Tokens/sec Not Recorded",
        ["simple", "full"],
        [0.0, 0.0],
        ylabel="Tokens/sec",
        note="Current local outputs do not record eval token counts; chart kept as placeholder.",
        ylim=(0, 1),
    )

    print(f"Backed up existing charts to {backup_dir}")
    print(f"Rewrote charts in {charts_dir}")


if __name__ == "__main__":
    main()
