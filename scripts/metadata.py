import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROVIDERS = ("local", "saas")
MODES = ("simple", "full")
SCORE_COLUMNS = ("BLEU-4", "ROUGE-L", "METEOR-lite", "CIDEr-lite")
MODEL_SIZE_HINTS = {
    "gemma4:26b": "Gemma 4 26B A4B; 4B active/token",
    "gemma4:e2b": "Gemma 4 E2B effective params",
    "gemma3:27b": "Gemma 3 27B params",
    "qwen3.5": "not recorded",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_under_run_dir(path: Path, run_dir: Path, label: str) -> Path:
    resolved_path = path.resolve()
    resolved_run_dir = run_dir.resolve()
    try:
        resolved_path.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside --run-dir. Got {path}; run_dir={run_dir}"
        ) from exc
    return path


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "not recorded"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_seconds(value: Any) -> str:
    if value in (None, ""):
        return "not recorded"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds == 0:
        return "0 sec"
    minutes = seconds / 60
    return f"{seconds:.1f} sec ({minutes:.1f} min)"


def fmt_rate(value: Any) -> str:
    if value is None:
        return "not recorded"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_percent(value: Any) -> str:
    if value in (None, ""):
        return "not recorded"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def safe_cell(value: Any) -> str:
    text = fmt(value)
    return text.replace("|", "/").replace("\n", "<br>")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(safe_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def read_outputs(run_dir: Path, provider: str, mode: str) -> list[dict[str, Any]]:
    output_dir = run_dir / "outputs" / provider / mode
    if not output_dir.exists():
        return []

    outputs: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            outputs.append(read_json(path))
        except (OSError, json.JSONDecodeError):
            outputs.append({"metadata": {"read_error": str(path)}})
    return outputs


def nested_get(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metadata_values(outputs: list[dict[str, Any]], key: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for output in outputs:
        value = output.get("metadata", {}).get(key)
        if value not in (None, ""):
            values[str(value)] += 1
    return values


def most_common_value(outputs: list[dict[str, Any]], key: str, fallback: Any = None) -> Any:
    values = metadata_values(outputs, key)
    if not values:
        return fallback
    return values.most_common(1)[0][0]


def output_duration(outputs: list[dict[str, Any]]) -> float | None:
    durations = [
        output.get("metadata", {}).get("total_duration_sec")
        for output in outputs
        if isinstance(output.get("metadata", {}).get("total_duration_sec"), (int, float))
    ]
    if not durations:
        return None
    return float(sum(durations))


def schema_quality(scores: dict[str, Any], provider: str, mode: str) -> str:
    cell = scores.get("cells", {}).get(provider, {}).get(mode, {})
    if not cell:
        return "not recorded"
    valid = fmt_rate(cell.get("schema_valid_rate"))
    completion = fmt_rate(cell.get("required_field_completion_rate"))
    count = cell.get("count")
    if count == 0:
        return "no output to validate"
    return f"JSON schema valid {valid}; field completion {completion}"


def score_text(scores: dict[str, Any], provider: str, mode: str) -> str:
    if provider == "saas":
        if mode == "simple":
            return "reference baseline; no candidate score"
        return "not scored; SaaS full reference missing"

    comparison_name = f"local_{mode}_vs_saas_{mode}"
    comparison = scores.get("evaluation_comparisons", {}).get(comparison_name, {})
    matched = comparison.get("matched_image_count", 0)
    if matched == 0:
        return "not evaluable; matched image count = 0"

    return (
        f"{SCORE_COLUMNS[0]} {fmt(comparison.get('bleu4'))}; "
        f"{SCORE_COLUMNS[1]} {fmt(comparison.get('rouge_l'))}; "
        f"{SCORE_COLUMNS[2]} {fmt(comparison.get('meteor_lite'))}; "
        f"{SCORE_COLUMNS[3]} {fmt(comparison.get('cider_lite'))}"
    )


def token_counts(record: dict[str, Any], provider: str, mode: str, output_count: int) -> tuple[str, str]:
    generation = record.get("generation", {}).get(provider, {})
    if output_count == 0:
        return "0", "0"

    by_mode = generation.get("by_mode", {}).get(mode, {})
    input_tokens = by_mode.get("input_token_count")
    output_tokens = by_mode.get("output_token_count")
    if input_tokens is None and provider == "saas":
        input_tokens = generation.get("input_token_count")
    if output_tokens is None and provider == "saas":
        output_tokens = generation.get("output_token_count")
    return fmt_int(input_tokens), fmt_int(output_tokens)


def local_duration_metric(
    provider: str,
    provider_generation: dict[str, Any],
    mode: str,
    key: str,
) -> str:
    if provider == "saas":
        return "not applicable (SaaS runs on provider infrastructure)"
    value = provider_generation.get("by_mode", {}).get(mode, {}).get(key)
    return fmt_seconds(value)


def platform_text(provider: str, platform: dict[str, Any]) -> str:
    if provider == "saas":
        return "not applicable (SaaS runs on provider infrastructure)"
    machine = platform.get("machine", "unknown machine")
    os_name = platform.get("os", "unknown OS")
    return f"server ({machine}, {os_name})"


def gpu_capacity_text(provider: str, platform: dict[str, Any]) -> str:
    if provider == "saas":
        return "not applicable (SaaS runs on provider infrastructure)"
    gpu = platform.get("gpu", "unknown GPU")
    vram = platform.get("vram_gb")
    if vram in (None, 0, "0"):
        return f"{gpu}; total VRAM not recorded"
    return f"{gpu}; {fmt(vram, 1)} GB"


def model_size_text(model: Any) -> str:
    if not model:
        return "not recorded"
    normalized = str(model).lower()
    if normalized in MODEL_SIZE_HINTS:
        return MODEL_SIZE_HINTS[normalized]
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", normalized)
    if match:
        return f"{match.group(1)}B params (inferred from model name)"
    if "gemini" in normalized:
        return "not published by Google / provider managed"
    return "not recorded"


def output_runtime_summary(outputs: list[dict[str, Any]], prefix: str) -> tuple[float | None, float | None]:
    weighted_sum = 0.0
    sample_count = 0
    peaks: list[float] = []
    for output in outputs:
        metadata = output.get("metadata", {})
        avg = metadata.get(f"{prefix}_avg")
        count = metadata.get("runtime_gpu_sample_count", 0)
        peak = metadata.get(f"{prefix}_peak")
        if isinstance(avg, (int, float)) and isinstance(count, int) and count > 0:
            weighted_sum += float(avg) * count
            sample_count += count
        if isinstance(peak, (int, float)):
            peaks.append(float(peak))
    avg_value = weighted_sum / sample_count if sample_count else None
    peak_value = max(peaks) if peaks else None
    return avg_value, peak_value


def aggregate_runtime_summary(
    provider_generation: dict[str, Any],
    mode: str,
    prefix: str,
) -> tuple[float | None, float | None]:
    mode_stats = provider_generation.get("by_mode", {}).get(mode, {})
    sample_sum = mode_stats.get(f"{prefix}_sample_sum")
    sample_count = mode_stats.get(f"{prefix}_sample_count")
    peak = mode_stats.get(f"{prefix}_peak")
    avg = None
    if isinstance(sample_sum, (int, float)) and isinstance(sample_count, int) and sample_count > 0:
        avg = float(sample_sum) / sample_count
    return avg, peak if isinstance(peak, (int, float)) and peak > 0 else None


def runtime_vram_text(
    provider: str,
    provider_generation: dict[str, Any],
    mode: str,
    outputs: list[dict[str, Any]],
) -> str:
    if provider == "saas":
        return "not applicable (SaaS runs on provider infrastructure)"

    avg, peak = output_runtime_summary(outputs, "runtime_vram_used_gb")
    if avg is None and peak is None:
        avg, peak = aggregate_runtime_summary(provider_generation, mode, "runtime_vram_used_gb")
    if avg is None and peak is None:
        return "not recorded; cannot be recovered after run"
    parts = []
    if avg is not None:
        parts.append(f"avg {float(avg):.1f} GB")
    if peak is not None:
        parts.append(f"peak {float(peak):.1f} GB")
    return "; ".join(parts)


def runtime_gpu_usage_text(
    provider: str,
    provider_generation: dict[str, Any],
    mode: str,
    outputs: list[dict[str, Any]],
) -> str:
    if provider == "saas":
        return "not applicable (SaaS runs on provider infrastructure)"

    avg, peak = output_runtime_summary(outputs, "runtime_gpu_utilization_percent")
    if avg is None and peak is None:
        avg, peak = aggregate_runtime_summary(
            provider_generation, mode, "runtime_gpu_utilization_percent"
        )
    if avg is None and peak is None:
        return "not recorded; cannot be calculated from existing output"
    parts = []
    if avg is not None:
        parts.append(f"avg {fmt_percent(avg)}")
    if peak is not None:
        parts.append(f"peak {fmt_percent(peak)}")
    return "; ".join(parts)


def dataset_split_text(split: dict[str, Any]) -> str:
    smoke_percent = split.get("smoke_percent")
    seed = split.get("seed")
    smoke_count = split.get("counts", {}).get("smoke")
    formal_count = split.get("counts", {}).get("formal")
    return f"small test / smoke {smoke_percent}% (seed {seed}; {smoke_count}/{formal_count} images)"


def provider_notes(
    record: dict[str, Any],
    scores: dict[str, Any],
    provider: str,
    mode: str,
    output_count: int,
) -> str:
    generation = record.get("generation", {}).get(provider, {})
    notes: list[str] = []

    if provider == "saas":
        notes.append(
            "request/quota/token tracked: "
            f"requests {fmt_int(generation.get('request_count'))}, "
            f"success {fmt_int(generation.get('api_success_count'))}, "
            f"retry {fmt_int(generation.get('retry_count'))}, "
            f"rate-limit {fmt_int(generation.get('rate_limit_errors'))}, "
            f"quota {fmt_int(generation.get('quota_errors'))}"
        )
        if mode == "full" and output_count == 0:
            notes.append("not run yet; same-mode full score cannot be calculated")
    else:
        if generation.get("by_mode", {}).get(mode, {}).get("input_token_count") is None:
            notes.append("local token / prefill / decode metrics were not recorded in this older run")
        else:
            notes.append("local token / prefill / decode metrics recorded from Ollama runtime response")
        if mode == "full":
            notes.append("SaaS full reference missing, so score is not official")

    gate = scores.get("baseline_quality_gate", {})
    if mode == "simple" and gate.get("approved_for_scoring") is False:
        notes.append("baseline manual review pending; score status is exploratory")
    return "; ".join(notes)


def build_metadata_rows(
    run_dir: Path,
    record: dict[str, Any],
    scores: dict[str, Any],
    split: dict[str, Any],
) -> list[list[Any]]:
    metadata = record.get("run_metadata", {})
    generation = record.get("generation", {})
    platform = metadata.get("platform", {})
    run_id = Path(metadata.get("run_dir", run_dir)).name
    rows: list[list[Any]] = []

    for provider in PROVIDERS:
        for mode in MODES:
            outputs = read_outputs(run_dir, provider, mode)
            output_count = len(outputs)
            provider_generation = generation.get(provider, {})
            model = most_common_value(outputs, "model", provider_generation.get("model"))
            duration = output_duration(outputs)
            if duration is None:
                duration = provider_generation.get("by_mode", {}).get(mode, {}).get("total_duration_sec")
            input_tokens, output_tokens = token_counts(record, provider, mode, output_count)

            rows.append(
                [
                    run_id,
                    dataset_split_text(split),
                    output_count,
                    "SaaS" if provider == "saas" else "local",
                    model,
                    model_size_text(model),
                    mode,
                    platform_text(provider, platform),
                    gpu_capacity_text(provider, platform),
                    runtime_vram_text(provider, provider_generation, mode, outputs),
                    runtime_gpu_usage_text(provider, provider_generation, mode, outputs),
                    input_tokens,
                    output_tokens,
                    fmt_seconds(duration),
                    local_duration_metric(provider, provider_generation, mode, "prefill_duration_sec"),
                    local_duration_metric(provider, provider_generation, mode, "decode_duration_sec"),
                    schema_quality(scores, provider, mode),
                    score_text(scores, provider, mode),
                    provider_notes(record, scores, provider, mode, output_count),
                ]
            )
    return rows


def build_summary(run_dir: Path) -> str:
    split = read_json(run_dir / "split_summary.json")
    record = read_json(run_dir / "experiment_record.json")
    scores = read_json(run_dir / "scores.json")

    metadata = record.get("run_metadata", {})
    smoke_counts = metadata.get("smoke_category_counts") or split.get("smoke_category_counts", {})
    formal_counts = metadata.get("formal_category_counts") or split.get("formal_category_counts", {})

    field_rows = [
        ["run id", "每次實驗的識別名稱；此報告使用 run directory 名稱。"],
        ["dataset split", "small test / smoke 或 full；目前為 10% smoke test。"],
        ["image count", "該 provider + prompt type 實際產出的 JSON 檔案數量。"],
        ["model type", "SaaS 或 local。"],
        ["model", "Gemini / Gemma / Qwen 等實際記錄到輸出檔或 experiment record 的模型名稱。"],
        ["model size / params", "模型參數量；Gemma 使用 Google 文件，Gemini 未公開則不推測。"],
        ["prompt type", "simple / full / chain-style；依指定 run 實際已有的輸出列出。"],
        ["platform", "server / local / VM；目前以 experiment_record 的 machine 與 OS 表示。"],
        ["GPU capacity / total VRAM", "硬體規格，不是 runtime usage；pipeline 目前記錄 GPU 型號與總 VRAM。"],
        ["runtime VRAM usage", "模型實際執行時使用的 VRAM；SaaS 不適用；新 local run 由 pipeline 執行時記錄。"],
        ["runtime GPU usage", "模型實際執行時 GPU utilization；SaaS 不適用；新 local run 由 pipeline 執行時記錄。"],
        ["input token", "SaaS 使用 provider usage；新 local run 使用 Ollama `prompt_eval_count`。"],
        ["output token", "SaaS 使用 provider usage；新 local run 使用 Ollama `eval_count`。"],
        ["duration time", "該列輸出檔 metadata 的 total_duration_sec 加總。"],
        ["prefill time", "新 local run 使用 Ollama `prompt_eval_duration` 記錄；SaaS 不適用。"],
        ["decode time", "新 local run 使用 Ollama `eval_duration` 記錄；SaaS 不適用。"],
        ["output format quality", "JSON schema valid rate 與 required field completion rate。"],
        ["score 1-4", "四大文字相似度分數：BLEU-4、ROUGE-L、METEOR-lite、CIDEr-lite。"],
        ["備註", "錯誤、quota、異常狀況、manual review 狀態。"],
    ]

    lines = [
        f"# {run_dir.name} Record Summary",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This file reorganizes the specified run metadata into report-ready tables.",
        "",
        "## 1. 欄位說明",
        "",
        table(["欄位", "說明"], field_rows),
        "",
        "## 2. 已記錄實驗資料總表",
        "",
        table(
            [
                "run id",
                "dataset split",
                "image count",
                "model type",
                "model",
                "model size / params",
                "prompt type",
                "platform",
                "GPU capacity / total VRAM",
                "runtime VRAM usage",
                "runtime GPU usage",
                "input token",
                "output token",
                "duration time",
                "prefill time",
                "decode time",
                "output format quality",
                "score 1-4",
                "備註",
            ],
            build_metadata_rows(run_dir, record, scores, split),
        ),
        "",
        "## 3. Model Size Source Notes",
        "",
        table(
            ["Model family", "Report rule"],
            [
                [
                    "Gemma 4",
                    "Use Google Gemma docs: E2B, E4B, 31B, and 26B A4B; A4B means 4B active parameters per token.",
                ],
                [
                    "Gemini",
                    "Google Gemini API docs list model families/capabilities but do not publish parameter counts, so this report records provider managed / not published.",
                ],
            ],
        ),
        "",
        "## 4. Dataset Split / Category Count",
        "",
        table(
            ["Scope", "Image Count"],
            [
                ["small test / smoke", split.get("counts", {}).get("smoke")],
                ["full / formal", split.get("counts", {}).get("formal")],
            ],
        ),
        "",
        table(
            ["Category", "Smoke Count", "Full/Formal Count"],
            [
                [category, smoke_counts.get(category, 0), formal_counts.get(category, 0)]
                for category in sorted(set(smoke_counts) | set(formal_counts))
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run-specific report-ready record tables.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <run-dir>/<run-name>_record_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / f"{run_dir.name}_record_summary.md"
    ensure_under_run_dir(output, run_dir, "--output")
    content = build_summary(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
