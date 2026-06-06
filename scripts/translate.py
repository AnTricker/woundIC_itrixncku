import argparse
import json
import platform
import subprocess
import threading
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama


TRANSLATION_RULES = [
    "Output valid JSON only.",
    "Use Traditional Chinese.",
    "Translate only string values inside the caption JSON.",
    "Keep the original JSON keys and nested structure exactly.",
    "Keep null, booleans, numbers, and empty arrays unchanged.",
    "Preserve important English professional terms in parentheses, for example 紅斑（erythema）, 滲出液（exudate）, 壞死（necrosis）, 焦痂（eschar）.",
    "Do not add new medical claims.",
    "Do not diagnose; preserve visual-description-only wording.",
    "This translation is for human review only and must not be used for scoring.",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def response_value(response: Any, key: str, default: Any = 0) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def query_gpu_sample() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    rows = []
    for line in result.stdout.splitlines():
        try:
            vram_mb, gpu_percent = [float(part.strip()) for part in line.split(",")[:2]]
            rows.append((vram_mb, gpu_percent))
        except (ValueError, TypeError):
            continue
    if not rows:
        return None
    return {
        "runtime_vram_used_gb": sum(row[0] for row in rows) / 1024,
        "runtime_gpu_utilization_percent": sum(row[1] for row in rows) / len(rows),
    }


class RuntimeGpuMonitor:
    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        sample = query_gpu_sample()
        if sample:
            self.samples.append(sample)
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(self.interval_sec):
            sample = query_gpu_sample()
            if sample:
                self.samples.append(sample)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        sample = query_gpu_sample()
        if sample:
            self.samples.append(sample)
        if not self.samples:
            return {
                "runtime_gpu_monitoring": "unavailable",
                "runtime_gpu_sample_count": 0,
            }
        vram_values = [sample["runtime_vram_used_gb"] for sample in self.samples]
        gpu_values = [sample["runtime_gpu_utilization_percent"] for sample in self.samples]
        return {
            "runtime_gpu_monitoring": "recorded",
            "runtime_gpu_monitoring_source": "nvidia-smi",
            "runtime_gpu_sample_count": len(self.samples),
            "runtime_vram_used_gb_avg": sum(vram_values) / len(vram_values),
            "runtime_vram_used_gb_peak": max(vram_values),
            "runtime_gpu_utilization_percent_avg": sum(gpu_values) / len(gpu_values),
            "runtime_gpu_utilization_percent_peak": max(gpu_values),
        }


def output_json_paths(run_dir: Path) -> list[Path]:
    output_root = run_dir / "outputs"
    if not output_root.exists():
        return []
    return sorted(path for path in output_root.glob("*/*/*.json") if path.is_file())


def translated_path(run_dir: Path, source_path: Path) -> Path:
    relative = source_path.relative_to(run_dir / "outputs")
    return run_dir / "output_translation" / relative


def provider_mode_from_path(run_dir: Path, source_path: Path) -> tuple[str, str]:
    relative = source_path.relative_to(run_dir / "outputs")
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Output path must look like outputs/<provider>/<mode>/<file>: {source_path}")
    return parts[0], parts[1]


def translation_prompt(caption: dict[str, Any]) -> str:
    return (
        "You are translating wound image-captioning JSON for a Traditional Chinese "
        "research review.\n\n"
        "Rules:\n"
        + "\n".join(f"- {rule}" for rule in TRANSLATION_RULES)
        + "\n\nInput caption JSON:\n"
        + json.dumps(caption, ensure_ascii=False, indent=2)
        + "\n\nOutput JSON shape:\n"
        + "{\n"
        + '  "caption": { ... same structure as input caption, translated to zh-TW ... },\n'
        + '  "translation_notes": []\n'
        + "}\n"
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


def translate_caption(model: str, caption: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    gpu_monitor = RuntimeGpuMonitor()
    gpu_monitor.start()
    try:
        response = ollama.chat(
            model=model,
            format="json",
            messages=[{"role": "user", "content": translation_prompt(caption)}],
        )
    finally:
        gpu_metadata = gpu_monitor.stop()
    result = parse_json_text(response["message"]["content"])
    if "caption" not in result:
        result = {"caption": result, "translation_notes": ["wrapped raw model JSON"]}
    result.setdefault("translation_notes", [])
    input_tokens = int(response_value(response, "prompt_eval_count", 0) or 0)
    output_tokens = int(response_value(response, "eval_count", 0) or 0)
    metrics = {
        "total_duration_sec": time.monotonic() - started,
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "total_token_count": input_tokens + output_tokens,
        "prefill_duration_sec": float(response_value(response, "prompt_eval_duration", 0) or 0)
        / 1_000_000_000,
        "decode_duration_sec": float(response_value(response, "eval_duration", 0) or 0)
        / 1_000_000_000,
        **gpu_metadata,
    }
    return result, metrics


def translated_bundle(
    *,
    run_dir: Path,
    source_path: Path,
    output_path: Path,
    source_bundle: dict[str, Any],
    model: str,
    translated_caption: dict[str, Any],
    notes: list[Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    provider, mode = provider_mode_from_path(run_dir, source_path)
    bundle = deepcopy(source_bundle)
    metadata = dict(bundle.get("metadata", {}))
    source_metadata = dict(source_bundle.get("metadata", {}))
    metadata["translation"] = {
        "created_at": now_iso(),
        "script": "scripts/translate.py",
        "status": "success",
        "run_dir": str(run_dir),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "translation_model": model,
        "translation_scope": "caption",
        "translation_rules": TRANSLATION_RULES,
        "provider": provider,
        "prompt_mode": mode,
        "source_provider": source_metadata.get("provider"),
        "source_saas_provider": source_metadata.get("saas_provider"),
        "source_model": source_metadata.get("model"),
        "source_mode": source_metadata.get("mode"),
        "source_created_at": source_metadata.get("created_at"),
        "source_image": source_metadata.get("image"),
        "source_image_name": source_metadata.get("image_name"),
        "source_raw_category": source_metadata.get("raw_category"),
        "source_target_category": source_metadata.get("target_category"),
        "original_outputs_modified": False,
        **metrics,
    }
    bundle["metadata"] = metadata
    bundle["caption"] = translated_caption
    bundle["translation_notes"] = notes
    return bundle


def fmt_int(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else "not recorded"


def fmt_seconds(value: Any) -> str:
    return f"{float(value):.1f} sec" if isinstance(value, (int, float)) else "not recorded"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "/").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def metric_total(records: list[dict[str, Any]], key: str) -> float | int | None:
    values = [record.get(key) for record in records if isinstance(record.get(key), (int, float))]
    return sum(values) if values else None


def runtime_summary(records: list[dict[str, Any]], prefix: str, suffix: str) -> str:
    values = [
        record.get(f"{prefix}_{suffix}")
        for record in records
        if isinstance(record.get(f"{prefix}_{suffix}"), (int, float))
    ]
    if not values:
        return "not recorded"
    if suffix == "avg":
        return f"{sum(values) / len(values):.1f}"
    return f"{max(values):.1f}"


def runtime_text(records: list[dict[str, Any]], prefix: str, unit: str) -> str:
    avg = runtime_summary(records, prefix, "avg")
    peak = runtime_summary(records, prefix, "peak")
    if avg == "not recorded" and peak == "not recorded":
        return "not recorded"
    return f"avg {avg}{unit}; peak {peak}{unit}"


def write_translation_summary(run_dir: Path, output_paths: list[Path]) -> Path:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in output_paths:
        bundle = read_json(path)
        translation = bundle.get("metadata", {}).get("translation", {})
        key = (
            str(translation.get("provider", "unknown")),
            str(translation.get("prompt_mode", "unknown")),
            str(translation.get("source_model", "unknown")),
            str(translation.get("translation_model", "unknown")),
        )
        groups[key].append(translation)

    rows = []
    for (provider, mode, source_model, translation_model), records in sorted(groups.items()):
        rows.append(
            [
                run_dir.name,
                provider,
                source_model,
                mode,
                translation_model,
                len(records),
                fmt_int(metric_total(records, "input_token_count")),
                fmt_int(metric_total(records, "output_token_count")),
                fmt_seconds(metric_total(records, "total_duration_sec")),
                fmt_seconds(metric_total(records, "prefill_duration_sec")),
                fmt_seconds(metric_total(records, "decode_duration_sec")),
                runtime_text(records, "runtime_vram_used_gb", " GB"),
                runtime_text(records, "runtime_gpu_utilization_percent", "%"),
                "success",
                "human review only; not used for scoring",
            ]
        )

    field_rows = [
        ["run id", "被翻譯輸出所屬的 run directory 名稱。"],
        ["source provider/model/mode", "被翻譯的原始 caption 來源。"],
        ["translation model", "執行繁中翻譯的 local Ollama model。"],
        ["translated count", "已產生翻譯 JSON 的檔案數。"],
        ["input/output token", "翻譯模型 Ollama `prompt_eval_count` / `eval_count` 加總。"],
        ["duration time", "每筆翻譯 wall-clock duration 加總。"],
        ["prefill/decode time", "Ollama `prompt_eval_duration` / `eval_duration` 加總。"],
        ["runtime VRAM/GPU usage", "翻譯模型執行期間由 `nvidia-smi` 取樣的 avg / peak。"],
        ["usage", "翻譯只供 human review；不回寫原始 output，不參與 scoring。"],
    ]
    summary_path = run_dir / "output_translation" / f"{run_dir.name}_translation_summary.md"
    content = "\n".join(
        [
            f"# {run_dir.name} Translation Summary",
            "",
            f"Source run: `{run_dir}`",
            "",
            f"Translation output root: `{run_dir / 'output_translation'}`",
            "",
            "## 1. 欄位說明",
            "",
            table(["欄位", "說明"], field_rows),
            "",
            "## 2. 已記錄翻譯資料總表",
            "",
            table(
                [
                    "run id",
                    "source provider",
                    "source model",
                    "source prompt type",
                    "translation model",
                    "translated count",
                    "input token",
                    "output token",
                    "duration time",
                    "prefill time",
                    "decode time",
                    "runtime VRAM usage",
                    "runtime GPU usage",
                    "status",
                    "usage",
                ],
                rows,
            ),
            "",
            "## 3. 輸出規則",
            "",
            "- 翻譯範圍：每筆原始結果 JSON 的 `caption`。",
            "- 翻譯檔案：鏡像寫入 `output_translation/<provider>/<mode>/`。",
            "- 原始 `outputs/` 不會被修改。",
            "- 翻譯內容只供人工檢閱，不參與任何 score 計算。",
            "",
        ]
    )
    summary_path.write_text(content, encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate every output JSON caption in a run into mirrored zh-TW files."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="gemma4:26b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    paths = output_json_paths(run_dir)
    if not paths:
        raise FileNotFoundError(f"No output JSON files found under {run_dir / 'outputs'}")

    translated_paths = []
    for source_path in paths:
        output_path = translated_path(run_dir, source_path)
        print(f"Translating {source_path} -> {output_path}")
        source_bundle = read_json(source_path)
        caption = source_bundle.get("caption")
        if not isinstance(caption, dict):
            raise ValueError(f"caption is missing or is not a JSON object: {source_path}")
        translation, metrics = translate_caption(args.model, caption)
        write_json(
            output_path,
            translated_bundle(
                run_dir=run_dir,
                source_path=source_path,
                output_path=output_path,
                source_bundle=source_bundle,
                model=args.model,
                translated_caption=translation["caption"],
                notes=translation.get("translation_notes", []),
                metrics=metrics,
            ),
        )
        translated_paths.append(output_path)
    summary_path = write_translation_summary(run_dir, translated_paths)
    print(f"Saved translated files under {run_dir / 'output_translation'}")
    print(f"Saved translation summary: {summary_path}")


if __name__ == "__main__":
    main()
