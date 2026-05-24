import argparse
import base64
import concurrent.futures
import csv
import json
import math
import mimetypes
import os
import platform
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import ollama
from jsonschema import Draft202012Validator

import main as prompt_runtime


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
SCHEMA_PATH = BASE_DIR / "wound_schema.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODES = ("simple", "full")
PROVIDERS = ("saas", "local")
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ENV_FILE = BASE_DIR / ".env"
DEFAULT_EXCLUDE_DIRS = "MM-SkinQA"
DEFAULT_QUALITY_SCHEMA_MIN = 0.95
DEFAULT_QUALITY_FIELD_MIN = 0.90
DEFAULT_SAAS_LIMITS = {
    "gemini-2.5-flash": {
        "rpm": 5,
        "rpd": 20,
        "tpm": 250_000,
    },
    "gemini-3.1-flash-lite": {
        "rpm": 15,
        "rpd": 500,
        "tpm": 250_000,
    },
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def experiment_record_path(run_dir: Path) -> Path:
    return run_dir / "experiment_record.json"


def empty_generation_stats(provider: str, model: str, saas_provider: str | None) -> dict[str, Any]:
    return {
        "provider": provider,
        "saas_provider": saas_provider,
        "model": model,
        "request_count": 0,
        "api_success_count": 0,
        "api_failure_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "retry_count": 0,
        "rate_limit_errors": 0,
        "quota_errors": 0,
        "network_errors": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "total_token_count": 0,
        "input_text_token_count": 0,
        "input_image_token_count": 0,
        "total_duration_sec": 0.0,
        "by_mode": {
            mode: {
                "output_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "schema_valid_count": 0,
                "total_duration_sec": 0.0,
            }
            for mode in MODES
        },
    }


def merge_generation_stats(
    existing: dict[str, Any],
    provider: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    provider_stats = dict(merged.get(provider, {}))
    for key, value in stats.items():
        if key == "by_mode":
            by_mode = dict(provider_stats.get("by_mode", {}))
            for mode, mode_stats in value.items():
                current = dict(by_mode.get(mode, {}))
                for stat_key, stat_value in mode_stats.items():
                    if isinstance(stat_value, (int, float)):
                        current[stat_key] = current.get(stat_key, 0) + stat_value
                    else:
                        current[stat_key] = stat_value
                by_mode[mode] = current
            provider_stats["by_mode"] = by_mode
        elif isinstance(value, (int, float)):
            provider_stats[key] = provider_stats.get(key, 0) + value
        else:
            provider_stats[key] = value
    merged[provider] = provider_stats
    return merged


def load_split_summary(run_dir: Path) -> dict[str, Any]:
    return read_optional_json(run_dir / "split_summary.json")


def saas_limits_for_model(model: str) -> dict[str, int | str]:
    normalized = model.lower()
    for key, limits in DEFAULT_SAAS_LIMITS.items():
        if key in normalized:
            return {"model": key, **limits}
    return {
        "model": model,
        "rpm": 0,
        "rpd": 0,
        "tpm": 0,
    }


def update_token_stats(stats: dict[str, Any] | None, usage: dict[str, Any]) -> None:
    if stats is None or not usage:
        return
    stats["input_token_count"] += usage.get("promptTokenCount", 0)
    stats["output_token_count"] += usage.get("candidatesTokenCount", 0)
    stats["total_token_count"] += usage.get("totalTokenCount", 0)
    for detail in usage.get("promptTokensDetails", []):
        modality = str(detail.get("modality", "")).lower()
        token_count = detail.get("tokenCount", 0)
        if modality == "text":
            stats["input_text_token_count"] += token_count
        elif modality == "image":
            stats["input_image_token_count"] += token_count


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 1, ""
    return completed.returncode, completed.stdout.strip()


def detect_gpu_info() -> dict[str, Any]:
    code, output = run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if code == 0 and output:
        names = []
        vram_values = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if parts:
                names.append(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                vram_values.append(round(int(parts[1]) / 1024, 1))
        return {
            "gpu": " + ".join(names) if names else "unknown",
            "vram_gb": sum(vram_values) if vram_values else 0,
            "gpu_detection_source": "nvidia-smi",
        }

    code, output = run_command(["lspci"])
    if code == 0 and output:
        gpu_lines = [
            line
            for line in output.splitlines()
            if re.search(r"nvidia|vga|3d|display", line, flags=re.IGNORECASE)
        ]
        if gpu_lines:
            gpu = "; ".join(gpu_lines)
            if "Device 2684" in gpu:
                gpu = "NVIDIA GeForce RTX 4090"
            return {
                "gpu": gpu,
                "vram_gb": 24 if "4090" in gpu else 0,
                "gpu_detection_source": "lspci",
            }

    machine = platform.node() or ""
    if "4090" in machine:
        return {
            "gpu": "NVIDIA GeForce RTX 4090",
            "vram_gb": 24,
            "gpu_detection_source": "machine-name-inference",
        }
    return {"gpu": "unknown", "vram_gb": 0, "gpu_detection_source": "unavailable"}


def detect_ram_gb() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3))
    except (AttributeError, OSError, ValueError):
        return 0


def manifest_category_counts_by_scope(run_dir: Path) -> dict[str, dict[str, int]]:
    try:
        rows = read_manifest(run_dir)
    except FileNotFoundError:
        return {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = row.get("category", "unknown")
        for scope in row.get("scopes", []):
            counts[scope][category] += 1
    return {scope: dict(counter) for scope, counter in sorted(counts.items())}


def build_run_metadata(run_dir: Path, score_scopes: set[str] | None = None) -> dict[str, Any]:
    split_summary = load_split_summary(run_dir)
    counts = split_summary.get("counts", {})
    category_counts_by_scope = split_summary.get("category_counts_by_scope") or (
        manifest_category_counts_by_scope(run_dir)
    )
    selected_sample_count = (
        sum(counts.get(scope, 0) for scope in score_scopes)
        if score_scopes is not None
        else counts.get("formal", split_summary.get("total", 0))
    )
    gpu_info = detect_gpu_info()
    return {
        "run_dir": str(run_dir),
        "timestamp": now_iso(),
        "scope": sorted(score_scopes) if score_scopes is not None else None,
        "dataset_root": split_summary.get("image_dir"),
        "smoke_percent": split_summary.get("smoke_percent"),
        "seed": split_summary.get("seed"),
        "excluded_dirs": split_summary.get("exclude_dirs", []),
        "sample_count": selected_sample_count,
        "total_image_count": split_summary.get("total", counts.get("formal", 0)),
        "scope_counts": counts,
        "category_counts": split_summary.get("category_counts", {}),
        "category_counts_by_scope": category_counts_by_scope,
        "smoke_category_counts": category_counts_by_scope.get("smoke", {}),
        "formal_category_counts": category_counts_by_scope.get(
            "formal",
            split_summary.get("category_counts", {}),
        ),
        "platform": {
            "machine": platform.node() or "unknown",
            "os": platform.system() or "unknown",
            "hardware": platform.machine() or "unknown",
            "gpu": gpu_info["gpu"],
            "vram_gb": gpu_info["vram_gb"],
            "gpu_detection_source": gpu_info["gpu_detection_source"],
            "ram_gb": detect_ram_gb(),
        },
        "metric_definitions": {
            "request_count": "SaaS/API model call attempts. Simple normally uses 1 per image; full normally uses 3 per image.",
            "api_success_count": "Successful SaaS/API model calls.",
            "api_failure_count": "SaaS/API model calls that still failed after retry policy.",
            "success_count": "Completed output bundles/files at provider+mode job level.",
            "failure_count": "Failed provider+mode output jobs.",
            "retry_count": "Extra attempts after a retryable SaaS/API error.",
        },
    }


def update_experiment_record(run_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    path = experiment_record_path(run_dir)
    record = read_optional_json(path)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(record.get(key), dict):
            record[key] = {**record[key], **value}
        else:
            record[key] = value
    write_json(path, record)
    return record


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def render_template(path: Path, **values: object) -> str:
    rendered = path.read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def image_to_inline_data(image_path: Path) -> dict[str, str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"mime_type": mime_type, "data": data}


def parse_json_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def iter_images(image_dir: Path, exclude_dirs: set[str] | None = None) -> list[Path]:
    if not image_dir.exists():
        return []
    normalized_excludes = {normalize_category(item) for item in (exclude_dirs or set())}
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(normalize_category(part) in normalized_excludes for part in path.parts)
    )


def normalize_category(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_dataset_category(image_path: Path, image_dir: Path) -> str:
    try:
        relative = image_path.relative_to(image_dir)
    except ValueError:
        return prompt_runtime.infer_category(image_path)
    if len(relative.parts) > 1:
        return normalize_category(relative.parts[0])
    return prompt_runtime.infer_category(image_path)


def is_supported_category(category: str) -> bool:
    return category in prompt_runtime.CATEGORY_PROMPTS


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.jsonl"


def read_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path(run_dir).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    run_dir = Path(args.run_dir)
    if not 0 <= args.smoke_percent <= 100:
        raise ValueError("--smoke-percent must be between 0 and 100")

    exclude_dirs = parse_csv_set(args.exclude_dirs)
    images = iter_images(image_dir, exclude_dirs)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rng = random.Random(args.seed)
    by_category: dict[str, list[Path]] = defaultdict(list)
    excluded_rows: list[dict[str, Any]] = []
    for image in images:
        category = infer_dataset_category(image, image_dir)
        if is_supported_category(category):
            by_category[category].append(image)
        else:
            excluded_rows.append(
                {
                    "image": str(image),
                    "image_name": image.name,
                    "category": category,
                    "reason": "unsupported_category_prompt",
                }
            )

    supported_total = sum(len(category_images) for category_images in by_category.values())
    if not supported_total:
        raise ValueError(
            "No supported wound categories found. Expected folder names or filename "
            f"prefixes matching: {', '.join(sorted(prompt_runtime.CATEGORY_PROMPTS))}"
        )

    total_smoke_count = round(supported_total * args.smoke_percent / 100)
    smoke_counts: dict[str, int] = {}
    fractional_parts: list[tuple[float, str]] = []
    assigned = 0
    for category, category_images in sorted(by_category.items()):
        exact = len(category_images) * args.smoke_percent / 100
        count = math.floor(exact)
        smoke_counts[category] = count
        assigned += count
        fractional_parts.append((exact - count, category))

    remaining = total_smoke_count - assigned
    for _, category in sorted(fractional_parts, reverse=True)[:remaining]:
        smoke_counts[category] += 1

    rows: list[dict[str, Any]] = []
    for category, category_images in sorted(by_category.items()):
        shuffled = category_images[:]
        rng.shuffle(shuffled)
        smoke_count = smoke_counts[category]

        for index, image in enumerate(shuffled):
            scopes = ["formal"]
            if index < smoke_count:
                scopes.append("smoke")
            rows.append(
                {
                    "image": str(image),
                    "image_name": image.name,
                    "category": category,
                    "scopes": scopes,
                }
            )

    rows.sort(key=lambda row: (row["category"], row["image_name"]))
    write_manifest(run_dir, rows)
    excluded_path = run_dir / "excluded_manifest.jsonl"
    write_jsonl(excluded_path, excluded_rows)
    summary = Counter(scope for row in rows for scope in row["scopes"])
    category_counts = {
        category: len(category_images)
        for category, category_images in sorted(by_category.items())
    }
    category_counts_by_scope = {
        scope: dict(
            Counter(row["category"] for row in rows if scope in row.get("scopes", []))
        )
        for scope in ("formal", "smoke")
    }
    excluded_counts = Counter(row["category"] for row in excluded_rows)
    write_json(
        run_dir / "split_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_dir": str(image_dir),
            "smoke_percent": args.smoke_percent,
            "seed": args.seed,
            "exclude_dirs": sorted(exclude_dirs),
            "counts": dict(summary),
            "category_counts": category_counts,
            "category_counts_by_scope": category_counts_by_scope,
            "formal_category_counts": category_counts_by_scope["formal"],
            "smoke_category_counts": category_counts_by_scope["smoke"],
            "excluded_counts": dict(excluded_counts),
            "total": len(rows),
            "excluded_total": len(excluded_rows),
        },
    )
    print(f"Saved manifest: {manifest_path(run_dir)}")
    print(f"Saved excluded manifest: {excluded_path}")
    print(f"Scope counts: {dict(summary)}")
    if exclude_dirs:
        print(f"Excluded directories: {sorted(exclude_dirs)}")
    if excluded_rows:
        print(f"Excluded unsupported categories: {dict(excluded_counts)}")


def output_path(run_dir: Path, provider: str, mode: str, image_path: Path) -> Path:
    return run_dir / "outputs" / provider / mode / f"{image_path.stem}.json"


def load_category_guidance(category: str) -> tuple[str, str]:
    return prompt_runtime.load_category_prompt(category)


def build_caption_prompt(category: str, mode: str) -> str:
    target_category, guidance = load_category_guidance(category)
    if mode == "simple":
        template = PROMPT_DIR / "simple_caption_template.md"
        return render_template(
            template,
            category=target_category,
            specific_instructions=guidance,
        )
    if mode == "full":
        return prompt_runtime.render_prompt(
            "caption",
            category=target_category,
            specific_instructions=guidance,
        )
    raise ValueError(f"Unknown prompt mode: {mode}")


def build_vqa_prompt(category: str) -> str:
    target_category, guidance = load_category_guidance(category)
    return prompt_runtime.render_prompt(
        "vqa",
        category=target_category,
        specific_instructions=guidance,
    )


def build_self_check_prompt(category: str, caption: dict[str, Any], vqa: dict[str, Any]) -> str:
    target_category, guidance = load_category_guidance(category)
    return prompt_runtime.render_prompt(
        "self_check",
        category=target_category,
        specific_instructions=guidance,
        caption_json=json.dumps(caption, ensure_ascii=False, indent=2),
        vqa_json=json.dumps(vqa, ensure_ascii=False, indent=2),
    )


def call_ollama_json(model: str, prompt: str, image_path: Path | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [str(image_path)]
    response = ollama.chat(model=model, format="json", messages=[message])
    return parse_json_text(response["message"]["content"])


def call_openai_json(model: str, prompt: str, image_path: Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if image_path is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
                "detail": "high",
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_object"}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=120) as client:
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
    data = response.json()
    text = data.get("output_text")
    if not text:
        parts = []
        for item in data.get("output", []):
            for content_item in item.get("content", []):
                if "text" in content_item:
                    parts.append(content_item["text"])
        text = "\n".join(parts)
    return parse_json_text(text)


def call_gemini_json(
    model: str,
    prompt: str,
    image_path: Path | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    parts: list[dict[str, Any]] = []
    if image_path is not None:
        parts.append({"inline_data": image_to_inline_data(image_path)})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    with httpx.Client(timeout=120) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    data = response.json()
    update_token_stats(stats, data.get("usageMetadata", {}))
    text = "\n".join(
        part.get("text", "")
        for candidate in data.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if part.get("text")
    )
    if not text.strip():
        finish_reasons = [
            candidate.get("finishReason", "unknown")
            for candidate in data.get("candidates", [])
        ]
        prompt_feedback = data.get("promptFeedback", {})
        snippet = json.dumps(data, ensure_ascii=False)[:2000]
        raise RuntimeError(
            "Gemini returned no text content. "
            f"finish_reasons={finish_reasons}; "
            f"prompt_feedback={prompt_feedback}; "
            f"response_snippet={snippet}"
        )
    try:
        return parse_json_text(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned text that could not be parsed as JSON. "
            f"finish_reasons={[candidate.get('finishReason', 'unknown') for candidate in data.get('candidates', [])]}; "
            f"text_snippet={text[:1000]}"
        ) from exc


def call_saas_json(
    provider: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider == "openai":
        return call_openai_json(model, prompt, image_path)
    if provider == "gemini":
        return call_gemini_json(model, prompt, image_path, stats)
    raise ValueError(f"Unknown SaaS provider: {provider}")


def is_quota_error(text: str) -> bool:
    lowered = text.lower()
    return "quota" in lowered or "resource_exhausted" in lowered


def is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "ratelimit" in text or "429" in text


def is_retryable_saas_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return is_rate_limit_error(exc) or is_quota_error(str(exc))


def record_saas_error(stats: dict[str, Any] | None, exc: Exception) -> None:
    if stats is None:
        return
    if is_rate_limit_error(exc):
        stats["rate_limit_errors"] += 1
    if is_quota_error(str(exc)):
        stats["quota_errors"] += 1
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        stats["network_errors"] += 1


def call_saas_json_with_retry(
    provider: str,
    model: str,
    prompt: str,
    image_path: Path | None,
    *,
    request_delay_sec: float,
    max_retries: int,
    backoff_base_sec: float,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    attempt = 0
    while True:
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        if stats is not None:
            stats["request_count"] += 1
        try:
            result = call_saas_json(provider, model, prompt, image_path, stats)
            if stats is not None:
                stats["api_success_count"] += 1
            return result
        except Exception as exc:
            record_saas_error(stats, exc)
            if attempt >= max_retries or not is_retryable_saas_error(exc):
                if stats is not None:
                    stats["api_failure_count"] += 1
                raise
            attempt += 1
            if stats is not None:
                stats["retry_count"] += 1
            time.sleep(backoff_base_sec * (2 ** (attempt - 1)))


def validate_caption(caption: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(read_json(SCHEMA_PATH))
    return prompt_runtime.validate_caption(caption, validator)


def generate_one(
    *,
    row: dict[str, Any],
    provider: str,
    model: str,
    mode: str,
    saas_provider: str | None = None,
    request_delay_sec: float = 0.0,
    max_retries: int = 0,
    backoff_base_sec: float = 1.0,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    image_path = Path(row["image"])
    category = row["category"]
    caption_prompt = build_caption_prompt(category, mode)
    if provider == "local":
        caption = call_ollama_json(model, caption_prompt, image_path)
    else:
        caption = call_saas_json_with_retry(
            saas_provider or provider,
            model,
            caption_prompt,
            image_path,
            request_delay_sec=request_delay_sec,
            max_retries=max_retries,
            backoff_base_sec=backoff_base_sec,
            stats=stats,
        )

    vqa = None
    self_check = None
    if mode == "full":
        vqa_prompt = build_vqa_prompt(category)
        if provider == "local":
            vqa = call_ollama_json(model, vqa_prompt, image_path)
        else:
            vqa = call_saas_json_with_retry(
                saas_provider or provider,
                model,
                vqa_prompt,
                image_path,
                request_delay_sec=request_delay_sec,
                max_retries=max_retries,
                backoff_base_sec=backoff_base_sec,
                stats=stats,
            )

        self_check_prompt = build_self_check_prompt(category, caption, vqa)
        if provider == "local":
            self_check = call_ollama_json(model, self_check_prompt)
        else:
            self_check = call_saas_json_with_retry(
                saas_provider or provider,
                model,
                self_check_prompt,
                None,
                request_delay_sec=request_delay_sec,
                max_retries=max_retries,
                backoff_base_sec=backoff_base_sec,
                stats=stats,
            )

    schema_errors = validate_caption(caption)
    duration_sec = time.monotonic() - started
    return {
        "metadata": {
            "image": str(image_path),
            "image_name": row["image_name"],
            "scopes": row["scopes"],
            "raw_category": category,
            "target_category": caption.get("category_specific_check", {}).get(
                "target_category", category
            ),
            "provider": provider,
            "saas_provider": saas_provider,
            "model": model,
            "mode": mode,
            "created_at": now_iso(),
            "total_duration_sec": duration_sec,
        },
        "caption": caption,
        "auxiliary_vqa": vqa,
        "self_check": self_check,
        "schema_validation": {
            "pass": not schema_errors,
            "errors": schema_errors,
        },
    }


def selected_rows(
    rows: list[dict[str, Any]],
    scopes: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if scopes.intersection(set(row.get("scopes", [])))]
    return filtered[:limit] if limit else filtered


def scope_arg(value: str) -> set[str]:
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if not parts or parts == {"none"}:
        return set()
    invalid = parts - {"smoke", "formal"}
    if invalid:
        raise ValueError(f"Invalid scope(s): {', '.join(sorted(invalid))}")
    return parts


def generate_outputs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    rows = read_manifest(run_dir)
    scopes = scope_arg(args.scopes)
    modes = tuple(args.modes.split(","))
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"Invalid mode: {mode}")

    provider = args.provider
    if provider == "local":
        model = args.local_model
        saas_provider = None
    else:
        saas_provider = args.saas_provider
        model = args.saas_model
    stats = empty_generation_stats(provider, model, saas_provider)

    if not scopes:
        print(f"Skipped {provider} generation because scopes=none")
        existing = read_optional_json(experiment_record_path(run_dir)).get("generation", {})
        update_experiment_record(
            run_dir,
            {
                "run_metadata": build_run_metadata(run_dir),
                "generation": merge_generation_stats(existing, provider, stats),
            },
        )
        return

    jobs = [
        (row, mode)
        for row in selected_rows(rows, scopes, args.limit)
        for mode in modes
    ]
    if not jobs:
        print(f"No rows selected for scopes={sorted(scopes)}")
        return

    def run_job(job: tuple[dict[str, Any], str]) -> tuple[Path, dict[str, Any]]:
        row, mode = job
        result = generate_one(
            row=row,
            provider=provider,
            model=model,
            mode=mode,
            saas_provider=saas_provider,
            request_delay_sec=getattr(args, "saas_request_delay_sec", 0.0),
            max_retries=getattr(args, "saas_max_retries", 0),
            backoff_base_sec=getattr(args, "saas_backoff_base_sec", 1.0),
            stats=stats if provider == "saas" else None,
        )
        path = output_path(run_dir, provider, mode, Path(row["image"]))
        write_json(path, result)
        return path, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                path, result = future.result()
                ok = "schema ok" if result["schema_validation"]["pass"] else "schema failed"
                mode = result.get("metadata", {}).get("mode")
                if mode in stats["by_mode"]:
                    stats["by_mode"][mode]["output_count"] += 1
                    stats["by_mode"][mode]["success_count"] += 1
                    stats["by_mode"][mode]["total_duration_sec"] += result.get(
                        "metadata", {}
                    ).get("total_duration_sec", 0.0)
                    if result.get("schema_validation", {}).get("pass"):
                        stats["by_mode"][mode]["schema_valid_count"] += 1
                stats["success_count"] += 1
                stats["total_duration_sec"] += result.get("metadata", {}).get(
                    "total_duration_sec", 0.0
                )
                print(f"Saved {path} ({ok})")
            except Exception as exc:
                stats["failure_count"] += 1
                print(f"Generation error: {exc}")
    existing = read_optional_json(experiment_record_path(run_dir)).get("generation", {})
    update_experiment_record(
        run_dir,
        {
            "run_metadata": build_run_metadata(run_dir, scopes),
            "generation": merge_generation_stats(existing, provider, stats),
        },
    )
    print(f"Saved {experiment_record_path(run_dir)}")


def prompt_call_count(mode: str) -> int:
    if mode == "simple":
        return 1
    if mode == "full":
        return 3
    raise ValueError(f"Unknown prompt mode: {mode}")


def prompt_char_count(row: dict[str, Any], mode: str) -> int:
    category = row["category"]
    total = len(build_caption_prompt(category, mode))
    if mode == "full":
        total += len(build_vqa_prompt(category))
        total += len(build_self_check_prompt(category, {}, {}))
    return total


def select_probe_row(
    rows: list[dict[str, Any]],
    scopes: set[str],
    seed: int,
) -> dict[str, Any]:
    selected = selected_rows(rows, scopes, 0)
    if not selected:
        raise ValueError(f"No rows available for probe scopes={sorted(scopes)}")
    return random.Random(seed).choice(selected)


def estimate_gemini_consumption(
    *,
    probe_results: dict[str, dict[str, Any]],
    selected_count: int,
    model: str,
) -> dict[str, Any]:
    limits = saas_limits_for_model(model)
    rpm = int(limits.get("rpm", 0) or 0)
    rpd = int(limits.get("rpd", 0) or 0)
    tpm = int(limits.get("tpm", 0) or 0)
    min_delay_for_rpm = 60 / rpm if rpm else None
    estimates: dict[str, Any] = {
        "basis": "one randomly selected image per mode",
        "selected_scope_image_count": selected_count,
        "model_limits": limits,
        "request_token_quota_relation": {
            "request_count": "Number of API attempts. Used for RPM pressure and approximates request-count RPD consumption.",
            "quota_consuming_request_count": "Requests that consume request quota. For planning, treat this as request_count, including blocked calls and retries.",
            "input_token_count": "Gemini usageMetadata.promptTokenCount. Used for TPM pressure.",
            "output_token_count": "Gemini usageMetadata.candidatesTokenCount.",
            "total_token_count": "Gemini usageMetadata.totalTokenCount.",
        },
        "modes": {},
        "combined_simple_full": {
            "estimated_request_count": 0,
            "estimated_quota_consuming_request_count": 0,
            "estimated_duration_sec": 0.0,
            "estimated_duration_with_configured_sleep_sec": 0.0,
            "estimated_rpm_safe_min_duration_sec": 0.0,
            "estimated_retry_count": 0,
            "estimated_rate_limit_errors": 0,
            "estimated_quota_errors": 0,
            "estimated_input_token_count": 0,
            "estimated_output_token_count": 0,
            "estimated_total_token_count": 0,
        },
        "cost": "unknown",
    }
    for mode, result in probe_results.items():
        probe = result["probe"]
        expected_requests = prompt_call_count(mode)
        per_image_requests = max(probe["request_count"], expected_requests)
        per_image_duration = probe["duration_sec"]
        per_image_retries = probe["retry_count"]
        per_image_rate_limits = probe["rate_limit_errors"]
        per_image_quota_errors = probe["quota_errors"]
        token_count = probe.get("token_count", {})
        estimated_input_tokens = token_count.get("input", 0) * selected_count
        estimated_output_tokens = token_count.get("output", 0) * selected_count
        estimated_total_tokens = token_count.get("total", 0) * selected_count
        configured_sleep_sec = probe["request_delay_sec"] * per_image_requests * selected_count
        estimated_duration_sec = per_image_duration * selected_count
        rpm_safe_min_duration_sec = (
            per_image_requests * selected_count * min_delay_for_rpm
            if min_delay_for_rpm is not None
            else 0.0
        )
        duration_for_tpm_sec = max(estimated_duration_sec, rpm_safe_min_duration_sec, 1.0)
        projected_input_tpm = estimated_input_tokens / (duration_for_tpm_sec / 60)
        mode_estimate = {
            "per_image_request_count": per_image_requests,
            "observed_request_count_per_image": probe["request_count"],
            "expected_request_count_per_image": expected_requests,
            "estimated_request_count": per_image_requests * selected_count,
            "estimated_quota_consuming_request_count": per_image_requests * selected_count,
            "per_image_duration_sec": per_image_duration,
            "estimated_duration_sec": estimated_duration_sec,
            "configured_request_delay_sec": probe["request_delay_sec"],
            "configured_sleep_sec": configured_sleep_sec,
            "recommended_min_request_delay_sec_for_rpm": min_delay_for_rpm,
            "rpm_safe": (
                True if min_delay_for_rpm is None else probe["request_delay_sec"] >= min_delay_for_rpm
            ),
            "estimated_rpm_safe_min_duration_sec": rpm_safe_min_duration_sec,
            "estimated_retry_count": per_image_retries * selected_count,
            "estimated_rate_limit_errors": per_image_rate_limits * selected_count,
            "estimated_quota_errors": per_image_quota_errors * selected_count,
            "estimated_input_token_count": estimated_input_tokens,
            "estimated_output_token_count": estimated_output_tokens,
            "estimated_total_token_count": estimated_total_tokens,
            "estimated_input_tpm": projected_input_tpm,
            "tpm_safe": True if not tpm else projected_input_tpm <= tpm,
            "rpd_safe": True if not rpd else per_image_requests * selected_count <= rpd,
        }
        estimates["modes"][mode] = mode_estimate
        combined = estimates["combined_simple_full"]
        combined["estimated_request_count"] += mode_estimate["estimated_request_count"]
        combined["estimated_quota_consuming_request_count"] += mode_estimate[
            "estimated_quota_consuming_request_count"
        ]
        combined["estimated_duration_sec"] += mode_estimate["estimated_duration_sec"]
        combined["estimated_duration_with_configured_sleep_sec"] += mode_estimate[
            "estimated_duration_sec"
        ]
        combined["estimated_rpm_safe_min_duration_sec"] += mode_estimate[
            "estimated_rpm_safe_min_duration_sec"
        ]
        combined["estimated_retry_count"] += mode_estimate["estimated_retry_count"]
        combined["estimated_rate_limit_errors"] += mode_estimate["estimated_rate_limit_errors"]
        combined["estimated_quota_errors"] += mode_estimate["estimated_quota_errors"]
        combined["estimated_input_token_count"] += mode_estimate["estimated_input_token_count"]
        combined["estimated_output_token_count"] += mode_estimate["estimated_output_token_count"]
        combined["estimated_total_token_count"] += mode_estimate["estimated_total_token_count"]
    combined = estimates["combined_simple_full"]
    combined_duration_for_tpm = max(
        combined["estimated_duration_sec"],
        combined["estimated_rpm_safe_min_duration_sec"],
        1.0,
    )
    combined["estimated_input_tpm"] = combined["estimated_input_token_count"] / (
        combined_duration_for_tpm / 60
    )
    combined["rpm_safe"] = all(
        mode_estimate["rpm_safe"] for mode_estimate in estimates["modes"].values()
    )
    combined["rpd_safe"] = (
        True if not rpd else combined["estimated_quota_consuming_request_count"] <= rpd
    )
    combined["tpm_safe"] = True if not tpm else combined["estimated_input_tpm"] <= tpm
    return estimates


def run_gemini_probe(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    rows = read_manifest(run_dir)
    scopes = scope_arg(args.scopes)
    if not scopes:
        raise ValueError("--scopes cannot be none for gemini-probe")
    selected = selected_rows(rows, scopes, args.limit)
    probe_row = select_probe_row(rows, scopes, args.seed)
    probe_id = f"gemini_probe_{now_iso().replace(':', '').replace('-', '')}"
    probe_dir = run_dir / "probes" / probe_id
    probe_results: dict[str, dict[str, Any]] = {}

    for mode in MODES:
        stats = empty_generation_stats("saas", args.saas_model, "gemini")
        started = time.monotonic()
        output_file = probe_dir / f"{mode}.json"
        error = None
        schema_valid = False
        stats_before = {
            key: stats.get(key, 0)
            for key in (
                "input_token_count",
                "output_token_count",
                "total_token_count",
                "input_text_token_count",
                "input_image_token_count",
            )
        }
        try:
            result = generate_one(
                row=probe_row,
                provider="saas",
                model=args.saas_model,
                mode=mode,
                saas_provider="gemini",
                request_delay_sec=args.saas_request_delay_sec,
                max_retries=args.saas_max_retries,
                backoff_base_sec=args.saas_backoff_base_sec,
                stats=stats,
            )
            stats["success_count"] += 1
            write_json(output_file, result)
            schema_valid = result.get("schema_validation", {}).get("pass", False)
            print(f"Saved Gemini {mode} probe: {output_file}")
        except Exception as exc:
            stats["failure_count"] += 1
            error = str(exc)
            print(f"Gemini {mode} probe failed: {exc}")
        duration_sec = time.monotonic() - started
        token_count = {
            "input": stats["input_token_count"] - stats_before["input_token_count"],
            "output": stats["output_token_count"] - stats_before["output_token_count"],
            "total": stats["total_token_count"] - stats_before["total_token_count"],
            "input_text": stats["input_text_token_count"] - stats_before["input_text_token_count"],
            "input_image": stats["input_image_token_count"] - stats_before["input_image_token_count"],
            "source": "Gemini usageMetadata",
        }
        probe_results[mode] = {
            "output_file": str(output_file) if error is None else None,
            "probe": {
                "mode": mode,
                "model": args.saas_model,
                "image": probe_row["image"],
                "image_name": probe_row["image_name"],
                "category": probe_row["category"],
                "scopes": probe_row["scopes"],
                "duration_sec": duration_sec,
                "request_count": stats["request_count"],
                "api_success_count": stats["api_success_count"],
                "api_failure_count": stats["api_failure_count"],
                "retry_count": stats["retry_count"],
                "success_count": stats["success_count"],
                "failure_count": stats["failure_count"],
                "rate_limit_errors": stats["rate_limit_errors"],
                "quota_errors": stats["quota_errors"],
                "network_errors": stats["network_errors"],
                "schema_valid": schema_valid,
                "quota_consuming": {
                    "request_count": stats["request_count"],
                    "rpd_units": stats["request_count"],
                    "rpm_window_units": stats["request_count"],
                    "input_token_count": token_count["input"],
                    "tpm_units": token_count["input"],
                    "note": "For planning, blocked calls and retries still count as request pressure.",
                },
                "token_count": token_count,
                "request_delay_sec": args.saas_request_delay_sec,
                "max_retries": args.saas_max_retries,
                "backoff_base_sec": args.saas_backoff_base_sec,
                "error": error,
            },
        }

    probe_record = {
        "created_at": now_iso(),
        "provider": "gemini",
        "model": args.saas_model,
        "probe_id": probe_id,
        "selection": {
            "seed": args.seed,
            "scopes": sorted(scopes),
            "selected_scope_image_count": len(selected),
            "image": probe_row["image"],
            "image_name": probe_row["image_name"],
            "category": probe_row["category"],
        },
        "results": probe_results,
        "prediction": estimate_gemini_consumption(
            probe_results=probe_results,
            selected_count=len(selected),
            model=args.saas_model,
        ),
        "notes": (
            "Prediction is a rough projection from one image. Full prompt uses three "
            "Gemini calls per image: caption, VQA, and self-check."
        ),
    }
    write_json(probe_dir / "gemini_probe_record.json", probe_record)
    update_experiment_record(
        run_dir,
        {
            "run_metadata": build_run_metadata(run_dir, scopes),
            "gemini_probe": probe_record,
        },
    )
    print(f"Saved Gemini probe record: {probe_dir / 'gemini_probe_record.json'}")
    print(f"Saved {experiment_record_path(run_dir)}")


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


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
    return " ".join(str(part) for part in parts if part)


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu4(candidate: str, reference: str) -> float:
    cand = tokenise(candidate)
    ref = tokenise(reference)
    if not cand or not ref:
        return 0.0
    precisions = []
    for n in range(1, 5):
        cand_ngrams = ngrams(cand, n)
        ref_ngrams = ngrams(ref, n)
        if not cand_ngrams:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precisions.append(max(overlap / sum(cand_ngrams.values()), 1e-9))
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / 4)


def rouge_l(candidate: str, reference: str) -> float:
    cand = tokenise(candidate)
    ref = tokenise(reference)
    if not cand or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
    for i, cand_token in enumerate(cand, 1):
        for j, ref_token in enumerate(ref, 1):
            if cand_token == ref_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def meteor_lite(candidate: str, reference: str) -> float:
    cand = tokenise(candidate)
    ref = tokenise(reference)
    if not cand or not ref:
        return 0.0
    overlap = sum((Counter(cand) & Counter(ref)).values())
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 0.0 if precision + recall == 0 else 10 * precision * recall / (recall + 9 * precision)


def cider_lite(candidate: str, reference: str) -> float:
    cand = tokenise(candidate)
    ref = tokenise(reference)
    if not cand or not ref:
        return 0.0
    scores = []
    for n in range(1, 5):
        cand_ngrams = ngrams(cand, n)
        ref_ngrams = ngrams(ref, n)
        if not cand_ngrams or not ref_ngrams:
            scores.append(0.0)
            continue
        keys = set(cand_ngrams) | set(ref_ngrams)
        dot = sum(cand_ngrams[key] * ref_ngrams[key] for key in keys)
        cand_norm = math.sqrt(sum(value * value for value in cand_ngrams.values()))
        ref_norm = math.sqrt(sum(value * value for value in ref_ngrams.values()))
        scores.append(dot / (cand_norm * ref_norm) if cand_norm and ref_norm else 0.0)
    return sum(scores) / len(scores)


def completion_rate(bundle: dict[str, Any]) -> float:
    caption = bundle.get("caption", {})
    fields = [
        ("image_observation", "body_site"),
        ("image_observation", "wound_presence"),
        ("image_observation", "visual_summary"),
        ("wound_features", "shape_pattern"),
        ("wound_features", "edges_margins"),
        ("wound_features", "wound_bed"),
        ("wound_features", "color"),
        ("wound_features", "texture"),
        ("wound_features", "fluid_exudate_bleeding"),
        ("wound_features", "periwound_skin"),
    ]
    complete = 0
    for section, field in fields:
        value = caption.get(section, {}).get(field)
        if value not in (None, "", [], {}):
            complete += 1
    return complete / len(fields)


def no_ruler_compliance(bundle: dict[str, Any]) -> float:
    measurement = (
        bundle.get("caption", {})
        .get("image_observation", {})
        .get("measurement_tool", {})
    )
    visible = measurement.get("visible")
    estimated_size = str(measurement.get("estimated_size", "")).lower()
    if visible is False:
        return 1.0 if estimated_size == "unknown" else 0.0
    return 1.0


def not_observed_usage(bundle: dict[str, Any]) -> float:
    text = json.dumps(bundle.get("caption", {}), ensure_ascii=False).lower()
    return 1.0 if "not observed" in text else 0.0


def safety_scope_compliance(bundle: dict[str, Any]) -> float:
    scope = bundle.get("caption", {}).get("safety_scope")
    return 1.0 if scope == "Visual description only; not a medical diagnosis." else 0.0


def self_metrics(bundles: list[dict[str, Any]]) -> dict[str, float | int]:
    if not bundles:
        return {
            "count": 0,
            "schema_valid_rate": 0.0,
            "required_field_completion_rate": 0.0,
            "not_observed_usage_rate": 0.0,
            "no_ruler_size_compliance_rate": 0.0,
            "safety_scope_compliance_rate": 0.0,
        }
    return {
        "count": len(bundles),
        "schema_valid_rate": sum(
            1 for bundle in bundles if bundle.get("schema_validation", {}).get("pass")
        )
        / len(bundles),
        "required_field_completion_rate": sum(completion_rate(bundle) for bundle in bundles)
        / len(bundles),
        "not_observed_usage_rate": sum(not_observed_usage(bundle) for bundle in bundles)
        / len(bundles),
        "no_ruler_size_compliance_rate": sum(no_ruler_compliance(bundle) for bundle in bundles)
        / len(bundles),
        "safety_scope_compliance_rate": sum(safety_scope_compliance(bundle) for bundle in bundles)
        / len(bundles),
    }


def load_bundles(
    run_dir: Path,
    provider: str,
    mode: str,
    scopes: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir = run_dir / "outputs" / provider / mode
    if not output_dir.exists():
        return {}
    bundles = {}
    for path in output_dir.glob("*.json"):
        if not path.is_file():
            continue
        bundle = read_json(path)
        bundle_scopes = set(bundle.get("metadata", {}).get("scopes", []))
        if scopes is not None and not scopes.intersection(bundle_scopes):
            continue
        bundles[path.stem] = bundle
    return bundles


def paired_scores(
    candidate_bundles: dict[str, dict[str, Any]],
    reference_bundles: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    common = sorted(set(candidate_bundles) & set(reference_bundles))
    unmatched = len(set(candidate_bundles) ^ set(reference_bundles))
    if not common:
        return {
            "paired_count": 0,
            "matched_image_count": 0,
            "unmatched_image_count": unmatched,
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "meteor_lite": 0.0,
            "cider_lite": 0.0,
        }
    scores = defaultdict(float)
    for key in common:
        candidate = caption_text(candidate_bundles[key])
        reference = caption_text(reference_bundles[key])
        scores["bleu4"] += bleu4(candidate, reference)
        scores["rouge_l"] += rouge_l(candidate, reference)
        scores["meteor_lite"] += meteor_lite(candidate, reference)
        scores["cider_lite"] += cider_lite(candidate, reference)
    return {
        "paired_count": len(common),
        "matched_image_count": len(common),
        "unmatched_image_count": unmatched,
        "bleu4": scores["bleu4"] / len(common),
        "rouge_l": scores["rouge_l"] / len(common),
        "meteor_lite": scores["meteor_lite"] / len(common),
        "cider_lite": scores["cider_lite"] / len(common),
    }


def paired_scores_by_category(
    candidate_bundles: dict[str, dict[str, Any]],
    reference_bundles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_category: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for key in sorted(set(candidate_bundles) & set(reference_bundles)):
        category = candidate_bundles[key].get("metadata", {}).get("raw_category", "unknown")
        candidate_group, reference_group = by_category.setdefault(category, ({}, {}))
        candidate_group[key] = candidate_bundles[key]
        reference_group[key] = reference_bundles[key]
    return {
        category: paired_scores(candidate_group, reference_group)
        for category, (candidate_group, reference_group) in sorted(by_category.items())
    }


def format_pct(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def write_scores_md(run_dir: Path, scores: dict[str, Any]) -> None:
    lines = [
        "# Evaluation Scores",
        "",
        "## Provider / Prompt Quality Table",
        "",
        "| Provider | Prompt mode | Count | Schema valid | Field completion | Not-observed | No-ruler compliance | Safety scope |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider in PROVIDERS:
        for mode in MODES:
            cell = scores["cells"].get(provider, {}).get(mode, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        provider,
                        mode,
                        str(cell.get("count", 0)),
                        format_pct(cell.get("schema_valid_rate", 0.0)),
                        format_pct(cell.get("required_field_completion_rate", 0.0)),
                        format_pct(cell.get("not_observed_usage_rate", 0.0)),
                        format_pct(cell.get("no_ruler_size_compliance_rate", 0.0)),
                        format_pct(cell.get("safety_scope_compliance_rate", 0.0)),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Official Same-Mode Local Scores",
            "",
            "| Comparison | Matched | Unmatched | BLEU-4 | ROUGE-L | METEOR-lite | CIDEr-lite |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("local_simple_vs_saas_simple", "local_full_vs_saas_full"):
        comparison = scores.get("evaluation_comparisons", {}).get(name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(comparison.get("matched_image_count", 0)),
                    str(comparison.get("unmatched_image_count", 0)),
                    format_pct(comparison.get("bleu4", 0.0)),
                    format_pct(comparison.get("rouge_l", 0.0)),
                    format_pct(comparison.get("meteor_lite", 0.0)),
                    format_pct(comparison.get("cider_lite", 0.0)),
                ]
            )
            + " |"
        )
    gate = scores.get("baseline_quality_gate", {})
    if gate:
        lines.extend(
            [
                "",
                "## Baseline Quality Gate",
                "",
                f"- Reference: {gate.get('reference_provider')} {gate.get('reference_mode')}",
                f"- Schema valid minimum: {gate.get('json_valid_rate_min')}",
                f"- Field completion minimum: {gate.get('field_completion_rate_min')}",
                f"- Manual review: {gate.get('manual_review_result')}",
                f"- Approved for scoring: {gate.get('approved_for_scoring')}",
            ]
        )
    if scores.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in scores["warnings"])
    (run_dir / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_model_name(bundles: dict[str, dict[str, Any]]) -> str:
    for bundle in bundles.values():
        model = bundle.get("metadata", {}).get("model")
        if model:
            return str(model)
    return "unknown"


def score_delta_summary(
    simple_scores: dict[str, Any],
    full_scores: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "derived_metric",
        "purpose": (
            "compare local full score against local simple score through same-mode "
            "SaaS references"
        ),
        "bleu_delta": full_scores.get("bleu4", 0.0) - simple_scores.get("bleu4", 0.0),
        "rouge_l_delta": full_scores.get("rouge_l", 0.0)
        - simple_scores.get("rouge_l", 0.0),
        "meteor_delta": full_scores.get("meteor_lite", 0.0)
        - simple_scores.get("meteor_lite", 0.0),
        "cider_lite_delta": full_scores.get("cider_lite", 0.0)
        - simple_scores.get("cider_lite", 0.0),
        "interpretation": "",
    }


def baseline_quality_gate(
    *,
    reference_provider: str,
    reference_mode: str,
    reference_cell: dict[str, Any],
) -> dict[str, Any]:
    schema_rate = reference_cell.get("schema_valid_rate", 0.0)
    field_rate = reference_cell.get("required_field_completion_rate", 0.0)
    metric_pass = (
        schema_rate >= DEFAULT_QUALITY_SCHEMA_MIN
        and field_rate >= DEFAULT_QUALITY_FIELD_MIN
    )
    return {
        "reference_provider": reference_provider,
        "reference_mode": reference_mode,
        "json_valid_rate_min": DEFAULT_QUALITY_SCHEMA_MIN,
        "field_completion_rate_min": DEFAULT_QUALITY_FIELD_MIN,
        "manual_review_sample_count": 20,
        "manual_review_result": "pending",
        "manual_review_pass": False,
        "metric_pass": metric_pass,
        "approved_for_scoring": False,
        "score_status": "exploratory",
        "notes": "Manual review is pending; local scores are not final results yet.",
    }


def score_outputs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    score_scopes = scope_arg(args.score_scopes)
    reference_bundles_by_mode = {
        mode: load_bundles(run_dir, "saas", mode, score_scopes) for mode in MODES
    }
    warnings = []
    for mode, bundles in reference_bundles_by_mode.items():
        if not bundles:
            warnings.append(
                f"No SaaS reference outputs found for mode={mode}, "
                f"scopes={sorted(score_scopes)}. Same-mode local score for {mode} "
                "will have zero matched images."
            )

    scores: dict[str, Any] = {
        "created_at": now_iso(),
        "reference_provider": "saas",
        "reference_mode": "same-mode",
        "score_scopes": sorted(score_scopes),
        "cells": {},
        "evaluation_comparisons": {},
        "score_summaries": [],
        "score_by_category": {},
        "warnings": warnings,
    }
    csv_rows = []
    for provider in PROVIDERS:
        scores["cells"][provider] = {}
        for mode in MODES:
            bundles = load_bundles(run_dir, provider, mode, score_scopes)
            bundle_list = list(bundles.values())
            cell = dict(self_metrics(bundle_list))
            if provider == "local":
                cell["vs_reference"] = paired_scores(bundles, reference_bundles_by_mode[mode])
                scores["score_by_category"][mode] = paired_scores_by_category(
                    bundles,
                    reference_bundles_by_mode[mode],
                )
            else:
                cell["vs_reference"] = {}
            if provider == "local" and cell["count"] and not cell["vs_reference"]["paired_count"]:
                warnings.append(
                    f"Local {mode} has {cell['count']} output(s), but none pair with SaaS "
                    f"{mode} reference on scopes={sorted(score_scopes)}."
                )
            scores["cells"][provider][mode] = cell
            if provider == "local":
                summary = {
                    "scope": ",".join(sorted(score_scopes)),
                    "reference_provider": "saas",
                    "reference_model": infer_model_name(reference_bundles_by_mode[mode]),
                    "reference_mode": mode,
                    "candidate_provider": "local",
                    "candidate_model": infer_model_name(bundles),
                    "candidate_mode": mode,
                    "sample_count": cell.get("count", 0),
                    "matched_image_count": cell["vs_reference"].get("matched_image_count", 0),
                    "unmatched_image_count": cell["vs_reference"].get("unmatched_image_count", 0),
                    "json_valid_rate": cell.get("schema_valid_rate", 0.0),
                    "field_completion_rate": cell.get("required_field_completion_rate", 0.0),
                    "not_observed_usage_rate": cell.get("not_observed_usage_rate", 0.0),
                    "bleu": cell["vs_reference"].get("bleu4", 0.0),
                    "rouge_l": cell["vs_reference"].get("rouge_l", 0.0),
                    "meteor": cell["vs_reference"].get("meteor_lite", 0.0),
                    "cider_lite": cell["vs_reference"].get("cider_lite", 0.0),
                    "overall_score": None,
                }
                scores["score_summaries"].append(summary)
            csv_rows.append(
                {
                    "provider": provider,
                    "mode": mode,
                    **cell,
                    **{f"vs_reference_{k}": v for k, v in cell["vs_reference"].items()},
                }
            )

    local_simple = scores["cells"].get("local", {}).get("simple", {})
    local_full = scores["cells"].get("local", {}).get("full", {})
    saas_simple = scores["cells"].get("saas", {}).get("simple", {})
    saas_full = scores["cells"].get("saas", {}).get("full", {})
    scores["evaluation_comparisons"] = {
        "local_simple_vs_saas_simple": {
            "type": "score",
            "purpose": "evaluate local model under the simple prompt baseline",
            "candidate": "local_simple",
            "reference": "saas_simple",
            **local_simple.get("vs_reference", {}),
            "json_valid_rate": local_simple.get("schema_valid_rate", 0.0),
        },
        "local_full_vs_saas_full": {
            "type": "score",
            "purpose": "evaluate local model under the full prompt pipeline",
            "candidate": "local_full",
            "reference": "saas_full",
            **local_full.get("vs_reference", {}),
            "json_valid_rate": local_full.get("schema_valid_rate", 0.0),
        },
        "saas_simple_vs_saas_full": {
            "type": "quality_check",
            "purpose": "validate whether full prompt improves SaaS baseline quality",
            "score_calculated": False,
            "manual_review_required": True,
            "json_valid_rate_simple": saas_simple.get("schema_valid_rate", 0.0),
            "json_valid_rate_full": saas_full.get("schema_valid_rate", 0.0),
            "field_completion_rate_simple": saas_simple.get(
                "required_field_completion_rate", 0.0
            ),
            "field_completion_rate_full": saas_full.get(
                "required_field_completion_rate", 0.0
            ),
            "not_observed_usage_rate_simple": saas_simple.get("not_observed_usage_rate", 0.0),
            "not_observed_usage_rate_full": saas_full.get("not_observed_usage_rate", 0.0),
            "manual_review_result": "pending",
            "notes": "",
        },
        "local_prompt_effect_summary": score_delta_summary(
            local_simple.get("vs_reference", {}),
            local_full.get("vs_reference", {}),
        ),
    }
    scores["baseline_quality_gate"] = baseline_quality_gate(
        reference_provider="saas",
        reference_mode=args.reference_mode,
        reference_cell=scores["cells"].get("saas", {}).get(args.reference_mode, {}),
    )

    write_json(run_dir / "scores.json", scores)
    write_scores_md(run_dir, scores)
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    update_experiment_record(
        run_dir,
        {
            "run_metadata": build_run_metadata(run_dir, score_scopes),
            "scores": scores,
            "score_summary": scores["score_summaries"],
            "evaluation_comparisons": scores["evaluation_comparisons"],
            "baseline_quality_gate": scores["baseline_quality_gate"],
        },
    )
    print(f"Saved {run_dir / 'scores.json'}")
    print(f"Saved {run_dir / 'scores.md'}")
    print(f"Saved {experiment_record_path(run_dir)}")


def write_fallback_png(path: Path, values: list[tuple[str, float]]) -> None:
    import struct
    import zlib

    width, height = 800, 480
    margin = 60
    pixels = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            index = (y * width + x) * 3
            pixels[index : index + 3] = bytes(color)

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                set_pixel(x, y, color)

    fill_rect(margin, height - margin, width - margin, height - margin + 2, (40, 40, 40))
    fill_rect(margin, margin, margin + 2, height - margin, (40, 40, 40))
    if values:
        max_value = max(value for _, value in values) or 1.0
        bar_area = width - margin * 2
        bar_width = max(16, bar_area // max(len(values) * 2, 1))
        colors = [(43, 109, 176), (219, 120, 66), (73, 152, 104), (154, 103, 186)]
        for index, (_, value) in enumerate(values):
            bar_height = int((height - margin * 2) * max(value, 0.0) / max_value)
            x0 = margin + index * (bar_width * 2) + bar_width // 2
            y0 = height - margin - bar_height
            fill_rect(x0, y0, x0 + bar_width, height - margin, colors[index % len(colors)])

    raw_rows = []
    for y in range(height):
        row_start = y * width * 3
        raw_rows.append(b"\x00" + bytes(pixels[row_start : row_start + width * 3]))
    raw = b"".join(raw_rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_bar_chart(path: Path, title: str, values: list[tuple[str, float]]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [label for label, _ in values]
        heights = [value for _, value in values]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, heights, color=["#2b6db0", "#db7842", "#499868", "#9a67ba"])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    except ModuleNotFoundError:
        write_fallback_png(path, values)


def generate_charts(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    scores = read_optional_json(run_dir / "scores.json")
    if not scores:
        raise FileNotFoundError(f"Missing scores.json in {run_dir}; run score first.")

    charts_dir = run_dir / "charts"
    cells = scores.get("cells", {})
    comparisons = scores.get("evaluation_comparisons", {})
    generation = read_optional_json(experiment_record_path(run_dir)).get("generation", {})
    split_summary = load_split_summary(run_dir)

    json_values = [
        (f"{provider}-{mode}", cells.get(provider, {}).get(mode, {}).get("schema_valid_rate", 0.0))
        for provider in PROVIDERS
        for mode in MODES
    ]
    write_bar_chart(charts_dir / "json_valid_rate.png", "JSON Valid Rate", json_values)

    score_values = [
        ("simple BLEU", comparisons.get("local_simple_vs_saas_simple", {}).get("bleu4", 0.0)),
        ("full BLEU", comparisons.get("local_full_vs_saas_full", {}).get("bleu4", 0.0)),
        ("simple CIDEr", comparisons.get("local_simple_vs_saas_simple", {}).get("cider_lite", 0.0)),
        ("full CIDEr", comparisons.get("local_full_vs_saas_full", {}).get("cider_lite", 0.0)),
    ]
    write_bar_chart(charts_dir / "score_comparison_table.png", "Same-Mode Scores", score_values)

    category_values = [
        (category, float(count))
        for category, count in split_summary.get("category_counts", {}).items()
    ]
    write_bar_chart(charts_dir / "category_counts.png", "Category Counts", category_values)

    matched_values = [
        (
            "simple matched",
            comparisons.get("local_simple_vs_saas_simple", {}).get("matched_image_count", 0),
        ),
        (
            "simple unmatched",
            comparisons.get("local_simple_vs_saas_simple", {}).get("unmatched_image_count", 0),
        ),
        (
            "full matched",
            comparisons.get("local_full_vs_saas_full", {}).get("matched_image_count", 0),
        ),
        (
            "full unmatched",
            comparisons.get("local_full_vs_saas_full", {}).get("unmatched_image_count", 0),
        ),
    ]
    write_bar_chart(charts_dir / "matched_image_count.png", "Matched vs Unmatched", matched_values)

    saas_generation = generation.get("saas", {})
    api_error_values = [
        ("rate limit", saas_generation.get("rate_limit_errors", 0)),
        ("quota", saas_generation.get("quota_errors", 0)),
        ("network", saas_generation.get("network_errors", 0)),
        ("retry", saas_generation.get("retry_count", 0)),
    ]
    write_bar_chart(charts_dir / "saas_api_errors.png", "SaaS API Errors", api_error_values)

    delta = comparisons.get("local_prompt_effect_summary", {})
    delta_values = [
        ("BLEU", delta.get("bleu_delta", 0.0)),
        ("ROUGE-L", delta.get("rouge_l_delta", 0.0)),
        ("METEOR", delta.get("meteor_delta", 0.0)),
        ("CIDEr", delta.get("cider_lite_delta", 0.0)),
    ]
    write_bar_chart(charts_dir / "same_mode_local_score_delta.png", "Full - Simple Delta", delta_values)

    category_score_values = []
    for mode, mode_scores in scores.get("score_by_category", {}).items():
        for category, category_scores in mode_scores.items():
            category_score_values.append(
                (f"{mode}-{category}", category_scores.get("cider_lite", 0.0))
            )
    write_bar_chart(charts_dir / "score_by_category.png", "CIDEr-lite by Category", category_score_values)

    local_modes = generation.get("local", {}).get("by_mode", {})
    latency_values = [
        (
            f"local-{mode}",
            mode_stats.get("total_duration_sec", 0.0)
            / max(mode_stats.get("output_count", 0), 1),
        )
        for mode, mode_stats in local_modes.items()
    ]
    saas_modes = generation.get("saas", {}).get("by_mode", {})
    latency_values.extend(
        (
            f"saas-{mode}",
            mode_stats.get("total_duration_sec", 0.0)
            / max(mode_stats.get("output_count", 0), 1),
        )
        for mode, mode_stats in saas_modes.items()
    )
    write_bar_chart(charts_dir / "latency_by_provider_mode.png", "Latency by Provider/Mode", latency_values)

    token_values = [
        ("local simple", 0.0),
        ("local full", 0.0),
    ]
    write_bar_chart(charts_dir / "tokens_per_second_local.png", "Local Tokens/sec", token_values)

    update_experiment_record(
        run_dir,
        {
            "charts": {
                "created_at": now_iso(),
                "charts_dir": str(charts_dir),
                "files": sorted(path.name for path in charts_dir.glob("*.png")),
            }
        },
    )
    print(f"Saved charts in {charts_dir}")


def build_generate_args(
    args: argparse.Namespace,
    provider: str,
    scopes: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=args.run_dir,
        provider=provider,
        scopes=scopes,
        modes=args.modes,
        local_model=args.local_model,
        saas_provider=args.saas_provider,
        saas_model=args.saas_model,
        workers=args.workers,
        limit=args.limit,
        saas_request_delay_sec=args.saas_request_delay_sec,
        saas_max_retries=args.saas_max_retries,
        saas_backoff_base_sec=args.saas_backoff_base_sec,
    )


def generate_both(args: argparse.Namespace) -> None:
    tasks = [
        build_generate_args(args, "local", args.local_scopes),
        build_generate_args(args, "saas", args.saas_scopes),
    ]
    if args.parallel_providers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(generate_outputs, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for task in tasks:
            generate_outputs(task)


def run_all(args: argparse.Namespace) -> None:
    prepare_dataset(args)
    generate_both(args)
    if args.stop_before_score:
        print(
            "Stopped before scoring. Review SaaS/local outputs, then run the score command."
        )
        return
    score_outputs(args)
    generate_charts(args)


def add_common_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scopes", default="smoke")
    parser.add_argument("--modes", default="simple,full")
    parser.add_argument("--local-model", default="qwen3.5")
    parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--saas-request-delay-sec", type=float, default=0.0)
    parser.add_argument("--saas-max-retries", type=int, default=3)
    parser.add_argument("--saas-backoff-base-sec", type=float, default=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation pipeline for wound VLM prompts.")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to env file containing API keys. Default: repo .env",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--image-dir", required=True)
    prepare_parser.add_argument("--run-dir", required=True)
    prepare_parser.add_argument("--smoke-percent", type=int, default=10)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--exclude-dirs",
        default=DEFAULT_EXCLUDE_DIRS,
        help="Comma-separated folder names to skip while scanning images.",
    )
    prepare_parser.set_defaults(func=prepare_dataset)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--image-dir", required=True)
    split_parser.add_argument("--run-dir", required=True)
    split_parser.add_argument("--smoke-percent", type=int, default=10)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument(
        "--exclude-dirs",
        default=DEFAULT_EXCLUDE_DIRS,
        help="Deprecated alias for prepare. Comma-separated folder names to skip.",
    )
    split_parser.set_defaults(func=prepare_dataset)

    local_parser = subparsers.add_parser("generate-local")
    add_common_generate_args(local_parser)
    local_parser.set_defaults(provider="local", scopes="smoke", func=generate_outputs)

    saas_parser = subparsers.add_parser("generate-saas")
    add_common_generate_args(saas_parser)
    saas_parser.set_defaults(provider="saas", scopes="smoke", func=generate_outputs)

    probe_parser = subparsers.add_parser("gemini-probe")
    probe_parser.add_argument("--run-dir", required=True)
    probe_parser.add_argument("--scopes", default="smoke")
    probe_parser.add_argument("--seed", type=int, default=42)
    probe_parser.add_argument("--limit", type=int, default=0)
    probe_parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    probe_parser.add_argument("--saas-request-delay-sec", type=float, default=0.0)
    probe_parser.add_argument("--saas-max-retries", type=int, default=3)
    probe_parser.add_argument("--saas-backoff-base-sec", type=float, default=2.0)
    probe_parser.set_defaults(func=run_gemini_probe)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--run-dir", required=True)
    generate_parser.add_argument("--local-scopes", default="smoke")
    generate_parser.add_argument("--saas-scopes", default="smoke")
    generate_parser.add_argument("--modes", default="simple,full")
    generate_parser.add_argument("--local-model", default="qwen3.5")
    generate_parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    generate_parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    generate_parser.add_argument("--workers", type=int, default=1)
    generate_parser.add_argument("--limit", type=int, default=0)
    generate_parser.add_argument("--parallel-providers", action="store_true")
    generate_parser.add_argument("--saas-request-delay-sec", type=float, default=0.0)
    generate_parser.add_argument("--saas-max-retries", type=int, default=3)
    generate_parser.add_argument("--saas-backoff-base-sec", type=float, default=2.0)
    generate_parser.set_defaults(func=generate_both)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-dir", required=True)
    score_parser.add_argument("--reference-mode", choices=MODES, default="full")
    score_parser.add_argument("--score-scopes", default="smoke")
    score_parser.set_defaults(func=score_outputs)

    charts_parser = subparsers.add_parser("charts")
    charts_parser.add_argument("--run-dir", required=True)
    charts_parser.set_defaults(func=generate_charts)

    run_parser = subparsers.add_parser("run-all")
    run_parser.add_argument("--image-dir", required=True)
    run_parser.add_argument("--run-dir", required=True)
    run_parser.add_argument("--smoke-percent", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--exclude-dirs",
        default=DEFAULT_EXCLUDE_DIRS,
        help="Comma-separated folder names to skip while scanning images.",
    )
    run_parser.add_argument("--local-scopes", default="smoke")
    run_parser.add_argument("--saas-scopes", default="smoke")
    run_parser.add_argument("--modes", default="simple,full")
    run_parser.add_argument("--local-model", default="qwen3.5")
    run_parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    run_parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--parallel-providers", action="store_true")
    run_parser.add_argument("--saas-request-delay-sec", type=float, default=0.0)
    run_parser.add_argument("--saas-max-retries", type=int, default=3)
    run_parser.add_argument("--saas-backoff-base-sec", type=float, default=2.0)
    run_parser.add_argument("--reference-mode", choices=MODES, default="full")
    run_parser.add_argument("--score-scopes", default="smoke")
    run_parser.add_argument("--stop-before-score", action="store_true")
    run_parser.set_defaults(func=run_all)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))
    if getattr(args, "saas_provider", None) == "openai" and (
        not getattr(args, "saas_model", None) or args.saas_model == DEFAULT_GEMINI_MODEL
    ):
        args.saas_model = DEFAULT_OPENAI_MODEL
    args.func(args)


if __name__ == "__main__":
    main()
