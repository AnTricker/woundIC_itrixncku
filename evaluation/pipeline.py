import argparse
import base64
import concurrent.futures
import csv
import json
import math
import mimetypes
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import ollama
from jsonschema import Draft202012Validator

import main as prompt_runtime


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE_DIR / "prompts"
SCHEMA_PATH = BASE_DIR / "wound_schema.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODES = ("simple", "full")
PROVIDERS = ("saas", "local")
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ENV_FILE = BASE_DIR / ".env"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def split_dataset(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    run_dir = Path(args.run_dir)
    if args.baseline_percent + args.inference_percent != 100:
        raise ValueError("--baseline-percent + --inference-percent must equal 100")

    images = iter_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rng = random.Random(args.seed)
    by_category: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_category[prompt_runtime.infer_category(image)].append(image)

    rows: list[dict[str, Any]] = []
    for category, category_images in sorted(by_category.items()):
        shuffled = category_images[:]
        rng.shuffle(shuffled)
        baseline_count = round(len(shuffled) * args.baseline_percent / 100)
        if args.baseline_percent > 0:
            baseline_count = max(1, baseline_count)
        if args.inference_percent > 0 and baseline_count == len(shuffled) and len(shuffled) > 1:
            baseline_count -= 1

        for index, image in enumerate(shuffled):
            split = "baseline" if index < baseline_count else "inference"
            rows.append(
                {
                    "image": str(image),
                    "image_name": image.name,
                    "category": category,
                    "split": split,
                }
            )

    rows.sort(key=lambda row: (row["split"], row["category"], row["image_name"]))
    write_manifest(run_dir, rows)
    summary = Counter(row["split"] for row in rows)
    write_json(
        run_dir / "split_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_dir": str(image_dir),
            "baseline_percent": args.baseline_percent,
            "inference_percent": args.inference_percent,
            "seed": args.seed,
            "counts": dict(summary),
            "total": len(rows),
        },
    )
    print(f"Saved manifest: {manifest_path(run_dir)}")
    print(f"Split counts: {dict(summary)}")


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


def call_gemini_json(model: str, prompt: str, image_path: Path | None = None) -> dict[str, Any]:
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
    text = "\n".join(
        part.get("text", "")
        for candidate in data.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if part.get("text")
    )
    return parse_json_text(text)


def call_saas_json(
    provider: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
) -> dict[str, Any]:
    if provider == "openai":
        return call_openai_json(model, prompt, image_path)
    if provider == "gemini":
        return call_gemini_json(model, prompt, image_path)
    raise ValueError(f"Unknown SaaS provider: {provider}")


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
) -> dict[str, Any]:
    image_path = Path(row["image"])
    category = row["category"]
    caption_prompt = build_caption_prompt(category, mode)
    if provider == "local":
        caption = call_ollama_json(model, caption_prompt, image_path)
    else:
        caption = call_saas_json(saas_provider or provider, model, caption_prompt, image_path)

    vqa = None
    self_check = None
    if mode == "full":
        vqa_prompt = build_vqa_prompt(category)
        if provider == "local":
            vqa = call_ollama_json(model, vqa_prompt, image_path)
        else:
            vqa = call_saas_json(saas_provider or provider, model, vqa_prompt, image_path)

        self_check_prompt = build_self_check_prompt(category, caption, vqa)
        if provider == "local":
            self_check = call_ollama_json(model, self_check_prompt)
        else:
            self_check = call_saas_json(saas_provider or provider, model, self_check_prompt)

    schema_errors = validate_caption(caption)
    return {
        "metadata": {
            "image": str(image_path),
            "image_name": row["image_name"],
            "split": row["split"],
            "raw_category": category,
            "target_category": caption.get("category_specific_check", {}).get(
                "target_category", category
            ),
            "provider": provider,
            "saas_provider": saas_provider,
            "model": model,
            "mode": mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
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
    splits: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["split"] in splits]
    return filtered[:limit] if limit else filtered


def split_arg(value: str) -> set[str]:
    parts = {part.strip() for part in value.split(",") if part.strip()}
    invalid = parts - {"baseline", "inference"}
    if invalid:
        raise ValueError(f"Invalid split(s): {', '.join(sorted(invalid))}")
    return parts


def generate_outputs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    rows = read_manifest(run_dir)
    splits = split_arg(args.splits)
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

    jobs = [
        (row, mode)
        for row in selected_rows(rows, splits, args.limit)
        for mode in modes
    ]
    if not jobs:
        print(f"No rows selected for splits={sorted(splits)}")
        return

    def run_job(job: tuple[dict[str, Any], str]) -> tuple[Path, dict[str, Any]]:
        row, mode = job
        result = generate_one(
            row=row,
            provider=provider,
            model=model,
            mode=mode,
            saas_provider=saas_provider,
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
                print(f"Saved {path} ({ok})")
            except Exception as exc:
                print(f"Generation error: {exc}")


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
    splits: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir = run_dir / "outputs" / provider / mode
    if not output_dir.exists():
        return {}
    bundles = {}
    for path in output_dir.glob("*.json"):
        if not path.is_file():
            continue
        bundle = read_json(path)
        split = bundle.get("metadata", {}).get("split")
        if splits is not None and split not in splits:
            continue
        bundles[path.stem] = bundle
    return bundles


def paired_scores(
    candidate_bundles: dict[str, dict[str, Any]],
    reference_bundles: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    common = sorted(set(candidate_bundles) & set(reference_bundles))
    if not common:
        return {
            "paired_count": 0,
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
        "bleu4": scores["bleu4"] / len(common),
        "rouge_l": scores["rouge_l"] / len(common),
        "meteor_lite": scores["meteor_lite"] / len(common),
        "cider_lite": scores["cider_lite"] / len(common),
    }


def format_pct(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def write_scores_md(run_dir: Path, scores: dict[str, Any]) -> None:
    lines = [
        "# Evaluation Scores",
        "",
        "## 2x2 Ablation Table",
        "",
        "| Provider | Prompt mode | Count | Schema valid | Field completion | Not-observed | No-ruler compliance | Safety scope | Paired refs | BLEU-4 | ROUGE-L | METEOR-lite | CIDEr-lite |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider in PROVIDERS:
        for mode in MODES:
            cell = scores["cells"].get(provider, {}).get(mode, {})
            local_ref = cell.get("vs_reference", {})
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
                        str(local_ref.get("paired_count", 0)),
                        format_pct(local_ref.get("bleu4", 0.0)),
                        format_pct(local_ref.get("rouge_l", 0.0)),
                        format_pct(local_ref.get("meteor_lite", 0.0)),
                        format_pct(local_ref.get("cider_lite", 0.0)),
                    ]
                )
                + " |"
            )
    if scores.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in scores["warnings"])
    (run_dir / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_outputs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    score_splits = split_arg(args.score_splits)
    reference_bundles = load_bundles(run_dir, "saas", args.reference_mode, score_splits)
    warnings = []
    if not reference_bundles:
        warnings.append(
            f"No SaaS reference outputs found for mode={args.reference_mode}, "
            f"splits={sorted(score_splits)}. Run generate-saas on the same score split "
            "before scoring semantic metrics."
        )

    scores: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference_provider": "saas",
        "reference_mode": args.reference_mode,
        "score_splits": sorted(score_splits),
        "cells": {},
        "warnings": warnings,
    }
    csv_rows = []
    for provider in PROVIDERS:
        scores["cells"][provider] = {}
        for mode in MODES:
            bundles = load_bundles(run_dir, provider, mode, score_splits)
            bundle_list = list(bundles.values())
            cell = dict(self_metrics(bundle_list))
            cell["vs_reference"] = paired_scores(bundles, reference_bundles)
            if provider == "local" and cell["count"] and not cell["vs_reference"]["paired_count"]:
                warnings.append(
                    f"Local {mode} has {cell['count']} output(s), but none pair with SaaS "
                    f"{args.reference_mode} reference on splits={sorted(score_splits)}."
                )
            scores["cells"][provider][mode] = cell
            csv_rows.append(
                {
                    "provider": provider,
                    "mode": mode,
                    **cell,
                    **{f"vs_reference_{k}": v for k, v in cell["vs_reference"].items()},
                }
            )

    write_json(run_dir / "scores.json", scores)
    write_scores_md(run_dir, scores)
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {run_dir / 'scores.json'}")
    print(f"Saved {run_dir / 'scores.md'}")


def build_generate_args(
    args: argparse.Namespace,
    provider: str,
    splits: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=args.run_dir,
        provider=provider,
        splits=splits,
        modes=args.modes,
        local_model=args.local_model,
        saas_provider=args.saas_provider,
        saas_model=args.saas_model,
        workers=args.workers,
        limit=args.limit,
    )


def generate_both(args: argparse.Namespace) -> None:
    tasks = [
        build_generate_args(args, "local", args.local_splits),
        build_generate_args(args, "saas", args.saas_splits),
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
    split_dataset(args)
    generate_both(args)
    if args.stop_before_score:
        print(
            "Stopped before scoring. Review SaaS baseline/eval outputs, then run the score command."
        )
        return
    score_outputs(args)


def add_common_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits", default="baseline")
    parser.add_argument("--modes", default="simple,full")
    parser.add_argument("--local-model", default="qwen3.5")
    parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation pipeline for wound VLM prompts.")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to env file containing API keys. Default: repo .env",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--image-dir", required=True)
    split_parser.add_argument("--run-dir", required=True)
    split_parser.add_argument("--baseline-percent", type=int, required=True)
    split_parser.add_argument("--inference-percent", type=int, required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.set_defaults(func=split_dataset)

    local_parser = subparsers.add_parser("generate-local")
    add_common_generate_args(local_parser)
    local_parser.set_defaults(provider="local", splits="inference", func=generate_outputs)

    saas_parser = subparsers.add_parser("generate-saas")
    add_common_generate_args(saas_parser)
    saas_parser.set_defaults(provider="saas", splits="baseline,inference", func=generate_outputs)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--run-dir", required=True)
    generate_parser.add_argument("--local-splits", default="inference")
    generate_parser.add_argument("--saas-splits", default="baseline,inference")
    generate_parser.add_argument("--modes", default="simple,full")
    generate_parser.add_argument("--local-model", default="qwen3.5")
    generate_parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    generate_parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    generate_parser.add_argument("--workers", type=int, default=1)
    generate_parser.add_argument("--limit", type=int, default=0)
    generate_parser.add_argument("--parallel-providers", action="store_true")
    generate_parser.set_defaults(func=generate_both)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-dir", required=True)
    score_parser.add_argument("--reference-mode", choices=MODES, default="full")
    score_parser.add_argument("--score-splits", default="inference")
    score_parser.set_defaults(func=score_outputs)

    run_parser = subparsers.add_parser("run-all")
    run_parser.add_argument("--image-dir", required=True)
    run_parser.add_argument("--run-dir", required=True)
    run_parser.add_argument("--baseline-percent", type=int, required=True)
    run_parser.add_argument("--inference-percent", type=int, required=True)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--local-splits", default="inference")
    run_parser.add_argument("--saas-splits", default="baseline,inference")
    run_parser.add_argument("--modes", default="simple,full")
    run_parser.add_argument("--local-model", default="qwen3.5")
    run_parser.add_argument("--saas-provider", choices=["gemini", "openai"], default="gemini")
    run_parser.add_argument("--saas-model", default=DEFAULT_GEMINI_MODEL)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--parallel-providers", action="store_true")
    run_parser.add_argument("--reference-mode", choices=MODES, default="full")
    run_parser.add_argument("--score-splits", default="inference")
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
