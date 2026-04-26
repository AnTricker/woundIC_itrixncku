import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import ollama
from jsonschema import Draft202012Validator


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
DEFAULT_IMAGE_DIR = BASE_DIR / "images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "qwen35"
DEFAULT_MODEL = "qwen3.5"
CAPTION_SCHEMA_PATH = BASE_DIR / "wound_schema.json"

PROMPT_FILES = {
    "caption": "caption_template.md",
    "vqa": "auxiliary_vqa_template.md",
    "self_check": "self_check_template.md",
}

CATEGORY_PROMPTS = {
    "abrasion": "abrasions.md",
    "abrasions": "abrasions.md",
    "bruise": "bruises.md",
    "bruises": "bruises.md",
    "contusion": "bruises.md",
    "contusions": "bruises.md",
    "burn": "burns.md",
    "burns": "burns.md",
    "cut": "cut.md",
    "cuts": "cut.md",
    "incision": "cut.md",
    "incisions": "cut.md",
    "ingrown_nail": "ingrown_nail.md",
    "ingrown_nails": "ingrown_nail.md",
    "ingrown": "ingrown_nail.md",
    "laceration": "laceration.md",
    "lacerations": "laceration.md",
    "laseration": "laceration.md",
    "laserations": "laceration.md",
    "stab_wound": "stab_wound.md",
    "stab_wounds": "stab_wound.md",
    "stab": "stab_wound.md",
    "puncture": "stab_wound.md",
    "punctures": "stab_wound.md",
}


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(load_text(path))


def parse_json_response(response: dict) -> dict:
    content = response["message"]["content"]
    return json.loads(content)


def infer_category(image_path: Path) -> str:
    stem = image_path.stem.lower()
    prefix = stem.split(" (", 1)[0]
    prefix = prefix.split(" ", 1)[0]
    return prefix.replace("-", "_")


def load_category_prompt(category: str) -> tuple[str, str]:
    prompt_file = CATEGORY_PROMPTS.get(category)
    if not prompt_file:
        raise FileNotFoundError(
            f"No category prompt mapping for '{category}'. "
            f"Known categories: {', '.join(sorted(CATEGORY_PROMPTS))}"
        )
    return prompt_file.removesuffix(".md"), load_text(PROMPT_DIR / prompt_file)


def render_prompt(template_name: str, **values: object) -> str:
    template = load_text(PROMPT_DIR / PROMPT_FILES[template_name])
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def chat_json(model: str, prompt: str, image_path: Path | None = None) -> dict:
    message = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [str(image_path)]
    response = ollama.chat(model=model, format="json", messages=[message])
    return parse_json_response(response)


def validate_caption(caption: dict, validator: Draft202012Validator) -> list[str]:
    errors = sorted(validator.iter_errors(caption), key=lambda err: list(err.path))
    return [
        f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors
    ]


def build_output_path(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f"{image_path.stem}.json"


def run_image(
    image_path: Path,
    model: str,
    stage: str,
    validator: Draft202012Validator,
) -> dict:
    raw_category = infer_category(image_path)
    target_category, category_guidance = load_category_prompt(raw_category)

    caption_prompt = render_prompt(
        "caption",
        category=target_category,
        specific_instructions=category_guidance,
    )
    caption = chat_json(model, caption_prompt, image_path)
    schema_errors = validate_caption(caption, validator)

    vqa = None
    if stage in {"caption-vqa", "full"}:
        vqa_prompt = render_prompt(
            "vqa",
            category=target_category,
            specific_instructions=category_guidance,
        )
        vqa = chat_json(model, vqa_prompt, image_path)

    self_check = None
    if stage == "full":
        self_check_prompt = render_prompt(
            "self_check",
            category=target_category,
            specific_instructions=category_guidance,
            caption_json=json.dumps(caption, ensure_ascii=False, indent=2),
            vqa_json=json.dumps(vqa or {}, ensure_ascii=False, indent=2),
        )
        self_check = chat_json(model, self_check_prompt)

    return {
        "metadata": {
            "image": str(image_path),
            "raw_category": raw_category,
            "target_category": target_category,
            "model": model,
            "stage": stage,
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


def iter_images(image_dir: Path) -> list[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(str(image_dir / pattern)))
    return sorted(paths)


def run_captioning(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = load_json(CAPTION_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    images = iter_images(image_dir)
    if args.limit:
        images = images[: args.limit]

    if not images:
        print(f"No images found in {image_dir}")
        return

    for image_path in images:
        print(f"Processing {image_path.name}...")
        try:
            result = run_image(image_path, args.model, args.stage, validator)
            output_path = build_output_path(output_dir, image_path)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ok = "schema ok" if result["schema_validation"]["pass"] else "schema failed"
            print(f"Saved {output_path} ({ok})")
        except Exception as exc:
            print(f"Error on {image_path.name}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wound image-captioning prompt pipeline with Ollama."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--stage",
        choices=["caption-only", "caption-vqa", "full"],
        default="full",
        help="Prompt pipeline stage for A/B testing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max image count for smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_captioning(parse_args())
