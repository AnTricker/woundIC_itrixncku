from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = BASE_DIR / "wound_schema.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "bertscore_field_scores"
SCHEMA_VERSION = "caption_field_scores.v1"
LANGUAGE = "en"
EXPECTED_MODEL = "roberta-large"
MISSING = object()

DYNAMIC_WOUND_FIELDS = (
    "shape_pattern",
    "edges_margins",
    "wound_bed",
    "color",
    "texture",
    "fluid_exudate_bleeding",
    "periwound_skin",
)
ENUM_WOUND_FIELDS = ("foreign_material_debris", "necrosis_eschar")
SEMANTIC_ARRAY_FIELDS = (
    "observed_supporting_features",
    "expected_but_not_observed",
)
PRESENCE_ARRAY_FIELDS = (
    ("category_specific_check", "differential_visual_conflicts"),
    ("uncertainty", "low_confidence_regions"),
    ("uncertainty", "cannot_determine"),
)
WOUND_PRESENCE_VALUES = {
    "open wound",
    "closed discoloration",
    "nail-fold lesion",
    "unclear",
}
OBSERVATION_ENUM_VALUES = {"observed", "not observed", "uncertain"}
NONCANONICAL_NEGATIVE_PREFIXES = (
    "absent",
    "no ",
    "none",
    "not observed",
    "not visible",
    "without ",
)


@dataclass(frozen=True)
class ScoreOptions:
    idf: bool = False
    batch_size: int = 8
    device: str | None = None
    use_fast_tokenizer: bool = False


ScoreFunction = Callable[
    [list[str], list[str], ScoreOptions],
    tuple[list[float], list[float], list[float], dict[str, Any]],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one English ref/can field-level BERTScore JSON database. "
            "Different caption fields are never concatenated."
        )
    )
    parser.add_argument("ref_dir", help="Reference caption JSON directory.")
    parser.add_argument("can_dir", help="Candidate caption JSON directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalized_label(value: str) -> str:
    return normalize_space(value).casefold()


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_.-") or "unknown"


def source_root_name(path: Path) -> str:
    parts = path.resolve().parts
    output_indexes = [index for index, part in enumerate(parts) if part.casefold() == "outputs"]
    if output_indexes:
        index = output_indexes[-1]
        if index > 0:
            return parts[index - 1]
    return path.resolve().name or path.name or "unknown"


def output_path(ref_dir: Path, can_dir: Path, output_dir: Path) -> Path:
    ref_root = safe_id(source_root_name(ref_dir))
    can_root = safe_id(source_root_name(can_dir))
    return output_dir / f"ref_{ref_root}__can_{can_root}_caption_field_scores.json"


def read_caption_files(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_dir():
        raise SystemExit(f"Caption directory does not exist or is not a directory: {path}")
    bundles: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for json_path in sorted(path.glob("*.json"), key=lambda item: item.name.casefold()):
        try:
            value = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "file": json_path.name,
                    "path": str(json_path.resolve()),
                    "type": "json_load_error",
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(value, dict):
            errors.append(
                {
                    "file": json_path.name,
                    "path": str(json_path.resolve()),
                    "type": "invalid_bundle_type",
                    "message": "Top-level JSON value must be an object.",
                }
            )
            continue
        bundles[json_path.name] = value
    return bundles, errors


def caption_for(bundle: dict[str, Any]) -> Any:
    return bundle.get("caption", MISSING)


def nested_value(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return MISSING
        current = current[key]
    return current


def raw_json_value(value: Any) -> Any:
    return "<missing>" if value is MISSING else value


def side_issue(value: Any, side: str, *, uncertain_is_unresolved: bool = True) -> str | None:
    if value is MISSING:
        return f"{side}_missing"
    if value is None:
        return f"{side}_null"
    if not isinstance(value, str):
        return f"{side}_wrong_type"
    if not normalize_space(value):
        return f"{side}_empty"
    if uncertain_is_unresolved and normalized_label(value) == "uncertain":
        return f"{side}_uncertain"
    return None


def unresolved_result(metric: str, ref_value: Any, can_value: Any, reasons: list[str]) -> dict[str, Any]:
    return {
        "metric": metric,
        "status": "unresolved",
        "ref_value": raw_json_value(ref_value),
        "can_value": raw_json_value(can_value),
        "unresolved_reasons": reasons,
    }


def queue_score(
    target: dict[str, Any],
    ref_text: str,
    can_text: str,
    pending: list[tuple[dict[str, Any], str, str]],
) -> None:
    target["status"] = "pending_bertscore"
    pending.append((target, normalize_space(can_text), normalize_space(ref_text)))


def direct_text_result(
    metric: str,
    ref_value: Any,
    can_value: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    reasons = [
        issue
        for issue in (side_issue(ref_value, "ref"), side_issue(can_value, "can"))
        if issue
    ]
    if reasons:
        return unresolved_result(metric, ref_value, can_value, reasons)
    result = {
        "metric": metric,
        "ref_value": ref_value,
        "can_value": can_value,
    }
    queue_score(result, ref_value, can_value, pending)
    return result


def categorical_result(ref_value: Any, can_value: Any) -> dict[str, Any]:
    reasons: list[str] = []
    for side, value in (("ref", ref_value), ("can", can_value)):
        issue = side_issue(value, side, uncertain_is_unresolved=False)
        if issue:
            reasons.append(issue)
        elif normalized_label(value) not in WOUND_PRESENCE_VALUES:
            reasons.append(f"{side}_invalid_value")
    if reasons:
        return unresolved_result("categorical", ref_value, can_value, reasons)
    matched = normalized_label(ref_value) == normalized_label(can_value)
    return {
        "metric": "categorical",
        "status": "matched" if matched else "mismatched",
        "ref_value": ref_value,
        "can_value": can_value,
        "exact_match": matched,
    }


def observation_state(value: Any, side: str, *, enum_only: bool) -> tuple[str | None, str | None]:
    issue = side_issue(value, side, uncertain_is_unresolved=False)
    if issue:
        return None, issue
    normalized = normalized_label(value)
    if enum_only and normalized not in OBSERVATION_ENUM_VALUES:
        return None, f"{side}_invalid_value"
    if normalized == "uncertain":
        return None, f"{side}_uncertain"
    if normalized.startswith("uncertain"):
        return None, f"{side}_uncertain"
    if normalized == "not observed":
        return "not_observed", None
    if normalized.startswith(NONCANONICAL_NEGATIVE_PREFIXES):
        return None, f"{side}_noncanonical_negative"
    return "observed", None


def observation_quadrant(ref_state: str, can_state: str) -> str:
    if ref_state == "observed" and can_state == "observed":
        return "both_observed"
    if ref_state == "not_observed" and can_state == "not_observed":
        return "both_not_observed"
    if ref_state == "observed":
        return "ref_only_observed"
    return "can_only_observed"


def dynamic_wound_result(
    ref_value: Any,
    can_value: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    ref_state, ref_issue = observation_state(ref_value, "ref", enum_only=False)
    can_state, can_issue = observation_state(can_value, "can", enum_only=False)
    reasons = [reason for reason in (ref_issue, can_issue) if reason]
    if reasons:
        return unresolved_result("dynamic_bertscore", ref_value, can_value, reasons)
    quadrant = observation_quadrant(ref_state, can_state)
    result = {
        "metric": "dynamic_bertscore",
        "status": "state_only",
        "quadrant": quadrant,
        "ref_value": ref_value,
        "can_value": can_value,
    }
    if quadrant == "both_observed":
        queue_score(result, ref_value, can_value, pending)
    return result


def enum_quadrant_result(ref_value: Any, can_value: Any) -> dict[str, Any]:
    ref_state, ref_issue = observation_state(ref_value, "ref", enum_only=True)
    can_state, can_issue = observation_state(can_value, "can", enum_only=True)
    reasons = [reason for reason in (ref_issue, can_issue) if reason]
    if reasons:
        return unresolved_result("enum_quadrant", ref_value, can_value, reasons)
    return {
        "metric": "enum_quadrant",
        "status": "state_only",
        "quadrant": observation_quadrant(ref_state, can_state),
        "ref_value": ref_value,
        "can_value": can_value,
    }


def validate_text_array(value: Any, side: str) -> tuple[list[str] | None, list[str]]:
    if value is MISSING:
        return None, [f"{side}_missing"]
    if value is None:
        return None, [f"{side}_null"]
    if not isinstance(value, list):
        return None, [f"{side}_wrong_type"]
    reasons = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            reasons.append(f"{side}_item_{index}_wrong_type")
        elif not normalize_space(item):
            reasons.append(f"{side}_item_{index}_empty")
    if reasons:
        return None, reasons
    return value, []


def array_empty_status(ref_items: list[str], can_items: list[str]) -> str | None:
    if not ref_items and not can_items:
        return "both_empty"
    if not ref_items:
        return "ref_empty"
    if not can_items:
        return "can_empty"
    return None


def semantic_array_result(
    ref_value: Any,
    can_value: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    ref_items, ref_reasons = validate_text_array(ref_value, "ref")
    can_items, can_reasons = validate_text_array(can_value, "can")
    reasons = ref_reasons + can_reasons
    if reasons:
        return unresolved_result("item_level_bertscore", ref_value, can_value, reasons)
    empty_status = array_empty_status(ref_items, can_items)
    result: dict[str, Any] = {
        "metric": "item_level_bertscore",
        "status": empty_status or "pending_bertscore",
        "ref_items": ref_items,
        "can_items": can_items,
    }
    if empty_status:
        return result
    pair_scores: list[dict[str, Any]] = []
    for ref_index, ref_item in enumerate(ref_items):
        for can_index, can_item in enumerate(can_items):
            pair = {
                "ref_index": ref_index,
                "can_index": can_index,
                "ref_item": ref_item,
                "can_item": can_item,
            }
            queue_score(pair, ref_item, can_item, pending)
            pair_scores.append(pair)
    result["pair_scores"] = pair_scores
    result["ref_best_scores"] = []
    result["can_best_scores"] = []
    return result


def presence_array_result(ref_value: Any, can_value: Any) -> dict[str, Any]:
    ref_items, ref_reasons = validate_text_array(ref_value, "ref")
    can_items, can_reasons = validate_text_array(can_value, "can")
    reasons = ref_reasons + can_reasons
    if reasons:
        return unresolved_result("empty_nonempty", ref_value, can_value, reasons)
    status = array_empty_status(ref_items, can_items) or "both_nonempty"
    return {
        "metric": "empty_nonempty",
        "status": status,
        "ref_value": ref_items,
        "can_value": can_items,
    }


def binary_result(ref_value: Any, can_value: Any) -> dict[str, Any]:
    reasons = []
    for side, value in (("ref", ref_value), ("can", can_value)):
        if value is MISSING:
            reasons.append(f"{side}_missing")
        elif value is None:
            reasons.append(f"{side}_null")
        elif not isinstance(value, bool):
            reasons.append(f"{side}_wrong_type")
    if reasons:
        return unresolved_result("binary", ref_value, can_value, reasons)
    if ref_value and can_value:
        status = "both_true"
    elif not ref_value and not can_value:
        status = "both_false"
    elif ref_value:
        status = "ref_only_true"
    else:
        status = "can_only_true"
    return {
        "metric": "binary",
        "status": status,
        "ref_value": ref_value,
        "can_value": can_value,
        "exact_match": ref_value == can_value,
    }


def measurement_description_result(
    ref_value: Any,
    can_value: Any,
    ref_visible: Any,
    can_visible: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    visible_reasons = []
    for side, value in (("ref", ref_visible), ("can", can_visible)):
        if not isinstance(value, bool):
            visible_reasons.append(f"{side}_visible_invalid")
    if visible_reasons:
        return unresolved_result("conditional_bertscore", ref_value, can_value, visible_reasons)
    if not (ref_visible and can_visible):
        return {
            "metric": "conditional_bertscore",
            "status": "not_applicable",
            "ref_value": raw_json_value(ref_value),
            "can_value": raw_json_value(can_value),
            "reason": "measurement_tool_not_visible_on_both_sides",
        }
    return direct_text_result("conditional_bertscore", ref_value, can_value, pending)


def estimated_size_result(
    ref_value: Any,
    can_value: Any,
    ref_visible: Any,
    can_visible: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    reasons = [
        issue
        for issue in (
            side_issue(ref_value, "ref", uncertain_is_unresolved=False),
            side_issue(can_value, "can", uncertain_is_unresolved=False),
        )
        if issue
    ]
    if reasons:
        return unresolved_result("conditional_bertscore", ref_value, can_value, reasons)
    ref_unknown = normalized_label(ref_value) == "unknown"
    can_unknown = normalized_label(can_value) == "unknown"
    if ref_unknown or can_unknown:
        if ref_unknown and can_unknown:
            status = "both_unknown"
        elif ref_unknown:
            status = "ref_unknown"
        else:
            status = "can_unknown"
        return {
            "metric": "conditional_bertscore",
            "status": status,
            "ref_value": ref_value,
            "can_value": can_value,
        }
    if not (isinstance(ref_visible, bool) and isinstance(can_visible, bool)):
        return unresolved_result(
            "conditional_bertscore",
            ref_value,
            can_value,
            ["measurement_visible_invalid"],
        )
    if not (ref_visible and can_visible):
        return {
            "metric": "conditional_bertscore",
            "status": "not_applicable",
            "ref_value": ref_value,
            "can_value": can_value,
            "reason": "measurement_tool_not_visible_on_both_sides",
        }
    return direct_text_result("conditional_bertscore", ref_value, can_value, pending)


def schema_anomalies(
    caption: Any,
    side: str,
    validator: Any,
) -> list[dict[str, Any]]:
    if caption is MISSING:
        return [
            {
                "side": side,
                "field_path": "caption",
                "type": "missing_caption",
                "raw_value": "<missing>",
                "message": "Bundle does not contain caption.",
            }
        ]
    errors = sorted(validator.iter_errors(caption), key=lambda error: list(error.absolute_path))
    return [
        {
            "side": side,
            "field_path": "caption"
            + ("." + ".".join(str(part) for part in error.absolute_path) if error.absolute_path else ""),
            "type": f"schema_{error.validator}_error",
            "raw_value": error.instance,
            "message": error.message,
        }
        for error in errors
    ]


def custom_anomalies(caption: Any, bundle: dict[str, Any], side: str) -> list[dict[str, Any]]:
    if not isinstance(caption, dict):
        return []
    anomalies: list[dict[str, Any]] = []

    def add(path: str, anomaly_type: str, raw_value: Any, message: str) -> None:
        anomalies.append(
            {
                "side": side,
                "field_path": path,
                "type": anomaly_type,
                "raw_value": raw_json_value(raw_value),
                "message": message,
            }
        )

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            seen: set[str] = set()
            for index, item in enumerate(value):
                item_path = f"{path}.{index}"
                if not isinstance(item, str):
                    add(item_path, "array_item_wrong_type", item, "Array item must be a string.")
                    continue
                normalized = normalized_label(item)
                if not normalized:
                    add(item_path, "array_item_empty", item, "Array item must not be empty.")
                elif normalized in seen:
                    add(item_path, "array_item_duplicate", item, "Duplicate array item after normalization.")
                seen.add(normalized)
        elif isinstance(value, str) and not normalize_space(value):
            add(path, "empty_string", value, "String field is empty.")

    visit(caption, "caption")

    for field in DYNAMIC_WOUND_FIELDS:
        value = nested_value(caption, "wound_features", field)
        if not isinstance(value, str):
            continue
        normalized = normalized_label(value)
        if normalized != "not observed" and normalized.startswith(NONCANONICAL_NEGATIVE_PREFIXES):
            add(
                f"caption.wound_features.{field}",
                "noncanonical_observation_state",
                value,
                'Use the canonical value "not observed" for an absent feature.',
            )

    target = nested_value(caption, "category_specific_check", "target_category")
    assigned = nested_value(bundle, "metadata", "raw_category")
    if isinstance(target, str) and isinstance(assigned, str):
        if normalized_label(target) != normalized_label(assigned):
            add(
                "caption.category_specific_check.target_category",
                "assigned_value_mismatch",
                target,
                f"target_category differs from assigned category: {assigned}",
            )

    measurement = nested_value(caption, "image_observation", "measurement_tool")
    if isinstance(measurement, dict):
        visible = measurement.get("visible", MISSING)
        description = measurement.get("description", MISSING)
        estimated_size = measurement.get("estimated_size", MISSING)
        if visible is False and description not in (None, MISSING):
            add(
                "caption.image_observation.measurement_tool.description",
                "measurement_state_inconsistent",
                description,
                "description must be null when measurement_tool.visible is false.",
            )
        if visible is False and isinstance(estimated_size, str):
            if normalized_label(estimated_size) != "unknown":
                add(
                    "caption.image_observation.measurement_tool.estimated_size",
                    "measurement_state_inconsistent",
                    estimated_size,
                    "estimated_size must be unknown when measurement_tool.visible is false.",
                )
        if visible is True and description is None:
            add(
                "caption.image_observation.measurement_tool.description",
                "measurement_state_inconsistent",
                description,
                "description should identify the visible measurement tool.",
            )
    return anomalies


def block_metadata(
    filename: str,
    ref_path: Path,
    can_path: Path,
    ref_bundle: dict[str, Any],
    can_bundle: dict[str, Any],
) -> dict[str, Any]:
    ref_metadata = ref_bundle.get("metadata", {}) if isinstance(ref_bundle.get("metadata"), dict) else {}
    can_metadata = can_bundle.get("metadata", {}) if isinstance(can_bundle.get("metadata"), dict) else {}
    category = (
        can_metadata.get("raw_category")
        or ref_metadata.get("raw_category")
        or nested_value(can_bundle, "caption", "category_specific_check", "target_category")
        or nested_value(ref_bundle, "caption", "category_specific_check", "target_category")
        or "unknown"
    )
    return {
        "block_id": filename,
        "category": raw_json_value(category),
        "ref_file": str((ref_path / filename).resolve()),
        "can_file": str((can_path / filename).resolve()),
        "ref_provider": ref_metadata.get("provider"),
        "can_provider": can_metadata.get("provider"),
        "ref_model": ref_metadata.get("model"),
        "can_model": can_metadata.get("model"),
        "ref_mode": ref_metadata.get("mode"),
        "can_mode": can_metadata.get("mode"),
    }


def build_block(
    filename: str,
    ref_bundle: dict[str, Any],
    can_bundle: dict[str, Any],
    ref_dir: Path,
    can_dir: Path,
    validator: Any,
    pending: list[tuple[dict[str, Any], str, str]],
) -> dict[str, Any]:
    ref_caption = caption_for(ref_bundle)
    can_caption = caption_for(can_bundle)
    fields: dict[str, Any] = {
        "image_observation": {},
        "wound_features": {},
        "category_specific_check": {},
        "uncertainty": {},
    }

    ref_body_site = nested_value(ref_caption, "image_observation", "body_site")
    can_body_site = nested_value(can_caption, "image_observation", "body_site")
    fields["image_observation"]["body_site"] = direct_text_result(
        "bertscore", ref_body_site, can_body_site, pending
    )
    fields["image_observation"]["wound_presence"] = categorical_result(
        nested_value(ref_caption, "image_observation", "wound_presence"),
        nested_value(can_caption, "image_observation", "wound_presence"),
    )
    fields["image_observation"]["visual_summary"] = direct_text_result(
        "bertscore",
        nested_value(ref_caption, "image_observation", "visual_summary"),
        nested_value(can_caption, "image_observation", "visual_summary"),
        pending,
    )

    ref_measurement = nested_value(ref_caption, "image_observation", "measurement_tool")
    can_measurement = nested_value(can_caption, "image_observation", "measurement_tool")
    ref_visible = nested_value(ref_measurement, "visible")
    can_visible = nested_value(can_measurement, "visible")
    ref_description = nested_value(ref_measurement, "description")
    can_description = nested_value(can_measurement, "description")
    ref_size = nested_value(ref_measurement, "estimated_size")
    can_size = nested_value(can_measurement, "estimated_size")
    fields["image_observation"]["measurement_tool"] = {
        "visible": binary_result(ref_visible, can_visible),
        "description": measurement_description_result(
            ref_description, can_description, ref_visible, can_visible, pending
        ),
        "estimated_size": estimated_size_result(
            ref_size, can_size, ref_visible, can_visible, pending
        ),
    }

    for field in DYNAMIC_WOUND_FIELDS:
        fields["wound_features"][field] = dynamic_wound_result(
            nested_value(ref_caption, "wound_features", field),
            nested_value(can_caption, "wound_features", field),
            pending,
        )
    for field in ENUM_WOUND_FIELDS:
        fields["wound_features"][field] = enum_quadrant_result(
            nested_value(ref_caption, "wound_features", field),
            nested_value(can_caption, "wound_features", field),
        )

    for field in SEMANTIC_ARRAY_FIELDS:
        fields["category_specific_check"][field] = semantic_array_result(
            nested_value(ref_caption, "category_specific_check", field),
            nested_value(can_caption, "category_specific_check", field),
            pending,
        )
    for section, field in PRESENCE_ARRAY_FIELDS:
        fields[section][field] = presence_array_result(
            nested_value(ref_caption, section, field),
            nested_value(can_caption, section, field),
        )

    anomalies = (
        schema_anomalies(ref_caption, "ref", validator)
        + schema_anomalies(can_caption, "can", validator)
        + custom_anomalies(ref_caption, ref_bundle, "ref")
        + custom_anomalies(can_caption, can_bundle, "can")
    )
    return {
        "fields": fields,
        "anomalies": anomalies,
        "metadata": block_metadata(filename, ref_dir, can_dir, ref_bundle, can_bundle),
    }


def load_bert_score_runtime() -> tuple[Any, str, int, Path, dict[str, str]]:
    try:
        import bert_score
        import torch
        import transformers
        from bert_score import score as bertscore_score
        from bert_score.utils import lang2model, model2layers
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: bert_score. Install project requirements before scoring."
        ) from exc
    model_type = lang2model[LANGUAGE]
    if model_type != EXPECTED_MODEL:
        raise RuntimeError(
            f"Unexpected upstream English model: expected {EXPECTED_MODEL}, got {model_type}"
        )
    num_layers = model2layers[model_type]
    baseline_path = (
        Path(bert_score.__file__).resolve().parent
        / "rescale_baseline"
        / LANGUAGE
        / f"{model_type}.tsv"
    )
    if not baseline_path.is_file():
        raise SystemExit(f"Official BERTScore baseline is missing: {baseline_path}")
    versions = {
        "bert_score_version": package_version("bert-score"),
        "transformers_version": str(getattr(transformers, "__version__", "unknown")),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
    }
    return bertscore_score, model_type, num_layers, baseline_path, versions


def compute_bert_scores(
    candidates: list[str],
    references: list[str],
    options: ScoreOptions,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    bertscore_score, model_type, num_layers, baseline_path, versions = load_bert_score_runtime()
    kwargs: dict[str, Any] = {
        "lang": LANGUAGE,
        "idf": options.idf,
        "rescale_with_baseline": True,
        "return_hash": True,
        "batch_size": options.batch_size,
        "verbose": False,
    }
    if options.device:
        kwargs["device"] = options.device
    if options.use_fast_tokenizer:
        kwargs["use_fast_tokenizer"] = True
    (precision, recall, f1), hash_code = bertscore_score(candidates, references, **kwargs)
    return (
        [float(value) for value in precision],
        [float(value) for value in recall],
        [float(value) for value in f1],
        {
            "language": LANGUAGE,
            "model": model_type,
            "num_layers": num_layers,
            "baseline_source": "bert_score_package_default",
            "baseline_path": str(baseline_path),
            "bertscore_hash": str(hash_code),
            "score_representation": "official_baseline_rescaled",
            **versions,
        },
    )


def apply_pending_scores(
    pending: list[tuple[dict[str, Any], str, str]],
    options: ScoreOptions,
    score_function: ScoreFunction,
) -> dict[str, Any]:
    if not pending:
        return {
            "language": LANGUAGE,
            "model": EXPECTED_MODEL,
            "score_representation": "official_baseline_rescaled",
            "status": "not_loaded_no_scoreable_pairs",
        }
    candidates = [can_text for _, can_text, _ in pending]
    references = [ref_text for _, _, ref_text in pending]
    precision, recall, f1, config = score_function(candidates, references, options)
    if not (len(precision) == len(recall) == len(f1) == len(pending)):
        raise RuntimeError("BERTScore backend returned an unexpected number of results.")
    for index, (target, _, _) in enumerate(pending):
        target["status"] = "scored"
        target["bertscore_precision"] = precision[index]
        target["bertscore_recall"] = recall[index]
        target["bertscore_f1"] = f1[index]
    return config


def finalize_semantic_arrays(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("metric") == "item_level_bertscore" and value.get("status") == "pending_bertscore":
            pairs = value["pair_scores"]
            ref_count = len(value["ref_items"])
            can_count = len(value["can_items"])
            value["ref_best_scores"] = [
                min(
                    (pair for pair in pairs if pair["ref_index"] == ref_index),
                    key=lambda pair: (-pair["bertscore_f1"], pair["can_index"]),
                )
                for ref_index in range(ref_count)
            ]
            value["can_best_scores"] = [
                min(
                    (pair for pair in pairs if pair["can_index"] == can_index),
                    key=lambda pair: (-pair["bertscore_f1"], pair["ref_index"]),
                )
                for can_index in range(can_count)
            ]
            value["status"] = "scored"
        else:
            for item in value.values():
                finalize_semantic_arrays(item)
    elif isinstance(value, list):
        for item in value:
            finalize_semantic_arrays(item)


def build_database(
    ref_dir: Path,
    can_dir: Path,
    options: ScoreOptions,
    *,
    limit: int = 0,
    score_function: ScoreFunction = compute_bert_scores,
    validator: Any | None = None,
) -> dict[str, Any]:
    ref_bundles, ref_load_errors = read_caption_files(ref_dir)
    can_bundles, can_load_errors = read_caption_files(can_dir)
    ref_names = set(ref_bundles)
    can_names = set(can_bundles)
    matched = sorted(ref_names & can_names, key=str.casefold)
    if limit > 0:
        matched = matched[:limit]
    if not matched:
        raise SystemExit("No matched valid JSON filenames were found between ref and can.")

    if validator is None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: jsonschema. Install project requirements before scoring."
            ) from exc
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
    pending: list[tuple[dict[str, Any], str, str]] = []
    blocks = {
        filename: build_block(
            filename,
            ref_bundles[filename],
            can_bundles[filename],
            ref_dir,
            can_dir,
            validator,
            pending,
        )
        for filename in matched
    }
    bertscore_config = apply_pending_scores(pending, options, score_function)
    finalize_semantic_arrays(blocks)
    for block in blocks.values():
        block["metadata"]["bertscore"] = bertscore_config

    return {
        "schema_version": SCHEMA_VERSION,
        "blocks": blocks,
        "report_metadata": {
            "ref_directory": str(ref_dir.resolve()),
            "can_directory": str(can_dir.resolve()),
            "ref_root_name": source_root_name(ref_dir),
            "can_root_name": source_root_name(can_dir),
            "matched_block_count": len(matched),
            "ref_only_file_count": len(ref_names - can_names),
            "can_only_file_count": len(can_names - ref_names),
            "ref_only_files": sorted(ref_names - can_names, key=str.casefold),
            "can_only_files": sorted(can_names - ref_names, key=str.casefold),
            "ref_load_errors": ref_load_errors,
            "can_load_errors": can_load_errors,
            "limit": limit,
            "bertscore": bertscore_config,
            "created_at": now_iso(),
        },
    }


def render_json_with_compact_field_results(data: Any) -> str:
    replacements: dict[str, str] = {}

    def replace_results(value: Any) -> Any:
        if isinstance(value, dict) and "metric" in value:
            marker = f"__COMPACT_FIELD_RESULT_{uuid.uuid4().hex}__"
            replacements[marker] = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            return marker
        if isinstance(value, dict):
            return {key: replace_results(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_results(item) for item in value]
        return value

    rendered = json.dumps(replace_results(data), ensure_ascii=False, indent=2, allow_nan=False)
    for marker, compact in replacements.items():
        rendered = rendered.replace(json.dumps(marker), compact)
    return rendered + "\n"


def atomic_write_database(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_json_with_compact_field_results(data)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    ref_dir = Path(args.ref_dir)
    can_dir = Path(args.can_dir)
    options = ScoreOptions(
        idf=args.idf,
        batch_size=args.batch_size,
        device=args.device,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )
    database = build_database(ref_dir, can_dir, options, limit=args.limit)
    destination = output_path(ref_dir, can_dir, Path(args.output_dir))
    atomic_write_database(destination, database)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
