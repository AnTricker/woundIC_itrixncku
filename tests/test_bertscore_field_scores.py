import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bertscore_field_scores.py"
SPEC = importlib.util.spec_from_file_location("bertscore_field_scores", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EmptyValidator:
    def iter_errors(self, value):
        return []


def fake_score(candidates, references, options):
    values = []
    for candidate, reference in zip(candidates, references):
        can_tokens = set(candidate.casefold().split())
        ref_tokens = set(reference.casefold().split())
        union = can_tokens | ref_tokens
        values.append(len(can_tokens & ref_tokens) / len(union) if union else 0.0)
    return values, values, values, {
        "language": "en",
        "model": "fake-roberta-large",
        "bertscore_hash": "fake-hash",
        "score_representation": "official_baseline_rescaled",
    }


def caption_bundle(*, model, candidate=False):
    caption = {
        "image_observation": {
            "body_site": "lower left arm" if candidate else "left forearm",
            "wound_presence": "open wound",
            "visual_summary": (
                "A shallow abrasion with a red raw surface."
                if candidate
                else "A superficial abrasion with an erythematous wound bed."
            ),
            "measurement_tool": {
                "visible": False,
                "description": None,
                "estimated_size": "unknown",
            },
        },
        "wound_features": {
            "shape_pattern": "irregular linear pattern" if candidate else "irregular streaked pattern",
            "edges_margins": "irregular margins" if candidate else "not observed",
            "wound_bed": "red raw wound bed" if candidate else "raw erythematous surface",
            "color": "erythematous red" if candidate else "red erythematous",
            "texture": "not observed",
            "fluid_exudate_bleeding": "not observed" if candidate else "pinpoint bleeding observed",
            "periwound_skin": "not observed",
            "foreign_material_debris": "not observed",
            "necrosis_eschar": "observed" if candidate else "uncertain",
        },
        "category_specific_check": {
            "target_category": "abrasions",
            "observed_supporting_features": (
                ["pinpoint bleeding", "superficial skin loss"]
                if candidate
                else ["superficial skin loss", "pinpoint bleeding"]
            ),
            "expected_but_not_observed": [] if candidate else ["embedded dirt"],
            "differential_visual_conflicts": [],
        },
        "uncertainty": {
            "low_confidence_regions": ["blurred upper margin"] if candidate else [],
            "cannot_determine": [],
        },
        "safety_scope": "Visual description only; not a medical diagnosis.",
    }
    return {
        "metadata": {
            "raw_category": "abrasions",
            "provider": "local" if candidate else "saas",
            "model": model,
            "mode": "simple",
        },
        "caption": caption,
    }


class FieldScoreTests(unittest.TestCase):
    def test_dynamic_and_enum_state_routing(self):
        pending = []
        scored = MODULE.dynamic_wound_result("raw surface", "red surface", pending)
        self.assertEqual(scored["quadrant"], "both_observed")
        self.assertEqual(scored["status"], "pending_bertscore")
        self.assertEqual(len(pending), 1)

        state_only = MODULE.dynamic_wound_result("not observed", "irregular", [])
        self.assertEqual(state_only["quadrant"], "can_only_observed")
        self.assertEqual(state_only["status"], "state_only")

        unresolved = MODULE.enum_quadrant_result("not observed", "none visible")
        self.assertEqual(unresolved["status"], "unresolved")
        self.assertIn("can_invalid_value", unresolved["unresolved_reasons"])

        uncertain = MODULE.dynamic_wound_result("uncertain", "not observed", [])
        self.assertEqual(uncertain["status"], "unresolved")
        self.assertIn("ref_uncertain", uncertain["unresolved_reasons"])

        noncanonical = MODULE.dynamic_wound_result("not observed", "none visible", [])
        self.assertEqual(noncanonical["status"], "unresolved")
        self.assertIn("can_noncanonical_negative", noncanonical["unresolved_reasons"])

    def test_semantic_array_scores_every_pair_and_best_per_item(self):
        pending = []
        result = MODULE.semantic_array_result(
            ["superficial skin loss", "pinpoint bleeding"],
            ["pinpoint bleeding", "superficial skin loss"],
            pending,
        )
        self.assertEqual(len(pending), 4)
        MODULE.apply_pending_scores(pending, MODULE.ScoreOptions(), fake_score)
        MODULE.finalize_semantic_arrays(result)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(len(result["pair_scores"]), 4)
        self.assertEqual([row["can_index"] for row in result["ref_best_scores"]], [1, 0])
        self.assertEqual([row["ref_index"] for row in result["can_best_scores"]], [1, 0])

        can_empty = MODULE.semantic_array_result(["embedded dirt"], [], [])
        self.assertEqual(can_empty["status"], "can_empty")

    def test_measurement_conditional_rules(self):
        pending = []
        description = MODULE.measurement_description_result(
            "ruler", "measurement ruler", True, True, pending
        )
        self.assertEqual(description["status"], "pending_bertscore")

        not_applicable = MODULE.measurement_description_result(None, None, False, False, [])
        self.assertEqual(not_applicable["status"], "not_applicable")

        both_unknown = MODULE.estimated_size_result("unknown", "unknown", False, False, [])
        self.assertEqual(both_unknown["status"], "both_unknown")

    def test_build_database_preserves_groups_and_filename_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref_dir = root / "runs" / "ref_root" / "outputs" / "saas" / "simple"
            can_dir = root / "runs" / "can_root" / "outputs" / "local" / "simple"
            ref_dir.mkdir(parents=True)
            can_dir.mkdir(parents=True)
            filename = "abrasions (27).json"
            (ref_dir / filename).write_text(
                json.dumps(caption_bundle(model="ref-model")), encoding="utf-8"
            )
            (can_dir / filename).write_text(
                json.dumps(caption_bundle(model="can-model", candidate=True)), encoding="utf-8"
            )

            database = MODULE.build_database(
                ref_dir,
                can_dir,
                MODULE.ScoreOptions(),
                score_function=fake_score,
                validator=EmptyValidator(),
            )
            self.assertEqual(list(database["blocks"]), [filename])
            block = database["blocks"][filename]
            self.assertEqual(
                list(block), ["fields", "anomalies", "metadata"]
            )
            self.assertEqual(
                list(block["fields"]),
                ["image_observation", "wound_features", "category_specific_check", "uncertainty"],
            )
            self.assertEqual(
                block["fields"]["wound_features"]["edges_margins"]["quadrant"],
                "can_only_observed",
            )
            self.assertEqual(
                block["fields"]["category_specific_check"]["expected_but_not_observed"]["status"],
                "can_empty",
            )
            self.assertEqual(
                block["fields"]["uncertainty"]["low_confidence_regions"]["status"],
                "ref_empty",
            )
            self.assertEqual(database["report_metadata"]["ref_root_name"], "ref_root")
            self.assertEqual(database["report_metadata"]["can_root_name"], "can_root")

    def test_anomaly_ledger_records_duplicates_target_and_measurement(self):
        bundle = caption_bundle(model="can-model", candidate=True)
        caption = bundle["caption"]
        caption["category_specific_check"]["target_category"] = "cuts"
        caption["category_specific_check"]["observed_supporting_features"] = ["red", " RED "]
        caption["wound_features"]["texture"] = "none visible"
        caption["image_observation"]["measurement_tool"] = {
            "visible": False,
            "description": "ruler",
            "estimated_size": "2 cm",
        }
        anomalies = MODULE.custom_anomalies(caption, bundle, "can")
        types = {item["type"] for item in anomalies}
        self.assertIn("assigned_value_mismatch", types)
        self.assertIn("array_item_duplicate", types)
        self.assertIn("measurement_state_inconsistent", types)
        self.assertIn("noncanonical_observation_state", types)

    def test_compact_leaf_json_and_atomic_overwrite(self):
        data = {
            "schema_version": "first",
            "blocks": {
                "x.json": {
                    "fields": {
                        "image_observation": {
                            "body_site": {
                                "metric": "bertscore",
                                "status": "scored",
                                "ref_value": "arm",
                                "can_value": "forearm",
                                "bertscore_f1": 0.5,
                            }
                        }
                    },
                    "anomalies": [],
                    "metadata": {},
                }
            },
        }
        rendered = MODULE.render_json_with_compact_field_results(data)
        self.assertIn(
            '"body_site": {"metric":"bertscore","status":"scored"', rendered
        )
        self.assertEqual(json.loads(rendered), data)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.json"
            MODULE.atomic_write_database(path, data)
            data["schema_version"] = "second"
            MODULE.atomic_write_database(path, data)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], "second")

    def test_output_filename_contains_ref_and_can_root_names(self):
        ref = Path("runs/ref_run/outputs/saas/simple")
        can = Path("runs/can_run/outputs/local/simple")
        path = MODULE.output_path(ref, can, Path("scores"))
        self.assertEqual(
            path.name,
            "ref_ref_run__can_can_run_caption_field_scores.json",
        )


if __name__ == "__main__":
    unittest.main()
