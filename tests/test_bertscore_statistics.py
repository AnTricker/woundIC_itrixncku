import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bertscore_statistics.py"
SPEC = importlib.util.spec_from_file_location("bertscore_statistics", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def score(value):
    return {
        "status": "scored",
        "bertscore_precision": value,
        "bertscore_recall": value,
        "bertscore_f1": value,
    }


def set_nested(root, path, value):
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def block(category="abrasions"):
    return {"fields": {}, "anomalies": [], "metadata": {"category": category}}


class StatisticsTests(unittest.TestCase):
    def test_chart_labels_use_compact_values(self):
        self.assertEqual(
            MODULE.bert_bar_label({"mean": 0.456, "std": 0.123}),
            r"$\mathbf{0.46}$ ± 0.12",
        )
        self.assertEqual(
            MODULE.bert_bar_label({"mean": 0.456, "std": None}),
            r"$\mathbf{0.46}$ ± N/A",
        )
        row = {"counts": {"can_missed": 2}, "rates": {"can_missed": 0.125}}
        self.assertEqual(MODULE.coverage_segment_label(row, "can_missed"), "2 (12.5%)")

    def test_array_uses_per_image_macro_average(self):
        first = block()
        second = block()
        set_nested(
            first["fields"],
            "category_specific_check.observed_supporting_features",
            {
                "status": "scored",
                "ref_best_scores": [score(0.0), score(1.0)],
                "can_best_scores": [score(0.25)],
            },
        )
        set_nested(
            second["fields"],
            "category_specific_check.observed_supporting_features",
            {
                "status": "scored",
                "ref_best_scores": [score(1.0)],
                "can_best_scores": [score(0.75)],
            },
        )
        aggregates = MODULE.aggregate_bertscore(
            {"abrasions": [("a.json", first), ("b.json", second)]}
        )["abrasions"]
        ref_row = next(
            row
            for row in aggregates
            if row["field_path"].endswith("observed_supporting_features")
            and row["direction"] == "ref_best"
        )
        can_row = next(
            row
            for row in aggregates
            if row["field_path"].endswith("observed_supporting_features")
            and row["direction"] == "can_best"
        )
        self.assertEqual(ref_row["f1"]["n"], 2)
        self.assertAlmostEqual(ref_row["f1"]["mean"], 0.75)
        self.assertAlmostEqual(can_row["f1"]["mean"], 0.5)

    def test_scalar_mean_and_sample_standard_deviation(self):
        first = block()
        second = block()
        set_nested(first["fields"], "image_observation.body_site", score(0.2))
        set_nested(second["fields"], "image_observation.body_site", score(0.4))
        rows = MODULE.aggregate_bertscore(
            {"abrasions": [("a.json", first), ("b.json", second)]}
        )["abrasions"]
        row = next(item for item in rows if item["field_path"] == "image_observation.body_site")
        self.assertAlmostEqual(row["precision"]["mean"], 0.3)
        self.assertAlmostEqual(row["precision"]["std"], math.sqrt(0.02))
        self.assertEqual(row["precision"]["n"], 2)

    def test_global_ylim_includes_negative_error_bar_and_zero(self):
        aggregates = {
            "abrasions": [
                {
                    "precision": {"mean": -0.2, "std": 0.15, "n": 2},
                    "recall": {"mean": 0.4, "std": 0.1, "n": 2},
                    "f1": {"mean": 0.1, "std": None, "n": 1},
                }
            ]
        }
        lower, upper = MODULE.global_bertscore_ylim(aggregates)
        self.assertLess(lower, -0.35)
        self.assertGreater(upper, 0.5)
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)

    def test_coverage_mapping_and_category_denominator(self):
        self.assertEqual(
            MODULE.coverage_outcome(
                {
                    "metric": "dynamic_bertscore",
                    "status": "scored",
                    "quadrant": "both_observed",
                }
            ),
            "both_present",
        )
        self.assertEqual(
            MODULE.coverage_outcome(
                {
                    "metric": "item_level_bertscore",
                    "status": "can_empty",
                }
            ),
            "can_missed",
        )
        self.assertEqual(
            MODULE.coverage_outcome(
                {"metric": "empty_nonempty", "status": "ref_empty"}
            ),
            "can_extra",
        )

        first = block()
        second = block()
        set_nested(
            first["fields"],
            "wound_features.shape_pattern",
            {
                "metric": "dynamic_bertscore",
                "status": "scored",
                "quadrant": "both_observed",
            },
        )
        set_nested(
            second["fields"],
            "wound_features.shape_pattern",
            {
                "metric": "dynamic_bertscore",
                "status": "state_only",
                "quadrant": "ref_only_observed",
            },
        )
        rows = MODULE.aggregate_coverage(
            {"abrasions": [("a.json", first), ("b.json", second)]}
        )["abrasions"]
        row = next(item for item in rows if item["field_path"] == "wound_features.shape_pattern")
        self.assertEqual(row["total"], 2)
        self.assertEqual(row["counts"]["both_present"], 1)
        self.assertEqual(row["counts"]["can_missed"], 1)
        self.assertAlmostEqual(sum(row["rates"].values()), 1.0)

    def test_wound_presence_excludes_unresolved_from_rate(self):
        items = []
        for index, status in enumerate(("matched", "matched", "mismatched", "unresolved")):
            item = block()
            set_nested(
                item["fields"],
                "image_observation.wound_presence",
                {"status": status},
            )
            items.append((f"{index}.json", item))
        row = MODULE.aggregate_wound_presence({"abrasions": items})[0]
        self.assertEqual(row["unresolved"], 1)
        self.assertAlmostEqual(row["exact_match_rate"], 2 / 3)

    def test_anomaly_path_resolution_and_compliance_levels(self):
        first = block()
        second = block()
        first["anomalies"] = [
            {
                "side": "can",
                "field_path": "caption.uncertainty",
                "type": "schema_required_error",
                "message": "'cannot_determine' is a required property",
                "raw_value": {},
            },
            {
                "side": "can",
                "field_path": "caption.wound_features",
                "type": "schema_additionalProperties_error",
                "message": "Additional properties are not allowed ('foo' was unexpected)",
                "raw_value": {},
            },
            {
                "side": "ref",
                "field_path": "caption.wound_features.texture",
                "type": "noncanonical_observation_state",
                "message": "Use not observed",
                "raw_value": "none visible",
            },
        ]
        second["anomalies"] = [
            {
                "side": "can",
                "field_path": "caption.category_specific_check.target_category",
                "type": "assigned_value_mismatch",
                "message": "mismatch",
                "raw_value": "cut",
            }
        ]
        blocks = {"a.json": first, "b.json": second}
        can_records = MODULE.normalized_anomaly_records(blocks, "can")
        ref_records = MODULE.normalized_anomaly_records(blocks, "ref")
        paths = {record["field_path"] for record in can_records}
        self.assertIn("caption.uncertainty.cannot_determine", paths)
        self.assertIn("caption.wound_features.foo", paths)
        self.assertEqual(len(ref_records), 1)
        summary = {row["level"]: row for row in MODULE.compliance_summary(blocks, can_records)}
        self.assertEqual(summary["strict_schema"]["failed"], 1)
        self.assertEqual(summary["extended_contract"]["failed"], 2)

    def test_report_naming_and_relative_assets(self):
        source = Path(
            "runs/bertscore_field_scores/"
            "ref_saas__can_gemma_caption_field_scores.json"
        )
        report, assets = MODULE.report_paths(source, Path("runs/bertscore_statistics"))
        self.assertEqual(report.name, "ref_saas__can_gemma_statistics.md")
        self.assertEqual(assets.name, "ref_saas__can_gemma_statistics_assets")
        image = assets / "bertscore_abrasions.png"
        self.assertEqual(
            MODULE.relative_image_path(report, image),
            "ref_saas__can_gemma_statistics_assets/bertscore_abrasions.png",
        )

    def test_atomic_markdown_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            MODULE.atomic_write_text(path, "first")
            MODULE.atomic_write_text(path, "second")
            self.assertEqual(path.read_text(encoding="utf-8"), "second")


if __name__ == "__main__":
    unittest.main()
