import unittest
from unittest import mock
import sys

for dependency in ("httpx", "jsonschema", "ollama"):
    sys.modules.setdefault(dependency, mock.MagicMock())
import pipeline


class PipelineModeTests(unittest.TestCase):
    def test_parse_modes_supports_simple_only(self) -> None:
        self.assertEqual(pipeline.parse_modes("simple"), ("simple",))

    def test_parse_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            pipeline.parse_modes("simple,unknown")


class SaasLimitTests(unittest.TestCase):
    def test_cli_quota_overrides_planning_defaults(self) -> None:
        limits = pipeline.saas_limits_for_model(
            "gemini-3.1-flash-lite", rpm=30, rpd=1000, tpm=500_000
        )

        self.assertEqual(limits["rpm"], 30)
        self.assertEqual(limits["rpd"], 1000)
        self.assertEqual(limits["tpm"], 500_000)
        self.assertEqual(limits["source"], "CLI override from project quota")

    def test_simple_only_estimate_has_no_full_combination_alias(self) -> None:
        probe_results = {
            "simple": {
                "probe": {
                    "request_count": 1,
                    "duration_sec": 10.0,
                    "retry_count": 0,
                    "rate_limit_errors": 0,
                    "quota_errors": 0,
                    "request_delay_sec": 4.0,
                    "token_count": {"input": 1000, "output": 100, "total": 1100},
                }
            }
        }

        estimate = pipeline.estimate_gemini_consumption(
            probe_results=probe_results,
            selected_count=431,
            model="gemini-3.1-flash-lite",
            rpm=15,
            rpd=500,
            tpm=250_000,
        )

        self.assertEqual(set(estimate["modes"]), {"simple"})
        self.assertEqual(estimate["combined_selected_modes"]["estimated_request_count"], 431)
        self.assertNotIn("combined_simple_full", estimate)


if __name__ == "__main__":
    unittest.main()
