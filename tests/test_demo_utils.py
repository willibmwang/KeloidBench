"""Correctness tests for the interactive KeloidBench demonstration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from demo_utils import (  # noqa: E402
    CURATED_EXAMPLES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    deterministic_summary,
    expression_qc,
    generate_grounded_summary,
    normalize_expression_table,
    program_coverage,
    score_sample,
)


class DemoInputTests(unittest.TestCase):
    def test_expression_normalization_collapses_duplicate_symbols(self):
        raw = pd.DataFrame([[1.0, 3.0, 7.0]], index=["sample"], columns=["postn", "POSTN", "KRT1"])
        normalized = normalize_expression_table(raw)
        self.assertEqual(normalized.columns.tolist(), ["KRT1", "POSTN"])
        self.assertEqual(float(normalized.loc["sample", "POSTN"]), 2.0)

    def test_program_coverage_marks_tiny_panel_unusable(self):
        coverage = program_coverage(["POSTN"])
        self.assertFalse(bool(coverage.loc[coverage["program"] == "ecm_profibrotic_rank", "usable"].iloc[0]))

    def test_nontransferable_accession_is_visible(self):
        expression = pd.DataFrame([[1.0]], columns=["POSTN"], index=["s"])
        qc = expression_qc(expression, accession="GSE173900")
        self.assertEqual(qc["transfer_status"], "non-transferable")


class DemoScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from demo_utils import load_product_bundle, load_feature_views

        _, cls.bundle = load_product_bundle()
        cls.features = load_feature_views()[cls.bundle["meta"]["feature_set"]]

    def test_live_score_obeys_abstention_policy(self):
        row = self.features.iloc[[0]]
        result = score_sample(
            row,
            self.bundle,
            is_expression=False,
            confidence_threshold=0.99,
        )
        self.assertEqual(result["decision"], "abstain")
        self.assertIn("prob_keloid", result)
        self.assertTrue(result["contributions"])

    def test_curated_examples_exercise_distinct_product_states(self):
        expected = {
            "confident_keloid": "keloid",
            "confident_unaffected": "non_keloid",
            "abstention": "abstain",
        }
        for slug, decision in expected.items():
            sample_id = CURATED_EXAMPLES[slug]["sample_id"]
            result = score_sample(
                self.features.loc[[sample_id]],
                self.bundle,
                is_expression=False,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            )
            self.assertEqual(result["decision"], decision, slug)

    def test_summary_never_calls_result_a_diagnosis(self):
        packet = {
            "prediction": {
                "decision": "abstain",
                "confidence": 0.52,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "prob_keloid": 0.52,
            },
            "quality_control": {
                "transfer_status": "unverified",
                "transfer_note": "Unseen cohort.",
            },
            "top_supporting_contributions": [],
        }
        summary = deterministic_summary(packet)
        self.assertIn("abstained", summary)
        self.assertIn("must not be interpreted as a clinical diagnosis", summary)

    def test_remote_narrative_redacts_sample_identifier(self):
        packet = {
            "sample": {"sample_id": "sensitive-id", "accession": "uploaded"},
            "prediction": {"decision": "abstain"},
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "Grounded summary"}}]}
        with patch.dict(
            "os.environ",
            {
                "KELOIDBENCH_LLM_API_URL": "https://example.invalid/v1/chat/completions",
                "KELOIDBENCH_LLM_MODEL": "test-model",
            },
            clear=False,
        ), patch("demo_utils.requests.post", return_value=response) as post:
            text, mode = generate_grounded_summary(packet)
        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(text, "Grounded summary")
        self.assertEqual(mode, "grounded_llm")
        self.assertNotIn("sensitive-id", prompt)
        self.assertIn("redacted", prompt)


if __name__ == "__main__":
    unittest.main()
