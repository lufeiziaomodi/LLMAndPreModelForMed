import json
import unittest

from apps.run_external_direct_eval import _redact_config_secrets
from evaluate.cli.workflows import _summarize_judge_results
from evaluate.explanation.faithfulness import compute_faithfulness_for_sample
from evaluate.explanation.processor import (
    compute_query_coverage,
    normalize_explanation_output,
)


class ExplanationEvaluationTests(unittest.TestCase):
    def test_biomedical_identifier_is_not_corrupted_by_list_cleanup(self):
        output = json.dumps(
            [{
                "analysis_steps": "1. CYP2C19) remains an explicit anchor. 2. CYP2C9 is shared.",
                "mechanism_summary": "CYP2C19) and CYP2C9 are retained.",
                "representative_pairs": ["A -> B"],
            }]
        )

        normalized = normalize_explanation_output(output)
        claim_text = " ".join(item["text"] for item in normalized["core_claims"])

        self.assertIn("CYP2C19)", claim_text)
        self.assertEqual(3, len(normalized["core_claims"]))

    def test_mechanism_summary_is_split_into_atomic_claims(self):
        output = json.dumps(
            [{
                "mechanism_summary": "CYP2C9 is shared; ABCB1 is also shared.",
                "representative_pairs": ["A -> B"],
            }]
        )

        normalized = normalize_explanation_output(output)

        self.assertEqual(2, len(normalized["core_claims"]))

    def test_query_coverage_uses_three_representatives_and_drops_duplicates(self):
        queries = "\n".join([
            "1. A -> A",
            "2. A -> B",
            "3. a -> b",
            "4. A -> C",
            "5. A -> D",
            "6. A -> E",
        ])
        metrics = compute_query_coverage(queries, ["A -> B", "A -> C", "A -> D"])

        self.assertEqual(4, metrics["total_queries"])
        self.assertEqual(3, metrics["query_coverage_denominator"])
        self.assertEqual(1.0, metrics["query_coverage"])
        self.assertEqual(1.0, metrics["representative_pair_precision"])

    def test_kg_grounding_uses_explicit_anchor_overlap(self):
        output = json.dumps(
            [{
                "analysis_steps": "1. The sentence reports reduced metabolism.",
                "mechanism_summary": "CYP2C9 is a shared enzyme anchor.",
                "representative_pairs": ["DIFLUCAN -> tolbutamide"],
            }]
        )
        metrics = compute_faithfulness_for_sample(
            sentence="DIFLUCAN reduces the metabolism of tolbutamide.",
            explanation=output,
            predicted_label="mechanism",
            kg_evidence="DIFLUCAN -> tolbutamide: Shared Nodes: [CYP2C9(Enzyme)]",
            queries="1. DIFLUCAN -> tolbutamide",
        )

        self.assertGreater(metrics["kg_grounded_ratio"], 0.0)
        self.assertEqual(["cyp2c9"], metrics["kg_grounded_claims"][0]["matched_anchors"])

    def test_config_snapshot_redacts_api_keys(self):
        redacted = _redact_config_secrets({"external_generation": {"api_key": "secret"}})
        self.assertEqual("${DASHSCOPE_API_KEY}", redacted["external_generation"]["api_key"])

    def test_judge_summary_excludes_request_failures(self):
        summary = _summarize_judge_results(
            [
                {"mechanism_overall_score": 8, "mechanism_overall_decision": "fair"},
                {"mechanism_overall_score": 10, "mechanism_overall_decision": "good"},
                {
                    "mechanism_overall_score": 0,
                    "mechanism_overall_decision": "poor",
                    "judge_short_rationale": "Judge error: HTTP 400",
                },
            ],
            "qwen-max",
        )

        self.assertEqual(3, summary["n_samples"])
        self.assertEqual(2, summary["n_valid_samples"])
        self.assertEqual(1, summary["n_failed_samples"])
        self.assertEqual(9.0, summary["mechanism_overall_score_mean"])
        self.assertEqual(0.5, summary["mechanism_good_rate"])


if __name__ == "__main__":
    unittest.main()
