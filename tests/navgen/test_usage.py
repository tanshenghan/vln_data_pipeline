import unittest

from navgen_pipeline.config import PipelineConfig
from navgen_pipeline.usage import aggregate_usage, text_event, video_event


class UsageTests(unittest.TestCase):
    def test_stage_specific_vlm_rates_fall_back_to_global_rates(self):
        config = PipelineConfig(
            llm_input_cny_per_1k=0.006,
            llm_output_cny_per_1k=0.030,
            vlm_input_cny_per_1k=0.0032,
            vlm_output_cny_per_1k=0.016,
        )
        self.assertEqual(config.text_rates_for_stage("planner"), (0.006, 0.030))
        self.assertEqual(config.text_rates_for_stage("vlm"), (0.0032, 0.016))
        fallback = PipelineConfig(llm_input_cny_per_1k=0.006, llm_output_cny_per_1k=0.030)
        self.assertEqual(fallback.text_rates_for_stage("vlm"), (0.006, 0.030))

    def test_text_usage_normalization_and_estimate(self):
        event = text_event(
            "vlm",
            "model",
            {
                "status": "completed",
                "latency_s": 1.25,
                "usage": {"input_tokens": 1200, "output_tokens": 300, "reasoning_tokens": 10},
            },
            input_rate_cny_per_1k=0.01,
            output_rate_cny_per_1k=0.02,
        )
        self.assertEqual(event["total_tokens"], 1500)
        self.assertEqual(event["cost_status"], "estimated")
        self.assertAlmostEqual(event["cost_cny"], 0.018)

    def test_video_duration_without_rate_is_explicitly_unpriced(self):
        event = video_event(
            "seedance",
            {"id": "task", "status": "succeeded", "usage": {"completion_tokens": 123}},
            requested_duration_s=8,
            actual_duration_s=8.04,
            latency_s=42.0,
        )
        summary = aggregate_usage([event])
        self.assertAlmostEqual(summary["video_actual_duration_s"], 8.04)
        self.assertEqual(summary["video_provider_total_tokens"], 123)
        self.assertIsNone(summary["estimated_cost_cny"])
        self.assertEqual(summary["cost_status"], "unpriced")

    def test_offline_events_are_not_applicable_not_configured(self):
        summary = aggregate_usage(
            [
                {
                    "kind": "video",
                    "stage": "mock_video",
                    "latency_s": 0.1,
                    "requested_duration_s": 15,
                    "actual_duration_s": 15.0,
                    "cost_cny": 0.0,
                    "cost_status": "not_applicable",
                }
            ]
        )
        self.assertEqual(summary["cost_status"], "not_applicable")
        self.assertEqual(summary["estimated_cost_cny"], 0.0)
        self.assertEqual(summary["billable_cost_cny"], 0.0)

    def test_video_provider_token_rate_can_be_explicitly_estimated(self):
        event = video_event(
            "seedance",
            {"id": "task", "status": "succeeded", "usage": {"completion_tokens": 2500}},
            requested_duration_s=8,
            actual_duration_s=8.0,
            latency_s=1.0,
            per_1k_tokens_cny=0.12,
        )
        self.assertEqual(event["provider_total_tokens"], 2500)
        self.assertEqual(event["cost_status"], "estimated")
        self.assertAlmostEqual(event["cost_cny"], 0.3)


if __name__ == "__main__":
    unittest.main()
