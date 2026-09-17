import unittest

from navgen_pipeline.planner import TemplatePlanner
from navgen_pipeline.prompting import DoubaoPromptRefiner, PromptCompiler


class PromptingTests(unittest.TestCase):
    def test_compiler_emits_scene_timeline_and_negative_anchors(self):
        sample = TemplatePlanner().plan_one(0)
        bundle = PromptCompiler().compile(sample)
        for marker in ("【镜头与格式】", "【场景锚点】", "【目标锚点】", "【时间线】", "【终点条件】", "【运动约束】", "【负面约束】"):
            self.assertIn(marker, bundle.prompt)
        self.assertGreater(len(bundle.prompt), 700)
        self.assertEqual(bundle.checks["quality"], "pass")
        self.assertEqual(bundle.checks["timeline_count"], len(sample.route.waypoints) - 1)

    def test_template_planner_persists_prompt_report(self):
        sample = TemplatePlanner().plan_one(1)
        self.assertEqual(sample.metadata["prompt_compiler"], "seedance-nav-v2")
        self.assertEqual(sample.metadata["prompt_report"]["source"], "deterministic_compiler")
        self.assertIn("【目标锚点】", sample.seedance_prompt)

    def test_refiner_preserves_facts_when_response_is_complete(self):
        sample = TemplatePlanner().plan_one(2)

        # Use valid JSON rather than relying on model-specific SDK objects.
        import json

        class JsonResponses:
            def generate_text(self, **kwargs):
                base = PromptCompiler().compile(sample)
                return json.dumps({"seedance_prompt": base.prompt, "negative_prompt": "keep all", "changes": ["wording"]}, ensure_ascii=False)

        bundle = DoubaoPromptRefiner(JsonResponses(), "prompt-model").refine(sample)
        self.assertEqual(bundle.source, "doubao_refiner")
        self.assertEqual(bundle.checks["quality"], "pass")
        self.assertTrue(bundle.checks["refiner_negative_fallback"])
        self.assertIn("穿墙", bundle.negative_prompt)
        self.assertIn("【时间线】", bundle.prompt)

    def test_refiner_falls_back_without_duplicate_prompt_sections(self):
        sample = TemplatePlanner().plan_one(0)

        import json

        class IncompleteResponses:
            def generate_text(self, **kwargs):
                return json.dumps(
                    {
                        "seedance_prompt": "只描述了一个房间，没有完整结构",
                        "negative_prompt": "keep all",
                        "changes": ["shortened"],
                    },
                    ensure_ascii=False,
                )

        base = PromptCompiler().compile(sample)
        bundle = DoubaoPromptRefiner(IncompleteResponses(), "prompt-model").refine(sample)
        self.assertEqual(bundle.prompt, base.prompt)
        self.assertTrue(bundle.checks["refiner_prompt_fallback"])
        self.assertEqual(bundle.prompt.count("【场景锚点】"), 1)


if __name__ == "__main__":
    unittest.main()
