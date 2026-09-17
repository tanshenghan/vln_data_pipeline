import unittest

from navgen_pipeline.ark_client import ArkResponsesClient, SeedanceClient
from navgen_pipeline.vlm_filter import DoubaoVLMFilter
from navgen_pipeline.planner import TemplatePlanner


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.user_agent = "test-agent"

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class ArkClientTests(unittest.TestCase):
    def test_responses_text_extraction_and_payload_shape(self):
        http = FakeHTTP([{"output": [{"content": [{"type": "output_text", "text": "{\"ok\": true}"}]}]}])
        text = ArkResponsesClient(http).generate_text("model-x", "system", "user")
        self.assertEqual(text, '{"ok": true}')
        method, path, payload = http.calls[0]
        self.assertEqual((method, path), ("POST", "/responses"))
        self.assertEqual(payload["model"], "model-x")
        self.assertEqual(payload["input"][0]["content"][-1]["type"], "input_text")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_seedance_task_lifecycle_and_nested_video_url(self):
        http = FakeHTTP([
            {"id": "cgt-test"},
            {"id": "cgt-test", "status": "running"},
            {"id": "cgt-test", "status": "succeeded", "data": {"content": {"video_url": "https://example/video.mp4"}}},
        ])
        client = SeedanceClient(http)
        created = client.create_task("video-model", [{"type": "text", "text": "test"}], duration=15)
        self.assertEqual(created["id"], "cgt-test")
        final = client.wait_for_task("cgt-test", poll_s=0.2, timeout_s=2.0)
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(client.video_url(final), "https://example/video.mp4")
        with self.assertRaises(ValueError):
            client.create_task("video-model", [{"type": "text", "text": "test"}], duration=3)

    def test_vlm_pass_is_downgraded_when_scores_fail_gates(self):
        sample = TemplatePlanner().plan_one(0)
        http = FakeHTTP([
            {
                "output": [{"content": [{"type": "output_text", "text": (
                    '{"decision":"pass","scene_consistency":0.9,'
                    '"instruction_alignment":0.9,"target_identity":0.1,'
                    '"target_visible_ratio":0.9,"valid_interval":[0,15],'
                    '"confidence":0.9,"evidence":[],"reason_codes":[]}'
                )}]}]
            }
        ])
        # No files are needed when a contact sheet path is supplied; use a
        # tiny fake path only to exercise the JSON gate after parsing.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sheet.jpg"
            image.write_bytes(b"not-an-image")
            # The adapter encodes the file before making the fake request.
            report = DoubaoVLMFilter(ArkResponsesClient(http), "vlm").evaluate(
                sample, [], contact_sheet=image
            )
        self.assertEqual(report.decision, "review")
        self.assertIn("target_identity_uncertain", report.reason_codes)

    def test_vlm_malformed_numeric_field_is_reviewable(self):
        sample = TemplatePlanner().plan_one(0)
        http = FakeHTTP([
            {
                "output": [{"content": [{"type": "output_text", "text": (
                    '{"decision":"pass","scene_consistency":"cabinet_01: beige",'
                    '"instruction_alignment":0.9,"target_identity":0.9,'
                    '"target_visible_ratio":0.9,"valid_interval":[0,15],'
                    '"confidence":0.9,"evidence":[],"reason_codes":[]}'
                )}]}]
            }
        ])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sheet.jpg"
            image.write_bytes(b"not-an-image")
            report = DoubaoVLMFilter(ArkResponsesClient(http), "vlm").evaluate(
                sample, [], contact_sheet=image
            )
        self.assertEqual(report.decision, "review")
        self.assertIn("malformed_numeric_output", report.reason_codes)

    def test_vlm_semantic_prose_scores_are_coerced_with_audit_marker(self):
        sample = TemplatePlanner().plan_one(0)
        http = FakeHTTP([
            {
                "output": [{"content": [{"type": "output_text", "text": (
                    '{"decision":"pass","scene_consistency":"场景从头到尾保持一致",'
                    '"instruction_alignment":"视频与指令匹配并完成终点",'
                    '"target_identity":"cabinet_01（餐边柜）",'
                    '"target_visible_ratio":"始终可见","valid_interval":[0,15],'
                    '"confidence":0.95,"evidence":[],"reason_codes":[]}'
                )}]}]
            }
        ])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sheet.jpg"
            image.write_bytes(b"not-an-image")
            report = DoubaoVLMFilter(ArkResponsesClient(http), "vlm").evaluate(
                sample, [], contact_sheet=image
            )
        self.assertEqual(report.decision, "pass")
        self.assertEqual(report.scene_consistency, 1.0)
        self.assertEqual(report.instruction_alignment, 1.0)
        self.assertIn("coerced_semantic_score", report.reason_codes)

    def test_vlm_observed_scene_label_is_normalized(self):
        sample = TemplatePlanner().plan_one(0)
        http = FakeHTTP([
            {
                "output": [{"content": [{"type": "output_text", "text": (
                    '{"decision":"review","scene_consistency":0.9,'
                    '"instruction_alignment":0.9,"target_identity":0.9,'
                    '"target_visible_ratio":0.9,"valid_interval":[0,15],'
                    '"confidence":0.9,"observed_scene_label":"走廊",'
                    '"observed_scene_confidence":0.92,"observed_scene_source":"vlm_video",'
                    '"evidence":[],"reason_codes":[]}'
                )}]}]
            }
        ])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sheet.jpg"
            image.write_bytes(b"not-an-image")
            report = DoubaoVLMFilter(ArkResponsesClient(http), "vlm").evaluate(
                sample, [], contact_sheet=image
            )
        self.assertEqual(report.observed_scene_label, "hallway")
        self.assertAlmostEqual(report.observed_scene_confidence, 0.92)
        self.assertEqual(report.observed_scene_source, "vlm_video")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
