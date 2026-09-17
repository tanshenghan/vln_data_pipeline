import unittest
from pathlib import Path

from navgen_pipeline.rxr import load_rxr_episodes


class RxRImportTests(unittest.TestCase):
    def test_source_batch_preserves_episode_identity_and_instruction(self):
        source = Path(__file__).parents[1] / "data" / "rxr_sn83yjsr3w2_episodes.json"
        samples = load_rxr_episodes(source, duration_s=15.0)

        self.assertEqual([sample.sample_id for sample in samples], ["rxr_13288", "rxr_13289", "rxr_13290"])
        self.assertEqual(samples[0].scene.scene_id, "SN83YJsR3w2")
        self.assertEqual(samples[0].metadata["source_episode_id"], "13288")
        self.assertEqual(samples[1].task.target_id, "bedroom_door_02")
        self.assertIn("outside seating area", samples[2].task.instruction_original.lower())
        self.assertNotIn("Wait, that's wrong", samples[2].task.instruction_original)

        for sample in samples:
            self.assertEqual(sample.metadata["source"], "RxR")
            self.assertEqual(sample.metadata["source_scene_id"], "SN83YJsR3w2")
            self.assertEqual(sample.route.waypoints[-1].t, 15.0)
            self.assertEqual(sample.route.waypoints[-1].x, sample.route.waypoints[-2].x)
            self.assertEqual(sample.route.waypoints[-1].y, sample.route.waypoints[-2].y)
            self.assertEqual(sample.validate(), [])


if __name__ == "__main__":
    unittest.main()
