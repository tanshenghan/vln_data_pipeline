import json
import shutil
import tempfile
import unittest
from pathlib import Path

from navgen_pipeline.config import PipelineConfig
from navgen_pipeline.pipeline import NavigationDataPipeline


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe are required")
class OfflinePipelineTests(unittest.TestCase):
    def test_small_batch_writes_auditable_samples(self):
        with tempfile.TemporaryDirectory(prefix="navgen-test-") as directory:
            root = Path(directory)
            config = PipelineConfig(duration_s=15.0, frame_sample_fps=1.0)
            records = NavigationDataPipeline(config=config, output_dir=root / "run", mode="offline").run_batch(3)
            self.assertEqual(len(records), 3)
            self.assertGreaterEqual(sum(record.quality_tier == "A" for record in records), 1)
            for record in records:
                sample_dir = root / "run" / "samples" / record.sample_id
                self.assertTrue((sample_dir / "sample.json").exists())
                self.assertTrue((sample_dir / "pre_qc.json").exists())
                self.assertTrue((sample_dir / "embedding_report.json").exists())
                self.assertTrue((sample_dir / "prompt_report.json").exists())
                self.assertTrue((sample_dir / "video.mp4").exists())
                self.assertTrue((sample_dir / "trajectory.jsonl").exists())
                self.assertTrue((sample_dir / "trajectory.json").exists())
                self.assertTrue((sample_dir / "planned_trajectory.jsonl").exists())
                self.assertTrue((sample_dir / "trajectory_plot.png").exists())
                self.assertTrue((sample_dir / "usage.json").exists())
                sample_artifact = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
                provenance = sample_artifact["trajectory_provenance"]
                self.assertIn("source", provenance)
                self.assertIn("metric_scale", provenance)
                self.assertIn("metrics_are_metric", provenance)
            summary = json.loads((root / "run" / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["count"], 3)
            self.assertEqual(summary["pipeline_contract"]["version"], "navgen-fixed-v1")
            self.assertEqual(summary["pipeline_contract"]["trajectory_variant"], "smoothed_scaled")
            self.assertEqual(summary["pipeline_contract"]["runtime_trajectory_variant"], "route_contract")
            self.assertIn("quality_tier_counts", summary)
            self.assertIn("usage", summary)
            self.assertIn("trajectory_data_path", summary["samples"][0])
            self.assertTrue((root / "run" / "usage_summary.json").exists())
            self.assertEqual(len((root / "run" / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
