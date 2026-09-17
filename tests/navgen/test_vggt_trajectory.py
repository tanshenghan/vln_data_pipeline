import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from navgen_pipeline.config import PipelineConfig
from navgen_pipeline.models import PoseSample
from navgen_pipeline.pipeline import NavigationDataPipeline
from navgen_pipeline.vggt_trajectory import (
    VGGTTrajectoryResult,
    _derive_raw_kinematics,
    _pose_encoding_to_matrices,
    camera_matrices_to_navigation_poses,
    trajectory_data_dict,
)
from navgen_pipeline.trajectory_variants import (
    build_vggt_variants,
    smooth_trajectory,
    variant_summary,
)


class VGGTTrajectoryTests(unittest.TestCase):
    def test_pose_encoding_decodes_opencv_extrinsics_and_fov(self):
        encoding = np.asarray(
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, math.pi / 2.0, math.pi / 2.0]],
            dtype=np.float64,
        )
        extrinsics, intrinsics = _pose_encoding_to_matrices(encoding, image_size=518, np=np)
        np.testing.assert_allclose(extrinsics[0, :, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(extrinsics[0, :, 3], [1.0, 2.0, 3.0], atol=1e-8)
        np.testing.assert_allclose(intrinsics[0, 0, 0], 259.0, atol=1e-8)
        np.testing.assert_allclose(intrinsics[0, 1, 1], 259.0, atol=1e-8)

    def test_camera_centres_use_first_camera_horizontal_frame(self):
        # For an OpenCV camera-from-world matrix [I | t], the camera centre is
        # -t.  The second camera therefore moves one unit along local +X.
        extrinsics = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )
        poses, metadata = camera_matrices_to_navigation_poses(extrinsics, np=np)
        self.assertEqual(metadata["coordinate_frame"], "vggt_first_camera_horizontal")
        self.assertAlmostEqual(poses[0][0], 0.0)
        self.assertAlmostEqual(poses[1][0], 1.0)
        self.assertAlmostEqual(poses[1][1], 0.0)

    def test_raw_kinematics_does_not_smooth_positions(self):
        samples = [
            PoseSample(t=0.0, x=0.0, y=0.0),
            PoseSample(t=1.0, x=1.0, y=0.0),
            PoseSample(t=2.0, x=1.2, y=0.0),
        ]
        _derive_raw_kinematics(samples)
        self.assertEqual([sample.x for sample in samples], [0.0, 1.0, 1.2])
        self.assertAlmostEqual(samples[1].vx, 0.6)

    def test_smoothed_scaled_is_the_production_default(self):
        config = PipelineConfig(trajectory_source="vggt")
        self.assertEqual(config.video_estimator, "vggt")
        self.assertEqual(config.vggt_input_size % 14, 0)
        self.assertEqual(config.vggt_postprocess, "smoothed_scaled")
        self.assertEqual(config.pipeline_profile, "fixed")
        self.assertEqual(config.pipeline_contract(mode="live")["version"], "navgen-fixed-v1")

    def test_fixed_profile_rejects_noncanonical_live_trajectory_settings(self):
        raw = PipelineConfig(vggt_postprocess="raw")
        self.assertIn("vggt_postprocess must be smoothed_scaled", "; ".join(raw.fixed_contract_errors(live=True)))
        self.assertIn("vggt_postprocess must be smoothed_scaled", "; ".join(raw.fixed_contract_errors(live=False)))
        experiment = PipelineConfig(pipeline_profile="experiment", vggt_postprocess="raw")
        self.assertEqual(experiment.fixed_contract_errors(live=True), [])
        self.assertEqual(experiment.fixed_contract_errors(live=False), [])
        self.assertEqual(PipelineConfig(trajectory_source="route_contract").fixed_contract_errors(live=False), [])
        with self.assertRaisesRegex(ValueError, "fixed pipeline contract violation"):
            NavigationDataPipeline(config=raw, output_dir=Path("/tmp/navgen-fixed-contract-test"), mode="live")
        with self.assertRaisesRegex(ValueError, "fixed pipeline contract violation"):
            NavigationDataPipeline(
                config=raw,
                output_dir=Path("/tmp/navgen-fixed-contract-offline-test"),
                mode="offline",
            )

    def test_seedance_auto_source_dispatches_to_configured_vggt_variant(self):
        fake_points = [
            PoseSample(t=0.0, x=0.0, y=0.0, source="vggt_monocular_estimate"),
            PoseSample(t=1.0, x=0.2, y=0.4, source="vggt_monocular_estimate"),
        ]
        fake_report = {
            "source": "vggt_monocular_estimate",
            "ground_truth": False,
            "metric_scale": "unknown",
            "metric_units": "relative",
            "metrics_are_metric": False,
            "temporal_smoothing": False,
            "scale_anchor": "none",
            "confidence": 0.5,
            "warnings": ["no_temporal_smoothing", "no_scale_anchor"],
            "estimator": "vggt_camera_head",
        }
        fake_result = VGGTTrajectoryResult(fake_points, fake_report)
        with tempfile.TemporaryDirectory(prefix="navgen-vggt-dispatch-") as directory:
            pipeline = NavigationDataPipeline(
                # Keep this dispatch test focused on source selection; choose
                # the raw variant explicitly so the expected points are not
                # altered by the production post-processing default.
                config=PipelineConfig(
                    pipeline_profile="experiment",
                    trajectory_source="auto",
                    video_estimator="vggt",
                    vggt_postprocess="raw",
                ),
                output_dir=Path(directory) / "run",
                mode="offline",
            )
            sample = pipeline.template_planner.plan_one(0)
            with patch("navgen_pipeline.pipeline.estimate_vggt_trajectory", return_value=fake_result) as estimator:
                source = pipeline._effective_trajectory_source({"backend": "seedance"})
                points, report, provenance = pipeline._build_trajectory(
                    sample, Path(directory) / "seedance.mp4", {"backend": "seedance"}
                )
        self.assertEqual(source, "vggt")
        estimator.assert_called_once()
        self.assertEqual([point.x for point in points], [0.0, 0.2])
        self.assertEqual(report.source, "vggt_monocular_estimate")
        self.assertFalse(report.ground_truth)
        self.assertEqual(report.metric_scale, "unknown")
        self.assertEqual(report.variant, "raw")
        self.assertFalse(provenance["temporal_smoothing"])
        self.assertEqual(provenance["scale_anchor"], "none")

    def test_default_vggt_build_selects_smoothed_scaled_variant(self):
        fake_points = [
            PoseSample(t=0.0, x=0.0, y=0.0, source="vggt_monocular_estimate"),
            PoseSample(t=1.0, x=0.2, y=0.4, source="vggt_monocular_estimate"),
            PoseSample(t=2.0, x=0.4, y=0.8, source="vggt_monocular_estimate"),
        ]
        fake_result = VGGTTrajectoryResult(
            fake_points,
            {
                "source": "vggt_monocular_estimate",
                "ground_truth": False,
                "metric_scale": "unknown",
                "metric_units": "relative",
                "metrics_are_metric": False,
                "temporal_smoothing": False,
                "scale_anchor": "none",
                "confidence": 0.5,
                "warnings": ["no_temporal_smoothing", "no_scale_anchor"],
                "estimator": "vggt_camera_head",
            },
        )
        with tempfile.TemporaryDirectory(prefix="navgen-vggt-default-") as directory:
            pipeline = NavigationDataPipeline(
                config=PipelineConfig(trajectory_source="vggt"),
                output_dir=Path(directory) / "run",
                mode="offline",
            )
            sample = pipeline.template_planner.plan_one(0)
            with patch("navgen_pipeline.pipeline.estimate_vggt_trajectory", return_value=fake_result):
                points, report, provenance = pipeline._build_trajectory(
                    sample, Path(directory) / "seedance.mp4", {"backend": "seedance"}
                )
        self.assertEqual(report.variant, "smoothed_scaled")
        self.assertTrue(report.temporal_smoothing)
        self.assertEqual(report.scale_anchor, "route_endpoint_displacement")
        self.assertFalse(report.ground_truth)
        self.assertEqual(provenance["variant"], "smoothed_scaled")
        self.assertEqual(points[0].x, sample.route.waypoints[0].x)
        self.assertEqual(points[-1].x, sample.route.waypoints[-2].x)

    def test_vggt_artifact_promotes_frame_and_scale_metadata(self):
        points = [
            PoseSample(t=0.0, x=0.0, y=0.0, frame_id="vggt_first_camera_horizontal"),
            PoseSample(t=1.0, x=0.1, y=0.2, frame_id="vggt_first_camera_horizontal"),
        ]
        result = VGGTTrajectoryResult(
            points,
            {
                "source": "vggt_monocular_estimate",
                "ground_truth": False,
                "metric_scale": "unknown",
                "metric_units": "relative",
                "metrics_are_metric": False,
            },
        )
        artifact = trajectory_data_dict(result)
        self.assertEqual(artifact["source"], "vggt_monocular_estimate")
        self.assertFalse(artifact["ground_truth"])
        self.assertEqual(artifact["coordinate_frame"], "vggt_first_camera_horizontal")
        self.assertEqual(artifact["metric_scale"], "unknown")
        self.assertEqual(artifact["metric_units"], "relative")
        self.assertFalse(artifact["metrics_are_metric"])

    def test_smoothing_preserves_endpoints_but_changes_interior(self):
        raw = [
            PoseSample(t=0.0, x=0.0, y=0.0, yaw=0.0),
            PoseSample(t=1.0, x=1.0, y=0.8, yaw=0.1),
            PoseSample(t=2.0, x=2.0, y=-0.6, yaw=0.2),
            PoseSample(t=3.0, x=3.0, y=0.0, yaw=0.3),
        ]
        smoothed = smooth_trajectory(raw, window=3)
        self.assertEqual((smoothed[0].x, smoothed[0].y), (raw[0].x, raw[0].y))
        self.assertEqual((smoothed[-1].x, smoothed[-1].y), (raw[-1].x, raw[-1].y))
        self.assertNotEqual((smoothed[1].x, smoothed[1].y), (raw[1].x, raw[1].y))
        self.assertEqual(smoothed[1].source, "vggt_monocular_estimate_smoothed")

    def test_smoothing_keeps_endpoint_neighbourhood_to_avoid_edge_spike(self):
        raw = [
            PoseSample(t=float(index), x=float(index), y=float(index % 2))
            for index in range(8)
        ]
        smoothed = smooth_trajectory(raw, window=5)
        # With radius=2 the first/last two samples stay as observed; the
        # interior is still filtered.
        self.assertEqual([(p.x, p.y) for p in smoothed[:2]], [(p.x, p.y) for p in raw[:2]])
        self.assertEqual([(p.x, p.y) for p in smoothed[-2:]], [(p.x, p.y) for p in raw[-2:]])
        self.assertNotEqual((smoothed[3].x, smoothed[3].y), (raw[3].x, raw[3].y))

    def test_endpoint_scale_variant_hits_declared_route_endpoints(self):
        raw = [
            PoseSample(t=0.0, x=0.0, y=0.0),
            PoseSample(t=1.0, x=1.0, y=1.0),
        ]
        result = VGGTTrajectoryResult(
            raw,
            {
                "source": "vggt_monocular_estimate",
                "ground_truth": False,
                "metric_scale": "unknown",
                "metric_units": "relative",
                "metrics_are_metric": False,
                "warnings": [],
            },
        )
        variants = build_vggt_variants(result, anchor_start=(2.0, 3.0), anchor_end=(6.0, 3.0))
        scaled = variants["scaled"]
        self.assertAlmostEqual(scaled.trajectory[0].x, 2.0)
        self.assertAlmostEqual(scaled.trajectory[0].y, 3.0)
        self.assertAlmostEqual(scaled.trajectory[-1].x, 6.0)
        self.assertAlmostEqual(scaled.trajectory[-1].y, 3.0)
        self.assertFalse(scaled.report["ground_truth"])
        self.assertEqual(scaled.report["metric_scale"], "route_endpoint_anchor")
        self.assertFalse(scaled.report["metrics_are_metric"])
        self.assertEqual(scaled.report["scale_anchor"], "route_endpoint_displacement")

    def test_variant_summary_contains_all_comparison_rows(self):
        points = [
            PoseSample(t=0.0, x=0.0, y=0.0),
            PoseSample(t=1.0, x=1.0, y=0.0),
        ]
        result = VGGTTrajectoryResult(
            points,
            {
                "source": "vggt_monocular_estimate",
                "ground_truth": False,
                "metric_scale": "unknown",
                "metric_units": "relative",
                "metrics_are_metric": False,
                "warnings": [],
            },
        )
        variants = build_vggt_variants(result, anchor_start=(0.0, 0.0), anchor_end=(2.0, 0.0))
        # The reference is the declared route in the same frame as the
        # endpoint anchor.  The raw VGGT points intentionally remain in an
        # arbitrary relative frame and are not suitable as that reference.
        reference = [
            PoseSample(t=0.0, x=0.0, y=0.0),
            PoseSample(t=1.0, x=2.0, y=0.0),
        ]
        rows = variant_summary(variants, reference=reference)
        self.assertEqual([row["variant"] for row in rows], ["raw", "smoothed", "scaled", "smoothed_scaled"])
        self.assertTrue(all(row["ground_truth"] is False for row in rows))
        self.assertFalse(rows[0]["reference_comparison_valid"])
        self.assertIsNone(rows[0]["endpoint_error_to_reference"])
        self.assertIsNone(rows[0]["path_length_ratio_to_reference"])
        self.assertTrue(rows[2]["reference_comparison_valid"])
        self.assertAlmostEqual(rows[2]["endpoint_error_to_reference"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
