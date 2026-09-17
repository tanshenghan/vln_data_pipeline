import math
import json
import tempfile
import unittest
from pathlib import Path

from navgen_pipeline.models import RouteContract, StopCondition, Waypoint
from navgen_pipeline.trajectory import TrajectoryGenerator, read_trajectory_file, trajectory_metrics
from navgen_pipeline.trajectory_filter import TrajectoryFilter


class TrajectoryFilterTests(unittest.TestCase):
    def _straight_route(self, distance=2.0, duration=10.0, obstacles=None):
        return RouteContract(
            waypoints=[Waypoint(0.0, 0.0, 0.0, 0.0), Waypoint(duration, distance, 0.0, 0.0)],
            max_speed_mps=0.8,
            max_accel_mps2=0.9,
            max_angular_speed_rps=1.4,
            max_angular_accel_rps2=2.5,
            clearance_m=0.18,
            obstacles=list(obstacles or []),
        )

    def test_smooth_route_passes_and_has_reasonable_metrics(self):
        route = self._straight_route(distance=2.0, duration=10.0)
        trajectory = TrajectoryGenerator(10.0).generate(route)
        report, _ = TrajectoryFilter().evaluate(
            trajectory,
            route,
            stop_condition=StopCondition(distance_min_m=0.8, distance_max_m=1.2, dwell_s=0.0),
            target_position=(3.0, 0.0),
            allow_review=False,
        )
        self.assertEqual(report.decision, "pass")
        self.assertLessEqual(report.metrics["max_speed_mps"], 0.9)

    def test_collision_is_hard_rejection(self):
        route = self._straight_route(
            distance=2.0,
            duration=10.0,
            obstacles=[{"x_min": 0.8, "x_max": 1.2, "y_min": -0.3, "y_max": 0.3}],
        )
        trajectory = TrajectoryGenerator(10.0).generate(route)
        report, _ = TrajectoryFilter().evaluate(trajectory, route)
        self.assertEqual(report.decision, "reject")
        self.assertIn("collision_or_insufficient_clearance", report.hard_failures)

    def test_terminal_dwell_is_measured(self):
        route = RouteContract(
            waypoints=[Waypoint(0.0, 0.0, 0.0, 0.0), Waypoint(3.0, 1.0, 0.0, 0.0), Waypoint(5.0, 1.0, 0.0, 0.0)],
            max_speed_mps=0.8,
            max_accel_mps2=0.9,
            max_angular_speed_rps=1.4,
            max_angular_accel_rps2=2.5,
        )
        trajectory = TrajectoryGenerator(10.0).generate(route)
        report, _ = TrajectoryFilter().evaluate(
            trajectory,
            route,
            stop_condition=StopCondition(distance_min_m=0.5, distance_max_m=1.5, dwell_s=1.5),
            target_position=(2.0, 0.0),
            allow_review=False,
        )
        self.assertEqual(report.decision, "pass")
        self.assertGreaterEqual(report.metrics["stop_duration_s"], 1.5)

    def test_abnormally_fast_route_is_rejected(self):
        route = self._straight_route(distance=8.0, duration=2.0)
        trajectory = TrajectoryGenerator(10.0).generate(route)
        report, _ = TrajectoryFilter().evaluate(trajectory, route)
        self.assertIn("abnormal_speed", report.hard_failures)

    def test_external_pose_jsonl_round_trip_is_strict(self):
        with tempfile.TemporaryDirectory(prefix="navgen-poses-") as directory:
            path = Path(directory) / "poses.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"t": t, "x": t * 0.1, "y": 0.0, "yaw": 0.0})
                    for t in (0.0, 0.1, 0.2)
                )
                + "\n",
                encoding="utf-8",
            )
            samples = read_trajectory_file(path)
            self.assertEqual(len(samples), 3)
            self.assertAlmostEqual(samples[-1].x, 0.02)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
