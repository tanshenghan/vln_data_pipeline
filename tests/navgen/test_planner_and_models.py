import json
import unittest

from navgen_pipeline.models import dataclass_to_dict
from navgen_pipeline.models import VLMReport
from navgen_pipeline.pipeline import _attach_observed_scene_semantics
from navgen_pipeline.planner import PlannerConfig, TemplatePlanner, sample_spec_from_dict


class PlannerAndModelTests(unittest.TestCase):
    def test_template_batch_has_valid_contracts_and_terminal_dwell(self):
        samples = TemplatePlanner(PlannerConfig(duration_s=15.0)).plan_batch(3)
        self.assertEqual([sample.sample_id for sample in samples], ["nav_000000", "nav_000001", "nav_000002"])
        for sample in samples:
            self.assertEqual(sample.validate(), [])
            self.assertAlmostEqual(sample.route.waypoints[-1].t, 15.0)
            self.assertAlmostEqual(sample.route.waypoints[-1].x, sample.route.waypoints[-2].x)
            self.assertAlmostEqual(sample.route.waypoints[-1].y, sample.route.waypoints[-2].y)
            self.assertIn(sample.task.target_id, {obj.id for obj in sample.scene.objects})

    def test_model_json_round_trip(self):
        sample = TemplatePlanner().plan_one(0)
        encoded = json.dumps(dataclass_to_dict(sample), ensure_ascii=False)
        decoded = json.loads(encoded)
        # The remote planner adapter accepts the same top-level shape.
        restored = sample_spec_from_dict(decoded, sample.sample_id, sample.seed)
        self.assertEqual(restored.sample_id, sample.sample_id)
        self.assertEqual(restored.task.target_id, sample.task.target_id)
        self.assertEqual(restored.route.validate(), [])

    def test_remote_3d_aliases_and_duration_override_are_normalized(self):
        raw = {
            "scene": {
                "scene_id": "remote_scene",
                "room_type": "open_plan_kitchen_dining",
                "width": 8,
                "depth": 6,
                "objects": [
                    {"object_id": "cabinet_01", "category": "buffet_cabinet", "position": [5.0, 0.9, 4.0], "size": [1.6, 1.0, 0.6], "is_target": True},
                    {"object_id": "table_01", "category": "table", "position": [3.0, 0.8, 2.0], "size": [1.0, 0.7, 1.0]},
                ],
                "obstacles": [
                    {"obstacle_id": "table_01", "position": [3.0, 0.8, 2.0], "size": [1.0, 0.7, 1.0]}
                ],
                "regions": [
                    {"id": "work_area", "label": "WORK AREA", "x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 6.0},
                    {"id": "dining_area", "label": "DINING AREA", "x_min": 4.0, "x_max": 8.0, "y_min": 0.0, "y_max": 6.0},
                ],
            },
            "task": {
                "type": "go_to_target",
                "target_object_id": "cabinet_01",
                "instruction": "前进到餐边柜旁并停下。",
                "actions": ["move_forward", "stop"],
                "start_pose": [0.8, 0.0, 0.8, 0.0],
                "stop_condition": "距离目标0.8-1.2米，目标可见比例80%",
            },
            "route_contract": {
                "expected_duration_s": 9.5,
                "waypoints": [
                    {"position": [0.8, 0.0, 0.8], "yaw_rad": 0.0},
                    {"position": [4.0, 0.0, 3.4], "yaw_rad": 0.2},
                    {"position": [4.2, 0.0, 3.8], "yaw_rad": 0.4},
                ],
            },
        }
        restored = sample_spec_from_dict(raw, "nav_remote", 11, duration_override_s=15)
        self.assertEqual(restored.task.type, "object_goal")
        self.assertEqual(restored.task.target_id, "cabinet_01")
        self.assertEqual(restored.task.start_pose, (0.8, 0.8, 0.0))
        self.assertAlmostEqual(restored.route.duration_s, 15.0)
        self.assertEqual(restored.route.waypoints[1].y, 3.4)
        self.assertAlmostEqual(restored.task.stop_condition.distance_min_m, 0.8)
        self.assertAlmostEqual(restored.task.stop_condition.distance_max_m, 1.2)
        self.assertEqual([region["label"] for region in restored.scene.regions], ["WORK AREA", "DINING AREA"])

    def test_observed_scene_semantics_are_separate_from_room_prior(self):
        sample = TemplatePlanner().plan_one(0)
        report = VLMReport(
            decision="pass",
            scene_consistency=0.95,
            instruction_alignment=0.95,
            target_identity=0.95,
            target_visible_ratio=0.9,
            valid_interval=(0.0, 15.0),
            confidence=0.9,
            observed_scene_label="hallway",
            observed_scene_confidence=0.92,
            observed_scene_source="vlm_video",
        )
        _attach_observed_scene_semantics(sample, report)
        self.assertEqual(sample.scene.room_type, "hallway_lounge")
        self.assertEqual(sample.scene.semantic_label, "hallway")
        self.assertEqual(sample.scene.semantic_source, "vlm_video")
        self.assertEqual(report.reason_codes, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
