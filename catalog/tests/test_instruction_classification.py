import json
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_semantic_explorer.instruction_classification import (
    ATOMIC_ACTIONS,
    FORCED_SCENE_SIM,
    FORCED_SCENE_SSV2,
    SCENES,
    TASK_TYPES,
    ClassificationUnit,
    build_classification_units,
    classify_units_resilient,
    load_checkpoint,
    load_unique_instructions,
    parse_classification_response,
)


class InstructionClassificationTest(unittest.TestCase):
    def test_malformed_batch_is_split_and_retried(self) -> None:
        units = [
            ClassificationUnit(unit_id="a", text="a", member_hashes=("a",), context=""),
            ClassificationUnit(unit_id="b", text="b", member_hashes=("b",), context=""),
        ]

        class SplitClient:
            def classify(self, batch):
                if len(batch) > 1:
                    raise RuntimeError("malformed batch")
                return {batch[0].unit_id: {"scene": "ok"}}

        self.assertEqual(
            classify_units_resilient(SplitClient(), units),
            {"a": {"scene": "ok"}, "b": {"scene": "ok"}},
        )

    def test_compact_scene_taxonomy_is_fixed(self) -> None:
        self.assertEqual(len(SCENES), 10)
        self.assertEqual(len(SCENES), len(set(SCENES)))
        self.assertNotIn("未知场景", SCENES)

    def test_inventory_is_aggregated_by_instruction_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            rows = [
                {
                    "instruction_hash": "h1",
                    "instruction": "pick up cup",
                    "domain": "real",
                    "dataset": "A",
                    "source_kind": "catalog_task",
                },
                {
                    "instruction_hash": "h1",
                    "instruction": "pick up cup",
                    "domain": "real",
                    "dataset": "B",
                    "source_kind": "lerobot_episode_jsonl",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            unique = load_unique_instructions(path)
            self.assertEqual(set(unique), {"h1"})
            self.assertEqual(unique["h1"].occurrence_count, 2)
            self.assertEqual(unique["h1"].dataset_names, ["A", "B"])

    def test_ssv2_templates_collapse_and_sim_scene_is_forced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            rows = [
                {
                    "instruction_hash": "ss1",
                    "instruction": "putting cup on table",
                    "instruction_template": "putting [something] on [something]",
                    "domain": "real",
                    "dataset": "Something-Something-V2",
                    "source_kind": "something_something_labels",
                },
                {
                    "instruction_hash": "ss2",
                    "instruction": "putting book on shelf",
                    "instruction_template": "putting [something] on [something]",
                    "domain": "real",
                    "dataset": "Something-Something-V2",
                    "source_kind": "something_something_labels",
                },
                {
                    "instruction_hash": "sim1",
                    "instruction": "open drawer",
                    "domain": "sim",
                    "dataset": "LIBERO",
                    "source_kind": "catalog_task",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            units, deterministic = build_classification_units(load_unique_instructions(path))
            self.assertFalse(deterministic)
            self.assertEqual(len(units), 2)
            template = next(unit for unit in units if unit.source == "llm_template")
            self.assertEqual(template.forced_scene, FORCED_SCENE_SSV2)
            self.assertEqual(set(template.member_hashes), {"ss1", "ss2"})
            sim = next(unit for unit in units if unit.unit_id == "sim1")
            self.assertEqual(sim.forced_scene, FORCED_SCENE_SIM)

    def test_vqa_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            row = {
                "instruction_hash": "v1",
                "instruction": "RefSpatial",
                "domain": "vqa",
                "dataset": "RefSpatial",
                "source_kind": "catalog_task",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            units, deterministic = build_classification_units(load_unique_instructions(path))
            self.assertFalse(units)
            self.assertEqual(deterministic["v1"]["task_type"], "视觉问答 / 空间理解")

    def test_cleaning_scene_override_preserves_dishwashing_as_kitchen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            rows = [
                {
                    "instruction_hash": "trash",
                    "instruction": "collect trash",
                    "domain": "real",
                    "dataset": "A",
                    "source_kind": "catalog_task",
                },
                {
                    "instruction_hash": "bowl",
                    "instruction": "clean bowl",
                    "domain": "real",
                    "dataset": "A",
                    "source_kind": "catalog_task",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            units, _ = build_classification_units(load_unique_instructions(path))
            by_id = {unit.unit_id: unit for unit in units}
            self.assertEqual(by_id["trash"].forced_scene, "清洁与维护")
            self.assertEqual(by_id["bowl"].forced_scene, "厨房与餐饮")

    def test_response_parser_enforces_fixed_labels_and_forced_scene(self) -> None:
        unit = ClassificationUnit(
            unit_id="u1",
            text="open a drawer",
            member_hashes=("h1",),
            context="",
            forced_scene="家居与日常生活",
        )
        response = {
            "output": [
                {
                    "content": [
                        {
                            "text": json.dumps(
                                [
                                    {
                                        "id": "u1",
                                        "scene": "家居与日常生活",
                                        "task_type": "开合 / 启闭",
                                        "primary_atomic_action": "打开",
                                        "atomic_actions": ["打开"],
                                        "confidence": 0.9,
                                        "reason": "抽屉开合",
                                    }
                                ],
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            ]
        }
        parsed = parse_classification_response(response, [unit])
        self.assertEqual(parsed["u1"]["task_type"], "开合 / 启闭")
        self.assertEqual(parsed["u1"]["primary_atomic_action"], "打开")
        self.assertEqual(parsed["u1"]["atomic_actions"], ["打开"])
        response["output"][0]["content"][0]["text"] = response["output"][0]["content"][0]["text"].replace(
            "家居与日常生活", "厨房与餐饮"
        )
        with self.assertRaisesRegex(ValueError, "forced scene mismatch"):
            parse_classification_response(response, [unit])

    def test_checkpoint_ignores_other_model_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            rows = [
                {
                    "instruction_hash": "keep",
                    "taxonomy_version": "compact-scenes-atomic-v3",
                    "classification_source": "llm",
                    "model": "chosen",
                },
                {
                    "instruction_hash": "drop",
                    "taxonomy_version": "compact-scenes-atomic-v3",
                    "classification_source": "llm",
                    "model": "other",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(set(load_checkpoint(path, model="chosen")), {"keep"})

    def test_task_taxonomy_has_fallback(self) -> None:
        self.assertIn("其他 / 复合任务", TASK_TYPES)
        self.assertIn("其他", ATOMIC_ACTIONS)
        self.assertIn("烹饪 / 制作", ATOMIC_ACTIONS)
        self.assertIn("挤压 / 形变 / 撕裂", ATOMIC_ACTIONS)
        self.assertEqual(len(ATOMIC_ACTIONS), 27)
        self.assertEqual(len(ATOMIC_ACTIONS), len(set(ATOMIC_ACTIONS)))


if __name__ == "__main__":
    unittest.main()
