from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.classification_overlay import enrich_instruction_classification
from catalog.instruction_classification import TAXONOMY_VERSION
from catalog.instruction_inventory import instruction_hash


class ClassificationOverlayTest(unittest.TestCase):
    def test_current_taxonomy_updates_semantics_and_keeps_pending_group(self) -> None:
        catalog = {
            "records": [
                {"native_task": "open drawer", "scene": "旧场景", "task_type": "旧类型"},
                {"native_task": "unknown task", "scene": "旧场景", "task_type": "旧类型"},
            ],
            "stats": {},
        }
        row = {
            "schema_version": 1,
            "taxonomy_version": TAXONOMY_VERSION,
            "instruction_hash": instruction_hash("open drawer"),
            "scene": "家居与日常生活",
            "task_type": "开合 / 启闭",
            "primary_atomic_action": "打开",
            "atomic_actions": ["打开"],
            "confidence": 0.97,
            "needs_review": False,
            "reason": "打开抽屉",
            "classification_source": "llm",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            enrich_instruction_classification(catalog, path)

        classified, pending = catalog["records"]
        self.assertEqual(classified["scene"], "家居与日常生活")
        self.assertEqual(classified["task_type"], "开合 / 启闭")
        self.assertEqual(classified["primary_atomic_action"], "打开")
        self.assertEqual(classified["atomic_actions"], ["打开"])
        self.assertEqual(classified["scene_source"], "instruction_classification")
        self.assertEqual(pending["primary_atomic_action"], "待分类")
        self.assertEqual(pending["atomic_actions"], [])
        self.assertEqual(
            catalog["stats"]["instruction_classification"],
            {"classified_records": 1, "pending_records": 1, "needs_review_records": 0},
        )

    def test_old_taxonomy_and_truncated_line_are_ignored(self) -> None:
        catalog = {"records": [{"native_task": "open drawer"}], "stats": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "taxonomy_version": "old",
                        "instruction_hash": instruction_hash("open drawer"),
                        "primary_atomic_action": "打开",
                    }
                )
                + "\n{",
                encoding="utf-8",
            )
            enrich_instruction_classification(catalog, path)
        self.assertEqual(catalog["records"][0]["primary_atomic_action"], "待分类")


if __name__ == "__main__":
    unittest.main()
