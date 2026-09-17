from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from catalog.instruction_inventory import (
    SOURCE_EGOVERSE,
    SOURCE_LEROBOT_JSONL,
    SOURCE_SSV2,
    ScanDiagnostics,
    build_instruction_inventory,
    make_record,
    normalize_instruction,
    scan_egoverse,
    scan_lerobot_jsonl,
    scan_something_something,
)


class InstructionInventoryTest(unittest.TestCase):
    def test_normalization_and_record_id_are_stable(self) -> None:
        self.assertEqual(normalize_instruction("  Pick\nUP  杯子 "), "pick up 杯子")
        kwargs = dict(
            domain="real",
            dataset="Demo",
            source_kind="fixture",
            source_unit="unit",
            source_record_id="1",
            occurrence_kind="episode",
            instruction="Pick up cup",
            instruction_field="task",
            metadata_path="/data/meta.json",
        )
        first = make_record(**kwargs)
        second = make_record(**kwargs)
        self.assertIsNotNone(first)
        self.assertEqual(first.record_id, second.record_id)

    def test_egoverse_reads_only_root_task_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "EgoVerse" / "episode-a"
            episode.mkdir(parents=True)
            (episode / "zarr.json").write_text(
                json.dumps(
                    {
                        "attributes": {
                            "task_name": "sort the stationery",
                            "embodiment": "eva_bimanual",
                            "total_frames": 42,
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = list(scan_egoverse(root))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source_kind, SOURCE_EGOVERSE)
            self.assertEqual(records[0].episode_id, "episode-a")
            self.assertEqual(records[0].episode_length, 42)

    def test_something_something_reads_all_labeled_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Something-Something-V2"
            directory.mkdir()
            archive = directory / "20bn-something-something-download-package-labels.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "labels/train.json",
                    json.dumps([{"id": "1", "label": "putting cup down", "template": "Putting [something] down"}]),
                )
                bundle.writestr(
                    "labels/validation.json",
                    json.dumps([{"id": "2", "label": "lifting a bowl", "template": "Lifting [something]"}]),
                )
                bundle.writestr("labels/test-answers.csv", "3;Moving something up\n")
            records = list(scan_something_something(root))
            self.assertEqual([record.split for record in records], ["train", "validation", "test"])
            self.assertTrue(all(record.source_kind == SOURCE_SSV2 for record in records))

    def test_egoverse_manifest_limits_scan_to_downloaded_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "EgoVerse"
            for episode_id in ("episode-a", "episode-b"):
                episode = dataset / episode_id
                episode.mkdir(parents=True)
                (episode / "zarr.json").write_text(
                    json.dumps({"attributes": {"task_name": f"task {episode_id}"}}), encoding="utf-8"
                )
            download = dataset / ".egoverse-download"
            download.mkdir()
            (download / "active_episodes.tsv").write_text(
                "episode_hash\tzarr_processed_path\n"
                "episode-b\ts3://bucket/episode-b.zarr\n"
                "episode-missing\ts3://bucket/episode-missing.zarr\n",
                encoding="utf-8",
            )
            diagnostics = ScanDiagnostics()
            records = list(scan_egoverse(root, diagnostics, metadata_workers=2))
            self.assertEqual([record.episode_id for record in records], ["episode-b"])
            self.assertEqual(diagnostics.missing_metadata_files[SOURCE_EGOVERSE], 1)

    def test_lerobot_jsonl_prefers_episode_task_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            conversion_root = Path(temporary) / "converted"
            meta = conversion_root / "RoboMIND-2.0-full" / "franka-cams" / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text(json.dumps({"robot_type": "franka"}), encoding="utf-8")
            (meta / "tasks.jsonl").write_text('{"task_index":0,"task":"fallback task"}\n', encoding="utf-8")
            (meta / "episodes.jsonl").write_text(
                '{"episode_index":7,"tasks":["open the drawer"],"length":88}\n', encoding="utf-8"
            )
            smoke_meta = conversion_root / ".writer-smoke" / "meta"
            smoke_meta.mkdir(parents=True)
            (smoke_meta / "episodes.jsonl").write_text(
                '{"episode_index":0,"tasks":["do not include"]}\n', encoding="utf-8"
            )
            records = list(scan_lerobot_jsonl([conversion_root]))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source_kind, SOURCE_LEROBOT_JSONL)
            self.assertEqual(records[0].episode_id, "7")
            self.assertEqual(records[0].instruction, "open the drawer")

    def test_build_writes_jsonl_and_summary_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            root.mkdir()
            episode = root / "EgoVerse" / "episode-a"
            episode.mkdir(parents=True)
            (episode / "zarr.json").write_text(
                json.dumps({"attributes": {"task_description": "wipe the table"}}), encoding="utf-8"
            )
            output = Path(temporary) / "inventory.jsonl"
            summary_output = Path(temporary) / "summary.json"
            summary = build_instruction_inventory(
                dataset_root=root,
                output=output,
                summary_output=summary_output,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["unique_instructions"], 1)
            self.assertEqual(rows[0]["instruction_field"], "attributes.task_description")
            self.assertEqual(json.loads(summary_output.read_text(encoding="utf-8"))["records"], 1)


if __name__ == "__main__":
    unittest.main()
