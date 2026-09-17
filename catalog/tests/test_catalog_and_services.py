from __future__ import annotations

import importlib.util
import gzip
import json
import math
import sys
import threading
import types
import unittest
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from catalog.catalog import localize_native_scene, scan_catalog, write_catalog
from catalog.conversion_status import CONVERSION_STATUSES, enrich_conversion_status
from catalog.embedding import aggregate_task_terms, project_embeddings
from catalog.serve import ExplorerRequestHandler


class DatasetSemanticCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "datasets"
        self.root.mkdir()
        self._build_fixture()

    def _mkdir(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _touch(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def _write_text(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _build_fixture(self) -> None:
        self._touch("RoboMIND/benchmark1_1_compressed/h5_sim_franka_3rgb/open_drawer.tar.gz.part-aa")
        self._touch("RoboMIND/benchmark1_1_compressed/h5_sim_franka_3rgb/open_drawer.tar.gz.part-ab")
        self._touch("RoboMIND/benchmark1_1_compressed/h5_franka_3rgb/pick_cup.tar.gz.part-aa")
        self._mkdir("RoboMIND/failure_data/drop_bowl")
        self._mkdir("RoboMIND/failure_data/drop_bowl/collision/100-2026_08_31_12_00_00")
        self._mkdir("RoboMINDv2/franka/pick_apple")
        self._mkdir("RoboMINDv2/franka/pick_apple/success_episodes/episode-a")
        self._mkdir("RoboMINDv2/franka/pick_apple/success_episodes/episode-b")
        self._mkdir("RoboMINDv2/franka_sim/open_drawer")
        self._mkdir("RoboMINDv2/franka_sim/open_drawer/success_episodes/episode-a")
        self._touch("IPEC-COMMUNITY/FastUMI-Data/clean_table.tar.gz.part-0000")
        self._touch("IPEC-COMMUNITY/FastUMI-Data/clean_table.tar.gz.part-0001")
        self._mkdir("IPEC-COMMUNITY/FastUMI_100k_lerobot/dual_arm/fold_towel")
        self._write_text(
            "IPEC-COMMUNITY/FastUMI_100k_lerobot/dual_arm/fold_towel/meta/info.json",
            '{"total_episodes": 2}',
        )
        self._mkdir("OpenDriveLab/FreeTacMan/plug_charger")
        self._touch("OpenDriveLab/FreeTacMan/plug_charger/plug_charger_0_traj.csv")
        self._mkdir("silicx/exUMI/insert_pen")
        self._touch("silicx/exUMI/insert_pen/000000.mp4")
        self._touch("silicx/exUMI/insert_pen/000001.mp4")
        self._mkdir("binghaohuang-robot/touch_in_the_wild-dataset/four_tasks/fluid_transfer")
        self._mkdir("binghaohuang-robot/touch_in_the_wild-dataset/in_the_wild_data/kitchen")
        realomin_task = (
            "genrobot2025/10Kh-RealOmin-OpenData/[STAGE 3]/domestic_services/kitchen/organization/" "organize_utensils"
        )
        self._mkdir(f"{realomin_task}/0")
        self._mkdir(f"{realomin_task}/1")
        self._touch("EgoDex/open_drawer.tar")
        self._mkdir("HiFi-UMI-2K-main/chunk-000/part-0000")

    def test_scans_supported_layouts_without_storage_duplicates(self) -> None:
        catalog = scan_catalog(self.root, generated_at="2026-08-31T00:00:00+08:00")

        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(
            [level["key"] for level in catalog["hierarchy"]],
            ["domain", "scene", "dataset", "task_type", "primary_atomic_action", "native_task"],
        )
        self.assertEqual(catalog["stats"]["records"], 12)
        self.assertEqual(catalog["stats"]["datasets"], 9)
        self.assertEqual(len({record["id"] for record in catalog["records"]}), 12)
        self.assertEqual(
            catalog["domain_labels"],
            {
                "real": "真机",
                "sim": "仿真",
                "umi": "UMI 数据集",
                "ego": "Ego",
                "vqa": "VQA 数据集",
            },
        )
        self.assertEqual(
            catalog["stats"]["by_domain"],
            {"real": 4, "sim": 2, "umi": 5, "ego": 1, "vqa": 0},
        )

        sim_tasks = [record for record in catalog["records"] if record["domain"] == "sim"]
        self.assertEqual({record["native_task"] for record in sim_tasks}, {"open_drawer"})
        self.assertEqual(len(sim_tasks), 2)

        realomin = next(record for record in catalog["records"] if record["dataset"] == "10Kh-RealOmin")
        self.assertEqual(realomin["native_scene"], "kitchen")
        self.assertEqual(realomin["scene"], "厨房与餐饮")
        self.assertEqual(realomin["scene_source"], "native_name")
        self.assertEqual(realomin["native_group"], "organization")
        self.assertEqual(realomin["episode_count"], 2)

        self.assertEqual(localize_native_scene("bedroom"), "卧室")
        self.assertEqual(localize_native_scene("living_room"), "客厅")
        self.assertEqual(localize_native_scene("kitchen"), "厨房与餐饮")
        self.assertEqual(localize_native_scene("study"), "书房")
        self.assertEqual(localize_native_scene("custom_scene"), "custom_scene")

        fastumi = [record for record in catalog["records"] if record["dataset"] == "FastUMI-Data"]
        self.assertEqual(len(fastumi), 1)
        self.assertEqual(fastumi[0]["native_task"], "clean_table")

        umi_datasets = {
            "FastUMI-Data",
            "FastUMI-100K",
            "FreeTacMan",
            "Touch-in-the-Wild",
            "exUMI",
        }
        self.assertTrue(
            all(record["domain"] == "umi" for record in catalog["records"] if record["dataset"] in umi_datasets)
        )

        unresolved_datasets = {item["dataset"] for item in catalog["unresolved"]}
        self.assertIn("Touch-in-the-Wild", unresolved_datasets)
        self.assertIn("HiFi-UMI-2K", unresolved_datasets)
        touch_unresolved = next(item for item in catalog["unresolved"] if item["dataset"] == "Touch-in-the-Wild")
        self.assertEqual(touch_unresolved["domain"], "umi")

        references = {item["dataset"]: item for item in catalog["reference_datasets"]}
        self.assertEqual(catalog["stats"]["reference_datasets"], 11)
        self.assertEqual(catalog["stats"]["reference_datasets_downloaded"], 3)
        self.assertEqual(references["HiFi-UMI-2K"]["availability"], "downloaded")
        self.assertEqual(references["FastUMI-100K"]["availability"], "downloaded")
        self.assertEqual(references["FastUMI-Data"]["availability"], "downloaded")
        self.assertEqual(references["UMI Original Dataset"]["availability"], "not_downloaded")
        self.assertEqual(references["RefSpatial"]["availability"], "not_downloaded")

        episode_counts = {
            (record["dataset"], record["native_task"]): record["episode_count"] for record in catalog["records"]
        }
        self.assertEqual(episode_counts[("RoboMIND", "drop_bowl")], 1)
        self.assertEqual(episode_counts[("RoboMINDv2", "pick_apple")], 2)
        self.assertEqual(episode_counts[("RoboMINDv2", "open_drawer")], 1)
        self.assertEqual(episode_counts[("FastUMI-100K", "fold_towel")], 2)
        self.assertEqual(episode_counts[("FreeTacMan", "plug_charger")], 1)
        self.assertEqual(episode_counts[("exUMI", "insert_pen")], 2)
        self.assertEqual(episode_counts[("10Kh-RealOmin", "organize_utensils")], 2)
        self.assertEqual(catalog["stats"]["records_with_episode_count"], 7)
        self.assertEqual(catalog["stats"]["episodes_known_total"], 11)

    def test_hifi_metadata_is_aggregated_into_task_rows(self) -> None:
        meta = self._mkdir("HiFi-UMI-2K-main/chunk-000/part-0000/meta")
        self._write_text(
            "HiFi-UMI-2K-main/chunk-000/part-0000/meta/info.json",
            '{"robot_type":"simpleai_umi_4.0"}',
        )
        self._touch("HiFi-UMI-2K-main/chunk-000/part-0000/meta/episodes/chunk-000/file-000.parquet")

        class FakeColumn:
            def to_pylist(self):
                return [["Pick up the cup."], ["Pick up the cup.", "Open the drawer."], None]

        class FakeBatch:
            def column(self, _index):
                return FakeColumn()

        class FakeParquetFile:
            schema_arrow = types.SimpleNamespace(names=["tasks"])

            def __init__(self, _path):
                pass

            def iter_batches(self, **_kwargs):
                return [FakeBatch()]

        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = FakeParquetFile
        pyarrow.parquet = parquet
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            catalog = scan_catalog(self.root, generated_at="fixed")

        records = [record for record in catalog["records"] if record["dataset"] == "HiFi-UMI-2K"]
        self.assertEqual({record["native_task"] for record in records}, {"Pick up the cup.", "Open the drawer."})
        self.assertEqual(
            {record["native_task"]: record["episode_count"] for record in records},
            {"Pick up the cup.": 2, "Open the drawer.": 1},
        )
        self.assertTrue(all(record["source_path"] == "HiFi-UMI-2K-main" for record in records))
        self.assertNotIn("HiFi-UMI-2K", {item["dataset"] for item in catalog["unresolved"]})

    def test_instruction_inventory_adds_metadata_only_datasets_without_duplicate_outputs(self) -> None:
        self._mkdir("EgoVerse")
        self._mkdir("Something-Something-V2")
        self._mkdir("tencent/Hy-Embodied-0.5-VLA-Data")
        inventory = self._write_text(
            "instruction_inventory.jsonl",
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "dataset": "EgoVerse",
                        "source_kind": "egoverse_zarr",
                        "instruction": "Pick up the cup",
                        "occurrence_kind": "episode",
                    },
                    {
                        "dataset": "EgoVerse",
                        "source_kind": "egoverse_zarr",
                        "instruction": "  pick up the cup  ",
                        "occurrence_kind": "episode",
                    },
                    {
                        "dataset": "Something-Something-V2",
                        "source_kind": "something_something_labels",
                        "instruction": "Putting a mug on a table",
                        "instruction_template": "Putting [something] on [something]",
                        "occurrence_kind": "video",
                    },
                    {
                        "dataset": "Something-Something-V2",
                        "source_kind": "something_something_labels",
                        "instruction": "Putting a pen on a book",
                        "instruction_template": "Putting something on something",
                        "occurrence_kind": "video",
                    },
                    {
                        "dataset": "Hy-Embodied-0.5-VLA-Data",
                        "source_kind": "hy_embodied_task_parquet",
                        "instruction": "Open the drawer",
                        "occurrence_kind": "task_definition",
                    },
                    {
                        "dataset": "RoboMIND-2.0-full",
                        "source_kind": "lerobot_episode_jsonl",
                        "instruction": "Duplicate converted task",
                        "occurrence_kind": "episode",
                    },
                )
            )
            + "\n",
        )

        catalog = scan_catalog(self.root, generated_at="fixed", instruction_inventory=inventory)
        records = {
            (record["dataset"], record["native_task"]): record
            for record in catalog["records"]
            if record["classification_source"] == "metadata_instruction_inventory"
        }

        self.assertEqual(len(records), 3)
        self.assertEqual(records[("EgoVerse", "Pick up the cup")]["episode_count"], 2)
        self.assertEqual(
            records[("Something-Something-V2", "Putting [something] on [something]")]["episode_count"],
            2,
        )
        self.assertIsNone(records[("Hy-Embodied-0.5-VLA-Data", "Open the drawer")]["episode_count"])
        self.assertNotIn("RoboMIND-2.0-full", {record["dataset"] for record in catalog["records"]})
        unresolved = {item["dataset"] for item in catalog["unresolved"]}
        self.assertNotIn("EgoVerse", unresolved)
        self.assertNotIn("Something-Something-V2", unresolved)
        self.assertNotIn("Hy-Embodied-0.5-VLA-Data", unresolved)

    def test_metadata_only_dataset_conversion_evidence_is_conservative(self) -> None:
        self._mkdir("EgoVerse")
        self._mkdir("Something-Something-V2")
        inventory = self._write_text(
            "instruction_inventory.jsonl",
            json.dumps(
                {
                    "dataset": "Something-Something-V2",
                    "source_kind": "something_something_labels",
                    "instruction": "Putting a mug on a table",
                    "instruction_template": "Putting [something] on [something]",
                    "occurrence_kind": "video",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "dataset": "EgoVerse",
                    "source_kind": "egoverse_zarr",
                    "instruction": "Pick up the cup",
                    "occurrence_kind": "episode",
                }
            )
            + "\n",
        )
        pfs_root = Path(self.temporary_directory.name) / "converted"
        output_meta = pfs_root / "Something-Something-V2" / "meta"
        output_meta.mkdir(parents=True)
        (output_meta / "info.json").write_text('{"total_episodes": 2}', encoding="utf-8")
        (output_meta / "canonical_80d.json").write_text("{}", encoding="utf-8")
        (output_meta / "tasks.jsonl").write_text(
            '{"task":"Putting [something] on [something]"}\n', encoding="utf-8"
        )

        catalog = scan_catalog(self.root, generated_at="fixed", instruction_inventory=inventory)
        enrich_conversion_status(catalog, pfs_root=pfs_root, checked_at="fixed")
        statuses = {
            record["dataset"]: record
            for record in catalog["records"]
            if record["dataset"] in {"EgoVerse", "Something-Something-V2"}
        }

        self.assertEqual(statuses["Something-Something-V2"]["conversion_status"], "partial")
        self.assertEqual(
            statuses["Something-Something-V2"]["conversion_source"], "lerobot_metadata_collection"
        )
        self.assertEqual(statuses["EgoVerse"]["conversion_status"], "not_converted")

    def test_vqa_scanner_only_adds_downloaded_allowlisted_roots(self) -> None:
        self._mkdir("IPEC-COMMUNITY/EO-Data1.5M")
        self._mkdir("JingkunAn/RefSpatial")
        self._mkdir("Random/UnknownVQA")

        catalog = scan_catalog(self.root, generated_at="fixed")

        vqa_records = [record for record in catalog["records"] if record["domain"] == "vqa"]
        self.assertEqual({record["dataset"] for record in vqa_records}, {"EO-Data1.5M", "RefSpatial"})
        self.assertTrue(all(record["scene"] == "视觉问答与空间理解" for record in vqa_records))
        self.assertTrue(all(record["scene_source"] == "manual" for record in vqa_records))
        self.assertTrue(all(record["task_type"] == "EmbodiedVQA" for record in vqa_records))
        self.assertTrue(all(record["task_type_source"] == "manual" for record in vqa_records))
        self.assertEqual(
            {record["source_path"] for record in vqa_records},
            {"IPEC-COMMUNITY/EO-Data1.5M", "JingkunAn/RefSpatial"},
        )
        self.assertNotIn("UnknownVQA", {record["dataset"] for record in vqa_records})

        references = {item["dataset"]: item for item in catalog["reference_datasets"]}
        self.assertEqual(references["EO-Data1.5M"]["availability"], "downloaded")
        self.assertEqual(references["RefSpatial"]["availability"], "downloaded")
        self.assertEqual(references["Omni3D"]["availability"], "not_downloaded")

    def test_lerobot_episode_metadata_requires_a_nonnegative_integer(self) -> None:
        base = "IPEC-COMMUNITY/FastUMI_100k_lerobot/dual_arm"
        self._write_text(f"{base}/zero_episodes/meta/info.json", '{"total_episodes": 0}')
        self._write_text(f"{base}/boolean_count/meta/info.json", '{"total_episodes": true}')
        self._write_text(f"{base}/invalid_json/meta/info.json", "{")

        catalog = scan_catalog(self.root, generated_at="fixed")
        counts = {
            record["native_task"]: record["episode_count"]
            for record in catalog["records"]
            if record["dataset"] == "FastUMI-100K"
        }

        self.assertEqual(counts["zero_episodes"], 0)
        self.assertIsNone(counts["boolean_count"])
        self.assertIsNone(counts["invalid_json"])

    def test_ids_and_order_are_deterministic(self) -> None:
        first = scan_catalog(self.root, generated_at="fixed")
        second = scan_catalog(self.root, generated_at="fixed")

        self.assertEqual(first, second)

    def test_conversion_status_defaults_to_unknown_without_output_mounts(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")

        enrich_conversion_status(catalog)

        self.assertTrue(all(record["conversion_status"] == "unknown" for record in catalog["records"]))
        self.assertEqual(
            catalog["stats"]["conversion_by_status"],
            {status: len(catalog["records"]) if status == "unknown" else 0 for status in CONVERSION_STATUSES},
        )
        self.assertIsNone(catalog["conversion"]["pfs_output_root"])
        self.assertIsNone(catalog["conversion"]["cfs_output_root"])

    def test_conversion_status_uses_only_structured_output_evidence(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        output_base = Path(self.temporary_directory.name) / "conversion_outputs"
        pfs_root = output_base / "pfs"
        cfs_root = output_base / "cfs"

        def write_json(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

        def write_jsonl(path: Path, payloads: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(f"{json.dumps(payload)}\n" for payload in payloads), encoding="utf-8")

        write_json(
            pfs_root / "RoboMIND-1.0-full-clean/cleaning_report.json",
            {
                "source_root": "/mnt/cfs/data/datasets/RoboMIND",
                "output_root": (pfs_root / "RoboMIND-1.0-full-clean").as_posix(),
                "total_episodes": 10,
                "total_frames": 100,
                "unresolved_errors": 0,
            },
        )
        write_json(
            pfs_root / "RoboMIND-1.0-full-clean/conversion_summary.json",
            {"datasets": [{"episodes": 10, "frames": 100}]},
        )
        write_json(
            pfs_root / "RoboMIND-2.0-full/cleaning_report.json",
            {
                "source_root": "/mnt/cfs/data/datasets/RoboMINDv2",
                "output_root": (pfs_root / "RoboMIND-2.0-full").as_posix(),
                "total_episodes": 20,
                "total_frames": 200,
                "unresolved_errors": 0,
            },
        )
        write_json(
            pfs_root / "RoboMIND-2.0-full/conversion_summary.json",
            {"datasets": [{"episodes": 20, "frames": 200}]},
        )
        write_json(
            pfs_root / "FastUMI-Data-80D-full-compact/.done.json",
            {"valid": True, "episodes": 1, "parquet_footers": 1, "done_markers": 1},
        )
        write_jsonl(
            pfs_root / "FastUMI-Data-80D-full-compact/meta/tasks.jsonl",
            [{"task": "clean table"}],
        )
        write_json(
            pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel/meta/info.json",
            {"total_episodes": 2},
        )
        write_json(
            pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel/meta/canonical_80d.json",
            {"schema_version": 1},
        )
        write_jsonl(
            pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel/meta/tasks.jsonl",
            [{"task": "fold a towel"}],
        )
        for marker in ("a.done", "b.done"):
            marker_path = pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel/.canonical_status" / marker
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
        write_json(pfs_root / "EgoDex/meta/info.json", {"total_episodes": 5})
        write_json(pfs_root / "EgoDex/meta/canonical_80d.json", {"schema_version": 1})
        write_jsonl(pfs_root / "EgoDex/meta/episodes.jsonl", [{"task_family": "open drawer"}])
        write_json(
            cfs_root / "FreeTacMan-80D-formal-v1/.done.json",
            {
                "valid": True,
                "roots": [{"valid": True, "episodes": 1, "parquet_footers": 1, "done_markers": 1}],
            },
        )
        write_jsonl(
            cfs_root / "FreeTacMan-80D-formal-v1/2cam/meta/tasks.jsonl",
            [{"task": "plug_charger"}],
        )
        write_json(
            cfs_root / "Touch-in-the-Wild-80D-clean-v4/processed_zarr/.done.json",
            {
                "valid": True,
                "episodes": 1,
                "parquet_footers": 1,
                "done_markers": 1,
                "exclusions": {"valid": True},
            },
        )
        write_jsonl(
            cfs_root / "Touch-in-the-Wild-80D-clean-v4/processed_zarr/meta/tasks.jsonl",
            [{"task": "four_tasks/fluid_transfer/fluid_transfer"}],
        )
        write_json(
            cfs_root / "exUMI-80D-formal-v4/.done.json",
            {
                "valid": True,
                "roots": [{"valid": True, "episodes": 1, "parquet_footers": 1, "done_markers": 1}],
            },
        )
        write_jsonl(
            cfs_root / "exUMI-80D-formal-v4/camera_only/meta/tasks.jsonl",
            [{"task": "insert_pen"}],
        )
        write_json(
            cfs_root
            / "10Kh-RealOmin-OpenData.formal-fragments-v2/fragments/chunk-000000/episode-000000000/validation.json",
            {"valid": True},
        )

        enrich_conversion_status(catalog, pfs_root=pfs_root, cfs_root=cfs_root, checked_at="checked")

        by_dataset: dict[str, list[dict]] = {}
        for record in catalog["records"]:
            by_dataset.setdefault(record["dataset"], []).append(record)

        robomind = by_dataset["RoboMIND"]
        failure = next(record for record in robomind if "failure_data" in record["source_path"])
        successes = [record for record in robomind if record is not failure]
        self.assertEqual(failure["conversion_status"], "excluded")
        self.assertTrue(all(record["conversion_status"] == "converted" for record in successes))
        self.assertTrue(all(record["conversion_status"] == "converted" for record in by_dataset["RoboMINDv2"]))
        for dataset in ("FastUMI-Data", "FastUMI-100K", "FreeTacMan", "Touch-in-the-Wild", "exUMI", "EgoDex"):
            self.assertTrue(all(record["conversion_status"] == "converted" for record in by_dataset[dataset]))
        self.assertTrue(all(record["conversion_status"] == "partial" for record in by_dataset["10Kh-RealOmin"]))
        self.assertTrue(
            all(record["conversion_checked_at"] == "checked" for record in catalog["records"]),
        )
        self.assertEqual(sum(catalog["stats"]["conversion_by_status"].values()), len(catalog["records"]))

    def test_realomin_published_multi_root_catalog_marks_dataset_converted(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        pfs_root = Path(self.temporary_directory.name) / "pfs"
        output = pfs_root / "10Kh-RealOmin-OpenData-80D-multiroot-v1"
        aggregate_info = output / "meta/aggregate_info.json"
        aggregate_info.parent.mkdir(parents=True)
        aggregate_info.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "valid": True,
                    "published": True,
                    "source_modified": False,
                    "source_mcap": 105,
                    "structurally_readable_mcap": 100,
                    "upstream_corrupt_mcap": 5,
                    "source_exclusions": 2,
                    "episodes": 98,
                    "parquet_files": 98,
                    "videos": 196,
                    "formal_root": output.as_posix(),
                    "evidence": {
                        "committed_markers": 98,
                        "valid_sources": 100,
                        "active_failures": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

        enrich_conversion_status(catalog, pfs_root=pfs_root, checked_at="checked")

        records = [record for record in catalog["records"] if record["dataset"] == "10Kh-RealOmin"]
        self.assertTrue(records)
        self.assertTrue(all(record["conversion_status"] == "converted" for record in records))
        self.assertTrue(all(record["conversion_output_path"] == output.as_posix() for record in records))
        self.assertTrue(
            all(record["conversion_source"] == "validated_multi_root_catalog" for record in records),
        )

    def test_fastumi_missing_exact_task_is_not_converted_only_when_output_root_has_evidence(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        template = next(record for record in catalog["records"] if record["dataset"] == "FastUMI-100K")
        missing = {**template, "id": "missing-task", "native_task": "missing_task"}
        catalog["records"].append(missing)
        pfs_root = Path(self.temporary_directory.name) / "pfs"
        info = pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel/meta/info.json"
        info.parent.mkdir(parents=True)
        info.write_text('{"total_episodes": 2}', encoding="utf-8")
        (info.parent / "canonical_80d.json").write_text("{}", encoding="utf-8")
        (info.parent / "tasks.jsonl").write_text('{"task":"fold towel"}\n', encoding="utf-8")
        marker_root = info.parents[1] / ".canonical_status"
        marker_root.mkdir()
        (marker_root / "a.done").touch()
        (marker_root / "b.done").touch()

        enrich_conversion_status(catalog, pfs_root=pfs_root)

        self.assertEqual(template["conversion_status"], "converted")
        self.assertEqual(missing["conversion_status"], "not_converted")
        self.assertIsNone(missing["conversion_evidence_path"])

    def test_complete_metadata_datasets_use_dataset_specific_evidence(self) -> None:
        pfs_root = Path(self.temporary_directory.name) / "pfs"

        def write_json(path: Path, payload) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

        def touch_many(root: Path, names: list[str]) -> None:
            root.mkdir(parents=True, exist_ok=True)
            for name in names:
                (root / name).touch()

        def make_single_root(name: str, status: str, *, stats: bool) -> Path:
            output = pfs_root / name
            write_json(output / "meta/info.json", {"total_episodes": 2, "total_frames": 20, "total_videos": 2})
            write_json(output / "meta/canonical_80d.json", {})
            (output / "meta/tasks.jsonl").write_text('{"task":"task"}\n', encoding="utf-8")
            touch_many(output / "data/chunk-000", ["episode_000000.parquet", "episode_000001.parquet"])
            touch_many(output / "videos/chunk-000/cam", ["episode_000000.mp4", "episode_000001.mp4"])
            touch_many(output / status / "episodes", ["0.json", "1.json"])
            if stats:
                touch_many(output / status / "stats", ["0.npz", "1.npz"])
            return output

        galaxea_source = self._mkdir("Galaxea-Open-World-Dataset/lerobot")
        (galaxea_source / "Task_A.tar.gz").touch()
        write_json(
            self.root / "Galaxea-Open-World-Dataset/.aihc_download_complete",
            {"success_count": 1, "failure_count": 0},
        )
        galaxea_output = pfs_root / "Galaxea-Open-World/Task_A"
        write_json(
            galaxea_output / "meta/info.json",
            {"total_episodes": 2, "total_frames": 20, "total_videos": 4},
        )
        write_json(galaxea_output / "meta/canonical_80d.json", {})
        (galaxea_output / "meta/tasks.jsonl").write_text('{"task":"task"}\n', encoding="utf-8")
        touch_many(galaxea_output / "data/chunk-000", ["0.parquet", "1.parquet"])
        touch_many(galaxea_output / "videos/chunk-000/cam", [f"{index}.mp4" for index in range(4)])

        write_json(
            self.root / "tencent/Hy-Embodied-0.5-VLA-Data/tables.json",
            {
                "total_tables": 1,
                "total_episodes": 2,
                "tables": [{"table_name": "table_000", "total_episodes": 2}],
            },
        )
        hy_output = pfs_root / "Hy-Embodied-0.5-VLA-Data-80D-starvla-v3"
        write_json(
            hy_output / "preparation_report.json",
            {
                "status": "passed",
                "stage": "conversion_and_cpu_loader_acceptance",
                "partial_scope": False,
                "official_and_starvla_loader_acceptance": "passed",
                "tables": 1,
                "episodes": 2,
                "rows": 20,
                "shards": 2,
            },
        )
        write_json(
            hy_output / "conversion/manifest.json",
            {"partial_scope": False, "tables": ["table_000"], "total_episodes": 2, "total_rows": 20, "total_shards": 2},
        )
        write_json(
            hy_output / "conversion/status.json",
            {
                "tables": ["table_000"],
                "total_episodes": 2,
                "total_rows": 20,
                "total_shards": 2,
                "completed_shards": 2,
            },
        )
        write_json(
            hy_output / "conversion/official_reader_acceptance.json",
            {"status": "passed", "tables": [{"table": "table_000"}]},
        )
        write_json(
            hy_output / "conversion/starvla_loader_acceptance/summary.json",
            {"status": "passed", "tables": [{"table": "table_000", "episodes": 2, "frames": 20}]},
        )
        write_json(
            hy_output / "table_000/preparation_report.json",
            {
                "status": "passed",
                "stage": "conversion_integrity",
                "partial_scope": False,
                "episodes": 2,
                "rows": 20,
                "files": 2,
            },
        )
        write_json(
            hy_output / "table_000/meta/info.json",
            {"total_episodes": 2, "total_frames": 20, "total_files": 2, "total_videos": 6},
        )
        write_json(hy_output / "table_000/meta/canonical_80d.json", {})
        (hy_output / "table_000/meta/tasks.parquet").touch()
        touch_many(hy_output / "table_000/data/chunk-000", ["0.parquet", "1.parquet"])

        let_output = make_single_root("LET-claw", ".let_status", stats=True)
        write_json(let_output / ".let_status/reindex_manifest.json", {"status": "COMPLETE", "episodes": 2})
        write_json(let_output / ".let_status/audit.json", {"status": "PASS", "episodes": 2, "frames": 20})
        write_json(
            let_output / ".let_status/progress.json",
            {"completed": 3, "total": 3, "failures": []},
        )
        write_json(let_output / ".let_status/failures.json", [])
        write_json(let_output / ".let_status/conversion_exclusions.json", [{"source": "bad"}])

        make_single_root("RT-1", ".rt1_status", stats=True)

        something_output = make_single_root("Something-Something-V2", ".something_v2_status", stats=False)
        labels_path = self.root / "Something-Something-V2/20bn-something-something-download-package-labels.zip"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(labels_path, "w") as archive:
            archive.writestr("labels/train.json", json.dumps([{"id": 1}]))
            archive.writestr("labels/validation.json", json.dumps([{"id": 2}]))
            archive.writestr("labels/test.json", json.dumps([]))
            archive.writestr("labels/labels.json", json.dumps({str(index): index for index in range(174)}))

        datasets = (
            "Galaxea-Open-World",
            "Hy-Embodied-0.5-VLA-Data",
            "LET-claw",
            "RT-1",
            "Something-Something-V2",
        )
        catalog = {
            "dataset_root": self.root.as_posix(),
            "generated_at": "fixed",
            "records": [
                {"id": f"record-{index}", "dataset": dataset, "native_task": "task", "source_path": dataset}
                for index, dataset in enumerate(datasets)
            ],
        }

        enrich_conversion_status(catalog, pfs_root=pfs_root, checked_at="checked")

        self.assertEqual(
            {record["dataset"]: record["conversion_status"] for record in catalog["records"]},
            {dataset: "converted" for dataset in datasets},
        )
        self.assertEqual(something_output.name, "Something-Something-V2")

    def test_hifi_conversion_parts_are_reported_as_partial(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        template = dict(catalog["records"][0])
        template.update(
            {
                "id": "hifi-task",
                "dataset": "HiFi-UMI-2K",
                "native_task": "Pick up the cup.",
                "source_path": "HiFi-UMI-2K-main",
            }
        )
        catalog["records"] = [template]
        pfs_root = Path(self.temporary_directory.name) / "pfs"
        meta = pfs_root / "HiFi-UMI-2K-80D-full/chunk-000/part-0000/meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text('{"total_episodes":3}', encoding="utf-8")
        (meta / "canonical_80d.json").write_text("{}", encoding="utf-8")
        (meta / "tasks.parquet").touch()

        class FakeColumn:
            def to_pylist(self):
                return ["Pick up the cup."]

        class FakeBatch:
            def column(self, _index):
                return FakeColumn()

        class FakeParquetFile:
            schema_arrow = types.SimpleNamespace(names=["task"])

            def __init__(self, _path):
                pass

            def iter_batches(self, **_kwargs):
                return [FakeBatch()]

        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = FakeParquetFile
        pyarrow.parquet = parquet
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            enrich_conversion_status(catalog, pfs_root=pfs_root, checked_at="checked")

        record = catalog["records"][0]
        self.assertEqual(record["conversion_status"], "partial")
        self.assertEqual(record["conversion_source"], "lerobot_part_metadata")
        self.assertIn("3 个转换回合", record["conversion_note"])

    def test_hifi_matching_source_and_output_partitions_are_converted(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        template = dict(catalog["records"][0])
        template.update(
            {
                "id": "hifi-task",
                "dataset": "HiFi-UMI-2K",
                "native_task": "Pick up the cup.",
                "source_path": "HiFi-UMI-2K-main",
            }
        )
        catalog["records"] = [template]
        source_meta = self.root / "HiFi-UMI-2K-main/chunk-000/part-0000/meta"
        source_meta.mkdir(parents=True, exist_ok=True)
        (source_meta / "info.json").write_text('{"total_episodes":3}', encoding="utf-8")
        pfs_root = Path(self.temporary_directory.name) / "pfs"
        output_meta = pfs_root / "HiFi-UMI-2K-80D-full/chunk-000/part-0000/meta"
        output_meta.mkdir(parents=True)
        (output_meta / "info.json").write_text('{"total_episodes":3}', encoding="utf-8")
        (output_meta / "canonical_80d.json").write_text("{}", encoding="utf-8")
        (output_meta / "tasks.parquet").touch()

        class FakeColumn:
            def to_pylist(self):
                return ["Pick up the cup."]

        class FakeBatch:
            def column(self, _index):
                return FakeColumn()

        class FakeParquetFile:
            schema_arrow = types.SimpleNamespace(names=["task"])

            def __init__(self, _path):
                pass

            def iter_batches(self, **_kwargs):
                return [FakeBatch()]

        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = FakeParquetFile
        pyarrow.parquet = parquet
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            enrich_conversion_status(catalog, pfs_root=pfs_root, checked_at="checked")

        record = catalog["records"][0]
        self.assertEqual(record["conversion_status"], "converted")
        self.assertEqual(record["conversion_source"], "lerobot_partition_parity")
        self.assertIn("逐一一致", record["conversion_note"])

    def test_embedding_terms_are_aggregated_by_scene_and_task(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        points = aggregate_task_terms(catalog["records"])

        self.assertEqual(len(points), len({(record["scene"], record["native_task"]) for record in catalog["records"]}))
        self.assertEqual(len(points), len({point["id"] for point in points}))
        self.assertEqual(
            sorted(record_id for point in points for record_id in point["record_ids"]),
            sorted(record["id"] for record in catalog["records"]),
        )
        open_drawer_points = [point for point in points if point["task"] == "open_drawer"]
        self.assertEqual(len(open_drawer_points), 2)
        self.assertEqual({point["scene"] for point in open_drawer_points}, {"仿真通用场景", "收纳与家具"})

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None,
        "NumPy is required for embedding projection",
    )
    def test_embedding_projection_is_finite_and_deterministic(self) -> None:
        import numpy as np

        embeddings = np.asarray(
            [[1.0, 0.0, 0.2], [0.9, 0.1, 0.2], [0.0, 1.0, 0.3], [0.1, 0.8, 0.4]],
            dtype=np.float32,
        )
        first_coordinates, first_ratios = project_embeddings(embeddings)
        second_coordinates, second_ratios = project_embeddings(embeddings)

        self.assertEqual(first_coordinates.tolist(), second_coordinates.tolist())
        self.assertEqual(first_ratios, second_ratios)
        self.assertEqual(first_coordinates.shape, (4, 2))
        self.assertTrue(all(math.isfinite(value) for value in first_coordinates.ravel()))

    def test_visible_directory_scanner_does_not_follow_task_symlinks(self) -> None:
        external = Path(self.temporary_directory.name) / "external_task"
        external.mkdir()
        symlink = self.root / "OpenDriveLab/FreeTacMan/symlink_task"
        try:
            symlink.symlink_to(external, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")

        catalog = scan_catalog(self.root, generated_at="fixed")
        task_names = {record["native_task"] for record in catalog["records"]}
        self.assertNotIn("symlink_task", task_names)

    def test_write_catalog_is_valid_json(self) -> None:
        catalog = scan_catalog(self.root, generated_at="fixed")
        output = Path(self.temporary_directory.name) / "output" / "catalog.json"

        write_catalog(catalog, output)

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), catalog)
        self.assertFalse(output.with_suffix(".json.tmp").exists())


class ExplorerServerTest(unittest.TestCase):
    def test_serves_supplied_catalog_and_static_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            catalog_path = temporary_path / "catalog.json"
            catalog_path.write_text('{"schema_version": 1, "records": []}', encoding="utf-8")
            web_root = Path(__file__).parents[1] / "web"
            ExplorerRequestHandler.catalog_path = catalog_path
            ExplorerRequestHandler.video_api_base_url = "http://127.0.0.1:8790"
            ExplorerRequestHandler.video_config_message = "test video service"
            handler = partial(ExplorerRequestHandler, directory=web_root.as_posix())
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(f"{base_url}/catalog.json", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
                self.assertEqual(json.loads(response.read()), {"schema_version": 1, "records": []})
            with urllib.request.urlopen(f"{base_url}/", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
                html = response.read().decode("utf-8")
                self.assertIn("数据集语义浏览器", html)
                self.assertIn("语义散点", html)
                self.assertIn('src="catalog.js?v=20260903-atomic-action"', html)
                self.assertRegex(html, r'src="app\.js\?v=[^"]+"')
                self.assertRegex(html, r'href="styles\.css\?v=[^"]+"')
            with urllib.request.urlopen(f"{base_url}/catalog.js", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "application/javascript; charset=utf-8")
                script = response.read().decode("utf-8")
                prefix = "globalThis.__DATASET_EXPLORER_CATALOG__="
                self.assertTrue(script.startswith(prefix))
                self.assertEqual(
                    json.loads(script[len(prefix) :].removesuffix(";\n")),
                    {"schema_version": 1, "records": []},
                )
            gzip_request = urllib.request.Request(
                f"{base_url}/catalog.js", headers={"Accept-Encoding": "gzip"}
            )
            with urllib.request.urlopen(gzip_request, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Encoding"], "gzip")
                self.assertEqual(response.headers["Vary"], "Accept-Encoding")
                self.assertIn(
                    b"globalThis.__DATASET_EXPLORER_CATALOG__=",
                    gzip.decompress(response.read()),
                )
            with urllib.request.urlopen(f"{base_url}/video-config.json", timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
                self.assertEqual(
                    json.loads(response.read()),
                    {
                        "schema_version": 1,
                        "enabled": True,
                        "api_base_url": "http://127.0.0.1:8790",
                        "message": "test video service",
                    },
                )

    def test_proxies_video_api_with_range_support(self) -> None:
        class VideoFixtureHandler(BaseHTTPRequestHandler):
            seen_requests: list[tuple[str, str, str | None, str | None]] = []

            def do_GET(self) -> None:
                self._respond(head_only=False)

            def do_HEAD(self) -> None:
                self._respond(head_only=True)

            def _respond(self, *, head_only: bool) -> None:
                self.seen_requests.append(
                    (self.command, self.path, self.headers.get("Range"), self.headers.get("Origin"))
                )
                if self.path.startswith("/v1/records/test/videos?"):
                    body = json.dumps({"record_id": "test", "items": []}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                elif self.path == "/v1/media/test-token" and self.headers.get("Range") == "bytes=0-0":
                    body = b"x"
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", "bytes 0-0/3")
                    self.send_header("Content-Length", "1")
                else:
                    body = b"missing"
                    self.send_response(404)
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(body)

            def log_message(self, message_format: str, *args: object) -> None:
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), VideoFixtureHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)

        with TemporaryDirectory() as temporary_directory:
            catalog_path = Path(temporary_directory) / "catalog.json"
            catalog_path.write_text('{"schema_version": 1, "records": []}', encoding="utf-8")
            web_root = Path(__file__).parents[1] / "web"
            old_base_url = ExplorerRequestHandler.video_api_base_url
            old_upstream = ExplorerRequestHandler.video_api_upstream
            old_message = ExplorerRequestHandler.video_config_message
            self.addCleanup(setattr, ExplorerRequestHandler, "video_api_base_url", old_base_url)
            self.addCleanup(setattr, ExplorerRequestHandler, "video_api_upstream", old_upstream)
            self.addCleanup(setattr, ExplorerRequestHandler, "video_config_message", old_message)
            ExplorerRequestHandler.catalog_path = catalog_path
            ExplorerRequestHandler.video_api_base_url = "./video-api"
            ExplorerRequestHandler.video_api_upstream = f"http://127.0.0.1:{upstream.server_address[1]}"
            ExplorerRequestHandler.video_config_message = "same-origin proxy ready"
            handler = partial(ExplorerRequestHandler, directory=web_root.as_posix())
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"

            with urllib.request.urlopen(f"{base_url}/video-config.json", timeout=2) as response:
                self.assertEqual(json.loads(response.read())["api_base_url"], "./video-api")
            with urllib.request.urlopen(
                f"{base_url}/video-api/v1/records/test/videos?offset=0&limit=24",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["record_id"], "test")
            media_request = urllib.request.Request(
                f"{base_url}/video-api/v1/media/test-token",
                headers={"Range": "bytes=0-0", "Origin": "https://example.invalid"},
            )
            with urllib.request.urlopen(media_request, timeout=2) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 0-0/3")
                self.assertEqual(response.read(), b"x")

        self.assertIn(("GET", "/v1/records/test/videos?offset=0&limit=24", None, None), VideoFixtureHandler.seen_requests)
        self.assertIn(("GET", "/v1/media/test-token", "bytes=0-0", None), VideoFixtureHandler.seen_requests)


if __name__ == "__main__":
    unittest.main()
