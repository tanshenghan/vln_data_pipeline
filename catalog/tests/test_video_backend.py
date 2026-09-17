from __future__ import annotations

import json
import subprocess
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from catalog.video_index import (
    build_video_index,
    format_segment_video_relative_path,
    format_video_relative_path,
    validate_video_path_template,
)
from catalog.video_serve import (
    AssetTokenManager,
    InvalidAssetToken,
    RangeNotSatisfiable,
    TranscodeManager,
    VideoRequestHandler,
    VideoService,
    VideoServiceError,
    build_argparser,
    parse_byte_range,
)


class DatasetVideoIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.pfs_root = self.base / "pfs"
        self.cfs_root = self.base / "cfs"
        self.native_root = self.base / "native"
        self.records = [
            {
                "id": "robomindv2-pick",
                "dataset": "RoboMINDv2",
                "native_group": "franka",
                "native_task": "pick_apple",
                "source_path": "RoboMINDv2/franka/pick_apple",
            },
            {
                "id": "robomindv2-other",
                "dataset": "RoboMINDv2",
                "native_group": "franka",
                "native_task": "pick_apple",
                "source_path": "RoboMINDv2/franka/other_pick_apple",
            },
            {
                "id": "robomind-one-a",
                "dataset": "RoboMIND",
                "native_group": "franka",
                "native_task": "pick_cup",
                "source_path": "RoboMIND/archive_a/pick_cup",
            },
            {
                "id": "robomind-one-b",
                "dataset": "RoboMIND",
                "native_group": "franka",
                "native_task": "pick_cup",
                "source_path": "RoboMIND/archive_b/pick_cup",
            },
            {
                "id": "fastumi-missing",
                "dataset": "FastUMI-100K",
                "native_group": "dual_arm",
                "native_task": "fold_towel",
                "source_path": "IPEC-COMMUNITY/FastUMI_100k_lerobot/dual_arm/fold_towel",
            },
            {
                "id": "touch-linked",
                "dataset": "Touch-in-the-Wild",
                "native_group": "four_tasks",
                "native_task": "fluid_transfer",
                "source_path": "binghaohuang-robot/touch_in_the_wild-dataset/four_tasks/fluid_transfer",
            },
            {
                "id": "realomin-unmapped",
                "dataset": "10Kh-RealOmin",
                "native_group": "organization",
                "native_task": "organize_utensils",
                "source_path": "genrobot2025/10Kh-RealOmin-OpenData/task/organize_utensils",
            },
            {
                "id": "realomin-other",
                "dataset": "10Kh-RealOmin",
                "native_group": "organization",
                "native_task": "organize_utensils",
                "source_path": "genrobot2025/10Kh-RealOmin-OpenData/other/organize_utensils",
            },
        ]
        self.catalog = {"schema_version": 1, "records": self.records}
        self._build_lerobot_fixture()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, payloads: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{json.dumps(payload)}\n" for payload in payloads), encoding="utf-8")

    @staticmethod
    def _video_info(camera_keys: list[str], total_episodes: int) -> dict:
        return {
            "total_episodes": total_episodes,
            "chunks_size": 1000,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {key: {"dtype": "video", "shape": [8, 8, 3]} for key in camera_keys},
        }

    def _video_path(self, root: Path, episode_index: int, camera_key: str) -> Path:
        return root / f"videos/chunk-000/{camera_key}/episode_{episode_index:06d}.mp4"

    def _build_lerobot_fixture(self) -> None:
        robomindv2 = self.pfs_root / "RoboMIND-2.0-full/franka-cams-test"
        self._write_json(robomindv2 / "meta/info.json", self._video_info(["observation.images.front", "wrist"], 2))
        self._write_jsonl(
            robomindv2 / "meta/episodes.jsonl",
            [
                {
                    "episode_index": 0,
                    "tasks": ["pick apple"],
                    "source_hdf5": (
                        "/mnt/cfs/data/datasets/RoboMINDv2/franka/pick_apple/success_episodes/episode-0/data.hdf5"
                    ),
                },
                {"episode_index": 1, "tasks": ["pick apple"]},
            ],
        )
        for camera in ("observation.images.front", "wrist"):
            path = self._video_path(robomindv2, 0, camera)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"0123456789")

        robomind = self.pfs_root / "RoboMIND-1.0-full-clean/franka-cams-test"
        self._write_json(robomind / "meta/info.json", self._video_info(["front"], 1))
        self._write_jsonl(robomind / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["pick cup"]}])
        robomind_video = self._video_path(robomind, 0, "front")
        robomind_video.parent.mkdir(parents=True, exist_ok=True)
        robomind_video.write_bytes(b"robomind")

        fastumi = self.pfs_root / "FastUMI-100K-80D-full/dual_arm/fold_towel"
        self._write_json(fastumi / "meta/info.json", self._video_info(["front"], 1))
        self._write_jsonl(fastumi / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["fold towel"]}])

        touch = self.cfs_root / "Touch-in-the-Wild-80D-clean-v4/processed_zarr"
        self._write_json(touch / "meta/info.json", self._video_info(["camera"], 1))
        self._write_jsonl(
            touch / "meta/episodes.jsonl",
            [{"episode_index": 0, "tasks": ["four_tasks/fluid_transfer/fluid_transfer"]}],
        )
        native_video = self.native_root / "touch/source.mp4"
        native_video.parent.mkdir(parents=True, exist_ok=True)
        native_video.write_bytes(b"linked-video")
        touch_video = self._video_path(touch, 0, "camera")
        touch_video.parent.mkdir(parents=True, exist_ok=True)
        try:
            touch_video.symlink_to(native_video)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        for global_index in (123, 124):
            realomin = (
                self.cfs_root
                / "10Kh-RealOmin-OpenData.formal-fragments-v2/fragments/chunk-000000"
                / f"episode-{global_index:09d}"
            )
            self._write_json(realomin / "meta/info.json", self._video_info(["front"], 1))
            self._write_jsonl(
                realomin / "meta/episodes.jsonl",
                [
                    {
                        "episode_index": 0,
                        "source_mcap": (
                            "/mnt/cfs/data/datasets/genrobot2025/10Kh-RealOmin-OpenData/task/"
                            f"organize_utensils/episode-{global_index}.mcap"
                        ),
                    }
                ],
            )
            realomin_video = self._video_path(realomin, 0, "front")
            realomin_video.parent.mkdir(parents=True, exist_ok=True)
            realomin_video.write_bytes(b"mapped-realomin")
        state = self.cfs_root / ".realomin_formal_fragments_v2_state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "status").write_text("COMPLETE\n", encoding="utf-8")
        self._write_jsonl(
            state / "valid_sources.jsonl",
            [
                {"index": 123, "source": "task/organize_utensils/episode-123.mcap"},
                {"index": 124, "source": "task/organize_utensils/episode-124.mcap"},
            ],
        )
        self._write_json(
            self.pfs_root / "10Kh-RealOmin-OpenData-80D-multiroot-v1/meta/aggregate_info.json",
            {"status": "COMPLETE", "valid": True, "published": True, "episodes": 2, "videos": 4},
        )

    def build_index(self, *, max_episodes_per_record: int = 200) -> dict:
        return build_video_index(
            self.catalog,
            pfs_root=self.pfs_root,
            cfs_root=self.cfs_root,
            generated_at="fixed",
            max_episodes_per_record=max_episodes_per_record,
        )

    def test_builds_compact_mappings_and_separates_conversion_from_playability(self) -> None:
        index = self.build_index()
        records = index["records"]

        robomindv2 = records["robomindv2-pick"]
        self.assertEqual(robomindv2["video_status"], "available")
        self.assertEqual(robomindv2["indexed_episode_count"], 1)
        self.assertEqual(robomindv2["indexed_video_count"], 2)
        self.assertEqual(robomindv2["existing_video_count"], 2)
        self.assertEqual(len(robomindv2["sources"]), 1)
        self.assertEqual(len(robomindv2["sources"][0]["episodes"]), 1)
        self.assertEqual(records["robomindv2-other"]["video_status"], "unmapped")

        self.assertEqual(records["robomind-one-a"]["video_status"], "partial")
        self.assertEqual(records["robomind-one-b"]["video_status"], "partial")
        self.assertEqual(records["robomind-one-a"]["mapping_statuses"], ["task_name"])

        fastumi = records["fastumi-missing"]
        self.assertEqual(fastumi["video_status"], "absent")
        self.assertEqual(fastumi["indexed_video_count"], 1)
        self.assertEqual(fastumi["existing_video_count"], 0)

        touch = records["touch-linked"]
        self.assertEqual(touch["video_status"], "linked")
        self.assertEqual(touch["linked_video_count"], 1)
        realomin = records["realomin-unmapped"]
        self.assertEqual(realomin["video_status"], "available")
        self.assertEqual(realomin["mapping_statuses"], ["source_path"])
        self.assertEqual(realomin["total_episode_count"], 2)
        self.assertEqual(realomin["indexed_episode_count"], 2)
        self.assertEqual(realomin["sources"][0]["episodes"][0]["episode_index"], 0)
        self.assertEqual(realomin["sources"][0]["episodes"][0]["display_episode_index"], 123)
        self.assertEqual(records["realomin-other"]["video_status"], "unmapped")

        self.assertEqual(index["stats"]["by_video_status"]["absent"], 1)
        self.assertEqual(index["stats"]["by_video_status"]["unmapped"], 2)
        self.assertNotIn("asset_id", json.dumps(index))

    def test_episode_cap_marks_record_partial_without_expanding_cameras(self) -> None:
        index = self.build_index(max_episodes_per_record=1)
        record = index["records"]["robomindv2-pick"]

        self.assertFalse(record["truncated"])
        self.assertEqual(record["indexed_episode_count"], 1)
        self.assertEqual(record["indexed_video_count"], 2)
        self.assertEqual(
            set(record["sources"][0]),
            {
                "root",
                "video_path_template",
                "chunks_size",
                "camera_keys",
                "source_codec",
                "episodes",
                "existing_video_count",
                "linked_video_count",
                "media_status",
            },
        )
        realomin = index["records"]["realomin-unmapped"]
        self.assertEqual(realomin["total_episode_count"], 2)
        self.assertEqual(realomin["indexed_episode_count"], 1)
        self.assertTrue(realomin["truncated"])

    def test_hifi_shared_video_uses_a_separate_episode_locator_cap(self) -> None:
        with patch("catalog.video_index._add_hifi_v3_mappings") as add_hifi:
            index = build_video_index(
                self.catalog,
                pfs_root=self.pfs_root,
                cfs_root=self.cfs_root,
                max_episodes_per_record=1,
                max_hifi_episodes_per_record=321,
            )

        self.assertEqual(add_hifi.call_args.kwargs["max_episodes_per_record"], 321)
        self.assertEqual(index["limits"]["max_episodes_per_record"], 1)
        self.assertEqual(index["limits"]["max_hifi_episodes_per_record"], 321)

    def test_completed_standard_and_hy_segmented_outputs_are_discovered(self) -> None:
        datasets = {
            "galaxea": ("Galaxea-Open-World", self.pfs_root / "Galaxea-Open-World/task-a"),
            "let": ("LET-claw", self.pfs_root / "LET-claw"),
            "rt1": ("RT-1", self.pfs_root / "RT-1"),
            "robocoin": ("RoboCOIN", self.pfs_root / "RoboCOIN/task-a"),
            "something": ("Something-Something-V2", self.pfs_root / "Something-Something-V2"),
        }
        records = []
        for record_id, (dataset, output) in datasets.items():
            task = f"{record_id} task"
            records.append(
                {
                    "id": record_id,
                    "dataset": dataset,
                    "native_group": "group",
                    "native_task": task,
                    "source_path": f"source/{record_id}",
                }
            )
            self._write_json(output / "meta/info.json", self._video_info(["front"], 1))
            self._write_jsonl(output / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
            self._write_jsonl(output / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": [task]}])
            video = self._video_path(output, 0, "front")
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")

        records.append(
            {
                "id": "hy-a",
                "dataset": "Hy-Embodied-0.5-VLA-Data",
                "native_group": "table",
                "native_task": "拿起红色杯子",
                "source_path": "source/hy-a",
            }
        )
        records.append(
            {
                "id": "hy-b",
                "dataset": "Hy-Embodied-0.5-VLA-Data",
                "native_group": "table",
                "native_task": "放下蓝色杯子",
                "source_path": "source/hy-b",
            }
        )
        hy = self.pfs_root / "Hy-Embodied-0.5-VLA-Data-80D-starvla-v3/table_000"
        self._write_json(
            hy / "meta/info.json",
            {
                "total_episodes": 1,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.images.cam_high": {"dtype": "video"},
                    "observation.images.cam_left_wrist": {"dtype": "video"},
                    "observation.images.cam_right_wrist": {"dtype": "video"},
                },
            },
        )

        segmented_rows = [
            {"episode_index": 0, "tasks": ["拿起红色杯子"]},
            {"episode_index": 1, "tasks": ["放下蓝色杯子"]},
        ]
        for camera in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            for segmented_row in segmented_rows:
                segmented_row[f"videos/{camera}/chunk_index"] = 0
                segmented_row[f"videos/{camera}/file_index"] = 0
                segmented_row[f"videos/{camera}/from_timestamp"] = float(segmented_row["episode_index"])
                segmented_row[f"videos/{camera}/to_timestamp"] = float(segmented_row["episode_index"] + 1)
        with patch(
            "catalog.video_index._iter_hifi_episode_rows",
            side_effect=lambda *_args: iter(segmented_rows),
        ):
            index = build_video_index({"records": records}, pfs_root=self.pfs_root)

        for record_id in datasets:
            self.assertEqual(index["records"][record_id]["video_status"], "available")
            self.assertEqual(index["records"][record_id]["indexed_episode_count"], 1)
        for record_id in ("hy-a", "hy-b"):
            self.assertEqual(index["records"][record_id]["video_status"], "available")
            self.assertEqual(index["records"][record_id]["indexed_episode_count"], 1)
            self.assertEqual(index["records"][record_id]["indexed_video_count"], 3)
            self.assertIsNone(index["records"][record_id]["video_note"])

    def test_video_template_rejects_path_escape_and_unsafe_camera(self) -> None:
        with self.assertRaises(ValueError):
            validate_video_path_template("../{video_key}/episode_{episode_index}.mp4")
        with self.assertRaises(ValueError):
            validate_video_path_template("videos/{unknown}/episode_{episode_index}.mp4")
        with self.assertRaises(ValueError):
            format_video_relative_path(
                "videos/{video_key}/episode_{episode_index}.mp4",
                chunks_size=1000,
                episode_index=0,
                camera_key="../camera",
            )
        self.assertEqual(
            format_segment_video_relative_path(
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                chunk_index=2,
                file_index=4,
                camera_key="front",
            ).as_posix(),
            "videos/front/chunk-002/file-004.mp4",
        )
        with self.assertRaises(ValueError):
            format_segment_video_relative_path(
                "../{video_key}/chunk-{chunk_index}/file-{file_index}.mp4",
                chunk_index=0,
                file_index=0,
                camera_key="front",
            )


class AssetTokenAndRangeTest(unittest.TestCase):
    def test_asset_token_rejects_tampering_and_expiry(self) -> None:
        manager = AssetTokenManager(b"a sufficiently long secret", ttl_seconds=10)
        token = manager.issue({"r": "record", "s": 0, "p": 1, "c": 2}, now=100)

        self.assertEqual(manager.verify(token, now=109)["r"], "record")
        with self.assertRaises(InvalidAssetToken):
            manager.verify(f"{token[:-1]}A", now=109)
        with self.assertRaises(InvalidAssetToken):
            manager.verify(token, now=110)
        with self.assertRaises(InvalidAssetToken):
            manager.verify("é.bad", now=100)

    def test_single_range_parser_supports_open_and_suffix_ranges(self) -> None:
        self.assertIsNone(parse_byte_range(None, 10))
        self.assertEqual(parse_byte_range("bytes=2-5", 10), (2, 5))
        self.assertEqual(parse_byte_range("bytes=7-", 10), (7, 9))
        self.assertEqual(parse_byte_range("bytes=-4", 10), (6, 9))
        self.assertEqual(parse_byte_range("bytes=0-99", 10), (0, 9))
        for value in ("items=0-1", "bytes=", "bytes=5-2", "bytes=20-", "bytes=0-1,3-4", "bytes=-0"):
            with self.subTest(value=value), self.assertRaises(RangeNotSatisfiable):
                parse_byte_range(value, 10)

    def test_transcode_cli_controls_are_explicit(self) -> None:
        args = build_argparser().parse_args(
            [
                "--index",
                "video-index.json",
                "--secret-file",
                "video-api.secret",
                "--cors-origin",
                "https://dataset.example.test",
                "--ffmpeg",
                "none",
                "--transcode-cache",
                "video-cache",
                "--transcode-workers",
                "2",
                "--transcode-threads",
                "3",
                "--transcode-timeout",
                "45",
                "--transcode-wait-timeout",
                "12",
                "--transcode-cache-max-gib",
                "8",
                "--transcode-min-free-gib",
                "2",
            ]
        )

        self.assertEqual(args.ffmpeg, "none")
        self.assertEqual(args.transcode_cache, Path("video-cache"))
        self.assertEqual(args.transcode_workers, 2)
        self.assertEqual(args.transcode_threads, 3)
        self.assertEqual(args.transcode_timeout, 45)
        self.assertEqual(args.transcode_wait_timeout, 12)
        self.assertEqual(args.transcode_cache_max_gib, 8)
        self.assertEqual(args.transcode_min_free_gib, 2)


class TranscodeManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.source = self.base / "source.mp4"
        self.source.write_bytes(b"source-video-bytes")
        self.cache = self.base / "cache"

    def manager(self, **overrides) -> TranscodeManager:
        cache_root = overrides.pop("cache_root", self.cache)
        options = {
            "ffmpeg": "/fake/ffmpeg",
            "ffprobe": "/fake/ffprobe",
            "forbidden_roots": (),
            "workers": 1,
            "threads": 2,
            "timeout_seconds": 5,
            "wait_timeout_seconds": 2,
            "cache_max_bytes": 32 * 1024 * 1024,
            "min_free_bytes": 0,
        }
        options.update(overrides)
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ):
            return TranscodeManager(cache_root, **options)

    @staticmethod
    def successful_process(args, **kwargs):
        if args[0] == "/fake/ffmpeg":
            Path(args[-1]).write_bytes(b"h264-compatible-video")
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(args, 0, stdout="h264\n", stderr="")

    def test_atomic_cache_uses_source_fingerprint_and_validates_h264(self) -> None:
        manager = self.manager()
        with patch(
            "catalog.video_serve.subprocess.run",
            side_effect=self.successful_process,
        ) as run:
            first = manager.materialize(self.source)
            second = manager.materialize(self.source)
            self.source.write_bytes(b"changed-source-video-with-a-new-size")
            third = manager.materialize(self.source)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(first.read_bytes(), b"h264-compatible-video")
        self.assertEqual(third.read_bytes(), b"h264-compatible-video")
        self.assertEqual(sum(call.args[0][0] == "/fake/ffmpeg" for call in run.call_args_list), 2)
        ffmpeg_call = next(call for call in run.call_args_list if call.args[0][0] == "/fake/ffmpeg")
        self.assertIn("-fs", ffmpeg_call.args[0])
        self.assertIs(ffmpeg_call.kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(ffmpeg_call.kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in self.cache.rglob("*")))

    def test_same_key_is_single_flight_and_different_key_is_busy(self) -> None:
        manager = self.manager()
        second_source = self.base / "second.mp4"
        second_source.write_bytes(b"second-source")
        started = threading.Event()
        release = threading.Event()
        ffmpeg_calls = []

        def blocking_process(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                ffmpeg_calls.append(args)
                started.set()
                self.assertTrue(release.wait(2))
                Path(args[-1]).write_bytes(b"h264-compatible-video")
                return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
            return subprocess.CompletedProcess(args, 0, stdout="h264\n", stderr="")

        results: list[Path] = []
        errors: list[Exception] = []

        def materialize_source() -> None:
            try:
                results.append(manager.materialize(self.source))
            except Exception as exc:
                errors.append(exc)

        with patch("catalog.video_serve.subprocess.run", side_effect=blocking_process):
            leader = threading.Thread(target=materialize_source)
            follower = threading.Thread(target=materialize_source)
            leader.start()
            self.assertTrue(started.wait(2))
            follower.start()
            with self.assertRaises(VideoServiceError) as caught:
                manager.materialize(second_source)
            self.assertEqual(caught.exception.status, 503)
            self.assertEqual(caught.exception.code, "transcode_busy")
            release.set()
            leader.join(2)
            follower.join(2)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(ffmpeg_calls), 1)

    def test_output_budget_reservation_is_atomic_across_workers(self) -> None:
        manager = self.manager(
            cache_root=self.base / "shared-budget-cache",
            workers=2,
            cache_max_bytes=12 * 1024 * 1024,
        )
        second_source = self.base / "second.mp4"
        second_source.write_bytes(b"second-source")
        started = threading.Event()
        release = threading.Event()
        errors: list[Exception] = []

        def blocking_process(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                started.set()
                self.assertTrue(release.wait(2))
                Path(args[-1]).write_bytes(b"h264-compatible-video")
                return subprocess.CompletedProcess(args, 0)
            return subprocess.CompletedProcess(args, 0, stdout="h264\n", stderr="")

        def first_job() -> None:
            try:
                manager.materialize(self.source)
            except Exception as exc:
                errors.append(exc)

        with patch("catalog.video_serve.subprocess.run", side_effect=blocking_process):
            thread = threading.Thread(target=first_job)
            thread.start()
            self.assertTrue(started.wait(2))
            with self.assertRaises(VideoServiceError) as full:
                manager.materialize(second_source)
            self.assertEqual(full.exception.status, 507)
            self.assertEqual(full.exception.code, "transcode_cache_full")
            release.set()
            thread.join(2)
        self.assertEqual(errors, [])

    def test_failed_transcode_is_not_published_and_capacity_is_enforced(self) -> None:
        manager = self.manager()

        def failing_process(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                Path(args[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"sensitive path")
            return subprocess.CompletedProcess(args, 0, stdout="mpeg4\n", stderr="")

        with patch("catalog.video_serve.subprocess.run", side_effect=failing_process):
            with self.assertRaises(VideoServiceError) as caught:
                manager.materialize(self.source)
        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(caught.exception.code, "transcode_failed")
        self.assertNotIn(self.base.as_posix(), caught.exception.message)
        self.assertFalse(any(path.suffix == ".mp4" for path in self.cache.rglob("*.mp4")))
        self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in self.cache.rglob("*")))

        full_manager = self.manager(cache_root=self.base / "full-cache", cache_max_bytes=1)
        with self.assertRaises(VideoServiceError) as full:
            full_manager.materialize(self.source)
        self.assertEqual(full.exception.status, 507)
        self.assertEqual(full.exception.code, "transcode_cache_full")

        limited_manager = self.manager(cache_root=self.base / "limited-cache", cache_max_bytes=9 * 1024 * 1024)

        def fill_limit(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                limit = int(args[args.index("-fs") + 1])
                with Path(args[-1]).open("wb") as handle:
                    handle.truncate(limit)
                return subprocess.CompletedProcess(args, 0)
            return subprocess.CompletedProcess(args, 0, stdout="h264\n", stderr="")

        with patch("catalog.video_serve.subprocess.run", side_effect=fill_limit):
            with self.assertRaises(VideoServiceError) as exceeded:
                limited_manager.materialize(self.source)
        self.assertEqual(exceeded.exception.status, 507)
        self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in (self.base / "limited-cache").rglob("*")))

    def test_source_change_is_not_published(self) -> None:
        manager = self.manager()

        def mutate_source(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                Path(args[-1]).write_bytes(b"h264-compatible-video")
                self.source.write_bytes(b"source-mutated-during-transcode")
                return subprocess.CompletedProcess(args, 0)
            return subprocess.CompletedProcess(args, 0, stdout="h264\n", stderr="")

        with patch("catalog.video_serve.subprocess.run", side_effect=mutate_source):
            with self.assertRaises(VideoServiceError) as changed:
                manager.materialize(self.source)
        self.assertEqual(changed.exception.status, 422)
        self.assertEqual(changed.exception.code, "transcode_source_changed")
        self.assertFalse(any(path.suffix == ".mp4" for path in self.cache.rglob("*.mp4")))

    def test_cache_rejects_symlink_and_dataset_root(self) -> None:
        dataset_root = self.base / "dataset"
        dataset_root.mkdir()
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ):
            with self.assertRaises(ValueError):
                TranscodeManager(
                    dataset_root / "cache",
                    ffmpeg="/fake/ffmpeg",
                    ffprobe="/fake/ffprobe",
                    forbidden_roots=(dataset_root,),
                )
        target = self.base / "target-cache"
        target.mkdir()
        symlink = self.base / "cache-link"
        try:
            symlink.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ):
            with self.assertRaises(ValueError):
                TranscodeManager(
                    symlink,
                    ffmpeg="/fake/ffmpeg",
                    ffprobe="/fake/ffprobe",
                    forbidden_roots=(),
                )

    def test_tool_validation_fails_before_cache_creation(self) -> None:
        cache = self.base / "invalid-tool-cache"
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ):
            with self.assertRaises(ValueError):
                TranscodeManager(
                    cache,
                    ffmpeg="/missing/ffmpeg",
                    ffprobe="/missing/ffprobe",
                    forbidden_roots=(),
                )
        self.assertFalse(cache.exists())

        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            TranscodeManager(
                self.base / "validated-tool-cache",
                ffmpeg="/fake/ffmpeg",
                ffprobe="/fake/ffprobe",
                forbidden_roots=(),
            )
        self.assertEqual(run.call_count, 2)
        encoder_call = run.call_args_list[1]
        self.assertIn("libx264", encoder_call.args[0])
        self.assertIs(encoder_call.kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(encoder_call.kwargs["stderr"], subprocess.DEVNULL)


class DatasetVideoServerTest(DatasetVideoIndexTest):
    # Reuse the realistic fixture helpers without rerunning the index-only test
    # methods under a live HTTP server.
    test_builds_compact_mappings_and_separates_conversion_from_playability = None
    test_episode_cap_marks_record_partial_without_expanding_cameras = None
    test_video_template_rejects_path_escape_and_unsafe_camera = None

    def setUp(self) -> None:
        super().setUp()
        self.index = self.build_index()
        segmented_root = self.pfs_root / "segmented"
        segmented_video = segmented_root / "videos/front/chunk-000/file-000.mp4"
        segmented_video.parent.mkdir(parents=True, exist_ok=True)
        segmented_video.write_bytes(b"shared-video")
        self.index["records"]["segmented"] = {
            "record_id": "segmented",
            "dataset": "HiFi-UMI-2K",
            "native_task": "shared video task",
            "video_status": "available",
            "total_episode_count": 1,
            "indexed_episode_count": 1,
            "total_video_count": 1,
            "sources": [
                {
                    "root": segmented_root.as_posix(),
                    "video_path_template": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    "chunks_size": 1000,
                    "camera_keys": ["front"],
                    "source_codec": "h264",
                    "segmented_media": True,
                    "episodes": [
                        {
                            "episode_index": 3,
                            "display_episode_index": 42,
                            "task": "shared video task",
                            "camera_segments": {"front": [0, 0, 12.5, 18.75]},
                        }
                    ],
                }
            ],
        }
        self.allowed_origin = "https://dataset.example.test"
        self.service = VideoService(
            self.index,
            secret=b"server-side hmac secret material",
            media_roots=[self.pfs_root, self.cfs_root, self.native_root],
            cors_origins=[self.allowed_origin],
            public_base_url="https://media.example.test",
            ffprobe=None,
        )
        VideoRequestHandler.service = self.service
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), VideoRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def request(self, path: str, *, method: str = "GET", origin: str | None = None, range_value: str | None = None):
        headers = {}
        if origin:
            headers["Origin"] = origin
        if range_value:
            headers["Range"] = range_value
        request = urllib.request.Request(f"{self.base_url}{path}", headers=headers, method=method)
        return urllib.request.urlopen(request, timeout=2)

    def test_list_api_returns_status_paginated_items_and_relative_identity(self) -> None:
        with self.request(
            "/v1/records/robomindv2-pick/videos?offset=1&limit=1",
            origin=self.allowed_origin,
        ) as response:
            payload = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], self.allowed_origin)
        self.assertEqual(payload["video_status"], "available")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["indexed_total"], 2)
        self.assertEqual(payload["source_total"], 2)
        self.assertEqual(payload["offset"], 1)
        self.assertEqual(len(payload["items"]), 1)
        item = payload["items"][0]
        self.assertEqual(item["playback_status"], "unverified")
        self.assertEqual(item["source_codec"], "unknown")
        self.assertTrue(item["stream_url"].startswith("https://media.example.test/v1/media/"))
        self.assertNotIn(self.pfs_root.as_posix(), json.dumps(payload))

        with self.request("/v1/records/fastumi-missing/videos", origin=self.allowed_origin) as response:
            missing = json.loads(response.read())
        self.assertEqual(missing["video_status"], "absent")
        self.assertEqual(missing["items"][0]["playback_status"], "missing")
        self.assertIsNone(missing["items"][0]["stream_url"])

        with self.request("/v1/records/realomin-unmapped/videos", origin=self.allowed_origin) as response:
            realomin = json.loads(response.read())
        self.assertEqual(realomin["video_status"], "available")
        self.assertEqual(realomin["source_episode_count"], 2)
        self.assertEqual(realomin["items"][0]["episode_index"], 123)

        with self.request("/v1/records/segmented/videos", origin=self.allowed_origin) as response:
            segmented = json.loads(response.read())
        self.assertEqual(segmented["items"][0]["episode_index"], 42)
        self.assertEqual(segmented["items"][0]["start_seconds"], 12.5)
        self.assertEqual(segmented["items"][0]["end_seconds"], 18.75)

    def test_media_get_and_head_support_byte_ranges(self) -> None:
        with self.request("/v1/records/robomindv2-pick/videos?limit=1", origin=self.allowed_origin) as response:
            item = json.loads(response.read())["items"][0]
        stream_path = urlsplit_path(item["stream_url"])

        with self.request(stream_path, origin=self.allowed_origin, range_value="bytes=2-5") as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertEqual(response.read(), b"2345")

        with self.request(stream_path, method="HEAD", origin=self.allowed_origin, range_value="bytes=-3") as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 7-9/10")
            self.assertEqual(response.headers["Content-Length"], "3")
            self.assertEqual(response.read(), b"")

        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(stream_path, origin=self.allowed_origin, range_value="bytes=100-")
        self.assertEqual(caught.exception.code, 416)
        self.assertEqual(caught.exception.headers["Content-Range"], "bytes */10")

    def test_exact_cors_and_realpath_allowlists_are_enforced(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/records/robomindv2-pick/videos", origin="https://evil.example")
        self.assertEqual(caught.exception.code, 403)
        self.assertIsNone(caught.exception.headers.get("Access-Control-Allow-Origin"))

        blocked_service = VideoService(
            self.index,
            secret=b"another server-side hmac secret",
            media_roots=[self.pfs_root, self.cfs_root],
            cors_origins=[self.allowed_origin],
            ffprobe=None,
        )
        blocked = blocked_service.list_videos("touch-linked", offset=0, limit=1)
        self.assertEqual(blocked["items"][0]["playback_status"], "blocked")
        self.assertIsNone(blocked["items"][0]["stream_url"])

        outside_index = json.loads(json.dumps(self.index))
        outside_index["records"]["robomindv2-pick"]["sources"][0]["root"] = self.native_root.as_posix()
        with self.assertRaises(ValueError):
            VideoService(
                outside_index,
                secret=b"another server-side hmac secret",
                media_roots=[self.native_root],
                cors_origins=[self.allowed_origin],
            )

    def test_preflight_returns_exact_cors_policy(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1/records/robomindv2-pick/videos",
            method="OPTIONS",
            headers={"Origin": self.allowed_origin, "Access-Control-Request-Method": "GET"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], self.allowed_origin)
            self.assertIn("Range", response.headers["Access-Control-Allow-Headers"])
            self.assertIn("Retry-After", response.headers["Access-Control-Expose-Headers"])

    def test_codec_cache_uses_each_file_fingerprint(self) -> None:
        source = self.index["records"]["robomindv2-pick"]["sources"][0]
        first = self._video_path(Path(source["root"]), 0, "observation.images.front")
        second = self._video_path(Path(source["root"]), 0, "wrist")
        self.service.ffprobe = "/fake/ffprobe"

        with patch(
            "catalog.video_serve.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="h264\n", stderr=""),
                subprocess.CompletedProcess([], 0, stdout="hevc\n", stderr=""),
            ],
        ) as run:
            self.assertEqual(self.service._probe_codec(source, "camera", first), "h264")
            self.assertEqual(self.service._probe_codec(source, "camera", second), "hevc")
            self.assertEqual(self.service._probe_codec(source, "camera", first), "h264")

        self.assertEqual(run.call_count, 2)

    def test_non_h264_is_transcoded_once_then_streamed_with_ranges(self) -> None:
        cache = self.base / "transcode-cache"
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ):
            service = VideoService(
                self.index,
                secret=b"transcode server-side hmac secret",
                media_roots=[self.pfs_root, self.cfs_root, self.native_root],
                cors_origins=[self.allowed_origin],
                public_base_url="https://media.example.test",
                ffprobe="/fake/ffprobe",
                ffmpeg="/fake/ffmpeg",
                transcode_cache=cache,
                transcode_workers=1,
                transcode_threads=2,
                transcode_cache_max_bytes=32 * 1024 * 1024,
                transcode_min_free_bytes=0,
            )
        VideoRequestHandler.service = service
        resolved_cache = cache.resolve()
        ffmpeg_commands: list[list[str]] = []

        def transcode_process(args, **kwargs):
            if args[0] == "/fake/ffmpeg":
                ffmpeg_commands.append(args)
                Path(args[-1]).write_bytes(b"H264-TRANSCODED-VIDEO")
                return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
            probed_path = Path(args[-1])
            codec = "h264" if resolved_cache in probed_path.parents else "mpeg4"
            return subprocess.CompletedProcess(args, 0, stdout=f"{codec}\n", stderr="")

        with patch("catalog.video_serve.subprocess.run", side_effect=transcode_process):
            with self.request("/v1/records/robomindv2-pick/videos?limit=1", origin=self.allowed_origin) as response:
                before = json.loads(response.read())["items"][0]
            self.assertEqual(before["playback_status"], "transcode_required")
            self.assertEqual(before["source_codec"], "mpeg4")
            self.assertEqual(before["playback_codec"], "h264")
            self.assertFalse(before["transcoded"])
            self.assertIsNotNone(before["stream_url"])

            stream_path = urlsplit_path(before["stream_url"])
            with self.request(stream_path, origin=self.allowed_origin, range_value="bytes=0-3") as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 0-3/21")
                self.assertEqual(response.read(), b"H264")

            with self.request("/v1/records/robomindv2-pick/videos?limit=1", origin=self.allowed_origin) as response:
                after = json.loads(response.read())["items"][0]
            self.assertEqual(after["playback_status"], "ready")
            self.assertEqual(after["source_codec"], "mpeg4")
            self.assertEqual(after["playback_codec"], "h264")
            self.assertTrue(after["transcoded"])

            cached_stream = urlsplit_path(after["stream_url"])
            with self.request(
                cached_stream, method="HEAD", origin=self.allowed_origin, range_value="bytes=-5"
            ) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Length"], "5")
                self.assertEqual(response.headers["Content-Range"], "bytes 16-20/21")

        self.assertEqual(len(ffmpeg_commands), 1)
        command = ffmpeg_commands[0]
        self.assertIn("libx264", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[command.index("-threads") + 1], "2")
        self.assertGreaterEqual(int(command[command.index("-fs") + 1]), 8 * 1024 * 1024)
        self.assertEqual(len(list(cache.glob("h264-v1/*/*.mp4"))), 1)
        self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in cache.rglob("*")))

    def test_transcoding_disabled_keeps_415_and_token_binds_source_fingerprint(self) -> None:
        service = VideoService(
            self.index,
            secret=b"disabled transcode hmac secret",
            media_roots=[self.pfs_root, self.cfs_root, self.native_root],
            cors_origins=[self.allowed_origin],
            ffprobe="/fake/ffprobe",
        )
        source = self._video_path(
            Path(self.index["records"]["robomindv2-pick"]["sources"][0]["root"]),
            0,
            "observation.images.front",
        )
        with patch(
            "catalog.video_serve.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="mpeg4\n", stderr=""),
        ):
            item = service.list_videos("robomindv2-pick", offset=0, limit=1)["items"][0]
            self.assertEqual(item["playback_status"], "unsupported")
            self.assertIsNone(item["stream_url"])
            with self.assertRaises(VideoServiceError) as unsupported:
                service.resolve_asset(item["asset_id"])
            self.assertEqual(unsupported.exception.status, 415)
            self.assertNotIn(source.as_posix(), unsupported.exception.message)

            source.write_bytes(b"source-file-was-replaced")
            with self.assertRaises(VideoServiceError) as stale:
                service.resolve_asset(item["asset_id"])
            self.assertEqual(stale.exception.status, 403)
            self.assertEqual(stale.exception.code, "invalid_asset")

    def test_explicit_transcode_cache_requires_both_tools(self) -> None:
        for ffmpeg, ffprobe in ((None, "/fake/ffprobe"), ("/fake/ffmpeg", None), (None, None)):
            with self.subTest(ffmpeg=ffmpeg, ffprobe=ffprobe), self.assertRaises(ValueError):
                VideoService(
                    self.index,
                    secret=b"fail-fast transcode hmac secret",
                    media_roots=[self.pfs_root, self.cfs_root, self.native_root],
                    cors_origins=[self.allowed_origin],
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    transcode_cache=self.base / "must-not-be-created",
                )
        self.assertFalse((self.base / "must-not-be-created").exists())


def urlsplit_path(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path


if __name__ == "__main__":
    unittest.main()
