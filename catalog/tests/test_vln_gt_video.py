"""GT source identity, lazy render caching, and browser media delivery."""
import json
import sqlite3
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from catalog.vln_catalog import build_database, connect, overview
from catalog.vln_gt_video import GTVideoService, build_index
from catalog.vln_serve import VLNRequestHandler
from catalog.vln_video_collection import build_collection


class GTVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "lerobot"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
        (self.dataset / "meta/beta_nav.json").write_text(json.dumps({"episodes": [
            {"episode_id": i, "scene_id": "mp3d/scan1/scan1.glb", "instruction": instruction,
             "frames": 41, "episode_index": i-1}
            for i, instruction in [(1, "Walk forward. "), (2, "Wrong instruction.")]
        ]}))
        annotations = self.root / "annotations/train"
        annotations.mkdir(parents=True)
        (annotations / "train.json").write_text(json.dumps({"episodes": [
            {"episode_id": i, "scene_id": "mp3d/scan1/scan1.glb", "instruction": {"instruction_text": "Walk forward."},
             "start_position": [0, 0, 0], "reference_path": [[0, 0, 0], [0, 0, 1]], "goals": [{"position": [0, 0, 1]}]}
            for i in (1, 2)
        ]}))
        config = self.root / "sources.json"
        config.write_text(json.dumps({"sources": [{"id": "test-r2r", "dataset": "R2R", "format": "vlnce",
                                                 "version": "test", "root": "annotations", "splits": ["train"],
                                                 "pattern": "{split}/{split}.json"}]}))
        self.catalog = self.root / "catalog.sqlite"
        build_database(config, self.catalog)
        self.index = self.root / "videos.sqlite"
        self.config = build_index(self.catalog, self.dataset, self.index, self.root / "videos", self.root / "render.py",
                                  Path("/usr/bin/python3"), "test-r2r")
        with connect(self.catalog) as db:
            self.meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
            self.ids = {row["episode_id"]: row["id"] for row in db.execute("SELECT id,episode_id FROM episodes")}
        self.service = GTVideoService(self.index, self.meta)
        self.addCleanup(self.service.pool.shutdown, wait=True)

    def complete(self, record_id):
        paths = self.service.paths(record_id)
        paths["video"].parent.mkdir(parents=True, exist_ok=True)
        paths["video"].write_bytes(b"0123456789")
        paths["poster"].write_bytes(b"poster")
        paths["report"].write_text(json.dumps(self.report(record_id)))

    def report(self, record_id):
        row = self.service.records[record_id]
        return {"fps": 8, "size": [470, 280], "horizon": 10, "layout": "rgb_only", "frames": 41,
                "instruction": row["instruction"], "input_provenance": {
                    "episode_index": row["episode_index"], "dataset_root": str(self.dataset),
                    "info_sha256": self.config["info_sha256"]}}

    def test_exact_instruction_binding_and_gt_filter_ignore_model_media(self):
        self.assertEqual(self.config["matched_records"], 1)
        self.assertEqual(self.config["unmatched_episodes"], 1)
        self.assertEqual(self.service.describe(self.ids["2"])["state"], "unavailable")
        with connect(self.catalog) as db:
            self.service.attach(db)
            result = overview(db, {"media": "yes"}, "id IN (SELECT record_id FROM gt_videos.records)")
        self.assertEqual(result["stats"]["records"], 1)
        self.assertEqual(result["stats"]["media"], 1)
        with self.assertRaises(ValueError):
            GTVideoService(self.index, {**self.meta, "generated_at": "different"})

    def test_native_eight_fps_cache_rejects_old_format_and_wrong_identity(self):
        identifier = self.ids["1"]
        self.complete(identifier)
        self.assertEqual(self.service.describe(identifier)["state"], "ready")
        for field, value in (("fps", 5), ("size", [610, 366]), ("instruction", "Other task")):
            report = self.report(identifier)
            report[field] = value
            self.service.paths(identifier)["report"].write_text(json.dumps(report))
            self.assertFalse(self.service.ready(identifier))
        self.assertIsNone(self.service.media_path("../../secret", "video"))

    def test_duplicate_requests_render_once_and_publish_complete_triplet(self):
        identifier = self.ids["1"]
        entered, release = threading.Event(), threading.Event()

        def run(command, **kwargs):
            self.assertIn("--rgb-only", command)
            self.assertEqual(command[command.index("--fps")+1], "8")
            entered.set()
            self.assertTrue(release.wait(5))
            output = Path(command[command.index("--output")+1])
            output.write_bytes(b"0123456789")
            output.with_suffix(".png").write_bytes(b"poster")
            output.with_suffix(".json").write_text(json.dumps(self.report(identifier)))
            return type("Result", (), {"returncode": 0})()

        with patch("catalog.vln_gt_video.subprocess.run", side_effect=run) as mocked:
            self.service.request(identifier)
            self.assertTrue(entered.wait(5))
            for _ in range(5):
                self.assertEqual(self.service.request(identifier)["state"], "rendering")
            self.assertIsNone(self.service.media_path(identifier, "video"))
            release.set()
            self.service.jobs[identifier].result(timeout=5)
            self.assertEqual(self.service.request(identifier)["state"], "ready")
            self.assertEqual(mocked.call_count, 1)
        report = json.loads(self.service.paths(identifier)["report"].read_text())
        self.assertEqual(report["catalog_record_id"], identifier)
        self.assertNotIn(".render-", report["video"])

    def test_http_video_range_status_and_unavailable_episode(self):
        self.complete(self.ids["1"])
        handler = type("GTHandler", (VLNRequestHandler,), {"database": self.catalog, "gt_videos": self.service,
                                                           "log_message": lambda *args: None})
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(self.root)))
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{server.server_port}/vln-api/"
        with urlopen(origin + "episode?id=" + self.ids["1"]) as response:
            self.assertEqual(json.load(response)["gt_video"]["fps"], 8)
        with urlopen(Request(origin + "gt-video?id=" + self.ids["2"], data=b"")) as response:
            self.assertEqual(json.load(response)["state"], "unavailable")
        url = origin + "gt-video-media?id=" + self.ids["1"] + "&kind=video"
        with urlopen(Request(url, headers={"Range": "bytes=2-5"})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
        with self.assertRaises(HTTPError) as error:
            urlopen(origin + "gt-video-media?id=../../secret&kind=video")
        self.assertEqual(error.exception.code, 404)

    def test_collection_binds_rxr_by_role_language_and_instruction_identity(self):
        rxr = self.root / "rxr"
        (rxr / "meta").mkdir(parents=True)
        (rxr / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
        item = json.loads((self.dataset / "meta/beta_nav.json").read_text())["episodes"][0]
        item["source_metadata"] = {"dataset": "rxr", "split": "train", "role": "guide", "language": "hi-IN", "instruction_id": "77"}
        (rxr / "meta/beta_nav.json").write_text(json.dumps({"episodes": [item]}))
        with sqlite3.connect(self.catalog) as db:
            db.row_factory = sqlite3.Row
            original = dict(db.execute("SELECT * FROM episodes WHERE id=?", (self.ids["1"],)).fetchone())
            for role in ("guide", "follower"):
                row = {**original, "id": "rxr-"+role, "source_id": "test-rxr", "dataset": "RxR", "role": role, "language": "hi-IN", "instruction_id": "77"}
                db.execute(f"INSERT INTO episodes ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
        index = self.root / "all.sqlite"
        sources = [{"root": str(root), "source_id": source, "dataset": dataset, "split": "train", "role": "guide"}
                   for root, source, dataset in [(self.dataset, "test-r2r", "R2R"), (rxr, "test-rxr", "RxR")]]
        config = build_collection(self.catalog, sources, index, self.root / "all-videos", self.root / "render.py", "/usr/bin/python3")
        service = GTVideoService(index, self.meta)
        self.addCleanup(service.pool.shutdown, wait=True)
        self.assertEqual(config["matched_records"], 2)
        self.assertIn("rxr-guide", service.records)
        self.assertNotIn("rxr-follower", service.records)
        self.assertIn("/rxr/train/guide/", str(service.paths("rxr-guide")["video"]))
        service.mark("rxr-guide", "rendering")
        self.assertEqual(service.describe("rxr-guide")["state"], "rendering")
        self.assertEqual(service.stats()["rendering"], 1)
        service.mark("rxr-guide", "error", "test failure")
        self.assertEqual(service.stats()["failed"], 1)
        self.assertEqual(service.stats()["pending"], 1)

    def test_collection_rejects_unvalidated_shards(self):
        index = self.root / "invalid.sqlite"
        with self.assertRaisesRegex(ValueError, "unvalidated shard"):
            build_collection(self.catalog, [{"shards_root": str(self.root), "source_id": "test-r2r", "dataset": "R2R", "split": "train", "role": "guide"}], index,
                             self.root / "videos", self.root / "render.py", "/usr/bin/python3")
        self.assertFalse(index.exists())

    def test_collection_binds_scalevln_basis_scene_without_allowing_traversal(self):
        scene_id = "hm3d/train/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb"
        item = json.loads((self.dataset / "meta/beta_nav.json").read_text())["episodes"][0]
        item.update(scene_id=scene_id, source_metadata={"dataset": "scalevln", "split": "train"})
        metadata = self.dataset / "meta/beta_nav.json"
        metadata.write_text(json.dumps({"episodes": [item]}))
        with sqlite3.connect(self.catalog) as db:
            db.row_factory = sqlite3.Row
            row = dict(db.execute("SELECT * FROM episodes WHERE id=?", (self.ids["1"],)).fetchone())
            row.update(id="scalevln-1", source_id="test-scalevln", dataset="ScaleVLN", scene="CFVBbU9Rsyb.basis")
            db.execute(f"INSERT INTO episodes ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
        sources = [{"root": str(self.dataset), "source_id": "test-scalevln", "dataset": "ScaleVLN", "split": "train", "role": "guide"}]
        index = self.root / "scalevln.sqlite"
        config = build_collection(self.catalog, sources, index, self.root / "videos", self.root / "render.py", "/usr/bin/python3")
        service = GTVideoService(index, self.meta)
        self.addCleanup(service.pool.shutdown, wait=True)
        self.assertEqual(config["matched_records"], 1)
        self.assertEqual(service.records["scalevln-1"]["scene"], "CFVBbU9Rsyb.basis")
        self.assertIn("/scalevln/train/guide/CFVBbU9Rsyb.basis/", str(service.paths("scalevln-1")["video"]))
        for bad_scene in ("../../CFVBbU9Rsyb.basis.glb", "hm3d/train/00337-CFVBbU9Rsyb/../CFVBbU9Rsyb.basis.glb"):
            with self.subTest(bad_scene=bad_scene):
                metadata.write_text(json.dumps({"episodes": [{**item, "scene_id": bad_scene}]}))
                with self.assertRaises(ValueError):
                    build_collection(self.catalog, sources, self.root / "unsafe.sqlite", self.root / "videos", self.root / "render.py", "/usr/bin/python3")

    def test_shared_video_preserves_each_instruction_and_discloses_provenance(self):
        original = self.ids["1"]
        self.complete(original)
        self.service.records[original]["content_hash"] = "verified-content-hash"
        alias = "same-route-new-instruction"
        self.service.records[alias] = {**self.service.records[original], "episode_id": "99", "episode_index": 99,
                                       "instruction": "Go to the same destination."}
        paths = self.service.paths(alias)
        paths["video"].parent.mkdir(parents=True)
        self.service.reuse(original, alias, paths)
        self.assertTrue(self.service.ready(alias))
        report = json.loads(paths["report"].read_text())
        self.assertEqual(report["instruction"], "Go to the same destination.")
        self.assertEqual(report["input_provenance"]["episode_index"], 99)
        self.assertEqual(report["input_provenance"]["shared_video_from_record_id"], original)
        self.assertEqual(paths["video"].stat().st_ino, self.service.paths(original)["video"].stat().st_ino)


if __name__ == "__main__":
    unittest.main()
