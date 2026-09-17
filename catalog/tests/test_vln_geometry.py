"""GT identity, action alignment, atomic indexing and map API regression tests."""
import contextlib
import gzip
import io
import json
import sqlite3
import threading
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from catalog.build_vln_geometry import build_geometry
from catalog.vln_catalog import connect
from catalog.vln_ground_truth import geometry_for_record, normalize_gt, open_geometry, read_gt_entries
from catalog.vln_serve import VLNRequestHandler
from catalog.tests import test_vln_catalog as catalog_tests
from catalog.tests.test_vln_catalog import episode


class VLNGeometryTests(unittest.TestCase):
    setUp = catalog_tests.VLNCatalogTests.setUp
    write = catalog_tests.VLNCatalogTests.write
    build = catalog_tests.VLNCatalogTests.build

    def fixture(self, gt=None, rows=None):
        self.write(rows or [episode(), episode(2)])
        self.build()
        self.gt_path = self.annotation.with_name("val_unseen_gt.json.gz")
        self.gt = gt or {"locations": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "actions": [2, 1, 3, 1, 0], "forward_steps": 2}
        with gzip.open(self.gt_path, "wt") as handle:
            json.dump({"1": self.gt}, handle)
        self.geometry = self.root / "vln-geometry.sqlite"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = build_geometry(self.database, self.geometry)
        with connect(self.database) as db:
            self.records = [json.loads(row[0]) for row in db.execute("SELECT detail FROM episodes ORDER BY episode_id")]
        return summary

    def test_keyed_stream_reader_spans_chunks_and_rejects_malformed_json(self):
        path = self.root / "gt.json.gz"
        entries = {"1": {"text": "हिन्दी" * 30000}, "2": {"locations": [[0, 0, 0]]}}
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False)
        self.assertEqual(dict(read_gt_entries(path)), entries)
        bad = self.root / "gt.json"
        for raw in ['{"1": {}', '{"1":{} ,}', '{"1": {}} junk', '{"1": []}']:
            with self.subTest(raw=raw):
                bad.write_text(raw)
                with self.assertRaises(ValueError):
                    list(read_gt_entries(bad))

    def test_turns_and_stop_do_not_consume_gt_locations(self):
        self.fixture()
        normalized = normalize_gt(self.gt, [0, 0, 0])
        self.assertEqual(normalized["action_location_indices"], [0, 0, 1, 1, 2, 2])
        self.assertEqual(normalized["length"], 2)
        self.assertEqual(normalize_gt({**self.gt, "actions": [1, 0]}, [0, 0, 0])["action_location_indices"], [])
        self.assertEqual(normalize_gt({**self.gt, "actions": [99, 1, 1]}, [0, 0, 0])["action_location_indices"], [])

    def test_invalid_coordinates_and_mismatched_start_cannot_be_attached(self):
        for gt, reason in [({"locations": []}, "missing_locations"),
                           ({"locations": [[float("nan"), 0, 0]]}, "invalid_coordinates"),
                           ({"locations": [[3, 0, 0]]}, "start_mismatch"),
                           ({"locations": [[0, 0, 0]], "actions": [True]}, "invalid_actions")]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                normalize_gt(gt, [0, 0, 0])
        summary = self.fixture(gt={"locations": [[10, 0, 0]], "actions": [0]})
        self.assertEqual(summary["ground_truth_records"], 0)
        with open_geometry(self.geometry) as db:
            self.assertEqual(geometry_for_record(db, self.records[0])["ground_truth"]["reason"], "start_mismatch")

    def test_exact_gt_identity_missing_entries_and_native_coordinate_guard(self):
        summary = self.fixture()
        self.assertEqual(summary["ground_truth_records"], 1)
        self.assertEqual(summary["missing_records"], 1)
        with sqlite3.connect(self.geometry) as db:
            db.execute("INSERT INTO scene_maps VALUES (?,?)", ("scan1", json.dumps({"floors": [1]})))
        with open_geometry(self.geometry) as db:
            result = geometry_for_record(db, self.records[0])
            self.assertEqual(result["ground_truth"]["locations"], self.gt["locations"])
            self.assertEqual(result["ground_truth"]["source_key"], "1")
            self.assertIsNotNone(result["topdown"])
            self.assertIsNone(geometry_for_record(db, {**self.records[0], "projection_axes": [0, 1]})["topdown"])
            self.assertEqual(geometry_for_record(db, self.records[1])["ground_truth"]["reason"], "gt_episode_missing")
        self.assertEqual(geometry_for_record(None, {**self.records[0], "split": "test"})["ground_truth"], {"available": False, "reason": "test_unpublished"})

    def test_truncated_or_duplicate_gt_keeps_previous_published_geometry(self):
        self.fixture()
        before = self.geometry.read_bytes()
        for raw in ['{"1": {}', '{"1": {},"1": {}}']:
            with gzip.open(self.gt_path, "wt") as handle:
                handle.write(raw)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                build_geometry(self.database, self.geometry)
            self.assertEqual(self.geometry.read_bytes(), before)
            self.assertFalse(list(self.root.glob(".vln-geometry-*.sqlite")))
        with self.assertRaises(ValueError):
            build_geometry(self.database, self.database)

    def test_http_gt_download_map_asset_and_stale_catalog(self):
        self.fixture()
        image = self.root / "floor.png"
        image.write_bytes(b"indexed-png-test")
        with sqlite3.connect(self.geometry) as db:
            db.execute("INSERT INTO assets VALUES (?,?)", ("floor1", str(image)))
            db.execute("INSERT INTO scene_maps VALUES (?,?)", ("scan1", json.dumps({"floors": [{"image": "floor1"}]})))
        handler = type("GTHandler", (VLNRequestHandler,), {"database": self.database, "log_message": lambda *args: None})
        web = Path(__file__).parents[1] / "vln_web"
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(web)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        origin = f"http://127.0.0.1:{server.server_port}/vln-api/"

        def get(path):
            return urllib.request.urlopen(origin + path)

        with get("episodes") as response:
            flags = {row["episode_id"]: row["has_gt"] for row in json.load(response)["items"]}
        self.assertEqual(flags, {"1": True, "2": False})
        with get("ground-truth?id=" + self.records[0]["id"]) as response:
            self.assertIn("attachment;", response.headers["Content-Disposition"])
            result = json.load(response)
            self.assertEqual(result["ground_truth"]["locations"], self.gt["locations"])
            self.assertEqual(result["ground_truth"]["actions"], self.gt["actions"])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            get("ground-truth?id=" + self.records[1]["id"])
        self.assertEqual(raised.exception.code, 404)
        with get("episode?id=" + self.records[0]["id"]) as response:
            self.assertEqual(json.load(response)["topdown"]["floors"][0]["image"], "floor1")
        with get("media?id=floor1") as response:
            self.assertEqual(response.read(), image.read_bytes())
        with sqlite3.connect(self.geometry) as db:
            info = json.loads(db.execute("SELECT value FROM metadata WHERE key='geometry'").fetchone()[0])
            info["catalog_generated_at"] = "old catalogue snapshot"
            db.execute("UPDATE metadata SET value=? WHERE key='geometry'", (json.dumps(info),))
        with get("episode?id=" + self.records[0]["id"]) as response:
            self.assertFalse(json.load(response)["ground_truth"]["available"])


if __name__ == "__main__":
    unittest.main()
