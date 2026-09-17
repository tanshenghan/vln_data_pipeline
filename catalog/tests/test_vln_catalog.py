"""Format, identity, publication and HTTP tests for the VLN browser."""
import contextlib
import gzip
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import urllib.parse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from data_web.vln_catalog import build_database, connect, normalize, overview, read_rows, where_clause
from data_web.vln_serve import VLNRequestHandler


def episode(identifier=1, **updates):
    return {
        "episode_id": identifier, "trajectory_id": 9,
        "scene_id": "mp3d/scan1/scan1.glb", "start_position": [0, 0, 0],
        "reference_path": [[0, 0, 0], [3, 0, 4]],
        "goals": [{"position": [3, 0, 4]}],
        "instruction": {"instruction_text": "Walk past the couch and turn left."},
        **updates,
    }


class VLNCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation = self.root / "data" / "val_unseen" / "val_unseen.json.gz"
        self.annotation.parent.mkdir(parents=True)
        self.source = {"id": "r2r-test-v1", "dataset": "R2R", "format": "vlnce", "version": "test-v1",
                       "root": "data", "splits": ["val_unseen"]}
        self.config = self.root / "sources.json"
        self.database = self.root / "catalog.sqlite"

    def write(self, rows, source=None, media=None):
        with gzip.open(self.annotation, "wt", encoding="utf-8") as handle:
            json.dump({"episodes": rows, "instruction_vocab": {"word_list": []}}, handle)
        self.config.write_text(json.dumps({"sources": [source or self.source], "media_catalogs": media or []}))

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_database(self.config, self.database)

    def test_streams_container_list_gzip_and_multilingual_jsonl(self):
        row = {"instruction": "हिन्दी తెలుగు " + "a" * 100000}
        for name, value in [("array.json", json.dumps([row])), ("wrapped.json", json.dumps({"vocab": {}, "episodes": [row], "tail": []}))]:
            path = self.root / name
            path.write_text(value)
            self.assertEqual(list(read_rows(path)), [row])
        path = self.root / "rxr.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n\n")
        self.assertEqual(list(read_rows(path)), [row])

    def test_truncated_and_malformed_inputs_are_rejected(self):
        path = self.root / "bad.json"
        for value in ['{"episodes":[{"id":1}]', '[{"id":1},]', '[{"id":1}] garbage', '{"other": []}']:
            with self.subTest(value=value):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    list(read_rows(path))

    def test_vlnce_coordinates_language_and_missing_test_targets(self):
        record = next(normalize(episode(), self.source, "val_unseen", "guide", self.annotation, 0, {}))
        self.assertEqual(record["path_length"], 5)
        self.assertEqual(record["projection_axes"], [0, 2])
        self.assertEqual(record["language"], "en")
        unknown = next(normalize(episode(reference_path=[], goals=[]), self.source, "test", "guide", self.annotation, 0, {}))
        self.assertIsNone(unknown["path_length"])
        self.assertIsNone(unknown["geodesic_distance"])
        self.assertIn("missing_goal", unknown["issues"])

    def test_nonfinite_rxr_word_timestamps_are_null(self):
        row = episode(instruction={"instruction_text": "తెలుగు", "language": "te-IN", "instruction_id": "32",
                                   "timed_instruction": [{"word": "తెలుగు", "start_time": float("nan"), "end_time": float("inf")}]})
        self.write([row], {**self.source, "dataset": "RxR"})
        self.build()
        with connect(self.database) as db:
            record = json.loads(db.execute("SELECT detail FROM episodes").fetchone()[0])
        self.assertEqual(record["language"], "te-IN")
        self.assertEqual(record["timed_instruction"][0]["start_time"], None)

    def test_route_counts_differ_from_instruction_counts_and_ids_survive_reordering(self):
        rows = [episode(1), episode(2)]
        self.write(rows)
        self.build()
        with connect(self.database) as db:
            stats = overview(db, {})["stats"]
            original = {row["episode_id"]: row["id"] for row in db.execute("SELECT id,episode_id FROM episodes")}
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["routes"], 1)
        self.write(list(reversed(rows)))
        self.build()
        with connect(self.database) as db:
            rebuilt = {row["episode_id"]: row["id"] for row in db.execute("SELECT id,episode_id FROM episodes")}
        self.assertEqual(original, rebuilt)
        record = next(normalize(rows[0], self.source, "train", "guide", self.annotation, 0, {}))
        self.assertNotEqual(record["id"], original["1"])

    def test_missing_configured_file_preserves_published_database(self):
        self.write([episode()])
        self.build()
        before = self.database.read_bytes()
        self.annotation.unlink()
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(before, self.database.read_bytes())
        self.assertFalse(list(self.root.glob(".vln-*.sqlite")))

    def test_duplicate_identity_rejects_publication(self):
        self.write([episode()])
        self.build()
        before = self.database.read_bytes()
        self.write([episode(), episode()])
        with self.assertRaises(Exception):
            self.build()
        self.assertEqual(before, self.database.read_bytes())

    def test_native_r2r_expands_instructions_and_never_bridges_missing_viewpoints(self):
        row = {"path_id": 1, "scan": "s", "path": ["a", "b", "c"], "instructions": ["Go left", "Turn left"]}
        source = {**self.source, "format": "r2r"}
        records = list(normalize(row, source, "train", "guide", self.annotation, 0, {"a": [0, 0, 0], "c": [3, 4, 0]}))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["route_id"], records[1]["route_id"])
        self.assertNotEqual(records[0]["id"], records[1]["id"])
        self.assertEqual(records[0]["reference_path"], [])
        self.assertEqual(records[0]["viewpoints"], ["a", "b", "c"])
        complete = next(normalize(row, source, "train", "guide", self.annotation, 0, {"a": [0, 0, 0], "b": [0, 4, 0], "c": [3, 4, 0]}))
        self.assertEqual(complete["path_length"], 7)
        self.assertEqual(complete["projection_axes"], [0, 1])

    def test_native_rxr_preserves_language_and_role(self):
        source = {**self.source, "dataset": "RxR", "format": "rxr"}
        row = {"instruction_id": 53, "path_id": 12, "scan": "s", "instruction": "दरवाजे तक चलें", "language": "hi-IN"}
        guide = next(normalize(row, source, "train", "guide", self.annotation, 0, {}))
        follower = next(normalize(row, source, "train", "follower", self.annotation, 0, {}))
        self.assertEqual(guide["instruction_id"], "53")
        self.assertEqual(guide["language"], "hi-IN")
        self.assertNotEqual(guide["id"], follower["id"])

    def test_bound_filters_and_literal_wildcards(self):
        self.write([episode(1), episode(2, instruction={"instruction_text": "100% success_"})])
        self.build()
        with connect(self.database) as db:
            self.assertEqual(overview(db, {"q": "%"})["stats"]["records"], 1)
            self.assertEqual(overview(db, {"q": "' OR 1=1 --"})["stats"]["records"], 0)
            self.assertEqual(overview(db, {"min_length": "6"})["stats"]["records"], 0)
        for params in [{"min_length": "nan"}, {"max_length": "-2"}, {"min_length": "10", "max_length": "2"}]:
            with self.assertRaises(ValueError):
                where_clause(params)

    def media_fixture(self):
        root = self.root / "public"
        root.mkdir()
        (root / "frame.webp").write_bytes(b"0123456789")
        report = {"episodes": [{"meta": {"sceneId": "scan1", "episodeId": 1, "split": "val_unseen",
                                          "instruction": episode()["instruction"]["instruction_text"]},
                                "events": [{"step": 0, "overlay": "/frame.webp"}], "assets": {}}]}
        path = self.root / "reports.json"
        path.write_text(json.dumps(report))
        return [{"source_id": self.source["id"], "catalog": str(path), "root": str(root)}]

    def test_media_matches_exact_source_scene_split_episode_and_instruction(self):
        media = self.media_fixture()
        self.write([episode(), episode(2)], media=media)
        self.build()
        with connect(self.database) as db:
            self.assertEqual(overview(db, {})["stats"]["media"], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM assets").fetchone()[0], 1)
        self.write([episode(instruction={"instruction_text": "A different instruction"})], media=media)
        self.build()
        with connect(self.database) as db:
            self.assertEqual(overview(db, {})["stats"]["media"], 0)

    def test_http_pagination_export_media_range_and_invalid_requests(self):
        self.write([episode(), episode(2)], media=self.media_fixture())
        self.build()
        handler = type("TestHandler", (VLNRequestHandler,), {"database": self.database, "log_message": lambda *args: None})
        web = Path(__file__).parents[1] / "vln_web"
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(web)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        origin = f"http://127.0.0.1:{server.server_port}"

        def get(path, **headers):
            return urllib.request.urlopen(urllib.request.Request(origin + path, headers=headers))

        with get("/vln-api/episodes?limit=1&offset=1") as response:
            result = json.load(response)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["stats"]["records"], 2)
        with get("/vln-api/export?media=yes") as response:
            exported = [json.loads(line) for line in response]
        self.assertEqual(len(exported), 1)
        self.assertIn("source_path", exported[0])
        payload = {"filters": {"media": "yes"}, "reviews": {exported[0]["id"]: {"decision": "keep", "note": "verified"}}}
        request = urllib.request.Request(origin + "/vln-api/export", data=urllib.parse.urlencode({"payload": json.dumps(payload)}).encode())
        with urllib.request.urlopen(request) as response:
            reviewed = json.loads(response.readline())
        self.assertEqual(reviewed["manual_decision"], "keep")
        self.assertEqual(reviewed["manual_note"], "verified")
        self.assertNotIn("timed_instruction", reviewed)
        with get("/vln-api/episode?id=" + exported[0]["id"]) as response:
            record = json.load(response)
        asset = record["media"][0]["frames"][0]["path"]
        with get("/vln-api/media?id=" + asset, Range="bytes=2-5") as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
        with get("/") as response:
            self.assertIn(b"VLN Atlas", response.read())
        for path, code in [("/vln-api/episodes?limit=no", 400), ("/vln-api/episode?id=missing", 404),
                           ("/vln-api/media?id=../../sources.json", 404), ("/sources.json", 404)]:
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                get(path)
            self.assertEqual(error.exception.code, code)


if __name__ == "__main__":
    unittest.main()
