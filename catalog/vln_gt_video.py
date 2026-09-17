"""Exact LeRobot/catalog bindings and a bounded, lazy GT-video cache."""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def fingerprint(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def build_index(catalog, dataset, output, video_root, renderer, python, source_id):
    import ijson

    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    info = json.loads((dataset / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0" or info.get("video_path"):
        raise ValueError("expected image-backed LeRobot v3")
    with readonly(catalog) as db:
        catalog_meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        candidates = {}
        for row in db.execute("SELECT id,episode_id,scene,instruction FROM episodes WHERE source_id=? AND dataset='R2R' AND split='train'", (source_id,)):
            key = (row["episode_id"], row["scene"], row["instruction"].strip())
            candidates.setdefault(key, []).append(row["id"])
    config = {"catalog_id": catalog_meta["catalog_id"], "catalog_generated_at": catalog_meta["generated_at"],
              "dataset_root": str(dataset), "output_root": str(Path(video_root).resolve()),
              "renderer": str(Path(renderer).resolve()), "python": str(Path(python).absolute()),
              "source_id": source_id, "fps": 8, "size": [470, 280], "horizon": 10,
              "source_manifest": fingerprint(dataset / "meta/beta_nav.json"),
              "info_sha256": hashlib.sha256((dataset / "meta/info.json").read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".sqlite", delete=False) as handle:
        temporary = Path(handle.name)
    count, unmatched = 0, 0
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY,episode_id TEXT,scene TEXT,instruction TEXT,frames INTEGER,episode_index INTEGER)")
            with (dataset / "meta/beta_nav.json").open("rb") as handle:
                for item in ijson.items(handle, "episodes.item", use_float=True):
                    episode_id, scene = str(item["episode_id"]), Path(item["scene_id"]).stem
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene) or not episode_id.isdecimal():
                        raise ValueError("unsafe scene/episode identifier")
                    ids = candidates.get((episode_id, scene, item["instruction"].strip()), [])
                    if len(ids) != 1:
                        unmatched += 1
                        continue
                    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (ids[0], episode_id, scene, item["instruction"], item["frames"], item["episode_index"]))
                    count += 1
            config.update(matched_records=count, unmatched_episodes=unmatched)
            if not count:
                raise ValueError("no LeRobot episodes match this catalog source/split/scene/instruction")
            db.execute("INSERT INTO metadata VALUES ('videos',?)", (json.dumps(config),))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return config


class GTVideoService:
    def __init__(self, index, catalog_metadata, workers=1):
        self.index = Path(index).resolve()
        with readonly(self.index) as db:
            self.config = json.loads(db.execute("SELECT value FROM metadata WHERE key='videos'").fetchone()[0])
            self.records = {row["record_id"]: dict(row) for row in db.execute("SELECT * FROM records")}
        for key, value in (("catalog_id", catalog_metadata["catalog_id"]), ("catalog_generated_at", catalog_metadata["generated_at"])):
            if self.config[key] != value:
                raise ValueError("GT video index belongs to a different catalog; rebuild it")
        self.collection = self.config.get("schema_version") == 2
        for source in self.config["datasets"].values() if self.collection else [self.config]:
            root = Path(source["dataset_root"])
            if fingerprint(root / "meta/beta_nav.json") != source["source_manifest"]:
                raise ValueError("LeRobot provenance changed; rebuild video index into a new output version")
            if hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest() != source["info_sha256"]:
                raise ValueError("LeRobot info.json changed")
        self.root = Path(self.config["output_root"])
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vln-gt-video")
        self.jobs = {}
        self.progress = self.index.with_suffix(".progress.sqlite") if self.collection else None
        if self.progress:
            with sqlite3.connect(self.progress, timeout=30) as db:
                db.execute("CREATE TABLE IF NOT EXISTS jobs(record_id TEXT PRIMARY KEY,state TEXT,updated REAL,error TEXT)")

    def source(self, record_id):
        return self.config["datasets"][self.records[record_id]["source_key"]] if self.collection else self.config

    def mark(self, record_id, state, error=None):
        if self.progress:
            with sqlite3.connect(self.progress, timeout=30) as db:
                db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?)", (record_id, state, time.time(), error))

    def stats(self):
        counts = {}
        by_group = {}
        if self.progress:
            with readonly(self.progress) as db:
                counts = dict(db.execute("SELECT state,count(*) FROM jobs GROUP BY state").fetchall())
                for identifier, state in db.execute("SELECT record_id,state FROM jobs"):
                    row = self.records.get(identifier)
                    if row:
                        key = (row["dataset"], row["split"], row["role"])
                        group = by_group.setdefault(key, {})
                        group[state] = group.get(state, 0) + 1
        coverage = []
        for row in self.config.get("coverage", []):
            group = by_group.get((row["dataset"], row["split"], row["role"]), {})
            coverage.append({**row, "ready": group.get("ready", 0), "failed": group.get("error", 0)})
        return {"records": len(self.records), "fps": 8, "ready": counts.get("ready", 0), "rendering": counts.get("rendering", 0),
                "failed": counts.get("error", 0), "pending": len(self.records)-sum(counts.values()),
                "coverage": coverage}

    def attach(self, db):
        db.execute("ATTACH DATABASE ? AS gt_videos", (self.index.as_uri() + "?mode=ro",))

    def paths(self, record_id):
        row = self.records[record_id]
        parent = self.root / row["dataset"].lower() / row["split"] / row["role"] if self.collection else self.root / "train"
        base = parent / row["scene"] / f'episode_{row["episode_id"]}' / "gt10_apos_opos"
        return {kind: base.with_suffix(ext) for kind, ext in (("video", ".mp4"), ("poster", ".png"), ("report", ".json"))}

    def ready(self, record_id):
        paths, row = self.paths(record_id), self.records[record_id]
        if not all(path.is_file() and path.resolve() == path for path in paths.values()):
            return False
        try:
            report = json.loads(paths["report"].read_text())
            provenance = report["input_provenance"]
            return (report["fps"] == 8 and report["size"] == [470, 280] and report["horizon"] == 10
                    and report["layout"] == "rgb_only" and report["frames"] == row["frames"]
                    and report["instruction"].strip() == row["instruction"].strip()
                    and provenance["episode_index"] == row["episode_index"]
                    and provenance["dataset_root"] == self.source(record_id)["dataset_root"]
                    and provenance["info_sha256"] == self.source(record_id)["info_sha256"])
        except (ValueError, KeyError, OSError):
            return False

    def describe(self, record_id):
        if record_id not in self.records:
            return {"state": "unavailable", "reason": "此样本尚缺对应的 LeRobot RGB 观测；原始指令和已公开 GT 仍可查看。"}
        row = self.records[record_id]
        state = "ready" if self.ready(record_id) else "pending"
        if state != "ready" and self.progress:
            with readonly(self.progress) as db:
                job = db.execute("SELECT state,updated FROM jobs WHERE record_id=?", (record_id,)).fetchone()
            if job and (job["state"] == "error" or (job["state"] == "rendering" and time.time()-job["updated"] < 1800)):
                state = job["state"]
        with self.lock:
            future = self.jobs.get(record_id)
            if state != "ready" and future:
                state = "rendering" if future.running() else "queued"
                if future.done():
                    state = "error"
        return {"state": state, "fps": 8, "size": [470, 280], "frames": row["frames"],
                "duration_seconds": row["frames"] / 8, "horizon": 10}

    def request(self, record_id, retry=False):
        status = self.describe(record_id)
        if status["state"] in {"unavailable", "ready"}:
            return status
        if status["state"] == "rendering":
            return status
        with self.lock:
            future = self.jobs.get(record_id)
            if future is None or (retry and future.done()):
                if sum(not job.done() for job in self.jobs.values()) >= 16:
                    return {**status, "state": "busy"}
                self.jobs[record_id] = self.pool.submit(self.render, record_id)
        return self.describe(record_id)

    def render(self, record_id):
        row, paths = self.records[record_id], self.paths(record_id)
        directory = paths["video"].parent
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".render.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.ready(record_id):
                self.mark(record_id, "ready")
                return
            self.mark(record_id, "rendering")
            try:
                if self.collection:
                    shared = self.root / ".shared"
                    shared.mkdir(parents=True, exist_ok=True)
                    key = row["content_hash"]
                    with (shared / f"{key}.lock").open("a") as shared_lock:
                        fcntl.flock(shared_lock, fcntl.LOCK_EX)
                        reference = shared / f"{key}.json"
                        owner = json.loads(reference.read_text()).get("record_id") if reference.is_file() else None
                        if owner in self.records and self.records[owner]["content_hash"] == key and self.ready(owner):
                            self.reuse(owner, record_id, paths)
                        else:
                            self._render(record_id, paths, row, directory)
                            temporary = reference.with_suffix(".tmp")
                            temporary.write_text(json.dumps({"record_id": record_id}))
                            temporary.replace(reference)
                else:
                    self._render(record_id, paths, row, directory)
                self.mark(record_id, "ready")
            except Exception as error:
                self.mark(record_id, "error", str(error)[-1500:])
                raise

    def reuse(self, owner, record_id, paths):
        """Only byte-identical observations/labels share a clip; identities stay separate."""
        original, row = self.paths(owner), self.records[record_id]
        source = self.source(record_id)
        report = json.loads(original["report"].read_text())
        rendered_provenance = report["input_provenance"]
        report.update(video=str(paths["video"]), preview=str(paths["poster"]), episode_id=row["episode_id"],
                      instruction=row["instruction"], catalog_record_id=record_id,
                      source_episode=source["dataset_root"], annotations=source["dataset_root"],
                      input_provenance={"format": "LeRobotDataset v3.0", "dataset_root": source["dataset_root"],
                                        "episode_index": row["episode_index"], "source_episode_id": row["episode_id"],
                                        "scene_id": row["scene"], "info_sha256": source["info_sha256"],
                                        "identical_observations_and_labels_sha256": row["content_hash"],
                                        "shared_video_from_record_id": owner,
                                        "rendered_source_provenance": rendered_provenance})
        with tempfile.TemporaryDirectory(prefix=".reuse-", dir=paths["video"].parent) as temporary:
            for kind in ("video", "poster"):
                target = Path(temporary) / paths[kind].name
                try:
                    os.link(original[kind], target)
                except OSError:
                    shutil.copyfile(original[kind], target)
                target.replace(paths[kind])
            target = Path(temporary) / paths["report"].name
            target.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.replace(paths["report"])
        if not self.ready(record_id):
            raise ValueError("shared video failed identity validation")

    def _render(self, record_id, paths, row, directory):
        # Publish a matching triplet only after a complete render. JSON is the commit marker.
        try:
            with tempfile.TemporaryDirectory(prefix=".render-", dir=directory) as temporary:
                output = Path(temporary) / paths["video"].name
                command = [self.config["python"], self.config["renderer"], self.source(record_id)["dataset_root"],
                           "--episode-id", row["episode_id"], "--scene", row["scene"], "--output", str(output),
                           "--fps", "8", "--horizon", "10", "--scale", "1", "--rgb-only",
                           "--hide-occluded", "--number-steps", "--preview-frame", str(min(24, row["frames"]-1))]
                if self.collection:
                    command = [self.config["python"], str(Path(__file__).with_name("vln_video_collection.py")),
                               "--index", str(self.index), "--record-id", record_id, "--output", str(output)]
                env = {**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/vln_gt_video_pycache", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800, check=False)
                if result.returncode:
                    raise RuntimeError(result.stderr[-6000:])
                report_path = output.with_suffix(".json")
                report = json.loads(report_path.read_text())
                report.update(video=str(paths["video"]), preview=str(paths["poster"]), catalog_record_id=record_id)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                for kind in ("video", "poster", "report"):
                    output.with_suffix(paths[kind].suffix).replace(paths[kind])
            if not self.ready(record_id):
                raise ValueError("rendered video failed GT media validation")
        except Exception:
            logger.exception("GT video render failed for %s", record_id)
            raise

    def media_path(self, record_id, kind):
        if record_id not in self.records or kind not in {"video", "poster", "report"} or not self.ready(record_id):
            return None
        return self.paths(record_id)[kind]
