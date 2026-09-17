"""Index all validated R2R/RxR observations and render cached metadata jobs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sqlite3
import tempfile
import zlib
from pathlib import Path

try:
    from .vln_gt_video import fingerprint, readonly
except ImportError:
    from vln_gt_video import fingerprint, readonly


def build_collection(catalog, sources, output, video_root, renderer, python):
    import ijson

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with readonly(catalog) as db:
        meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        candidates = {}
        totals = [dict(row) for row in db.execute("SELECT dataset,split,role,count(*) total FROM episodes GROUP BY dataset,split,role")]
        for row in db.execute("SELECT id,source_id,dataset,split,role,language,episode_id,instruction_id,scene,instruction FROM episodes"):
            key = tuple(row[k] for k in ("source_id", "dataset", "split", "role", "episode_id", "scene")) + (row["instruction"].strip(),)
            candidates.setdefault(key, []).append(dict(row))
    config = {"schema_version": 2, "catalog_id": meta["catalog_id"], "catalog_generated_at": meta["generated_at"],
              "output_root": str(Path(video_root).resolve()), "renderer": str(Path(renderer).resolve()),
              "python": str(Path(python).absolute()), "fps": 8, "size": [470, 280], "horizon": 10,
              "datasets": {}, "coverage": totals}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".sqlite", delete=False) as handle:
        temporary = Path(handle.name)
    unmatched = 0
    try:
        with sqlite3.connect(temporary) as db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE records (record_id TEXT PRIMARY KEY,episode_id TEXT,scene TEXT,instruction TEXT,
                    frames INTEGER,episode_index INTEGER,source_key TEXT,dataset TEXT,split TEXT,role TEXT,
                    language TEXT,instruction_id TEXT,content_hash TEXT);
                CREATE TABLE episode_metadata (record_id TEXT PRIMARY KEY,payload BLOB NOT NULL);
            """)
            for source in sources:
                roots = sorted(Path(source["shards_root"]).resolve().glob("*/meta/info.json")) if source.get("shards_root") else [Path(source["root"]).resolve() / "meta/info.json"]
                for info_path in roots:
                    root = info_path.parent.parent
                    if source.get("shards_root"):
                        validation = root / "meta/validation.json"
                        if not validation.is_file() or json.loads(validation.read_text()).get("complete") is not True:
                            raise ValueError(f"unvalidated shard: {root}")
                    info = json.loads(info_path.read_text())
                    if info.get("codebase_version") != "v3.0" or info.get("video_path"):
                        raise ValueError(f"expected image-backed LeRobot v3: {root}")
                    source_key = hashlib.sha256(str(root).encode()).hexdigest()[:24]
                    config["datasets"][source_key] = {"dataset_root": str(root), "source_manifest": fingerprint(root / "meta/beta_nav.json"),
                                                     "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest()}
                    with (root / "meta/beta_nav.json").open("rb") as handle:
                        for item in ijson.items(handle, "episodes.item", use_float=True):
                            scene, episode_id = Path(item["scene_id"]).stem, str(item["episode_id"])
                            if not re.fullmatch(r"[A-Za-z0-9_-]+", scene) or not episode_id.isdecimal():
                                raise ValueError("unsafe source scene/episode identifier")
                            provenance = item.get("source_metadata", {})
                            for field in ("dataset", "split", "role"):
                                if field in provenance and provenance[field].lower() != source[field].lower():
                                    raise ValueError(f"source {field} mismatch: {root} / {episode_id}")
                            key = tuple(source[k] for k in ("source_id", "dataset", "split", "role")) + (episode_id, scene, item["instruction"].strip())
                            matches = candidates.get(key, [])
                            matches = [r for r in matches if ("instruction_id" not in provenance or r["instruction_id"] == str(provenance["instruction_id"])) and ("language" not in provenance or r["language"] == provenance["language"])]
                            if len(matches) != 1:
                                unmatched += 1
                                continue
                            record = matches[0]
                            physical = {k: item.get(k) for k in ("scene_id", "frames", "camera", "pointing_config", "observation_hashes", "array_hashes")}
                            physical["samples_sha256"] = item.get("source_hashes", {}).get("samples.jsonl")
                            required = {"observation.world_from_body", "observation.world_from_camera_cv", "action.fm.waypoints_ego",
                                        "action.apos.pixel", "action.apos.valid_mask", "action.opos.pixel", "action.opos.valid_mask"}
                            if (len(item.get("observation_hashes", [])) != item["frames"]
                                    or not required.issubset(item.get("array_hashes", {})) or not physical["samples_sha256"]):
                                physical["no_reuse_without_full_hash_proof"] = record["id"]
                            content_hash = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                            values = (record["id"], episode_id, scene, item["instruction"], item["frames"], item["episode_index"], source_key,
                                      source["dataset"], source["split"], source["role"], record["language"], record["instruction_id"], content_hash)
                            db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
                            payload = zlib.compress(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode(), 1)
                            db.execute("INSERT INTO episode_metadata VALUES (?,?)", (record["id"], payload))
                    if len(config["datasets"]) % 100 == 0:
                        print(json.dumps({"indexed_datasets": len(config["datasets"]), "records": db.execute("SELECT count(*) FROM records").fetchone()[0]}), flush=True)
            for row in config["coverage"]:
                row["observations"] = db.execute("SELECT count(*) FROM records WHERE dataset=? AND split=? AND role=?", (row["dataset"], row["split"], row["role"])).fetchone()[0]
                row["missing_observations"] = row["total"] - row["observations"]
            config["matched_records"] = db.execute("SELECT count(*) FROM records").fetchone()[0]
            config["unique_clips"] = db.execute("SELECT count(DISTINCT content_hash) FROM records").fetchone()[0]
            config["unmatched_episodes"] = unmatched
            if not config["matched_records"]:
                raise ValueError("no matching observation records")
            db.execute("CREATE INDEX records_coverage ON records(dataset,split,role)")
            db.execute("CREATE INDEX records_content ON records(content_hash)")
            db.execute("INSERT INTO metadata VALUES ('videos',?)", (json.dumps(config),))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {k: v for k, v in config.items() if k != "datasets"}


def render_cached(index, record_id, output):
    with readonly(index) as db:
        config = json.loads(db.execute("SELECT value FROM metadata WHERE key='videos'").fetchone()[0])
        row = dict(db.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone())
        metadata = json.loads(zlib.decompress(db.execute("SELECT payload FROM episode_metadata WHERE record_id=?", (record_id,)).fetchone()[0]))
    source = config["datasets"][row["source_key"]]
    root = Path(source["dataset_root"])
    if fingerprint(root / "meta/beta_nav.json") != source["source_manifest"]:
        raise ValueError("source metadata changed since indexing")
    spec = importlib.util.spec_from_file_location("vln_gt_renderer", config["renderer"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episode = module.load_lerobot_episode(root, row["episode_id"], scene=row["scene"], metadata=metadata)
    result = module.render_video(episode, output, fps=8, horizon=10, scale=1, rgb_only=True,
                                 hide_occluded=True, number_steps=True, preview_frame=min(24, row["frames"]-1))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_cached(args.index, args.record_id, args.output)


if __name__ == "__main__":
    main()
