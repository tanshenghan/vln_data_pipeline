#!/usr/bin/env python3
"""Index dataset GT and build CPU NavMesh maps alongside a VLN catalogue."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .vln_catalog import connect, json_text, stable_id
    from .vln_ground_truth import normalize_gt, read_gt_entries
except ImportError:
    from vln_catalog import connect, json_text, stable_id
    from vln_ground_truth import normalize_gt, read_gt_entries


def gt_path_for(annotation, source, split, role):
    if source.get("gt_pattern"):
        # Annotation roots in catalogue metadata may originally be relative.
        # Resolve a configured pattern relative to the source root supplied by
        # the caller; default sibling discovery works after source relocation.
        root = Path(source["root"])
        if not root.is_absolute():
            raise ValueError("An explicit gt_pattern requires an absolute source root")
        path = (root / source["gt_pattern"].format(split=split, role=role)).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("GT path escapes source root")
        return path
    name = annotation.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            for extension in (".json.gz", ".json"):
                candidate = annotation.with_name(stem + "_gt" + extension)
                if candidate.is_file():
                    return candidate
            return annotation.with_name(stem + "_gt.json.gz")
    return None


def build_geometry(catalog_path, output, scene_root=None, habitat_python=None, workers=4, meters_per_pixel=0.04):
    output = output.resolve()
    if output == catalog_path.resolve():
        raise ValueError("Geometry output must differ from the source catalogue")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vln-geometry-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    db = sqlite3.connect(temporary)
    catalog = connect(catalog_path)
    reports = []
    try:
        db.executescript("""
            CREATE TABLE ground_truth(record_id TEXT PRIMARY KEY,scene TEXT,point_count INTEGER,action_count INTEGER,payload TEXT);
            CREATE TABLE gt_issues(record_id TEXT PRIMARY KEY,reason TEXT);
            CREATE TABLE scene_maps(scene TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE assets(id TEXT PRIMARY KEY,path TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        """)
        metadata = json.loads(catalog.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        sources = {source["id"]: source for source in metadata["sources"]}
        for spec in metadata["inputs"]:
            source_id, split, role = spec["source_id"], spec["split"], spec["role"]
            gt_path = gt_path_for(Path(spec["path"]), sources[source_id], split, role)
            expected = defaultdict(list)
            for row in catalog.execute("SELECT id,episode_id,scene,json_extract(detail,'$.start_position') start FROM episodes WHERE source_id=? AND split=? AND role=?",
                                       (source_id, split, role)):
                expected[row["episode_id"]].append(dict(row))
            report = {"source_id": source_id, "split": split, "role": role, "path": str(gt_path) if gt_path else None,
                      "expected": sum(len(rows) for rows in expected.values()), "matched": 0, "rejected": 0, "unmatched_gt": 0}
            if gt_path is None or not gt_path.is_file():
                reason = "test_unpublished" if split == "test" else "gt_file_missing"
                for rows in expected.values():
                    db.executemany("INSERT INTO gt_issues VALUES (?,?)", [(row["id"], reason) for row in rows])
                report["status"] = reason
                reports.append(report)
                print(f"GT {source_id} {split} {role}: {reason}", flush=True)
                continue
            print(f"GT indexing {source_id} {split} {role}: {gt_path}", flush=True)
            seen = set()
            reasons = Counter()
            for key, entry in read_gt_entries(gt_path):
                if key in seen:
                    raise ValueError(f"Duplicate GT episode key {key} in {gt_path}")
                seen.add(key)
                rows = expected.get(key, [])
                if not rows:
                    report["unmatched_gt"] += 1
                for row in rows:
                    try:
                        payload = normalize_gt(entry, json.loads(row["start"]) if row["start"] else None)
                    except ValueError as error:
                        reason = str(error)
                        db.execute("INSERT INTO gt_issues VALUES (?,?)", (row["id"], reason))
                        report["rejected"] += 1
                        reasons[reason] += 1
                        continue
                    payload.update({"source_path": str(gt_path), "source_key": key, "kind": "dataset_evaluation_gt"})
                    db.execute("INSERT INTO ground_truth VALUES (?,?,?,?,?)",
                               (row["id"], row["scene"], payload["point_count"], payload["action_count"], json_text(payload)))
                    report["matched"] += 1
            for key in expected.keys() - seen:
                db.executemany("INSERT INTO gt_issues VALUES (?,?)", [(row["id"], "gt_episode_missing") for row in expected[key]])
            report.update({"status": "indexed", "rejection_reasons": dict(reasons)})
            reports.append(report)
            db.commit()
            print(f"GT matched {report['matched']}/{report['expected']}, rejected {report['rejected']}", flush=True)
        map_results = []
        if scene_root and habitat_python:
            scenes = [row[0] for row in catalog.execute("SELECT DISTINCT scene FROM episodes WHERE format='vlnce' ORDER BY scene")]
            with tempfile.TemporaryDirectory(prefix="vln-map-build-", dir=output.parent) as job_dir:
                job_dir = Path(job_dir)
                scene_list, manifest = job_dir / "scenes.json", job_dir / "maps.json"
                scene_list.write_text(json_text(scenes))
                command = [str(habitat_python), str(Path(__file__).with_name("vln_topdown.py")),
                           "--scenes", str(scene_list), "--scene-root", str(scene_root),
                           "--output-dir", str(output.parent / "topdown"), "--manifest", str(manifest),
                           "--workers", str(workers), "--meters-per-pixel", str(meters_per_pixel)]
                subprocess.run(command, check=True, env={**os.environ, "HABITAT_SIM_LOG": "quiet", "MAGNUM_LOG": "quiet"})
                map_results = json.loads(manifest.read_text())
            for result in map_results:
                if result.get("error") or not result.get("floors"):
                    continue
                for floor in result["floors"]:
                    path = floor.pop("image_path")
                    asset_id = stable_id(path)
                    db.execute("INSERT OR IGNORE INTO assets VALUES (?,?)", (asset_id, path))
                    floor["image"] = asset_id
                db.execute("INSERT INTO scene_maps VALUES (?,?)", (result["scene"], json_text(result)))
        summary = {"schema_version": 1, "catalog_id": metadata["catalog_id"], "catalog_generated_at": metadata["generated_at"],
                   "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": reports,
                   "ground_truth_records": db.execute("SELECT count(*) FROM ground_truth").fetchone()[0],
                   "missing_records": db.execute("SELECT count(*) FROM gt_issues").fetchone()[0],
                   "mapped_scenes": db.execute("SELECT count(*) FROM scene_maps").fetchone()[0],
                   "map_errors": [row for row in map_results if row.get("error")]}
        db.execute("INSERT INTO metadata VALUES ('geometry',?)", (json_text(summary),))
        db.commit()
        db.close()
        catalog.close()
        os.replace(temporary, output)
        return summary
    except BaseException:
        db.close()
        catalog.close()
        Path(temporary).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, help="MP3D root containing scan/scan.navmesh")
    parser.add_argument("--habitat-python", type=Path, help="Python with habitat_sim, numpy and Pillow installed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--meters-per-pixel", type=float, default=0.04)
    args = parser.parse_args()
    if bool(args.scene_root) != bool(args.habitat_python):
        parser.error("scene-root and habitat-python must be provided together")
    summary = build_geometry(args.catalog, args.output, args.scene_root, args.habitat_python, args.workers, args.meters_per_pixel)
    print(json_text(summary))


if __name__ == "__main__":
    main()
