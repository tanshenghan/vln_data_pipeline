"""Native, read-only ObjectNav catalog; not a VLN expert-trajectory adapter.

Keep millions of native episodes in their original per-scene compressed files.
The small manifest indexes counts/goal categories; a page reads only one scene.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re

try:
    from .download_habitat_assets import PACKAGES, sha256, atomic_json
except ImportError:
    from download_habitat_assets import PACKAGES, sha256, atomic_json


DATASETS = {
    "mp3d_objectnav_v1": {"label": "MP3D ObjectNav v1", "family": "mp3d", "version": "v1", "semantics": "MP3D"},
    "hm3d_objectnav_v1": {"label": "HM3D ObjectNav v1", "family": "hm3d", "version": "v1", "semantics": "HM3DSem v0.1"},
    "hm3d_objectnav_v2": {"label": "HM3D ObjectNav v2", "family": "hm3d", "version": "v2", "semantics": "HM3DSem v0.2"},
}
MANIFEST = "objectnav_source_manifest.json"
COLLECTION_NOTE = "仅原始目标导航数据浏览；尚未接入轨迹生成与 GT/LeRobot 采集验收。"


def spec(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unknown ObjectNav dataset/version")
    return DATASETS[dataset]


def dataset_root(source_root, dataset):
    entry = spec(dataset)
    return Path(source_root) / "datasets/objectnav" / entry["family"] / entry["version"]


def scene_relative(dataset, scene_id, *, semantic=False):
    entry = spec(dataset)
    if entry["family"] == "mp3d":
        match = re.fullmatch(r"mp3d/([A-Za-z0-9]{11})/\1\.glb", scene_id)
        if not match:
            raise ValueError("Invalid MP3D ObjectNav scene path")
        return Path(scene_id)
    prefix = "hm3d" if entry["version"] == "v1" else "hm3d_v0.2"
    match = re.fullmatch(re.escape(prefix) + r"/(train|val|minival)/([0-9]{5}-([A-Za-z0-9]{11}))/\3\.basis\.glb", scene_id)
    if not match:
        raise ValueError("Invalid HM3D ObjectNav scene path/version")
    # Geometry changed too: never bind v1 episodes to v0.2 meshes/navmeshes.
    base = Path("hm3d/versions/v0.1") if entry["version"] == "v1" else Path("hm3d")
    split = "val" if match[1] == "minival" else match[1]
    return base / split / match[2] / f"{match[3]}.basis.glb"


def content_path(root, relative):
    if not re.fullmatch(r"(?:train|val|val_mini)/content/[A-Za-z0-9]{11}\.json\.gz", relative):
        raise ValueError("Invalid ObjectNav source path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("ObjectNav source is missing or escapes its dataset root")
    return path


def read_native(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("episodes"), list) or not isinstance(data.get("goals_by_category"), dict):
        raise ValueError("Unsupported native ObjectNav annotation layout")
    return data


def goal_key(episode):
    return Path(episode["scene_id"]).name + "_" + episode["object_category"]


def prepare(source_root, dataset):
    """Verify every input against the installed official archive and count rows."""
    entry = spec(dataset)
    source_root = Path(source_root)
    root = dataset_root(source_root, dataset)
    package = f"objectnav_{entry['family']}_{entry['version']}.zip"
    installed = json.loads((source_root / "installation_manifests/habitat" / (package + ".installed.json")).read_text())
    if not installed.get("complete") or installed.get("url") != PACKAGES[package]:
        raise ValueError("Official ObjectNav installation is not verified")
    records = []
    for file in installed["files"]:
        path = source_root / file["path"]
        if not path.is_relative_to(root) or path.parent.name != "content":
            continue
        relative = str(path.relative_to(root))
        path = content_path(root, relative)
        if path.stat().st_size != file["bytes"] or sha256(path) != file["sha256"]:
            raise ValueError(f"ObjectNav annotation changed: {relative}")
        native = read_native(path)
        episodes, goals = native["episodes"], native["goals_by_category"]
        if not episodes:
            raise ValueError(f"Empty native scene: {relative}")
        scene_id = episodes[0]["scene_id"]
        scene_relative(dataset, scene_id)
        categories, with_paths, explicit_stop = Counter(), 0, 0
        for row in episodes:
            if row.get("scene_id") != scene_id or "episode_id" not in row:
                raise ValueError(f"Scene or episode identity mismatch: {relative}")
            if not isinstance(row.get("object_category"), str) or goal_key(row) not in goals:
                raise ValueError(f"Unresolved native goal category: {relative}")
            categories[row["object_category"]] += 1
            paths = row.get("shortest_paths")
            with_paths += bool(paths)
            explicit_stop += bool(paths and any(p and p[-1] == 0 for p in paths))
        record = {"split": path.parent.parent.name, "scene": path.name.removesuffix(".json.gz"),
                  "source_scene_id": scene_id, "file": relative, "bytes": file["bytes"], "sha256": file["sha256"],
                  "episodes": len(episodes), "categories": dict(sorted(categories.items())),
                  "episodes_with_native_shortest_paths": with_paths, "episodes_with_explicit_action_0": explicit_stop,
                  "goal_groups": {k: {"objects": len(v), "view_points": sum(len(g.get("view_points") or []) for g in v)}
                                  for k, v in goals.items()}}
        records.append(record)
        print(json.dumps({"event": "objectnav_index", "dataset": dataset, "split": record["split"],
                          "scene": record["scene"], "episodes": record["episodes"]}), flush=True)
    if not records:
        raise ValueError("No installed ObjectNav scene annotations")
    manifest = {"schema_version": 1, "complete": True, "dataset": dataset, **entry,
                "source_url": installed["url"], "archive_sha256": installed["archive_sha256"],
                "task_type": "object_navigation", "original_language_instructions": False,
                "expert_replay_adapter": False, "web_collection_accepted": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scenes": sorted(records, key=lambda r: (r["split"], r["scene"]))}
    atomic_json(root / MANIFEST, manifest)
    (root / MANIFEST).chmod(0o664)
    return manifest


def manifest(source_root, dataset):
    path = dataset_root(source_root, dataset) / MANIFEST
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("dataset") != dataset or value.get("complete") is not True:
        raise ValueError("ObjectNav source index is not complete")
    return value


def asset_status(scenes_root, dataset, scene_id):
    root = Path(scenes_root).resolve()
    geometry = root / scene_relative(dataset, scene_id)
    semantic = root / scene_relative(dataset, scene_id, semantic=True)

    def present(path):
        try:
            return path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    # MP3D's existing RGB/navmesh deployment need not contain semantic assets.
    if spec(dataset)["family"] == "mp3d":
        semantics = present(semantic.with_suffix(".house")) and present(semantic.with_suffix(".semantic.ply"))
    else:
        semantics = all(present(semantic.with_name(semantic.name.replace(".basis.glb", suffix)))
                        for suffix in (".semantic.glb", ".semantic.txt"))
    return {"geometry_ready": present(geometry) and present(geometry.with_suffix(".navmesh")),
            "semantic_assets_ready": semantics, "can_collect": False,
            "collection_note": COLLECTION_NOTE}


def overview(source_root, scenes_root):
    result = []
    for dataset, entry in DATASETS.items():
        item = {"id": dataset, **entry, "available": False, "can_collect": False, "collection_note": COLLECTION_NOTE}
        try:
            data = manifest(source_root, dataset)
        except FileNotFoundError:
            result.append(item)
            continue
        splits = {}
        for scene in data["scenes"]:
            split = splits.setdefault(scene["split"], {"episodes": 0, "scenes": []})
            split["episodes"] += scene["episodes"]
            split["scenes"].append({k: scene[k] for k in ("scene", "episodes", "categories", "source_scene_id")}
                                   | asset_status(scenes_root, dataset, scene["source_scene_id"]))
        item.update(available=True, splits=splits, source_url=data["source_url"], archive_sha256=data["archive_sha256"])
        result.append(item)
    return {"items": result}


@lru_cache(maxsize=1)
def _cached_scene(path, mtime, size, expected_sha):
    # Cache only one decompressed scene, never millions of episodes globally.
    if sha256(path) != expected_sha:
        raise ValueError("ObjectNav scene file failed provenance hash verification")
    return read_native(path)


def selected_scene(source_root, dataset, split, scene):
    data = manifest(source_root, dataset)
    descriptor = next((r for r in data["scenes"] if r["split"] == split and r["scene"] == scene), None)
    if descriptor is None:
        raise ValueError("ObjectNav split/scene is not in the installed source index")
    path = content_path(dataset_root(source_root, dataset), descriptor["file"])
    st = path.stat()
    if st.st_size != descriptor["bytes"]:
        raise ValueError("ObjectNav source size changed")
    native = _cached_scene(str(path), st.st_mtime_ns, st.st_size, descriptor["sha256"])
    return data, descriptor, native


def summarize_row(dataset, descriptor, row, ordinal):
    identity = [dataset, descriptor["split"], descriptor["scene"], descriptor["sha256"], str(row["episode_id"]), ordinal]
    return {"id": hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24],
            "dataset": dataset, "split": descriptor["split"], "scene": descriptor["scene"],
            "source_episode_id": str(row["episode_id"]), "source_row": ordinal,
            "object_category": row["object_category"], "start_position": row.get("start_position"),
            "geodesic_distance": (row.get("info") or {}).get("geodesic_distance"),
            "has_native_shortest_paths": bool(row.get("shortest_paths")), "can_collect": False}


def page(source_root, scenes_root, dataset, split, scene, category="", offset=0, limit=30):
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("Invalid ObjectNav pagination")
    _, descriptor, native = selected_scene(source_root, dataset, split, scene)
    if category and category not in descriptor["categories"]:
        raise ValueError("Unknown object category in the selected scene")
    total = descriptor["categories"][category] if category else descriptor["episodes"]
    items, index = [], 0
    for ordinal, row in enumerate(native["episodes"]):
        if category and row["object_category"] != category:
            continue
        if index >= offset:
            items.append(summarize_row(dataset, descriptor, row, ordinal))
            if len(items) >= limit:
                break
        index += 1
    return {"items": items, "total": total, "offset": offset, "limit": limit,
            **asset_status(scenes_root, dataset, descriptor["source_scene_id"])}


def detail(source_root, scenes_root, dataset, split, scene, ordinal):
    data, descriptor, native = selected_scene(source_root, dataset, split, scene)
    if not 0 <= ordinal < len(native["episodes"]):
        raise ValueError("ObjectNav source row is out of range")
    row = native["episodes"][ordinal]
    goals = native["goals_by_category"][goal_key(row)]
    return {**summarize_row(dataset, descriptor, row, ordinal), "source_episode": row,
            "source_file": descriptor["file"], "source_sha256": descriptor["sha256"],
            "source_url": data["source_url"], "semantic_version": data["semantics"],
            "coordinate_frame": "Habitat world, Y-up, metres; quaternion [x,y,z,w]",
            "goals_key": goal_key(row), "goal_objects": [
                {k: v for k, v in goal.items() if k != "view_points"} | {
                    "view_point_count": len(goal.get("view_points") or []),
                    "view_point_preview": (goal.get("view_points") or [])[:3]}
                for goal in goals],
            "original_language_instructions": False,
            "trajectory_note": "原始目标位置是物体中心，不是机器人终点；view_points 是候选观测位姿。未生成轨迹、STOP 或训练标签。",
            **asset_status(scenes_root, dataset, descriptor["source_scene_id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        result = prepare(args.source_root, dataset)
        print(json.dumps({"dataset": dataset, "complete": result["complete"],
                          "episodes_including_val_mini": sum(s["episodes"] for s in result["scenes"])}), flush=True)


if __name__ == "__main__":
    main()
