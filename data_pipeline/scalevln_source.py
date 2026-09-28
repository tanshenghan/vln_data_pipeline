"""Pinned StreamVLN ScaleVLN-CE adapter; never treats arbitrary EOF as STOP."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catalog.vln_catalog import read_rows

REVISION = "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"
REPOSITORY = "cywan/StreamVLN-Trajectory-Data"
SOURCE_ID = "scalevln-streamvln-150k"
SOURCE_HASHES = {
    "scalevln_subset_150k.json.gz": "23e31a057858691e239d019a7605c622bfe266aa6e0f9a68ddcdd7830d81abdf",
    "annotations.json": "2952d8ec63c156e9fed6032217315dfd52c8a6587a8b1af55445633463b4442e",
}
STOP_CONVENTION = "streamvln_published_omitted_stop_v1"
STOP_REFERENCE = f"https://huggingface.co/datasets/{REPOSITORY}/blob/{REVISION}/README.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_relative(scene_id):
    """Canonical shared layout: hm3d/train/<number-id>/<id>.basis.glb."""
    value = str(scene_id).removeprefix("data/scene_datasets/")
    parts = Path(value).parts
    if len(parts) == 3 and parts[0] == "hm3d":
        parts = (parts[0], "train", *parts[1:])
    if len(parts) != 4 or parts[:2] != ("hm3d", "train"):
        raise ValueError(f"ScaleVLN requires an HM3D train scene, got {scene_id!r}")
    if not re.fullmatch(r"[0-9]{5}-[A-Za-z0-9]+", parts[2]):
        raise ValueError(f"Invalid HM3D scene directory: {scene_id!r}")
    if parts[3] != parts[2].split("-", 1)[1] + ".basis.glb":
        raise ValueError(f"HM3D scene filename does not match its directory: {scene_id!r}")
    return Path(*parts)


def replay_actions(actions):
    """Apply ONLY the pinned publisher's dummy/omitted-STOP representation.

    The collector additionally verifies real simulator endpoint success before
    supervising STOP. This is not a rule for generic truncated trajectories.
    """
    if (not isinstance(actions, list) or len(actions) < 2 or actions[0] != -1
            or any(type(a) is not int or a not in (1, 2, 3) for a in actions[1:])):
        raise ValueError("ScaleVLN annotation must contain one leading -1 dummy followed by 1/2/3 actions")
    return [*actions[1:], 0]


def point(value, length=3):
    if (not isinstance(value, list) or len(value) != length
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value)):
        raise ValueError("Non-finite or malformed ScaleVLN pose/reference point")
    return value


def normalize_episode(episode, annotation):
    eid = str(episode["episode_id"])
    if not re.fullmatch(r"[0-9]+", eid) or str(annotation["id"]) != eid:
        raise ValueError("ScaleVLN episode/annotation ID mismatch")
    relative = scene_relative(episode["scene_id"])
    text = episode["instruction"]["instruction_text"]
    if not isinstance(text, str) or not text.strip() or annotation.get("instructions") != [text]:
        raise ValueError(f"ScaleVLN instruction mismatch: {eid}")
    expected_video = f"images/{relative.parts[2]}_scalevln_{int(eid):06d}"
    if annotation.get("video") != expected_video:
        raise ValueError(f"ScaleVLN scene/video identity mismatch: {eid}")
    route = [point(p) for p in episode["reference_path"]]
    if len(route) < 2:
        raise ValueError(f"Missing ScaleVLN reference polyline: {eid}")
    point(episode["start_position"])
    rotation = point(episode["start_rotation"], 4)
    if abs(sum(x * x for x in rotation) - 1) > 1e-5:
        raise ValueError(f"Non-unit source quaternion: {eid}")
    point(episode["goals"][0]["position"])
    actions = replay_actions(annotation["actions"])
    normalized = dict(episode, scene_id=relative.as_posix(),
                      instruction={**episode["instruction"], "language": "en"})
    normalized["source_metadata"] = {
        "dataset": "scalevln", "split": "train", "language": "en",
        "source_id": SOURCE_ID, "revision": REVISION,
        "original_scene_id": episode["scene_id"], "annotation_id": annotation["id"],
        "source_video_id": annotation["video"], "synthetic_instruction": True,
        "source_action_count": len(annotation["actions"]), "replay_frame_count": len(actions),
        "source_turn_angle_deg": 15, "replay_turn_angle_deg": 15, "forward_step_m": .25,
        "stop_convention": STOP_CONVENTION, "stop_reference": STOP_REFERENCE,
        "action_conversion": "remove the leading -1 dummy; restore publisher-omitted final STOP; no resampling",
        "route_reference": "sparse source subgoal polyline, NOT per-frame body-pose GT",
    }
    return normalized, {"actions": actions, "locations": route, "locations_source": "source_reference_path"}


def prepare_source(episodes_path, annotations_path, output, readme=None):
    """Normalize the complete pinned release, independent of installed scenes."""
    inputs = {"scalevln_subset_150k.json.gz": Path(episodes_path), "annotations.json": Path(annotations_path)}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if hashes != SOURCE_HASHES:
        raise ValueError("Input SHA-256 does not match the supported StreamVLN revision; refusing to invent STOP semantics")
    annotations = {}
    for row in read_rows(inputs["annotations.json"]):
        eid = str(row["id"])
        if eid in annotations:
            raise ValueError(f"Duplicate ScaleVLN action annotation: {eid}")
        replay_actions(row["actions"])
        annotations[eid] = row
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "train").mkdir()
    (output / "raw").mkdir()
    for name, path in inputs.items():
        shutil.copyfile(path, output / "raw" / name)
    if readme:
        shutil.copyfile(readme, output / "raw" / "README.md")
    counts, seen, frames = Counter(), set(), 0
    ep_file, gt_file = output / "train/train.json.gz", output / "train/train_gt.json.gz"
    with gzip.open(ep_file, "wt", encoding="utf-8") as eps, gzip.open(gt_file, "wt", encoding="utf-8") as gts:
        eps.write('{"episodes":[')
        gts.write("{")
        for row in read_rows(inputs["scalevln_subset_150k.json.gz"]):
            eid = str(row["episode_id"])
            if eid in seen or eid not in annotations:
                raise ValueError(f"Duplicate episode or missing action annotation: {eid}")
            normalized, expert = normalize_episode(row, annotations.pop(eid))
            if seen:
                eps.write(",")
                gts.write(",")
            json.dump(normalized, eps, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            gts.write(json.dumps(eid) + ":")
            json.dump(expert, gts, allow_nan=False, separators=(",", ":"))
            seen.add(eid)
            counts[normalized["scene_id"]] += 1
            frames += len(expert["actions"])
        eps.write("]}")
        gts.write("}")
    if annotations or not seen:
        raise ValueError("Episode/annotation sets differ or are empty; prepared source is not ready")
    manifest = {
        "format": "scalevln_streamvln_prepared_v1", "complete": True,
        "repository": REPOSITORY, "revision": REVISION, "raw_sha256": hashes,
        "episodes": len(seen), "frames": frames, "scene_counts": dict(sorted(counts.items())),
        "stop_convention": STOP_CONVENTION, "stop_reference": STOP_REFERENCE,
        "prepared_sha256": {str(p.relative_to(output)): sha256(p) for p in (ep_file, gt_file)},
        "adapter_sha256": sha256(__file__),
    }
    # The completion marker is written last. Never publish partial conversion.
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepared_manifest(root, *, verify_hashes=False):
    manifest = json.loads((Path(root) / "source_manifest.json").read_text())
    if (manifest.get("format") != "scalevln_streamvln_prepared_v1" or manifest.get("complete") is not True
            or manifest.get("revision") != REVISION or manifest.get("raw_sha256") != SOURCE_HASHES
            or manifest.get("stop_convention") != STOP_CONVENTION):
        raise ValueError("ScaleVLN source preparation/provenance is incomplete or unsupported")
    for relative in ("train/train.json.gz", "train/train_gt.json.gz"):
        if not (Path(root) / relative).is_file():
            raise FileNotFoundError(f"ScaleVLN prepared source is missing: {relative}")
        if verify_hashes and sha256(Path(root) / relative) != manifest["prepared_sha256"].get(relative):
            raise ValueError(f"ScaleVLN prepared source hash mismatch: {relative}")
    return manifest


def scene_ready(scenes_root, scene_id):
    scene = Path(scenes_root) / scene_relative(scene_id)
    try:
        for path in (scene, scene.with_suffix(".navmesh")):
            with path.open("rb") as handle:
                if not handle.read(1):
                    return False
        return True
    except OSError:
        return False


def availability(root, scenes_root):
    """Report partial scene coverage without marking other datasets unready."""
    try:
        manifest = prepared_manifest(root)
        deployed = {scene: count for scene, count in manifest["scene_counts"].items() if scene_ready(scenes_root, scene)}
        return {"source_ready": True, "total_episodes": manifest["episodes"],
                "total_scenes": len(manifest["scene_counts"]), "ready_episodes": sum(deployed.values()),
                "ready_scenes": len(deployed), "scene_paths": list(deployed),
                "can_collect": bool(deployed), "full_scene_coverage": len(deployed) == len(manifest["scene_counts"])}
    except (OSError, ValueError, KeyError) as exc:
        return {"source_ready": False, "can_collect": False, "ready_episodes": 0, "ready_scenes": 0,
                "scene_paths": [], "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New shared datasets/ScaleVLN_StreamVLN directory")
    parser.add_argument("--readme", type=Path, help="Optional pinned publisher dataset card to preserve")
    args = parser.parse_args()
    manifest = prepare_source(args.episodes, args.annotations, args.output, args.readme)
    print(json.dumps({k: v for k, v in manifest.items() if k != "scene_counts"}, indent=2))


if __name__ == "__main__":
    main()
