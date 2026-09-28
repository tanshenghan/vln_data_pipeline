"""Pinned public SRDF round-3 routes. Discrete viewpoints are NOT replay GT."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catalog.vln_catalog import read_rows

REPOSITORY = "ZunWang/SRDF"
REVISION = "785901fab701bd2cad1f47204d56523e86a8725d"
ANNOTATION = "mantis.hm3d_round3_greedy_ndtw0.9_ranked_414k_rouge0.85.json"
SOURCE_ID = "srdf-hm3d-round3-public"
GRAPH_REPOSITORY = "OpenGVLab/ScaleVLN"
GRAPH_REVISION = "8ef638db28d49184e23ad3edf8aa5281fd8db7e3"
GRAPH_ARCHIVE = "r2r_preprocess_data.zip"
SOURCE_HASHES = {
    ANNOTATION: "4b221177d3cb71657f5db90275af5933e5bcd86a3a9fd048baba3d2fe4e53e5f",
    GRAPH_ARCHIVE: "d1820ae38a115e577df3aeb09360d6f2313b82b77a59476dff1386b04ab7345a",
}
CE_URL = "https://huggingface.co/datasets/SongzeLi/SRDF_CE_400k"
COLLECTION_REASON = (
    "SRDF 公开 round-3 版仅有离散路标路线，没有连续动作/逐帧位姿；"
    "SRDF-CE 400k 需作者批准 Hugging Face 访问后另行适配与采集验收。"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(directory):
    """Download only two pinned public inputs; no weights, images or credentials."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specs = [(REPOSITORY, REVISION, "data/" + ANNOTATION, ANNOTATION),
             (GRAPH_REPOSITORY, GRAPH_REVISION, GRAPH_ARCHIVE, GRAPH_ARCHIVE)]
    for repo, revision, remote, name in specs:
        target = directory / name
        if target.exists():
            if sha256(target) != SOURCE_HASHES[name]:
                raise ValueError(f"Existing input hash mismatch; not overwriting: {target}")
            continue
        partial = target.with_name(name + ".partial")
        size = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={size}-"} if size else {}
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}?download=true"
        print(json.dumps({"download": name, "resume_bytes": size}), flush=True)
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            if size and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {size}-")):
                raise ValueError("Server did not honor the resume range; existing partial preserved")
            with partial.open("ab" if size else "xb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        if sha256(partial) != SOURCE_HASHES[name]:
            raise ValueError(f"Downloaded SHA-256 mismatch: {name}")
        partial.rename(target)
        print(json.dumps({"verified": name, "bytes": target.stat().st_size}), flush=True)


def normalize(row):
    """Preserve each official instruction ID, heading and viewpoint sequence."""
    scan, route = row["scan"], row["path"]
    if not isinstance(scan, str) or not re.fullmatch(r"[0-9]{5}-[A-Za-z0-9]+", scan):
        raise ValueError("Invalid SRDF HM3D scan")
    if (not isinstance(route, list) or len(route) < 2
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9]{5}", v) for v in route)):
        raise ValueError("Invalid SRDF discrete viewpoint path")
    heading = row["heading"]
    if isinstance(heading, bool) or not isinstance(heading, (int, float)) or not math.isfinite(heading):
        raise ValueError("Non-finite SRDF heading")
    path_id = row["path_id"]
    if not isinstance(path_id, str) or not path_id.startswith("hm3d_" + scan + "__"):
        raise ValueError("SRDF path/scan identity mismatch")
    ids, texts = row["instr_ids"], row["instructions"]
    if not isinstance(ids, list) or not isinstance(texts, list) or not ids or len(ids) != len(texts):
        raise ValueError("SRDF instruction IDs/texts do not align")
    for iid, text in zip(ids, texts):
        if not isinstance(iid, str) or not re.fullmatch(re.escape(path_id) + r"_[0-9]+", iid):
            raise ValueError("SRDF instruction/path identity mismatch")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty SRDF instruction")
        yield {"episode_id": iid, "instruction_id": iid, "trajectory_id": path_id,
               "scan": scan, "path_id": path_id, "path": route, "heading": heading,
               "instructions": [text], "language": "en"}


def prepare_source(annotation, graph_archive, output):
    inputs = {ANNOTATION: Path(annotation), GRAPH_ARCHIVE: Path(graph_archive)}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if hashes != SOURCE_HASHES:
        raise ValueError("SRDF/ScaleVLN input SHA-256 differs from the pinned official release")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "train").mkdir()
    (output / "connectivity").mkdir()
    for name, path in inputs.items():
        shutil.copyfile(path, output / "raw" / name)
    counts, seen, routes = Counter(), set(), set()
    prepared = output / "train/train.jsonl"
    viewpoints = {}
    with prepared.open("x", encoding="utf-8") as stream:
        for row in read_rows(Path(annotation)):
            for normalized in normalize(row):
                iid, scan = normalized["episode_id"], normalized["scan"]
                if iid in seen:
                    raise ValueError(f"Duplicate SRDF instruction ID: {iid}")
                seen.add(iid)
                routes.add(normalized["trajectory_id"])
                counts[scan] += 1
                viewpoints.setdefault(scan, set()).update(normalized["path"])
                stream.write(json.dumps(normalized, ensure_ascii=False, allow_nan=False) + "\n")
    if not seen:
        raise ValueError("Empty SRDF source")
    graph_hashes = {}
    with zipfile.ZipFile(graph_archive) as archive:
        for scan, needed in sorted(viewpoints.items()):
            matches = [n for n in archive.namelist()
                       if n.endswith("/" + scan + "_connectivity.json")
                       and "connectivity" in Path(n).parts]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one official connectivity graph for {scan}: {matches}")
            # Never extract archive paths. Only write validated scan-based filenames.
            with archive.open(matches[0]) as stream:
                graph = json.load(stream)
            compact, available = [], set()
            for item in graph:
                if not item.get("included"):
                    continue
                pose = item.get("pose", [])
                if len(pose) != 16 or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in pose):
                    raise ValueError(f"Invalid graph pose for {scan}")
                vid = str(item["image_id"])
                if vid in available:
                    raise ValueError(f"Duplicate graph viewpoint: {scan}/{vid}")
                available.add(vid)
                compact.append({"image_id": vid, "included": True, "pose": pose})
            if not needed <= available:
                raise ValueError(f"SRDF route contains missing graph viewpoints: {scan}")
            target = output / "connectivity" / (scan + "_connectivity.json")
            target.write_text(json.dumps(compact, allow_nan=False) + "\n")
            graph_hashes[str(target.relative_to(output))] = sha256(target)
    manifest = {
        "format": "srdf_discrete_round3_v1", "complete": True,
        "source_id": SOURCE_ID, "repository": REPOSITORY, "revision": REVISION,
        "graph_repository": GRAPH_REPOSITORY, "graph_revision": GRAPH_REVISION,
        "raw_sha256": hashes, "prepared_sha256": {"train/train.jsonl": sha256(prepared), **graph_hashes},
        "episodes": len(seen), "routes": len(routes), "scene_counts": dict(sorted(counts.items())),
        "release_scope": "public intermediate HM3D round-3 subset; NOT the unrecoverable final SRDF-20M",
        "trajectory_kind": "discrete_viewpoint_reference_only", "can_collect": False,
        "coordinate_frame": "unchanged ScaleVLN connectivity poses; not converted to Habitat body poses",
        "has_expert_actions": False, "has_frame_poses": False,
        "continuous_release_url": CE_URL, "collection_reason": COLLECTION_REASON,
        "adapter_sha256": sha256(__file__),
    }
    # This is the only completion marker, written after every source/path check.
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def prepared_manifest(root, *, verify_hashes=False):
    root = Path(root)
    manifest = json.loads((root / "source_manifest.json").read_text())
    if (manifest.get("format") != "srdf_discrete_round3_v1" or manifest.get("complete") is not True
            or manifest.get("revision") != REVISION or manifest.get("graph_revision") != GRAPH_REVISION
            or manifest.get("source_id") != SOURCE_ID or manifest.get("raw_sha256") != SOURCE_HASHES):
        raise ValueError("SRDF preparation/provenance is incomplete or unsupported")
    expected = {"train/train.jsonl"}
    for scan in manifest["scene_counts"]:
        if not re.fullmatch(r"[0-9]{5}-[A-Za-z0-9]+", scan):
            raise ValueError("Unsafe SRDF manifest scan")
        expected.add(f"connectivity/{scan}_connectivity.json")
    if set(manifest.get("prepared_sha256", {})) != expected:
        raise ValueError("SRDF prepared file inventory mismatch")
    for relative in expected:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"SRDF prepared source missing: {relative}")
        if verify_hashes and sha256(path) != manifest["prepared_sha256"][relative]:
            raise ValueError(f"SRDF prepared hash mismatch: {relative}")
    return manifest


def availability(root, scenes_root):
    from data_pipeline.scalevln_source import scene_ready
    result = {"source_ready": False, "can_collect": False, "collection_reason": COLLECTION_REASON,
              "continuous_release_url": CE_URL, "trajectory_kind": "discrete_viewpoint_reference_only"}
    try:
        manifest = prepared_manifest(root)
        ready = {scan: n for scan, n in manifest["scene_counts"].items()
                 if scene_ready(scenes_root, f"hm3d/train/{scan}/{scan.split('-', 1)[1]}.basis.glb")}
        result.update(source_ready=True, total_episodes=manifest["episodes"], total_routes=manifest["routes"],
                      total_scenes=len(manifest["scene_counts"]), ready_scenes=len(ready),
                      scene_ready_episodes=sum(ready.values()), version="public HM3D round-3 (discrete)",
                      release_scope=manifest["release_scope"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download", help="Only public SRDF round-3 + ScaleVLN graphs, ~786 MB")
    get.add_argument("--directory", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--annotation", type=Path, required=True)
    prepare.add_argument("--graphs", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True, help="NEW shared datasets/SRDF directory")
    args = parser.parse_args()
    if args.command == "download":
        download(args.directory)
    else:
        result = prepare_source(args.annotation, args.graphs, args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ("scene_counts", "prepared_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
