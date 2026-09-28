"""Install verified official archives without replacing existing shared assets.

May run with --watch alongside download_habitat_assets.py. Never reads credentials.
HM3D v0.1 has separate geometry AND semantics; v0.2 is the canonical HM3D tree.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
import time
import zipfile

try:
    from .download_habitat_assets import PACKAGES, atomic_json, sha256
except ImportError:
    from download_habitat_assets import PACKAGES, atomic_json, sha256


SCENE = r"[0-9]{5}-[A-Za-z0-9]{11}"
OBJECTNAV = {
    "objectnav_mp3d_v1.zip": ("mp3d", "v1", ""),
    "objectnav_hm3d_v1.zip": ("hm3d", "v1", "objectnav_hm3d_v1"),
    "objectnav_hm3d_v2.zip": ("hm3d", "v2", "objectnav_hm3d_v2"),
}


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\x00" in name
            or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def destination(name, member, source_root):
    """Allow only the publisher's known layout, never arbitrary extraction paths."""
    path = safe_name(member)
    if name in OBJECTNAV:
        family, version, prefix = OBJECTNAV[name]
        parts = path.parts
        if prefix:
            if parts[0] != prefix:
                raise ValueError("Unexpected ObjectNav archive prefix")
            parts = parts[1:]
        if not parts:
            return None
        relative = PurePosixPath(*parts)
        if not re.fullmatch(r"(?:train|val|val_mini)/(?:content/[A-Za-z0-9]+|train|val|val_mini)\.json\.gz", str(relative)):
            raise ValueError(f"Unexpected ObjectNav file: {relative}")
        return source_root / "datasets/objectnav" / family / version / relative
    split = name.split("-")[1]
    base = source_root / "scene_datasets/hm3d"
    if "v0.1" in name or name in ("hm3d-train-habitat.tar", "hm3d-val-habitat.tar"):
        base = base / "versions/v0.1"
    if "configs" in name:
        if len(path.parts) != 1 or not re.fullmatch(r"hm3d_(?:annotated_)?(?:(?:train|val)_)?basis\.scene_dataset_config\.json", path.name):
            raise ValueError(f"Unexpected HM3D config: {member}")
        return base / split / path if f"_{split}_" in path.name else base / path
    suffix = r"semantic\.(?:glb|txt)" if "semantic-annots" in name else r"basis\.(?:glb|navmesh|scene_instance\.json)"
    if len(path.parts) != 2 or not re.fullmatch(SCENE, path.parts[0]):
        raise ValueError(f"Unexpected HM3D scene: {member}")
    scene_hash = path.parts[0].split("-", 1)[1]
    if not re.fullmatch(re.escape(scene_hash) + r"\." + suffix, path.name):
        raise ValueError(f"Unexpected HM3D scene asset: {member}")
    return base / split / path


@contextmanager
def archive_files(archive):
    """Yield (name, size, open) records, rejecting all special files and links."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            def members():
                for item in handle.infolist():
                    safe_name(item.filename)
                    mode = item.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError("ZIP special file/link rejected")
                    if item.is_dir():
                        continue
                    yield item.filename, item.file_size, lambda item=item: handle.open(item)
            yield members()
    else:
        with tarfile.open(archive, "r:*") as handle:
            def members():
                for item in handle:
                    safe_name(item.name)
                    if item.isdir():
                        continue
                    if not item.isfile() or item.issparse():
                        raise ValueError("TAR special file/link rejected")
                    yield item.name, item.size, lambda item=item: handle.extractfile(item)
            yield members()


def checked_path(path, source_root):
    resolved = path.resolve()
    if not resolved.is_relative_to(source_root.resolve()):
        raise ValueError(f"Shared destination escapes source root: {path}")
    # Existing example aliases are permitted, but cannot escape the HM3D tree.
    hm3d = source_root / "scene_datasets/hm3d"
    if path.is_relative_to(hm3d) and not resolved.is_relative_to(hm3d.resolve()):
        raise ValueError("HM3D symlink escapes its scene root")
    return path


def mkdir_shared(path, boundary):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        if current == boundary:
            break
        current = current.parent
    for folder in reversed(missing):
        folder.mkdir(exist_ok=True)
        os.chown(folder, -1, boundary.stat().st_gid)
        folder.chmod(0o2775)


def publish_file(staged, target, source_root, digest):
    checked_path(target, source_root)
    mkdir_shared(target.parent, source_root)
    if target.exists() or target.is_symlink():
        if not target.is_file() or sha256(target) != digest:
            raise ValueError(f"Existing asset differs; not overwritten: {target}")
        return
    # Atomic no-clobber publication on the destination filesystem.
    try:
        os.link(staged, target)
    except FileExistsError:
        if not target.is_file() or sha256(target) != digest:
            raise ValueError(f"Concurrent destination conflict: {target}") from None


def install_package(name, download_root, source_root):
    source_root = Path(source_root).resolve()
    archive = Path(download_root) / name
    download = json.loads(archive.with_name(name + ".download.json").read_text())
    if (not download.get("complete") or download.get("url") != PACKAGES[name]
            or archive.stat().st_size != download.get("bytes") or sha256(archive) != download.get("sha256")):
        raise ValueError(f"Unverified archive: {name}")
    reports = source_root / "installation_manifests/habitat"
    mkdir_shared(reports, source_root)
    report = reports / (name + ".installed.json")
    # Even a completed installation is checked if explicitly re-run.
    print(json.dumps({"event": "install", "file": name}), flush=True)
    files, seen = [], set()
    with tempfile.TemporaryDirectory(prefix=".habitat-install-", dir=source_root) as temporary:
        staging = Path(temporary)
        remaining = shutil.disk_usage(staging).free - 5 * 1024**3
        with archive_files(archive) as members:
            for member, size, opener in members:
                target = destination(name, member, source_root)
                if target is None or target in seen:
                    raise ValueError("Empty or duplicate archive file")
                seen.add(target)
                if size < 0 or size > remaining:
                    raise ValueError("Insufficient free space for safe archive staging")
                remaining -= size
                staged = staging / str(len(files))
                digest, copied = hashlib.sha256(), 0
                with opener() as src, staged.open("xb") as dst:
                    while block := src.read(4 * 1024**2):
                        copied += len(block)
                        if copied > size:
                            raise ValueError("Archive member exceeds declared size")
                        digest.update(block)
                        dst.write(block)
                if copied != size:
                    raise ValueError("Truncated archive member")
                staged.chmod(0o664)
                os.chown(staged, -1, source_root.stat().st_gid)
                checked_path(target, source_root)
                # Check every conflict before publishing any file in this archive.
                if target.exists() or target.is_symlink():
                    if not target.is_file() or sha256(target) != digest.hexdigest():
                        raise ValueError(f"Existing asset differs; not overwritten: {target}")
                files.append({"path": str(target.relative_to(source_root)), "bytes": copied,
                              "sha256": digest.hexdigest()})
        if not files:
            raise ValueError("Empty archive")
        for index, item in enumerate(files):
            publish_file(staging / str(index), source_root / item["path"], source_root, item["sha256"])
    result = {"complete": True, "package": name, "url": download["url"],
              "archive_sha256": download["sha256"], "files": files,
              "installed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(report, result)
    report.chmod(0o664)
    print(json.dumps({"event": "installed", "file": name, "files": len(files)}), flush=True)
    return result


def inventory(source_root, version="v0.2"):
    base = source_root / "scene_datasets/hm3d"
    if version == "v0.1":
        base = base / "versions/v0.1"
    counts = {}
    for split, expected in (("train", 800), ("val", 100)):
        complete = [p.parent.name for p in (base / split).glob("*/*.basis.glb")
                    if p.is_file() and p.with_suffix(".navmesh").is_file()]
        counts[split] = {"ready_scenes": len(set(complete)), "expected_scenes": expected}
    return {"hm3d": counts, "public_geometry_complete": all(
        v["ready_scenes"] == v["expected_scenes"] for v in counts.values()),
        "test_split": "not_publicly_released; not downloaded"}


def consumer_coverage(source_root):
    """Measure real input coverage without scheduling GPU jobs or rebuilding Atlas."""
    try:
        from .scalevln_source import availability
        from .objectnav_source import overview
    except ImportError:
        from scalevln_source import availability
        from objectnav_source import overview
    scenes = source_root / "scene_datasets"
    scale = availability(source_root / "datasets/ScaleVLN_StreamVLN", scenes)
    result = {"scalevln": {k: v for k, v in scale.items() if k != "scene_paths"}, "objectnav": {}}
    for dataset in overview(source_root, scenes)["items"]:
        result["objectnav"][dataset["id"]] = {
            "source_index_ready": dataset["available"], "can_collect": False,
            "splits": {split: {"scenes": len(part["scenes"]), "episodes": part["episodes"],
                "geometry_ready_scenes": sum(s["geometry_ready"] for s in part["scenes"]),
                "semantic_ready_scenes": sum(s["semantic_assets_ready"] for s in part["scenes"])}
                for split, part in dataset.get("splits", {}).items()}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--packages", nargs="+", choices=sorted(PACKAGES), default=list(PACKAGES))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    if not args.source_root.is_dir():
        raise ValueError("Shared source root must already exist")
    reports = args.source_root / "installation_manifests/habitat"
    mkdir_shared(reports, args.source_root)
    with (reports / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done, failed = set(), {}
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            for name in args.packages:
                if name in done or name in failed:
                    continue
                report = args.downloads / (name + ".download.json")
                if not report.is_file() or not json.loads(report.read_text()).get("complete"):
                    continue
                try:
                    install_package(name, args.downloads, args.source_root)
                    done.add(name)
                except Exception as exc:
                    failed[name] = str(exc)
                    print(json.dumps({"event": "install_failed", "file": name, "error": str(exc)}), flush=True)
            pending = sorted(set(args.packages) - done - set(failed))
            state = {"complete": len(done) == len(args.packages),
                     "installed": sorted(done), "failed": failed, "pending": pending,
                     "legacy_v0_1": inventory(args.source_root, "v0.1"), **inventory(args.source_root),
                     "consumer_coverage": consumer_coverage(args.source_root),
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(reports / "status.json", state)
            (reports / "status.json").chmod(0o664)
            finished_packages = set()
            for summary in args.downloads.glob("*download_summary.json"):
                finished_packages.update(json.loads(summary.read_text()).get("packages", []))
            downloads_finished = set(args.packages).issubset(finished_packages)
            if not args.watch or not pending or downloads_finished or time.monotonic() >= deadline:
                print(json.dumps(state), flush=True)
                raise SystemExit(0 if state["complete"] else 1)
            time.sleep(30)


if __name__ == "__main__":
    main()
