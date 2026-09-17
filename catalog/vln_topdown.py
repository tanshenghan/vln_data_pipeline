#!/usr/bin/env python3
"""Render navigability maps from NavMesh, with CPU PathFinder only.

Run this module with a Python environment containing habitat_sim, numpy and PIL.
No simulator, GPU context, RGB rendering, or source-file modification is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def height_slices(vertices, indices):
    import numpy as np
    triangles = vertices[indices.reshape(-1, 3)]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    heights = triangles[:, :, 1].mean(axis=1)
    spans = np.ptp(triangles[:, :, 1], axis=1)
    bins = {}
    for height, weight, span in zip(heights, area, spans):
        if span <= 0.2:
            key = round(float(height), 1)
            bins[key] = bins.get(key, 0) + float(weight)
    if not bins:
        return [float(np.median(vertices[:, 1]))]
    threshold = max(2.0, max(bins.values()) * 0.01)
    candidates = sorted(bins, key=lambda height: -bins[height])
    selected = []
    for height in candidates:
        if bins[height] >= threshold and all(abs(height - previous) >= 1.4 for previous in selected):
            selected.append(height)
    return sorted(selected or [candidates[0]])


def render_scene(job):
    import habitat_sim
    import numpy as np
    from PIL import Image

    scene, root, destination, requested_mpp, force = job
    navmesh = Path(root) / scene / f"{scene}.navmesh"
    if not navmesh.is_file():
        return {"scene": scene, "error": "navmesh_missing"}
    fingerprint = hashlib.sha256(json.dumps([str(navmesh), navmesh.stat().st_size, navmesh.stat().st_mtime_ns,
                                            requested_mpp, "navmesh-slices-v1"]).encode()).hexdigest()[:16]
    output = Path(destination) / scene / fingerprint
    manifest = output / "map.json"
    if manifest.is_file() and not force:
        cached = json.loads(manifest.read_text())
        if all(Path(floor["image_path"]).is_file() for floor in cached["floors"]):
            return cached
    finder = habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(navmesh)):
        return {"scene": scene, "error": "navmesh_unreadable"}
    vertices = np.asarray(finder.build_navmesh_vertices())
    indices = np.asarray(finder.build_navmesh_vertex_indices())
    if not len(vertices) or not len(indices):
        return {"scene": scene, "error": "navmesh_empty"}
    lower, upper = finder.get_bounds()
    # Bound very large meshes without silently claiming 4 cm resolution.
    mpp = max(requested_mpp, float(max(upper[0] - lower[0], upper[2] - lower[2])) / 2400)
    output.mkdir(parents=True, exist_ok=True)
    floors = []
    for index, height in enumerate(height_slices(vertices, indices)):
        grid = finder.get_topdown_view(mpp, height, 0.75)
        if not grid.any():
            continue
        free = grid.astype(bool)
        padded = np.pad(free, 1, constant_values=False)
        inner = free & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        colors = np.empty((*free.shape, 3), dtype=np.uint8)
        colors[:] = (222, 229, 239)
        colors[free] = (250, 252, 255)
        colors[free & ~inner] = (109, 128, 153)
        image_path = output / f"floor-{index}.png"
        temporary = output / f"floor-{index}.tmp.png"
        Image.fromarray(colors).save(temporary, optimize=True)
        temporary.replace(image_path)
        rows, columns = free.shape
        floors.append({
            "id": str(index), "height": height, "height_tolerance": 0.75,
            "image_path": str(image_path.resolve()), "width": columns, "height_pixels": rows,
            "origin": [float(lower[0]), float(lower[2])], "meters_per_pixel": mpp,
            "navigable_area_m2": round(float(free.sum()) * mpp * mpp, 2),
            "world_bounds": [float(lower[0]), float(lower[2]), float(lower[0])+columns*mpp, float(lower[2])+rows*mpp],
        })
    result = {"scene": scene, "kind": "navmesh_height_slices", "source_path": str(navmesh),
              "floor_method": "area_weighted_navmesh_height_peaks", "coordinate_system": "habitat_xz",
              "requested_meters_per_pixel": requested_mpp, "floors": floors}
    temp_manifest = output / "map.tmp.json"
    temp_manifest.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    temp_manifest.replace(manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--meters-per-pixel", type=float, default=0.04)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.meters_per_pixel) or not 0.01 <= args.meters_per_pixel <= 0.5:
        parser.error("meters-per-pixel must be between 0.01 and 0.5")
    scenes = json.loads(args.scenes.read_text())
    results = []
    jobs = [(scene, str(args.scene_root.resolve()), str(args.output_dir.resolve()), args.meters_per_pixel, args.force) for scene in scenes]
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        pending = {executor.submit(render_scene, job): job[0] for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"Map {len(results)}/{len(scenes)} {result['scene']}: {len(result.get('floors', []))} slices {result.get('error', '')}", flush=True)
    args.manifest.write_text(json.dumps(sorted(results, key=lambda item: item["scene"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
