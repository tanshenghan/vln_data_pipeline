"""Ground-truth sidecar indexing and read-only geometry lookups."""
from __future__ import annotations

import gzip
import json
import math
import sqlite3

try:
    from .vln_catalog import xyz
except ImportError:
    from vln_catalog import xyz


def read_gt_entries(path):
    """Decode one keyed GT entry at a time, including compressed training files."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        buffer, eof = "", False
        decoder = json.JSONDecoder()

        def fill():
            nonlocal buffer, eof
            chunk = handle.read(65536)
            buffer += chunk
            eof = not chunk

        def space():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        def take(character):
            nonlocal buffer
            space()
            if not buffer.startswith(character):
                raise ValueError(f"Invalid GT JSON, expected {character}: {path}")
            buffer = buffer[1:]

        def value():
            nonlocal buffer
            space()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    return result
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Truncated GT JSON: {path}") from None
                    fill()

        take("{")
        space()
        if not buffer.startswith("}"):
            while True:
                key = value()
                if not isinstance(key, str):
                    raise ValueError(f"GT entry ID must be a string: {path}")
                take(":")
                entry = value()
                if not isinstance(entry, dict):
                    raise ValueError(f"GT entry must be an object: {path}:{key}")
                yield key, entry
                space()
                if buffer.startswith("}"):
                    break
                take(",")
        take("}")
        space()
        if buffer:
            raise ValueError(f"Unexpected trailing GT JSON: {path}")


def normalize_gt(entry, start):
    # ScaleVLN's reference_path is a sparse route guide, not per-step GT.
    # Even coincidentally matching action counts cannot establish alignment.
    if entry.get("locations_source") == "source_reference_path":
        raise ValueError("reference_path_only")
    raw = entry.get("locations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing_locations")
    points = [xyz(item) for item in raw]
    if any(point is None for point in points):
        raise ValueError("invalid_coordinates")
    start = xyz(start)
    error = math.dist(start, points[0]) if start else None
    if error is not None and error > 0.5:
        raise ValueError("start_mismatch")
    actions = entry.get("actions") or []
    if not isinstance(actions, list) or any(type(action) is not int for action in actions):
        raise ValueError("invalid_actions")
    # GT locations usually contain translations only. Turns and STOP do not
    # consume a location. Reconstruct this association only when counts agree.
    aligned = all(action in {0, 1, 2, 3} for action in actions) and actions.count(1) == len(points) - 1
    indices = [0] if aligned and actions else []
    if indices:
        for action in actions:
            indices.append(indices[-1] + int(action == 1))
    return {
        "locations": points, "actions": actions, "action_location_indices": indices,
        "point_count": len(points), "action_count": len(actions),
        "length": round(sum(math.dist(a, b) for a, b in zip(points, points[1:])), 4),
        "start_error": error, "forward_steps": entry.get("forward_steps"),
        "coordinate_system": "habitat_xyz_y_up", "units": "meters",
        "action_alignment": "forward_steps_only" if indices else "unavailable",
        "action_names": {"0": "STOP", "1": "FORWARD", "2": "LEFT", "3": "RIGHT"},
    }


def open_geometry(path):
    if not path.is_file():
        return None
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def geometry_for_record(db, record):
    missing = {"available": False, "reason": "test_unpublished" if record["split"] == "test" else "not_indexed"}
    result = {"ground_truth": missing, "topdown": None}
    if db is None:
        return result
    row = db.execute("SELECT payload FROM ground_truth WHERE record_id=?", (record["id"],)).fetchone()
    if row:
        result["ground_truth"] = {"available": True, **json.loads(row[0])}
    else:
        issue = db.execute("SELECT reason FROM gt_issues WHERE record_id=?", (record["id"],)).fetchone()
        if issue:
            result["ground_truth"]["reason"] = issue[0]
    # Habitat maps cannot be overlaid on native Matterport X/Y coordinates.
    if record.get("projection_axes") == [0, 2]:
        row = db.execute("SELECT payload FROM scene_maps WHERE scene=?", (record["scene"],)).fetchone()
        if row:
            result["topdown"] = json.loads(row[0])
    return result
