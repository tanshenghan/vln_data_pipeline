"""Read VLN annotations into a paginated, read-only SQLite catalogue.

Only configured annotation files are read. Habitat and model dependencies are not
needed; building the catalogue never runs a simulator or modifies source data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
SPLITS = ("train", "val_seen", "val_unseen", "test")
FILTERS = ("dataset", "source_id", "split", "scene", "language", "role")
SUMMARY_COLUMNS = "id,source_id,dataset,version,format,split,scene,language,role,episode_id,instruction_id,trajectory_id,route_id,instruction,word_count,path_length,geodesic_distance,point_count,has_media,issues"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json_text(parts).encode()).hexdigest()[:24]


def json_safe(value: Any) -> Any:
    """RxR uses NaN for unavailable word timestamps; expose JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL, JSON list, or Habitat {episodes: [...]} annotation.

    Decode one episode at a time, including large RxR train files. Do not load
    instruction vocabularies or unrelated trailing fields into the catalogue.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        if ".jsonl" in path.suffixes:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{number}: expected an object")
                    yield row
            return
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            chunk = handle.read(65536)
            buffer += chunk
            eof = not chunk

        def whitespace() -> None:
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        def decode() -> Any:
            nonlocal buffer
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Truncated or invalid JSON: {path}") from None
                    refill()

        whitespace()
        wrapped = buffer.startswith("{")
        if wrapped:
            buffer = buffer[1:]
            while True:
                whitespace()
                key = decode()
                whitespace()
                if not buffer.startswith(":"):
                    raise ValueError(f"Invalid annotation object: {path}")
                buffer = buffer[1:]
                whitespace()
                if key == "episodes":
                    break
                decode()
                whitespace()
                if not buffer.startswith(","):
                    raise ValueError(f"Missing episodes array: {path}")
                buffer = buffer[1:]
        whitespace()
        if not buffer.startswith("["):
            raise ValueError(f"Expected episodes array or JSONL: {path}")
        buffer = buffer[1:]
        whitespace()
        if not buffer.startswith("]"):
            while True:
                row = decode()
                if not isinstance(row, dict):
                    raise ValueError(f"Expected episode object: {path}")
                yield row
                whitespace()
                if buffer.startswith("]"):
                    break
                if not buffer.startswith(","):
                    raise ValueError(f"Invalid episode separator: {path}")
                buffer = buffer[1:]
                whitespace()
        buffer = buffer[1:]
        # Validate the remainder too: a truncated container must never publish.
        whitespace()
        if wrapped:
            while buffer.startswith(","):
                buffer = buffer[1:]
                whitespace()
                decode()
                whitespace()
                if not buffer.startswith(":"):
                    raise ValueError(f"Invalid annotation trailer: {path}")
                buffer = buffer[1:]
                whitespace()
                decode()
                whitespace()
            if not buffer.startswith("}"):
                raise ValueError(f"Truncated annotation object: {path}")
            buffer = buffer[1:]
        whitespace()
        if buffer:
            raise ValueError(f"Unexpected trailing JSON: {path}")


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    point = [finite(item) for item in value]
    return point if all(item is not None for item in point) else None


def load_connectivity(root: Path | None, scene: str) -> dict[str, list[float]]:
    if root is None:
        return {}
    # scene identifiers must not escape the explicitly configured directory.
    if not re.fullmatch(r"[\w-]+", scene):
        return {}
    path = root / f"{scene}_connectivity.json"
    if not path.is_file():
        return {}
    result = {}
    for row in json.loads(path.read_text()):
        pose = row.get("pose", [])
        if row.get("included") and len(pose) == 16:
            point = xyz([pose[3], pose[7], pose[11]])
            if point is not None:
                result[str(row["image_id"])] = point
    return result


def normalize(row: dict[str, Any], source: dict[str, Any], split: str, role: str,
              filename: Path, ordinal: int, connectivity: dict[str, list[float]]) -> Iterator[dict[str, Any]]:
    fmt = source.get("format", "vlnce")
    raw_instruction = row.get("instruction", {})
    instruction_meta = raw_instruction if isinstance(raw_instruction, dict) else {}
    scene_path = str(row.get("scene_id") or row.get("scan") or "")
    scene = Path(scene_path).stem or "unknown"
    if not any(row.get(field) is not None for field in ("episode_id", "instruction_id", "path_id")):
        raise ValueError(f"Missing stable episode/instruction/path ID: {filename}, row {ordinal}")
    episode_id = str(row.get("episode_id", row.get("instruction_id", row.get("path_id", ordinal))))
    trajectory_id = str(row.get("trajectory_id", row.get("path_id", episode_id)))
    role = str((row.get("info") or {}).get("role") or row.get("role") or role)
    language = str(instruction_meta.get("language") or row.get("language") or
                   ("en" if source["dataset"] == "R2R" else "unknown"))
    viewpoints = row.get("path", []) if fmt != "vlnce" else []
    viewpoints = [str(item) for item in viewpoints] if isinstance(viewpoints, list) else []
    raw_points = row.get("reference_path", []) if fmt == "vlnce" else [connectivity.get(v) for v in viewpoints]
    points = [xyz(item) for item in raw_points] if isinstance(raw_points, list) else []
    complete_path = bool(points) and all(point is not None for point in points)
    points = points if complete_path else []  # Never bridge missing viewpoints.
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:])) if len(points) >= 2 else None
    goals = [point for goal in (row.get("goals") or []) if (point := xyz(goal.get("position"))) is not None]
    start = xyz(row.get("start_position")) or (points[0] if points else None)
    if fmt != "vlnce" and points:
        goals = [points[-1]]
    distance = finite((row.get("info") or {}).get("geodesic_distance"))
    if distance is not None and distance < 0:
        distance = None
    texts = row.get("instructions") if fmt == "r2r" else None
    if not isinstance(texts, list):
        texts = [instruction_meta.get("instruction_text", raw_instruction if isinstance(raw_instruction, str) else "")]
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Non-text instruction in {filename}, row {ordinal}")
        instruction_id = str(instruction_meta.get("instruction_id", row.get("instruction_id", f"{episode_id}_{index}")))
        issues = []
        if not text.strip():
            issues.append("missing_instruction")
        if not points:
            issues.append("missing_geometry")
        if not goals:
            issues.append("missing_goal")
        if scene == "unknown":
            issues.append("missing_scene")
        key = stable_id(source["id"], split, role, scene, episode_id, instruction_id)
        yield {
            "id": key, "source_id": source["id"], "dataset": source["dataset"],
            "version": source.get("version", "unspecified"), "format": fmt,
            "split": split, "scene": scene, "scene_path": scene_path,
            "language": language, "role": role, "episode_id": episode_id,
            "instruction_id": instruction_id, "trajectory_id": trajectory_id,
            "route_id": stable_id(source["id"], split, role, scene, trajectory_id),
            "instruction": text, "word_count": len(text.split()),
            "path_length": round(length, 3) if length is not None else None,
            "geodesic_distance": distance, "point_count": len(points),
            "reference_path": points, "viewpoints": viewpoints,
            "projection_axes": [0, 2] if fmt == "vlnce" else [0, 1],
            "start_position": start, "start_rotation": json_safe(row.get("start_rotation")),
            "heading": finite(row.get("heading")), "goals": goals,
            "timed_instruction": json_safe(instruction_meta.get("timed_instruction", row.get("timed_instruction", []))),
            "annotator_id": instruction_meta.get("annotator_id", row.get("annotator_id")),
            "source_path": str(filename), "source_row": ordinal,
            "has_media": 0, "media": [], "issues": issues,
        }


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def import_reports(config: dict[str, Any], base: Path) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    reports: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for spec in config.get("media_catalogs", []):
        path = resolve_path(spec["catalog"], base)
        root = resolve_path(spec["root"], base)
        catalog = json.loads(path.read_text(encoding="utf-8"))

        def asset(url: Any) -> str | None:
            if not isinstance(url, str):
                return None
            file = (root / url.lstrip("/")).resolve()
            if file.is_relative_to(root) and file.is_file() and file.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg", ".mp4"}:
                return str(file)
            return None

        for report in catalog.get("episodes", []):
            meta = report["meta"]
            # Require an explicit source and split to avoid cross-dataset joins.
            key = (spec["source_id"], str(meta["split"]), str(meta["sceneId"]), str(meta["episodeId"]))
            frames = [{"path": file, "step": event.get("step", i), "action": event.get("action")}
                      for i, event in enumerate(report.get("events", []))
                      if (file := asset(event.get("overlay")))]
            media = {"label": spec.get("label", "已有模型运行"), "run": meta.get("run", {}),
                     "instruction": meta.get("instruction", ""), "frames": frames,
                     "map": asset(report.get("assets", {}).get("finalTrajectory")),
                     "video": asset(report.get("assets", {}).get("video")),
                     "result": report.get("result", {})}
            if frames or media["map"] or media["video"]:
                reports.setdefault(key, []).append(media)
    return reports


def build_database(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    if not sources or len({s["id"] for s in sources}) != len(sources):
        raise ValueError("Configure at least one source; source IDs must be unique")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_index = import_reports(config, config_path.parent)
    fd, temporary = tempfile.mkstemp(prefix=".vln-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    connection = sqlite3.connect(temporary)
    counts = Counter()
    inputs = []
    try:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, source_id TEXT, dataset TEXT, version TEXT, format TEXT,
                split TEXT, scene TEXT, language TEXT, role TEXT, episode_id TEXT,
                instruction_id TEXT, trajectory_id TEXT, route_id TEXT,
                instruction TEXT, word_count INTEGER, path_length REAL, geodesic_distance REAL,
                point_count INTEGER, has_media INTEGER, issues TEXT, detail TEXT NOT NULL
            );
            CREATE TABLE assets (id TEXT PRIMARY KEY, path TEXT NOT NULL);
        """)
        for source in sources:
            fmt = source.get("format", "vlnce")
            if fmt not in {"vlnce", "r2r", "rxr"}:
                raise ValueError(f"Unsupported source format: {fmt}")
            root = resolve_path(source["root"], config_path.parent)
            graph_root = resolve_path(source["connectivity"], config_path.parent) if source.get("connectivity") else None
            graphs: dict[str, dict[str, list[float]]] = {}
            for split in source.get("splits", SPLITS):
                for role in source.get("roles", ["guide"]):
                    pattern = source.get("pattern", "{split}/{split}.json.gz")
                    path = (root / pattern.format(split=split, role=role)).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError(f"Configured annotation is missing or outside its root: {path}")
                    print(f"Indexing {source['dataset']} {split} {role}: {path}", flush=True)
                    count = 0
                    for ordinal, row in enumerate(read_rows(path)):
                        scene = Path(str(row.get("scan") or row.get("scene_id") or "")).stem
                        if fmt != "vlnce" and scene not in graphs:
                            graphs[scene] = load_connectivity(graph_root, scene)
                        for record in normalize(row, source, split, role, path, ordinal, graphs.get(scene, {})):
                            candidates = report_index.get((source["id"], split, record["scene"], record["episode_id"]), [])
                            # A matching episode number is insufficient when a source was reprocessed.
                            record["media"] = json.loads(json_text([
                                m for m in candidates if m["instruction"].strip() == record["instruction"].strip()
                            ]))
                            for media in record["media"]:
                                for item, field in [(media, "map"), (media, "video")] + [(f, "path") for f in media["frames"]]:
                                    file = item.get(field)
                                    if file:
                                        asset_id = stable_id(file)
                                        connection.execute("INSERT OR IGNORE INTO assets VALUES (?, ?)", (asset_id, file))
                                        item[field] = asset_id
                            record["has_media"] = int(bool(record["media"]))
                            values = [json_text(record[key]) if key == "issues" else record[key]
                                      for key in SUMMARY_COLUMNS.split(",")]
                            connection.execute(f"INSERT INTO episodes VALUES ({','.join('?' for _ in range(21))})",
                                               [*values, json_text(record)])
                            count += 1
                    counts[source["dataset"]] += count
                    inputs.append({"source_id": source["id"], "dataset": source["dataset"],
                                   "split": split, "role": role, "path": str(path), "records": count})
                    connection.commit()
        if not sum(counts.values()):
            raise ValueError("No VLN records were found; previous catalogue was preserved")
        connection.executescript("""
            CREATE INDEX idx_episodes_dataset_split_scene ON episodes(dataset,split,scene);
            CREATE INDEX idx_episodes_language ON episodes(language);
            CREATE INDEX idx_episodes_source ON episodes(source_id);
            CREATE INDEX idx_episodes_media ON episodes(has_media) WHERE has_media=1;
            CREATE INDEX idx_episodes_route ON episodes(route_id);
            CREATE INDEX idx_episodes_overview ON episodes(dataset,split,scene,role,language,source_id,has_media,path_length,route_id);
            PRAGMA optimize;
        """)
        metadata = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
                    "catalog_id": stable_id(sorted(source["id"] for source in sources)),
                    "sources": sources, "inputs": inputs, "counts": dict(counts)}
        connection.execute("INSERT INTO metadata VALUES ('catalog', ?)", (json_text(metadata),))
        connection.commit()
        connection.close()
        os.replace(temporary, output)
        return metadata
    except BaseException:
        connection.close()
        Path(temporary).unlink(missing_ok=True)
        raise


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def where_clause(params: dict[str, str], media_column="has_media") -> tuple[str, list[Any]]:
    conditions, values = [], []
    for field in FILTERS:
        if params.get(field):
            conditions.append(f"{field} = ?")
            values.append(params[field])
    if params.get("q", "").strip():
        query = params["q"].strip()
        if len(query) > 500:
            raise ValueError("搜索内容最多 500 个字符")
        query = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        conditions.append("(instruction LIKE ? ESCAPE '\\' OR episode_id LIKE ? ESCAPE '\\' OR trajectory_id LIKE ? ESCAPE '\\' OR scene LIKE ? ESCAPE '\\')")
        values.extend([query] * 4)
    for parameter, operator in (("min_length", ">="), ("max_length", "<=")):
        if params.get(parameter):
            number = finite(params[parameter])
            if number is None or number < 0:
                raise ValueError("路径长度必须是非负数")
            conditions.append(f"path_length {operator} ?")
            values.append(number)
    if params.get("min_length") and params.get("max_length") and float(params["min_length"]) > float(params["max_length"]):
        raise ValueError("最短路径长度不能大于最长路径长度")
    if params.get("media") == "yes":
        conditions.append(f"({media_column})=1")
    elif params.get("media") == "no":
        conditions.append(f"({media_column})=0")
    if params.get("quality") == "issues":
        conditions.append("issues != '[]'")
    elif params.get("quality") == "complete":
        conditions.append("issues = '[]'")
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), values


def summary_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["issues"] = json.loads(record["issues"])
    return record


def overview(connection: sqlite3.Connection, params: dict[str, str], media_column="has_media") -> dict[str, Any]:
    clause, values = where_clause(params, media_column)
    stats = dict(connection.execute("SELECT count(*) records, count(DISTINCT route_id) routes, "
                                    f"count(DISTINCT scene) scenes, coalesce(sum({media_column}),0) media, "
                                    "avg(path_length) mean_path_length FROM episodes" + clause, values).fetchone())
    facets = {}
    for field in FILTERS:
        scope = {key: value for key, value in params.items() if key != field}
        field_clause, field_values = where_clause(scope, media_column)
        facets[field] = [dict(row) for row in connection.execute(
            f"SELECT {field} value, count(*) count FROM episodes{field_clause} GROUP BY {field} ORDER BY {field}", field_values)]
    return {"stats": stats, "facets": facets}
