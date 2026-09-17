"""Build a bounded task-to-video index from formal LeRobot outputs.

The index contains only metadata-derived episode and camera locators.  It never
copies media and it never accepts a path from a browser client.  Dataset-specific
root discovery is deliberately shallow and restricted to the two conversion
roots supplied by the operator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

VIDEO_INDEX_SCHEMA_VERSION = 1
VIDEO_STATUSES = ("available", "linked", "partial", "absent", "unmapped")
VIDEO_STATUS_LABELS = {
    "available": "可直接播放",
    "linked": "符号链接媒体",
    "partial": "部分视频可用",
    "absent": "转换元数据存在但视频缺失",
    "unmapped": "尚未映射到任务词条",
}
NO_STANDALONE_VIDEO_NOTES: dict[str, str] = {}
DEFAULT_MAX_EPISODES_PER_RECORD = 200
DEFAULT_MAX_HIFI_EPISODES_PER_RECORD = 10_000
DEFAULT_MAX_LEROBOT_ROOTS = 100_000
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSONL_BYTES = 512 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_FORMAT_SPEC_RE = re.compile(r"0?\d*d?")
_ALLOWED_TEMPLATE_FIELDS = {"episode_chunk", "video_key", "episode_index"}
_ALLOWED_SEGMENT_TEMPLATE_FIELDS = {"video_key", "chunk_index", "file_index"}


@dataclass(frozen=True)
class LeRobotRoot:
    """One trusted LeRobot repository discovered beneath a formal output root."""

    path: Path
    dataset: str
    native_group: str | None = None
    native_task: str | None = None
    record_ids: tuple[str, ...] = ()
    display_episode_index: int | None = None
    validated_media: bool = False


@dataclass(frozen=True)
class DiscoveryResult:
    """Shallow discovery result plus its global safety-cap status."""

    roots: tuple[LeRobotRoot, ...]
    truncated: bool


def normalize_task(value: str) -> str:
    """Normalize task labels without collapsing non-Latin instructions."""

    # ``[^a-z0-9]`` used to collapse every Chinese-only instruction to the
    # same empty key.  Keep letters and digits from every Unicode script while
    # still treating punctuation and path separators as equivalent spacing.
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def _is_visible_directory(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_dir()
    except OSError:
        return False


def _is_regular_metadata_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file()
    except OSError:
        return False


def _sorted_directories(path: Path) -> list[Path]:
    if not _is_visible_directory(path):
        return []
    try:
        return sorted((item for item in path.iterdir() if _is_visible_directory(item)), key=lambda item: item.name)
    except OSError:
        return []


def _is_lerobot_root(path: Path) -> bool:
    return _is_visible_directory(path) and _is_regular_metadata_file(path / "meta" / "info.json")


def discover_lerobot_roots(
    *,
    pfs_root: Path | None,
    cfs_root: Path | None,
    max_roots: int = DEFAULT_MAX_LEROBOT_ROOTS,
    datasets: set[str] | None = None,
) -> DiscoveryResult:
    """Discover known formal layouts without recursively walking arbitrary paths."""

    if max_roots <= 0:
        raise ValueError("max_roots must be positive")

    roots: list[LeRobotRoot] = []
    seen: set[Path] = set()
    truncated = False

    def add(
        path: Path,
        dataset: str,
        native_group: str | None = None,
        native_task: str | None = None,
        *,
        validated_media: bool = False,
    ) -> bool:
        nonlocal truncated
        if not _is_lerobot_root(path):
            return True
        resolved = path.resolve()
        if resolved in seen:
            return True
        if len(roots) >= max_roots:
            truncated = True
            return False
        seen.add(resolved)
        roots.append(
            LeRobotRoot(
                path=resolved,
                dataset=dataset,
                native_group=native_group,
                native_task=native_task,
                validated_media=validated_media,
            )
        )
        return True

    def add_self_or_children(base: Path, dataset: str) -> bool:
        if _is_lerobot_root(base):
            return add(base, dataset)
        for child in _sorted_directories(base):
            if not add(child, dataset):
                return False
        return True

    enabled = datasets or {
        "RoboMIND",
        "RoboMINDv2",
        "FastUMI-Data",
        "FastUMI-100K",
        "EgoDex",
        "FreeTacMan",
        "Touch-in-the-Wild",
        "exUMI",
        "Galaxea-Open-World",
        "Hy-Embodied-0.5-VLA-Data",
        "LET-claw",
        "RT-1",
        "RoboCOIN",
        "Something-Something-V2",
        "HiFi-UMI-2K",
    }

    if pfs_root:
        pfs = pfs_root.resolve()
        if "RoboMIND" in enabled and not add_self_or_children(pfs / "RoboMIND-1.0-full-clean", "RoboMIND"):
            return DiscoveryResult(tuple(roots), True)
        if "RoboMINDv2" in enabled and not add_self_or_children(pfs / "RoboMIND-2.0-full", "RoboMINDv2"):
            return DiscoveryResult(tuple(roots), True)
        if "FastUMI-Data" in enabled and not add(pfs / "FastUMI-Data-80D-full-compact", "FastUMI-Data"):
            return DiscoveryResult(tuple(roots), True)
        if "FastUMI-100K" in enabled:
            fastumi_root = pfs / "FastUMI-100K-80D-full"
            for group in _sorted_directories(fastumi_root):
                for task in _sorted_directories(group):
                    if not add(task, "FastUMI-100K", group.name, task.name):
                        return DiscoveryResult(tuple(roots), True)
        if "EgoDex" in enabled and not add(pfs / "EgoDex", "EgoDex"):
            return DiscoveryResult(tuple(roots), True)
        if "Galaxea-Open-World" in enabled:
            for task_root in _sorted_directories(pfs / "Galaxea-Open-World"):
                if not add(task_root, "Galaxea-Open-World", validated_media=True):
                    return DiscoveryResult(tuple(roots), True)
        if "Hy-Embodied-0.5-VLA-Data" in enabled:
            for table_root in _sorted_directories(pfs / "Hy-Embodied-0.5-VLA-Data-80D-starvla-v3"):
                if not add(table_root, "Hy-Embodied-0.5-VLA-Data", validated_media=True):
                    return DiscoveryResult(tuple(roots), True)
        for dataset, output_name in (
            ("LET-claw", "LET-claw"),
            ("RT-1", "RT-1"),
            ("Something-Something-V2", "Something-Something-V2"),
        ):
            if dataset in enabled and not add(pfs / output_name, dataset, validated_media=True):
                return DiscoveryResult(tuple(roots), True)
        if "RoboCOIN" in enabled:
            for task_root in _sorted_directories(pfs / "RoboCOIN"):
                if not add(task_root, "RoboCOIN", validated_media=True):
                    return DiscoveryResult(tuple(roots), True)
        if "HiFi-UMI-2K" in enabled:
            hifi_root = pfs / "HiFi-UMI-2K-80D-full"
            for chunk_root in _sorted_directories(hifi_root):
                for part_root in _sorted_directories(chunk_root):
                    if not add(part_root, "HiFi-UMI-2K", validated_media=True):
                        return DiscoveryResult(tuple(roots), True)

    if cfs_root:
        cfs = cfs_root.resolve()
        if "FreeTacMan" in enabled:
            freetac = cfs / "FreeTacMan-80D-formal-v1"
            for name in ("2cam", "3cam"):
                if not add(freetac / name, "FreeTacMan"):
                    return DiscoveryResult(tuple(roots), True)
        if "Touch-in-the-Wild" in enabled and not add(
            cfs / "Touch-in-the-Wild-80D-clean-v4" / "processed_zarr", "Touch-in-the-Wild"
        ):
            return DiscoveryResult(tuple(roots), True)
        if "exUMI" in enabled:
            exumi = cfs / "exUMI-80D-formal-v4"
            for name in ("camera_only", "with_tactile"):
                if not add(exumi / name, "exUMI"):
                    return DiscoveryResult(tuple(roots), True)
    roots.sort(key=lambda item: (item.dataset, item.path.as_posix()))
    return DiscoveryResult(tuple(roots), truncated)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        if not _is_regular_metadata_file(path) or path.stat().st_size > _MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not _is_regular_metadata_file(path):
        return
    try:
        if path.stat().st_size > _MAX_JSONL_BYTES:
            return
        with path.open("r", encoding="utf-8") as handle:
            while True:
                line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
                if not line:
                    return
                if len(line) > _MAX_JSONL_LINE_BYTES:
                    return
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    return
                if isinstance(payload, dict):
                    yield payload
    except (OSError, UnicodeError):
        return


def validate_video_path_template(template: str) -> str:
    """Validate the small formatting language accepted from LeRobot metadata."""

    if not isinstance(template, str) or not template or "\\" in template or "\x00" in template:
        raise ValueError("invalid video_path template")
    formatter = string.Formatter()
    fields: set[str] = set()
    try:
        for _, field_name, format_spec, conversion in formatter.parse(template):
            if field_name is None:
                continue
            if field_name not in _ALLOWED_TEMPLATE_FIELDS or conversion is not None:
                raise ValueError("unsupported video_path template field")
            if not _FORMAT_SPEC_RE.fullmatch(format_spec):
                raise ValueError("unsupported video_path template format")
            fields.add(field_name)
    except ValueError as exc:
        raise ValueError("invalid video_path template") from exc
    if not {"episode_index", "video_key"}.issubset(fields):
        raise ValueError("video_path template must contain episode_index and video_key")
    rendered = template.format(episode_chunk=0, episode_index=0, video_key="camera")
    relative = PurePosixPath(rendered)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("video_path template escapes the LeRobot root")
    return template


def format_video_relative_path(
    template: str,
    *,
    chunks_size: int,
    episode_index: int,
    camera_key: str,
) -> PurePosixPath:
    """Render a trusted relative media path and recheck path containment syntax."""

    validate_video_path_template(template)
    if not isinstance(chunks_size, int) or isinstance(chunks_size, bool) or chunks_size <= 0:
        raise ValueError("chunks_size must be a positive integer")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    if not isinstance(camera_key, str) or not camera_key or "/" in camera_key or "\\" in camera_key:
        raise ValueError("camera_key must be one path segment")
    relative = PurePosixPath(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=camera_key,
        )
    )
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("rendered video path escapes the LeRobot root")
    return relative


def format_segment_video_relative_path(
    template: str,
    *,
    chunk_index: int,
    file_index: int,
    camera_key: str,
) -> PurePosixPath:
    """Render the shared-video locator used by LeRobot v3 episode metadata."""

    if not isinstance(template, str) or not template or "\\" in template or "\x00" in template:
        raise ValueError("invalid segmented video_path template")
    formatter = string.Formatter()
    fields: set[str] = set()
    try:
        for _, field_name, format_spec, conversion in formatter.parse(template):
            if field_name is None:
                continue
            if field_name not in _ALLOWED_SEGMENT_TEMPLATE_FIELDS or conversion is not None:
                raise ValueError("unsupported segmented video_path template field")
            if not _FORMAT_SPEC_RE.fullmatch(format_spec):
                raise ValueError("unsupported segmented video_path template format")
            fields.add(field_name)
    except ValueError as exc:
        raise ValueError("invalid segmented video_path template") from exc
    if not {"video_key", "chunk_index", "file_index"}.issubset(fields):
        raise ValueError("segmented video_path template is missing a locator field")
    for value, label in ((chunk_index, "chunk_index"), (file_index, "file_index")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if not isinstance(camera_key, str) or not camera_key or "/" in camera_key or "\\" in camera_key:
        raise ValueError("camera_key must be one path segment")
    relative = PurePosixPath(
        template.format(video_key=camera_key, chunk_index=chunk_index, file_index=file_index)
    )
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("rendered segmented video path escapes the LeRobot root")
    return relative


def _positive_integer(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _video_camera_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    cameras: list[str] = []
    for key, value in features.items():
        if not isinstance(key, str) or not isinstance(value, dict) or value.get("dtype") != "video":
            continue
        if not key or "/" in key or "\\" in key or key in {".", ".."}:
            continue
        cameras.append(key)
    return sorted(set(cameras))


def _task_metadata(root: Path) -> tuple[dict[int, str], list[str]]:
    by_index: dict[int, str] = {}
    tasks: list[str] = []
    for payload in _iter_jsonl(root / "meta" / "tasks.jsonl"):
        task = payload.get("task")
        if not isinstance(task, str) or not task:
            continue
        tasks.append(task)
        task_index = payload.get("task_index")
        if isinstance(task_index, int) and not isinstance(task_index, bool) and task_index >= 0:
            by_index[task_index] = task
    return by_index, tasks


def _episode_tasks(payload: dict[str, Any], tasks_by_index: dict[int, str], root_tasks: list[str]) -> list[str]:
    tasks: list[str] = []
    direct_task = payload.get("task")
    if isinstance(direct_task, str) and direct_task:
        tasks.append(direct_task)
    family = payload.get("task_family")
    if isinstance(family, str) and family:
        tasks.append(family)
    values = payload.get("tasks")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str) and value:
                tasks.append(value)
            elif isinstance(value, int) and not isinstance(value, bool) and value in tasks_by_index:
                tasks.append(tasks_by_index[value])
    indices = payload.get("task_indices")
    if isinstance(indices, list):
        for value in indices:
            if isinstance(value, int) and not isinstance(value, bool) and value in tasks_by_index:
                tasks.append(tasks_by_index[value])
    if not tasks and len(root_tasks) == 1:
        tasks.append(root_tasks[0])
    return list(dict.fromkeys(tasks))


def _episode_source_values(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("source_hdf5", "source_mcap", "source_path", "embedding_source"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _dataset_relative_path(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    for marker in ("/mnt/cfs/data/datasets/", "/mnt/pfs/data/datasets/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1].lstrip("/")
    return normalized.lstrip("/") if not normalized.startswith("/") else None


def _source_record_ids(
    dataset: str,
    source_values: Iterable[str],
    source_index: dict[tuple[str, str], tuple[str, ...]],
) -> set[str]:
    record_ids: set[str] = set()
    for value in source_values:
        relative = _dataset_relative_path(value)
        if not relative:
            continue
        parts = PurePosixPath(relative).parts
        for length in range(len(parts), 0, -1):
            prefix = PurePosixPath(*parts[:length]).as_posix()
            matches = source_index.get((dataset, prefix))
            if matches:
                record_ids.update(matches)
                break
    return record_ids


def _realomin_state_root(cfs_root: Path) -> Path | None:
    """Locate the trusted completion state beside either a test or production root."""

    name = ".realomin_formal_fragments_v2_state"
    candidates = (cfs_root / name, cfs_root.parent.parent / name)
    for candidate in candidates:
        try:
            if candidate.is_dir() and (candidate / "status").read_text(encoding="utf-8").strip() == "COMPLETE":
                return candidate
        except (OSError, UnicodeError):
            continue
    return None


def _realomin_excluded_indexes(state_root: Path) -> set[int]:
    excluded: set[int] = set()
    for path in (state_root / "source_exclusions").glob("*/*.json"):
        payload = _read_json_object(path)
        index = payload.get("index") if payload else None
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            excluded.add(index)
    return excluded


def _discover_realomin_record_roots(
    *,
    pfs_root: Path | None,
    cfs_root: Path | None,
    source_index: dict[tuple[str, str], tuple[str, ...]],
    max_episodes_per_record: int,
    max_roots: int,
) -> tuple[tuple[LeRobotRoot, ...], dict[str, int]]:
    """Select only catalog-relevant 10Kh fragments from the 667k-entry manifest.

    The manifest carries a stable global index and the exact source MCAP path.
    This avoids walking hundreds of thousands of one-episode repositories while
    retaining strict source-path mapping and bounded playback rows per record.
    """

    if pfs_root is None or cfs_root is None or max_roots <= 0:
        return (), {}
    state_root = _realomin_state_root(cfs_root)
    if state_root is None:
        return (), {}
    manifest = state_root / "valid_sources.jsonl"
    fragments = cfs_root / "10Kh-RealOmin-OpenData.formal-fragments-v2" / "fragments"
    aggregate = _read_json_object(
        pfs_root / "10Kh-RealOmin-OpenData-80D-multiroot-v1" / "meta" / "aggregate_info.json"
    )
    aggregate_episodes = _positive_integer(aggregate.get("episodes"), 0) if aggregate else 0
    aggregate_videos = _positive_integer(aggregate.get("videos"), 0) if aggregate else 0
    if (
        not aggregate
        or aggregate.get("status") != "COMPLETE"
        or aggregate.get("valid") is not True
        or aggregate.get("published") is not True
        or aggregate_videos != aggregate_episodes * 2
        or not _is_regular_metadata_file(manifest)
        or not _is_visible_directory(fragments)
    ):
        return (), {}

    excluded = _realomin_excluded_indexes(state_root)
    total_by_record: dict[str, int] = defaultdict(int)
    selected_by_record: dict[str, int] = defaultdict(int)
    roots: list[LeRobotRoot] = []
    dataset_prefix = "genrobot2025/10Kh-RealOmin-OpenData/"
    for payload in _iter_jsonl(manifest):
        index = payload.get("index")
        source = payload.get("source")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in excluded
            or not isinstance(source, str)
            or not source
        ):
            continue
        record_ids = _source_record_ids("10Kh-RealOmin", (f"{dataset_prefix}{source.lstrip('/')}",), source_index)
        if not record_ids:
            continue
        for record_id in record_ids:
            total_by_record[record_id] += 1
        selected_ids = tuple(
            sorted(record_id for record_id in record_ids if selected_by_record[record_id] < max_episodes_per_record)
        )
        if not selected_ids or len(roots) >= max_roots:
            continue
        for record_id in selected_ids:
            selected_by_record[record_id] += 1
        roots.append(
            LeRobotRoot(
                path=(fragments / f"chunk-{index // 1000:06d}" / f"episode-{index:09d}").resolve(),
                dataset="10Kh-RealOmin",
                record_ids=selected_ids,
                display_episode_index=index,
                validated_media=True,
            )
        )
    return tuple(roots), dict(total_by_record)


def _catalog_digest(records: list[dict[str, Any]]) -> str:
    identity = [
        {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "native_group": record.get("native_group"),
            "native_task": record.get("native_task"),
            "source_path": record.get("source_path"),
        }
        for record in records
    ]
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _iter_hifi_episode_rows(root: Path, cameras: list[str]) -> Iterator[dict[str, Any]]:
    """Read only the v3 episode locator columns needed for shared MP4 playback."""

    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return
    columns = ["episode_index", "tasks"]
    for camera in cameras:
        prefix = f"videos/{camera}"
        columns.extend(
            (
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            )
        )
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        if not _is_regular_metadata_file(path):
            continue
        try:
            parquet_file = parquet.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=4096, columns=columns):
                yield from batch.to_pylist()
        except (OSError, ValueError, TypeError):
            continue


def _add_hifi_v3_mappings(
    *,
    roots: Iterable[LeRobotRoot],
    mutable_records: dict[str, dict[str, Any]],
    task_index: dict[tuple[str, str], list[str]],
    max_episodes_per_record: int,
) -> None:
    """Map segmented v3 episodes stored as time ranges inside shared MP4s."""

    display_episode_index = 0
    for root_spec in roots:
        if root_spec.dataset not in {"HiFi-UMI-2K", "Hy-Embodied-0.5-VLA-Data"}:
            continue
        info = _read_json_object(root_spec.path / "meta" / "info.json")
        if not info:
            continue
        cameras = _video_camera_keys(info)
        template = info.get("video_path")
        if not cameras or not isinstance(template, str):
            continue
        try:
            format_segment_video_relative_path(template, chunk_index=0, file_index=0, camera_key=cameras[0])
        except ValueError:
            continue
        root_key = root_spec.path.as_posix()
        for payload in _iter_hifi_episode_rows(root_spec.path, cameras):
            episode_index = payload.get("episode_index")
            if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
                continue
            current_display_index = display_episode_index
            display_episode_index += 1
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                continue
            task_labels = [task for task in tasks if isinstance(task, str) and task]
            record_ids: set[str] = set()
            for task in task_labels:
                record_ids.update(task_index.get((root_spec.dataset, normalize_task(task)), ()))
            if not record_ids:
                continue

            camera_segments: dict[str, list[int | float]] = {}
            for camera in cameras:
                prefix = f"videos/{camera}"
                chunk_index = payload.get(f"{prefix}/chunk_index")
                file_index = payload.get(f"{prefix}/file_index")
                start_seconds = payload.get(f"{prefix}/from_timestamp")
                end_seconds = payload.get(f"{prefix}/to_timestamp")
                if (
                    not isinstance(chunk_index, int)
                    or isinstance(chunk_index, bool)
                    or chunk_index < 0
                    or not isinstance(file_index, int)
                    or isinstance(file_index, bool)
                    or file_index < 0
                    or not isinstance(start_seconds, (int, float))
                    or isinstance(start_seconds, bool)
                    or not isinstance(end_seconds, (int, float))
                    or isinstance(end_seconds, bool)
                    or float(start_seconds) < 0
                    or float(end_seconds) <= float(start_seconds)
                ):
                    continue
                camera_segments[camera] = [chunk_index, file_index, float(start_seconds), float(end_seconds)]
            if not camera_segments:
                continue

            task_label = task_labels[0] if task_labels else ""
            for record_id in sorted(record_ids):
                target = mutable_records.get(record_id)
                if not target:
                    continue
                target["total_episode_count"] += 1
                target["total_video_count"] += len(camera_segments)
                target["mapping_statuses"].add("task_name")
                if target["selected_episode_count"] >= max_episodes_per_record:
                    continue
                target["selected_episode_count"] += 1
                source = target["sources"].setdefault(
                    root_key,
                    {
                        "root": root_key,
                        "video_path_template": template,
                        "chunks_size": _positive_integer(info.get("chunks_size"), 1000),
                        "camera_keys": cameras,
                        "source_codec": "h264",
                        "validated_media": True,
                        "segmented_media": True,
                        "episodes": [],
                    },
                )
                episode_entry: dict[str, Any] = {
                    "episode_index": episode_index,
                    "display_episode_index": current_display_index,
                    "task": task_label,
                }
                segment_values = list(camera_segments.values())
                if segment_values and all(value == segment_values[0] for value in segment_values[1:]):
                    episode_entry["segment"] = segment_values[0]
                else:
                    episode_entry["camera_segments"] = camera_segments
                source["episodes"].append(episode_entry)


def build_video_index(
    catalog: dict[str, Any],
    *,
    pfs_root: Path | None = None,
    cfs_root: Path | None = None,
    generated_at: str | None = None,
    max_episodes_per_record: int = DEFAULT_MAX_EPISODES_PER_RECORD,
    max_hifi_episodes_per_record: int = DEFAULT_MAX_HIFI_EPISODES_PER_RECORD,
    max_lerobot_roots: int = DEFAULT_MAX_LEROBOT_ROOTS,
) -> dict[str, Any]:
    """Build a bounded record-to-episode/camera index from trusted roots."""

    if max_episodes_per_record <= 0:
        raise ValueError("max_episodes_per_record must be positive")
    if max_hifi_episodes_per_record <= 0:
        raise ValueError("max_hifi_episodes_per_record must be positive")
    records = catalog.get("records")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("catalog.records must be a list of objects")

    by_id: dict[str, dict[str, Any]] = {}
    task_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    touch_task_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    source_index_mutable: dict[tuple[str, str], list[str]] = defaultdict(list)
    hint_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for record in records:
        record_id = record.get("id")
        dataset = record.get("dataset")
        native_group = record.get("native_group")
        native_task = record.get("native_task")
        source_path = record.get("source_path")
        if not all(isinstance(value, str) and value for value in (record_id, dataset, native_task, source_path)):
            raise ValueError("every catalog record needs nonempty id, dataset, native_task, and source_path")
        if record_id in by_id:
            raise ValueError(f"duplicate catalog record id: {record_id}")
        by_id[record_id] = record
        task_index[(dataset, normalize_task(native_task))].append(record_id)
        source_index_mutable[(dataset, PurePosixPath(source_path).as_posix())].append(record_id)
        if isinstance(native_group, str) and native_group:
            hint_index[(dataset, native_group, native_task)].append(record_id)
            touch_expected = normalize_task(f"{native_group}/{native_task}/{native_task}")
            touch_task_index[(dataset, touch_expected)].append(record_id)

    source_index = {key: tuple(value) for key, value in source_index_mutable.items()}
    dataset_names = {record["dataset"] for record in records}
    discovery = discover_lerobot_roots(
        pfs_root=pfs_root,
        cfs_root=cfs_root,
        max_roots=max_lerobot_roots,
        datasets=dataset_names,
    )
    remaining_roots = max(0, max_lerobot_roots - len(discovery.roots))
    realomin_roots, realomin_episode_counts = _discover_realomin_record_roots(
        pfs_root=pfs_root,
        cfs_root=cfs_root,
        source_index=source_index,
        max_episodes_per_record=max_episodes_per_record,
        max_roots=remaining_roots,
    )
    discovery = DiscoveryResult(discovery.roots + realomin_roots, discovery.truncated)

    mutable_records: dict[str, dict[str, Any]] = {}
    for record_id, record in by_id.items():
        realomin_episodes = realomin_episode_counts.get(record_id, 0)
        mutable_records[record_id] = {
            "record_id": record_id,
            "dataset": record["dataset"],
            "native_task": record["native_task"],
            "source_path": record["source_path"],
            "total_episode_count": realomin_episodes,
            "total_video_count": realomin_episodes * 2,
            "selected_episode_count": 0,
            "mapping_statuses": set(),
            "sources": {},
        }

    for root_spec in discovery.roots:
        info = (
            {
                "chunks_size": 1000,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.robot0": {"dtype": "video"},
                    "observation.images.robot1": {"dtype": "video"},
                },
            }
            if root_spec.display_episode_index is not None
            else _read_json_object(root_spec.path / "meta" / "info.json")
        )
        if not info:
            continue
        cameras = _video_camera_keys(info)
        if not cameras:
            continue
        chunks_size = _positive_integer(info.get("chunks_size"), 1000)
        template_value = info.get("video_path")
        template = (
            template_value
            if isinstance(template_value, str)
            else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        try:
            validate_video_path_template(template)
        except ValueError:
            continue
        tasks_by_index, root_tasks = (
            ({}, []) if root_spec.display_episode_index is not None else _task_metadata(root_spec.path)
        )
        hinted_ids = set(
            hint_index.get(
                (root_spec.dataset, root_spec.native_group or "", root_spec.native_task or ""),
                (),
            )
        )
        episodes_path = root_spec.path / "meta" / "episodes.jsonl"
        episode_payloads: Iterable[dict[str, Any]] = (
            ({"episode_index": 0},) if root_spec.display_episode_index is not None else _iter_jsonl(episodes_path)
        )
        if not _is_regular_metadata_file(episodes_path) and hinted_ids:
            total_episodes = _positive_integer(info.get("total_episodes"), 0)
            episode_payloads = ({"episode_index": index} for index in range(total_episodes))

        for payload in episode_payloads:
            episode_index = payload.get("episode_index")
            if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
                continue
            tasks = _episode_tasks(payload, tasks_by_index, root_tasks)
            record_ids = set(root_spec.record_ids) or _source_record_ids(
                root_spec.dataset, _episode_source_values(payload), source_index
            )
            mapping_status = "source_path"
            if not record_ids and hinted_ids:
                record_ids = set(hinted_ids)
                mapping_status = "exact_output_root"
            if not record_ids:
                # RoboMINDv2 has source_hdf5 evidence for almost all episodes.
                # Accept its task-name fallback only when it identifies one
                # catalog row; ambiguous duplicate names remain unmapped.
                task_candidates: set[str] = set()
                for task in tasks:
                    task_candidates.update(task_index.get((root_spec.dataset, normalize_task(task)), ()))
                    if root_spec.dataset == "Touch-in-the-Wild":
                        task_candidates.update(touch_task_index.get((root_spec.dataset, normalize_task(task)), ()))
                if root_spec.dataset != "RoboMINDv2" or len(task_candidates) == 1:
                    record_ids.update(task_candidates)
                mapping_status = "task_name"
            if not record_ids:
                continue

            task_label = tasks[0] if tasks else str(root_spec.native_task or "")
            root_key = root_spec.path.as_posix()
            for record_id in sorted(record_ids):
                target = mutable_records.get(record_id)
                if not target:
                    continue
                if not root_spec.record_ids:
                    target["total_episode_count"] += 1
                    target["total_video_count"] += len(cameras)
                target["mapping_statuses"].add(mapping_status)
                if target["selected_episode_count"] >= max_episodes_per_record:
                    continue
                target["selected_episode_count"] += 1
                source = target["sources"].setdefault(
                    root_key,
                    {
                        "root": root_key,
                        "video_path_template": template,
                        "chunks_size": chunks_size,
                        "camera_keys": cameras,
                        "source_codec": "h264" if root_spec.validated_media else None,
                        "episodes": [],
                    },
                )
                if root_spec.validated_media:
                    source["validated_media"] = True
                episode_entry = {"episode_index": episode_index, "task": task_label}
                if root_spec.display_episode_index is not None:
                    episode_entry["display_episode_index"] = root_spec.display_episode_index
                source["episodes"].append(episode_entry)

    _add_hifi_v3_mappings(
        roots=discovery.roots,
        mutable_records=mutable_records,
        task_index=task_index,
        # HiFi stores lightweight time-range locators into shared MP4s, so it
        # can safely retain far more episodes than one-file-per-camera outputs.
        max_episodes_per_record=max_hifi_episodes_per_record,
    )

    indexed_records: dict[str, dict[str, Any]] = {}
    discovered_datasets = {root.dataset for root in discovery.roots}
    media_directory_cache: dict[Path, dict[str, bool]] = {}

    def media_directory_entries(path: Path) -> dict[str, bool]:
        cached = media_directory_cache.get(path)
        if cached is not None:
            return cached
        entries: dict[str, bool] = {}
        try:
            with os.scandir(path) as iterator:
                for entry in iterator:
                    try:
                        if entry.is_file(follow_symlinks=True):
                            entries[entry.name] = entry.is_symlink()
                    except OSError:
                        continue
        except OSError:
            pass
        media_directory_cache[path] = entries
        return entries

    for record_id in sorted(mutable_records):
        target = mutable_records[record_id]
        sources = sorted(target["sources"].values(), key=lambda source: source["root"])
        existing_video_count = 0
        linked_video_count = 0
        for source in sources:
            source["episodes"].sort(key=lambda episode: episode["episode_index"])
            expected_videos = len(source["episodes"]) * len(source["camera_keys"])
            if source.get("validated_media") is True:
                source_existing = expected_videos
                source_linked = 0
            else:
                source_existing = 0
                source_linked = 0
                root = Path(source["root"])
                for episode in source["episodes"]:
                    for camera in source["camera_keys"]:
                        relative = format_video_relative_path(
                            source["video_path_template"],
                            chunks_size=source["chunks_size"],
                            episode_index=episode["episode_index"],
                            camera_key=camera,
                        )
                        candidate = root.joinpath(*relative.parts)
                        entries = media_directory_entries(candidate.parent)
                        exists = candidate.name in entries
                        linked = entries.get(candidate.name, False)
                        if exists:
                            source_existing += 1
                            source_linked += int(linked)
            source["existing_video_count"] = source_existing
            source["linked_video_count"] = source_linked
            if source_existing == 0:
                source["media_status"] = "absent"
            elif source_linked == source_existing:
                source["media_status"] = "linked"
            elif source_existing < expected_videos:
                source["media_status"] = "partial"
            else:
                source["media_status"] = "available"
            existing_video_count += source_existing
            linked_video_count += source_linked
        indexed_episode_count = sum(len(source["episodes"]) for source in sources)
        indexed_video_count = sum(len(source["episodes"]) * len(source["camera_keys"]) for source in sources)
        truncated = discovery.truncated or target["total_episode_count"] > indexed_episode_count
        missing_video_count = max(0, indexed_video_count - existing_video_count)
        video_note = NO_STANDALONE_VIDEO_NOTES.get(target["dataset"])
        if target["total_episode_count"] == 0:
            video_status = (
                "absent"
                if video_note or target["dataset"] not in discovered_datasets
                else "unmapped"
            )
        elif existing_video_count == 0:
            video_status = "absent"
        elif (
            truncated
            or missing_video_count
            or (target["dataset"] == "RoboMIND" and "task_name" in target["mapping_statuses"])
        ):
            video_status = "partial"
        elif linked_video_count == existing_video_count:
            video_status = "linked"
        else:
            video_status = "available"
        indexed_records[record_id] = {
            "record_id": record_id,
            "dataset": target["dataset"],
            "native_task": target["native_task"],
            "status": video_status,
            "video_status": video_status,
            "total_episode_count": target["total_episode_count"],
            "indexed_episode_count": indexed_episode_count,
            "total_video_count": target["total_video_count"],
            "indexed_video_count": indexed_video_count,
            "existing_video_count": existing_video_count,
            "linked_video_count": linked_video_count,
            "missing_video_count": missing_video_count,
            "camera_keys": sorted({camera for source in sources for camera in source["camera_keys"]}),
            "mapping_statuses": sorted(target["mapping_statuses"]),
            "truncated": truncated,
            "video_note": video_note,
            "sources": sources,
        }

    allowed_roots = [root.resolve().as_posix() for root in (pfs_root, cfs_root) if root is not None]
    status_counts = {
        status: sum(record["video_status"] == status for record in indexed_records.values()) for status in VIDEO_STATUSES
    }
    available_records = sum(status_counts[status] for status in ("available", "linked", "partial"))
    effective_generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": VIDEO_INDEX_SCHEMA_VERSION,
        "generated_at": effective_generated_at,
        "catalog_digest": _catalog_digest(records),
        "allowed_conversion_roots": allowed_roots,
        "video_status_labels": VIDEO_STATUS_LABELS,
        "limits": {
            "max_episodes_per_record": max_episodes_per_record,
            "max_hifi_episodes_per_record": max_hifi_episodes_per_record,
            "max_lerobot_roots": max_lerobot_roots,
        },
        "discovery": {
            "lerobot_roots": len(discovery.roots),
            "truncated": discovery.truncated,
        },
        "stats": {
            "records": len(indexed_records),
            "available_records": available_records,
            "by_video_status": status_counts,
            "indexed_episodes": sum(record["indexed_episode_count"] for record in indexed_records.values()),
            "indexed_videos": sum(record["indexed_video_count"] for record in indexed_records.values()),
        },
        "records": indexed_records,
    }


def write_video_index(index: dict[str, Any], output: Path) -> None:
    """Atomically write a JSON video index."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
