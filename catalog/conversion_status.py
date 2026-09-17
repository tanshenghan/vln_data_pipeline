"""Attach structured LeRobot conversion evidence to catalog records.

The native dataset scanner and the conversion-output scanner deliberately stay
separate.  Native records can therefore be rebuilt on machines that do not mount
the LeRobot output roots; in that case every record is reported as ``unknown``
instead of being incorrectly labelled as not converted.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

CONVERSION_STATUSES = ("converted", "partial", "not_converted", "excluded", "unknown")
CONVERSION_STATUS_LABELS = {
    "converted": "已转换",
    "partial": "部分转换",
    "not_converted": "未转换",
    "excluded": "已排除",
    "unknown": "状态未知",
}
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_JSONL_BYTES = 128 * 1024 * 1024

# These datasets are represented in the instruction inventory, but their
# converted outputs are either one LeRobot root or a collection of independent
# LeRobot roots. Without a dataset-level completion report we intentionally
# expose the evidence as ``partial`` rather than claiming full conversion.
GENERIC_PARTIAL_OUTPUTS = {
    "Hy-Embodied-0.5-VLA-Data": "Hy-Embodied-0.5-VLA-Data-80D-starvla-v3",
    "Something-Something-V2": "Something-Something-V2",
    "Galaxea-Open-World": "Galaxea-Open-World",
    "LET-claw": "LET-claw",
    "RT-1": "RT-1",
    "RoboCOIN": "RoboCOIN",
}


def clear_conversion_status_caches() -> None:
    """Clear filesystem-derived caches before a later rescan in the same process."""

    _jsonl_values.cache_clear()
    _fastumi_snapshot_available.cache_clear()
    _hifi_output_task_index.cache_clear()
    _hifi_partition_episode_counts.cache_clear()
    _lerobot_metadata_evidence.cache_clear()
    _galaxea_complete.cache_clear()
    _hy_embodied_complete.cache_clear()
    _let_complete.cache_clear()
    _single_root_marker_complete.cache_clear()
    _something_label_count.cache_clear()


def _read_small_json(path: Path) -> dict[str, Any] | None:
    """Read a bounded JSON object without following report-controlled paths."""

    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_REPORT_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_count(payload: dict[str, Any] | None, *keys: str) -> bool:
    if not payload:
        return False
    return any((_nonnegative_int(payload.get(key)) or 0) > 0 for key in keys)


def _validated_report(path: Path) -> bool:
    payload = _read_small_json(path)
    return bool(payload and payload.get("valid") is True)


def _normalize_task(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@lru_cache(maxsize=64)
def _jsonl_values(path: Path, key: str) -> frozenset[str]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSONL_BYTES:
            return frozenset()
        values: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(line) > _MAX_REPORT_BYTES:
                    return frozenset()
                payload = json.loads(line)
                value = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(value, str) and value:
                    values.add(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return frozenset()
    return frozenset(values)


def _task_matches(path: Path, expected: str, *, key: str = "task") -> bool:
    expected_normalized = _normalize_task(expected)
    return bool(
        expected_normalized
        and expected_normalized in {_normalize_task(value) for value in _jsonl_values(path, key)}
    )


def _done_counts_valid(payload: dict[str, Any]) -> bool:
    episodes = _nonnegative_int(payload.get("episodes"))
    footers = _nonnegative_int(payload.get("parquet_footers"))
    markers = _nonnegative_int(payload.get("done_markers"))
    return episodes is not None and episodes > 0 and footers == episodes and markers == episodes


def _done_report_complete(path: Path, *, require_valid_exclusions: bool = False) -> bool:
    payload = _read_small_json(path)
    if not payload or payload.get("valid") is not True:
        return False
    if require_valid_exclusions:
        exclusions = payload.get("exclusions")
        if not isinstance(exclusions, dict) or exclusions.get("valid") is not True:
            return False
    roots = payload.get("roots")
    if isinstance(roots, list):
        return bool(roots) and all(
            isinstance(root, dict) and root.get("valid") is True and _done_counts_valid(root) for root in roots
        )
    return _done_counts_valid(payload)


def _robomind_reports_complete(output: Path) -> bool:
    cleaning_path = output / "cleaning_report.json"
    summary_path = output / "conversion_summary.json"
    cleaning = _read_small_json(cleaning_path)
    summary = _read_small_json(summary_path)
    if not cleaning or not summary or not _positive_count(cleaning, "total_episodes", "total_frames"):
        return False
    if _nonnegative_int(cleaning.get("unresolved_errors")) != 0:
        return False
    reported_output = cleaning.get("output_root")
    same_formal_output = reported_output == output.as_posix()
    mounted_counterpart = (
        isinstance(reported_output, str)
        and Path(reported_output).name == output.name
        and "/data/datasets/lerobotv3/high_priority_80d/" in reported_output
    )
    if not same_formal_output and not mounted_counterpart:
        return False
    source_root = cleaning.get("source_root")
    if not isinstance(source_root, str) or not source_root.startswith("/mnt/cfs/data/datasets/"):
        return False
    datasets = summary.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return False
    episode_counts = [_nonnegative_int(item.get("episodes")) for item in datasets if isinstance(item, dict)]
    frame_counts = [_nonnegative_int(item.get("frames")) for item in datasets if isinstance(item, dict)]
    if len(episode_counts) != len(datasets) or len(frame_counts) != len(datasets):
        return False
    return sum(episode_counts) == cleaning.get("total_episodes") and sum(frame_counts) == cleaning.get("total_frames")


def _canonical_task_complete(output: Path) -> bool:
    info = _read_small_json(output / "meta" / "info.json")
    episodes = _nonnegative_int(info.get("total_episodes")) if info else None
    if not episodes or not (output / "meta" / "canonical_80d.json").is_file():
        return False
    if not _jsonl_values(output / "meta" / "tasks.jsonl", "task"):
        return False
    marker_root = output / ".canonical_status"
    try:
        if marker_root.is_symlink() or not marker_root.is_dir():
            return False
        marker_count = sum(1 for item in marker_root.iterdir() if item.is_file() and item.suffix == ".done")
    except OSError:
        return False
    return marker_count == episodes


@lru_cache(maxsize=8)
def _fastumi_snapshot_available(output_root: Path) -> bool:
    try:
        if output_root.is_symlink() or not output_root.is_dir():
            return False
        for group in output_root.iterdir():
            if group.is_symlink() or not group.is_dir():
                continue
            for task in group.iterdir():
                if task.is_dir() and not task.is_symlink() and _canonical_task_complete(task):
                    return True
    except OSError:
        return False
    return False


@lru_cache(maxsize=4)
def _hifi_output_task_index(output_root: Path) -> tuple[frozenset[str], Path | None, int, int]:
    """Index task names backed by structurally readable HiFi conversion parts."""

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return frozenset(), None, 0, 0

    tasks: set[str] = set()
    evidence_path: Path | None = None
    valid_parts = 0
    converted_episodes = 0
    try:
        info_paths = sorted(output_root.glob("chunk-*/part-*/meta/info.json"))
        for info_path in info_paths:
            info = _read_small_json(info_path)
            episode_count = _nonnegative_int(info.get("total_episodes")) if info else None
            tasks_path = info_path.parent / "tasks.parquet"
            canonical = info_path.parent / "canonical_80d.json"
            if not episode_count or not canonical.is_file() or tasks_path.is_symlink() or not tasks_path.is_file():
                continue
            try:
                parquet = pq.ParquetFile(tasks_path)
                task_column = "task" if "task" in parquet.schema_arrow.names else None
                if task_column is None:
                    continue
                part_tasks: set[str] = set()
                for batch in parquet.iter_batches(columns=[task_column], batch_size=4096):
                    for value in batch.column(0).to_pylist():
                        if isinstance(value, str) and value.strip():
                            part_tasks.add(_normalize_task(value))
            except (OSError, ValueError):
                continue
            if not part_tasks:
                continue
            tasks.update(part_tasks)
            valid_parts += 1
            converted_episodes += episode_count
            evidence_path = evidence_path or info_path
    except OSError:
        return frozenset(), None, 0, 0
    return frozenset(tasks), evidence_path, valid_parts, converted_episodes


@lru_cache(maxsize=8)
def _hifi_partition_episode_counts(root: Path, require_output_artifacts: bool) -> tuple[tuple[str, int], ...]:
    """Return stable HiFi partition counts, optionally validating LeRobot artifacts."""

    partitions: dict[str, int] = {}
    try:
        for info_path in root.glob("chunk-*/part-*/meta/info.json"):
            if info_path.is_symlink():
                return ()
            info = _read_small_json(info_path)
            episode_count = _nonnegative_int(info.get("total_episodes")) if info else None
            if episode_count is None:
                return ()
            meta_root = info_path.parent
            if require_output_artifacts and not (
                (meta_root / "canonical_80d.json").is_file() and (meta_root / "tasks.parquet").is_file()
            ):
                return ()
            relative = info_path.relative_to(root)
            partition = "/".join(relative.parts[:2])
            if partition in partitions:
                return ()
            partitions[partition] = episode_count
    except (OSError, ValueError):
        return ()
    return tuple(sorted(partitions.items()))


@lru_cache(maxsize=16)
def _lerobot_metadata_evidence(output_root: Path) -> tuple[Path | None, int, int]:
    """Summarize structurally valid top-level or one-level LeRobot outputs."""

    try:
        candidates = list(output_root.glob("meta/info.json"))
        candidates.extend(output_root.glob("*/meta/info.json"))
    except OSError:
        return None, 0, 0

    evidence_path: Path | None = None
    valid_roots = 0
    converted_episodes = 0
    for info_path in candidates:
        info = _read_small_json(info_path)
        episode_count = _nonnegative_int(info.get("total_episodes")) if info else None
        meta_root = info_path.parent
        has_task_metadata = (meta_root / "tasks.jsonl").is_file() or (meta_root / "tasks.parquet").is_file()
        if not episode_count or not has_task_metadata or not (meta_root / "canonical_80d.json").is_file():
            continue
        evidence_path = evidence_path or info_path
        valid_roots += 1
        converted_episodes += episode_count
    return evidence_path, valid_roots, converted_episodes


def _has_lerobot_task_metadata(meta_root: Path) -> bool:
    return (meta_root / "tasks.jsonl").is_file() or (meta_root / "tasks.parquet").is_file()


def _file_count(root: Path, pattern: str) -> int:
    try:
        return sum(1 for path in root.glob(pattern) if path.is_file())
    except OSError:
        return -1


def _direct_file_count(root: Path) -> int:
    try:
        with os.scandir(root) as entries:
            return sum(1 for entry in entries if entry.is_file(follow_symlinks=False))
    except OSError:
        return -1


@lru_cache(maxsize=4)
def _galaxea_complete(dataset_root: Path, output: Path) -> tuple[Path | None, int, int]:
    """Validate the full Galaxea archive set against all converted task roots."""

    source_root = dataset_root / "Galaxea-Open-World-Dataset"
    archive_root = source_root / "lerobot"
    marker = _read_small_json(source_root / ".aihc_download_complete")
    if not marker or marker.get("failure_count") != 0:
        return None, 0, 0
    expected_roots = _nonnegative_int(marker.get("success_count"))
    if not expected_roots:
        return None, 0, 0
    try:
        source_names = {
            path.name.removesuffix(".tar.gz")
            for path in archive_root.iterdir()
            if path.is_file() and path.name.endswith(".tar.gz")
        }
        output_roots = {
            path.name: path
            for path in output.iterdir()
            if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
        }
    except OSError:
        return None, 0, 0
    if len(source_names) != expected_roots or source_names != set(output_roots):
        return None, 0, 0

    total_episodes = 0
    evidence: Path | None = None
    for name in sorted(output_roots):
        task_root = output_roots[name]
        meta_root = task_root / "meta"
        info_path = meta_root / "info.json"
        info = _read_small_json(info_path)
        episodes = _nonnegative_int(info.get("total_episodes")) if info else None
        videos = _nonnegative_int(info.get("total_videos")) if info else None
        if not episodes or videos is None:
            return None, 0, 0
        if not (meta_root / "canonical_80d.json").is_file() or not _has_lerobot_task_metadata(meta_root):
            return None, 0, 0
        if _file_count(task_root / "data", "**/*.parquet") != episodes:
            return None, 0, 0
        if _file_count(task_root / "videos", "**/*.mp4") != videos:
            return None, 0, 0
        evidence = evidence or info_path
        total_episodes += episodes
    return evidence, len(output_roots), total_episodes


@lru_cache(maxsize=4)
def _hy_embodied_complete(dataset_root: Path, output: Path) -> tuple[Path | None, int, int]:
    """Validate the accepted StarVLA v3 conversion and all reported tables."""

    report_path = output / "preparation_report.json"
    report = _read_small_json(report_path)
    manifest = _read_small_json(output / "conversion/manifest.json")
    status = _read_small_json(output / "conversion/status.json")
    official = _read_small_json(output / "conversion/official_reader_acceptance.json")
    starvla = _read_small_json(output / "conversion/starvla_loader_acceptance/summary.json")
    if not all((report, manifest, status, official, starvla)):
        return None, 0, 0
    if (
        report.get("status") != "passed"
        or report.get("stage") != "conversion_and_cpu_loader_acceptance"
        or report.get("partial_scope") is not False
        or report.get("official_and_starvla_loader_acceptance") != "passed"
        or official.get("status") != "passed"
        or starvla.get("status") != "passed"
        or manifest.get("partial_scope") is not False
    ):
        return None, 0, 0

    table_names = manifest.get("tables")
    official_tables = official.get("tables")
    starvla_tables = starvla.get("tables")
    if (
        not isinstance(table_names, list)
        or not table_names
        or not all(isinstance(name, str) and name for name in table_names)
        or len(set(table_names)) != len(table_names)
        or not isinstance(official_tables, list)
        or not isinstance(starvla_tables, list)
    ):
        return None, 0, 0

    official_names = {item.get("table") for item in official_tables if isinstance(item, dict)}
    starvla_by_name = {
        item.get("table"): item
        for item in starvla_tables
        if isinstance(item, dict) and isinstance(item.get("table"), str)
    }
    expected_names = set(table_names)
    table_count = _nonnegative_int(report.get("tables"))
    expected_episodes = _nonnegative_int(report.get("episodes"))
    expected_rows = _nonnegative_int(report.get("rows"))
    expected_shards = _nonnegative_int(report.get("shards"))
    if (
        official_names != expected_names
        or set(starvla_by_name) != expected_names
        or table_count != len(table_names)
        or expected_episodes is None
        or expected_rows is None
        or expected_shards is None
        or manifest.get("total_episodes") != expected_episodes
        or manifest.get("total_rows") != expected_rows
        or manifest.get("total_shards") != expected_shards
        or status.get("total_episodes") != expected_episodes
        or status.get("total_rows") != expected_rows
        or status.get("total_shards") != expected_shards
        or status.get("completed_shards") != expected_shards
        or status.get("tables") != table_names
    ):
        return None, 0, 0

    total_episodes = 0
    total_frames = 0
    total_files = 0
    for name in table_names:
        task_root = output / name
        meta_root = task_root / "meta"
        info = _read_small_json(meta_root / "info.json")
        table_report = _read_small_json(task_root / "preparation_report.json")
        accepted = starvla_by_name[name]
        episodes = _nonnegative_int(info.get("total_episodes")) if info else None
        frames = _nonnegative_int(info.get("total_frames")) if info else None
        files = _nonnegative_int(info.get("total_files")) if info else None
        videos = _nonnegative_int(info.get("total_videos")) if info else None
        if (
            not table_report
            or table_report.get("status") != "passed"
            or table_report.get("stage") != "conversion_integrity"
            or table_report.get("partial_scope") is not False
            or episodes is None
            or frames is None
            or files is None
            or videos is None
            or table_report.get("episodes") != episodes
            or table_report.get("rows") != frames
            or table_report.get("files") != files
            or accepted.get("episodes") != episodes
            or accepted.get("frames") != frames
        ):
            return None, 0, 0
        if not (meta_root / "canonical_80d.json").is_file() or not _has_lerobot_task_metadata(meta_root):
            return None, 0, 0
        total_episodes += episodes
        total_frames += frames
        total_files += files
    if total_episodes != expected_episodes or total_frames != expected_rows or total_files != expected_shards:
        return None, 0, 0
    return report_path, len(table_names), total_episodes


@lru_cache(maxsize=4)
def _single_root_marker_complete(
    output: Path,
    status_name: str,
    *,
    require_stats: bool,
) -> tuple[Path | None, int]:
    """Validate a single LeRobot root using per-episode durable markers."""

    meta_root = output / "meta"
    info_path = meta_root / "info.json"
    info = _read_small_json(info_path)
    episodes = _nonnegative_int(info.get("total_episodes")) if info else None
    videos = _nonnegative_int(info.get("total_videos")) if info else None
    if not episodes or videos != episodes:
        return None, 0
    if not (meta_root / "canonical_80d.json").is_file() or not _has_lerobot_task_metadata(meta_root):
        return None, 0
    status_root = output / status_name
    if _direct_file_count(status_root / "episodes") != episodes:
        return None, 0
    if require_stats and _direct_file_count(status_root / "stats") != episodes:
        return None, 0
    return info_path, episodes


def _read_small_json_list(path: Path) -> list[Any] | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_REPORT_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, list) else None


@lru_cache(maxsize=4)
def _let_complete(output: Path) -> tuple[Path | None, int]:
    """Validate LET completion, including its explicit source exclusions."""

    status_root = output / ".let_status"
    manifest_path = status_root / "reindex_manifest.json"
    manifest = _read_small_json(manifest_path)
    audit = _read_small_json(status_root / "audit.json")
    progress = _read_small_json(status_root / "progress.json")
    failures = _read_small_json_list(status_root / "failures.json")
    exclusions = _read_small_json_list(status_root / "conversion_exclusions.json")
    evidence, episodes = _single_root_marker_complete(output, ".let_status", require_stats=True)
    if not evidence or not manifest or not audit or not progress or failures != [] or exclusions is None:
        return None, 0
    if manifest.get("status") != "COMPLETE" or manifest.get("episodes") != episodes:
        return None, 0
    if audit.get("status") != "PASS" or audit.get("episodes") != episodes:
        return None, 0
    completed = _nonnegative_int(progress.get("completed"))
    total = _nonnegative_int(progress.get("total"))
    if completed != total or progress.get("failures") != [] or total != episodes + len(exclusions):
        return None, 0
    return manifest_path, episodes


@lru_cache(maxsize=4)
def _something_label_count(dataset_root: Path) -> int:
    labels = dataset_root / "Something-Something-V2/20bn-something-something-download-package-labels.zip"
    try:
        if labels.is_symlink() or not labels.is_file() or labels.stat().st_size > _MAX_REPORT_BYTES:
            return 0
        with ZipFile(labels) as archive:
            total = 0
            for name in ("labels/train.json", "labels/validation.json", "labels/test.json"):
                payload = json.loads(archive.read(name))
                if not isinstance(payload, list):
                    return 0
                total += len(payload)
            classes = json.loads(archive.read("labels/labels.json"))
            if not isinstance(classes, dict) or len(classes) != 174:
                return 0
            return total
    except (OSError, KeyError, BadZipFile, UnicodeError, json.JSONDecodeError):
        return 0


def _first_valid_fragment_report(fragments_root: Path) -> Path | None:
    """Return one structured fragment report, stopping at the first match.

    This intentionally proves only partial conversion.  It does not enumerate a
    potentially very large 10Kh output tree merely to build the catalog.
    """

    try:
        chunks: Iterable[Path] = sorted(
            (item for item in fragments_root.iterdir() if item.is_dir() and item.name.startswith("chunk-")),
            key=lambda item: item.name,
        )
        for chunk in chunks:
            for episode in sorted(
                (item for item in chunk.iterdir() if item.is_dir() and item.name.startswith("episode-")),
                key=lambda item: item.name,
            ):
                report = episode / "validation.json"
                if _validated_report(report):
                    return report
    except OSError:
        return None
    return None


def _realomin_aggregate_complete(output: Path) -> bool:
    """Validate the published multi-root completion report for 10Kh-RealOmin."""

    payload = _read_small_json(output / "meta" / "aggregate_info.json")
    if not payload:
        return False
    if not (
        payload.get("status") == "COMPLETE"
        and payload.get("valid") is True
        and payload.get("published") is True
        and payload.get("source_modified") is False
    ):
        return False

    source_mcap = _nonnegative_int(payload.get("source_mcap"))
    readable_mcap = _nonnegative_int(payload.get("structurally_readable_mcap"))
    corrupt_mcap = _nonnegative_int(payload.get("upstream_corrupt_mcap"))
    exclusions = _nonnegative_int(payload.get("source_exclusions"))
    episodes = _nonnegative_int(payload.get("episodes"))
    parquet_files = _nonnegative_int(payload.get("parquet_files"))
    videos = _nonnegative_int(payload.get("videos"))
    if None in (source_mcap, readable_mcap, corrupt_mcap, exclusions, episodes, parquet_files, videos):
        return False
    if not episodes or source_mcap != readable_mcap + corrupt_mcap:
        return False
    if episodes != readable_mcap - exclusions or parquet_files != episodes or videos < episodes:
        return False

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        return False
    if _nonnegative_int(evidence.get("committed_markers")) != episodes:
        return False
    if _nonnegative_int(evidence.get("valid_sources")) != readable_mcap:
        return False
    if _nonnegative_int(evidence.get("active_failures")) != 0:
        return False

    reported_output = payload.get("formal_root")
    if reported_output == output.as_posix():
        return True
    return bool(
        isinstance(reported_output, str)
        and Path(reported_output).name == output.name
        and "/data/datasets/lerobotv3/high_priority_80d/" in reported_output
    )


def _status_payload(
    status: str,
    *,
    checked_at: str,
    output_path: Path | None = None,
    evidence_path: Path | None = None,
    source: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if status not in CONVERSION_STATUSES:
        status = "unknown"
    return {
        "conversion_status": status,
        "conversion_output_path": output_path.as_posix() if output_path else None,
        "conversion_evidence_path": evidence_path.as_posix() if evidence_path else None,
        "conversion_source": source,
        "conversion_note": note,
        "conversion_checked_at": checked_at,
    }


def _record_status(
    record: dict[str, Any],
    *,
    dataset_root: Path | None,
    pfs_root: Path | None,
    cfs_root: Path | None,
    checked_at: str,
) -> dict[str, Any]:
    dataset = record.get("dataset")
    source_path = str(record.get("source_path") or "")

    if dataset == "RoboMIND" and pfs_root:
        output = pfs_root / "RoboMIND-1.0-full-clean"
        report = output / "cleaning_report.json"
        if _robomind_reports_complete(output):
            if "/failure_data/" in f"/{source_path.strip('/')}" or source_path.startswith("RoboMIND/failure_data/"):
                return _status_payload(
                    "excluded",
                    checked_at=checked_at,
                    output_path=output,
                    evidence_path=report,
                    source="cleaning_report",
                    note="正式转换按规则排除了 failure_data 演示。",
                )
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="cleaning_report",
                note="正式清洗输出已完成；原始分卷路径与清洗后任务采用聚合对应。",
            )

    if dataset == "RoboMINDv2" and pfs_root:
        output = pfs_root / "RoboMIND-2.0-full"
        report = output / "cleaning_report.json"
        if _robomind_reports_complete(output):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="cleaning_report",
            )

    if dataset == "FastUMI-Data" and pfs_root:
        output = pfs_root / "FastUMI-Data-80D-full-compact"
        report = output / ".done.json"
        if _done_report_complete(report) and _task_matches(
            output / "meta" / "tasks.jsonl", str(record.get("native_task") or "")
        ):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="validation_report",
            )

    if dataset == "FastUMI-100K" and pfs_root:
        output_root = pfs_root / "FastUMI-100K-80D-full"
        native_group = str(record.get("native_group") or "")
        native_task = str(record.get("native_task") or "")
        output = output_root / native_group / native_task
        report = output / "meta" / "info.json"
        if _canonical_task_complete(output):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="lerobot_meta_info",
            )
        if _fastumi_snapshot_available(output_root):
            return _status_payload(
                "not_converted",
                checked_at=checked_at,
                output_path=output,
                source="exact_task_output_check",
                note="正式输出根存在，但未找到该原生任务的有效 meta/info.json。",
            )

    if dataset == "HiFi-UMI-2K" and pfs_root:
        output = pfs_root / "HiFi-UMI-2K-80D-full"
        converted_tasks, evidence, valid_parts, converted_episodes = _hifi_output_task_index(output)
        source_path = Path(str(record.get("source_path") or ""))
        if source_path.is_absolute():
            source = source_path
        else:
            source = dataset_root / source_path if dataset_root else None
        source_partitions = _hifi_partition_episode_counts(source, False) if source else ()
        output_partitions = _hifi_partition_episode_counts(output, True)
        conversion_complete = bool(source_partitions and source_partitions == output_partitions)
        if converted_tasks and _normalize_task(str(record.get("native_task") or "")) in converted_tasks:
            note_suffix = (
                "输出分块及 episode 数与原始 metadata 逐一一致。"
                if conversion_complete
                else "尚未证明输出与全部原始分块逐一一致，因此保守标记为部分转换。"
            )
            return _status_payload(
                "converted" if conversion_complete else "partial",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="lerobot_partition_parity" if conversion_complete else "lerobot_part_metadata",
                note=(
                    f"已发现 {valid_parts} 个结构可读的 LeRobot 分块、"
                    f"{converted_episodes} 个转换回合；{note_suffix}"
                ),
            )
        if converted_tasks:
            return _status_payload(
                "not_converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="exact_task_output_check",
                note="已存在 HiFi 正式转换分块，但转换任务表中未找到该 instruction。",
            )

    if dataset == "Galaxea-Open-World" and pfs_root and dataset_root:
        output = pfs_root / "Galaxea-Open-World"
        evidence, valid_roots, converted_episodes = _galaxea_complete(dataset_root, output)
        if evidence:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="source_archive_output_parity",
                note=(
                    f"{valid_roots} 个源任务包与 LeRobot 任务根逐一对应；"
                    f"{converted_episodes} 个 episode 的数据与视频结构完整。"
                ),
            )

    if dataset == "Hy-Embodied-0.5-VLA-Data" and pfs_root and dataset_root:
        output = pfs_root / "Hy-Embodied-0.5-VLA-Data-80D-starvla-v3"
        evidence, valid_roots, converted_episodes = _hy_embodied_complete(dataset_root, output)
        if evidence:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="validated_starvla_v3_acceptance",
                note=(
                    f"StarVLA v3 转换与双读取器验收通过：{valid_roots} 个数据表、"
                    f"{converted_episodes} 个 episode。"
                ),
            )

    if dataset == "LET-claw" and pfs_root:
        output = pfs_root / "LET-claw"
        evidence, converted_episodes = _let_complete(output)
        if evidence:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="validated_completion_manifest",
                note=f"完成清单与审计均通过；{converted_episodes} 个有效 episode，源损坏项已明确排除。",
            )

    if dataset == "RT-1" and pfs_root:
        output = pfs_root / "RT-1"
        evidence, converted_episodes = _single_root_marker_complete(output, ".rt1_status", require_stats=True)
        if evidence:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="per_episode_output_validation",
                note=f"{converted_episodes} 个 episode 的状态、统计、Parquet 与视频数量一致。",
            )

    if dataset == "Something-Something-V2" and pfs_root and dataset_root:
        output = pfs_root / "Something-Something-V2"
        evidence, converted_episodes = _single_root_marker_complete(
            output, ".something_v2_status", require_stats=False
        )
        if evidence and converted_episodes == _something_label_count(dataset_root):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="label_and_episode_output_parity",
                note=f"标签清单与转换输出均为 {converted_episodes} 个视频 episode。",
            )

    if dataset in GENERIC_PARTIAL_OUTPUTS and pfs_root:
        output = pfs_root / GENERIC_PARTIAL_OUTPUTS[dataset]
        evidence, valid_roots, converted_episodes = _lerobot_metadata_evidence(output)
        if evidence:
            return _status_payload(
                "partial",
                checked_at=checked_at,
                output_path=output,
                evidence_path=evidence,
                source="lerobot_metadata_collection",
                note=(
                    f"已发现 {valid_roots} 个结构可读的 LeRobot 根、"
                    f"{converted_episodes} 个转换回合；"
                    "缺少覆盖全部源数据的全局完成校验报告，因此保守标记为部分转换。"
                ),
            )

    if dataset == "EgoVerse" and pfs_root:
        output = pfs_root / "EgoVerse"
        return _status_payload(
            "not_converted",
            checked_at=checked_at,
            output_path=output,
            source="exact_dataset_output_check",
            note="PFS 正式转换根已挂载，但未找到 EgoVerse 的 LeRobot 输出。",
        )

    if dataset == "FreeTacMan" and cfs_root:
        output = cfs_root / "FreeTacMan-80D-formal-v1"
        report = output / ".done.json"
        task = str(record.get("native_task") or "")
        task_match = any(_task_matches(output / root / "meta" / "tasks.jsonl", task) for root in ("2cam", "3cam"))
        if _done_report_complete(report) and task_match:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="validation_report",
            )

    if dataset == "Touch-in-the-Wild" and cfs_root:
        output = cfs_root / "Touch-in-the-Wild-80D-clean-v4" / "processed_zarr"
        report = output / ".done.json"
        expected_task = "/".join(
            [
                str(record.get("native_group") or ""),
                str(record.get("native_task") or ""),
                str(record.get("native_task") or ""),
            ]
        )
        if _done_report_complete(report, require_valid_exclusions=True) and _task_matches(
            output / "meta" / "tasks.jsonl", expected_task
        ):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="validation_report",
            )

    if dataset == "exUMI" and cfs_root:
        output = cfs_root / "exUMI-80D-formal-v4"
        report = output / ".done.json"
        task = str(record.get("native_task") or "")
        task_match = any(
            _task_matches(output / root / "meta" / "tasks.jsonl", task) for root in ("camera_only", "with_tactile")
        )
        if _done_report_complete(report) and task_match:
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="validation_report",
            )

    if dataset == "EgoDex" and pfs_root:
        output = pfs_root / "EgoDex"
        report = output / "meta" / "info.json"
        canonical = output / "meta" / "canonical_80d.json"
        task_families = output / "meta" / "episodes.jsonl"
        if (
            canonical.is_file()
            and _positive_count(_read_small_json(report), "total_episodes")
            and _task_matches(task_families, str(record.get("native_task") or ""), key="task_family")
        ):
            return _status_payload(
                "converted",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="lerobot_meta_info",
            )

    if dataset == "10Kh-RealOmin":
        if pfs_root:
            output = pfs_root / "10Kh-RealOmin-OpenData-80D-multiroot-v1"
            report = output / "meta" / "aggregate_info.json"
            if _realomin_aggregate_complete(output):
                return _status_payload(
                    "converted",
                    checked_at=checked_at,
                    output_path=output,
                    evidence_path=report,
                    source="validated_multi_root_catalog",
                    note="完整数据集已发布为经过校验的多根 LeRobot 目录；片段保持不可变。",
                )
        output = cfs_root / "10Kh-RealOmin-OpenData.formal-fragments-v2" if cfs_root else None
        report = _first_valid_fragment_report(output / "fragments") if output else None
        if report:
            return _status_payload(
                "partial",
                checked_at=checked_at,
                output_path=output,
                evidence_path=report,
                source="validated_fragment",
                note="已存在通过校验的正式片段，但片段尚未与语义任务逐条对应。",
            )

    return _status_payload("unknown", checked_at=checked_at)


def enrich_conversion_status(
    catalog: dict[str, Any],
    *,
    pfs_root: Path | None = None,
    cfs_root: Path | None = None,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Mutate and return ``catalog`` with per-record conversion evidence."""

    effective_checked_at = checked_at or str(catalog.get("generated_at") or "unknown")
    dataset_root_value = catalog.get("dataset_root")
    dataset_root = Path(dataset_root_value) if isinstance(dataset_root_value, str) and dataset_root_value else None
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError("catalog.records must be a list")

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("catalog.records must contain objects")
        record.update(
            _record_status(
                record,
                dataset_root=dataset_root,
                pfs_root=pfs_root,
                cfs_root=cfs_root,
                checked_at=effective_checked_at,
            )
        )

    counts = Counter(str(record.get("conversion_status") or "unknown") for record in records)
    by_status = {status: counts.get(status, 0) for status in CONVERSION_STATUSES}
    catalog["conversion_schema_version"] = 1
    catalog["conversion"] = {
        "schema_version": 1,
        "checked_at": effective_checked_at,
        "pfs_output_root": pfs_root.as_posix() if pfs_root else None,
        "cfs_output_root": cfs_root.as_posix() if cfs_root else None,
        "status_labels": CONVERSION_STATUS_LABELS,
        "by_status": by_status,
    }
    stats = catalog.setdefault("stats", {})
    stats["conversion_by_status"] = by_status
    stats["converted_records"] = by_status["converted"]
    stats["partial_conversion_records"] = by_status["partial"]
    return catalog
