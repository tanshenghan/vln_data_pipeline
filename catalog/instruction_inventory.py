"""Read-only extraction of task instructions from heterogeneous dataset metadata.

The inventory is deliberately separate from semantic classification.  Each output
row describes one traceable instruction occurrence and keeps enough provenance to
map a future label back to a task definition or episode.  Media payloads are never
opened and pickle files are intentionally ignored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1
SOURCE_CATALOG = "catalog_task"
SOURCE_EGOVERSE = "egoverse_zarr"
SOURCE_SSV2 = "something_something_labels"
SOURCE_HIFI = "hifi_umi_lerobot_parquet"
SOURCE_HY = "hy_embodied_task_parquet"
SOURCE_LEROBOT_JSONL = "lerobot_episode_jsonl"
MAX_ISSUE_SAMPLES = 100
MAX_CONVERSION_META_DEPTH = 2
CONVERSION_TEST_MARKERS = ("smoke", "preflight", "integration-test")


class ScanDiagnostics:
    """Bounded diagnostics so a damaged metadata row does not abort the inventory."""

    def __init__(self) -> None:
        self.files_scanned: Counter[str] = Counter()
        self.missing_metadata_files: Counter[str] = Counter()
        self.missing_instruction: Counter[str] = Counter()
        self.issue_count = 0
        self.issue_samples: list[dict[str, Any]] = []

    def scanned(self, source_kind: str) -> None:
        self.files_scanned[source_kind] += 1

    def missing(self, source_kind: str) -> None:
        self.missing_instruction[source_kind] += 1

    def metadata_missing(self, source_kind: str) -> None:
        self.missing_metadata_files[source_kind] += 1

    def issue(self, source_kind: str, path: str | Path, error: BaseException, *, line: int | None = None) -> None:
        self.issue_count += 1
        if len(self.issue_samples) >= MAX_ISSUE_SAMPLES:
            return
        self.issue_samples.append(
            {
                "source_kind": source_kind,
                "path": str(path),
                "line": line,
                "error_type": type(error).__name__,
                "message": str(error)[:500],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_scanned": dict(sorted(self.files_scanned.items())),
            "missing_metadata_files": dict(sorted(self.missing_metadata_files.items())),
            "missing_instruction": dict(sorted(self.missing_instruction.items())),
            "issue_count": self.issue_count,
            "issue_samples": self.issue_samples,
            "issue_samples_truncated": self.issue_count > len(self.issue_samples),
        }


@dataclass(frozen=True)
class InstructionRecord:
    """One instruction occurrence with stable identity and source provenance."""

    schema_version: int
    record_id: str
    instruction_hash: str
    domain: str
    dataset: str
    source_kind: str
    source_unit: str
    source_record_id: str
    occurrence_kind: str
    instruction: str
    normalized_instruction: str
    instruction_field: str
    metadata_path: str
    episode_id: str | None = None
    split: str | None = None
    native_scene: str | None = None
    embodiment: str | None = None
    task_index: int | None = None
    episode_length: int | None = None
    episode_count: int | None = None
    instruction_template: str | None = None


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_instruction(value: Any) -> str:
    """Return normalized display text without changing the original language."""

    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_instruction(value: Any) -> str:
    return clean_instruction(value).casefold()


def instruction_hash(value: str) -> str:
    return hashlib.sha256(normalize_instruction(value).encode("utf-8")).hexdigest()


def _stable_record_id(parts: Iterable[Any]) -> str:
    key = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.blake2s(key.encode("utf-8"), digest_size=16).hexdigest()


def make_record(
    *,
    domain: str,
    dataset: str,
    source_kind: str,
    source_unit: str,
    source_record_id: Any,
    occurrence_kind: str,
    instruction: Any,
    instruction_field: str,
    metadata_path: str | Path,
    episode_id: Any = None,
    split: str | None = None,
    native_scene: str | None = None,
    embodiment: str | None = None,
    task_index: Any = None,
    episode_length: Any = None,
    episode_count: Any = None,
    instruction_template: Any = None,
) -> InstructionRecord | None:
    text = clean_instruction(instruction)
    if not text:
        return None
    normalized = normalize_instruction(text)
    source_record_text = str(source_record_id)
    episode_text = None if episode_id is None else str(episode_id)
    digest = instruction_hash(text)
    record_id = _stable_record_id(
        (source_kind, dataset, source_unit, source_record_text, episode_text, instruction_field, digest)
    )
    return InstructionRecord(
        schema_version=SCHEMA_VERSION,
        record_id=record_id,
        instruction_hash=digest,
        domain=domain,
        dataset=dataset,
        source_kind=source_kind,
        source_unit=source_unit,
        source_record_id=source_record_text,
        occurrence_kind=occurrence_kind,
        instruction=text,
        normalized_instruction=normalized,
        instruction_field=instruction_field,
        metadata_path=str(metadata_path),
        episode_id=episode_text,
        split=split,
        native_scene=clean_instruction(native_scene) or None,
        embodiment=clean_instruction(embodiment) or None,
        task_index=_optional_int(task_index),
        episode_length=_optional_int(episode_length),
        episode_count=_optional_int(episode_count),
        instruction_template=clean_instruction(instruction_template) or None,
    )


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _read_json(
    path: Path,
    *,
    diagnostics: ScanDiagnostics | None = None,
    source_kind: str = "json",
) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if diagnostics:
            diagnostics.scanned(source_kind)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if diagnostics is None:
            raise
        diagnostics.issue(source_kind, path, exc)
        return None


def _read_jsonl(
    path: Path,
    *,
    diagnostics: ScanDiagnostics | None = None,
    source_kind: str = "jsonl",
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if diagnostics:
                diagnostics.scanned(source_kind)
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if diagnostics is None:
                        raise
                    diagnostics.issue(source_kind, path, exc, line=line_number)
                    continue
                if isinstance(value, dict):
                    yield line_number, value
    except (OSError, UnicodeError) as exc:
        if diagnostics is None:
            raise
        diagnostics.issue(source_kind, path, exc)


def _read_parquet_rows(
    path: Path,
    columns: list[str],
    *,
    diagnostics: ScanDiagnostics | None = None,
    source_kind: str = "parquet",
) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError("pyarrow is required to scan parquet metadata") from exc

    try:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected = [column for column in columns if column in available]
        if diagnostics:
            diagnostics.scanned(source_kind)
        if not selected:
            return
        for batch in parquet.iter_batches(columns=selected, batch_size=4096):
            yield from batch.to_pylist()
    except (OSError, ValueError) as exc:
        if diagnostics is None:
            raise
        diagnostics.issue(source_kind, path, exc)


def scan_catalog_records(
    catalog_path: Path, diagnostics: ScanDiagnostics | None = None
) -> Iterator[InstructionRecord]:
    catalog = _read_json(catalog_path, diagnostics=diagnostics, source_kind=SOURCE_CATALOG)
    if not isinstance(catalog, dict):
        return
    for row in catalog.get("records", []):
        if not isinstance(row, dict):
            continue
        record = make_record(
            domain=row.get("domain") or "unknown",
            dataset=row.get("dataset") or "unknown",
            source_kind=SOURCE_CATALOG,
            source_unit=row.get("source_path") or "catalog",
            source_record_id=row.get("id") or row.get("source_path") or "unknown",
            occurrence_kind="task_path",
            instruction=row.get("native_task"),
            instruction_field="native_task",
            metadata_path=catalog_path,
            native_scene=row.get("native_scene"),
            episode_count=row.get("episode_count"),
        )
        if record:
            yield record
        elif diagnostics:
            diagnostics.missing(SOURCE_CATALOG)


def _batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _egoverse_candidates(root: Path) -> Iterator[tuple[str, Path]]:
    manifest = root / ".egoverse-download" / "active_episodes.tsv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                episode_id = clean_instruction(row.get("episode_hash"))
                if episode_id:
                    yield episode_id, root / episode_id / "zarr.json"
        return
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                yield entry.name, Path(entry.path) / "zarr.json"


def _read_egoverse_metadata(
    candidate: tuple[str, Path],
) -> tuple[str, Path, dict[str, Any] | None, BaseException | None]:
    episode_id, metadata_path = candidate
    if not metadata_path.is_file():
        return episode_id, metadata_path, None, None
    try:
        value = _read_json(metadata_path)
        attributes = value.get("attributes", {}) if isinstance(value, dict) else {}
        return episode_id, metadata_path, attributes if isinstance(attributes, dict) else {}, None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return episode_id, metadata_path, None, exc


def scan_egoverse(
    dataset_root: Path,
    diagnostics: ScanDiagnostics | None = None,
    *,
    metadata_workers: int = 16,
) -> Iterator[InstructionRecord]:
    root = dataset_root / "EgoVerse"
    if not root.is_dir():
        return
    workers = max(1, metadata_workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="egoverse-meta") as executor:
        for batch in _batches(_egoverse_candidates(root), workers * 16):
            for episode_id, metadata_path, attributes, error in executor.map(_read_egoverse_metadata, batch):
                if error is not None:
                    if diagnostics:
                        diagnostics.issue(SOURCE_EGOVERSE, metadata_path, error)
                    continue
                if attributes is None:
                    if diagnostics:
                        diagnostics.metadata_missing(SOURCE_EGOVERSE)
                    continue
                if diagnostics:
                    diagnostics.scanned(SOURCE_EGOVERSE)
                field = "task_name" if clean_instruction(attributes.get("task_name")) else "task_description"
                record = make_record(
                    domain="ego",
                    dataset="EgoVerse",
                    source_kind=SOURCE_EGOVERSE,
                    source_unit=episode_id,
                    source_record_id=episode_id,
                    occurrence_kind="episode",
                    instruction=attributes.get(field),
                    instruction_field=f"attributes.{field}",
                    metadata_path=metadata_path,
                    episode_id=episode_id,
                    embodiment=attributes.get("embodiment"),
                    episode_length=attributes.get("total_frames"),
                )
                if record:
                    yield record
                elif diagnostics:
                    diagnostics.missing(SOURCE_EGOVERSE)


def scan_something_something(
    dataset_root: Path, diagnostics: ScanDiagnostics | None = None
) -> Iterator[InstructionRecord]:
    archive = dataset_root / "Something-Something-V2" / "20bn-something-something-download-package-labels.zip"
    if not archive.is_file():
        return
    with zipfile.ZipFile(archive) as bundle:
        for split, member in (("train", "labels/train.json"), ("validation", "labels/validation.json")):
            with bundle.open(member) as handle:
                rows = json.load(handle)
            if diagnostics:
                diagnostics.scanned(SOURCE_SSV2)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sample_id = row.get("id")
                field = "label" if clean_instruction(row.get("label")) else "template"
                record = make_record(
                    domain="ego",
                    dataset="Something-Something-V2",
                    source_kind=SOURCE_SSV2,
                    source_unit=split,
                    source_record_id=sample_id,
                    occurrence_kind="video",
                    instruction=row.get(field),
                    instruction_field=field,
                    metadata_path=f"{archive}!{member}",
                    episode_id=sample_id,
                    split=split,
                    instruction_template=row.get("template"),
                )
                if record:
                    yield record
                elif diagnostics:
                    diagnostics.missing(SOURCE_SSV2)

        member = "labels/test-answers.csv"
        with bundle.open(member) as raw_handle:
            if diagnostics:
                diagnostics.scanned(SOURCE_SSV2)
            rows = csv.reader((line.decode("utf-8") for line in raw_handle), delimiter=";")
            for row in rows:
                if len(row) < 2:
                    continue
                sample_id, template = row[0], row[1]
                record = make_record(
                    domain="ego",
                    dataset="Something-Something-V2",
                    source_kind=SOURCE_SSV2,
                    source_unit="test",
                    source_record_id=sample_id,
                    occurrence_kind="video",
                    instruction=template,
                    instruction_field="test-answers.csv.template",
                    metadata_path=f"{archive}!{member}",
                    episode_id=sample_id,
                    split="test",
                    instruction_template=template,
                )
                if record:
                    yield record
                elif diagnostics:
                    diagnostics.missing(SOURCE_SSV2)


def scan_hifi_umi(dataset_root: Path, diagnostics: ScanDiagnostics | None = None) -> Iterator[InstructionRecord]:
    root = dataset_root / "HiFi-UMI-2K-main"
    if not root.is_dir():
        return
    for part in sorted(root.glob("chunk-*/part-*")):
        meta = part / "meta"
        info_path = meta / "info.json"
        info = (
            _read_json(info_path, diagnostics=diagnostics, source_kind=SOURCE_HIFI) if info_path.is_file() else {}
        )
        embodiment = info.get("robot_type") if isinstance(info, dict) else None
        files = sorted((meta / "episodes").glob("chunk-*/file-*.parquet"))
        emitted = False
        for path in files:
            for row in _read_parquet_rows(
                path,
                ["episode_index", "tasks", "length"],
                diagnostics=diagnostics,
                source_kind=SOURCE_HIFI,
            ):
                tasks = row.get("tasks")
                if isinstance(tasks, str):
                    tasks = [tasks]
                if not isinstance(tasks, list):
                    continue
                episode_id = row.get("episode_index")
                for task_position, task in enumerate(tasks):
                    record = make_record(
                        domain="umi",
                        dataset="HiFi-UMI-2K",
                        source_kind=SOURCE_HIFI,
                        source_unit=part.relative_to(root).as_posix(),
                        source_record_id=f"{episode_id}:{task_position}",
                        occurrence_kind="episode",
                        instruction=task,
                        instruction_field="tasks[]",
                        metadata_path=path,
                        episode_id=episode_id,
                        embodiment=embodiment,
                        episode_length=row.get("length"),
                    )
                    if record:
                        emitted = True
                        yield record
                    elif diagnostics:
                        diagnostics.missing(SOURCE_HIFI)
        if emitted:
            continue
        tasks_path = meta / "tasks.parquet"
        if tasks_path.is_file():
            yield from _scan_task_parquet(
                tasks_path,
                domain="umi",
                dataset="HiFi-UMI-2K",
                source_kind=SOURCE_HIFI,
                source_unit=part.relative_to(root).as_posix(),
                embodiment=embodiment,
                diagnostics=diagnostics,
            )


def scan_hy_embodied(dataset_root: Path, diagnostics: ScanDiagnostics | None = None) -> Iterator[InstructionRecord]:
    root = dataset_root / "tencent" / "Hy-Embodied-0.5-VLA-Data"
    if not root.is_dir():
        return
    for tasks_path in sorted(root.glob("table_*/meta/tasks.parquet")):
        table = tasks_path.parents[1]
        yield from _scan_task_parquet(
            tasks_path,
            domain="real",
            dataset="Hy-Embodied-0.5-VLA-Data",
            source_kind=SOURCE_HY,
            source_unit=table.relative_to(root).as_posix(),
            diagnostics=diagnostics,
        )


def _scan_task_parquet(
    path: Path,
    *,
    domain: str,
    dataset: str,
    source_kind: str,
    source_unit: str,
    embodiment: str | None = None,
    diagnostics: ScanDiagnostics | None = None,
) -> Iterator[InstructionRecord]:
    for row_number, row in enumerate(
        _read_parquet_rows(
            path,
            ["task_index", "task", "instruction"],
            diagnostics=diagnostics,
            source_kind=source_kind,
        ),
        1,
    ):
        field = "task" if clean_instruction(row.get("task")) else "instruction"
        task_index = row.get("task_index")
        record = make_record(
            domain=domain,
            dataset=dataset,
            source_kind=source_kind,
            source_unit=source_unit,
            source_record_id=task_index if task_index is not None else row_number,
            occurrence_kind="task_definition",
            instruction=row.get(field),
            instruction_field=field,
            metadata_path=path,
            embodiment=embodiment,
            task_index=task_index,
        )
        if record:
            yield record
        elif diagnostics:
            diagnostics.missing(source_kind)


def _infer_domain(dataset: str) -> str:
    text = dataset.casefold()
    if "umi" in text:
        return "umi"
    if any(token in text for token in ("libero", "rlbench", "calvin", "maniskill", "metaworld", "sim")):
        return "sim"
    if any(token in text for token in ("ego", "something-something")):
        return "ego"
    if any(token in text for token in ("vqa", "ground", "refspatial")):
        return "vqa"
    return "real"


def _iter_meta_directories(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    seen: set[Path] = set()
    for depth in range(1, MAX_CONVERSION_META_DEPTH + 1):
        pattern = "/".join(["*"] * depth + ["meta", "episodes.jsonl"])
        for episodes_path in root.glob(pattern):
            relative_parts = episodes_path.relative_to(root).parts
            if any(
                part.startswith(".") or any(marker in part.casefold() for marker in CONVERSION_TEST_MARKERS)
                for part in relative_parts
            ):
                continue
            if not episodes_path.is_file() or episodes_path.parent in seen:
                continue
            seen.add(episodes_path.parent)
            yield episodes_path.parent


def scan_lerobot_jsonl(
    conversion_roots: Iterable[Path], diagnostics: ScanDiagnostics | None = None
) -> Iterator[InstructionRecord]:
    for conversion_root in conversion_roots:
        if not conversion_root.is_dir():
            continue
        for meta in sorted(_iter_meta_directories(conversion_root)):
            unit = meta.parent.relative_to(conversion_root).as_posix()
            dataset = unit.split("/", 1)[0]
            info_path = meta / "info.json"
            info = (
                _read_json(info_path, diagnostics=diagnostics, source_kind=SOURCE_LEROBOT_JSONL)
                if info_path.is_file()
                else {}
            )
            embodiment = None
            if isinstance(info, dict):
                embodiment = info.get("robot_type") or info.get("embodiment")
            if not embodiment and "/" in unit:
                embodiment = unit.split("/", 1)[1]
            episodes_path = meta / "episodes.jsonl"
            emitted = False
            for line_number, row in _read_jsonl(
                episodes_path, diagnostics=diagnostics, source_kind=SOURCE_LEROBOT_JSONL
            ):
                tasks = row.get("tasks")
                field = "tasks[]"
                if tasks is None:
                    tasks = row.get("task") or row.get("instruction")
                    field = "task" if row.get("task") is not None else "instruction"
                if isinstance(tasks, str):
                    tasks = [tasks]
                if not isinstance(tasks, list):
                    continue
                episode_id = row.get("episode_index", line_number - 1)
                for task_position, task in enumerate(tasks):
                    record = make_record(
                        domain=_infer_domain(dataset),
                        dataset=dataset,
                        source_kind=SOURCE_LEROBOT_JSONL,
                        source_unit=unit,
                        source_record_id=f"{episode_id}:{task_position}",
                        occurrence_kind="episode",
                        instruction=task,
                        instruction_field=field,
                        metadata_path=episodes_path,
                        episode_id=episode_id,
                        embodiment=embodiment,
                        episode_length=row.get("length") or row.get("frames"),
                    )
                    if record:
                        emitted = True
                        yield record
                    elif diagnostics:
                        diagnostics.missing(SOURCE_LEROBOT_JSONL)
            if emitted:
                continue
            tasks_path = meta / "tasks.jsonl"
            if not tasks_path.is_file():
                continue
            for line_number, row in _read_jsonl(
                tasks_path, diagnostics=diagnostics, source_kind=SOURCE_LEROBOT_JSONL
            ):
                field = "task" if clean_instruction(row.get("task")) else "instruction"
                task_index = row.get("task_index", line_number - 1)
                record = make_record(
                    domain=_infer_domain(dataset),
                    dataset=dataset,
                    source_kind=SOURCE_LEROBOT_JSONL,
                    source_unit=unit,
                    source_record_id=task_index,
                    occurrence_kind="task_definition",
                    instruction=row.get(field),
                    instruction_field=field,
                    metadata_path=tasks_path,
                    embodiment=embodiment,
                    task_index=task_index,
                )
                if record:
                    yield record
                elif diagnostics:
                    diagnostics.missing(SOURCE_LEROBOT_JSONL)


def known_unextractable_sources(dataset_root: Path) -> list[dict[str, str]]:
    definitions = (
        (
            "Egocentric-100K",
            dataset_root / "Egocentric-100K",
            "TAR 内的 clip JSON 仅含视频技术信息, 未发现 instruction/task 字段。",
        ),
        (
            "Touch-in-the-Wild/in_the_wild_data",
            dataset_root / "binghaohuang-robot" / "touch_in_the_wild-dataset" / "in_the_wild_data",
            "Zarr 与 session 计划提供场景和 episode 边界, 未发现安全可读的 instruction 字段; pickle 不反序列化。",
        ),
    )
    return [
        {"dataset": dataset, "path": str(path), "status": "no_instruction_metadata", "reason": reason}
        for dataset, path, reason in definitions
        if path.exists()
    ]


class InventoryWriter:
    """Stream records to an atomic JSONL file while collecting compact statistics."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.records = 0
        self.duplicate_records_skipped = 0
        self._seen_ids: set[str] = set()
        self._unique_instructions: set[str] = set()
        self._dataset_instructions: dict[str, set[str]] = defaultdict(set)
        self.by_dataset: Counter[str] = Counter()
        self.by_source_kind: Counter[str] = Counter()

    def write(self, sources: Iterable[Iterable[InstructionRecord]]) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for source in sources:
                    for record in source:
                        if record.record_id in self._seen_ids:
                            self.duplicate_records_skipped += 1
                            continue
                        self._seen_ids.add(record.record_id)
                        self._unique_instructions.add(record.instruction_hash)
                        self._dataset_instructions[record.dataset].add(record.instruction_hash)
                        self.by_dataset[record.dataset] += 1
                        self.by_source_kind[record.source_kind] += 1
                        handle.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
                        handle.write("\n")
                        self.records += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def summary(
        self,
        *,
        generated_at: str,
        dataset_root: Path,
        catalog_path: Path | None,
        diagnostics: ScanDiagnostics,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "dataset_root": str(dataset_root),
            "catalog_path": str(catalog_path) if catalog_path else None,
            "output": str(self.output),
            "records": self.records,
            "unique_instructions": len(self._unique_instructions),
            "duplicate_records_skipped": self.duplicate_records_skipped,
            "by_dataset": {
                dataset: {
                    "records": count,
                    "unique_instructions": len(self._dataset_instructions[dataset]),
                }
                for dataset, count in sorted(self.by_dataset.items())
            },
            "by_source_kind": dict(sorted(self.by_source_kind.items())),
            "diagnostics": diagnostics.as_dict(),
            "unextractable_sources": known_unextractable_sources(dataset_root),
        }


def write_json_atomic(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_instruction_inventory(
    *,
    dataset_root: Path,
    output: Path,
    summary_output: Path,
    catalog_path: Path | None = None,
    conversion_roots: Iterable[Path] = (),
    metadata_workers: int = 16,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist: {dataset_root}")
    resolved_catalog = catalog_path.resolve() if catalog_path else None
    resolved_conversions = tuple(path.resolve() for path in conversion_roots)
    diagnostics = ScanDiagnostics()
    sources: list[Iterable[InstructionRecord]] = []
    if resolved_catalog and resolved_catalog.is_file():
        sources.append(scan_catalog_records(resolved_catalog, diagnostics))
    sources.extend(
        (
            scan_egoverse(dataset_root, diagnostics, metadata_workers=metadata_workers),
            scan_something_something(dataset_root, diagnostics),
            scan_hifi_umi(dataset_root, diagnostics),
            scan_hy_embodied(dataset_root, diagnostics),
            scan_lerobot_jsonl(resolved_conversions, diagnostics),
        )
    )
    writer = InventoryWriter(output.resolve())
    writer.write(sources)
    summary = writer.summary(
        generated_at=utc_now(),
        dataset_root=dataset_root,
        catalog_path=resolved_catalog,
        diagnostics=diagnostics,
    )
    write_json_atomic(summary, summary_output.resolve())
    return summary
