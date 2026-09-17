"""Build a task-level semantic catalog from the downloaded dataset tree.

The scanner intentionally inspects only names and shallow metadata-bearing directory
levels. It never opens media, moves data, extracts archives, or treats technical
storage directories (chunks, tables, UUID episodes) as semantic tasks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

DOMAIN_LABELS = {
    "real": "真机",
    "sim": "仿真",
    "umi": "UMI 数据集",
    "ego": "Ego",
    "vqa": "VQA 数据集",
}
UNKNOWN_GROUP = "未提供原生分组"
UNKNOWN_SCENE = "其他 / 待人工标注"
UNKNOWN_TASK_TYPE = "复合任务 / 待人工标注"
NATIVE_SCENE_LABELS = {
    "bedroom": "卧室",
    "living room": "客厅",
    "kitchen": "厨房与餐饮",
    "study": "书房",
}

# The Feishu research document defines UMI as a peer data domain and reserves
# three VQA families.  VQA dataset names come from the matching P2 processing
# ledger.  Entries are declarative and shallow: a missing directory remains a
# reference-catalog item and never becomes a task record.
UMI_REFERENCE_SOURCES = (
    (
        "HiFi-UMI-2K",
        "HiFi-UMI-2K",
        ("HiFi-UMI-2K-main", "HiFi-UMI-2K"),
        "https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K",
    ),
    (
        "FastUMI-100K",
        "FastUMI-100K",
        ("IPEC-COMMUNITY/FastUMI_100k_lerobot",),
        "https://github.com/MrKeee/FastUMI-100K",
    ),
    (
        "FastUMI-Data",
        "FastUMI Dataset",
        ("IPEC-COMMUNITY/FastUMI-Data",),
        "https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data",
    ),
    (
        "UMI Original Dataset",
        "UMI Original Dataset",
        (),
        "https://umi-data.github.io/",
    ),
)

VQA_REFERENCE_SOURCES = (
    ("VQA / Grounding", "LocateAnything", ("LocateAnything",), "LocateAnything 138M query release"),
    ("2/3D Grounding", "Omni3D", ("facebookresearch/omni3d",), "facebookresearch/omni3d"),
    ("EmbodiedVQA", "RefSpatial", ("JingkunAn/RefSpatial",), "JingkunAn/RefSpatial"),
    ("EmbodiedVQA", "EO-Data1.5M", ("IPEC-COMMUNITY/EO-Data1.5M",), "IPEC-COMMUNITY/EO-Data1.5M"),
    ("EmbodiedPlanning", "ShareRobot", ("BAAI/ShareRobot",), "BAAI/ShareRobot"),
    ("EmbodiedPlanning", "EgoPlanIT", ("ChenYi99/EgoPlan",), "ChenYi99/EgoPlan"),
    ("2/3D Grounding", "CA-1M", ("apple/ml-cubifyanything",), "apple/ml-cubifyanything"),
)

# Datasets whose semantic tasks live in metadata rather than useful directory
# names.  The full instruction inventory is produced by the read-only metadata
# scanner; the catalog consumes only these canonical, non-duplicate datasets.
INVENTORY_TASK_SOURCES = {
    "EgoVerse": {
        "domain": "ego",
        "source_kind": "egoverse_zarr",
        "source_path": "EgoVerse",
        "use_template": False,
    },
    "Something-Something-V2": {
        "domain": "ego",
        "source_kind": "something_something_labels",
        "source_path": "Something-Something-V2",
        "use_template": True,
    },
    "Hy-Embodied-0.5-VLA-Data": {
        "domain": "real",
        "source_kind": "hy_embodied_task_parquet",
        "source_path": "tencent/Hy-Embodied-0.5-VLA-Data",
        "use_template": False,
    },
    "Galaxea-Open-World": {
        "domain": "real",
        "source_kind": "lerobot_episode_jsonl",
        "source_path": "Galaxea-Open-World-Dataset",
        "use_template": False,
    },
    "LET-claw": {
        "domain": "real",
        "source_kind": "lerobot_episode_jsonl",
        "source_path": "LejuRobotics/LET-Base-Dataset",
        "use_template": False,
    },
    "RT-1": {
        "domain": "real",
        "source_kind": "lerobot_episode_jsonl",
        "source_path": "/mnt/pfs/data/datasets/lerobotv3/high_priority_80d/RT-1",
        "use_template": False,
    },
    "RoboCOIN": {
        "domain": "real",
        "source_kind": "lerobot_episode_jsonl",
        "source_path": "RoboCOIN",
        "use_template": False,
    },
}


@dataclass(frozen=True)
class TaskRecord:
    """One task-to-source-path relationship in the catalog."""

    id: str
    domain: str
    dataset: str
    native_group: str
    native_scene: str | None
    scene: str
    scene_source: str
    task_type: str
    task_type_source: str
    native_task: str
    source_path: str
    data_role: str
    classification_source: str
    episode_count: int | None = None
    primary_atomic_action: str = "待分类"
    atomic_actions: list[str] = field(default_factory=list)


def normalized(name: str) -> str:
    """Normalize a task name for conservative keyword classification."""

    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def localize_native_scene(scene: str) -> str:
    """Return the Chinese catalog label for a known native scene name.

    The original dataset value remains available in ``native_scene`` for
    provenance and path matching; only the normalized semantic ``scene`` field
    is localized.
    """

    return NATIVE_SCENE_LABELS.get(normalized(scene), scene)


def _has_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def infer_scene(task: str, *, domain: str, context: str = "") -> str:
    """Infer a broad scene from a native task name.

    The returned value is always marked as a heuristic by callers. A native scene,
    when present, takes precedence over this value.
    """

    text = normalized(f"{context} {task}")
    if domain == "sim":
        return "仿真通用场景"
    if _has_any(text, ("trash", "garbage", "dustbin", "waste", "throw away", "sweep")):
        return "清洁与废弃物"
    if _has_any(text, ("towel", "cloth", "laundry", "fold", "hang", "zip clothes")):
        return "织物与整理"
    if _has_any(
        text,
        (
            "battery",
            "gear",
            "machine",
            "switch",
            "panel",
            "conveyor",
            "cylinder",
            "nut",
            "plug",
            "pipe",
            "stamp",
            "tool",
            "foil",
            "peg",
        ),
    ):
        return "工业与精密操作"
    if _has_any(
        text,
        (
            "apple",
            "banana",
            "pear",
            "strawber",
            "fruit",
            "bread",
            "potato",
            "corn",
            "eggplant",
            "egg",
            "pepper",
            "tomato",
            "avocado",
            "plum",
            "kiwi",
            "pumpkin",
            "carrot",
            "lemon",
            "mango",
            "orange",
            "cucumber",
            "vegetable",
            "brocolli",
            "broccoli",
            "dish",
            "fork",
            "spoon",
            "chopstick",
            "bowl",
            "plate",
            "cup",
            "mug",
            "pot",
            "oven",
            "toaster",
            "ricecooker",
            "roaster",
            "seasoning",
            "water",
            "kitchen",
            "hotdog",
            "beef",
            "coke",
        ),
    ):
        return "厨房与餐饮"
    if _has_any(
        text,
        ("drawer", "cabinet", "chest", "box", "shelf", "rack", "holder", "basket", "container", "suitcase"),
    ):
        return "收纳与家具"
    if _has_any(
        text,
        (
            "marker",
            "pen",
            "pencil",
            "cube",
            "block",
            "ball",
            "doll",
            "toy",
            "brick",
            "shape",
            "lamp",
            "laptop",
            "paper",
            "flower",
            "domino",
            "jenga",
            "whiteboard",
            "book",
            "stationery",
        ),
    ):
        return "桌面物体与办公玩具"
    return UNKNOWN_SCENE


def infer_task_type(task: str) -> str:
    """Infer one broad operation type from a native task name."""

    text = normalized(task)
    if _has_any(text, ("clean", "wipe", "brush", "erase", "sweep")):
        return "清洁 / 擦拭"
    if _has_any(text, ("fold", "hang", "zip clothes")):
        return "折叠 / 悬挂"
    if _has_any(text, ("insert", "insertion", "assemble", "plug", "gear", "nut", "stamp", "seal", "peg in")):
        return "插入 / 装配 / 精密定位"
    if _has_any(text, ("sort", "classify", "organize", "tidy", "rearrange", "get the cubes in line", "table bussing")):
        return "分类 / 整理"
    if _has_any(text, ("open", "close", "uncap", "cap lid", "cover pot lid", "unscrew", "screw bottle")):
        return "开合 / 关节物体操作"
    if _has_any(text, ("stack", "piled", "pile", "upright", "flip", "arrange", "topple")):
        return "堆叠 / 翻转 / 摆放"
    if _has_any(text, ("pour", "fluid transfer", "water cup")):
        return "倾倒 / 倒水"
    if _has_any(
        text,
        ("push", "pull", "press", "rotate", "switch", "lamp on", "lamp off", "turn on", "turn off", "toss"),
    ):
        return "推拉 / 按压 / 旋转"
    if _has_any(
        text,
        (
            "pick",
            "place",
            "put",
            "move",
            "take",
            "pack",
            "unpack",
            "throw",
            "collect",
            "gather",
            "lay out",
            "remove",
            "lift",
            "cover beef",
        ),
    ):
        return "拿取 / 放置 / 搬运"
    return UNKNOWN_TASK_TYPE


def _record_id(dataset: str, native_group: str, task: str, source_path: str, data_role: str) -> str:
    key = "\x1f".join((dataset, native_group, task, source_path, data_role))
    return hashlib.blake2s(key.encode("utf-8"), digest_size=10).hexdigest()


def _make_record(
    *,
    root: Path,
    domain: str,
    dataset: str,
    task: str,
    source: Path,
    data_role: str,
    native_group: str = UNKNOWN_GROUP,
    native_scene: str | None = None,
    classification_source: str = "native_name",
    episode_count: int | None = None,
    context: str = "",
    scene_override: str | None = None,
    task_type_override: str | None = None,
) -> TaskRecord:
    source_path = source.relative_to(root).as_posix()
    localized_native_scene = localize_native_scene(native_scene) if native_scene else None
    scene = scene_override or localized_native_scene or infer_scene(task, domain=domain, context=context)
    scene_source = "manual" if scene_override else "native_name" if native_scene else "name_heuristic"
    return TaskRecord(
        id=_record_id(dataset, native_group, task, source_path, data_role),
        domain=domain,
        dataset=dataset,
        native_group=native_group,
        native_scene=native_scene,
        scene=scene,
        scene_source=scene_source,
        task_type=task_type_override or infer_task_type(task),
        task_type_source="manual" if task_type_override else "name_heuristic",
        native_task=task,
        source_path=source_path,
        data_role=data_role,
        classification_source=classification_source,
        episode_count=episode_count,
    )


def _visible_directories(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        child for child in path.iterdir() if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")
    )


def _count_visible_directories(path: Path) -> int | None:
    """Count one directory level without sorting or following symlinks."""

    try:
        with os.scandir(path) as entries:
            return sum(entry.is_dir(follow_symlinks=False) and not entry.name.startswith(".") for entry in entries)
    except OSError:
        return None


def _count_robomind_v2_episodes(task_dir: Path) -> int | None:
    """Count native episode directories without descending into their payloads."""

    container = task_dir / "success_episodes"
    if not container.is_dir() or container.is_symlink():
        return None
    return _count_visible_directories(container)


def _count_robomind_failure_episodes(task_dir: Path) -> int | None:
    """Count ``task / failure reason / episode`` directories."""

    reason_dirs = _visible_directories(task_dir)
    if not reason_dirs:
        return None
    count = sum(_count_visible_directories(reason_dir) or 0 for reason_dir in reason_dirs)
    return count or None


def _count_lerobot_metadata_episodes(task_dir: Path) -> int | None:
    """Read the small LeRobot info file instead of enumerating payload shards."""

    info_path = task_dir / "meta" / "info.json"
    if not info_path.is_file() or info_path.is_symlink():
        return None
    try:
        if info_path.stat().st_size > 1_000_000:
            return None
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    count = payload.get("total_episodes") if isinstance(payload, dict) else None
    return count if type(count) is int and count >= 0 else None


def _count_freetacman_episodes(task_dir: Path) -> int | None:
    """FreeTacMan stores one trajectory CSV per episode beside its videos."""

    with os.scandir(task_dir) as entries:
        count = sum(entry.is_file(follow_symlinks=False) and entry.name.endswith("_traj.csv") for entry in entries)
    return count or None


def _count_exumi_episodes(task_dir: Path) -> int | None:
    """exUMI stores one zero-padded MP4 per episode at the task root."""

    with os.scandir(task_dir) as entries:
        count = sum(
            entry.is_file(follow_symlinks=False) and re.fullmatch(r"\d+\.mp4", entry.name) is not None
            for entry in entries
        )
    return count or None


def _scan_robomind_v1(root: Path) -> list[TaskRecord]:
    base = root / "RoboMIND"
    records: list[TaskRecord] = []
    archive_pattern = re.compile(r"(?P<task>.+)\.tar\.gz\.part-[^.]+$")
    seen: set[tuple[str, str]] = set()
    for part in sorted(base.glob("benchmark*_compressed/*/*.tar.gz.part-*")):
        match = archive_pattern.fullmatch(part.name)
        if not match:
            continue
        group = part.parent.relative_to(base).as_posix()
        task = match.group("task")
        if (group, task) in seen:
            continue
        seen.add((group, task))
        domain = "sim" if "/h5_sim" in f"/{group}" else "real"
        records.append(
            _make_record(
                root=root,
                domain=domain,
                dataset="RoboMIND",
                native_group=group,
                task=task,
                source=part.parent / f"{task}.tar.gz.part-*",
                data_role="success archive",
                context=group,
            )
        )
    failure_base = base / "failure_data"
    for task_dir in _visible_directories(failure_base):
        records.append(
            _make_record(
                root=root,
                domain="real",
                dataset="RoboMIND",
                native_group="failure_data",
                task=task_dir.name,
                source=task_dir,
                data_role="failure directory",
                episode_count=_count_robomind_failure_episodes(task_dir),
            )
        )
    return records


def _scan_robomind_v2(root: Path) -> list[TaskRecord]:
    base = root / "RoboMINDv2"
    records: list[TaskRecord] = []
    for group_dir in _visible_directories(base):
        domain = "sim" if "sim" in normalized(group_dir.name).split() else "real"
        for task_dir in _visible_directories(group_dir):
            records.append(
                _make_record(
                    root=root,
                    domain=domain,
                    dataset="RoboMINDv2",
                    native_group=group_dir.name,
                    task=task_dir.name,
                    source=task_dir,
                    data_role="task directory",
                    context=group_dir.name,
                    episode_count=_count_robomind_v2_episodes(task_dir),
                )
            )
    return records


def _archive_task_name(name: str) -> str | None:
    match = re.fullmatch(r"(.+?)\.tar\.gz(?:\.part-[^.]+)?", name)
    return match.group(1) if match else None


def _scan_fastumi(root: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    archive_base = root / "IPEC-COMMUNITY" / "FastUMI-Data"
    seen: set[str] = set()
    if archive_base.is_dir():
        for path in sorted(archive_base.iterdir()):
            if not path.is_file():
                continue
            task = _archive_task_name(path.name)
            if task is None or task in seen:
                continue
            seen.add(task)
            records.append(
                _make_record(
                    root=root,
                    domain="umi",
                    dataset="FastUMI-Data",
                    task=task,
                    source=archive_base / f"{task}.tar.gz*",
                    data_role="task archive",
                )
            )
    lerobot_base = root / "IPEC-COMMUNITY" / "FastUMI_100k_lerobot"
    for arm_dir in _visible_directories(lerobot_base):
        for task_dir in _visible_directories(arm_dir):
            records.append(
                _make_record(
                    root=root,
                    domain="umi",
                    dataset="FastUMI-100K",
                    native_group=arm_dir.name,
                    task=task_dir.name,
                    source=task_dir,
                    data_role="task directory",
                    episode_count=_count_lerobot_metadata_episodes(task_dir),
                )
            )
    return records


def _clean_metadata_instruction(value: Any) -> str:
    """Normalize metadata text for display and stable task aggregation."""

    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _scan_hifi_umi(root: Path) -> list[TaskRecord]:
    """Aggregate HiFi-UMI metadata into one catalog row per instruction.

    ``chunk-* / part-*`` are technical storage partitions rather than semantic
    tasks.  Emitting every episode would also make the browser unusable, so
    identical instruction text is grouped and its episode occurrence count is
    retained on the task row.
    """

    base = root / "HiFi-UMI-2K-main"
    if not base.is_dir() or base.is_symlink():
        return []
    episode_files = sorted(base.glob("chunk-*/part-*/meta/episodes/chunk-*/file-*.parquet"))
    if not episode_files:
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    task_counts: Counter[str] = Counter()
    display_tasks: dict[str, str] = {}
    embodiment = UNKNOWN_GROUP
    for path in episode_files:
        if path.is_symlink():
            continue
        if embodiment == UNKNOWN_GROUP:
            info_path = path.parents[2] / "info.json"
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                robot_type = info.get("robot_type") if isinstance(info, dict) else None
                if isinstance(robot_type, str) and robot_type.strip():
                    embodiment = robot_type.strip()
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        try:
            parquet = pq.ParquetFile(path)
            if "tasks" not in parquet.schema_arrow.names:
                continue
            for batch in parquet.iter_batches(columns=["tasks"], batch_size=4096):
                for tasks in batch.column(0).to_pylist():
                    if isinstance(tasks, str):
                        tasks = [tasks]
                    if not isinstance(tasks, list):
                        continue
                    for value in tasks:
                        task = _clean_metadata_instruction(value)
                        if not task:
                            continue
                        key = task.casefold()
                        display_tasks.setdefault(key, task)
                        task_counts[key] += 1
        except (OSError, ValueError):
            continue

    return [
        _make_record(
            root=root,
            domain="umi",
            dataset="HiFi-UMI-2K",
            native_group=embodiment,
            task=display_tasks[key],
            source=base,
            data_role="episode instruction aggregate",
            classification_source="metadata_instruction",
            episode_count=task_counts[key],
            context=embodiment,
        )
        for key in sorted(task_counts)
    ]


def _scan_instruction_inventory_tasks(root: Path, inventory_path: Path | None) -> list[TaskRecord]:
    """Build compact task rows for metadata-only datasets from the JSONL inventory."""

    if inventory_path is None or not inventory_path.is_file() or inventory_path.is_symlink():
        return []
    task_counts: Counter[tuple[str, str]] = Counter()
    display_tasks: dict[tuple[str, str], str] = {}
    episode_counts: Counter[tuple[str, str]] = Counter()
    try:
        with inventory_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                dataset = row.get("dataset")
                spec = INVENTORY_TASK_SOURCES.get(dataset)
                if spec is None or row.get("source_kind") != spec["source_kind"]:
                    continue
                raw_task = row.get("instruction_template") if spec["use_template"] else row.get("instruction")
                task = _clean_metadata_instruction(raw_task)
                if not task:
                    continue
                # Something-Something ships both bracketed and plain variants
                # of each of its 174 action templates. They are presentation
                # variants of the same class, so ignore placeholder brackets
                # and punctuation when building the aggregation key.
                aggregation_key = (normalized(task) or task.casefold()) if spec["use_template"] else task.casefold()
                key = (dataset, aggregation_key)
                display_tasks.setdefault(key, task)
                task_counts[key] += 1
                if row.get("occurrence_kind") in {"episode", "video"}:
                    episode_counts[key] += 1
    except (OSError, UnicodeError):
        return []

    records: list[TaskRecord] = []
    for key in sorted(task_counts):
        dataset, _ = key
        spec = INVENTORY_TASK_SOURCES[dataset]
        task = display_tasks[key]
        source_path = str(spec["source_path"])
        source = Path(source_path)
        if not source.is_absolute():
            source = root / source
        if not source.exists():
            continue
        episode_count = episode_counts[key] or None
        records.append(
            TaskRecord(
                id=_record_id(dataset, UNKNOWN_GROUP, task, source_path, "metadata instruction aggregate"),
                domain=str(spec["domain"]),
                dataset=dataset,
                native_group=UNKNOWN_GROUP,
                native_scene=None,
                scene=infer_scene(task, domain=str(spec["domain"]), context=dataset),
                scene_source="name_heuristic",
                task_type=infer_task_type(task),
                task_type_source="name_heuristic",
                native_task=task,
                source_path=source_path,
                data_role="episode instruction aggregate" if episode_count is not None else "task metadata aggregate",
                classification_source="metadata_instruction_inventory",
                episode_count=episode_count,
            )
        )
    return records


def _scan_direct_task_dataset(
    root: Path,
    *,
    relative_path: str,
    dataset: str,
    domain: str = "real",
    episode_counter: Callable[[Path], int | None] | None = None,
) -> list[TaskRecord]:
    base = root / relative_path
    return [
        _make_record(
            root=root,
            domain=domain,
            dataset=dataset,
            task=task_dir.name,
            source=task_dir,
            data_role="task directory",
            episode_count=episode_counter(task_dir) if episode_counter else None,
        )
        for task_dir in _visible_directories(base)
    ]


def _scan_touch(root: Path) -> tuple[list[TaskRecord], list[dict[str, Any]]]:
    base = root / "binghaohuang-robot" / "touch_in_the_wild-dataset"
    records: list[TaskRecord] = []
    unresolved: list[dict[str, Any]] = []
    for group in ("four_tasks", "indoor_data"):
        for task_dir in _visible_directories(base / group):
            records.append(
                _make_record(
                    root=root,
                    domain="umi",
                    dataset="Touch-in-the-Wild",
                    native_group=group,
                    task=task_dir.name,
                    source=task_dir,
                    data_role="task directory",
                )
            )
    for scene_dir in _visible_directories(base / "in_the_wild_data"):
        unresolved.append(
            {
                "dataset": "Touch-in-the-Wild",
                "domain": "umi",
                "source_path": scene_dir.relative_to(root).as_posix(),
                "native_scene": scene_dir.name,
                "available_level": "scene",
                "reason": "目录只提供采集场景; 具体任务需要从内部 metadata 提取。",
            }
        )
    return records, unresolved


def _realomin_native_fields(parts: tuple[str, ...]) -> tuple[str, str | None]:
    if len(parts) >= 5 and parts[0] == "[STAGE 3]" and parts[1] == "domestic_services":
        return parts[-2], parts[2]
    if len(parts) >= 2:
        return parts[-2], None
    return "Stage2", None


def _scan_realomin(root: Path) -> list[TaskRecord]:
    base = root / "genrobot2025" / "10Kh-RealOmin-OpenData"
    records: list[TaskRecord] = []
    if not base.is_dir():
        return records
    for current, dir_names, _ in os.walk(base):
        dir_names[:] = [name for name in dir_names if not name.startswith(".")]
        episode_names = [name for name in dir_names if re.fullmatch(r"[0-9]+", name)]
        if not episode_names:
            continue
        task_dir = Path(current)
        relative_parts = task_dir.relative_to(base).parts
        if not relative_parts:
            continue
        native_group, native_scene = _realomin_native_fields(relative_parts)
        records.append(
            _make_record(
                root=root,
                domain="real",
                dataset="10Kh-RealOmin",
                native_group=native_group,
                native_scene=native_scene,
                task=relative_parts[-1],
                source=task_dir,
                data_role="task directory",
                episode_count=len(episode_names),
                context="/".join(relative_parts),
            )
        )
        dir_names[:] = [name for name in dir_names if name not in episode_names]
    return sorted(records, key=lambda record: record.source_path)


def _scan_egodex(root: Path) -> list[TaskRecord]:
    base = root / "EgoDex"
    if not base.is_dir():
        return []
    return [
        _make_record(
            root=root,
            domain="ego",
            dataset="EgoDex",
            task=archive.stem,
            source=archive,
            data_role="task archive",
        )
        for archive in sorted(base.glob("*.tar"))
    ]


def _scan_vqa(root: Path) -> list[TaskRecord]:
    """Add one shallow record for each downloaded, allow-listed VQA dataset root."""

    records: list[TaskRecord] = []
    for category, dataset, relative_paths, _ in VQA_REFERENCE_SOURCES:
        source = next(
            (
                root / relative_path
                for relative_path in relative_paths
                if (root / relative_path).is_dir() and not (root / relative_path).is_symlink()
            ),
            None,
        )
        if source is None:
            continue
        records.append(
            _make_record(
                root=root,
                domain="vqa",
                dataset=dataset,
                native_group="飞书文档分类",
                task=dataset,
                source=source,
                data_role="dataset directory",
                classification_source="reference_document",
                scene_override="视觉问答与空间理解",
                task_type_override=category,
            )
        )
    return records


def _reference_datasets(root: Path) -> list[dict[str, Any]]:
    """Describe document-catalog datasets without claiming missing paths exist."""

    references: list[dict[str, Any]] = []
    definitions = [
        ("umi", "UMI", dataset, document_name, paths, source_reference)
        for dataset, document_name, paths, source_reference in UMI_REFERENCE_SOURCES
    ] + [
        ("vqa", category, dataset, dataset, paths, source_reference)
        for category, dataset, paths, source_reference in VQA_REFERENCE_SOURCES
    ]
    for domain, category, dataset, document_name, relative_paths, source_reference in definitions:
        available_path = next(
            (
                relative_path
                for relative_path in relative_paths
                if (root / relative_path).is_dir() and not (root / relative_path).is_symlink()
            ),
            None,
        )
        references.append(
            {
                "dataset": dataset,
                "document_name": document_name,
                "domain": domain,
                "category": category,
                "availability": "downloaded" if available_path else "not_downloaded",
                "source_path": available_path,
                "expected_path": relative_paths[0] if relative_paths else None,
                "source_reference": source_reference,
                "classification_source": "feishu_document"
                if domain == "umi"
                else "feishu_taxonomy_and_processing_ledger",
            }
        )
    return references


def _unresolved_sources(root: Path, *, resolved_datasets: set[str] | None = None) -> list[dict[str, Any]]:
    resolved_datasets = resolved_datasets or set()
    definitions = (
        (
            "HiFi-UMI-2K",
            "umi",
            "HiFi-UMI-2K-main",
            "chunk / part",
            "chunk-* 和 part-* 是存储分块, 不是任务。",
        ),
        (
            "RoboMIND-2.0-full LeRobot v3",
            "real",
            "lerobotv3/high_priority_80d/RoboMIND-2.0-full",
            "embodiment/camera schema",
            "任务需要从子集 metadata 读取。",
        ),
        (
            "Hy-Embodied-0.5-VLA-Data",
            "real",
            "tencent/Hy-Embodied-0.5-VLA-Data",
            "table",
            "table_* 是 Lance 存储表; 任务信息需要从 metadata 读取。",
        ),
        (
            "Egocentric-100K",
            "ego",
            "Egocentric-100K",
            "factory / worker",
            "一级目录是采集组织结构, 不是任务。",
        ),
        (
            "EgoVerse",
            "ego",
            "EgoVerse",
            "episode UUID",
            "任务名位于 episode 的 zarr.json attributes.task_name。",
        ),
        (
            "Something-Something-V2",
            "ego",
            "Something-Something-V2",
            "video id",
            "任务分类位于 labels zip 的 template 字段。",
        ),
    )
    unresolved: list[dict[str, Any]] = []
    for dataset, domain, relative_path, level, reason in definitions:
        if dataset in resolved_datasets:
            continue
        path = root / relative_path
        if path.exists():
            unresolved.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "source_path": path.relative_to(root).as_posix(),
                    "native_scene": None,
                    "available_level": level,
                    "reason": reason,
                }
            )
    return unresolved


def scan_catalog(
    root: Path,
    *,
    generated_at: str | None = None,
    instruction_inventory: Path | None = None,
) -> dict[str, Any]:
    """Scan supported native task-name layouts beneath ``root``."""

    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist or is not a directory: {root}")

    records: list[TaskRecord] = []
    records.extend(_scan_robomind_v1(root))
    records.extend(_scan_robomind_v2(root))
    records.extend(_scan_fastumi(root))
    records.extend(_scan_hifi_umi(root))
    records.extend(_scan_instruction_inventory_tasks(root, instruction_inventory))
    records.extend(
        _scan_direct_task_dataset(
            root,
            relative_path="OpenDriveLab/FreeTacMan",
            dataset="FreeTacMan",
            domain="umi",
            episode_counter=_count_freetacman_episodes,
        )
    )
    records.extend(
        _scan_direct_task_dataset(
            root,
            relative_path="silicx/exUMI",
            dataset="exUMI",
            domain="umi",
            episode_counter=_count_exumi_episodes,
        )
    )
    touch_records, touch_unresolved = _scan_touch(root)
    records.extend(touch_records)
    records.extend(_scan_realomin(root))
    records.extend(_scan_egodex(root))
    records.extend(_scan_vqa(root))

    records.sort(
        key=lambda record: (record.domain, record.dataset, record.native_group, record.native_task, record.source_path)
    )
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = [record_id for record_id, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"catalog contains duplicate stable ids: {duplicates[:3]}")

    domain_counts = Counter(record.domain for record in records)
    datasets = sorted({record.dataset for record in records})
    unresolved = touch_unresolved + _unresolved_sources(root, resolved_datasets=set(datasets))
    reference_datasets = _reference_datasets(root)
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": root.as_posix(),
        "hierarchy": [
            {"key": "domain", "label": "数据域"},
            {"key": "scene", "label": "场景"},
            {"key": "dataset", "label": "数据集"},
            {"key": "task_type", "label": "任务类型"},
            {"key": "primary_atomic_action", "label": "原子动作"},
            {"key": "native_task", "label": "具体任务"},
        ],
        "domain_labels": DOMAIN_LABELS,
        "stats": {
            "records": len(records),
            "unique_tasks": len({(record.dataset, record.native_task) for record in records}),
            "datasets": len(datasets),
            "dataset_names": datasets,
            "by_domain": {domain: domain_counts.get(domain, 0) for domain in DOMAIN_LABELS},
            "unresolved_sources": len(unresolved),
            "reference_datasets": len(reference_datasets),
            "reference_datasets_downloaded": sum(
                item["availability"] == "downloaded" for item in reference_datasets
            ),
            "records_with_episode_count": sum(record.episode_count is not None for record in records),
            "episodes_known_total": sum(record.episode_count or 0 for record in records),
        },
        "records": [asdict(record) for record in records],
        "unresolved": unresolved,
        "reference_datasets": reference_datasets,
    }


def write_catalog(catalog: dict[str, Any], output: Path) -> None:
    """Write a catalog atomically as UTF-8 JSON."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
