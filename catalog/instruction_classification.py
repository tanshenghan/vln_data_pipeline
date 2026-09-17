# ruff: noqa: RUF001
"""Classify unique instruction text into a compact, fixed semantic taxonomy.

The classifier operates on ``instruction_hash`` rather than every occurrence.
Something-Something-V2 labels are additionally collapsed by their native action
template before an API call, then expanded back to every unique instruction.
Results remain a sidecar index and never rewrite source metadata or the inventory.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

CLASSIFICATION_SCHEMA_VERSION = 1
TAXONOMY_VERSION = "compact-scenes-atomic-v3"

# Keep this deliberately small. These labels are the website's second-level
# navigation and should describe broad environments, not individual objects.
SCENES = (
    "厨房与餐饮",
    "家居与日常生活",
    "桌面与办公",
    "清洁与维护",
    "工业与工具",
    "织物与穿戴",
    "户外与公共环境",
    "通用操作场景",
    "仿真通用场景",
    "视觉问答与空间理解",
)

TASK_TYPES = (
    "拿取 / 放置 / 搬运",
    "开合 / 启闭",
    "推拉 / 按压 / 旋转",
    "插入 / 装配 / 连接",
    "倾倒 / 填充 / 转移",
    "清洁 / 擦拭 / 废弃",
    "分类 / 整理 / 收纳",
    "折叠 / 穿戴 / 织物处理",
    "切割 / 烹饪 / 食物处理",
    "堆叠 / 排列 / 翻转",
    "视觉问答 / 空间理解",
    "其他 / 复合任务",
)

# Atomic actions are intentionally more granular than scenes. A task keeps an
# ordered sequence but has exactly one primary action for use as a tree level.
ATOMIC_ACTIONS = (
    "抓取",
    "放置",
    "移动",
    "持握 / 保持",
    "打开",
    "关闭",
    "推动",
    "拉动",
    "按压",
    "旋转 / 滚动",
    "插入 / 连接",
    "拔出 / 拆卸",
    "倾倒 / 填充",
    "舀取 / 涂抹 / 撒布",
    "清洁 / 擦拭",
    "折叠 / 展开",
    "扣合 / 系紧",
    "切割 / 加工",
    "烹饪 / 制作",
    "排列 / 堆叠 / 翻转",
    "分类 / 整理",
    "接触 / 敲击",
    "挤压 / 形变 / 撕裂",
    "释放 / 抛掷",
    "书写 / 输入",
    "观察 / 判断",
    "其他",
)

SPECIAL_DATASET_SSV2 = "Something-Something-V2"
FORCED_SCENE_SIM = "仿真通用场景"
FORCED_SCENE_VQA = "视觉问答与空间理解"
FORCED_SCENE_SSV2 = "通用操作场景"
DEFAULT_REVIEW_THRESHOLD = 0.72
MAX_CONTEXT_VALUES = 12


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _bounded_add(values: set[str], value: Any) -> None:
    text = _clean(value)
    if text and len(values) < MAX_CONTEXT_VALUES:
        values.add(text)


@dataclass
class UniqueInstruction:
    instruction_hash: str
    instruction: str
    occurrence_count: int = 0
    domains: Counter[str] = field(default_factory=Counter)
    datasets: Counter[str] = field(default_factory=Counter)
    source_kinds: Counter[str] = field(default_factory=Counter)
    native_scenes: set[str] = field(default_factory=set)
    templates: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any]) -> None:
        self.occurrence_count += 1
        for field_name, counter in (
            ("domain", self.domains),
            ("dataset", self.datasets),
            ("source_kind", self.source_kinds),
        ):
            value = _clean(row.get(field_name))
            if value:
                counter[value] += 1
        _bounded_add(self.native_scenes, row.get("native_scene"))
        _bounded_add(self.templates, row.get("instruction_template"))

    @property
    def dominant_domain(self) -> str:
        return self.domains.most_common(1)[0][0] if self.domains else ""

    @property
    def dataset_names(self) -> list[str]:
        return sorted(self.datasets)

    @property
    def template(self) -> str | None:
        return sorted(self.templates)[0] if len(self.templates) == 1 else None


@dataclass(frozen=True)
class ClassificationUnit:
    unit_id: str
    text: str
    member_hashes: tuple[str, ...]
    context: str
    forced_scene: str | None = None
    source: str = "llm"


def load_unique_instructions(path: str | Path) -> dict[str, UniqueInstruction]:
    """Stream the inventory and aggregate traceable context per instruction hash."""

    unique: dict[str, UniqueInstruction] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            digest = _clean(row.get("instruction_hash"))
            instruction = _clean(row.get("instruction"))
            if not digest or not instruction:
                raise ValueError(f"missing instruction_hash/instruction at {path}:{line_number}")
            item = unique.get(digest)
            if item is None:
                item = UniqueInstruction(digest, instruction)
                unique[digest] = item
            item.add(row)
    return unique


def forced_scene_for(item: UniqueInstruction) -> str | None:
    domains = set(item.domains)
    datasets = set(item.datasets)
    if domains == {"vqa"}:
        return FORCED_SCENE_VQA
    if domains == {"sim"}:
        return FORCED_SCENE_SIM
    if datasets == {SPECIAL_DATASET_SSV2}:
        return FORCED_SCENE_SSV2
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", item.instruction.casefold())
    cleaning_terms = (
        "clean",
        "wipe",
        "sweep",
        "vacuum",
        "dust",
        "trash",
        "garbage",
        "waste",
        "debris",
        "清洁",
        "清理",
        "擦拭",
        "打扫",
        "垃圾",
    )
    kitchen_terms = (
        "bowl",
        "dish",
        "plate",
        "cup",
        "pot",
        "pan",
        "餐具",
        "碗",
        "盘",
        "杯",
        "锅",
    )
    if any(term in text for term in cleaning_terms):
        if any(term in text for term in kitchen_terms):
            return "厨房与餐饮"
        return "清洁与维护"
    return None


def _unit_context(item: UniqueInstruction) -> str:
    parts = []
    if item.datasets:
        parts.append(f"数据集={','.join(item.dataset_names)}")
    if item.domains:
        parts.append(f"数据域={','.join(sorted(item.domains))}")
    if item.native_scenes:
        parts.append(f"原生场景={','.join(sorted(item.native_scenes))}")
    return "；".join(parts)


def build_classification_units(
    unique: dict[str, UniqueInstruction],
) -> tuple[list[ClassificationUnit], dict[str, dict[str, Any]]]:
    """Build API units and deterministic rows.

    VQA dataset-name placeholders need no model call. SSV2 instructions sharing
    the same native template share one API call while retaining independent rows.
    """

    deterministic: dict[str, dict[str, Any]] = {}
    template_members: dict[str, list[str]] = {}
    regular: list[ClassificationUnit] = []

    for digest, item in unique.items():
        forced_scene = forced_scene_for(item)
        if forced_scene == FORCED_SCENE_VQA:
            deterministic[digest] = {
                "scene": FORCED_SCENE_VQA,
                "task_type": "视觉问答 / 空间理解",
                "primary_atomic_action": "观察 / 判断",
                "atomic_actions": ["观察 / 判断"],
                "confidence": 1.0,
                "reason": "VQA 数据域固定映射",
                "classification_source": "domain_rule",
            }
            continue

        if forced_scene == FORCED_SCENE_SSV2 and item.template:
            template_members.setdefault(item.template, []).append(digest)
            continue

        regular.append(
            ClassificationUnit(
                unit_id=digest,
                text=item.instruction,
                member_hashes=(digest,),
                context=_unit_context(item),
                forced_scene=forced_scene,
                source="llm_forced_scene" if forced_scene else "llm",
            )
        )

    template_units = []
    for template, members in sorted(template_members.items()):
        template_digest = hashlib.blake2s(template.encode("utf-8"), digest_size=12).hexdigest()
        template_units.append(
            ClassificationUnit(
                unit_id=f"template:{template_digest}",
                text=template,
                member_hashes=tuple(sorted(members)),
                context=f"数据集={SPECIAL_DATASET_SSV2}；这是动作模板，方括号表示任意物体",
                forced_scene=FORCED_SCENE_SSV2,
                source="llm_template",
            )
        )
    return regular + template_units, deterministic


def _system_prompt() -> str:
    scene_lines = "\n".join(f"- {label}" for label in SCENES)
    task_lines = "\n".join(f"- {label}" for label in TASK_TYPES)
    atomic_lines = "\n".join(f"- {label}" for label in ATOMIC_ACTIONS)
    return f"""你是机器人数据集指令分类器。只按给定固定枚举分类，不得创造新标签。

场景是宽泛环境，不是动作或物体类别。固定场景（共 {len(SCENES)} 类）：
{scene_lines}

任务类型：
{task_lines}

原子动作：
{atomic_lines}

场景判定规则：
1. 做饭、餐具、食物处理、饮料等归“厨房与餐饮”。
2. 卧室、客厅、浴室、宠物照料及一般家庭活动归“家居与日常生活”。
3. 文具、键盘、书本、文件以及明确桌面办公活动归“桌面与办公”。
4. 垃圾处理、清扫、擦洗、设备保养在没有更明确地点时优先归“清洁与维护”；清洗餐具仍归“厨房与餐饮”。
5. 装配线、机器、维修、工具、零件操作归“工业与工具”。
6. 衣物、毛巾、布料、鞋帽的处理归“织物与穿戴”。
7. 校园、商店、道路、花园等非家庭公共或户外活动归“户外与公共环境”。
8. 只有物体与动作、无法可靠推出环境时归“通用操作场景”，不要猜测具体房间。
9. forced_scene 非空时必须原样采用。

任务类型按核心目标选一类；长指令包含多个必要步骤时选“其他 / 复合任务”。
atomic_actions 记录指令明确要求的有序动作，最多 6 个，不要加入指令没有表达的机器人底层动作。
primary_atomic_action 是任务最终目标或最能区分任务的一个动作，用于树状目录分组，且必须出现在 atomic_actions 中。
例：“拿起杯子并放到托盘”可输出 [“抓取”,“移动”,“放置”]，主原子动作是“放置”。
拉拉链、系鞋带、扣紧带子统一使用“扣合 / 系紧”。
蒸、烤、煎、制作食物使用“烹饪 / 制作”；打字、写字使用“书写 / 输入”。
持住不动使用“持握 / 保持”；投掷或主动释放使用“释放 / 抛掷”；触碰或敲击使用“接触 / 敲击”。
弯折、挤压、拉伸、撕裂使用“挤压 / 形变 / 撕裂”；舀取、涂抹、撒布使用“舀取 / 涂抹 / 撒布”。
confidence 为 0 到 1。reason 使用不超过 18 个中文字的简短依据。
输入是 JSON 数组。输出必须是同样数量的纯 JSON 数组，不要 Markdown，不要额外文字；每项格式：
{{"id":"原id","scene":"固定场景之一","task_type":"固定任务类型之一","primary_atomic_action":"固定原子动作之一","atomic_actions":["固定原子动作"],"confidence":0.0,"reason":"简短依据"}}"""


def _extract_response_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for output in response.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def parse_classification_response(
    response: dict[str, Any], units: Iterable[ClassificationUnit]
) -> dict[str, dict[str, Any]]:
    units_by_id = {unit.unit_id: unit for unit in units}
    text = _extract_response_text(response)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("model response does not contain a JSON array")
    try:
        values = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model JSON: {exc}") from exc
    if not isinstance(values, list):
        raise ValueError("model response must be a JSON array")

    parsed: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("model result item is not an object")
        unit_id = _clean(value.get("id"))
        unit = units_by_id.get(unit_id)
        if unit is None or unit_id in parsed:
            raise ValueError(f"unexpected or duplicate result id: {unit_id}")
        scene = _clean(value.get("scene"))
        task_type = _clean(value.get("task_type"))
        if scene not in SCENES:
            raise ValueError(f"invalid scene for {unit_id}: {scene}")
        if task_type not in TASK_TYPES:
            raise ValueError(f"invalid task type for {unit_id}: {task_type}")
        primary_atomic_action = _clean(value.get("primary_atomic_action"))
        atomic_actions_value = value.get("atomic_actions")
        if primary_atomic_action not in ATOMIC_ACTIONS:
            raise ValueError(f"invalid primary atomic action for {unit_id}: {primary_atomic_action}")
        if not isinstance(atomic_actions_value, list) or not 1 <= len(atomic_actions_value) <= 6:
            raise ValueError(f"invalid atomic action sequence for {unit_id}")
        atomic_actions = [_clean(action) for action in atomic_actions_value]
        if any(action not in ATOMIC_ACTIONS for action in atomic_actions):
            raise ValueError(f"invalid atomic action sequence for {unit_id}")
        if primary_atomic_action not in atomic_actions:
            raise ValueError(f"primary atomic action missing from sequence for {unit_id}")
        atomic_actions = list(dict.fromkeys(atomic_actions))
        if unit.forced_scene and scene != unit.forced_scene:
            raise ValueError(f"forced scene mismatch for {unit_id}: {scene}")
        try:
            confidence = min(1.0, max(0.0, float(value.get("confidence"))))
        except (TypeError, ValueError):
            raise ValueError(f"invalid confidence for {unit_id}") from None
        parsed[unit_id] = {
            "scene": scene,
            "task_type": task_type,
            "primary_atomic_action": primary_atomic_action,
            "atomic_actions": atomic_actions,
            "confidence": round(confidence, 4),
            "reason": _clean(value.get("reason"))[:80],
        }

    missing = set(units_by_id) - set(parsed)
    if missing:
        raise ValueError(f"model omitted {len(missing)} result ids")
    return parsed


class ResponsesAPIClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/v1/responses"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def classify(self, units: list[ClassificationUnit]) -> dict[str, dict[str, Any]]:
        input_items = [
            {
                "id": unit.unit_id,
                "text": unit.text,
                "context": unit.context,
                "forced_scene": unit.forced_scene,
            }
            for unit in units
        ]
        payload = {
            "model": self.model,
            "instructions": _system_prompt(),
            "input": json.dumps(input_items, ensure_ascii=False, separators=(",", ":")),
            "max_output_tokens": max(1200, min(12000, len(units) * 150)),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "BetaData-instruction-classifier/1",
            },
        )
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.load(response)
                return parse_classification_response(value, units)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise RuntimeError(f"classification request failed after retries: {last_error}")


def _batches(values: list[ClassificationUnit], size: int) -> Iterator[list[ClassificationUnit]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def classify_units_resilient(
    client: ResponsesAPIClient, units: list[ClassificationUnit]
) -> dict[str, dict[str, Any]]:
    """Split a repeatedly malformed model batch so one response cannot stop the run."""

    try:
        return client.classify(units)
    except RuntimeError:
        if len(units) <= 1:
            raise
        middle = len(units) // 2
        return {
            **classify_units_resilient(client, units[:middle]),
            **classify_units_resilient(client, units[middle:]),
        }


def _classification_row(
    item: UniqueInstruction,
    result: dict[str, Any],
    *,
    source: str,
    model: str | None,
    review_threshold: float,
    classified_at: str,
) -> dict[str, Any]:
    confidence = float(result["confidence"])
    task_type = str(result["task_type"])
    primary_atomic_action = str(result["primary_atomic_action"])
    return {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "instruction_hash": item.instruction_hash,
        "instruction": item.instruction,
        "scene": result["scene"],
        "task_type": task_type,
        "primary_atomic_action": primary_atomic_action,
        "atomic_actions": list(result["atomic_actions"]),
        "confidence": confidence,
        "needs_review": (
            confidence < review_threshold
            or task_type == "其他 / 复合任务"
            or primary_atomic_action == "其他"
        ),
        "reason": result.get("reason") or "",
        "classification_source": source,
        "model": model,
        "classified_at": classified_at,
        "occurrence_count": item.occurrence_count,
        "datasets": item.dataset_names,
        "domains": sorted(item.domains),
        "native_scenes": sorted(item.native_scenes),
        "instruction_template": item.template,
    }


def load_checkpoint(
    path: str | Path, *, model: str, taxonomy_version: str = TAXONOMY_VERSION
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    checkpoint = Path(path)
    if not checkpoint.exists():
        return rows
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at line {line_number}: {exc}") from exc
            if row.get("taxonomy_version") != taxonomy_version:
                continue
            if row.get("classification_source") != "domain_rule" and row.get("model") != model:
                continue
            digest = _clean(row.get("instruction_hash"))
            if digest:
                rows[digest] = row
    return rows


def _write_atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _summary(rows: Iterable[dict[str, Any]], *, model: str, api_calls: int) -> dict[str, Any]:
    row_list = list(rows)
    scene_unique = Counter(row["scene"] for row in row_list)
    scene_occurrences = Counter()
    task_unique = Counter(row["task_type"] for row in row_list)
    task_occurrences = Counter()
    primary_atomic_unique = Counter(row["primary_atomic_action"] for row in row_list)
    primary_atomic_occurrences = Counter()
    atomic_membership_unique = Counter()
    sources = Counter(row["classification_source"] for row in row_list)
    for row in row_list:
        occurrences = int(row["occurrence_count"])
        scene_occurrences[row["scene"]] += occurrences
        task_occurrences[row["task_type"]] += occurrences
        primary_atomic_occurrences[row["primary_atomic_action"]] += occurrences
        for atomic_action in row["atomic_actions"]:
            atomic_membership_unique[atomic_action] += 1
    return {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "generated_at": now_iso(),
        "model": model,
        "unique_instructions": len(row_list),
        "occurrences": sum(int(row["occurrence_count"]) for row in row_list),
        "needs_review": sum(bool(row["needs_review"]) for row in row_list),
        "api_calls": api_calls,
        "scenes": list(SCENES),
        "task_types": list(TASK_TYPES),
        "atomic_actions": list(ATOMIC_ACTIONS),
        "by_scene_unique": dict(sorted(scene_unique.items())),
        "by_scene_occurrences": dict(sorted(scene_occurrences.items())),
        "by_task_type_unique": dict(sorted(task_unique.items())),
        "by_task_type_occurrences": dict(sorted(task_occurrences.items())),
        "by_primary_atomic_action_unique": dict(sorted(primary_atomic_unique.items())),
        "by_primary_atomic_action_occurrences": dict(sorted(primary_atomic_occurrences.items())),
        "by_atomic_action_membership_unique": dict(sorted(atomic_membership_unique.items())),
        "by_source": dict(sorted(sources.items())),
    }


def classify_inventory(
    *,
    inventory_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    checkpoint_path: str | Path,
    api_key: str,
    api_base: str,
    model: str,
    batch_size: int = 32,
    workers: int = 4,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    limit_units: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    unique = load_unique_instructions(inventory_path)
    units, deterministic = build_classification_units(unique)
    checkpoint = Path(checkpoint_path)
    completed = load_checkpoint(checkpoint, model=model)
    classified_at = now_iso()

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as handle:
        for digest, result in deterministic.items():
            if digest in completed:
                continue
            row = _classification_row(
                unique[digest],
                result,
                source="domain_rule",
                model=None,
                review_threshold=review_threshold,
                classified_at=classified_at,
            )
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed[digest] = row
        handle.flush()
        os.fsync(handle.fileno())

    pending = [unit for unit in units if any(digest not in completed for digest in unit.member_hashes)]
    if limit_units is not None:
        pending = pending[:limit_units]
    batches = list(_batches(pending, batch_size))
    client = ResponsesAPIClient(api_key=api_key, base_url=api_base, model=model)
    api_calls = 0
    started_at = time.monotonic()
    print(
        json.dumps(
            {
                "event": "classification_start",
                "inventory_unique": len(unique),
                "already_completed": len(completed),
                "pending_units": len(pending),
                "batches": len(batches),
                "model": model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def request_batch(batch: list[ClassificationUnit]) -> tuple[list[ClassificationUnit], dict[str, dict[str, Any]]]:
        return batch, classify_units_resilient(client, batch)

    with checkpoint.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as executor:
        batch_iterator = iter(batches)
        in_flight: dict[
            Future[tuple[list[ClassificationUnit], dict[str, dict[str, Any]]]], list[ClassificationUnit]
        ] = {}
        completed_batches = 0
        failed_batches = 0

        def submit_next() -> bool:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                return False
            in_flight[executor.submit(request_batch, batch)] = batch
            return True

        for _ in range(min(workers, len(batches))):
            submit_next()

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                submitted_batch = in_flight.pop(future)
                completed_batches += 1
                try:
                    batch, results = future.result()
                except RuntimeError as error:
                    failed_batches += 1
                    print(
                        json.dumps(
                            {
                                "event": "classification_batch_deferred",
                                "unit_count": len(submitted_batch),
                                "error": str(error),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    submit_next()
                    continue
                api_calls += 1
                for unit in batch:
                    result = results[unit.unit_id]
                    for digest in unit.member_hashes:
                        if digest in completed:
                            continue
                        row = _classification_row(
                            unique[digest],
                            result,
                            source=unit.source,
                            model=model,
                            review_threshold=review_threshold,
                            classified_at=classified_at,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        completed[digest] = row
                handle.flush()
                if completed_batches == 1 or completed_batches % 10 == 0 or completed_batches == len(batches):
                    print(
                        json.dumps(
                            {
                                "event": "classification_progress",
                                "completed_batches": completed_batches,
                                "total_batches": len(batches),
                                "classified_unique": len(completed),
                                "elapsed_seconds": round(time.monotonic() - started_at, 1),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                submit_next()

    ordered_rows = [completed[digest] for digest in sorted(completed)]
    _write_atomic_jsonl(Path(output_path), ordered_rows)
    summary = _summary(ordered_rows, model=model, api_calls=api_calls)
    summary["inventory_unique_instructions"] = len(unique)
    summary["complete"] = len(completed) == len(unique)
    summary["remaining_unique_instructions"] = len(unique) - len(completed)
    summary["deferred_batches"] = failed_batches
    _write_atomic_json(Path(summary_path), summary)
    return summary
