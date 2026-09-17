"""Overlay instruction classifications onto task-level catalog records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .instruction_classification import ATOMIC_ACTIONS, SCENES, TASK_TYPES, TAXONOMY_VERSION
    from .instruction_inventory import instruction_hash
except ImportError:
    from instruction_classification import ATOMIC_ACTIONS, SCENES, TASK_TYPES, TAXONOMY_VERSION
    from instruction_inventory import instruction_hash


UNCLASSIFIED_ATOMIC_ACTION = "待分类"


def load_instruction_classifications(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load the latest valid current-taxonomy row for each instruction hash."""

    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A classifier may be appending while the hourly scanner reads.
                # Ignore an incomplete line and keep the last published mapping.
                continue
            digest = row.get("instruction_hash")
            primary = row.get("primary_atomic_action")
            actions = row.get("atomic_actions")
            if (
                row.get("taxonomy_version") != TAXONOMY_VERSION
                or not isinstance(digest, str)
                or primary not in ATOMIC_ACTIONS
                or not isinstance(actions, list)
                or primary not in actions
                or any(action not in ATOMIC_ACTIONS for action in actions)
                or row.get("scene") not in SCENES
                or row.get("task_type") not in TASK_TYPES
            ):
                continue
            rows[digest] = row
    return rows


def enrich_instruction_classification(
    catalog: dict[str, Any], classification_path: str | Path | None
) -> dict[str, Any]:
    """Apply semantic fields without changing native task names or paths."""

    classifications = load_instruction_classifications(classification_path)
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError("catalog.records must be a list")

    applied = 0
    needs_review = 0
    for record in records:
        record["primary_atomic_action"] = UNCLASSIFIED_ATOMIC_ACTION
        record["atomic_actions"] = []
        row = classifications.get(instruction_hash(str(record.get("native_task") or "")))
        if row is None:
            continue
        record["scene"] = row["scene"]
        record["scene_source"] = "instruction_classification"
        record["task_type"] = row["task_type"]
        record["task_type_source"] = "instruction_classification"
        record["primary_atomic_action"] = row["primary_atomic_action"]
        record["atomic_actions"] = list(row["atomic_actions"])
        record["semantic_confidence"] = row.get("confidence")
        record["semantic_needs_review"] = bool(row.get("needs_review"))
        record["semantic_reason"] = row.get("reason") or ""
        record["semantic_classification_source"] = row.get("classification_source") or "unknown"
        record["semantic_taxonomy_version"] = TAXONOMY_VERSION
        applied += 1
        needs_review += int(record["semantic_needs_review"])

    pending = len(records) - applied
    catalog.setdefault("stats", {})["instruction_classification"] = {
        "classified_records": applied,
        "pending_records": pending,
        "needs_review_records": needs_review,
    }
    catalog["instruction_classification"] = {
        "taxonomy_version": TAXONOMY_VERSION,
        "source_path": Path(classification_path).as_posix() if classification_path is not None else None,
        "classified_records": applied,
        "pending_records": pending,
        "needs_review_records": needs_review,
    }
    return catalog
