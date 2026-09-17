#!/usr/bin/env python3
"""Safely rescan dataset and conversion roots and publish a changed catalog.

The hourly mode deliberately preserves the existing embedding payload. New or
renamed task terms therefore appear in the tree immediately and can be added to
the semantic scatter later by an explicit full catalog build.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from .catalog import scan_catalog, write_catalog
    from .classification_overlay import enrich_instruction_classification
    from .conversion_status import clear_conversion_status_caches, enrich_conversion_status
except ImportError:
    from catalog import scan_catalog, write_catalog
    from classification_overlay import enrich_instruction_classification
    from conversion_status import clear_conversion_status_caches, enrich_conversion_status


PROTECTED_CONVERSION_REGRESSIONS = {
    "converted": {"partial", "not_converted", "excluded", "unknown"},
    "partial": {"not_converted", "unknown"},
    "excluded": {"partial", "not_converted", "unknown"},
}


class PublicationRejected(RuntimeError):
    """Raised when a candidate catalog fails an automatic-publication guard."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _record_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError("catalog.records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("catalog.records must contain objects with string ids")
        result[record["id"]] = record
    if len(result) != len(records):
        raise ValueError("catalog contains duplicate record ids")
    return result


def comparable_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without scan timestamps or the manually refreshed embedding."""

    payload = copy.deepcopy(catalog)
    payload.pop("generated_at", None)
    payload.pop("embedding", None)
    payload.pop("catalog_sync", None)
    conversion = payload.get("conversion")
    if isinstance(conversion, dict):
        conversion.pop("checked_at", None)
    records = payload.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                record.pop("conversion_checked_at", None)
        records.sort(key=lambda record: str(record.get("id") or ""))
    return payload


def catalog_digest(catalog: dict[str, Any]) -> str:
    encoded = json.dumps(comparable_catalog(catalog), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def catalog_diff(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    old_records = _record_map(previous)
    new_records = _record_map(candidate)
    added = sorted(set(new_records) - set(old_records))
    removed = sorted(set(old_records) - set(new_records))
    conversion_changes = []
    changed_records = []
    for record_id in sorted(set(old_records) & set(new_records)):
        old_record = copy.deepcopy(old_records[record_id])
        new_record = copy.deepcopy(new_records[record_id])
        old_record.pop("conversion_checked_at", None)
        new_record.pop("conversion_checked_at", None)
        old_status = str(old_record.get("conversion_status") or "unknown")
        new_status = str(new_record.get("conversion_status") or "unknown")
        if old_status != new_status:
            conversion_changes.append(
                {
                    "id": record_id,
                    "dataset": new_record.get("dataset"),
                    "task": new_record.get("native_task"),
                    "from": old_status,
                    "to": new_status,
                }
            )
        if old_record != new_record:
            changed_records.append(record_id)
    return {
        "changed": catalog_digest(previous) != catalog_digest(candidate),
        "added_ids": added,
        "removed_ids": removed,
        "changed_record_ids": changed_records,
        "conversion_changes": conversion_changes,
    }


def validate_candidate(
    previous: dict[str, Any],
    candidate: dict[str, Any],
    diff: dict[str, Any],
    *,
    minimum_records: int,
    allow_removals: bool,
    max_unknown_increase: int,
) -> None:
    candidate_records = _record_map(candidate)
    if len(candidate_records) < minimum_records:
        raise PublicationRejected(
            f"candidate has {len(candidate_records)} records, below minimum {minimum_records}"
        )
    removed_ids = diff.get("removed_ids") or []
    if removed_ids and not allow_removals:
        raise PublicationRejected(f"candidate removes {len(removed_ids)} records; manual review required")

    regressions = []
    for change in diff.get("conversion_changes") or []:
        if change["to"] in PROTECTED_CONVERSION_REGRESSIONS.get(change["from"], set()):
            regressions.append(change)
    if regressions:
        sample = ", ".join(f"{item['dataset']}:{item['from']}->{item['to']}" for item in regressions[:3])
        raise PublicationRejected(f"candidate contains {len(regressions)} protected conversion regressions: {sample}")

    old_unknown = int(previous.get("stats", {}).get("conversion_by_status", {}).get("unknown", 0))
    new_unknown = int(candidate.get("stats", {}).get("conversion_by_status", {}).get("unknown", 0))
    if new_unknown - old_unknown > max_unknown_increase:
        raise PublicationRejected(
            f"unknown conversion records increase by {new_unknown - old_unknown}, limit {max_unknown_increase}"
        )


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _conversion_counts(catalog: dict[str, Any]) -> dict[str, int]:
    values = catalog.get("stats", {}).get("conversion_by_status", {})
    return {str(key): int(value) for key, value in values.items()}


def _base_status(previous_status: dict[str, Any] | None, *, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "service": "BetaData catalog sync",
        "mode": "hourly_shallow_scan_embedding_preserved",
        "scan_started_at": started_at,
        "last_success_at": (previous_status or {}).get("last_success_at"),
        "last_publish_at": (previous_status or {}).get("last_publish_at"),
        "next_run_at": (previous_status or {}).get("next_run_at"),
    }


def run_sync(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    started_monotonic = time.monotonic()
    started_at = now_iso()
    previous_status = _load_json(args.status_path) if args.status_path.is_file() else None
    status = _base_status(previous_status, started_at=started_at)
    published = False
    try:
        for label, path in (
            ("dataset root", args.root),
            ("PFS conversion root", args.conversion_pfs_root),
            ("CFS conversion root", args.conversion_cfs_root),
        ):
            if not path.is_dir():
                raise PublicationRejected(f"{label} is unavailable: {path}")
        if not args.catalog.is_file():
            raise PublicationRejected(f"current catalog is unavailable: {args.catalog}")

        previous = _load_json(args.catalog)
        clear_conversion_status_caches()
        candidate = scan_catalog(
            args.root,
            generated_at=started_at,
            instruction_inventory=args.instruction_inventory,
        )
        enrich_instruction_classification(candidate, args.instruction_classification)
        enrich_conversion_status(
            candidate,
            pfs_root=args.conversion_pfs_root,
            cfs_root=args.conversion_cfs_root,
            checked_at=started_at,
        )
        diff = catalog_diff(previous, candidate)
        validate_candidate(
            previous,
            candidate,
            diff,
            minimum_records=args.minimum_records,
            allow_removals=args.allow_removals,
            max_unknown_increase=args.max_unknown_increase,
        )

        if diff["changed"]:
            if isinstance(previous.get("embedding"), dict):
                candidate["embedding"] = previous["embedding"]
            candidate["catalog_sync"] = {
                "mode": "automatic_shallow_scan",
                "published_at": now_iso(),
                "embedding": "preserved_not_rebuilt",
            }
            write_catalog(candidate, args.catalog)
            published = True

        completed_at = now_iso()
        active_catalog = candidate if published else previous
        status.update(
            {
                "result": "published" if published else "unchanged",
                "published": published,
                "scan_completed_at": completed_at,
                "last_success_at": completed_at,
                "last_publish_at": completed_at if published else status.get("last_publish_at"),
                "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                "catalog_path": args.catalog.as_posix(),
                "catalog_generated_at": active_catalog.get("generated_at"),
                "catalog_digest": catalog_digest(active_catalog),
                "embedding_updated": False,
                "embedding_preserved": isinstance(active_catalog.get("embedding"), dict),
                "stats": {
                    "records": int(active_catalog.get("stats", {}).get("records", 0)),
                    "datasets": int(active_catalog.get("stats", {}).get("datasets", 0)),
                    "conversion_by_status": _conversion_counts(active_catalog),
                },
                "diff": {
                    "added": len(diff["added_ids"]),
                    "removed": len(diff["removed_ids"]),
                    "changed_records": len(diff["changed_record_ids"]),
                    "conversion_changes": len(diff["conversion_changes"]),
                    "added_ids": diff["added_ids"][:100],
                    "removed_ids": diff["removed_ids"][:100],
                    "conversion_change_sample": diff["conversion_changes"][:100],
                },
            }
        )
        _write_json_atomic(status, args.status_path)
        return status, True
    except Exception as error:
        status.update(
            {
                "result": "blocked" if isinstance(error, PublicationRejected) else "error",
                "scan_completed_at": now_iso(),
                "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                "error": str(error),
                "published": False,
            }
        )
        _write_json_atomic(status, args.status_path)
        return status, False


def _print_status(status: dict[str, Any]) -> None:
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


def _run_locked(args: argparse.Namespace) -> bool:
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{now_iso()} catalog sync skipped: another scan holds {args.lock_path}", flush=True)
            return True
        status, successful = run_sync(args)
        _print_status(status)
        return successful


def _update_next_run(status_path: Path, next_run: datetime) -> None:
    status = _load_json(status_path) if status_path.is_file() else _base_status(None, started_at=now_iso())
    status["next_run_at"] = next_run.astimezone().isoformat(timespec="seconds")
    status["scheduler_pid"] = os.getpid()
    _write_json_atomic(status, status_path)


def watch(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(f"{now_iso()} BetaData catalog sync scheduler started: interval={args.interval_seconds}s", flush=True)
    while not stop_event.is_set():
        cycle_started = time.monotonic()
        _run_locked(args)
        delay = max(1.0, args.interval_seconds - (time.monotonic() - cycle_started))
        next_run = datetime.now().astimezone() + timedelta(seconds=delay)
        _update_next_run(args.status_path, next_run)
        stop_event.wait(delay)
    print(f"{now_iso()} BetaData catalog sync scheduler stopped", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("once", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--instruction-inventory", type=Path)
    parser.add_argument("--instruction-classification", type=Path)
    parser.add_argument("--conversion-pfs-root", type=Path, required=True)
    parser.add_argument("--conversion-cfs-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=1000)
    parser.add_argument("--max-unknown-increase", type=int, default=0)
    parser.add_argument("--allow-removals", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.minimum_records < 1:
        raise SystemExit("--minimum-records must be positive")
    if args.max_unknown_increase < 0:
        raise SystemExit("--max-unknown-increase must be nonnegative")
    if args.interval_seconds < 60:
        raise SystemExit("--interval-seconds must be at least 60")
    if args.mode == "watch":
        raise SystemExit(watch(args))
    raise SystemExit(0 if _run_locked(args) else 1)


if __name__ == "__main__":
    main()
