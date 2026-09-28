"""Continuously feed disjoint, bounded GT batches to local and cloud schedulers."""
from contextlib import contextmanager
import fcntl
import uuid

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from apps.api.catalog_service import resolve_selection
from apps.api.db import (CloudExecution, CollectionBatch, CollectionClaim, CollectionPlan,
                         DataSample, Run, now, serialize, session)
from apps.api.schemas import RunRequest
from apps.api.settings import settings
from catalog.vln_catalog import connect, where_clause
from workers.cloud_config import load_pools, validate_cloud

ACTIVE = {"submitting", "submitted", "uncertain", "stopping"}
TERMINAL = {"succeeded", "failed", "cancelled"}


@contextmanager
def allocation_lock():
    """Serialize both feeders, without blocking cloud lease renewal or workers.

    CollectionClaim's unique key remains the durable duplicate-work guard.
    The common lock also keeps capacity/progress snapshots fresh across feeders.
    """
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    with (settings.runtime_root / "collection-allocation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def device_error(plan, device, error):
    """One feeder must not erase another selected device's failure reason."""
    errors = dict(plan.progress.get("device_errors", {}))
    if error:
        errors[device] = error
    else:
        errors.pop(device, None)
    plan.progress = {**plan.progress, "device_errors": errors}
    plan.error = ("采集批次失败，请检查日志后继续 / 重试" if plan.progress.get("batches_failed")
                  else "；".join(errors.values()) or None)


def candidates(cfg):
    """Read a fresh catalog scope; no count limit or implicit change of split."""
    clause, values = where_clause({"dataset": {"r2r": "R2R", "rxr": "RxR", "scalevln": "ScaleVLN"}[cfg["dataset"]],
                                   "split": cfg["split"], "role": cfg["role"]})
    catalog = connect(settings.catalog_db)
    try:
        rows = [dict(r) for r in catalog.execute(
            "SELECT id,episode_id,language,scene FROM episodes" + clause + " ORDER BY scene,episode_id,id", values)
                if (cfg["languages"] == ["all"] or r["language"] in cfg["languages"])
                and (not cfg["episode_ids"] or r["episode_id"] in cfg["episode_ids"])
                and (not cfg["source_ids"] or r["id"] in cfg["source_ids"])]
    finally:
        catalog.close()
    if not rows:
        raise ValueError("所选持续采集范围中没有源轨迹")
    if (cfg["episode_ids"] and set(cfg["episode_ids"]) != {r["episode_id"] for r in rows}) or (
            cfg["source_ids"] and set(cfg["source_ids"]) != {r["id"] for r in rows}):
        raise ValueError("指定的源轨迹不完整，请检查数据集、划分、角色、语言和 ID")
    if len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("来源目录中存在重复 episode，请先指定唯一版本的源样本")
    return rows


def prepare(payload):
    cfg = payload.model_dump(mode="json", exclude={"compute_pools"})
    cfg.update(count=1, full_dataset=False, compute_pool="auto")
    if payload.execution != "local":
        pools = {p.id: p for p in load_pools()}
        if any(pid not in pools or not pools[pid].enabled for pid in payload.compute_pools if pid != "local"):
            raise ValueError("所选采集设备尚未配置或未启用")
    rows = candidates(cfg)
    return cfg, rows


def reserved(db):
    claimed = set(db.scalars(select(CollectionClaim.source_id)))
    completed = set(db.scalars(select(DataSample.source_record_id).join(Run, Run.id == DataSample.run_id)
        .where(DataSample.origin == "gt", DataSample.quality == "validated", Run.status == "succeeded")))
    # Also respect ordinary explicit batches which are already in flight.
    active_runs = list(db.scalars(select(Run).where(Run.kind == "gt", Run.status.not_in(TERMINAL))))
    in_flight = {sid for r in active_runs for sid in r.config.get("source_ids", [])}
    return claimed | completed | in_flight, completed, active_runs


def batch_config(plan, rows, pool_id):
    cfg = dict(plan.config)
    cfg.pop("plan_request_payload", None)
    cfg.update(episode_ids=[r["episode_id"] for r in rows], source_ids=[r["id"] for r in rows],
               execution="cloud_idle", full_dataset=False, count=len(rows), compute_pool=pool_id,
               collection_plan_id=plan.id, workers=1)
    cfg["request_payload"] = RunRequest.model_validate(cfg).model_dump(mode="json")
    cfg = resolve_selection(cfg)
    cfg["capacity"] = validate_cloud(cfg)
    return cfg


def preview(payload):
    cfg, rows = prepare(payload)
    with session() as db:
        taken, completed, _ = reserved(db)
    remaining = [r for r in rows if r["id"] not in taken]
    return {"episodes": len(rows), "remaining": len(remaining),
            "completed": sum(r["id"] in completed for r in rows),
            "batch_size": cfg["routes_per_shard"], "compute_pools": payload.compute_pools,
            "basis": "持续采集该范围内未完成的轨迹；每批启动前单独检查容量。采完后等待目录新增轨迹。"}


def summaries():
    with session() as db:
        result = []
        for plan in db.scalars(select(CollectionPlan).order_by(CollectionPlan.created_at.desc()).limit(50)):
            item = serialize(plan)
            batches = db.scalars(select(Run).join(CollectionBatch, CollectionBatch.run_id == Run.id)
                .where(CollectionBatch.plan_id == plan.id).order_by(Run.created_at.desc()).limit(10)).all()
            item["batches"] = [{"id": r.id, "name": r.name, "status": r.status,
                                "compute_pool": "local" if r.config.get("execution") == "local" else r.config["compute_pool"]} for r in batches]
            if plan.control == "run" and plan.status != "needs_attention" and any(r.status == "running" for r in batches):
                item["status"] = "running"
            result.append(item)
        return result


def control_plan(plan_id, action):
    if action not in {"pause", "resume", "cancel", "retry"}:
        raise ValueError("不支持的持续采集操作")
    with session() as db:
        plan = db.scalar(select(CollectionPlan).where(CollectionPlan.id == plan_id).with_for_update())
        if not plan:
            raise LookupError("持续采集计划不存在")
        if plan.control == "cancel":
            raise ValueError("该持续采集计划已停止")
        batches = list(db.scalars(select(CollectionBatch).where(CollectionBatch.plan_id == plan_id)))
        for batch in batches:
            run = db.get(Run, batch.run_id)
            binding = db.get(CloudExecution, run.id)
            active = (binding and binding.state in ACTIVE) or (
                run.config.get("execution") == "local" and run.status == "running")
            if action in {"resume", "retry"}:
                if not batch.paused_by_plan and not (action == "retry" and run.status == "failed"):
                    continue
                if active:
                    raise ValueError("正在暂停当前采集阶段，确认停止后即可继续")
                if run.status not in {"succeeded", "cancelled"}:
                    run.control, run.status = "run", "queued"
                    run.error = None
                    if binding:
                        binding.failures = 0
                batch.paused_by_plan = 0
            elif run.status not in {"succeeded", "cancelled"}:
                if action == "pause" and (run.control != "run" or run.status == "failed"):
                    continue
                batch.paused_by_plan = int(action == "pause")
                run.control = action
                if not active:
                    run.status = "paused" if action == "pause" else "cancelled"
            run.updated_at = now()
        plan.control = "run" if action in {"resume", "retry"} else action
        plan.status = {"pause": "paused", "cancel": "cancelled"}.get(action, "waiting")
        plan.error, plan.updated_at = None, now()
        return serialize(plan)


def refill(pools, snapshots):
    """Called under the cloud controller's process lock, after lease renewal."""
    with allocation_lock() as acquired:
        if acquired:
            _refill(pools, snapshots)


def _refill(pools, snapshots):
    with session() as db:
        # Release cancelled work only after the cloud scheduler confirmed it stopped.
        cancelled = select(Run.id).where(Run.status == "cancelled")
        db.execute(delete(CollectionClaim).where(CollectionClaim.run_id.in_(cancelled)))
        plans = list(db.scalars(select(CollectionPlan).where(CollectionPlan.control == "run")
                               .order_by(CollectionPlan.created_at)))
    for plan in plans:
        if not any(pid != "local" for pid in plan.compute_pools):
            continue
        try:
            refill_one(plan, pools, snapshots)
        except Exception as exc:
            with session() as db:
                current = db.get(CollectionPlan, plan.id)
                if current.control == "run":
                    device_error(current, "cloud", str(exc) if isinstance(exc, (ValueError, OSError)) else type(exc).__name__)
                    current.updated_at = now()


def refill_one(plan, pools, snapshots):
    rows = candidates(plan.config)
    with session() as db:
        taken, completed, active_runs = reserved(db)
        children = list(db.scalars(select(Run).join(CollectionBatch, CollectionBatch.run_id == Run.id)
                                  .where(CollectionBatch.plan_id == plan.id)))
        # A full-split job owns this scope until it finishes.
        if any(r.config.get("full_dataset") and all(r.config.get(k) == plan.config[k]
                   for k in ("dataset", "split", "role")) for r in active_runs):
            taken.update(r["id"] for r in rows)
        pending = [r for r in rows if r["id"] not in taken]
        failed = [r for r in children if r.status == "failed"]
        current = db.get(CollectionPlan, plan.id)
        if current.control != "run":
            return
        current.progress = {**current.progress, "total": len(rows), "completed": sum(r["id"] in completed for r in rows),
            "remaining": len(pending), "assigned_or_skipped": sum(r["id"] in taken - completed for r in rows),
            "batches_completed": sum(r.status == "succeeded" for r in children), "batches_failed": len(failed)}
        current.status = "needs_attention" if failed else "running" if any(r.status == "running" for r in children) else "waiting" if pending or any(r.status not in TERMINAL for r in children) else "watching"
        device_error(current, "cloud", None)
        current.updated_at = now()
    if failed or not pending:
        return
    for pool in pools:
        snapshot = snapshots.get(pool.id, {})
        if (pool.id not in plan.compute_pools or not pool.enabled or
                plan.config["dataset"] not in pool.validated_datasets or not snapshot.get("idle") or
                not snapshot.get("observed") or snapshot.get("pending_foreground")):
            continue
        # Bound both queued and running work, including standalone acceptance jobs.
        occupied = max(snapshot.get("active_jobs", 0), sum(
            r.control == "run" and r.status != "paused" and r.config.get("execution") == "cloud_idle" and
            pool.id in r.config.get("eligible_pools", []) for r in active_runs))
        for _ in range(max(0, pool.max_active_jobs - occupied)):
            batch = pending[:plan.config["routes_per_shard"]]
            if not batch:
                break
            try:
                cfg = batch_config(plan, batch, pool.id)
            except (ValueError, OSError) as exc:
                with session() as db:
                    current = db.get(CollectionPlan, plan.id)
                    if current.control == "run":
                        device_error(current, pool.id, f"{pool.name}：{exc}")
                break  # Other selected pools may still be usable.
            run_id = str(uuid.uuid4())
            try:
                with session() as db:
                    current = db.scalar(select(CollectionPlan).where(CollectionPlan.id == plan.id).with_for_update())
                    if current.control != "run":
                        return
                    device_error(current, pool.id, None)
                    run = Run(id=run_id, request_key="continuous-" + run_id, kind="gt",
                              name=f"{plan.name} · {pool.name} · {run_id[:8]}", config=cfg)
                    db.add(run)
                    db.flush()
                    db.add(CollectionBatch(run_id=run_id, plan_id=plan.id))
                    db.add_all([CollectionClaim(source_id=r["id"], run_id=run_id) for r in batch])
                    db.flush()
            except IntegrityError:
                return  # Another plan claimed it; reread durable claims next cycle.
            active_runs.append(run)
            pending = pending[len(batch):]
