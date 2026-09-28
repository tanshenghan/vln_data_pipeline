"""One bounded local GT batch at a time, fair across durable collection plans."""
from pathlib import Path
import subprocess
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from apps.api.catalog_service import resolve_selection
from apps.api.db import CollectionBatch, CollectionClaim, CollectionPlan, Run, now, session
from apps.api.preflight import validate_gt
from apps.api.schemas import RunRequest
from apps.api.settings import settings
from workers.collection_plans import TERMINAL, allocation_lock, candidates, device_error, reserved


def gpu_idle():
    """Fail closed if monitoring fails; never terminate another user's process."""
    try:
        result = subprocess.run(["nvidia-smi", f"--id={settings.gpu_id}",
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode or result.stdout.strip():
            return False
        result = subprocess.run(["nvidia-smi", f"--id={settings.gpu_id}",
            "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode:
            return False
        utilization, memory = map(int, result.stdout.strip().split(","))
        return utilization < 10 and memory < 512
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def ready_rows(cfg, rows):
    if cfg["dataset"] != "scalevln":
        return rows
    from data_pipeline.scalevln_source import availability
    ready = availability(settings.dataset_root("scalevln"), settings.scenes_root)
    scenes = {Path(path).stem for path in ready["scene_paths"]}
    return [row for row in rows if row["scene"] in scenes]


def batch_config(plan, rows):
    cfg = dict(plan.config)
    cfg.pop("plan_request_payload", None)
    cfg.update(episode_ids=[r["episode_id"] for r in rows], source_ids=[r["id"] for r in rows],
               execution="local", compute_pool="auto", full_dataset=False, count=len(rows),
               collection_plan_id=plan.id, workers=1)
    cfg["request_payload"] = RunRequest.model_validate(cfg).model_dump(mode="json")
    cfg = resolve_selection(cfg)
    cfg["capacity"] = validate_gt(cfg)
    return cfg


def refill_local():
    if settings.cloud_worker:
        return
    with allocation_lock() as acquired:
        if acquired:
            _refill()


def _refill():
    with session() as db:
        cancelled = select(Run.id).where(Run.status == "cancelled")
        db.execute(delete(CollectionClaim).where(CollectionClaim.run_id.in_(cancelled)))
        plans = [p for p in db.scalars(select(CollectionPlan).where(CollectionPlan.control == "run"))
                 if "local" in p.compute_pools]
        last_batch = dict(db.execute(select(CollectionBatch.plan_id, func.max(Run.created_at))
            .join(Run, Run.id == CollectionBatch.run_id)
            .where(Run.config["execution"].as_string() == "local").group_by(CollectionBatch.plan_id)).all())
        # Plans which have never received work go first; then least recently fed.
        plans.sort(key=lambda p: (p.id in last_batch, last_batch.get(p.id, p.created_at), p.id))
        taken, completed, active = reserved(db)
        occupied = any(r.config.get("execution", "local") == "local" and
                       r.status in {"queued", "running", "waiting"} for r in active)
    idle = None
    for plan in plans:
        try:
            rows = candidates(plan.config)
            if any(r.config.get("full_dataset") and all(r.config.get(k) == plan.config[k]
                    for k in ("dataset", "split", "role")) for r in active):
                taken.update(r["id"] for r in rows)
            pending = [r for r in rows if r["id"] not in taken]
            available = ready_rows(plan.config, pending)
            with session() as db:
                current = db.get(CollectionPlan, plan.id)
                if current.control != "run":
                    continue
                children = list(db.scalars(select(Run).join(CollectionBatch, CollectionBatch.run_id == Run.id)
                    .where(CollectionBatch.plan_id == plan.id)))
                failed = any(r.status == "failed" for r in children)
                current.progress = {**current.progress, "total": len(rows), "completed": sum(r["id"] in completed for r in rows),
                    "remaining": len(pending), "ready_remaining": len(available),
                    "assigned_or_skipped": sum(r["id"] in taken - completed for r in rows),
                    "batches_completed": sum(r.status == "succeeded" for r in children),
                    "batches_failed": sum(r.status == "failed" for r in children)}
                current.status = ("needs_attention" if failed else "running" if any(r.status == "running" for r in children)
                    else "waiting" if pending or any(r.status not in TERMINAL for r in children) else "watching")
                device_error(current, "local", "等待所选源轨迹的场景和 NavMesh 就绪" if pending and not available else None)
                current.updated_at = now()
            if failed or occupied or not available:
                continue
            if idle is None:
                idle = gpu_idle()
            if not idle:
                with session() as db:
                    device_error(db.get(CollectionPlan, plan.id), "local", "GPU0 正被占用或状态不可确认，等待空闲")
                continue
            batch = available[:plan.config["routes_per_shard"]]
            cfg = batch_config(plan, batch)
            run_id = str(uuid.uuid4())
            with session() as db:
                current = db.scalar(select(CollectionPlan).where(CollectionPlan.id == plan.id).with_for_update())
                if current.control != "run":
                    continue
                db.add(Run(id=run_id, request_key="continuous-" + run_id, kind="gt",
                    name=f"{plan.name} · GPU0 · {run_id[:8]}", config=cfg))
                db.flush()
                db.add(CollectionBatch(run_id=run_id, plan_id=plan.id))
                db.add_all([CollectionClaim(source_id=r["id"], run_id=run_id) for r in batch])
                db.flush()
            occupied = True
        except IntegrityError:
            return  # A cloud/local plan claimed these sources; reread next cycle.
        except (OSError, ValueError) as exc:
            with session() as db:
                current = db.get(CollectionPlan, plan.id)
                if current.control == "run":
                    device_error(current, "local", f"GPU0：{exc}")
                    current.updated_at = now()
