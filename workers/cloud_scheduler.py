"""Independent idle-job controller; never shares the busy local Celery worker."""
import argparse
import fcntl
import json
import logging
import time
from datetime import timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from apps.api.db import Base, CloudExecution, CollectionBatch, CollectionPlan, Run, engine, now, session
from apps.api.settings import settings
from workers.cloud_config import Pool, load_pools
from workers.cloud_providers import CloudError, Provider
from workers.command_runner import atomic_json

log = logging.getLogger(__name__)
ACTIVE = {"submitting", "submitted", "uncertain", "stopping"}


def due(value):
    return value is None or value.replace(tzinfo=timezone.utc) <= now()


def save(run_id, *, binding=None, **changes):
    with session() as db:
        row = db.get(Run, run_id)
        for key, value in changes.items():
            setattr(row, key, value)
        row.updated_at = now()
        if binding:
            execution = db.get(CloudExecution, run_id)
            for key, value in binding.items():
                setattr(execution, key, value)


def lease(binding, pool, action):
    directory = Path(binding.directory)
    pool.check_storage()
    atomic_json(directory / "cloud_control.json", {"action": action, "generation": binding.generation,
                "deadline": time.time() + pool.lease_seconds})


def read_result(binding):
    try:
        result = json.loads((Path(binding.directory) / "cloud_result.json").read_text())
        return result if result.get("generation") == binding.generation else None
    except (OSError, ValueError):
        return None


def import_result(run_id, binding, result):
    from workers.tasks import register
    from contracts.check import check_metadata
    from workers.sample_catalog import register_samples, register_vlnverse_candidates
    from workers.catalog_sync import sync_gt_catalog
    directory = Path(binding.directory).resolve()
    artifacts = result.get("artifacts", [])
    if not artifacts:
        raise ValueError("云任务未返回可校验的产物清单")
    if not any(a["kind"] == "dataset" for a in artifacts):
        with session() as db:
            dataset = db.get(Run, run_id).config["dataset"]
        summary = directory / "trajectory_review/summary.json"
        if dataset != "vlnverse" or not summary.is_file() or json.loads(summary.read_text()).get("accepted") != 0:
            raise ValueError("云任务缺少正式 GT 数据集")
    for item in artifacts:
        relative = Path(item["path"])
        path = (directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(directory) or not path.exists():
            raise ValueError("云产物路径无效或尚未写入持久化盘")
        if item["kind"] == "dataset":
            check_metadata(path)
            report = json.loads((path / "meta/platform_validation.json").read_text())
            if not all(report.get(k) is True for k in ("complete", "self_contained", "official_loader_all_frames",
                                                       "all_30_source_arrays_exact", "geometry_recomputed")):
                raise ValueError("云产物缺少完整独立校验")
    from workers.output_storage import poll_cloud_archive
    directory = poll_cloud_archive(run_id, directory)
    if directory is None:
        return False
    for item in artifacts:
        register(run_id, item["kind"], directory / item["path"], item.get("details"))
    if (directory / "trajectory_review/summary.json").exists():
        register_vlnverse_candidates(run_id)
    register_samples(run_id)
    sync_gt_catalog()


class Scheduler:
    def __init__(self, provider_factory=Provider):
        self.provider_factory = provider_factory
        self.clear_since = {}

    def bind(self, row, pool):
        pool.check_storage()
        directory = pool.local_storage_root / "runs" / row.id
        directory.mkdir(parents=True, exist_ok=True)
        with session() as db:
            if db.get(CloudExecution, row.id):
                return
            db.add(CloudExecution(run_id=row.id, pool_id=pool.id, directory=str(directory),
                                  pool_digest=pool.digest(), pool_spec=pool.model_dump(mode="json")))
        atomic_json(directory / "config.json", row.config)

    def stop(self, row, binding, pool, provider, reason):
        lease(binding, pool, "cancel" if row.control == "cancel" else "pause")
        if binding.state != "stopping":
            save(row.id, binding={"state": "stopping", "stop_reason": reason,
                                  "not_before": now() + timedelta(seconds=pool.stop_grace_seconds)},
                 status="running", stage="cloud_yielding", progress={"pool": pool.name, "reason": reason})
        elif due(binding.not_before):
            provider.stop(binding.job_id)
            # Stop acknowledgement does not imply GPU release. Retain the ID
            # and inspect it on later ticks until the provider confirms terminal.

    def latest_control(self, run_id, binding):
        with session() as db:
            row = db.get(Run, run_id)
            parent = db.get(CollectionPlan, row.config["collection_plan_id"]) if row.config.get("collection_plan_id") else None
            if parent and parent.control != "run" and row.control == "run" and row.status not in {"succeeded", "cancelled", "failed"}:
                row.control = parent.control
                batch = db.get(CollectionBatch, row.id)
                if batch:
                    batch.paused_by_plan = int(parent.control == "pause")
                if binding.state not in ACTIVE:
                    row.status = "paused" if parent.control == "pause" else "cancelled"
        return row

    def finish(self, row, binding, pool, job):
        # User intent may have changed while the provider request was in flight.
        row = self.latest_control(row.id, binding)
        result = read_result(binding)
        if row.control in {"pause", "cancel"}:
            save(row.id, binding={"state": "waiting", "job_id": None, "job_name": None, "stop_reason": None},
                 status="paused" if row.control == "pause" else "cancelled", stage="cloud_stopped")
        elif result and result["status"] == "succeeded" and job.state == "succeeded":
            if import_result(row.id, binding, result) is False:
                save(row.id, status="running", stage="cloud_archiving", error=None,
                     progress={"pool": pool.name, "stage": "cloud_archiving"})
                return
            save(row.id, binding={"state": "done"}, status="succeeded", stage="complete",
                 error=None, progress={"complete": True, "pool": pool.name})
        else:
            yielded = binding.stop_reason in {"foreground", "monitor_unavailable", "disabled"} or job.preempted
            yielded = yielded or bool(result and result["status"] == "yielded")
            failures = binding.failures + (0 if yielded else 1)
            failed = failures >= pool.max_failures or binding.stop_reason == "config_changed"
            save(row.id, binding={"state": "waiting", "job_id": None, "job_name": None,
                    "stop_reason": None, "failures": failures,
                    "not_before": now() + timedelta(seconds=pool.cooldown_seconds)},
                 status="failed" if failed else "waiting", stage="cloud_waiting",
                 error=("云端执行失败，查看持久化阶段日志后重试" if failed else None),
                 progress={"pool": pool.name, "reason": "资源已让出，等待空闲后续跑" if yielded else "云任务中断，冷却后重试",
                           "failures": failures})

    def reconcile(self, row, binding, pool, provider, jobs, blocked, config_changed=False):
        # Plan/user controls may have changed during the queue API request.
        row = self.latest_control(row.id, binding)
        # Recover the exact name after a crash/timeout between CreateJob and DB
        # commit. Absence in a listing never proves that a timed-out create failed.
        if binding.state in {"submitting", "uncertain"} and not binding.job_id:
            matches = [j for j in jobs if j.name == binding.job_name]
            if len(matches) != 1:
                save(row.id, binding={"state": "uncertain"}, stage="cloud_submit_unknown",
                     error="云提交结果待核对；按唯一任务名查询中，不会重复提交")
                return
            binding.job_id = matches[0].id
            binding.state = "submitted"
            save(row.id, binding={"job_id": binding.job_id, "state": "submitted"}, error=None)
        if binding.job_id and binding.state in ACTIVE:
            job = provider.get(binding.job_id)
            if job.name != binding.job_name:
                raise CloudError("云任务 ID 与已记录任务名不匹配，禁止操作")
            if job.terminal:
                self.finish(row, binding, pool, job)
                return
            reason = ("config_changed" if config_changed else "manual" if row.control != "run" else
                      "disabled" if not pool.enabled else "foreground" if blocked else binding.stop_reason)
            if reason:
                self.stop(row, binding, pool, provider, reason)
                return
            lease(binding, pool, "run")
            progress = {"pool": pool.name, "job_id": job.id, "provider_state": job.state}
            try:
                worker = json.loads((Path(binding.directory) / "cloud_progress.json").read_text())
                if worker.get("generation") == binding.generation:
                    progress["worker"] = worker.get("progress", {})
            except (OSError, ValueError):
                pass
            save(row.id, status="running" if job.state == "running" else "waiting",
                 stage="cloud_running" if job.state == "running" else "cloud_waiting", heartbeat=now(), progress=progress)
            return
        if row.control != "run" or row.status in {"paused", "failed", "cancelled", "succeeded"}:
            return
        if config_changed:
            save(row.id, status="failed", error="云镜像、存储或执行配置已改变；恢复原配置后续跑，或创建新任务")
            return
        if blocked or not pool.enabled or not due(binding.not_before):
            save(row.id, status="waiting", stage="cloud_waiting", progress={"pool": pool.name, "reason": "等待空闲资源"})
            return
        generation = binding.generation + 1
        name = f"vln-{row.id.replace('-', '')}-g{generation}"
        payload = provider.payload(row.id, generation, name)  # Validate before persisting submit intent.
        binding.generation = generation
        lease(binding, pool, "run")
        atomic_json(Path(binding.directory) / "config.json", row.config)
        save(row.id, binding={"generation": generation, "job_name": name, "state": "submitting", "stop_reason": None},
             status="waiting", stage="cloud_submitting", error=None, attempt=row.attempt + 1)
        # Never automatically replay CreateJob, even for a transport exception.
        job_id = provider.submit(payload)
        save(row.id, binding={"job_id": job_id, "state": "submitted"})

    def tick(self):
        pools = load_pools()
        with session() as db:
            rows = [r for r in db.scalars(select(Run).order_by(Run.created_at)) if r.config.get("execution") == "cloud_idle"]
            bindings = {b.run_id: b for b in db.scalars(select(CloudExecution))}
        snapshots = {}
        # Removed pool configurations still need their recorded jobs stopped.
        configured = {p.id: p for p in pools}
        for binding in bindings.values():
            if binding.pool_id not in configured:
                pool = Pool.model_validate(binding.pool_spec)
                pool.enabled = False
                configured[pool.id] = pool
        for pool in configured.values():
            owned = [b for b in bindings.values() if b.pool_id == pool.id and b.state in ACTIVE]
            snapshot = {"observed": False, "active_jobs": len(owned)}
            snapshots[pool.id] = snapshot
            if not pool.enabled and not owned:
                snapshot["reason"] = "未启用"
                continue
            provider = None
            try:
                pool.credentials()
                pool.check_storage()
                provider = self.provider_factory(pool)
                jobs = provider.list_jobs()
                owned_ids = {b.job_id for b in owned}
                owned_names = {b.job_name for b in owned}
                pending = sum(j.state == "pending" and not j.idle and j.id not in owned_ids and j.name not in owned_names for j in jobs)
                if pending:
                    self.clear_since.pop(pool.id, None)
                else:
                    self.clear_since.setdefault(pool.id, time.monotonic())
                stable = not pending and time.monotonic() - self.clear_since[pool.id] >= pool.idle_seconds
                snapshot.update(observed=True, pending_foreground=pending,
                                reason="等待其他任务获得资源" if pending else "空闲观察中" if not stable else "")
                active_count = len(owned)
                for row in rows:
                    binding = bindings.get(row.id)
                    if not binding:
                        if (not pool.enabled or not stable or active_count >= pool.max_active_jobs or
                            row.control != "run" or row.status not in {"queued", "waiting"} or
                            pool.id not in row.config.get("eligible_pools", [])):
                            continue
                        self.bind(row, pool)
                        with session() as db:
                            binding = db.get(CloudExecution, row.id)
                        bindings[row.id] = binding
                    if binding.pool_id != pool.id or binding.state == "done":
                        continue
                    if binding.state not in ACTIVE and (not stable or active_count >= pool.max_active_jobs):
                        continue
                    old_active = binding.state in ACTIVE
                    frozen = Pool.model_validate(binding.pool_spec)
                    frozen.enabled = pool.enabled
                    frozen.max_failures = pool.max_failures
                    changed = binding.pool_digest != pool.digest()
                    # Route stop/get through the original account/queue if an
                    # operator changed the configuration while a job was live.
                    original = self.provider_factory(frozen) if changed else provider
                    try:
                        self.reconcile(row, binding, frozen, original, jobs, bool(pending), changed)
                    except Exception as exc:
                        reason = str(exc) if isinstance(exc, (CloudError, ValueError)) else type(exc).__name__
                        save(row.id, error=f"云调度暂不可用：{reason}；保留任务 ID 与断点")
                        log.warning("Cloud run %s: %s", row.id, reason)
                        stable = False
                        self.clear_since.pop(pool.id, None)
                        snapshot.update(observed=False, reason="云任务状态待核对，暂停新任务派发")
                    finally:
                        if original is not provider:
                            original.close()
                    with session() as db:
                        current = db.get(CloudExecution, row.id)
                    active_count += int(current.state in ACTIVE) - int(old_active)
                snapshot["active_jobs"] = active_count
                snapshot["idle"] = stable
            except Exception as exc:
                self.clear_since.pop(pool.id, None)
                snapshot["reason"] = f"监控不可用（{type(exc).__name__}），暂停派发"
                log.warning("Cloud pool %s: %s", pool.id, type(exc).__name__)
                # A lost API/control connection must never become 'idle'. The
                # remote lease watchdog releases resources even if StopJob fails.
                for binding in owned:
                    try:
                        frozen = Pool.model_validate(binding.pool_spec)
                        lease(binding, frozen, "pause")
                        save(binding.run_id, binding={"stop_reason": "monitor_unavailable"}, stage="cloud_yielding")
                    except (OSError, ValueError):
                        pass
            finally:
                if provider:
                    provider.close()
        atomic_json(settings.runtime_root / "cloud_status.json", {"updated": time.time(), "pools": snapshots})
        from workers.collection_plans import refill
        refill(pools, snapshots)
        return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one scheduling cycle (may create/stop jobs)")
    parser.add_argument("--check", action="store_true", help="Read-only configuration and cloud API check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.check:
        for pool in load_pools():
            reason = pool.ready()
            if not reason:
                provider = Provider(pool)
                try:
                    provider.payload("00000000-0000-0000-0000-000000000000", 1, "vln-check",
                                     authenticate_image=False)
                    count = len(provider.list_jobs())
                    reason = f"可连接，读取 {count} 个任务；本检查不执行 GPU 验证"
                finally:
                    provider.close()
            print(f"{pool.id}: {reason}")
        return
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    with (settings.runtime_root / "cloud_scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = Scheduler()
        while True:
            try:
                scheduler.tick()
            except Exception as exc:
                log.error("Cloud configuration/control failure: %s", type(exc).__name__)
            if args.once:
                break
            time.sleep(max(5, settings.cloud_poll_seconds))


if __name__ == "__main__":
    main()
