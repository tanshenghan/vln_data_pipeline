"""Run the existing GT pipeline inside a disposable, preemptible cloud job.

All stages/checkpoints/artifacts live on a persistent POSIX volume. Only the
container-local bookkeeping DB is disposable; no central DB or Redis access is
needed. The supervisor owns every child process group and enforces a lease.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from uuid import UUID

from workers.command_runner import atomic_json


def action(directory, generation, timestamp=None):
    try:
        value = json.loads((directory / "cloud_control.json").read_text())
        if value.get("generation") != generation or value.get("deadline", 0) <= (timestamp or time.time()):
            return "pause"
        return value["action"] if value.get("action") in {"run", "pause", "cancel"} else "pause"
    except (OSError, ValueError):
        return "pause"


def recover_interrupted(directory):
    """Called only after acquiring the cross-container lock on the shared disk.

PID identities from another container/host cannot be used to attach. Completed
stages and shards are untouched. Interrupted attempts get an explicit exit so
the existing archive/retry mechanism can safely restart their scratch output.
    """
    plan = directory / "dataset.work/plan.json"
    if plan.is_file():
        try:
            json.loads(plan.read_text())
        except ValueError:
            # A native eviction can interrupt the initial non-atomic plan write.
            # No shards exist at that point. Never discard a plan with outputs.
            if any((plan.parent / "shards").glob("*/meta/build_shard.json")) or (directory / "dataset").exists():
                raise ValueError("云采集计划损坏但已有产物，请人工核对后恢复")
            plan.rename(plan.with_name(f"plan.interrupted-{time.time_ns()}.json"))
    for stage in (directory / "stages").glob("*"):
        if (stage / "success.json").exists():
            continue
        for attempt in stage.glob("attempt-*"):
            if not (attempt / "exit.json").exists():
                atomic_json(attempt / "exit.json", {"returncode": -15, "finished": time.time(), "cloud_interrupted": True})


def validate_environment(cfg):
    from apps.api.settings import settings
    from apps.api.capacity import estimate_gt
    from apps.api.preflight import readable_file
    from data_pipeline.collect_r2r import sha256
    from data_pipeline.vlnce_source import source_paths
    if cfg["dataset"] != "vlnverse":
        paths = source_paths(settings.dataset_root(cfg["dataset"]), cfg["dataset"], cfg["split"], cfg["role"])
        if [sha256(p) for p in paths] != cfg["cloud_source_sha256"]:
            raise ValueError("云端源数据与提交时冻结的源数据不同")
    interpreters = [settings.lerobot_python, settings.isaac_python if cfg["dataset"] == "vlnverse" else settings.habitat_python]
    if any(not p.is_file() or not os.access(p, os.X_OK) for p in interpreters):
        raise ValueError("云镜像缺少采集或导出解释器")
    bundle = settings.rvq_bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    files = manifest["codebook_files"] + manifest.get("distance_files", [])
    files += [manifest[k] for k in ("jacobian_weights_file", "alpha_file") if manifest.get(k)]
    if any(not readable_file(bundle / name) for name in files):
        raise ValueError("云端 RVQ 码本不完整")
    estimate = estimate_gt(cfg)
    if not estimate["can_start"]:
        # Previously completed outputs already occupy part of the peak budget.
        # Failed scratch is deliberately not credited as reusable capacity.
        directory = settings.runs_root / cfg["cloud_run_id"]
        reusable = [directory / "dataset.work/shards", directory / "dataset"]
        if (directory / "stages/capture/success.json").is_file():
            reusable.extend([directory / "capture", directory / "capture_archive"])
        present = sum(p.stat().st_size for root in reusable for p in root.rglob("*") if p.is_file()) / 1024**3
        if estimate["free_gb"] < max(settings.minimum_free_gb, estimate["estimated_peak_gb"] - present):
            raise ValueError("云端持久化存储剩余容量不足")


def execute(args, directory):
    from sqlalchemy import select
    from apps.api.db import Base, Artifact, Run, engine, session, update_run
    from workers.tasks import collect_gt, Paused, Cancelled
    cfg = json.loads((directory / "config.json").read_text())
    Base.metadata.create_all(engine)
    with session() as db:
        db.add(Run(id=args.run_id, request_key=args.run_id, kind="gt", name=cfg.get("name", ""), config=cfg))
    status = "failed"
    try:
        validate_environment({**cfg, "cloud_run_id": args.run_id})
        collect_gt(args.run_id, cfg, publish_catalog=False)
        status = "succeeded"
    except (Paused, Cancelled):
        status = "yielded"
    except Exception as exc:
        # Full local stage logs remain on the volume; keep this manifest free of secrets.
        print(f"Cloud pipeline failed: {type(exc).__name__}", flush=True)
    with session() as db:
        artifacts = [{"kind": a.kind, "path": str(Path(a.path).relative_to(directory)), "details": a.details}
                     for a in db.scalars(select(Artifact).where(Artifact.run_id == args.run_id))]
    atomic_json(directory / "cloud_result.json", {"generation": args.generation, "status": status, "artifacts": artifacts})
    return 0 if status == "succeeded" else 75 if status == "yielded" else 1


def terminate_group(process):
    # All cloud stage runners inherit this group (unlike detached local runners).
    # A SIGSTOP would leave GPU memory allocated and is deliberately never used.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Even if the group leader exited, a renderer may have ignored TERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(args, directory):
    with (directory / "cloud_worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75  # Another container still owns these checkpoints.
        if action(directory, args.generation) != "run":
            atomic_json(directory / "cloud_result.json", {"generation": args.generation, "status": "yielded", "artifacts": []})
            return 75
        recover_interrupted(directory)
        stop_requested = False
        def stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with tempfile.TemporaryDirectory(prefix="vln-cloud-") as runtime:
            env = {**os.environ, "VLN_CLOUD_WORKER": "true", "VLN_GPU_ID": "0",
                   "VLN_RESOURCE_ROOT": str(args.storage_root), "VLN_RUNTIME_ROOT": runtime,
                   "VLN_DATABASE_URL": "sqlite:///" + runtime + "/worker.sqlite", "PYTHONUNBUFFERED": "1"}
            command = [sys.executable, "-m", "workers.cloud_entry", *sys.argv[1:], "--execute"]
            process = subprocess.Popen(command, env=env, start_new_session=True, pass_fds=(lock.fileno(),))
            requested_at = None
            try:
                while process.poll() is None:
                    intent = "pause" if stop_requested else action(directory, args.generation)
                    if intent != "run":
                        requested_at = requested_at or time.monotonic()
                        if intent == "cancel" or time.monotonic() - requested_at >= args.grace_seconds:
                            terminate_group(process)
                            atomic_json(directory / "cloud_result.json", {"generation": args.generation,
                                        "status": "yielded", "artifacts": []})
                            return 75
                    # Read the disposable DB only after the child creates it.
                    try:
                        import sqlite3
                        with sqlite3.connect(f"file:{runtime}/worker.sqlite?mode=ro", uri=True, timeout=1) as db:
                            record = db.execute("SELECT stage, progress FROM runs WHERE id=?", (args.run_id,)).fetchone()
                        if record:
                            atomic_json(directory / "cloud_progress.json", {"generation": args.generation,
                                        "stage": record[0], "progress": json.loads(record[1]), "updated": time.time()})
                    except (sqlite3.Error, OSError, ValueError):
                        pass
                    time.sleep(1)
                return process.returncode
            finally:
                # Includes orphaned renderers after a wrapper failure.
                terminate_group(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=lambda v: str(UUID(v)), required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--storage-id", required=True)
    parser.add_argument("--grace-seconds", type=int, default=15)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.storage_root = args.storage_root.resolve()
    if (args.storage_root / ".vln-volume-id").read_text().strip() != args.storage_id:
        raise ValueError("云持久化存储卷标不匹配")
    directory = args.storage_root / "runs" / args.run_id
    # The controller must have written the config onto the SAME volume.
    if not (directory / "config.json").is_file():
        raise ValueError("未找到任务配置；请核对本机与云端是否挂载同一持久化盘")
    if (directory / "config.json").stat().st_uid != os.geteuid():
        raise ValueError("云任务运行 UID 必须与任务配置文件的所有者一致；请检查镜像用户和云任务运行身份")
    return execute(args, directory) if args.execute else supervise(args, directory)


if __name__ == "__main__":
    raise SystemExit(main())
