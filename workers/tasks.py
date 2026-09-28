"""Durable orchestration around the unmodified upstream production stages."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from sqlalchemy import select
from apps.api.db import Artifact, CloudExecution, Run, now, session, update_run
from apps.api.settings import ROOT, settings
from .celery_app import celery
from .command_runner import atomic_json, identity


class Paused(Exception):
    pass


class Cancelled(Exception):
    pass


def run_dir(run_id):
    if not settings.cloud_worker:
        with session() as db:
            binding = db.get(CloudExecution, run_id)
            if binding:
                archived = settings.runs_root / run_id
                if settings.shared_run_permissions and (archived / ".shared-cloud-output.json").is_file():
                    return archived
                return Path(binding.directory)
    return settings.runs_root / run_id


def control(run_id):
    if settings.cloud_worker:
        # The supervisor also enforces the lease while a stage owns the GPU.
        return json.loads((run_dir(run_id) / "cloud_control.json").read_text())["action"]
    with session() as db:
        return db.get(Run, run_id).control


def execute(run_id, stage, argv, env=None):
    root = run_dir(run_id) / "stages" / stage
    root.mkdir(parents=True, exist_ok=True)
    spec = {"argv": list(map(str, argv)), "cwd": str(ROOT), "env": env or {}}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if stage in {"capture", "collect_shards"}:
        spec["gpu_lock"] = str(settings.runtime_root / "gpu.lock")
    successful = root / "success.json"
    if successful.exists():
        if json.loads(successful.read_text())["digest"] != digest:
            raise RuntimeError("已完成阶段的配置发生变化，请创建新任务")
        return
    intent = control(run_id)
    if intent == "pause":
        raise Paused()
    if intent == "cancel":
        raise Cancelled()
    attempts = sorted(root.glob("attempt-*"))
    last = attempts[-1] if attempts else None
    if last and (last / "command.json").exists():
        previous = json.loads((last / "command.json").read_text())
        if {k:v for k,v in previous.items() if k != "gpu_lock"} != {k:v for k,v in spec.items() if k != "gpu_lock"}:
            raise RuntimeError("阶段执行参数发生变化，请创建新任务")
        if (last / "exit.json").exists() and json.loads((last / "exit.json").read_text())["returncode"] == 0:
            # A worker can stop after the runner finishes but before acknowledging success.
            atomic_json(successful, {"digest": digest, "attempt": last.name, "finished": time.time()})
            return
    attached = False
    if last and (last / "process.json").exists() and not (last / "exit.json").exists():
        info = json.loads((last / "process.json").read_text())
        attached = identity(info["pid"]) == info["start"]
        if not attached:
            raise RuntimeError(f"阶段 {stage} 执行结果未知；保留产物并核对后再续跑")
    runner = None
    if not attached:
        last = root / f"attempt-{len(attempts)+1:04d}"
        last.mkdir()
        atomic_json(last / "command.json", spec)
        runner = subprocess.Popen([sys.executable, "-m", "workers.command_runner", str(last)],
                         cwd=ROOT, start_new_session=not settings.cloud_worker,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    update_run(run_id, stage=stage, status="running", heartbeat=now())
    while not (last / "exit.json").exists():
        if runner is not None and runner.poll() is not None and not (last / "exit.json").exists():
            raise RuntimeError(f"阶段 {stage} 执行器异常退出，无完成记录；保留现场等待核对")
        intent = control(run_id)
        if stage == "collect_shards":
            atomic_json(run_dir(run_id) / "control.json", {"action": intent})
        if intent == "cancel" and not settings.cloud_worker and (last / "process.json").exists():
            info = json.loads((last / "process.json").read_text())
            if identity(info["pid"]) == info["start"]:
                os.killpg(info["pid"], signal.SIGTERM)
        progress = {"stage": stage, "log": str(last.relative_to(run_dir(run_id))) + "/output.log"}
        log_path = last / "output.log"
        if log_path.exists():
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 12000))
                for line in handle.read().decode("utf-8", errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            for key in ("completed", "total", "episode_id", "frames", "stage", "task_id"):
                                if key in event:
                                    progress[key] = event[key]
                    except ValueError:
                        continue
        batch_status = run_dir(run_id) / "dataset.work/status.json"
        if batch_status.exists():
            try:
                progress.update(json.loads(batch_status.read_text()))
            except (ValueError, OSError):
                pass
        update_run(run_id, heartbeat=now(), progress=progress)
        if (last / "process.json").exists():
            info = json.loads((last / "process.json").read_text())
            if identity(info["pid"]) != info["start"] and not (last / "exit.json").exists():
                raise RuntimeError(f"阶段 {stage} 进程已退出但无完成记录，请核对产物")
        time.sleep(2)
    result = json.loads((last / "exit.json").read_text())
    if control(run_id) == "cancel":
        raise Cancelled()
    if result["returncode"] == 75 and stage == "collect_shards":
        raise Paused()
    if result["returncode"]:
        raise RuntimeError(f"阶段 {stage} 失败，退出码 {result['returncode']}，查看阶段日志")
    atomic_json(successful, {"digest": digest, "attempt": last.name, "finished": time.time()})


def common_env():
    return {"HABITAT_PYTHON": str(settings.habitat_python), "LEROBOT_PYTHON": str(settings.lerobot_python),
            "BETA_GPU_ID": str(settings.gpu_id), "BETA_SCENES_DIR": str(settings.scenes_root),
            "BETA_RVQ_BUNDLE": str(settings.rvq_bundle),
            "BETA_VLNVERSE_HDR": str(settings.vlnverse_hdr),
            "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT), "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1"}


def archive_failed_output(run_id, stage, output):
    """Only archive this run's incomplete output after a recorded nonzero exit."""
    root = run_dir(run_id) / "stages" / stage
    if (root / "success.json").exists() or not output.exists():
        return
    attempts = sorted(root.glob("attempt-*"))
    if not attempts:
        raise RuntimeError(f"{stage} 输出已存在且无执行记录，需要核对")
    exit_file = attempts[-1] / "exit.json"
    process_file = attempts[-1] / "process.json"
    if not exit_file.exists() and process_file.exists():
        info = json.loads(process_file.read_text())
        if identity(info["pid"]) == info["start"]:
            return
    if exit_file.exists() and json.loads(exit_file.read_text())["returncode"] == 0:
        # execute() will recover the completed stage without overwriting its output.
        return
    if not exit_file.exists():
        raise RuntimeError(f"{stage} 输出状态不明确，禁止覆盖")
    archived = run_dir(run_id) / "failed_outputs"
    archived.mkdir(exist_ok=True)
    output.rename(archived / f"{stage}-{time.time_ns()}")


def register(run_id, kind, path, details=None):
    path = Path(path).resolve()
    artifact_id = hashlib.sha256(f"{run_id}:{path}".encode()).hexdigest()[:24]
    with session() as db:
        existing = db.get(Artifact, artifact_id)
        if existing:
            existing.details = details or {}
            existing.kind = kind
        else:
            db.add(Artifact(id=artifact_id, run_id=run_id, kind=kind, name=path.name,
                            path=str(path), details=details or {}))


def collect_gt(run_id, cfg, *, publish_catalog=True):
    directory = run_dir(run_id)
    env = common_env()
    env["BETA_DATA_ROOT"] = str(settings.dataset_root(cfg["dataset"]))
    output = directory / "dataset"
    if cfg["full_dataset"]:
        archive_failed_output(run_id, "collect_shards", output)
        atomic_json(directory / "control.json", {"action": control(run_id)})
        execute(run_id, "collect_shards", [settings.lerobot_python, "data_pipeline/build_r2r_train.py",
            "--output-dir", output, "--dataset", cfg["dataset"], "--split", cfg["split"],
            "--role", cfg["role"], "--languages", *cfg["languages"], "--data-root", env["BETA_DATA_ROOT"],
            "--scenes-dir", env["BETA_SCENES_DIR"], "--rvq-bundle", env["BETA_RVQ_BUNDLE"],
            "--workers", cfg["workers"], "--gpus", settings.gpu_id,
            "--routes-per-shard", cfg["routes_per_shard"], "--control-file", directory / "control.json"], env)
    else:
        ids = directory / "episode_ids.json"
        atomic_json(ids, cfg["episode_ids"])
        raw = directory / "capture"
        archive_failed_output(run_id, "capture", raw)
        if cfg["dataset"] == "vlnverse":
            roots = [arg for root in settings.vlnverse_scene_roots if root for arg in ["--scene-root", root]]
            execute(run_id, "capture", [sys.executable, "-m", "workers.vlnverse_capture", settings.isaac_python,
                "--config", directory / "config.json", "--output", raw,
                "--rvq-bundle", env["BETA_RVQ_BUNDLE"], *roots],
                {**env, "OMNI_KIT_ACCEPT_EULA": "YES"})
            execute(run_id, "vlm_review", [sys.executable, "data_pipeline/review_vlnverse.py", raw,
                "--env-file", settings.navgen_env], env)
            register_vlnverse_candidates(run_id, raw, publish_catalog=publish_catalog)
            review = json.loads((raw / "vlm_review_summary.json").read_text())
            if not review["accepted"]:
                register(run_id, "capture", raw, {"candidate":True,"ground_truth":False})
                return
            execute(run_id, "annotate", [settings.lerobot_python, "data_pipeline/collect_vlnverse.py",
                "--output", raw, "--annotate"], env)
        else:
            execute(run_id, "capture", ["bash", "data_pipeline/run.sh", raw, "-1", "--dataset", cfg["dataset"],
                "--split", cfg["split"], "--role", cfg["role"], "--languages", *cfg["languages"],
                "--episode-ids-file", ids], env)
        execute(run_id, "validate_capture", [settings.lerobot_python, "data_pipeline/validate_r2r.py", raw], env)
        archive_failed_output(run_id, "export", output)
        execute(run_id, "export", [settings.lerobot_python, "data_pipeline/export_lerobot.py", raw,
            "--output-dir", output, "--repo-id", f"beta-nav/{cfg['dataset']}-{run_id[:8]}"], env)
        # Prove the final loader does not depend on original scratch paths, while retaining an audit copy.
        if raw.exists() and cfg["dataset"] != "vlnverse":
            raw.rename(directory / "capture_archive")
        elif cfg["dataset"] == "vlnverse":
            register(run_id,"capture",raw,{"ground_truth":False,"contains_candidates":True})
    execute(run_id, "validate_lerobot", [settings.lerobot_python, "-m", "contracts.check", output,
        "--report", output / "meta/platform_validation.json"], env)
    metadata = json.loads((output / "meta/beta_nav.json").read_text())
    for index, episode in enumerate(metadata["episodes"]):
        eid = str(episode["episode_id"])
        movie = output / "review" / f"episode_{eid}_gt.mp4"
        execute(run_id, f"preview_{index:06d}", [settings.lerobot_python, "visual/render_episode_gt.py", output,
            "--episode-id", eid, "--output", movie, "--fps", "8", "--scale", "1"], env)
        register(run_id, "video", movie, {"episode_id": eid, "ground_truth": True})
    register(run_id, "dataset", output, {"format": "beta_nav_lerobot_v1", "ground_truth": True,
             "episodes": [{k:v for k,v in e.items() if k in {"episode_id","scene_id","frames","instruction","source_metadata"}} for e in metadata["episodes"]]})
    for name in ["platform_validation.json", "beta_nav.json", "info.json", "stats.json"]:
        register(run_id, "report", output / "meta" / name)
    if publish_catalog:
        from .catalog_sync import sync_gt_catalog
        sync_gt_catalog()
        from .sample_catalog import register_samples
        register_samples(run_id)


def register_vlnverse_candidates(run_id, raw, *, publish_catalog=True):
    """Make reviewed candidates inspectable even when none qualifies for export."""
    import shutil
    audit = run_dir(run_id) / "trajectory_review"
    audit.mkdir(exist_ok=True)
    summary = json.loads((raw / "vlm_review_summary.json").read_text())
    shutil.copyfile(raw / "vlm_review_summary.json", audit / "summary.json")
    register(run_id, "report", audit / "summary.json", {"semantic_review": True})
    for report in summary["episodes"]:
        eid=report["episode_id"]
        folder=raw/("episodes" if report["accepted"] else "candidates")/eid
        target=audit/eid;target.mkdir(exist_ok=True)
        for name in ("episode.json","vlm_review.json","vlm_request.json","vlm_response.json","model_responses.jsonl"):
            if (folder/name).exists():
                shutil.copyfile(folder/name,target/name)
                register(run_id,"report",target/name,{"episode_id":eid})
        movie=target/"observations.mp4"
        execute(run_id,"observations_"+eid,["ffmpeg","-nostdin","-y","-loglevel","error","-framerate","8",
            "-i",folder/"rgb/%06d.png","-c:v","libx264","-pix_fmt","yuv420p",movie])
        register(run_id,"video",movie,{"episode_id":eid,"ground_truth":report["accepted"],
            "trajectory_source":report["trajectory_source"],"semantic_decision":report["decision"],"candidate":not report["accepted"]})
    register(run_id,"capture",raw,{"ground_truth":False,"contains_candidates":bool(summary["candidates"])})
    if publish_catalog:
        from .sample_catalog import register_vlnverse_candidates as catalog_candidates
        catalog_candidates(run_id)


def generate_seedance(run_id, cfg):
    try:
        execute(run_id, "seedance_pipeline", [settings.navgen_python, "-m", "workers.seedance_entry", run_id],
                {"PYTHONUNBUFFERED": "1", "VLN_RUN_CONFIG": str(run_dir(run_id) / "config.json"),
                 "VLN_NAVGEN_ENV": str(settings.navgen_env), "VLN_VGGT_REPO": str(settings.vggt_repo)})
    finally:
        register_seedance_artifacts(run_id)


def register_seedance_artifacts(run_id):
    output = run_dir(run_id) / "generation"
    missing = {"depth": "未提供", "calibration": "未提供可信标定", "body_pose": "未提供 GT",
               "fm": "未提供", "rvq": "未提供", "apos": "未提供", "opos": "未提供"}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".mp4", ".json", ".jsonl", ".png", ".jpg", ".txt"}:
            kind = "video" if path.suffix == ".mp4" else "image" if path.suffix in {".png", ".jpg"} else "report"
            register(run_id, kind, path,
                     {"ground_truth": False, "missing": missing, "trajectory": "估计轨迹，非 GT",
                      "relative_path": str(path.relative_to(output))})
    from .sample_catalog import register_samples
    register_samples(run_id)


@celery.task(name="workers.tasks.execute_run")
def execute_run(run_id):
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with session() as db:
            row = db.get(Run, run_id)
            if not row or row.status not in {"queued", "running"}:
                return
            cfg = row.config
            if cfg.get("execution") == "cloud_idle":
                return  # Only the independent cloud controller may dispatch this run.
            row.attempt += 1
            row.error = None
            row.status = "running"
            row.heartbeat = now()
            kind = row.kind
        atomic_json(directory / "config.json", cfg)
        try:
            if kind == "gt":
                # The detached capture runner owns the GPU lock across worker restarts.
                collect_gt(run_id, cfg)
            else:
                generate_seedance(run_id, cfg)
            update_run(run_id, status="succeeded", stage="complete", heartbeat=now(), progress={"complete": True})
        except Paused:
            update_run(run_id, status="paused", heartbeat=now())
        except Cancelled:
            update_run(run_id, status="cancelled", heartbeat=now())
        except Exception as exc:
            update_run(run_id, status="failed", error=str(exc)[:4000], heartbeat=now())
            raise
        finally:
            from .output_storage import share_run_outputs
            share_run_outputs(directory)


@celery.task(name="workers.tasks.reconcile")
def reconcile():
    from .local_collection import refill_local
    refill_local()
    # The per-run file lock is the authoritative guard; DB heartbeat alone never kills/restarts a process.
    with session() as db:
        rows = db.scalars(select(Run).where(Run.status.in_(["queued", "running"]))).all()
        ids = [r.id for r in rows if r.config.get("execution") != "cloud_idle" and
               (r.status == "queued" or not r.heartbeat or r.heartbeat < now() - timedelta(seconds=90))]
    for run_id in ids:
        execute_run.delay(run_id)
