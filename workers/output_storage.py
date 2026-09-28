"""Opt-in sharing of run outputs only; never widen credential/runtime permissions."""
import os
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from apps.api.settings import settings

_archive_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vln-output-archive")
_archive_jobs = {}


def poll_cloud_archive(run_id, directory):
    """Copy outside the controller thread so other GPU leases keep renewing."""
    source = Path(directory).resolve()
    if not settings.shared_run_permissions or settings.cloud_worker:
        return source
    marker = settings.runs_root / run_id / ".shared-cloud-output.json"
    if marker.is_file():
        result = cloud_archive(run_id, source)
        _archive_jobs.pop(run_id, None)
        return result
    job = _archive_jobs.get(run_id)
    if job:
        if job["source"] != source:
            raise ValueError("正在归档的云产物来源发生变化")
        if not job["future"].done():
            return None
        try:
            return job["future"].result()
        except Exception:
            if job["retry_at"] is None:
                job["retry_at"] = time.monotonic() + 60
                raise
            if time.monotonic() < job["retry_at"]:
                raise
    _archive_jobs[run_id] = {"source": source, "retry_at": None,
        "future": _archive_executor.submit(cloud_archive, run_id, source)}
    return None


def is_shared_output(path):
    return (settings.shared_run_permissions and not settings.cloud_worker
            and Path(path).resolve().is_relative_to(settings.runs_root.resolve()))


def share_run_outputs(directory):
    directory = Path(directory)
    if not is_shared_output(directory) or directory.is_symlink():
        return
    # Do not follow source-asset links outside the run. Explicitly private files
    # produced by export libraries need this even when the parent has default ACLs.
    for base, dirs, files in os.walk(directory, followlinks=False):
        for path in [Path(base), *(Path(base) / name for name in files)]:
            if path.is_symlink():
                continue
            mode = path.stat().st_mode & 0o7777
            path.chmod(mode | (0o777 if path.is_dir() else 0o666))


def cloud_archive(run_id, directory):
    """Publish completed cloud outputs locally; remote checkpoints stay on PFS."""
    source = Path(directory).resolve()
    target = settings.runs_root / run_id
    if not settings.shared_run_permissions or settings.cloud_worker:
        return source
    if target.parent.resolve() != settings.runs_root.resolve() or target.name != run_id:
        raise ValueError("Invalid run archive destination")
    marker = target / ".shared-cloud-output.json"
    if target.is_symlink():
        raise ValueError("共享输出目标不能是符号链接")
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text()).get("source") == str(source):
            return target
        raise ValueError("共享目录存在同名未验收产物，禁止覆盖")
    names = [name for name in ("dataset", "capture", "capture_archive", "trajectory_review",
        "stages", "config.json", "episode_ids.json") if (source / name).exists()]
    files = []
    for name in names:
        root = source / name
        paths = [root, *root.rglob("*")] if root.is_dir() else [root]
        for path in paths:
            if path.is_symlink():
                raise ValueError("云产物含符号链接，需核对后归档")
            if path.is_file():
                files.append(path)
    needed = sum(p.stat().st_size for p in files)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(settings.runs_root).free < needed + settings.minimum_free_gb * 1024**3:
        raise ValueError("共享输出盘空间不足，云端原始产物仍保留")
    staging = Path(tempfile.mkdtemp(prefix=f".cloud-{run_id}-", dir=settings.runs_root))
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()
    try:
        for name in names:
            root = source / name
            if root.is_dir():
                shutil.copytree(root, staging / name)
            else:
                shutil.copy2(root, staging / name)
        hashes = {}
        for path in files:
            relative = path.relative_to(source)
            value = digest(path)
            if digest(staging / relative) != value:
                raise ValueError("云产物归档 SHA256 校验失败")
            hashes[str(relative)] = value
        (staging / marker.name).write_text(json.dumps({"source": str(source), "files": hashes,
            "bytes": needed, "complete": True}, ensure_ascii=False, indent=2) + "\n")
        share_run_outputs(staging)
        if target.exists() or target.is_symlink():
            raise ValueError("归档发布前出现同名目录，禁止覆盖")
        staging.rename(target)
        return target
    except Exception:
        # Only our unpublished mkdtemp copy is removed; source and published runs remain intact.
        shutil.rmtree(staging)
        raise
