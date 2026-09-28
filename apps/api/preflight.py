import json
import shutil
import subprocess
import time
import socket
from urllib.parse import urlparse
from dotenv import dotenv_values
from .settings import settings
from .capacity import estimate_gt

_service_cache = (0, {})


def readable_file(path):
    """Shared paths may exist but be inaccessible to this service account."""
    try:
        with path.open("rb") as stream:
            stream.read(1)
        return True
    except OSError:
        return False


def services():
    global _service_cache
    if time.monotonic() - _service_cache[0] > 5:
        names = ["vln-worker", "vln-beat", "vln-atlas"]
        try:
            result = subprocess.run(["systemctl", "--user", "is-active", *names], capture_output=True, text=True, timeout=4)
            states = result.stdout.splitlines()
            checks = {name: i < len(states) and states[i] == "active" for i, name in enumerate(names)}
        except (OSError, subprocess.TimeoutExpired):
            checks = {name: False for name in names}
        _service_cache = (time.monotonic(), checks)
    return _service_cache[1]


def health():
    root = settings.runs_root
    checks = {
        "habitat": settings.habitat_python.exists(), "lerobot": settings.lerobot_python.exists(),
        "catalog": readable_file(settings.catalog_db), "rvq": readable_file(settings.rvq_bundle / "manifest.json"),
        "r2r": all(readable_file(settings.dataset_root("r2r") / "train" / name)
                   for name in ("train.json.gz", "train_gt.json.gz")),
        "rxr": all(readable_file(settings.dataset_root("rxr") / "train" / name)
                   for name in ("train_guide.json.gz", "train_guide_gt.json.gz")),
    }
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        gpu = r.stdout.strip()
        checks["gpu"] = r.returncode == 0
        checks["gpu"] = checks["gpu"] and 0 <= settings.gpu_id < len(r.stdout.strip().splitlines())
    except (OSError, subprocess.TimeoutExpired):
        gpu = "不可用"
        checks["gpu"] = False
    free = shutil.disk_usage(root).free / 1024**3 if root.exists() else 0
    checks["disk"] = free >= settings.minimum_free_gb
    report = settings.runtime_root / "environment_validation.json"
    verified = json.loads(report.read_text()) if report.exists() else {}
    checks["runtime_verified"] = bool(verified.get("habitat") and verified.get("lerobot"))
    checks.update(services())
    seedance_ready = bool(dotenv_values(settings.navgen_env).get("ARK_API_KEY")) if settings.navgen_env.exists() else False
    isaac_report = settings.runtime_root / "vlnverse_environment_validation.json"
    isaac_verified = json.loads(isaac_report.read_text()) if isaac_report.exists() else {}
    endpoint=urlparse(settings.lightnav_endpoint)
    try:
        with socket.create_connection((endpoint.hostname,endpoint.port or 80),timeout=1):lightnav_ready=True
    except OSError:lightnav_ready=False
    lightnav_report = settings.runtime_root / "vlnverse_lightnav_validation.json"
    lightnav_verified = json.loads(lightnav_report.read_text()) if lightnav_report.exists() else {}
    lightnav_ready = lightnav_ready and bool(lightnav_verified.get("complete"))
    from data_pipeline.scalevln_source import availability
    scalevln = availability(settings.dataset_root("scalevln"), settings.scenes_root)
    scalevln.pop("scene_paths", None)
    scale_report = settings.runtime_root / "scalevln_environment_validation.json"
    scalevln["validated"] = bool(json.loads(scale_report.read_text()).get("complete")) if scale_report.exists() else False
    from data_pipeline import srdf_source
    srdf = srdf_source.availability(settings.dataset_root("srdf"), settings.scenes_root)
    return {"scalevln": scalevln, "srdf": srdf, "lightnav_ready": lightnav_ready, "lightnav_validation": lightnav_verified,
            "vlnverse_ready": bool(isaac_verified.get("complete") and settings.isaac_python.exists() and seedance_ready),
            "vlnverse_validation": isaac_verified, "ready": all(checks.values()), "checks": checks, "seedance_ready": seedance_ready, "gpu": gpu, "free_gb": round(free, 1),
            "gpu_id": settings.gpu_id, "workstation": True, "validation": verified}


def validate_gt(cfg):
    h = health()
    if not h["ready"]:
        raise ValueError("工作站采集环境未就绪：" + "、".join(k for k,v in h["checks"].items() if not v))
    if cfg["dataset"] == "scalevln" and not h.get("scalevln", {}).get("can_collect"):
        raise ValueError("ScaleVLN 源数据、HM3D 场景或 navmesh 尚未就绪")
    if cfg["dataset"] == "vlnverse" and not h.get("vlnverse_ready"):
        raise ValueError("VLNVerse 原生 USD 渲染环境尚未通过工作站实测")
    if cfg["dataset"] == "vlnverse" and any(not r["has_reference_gt"] for r in cfg.get("source_records",[])) and not h["lightnav_ready"]:
        raise ValueError("无公开路线需要 LightNav 服务，目前服务不可达")
    try:
        estimate = estimate_gt(cfg)
    except FileNotFoundError as exc:
        raise ValueError("缺少源数据或专家 GT：" + str(exc)) from exc
    bundle = settings.rvq_bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    names = manifest["codebook_files"] + manifest.get("distance_files", [])
    names += [manifest[k] for k in ("jacobian_weights_file", "alpha_file") if manifest.get(k)]
    if any(not readable_file(bundle / name) for name in names):
        raise ValueError("RVQ 码本文件不完整或当前账号不可读")
    if not estimate["can_start"]:
        raise ValueError(f"磁盘容量不足：保守峰值预算 {estimate['estimated_peak_gb']} GiB，可用 {estimate['free_gb']} GiB；请减少采集范围或配置更大的数据盘")
    return estimate
