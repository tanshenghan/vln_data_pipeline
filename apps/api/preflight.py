import json
import shutil
import subprocess
import time
from dotenv import dotenv_values
from .settings import settings
from .capacity import estimate_gt

_service_cache = (0, {})


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
    root = settings.resource_root
    checks = {
        "habitat": settings.habitat_python.exists(), "lerobot": settings.lerobot_python.exists(),
        "catalog": settings.catalog_db.exists(), "rvq": (root / "action_tokenizer/manifest.json").exists(),
        "r2r": (settings.dataset_root("r2r") / "train/train_gt.json.gz").exists(),
        "rxr": (settings.dataset_root("rxr") / "train/train_guide_gt.json.gz").exists(),
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
    return {"ready": all(checks.values()), "checks": checks, "seedance_ready": seedance_ready, "gpu": gpu, "free_gb": round(free, 1),
            "gpu_id": settings.gpu_id, "workstation": True, "validation": verified}


def validate_gt(cfg):
    h = health()
    if not h["ready"]:
        raise ValueError("工作站采集环境未就绪：" + "、".join(k for k,v in h["checks"].items() if not v))
    try:
        estimate = estimate_gt(cfg)
    except FileNotFoundError as exc:
        raise ValueError("缺少源数据或专家 GT：" + str(exc)) from exc
    bundle = settings.resource_root / "action_tokenizer"
    manifest = json.loads((bundle / "manifest.json").read_text())
    names = manifest["codebook_files"] + manifest.get("distance_files", [])
    names += [manifest[k] for k in ("jacobian_weights_file", "alpha_file") if manifest.get(k)]
    if any(not (bundle / name).is_file() for name in names):
        raise ValueError("RVQ 码本文件不完整")
    if not estimate["can_start"]:
        raise ValueError(f"磁盘容量不足：保守峰值预算 {estimate['estimated_peak_gb']} GiB，可用 {estimate['free_gb']} GiB；请减少采集范围或配置更大的数据盘")
    return estimate
