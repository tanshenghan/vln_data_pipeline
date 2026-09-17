import json
import shutil
import subprocess
from dotenv import dotenv_values
from .settings import settings


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
    except (OSError, subprocess.TimeoutExpired):
        gpu = "不可用"
        checks["gpu"] = False
    free = shutil.disk_usage(root).free / 1024**3 if root.exists() else 0
    checks["disk"] = free >= settings.minimum_free_gb
    report = settings.runtime_root / "environment_validation.json"
    verified = json.loads(report.read_text()) if report.exists() else {}
    checks["runtime_verified"] = bool(verified.get("habitat") and verified.get("lerobot"))
    seedance_ready = bool(dotenv_values(settings.navgen_env).get("ARK_API_KEY")) if settings.navgen_env.exists() else False
    return {"ready": all(checks.values()), "checks": checks, "seedance_ready": seedance_ready, "gpu": gpu, "free_gb": round(free, 1),
            "workstation": True, "validation": verified}


def validate_gt(cfg):
    h = health()
    if not h["ready"]:
        raise ValueError("工作站采集环境未就绪：" + "、".join(k for k,v in h["checks"].items() if not v))
    root = settings.dataset_root(cfg["dataset"])
    name = cfg["split"] + ("_" + cfg["role"] if cfg["dataset"] == "rxr" else "")
    for f in [root / cfg["split"] / f"{name}.json.gz", root / cfg["split"] / f"{name}_gt.json.gz"]:
        if not f.is_file():
            raise ValueError(f"缺少源数据或 GT：{f.name}")
    for record in cfg.get("source_records", []):
        scene = settings.resource_root / "scene_datasets/mp3d" / record["scene"] / f"{record['scene']}.glb"
        if not scene.exists() or not scene.with_suffix(".navmesh").exists():
            raise ValueError(f"场景尚未就绪：{record['scene']}")
