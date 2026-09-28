"""Operator-owned pool configuration. No credentials or job templates in HTTP responses."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apps.api.settings import settings


class Pool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    provider: Literal["volcengine", "baidu"]
    enabled: bool = False
    region: str
    endpoint: str
    queue_id: str
    resource_pool_id: str = ""
    project_name: str = "default"
    # All these queues are observed for waiting foreground jobs. The account
    # must be allowed to see other users' jobs, not only its own submissions.
    monitor_queue_ids: list[str] = Field(default_factory=list)
    access_key_env: str
    secret_key_env: str
    session_token_env: str = ""
    # Optional Baidu CCR: mint a short-lived pull password per attempt instead
    # of persisting registry passwords in job templates or checkpoint files.
    baidu_registry_id: str = Field(default="", pattern=r"^(|ccr-[a-z0-9]+)$")
    baidu_registry_username: str = ""
    template_file: Path
    local_storage_root: Path
    remote_storage_root: Path
    storage_id: str
    remote_repo: Path = Path("/opt/VLN_Data_Pipeline")
    remote_python: str = "/opt/platform/bin/python"
    # Some providers bootstrap containers as root. Drop to the storage owner's
    # numeric identity before opening logs or starting any collection process.
    run_as_user: str = Field(default="", pattern=r"^(|[1-9][0-9]*:[1-9][0-9]*)$")
    worker_env: dict[str, str] = Field(default_factory=dict)
    validated_datasets: list[Literal["r2r", "rxr", "vlnverse"]] = Field(default_factory=list)
    max_active_jobs: int = Field(default=1, ge=1, le=64)
    idle_seconds: int = Field(default=120, ge=0, le=3600)
    cooldown_seconds: int = Field(default=120, ge=0, le=3600)
    lease_seconds: int = Field(default=90, ge=30, le=600)
    stop_grace_seconds: int = Field(default=15, ge=0, le=60)
    max_failures: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_paths(self):
        from urllib.parse import urlsplit
        url = urlsplit(self.endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.query or url.path not in {"", "/"}:
            raise ValueError("云 API endpoint 必须是 HTTPS 服务根地址")
        if not all(p.is_absolute() for p in (self.local_storage_root, self.remote_storage_root, self.remote_repo)):
            raise ValueError("云存储和代码路径必须为绝对路径")
        if self.provider == "baidu" and self.resource_pool_id != "aihc-serverless":
            raise ValueError("百度闲时任务需要全托管资源池，resource_pool_id 必须为 aihc-serverless")
        if self.baidu_registry_id and (self.provider != "baidu" or not self.baidu_registry_username):
            raise ValueError("CCR 临时凭据需要百度资源池和镜像仓库用户名")
        if self.lease_seconds < settings.cloud_poll_seconds * 3:
            raise ValueError("云租约至少为调度轮询间隔的三倍")
        if self.monitor_queue_ids and self.queue_id not in self.monitor_queue_ids:
            raise ValueError("monitor_queue_ids 必须包含采集任务所在队列")
        protected = {"VLN_CLOUD_WORKER", "VLN_DATABASE_URL", "VLN_RESOURCE_ROOT", "VLN_RUNTIME_ROOT", "VLN_GPU_ID"}
        if protected.intersection(self.worker_env):
            raise ValueError("worker_env 不能覆盖执行器管理的数据库、输出或 GPU 配置")
        return self

    def template(self):
        content = self.template_file.read_text()
        if "REPLACE_" in content:
            raise ValueError("请先填写云任务模板中的镜像、资源规格和存储 ID")
        return json.loads(content)

    def digest(self):
        # Operational toggles can change while paused; image, mounts, resource
        # spec and interpreter paths cannot silently change a resumed pipeline.
        excluded = {"enabled", "validated_datasets", "max_active_jobs", "idle_seconds", "cooldown_seconds", "max_failures"}
        value = {"pool": self.model_dump(mode="json", exclude=excluded), "template": self.template()}
        if not self.run_as_user:
            # Preserve fingerprints of existing deployments using image USER.
            value["pool"].pop("run_as_user")
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def check_storage(self):
        marker = self.local_storage_root / ".vln-volume-id"
        if not marker.is_file() or marker.read_text().strip() != self.storage_id:
            raise ValueError(f"{self.name} 的持久化盘未挂载或卷标不匹配")

    def credentials(self):
        file_values = dotenv_values(settings.cloud_credentials_file) if settings.cloud_credentials_file.is_file() else {}
        def get(key):
            return (os.environ.get(key) or file_values.get(key) or "") if key else ""
        ak, sk = get(self.access_key_env), get(self.secret_key_env)
        if not ak or not sk:
            raise ValueError(f"{self.name} 缺少 AK/SK 配置")
        return ak, sk, get(self.session_token_env)

    def ready(self):
        if not self.enabled:
            return "未启用"
        try:
            self.credentials()
            self.check_storage()
            self.template()
            if not self.validated_datasets:
                return "尚未验证云端采集镜像和数据"
        except (OSError, ValueError) as exc:
            return str(exc)
        return ""


def load_pools():
    path = settings.cloud_pools_file
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("pools"), list):
        raise ValueError("云资源池配置必须包含 pools 列表")
    pools = []
    for item in raw["pools"]:
        if not isinstance(item, dict) or "template_file" not in item:
            raise ValueError("云资源池缺少任务模板配置")
        item = dict(item)
        item["template_file"] = str((path.parent / item["template_file"]).resolve())
        pools.append(Pool.model_validate(item))
    if len({p.id for p in pools}) != len(pools):
        raise ValueError("云资源池 ID 不得重复")
    return pools


def cloud_status():
    try:
        pools = load_pools()
    except (OSError, ValueError, KeyError):
        return {"pools": [], "ready": False, "error": "云资源池配置无效，请检查服务日志与配置文件"}
    try:
        snapshot = json.loads((settings.runtime_root / "cloud_status.json").read_text())
    except (OSError, ValueError):
        snapshot = {}
    fresh = 0 <= time.time() - snapshot.get("updated", 0) <= max(60, settings.cloud_poll_seconds * 4)
    items = []
    for pool in pools:
        reason = pool.ready()
        live = snapshot.get("pools", {}).get(pool.id, {}) if fresh else {}
        ready = not reason and bool(live.get("observed"))
        items.append({"id": pool.id, "name": pool.name, "provider": pool.provider, "enabled": pool.enabled,
                      "ready": ready, "datasets": pool.validated_datasets,
                      "reason": reason or live.get("reason") or ("" if ready else "等待云调度服务连接"),
                      "pending_foreground": live.get("pending_foreground"),
                      "active_jobs": live.get("active_jobs", 0), "max_active_jobs": pool.max_active_jobs})
    return {"pools": items, "ready": any(p["ready"] for p in items), "scheduler_alive": fresh,
            "updated": snapshot.get("updated")}


def validate_cloud(cfg):
    from apps.api.capacity import estimate_gt
    from data_pipeline.vlnce_source import source_paths
    from data_pipeline.collect_r2r import sha256
    status = cloud_status()
    available = {p["id"] for p in status["pools"] if p["ready"] and cfg["dataset"] in p["datasets"]}
    target = cfg["compute_pool"]
    pools = [p for p in load_pools() if p.id in available and (
        target in {"auto", p.id} or
        (target == "volc-h20" and p.provider == "volcengine" and p.id.startswith("volc-h20-"))
    )]
    if not pools:
        raise ValueError("没有就绪且支持该数据集的云资源池；请配置 AK/SK、队列、镜像及持久化存储，并启动云调度服务")
    estimates = []
    for pool in pools:
        pool.check_storage()
        estimate = estimate_gt(cfg, storage_root=pool.local_storage_root / "runs", verify_scenes=False)
        if estimate["can_start"]:
            estimates.append((pool, estimate))
    if not estimates:
        raise ValueError("云端持久化盘容量不足，请减少采集范围或扩容")
    cfg["eligible_pools"] = [p.id for p, _ in estimates]
    if cfg["dataset"] != "vlnverse":
        paths = source_paths(settings.dataset_root(cfg["dataset"]), cfg["dataset"], cfg["split"], cfg["role"])
        cfg["cloud_source_sha256"] = [sha256(p) for p in paths]
    return {**estimates[0][1], "execution": "cloud_idle", "gpu_id": None,
            "pools": [p.name for p, _ in estimates],
            "basis": "云端挂载持久化盘容量预算；容器启动后复核源数据、场景、码本与采集环境。"}
