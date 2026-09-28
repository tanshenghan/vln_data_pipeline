"""Queue small native idle acceptance jobs, then enable only verified datasets.

Run with the controller's environment and database. Jobs use the normal durable
scheduler, including foreground demand, cooldown, leases and manual controls.
A persistent ledger makes restarting this command reuse the same jobs.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time
from uuid import uuid4

from apps.api.catalog_service import resolve_selection
from apps.api.db import CloudExecution, Run, session
from apps.api.schemas import RunRequest
from apps.api.settings import settings
from data_pipeline.collect_r2r import sha256
from data_pipeline.vlnce_source import source_paths
from workers.cloud_config import load_pools
from workers.cloud_providers import CloudError, Provider
from workers.command_runner import atomic_json


def current_pool(pool_id):
    return next(p for p in load_pools() if p.id == pool_id)


def update_pool(pool, datasets=(), *, expected_digest):
    """Read-modify-write only after rechecking the deployment being accepted."""
    original = settings.cloud_pools_file.read_bytes()
    if current_pool(pool.id).digest() != expected_digest:
        raise ValueError("资源池执行配置已改变，保留验收结果但不自动启用")
    value = json.loads(original)
    raw = next(p for p in value["pools"] if p["id"] == pool.id)
    raw["enabled"] = True
    raw["validated_datasets"] = sorted(set(raw.get("validated_datasets", [])) | set(datasets))
    if settings.cloud_pools_file.read_bytes() != original:
        raise ValueError("资源池配置正在被修改，请稍后重新执行验收检查")
    atomic_json(settings.cloud_pools_file, value)


def accept(pool, records, *, expected_digest):
    """Require matching bindings, cloud success and independent output reports."""
    provider = Provider(pool)
    try:
        for dataset, record in records.items():
            with session() as db:
                row = db.get(Run, record["id"])
                binding = db.get(CloudExecution, record["id"])
            if (not row or row.status != "succeeded" or row.control != "run" or
                    row.config.get("dataset") != dataset or not binding or binding.state != "done" or
                    binding.pool_id != pool.id or binding.pool_digest != expected_digest):
                raise ValueError("验收任务尚未成功，或执行配置与当前资源池不同")
            job = provider.get(binding.job_id)
            if job.name != binding.job_name or job.state != "succeeded" or not job.idle:
                raise ValueError("云平台未确认验收任务以闲时模式成功完成")
            directory = Path(binding.directory)
            result = json.loads((directory / "cloud_result.json").read_text())
            report = json.loads((directory / "dataset/meta/platform_validation.json").read_text())
            if (result.get("generation") != binding.generation or result.get("status") != "succeeded" or
                    not all(report.get(k) is True for k in ("complete", "self_contained", "official_loader_all_frames",
                                                           "all_30_source_arrays_exact", "geometry_recomputed"))):
                raise ValueError("验收产物未通过完整独立校验")
        update_pool(pool, records, expected_digest=expected_digest)
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--datasets", nargs="+", choices=["r2r", "rxr"], default=["r2r", "rxr"])
    args = parser.parse_args()
    pool = current_pool(args.pool)
    pool.credentials()
    pool.check_storage()
    pool.template()
    path = settings.runtime_root / "cloud-validation" / f"{pool.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(pool, args.datasets, path)


def verify(pool, datasets, path):
    ledger = json.loads(path.read_text()) if path.is_file() else {"pool_digest": pool.digest(), "runs": {}}
    if ledger["pool_digest"] != pool.digest():
        raise ValueError("资源池与已记录验收配置不同，请核对后另建验收记录；不会重复提交旧任务")
    expected_digest = ledger["pool_digest"]
    records = ledger["runs"]
    for dataset in dict.fromkeys(datasets):
        if dataset not in records:
            cfg = resolve_selection(RunRequest(dataset=dataset, count=1, execution="cloud_idle", compute_pool=pool.id,
                                    name=f"云端验收 · {pool.name} · {dataset.upper()}").model_dump(mode="json"))
            cfg["eligible_pools"] = [pool.id]
            cfg["cloud_source_sha256"] = [sha256(p) for p in source_paths(settings.dataset_root(dataset), dataset,
                                                                        cfg["split"], cfg["role"])]
            records[dataset] = {"id": str(uuid4()), "config": cfg}
            atomic_json(path, ledger)  # Persist identity before any database insert.
        record = records[dataset]
        with session() as db:
            if not db.get(Run, record["id"]):
                db.add(Run(id=record["id"], request_key="cloud-acceptance-" + record["id"], kind="gt",
                           name=record["config"]["name"], config=record["config"]))
    update_pool(pool, expected_digest=expected_digest)
    selected = {dataset: records[dataset] for dataset in dict.fromkeys(datasets)}
    previous = None
    while True:
        if current_pool(pool.id).digest() != expected_digest:
            raise ValueError("资源池执行配置改变，停止自动验收")
        with session() as db:
            states = {dataset: db.get(Run, record["id"]).status for dataset, record in selected.items()}
        if states != previous:
            print(json.dumps({"pool": pool.id, "states": states,
                              "runs": {k: v["id"] for k, v in selected.items()}}, ensure_ascii=False), flush=True)
            previous = states
        if any(state in {"failed", "cancelled"} for state in states.values()):
            raise ValueError("验收任务失败或已取消；请检查对应任务日志，不会启用该数据集")
        if all(state == "succeeded" for state in states.values()):
            try:
                accept(pool, selected, expected_digest=expected_digest)
            except (OSError, CloudError) as exc:
                # A temporary gateway/API outage must not abandon queued
                # acceptance or cause another GPU job to be submitted.
                print(f"验收结果暂不可读（{type(exc).__name__}），保留任务等待重查", flush=True)
                time.sleep(15)
                continue
            ledger["accepted"] = sorted(set(ledger.get("accepted", [])) | set(selected))
            atomic_json(path, ledger)
            print(json.dumps({"pool": pool.id, "validated_datasets": ledger["accepted"]}), flush=True)
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
