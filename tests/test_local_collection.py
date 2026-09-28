"""Local continuous collection is bounded, restart-safe and never claims a busy GPU."""
import json
import sqlite3
from unittest.mock import Mock

import pytest
from sqlalchemy import select

from apps.api import db
from apps.api.settings import settings
from tests.test_platform import platform
from tests.test_collection_plans import batches, complete
from workers import local_collection as local
from workers.command_runner import atomic_json
from workers.output_storage import share_run_outputs


@pytest.fixture
def workstation(platform, monkeypatch):
    monkeypatch.setattr(local, "gpu_idle", lambda: True)
    monkeypatch.setattr(local, "validate_gt", lambda cfg: {"can_start": True})
    with sqlite3.connect(settings.catalog_db) as catalog:
        catalog.row_factory = sqlite3.Row
        template = dict(catalog.execute("SELECT * FROM episodes LIMIT 1").fetchone())
        for dataset in ("RxR", "R2R", "ScaleVLN"):
            for i in range(5):
                row = dict(template, id=f"{dataset}-{i}", episode_id=str(20000+i), dataset=dataset)
                detail = json.loads(row["detail"])
                detail.update({key: row[key] for key in ("id", "episode_id", "dataset")})
                row["detail"] = json.dumps(detail)
                catalog.execute(f"INSERT INTO episodes ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
    return platform


def create(client, dataset="rxr", **changes):
    response = client.post("/api/collection-plans", json={"dataset": dataset, "execution": "local",
        "compute_pools": ["local"], "routes_per_shard": 2, **changes},
        headers={"Idempotency-Key": "local-plan-" + dataset})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_local_fairness_bounded_restart_and_no_duplicate(workstation):
    client, dispatch = workstation
    a = create(client, "rxr")
    b = create(client, "r2r")
    local.refill_local()
    first = batches(a)
    assert len(first) == 1 and not batches(b)
    assert first[0].config["execution"] == "local"
    for _ in range(3):
        local.refill_local()
    assert len(batches(a)) == 1 and not batches(b)
    complete(first[0])
    local.refill_local()
    assert len(batches(b)) == 1
    complete(batches(b)[0])
    local.refill_local()
    assert len(batches(a)) == 2
    assert not set(batches(a)[0].config["source_ids"]) & set(batches(a)[1].config["source_ids"])
    dispatch.assert_not_called()  # The regular reconcile task dispatches persisted runs.


def test_busy_gpu_and_failed_monitor_do_not_allocate(workstation, monkeypatch):
    client, _ = workstation
    pid = create(client)
    monkeypatch.setattr(local, "gpu_idle", lambda: False)
    local.refill_local()
    assert not batches(pid)
    assert "GPU0" in client.get("/api/collection-plans").json()[0]["error"]


def test_capacity_failure_waits_without_claiming(workstation, monkeypatch):
    client, _ = workstation
    pid = create(client)
    def full(_):
        raise ValueError("磁盘容量不足")
    monkeypatch.setattr(local, "validate_gt", full)
    local.refill_local()
    assert not batches(pid)
    assert "容量不足" in client.get("/api/collection-plans").json()[0]["error"]


def test_running_local_pause_waits_for_actual_stage_exit(workstation):
    client, _ = workstation
    pid = create(client)
    local.refill_local()
    run = batches(pid)[0]
    db.update_run(run.id, status="running")
    assert client.post(f"/api/collection-plans/{pid}/pause").status_code == 200
    assert batches(pid)[0].status == "running"
    assert batches(pid)[0].control == "pause"
    assert client.post(f"/api/collection-plans/{pid}/resume").status_code == 409
    db.update_run(run.id, status="paused")
    assert client.post(f"/api/collection-plans/{pid}/resume").status_code == 200
    assert batches(pid)[0].status == "queued"
    local.refill_local()
    assert len(batches(pid)) == 1


def test_failed_batch_requires_explicit_retry(workstation):
    client, _ = workstation
    pid = create(client)
    local.refill_local()
    db.update_run(batches(pid)[0].id, status="failed")
    local.refill_local()
    assert len(batches(pid)) == 1
    assert client.get("/api/collection-plans").json()[0]["status"] == "needs_attention"
    client.post(f"/api/collection-plans/{pid}/retry").raise_for_status()
    assert batches(pid)[0].status == "queued"


def test_scalevln_waits_for_missing_scenes_then_refills(workstation, monkeypatch):
    from data_pipeline import scalevln_source
    client, _ = workstation
    pid = create(client, "scalevln")
    monkeypatch.setattr(scalevln_source, "availability", lambda *_: {"scene_paths": []})
    local.refill_local()
    assert not batches(pid)
    monkeypatch.setattr(scalevln_source, "availability", lambda *_: {"scene_paths": ["hm3d/train/example.glb"]})
    local.refill_local()
    assert len(batches(pid)) == 1
    assert batches(pid)[0].config["count"] == 2


def test_fixed_cloud_request_does_not_create_a_continuous_plan(workstation, monkeypatch):
    from apps.api import main
    client, dispatch = workstation
    monkeypatch.setattr(main, "validate_cloud", lambda cfg: {"can_start": True})
    response = client.post("/api/runs", json={"dataset": "rxr", "execution": "cloud_idle",
        "compute_pool": "baidu-pro6000", "count": 1}, headers={"Idempotency-Key": "fixed-cloud-one"})
    assert response.status_code == 202, response.text
    assert response.json()["config"]["count"] == 1
    assert client.get("/api/collection-plans").json() == []
    dispatch.assert_not_called()


def test_bad_local_pool_and_unvalidated_scale_cloud_rejected(workstation):
    client, _ = workstation
    for changes in ({"compute_pools": ["baidu-b200"]}, {"compute_pools": ["local", "baidu-b200"]},
                    {"dataset": "scalevln", "execution": "cloud_idle", "compute_pools": ["baidu-b200"]}):
        response = client.post("/api/collection-plans", json={"dataset": "rxr", "execution": "local",
            "compute_pools": ["local"], **changes}, headers={"Idempotency-Key": "invalid-local-plan"})
        assert response.status_code == 422


def test_sharing_only_affects_outputs_and_never_follows_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "resource_root", tmp_path / "resources")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    root = settings.runs_root / "run-id"
    root.mkdir(parents=True)
    outside = tmp_path / "private.json"
    atomic_json(outside, {"private": True})
    (root / "link.json").symlink_to(outside)
    atomic_json(root / "shared.json", {})
    private_export = root / "export.json"
    private_export.write_text("{}")
    private_export.chmod(0o600)
    share_run_outputs(root)
    assert outside.stat().st_mode & 0o777 == 0o600
    assert (root / "shared.json").stat().st_mode & 0o777 == 0o666
    assert private_export.stat().st_mode & 0o777 == 0o666


def test_gpu_probe_fails_closed(monkeypatch):
    monkeypatch.setattr(local.subprocess, "run", Mock(return_value=Mock(returncode=0, stdout="1234\n")))
    assert not local.gpu_idle()
    monkeypatch.setattr(local.subprocess, "run", Mock(side_effect=OSError("unavailable")))
    assert not local.gpu_idle()


def test_cloud_archive_verified_idempotent_and_no_control_secrets(tmp_path, monkeypatch):
    from workers.output_storage import cloud_archive
    monkeypatch.setattr(settings, "resource_root", tmp_path / "shared")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    source = tmp_path / "cloud/run"
    (source / "dataset").mkdir(parents=True)
    (source / "dataset/data.bin").write_bytes(b"verified output")
    (source / "cloud_control.json").write_text('{"lease": 1}')
    (source / "private.env").write_text('private configuration')
    target = cloud_archive("run-one", source)
    assert target == settings.runs_root / "run-one"
    assert (target / "dataset/data.bin").read_bytes() == b"verified output"
    assert not (target / "cloud_control.json").exists()
    assert not (target / "private.env").exists()
    assert json.loads((target / ".shared-cloud-output.json").read_text())["complete"]
    assert cloud_archive("run-one", source) == target
    assert (source / "dataset/data.bin").exists()


def test_cloud_archive_rejects_links_collisions_and_full_disk(tmp_path, monkeypatch):
    from workers.output_storage import cloud_archive
    monkeypatch.setattr(settings, "resource_root", tmp_path / "shared")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    source = tmp_path / "cloud"
    (source / "dataset").mkdir(parents=True)
    outside = tmp_path / "secret"
    outside.write_text("private")
    (source / "dataset/escape").symlink_to(outside)
    with pytest.raises(ValueError, match="符号链接"):
        cloud_archive("one", source)
    (source / "dataset/escape").unlink()
    (settings.runs_root / "existing").mkdir(parents=True)
    with pytest.raises(ValueError, match="禁止覆盖"):
        cloud_archive("existing", source)
    monkeypatch.setattr("workers.output_storage.shutil.disk_usage", lambda _: Mock(free=0))
    with pytest.raises(ValueError, match="空间不足"):
        cloud_archive("two", source)


def test_cloud_archive_poll_does_not_block_lease_thread(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from workers import output_storage as storage
    monkeypatch.setattr(settings, "resource_root", tmp_path / "shared")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    monkeypatch.setattr(storage, "_archive_jobs", {})
    future = Future()
    executor = Mock()
    executor.submit.return_value = future
    monkeypatch.setattr(storage, "_archive_executor", executor)
    assert storage.poll_cloud_archive("run-async", tmp_path / "cloud") is None
    assert storage.poll_cloud_archive("run-async", tmp_path / "cloud") is None
    assert executor.submit.call_count == 1
    target = settings.runs_root / "run-async"
    future.set_result(target)
    assert storage.poll_cloud_archive("run-async", tmp_path / "cloud") == target


def test_cloud_archive_failure_has_retry_backoff(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from workers import output_storage as storage
    monkeypatch.setattr(settings, "resource_root", tmp_path / "shared")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    monkeypatch.setattr(storage, "_archive_jobs", {})
    monkeypatch.setattr(storage.time, "monotonic", lambda: 10)
    future = Future()
    executor = Mock()
    executor.submit.return_value = future
    monkeypatch.setattr(storage, "_archive_executor", executor)
    storage.poll_cloud_archive("run-failed", tmp_path / "cloud")
    future.set_exception(ValueError("full"))
    for _ in range(2):
        with pytest.raises(ValueError, match="full"):
            storage.poll_cloud_archive("run-failed", tmp_path / "cloud")
    assert executor.submit.call_count == 1
    monkeypatch.setattr(storage.time, "monotonic", lambda: 71)
    assert storage.poll_cloud_archive("run-failed", tmp_path / "cloud") is None
    assert executor.submit.call_count == 2


def test_failed_cloud_copy_cleans_only_unpublished_scratch(tmp_path, monkeypatch):
    from workers import output_storage as storage
    monkeypatch.setattr(settings, "resource_root", tmp_path / "shared")
    monkeypatch.setattr(settings, "shared_run_permissions", True)
    source = tmp_path / "cloud"
    (source / "dataset").mkdir(parents=True)
    (source / "dataset/data").write_text("original")
    monkeypatch.setattr(storage.shutil, "copytree", Mock(side_effect=OSError("interrupted copy")))
    with pytest.raises(OSError, match="interrupted"):
        storage.cloud_archive("run-copy-failed", source)
    assert list(settings.runs_root.iterdir()) == []
    assert (source / "dataset/data").read_text() == "original"
