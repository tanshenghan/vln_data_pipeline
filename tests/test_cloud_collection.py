"""Cloud lifecycle tests run against isolated SQL/FS and simulated provider APIs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import httpx
import pytest

from tests.test_platform import platform
from apps.api import db, main
from apps.api.settings import settings
from workers import cloud_scheduler
from workers.cloud_config import Pool, cloud_status
from workers.cloud_entry import action, recover_interrupted, terminate_group
from workers.cloud_providers import CloudError, Job, Provider, normalize_job, signed_headers
from workers.command_runner import atomic_json


@pytest.fixture
def cloud(platform, tmp_path, monkeypatch):
    client, dispatch = platform
    storage = tmp_path / "cloud-volume"
    storage.mkdir()
    (storage / ".vln-volume-id").write_text("test-volume")
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"jobSpec": {"replicas": 1, "image": "test/image", "resources": []}}))
    pool = Pool(id="baidu-pro6000", name="PRO6000", provider="baidu", enabled=True, region="bj",
                endpoint="https://aihc.bj.baidubce.com", resource_pool_id="aihc-serverless", queue_id="queue-test",
                access_key_env="TEST_CLOUD_AK", secret_key_env="TEST_CLOUD_SK", template_file=template,
                local_storage_root=storage, remote_storage_root=Path("/mnt/cloud"), storage_id="test-volume",
                validated_datasets=["rxr"], idle_seconds=0, cooldown_seconds=0, stop_grace_seconds=0)
    monkeypatch.setenv("TEST_CLOUD_AK", "fake-ak")
    monkeypatch.setenv("TEST_CLOUD_SK", "fake-sk")
    config = tmp_path / "pools.json"
    config.write_text(json.dumps({"pools": [pool.model_dump(mode="json")]}))
    monkeypatch.setattr(settings, "cloud_pools_file", config)
    monkeypatch.setattr(cloud_scheduler, "load_pools", lambda: [pool])
    def validate(cfg):
        cfg["eligible_pools"] = [pool.id]
        return {"can_start": True}
    monkeypatch.setattr(main, "validate_cloud", validate)
    fake = FakeCloud()
    scheduler = cloud_scheduler.Scheduler(lambda p: FakeProvider(p, fake))
    return client, dispatch, pool, fake, scheduler


class FakeCloud:
    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.stops = []
        self.fail_monitor = False
        self.timeout_submit = False
        self.accept_timeout = True
        self.hide_jobs = False


class FakeProvider:
    def __init__(self, pool, cloud):
        self.pool, self.cloud = pool, cloud

    def payload(self, *args):
        return Provider.payload(self, *args)

    def submit(self, payload):
        cloud = self.cloud
        cloud.submissions.append(payload)
        job = Job(f"job-{len(cloud.submissions)}", payload["name"], "running", idle=True)
        if not cloud.timeout_submit or cloud.accept_timeout:
            cloud.jobs[job.id] = job
        if cloud.timeout_submit:
            raise CloudError("timeout after submission")
        return job.id

    def list_jobs(self):
        if self.cloud.fail_monitor:
            raise CloudError("monitor unavailable")
        return [] if self.cloud.hide_jobs else list(self.cloud.jobs.values())

    def get(self, job_id):
        return self.cloud.jobs[job_id]

    def stop(self, job_id):
        self.cloud.stops.append(job_id)
        self.cloud.jobs[job_id] = replace(self.cloud.jobs[job_id], state="stopped")

    def close(self):
        pass


def submit(client, key="cloud-request-one"):
    response = client.post("/api/runs", json={"dataset": "rxr", "execution": "cloud_idle"},
                           headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def read(run_id):
    with db.session() as session:
        return session.get(db.Run, run_id), session.get(db.CloudExecution, run_id)


def test_cloud_never_dispatches_local_gpu_and_is_idempotent(cloud):
    client, dispatch, _, fake, scheduler = cloud
    rid = submit(client)
    assert submit(client) == rid
    dispatch.assert_not_called()
    scheduler.tick()
    scheduler.tick()
    assert len(fake.submissions) == 1
    assert read(rid)[1].job_id == "job-1"
    from workers.tasks import execute_run, reconcile
    execute_run(rid)
    reconcile()
    dispatch.assert_not_called()
    assert read(rid)[0].status == "running"


def test_pending_foreground_stops_only_ours_then_resumes_same_directory(cloud):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    binding = read(rid)[1]
    checkpoint = Path(binding.directory) / "dataset.work/shards/00000/meta/build_shard.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{"complete":true}')
    original = checkpoint.stat().st_mtime_ns
    fake.jobs["other"] = Job("other", "someone-elses-training", "pending")
    scheduler.tick()
    assert read(rid)[1].state == "stopping"
    scheduler.tick()
    scheduler.tick()
    assert read(rid)[0].status == "waiting"
    assert fake.stops == ["job-1"]
    assert len(fake.submissions) == 1
    fake.jobs["other"] = replace(fake.jobs["other"], state="running")
    scheduler.tick()
    assert len(fake.submissions) == 2
    assert read(rid)[1].generation == 2
    assert read(rid)[1].directory == binding.directory
    assert checkpoint.stat().st_mtime_ns == original


def test_manual_pause_and_cancel_do_not_auto_resume(cloud):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    client.post(f"/api/runs/{rid}/pause").raise_for_status()
    assert client.post(f"/api/runs/{rid}/retry").status_code == 409
    for _ in range(5):
        scheduler.tick()
    assert read(rid)[0].status == "paused"
    assert len(fake.submissions) == 1
    client.post(f"/api/runs/{rid}/resume").raise_for_status()
    scheduler.tick()
    assert len(fake.submissions) == 2
    client.post(f"/api/runs/{rid}/cancel").raise_for_status()
    for _ in range(5):
        scheduler.tick()
    assert read(rid)[0].status == "cancelled"
    assert len(fake.submissions) == 2


def test_cancel_before_binding(cloud):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    client.post(f"/api/runs/{rid}/cancel").raise_for_status()
    scheduler.tick()
    assert not fake.submissions
    assert read(rid)[0].status == "cancelled"


def test_pending_archive_is_not_premature_success_and_keeps_polling(cloud, monkeypatch):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    binding = read(rid)[1]
    atomic_json(Path(binding.directory) / "cloud_result.json", {
        "generation": binding.generation, "status": "succeeded", "artifacts": []})
    fake.jobs[binding.job_id] = replace(fake.jobs[binding.job_id], state="succeeded")
    importer = Mock(return_value=False)
    monkeypatch.setattr(cloud_scheduler, "import_result", importer)
    scheduler.tick()
    assert read(rid)[0].stage == "cloud_archiving"
    assert read(rid)[0].status == "running"
    scheduler.tick()
    assert importer.call_count == 2
    importer.return_value = None
    scheduler.tick()
    assert read(rid)[0].status == "succeeded"


@pytest.mark.parametrize("accepted", [True, False])
def test_uncertain_submit_recovers_by_name_without_duplicate(cloud, accepted):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    fake.timeout_submit = True
    fake.accept_timeout = accepted
    scheduler.tick()
    restarted = cloud_scheduler.Scheduler(lambda p: FakeProvider(p, fake))
    restarted.tick()
    restarted.tick()
    assert len(fake.submissions) == 1
    binding = read(rid)[1]
    assert binding.state == ("submitted" if accepted else "uncertain")
    if not accepted:
        client.post(f"/api/runs/{rid}/pause")
        assert client.post(f"/api/runs/{rid}/retry").status_code == 409


def test_monitor_failure_is_not_idle_and_worker_lease_yields(cloud):
    client, _, pool, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    binding = read(rid)[1]
    fake.fail_monitor = True
    result = scheduler.tick()
    assert result[pool.id]["observed"] is False
    assert action(Path(binding.directory), binding.generation) == "pause"
    assert len(fake.submissions) == 1
    fake.fail_monitor = False
    for _ in range(3):
        scheduler.tick()
    assert fake.stops == ["job-1"]


def test_native_preemption_and_failures_have_different_retry_budgets(cloud):
    client, _, pool, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    fake.jobs["job-1"] = replace(fake.jobs["job-1"], state="failed", reason="Preempted by queue owner")
    scheduler.tick()
    assert read(rid)[1].failures == 0
    for _ in range(pool.max_failures):
        scheduler.tick()
        binding = read(rid)[1]
        fake.jobs[binding.job_id] = replace(fake.jobs[binding.job_id], state="failed", reason="invalid image")
        scheduler.tick()
    assert read(rid)[0].status == "failed"
    before = len(fake.submissions)
    scheduler.tick()
    assert len(fake.submissions) == before


def test_pool_limit_and_source_only_preflight(cloud, monkeypatch):
    client, _, pool, fake, scheduler = cloud
    rid = submit(client)
    other = submit(client, "cloud-request-two")
    scheduler.tick()
    assert len(fake.submissions) == pool.max_active_jobs
    assert read(other)[1] is None
    pool.enabled = False
    for _ in range(3):
        scheduler.tick()
    assert fake.stops == ["job-1"]
    assert read(rid)[0].status == "waiting"


def test_configuration_and_missing_mount_never_start(cloud):
    client, _, pool, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    pool.remote_repo = Path("/changed/code")
    for _ in range(3):
        scheduler.tick()
    assert read(rid)[0].status == "failed"
    (pool.local_storage_root / ".vln-volume-id").unlink()
    scheduler.tick()
    assert len(fake.submissions) == 1


def test_accepting_another_dataset_does_not_interrupt_existing_job(cloud):
    client, _, pool, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    pool.validated_datasets.append("r2r")
    scheduler.tick()
    assert read(rid)[0].status == "running"
    assert len(fake.submissions) == 1
    assert not fake.stops


def test_baidu_current_instance_reclamation_does_not_consume_failure_budget():
    raw = {"jobId": "idle-job", "name": "collector", "status": "Failed", "idleTask": True,
           "pods": [{"status": "Preempted", "reason": "Idle resources reclaimed"}]}
    assert normalize_job(raw, "baidu").preempted
    raw["historyPods"] = raw["pods"]
    raw["pods"] = [{"status": "Failed", "reason": "application exited with code 1"}]
    assert not normalize_job(raw, "baidu").preempted
    raw["pods"] = None  # AIHC history/list responses may explicitly return null.
    assert not normalize_job(raw, "baidu").preempted


def test_acceptance_requires_native_idle_success_and_complete_report(cloud, monkeypatch):
    from configs import verify_cloud_pool
    client, _, pool, fake, scheduler = cloud
    pool.validated_datasets = []
    atomic_json(settings.cloud_pools_file, {"pools": [pool.model_dump(mode="json")]})
    rid = submit(client)
    scheduler.tick()
    binding = read(rid)[1]
    digest = pool.digest()
    records = {"rxr": {"id": rid}}
    monkeypatch.setattr(verify_cloud_pool, "Provider", lambda p: FakeProvider(p, fake))
    with pytest.raises(ValueError, match="尚未成功"):
        verify_cloud_pool.accept(pool, records, expected_digest=digest)
    with db.session() as session:
        session.get(db.Run, rid).status = "succeeded"
        session.get(db.CloudExecution, rid).state = "done"
    fake.jobs[binding.job_id] = replace(fake.jobs[binding.job_id], state="succeeded", idle=False)
    with pytest.raises(ValueError, match="闲时模式"):
        verify_cloud_pool.accept(pool, records, expected_digest=digest)
    fake.jobs[binding.job_id] = replace(fake.jobs[binding.job_id], idle=True)
    root = Path(binding.directory)
    (root / "dataset/meta").mkdir(parents=True)
    atomic_json(root / "cloud_result.json", {"generation": binding.generation, "status": "succeeded"})
    report = {k: True for k in ("complete", "self_contained", "official_loader_all_frames",
                               "all_30_source_arrays_exact", "geometry_recomputed")}
    atomic_json(root / "dataset/meta/platform_validation.json", {**report, "geometry_recomputed": False})
    with pytest.raises(ValueError, match="完整独立校验"):
        verify_cloud_pool.accept(pool, records, expected_digest=digest)
    assert json.loads(settings.cloud_pools_file.read_text())["pools"][0]["validated_datasets"] == []
    atomic_json(root / "dataset/meta/platform_validation.json", report)
    verify_cloud_pool.accept(pool, records, expected_digest=digest)
    assert json.loads(settings.cloud_pools_file.read_text())["pools"][0]["validated_datasets"] == ["rxr"]
    template = json.loads(pool.template_file.read_text())
    template["jobSpec"]["image"] = "changed/image"
    atomic_json(pool.template_file, template)
    with pytest.raises(ValueError, match="执行配置已改变"):
        verify_cloud_pool.accept(pool, records, expected_digest=digest)


def test_stale_pool_status_disables_submission_and_never_exposes_keys(cloud):
    _, _, _, _, scheduler = cloud
    scheduler.tick()
    status = cloud_status()
    assert status["ready"]
    assert "fake-ak" not in json.dumps(status) and "fake-sk" not in json.dumps(status)
    atomic_json(settings.runtime_root / "cloud_status.json", {"updated": time.time() - 1000})
    assert not cloud_status()["ready"]


@pytest.mark.parametrize(("target", "unready", "expected"), [
    ("volc-h20", [], ["volc-h20-q1", "volc-h20-q2"]),
    ("volc-h20", ["volc-h20-q1"], ["volc-h20-q2"]),
    ("volc-h20", ["volc-h20-q1", "volc-h20-q2"], []),
    ("volc-h20-q1", [], ["volc-h20-q1"]),
    ("auto", [], ["baidu-pro6000", "volc-h20-q1", "volc-h20-q2"]),
])
def test_h20_selection_uses_only_ready_h20_queues(cloud, monkeypatch, target, unready, expected):
    from apps.api import capacity
    from data_pipeline import vlnce_source
    from workers import cloud_config
    _, _, base, _, _ = cloud
    pools = [base] + [base.model_copy(update={"id": f"volc-h20-q{i}", "provider": "volcengine"})
                      for i in (1, 2)]
    monkeypatch.setattr(cloud_config, "load_pools", lambda: pools)
    monkeypatch.setattr(cloud_config, "cloud_status", lambda: {"pools": [
        {"id": p.id, "ready": p.id not in unready, "datasets": ["rxr"]} for p in pools
    ]})
    monkeypatch.setattr(capacity, "estimate_gt", lambda *a, **kw: {"can_start": True})
    monkeypatch.setattr(vlnce_source, "source_paths", lambda *a: [])
    cfg = {"compute_pool": target, "dataset": "rxr", "split": "train", "role": "guide"}
    if not expected:
        with pytest.raises(ValueError, match="没有就绪"):
            cloud_config.validate_cloud(cfg)
    else:
        cloud_config.validate_cloud(cfg)
        assert cfg["eligible_pools"] == expected


def test_result_must_match_generation_and_pass_import(cloud, monkeypatch):
    client, _, _, fake, scheduler = cloud
    rid = submit(client)
    scheduler.tick()
    binding = read(rid)[1]
    atomic_json(Path(binding.directory) / "cloud_result.json", {"generation": binding.generation, "status": "succeeded", "artifacts": []})
    fake.jobs[binding.job_id] = replace(fake.jobs[binding.job_id], state="succeeded")
    importer = Mock()
    monkeypatch.setattr(cloud_scheduler, "import_result", importer)
    scheduler.tick()
    assert read(rid)[0].status == "succeeded"
    importer.assert_called_once()


def test_checkpoint_recovery_and_expired_lease(tmp_path):
    done = tmp_path / "stages/done"
    interrupted = tmp_path / "stages/capture/attempt-0001"
    done.mkdir(parents=True)
    interrupted.mkdir(parents=True)
    (done / "success.json").write_text('{"digest":"same"}')
    stamp = (done / "success.json").stat().st_mtime_ns
    recover_interrupted(tmp_path)
    assert json.loads((interrupted / "exit.json").read_text())["cloud_interrupted"]
    assert (done / "success.json").stat().st_mtime_ns == stamp
    atomic_json(tmp_path / "cloud_control.json", {"action": "run", "generation": 2, "deadline": 100})
    assert action(tmp_path, 2, timestamp=90) == "run"
    assert action(tmp_path, 1, timestamp=90) == "pause"
    assert action(tmp_path, 2, timestamp=101) == "pause"


def test_evicted_initial_plan_can_restart_without_discarding_shards(tmp_path):
    plan = tmp_path / "dataset.work/plan.json"
    plan.parent.mkdir()
    plan.write_text('{"shards": [')
    recover_interrupted(tmp_path)
    assert not plan.exists()
    assert len(list(plan.parent.glob("plan.interrupted-*.json"))) == 1
    plan.write_text('{"shards": [')
    completed = plan.parent / "shards/00000/meta/build_shard.json"
    completed.parent.mkdir(parents=True)
    completed.write_text('{"complete":true}')
    with pytest.raises(ValueError, match="已有产物"):
        recover_interrupted(tmp_path)
    assert completed.read_text() == '{"complete":true}'


def test_invalid_pool_config_returns_preflight_error(platform, tmp_path, monkeypatch):
    from workers.cloud_config import validate_cloud
    client, _ = platform
    config = tmp_path / "invalid-pools.json"
    config.write_text('{}')
    monkeypatch.setattr(settings, "cloud_pools_file", config)
    monkeypatch.setattr(main, "validate_cloud", validate_cloud)
    response = client.post("/api/preflight", json={"dataset": "rxr", "execution": "cloud_idle"})
    assert response.status_code == 422
    assert not client.get("/api/compute-pools").json()["ready"]


def test_cloud_container_rejects_uid_mismatch_before_start(tmp_path, monkeypatch):
    from workers import cloud_entry
    rid = "00000000-0000-0000-0000-000000000001"
    directory = tmp_path / "runs" / rid
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{}')
    (tmp_path / ".vln-volume-id").write_text("test-volume")
    monkeypatch.setattr(sys, "argv", ["cloud_entry", "--run-id", rid, "--generation", "1",
        "--storage-root", str(tmp_path), "--storage-id", "test-volume"])
    monkeypatch.setattr(os, "geteuid", lambda: (directory / "config.json").stat().st_uid + 1)
    with pytest.raises(ValueError, match="UID"):
        cloud_entry.main()


@pytest.mark.parametrize("cloud_worker,expected", [("true", "GPU-assigned-only"), ("false", "unset")])
def test_habitat_entry_preserves_cloud_device_assignment(tmp_path, cloud_worker, expected):
    interpreter = tmp_path / "fake-habitat"
    interpreter.write_text('#!/usr/bin/env bash\nprintf "%s" "${CUDA_VISIBLE_DEVICES:-unset}"\n')
    interpreter.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.check_output(["bash", str(root / "data_pipeline/run.sh"), str(tmp_path / "output"), "1"],
        env={**os.environ, "HABITAT_PYTHON": str(interpreter), "VLN_CLOUD_WORKER": cloud_worker,
             "CUDA_VISIBLE_DEVICES": "GPU-assigned-only"}, text=True)
    assert result == expected


def test_stop_releases_entire_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)"
    parent = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(90)"
    process = subprocess.Popen([sys.executable, "-c", parent, str(pid_file), child], start_new_session=True)
    try:
        for _ in range(100):
            if pid_file.exists():
                break
            time.sleep(.02)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        terminate_group(process)
        for _ in range(100):
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("GPU child process survived termination")
    finally:
        terminate_group(process)


def test_baidu_wire_contract_pagination_and_idle_priority(cloud):
    _, _, pool, _, _ = cloud
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.params["action"] == "DescribeDevInstances":
            return httpx.Response(200, json={"devInstances": [], "totalCount": 0})
        if request.url.params["action"] == "DescribeQueue":
            return httpx.Response(200, json={"queueId": "queue-test", "queueType": "Elastic"})
        body = json.loads(request.content)
        if request.url.params["action"] == "DescribeJobs":
            page = body["pageNumber"]
            jobs = [{"jobId": str(i), "name": str(i), "status": "Created"} for i in range(100)] if page == 1 else [{"jobId": "100", "name": "foreground", "status": "Created"}]
            return httpx.Response(200, json={"jobs": jobs, "totalCount": 101})
        return httpx.Response(200, json={"jobId": "created-job"})
    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(respond)))
    assert len(provider.list_jobs()) == 101
    payload = provider.payload("00000000-0000-0000-0000-000000000000", 1, "vln-test")
    assert payload["idleTask"] is True and "priority" not in payload
    assert provider.submit(payload) == "created-job"
    assert requests[-1].url.params["queueID"] == "queue-test"
    assert requests[-1].headers["X-API-Version"] == "v2"
    assert requests[-1].headers["Authorization"].startswith("bce-auth-v1/fake-ak/")
    # Golden vector cross-checked against bce-python-sdk 0.9.79, using fake keys.
    headers = signed_headers(pool, "DescribeJobs", {"action": "DescribeJobs", "resourcePoolId": "aihc-serverless"}, b'{}',
                             timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert headers["Authorization"] == ("bce-auth-v1/fake-ak/2026-09-27T00:00:00Z/1800/content-type;host/"
                                         "def18a56010970ea5cfc545b6847a619bef6c677c58b1b335c0b29f4d22a1f5a")
    provider.close()


def test_baidu_observes_queue_and_development_workloads_without_blocking_own_idle_job(cloud):
    _, _, pool, _, _ = cloud
    detail = {"queueId": pool.queue_id, "queueType": "Elastic", "inqueueJobs": 1}
    devs = []

    def respond(request):
        action = request.url.params["action"]
        if action == "DescribeJobs":
            return httpx.Response(200, json={"totalCount": 2, "jobs": [
                {"jobId": "ours", "name": "vln-ours", "status": "Pending", "idleTask": True},
                {"jobId": "old", "name": "old", "status": "ManualTermination"}]})
        assert request.method == "GET" and request.headers["version"] == "v2"
        assert "resourcePoolId" not in request.url.params and "queueID" not in request.url.params
        if action == "DescribeDevInstances":
            assert request.url.params["queueName"] == pool.queue_id
            assert request.url.params["onlyMyDevs"] == "false"
            return httpx.Response(200, json={"totalCount": len(devs), "devInstances": devs})
        assert action == "DescribeQueue" and request.url.params["queueId"] == pool.queue_id
        return httpx.Response(200, json=detail)

    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(respond)))
    jobs = provider.list_jobs()
    assert next(job for job in jobs if job.id == "old").terminal
    assert not any(job.state == "pending" and not job.idle for job in jobs)

    devs.append({"id": "dev-other-user", "queueName": pool.queue_id, "status": 2})
    detail["inqueueJobs"] = 3  # Collector, development instance, and another workload.
    jobs = provider.list_jobs()
    assert sum(job.state == "pending" and not job.idle for job in jobs) == 2
    assert next(job for job in jobs if job.id == "dev:dev-other-user").state == "pending"

    devs[0]["status"] = 99
    with pytest.raises(CloudError, match="未知百度开发机状态"):
        provider.list_jobs()
    devs.clear()
    # Live AIHC returns the physical parent with the selected child nested.
    child = dict(detail)
    detail.clear()
    detail.update(queueId="parent-physical", queueType="Physical", children=[child, {
        "queueId": "other-queue", "queueType": "Elastic", "inqueueJobs": 100}])
    assert sum(job.state == "pending" and not job.idle for job in provider.list_jobs()) == 2
    detail.clear()
    detail.update(queueId=pool.queue_id)
    detail["queueType"] = "Physical"
    with pytest.raises(CloudError, match="弹性任务队列"):
        provider.list_jobs()
    provider.close()


def test_volc_wire_contract_forces_preemptible_and_disables_retention(cloud):
    _, _, pool, _, _ = cloud
    pool.provider = "volcengine"
    pool.region = "cn-beijing"
    pool.endpoint = "https://open.volcengineapi.com"
    pool.template_file.write_text(json.dumps({"ResourceConfig": {"Priority": 9, "Preemptible": False,
        "ResourceReservation": True, "HoldingTimeSeconds": 3600, "Roles": [{"Replicas": 1}]},
        "RuntimeConfig": {"Image": {"Url": "test/image"}}}))
    provider = Provider(pool)
    payload = provider.payload("00000000-0000-0000-0000-000000000000", 1, "vln-test")
    config = payload["ResourceConfig"]
    assert config["Preemptible"] and config["Priority"] == 1
    assert config["HoldingTimeSeconds"] == 0 and not config["ResourceReservation"]
    # A provider may need root for its bootstrap; private files must only be
    # opened inside the shell running as the PFS owner.
    original_digest = pool.digest()
    pool.run_as_user = "1022:1024"
    switched = provider.payload("00000000-0000-0000-0000-000000000000", 1, "vln-test")
    import shlex
    command = shlex.split(switched["RuntimeConfig"]["Command"])
    assert command[:7] == ["exec", "setpriv", "--reuid=1022", "--regid=1024", "--clear-groups", "/bin/sh", "-c"]
    assert command[7] == payload["RuntimeConfig"]["Command"]
    assert pool.digest() != original_digest
    pool.run_as_user = ""
    assert pool.digest() == original_digest
    headers = signed_headers(pool, "ListJobs", {"Action": "ListJobs", "Version": "2024-07-01"}, b'{}',
                             timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert "20260927/cn-beijing/ml_platform/request" in headers["Authorization"]
    assert headers["X-Date"] == "20260927T000000Z"
    # Golden vector cross-checked against volcengine 1.0.228, using fake keys.
    assert headers["Authorization"].endswith("Signature=c2b7e95a0f5b527ac81e311f4687d108c591b92b10ddb4c92be0cecc4a2036dc")
    provider.close()


def test_volc_monitors_other_users_development_and_service_scale_out(cloud):
    _, _, pool, _, _ = cloud
    pool.provider = "volcengine"
    pool.region, pool.endpoint = "cn-beijing", "https://open.volcengineapi.com"
    dev = {"Id": "someone-elses-dev", "ResourceQueueId": pool.queue_id, "Status": {"State": "Pending"}}
    role = {"ExpectedReplicias": 2, "AvailableReplicas": 1,
            "ExpectedUpdatedReplicas": 2, "UpdatedAvailableReplicas": 1}
    deployment = {"Id": "service-scale-out", "Status": {"State": "Inservice"}, "RoleStatus": [{"Status": role}]}
    requests = []

    def respond(request):
        action = request.url.params['Action']
        payload = json.loads(request.content)
        requests.append((action, payload))
        assert payload['PageSize'] == 100  # Live v2 rejects PageSize=1.
        if action == 'ListJobs':
            # Live ML jobs spend image-pull/startup time in Deploying.
            result = {'Items': [{'Id':'ours','Name':'vln-ours','Status':{'State':'Deploying'},
                                 'ResourceConfig':{'Preemptible':True}}], 'TotalCount':1}
        elif action == 'ListDevInstances':
            assert payload['ResourceQueueIds'] == [pool.queue_id]
            assert 'Relationship' not in payload and 'CreateUserIds' not in payload
            result = {'List':[dev], 'TotalCount':1}
        else:
            assert action == 'ListDeployments' and payload['ResourceQueueId'] == pool.queue_id
            result = {'Items':[deployment], 'TotalCount':1}
        return httpx.Response(200, json={'Result':result})

    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(respond)))
    jobs = provider.list_jobs()
    assert {j.id for j in jobs if j.state == 'pending' and not j.idle} == {
        'dev:someone-elses-dev', 'deployment:service-scale-out'}
    dev['Status']['State'] = 'Running'
    role.update(AvailableReplicas=2, UpdatedAvailableReplicas=2)
    assert not any(j.state == 'pending' and not j.idle for j in provider.list_jobs())
    deployment['RoleStatus'] = []
    with pytest.raises(CloudError, match='缺少实例数量'):
        provider.list_jobs()
    deployment['RoleStatus'] = [{'Status':role}]
    dev['ResourceQueueId'] = 'unexpected-queue'
    with pytest.raises(CloudError, match='队列筛选'):
        provider.list_jobs()
    provider.close()


def test_volc_development_pagination_must_be_complete(cloud):
    _, _, pool, _, _ = cloud
    pool.provider = 'volcengine'
    seen_pages = []
    broken = False
    def respond(request):
        action = request.url.params['Action']
        payload = json.loads(request.content)
        if action == 'ListJobs' or action == 'ListDeployments':
            result = {'Items':[], 'TotalCount':0}
        else:
            page = payload['PageNumber']
            seen_pages.append(page)
            ids = range(100) if page == 1 else ([] if broken else [100])
            result = {'List':[{'Id':str(i), 'ResourceQueueId':pool.queue_id,
                               'Status':{'State':'Pending' if i==100 else 'Running'}} for i in ids],
                      'TotalCount':101}
        return httpx.Response(200, json={'Result':result})
    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(respond)))
    assert sum(j.state == 'pending' for j in provider.list_jobs()) == 1
    assert seen_pages == [1, 2]
    broken = True
    with pytest.raises(CloudError, match='分页结果不完整'):
        provider.list_jobs()
    provider.close()


def test_baidu_refreshes_private_image_password_before_each_attempt(cloud):
    _, _, pool, _, _ = cloud
    pool.baidu_registry_id = "ccr-test123"
    pool.baidu_registry_username = "test-user"
    image = "ccr-test123-vpc.cnc.bj.baidubce.com/team/collector@sha256:abc"
    pool.template_file.write_text(json.dumps({"jobSpec": {"replicas": 1, "image": image}}))
    digest = pool.digest()
    calls = []

    def respond(request):
        calls.append(request)
        assert request.url.host == "ccr.bj.baidubce.com"
        assert request.url.path == "/v1/instances/ccr-test123/credential"
        assert json.loads(request.content) == {"duration": 1}
        assert "/host;x-bce-date/" in request.headers["authorization"]
        return httpx.Response(200, json={"password": f"test-password-{len(calls)}"})

    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(respond)))
    preview = provider.payload("run", 1, "check", authenticate_image=False)
    assert not calls and "imageConfig" not in preview["jobSpec"]
    first = provider.payload("run", 1, "first")
    second = provider.payload("run", 2, "second")
    assert first["jobSpec"]["imageConfig"]["password"] == "test-password-1"
    assert second["jobSpec"]["imageConfig"]["password"] == "test-password-2"
    assert pool.digest() == digest
    assert "password" not in pool.template_file.read_text()
    assert "password" not in pool.model_dump_json()
    provider.close()

    provider = Provider(pool, httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(403, json={"message": "sensitive-provider-body"}))))
    with pytest.raises(CloudError, match="HTTP 403") as error:
        provider.payload("run", 3, "third")
    assert "sensitive" not in str(error.value)
    provider.close()


def test_unknown_provider_state_is_not_idle():
    with pytest.raises(CloudError):
        normalize_job({"jobId": "job", "name": "unknown", "status": "brand-new-state"}, "baidu")
