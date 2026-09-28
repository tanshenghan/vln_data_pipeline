"""Continuous plans must feed fresh batches without bypassing idle scheduling."""
import json
import sqlite3
from dataclasses import replace

import pytest
from sqlalchemy import select

from apps.api import db
from apps.api.settings import settings
from workers import collection_plans as plans, cloud_scheduler
from workers import local_collection as local
from workers.cloud_providers import Job
from tests.test_cloud_collection import cloud, FakeProvider, read
from tests.test_platform import platform


@pytest.fixture
def continuous(cloud, monkeypatch, tmp_path):
    client, dispatch, pro, fake, scheduler = cloud
    storage = tmp_path / "b200-volume"
    storage.mkdir()
    (storage / ".vln-volume-id").write_text("test-volume")
    b200 = pro.model_copy(update={"id": "baidu-b200", "name": "B200", "local_storage_root": storage})
    pools = [pro, b200]
    monkeypatch.setattr(plans, "load_pools", lambda: pools)
    monkeypatch.setattr(cloud_scheduler, "load_pools", lambda: pools)
    def validate(cfg):
        cfg["eligible_pools"] = [cfg["compute_pool"]]
        return {"can_start": True}
    monkeypatch.setattr(plans, "validate_cloud", validate)
    with sqlite3.connect(settings.catalog_db) as catalog:
        catalog.row_factory = sqlite3.Row
        template = dict(catalog.execute("SELECT * FROM episodes LIMIT 1").fetchone())
        for i in range(6):
            row = dict(template)
            row.update(id=f"source-{i}", episode_id=str(10300 + i))
            detail = json.loads(row["detail"])
            detail.update(id=row["id"], episode_id=row["episode_id"])
            row["detail"] = json.dumps(detail)
            catalog.execute(f"INSERT INTO episodes ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
    return client, dispatch, pools, fake, scheduler


def create(client, key="continuous-plan-one", **changes):
    payload = {"dataset": "rxr", "compute_pools": ["baidu-pro6000", "baidu-b200"],
               "routes_per_shard": 2, **changes}
    response = client.post("/api/collection-plans", json=payload, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def batches(plan_id):
    with db.session() as session:
        return list(session.scalars(select(db.Run).join(db.CollectionBatch, db.CollectionBatch.run_id == db.Run.id)
                                   .where(db.CollectionBatch.plan_id == plan_id).order_by(db.Run.created_at)))


def complete(run, fake=None):
    db.update_run(run.id, status="succeeded", stage="complete")
    with db.session() as session:
        binding = session.get(db.CloudExecution, run.id)
        if binding:
            binding.state = "done"
        for record in run.config["source_records"]:
            session.add(db.DataSample(id="sample-" + record["id"], run_id=run.id, origin="gt", quality="validated",
                source_record_id=record["id"], details={}, **{key:str(record[key]) for key in
                ("dataset", "split", "role", "language", "scene", "episode_id", "instruction")}))


def test_plan_is_idempotent_and_pending_pool_can_be_selected(continuous):
    client, dispatch, pools, fake, scheduler = continuous
    pools[1].validated_datasets = []
    pid = create(client)
    assert create(client) == pid
    assert len(client.get('/api/collection-plans').json()) == 1
    response = client.post('/api/collection-plans', json={"dataset":"rxr", "compute_pools":[pools[0].id]},
                           headers={"Idempotency-Key":"continuous-plan-one"})
    assert response.status_code == 409
    preview = client.post('/api/collection-plans/preflight', json={"dataset":"rxr", "compute_pools":[pools[1].id]}).json()
    assert preview['episodes'] == 7  # count=1 must not silently limit a continuous plan.
    assert not batches(pid)
    scheduler.tick()
    assert len(batches(pid)) == 1
    assert batches(pid)[0].config['compute_pool'] == pools[0].id
    assert not fake.submissions  # Feeder queues; only next scheduler cycle may submit.
    dispatch.assert_not_called()


def test_disjoint_batches_restart_and_refill_after_success(continuous):
    client, dispatch, pools, fake, scheduler = continuous
    pid = create(client)
    scheduler.tick()
    first = batches(pid)
    assert len(first) == 2
    assert {r.config['compute_pool'] for r in first} == {p.id for p in pools}
    assert all(len(r.config['source_ids']) == 2 for r in first)
    assert len({sid for r in first for sid in r.config['source_ids']}) == 4
    restarted = cloud_scheduler.Scheduler(lambda p: FakeProvider(p, fake))
    for _ in range(3):
        restarted.tick()
    assert len(fake.submissions) == 2
    assert len(batches(pid)) == 2
    complete(first[0])
    restarted.tick()
    all_batches = batches(pid)
    assert len(all_batches) == 3
    ids = [sid for r in all_batches for sid in r.config['source_ids']]
    assert len(ids) == len(set(ids)) == 6
    dispatch.assert_not_called()


def test_foreground_pressure_does_not_feed_and_yields_existing_batches(continuous):
    client, _, _, fake, scheduler = continuous
    pid = create(client)
    fake.jobs['foreground'] = Job('foreground', 'another-user', 'pending')
    scheduler.tick()
    assert not batches(pid)
    fake.jobs['foreground'] = replace(fake.jobs['foreground'], state='running')
    scheduler.tick()
    scheduler.tick()
    assert len(fake.submissions) == 2
    fake.jobs['foreground'] = replace(fake.jobs['foreground'], state='pending')
    for _ in range(3):
        scheduler.tick()
    assert len(fake.stops) == 2
    assert len(fake.submissions) == 2
    assert all(r.status == 'waiting' for r in batches(pid))
    fake.jobs['foreground'] = replace(fake.jobs['foreground'], state='running')
    scheduler.tick()
    assert len(fake.submissions) == 4
    assert len(batches(pid)) == 2


def test_pause_restart_resume_and_cancel_preserve_user_intent(continuous):
    client, _, _, fake, scheduler = continuous
    pid = create(client)
    scheduler.tick()
    scheduler.tick()
    assert client.post(f'/api/collection-plans/{pid}/pause').status_code == 200
    assert client.post(f'/api/collection-plans/{pid}/resume').status_code == 409
    scheduler = cloud_scheduler.Scheduler(lambda p: FakeProvider(p, fake))
    for _ in range(5):
        scheduler.tick()
    assert all(r.status == 'paused' for r in batches(pid))
    assert len(fake.submissions) == 2
    assert client.post(f'/api/runs/{batches(pid)[0].id}/retry').status_code == 409
    client.post(f'/api/collection-plans/{pid}/resume').raise_for_status()
    scheduler.tick()
    assert len(fake.submissions) == 4
    client.post(f'/api/collection-plans/{pid}/cancel').raise_for_status()
    for _ in range(5):
        scheduler.tick()
    assert all(r.status == 'cancelled' for r in batches(pid))
    with db.session() as session:
        assert not list(session.scalars(select(db.CollectionClaim)))
    assert len(fake.submissions) == 4


def test_environment_promotes_later_and_overlapping_plans_do_not_duplicate(continuous):
    client, _, pools, fake, scheduler = continuous
    pools[1].validated_datasets = []
    pid = create(client)
    second = create(client, key='overlapping-plan')
    scheduler.tick()
    assert len(batches(pid)) == 1
    assert not batches(second)
    pools[1].validated_datasets = ['rxr']
    scheduler.tick()
    assert len(batches(pid)) == 2
    assert not batches(second)
    ids = [sid for r in batches(pid) for sid in r.config['source_ids']]
    assert len(set(ids)) == 4


def test_monitor_and_capacity_failure_never_create_a_batch(continuous, monkeypatch):
    client, _, _, fake, scheduler = continuous
    pid = create(client)
    fake.fail_monitor = True
    scheduler.tick()
    assert not batches(pid)
    fake.fail_monitor = False
    def full(_):
        raise ValueError('云端持久化盘容量不足')
    monkeypatch.setattr(plans, 'validate_cloud', full)
    scheduler.tick()
    assert not batches(pid)
    assert not fake.submissions
    assert '容量不足' in client.get('/api/collection-plans').json()[0]['error']


def test_exhaustion_watches_new_sources_and_failure_requires_retry(continuous):
    client, _, pools, fake, scheduler = continuous
    pid = create(client, episode_ids=['10288'])
    scheduler.tick()
    only = batches(pid)[0]
    db.update_run(only.id, status='failed')
    scheduler.tick()
    assert client.get('/api/collection-plans').json()[0]['status'] == 'needs_attention'
    assert not fake.submissions
    client.post(f'/api/collection-plans/{pid}/retry').raise_for_status()
    assert batches(pid)[0].status == 'queued'
    complete(batches(pid)[0])
    for _ in range(3):
        scheduler.tick()
    assert len(batches(pid)) == 1
    plan = client.get('/api/collection-plans').json()[0]
    assert plan['status'] == 'watching'
    assert plan['progress']['completed'] == 1
    # A new plan over the same source also skips the verified sample.
    duplicate = create(client, key='already-collected', episode_ids=['10288'])
    scheduler.tick()
    assert not batches(duplicate)


def test_scope_and_cross_site_writes_are_validated(continuous):
    client, _, _, _, _ = continuous
    payload = {'dataset':'rxr', 'compute_pools':['baidu-pro6000']}
    for changes in ({'compute_pools':[]}, {'compute_pools':['unknown']}, {'episode_ids':['missing']},
                    {'dataset':'scalevln'}, {'split':'test'}, {'compute_pools':['baidu-pro6000']*2}):
        assert client.post('/api/collection-plans', json={**payload, **changes},
                           headers={'Idempotency-Key':'bad-plan-request'}).status_code == 422
    assert client.post('/api/collection-plans', json=payload,
        headers={'Idempotency-Key':'foreign-origin-request', 'Origin':'https://unrelated.example'}).status_code == 403


def test_new_catalog_sources_are_picked_up_after_exhaustion(continuous):
    client, _, _, _, scheduler = continuous
    pid = create(client, routes_per_shard=64)
    scheduler.tick()
    complete(batches(pid)[0])
    scheduler.tick()
    assert client.get('/api/collection-plans').json()[0]['status'] == 'watching'
    with sqlite3.connect(settings.catalog_db) as catalog:
        catalog.row_factory = sqlite3.Row
        row = dict(catalog.execute('SELECT * FROM episodes LIMIT 1').fetchone())
        row.update(id='source-added-later', episode_id='20000')
        detail = json.loads(row['detail'])
        detail.update(id=row['id'], episode_id=row['episode_id'])
        row['detail'] = json.dumps(detail)
        catalog.execute(f"INSERT INTO episodes ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", list(row.values()))
    scheduler.tick()
    assert len(batches(pid)) == 2
    assert batches(pid)[-1].config['source_ids'] == ['source-added-later']


def test_plan_resume_preserves_individually_paused_batch(continuous):
    client, _, _, _, scheduler = continuous
    pid = create(client)
    scheduler.tick()
    first = batches(pid)[0]
    client.post(f'/api/runs/{first.id}/pause').raise_for_status()
    client.post(f'/api/collection-plans/{pid}/pause').raise_for_status()
    client.post(f'/api/collection-plans/{pid}/resume').raise_for_status()
    assert read(first.id)[0].control == 'pause'
    assert read(first.id)[0].status == 'paused'


def test_one_pool_capacity_problem_does_not_block_another(continuous, monkeypatch):
    client, _, _, _, scheduler = continuous
    pid = create(client)
    def validate(cfg):
        if cfg['compute_pool'] == 'baidu-pro6000':
            raise ValueError('容量不足')
        cfg['eligible_pools'] = [cfg['compute_pool']]
        return {'can_start':True}
    monkeypatch.setattr(plans, 'validate_cloud', validate)
    scheduler.tick()
    assert len(batches(pid)) == 1
    assert batches(pid)[0].config['compute_pool'] == 'baidu-b200'


@pytest.fixture
def mixed(continuous, monkeypatch):
    monkeypatch.setattr(local, "gpu_idle", lambda: True)
    monkeypatch.setattr(local, "validate_gt", lambda cfg: {"can_start": True})
    return continuous


@pytest.mark.parametrize("cloud_ids", [
    ["baidu-pro6000"], ["baidu-b200"], ["baidu-pro6000", "baidu-b200"]])
@pytest.mark.parametrize("local_first", [True, False])
def test_mixed_devices_claim_disjoint_batches_in_either_order(mixed, cloud_ids, local_first):
    client, dispatch, _, fake, scheduler = mixed
    devices = ["local", *cloud_ids]
    pid = create(client, execution="mixed", compute_pools=devices)
    preview = client.post("/api/collection-plans/preflight", json={
        "dataset": "rxr", "execution": "mixed", "compute_pools": devices}).json()
    assert preview["compute_pools"] == devices
    assert preview["episodes"] == 7
    feeders = [local.refill_local, scheduler.tick]
    for feed in feeders if local_first else reversed(feeders):
        feed()
    first = batches(pid)
    assert len(first) == len(devices)
    assert {r.config["execution"] for r in first} == {"local", "cloud_idle"}
    assert {"local" if r.config["execution"] == "local" else r.config["compute_pool"]
            for r in first} == set(devices)
    ids = [sid for r in first for sid in r.config["source_ids"]]
    assert len(ids) == len(set(ids)) == 2 * len(devices)
    assert all(r.config["request_payload"]["execution"] == r.config["execution"] for r in first)
    # Restarted controllers do not duplicate queued work or dispatch local work to cloud.
    restarted = cloud_scheduler.Scheduler(lambda p: FakeProvider(p, fake))
    for _ in range(2):
        local.refill_local()
        restarted.tick()
    assert len(batches(pid)) == len(devices)
    assert len(fake.submissions) == len(cloud_ids)
    dispatch.assert_not_called()


def test_mixed_unverified_cloud_does_not_block_gpu0(mixed):
    client, _, pools, fake, scheduler = mixed
    for pool in pools:
        pool.validated_datasets = []
    pid = create(client, execution="mixed", compute_pools=["local", *[p.id for p in pools]])
    scheduler.tick()
    local.refill_local()
    assert len(batches(pid)) == 1
    assert batches(pid)[0].config["execution"] == "local"
    assert not fake.submissions


def test_mixed_busy_gpu0_does_not_block_cloud_or_lose_reason(mixed, monkeypatch):
    client, _, pools, _, scheduler = mixed
    pid = create(client, execution="mixed", compute_pools=["local", *[p.id for p in pools]])
    monkeypatch.setattr(local, "gpu_idle", lambda: False)
    local.refill_local()
    scheduler.tick()
    assert len(batches(pid)) == 2
    assert all(r.config["execution"] == "cloud_idle" for r in batches(pid))
    plan = client.get("/api/collection-plans").json()[0]
    assert "GPU0" in plan["error"]
    assert "local" in plan["progress"]["device_errors"]
    monkeypatch.setattr(local, "gpu_idle", lambda: True)
    local.refill_local()
    assert len(batches(pid)) == 3
    assert client.get("/api/collection-plans").json()[0]["error"] is None


def test_mixed_cloud_error_survives_local_progress(mixed, monkeypatch):
    client, _, _, _, scheduler = mixed
    pid = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000"])
    def full(_):
        raise ValueError("云端磁盘容量不足")
    monkeypatch.setattr(plans, "validate_cloud", full)
    scheduler.tick()
    local.refill_local()
    assert len(batches(pid)) == 1
    plan = client.get("/api/collection-plans").json()[0]
    assert "云端磁盘容量不足" in plan["error"]
    assert "baidu-pro6000" in plan["progress"]["device_errors"]


def test_mixed_pause_resume_and_cancel_apply_to_both_feeders(mixed):
    client, _, _, _, scheduler = mixed
    pid = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000", "baidu-b200"])
    local.refill_local()
    scheduler.tick()
    assert len(batches(pid)) == 3
    client.post(f"/api/collection-plans/{pid}/pause").raise_for_status()
    local.refill_local()
    scheduler.tick()
    assert len(batches(pid)) == 3
    assert all(r.status == "paused" and r.control == "pause" for r in batches(pid))
    client.post(f"/api/collection-plans/{pid}/resume").raise_for_status()
    assert all(r.status == "queued" and r.control == "run" for r in batches(pid))
    client.post(f"/api/collection-plans/{pid}/cancel").raise_for_status()
    scheduler.tick()
    local.refill_local()
    assert all(r.status == "cancelled" for r in batches(pid))
    with db.session() as session:
        assert not list(session.scalars(select(db.CollectionClaim)))


def test_mixed_failure_is_visible_while_other_device_runs(mixed):
    client, _, _, _, scheduler = mixed
    pid = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000"])
    local.refill_local()
    scheduler.tick()
    first, second = batches(pid)
    db.update_run(first.id, status="failed")
    db.update_run(second.id, status="running")
    local.refill_local()
    plan = client.get("/api/collection-plans").json()[0]
    assert plan["status"] == "needs_attention"
    assert plan["progress"]["batches_failed"] == 1
    assert len(batches(pid)) == 2


def test_mixed_feeders_share_nonblocking_allocation_lock(mixed):
    client, _, _, _, scheduler = mixed
    pid = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000"])
    with plans.allocation_lock() as acquired:
        assert acquired
        local.refill_local()
        scheduler.tick()
        assert not batches(pid)
    local.refill_local()
    scheduler.tick()
    assert len(batches(pid)) == 2


def test_cloud_refills_do_not_distort_gpu0_fairness(mixed):
    client, _, _, _, scheduler = mixed
    a = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000"])
    b = create(client, key="second-local-plan", execution="local", compute_pools=["local"])
    local.refill_local()
    complete(batches(a)[0])
    local.refill_local()
    assert len(batches(b)) == 1
    scheduler.tick()  # A newer cloud batch on A must not move A behind B for GPU0.
    complete(batches(b)[0])
    local.refill_local()
    assert sum(r.config["execution"] == "local" for r in batches(a)) == 2
    assert len(batches(b)) == 1


def test_durable_claim_conflict_rolls_back_entire_mixed_batch(mixed, monkeypatch):
    client, _, _, _, _ = mixed
    pid = create(client, execution="mixed", compute_pools=["local", "baidu-pro6000"])
    original = local.batch_config
    def raced(plan, rows):
        cfg = original(plan, rows)
        with db.session() as session:
            session.add(db.Run(id="competing-run", request_key="competing-request", kind="gt",
                               name="Already claimed", config=cfg))
            session.flush()
            session.add(db.CollectionClaim(source_id=rows[0]["id"], run_id="competing-run"))
        return cfg
    monkeypatch.setattr(local, "batch_config", raced)
    local.refill_local()
    assert not batches(pid)
    with db.session() as session:
        assert len(list(session.scalars(select(db.Run)))) == 1
        assert len(list(session.scalars(select(db.CollectionClaim)))) == 1


@pytest.mark.parametrize("changes", [
    {"compute_pools": ["local"]}, {"compute_pools": ["baidu-pro6000"]},
    {"compute_pools": ["local", "local"]}, {"compute_pools": ["local", "unknown"]},
    {"dataset": "scalevln"}])
def test_mixed_plan_rejects_invalid_devices_and_unsupported_dataset(mixed, changes):
    client, _, _, _, _ = mixed
    payload = {"dataset": "rxr", "execution": "mixed", "compute_pools": ["local", "baidu-pro6000"], **changes}
    for endpoint in ("/api/collection-plans/preflight", "/api/collection-plans"):
        response = client.post(endpoint, json=payload, headers={"Idempotency-Key": "invalid-mixed-plan"})
        assert response.status_code == 422, response.text


def test_mixed_execution_is_only_a_plan_mode_not_a_run_mode(mixed):
    client, _, _, _, _ = mixed
    response = client.post("/api/runs", json={"dataset": "rxr", "execution": "mixed"},
                           headers={"Idempotency-Key": "invalid-mixed-run"})
    assert response.status_code == 422
