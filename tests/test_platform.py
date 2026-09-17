import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from apps.api import db, main
from apps.api.settings import settings
from catalog.vln_catalog import build_database
from workers.tasks import execute, run_dir, archive_failed_output
from workers.seedance_entry import DurableProvider


@pytest.fixture
def platform(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread":False}, poolclass=StaticPool)
    monkeypatch.setattr(db,"Session",sessionmaker(engine,expire_on_commit=False))
    monkeypatch.setattr(main,"engine",engine)
    monkeypatch.setattr(settings,"runtime_root",tmp_path/"runtime")
    monkeypatch.setattr(settings,"resource_root",tmp_path/"resources")
    settings.runtime_root.mkdir();settings.resource_root.mkdir()
    source=tmp_path/"source";source.mkdir()
    (source/"train.json").write_text(json.dumps({"episodes":[{
        "episode_id":10288,"trajectory_id":1,"scene_id":"mp3d/example/example.glb",
        "instruction":{"instruction_text":"Walk forward.","language":"en-US"},
        "start_position":[0,0,0],"goals":[{"position":[1,0,0]}],"reference_path":[[0,0,0],[1,0,0]]}]}))
    config=tmp_path/"sources.json"
    config.write_text(json.dumps({"sources":[{"id":"rxr-test","dataset":"RxR","version":"test","format":"vlnce",
                    "root":str(source),"pattern":"{split}.json","splits":["train"],"roles":["guide"]}]}))
    build_database(config,settings.catalog_db)
    monkeypatch.setattr(main,"validate_gt",lambda cfg:None)
    dispatch=Mock();monkeypatch.setattr(main.execute_run,"delay",dispatch)
    with TestClient(main.app) as client:
        yield client,dispatch
    engine.dispose()


def test_rxr_selection_and_idempotency(platform):
    client,dispatch=platform
    payload={"dataset":"rxr","episode_ids":["10288"]}
    headers={"Idempotency-Key":"test-one-request"}
    first=client.post('/api/runs',json=payload,headers=headers)
    assert first.status_code==202,first.text
    second=client.post('/api/runs',json=payload,headers=headers)
    assert second.json()['id']==first.json()['id']
    assert dispatch.call_count==1
    conflict=client.post('/api/runs',json={**payload,"episode_ids":["10289"]},headers=headers)
    assert conflict.status_code==409
    assert first.json()['config']['source_records'][0]['instruction']=='Walk forward.'


def test_cross_site_start_rejected(platform):
    client,dispatch=platform
    response=client.post('/api/runs',json={},headers={"Origin":"https://unrelated.example","Idempotency-Key":"external-origin"})
    assert response.status_code==403
    dispatch.assert_not_called()


def test_reviews_persist_and_release_requires_completed_kept_run(platform):
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'review-test-run'}).json()['id']
    assert client.post('/api/releases/'+rid).status_code==409
    assert client.post('/api/reviews/'+rid,json={'decision':'keep','note':'检查通过'}).status_code==200
    assert client.get('/api/runs/'+rid).json()['reviews'][0]['note']=='检查通过'
    db.update_run(rid,status='succeeded')
    release=client.post('/api/releases/'+rid)
    assert release.status_code==200
    assert client.post('/api/releases/'+rid).json()['id']==release.json()['id']


def test_successful_stage_not_executed_twice(platform):
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'stage-resume-test'}).json()['id']
    target=run_dir(rid)/'counter.txt'
    command=[sys.executable,'-c',"from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text()+'x' if p.exists() else 'x')",str(target)]
    execute(rid,'counter',command)
    execute(rid,'counter',command)
    assert target.read_text()=='x'


def test_unknown_output_not_overwritten(platform):
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'unknown-output-test'}).json()['id']
    target=run_dir(rid)/'capture';target.mkdir(parents=True)
    (target/'evidence').write_text('preserve')
    with pytest.raises(RuntimeError,match='无执行记录'):
        archive_failed_output(rid,'capture',target)
    assert (target/'evidence').read_text()=='preserve'


def test_provider_resume_never_submits_twice(tmp_path):
    provider=Mock();provider.create_task.return_value={'id':'provider-1'}
    wrapper=DurableProvider(provider,tmp_path/'journal')
    assert wrapper.create_task(model='test')['id']=='provider-1'
    assert DurableProvider(provider,tmp_path/'journal').create_task(model='test')['id']=='provider-1'
    assert provider.create_task.call_count==1


def test_lost_provider_response_requires_reconciliation(tmp_path):
    provider=Mock();provider.create_task.side_effect=TimeoutError('response lost')
    wrapper=DurableProvider(provider,tmp_path/'journal')
    with pytest.raises(TimeoutError):wrapper.create_task(model='test')
    with pytest.raises(RuntimeError,match='禁止自动重提'):wrapper.create_task(model='test')
    assert provider.create_task.call_count==1


def test_stage_completed_before_worker_ack_is_recovered(platform):
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'recover-ack-window'}).json()['id']
    target=run_dir(rid)/'counter.txt'
    command=[sys.executable,'-c',"from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text()+'x' if p.exists() else 'x')",str(target)]
    execute(rid,'counter',command)
    (run_dir(rid)/'stages/counter/success.json').unlink()
    archive_failed_output(rid,'counter',target)
    execute(rid,'counter',command)
    assert target.read_text()=='x'
    assert len(list((run_dir(rid)/'stages/counter').glob('attempt-*')))==1


def test_duplicate_delivery_does_not_retry_failed_run(platform,monkeypatch):
    from workers import tasks
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'failed-duplicate-delivery'}).json()['id']
    db.update_run(rid,status='failed',error='intentional failure')
    collect=Mock();monkeypatch.setattr(tasks,'collect_gt',collect)
    tasks.execute_run.run(rid)
    collect.assert_not_called()
    assert client.get('/api/runs/'+rid).json()['status']=='failed'


def test_failed_generation_exposes_partial_video(platform,monkeypatch):
    from workers import tasks
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'failed-generation-artifacts'}).json()['id']
    def fail(*args,**kwargs):
        directory=run_dir(rid)/'generation/work/sample'
        directory.mkdir(parents=True)
        (directory/'generated.mp4').write_bytes(b'partial artifact preserved')
        raise RuntimeError('downstream estimator failed')
    monkeypatch.setattr(tasks,'execute',fail)
    with pytest.raises(RuntimeError,match='estimator'):
        tasks.generate_seedance(rid,{})
    artifacts=client.get('/api/runs/'+rid).json()['artifacts']
    assert len(artifacts)==1
    assert artifacts[0]['kind']=='video'
    assert artifacts[0]['details']['ground_truth'] is False


def test_provider_poll_retries_reads_without_new_submission(tmp_path):
    from navgen_pipeline.ark_client import ArkHTTPError
    provider=Mock()
    provider.get_task.side_effect=[ArkHTTPError(0,'temporary network failure'),{'id':'provider-1','status':'succeeded'}]
    wrapper=DurableProvider(provider,tmp_path/'journal')
    result=wrapper.wait_for_task('provider-1',poll_s=.01,timeout_s=2)
    assert result['status']=='succeeded'
    assert provider.get_task.call_count==2
    provider.create_task.assert_not_called()


def test_quality_findings_do_not_fail_generation_execution():
    from workers.seedance_entry import execution_errors
    assert execution_errors(['seedance_rgb_video_is_not_ground_truth','instruction_path_direction_mismatch'])==[]
    assert execution_errors(['pipeline_error:network failed','monocular_scale_is_unknown'])==['pipeline_error:network failed']
