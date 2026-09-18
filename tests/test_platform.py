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
    assert client.post('/api/releases/'+rid).status_code==409
    from workers.tasks import register
    from contracts.check import CONTRACT_PATH
    dataset=run_dir(rid)/'dataset';(dataset/'meta').mkdir(parents=True)
    contract=json.loads(CONTRACT_PATH.read_text())
    (dataset/'meta/beta_nav.json').write_text(json.dumps({**contract['metadata'],'features':contract['features'],'episodes':[{k:contract[k] for k in ('pointing_config','camera')}]}))
    report={k:True for k in ('complete','self_contained','official_loader_all_frames','all_30_source_arrays_exact','geometry_recomputed')}
    (dataset/'meta/platform_validation.json').write_text(json.dumps(report))
    register(rid,'dataset',dataset)
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


def test_preview_snapshot_reused_and_changed_parameters_rejected(platform,monkeypatch):
    client,dispatch=platform
    monkeypatch.setattr(main,'health',lambda:{'seedance_ready':True})
    payload={'kind':'seedance','operation':'preview','dataset':'rxr','episode_ids':['10288'],'duration':8}
    preview=client.post('/api/runs',json=payload,headers={'Idempotency-Key':'frozen-preview-test'}).json()
    rid=preview['id'];db.update_run(rid,status='succeeded')
    assert 'Prompt' in client.post('/api/releases/'+rid).json()['detail']
    frozen=[{'sample_id':'frozen-test','seedance_prompt':'This is the exact preview, unchanged.'}]
    target=run_dir(rid)/'generation/prepared_samples.json';target.parent.mkdir(parents=True)
    target.write_text(json.dumps(frozen))
    actual={**payload,'operation':'generate','preview_id':rid}
    started=client.post('/api/runs',json=actual,headers={'Idempotency-Key':'frozen-generate-test'})
    assert started.status_code==202,started.text
    assert started.json()['config']['prepared_samples']==frozen
    repeat=client.post('/api/runs',json=actual,headers={'Idempotency-Key':'frozen-generate-test'})
    assert repeat.json()['id']==started.json()['id']
    changed=client.post('/api/runs',json={**actual,'duration':9},headers={'Idempotency-Key':'changed-preview-test'})
    assert changed.status_code==422
    assert '重新预览' in changed.json()['detail']
    assert dispatch.call_count==2


def test_production_catalog_filters_export_and_non_gt_identity(platform):
    from apps.api.db import DataSample
    client,_=platform
    first=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'catalog-parent-one'}).json()
    source_id=first['config']['source_ids'][0]
    with db.session() as session:
        for origin,quality,media in [('gt','validated','video-1'),('seedance','review',None)]:
            session.add(DataSample(id='produced_'+origin,run_id=first['id'],origin=origin,dataset='RxR',split='train',role='guide',language='en-US',
                scene='example',episode_id='10288',instruction='Walk forward.',source_record_id=source_id,quality=quality,
                details={'route_id':origin,'path_length':3.5 if origin=='gt' else None,'video_artifact_id':media,'artifact_ids':[],
                         'missing':{} if origin=='gt' else {'depth':'未提供'}}))
    r=client.get('/vln-api/episodes',params={'origin':'produced','dataset':'RxR','limit':1}).json()
    assert r['stats']['records']==2 and len(r['items'])==1
    assert r['stats']['media']==1
    assert client.get('/vln-api/episodes',params={'origin':'seedance','quality':'issues'}).json()['stats']['records']==1
    detail=client.get('/vln-api/episode',params={'id':'produced_seedance'}).json()
    assert detail['ground_truth'] is False
    assert detail['missing']['depth']=='未提供'
    assert detail['source_record_id']==source_id
    rows=client.get('/api/sources/'+source_id+'/derived').json()
    assert {r['origin'] for r in rows}=={'gt','seedance'}
    exported=client.post('/vln-api/export',data={'payload':json.dumps({'filters':{'origin':'produced','media':'yes','min_length':'3'}})})
    assert exported.status_code==200,exported.text
    assert [r['id'] for r in exported.json()['samples']]==['produced_gt']


def test_blank_observed_semantics_are_missing_not_invented():
    from dataclasses import asdict
    from navgen_pipeline.planner import TemplatePlanner,sample_spec_from_dict
    sample=TemplatePlanner().plan_one(0)
    data=asdict(sample)
    data['scene'].update(semantic_label='',semantic_confidence=' ',semantic_source='')
    parsed=sample_spec_from_dict(data,sample.sample_id,sample.seed)
    assert parsed.scene.semantic_confidence is None
    assert parsed.scene.semantic_label is None
    assert parsed.scene.semantic_source is None


def test_planner_response_is_reused_after_parser_failure(tmp_path):
    from workers.seedance_entry import InstructionClient
    client=Mock();client.call_history=[{'model':'planner','usage':{'input_tokens':100}}]
    client.generate_text.return_value='{"scene":{"semantic_confidence":""}}'
    wrapper=InstructionClient(client,'walk forward',tmp_path/'responses')
    first=wrapper.generate_text(model='planner',system='JSON',user_text='Plan')
    client.call_history=[]
    second=wrapper.generate_text(model='planner',system='JSON',user_text='Plan')
    assert first==second
    assert client.generate_text.call_count==1
    assert client.call_history[0]['usage']['input_tokens']==100


def test_dataset_archive_contains_files_and_path_escape_is_denied(platform):
    import io,tarfile
    from workers.tasks import register
    client,_=platform
    rid=client.post('/api/runs',json={'dataset':'rxr'},headers={'Idempotency-Key':'archive-run'}).json()['id']
    directory=run_dir(rid)/'dataset';(directory/'meta').mkdir(parents=True)
    (directory/'meta/info.json').write_text('{"test":true}')
    register(rid,'dataset',directory)
    artifact=client.get('/api/runs/'+rid).json()['artifacts'][0]
    response=client.get('/api/artifacts/'+artifact['id'],params={'archive':'tar'})
    assert response.status_code==200
    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        assert archive.extractfile('./meta/info.json').read()==b'{"test":true}'
    escaped=client.get('/api/artifacts/'+artifact['id'],params={'file':'../../../../etc/passwd'})
    assert escaped.status_code==403


def test_cancel_paused_batch_is_terminal_without_redispatch(platform):
    client,dispatch=platform
    rid=client.post('/api/runs',json={'dataset':'rxr','full_dataset':True},headers={'Idempotency-Key':'cancel-paused-batch'}).json()['id']
    assert client.post('/api/runs/'+rid+'/pause').status_code==200
    assert client.get('/api/runs/'+rid).json()['status']=='paused'
    assert client.post('/api/runs/'+rid+'/cancel').status_code==200
    assert client.get('/api/runs/'+rid).json()['status']=='cancelled'
    assert json.loads((run_dir(rid)/'control.json').read_text())=={'action':'cancel'}
    assert dispatch.call_count==1


@pytest.mark.parametrize('decision,accepted,terminal,valid,expected', [
    ('pass',True,True,True,200),
    ('reject',True,True,True,409),
    ('uncertain',False,True,True,409),
    ('pass',True,False,True,409),
    ('pass',True,True,False,409),
])
def test_vlnverse_release_uses_automatic_review_without_manual_gate(platform,decision,accepted,terminal,valid,expected):
    client,_=platform
    from workers.tasks import register
    from contracts.check import CONTRACT_PATH
    rid='automatic-vlnverse-test'
    with db.session() as session:
        session.add(db.Run(id=rid,request_key=rid,kind='gt',name='Automatic GT',status='succeeded',config={'dataset':'vlnverse'}))
    dataset=run_dir(rid)/'dataset';(dataset/'meta').mkdir(parents=True)
    contract=json.loads(CONTRACT_PATH.read_text())
    episode={k:contract[k] for k in ('pointing_config','camera')}
    episode['vlm_review']={'decision':decision,'accepted':accepted,'explicit_terminal':terminal}
    (dataset/'meta/beta_nav.json').write_text(json.dumps({**contract['metadata'],'features':contract['features'],'episodes':[episode]}))
    report={k:valid for k in ('complete','self_contained','official_loader_all_frames','all_30_source_arrays_exact','geometry_recomputed')}
    (dataset/'meta/platform_validation.json').write_text(json.dumps(report))
    register(rid,'dataset',dataset)
    assert client.post('/api/releases/'+rid).status_code==expected
    with db.session() as session:
        assert session.query(db.Review).count()==0
