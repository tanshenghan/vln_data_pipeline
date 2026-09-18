import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from apps.api.settings import settings
from apps.api.catalog_service import resolve_selection
from apps.api.schemas import RunRequest
from catalog import vlnverse_data, vlnverse_scenes
from configs.sync_vlnverse import matches
from data_pipeline.vlnverse_source import replay_plan, canonical_pose, body_pose_source, SOURCE_TO_BETA


def source_record():
    return {'episode_id':'kujiale_0001_1_0','scene_id':'vlnverse/kujiale_0001','scan':'kujiale_0001',
        'start_position':[0,0,0],'start_rotation':[1,0,0,0],
        'instruction':{'instruction_text':{'natural':'Walk ahead.'}},
        'reference_path':[[0,0,0],[.25,0,0],[.5,0,0]],'goals':{'position':[.5,0,0],'radius':3}}


@pytest.fixture
def verse(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'resource_root',tmp_path)
    monkeypatch.setattr(settings,'vlnverse_managed_scene_root',None)
    monkeypatch.setattr(settings,'vlnverse_scene_root',None)
    root=settings.dataset_root('vlnverse');root.mkdir(parents=True)
    raw=source_record()
    for split in ['val','test']:
        row=dict(raw)
        if split=='test':row.pop('reference_path');row.pop('goals')
        with gzip.open(root/f'coarse_{split}.json.gz','wt') as f:json.dump({'episodes':[row]},f)
    scene=tmp_path/'scene_datasets/vlnverse/kujiale_0001';scene.mkdir(parents=True)
    (scene/'kujiale_0001.usda').write_text('#usda 1.0\n(upAxis = "Z"\nmetersPerUnit = 1)')
    return root,scene


def test_catalog_preserves_test_absence_and_transforms_reference(verse):
    rows=vlnverse_data.records(verse[0]);test=next(r for r in rows if r['split']=='test')
    val=next(r for r in rows if r['split']=='val_seen')
    assert test['has_reference_gt'] is False and test['reference_path']==[]
    assert val['has_reference_gt'] and val['reference_path'][-1]==[.5,0.,0.]
    assert val['source_record']==source_record()
    assert len(val['source_file_sha256'])==64


def test_select_only_installed_public_reference_and_freeze_source(verse):
    cfg=RunRequest(dataset='vlnverse',split='val_seen',role='coarse').model_dump()
    result=resolve_selection(cfg)
    assert result['source_records'][0]['source_record']==source_record()
    assert result['episode_ids']==['kujiale_0001_1_0']
    cfg['source_ids']=['vlnverse_missing']
    with pytest.raises(ValueError,match='未找到'):resolve_selection(cfg)


def test_z_up_conversion_is_right_handed_and_preserves_forward():
    np.testing.assert_allclose(SOURCE_TO_BETA@SOURCE_TO_BETA.T,np.eye(3))
    assert np.linalg.det(SOURCE_TO_BETA)==pytest.approx(1)
    for yaw in [0,np.pi/2,np.pi]:
        pose=canonical_pose(body_pose_source([1,2,3],yaw))
        np.testing.assert_allclose(pose[:3,3],[1,3,-2])
        np.testing.assert_allclose(pose[:3,1],[0,1,0])
        np.testing.assert_allclose(-pose[:3,2],SOURCE_TO_BETA@np.array([np.cos(yaw),np.sin(yaw),0]))


def test_reference_replay_terminal_and_no_synthetic_test_path():
    raw=source_record();plan=replay_plan({'source_record':raw})
    assert len(plan['poses_source'])==3
    assert plan['commands'][-1]=='stop_at_verified_goal'
    raw.pop('reference_path')
    with pytest.raises(ValueError,match='public reference'):replay_plan({'source_record':raw})


def test_heading_transition_retains_exact_source_waypoints():
    raw=source_record();raw['reference_path']=[[0,0,0],[0,.25,0]];raw['goals']['position']=[0,.25,0]
    plan=replay_plan({'source_record':raw})
    assert len(plan['poses_source'])==8
    np.testing.assert_allclose(plan['poses_source'][:-1,:3,3],0)
    np.testing.assert_allclose(plan['poses_source'][-1,:3,3],[0,.25,0])
    raw['goals']['position']=[10,0,0]
    with pytest.raises(ValueError,match='endpoint'):replay_plan({'source_record':raw})


def test_scene_asset_boundary(verse,tmp_path):
    root=verse[1].parent
    assert vlnverse_scenes.detail(root,'kujiale_0001')['up_axis']=='Z'
    with pytest.raises(ValueError):vlnverse_scenes.detail(root,'../kujiale_0001')
    with pytest.raises(ValueError):vlnverse_scenes.media_path(root,'kujiale_0001','secret')
    external=tmp_path/'external';external.write_text('not scene media')
    (verse[1]/'occupancy.png').symlink_to(external)
    with pytest.raises(FileNotFoundError):vlnverse_scenes.media_path(root,'kujiale_0001','occupancy')


def test_sync_checks_both_git_and_lfs_hashes(tmp_path):
    p=tmp_path/'asset';p.write_bytes(b'abc')
    git={'size':3,'oid':hashlib.sha1(b'blob 3\0abc').hexdigest()}
    lfs={'size':3,'oid':'unused','lfs':{'oid':hashlib.sha256(b'abc').hexdigest()}}
    assert matches(p,git) and matches(p,lfs)
    p.write_bytes(b'abd')
    assert not matches(p,git) and not matches(p,lfs)


def test_vlnverse_request_rejects_unsupported_paths():
    for kwargs in [{'role':'guide'},{'role':'coarse','full_dataset':True},{'role':'coarse','kind':'seedance'}]:
        with pytest.raises(ValueError):RunRequest(dataset='vlnverse',**kwargs)


def test_scene_inventory_follows_pagination(tmp_path,monkeypatch):
    from configs import sync_vlnverse as syncer
    calls=[]
    def curl(argv,**kwargs):
        url=argv[-3];calls.append(url)
        page=Path(argv[-1]);headers=Path(argv[argv.index('-D')+1])
        n=len(calls)
        page.write_text(json.dumps([{'type':'file','path':f'kujiale_0001/{n}.usd','size':1,'oid':'x'}]))
        headers.write_text('link: <https://huggingface.co/api/datasets/page2>; rel="next"\n' if n==1 else '')
    monkeypatch.setattr(syncer.subprocess,'run',curl)
    rows=syncer.tree('kujiale_0001','revision',tmp_path)
    assert len(calls)==2 and len(rows)==2
    assert syncer.tree('kujiale_0001','revision',tmp_path)==rows
    assert len(calls)==2  # Exact revision inventory can be resumed locally.


def test_complete_sync_uses_root_tree_not_truncated_siblings(tmp_path,monkeypatch):
    from argparse import Namespace
    from configs import sync_vlnverse as syncer
    scenes=['kujiale_0001','kujiale_0002'];root=tmp_path/'assets';root.mkdir()
    def row(path):
        return dict(type='file',path=path,size=3,oid=hashlib.sha1(b'blob 3\0abc').hexdigest())
    for scene in scenes:
        (root/scene).mkdir();(root/scene/'scene.usd').write_bytes(b'abc')
    (root/'.gitattributes').write_bytes(b'abc')
    info=tmp_path/'info.json'
    info.write_text(json.dumps({'sha':'pinned','siblings':[{'rfilename':'kujiale_0001/scene.usd'}]}))
    def curl(argv,**kwargs):
        Path(argv[-1]).write_text(json.dumps([{'type':'directory','path':s} for s in scenes]+[row('.gitattributes')]))
    monkeypatch.setattr(syncer.subprocess,'run',curl)
    monkeypatch.setattr(syncer,'tree',lambda scene,revision,metadata:[row(scene+'/scene.usd')])
    state=tmp_path/'state'
    syncer.sync(Namespace(root=root,state=state,repo_info=info,reuse=[],workers=4))
    report=json.loads((state/'status.json').read_text())
    assert report['status']=='complete' and report['completed_scenes']==2
    assert report['verified_files']==report['total_files']==3
    assert all((root/s/'platform_asset_manifest.json').exists() for s in scenes)


def test_test_split_selects_inference_without_fabricating_reference(verse):
    cfg=resolve_selection(RunRequest(dataset='vlnverse',split='test',role='coarse').model_dump())
    assert cfg['collection_strategy']=='public_reference_else_lightnav_vlm'
    record=cfg['source_records'][0]
    assert not record['has_reference_gt'] and not record['goals'] and not record['reference_path']
    assert 'reference_path' not in record['source_record']
    with pytest.raises(ValueError):RunRequest(dataset='r2r',split='test')


def test_lightnav_action_is_bounded_and_invalid_stop_is_rejected():
    from data_pipeline.vlnverse_lightnav import command
    reply={'actions':{'actions':[[0,0,0],[4,12,-4]]+[[0,0,0]]*8},'stop':False}
    forward,yaw=command(reply)
    assert forward==.25 and yaw==pytest.approx(-np.pi/6)
    reply['stop']=True
    with pytest.raises(ValueError,match='evidence'):command(reply)
    reply['raw_text']='<act_l0_6><act_l1_122><act_l2_174>'
    assert command(reply)==(0,0)
    reply['actions']['actions'][0][0]=float('nan')
    with pytest.raises(ValueError,match='waypoints'):command(reply)


def test_review_requires_positive_evidence_and_known_frame_ids():
    from data_pipeline.review_vlnverse import parse_review
    row={'decision':'pass','confidence':.9,'reason':'The endpoint is visible.','evidence_frame_ids':[3],
         'route_correct':True,'destination_reached':True,'limitations':[]}
    assert parse_review(json.dumps(row),[0,3])['decision']=='pass'
    with pytest.raises(ValueError,match='evidence'):parse_review(json.dumps(row),[0,2])
    row['destination_reached']=None
    with pytest.raises(ValueError,match='requires'):parse_review(json.dumps(row),[0,3])


def test_review_retains_nonstop_candidate_even_if_vlm_passes(tmp_path,monkeypatch):
    from data_pipeline import review_vlnverse as reviewer
    root=tmp_path/'capture';folder=root/'episodes/e';folder.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps({'episode_ids':['e'],'episodes':1}))
    (folder/'0.png').write_bytes(b'fake image fixture')
    (folder/'frames.jsonl').write_text(json.dumps({'frame_id':0,'rgb':'0.png','collision_from_previous_action':False})+'\n')
    (folder/'episode.json').write_text(json.dumps({'episode_id':'e','instruction':'Go to door.',
        'terminal_command':'step_limit','trajectory_source':'lightnav_inference','collision_frames':0}))
    env=tmp_path/'review.env';env.write_text('ARK_API_KEY=test-only-not-a-secret\n')
    calls=[]
    def generate(self,*args,**kwargs):
        calls.append(1)
        return json.dumps({'decision':'pass','confidence':.9,'reason':'At door.','evidence_frame_ids':[0],
            'route_correct':True,'destination_reached':True,'limitations':[]})
    monkeypatch.setattr(reviewer.ArkResponsesClient,'generate_text',generate)
    result=reviewer.review(root,env)
    assert result['accepted']==0 and result['candidates']==1
    assert not folder.exists() and (root/'candidates/e/0.png').exists()
    assert json.loads((root/'manifest.json').read_text())['episode_ids']==[]
    assert reviewer.review(root,env)['candidates']==1
    assert len(calls)==1  # Saved provider reply is reused; no second paid call.


def test_failed_candidate_never_advertises_gt():
    from apps.api.db import DataSample
    from apps.api.produced_catalog import public_sample
    row=DataSample(id='candidate',run_id='run',origin='gt',source_record_id='source',
        details={'candidate':True,'video_artifact_id':'video'},quality='rejected_vlm')
    result=public_sample(row)
    assert result['has_media'] and not result['ground_truth'] and not result['has_gt'] and not result['has_gt_video']
