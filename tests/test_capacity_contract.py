import copy
import gzip
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import pytest
from apps.api import capacity
from apps.api.settings import settings
from contracts.check import check_metadata, CONTRACT_PATH


@pytest.mark.parametrize('dataset,expected', [('r2r', 4), ('rxr', 6)])
def test_capacity_counts_actual_replay_frames_and_checks_disk(tmp_path, monkeypatch, dataset, expected):
    monkeypatch.setattr(settings, 'resource_root', tmp_path)
    monkeypatch.setattr(settings, 'source_root', None)
    monkeypatch.setattr(settings, 'rvq_bundle_root', None)
    folder = settings.dataset_root(dataset) / 'val_seen'
    folder.mkdir(parents=True)
    stem = 'val_seen' + ('_guide' if dataset == 'rxr' else '')
    values = [(stem+'.json.gz', {'episodes':[{'episode_id':1,'scene_id':'mp3d/scene/scene.glb','instruction':{'language':'en-US'}}]}),
              (stem+'_gt.json.gz', {'1':{'actions':[2,1,3,0]}})]
    if dataset == 'rxr':
        # Original RxR files encode unavailable speech timing as bare NaN.
        values[0][1]['episodes'][0]['instruction']['timed_instruction'] = [
            {'word': 'forward', 'start_time': float('nan'), 'end_time': float('nan')}]
    for filename, data in values:
        with gzip.open(folder/filename, 'wt') as stream: json.dump(data,stream)
    scene = tmp_path/'scene_datasets/mp3d/scene/scene.glb'
    scene.parent.mkdir(parents=True);scene.touch();scene.with_suffix('.navmesh').touch()
    cfg = dict(dataset=dataset,split='val_seen',role='guide',languages=['all'],episode_ids=['1'],full_dataset=False)
    result = capacity.estimate_gt(cfg)
    assert result['frames']==expected and result['episodes']==1
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    assert capacity.estimate_gt(cfg)['can_start'] is False
    cfg['episode_ids']=['missing']
    with pytest.raises(ValueError,match='完整'):capacity.estimate_gt(cfg)


def test_contract_rejects_false_time_and_shape_claims(tmp_path):
    contract=json.loads(CONTRACT_PATH.read_text())
    metadata={**contract['metadata'],'features':contract['features'],'episodes':[{k:contract[k] for k in ('pointing_config','camera')}]}
    path=tmp_path/'meta/beta_nav.json';path.parent.mkdir()
    path.write_text(json.dumps(metadata))
    assert check_metadata(tmp_path)['metadata_compatible']
    bad=copy.deepcopy(metadata);bad['fps']=8;path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='fps'):check_metadata(tmp_path)
    bad=copy.deepcopy(metadata);bad['features']['action.fm.waypoints_ego']['shape']=[10,3]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='waypoints_ego'):check_metadata(tmp_path)


def test_current_contract_does_not_relabel_historical_opos(tmp_path):
    contract=json.loads(CONTRACT_PATH.read_text())
    metadata={**contract['metadata'],'features':contract['features'],'episodes':[{k:contract[k] for k in ('pointing_config','camera')}]}
    metadata['episodes'][0]['pointing_config']['opos_visibility_window']='all'
    path=tmp_path/'meta/beta_nav.json';path.parent.mkdir();path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='opos_visibility_window'):check_metadata(tmp_path)
