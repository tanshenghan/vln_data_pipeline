"""Verify evidence from a real completed web/worker LightNav collection, including semantic review."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apps.api.db import session,Run
from apps.api.settings import settings
from workers.command_runner import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-id');group.add_argument('--collect',action='store_true')
    parser.add_argument('--episode-id',default='kujiale_0010_36_1')
    parser.add_argument('--max-steps',type=int,default=80)
    args=parser.parse_args()
    if args.collect:
        import uuid
        from apps.api.schemas import RunRequest
        from apps.api.catalog_service import resolve_selection
        from apps.api.capacity import estimate_gt
        from workers.tasks import execute_run
        request=RunRequest(dataset='vlnverse',split='test',role='coarse',episode_ids=[args.episode_id],lightnav_max_steps=args.max_steps)
        config=resolve_selection(request.model_dump(mode='json'))
        config['request_payload']=request.model_dump(mode='json')
        if not estimate_gt(config)['can_start']:raise ValueError('insufficient capture storage')
        args.run_id=str(uuid.uuid4())
        with session() as db:
            db.add(Run(id=args.run_id,request_key='acceptance-'+args.run_id,kind='gt',
                name='VLNVerse LightNav deployment acceptance',config=config))
        print(json.dumps({'acceptance_run_id':args.run_id}),flush=True)
        execute_run.run(args.run_id)
    with session() as db:
        run=db.get(Run,args.run_id)
        if not run or run.status!='succeeded' or run.config['dataset']!='vlnverse':raise ValueError('requires completed VLNVerse run')
        config=run.config
    root=settings.runs_root/args.run_id
    summary=json.loads((root/'trajectory_review/summary.json').read_text())
    reports=[]
    for row in summary['episodes']:
        if row['trajectory_source']!='lightnav_inference':continue
        folder=root/'capture'/('episodes' if row['accepted'] else 'candidates')/row['episode_id']
        frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
        responses=[json.loads(x) for x in (folder/'model_responses.jsonl').read_text().splitlines()]
        assert len(frames)>1 and len(frames)==len(responses)
        assert [r['seq'] for r in responses]==list(range(len(frames)))
        assert all(r['rc']==0 and r.get('raw_text') for r in responses)
        assert all((folder/f['rgb']).is_file() and (folder/f['depth']).is_file() for f in frames)
        poses=np.array([f['world_from_body'] for f in frames]);assert np.isfinite(poses).all()
        assert row['call']['status']=='completed' and row['model'] and row['decision'] in {'pass','reject','uncertain'}
        if not row['accepted']:assert not (root/'dataset/meta/beta_nav.json').exists() or row['episode_id'] not in [e['episode_id'] for e in json.loads((root/'dataset/meta/beta_nav.json').read_text())['episodes']]
        reports.append({'episode_id':row['episode_id'],'frames':len(frames),'inference_calls':len(responses),
            'path_length_m':float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).sum()),
            'review_decision':row['decision'],'accepted':row['accepted']})
    assert reports,'no LightNav evidence'
    report={'complete':True,'run_id':args.run_id,'episodes':reports,'endpoint':config['lightnav_endpoint'],
        'meaning':'real RGB-conditioned inference, native USD observations, VLM review and candidate/export gating verified; not navigation success guarantee'}
    atomic_json(settings.runtime_root/'vlnverse_lightnav_validation.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
