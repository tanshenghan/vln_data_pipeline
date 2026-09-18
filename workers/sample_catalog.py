"""Register immutable run/sample identities without changing source records."""
import hashlib
import json
from pathlib import Path
from sqlalchemy import select
from apps.api.db import DataSample,Artifact,Run,session
from apps.api.settings import settings
from catalog.vln_catalog import connect


def register_samples(run_id):
    directory=settings.runs_root/run_id
    with session() as db:
        run=db.get(Run,run_id)
        if not run or run.config.get('operation')=='preview':return
        artifacts=db.scalars(select(Artifact).where(Artifact.run_id==run_id)).all()
        cfg=run.config
        sources=cfg.get('source_records',[])
        source_by_id={r['id']:r for r in sources}
        by_path={str(Path(a.path).resolve()):a for a in artifacts}
        pending=[]
        if run.kind=='gt':
            metadata=directory/'dataset/meta/beta_nav.json'
            if not metadata.exists():return
            catalog=connect(settings.catalog_db)
            try:
                for episode in json.loads(metadata.read_text())['episodes']:
                    eid=str(episode['episode_id']);scene=Path(episode['scene_id']).stem
                    matches=[r for r in sources if str(r['episode_id'])==eid and r['scene']==scene and r['instruction'].strip()==episode['instruction'].strip()]
                    if not matches:
                        rows=catalog.execute('SELECT detail FROM episodes WHERE dataset=? AND split=? AND role=? AND episode_id=? AND scene=?',
                            ({'r2r':'R2R','rxr':'RxR'}[cfg['dataset']],cfg['split'],cfg['role'],eid,scene)).fetchall()
                        matches=[json.loads(r[0]) for r in rows if json.loads(r[0])['instruction'].strip()==episode['instruction'].strip()]
                    if len(matches)!=1:raise ValueError('采集产物无法唯一匹配来源样本：'+eid)
                    source=matches[0]
                    videos=[a for a in artifacts if a.kind=='video' and a.details.get('episode_id')==eid]
                    videos.sort(key=lambda a: not a.name.endswith('_gt.mp4'))
                    relevant=[a for a in artifacts if a.kind!='video']+videos
                    details={'path_length':source.get('path_length'),'route_id':source['route_id'],
                        'source_version':source['version'],'frames':episode['frames'],'source_instruction_id':source['instruction_id'],
                        'artifact_ids':[a.id for a in relevant],'video_artifact_id':videos[0].id if videos else None,
                        'missing':{},'issues':[],'run_name':run.name}
                    if cfg['dataset']=='vlnverse':
                        details.update(trajectory_source=episode.get('trajectory_source'),vlm_report=episode.get('vlm_review'),semantic_quality=episode.get('semantic_quality'))
                    pending.append((str(episode['episode_index']),source,'validated',details))
            finally:catalog.close()
        else:
            manifest=directory/'generation/manifest.jsonl'
            if not manifest.exists():return
            for line in manifest.read_text().splitlines():
                row=json.loads(line);spec=row['sample_spec'];meta=spec.get('metadata',{})
                sid=meta.get('source_record_id') or row['sample_id'].removeprefix('source_')
                source=source_by_id.get(sid)
                if source is None:
                    source={'id':None,'dataset':'Seedance','split':'generated','role':'generated','language':'unknown',
                        'scene':spec['scene']['scene_id'],'episode_id':row['sample_id'],
                        'instruction':spec['task'].get('instruction_original') or spec['task']['instruction_zh']}
                sample_prefix=(directory/'generation/samples'/row['sample_id']).resolve()
                relevant=[a for a in artifacts if Path(a.path).resolve().is_relative_to(sample_prefix)]
                video=by_path.get(str(Path(row['video_path']).resolve())) if row.get('video_path') else None
                details={'path_length':None,'route_id':run_id+':'+row['sample_id'],'source_version':source.get('version'),
                    'sample_id':row['sample_id'],'artifact_ids':[a.id for a in relevant],
                    'video_artifact_id':video.id if video else None,'issues':row.get('errors',[]),
                    'missing':{'depth':'未提供','calibration':'未提供可信标定','body_pose':'未提供 GT',
                               'fm':'未提供','rvq':'未提供','apos':'未提供','opos':'未提供'},
                    'trajectory':'估计轨迹，非 GT','quality_tier':row['quality_tier'],
                    'trajectory_report':row.get('trajectory_report'),'vlm_report':row.get('vlm_report'),
                    'usage':row.get('usage_report'),'prompt':spec.get('seedance_prompt'),'run_name':run.name}
                pending.append((row['sample_id'],source,row['status'],details))
        for key,source,quality,details in pending:
            sample_id='produced_'+hashlib.sha256((run_id+':'+key).encode()).hexdigest()[:32]
            values={field:str(source[field]) for field in ('dataset','split','role','language','scene','episode_id','instruction')}
            values.update(run_id=run_id,origin=run.kind,source_record_id=source['id'],quality=quality,details=details)
            existing=db.get(DataSample,sample_id)
            if existing:
                for field,value in values.items():setattr(existing,field,value)
            else:db.add(DataSample(id=sample_id,**values))


def register_vlnverse_candidates(run_id):
    directory=settings.runs_root/run_id/'trajectory_review'
    report=json.loads((directory/'summary.json').read_text())
    with session() as db:
        run=db.get(Run,run_id)
        sources={r['episode_id']:r for r in run.config['source_records']}
        artifacts=db.scalars(select(Artifact).where(Artifact.run_id==run_id)).all()
        for row in report['episodes']:
            if row['accepted']:continue
            eid=row['episode_id'];source=sources[eid]
            metadata=json.loads((directory/eid/'episode.json').read_text())
            relevant=[a for a in artifacts if a.details.get('episode_id')==eid or a.kind=='capture']
            videos=[a for a in relevant if a.kind=='video']
            sample_id='produced_'+hashlib.sha256((run_id+':candidate:'+eid).encode()).hexdigest()[:32]
            values={field:str(source[field]) for field in ('dataset','split','role','language','scene','episode_id','instruction')}
            values.update(run_id=run_id,origin='gt',source_record_id=source['id'],quality=metadata['semantic_quality'],
                details={'candidate':True,'trajectory_source':row['trajectory_source'],'vlm_report':row,
                    'frames':metadata['frames'],'terminal_command':metadata['terminal_command'],
                    'artifact_ids':[a.id for a in relevant],'video_artifact_id':videos[0].id if videos else None,
                    'run_name':run.name,'issues':[row['reason']],
                    'missing':{'training_export':'审核未通过或缺少明确 STOP，保留原始采集供复核'},
                    'route_id':source['route_id'],'source_version':source['version']})
            existing=db.get(DataSample,sample_id)
            if existing:
                for key,value in values.items():setattr(existing,key,value)
            else:db.add(DataSample(id=sample_id,**values))
