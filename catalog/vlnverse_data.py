"""VLNVerse annotations stay separate from the existing MP3D index snapshot."""
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
from .vlnverse_scenes import usd_path


def canonical_point(p):
    return [float(p[0]),float(p[2]),-float(p[1])]


@lru_cache(maxsize=4)
def load_records(root, stamps):
    result=[]
    for filename,mtime,size in stamps:
        path=Path(root)/filename
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        granularity,raw_split=filename.removesuffix('.json.gz').split('_',1)
        split={'val':'val_seen'}.get(raw_split,raw_split)
        with gzip.open(path,'rt') as f:rows=json.load(f)['episodes']
        for row in rows:
            eid=str(row['episode_id']);scene=row.get('scan') or Path(row['scene_id']).name
            texts=row['instruction']['instruction_text']
            instruction=texts.get('natural') or texts.get('formal') if isinstance(texts,dict) else texts
            points=[canonical_point(p) for p in (row.get('reference_path') or [])]
            key='vlnverse_'+hashlib.sha256(f'{filename}:{eid}:natural'.encode()).hexdigest()[:24]
            source_id='vlnverse-'+granularity
            raw_goals=row.get('goals') or []
            if isinstance(raw_goals,dict):raw_goals=[raw_goals]
            goals=[canonical_point(g['position']) for g in raw_goals if g.get('position') is not None]
            length=sum(math.dist(a,b) for a,b in zip(points,points[1:])) if len(points)>1 else None
            result.append(dict(id=key,source_id=source_id,dataset='VLNVerse',version=digest[:16],format='vlnverse',
                split=split,role=granularity,language='en',scene=scene,scene_path=row['scene_id'],episode_id=eid,
                instruction_id=eid+'_natural',trajectory_id=str(row.get('trajectory_id',eid)),route_id=source_id+':'+eid,
                instruction=instruction,word_count=len(instruction.split()),reference_path=points,point_count=len(points),
                path_length=length,geodesic_distance=None,start_position=canonical_point(row['start_position']),
                start_rotation=row['start_rotation'],goals=goals,projection_axes=[0,2],has_media=0,media=[],
                issues=[] if points and goals else ['no_public_reference_trajectory'],
                source_path=str(path),source_file_sha256=digest,source_record=row,
                source_coordinate_frame='USD Z-up, metres; quaternion convention is resolved by collection adapter',
                geometry_transform={'source_to_canonical':[[1,0,0],[0,0,1],[0,-1,0]],'units':'metre'},
                has_reference_gt=bool(points and goals)))
    return result


def records(root):
    root=Path(root)
    files=sorted(root.glob('*.json.gz'))
    return load_records(str(root),tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in files))


def browse(root,scene_roots,params,limit=30,offset=0):
    rows=records(root)
    for key in ('split','role','scene','language'):
        if params.get(key):rows=[r for r in rows if r[key]==params[key]]
    q=params.get('q','').strip().lower()
    if q:rows=[r for r in rows if q in (r['instruction']+' '+r['episode_id']+' '+r['scene']).lower()]
    def available(record):
        try:usd_path(scene_roots,record['scene']);return True
        except (OSError,ValueError):return False
    if params.get('available')=='yes':rows=[r for r in rows if available(r)]
    items=[]
    for record in rows[offset:offset+limit]:
        item={k:v for k,v in record.items() if k not in ('source_record','source_path')}
        item['scene_available']=available(record)
        items.append(item)
    return {'items':items,'stats':{'records':len(rows),'scenes':len({r['scene'] for r in rows})},
            'facets':{},'index_ready':True,'limit':limit,'offset':offset}
