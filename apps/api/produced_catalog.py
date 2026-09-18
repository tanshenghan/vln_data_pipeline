"""Federated production catalog used by the existing Atlas browsing layout."""
import json
from sqlalchemy import select, or_, func
from .db import DataSample, Artifact, Run, Release, Review, session, serialize


def public_sample(row, artifacts=None):
    result=serialize(row)
    result.update(row.details)
    gt = row.origin=='gt' and not row.details.get('candidate',False)
    result.update(id=row.id, source_id=row.run_id, version=row.run_id,
                  origin=row.origin, source_record_id=row.source_record_id,
                  ground_truth=gt, has_gt=gt,
                  has_gt_video=gt and bool(row.details.get('video_artifact_id')),
                  has_media=bool(row.details.get('video_artifact_id')),
                  path_length=row.details.get('path_length'))
    if artifacts is not None:
        result['artifacts']=[serialize(a) for a in artifacts]
    return result


def conditions(params):
    filters=[]
    origin=params.get('origin','produced')
    if origin in {'gt','seedance'}:filters.append(DataSample.origin==origin)
    for name in ('dataset','split','role','language','scene','quality','source_record_id'):
        value=params.get(name)
        if not value:continue
        if name=='quality':
            if value=='complete':filters.append(DataSample.quality.in_(['accepted','validated']))
            elif value=='issues':filters.append(DataSample.quality.not_in(['accepted','validated']))
            else:filters.append(DataSample.quality==value)
        else:filters.append(getattr(DataSample,name)==value)
    if params.get('source_id'):filters.append(DataSample.run_id==params['source_id'])
    if params.get('q'):
        q='%'+params['q'].replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
        filters.append(or_(*[column.ilike(q,escape='\\') for column in
                            (DataSample.instruction,DataSample.episode_id,DataSample.scene,DataSample.run_id)]))
    media = DataSample.details['video_artifact_id'].as_string().is_not(None)
    if params.get('media')=='yes':filters.append(media)
    elif params.get('media')=='no':filters.append(~media)
    length = DataSample.details['path_length'].as_float()
    if params.get('min_length'):filters.append(length>=float(params['min_length']))
    if params.get('max_length'):filters.append(length<=float(params['max_length']))
    return filters


def browse(params,limit=40,offset=0):
    with session() as db:
        filters=conditions(params)
        rows=db.scalars(select(DataSample).where(*filters).order_by(DataSample.created_at.desc(),DataSample.id).limit(limit).offset(offset)).all()
        facets={}
        for field in ('dataset','split','role','language','scene','run_id'):
            column=getattr(DataSample,field)
            counts=db.execute(select(column,func.count()).where(*filters).group_by(column).order_by(column)).all()
            facets['source_id' if field=='run_id' else field]=[{'value':k,'count':v} for k,v in counts]
        summary=db.execute(select(func.count(),func.count(func.distinct(DataSample.details['route_id'].as_string())),
            func.count(func.distinct(DataSample.scene)),func.avg(DataSample.details['path_length'].as_float())).where(*filters)).one()
        media=db.scalar(select(func.count()).select_from(DataSample).where(*filters,DataSample.details['video_artifact_id'].as_string().is_not(None)))
        stats=dict(zip(('records','routes','scenes','mean_path_length'),summary));stats['media']=media
        tree=[dict(dataset=d,split=s,scene=c,count=n) for d,s,c,n in db.execute(select(
            DataSample.dataset,DataSample.split,DataSample.scene,func.count()).where(*filters).group_by(
                DataSample.dataset,DataSample.split,DataSample.scene))]
        return {'items':[public_sample(r) for r in rows],'stats':stats,'facets':facets,
                'offset':offset,'limit':limit,'origin':params.get('origin','produced'),'tree':tree}


def detail(sample_id):
    with session() as db:
        row=db.get(DataSample,sample_id)
        if not row:return None
        ids=row.details.get('artifact_ids',[])
        artifacts=db.scalars(select(Artifact).where(Artifact.id.in_(ids))).all() if ids else []
        result=public_sample(row,artifacts)
        result['run']=serialize(db.get(Run,row.run_id))
        result['reviews']=[serialize(r) for r in db.scalars(select(Review).where(Review.subject_id==row.run_id).order_by(Review.created_at.desc()))]
        release=db.scalar(select(Release).where(Release.run_id==row.run_id))
        result['release']=serialize(release) if release else None
        return result


def derived(source_id):
    with session() as db:
        return [public_sample(r) for r in db.scalars(select(DataSample).where(
            DataSample.source_record_id==source_id).order_by(DataSample.created_at.desc()))]


def export_records(params):
    with session() as db:
        rows=db.scalars(select(DataSample).where(*conditions(params)).order_by(DataSample.created_at,DataSample.id).execution_options(yield_per=1000))
        yield '{"filters":'+json.dumps(params,ensure_ascii=False)+',"samples":['
        first=True
        for row in rows:
            value=public_sample(row)
            if not first:yield ','
            first=False
            yield json.dumps(value,ensure_ascii=False)
        yield ']}'
