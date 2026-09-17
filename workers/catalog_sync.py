"""Publish validated GT runs into the original Atlas video collection."""
import fcntl
import json
from pathlib import Path
import tempfile
from sqlalchemy import select
from apps.api.db import Artifact, Run, session
from apps.api.settings import ROOT, settings
from catalog.vln_video_collection import build_collection


def sync_gt_catalog():
    with (settings.runtime_root / 'catalog-sync.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with session() as db:
            rows = db.execute(select(Artifact, Run).join(Run, Artifact.run_id == Run.id)
                              .where(Artifact.kind == 'dataset', Run.kind == 'gt')
                              .order_by(Run.created_at.desc())).all()
            sources = []
            for artifact, run in rows:
                dataset = Path(artifact.path)
                report = dataset / 'meta/platform_validation.json'
                if not report.exists() or not json.loads(report.read_text()).get('complete'):
                    continue
                cfg = run.config
                records = cfg.get('source_records', [])
                source_id = records[0]['source_id'] if records else {'r2r':'r2r-vlnce-v1-3','rxr':'rxr-vlnce-v0'}[cfg['dataset']]
                sources.append({'root':str(dataset),'source_id':source_id,
                    'dataset':{'r2r':'R2R','rxr':'RxR'}[cfg['dataset']], 'split':cfg['split'],'role':cfg['role']})
        if not sources:
            return {'matched_records':0}
        with tempfile.TemporaryDirectory(dir=settings.runtime_root) as scratch:
            candidate = Path(scratch) / 'vln-videos.sqlite'
            result = build_collection(settings.catalog_db, sources, candidate,
                settings.runtime_root/'gt-videos', ROOT/'visual/render_episode_gt.py', settings.lerobot_python)
            candidate.replace(settings.runtime_root/'vln-videos.sqlite')
        return result


if __name__ == '__main__':
    print(json.dumps(sync_gt_catalog(),ensure_ascii=False))
