"""Real workstation acceptance; never mark Isaac ready from import checks alone."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apps.api.settings import settings
from apps.api.catalog_service import resolve_selection
from apps.api.schemas import RunRequest
from workers.command_runner import atomic_json
from workers.tasks import common_env


def main():
    report_path=settings.runtime_root/'vlnverse_environment_validation.json'
    output=settings.runs_root/'_validation'/('vlnverse-'+uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    report={'complete':False,'status':'running','directory':str(output),'python':str(settings.isaac_python),
            'started_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(report_path,report)
    env={**os.environ,**common_env(),'OMNI_KIT_ACCEPT_EULA':'YES'}
    def execute(stage,argv,gpu=False):
        report['stage']=stage;atomic_json(report_path,report)
        with (output/(stage+'.log')).open('w') as log:
            if gpu:
                with (settings.runtime_root/'gpu.lock').open('w') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=resolve_selection(RunRequest(dataset='vlnverse',split='val_seen',role='coarse',episode_ids=['kujiale_0254_32_2']).model_dump())
        config=output/'config.json';atomic_json(config,cfg)
        capture=output/'capture';dataset=output/'dataset'
        roots=[arg for p in settings.vlnverse_scene_roots if p for arg in ['--scene-root',p]]
        execute('capture',[sys.executable,'-m','workers.vlnverse_capture',settings.isaac_python,'--config',config,
            '--output',capture,'--rvq-bundle',settings.rvq_bundle,*roots],gpu=True)
        execute('annotate',[settings.lerobot_python,'data_pipeline/collect_vlnverse.py','--output',capture,'--annotate'])
        execute('validate_capture',[settings.lerobot_python,'data_pipeline/validate_r2r.py',capture])
        execute('export',[settings.lerobot_python,'data_pipeline/export_lerobot.py',capture,
            '--output-dir',dataset,'--repo-id','beta-nav/vlnverse-environment-acceptance'])
        capture.rename(output/'capture_archive')
        execute('independent_contract',[settings.lerobot_python,'-m','contracts.check',dataset,
            '--report',dataset/'meta/platform_validation.json'])
        execute('preview',[settings.lerobot_python,'visual/render_episode_gt.py',dataset,
            '--episode-id','kujiale_0254_32_2','--output',output/'gt-review.mp4','--fps','8','--scale','1'])
        validation=json.loads((dataset/'meta/platform_validation.json').read_text())
        if not validation.get('complete'):raise ValueError('independent validator did not report completion')
        report.update(complete=True,status='succeeded',stage='complete',validation=validation,
                      finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(report_path,report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));atomic_json(report_path,report)
        raise


if __name__=='__main__':main()
