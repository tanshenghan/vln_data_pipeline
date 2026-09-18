"""Install the workstation's persistent full VLNVerse scene synchronization."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apps.api.settings import settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-info',type=Path,default=ROOT/'runtime/vlnverse_repo_info.json')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    target=settings.vlnverse_managed_scene_root
    if target is None:parser.error('configure VLN_VLNVERSE_MANAGED_SCENE_ROOT first')
    if not args.repo_info.is_file():parser.error('provide the HF repository revision metadata JSON')
    command=[sys.executable,str(ROOT/'configs/sync_vlnverse.py'),'--root',str(target),
             '--state',str(settings.runtime_root/'vlnverse_sync'),'--repo-info',str(args.repo_info.resolve()),
             '--workers',str(args.workers)]
    for path in (settings.vlnverse_scene_root,settings.resource_root/'scene_datasets/vlnverse'):
        if path and path.is_dir():command+=['--reuse',str(path)]
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    (units/'vln-vlnverse-sync.service').write_text(f'''[Unit]
Description=VLNVerse complete scene mirror with hash verification
After=network-online.target

[Service]
Type=simple
WorkingDirectory={ROOT}
ExecStart={shlex.join(command)}
Restart=on-failure
RestartSec=30
Nice=10

[Install]
WantedBy=default.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now','vln-vlnverse-sync'],check=True)


if __name__=='__main__':main()
