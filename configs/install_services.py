"""Install dedicated user services. Run after dependencies and source data are ready."""
from pathlib import Path
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UNITS = Path.home() / ".config/systemd/user"
UNITS.mkdir(parents=True, exist_ok=True)
(ROOT / "runtime").mkdir(exist_ok=True)
commands = {
    "vln-api": f"{ROOT}/.venv/bin/python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8800",
    "vln-worker": f"{ROOT}/.venv/bin/celery -A workers.celery_app:celery worker --pool=solo --concurrency=1 --loglevel=INFO --hostname=vln-workstation@%H",
    "vln-beat": f"{ROOT}/.venv/bin/celery -A workers.celery_app:celery beat --loglevel=INFO --schedule={ROOT}/runtime/celerybeat",
    "vln-cloud": f"{ROOT}/.venv/bin/python -m workers.cloud_scheduler",
    "vln-atlas": f"{ROOT}/.venv/bin/python -m catalog.serve --vln-db {ROOT}/runtime/vln.sqlite --host 127.0.0.1 --port 8788",
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", nargs="+", choices=list(commands), help="Install only the named services; preserve other deployment units")
args = parser.parse_args()
if args.only:
    commands = {name: commands[name] for name in args.only}
for name, command in commands.items():
    # Solo worker's detached command runners keep running across worker restarts.
    kill = "process" if name == "vln-worker" else "control-group"
    (UNITS / f"{name}.service").write_text(f"""[Unit]
Description=VLN Data Pipeline {name}
After=network-online.target

[Service]
Type=simple
WorkingDirectory={ROOT}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH={ROOT}:{ROOT}/src
ExecStart={command}
Restart=always
RestartSec=5
KillMode={kill}
TimeoutStopSec=15

[Install]
WantedBy=default.target
""")
subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
subprocess.run(["systemctl", "--user", "enable", "--now", *[n + ".service" for n in commands]], check=True)
print("Services installed; inspect systemctl --user status " + " ".join(commands))
