"""Require an explicit capture completion marker even when Kit exits with code 0."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    python, *args = sys.argv[1:]
    output = Path(args[args.index('--output') + 1])
    result = subprocess.run([python, 'data_pipeline/collect_vlnverse.py', *args])
    marker = output / 'render_complete.json'
    if result.returncode:
        raise SystemExit(result.returncode)
    if not marker.is_file() or not json.loads(marker.read_text()).get('complete'):
        failure = output / 'failure.json'
        detail = json.loads(failure.read_text()).get('error') if failure.exists() else 'renderer did not finish'
        raise RuntimeError('Native USD capture incomplete: ' + str(detail))


if __name__ == '__main__':main()
