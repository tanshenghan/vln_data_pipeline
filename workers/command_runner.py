"""A durable subprocess boundary; exit status survives a Celery/API restart."""
import json
import fcntl
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def identity(pid):
    try:
        # Parse after comm, which can contain spaces or parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, IndexError):
        return None


def main():
    directory = Path(sys.argv[1])
    spec = json.loads((directory / "command.json").read_text())
    atomic_json(directory / "process.json", {"pid": os.getpid(), "start": identity(os.getpid()), "started": time.time()})
    env = dict(os.environ, **spec.get("env", {}))
    child = None
    def stop(signum, frame):
        if child is not None:
            child.terminate()
        else:
            raise InterruptedError("cancelled while waiting for resources")
    signal.signal(signal.SIGTERM, stop)
    code = 1
    with (directory / "output.log").open("ab", buffering=0) as log:
        try:
            with ExitStack() as resources:
                if spec.get("gpu_lock"):
                    lock = resources.enter_context(Path(spec["gpu_lock"]).open("w"))
                    log.write(b'{"stage":"waiting_gpu"}\n')
                    fcntl.flock(lock, fcntl.LOCK_EX)
                child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
        except Exception as exc:
            log.write((f"Runner error: {type(exc).__name__}: {exc}\n").encode())
    atomic_json(directory / "exit.json", {"returncode": code, "finished": time.time()})


if __name__ == "__main__":
    main()
