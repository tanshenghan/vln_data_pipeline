"""Accept a workstation only after real capture, export and independent readback."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_environment(settings, cfg, environment):
    """Keep every stage's evidence; never infer readiness from imports alone."""
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    with (settings.runtime_root / "environment_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _verify(settings, cfg, environment)


def _verify(settings, cfg, environment):
    output = settings.runs_root / "_validation" / (cfg["dataset"] + "-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    report_path = settings.runtime_root / ("scalevln_environment_validation.json"
        if cfg["dataset"] == "scalevln" else "environment_validation.json")
    report = {
        "complete": False, "habitat": False, "lerobot": False, "status": "running",
        "directory": str(output), "dataset": cfg["dataset"], "episode_ids": cfg["episode_ids"],
        "habitat_python": str(settings.habitat_python), "lerobot_python": str(settings.lerobot_python),
        "source_root": str(settings.source_resource_root), "scenes_root": str(settings.scenes_root),
        "rvq_bundle": str(settings.rvq_bundle), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(report_path, report)
    env = {**os.environ, **environment, "BETA_DATA_ROOT": str(settings.dataset_root(cfg["dataset"]))}

    def execute(stage, argv, gpu=False):
        report["stage"] = stage
        atomic_json(report_path, report)
        command = list(map(str, argv))
        atomic_json(output / (stage + ".command.json"), {"argv": command, "cwd": str(ROOT)})
        print(json.dumps({"stage": stage, "directory": str(output)}), flush=True)
        with (output / (stage + ".log")).open("w") as log:
            if gpu:
                with (settings.runtime_root / "gpu.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        manifest = settings.rvq_bundle / "manifest.json"
        report["rvq_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        atomic_json(output / "config.json", cfg)
        ids = output / "episode_ids.json"
        atomic_json(ids, cfg["episode_ids"])
        capture, dataset = output / "capture", output / "dataset"
        execute("capture", ["bash", "data_pipeline/run.sh", capture, "-1", "--dataset", cfg["dataset"],
            "--split", cfg["split"], "--role", cfg["role"], "--languages", *cfg["languages"],
            "--episode-ids-file", ids], gpu=True)
        execute("validate_capture", [settings.lerobot_python, "data_pipeline/validate_r2r.py", capture])
        execute("export", [settings.lerobot_python, "data_pipeline/export_lerobot.py", capture,
            "--output-dir", dataset, "--repo-id", "beta-nav/" + output.name])
        # The independent official loader must work without the original capture path.
        capture.rename(output / "capture_archive")
        execute("independent_contract", [settings.lerobot_python, "-m", "contracts.check", dataset,
            "--report", dataset / "meta/platform_validation.json"])
        validation = json.loads((dataset / "meta/platform_validation.json").read_text())
        if validation.get("complete") is not True:
            raise ValueError("independent validator did not report completion")
        execute("preview", [settings.lerobot_python, "visual/render_episode_gt.py", dataset,
            "--episode-id", cfg["episode_ids"][0], "--output", output / "gt-review.mp4",
            "--fps", "8", "--scale", "1"])
        if not (output / "gt-review.mp4").is_file() or not (output / "gt-review.mp4").stat().st_size:
            raise ValueError("preview did not produce a nonempty video")
        report.update(complete=True, habitat=True, lerobot=True, status="succeeded", stage="complete",
                      validation=validation, finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        report.update(status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        atomic_json(output / "environment_validation.json", report)
        atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["r2r", "rxr", "scalevln"], default="r2r")
    parser.add_argument("--split", choices=["train", "val_seen", "val_unseen"], default="train")
    parser.add_argument("--role", choices=["guide", "follower"], default="guide")
    parser.add_argument("--languages", nargs="+", choices=["all", "en-US", "en-IN", "hi-IN", "te-IN"], default=["all"])
    parser.add_argument("--episode-id", help="One public expert episode; defaults to the first catalog match")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from apps.api.catalog_service import resolve_selection
    from apps.api.schemas import RunRequest
    from apps.api.settings import settings
    from workers.tasks import common_env

    request = RunRequest(dataset=args.dataset, split=args.split, role=args.role, languages=args.languages,
                         count=1, episode_ids=[args.episode_id] if args.episode_id else [])
    cfg = resolve_selection(request.model_dump())
    print(json.dumps(verify_environment(settings, cfg, common_env()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
