"""The readiness marker is earned only after the entire acceptance chain passes."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from data_pipeline.verify_environment import verify_environment


@pytest.mark.parametrize("failure", [None, "capture", "export", "contract", "preview"])
def test_acceptance_only_marks_ready_after_all_real_stages(tmp_path, monkeypatch, failure):
    bundle = tmp_path / "shared/action_tokenizer"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    settings = SimpleNamespace(
        resource_root=tmp_path / "personal", runs_root=tmp_path / "personal/runs", runtime_root=tmp_path / "runtime",
        source_resource_root=tmp_path / "shared", scenes_root=tmp_path / "shared/scene_datasets",
        rvq_bundle=bundle, habitat_python=Path("/habitat/bin/python"), lerobot_python=Path("/lerobot/bin/python"),
        dataset_root=lambda dataset: tmp_path / "shared/datasets" / dataset,
    )
    cfg = {"dataset": "r2r", "split": "train", "role": "guide", "languages": ["all"], "episode_ids": ["7"]}
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"]
        report = json.loads((settings.runtime_root / "environment_validation.json").read_text())
        assert not report["habitat"] and not report["lerobot"]
        output = Path(report["directory"])
        if report["stage"] == failure:
            raise subprocess.CalledProcessError(1, command)
        if report["stage"] == "capture":
            (output / "capture").mkdir()
        elif report["stage"] == "independent_contract":
            assert not (output / "capture").exists()
            assert (output / "capture_archive").is_dir()
            path = output / "dataset/meta/platform_validation.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"complete": failure != "contract"}))
        elif report["stage"] == "preview":
            (output / "gt-review.mp4").write_bytes(b"test preview")

    monkeypatch.setattr(subprocess, "run", run)
    if failure:
        with pytest.raises((subprocess.CalledProcessError, ValueError)):
            verify_environment(settings, cfg, {})
    else:
        verify_environment(settings, cfg, {})
        assert len(commands) == 5
    report = json.loads((settings.runtime_root / "environment_validation.json").read_text())
    assert report["complete"] is (failure is None)
    assert report["habitat"] is (failure is None)
    assert report["lerobot"] is (failure is None)
    assert report["status"] == ("succeeded" if failure is None else "failed")
    assert json.loads((Path(report["directory"]) / "environment_validation.json").read_text()) == report
    assert not (settings.source_resource_root / "validation").exists()
