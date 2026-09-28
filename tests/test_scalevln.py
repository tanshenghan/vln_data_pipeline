"""Platform selection/estimates keep absent HM3D scenes out of default jobs."""
import json

import pytest
from pydantic import ValidationError

from apps.api import capacity
from apps.api.catalog_service import browse, resolve_selection
from apps.api.schemas import RunRequest
from apps.api.settings import settings
from catalog.vln_catalog import build_database
from data_pipeline.tests.test_scalevln_source import prepared_fixture


@pytest.mark.parametrize("changes", [dict(split="val_seen"), dict(split="test"), dict(role="follower"),
    dict(languages=["en-US"]), dict(full_dataset=True), dict(kind="seedance"), dict(execution="cloud_idle")])
def test_scalevln_unsupported_modes_rejected(changes):
    with pytest.raises(ValidationError):
        RunRequest(dataset="scalevln", **changes)


def test_default_selection_skips_missing_scenes_explicit_selection_fails_preflight(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    datasets = shared / "datasets"
    datasets.mkdir(parents=True)
    root = prepared_fixture(datasets, monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(settings, "source_root", shared)
    monkeypatch.setattr(settings, "runtime_root", runtime)
    monkeypatch.setattr(settings, "resource_root", tmp_path / "personal")
    scene = settings.scenes_root / "hm3d/train/00000-example/example.basis.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"mesh")
    scene.with_suffix(".navmesh").write_bytes(b"navmesh")
    cfg_path = tmp_path / "catalog.json"
    cfg_path.write_text(json.dumps({"sources": [{"id": "scalevln-streamvln-150k", "dataset": "ScaleVLN",
        "format": "vlnce", "root": str(root), "splits": ["train"], "roles": ["guide"]}]}))
    build_database(cfg_path, settings.catalog_db)
    cfg = resolve_selection(RunRequest(dataset="scalevln").model_dump())
    assert cfg["episode_ids"] == ["1"]
    assert cfg["source_records"][0]["dataset"] == "ScaleVLN"
    estimate = capacity.estimate_gt(cfg)
    assert (estimate["episodes"], estimate["scenes"], estimate["frames"]) == (1, 1, 4)
    assert settings.runs_root.is_relative_to(tmp_path / "personal")
    with pytest.raises(ValueError, match="减少数量"):
        resolve_selection(RunRequest(dataset="scalevln", count=2).model_dump())
    missing = resolve_selection(RunRequest(dataset="scalevln", episode_ids=["2"]).model_dump())
    with pytest.raises(ValueError, match="场景或 navmesh"):
        capacity.estimate_gt(missing)
    rows = browse({"dataset": "ScaleVLN"})["items"]
    assert {r["episode_id"]: r["scene_ready"] for r in rows} == {"1": True, "2": False}
    assert not (shared / "runs").exists()
