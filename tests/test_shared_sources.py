"""Shared annotation/scene inputs must not redirect run outputs or indexes."""
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api import capacity, db
from apps.api.catalog_service import resolve_selection
from apps.api.preflight import readable_file
from apps.api.schemas import RunRequest
from apps.api.settings import Settings, settings
from catalog.vln_catalog import build_database
from configs.build_source_catalog import source_configuration
from workers.tasks import common_env, run_dir


@pytest.fixture
def private_database(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "platform.sqlite"))
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "Session", sessionmaker(engine, expire_on_commit=False))
    yield
    engine.dispose()


def test_original_deployment_layout_remains_default(tmp_path):
    config = Settings(_env_file=None, resource_root=tmp_path, source_root=None)
    assert config.dataset_root("r2r") == tmp_path / "datasets/R2R_VLNCE_v1-3_preprocessed"
    assert config.dataset_root("rxr") == tmp_path / "datasets/RxR_VLNCE_v0"
    assert config.scenes_root == tmp_path / "scene_datasets"
    assert config.rvq_bundle == tmp_path / "action_tokenizer"
    assert config.runs_root == tmp_path / "runs"


def test_shared_source_selection_capacity_and_worker_keep_outputs_private(tmp_path, monkeypatch, private_database):
    shared = tmp_path / "shared"
    private = tmp_path / "personal"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(settings, "source_root", shared)
    monkeypatch.setattr(settings, "rvq_bundle_root", None)
    monkeypatch.setattr(settings, "resource_root", private)
    monkeypatch.setattr(settings, "runtime_root", runtime)
    monkeypatch.setattr(settings, "vlnverse_managed_scene_root", None)
    monkeypatch.setattr(settings, "vlnverse_scene_root", tmp_path / "shared-kujiale")
    root = settings.dataset_root("r2r") / "train"
    root.mkdir(parents=True)
    episode = {"episode_id": 7, "scene_id": "mp3d/example/example.glb",
               "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
               "instruction": {"instruction_text": "Walk forward."},
               "reference_path": [[0, 0, 0], [0, 0, -0.25]],
               "goals": [{"position": [0, 0, -0.25]}]}
    for name, data in [("train.json.gz", {"episodes": [episode]}),
                       ("train_gt.json.gz", {"7": {"actions": [1, 0]}})]:
        with gzip.open(root / name, "wt") as handle:
            json.dump(data, handle)
    scene = settings.scenes_root / "mp3d/example/example.glb"
    scene.parent.mkdir(parents=True)
    scene.touch()
    scene.with_suffix(".navmesh").touch()
    shared_files = [p for p in shared.rglob("*") if p.is_file()]
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in shared_files}

    # Indexing and selection read the shared source; neither creates output in it.
    config = source_configuration()
    assert [s["root"] for s in config["sources"]] == [str(settings.dataset_root(d)) for d in ("r2r", "rxr")]
    config["sources"] = [{**config["sources"][0], "splits": ["train"]}]
    config_path = runtime / "sources.local.json"
    config_path.write_text(json.dumps(config))
    build_database(config_path, settings.catalog_db)
    cfg = resolve_selection(RunRequest().model_dump())
    assert cfg["episode_ids"] == ["7"]
    estimate = capacity.estimate_gt(cfg)
    assert (estimate["episodes"], estimate["scenes"], estimate["frames"]) == (1, 1, 2)
    assert settings.catalog_db.parent == runtime
    assert run_dir("test-run") == private / "runs/test-run"
    assert (private / "runs").is_dir()
    assert not (shared / "runs").exists()
    assert not (private / "datasets").exists()
    assert {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in shared_files} == before

    env = common_env()
    assert env["BETA_SCENES_DIR"] == str(shared / "scene_datasets")
    assert env["BETA_RVQ_BUNDLE"] == str(shared / "action_tokenizer")
    assert env["BETA_VLNVERSE_HDR"] == str(shared / "render_assets/limpopo_golf_course_4k.hdr")
    assert settings.vlnverse_scene_roots == [None, tmp_path / "shared-kujiale", shared / "scene_datasets/vlnverse"]


def test_readonly_model_bundle_override_keeps_source_and_output_roots(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    private = tmp_path / "personal"
    bundle = tmp_path / "model/action_tokenizer"
    monkeypatch.setattr(settings, "source_root", shared)
    monkeypatch.setattr(settings, "resource_root", private)
    monkeypatch.setattr(settings, "rvq_bundle_root", bundle)
    assert settings.rvq_bundle == bundle
    assert settings.dataset_root("r2r") == shared / "datasets/R2R_VLNCE_v1-3_preprocessed"
    assert settings.scenes_root == shared / "scene_datasets"
    assert settings.runs_root == private / "runs"
    assert common_env()["BETA_RVQ_BUNDLE"] == str(bundle)
    assert common_env()["PYTHONDONTWRITEBYTECODE"] == "1"


def test_unreadable_shared_asset_is_not_ready_and_does_not_crash_health(tmp_path):
    path = tmp_path / "manifest.json"
    assert not readable_file(path)
    path.write_text("{}")
    assert readable_file(path)
    denied = Mock(spec=Path)
    denied.open.side_effect = PermissionError("shared asset is private")
    assert not readable_file(denied)
