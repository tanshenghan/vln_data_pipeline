import json

import pytest
from pydantic import ValidationError

from apps.api.catalog_service import browse, resolve_selection
from apps.api.schemas import RunRequest, CollectionPlanRequest
from apps.api.settings import settings
from catalog.vln_catalog import build_database, connect
from configs.build_source_catalog import source_configuration
from data_pipeline import srdf_source
from data_pipeline.tests.test_srdf_source import prepared_fixture, native_row


def test_source_registry_catalog_and_shared_paths(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    datasets = shared / "datasets"
    datasets.mkdir(parents=True)
    prepared_fixture(datasets, monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(settings, "source_root", shared)
    monkeypatch.setattr(settings, "runtime_root", runtime)
    monkeypatch.setattr(settings, "resource_root", tmp_path / "personal")
    source = next(s for s in source_configuration()["sources"] if s["dataset"] == "SRDF")
    assert source["id"] == srdf_source.SOURCE_ID and source["format"] == "r2r"
    config = tmp_path / "catalog.json"
    config.write_text(json.dumps({"sources": [source]}))
    build_database(config, settings.catalog_db)
    result = browse({"dataset": "SRDF", "split": "train", "language": "en"}, limit=1)
    assert result["stats"]["records"] == 2 and len(result["items"]) == 1
    assert not result["items"][0]["can_collect"]
    with connect(settings.catalog_db) as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT detail FROM episodes ORDER BY instruction_id")]
    assert [r["instruction_id"] for r in rows] == native_row()["instr_ids"]
    assert rows[0]["reference_path"] == [[0, 2, 1.5], [1, 2, 1.5]]
    assert rows[0]["projection_axes"] == [0, 1]  # Original graph frame, not Habitat X/Z.
    assert not rows[0]["issues"] and rows[0]["start_rotation"] is None
    assert settings.runs_root.is_relative_to(tmp_path / "personal")
    assert not (shared / "runs").exists()


@pytest.mark.parametrize("changes", [{}, {"kind": "seedance"}, {"execution": "cloud_idle"},
                                     {"full_dataset": True}, {"split": "val_unseen"}])
def test_discrete_source_cannot_submit_capture(changes):
    with pytest.raises(ValidationError, match="离散路标"):
        RunRequest(dataset="srdf", **changes)
    with pytest.raises(ValueError, match="离散路标"):
        resolve_selection({"dataset": "srdf"})


def test_discrete_source_cannot_create_continuous_plan():
    with pytest.raises(ValidationError, match="离散路标"):
        CollectionPlanRequest(dataset="srdf", compute_pools=["local"], execution="local")
