"""ObjectNav read-only API does not mutate platform jobs or enable fake GT."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from apps.api.main import app
from apps.api.settings import settings
from apps.api.schemas import RunRequest
from data_pipeline.tests.test_objectnav_source import fixture_source


def test_objectnav_native_source_api(tmp_path, monkeypatch):
    fixture_source(tmp_path)
    monkeypatch.setattr(settings, "source_root", tmp_path)
    client = TestClient(app)
    response = client.get("/api/objectnav/datasets")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    params = {"dataset": "hm3d_objectnav_v2", "split": "train", "scene": "ABCDEFGHIJK"}
    page = client.get("/api/objectnav/episodes", params=params)
    assert page.status_code == 200 and page.json()["total"] == 3
    detail = client.get("/api/objectnav/episode", params={**params, "source_row": 0})
    assert detail.status_code == 200 and not detail.json()["can_collect"]
    assert client.get("/api/objectnav/episode", params={**params, "source_row": -1}).status_code == 422
    assert client.get("/api/objectnav/episodes", params={**params, "dataset": "../r2r"}).status_code == 422


@pytest.mark.parametrize("dataset", ["mp3d_objectnav_v1", "hm3d_objectnav_v1", "hm3d_objectnav_v2"])
def test_native_goals_cannot_be_submitted_as_vln_expert_replay(dataset):
    with pytest.raises(ValidationError):
        RunRequest(dataset=dataset)
