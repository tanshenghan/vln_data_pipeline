"""Native object goals are not language instructions or terminal-ground GT."""
import gzip
import json

import pytest

from data_pipeline import objectnav_source as source
from data_pipeline.download_habitat_assets import PACKAGES, sha256


def fixture_source(tmp_path, dataset="hm3d_objectnav_v2"):
    entry = source.spec(dataset)
    root = source.dataset_root(tmp_path, dataset)
    path = root / "train/content/ABCDEFGHIJK.json.gz"
    path.parent.mkdir(parents=True)
    prefix = "hm3d" if entry["version"] == "v1" else "hm3d_v0.2"
    scene_id = f"{prefix}/train/00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"
    if entry["family"] == "mp3d":
        scene_id = "mp3d/ABCDEFGHIJK/ABCDEFGHIJK.glb"
    episode = {"episode_id": "0", "scene_id": scene_id, "object_category": "chair",
               "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
               "goals": [], "shortest_paths": None, "info": {"geodesic_distance": 3.0}}
    episodes = [episode, {**episode, "object_category": "bed"}, {**episode, "episode_id": "1"}]
    goals = {source.goal_key(e): [{"position": [1, 1, -3], "object_id": 42,
              "object_category": e["object_category"], "view_points": [
                  {"agent_state": {"position": [1, 0, -2], "rotation": [0, 0, 0, 1]}, "iou": .3}
                  for _ in range(4)]}] for e in episodes}
    with gzip.open(path, "wt") as handle:
        json.dump({"episodes": episodes, "goals_by_category": goals}, handle)
    installed = tmp_path / "installation_manifests/habitat"
    installed.mkdir(parents=True, exist_ok=True)
    package = f"objectnav_{entry['family']}_{entry['version']}.zip"
    (installed / (package + ".installed.json")).write_text(json.dumps({
        "complete": True, "url": PACKAGES[package], "archive_sha256": "a" * 64,
        "files": [{"path": str(path.relative_to(tmp_path)), "bytes": path.stat().st_size, "sha256": sha256(path)}]}))
    source.prepare(tmp_path, dataset)
    return path, scene_id


@pytest.mark.parametrize("dataset", list(source.DATASETS))
def test_all_three_versions_are_browsable_not_collectable(tmp_path, dataset):
    fixture_source(tmp_path, dataset)
    item = next(i for i in source.overview(tmp_path, tmp_path / "scene_datasets")["items"] if i["id"] == dataset)
    assert item["available"] and not item["can_collect"]
    assert item["splits"]["train"]["episodes"] == 3
    assert not item["splits"]["train"]["scenes"][0]["geometry_ready"]


def test_category_pagination_keeps_native_ids_and_file_ordinal(tmp_path):
    fixture_source(tmp_path)
    data = source.page(tmp_path, tmp_path / "scene_datasets", "hm3d_objectnav_v2", "train", "ABCDEFGHIJK", "chair", 1, 1)
    assert data["total"] == 2 and len(data["items"]) == 1
    assert data["items"][0]["source_row"] == 2
    assert data["items"][0]["source_episode_id"] == "1"
    assert "instruction" not in data["items"][0]
    first = source.detail(tmp_path, tmp_path / "scene_datasets", "hm3d_objectnav_v2", "train", "ABCDEFGHIJK", 0)
    duplicate_native_id = source.detail(tmp_path, tmp_path / "scene_datasets", "hm3d_objectnav_v2", "train", "ABCDEFGHIJK", 1)
    assert first["id"] != duplicate_native_id["id"]
    assert first["source_episode"]["goals"] == []
    assert first["goal_objects"][0]["view_point_count"] == 4
    assert len(first["goal_objects"][0]["view_point_preview"]) == 3
    assert not first["original_language_instructions"]
    assert "reference_path" not in first and "actions" not in first


def test_geometry_and_versioned_semantics_checked_independently(tmp_path):
    _, native_scene = fixture_source(tmp_path, "hm3d_objectnav_v1")
    scenes = tmp_path / "scene_datasets"
    geometry = scenes / source.scene_relative("hm3d_objectnav_v1", native_scene)
    geometry.parent.mkdir(parents=True)
    geometry.write_bytes(b"geometry")
    geometry.with_suffix(".navmesh").write_bytes(b"navmesh")
    # Wrong-version v0.2 semantics must not make v1 ready.
    other_geometry = scenes / "hm3d/train/00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"
    other_geometry.parent.mkdir(parents=True)
    for suffix in (".semantic.glb", ".semantic.txt"):
        other_geometry.with_name(other_geometry.name.replace(".basis.glb", suffix)).write_bytes(b"v0.2")
    state = source.asset_status(scenes, "hm3d_objectnav_v1", native_scene)
    assert state["geometry_ready"] and not state["semantic_assets_ready"]
    semantic = scenes / source.scene_relative("hm3d_objectnav_v1", native_scene, semantic=True)
    semantic.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".semantic.glb", ".semantic.txt"):
        semantic.with_name(semantic.name.replace(".basis.glb", suffix)).write_bytes(b"v0.1")
    state = source.asset_status(scenes, "hm3d_objectnav_v1", native_scene)
    assert state["semantic_assets_ready"] and not state["can_collect"]


def test_v2_minival_uses_val_assets_without_rewriting_original_scene():
    native = "hm3d_v0.2/minival/00800-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"
    assert str(source.scene_relative("hm3d_objectnav_v2", native)) == "hm3d/val/00800-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"


def test_v1_uses_legacy_geometry_even_without_semantic_rendering():
    native = "hm3d/train/00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"
    assert str(source.scene_relative("hm3d_objectnav_v1", native)) == "hm3d/versions/v0.1/train/00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"


@pytest.mark.parametrize("native", ["../a.glb", "/hm3d/a.glb", "hm3d/train/00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb",
    "hm3d_v0.2/train/../ABCDEFGHIJK.basis.glb", "hm3d_v0.2/train/00000-ABCDEFGHIJK/wrong.basis.glb"])
def test_wrong_version_and_unsafe_scene_paths_rejected(native):
    with pytest.raises(ValueError):
        source.scene_relative("hm3d_objectnav_v2", native)


@pytest.mark.parametrize("relative", ["../outside.json.gz", "/train/content/ABCDEFGHIJK.json.gz", "train/content/../x.json.gz"])
def test_arbitrary_files_cannot_be_read(tmp_path, relative):
    with pytest.raises(ValueError):
        source.content_path(tmp_path, relative)


def test_mutated_source_rejected_before_read(tmp_path):
    path, _ = fixture_source(tmp_path)
    with gzip.open(path, "wt") as handle:
        json.dump({"episodes": [], "goals_by_category": {}}, handle)
    with pytest.raises(ValueError, match="changed|hash"):
        source.page(tmp_path, tmp_path / "scene_datasets", "hm3d_objectnav_v2", "train", "ABCDEFGHIJK")


def test_missing_catalog_does_not_imply_readiness(tmp_path):
    assert not any(item["available"] for item in source.overview(tmp_path, tmp_path / "scenes")["items"])


def test_unknown_scene_invalid_offset_and_category_rejected(tmp_path):
    fixture_source(tmp_path)
    for changes in ({"scene": "../escape"}, {"category": "not-in-source"}, {"offset": -1}, {"limit": 101}):
        args = {"dataset": "hm3d_objectnav_v2", "split": "train", "scene": "ABCDEFGHIJK", **changes}
        with pytest.raises(ValueError):
            source.page(tmp_path, tmp_path / "scene_datasets", **args)
