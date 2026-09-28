"""CPU checks for the isolated native probe; they do not imply GPU readiness."""
import gzip
import json
from pathlib import Path

import pytest

from data_pipeline.probe_evt_bench import prepare_workspace, scene_relative, select_episode, sha256


@pytest.mark.parametrize("value", ["/mp3d/a.glb", "mp3d/../private.glb", "hm3d/a/../../x.glb", "other/x.glb", "mp3d/a.json"])
def test_reject_unsafe_or_unknown_scene_ids(value):
    with pytest.raises(ValueError):
        scene_relative(value)


def test_prefix_and_hm3d_layout():
    assert scene_relative("data/scene_datasets/hm3d/train/001-a/a.basis.glb") == Path("hm3d/train/001-a/a.basis.glb")


def test_select_only_readable_requested_scene_and_id(tmp_path):
    scenes = tmp_path / "scenes"
    scene = scenes / "mp3d/a/a.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"test-only mesh")
    source = tmp_path / "train.json.gz"
    episodes = [
        {"episode_id": "missing", "scene_id": "hm3d/train/b/b.basis.glb"},
        {"episode_id": "7", "scene_id": "mp3d/a/a.glb", "info": {"robot_position": [1, 2, 3]}},
    ]
    with gzip.open(source, "wt") as stream:
        json.dump({"episodes": episodes}, stream)
    selected, relative = select_episode(source, scenes, "mp3d")
    assert selected == episodes[1]
    assert relative == Path("mp3d/a/a.glb")
    with pytest.raises(ValueError, match="No readable"):
        select_episode(source, scenes, "hm3d")
    with pytest.raises(ValueError, match="No readable"):
        select_episode(source, scenes, "mp3d", "missing")


def test_native_cache_cannot_write_into_shared_scene_root(tmp_path):
    upstream, humans, scenes = [tmp_path / name for name in ("upstream", "humans", "scenes")]
    relative = Path("mp3d/a/a.glb")
    scene = scenes / relative
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"immutable input")
    original_hash = sha256(scene)
    output = tmp_path / "new-probe"
    work = prepare_workspace(output, upstream, humans, scenes, relative)
    staged = work / "data/scene_datasets" / relative
    assert staged.is_symlink() and staged.resolve() == scene
    cache = work / "data/scene_datasets/navmeshes"
    cache.mkdir()
    (cache / "a.navmesh").write_bytes(b"new per-probe native cache")
    assert not (scenes / "navmeshes").exists()
    assert sha256(scene) == original_hash
    with pytest.raises(FileExistsError):
        prepare_workspace(output, upstream, humans, scenes, relative)
