"""ScaleVLN joins/STOP semantics and shared-scene safety, without GPU rendering."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from data_pipeline import scalevln_source as source
from data_pipeline.vlnce_source import load_source, source_paths
from data_pipeline import collect_r2r as collector
from catalog.vln_ground_truth import normalize_gt


def example(eid=1, scene="00000-example"):
    name = scene.split("-", 1)[1]
    episode = {"episode_id": eid, "trajectory_id": f"scalevln_{scene}_{eid}",
        "scene_id": f"hm3d/{scene}/{name}.basis.glb",
        "start_position": [0., 0., 0.], "start_rotation": [0., 0., 0., 1.],
        "goals": [{"position": [0., 0., -.5], "radius": .25}],
        "instruction": {"instruction_text": "Walk to the door and stop."},
        "reference_path": [[0., 0., 0.], [0., 0., -.5]]}
    annotation = {"id": eid, "video": f"images/{scene}_scalevln_{eid:06d}",
                  "instructions": [episode["instruction"]["instruction_text"]], "actions": [-1, 1, 2, 1]}
    return episode, annotation


def prepared_fixture(tmp_path, monkeypatch):
    originals = [example(), example(2, "00001-missing")]
    episodes, annotations = tmp_path / "episodes.json.gz", tmp_path / "annotations.json"
    with gzip.open(episodes, "wt") as handle:
        json.dump({"episodes": [e for e, _ in originals]}, handle)
    annotations.write_text(json.dumps([a for _, a in originals]))
    monkeypatch.setattr(source, "SOURCE_HASHES", {
        "scalevln_subset_150k.json.gz": source.sha256(episodes), "annotations.json": source.sha256(annotations)})
    root = tmp_path / "ScaleVLN_StreamVLN"
    source.prepare_source(episodes, annotations, root)
    return root


def test_normalization_preserves_identity_motion_and_stop_provenance():
    episode, annotation = example()
    before = copy.deepcopy((episode, annotation))
    normalized, gt = source.normalize_episode(episode, annotation)
    assert (episode, annotation) == before
    assert normalized["scene_id"] == "hm3d/train/00000-example/example.basis.glb"
    assert gt["actions"] == [1, 2, 1, 0]  # no duplicated turns or leading fake STOP
    assert gt["locations"] == episode["reference_path"]
    assert gt["locations_source"] == "source_reference_path"
    meta = normalized["source_metadata"]
    assert meta["stop_convention"] == source.STOP_CONVENTION
    assert meta["synthetic_instruction"] and meta["replay_frame_count"] == 4


def test_sparse_reference_path_is_never_indexed_as_dense_ground_truth():
    episode, gt = source.normalize_episode(*example())
    # Even an accidental count match must not imply source trajectory GT.
    gt["actions"] = [1, 0]
    with pytest.raises(ValueError, match="reference_path_only"):
        normalize_gt(gt, episode["start_position"])


@pytest.mark.parametrize("actions", [[], [-1], [1, 2, 1], [-1, 0, 1], [-1, 1, -1], [-1, 4], [-1, True]])
def test_never_interpret_unknown_or_truncated_encoding_as_stop(actions):
    with pytest.raises(ValueError):
        source.replay_actions(actions)


@pytest.mark.parametrize("scene", ["../a.glb", "/hm3d/a.glb", "hm3d/../x/x.basis.glb",
    "mp3d/00000-example/example.glb", "hm3d/val/00000-example/example.basis.glb",
    "hm3d/00000-example/wrong.basis.glb"])
def test_reject_wrong_scene_family_and_traversal(scene):
    with pytest.raises(ValueError):
        source.scene_relative(scene)


@pytest.mark.parametrize("field,value", [("id", 9), ("instructions", ["A different instruction"]), ("video", "images/wrong")])
def test_join_rejects_matching_id_with_wrong_instruction_or_scene(field, value):
    episode, annotation = example()
    annotation[field] = value
    with pytest.raises(ValueError):
        source.normalize_episode(episode, annotation)


def test_prepare_is_complete_pinned_immutable_and_train_only(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    metadata = source.prepared_manifest(root, verify_hashes=True)
    assert metadata["episodes"] == 2 and metadata["frames"] == 8
    episodes, gt, selection = load_source(root, "scalevln", episode_ids=["1"])
    assert len(episodes) == 1 and set(gt) == {"1"}
    assert selection["source_episode_count"] == 2
    assert selection["preparation"]["revision"] == source.REVISION
    assert gt["1"]["actions"] == [1, 2, 1, 0]
    with pytest.raises(ValueError, match="only train"):
        source_paths(root, "scalevln", "val_seen")
    with pytest.raises(ValueError, match="follower"):
        source_paths(root, "scalevln", role="follower")
    with pytest.raises(ValueError, match="only supported for RxR"):
        load_source(root, "scalevln", languages=["en-US"])
    (root / "train/train_gt.json.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_source(root, "scalevln", episode_ids=["1"])


def test_unpinned_inputs_do_not_create_output(tmp_path):
    episodes = tmp_path / "episodes.gz"
    annotations = tmp_path / "actions.json"
    episodes.write_bytes(b"not the pinned release")
    annotations.write_text("[]")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="SHA-256"):
        source.prepare_source(episodes, annotations, output)
    assert not output.exists()


def test_directory_presence_does_not_claim_full_scene_coverage(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    scenes = tmp_path / "shared-scenes"
    mesh = scenes / "hm3d/train/00000-example/example.basis.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    assert not source.availability(root, scenes)["can_collect"]
    mesh.with_suffix(".navmesh").write_bytes(b"navigation mesh")
    before = {p: p.read_bytes() for p in scenes.rglob("*") if p.is_file()}
    ready = source.availability(root, scenes)
    assert ready["ready_episodes"] == 1 and ready["total_episodes"] == 2
    assert ready["ready_scenes"] == 1 and not ready["full_scene_coverage"]
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("distance,success", [(.2, True), (.5, False), (1.0, False), (float("inf"), False)])
def test_restored_stop_requires_verified_simulator_goal(tmp_path, monkeypatch, distance, success):
    episode, gt = source.normalize_episode(*example())
    sim = Mock()
    sim.geodesic_distance.return_value = distance
    monkeypatch.setattr(collector, "capture", lambda *args: {"world_from_body": np.eye(4).tolist()})
    export = Mock(return_value={"frames": 4})
    monkeypatch.setattr(collector, "export_targets", export)
    if success:
        result = collector.collect_episode(sim, episode, gt, tmp_path / "capture", {}, None)
        assert result["terminal_command"]["goal_verified"]
        export.assert_called_once()
    else:
        with pytest.raises(ValueError):
            collector.collect_episode(sim, episode, gt, tmp_path / "capture", {}, None)
        export.assert_not_called()
