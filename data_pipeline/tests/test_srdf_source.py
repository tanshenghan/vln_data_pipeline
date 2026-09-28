"""SRDF source identities/geometry/provenance, never fabricated action labels."""
import json
import zipfile

import pytest

from data_pipeline import srdf_source as srdf


def native_row():
    pid = "hm3d_00000-example__fake_00000_00001"
    return {"scan": "00000-example", "path_id": pid, "path": ["00000", "00001"],
            "heading": .5, "instr_ids": [pid + "_0", pid + "_1"],
            "instructions": ["Walk to the door.", "Go forward, then wait."]}


def prepared_fixture(tmp_path, monkeypatch):
    annotation = tmp_path / "source.json"
    annotation.write_text(json.dumps([native_row()]))
    archive = tmp_path / "graphs.zip"
    graph = []
    for i in range(2):
        graph.append({"image_id": f"{i:05}", "included": True,
                      "pose": [1, 0, 0, i, 0, 1, 0, 2, 0, 0, 1, 1.5, 0, 0, 0, 1]})
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("connectivity/00000-example_connectivity.json", json.dumps(graph))
        stream.writestr("../../escape.json", "unrelated entry is never extracted")
    monkeypatch.setattr(srdf, "SOURCE_HASHES", {srdf.ANNOTATION: srdf.sha256(annotation),
                                             srdf.GRAPH_ARCHIVE: srdf.sha256(archive)})
    output = tmp_path / "SRDF"
    srdf.prepare_source(annotation, archive, output)
    return output


def test_preparation_retains_instructions_and_reference_only(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    manifest = srdf.prepared_manifest(root, verify_hashes=True)
    rows = list(srdf.read_rows(root / "train/train.jsonl"))
    original = native_row()
    assert manifest["episodes"] == 2 and manifest["routes"] == 1
    assert [r["episode_id"] for r in rows] == original["instr_ids"]
    assert [r["instructions"][0] for r in rows] == original["instructions"]
    assert all(r["path"] == original["path"] and r["heading"] == original["heading"] for r in rows)
    assert not manifest["can_collect"] and not manifest["has_expert_actions"]
    assert not (root / "train/train_gt.json.gz").exists()
    assert not (tmp_path / "escape.json").exists()
    assert (root / "raw" / srdf.ANNOTATION).is_file()
    (root / "train/train.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        srdf.prepared_manifest(root, verify_hashes=True)


def test_all_scenes_installed_still_does_not_enable_collection(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    scenes = tmp_path / "scene_datasets"
    scene = scenes / "hm3d/train/00000-example/example.basis.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"mesh")
    scene.with_suffix(".navmesh").write_bytes(b"navmesh")
    ready = srdf.availability(root, scenes)
    assert ready["source_ready"] and ready["ready_scenes"] == 1
    assert ready["scene_ready_episodes"] == 2 and not ready["can_collect"]
    assert not srdf.availability(tmp_path / "absent", scenes)["source_ready"]


@pytest.mark.parametrize("changes", [
    {"scan": "../escape"}, {"path": ["00000", "../bad"]}, {"path": ["00000"]},
    {"heading": float("nan")}, {"heading": True}, {"path_id": "different-scene"},
    {"instructions": ["one"]}, {"instructions": ["", "ok"]}, {"instr_ids": ["a", "b"]},
])
def test_malformed_rows_are_rejected(changes):
    with pytest.raises(ValueError):
        list(srdf.normalize({**native_row(), **changes}))


def test_hash_mismatch_never_publishes_source(tmp_path):
    a, g, output = tmp_path / "a", tmp_path / "g", tmp_path / "out"
    a.write_text("[]")
    g.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA-256"):
        srdf.prepare_source(a, g, output)
    assert not output.exists()


def test_missing_graph_node_has_no_completion_marker(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    annotation = tmp_path / "source.json"
    row = native_row()
    row["path"][-1] = "99999"
    annotation.write_text(json.dumps([row]))
    monkeypatch.setitem(srdf.SOURCE_HASHES, srdf.ANNOTATION, srdf.sha256(annotation))
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="missing graph viewpoints"):
        srdf.prepare_source(annotation, tmp_path / "graphs.zip", output)
    assert not (output / "source_manifest.json").exists()


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    original = (root / "source_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        srdf.prepare_source(tmp_path / "source.json", tmp_path / "graphs.zip", root)
    assert (root / "source_manifest.json").read_bytes() == original
