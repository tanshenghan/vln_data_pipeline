"""Offline safety tests: no network, credential values or live shared data."""
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from data_pipeline import download_habitat_assets as download
from data_pipeline import install_habitat_assets as install


def package(tmp_path, name, files):
    root = tmp_path / "downloads"
    root.mkdir(exist_ok=True)
    archive = root / name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "w") as z:
            for member, data in files:
                z.writestr(member, data)
    else:
        with tarfile.open(archive, "w:gz" if name.endswith(".gz") else "w") as t:
            for member, data in files:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
    download.atomic_json(root / (name + ".download.json"), {
        "complete": True, "url": download.PACKAGES[name],
        "bytes": archive.stat().st_size, "sha256": download.sha256(archive)})
    source = tmp_path / "shared"
    source.mkdir(exist_ok=True)
    return root, source


@pytest.mark.parametrize("name", ["/etc/passwd", "../escape", "a/../../escape", "a\\b", "./a", "a//b"])
def test_traversal_rejected(name):
    with pytest.raises(ValueError):
        install.safe_name(name)


@pytest.mark.parametrize("name,prefix,family,version", [
    ("objectnav_mp3d_v1.zip", "", "mp3d", "v1"),
    ("objectnav_hm3d_v1.zip", "objectnav_hm3d_v1/", "hm3d", "v1"),
    ("objectnav_hm3d_v2.zip", "objectnav_hm3d_v2/", "hm3d", "v2"),
])
def test_native_objectnav_layout_and_provenance(tmp_path, name, prefix, family, version):
    root, source = package(tmp_path, name, [(prefix + "train/content/ABCDEFGHIJK.json.gz", b"example")])
    result = install.install_package(name, root, source)
    target = source / "datasets/objectnav" / family / version / "train/content/ABCDEFGHIJK.json.gz"
    assert target.read_bytes() == b"example"
    assert result["complete"] and result["url"] == download.PACKAGES[name]
    assert install.install_package(name, root, source)["files"] == result["files"]


def test_scene_example_link_preserved_and_conflicts_never_overwritten(tmp_path):
    name = "hm3d-train-habitat-v0.2.tar"
    member = "00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb"
    root, source = package(tmp_path, name, [(member, b"original")])
    base = source / "scene_datasets/hm3d"
    example = base / "example/00000-ABCDEFGHIJK"
    example.mkdir(parents=True)
    (base / "train").mkdir()
    alias = base / "train/00000-ABCDEFGHIJK"
    alias.symlink_to("../example/00000-ABCDEFGHIJK")
    (example / "ABCDEFGHIJK.basis.glb").write_bytes(b"original")
    install.install_package(name, root, source)
    assert alias.is_symlink()
    (example / "ABCDEFGHIJK.basis.glb").write_bytes(b"user-edited")
    with pytest.raises(ValueError, match="not overwritten"):
        install.install_package(name, root, source)
    assert (example / "ABCDEFGHIJK.basis.glb").read_bytes() == b"user-edited"


def test_semantic_versions_never_share_annotation_paths(tmp_path):
    targets = []
    for name in ("hm3d-train-semantic-annots-v0.1.tar.gz", "hm3d-train-semantic-annots-v0.2.tar"):
        root, source = package(tmp_path, name, [("00000-ABCDEFGHIJK/ABCDEFGHIJK.semantic.glb", name.encode())])
        result = install.install_package(name, root, source)
        targets.append(source / result["files"][0]["path"])
    assert targets[0] != targets[1]
    assert "versions/v0.1" in str(targets[0])
    assert targets[0].read_bytes() != targets[1].read_bytes()


def test_legacy_geometry_is_not_replaced_by_v02(tmp_path):
    paths = []
    for name in ("hm3d-train-habitat.tar", "hm3d-train-habitat-v0.2.tar"):
        root, source = package(tmp_path, name, [("00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb", name.encode())])
        result = install.install_package(name, root, source)
        paths.append(source / result["files"][0]["path"])
    assert paths[0] != paths[1]
    assert "versions/v0.1" in str(paths[0])
    assert paths[0].read_bytes() != paths[1].read_bytes()


def test_configs_at_correct_relative_level(tmp_path):
    root = tmp_path
    assert install.destination("hm3d-train-configs.tar", "hm3d_basis.scene_dataset_config.json", root) == root / "scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json"
    assert install.destination("hm3d-train-semantic-configs-v0.1.tar", "hm3d_annotated_train_basis.scene_dataset_config.json", root) == root / "scene_datasets/hm3d/versions/v0.1/train/hm3d_annotated_train_basis.scene_dataset_config.json"


def test_tampered_archive_never_installed(tmp_path):
    name = "objectnav_mp3d_v1.zip"
    root, source = package(tmp_path, name, [("train/train.json.gz", b"data")])
    with (root / name).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Unverified"):
        install.install_package(name, root, source)
    assert not (source / "datasets").exists()


def test_external_symlink_rejected(tmp_path):
    name = "hm3d-train-habitat-v0.2.tar"
    root, source = package(tmp_path, name, [("00000-ABCDEFGHIJK/ABCDEFGHIJK.basis.glb", b"data")])
    base = source / "scene_datasets/hm3d"
    base.mkdir(parents=True)
    (base / "train").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        install.install_package(name, root, source)
    assert not (tmp_path / "00000-ABCDEFGHIJK").exists()


def test_archive_symlinks_rejected(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as t:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        t.addfile(info)
    with pytest.raises(ValueError, match="link rejected"):
        with install.archive_files(archive) as members:
            list(members)


def test_partial_scene_not_counted_as_ready(tmp_path):
    scene = tmp_path / "scene_datasets/hm3d/train/00000-ABCDEFGHIJK"
    scene.mkdir(parents=True)
    (scene / "ABCDEFGHIJK.basis.glb").write_bytes(b"mesh")
    assert install.inventory(tmp_path)["hm3d"]["train"]["ready_scenes"] == 0
    (scene / "ABCDEFGHIJK.basis.navmesh").write_bytes(b"navmesh")
    report = install.inventory(tmp_path)
    assert report["hm3d"]["train"]["ready_scenes"] == 1
    assert not report["public_geometry_complete"]


def test_missing_consumer_inputs_not_claimed_ready(tmp_path):
    coverage = install.consumer_coverage(tmp_path)
    assert not coverage["scalevln"]["source_ready"]
    assert len(coverage["objectnav"]) == 3
    assert not any(v["can_collect"] or v["source_index_ready"] for v in coverage["objectnav"].values())


def fake_http(monkeypatch, responses):
    calls = []

    class Response:
        def __init__(self, status, headers):
            self.status_code, self.headers = status, headers
            self.is_redirect = status in (301, 302, 303, 307, 308)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_bytes(self, **kwargs):
            yield b"binary archive preview"

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, target, **kwargs):
            calls.append((method, target, kwargs))
            return Response(*responses.pop(0))

    monkeypatch.setitem(__import__("sys").modules, "httpx", SimpleNamespace(Client=Client))
    return calls


def test_access_probe_sends_credentials_only_to_official_api(monkeypatch):
    calls = fake_http(monkeypatch, [(302, {"location": "https://bucket.s3.amazonaws.com/file?signed=private"}),
                                   (206, {"content-range": "bytes 0-1023/4096", "etag": '"release"'})])
    auth = ("dummy_id", "dummy_secret")
    result = download.probe(download.MATTERPORT + "hm3d-train-configs.tar", auth)
    assert result["bytes"] == 4096
    assert calls[0][0] == "GET" and calls[0][2]["auth"] == auth
    assert calls[1][2]["auth"] is None
    assert "signed" not in json.dumps(result) and "dummy_secret" not in json.dumps(result)


def test_access_probe_rejects_non_official_redirect(monkeypatch):
    calls = fake_http(monkeypatch, [(302, {"location": "https://example.invalid/file"})])
    with pytest.raises(ValueError, match="HTTPS object store"):
        download.probe(download.MATTERPORT + "file", ("dummy", "secret"))
    assert len(calls) == 1


def test_changed_partial_is_preserved_not_resumed(tmp_path, monkeypatch):
    name = "objectnav_mp3d_v1.zip"
    partial = tmp_path / (name + ".part")
    partial.write_bytes(b"old partial")
    download.atomic_json(tmp_path / (name + ".download.json"), {"bytes": 100, "etag": "old"})
    monkeypatch.setattr(download, "probe", lambda *args: {"bytes": 101, "etag": "new"})
    with pytest.raises(ValueError, match="different source version"):
        download.download(name, tmp_path, ("dummy", "secret"))
    assert partial.read_bytes() == b"old partial"


def test_existing_completed_archive_is_hash_checked(tmp_path, monkeypatch):
    name = "objectnav_mp3d_v1.zip"
    (tmp_path / name).write_bytes(b"verified")
    report = {"complete": True, "bytes": 8, "etag": "release", "url": download.PACKAGES[name],
              "sha256": download.sha256(tmp_path / name)}
    download.atomic_json(tmp_path / (name + ".download.json"), report)
    monkeypatch.setattr(download, "probe", lambda *args: {"bytes": 8, "etag": "release"})
    assert download.download(name, tmp_path, ("dummy", "secret")) == report
    (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        download.download(name, tmp_path, ("dummy", "secret"))
    assert (tmp_path / name).read_bytes() == b"tampered"
