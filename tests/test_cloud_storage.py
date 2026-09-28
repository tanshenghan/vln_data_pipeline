import json
import os
import sys
from pathlib import Path

import pytest

from configs import install_cloud_storage as installer


@pytest.mark.parametrize("sftp_mode", ["setpriv", "login"])
def test_second_gateway_preserves_existing_baidu_unit(tmp_path, monkeypatch, sftp_mode):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    mount = tmp_path / "mount"
    mount.mkdir()
    identity, known = tmp_path / "key", tmp_path / "known_hosts"
    identity.touch()
    known.touch()
    binary = tmp_path / "sshfs"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    units = tmp_path / ".config/systemd/user"
    units.mkdir(parents=True)
    previous = units / "vln-cloud-storage.service"
    previous.write_text("existing Baidu gateway\n")
    cfg = tmp_path / "volc.json"
    cfg.write_text(json.dumps({"target":"root@h20.example", "port":666,
        "remote_directory":"/H20_vepfs/lijiaqi/vln-pipeline", "mountpoint":str(mount),
        "identity_file":str(identity), "known_hosts_file":str(known),
        "uid":os.getuid(), "gid":os.getgid(), "sftp_server":"/usr/lib/openssh/sftp-server",
        **({"sftp_mode": "login"} if sftp_mode == "login" else {})}))
    monkeypatch.setattr(sys, "argv", ["install", "--config", str(cfg), "--sshfs-binary", str(binary),
                                      "--service-name", "vln-cloud-storage-volc"])
    commands = []
    monkeypatch.setattr(installer.subprocess, "run", lambda args, **kwargs: commands.append(args))
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/" + name)
    installer.main()
    assert previous.read_text() == "existing Baidu gateway\n"
    unit = (units / "vln-cloud-storage-volc.service").read_text()
    assert "H20_vepfs" in unit and "StrictHostKeyChecking=yes" in unit
    assert ("idmap=user" in unit) == (sftp_mode == "login")
    assert ("setpriv" in unit) == (sftp_mode == "setpriv")
    assert commands[-1] == ["systemctl", "--user", "restart", "vln-cloud-storage-volc"]
    assert all("vln-cloud-storage" not in command for command in commands)


def test_gateway_unit_name_cannot_escape_unit_directory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install", "--config", "/unused", "--service-name", "../other"])
    with pytest.raises(ValueError, match="Service name"):
        installer.main()
