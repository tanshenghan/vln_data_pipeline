"""Install a user service bridging an existing PFS directory through SSH/SFTP.

The remote directory, image user and volume markers must already exist. This
installer never creates a volume marker or changes a cloud development instance.
See docs/CLOUD_COLLECTION.md for the shared-storage protocol and its limits.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def unit_arg(value):
    value = str(value)
    if any(c in value for c in "\n\r\0"):
        raise ValueError("Service arguments must not contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sshfs-binary", type=Path, help="Optional locally extracted sshfs executable to install")
    parser.add_argument("--service-name", default="vln-cloud-storage",
                        help="Use a distinct unit for each gateway, e.g. vln-cloud-storage-volc")
    args = parser.parse_args()
    if not re.fullmatch(r"vln-cloud-storage(?:-[a-z0-9-]+)?", args.service_name):
        raise ValueError("Service name must start with vln-cloud-storage and contain only lowercase letters, digits or hyphens")
    cfg = json.loads(args.config.read_text())
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+", cfg["target"]):
        raise ValueError("target must be user@hostname")
    uid, gid = int(cfg["uid"]), int(cfg["gid"])
    if (uid, gid) != (os.getuid(), os.getgid()):
        raise ValueError("The bridge must run as the controller's numeric UID/GID")
    sftp_mode = cfg.get("sftp_mode", "setpriv")
    if sftp_mode not in {"setpriv", "login"}:
        raise ValueError("sftp_mode must be setpriv or login")
    port = int(cfg["port"])
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    for name in ("remote_directory", "mountpoint", "identity_file", "known_hosts_file", "sftp_server"):
        if not Path(cfg[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    for name in ("identity_file", "known_hosts_file"):
        if not Path(cfg[name]).is_file():
            raise ValueError(f"Missing {name}")
    mount = Path(cfg["mountpoint"])
    if not mount.is_dir():
        raise ValueError("Create an empty, user-owned mountpoint before installing")
    if not os.path.ismount(mount) and any(mount.iterdir()):
        raise ValueError("Refusing to hide existing local files with a mount")
    binary = Path.home() / ".local/lib/vln-pipeline/sshfs"
    if args.sshfs_binary:
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.sshfs_binary, binary)
    elif not binary.is_file():
        installed = shutil.which("sshfs")
        if not installed:
            raise ValueError("Install sshfs or supply --sshfs-binary")
        binary = Path(installed)
    unmount = shutil.which("fusermount3")
    if not unmount:
        raise ValueError("FUSE 3 is required")
    command = [str(binary), "-f", f"{cfg['target']}:{cfg['remote_directory']}", str(mount), "-p", str(port)]
    # An ordinary development-machine login needs no sudo. Its image must run
    # under the same *remote* UID/GID; SSHFS maps that owner to the controller.
    server = (shlex.quote(cfg["sftp_server"]) if sftp_mode == "login" else
              shlex.join(["setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", cfg["sftp_server"]]))
    options = [f"IdentityFile={cfg['identity_file']}", f"UserKnownHostsFile={cfg['known_hosts_file']}",
               "StrictHostKeyChecking=yes", "IdentitiesOnly=yes", "BatchMode=yes", "ConnectTimeout=8",
               "ServerAliveInterval=5", "ServerAliveCountMax=2", "dir_cache=no", "direct_io",
               "attr_timeout=0", "entry_timeout=0", "negative_timeout=0", "sshfs_sync", "auto_unmount",
               f"sftp_server={server}"]
    if sftp_mode == "login":
        options.append("idmap=user")
    for option in options:
        command.extend(["-o", option])
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    (units / f"{args.service_name}.service").write_text(f"""[Unit]
Description=VLN PFS control and artifact bridge
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart={' '.join(unit_arg(part) for part in command)}
ExecStop={unit_arg(unmount)} -u {unit_arg(mount)}
Restart=on-failure
RestartSec=10
TimeoutStopSec=20

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", args.service_name], check=True)
    subprocess.run(["systemctl", "--user", "restart", args.service_name], check=True)
    print("Storage service installed. Verify the remote volume markers before enabling pools.")


if __name__ == "__main__":
    main()
