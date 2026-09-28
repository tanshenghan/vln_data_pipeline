"""Snapshot the validated workstation runtimes into a cloud image build context.

Requires conda-pack==0.8.1 in THIS script's environment. Reads source runtimes;
never installs into or changes a shared Habitat/LeRobot environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.api.settings import settings


def python_info(python):
    code = "import json,sys,sysconfig; print(json.dumps(dict(version=list(sys.version_info[:2]), base=sys.base_prefix, site=sysconfig.get_path('purelib'))))"
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def package_directory(source, output):
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        for path in sorted(source.iterdir()):
            archive.add(path, arcname=path.name, filter=lambda member: None if "__pycache__" in Path(member.name).parts else member)


def stage_code(output):
    # Inspect the build context before docker sends it. Only code/config file
    # types under explicit source directories are copied; no datasets or keys.
    allowed = {".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".html", ".css", ".js", ".md", ".toml"}
    excluded = {"runtime", "releases", "archive", "__pycache__", "node_modules", "tests"}
    sources = {}
    for name in ("apps", "workers", "data_pipeline", "catalog", "contracts", "src", "habitat_server", "visual"):
        for path in sorted((ROOT / name).rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part.startswith(".") or part in excluded for part in relative.parts):
                continue
            if not path.is_file() or path.is_symlink() or path.suffix not in allowed or ".local." in path.name:
                continue
            if relative == Path("data_pipeline/config.py"):
                continue
            target = output / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            sources[str(relative)] = digest(target)
    shutil.copyfile(ROOT / "configs/cloud/Dockerfile", output / "Dockerfile")
    return sources


def prepare(output):
    import conda_pack
    if output.exists():
        raise FileExistsError("Use a new build context directory; existing snapshots are not overwritten")
    platform = python_info(Path(sys.executable))
    lerobot = python_info(settings.lerobot_python)
    if platform["version"] != [3, 10] or lerobot["version"] != [3, 11]:
        raise ValueError("This image recipe requires platform Python 3.10 and LeRobot Python 3.11")
    output.mkdir(parents=True)
    sources = stage_code(output)
    print("Packing read-only Habitat environment…", flush=True)
    conda_pack.pack(prefix=str(settings.habitat_python.parent.parent), output=str(output / "habitat.tar.gz"),
                    dest_prefix="/opt/habitat", ignore_missing_files=True, compress_level=1)
    # Pip has replaced numpy in the deployed conda environment. conda-pack's
    # ignore_missing_files preserves the ACTUAL files rather than restoring stale
    # conda metadata versions. Imports are checked during image construction.
    for source, name in [(Path(platform["site"]), "platform-site.tar.gz"),
                         (Path(lerobot["base"]), "python311.tar.gz"),
                         (Path(lerobot["site"]), "lerobot-site.tar.gz")]:
        print(f"Packing {name}…", flush=True)
        package_directory(source, output / name)
    manifest = {"format": "vln-runtime-snapshot-v1", "sources": sources,
                "archives": {p.name: digest(p) for p in sorted(output.glob("*.tar.gz"))},
                "validation": "Build-time imports only; requires separate GPU/cloud validation"}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Build context ready: {output}", flush=True)


def refresh_code(output):
    manifest_path = output / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "vln-runtime-snapshot-v1":
        raise ValueError("Not a runtime snapshot build context")
    shutil.rmtree(output / "code")
    manifest["sources"] = stage_code(output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated code; runtime archives retained: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-code", action="store_true", help="Update code in an existing snapshot without repacking runtimes")
    args = parser.parse_args()
    (refresh_code if args.refresh_code else prepare)(args.output.resolve())
