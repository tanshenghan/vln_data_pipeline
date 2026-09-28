"""Resumable official Habitat downloads. Credentials never enter argv or logs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit

MATTERPORT = "https://api.matterport.com/resources/habitat/"
HABITAT = "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/"
PACKAGES = {
    **{f"hm3d-{split}-{kind}-v0.2.tar": MATTERPORT + f"hm3d-{split}-{kind}-v0.2.tar"
       for split in ("train", "val") for kind in ("habitat", "semantic-annots", "semantic-configs")},
    **{f"hm3d-{split}-configs.tar": MATTERPORT + f"hm3d-{split}-configs.tar" for split in ("train", "val")},
    # v0.1 geometry is not interchangeable with v0.2 (publisher debris fixes).
    **{f"hm3d-{split}-habitat.tar": MATTERPORT + f"hm3d-{split}-habitat.tar" for split in ("train", "val")},
    **{f"hm3d-{split}-semantic-annots-v0.1.tar.gz": MATTERPORT + f"hm3d-{split}-semantic-annots-v0.1.tar.gz"
       for split in ("train", "val")},
    **{f"hm3d-{split}-semantic-configs-v0.1.tar": MATTERPORT + f"hm3d-{split}-semantic-configs-v0.1.tar"
       for split in ("train", "val")},
    "objectnav_mp3d_v1.zip": HABITAT + "m3d/v1/objectnav_mp3d_v1.zip",
    "objectnav_hm3d_v1.zip": HABITAT + "hm3d/v1/objectnav_hm3d_v1.zip",
    "objectnav_hm3d_v2.zip": HABITAT + "hm3d/v2/objectnav_hm3d_v2.zip",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def probe(url, auth):
    """GET range, not HEAD: the publisher signs download URLs for GET only."""
    import httpx

    headers = {"Range": "bytes=0-1023", "Cache-Control": "no-cache"}
    with httpx.Client(timeout=40) as client:
        target = url
        for step in range(4):
            # Only the official API receives the account credential.
            credential = auth if target.startswith(MATTERPORT) else None
            with client.stream("GET", target, auth=credential, headers=headers) as response:
                if response.is_redirect:
                    target = response.headers["location"]
                    host = urlsplit(target)
                    if host.scheme != "https" or not (host.hostname or "").endswith(".amazonaws.com"):
                        raise ValueError("Official download did not authorize access to an HTTPS object store")
                    continue
                if response.status_code not in (200, 206):
                    raise ValueError(f"Download access check failed: HTTP {response.status_code}")
                value = response.headers.get("content-range", "")
                total = int(value.rsplit("/", 1)[1]) if value else int(response.headers["content-length"])
                next(response.iter_bytes(chunk_size=1024), b"")
                return {"bytes": total, "etag": response.headers.get("etag"),
                        "last_modified": response.headers.get("last-modified")}
    raise ValueError("Too many download redirects")


def download(name, root, auth):
    url = PACKAGES[name]
    final, partial = root / name, root / (name + ".part")
    report = root / (name + ".download.json")
    metadata = probe(url, auth)
    if final.exists():
        saved = json.loads(report.read_text()) if report.exists() else {}
        if (saved.get("complete") and saved.get("url") == url and saved.get("etag") == metadata["etag"]
                and final.stat().st_size == metadata["bytes"] and sha256(final) == saved.get("sha256")):
            return saved
        raise ValueError(f"Existing archive is unverified or changed; refusing to overwrite: {name}")
    previous = json.loads(report.read_text()) if report.exists() else {}
    if partial.exists() and partial.stat().st_size and (
            previous.get("etag") != metadata["etag"] or previous.get("bytes") != metadata["bytes"]):
        raise ValueError(f"Partial archive belongs to a different source version: {name}")
    state = {"complete": False, "url": url, **metadata, "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(report, state)
    options = ["url = " + json.dumps(url), "output = " + json.dumps(str(partial))]
    if url.startswith(MATTERPORT):
        options.append("user = " + json.dumps(":".join(auth)))
    # Curl strips Authorization on a cross-host redirect. Never use location-trusted.
    if metadata["etag"]:
        options.append("header = " + json.dumps("If-Match: " + metadata["etag"]))
    argv = ["curl", "--config", "-", "--location", "--fail", "--silent", "--show-error",
            "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-",
            "--retry", "8", "--retry-delay", "5", "--connect-timeout", "30",
            "--speed-time", "120", "--speed-limit", "1024"]
    print(json.dumps({"event": "download", "file": name, "bytes": metadata["bytes"]}), flush=True)
    with (root / (name + ".curl.log")).open("ab") as log:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=log)
        process.stdin.write(("\n".join(options) + "\n").encode())
        process.stdin.close()
        while process.poll() is None:
            print(json.dumps({"event": "progress", "file": name,
                              "downloaded": partial.stat().st_size if partial.exists() else 0,
                              "total": metadata["bytes"]}), flush=True)
            time.sleep(30)
        if process.returncode:
            raise RuntimeError(f"Download failed for {name}; curl exit {process.returncode}; partial retained")
    if partial.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Archive size mismatch: {name}")
    state.update(sha256=sha256(partial), complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    partial.replace(final)
    atomic_json(report, state)
    print(json.dumps({"event": "complete", "file": name, "bytes": metadata["bytes"]}), flush=True)
    return state


def main():
    from dotenv import dotenv_values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packages", nargs="+", choices=sorted(PACKAGES), default=list(PACKAGES))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--summary-name", default="download_summary.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z_]*download_summary\.json", args.summary_name):
        raise ValueError("Summary name must be a local download_summary.json basename")
    if args.credentials.stat().st_mode & 0o077:
        raise ValueError("Credentials file must be private (chmod 600)")
    config = dotenv_values(args.credentials)
    auth = tuple(config.get(k, "") for k in ("MATTERPORT_API_TOKEN_ID", "MATTERPORT_API_TOKEN_SECRET"))
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", part) for part in auth):
        raise ValueError("Missing or malformed Matterport credentials")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(download, name, args.output, auth): name for name in args.packages}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                # Never print HTTP exceptions: their URLs can contain signed credentials.
                failures.append({"file": jobs[job], "error_type": type(exc).__name__})
                print(json.dumps({"event": "failed", **failures[-1]}), flush=True)
    atomic_json(args.output / args.summary_name, {"complete": not failures, "failures": failures,
                "packages": args.packages, "finished_at": datetime.now(timezone.utc).isoformat()})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
