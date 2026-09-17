#!/usr/bin/env python3
"""Serve paginated LeRobot video metadata and allow-listed MP4 byte ranges."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, quote, unquote, urlsplit

try:
    from .video_index import VIDEO_INDEX_SCHEMA_VERSION, format_segment_video_relative_path, format_video_relative_path
except ImportError:
    from video_index import (  # type: ignore[no-redef]
        VIDEO_INDEX_SCHEMA_VERSION,
        format_segment_video_relative_path,
        format_video_relative_path,
    )

_MAX_INDEX_BYTES = 512 * 1024 * 1024
_MAX_TOKEN_BYTES = 4096
_MAX_PAGE_SIZE = 100
_STREAM_CHUNK_BYTES = 1024 * 1024
_RECORD_ROUTE = re.compile(r"^/v1/records/([^/]+)/videos$")
_MEDIA_ROUTE = re.compile(r"^/v1/media/([^/]+)$")
_H264_CODECS = {"h264", "avc", "avc1"}
_EXPOSE_HEADERS = "Accept-Ranges, Content-Length, Content-Range, Content-Type, Cache-Control, Retry-After"
_TRANSCODE_PROFILE = "h264-v1"
_TRANSCODE_EXPANSION_FACTOR = 4
_MIN_TRANSCODE_OUTPUT_BYTES = 8 * 1024 * 1024
_TRANSCODE_MUXER_GUARD_BYTES = 1024 * 1024
_DEFAULT_TOKEN_TTL_SECONDS = 24 * 60 * 60


class VideoServiceError(Exception):
    """HTTP-safe service error which never exposes an absolute media path."""

    def __init__(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


class InvalidAssetToken(ValueError):
    """Raised when a token is malformed, forged, or expired."""


class RangeNotSatisfiable(ValueError):
    """Raised when a single HTTP byte range cannot be satisfied."""


@dataclass(frozen=True)
class MediaAsset:
    """A resolved media target derived only from the trusted index."""

    path: Path
    record_id: str
    episode_index: int
    camera_key: str
    codec: str | None


@dataclass
class _TranscodeJob:
    event: threading.Event
    error: VideoServiceError | None = None


def _file_fingerprint(path: Path) -> str:
    """Return an opaque identity for the currently resolved regular file."""

    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("media is not a regular file")
    payload = json.dumps(
        {
            "path": resolved.as_posix(),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TranscodeManager:
    """Materialize validated H.264 MP4s into an atomic, bounded cache."""

    def __init__(
        self,
        cache_root: Path,
        *,
        ffmpeg: str,
        ffprobe: str,
        forbidden_roots: tuple[Path, ...],
        workers: int = 1,
        threads: int = 4,
        timeout_seconds: int = 600,
        wait_timeout_seconds: int = 120,
        cache_max_bytes: int = 100 * 1024**3,
        min_free_bytes: int = 20 * 1024**3,
    ) -> None:
        if workers <= 0 or threads <= 0 or timeout_seconds <= 0 or wait_timeout_seconds <= 0:
            raise ValueError("transcode concurrency and timeout values must be positive")
        if cache_max_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("transcode cache limits are invalid")
        self._validate_tools(ffmpeg, ffprobe)
        requested_root = Path(os.path.abspath(cache_root))
        if requested_root.is_symlink():
            raise ValueError("transcode cache directory cannot be a symbolic link")
        requested_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise ValueError("transcode cache directory must be a real directory")
        self.cache_root = requested_root.resolve(strict=True)
        resolved_forbidden_roots = tuple(root.resolve() for root in forbidden_roots)
        if _contained_by(self.cache_root, resolved_forbidden_roots):
            raise ValueError("transcode cache directory cannot be inside a dataset root")
        self.profile_root = self.cache_root / _TRANSCODE_PROFILE
        self.profile_root.mkdir(mode=0o700, exist_ok=True)
        if self.profile_root.is_symlink() or not self.profile_root.is_dir():
            raise ValueError("transcode cache profile directory is invalid")
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.threads = threads
        self.timeout_seconds = timeout_seconds
        self.wait_timeout_seconds = wait_timeout_seconds
        self.cache_max_bytes = cache_max_bytes
        self.min_free_bytes = min_free_bytes
        self._worker_slots = threading.BoundedSemaphore(workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, _TranscodeJob] = {}
        self._validated_cache: dict[tuple[str, int, int], bool] = {}
        self._reserved_bytes = 0
        self._cache_bytes = self._measure_cache_bytes()

    @staticmethod
    def _validate_tools(ffmpeg: str, ffprobe: str) -> None:
        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            encoder = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=2x2:d=0.04",
                    "-frames:v",
                    "1",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("transcode cache requires working ffmpeg, ffprobe, and libx264") from exc
        if probe.returncode != 0 or encoder.returncode != 0:
            raise ValueError("transcode cache requires working ffmpeg, ffprobe, and libx264")

    def _measure_cache_bytes(self) -> int:
        total = 0
        try:
            for shard in self.profile_root.iterdir():
                if shard.is_symlink() or not shard.is_dir():
                    continue
                for item in shard.iterdir():
                    if item.is_symlink() or not item.is_file() or item.suffix != ".mp4":
                        continue
                    total += item.stat().st_size
        except OSError:
            return total
        return total

    @staticmethod
    def _cache_key(source: Path) -> str:
        payload = f"{_TRANSCODE_PROFILE}:{_file_fingerprint(source)}".encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _target_for_key(self, key: str) -> Path:
        shard = self.profile_root / key[:2]
        shard.mkdir(mode=0o700, exist_ok=True)
        if shard.is_symlink() or not shard.is_dir():
            raise VideoServiceError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "transcode_failed",
                "The compatibility video cache is invalid.",
            )
        return shard / f"{key}.mp4"

    def _probe_cached_h264(self, path: Path) -> bool:
        try:
            metadata = path.stat()
        except OSError:
            return False
        cache_key = (path.as_posix(), metadata.st_size, metadata.st_mtime_ns)
        with self._lock:
            cached = self._validated_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path.as_posix(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(30, self.timeout_seconds),
            )
            codec = (
                result.stdout.strip().splitlines()[0].lower() if result.returncode == 0 and result.stdout.strip() else ""
            )
            valid = codec in _H264_CODECS
        except (OSError, subprocess.SubprocessError):
            valid = False
        with self._lock:
            self._validated_cache[cache_key] = valid
        return valid

    def cached_path(self, source: Path) -> Path | None:
        try:
            key = self._cache_key(source)
            target = self._target_for_key(key)
            if target.is_symlink() or not target.is_file() or target.stat().st_size <= 0:
                return None
            resolved = target.resolve(strict=True)
            if not _contained_by(resolved, (self.cache_root,)):
                return None
            return resolved if self._probe_cached_h264(resolved) else None
        except (OSError, VideoServiceError):
            return None

    def _reserve_capacity(self, source_bytes: int) -> tuple[int, int]:
        with self._lock:
            free_bytes = shutil.disk_usage(self.cache_root).free
            quota_budget = self.cache_max_bytes - self._cache_bytes - self._reserved_bytes
            disk_budget = free_bytes - self.min_free_bytes - self._reserved_bytes
            desired_output = max(source_bytes * _TRANSCODE_EXPANSION_FACTOR, _MIN_TRANSCODE_OUTPUT_BYTES)
            reserve_budget = min(desired_output + _TRANSCODE_MUXER_GUARD_BYTES, quota_budget, disk_budget)
            if reserve_budget < _MIN_TRANSCODE_OUTPUT_BYTES + _TRANSCODE_MUXER_GUARD_BYTES:
                raise VideoServiceError(
                    HTTPStatus.INSUFFICIENT_STORAGE,
                    "transcode_cache_full",
                    "The compatibility video cache has insufficient capacity.",
                )
            self._reserved_bytes += reserve_budget
            return reserve_budget, reserve_budget - _TRANSCODE_MUXER_GUARD_BYTES

    def _release_capacity(self, expected_bytes: int) -> None:
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - expected_bytes)

    def _publish(self, temporary: Path, target: Path, source: Path, source_fingerprint: str) -> Path:
        try:
            actual_bytes = temporary.stat().st_size
            previous_bytes = target.stat().st_size if target.is_file() and not target.is_symlink() else 0
            with self._lock:
                projected = self._cache_bytes - previous_bytes + actual_bytes
                free_bytes = shutil.disk_usage(self.cache_root).free
                if projected > self.cache_max_bytes or free_bytes < self.min_free_bytes:
                    raise VideoServiceError(
                        HTTPStatus.INSUFFICIENT_STORAGE,
                        "transcode_cache_full",
                        "The compatibility video cache has insufficient capacity.",
                    )
                if target.is_symlink():
                    raise VideoServiceError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "transcode_failed",
                        "The compatibility video cache is invalid.",
                    )
                if _file_fingerprint(source) != source_fingerprint:
                    raise VideoServiceError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "transcode_source_changed",
                        "The source video changed during compatibility conversion.",
                    )
                os.replace(temporary, target)
                self._cache_bytes = projected
            return target.resolve(strict=True)
        except OSError as exc:
            raise VideoServiceError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "transcode_failed",
                "The compatibility video could not be published.",
            ) from exc

    def _run_transcode(self, source: Path, target: Path) -> Path:
        source_size = source.stat().st_size
        source_fingerprint = _file_fingerprint(source)
        reserve_budget, output_limit = self._reserve_capacity(source_size)
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=".tmp.mp4", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                result = subprocess.run(
                    [
                        self.ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        source.as_posix(),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-sn",
                        "-dn",
                        "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-tag:v",
                        "avc1",
                        "-threads",
                        str(self.threads),
                        "-movflags",
                        "+faststart",
                        "-map_metadata",
                        "-1",
                        "-fs",
                        str(output_limit),
                        temporary.as_posix(),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise VideoServiceError(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "transcode_timeout",
                    "The compatibility video conversion timed out.",
                ) from exc
            except OSError as exc:
                raise VideoServiceError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "transcode_failed",
                    "The compatibility video conversion failed.",
                ) from exc
            actual_bytes = temporary.stat().st_size if temporary.is_file() else 0
            if actual_bytes >= output_limit or actual_bytes > reserve_budget:
                raise VideoServiceError(
                    HTTPStatus.INSUFFICIENT_STORAGE,
                    "transcode_cache_full",
                    "The compatibility video exceeded its cache budget.",
                )
            if result.returncode != 0 or actual_bytes <= 0:
                raise VideoServiceError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "transcode_failed",
                    "The compatibility video conversion failed.",
                )
            if not self._probe_cached_h264(temporary):
                raise VideoServiceError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "transcode_failed",
                    "The compatibility video failed validation.",
                )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            published = self._publish(temporary, target, source, source_fingerprint)
            return published
        finally:
            self._release_capacity(reserve_budget)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def materialize(self, source: Path) -> Path:
        cached = self.cached_path(source)
        if cached:
            return cached
        try:
            key = self._cache_key(source)
            target = self._target_for_key(key)
        except OSError as exc:
            raise VideoServiceError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "transcode_failed",
                "The compatibility video cache is unavailable.",
            ) from exc

        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                job = _TranscodeJob(threading.Event())
                self._jobs[key] = job
                leader = True
            else:
                leader = False
        if not leader:
            if not job.event.wait(self.wait_timeout_seconds):
                raise VideoServiceError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "transcode_pending",
                    "The compatibility video is still being prepared.",
                    headers={"Retry-After": "5"},
                )
            if job.error:
                raise VideoServiceError(job.error.status, job.error.code, job.error.message, headers=job.error.headers)
            cached = self.cached_path(source)
            if cached:
                return cached
            raise VideoServiceError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "transcode_failed",
                "The compatibility video conversion failed.",
            )

        try:
            cached = self.cached_path(source)
            if cached:
                return cached
            if not self._worker_slots.acquire(blocking=False):
                raise VideoServiceError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "transcode_busy",
                    "All compatibility video workers are busy.",
                    headers={"Retry-After": "5"},
                )
            try:
                return self._run_transcode(source, target)
            finally:
                self._worker_slots.release()
        except VideoServiceError as exc:
            job.error = exc
            raise
        except OSError as exc:
            error = VideoServiceError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "transcode_failed",
                "The compatibility video conversion failed.",
            )
            job.error = error
            raise error from exc
        finally:
            with self._lock:
                self._jobs.pop(key, None)
                job.event.set()


def _base64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _base64url_decode(payload: str) -> bytes:
    if not payload or len(payload) > _MAX_TOKEN_BYTES or not re.fullmatch(r"[A-Za-z0-9_-]+", payload):
        raise InvalidAssetToken("invalid base64url component")
    padding = "=" * (-len(payload) % 4)
    try:
        return base64.b64decode(payload + padding, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidAssetToken("invalid base64url component") from exc


class AssetTokenManager:
    """Issue self-contained locators whose content cannot be forged by clients."""

    def __init__(self, secret: bytes, *, ttl_seconds: int = 3600) -> None:
        if len(secret) < 16:
            raise ValueError("asset token secret must contain at least 16 bytes")
        if ttl_seconds <= 0:
            raise ValueError("asset token ttl must be positive")
        self._secret = secret
        self.ttl_seconds = ttl_seconds

    def issue(self, locator: dict[str, Any], *, now: int | None = None) -> str:
        issued_at = int(time.time()) if now is None else now
        payload = {**locator, "x": issued_at + self.ttl_seconds}
        encoded = _base64url_encode(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        signature = _base64url_encode(hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest())
        return f"{encoded}.{signature}"

    def verify(self, token: str, *, now: int | None = None) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > _MAX_TOKEN_BYTES or token.count(".") != 1:
            raise InvalidAssetToken("malformed asset token")
        encoded, supplied_signature = token.split(".", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded) or not re.fullmatch(r"[A-Za-z0-9_-]+", supplied_signature):
            raise InvalidAssetToken("malformed asset token")
        expected_signature = _base64url_encode(
            hmac.new(self._secret, encoded.encode("ascii", errors="strict"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise InvalidAssetToken("asset token signature mismatch")
        try:
            payload = json.loads(_base64url_decode(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidAssetToken("invalid asset token payload") from exc
        if not isinstance(payload, dict):
            raise InvalidAssetToken("invalid asset token payload")
        expires_at = payload.get("x")
        effective_now = int(time.time()) if now is None else now
        if not isinstance(expires_at, int) or isinstance(expires_at, bool) or expires_at <= effective_now:
            raise InvalidAssetToken("asset token expired")
        return payload


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, rejecting multi-range requests."""

    if header is None:
        return None
    if size <= 0 or not header.startswith("bytes="):
        raise RangeNotSatisfiable("unsupported byte range")
    specification = header[6:].strip()
    if not specification or "," in specification or specification.count("-") != 1:
        raise RangeNotSatisfiable("only one byte range is supported")
    start_text, end_text = (item.strip() for item in specification.split("-", 1))
    if not start_text:
        if not end_text.isdigit():
            raise RangeNotSatisfiable("invalid suffix byte range")
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise RangeNotSatisfiable("invalid suffix byte range")
        return max(0, size - suffix_length), size - 1
    if not start_text.isdigit() or (end_text and not end_text.isdigit()):
        raise RangeNotSatisfiable("invalid byte range")
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start >= size or end < start:
        raise RangeNotSatisfiable("unsatisfiable byte range")
    return start, min(end, size - 1)


def _contained_by(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _load_index(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_INDEX_BYTES:
            raise ValueError("video index must be a bounded regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("video index is unavailable or invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != VIDEO_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported video index schema")
    return payload


def _normalize_public_base_url(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("public_base_url must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def _validate_cors_origins(origins: list[str] | tuple[str, ...]) -> frozenset[str]:
    validated: set[str] = set()
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"invalid exact CORS origin: {origin}")
        validated.add(f"{parsed.scheme}://{parsed.netloc}")
    if not validated:
        raise ValueError("at least one exact CORS origin is required")
    return frozenset(validated)


class VideoService:
    """Read-only video metadata and media resolver."""

    def __init__(
        self,
        index: dict[str, Any],
        *,
        secret: bytes,
        media_roots: list[Path] | tuple[Path, ...] | None,
        cors_origins: list[str] | tuple[str, ...],
        public_base_url: str | None = None,
        token_ttl_seconds: int = _DEFAULT_TOKEN_TTL_SECONDS,
        ffprobe: str | None = None,
        ffmpeg: str | None = None,
        transcode_cache: Path | None = None,
        transcode_workers: int = 1,
        transcode_threads: int = 4,
        transcode_timeout_seconds: int = 600,
        transcode_wait_timeout_seconds: int = 120,
        transcode_cache_max_bytes: int = 100 * 1024**3,
        transcode_min_free_bytes: int = 20 * 1024**3,
    ) -> None:
        if index.get("schema_version") != VIDEO_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported video index schema")
        records = index.get("records")
        if not isinstance(records, dict):
            raise ValueError("video index records must be an object")
        conversion_roots = index.get("allowed_conversion_roots")
        if not isinstance(conversion_roots, list) or not all(isinstance(root, str) for root in conversion_roots):
            raise ValueError("video index needs allowed_conversion_roots")
        self.conversion_roots = tuple(Path(root).resolve() for root in conversion_roots)
        if not self.conversion_roots:
            raise ValueError("video index has no allowed conversion roots")
        configured_media_roots = list(media_roots or [Path(root) for root in conversion_roots])
        self.media_roots = tuple(Path(root).resolve() for root in configured_media_roots)
        if not self.media_roots:
            raise ValueError("at least one media root is required")
        self.cors_origins = _validate_cors_origins(cors_origins)
        self.public_base_url = _normalize_public_base_url(public_base_url)
        self.token_manager = AssetTokenManager(secret, ttl_seconds=token_ttl_seconds)
        self.ffprobe = ffprobe
        self._codec_cache: dict[str, str | None] = {}
        if transcode_cache is not None and (not ffmpeg or not ffprobe):
            raise ValueError("transcode cache requires both ffmpeg and ffprobe")
        self.transcoder = (
            TranscodeManager(
                transcode_cache,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                forbidden_roots=tuple(dict.fromkeys((*self.conversion_roots, *self.media_roots))),
                workers=transcode_workers,
                threads=transcode_threads,
                timeout_seconds=transcode_timeout_seconds,
                wait_timeout_seconds=transcode_wait_timeout_seconds,
                cache_max_bytes=transcode_cache_max_bytes,
                min_free_bytes=transcode_min_free_bytes,
            )
            if transcode_cache is not None and ffmpeg and ffprobe
            else None
        )
        self.records = records
        self._validate_records()

    @classmethod
    def from_file(
        cls,
        index_path: Path,
        **kwargs: Any,
    ) -> "VideoService":
        return cls(_load_index(index_path), **kwargs)

    def _validate_records(self) -> None:
        for record_id, record in self.records.items():
            if not isinstance(record_id, str) or not record_id or not isinstance(record, dict):
                raise ValueError("invalid record in video index")
            sources = record.get("sources")
            if not isinstance(sources, list):
                raise ValueError("video record sources must be a list")
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("video source must be an object")
                root_value = source.get("root")
                if not isinstance(root_value, str):
                    raise ValueError("video source root is missing")
                root = Path(root_value).resolve()
                if not _contained_by(root, self.conversion_roots):
                    raise ValueError("video source root is outside the indexed conversion roots")
                template = source.get("video_path_template")
                chunks_size = source.get("chunks_size")
                cameras = source.get("camera_keys")
                episodes = source.get("episodes")
                if not isinstance(template, str) or not isinstance(cameras, list) or not isinstance(episodes, list):
                    raise ValueError("video source metadata is invalid")
                segmented = source.get("segmented_media") is True
                for camera in cameras:
                    if segmented:
                        format_segment_video_relative_path(template, chunk_index=0, file_index=0, camera_key=camera)
                    else:
                        format_video_relative_path(template, chunks_size=chunks_size, episode_index=0, camera_key=camera)
                for episode in episodes:
                    if not isinstance(episode, dict):
                        raise ValueError("video episode metadata is invalid")
                    episode_index = episode.get("episode_index")
                    if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
                        raise ValueError("video episode index is invalid")
                    display_index = episode.get("display_episode_index")
                    if display_index is not None and (
                        not isinstance(display_index, int) or isinstance(display_index, bool) or display_index < 0
                    ):
                        raise ValueError("video display episode index is invalid")
                    if segmented:
                        segments = episode.get("camera_segments")
                        common_segment = episode.get("segment")
                        if common_segment is None and not isinstance(segments, dict):
                            raise ValueError("segmented video episode locator is missing")
                        for camera in cameras:
                            self._segment_values(common_segment if common_segment is not None else segments.get(camera))

    def origin_allowed(self, origin: str | None) -> bool:
        return origin is None or origin in self.cors_origins

    def _iter_items(self, record: dict[str, Any]) -> Iterator[tuple[int, int, int, dict[str, Any], dict[str, Any], str]]:
        for source_index, source in enumerate(record["sources"]):
            for episode_position, episode in enumerate(source["episodes"]):
                for camera_position, camera_key in enumerate(source["camera_keys"]):
                    yield source_index, episode_position, camera_position, source, episode, camera_key

    def _metadata_path(
        self,
        source: dict[str, Any],
        episode: dict[str, Any],
        camera_key: str,
    ) -> Path:
        root = Path(source["root"]).resolve()
        if source.get("segmented_media") is True:
            segment = episode.get("segment")
            if segment is None:
                segment = episode.get("camera_segments", {}).get(camera_key)
            chunk_index, file_index, _, _ = self._segment_values(segment)
            relative = format_segment_video_relative_path(
                source["video_path_template"],
                chunk_index=chunk_index,
                file_index=file_index,
                camera_key=camera_key,
            )
        else:
            relative = format_video_relative_path(
                source["video_path_template"],
                chunks_size=source["chunks_size"],
                episode_index=episode["episode_index"],
                camera_key=camera_key,
            )
        return root.joinpath(*relative.parts)

    @staticmethod
    def _segment_values(value: Any) -> tuple[int, int, float, float]:
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("invalid segmented video locator")
        chunk_index, file_index, start_seconds, end_seconds = value
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
            or not isinstance(file_index, int)
            or isinstance(file_index, bool)
            or file_index < 0
            or not isinstance(start_seconds, (int, float))
            or isinstance(start_seconds, bool)
            or not isinstance(end_seconds, (int, float))
            or isinstance(end_seconds, bool)
            or float(start_seconds) < 0
            or float(end_seconds) <= float(start_seconds)
        ):
            raise ValueError("invalid segmented video locator")
        return chunk_index, file_index, float(start_seconds), float(end_seconds)

    def _resolved_media_path(self, source: dict[str, Any], episode: dict[str, Any], camera_key: str) -> Path:
        candidate = self._metadata_path(source, episode, camera_key)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available.") from exc
        if not _contained_by(resolved, self.media_roots):
            raise VideoServiceError(HTTPStatus.FORBIDDEN, "media_path_blocked", "Video path is outside the allowlist.")
        if not resolved.is_file():
            raise VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available.")
        return resolved

    def _probe_codec(self, source: dict[str, Any], camera_key: str, path: Path) -> str | None:
        declared = source.get("source_codec")
        if isinstance(declared, str) and declared:
            return declared.lower()
        try:
            cache_key = _file_fingerprint(path)
        except OSError:
            return None
        if cache_key in self._codec_cache:
            return self._codec_cache[cache_key]
        if not self.ffprobe:
            self._codec_cache[cache_key] = None
            return None
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path.as_posix(),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
            codec = (
                result.stdout.strip().splitlines()[0].lower()
                if result.returncode == 0 and result.stdout.strip()
                else None
            )
        except (OSError, subprocess.SubprocessError):
            codec = None
        self._codec_cache[cache_key] = codec
        return codec

    def _asset_status(
        self,
        source: dict[str, Any],
        episode: dict[str, Any],
        camera_key: str,
    ) -> tuple[str | None, str, str | None, bool, str | None]:
        try:
            path = self._resolved_media_path(source, episode, camera_key)
        except (VideoServiceError, ValueError) as exc:
            if isinstance(exc, VideoServiceError) and exc.code == "media_path_blocked":
                return None, "blocked", None, False, None
            return None, "missing", None, False, None
        try:
            fingerprint = _file_fingerprint(path)
        except OSError:
            return None, "missing", None, False, None
        codec = self._probe_codec(source, camera_key, path)
        if codec and codec not in _H264_CODECS:
            if self.transcoder:
                cached = self.transcoder.cached_path(path)
                return codec, "ready" if cached else "transcode_required", "h264", bool(cached), fingerprint
            return codec, "unsupported", None, False, fingerprint
        return codec, "ready" if codec else "unverified", "h264" if codec else None, False, fingerprint

    def _stream_url(self, asset_id: str) -> str:
        path = f"/v1/media/{quote(asset_id, safe='')}"
        return f"{self.public_base_url}{path}" if self.public_base_url else path

    def list_videos(
        self,
        record_id: str,
        *,
        offset: int,
        limit: int,
        now: int | None = None,
    ) -> dict[str, Any]:
        record = self.records.get(record_id)
        if not isinstance(record, dict):
            raise VideoServiceError(HTTPStatus.NOT_FOUND, "record_not_found", "No video index exists for this record.")
        if offset < 0 or not 1 <= limit <= _MAX_PAGE_SIZE:
            raise VideoServiceError(HTTPStatus.BAD_REQUEST, "invalid_pagination", "offset or limit is invalid.")
        total = sum(len(source["episodes"]) * len(source["camera_keys"]) for source in record["sources"])
        items: list[dict[str, Any]] = []
        for position, (source_index, episode_position, camera_position, source, episode, camera_key) in enumerate(
            self._iter_items(record)
        ):
            if position < offset:
                continue
            if len(items) >= limit:
                break
            codec, playback_status, playback_codec, transcoded, fingerprint = self._asset_status(
                source, episode, camera_key
            )
            locator = {"r": record_id, "s": source_index, "p": episode_position, "c": camera_position}
            if fingerprint:
                locator["f"] = fingerprint
            asset_id = self.token_manager.issue(locator, now=now)
            item = {
                    "asset_id": asset_id,
                    "episode_index": episode.get("display_episode_index", episode["episode_index"]),
                    "camera_key": camera_key,
                    "task": episode.get("task") or record.get("native_task") or "",
                    "stream_url": (
                        self._stream_url(asset_id)
                        if playback_status in {"ready", "unverified", "transcode_required"}
                        else None
                    ),
                    "source_codec": codec or "unknown",
                    "playback_codec": playback_codec or "unknown",
                    "playback_status": playback_status,
                    "transcoded": transcoded,
                }
            if source.get("segmented_media") is True:
                segment = episode.get("segment")
                if segment is None:
                    segment = episode["camera_segments"][camera_key]
                _, _, start_seconds, end_seconds = self._segment_values(segment)
                item["start_seconds"] = start_seconds
                item["end_seconds"] = end_seconds
            items.append(item)
        return {
            "record_id": record_id,
            "video_status": record.get("video_status", record.get("status", "unmapped")),
            "total": total,
            "indexed_total": total,
            "source_total": record.get("total_video_count", total),
            "indexed_episode_count": record.get("indexed_episode_count", 0),
            "source_episode_count": record.get("total_episode_count", 0),
            "offset": offset,
            "limit": limit,
            "items": items,
            "truncated": bool(record.get("truncated")),
            "video_note": record.get("video_note") if isinstance(record.get("video_note"), str) else None,
        }

    def resolve_asset(self, token: str, *, now: int | None = None) -> MediaAsset:
        try:
            locator = self.token_manager.verify(token, now=now)
        except InvalidAssetToken as exc:
            raise VideoServiceError(HTTPStatus.FORBIDDEN, "invalid_asset", "Video link is invalid or expired.") from exc
        record_id = locator.get("r")
        indexes = [locator.get(key) for key in ("s", "p", "c")]
        if not isinstance(record_id, str) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in indexes
        ):
            raise VideoServiceError(HTTPStatus.FORBIDDEN, "invalid_asset", "Video link is invalid or expired.")
        record = self.records.get(record_id)
        if not isinstance(record, dict):
            raise VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available.")
        source_index, episode_position, camera_position = indexes
        try:
            source = record["sources"][source_index]
            episode = source["episodes"][episode_position]
            camera_key = source["camera_keys"][camera_position]
        except (IndexError, KeyError, TypeError):
            raise VideoServiceError(HTTPStatus.FORBIDDEN, "invalid_asset", "Video link is invalid or expired.") from None
        path = self._resolved_media_path(source, episode, camera_key)
        try:
            fingerprint = _file_fingerprint(path)
        except OSError as exc:
            raise VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available.") from exc
        if locator.get("f") != fingerprint:
            raise VideoServiceError(HTTPStatus.FORBIDDEN, "invalid_asset", "Video link is invalid or expired.")
        codec = self._probe_codec(source, camera_key, path)
        if codec and codec not in _H264_CODECS:
            if not self.transcoder:
                raise VideoServiceError(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_codec",
                    "This video needs an H.264 compatibility transcode before browser playback.",
                )
            path = self.transcoder.materialize(path)
            codec = "h264"
        return MediaAsset(path, record_id, episode["episode_index"], camera_key, codec)


class VideoRequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter for :class:`VideoService`."""

    protocol_version = "HTTP/1.1"
    service: VideoService

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        if origin not in self.service.cors_origins:
            self._send_error(VideoServiceError(HTTPStatus.FORBIDDEN, "origin_forbidden", "Origin is not allowed."))
            return
        requested_method = self.headers.get("Access-Control-Request-Method", "GET").upper()
        if requested_method not in {"GET", "HEAD"}:
            self._send_error(
                VideoServiceError(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Method is not allowed.")
            )
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers(origin)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._route(head_only=False)

    def do_HEAD(self) -> None:
        self._route(head_only=True)

    def _route(self, *, head_only: bool) -> None:
        origin = self.headers.get("Origin")
        if not self.service.origin_allowed(origin):
            self._send_error(VideoServiceError(HTTPStatus.FORBIDDEN, "origin_forbidden", "Origin is not allowed."))
            return
        parsed = urlsplit(self.path)
        media_match = _MEDIA_ROUTE.fullmatch(parsed.path)
        if media_match:
            self._serve_media(unquote(media_match.group(1)), origin=origin, head_only=head_only)
            return
        record_match = _RECORD_ROUTE.fullmatch(parsed.path)
        if record_match:
            if head_only:
                self._send_error(
                    VideoServiceError(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Use GET for video lists."),
                    origin=origin,
                    head_only=True,
                )
                return
            self._serve_record_videos(unquote(record_match.group(1)), parsed.query, origin=origin)
            return
        self._send_error(VideoServiceError(HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found."), origin=origin)

    def _serve_record_videos(self, record_id: str, query: str, *, origin: str | None) -> None:
        try:
            parameters = parse_qs(query, keep_blank_values=True, strict_parsing=False)
            if set(parameters) - {"offset", "limit"}:
                raise ValueError
            if any(len(values) != 1 for values in parameters.values()):
                raise ValueError
            offset = int(parameters.get("offset", ["0"])[0])
            limit = int(parameters.get("limit", ["24"])[0])
            payload = self.service.list_videos(record_id, offset=offset, limit=limit)
        except ValueError:
            self._send_error(
                VideoServiceError(HTTPStatus.BAD_REQUEST, "invalid_pagination", "offset or limit is invalid."),
                origin=origin,
            )
            return
        except VideoServiceError as exc:
            self._send_error(exc, origin=origin)
            return
        self._send_json(HTTPStatus.OK, payload, origin=origin)

    def _serve_media(self, asset_id: str, *, origin: str | None, head_only: bool) -> None:
        try:
            asset = self.service.resolve_asset(asset_id)
            size = asset.path.stat().st_size
            selected_range = parse_byte_range(self.headers.get("Range"), size)
        except RangeNotSatisfiable:
            try:
                size = asset.path.stat().st_size
            except (NameError, OSError):
                size = 0
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self._send_common_headers(origin)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        except (OSError, VideoServiceError) as exc:
            error = (
                exc
                if isinstance(exc, VideoServiceError)
                else VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available.")
            )
            self._send_error(error, origin=origin, head_only=head_only)
            return

        start, end = selected_range if selected_range else (0, size - 1)
        response_status = HTTPStatus.PARTIAL_CONTENT if selected_range else HTTPStatus.OK
        content_length = end - start + 1
        try:
            handle = asset.path.open("rb")
        except OSError:
            self._send_error(
                VideoServiceError(HTTPStatus.NOT_FOUND, "media_missing", "Video file is not available."),
                origin=origin,
                head_only=head_only,
            )
            return
        with handle:
            self.send_response(response_status)
            self._send_common_headers(origin)
            self.send_header("Content-Type", mimetypes.guess_type(asset.path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "private, max-age=300")
            if selected_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return
            handle.seek(start)
            remaining = content_length
            try:
                while remaining:
                    chunk = handle.read(min(_STREAM_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        origin: str | None,
        head_only: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._send_common_headers(origin)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_error(
        self,
        error: VideoServiceError,
        *,
        origin: str | None = None,
        head_only: bool = False,
    ) -> None:
        self._send_json(
            error.status,
            {"error": {"code": error.code, "message": error.message}},
            origin=origin,
            head_only=head_only,
            extra_headers=error.headers,
        )

    def _send_common_headers(self, origin: str | None) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Vary", "Origin")
        if origin in self.service.cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Expose-Headers", _EXPOSE_HEADERS)

    def log_message(self, message_format: str, *args: Any) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/v1/media/"):
            path = "/v1/media/<redacted>"
        print(f"{self.client_address[0]} {self.command} {path}")


def _read_secret(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
            raise ValueError("secret file must be a small regular file")
        secret = path.read_bytes().strip()
    except OSError as exc:
        raise ValueError("secret file is unavailable") from exc
    if len(secret) < 16:
        raise ValueError("secret file must contain at least 16 bytes")
    return secret


def _resolve_tool(value: str) -> str | None:
    if value == "none":
        return None
    return shutil.which(value) if value != "auto" else shutil.which("ffmpeg")


def _resolve_ffprobe(value: str) -> str | None:
    if value == "none":
        return None
    return shutil.which(value) if value != "auto" else shutil.which("ffprobe")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="video-index.json generated by build_video_index.py")
    parser.add_argument(
        "--secret-file",
        type=Path,
        required=True,
        help="Private HMAC secret file (chmod 600 recommended)",
    )
    parser.add_argument("--cors-origin", action="append", required=True, help="Exact allowed browser origin; repeatable")
    parser.add_argument(
        "--media-root",
        action="append",
        type=Path,
        help="Allowed resolved media root; repeat for native roots targeted by trusted video symlinks",
    )
    parser.add_argument("--public-base-url", help="Public HTTPS media origin used in generated stream URLs")
    parser.add_argument("--token-ttl", type=int, default=_DEFAULT_TOKEN_TTL_SECONDS)
    parser.add_argument("--ffprobe", default="auto", help="ffprobe path, 'auto', or 'none'")
    parser.add_argument("--ffmpeg", default="auto", help="ffmpeg path, 'auto', or 'none'")
    parser.add_argument("--transcode-cache", type=Path, help="H.264 compatibility cache outside all dataset roots")
    parser.add_argument("--transcode-workers", type=int, default=1)
    parser.add_argument("--transcode-threads", type=int, default=4)
    parser.add_argument("--transcode-timeout", type=int, default=600, help="Leader ffmpeg timeout in seconds")
    parser.add_argument(
        "--transcode-wait-timeout",
        type=int,
        default=120,
        help="Same-asset follower wait timeout in seconds",
    )
    parser.add_argument("--transcode-cache-max-gib", type=float, default=100.0)
    parser.add_argument("--transcode-min-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind localhost and expose only through a trusted HTTPS proxy",
    )
    parser.add_argument("--port", type=int, default=8790)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ffprobe = _resolve_ffprobe(args.ffprobe)
    ffmpeg = _resolve_tool(args.ffmpeg)
    service = VideoService.from_file(
        args.index.resolve(),
        secret=_read_secret(args.secret_file.resolve()),
        media_roots=args.media_root,
        cors_origins=args.cors_origin,
        public_base_url=args.public_base_url,
        token_ttl_seconds=args.token_ttl,
        ffprobe=ffprobe,
        ffmpeg=ffmpeg,
        transcode_cache=args.transcode_cache,
        transcode_workers=args.transcode_workers,
        transcode_threads=args.transcode_threads,
        transcode_timeout_seconds=args.transcode_timeout,
        transcode_wait_timeout_seconds=args.transcode_wait_timeout,
        transcode_cache_max_bytes=int(args.transcode_cache_max_gib * 1024**3),
        transcode_min_free_bytes=int(args.transcode_min_free_gib * 1024**3),
    )
    VideoRequestHandler.service = service
    with ThreadingHTTPServer((args.host, args.port), VideoRequestHandler) as server:
        print(f"Dataset video API: http://{args.host}:{args.port}")
        print(f"CORS origins: {', '.join(sorted(service.cors_origins))}")
        print(f"H.264 transcode cache: {'enabled' if service.transcoder else 'disabled'}")
        server.serve_forever()


if __name__ == "__main__":
    main()
