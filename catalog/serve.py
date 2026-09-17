#!/usr/bin/env python3
"""Serve the dataset semantic tree explorer on localhost."""

from __future__ import annotations

import argparse
import gzip
import json
from functools import partial
from http import HTTPStatus
from http.client import HTTPConnection, HTTPSConnection, HTTPException
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class ExplorerRequestHandler(SimpleHTTPRequestHandler):
    """Serve static assets plus one catalog file supplied by the operator."""

    catalog_path: Path
    video_api_base_url: str | None = None
    video_api_upstream: str | None = None
    video_config_message = "视频播放器已就绪, 服务器尚未配置媒体入口."

    def end_headers(self) -> None:
        # This UI is commonly opened through a forwarded proxy URL. Prevent a
        # previously cached app.js from keeping stale root-relative API paths.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        request_path = self.path.partition("?")[0]
        if self._is_video_api_request(request_path):
            self._proxy_video_api()
            return
        if request_path == "/catalog.json":
            self._serve_catalog()
            return
        if request_path == "/catalog.js":
            self._serve_catalog_script()
            return
        if request_path == "/video-config.json":
            self._serve_video_config()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        request_path = self.path.partition("?")[0]
        if self._is_video_api_request(request_path):
            self._proxy_video_api(head_only=True)
            return
        super().do_HEAD()

    def do_OPTIONS(self) -> None:
        request_path = self.path.partition("?")[0]
        if not self._is_video_api_request(request_path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _is_video_api_request(self, request_path: str) -> bool:
        return bool(self.video_api_upstream) and (
            request_path == "/video-api" or request_path.startswith("/video-api/")
        )

    def _proxy_video_api(self, *, head_only: bool = False) -> None:
        parsed_request = urlsplit(self.path)
        upstream_path = parsed_request.path.removeprefix("/video-api") or "/"
        if parsed_request.query:
            upstream_path = f"{upstream_path}?{parsed_request.query}"
        parsed_upstream = urlsplit(self.video_api_upstream or "")
        connection_class = HTTPSConnection if parsed_upstream.scheme == "https" else HTTPConnection
        connection = connection_class(parsed_upstream.hostname, parsed_upstream.port, timeout=660)
        request_headers = {
            name: value
            for name in ("Accept", "Range", "If-Range", "If-None-Match", "If-Modified-Since", "User-Agent")
            if (value := self.headers.get(name))
        }
        try:
            connection.request("HEAD" if head_only else "GET", upstream_path, headers=request_headers)
            response = connection.getresponse()
        except (OSError, HTTPException):
            connection.close()
            payload = json.dumps(
                {"error": {"code": "video_upstream_unavailable", "message": "视频服务暂时不可用。"}},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(HTTPStatus.BAD_GATEWAY)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)
            return

        self.send_response(response.status, response.reason)
        forwarded_headers = {
            "accept-ranges",
            "content-disposition",
            "content-length",
            "content-range",
            "content-type",
            "etag",
            "last-modified",
            "retry-after",
            "x-content-type-options",
        }
        for name, value in response.getheaders():
            if name.lower() in forwarded_headers:
                self.send_header(name, value)
        self.end_headers()
        if not head_only:
            try:
                while chunk := response.read(1024 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        connection.close()

    def _serve_catalog(self) -> None:
        try:
            payload = self.catalog_path.read_bytes()
            json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            message = f"catalog unavailable: {exc}\n".encode()
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        self._serve_runtime_payload(payload, "application/json; charset=utf-8")

    def _serve_catalog_script(self) -> None:
        try:
            catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            serialized = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
            serialized = serialized.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            payload = f"globalThis.__DATASET_EXPLORER_CATALOG__={serialized};\n".encode("utf-8")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            message = f"catalog unavailable: {exc}\n".encode()
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        self._serve_runtime_payload(payload, "application/javascript; charset=utf-8")

    def _serve_runtime_payload(self, payload: bytes, content_type: str) -> None:
        """Send large generated assets with browser-negotiated gzip compression."""

        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        if accepts_gzip:
            payload = gzip.compress(payload, compresslevel=6)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Vary", "Accept-Encoding")
        if accepts_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_video_config(self) -> None:
        payload = json.dumps(
            {
                "schema_version": 1,
                "enabled": bool(self.video_api_base_url),
                "api_base_url": self.video_api_base_url or "",
                "message": self.video_config_message,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    catalog_group = parser.add_mutually_exclusive_group(required=True)
    catalog_group.add_argument("--catalog", type=Path, help="catalog.json generated by build_catalog.py")
    catalog_group.add_argument("--vln-db", type=Path, help="VLN SQLite index generated by build_vln_catalog.py")
    parser.add_argument("--vln-geometry", type=Path, help="GT/map sidecar; defaults to vln-geometry.sqlite beside the VLN index")
    parser.add_argument("--vln-videos", type=Path, help="LeRobot GT video index; defaults to vln-videos.sqlite beside the VLN index")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Keep the default when using SSH forwarding.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--video-api-base-url",
        help="Browser-visible video API origin, for example http://127.0.0.1:8790 when both ports are forwarded.",
    )
    parser.add_argument(
        "--video-api-upstream",
        help="Internal video API origin proxied below ./video-api, for example http://127.0.0.1:8790.",
    )
    parser.add_argument("--video-config-message")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.vln_db:
        try:
            from .vln_serve import serve
        except ImportError:
            from vln_serve import serve
        serve(args.vln_db, args.host, args.port, args.vln_geometry, args.vln_videos)
        return
    catalog_path = args.catalog.resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"catalog does not exist: {catalog_path}")
    video_api_base_url = args.video_api_base_url.rstrip("/") if args.video_api_base_url else None
    video_api_upstream = args.video_api_upstream.rstrip("/") if args.video_api_upstream else None
    if video_api_base_url and video_api_upstream:
        raise ValueError("--video-api-base-url and --video-api-upstream cannot be combined")
    if video_api_base_url:
        parsed_video_url = urlsplit(video_api_base_url)
        if parsed_video_url.scheme not in {"http", "https"} or not parsed_video_url.netloc:
            raise ValueError("--video-api-base-url must be an absolute HTTP(S) URL")
    if video_api_upstream:
        parsed_upstream = urlsplit(video_api_upstream)
        if (
            parsed_upstream.scheme not in {"http", "https"}
            or not parsed_upstream.hostname
            or parsed_upstream.username
            or parsed_upstream.password
            or parsed_upstream.path not in {"", "/"}
            or parsed_upstream.query
            or parsed_upstream.fragment
        ):
            raise ValueError("--video-api-upstream must be an HTTP(S) origin without credentials or a path")
        video_api_base_url = "./video-api"
    web_root = Path(__file__).with_name("web")
    ExplorerRequestHandler.catalog_path = catalog_path
    ExplorerRequestHandler.video_api_base_url = video_api_base_url
    ExplorerRequestHandler.video_api_upstream = video_api_upstream
    ExplorerRequestHandler.video_config_message = args.video_config_message or (
        "视频服务已通过当前页面接通。" if video_api_upstream else ExplorerRequestHandler.video_config_message
    )
    handler = partial(ExplorerRequestHandler, directory=web_root.as_posix())
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Dataset semantic explorer: http://{args.host}:{args.port}")
        print(f"Catalog: {catalog_path}")
        if video_api_upstream:
            print(f"Video API proxy: ./video-api -> {video_api_upstream}")
        server.serve_forever()


if __name__ == "__main__":
    main()
