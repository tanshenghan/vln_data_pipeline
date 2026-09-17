"""HTTP endpoints for the VLN workspace; assets are served only by indexed ID."""
from __future__ import annotations

import json
import mimetypes
import sqlite3
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

try:
    from .vln_catalog import SCHEMA_VERSION, SUMMARY_COLUMNS, connect, json_text, overview, summary_record, where_clause
    from .video_serve import RangeNotSatisfiable, parse_byte_range
    from .vln_ground_truth import geometry_for_record, open_geometry
    from .vln_gt_video import GTVideoService
except ImportError:
    from vln_catalog import SCHEMA_VERSION, SUMMARY_COLUMNS, connect, json_text, overview, summary_record, where_clause
    from video_serve import RangeNotSatisfiable, parse_byte_range
    from vln_ground_truth import geometry_for_record, open_geometry
    from vln_gt_video import GTVideoService


class VLNRequestHandler(SimpleHTTPRequestHandler):
    database: Path
    geometry_path: Path | None = None
    gt_videos: GTVideoService | None = None

    def media_column(self):
        return "id IN (SELECT record_id FROM gt_videos.records)" if self.gt_videos else "has_media"

    def geometry_db(self, catalog):
        path = self.geometry_path or self.database.with_name("vln-geometry.sqlite")
        geometry = open_geometry(path)
        if geometry is not None:
            meta = json.loads(catalog.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
            info = json.loads(geometry.execute("SELECT value FROM metadata WHERE key='geometry'").fetchone()[0])
            if info.get("catalog_id") != meta["catalog_id"] or info.get("catalog_generated_at") != meta["generated_at"]:
                geometry.close()
                return None
        return geometry

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, value, status=200):
        payload = json_text(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        """Stream a manifest with local reviews; nothing is stored server-side."""
        parsed = urlsplit(self.path)
        if parsed.path == "/vln-api/gt-video":
            params = parse_qs(parsed.query)
            record_id = params.get("id", [""])[-1]
            status = self.gt_videos.request(record_id, retry=params.get("retry") == ["1"]) if self.gt_videos else {"state": "unavailable"}
            self.send_json(status)
            return
        if urlsplit(self.path).path != "/vln-api/export":
            self.send_json({"error": "未找到该接口"}, 404)
            return
        db = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("导出筛选和备注请求为空或超过 16 MiB")
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(parse_qs(body).get("payload", [""])[0])
            params = payload.get("filters", {})
            reviews = payload.get("reviews", {})
            if not isinstance(params, dict) or not all(isinstance(v, str) for v in params.values()):
                raise ValueError("导出筛选格式无效")
            if not isinstance(reviews, dict):
                raise ValueError("人工标记格式无效")
            for review in reviews.values():
                if (not isinstance(review, dict) or review.get("decision") not in {"unreviewed", "keep", "exclude", "review"}
                        or not isinstance(review.get("note", ""), str)):
                    raise ValueError("人工标记格式无效")
            db = connect(self.database)
            if self.gt_videos:
                self.gt_videos.attach(db)
            self.export(db, params, reviews)
        except (ValueError, UnicodeError, AttributeError) as error:
            self.send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (sqlite3.Error, OSError):
            self.send_json({"error": "导出暂时失败，请检查目录后重试。"}, 503)
        finally:
            if db is not None:
                db.close()

    def do_GET(self):
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/vln-api/"):
            if parsed.path not in {"/", "/index.html", "/app.js", "/styles.css", "/favicon.svg"}:
                self.send_error(404)
                return
            return super().do_HEAD() if self.command == "HEAD" else super().do_GET()
        params = {key: value[-1] for key, value in parse_qs(parsed.query).items()}
        db = None
        geometry = None
        try:
            db = connect(self.database)
            if self.gt_videos:
                self.gt_videos.attach(db)
            geometry = self.geometry_db(db)
            if parsed.path == "/vln-api/catalog":
                meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
                tree = [dict(row) for row in db.execute("SELECT dataset,split,scene,count(*) count FROM episodes GROUP BY dataset,split,scene ORDER BY dataset,split,scene")]
                geometry_info = json.loads(geometry.execute("SELECT value FROM metadata WHERE key='geometry'").fetchone()[0]) if geometry is not None else None
                self.send_json({**meta, **overview(db, {}, self.media_column()), "tree": tree, "geometry": geometry_info,
                                "gt_videos": self.gt_videos.stats() if self.gt_videos else None})
            elif parsed.path == "/vln-api/episodes":
                limit = min(100, max(1, int(params.get("limit", "40"))))
                offset = max(0, int(params.get("offset", "0")))
                clause, values = where_clause(params, self.media_column())
                video_order = f"({self.media_column()}) DESC," if self.gt_videos else ""
                rows = db.execute(f"SELECT {SUMMARY_COLUMNS} FROM episodes{clause} ORDER BY {video_order}dataset,split,scene,episode_id,id LIMIT ? OFFSET ?",
                                  [*values, limit, offset])
                payload = {"items": [summary_record(row) for row in rows], "limit": limit, "offset": offset}
                available = set()
                if geometry is not None and payload["items"]:
                    ids = [record["id"] for record in payload["items"]]
                    available = {row[0] for row in geometry.execute(f"SELECT record_id FROM ground_truth WHERE record_id IN ({','.join('?' for _ in ids)})", ids)}
                for record in payload["items"]:
                    record["has_gt"] = record["id"] in available
                    record["has_gt_video"] = bool(self.gt_videos and record["id"] in self.gt_videos.records)
                payload.update(overview(db, params, self.media_column()))
                self.send_json(payload)
            elif parsed.path == "/vln-api/episode":
                row = db.execute("SELECT detail FROM episodes WHERE id=?", (params.get("id", ""),)).fetchone()
                record = json.loads(row[0]) if row else {"error": "未找到该样本"}
                if row:
                    record.update(geometry_for_record(geometry, record))
                    record["gt_video"] = self.gt_videos.describe(record["id"]) if self.gt_videos else {"state": "unavailable", "reason": "尚未接入此样本的 LeRobot RGB 数据。"}
                self.send_json(record, 200 if row else 404)
            elif parsed.path == "/vln-api/gt-video":
                self.send_json(self.gt_videos.describe(params.get("id", "")) if self.gt_videos else {"state": "unavailable"})
            elif parsed.path == "/vln-api/video-progress":
                self.send_json(self.gt_videos.stats() if self.gt_videos else {"records": 0, "coverage": []})
            elif parsed.path == "/vln-api/gt-video-media":
                path = self.gt_videos.media_path(params.get("id", ""), params.get("kind", "")) if self.gt_videos else None
                if path is None:
                    self.send_json({"error": "GT 视频尚未生成或媒体不存在"}, 404)
                else:
                    self.send_media_path(path)
            elif parsed.path == "/vln-api/ground-truth":
                row = db.execute("SELECT detail FROM episodes WHERE id=?", (params.get("id", ""),)).fetchone()
                if row is None:
                    self.send_json({"error": "未找到该样本"}, 404)
                else:
                    record = json.loads(row[0])
                    ground_truth = geometry_for_record(geometry, record)["ground_truth"]
                    if not ground_truth["available"]:
                        self.send_json({"error": "该样本没有已接入的 GT", "reason": ground_truth["reason"]}, 404)
                    else:
                        payload = {key: record[key] for key in ("id", "source_id", "dataset", "version", "split", "scene", "language", "role", "episode_id", "instruction_id", "trajectory_id", "instruction", "projection_axes", "start_position", "start_rotation", "goals", "reference_path")}
                        payload["ground_truth"] = ground_truth
                        content = json_text(payload).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Disposition", f'attachment; filename="vln-gt-{record["id"]}.json"')
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        if self.command != "HEAD":
                            self.wfile.write(content)
            elif parsed.path == "/vln-api/export":
                self.export(db, params)
            elif parsed.path == "/vln-api/media":
                self.media(db, params.get("id", ""), geometry)
            else:
                self.send_json({"error": "未找到该接口"}, 404)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (sqlite3.Error, OSError):
            self.send_json({"error": "目录暂时不可用，请检查索引文件后重试。"}, 503)
        finally:
            if geometry is not None:
                geometry.close()
            if db is not None:
                db.close()

    def export(self, db, params, reviews=None):
        reviews = reviews or {}
        exported_at = datetime.now(timezone.utc).isoformat()
        clause, values = where_clause(params, self.media_column())
        rows = db.execute("SELECT detail FROM episodes" + clause + " ORDER BY dataset,split,scene,episode_id,id", values)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="vln-manifest.jsonl"')
        self.end_headers()
        if self.command == "HEAD":
            return
        for row in rows:
            record = json.loads(row[0])
            record.pop("media", None)
            record.pop("timed_instruction", None)
            record["manifest_schema_version"] = 1
            record["manifest_filters"] = params
            review = reviews.get(record["id"], {})
            record["manual_decision"] = review.get("decision", "unreviewed")
            record["manual_note"] = review.get("note", "")
            record["manual_updated_at"] = review.get("updated_at")
            record["manifest_exported_at"] = exported_at
            self.wfile.write((json_text(record) + "\n").encode("utf-8"))

    def media(self, db, asset_id, geometry=None):
        row = db.execute("SELECT path FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None and geometry is not None:
            row = geometry.execute("SELECT path FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            self.send_json({"error": "未找到该媒体"}, 404)
            return
        path = Path(row[0])
        self.send_media_path(path)

    def send_media_path(self, path):
        # The stored path was resolved during indexing; reject changed symlinks.
        if path.resolve() != path or not path.is_file():
            self.send_json({"error": "该媒体文件已不可用"}, 404)
            return
        with path.open("rb") as handle:
            size = path.stat().st_size
            try:
                selected = parse_byte_range(self.headers.get("Range"), size)
            except RangeNotSatisfiable:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = selected if selected is not None else (0, size - 1)
            self.send_response(206 if selected is not None else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if selected is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(database: Path, host: str, port: int, geometry_path: Path | None = None, videos_path: Path | None = None):
    if not database.is_file():
        raise FileNotFoundError(f"Build a VLN catalogue first: {database}")
    with connect(database) as db:
        meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported VLN catalogue schema; rebuild the index")
    VLNRequestHandler.database = database.resolve()
    VLNRequestHandler.geometry_path = geometry_path.resolve() if geometry_path else None
    videos_path = videos_path or database.with_name("vln-videos.sqlite")
    VLNRequestHandler.gt_videos = GTVideoService(videos_path, meta) if videos_path.is_file() else None
    handler = partial(VLNRequestHandler, directory=str(Path(__file__).with_name("vln_web")))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"VLN 数据浏览器: http://{host}:{server.server_port}", flush=True)
        server.serve_forever()
