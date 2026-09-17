import hashlib
import json
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from .db import Base, engine, Run, Artifact, Review, Release, session, serialize, now
from .settings import settings, ROOT
from .schemas import RunRequest, ReviewRequest
from .catalog_service import browse, detail, resolve_selection
from .preflight import health, validate_gt
from workers.tasks import execute_run, run_dir


@asynccontextmanager
async def lifespan(app):
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="VLN Data Pipeline", lifespan=lifespan)


@app.middleware("http")
async def same_origin_writes(request, call_next):
    # LAN application; prevent an unrelated website from starting expensive jobs via the user's browser.
    if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return Response("跨站写请求被拒绝", status_code=403)
    return await call_next(request)


@app.get("/api/health")
def get_health():
    result = health()
    with session() as db:
        result["runs"] = dict(db.execute(select(Run.status, func.count()).group_by(Run.status)).all())
    return result


@app.get("/api/sources")
def sources(request: Request, limit: int = 30, offset: int = 0):
    try:
        return browse(dict(request.query_params), min(100, max(1, limit)), max(0, offset))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/sources/{record_id}")
def source_detail(record_id: str):
    record = detail(record_id)
    if not record:
        raise HTTPException(404, "样本不存在")
    return record


@app.get("/api/runs")
def runs():
    with session() as db:
        return [serialize(r) for r in db.scalars(select(Run).order_by(Run.created_at.desc()).limit(200))]


@app.post("/api/runs", status_code=202)
def create_run(payload: RunRequest, idempotency_key: str = Header(min_length=8, max_length=128)):
    cfg = payload.model_dump()
    with session() as db:
        existing = db.scalar(select(Run).where(Run.request_key == idempotency_key))
        if existing:
            if existing.config.get("request_payload") != cfg:
                raise HTTPException(409, "同一请求 ID 不能更改参数")
            return serialize(existing)
    try:
        if not cfg["full_dataset"] and (cfg["kind"] == "gt" or cfg["source_ids"] or cfg["episode_ids"]):
            cfg = resolve_selection(cfg)
        if cfg["kind"] == "gt":
            validate_gt(cfg)
        elif not health()["seedance_ready"]:
            raise ValueError("Seedance API 密钥尚未配置，生成任务未提交")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    run_id = str(uuid.uuid4())
    cfg["request_payload"] = payload.model_dump()
    try:
        with session() as db:
            row = Run(id=run_id, request_key=idempotency_key, kind=cfg["kind"],
                      name=cfg["name"] or f"{cfg['dataset'].upper()} {'GT 采集' if cfg['kind']=='gt' else 'Seedance'}", config=cfg)
            db.add(row)
            db.flush()
            result = serialize(row)
    except IntegrityError:
        with session() as db:
            row = db.scalar(select(Run).where(Run.request_key == idempotency_key))
            if row.config["request_payload"] != payload.model_dump():
                raise HTTPException(409, "请求 ID 冲突")
            return serialize(row)
    try:
        execute_run.delay(run_id)
    except Exception:
        # Transaction committed; reconciler will dispatch when the broker recovers.
        result["dispatch"] = "等待队列恢复，任务已保存"
    return result


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    with session() as db:
        row = db.get(Run, run_id)
        if not row:
            raise HTTPException(404, "任务不存在")
        result = serialize(row)
        result["artifacts"] = [serialize(a) for a in db.scalars(select(Artifact).where(Artifact.run_id == run_id))]
        result["reviews"] = [serialize(r) for r in db.scalars(select(Review).where(Review.subject_id == run_id).order_by(Review.created_at.desc()))]
        result["release"] = next((serialize(r) for r in db.scalars(select(Release).where(Release.run_id == run_id))), None)
    directory = run_dir(run_id)
    if row.kind == "seedance":
        generation = directory / "generation"
        usage = generation / "usage_summary.json"
        manifest = generation / "manifest.jsonl"
        try:
            result["generation"] = {
                "usage": json.loads(usage.read_text()) if usage.exists() else None,
                "samples": [{k:r.get(k) for k in ("sample_id", "status", "quality_tier", "errors", "trajectory_report", "vlm_report", "prompt_report")}
                            for r in (json.loads(line) for line in manifest.read_text().splitlines())] if manifest.exists() else [],
            }
        except (OSError, ValueError):
            result["generation"] = {"usage": None, "samples": [], "updating": True}
    result["stages"] = [{"name":p.name, "complete":(p / "success.json").exists(),
                         "attempts":len(list(p.glob('attempt-*')))} for p in sorted((directory / "stages").glob("*"))]
    return result


@app.get("/api/runs/{run_id}/logs")
def logs(run_id: str):
    with session() as db:
        if not db.get(Run, run_id):
            raise HTTPException(404, "任务不存在")
    logs = sorted((run_dir(run_id) / "stages").glob("*/attempt-*/output.log"), key=lambda p:p.stat().st_mtime)
    pieces = []
    for path in logs[-8:]:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 16000))
            pieces.append(f"[{path.parent.parent.name} / {path.parent.name}]\n" + handle.read().decode("utf-8", errors="replace"))
    return {"text": "\n".join(pieces)}


@app.post("/api/runs/{run_id}/{action}")
def operate(run_id: str, action: str):
    if action not in {"pause", "cancel", "resume", "retry"}:
        raise HTTPException(404)
    with session() as db:
        row = db.scalar(select(Run).where(Run.id == run_id).with_for_update())
        if not row:
            raise HTTPException(404)
        if row.status in {"succeeded", "cancelled"}:
            raise HTTPException(409, "任务已结束")
        if action in {"resume", "retry"}:
            if row.status not in {"failed", "paused"}:
                raise HTTPException(409, "任务尚未停止，不能重复执行")
            row.control, row.status = "run", "queued"
        else:
            row.control = action
            if row.status == "queued":
                row.status = "paused" if action == "pause" else "cancelled"
        row.updated_at = now()
    if action in {"retry", "resume"}:
        execute_run.delay(run_id)
    return {"ok": True, "note": "暂停将在当前阶段完成后生效" if action == "pause" else "操作已记录"}


@app.post("/api/reviews/{run_id}")
def review(run_id: str, payload: ReviewRequest):
    with session() as db:
        if not db.get(Run, run_id):
            raise HTTPException(404)
        record = Review(id=str(uuid.uuid4()), subject_id=run_id, **payload.model_dump())
        db.add(record)
        db.flush()
        return serialize(record)


@app.post("/api/releases/{run_id}")
def publish(run_id: str):
    with session() as db:
        row = db.scalar(select(Run).where(Run.id == run_id).with_for_update())
        if not row or row.status != "succeeded":
            raise HTTPException(409, "仅完成的任务可以发布")
        previous = db.scalar(select(Release).where(Release.run_id == run_id))
        if previous:
            return serialize(previous)
        review = db.scalar(select(Review).where(Review.subject_id == run_id).order_by(Review.created_at.desc()).limit(1))
        if not review or review.decision != "keep":
            raise HTTPException(409, "请先审核并保留该任务产物")
        artifacts = db.scalars(select(Artifact).where(Artifact.run_id == run_id)).all()
        manifest = {"run": serialize(row), "artifacts": [serialize(a) for a in artifacts], "files": {}}
        for a in artifacts:
            path = Path(a.path)
            paths = path.rglob("*") if path.is_dir() else [path]
            for file in paths:
                if file.is_file():
                    h = hashlib.sha256()
                    with file.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024*1024), b""):
                            h.update(block)
                    manifest["files"][str(file.relative_to(run_dir(run_id)))] = h.hexdigest()
        release = Release(id=str(uuid.uuid4()), run_id=run_id, name=row.name, manifest=manifest)
        db.add(release)
        db.flush()
        return serialize(release)


@app.get("/api/artifacts/{artifact_id}")
def artifact(artifact_id: str, request: Request, file: str = ""):
    with session() as db:
        row = db.get(Artifact, artifact_id)
        if not row:
            raise HTTPException(404)
        path = Path(row.path).resolve()
        base = run_dir(row.run_id).resolve()
        if file:
            candidate = (path / file).resolve()
            if not path.is_dir() or not candidate.is_relative_to(path):
                raise HTTPException(403)
            path = candidate
        if not path.is_relative_to(base) or not path.exists():
            raise HTTPException(404)
        if path.is_dir():
            return {"files": [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]}
        return FileResponse(path, filename=path.name, content_disposition_type="inline")


@app.get("/api/runs/{run_id}/manifest")
def export_manifest(run_id: str):
    result = run_detail(run_id)
    return Response(json.dumps(result, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'})


@app.api_route("/atlas/{path:path}", methods=["GET", "POST", "HEAD"])
async def atlas(path: str, request: Request):
    client = httpx.AsyncClient(timeout=60)
    try:
        upstream = await client.send(client.build_request(
            request.method, settings.atlas_url + "/" + path,
            params=request.query_params, content=await request.body(),
            headers={k:v for k,v in request.headers.items() if k.lower() in
                     {"range", "if-range", "content-type", "if-none-match", "if-modified-since"}}), stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(503, "数据浏览服务正在启动")

    async def close():
        await upstream.aclose()
        await client.aclose()

    headers = {k:v for k,v in upstream.headers.items() if k.lower() in
               {"content-type", "content-range", "accept-ranges", "content-disposition",
                "content-length", "content-encoding", "etag", "last-modified"}}
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                             headers=headers, background=BackgroundTask(close))


@app.api_route("/vln-api/{path:path}", methods=["GET", "POST", "HEAD"])
async def atlas_api(path: str, request: Request):
    return await atlas("vln-api/" + path, request)


app.mount("/atlas-assets", StaticFiles(directory=ROOT / "catalog/vln_web"), name="atlas-assets")
app.mount("/", StaticFiles(directory=ROOT / "apps/web", html=True), name="web")
