import json
from pathlib import Path
from catalog.vln_catalog import connect, where_clause, SUMMARY_COLUMNS, summary_record, overview
from .settings import settings
from catalog import vlnverse_data, vlnverse_scenes


def browse(params, limit=30, offset=0):
    if params.get("dataset", "").lower() == "vlnverse":
        return vlnverse_data.browse(settings.dataset_root("vlnverse"), settings.vlnverse_scene_roots, params, limit, offset)
    if not settings.catalog_db.exists():
        return {"items": [], "stats": {"records": 0}, "facets": {}, "index_ready": False}
    db = connect(settings.catalog_db)
    try:
        clause, values = where_clause(params)
        items = [summary_record(r) for r in db.execute(
            f"SELECT {SUMMARY_COLUMNS} FROM episodes{clause} ORDER BY dataset,split,scene,episode_id,id LIMIT ? OFFSET ?",
            [*values, limit, offset])]
        if any(r["dataset"] == "ScaleVLN" for r in items):
            from data_pipeline.scalevln_source import availability
            ready = {Path(p).stem for p in availability(settings.dataset_root("scalevln"), settings.scenes_root)["scene_paths"]}
            for item in items:
                if item["dataset"] == "ScaleVLN":
                    item["scene_ready"] = item["scene"] in ready
        for item in items:
            if item["dataset"] == "SRDF":
                from data_pipeline.srdf_source import COLLECTION_REASON
                item.update(can_collect=False, trajectory_kind="discrete_viewpoint_reference_only",
                            collection_reason=COLLECTION_REASON)
        return {"items": items, **overview(db, params), "index_ready": True, "offset": offset, "limit": limit}
    finally:
        db.close()


def detail(record_id):
    if record_id.startswith("vlnverse_"):
        return next((r for r in vlnverse_data.records(settings.dataset_root("vlnverse")) if r["id"] == record_id), None)
    db = connect(settings.catalog_db)
    try:
        row = db.execute("SELECT detail FROM episodes WHERE id=?", (record_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        db.close()


def resolve_selection(cfg):
    if cfg["dataset"] == "srdf":
        from data_pipeline.srdf_source import COLLECTION_REASON
        raise ValueError(COLLECTION_REASON)
    if cfg["dataset"] == "vlnverse":
        return resolve_vlnverse(cfg)
    db = connect(settings.catalog_db)
    try:
        params = {"dataset": {"r2r": "R2R", "rxr": "RxR", "scalevln": "ScaleVLN"}[cfg["dataset"]], "split": cfg["split"], "role": cfg["role"]}
        clause, values = where_clause(params)
        if cfg["dataset"] == "scalevln" and not cfg["episode_ids"] and not cfg["source_ids"]:
            from data_pipeline.scalevln_source import availability
            ready = availability(settings.dataset_root("scalevln"), settings.scenes_root)
            scenes = [Path(p).stem for p in ready["scene_paths"]]
            if not scenes:
                raise ValueError("ScaleVLN 尚无已就绪的 HM3D 场景和 navmesh")
            clause += " AND scene IN (" + ",".join("?" for _ in scenes) + ")"
            values += scenes
        rows = db.execute("SELECT id,episode_id,scene,language,instruction,detail FROM episodes" + clause + " ORDER BY scene,episode_id", values)
        requested = set(cfg["episode_ids"])
        source_ids = set(cfg["source_ids"])
        selected = []
        for row in rows:
            if cfg["languages"] != ["all"] and row["language"] not in cfg["languages"]:
                continue
            if source_ids and row["id"] not in source_ids:
                continue
            if requested and row["episode_id"] not in requested:
                continue
            selected.append(dict(row))
            if not requested and not source_ids and len(selected) >= cfg["count"]:
                break
        if not selected or (requested and requested != {r["episode_id"] for r in selected}) or (source_ids and source_ids != {r["id"] for r in selected}):
            raise ValueError("未找到匹配的源 episode；请检查数据集、split、role、语言和 ID")
        if cfg["dataset"] == "scalevln" and not requested and not source_ids and len(selected) < cfg["count"]:
            raise ValueError(f"当前 HM3D 场景仅覆盖 {len(selected)} 条 ScaleVLN 路线，请减少数量或补齐场景")
        cfg["episode_ids"] = list(dict.fromkeys(r["episode_id"] for r in selected))
        cfg["source_ids"] = [r["id"] for r in selected]
        cfg["source_records"] = [json.loads(r["detail"]) for r in selected]
        return cfg
    finally:
        db.close()


def resolve_vlnverse(cfg):
    requested, source_ids = set(cfg["episode_ids"]), set(cfg["source_ids"])
    selected = []
    for row in vlnverse_data.records(settings.dataset_root("vlnverse")):
        if row["split"] != cfg["split"] or row["role"] != cfg["role"]:
            continue
        if (requested and row["episode_id"] not in requested) or (source_ids and row["id"] not in source_ids):
            continue
        if not requested and not source_ids:
            try:
                vlnverse_scenes.usd_path(settings.vlnverse_scene_roots, row["scene"])
            except (OSError, ValueError):
                continue
        selected.append(row)
        if not requested and not source_ids and len(selected) >= cfg["count"]:
            break
    if not selected or (requested and requested != {r["episode_id"] for r in selected}) or (source_ids and source_ids != {r["id"] for r in selected}):
        raise ValueError("未找到匹配的 VLNVerse 轨迹；请从已部署场景选择样本")
    cfg["episode_ids"] = [r["episode_id"] for r in selected]
    cfg["source_ids"] = [r["id"] for r in selected]
    cfg["source_records"] = selected
    cfg["collection_strategy"] = "public_reference_else_lightnav_vlm"
    cfg["lightnav_endpoint"] = settings.lightnav_endpoint
    cfg["lightnav_model_provenance"] = (json.loads(settings.lightnav_model_manifest.read_text())
        if settings.lightnav_model_manifest and settings.lightnav_model_manifest.is_file() else settings.lightnav_model_provenance)
    return cfg
