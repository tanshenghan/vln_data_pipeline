"""Read source frame counts without loading entire trajectory corpora into RAM."""
from functools import lru_cache
import gzip
from pathlib import Path
import shutil
import tempfile
import ijson
from catalog.vln_catalog import read_rows
from data_pipeline.vlnce_source import source_paths, rxr_replay_actions
from .settings import settings


@lru_cache(maxsize=12)
def inventory(episode_path, gt_path, episode_stamp, gt_stamp, dataset):
    records = {}
    # Native RxR annotations contain NaN speech timestamps. The catalog reader
    # streams these original episodes without requiring strict JSON numbers;
    # timing fields are not involved in frame counts or storage estimates.
    for row in read_rows(Path(episode_path)):
        eid = str(row["episode_id"])
        if eid in records:
            raise ValueError("源数据存在重复 episode ID")
        records[eid] = {"scene": Path(row["scene_id"]).stem,
                        "scene_path": row["scene_id"],
                        "language": row["instruction"].get("language"), "frames": None}
    with gzip.open(gt_path, "rb") as stream:
        for eid, row in ijson.kvitems(stream, ""):
            if eid in records:
                actions = row["actions"]
                if dataset == "rxr":
                    actions = rxr_replay_actions(actions)
                if not actions or actions[-1] != 0 or any(a not in (1, 2, 3) for a in actions[:-1]):
                    raise ValueError(f"专家动作或 STOP 无效：{eid}")
                records[eid]["frames"] = len(actions)
    return records


def estimate_gt(cfg, *, storage_root=None, verify_scenes=True):
    if cfg["dataset"] == "vlnverse":
        from catalog.vlnverse_scenes import usd_path
        from data_pipeline.vlnverse_source import replay_plan
        selected = {}
        for record in cfg.get("source_records", []):
            if verify_scenes:
                scene = usd_path(settings.vlnverse_scene_roots, record["scene"])
                # Platform downloads become available only after every dependency is verified.
                if not (scene.parent / "platform_asset_manifest.json").is_file():
                    raise ValueError("场景资源下载校验尚未完成：" + record["scene"])
            selected[record["episode_id"]] = {"scene": record["scene"], "frames": (len(replay_plan(record)["poses_source"]) if record["has_reference_gt"] else cfg.get("lightnav_max_steps", 500)+1)}
        if not selected:
            raise ValueError("缺少已冻结的 VLNVerse 参考轨迹")
        scenes = {r["scene"] for r in selected.values()}
    else:
        paths = source_paths(settings.dataset_root(cfg["dataset"]), cfg["dataset"], cfg["split"], cfg["role"])
        records = inventory(*paths, *((p.stat().st_mtime_ns, p.stat().st_size) for p in paths), cfg["dataset"])
        wanted = None if cfg["full_dataset"] else set(cfg["episode_ids"])
        selected = {eid: row for eid, row in records.items()
                    if (wanted is None or eid in wanted) and
                    (cfg["languages"] == ["all"] or row["language"] in cfg["languages"])}
        if not selected or (wanted is not None and wanted != set(selected)):
            raise ValueError("容量预检未找到完整的所选源数据")
        if any(row["frames"] is None for row in selected.values()):
            raise ValueError("所选 episode 缺少专家 GT")
        scenes = {row["scene"] for row in selected.values()}
        if cfg["dataset"] == "scalevln" and verify_scenes:
            from data_pipeline.scalevln_source import scene_ready
            for scene_id in {row["scene_path"] for row in selected.values()}:
                if not scene_ready(settings.scenes_root, scene_id):
                    raise ValueError("ScaleVLN 场景或 navmesh 尚未就绪：" + scene_id)
        for name in scenes if verify_scenes and cfg["dataset"] != "scalevln" else []:
            scene = settings.scenes_root / "mp3d" / name / (name + ".glb")
            if not scene.is_file() or not scene.with_suffix(".navmesh").is_file():
                raise ValueError(f"场景尚未就绪：{name}")
    destination = storage_root or settings.runs_root
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=destination) as probe:
            probe.write(b"output permission check")
            probe.flush()
    except OSError as exc:
        raise ValueError("采集输出目录不可写") from exc
    frames = sum(row["frames"] for row in selected.values())
    # RGB+float32 depth+boolean validity, labels/metadata, scratch/export/merge copies.
    # This is conservative uncompressed planning capacity, never a promised final size.
    per_frame = 470 * 280 * (3 + 4 + 1) + 65536
    copies = 3 if cfg["full_dataset"] else 2
    required = frames * per_frame * copies / 1024**3 + settings.minimum_free_gb
    free = shutil.disk_usage(destination).free / 1024**3
    return {"episodes": len(selected), "scenes": len(scenes), "frames": frames,
            "estimated_peak_gb": round(required, 2), "free_gb": round(free, 2),
            "can_start": free >= required, "gpu_id": settings.gpu_id,
            "basis": "未压缩 RGB/深度/有效性、标签及采集/导出/合并副本的保守容量预算，含保留空间；实际大小取决于压缩率"}
