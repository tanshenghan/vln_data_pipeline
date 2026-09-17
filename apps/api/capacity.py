"""Read source frame counts without loading entire trajectory corpora into RAM."""
from functools import lru_cache
import gzip
from pathlib import Path
import shutil
import tempfile
import ijson
from data_pipeline.vlnce_source import source_paths, rxr_replay_actions
from .settings import settings


@lru_cache(maxsize=12)
def inventory(episode_path, gt_path, episode_stamp, gt_stamp, dataset):
    records = {}
    with gzip.open(episode_path, "rb") as stream:
        for row in ijson.items(stream, "episodes.item"):
            eid = str(row["episode_id"])
            if eid in records:
                raise ValueError("源数据存在重复 episode ID")
            records[eid] = {"scene": Path(row["scene_id"]).stem,
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


def estimate_gt(cfg):
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
    for name in scenes:
        scene = settings.resource_root / "scene_datasets/mp3d" / name / (name + ".glb")
        if not scene.is_file() or not scene.with_suffix(".navmesh").is_file():
            raise ValueError(f"场景尚未就绪：{name}")
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=settings.runs_root) as probe:
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
    free = shutil.disk_usage(settings.runs_root).free / 1024**3
    return {"episodes": len(selected), "scenes": len(scenes), "frames": frames,
            "estimated_peak_gb": round(required, 2), "free_gb": round(free, 2),
            "can_start": free >= required, "gpu_id": settings.gpu_id,
            "basis": "未压缩 RGB/深度/有效性、标签及采集/导出/合并副本的保守容量预算，含保留空间；实际大小取决于压缩率"}
