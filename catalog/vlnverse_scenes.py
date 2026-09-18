"""Read-only discovery of locally configured VLNVerse USD scene assets."""
import json
from pathlib import Path
import re


def scene_directory(root, scene_id):
    if root is None or not re.fullmatch(r"kujiale_[0-9]+(?:_fix)?", scene_id):
        raise ValueError("无效的 Kujiale 场景 ID")
    for entry in root if isinstance(root, (list, tuple)) else [root]:
        if entry is None:
            continue
        base = Path(entry).resolve()
        directory = (base / scene_id).resolve()
        if directory.is_relative_to(base) and directory.is_dir():
            return directory
    raise FileNotFoundError(scene_id)


def usd_path(root, scene_id):
    directory = scene_directory(root, scene_id)
    for name in (scene_id + ".usda", "start_result_navigation.usd"):
        path = (directory / name).resolve()
        if path.is_relative_to(directory) and path.is_file():
            return path
    raise FileNotFoundError("USD 主场景文件缺失")


def media_path(root, scene_id, kind):
    directory = scene_directory(root, scene_id)
    names = {"thumbnail": f".thumbs/256x256/{scene_id}.usda.png", "occupancy": "occupancy.png"}
    if kind not in names:
        raise ValueError("不支持的场景媒体类型")
    path = (directory / names[kind]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise FileNotFoundError(kind)
    return path


def detail(root, scene_id):
    directory = scene_directory(root, scene_id)
    usd = usd_path(root, scene_id)
    with usd.open(errors="replace") as stream:
        header = stream.read(16384)
    axis = re.search(r'upAxis\s*=\s*"([XYZ])"', header)
    units = re.search(r'metersPerUnit\s*=\s*([0-9.eE+-]+)', header)
    rooms = json.loads((directory / "rooms.json").read_text()) if (directory / "rooms.json").is_file() else []
    occupancy = json.loads((directory / "occupancy.json").read_text()) if (directory / "occupancy.json").is_file() else None
    media = {}
    for kind in ("thumbnail", "occupancy"):
        try:
            media_path(root, scene_id, kind)
            media[kind] = f"/api/scenes/{scene_id}/media/{kind}"
        except FileNotFoundError:
            media[kind] = None
    return {"id": scene_id, "dataset": "VLNVerse", "family": "Kujiale", "format": "USD",
            "up_axis": axis.group(1) if axis else None, "meters_per_unit": float(units.group(1)) if units else None,
            "rooms": rooms, "room_count": len(rooms), "room_types": sorted({r.get("room_type", "unknown") for r in rooms}),
            "occupancy": occupancy, "media": media, "source": "https://huggingface.co/datasets/Eyz/VLNVerse_scene",
            "asset_status": "主场景文件已发现；完整渲染依赖在采集预检中验证"}


def discover(root):
    if isinstance(root, (list, tuple)):
        found = {}
        configured = False
        for entry in root:
            result = discover(entry)
            configured |= result["configured"]
            for item in result["items"]:
                found.setdefault(item["id"], item)
        return {"items": [found[key] for key in sorted(found)], "configured": configured, "total": len(found)}
    if root is None or not Path(root).is_dir():
        return {"items": [], "configured": False, "total": 0}
    items = []
    for path in sorted(Path(root).glob("kujiale_*")):
        try:
            record = detail(root, path.name)
            items.append({k: record[k] for k in ("id", "dataset", "family", "format", "up_axis", "meters_per_unit", "room_count", "room_types", "media")})
        except (ValueError, OSError):
            continue
    return {"items": items, "configured": True, "total": len(items)}
