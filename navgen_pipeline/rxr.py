"""Import RxR/Matterport episodes into the NavGen sample contract.

The source episodes contain a natural-language route and a sparse Matterport
reference path, but no RGB clip that can be sent to Seedance.  This importer
keeps those facts intact and projects the reference path onto a small local
ground plane used by the video prompt.  The projection is a declared route
prior; it is never labelled as an observed or ground-truth pose stream.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import ObjectSpec, RouteContract, SampleSpec, SceneManifest, StopCondition, TaskSpec, Waypoint


RXR_IMPORTER_VERSION = "rxr-import-v1"
DEFAULT_LOCAL_SCALE = 0.30
DEFAULT_ANCHOR = (0.8, 0.8)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point3(value: Any, name: str) -> Tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"{name} must contain [x, y, z]")
    return tuple(_finite_float(item, name) for item in value[:3])  # type: ignore[return-value]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _yaw_between(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.atan2(right[1] - left[1], right[0] - left[0])


def _localize_reference_path(
    reference_path: Sequence[Sequence[float]],
    *,
    scale: float,
    anchor: Tuple[float, float],
    duration_s: float,
    dwell_s: float = 1.2,
) -> Tuple[List[Waypoint], List[Tuple[float, float]]]:
    """Map Matterport ``(x, height, z)`` points to local ``(x, y)`` poses.

    Matterport's x axis is reversed here so the route grows away from the
    local entrance.  The transformation is uniform and therefore preserves
    the turns and relative geometry of the supplied reference path.
    """

    if len(reference_path) < 2:
        raise ValueError("RxR reference_path must contain at least two points")
    source_points = [_point3(point, "reference_path point") for point in reference_path]
    source_start_x, _, source_start_z = source_points[0]
    local_points: List[Tuple[float, float]] = []
    for source_x, _, source_z in source_points:
        point = (
            anchor[0] + (source_start_x - source_x) * scale,
            anchor[1] + (source_z - source_start_z) * scale,
        )
        if not local_points or math.hypot(point[0] - local_points[-1][0], point[1] - local_points[-1][1]) > 1e-6:
            local_points.append(point)
    if len(local_points) < 2:
        raise ValueError("RxR reference_path collapses to one local point")
    duration = _finite_float(duration_s, "duration_s")
    dwell = min(max(0.0, float(dwell_s)), max(0.0, duration - 0.1))
    moving_duration = max(0.1, duration - dwell)
    distances = [
        math.hypot(local_points[index + 1][0] - local_points[index][0], local_points[index + 1][1] - local_points[index][1])
        for index in range(len(local_points) - 1)
    ]
    total_distance = sum(distances)
    if total_distance <= 1e-9:
        raise ValueError("RxR reference_path has zero ground-plane length")
    times = [0.0]
    for distance in distances:
        times.append(times[-1] + moving_duration * distance / total_distance)
    yaws: List[float] = []
    for index, point in enumerate(local_points):
        if index == 0:
            yaw = _yaw_between(local_points[0], local_points[1])
        elif index == len(local_points) - 1:
            yaw = _yaw_between(local_points[-2], local_points[-1])
        else:
            incoming = _yaw_between(local_points[index - 1], point)
            outgoing = _yaw_between(point, local_points[index + 1])
            yaw = math.atan2(
                math.sin(incoming) + math.sin(outgoing),
                math.cos(incoming) + math.cos(outgoing),
            )
        yaws.append(yaw)
    # Keep the moving endpoint and terminal dwell as two strictly increasing
    # anchors.  This is important for both Seedance's timeline and the
    # trajectory filter's dwell check.
    moving_end_t = max(times[-2] + 1e-3, duration - dwell) if len(times) > 1 else duration
    times[-1] = moving_end_t
    waypoints = [Waypoint(t, point[0], point[1], yaws[index]) for index, (t, point) in enumerate(zip(times, local_points))]
    waypoints.append(Waypoint(duration, local_points[-1][0], local_points[-1][1], yaws[-1]))
    return waypoints, local_points


def _route_direction(points: Sequence[Tuple[float, float]], index: int) -> Tuple[float, float]:
    if len(points) < 2:
        return (1.0, 0.0)
    if index <= 0:
        left, right = points[0], points[1]
    elif index >= len(points) - 1:
        left, right = points[-2], points[-1]
    else:
        left, right = points[index - 1], points[index + 1]
    dx, dy = right[0] - left[0], right[1] - left[1]
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def _near_route(
    points: Sequence[Tuple[float, float]],
    index: int,
    *,
    side: float = 1.0,
    offset: float = 0.65,
    width: float = 8.5,
    depth: float = 5.5,
) -> Tuple[float, float]:
    """Place a semantic landmark beside (rather than on) a route point."""

    index = max(0, min(len(points) - 1, int(index)))
    x, y = points[index]
    dx, dy = _route_direction(points, index)
    candidate = (x - dy * offset * side, y + dx * offset * side)
    return (_clamp(candidate[0], 0.45, width - 0.45), _clamp(candidate[1], 0.45, depth - 0.45))


def _ahead_of_end(
    points: Sequence[Tuple[float, float]],
    *,
    distance: float = 0.9,
    width: float = 8.5,
    depth: float = 5.5,
) -> Tuple[float, float]:
    end = points[-1]
    dx, dy = _route_direction(points, len(points) - 1)
    return (_clamp(end[0] + dx * distance, 0.45, width - 0.45), _clamp(end[1] + dy * distance, 0.45, depth - 0.45))


def _landmark(
    object_id: str,
    category: str,
    position: Tuple[float, float],
    *,
    size: Tuple[float, float] = (0.55, 0.55),
    attributes: Optional[Mapping[str, Any]] = None,
    color_rgb: Optional[Tuple[int, int, int]] = None,
) -> ObjectSpec:
    return ObjectSpec(
        id=object_id,
        category=category,
        position=position,
        size=size,
        attributes=dict(attributes or {}),
        color_rgb=color_rgb,
        dynamic=False,
    )


def _scene_objects(
    points: Sequence[Tuple[float, float]],
    target_id: str,
    *,
    width: float,
    depth: float,
) -> List[ObjectSpec]:
    """Create stable visual anchors shared by the three instructions."""

    end_target = _ahead_of_end(points, width=width, depth=depth)
    objects = [
        _landmark("potted_plants_01", "potted_plants", _near_route(points, 0, side=1.0, offset=1.05, width=width, depth=depth), size=(0.65, 0.65), attributes={"count": 2, "location": "corridor entrance"}, color_rgb=(92, 145, 92)),
        _landmark("door_mat_01", "door_mat", _near_route(points, 1, side=-1.0, offset=0.35, width=width, depth=depth), size=(0.85, 0.38), attributes={"texture": "woven", "state": "on_floor"}, color_rgb=(139, 111, 82)),
        _landmark("glass_door_01", "glass_door", _near_route(points, 2, side=1.0, offset=0.6, width=width, depth=depth), size=(0.15, 1.15), attributes={"panels": 3, "state": "openable"}, color_rgb=(153, 196, 207)),
        _landmark("stairs_01", "stairs", _near_route(points, 3, side=-1.0, offset=0.85, width=width, depth=depth), size=(1.15, 0.8), attributes={"direction": "up", "side": "left"}, color_rgb=(142, 119, 94)),
        _landmark("portrait_01", "portrait", _near_route(points, 6, side=1.0, offset=0.75, width=width, depth=depth), size=(0.18, 0.72), attributes={"subject": "woman", "mounted": True}, color_rgb=(120, 91, 74)),
        _landmark("bedroom_door_01", "bedroom_door", _near_route(points, 7, side=-1.0, offset=0.62, width=width, depth=depth), size=(0.18, 1.1), attributes={"state": "open"}, color_rgb=(177, 145, 109)),
        _landmark("tiny_table_01", "side_table", _near_route(points, 7, side=1.0, offset=1.05, width=width, depth=depth), size=(0.55, 0.55), attributes={"scale": "small"}, color_rgb=(153, 108, 74)),
        _landmark("square_mirror_01", "mirror", _near_route(points, 7, side=1.0, offset=1.45, width=width, depth=depth), size=(0.12, 0.58), attributes={"shape": "square"}, color_rgb=(169, 190, 198)),
        _landmark("armoire_01", "armoire", _near_route(points, 8, side=-1.0, offset=0.85, width=width, depth=depth), size=(0.7, 0.5), attributes={"wood": "dark", "side": "left"}, color_rgb=(112, 77, 54)),
        _landmark("bed_01", "bed", _near_route(points, 8, side=1.0, offset=1.05, width=width, depth=depth), size=(1.25, 0.7), attributes={"side": "right"}, color_rgb=(177, 181, 190)),
        _landmark("wash_basin_01", "wash_basin", end_target, size=(0.55, 0.42), attributes={"room": "bathroom", "is_target": target_id == "wash_basin_01"}, color_rgb=(205, 215, 217)),
        _landmark("bedroom_door_02", "bedroom_door", _near_route(points, 9, side=-1.0, offset=0.8, width=width, depth=depth), size=(0.18, 1.1), attributes={"state": "open", "is_target": target_id == "bedroom_door_02"}, color_rgb=(184, 151, 113)),
        _landmark("bed_02", "bed", end_target, size=(1.3, 0.72), attributes={"room": "far bedroom", "is_target": target_id == "bed_02"}, color_rgb=(167, 177, 190)),
        _landmark("ottoman_01", "ottoman", _near_route(points, 9, side=1.0, offset=1.35, width=width, depth=depth), size=(0.58, 0.48), attributes={"on_floor": True}, color_rgb=(137, 109, 87)),
    ]
    # Mark exactly one visual anchor as the task target.  The target can share
    # a position with another landmark because the source episode only gives a
    # goal region; the prompt and manifest retain the ambiguity explicitly.
    for obj in objects:
        if obj.id == target_id:
            obj.attributes["is_target"] = True
    return objects


def _canonical_actions(episode_id: str) -> List[str]:
    if episode_id == "13288":
        return ["turn_around", "move_forward", "turn_right", "exit_glass_door", "turn_left", "passage", "enter_bedroom", "face_wash_basin", "stop"]
    if episode_id == "13289":
        return ["turn_around", "move_forward", "turn_right", "open_door", "turn_left", "narrow_path", "turn_left", "enter_bedroom", "walk_to_open_door", "stop"]
    return ["move_forward", "turn_right", "enter_glass_door", "follow_portrait_hallway", "enter_bedroom", "face_bathroom", "observe_far_bedroom", "stop"]


def rxr_episode_to_sample(
    episode: Mapping[str, Any],
    *,
    duration_s: float = 15.0,
    local_scale: float = DEFAULT_LOCAL_SCALE,
    anchor: Tuple[float, float] = DEFAULT_ANCHOR,
    sample_prefix: str = "rxr",
) -> SampleSpec:
    """Convert one JSON episode record into a validated :class:`SampleSpec`."""

    episode_id = str(episode.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("RxR episode is missing episode_id")
    instruction = str(episode.get("instruction", episode.get("instruction_text", ""))).strip()
    if not instruction:
        raise ValueError(f"RxR episode {episode_id} is missing instruction")
    reference_path = episode.get("reference_path")
    if not isinstance(reference_path, Sequence) or isinstance(reference_path, (str, bytes)):
        raise ValueError(f"RxR episode {episode_id} is missing reference_path")
    duration = float(max(4, min(15, round(_finite_float(duration_s, "duration_s")))))
    scale = _finite_float(local_scale, "local_scale")
    if scale <= 0:
        raise ValueError("local_scale must be positive")
    waypoints, local_points = _localize_reference_path(
        reference_path,
        scale=scale,
        anchor=anchor,
        duration_s=duration,
    )
    width, depth = 8.5, 5.5
    target_id = str(episode.get("target_id") or "bed_02")
    allowed_targets = {"wash_basin_01", "bedroom_door_02", "bed_02"}
    if target_id not in allowed_targets:
        raise ValueError(f"unsupported RxR target_id {target_id!r} for episode {episode_id}")
    objects = _scene_objects(local_points, target_id, width=width, depth=depth)
    target = next(obj for obj in objects if obj.id == target_id)
    source_start = _point3(episode.get("start_position", reference_path[0]), "start_position")
    source_goal = _point3(episode.get("goal_position", reference_path[-1]), "goal_position")
    start_yaw = waypoints[0].yaw
    task = TaskSpec(
        type="instruction_following",
        target_id=target.id,
        instruction_zh=instruction,
        instruction_original=instruction,
        canonical_actions=_canonical_actions(episode_id),
        start_pose=(waypoints[0].x, waypoints[0].y, start_yaw),
        stop_condition=StopCondition(
            distance_min_m=0.6,
            distance_max_m=1.6,
            face_target=True,
            dwell_s=1.2,
            target_visible_ratio=0.75,
        ),
        difficulty={
            "source": "RxR",
            "language": str(episode.get("language", "en")),
            "turns": max(1, len(waypoints) - 2),
            "reference_goal_radius_m": float(episode.get("goal_radius_m", 3.0)),
        },
    )
    route = RouteContract(
        waypoints=waypoints,
        robot_profile="indoor_diff_drive_v1",
        max_speed_mps=0.7,
        # The sparse RxR anchors include genuine corners.  These limits leave
        # room for the fixed smoothstep interpolator without classifying the
        # source route itself as an impossible acceleration spike.
        max_accel_mps2=1.5,
        max_angular_speed_rps=1.2,
        max_angular_accel_rps2=2.5,
        clearance_m=0.18,
        goal_region={
            "target_id": target.id,
            "center": [target.position[0], target.position[1]],
            "radius_m": float(episode.get("goal_radius_m", 3.0)),
            "source": "rxr_goal_region",
        },
        expected_duration_s=duration,
        # Exact Matterport geometry is not included in this small import file;
        # keep obstacles empty rather than inventing collision boxes.
        obstacles=[],
    )
    scene = SceneManifest(
        scene_id=str(episode.get("scene_id", "SN83YJsR3w2")),
        room_type="matterport_indoor_multi_room",
        objects=objects,
        obstacles=[],
        regions=[],
        width_m=width,
        depth_m=depth,
        camera_height_m=1.2,
        fps=24,
        metadata={
            "generator": RXR_IMPORTER_VERSION,
            "source_dataset": "RxR / VLN-CE guide trajectories",
            "source_scene_asset": str(episode.get("scene_asset", "mp3d/SN83YJsR3w2/SN83YJsR3w2.glb")),
            "geometry_source": "rxr_reference_path_only",
            "coordinate_transform": "local_x=anchor_x+(source_start_x-source_x)*scale; local_y=anchor_y+(source_z-source_start_z)*scale",
            "local_scale": scale,
            "local_anchor": list(anchor),
            "source_start_position": list(source_start),
            "source_goal_position": list(source_goal),
            "source_reference_path": [list(_point3(item, "reference_path point")) for item in reference_path],
            "semantic_regions": "not_inferred_until_vlm_or_human_review",
        },
    )
    sample = SampleSpec(
        sample_id=f"{sample_prefix}_{episode_id}",
        scene=scene,
        task=task,
        route=route,
        # A non-empty placeholder keeps the dataclass contract valid before
        # the fixed Prompt Compiler writes the full Seedance prompt.
        seedance_prompt="待固定 Prompt Compiler 编译 Seedance 场景、时间线与负面约束。",
        seed=int(episode_id) if episode_id.isdigit() else abs(hash(episode_id)) % (2**31),
        planner_source="rxr_episode_import",
        metadata={
            "source": "RxR",
            "source_episode_id": episode_id,
            "source_scene_id": str(episode.get("scene_id", "SN83YJsR3w2")),
            "source_trajectory_id": str(episode.get("trajectory_id", "")),
            "source_instruction_id": str(episode.get("instruction_id", "")),
            "source_language": str(episode.get("language", "en")),
            "source_goal_radius_m": float(episode.get("goal_radius_m", 3.0)),
            "route_prior": {
                "kind": "rxr_reference_path_projection",
                "scale_m_per_source_unit": scale,
                "point_count": len(reference_path),
                "ground_truth": False,
            },
            "planner_stage": "rxr_episode_import",
            "task_summary_zh": str(episode.get("task_summary_zh", "")),
        },
    )
    errors = sample.validate()
    if errors:
        raise ValueError(f"invalid imported RxR sample {sample.sample_id}: {errors}")
    return sample


def load_rxr_episodes(
    path: Path | str,
    *,
    duration_s: float = 15.0,
    local_scale: float = DEFAULT_LOCAL_SCALE,
    sample_prefix: str = "rxr",
) -> List[SampleSpec]:
    """Load and convert all episodes in a NavGen RxR JSON file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"RxR episode file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid RxR episode JSON: {exc.msg}") from exc
    if isinstance(payload, Mapping):
        records = payload.get("episodes", payload.get("samples", []))
    else:
        records = payload
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("RxR episode file must contain an episodes array")
    samples = [
        rxr_episode_to_sample(
            record,
            duration_s=duration_s,
            local_scale=local_scale,
            sample_prefix=sample_prefix,
        )
        for record in records
        if isinstance(record, Mapping)
    ]
    if not samples:
        raise ValueError("RxR episode file contains no episode records")
    return samples


__all__ = ["RXR_IMPORTER_VERSION", "load_rxr_episodes", "rxr_episode_to_sample"]
