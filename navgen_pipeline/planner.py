"""Scene/task planning for the navigation data pipeline."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import (
    ObjectSpec,
    RouteContract,
    SampleSpec,
    SceneManifest,
    StopCondition,
    TaskSpec,
    Waypoint,
)
from .prompting import PromptCompiler


TASK_TYPES = ("object_goal", "instruction_following", "target_tracking")


@dataclass
class PlannerConfig:
    # Fifteen seconds is the upper end of the current Seedance single-clip
    # range and leaves room for acceleration, approach and terminal dwell.
    duration_s: float = 15.0
    camera_height_m: float = 1.2
    fps: int = 24
    robot_profile: str = "indoor_diff_drive_v1"


def _yaw_between(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)


def _scene_templates() -> List[Dict[str, Any]]:
    """Small deterministic scene library used by the offline smoke test."""

    return [
        {
            "room_type": "hallway_lounge",
            "width_m": 8.0,
            "depth_m": 5.5,
            "objects": [
                ObjectSpec(
                    id="door_01",
                    category="door",
                    attributes={"state": "closed", "side": "left"},
                    position=(1.0, 2.8),
                    size=(0.12, 1.0),
                    color_rgb=(120, 90, 70),
                ),
                ObjectSpec(
                    id="cabinet_01",
                    category="buffet_cabinet",
                    attributes={"doors": 4, "decorations": True},
                    position=(5.8, 4.3),
                    size=(1.6, 0.55),
                    color_rgb=(188, 162, 126),
                ),
                ObjectSpec(
                    id="shelf_01",
                    category="bookshelf",
                    attributes={"shelves": 5},
                    position=(6.9, 2.0),
                    size=(0.65, 0.45),
                    color_rgb=(122, 82, 52),
                ),
                ObjectSpec(
                    id="chair_01",
                    category="chair",
                    attributes={"movable": False},
                    position=(4.0, 2.6),
                    size=(0.65, 0.65),
                    color_rgb=(75, 105, 118),
                ),
                ObjectSpec(
                    id="person_01",
                    category="person",
                    attributes={"shirt_color": "light_green", "role": "target"},
                    position=(2.1, 4.1),
                    size=(0.45, 0.45),
                    dynamic=True,
                    color_rgb=(142, 190, 142),
                ),
            ],
        },
        {
            "room_type": "study_room",
            "width_m": 7.0,
            "depth_m": 6.0,
            "objects": [
                ObjectSpec(
                    id="desk_01",
                    category="desk",
                    attributes={"wood": "oak"},
                    position=(4.8, 4.8),
                    size=(1.5, 0.8),
                    color_rgb=(156, 110, 72),
                ),
                ObjectSpec(
                    id="lamp_01",
                    category="lamp",
                    attributes={"state": "on"},
                    position=(5.0, 4.3),
                    size=(0.25, 0.25),
                    color_rgb=(236, 208, 110),
                ),
                ObjectSpec(
                    id="bookcase_01",
                    category="bookcase",
                    attributes={"shelves": 4},
                    position=(1.1, 4.4),
                    size=(0.7, 0.55),
                    color_rgb=(105, 74, 50),
                ),
                ObjectSpec(
                    id="plant_01",
                    category="plant",
                    attributes={"pot": "white"},
                    position=(2.5, 1.8),
                    size=(0.55, 0.55),
                    color_rgb=(84, 145, 85),
                ),
            ],
        },
        {
            "room_type": "open_office",
            "width_m": 9.0,
            "depth_m": 6.5,
            "objects": [
                ObjectSpec(
                    id="printer_01",
                    category="printer",
                    attributes={"color": "white"},
                    position=(7.2, 4.9),
                    size=(0.7, 0.55),
                    color_rgb=(224, 224, 218),
                ),
                ObjectSpec(
                    id="sofa_01",
                    category="sofa",
                    attributes={"seats": 3},
                    position=(2.2, 4.8),
                    size=(1.8, 0.8),
                    color_rgb=(80, 107, 126),
                ),
                ObjectSpec(
                    id="table_01",
                    category="table",
                    attributes={"round": True},
                    position=(4.5, 2.8),
                    size=(1.0, 1.0),
                    color_rgb=(167, 133, 95),
                ),
            ],
        },
    ]


def _obstacle_for(obj: ObjectSpec, padding: float = 0.1) -> Dict[str, float]:
    half_x = obj.size[0] / 2 + padding
    half_y = obj.size[1] / 2 + padding
    return {
        "x_min": obj.position[0] - half_x,
        "x_max": obj.position[0] + half_x,
        "y_min": obj.position[1] - half_y,
        "y_max": obj.position[1] + half_y,
        "object_id": obj.id,
    }


def _with_terminal_dwell(points: List[Waypoint], duration_s: float, dwell_s: float) -> List[Waypoint]:
    """Append an explicit stationary segment for stop-condition validation."""

    if not points or dwell_s <= 0:
        return points
    last = points[-1]
    terminal_start = max(points[-2].t + 1e-3 if len(points) > 1 else 0.0, duration_s - dwell_s)
    points[-1] = Waypoint(terminal_start, last.x, last.y, last.yaw)
    points.append(Waypoint(duration_s, last.x, last.y, last.yaw))
    return points


def _approach_point(
    scene: SceneManifest,
    start: Tuple[float, float, float],
    target: ObjectSpec,
    distance_m: float = 1.0,
) -> Tuple[float, float]:
    """Return a point in front of ``target`` while staying inside the room."""

    dx = start[0] - target.position[0]
    dy = start[1] - target.position[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = -1.0, 0.0, 1.0
    x = target.position[0] + distance_m * dx / norm
    y = target.position[1] + distance_m * dy / norm
    # Leave a small margin for the robot footprint and avoid ending outside a
    # generated room when the target is close to a wall.
    x = min(max(0.45, x), max(0.45, scene.width_m - 0.45))
    y = min(max(0.45, y), max(0.45, scene.depth_m - 0.45))
    return x, y


def _timed_path(
    coordinates: Sequence[Tuple[float, float]],
    duration_s: float,
    dwell_s: float,
    final_yaw: Optional[float] = None,
    start_yaw: Optional[float] = None,
) -> List[Waypoint]:
    """Parameterise a short polyline by distance and append terminal dwell.

    Equal-distance timing prevents a short final segment from creating a speed
    spike.  The trajectory generator applies a smoothstep between these
    anchors, so the resulting path has zero velocity at intentional corners.
    """

    if len(coordinates) < 2:
        raise ValueError("a path needs at least two coordinates")
    cleaned: List[Tuple[float, float]] = []
    for coordinate in coordinates:
        point = (float(coordinate[0]), float(coordinate[1]))
        if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(point)
    if len(cleaned) < 2:
        raise ValueError("a path needs two distinct coordinates")
    if duration_s <= 0 or dwell_s < 0 or duration_s <= dwell_s:
        raise ValueError("duration must be greater than non-negative dwell")
    distances = [
        math.hypot(cleaned[index + 1][0] - cleaned[index][0], cleaned[index + 1][1] - cleaned[index][1])
        for index in range(len(cleaned) - 1)
    ]
    total_distance = sum(distances)
    if total_distance <= 1e-6:
        raise ValueError("path length must be positive")
    moving_duration = duration_s - dwell_s
    times = [0.0]
    for distance in distances:
        times.append(times[-1] + moving_duration * distance / total_distance)

    def segment_yaw(left: Tuple[float, float], right: Tuple[float, float]) -> float:
        return math.atan2(right[1] - left[1], right[0] - left[0])

    yaws: List[float] = []
    for index, point in enumerate(cleaned):
        if index == 0:
            yaw = start_yaw if start_yaw is not None else segment_yaw(cleaned[0], cleaned[1])
        elif index == len(cleaned) - 1:
            yaw = final_yaw if final_yaw is not None else segment_yaw(cleaned[-2], cleaned[-1])
        else:
            incoming = segment_yaw(cleaned[index - 1], point)
            outgoing = segment_yaw(point, cleaned[index + 1])
            # Circular mean keeps the camera orientation aligned with a bend.
            yaw = math.atan2(math.sin(incoming) + math.sin(outgoing), math.cos(incoming) + math.cos(outgoing))
        yaws.append(yaw)
    points = [Waypoint(t, xy[0], xy[1], yaws[index]) for index, (t, xy) in enumerate(zip(times, cleaned))]
    return _with_terminal_dwell(points, duration_s, dwell_s)


def _route_for(
    scene: SceneManifest, task_type: str, target: ObjectSpec, duration_s: float
) -> tuple[TaskSpec, RouteContract]:
    start = (0.8, 0.8, 0.0)
    target_name = {
        "buffet_cabinet": "餐边柜",
        "bookcase": "书架",
        "bookshelf": "书架",
        "desk": "书桌",
        "printer": "打印机",
        "sofa": "沙发",
        "table": "桌子",
        "plant": "盆栽",
    }.get(target.category, target.category)
    if task_type == "target_tracking":
        # Keep a broad, obstacle-free S curve.  The endpoint remains within a
        # generous 1--3.5 m tracking radius of the moving person.
        end_x = min(scene.width_m - 1.0, target.position[0] + 2.5)
        end_y = min(scene.depth_m - 0.9, target.position[1] + 0.2)
        if scene.room_type == "hallway_lounge":
            coordinates = [start[:2], (1.8, 1.4), (2.8, 2.5), (3.8, 3.7), (end_x, end_y)]
        elif scene.room_type == "study_room":
            coordinates = [start[:2], (1.5, 1.3), (2.3, 2.4), (3.2, 3.7), (end_x, end_y)]
        else:
            coordinates = [start[:2], (1.5, 1.4), (2.2, 2.2), (3.2, 3.8), (end_x, end_y)]
        task = TaskSpec(
            type=task_type,
            target_id=target.id,
            instruction_zh="跟随穿浅绿色上衣的人，保持目标在画面中并在其前方安全距离处停下。",
            canonical_actions=["follow_target", "maintain_visibility", "stop"],
            start_pose=start,
            stop_condition=StopCondition(
                distance_min_m=1.0,
                distance_max_m=3.5,
                face_target=False,
                dwell_s=1.0,
                target_visible_ratio=0.75,
            ),
            difficulty={"occlusion": "low", "turns": 2, "dynamic_target": True},
        )
        points = _timed_path(coordinates, duration_s, task.stop_condition.dwell_s)
    elif task_type == "instruction_following":
        goal = _approach_point(scene, start, target, 1.0)
        if scene.room_type == "hallway_lounge":
            coordinates = [start[:2], (2.5, 0.9), (3.1, 3.5), goal]
        elif scene.room_type == "study_room":
            coordinates = [start[:2], (3.2, 1.0), (3.2, 3.5), goal]
        else:
            coordinates = [start[:2], (3.2, 1.0), (3.3, 3.8), goal]
        task = TaskSpec(
            type=task_type,
            target_id=target.id,
            instruction_zh=f"从入口直行，向左绕过障碍物，继续前进到{target_name}旁并停下。",
            canonical_actions=["move_forward", "turn_left", "pass_obstacle", "approach_target", "stop"],
            start_pose=start,
            stop_condition=StopCondition(
                distance_min_m=0.7,
                distance_max_m=1.5,
                face_target=True,
                dwell_s=1.5,
                target_visible_ratio=0.8,
            ),
            difficulty={"occlusion": "low", "turns": 1, "distractors": 1},
        )
        points = _timed_path(
            coordinates,
            duration_s,
            task.stop_condition.dwell_s,
            final_yaw=_yaw_between(goal[0], goal[1], target.position[0], target.position[1]),
        )
    else:
        goal = _approach_point(scene, start, target, 1.0)
        if scene.room_type == "hallway_lounge":
            coordinates = [start[:2], (3.2, 1.5), (3.2, 3.6), goal]
        elif scene.room_type == "study_room":
            coordinates = [start[:2], (2.6, 1.0), (2.7, 3.2), goal]
        else:
            coordinates = [start[:2], (2.2, 1.3), (3.1, 3.5), goal]
        task = TaskSpec(
            type=task_type,
            target_id=target.id,
            instruction_zh=f"从入口前进到{target_name}旁，在能清楚看见目标细节的位置停下。",
            canonical_actions=["move_forward", "approach_target", "face_target", "stop"],
            start_pose=start,
            stop_condition=StopCondition(
                distance_min_m=0.7,
                distance_max_m=1.6,
                face_target=True,
                dwell_s=1.2,
                target_visible_ratio=0.85,
            ),
            difficulty={"occlusion": "none", "turns": 0, "viewpoint_goal": True},
        )
        points = _timed_path(
            coordinates,
            duration_s,
            task.stop_condition.dwell_s,
            final_yaw=_yaw_between(goal[0], goal[1], target.position[0], target.position[1]),
        )

    obstacles = [_obstacle_for(obj) for obj in scene.objects if obj.id != target.id and not obj.dynamic]
    # Store the same padded geometry in the scene manifest so downstream
    # consumers do not have to guess whether obstacles lived only in the
    # route contract.
    scene.obstacles = list(obstacles)
    route = RouteContract(
        waypoints=points,
        robot_profile="indoor_diff_drive_v1",
        max_speed_mps=0.8,
        max_accel_mps2=0.9,
        max_angular_speed_rps=1.4,
        max_angular_accel_rps2=2.5,
        clearance_m=0.18,
        goal_region={"target_id": target.id, "stop_condition": task.stop_condition.__dict__},
        expected_duration_s=duration_s,
        obstacles=obstacles,
    )
    return task, route


def _seedance_prompt(scene: SceneManifest, task: TaskSpec, route: RouteContract) -> str:
    target = scene.object_by_id(task.target_id) if task.target_id else None
    target_desc = target.category if target else "the designated goal"
    timeline = "；".join(
        f"{route.waypoints[i].t:.1f}-{route.waypoints[i + 1].t:.1f}秒沿路线移动"
        for i in range(len(route.waypoints) - 1)
    )
    return (
        "第一视角室内移动机器人相机，镜头高度约 "
        f"{scene.camera_height_m:.1f} 米，{scene.fps} fps，一镜到底，"
        f"时长约 {route.duration_s:.1f} 秒。场景是 {scene.room_type}，"
        f"目标是 {target_desc}。时间线：{timeline}。"
        f"导航指令：{task.instruction_zh}"
        "保持墙壁、地面、门、家具和目标外观及相对位置稳定；"
        "机器人运动连续平滑，不得剪辑、跳帧、传送、穿墙、穿家具、"
        "突然改变相机高度或让目标身份发生变化。"
        "不添加与任务无关的字幕、片头或背景音乐。"
    )


class TemplatePlanner:
    """Deterministic planner used for reproducible local generation."""

    def __init__(self, config: Optional[PlannerConfig] = None, seed: int = 7) -> None:
        self.config = config or PlannerConfig()
        self.random = random.Random(seed)

    def plan_one(self, index: int, task_type: Optional[str] = None) -> SampleSpec:
        template = _scene_templates()[index % len(_scene_templates())]
        objects = list(template["objects"])
        scene = SceneManifest(
            scene_id=f"scene_{index:04d}",
            room_type=str(template["room_type"]),
            objects=objects,
            width_m=float(template["width_m"]),
            depth_m=float(template["depth_m"]),
            camera_height_m=self.config.camera_height_m,
            fps=self.config.fps,
            metadata={"generator": "template", "template_index": index % len(_scene_templates())},
        )
        chosen_type = task_type or TASK_TYPES[index % len(TASK_TYPES)]
        if chosen_type == "target_tracking":
            target = next((obj for obj in objects if obj.dynamic), None)
            if target is None:
                # Add a deterministic synthetic person to scenes that did not
                # originally contain a dynamic agent.
                target = ObjectSpec(
                    id="person_01",
                    category="person",
                    attributes={"shirt_color": "light_green", "role": "target"},
                    position=(2.0, 3.6),
                    size=(0.45, 0.45),
                    dynamic=True,
                    color_rgb=(142, 190, 142),
                )
                objects.append(target)
                scene.objects = objects
        elif chosen_type == "instruction_following" and scene.object_by_id("cabinet_01"):
            target = scene.object_by_id("cabinet_01")  # type: ignore[assignment]
        else:
            preferred = ("cabinet_01", "bookcase_01", "bookshelf_01", "desk_01", "printer_01", "sofa_01", "table_01")
            target = next((scene.object_by_id(object_id) for object_id in preferred if scene.object_by_id(object_id)), None)
            target = target or next((obj for obj in objects if not obj.dynamic), objects[0])
        task, route = _route_for(scene, chosen_type, target, self.config.duration_s)
        sample = SampleSpec(
            sample_id=f"nav_{index:06d}",
            scene=scene,
            task=task,
            route=route,
            seedance_prompt=_seedance_prompt(scene, task, route),
            seed=self.random.randint(0, 2**31 - 1),
            planner_source="template",
            metadata={"task_type": chosen_type},
        )
        # Compile a full camera/scene/timeline prompt instead of handing the
        # video model only a short navigation sentence.  The compiler is
        # deterministic, so offline and live runs share the same anchors.
        PromptCompiler().apply(sample)
        errors = sample.validate()
        if errors:
            raise ValueError(f"invalid generated sample {sample.sample_id}: {errors}")
        return sample

    def plan_batch(self, count: int) -> List[SampleSpec]:
        if count < 1:
            raise ValueError("count must be positive")
        return [self.plan_one(i) for i in range(count)]


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response that may contain markdown."""

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("``"))
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("planner response must be a JSON object")
    return value


def _ground_point(value: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Normalize 2D ``[x, y]`` and robotics 3D ``[x, y, z]`` points.

    Remote planners often use y-up coordinates.  The internal route contract
    is a ground-plane (x, y) representation, so map the third coordinate to
    the internal y axis and retain the original height elsewhere when useful.
    """

    if isinstance(value, Mapping):
        if "x" in value and ("z" in value or "y" in value):
            return float(value.get("x", default[0])), float(value.get("z", value.get("y", default[1])))
        value = value.get("position", value.get("center", default))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return default
        if len(numbers) >= 3:
            return numbers[0], numbers[2]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
    return default


def _normalize_start_pose(
    value: Any, default: Tuple[float, float, float] = (0.8, 0.8, 0.0)
) -> Tuple[float, float, float]:
    """Normalize 2D ``[x, y, yaw]`` or 3D ``[x, y, z, yaw]`` poses.

    A three-element waypoint position is interpreted as (x, y-up, z), while
    a three-element *start pose* conventionally means (x, y, yaw).  Keeping a
    dedicated helper avoids accidentally using yaw as a ground coordinate.
    """

    if isinstance(value, Mapping):
        x = value.get("x", default[0])
        y = value.get("z", value.get("y", default[1]))
        yaw = value.get("yaw", value.get("yaw_rad", value.get("heading", default[2])))
        try:
            return float(x), float(y), float(yaw)
        except (TypeError, ValueError):
            return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return default
        if len(numbers) >= 4:
            return numbers[0], numbers[2], numbers[3]
        if len(numbers) >= 3:
            return numbers[0], numbers[1], numbers[2]
        if len(numbers) >= 2:
            return numbers[0], numbers[1], default[2]
    return default


def _ground_size(value: Any, default: Tuple[float, float] = (0.5, 0.5)) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        value = value.get("size", default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            numbers = [abs(float(item)) for item in value]
        except (TypeError, ValueError):
            return default
        if len(numbers) >= 3:
            return max(numbers[0], 0.05), max(numbers[2], 0.05)
        if len(numbers) >= 2:
            return max(numbers[0], 0.05), max(numbers[1], 0.05)
    return default


def _normalize_object(raw: Mapping[str, Any], index: int) -> ObjectSpec:
    item = dict(raw)
    object_id = str(item.get("id", item.get("object_id", f"object_{index:03d}")))
    category = str(item.get("category", item.get("name", "object")))
    position = _ground_point(item.get("position", item.get("center", (0.0, 0.0))))
    size = _ground_size(item.get("size", (0.5, 0.5)))
    attributes = dict(item.get("attributes", {})) if isinstance(item.get("attributes", {}), Mapping) else {}
    # Keep useful model fields without passing unsupported constructor keys.
    for key in ("is_target", "role", "material", "color", "shirt_color"):
        if key in item and key not in attributes:
            attributes[key] = item[key]
    if isinstance(item.get("position"), Sequence) and len(item["position"]) >= 3:
        attributes.setdefault("height_coordinate_m", float(item["position"][1]))
    dynamic = bool(item.get("dynamic", item.get("is_dynamic", False))) or any(
        marker in category.lower() for marker in ("moving", "person", "human", "pet", "animal")
    )
    color = item.get("color_rgb")
    color_rgb = tuple(int(v) for v in color[:3]) if isinstance(color, Sequence) and len(color) >= 3 and not isinstance(color, (str, bytes)) else None
    return ObjectSpec(
        id=object_id,
        category=category,
        attributes=attributes,
        position=position,
        size=size,
        dynamic=dynamic,
        color_rgb=color_rgb,
    )


def _normalize_obstacle(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(raw)
    if all(key in item for key in ("x_min", "x_max", "y_min", "y_max")):
        return {
            "x_min": float(item["x_min"]),
            "x_max": float(item["x_max"]),
            "y_min": float(item["y_min"]),
            "y_max": float(item["y_max"]),
            "object_id": str(item.get("object_id", item.get("id", f"obstacle_{index:03d}"))),
        }
    center = _ground_point(item.get("position", item.get("center", (0.0, 0.0))))
    size = _ground_size(item.get("size", (0.5, 0.5)))
    return {
        "x_min": center[0] - size[0] / 2.0,
        "x_max": center[0] + size[0] / 2.0,
        "y_min": center[1] - size[1] / 2.0,
        "y_max": center[1] + size[1] / 2.0,
        "object_id": str(item.get("obstacle_id", item.get("id", f"obstacle_{index:03d}"))),
        "category": str(item.get("category", "obstacle")),
    }


def _normalize_stop_condition(value: Any) -> StopCondition:
    if isinstance(value, Mapping):
        data = dict(value)
        distance = data.get("distance_range_m", data.get("distance_range"))
        if isinstance(distance, Sequence) and len(distance) >= 2:
            data.setdefault("distance_min_m", distance[0])
            data.setdefault("distance_max_m", distance[1])
        return StopCondition(
            distance_min_m=float(data.get("distance_min_m", 0.6)),
            distance_max_m=float(data.get("distance_max_m", 1.6)),
            face_target=bool(data.get("face_target", True)),
            dwell_s=float(data.get("dwell_s", 1.0)),
            target_visible_ratio=float(data.get("target_visible_ratio", 0.8)),
        )
    text = str(value or "")
    distance_range_match = re.search(
        r"(?:距离|distance)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(?:-|–|~|至|to)\s*(\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    distance_match = re.search(
        r"(?:距离|distance)[^0-9]{0,12}(?:不超过|<=|less than|under)?[^0-9]*(\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    ratio_match = re.search(r"(?:比例|占画面|ratio)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*%?", text, re.I)
    if distance_range_match:
        distance_min = float(distance_range_match.group(1))
        max_distance = float(distance_range_match.group(2))
    else:
        max_distance = float(distance_match.group(1)) if distance_match else 1.2
        distance_min = min(max_distance, 0.8)
    visible = float(ratio_match.group(1)) / 100.0 if ratio_match else 0.8
    return StopCondition(
        distance_min_m=max(0.0, min(distance_min, max_distance)),
        distance_max_m=max_distance,
        face_target=True,
        dwell_s=1.0,
        target_visible_ratio=max(0.0, min(1.0, visible)),
    )


def _normalize_waypoints(items: Any, duration_s: float) -> List[Waypoint]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []
    points: List[Tuple[float, float, float]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        x, y = _ground_point(item.get("position", item.get("point", item.get("coordinates", (0.0, 0.0)))))
        yaw = item.get("yaw", item.get("yaw_rad", item.get("heading", 0.0)))
        try:
            yaw_value = float(yaw)
        except (TypeError, ValueError):
            yaw_value = 0.0
        points.append((x, y, yaw_value))
    if len(points) < 2:
        return []
    # Parameterize by ground distance and reserve one second (or 10%) for a
    # terminal dwell.  This turns model waypoints without timestamps into the
    # strictly increasing internal route contract.
    distances = [math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)]
    total = sum(distances)
    dwell = min(1.2, max(0.5, duration_s * 0.1))
    moving = max(0.1, duration_s - dwell)
    times = [0.0]
    if total > 1e-6:
        for distance in distances:
            times.append(times[-1] + moving * distance / total)
    else:
        step = moving / max(1, len(points) - 1)
        times = [index * step for index in range(len(points))]
    # Ensure the final anchor is stationary through the requested duration.
    waypoints = [Waypoint(t, x, y, yaw) for t, (x, y, yaw) in zip(times, points)]
    waypoints[-1] = Waypoint(max(waypoints[-2].t + 1e-3, duration_s - dwell), *points[-1])
    waypoints.append(Waypoint(duration_s, *points[-1]))
    return waypoints


def sample_spec_from_dict(
    data: Mapping[str, Any], sample_id: str, seed: int, duration_override_s: Optional[float] = None
) -> SampleSpec:
    """Convert a planner JSON object into the local data contract."""

    scene_data = dict(data["scene"])
    for field in ("semantic_label", "semantic_confidence", "semantic_source"):
        if isinstance(scene_data.get(field), str) and not scene_data[field].strip():
            scene_data[field] = None
    objects = [
        _normalize_object(dict(item), index)
        for index, item in enumerate(scene_data.get("objects", []))
        if isinstance(item, Mapping)
    ]
    raw_obstacles = scene_data.get("obstacles", [])
    obstacles = [
        _normalize_obstacle(dict(item), index)
        for index, item in enumerate(raw_obstacles)
        if isinstance(item, Mapping)
    ]
    regions = [
        dict(item)
        for item in (scene_data.get("regions") or [])
        if isinstance(item, Mapping)
    ]
    scene = SceneManifest(
        scene_id=str(scene_data.get("scene_id", sample_id + "_scene")),
        room_type=str(scene_data.get("room_type", "unknown")),
        objects=objects,
        obstacles=obstacles,
        width_m=float(scene_data.get("width_m", scene_data.get("width", 8.0))),
        depth_m=float(scene_data.get("depth_m", scene_data.get("depth", 6.0))),
        camera_height_m=float(scene_data.get("camera_height_m", 1.2)),
        fps=int(scene_data.get("fps", 24)),
        reference_images=list(scene_data.get("reference_images", [])),
        metadata=dict(scene_data.get("metadata", {})),
        regions=regions,
        semantic_label=(str(scene_data["semantic_label"]) if scene_data.get("semantic_label") is not None else None),
        semantic_confidence=(float(scene_data["semantic_confidence"]) if scene_data.get("semantic_confidence") is not None else None),
        semantic_source=(str(scene_data["semantic_source"]) if scene_data.get("semantic_source") is not None else None),
    )
    task_data = dict(data["task"])
    stop_value = task_data.get("stop_condition", {})
    stop_condition = _normalize_stop_condition(stop_value)
    task_type = str(task_data.get("type", "object_goal"))
    task_type = {
        "go_to_target": "object_goal",
        "navigate_to_target": "object_goal",
        "navigate": "object_goal",
        "navigation": "object_goal",
        "point_goal": "object_goal",
        "object_navigation": "object_goal",
        "follow": "target_tracking",
        "follow_target": "target_tracking",
        "instruction_follow": "instruction_following",
    }.get(task_type, task_type)
    target_id = task_data.get("target_id", task_data.get("target_object_id"))
    if not target_id:
        target = next((obj for obj in objects if obj.attributes.get("is_target") is True), None)
        target_id = target.id if target is not None else (objects[0].id if objects else None)
    raw_start = task_data.get("start_pose", task_data.get("start", [0.8, 0.8, 0.0]))
    start_x, start_y, start_yaw = _normalize_start_pose(raw_start)
    raw_actions = task_data.get("canonical_actions", task_data.get("actions", []))
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    task = TaskSpec(
        type=task_type,
        target_id=str(target_id) if target_id is not None else None,
        instruction_zh=str(task_data.get("instruction_zh", task_data.get("instruction", ""))),
        canonical_actions=list(raw_actions) if isinstance(raw_actions, Sequence) else [],
        start_pose=(start_x, start_y, start_yaw),
        stop_condition=stop_condition,
        difficulty=dict(task_data.get("difficulty", {}))
        if isinstance(task_data.get("difficulty", {}), Mapping)
        else {},
        instruction_original=task_data.get("instruction_original"),
    )
    route_data = dict(data.get("route_contract", data.get("route", {})))
    raw_duration = route_data.get("expected_duration_s", route_data.get("duration_s", 15.0))
    try:
        expected_duration = float(duration_override_s if duration_override_s is not None else raw_duration)
    except (TypeError, ValueError):
        expected_duration = 15.0
    # Seedance requires an integer duration.  Keep the model's intent while
    # normalizing to the nearest valid contract value.
    expected_duration = float(max(4, min(15, round(expected_duration))))
    waypoints = _normalize_waypoints(
        route_data.get("waypoints", route_data.get("points", [])), expected_duration
    )
    route_obstacles = [
        _normalize_obstacle(dict(item), index)
        for index, item in enumerate(route_data.get("obstacles") or obstacles)
        if isinstance(item, Mapping)
    ]
    goal_region = dict(route_data.get("goal_region", {})) if isinstance(route_data.get("goal_region", {}), Mapping) else {}
    if "center" in goal_region:
        goal_region["center"] = list(_ground_point(goal_region["center"]))
    route = RouteContract(
        waypoints=waypoints,
        robot_profile=str(route_data.get("robot_profile", "indoor_diff_drive_v1")),
        max_speed_mps=float(route_data.get("max_speed_mps", 0.7)),
        max_accel_mps2=float(route_data.get("max_accel_mps2", 0.8)),
        max_angular_speed_rps=float(route_data.get("max_angular_speed_rps", 1.2)),
        max_angular_accel_rps2=float(route_data.get("max_angular_accel_rps2", 2.0)),
        clearance_m=float(route_data.get("clearance_m", 0.18)),
        goal_region=goal_region,
        expected_duration_s=expected_duration,
        obstacles=route_obstacles or list(scene.obstacles),
    )
    prompt = str(data.get("seedance_prompt", "")) or _seedance_prompt(scene, task, route)
    sample = SampleSpec(
        sample_id=sample_id,
        scene=scene,
        task=task,
        route=route,
        seedance_prompt=prompt,
        seed=seed,
        planner_source="doubao",
        metadata=dict(data.get("metadata", {})),
    )
    # Normalize model-produced plans through the same deterministic compiler;
    # this prevents a remote planner from omitting camera, continuity or
    # negative constraints required by Seedance.
    PromptCompiler().apply(sample, source="planner_compiler")
    errors = sample.validate()
    if errors:
        raise ValueError(f"invalid model-generated sample {sample_id}: {errors}")
    return sample


class DoubaoPlanner:
    """Planner backed by the OpenAI-compatible Ark Responses endpoint."""

    def __init__(self, responses_client: Any, model: str, duration_s: float = 15.0) -> None:
        self.client = responses_client
        self.model = model
        self.duration_s = float(duration_s)

    def plan_one(self, index: int, seed: int) -> SampleSpec:
        system = (
            "你是机器人导航数据规划器。只输出一个严格 JSON 对象，不要 markdown。"
            "场景必须包含可观察物体、唯一 target_id、可执行的 waypoints，"
            "并让 instruction_zh 与 route_contract 的动作顺序完全一致。"
            "只有能够明确给出边界和标签时才填写 scene.regions，否则返回空数组，不能猜造房间分区。"
            "semantic_label/semantic_confidence/semantic_source 属于视频审核后的观测字段；规划阶段必须返回 JSON null，"
            "不能把 room_type 先验冒充视觉语义。"
        )
        user = (
            "生成一个室内移动机器人导航样本。场景中至少包含一个目标物和一个障碍物；"
            "视频将由第一视角模型生成，因此任务必须能从 RGB 视频观察验证。"
            f"route_contract.expected_duration_s 必须设为 {self.duration_s:.0f} 秒。"
            "输出字段：scene(scene_id,room_type,semantic_label,semantic_confidence,semantic_source,objects,obstacles,regions,width_m,depth_m,camera_height_m,fps),"
            "task(type,target_id,instruction_zh,canonical_actions,start_pose,stop_condition),"
            "route_contract(waypoints,robot_profile,max_speed_mps,max_accel_mps2,"
            "max_angular_speed_rps,max_angular_accel_rps2,clearance_m,goal_region,expected_duration_s)。"
            f"本次样本随机种子仅用于记录：{seed}。"
        )
        text = self.client.generate_text(
            model=self.model,
            system=system,
            user_text=user,
        )
        return sample_spec_from_dict(
            _extract_json(text), f"nav_{index:06d}", seed, duration_override_s=self.duration_s
        )

    def plan_batch(self, count: int, seed: int = 7) -> List[SampleSpec]:
        return [self.plan_one(i, seed + i) for i in range(count)]
