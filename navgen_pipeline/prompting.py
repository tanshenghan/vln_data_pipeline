"""Prompt compilation and optional model-assisted refinement for Seedance.

The video model should not have to infer a navigation task from one short
sentence.  This module turns the structured scene/task/route contract into a
repeatable, camera-aware prompt with explicit identity anchors, a timeline and
negative constraints.  A Doubao refiner is optional: the deterministic
compiler remains the safety net and makes offline runs reproducible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .models import ObjectSpec, SampleSpec


PROMPT_COMPILER_VERSION = "seedance-nav-v2"


_CATEGORY_NAMES = {
    "buffet_cabinet": "浅米色四门餐边柜",
    "bookcase": "深色木质书架",
    "bookshelf": "深色木质书架",
    "desk": "橡木书桌",
    "door": "室内木门",
    "chair": "蓝灰色椅子",
    "lamp": "暖黄色台灯",
    "plant": "白盆绿植",
    "printer": "白色打印机",
    "sofa": "蓝灰色三人沙发",
    "table": "圆形木桌",
    "person": "穿浅绿色上衣的人",
    # RxR/Matterport landmarks used by the episode importer.  Concrete
    # descriptions help Seedance preserve object identity across the clip.
    "potted_plants": "一组盆栽绿植",
    "door_mat": "地面上的编织门垫",
    "glass_door": "三联玻璃门",
    "stairs": "室内楼梯",
    "portrait": "墙上的女士画像",
    "bedroom_door": "卧室开放门",
    "side_table": "小边桌",
    "mirror": "方形墙镜",
    "armoire": "深色木质衣柜",
    "bed": "卧室床",
    "wash_basin": "白色洗手盆",
    "ottoman": "地面脚凳",
}


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, Mapping):
                return dict(parsed)
    raise ValueError("prompt refiner response did not contain a JSON object")


def _degrees(radians: float) -> int:
    return int(round(math.degrees(float(radians))))


def _position_hint(scene_width: float, scene_depth: float, obj: ObjectSpec) -> str:
    horizontal = "左侧" if obj.position[0] < scene_width * 0.4 else "右侧" if obj.position[0] > scene_width * 0.65 else "中部"
    vertical = "远端" if obj.position[1] > scene_depth * 0.62 else "入口附近" if obj.position[1] < scene_depth * 0.32 else "中段"
    return f"{horizontal}{vertical}"


def _object_description(scene_width: float, scene_depth: float, obj: ObjectSpec) -> str:
    name = _CATEGORY_NAMES.get(obj.category, obj.category)
    attributes = []
    for key, value in obj.attributes.items():
        if value is True:
            attributes.append(str(key))
        elif value not in (False, None, ""):
            attributes.append(f"{key}={value}")
    suffix = f"（{', '.join(attributes)}）" if attributes else ""
    dynamic = "，会沿任务路线连续移动" if obj.dynamic else "，保持静止"
    return f"{obj.id}: {name}{suffix}，位于场景{_position_hint(scene_width, scene_depth, obj)}{dynamic}"


@dataclass
class PromptBundle:
    """Compiled prompt plus machine-readable checks retained for audit."""

    prompt: str
    negative_prompt: str
    version: str = PROMPT_COMPILER_VERSION
    sections: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic_compiler"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "sections": list(self.sections),
            "checks": dict(self.checks),
            "prompt_chars": len(self.prompt),
            "negative_prompt_chars": len(self.negative_prompt),
            "seedance_prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
        }


class PromptCompiler:
    """Compile a structured sample into a camera- and motion-aware prompt."""

    version = PROMPT_COMPILER_VERSION

    def compile(self, sample: SampleSpec, source: str = "deterministic_compiler") -> PromptBundle:
        scene = sample.scene
        task = sample.task
        route = sample.route
        target = scene.object_by_id(task.target_id) if task.target_id else None
        target_description = _object_description(scene.width_m, scene.depth_m, target) if target else "target object defined by the task"
        static_objects = [
            _object_description(scene.width_m, scene.depth_m, obj)
            for obj in scene.objects
            if obj is not target
        ]
        if not static_objects:
            static_objects = ["场景中无额外障碍物；保持墙面和地面纹理稳定"]
        region_anchor = (
            "；".join(
                f"{region.get('label', region.get('name', '区域'))}"
                f"=x[{float(region['x_min']):.2f},{float(region['x_max']):.2f}]"
                f",y[{float(region['y_min']):.2f},{float(region['y_max']):.2f}]"
                for region in scene.regions
                if all(key in region for key in ("x_min", "x_max", "y_min", "y_max"))
            )
            or "未提供房间分区；整个 manifest 视为一个连续开放空间"
        )

        timeline: List[str] = []
        for index, (left, right) in enumerate(zip(route.waypoints, route.waypoints[1:]), start=1):
            if abs(right.x - left.x) < 1e-5 and abs(right.y - left.y) < 1e-5:
                action = "终点停留并保持构图"
            elif index == len(route.waypoints) - 1:
                action = "接近目标、对准目标并平稳停下"
            else:
                action = "沿规划路线连续移动"
            timeline.append(
                f"{left.t:.1f}–{right.t:.1f}s：{action}；相机约在({left.x:.2f},{left.y:.2f})移动至({right.x:.2f},{right.y:.2f})，朝向约{_degrees(right.yaw)}°"
            )

        actions = " → ".join(task.canonical_actions) if task.canonical_actions else "move → approach → stop"
        negative_prompt = (
            "禁止剪辑、转场、跳帧、回放、传送、穿墙、穿家具、碰撞、漂浮、\n"
            "禁止房间布局或物体形状/颜色/材质变化，禁止目标身份切换或消失后凭空出现，\n"
            "禁止相机高度、视场角和镜头畸变突然变化，禁止不符合差速轮机器人运动学的横移和抖动，\n"
            "禁止字幕、文字、水印、片头、片尾、背景音乐和无关人物。"
        )
        prompt = "\n".join(
            [
                "【输出目标】生成一段真实感室内移动机器人 RGB 导航视频；画面应像机器人相机实拍，而不是平面示意图。",
                f"【镜头与格式】第一视角、相机高度约{scene.camera_height_m:.2f}m、{scene.fps}fps、{route.duration_s:.1f}s、{scene.width_m:.1f}m×{scene.depth_m:.1f}m 室内空间；16:9，单镜头连续拍摄，正常透视和自然运动模糊。",
                f"【场景锚点】房间类型：{scene.room_type}；语义区域：{region_anchor}。墙面、地板、门、家具的空间关系从第一帧到最后一帧固定。静态物体：{'；'.join(static_objects)}。",
                f"【目标锚点】{target_description}。目标必须从出现开始保持同一外观、颜色、尺寸和身份；任务要求的目标可见比例约{task.stop_condition.target_visible_ratio:.2f}。",
                f"【导航语义】原始指令：{task.instruction_zh}；动作顺序：{actions}。",
                "【时间线】" + "；".join(timeline),
                f"【终点条件】在距离目标{task.stop_condition.distance_min_m:.2f}–{task.stop_condition.distance_max_m:.2f}m处结束；{('镜头朝向目标' if task.stop_condition.face_target else '保持目标在前方视野')}；最后{task.stop_condition.dwell_s:.1f}s 完全静止并保持目标可见。",
                f"【运动约束】移动机器人 profile={route.robot_profile}；最大线速度{route.max_speed_mps:.2f}m/s，最大加速度{route.max_accel_mps2:.2f}m/s²，最大角速度{route.max_angular_speed_rps:.2f}rad/s；沿可通行区域平滑减速，障碍物安全间隙至少{route.clearance_m:.2f}m。",
                "【连续性约束】相邻帧必须属于同一空间；曝光、白平衡和光照缓慢变化；保持目标与相机的真实遮挡关系，不要用生成式变形填补运动。",
                "【负面约束】" + negative_prompt,
            ]
        )
        required_markers = ["【镜头与格式】", "【场景锚点】", "【目标锚点】", "【时间线】", "【终点条件】", "【运动约束】", "【负面约束】"]
        checks = {
            "has_required_sections": all(marker in prompt for marker in required_markers),
            "waypoint_count": len(route.waypoints),
            "timeline_count": len(timeline),
            "duration_integer_for_seedance": abs(route.duration_s - round(route.duration_s)) < 1e-6 and 4 <= round(route.duration_s) <= 15,
            "target_anchor_present": target is not None,
            "instruction_present": bool(task.instruction_zh.strip()),
            "negative_constraints_present": len(negative_prompt) >= 100,
        }
        checks["quality"] = "pass" if all(value is True for key, value in checks.items() if key not in {"waypoint_count", "timeline_count"}) else "review"
        return PromptBundle(
            prompt=prompt,
            negative_prompt=negative_prompt,
            version=self.version,
            sections=[marker.strip("【】") for marker in required_markers],
            checks=checks,
            source=source,
        )

    def apply(self, sample: SampleSpec, source: str = "deterministic_compiler") -> PromptBundle:
        bundle = self.compile(sample, source=source)
        sample.seedance_prompt = bundle.prompt
        sample.metadata["prompt_compiler"] = bundle.version
        sample.metadata["prompt_report"] = bundle.to_dict()
        return bundle


class DoubaoPromptRefiner:
    """Optional second model pass that improves wording without changing facts."""

    def __init__(self, responses_client: Any, model: str, compiler: Optional[PromptCompiler] = None) -> None:
        self.client = responses_client
        self.model = model
        self.compiler = compiler or PromptCompiler()

    def refine(self, sample: SampleSpec) -> PromptBundle:
        base = self.compiler.compile(sample)
        system = (
            "你是 Seedance 2.0 的视频提示词编辑器。只输出严格 JSON，不要 markdown。"
            "只能改写表达，不能改变场景物体、目标身份、时间线、路线坐标、终点条件或运动上限。"
            "必须保留真实室内第一视角、连续单镜头和全部负面约束。"
            "JSON 字段必须是 seedance_prompt、negative_prompt、changes。"
        )
        user = json.dumps(
            {
                "scene": {
                    "room_type": sample.scene.room_type,
                    "dimensions_m": [sample.scene.width_m, sample.scene.depth_m],
                    "regions": list(sample.scene.regions),
                    "objects": [
                        {"id": obj.id, "category": obj.category, "attributes": obj.attributes, "position": obj.position, "dynamic": obj.dynamic}
                        for obj in sample.scene.objects
                    ],
                },
                "task": {
                    "type": sample.task.type,
                    "target_id": sample.task.target_id,
                    "instruction_zh": sample.task.instruction_zh,
                    "stop_condition": sample.task.stop_condition.__dict__,
                },
                "route": {
                    "waypoints": [point.__dict__ for point in sample.route.waypoints],
                    "limits": {
                        "max_speed_mps": sample.route.max_speed_mps,
                        "max_accel_mps2": sample.route.max_accel_mps2,
                        "clearance_m": sample.route.clearance_m,
                    },
                },
                "base_prompt": base.prompt,
                "base_negative_prompt": base.negative_prompt,
            },
            ensure_ascii=False,
        )
        response = self.client.generate_text(model=self.model, system=system, user_text=user, max_output_tokens=2600)
        data = _json_object(response)
        refined_prompt = str(data.get("seedance_prompt", "")).strip()
        refined_negative = str(data.get("negative_prompt", "")).strip() or base.negative_prompt
        if not refined_prompt:
            raise ValueError("prompt refiner returned an empty seedance_prompt")
        # A model may accidentally omit a required section.  Do not append the
        # complete deterministic prompt: that creates duplicate sections and
        # can make Seedance resolve contradictory wording.  Fall back to the
        # single, compiler-produced prompt and retain an audit flag instead.
        required = [
            "【场景锚点】",
            "【目标锚点】",
            "【时间线】",
            "【终点条件】",
            "【运动约束】",
            "【负面约束】",
        ]
        prompt_fallback = False
        if not all(marker in refined_prompt for marker in required):
            refined_prompt = base.prompt
            prompt_fallback = True
        # Do not let a refiner silently weaken the safety/continuity list.
        # Short or generic output is replaced by the deterministic list that
        # was constructed from the route contract.
        negative_fallback = len(refined_negative) < 100 or not all(
            phrase in refined_negative for phrase in ("穿墙", "禁止房间布局", "禁止字幕")
        )
        if negative_fallback:
            refined_negative = base.negative_prompt
        bundle = PromptBundle(
            prompt=refined_prompt,
            negative_prompt=refined_negative,
            version=self.compiler.version,
            sections=base.sections,
            checks={
                **base.checks,
                "model_changes": data.get("changes", []),
                "refiner_prompt_fallback": prompt_fallback,
                "refiner_negative_fallback": negative_fallback,
            },
            source="doubao_refiner",
        )
        bundle.checks["quality"] = "pass" if all(marker in bundle.prompt for marker in required) else "review"
        return bundle
