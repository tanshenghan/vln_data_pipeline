"""Semantic/visual filtering with an offline heuristic and Ark VLM backend."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ark_client import ArkResponsesClient, file_to_data_url
from .embedding import LocalEmbedding
from .models import SampleSpec, VLMReport
from .video import VideoInfo

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class VLMThresholds:
    scene_pass: float = 0.85
    alignment_pass: float = 0.85
    # The local colour-presence heuristic sees small furniture targets at a
    # lower pixel ratio than a full VLM.  Keep the threshold high enough to
    # reject missing targets while allowing a clearly rendered 720p target;
    # live mode still uses the stricter semantic judge.
    identity_pass: float = 0.55
    confidence_pass: float = 0.80
    review_floor: float = 0.60
    tracking_visible_ratio: float = 0.75


def _color_visibility(path: Path, rgb: Tuple[int, int, int], tolerance: int = 35) -> float:
    if Image is None:
        return 0.5
    try:
        image = Image.open(path).convert("RGB").resize((64, 36))
        pixels = list(image.getdata())
    except Exception:
        return 0.0
    hits = 0
    for pixel in pixels:
        distance = math.sqrt(sum((int(pixel[i]) - int(rgb[i])) ** 2 for i in range(3)))
        if distance <= tolerance * math.sqrt(3):
            hits += 1
    # A target generally occupies more than a handful of pixels in the mock
    # video.  Cap the score so a tiny accidental color patch is not a pass.
    ratio = hits / max(1, len(pixels))
    return min(1.0, ratio / 0.025)


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


_UNIT_SCORE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:0?\.\d+|0|1(?:\.0+)?)(?![A-Za-z0-9_])")
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)?)\s*%")

# Keep observed room semantics intentionally small and auditable.  The VLM
# may describe a room in Chinese or English, but an arbitrary prose sentence
# must not become a map label.  ``room_type`` remains the planner prior.
_SCENE_LABEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hallway": ("hallway", "corridor", "passage", "走廊", "过道", "通道"),
    "living_room": ("living room", "lounge", "客厅", "起居室"),
    "study": ("study room", "study", "书房"),
    "office": ("office", "办公室"),
    "kitchen": ("kitchen", "厨房"),
    "dining_room": ("dining room", "dining", "餐厅", "饭厅"),
    "bedroom": ("bedroom", "卧室"),
    "bathroom": ("bathroom", "bath", "卫生间", "浴室"),
    "open_space": ("open space", "open plan", "开放空间", "开放式空间"),
}


def _coerce_scene_label(value: Any) -> Tuple[Optional[str], bool]:
    """Normalize an observed room label without guessing from free prose."""

    if value is None:
        return None, False
    if not isinstance(value, str):
        return None, True
    text = re.sub(r"\s+", " ", value.strip().lower())
    if not text:
        return None, True
    matches = {
        canonical
        for canonical, aliases in _SCENE_LABEL_ALIASES.items()
        if any(alias in text for alias in aliases)
    }
    if len(matches) == 1:
        return next(iter(matches)), False
    return None, True


def _coerce_unit_score(value: Any) -> Tuple[float, bool]:
    """Return a finite [0, 1] score and whether the source was malformed.

    VLMs occasionally put a prose object description in a numeric field.  A
    malformed value must become an auditable review signal, not crash the
    whole sample or accidentally turn an object id such as ``cabinet_01``
    into a score of one.
    """

    if isinstance(value, bool):
        return 0.0, True
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return max(0.0, min(1.0, number)), False
        return 0.0, True
    if isinstance(value, str):
        text = value.strip()
        percent = _PERCENT_RE.search(text)
        if percent:
            number = float(percent.group(1)) / 100.0
            return max(0.0, min(1.0, number)), False
        match = _UNIT_SCORE_RE.search(text)
        if match:
            number = float(match.group(0))
            return max(0.0, min(1.0, number)), False
    return 0.0, True


def _coerce_semantic_score(field_name: str, value: Any) -> Tuple[float, bool]:
    """Map an unambiguous natural-language verdict to a unit score.

    Seed VLMs occasionally follow the *meaning* of the requested schema but
    emit a sentence instead of a number (for example, ``"场景保持一致"``).
    We accept only explicit positive/negative cues and keep all other prose as
    malformed so it enters manual review rather than being guessed.
    """

    if not isinstance(value, str):
        return 0.0, False
    text = re.sub(r"\s+", "", value.strip().lower())
    if not text:
        return 0.0, False
    vocab: Dict[str, Tuple[Sequence[str], Sequence[str]]] = {
        "scene_consistency": (
            ("不一致", "不稳定", "发生变化", "明显变化", "漂移", "跳变", "消失", "不同空间", "inconsistent", "scene drift", "drift", "changed", "disappear", "different scene"),
            ("未发生变化", "没有明显变化", "无明显变化", "保持一致", "场景一致", "同一空间", "一致", "稳定", "consistent", "same scene", "same space", "stable", "no drift"),
        ),
        "instruction_alignment": (
            ("不匹配", "未匹配", "未遵循", "不符合", "偏离", "未完成", "错误方向", "穿墙", "mismatch", "notfollowed", "deviat", "wrong direction", "failed"),
            ("匹配", "遵循", "符合", "完成", "对齐", "沿路线", "达到终点", "执行正确", "instructionfollowed", "navigationprogress", "aligned", "matches", "followed"),
        ),
        "target_visible_ratio": (
            ("不可见", "未看见", "消失", "遮挡严重", "notvisible", "disappeared", "occluded"),
            ("始终可见", "保持可见", "清晰可见", "目标可见", "可见率", "visible", "inview", "clearlyvisible"),
        ),
    }
    negative, positive = vocab.get(field_name, ((), ()))
    # Negative cues win when a sentence contains both a quoted field name and
    # a corrective qualifier (e.g. ``"不一致，但目标匹配"``).
    if any(marker in text for marker in negative):
        return 0.0, True
    if any(marker in text for marker in positive):
        return 1.0, True
    return 0.0, False


def _coerce_time(value: Any, default: float) -> Tuple[float, bool]:
    """Coerce a finite timestamp while retaining a malformed-value flag."""

    if isinstance(value, bool):
        return default, True
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, False) if math.isfinite(number) else (default, True)
    if isinstance(value, str):
        match = re.search(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", value)
        if match:
            number = float(match.group(0))
            return (number, False) if math.isfinite(number) else (default, True)
    return default, True


class HeuristicVLMFilter:
    """Deterministic local judge used when an Ark VLM is not available."""

    def __init__(self, thresholds: Optional[VLMThresholds] = None, embedding: Optional[LocalEmbedding] = None) -> None:
        self.thresholds = thresholds or VLMThresholds()
        self.embedding = embedding or LocalEmbedding()

    def evaluate(
        self,
        sample: SampleSpec,
        frames: Sequence[Path],
        video_info: Optional[VideoInfo] = None,
        embedding_report: Optional[Mapping[str, Any]] = None,
    ) -> VLMReport:
        if not frames:
            return VLMReport(
                decision="reject",
                scene_consistency=0.0,
                instruction_alignment=0.0,
                target_identity=0.0,
                target_visible_ratio=0.0,
                valid_interval=(0.0, 0.0),
                confidence=0.0,
                reason_codes=["no_frames"],
                source="heuristic",
            )
        temporal = self.embedding.temporal_consistency(frames)
        # Local embeddings are intentionally conservative.  Mix in a stable
        # baseline because normal camera motion changes pixels even in a valid
        # scene; the remote VLM remains the authoritative semantic judge.
        scene_score = max(0.0, min(1.0, 0.35 + 0.65 * temporal))
        target = sample.scene.object_by_id(sample.task.target_id) if sample.task.target_id else None
        if target is not None and target.color_rgb is not None:
            visibility_values = [_color_visibility(frame, target.color_rgb) for frame in frames]
            visible_ratio = sum(1 for value in visibility_values if value >= 0.45) / len(visibility_values)
            identity_score = sum(visibility_values) / len(visibility_values)
        else:
            visible_ratio = 0.85
            identity_score = 0.85
        # A structurally valid contract and a decodable video provide a useful
        # local alignment prior; an Ark VLM can override it in live mode.
        alignment = 0.90 if sample.task.canonical_actions and sample.route.waypoints else 0.70
        confidence = min(scene_score, alignment, max(identity_score, 0.0))
        reasons: List[str] = []
        duplicate_pairs = embedding_report.get("duplicate_frame_pairs", []) if embedding_report else []
        if len(frames) >= 3 and isinstance(duplicate_pairs, Sequence) and len(duplicate_pairs) >= len(frames):
            reasons.append("frozen_or_duplicate_video")
        if scene_score < self.thresholds.scene_pass:
            reasons.append("temporal_scene_drift")
        if alignment < self.thresholds.alignment_pass:
            reasons.append("instruction_alignment_uncertain")
        if sample.task.type == "target_tracking" and visible_ratio < self.thresholds.tracking_visible_ratio:
            reasons.append("target_visibility_gap")
        if target is not None and target.color_rgb is not None and identity_score < self.thresholds.identity_pass:
            reasons.append("target_identity_uncertain")
        # A local heuristic cannot prove subtle semantic alignment.  Mark a
        # borderline result for review instead of silently accepting it.
        if reasons:
            decision = "review" if confidence >= self.thresholds.review_floor else "reject"
        elif confidence >= self.thresholds.confidence_pass:
            decision = "pass"
        else:
            decision = "review"
        duration = video_info.duration_s if video_info else sample.route.duration_s
        return VLMReport(
            decision=decision,
            scene_consistency=scene_score,
            instruction_alignment=alignment,
            target_identity=max(0.0, min(1.0, identity_score)),
            target_visible_ratio=max(0.0, min(1.0, visible_ratio)),
            valid_interval=(0.0, max(0.0, duration)),
            confidence=max(0.0, min(1.0, confidence)),
            evidence=[
                {"frame": str(frames[0]), "reason_code": "first_keyframe"},
                {"frame": str(frames[-1]), "reason_code": "last_keyframe"},
            ],
            reason_codes=reasons,
            source="heuristic",
        )


class DoubaoVLMFilter:
    """VLM judge backed by the Ark Responses-compatible endpoint."""

    def __init__(self, client: ArkResponsesClient, model: str, thresholds: Optional[VLMThresholds] = None) -> None:
        self.client = client
        self.model = model
        self.thresholds = thresholds or VLMThresholds()

    def evaluate(
        self,
        sample: SampleSpec,
        frames: Sequence[Path],
        contact_sheet: Optional[Path] = None,
        video_info: Optional[VideoInfo] = None,
    ) -> VLMReport:
        if not frames and contact_sheet is None:
            raise ValueError("at least one frame or contact sheet is required for VLM filtering")
        image_paths = [contact_sheet] if contact_sheet else list(frames[:16])
        image_urls = [file_to_data_url(path) for path in image_paths if path is not None]
        manifest = {
            "scene": {
                "room_type": sample.scene.room_type,
                "objects": [
                    {"id": obj.id, "category": obj.category, "attributes": obj.attributes}
                    for obj in sample.scene.objects
                ],
            },
            "task": {
                "type": sample.task.type,
                "target_id": sample.task.target_id,
                "instruction": sample.task.instruction_zh,
                "canonical_actions": sample.task.canonical_actions,
            },
        }
        system = (
            "你是导航数据质量审核器。只输出 JSON，不输出 markdown。"
            "只能依据输入帧和场景清单判断；不确定时使用 review，不能猜测。"
            "左右方向以相机视角为准；若改变目标、左右方向或终点，不能静默改写原指令。"
            "observed_scene_label 只能从 hallway、living_room、study、office、kitchen、dining_room、"
            "bedroom、bathroom、open_space 中选择；看不清时必须为 null。"
            "scene_consistency、instruction_alignment、target_identity、target_visible_ratio、confidence"
            "必须是 0 到 1 的 JSON 数字，不要把描述性句子放进这些字段。"
        )
        user = (
            "请检查以下视频关键帧与导航任务。分别判断 scene consistency 和 navigation alignment，"
            "确认目标身份、目标可见率、有效起止时间，并给出 evidence。"
            "返回字段：decision(pass/review/reject), scene_consistency, instruction_alignment, "
            "target_identity, target_visible_ratio, valid_interval([start,end]), confidence, "
            "observed_scene_label(受控标签或null), observed_scene_confidence(0到1或null), "
            "observed_scene_source(例如vlm_video或null), evidence([{time,reason_code}]), "
            "corrected_instruction(null或保留语义的改写), reason_codes。"
            "数值字段示例：scene_consistency:0.95, instruction_alignment:0.90, "
            "target_identity:1.0, target_visible_ratio:0.85, confidence:0.92。\n"
            f"场景与任务 JSON：{json.dumps(manifest, ensure_ascii=False)}"
        )
        text = self.client.generate_text(model=self.model, system=system, user_text=user, images=image_urls)
        data = _parse_json_object(text)
        duration = video_info.duration_s if video_info else sample.route.duration_s
        interval = data.get("valid_interval", [0.0, duration])
        malformed_fields: List[str] = []
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            interval = [0.0, duration]
            malformed_fields.append("valid_interval")
        start_time, start_bad = _coerce_time(interval[0], 0.0)
        end_time, end_bad = _coerce_time(interval[1], duration)
        if start_bad:
            malformed_fields.append("valid_interval[0]")
        if end_bad:
            malformed_fields.append("valid_interval[1]")
        score_values: Dict[str, float] = {}
        coerced_fields: List[str] = []
        semantic_coerced_fields: List[str] = []
        target = sample.scene.object_by_id(sample.task.target_id) if sample.task.target_id else None
        for field_name in (
            "scene_consistency",
            "instruction_alignment",
            "target_identity",
            "target_visible_ratio",
            "confidence",
        ):
            raw_value = data.get(field_name, 0.0)
            score, malformed = _coerce_unit_score(raw_value)
            # Seed VLMs sometimes use ``target_identity`` as a short
            # description (for example ``cabinet_01（四门餐边柜）``) despite
            # the numeric schema.  If that description explicitly names the
            # requested target, retain the semantic signal while recording a
            # schema-coercion audit marker.  Unknown prose remains a hard
            # review condition via ``malformed_numeric_output`` below.
            if malformed and field_name == "target_identity" and target is not None:
                text_value = str(raw_value).lower()
                hints = [str(target.id).lower(), str(target.category).lower()]
                if any(hint and hint in text_value for hint in hints):
                    score, malformed = 1.0, False
                    coerced_fields.append(field_name)
            if malformed:
                semantic_score, semantic_match = _coerce_semantic_score(field_name, raw_value)
                if semantic_match:
                    score, malformed = semantic_score, False
                    semantic_coerced_fields.append(field_name)
            score_values[field_name] = score
            if malformed:
                malformed_fields.append(field_name)
        raw_scene_label = data.get("observed_scene_label", data.get("scene_label"))
        observed_scene_label, scene_label_bad = _coerce_scene_label(raw_scene_label)
        if scene_label_bad:
            malformed_fields.append("observed_scene_label")
        observed_scene_confidence: Optional[float] = None
        raw_scene_confidence = data.get("observed_scene_confidence")
        if raw_scene_confidence is not None:
            observed_scene_confidence, scene_confidence_bad = _coerce_unit_score(raw_scene_confidence)
            if scene_confidence_bad:
                malformed_fields.append("observed_scene_confidence")
        observed_scene_source = data.get("observed_scene_source")
        if observed_scene_source is not None and not isinstance(observed_scene_source, str):
            observed_scene_source = None
            malformed_fields.append("observed_scene_source")
        elif isinstance(observed_scene_source, str):
            # The response itself came from the VLM.  Do not allow it to claim
            # a human/simulator source; those annotations must be attached by
            # an external audited step.
            observed_scene_source = "vlm_video"
        elif observed_scene_label is not None and observed_scene_confidence is not None:
            observed_scene_source = "vlm_video"
        decision = str(data.get("decision", "review")).strip().lower()
        if decision not in {"pass", "review", "reject"}:
            decision = "review"
            malformed_fields.append("decision")
        report = VLMReport(
            decision=decision,
            scene_consistency=score_values["scene_consistency"],
            instruction_alignment=score_values["instruction_alignment"],
            target_identity=score_values["target_identity"],
            target_visible_ratio=score_values["target_visible_ratio"],
            valid_interval=(start_time, end_time),
            confidence=score_values["confidence"],
            evidence=list(data.get("evidence", [])) if isinstance(data.get("evidence", []), list) else [],
            corrected_instruction=data.get("corrected_instruction"),
            reason_codes=list(data.get("reason_codes", [])) if isinstance(data.get("reason_codes", []), list) else [],
            source="doubao-vlm",
            observed_scene_label=observed_scene_label,
            observed_scene_confidence=observed_scene_confidence,
            observed_scene_source=(str(observed_scene_source).strip() or None) if isinstance(observed_scene_source, str) else None,
        )
        errors = report.validate()
        if errors:
            raise ValueError(f"invalid VLM JSON output: {errors}")
        if malformed_fields:
            # Preserve the model's usable fields but never call malformed
            # output a clean pass.  The exact fields are retained as evidence
            # so a later prompt/model revision can be audited.
            report.reason_codes.append("malformed_numeric_output")
            report.evidence.append(
                {"reason_code": "malformed_numeric_output", "fields": sorted(set(malformed_fields))}
            )
            if report.decision == "pass":
                report.decision = "review"
        if coerced_fields:
            report.reason_codes.append("coerced_target_identity_description")
            report.evidence.append(
                {"reason_code": "coerced_target_identity_description", "fields": sorted(set(coerced_fields))}
            )
        if semantic_coerced_fields:
            report.reason_codes.append("coerced_semantic_score")
            report.evidence.append(
                {"reason_code": "coerced_semantic_score", "fields": sorted(set(semantic_coerced_fields))}
            )
        # Do not trust a model-provided ``decision`` in isolation.  A model
        # may accidentally emit ``pass`` with a low score; enforce the same
        # gates used by the local judge and retain explicit reason codes for
        # auditability.
        failed_gates: List[str] = []
        if report.scene_consistency < self.thresholds.scene_pass:
            failed_gates.append("temporal_scene_drift")
        if report.instruction_alignment < self.thresholds.alignment_pass:
            failed_gates.append("instruction_alignment_uncertain")
        if report.target_identity < self.thresholds.identity_pass:
            failed_gates.append("target_identity_uncertain")
        if sample.task.type == "target_tracking" and report.target_visible_ratio < self.thresholds.tracking_visible_ratio:
            failed_gates.append("target_visibility_gap")
        if report.confidence < self.thresholds.confidence_pass:
            failed_gates.append("low_vlm_confidence")
        for reason in failed_gates:
            if reason not in report.reason_codes:
                report.reason_codes.append(reason)
        if failed_gates and report.decision == "pass":
            report.decision = "review" if report.confidence >= self.thresholds.review_floor else "reject"
        return report
