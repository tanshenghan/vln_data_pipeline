"""Serializable data contracts used by the navigation data pipeline.

The project deliberately uses dataclasses instead of a mandatory validation
dependency so that the offline smoke test works in a minimal Python image.
The ``validate`` methods are strict enough to catch the common label/route
mistakes before an expensive video request is submitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _as_float_tuple(values: Sequence[float], length: int, name: str) -> Tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(float(v) for v in values)


@dataclass
class ObjectSpec:
    id: str
    category: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (0.5, 0.5)
    dynamic: bool = False
    color_rgb: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        self.position = _as_float_tuple(self.position, 2, "position")  # type: ignore[assignment]
        self.size = _as_float_tuple(self.size, 2, "size")  # type: ignore[assignment]
        if not self.id or not self.category:
            raise ValueError("object id and category are required")
        if any(v <= 0 for v in self.size):
            raise ValueError("object size must be positive")
        if self.color_rgb is not None:
            if len(self.color_rgb) != 3 or any(not 0 <= int(v) <= 255 for v in self.color_rgb):
                raise ValueError("color_rgb must contain three values in [0, 255]")
            self.color_rgb = tuple(int(v) for v in self.color_rgb)  # type: ignore[assignment]


@dataclass
class SceneManifest:
    scene_id: str
    room_type: str
    objects: List[ObjectSpec] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    width_m: float = 8.0
    depth_m: float = 6.0
    camera_height_m: float = 1.2
    fps: int = 24
    reference_images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional semantic regions.  A scene may intentionally be one open
    # space; consumers must not invent room boundaries when this is empty.
    regions: List[Dict[str, Any]] = field(default_factory=list)
    # ``room_type`` is the planner's prior.  These fields are populated only
    # when a VLM, human annotator, or simulator provides an observed semantic
    # label; keeping the provenance separate prevents a prompt prior from
    # being rendered as visual ground truth.
    semantic_label: Optional[str] = None
    semantic_confidence: Optional[float] = None
    semantic_source: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.scene_id:
            errors.append("scene_id is empty")
        if not self.room_type:
            errors.append("room_type is empty")
        if self.width_m <= 0 or self.depth_m <= 0:
            errors.append("scene dimensions must be positive")
        if self.camera_height_m <= 0:
            errors.append("camera_height_m must be positive")
        if self.fps <= 0:
            errors.append("fps must be positive")
        ids = [obj.id for obj in self.objects]
        if len(ids) != len(set(ids)):
            errors.append("object ids must be unique")
        for index, obstacle in enumerate(self.obstacles):
            try:
                x_min, x_max = float(obstacle["x_min"]), float(obstacle["x_max"])
                y_min, y_max = float(obstacle["y_min"]), float(obstacle["y_max"])
                if x_max <= x_min or y_max <= y_min:
                    errors.append(f"obstacle {index} has invalid bounds")
            except (KeyError, TypeError, ValueError):
                errors.append(f"obstacle {index} is missing numeric bounds")
        for index, region in enumerate(self.regions):
            try:
                x_min, x_max = float(region["x_min"]), float(region["x_max"])
                y_min, y_max = float(region["y_min"]), float(region["y_max"])
                if x_max <= x_min or y_max <= y_min:
                    errors.append(f"region {index} has invalid bounds")
                if x_min < 0 or y_min < 0 or x_max > self.width_m or y_max > self.depth_m:
                    errors.append(f"region {index} is outside scene bounds")
                if not (region.get("label") or region.get("name")):
                    errors.append(f"region {index} is missing label/name")
            except (KeyError, TypeError, ValueError):
                errors.append(f"region {index} is missing numeric bounds")
        if self.semantic_confidence is not None:
            try:
                confidence = float(self.semantic_confidence)
                if not 0.0 <= confidence <= 1.0:
                    errors.append("semantic_confidence must be in [0, 1]")
            except (TypeError, ValueError):
                errors.append("semantic_confidence must be numeric")
        if self.semantic_label is not None and not str(self.semantic_label).strip():
            errors.append("semantic_label cannot be empty")
        return errors

    def object_by_id(self, object_id: str) -> Optional[ObjectSpec]:
        return next((obj for obj in self.objects if obj.id == object_id), None)


@dataclass
class StopCondition:
    distance_min_m: float = 0.6
    distance_max_m: float = 1.6
    face_target: bool = True
    dwell_s: float = 1.0
    target_visible_ratio: float = 0.8

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.distance_min_m < 0 or self.distance_max_m < self.distance_min_m:
            errors.append("invalid stop distance range")
        if self.dwell_s < 0:
            errors.append("dwell_s cannot be negative")
        if not 0 <= self.target_visible_ratio <= 1:
            errors.append("target_visible_ratio must be in [0, 1]")
        return errors


@dataclass
class TaskSpec:
    type: str
    target_id: Optional[str]
    instruction_zh: str
    canonical_actions: List[str] = field(default_factory=list)
    start_pose: Tuple[float, float, float] = (0.8, 0.8, 0.0)
    stop_condition: StopCondition = field(default_factory=StopCondition)
    difficulty: Dict[str, Any] = field(default_factory=dict)
    instruction_original: Optional[str] = None

    def __post_init__(self) -> None:
        self.start_pose = _as_float_tuple(self.start_pose, 3, "start_pose")  # type: ignore[assignment]
        if self.instruction_original is None:
            self.instruction_original = self.instruction_zh


@dataclass
class Waypoint:
    t: float
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.x = float(self.x)
        self.y = float(self.y)
        self.yaw = float(self.yaw)


@dataclass
class RouteContract:
    waypoints: List[Waypoint]
    robot_profile: str = "indoor_diff_drive_v1"
    max_speed_mps: float = 0.7
    max_accel_mps2: float = 0.8
    max_angular_speed_rps: float = 1.2
    max_angular_accel_rps2: float = 2.0
    clearance_m: float = 0.18
    goal_region: Dict[str, Any] = field(default_factory=dict)
    expected_duration_s: float = 8.0
    obstacles: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.waypoints) < 2:
            errors.append("at least two waypoints are required")
        previous_t: Optional[float] = None
        for index, point in enumerate(self.waypoints):
            if point.t < 0:
                errors.append(f"waypoint {index} has negative time")
            if previous_t is not None and point.t <= previous_t:
                errors.append("waypoint timestamps must be strictly increasing")
            previous_t = point.t
        for name in (
            "max_speed_mps",
            "max_accel_mps2",
            "max_angular_speed_rps",
            "max_angular_accel_rps2",
        ):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.expected_duration_s <= 0:
            errors.append("expected_duration_s must be positive")
        if self.clearance_m < 0:
            errors.append("clearance_m cannot be negative")
        for index, obstacle in enumerate(self.obstacles):
            try:
                x_min, x_max = float(obstacle["x_min"]), float(obstacle["x_max"])
                y_min, y_max = float(obstacle["y_min"]), float(obstacle["y_max"])
                if x_max <= x_min or y_max <= y_min:
                    errors.append(f"route obstacle {index} has invalid bounds")
            except (KeyError, TypeError, ValueError):
                errors.append(f"route obstacle {index} is missing numeric bounds")
        return errors

    @property
    def duration_s(self) -> float:
        return self.waypoints[-1].t if self.waypoints else self.expected_duration_s


@dataclass
class PoseSample:
    t: float
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    omega: float = 0.0
    alpha: float = 0.0
    confidence: float = 1.0
    # Coordinate-frame/provenance hints are optional for backward
    # compatibility with the original route JSONL contract.
    frame_id: str = "map"
    source: str = ""


@dataclass
class SampleSpec:
    sample_id: str
    scene: SceneManifest
    task: TaskSpec
    route: RouteContract
    seedance_prompt: str
    seed: int
    planner_source: str = "template"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors = self.scene.validate() + self.task.stop_condition.validate() + self.route.validate()
        if self.task.type not in {"object_goal", "instruction_following", "target_tracking"}:
            errors.append(f"unsupported task type: {self.task.type}")
        if not self.task.instruction_zh.strip():
            errors.append("task instruction is empty")
        if self.task.target_id and self.scene.object_by_id(self.task.target_id) is None:
            errors.append(f"target_id {self.task.target_id!r} does not exist in scene")
        if not self.seedance_prompt.strip():
            errors.append("seedance_prompt is empty")
        if self.seed < 0:
            errors.append("seed must be non-negative for reproducible generation")
        return errors


@dataclass
class VLMReport:
    decision: str
    scene_consistency: float
    instruction_alignment: float
    target_identity: float
    target_visible_ratio: float
    valid_interval: Tuple[float, float]
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    corrected_instruction: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)
    source: str = "heuristic"
    # Optional observed scene semantics.  These are deliberately separate
    # from the planner's ``SceneManifest.room_type`` prior.  A label is only
    # promoted to the manifest when the VLM supplies sufficient confidence and
    # an auditable source.
    observed_scene_label: Optional[str] = None
    observed_scene_confidence: Optional[float] = None
    observed_scene_source: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.decision not in {"pass", "review", "reject"}:
            errors.append("decision must be pass, review or reject")
        for name in (
            "scene_consistency",
            "instruction_alignment",
            "target_identity",
            "target_visible_ratio",
            "confidence",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                errors.append(f"{name} must be in [0, 1]")
        if self.valid_interval[1] < self.valid_interval[0]:
            errors.append("invalid valid_interval")
        if self.observed_scene_confidence is not None:
            try:
                confidence = float(self.observed_scene_confidence)
                if not 0.0 <= confidence <= 1.0:
                    errors.append("observed_scene_confidence must be in [0, 1]")
            except (TypeError, ValueError):
                errors.append("observed_scene_confidence must be numeric")
        if self.observed_scene_label is not None and not str(self.observed_scene_label).strip():
            errors.append("observed_scene_label cannot be empty")
        return errors


@dataclass
class TrajectoryReport:
    decision: str
    metrics: Dict[str, float]
    hard_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repaired: bool = False
    confidence: float = 1.0
    source: str = "route_contract"
    # A route contract is a plan/prior, not an observed robot pose.  Keep
    # provenance explicit so a dashboard or dataset consumer cannot silently
    # treat a generated path as ground truth.
    ground_truth: bool = False
    metric_scale: str = "known"
    estimator: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Units/processing flags are kept at the report's top level so consumers
    # do not need to know the estimator-specific provenance schema.
    metric_units: str = "unknown"
    metrics_are_metric: bool = False
    temporal_smoothing: bool = False
    scale_anchor: str = "none"
    # Name of the deterministic post-processing variant, when applicable
    # (for example ``smoothed_scaled`` for the production VGGT path).
    variant: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.decision not in {"pass", "review", "reject"}:
            errors.append("decision must be pass, review or reject")
        if not 0 <= self.confidence <= 1:
            errors.append("confidence must be in [0, 1]")
        return errors


@dataclass
class SampleRecord:
    sample_id: str
    status: str
    sample_spec: Dict[str, Any]
    pipeline_contract: Optional[Dict[str, Any]] = None
    # A is directly usable for training, B requires human/VLM review, and
    # reject is retained only for audit/debugging.
    quality_tier: str = "reject"
    video_path: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)
    contact_sheet_path: Optional[str] = None
    prompt_report: Optional[Dict[str, Any]] = None
    trajectory_path: Optional[str] = None
    trajectory_plot_path: Optional[str] = None
    vlm_report: Optional[Dict[str, Any]] = None
    trajectory_report: Optional[Dict[str, Any]] = None
    embedding_report: Optional[Dict[str, Any]] = None
    usage_report: Optional[Dict[str, Any]] = None
    # ``trajectory_path`` is the line-oriented source file.  These two
    # optional paths point to browser/download-friendly artifacts emitted by
    # the packager when an observed and/or planned path is available.
    trajectory_data_path: Optional[str] = None
    planned_trajectory_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)


def dataclass_to_dict(value: Any) -> Any:
    """Convert nested dataclasses into JSON-friendly dictionaries."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    return value


def dict_to_waypoints(items: Iterable[Mapping[str, Any]]) -> List[Waypoint]:
    return [Waypoint(**dict(item)) for item in items]
