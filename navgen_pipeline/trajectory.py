"""Trajectory generation and lightweight reconstruction primitives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import PoseSample, RouteContract, Waypoint


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _interpolate_angle(left: float, right: float, amount: float) -> float:
    return wrap_angle(left + amount * wrap_angle(right - left))


def interpolate_route(route: RouteContract, t: float) -> Tuple[float, float, float]:
    points = route.waypoints
    if not points:
        return 0.0, 0.0, 0.0
    if t <= points[0].t:
        point = points[0]
        return point.x, point.y, point.yaw
    if t >= points[-1].t:
        point = points[-1]
        return point.x, point.y, point.yaw
    for left, right in zip(points, points[1:]):
        if left.t <= t <= right.t:
            span = right.t - left.t
            u = (t - left.t) / span if span > 0 else 0.0
            # Cubic smoothstep gives zero velocity at each anchor while
            # keeping the peak speed/acceleration bounded for short indoor
            # clips.  (A quintic minimum-jerk curve is smoother in theory,
            # but its 1.875x peak-speed factor makes a 4--15 s video reject
            # otherwise reasonable routes.)
            s = u * u * (3.0 - 2.0 * u)
            return (
                left.x + s * (right.x - left.x),
                left.y + s * (right.y - left.y),
                _interpolate_angle(left.yaw, right.yaw, s),
            )
    last = points[-1]
    return last.x, last.y, last.yaw


def _times(duration_s: float, hz: float) -> List[float]:
    if duration_s <= 0 or hz <= 0:
        raise ValueError("duration and hz must be positive")
    count = max(2, int(round(duration_s * hz)) + 1)
    return [min(duration_s, index / hz) for index in range(count)]


class TrajectoryGenerator:
    def __init__(self, sample_hz: float = 10.0) -> None:
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self.sample_hz = sample_hz

    def generate(self, route: RouteContract, confidence: float = 1.0, source: str = "route_contract") -> List[PoseSample]:
        errors = route.validate()
        if errors:
            raise ValueError(f"invalid route: {errors}")
        times = _times(route.duration_s, self.sample_hz)
        positions = [interpolate_route(route, t) for t in times]
        samples = [
            PoseSample(t=t, x=p[0], y=p[1], yaw=p[2], confidence=confidence, source=source)
            for t, p in zip(times, positions)
        ]

        def derivative(values: Sequence[float]) -> List[float]:
            result: List[float] = []
            for i in range(len(values)):
                if i == 0:
                    dt = max(1e-9, times[1] - times[0])
                    result.append((values[1] - values[0]) / dt)
                elif i == len(values) - 1:
                    dt = max(1e-9, times[-1] - times[-2])
                    result.append((values[-1] - values[-2]) / dt)
                else:
                    dt = max(1e-9, times[i + 1] - times[i - 1])
                    result.append((values[i + 1] - values[i - 1]) / dt)
            return result

        vx = derivative([p[0] for p in positions])
        vy = derivative([p[1] for p in positions])
        ax = derivative(vx)
        ay = derivative(vy)
        unwrapped_yaw: List[float] = [positions[0][2]]
        for _, _, yaw in positions[1:]:
            unwrapped_yaw.append(unwrapped_yaw[-1] + wrap_angle(yaw - unwrapped_yaw[-1]))
        omega = derivative(unwrapped_yaw)
        alpha = derivative(omega)
        for i, sample in enumerate(samples):
            sample.vx = vx[i]
            sample.vy = vy[i]
            sample.ax = ax[i]
            sample.ay = ay[i]
            sample.omega = omega[i]
            sample.alpha = alpha[i]
        return samples

    def reconstruct_from_positions(
        self,
        observations: Sequence[Tuple[float, float, float, float]],
        confidence: float = 0.5,
        source: str = "reconstructed",
    ) -> List[PoseSample]:
        """Convert timestamped (t, x, y, yaw) observations into a trajectory.

        This is intentionally conservative: it does not invent metric scale.
        Callers should set a low confidence when the observations came from
        monocular video without a scale anchor.
        """

        if len(observations) < 2:
            raise ValueError("at least two observations are required")
        points = [Waypoint(float(t), float(x), float(y), float(yaw)) for t, x, y, yaw in observations]
        route = RouteContract(
            waypoints=points,
            expected_duration_s=points[-1].t,
            max_speed_mps=1e6,
            max_accel_mps2=1e6,
            max_angular_speed_rps=1e6,
            max_angular_accel_rps2=1e6,
        )
        return self.generate(route, confidence=confidence, source=source)


def pose_to_dict(sample: PoseSample) -> Dict[str, Any]:
    return {
        "t": sample.t,
        "x": sample.x,
        "y": sample.y,
        "z": sample.z,
        "yaw": sample.yaw,
        "vx": sample.vx,
        "vy": sample.vy,
        "vz": sample.vz,
        "ax": sample.ax,
        "ay": sample.ay,
        "az": sample.az,
        "omega": sample.omega,
        "alpha": sample.alpha,
        "confidence": sample.confidence,
        "frame_id": sample.frame_id,
        "source": sample.source,
    }


def write_trajectory_jsonl(path: Path, trajectory: Sequence[PoseSample]) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in trajectory:
            handle.write(json.dumps(pose_to_dict(sample), ensure_ascii=False) + "\n")
    return path


def read_trajectory_jsonl(path: Path | str) -> List[PoseSample]:
    """Read an externally recorded pose stream.

    The importer intentionally accepts the small common subset ``t, x, y,
    yaw`` and preserves optional velocity/acceleration fields when present.
    It is suitable for a JSONL export made from ROS ``/tf``/``/odom`` or a
    simulator.  No interpolation or metric scale is invented here; callers
    must provide poses in a declared coordinate frame.
    """

    import json

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"trajectory file does not exist: {source}")
    samples: List[PoseSample] = []
    has_velocity_fields = False
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trajectory JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"trajectory line {line_number} must be an object")
            missing = [name for name in ("t", "x", "y", "yaw") if name not in value]
            if missing:
                raise ValueError(f"trajectory line {line_number} is missing: {', '.join(missing)}")
            try:
                has_velocity_fields = has_velocity_fields or any(name in value for name in ("vx", "vy", "omega"))
                sample = PoseSample(
                    t=float(value["t"]),
                    x=float(value["x"]),
                    y=float(value["y"]),
                    z=float(value.get("z", 0.0)),
                    yaw=float(value["yaw"]),
                    vx=float(value.get("vx", 0.0)),
                    vy=float(value.get("vy", 0.0)),
                    vz=float(value.get("vz", 0.0)),
                    ax=float(value.get("ax", 0.0)),
                    ay=float(value.get("ay", 0.0)),
                    az=float(value.get("az", 0.0)),
                    omega=float(value.get("omega", 0.0)),
                    alpha=float(value.get("alpha", 0.0)),
                    confidence=float(value.get("confidence", 1.0)),
                    frame_id=str(value.get("frame_id", "map")),
                    source=str(value.get("source", "")),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"trajectory line {line_number} contains non-numeric pose values") from exc
            values = (sample.t, sample.x, sample.y, sample.z, sample.yaw)
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"trajectory line {line_number} contains non-finite pose values")
            if samples and sample.t <= samples[-1].t:
                raise ValueError("trajectory timestamps must be strictly increasing")
            samples.append(sample)
    if len(samples) < 2:
        raise ValueError("trajectory file must contain at least two poses")
    if not has_velocity_fields:
        _derive_kinematics(samples)
    return samples


def _derive_kinematics(samples: Sequence[PoseSample]) -> None:
    """Fill velocity/acceleration fields for pose files containing positions only."""

    if len(samples) < 2:
        return

    def derivative(values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for index in range(len(values)):
            if index == 0:
                dt = max(1e-9, samples[1].t - samples[0].t)
                result.append((values[1] - values[0]) / dt)
            elif index == len(values) - 1:
                dt = max(1e-9, samples[-1].t - samples[-2].t)
                result.append((values[-1] - values[-2]) / dt)
            else:
                dt = max(1e-9, samples[index + 1].t - samples[index - 1].t)
                result.append((values[index + 1] - values[index - 1]) / dt)
        return result

    vx = derivative([sample.x for sample in samples])
    vy = derivative([sample.y for sample in samples])
    unwrapped = [samples[0].yaw]
    for sample in samples[1:]:
        unwrapped.append(unwrapped[-1] + wrap_angle(sample.yaw - unwrapped[-1]))
    omega = derivative(unwrapped)
    ax = derivative(vx)
    ay = derivative(vy)
    alpha = derivative(omega)
    for index, sample in enumerate(samples):
        sample.vx, sample.vy = vx[index], vy[index]
        sample.ax, sample.ay = ax[index], ay[index]
        sample.omega, sample.alpha = omega[index], alpha[index]


def read_trajectory_file(path: Path | str) -> List[PoseSample]:
    """Read either NavGen ``trajectory.jsonl`` or its JSON download artifact."""

    import json

    source = Path(path)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_trajectory_jsonl(source)
    if source.suffix.lower() != ".json":
        raise ValueError("trajectory file must use .jsonl, .ndjson or .json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid trajectory JSON: {exc.msg}") from exc
    points: Any = value.get("points") if isinstance(value, Mapping) else value
    if not isinstance(points, list):
        raise ValueError("trajectory JSON must contain a points array")
    # Reuse the strict JSONL parser so both formats have identical validation.
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        temporary = Path(handle.name)
        for point in points:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")
    try:
        return read_trajectory_jsonl(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def trajectory_metrics(trajectory: Sequence[PoseSample]) -> Dict[str, float]:
    if not trajectory:
        return {"duration_s": 0.0, "path_length_m": 0.0, "max_speed_mps": 0.0, "max_accel_mps2": 0.0, "max_angular_speed_rps": 0.0, "max_jerk_mps3": 0.0}
    path_length = 0.0
    max_speed = 0.0
    max_accel = 0.0
    max_omega = 0.0
    max_jerk = 0.0
    previous_accel: Optional[Tuple[float, float]] = None
    previous_time = trajectory[0].t
    for left, right in zip(trajectory, trajectory[1:]):
        path_length += math.hypot(right.x - left.x, right.y - left.y)
    for sample in trajectory:
        speed = math.hypot(sample.vx, sample.vy)
        accel = math.hypot(sample.ax, sample.ay)
        max_speed = max(max_speed, speed)
        max_accel = max(max_accel, accel)
        max_omega = max(max_omega, abs(sample.omega))
        if previous_accel is not None:
            dt = max(1e-6, sample.t - previous_time)
            max_jerk = max(max_jerk, math.hypot(sample.ax - previous_accel[0], sample.ay - previous_accel[1]) / dt)
        previous_accel = (sample.ax, sample.ay)
        previous_time = sample.t
    return {
        "duration_s": max(0.0, trajectory[-1].t - trajectory[0].t),
        "path_length_m": path_length,
        "max_speed_mps": max_speed,
        "max_accel_mps2": max_accel,
        "max_angular_speed_rps": max_omega,
        "max_jerk_mps3": max_jerk,
    }
