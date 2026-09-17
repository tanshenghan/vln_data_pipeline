"""Kinematic, geometric and task-level trajectory quality gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import PoseSample, RouteContract, StopCondition
from .trajectory import TrajectoryGenerator, trajectory_metrics, wrap_angle


@dataclass
class FilterThresholds:
    speed_tolerance: float = 0.08
    accel_tolerance: float = 0.12
    angular_speed_tolerance: float = 0.12
    angular_accel_tolerance: float = 0.15
    jitter_reversal_limit: int = 8
    stop_speed_mps: float = 0.06
    stop_omega_rps: float = 0.10
    min_duration_s: float = 0.5


def _distance_to_rect(x: float, y: float, rect: Dict[str, Any]) -> float:
    x_min = float(rect.get("x_min", -math.inf))
    x_max = float(rect.get("x_max", math.inf))
    y_min = float(rect.get("y_min", -math.inf))
    y_max = float(rect.get("y_max", math.inf))
    dx = max(x_min - x, 0.0, x - x_max)
    dy = max(y_min - y, 0.0, y - y_max)
    outside = math.hypot(dx, dy)
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return -min(x - x_min, x_max - x, y - y_min, y_max - y)
    return outside


def _min_clearance(trajectory: Sequence[PoseSample], obstacles: Sequence[Dict[str, Any]]) -> float:
    if not obstacles or not trajectory:
        return math.inf
    # Check points between samples as well as the samples themselves.  At a
    # low trajectory rate a fast segment could otherwise "tunnel" through a
    # thin obstacle and falsely pass the geometry gate.
    minimum = math.inf
    for left, right in zip(trajectory, trajectory[1:]):
        for step in range(6):
            amount = step / 5.0
            x = left.x + amount * (right.x - left.x)
            y = left.y + amount * (right.y - left.y)
            for obstacle in obstacles:
                minimum = min(minimum, _distance_to_rect(x, y, obstacle))
    if len(trajectory) == 1:
        for obstacle in obstacles:
            minimum = min(minimum, _distance_to_rect(trajectory[0].x, trajectory[0].y, obstacle))
    return minimum


def _direction_reversals(trajectory: Sequence[PoseSample]) -> int:
    signs: List[int] = []
    for sample in trajectory:
        speed = math.hypot(sample.vx, sample.vy)
        if speed < 1e-4:
            continue
        projection = sample.vx * math.cos(sample.yaw) + sample.vy * math.sin(sample.yaw)
        sign = 1 if projection >= 0 else -1
        if not signs or sign != signs[-1]:
            signs.append(sign)
    return max(0, len(signs) - 1)


def _stop_duration(trajectory: Sequence[PoseSample], thresholds: FilterThresholds) -> float:
    if not trajectory:
        return 0.0
    end = trajectory[-1].t
    start = end
    for sample in reversed(trajectory):
        if math.hypot(sample.vx, sample.vy) <= thresholds.stop_speed_mps and abs(sample.omega) <= thresholds.stop_omega_rps:
            start = sample.t
        else:
            break
    return max(0.0, end - start)


def _smooth_positions(trajectory: Sequence[PoseSample], radius: int = 2) -> List[Tuple[float, float, float, float]]:
    if len(trajectory) < 3:
        return [(s.t, s.x, s.y, s.yaw) for s in trajectory]
    result: List[Tuple[float, float, float, float]] = []
    for index, sample in enumerate(trajectory):
        if index == 0 or index == len(trajectory) - 1:
            result.append((sample.t, sample.x, sample.y, sample.yaw))
            continue
        lo = max(0, index - radius)
        hi = min(len(trajectory), index + radius + 1)
        window = trajectory[lo:hi]
        mean_x = sum(item.x for item in window) / len(window)
        mean_y = sum(item.y for item in window) / len(window)
        # Circular mean for yaw.
        mean_yaw = math.atan2(sum(math.sin(item.yaw) for item in window), sum(math.cos(item.yaw) for item in window))
        result.append((sample.t, mean_x, mean_y, mean_yaw))
    return result


class TrajectoryFilter:
    def __init__(self, thresholds: Optional[FilterThresholds] = None) -> None:
        self.thresholds = thresholds or FilterThresholds()
        self.generator = TrajectoryGenerator(sample_hz=10.0)

    def evaluate(
        self,
        trajectory: Sequence[PoseSample],
        route: RouteContract,
        stop_condition: Optional[StopCondition] = None,
        target_position: Optional[Tuple[float, float]] = None,
        allow_review: bool = True,
        repair_soft_failures: bool = True,
        source: str = "route_contract",
    ) -> Tuple[TrajectoryReportLike, List[PoseSample]]:
        """Evaluate a trajectory and return a report plus possibly repaired path."""

        report, repaired = self._evaluate_once(
            trajectory,
            route,
            stop_condition=stop_condition,
            target_position=target_position,
            allow_review=allow_review,
            source=source,
        )
        # Only repair soft smoothness violations.  Never smooth away a
        # collision or a hard kinematic violation.
        repairable = (
            repair_soft_failures
            and set(report.hard_failures).issubset({"jitter", "high_direction_reversal"})
            and bool(report.hard_failures)
        )
        if repairable:
            smoothed = _smooth_positions(trajectory)
            if len(smoothed) >= 2:
                try:
                    repaired_trajectory = self.generator.reconstruct_from_positions(
                        smoothed, confidence=min(s.confidence for s in trajectory), source=source
                    )
                    second, _ = self._evaluate_once(
                        repaired_trajectory,
                        route,
                        stop_condition=stop_condition,
                        target_position=target_position,
                        allow_review=allow_review,
                        source=source,
                    )
                    if second.decision == "pass":
                        second.repaired = True
                        return second, repaired_trajectory
                except Exception:
                    pass
        return report, list(repaired)

    def _evaluate_once(
        self,
        trajectory: Sequence[PoseSample],
        route: RouteContract,
        stop_condition: Optional[StopCondition],
        target_position: Optional[Tuple[float, float]],
        allow_review: bool,
        source: str,
    ) -> Tuple["TrajectoryReport", List[PoseSample]]:
        from .models import TrajectoryReport

        hard: List[str] = []
        warnings: List[str] = []
        if len(trajectory) < 2:
            hard.append("too_few_samples")
            return TrajectoryReport("reject", {}, hard, warnings, confidence=0.0, source=source), list(trajectory)
        times = [sample.t for sample in trajectory]
        if any(right <= left for left, right in zip(times, times[1:])):
            hard.append("non_monotonic_timestamps")
        metrics = trajectory_metrics(trajectory)
        if metrics["duration_s"] < self.thresholds.min_duration_s:
            hard.append("duration_too_short")
        speed_limit = route.max_speed_mps * (1.0 + self.thresholds.speed_tolerance)
        accel_limit = route.max_accel_mps2 * (1.0 + self.thresholds.accel_tolerance)
        omega_limit = route.max_angular_speed_rps * (1.0 + self.thresholds.angular_speed_tolerance)
        alpha_limit = route.max_angular_accel_rps2 * (1.0 + self.thresholds.angular_accel_tolerance)
        if metrics["max_speed_mps"] > speed_limit:
            hard.append("abnormal_speed")
        if metrics["max_accel_mps2"] > accel_limit:
            hard.append("abnormal_acceleration")
        if metrics["max_angular_speed_rps"] > omega_limit:
            hard.append("abnormal_angular_speed")
        max_alpha = max((abs(sample.alpha) for sample in trajectory), default=0.0)
        metrics["max_angular_accel_rps2"] = float(max_alpha)
        if max_alpha > alpha_limit:
            hard.append("abnormal_angular_acceleration")
        reversals = _direction_reversals(trajectory)
        if reversals > self.thresholds.jitter_reversal_limit:
            hard.append("high_direction_reversal")
        if metrics["max_jerk_mps3"] > max(10.0, route.max_accel_mps2 * 15.0):
            hard.append("jitter")

        clearance = _min_clearance(trajectory, route.obstacles)
        if clearance < route.clearance_m:
            hard.append("collision_or_insufficient_clearance")
        if not math.isfinite(clearance):
            warnings.append("no_obstacle_geometry_available")

        if target_position is not None and stop_condition is not None:
            end = trajectory[-1]
            target_distance = math.hypot(end.x - target_position[0], end.y - target_position[1])
            metrics["endpoint_target_distance_m"] = target_distance
            if not (stop_condition.distance_min_m <= target_distance <= stop_condition.distance_max_m):
                hard.append("endpoint_outside_goal_region")
            stop_duration = _stop_duration(trajectory, self.thresholds)
            metrics["stop_duration_s"] = stop_duration
            if stop_duration + 1e-6 < stop_condition.dwell_s:
                hard.append("insufficient_stop_dwell")
        else:
            warnings.append("goal_geometry_not_checked")

        confidence = 1.0
        if warnings:
            confidence -= min(0.35, 0.08 * len(warnings))
        if hard:
            confidence -= min(0.8, 0.15 * len(hard))
        confidence = max(0.0, min(1.0, confidence))
        if hard:
            decision = "reject"
        elif warnings and allow_review:
            decision = "review"
        else:
            decision = "pass"
        report = TrajectoryReport(
            decision=decision,
            metrics={**metrics, "direction_reversals": float(reversals), "min_clearance_m": float(clearance) if math.isfinite(clearance) else -1.0},
            hard_failures=hard,
            warnings=warnings,
            confidence=confidence,
            source=source,
        )
        return report, list(trajectory)


# A forward declaration alias keeps type checkers happy without introducing a
# runtime circular import in the public method signature.
TrajectoryReportLike = Any
