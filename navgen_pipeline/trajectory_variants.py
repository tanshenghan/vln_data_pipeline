"""Deterministic post-processing variants for a raw VGGT trajectory.

The first VGGT integration deliberately returns an untouched relative camera
trajectory.  This module keeps that raw result as a comparison baseline while
the production pipeline selects ``smoothed_scaled``:

``raw``
    Original VGGT poses and finite-difference derivatives.
``smoothed``
    A small triangular moving average over position/yaw, with no scale fit.
``scaled``
    Raw poses mapped by an endpoint similarity transform to a declared route
    prior.  This is an assumption, not ground truth.
``smoothed_scaled``
    Smoothing followed by the same declared endpoint similarity transform.

All variants preserve timestamps and ``ground_truth=False``.  Scale-anchored
variants are explicitly marked ``metrics_are_metric=False`` because a route
contract is not an external pose measurement.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import PoseSample
from .trajectory import trajectory_metrics, wrap_angle


VGGTTrajectoryVariantName = str
VGGTTrajectoryVariantNames: Tuple[str, ...] = (
    "raw",
    "smoothed",
    "scaled",
    "smoothed_scaled",
)


@dataclass
class TrajectoryVariant:
    """One post-processing result and its provenance report."""

    name: VGGTTrajectoryVariantName
    trajectory: List[PoseSample]
    report: Dict[str, Any]


def _copy_with_kinematics(
    points: Sequence[PoseSample],
    *,
    source: str,
) -> List[PoseSample]:
    """Copy poses and derive finite-difference kinematics without filtering."""

    copied = [replace(point, source=source) for point in points]
    if len(copied) < 2:
        return copied

    def derivative(values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for index in range(len(values)):
            if index == 0:
                dt = max(1e-9, copied[1].t - copied[0].t)
                result.append((values[1] - values[0]) / dt)
            elif index == len(values) - 1:
                dt = max(1e-9, copied[-1].t - copied[-2].t)
                result.append((values[-1] - values[-2]) / dt)
            else:
                dt = max(1e-9, copied[index + 1].t - copied[index - 1].t)
                result.append((values[index + 1] - values[index - 1]) / dt)
        return result

    vx = derivative([point.x for point in copied])
    vy = derivative([point.y for point in copied])
    vz = derivative([point.z for point in copied])
    ax = derivative(vx)
    ay = derivative(vy)
    az = derivative(vz)
    unwrapped_yaw = [copied[0].yaw]
    for point in copied[1:]:
        unwrapped_yaw.append(unwrapped_yaw[-1] + wrap_angle(point.yaw - unwrapped_yaw[-1]))
    omega = derivative(unwrapped_yaw)
    alpha = derivative(omega)
    for index, point in enumerate(copied):
        point.vx, point.vy, point.vz = vx[index], vy[index], vz[index]
        point.ax, point.ay, point.az = ax[index], ay[index], az[index]
        point.omega, point.alpha = omega[index], alpha[index]
    return copied


def smooth_trajectory(
    trajectory: Sequence[PoseSample],
    *,
    window: int = 5,
    source: str = "vggt_monocular_estimate_smoothed",
) -> List[PoseSample]:
    """Apply a light triangular moving average while preserving edge samples.

    Keeping a small endpoint neighbourhood unchanged makes comparisons
    reproducible and avoids creating an artificial velocity spike at the
    first/last finite-difference sample.  The operation is intentionally
    simple and dependency-free; it is a post-processing experiment, not a
    learned smoother.
    """

    if window < 3:
        raise ValueError("smoothing window must be at least 3")
    if window % 2 == 0:
        window += 1
    if len(trajectory) < 3:
        return _copy_with_kinematics(trajectory, source=source)

    radius = min(window // 2, (len(trajectory) - 1) // 2)
    if radius < 1:
        return _copy_with_kinematics(trajectory, source=source)
    unwrapped_yaw = [trajectory[0].yaw]
    for point in trajectory[1:]:
        unwrapped_yaw.append(unwrapped_yaw[-1] + wrap_angle(point.yaw - unwrapped_yaw[-1]))

    smoothed: List[PoseSample] = []
    for index, point in enumerate(trajectory):
        # A moving average next to a hard endpoint can turn a gentle first
        # step into a large finite-difference velocity.  Keep ``radius``
        # samples at each edge and smooth only the interior where a complete
        # neighbourhood is available.
        if index < radius or index >= len(trajectory) - radius:
            smoothed.append(replace(point, source=source))
            continue
        lo = max(0, index - radius)
        hi = min(len(trajectory) - 1, index + radius)
        weighted: List[Tuple[PoseSample, float, float]] = []
        for neighbour in range(lo, hi + 1):
            distance = abs(neighbour - index)
            weight = float(radius + 1 - distance)
            weighted.append((trajectory[neighbour], weight, unwrapped_yaw[neighbour]))
        total = sum(item[1] for item in weighted)
        values = {
            "x": sum(item[0].x * item[1] for item in weighted) / total,
            "y": sum(item[0].y * item[1] for item in weighted) / total,
            "z": sum(item[0].z * item[1] for item in weighted) / total,
            "yaw": sum(item[2] * item[1] for item in weighted) / total,
        }
        smoothed.append(replace(point, **values, source=source))
    for point in smoothed:
        point.yaw = wrap_angle(point.yaw)
    return _copy_with_kinematics(smoothed, source=source)


def _endpoint_similarity(
    trajectory: Sequence[PoseSample],
    anchor_start: Tuple[float, float],
    anchor_end: Tuple[float, float],
    *,
    source: str,
) -> Tuple[List[PoseSample], Dict[str, Any]]:
    """Map a relative path to declared start/end coordinates.

    The transform uses the observed endpoint displacement for orientation and
    scale, then translates the first point to ``anchor_start``.  It is
    deliberately labelled as a route-prior anchor by callers.
    """

    if len(trajectory) < 2:
        return _copy_with_kinematics(trajectory, source=source), {
            "scale_factor": None,
            "rotation_rad": None,
            "available": False,
            "warning": "too_few_points_for_endpoint_anchor",
        }
    observed_start = (float(trajectory[0].x), float(trajectory[0].y))
    observed_end = (float(trajectory[-1].x), float(trajectory[-1].y))
    observed_dx = observed_end[0] - observed_start[0]
    observed_dy = observed_end[1] - observed_start[1]
    target_dx = float(anchor_end[0]) - float(anchor_start[0])
    target_dy = float(anchor_end[1]) - float(anchor_start[1])
    observed_norm = math.hypot(observed_dx, observed_dy)
    target_norm = math.hypot(target_dx, target_dy)
    if observed_norm < 1e-9 or target_norm < 1e-9:
        return _copy_with_kinematics(trajectory, source=source), {
            "scale_factor": None,
            "rotation_rad": None,
            "available": False,
            "warning": "endpoint_displacement_too_small_for_scale_anchor",
        }
    scale = target_norm / observed_norm
    rotation = math.atan2(target_dy, target_dx) - math.atan2(observed_dy, observed_dx)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    anchored: List[PoseSample] = []
    for point in trajectory:
        local_x = float(point.x) - observed_start[0]
        local_y = float(point.y) - observed_start[1]
        transformed_x = float(anchor_start[0]) + scale * (cosine * local_x - sine * local_y)
        transformed_y = float(anchor_start[1]) + scale * (sine * local_x + cosine * local_y)
        anchored.append(
            replace(
                point,
                x=transformed_x,
                y=transformed_y,
                z=float(point.z) * scale,
                yaw=wrap_angle(float(point.yaw) + rotation),
                source=source,
            )
        )
    return _copy_with_kinematics(anchored, source=source), {
        "scale_factor": float(scale),
        "rotation_rad": float(rotation),
        "available": True,
        "observed_start": [observed_start[0], observed_start[1]],
        "observed_end": [observed_end[0], observed_end[1]],
        "anchor_start": [float(anchor_start[0]), float(anchor_start[1])],
        "anchor_end": [float(anchor_end[0]), float(anchor_end[1])],
    }


def _variant_report(
    base_report: Mapping[str, Any],
    name: str,
    trajectory: Sequence[PoseSample],
    *,
    smoothing_window: Optional[int] = None,
    anchor: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy the VGGT report and append explicit post-processing metadata."""

    report: Dict[str, Any] = copy.deepcopy(dict(base_report))
    base_source = str(report.get("source", "vggt_monocular_estimate"))
    suffix = {
        "raw": "",
        "smoothed": "_smoothed",
        "scaled": "_scaled",
        "smoothed_scaled": "_smoothed_scaled",
    }.get(name, f"_{name}")
    report["base_source"] = "vggt_monocular_estimate"
    report["source"] = f"{base_source}{suffix}" if suffix else base_source
    report["variant"] = name
    report["ground_truth"] = False
    report["temporal_smoothing"] = name in {"smoothed", "smoothed_scaled"}
    report["smoothing_method"] = "triangular_moving_average" if report["temporal_smoothing"] else "none"
    if smoothing_window is not None:
        report["smoothing_window"] = int(smoothing_window)
    scaled = name in {"scaled", "smoothed_scaled"}
    report["scale_anchor"] = "route_endpoint_displacement" if scaled else "none"
    report["metric_scale"] = "route_endpoint_anchor" if scaled else "unknown"
    report["metric_units"] = "m" if scaled else "relative"
    # A route prior supplies an assumed scale; it is not an independent metric
    # measurement and must not be advertised as ground truth.
    report["metrics_are_metric"] = False
    report["scale_anchor_provenance"] = dict(anchor or {"available": False})
    warnings = list(report.get("warnings", []))
    if report["temporal_smoothing"]:
        # The raw report advertises that it was not smoothed.  Once a
        # variant has actually been filtered, retaining that warning would
        # make the provenance self-contradictory.
        warnings = [item for item in warnings if str(item) != "no_temporal_smoothing"]
        warnings.append("temporal_smoothing_applied")
    anchor_available = bool((anchor or {}).get("available", False))
    if scaled and anchor_available:
        # Likewise, a successful route-prior transform is no longer an
        # unanchored trajectory.  The explicit route-prior warning below
        # still makes clear that this is not an external metric measurement.
        warnings = [item for item in warnings if str(item) != "no_scale_anchor"]
        warnings.append("scale_anchor_uses_route_prior_not_ground_truth")
    elif scaled:
        warnings.append("no_scale_anchor")
    report["warnings"] = list(dict.fromkeys(str(item) for item in warnings))
    report["metrics"] = trajectory_metrics(trajectory)
    return report


def build_vggt_variants(
    raw_result: Any,
    *,
    anchor_start: Optional[Tuple[float, float]] = None,
    anchor_end: Optional[Tuple[float, float]] = None,
    smoothing_window: int = 5,
) -> Dict[str, TrajectoryVariant]:
    """Build raw/smoothed/scaled variants from one VGGT inference result.

    ``raw_result`` is intentionally duck-typed to avoid importing the VGGT
    model module at import time; it must expose ``trajectory`` and ``report``.
    When anchors are omitted, scaled variants remain unchanged and carry an
    explicit unavailable-anchor warning.
    """

    raw_trajectory = [replace(point) for point in raw_result.trajectory]
    base_report = copy.deepcopy(dict(raw_result.report))
    raw_report = _variant_report(base_report, "raw", raw_trajectory)
    variants: Dict[str, TrajectoryVariant] = {
        "raw": TrajectoryVariant("raw", raw_trajectory, raw_report)
    }
    smoothed = smooth_trajectory(
        raw_trajectory,
        window=smoothing_window,
        source="vggt_monocular_estimate_smoothed",
    )
    variants["smoothed"] = TrajectoryVariant(
        "smoothed",
        smoothed,
        _variant_report(base_report, "smoothed", smoothed, smoothing_window=smoothing_window),
    )

    anchor_available = anchor_start is not None and anchor_end is not None
    if anchor_available:
        scaled, anchor_meta = _endpoint_similarity(
            raw_trajectory,
            anchor_start,  # type: ignore[arg-type]
            anchor_end,  # type: ignore[arg-type]
            source="vggt_monocular_estimate_scaled",
        )
        smoothed_scaled, smoothed_anchor_meta = _endpoint_similarity(
            smoothed,
            anchor_start,  # type: ignore[arg-type]
            anchor_end,  # type: ignore[arg-type]
            source="vggt_monocular_estimate_smoothed_scaled",
        )
    else:
        scaled = _copy_with_kinematics(raw_trajectory, source="vggt_monocular_estimate_scaled")
        smoothed_scaled = _copy_with_kinematics(smoothed, source="vggt_monocular_estimate_smoothed_scaled")
        anchor_meta = {"available": False, "warning": "route_endpoint_anchor_not_provided"}
        smoothed_anchor_meta = dict(anchor_meta)
    variants["scaled"] = TrajectoryVariant(
        "scaled",
        scaled,
        _variant_report(base_report, "scaled", scaled, anchor=anchor_meta),
    )
    variants["smoothed_scaled"] = TrajectoryVariant(
        "smoothed_scaled",
        smoothed_scaled,
        _variant_report(
            base_report,
            "smoothed_scaled",
            smoothed_scaled,
            smoothing_window=smoothing_window,
            anchor=smoothed_anchor_meta,
        ),
    )
    # Make the unavailable case unambiguous: the report still has the variant
    # name but never claims a successful metric transform.
    if not anchor_available:
        for name in ("scaled", "smoothed_scaled"):
            variants[name].report["metric_scale"] = "unknown"
            variants[name].report["metric_units"] = "relative"
            variants[name].report["scale_anchor"] = "unavailable"
            variants[name].report["warnings"] = list(
                dict.fromkeys(
                    list(variants[name].report.get("warnings", []))
                    + ["route_endpoint_anchor_not_provided"]
                )
            )
    return variants


def variant_summary(
    variants: Mapping[str, TrajectoryVariant],
    *,
    reference: Optional[Sequence[PoseSample]] = None,
) -> List[Dict[str, Any]]:
    """Return compact comparable metrics for a UI/JSON report."""

    reference_end: Optional[Tuple[float, float]] = None
    reference_length: Optional[float] = None
    if reference:
        reference_end = (float(reference[-1].x), float(reference[-1].y))
        reference_length = float(trajectory_metrics(reference)["path_length_m"])
    rows: List[Dict[str, Any]] = []
    for name in VGGTTrajectoryVariantNames:
        variant = variants.get(name)
        if variant is None:
            continue
        metrics = variant.report.get("metrics", {})
        end_error = None
        # Raw/smoothed VGGT paths live in an arbitrary frame.  Comparing their
        # endpoint or length directly to a metre-valued route would fabricate
        # an accuracy number, so only an explicitly endpoint-anchored variant
        # gets reference-comparison metrics.
        reference_comparable = (
            reference_end is not None
            and variant.trajectory
            and variant.report.get("scale_anchor") == "route_endpoint_displacement"
            and bool(variant.report.get("scale_anchor_provenance", {}).get("available", False))
        )
        if reference_comparable:
            end_error = math.hypot(
                float(variant.trajectory[-1].x) - reference_end[0],
                float(variant.trajectory[-1].y) - reference_end[1],
            )
        path_length = metrics.get("path_length_m")
        rows.append(
            {
                "variant": name,
                "source": variant.report.get("source"),
                "ground_truth": bool(variant.report.get("ground_truth", False)),
                "metric_scale": variant.report.get("metric_scale", "unknown"),
                "metric_units": variant.report.get("metric_units", "relative"),
                "metrics_are_metric": bool(variant.report.get("metrics_are_metric", False)),
                "temporal_smoothing": bool(variant.report.get("temporal_smoothing", False)),
                "scale_anchor": variant.report.get("scale_anchor", "none"),
                "path_length": path_length,
                "max_speed": metrics.get("max_speed_mps"),
                "max_accel": metrics.get("max_accel_mps2"),
                "max_jerk": metrics.get("max_jerk_mps3"),
                "reference_comparison_valid": bool(reference_comparable),
                "endpoint_error_to_reference": end_error,
                "path_length_ratio_to_reference": (
                    float(path_length) / reference_length
                    if reference_comparable
                    and isinstance(path_length, (int, float))
                    and reference_length
                    and reference_length > 0
                    else None
                ),
                "warnings": list(variant.report.get("warnings", [])),
            }
        )
    return rows


def comparison_data_dict(
    variants: Mapping[str, TrajectoryVariant],
    *,
    source_video: Optional[str] = None,
    anchor_start: Optional[Tuple[float, float]] = None,
    anchor_end: Optional[Tuple[float, float]] = None,
    reference: Optional[Sequence[PoseSample]] = None,
) -> Dict[str, Any]:
    """Serialize all variants in one self-contained comparison artifact."""

    from .trajectory import pose_to_dict

    return {
        "schema_version": "navgen-vggt-comparison-v1",
        "source_video": source_video,
        "anchor": {
            "type": "route_endpoint_similarity",
            "start": list(anchor_start) if anchor_start is not None else None,
            "end": list(anchor_end) if anchor_end is not None else None,
            "is_ground_truth": False,
        },
        "variants": {
            name: {
                "report": variant.report,
                "points": [pose_to_dict(point) for point in variant.trajectory],
            }
            for name, variant in variants.items()
        },
        "summary": variant_summary(variants, reference=reference),
    }
