"""Estimate camera motion from a video when no pose telemetry is available.

The estimator in this module is deliberately labelled as *video VO* rather
than ground truth.  A Seedance clip contains RGB frames only; the homography
based estimate can recover a useful relative motion trace, but it cannot
prove a robot pose or an absolute metric scale on its own.  Callers may anchor
the first and last pose to a known scene/route so that the trace can be drawn
in scene coordinates.  The returned provenance makes that anchoring explicit.

The dependency on OpenCV/NumPy is optional.  Importing the rest of NavGen and
running the offline mock pipeline must continue to work without the vision
extra installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import PoseSample
from .trajectory import TrajectoryGenerator, trajectory_metrics, wrap_angle

try:  # pragma: no cover - availability depends on the workstation image
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


@dataclass
class VideoTrajectoryResult:
    """A trajectory plus auditable estimator metadata."""

    trajectory: List[PoseSample]
    report: Dict[str, Any]


@dataclass
class _Motion:
    """One relative camera-motion observation between two sampled frames."""

    dx: float
    forward: float
    yaw: float
    inliers: int
    matches: int
    reprojection_error: float
    confidence: float


def _require_vision() -> None:
    if cv2 is None or np is None:
        raise RuntimeError(
            "video trajectory estimation requires the optional vision dependencies; "
            "install with `pip install -e '.[vision]'`"
        )


def _intrinsics(width: int, height: int, focal_scale: float) -> Any:
    # Seedance does not expose the camera calibration.  This is an explicit
    # approximation, recorded in the report, rather than a hidden constant.
    focal = max(float(width), 1.0) * max(0.35, min(2.0, float(focal_scale)))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _frame_features(gray: Any, sift: Any) -> Tuple[List[Any], Any]:
    """Detect mostly floor-region features, with a full-frame fallback."""

    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    # Lower-half features are less affected by independently moving objects
    # and are the best available approximation to a ground-plane homography.
    mask[int(height * 0.48) :, :] = 255
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < 24:
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints or [], descriptors


def _reprojection_error(source: Any, target: Any, homography: Any, mask: Any) -> float:
    if homography is None or mask is None:
        return float("inf")
    try:
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        distances = np.linalg.norm(projected - target, axis=1)
        inlier_mask = mask.ravel().astype(bool)
        values = distances[inlier_mask]
        return float(np.median(values)) if values.size else float("inf")
    except Exception:
        return float("inf")


def _candidate_motion(
    homography: Any,
    camera_matrix: Any,
    source_points: Any,
    target_points: Any,
    inlier_mask: Any,
    previous_yaw: Optional[float],
    forward_hint: Optional[bool],
) -> Optional[_Motion]:
    """Choose a physically plausible floor-plane homography solution.

    Homography decomposition returns four sign/plane solutions.  We favour a
    near-vertical plane normal (the floor in camera coordinates), continuity
    of yaw, and the direction implied by global image expansion/contraction.
    This is still an estimate; ambiguity is retained in the report.
    """

    try:
        count, rotations, translations, normals = cv2.decomposeHomographyMat(homography, camera_matrix)
    except Exception:
        return None
    if not count:
        return None
    reprojection = _reprojection_error(source_points, target_points, homography, inlier_mask)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    matches = int(len(source_points))
    candidates: List[Tuple[float, _Motion, float]] = []
    for rotation, translation, normal in zip(rotations, translations, normals):
        try:
            vector = np.asarray(translation, dtype=float).reshape(-1)
            normal_vector = np.asarray(normal, dtype=float).reshape(-1)
            if vector.size < 3 or normal_vector.size < 3:
                continue
            # Camera x is lateral and camera z is forward.  The sign is
            # normalised below using the image-scale hint.
            lateral = float(vector[0])
            forward = -float(vector[2])
            yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
            vertical_normal = abs(float(normal_vector[1]))
            continuity = 0.0 if previous_yaw is None else abs(wrap_angle(yaw - previous_yaw))
            direction_bonus = 0.0
            if forward_hint is not None and abs(forward) > 1e-9:
                direction_bonus = 0.45 if (forward > 0) == forward_hint else -0.45
            # The score is intentionally conservative: a noisy translation
            # must not beat a solution with a clearly vertical plane normal.
            score = 2.2 * vertical_normal - 0.35 * continuity + direction_bonus
            confidence = max(
                0.0,
                min(
                    1.0,
                    0.45 * min(1.0, inliers / 80.0)
                    + 0.35 * min(1.0, inliers / max(matches, 1) / 0.7)
                    + 0.20 * min(1.0, 3.0 / max(reprojection, 1e-3)),
                ),
            )
            candidates.append(
                (
                    score,
                    _Motion(
                        dx=lateral,
                        forward=forward,
                        yaw=yaw,
                        inliers=inliers,
                        matches=matches,
                        reprojection_error=reprojection,
                        confidence=confidence,
                    ),
                    vertical_normal,
                )
            )
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    motion = candidates[0][1]
    # The opposite solution has the same rotation.  Prefer the sign that
    # agrees with the observed forward/backward image-scale cue when possible.
    if forward_hint is not None and abs(motion.forward) > 1e-9 and (motion.forward > 0) != forward_hint:
        for _, alternative, vertical_normal in candidates[1:]:
            if abs(alternative.yaw - motion.yaw) < 1e-4 and (alternative.forward > 0) == forward_hint:
                if vertical_normal >= candidates[0][2] - 0.18:
                    motion = alternative
                    break
    return motion


def _read_sampled_frames(path: Path, sample_hz: float) -> Iterable[Tuple[float, Any, float, int, int]]:
    """Yield ``(time, gray, fps, width, height)`` at a bounded rate."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for trajectory estimation: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(fps / max(sample_hz, 0.5))))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Keeping the working width bounded makes the estimator fast
                # enough for a workstation batch without changing timestamps.
                if gray.shape[1] > 960:
                    scale = 960.0 / gray.shape[1]
                    gray = cv2.resize(gray, (960, max(1, int(gray.shape[0] * scale))))
                yield index / fps, gray, fps, width, height
            index += 1
    finally:
        capture.release()


def _smooth_positions(points: Sequence[Tuple[float, float]], radius: int = 1) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    result: List[Tuple[float, float]] = []
    for index, point in enumerate(points):
        if index == 0 or index == len(points) - 1:
            result.append(point)
            continue
        lo = max(0, index - radius)
        hi = min(len(points), index + radius + 1)
        window = points[lo:hi]
        result.append((sum(p[0] for p in window) / len(window), sum(p[1] for p in window) / len(window)))
    return result


def _anchor_to_poses(
    positions: Sequence[Tuple[float, float]],
    yaws: Sequence[float],
    start_pose: Optional[Tuple[float, float, float]],
    end_pose: Optional[Tuple[float, float, float]],
) -> Tuple[List[Tuple[float, float]], List[float], Dict[str, Any]]:
    """Apply a similarity transform using optional known endpoint anchors."""

    if not positions:
        return [], list(yaws), {"metric_scale": "unknown", "anchor": "none"}
    raw_start = positions[0]
    transformed = [(p[0] - raw_start[0], p[1] - raw_start[1]) for p in positions]
    rotation = 0.0
    scale = 1.0
    anchor = "relative"
    if start_pose is not None and end_pose is not None and len(positions) >= 2:
        raw_dx = positions[-1][0] - positions[0][0]
        raw_dy = positions[-1][1] - positions[0][1]
        target_dx = float(end_pose[0]) - float(start_pose[0])
        target_dy = float(end_pose[1]) - float(start_pose[1])
        raw_length = math.hypot(raw_dx, raw_dy)
        target_length = math.hypot(target_dx, target_dy)
        if raw_length > 1e-6 and target_length > 1e-6:
            scale = target_length / raw_length
            rotation = math.atan2(target_dy, target_dx) - math.atan2(raw_dy, raw_dx)
            anchor = "route_start_end"
    cosine, sine = math.cos(rotation), math.sin(rotation)
    origin_x = float(start_pose[0]) if start_pose is not None else 0.0
    origin_y = float(start_pose[1]) if start_pose is not None else 0.0
    output: List[Tuple[float, float]] = []
    for x, y in transformed:
        output.append((origin_x + scale * (cosine * x - sine * y), origin_y + scale * (sine * x + cosine * y)))
    yaw_offset = rotation + (float(start_pose[2]) if start_pose is not None else 0.0)
    output_yaws = [wrap_angle(float(yaw) + yaw_offset) for yaw in yaws]
    metadata: Dict[str, Any] = {
        "metric_scale": "endpoint_anchored" if anchor == "route_start_end" else "unknown",
        "anchor": anchor,
        "scale_factor": scale,
        "rotation_anchor_rad": rotation,
    }
    if end_pose is not None and output:
        metadata["endpoint_position_error_m"] = math.hypot(
            output[-1][0] - float(end_pose[0]), output[-1][1] - float(end_pose[1])
        )
        metadata["endpoint_yaw_error_rad"] = abs(wrap_angle(output_yaws[-1] - float(end_pose[2])))
    return output, output_yaws, metadata


def estimate_video_trajectory(
    video_path: Path | str,
    *,
    sample_hz: float = 10.0,
    start_pose: Optional[Tuple[float, float, float]] = None,
    end_pose: Optional[Tuple[float, float, float]] = None,
    focal_scale: float = 0.9,
    min_inliers: int = 20,
) -> VideoTrajectoryResult:
    """Estimate a 2-D camera trace from an RGB video.

    ``start_pose``/``end_pose`` are optional scene anchors.  Supplying them
    makes the output drawable in the scene's metric coordinate frame, but it
    does not turn the result into ground truth; ``report['ground_truth']`` is
    always ``False`` for this function.
    """

    _require_vision()
    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"video does not exist or is empty: {path}")
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")

    frames = list(_read_sampled_frames(path, sample_hz))
    if len(frames) < 2:
        raise ValueError("video contains fewer than two usable frames")
    first_gray = frames[0][1]
    working_height, working_width = first_gray.shape[:2]
    camera_matrix = _intrinsics(working_width, working_height, focal_scale)
    sift = cv2.SIFT_create(nfeatures=2800, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    raw_positions: List[Tuple[float, float]] = [(0.0, 0.0)]
    raw_yaws: List[float] = [0.0]
    times: List[float] = [float(frames[0][0])]
    motions: List[_Motion] = []
    previous_keypoints, previous_descriptors = _frame_features(first_gray, sift)
    previous_yaw = 0.0
    valid_pairs = 0
    rejected_pairs = 0
    previous_scale_hint: Optional[bool] = None

    for timestamp, gray, _, _, _ in frames[1:]:
        keypoints, descriptors = _frame_features(gray, sift)
        motion: Optional[_Motion] = None
        if previous_descriptors is not None and descriptors is not None and len(previous_descriptors) >= 8 and len(descriptors) >= 8:
            try:
                matches = matcher.knnMatch(previous_descriptors, descriptors, k=2)
                good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
                if len(good) >= max(8, min_inliers):
                    source_points = np.float32([previous_keypoints[m.queryIdx].pt for m in good])
                    target_points = np.float32([keypoints[m.trainIdx].pt for m in good])
                    homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
                    if homography is not None and mask is not None and int(mask.sum()) >= min_inliers:
                        affine, affine_mask = cv2.estimateAffinePartial2D(
                            source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
                        )
                        scale_hint: Optional[bool] = previous_scale_hint
                        if affine is not None:
                            a, c = float(affine[0, 0]), float(affine[1, 0])
                            scale_hint = math.hypot(a, c) > 1.002
                        motion = _candidate_motion(
                            homography,
                            camera_matrix,
                            source_points,
                            target_points,
                            mask,
                            previous_yaw,
                            scale_hint,
                        )
                        if motion is not None:
                            previous_scale_hint = scale_hint
            except Exception:
                motion = None
        if motion is None or motion.inliers < min_inliers:
            rejected_pairs += 1
            motion = _Motion(0.0, 0.0, 0.0, 0, 0, float("inf"), 0.0)
        else:
            valid_pairs += 1
        motions.append(motion)
        old_x, old_y = raw_positions[-1]
        old_yaw = raw_yaws[-1]
        # ``forward`` is positive in the robot's local forward direction.
        cosine, sine = math.cos(old_yaw), math.sin(old_yaw)
        raw_positions.append(
            (
                old_x + cosine * motion.dx - sine * motion.forward,
                old_y + sine * motion.dx + cosine * motion.forward,
            )
        )
        raw_yaws.append(wrap_angle(old_yaw + motion.yaw))
        times.append(float(timestamp))
        previous_keypoints, previous_descriptors = keypoints, descriptors
        previous_yaw = motion.yaw

    # A median filter removes isolated decomposition flips while retaining the
    # broad turns visible in the clip.  The first/last samples stay exact.
    smoothed_xy = _smooth_positions(raw_positions, radius=1)
    anchored_xy, anchored_yaw, anchor_metadata = _anchor_to_poses(smoothed_xy, raw_yaws, start_pose, end_pose)
    # Even a perfect feature fit cannot establish metric ground truth from a
    # monocular generated clip.  Cap the confidence so downstream quality
    # gates keep this artifact in review until a pose stream is attached.
    confidence = max(0.05, min(0.55, 0.40 * valid_pairs / max(len(motions), 1) + 0.15 * (1.0 if anchor_metadata["anchor"] == "route_start_end" else 0.0)))
    trajectory = [
        PoseSample(
            t=t,
            x=xy[0],
            y=xy[1],
            yaw=yaw,
            confidence=confidence,
            frame_id="scene_xy_anchored",
            source="video_vo_estimate",
        )
        for t, xy, yaw in zip(times, anchored_xy, anchored_yaw)
    ]
    # Recompute derivatives from the measured positions.  The generator's
    # smoothing keeps downstream kinematic filters deterministic.
    if len(trajectory) >= 2:
        observations = [(sample.t, sample.x, sample.y, sample.yaw) for sample in trajectory]
        trajectory = TrajectoryGenerator(sample_hz=max(sample_hz, 1.0)).reconstruct_from_positions(
            observations, confidence=confidence, source="video_vo_estimate"
        )
        for sample in trajectory:
            sample.frame_id = "scene_xy_anchored"
            sample.source = "video_vo_estimate"
    metrics = trajectory_metrics(trajectory)
    warnings: List[str] = [
        "monocular_video_scale_is_not_observed",
        "camera_calibration_is_approximated",
        "not_ground_truth_pose",
    ]
    if anchor_metadata.get("anchor") == "route_start_end":
        warnings.append("metric_scale_anchored_to_route_endpoints")
    if rejected_pairs:
        warnings.append(f"{rejected_pairs}_frame_pairs_rejected")
    report: Dict[str, Any] = {
        "source": "video_vo_estimate",
        "estimator": "sift_floor_homography_v1",
        "ground_truth": False,
        "metric_scale_observed": anchor_metadata.get("metric_scale") == "endpoint_anchored",
        "confidence": confidence,
        "frame_count": len(frames),
        "sample_count": len(trajectory),
        "valid_pair_count": valid_pairs,
        "rejected_pair_count": rejected_pairs,
        "valid_pair_ratio": valid_pairs / max(len(motions), 1),
        "median_inliers": float(np.median([m.inliers for m in motions if m.inliers])) if any(m.inliers for m in motions) else 0.0,
        "median_reprojection_error_px": float(
            np.median([m.reprojection_error for m in motions if math.isfinite(m.reprojection_error)])
        )
        if any(math.isfinite(m.reprojection_error) for m in motions)
        else None,
        "focal_scale_assumption": focal_scale,
        "sample_hz": sample_hz,
        "effective_sample_hz": (
            1.0 / float(np.median(np.diff(np.asarray(times, dtype=float))))
            if len(times) > 1
            else 0.0
        ),
        "warnings": warnings,
        "metrics": metrics,
        **anchor_metadata,
    }
    return VideoTrajectoryResult(trajectory=trajectory, report=report)


def trajectory_data_dict(result: VideoTrajectoryResult) -> Dict[str, Any]:
    """Return a browser/download friendly trajectory artifact."""

    from .trajectory import pose_to_dict

    coordinate_frame = result.report.get("coordinate_frame")
    if not coordinate_frame and result.trajectory:
        coordinate_frame = result.trajectory[0].frame_id
    return {
        "schema_version": "navgen-trajectory-v2",
        "source": result.report.get("source"),
        "ground_truth": bool(result.report.get("ground_truth", False)),
        "metric_scale": result.report.get("metric_scale", "unknown"),
        "metric_units": result.report.get("metric_units", "metric"),
        "metrics_are_metric": bool(result.report.get("metrics_are_metric", False)),
        "coordinate_frame": coordinate_frame,
        "report": result.report,
        "points": [pose_to_dict(sample) for sample in result.trajectory],
    }
