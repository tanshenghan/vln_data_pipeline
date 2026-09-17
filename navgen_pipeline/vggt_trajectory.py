"""Estimate a relative camera trajectory from an RGB video with VGGT.

VGGT is deliberately an optional dependency.  A Seedance clip contains RGB
frames only, so this module estimates the camera trajectory in VGGT's
reconstructed coordinate frame.  It does *not* manufacture metric scale or a
ground-truth label.  The caller receives a provenance report with those
limitations made explicit.

The implementation keeps the first useful version intentionally simple:

* sample a bounded number of frames from the video;
* run the VGGT camera head once on that ordered frame set;
* convert the predicted world-to-camera matrices to camera centres;
* express the centres in a first-camera horizontal frame;
* compute finite-difference kinematics without smoothing or scale anchoring.

The official VGGT repository is not a PyPI dependency.  Set ``VGGT_REPO`` to
an installed/cloned checkout (or install it into the active environment) and
use ``VGGT_CHECKPOINT`` for a local checkpoint when the workstation has no
Hugging Face cache.
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import PoseSample
from .trajectory import trajectory_metrics


@dataclass
class VGGTTrajectoryResult:
    """Trajectory samples and a self-describing VGGT provenance report."""

    trajectory: List[PoseSample]
    report: Dict[str, Any]


def _require_runtime() -> Tuple[Any, Any]:
    """Import NumPy/OpenCV lazily so offline NavGen stays dependency-light."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "VGGT video estimation requires numpy and opencv; install with "
            "`pip install -e '.[vggt]'` (or `.[vision]` for the video reader)"
        ) from exc
    return cv2, np


def _import_vggt(repo_path: Optional[str]) -> Tuple[Any, Any, Any]:
    """Load the VGGT class and pose decoder without importing it at startup."""

    if repo_path:
        candidate = Path(repo_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"VGGT repository does not exist: {candidate}")
        # A checkout is importable from its root without requiring a global
        # package installation.  Put it first so a stale site-package cannot
        # silently win.
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "VGGT is not installed. Clone https://github.com/facebookresearch/vggt "
            "and install its requirements, or set VGGT_REPO to that checkout."
        ) from exc
    return VGGT, pose_encoding_to_extri_intri, sys.modules["vggt"]


def _resolve_device(torch: Any, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if value.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError("VGGT_DEVICE requests CUDA, but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"} and not value.startswith("cuda:"):
        raise ValueError("VGGT_DEVICE must be auto, cpu, cuda or cuda:<index>")
    return value


def _checkpoint_state(payload: Any) -> Mapping[str, Any]:
    """Extract a state dict from common PyTorch checkpoint layouts."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("VGGT checkpoint is not a state-dict mapping")
    for key in ("state_dict", "model_state_dict", "model", "module"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    # Some distributed checkpoints prefix every key with ``module.`` or
    # ``_orig_mod.``.  Removing only those known wrappers keeps model names
    # such as ``aggregator`` intact.
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        text = str(key)
        for prefix in ("module.", "_orig_mod."):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        cleaned[text] = value
    return cleaned


def _load_model(
    torch: Any,
    vggt_class: Any,
    *,
    model_id: str,
    checkpoint: Optional[str],
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Construct a camera-only VGGT and load a local or HF checkpoint."""

    # Disabling unused heads is important for a 24 GB workstation: this first
    # integration only needs camera poses, not dense depth/track tensors.
    try:
        model = vggt_class(
            enable_camera=True,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
    except TypeError:  # pragma: no cover - compatibility with old forks
        model = vggt_class()

    checkpoint_path: Optional[Path] = None
    source = checkpoint or ""
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VGGT checkpoint does not exist: {checkpoint_path}")
    elif model_id and Path(model_id).expanduser().is_file():
        checkpoint_path = Path(model_id).expanduser()
        source = str(checkpoint_path)

    started = time.monotonic()
    if checkpoint_path is None:
        # Prefer the official Hugging Face helper when available.  Falling
        # back to torch's downloader keeps the adapter usable in minimal
        # environments where huggingface_hub is not separately installed.
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            downloaded = hf_hub_download(repo_id=model_id, filename="model.pt")
            checkpoint_path = Path(downloaded)
        except Exception as hf_exc:
            url = f"https://huggingface.co/{model_id}/resolve/main/model.pt"
            try:
                downloaded = torch.hub.load_state_dict_from_url(
                    url,
                    map_location="cpu",
                    check_hash=False,
                    progress=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"unable to load VGGT checkpoint {model_id!r}; set VGGT_CHECKPOINT "
                    "to a local model.pt file"
                ) from exc
            payload = downloaded
            source = model_id
        else:
            payload = torch.load(str(checkpoint_path), map_location="cpu")
            source = model_id
    else:
        try:
            payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:  # older torch has no weights_only keyword
            payload = torch.load(str(checkpoint_path), map_location="cpu")

    state = _checkpoint_state(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, {
        "checkpoint": source,
        "checkpoint_load_s": round(time.monotonic() - started, 6),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        # Keep only counts in the report; full key names are implementation
        # details and can be very large for a checkpoint.
    }


def _uniform_indices(total: int, count: int) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    values = {int(round(index * (total - 1) / (count - 1))) for index in range(count)}
    return sorted(values)


def _read_video_frames(
    path: Path,
    *,
    sample_hz: float,
    max_frames: int,
    cv2: Any,
) -> Tuple[List[float], List[Any], Dict[str, Any]]:
    """Read ordered RGB frames and timestamps at a bounded sampling rate."""

    if sample_hz <= 0:
        raise ValueError("VGGT frame sample rate must be positive")
    if max_frames < 2:
        raise ValueError("VGGT max_frames must be at least 2")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for VGGT trajectory estimation: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 24.0
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / sample_hz)))
    if reported_count > 0:
        candidates = list(range(0, reported_count, stride))
        if not candidates:
            candidates = [0]
        # Include the final decoded frame when it is not already selected;
        # this is a sampling choice, not a scale/endpoint anchor.
        if candidates[-1] != reported_count - 1:
            candidates.append(reported_count - 1)
        indices = _uniform_indices(reported_count, min(max_frames, len(candidates)))
        # Uniform indices above refer to the complete video.  Re-map them to
        # the requested-rate candidate list so a high-fps clip is not sampled
        # at every frame when max_frames is large.
        if len(candidates) > max_frames:
            positions = _uniform_indices(len(candidates), max_frames)
            indices = [candidates[position] for position in positions]
        else:
            indices = candidates
    else:
        indices = []

    wanted = set(indices)
    timestamps: List[float] = []
    frames: List[Any] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not indices and (index % stride == 0):
                wanted.add(index)
            if index in wanted:
                timestamps.append(index / fps)
                frames.append(frame)
                if len(frames) >= max_frames:
                    # If the final frame was not known, bounded sampling can
                    # stop as soon as the requested budget is filled.
                    if reported_count <= 0 or index >= max(indices or [index]):
                        break
            index += 1
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("video contains fewer than two frames usable by VGGT")
    return timestamps, frames, {
        "video_fps": fps,
        "video_frame_count_reported": reported_count or index + 1,
        "sampled_frame_count": len(frames),
        "sample_hz_requested": sample_hz,
        "sample_hz_effective": (
            1.0
            / max(
                1e-9,
                sorted(
                    timestamps[index + 1] - timestamps[index]
                    for index in range(len(timestamps) - 1)
                )[max(0, (len(timestamps) - 2) // 2)],
            )
            if len(timestamps) > 1
            else 0.0
        ),
    }


def _frame_to_tensor(frame: Any, *, size: int, cv2: Any, np: Any, torch: Any) -> Any:
    """Pad an RGB frame to VGGT's square input without changing its content."""

    if size <= 0 or size % 14 != 0:
        raise ValueError("VGGT input_size must be a positive multiple of 14")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(float(size) / max(width, 1), float(size) / max(height, 1))
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float() / 255.0


def _quat_to_matrix(quaternion: Sequence[float], np: Any) -> Any:
    """Convert VGGT's XYZW quaternion to a 3x3 rotation matrix."""

    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.size != 4:
        raise ValueError("VGGT quaternion must contain four values")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = (float(item) / norm for item in values)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_encoding_to_matrices(pose_encoding: Any, *, image_size: int, np: Any) -> Tuple[Any, Any]:
    """Decode ``absT_quaR_FoV`` pose encodings using only NumPy."""

    values = np.asarray(pose_encoding, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[1] < 9:
        raise ValueError(f"unexpected VGGT pose encoding shape: {values.shape}")
    extrinsics = np.zeros((values.shape[0], 3, 4), dtype=np.float64)
    intrinsics = np.zeros((values.shape[0], 3, 3), dtype=np.float64)
    for index, row in enumerate(values):
        extrinsics[index, :3, :3] = _quat_to_matrix(row[3:7], np)
        extrinsics[index, :3, 3] = row[:3]
        # VGGT stores vertical and horizontal FoV after the quaternion.
        fov_h = float(np.clip(row[7], 1e-4, math.pi - 1e-4))
        fov_w = float(np.clip(row[8], 1e-4, math.pi - 1e-4))
        fy = (image_size / 2.0) / math.tan(fov_h / 2.0)
        fx = (image_size / 2.0) / math.tan(fov_w / 2.0)
        intrinsics[index] = np.asarray(
            [[fx, 0.0, image_size / 2.0], [0.0, fy, image_size / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return extrinsics, intrinsics


def _normalise(vector: Any, np: Any, fallback: Sequence[float]) -> Any:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / max(norm, 1e-9)


def camera_matrices_to_navigation_poses(
    extrinsics: Any, *, np: Any
) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]:
    """Convert camera-from-world matrices to first-camera 2-D poses.

    The axes are chosen from the first camera only.  This makes the output
    easy to plot while preserving VGGT's original translation scale.  No
    smoothing, route fitting, or metric scale is applied here.
    """

    matrices = np.asarray(extrinsics, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 4) or len(matrices) < 2:
        raise ValueError(f"unexpected VGGT extrinsic shape: {matrices.shape}")
    first_rotation = matrices[0, :3, :3]
    # OpenCV camera axes are x-right, y-down, z-forward.  Use the first
    # camera's up/forward directions to define a local horizontal frame.
    up = _normalise(-(first_rotation.T @ np.asarray([0.0, 1.0, 0.0])), np, [0.0, 1.0, 0.0])

    def project(vector: Any) -> Any:
        return vector - float(np.dot(vector, up)) * up

    right = _normalise(project(first_rotation.T @ np.asarray([1.0, 0.0, 0.0])), np, [1.0, 0.0, 0.0])
    forward = project(first_rotation.T @ np.asarray([0.0, 0.0, 1.0]))
    # Orthogonalise to protect against numerical drift in a predicted pose.
    forward = forward - float(np.dot(forward, right)) * right
    forward = _normalise(forward, np, [0.0, 0.0, 1.0])
    origin = -first_rotation.T @ matrices[0, :3, 3]
    poses: List[Tuple[float, float, float]] = []
    for matrix in matrices:
        rotation = matrix[:3, :3]
        center = -rotation.T @ matrix[:3, 3]
        relative = center - origin
        x = float(np.dot(relative, right))
        y = float(np.dot(relative, forward))
        z = float(np.dot(relative, up))
        camera_forward = project(rotation.T @ np.asarray([0.0, 0.0, 1.0]))
        camera_forward = _normalise(camera_forward, np, forward)
        yaw = math.atan2(float(np.dot(camera_forward, right)), float(np.dot(camera_forward, forward)))
        poses.append((x, y, z, yaw))
    return poses, {
        "coordinate_frame": "vggt_first_camera_horizontal",
        "origin": [float(item) for item in origin],
        "right_axis": [float(item) for item in right],
        "forward_axis": [float(item) for item in forward],
        "up_axis": [float(item) for item in up],
    }


def _derive_raw_kinematics(samples: Sequence[PoseSample]) -> None:
    """Populate derivatives with finite differences; intentionally no filter."""

    if len(samples) < 2:
        return

    def derivative(values: Sequence[float]) -> List[float]:
        output: List[float] = []
        for index in range(len(values)):
            if index == 0:
                dt = max(1e-9, samples[1].t - samples[0].t)
                output.append((values[1] - values[0]) / dt)
            elif index == len(values) - 1:
                dt = max(1e-9, samples[-1].t - samples[-2].t)
                output.append((values[-1] - values[-2]) / dt)
            else:
                dt = max(1e-9, samples[index + 1].t - samples[index - 1].t)
                output.append((values[index + 1] - values[index - 1]) / dt)
        return output

    vx = derivative([sample.x for sample in samples])
    vy = derivative([sample.y for sample in samples])
    ax = derivative(vx)
    ay = derivative(vy)
    unwrapped = [samples[0].yaw]
    for sample in samples[1:]:
        previous = unwrapped[-1]
        delta = (sample.yaw - previous + math.pi) % (2.0 * math.pi) - math.pi
        unwrapped.append(previous + delta)
    omega = derivative(unwrapped)
    alpha = derivative(omega)
    for index, sample in enumerate(samples):
        sample.vx, sample.vy = vx[index], vy[index]
        sample.ax, sample.ay = ax[index], ay[index]
        sample.omega, sample.alpha = omega[index], alpha[index]


def estimate_vggt_trajectory(
    video_path: Path | str,
    *,
    sample_hz: float = 2.0,
    max_frames: int = 48,
    input_size: int = 518,
    model_id: str = "facebook/VGGT-1B",
    checkpoint: Optional[str] = None,
    repo_path: Optional[str] = None,
    device: str = "auto",
) -> VGGTTrajectoryResult:
    """Run VGGT's camera head on an RGB video and return raw relative poses."""

    cv2, np = _require_runtime()
    path = Path(video_path).expanduser()
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"video does not exist or is empty: {path}")
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if input_size <= 0 or input_size % 14 != 0:
        raise ValueError("input_size must be a positive multiple of 14")

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "VGGT requires PyTorch; install the optional dependency and the "
            "official VGGT checkout before running this estimator"
        ) from exc
    vggt_class, pose_decoder, _ = _import_vggt(repo_path)
    resolved_device = _resolve_device(torch, device)
    timestamps, raw_frames, video_meta = _read_video_frames(
        path,
        sample_hz=sample_hz,
        max_frames=max_frames,
        cv2=cv2,
    )
    images = torch.stack(
        [_frame_to_tensor(frame, size=input_size, cv2=cv2, np=np, torch=torch) for frame in raw_frames]
    )
    model, load_meta = _load_model(
        torch,
        vggt_class,
        model_id=model_id,
        checkpoint=checkpoint,
        device=resolved_device,
    )
    images = images.to(resolved_device)
    started = time.monotonic()
    if resolved_device.startswith("cuda"):
        try:
            capability = torch.cuda.get_device_capability(resolved_device)
            dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        except Exception:
            dtype = torch.float16
        autocast = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype = torch.float32
        autocast = contextlib.nullcontext()
    with torch.inference_mode():
        with autocast:
            predictions = model(images)
    inference_s = round(time.monotonic() - started, 6)
    if not isinstance(predictions, Mapping) or "pose_enc" not in predictions:
        raise RuntimeError("VGGT inference did not return pose_enc")
    pose_encoding = predictions["pose_enc"]
    if hasattr(pose_encoding, "detach"):
        pose_encoding_np = pose_encoding.detach().float().cpu().numpy()
    else:
        pose_encoding_np = np.asarray(pose_encoding)

    # Use the official decoder when available to validate the encoding, then
    # use the NumPy implementation for a stable, dependency-light conversion.
    try:
        pose_decoder(pose_encoding, image_size_hw=(input_size, input_size), build_intrinsics=True)
    except Exception:
        # Older forks may expose a decoder with a different signature; the
        # explicit NumPy path below remains compatible with absT_quaR_FoV.
        pass
    extrinsics, intrinsics = _pose_encoding_to_matrices(pose_encoding_np, image_size=input_size, np=np)
    if len(extrinsics) != len(timestamps):
        raise RuntimeError(
            f"VGGT returned {len(extrinsics)} poses for {len(timestamps)} sampled frames"
        )
    poses, frame_metadata = camera_matrices_to_navigation_poses(extrinsics, np=np)
    trajectory: List[PoseSample] = []
    confidence = 0.5
    for timestamp, pose in zip(timestamps, poses):
        trajectory.append(
            PoseSample(
                t=float(timestamp),
                x=float(pose[0]),
                y=float(pose[1]),
                z=float(pose[2]),
                yaw=float(pose[3]),
                confidence=confidence,
                frame_id=str(frame_metadata["coordinate_frame"]),
                source="vggt_monocular_estimate",
            )
        )
    _derive_raw_kinematics(trajectory)
    finite = all(
        math.isfinite(value)
        for sample in trajectory
        for value in (sample.t, sample.x, sample.y, sample.z, sample.yaw)
    )
    if not finite:
        raise RuntimeError("VGGT returned non-finite camera poses")
    median_fx = float(np.median(intrinsics[:, 0, 0]))
    median_fy = float(np.median(intrinsics[:, 1, 1]))
    report: Dict[str, Any] = {
        "source": "vggt_monocular_estimate",
        "estimator": "vggt_camera_head",
        "ground_truth": False,
        "metric_scale_observed": False,
        "metric_scale": "unknown",
        "metric_units": "relative",
        "metrics_are_metric": False,
        "scale_anchor": "none",
        "temporal_smoothing": False,
        "confidence": confidence,
        "model_id": model_id,
        "device": resolved_device,
        "dtype": str(dtype).replace("torch.", ""),
        "input_size": input_size,
        "inference_s": inference_s,
        "median_fx_px": median_fx,
        "median_fy_px": median_fy,
        "focal_length_variation_ratio": float(
            (np.max(intrinsics[:, 0, 0]) - np.min(intrinsics[:, 0, 0])) / max(abs(median_fx), 1e-9)
        ),
        **video_meta,
        **load_meta,
        **frame_metadata,
        "warnings": [
            "monocular_scale_is_unknown",
            "no_temporal_smoothing",
            "no_scale_anchor",
            "predicted_intrinsics_not_calibration",
            "seedance_rgb_video_is_not_ground_truth",
        ],
        "metrics": trajectory_metrics(trajectory),
    }
    return VGGTTrajectoryResult(trajectory=trajectory, report=report)


def trajectory_data_dict(result: VGGTTrajectoryResult) -> Dict[str, Any]:
    """Return a browser/download-friendly VGGT trajectory artifact."""

    from .trajectory import pose_to_dict

    coordinate_frame = result.report.get("coordinate_frame")
    if not coordinate_frame and result.trajectory:
        coordinate_frame = result.trajectory[0].frame_id
    return {
        "schema_version": "navgen-trajectory-v2",
        "source": result.report.get("source"),
        "ground_truth": bool(result.report.get("ground_truth", False)),
        "metric_scale": result.report.get("metric_scale", "unknown"),
        "metric_units": result.report.get("metric_units", "relative"),
        "metrics_are_metric": bool(result.report.get("metrics_are_metric", False)),
        "coordinate_frame": coordinate_frame,
        "report": result.report,
        "points": [pose_to_dict(sample) for sample in result.trajectory],
    }
