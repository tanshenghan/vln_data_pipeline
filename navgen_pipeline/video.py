"""Video utilities, inexpensive pre-QC, and an offline Seedance stand-in."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import PoseSample, RouteContract, SampleSpec, SceneManifest

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


@dataclass
class VideoInfo:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    frames: Optional[int]
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _run(command: Sequence[str], timeout_s: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )


def probe_video(path: Path) -> VideoInfo:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"video does not exist or is empty: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-500:]}")
    data = json.loads(result.stdout or "{}")
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise RuntimeError(f"no video stream found in {path}")
    stream = streams[0]
    rate = str(stream.get("r_frame_rate", "0/1"))
    try:
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    format_data = data.get("format", {})
    duration = float(format_data.get("duration") or 0.0)
    frame_value = stream.get("nb_frames")
    frames: Optional[int]
    try:
        frames = int(frame_value) if frame_value not in (None, "N/A") else None
    except (TypeError, ValueError):
        frames = None
    return VideoInfo(
        path=str(path),
        duration_s=duration,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        frames=frames,
        size_bytes=int(format_data.get("size") or path.stat().st_size),
    )


def extract_frames(path: Path, output_dir: Path, sample_fps: float = 1.0) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale frames when rerunning a sample.
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "3",
        str(output_dir / "frame_%05d.jpg"),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[-500:]}")
    return sorted(output_dir.glob("frame_*.jpg"))


def _mean_brightness(path: Path) -> float:
    if Image is None:
        return 0.5
    try:
        image = Image.open(path).convert("L").resize((16, 16))
        values = list(image.getdata())
        return sum(values) / (255.0 * len(values))
    except Exception:
        return 0.0


def make_contact_sheet(frame_paths: Sequence[Path], output_path: Path, columns: int = 4) -> Optional[Path]:
    if Image is None or not frame_paths:
        return None
    thumbnails = []
    for index, path in enumerate(frame_paths):
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((320, 180))
            canvas = Image.new("RGB", (320, 180), (30, 30, 30))
            canvas.paste(image, ((320 - image.width) // 2, (180 - image.height) // 2))
            # A small index lets a VLM reviewer map evidence back to the
            # sampled timestamp even when the contact sheet is the only image
            # sent to the remote endpoint.
            ImageDraw.Draw(canvas).rectangle((4, 4, 66, 20), fill=(0, 0, 0))
            ImageDraw.Draw(canvas).text((8, 6), f"frame {index:02d}", fill=(245, 245, 245))
            thumbnails.append(canvas)
        except Exception:
            continue
    if not thumbnails:
        return None
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * 320, rows * 180), (10, 10, 10))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * 320, (index // columns) * 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def make_trajectory_plot(
    scene: SceneManifest,
    trajectory: Sequence[PoseSample],
    output_path: Path,
    target_id: Optional[str] = None,
    obstacles: Optional[Sequence[Mapping[str, Any]]] = None,
    reference_trajectory: Optional[Sequence[PoseSample]] = None,
    trajectory_source: str = "route_contract",
    relative_only: bool = False,
    width: int = 800,
    height: int = 520,
) -> Optional[Path]:
    """Render a dependency-light top-down trajectory audit image.

    This is intentionally a diagnostic plot rather than a photorealistic
    map.  It makes collision checks, start/end points and the target region
    inspectable in a browser or an annotation review queue without requiring
    matplotlib.
    ``relative_only`` is used for monocular estimates whose scale/frame cannot
    be aligned to the scene manifest.  In that mode the image deliberately
    omits object/obstacle footprints and draws only a fitted relative frame.
    """

    if Image is None or ImageDraw is None or not trajectory:
        return None
    image = Image.new("RGB", (width, height), (17, 25, 45))
    draw = ImageDraw.Draw(image)
    # Room and grid.
    margin_x, margin_y = 64, 44
    room_left, room_top = margin_x, margin_y
    room_right, room_bottom = width - margin_x, height - margin_y
    draw.rectangle((room_left, room_top, room_right, room_bottom), fill=(31, 48, 78), outline=(116, 150, 206), width=2)
    for gx in range(1, 8):
        x = room_left + (room_right - room_left) * gx / 8
        draw.line((x, room_top, x, room_bottom), fill=(46, 67, 101), width=1)
    for gy in range(1, 6):
        y = room_top + (room_bottom - room_top) * gy / 6
        draw.line((room_left, y, room_right, y), fill=(46, 67, 101), width=1)

    relative_bounds: Optional[Tuple[float, float, float, float]] = None
    if relative_only:
        xs = [float(sample.x) for sample in trajectory]
        ys = [float(sample.y) for sample in trajectory]
        relative_bounds = (min(xs), max(xs), min(ys), max(ys))

    def point(x: float, y: float) -> tuple[int, int]:
        if relative_bounds is None:
            return _world_to_canvas(scene, x, y, width, height)
        min_x, max_x, min_y, max_y = relative_bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        return (
            int(room_left + ((x - min_x) / span_x) * (room_right - room_left)),
            int(room_bottom - ((y - min_y) / span_y) * (room_bottom - room_top)),
        )

    # Obstacles are stored in the route contract and are already padded by
    # the planner.  Render them as translucent-looking solid blocks.
    if not relative_only:
        for obstacle in (obstacles if obstacles is not None else scene.obstacles):
            x0, y0 = point(float(obstacle.get("x_min", 0.0)), float(obstacle.get("y_max", 0.0)))
            x1, y1 = point(float(obstacle.get("x_max", 0.0)), float(obstacle.get("y_min", 0.0)))
            draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=(116, 78, 91), outline=(215, 132, 145), width=2)
            if obstacle.get("object_id"):
                draw.text((min(x0, x1) + 3, min(y0, y1) + 3), str(obstacle["object_id"]), fill=(244, 205, 210))

    # Draw static object footprints that are not represented in route.obstacles
    # (for example the target itself).
    if not relative_only:
        for obj in scene.objects:
            cx, cy = point(obj.position[0], obj.position[1])
            half_w = max(4, int(obj.size[0] / max(scene.width_m, 1e-6) * (width - 2 * margin_x) / 2))
            half_h = max(4, int(obj.size[1] / max(scene.depth_m, 1e-6) * (height - 2 * margin_y) / 2))
            outline = (247, 192, 84) if obj.id == target_id else (101, 126, 163)
            draw.rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h), outline=outline, width=2)

    # Keep the planned route and the observed/reconstructed path separate.
    # Previously the dashboard drew a single generated curve twice, which
    # made a route prior look like an executed robot trace.
    if reference_trajectory and not relative_only:
        reference_path = [point(sample.x, sample.y) for sample in reference_trajectory]
        if len(reference_path) > 1:
            for left, right in zip(reference_path, reference_path[1:]):
                # Pillow has no portable dashed-line primitive; short
                # alternating segments are sufficient for an audit image.
                for index in range(0, 10, 2):
                    amount0 = index / 10.0
                    amount1 = min(1.0, (index + 1) / 10.0)
                    x0 = left[0] + amount0 * (right[0] - left[0])
                    y0 = left[1] + amount0 * (right[1] - left[1])
                    x1 = left[0] + amount1 * (right[0] - left[0])
                    y1 = left[1] + amount1 * (right[1] - left[1])
                    draw.line((x0, y0, x1, y1), fill=(73, 167, 218), width=3)
    path = [point(sample.x, sample.y) for sample in trajectory]
    if len(path) > 1:
        draw.line(path, fill=(87, 225, 177), width=4, joint="curve")
    start = path[0]
    end = path[-1]
    draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill=(67, 210, 132), outline=(220, 255, 230), width=2)
    draw.ellipse((end[0] - 8, end[1] - 8, end[0] + 8, end[1] + 8), fill=(247, 188, 65), outline=(255, 244, 190), width=2)
    draw.text((room_left, 14), "trajectory audit", fill=(224, 235, 255))
    draw.text((room_left + 150, 14), "start", fill=(67, 210, 132))
    source_label = str(trajectory_source).replace("_", " ")
    draw.text((room_left + 330, 14), source_label, fill=(167, 205, 228))
    if relative_only:
        draw.text((room_left, height - 27), "relative frame · no metric/scene alignment", fill=(242, 202, 119))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def make_trajectory_comparison_plot(
    variants: Mapping[str, Any],
    output_path: Path,
    *,
    reference_trajectory: Optional[Sequence[PoseSample]] = None,
    width: int = 1200,
    height: int = 760,
) -> Optional[Path]:
    """Render raw/post-processed VGGT paths in comparable small multiples.

    Each panel normalizes only its display frame; the trajectory files retain
    their original values.  This avoids visually pretending that raw and
    route-anchored paths share metric coordinates while still making jitter
    and endpoint shape changes easy to inspect.
    """

    if Image is None or ImageDraw is None:
        return None
    names = [name for name in ("raw", "smoothed", "scaled", "smoothed_scaled") if name in variants]
    if not names:
        return None
    image = Image.new("RGB", (width, height), (12, 20, 36))
    draw = ImageDraw.Draw(image)
    columns = 2
    rows = math.ceil(len(names) / columns)
    outer_x, outer_y = 34, 56
    gap_x, gap_y = 22, 26
    panel_width = (width - 2 * outer_x - gap_x) // columns
    panel_height = (height - 2 * outer_y - gap_y * (rows - 1)) // rows
    colors = {
        "raw": (83, 181, 239),
        "smoothed": (120, 224, 167),
        "scaled": (245, 196, 70),
        "smoothed_scaled": (213, 145, 255),
    }

    def unwrap_variant(value: Any) -> Tuple[Sequence[PoseSample], Mapping[str, Any]]:
        if hasattr(value, "trajectory") and hasattr(value, "report"):
            return value.trajectory, value.report
        if isinstance(value, Mapping):
            points = value.get("trajectory") or value.get("points") or []
            report = value.get("report") if isinstance(value.get("report"), Mapping) else {}
            return points, report
        return value, {}

    def panel_point(
        sample: PoseSample,
        points: Sequence[PoseSample],
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Tuple[int, int]:
        xs = [float(item.x) for item in points]
        ys = [float(item.y) for item in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        pad = 0.08
        return (
            int(left + pad * (right - left) + ((float(sample.x) - min_x) / span_x) * (1.0 - 2 * pad) * (right - left)),
            int(bottom - pad * (bottom - top) - ((float(sample.y) - min_y) / span_y) * (1.0 - 2 * pad) * (bottom - top)),
        )

    draw.text((outer_x, 18), "VGGT trajectory comparison", fill=(229, 239, 255))
    draw.text(
        (outer_x + 300, 18),
        "raw baseline · smoothing · route-prior scale anchor",
        fill=(132, 165, 198),
    )
    for index, name in enumerate(names):
        row, column = divmod(index, columns)
        left = outer_x + column * (panel_width + gap_x)
        top = outer_y + row * (panel_height + gap_y)
        right, bottom = left + panel_width, top + panel_height
        draw.rounded_rectangle((left, top, right, bottom), radius=8, fill=(21, 35, 58), outline=(65, 93, 129), width=2)
        points, report = unwrap_variant(variants[name])
        if not points:
            draw.text((left + 14, top + 14), f"{name}: no points", fill=(237, 170, 150))
            continue
        color = colors.get(name, (176, 203, 232))
        chart_left, chart_top = left + 18, top + 54
        chart_right, chart_bottom = right - 18, bottom - 58
        for grid in range(1, 5):
            gx = chart_left + (chart_right - chart_left) * grid / 5
            gy = chart_top + (chart_bottom - chart_top) * grid / 5
            draw.line((gx, chart_top, gx, chart_bottom), fill=(35, 56, 84), width=1)
            draw.line((chart_left, gy, chart_right, gy), fill=(35, 56, 84), width=1)
        mapped = [panel_point(point, points, chart_left, chart_top, chart_right, chart_bottom) for point in points]
        if len(mapped) > 1:
            draw.line(mapped, fill=color, width=4, joint="curve")
        start, end = mapped[0], mapped[-1]
        draw.ellipse((start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6), fill=(74, 223, 157), outline=(228, 255, 238), width=2)
        draw.ellipse((end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7), fill=(247, 196, 70), outline=(255, 244, 190), width=2)
        metrics = report.get("metrics", {}) if isinstance(report, Mapping) else {}
        units = str(report.get("metric_units", "relative")) if isinstance(report, Mapping) else "relative"
        scale = str(report.get("metric_scale", "unknown")) if isinstance(report, Mapping) else "unknown"
        smoothing = "yes" if report.get("temporal_smoothing") else "no"
        path_length = metrics.get("path_length_m") if isinstance(metrics, Mapping) else None
        max_speed = metrics.get("max_speed_mps") if isinstance(metrics, Mapping) else None
        max_accel = metrics.get("max_accel_mps2") if isinstance(metrics, Mapping) else None
        draw.text((left + 14, top + 14), name, fill=color)
        draw.text(
            (left + 14, bottom - 43),
            f"L={float(path_length):.3f} {units}  v={float(max_speed):.3f} {units}/s  a={float(max_accel):.3f} {units}/s²"
            if all(isinstance(value, (int, float)) for value in (path_length, max_speed, max_accel))
            else "metrics unavailable",
            fill=(191, 211, 233),
        )
        draw.text((left + 14, bottom - 24), f"smoothing={smoothing} · scale={scale}", fill=(128, 158, 190))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return output_path


def video_pre_qc(path: Path, expected_duration_s: Optional[float] = None) -> Dict[str, Any]:
    """Cheap checks executed before an expensive VLM request."""

    reasons: List[str] = []
    try:
        info = probe_video(path)
    except Exception as exc:
        return {"decision": "reject", "metrics": {}, "reason_codes": ["decode_error", str(exc)]}
    if info.duration_s <= 0 or info.width <= 0 or info.height <= 0:
        reasons.append("invalid_video_dimensions_or_duration")
    if expected_duration_s is not None and abs(info.duration_s - expected_duration_s) > max(1.5, expected_duration_s * 0.35):
        reasons.append("duration_out_of_expected_range")
    return {
        "decision": "reject" if reasons else "pass",
        "metrics": info.to_dict(),
        "reason_codes": reasons,
    }


def _world_to_canvas(scene: SceneManifest, x: float, y: float, width: int, height: int) -> tuple[int, int]:
    margin_x, margin_y = 50.0, 40.0
    usable_w = max(1.0, width - 2 * margin_x)
    usable_h = max(1.0, height - 2 * margin_y)
    px = int(margin_x + (x / scene.width_m) * usable_w)
    # World y=0 is shown at the bottom of the image.
    py = int(height - margin_y - (y / scene.depth_m) * usable_h)
    return px, py


def _interpolate_route(route: RouteContract, t: float) -> tuple[float, float, float]:
    points = route.waypoints
    if not points:
        return 0.0, 0.0, 0.0
    if t <= points[0].t:
        return points[0].x, points[0].y, points[0].yaw
    if t >= points[-1].t:
        return points[-1].x, points[-1].y, points[-1].yaw
    for left, right in zip(points, points[1:]):
        if left.t <= t <= right.t:
            span = right.t - left.t
            u = (t - left.t) / span if span else 0.0
            # Smoothstep reduces visual and kinematic discontinuities.
            s = u * u * (3.0 - 2.0 * u)
            x = left.x + s * (right.x - left.x)
            y = left.y + s * (right.y - left.y)
            yaw = left.yaw + s * (right.yaw - left.yaw)
            return x, y, yaw
    last = points[-1]
    return last.x, last.y, last.yaw


class MockVideoGenerator:
    """Create a tiny deterministic MP4 for offline pipeline verification.

    It intentionally draws simple geometric rooms and colored targets.  This
    is a plumbing test, not a substitute for a Seedance quality evaluation.
    """

    def __init__(self, width: int = 640, height: int = 360, keep_frames: bool = False) -> None:
        self.width = width
        self.height = height
        self.keep_frames = keep_frames

    def generate(self, sample: SampleSpec, output_path: Path) -> Dict[str, Any]:
        if Image is None or ImageDraw is None:
            raise RuntimeError("Pillow is required for MockVideoGenerator")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_dir = output_path.parent / f".{sample.sample_id}_mock_frames"
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)
        fps = int(sample.scene.fps)
        duration = max(1.0, sample.route.duration_s)
        frame_count = max(2, int(round(duration * fps)))
        target = sample.scene.object_by_id(sample.task.target_id) if sample.task.target_id else None
        for index in range(frame_count):
            t = index / fps
            image = Image.new("RGB", (self.width, self.height), (224, 222, 211))
            draw = ImageDraw.Draw(image)
            # Floor and back wall.
            draw.rectangle((0, self.height * 0.58, self.width, self.height), fill=(158, 139, 116))
            draw.line((0, self.height * 0.58, self.width, self.height * 0.58), fill=(95, 81, 67), width=3)
            # Static objects are rendered with stable positions.
            for obj in sample.scene.objects:
                if obj.dynamic:
                    continue
                cx, cy = _world_to_canvas(sample.scene, obj.position[0], obj.position[1], self.width, self.height)
                half_w = max(8, int(obj.size[0] / sample.scene.width_m * (self.width - 100) / 2))
                half_h = max(8, int(obj.size[1] / sample.scene.depth_m * (self.height - 80) / 2))
                color = obj.color_rgb or (130, 130, 130)
                draw.rounded_rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h), radius=5, fill=color, outline=(40, 40, 40), width=2)
                if obj.category in {"buffet_cabinet", "bookcase", "bookshelf"}:
                    for divider in range(1, 4):
                        x = cx - half_w + int((2 * half_w) * divider / 4)
                        draw.line((x, cy - half_h, x, cy + half_h), fill=(70, 55, 42), width=2)
            # Dynamic targets follow a smooth route offset, preserving identity.
            if target is not None and target.dynamic:
                rx, ry, _ = _interpolate_route(sample.route, t)
                phase = min(1.0, t / duration)
                tx = target.position[0] + 0.8 * math.sin(phase * math.pi)
                ty = target.position[1] + 0.5 * phase
                # Include a small relationship to the robot route so tracking
                # examples are visually non-static.
                tx = 0.65 * tx + 0.35 * (rx + 0.5)
                ty = 0.65 * ty + 0.35 * (ry + 0.5)
                cx, cy = _world_to_canvas(sample.scene, tx, ty, self.width, self.height)
                color = target.color_rgb or (142, 190, 142)
                draw.ellipse((cx - 12, cy - 24, cx + 12, cy + 24), fill=color, outline=(30, 80, 30), width=2)
                draw.ellipse((cx - 8, cy - 40, cx + 8, cy - 24), fill=(194, 150, 116), outline=(50, 40, 30), width=1)
            # A subtle camera/robot motion cue prevents duplicate frames.
            rx, ry, _ = _interpolate_route(sample.route, t)
            cx, cy = _world_to_canvas(sample.scene, rx, ry, self.width, self.height)
            draw.polygon([(cx, cy - 10), (cx - 8, cy + 8), (cx + 8, cy + 8)], fill=(35, 45, 55))
            image.save(frame_dir / f"frame_{index:05d}.png")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = _run(command, timeout_s=max(120.0, duration * 20.0))
        if result.returncode != 0:
            raise RuntimeError(f"mock ffmpeg failed: {result.stderr[-500:]}")
        if not self.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
        return {
            "backend": "mock",
            "path": str(output_path),
            "duration_s": duration,
            "fps": fps,
            "frame_count": frame_count,
        }
