"""Runtime configuration with safe environment-variable handling."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ark_client import DEFAULT_BASE_URL


# The production contract is intentionally explicit.  Experiment commands can
# still exercise alternative estimators/variants, but a live run using the
# default profile must stay on this ordered path.
FIXED_PIPELINE_NAME = "seedance_vggt_navigation"
FIXED_PIPELINE_VERSION = "navgen-fixed-v1"
EXPERIMENT_PIPELINE_NAME = "seedance_navigation_experiment"
EXPERIMENT_PIPELINE_VERSION = "navgen-experiment-v1"
FIXED_PIPELINE_STAGES: Tuple[str, ...] = (
    "scene_task_planner",
    "prompt_compiler",
    "seedance_video",
    "pre_qc",
    "vlm_filter",
    "vggt_trajectory",
    "trajectory_filter",
    "dataset_packager",
)
FIXED_TRAJECTORY_SOURCE = "vggt"
FIXED_TRAJECTORY_VARIANT = "smoothed_scaled"
FIXED_SMOOTHING_WINDOW = 5


@dataclass
class PipelineConfig:
    # ``fixed`` is the production contract; ``experiment`` leaves the
    # estimator/post-processing switches available for offline A/B work.
    pipeline_profile: str = "fixed"
    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    planner_model: str = "ep-20260624154257-tl6bm"
    # Optional second pass that edits the compiled prompt while preserving all
    # scene/route facts.  It uses the same Ark Responses-compatible model by
    # default, but can be pointed at a dedicated endpoint.
    prompt_model: str = "ep-20260624154257-tl6bm"
    vlm_model: str = "doubao-seed-2-0-pro-260215"
    embedding_model: str = "ep-20260610171621-gs9vd"
    video_model: str = "ep-20260630104958-tmgzb"
    # Keep the default aligned with the planner.  A longer clip leaves enough
    # time for acceleration, approach and terminal dwell while staying within
    # Seedance's 4--15 s single-clip range.
    duration_s: float = 15.0
    fps: int = 24
    trajectory_hz: float = 10.0
    # ``auto`` uses the configured RGB estimator for real Seedance clips and
    # keeps the deterministic route contract for the offline mock.  Use
    # ``external`` when a simulator/robot pose file is available.
    trajectory_source: str = "auto"
    external_trajectory_path: Optional[str] = None
    video_vo_anchor_endpoints: bool = True
    video_vo_focal_scale: float = 0.9
    # VGGT is an optional, RGB-only estimator for Seedance clips.  It is kept
    # separate from the older OpenCV VO settings so experiments can switch
    # between the two without changing the sample contract.
    video_estimator: str = "vggt"
    vggt_model: str = "facebook/VGGT-1B"
    vggt_checkpoint: Optional[str] = None
    vggt_repo: Optional[str] = None
    vggt_device: str = "auto"
    vggt_frame_hz: float = 2.0
    vggt_max_frames: int = 48
    vggt_input_size: int = 518
    # The validated production path uses temporal smoothing followed by a
    # route-endpoint similarity transform.  The transform is an explicit
    # prior (never ground truth); ``raw``, ``smoothed`` and ``scaled`` remain
    # available for offline A/B experiments via the CLI.
    vggt_postprocess: str = "smoothed_scaled"
    vggt_smoothing_window: int = 5
    frame_sample_fps: float = 1.0
    resolution: str = "720p"
    ratio: str = "16:9"
    generate_audio: bool = False
    seeds_per_prompt: int = 1
    poll_s: float = 5.0
    task_timeout_s: float = 900.0
    # Optional account-specific rates used only for local cost estimates.
    # Leave unset to report usage while avoiding an invented price.
    llm_input_cny_per_1k: Optional[float] = None
    llm_output_cny_per_1k: Optional[float] = None
    # The VLM endpoint in the default pipeline is Doubao-Seed-2.0-pro,
    # whose public card is different from the Seed-2.1-pro planner card.
    # Keeping a separate pair prevents a blended rate from hiding the
    # model-specific assumption in an audit report.
    vlm_input_cny_per_1k: Optional[float] = None
    vlm_output_cny_per_1k: Optional[float] = None
    seedance_cny_per_second: Optional[float] = None
    seedance_cny_per_video: Optional[float] = None
    # Optional fallback when the gateway exposes a provider token counter for
    # video and the account is billed by that counter.  Keep this separate
    # from text-token rates because the units are provider-specific.
    seedance_cny_per_1k_tokens: Optional[float] = None
    use_remote_planner: bool = False
    use_remote_prompt: bool = False
    use_remote_vlm: bool = False
    use_remote_embedding: bool = False

    def __post_init__(self) -> None:
        if self.pipeline_profile not in {"fixed", "experiment"}:
            raise ValueError("pipeline_profile must be fixed or experiment")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.fps <= 0 or self.trajectory_hz <= 0 or self.frame_sample_fps <= 0:
            raise ValueError("fps, trajectory_hz and frame_sample_fps must be positive")
        if self.trajectory_source not in {"auto", "route_contract", "video_vo", "vggt", "external"}:
            raise ValueError("trajectory_source must be auto, route_contract, video_vo, vggt or external")
        if self.video_estimator not in {"video_vo", "vggt"}:
            raise ValueError("video_estimator must be video_vo or vggt")
        if self.video_vo_focal_scale <= 0:
            raise ValueError("video_vo_focal_scale must be positive")
        if self.vggt_frame_hz <= 0:
            raise ValueError("vggt_frame_hz must be positive")
        if self.vggt_max_frames < 2:
            raise ValueError("vggt_max_frames must be at least 2")
        if self.vggt_input_size <= 0 or self.vggt_input_size % 14 != 0:
            raise ValueError("vggt_input_size must be a positive multiple of 14")
        if self.vggt_postprocess not in {"raw", "smoothed", "scaled", "smoothed_scaled"}:
            raise ValueError("vggt_postprocess must be raw, smoothed, scaled or smoothed_scaled")
        if self.vggt_smoothing_window < 3:
            raise ValueError("vggt_smoothing_window must be at least 3")
        if self.vggt_device.strip().lower() not in {"auto", "cpu", "cuda"} and not self.vggt_device.strip().lower().startswith("cuda:"):
            raise ValueError("vggt_device must be auto, cpu, cuda or cuda:<index>")
        if self.poll_s <= 0 or self.task_timeout_s <= 0:
            raise ValueError("poll_s and task_timeout_s must be positive")
        for name in (
            "llm_input_cny_per_1k",
            "llm_output_cny_per_1k",
            "vlm_input_cny_per_1k",
            "vlm_output_cny_per_1k",
            "seedance_cny_per_second",
            "seedance_cny_per_video",
            "seedance_cny_per_1k_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def fixed_pipeline(self) -> bool:
        """Whether this configuration opts into the production contract."""

        return self.pipeline_profile == "fixed"

    def fixed_contract_errors(self, *, live: bool) -> List[str]:
        """Return deviations that would make a fixed run non-canonical.

        Offline/mock runs may use ``auto`` (which resolves to the deterministic
        route contract), but they must not be allowed to emit a raw/video-VO
        artifact carrying the production contract.  Explicit A/B settings
        therefore require ``pipeline_profile=experiment`` in every mode.
        """

        if not self.fixed_pipeline:
            return []
        errors: List[str] = []
        allowed_sources = {"auto", "vggt"} if live else {"auto", "route_contract", "vggt"}
        if self.trajectory_source not in allowed_sources:
            allowed = "/".join(sorted(allowed_sources))
            errors.append(f"trajectory_source must be {allowed} in the fixed {'live' if live else 'offline'} pipeline")
        if self.video_estimator != "vggt":
            errors.append(f"video_estimator must be vggt in the fixed {'live' if live else 'offline'} pipeline")
        if self.vggt_postprocess != FIXED_TRAJECTORY_VARIANT:
            errors.append(
                f"vggt_postprocess must be {FIXED_TRAJECTORY_VARIANT} in the fixed {'live' if live else 'offline'} pipeline"
            )
        if self.vggt_smoothing_window != FIXED_SMOOTHING_WINDOW:
            errors.append(
                f"vggt_smoothing_window must be {FIXED_SMOOTHING_WINDOW} in the fixed {'live' if live else 'offline'} pipeline"
            )
        return errors

    def pipeline_contract(self, *, mode: Optional[str] = None) -> Dict[str, Any]:
        """Return the versioned, secret-free contract embedded in run artifacts."""

        live_mode = mode == "live" or (mode == "auto" and bool(self.api_key))
        runtime_source = (
            FIXED_TRAJECTORY_SOURCE
            if self.fixed_pipeline and live_mode
            else "route_contract"
            if self.fixed_pipeline
            else self.trajectory_source
        )
        runtime_variant = (
            FIXED_TRAJECTORY_VARIANT
            if self.fixed_pipeline and live_mode
            else "route_contract"
            if self.fixed_pipeline
            else self.vggt_postprocess
        )
        return {
            "name": FIXED_PIPELINE_NAME if self.fixed_pipeline else EXPERIMENT_PIPELINE_NAME,
            "version": FIXED_PIPELINE_VERSION if self.fixed_pipeline else EXPERIMENT_PIPELINE_VERSION,
            "profile": self.pipeline_profile,
            "mode": mode,
            "stages": list(FIXED_PIPELINE_STAGES),
            "trajectory_source": FIXED_TRAJECTORY_SOURCE if self.fixed_pipeline else self.trajectory_source,
            "trajectory_variant": FIXED_TRAJECTORY_VARIANT if self.fixed_pipeline else self.vggt_postprocess,
            "smoothing_window": FIXED_SMOOTHING_WINDOW if self.fixed_pipeline else self.vggt_smoothing_window,
            "runtime_trajectory_source": runtime_source,
            "runtime_trajectory_variant": runtime_variant,
        }

    @property
    def seedance_duration_valid(self) -> bool:
        """Whether this configuration can be sent to Seedance 2.0."""

        return abs(self.duration_s - round(self.duration_s)) < 1e-6 and 4 <= round(self.duration_s) <= 15

    @classmethod
    def from_env(cls, **overrides: Any) -> "PipelineConfig":
        # ``.env`` is intentionally a tiny, dependency-free convenience for
        # workstation runs.  Explicit process environment variables always
        # win, and the file is ignored by git (see ``.gitignore``).  This
        # keeps the documented ``cp .env.example .env`` workflow honest
        # without requiring python-dotenv in the offline smoke-test image.
        _load_dotenv(Path.cwd() / ".env")

        def env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw is not None else default
            except ValueError:
                return default

        def env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        def env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        def env_optional_float(name: str) -> Optional[float]:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return None
            try:
                value = float(raw)
            except ValueError:
                return None
            return value if value >= 0 else None

        values: Dict[str, Any] = {
            "pipeline_profile": os.getenv("NAVGEN_PIPELINE_PROFILE", "fixed"),
            "base_url": os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL),
            "api_key": os.getenv("ARK_API_KEY"),
            "planner_model": os.getenv("PLANNER_MODEL", "ep-20260624154257-tl6bm"),
            "prompt_model": os.getenv("PROMPT_MODEL", os.getenv("PLANNER_MODEL", "ep-20260624154257-tl6bm")),
            "vlm_model": os.getenv("VLM_MODEL", "doubao-seed-2-0-pro-260215"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "ep-20260610171621-gs9vd"),
            "video_model": os.getenv("VIDEO_MODEL", "ep-20260630104958-tmgzb"),
            "duration_s": env_float("NAV_DURATION_S", 15.0),
            "fps": env_int("NAV_FPS", 24),
            "trajectory_hz": env_float("TRAJECTORY_HZ", 10.0),
            "trajectory_source": os.getenv("TRAJECTORY_SOURCE", "auto"),
            "external_trajectory_path": os.getenv("EXTERNAL_TRAJECTORY_PATH"),
            "video_vo_anchor_endpoints": env_bool("VIDEO_VO_ANCHOR_ENDPOINTS", True),
            "video_vo_focal_scale": env_float("VIDEO_VO_FOCAL_SCALE", 0.9),
            "video_estimator": os.getenv("VIDEO_ESTIMATOR", "vggt"),
            "vggt_model": os.getenv("VGGT_MODEL", "facebook/VGGT-1B"),
            "vggt_checkpoint": os.getenv("VGGT_CHECKPOINT"),
            "vggt_repo": os.getenv("VGGT_REPO"),
            "vggt_device": os.getenv("VGGT_DEVICE", "auto"),
            "vggt_frame_hz": env_float("VGGT_FRAME_HZ", 2.0),
            "vggt_max_frames": env_int("VGGT_MAX_FRAMES", 48),
            "vggt_input_size": env_int("VGGT_INPUT_SIZE", 518),
            "vggt_postprocess": os.getenv("VGGT_POSTPROCESS", "smoothed_scaled"),
            "vggt_smoothing_window": env_int("VGGT_SMOOTHING_WINDOW", 5),
            "frame_sample_fps": env_float("FRAME_SAMPLE_FPS", 1.0),
            "resolution": os.getenv("SEEDANCE_RESOLUTION", "720p"),
            "ratio": os.getenv("SEEDANCE_RATIO", "16:9"),
            "generate_audio": env_bool("SEEDANCE_GENERATE_AUDIO", False),
            "poll_s": env_float("ARK_POLL_S", 5.0),
            "task_timeout_s": env_float("ARK_TASK_TIMEOUT_S", 900.0),
            "llm_input_cny_per_1k": env_optional_float("LLM_INPUT_CNY_PER_1K"),
            "llm_output_cny_per_1k": env_optional_float("LLM_OUTPUT_CNY_PER_1K"),
            "vlm_input_cny_per_1k": env_optional_float("VLM_INPUT_CNY_PER_1K"),
            "vlm_output_cny_per_1k": env_optional_float("VLM_OUTPUT_CNY_PER_1K"),
            "seedance_cny_per_second": env_optional_float("SEEDANCE_CNY_PER_SECOND"),
            "seedance_cny_per_video": env_optional_float("SEEDANCE_CNY_PER_VIDEO"),
            "seedance_cny_per_1k_tokens": env_optional_float("SEEDANCE_CNY_PER_1K_TOKENS"),
            "use_remote_planner": env_bool("USE_REMOTE_PLANNER", False),
            "use_remote_prompt": env_bool("USE_REMOTE_PROMPT", False),
            "use_remote_vlm": env_bool("USE_REMOTE_VLM", False),
            "use_remote_embedding": env_bool("USE_REMOTE_EMBEDDING", False),
        }
        # CLI parsers use ``None`` for an omitted optional flag so that an
        # environment setting is not accidentally overwritten by ``False``.
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def safe_dict(self, *, mode: Optional[str] = None) -> Dict[str, Any]:
        result = asdict(self)
        result["api_key"] = "<set>" if self.api_key else None
        result["seedance_duration_valid"] = self.seedance_duration_valid
        result["pipeline_contract"] = self.pipeline_contract(mode=mode)
        return result

    def text_rates_for_stage(self, stage: str) -> tuple[Optional[float], Optional[float]]:
        """Return the input/output rates applicable to a text stage.

        Planner and prompt-refiner calls use the Seed-2.1-pro rate pair.
        The default VLM endpoint is Seed-2.0-pro, so callers can provide its
        separate pair without changing the public, backwards-compatible
        global ``LLM_*`` environment variables.
        """

        if stage == "vlm":
            return (
                self.vlm_input_cny_per_1k
                if self.vlm_input_cny_per_1k is not None
                else self.llm_input_cny_per_1k,
                self.vlm_output_cny_per_1k
                if self.vlm_output_cny_per_1k is not None
                else self.llm_output_cny_per_1k,
            )
        return self.llm_input_cny_per_1k, self.llm_output_cny_per_1k

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_dotenv(path: Path) -> None:
    """Load simple ``KEY=VALUE`` pairs without ever logging their values.

    This deliberately supports only the subset needed for deployment
    configuration.  Shell expansion, commands and multiline values are not
    evaluated; malformed lines are skipped.  Existing environment variables
    take precedence over values in the file.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError):
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" for char in key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
