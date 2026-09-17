"""Command line entry points for the navigation data pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .config import PipelineConfig
from .pipeline import NavigationDataPipeline
from .planner import PlannerConfig, TemplatePlanner
from .prompting import PromptCompiler
from .rxr import load_rxr_episodes
from .trajectory import write_trajectory_jsonl
from .trajectory import read_trajectory_file
from .trajectory_variants import (
    VGGTTrajectoryVariantNames,
    build_vggt_variants,
    comparison_data_dict,
)
from .usage import aggregate_usage
from .video import make_trajectory_comparison_plot
from .video_trajectory import estimate_video_trajectory, trajectory_data_dict
from .vggt_trajectory import VGGTTrajectoryResult, estimate_vggt_trajectory, trajectory_data_dict as vggt_trajectory_data_dict


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1, help="number of samples")
    parser.add_argument("--out", type=Path, default=Path("runs/default"), help="output directory")
    parser.add_argument("--duration", type=float, default=15.0, help="video/route duration in seconds (4-15 for Seedance)")


def _add_vggt_options(parser: argparse.ArgumentParser, *, include_variant: bool = True) -> None:
    """Add shared VGGT runtime and post-processing options to a CLI command."""

    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-checkpoint", type=Path, default=None)
    parser.add_argument("--vggt-repo", type=Path, default=None)
    parser.add_argument("--vggt-device", default="auto")
    parser.add_argument("--vggt-frame-hz", type=float, default=2.0)
    parser.add_argument("--vggt-max-frames", type=int, default=48)
    parser.add_argument("--vggt-input-size", type=int, default=518)
    parser.add_argument("--vggt-smoothing-window", type=int, default=5)
    if include_variant:
        parser.add_argument(
            "--vggt-postprocess",
            choices=VGGTTrajectoryVariantNames,
            default="smoothed_scaled",
            help="post-processing variant; smoothed_scaled is the production default",
        )
    parser.add_argument(
        "--vggt-anchor-start",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="declared route-prior start (required for scaled variants)",
    )
    parser.add_argument(
        "--vggt-anchor-end",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="declared route-prior end (required for scaled variants)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and filter synthetic navigation samples")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="run dependency-light offline smoke test")
    _add_common(smoke)
    run = sub.add_parser("run", help="run offline, auto or live pipeline")
    _add_common(run)
    run.add_argument("--mode", choices=("offline", "auto", "live"), default="offline")
    run.add_argument(
        "--pipeline-profile",
        choices=("fixed", "experiment"),
        default=None,
        help="fixed production stage contract (default); experiment enables offline A/B overrides",
    )
    # ``None`` means the flag was omitted, allowing USE_REMOTE_* in .env or
    # the process environment to remain effective.  Passing an explicit
    # false value is still possible through the Python API.
    run.add_argument("--remote-planner", action="store_true", default=None)
    run.add_argument(
        "--remote-prompt",
        action="store_true",
        default=None,
        help="use Doubao to refine the compiled Seedance prompt (deterministic compiler remains the fallback)",
    )
    run.add_argument("--remote-vlm", action="store_true", default=None)
    run.add_argument("--remote-embedding", action="store_true", default=None)
    run.add_argument(
        "--trajectory-source",
        choices=("auto", "route_contract", "video_vo", "vggt", "external"),
        default=None,
        help="pose source: auto uses VGGT for live Seedance and the route prior for offline mock",
    )
    run.add_argument(
        "--trajectory-poses",
        type=Path,
        default=None,
        help="JSONL pose stream for --trajectory-source external (supports {sample_id})",
    )
    run.add_argument(
        "--no-video-vo-anchor",
        action="store_true",
        default=None,
        help="do not anchor video VO to route endpoints",
    )
    run.add_argument(
        "--vggt-postprocess",
        choices=VGGTTrajectoryVariantNames,
        default=None,
        help="VGGT variant for live Seedance (smoothed_scaled is the production default)",
    )
    run.add_argument("--vggt-smoothing-window", type=int, default=None)
    fixed = sub.add_parser("run-fixed", help="run the versioned fixed live production pipeline")
    _add_common(fixed)
    fixed.set_defaults(mode="live", pipeline_profile="fixed")
    episodes = sub.add_parser(
        "run-episodes",
        help="run supplied RxR/Matterport episodes through the fixed Seedance/VLM/VGGT pipeline",
    )
    _add_common(episodes)
    episodes.add_argument("--episodes", type=Path, required=True, help="NavGen RxR episode JSON file")
    episodes.add_argument("--mode", choices=("offline", "auto", "live"), default="live")
    episodes.add_argument(
        "--pipeline-profile",
        choices=("fixed", "experiment"),
        default="fixed",
        help="fixed is the production contract; experiment is only for explicit A/B runs",
    )
    episodes.add_argument("--local-scale", type=float, default=0.30, help="uniform Matterport-to-local route scale")
    episodes.add_argument("--sample-prefix", default="rxr", help="sample ID prefix")
    episodes.add_argument("--remote-planner", action="store_true", default=None, help="unused for supplied episodes")
    episodes.add_argument(
        "--remote-prompt",
        action="store_true",
        default=None,
        help="run the Doubao prompt-refiner after deterministic compilation",
    )
    episodes.add_argument("--remote-vlm", action="store_true", default=None)
    episodes.add_argument("--remote-embedding", action="store_true", default=None)
    episodes.add_argument(
        "--trajectory-source",
        choices=("auto", "route_contract", "video_vo", "vggt", "external"),
        default=None,
    )
    episodes.add_argument("--trajectory-poses", type=Path, default=None)
    episodes.add_argument("--no-video-vo-anchor", action="store_true", default=None)
    episodes.add_argument("--vggt-postprocess", choices=VGGTTrajectoryVariantNames, default=None)
    episodes.add_argument("--vggt-smoothing-window", type=int, default=None)
    episodes.add_argument("--vggt-model", default=None)
    episodes.add_argument("--vggt-checkpoint", type=Path, default=None)
    episodes.add_argument("--vggt-repo", type=Path, default=None)
    episodes.add_argument("--vggt-device", default=None)
    episodes.add_argument("--vggt-frame-hz", type=float, default=None)
    episodes.add_argument("--vggt-max-frames", type=int, default=None)
    episodes.add_argument("--vggt-input-size", type=int, default=None)
    plan = sub.add_parser("plan", help="write deterministic scene/task plans and compiled Seedance prompts")
    _add_common(plan)
    estimate = sub.add_parser(
        "estimate-trajectory", help="estimate a camera trajectory from an RGB video for review"
    )
    estimate.add_argument("--video", type=Path, required=True)
    estimate.add_argument("--out", type=Path, default=Path("runs/video_trajectory"))
    estimate.add_argument(
        "--estimator",
        choices=("opencv", "vggt"),
        default="vggt",
        help="RGB estimator to use (VGGT is the default for Seedance clips)",
    )
    estimate.add_argument("--sample-hz", type=float, default=10.0)
    estimate.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=None)
    estimate.add_argument("--end", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=None)
    estimate.add_argument("--focal-scale", type=float, default=0.9)
    _add_vggt_options(estimate)
    compare = sub.add_parser(
        "compare-trajectory", help="run one VGGT inference and compare raw/smoothed/scaled variants"
    )
    compare.add_argument("--video", type=Path, required=True)
    compare.add_argument("--out", type=Path, default=Path("runs/vggt_comparison"))
    compare.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="optional planned/reference trajectory JSONL or JSON for endpoint/path comparisons",
    )
    _add_vggt_options(compare, include_variant=False)
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "count") and args.count < 1:
        raise SystemExit("--count must be positive")
    if args.command == "plan":
        planner = TemplatePlanner(PlannerConfig(duration_s=args.duration))
        plans = planner.plan_batch(args.count)
        args.out.mkdir(parents=True, exist_ok=True)
        prompt_root = args.out / "prompts"
        prompt_root.mkdir(parents=True, exist_ok=True)
        compiler = PromptCompiler()
        prompt_manifest = []
        for plan in plans:
            bundle = compiler.apply(plan)
            sample_dir = prompt_root / plan.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "seedance_prompt.txt").write_text(bundle.prompt, encoding="utf-8")
            (sample_dir / "negative_prompt.txt").write_text(bundle.negative_prompt, encoding="utf-8")
            (sample_dir / "prompt_report.json").write_text(
                json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            prompt_manifest.append(
                {
                    "sample_id": plan.sample_id,
                    "prompt": str(sample_dir / "seedance_prompt.txt"),
                    "negative_prompt": str(sample_dir / "negative_prompt.txt"),
                    "report": str(sample_dir / "prompt_report.json"),
                    "quality": bundle.checks.get("quality"),
                    "compiler_version": bundle.version,
                }
            )
        path = args.out / "plans.json"
        path.write_text(
            json.dumps([_safe_plan(plan) for plan in plans], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        prompt_manifest_path = args.out / "prompt_manifest.jsonl"
        prompt_manifest_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in prompt_manifest) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"plans": args.count, "path": str(path), "prompt_manifest": str(prompt_manifest_path)},
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "estimate-trajectory":
        try:
            if args.estimator == "vggt":
                raw_result = estimate_vggt_trajectory(
                    args.video,
                    sample_hz=args.vggt_frame_hz,
                    max_frames=args.vggt_max_frames,
                    input_size=args.vggt_input_size,
                    model_id=args.vggt_model,
                    checkpoint=str(args.vggt_checkpoint) if args.vggt_checkpoint else None,
                    repo_path=str(args.vggt_repo) if args.vggt_repo else None,
                    device=args.vggt_device,
                )
                variants = build_vggt_variants(
                    raw_result,
                    anchor_start=tuple(args.vggt_anchor_start) if args.vggt_anchor_start else None,
                    anchor_end=tuple(args.vggt_anchor_end) if args.vggt_anchor_end else None,
                    smoothing_window=args.vggt_smoothing_window,
                )
                selected = variants[args.vggt_postprocess]
                result = VGGTTrajectoryResult(selected.trajectory, selected.report)
                data_dict = vggt_trajectory_data_dict(result)
            else:
                result = estimate_video_trajectory(
                    args.video,
                    sample_hz=args.sample_hz,
                    start_pose=tuple(args.start) if args.start else None,
                    end_pose=tuple(args.end) if args.end else None,
                    focal_scale=args.focal_scale,
                )
                data_dict = trajectory_data_dict(result)
            args.out.mkdir(parents=True, exist_ok=True)
            jsonl_path = write_trajectory_jsonl(args.out / "trajectory.jsonl", result.trajectory)
            json_path = args.out / "trajectory.json"
            json_path.write_text(
                json.dumps(data_dict, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {"trajectory": str(jsonl_path), "data": str(json_path), "report": result.report},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2

    if args.command == "compare-trajectory":
        try:
            raw_result = estimate_vggt_trajectory(
                args.video,
                sample_hz=args.vggt_frame_hz,
                max_frames=args.vggt_max_frames,
                input_size=args.vggt_input_size,
                model_id=args.vggt_model,
                checkpoint=str(args.vggt_checkpoint) if args.vggt_checkpoint else None,
                repo_path=str(args.vggt_repo) if args.vggt_repo else None,
                device=args.vggt_device,
            )
            anchor_start = tuple(args.vggt_anchor_start) if args.vggt_anchor_start else None
            anchor_end = tuple(args.vggt_anchor_end) if args.vggt_anchor_end else None
            variants = build_vggt_variants(
                raw_result,
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                smoothing_window=args.vggt_smoothing_window,
            )
            reference = read_trajectory_file(args.reference) if args.reference else None
            args.out.mkdir(parents=True, exist_ok=True)
            variant_paths: Dict[str, Dict[str, str]] = {}
            for name, variant in variants.items():
                variant_dir = args.out / name
                variant_dir.mkdir(parents=True, exist_ok=True)
                variant_result = VGGTTrajectoryResult(variant.trajectory, variant.report)
                jsonl_path = write_trajectory_jsonl(variant_dir / "trajectory.jsonl", variant.trajectory)
                json_path = variant_dir / "trajectory.json"
                json_path.write_text(
                    json.dumps(vggt_trajectory_data_dict(variant_result), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                # Store paths relative to the comparison directory so the
                # artifact can be copied into a browser/static bundle.
                variant_paths[name] = {
                    "jsonl": str(Path(name) / "trajectory.jsonl"),
                    "json": str(Path(name) / "trajectory.json"),
                }
            comparison = comparison_data_dict(
                variants,
                # Do not leak a workstation's absolute filesystem path into
                # a static/browser artifact.  The generated clip's basename
                # is sufficient to identify the input; callers can retain
                # the full path in their job metadata if needed.
                source_video=args.video.name,
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                reference=reference,
            )
            comparison["variant_paths"] = variant_paths
            comparison_plot = make_trajectory_comparison_plot(
                variants,
                args.out / "comparison.png",
                reference_trajectory=reference,
            )
            comparison["comparison_plot"] = "comparison.png" if comparison_plot else None
            comparison_path = args.out / "comparison.json"
            comparison_path.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "comparison": str(comparison_path),
                        "plot": str(comparison_plot) if comparison_plot else None,
                        "summary": comparison["summary"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2

    supplied_samples = None
    if args.command == "run-episodes":
        try:
            supplied_samples = load_rxr_episodes(
                args.episodes,
                duration_s=args.duration,
                local_scale=args.local_scale,
                sample_prefix=args.sample_prefix,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        # The episode file, rather than --count, defines the batch size.
        args.count = len(supplied_samples)

    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.command != "smoke" and args.mode == "live":
        rounded_duration = round(args.duration)
        if abs(args.duration - rounded_duration) > 1e-6 or not 4 <= rounded_duration <= 15:
            raise SystemExit("live Seedance runs require an integer --duration in [4, 15]")

    mode = "offline" if args.command == "smoke" else args.mode
    config = PipelineConfig.from_env(
        pipeline_profile=getattr(args, "pipeline_profile", None),
        duration_s=args.duration,
        use_remote_planner=getattr(args, "remote_planner", None),
        use_remote_prompt=getattr(args, "remote_prompt", None),
        use_remote_vlm=getattr(args, "remote_vlm", None),
        use_remote_embedding=getattr(args, "remote_embedding", None),
        trajectory_source=getattr(args, "trajectory_source", None),
        external_trajectory_path=(
            str(getattr(args, "trajectory_poses", None))
            if getattr(args, "trajectory_poses", None) is not None
            else None
        ),
        video_vo_anchor_endpoints=(
            not args.no_video_vo_anchor if getattr(args, "no_video_vo_anchor", None) is not None else None
        ),
        vggt_postprocess=getattr(args, "vggt_postprocess", None),
        vggt_smoothing_window=getattr(args, "vggt_smoothing_window", None),
        vggt_model=getattr(args, "vggt_model", None),
        vggt_checkpoint=(
            str(getattr(args, "vggt_checkpoint", None))
            if getattr(args, "vggt_checkpoint", None) is not None
            else None
        ),
        vggt_repo=(
            str(getattr(args, "vggt_repo", None))
            if getattr(args, "vggt_repo", None) is not None
            else None
        ),
        vggt_device=getattr(args, "vggt_device", None),
        vggt_frame_hz=getattr(args, "vggt_frame_hz", None),
        vggt_max_frames=getattr(args, "vggt_max_frames", None),
        vggt_input_size=getattr(args, "vggt_input_size", None),
    )
    # Live mode implies remote planner/VLM; embedding remains opt-in because
    # its SDK is an optional dependency.
    if mode == "live":
        # Embeddings remain opt-in because the SDK is optional.  Preserve an
        # environment setting when the CLI flag was omitted.
        if getattr(args, "remote_embedding", None) is not None:
            config.use_remote_embedding = bool(args.remote_embedding)
    try:
        pipeline = NavigationDataPipeline(config=config, output_dir=args.out, mode=mode)
        records = pipeline.run_batch(args.count, samples=supplied_samples)
    except (RuntimeError, ValueError, TimeoutError) as exc:
        # Keep credential/configuration failures readable in batch jobs and
        # avoid printing a traceback that could include provider internals.
        print(json.dumps({"error": str(exc), "mode": mode}, ensure_ascii=False), file=sys.stderr)
        return 2
    summary = {
        "mode": mode,
        "pipeline_contract": config.pipeline_contract(mode=mode),
        "out": str(args.out),
        "status_counts": {status: sum(record.status == status for record in records) for status in sorted({r.status for r in records})},
        "quality_tier_counts": {tier: sum(record.quality_tier == tier for record in records) for tier in sorted({r.quality_tier for r in records})},
        "samples": [
            {
                "sample_id": record.sample_id,
                "status": record.status,
                "quality_tier": record.quality_tier,
                "video": record.video_path,
                "trajectory": record.trajectory_path,
                "trajectory_data": record.trajectory_data_path,
                "planned_trajectory": record.planned_trajectory_path,
                "trajectory_plot": record.trajectory_plot_path,
                "errors": record.errors,
            }
            for record in records
        ],
        "usage": aggregate_usage(
            event
            for record in records
            if isinstance(record.usage_report, dict)
            for event in record.usage_report.get("events", [])
            if isinstance(event, dict)
        ),
        "usage_summary": str(args.out / "usage_summary.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _safe_plan(plan: Any) -> Dict[str, Any]:
    from .models import dataclass_to_dict

    return dataclass_to_dict(plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
