"""End-to-end orchestration for offline and live navigation data generation."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ark_client import ArkEmbeddingClient, ArkHTTPClient, ArkResponsesClient, SeedanceClient
from .config import PipelineConfig
from .embedding import LocalEmbedding, extract_embedding_vectors, nearest_duplicate_pairs
from .models import PoseSample, SampleRecord, SampleSpec, TrajectoryReport, VLMReport, dataclass_to_dict
from .planner import DoubaoPlanner, PlannerConfig, TemplatePlanner
from .prompting import DoubaoPromptRefiner, PromptCompiler
from .trajectory import TrajectoryGenerator, pose_to_dict, read_trajectory_file, write_trajectory_jsonl
from .trajectory_filter import TrajectoryFilter
from .usage import aggregate_usage, text_event, video_event
from .video import MockVideoGenerator, extract_frames, make_contact_sheet, make_trajectory_plot, probe_video, video_pre_qc
from .video_trajectory import estimate_video_trajectory
from .vggt_trajectory import estimate_vggt_trajectory
from .vlm_filter import DoubaoVLMFilter, HeuristicVLMFilter
from .trajectory_variants import build_vggt_variants


def _redact_provider_payload(value: Any) -> Any:
    """Remove signed media URLs before provider responses enter an artifact."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"url", "video_url", "download_url", "uri", "signed_url"}:
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _redact_provider_payload(item)
        return result
    if isinstance(value, list):
        return [_redact_provider_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_provider_payload(item) for item in value]
    return value


def _safe_error(exc: BaseException) -> str:
    """Keep audit errors useful without persisting signed URLs or query data."""

    message = str(exc)
    return re.sub(r"https?://[^\s\]\}\)]+", "<redacted-url>", message)


def _scene_prior_matches_observation(prior: str, observed: str) -> bool:
    """Conservatively compare a planner room prior with a VLM label."""

    prior_text = re.sub(r"[_-]+", " ", str(prior).strip().lower())
    observed_text = str(observed).strip().lower()
    groups = {
        "hallway": ("hallway", "corridor", "passage", "走廊", "过道", "通道"),
        "living_room": ("living", "lounge", "客厅", "起居"),
        "study": ("study", "书房"),
        "office": ("office", "办公室"),
        "kitchen": ("kitchen", "厨房"),
        "dining_room": ("dining", "餐厅", "饭厅"),
        "bedroom": ("bedroom", "卧室"),
        "bathroom": ("bathroom", "bath", "卫生间", "浴室"),
        "open_space": ("open", "开放"),
    }
    matched = {
        canonical
        for canonical, markers in groups.items()
        if any(marker in prior_text for marker in markers)
    }
    if matched:
        return observed_text in matched
    # Unknown priors are not evidence of a mismatch; keep the VLM label but
    # leave the sample available for review.
    return True


def _attach_observed_scene_semantics(sample: SampleSpec, report: VLMReport) -> None:
    """Copy a confident VLM room label into the manifest with provenance."""

    label = report.observed_scene_label
    confidence = report.observed_scene_confidence
    if not label or confidence is None or confidence < 0.7:
        return
    source = report.observed_scene_source or report.source
    sample.scene.semantic_label = str(label)
    sample.scene.semantic_confidence = float(confidence)
    sample.scene.semantic_source = str(source)
    sample.metadata["scene_semantics"] = {
        "label": sample.scene.semantic_label,
        "confidence": sample.scene.semantic_confidence,
        "source": sample.scene.semantic_source,
        "planner_prior": sample.scene.room_type,
    }
    if not _scene_prior_matches_observation(sample.scene.room_type, sample.scene.semantic_label):
        if "scene_semantic_mismatch" not in report.reason_codes:
            report.reason_codes.append("scene_semantic_mismatch")
        if report.decision == "pass":
            report.decision = "review"


class DatasetPackager:
    """Persist artifacts and reports without discarding rejected samples."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def pack(
        self,
        sample: SampleSpec,
        status: str,
        video_path: Optional[Path],
        frames: Sequence[Path],
        contact_sheet: Optional[Path],
        pre_qc: Mapping[str, Any],
        vlm_report: Optional[VLMReport],
        trajectory: Sequence[PoseSample],
        trajectory_report: Optional[Any],
        embedding_report: Mapping[str, Any],
        usage_report: Optional[Mapping[str, Any]] = None,
        errors: Optional[Sequence[str]] = None,
        trajectory_plot: Optional[Path] = None,
        planned_trajectory: Optional[Sequence[PoseSample]] = None,
    ) -> SampleRecord:
        sample_dir = self.root / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        stored_video: Optional[Path] = None
        if video_path is not None and video_path.exists():
            stored_video = sample_dir / "video.mp4"
            if video_path.resolve() != stored_video.resolve():
                shutil.copy2(video_path, stored_video)
        stored_frames: List[Path] = []
        if frames:
            frame_dir = sample_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            for index, frame in enumerate(frames):
                destination = frame_dir / f"frame_{index:05d}{frame.suffix.lower() or '.jpg'}"
                if frame.resolve() != destination.resolve():
                    shutil.copy2(frame, destination)
                stored_frames.append(destination)
        stored_sheet: Optional[Path] = None
        if contact_sheet is not None and contact_sheet.exists():
            stored_sheet = sample_dir / "contact_sheet.jpg"
            if contact_sheet.resolve() != stored_sheet.resolve():
                shutil.copy2(contact_sheet, stored_sheet)
        stored_plot: Optional[Path] = None
        if trajectory_plot is not None and trajectory_plot.exists():
            stored_plot = sample_dir / "trajectory_plot.png"
            if trajectory_plot.resolve() != stored_plot.resolve():
                shutil.copy2(trajectory_plot, stored_plot)
        trajectory_path: Optional[Path] = None
        if trajectory:
            trajectory_path = sample_dir / "trajectory.jsonl"
            write_trajectory_jsonl(trajectory_path, trajectory)
        planned_trajectory_path: Optional[Path] = None
        if planned_trajectory:
            planned_trajectory_path = sample_dir / "planned_trajectory.jsonl"
            write_trajectory_jsonl(planned_trajectory_path, planned_trajectory)
        vlm_dict = dataclass_to_dict(vlm_report) if vlm_report is not None else None
        traj_dict = dataclass_to_dict(trajectory_report) if trajectory_report is not None else None
        trajectory_data_path: Optional[Path] = None
        if trajectory_report is not None:
            # Keep a single self-describing artifact for browser clients and
            # downstream annotators.  JSONL remains the canonical stream for
            # ingestion; this wrapper adds report/provenance and the planned
            # prior without conflating either path with ground truth.
            trajectory_data_path = sample_dir / "trajectory.json"
            trajectory_data = {
                "schema_version": "navgen-trajectory-v2",
                "pipeline_contract": (
                    dict(sample.metadata.get("pipeline_contract", {}))
                    if isinstance(sample.metadata.get("pipeline_contract"), Mapping)
                    else None
                ),
                "source": traj_dict.get("source") if isinstance(traj_dict, Mapping) else None,
                "ground_truth": bool(traj_dict.get("ground_truth", False)) if isinstance(traj_dict, Mapping) else False,
                "metric_scale": traj_dict.get("metric_scale", "unknown") if isinstance(traj_dict, Mapping) else "unknown",
                "metric_units": traj_dict.get("metric_units", "unknown") if isinstance(traj_dict, Mapping) else "unknown",
                "metrics_are_metric": bool(traj_dict.get("metrics_are_metric", False)) if isinstance(traj_dict, Mapping) else False,
                "report": traj_dict,
                "points": [pose_to_dict(item) for item in trajectory],
                "planned_points": [pose_to_dict(item) for item in (planned_trajectory or [])],
                "coordinate_frame": (
                    trajectory[0].frame_id
                    if trajectory
                    else (planned_trajectory[0].frame_id if planned_trajectory else None)
                ),
            }
            trajectory_data_path.write_text(
                json.dumps(trajectory_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # Keep rejected samples for audit, but make training eligibility an
        # explicit field rather than relying on a caller to infer it from a
        # status string.  A route contract is only a plan.  A real Seedance
        # clip with a video-only estimate must stay in review (B) until an
        # external simulator/robot pose stream is attached.  The offline mock
        # remains A solely so the dependency-light plumbing test can exercise
        # the complete packaging path.
        corrected_instruction = vlm_dict.get("corrected_instruction") if isinstance(vlm_dict, Mapping) else None
        trajectory_source = str(traj_dict.get("source", "unknown")) if isinstance(traj_dict, Mapping) else "unknown"
        trajectory_ground_truth = bool(traj_dict.get("ground_truth", False)) if isinstance(traj_dict, Mapping) else False
        generation_backend = (
            str(pre_qc.get("generation", {}).get("backend", ""))
            if isinstance(pre_qc.get("generation"), Mapping)
            else ""
        )
        eligible_pose = trajectory_ground_truth or generation_backend == "mock"
        if status == "accepted" and not corrected_instruction and eligible_pose:
            quality_tier = "A"
        elif status == "review" or (status == "accepted" and corrected_instruction):
            quality_tier = "B"
        elif status == "accepted":
            quality_tier = "B"
        else:
            quality_tier = "reject"
        spec_dict = dataclass_to_dict(sample)
        prompt_report = sample.metadata.get("prompt_report") if isinstance(sample.metadata, Mapping) else None
        # Promote the selected post-processing flags into the compact sample
        # provenance block.  Consumers should be able to identify the
        # canonical production variant without unpacking the full report.
        trajectory_provenance: Dict[str, Any] = {
            "source": trajectory_source,
            "ground_truth": trajectory_ground_truth,
            "training_eligible_pose": eligible_pose,
        }
        if isinstance(traj_dict, Mapping):
            for key in (
                "variant",
                "metric_scale",
                "metric_units",
                "metrics_are_metric",
                "temporal_smoothing",
                "scale_anchor",
                "estimator",
            ):
                if key in traj_dict:
                    trajectory_provenance[key] = traj_dict[key]
        sample_json = {
            "sample_id": sample.sample_id,
            "status": status,
            "sample_spec": spec_dict,
            "pipeline_contract": (
                dict(sample.metadata.get("pipeline_contract", {}))
                if isinstance(sample.metadata.get("pipeline_contract"), Mapping)
                else None
            ),
            "effective_instruction": corrected_instruction or sample.task.instruction_zh,
            "video": {"path": str(stored_video) if stored_video else None, "pre_qc": dict(pre_qc)},
            "frames": [str(path) for path in stored_frames],
            "contact_sheet": str(stored_sheet) if stored_sheet else None,
            "trajectory_path": str(trajectory_path) if trajectory_path else None,
            "trajectory_data": str(trajectory_data_path) if trajectory_data_path else None,
            "planned_trajectory_path": str(planned_trajectory_path) if planned_trajectory_path else None,
            "trajectory_plot": str(stored_plot) if stored_plot else None,
            "vlm_report": vlm_dict,
            "trajectory_report": traj_dict,
            "trajectory_provenance": trajectory_provenance,
            "prompt_report": dict(prompt_report) if isinstance(prompt_report, Mapping) else None,
            "embedding_report": dict(embedding_report),
            "usage": dict(usage_report) if isinstance(usage_report, Mapping) else None,
            "quality": {
                "tier": quality_tier,
                "training_eligible": quality_tier == "A",
                "selection_rule": "VLM pass + trajectory pass + observed ground-truth pose (or offline mock)",
            },
            "errors": list(errors or []),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (sample_dir / "sample.json").write_text(json.dumps(sample_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "pre_qc.json").write_text(json.dumps(dict(pre_qc), ensure_ascii=False, indent=2), encoding="utf-8")
        if vlm_dict is not None:
            (sample_dir / "vlm_report.json").write_text(json.dumps(vlm_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        if traj_dict is not None:
            (sample_dir / "trajectory_report.json").write_text(json.dumps(traj_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(prompt_report, Mapping):
            (sample_dir / "prompt_report.json").write_text(json.dumps(dict(prompt_report), ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "embedding_report.json").write_text(json.dumps(dict(embedding_report), ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(usage_report, Mapping):
            (sample_dir / "usage.json").write_text(json.dumps(dict(usage_report), ensure_ascii=False, indent=2), encoding="utf-8")
        return SampleRecord(
            sample_id=sample.sample_id,
            status=status,
            sample_spec=spec_dict,
            pipeline_contract=(
                dict(sample.metadata.get("pipeline_contract", {}))
                if isinstance(sample.metadata.get("pipeline_contract"), Mapping)
                else None
            ),
            quality_tier=quality_tier,
            video_path=str(stored_video) if stored_video else None,
            frame_paths=[str(path) for path in stored_frames],
            contact_sheet_path=str(stored_sheet) if stored_sheet else None,
            prompt_report=dict(prompt_report) if isinstance(prompt_report, Mapping) else None,
            trajectory_path=str(trajectory_path) if trajectory_path else None,
            trajectory_plot_path=str(stored_plot) if stored_plot else None,
            trajectory_data_path=str(trajectory_data_path) if trajectory_data_path else None,
            planned_trajectory_path=str(planned_trajectory_path) if planned_trajectory_path else None,
            vlm_report=vlm_dict,
            trajectory_report=traj_dict,
            embedding_report=dict(embedding_report),
            usage_report=dict(usage_report) if isinstance(usage_report, Mapping) else None,
            errors=list(errors or []),
        )


class NavigationDataPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        output_dir: Path | str = Path("runs/default"),
        mode: str = "offline",
    ) -> None:
        if mode not in {"offline", "live", "auto"}:
            raise ValueError("mode must be offline, live or auto")
        self.config = config or PipelineConfig.from_env()
        self.mode = mode
        self._live_run = mode == "live" or (mode == "auto" and bool(self.config.api_key))
        contract_errors = self.config.fixed_contract_errors(live=self._live_run)
        if contract_errors:
            details = "; ".join(contract_errors)
            raise ValueError(
                f"fixed pipeline contract violation: {details}; "
                "use pipeline_profile='experiment' only for explicit non-canonical/A-B runs"
            )
        self.pipeline_contract = self.config.pipeline_contract(mode=mode)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.output_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.packager = DatasetPackager(self.output_dir / "samples")
        self.template_planner = TemplatePlanner(
            PlannerConfig(duration_s=self.config.duration_s, camera_height_m=1.2, fps=self.config.fps), seed=7
        )
        self.http: Optional[ArkHTTPClient] = None
        self.responses: Optional[ArkResponsesClient] = None
        self.seedance: Optional[SeedanceClient] = None
        self.remote_planner: Optional[DoubaoPlanner] = None
        self.remote_prompt: Optional[DoubaoPromptRefiner] = None
        self.remote_vlm: Optional[DoubaoVLMFilter] = None
        self.remote_embedding: Optional[ArkEmbeddingClient] = None
        self.local_embedding = LocalEmbedding()
        self.prompt_compiler = PromptCompiler()
        self._prompt_fallback_errors: List[str] = []
        self._planner_fallback_error: Optional[str] = None
        self.mock_video = MockVideoGenerator()
        self.trajectory_generator = TrajectoryGenerator(sample_hz=self.config.trajectory_hz)
        self.trajectory_filter = TrajectoryFilter()
        self._configure_backends()

    def _append_usage_event(self, sample: SampleSpec, event: Mapping[str, Any]) -> None:
        events = sample.metadata.setdefault("usage_events", [])
        if isinstance(events, list):
            events.append(dict(event))

    def _capture_response_usage(
        self,
        sample: SampleSpec,
        history_start: int,
        stage: str,
        model: str,
    ) -> int:
        """Attach new secret-free Responses call records to one sample."""

        if self.responses is None:
            return history_start
        history = self.responses.call_history[history_start:]
        input_rate, output_rate = self.config.text_rates_for_stage(stage)
        for call in history:
            self._append_usage_event(
                sample,
                text_event(
                    stage,
                    model,
                    call,
                    input_rate_cny_per_1k=input_rate,
                    output_rate_cny_per_1k=output_rate,
                ),
            )
        return history_start + len(history)

    @property
    def live_enabled(self) -> bool:
        return bool(self.config.api_key)

    def _configure_backends(self) -> None:
        wants_live = self.mode == "live" or (self.mode == "auto" and self.live_enabled)
        if not wants_live:
            return
        if not self.live_enabled:
            raise RuntimeError("live mode requested but ARK_API_KEY is not set")
        if not self.config.seedance_duration_valid:
            raise ValueError("live Seedance runs require an integer duration in [4, 15] seconds")
        self.http = ArkHTTPClient(base_url=self.config.base_url, api_key=self.config.api_key)
        self.responses = ArkResponsesClient(self.http)
        self.seedance = SeedanceClient(self.http)
        self.remote_planner = DoubaoPlanner(
            self.responses, self.config.planner_model, duration_s=self.config.duration_s
        )
        if self.config.use_remote_prompt:
            self.remote_prompt = DoubaoPromptRefiner(self.responses, self.config.prompt_model, self.prompt_compiler)
        self.remote_vlm = DoubaoVLMFilter(self.responses, self.config.vlm_model)
        if self.config.use_remote_embedding:
            self.remote_embedding = ArkEmbeddingClient(
                self.config.embedding_model, base_url=self.config.base_url, api_key=self.config.api_key
            )

    def prepare_samples(self, samples: Sequence[SampleSpec]) -> List[SampleSpec]:
        """Run the canonical prompt stage for planner- or importer-produced samples.

        Imported RxR episodes intentionally skip the generative scene planner:
        their instruction and reference path are the source-of-record.  They
        still pass through exactly the same deterministic compiler and optional
        Doubao prompt-refiner as ordinary planned samples.
        """

        prepared = list(samples)
        # Always run the deterministic compiler.  In live mode an optional
        # Doubao refiner can improve wording, but it is never allowed to
        # replace the structured facts or safety constraints.
        for sample in prepared:
            try:
                bundle = self.prompt_compiler.apply(sample)
                if self.remote_prompt is not None:
                    prompt_history_start = len(self.responses.call_history) if self.responses is not None else 0
                    try:
                        refined = self.remote_prompt.refine(sample)
                        sample.seedance_prompt = refined.prompt
                        sample.metadata["prompt_report"] = refined.to_dict()
                        sample.metadata["prompt_refiner_model"] = self.config.prompt_model
                    except Exception as exc:
                        self._prompt_fallback_errors.append(f"{sample.sample_id}: {_safe_error(exc)}")
                        sample.metadata["prompt_report"] = bundle.to_dict()
                        sample.metadata["prompt_report"]["refiner_fallback"] = _safe_error(exc)
                    finally:
                        self._capture_response_usage(
                            sample,
                            prompt_history_start,
                            "prompt_refiner",
                            self.config.prompt_model,
                        )
                sample.metadata["prompt_report"]["planner_source"] = sample.planner_source
                sample.metadata.setdefault("planner_stage", sample.planner_source)
            except Exception as exc:
                # A prompt compile failure is fatal for the sample, but keep a
                # readable error for the batch summary instead of sending a
                # weak one-line prompt to Seedance.
                self._prompt_fallback_errors.append(f"{sample.sample_id}: compiler_error:{exc}")
                if self._live_run and self.config.fixed_pipeline:
                    raise RuntimeError(
                        f"fixed pipeline prompt compilation failed for {sample.sample_id}: {_safe_error(exc)}"
                    ) from exc
        return prepared

    def plan_samples(self, count: int) -> List[SampleSpec]:
        planner_history_start = len(self.responses.call_history) if self.responses is not None else 0
        if self.remote_planner is not None:
            try:
                samples = self.remote_planner.plan_batch(count)
            except Exception as exc:
                # Keep a reproducible fallback for a transient planner error;
                # the reason is recorded in run_summary.
                self._planner_fallback_error = _safe_error(exc)
                if self._live_run and self.config.fixed_pipeline:
                    # A fixed live run must never silently replace a failed
                    # model-generated scene/task with a template and then
                    # charge/send a video request for it.
                    raise RuntimeError(
                        f"fixed pipeline planner failed: {self._planner_fallback_error}"
                    ) from exc
                samples = self.template_planner.plan_batch(count)
        else:
            samples = self.template_planner.plan_batch(count)

        # A planner batch emits one Responses call per sample.  Preserve the
        # call usage beside the corresponding sample; if a partial batch
        # failed, attach the remaining error event to the first fallback
        # sample so it is still visible in the run audit.
        if self.responses is not None and samples:
            planner_calls = self.responses.call_history[planner_history_start:]
            planner_input_rate, planner_output_rate = self.config.text_rates_for_stage("planner")
            for index, call in enumerate(planner_calls):
                target_sample = samples[min(index, len(samples) - 1)]
                self._append_usage_event(
                    target_sample,
                    text_event(
                        "planner",
                        self.config.planner_model,
                        call,
                        input_rate_cny_per_1k=planner_input_rate,
                        output_rate_cny_per_1k=planner_output_rate,
                    ),
                )
        return self.prepare_samples(samples)

    def _video_content(self, sample: SampleSpec) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": sample.seedance_prompt}]
        for reference in sample.scene.reference_images:
            # Reference URLs must be reachable by Ark.  Local paths are not
            # silently sent as URLs; they are retained in metadata instead.
            if str(reference).startswith(("http://", "https://", "data:")):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": reference},
                        "role": "reference_image",
                    }
                )
        return content

    def _generate_video(self, sample: SampleSpec) -> Tuple[Path, Dict[str, Any]]:
        output_path = self.work_dir / sample.sample_id / "generated.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        wants_live = self.mode == "live" or (self.mode == "auto" and self.live_enabled)
        if not wants_live:
            metadata = self.mock_video.generate(sample, output_path)
            metadata = dict(metadata)
            metadata["generation_latency_s"] = round(time.monotonic() - started, 6)
            metadata["requested_duration_s"] = sample.route.duration_s
            return output_path, metadata
        if self.seedance is None:
            raise RuntimeError("Seedance client is not configured")
        task = self.seedance.create_task(
            model=self.config.video_model,
            content=self._video_content(sample),
            duration=int(round(sample.route.duration_s)),
            resolution=self.config.resolution,
            ratio=self.config.ratio,
            generate_audio=self.config.generate_audio,
            seed=sample.seed,
        )
        task_id = str(task["id"])
        final = self.seedance.wait_for_task(task_id, poll_s=self.config.poll_s, timeout_s=self.config.task_timeout_s)
        task_path = output_path.parent / "seedance_task.json"
        # Signed object-store URLs are credentials.  Keep the task response
        # useful for audit while removing URL/query material before writing it
        # to disk or surfacing it through the dashboard.
        safe_final = _redact_provider_payload(final)
        task_path.write_text(json.dumps(safe_final, ensure_ascii=False, indent=2), encoding="utf-8")
        status = str(final.get("status", "")).lower()
        if status not in {"succeeded", "success", "completed"}:
            raise RuntimeError(f"Seedance task {task_id} ended with status {status}")
        url = self.seedance.video_url(final)
        if not url:
            raise RuntimeError(f"Seedance task {task_id} succeeded without video_url")
        self.seedance.download(url, output_path)
        task_cost = final.get("cost_cny", final.get("estimated_cost_cny"))
        task_billing = final.get("billing", final.get("bill"))
        return output_path, {
            "backend": "seedance",
            "task_id": task_id,
            "video_url_present": True,
            "task_status": status,
            "task_usage": dict(final.get("usage", {})) if isinstance(final.get("usage"), Mapping) else {},
            "task_cost_cny": task_cost,
            "task_billing": dict(task_billing) if isinstance(task_billing, Mapping) else None,
            "task_latency_s": round(time.monotonic() - started, 6),
            "requested_duration_s": sample.route.duration_s,
            "path": str(output_path),
        }

    def _embedding_report(self, sample: SampleSpec, frames: Sequence[Path]) -> Dict[str, Any]:
        local_temporal = self.local_embedding.temporal_consistency(frames)
        report: Dict[str, Any] = {
            "backend": "local",
            "temporal_consistency": local_temporal,
            "frame_count": len(frames),
            "duplicate_frame_pairs": [
                {"left": left, "right": right, "similarity": score}
                for left, right, score in nearest_duplicate_pairs(frames, self.local_embedding, threshold=0.999)
            ],
        }
        if self.remote_embedding is not None and frames:
            try:
                # Keep the remote request compact; a contact sheet is created
                # later, so use the first few keyframes here.
                from .ark_client import file_to_data_url

                inputs: List[Dict[str, Any]] = [
                    {"type": "text", "text": sample.task.instruction_zh}
                ]
                for frame in frames[:8]:
                    inputs.append({"type": "image_url", "image_url": {"url": file_to_data_url(frame)}})
                response = self.remote_embedding.create(inputs)
                vectors = extract_embedding_vectors(response)
                report.update({"backend": "ark+local", "remote_vector_count": len(vectors)})
            except Exception as exc:
                report["remote_error"] = _safe_error(exc)
        return report

    def _vlm_report(
        self,
        sample: SampleSpec,
        frames: Sequence[Path],
        contact_sheet: Optional[Path],
        video_info: Any,
        embedding_report: Mapping[str, Any],
    ) -> VLMReport:
        if self.remote_vlm is not None:
            try:
                return self.remote_vlm.evaluate(sample, frames, contact_sheet=contact_sheet, video_info=video_info)
            except Exception as exc:
                # A failed remote judge is a review condition, never an
                # implicit pass.  Fall back to local checks so the sample is
                # still auditable.
                fallback = HeuristicVLMFilter().evaluate(sample, frames, video_info, embedding_report)
                fallback.decision = "review"
                fallback.reason_codes.append("remote_vlm_error")
                fallback.evidence.append({"error": _safe_error(exc), "reason_code": "remote_vlm_error"})
                return fallback
        return HeuristicVLMFilter(embedding=self.local_embedding).evaluate(sample, frames, video_info, embedding_report)

    def _effective_trajectory_source(self, generation_meta: Mapping[str, Any]) -> str:
        configured = self.config.trajectory_source
        if configured == "auto":
            # A mock clip has a known synthetic route by construction.  A
            # real Seedance clip has no pose telemetry, so estimate motion
            # from the RGB frames instead of silently reusing the plan.  VGGT
            # is the default RGB estimator; VIDEO_ESTIMATOR=video_vo keeps the
            # previous OpenCV implementation available for A/B tests.
            if generation_meta.get("backend") == "seedance":
                return self.config.video_estimator
            return "route_contract"
        return configured

    def _external_trajectory_path(self, sample: SampleSpec) -> Optional[Path]:
        configured = self.config.external_trajectory_path
        if not configured:
            return None
        rendered = str(configured).replace("{sample_id}", sample.sample_id)
        candidate = Path(rendered)
        if candidate.is_dir():
            for name in ("trajectory.jsonl", "trajectory.json"):
                nested = candidate / sample.sample_id / name
                if nested.exists():
                    return nested
                nested = candidate / name
                if nested.exists():
                    return nested
            return candidate / "trajectory.jsonl"
        return candidate

    @staticmethod
    def _route_endpoint_anchor(sample: SampleSpec) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return the declared route endpoints used only by opt-in scaling."""

        waypoints = sample.route.waypoints
        if len(waypoints) < 2:
            return None
        start = waypoints[0]
        end = waypoints[-1]
        # A repeated final waypoint represents terminal dwell, not another
        # movement anchor.  Use the last moving waypoint when present.
        if len(waypoints) >= 2 and end.x == waypoints[-2].x and end.y == waypoints[-2].y:
            end = waypoints[-2]
        return (float(start.x), float(start.y)), (float(end.x), float(end.y))

    def _build_trajectory(
        self,
        sample: SampleSpec,
        video_path: Path,
        generation_meta: Mapping[str, Any],
    ) -> Tuple[List[PoseSample], TrajectoryReport, Dict[str, Any]]:
        """Build an observed trajectory and retain an explicit provenance record."""

        source = self._effective_trajectory_source(generation_meta)
        if source == "route_contract":
            trajectory = self.trajectory_generator.generate(sample.route, confidence=1.0, source="route_contract")
            report = TrajectoryReport(
                decision="pass",
                metrics={},
                confidence=1.0,
                source="route_contract",
                ground_truth=False,
                metric_scale="route_contract",
                estimator="route_contract_interpolator",
                provenance={"kind": "planned_prior", "observed": False},
                metric_units="m",
                metrics_are_metric=True,
            )
            return trajectory, report, report.provenance

        if source == "external":
            path = self._external_trajectory_path(sample)
            if path is None or not path.exists():
                report = TrajectoryReport(
                    decision="review",
                    metrics={},
                    warnings=["external_trajectory_file_missing"],
                    confidence=0.0,
                    source="external_ground_truth_unavailable",
                    ground_truth=False,
                    metric_scale="unknown",
                    provenance={"configured_path": str(path) if path else None},
                )
                return [], report, report.provenance
            trajectory = read_trajectory_file(path)
            report = TrajectoryReport(
                decision="pass",
                metrics={},
                confidence=min(1.0, max(0.0, min(pose.confidence for pose in trajectory))),
                source="external_ground_truth",
                ground_truth=True,
                metric_scale="external_declared",
                estimator="external_pose_stream",
                provenance={"path": str(path), "frame": "caller_declared"},
                metric_units="m",
                metrics_are_metric=True,
            )
            return trajectory, report, report.provenance

        if source == "video_vo":
            # The final repeated waypoint is the terminal dwell.  Anchor to
            # the last moving waypoint, not to a synthetic extra stop sample.
            end_waypoint = sample.route.waypoints[-1]
            if len(sample.route.waypoints) >= 2 and sample.route.waypoints[-1].x == sample.route.waypoints[-2].x and sample.route.waypoints[-1].y == sample.route.waypoints[-2].y:
                end_waypoint = sample.route.waypoints[-2]
            result = estimate_video_trajectory(
                video_path,
                sample_hz=self.config.trajectory_hz,
                # The route's first yaw is the declared camera/robot heading;
                # TaskSpec historically stored only a coarse start pose and
                # may leave its yaw at zero.
                start_pose=(sample.route.waypoints[0].x, sample.route.waypoints[0].y, sample.route.waypoints[0].yaw),
                end_pose=(end_waypoint.x, end_waypoint.y, end_waypoint.yaw)
                if self.config.video_vo_anchor_endpoints
                else None,
                focal_scale=self.config.video_vo_focal_scale,
            )
            report = TrajectoryReport(
                decision="review",
                metrics={str(key): float(value) for key, value in result.report.get("metrics", {}).items() if isinstance(value, (int, float))},
                warnings=list(result.report.get("warnings", [])),
                confidence=float(result.report.get("confidence", 0.0)),
                source="video_vo_estimate",
                ground_truth=False,
                metric_scale=str(result.report.get("metric_scale", "unknown")),
                estimator=str(result.report.get("estimator", "video_vo")),
                provenance=dict(result.report),
                metric_units=str(
                    result.report.get(
                        "metric_units",
                        "m" if result.report.get("metric_scale") != "unknown" else "relative",
                    )
                ),
                metrics_are_metric=False,
                scale_anchor=str(result.report.get("anchor", "none")),
            )
            # Video VO is intentionally never auto-promoted to pass/A.  It is
            # useful for review and visualization, but a robot/simulator pose
            # stream is required before training eligibility is granted.
            return result.trajectory, report, report.provenance

        if source == "vggt":
            raw_result = estimate_vggt_trajectory(
                video_path,
                sample_hz=self.config.vggt_frame_hz,
                max_frames=self.config.vggt_max_frames,
                input_size=self.config.vggt_input_size,
                model_id=self.config.vggt_model,
                checkpoint=self.config.vggt_checkpoint,
                repo_path=self.config.vggt_repo,
                device=self.config.vggt_device,
            )
            anchor = self._route_endpoint_anchor(sample)
            variants = build_vggt_variants(
                raw_result,
                anchor_start=anchor[0] if anchor else None,
                anchor_end=anchor[1] if anchor else None,
                smoothing_window=self.config.vggt_smoothing_window,
            )
            selected = variants[self.config.vggt_postprocess]
            report = TrajectoryReport(
                decision="review",
                metrics={
                    str(key): float(value)
                    for key, value in selected.report.get("metrics", {}).items()
                    if isinstance(value, (int, float))
                },
                warnings=list(selected.report.get("warnings", [])),
                confidence=float(selected.report.get("confidence", 0.0)),
                source=str(selected.report.get("source", "vggt_monocular_estimate")),
                ground_truth=False,
                metric_scale=str(selected.report.get("metric_scale", "unknown")),
                estimator=str(selected.report.get("estimator", "vggt_camera_head")),
                provenance=dict(selected.report),
                metric_units=str(selected.report.get("metric_units", "relative")),
                metrics_are_metric=bool(selected.report.get("metrics_are_metric", False)),
                temporal_smoothing=bool(selected.report.get("temporal_smoothing", False)),
                scale_anchor=str(selected.report.get("scale_anchor", "none")),
                variant=(
                    str(selected.report.get("variant"))
                    if selected.report.get("variant") is not None
                    else None
                ),
            )
            # The production default is ``smoothed_scaled``.  Every selected
            # variant is explicitly marked in provenance and remains
            # review-only/non-ground-truth because the scale comes from a
            # route prior rather than pose telemetry.
            return selected.trajectory, report, report.provenance

        raise ValueError(f"unsupported trajectory source: {source}")

    def run_one(self, sample: SampleSpec) -> SampleRecord:
        # Carry the versioned contract into every sample so downstream jobs
        # can reject artifacts produced by a different stage order/config.
        sample.metadata["pipeline_contract"] = dict(self.pipeline_contract)
        errors: List[str] = []
        video_path: Optional[Path] = None
        frames: List[Path] = []
        contact_sheet: Optional[Path] = None
        trajectory_plot: Optional[Path] = None
        pre_qc: Dict[str, Any] = {"decision": "reject", "metrics": {}, "reason_codes": []}
        vlm_report: Optional[VLMReport] = None
        trajectory: List[PoseSample] = []
        planned_trajectory: List[PoseSample] = []
        trajectory_report: Optional[Any] = None
        embedding_report: Dict[str, Any] = {}
        generation_meta: Dict[str, Any] = {}
        generation_started = time.monotonic()
        try:
            video_path, generation_meta = self._generate_video(sample)
            pre_qc = video_pre_qc(video_path, expected_duration_s=sample.route.duration_s)
            pre_qc["generation"] = generation_meta
            if generation_meta.get("backend") == "seedance":
                task_response = {
                    "id": generation_meta.get("task_id"),
                    "status": generation_meta.get("task_status", "unknown"),
                    "usage": generation_meta.get("task_usage", {}),
                }
                if generation_meta.get("task_cost_cny") is not None:
                    task_response["cost_cny"] = generation_meta.get("task_cost_cny")
                if isinstance(generation_meta.get("task_billing"), Mapping):
                    task_response["billing"] = dict(generation_meta["task_billing"])
                self._append_usage_event(
                    sample,
                    video_event(
                        self.config.video_model,
                        task_response,
                        requested_duration_s=generation_meta.get("requested_duration_s", sample.route.duration_s),
                        actual_duration_s=pre_qc.get("metrics", {}).get("duration_s"),
                        latency_s=float(generation_meta.get("task_latency_s", 0.0) or 0.0),
                        resolution=self.config.resolution,
                        ratio=self.config.ratio,
                        size_bytes=pre_qc.get("metrics", {}).get("size_bytes"),
                        per_second_cny=self.config.seedance_cny_per_second,
                        per_video_cny=self.config.seedance_cny_per_video,
                        per_1k_tokens_cny=self.config.seedance_cny_per_1k_tokens,
                    ),
                )
            elif generation_meta:
                # Offline/mock generation still gets duration and latency
                # accounting, but is explicitly non-billable.
                self._append_usage_event(
                    sample,
                    {
                        "kind": "video",
                        "stage": "mock_video",
                        "model": "mock",
                        "status": "completed",
                        "latency_s": float(generation_meta.get("generation_latency_s", 0.0) or 0.0),
                        "requested_duration_s": float(sample.route.duration_s),
                        "actual_duration_s": pre_qc.get("metrics", {}).get("duration_s"),
                        "resolution": (
                            f"{pre_qc.get('metrics', {}).get('width')}x{pre_qc.get('metrics', {}).get('height')}"
                            if pre_qc.get("metrics", {}).get("width") and pre_qc.get("metrics", {}).get("height")
                            else None
                        ),
                        "ratio": self.config.ratio,
                        "size_bytes": pre_qc.get("metrics", {}).get("size_bytes"),
                        "cost_cny": 0.0,
                        "cost_status": "not_applicable",
                        "cost_basis": "offline_mock",
                    },
                )
            if pre_qc.get("decision") != "pass":
                errors.extend(str(item) for item in pre_qc.get("reason_codes", []))
            else:
                frame_dir = video_path.parent / "frames"
                frames = extract_frames(video_path, frame_dir, sample_fps=self.config.frame_sample_fps)
                contact_sheet = make_contact_sheet(frames, video_path.parent / "contact_sheet.jpg")
                video_info = probe_video(video_path)
                embedding_report = self._embedding_report(sample, frames)
                vlm_history_start = len(self.responses.call_history) if self.responses is not None else 0
                vlm_report = self._vlm_report(sample, frames, contact_sheet, video_info, embedding_report)
                _attach_observed_scene_semantics(sample, vlm_report)
                self._capture_response_usage(sample, vlm_history_start, "vlm", self.config.vlm_model)
                if vlm_report.decision != "pass":
                    errors.extend(vlm_report.reason_codes or ["vlm_not_pass"])
                # Build the observed pose stream separately from the planned
                # route.  In live mode this is an RGB estimate (VGGT by
                # default, or the legacy video VO) or an explicitly supplied
                # external pose file; the route contract is kept only as the
                # planned reference.
                try:
                    trajectory, trajectory_provenance_report, trajectory_provenance = self._build_trajectory(
                        sample, video_path, generation_meta
                    )
                except Exception as exc:
                    # A missing OpenCV extra or a corrupt pose file must not
                    # cause the observed video to be relabelled as the plan.
                    trajectory = []
                    trajectory_provenance_report = TrajectoryReport(
                        decision="review",
                        metrics={},
                        warnings=["trajectory_estimator_unavailable"],
                        confidence=0.0,
                        source="trajectory_unavailable",
                        ground_truth=False,
                        metric_scale="unknown",
                        provenance={"error": _safe_error(exc)},
                    )
                    trajectory_provenance = trajectory_provenance_report.provenance
                    errors.append(f"trajectory_estimator_error:{_safe_error(exc)}")
                target = sample.scene.object_by_id(sample.task.target_id) if sample.task.target_id else None
                target_position = target.position if target is not None else None
                if trajectory:
                    estimated_source = trajectory_provenance_report.source.startswith(
                        ("video_vo_estimate", "vggt_monocular_estimate")
                    )
                    filter_route = sample.route
                    filter_target_position = target_position
                    if (
                        not trajectory_provenance_report.ground_truth
                        and trajectory_provenance_report.metric_scale == "unknown"
                    ):
                        # A monocular trajectory lives in VGGT's arbitrary
                        # reconstruction frame.  Comparing it with scene
                        # metre bounds or a metre-valued goal would create
                        # false collision/endpoint failures.  Keep the raw
                        # points and explicitly report that metric geometry
                        # was deferred to review.
                        filter_route = replace(sample.route, obstacles=[])
                        filter_target_position = None
                    trajectory_report, trajectory = self.trajectory_filter.evaluate(
                        trajectory,
                        filter_route,
                        stop_condition=sample.task.stop_condition,
                        target_position=filter_target_position,
                        # A video-only estimate is already a review-tier
                        # artifact; do not let the filter's soft warnings
                        # become an accidental training pass.
                        allow_review=True,
                        repair_soft_failures=not estimated_source,
                        source=trajectory_provenance_report.source,
                    )
                    trajectory_report.ground_truth = trajectory_provenance_report.ground_truth
                    trajectory_report.metric_scale = trajectory_provenance_report.metric_scale
                    trajectory_report.estimator = trajectory_provenance_report.estimator
                    trajectory_report.provenance = trajectory_provenance
                    trajectory_report.metric_units = str(
                        trajectory_provenance_report.provenance.get(
                            "metric_units",
                            trajectory_provenance_report.metric_units
                            or ("m" if trajectory_provenance_report.metric_scale != "unknown" else "relative"),
                        )
                    )
                    trajectory_report.metrics_are_metric = bool(
                        trajectory_provenance_report.provenance.get(
                            "metrics_are_metric", trajectory_provenance_report.metrics_are_metric
                        )
                    )
                    trajectory_report.temporal_smoothing = bool(
                        trajectory_provenance_report.provenance.get(
                            "temporal_smoothing", trajectory_provenance_report.temporal_smoothing
                        )
                    )
                    trajectory_report.scale_anchor = str(
                        trajectory_provenance_report.provenance.get(
                            "scale_anchor", trajectory_provenance_report.scale_anchor
                        )
                    )
                    trajectory_report.variant = trajectory_provenance_report.variant or (
                        str(trajectory_provenance_report.provenance.get("variant"))
                        if trajectory_provenance_report.provenance.get("variant") is not None
                        else None
                    )
                    if filter_target_position is None and target_position is not None:
                        trajectory_report.provenance["metric_geometry_checked"] = False
                        trajectory_report.warnings.append("metric_geometry_deferred_unknown_scale")
                    trajectory_report.confidence = min(
                        trajectory_report.confidence, trajectory_provenance_report.confidence
                    )
                    trajectory_report.warnings = list(
                        dict.fromkeys(trajectory_provenance_report.warnings + trajectory_report.warnings)
                    )
                    if estimated_source and trajectory_report.hard_failures:
                        # Absolute speed/acceleration are not trustworthy when
                        # the camera calibration and monocular scale are only
                        # approximated.  Preserve the raw filter findings for
                        # audit, but downgrade those non-collision findings to
                        # human review instead of deleting the observed path.
                        geometry_failures = {"collision_or_insufficient_clearance"}
                        if not geometry_failures.intersection(trajectory_report.hard_failures):
                            trajectory_report.provenance["filter_hard_failures"] = list(trajectory_report.hard_failures)
                            trajectory_report.warnings.append("rgb_pose_estimate_kinematics_require_review")
                            trajectory_report.hard_failures = []
                            trajectory_report.decision = "review"
                    if estimated_source and trajectory_report.decision == "pass":
                        trajectory_report.decision = "review"
                    if trajectory_report.decision != "pass":
                        errors.extend(trajectory_report.hard_failures or trajectory_report.warnings or ["trajectory_not_pass"])
                else:
                    trajectory_report = trajectory_provenance_report
                    errors.extend(trajectory_report.warnings or ["trajectory_not_available"])
                sample.metadata["trajectory_provenance"] = trajectory_provenance
                planned_trajectory = self.trajectory_generator.generate(sample.route, confidence=1.0, source="route_contract")
                trajectory_plot = make_trajectory_plot(
                    sample.scene,
                    trajectory,
                    video_path.parent / "trajectory_plot.png",
                    target_id=sample.task.target_id,
                    obstacles=sample.route.obstacles,
                    # The workstation's primary visualization is the
                    # observed VGGT path only.  Keep the route contract in
                    # ``planned_trajectory.jsonl`` for audit/download, but
                    # do not draw it as a second line that could be mistaken
                    # for executed motion.
                    reference_trajectory=None,
                    trajectory_source=trajectory_report.source if trajectory_report is not None else "unknown",
                    relative_only=(
                        trajectory_report is not None
                        and not trajectory_report.ground_truth
                        and trajectory_report.metric_scale == "unknown"
                    ),
                )
        except Exception as exc:
            errors.append(f"pipeline_error:{_safe_error(exc)}")
            if not generation_meta and self.mode in {"live", "auto"} and self.responses is not None:
                # A create/download failure may happen before a task response
                # exists.  Keep a latency/error event without retaining the
                # request body or credential material.
                self._append_usage_event(
                    sample,
                    {
                        "kind": "video",
                        "stage": "seedance",
                        "model": self.config.video_model,
                        "status": "error",
                        "latency_s": round(time.monotonic() - generation_started, 6),
                        "requested_duration_s": float(sample.route.duration_s),
                        "actual_duration_s": None,
                        "provider_usage": {},
                        "provider_input_tokens": 0,
                        "provider_output_tokens": 0,
                        "provider_total_tokens": 0,
                        "cost_cny": None,
                        "cost_status": "unpriced",
                        "cost_basis": "task_failed",
                        "error_type": type(exc).__name__,
                    },
                )

        if pre_qc.get("decision") != "pass":
            status = "rejected_pre_qc"
        elif vlm_report is None or vlm_report.decision == "reject":
            status = "rejected_vlm"
        elif trajectory_report is None or trajectory_report.decision == "reject":
            status = "rejected_trajectory"
        elif vlm_report.decision == "review" or trajectory_report.decision == "review":
            status = "review"
        else:
            status = "accepted"
        usage_report = aggregate_usage(
            sample.metadata.get("usage_events", [])
            if isinstance(sample.metadata.get("usage_events", []), list)
            else []
        )
        return self.packager.pack(
            sample=sample,
            status=status,
            video_path=video_path,
            frames=frames,
            contact_sheet=contact_sheet,
            trajectory_plot=trajectory_plot,
            pre_qc=pre_qc,
            vlm_report=vlm_report,
            trajectory=trajectory,
            trajectory_report=trajectory_report,
            embedding_report=embedding_report,
            usage_report=usage_report,
            errors=errors,
            planned_trajectory=planned_trajectory,
        )

    def run_batch(
        self,
        count: int = 1,
        samples: Optional[Sequence[SampleSpec]] = None,
    ) -> List[SampleRecord]:
        """Run a batch, optionally starting from externally supplied samples.

        ``samples`` is used for source datasets such as RxR.  Supplying it
        bypasses only the generative planner; prompt compilation and every
        downstream fixed stage remain unchanged.  This makes the import path
        auditable without asking a planner model to rewrite a ground-truth
        instruction.
        """

        if samples is None:
            prepared_samples = self.plan_samples(count)
        else:
            prepared_samples = self.prepare_samples(samples)
            if not prepared_samples:
                raise ValueError("run_batch requires at least one supplied sample")
            count = len(prepared_samples)
        records = [self.run_one(sample) for sample in prepared_samples]
        counts = Counter(record.status for record in records)
        tier_counts = Counter(record.quality_tier for record in records)
        usage_events: List[Mapping[str, Any]] = []
        for record in records:
            if isinstance(record.usage_report, Mapping) and isinstance(record.usage_report.get("events"), list):
                usage_events.extend(item for item in record.usage_report["events"] if isinstance(item, Mapping))
        usage_summary = aggregate_usage(usage_events)
        summary = {
            "mode": self.mode,
            "pipeline_contract": dict(self.pipeline_contract),
            "config": self.config.safe_dict(mode=self.mode),
            "count": count,
            "status_counts": dict(counts),
            "quality_tier_counts": dict(tier_counts),
            "training_eligible_count": tier_counts.get("A", 0),
            "usage": usage_summary,
            "samples": [
                {
                    "sample_id": record.sample_id,
                    "status": record.status,
                    "quality_tier": record.quality_tier,
                    "video_path": record.video_path,
                    "trajectory_path": record.trajectory_path,
                    "trajectory_data_path": record.trajectory_data_path,
                    "planned_trajectory_path": record.planned_trajectory_path,
                    "trajectory_plot_path": record.trajectory_plot_path,
                    "usage": {
                        key: record.usage_report.get(key)
                        for key in (
                            "request_count",
                            "input_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                            "total_tokens",
                            "video_provider_total_tokens",
                            "video_provider_input_tokens",
                            "video_provider_output_tokens",
                            "latency_s",
                            "video_requested_duration_s",
                            "video_actual_duration_s",
                            "estimated_cost_cny",
                            "billable_cost_cny",
                            "cost_status",
                        )
                    }
                    if isinstance(record.usage_report, Mapping)
                    else None,
                    "errors": record.errors,
                }
                for record in records
            ],
        }
        if self._planner_fallback_error:
            summary["planner_fallback_error"] = self._planner_fallback_error
        if self._prompt_fallback_errors:
            summary["prompt_fallback_errors"] = list(self._prompt_fallback_errors)
        (self.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "usage_summary.json").write_text(
            json.dumps(usage_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(dataclass_to_dict(record), ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return records
