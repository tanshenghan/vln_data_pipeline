"""Preserve NavGen production stages with durable provider submission identity."""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from navgen_pipeline.config import PipelineConfig, _load_dotenv
from navgen_pipeline.ark_client import ArkHTTPError
from navgen_pipeline.models import ObjectSpec, RouteContract, SampleSpec, SampleRecord, SceneManifest, StopCondition, TaskSpec, Waypoint
from navgen_pipeline.pipeline import NavigationDataPipeline
from navgen_pipeline.rxr import _localize_reference_path
from .command_runner import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def restore_sample(data):
    data = dict(data)
    scene = dict(data.pop("scene")); scene["objects"] = [ObjectSpec(**o) for o in scene["objects"]]
    task = dict(data.pop("task")); task["stop_condition"] = StopCondition(**task["stop_condition"])
    route = dict(data.pop("route")); route["waypoints"] = [Waypoint(**w) for w in route["waypoints"]]
    return SampleSpec(scene=SceneManifest(**scene), task=TaskSpec(**task), route=RouteContract(**route), **data)


def source_sample(record, duration):
    """Generic source adapter; no hard-coded demo scene, object or instruction."""
    path = record.get("reference_path")
    if not path or len(path) < 2:
        raise ValueError("源样本没有可用参考路线")
    points, local = _localize_reference_path(path, scale=0.3, anchor=(0.8, 0.8), duration_s=duration)
    dx, dy = max(0, 0.5-min(p.x for p in points)), max(0, 0.5-min(p.y for p in points))
    for p in points:
        p.x += dx; p.y += dy
    provenance = {"source_id":record["source_id"], "source_episode_id":record["episode_id"],
                  "source_scene_id":record["scene"], "source_instruction_id":record.get("instruction_id"),
                  "source_reference_path":path, "source_split":record["split"], "source_role":record["role"],
                  "source_language":record["language"], "ground_truth":False,
                  "coordinate_transform":{"scale_prior":0.3,"x_axis":"negative source X", "y_axis":"source Z", "offset":[0.8+dx,0.8+dy]},
                  "geometry_source":"source_reference_path_only", "source_dataset":record["dataset"]}
    scene = SceneManifest(scene_id=record["scene"], room_type="indoor_semantics_unconfirmed",
                          width_m=max(8,max(p.x for p in points)+1),depth_m=max(6,max(p.y for p in points)+1),metadata=provenance)
    task = TaskSpec(type="instruction_following",target_id=None,instruction_zh=record["instruction"],
                    instruction_original=record["instruction"],start_pose=(points[0].x,points[0].y,points[0].yaw),
                    stop_condition=StopCondition(dwell_s=1.2))
    route = RouteContract(waypoints=points,expected_duration_s=duration,goal_region={"center":[points[-1].x,points[-1].y],"source":"route_prior"})
    seed = int(hashlib.sha256(record["id"].encode()).hexdigest()[:8],16) % (2**31)
    return SampleSpec(sample_id="source_"+record["id"],scene=scene,task=task,route=route,seedance_prompt="等待编译",
                      seed=seed,planner_source="source_episode_import",metadata=provenance)


class DurableProvider:
    def __init__(self, delegate, directory):
        self.delegate, self.directory = delegate, directory
        directory.mkdir(exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def wait_for_task(self, task_id, poll_s=5.0, timeout_s=900.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                result = self.delegate.get_task(task_id)
                status = str(result.get("status", "")).lower()
                print(json.dumps({"stage":"seedance_"+status,"task_id":task_id}),flush=True)
                if status in {"succeeded","success","completed","failed","expired","cancelled"}:
                    return result
            except ArkHTTPError as exc:
                if exc.status not in {0,408,429,500,502,503,504}:
                    raise
                print(json.dumps({"stage":"seedance_connection_retry","task_id":task_id}),flush=True)
            time.sleep(max(0.2,min(poll_s,5.0)))
        raise TimeoutError("供应商查询超时；任务 ID 已保存，续跑将查询原任务")

    def create_task(self, **kwargs):
        key = hashlib.sha256(json.dumps(kwargs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        journal = self.directory / (key + ".json")
        if journal.exists():
            saved = json.loads(journal.read_text())
            if saved.get("task_id"):
                print(json.dumps({"stage":"seedance_resume","task_id":saved["task_id"]}),flush=True)
                return {"id":saved["task_id"]}
            raise RuntimeError("上次提交结果未知；禁止自动重提，请核对供应商任务后处理")
        # Persist intent before the network request. If the response is lost, never charge again blindly.
        fd = os.open(journal, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,"w") as f:
            json.dump({"state":"submitting","request_sha256":key,"created":time.time()},f)
            f.flush();os.fsync(f.fileno())
        response = self.delegate.create_task(**kwargs)
        atomic_json(journal,{"state":"submitted","task_id":str(response["id"]),"request_sha256":key})
        print(json.dumps({"stage":"seedance_submitted","task_id":response["id"]}),flush=True)
        return response


class InstructionClient:
    def __init__(self, client, instruction):
        self.client, self.instruction = client, instruction

    def generate_text(self, **kwargs):
        kwargs["user_text"] += "\n用户要求的导航任务：" + self.instruction + "\n请围绕该任务规划场景与路线。"
        return self.client.generate_text(**kwargs)


class PersistentPipeline(NavigationDataPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_file = self.output_dir / "prepared_samples.json"
        self.seedance = DurableProvider(self.seedance, self.output_dir / "provider_tasks")

    def plan_samples(self, count):
        if self.plan_file.exists():
            return [restore_sample(s) for s in json.loads(self.plan_file.read_text())]
        return super().plan_samples(count)

    def run_one(self, sample):
        manifest = self.output_dir / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                saved = json.loads(line)
                if saved["sample_id"] != sample.sample_id or execution_errors(saved.get("errors", [])):
                    continue
                previous = saved["sample_spec"]
                current = asdict(sample)
                fields = ("scene", "task", "route", "seed", "seedance_prompt")
                if any(json.dumps(previous[k],sort_keys=True) != json.dumps(current[k],sort_keys=True) for k in fields):
                    raise RuntimeError("已生成样本的定义发生变化，请创建新任务")
                assets = [saved.get(k) for k in ("video_path", "trajectory_path", "trajectory_data_path") if saved.get(k)]
                if assets and all(Path(p).is_file() and Path(p).stat().st_size for p in assets):
                    print(json.dumps({"stage":"sample_reused","sample_id":sample.sample_id}),flush=True)
                    return SampleRecord(**saved)
        return super().run_one(sample)

    def prepare_samples(self, samples):
        if self.plan_file.exists():
            return [restore_sample(s) for s in json.loads(self.plan_file.read_text())]
        prepared = super().prepare_samples(samples)
        atomic_json(self.plan_file,[asdict(s) for s in prepared])
        return prepared

    def _generate_video(self, sample):
        print(json.dumps({"stage":"seedance_video","sample_id":sample.sample_id}),flush=True)
        cache = self.work_dir / sample.sample_id / "generation_cache.json"
        video = self.work_dir / sample.sample_id / "generated.mp4"
        if cache.exists() and video.exists():
            return video,json.loads(cache.read_text())
        result,meta = super()._generate_video(sample)
        atomic_json(cache,meta)
        return result,meta

    def _build_trajectory(self, *args, **kwargs):
        print(json.dumps({"stage":"vggt_trajectory"}),flush=True)
        with (ROOT / "runtime/gpu.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return super()._build_trajectory(*args,**kwargs)

    def _vlm_report(self, *args, **kwargs):
        print(json.dumps({"stage":"vlm_filter"}),flush=True)
        return super()._vlm_report(*args,**kwargs)


def execution_errors(errors):
    # Upstream errors also contains quality rejections and non-GT limitations.
    # They remain review findings and do not mean the production process crashed.
    return [e for e in errors if str(e).startswith("pipeline_error:")]


def main():
    cfg_path=Path(os.environ["VLN_RUN_CONFIG"])
    request=json.loads(cfg_path.read_text())
    _load_dotenv(Path(os.environ.get("VLN_NAVGEN_ENV",str(ROOT.parent / ".env"))))
    config=PipelineConfig.from_env()
    if not config.api_key:
        raise RuntimeError("Seedance API 密钥未配置")
    config.duration_s=request["duration"]
    config.vggt_repo=os.environ.get("VLN_VGGT_REPO",str(Path.home()/".cache/navgen/vggt"))
    config.pipeline_profile="fixed"
    pipeline=PersistentPipeline(config,output_dir=cfg_path.parent/"generation",mode="live")
    records=request.get("source_records",[])
    if request.get("instruction") and not records:
        pipeline.remote_planner.client=InstructionClient(pipeline.remote_planner.client,request["instruction"])
    samples=[source_sample(r,request["duration"]) for r in records] if records else None
    results=pipeline.run_batch(count=request["count"],samples=samples)
    failures=[(r.sample_id,execution_errors(r.errors)) for r in results if execution_errors(r.errors)]
    if failures:
        raise RuntimeError("生成流程存在执行错误，原始产物已保留："+json.dumps(failures,ensure_ascii=False))


if __name__ == "__main__":
    main()
