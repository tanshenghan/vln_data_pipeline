"""Render an EVT-Bench native tracking episode; NOT a BetaNav GT export.

Run in the separate TrackVLA/Habitat 0.3.1 + Bullet environment. Shared
scene/robot/human assets are read-only inputs; native navmesh caches live in
a new private output directory. No inference checkpoint is needed: the
controller is the upstream GT-bounding-box baseline, not TrackVLA or LightNav
and not an expert trajectory suitable for imitation training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_relative(scene_id):
    value = str(scene_id).removeprefix("data/scene_datasets/")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe scene ID: {scene_id}")
    if path.parts[0] not in {"mp3d", "hm3d"} or path.suffix != ".glb":
        raise ValueError(f"Unsupported scene ID: {scene_id}")
    return path


def select_episode(source, scenes_root, scene_family, episode_id=None):
    with gzip.open(source, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    for episode in episodes:
        if episode_id is not None and str(episode["episode_id"]) != episode_id:
            continue
        relative = scene_relative(episode["scene_id"])
        if relative.parts[0] == scene_family and (scenes_root / relative).is_file():
            return episode, relative
    raise ValueError(f"No readable {scene_family} scene for the requested episode in {source}")


def prepare_workspace(output, upstream, humanoids, scenes_root, relative):
    # Never reuse an output/cache from an earlier probe.
    output.mkdir(parents=True, exist_ok=False)
    work = output / "workspace"
    data = work / "data"
    data.mkdir(parents=True)
    (data / "robots").symlink_to(upstream / "data/robots", target_is_directory=True)
    (data / "humanoids").symlink_to(humanoids, target_is_directory=True)
    (data / "default.physics_config.json").symlink_to(upstream / "data/default.physics_config.json")
    (work / "humanoid_infos.json").symlink_to(upstream / "humanoid_infos.json")
    scene = data / "scene_datasets" / relative
    scene.parent.mkdir(parents=True)
    scene.symlink_to(scenes_root / relative)
    # In particular, scene_datasets itself must not be a symlink: the upstream
    # simulator creates scene_datasets/navmeshes/<scene>.navmesh when absent.
    return work


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def probe(args):
    upstream = args.upstream.resolve(strict=True)
    scenes = args.scenes_root.resolve(strict=True)
    humans = args.humanoids.resolve(strict=True)
    output = args.output.absolute()
    source = upstream / f"data/datasets/track/{args.mode}/{args.split}/{args.split}.json.gz"
    episode, relative = select_episode(source, scenes, args.scene_family, args.episode_id)
    if episode["info"]["episode_mode"] != args.mode.lower():
        raise ValueError("Episode mode does not match the selected source")
    work = prepare_workspace(output, upstream, humans, scenes, relative)
    write_json(output / "source_episode.json", episode)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "schema": "evt-native-probe-v1", "status": "running", "native_probe_passed": False,
        "betanav_gt_complete": False, "web_collection_accepted": False,
        "controller": "upstream GTBBoxAgent (privileged ground-truth bounding box baseline; not expert GT)",
        "task": "dynamic_human_following", "dataset": "EVT-Bench", "mode": args.mode,
        "split": args.split, "episode_id": str(episode["episode_id"]),
        "instruction": episode["info"]["instruction"], "source_scene_id": episode["scene_id"],
        "target_avatar": episode["info"]["main_humanoid_name"],
        "target_semantic_id": episode["info"]["main_human_semantic_id"],
        "upstream": {"url": "https://github.com/ShaoanWang/TrackVLA", "commit": commit},
        "source": {"path": str(source), "sha256": sha256(source)},
        "scene": {"path": str(scenes / relative), "sha256": sha256(scenes / relative)},
        "max_steps": args.steps, "output": str(output),
        "sampling": "initial observation, then one observation per native env.step",
        "pose_convention": "Habitat world: Y up, metres; matrices world_from_agent_base",
        "time_convention": "recorded simulator world time, not video FPS",
        "excluded": ["depth", "BetaNav FM/RVQ/apos/opos", "terminal STOP labels", "LeRobot export"],
    }
    write_json(output / "report.json", report)
    old_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(upstream))
        import imageio.v2 as imageio
        import numpy as np
        import habitat
        import habitat_sim
        from habitat.config import read_write
        from habitat.datasets import make_dataset
        from evt_bench.default import get_config
        from baseline_agent import GTBBoxAgent
        from omegaconf import OmegaConf

        if not habitat_sim.built_with_bullet:
            raise RuntimeError("EVT-Bench requires Habitat-Sim built with Bullet")
        os.chdir(work)
        config = get_config(str(upstream / f"habitat-lab/habitat/config/benchmark/nav/track/track_infer_{args.mode.lower()}.yaml"))
        with read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
            config.habitat.environment.iterator_options.shuffle = False
            config.habitat.environment.iterator_options.cycle = False
        # Keep the native inference dynamics. Both train and val smoke probes
        # use original episodes, but neither is a full benchmark evaluation.
        write_json(output / "native_config.json", OmegaConf.to_container(config, resolve=True))
        random.seed(config.habitat.simulator.seed)
        np.random.seed(config.habitat.simulator.seed)
        dataset = make_dataset(id_dataset="PointNav-v1")
        dataset.from_json(json.dumps({"episodes": [episode]}), scenes_dir="data/scene_datasets")
        controller = GTBBoxAgent(str(output / "baseline"))
        action_names = ["agent_0_humanoid_navigate_action", "agent_1_base_velocity"]
        action_names += [f"agent_{idx}_oracle_nav_randcoord_action_obstacle" for idx in range(2, 9)]
        records = []
        for subdir in ("rgb", "panoptic"):
            (output / subdir).mkdir()
        with habitat.TrackEnv(config=config, dataset=dataset) as env:
            env.seed(config.habitat.simulator.seed)
            env.reset()
            video = imageio.get_writer(str(output / "review.mp4"), fps=10, codec="libx264")
            try:
                for step in range(args.steps + 1):
                    if step:
                        before = env.sim.get_sensor_observations()
                        detector = env.task._get_observations(env.current_episode)
                        velocity = controller.act(before, detector, env.current_episode.episode_id)
                        controller.rgb_list.clear()  # This probe streams its own review video.
                        env.step({"action": tuple(action_names), "action_args": {"agent_1_base_vel": velocity}})
                    obs = env.sim.get_sensor_observations()
                    rgb = np.asarray(obs["agent_1_articulated_agent_jaw_rgb"])[..., :3]
                    panoptic = np.asarray(obs["agent_1_articulated_agent_jaw_panoptic"])
                    robot = env.sim.agents_mgr[1].articulated_agent
                    target = env.sim.agents_mgr[0].articulated_agent
                    metrics = env.get_metrics()
                    record = {
                        "frame_index": step, "sim_time_s": float(env.sim.get_world_time()),
                        "world_from_robot_base": np.asarray(robot.base_transformation).tolist(),
                        "world_from_target_base": np.asarray(target.base_transformation).tolist(),
                        "robot_base_position_m": np.asarray(robot.base_pos).tolist(),
                        "target_base_position_m": np.asarray(target.base_pos).tolist(),
                        "target_pixels": int(np.count_nonzero(panoptic == report["target_semantic_id"])),
                        "metrics": {key: float(value) for key, value in metrics.items()
                                    if isinstance(value, (int, float, np.number, bool))},
                    }
                    records.append(record)
                    imageio.imwrite(output / "rgb" / f"{step:06d}.png", rgb)
                    np.save(output / "panoptic" / f"{step:06d}.npy", panoptic, allow_pickle=False)
                    video.append_data(rgb)
                    if env.episode_over:
                        break
            finally:
                video.close()
            report["termination"] = "native_episode_over" if env.episode_over else "probe_step_limit"
            report["native_episode_over"] = bool(env.episode_over)
        write_json(output / "frames.json", records)
        # Re-open the written media and poses, not just the in-memory objects.
        saved = json.loads((output / "frames.json").read_text())
        robot_pos = np.array([r["robot_base_position_m"] for r in saved])
        target_pos = np.array([r["target_base_position_m"] for r in saved])
        robot_motion = float(np.linalg.norm(np.diff(robot_pos, axis=0), axis=1).sum())
        target_motion = float(np.linalg.norm(np.diff(target_pos, axis=0), axis=1).sum())
        for row in saved:
            index = row["frame_index"]
            rgb = imageio.imread(output / "rgb" / f"{index:06d}.png")
            panoptic = np.load(output / "panoptic" / f"{index:06d}.npy", allow_pickle=False)
            if rgb.shape != (480, 640, 3) or panoptic.shape != rgb.shape[:2]:
                raise ValueError("Native RGB/panoptic shapes do not match")
            for key in ("world_from_robot_base", "world_from_target_base"):
                pose = np.array(row[key])
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError("Invalid native pose")
            if int(np.count_nonzero(panoptic == report["target_semantic_id"])) != row["target_pixels"]:
                raise ValueError("Saved target mask does not match the target semantic ID")
        if robot_motion <= 0.01 or target_motion <= 0.01 or not any(r["target_pixels"] for r in saved):
            raise ValueError("Probe did not observe moving robot, moving target and visible target")
        report.update(status="succeeded", native_probe_passed=True, frame_count=len(saved),
                      robot_path_m=robot_motion, target_path_m=target_motion,
                      frames_with_visible_target=sum(r["target_pixels"] > 0 for r in saved),
                      habitat_sim_version=habitat_sim.__version__)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.chdir(old_cwd)
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--humanoids", type=Path, required=True, help="Directory containing humanoid_data/")
    parser.add_argument("--output", type=Path, required=True, help="New personal directory; never reused")
    parser.add_argument("--mode", choices=["STT", "DT", "AT"], default="STT")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--scene-family", choices=["mp3d", "hm3d"], default="mp3d")
    parser.add_argument("--episode-id")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 300:
        parser.error("--steps must be between 1 and 300")
    print(json.dumps(probe(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
