#!/usr/bin/env python3
"""Capture USD reference-waypoint replay with Isaac; annotate in the export env.

The camera follows public positions and explicitly planned heading transitions.
This is a kinematic replay, not an original action trace or dynamic robot rollout.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def annotate(output):
    from data_pipeline.collect_r2r import export_targets
    from data_pipeline.pointing import PointingConfig
    from lightnav.traj_vocab import load_rvq_bundle
    manifest = json.loads((output / 'manifest.json').read_text())
    bundle = load_rvq_bundle(manifest['rvq_bundle'], horizon=10)
    results = []
    for folder in sorted((output / 'episodes').iterdir()):
        metadata = json.loads((folder / 'episode.json').read_text())
        frames = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        if metadata['terminal_command'] not in {'stop_at_verified_goal', 'model_stop'}:
            raise ValueError('terminal STOP was not verified')
        if metadata.get('trajectory_source') == 'lightnav_inference' and not metadata.get('vlm_review', {}).get('accepted'):
            raise ValueError('LightNav candidate has not passed semantic review')
        result = export_targets(folder, frames, bundle, True, metadata['camera'], PointingConfig())
        metadata.update(result, complete=True, opos_status='terminal_ground_complete')
        write_json(folder / 'episode.json', metadata)
        results.append(metadata)
    write_json(output / 'summary.json', dict(complete=True, episodes=len(results),
        frames=sum(r['frames'] for r in results), camera=manifest['camera'], output_dir=str(output),
        full_rvq_10_chunks=sum(r['full_rvq_10_chunks'] for r in results),
        full_fm_30_chunks=sum(r['full_fm_30_chunks'] for r in results)))


def collect(config, output, scene_roots, bundle_path):
    # Import Kit before any pxr/omni module. Keep it isolated from LeRobot/Habitat.
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting',
        'active_gpu': int(os.environ.get('BETA_GPU_ID', '0')), 'multi_gpu': False,
        'width': 470, 'height': 280, 'anti_aliasing': 0})
    try:
        import hashlib
        import math
        import numpy as np
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux, Sdf
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from catalog.vlnverse_scenes import usd_path
        from data_pipeline.geometry import pinhole_intrinsics
        from data_pipeline.pointing import PointingConfig
        from data_pipeline.vlnverse_source import replay_plan, canonical_pose, SOURCE_TO_BETA, body_pose_source
        from data_pipeline.vlnverse_lightnav import initial_state, LightNavSession, StaticCollision, command, PROFILE

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        camera = dict(width=470, height=280, hfov_deg=120., position_body_native=[0., .88, 0.],
            orientation_euler=[0., 0., 0.], K=pinhole_intrinsics(470, 280, 120.).tolist(),
            min_depth_m=0., max_depth_m=10., depth_unit='metre', depth_type='camera_z',
            depth_normalized=False, depth_clipped_to_range=True, invalid_depth='use each depth NPZ valid mask')
        environment_hdr = Path(os.environ.get('BETA_VLNVERSE_HDR', ''))
        records = config['source_records']
        # Validate the entire frozen selection before writing any output.
        plans = [replay_plan(r) if r['has_reference_gt'] else None for r in records]
        max_steps = int(config.get('lightnav_max_steps', 500))
        paths = [usd_path(scene_roots, r['scene']) for r in records]
        for record in records:
            eid = record['episode_id']
            if not eid or Path(eid).name != eid or eid in {'.', '..'}:
                raise ValueError('unsafe episode ID')
        output.mkdir(parents=True, exist_ok=False)
        import shutil
        snapshots=output/'source_code';snapshots.mkdir()
        for filename in ('collect_vlnverse.py','vlnverse_source.py','vlnverse_lightnav.py'):
            shutil.copyfile(ROOT/'data_pipeline'/filename,snapshots/filename)
        manifest = dict(format='beta_nav_v2', created_utc=datetime.now(timezone.utc).isoformat(),
            collection='VLNVerse automatic public reference replay or LightNav inference in native USD',
            robot_type='vlnverse_isaac_auto', renderer='Isaac Sim 5.1 RTX native USD',
            source_selection={'dataset': 'vlnverse', 'split': config['split'], 'role': config['role'],
                              'source_ids': config['source_ids'], 'source_records': records},
            split=config['split'], episodes=len(records), episode_ids=[r['episode_id'] for r in records],
            source_sha256={r['source_path']: r['source_file_sha256'] for r in records},
            scene_entry_sha256={str(p): digest(p) for p in paths},
            scene_assets={record['scene']: {
                'repo': 'Eyz/VLNVerse_scene',
                'revision': json.loads((scene.parent/'platform_asset_manifest.json').read_text())['revision'],
                'file_manifest_sha256': digest(scene.parent/'platform_asset_manifest.json')}
                for record, scene in zip(records, paths)},
            collector_sha256=digest(snapshots/'collect_vlnverse.py'), lightnav_adapter_sha256=digest(snapshots/'vlnverse_lightnav.py'), source_adapter_sha256=digest(ROOT/'data_pipeline/vlnverse_source.py'),
            geometry_code_sha256=digest(ROOT/'data_pipeline/geometry.py'),
            pointing_code_sha256=digest(ROOT/'data_pipeline/pointing.py'),
            camera=camera, python=sys.executable, world_axes='BetaNav +Y up; body +X right, +Y up, -Z forward',
            camera_axes='OpenCV +X right, +Y down, +Z forward',
            source_coordinate_frame='USD Z-up metres, start_rotation wxyz, initial forward +X',
            source_to_canonical=SOURCE_TO_BETA.tolist(),
            sampling_mode='vlnverse_auto_v1', sampling_definition='per-episode reference replay or closed-loop LightNav',
            lightnav_profile=PROFILE, lightnav_endpoint=config.get('lightnav_endpoint'), lightnav_max_steps=max_steps,
            lightnav_model_provenance=config.get('lightnav_model_provenance'),
            physical_dt_s=None, collision_policy='not simulated; public reference waypoints replayed kinematically',
            rvq_horizon=10, fm_horizon=30, fm_representation='se2_diff', fm_normalization='none',
            rvq_bundle=str(bundle_path), rvq_sha256={p.name: digest(p) for p in bundle_path.iterdir() if p.is_file()},
            pointing_config=PointingConfig().metadata(), discrete_action_labels=False,
            terminal_policy='reference goal verified within .05m or explicit model STOP; semantic VLM review stored separately')
        manifest['scene_overrides'] = []
        write_json(output/'manifest.json', manifest)
        carb.settings.get_settings().set('/rtx/post/dlss/execMode', 2)
        rep.orchestrator.set_capture_on_play(False)
        context = omni.usd.get_context()
        current_scene = None
        for index, (record, plan, scene) in enumerate(zip(records, plans, paths)):
            if current_scene != scene:
                if current_scene is not None:
                    rgb_annot.detach([product]); depth_annot.detach([product]); product.destroy()
                if not context.open_stage(str(scene)):
                    raise RuntimeError('cannot open USD scene: '+str(scene))
                for _ in range(10): app.update()
                for _ in range(1200):
                    if context.get_stage_loading_status()[2] == 0: break
                    app.update()
                else: raise RuntimeError('USD scene dependencies did not finish loading')
                stage = context.get_stage()
                if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
                    raise ValueError('expected metre Z-up USD scene')
                # Author the temporary rig only in the anonymous session layer.
                stage.SetEditTarget(stage.GetSessionLayer())
                for prim in stage.Traverse():
                    if prim.IsA(UsdLux.DomeLight):
                        attr = UsdLux.DomeLight(prim).GetTextureFileAttr()
                        asset = attr.Get()
                        if asset and not asset.resolvedPath and Path(asset.path).name == 'limpopo_golf_course_4k.hdr':
                            if not environment_hdr.is_file():
                                raise FileNotFoundError('scene references missing limpopo HDR; configure BETA_VLNVERSE_HDR')
                            attr.Set(Sdf.AssetPath(str(environment_hdr.resolve())))
                            manifest['scene_overrides'].append({'scene':record['scene'], 'prim':str(prim.GetPath()),
                                'original_asset':asset.path, 'resolved_asset':str(environment_hdr.resolve()),
                                'sha256':digest(environment_hdr), 'reason':'resolve omitted named HDR dependency'})
                write_json(output/'manifest.json', manifest)
                rig = UsdGeom.Xform.Define(stage, '/VLNPlatformCapture')
                transform = rig.MakeMatrixXform()
                cam = UsdGeom.Camera.Define(stage, '/VLNPlatformCapture/Camera')
                cam.AddTranslateOp().Set(Gf.Vec3d(0, .88, 0))
                aperture = 20.955
                cam.GetHorizontalApertureAttr().Set(aperture)
                cam.GetVerticalApertureAttr().Set(aperture*280/470)
                cam.GetFocalLengthAttr().Set(aperture/(2*math.tan(math.radians(60))))
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(.001, 1000.))
                product = rep.create.render_product(str(cam.GetPath()), (470, 280))
                rgb_annot = rep.AnnotatorRegistry.get_annotator('rgb')
                depth_annot = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                rgb_annot.attach([product]); depth_annot.attach([product])
                current_scene = scene
                collision = None
            if plan is None and collision is None:
                collision = StaticCollision(stage)
            folder = output/'episodes'/record['episode_id']
            (folder/'rgb').mkdir(parents=True); (folder/'depth').mkdir()
            frames = []
            client = None; last_collision = None; termination = None; stagnant = 0
            if plan is None:
                position, yaw = initial_state(record)
                if not collision.floor_ok(position):raise ValueError('start pose lacks verified floor support')
                client = LightNavSession(config.get('lightnav_endpoint','ws://127.0.0.1:8051'), 'vln-pipeline-'+record['episode_id'])
                desired = body_pose_source(position,yaw)
            total_frames = len(plan['poses_source']) if plan else max_steps + 1
            with (folder/'frames.jsonl').open('w') as stream:
                for t in range(total_frames):
                    if plan:desired=plan['poses_source'][t]
                    transform.Set(Gf.Matrix4d(desired.T.tolist()))
                    rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)
                    rgb = np.asarray(rgb_annot.get_data())[:, :, :3].copy()
                    depth = np.asarray(depth_annot.get_data(), dtype=np.float32).squeeze().copy()
                    if rgb.shape != (280, 470, 3) or rgb.dtype != np.uint8 or depth.shape != (280, 470):
                        raise ValueError('unexpected rendered sensor shape/type')
                    valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
                    depth = np.clip(np.nan_to_num(depth, nan=0., posinf=10., neginf=0.), 0., 10.)
                    if t == 0 and (float(valid.mean()) < .01 or np.std(rgb) < 1.):
                        raise RuntimeError('rendered observation is empty; scene is not ready')
                    actual = np.array(rig.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    actual_cam = np.array(cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    np.testing.assert_allclose(actual, desired, atol=1e-6)
                    body = canonical_pose(actual)
                    cv = canonical_pose(actual_cam) @ np.diag([1., -1., -1., 1.])
                    np.testing.assert_allclose((np.linalg.inv(body)@cv)[:3,3], [0,.88,0], atol=1e-6)
                    rgb_name, depth_name = f'rgb/{t:06d}.png', f'depth/{t:06d}.npz'
                    Image.fromarray(rgb).save(folder/rgb_name)
                    np.savez_compressed(folder/depth_name, depth_m=depth, valid=valid)
                    frame = dict(frame_id=t, rgb=rgb_name, depth=depth_name,
                        world_from_body=body.tolist(), world_from_camera_cv=cv.tolist(),
                        world_from_depth_camera_cv=cv.tolist(), collision_from_previous_action=last_collision,
                        valid_depth_fraction=float(valid.mean()))
                    stream.write(json.dumps(frame)+'\n'); stream.flush(); frames.append(frame)
                    print(json.dumps({'episode_id':record['episode_id'], 'frames':t+1, 'total_frames':total_frames}), flush=True)
                    if client:
                        response = client.predict(folder/rgb_name, record['instruction'], t)
                        with (folder/'model_responses.jsonl').open('a') as audit:
                            audit.write(json.dumps({'frame_id':t, **response})+'\n')
                        if response['stop']:
                            termination = 'model_stop';break
                        if t == max_steps:
                            termination = 'step_limit';break
                        forward, turn = command(response)
                        previous=position.copy();old_yaw=yaw
                        position,yaw,last_collision=collision.advance(position,yaw,forward,turn)
                        stagnant = stagnant+1 if np.linalg.norm(position-previous)<1e-4 and abs(yaw-old_yaw)<1e-4 else 0
                        if stagnant >= 20:
                            termination='stalled';break
                        desired=body_pose_source(position,yaw)
            if client:client.close()
            metadata = dict(episode_id=record['episode_id'], trajectory_id=record['trajectory_id'],
                scene_id=record['scene'], instruction=record['instruction'], camera=camera,
                frames=len(frames), pointing_config=PointingConfig().metadata(), complete=False,
                goal_distance_m=plan['goal_distance_m'] if plan else None,
                terminal_command=plan['commands'][-1] if plan else termination,
                trajectory_source='public_reference' if plan else 'lightnav_inference',
                collision_policy=manifest['collision_policy'] if plan else PROFILE,
                collision_frames=sum(bool(f['collision_from_previous_action']) for f in frames),
                source_reference_path=record['reference_path'], goal_position=record['goals'][0] if record['goals'] else None,
                source_metadata=record, sampling=plan['sampling'] if plan else PROFILE)
            write_json(folder/'episode.json', metadata)
            print(json.dumps({'completed':index+1, 'total':len(records), 'episode_id':record['episode_id'], 'frames':len(frames)}), flush=True)
        write_json(output/'render_complete.json', {'complete':True, 'episodes':len(records)})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        if output.is_dir():write_json(output/'failure.json', {'error':repr(exc)})
        raise
    finally:
        app.close(wait_for_replicator=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scene-root', action='append', type=Path, default=[])
    parser.add_argument('--rvq-bundle', type=Path)
    parser.add_argument('--annotate', action='store_true')
    args = parser.parse_args()
    if args.annotate:
        if not (args.output/'render_complete.json').is_file():
            raise ValueError('render stage did not complete')
        annotate(args.output)
    else:
        collect(json.loads(args.config.read_text()), args.output, args.scene_root, args.rvq_bundle)


if __name__ == '__main__':
    main()
