"""Pose replay planning and explicit conversion from USD Z-up to BetaNav axes."""
import math
import numpy as np

SOURCE_TO_BETA=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])


def body_pose_source(position,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    pose=np.eye(4)
    pose[:3,:3]=[[s,0,-c],[-c,0,-s],[0,1,0]]
    pose[:3,3]=position
    return pose


def canonical_pose(pose):
    transform=np.eye(4);transform[:3,:3]=SOURCE_TO_BETA
    return transform@pose


def replay_plan(record,turn_step_deg=15):
    raw=record['source_record']
    path=np.asarray(raw.get('reference_path') or [],dtype=float)
    goals=raw.get('goals') or []
    if isinstance(goals,dict):goals=[goals]
    if path.ndim!=2 or path.shape[1:]!=(3,) or len(path)<2 or not goals or not np.isfinite(path).all():
        raise ValueError('VLNVerse episode lacks a public reference trajectory/goal')
    if not np.allclose(path[0],raw['start_position'],atol=.05):
        raise ValueError('reference path does not start at the source start position')
    # VLNVerse/Isaac start_rotation is scalar-first wxyz. Do not interpret it as Habitat xyzw.
    w,x,y,z=map(float,raw['start_rotation'])
    if not np.isclose(w*w+x*x+y*y+z*z,1,atol=1e-5) or abs(x)>1e-6 or abs(y)>1e-6:
        raise ValueError('expected a normalized upright source quaternion in wxyz order')
    yaw=math.atan2(2*w*z,1-2*z*z)
    poses=[body_pose_source(path[0],yaw)]
    commands=['start']
    position=path[0]
    for target in path[1:]:
        delta=target-position
        if np.linalg.norm(delta)<1e-8:continue
        if abs(delta[2])>.05:
            raise ValueError('non-planar reference motion requires a separate motion profile')
        desired=math.atan2(delta[1],delta[0])
        turn=(desired-yaw+math.pi)%(2*math.pi)-math.pi
        steps=max(1,math.ceil(abs(turn)/math.radians(turn_step_deg)))
        if abs(turn)>1e-8:
            for i in range(1,steps+1):
                poses.append(body_pose_source(position,yaw+turn*i/steps));commands.append('rotate')
        yaw+=turn
        poses.append(body_pose_source(target,yaw));commands.append('reference_waypoint')
        position=target
    distance=min(float(np.linalg.norm(position-np.asarray(g['position']))) for g in goals)
    if distance>.05:raise ValueError('reference endpoint is not the supplied goal')
    # Explicit terminal command, rather than treating end of input as an implicit STOP.
    commands[-1]='stop_at_verified_goal'
    if len(poses)>3000:raise ValueError('episode exceeds collection frame limit')
    return {'poses_source':np.asarray(poses),'commands':commands,'goal_distance_m':distance,
            'sampling':'source reference waypoints with <=15 degree in-place heading transitions; no dynamic robot rollout'}
