"""Native USD kinematic rollout: isolated LightNav session and static mesh collision queries."""
import base64
import json
import math
import numpy as np
from data_pipeline.vlnverse_source import body_pose_source

PROFILE = {'name': 'lightnav_unicycle_static_usd_v1', 'dt_s': .1,
           'linear_mps': [0., 2.5], 'angular_dps': [-300., 300.],
           'body_radius_m': .18, 'body_height_m': .90, 'substep_m': .04,
           'collision': 'static USD triangle mesh sphere sweeps; no sliding; floor support ray',
           'physics': 'kinematic controller, not a dynamic robot; lateral prediction discarded'}


def initial_state(record):
    raw = record['source_record']
    position = np.asarray(raw['start_position'], dtype=float)
    w,x,y,z = map(float, raw['start_rotation'])
    if position.shape != (3,) or not np.isfinite(position).all() or abs(x)+abs(y)>1e-6 or not np.isclose(w*w+z*z,1):
        raise ValueError('invalid upright VLNVerse start pose')
    return position, math.atan2(2*w*z, 1-2*z*z)


def command(response):
    if type(response.get('stop')) is not bool:
        raise ValueError('LightNav must return an explicit boolean STOP')
    data = response.get('actions')
    if isinstance(data,dict): data=data.get('actions')
    arr=np.asarray(data,dtype=float)
    if arr.shape == (1,10,3):arr=arr[0]
    if arr.shape == (30,):arr=arr.reshape(10,3)
    if arr.shape != (10,3) or not np.isfinite(arr).all():
        raise ValueError('invalid LightNav RVQ-10 waypoints')
    if response['stop']:
        # A server decode failure must not masquerade as a real predicted STOP.
        if not response.get('raw_text') or '<act_' not in response['raw_text']:
            raise ValueError('STOP lacks model action token evidence')
        return 0.,0.
    idx=next((i for i,p in enumerate(arr) if abs(p[0])>1e-6 or abs(p[2])>1e-6),0)
    return float(np.clip(arr[idx,0],0,.25)), float(np.clip(arr[idx,2],-math.pi/6,math.pi/6))


class LightNavSession:
    def __init__(self, endpoint, client_id):
        from websockets.sync.client import connect
        self.ws=connect(endpoint,open_timeout=15,max_size=8*1024*1024,close_timeout=5)
        self.request('login',{'clientId':client_id})
        self.request('reset',{})

    def request(self, action, data):
        self.ws.send(json.dumps({'action':action,'data':data}))
        reply=json.loads(self.ws.recv(timeout=180))
        body=reply.get('data',{})
        if reply.get('action')!=action or body.get('rc')!=0:
            raise RuntimeError('LightNav request failed: '+str(body.get('msg','invalid response'))[:300])
        return body

    def predict(self, path, instruction, seq):
        body=self.request('next',{'seq':seq,'instruction':instruction,
            'image':base64.b64encode(path.read_bytes()).decode()})
        if body.get('seq')!=seq:raise ValueError('LightNav sequence mismatch')
        command(body)
        return body

    def close(self):self.ws.close()


class StaticCollision:
    def __init__(self, stage):
        from pxr import UsdGeom, UsdPhysics, UsdUtils
        from omni.physx import get_physx_simulation_interface, get_physx_scene_query_interface
        # All edits stay in the caller's anonymous session layer.
        for prim in list(stage.Traverse()):
            if prim.IsInstance():prim.SetInstanceable(False)
        count=0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
                count+=1
        if not count:raise ValueError('USD has no collision meshes')
        UsdPhysics.Scene.Define(stage,'/VLNPlatformPhysics')
        self.sim=get_physx_simulation_interface()
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if stage_id<0:stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        if not self.sim.attach_stage(stage_id):raise RuntimeError('PhysX scene attach failed')
        self.sim.simulate(1/60,0)
        self.sim.fetch_results()
        self.query=get_physx_scene_query_interface()
        self.meshes=count

    def floor_ok(self, position):
        from pxr import Gf
        origin=np.asarray(position)+[0,0,.2]
        hit=self.query.raycast_closest(Gf.Vec3f(*map(float,origin)),Gf.Vec3f(0,0,-1),.4,True)
        return bool(hit['hit'] and abs(float(hit['position'][2])-position[2])<=.12)

    def advance(self, position, yaw, forward, turn):
        from pxr import Gf
        position=position.copy(); blocked=False
        steps=max(1,math.ceil(forward/PROFILE['substep_m']),math.ceil(abs(turn)/math.radians(5)))
        for _ in range(steps):
            heading=yaw+turn/(2*steps)
            direction=np.array([math.cos(heading),math.sin(heading),0.])
            distance=forward/steps
            if distance>1e-8:
                for height in (.20,.36,.52,.68,.84):
                    origin=position+[0,0,height]
                    hit=self.query.sweep_sphere_closest(.18,Gf.Vec3f(*map(float,origin)),Gf.Vec3f(*map(float,direction)),distance+.015,True)
                    if hit['hit']:blocked=True;break
                target=position+direction*distance
                if not blocked and self.floor_ok(target):position=target
                else:blocked=True
            yaw+=turn/steps
        return position,yaw,blocked
