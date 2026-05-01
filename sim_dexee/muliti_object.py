import time
import mujoco.viewer
from mujoco.renderer import Renderer

import numpy as np
import xml.etree.ElementTree as ET
from random import sample
import os
import trimesh


objects_path = "shadow_dexee/mesh/"
object_nums_all = len(os.listdir(objects_path))
obj_nums_in_scene = 3
assert obj_nums_in_scene <= object_nums_all

idxs = sample(range(object_nums_all), obj_nums_in_scene)

tree = ET.parse('shadow_dexee/scene.xml')
root = tree.getroot()
for idx in idxs:
    new_mesh = ET.Element('include')
    new_mesh.set('file', 'mesh/mesh_' + str(idx) + '.xml')
    root.insert(1, new_mesh)
tree.write('shadow_dexee/temp.xml')

m = mujoco.MjModel.from_xml_path('shadow_dexee/temp.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# Define camera parameters and init renderer.
height = 600
width = 600
camera_id = m.cam("camera_1").id
renderer = Renderer(m, height=height, width=width)

# Intrinsic matrix.
fov = m.cam_fovy[camera_id]
theta = np.deg2rad(fov)
fx = width / 2 / np.tan(theta / 2)
fy = height / 2 / np.tan(theta / 2)
cx = (width - 1) / 2.0
cy = (height - 1) / 2.0
intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Extrinsic matrix.
cam_pos = d.cam_xpos[camera_id]
cam_rot = d.cam_xmat[camera_id].reshape(3, 3)
extr = np.eye(4)
extr[:3, :3] = -cam_rot#cam_rot.T
extr[:3, 3] = cam_pos

def render_depth(
    renderer: mujoco.Renderer,
    camera_id: int,
) -> np.ndarray:
    renderer.update_scene(d, camera=camera_id)
    renderer.enable_depth_rendering()
    depth = renderer.render()
    return depth

def depth_to_pointcloud(
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    depth_trunc: float = 20.0,
) -> np.ndarray:
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
    valid = (depth > 0) & (depth < depth_trunc)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
    y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
    xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
    mask = np.isnan(xyz[:, 2])
    xyz = xyz[~mask]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_t = (extr @ xyz_h.T).T
    return xyz_t[:, :3]

# d.mocap_pos shape=[1, 3]
# d.mocap_quat shape=[1, 4] wxyz
d.mocap_pos[0] = [0, 0., 0.]
d.mocap_quat[0] = [0, 1, 0, 0]

# d.qpos shape=7+12+7*obj_nums_in_scene, first 7 for gripper_base, next 12 for 12 finger joints, then each mesh has 7, 3 for pos and 4 for quat(wxyz)
# set initial qpos
d.qpos = [0, 0., 0., 0, 1, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0.2, 0, -0.07, 1, 0, 0, 0, 0, 0, -0.07, 1, 0, 0, 0, -0.2, 0, -0.07, 1, 0, 0, 0]

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 0

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    counter=0
    while viewer.is_running() and time.time() - start < 300:
        counter+=1
        # set pos and quat of gripper base
        # print(d.time)
        if d.time > 0:
            d.mocap_pos[0] = d.mocap_pos[0] + [0, 0, 0.1]
        d.mocap_quat[0] = [0, 1, 0, 0]

        # set control of finger joints
        d.ctrl = [0, -0.4, 0, 0, 0, -0.4, 0, 0, 0, -0.4, 0, 0]

        for _ in range(20):
            mujoco.mj_step(m, d)

        if counter%20==0:
            depth = render_depth(renderer, camera_id)

            # print(depth.shape)
            pointcloud = depth_to_pointcloud(depth, intr, extr)
            pc = trimesh.PointCloud(pointcloud)
            pc.show()
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
