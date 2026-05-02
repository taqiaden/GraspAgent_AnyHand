import mujoco
import mujoco.viewer
from mujoco.renderer import Renderer
import numpy as np
import open3d as o3d

XML = """
<mujoco>
  <default>
    <geom mass=".01" solref="-1000 0"/>
  </default>
  <visual>
    <rgba haze="0 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1=".2 .2 .2" rgb2=".3 .3 .3" markrgb=".8 .8 .8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2 2" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="camera" pos="0 -2 .5" axisangle="1 0 0 90" fovy="45"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.3" rgba=".2 .5 .2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Define camera parameters and init renderer.
    height = 480
    width = 640
    fps = 30
    camera_id = model.cam("camera").id
    renderer = Renderer(model, height=height, width=width)

    # Intrinsic matrix.
    fov = model.cam_fovy[camera_id]
    theta = np.deg2rad(fov)
    fx = width / 2 / np.tan(theta / 2)
    fy = height / 2 / np.tan(theta / 2)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Extrinsic matrix.
    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    print(cam_pos)
    print(cam_rot)
    exit()
    extr = np.eye(4)
    extr[:3, :3] = cam_rot.T
    extr[:3, 3] = cam_pos

    def render_rgbd(
        renderer: mujoco.Renderer,
        camera_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        renderer.update_scene(data, camera=camera_id)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        rgb = renderer.render()
        return rgb, depth

    def rgbd_to_pointcloud(
        rgb: np.ndarray,
        depth: np.ndarray,
        intr: np.ndarray,
        extr: np.ndarray,
        depth_trunc: float = 20.0,
    ):
        cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
        valid = (depth > 0) & (depth < depth_trunc)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
        y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
        xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
        color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
        mask = np.isnan(xyz[:, 2])
        xyz = xyz[~mask]
        color = color[~mask]
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_t = (extr @ xyz_h.T).T
        xyzrgb = np.hstack([xyz_t[:, :3], color])
        return xyzrgb

    # Simulate for 10 seconds and capture RGB-D images at fps Hz.
    xyzrgbs: list[np.ndarray] = []
    mujoco.mj_resetData(model, data)
    while data.time < 10.0:
        mujoco.mj_step(model, data)
        if len(xyzrgbs) < data.time * fps:
            rgb, depth = render_rgbd(renderer, camera_id)
            xyzrgb = rgbd_to_pointcloud(rgb, depth, intr, extr)
            xyzrgbs.append(xyzrgb)

    # Visualize in open3d.
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgbs[0][:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgbs[0][:, 3:])
    vis.add_geometry(pcd)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    vis.add_geometry(frame)

    counter: int = 1

    def update_pc(vis):
        global counter
        if counter < len(xyzrgbs) - 1:
            pcd.points = o3d.utility.Vector3dVector(xyzrgbs[counter][:, :3])
            pcd.colors = o3d.utility.Vector3dVector(xyzrgbs[counter][:, 3:])
            vis.update_geometry(pcd)
            counter += 1

    vis.register_animation_callback(update_pc)
    vis.run()
    vis.destroy_window()
    renderer.close()