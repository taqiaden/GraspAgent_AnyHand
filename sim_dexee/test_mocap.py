import time

import mujoco
import mujoco.viewer

import numpy as np
import xml.etree.ElementTree as ET


tree = ET.parse('shadow_dexee/scene.xml')
root = tree.getroot()
new_mesh = ET.Element('include')
new_mesh.set('file', 'mesh/mesh_0.xml')
root.insert(1, new_mesh)
tree.write('shadow_dexee/temp.xml')

m = mujoco.MjModel.from_xml_path('shadow_dexee/temp.xml')
d = mujoco.MjData(m)

d.mocap_pos[0] = [0, 0, 0.35]
d.qpos = [0, 0, 0.35, 0, 1, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, 0, -0.07, 1, 0, 0, 0]

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        if d.time > 0.5:
            d.mocap_pos[0] = d.mocap_pos[0] + [0, 0, 0.0001]
        d.mocap_quat[0] = [0, 1, 0, 0]
        d.ctrl = [0, -0.4, 0, 0, 0, -0.4, 0, 0, 0, -0.4, 0, 0]
        print(d.qpos)
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

