from ikpy.chain import Chain
import numpy as np
import trimesh
from ikpy.chain import Chain
from ikpy.link import URDFLink

# Define the kinematic chain
my_chain = Chain(name='robot_arm', links=[
    URDFLink(
        name="link1",
        bounds=[-6.27, 6.27],
        origin_translation=[0, 0, 0.147],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="link2",
        bounds=[-6.27, 6.27],
        origin_translation=[0, 0, 0],
        origin_orientation=[-np.pi/2, np.pi/2, -np.pi],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="link3",
        bounds=[-2.79, 2.79],
        origin_translation=[-0.377, 0, 0.025],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="link4",
        bounds=[-6.27, 6.27],
        origin_translation=[-0.307, 0, 0.116],
        origin_orientation=[0, 0, -np.pi/2],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="link5",
        bounds=[-6.27, 6.27],
        origin_translation=[0, -0.116, 0],
        origin_orientation=[np.pi/2, 0, 0],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="link6",
        bounds=[-6.27, 6.27],
        origin_translation=[0, 0.105, 0],
        origin_orientation=[-np.pi/2, 0, 0],
        rotation=[0, 0, 1]
    ),
],active_links_mask=[True, True, True, True, True, True])
# Compute the inverse kinematics for a given target position
# target_position = np.array([-268.7269, -616.4493, 400.9593])/1000
# target_orientation = np.deg2rad([100.7061, -74.7654, -101.8784])
# target_orientation_matrix = trimesh.transformations.euler_matrix(*target_orientation)[:3, :3]

# ik_solution = my_chain.inverse_kinematics(target_position, target_orientation_matrix, initial_position=np.deg2rad([180,0,-90,90,90,-30]), orientation_mode="all")
# print("IK Solution:", np.rad2deg(ik_solution))

# fk_position = my_chain.forward_kinematics(ik_solution)[:3, 3]
# fk_rpy = trimesh.transformations.euler_from_matrix(my_chain.forward_kinematics(ik_solution))
# print(f"计算位置: {fk_position}")
# print(f"目标位置: {target_position}")
# print(f"计算欧拉角: {np.rad2deg(fk_rpy)}")
# print(f"目标欧拉角: {np.rad2deg(target_orientation)}")

def cr7_ik(T):
    target_position = T[:3, 3]
    target_orientation_matrix = T[:3, :3]
    ik_solution = my_chain.inverse_kinematics(target_position, target_orientation_matrix, initial_position=np.deg2rad([180,0,-90,90,90,-30]), orientation_mode="all")
    fk_T = my_chain.forward_kinematics(ik_solution)
    if np.max(np.abs(fk_T-T)) < 1e-4:
        return ik_solution
    else:
        return None


if __name__ == "__main__":
    xyz = np.array([-268.7269, -616.4493, 400.9593])/1000
    rpy = np.deg2rad([100.7061, -74.7654, -101.8784])
    T = trimesh.transformations.euler_matrix(*rpy)
    T[:3, 3] = xyz
    ik_res = cr7_ik(T)
    if ik_res is not None:
        print(np.rad2deg(ik_res))
    else:
        print("no solution")