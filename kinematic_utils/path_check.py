import copy

import trimesh
from colorama import Fore

from kinematic_utils.ik_cr7 import cr7_ik
from kinematic_utils.rrt_planner import *

class kinematic_checker():
    def __init__(self):
        p.connect(p.DIRECT)  # initialize this only once, not every time
        self.planner = RRTConnectPlanner()  # initialize this only once, not every time

    def T_hand_to_T_tcp(self,T):
        T = copy.deepcopy(T)
        T[:3, 3] = T[:3, 3] + np.array([-0.5, 0.0, 0.19])
        T_arm_2_hand = np.array([[-0.5, np.cos(np.pi / 6), 0, -0.08],
                                 [0, 0, -1, 0.14],
                                 [-np.cos(np.pi / 6), -0.5, 0, -0.02],
                                 [0, 0, 0, 1]])
        T_tcp = T @ T_arm_2_hand
        return T_tcp

    def kinematic_plan_exist(self,quat, shifted_point,check_plan_path=True):
        while True:
            try:
                T = trimesh.transformations.quaternion_matrix(quat)
                T[:3, 3] = shifted_point

                T_pre = copy.deepcopy(T)
                T_pre[:3, 3] = T_pre[:3, 3] - 0.0866 * trimesh.transformations.unit_vector(
                    T_pre[:3, 0]) + 0.05 * trimesh.transformations.unit_vector(T_pre[:3, 1])

                T_tcp_final = self.T_hand_to_T_tcp(T)
                T_tcp_pre = self.T_hand_to_T_tcp(T_pre)

                '''trajectory'''
                joint = cr7_ik(T_tcp_final)
                if joint is None:
                    print('no joint for final pose')
                    return False
                joint = cr7_ik(T_tcp_pre)  # already in rad
                if joint is None:
                    print('no joint for pre pose')
                    return False
                q_goal = joint
                q_start = np.deg2rad([268, -35, -70, 30, 2.5, -30])

                # 输出规划好的运动路径
                if check_plan_path:
                    smooth_traj, _ = plan_path(self.planner, q_start, q_goal)

                    if smooth_traj is None:
                        # print('no path')
                        return False
                # print('path found')

                return True
            except Exception as e:
                print(Fore.RED,' Error while searching for a trajectory', Fore.RESET)
                # traceback.print_exc()

                p.connect(p.DIRECT)  # initialize this only once, not every time
                self.planner = RRTConnectPlanner()  # initialize this only once, not every time


if __name__ == "__main__":
    # p.connect(p.DIRECT)  # initialize this only once, not every time
    # planner = RRTConnectPlanner()  # initialize this only once, not every time
    kinematics=kinematic_checker()

    xyz = np.array([-268.7269, -616.4493, 400.9593])/1000
    rpy = np.deg2rad([100.7061, -74.7654, -101.8784])
    quat = trimesh.transformations.quaternion_from_euler(*rpy)

    print(kinematics.kinematic_plan_exist(quat, xyz))