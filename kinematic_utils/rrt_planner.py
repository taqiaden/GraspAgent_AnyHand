import numpy as np
import pybullet as p
import time

from Configurations.config import where_am_i

full_joint_names = [
    "world2arm_fixed",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "RArm6_to_hand_fixed",
    "box_joint",
    "RTh_joint1",
    "RTh_joint2",
    "RTh_joint3",
    "RFF_joint1",
    "RFF_joint2",
    "RFF_joint3",
    "RMF_joint1",
    "RMF_joint2",
    "RMF_joint3",
    "RRF_joint1",
    "RRF_joint2",
    "RRF_joint3",
    "RLF_joint1",
    "RLF_joint2",
    "RLF_joint3",
]
planning_joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]
# urdf文件标定偏差量
joint_limits = [(-2 * np.pi, 2 * np.pi)] * len(planning_joint_names)
import os


urdf_path = r'/kinematic_utils/cr7_robot_right.urdf'


class RRTConnectPlanner:
    def __init__(
            self,
            urdf_path=urdf_path,
            full_joint_names=full_joint_names,
            planning_joint_names=planning_joint_names,
            joint_limits=joint_limits,
            obstacle_pos=None,
            obstacle_radius=0.0,
            obstacle_config_data=None,
            outer_init_flag: bool = False,
            remaining_obj_id_list: list = None,
            cam2base_matrix: np.array = None,
    ):
        p.setGravity(0, 0, -9.8)
        # 更严格的穿透检测
        p.setPhysicsEngineParameter(contactBreakingThreshold=1e-5)

        # 加载机器人 URDF
        self.robot = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )

        # 设置相机视角，便于观察
        p.resetDebugVisualizerCamera(
            cameraDistance=0.7,
            cameraYaw=-45,
            cameraPitch=-30,  # 俯仰
            cameraTargetPosition=[-0.5, 0, 0.1]
        )

        # 设置手部关节角度
        hand_idxs = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        hand_angle = np.zeros(len(hand_idxs))
        hand_angle = np.array(
            [0, 0, 0, 0,  # 大拇指
             0, 0, 0,
             0, 0, 0]
        )
        for hand_idx, angle in zip(hand_idxs, hand_angle):
            p.resetJointState(self.robot, hand_idx, angle)
        # p.resetJointState(self.robot, 8, 1.5)
        # max_force=500
        # mode = p.POSITION_CONTROL

        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.robot,
        #     jointIndices=hand_idxs,
        #     controlMode=mode,
        #     targetPositions=hand_angle,
        #     forces=[max_force] * len(hand_idxs)
        # )
        self.obstacle = None
        self.obstacles_idxs = []
        self.scene_obstacles_idxs = []
        self.object_obstacles_idxs = []
        self.attached_objects = []

        # 试验台立柱
        pillar_half_extents = [0.03, 0.10, 0.9]
        pillar_pos = [0.22, 0, 0]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=pillar_half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=pillar_half_extents, rgbaColor=[0.8, 0.6, 0.4, 0.5]
        )
        pid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pillar_pos,
        )
        self.obstacles_idxs.append(pid)
        # 立柱相机
        cam_half_extents = [0.04, 0.065, 0.025]
        cam_pos = [0.16, 0, 0.51]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=cam_half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=cam_half_extents, rgbaColor=[0.8, 0.6, 0.4, 0.5]
        )
        cid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=cam_pos,
        )
        self.obstacles_idxs.append(cid)

        # # 桌面
        # table_half_extents = [1, 1, 0.01]
        # table_pos = [0, 0, -0.011]
        # 桌面高度应该是0.064（基座坐标系）

        table_half_extents = [0.5, 1, 0.062]  # 0.065-0.068 实际桌面高度0.05+0.01
        table_pos = [-0.6, 0, -0.01]

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.6, 0.4, 0.5]
        )
        tid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=table_pos,
        )
        self.obstacles_idxs.append(tid)

        for _ in range(10):
            p.stepSimulation()

        # 关节映射
        self.full_joint_names = full_joint_names
        self.planning_joint_names = planning_joint_names
        self.joint_limits = joint_limits
        self.planning_indices = [
            full_joint_names.index(n) for n in planning_joint_names
        ]
        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode()
            if name in full_joint_names:
                self.joint_name_to_index[name] = i
        missing = set(full_joint_names) - set(self.joint_name_to_index.keys())
        if missing:
            print(f"[WARN] Joint mapping missing: {missing}")

    def connect_object_to_end_effector(self, object_id, ee_link_index=7):
        """
        将抓取的物体连接到机器人的末端执行器上，模拟抓取效果

        Args:
            object_id: 物体在PyBullet中的ID
            ee_link_index: 末端执行器链接索引，默认为7

        Returns:
            constraint_id: 创建的约束ID，可用于后续移除约束
        """
        # 创建一个固定约束，将物体固定到机器人末端执行器
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=6,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, -0, 0],
            childFramePosition=[-0.040, -0.06, -0.1232],  # 四舍五入的相对位置,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )
        # 设置约束的一些物理参数以减少抖动
        p.changeConstraint(constraint_id, maxForce=2000)

        # 记录已附着的物体
        self.attached_objects.append(object_id)

        print("连接id", constraint_id)
        return constraint_id

    def get_object_idx(self):
        return self.object_obstacles_idxs

    def get_zip_object_idx(self):
        return self.real2sim_zip

    def sample(self):
        return np.array([np.random.uniform(l, h) for l, h in self.joint_limits])

    def nearest(self, tree, q_rand):
        dists = [np.linalg.norm(q - q_rand) for q, _ in tree]
        idx = int(np.argmin(dists))
        return idx, tree[idx][0]

    def steer(self, q_from, q_to, step_size=0.2):
        d = q_to - q_from
        dist = np.linalg.norm(d)
        if dist < step_size:
            return q_to.copy()
        return q_from + d / dist * step_size

    def collision_free(self, q_partial, static_full):
        fa = static_full.copy()
        for i, qi in enumerate(q_partial):
            fa[self.planning_indices[i]] = qi
        for name, ang in zip(self.full_joint_names, fa):
            idx = self.joint_name_to_index[name]
            p.resetJointState(self.robot, idx, ang)
            # p.setJointMotorControl2(
            #     bodyUniqueId=self.robot,
            #     jointIndex=idx,
            #     controlMode=p.POSITION_CONTROL,
            #     targetPosition=ang
            # )
            # p.stepSimulation()

        if self.obstacle and p.getClosestPoints(
                self.robot, self.obstacle, distance=0.0
        ):
            return False
        for oid in self.obstacles_idxs:
            if p.getClosestPoints(self.robot, oid, distance=0.0):
                return False

        # 添加自碰撞检测
        # 获取所有链接的索引
        link_indices = list(range(p.getNumJoints(self.robot)))
        known_links = [(8, 12), (8, 15), (8, 18), (8, 21), (12, 15), (15, 18)]
        # 检查自碰撞 - 排除相邻链接
        for i in range(len(link_indices)):
            for j in range(i + 2, len(link_indices)):  # i+2 跳过直接相邻的链接
                # 特殊处理: 跳过某些已知的非碰撞链接对
                if (i, j) in known_links or (j, i) in known_links:
                    continue
                # 这里可以根据具体机器人结构调整
                closest_points = p.getClosestPoints(
                    bodyA=self.robot,
                    bodyB=self.robot,
                    distance=0.0,
                    linkIndexA=i,
                    linkIndexB=j
                )
                if closest_points:
                    return False
        # 添加自碰撞检测

        # # 检查已附着物体与环境障碍物的碰撞
        # for attached_obj_id in self.attached_objects:
        #     for oid in self.obstacles_idxs:
        #         if oid != attached_obj_id and oid not in self.attached_objects:
        #             if p.getClosestPoints(attached_obj_id, oid, distance=0.0):
        #                 return False
        return True

    def collision_free_edge(self, q_from, q_to, static_full, n=10):
        for a in np.linspace(0, 1, n):
            if not self.collision_free(q_from + a * (q_to - q_from), static_full):
                return False
        return True

    def _reconstruct(self, tree, idx):
        path = []
        while idx is not None:
            q, pid = tree[idx]
            path.append(q.copy())
            idx = pid
        return list(reversed(path))

    def plan(
            self,
            q_start,
            q_goal,
            static_full,
            max_iters=5000,
            step_size=0.2,
            goal_thresh=0.1,
    ):
        tree_s = [(q_start.copy(), None)]
        tree_g = [(q_goal.copy(), None)]
        for _ in range(max_iters):
            q_rand = self.sample()
            # extend from start
            i_s, q_near_s = self.nearest(tree_s, q_rand)
            q_new_s = self.steer(q_near_s, q_rand, step_size)
            if not self.collision_free_edge(q_near_s, q_new_s, static_full):
                continue
            tree_s.append((q_new_s.copy(), i_s))
            # extend from goal
            reached = False
            while True:
                i_g, q_near_g = self.nearest(tree_g, q_new_s)
                q_new_g = self.steer(q_near_g, q_new_s, step_size)
                if not self.collision_free_edge(q_near_g, q_new_g, static_full):
                    break
                tree_g.append((q_new_g.copy(), i_g))
                if np.linalg.norm(q_new_g - q_new_s) < goal_thresh:
                    reached = True
                    break
            if reached:
                ps = self._reconstruct(tree_s, len(tree_s) - 1)
                pg = self._reconstruct(tree_g, len(tree_g) - 1)
                return ps + pg[-2::-1]
            tree_s, tree_g = tree_g, tree_s
        print("目标不可达")
        return None

    @staticmethod
    def time_parameterize(path, max_velocity):
        times = [0.0]
        traj = [(path[0], 0.0)]
        for i in range(1, len(path)):
            dq = np.abs(path[i] - path[i - 1])
            dt = (
                np.max(dq / max_velocity)
                if np.iterable(max_velocity)
                else np.max(dq) / max_velocity
            )
            t = times[-1] + dt
            times.append(t)
            traj.append((path[i], t))
        return traj


def simplify_path(path, static_full, planner, n_checks=20):
    """
    对关节空间路径做“直连剪枝”：尝试用后续关键点直接连接前一点，
    如果在插值后都无碰撞，则跳过中间点，去除冗余绕行。
    """
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        # 从末尾开始寻找可直连的最远点
        j = len(path) - 1
        while j > i + 1:
            if planner.collision_free_edge(path[i], path[j], static_full, n=n_checks):
                break
            j -= 1
        # 连接到 j 点
        simplified.append(path[j])
        i = j
    return simplified


def get_object_position(self, object_id):
    """
    获取物体当前的位置

    Args:
        object_id: 物体在PyBullet中的ID

    Returns:
        position: 物体的[x, y, z]坐标位置
    """
    pos, _ = p.getBasePositionAndOrientation(object_id)
    return list(pos)


def smooth_trajectory(traj_time, dt=1 / 240.0):
    """
    对关键帧 traj_time 做等时线性插值，输出固定频率轨迹。
    """
    qs = [q for q, _ in traj_time]
    ts = [t for _, t in traj_time]
    t_end = ts[-1]
    result = []
    t_samples = np.arange(0.0, t_end + 1e-8, dt)
    # print("qs -1",qs[-3:])
    # print("t_sample -3:",t_samples[-3:])
    for t in t_samples:
        if t >= t_end:
            q = qs[-1]
        else:
            # 找到所属线段
            idx = next(i for i in range(len(ts) - 1) if ts[i] <= t < ts[i + 1])
            t1, t2 = ts[idx], ts[idx + 1]
            q1, q2 = qs[idx], qs[idx + 1]
            alpha = (t - t1) / (t2 - t1)
            q = q1 + alpha * (q2 - q1)
        result.append((q, t))
    result.append((qs[-1], t_end))
    return result


def plan_path(
        planner: RRTConnectPlanner = None,
        q_start=None,
        q_goal=None,
):
    # 获取当前所有关节静态状态
    static_full = []
    for name in full_joint_names:
        idx = planner.joint_name_to_index.get(name)
        static_full.append(
            p.getJointState(planner.robot, idx)[0] if idx is not None else 0.0
        )
    static_full = np.array(static_full)
    # print("关节状态为",static_full)

    # print("目标关节角度:", q_goal)
    # 碰撞预检
    if not planner.collision_free(q_start, static_full):
        # print("Error: 起点与障碍物发生碰撞")
        return None, None
    if not planner.collision_free(q_goal, static_full):
        # print("Error: 终点与障碍物发生碰撞")
        return None, None

    # RRT-Connect 规划
    np.random.seed(1234)
    raw_path = planner.plan(q_start, q_goal, static_full)

    if raw_path is not None:
        if np.allclose(raw_path[0], q_goal, atol=1e-6):
            raw_path = raw_path[::-1]
    else:
        print("raw path is none")
        p.resetSimulation()
        return None, None

    # 1. 先对原始离散路径做剪枝，去除多余绕行
    pruned_path = simplify_path(raw_path, static_full, planner, n_checks=20)

    # 2. 时间参数化
    traj_time = planner.time_parameterize(pruned_path, max_velocity=1)

    # print(traj_time[-2:])
    # 3. 等时线性插值生成 240Hz 平滑轨迹
    smooth_traj = smooth_trajectory(traj_time, dt=4 / 240.0)
    # print(smooth_traj[-2:])
    path = []
    for q, _ in smooth_traj:
        q = q * 180 / np.pi
        # print(f"time={_:.2f}, q={q}")
        path.append(q)
    return smooth_traj, path


if __name__ == "__main__":
    vis = True
    if vis:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    planner = RRTConnectPlanner()

    if vis:
        q = np.deg2rad([268, -35, -70, 30, 2.5, -30])
        # q=np.deg2rad([186, 5, -115, 75, 74, -83.5])
        for name, angle in zip(planning_joint_names, q):
            idx = planner.joint_name_to_index[name]
            p.resetJointState(planner.robot, idx, angle)
        p.stepSimulation()
        input("静态展示机械臂")
        exit()

    # 出发与目标关节配置
    q_start = np.array([268, -35, -70, 30, 2.5, -30])
    q_goal = np.array([242.88754, 8.176451, -143.289046, 100.310021, 67.362005, -15.370145])
    q_start = np.deg2rad(q_start)
    q_goal = np.deg2rad(q_goal)

    # 输出规划好的运动路径
    smooth_traj, _ = plan_path(planner, q_start, q_goal)

    if vis:
        # 设置相机视角，便于观察
        p.resetDebugVisualizerCamera(
            cameraDistance=1.46,
            cameraYaw=-87,
            cameraPitch=-35,  # 俯仰
            cameraTargetPosition=[0.4, -0.15, -0.42]
        )
    path = [q[0] * 180 / np.pi for q in smooth_traj]

    print(path[-2:])
    print(len(path))
    print("抓取位置关节角", np.rad2deg(q_goal))

    np.save("path.npy", np.asarray(path))

    if vis:
        while True:
            try:
                for q, _ in smooth_traj:
                    for name, angle in zip(planning_joint_names, q):
                        idx = planner.joint_name_to_index[name]
                        p.resetJointState(planner.robot, idx, angle)
                    p.stepSimulation()
                    time.sleep(8 / 240)
                time.sleep(1)
            except KeyboardInterrupt:
                break

    p.disconnect()

