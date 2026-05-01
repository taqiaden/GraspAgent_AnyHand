import os

os.environ["MUJOCO_GL"] = "osmesa"
import os
import re
import time
import torch.nn.functional as F
import mujoco
import numpy as np
import torch

from Configurations.config import device
from  kinematic_utils.path_check import  kinematic_checker
from  training.sample_random_grasp import quat_between_batch
from  utils.Multi_finger_hand_env import MojocoMultiFingersEnv
from  utils.quat_operations import quat_rotate_vector, quat_mul, bulk_quat_mul, quat_between

def next_video_name(dir_path=".", prefix="simulation", ext=".mp4"):
    os.makedirs(dir_path, exist_ok=True)

    pattern = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(ext)}$")
    max_idx = 0

    for f in os.listdir(dir_path):
        m = pattern.match(f)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    return os.path.join(dir_path, f"{prefix}_{next_idx:03d}{ext}")

class CasiaHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root,max_obj_per_scene=2,is_tendon_control=False,objects_path=None):
        # self.scene_xml_file='/scene.xml' if is_tendon_control else '/scene_s.xml'
        self.hand_xml_file="CasiaHand/hand.xml" if is_tendon_control else "CasiaHand/hand_s.xml"
        super().__init__(root=root,max_obj_per_scene=max_obj_per_scene,key='CasiaHand_s',objects_path=objects_path)
        self.is_tendon_control=is_tendon_control

        self.root=root

        # self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0]

        self.default_ctrl=self.decode_finger_ctrl([0.,0.,0.])

        self.contact_pads_geom_ids=[[2,3,4],[12,17,22,23],[26,31,36,37,40,45,50,51,54,59,64,65,68,73,78,79]] # (pad1,pad2,pad3), ft1, (ft2,ft3,ft4,ft5)
        # self.contact_pads_geom_ids=[[12,17,22,23],[26,31,36,37,40,45,50,51,54,59,64,65,68,73,78,79]] # (pad1,pad2,pad3), ft1, (ft2,ft3,ft4,ft5)

        # self.intilize_finger_joints()

        # self.contact_pads_info()
    def  max_finger_ctrl(self):
        # print(args)
        if self.is_tendon_control:
            j_th = 0.091 - 0.027
            j_fm = 0.091 -  0.037  # forefinger and midmiddle finger
            j_rl = 0.091 -  0.037  # ring finger and little finger
        else:
            # j form 0 to 1 represent open to close
            j_th =   1.5
            j_fm =   1.5
            j_rl =   1.5
        return [j_th, j_fm, j_fm, j_rl, j_rl]

    def  decode_finger_ctrl(self,fingers):
        # print(args)
        if self.is_tendon_control:
            j_th = 0.091 - fingers[0] * 0.027
            j_fm = 0.091 - fingers[1] * 0.037  # forefinger and mid_middle finger
            j_rl = 0.091 - fingers[2] * 0.037  # ring finger and little finger
        else:
            # j form 0 to 1 represent open to close
            j_th = fingers[0] *  1.5
            j_fm = fingers[1] *  1.5
            j_rl = fingers[2] *  1.5
        return [j_th, j_fm, j_fm, j_rl, j_rl]

    def check_fingers_scope(self,fingers):
        fingers=np.array(fingers)

        return  np.all((fingers >= 0) & (fingers <= 1))

    def clip_fingers_to_scope(self,hand_fingers):
        return torch.clamp(torch.tensor(hand_fingers),min=0.01,max=0.99).tolist()

    def safety_fingers_check(self):
        fingers_state=self.d.qpos[3+4:3+4+15]
        for i in range(5):
            cumulative=fingers_state[i*5:i*5+3]
            if sum(cumulative)>4: return False

        return True

    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=600,hard_level=0.,shake=True,update_obj_prob=None):
        self.restore_simulation_state()

        if obj_pose is None: obj_pose=self.objects_poses
        in_scope=True
        # in_scope = self.check_fingers_scope(hand_fingers)
        # if not in_scope: hand_fingers = self.clip_fingers_to_scope(hand_fingers)
        grasped_obj=None

        warning_flag = False


        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        # try:
        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose
        # except:
        #     print(len(self.default_finger_joints),' ',len(hand_pos),' ',len(hand_quat),' ',len(obj_pose),' ',len(self.objects))
        #     assert False
        self.d.ctrl *= 0
        mujoco.mj_step(self.m, self.d)
        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        if ini_contact_with_obj or ini_contact_with_floor:
            # self.static_view(1000)
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor,None,None,None,warning_flag,grasped_obj
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++',self.default_finger_joints)
        delta=[0, 0, 0.003]
        decoded_fingers = self.decode_finger_ctrl(hand_fingers)
        # max_fingers = self.max_finger_ctrl()
        self.d.ctrl = decoded_fingers
        shake_amp = .003
        shake_f = 20  # Hz

        for i in range(600):
            if i==200:
                _, collide_with_floor = self.check_hand_contact()
                if collide_with_floor:
                    # self.static_view(1000)
                    return in_scope, False, ini_contact_with_obj, collide_with_floor, None, None, None, warning_flag, grasped_obj

            #Rise phase
            if 200 < i < 400:
                self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

            # shake phase
            if 500 > i > 400:
                if i==401:
                    grasp_success, n_grasp_contact1, self_collide1,max_force1,max_penetration1 = self.check_valid_grasp(minimum_contact_points=0)
                    if not grasp_success or not shake: break
                # self.d.ctrl = max_fingers #if i < 400 else decoded_fingers
                t = i * self.m.opt.timestep
                phase = 2 * np.pi * shake_f * t
                shake = shake_amp * np.array([np.sin(phase),
                                              np.sin(phase + 2.1),
                                              np.sin(phase + 4.2)])
                # shake = shake_amp * np.sin(2 * np.pi * shake_f * t)
                self.d.mocap_pos[0] += shake  # vertical shake (z)
            mujoco.mj_step(self.m, self.d)

            qpos = self.d.qpos
            qvel = self.d.qvel
            qacc=self.d.qacc
            MAX_MAG = 1e6
            bad = (
                    (not np.all(np.isfinite(qpos))) or
                    (not np.all(np.isfinite(qvel))) or
                    (not np.all(np.isfinite(qacc))) or
                    np.any(np.abs(qpos) > MAX_MAG) or
                    np.any(np.abs(qvel) > MAX_MAG)
            )
            if bad:
                warning_flag = True

        if grasp_success:
            stable_grasp,n_grasp_contact2,self_collide2,max_force2,max_penetration2 = self.check_valid_grasp(minimum_contact_points=0)
            # print(f'---test------------------------------{max_force1,max_penetration1,max_force2,max_penetration2}')
            grasped_obj = self.get_grasped_obj()
            if update_obj_prob is not None and not warning_flag:

                # print(f'grasped_obj_: {grasped_obj}')

                # s=1.0 if stable_grasp else 0.9
                # if stable_grasp:print(Fore.GREEN,f"object {grasped_obj} grasped successfully",Fore.RESET)
                self.step_obj_prop(grasped_obj,scale=update_obj_prob)

            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,min(n_grasp_contact1,n_grasp_contact2),self_collide1 or self_collide2,stable_grasp,warning_flag,grasped_obj
        else:
            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact1,self_collide1,None,warning_flag,grasped_obj

    def view_grasp(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=300,hard_level=0.   ):
        self.restore_simulation_state()
        if obj_pose is None: obj_pose=self.objects_poses

        in_scope =True# self.check_fingers_scope(hand_fingers)
        # if not in_scope:hand_fingers = self.clip_fingers_to_scope(hand_fingers)
        # v2 = quat_rotate_vector(hand_quat, [0, 1, 0])
        # if v2[-1]<0:in_scope=False

        # if not in_scope:
        #     return in_scope,None,None, None

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose
        self.d.ctrl *= 0
        mujoco.mj_step(self.m, self.d)
        self.static_view(1000)

        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        # self.static_view(1000)

        delta=[0, 0, 0.003]
        decoded_fingers=self.decode_finger_ctrl(hand_fingers)
        # max_fingers=self.max_finger_ctrl()
        self.d.ctrl = decoded_fingers
        shake_amp = .003
        shake_f = 20  # Hz

        # video_path = next_video_name("CasiaHand_sim_clips", prefix="simulation")
        # writer = imageio.get_writer(video_path, fps=30)

        # Off-screen renderer for recording
        # renderer = mujoco.Renderer(self.m, width=640, height=480)

        # for i in range(70):
        #     mujoco.mj_step(self.m, self.d)
        #     if i==10:self.static_view(1000)



        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            for i in range(70,600):
                step_start = time.time()


                if 200 < i < 400:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

                # shake phase
                if 500 > i > 400:
                    # self.d.ctrl = max_fingers #if i<400 else decoded_fingers

                    t = i * self.m.opt.timestep
                    phase = 2 * np.pi * shake_f * t
                    shake = shake_amp * np.array([
                        np.sin(phase),
                        np.sin(phase + 2.1),
                        np.sin(phase + 4.2)
                    ])
                    self.d.mocap_pos[0] += shake

                mujoco.mj_step(self.m, self.d)
                viewer.sync()

                # -------- capture frame from renderer --------
                # renderer.update_scene(self.d)
                # frame = renderer.render()  # RGB uint8
                # writer.append_data(frame)

                # maintain real-time speed
                dt = self.m.opt.timestep
                time.sleep(max(0, dt - (time.time() - step_start)))

        # writer.close()
        # print("Saved video to simulation.mp4")

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success,n_grasp_contact,self_collide,max_force2,max_penetration2 = self.check_valid_grasp(minimum_contact_points=0,view=False)
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # print(Fore.CYAN,f'final d.mocap_quat[0] {self.d.mocap_quat[0]}',Fore.RESET)
        # print(Fore.CYAN,f'final d.qpos[3:3+4] {self.d.qpos[3:3 + 4]}',Fore.RESET)

        grasped_obj=self.get_grasped_obj()
        print(f'grasped_obj: {grasped_obj}')

        if grasp_success:self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact,self_collide,None

    def get_grasped_obj(self):
        k = 3 + 4 + len(self.default_finger_joints)
        objects_poses = self.d.qpos[k:]
        max_elevation = None
        grasped_obj = None
        for i in range(len(self.objects)):
            n = ((len(self.objects) - i - 1) * 7)
            pose = objects_poses[n:n + 3]
            # quat = objects_poses[n + 3:n + 7]
            if max_elevation is None:
                max_elevation = pose[-1]
                grasped_obj = self.objects[i]
            else:
                if pose[-1] > max_elevation:
                    max_elevation = pose[-1]
                    grasped_obj = self.objects[i]

        return grasped_obj

def sample_quat(size,f=0.5,ref_quat=None):
    ref_quat = torch.tensor([[0., 1., 0., 0.]],device=device) if ref_quat is None else ref_quat

    beta_quat=torch.zeros((size,4),device=device)
    beta_quat[:,[0,3]]=torch.randn((size, 2), device=device)
    beta_quat = F.normalize(beta_quat, dim=-1)

    approach=(torch.rand((10000, 3), device=device))

    approach[:,[0,2]]=2*(approach[:,[0,2]]-0.5)
    U=approach[:,1]
    k=2
    approach[:, 1] = (1 + torch.sign(2*U - 1) * torch.abs(2*U - 1) ** (1 / (k + 1))) / 2 # this is the CDF inversion of the Probability function defined as (x/0.5-1)^k

    # y_np =x.cpu().numpy()
    # # Plot histogram
    # plt.hist(y_np, bins=50, range=(0, 1), density=True, alpha=0.7, color='skyblue')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Histogram of Tensor Data')
    # plt.show()

    approach[:, :2]*=f
    approach = F.normalize(approach, dim=-1)

    approach_quat=quat_between_batch(torch.tensor([0.0, 1.0, 0.0],device=device),approach)
    approach_quat = F.normalize(approach_quat, dim=-1)

    quat=bulk_quat_mul(beta_quat,ref_quat)

    quat=bulk_quat_mul(approach_quat,quat)
    quat = F.normalize(quat, dim=-1)

    return quat

if __name__ == "__main__":

    root_dir = os.getcwd()  # current working directory

    env=CasiaHandEnv(root=root_dir +  "/sim_dexee/hands_and_objects/",max_obj_per_scene=5,is_tendon_control=False)

    env.view_geom_names_and_ids()



    kinematics = kinematic_checker()
    for i in range(1000):
        # env.prepare_obj_mesh()
        # env.initialize()
        env.drop_new_obj(selected_index=258,obj_pose=[0, 0.3, 0.2],obj_quat=[1,0,0,0], stablize=True)



        from  training.CH_training import process_pose
        while True:

            shifted_point = np.array([-268.7269, -616.4493, 400.9593]) / 1000
            # shifted_point[0:2]*=0
            rpy = np.deg2rad([100.7061, -74.7654, -101.8784])
            # quat = trimesh.transformations.quaternion_from_euler(*rpy)

            # fingers[0]=1.
            # fingers[1]=1.
            # fingers[2]=1.
            fingers=[1,1,1.]

            env.manual_view(pos=shifted_point.tolist(), quat=[0,1,0,0], fingers=fingers)

            # env.passive_viewer(pos=shifted_point.tolist(), quat=quat.tolist(),fingers=fingers)

            # for i in range(3):
            #     k=7+i
            #     print(f'----joint {i+1}')
            #     print(env.d.qpos[k:k+1])
            #     print(env.d.qpos[k+3:3+k+1])
            #     print(env.d.qpos[k+6:6+k+1])
            #     print(env.d.qpos[k+9:9+k+1])
            #     print(env.d.qpos[k+12:12+k+1])

        # env.get_scene_preception(view=True)