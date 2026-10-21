import os
from torch import nn
import torch.nn.functional as F
from Configurations.config import device
from  model.SH_model import SH_model_key, SH_G, SH_D
from  sim_dexee.Shadow_hand_env import ShadowHandEnv
from  training.abstract_training_module import AbstractGraspAgentTraining
from  training.sample_random_grasp import generate_random_SH_poses
from  utils.quat_operations import  grasp_frame_to_quat, quat_between
from utils. IO_utils import custom_print
from utils. cuda_utils import cuda_memory_report
import torch
from utils.args import get_args, read_config

bce_loss = nn.BCELoss()
bce_with_logits=nn.BCEWithLogitsLoss()

print = custom_print

def process_fingers(target_pose_):
    fingers = torch.clip(target_pose_[8:] + 0.5, 0, 1)
    fingers = fingers
    return fingers

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point
    delta=target_pose_[5:5 + 3].cpu().numpy()/15

    target_point_=target_point_+delta

    alpha=target_pose_[:3]

    beta=target_pose_[3:5]

    alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
    beta = F.normalize(beta, p=2, dim=0, eps=1e-8)

    approach_ref = torch.tensor([0.0, 0., 1.0], device=device)

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.],device=device))
    quat=grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers=process_fingers(target_pose_).cpu().tolist()

    assert all(x == x for x in quat), f"quat contains NaN, {quat,alpha,beta}"
    assert all(x == x for x in fingers), f"fingers contains NaN, {fingers}"

    if view:
        print()
        print('alpha: ', alpha)
        print('beta: ', beta)
        print('delta: ', delta)

        print('target_pose: ', target_pose)

        print('fingers: ', fingers)
        print('target_point_: ', target_point_)

    return quat,fingers,target_point_

class TrainGraspGAN(AbstractGraspAgentTraining):
    def __init__(self, args, epochs=1):

        super().__init__(args=args,sampler_policy_model=SH_G,critic_model=SH_D, epochs=epochs, model_key=SH_model_key,
                          randomization_unit=generate_random_SH_poses,
                         process_pose=process_pose, n_joints=3,train_policy_only=True,explore_mode=False
                         ,static_joints=[0,1,2])

        root_dir = os.getcwd()  # current working directory

        self.sim_env = ShadowHandEnv(root=root_dir + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)


def train_N_grasp_GAN(args,n=1):
    Train_grasp_GAN = TrainGraspGAN(args)
    torch.cuda.empty_cache()

    for i in range(n):
        cuda_memory_report()
        Train_grasp_GAN.initialize()
        Train_grasp_GAN.begin(iterations=10)

if __name__ == "__main__":
    args = get_args()

    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)

    train_N_grasp_GAN(args,n=10000)