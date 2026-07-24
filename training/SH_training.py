import argparse
import configparser
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

bce_loss = nn.BCELoss()
bce_with_logits=nn.BCEWithLogitsLoss()

print = custom_print

def process_fingers(target_pose_):
    fingers = torch.clip(target_pose_[11:] + 0.5, 0, 1)
    fingers = fingers
    return fingers

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point
    delta=target_pose_[5:5 + 3].cpu().numpy()/15

    target_point_=target_point_+delta

    zeta=target_pose_[8:8 + 3].cpu().numpy()/15
    pre_grasp_point=target_point_+zeta

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
        print('zeta: ', zeta)

        print('pre_grasp_point: ', pre_grasp_point)
        print('target_pose: ', target_pose)

        print('fingers: ', fingers)
        print('target_point_: ', target_point_)

    return quat,fingers,target_point_.tolist(),pre_grasp_point.tolist()

class TrainGraspGAN(AbstractGraspAgentTraining):
    def __init__(self, args, epochs=1):

        super().__init__(args=args,sampler_policy_model=SH_G,critic_model=SH_D, epochs=epochs, model_key=SH_model_key,
                         test_mode=True, randomization_unit=generate_random_SH_poses,
                         process_pose=process_pose, n_joints=3,train_policy_only=False,explore_mode=False
                         ,static_joints=[0,1,2])

        root_dir = os.getcwd()  # current working directory

        self.sim_env = ShadowHandEnv(root=root_dir + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)
        # self.sim_env.plt_obj_dict_statistics()

def train_N_grasp_GAN(args,n=1):

    Train_grasp_GAN = TrainGraspGAN(args)

    torch.cuda.empty_cache()

    for i in range(n):
        cuda_memory_report()
        Train_grasp_GAN.initialize()
        # Train_grasp_GAN.show_overlaid_graphs()
        Train_grasp_GAN.begin(iterations=10)

def read_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the config file"
    )

    parser.add_argument(
        "--load_last_optimizer",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load last optimizer state (default: True). Use true/false."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )


    parser.add_argument(
        "--catch_exceptions",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Wrap the execution with try and except (default: True). Use true/false."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)

    print("Config path:", os.path.abspath(config_path))
    print("load_last_optimizer:", args.load_last_optimizer)

    train_N_grasp_GAN(args,n=10000)
