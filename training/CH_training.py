import argparse
import configparser
import os
import torch.nn.functional as F
from Configurations.config import device
from  model.CH_model import CH_model_key, CH_D, CH_G
from sim_dexee.Casia_hand_env import CasiaHandEnv
from  training.abstract_training_module import AbstractGraspAgentTraining
from  training.sample_random_grasp import ch_pose_interpolation
from  utils.quat_operations import  grasp_frame_to_quat, quat_between
from utils. cuda_utils import cuda_memory_report
import torch

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point
    delta=target_pose_[5:5 + 3].cpu().numpy()/30

    target_point_=target_point_+delta

    alpha=target_pose_[:3]

    beta=target_pose_[3:5]

    alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
    beta = F.normalize(beta, p=2, dim=0, eps=1e-8)

    approach_ref=torch.tensor([0.866, -0.5, 0],device=device)

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.],device=device))
    quat=grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers = target_pose[5+3:].cpu().numpy().tolist()

    fingers=[min(1,max(0,x+0.5)) for x in fingers]

    assert all(x == x for x in quat), f"quat contains NaN, {quat}"
    assert all(x == x for x in fingers), f"fingers contains NaN, {fingers}"

    if view:
        print()
        print('quat: ',quat)
        print('fingers: ',fingers)
        print('target_point_: ',target_point_)

    return quat,fingers,target_point_.tolist()

class TrainGraspGAN(AbstractGraspAgentTraining):
    def __init__(self, args, epochs=1):

        super().__init__(args=args, sampler_policy_model=CH_G,critic_model=CH_D,epochs=epochs ,model_key=CH_model_key,
                         test_mode=False,pose_interpolation=ch_pose_interpolation,
                         process_pose=process_pose,n_param=11,check_kinematics=True)

        self.sim_env = CasiaHandEnv(root=os.getcwd() + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)

def train_N_grasp_GAN(args,n=1):
    Train_grasp_GAN = TrainGraspGAN(args)
    torch.cuda.empty_cache()

    for i in range(n):
        cuda_memory_report()
        Train_grasp_GAN.initialize()
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

    # Normalize filename (avoid config.ini.ini)
    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)

    print("Config path:", os.path.abspath(config_path))
    print("load_last_optimizer:", args.load_last_optimizer)

    train_N_grasp_GAN(args,n=10000)
