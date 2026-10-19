import argparse
import configparser
import os
import torch.nn.functional as F
from Configurations.config import device
from  model.CH_model import CH_model_key, CH_D, CH_G
from sim_dexee.Casia_hand_env import CasiaHandEnv
from  training.abstract_training_module import AbstractGraspAgentTraining
from  training.sample_random_grasp import generate_random_CH_poses
from  utils.quat_operations import  grasp_frame_to_quat, quat_between
from utils. cuda_utils import cuda_memory_report
import torch
from utils.args import get_args

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

    fingers = target_pose[8:].cpu().numpy().tolist()

    fingers=[min(1,max(0,x+0.5)) for x in fingers]

    assert all(x == x for x in quat), f"quat contains NaN, {quat}"
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

        super().__init__(args=args, sampler_policy_model=CH_G,critic_model=CH_D,epochs=epochs ,model_key=CH_model_key,
                         test_mode=args.test_mode,randomization_unit=generate_random_CH_poses,
                         process_pose=process_pose,n_joints=3,check_kinematics=False,train_policy_only=True,explore_mode=False,
                         domain_randomization=False)

        self.sim_env = CasiaHandEnv(root=os.getcwd() + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)
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

def  run_test():
    from rich.live import Live
    from utils.test_board import generate_combined_table
    import time
    Train_grasp_GAN = TrainGraspGAN(args)
    Train_grasp_GAN.initialize()
    hand_name = 'CasiaHand'
    i=0
    steps=args.steps
    # while True:
    #     Train_grasp_GAN.begin(iterations=1)

    with Live(generate_combined_table(Train_grasp_GAN.metric_dict,i,steps,hand_name=hand_name), refresh_per_second=4, screen=True) as live:
        while True:
            i = (Train_grasp_GAN.metric_dict['test_success']
                 + Train_grasp_GAN.metric_dict['test_failure'])
            live.update(generate_combined_table(Train_grasp_GAN.metric_dict, i, steps, hand_name=hand_name))
            if i < steps:
                Train_grasp_GAN.begin(iterations=1)
            else:
                time.sleep(0.25)

if __name__ == "__main__":
    args = get_args()

    # Normalize filename (avoid config.ini.ini)
    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)


    if args.test_mode:
        run_test()
    else:
        train_N_grasp_GAN(args,n=10000)