import os
from  model.CH_model import CH_model_key, CH_D, CH_G
from sim_dexee.Casia_hand_env import CasiaHandEnv
from test_board.abstract_test import AbstractGraspAgentTesting
from  training.sample_random_grasp import generate_random_CH_poses
from utils.args import get_args, read_config
from  training.CH_training import process_pose

class TrainGraspGAN(AbstractGraspAgentTesting):
    def __init__(self, args, epochs=1):

        super().__init__(args=args, sampler_policy_model=CH_G,critic_model=CH_D,model_key=CH_model_key,
                         randomization_unit=generate_random_CH_poses,
                         process_pose=process_pose,n_joints=3,check_kinematics=False,
                         domain_randomization=False)

        self.sim_env = CasiaHandEnv(root=os.getcwd() + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)
        # self.sim_env.plt_obj_dict_statistics()



def  run_test():
    from rich.live import Live
    from utils.test_board import generate_combined_table
    import time
    Train_grasp_GAN = TrainGraspGAN(args)
    Train_grasp_GAN.initialize()
    hand_name = 'CasiaHand'
    Train_grasp_GAN.metric_dict['test_success'] = 43
    Train_grasp_GAN.metric_dict['test_failure'] = 1
    i=0
    steps=args.steps
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

    run_test()
