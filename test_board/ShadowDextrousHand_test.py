import os
from torch import nn
from  model.SH_5F_model import SH_model_key, SH_G, SH_D
from  sim_dexee.Shadow_hand_five_fingers_env import ShadowHandEnv
from test_board.abstract_test import AbstractGraspAgentTesting
from training.ShadoDextrousHand_training import process_pose
from  training.sample_random_grasp import  generate_random_SH_5F_poses
from utils. IO_utils import custom_print
from utils.args import get_args, read_config

bce_loss = nn.BCELoss()
bce_with_logits = nn.BCEWithLogitsLoss()


print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


class TrainGraspGAN(AbstractGraspAgentTesting):
    def __init__(self, args,  epochs=1 ):

        super().__init__(args=args,sampler_policy_model=SH_G,critic_model=SH_D, model_key=SH_model_key,
                         randomization_unit=generate_random_SH_5F_poses,
                         process_pose=process_pose,n_joints=20,
                         static_joints=[0,1])

        root_dir = os.getcwd()  # current working directory

        self.sim_env = ShadowHandEnv(root=root_dir + "/sim_dexee/hands_and_objects/", max_obj_per_scene=10)
        # self.sim_env.plt_obj_dict_statistics()



def  run_test():
    from rich.live import Live
    from utils.test_board import generate_combined_table
    import time
    test_module = TrainGraspGAN(args)
    test_module.initialize()
    hand_name = 'Shadow Dexterous Hand'
    i=0
    steps=args.steps
    with Live(generate_combined_table(test_module.metric_dict,i,steps,hand_name=hand_name), refresh_per_second=4, screen=True) as live:
        while True:
            i = (test_module.metric_dict['test_success']
                 + test_module.metric_dict['test_failure'])
            live.update(generate_combined_table(test_module.metric_dict, i, steps, hand_name=hand_name))
            if i < steps:
                test_module.begin(iterations=1)
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
