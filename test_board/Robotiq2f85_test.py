import os
from model.r_2f85_model import R_2F85_D, R_2F85_G, R_2F85_model_key
from sim_dexee.r_2f85_env import R2F85Env
from test_board.abstract_test import AbstractGraspAgentTesting
from training.Robotiq2f85_training import process_pose
from  training.sample_random_grasp import generate_random_r_2f85_poses
from utils.args import get_args, read_config


class TrainGraspGAN(AbstractGraspAgentTesting):
    def __init__(self, args,epochs=1):

        super().__init__(args=args,sampler_policy_model=R_2F85_G,critic_model=R_2F85_D, model_key=R_2F85_model_key,
                         randomization_unit=generate_random_r_2f85_poses,
                         process_pose=process_pose,n_joints=0)

        self.sim_env = R2F85Env(root=os.getcwd() + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)
        # self.sim_env.plt_obj_dict_statistics()



def  run_test():
    from rich.live import Live
    from utils.test_board import generate_combined_table
    import time
    test_module = TrainGraspGAN(args)
    test_module.initialize()
    hand_name = 'Robotiq 2F85'
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

    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    config = read_config(config_path)

    run_test()
