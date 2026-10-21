import os
from  model.allergo_model import Allergo_model_key, Allergo_G, Allergo_D
from  sim_dexee.allegro_hand_env import AllegroHandEnv
from test_board.abstract_test import AbstractGraspAgentTesting
from training.AllergoHand_training import cip_fingers, process_pose
from  training.sample_random_grasp import generate_random_Allergo_poses
from utils.args import get_args, read_config


class TrainGraspGAN(AbstractGraspAgentTesting):
    def __init__(self, args, epochs=1):

        super().__init__(args=args,sampler_policy_model=Allergo_G,critic_model=Allergo_D,model_key=Allergo_model_key,
                         randomization_unit=generate_random_Allergo_poses,cip_fingers=cip_fingers,
                         process_pose=process_pose,n_joints=16)

        self.sim_env = AllegroHandEnv(root=os.getcwd() + "/sim_dexee/hands_and_objects/",max_obj_per_scene=10)
        # self.sim_env.plt_obj_dict_statistics()


def  run_test():
    from rich.live import Live
    from utils.test_board import generate_combined_table
    import time
    test_module = TrainGraspGAN(args)
    test_module.initialize()
    hand_name = 'Allegro Hand V4'
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

    # Read config
    config = read_config(config_path)

    run_test()
