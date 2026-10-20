import os
import random
import traceback
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from Configurations.config import device
from kinematic_utils.path_check import kinematic_checker
from utils.check_point_conventions import GANWrapper
from utils.domain_randomization import add_reflective_blob_noise, add_depth_noise
from utils.plot_utils import plot_distribution, plot_distribution_overlayed
from utils.rl.masked_categorical import MaskedCategorical
from utils.dynamic_dataset import DynamicDataManagement
from utils.training_satatistics import MovingRate, TrainingTracker
import numpy as np
import builtins

original_print = builtins.print

def custom_print(*args, **kwargs):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):
            formatted_args.append(f"{arg:.4f}")
        else:
            formatted_args.append(arg)
    original_print(*formatted_args, **kwargs)

builtins.print = custom_print


bce_with_logits = nn.BCEWithLogitsLoss()


def logits_to_probs(logits):
    return F.sigmoid(logits)

def visualize_depth_with_flat_index(depth, i):
    """
    depth: (H, W) depth map, e.g. (600, 600)
    i: index into depth.reshape(-1)
    """
    H, W = depth.shape

    # Convert flat index back to 2D index
    row, col = np.unravel_index(i, (H, W))

    plt.figure(figsize=(6, 6))
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth')

    # Highlight the selected point
    plt.scatter(col, row, c='red', s=80, marker='x')

    plt.title(f"Flat index {i} → (row={row}, col={col})")
    plt.axis('off')
    plt.show()


class AbstractGraspAgentTesting:
    def __init__(self, args, sampler_policy_model, critic_model, model_key='test',
                 randomization_unit=None,
                 process_pose=None,
                 n_joints=0,
                 track_statistics_history=False,
                 check_kinematics=False,
                 shake=False
                 , domain_randomization=False
                 , cip_fingers=None
                 , static_joints=None):

        self.args = args
        self.model_key = model_key
        self.max_n = 100
        self.domain_randomization = domain_randomization
        self.static_joints = [] if static_joints is None else static_joints

        self.view_env = args.view

        self.cip_fingers = cip_fingers

        self.shake = shake

        self.check_kinematics = check_kinematics

        self.kinematics = kinematic_checker() if check_kinematics else None

        self.sampler_policy_model = sampler_policy_model
        self.critic_model = critic_model

        '''hand specific fucntions'''
        self.randomization_unit = randomization_unit
        self.process_pose = process_pose
        self.sim_env = None

        ''''''
        self.track_statistics_history = track_statistics_history


        '''Moving rates'''
        self.skip_rate = None
        self.balanced_set_grasp_quality_statistics = None
        self.grasp_quality_statistics = None

        '''initialize statistics records'''
        self.sampler_loss_statistics = None
        self.critic_loss_statistics = None
        self.critic_loss_statistics = None

        self.n_param = n_joints + 8

        self.max_scenes = 1000

        self.DDM = DynamicDataManagement(key=self.model_key + '_synthesized_dynamic_data')

        self.skipped_last = True


        self.moving_std = torch.load(self.model_key + '_moving_std') if os.path.exists(
            self.model_key + '_moving_std') else None
        self.moving_range = torch.load(self.model_key + '_moving_range') if os.path.exists(
            self.model_key + '_moving_range') else None

        self.gan = GANWrapper(self.model_key, self.sampler_policy_model, self.critic_model)
        self.gan.ini_generator(train=False)

        torch.set_printoptions(
            precision=4,  # Number of decimal places
            sci_mode=False  # Disable scientific notation
        )
        np.set_printoptions(suppress=True, precision=4)


        self.metric_dict = {}
        self.metric_dict['test_success'] = 0
        self.metric_dict['test_failure'] = 0

    def initialize(self):

        '''Moving rates'''
        self.skip_rate = MovingRate(self.model_key + '_skip_rate',
                                    decay_rate=0.1,
                                    initial_val=1., track_history=self.track_statistics_history)

        self.Ave_uniquness = MovingRate(self.model_key + 'Ave_uniquness',
                                        decay_rate=0.01,
                                        initial_val=0., load_last=True, track_history=self.track_statistics_history)

        self.Ave_importance = MovingRate(self.model_key + '_Ave_importance',
                                         decay_rate=0.01,
                                         initial_val=0., load_last=True, track_history=self.track_statistics_history)

        self.learn_from_heurastic_rate = MovingRate(self.model_key + '_learn_from_heurastic_rate',
                                                    decay_rate=0.01,
                                                    initial_val=0., load_last=True,
                                                    track_history=self.track_statistics_history)

        self.dist_bias = MovingRate(self.model_key + '_dist_bias',
                                    decay_rate=0.01,
                                    initial_val=0., load_last=True, track_history=self.track_statistics_history)

        self.collision_tendency = MovingRate(self.model_key + '_collision_tendency',
                                             decay_rate=0.01,
                                             initial_val=0., load_last=True,
                                             track_history=self.track_statistics_history)

        self.data_update_rate = MovingRate(self.model_key + '_data_update_rate',
                                           decay_rate=0.01,
                                           initial_val=0., load_last=True, track_history=self.track_statistics_history)

        self.discrimination_dist = MovingRate(self.model_key + '_discrimination_dist',
                                              decay_rate=0.01,
                                              initial_val=0., load_last=True,
                                              track_history=self.track_statistics_history)

        self.confidence_indicator = MovingRate(self.model_key + '_confidence_indicator',
                                               decay_rate=0.01,
                                               initial_val=0., load_last=True,
                                               track_history=self.track_statistics_history)

        self.Ave_max_prop = MovingRate(self.model_key + '_Ave_max_prop',
                                       decay_rate=0.01,
                                       initial_val=1., load_last=True, track_history=self.track_statistics_history)

        self.random_sampler_acceptance_rate = MovingRate(self.model_key + '_random_sampler_acceptance_rate',
                                                         decay_rate=0.01,
                                                         initial_val=0., load_last=True,
                                                         track_history=self.track_statistics_history)

        self.buffer_replay_rate = MovingRate(self.model_key + '_buffer_replay_rate',
                                             decay_rate=0.01,
                                             initial_val=0., load_last=True,
                                             track_history=self.track_statistics_history)

        '''initialize statistics records'''
        self.balanced_set_grasp_quality_statistics = TrainingTracker(
            name=self.model_key + '_balanced_set_grasp_quality',
            track_label_balance=False, track_history=self.track_statistics_history)

        self.balanced_set_grasp_safety_statistics = TrainingTracker(name=self.model_key + '_balanced_set_grasp_safety',
                                                                    track_label_balance=False,
                                                                    track_history=self.track_statistics_history)

        self.cond_argmax_policy_statistics = TrainingTracker(name=self.model_key + '_cond_argmax_policy',
                                                             track_label_balance=False,
                                                             track_history=self.track_statistics_history)

        self.argmax_policy_statistics = TrainingTracker(name=self.model_key + '_argmax_policy',
                                                        track_label_balance=False,
                                                        track_history=self.track_statistics_history)

        self.sampler_loss_statistics = TrainingTracker(name=self.model_key + '_sampler_loss',
                                                       track_label_balance=False,
                                                       track_history=self.track_statistics_history)

        self.grasp_quality_statistics = TrainingTracker(name=self.model_key + '_grasp_quality',
                                                        track_label_balance=False, decay_rate=0.01,
                                                        track_history=self.track_statistics_history)

        self.grasp_safety_statistics = TrainingTracker(name=self.model_key + '_grasp_safety',
                                                       track_label_balance=False, decay_rate=0.01,
                                                       track_history=self.track_statistics_history)

        self.critic_loss_statistics = TrainingTracker(name=self.model_key + '_critic_loss',
                                                      track_label_balance=False,
                                                      track_history=self.track_statistics_history)

        self.metric_dict['uniqueness'] = self.Ave_uniquness.val
        self.metric_dict['collision'] = self.collision_tendency.val
        self.metric_dict['discrimination_dist'] = self.discrimination_dist.val
        self.metric_dict['confidence'] = self.confidence_indicator.val
        self.metric_dict['heuristic_efficiency'] = self.random_sampler_acceptance_rate.val
        self.metric_dict['replay_rate'] = self.learn_from_heurastic_rate.val

    def check_collision(self, target_point, target_pose, view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=False)
        return self.sim_env.check_collision(hand_pos=shifted_point.tolist(), hand_quat=quat, hand_fingers=fingers,
                                            view=view)

    def evaluate_grasp(self, target_point, target_pose, view=False, hard_level=0, shake=False, check_kinematics=False,
                       update_obj_prob=None):
        grasped_obj = None
        warning_flag = False
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=False)

            if view:
                grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp = self.sim_env.view_grasp(
                    hand_pos=shifted_point.tolist(), hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level)
                warning_flag = False
            else:
                grasp_success, contact_with_obj, contact_with_floor, warning_flag, grasped_obj = self.sim_env.check_graspness(
                    hand_pos=shifted_point.tolist(), hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level, shake=shake, update_obj_prob=update_obj_prob)

            initial_collision = contact_with_obj or contact_with_floor


            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor and not warning_flag:
                    if check_kinematics:
                        plan_found = self.kinematics.kinematic_plan_exist(quat, shifted_point)
                    else:
                        plan_found = True
                    return grasp_success, initial_collision, plan_found, grasped_obj, warning_flag

        return False, initial_collision, None, grasped_obj, warning_flag

    def iterate(self, pc, floor_mask, grasp_pose,
                                 grasp_quality):

        grasp_quality = grasp_quality[0, 0].reshape(-1)
        grasp_pose_PW = grasp_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
        selection_mask = (~floor_mask.clone()) & (grasp_quality > 0.5)

        selection_p = 0.001 + grasp_quality ** 2

        avaliable_iterations = selection_mask.sum()

        if avaliable_iterations < 3: return [], [], None

        n = int(min(self.max_n, avaliable_iterations))

        t = 0
        print('test')
        while t < n:
            t += 1
            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
            target_index = torch.argmax(dist.probs).item()

            selection_mask[target_index] *= False

            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = grasp_pose_PW[target_index]

            grasp_success, initial_collision, plan_found, grasped_obj, warning_flag = self.evaluate_grasp(
                target_point, target_generated_pose, view=self.view_env, shake=self.shake)
            print(f'{t}')
            if warning_flag : return None
            if  initial_collision: continue
            if grasp_success:
                self.metric_dict['test_success'] += 1
            else:
                self.metric_dict['test_failure'] += 1

            return None
        return None

    def step(self):
        self.sim_env.max_obj_per_scene = 10
        self.sim_env.remove_objects(n=len(self.sim_env.objects))
        n_objs = random.randint(self.args.min_obj, self.args.max_obj)

        self.sim_env.drop_new_obj(selected_index=None, stablize=True, n=n_objs,
                                  random_sampling=True)

        if len(self.sim_env.objects) == 0: return

        '''get scene perception'''
        depth, pc, floor_mask = self.sim_env.get_scene_preception(view=False)

        floor_mask = torch.from_numpy(floor_mask).to(device)
        clean_depth = torch.from_numpy(depth).to(device)  # [600.600]
        depth = torch.from_numpy(depth).to(device)  # [600.600]

        if self.domain_randomization:
            depth = add_reflective_blob_noise(clean_depth, n_blobs=np.random.randint(1, 10),
                                              blob_radius=np.random.random(), outlier_scale=0.02)
            depth = add_depth_noise(depth, keep_mask=floor_mask.reshape(600, 600), sigma=0.002)
            pc, _ = self.sim_env.depth_to_pointcloud(depth.cpu().numpy(), self.sim_env.intr, self.sim_env.extr)
        pc = torch.from_numpy(pc).to(device)

        grasp_pose, grasp_quality_logits, features2, features3, grasp_collision_logits = self.gan.generator(
            depth[None, None, ...], detach_sampler=True, detach_quality=True, detach_collision=True)

        grasp_quality = logits_to_probs(grasp_quality_logits)

        self.iterate(pc, floor_mask, grasp_pose,grasp_quality.detach() )


    def begin(self, iterations=10):
        with torch.no_grad():
            for i in range(iterations):
                if self.args.catch_exceptions:
                    try:
                        self.step()
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                        self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)
                else:
                    self.step()


    def show_overlaid_graphs(self, iterations=5, load_from_dataset=True):

        with torch.no_grad():

            grasp_quality_data = []

            for i in range(iterations):

                self.sim_env.max_obj_per_scene = 10

                if load_from_dataset:
                    self.sim_env.reload()

                else:
                    self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)

                    self.sim_env.drop_new_obj(selected_index=None, stablize=True,
                                              n=random.randint(5, self.sim_env.max_obj_per_scene))

                '''get scene perception'''
                depth, pc, floor_mask = self.sim_env.get_scene_preception(view=False)

                floor_mask = torch.from_numpy(floor_mask).to(device)

                depth = torch.from_numpy(depth).to(device)  # [600.600]

                with torch.no_grad():
                    self.gan.generator.eval()
                    grasp_pose, grasp_quality_logits, features2, features3, grasp_collision_logits = self.gan.generator(
                        depth[None, None, ...], detach_sampler=True, detach_quality=True, detach_collision=True)

                    grasp_quality = logits_to_probs(grasp_quality_logits)

                    # annealing_factor = (1 -     grasp_quality.detach()) #torch.ones_like(grasp_pose[:,0:1])
                    # grasp_pose_ref = self.pose_interpolation(grasp_pose,
                    #                                          annealing_factor=annealing_factor)
                    # standarized_depth_ = depth_normalization(depth[None, None, ...])
                    # gripper_pose_x = torch.cat([grasp_pose_ref, standarized_depth_], dim=1)
                    # grasp_quality_logits = self.gan.generator.grasp_quality_(features, gripper_pose_x)
                    # grasp_quality = logits_to_probs(grasp_quality_logits)

                    grasp_quality_c = grasp_quality[0, 0].reshape(-1).clone().detach()
                    grasp_quality_c = grasp_quality_c[~floor_mask]
                    grasp_quality_data.append(grasp_quality_c.numpy())

            if len(grasp_quality_data) == 1:
                plot_distribution(grasp_quality_data[0])
            else:
                plot_distribution_overlayed(grasp_quality_data, name='Scene')




