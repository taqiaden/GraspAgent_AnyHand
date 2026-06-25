import os
import random
import time
import traceback
from collections import deque
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from Configurations.config import device
from kinematic_utils.path_check import kinematic_checker
from model.abstract_model import depth_normalization
from  utils.Voxel_operations import crop_cube, view_3d_occupancy_grid
from utils.check_point_conventions import GANWrapper
from utils.cuda_utils import cuda_memory_report
from utils.domain_randomization import add_reflective_blob_noise, add_depth_noise
from utils.plot_utils import plot_distribution, plot_distribution_overlayed
from utils.report_utils import progress_indicator
from utils.rl.masked_categorical import MaskedCategorical
from  utils.Online_clustering import OnlingClustering
from  utils.dynamic_dataset import DynamicDataManagement, SynthesisedData
from utils.training_satatistics import MovingRate, TrainingTracker
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import numpy as np
from utils.visualiztion import view_npy_open3d

print_details=True

bce_with_logits=nn.BCEWithLogitsLoss()

def hinge_loss(positive, negative, margin, k=1.):
    loss = torch.clamp((negative.squeeze() - positive.squeeze()) + margin * k, 0.)
    return loss

def c_loss(pred, label):
    loss=bce_with_logits(pred,label)
    return loss

def logits_to_probs(logits):
    return F.sigmoid(logits)

def weighted_scatter_loss(x, weights,eps=1e-6):

    N, M = x.shape

    if N > 1000:
        idx = torch.randperm(N, device=x.device)[:1000]
        x = x[idx]
        weights=weights[idx]


    weights=weights/(weights.sum()+eps)

    loss=((x[:,5:]*weights[:,None]).sum(dim=0)**2).mean()#/(M-5)


    weights=weights[:,None]*weights[None,:]

    diff=x[:,None,:]-x[None,:,:]

    dist=diff.abs()

    weighted_dif= weights[:,:,None] * (1-dist).clamp(min=0.)**2

    loss+=weighted_dif.sum()/(x.shape[1]+eps)

    return loss



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

class AbstractGraspAgentTraining:
    def __init__(self, args,sampler_policy_model,critic_model,  epochs=1 ,model_key='test',
                 test_mode=False,
                 randomization_unit=None,
                 process_pose=None,
                 n_param=1,
                 track_statistics_history=False,
                 check_kinematics=False,
                 exclude_collision_from_grasp_quality=True,
                 shake=False,
                 force_balance=True
                 ,train_policy_only=False
                 ,domain_randomization=False):

        self.args = args
        self.model_key=model_key
        self.test_mode=test_mode
        self.max_n=5 if test_mode else 30
        self.train_policy_only=train_policy_only
        self.view=False
        self.synthesizie_only=False
        self.domain_randomization=domain_randomization

        self.force_balance=force_balance

        self.shake=shake

        self.exclude_collision_from_grasp_quality=exclude_collision_from_grasp_quality

        self.activate_grad_clipping=False

        self.check_kinematics=check_kinematics

        self.kinematics = kinematic_checker() if check_kinematics else None


        self.sampler_policy_model=sampler_policy_model
        self.critic_model=critic_model

        '''hand specific fucntions'''
        self.randomization_unit=randomization_unit
        self.process_pose=process_pose
        self.sim_env=None

        ''''''
        self.track_statistics_history=track_statistics_history

        
        self.batch_size = 2

        self.iter_per_scene = 1

        self.epochs = epochs

        '''Moving rates'''
        self.skip_rate = None
        self.balanced_set_grasp_quality_statistics = None
        self.balanced_set_collision_statistics = None
        self.grasp_quality_statistics = None

        '''initialize statistics records'''
        self.sampler_loss_statistics = None
        self.critic_loss_statistics = None
        self.critic_loss_statistics = None


        self.n_param = n_param

        self.max_scenes = 1000


        self.DDM = DynamicDataManagement(key=self.model_key + '_synthesized_dynamic_data')

        self.loaded_synthesised_data = None

        self.skipped_last=True

        approach_centers = torch.tensor([[0., 1., 0],[0., -1., 0],[1., 0, 0],[-1., 0, 0],[0., 0, -1]], device=device)
        beta_centers=torch.tensor([[0., 1],[0., -1],[1., 0],[-1., 0]], device=device)
        # Repeat approach_centers n_beta times
        alpha_repeated = approach_centers.repeat_interleave(beta_centers.shape[0] , dim=0)  # (20, 3)
        # Tile beta_centers n_alpha times
        beta_tiled = beta_centers.repeat(approach_centers.shape[0] , 1)  # (20, 2)
        # Concatenate along dimension 1
        alpha_beta = torch.cat([alpha_repeated, beta_tiled], dim=1)  # (20, 5)




        self.moving_std=torch.load( self.model_key + '_moving_std') if os.path.exists(self.model_key + '_moving_std') else None
        self.moving_range=torch.load(self.model_key + '_moving_range') if os.path.exists(self.model_key + '_moving_range') else None

        if  torch.isnan(self.moving_std).any():self.moving_std=None
        if  torch.isnan(self.moving_range).any():self.moving_range=None




        self.approach_beta_clusters=OnlingClustering(key_name=self.model_key+'_approach_beta_clusters',number_of_centers=8,vector_size=5,decay_rate=0.1,use_euclidean_dist=False,static_centers=alpha_beta)

        self.gan = GANWrapper(self.model_key, self.sampler_policy_model, self.critic_model)
        self.gan.ini_models(train=True)

        if not self.test_mode:
            self.load_optimizers()
            


    def initialize(self):

        '''Moving rates'''
        self.skip_rate = MovingRate(self.model_key + '_skip_rate',
                                    decay_rate=0.1,
                                    initial_val=1.,track_history=self.track_statistics_history)



        self.Ave_uniquness = MovingRate(self.model_key + 'Ave_uniquness',
                                                       decay_rate=0.01,
                                                       initial_val=0.,load_last=True,track_history=self.track_statistics_history)

        self.discrimination_dist=MovingRate(self.model_key + '_discrimination_dist',
                                                       decay_rate=0.01,
                                                       initial_val=0.,load_last=True,track_history=self.track_statistics_history)

        self.confidence_indicator = MovingRate(self.model_key + '_confidence_indicator',
                                                       decay_rate=0.01,
                                                       initial_val=0.,load_last=True,track_history=self.track_statistics_history)

        self.random_sampler_acceptance_rate = MovingRate(self.model_key + '_random_sampler_acceptance_rate',
                                                       decay_rate=0.01,
                                                       initial_val=0.,load_last=True,track_history=self.track_statistics_history)

        '''initialize statistics records'''
        self.balanced_set_grasp_quality_statistics = TrainingTracker(name=self.model_key + '_balanced_set_grasp_quality',
                                                            track_label_balance=False,track_history=self.track_statistics_history)
        self.argmax_grasp_quality_statistics = TrainingTracker(name=self.model_key + '_argmax_grasp_quality',
                                                            track_label_balance=False,track_history=self.track_statistics_history)



        self.balanced_set_collision_statistics = TrainingTracker(name=self.model_key + '_balanced_set_collision',
                                                    track_label_balance=False, decay_rate=0.01,track_history=self.track_statistics_history)

        self.sampler_loss_statistics = TrainingTracker(name=self.model_key + '_sampler_loss',
                                                          track_label_balance=False,track_history=self.track_statistics_history)

        self.grasp_quality_statistics = TrainingTracker(name=self.model_key + '_grasp_quality',
                                                        track_label_balance=False, decay_rate=0.01,track_history=self.track_statistics_history)

        self.critic_loss_statistics = TrainingTracker(name=self.model_key + '_critic_loss',
                                                 track_label_balance=False,track_history=self.track_statistics_history)




    def load_optimizers(self):
        print(f'Load optimizers')
        '''load  models'''

        # gan.generator.back_bone2_.apply(init_weights_he_normal)
        # gan.generator.grasp_quality_.apply(init_weights_he_normal)

        # gan.critic.back_bone.apply(gan_init_with_norms)
        # gan.critic.decoder.apply(gan_init_with_norms)

        sampler_params = []
        sampler_params += list(self.gan.generator.back_bone.parameters())
        sampler_params += list(self.gan.generator.PoseSampler.parameters())

        policy_params = []
        policy_params += list(self.gan.generator.back_bone2_.parameters())
        policy_params += list(self.gan.generator.grasp_quality_.parameters())
        policy_params += list(self.gan.generator.collision.parameters())

        self.gan.critic_adam_optimizer(learning_rate=self.args.lr, beta1=0.9, beta2=0.999,weight_decay_=0.)
        # self.gan.critic_sgd_optimizer(learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        # self.gan.generator_adam_optimizer(param_group=policy_params,learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        self.gan.generator_sgd_optimizer(param_group=policy_params,learning_rate=self.args.lr*10,momentum=0.)
        self.gan.sampler_optimizer = torch.optim.SGD(sampler_params, lr=self.args.lr*10,
                                               momentum=0)
        # self.gan.sampler_adam_optimizer(param_group=sampler_params,learning_rate=self.args.lr,beta1=0.9, beta2=0.999,weight_decay_=0.)

        # gan.sampler_optimizer =torch.optim.Adam(sampler_params, lr=self.args.lr   )

    def pose_interpolation(self, gripper_pose, annealing_factor):
        ref_pose = gripper_pose.detach().clone()
        n = ref_pose.shape[1]
        assert ref_pose.shape[0] == 1

        if torch.isnan(ref_pose).any():
            print(f'ref_pose is nan: {ref_pose}')
            exit()

        sampling_ratios=annealing_factor

        # if len(self.DDM)<self.max_scenes:
        # sampling_ratios = torch.where(annealing_factor > 0.5 , torch.tensor(1.0), sampling_ratios)
        # else:
        #     sampling_ratios = torch.where(annealing_factor > 0.85 , torch.tensor(1.0), sampling_ratios)

        sampled_pose = self.randomization_unit(ref_pose[0, 0].numel()).reshape(600, 600, n).permute(2, 0, 1)[
            None, ...]

        sampled_pose = sampled_pose * sampling_ratios + (1 - sampling_ratios) * ref_pose
        assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios.min()}, {sampled_pose.max()}'

        sampled_pose[:, 0:3] = F.normalize(sampled_pose[:, 0:3], dim=1)
        sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)

        return sampled_pose

    def step_discriminator(self, cropped_local_point_clouds, depth,  grasp_pose, grasp_pose_ref, pairs):

        '''zero grad'''
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        '''self supervised critic learning'''
        with torch.no_grad():

            generated_grasps_stack = []
            for pair in pairs:
                index = pair[0]

                pred_pose = grasp_pose[index]
                label_pose = grasp_pose_ref[index]
                pair_pose = torch.stack([pred_pose, label_pose])
                generated_grasps_stack.append(pair_pose)
            generated_grasps_stack = torch.stack(generated_grasps_stack)

        scores = self.gan.critic( generated_grasps_stack,  cropped_local_point_clouds)


        generated_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin = pairs[j][2]

            assert margin >= 0 , f'margin=,{margin}'

            if k > 0:
                loss += (hinge_loss(positive=ref_scores[j], negative=generated_scores[j],
                                    margin=margin)) / self.batch_size

            else:
                loss += (hinge_loss(positive=generated_scores[j], negative=ref_scores[j],
                                    margin=margin) ) / self.batch_size

        loss.backward()

        self.critic_loss_statistics.loss = loss.item()

        if self.activate_grad_clipping: self.critic_gradient_clipping()

        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        if print_details:print(Fore.LIGHTYELLOW_EX, f'd_loss={loss.item()}',
              Fore.RESET)

    def critic_gradient_clipping(self):
        '''GRADIENT CLIPPING'''
        params = list(self.gan.critic.back_bone.parameters())
        backbone_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        params = list(self.gan.critic.decoder.parameters())

        decoder_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        norm = torch.nn.utils.clip_grad_norm_(self.gan.critic.parameters(), max_norm=1.0)
        if print_details: print(Fore.LIGHTGREEN_EX, f' C  norm : {norm}, backbone_norm:{backbone_norm}, decoder_norm={decoder_norm}',
              Fore.RESET)

    def print_pairs_info(self, pairs, grasp_pose, grasp_pose_ref):
        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin = pairs[j][2]

            target_generated_pose = grasp_pose[target_index].detach()
            target_ref_pose = grasp_pose_ref[target_index].detach()

            if k < 0:
                print(Fore.GREEN,
                      f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',
                      Fore.RESET)
            elif k > 0:
                print(Fore.LIGHTCYAN_EX,
                      f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',
                      Fore.RESET)

    def get_generator_loss(self, cropped_local_point_clouds, depth, clean_depth, grasp_pose, grasp_pose_ref, pairs, floor_mask    ):

        grasp_pose = grasp_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
        grasp_pose_ref = grasp_pose_ref[0].permute(1, 2, 0).reshape(360000, self.n_param)

        generated_grasps_stack = []
        for pair in pairs:
            index = pair[0]
            pred_pose = grasp_pose[index]

            label_pose = grasp_pose_ref[index]
            pair_pose = torch.stack([pred_pose, label_pose])
            generated_grasps_stack.append(pair_pose)

        generated_grasps_stack = torch.stack(generated_grasps_stack)

        scores = self.gan.critic( generated_grasps_stack, cropped_local_point_clouds,detach_backbone=True)

        gen_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            margin=pairs[j][2]
            loss += (hinge_loss(positive=gen_scores[j], negative=ref_scores[j], margin=margin) ) / self.batch_size

        return loss

    def supplemetary_statistics(self,probs,pc,grasp_pose_PW,floor_mask):
        try:
            for l in range(10):
                dist = MaskedCategorical(probs=probs.clamp(min=0.1),mask=(~floor_mask)&(probs>0.5))
                grasp_target_index = dist.probs.argmax()
                grasp_target_point = pc[grasp_target_index]
                grasp_prediction_ = probs[grasp_target_index].squeeze().clone()

                probs[grasp_target_index]=float('-inf')

                grasp_target_pose = grasp_pose_PW[grasp_target_index].detach()
                grasp_success, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, plan_found, grasped_obj = self.evaluate_grasp(
                    grasp_target_point, grasp_target_pose, view=False,
                    shake=self.shake, check_kinematics=False,
                    update_obj_prob=None)

                if not initial_collision:
                    label = torch.ones_like(grasp_prediction_) if grasp_success else torch.zeros_like(
                        grasp_prediction_)
                    self.argmax_grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                                      grasp_prediction_.detach())
                    break



        except Exception as e:
            print(Fore.RED, f'Error track statistics: {str(e)}',Fore.RESET)

    def get_repulsive_loss(self,depth,grasp_pose,features,floor_mask):
        standarized_depth_=depth_normalization(depth[None,None,...])

        gripper_pose_x = torch.cat([grasp_pose, standarized_depth_], dim=1)

        grasp_quality_x = self.gan.generator.grasp_quality_(features, gripper_pose_x)
        grasp_quality_x = grasp_quality_x[0, 0].reshape(-1)

        grasp_quality_obj_x = grasp_quality_x[~floor_mask]

        grasp_quality_obj_x=logits_to_probs(grasp_quality_obj_x)

        high_quality = grasp_quality_obj_x[grasp_quality_obj_x >= 0.5]
        low_quality = grasp_quality_obj_x[grasp_quality_obj_x < 0.5]

        # Only update if both groups have elements
        if len(high_quality) > 0 and len(low_quality) > 0:
            self.discrimination_dist.update(
                high_quality.mean().item() - low_quality.mean().item()
            )

            self.confidence_indicator.update(high_quality.mean().item() )

        loss_p = ((torch.clamp(1.0- high_quality, min=0.)*2)**2).mean()
        loss_n = ((torch.clamp(low_quality, min=0.)*2)**2).mean()

        print(f'loss_p: {loss_p.item()},  loss_n: {loss_n.item()}')

        return loss_p

    def step_policy(self, cropped_local_point_clouds, depth, clean_depth, floor_mask, pc, grasp_pose_ref, pairs     ):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)

        '''generated grasps'''
        grasp_pose, grasp_quality_logits ,features,grasp_collision_logits= self.gan.generator(
            depth[None, None, ...])

        grasp_pose_PW = grasp_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
        grasp_quality_logits = grasp_quality_logits[0, 0].reshape(-1)
        grasp_collision_logits = grasp_collision_logits[0, 0].reshape(-1)
        probs = logits_to_probs(grasp_quality_logits)

        self.supplemetary_statistics( probs.clone().detach(), pc, grasp_pose_PW,floor_mask)

        grasp_quality_loss_=self.get_grasp_quality_loss(probs,grasp_quality_logits,floor_mask,pc,grasp_pose_PW,random_sampling=False)

        floor_quality_loss=probs[floor_mask].clamp(min=0).mean()

        collision_loss_=self.get_grasp_collision_loss(probs, grasp_collision_logits, floor_mask, pc, grasp_pose_PW,random_sampling=True)


        policy_loss =    grasp_quality_loss_ +collision_loss_+floor_quality_loss
        policy_loss.backward()
        if self.activate_grad_clipping: self.policy_gradient_clipping()
        self.gan.generator_optimizer.step()
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)

        scatter_loss=torch.tensor([0.],device=device)
        grasp_sampling_loss=torch.tensor([0.],device=device)
        contrast_loss=torch.tensor([0.],device=device)
        if len(pairs) == self.batch_size:
            grasp_sampling_loss = self.get_generator_loss(cropped_local_point_clouds,
                                                            depth, clean_depth, grasp_pose, grasp_pose_ref,
                                                            pairs, floor_mask)

            assert not torch.isnan(grasp_sampling_loss).any(), f'{grasp_sampling_loss}'

            weight=(1-logits_to_probs(grasp_quality_logits[~floor_mask]).detach())**2

            scatter_loss = weighted_scatter_loss(grasp_pose.reshape(self.n_param, -1).permute(1, 0)[~floor_mask],weights=weight) if len(
                pairs) == self.batch_size else torch.tensor(
                [0.], device=grasp_pose.device)
            contrast_loss=self.get_repulsive_loss( depth, grasp_pose, features.detach(), floor_mask)
            with torch.no_grad():
                self.sampler_loss_statistics.loss = grasp_sampling_loss.item()

            sampler_loss = grasp_sampling_loss   + scatter_loss+contrast_loss
            sampler_loss.backward()
            if self.activate_grad_clipping: self.policy_gradient_clipping()
            self.gan.sampler_optimizer.step()

        if print_details: print(Fore.LIGHTYELLOW_EX,
              f'grasp_sampling_loss={grasp_sampling_loss.item()}, floor_quality_loss={floor_quality_loss.item()}, grasp_quality_loss_={grasp_quality_loss_.item()}, scatter_loss={scatter_loss.item()}, contrast_loss={contrast_loss.item()}, contrast_loss={contrast_loss.item()}',
              Fore.RESET)


        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)

    def get_grasp_quality_loss(self,probs,grasp_quality_logits,floor_mask,pc,grasp_pose_PW,random_sampling=False):

        grasp_quality_loss_ = torch.tensor(0., device=device)

        start = time.time()
        positive_counter = 0
        negative_counter = 0
        n = 2
        s = int(n / 2)
        for k in range(n):
            '''grasp quality'''
            while True:
                # if random_sampling:
                dist = MaskedCategorical(probs=torch.rand_like(probs), mask=~floor_mask)
                # else:
                #     dist = MaskedCategorical(probs=probs.clamp(min=0.1), mask=~floor_mask)
                grasp_target_index = dist.sample()

                grasp_target_point = pc[grasp_target_index]
                grasp_prediction_logits = grasp_quality_logits[grasp_target_index].squeeze()
                grasp_target_pose = grasp_pose_PW[grasp_target_index].detach()

                grasp_success, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, plan_found, grasped_obj = self.evaluate_grasp(
                    grasp_target_point, grasp_target_pose, view=False,
                    shake=self.shake, check_kinematics=False,
                    update_obj_prob=None)

                if warning_flag: continue
                if time.time() - start > 5 * s or self.skip_rate.val > 0.9:
                    # print(Fore.RED, f'quality policy exploration timeout', Fore.RESET)
                    break
                if initial_collision and self.exclude_collision_from_grasp_quality:continue
                label = torch.ones_like(grasp_prediction_logits) if grasp_success else torch.zeros_like(grasp_prediction_logits)
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       logits_to_probs(grasp_prediction_logits.detach()))

                if self.force_balance:
                    if grasp_success and positive_counter >= s: continue
                    if (not grasp_success) and negative_counter >= s: continue

                if not random_sampling:
                    u = self.approach_beta_clusters.get_uniqueness_score(grasp_target_pose[0:5]).item()
                    not_unique = self.Ave_uniquness.lower_rejection_criteria(u, k=2.,
                                                                               report=False)
                    if (logits_to_probs(grasp_prediction_logits)>np.random.rand() or not_unique) and  grasp_success:continue
                    if logits_to_probs(grasp_prediction_logits)<np.random.rand() and not grasp_success:continue

                break
            if grasp_success:
                positive_counter += 1
            else:
                negative_counter += 1
            label = torch.ones_like(grasp_prediction_logits) if grasp_success else torch.zeros_like(grasp_prediction_logits)
            grasp_quality_loss = c_loss(grasp_prediction_logits, label)

            with torch.no_grad():
                self.balanced_set_grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.balanced_set_grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                           logits_to_probs(
                                                                               grasp_prediction_logits.detach()))

            grasp_quality_loss_ += grasp_quality_loss / n

        return grasp_quality_loss_

    def get_grasp_collision_loss(self,probs,grasp_collision_logits,floor_mask,pc,grasp_pose_PW,random_sampling=False):
        grasp_collision_loss_ = torch.tensor(0., device=device)

        start = time.time()
        positive_counter = 0
        negative_counter = 0
        n = 2
        s = int(n / 2)
        for k in range(n):
            '''grasp quality'''
            while True:
                if random_sampling:
                    dist = MaskedCategorical(probs=torch.rand_like(probs), mask=~floor_mask)
                else:
                    dist = MaskedCategorical(probs=probs.clamp(min=0.1), mask=~floor_mask)
                grasp_target_index = dist.sample()

                grasp_target_point = pc[grasp_target_index]
                grasp_prediction_logits = grasp_collision_logits[grasp_target_index].squeeze()
                grasp_target_pose = grasp_pose_PW[grasp_target_index].detach()
                
                contact_with_obj , contact_with_floor=self.check_collision(grasp_target_point, grasp_target_pose)
                collision = contact_with_obj and contact_with_floor

                if time.time() - start > 5 * s or self.skip_rate.val > 0.9:
                    break

                if collision and positive_counter >= s: continue
                if (not collision) and negative_counter >= s: continue

                break
            if collision:
                positive_counter += 1
            else:
                negative_counter += 1
            label = torch.ones_like(grasp_prediction_logits) if collision else torch.zeros_like(grasp_prediction_logits)
            grasp_collision_loss = c_loss(grasp_prediction_logits, label)


            with torch.no_grad():
                self.balanced_set_collision_statistics.loss = grasp_collision_loss.item()
                self.balanced_set_collision_statistics.update_confession_matrix(label.detach(),
                                                                           logits_to_probs(
                                                                               grasp_prediction_logits.detach()))
                
            grasp_collision_loss_ += grasp_collision_loss / n

        return grasp_collision_loss_

    def policy_gradient_clipping(self):
        '''GRADIENT CLIPPING'''
        norm1 = torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone.parameters(), max_norm=float('inf'))
        norm2 = torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone2_.parameters(), max_norm=float('inf'))
        # norm3=torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone3_.parameters(), self.max_norm=float('inf'))
        norm = torch.nn.utils.clip_grad_norm_(self.gan.generator.parameters(), max_norm=5.0)


        if print_details:print(Fore.LIGHTGREEN_EX, f' G norm : {norm}, backbone1:{norm1}, backbone2: {norm2}, ', Fore.RESET)

    def check_collision(self, target_point, target_pose, view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=view)

        return self.sim_env.check_collision(hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers, view=view)

    def evaluate_grasp(self, target_point, target_pose, view=False, hard_level=0, shake=False, check_kinematics=False,
                       update_obj_prob=None):
        grasped_obj = None
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=self.test_mode)

            if view:
                in_scope, grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp = self.sim_env.view_grasp(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level)
                warning_flag = False
            else:
                in_scope, grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp, warning_flag, grasped_obj = self.sim_env.check_graspness(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level, shake=shake, update_obj_prob=update_obj_prob)

            initial_collision = contact_with_obj or contact_with_floor

            if warning_flag and print_details: print(Fore.RED, f' ----------------------------- warning_flag', Fore.RESET)


            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    if check_kinematics:
                        plan_found = self.kinematics.kinematic_plan_exist(quat, shifted_point)
                    else: plan_found=True
                    return grasp_success, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, plan_found, grasped_obj

        return False, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, None, grasped_obj

    def sample_contrastive_pairs(self, pc, floor_mask, grasp_pose, grasp_pose_ref,
                                 grasp_quality ):
        start = time.time()

        d_pairs = []
        g_pairs = []
        all_pairs = []

        selection_mask = (~floor_mask)
        grasp_quality = grasp_quality[0, 0].reshape(-1)
        grasp_pose_PW = grasp_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
        clipped_grasp_pose_PW = grasp_pose_PW.clone()
        clipped_grasp_pose_PW[:, 5:5 + 3] = torch.clip(clipped_grasp_pose_PW[:, 5:5 + 3], 0, 1)
        grasp_pose_ref_PW = grasp_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)

        selection_p =torch.rand_like(grasp_quality) #if self.loaded_synthesised_data is None else grasp_quality
        if self.test_mode: selection_p = 0.001  + grasp_quality ** 2

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations < 3: return [], [], None

        n = int(min(self.max_n, avaliable_iterations))

        if print_details:print(Fore.LIGHTBLACK_EX, '# Available candidates =', avaliable_iterations.item(), Fore.RESET)

        counter = 0
        sampler_samples = 0
        sampled_obj_ids = []

        t = 0
        while t < n:
            time_out = time.time() - start
            if time_out > 5 and not self.test_mode: break
            t += 1

            importance = None
            if self.loaded_synthesised_data is not None and len(self.loaded_synthesised_data) > 0 :
                target_index, _, _, importance, _ = self.loaded_synthesised_data.sample_pop()
            else:
                dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
                target_index = torch.argmax(dist.probs).item()

            selection_mask[target_index] *= False

            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = grasp_pose_PW[target_index]
            target_ref_pose = grasp_pose_ref_PW[target_index]

            if self.test_mode:
                contact_with_obj, contact_with_floor = self.check_collision(target_point, target_ref_pose,
                                                                            view=False)
                if contact_with_obj or contact_with_floor: continue

                view_r = self.evaluate_grasp(
                    target_point, target_ref_pose, view=True, shake=self.shake)
                if print_details:print(Fore.LIGHTCYAN_EX,
                      f'return f1: {view_r}, quality_score: {grasp_quality[target_index].item()}, max score={grasp_quality.max().item()}')

                g_pairs.append((target_index, 1, 1))
                d_pairs.append((target_index, 1, 1))
                return d_pairs, g_pairs, 1

            ref_success, ref_initial_collision, ref_n_grasp_contact, ref_self_collide, stable_ref_grasp, warning_flag, ref_plan_found, ref_grasped_obj = self.evaluate_grasp(
                target_point, target_ref_pose, view=False, shake=self.shake, update_obj_prob=None, check_kinematics=self.check_kinematics)


            if self.loaded_synthesised_data is None :self.random_sampler_acceptance_rate.update(ref_success)

            if warning_flag:
                break
            gen_success, gen_initial_collision, gen_n_grasp_contact, gen_self_collide, stable_gen_grasp, warning_flag, gen_plan_found, gen_grasped_obj = self.evaluate_grasp(
                target_point, target_generated_pose, view=False, shake=self.shake, check_kinematics=self.check_kinematics,
                update_obj_prob=grasp_quality[target_index].item() if self.loaded_synthesised_data is None  else None)

            if self.check_kinematics:
                ref_success=ref_success and ref_plan_found
                gen_success=gen_success and gen_plan_found

            if warning_flag:
                break

            if t == 1 and self.skip_rate() > 0.9 and print_details:
                print(
                    f' ref ---- {target_ref_pose}, {ref_success, ref_initial_collision, ref_n_grasp_contact, ref_self_collide}')
                print()
                print(
                    f' gen ---- {target_generated_pose}, {gen_success, gen_initial_collision, gen_n_grasp_contact, gen_self_collide}')

            if gen_success:
                u = self.approach_beta_clusters.get_uniqueness_score(grasp_pose_PW[target_index][0:5]).item()
                u=min(u,0.99)

                importance = max(0.01,
                                 grasp_quality[target_index].item()*u)
                all_pairs.append(
                    (target_index, target_point, grasp_pose_PW[target_index], importance, gen_grasped_obj))

                if self.Ave_uniquness.lower_rejection_criteria(u, k=2.,report=False): continue

            elif ref_success:
                u = self.approach_beta_clusters.get_uniqueness_score(grasp_pose_ref_PW[target_index][0:5]).item()
                u=min(u,0.99)
                # if (importance is not None and importance>0.1) or len(self.DDM)<self.max_scenes:
                importance = u*importance if importance is not None else min(0.5,max(0.01,u*(1-grasp_quality[target_index].item())))
                # if importance>0.1:
                all_pairs.append(
                    (target_index, target_point, grasp_pose_ref_PW[target_index], importance, ref_grasped_obj))

                if self.Ave_uniquness.lower_rejection_criteria(u, k=2.,report=False): continue

                if ref_grasped_obj in sampled_obj_ids:
                    if len(self.loaded_synthesised_data) > 0: continue
                else:
                    sampled_obj_ids.append(sampled_obj_ids)

            if not ref_success and not gen_success:
                if self.loaded_synthesised_data is None: self.sim_env.update_obj_info(1e-2, decay=0.99)
                continue
            elif ref_success and not gen_success:
                k=1
            elif gen_success and not ref_success:
                k=-1
            else:
                continue

            if k == 1:
                sampler_samples += 1

            counter += 1
            t = 0
            hh = (counter / self.batch_size) ** 2
            n = int(min(hh * self.max_n + n, avaliable_iterations))

            if len(d_pairs) < self.batch_size and  (ref_success ^ gen_success ):
                # if (importance > 0.1) or (self.skip_rate.val > 0.5):
                margin = 0 if ref_initial_collision or gen_initial_collision else  ((1-(0.5-  grasp_quality[target_index]).abs().item()*2)**2 if k>0 else ((0.5-  grasp_quality[target_index]).abs().item()*2)**2)

                d_pairs.append((target_index, k, margin,  target_point))

                # if (self.loaded_synthesised_data is None or len(self.loaded_synthesised_data) == 0):
                    # superior_pose = target_ref_pose if k > 0 else target_generated_pose
                self.approach_beta_clusters.update(target_generated_pose[0:5].detach().clone(),influence_factor=grasp_quality[target_index].item())

            if len(g_pairs) < self.batch_size and ref_success and not gen_success:

                margin =  0 if ref_initial_collision or gen_initial_collision else ((0.5-  grasp_quality[target_index]).abs().item()*2)**2

                g_pairs.append((target_index, k, margin, target_point))

            if len(d_pairs) == self.batch_size and len(g_pairs) == self.batch_size: break

        self.update_synthesised_data(all_pairs,pc)

        return d_pairs, g_pairs, sampler_samples

    def update_synthesised_data(self,all_pairs,pc):
        if len(all_pairs) > 0:
            '''Update dynamic data'''
            self.sim_env.restore_simulation_state()
            synthesised_data_obj = SynthesisedData()
            synthesised_data_obj.obj_ids = self.sim_env.objects
            synthesised_data_obj.obj_poses = self.sim_env.objects_poses

            assert 7 * len(self.sim_env.objects) == len(self.sim_env.objects_poses)

            for pair in all_pairs:
                target_index, target_point, pose, importance, grasped_object = pair

                target_point = pc[target_index]

                U_alpha_beta_score = self.approach_beta_clusters.get_uniqueness_score(
                    pose[0:5].detach()).item()

                synthesised_data_obj.target_indexes.append(target_index)
                synthesised_data_obj.grasp_target_points.append(target_point.cpu().numpy())
                synthesised_data_obj.grasp_parameters.append(pose.cpu().numpy())
                synthesised_data_obj.importance.append(importance)
                synthesised_data_obj.grasped_objects.append(grasped_object)

                synthesised_data_obj.uniqueness.append(U_alpha_beta_score)

            if self.loaded_synthesised_data is not None:
                synthesised_data_obj.id = self.loaded_synthesised_data.id
                for n in range(len(self.loaded_synthesised_data.target_indexes)):
                    target_index = self.loaded_synthesised_data.target_indexes[n]
                    # if target_index in synthesised_data_obj.target_indexes: continue

                    if len(self.loaded_synthesised_data.grasp_parameters[n])!=self.n_param:
                        if print_details:print(Fore.RED,f'old record has different pose shape', Fore.RESET)
                        continue

                    if self.loaded_synthesised_data.grasped_objects[n] is None: continue
                    U_alpha_beta_score = self.approach_beta_clusters.get_uniqueness_score(
                        torch.tensor(self.loaded_synthesised_data.grasp_parameters[n][0:5]).to(device)).item()

                    synthesised_data_obj.target_indexes.append(target_index)
                    synthesised_data_obj.grasp_target_points.append(self.loaded_synthesised_data.grasp_target_points[n])
                    synthesised_data_obj.grasp_parameters.append(self.loaded_synthesised_data.grasp_parameters[n])
                    synthesised_data_obj.importance.append(self.loaded_synthesised_data.importance[n])
                    synthesised_data_obj.grasped_objects.append(self.loaded_synthesised_data.grasped_objects[n])

                    synthesised_data_obj.uniqueness.append(U_alpha_beta_score)

            '''update'''
            if self.loaded_synthesised_data is None:
                if len(self.DDM)>=self.max_scenes:
                    importance, uniqueness = synthesised_data_obj.unique_obj_max_scores()
                    ave_uniqueness = sum(uniqueness)/len(uniqueness)

                    self.Ave_uniquness.update(ave_uniqueness)

                    if  not self.Ave_uniquness.lower_rejection_criteria(ave_uniqueness, k=2.):
                        self.DDM.save_data_point(synthesised_data_obj)
                        if len(self.DDM)-len(self.DDM.low_quality_samples_tracker)<self.max_scenes:
                            if print_details:print(Fore.GREEN,
                                  f'Replace sample, criteria: ave_uniqueness : { ave_uniqueness} ',
                                  Fore.RESET)
                        else:
                            print(Fore.GREEN, f'Add new sample, criteria: ave_uniqueness : { ave_uniqueness}',
                                  Fore.RESET)
                    else:
                        if print_details:print(Fore.LIGHTYELLOW_EX,
                              f'Ignore new sample, criteria: ave_uniqueness : { ave_uniqueness} ',
                              Fore.RESET)
                else:
                    print(Fore.GREEN,f'Add new sample',
                    Fore.RESET)
                    self.DDM.save_data_point(synthesised_data_obj)
            else:
                importance, uniqueness = synthesised_data_obj.unique_obj_max_scores()
                ave_uniqueness = sum(uniqueness)/len(uniqueness)
                max_importance=max(importance)
                self.Ave_uniquness.update(ave_uniqueness)

                self.DDM.update_old_record(synthesised_data_obj)

                not_unique = self.Ave_uniquness.lower_rejection_criteria(ave_uniqueness, k=2.,report=print_details)

                if len(self.DDM) >= self.max_scenes and (not_unique or (max_importance<0.1)):# ( (c_Importance and c_Uniquness) or (c_Importance_too_confident and c_Uniquness)) :
                    if print_details:print(Fore.LIGHTRED_EX,
                          f'poor sample detected, criteria: not_unique: { not_unique},  ave_uniqueness: { ave_uniqueness}, max_importance:{max_importance} ',
                          Fore.RESET)
                    self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

                    if len(self.DDM) > self.max_scenes: self.DDM.try_compress()

            self.skip_rate.update(0.)
        else:
            self.skip_rate.update(1.)

            if self.loaded_synthesised_data is not None:
                if print_details:print(Fore.LIGHTRED_EX, 'Poses not found for the scene, to be replaced', Fore.RESET)

                self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)
                if len(self.DDM)>self.max_scenes: self.DDM.try_compress()

    def prepare_voxels(self, pairs, depth, pc, full_pointcloud, view=False):
        '''prepare cropped point clouds''''''prepare cropped point clouds'''
        radius = 0.13
        batch_features_list = []
        batch_indices_list = []
        space_range = 2.0
        voxel_size = 0.02
        grid_size = int(space_range / voxel_size)
        b = 0
        for pair in pairs:
            index = pair[0]

            center = pc[index]

            sub_pc = crop_cube(full_pointcloud, center, cube_size=2 * radius)
            sub_pc -= center
            sub_pc /= radius

            if view:
                visualize_depth_with_flat_index(depth.cpu().numpy(), index)
                view_npy_open3d(sub_pc.cpu().numpy(), view_coordinate=True)

            coords = ((sub_pc + 1.0) / space_range * grid_size).floor().int()

            # Safety clamp
            coords = torch.clamp(coords, 0, grid_size - 1)

            # Unique voxels
            voxel_coords, inverse = torch.unique(
                coords, dim=0, return_inverse=True
            )

            voxel_features = scatter_mean(
                sub_pc, inverse, dim=0
            )

            batch_size = 1
            batch_indices = torch.zeros(
                (voxel_coords.shape[0], 1),
                dtype=torch.int32,
                device=sub_pc.device
            ) + b

            indices = torch.cat([
                batch_indices,
                voxel_coords[:, [2, 1, 0]]  # z, y, x
            ], dim=1)

            batch_indices_list.append(indices)
            batch_features_list.append(voxel_features)

            b += 1

        batch_features = torch.cat(batch_features_list, dim=0)
        batch_indices = torch.cat(batch_indices_list, dim=0)

        cropped_local_point_clouds = spconv.SparseConvTensor(
            features=batch_features.float(),  # (M, C=3)
            indices=batch_indices,  # (M, 4)
            spatial_shape=[grid_size] * 3,
            batch_size=self.batch_size
        )

        if view:
            x = cropped_local_point_clouds.dense()
            x = (x != 0).any(dim=1, keepdim=False).float().cpu().numpy()[0]
            view_3d_occupancy_grid(x)

        return cropped_local_point_clouds

    def step(self, report=False):

        self.sim_env.max_obj_per_scene = 10

        if (self.skipped_last or self.skip_rate.val>np.random.random()**2 or len(self.DDM) < 100) and not self.train_policy_only:

            self.loaded_synthesised_data = self.DDM.load_random_sample()
            self.sim_env.objects = deque(self.loaded_synthesised_data.obj_ids)
            self.sim_env.objects_poses = self.loaded_synthesised_data.obj_poses
            self.sim_env.reload()

        else:
            self.loaded_synthesised_data = None

            self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)

            min_=0 if self.train_policy_only else 5
            self.sim_env.drop_new_obj(selected_index=None, stablize=True,n=random.randint(min_, self.sim_env.max_obj_per_scene ))

        '''get scene perception'''
        depth, pc, floor_mask = self.sim_env.get_scene_preception(view=False)

        full_objects_pc = self.sim_env.get_obj_point_clouds(view=False)
        full_pointcloud = np.vstack([pc[floor_mask], full_objects_pc])
        floor_mask = torch.from_numpy(floor_mask).to(device)
        full_pointcloud = torch.from_numpy(full_pointcloud).to(device)
        clean_depth = torch.from_numpy(depth).to(device)  # [600.600]
        depth = torch.from_numpy(depth).to(device)  # [600.600]

        if self.domain_randomization:
            depth=add_reflective_blob_noise(clean_depth,n_blobs=np.random.randint(5,10), blob_radius=np.random.uniform(1, 3), outlier_scale=0.02)
            depth=add_depth_noise(depth,keep_mask=floor_mask.reshape(600,600))
            pc, _ = self.sim_env.depth_to_pointcloud(depth.cpu().numpy(), self.sim_env.intr, self.sim_env.extr)

        pc = torch.from_numpy(pc).to(device)


        for k in range(self.iter_per_scene):

            with torch.no_grad():
                self.gan.generator.eval()
                grasp_pose, grasp_quality_logits,features,grasp_collision_logits = self.gan.generator( depth[None, None, ...],detach_backbone=True)
                self.gan.generator.train()

                grasp_quality = logits_to_probs(grasp_quality_logits)

                annealing_factor = (1 - grasp_quality.detach()).clamp(min=self.skip_rate.val ** 2)
                if print_details:print(Fore.LIGHTYELLOW_EX,
                      f'mean_annealing_factor= {annealing_factor.mean()},max_annealing_factor= {annealing_factor.max()},min_annealing_factor= {annealing_factor.min()}, skip rate={self.skip_rate.val}',
                      Fore.RESET)


                grasp_pose_ref =  self.pose_interpolation(grasp_pose,
                                                         annealing_factor=annealing_factor)  # [b,self.n_param,600,600]
                if self.loaded_synthesised_data is not None:
                    '''inject saved poses'''

                    grasp_pose_ref = grasp_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
                    grasp_pose_gen = grasp_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)


                    for t in range(len(self.loaded_synthesised_data.target_indexes)):
                        index = self.loaded_synthesised_data.target_indexes[t]
                        pose = self.loaded_synthesised_data.grasp_parameters[t]

                        pose = torch.tensor(pose).to(device)

                        if pose.shape==grasp_pose_ref[index].shape:
                            option1=pose
                            option2=pose*0.9+grasp_pose_gen[index]*0.1
                            u1 = self.approach_beta_clusters.get_uniqueness_score(
                                option1[0:5]).item()
                            u2 = self.approach_beta_clusters.get_uniqueness_score(
                                option2[0:5]).item()
                            grasp_pose_ref[index] = option1 if u1>u2 else option2
                        elif pose.shape[0]>=5:
                            grasp_pose_ref[index][0:5] = pose[0:5]

                    grasp_pose_ref = grasp_pose_ref.reshape(600, 600, self.n_param).permute(2, 0, 1).unsqueeze(0)

                if report and k == 0:
                    self.view_result(grasp_pose, (~floor_mask) & (grasp_quality.reshape(-1)>0.5),(~floor_mask) & (grasp_quality.reshape(-1)<0.5))

                d_pairs, g_pairs = [], []
                if not self.train_policy_only:

                    d_pairs, g_pairs, sampler_samples = self.sample_contrastive_pairs(pc, floor_mask, grasp_pose,
                                                                                      grasp_pose_ref,
                                                                                     grasp_quality.detach() )
                    if self.synthesizie_only: break

            if self.test_mode:
                if len(d_pairs) > 0 and self.view:
                    self.prepare_voxels(d_pairs, depth, pc, full_pointcloud, view=self.view)
                return

            grasp_pose = grasp_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
            grasp_pose_ref_pixel = None if self.train_policy_only else grasp_pose_ref
            grasp_pose_ref = None if self.train_policy_only else grasp_pose_ref[0].permute(1, 2, 0).reshape(360000,
                                                                                                           self.n_param)

            if not self.train_policy_only and len(d_pairs) == self.batch_size:

                d_cropped_local_point_clouds = self.prepare_voxels(d_pairs, depth, pc, full_pointcloud)
                # d_cropped_local_point_clouds=None
                self.step_discriminator(d_cropped_local_point_clouds, depth,  grasp_pose, grasp_pose_ref, d_pairs)
                if print_details:self.print_pairs_info(d_pairs, grasp_pose, grasp_pose_ref)

                self.skipped_last=False
            else:
                self.skipped_last=True


            if not self.train_policy_only and len(g_pairs) == self.batch_size:

                g_cropped_local_point_clouds = self.prepare_voxels(g_pairs, depth, pc, full_pointcloud)

                self.step_policy(g_cropped_local_point_clouds, depth, clean_depth, floor_mask, pc, grasp_pose_ref_pixel,
                                    g_pairs)
                if print_details:self.print_pairs_info(g_pairs, grasp_pose, grasp_pose_ref)

            elif self.skip_rate.val < 0.5 or self.train_policy_only:

                self.step_policy(None, depth, clean_depth, floor_mask, pc, grasp_pose_ref_pixel, g_pairs  )

            if not self.train_policy_only and not (
                    (len(d_pairs) == self.batch_size) or (len(g_pairs) == self.batch_size)) and not self.test_mode:
                if k == 0:
                    self.sim_env.remove_objects(n=2)
                    break



    def view_result(self, grasp_poses=None, mask=None,mask2=None):
        with torch.no_grad():


            cuda_memory_report()

            if grasp_poses is not None:
                grasp_poses_ = grasp_poses[0].permute(1, 2, 0).reshape(360000, self.n_param).detach()  # .cpu().numpy()
                if mask is not None:grasp_poses_1 = grasp_poses_[mask]
                if mask2 is not None: grasp_poses_2 = grasp_poses_[mask2]
                if grasp_poses_1.numel() > 0:
                    pose_std=torch.std(grasp_poses_1, dim=0).cpu()
                    print(f'Valid grasp_pose parameters std = {pose_std}' )
                    pose_range=(torch.max(grasp_poses_1, dim=0)[0]-torch.min(grasp_poses_1, dim=0)[0]).cpu()
                    print(f'Valid grasp_pose parameters range = {pose_range}')

                    pose_std2=torch.std(grasp_poses_2, dim=0).cpu()
                    print(f'Invalid grasp_pose parameters std = {pose_std2}' )
                    pose_range2=(torch.max(grasp_poses_2, dim=0)[0]-torch.min(grasp_poses_2, dim=0)[0]).cpu()
                    print(f'Invalid grasp_pose parameters range = {pose_range2}')

                    if not torch.isnan(pose_std).any():
                        self.moving_std=pose_std if self.moving_std is None else self.moving_std *0.9+pose_std*0.1

                    if not torch.isnan(pose_range).any():
                        self.moving_range=pose_range if self.moving_range is None else self.moving_range *0.9+pose_range*0.1

                    print(f'alpha parameters moving std = {self.moving_std[0:3].mean()}')
                    print(f'alpha parameters range = {self.moving_range[0:3].mean()}')
                    print(f'beta parameters moving std = {self.moving_std[3:5].mean()}')
                    print(f'beta parameters range = {self.moving_range[3:5].mean()}')
                    print(f'delta parameters moving std = {self.moving_std[5:8].mean()}')
                    print(f'delta parameters range = {self.moving_range[5:8].mean()}')
                    print(f'joints parameters moving std = {self.moving_std[8:].mean()}')
                    print(f'joints parameters range = {self.moving_range[8:].mean()}')
                    torch.save(self.moving_std,self.model_key+'_moving_std')
                    torch.save(self.moving_range,self.model_key+'_moving_range')


                else: print('No valid grasps')

            self.sampler_loss_statistics.print()
            self.critic_loss_statistics.print()

            self.skip_rate.view()

            self.Ave_uniquness.view()
            self.random_sampler_acceptance_rate.view()
            self.confidence_indicator.view()
            self.discrimination_dist.view()

            self.balanced_set_grasp_quality_statistics.print()
            self.balanced_set_collision_statistics.print()

            self.grasp_quality_statistics.print()

            self.argmax_grasp_quality_statistics.print()
            self.approach_beta_clusters.view()


    def save_statistics(self):
        self.sim_env.save_obj_dict()

        self.skip_rate.save()

        self.random_sampler_acceptance_rate.save()

        self.critic_loss_statistics.save()
        self.sampler_loss_statistics.save()
        self.approach_beta_clusters.save()


        self.argmax_grasp_quality_statistics.save()

        self.Ave_uniquness.save()
        self.confidence_indicator.save()
        self.discrimination_dist.save()

        self.balanced_set_grasp_quality_statistics.save()

        self.balanced_set_collision_statistics.save()
        self.grasp_quality_statistics.save()

        self.sim_env.save_obj_dict()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()


    def begin(self, iterations=10):
        context = torch.no_grad() if self.test_mode else torch.enable_grad()

        with context:
            pi = progress_indicator('Begin new training round: ', max_limit=iterations)

            print(f'# Synthesised scenes = {len(self.DDM)}')

            for i in range(iterations):
                if self.skip_rate.val > 0.8:
                    self.batch_size = 1
                    self.iter_per_scene = 1  # 5
                    self.sim_env.max_obj_per_scene = 1
                elif self.skip_rate.val < 0.4:
                    self.batch_size = 2
                    self.iter_per_scene = 1
                    self.sim_env.max_obj_per_scene = int(7 * np.random.rand())

                if self.args.catch_exceptions:
                    try:
                        self.step(report=i == iterations - 1)
                        pi.step(i)
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                        self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)
                        if self.loaded_synthesised_data is not None: self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

                else:
                    self.step( report=i == iterations - 1)
                    pi.step(i)

            pi.end()
            
            if not self.test_mode:
                self.export_check_points()
                self.save_statistics()


    def show_overlaid_graphs(self, iterations=5,load_from_dataset=True):

        with torch.no_grad() :
            
            grasp_quality_data=[]

            for i in range(iterations):

                self.sim_env.max_obj_per_scene = 10

                if load_from_dataset:

                    self.loaded_synthesised_data = self.DDM.load_random_sample()
                    self.sim_env.objects = deque(self.loaded_synthesised_data.obj_ids)
                    self.sim_env.objects_poses = self.loaded_synthesised_data.obj_poses

                    self.sim_env.reload()

                else:
                    self.loaded_synthesised_data = None

                    self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)

                    self.sim_env.drop_new_obj(selected_index=None, stablize=True,
                                              n=random.randint(5, self.sim_env.max_obj_per_scene))


                '''get scene perception'''
                depth, pc, floor_mask = self.sim_env.get_scene_preception(view=False)

                floor_mask = torch.from_numpy(floor_mask).to(device)

                depth = torch.from_numpy(depth).to(device)  # [600.600]

                with torch.no_grad():
                    self.gan.generator.eval()
                    grasp_pose, grasp_quality_logits, features, grasp_collision_logits = self.gan.generator(
                        depth[None, None, ...], detach_backbone=True)

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
                        
            if len(grasp_quality_data)==1:plot_distribution(grasp_quality_data[0])
            else: plot_distribution_overlayed(grasp_quality_data, name='Scene')




