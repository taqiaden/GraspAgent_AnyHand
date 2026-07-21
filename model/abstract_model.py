import torch
from torch import nn
from Configurations.config import device
from  model.sparse_encoder import SparseEncoderIN
from  model.Decoders import CriticDecoder, FilmModulatedDecoder
from  utils.model_init import init_weights_he_normal
from model.resunet import res_unet

def depth_normalization(depth):
    max_ = 1.3
    min_ = 1.15
    standarized_depth_ = (depth.clone() - min_) / (max_ - min_)
    standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
    return standarized_depth_
class G(nn.Module):
    def __init__(self,sampler_decoder,n_params,static_joints=None):
        super().__init__()
        self.static_joints=[] if static_joints is None else static_joints

        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0., activation=nn.ReLU(), IN_affine=False,
                                  activate_skip=False).to(device)

        self.back_bone2_ = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                    relu_negative_slope=0., activation=nn.ReLU(), IN_affine=False, activate_skip=False).to(device)

        self.back_bone3_ = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                    relu_negative_slope=0., activation=nn.ReLU(), IN_affine=False, activate_skip=False).to(device)

        self.PoseSampler = sampler_decoder


        self.grasp_quality_=FilmModulatedDecoder( 64, n_params, 1,
        activation=nn.SiLU(),  normalize=True).to(device)

        self.collision=FilmModulatedDecoder( 64, 8+1+len(self.static_joints), 1,
        activation=nn.SiLU(),  normalize=True).to(device)

        self.back_bone.apply(init_weights_he_normal)
        self.back_bone2_.apply(init_weights_he_normal)
        self.back_bone3_.apply(init_weights_he_normal)

        self.grasp_quality_.apply(init_weights_he_normal)
        self.collision.apply(init_weights_he_normal)

    def forward(self, depth,  detach_sampler=False,detach_quality=False,detach_collision=False):
        standarized_depth_=depth_normalization(depth)

        '''backbones'''
        if detach_sampler:
            with torch.no_grad():
                features = self.back_bone(standarized_depth_)
                dense_grasp_pose = self.PoseSampler(features, standarized_depth_)

        else:
            features = self.back_bone(standarized_depth_)
            dense_grasp_pose = self.PoseSampler(features, standarized_depth_)

        detached_dense_grasp_pose = dense_grasp_pose.detach().clone()
        detached_dense_grasp_pose = torch.cat([detached_dense_grasp_pose, standarized_depth_], dim=1)

        if detach_quality:
            with torch.no_grad():
                features2 = self.back_bone2_(standarized_depth_)
                grasp_quality_logits = self.grasp_quality_(features2, detached_dense_grasp_pose)


        else:
            features2 = self.back_bone2_(standarized_depth_)
            grasp_quality_logits = self.grasp_quality_(features2, detached_dense_grasp_pose)

        # print('G b1 max val= ', features.max().item(), 'mean:', features.mean().item(), ' std:',
        #       features.std(dim=1).mean().item())
        # print('G b2 max val= ', features2.max().item(), 'mean:', features2.mean().item(), ' std:',
        #       features2.std(dim=1).mean().item())

        detached_dense_grasp_pose = torch.cat([detached_dense_grasp_pose[:, 0:5], detached_dense_grasp_pose[:, 8:11],detached_dense_grasp_pose[:,self.static_joints],standarized_depth_], dim=1)
        if detach_collision:
            with torch.no_grad():
                features3 = self.back_bone3_(standarized_depth_)
                collision = self.collision(features3, detached_dense_grasp_pose)

        else:
            features3 = self.back_bone3_(standarized_depth_)
            collision = self.collision(features3, detached_dense_grasp_pose)


        return dense_grasp_pose, grasp_quality_logits,features2.detach(),collision

class C(nn.Module):
    def __init__(self,n_params):
        super().__init__()

        self.back_bone = SparseEncoderIN().to(device)

        self.decoder = CriticDecoder(in_c1=512 , in_c2=n_params  ).to(device)

        self.back_bone.apply(init_weights_he_normal)
        self.decoder.apply(init_weights_he_normal)

    def forward(self,  pose,  cropped_local_point_clouds, detach_backbone=False):

        if detach_backbone:
            with torch.no_grad():
                anchor = self.back_bone(cropped_local_point_clouds)
        else:
            anchor = self.back_bone(cropped_local_point_clouds)

        # print('D max val= ', anchor.max().item(), 'mean:', anchor.mean().item(),
        #       ' std:',
        #       anchor.std(dim=1).mean().item())

        scores = self.decoder(anchor[:,None], pose)

        return scores
