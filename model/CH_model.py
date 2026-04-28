import torch.nn.functional as F
from  model.Decoders import CriticDecoder, FilmModulatedDecoder
from model.abstract_model import G, C
from  model.sparse_encoder import SparseEncoderIN
from  utils.NN_tools import replace_instance_with_groupnorm
from  utils.model_init import init_weights_he_normal
from  utils.positional_encoding import PositionalEncoding_2d, LearnableRBFEncoding2D
from model.resunet import res_unet
import torch
import torch.nn as nn


CH_model_key = 'CH_model'
CH_model_key2 = 'CH_model2'



class CHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()


        self.alpha = FilmModulatedDecoder(in_c1=64, in_c2= 1+3, out_c=3,activation=nn.SiLU(), normalize=False).to(
            'cuda')
        self.beta = FilmModulatedDecoder(in_c1=64, in_c2= 1+3+3, out_c=2,activation=nn.SiLU(), normalize=False).to(
            'cuda')
        self.fingers=FilmModulatedDecoder(in_c1=64, in_c2=1+5+3, out_c=3,activation=nn.SiLU(),normalize=False).to(
            'cuda')

        self.delta=FilmModulatedDecoder(in_c1=64, in_c2=1, out_c=3, activation=nn.SiLU(),normalize=False).to(
            'cuda')

        self.biases = nn.Parameter(torch.tensor([0.]*6, dtype=torch.float32, device='cuda'), requires_grad=True).reshape(1,-1,1,1)


    def forward(self, features,depth ):

        delta= self.delta(features, depth)
        # delta=F.tanh(delta)+self.biases[:,0:3]

        alpha = self.alpha(features,torch.cat([depth,delta], dim=1))
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1))
        beta = F.normalize(beta, dim=1)


        fingers=self.fingers(features,torch.cat([alpha,delta,beta,depth], dim=1))
        # fingers=F.tanh(fingers)+self.biases[:,3:]


        pose = torch.cat([alpha,beta,delta,fingers], dim=1)

        return pose


class CH_G(G):
    def __init__(self):
        super().__init__(CHPoseSampler(),12)

class CH_D(C):
    def __init__(self):
        super().__init__(n_params=11)

