import torch.nn.functional as F
from  model.Decoders import  FilmModulatedDecoder
from model.abstract_model import C, G
import torch
import torch.nn as nn

SH_model_key = 'SH_model'


class SHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()


        self.delta = FilmModulatedDecoder(in_c1=64, in_c2= 1, out_c=3, activation=nn.SiLU(), normalize=True).to(device)
        self.alpha = FilmModulatedDecoder(in_c1=64, in_c2= 1+3, out_c=3, activation=nn.SiLU(), normalize=False).to(device)

        self.beta = FilmModulatedDecoder(in_c1=64, in_c2= 1+3+3, out_c=2,activation=nn.SiLU(), normalize=False).to(device)


        self.fingers=FilmModulatedDecoder(in_c1=64, in_c2=1+5+3, out_c=3, activation=nn.SiLU(),normalize=False).to(device)




    def forward(self, features,depth):
        delta = self.delta(features, depth)

        alpha = self.alpha(features,torch.cat([depth,delta],dim=1))
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1))
        beta = F.normalize(beta, dim=1)

        fingers= self.fingers(features, torch.cat([alpha,beta,delta,depth], dim=1))

        pose = torch.cat([alpha,beta,delta,fingers], dim=1)

        return pose


class SH_G(G):
    def __init__(self):
        super().__init__(SHPoseSampler(),12)


class SH_D(C):
    def __init__(self):
        super().__init__(n_params=11)
