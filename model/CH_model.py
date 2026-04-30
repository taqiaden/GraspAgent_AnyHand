import torch.nn.functional as F

from Configurations.config import device
from  model.Decoders import  FilmModulatedDecoder
from model.abstract_model import G, C
import torch
import torch.nn as nn


CH_model_key = 'CH_model'
CH_model_key2 = 'CH_model2'



class CHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()


        self.alpha = FilmModulatedDecoder(in_c1=64, in_c2= 1+3, out_c=3,activation=nn.SiLU(), normalize=False).to(device)
        self.beta = FilmModulatedDecoder(in_c1=64, in_c2= 1+3+3, out_c=2,activation=nn.SiLU(), normalize=False).to(device)
        self.fingers=FilmModulatedDecoder(in_c1=64, in_c2=1+5+3, out_c=3,activation=nn.SiLU(),normalize=False).to(device)

        self.delta=FilmModulatedDecoder(in_c1=64, in_c2=1, out_c=3, activation=nn.SiLU(),normalize=False).to(device)



    def forward(self, features,depth ):

        delta= self.delta(features, depth)

        alpha = self.alpha(features,torch.cat([depth,delta], dim=1))
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1))
        beta = F.normalize(beta, dim=1)


        fingers=self.fingers(features,torch.cat([alpha,delta,beta,depth], dim=1))


        pose = torch.cat([alpha,beta,delta,fingers], dim=1)

        return pose


class CH_G(G):
    def __init__(self):
        super().__init__(CHPoseSampler(),12)

class CH_D(C):
    def __init__(self):
        super().__init__(n_params=11)

