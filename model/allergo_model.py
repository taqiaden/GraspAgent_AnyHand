
import torch.nn.functional as F
from  model.Decoders import  FilmModulatedDecoder
from model.abstract_model import C, G
import torch
import torch.nn as nn

Allergo_model_key = 'Allergo_model'

class ALlergoPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.delta = FilmModulatedDecoder(in_c1=64, in_c2= 1, out_c=3, activation=nn.SiLU(),normalize=True).to(
            'cuda')

        self.alpha = FilmModulatedDecoder(in_c1=64, in_c2= 1+3, out_c=3,activation=nn.SiLU(),normalize=True).to(
            'cuda')
        self.beta = FilmModulatedDecoder(in_c1=64, in_c2= 4+3, out_c=2, activation=nn.SiLU(),normalize=True).to(
            'cuda')

        self.fingers=FilmModulatedDecoder(in_c1=64, in_c2=9, out_c=16, activation=nn.SiLU(),normalize=True).to(
            'cuda')

        self.biases = nn.Parameter(torch.tensor([0.]*19, dtype=torch.float32, device='cuda'), requires_grad=True).reshape(1,-1,1,1)


    def forward(self, features,depth ):

        delta = self.delta(features,depth)

        alpha = self.alpha(features,torch.cat([depth,delta],dim=1))
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1))
        beta = F.normalize(beta, dim=1)

        fingers= self.fingers(features, torch.cat([depth,delta,alpha,beta], dim=1))

        pose = torch.cat([alpha,beta,delta,fingers], dim=1) #28

        return pose


class Allergo_G(G):
    def __init__(self):
        super().__init__(ALlergoPoseSampler(),25)


class Allergo_D(C):
    def __init__(self):
        super().__init__(n_params=24)

