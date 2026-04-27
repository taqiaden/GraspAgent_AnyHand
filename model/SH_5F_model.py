import torch.nn.functional as F
from  model.Decoders import  FilmModulatedDecoder
from model.abstract_model import G, C
import torch
import torch.nn as nn

SH_model_key = 'SH_5F_model'

class SHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.delta = FilmModulatedDecoder(in_c1=64, in_c2= 1, out_c=3,activation=nn.SiLU(), normalize=False).to(
            'cuda')

        self.alpha = FilmModulatedDecoder(in_c1=64, in_c2= 1+3, out_c=3,activation=nn.SiLU(), normalize=False).to(
            'cuda')
        self.beta = FilmModulatedDecoder(in_c1=64, in_c2= 4+3, out_c=2,activation=nn.SiLU(), normalize=False).to(
            'cuda')

        self.gamma = FilmModulatedDecoder(in_c1=64, in_c2= 7+2, out_c=2,activation=nn.SiLU(), normalize=False).to(
            'cuda')
        self.fingers=FilmModulatedDecoder(in_c1=64, in_c2=9+2, out_c=18,activation=nn.SiLU(),normalize=False).to(
            'cuda')

        self.biases = nn.Parameter(torch.tensor([0.]*23, dtype=torch.float32, device='cuda'), requires_grad=True).reshape(1,-1,1,1)


    def forward(self, features,depth ):

        delta = self.delta(features,depth.detach())
        delta=F.tanh(delta)+self.biases[:,0:3]

        alpha = self.alpha(features,torch.cat([depth,delta],dim=1).detach())
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1).detach())
        beta = F.normalize(beta, dim=1)


        gamma=self.gamma(features,torch.cat([depth,delta,alpha,beta], dim=1).detach())
        gamma=F.tanh(gamma)+self.biases[:,3:5]

        fingers= self.fingers(features, torch.cat([depth,delta,alpha,beta,gamma], dim=1).detach())
        fingers=F.tanh(fingers)+self.biases[:,5:]

        pose = torch.cat([alpha,beta,delta,gamma,fingers], dim=1) #28

        return pose

class SH_G(G):
    def __init__(self):
        super().__init__(SHPoseSampler(),29)

class SH_D(C):
    def __init__(self):
        super().__init__(n_params=28)

