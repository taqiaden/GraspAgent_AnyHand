import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2D(nn.Module):
    def __init__(self,channels,elementwise_affine=True):
        super().__init__()
        self.norm=nn.LayerNorm([channels],elementwise_affine=elementwise_affine)

    def forward(self,x):
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=x.permute(0,3,1,2)
        return x


class MahalanobisDistance(nn.Module):
    def __init__(self, dim=64, out_dim=None, normalize=False):
        """
        dim: input feature dimension (64)
        out_dim: projected dimension (default = dim)
        normalize: whether to L2-normalize inputs before distance
        """
        super().__init__()
        out_dim =  dim if out_dim is None else out_dim
        self.normalize = normalize

        # W defines M = W^T W
        self.W = nn.Linear(dim, out_dim, bias=False)

        # Small-gain initialization for stability
        nn.init.kaiming_normal_(self.W.weight, nonlinearity="linear")
        self.W.weight.data *= 0.5

    def forward(self, main, others):
        """
        main:   [B, 1, 64]
        others: [B, N, 64]

        returns:
            dist: [B, N]
        """
        if self.normalize:
            main = F.normalize(main, dim=-1)
            others = F.normalize(others, dim=-1)

        # Broadcast main to [B, N, 64]
        diff = main - others          # [B, N, 64]

        # Apply learned transform
        z = self.W(diff)              # [B, N, out_dim]

        # Squared Mahalanobis distance
        dist = (z * z).sum(dim=-1)    # [B, N]

        return dist


class CriticDecoder(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()

        self.context_proj = nn.Sequential(
            nn.Linear(in_c1, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64, bias=True),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(in_c2, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64, bias=True),
        )

        self.dist = MahalanobisDistance(dim=64,normalize=True).to('cuda')

    def forward(self, context, condition):
        condition = self.cond_proj(condition)

        context = self.context_proj(context)

        x = self.dist(main=context, others=condition)
        return x



class FilmModulatedDecoder(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,
                 activation=None,normalize=False):
        super().__init__()

        mid_c=max(in_c1,in_c2)
        mid_c+=mid_c%2

        self.gamma = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')
        self.beta = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')


        self.condition_proj =nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            activation,
            nn.Conv2d(mid_c , max(48,5*out_c), kernel_size=1,bias=True),
            LayerNorm2D(max(48,5*out_c)),
            activation,
            nn.Conv2d(max(48,5*out_c), max(32,3*out_c), kernel_size=1,bias=True),
            LayerNorm2D(max(32,3*out_c)),
            activation,
            nn.Conv2d(max(32,3*out_c), out_c, kernel_size=1,bias=True)
        ).to('cuda') if normalize else  nn.Sequential(
            activation,
            nn.Conv2d(mid_c , max(48,5*out_c), kernel_size=1,bias=True),
            activation,
            nn.Conv2d(max(48,5*out_c), max(32,3*out_c), kernel_size=1,bias=True),
            activation,
            nn.Conv2d(max(32,3*out_c), out_c, kernel_size=1,bias=True)
        ).to('cuda')

    def forward(self, context, condition):

        condition = self.condition_proj(condition)

        gamma = self.gamma(context)
        beta = self.beta(context)

        x = condition * gamma+beta

        x = self.d(x)

        return x











