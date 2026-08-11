from model.abstract_model import G, C
from model.Decoders import PoseSampler

SH_model_key = 'SH_5F_model'


class SH_G(G):
    def __init__(self):
        super().__init__(PoseSampler(n_joint=20),29)

class SH_D(C):
    def __init__(self):
        super().__init__(n_params=28)

