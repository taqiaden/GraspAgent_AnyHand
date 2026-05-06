from model.abstract_model import C, G
from model.Decoders import PoseSampler
SH_model_key = 'SH_model'





class SH_G(G):
    def __init__(self):
        super().__init__(PoseSampler(n_joint=3),12)


class SH_D(C):
    def __init__(self):
        super().__init__(n_params=11)
