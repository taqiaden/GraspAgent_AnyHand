from model.abstract_model import G, C
from model.Decoders import PoseSampler


CH_model_key = 'CH_model'


class CH_G(G):
    def __init__(self):
        super().__init__(PoseSampler(n_joint=3),12)

class CH_D(C):
    def __init__(self):
        super().__init__(n_params=11)

