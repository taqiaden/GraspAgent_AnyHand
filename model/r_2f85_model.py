from model.abstract_model import C, G
from model.Decoders import PoseSampler

R_2F85_model_key = 'R_2F85_model'


class R_2F85_G(G):
    def __init__(self):
        super().__init__(PoseSampler(n_joint=0),9+3)

class R_2F85_D(C):
    def __init__(self):
        super().__init__(n_params=8+3)
