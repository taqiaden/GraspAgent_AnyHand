from model.abstract_model import C, G
from model.Decoders import PoseSampler
Allergo_model_key = 'Allergo_model'


class Allergo_G(G):
    def __init__(self):
        super().__init__(PoseSampler(n_joint=16),25)


class Allergo_D(C):
    def __init__(self):
        super().__init__(n_params=24)

