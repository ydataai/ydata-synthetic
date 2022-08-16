"""
    Main synthesizer class
"""
from enum import Enum, unique

from ..regular.vanillagan.model import VanilllaGAN
from ..regular.cgan.model import CGAN
from ..regular.wgan.model import WGAN
from ..regular.wgangp.model import WGAN_GP
from ..regular.cramergan.model import CRAMERGAN
from ..regular.dragan.model import DRAGAN

@unique
class Model(Enum):
    VANILLA = 'gan'
    CONDITIONAL = 'cgan'
    WASSERTEIN =  'wgan'
    WASSERTEINGP ='wgangp'
    CRAMER = 'cramer'
    DEEPREGRET = 'dragan'

    __MAPPING__ = {
        VANILLA : VanilllaGAN,
        CONDITIONAL: CGAN,
        WASSERTEIN: WGAN,
        WASSERTEINGP: WGAN_GP,
        CRAMER: CRAMERGAN,
        DEEPREGRET: DRAGAN
    }

    @property
    def function(self):
        return self.__MAPPING__[self.value]

class RegularSynthesizer():
    "Abstraction class "
    def __new__(cls, modelname: str, model_parameters, **kwargs):
        return Model(modelname).function(model_parameters, **kwargs)