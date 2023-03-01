"""
    Main synthesizer class
"""
from enum import Enum, unique

from joblib import load

from tensorflow import config as tfconfig

from ..regular.vanillagan.model import VanilllaGAN
from ..regular.cgan.model import CGAN
from ..regular.wgan.model import WGAN
from ..regular.wgangp.model import WGAN_GP
from ..regular.cwgangp.model import CWGANGP
from ..regular.cramergan.model import CRAMERGAN
from ..regular.dragan.model import DRAGAN

from ...utils.gumbel_softmax import GumbelSoftmaxActivation


@unique
class Model(Enum):
    VANILLA = 'gan'
    CONDITIONAL = 'cgan'
    WASSERTEIN =  'wgan'
    WASSERTEINGP ='wgangp'
    CWASSERTEINGP = 'cwgangp'
    CRAMER = 'cramer'
    DEEPREGRET = 'dragan'

    __MAPPING__ = {
        VANILLA : VanilllaGAN,
        CONDITIONAL: CGAN,
        WASSERTEIN: WGAN,
        WASSERTEINGP: WGAN_GP,
        CWASSERTEINGP: CWGANGP,
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

    @staticmethod
    def load(path):
        """
        ### Description:
        Loads a saved synthesizer from a pickle.

        ### Args:
        `path` (str): Path to read the synthesizer pickle from.
        """
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except (ValueError, RuntimeError):
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth