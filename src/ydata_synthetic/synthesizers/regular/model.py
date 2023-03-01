"""
    Main synthesizer class
"""
from enum import Enum, unique

from joblib import load

from tensorflow import config as tfconfig

from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from ydata_synthetic.synthesizers.regular.cwgangp.model import CWGANGP
from ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN
from ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from ydata_synthetic.synthesizers.regular.ctgan.model import CTGAN


@unique
class Model(Enum):
    VANILLA = 'gan'
    CONDITIONAL = 'cgan'
    WASSERTEIN =  'wgan'
    WASSERTEINGP ='wgangp'
    CWASSERTEINGP = 'cwgangp'
    CRAMER = 'cramer'
    DEEPREGRET = 'dragan'
    CONDITIONALTABULAR = 'ctgan'

    __MAPPING__ = {
        VANILLA : VanilllaGAN,
        CONDITIONAL: CGAN,
        WASSERTEIN: WGAN,
        WASSERTEINGP: WGAN_GP,
        CWASSERTEINGP: CWGANGP,
        CRAMER: CRAMERGAN,
        DEEPREGRET: DRAGAN,
        CONDITIONALTABULAR: CTGAN
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
        if isinstance(synth, dict):
            return CTGAN.load(synth)
        return synth