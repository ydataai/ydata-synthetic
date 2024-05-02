"""
Main synthesizer class
"""
from enum import Enum, unique
from joblib import load, dump
from typing import Any, Dict, Union

import tensorflow as tf

from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from ydata_synthetic.synthesizers.regular.cwgangp.model import CWGANGP
from ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN
from ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from ydata_synthetic.synthesizers.regular.ctgan.model import CTGAN
from ydata_synthetic.synthesizers.regular.gmm.model import GMM


@unique
class Model(Enum):
    VANILLA = 'gan'
    CONDITIONAL = 'cgan'
    WASSERTEIN = 'wgan'
    WASSERTEINGP = 'wgangp'
    CWASSERTEINGP = 'cwgangp'
    CRAMER = 'cramer'
    DEEPREGRET = 'dragan'
    CONDITIONALTABULAR = 'ctgan'
    FAST = 'fast'

    __MAPPING__ = {
        VANILLA: VanilllaGAN,
        CONDITIONAL: CGAN,
        WASSERTEIN: WGAN,
        WASSERTEINGP: WGAN_GP,
        CWASSERTEINGP: CWGANGP,
        CRAMER: CRAMERGAN,
        DEEPREGRET: DRAGAN,
        CONDITIONALTABULAR: CTGAN,
        FAST: GMM
    }

    @property
    def function(self):
        return self.__MAPPING__[self.value]

class RegularSynthesizer():
    """
    Abstraction class for synthetic data generation.
    """
    def __init__(self, modelname: str, model_parameters: Union[Dict[str, Any], None] = None, **kwargs):
        """
        Initializes the synthesizer object.

        Args:
            modelname (str): Name of the synthesizer model.
            model_parameters (Dict[str, Any], optional): Model parameters. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.modelname = modelname
        self.model_parameters = model_parameters
        self.model = None
        if Model(modelname) == Model.FAST:
            self.model = Model(modelname).function(**kwargs)
        else:
            self.model = Model(modelname).function(model_parameters, **kwargs)

    def __new__(cls, modelname: str, model_parameters: Union[Dict[str, Any], None] = None, **kwargs):
        return super().__new__(cls)

    def save(self, path: str):
        """
        Saves the synthesizer object to a pickle file.

        Args:
            path (str): Path to save the synthesizer pickle.
        """
        dump(self.__dict__, path)

    @staticmethod
    def load(path: str):
        """
        Loads a saved synthesizer from a pickle.

        Args:
            path (str): Path to read the synthesizer pickle from.

        Returns:
            Union[RegularSynthesizer, CTGAN]: The loaded synthesizer object.
        """
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tf.config.experimental.set_memory_growth(gpu_devices[0], True)
            except (ValueError, RuntimeError):
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        if isinstance(synth, dict) and Model(list(synth.keys())[0]) == Model.FAST:
            return GMM.load(synth)
        return RegularSynthesizer(**synth)
