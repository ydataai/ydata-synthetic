"""
    Main time-series synthesizer class
"""
from enum import Enum, unique
import os
import logging
from joblib import load

from tensorflow import config as tfconfig

from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN
from ydata_synthetic.synthesizers.timeseries.doppelganger.model import DoppelGANger

from ydata_synthetic.utils.logger import SynthesizersLogger

logger = SynthesizersLogger(name='timseriesSynthesizer.logger')
logger.setLevel(logging.INFO)

@unique
class Model(Enum):
    TIMEGAN = 'timegan'
    DOPPELGANGER = 'doppelganger'

    __MAPPING__ = {
        TIMEGAN : TimeGAN,
        DOPPELGANGER: DoppelGANger
    }

    @property
    def function(self):
        return self.__MAPPING__[self.value]

class TimeSeriesSynthesizer():
    "Abstraction class "
    def __new__(cls, modelname: str, model_parameters=None, **kwargs):
        logger.info_def_report(model=modelname)
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
        if os.path.isdir(path):
            return DoppelGANger.load(path)
        return load(path)
