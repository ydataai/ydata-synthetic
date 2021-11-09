from collections import namedtuple
from enum import Enum
from typing import List, Union

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor

_regular_processor_parameters = ['num_cols', 'cat_cols', 'pos_idx']
_regular_processor_parameters_df = [[], [], False]

RegProcessorArguments = namedtuple('RegularProcessorParameters',
                                    _regular_processor_parameters,
                                    defaults=_regular_processor_parameters_df)

class RegularModels(Enum):
    "Supported models for the Regular Data Processor."
    CGAN = 'CGAN'
    CRAMERGAN = 'CramerGAN'
    DRAGAN = 'DRAGAN'
    GAN = 'VanillaGAN'
    WGAN = 'WGAN'
    WGAN_GP = 'WGAN_GP'

@typechecked
class RegularDataProcessor(BaseProcessor):
    """
    Main class for Regular/Tabular Data Preprocessing.
    It works like any other transformer in scikit learn with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings/list of ints):
            List of names of numerical columns or positional indexes (if pos_idx was set to True).
        cat_cols (list of strings/list of ints):
            List of names of categorical columns or positional indexes (if pos_idx was set to True).
        pos_idx (bool):
            Specifies if the passed col IDs are names or positional indexes (column numbers).
    """
    def __init__(self, reg_processor_args: RegProcessorArguments):
        super().__init__(num_cols = reg_processor_args.num_cols,
                         cat_cols = reg_processor_args.cat_cols,
                         pos_idx = reg_processor_args.pos_idx)

        self.num_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        self.cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
