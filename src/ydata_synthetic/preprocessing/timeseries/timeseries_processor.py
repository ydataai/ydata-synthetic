from collections import namedtuple
from enum import Enum
from typing import List, Union

from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor

#TODO: Ellaborate the ts processor arguments
_ts_processor_parameters = ['num_cols', 'cat_cols', 'pos_idx']
_ts_processor_parameters_df = [[], [], False]

TSProcessorArguments = namedtuple('TimeSeriesProcessorArguments',
                                    _ts_processor_parameters,
                                    defaults=_ts_processor_parameters_df)


class TimeSeriesModels(Enum):
    "Supported models for the TimeSeries Data Processor."
    TIMEGAN = 'TIMEGAN'
    TSCWGAN = 'TSCWGAN'


@typechecked
class TimeSeriesDataProcessor(BaseProcessor):
    """
    Not implemented.
    """
    def __init__(self, *, num_cols: Union[List[str], List[int]] = None, cat_cols: Union[List[str], List[int]] = None,
                 pos_idx: bool = False):
        raise NotImplementedError
