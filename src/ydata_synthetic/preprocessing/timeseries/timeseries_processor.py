"Implementation of a TimeSeries DataProcessor."
from enum import Enum
from typing import List, Optional

from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor


class TimeSeriesModels(Enum):
    "Supported models for the TimeSeries Data Processor."
    TIMEGAN = 'TIMEGAN'
    TSCWGAN = 'TSCWGAN'


@typechecked
class TimeSeriesDataProcessor(BaseProcessor):
    """
    Not implemented.
    """
    def __init__(self, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
        raise NotImplementedError
