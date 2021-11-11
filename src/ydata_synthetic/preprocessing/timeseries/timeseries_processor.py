from typing import List, Union

from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor


@typechecked
class TimeSeriesDataProcessor(BaseProcessor):
    """
    Not implemented.
    """
    def __init__(self, *, num_cols: Union[List[str], List[int]] = None, cat_cols: Union[List[str], List[int]] = None,
                 pos_idx: bool = False):
        raise NotImplementedError
