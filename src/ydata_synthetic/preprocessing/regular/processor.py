from typing import List, Union

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor

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
    def __init__(self, *, num_cols: Union[List[str], List[int]] = None, cat_cols: Union[List[str], List[int]] = None,
                 pos_idx: bool = False):
        super().__init__(num_cols = num_cols, cat_cols = cat_cols, pos_idx = pos_idx)

        self.num_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        self.cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
