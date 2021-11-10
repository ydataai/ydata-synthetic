from abc import ABC, abstractmethod
from typing import List, Union

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from typeguard import typechecked

@typechecked
class AbstractProcessor(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class for Data Preprocessing.
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

        self._col_map = {'numerical': [] if num_cols is None else num_cols,
                         'categorical': [] if cat_cols is None else cat_cols}

        self._col_idxs = {'numerical': None,
                         'categorical': None}

        self._pipeline = {'numerical': None,
                          'categorical': None}

        self._types = None
        self.col_order_ = None
        self.pos_idx = pos_idx

    @abstractmethod
    def fit(self, X: DataFrame):
        """Fits the DataProcessor to a passed DataFrame.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the num/cat columns defined in initialization.
        Returns:
            self (DataProcessor): The fitted data processor.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: DataFrame) -> ndarray:
        """Transforms the passed DataFrame with the fit DataProcessor.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the columns types defined in initialization.
        Returns:
            transformed (ndarray):
                Processed version of the passed DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, X: ndarray) -> DataFrame:
        """Inverts the data transformation pipelines on a passed DataFrame.
        Args:
            X (ndarray):
                Numpy array to be brought back to the original data format.
                Should share the schema of data transformed by this DataProcessor.
                Can be used to revert transformations of training data or for synthetic samples.
        Returns:
            result (DataFrame):
                DataFrame with all performed transformations inverted.
        """
        raise NotImplementedError
