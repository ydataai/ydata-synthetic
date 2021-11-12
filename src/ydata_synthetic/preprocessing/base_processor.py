from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from typeguard import typechecked

@typechecked
class BaseProcessor(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for Data Preprocessing.
    It works like any other transformer in scikit learn with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns.
        cat_cols (list of strings):
            List of names of categorical columns.
    """
    def __init__(self, *, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):

        self._col_map = {'numerical': [] if num_cols is None else num_cols,
                         'categorical': [] if cat_cols is None else cat_cols}

        self._col_idxs = {'numerical': None,
                         'categorical': None}

        self._pipeline = {'numerical': None,
                          'categorical': None}

        self._types = None
        self.col_order_ = None

    @property
    def pipeline(self) -> Dict[str, BaseEstimator]:
        """Returns a dictionary mapping column type names to its respective pipeline."""
        return self._pipeline

    @property
    def col_map(self) -> Dict[str, List[str]]:
        """Returns a dictionary mapping column type names to its respective list of members."""
        return self._col_map

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
