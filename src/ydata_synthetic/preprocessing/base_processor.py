"Implements a BaseProcessor Class, not meant to be directly instantiated."
from abc import ABC, abstractmethod
from typing import List, Optional

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
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
    def __init__(self, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):

        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols

        self._num_pipeline = None
        self._cat_pipeline = None

        self._types = None

    @property
    def num_pipeline(self) -> BaseEstimator:
        """Returns the pipeline applied to numerical columns."""
        return self._num_pipeline

    @property
    def cat_pipeline(self) -> BaseEstimator:
        """Returns the pipeline applied to categorical columns."""
        return self._cat_pipeline

    @property
    def types(self) -> Series:
        """Returns a Series with the dtypes of each column in the fitted DataFrame."""
        return self._types

    def _check_is_fitted(self):
        """Checks if the processor is fitted by testing the numerical pipeline.
        Raises NotFittedError if not."""
        if self._num_pipeline is None:
            raise NotFittedError("This data processor has not yet been fitted.")

    # pylint: disable=C0103
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

    # pylint: disable=C0103
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

    # pylint: disable=C0103
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
