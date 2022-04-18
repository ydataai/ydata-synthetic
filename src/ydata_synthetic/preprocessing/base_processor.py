"Base class of Data Preprocessors, do not instantiate this class directly."
from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import List, Optional

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from typeguard import typechecked


# pylint: disable=R0902
@typechecked
class BaseProcessor(ABC, BaseEstimator, TransformerMixin):
    """
    This data processor works like a scikit learn transformer in with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns.
        cat_cols (list of strings):
            List of names of categorical columns.
    """
    def __init__(self, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols

        self._num_pipeline = None  # To be overriden by child processors
        self._cat_pipeline = None  # To be overriden by child processors

        self._col_transform_info = None  # Metadata object mapping inputs/outputs of each pipeline

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

    @property
    def col_transform_info(self) -> SimpleNamespace:
        """Returns a ProcessorInfo object specifying input/output feature mappings of this processor's pipelines."""
        self._check_is_fitted()
        if self._col_transform_info is None:
            self._col_transform_info = self.__create_metadata_synth()
        return self._col_transform_info

    def __create_metadata_synth(self) -> SimpleNamespace:
        def new_pipeline_info(feat_in, feat_out):
            return SimpleNamespace(feat_names_in = feat_in, feat_names_out = feat_out)
        if self.num_cols:
            num_info = new_pipeline_info(self.num_pipeline.feature_names_in_, self.num_pipeline.get_feature_names_out())
        else:
            num_info = new_pipeline_info([], [])
        if self.cat_cols:
            cat_info = new_pipeline_info(self.cat_pipeline.feature_names_in_, self.cat_pipeline.get_feature_names_out())
        else:
            cat_info = new_pipeline_info([], [])
        return SimpleNamespace(numerical=num_info, categorical=cat_info)

    def _check_is_fitted(self):
        """Checks if the processor is fitted by testing the numerical pipeline.
        Raises NotFittedError if not."""
        if self._num_pipeline is None:
            raise NotFittedError("This data processor has not yet been fitted.")

    def _validate_cols(self, x_cols):
        """Ensures validity of the passed numerical and categorical columns.
        The following is verified:
            1) Num cols and cat cols are disjoint sets;
            2) The union of these sets should equal x_cols;.
        Assertion errors are raised in case any of the tests fails."""
        missing = set(x_cols).difference(set(self.num_cols).union(set(self.cat_cols)))
        intersection = set(self.num_cols).intersection(set(self.cat_cols))
        assert intersection == set(), f"num_cols and cat_cols share columns {intersection} but should be disjoint."
        assert missing == set(), f"The columns {missing} of the provided dataset were not attributed to a pipeline."

    # pylint: disable=C0103
    @abstractmethod
    def fit(self, X: DataFrame) -> BaseProcessor:
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
            transformed (ndarray): Processed version of the passed DataFrame.
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
