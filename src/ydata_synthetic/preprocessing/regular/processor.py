"Implementation of a Regular DataProcessor."
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from numpy import concatenate, ndarray, split, zeros
from pandas import DataFrame, concat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import BaseProcessor


class RegularModels(Enum):
    "Supported models for the Regular Data Processor."
    CGAN = 'CGAN'
    CRAMERGAN = 'CramerGAN'
    DRAGAN = 'DRAGAN'
    GAN = 'VanillaGAN'
    WGAN = 'WGAN'
    WGAN_GP = 'WGAN_GP'
    CWGAN_GP = 'CWGAN_GP'


@typechecked
class RegularDataProcessor(BaseProcessor):
    """
    Main class for Regular/Tabular Data Preprocessing.
    It works like any other transformer in scikit learn with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns.
        cat_cols (list of strings):
            List of names of categorical columns.
    """
    def __init__(self, num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
        super().__init__(num_cols, cat_cols)

        self._col_order_ = None
        self._num_col_idx_ = None
        self._cat_col_idx_ = None

    # pylint: disable=W0106
    def fit(self, X: DataFrame) -> RegularDataProcessor:
        """Fits the DataProcessor to a passed DataFrame.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the num/cat columns defined in initialization.
        Returns:
            self (RegularDataProcessor): The fitted data processor.
        """
        self._validate_cols(X.columns)

        self._col_order_ = [c for c in X.columns if c in self.num_cols + self.cat_cols]

        self._types = X.dtypes

        self._num_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
        ])
        self._cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
        ])

        self.num_pipeline.fit(X[self.num_cols]) if self.num_cols else zeros([len(X), 0])
        self.cat_pipeline.fit(X[self.cat_cols]) if self.num_cols else zeros([len(X), 0])

        self._num_col_idx_ = len(self.num_pipeline.get_feature_names_out())
        self._cat_col_idx_ = self._num_col_idx_ + len(self.cat_pipeline.get_feature_names_out())

        return self

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
        self._check_is_fitted()

        num_data = self.num_pipeline.transform(X[self.num_cols]) if self.num_cols else zeros([len(X), 0])
        cat_data = self.cat_pipeline.transform(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0])

        transformed = concatenate([num_data, cat_data], axis=1)

        return transformed

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
        self._check_is_fitted()

        num_data, cat_data, _ = split(X, [self._num_col_idx_, self._cat_col_idx_], axis=1)

        num_data = self.num_pipeline.inverse_transform(num_data) if self.num_cols else zeros([len(X), 0])
        cat_data = self.cat_pipeline.inverse_transform(cat_data) if self.cat_cols else zeros([len(X), 0])

        result = concat([DataFrame(num_data, columns=self.num_cols),
                         DataFrame(cat_data, columns=self.cat_cols)], axis=1)

        result = result.loc[:, self._col_order_]

        for col in result.columns:
            result[col]=result[col].astype(self._types[col])

        return result
