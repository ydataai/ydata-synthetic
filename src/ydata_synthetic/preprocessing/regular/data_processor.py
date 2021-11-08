from typing import List

from numpy import concatenate, ndarray, split, zeros
from pandas import concat, DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typeguard import typechecked


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Main class for Data Preprocessing. It is a base version.
    It works like any other transformer in scikit lear with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns
        cat_cols (list of strings):
            List of names of categorical columns
    """
    @typechecked
    def __init__(self, *, num_cols: List[str] = None, cat_cols: List[str] = None):

        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols

        self.num_col_idx_ = None
        self.cat_col_idx_ = None

        self.num_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        self.cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        self._types = None
        self.col_order_ = None

    @typechecked
    def fit(self, X: DataFrame):
        self.col_order_ = [c for c in X.columns if c in self.num_cols + self.cat_cols]
        self._types = X.dtypes

        self.num_pipeline.fit(X[self.num_cols]) if self.num_cols else zeros([len(X), 0])
        self.cat_pipeline.fit(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0])

        return self

    def transform(self, X: DataFrame) -> ndarray:
        num_data = self.num_pipeline.transform(X[self.num_cols]) if self.num_cols else zeros([len(X), 0])
        cat_data = self.cat_pipeline.transform(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0])

        transformed = concatenate([num_data, cat_data], axis=1)

        self.num_col_idx_ = num_data.shape[1]
        self.cat_col_idx_ = self.num_col_idx_ + cat_data.shape[1]

        return transformed

    def fit_transform(self, X: DataFrame, y=None, **fit_params):
        return super().fit_transform(X)

    def inverse_transform(self, X: ndarray) -> DataFrame:
        num_data, cat_data, _ = split(X, [self.num_col_idx_, self.cat_col_idx_], axis=1)

        num_data = self.num_pipeline.inverse_transform(num_data) if self.num_cols else zeros([len(X), 0])
        cat_data = self.cat_pipeline.inverse_transform(cat_data) if self.cat_cols else zeros([len(X), 0])

        result = concat([DataFrame(num_data, columns=self.num_cols),
                            DataFrame(cat_data, columns=self.cat_cols),], axis=1)

        result = result.loc[:, self.col_order_]

        for col in result.columns:
            result[col]=result[col].astype(self._types[col])

        return result
