from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, List, Optional, Tuple

from numpy import concatenate, ndarray, zeros, ones, expand_dims, reshape, sum as npsum, repeat, array_split, asarray, amin, amax, stack
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ydata_synthetic.preprocessing.base_processor import BaseProcessor

class ColumnMetadata:
    """
    Dataclass that stores the metadata of each column.
    """
    def __init__(self, discrete: bool, output_dim: int, name: str, real: bool = True):
        self.discrete = discrete
        self.output_dim = output_dim
        self.name = name
        self.real = real

class DoppelGANgerProcessor(BaseProcessor):
    """
    Main class for class the DoppelGANger preprocessing.
    It works like any other transformer in scikit learn with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns.
        cat_cols (list of strings):
            List of categorical columns.
        measurement_cols (list of strings):
            List of measurement columns.
        sequence_length (int):
            Sequence length.
    """
    SUPPORTED_MODEL = 'DoppelGANger'

    def __init__(self, num_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 measurement_cols: Optional[List[str]] = None,
                 sequence_length: Optional[int] = None,
                 sample_length: Optional[int] = None,
                 normalize_tanh: Optional[bool] = None):
        super().__init__(num_cols, cat_cols)

        self.sequence_length = sequence_length
        self.sample_length = sample_length
        self.normalize_tanh = normalize_tanh

        self._validate_input_data()

        self._measurement_num_cols = [c for c in self.num_cols if c in measurement_cols]
        self._measurement_cat_cols = [c for c in self.cat_cols if c in measurement_cols]
        self._attribute_num_cols = [c for c in self.num_cols if c not in measurement_cols]
        self._attribute_cat_cols = [c for c in self.cat_cols if c not in measurement_cols]
        self._measurement_cols_metadata = None
        self._attribute_cols_metadata = None
        self._measurement_one_hot_cat_cols = None
        self._attribute_one_hot_cat_cols = None
        self._has_attributes = bool(self._attribute_num_cols or self._attribute_cat_cols)
        self._eps = 1e-4

    def _validate_input_data(self):
        if self.num_cols is None or self.cat_cols is None:
            raise ValueError("Both num_cols and cat_cols cannot be None.")

    def fit(self, X: DataFrame) -> DoppelGANgerProcessor:
        """Fits the data processor to a passed DataFrame.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the num/cat columns defined in initialization.
        Returns:
            self (DoppelGANgerProcessor): The fitted data processor.
        """
        self._validate_cols(X.columns)

        measurement_cols = self._measurement_num_cols + self._measurement_cat_cols
        if not measurement_cols:
            raise ValueError("At least one measurement column must be supplied.")
        if not all(c in self.num_cols + self.cat_cols for c in measurement_cols):
            raise ValueError("At least one of the supplied measurement columns does not exist in the dataset.")
        if self.sequence_length is None:
            raise ValueError("The sequence length is mandatory.")

        self._col_order_ = [c for c in X.columns if c in self.num_cols + self.cat_cols]
        self._types = X.dtypes
        self._num_pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
        ])
        self._cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')),
        ])
        self._num_pipeline.fit(X[self._attribute_num_cols]) if self._attribute_num_cols else zeros([len(X), 0])
        self._cat_pipeline.fit(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0])

        return self

    def transform(self, X: DataFrame) -> Tuple[ndarray, ndarray]:
        """Transforms the passed DataFrame with the fit DataProcessor.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the columns types defined in initialization.
        Returns:
            transformed (ndarray, ndarray):
                Processed version of the passed DataFrame.
        """
        self._check_is_fitted()

        one_hot_cat_cols_out = self._cat_pipeline.get_feature_names_out() if self.cat_cols else []
        cat_data = DataFrame(self._cat_pipeline.transform(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0]), columns=one_hot_cat_cols_out)

        self._measurement_one_hot_cat_cols = [c for c in one_hot_cat_cols_out if c.startswith(tuple(self._measurement_cat_cols))]  # .split("_")[0]
        self._measurement_cols_metadata = [ColumnMetadata(discrete=False,
                                                          output_dim=1,
                                                          name=c) for c in self._measurement_num_cols]
        measurement_cat_data = cat_data[self._measurement_one_hot_cat_cols].to_numpy() if self._measurement_one_hot_cat_cols else zeros([len(X), 0])
        self._measurement_cols_metadata += [ColumnMetadata(discrete=True,
                                                           output_dim=X[c].nunique() if X[c].nunique() != 2 else 1,
                                                           name=c) for c in self._measurement_cat_cols]
        data_features = concatenate([X[self._measurement_num_cols].to_numpy(), measurement_cat_data], axis=1)

        if self._has_attributes:
            self._attribute_one_hot_cat_cols = [c for c in one_hot_cat_cols_out if c.startswith(tuple(self._attribute_cat_cols))]  # .split("_")[0]
            attribute_num_data = self._num_pipeline.transform(X[self._attribute_num_cols]) if self._attribute_num_cols else zeros([len(X), 0])
            self._attribute_cols_metadata = [ColumnMetadata(discrete=False,
                                                            output_dim=1,
                                                            name=c) for c in self._attribute_num_cols]
            attribute_cat_data = cat_data[self._attribute_one_hot_cat_cols].to_numpy() if self._attribute_one_hot_cat_cols else zeros([len(X), 0])
            self._attribute_cols_metadata += [ColumnMetadata(discrete=True,
                                                             output_dim=X[c].nunique() if X[c].nunique() != 2 else 1,
                                                             name=c) for c in self._attribute_cat_cols]
            data_attributes = concatenate([attribute_num_data, attribute_cat_data], axis=1)
        else:
            data_attributes = zeros((data_features.shape[0], 0))
            self._attribute_one_hot_cat_cols = []
            self._attribute_cols_metadata = []

        num_samples = int(X.shape[0] / self.sequence_length)
        data_features = asarray(array_split(data_features, num_samples))

        additional_attributes = []
        for ix, col_meta in enumerate(self._measurement_cols_metadata):
            if not col_meta.discrete:
                col_data = X[col_meta.name].to_numpy().reshape(num_samples, -1)
                max_col = amax(col_data, axis=1) + self._eps
                min_col = amin(col_data, axis=1) - self._eps
                additional_attributes.append((max_col + min_col) / 2.0)
                additional_attributes.append((max_col - min_col) / 2.0)
                self._attribute_cols_metadata += [ColumnMetadata(discrete=False,
                                                                 output_dim=1,
                                                                 name=f"addi_{col_meta.name}_{ix}",
                                                                 real=False) for ix in range (1, 3)]
                max_col = expand_dims(max_col, axis=1)
                min_col = expand_dims(min_col, axis=1)
                data_features[:, :, ix] = (data_features[:, :, ix] - min_col) / (max_col - min_col)
                if self.normalize_tanh:
                    data_features[:, :, ix] = data_features[:, :, ix] * 2.0 - 1.0

        data_attributes = asarray(array_split(data_attributes, num_samples))
        data_attributes = data_attributes.mean(axis=1)

        if additional_attributes:
            additional_attributes = stack(additional_attributes, axis=1)
            data_attributes = concatenate([data_attributes, additional_attributes], axis=1)

        data_features = self.add_gen_flag(data_features, sample_len=self.sample_length)
        self._measurement_cols_metadata += [ColumnMetadata(discrete=True, output_dim=2, name="gen_flags")]
        return data_features, data_attributes

