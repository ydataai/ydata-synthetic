from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass

from numpy import concatenate, ndarray, zeros, ones, expand_dims, reshape, sum as npsum, repeat, array_split, asarray, amin, amax, stack
from pandas import DataFrame
from typeguard import typechecked
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ydata_synthetic.preprocessing.base_processor import BaseProcessor


@dataclass
class ColumnMetadata:
    """
    Dataclass that stores the metadata of each column.
    """
    discrete: bool
    output_dim: int
    name: str
    real: bool = True


@typechecked
class DoppelGANgerProcessor(BaseProcessor):
    """
    Main class for class the DoppelGANger preprocessing.
    It works like any other transformer in scikit learn with the methods fit, transform and inverse transform.
    Args:
        num_cols (list of strings):
            List of names of numerical columns.
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

        if num_cols is None:
            num_cols = []
        if cat_cols is None:
            cat_cols = []
        if measurement_cols is None:
            measurement_cols = []
        if normalize_tanh is None:
            normalize_tanh = False

        self._col_order_ = None
        self.sequence_length = sequence_length
        self.sample_length = sample_length
        self.normalize_tanh = normalize_tanh

        if self.sequence_length is not None and self.sample_length is not None:
            if self.sequence_length % self.sample_length != 0:
                raise ValueError("The sequence length must be a multiple of the sample length.")

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

    @property
    def measurement_cols_metadata(self):
        return self._measurement_cols_metadata

    @property
    def attribute_cols_metadata(self):
        return self._attribute_cols_metadata

    def add_gen_flag(self, data_features: ndarray, sample_len: int):
        num_sample = data_features.shape[0]
        length = data_features.shape[1]
        data_gen_flag = ones((num_sample, length))
        data_gen_flag = expand_dims(data_gen_flag, 2)
        shift_gen_flag = concatenate(
            [data_gen_flag[:, 1:, :],
            zeros((data_gen_flag.shape[0], 1, 1))],
            axis=1)
        data_gen_flag_t = reshape(
            data_gen_flag,
            [num_sample, int(length / sample_len), sample_len])
        data_gen_flag_t = npsum(data_gen_flag_t, 2)
        data_gen_flag_t = data_gen_flag_t > 0.5
        data_gen_flag_t = repeat(data_gen_flag_t, sample_len, axis=1)
        data_gen_flag_t = expand_dims(data_gen_flag_t, 2)
        data_features = concatenate(
            [data_features,
            shift_gen_flag,
            (1 - shift_gen_flag) * data_gen_flag_t],
            axis=2)

        return data_features

    # pylint: disable=W0106
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

    def transform(self, X: DataFrame) -> tuple[ndarray, ndarray]:
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

    def inverse_transform(self, X_features: ndarray, X_attributes: ndarray, gen_flags: ndarray) -> list[DataFrame]:
        """Inverts the data transformation pipelines on a passed DataFrame.
        Args:
            X_features (ndarray):
                Numpy array with the measurement data to be brought back to the original format.
            X_attributes (ndarray):
                Numpy array with the attribute data to be brought back to the original format.
            gen_flags (ndarray):
                Numpy array with the flags indicating the activation of features.
        Returns:
            result (DataFrame):
                DataFrame with all performed transformations inverted.
        """
        self._check_is_fitted()

        addi_cols_idx = addi_cols_idx_start = sum([c.output_dim for c in self._attribute_cols_metadata if c.real])
        for m_col_ix in range(len(self._measurement_num_cols)):
            max_plus_min = X_attributes[:, addi_cols_idx]
            max_minus_min = X_attributes[:, addi_cols_idx + 1]
            max_val = expand_dims(max_plus_min + max_minus_min, axis=1)
            min_val = expand_dims(max_plus_min - max_minus_min, axis=1)
            if self.normalize_tanh:
                X_features[:, :, m_col_ix] = (X_features[:, :, m_col_ix] + 1.0) / 2.0
            X_features[:, :, m_col_ix] = X_features[:, :, m_col_ix] * (max_val - min_val) + min_val
            addi_cols_idx += 2

        X_features = X_features * expand_dims(gen_flags, axis=2)
        X_attributes = X_attributes[:, :addi_cols_idx_start]

        num_samples = X_attributes.shape[0]
        if self._has_attributes:
            X_attributes = repeat(X_attributes.reshape((num_samples, 1, X_attributes.shape[1])), repeats=X_features.shape[1], axis=1)
            generated_data = concatenate((X_features, X_attributes), axis=2)
        else:
            generated_data = X_features
        output_cols = self._measurement_num_cols + self._measurement_one_hot_cat_cols + self._attribute_num_cols + self._attribute_one_hot_cat_cols
        one_hot_cat_cols = self._measurement_one_hot_cat_cols + self._attribute_one_hot_cat_cols

        samples = []
        for i in range(num_samples):
            df = DataFrame(generated_data[i], columns=output_cols)
            df_num_feat = df[self._measurement_num_cols].to_numpy()
            df_num_attr = self._num_pipeline.inverse_transform(df[self._attribute_num_cols]) if self._attribute_num_cols else zeros([len(df), 0])
            df_cat = self._cat_pipeline.inverse_transform(df[one_hot_cat_cols]) if self.cat_cols else zeros([len(df), 0])
            df = DataFrame(concatenate((df_num_feat, df_num_attr, df_cat), axis=1), columns=self._measurement_num_cols+self._attribute_num_cols+self.cat_cols)
            df = df.loc[:, self._col_order_]
            for col in df.columns:
                df[col] = df[col].astype(self._types[col])
            samples.append(df)

        return samples
