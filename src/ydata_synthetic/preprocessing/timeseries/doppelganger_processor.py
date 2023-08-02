from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass

from numpy import concatenate, ndarray, zeros, ones, expand_dims, reshape, sum as npsum, repeat, array_split, asarray
from pandas import DataFrame
from typeguard import typechecked

from ydata_synthetic.preprocessing.regular.processor import RegularDataProcessor


@dataclass
class ColumnMetadata:
    """
    Dataclass that stores the metadata of each column.
    """
    discrete: bool
    output_dim: int
    name: str


@typechecked
class DoppelGANgerProcessor(RegularDataProcessor):
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
                 sequence_length: Optional[int] = None):
        super().__init__(num_cols, cat_cols)

        if num_cols is None:
            num_cols = []
        if cat_cols is None:
            cat_cols = []
        if measurement_cols is None:
            measurement_cols = []
        self.sequence_length = sequence_length
        self._measurement_num_cols = [c for c in self.num_cols if c in measurement_cols]
        self._measurement_cat_cols = [c for c in self.cat_cols if c in measurement_cols]
        self._attribute_num_cols = [c for c in self.num_cols if c not in measurement_cols]
        self._attribute_cat_cols = [c for c in self.cat_cols if c not in measurement_cols]
        self._measurement_cols_metadata = None
        self._attribute_cols_metadata = None
        self._measurement_one_hot_cat_cols = None
        self._attribute_one_hot_cat_cols = None
        self._has_attributes = self._attribute_num_cols or self._attribute_cat_cols

    @property
    def measurement_cols_metadata(self):
        return self._measurement_cols_metadata

    @property
    def attribute_cols_metadata(self):
        return self._attribute_cols_metadata

    def add_gen_flag(self, data_features: ndarray, sample_len: int):
        num_sample = data_features.shape[0]
        length = data_features.shape[1]
        if length % sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
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

        measurement_cols = self._measurement_num_cols + self._measurement_cat_cols
        if not measurement_cols:
            raise ValueError("At least one measurement column must be supplied.")
        if not all(c in self.num_cols + self.cat_cols for c in measurement_cols):
            raise ValueError("At least one of the supplied measurement columns does not exist in the dataset.")
        if self.sequence_length is None:
            raise ValueError("The sequence length is mandatory.")

        num_data = DataFrame(self.num_pipeline.transform(X[self.num_cols]) if self.num_cols else zeros([len(X), 0]), columns=self.num_cols)
        one_hot_cat_cols = self.cat_pipeline.get_feature_names_out()
        cat_data = DataFrame(self.cat_pipeline.transform(X[self.cat_cols]) if self.cat_cols else zeros([len(X), 0]), columns=one_hot_cat_cols)

        self._measurement_one_hot_cat_cols = [c for c in one_hot_cat_cols if c.split("_")[0] in self._measurement_cat_cols]
        measurement_num_data = num_data[self._measurement_num_cols].to_numpy() if self._measurement_num_cols else zeros([len(X), 0])
        self._measurement_cols_metadata = [ColumnMetadata(discrete=False, output_dim=1, name=c) for c in self._measurement_num_cols]
        measurement_cat_data = cat_data[self._measurement_one_hot_cat_cols].to_numpy() if self._measurement_one_hot_cat_cols else zeros([len(X), 0])
        self._measurement_cols_metadata += [ColumnMetadata(discrete=True, output_dim=X[c].nunique(), name=c) for c in self._measurement_cat_cols]
        data_features = concatenate([measurement_num_data, measurement_cat_data], axis=1)

        if self._has_attributes:
            self._attribute_one_hot_cat_cols = [c for c in one_hot_cat_cols if c.split("_")[0] in self._attribute_cat_cols]
            attribute_num_data = num_data[self._attribute_num_cols].to_numpy() if self._attribute_num_cols else zeros([len(X), 0])
            self._attribute_cols_metadata = [ColumnMetadata(discrete=False, output_dim=1, name=c) for c in self._attribute_num_cols]
            attribute_cat_data = cat_data[self._attribute_one_hot_cat_cols].to_numpy() if self._attribute_one_hot_cat_cols else zeros([len(X), 0])
            self._attribute_cols_metadata += [ColumnMetadata(discrete=True, output_dim=X[c].nunique(), name=c) for c in self._attribute_cat_cols]
            data_attributes = concatenate([attribute_num_data, attribute_cat_data], axis=1)
        else:
            self._attribute_one_hot_cat_cols = []
            data_attributes = zeros((data_features.shape[0], 1))
            self._attribute_cols_metadata = [ColumnMetadata(discrete=False, output_dim=1, name="zeros_attribute")]

        num_samples = int(X.shape[0] / self.sequence_length)
        data_features = asarray(array_split(data_features, num_samples))
        data_attributes = asarray(array_split(data_attributes, num_samples))

        data_features = self.add_gen_flag(data_features, sample_len=self.sequence_length)
        self._measurement_cols_metadata += [ColumnMetadata(discrete=True, output_dim=2, name="gen_flags")]
        return data_features, data_attributes.mean(axis=1)

    def inverse_transform(self, X_features: ndarray, X_attributes: ndarray) -> list[DataFrame]:
        """Inverts the data transformation pipelines on a passed DataFrame.
        Args:
            X_features (ndarray):
                Numpy array with the measurement data to be brought back to the original format.
            X_attributes (ndarray):
                Numpy array with the attribute data to be brought back to the original format.
        Returns:
            result (DataFrame):
                DataFrame with all performed transformations inverted.
        """
        self._check_is_fitted()

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
            df_num = self.num_pipeline.inverse_transform(df[self.num_cols]) if self.num_cols else zeros([len(df), 0])
            df_cat = self.cat_pipeline.inverse_transform(df[one_hot_cat_cols].round(0)) if self.cat_cols else zeros([len(df), 0])
            df = DataFrame(concatenate((df_num, df_cat), axis=1), columns=self.num_cols+self.cat_cols)
            df = df.loc[:, self._col_order_]
            for col in df.columns:
                df[col] = df[col].astype(self._types[col])
            samples.append(df)

        return samples
