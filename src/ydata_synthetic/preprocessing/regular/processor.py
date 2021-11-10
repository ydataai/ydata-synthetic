from collections import namedtuple
from enum import Enum
from typing import List, Union

from numpy import concatenate, ndarray, split, zeros
from pandas import DataFrame, concat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typeguard import typechecked

from ydata_synthetic.preprocessing.base_processor import AbstractProcessor

_regular_processor_parameters = ['num_cols', 'cat_cols', 'pos_idx']
_regular_processor_parameters_df = [[], [], False]

RegProcessorArguments = namedtuple('RegularProcessorParameters',
                                    _regular_processor_parameters,
                                    defaults=_regular_processor_parameters_df)

class RegularModels(Enum):
    "Supported models for the Regular Data Processor."
    CGAN = 'CGAN'
    CRAMERGAN = 'CramerGAN'
    DRAGAN = 'DRAGAN'
    GAN = 'VanillaGAN'
    WGAN = 'WGAN'
    WGAN_GP = 'WGAN_GP'

@typechecked
class RegularDataProcessor(AbstractProcessor):
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
    def __init__(self, reg_processor_args: RegProcessorArguments):
        super().__init__(num_cols = reg_processor_args.num_cols,
                         cat_cols = reg_processor_args.cat_cols,
                         pos_idx = reg_processor_args.pos_idx)

        self._pipeline['numerical'] = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        self._pipeline['categorical'] = Pipeline([
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

    def fit(self, X: DataFrame):
        """Fits the DataProcessor to a passed DataFrame.
        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the num/cat columns defined in initialization.
        Returns:
            self (RegularDataProcessor): The fitted data processor.
        """
        if self.pos_idx:
            self._col_map = {key: list(X.columns[val]) for key, val in self._col_map}
        self.col_order_ = [c for c in X.columns if c in sum(self._col_map.values(), [])]
        self._types = X.dtypes

        for col_type in self._pipeline:
            self._pipeline[col_type].fit(X[self._col_map[col_type]]) if self._col_map[col_type] else zeros([len(X), 0])

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
        data = {}
        i = 0
        for col_type, pipeline in self._pipeline.items():
            data[col_type] = pipeline.transform(X[self._col_map[col_type]]) if self._col_map[col_type] else zeros([len(X), 0])
            self._col_idxs[col_type] = i + data[col_type].shape[1]
            i += data[col_type].shape[1]

        transformed = concatenate(list(data.values()), axis=1)

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
        data_blocks = split(X, [col_type_idx for col_type_idx in self._col_idxs.values()], axis=1)

        data = {}
        for i, (col_type, pipeline) in enumerate(self._pipeline.items()):
            data[col_type] = pipeline.inverse_transform(data_blocks[i]) if self._col_map[col_type] else zeros([len(X), 0])

        result = concat([DataFrame(data_, columns=cols) for data_, cols in zip(data.values(), self._col_map.values())], axis=1)

        result = result.loc[:, self.col_order_]

        for col in result.columns:
            result[col]=result[col].astype(self._types[col])

        return result
