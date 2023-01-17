# Inverts all preprocessing pipelines provided in the preprocessing examples
from typing import Union

from pandas import DataFrame, concat

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, MinMaxScaler


def inverse_transform(data: DataFrame, processor: Union[Pipeline, ColumnTransformer, PowerTransformer,
                                                           OneHotEncoder, StandardScaler, MinMaxScaler]) -> DataFrame:
    """Inverts data transformations taking place in a standard sklearn processor.
    Supported processes are sklearn pipelines, column transformers or base estimators like standard scalers.

    Args:
        data (DataFrame): The data object that needs inversion of preprocessing
        processor (Union[Pipeline, ColumnTransformer, BaseEstimator]): The processor applied on the original data

    Returns:
        inv_data (DataFrame): The data object after inverting preprocessing"""
    inv_data = data.copy()
    if isinstance(processor, (PowerTransformer, OneHotEncoder, StandardScaler, MinMaxScaler, Pipeline)):
        inv_data = DataFrame(processor.inverse_transform(data), columns=processor.feature_names_in_ if hasattr(processor, "feature_names_in") else None)
    elif isinstance(processor, ColumnTransformer):
        output_indices = processor.output_indices_
        assert isinstance(data, DataFrame), "The data to be inverted from a ColumnTransformer has to be a Pandas DataFrame."
        for t_name, t, t_cols in processor.transformers_[::-1]:
            slice_ = output_indices[t_name]
            t_indices = list(range(slice_.start, slice_.stop, 1 if slice_.step is None else slice_.step))
            if t == 'drop':
                continue
            elif t == 'passthrough':
                inv_cols = DataFrame(data.iloc[:,t_indices].values, columns = t_cols, index = data.index)
                inv_col_names = inv_cols.columns
            else:
                inv_cols = DataFrame(t.inverse_transform(data.iloc[:,t_indices].values), columns = t_cols, index = data.index)
                inv_col_names = inv_cols.columns
            if set(inv_col_names).issubset(set(inv_data.columns)):
                inv_data[inv_col_names] = inv_cols[inv_col_names]
            else:
                inv_data = concat([inv_data, inv_cols], axis=1)
    else:
        print('The provided data processor is not supported and cannot be inverted with this method.')
        return None
    return inv_data[processor.feature_names_in_] if hasattr(processor, "feature_names_in") else inv_data
