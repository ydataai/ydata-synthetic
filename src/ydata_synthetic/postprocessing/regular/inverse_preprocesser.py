# Inverts all preprocessing pipelines provided in the preprocessing examples
from os import listdir
from os.path import isfile, join
import sys
from typing import Union

import pandas as pd
from numpy import isclose, ndarray

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler


def inverter(data: pd.DataFrame, processor: Union[Pipeline, ColumnTransformer, PowerTransformer, OneHotEncoder, StandardScaler]) -> pd.DataFrame:
    """Inverts data transformations taking place in a standard sklearn processor.
    Supported processes are sklearn pipelines, column transformers or base estimators like standard scalers.

    Args:
        data (pd.DataFrame): The data object that needs inversion of preprocessing
        processor (Union[Pipeline, ColumnTransformer, BaseEstimator]): The processor applied on the original data

    Returns:
        inv_data (pd.DataFrame): The data object after inverting preprocessing"""
    inv_data = data.copy()
    if isinstance(processor, (PowerTransformer, OneHotEncoder, StandardScaler, Pipeline)):
        inv_data = pd.DataFrame(processor.inverse_transform(data), columns=processor.feature_names_in_)
    elif isinstance(processor, ColumnTransformer):
        to_drop = []
        output_indices = processor.output_indices_
        assert isinstance(data, pd.DataFrame), "The data to be inverted from a ColumnTransformer has to be a Pandas DataFrame."
        for t_name, t, t_cols in processor.transformers_[::-1]:
            slice_ = output_indices[t_name]
            t_indices = list(range(slice_.start, slice_.stop, 1 if slice_.step is None else slice_.step))
            if t_name == 'remainder':
                continue
            else:
                inv_cols = pd.DataFrame(t.inverse_transform(data.iloc[:,t_indices].values), columns = t_cols, index = data.index)
            if len(t_indices) != len(t_cols):
                to_drop += t_indices
            inv_data = pd.concat([inv_data, inv_cols], axis=1)
        inv_data.drop(columns=to_drop, inplace=True)
    else:
        print('The provided data processor is not supported and cannot be inverted with this method.')
        return None
    return inv_data

if __name__ == '__main__':
    def non_recovered_cols(orig_data, inv_data):
        comparable_cols = [col for col in inv_data.columns if col in orig_data.columns]
        non_recovered_cols = sum(~isclose(data[comparable_cols], inv_data[comparable_cols]).any(0))
        if non_recovered_cols > 0:
            print(f"Warning, {non_recovered_cols}/{len(comparable_cols)} are not similar to the original data after inversion.")
        else:
            print(f"All {len(comparable_cols)} columns were successfully inverted.")

    from ydata_synthetic.preprocessing.regular.adult import transformations
    data, processed_data, processor = transformations()
    inv_data = inverter(processed_data, processor)
    non_recovered_cols(data, inv_data)

    from ydata_synthetic.preprocessing.regular.breast_cancer_wisconsin import transformations
    data, processed_data, processor = transformations()
    inv_data = inverter(processed_data, processor)
    non_recovered_cols(data, inv_data)

    from ydata_synthetic.preprocessing.regular.credit_fraud import transformations
    data = pd.read_csv('data/creditcard.csv', index_col=[0])
    data, processed_data, processor = transformations(data)
    inv_data = inverter(processed_data, processor)
    non_recovered_cols(data, inv_data)
