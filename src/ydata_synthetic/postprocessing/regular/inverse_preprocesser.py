# Inverts all preprocessing pipelines provided in the preprocessing examples
from os import listdir
from os.path import isfile, join
import sys
from typing import Union

import pandas as pd
import numpy as np

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
        inv_data = processor.inverse_transform(data)
    elif isinstance(processor, ColumnTransformer):
        assert isinstance(data, pd.DataFrame), "The data to be inverted from a ColumnTransformer has to be a Pandas DataFrame."
        for t_name, t, t_cols in processor.transformers_[::-1]:
            assert set(t_cols).issubset(set(data.columns)), "The transformed cols cannot be mapped to the provided DataFrame."
            inv_data[t_cols] = t.inverse_transform(data[t_cols])
    else:
        print('The provided data processor is not supported and cannot be inverted with this method.')
        return None
    return inv_data

def __test_inverter():
    preproc_path = 'src/ydata_synthetic/preprocessing/regular'
    preproc_scripts = [f.split('.')[0] for f in listdir(preproc_path) if isfile(join(preproc_path, f)) and not f.startswith('_')]
    eval_str = "from ydata_synthetic.preprocessing.regular.{} import transformations"
    data_pointers = {'cardiovascular': None,
                     'credit_fraud': pd.read_csv('data/creditcard.csv', index_col=[0])}
    for script in preproc_scripts:
        try:
            exec(eval_str.format(script))
            data = data_pointers.get(script)
            if data is not None:
                data, processed_data, processor = transformations(data)
            else:
                data, processed_data, processor = transformations()
            inv_data = inverter(processed_data, processor)
            non_recovered_cols = sum(~np.isclose(data, inv_data).any(0))
            if non_recovered_cols > 0:
                print(f"Warning, {non_recovered_cols} are not similar to the original data after inversion.")
        except:
            continue

if __name__ == '__main__':
    __test_inverter()
    from ydata_synthetic.preprocessing.regular.credit_fraud import *
    data = pd.read_csv('data/creditcard.csv', index_col=[0])
    data, processed_data, processor = transformations(data)
    inv_data = inverter(processed_data, processor)
    print('!')
