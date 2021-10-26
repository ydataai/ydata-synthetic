# Inverts all preprocessing pipelines provided in the preprocessing examples
from os import listdir
from os.path import isfile, join
import sys
from typing import Union

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator


def inverter(data: pd.DataFrame, processor: Union[Pipeline, ColumnTransformer, BaseEstimator]) -> pd.DataFrame:
    """Inverts data transformations taking place in a standard sklearn processor.
    Supported processes are sklearn pipelines, column transformers or base estimators like standard scalers.

    Args:
        data (pd.DataFrame): The data object that needs inversion of preprocessing
        processor (Union[Pipeline, ColumnTransformer, BaseEstimator]): The processor applied on the original data

    Returns:
        inv_data (pd.DataFrame): The data object after inverting preprocessing"""

    return data


def __test_inverter():
    preproc_path = 'src/ydata_synthetic/preprocessing/regular'
    preproc_scripts = [f.split('.')[0] for f in listdir(preproc_path) if isfile(join(preproc_path, f)) and not f.startswith('_')]
    eval_str = "from ydata_synthetic.preprocessing.regular.{} import transformations"
    for script in preproc_scripts:
        try:
            print(eval_str.format(script))
            exec(eval_str.format(script))
            data, processed_data, processor = transformations()




if __name__ == '__main__':
    __test_inverter()
