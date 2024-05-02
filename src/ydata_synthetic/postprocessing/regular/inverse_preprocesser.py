# Inverts all preprocessing pipelines provided in the preprocessing examples
import pandas as pd
import numpy as np
from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, FunctionTransformer

def inverse_transform(data: pd.DataFrame, processor: Union[Pipeline, ColumnTransformer, BaseEstimator]) -> pd.DataFrame:
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
        for t_name, t in processor.transformers_:
            if t_name == 'drop':
                continue
            elif t_name == 'passthrough':
                inv_data[t[1]] = data[t[1]]
            else:
                t_data = data.iloc[:, processor.get_feature_names_out()[processor.transformers_[::-1].index((t_name, t))[0][1]:processor.transformers_[::-1].index((t_name, t))[0][1] + len(t[1])]]
                if isinstance(t[0], FunctionTransformer):
                    inv_data.iloc[:, processor.transformers_[::-1].index((t_name, t))[0][1]:processor.transformers_[::-
