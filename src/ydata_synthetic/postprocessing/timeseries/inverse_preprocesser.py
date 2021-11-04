from typing import Union, List

from ydata_synthetic.postprocessing.regular import inverse_preprocesser

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, MinMaxScaler

from pandas import DataFrame

def inverse_transform(data: List, processor: Union[Pipeline, ColumnTransformer, PowerTransformer, OneHotEncoder,
                                                   StandardScaler, MinMaxScaler]):
    if isinstance(data, list):
        data = DataFrame(data)
        return inverse_preprocesser.inverse_transform(data, processor).tolist()
    else:
        return inverse_preprocesser.inverse_transform(data, processor)
