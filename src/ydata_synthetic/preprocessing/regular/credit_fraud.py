#Data transformations to be applied
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def transformations(data):
    #Log transformation to Amount variable
    processed_data = data.copy()
    data_cols = list(data.columns[data.columns != 'Class'])

    data_transformer = Pipeline(steps=[
        ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True, copy=True))])

    preprocessor = ColumnTransformer(
        transformers = [('power', data_transformer, data_cols)])
    processed_data[data_cols] = preprocessor.fit_transform(data[data_cols])

    return data, processed_data, preprocessor
