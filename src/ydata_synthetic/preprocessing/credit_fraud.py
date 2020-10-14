#Data transformations to be aplied 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PowerTransformer

def transformations(data):
    #Log transformation to Amount variable
    data_cols = list(data.columns[data.columns != 'Class'])
  
    #data[data_cols] = StandardScaler().fit_transform(data[data_cols])
    data[data_cols] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(data[data_cols])
    
    return data