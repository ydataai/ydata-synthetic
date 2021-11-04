"""
    Get the stock data from Yahoo finance data
    Data from the period 01 January 2017 - 24 January 2021
"""
from typing import Union, List

import pandas as pd

from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

def transformations(path, seq_len: int, cols: Union[str, List] = None):
    """Apply min max scaling and roll windows of a temporal dataset.

    Args:
        path(str): path to a csv temporal dataframe
        seq_len(int): length of the rolled sequences
        cols (Union[str, List]): Column or list of columns to be used"""
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(cols, list):
        stock_df = pd.read_csv(path)[cols]
    else:
        stock_df = pd.read_csv(path)
    try:
        stock_df = stock_df.set_index('Date').sort_index()
    except:
        stock_df=stock_df
    #Data transformations to be applied prior to be used with the synthesizer model
    data, processed_data, scaler = real_data_loading(stock_df.values, seq_len=seq_len)

    return data, processed_data, scaler
