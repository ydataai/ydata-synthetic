"""
    Get the stock data from Yahoo finance data
    Data from the period 01 January 2017 - 24 January 2021
"""
import os
import requests as req
import pandas as pd

from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

def transformations(seq_len: int):
    try:
        file_path = os.path.join(os.path.dirname(os.path.join('..', os.path.dirname(__file__))), 'data')
        stock_df = pd.read_csv(os.path.join(file_path, 'stock.csv'))
    except:
        stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1483228800&period2=1611446400&interval=1d&events=history&includeAdjustedClose=true'
        request = req.get(stock_url)
        url_content = request.content

        file_path = os.path.join(os.path.dirname(os.path.join('..', os.path.dirname(__file__))), 'data')
        stock_csv = open(os.path.join(file_path, 'stock.csv'), 'wb')
        stock_csv.write(url_content)
        # Reading the stock data
        stock_df = pd.read_csv('../data/stock.csv')

    try:
        stock_df = stock_df.set_index('Date').sort_index()
    except:
        stock_df=stock_df
    #Data transformations to be applied prior to be used with the synthesizer model
    processed_data = real_data_loading(stock_df.values, seq_len=seq_len)

    return processed_data
