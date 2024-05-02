"""
Get the stock data from Yahoo finance data
-----------------------------------------
This function retrieves the stock data from a CSV file that contains data downloaded from Yahoo finance.
The data is expected to be in a specific format with a 'Date' column.

Data from the period 01 January 2017 - 24 January 2021
-----------------------------------------------------
The function currently supports data from this specific time period.

Parameters
----------
path : str
    The file path of the CSV file containing the stock data.

seq_len: int
    The length of the sequence to be used for data transformations.

Returns
-------
processed_data : numpy array
    The transformed data ready to be used with the synthesizer model.

Raises
------
KeyError
    If the 'Date' column is not found in the CSV file.
"""
import pandas as pd

from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

def transformations(path, seq_len: int):
    stock_df = pd.read_csv(path)
    # Set the 'Date' column as the index and sort the data by date
    try:
        stock_df = stock_df.set_index('Date').sort_index()
    except KeyError:
        # Raise an error if the 'Date' column is not found
        raise KeyError("The 'Date' column was not found in the CSV file.")
    
    # Data transformations to be applied prior to be used with the synthesizer model
    processed_data = real_data_loading(stock_df.values, seq_len=seq_len)

    return processed_data
