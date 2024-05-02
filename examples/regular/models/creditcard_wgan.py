# Install ydata-synthetic library
# pip install ydata-synthetic

import os
import sys
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from urllib.error import HTTPError
from typing import List, Dict, Any

import sklearn.cluster as cluster
from ydata_synthetic.utils.cache import cache_file
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer

def download_file(file_url: str, file_path: str) -> None:
    """Download a file from a URL and save it to a local path.

    Args:
        file_url (str): The URL of the file to download.
        file_path (str): The local path to save the file to.

    Raises:
        HTTPError: If the file cannot be downloaded.
    """
    try:
        urlretrieve(file_url, file_path)
    except HTTPError as e:
        print(f"An error occurred while downloading the file: {e}")
        sys.exit(1)

def load_creditcard_data(file_path: str) -> pd.DataFrame:
    """Load the creditcard data from a local file.

    Args:
        file_path (str): The local path to the creditcard data file.

    Returns:
        pd.DataFrame: The creditcard data.
    """
    try:
        return pd.read_csv(file_path, index_col=[0])
    except FileNotFoundError as e:
        print(f"The creditcard data file cannot be found: {e}")
        sys.exit(1)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process the creditcard data.

    Args:
        data (pd.DataFrame): The creditcard data.

    Returns:
        pd.DataFrame: The processed creditcard data.
    """
    num_cols = list(data.columns[data.columns != 'Class'])
    cat_cols = ['Class']

    print("Dataset columns: {}".format(num_cols))

    sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']
    processed_data = data[sorted_cols].copy()

    return processed_data

def train_gan(synth: RegularSynthesizer, train_data: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> None:
    """Train the GAN model.

    Args:
        synth (RegularSynthesizer): The synthesizer to train.
        train_data (pd.DataFrame): The training data.
        num_cols (List[str]): The names of the numerical columns.
        cat_cols (List[str]): The names of the categorical columns.
    """
    # Define the GAN and training parameters
    noise_dim = 100  # Changed the way the noise dimension is defined
    dim = 128
    batch_size = 128

    log_step = 100
    epochs = 500 + 1
    learning_rate = 5e-4
    beta_1 = 0.5
    beta_2 = 0.9
    models_dir = '../cache'

    model_parameters = ModelParameters(batch_size=batch_size,
                                       lr=learning_rate,
                                       betas=(beta_1, beta_2),
                                       noise_dim=noise_dim,
                                       layers_dim=dim)

    train_args = TrainParameters(epochs=epochs,
                                 sample_interval=log_step)

    test_size = 492  # number of fraud cases

    # Training the CRAMERGAN model
    synth.fit(data=train_data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

    # Saving the synthesizer to later generate new events
    synth.save(path='creditcard_wgan_model.pkl')

if __name__ == "__main__":
    # Download the creditcard data file
    data_url = 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv'
    data_path = 'creditcard.csv'
    download_file(data_url, data_path)

    # Load the creditcard data
    data = load_creditcard_data(data_path)

    # Process the creditcard data
    processed_data = process_data(data)

    # For the purpose of this example we will only synthesize the minority class
    train_data = processed_data.loc[processed_data['Class'] == 1].copy()

    print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))

    # KMeans clustering
    algorithm = cluster.KMeans
    args, kwds = (), {'n_clusters': 2, 'random_state': 0}
    labels = algorithm(*args, **kwds).fit_predict(train_data[num_cols])

    # Add the clusters to the training data
    train_data['Class'] = labels

    # Train the GAN model
    synth = RegularSynthesizer(modelname='wgan')
    train_gan(synth, train_data, num_cols, cat_cols)

    # Load the trained synthesizer
    if os.path.exists('creditcard_wgan_model.pkl'):
        synth = RegularSynthesizer.load(path='creditcard_wgan_model.pkl')

        # Sample data from the trained synthesizer
        data_sample = synth.sample(100000)
    else:
        print("The trained synthesizer does not exist.")
