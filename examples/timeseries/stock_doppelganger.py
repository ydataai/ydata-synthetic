
# Importing necessary libraries
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import pandas as pd
from os import path

# Read the data
stock_data = processed_stock(path='../../data/stock_data.csv', seq_len=24)
stock_data = [pd.DataFrame(sd, columns = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]) for sd in stock_data]
stock_data = pd.concat(stock_data).reset_index(drop=True)

# Define model parameters
model_args = ModelParameters(batch_size=100,
                             lr=0.001,
                             betas=(0.5, 0.9),
                             latent_dim=3,
                             gp_lambda=10,
                             pac=10)

train_args = TrainParameters(epochs=500, sequence_length=24,
                             measurement_cols=["Open", "High", "Low", "Close", "Adj_Close", "Volume"])

# Training the DoppelGANger synthesizer
if path.exists('doppelganger_stock'):
    model_dop_gan = TimeSeriesSynthesizer.load('doppelganger_stock')
else:
    model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=model_args)
    model_dop_gan.fit(stock_data, train_args, num_cols=["Open", "High", "Low", "Close", "Adj_Close", "Volume"])

# Generating new synthetic samples
synth_data = model_dop_gan.sample(n_samples=500)
print(synth_data[0])
