"""
    TimeGAN architecture example file
"""

# Importing necessary libraries
from os import path
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define model parameters
gan_args = ModelParameters(batch_size=128,
                           lr=5e-4,
                           noise_dim=32,
                           layers_dim=128,
                           latent_dim=24,
                           gamma=1)

train_args = TrainParameters(epochs=50000,
                             sequence_length=24,
                             number_sequences=6)

# Read the data
stock_data = pd.read_csv("../../data/stock_data.csv")
cols = list(stock_data.columns)

# Training the TimeGAN synthesizer
if path.exists('synthesizer_stock.pkl'):
    synth = TimeSeriesSynthesizer.load('synthesizer_stock.pkl')
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(stock_data, train_args, num_cols=cols)
    synth.save('synthesizer_stock.pkl')

# Generating new synthetic samples
stock_data_blocks = processed_stock(path='../../data/stock_data.csv', seq_len=24)
synth_data = synth.sample(n_samples=len(stock_data_blocks))
print(synth_data[0].shape)

# Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,25))
obs = np.random.randint(len(stock_data_blocks))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': stock_data_blocks[obs][:, j],
                   'Synthetic': synth_data[obs].iloc[:, j]})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
