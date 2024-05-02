import tensorflow as tf
import pandas as pd
import numpy as np
from pmlb import fetch_data
from ydata_synthetic.synthesizers.regular import RegularSynthesizer

# Load data and define the data processor parameters
data = fetch_data('adult')
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', 'target']

# Define the training parameters
noise_dim = 128
dim = 128
batch_size = 500
log_step = 100
epochs = 500 + 1
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([400, 450], [3e-3, 5e-4])
beta_1 = 0.5
beta_2 = 0.9
models_dir = '../cache'

# Define the model parameters
gan_args = RegularSynthesizer.ModelParameters(batch_size=batch_size,
                                              lr=learning_rate,
                                              betas=(beta_1, beta_2),
                                              noise_dim=noise_dim,
                                              layers_dim=dim)

# Define the training parameters
train_args = RegularSynthesizer.TrainParameters(epochs=epochs,
                                                sample_interval=log_step)

# Initialize the synthesizer
synth = RegularSynthesizer(modelname='wgangp', model_parameters=gan_args, n_critic=2)

# Train the synthesizer
synth.fit(data, train_args, num_cols, cat_cols)

# Save the trained synthesizer
synth.save('adult_wgangp_model.pkl')

# Load the trained synthesizer
synth = RegularSynthesizer.load('adult_wgangp_model.pkl')

# Sample data from the trained synthesizer
synth_data = synth.sample(1000)
