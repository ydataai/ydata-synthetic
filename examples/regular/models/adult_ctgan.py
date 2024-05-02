import pandas as pd
from pmlb import fetch_data

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Load data and define the data processor parameters
data = fetch_data('adult')
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', 'target']

# Defining the training parameters
batch_size = 500
epochs = 500 + 1
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

# Define the model parameters
ctgan_args = ModelParameters(batch_size=batch_size,
                             learning_rate=learning_rate,
                             betas=(beta_1, beta_2))

# Define the training arguments
train_args = TrainParameters(epochs=epochs)

# Initialize the synthesizer
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)

# Train the synthesizer
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# Save the trained synthesizer
synth.save('adult_ctgan_model.pkl')

# Load the trained synthesizer
synth = RegularSynthesizer.load('adult_ctgan_model.pkl')

# Sample data from the trained synthesizer
synth_data = synth.sample(1000)
print(synth_data)
