# Import necessary libraries and modules
from pmlb import fetch_data  # Import fetch_data function from pmlb library to load data

from ydata_synthetic.synthesizers.regular import RegularSynthesizer  # Import RegularSynthesizer class from ydata_synthetic library
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters  # Import ModelParameters and TrainParameters classes from ydata_synthetic library

# Load data from the UCI Machine Learning Repository and define the data processor parameters
data = fetch_data('adult')  # Fetch the 'adult' dataset from the UCI Machine Learning Repository
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']  # Define the numerical columns in the dataset
cat_cols = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'target']  # Define the categorical columns in the dataset

# Define the DRAGAN training parameters
noise_dim = 128  # Set the dimensionality of the noise vector
dim = 128  # Set the dimensionality of the layers
batch_size = 500  # Set the batch size for training

log_step = 100  # Set the logging frequency during training
epochs = 500 + 1  # Set the number of training epochs
learning_rate = 1e-5  # Set the learning rate for the optimizer
beta_1 = 0.5  # Set the first momentum term for the optimizer
beta_2 = 0.9  # Set the second momentum term for the optimizer
models_dir = '../cache'  # Set the directory path to save the trained models

# Initialize the model parameters and training arguments
gan_args = ModelParameters(batch_size=batch_size,  # Set the batch size
                           lr=learning_rate,  # Set the learning rate
                           betas=(beta_1, beta_2),  # Set the momentum terms
                           noise_dim=noise_dim,  # Set the dimensionality of the noise vector
                           layers_dim=dim)  # Set the dimensionality of the layers

train_args = TrainParameters(epochs=epochs,  # Set the number of training epochs
                             sample_interval=log_step)  # Set the logging frequency during training

# Initialize the synthetic data generator with the DRAGAN model
synth = RegularSynthesizer(modelname='dragan',  # Set the model name to 'dragan'
                           model_parameters=gan_args,  # Set the model parameters
                           n_discriminator=3)  # Set the number of discriminators

# Train the synthetic data generator on the loaded data
synth.fit(data=data,  # Set the input data
          train_arguments=train_args,  # Set the training arguments
          num_cols=num_cols,  # Set the numerical columns in the input data
          cat_cols=cat_cols)  # Set the categorical columns in the input data

# Save the trained synthetic data generator
synth.save('adult_dragan_model.pkl')

# Load the saved synthetic data generator
synthesizer = RegularSynthesizer.load('adult_dragan_model.pkl')

# Generate synthetic data samples from the loaded synthetic data generator
synthesizer.sample(1000)  # Set the number of synthetic data samples to generate
