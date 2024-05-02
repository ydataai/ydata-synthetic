import pandas as pd  # Importing pandas library for data manipulation
import numpy as np  # Importing numpy library for numerical operations
from sklearn import cluster  # Importing KMeans clustering algorithm from sklearn library

from ydata_synthetic.utils.cache import cache_file  # Importing cache_file function from ydata_synthetic.utils.cache
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters  # Importing ModelParameters and TrainParameters from ydata_synthetic.synthesizers
from ydata_synthetic.synthesizers.regular import RegularSynthesizer  # Importing RegularSynthesizer from ydata_synthetic.synthesizers.regular

# Read the original data and have it preprocessed
data_path = cache_file('creditcard.csv', 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv')
data = pd.read_csv(data_path, index_col=[0])  # Reading the csv file into a pandas DataFrame

# Data processing and analysis
num_cols = list(data.columns[ data.columns != 'Class' ])  # List of numerical column names
cat_cols = []  # List of categorical column names (empty in this case)

print('Dataset columns: {}'.format(num_cols))  # Printing the dataset columns
sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']
processed_data = data[ sorted_cols ].copy()  # Creating a copy of the data with sorted columns

# For the purpose of this example we will only synthesize the minority class
train_data = processed_data.loc[processed_data['Class'] == 1].copy()  # Selecting the minority class for training

# Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional WGANGP
print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))
algorithm = cluster.KMeans  # Initializing KMeans algorithm
args, kwds = (), {'n_clusters':2, 'random_state':0}  # Defining the arguments and keyword arguments for KMeans
labels = algorithm(*args, **kwds).fit_predict(train_data[ num_cols ])  # Fitting the KMeans algorithm on numerical data

fraud_w_classes = train_data.copy()  # Creating a copy of the training data
fraud_w_classes['Class'] = labels  # Adding the KMeans labels to the copy

# ----------------------------
#    GAN Training
# ----------------------------

# Define the Conditional WGANGP and training parameters
noise_dim = 32  # Dimension of the noise vector
dim = 128  # Dimension of the generator and discriminator
batch_size = 64  # Batch size for training
beta_1 = 0.5  # Beta1 hyperparameter for Adam optimizer
beta_2 = 0.9  # Beta2 hyperparameter for Adam optimizer

log_step = 100  # Logging step for printing training progress
epochs = 500 + 1  # Number of training epochs
learning_rate = 5e-4  # Learning rate for the optimizer
models_dir = '../cache'  # Directory for saving the trained models

# Test here the new inputs
gan_args = ModelParameters(batch_size=batch_size,  # Model parameters with batch size, learning rate, betas, noise dimension, and layer dimensions
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,  # Train parameters with epochs, cache prefix, sample interval, label dimension, and labels
                             cache_prefix='',
                             sample_interval=log_step,
                             label_dim=-1,
                             labels=(0,1))

# Create a bining for the 'Amount' column
fraud_w_classes
