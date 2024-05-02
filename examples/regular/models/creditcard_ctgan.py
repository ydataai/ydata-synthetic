"""
    CTGAN architecture example file

This script demonstrates how to use the CTGAN architecture for synthetic data generation.
It uses the creditcard dataset available at https://datahub.io/machine-learning/creditcard/r/creditcard.csv
"""

import pandas as pd  # Importing pandas library for data manipulation
from sklearn import cluster  # Importing clustering algorithms from sklearn

# Reading the original data and preprocessing it
data_path = cache_file('creditcard.csv', 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv')
data = pd.read_csv(data_path, index_col=[0])

# Data processing and analysis
num_cols = list(data.columns[data.columns != 'Class'])  # List of numerical columns
cat_cols = []  # List of categorical columns

print('Dataset columns: {}'.format(num_cols))
sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19',
               'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15',
               'V9', 'V23', 'Class']
processed_data = data[sorted_cols].copy()  # Copying the data with sorted columns

# Selecting the minority class for synthesis
train_data = processed_data.loc[processed_data['Class'] == 1].copy()

# Creating a new class column using KMeans for conditional GAN
print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))
algorithm = cluster.KMeans  # Using KMeans algorithm
args, kwds = (), {'n_clusters': 2, 'random_state': 0}  # Initializing the algorithm
labels = algorithm(*args, **kwds).fit_predict(train_data[num_cols])  # Fitting the algorithm

fraud_w_classes = train_data.copy()  # Copying the data
fraud_w_classes['Class'] = labels  # Adding the new class column

#----------------------------
#    CTGAN Training
#----------------------------

batch_size = 500  # Setting batch size
epochs = 500 + 1  # Setting number of epochs
learning_rate = 2e-4  # Setting learning rate
beta_1 = 0.5  # Setting beta1 value
beta_2 = 0.9  # Setting beta2 value

ctgan_args = ModelParameters(batch_size=batch_size,  # Model parameters
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

train_args = TrainParameters(epochs=epochs)  # Training parameters

# Preprocessing the data for CTGAN
fraud_w_classes['Amount'] = pd.cut(fraud_w_classes['Amount'], 5).cat.codes  # Binning the 'Amount' column

# Initializing the CTGAN
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)

# Training the CTGAN
synth.fit(data=fraud_w_classes, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# Saving the synthesizer
synth.save('creditcard_ctgan_model.pkl')

# Loading the synthesizer
synthesizer = RegularSynthesizer.load('creditcard_ctgan_model.pkl')

# Sampling from the synthesizer
sample = synthesizer.sample(1000)  # Sampling 1000 records
print(sample)  # Printing the sampled records
