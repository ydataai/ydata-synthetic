from ydata_synthetic.synthesizers.regular import CGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

import pandas as pd
import numpy as np
from sklearn import cluster

model = CGAN

#Read the original data and have it preprocessed
data = pd.read_csv('data/creditcard.csv', index_col=[0])

#List of columns different from the Class column
num_cols = list(data.columns[ data.columns != 'Class' ])
cat_cols = []  # Condition features are not preprocessed and therefore not listed here

print('Dataset columns: {}'.format(num_cols))
sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']
data = data[ sorted_cols ].copy()

#For the purpose of this example we will only synthesize the minority class
train_data = data.loc[ data['Class']==1 ].copy()

#Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional GAN
print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))
algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train_data[ num_cols ])

print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

fraud_w_classes = train_data.copy()
fraud_w_classes['Class'] = labels

#----------------------------
#    GAN Training
#----------------------------

#Define the Conditional GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 128
beta_1 = 0.5
beta_2 = 0.9

log_step = 100
epochs = 300 + 1
learning_rate = 5e-4
models_dir = './cache'

#Test here the new inputs
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             cache_prefix='',
                             sample_interval=log_step,
                             label_dim=-1,
                             labels=(0,1))

#Init the Conditional GAN providing the index of the label column as one of the arguments
synthesizer = model(model_parameters=gan_args, num_classes=2)

#Training the Conditional GAN
synthesizer.train(data=fraud_w_classes, label_col="Class", train_arguments=train_args,
                  num_cols=num_cols, cat_cols=cat_cols)

#Saving the synthesizer
synthesizer.save('cgan_synthtrained.pkl')

#Loading the synthesizer
synthesizer = model.load('cgan_synthtrained.pkl')

#Sampling from the synthesizer
cond_array = np.array([0])
# Synthesizer samples are returned in the original format (inverse_transform of internal processing already took place)
sample = synthesizer.sample(cond_array, 1000)
