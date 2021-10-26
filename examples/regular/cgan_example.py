from ydata_synthetic.synthesizers.regular import CGAN
from ydata_synthetic.preprocessing.regular.credit_fraud import transformations
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

import pandas as pd
import numpy as np
from sklearn import cluster

#Read the original data and have it preprocessed
data = pd.read_csv('data/creditcard.csv', index_col=[0])

#List of columns different from the Class column
data_cols = list(data.columns[ data.columns != 'Class' ])
label_cols = ['Class']

print('Dataset columns: {}'.format(data_cols))
sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']

#Before training the GAN do not forget to apply the required data transformations
#To ease here we've applied a PowerTransformation
data = data[ sorted_cols ].copy()
data, proc_data, _ = transformations(data)

#For the purpose of this example we will only synthesize the minority class
train_data = proc_data.loc[ data['Class']==1 ].copy()

#Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional GAN
print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))
algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train_data[ data_cols ])

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

train_sample = fraud_w_classes.copy().reset_index(drop=True)
train_sample = pd.get_dummies(train_sample, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [list(train_sample.columns).index(i) for i in train_sample.columns if 'Class' in i ]
data_cols = [ i for i in train_sample.columns if i not in label_cols ]
train_sample[ data_cols ] = train_sample[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train_sample[ data_cols ]

#Test here the new inputs
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           n_cols=train_sample.shape[1] - len(label_cols),  # Don't count the label columns here
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             cache_prefix='',
                             sample_interval=log_step,
                             label_dim=-1,
                             labels=(0,1))

#Init the Conditional GAN providing the index of the label column as one of the arguments
synthesizer = CGAN(model_parameters=gan_args, num_classes=2)

#Training the Conditional GAN
synthesizer.train(data=train_sample, label="Class",train_arguments=train_args)

#Saving the synthesizer
synthesizer.save('cgan_synthtrained.pkl')
