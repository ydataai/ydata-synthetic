#Install ydata-synthetic lib
# pip install ydata-synthetic
import sklearn.cluster as cluster
import numpy as np
import pandas as pd

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import CRAMERGAN
from ydata_synthetic.preprocessing.regular.credit_fraud import transformations

model = CRAMERGAN

#Read the original data and have it preprocessed
data = pd.read_csv('data/creditcard.csv', index_col=[0])

#Data processing and analysis
data_cols = list(data.columns[ data.columns != 'Class' ])
label_cols = ['Class']

print('Dataset columns: {}'.format(data_cols))
sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']
processed_data = data[ sorted_cols ].copy()

#Before training the GAN do not forget to apply the required data transformations
#To ease here we've applied a PowerTransformation
_, data, _ = transformations(data)

#For the purpose of this example we will only synthesize the minority class
train_data = data.loc[ data['Class']==1 ].copy()

print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train_data[ data_cols ])

print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

fraud_w_classes = train_data.copy()
fraud_w_classes['Class'] = labels

# GAN training
#Define the GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
epochs = 300+1
learning_rate = 5e-4
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

train_sample = fraud_w_classes.copy().reset_index(drop=True)
train_sample = pd.get_dummies(train_sample, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train_sample.columns if 'Class' in i ]
data_cols = [ i for i in train_sample.columns if i not in label_cols ]
train_sample[ data_cols ] = train_sample[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train_sample[ data_cols ]

model_parameters = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           n_cols=train_sample.shape[1],
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

test_size = 492 # number of fraud cases
noise_dim = 32

#Training the CRAMERGAN model
synthesizer = model(model_parameters, gradient_penalty_weight=10)
synthesizer.train(train_sample, train_args)

#Saving the synthesizer to later generate new events
synthesizer.save(path='models/cramergan_creditcard.pkl')

#Loading the synthesizer
synth = model.load(path='models/cramergan_creditcard.pkl')

#Sampling the data
#Note that the data returned it is not inverse processed.
data_sample = synth.sample(100000)
