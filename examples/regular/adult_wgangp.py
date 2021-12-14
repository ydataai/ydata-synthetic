from pmlb import fetch_data

from ydata_synthetic.synthesizers.regular import WGAN_GP
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

model = WGAN_GP

#Load data and define the data processor parameters
data = fetch_data('adult')
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', 'target']

#Defining the training parameters

noise_dim = 128
dim = 128
batch_size = 500

log_step = 100
epochs = 300+1
learning_rate = [5e-4, 3e-3]
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

synthesizer = model(gan_args, n_critic=2)
synthesizer.train(data, train_args, num_cols, cat_cols)

synthesizer.save('test.pkl')

synthesizer = model.load('test.pkl')
synth_data = synthesizer.sample(1000)
