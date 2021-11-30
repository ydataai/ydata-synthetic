from numpy import reshape

from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TSCWGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.postprocessing.regular.inverse_preprocesser import inverse_transform

model = TSCWGAN

#Define the GAN and training parameters
noise_dim = 32
dim = 128
seq_len = 48
cond_dim = 24
batch_size = 128

log_step = 100
epochs = 300+1
learning_rate = 5e-4
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'
critic_iter = 5

# Get transformed data stock - Univariate
data, processed_data, scaler = processed_stock(path='./data/stock_data.csv', seq_len=seq_len, cols = ['Open'])
data_sample = processed_data[0]

model_parameters = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           n_cols=seq_len,
                           layers_dim=dim,
                           condition = cond_dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step,
                             critic_iter=critic_iter)

#Training the TSCWGAN model
synthesizer = model(model_parameters, gradient_penalty_weight=10)
synthesizer.train(processed_data, train_args)

#Saving the synthesizer to later generate new events
synthesizer.save(path='./tscwgan_stock.pkl')

#Loading the synthesizer
synth = model.load(path='./tscwgan_stock.pkl')

#Sampling the data
#Note that the data returned is not inverse processed.
cond_index = 100  # Arbitrary sequence for conditioning
cond_array = reshape(processed_data[cond_index][:cond_dim], (1,-1))

data_sample = synth.sample(cond_array, 1000, 100)

# Inverting the scaling of the synthetic samples
data_sample = inverse_transform(data_sample, scaler)
