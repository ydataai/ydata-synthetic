from ydata_synthetic.preprocessing.regular.adult import transformations
from ydata_synthetic.synthesizers.regular import DRAGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

#Load and process the data
data, processed_data, preprocessor = transformations()

# WGAN_GP training
#Defininf the training parameters of WGAN_GP

noise_dim = 128
dim = 128
batch_size = 500

log_step = 100
epochs = 300+1
learning_rate = 1e-5
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           n_cols=processed_data.shape[1],
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

synthesizer = DRAGAN(gan_args, n_discriminator=3)
synthesizer.train(processed_data, train_args)
synthesizer.save('adult_synth.pkl')
