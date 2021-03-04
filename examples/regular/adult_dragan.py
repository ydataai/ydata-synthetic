from ydata_synthetic.preprocessing.regular.adult import transformations
from ydata_synthetic.synthesizers.regular import DRAGAN

#Load and process the data
data, processed_data, preprocessor = transformations()

# WGAN_GP training
#Defininf the training parameters of WGAN_GP

noise_dim = 128
dim = 128
batch_size = 500

log_step = 100
epochs = 200+1
learning_rate = 1e-5
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, processed_data.shape[1], dim]
train_args = ['', epochs, log_step]

synthesizer = DRAGAN(gan_args, n_discriminator=3)
synthesizer.train(processed_data, train_args)

synth_data = synthesizer.sample(1000)
