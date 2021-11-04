"""
Conditional time-series Wasserstein GAN.
Based on: https://www.naun.org/main/NAUN/neural/2020/a082016-004(2020).pdf
And on: https://github.com/CasperHogenboom/WGAN_financial_time-series
"""
from tqdm import trange
from numpy.random import normal
from pandas import DataFrame

from tensorflow import concat, float32, convert_to_tensor, reshape, GradientTape, reduce_mean, make_ndarray, make_tensor_proto, tile, expand_dims
from tensorflow import data as tfdata
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, Dense, LeakyReLU, Flatten, Add


from ydata_synthetic.synthesizers.gan import BaseModel
from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.loss import Mode, gradient_penalty

class TSCWGAN(BaseModel):

    __MODEL__='TSCWGAN'

    def __init__(self, model_parameters, gradient_penalty_weight=10):
        """Create a base TSCWGAN."""
        self.gradient_penalty_weight = gradient_penalty_weight
        self.cond_dim = model_parameters.condition
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim + self.cond_dim, 1), dim=self.layers_dim, data_dim=self.data_dim)
        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim + self.cond_dim, 1), dim=self.layers_dim)

        self.g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.c_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # The generator takes noise as input and generates records
        noise = Input(shape=self.noise_dim, batch_size=self.batch_size)
        cond = Input(shape=self.cond_dim, batch_size=self.batch_size)
        gen = concat([cond, noise], axis=1)
        gen = self.generator(gen)
        score = concat([cond, gen], axis=1)
        score = self.critic(score)

    def train(self, data, train_arguments: TrainParameters):
        real_batches = self.get_batch_data(data)
        noise_batches = self.get_batch_noise()

        for epoch in trange(train_arguments.epochs):
            for i in range(train_arguments.critic_iter):
                real_batch = next(real_batches)
                noise_batch = next(noise_batches)[:len(real_batch)]  # Truncate the noise tensor in the shape of the real data tensor

                c_loss = self.update_critic(real_batch, noise_batch)

            real_batch = next(real_batches)
            noise_batch = next(noise_batches)[:len(real_batch)]

            g_loss = self.update_generator(real_batch, noise_batch)

            print(
                "Epoch: {} | critic_loss: {} | gen_loss: {}".format(
                    epoch, c_loss, g_loss
                ))

        self.g_optimizer = self.g_optimizer.get_config()
        self.c_optimizer = self.c_optimizer.get_config()

    def update_critic(self, real_batch, noise_batch):
        with GradientTape() as c_tape:
            fake_batch, cond_batch = self._make_fake_batch(real_batch, noise_batch)

            # Real and fake records with conditions
            real_batch_ = concat([cond_batch, real_batch], axis=1)
            fake_batch_ = concat([cond_batch, fake_batch], axis=1)

            c_loss = self.c_lossfn(real_batch_, fake_batch_)

        c_gradient = c_tape.gradient(c_loss, self.critic.trainable_variables)

        # Update the weights of the critic using the optimizer
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )
        return c_loss

    def update_generator(self, real_batch, noise_batch):
        with GradientTape() as g_tape:
            fake_batch, cond_batch = self._make_fake_batch(real_batch, noise_batch)

            # Fake records with conditions
            fake_batch_ = concat([cond_batch, fake_batch], axis=1)

            g_loss = self.g_lossfn(fake_batch_)

        g_gradient = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the optimizer
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )
        return g_loss

    def c_lossfn(self, real_batch_, fake_batch_):
        score_fake = self.critic(fake_batch_)
        score_real = self.critic(real_batch_)
        grad_penalty = self.gradient_penalty(real_batch_, fake_batch_)
        c_loss = reduce_mean(score_fake) - reduce_mean(score_real) + grad_penalty
        return c_loss

    def g_lossfn(self, fake_batch_):
        score_fake = self.critic(fake_batch_)
        g_loss = - reduce_mean(score_fake)
        return g_loss

    def _make_fake_batch(self, real_batch, noise_batch):
        """Generate a batch of fake records and return it with the batch of used conditions.
        Conditions are the first elements of records in the real batch."""
        cond_batch = real_batch[:, :self.cond_dim]
        gen_input = concat([cond_batch, noise_batch], axis=1)
        return self.generator(gen_input, training=True), cond_batch

    def gradient_penalty(self, real, fake):
        gp = gradient_penalty(self.critic, real, fake, mode=Mode.DRAGAN)
        return gp

    def _generate_noise(self):
        "Gaussian noise for the generator input."
        while True:
            yield normal(size=self.noise_dim)

    def get_batch_noise(self):
        "Create a batch iterator for the generator gaussian noise input."
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size)
                                .repeat())

    def get_batch_data(self, data, n_windows= None):
        if not n_windows:
            n_windows = len(data)
        data = reshape(convert_to_tensor(data, dtype=float32), shape=(-1, self.data_dim))
        return iter(tfdata.Dataset.from_tensor_slices(data)
                                .shuffle(buffer_size=n_windows)
                                .batch(self.batch_size).repeat())

    def sample(self, cond_array, n_samples):
        """Provided that cond_array is passed, produce n_samples for each condition vector in cond_array."""
        assert len(cond_array.shape) == 2, "Condition array should have 2 dimensions."
        assert cond_array.shape[1] == self.cond_dim, \
            f"Each sequence in the condition array should have a {self.cond_dim} length."
        n_conds = cond_array.shape[0]
        steps = n_samples // self.batch_size + 1
        data = []
        z_dist = self.get_batch_noise()
        for seq in range(n_conds):
            cond_seq = expand_dims(convert_to_tensor(cond_array.iloc[seq], float32), axis=0)
            cond_seq = tile(cond_seq, multiples=[self.batch_size, 1])
            for step in trange(steps, desc=f'Synthetic data generation - Condition {seq+1}/{n_conds}'):
                gen_input = concat([cond_seq, next(z_dist)], axis=1)
                records = make_ndarray(make_tensor_proto(self.generator(gen_input, training=False)))
                data.append(records)
        return DataFrame(concat(data, axis=0))


class Generator(Model):
    """Conditional generator with skip connections."""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim):
        # Define input - Expected input shape is (batch_size, seq_len, noise_dim). noise_dim = Z + cond
        noise_input = Input(shape = input_shape, batch_size = self.batch_size)

        # Compose model
        proc_input = Sequential(layers=[
            Conv1D(filters=dim, kernel_size=1, input_shape = input_shape),
            LeakyReLU(),
            Conv1D(dim, kernel_size=5, dilation_rate=2, padding="same"),
            LeakyReLU()
        ], name='input_to_latent')(noise_input)

        block_cnn = Sequential(layers=[
            Conv1D(filters=dim, kernel_size=3, dilation_rate=2, padding="same"),
            LeakyReLU()
        ], name='block_cnn')
        for i in range(3):
            if i == 0:
                cnn_block_i = proc_input
                cnn_block_o = block_cnn(proc_input)
            else:
                cnn_block_o = block_cnn(cnn_block_i)
            cnn_block_i = Add()([cnn_block_i, cnn_block_o])

        shift = Sequential(layers=[
            Conv1D(filters=10, kernel_size=3, dilation_rate=2, padding="same"),
            LeakyReLU(),
            Flatten(),
            Dense(dim*2),
            LeakyReLU()
        ], name='block_shift')(cnn_block_i)

        block = Sequential(layers=[
            Dense(dim*2),
            LeakyReLU()
        ], name='block')
        for i in range(3):
            if i == 0:
                block_i = shift
                block_o = block(shift)
            else:
                block_o = block(block_i)
            block_i = Add()([block_i, block_o])

        output = Dense(data_dim, name='latent_to_ouput')(block_i)
        return Model(inputs = noise_input, outputs = output, name='SkipConnectionGenerator')

class Critic(Model):
    """Conditional Wasserstein Critic with skip connections."""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        # Define input - Expected input shape is X + condition
        record_input = Input(shape = input_shape, batch_size = self.batch_size)

        # Compose model
        proc_record = Sequential(layers=[
            Dense(dim*2,),
            LeakyReLU()
        ], name='ts_to_latent')(record_input)

        block = Sequential(layers=[
            Dense(dim*2),
            LeakyReLU()
        ], name='block')
        for i in range(7):
            if i == 0:
                block_i = proc_record
                block_o = block(proc_record)
            else:
                block_o = block(block_i)
            block_i = Add()([block_i, block_o])

        output = Dense(1, name = 'latent_to_score')(block_i)
        return Model(inputs=record_input, outputs=output, name='SkipConnectionCritic')
