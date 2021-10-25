import os
from os import path
import numpy as np
from tqdm import trange

from ydata_synthetic.synthesizers.gan import BaseModel
from ydata_synthetic.synthesizers import TrainParameters

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

class CRAMERGAN(BaseModel):

    __MODEL__='CRAMERGAN'

    def __init__(self, model_parameters, gradient_penalty_weight=10):
        # As recommended in WGAN paper - https://arxiv.org/pdf/1705.10743.pdf
        # Cramer GAN- Introducing a anew distance as a solution to biased Wassertein Gradient
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        self.g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.c_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # The generator takes noise as input and generates records
        z = Input(shape=(self.noise_dim,), batch_size=self.batch_size)
        fake = self.generator(z, training=True)
        logits = self.critic(fake, training=True)

        # Compile the critic
        self.critic.compile(loss=self.c_lossfn,
                            optimizer=self.c_optimizer,
                            metrics=['accuracy'])

        # Generator and critic model
        _model = Model(z, logits)
        _model.compile(loss=self.g_lossfn, optimizer=self.g_optimizer)

    def gradient_penalty(self, real, fake, fake2):
        epsilon = tf.random.uniform([real.shape[0], 1], 0.0, 1.0, dtype=tf.dtypes.float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with tf.GradientTape() as t:
            t.watch(x_hat)
            f_x_hat = self.f_crit(x_hat, fake2)
        gradients = t.gradient(f_x_hat, x_hat)
        c_dx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
        c_regularizer = (c_dx - 1.0) ** 2
        return c_regularizer

    def update_gradients(self, x):
        """
        Compute the gradients for both the Generator and the Critic
        :param x: real data event
        :return: generator gradients, critic gradients
        """
        # Update the gradients of critic for n_critic times (Training the critic)

        ##New generator gradient_tape
        noise= tf.random.normal([x.shape[0], self.noise_dim], dtype=tf.dtypes.float32)
        noise2= tf.random.normal([x.shape[0], self.noise_dim], dtype=tf.dtypes.float32)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake=self.generator(noise, training=True)
            fake2=self.generator(noise2, training=True)

            g_loss = self.g_lossfn(x, fake, fake2)

            c_loss = self.c_lossfn(x, fake, fake2)

        # Get the gradients of the generator
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        c_gradient = d_tape.gradient(c_loss, self.critic.trainable_variables)
        # Update the weights of the critic using the optimizer
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )

        return c_loss, g_loss

    def g_lossfn(self, real, fake, fake2):
        """
        :param real: A real sample
        :param fake: A fake sample
        :param fak2: A second fake sample
        :return: Loss of the generator
        """
        g_loss = tf.norm(self.critic(real, training=True) - self.critic(fake, training=True), axis=1) + \
                 tf.norm(self.critic(real, training=True) - self.critic(fake2, training=True), axis=1) - \
                 tf.norm(self.critic(fake, training=True) - self.critic(fake2, training=True), axis=1)
        return tf.reduce_mean(g_loss)

    def f_crit(self, real, fake):
        """
        Computes the critic distance function f between two samples
        :param real: A real sample
        :param fake: A fake sample
        :return: Loss of the critic
        """
        return tf.norm(self.critic(real, training=True) - self.critic(fake, training=True), axis=1) - tf.norm(self.critic(real, training=True), axis=1)

    def c_lossfn(self, real, fake, fake2):
        """
        :param real: A real sample
        :param fake: A fake sample
        :param fak2: A second fake sample
        :return: Loss of the critic
        """
        f_real = self.f_crit(real, fake2)
        f_fake = self.f_crit(fake, fake2)
        loss_surrogate = f_real - f_fake
        gp = self.gradient_penalty(real, fake, fake2)
        return tf.reduce_mean(- loss_surrogate + self.gradient_penalty_weight*gp)

    def get_data_batch(self, train, batch_size, seed=0):
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly
        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  # wasteful to shuffle every time
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        x = train.loc[train_ix[start_i: stop_i]].values
        return np.reshape(x, (batch_size, -1))

    def train_step(self, train_data):
        critic_loss, g_loss = self.update_gradients(train_data)
        return critic_loss, g_loss

    def train(self, data, train_arguments: TrainParameters):
        iterations = int(abs(data.shape[0] / self.batch_size) + 1)

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('..\cramergan_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for epoch in trange(train_arguments.epochs):
                for iteration in range(iterations):
                    batch_data = self.get_data_batch(data, self.batch_size)
                    c_loss, g_loss = self.train_step(batch_data)

                    if iteration % train_arguments.sample_interval == 0:
                        # Test here data generation step
                        # save model checkpoints
                        if path.exists('./cache') is False:
                            os.mkdir('./cache')
                        model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                        self.generator.save_weights(model_checkpoint_base_name.format('generator', iteration))
                        self.critic.save_weights(model_checkpoint_base_name.format('critic', iteration))

                print(
                    "Epoch: {} | critic_loss: {} | gen_loss: {}".format(
                        epoch, c_loss, g_loss
                    ))

        self.g_optimizer=self.g_optimizer.get_config()
        self.critic_optimizer=self.c_optimizer.get_config()


class Generator(tf.keras.Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)

class Critic(tf.keras.Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1)(x)
        return Model(inputs=input, outputs=x)
