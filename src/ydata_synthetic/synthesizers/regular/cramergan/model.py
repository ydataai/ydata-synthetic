import os
from os import path
import numpy as np
import tqdm

from ydata_synthetic.synthesizers import gan
from ydata_synthetic.synthesizers.loss import Mode, gradient_penalty

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

class CRAMERGAN(gan.Model):

    def __init__(self, model_parameters, gradient_penalty_weight=10):
        # As recommended in WGAN paper - https://arxiv.org/pdf/1705.10743.pdf
        # Cramer GAN- Introducing a anew distance as a solution to biased Wassertein Gradient
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        self.critic = Critic(self.discriminator)

        self.g_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.critic_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)

    def gradient_penalty(self, real, fake):
        gp = gradient_penalty(self.critic, real, fake, mode=Mode.CRAMER)
        return gp

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

            g_loss = tf.reduce_mean(
                self.critic(x, fake2) - self.critic(fake, fake2)
            )

            d_loss = -g_loss
            gp = self.gradient_penalty(x, [fake, fake2])

            critic_loss = d_loss + gp*self.gradient_penalty_weight

        # Get the gradients of the generator
        gen_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator
        self.g_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        d_gradient = d_tape.gradient(critic_loss, self.discriminator.trainable_variables)
        # Update the weights of the critic using the optimizer
        self.critic_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )

        return critic_loss, g_loss

    def d_lossfn(self, real, g_loss, noise):
        """
        passes through the network and computes the losses
        """
        # run noise through generator
        fake = self.generator(noise)
        # discriminate x and x_gen
        logits_real = self.critic(real)
        logits_fake = self.critic(fake)

        # gradient penalty
        gp = self.gradient_penalty(real, fake)
        # getting the loss of the discriminator.
        d_loss = (tf.reduce_mean(logits_fake)
                  - tf.reduce_mean(logits_real)
                  + gp * self.gradient_penalty_weight)
        return d_loss

    def g_lossfn(self, real):
        """
        :param real: Data batch we are analyzing
        :return: Loss of the generator
        """
        # generating noise from a uniform distribution
        noise = tf.random.normal([real.shape[0], self.noise_dim], dtype=tf.dtypes.float32)

        fake = self.generator(noise)
        logits_fake = self.critic(fake)
        g_loss = -tf.reduce_mean(logits_fake)
        return g_loss

    def get_data_batch(self, train, batch_size):
        buffer_size = len(train)
        train_loader = tf.data.Dataset.from_tensor_slices(train) \
            .batch(batch_size).shuffle(buffer_size)
        return train_loader

    def train_step(self, train_data):
        critic_loss, g_loss = self.update_gradients(train_data)
        return critic_loss, g_loss

    def train(self, data, train_arguments):
        [cache_prefix, iterations, sample_interval] = train_arguments
        train_loader = self.get_data_batch(data, self.batch_size)

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('..\wgan_gp_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for iteration in tqdm.trange(iterations):
                for batch_data in train_loader:
                    batch_data = tf.cast(batch_data, dtype=tf.float32)
                    critic_loss, g_loss = self.train_step(batch_data)

                    print(
                        "Iteration: {} | critic_loss: {} | gen_loss: {}".format(
                            iteration, critic_loss, g_loss
                        ))

                    if iteration % sample_interval == 0:
                        # Test here data generation step
                        # save model checkpoints
                        if path.exists('./cache') is False:
                            os.mkdir('./cache')
                        model_checkpoint_base_name = './cache/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                        self.generator.save_weights(model_checkpoint_base_name.format('generator', iteration))
                        self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', iteration))


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

class Discriminator(tf.keras.Model):
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

class Critic(object):
    def __init__(self, h):
        self.h = h

    def __call__(self, x, x_):
        return tf.norm(self.h(x, training=True) - self.h(x_, training=True), axis=1) - tf.norm(self.h(x, training=True), axis=1)
