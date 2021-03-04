import os
from os import path

import numpy as np
import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model, initializers

from ydata_synthetic.synthesizers import gan
from ydata_synthetic.synthesizers.loss import gradient_penalty

import pandas as pd

class DRAGAN(gan.Model):

    def __init__(self, model_parameters, n_discriminator, gradient_penalty_weight=10):
        # As recommended in DRAGAN paper - https://arxiv.org/abs/1705.07215
        self.n_discriminator = n_discriminator
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self):
        # define generator/discriminator
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        self.g_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)
        self.d_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)

    def gradient_penalty(self, real, fake):
        gp = gradient_penalty(self.discriminator, real, fake, mode='dragan')
        return gp

    def update_gradients(self, x):
        """
        Compute the gradients for both the Generator and the Discriminator
        :param x: real data event
        :return: generator gradients, discriminator gradients
        """
        # Update the gradients of critic for n_critic times (Training the critic)
        for _ in range(self.n_discriminator):
            with tf.GradientTape() as d_tape:
                d_loss = self.d_lossfn(x)
            # Get the gradients of the critic
            d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the critic using the optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Update the generator
        with tf.GradientTape() as g_tape:
            gen_loss = self.g_lossfn(x)

        # Get the gradients of the generator
        gen_gradients = g_tape.gradient(gen_loss, self.generator.trainable_variables)

        # Update the weights of the generator
        self.g_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return d_loss, gen_loss

    def d_lossfn(self, real):
        """
        Calculates the critic losses
        """
        noise = tf.random.normal((self.batch_size, self.noise_dim), dtype=tf.dtypes.float64)
        # run noise through generator
        fake = self.generator(noise)
        # discriminate x and x_gen
        logits_real = self.discriminator(real, training=True)
        logits_fake = self.discriminator(fake, training=True)

        # gradient penalty
        gp = self.gradient_penalty(real, fake)

        # getting the loss of the discriminator.
        d_loss = (tf.reduce_mean(logits_fake)
                  - tf.reduce_mean(logits_real)
                  + gp * self.gradient_penalty_weight)
        return d_loss

    # generator loss
    def g_lossfn(self, real):
        """
        Calculates the Generator losses
        :param real: Data batch we are analyzing
        :return: Loss of the generator
        """
        # generating noise from a uniform distribution
        noise = tf.random.normal((real.shape[0], self.noise_dim), dtype=tf.float64)

        fake = self.generator(noise, training=True)
        logits_fake = self.discriminator(fake, training=True)
        g_loss = -tf.reduce_mean(logits_fake)
        return g_loss

    def get_data_batch(self, train, batch_size):
        buffer_size = len(train)
        #tensor_data = pd.concat([x_train, y_train], axis=1)
        train_loader = tf.data.Dataset.from_tensor_slices(train) \
            .batch(batch_size).shuffle(buffer_size)
        return train_loader

    def train_step(self, train_data):
        d_loss, g_loss = self.update_gradients(train_data)
        return d_loss, g_loss

    def train(self, data, train_arguments):
        [cache_prefix, iterations, sample_interval] = train_arguments
        train_loader = self.get_data_batch(data, self.batch_size)

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('..\dragan_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for iteration in tqdm.trange(iterations):
                for batch_data in train_loader:
                    batch_data = tf.cast(batch_data, dtype=tf.float32)
                    d_loss, g_loss = self.train_step(batch_data)

                    print(
                        "Iteration: {} | disc_loss: {} | gen_loss: {}".format(
                            iteration, d_loss, g_loss
                        ))

                    if iteration % sample_interval == 0:
                        # Test here data generation step
                        # save model checkpoints
                        if path.exists('./cache') is False:
                            os.mkdir('./cache')
                        model_checkpoint_base_name = './cache/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                        self.generator.save_weights(model_checkpoint_base_name.format('generator', iteration))
                        self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', iteration))


class Discriminator(Model):
  def __init__(self, batch_size):
    self.batch_size = batch_size

  def build_model(self, input_shape, dim):
    input = Input(shape=input_shape, batch_size=self.batch_size)
    x = Dense(dim * 4, kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.5), activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Dense(dim * 2, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(dim, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input, outputs=x)

class Generator(Model):
  def __init__(self, batch_size):
    self.batch_size = batch_size

  def build_model(self, input_shape, dim, data_dim):
    input = Input(shape=input_shape, batch_size = self.batch_size)
    x = Dense(dim, kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.5), activation='relu')(input)
    x = Dense(dim * 2, activation='relu')(x)
    x = Dense(dim * 4, activation='relu')(x)
    x = Dense(data_dim)(x)
    return Model(inputs=input, outputs=x)

