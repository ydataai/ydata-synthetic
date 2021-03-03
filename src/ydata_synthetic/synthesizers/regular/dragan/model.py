import os
from os import path
import numpy as np
import tqdm

from ydata_synthetic.synthesizers import gan
from ydata_synthetic.synthesizers.loss import gradient_penalty

from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras import Model, initializers
from tensorflow.keras.optimizers import Adam

import os
from os import path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pandas as pd

class DRAGAN(gan):

    def __init__(self, model_parameters, n_critic, gradient_penalty_weight=10):
        # As recommended in DRAGAN paper - https://arxiv.org/abs/1705.07215
        self.n_critic = n_critic
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self):
        # define generator/discriminator
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.discriminator = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        self.g_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)
        self.critic_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)

    def gradient_penalty(self, real):
        gp = gradient_penalty(self.discriminator, real, mode='dragan')
        return gp

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
        gp = self.gradient_penalty(real)

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
        #tensor_data = pd.concat([x_train, y_train], axis=1)
        train_loader = tf.data.Dataset.from_tensor_slices(train) \
            .batch(batch_size).shuffle(self.BUFFER_SIZE)
        return train_loader

    @tf.function
    def train_step(self, real):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            noise = tf.random.normal((self.batch_size, self.noise_dim), dtype=tf.float64)

            fake = self.generator(noise)
            logits_fake = self.discriminator(fake)
            logits_fake = tf.cast(logits_fake, dtype=tf.float64)
            g_loss = -tf.reduce_mean(logits_fake)
            logits_real = self.discriminator(real)
            logits_real = tf.cast(logits_real, dtype=tf.float64)
            # gradient penalty
            gp = self.gradient_penalty(real)

            # getting the loss of the discriminator.
            d_loss = (tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real) + gp * self.gradient_penalty_weight)

            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

            self.d_optim.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            self.g_optim.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        print('gen_loss:{} and dis_loss:{}'.format(g_loss, d_loss))
        return g_loss, d_loss, fake

    # training loop
    def train(self, x_train, y_train):
        train_loader = self.get_data_batch(x_train, y_train, self.batch_size)
        for step in range(self.epochs):
            for data_batch in train_loader:
                if data_batch.shape[0] == self.batch_size:
                    g_loss, d_loss, generated_output = self.train_step(data_batch)
                    # total_loss = g_loss + d_loss
                    g_loss, d_loss = tf.reduce_sum(g_loss), tf.reduce_sum(d_loss)

                    template = '[{}/{}]  Dis_loss={} Gen_loss={}'
                    print(template.format(step, self.epochs, d_loss, g_loss))

            if path.exists('./cache') is False:
                os.mkdir('./cache')
            model_checkpoint_base_name = './cache/' + 'generated_seq_data' + '_{}_model_weights_step_{}'
            self.generator.save_weights(model_checkpoint_base_name.format('generator', step))
            self.discriminator.save_weights(model_checkpoint_base_name.format('critic', step))


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
