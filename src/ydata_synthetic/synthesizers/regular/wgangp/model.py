import os
from os import path
import numpy as np
import tqdm
from functools import partial

from ydata_synthetic.synthesizers import gan

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

class WGAN_GP(gan.Model):
    
    GRADIENT_PENALTY_WEIGHT = 10
    
    def __init__(self, model_parameters, n_critic):
        # As recommended in WGAN paper - https://arxiv.org/abs/1701.07875
        # WGAN-GP - WGAN with Gradient Penalty
        self.n_critic = n_critic
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        self.g_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.critic_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty(self, real, fake):
        epsilon = tf.random.uniform([real.shape[0], 1], 0.0, 1.0, dtype=tf.dtypes.float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def compute_gradients(self, x):
        """
        Compute the gradients for both the Generator and the Critic
        :param x: real data event
        :return: generator gradients, critic gradients
        """
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            d_loss, g_loss = self.compute_loss(x)

        gen_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        disc_gradients = d_tape.gradient(d_loss, self.critic.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, ggradients, dgradients):
        self.g_optimizer.apply_gradients(
            zip(ggradients, self.generator.trainable_variables)
        )
        self.critic_optimizer.apply_gradients(
            zip(dgradients, self.critic.trainable_variables)
        )

    def compute_loss(self, real):
        """ 
        passes through the network and computes the losses
        """
        # generating noise from a uniform distribution

        noise = tf.random.normal([real.shape[0], self.noise_dim], dtype=tf.dtypes.float32)

        # run noise through generator
        fake = self.generator(noise)
        # discriminate x and x_gen
        logits_real = self.critic(real)
        logits_fake = self.critic(fake)

        # gradient penalty
        d_regularizer = self.gradient_penalty(real, fake)
        ### losses
        d_loss = (
                tf.reduce_mean(logits_real)
                - tf.reduce_mean(logits_fake)
                + d_regularizer * self.GRADIENT_PENALTY_WEIGHT
        )

        # losses of fake with label "1"
        g_loss = tf.reduce_mean(logits_fake)
        return d_loss, g_loss

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

    @tf.function
    def train_step(self, train_data):
        g_gradients, d_gradients = self.compute_gradients(train_data)
        self.apply_gradients(g_gradients, d_gradients)

    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('../wgan_gp_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for epoch in tqdm.trange(epochs):
                batch_data = self.get_data_batch(data, self.batch_size).astype(np.float32)
                self.train_step(batch_data)
                loss = self.compute_loss(batch_data)

                print(
                    "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                        epoch, loss[0], loss[1]
                    ))

                if epoch % sample_interval == 0:
                    # Test here data generation step
                    # save model checkpoints
                    if path.exists('./cache') is False:
                        os.mkdir('./cache')
                    model_checkpoint_base_name = './cache/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                    self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                    self.critic.save_weights(model_checkpoint_base_name.format('critic', epoch))

    def load(self, path):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator

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