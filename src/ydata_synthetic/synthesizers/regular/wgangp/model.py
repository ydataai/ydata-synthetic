"""
    WGANGP architecture model implementation
"""

import os
from os import path
from typing import List, NamedTuple, Optional

from tqdm import trange
import numpy as np

import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

#Import ydata synthetic classes
from ....synthesizers import TrainParameters
from ....synthesizers.gan import BaseModel

class WGAN_GP(BaseModel):

    __MODEL__='WGAN_GP'

    def __init__(self, model_parameters, n_generator:int=1, n_critic:int=1, gradient_penalty_weight:int=10):
        # As recommended in WGAN paper - https://arxiv.org/abs/1701.07875
        # WGAN-GP - WGAN with Gradient Penalty
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,
                        activation_info=activation_info, tau = self.tau)

        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        c_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        return g_optimizer, c_optimizer

    def gradient_penalty(self, real, fake):
        epsilon = tf.random.uniform([real.shape[0], 1], minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    @tf.function
    def update_gradients(self, x, g_optimizer, c_optimizer):
        """
        Compute the gradients for both the Generator and the Critic
        :param x: real data event
        :return: generator gradients, critic gradients
        """
        for _ in range(self.n_critic):
            with tf.GradientTape() as d_tape:
                critic_loss = self.c_lossfn(x)
            # Get the gradients of the critic
            d_gradient = d_tape.gradient(critic_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the optimizer
            c_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )

        ##Add here the n_generator
        # Update the generator
        for _ in range(self.n_generator):
            with tf.GradientTape() as g_tape:
                gen_loss = self.g_lossfn(x)
            # Get the gradients of the generator
            gen_gradients = g_tape.gradient(gen_loss, self.generator.trainable_variables)
            # Update the weights of the generator
            g_optimizer.apply_gradients(
                zip(gen_gradients, self.generator.trainable_variables)
            )

        return critic_loss, gen_loss

    def c_lossfn(self, real):
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
        gp = self.gradient_penalty(real, fake)
        # getting the loss of the critic.
        c_loss = (tf.reduce_mean(logits_fake)
                  - tf.reduce_mean(logits_real)
                  + gp * self.gradient_penalty_weight)
        return c_loss

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

    def get_data_batch(self, train, batch_size, seed=0):
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly
        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(train.shape[0], replace=False, size=len(train))  # wasteful to shuffle every time
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        return train[train_ix[start_i: stop_i]]

    def train_step(self, train_data, optimizers):
        cri_loss, ge_loss = self.update_gradients(train_data, *optimizers)
        return cri_loss, ge_loss

    def fit(self, data, train_arguments: TrainParameters, num_cols: List[str], cat_cols: List[str]):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        super().fit(data, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        optimizers = self.define_gan(self.processor.col_transform_info)

        iterations = int(abs(data.shape[0]/self.batch_size)+1)

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('..\wgan_gp_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for epoch in trange(train_arguments.epochs):
                for _ in range(iterations):
                    batch_data = self.get_data_batch(processed_data, self.batch_size).astype(np.float32)
                    cri_loss, ge_loss = self.train_step(batch_data, optimizers)

                print(
                    "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                        epoch, cri_loss, ge_loss
                    ))

                if epoch % train_arguments.sample_interval == 0:
                    # Test here data generation step
                    # save model checkpoints
                    if path.exists('./cache') is False:
                        os.mkdir('./cache')
                    model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                    self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                    self.critic.save_weights(model_checkpoint_base_name.format('critic', epoch))


class Generator(tf.keras.Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None, tau: Optional[float] = None):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        #if activation_info:
        #    x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
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
