"""CWGANGP implementation."""
import os
from os import path
from typing import List, Optional, NamedTuple

from tqdm import trange

import numpy as np
from numpy import hstack
from pandas import DataFrame
from tensorflow import dtypes, GradientTape, reduce_sum, reduce_mean, sqrt, random
from keras import Model
from keras.layers import (Dense, Dropout, Input, concatenate, LeakyReLU)
from keras.optimizers import Adam

#Import ydata synthetic classes
from ....synthesizers import TrainParameters
from ....synthesizers.gan import ConditionalModel
from ....synthesizers.regular.wgangp.model import WGAN_GP

class CWGANGP(ConditionalModel, WGAN_GP):

    __MODEL__='CWGAN_GP'

    def __init__(self, model_parameters,
                 n_generator: Optional[int]=1,
                 n_critic: Optional[int]=1,
                 gradient_penalty_weight:int=10):
        """
        Adapts the WGAN_GP synthesizer implementation to be conditional.

        Several conditional WGAN implementations can be found online, here are a few:
            https://cameronfabbri.github.io/papers/conditionalWGAN.pdf
            https://www.sciencedirect.com/science/article/abs/pii/S0020025519309715
            https://arxiv.org/pdf/2008.09202.pdf
        """
        WGAN_GP.__init__(self, model_parameters,
                         n_generator=n_generator,
                         n_critic=n_critic,
                         gradient_penalty_weight=gradient_penalty_weight)

    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,),
                        label_shape=(self.label_dim, ),
                        dim=self.layers_dim,
                        data_dim=self.data_dim,
                        activation_info = activation_info,
                        tau = self.tau)

        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,),
                        label_shape=(self.label_dim,),
                        dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        c_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        return g_optimizer, c_optimizer

    def gradient_penalty(self, real, fake, label):
        epsilon = random.uniform([real.shape[0], 1], 0.0, 1.0, dtype=dtypes.float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic([x_hat, label])
        gradients = t.gradient(d_hat, x_hat)
        ddx = sqrt(reduce_sum(gradients ** 2))
        d_regularizer = reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    @staticmethod
    def get_data_batch(data, batch_size, seed=0):
        "Produce real data batches from the passed data object."
        start_i = (batch_size * seed) % len(data)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(data)
        np.random.seed(shuffle_seed)
        data_ix = np.random.choice(data.shape[0], replace=False, size=len(data))  # wasteful to shuffle every time
        return dtypes.cast(data[data_ix[start_i: stop_i]], dtype=dtypes.float32)

    def c_lossfn(self, real):
        "Forward pass on the critic and computes the loss."
        real, label = real
        # generating noise from a uniform distribution
        noise = random.uniform([real.shape[0], self.noise_dim], minval=0.999, maxval=1.0 , dtype=dtypes.float32)
        # run noise through generator
        fake = self.generator([noise, label])
        # discriminate x and x_gen
        logits_real = self.critic([real, label])
        logits_fake = self.critic([fake, label])

        # gradient penalty
        gp = self.gradient_penalty(real, fake, label)
        # getting the loss of the critic.
        c_loss = (reduce_mean(logits_fake)
                  - reduce_mean(logits_real)
                  + gp * self.gradient_penalty_weight)
        return c_loss

    def g_lossfn(self, real):
        """
        Forward pass on the generator and computes the loss.

        :param real: Data batch we are analyzing
        :return: Loss of the generator
        """
        real, label = real

        # generating noise from a uniform distribution
        noise = random.uniform([real.shape[0], self.noise_dim], minval=0.0, maxval=0.001 ,dtype=dtypes.float32)

        fake = self.generator([noise, label])
        logits_fake = self.critic([fake, label])
        g_loss = -reduce_mean(logits_fake)
        return g_loss

    def fit(self, data: DataFrame,
            label_cols: List[str],
            train_arguments: TrainParameters,
            num_cols: List[str],
            cat_cols: List[str]):
        """
        Train the synthesizer on a provided dataset based on a specified condition column.

        Args:
            data: A pandas DataFrame with the data to be synthesized
            label: The name of the column to be used as a label and condition for the training
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        data, label = self._prep_fit(data, label_cols, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        self.label_dim = len(label_cols)

        #Init the GAN model and optimizers
        optimizers = self.define_gan(self.processor.col_transform_info)

        # Merging labels with processed data
        processed_data = hstack([processed_data, label])

        iterations = int(abs(processed_data.shape[0] / self.batch_size) + 1)
        print(f'Number of iterations per epoch: {iterations}')

        for epoch in trange(train_arguments.epochs):
            for _ in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                batch_x = self.get_data_batch(processed_data, self.batch_size)  # Batches are retrieved with labels
                batch_x, label = batch_x[:, :-self.label_dim], batch_x[:, -self.label_dim:]  # Separate labels from batch

                cri_loss, ge_loss = self.train_step((batch_x, label), optimizers)

            print(
                "Epoch: {} | critic_loss: {} | gen_loss: {}".format(
                    epoch, cri_loss, ge_loss
                ))

            # If at save interval => save model state and generated image samples
            if epoch % train_arguments.sample_interval == 0:
                self._run_checkpoint(train_arguments, epoch)

    def _run_checkpoint(self, train_arguments, epoch):
        "Run checkpoint. Store model state and generated samples."
        if path.exists('./cache') is False:
            os.mkdir('./cache')
        model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
        self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
        self.critic.save_weights(model_checkpoint_base_name.format('critic', epoch))


act_leakyr = LeakyReLU(alpha=0.2)
# pylint: disable=R0903,D203
class Generator():

    "Standard discrete conditional generator."
    def __init__(self, batch_size):
        "Sets the properties of the generator."
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None, tau: Optional[float] = None):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label_v = Input(shape=label_shape)
        x = concatenate([noise, label_v])
        x = Dense(dim, activation=act_leakyr)(x)
        x = Dense(dim * 2, activation=act_leakyr)(x)
        x = Dense(dim * 4, activation=act_leakyr)(x)
        x = Dense(data_dim)(x)
        #if activation_info:
        #    x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
        return Model(inputs=[noise, label_v], outputs=x)

# pylint: disable=R0903,D203
class Critic():
    "Conditional Critic."
    def __init__(self, batch_size):
        "Sets the properties of the critic."
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape,dim):
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=label_shape, batch_size=self.batch_size)
        input_ = concatenate([events, label])
        x = Dense(dim * 4, activation=act_leakyr)(input_)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation=act_leakyr)(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation=act_leakyr)(x)
        x = Dense(1)(x)
        return Model(inputs=[events, label], outputs=x)
