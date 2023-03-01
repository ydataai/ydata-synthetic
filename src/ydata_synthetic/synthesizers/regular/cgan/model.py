"""
    CGAN architecture implementation file
"""
import os
from os import path
from typing import List, Optional, NamedTuple

from tqdm import trange

import numpy as np
from numpy import hstack
from pandas import DataFrame

from tensorflow import random
from tensorflow import data as tfdata
from tensorflow import dtypes
from keras import Model
from keras.layers import (Dense, Dropout, Input, concatenate)
from keras.optimizers import Adam

#Import ydata synthetic classes
from ....synthesizers import TrainParameters
from ....synthesizers.gan import ConditionalModel

class CGAN(ConditionalModel):
    "CGAN model for discrete conditions"

    __MODEL__='CGAN'

    def __init__(self, model_parameters):
        self._col_order = None
        super().__init__(model_parameters)

    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,),
                        label_shape=(self.label_dim),
                        dim=self.layers_dim, data_dim=self.data_dim,
                        activation_info = activation_info, tau = self.tau)

        self.discriminator = Discriminator(self.batch_size). \
            build_model(input_shape=(self.data_dim,),
                        label_shape=(self.label_dim,),
                        dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        d_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.noise_dim,))
        label = Input(shape=(1,))  # A label vector is expected
        record = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator([record, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self._model = Model([noise, label], validity)
        self._model.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    def _generate_noise(self):
        "Gaussian noise for the generator input."
        while True:
            yield random.uniform(shape=(self.noise_dim,))

    def get_batch_noise(self):
        "Create a batch iterator for the generator gaussian noise input."
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=dtypes.float32)
                                .batch(self.batch_size)
                                .repeat())

    def get_data_batch(self, data, batch_size, seed=0):
        "Produce real data batches from the passed data object."
        start_i = (batch_size * seed) % len(data)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(data)
        np.random.seed(shuffle_seed)
        data_ix = np.random.choice(data.shape[0], replace=False, size=len(data))  # wasteful to shuffle every time
        return data[data_ix[start_i: stop_i]]

    def fit(self,
            data: DataFrame,
            label_cols: List[str],
            train_arguments: TrainParameters,
            num_cols: List[str],
            cat_cols: List[str]):
        """
        Args:
            data: A pandas DataFrame with the data to be synthesized
            label_cols: The name of the column to be used as a label and condition for the training
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        data, label = self._prep_fit(data,label_cols,num_cols,cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        self.label_dim = len(label_cols)

        # Init the GAN model and optimizers
        self.define_gan(self.processor.col_transform_info)

        # Merging labels with processed data
        processed_data = hstack([processed_data, label])

        noise_batches = self.get_batch_noise()

        iterations = int(abs(processed_data.shape[0] / self.batch_size) + 1)
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in trange(train_arguments.epochs):
            for _ in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                batch_x = self.get_data_batch(processed_data, self.batch_size)  # Batches are retrieved with labels
                batch_x, label = batch_x[:, :-1], batch_x[:, -1]  # Separate labels from batch
                noise = next(noise_batches)

                # Generate a batch of new records
                gen_records = self.generator([noise, label], training=True)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([batch_x, label], valid)  # Separate labels
                d_loss_fake = self.discriminator.train_on_batch([gen_records, label], fake)  # Separate labels
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = next(noise_batches)
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self._model.train_on_batch([noise, label], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save model state and generated image samples
            if epoch % train_arguments.sample_interval == 0:
                self._run_checkpoint(train_arguments, epoch, label)

    def _run_checkpoint(self, train_arguments, epoch, label):
        "Run checkpoint. Store model state and generated samples."
        if path.exists('./cache') is False:
            os.mkdir('./cache')
        model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
        self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
        self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

# pylint: disable=R0903
class Generator():
    "Standard discrete conditional generator."
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None, tau: Optional[float] = None):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label_v = Input(shape=label_shape)
        x = concatenate([noise, label_v])
        x = Dense(dim, activation='relu')(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        #if activation_info:
        #    x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
        return Model(inputs=[noise, label_v], outputs=x)


# pylint: disable=R0903
class Discriminator():
    "Standard discrete conditional discriminator."
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape, dim):
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=label_shape, batch_size=self.batch_size)
        input_ = concatenate([events, label])
        x = Dense(dim * 4, activation='relu')(input_)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[events, label], outputs=x)
