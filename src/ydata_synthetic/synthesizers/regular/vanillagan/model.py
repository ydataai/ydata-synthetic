"""
    Vanilla GAN architecture model implementation
"""
import os
from os import path
from typing import List, Optional, NamedTuple

import numpy as np
from tqdm import trange

import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras import Model
from keras.optimizers import Adam

#Import ydata synthetic classes
from ....synthesizers.gan import BaseModel
from ....synthesizers import TrainParameters
from ....utils.gumbel_softmax import GumbelSoftmaxActivation

class VanilllaGAN(BaseModel):

    __MODEL__='GAN'

    def __init__(self, model_parameters):
        super().__init__(model_parameters)

    def define_gan(self, activation_info: Optional[NamedTuple]):
        self.generator = Generator(self.batch_size).\
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,)

        self.discriminator = Discriminator(self.batch_size).\
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        d_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self._model = Model(z, validity)
        self._model.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
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
        self.define_gan(self.processor.col_transform_info)

        iterations = int(abs(data.shape[0]/self.batch_size)+1)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in trange(train_arguments.epochs):
            for _ in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                batch_data = self.get_data_batch(processed_data, self.batch_size)
                noise = tf.random.normal((self.batch_size, self.noise_dim))

                # Generate a batch of events
                gen_data = self.generator(noise, training=True)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(batch_data, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = tf.random.normal((self.batch_size, self.noise_dim))
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self._model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated events
            if epoch % train_arguments.sample_interval == 0:
                #Test here data generation step
                # save model checkpoints
                if path.exists('./cache') is False:
                    os.mkdir('./cache')
                model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

                #Here is generating the data
                z = tf.random.normal((432, self.noise_dim))
                gen_data = self.generator(z)
                print('generated_data')


class Generator(tf.keras.Model):
    def __init__(self, batch_size):
        self.batch_size=batch_size

    def build_model(self, input_shape, dim, data_dim):
        input= Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)

class Discriminator(tf.keras.Model):
    def __init__(self,batch_size):
        self.batch_size=batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)
