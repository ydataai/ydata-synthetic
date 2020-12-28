import os
from os import path
import numpy as np
from tqdm import tqdm

from ydata_synthetic.synthesizers import gan

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

#Auxiliary Keras backend class to calculate the Random Weighted average
#https://stackoverflow.com/questions/58133430/how-to-substitute-keras-layers-merge-merge-in-tensorflow-keras
class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class WGAN(gan.Model):

    def __init__(self, model_parameters, n_critic):
        # As recommended in WGAN paper - https://arxiv.org/abs/1701.07875
        # WGAN-GP - WGAN with Gradient Penalty
        self.n_critic = n_critic
        super().__init__(model_parameters)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def define_gan(self):
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.critic = Critic(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.critic_optimizer = Adam(self.lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Build and compile the critic
        self.critic.compile(loss=self.wasserstein_loss,
                                   optimizer=self.critic_optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)
        # The discriminator takes generated images as input and determines validity
        validity = self.critic(record)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        #For the WGAN model use the Wassertein loss
        self._model = Model(z, validity)
        self._model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

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

    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments

        #Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('.', 'summaries', 'train'))

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = -np.ones((self.batch_size, 1))

        with train_summary_writer.as_default():
            for epoch in tqdm.trange(epochs, desc='Epoch Iterations'):

                for _ in range(self.n_critic):
                    # ---------------------
                    #  Train the Critic
                    # ---------------------
                    batch_data = self.get_data_batch(data, self.batch_size)
                    noise = tf.random.normal((self.batch_size, self.noise_dim))

                    # Generate a batch of events
                    gen_data = self.generator(noise)

                    # Train the Critic
                    d_loss_real = self.critic.train_on_batch(batch_data, valid)
                    d_loss_fake = self.critic.train_on_batch(gen_data, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = tf.random.normal((self.batch_size, self.noise_dim))
                # Train the generator (to have the critic label samples as valid)
                g_loss = self._model.train_on_batch(noise, valid)
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

                #If at save interval => save generated events
                if epoch % sample_interval == 0:
                    # Test here data generation step
                    # save model checkpoints
                    if path.exists('./cache') is False:
                        os.mkdir('./cache')
                    model_checkpoint_base_name = './cache/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                    self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                    self.critic.save_weights(model_checkpoint_base_name.format('critic', epoch))


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