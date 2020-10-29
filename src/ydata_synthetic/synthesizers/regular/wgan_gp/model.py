import os
from os import path
import numpy as np
from tqdm import tqdm
from functools import partial

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

class WGAN_GP(gan.Model):

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

        # Freeze the generator while discriminator training
        self.generator.trainable = False

        #Real event input
        real_event = Input(shape=self.data_dim)

        #Random noise object       
        z = Input(shape=(self.noise_dim,))
        #Generate new record using the generator from noise
        record = self.generator(z)

        # Discriminator determines validity of the real and fake events
        fake = self.critic(record)
        valid = self.critic(real_event)

        # Construct weighted average between real and the fake envents
        interpolated_img = RandomWeightedAverage()([real_event, record])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=validity_interpolated)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self._model_critic = Model(inputs=[real_event, z],
                                  outputs=[valid, fake, validity_interpolated],
                                  metrics=['accuracy'])

        self._model_critic.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # Computational graph for the Generator
        #Freeze the critic training while training the generator
        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = Input(shape=(self.noise_dim,))
        fake_record = self.generator(z_gen)
        valid = self.critic(fake_record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # For the WGAN model use the Wassertein loss
        self._model = Model(z_gen, valid)
        self._model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_pred, averaged_samples):
        """
        Computing gradient penalty based on the prediction for real, fake and weighted events
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]

        gradients_sqr = K.square(gradients)

        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))

        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)

        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


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
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = -np.zeros((self.batch_size, 1))

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
                    d_loss = self._model_critic.train_on_batch([batch_data, noise],
                                                                [valid, fake, dummy])

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = tf.random.normal((self.batch_size, self.noise_dim))
                # Train the generator (to have the critic label samples as valid)
                g_loss = self.model.train_on_batch(noise, valid)
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