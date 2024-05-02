"""
    WGAN architecture model implementation
"""

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.models import Model
from tqdm import trange

# Import synthetic classes and base model
from ydata_synthesizers import TrainParameters
from ydata_synthesizers.base import BaseGANModel

# Import custom Keras layer for Random Weighted Average
from keras_random_weighted_average import RandomWeightedAverage

class WGAN(BaseGANModel):
    """WGAN model class"""
    def __init__(self, model_parameters, n_critic, clip_value=0.01):
        """
        Initialize WGAN model

        Args:
            model_parameters: model parameters
            n_critic: number of critic iterations
            clip_value: value for clipping weights
        """
        self.n_critic = n_critic
        self.clip_value = clip_value
        super().__init__(model_parameters)

    def wasserstein_loss(self, y_true, y_pred):
        """
        Calculate Wasserstein loss

        Args:
            y_true: ground truth
            y_pred: predictions

        Returns:
            Wasserstein loss
        """
        return K.mean(y_true * y_pred)

    def define_gan(self, activation_info=None):
        """
        Define the trainable model components

        Args:
            activation_info: activation information

        Returns:
            generator and critic optimizers
        """
        # Initialize generator and critic
        self.generator = Generator(self.batch_size).build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim, activation_info=activation_info, tau=self.tau)
        self.critic = Critic(self.batch_size).build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        # Compile critic
        optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        critic_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        self.critic.compile(loss=self.wasserstein_loss, optimizer=critic_optimizer, metrics=['accuracy'])

        # Create the combined model
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)
        validity = self.critic(record)

        self.critic.trainable = False

        self._model = Model(z, validity)
        self._model.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        """
        Get real data batches from the passed data object

        Args:
            train: real data
            batch_size: batch size
            seed: seed for randomization

        Returns:
            data batch
        """
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

    def fit(self, data, train_arguments: TrainParameters, num_cols, cat_cols):
        """
        Fit a synthesizer model to a given input dataset

        Args:
            data: a pandas DataFrame or a Numpy array with the data to be synthesized
            train_arguments: GAN training arguments
            num_cols: list of columns of the data object to be handled as numerical
            cat_cols: list of columns of the data object to be handled as categorical
        """
        super().fit(data, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        self.define_gan(self.processor.col_transform_info)

        # Create a summary file
        iterations = int(abs(data.shape[0]/self.batch_size)+1)
        train_summary_writer = tf.summary.create_file_writer(os.path.join('.', 'summaries', 'train'))

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = -np.ones((self.batch_size, 1))

        with train_summary_writer.as_default():
            for epoch in trange(train_arguments.epochs, desc='Epoch Iterations'):
                for _ in range(iterations):
                    for _ in range(self.n_critic):
                        # Train the Critic
                        batch_data = self.get_data_batch(processed_data, self.batch_size)
                        noise = tf.random.normal((self.batch_size, self.noise_dim))

                        # Generate a batch of events
                        gen_data = self.generator(noise)

                        # Train the critic
                        d_loss_real = self.critic.train_on_batch(batch_data, valid)
                        d_loss_fake = self.critic.train_on_batch(gen_data, fake)
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                        # Clip weights
                        for l in self.critic.layers:
                            weights = l.get_weights()
                            weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                            l.set_weights(weights)

                    # Train the Generator
                    noise = tf.random.normal((self.batch_size, self.noise_dim))
                    g_loss = self._model.train_on_batch(noise, valid)

                    # Print progress
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

                # Save generated events
                if epoch % train_arguments.sample_interval == 0:
                    # Save model checkpoints
                    if os.path.exists('./cache') is False:
                        os.mkdir('./cache')
                    model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                    self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                    self.critic.save_weights(model_checkpoint_base_name.format('critic', epoch))

class Generator(tf.keras.Model):
    """Generator model class"""
    def __init__(self, batch_size):
        """
        Initialize generator model

        Args:
            batch_size: batch size
        """
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim, activation_info=None, tau=None):
        """
        Create generator model components

        Args:
            input_shape: input dimensionality
            dim: hidden layers dimensions
            data_dim: Output dimensionality
            activation_info: activation information
            tau: Gumbel-Softmax non-negative temperature

        Returns:
            Generator model
        """
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        # if activation_info:
        #     x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
        return Model(inputs=input, outputs=x)

class Critic(tf.keras.Model):
    """Critic model class"""
    def __init__(self, batch_size):
        """
        Initialize critic model

        Args:
            batch_size: batch size
        """
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        """
        Create critic model components

        Args:
            input_shape: input dimensionality
            dim: hidden layers size

        Returns:
            Critic model
        """
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1)(x)
        return Model(inputs=input, outputs=x)
