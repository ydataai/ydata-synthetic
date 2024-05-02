import os
import numpy as np
from typing import List, Optional, NamedTuple
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model, load_model

# Import ydata synthetic classes
from ....synthesizers.base import BaseGANModel
from ....synthesizers import TrainParameters

class VanillaGAN(BaseGANModel):
    __MODEL__='GAN'

    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)

    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        """Define the trainable model components.

        Args:
            activation_info (Optional[NamedTuple], optional): Defaults to None.

        Returns:
            (generator_optimizer, critic_optimizer): Generator and critic optimizers
        """
        self.generator = Generator(self.batch_size, self.noise_dim, self.data_dim, self.layers_dim).build_model()
        self.discriminator = Discriminator(self.batch_size, self.data_dim, self.layers_dim).build_model()

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
        """Get real data batches from the passed data object.

        Args:
            train: real data
            batch_size: batch size
            seed (int, optional):Defaults to 0.

        Returns:
            data batch
        """
        np.random.seed(seed)
        data_indices = np.random.permutation(len(train))
        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        return train.iloc[data_indices[start_i: stop_i]]

    def fit(self, data, train_arguments: TrainParameters, num_cols: List[str], cat_cols: List[str]):
        """Fit a synthesizer model to a given input dataset.

        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            train_arguments: GAN training arguments.
            num_cols (List[str]): List of columns of the data object to be handled as numerical
            cat_cols (List[str]): List of columns of the data object to be handled as categorical
        """
        super().fit(data, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        self.define_gan(self.processor.col_transform_info)

        iterations = int(abs(data.shape[0]/self.batch_size)+1)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(train_arguments.epochs):
            for _ in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                batch_data = self.get_data_batch(processed_data, self.batch_size)
                noise = tf.random.normal((self.batch_size, self.noise_dim))

                # Generate a batch of events
                gen_data = self.generator(noise, training=True)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(batch_data.values, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = tf.random.normal((self.batch_size, self.noise_dim))
                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self._model.train_on_batch(noise, valid)

            # Plot the progress
            print(f"Epoch {epoch + 1}/{train_arguments.epochs} \
                  [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] \
                  [G loss: {g_loss:.4f}]")

            # If at save interval => save generated events
            if (epoch + 1) % train_arguments.sample_interval == 0:
                # Test here data generation step
                # save model checkpoints
                if not os.path.exists('./cache'):
                    os.mkdir('./cache')
                model_checkpoint_base_name = f'./cache/{train_arguments.cache_prefix}_{self.__MODEL__}_model_weights_step_{epoch}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator'))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator'))

                # Generate and save synthetic data
                z = tf.random.normal((432, self.noise_dim))
                gen_data = self.generator(z)
                self.save_synthetic_data(gen_data)

@dataclass
class GeneratorParameters:
    batch_size: int
    noise_dim: int
    data_dim: int
    layers_dim: List[int]

@dataclass
class DiscriminatorParameters:
    batch_size: int
    data_dim: int
    layers_dim: List[int]

class Generator(tf.keras.Model):
    def __init__(self, batch_size, noise_dim, data_dim, layers_dim):
        super().__init__()
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.layers_dim = layers_dim

        self.dense1 = Dense(layers_dim[0], activation='relu')
        self.dense2 = Dense(layers_dim[1], activation='relu')
        self.dense3 = Dense(layers_dim[2], activation='relu')
        self.dense4 = Dense(data_dim)

    def build_model(self):
        """Create model components.

        Returns:
            Generator model
        """
        input = Input(shape=(self.noise_dim,), batch_size=self.batch_size)
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return Model(inputs=input, outputs=output)

class Discriminator(tf.keras.Model):
    def __init__(self, batch_size, data_dim, layers_dim):
        super().__init__()
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.layers_dim = layers_dim

        self.dense1 = Dense(layers_dim[0], activation='relu')
        self.dense2 = Dense(layers_dim[1], activation='relu')
        self.dense3 = Dense(layers_dim[2], activation='relu')
        self.dense4 = Dense(1, activation='sigmoid')

    def build_model(self):
        """Create model components.

        Returns:
            Discriminator model
        """
        input = Input(shape=(self.data_dim,), batch_size=self.batch_size)
        x = self.dense1(input)
        x = Dropout(0.1)(x)
        x = self.dense2(x)
        x = Dropout(0.1)(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return Model(inputs=input, outputs=output)
