"""CGAN implementation"""
import os
from os import path
from typing import List, Union

import numpy as np
from numpy import array, vstack, zeros, empty
from numpy.random import normal
from pandas import DataFrame
from tensorflow import convert_to_tensor, dtypes, expand_dims, tile, concat, constant
from tensorflow import data as tfdata
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.gan import BaseModel


class CGAN(BaseModel):

    __MODEL__='CGAN'

    def __init__(self, model_parameters, num_classes):
        self.num_classes = num_classes
        super().__init__(model_parameters)
        self._mask = None

    def define_gan(self):
        self.generator = Generator(self.batch_size, self.num_classes). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size, self.num_classes). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        d_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        label = Input(shape=(self.num_classes,))
        record = self.generator([z, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator([record, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self._model = Model([z, label], validity)
        self._model.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    def _generate_noise(self):
        "Gaussian noise for the generator input."
        while True:
            yield normal(size=self.noise_dim)

    def get_batch_noise(self):
        "Create a batch iterator for the generator gaussian noise input."
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=dtypes.float32)
                                .batch(self.batch_size)
                                .repeat())

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
        # np.random.seed(seed)
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(train.shape[0], replace=False, size=len(train))  # wasteful to shuffle every time
        return train[train_ix[start_i: stop_i]]

    def __build_mask(self, label_col: str):
        num_vars = list(self.processor._num_pipeline.get_feature_names_out())
        cat_vars = list(self.processor._cat_pipeline.get_feature_names_out())
        mask = zeros(len(num_vars) + len(cat_vars), bool)
        mask[[i for i, col in enumerate(num_vars + cat_vars) if ''.join(col.split('_')[:-1]) == label_col]] = True
        self._mask = constant(mask)

    def train(self, data: Union[DataFrame, array], label_col: str, train_arguments: TrainParameters, num_cols: List[str],
              cat_cols: List[str], preprocess: bool = True):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            label: The name of the column to be used as a label and condition for the training
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
            preprocess: If True preprocess the data before using in train session
        """
        super().train(data, num_cols, cat_cols, preprocess)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1] - len(self.processor.cat_pipeline.get_feature_names_out())
        self.define_gan()

        self.__build_mask(label_col)

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
                batch_x = self.get_data_batch(processed_data, self.batch_size)
                batch_x, label = batch_x[:, ~self._mask ], batch_x[:, self._mask ]
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

            # If at save interval => save generated image samples
            if epoch % train_arguments.sample_interval == 0:
                # Test here data generation step
                # save model checkpoints
                if path.exists('./cache') is False:
                    os.mkdir('./cache')
                model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

                #Here is generating synthetic data from an arbitrary condition
                gen_data = self.sample(label[0], 1000)

    def sample(self, cond_array, n_samples, inverse_transform: bool = True):
        """Produce n_samples by conditioning the generator with cond_array."""
        assert cond_array.shape[0] == self.num_classes, \
            f"The condition sequence should have a {self.num_classes} length."
        steps = n_samples // self.batch_size + 1
        data = []
        z_dist = self.get_batch_noise()
        cond_seq = expand_dims(convert_to_tensor(cond_array, dtypes.float32), axis=0)
        cond_seq = tile(cond_seq, multiples=[self.batch_size, 1])
        for step in trange(steps, desc=f'Synthetic data generation'):
            records = empty(shape=(self.batch_size, self._mask.shape[0]))
            records[:, ~self._mask] = self.generator([next(z_dist), cond_seq], training=False)
            records[:, self._mask] = cond_seq
            data.append(records)
        data = array(vstack(data))
        if inverse_transform:
            return self.processor.inverse_transform(data)
        return data

class Generator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim, data_dim):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(self.num_classes,), batch_size=self.batch_size, dtype='int32')
        label_dense = Dense(dim, activation ='relu')(label)
        noise_dense = Dense(dim, activation = 'relu')(noise)
        input = concat([noise_dense, label_dense], axis=1)

        x = Dropout(0.5)(input)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=[noise, label], outputs=x)

class Discriminator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim):
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(self.num_classes,), batch_size=self.batch_size, dtype='int32')
        label_dense = Dense(dim)(label)
        events_dense = Dense(dim)(events)
        input = concat([events_dense, label_dense], axis = 1)

        x = Dropout(0.5)(input)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.5)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[events, label], outputs=x)
