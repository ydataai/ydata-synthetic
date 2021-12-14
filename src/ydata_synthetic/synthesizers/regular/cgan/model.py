"""CGAN implementation"""
import os
from os import path
from typing import List, Tuple, Union

import numpy as np
from numpy import array, empty, hstack, ndarray, vstack, save
from numpy.random import normal
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype
from tensorflow import convert_to_tensor
from tensorflow import data as tfdata
from tensorflow import dtypes, expand_dims, tile
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Dropout, Embedding, Flatten, Input,
                                     multiply)
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.gan import BaseModel
from ydata_synthetic.utils.gumbel_softmax import GumbelSoftmaxActivation


class CGAN(BaseModel):
    "CGAN model for discrete conditions."

    __MODEL__='CGAN'

    def __init__(self, model_parameters, num_classes):
        self.num_classes = num_classes
        self._label_col = None
        super().__init__(model_parameters)

    @property
    def label_col(self) -> str:
        "Returns the name of the label used for conditioning the model."
        return self._label_col

    @label_col.setter
    def label_col(self, data_label: Tuple[Union[DataFrame, array], str]):
        "Validates the label_col format, raises ValueError if invalid."
        data, label_col = data_label
        assert label_col in data.columns, f"The column {label_col} could not be found on the provided dataset and \
            cannot be used as condition."
        assert data[label_col].isna().sum() == 0, "The label column contains NaN values, please impute or drop the \
            respective records before proceeding."
        assert is_float_dtype(data[label_col]) or is_integer_dtype(float), "The label column is expected to be an \
            integer or a float dtype to ensure the function of the embedding layer."
        unique_frac = data[label_col].nunique()/len(data.index)
        assert unique_frac < 1, "The provided column {label_col} is constituted by unique values and is not suitable \
            to be used as condition."

    def define_gan(self):
        self.generator = Generator(self.batch_size, self.num_classes). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,
                        activation_info = activation_info)

        self.discriminator = Discriminator(self.batch_size, self.num_classes). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

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
            yield normal(size=self.noise_dim)

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

    def train(self, data: Union[DataFrame, array], label_col: str, train_arguments: TrainParameters, num_cols: List[str],
              cat_cols: List[str]):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            label: The name of the column to be used as a label and condition for the training
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        # Validating the label column
        self.label_col = (data, label_col)

        # Separating labels from the rest of the data to fit the data processor
        data, label = data.loc[:, data.columns != label_col], expand_dims(data[label_col], 1)

        super().train(data, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        self.define_gan(self.processor.col_transform_info if preprocess else None)

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
        save('./cache/' + train_arguments.cache_prefix + f'_sample_{epoch}.npy', self.sample(array([label[0]]), 1000))

    def sample(self, condition: ndarray, n_samples: int,) -> ndarray:
        """Produce n_samples by conditioning the generator with condition."""
        assert condition.shape[0] == 1, \
            "A condition with cardinality one is expected."
        steps = n_samples // self.batch_size + 1
        data = []
        z_dist = self.get_batch_noise()
        condition = expand_dims(convert_to_tensor(condition, dtypes.float32), axis=0)
        cond_seq = tile(condition, multiples=[self.batch_size, 1])
        for _ in trange(steps, desc='Synthetic data generation'):
            records = empty(shape=(self.batch_size, self.data_dim))
            records = self.generator([next(z_dist), cond_seq], training=False)
            data.append(records)
        data = self.processor.inverse_transform(array(vstack(data)))
        data[self.label_col] = tile(condition, multiples=[data.shape[0], 1])
        return data


# pylint: disable=R0903
class Generator():
    "Standard discrete conditional generator."
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        input = multiply([noise, label_embedding])

        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        if activation_info:
            x = GumbelSoftmaxActivation(activation_info).call(x)
        return Model(inputs=[noise, label], outputs=x)


# pylint: disable=R0903
class Discriminator():
    "Standard discrete conditional discriminator."
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim):
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        events_flat = Flatten()(events)
        input = multiply([events_flat, label_embedding])

        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[events, label], outputs=x)
