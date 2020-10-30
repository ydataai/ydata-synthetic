import os
import tqdm

import pandas as pd
import tensorflow as tf
from tensorflow.python import keras

class Model():
    def __init__(
            self,
            model_parameters
    ):
        self._model_parameters = model_parameters
        [self.batch_size, self.lr, self.beta_1, self.beta_2, self.noise_dim,
         self.data_dim, self.layers_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    def define_gan(self):
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        return network.trainable_variables

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def model_name(self):
        return self.__class__.__name__

    def train(self, data, train_arguments):
        raise NotImplementedError

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for step in tqdm.trange(steps):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(pd.DataFrame(records))
        return pd.concat(data)

    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return