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
        """
        Initialize the latent variables.

        Args:
            self: (todo): write your description
            model_parameters: (str): write your description
        """
        self._model_parameters = model_parameters
        [self.batch_size, self.lr, self.beta_1, self.beta_2, self.noise_dim,
         self.data_dim, self.layers_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        """
        Call the model.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        return self.model(inputs=inputs, **kwargs)

    def define_gan(self):
        """
        Define a list of the current class.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        """
        Return the variables of the given network.

        Args:
            self: (todo): write your description
            network: (todo): write your description
        """
        return network.trainable_variables

    @property
    def model_parameters(self):
        """
        Returns the model parameters.

        Args:
            self: (todo): write your description
        """
        return self._model_parameters

    @property
    def model_name(self):
        """
        The name of the model.

        Args:
            self: (todo): write your description
        """
        return self.__class__.__name__

    def train(self, data, train_arguments):
        """
        Train the model.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            train_arguments: (todo): write your description
        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Return a tensor.

        Args:
            self: (todo): write your description
            n_samples: (int): write your description
        """
        steps = n_samples // self.batch_size + 1
        data = []
        for step in tqdm.trange(steps):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(pd.DataFrame(records))
        return pd.concat(data)

    def save(self, path, name):
        """
        Save the model to disk.

        Args:
            self: (todo): write your description
            path: (str): write your description
            name: (str): write your description
        """
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return