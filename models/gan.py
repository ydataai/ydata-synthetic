from abc import ABC
from abc import abstractmethod

import os
from easydict import EasyDict as edict
from tensorflow.python import keras

class Model(ABC):
    def __init__(
            self,
            model_parameters: edict = None,
    ):
        self._model_parameters = model_parameters
        [self.batch_size, self.lr, self.noise_dim,
         self.data_dim, self.layers_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    @abstractmethod
    def define_gan(self) -> keras.Model:
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        return network.trainable_variables

    @property
    def model(self):
        return self._model

    @property
    def model_parameters(self) -> edict:
        return self._model_parameters

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def train(self, data, train_arguments):
        raise NotImplementedError

    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return