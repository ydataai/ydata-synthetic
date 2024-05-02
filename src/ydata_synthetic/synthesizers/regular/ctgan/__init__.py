# models.py

import torch
from .model import CTGAN

class GenerativeModel:
    """
    A base class for generative models.
    """
    def __init__(self):
        self.model = None

    def train(self, train_data, valid_data=None, **kwargs):
        """
        Trains the generative model on the given train data.

        :param train_data: The training data.
        :param valid_data: The validation data.
        :param kwargs: Additional keyword arguments for the training process.
        """
        raise NotImplementedError

    def generate(self, num_samples, **kwargs):
        """
        Generates new samples using the trained model.

        :param num_samples: The number of samples to generate.
        :param kwargs: Additional keyword arguments for the generation process.
        :return: A tensor of shape (num_samples, num_features) containing the generated samples.
        """
        raise NotImplementedError


class CTGANModel(GenerativeModel):
    """
    A CTGAN model for generating synthetic tabular data.
    """
    def __init__(self, num_units=128, num_layers=3, learning_rate=1e-3, **kwargs):
        """
        Initializes the CTGAN model.

        :param num_units: The number of units in each layer of the generator and discriminator.
        :param num_layers: The number of layers in the generator and discriminator.
        :param learning_rate: The learning rate for the optimizer.
        :param kwargs: Additional keyword arguments for the CTGAN model.
        """
        super().__init__()
        self.model = CTGAN(num_units=num_units, num_layers=num_layers, learning_rate=learning_rate, **kwargs)

    def train(self, train_data, valid_data=None, num_epochs=100, batch_size=64, **kwargs):
        """
        Trains the CTGAN model on the given train data.

        :param train_data: The training data.
        :param valid_data: The validation data.
        :param num_epochs: The number of training epochs.
        :param batch_size: The batch size for training.
        :param kwargs: Additional keyword arguments for the CTGAN model.
        """
        self.model.train(train_data, valid_data, num_epochs, batch_size, **kwargs)

    def generate(self, num_samples, **kwargs):
        """
        Generates new samples using the trained CTGAN model.

        :param num_samples: The number of samples to generate.
        :param kwargs: Additional keyword arguments for the CTGAN model.
        :return: A tensor of shape (num_samples, num_features) containing the generated samples.
        """
        return self.model.generate(num_samples, **kwargs)
