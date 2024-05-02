"""
    CGAN architecture implementation file
"""
import os
from os import path
from typing import Optional, NamedTuple, List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Dropout, Input, concatenate)
from tensorflow.keras.optimizers import Adam

# Import ydata synthetic classes
from ....synthesizers import TrainParameters
from ....synthesizers.base import ConditionalModel

class CGAN(ConditionalModel):
    """
    CGAN model for discrete conditions
    """

    __MODEL__ = 'CGAN'

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initialize the CGAN model

        Args:
            model_parameters: Model parameters
        """
        self._col_order = None
        super().__init__(model_parameters)

    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        """
        Define the trainable model components.

        Args:
            activation_info (Optional[NamedTuple]): Activation information
        """
        self.generator = Generator(self.batch_size).build_model(
            input_shape=(self.noise_dim,),
            label_shape=(self.label_dim,),
            dim=self.layers_dim,
            data_dim=self.data_dim,
            activation_info=activation_info,
            tau=self.tau
        )

        self.discriminator = Discriminator(self.batch_size).build_model(
            input_shape=(self.data_dim,),
            label_shape=(self.label_dim,),
            dim=self.layers_dim
        )

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

class Generator():
    """
    Standard discrete conditional generator.
    """
    def __init__(self, batch_size):
        """
        Initialize the generator

        Args:
            batch_size: Batch size
        """
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None, tau: Optional[float] = None) -> Model:
        """
        Create model components.

        Args:
            input_shape: Input shape
            label_shape: Label shape
            dim: Hidden layers dimensions
            data_dim: Output dimensionality
            activation_info (Optional[NamedTuple]): Activation information
            tau (Optional[float]): Gumbel-Softmax non-negative temperature

        Returns:
            Generator model
        """
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label_v = Input(shape=label_shape)
        x = concatenate([noise, label_v])
        x = Dense(dim, activation='relu')(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)

        if activation_info:
            x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)

        return Model(inputs=[noise, label_v], outputs=x)

class Discriminator():
    """
    Standard discrete conditional discriminator.
    """
    def __init__(self, batch_size):
        """
        Initialize the discriminator

        Args:
            batch_size: Batch size
        """
        self.batch_size = batch_size

    def build_model(self, input_shape, label_shape, dim) -> Model:
        """
        Create model components.

        Args:
            input_shape: Input shape
            label_shape: Label shape
            dim: Hidden layers dimensions

        Returns:
            Discriminator model
        """
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=label_shape, batch_size=self.batch_size)
        input_ = concatenate([events, label])
        x = Dense(dim * 4, activation='relu')(input_)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[events, label], outputs=x)

# Gumbel-Softmax activation layer
class GumbelSoftmaxActivation(tf.keras.layers.Layer):
    """
    Gumbel-Softmax activation layer
    """
    def __init__(self, activation_info: NamedTuple, tau: Optional[float] = None, **kwargs):
        """
        Initialize the Gumbel-Softmax activation layer

        Args:
            activation_info: Activation information
            tau (Optional[float]): Non-negative temperature
            **kwargs: Additional keyword arguments
        """
        self.activation = activation_info.activation
        self.tau = tau
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer

        Args:
            input_shape: Input shape
        """
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        Call the layer

        Args:
            inputs: Input tensor
            training (bool, optional): Training flag
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor
        """
        if self.tau is None:
            self.tau = tf.constant(1.0, dtype=tf.float32)

        uniform_noise = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1)
        gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + 1e-20))
        y = (inputs + gumbel_noise) / self.tau
        y = tf.nn.softmax(y, axis=-1)

        if self.activation is not None:
            y = self.activation(y)

        return y

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape

        Args:
            input_shape: Input shape

        Returns:
            Output shape
        """
        return input_shape[0], input_shape[1], np.prod(input_shape[2:])

    def get_config(self):
        """
        Get the layer configuration

        Returns:
            Layer configuration dictionary
        """
        config = {
            'activation': self.activation,
            'tau': self.tau
        }
        base_config = super().get_config()
        return {**base_config, **config}
