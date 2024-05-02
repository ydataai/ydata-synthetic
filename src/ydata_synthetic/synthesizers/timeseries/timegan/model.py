"""
    TimeGAN class implemented accordingly with:
    Original code can be found here: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
import numpy as np
from pandas import DataFrame
from tensorflow import function
from tensorflow import data as tfdata
from tensorflow import nn
from tensorflow.keras import Model
from tensorflow.keras.layers import (GRU, LSTM, Dense, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (BinaryCrossentropy, MeanSquaredError)
from ydata_synthetic.synthesizers.base import BaseGANModel, ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

def make_net(model: Model,
             n_layers: int,
             hidden_units: int,
             output_units: int,
             net_type: str = 'GRU') -> Model:
    if net_type == 'GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                          return_sequences=True,
                          name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                          return_sequences=True,
                          name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model

class TimeGAN(BaseGANModel):
    """
    TimeGAN class.
    """
    __MODEL__ = 'TimeGAN'

    def __init__(self,
                 model_parameters: ModelParameters):
        """
        Initialize the TimeGAN class.

        Args:
            model_parameters: ModelParameters object.
        """
        super().__init__(model_parameters)
        self.seq_len = None
        self.n_seq = None
        self.hidden_dim = model_parameters.latent_dim
        self.gamma = model_parameters.gamma
        self.num_cols = None

    def fit(self,
            data: DataFrame,
            train_arguments: TrainParameters,
            num_cols: List[str],
            non_seq_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None):
        """
        Fit the TimeGAN model.

        Args:
            data: DataFrame object.
            train_arguments: TrainParameters object.
            num_cols: List of numerical column names.
            non_seq_cols: List of non-sequential column names.
            cat_cols: List of categorical column names.

        Raises:
            NotImplementedError: If categorical columns are provided.
        """
        super().fit(data=data,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    train_arguments=train_arguments)
        if cat_cols:
            raise NotImplementedError("TimeGAN does not support categorical features.")
        self.num_cols = num_cols
        self.seq_len = train_arguments.sequence_length
        self.n_seq = train_arguments.number_sequences
        processed_data = real_data_loading(data[self.num_cols].values,
                                           seq_len=self.seq_len)
        self.train(data=processed_data,
                   train_steps=train_arguments.epochs)

    def sample(self,
               n_samples: int) -> List[DataFrame]:
        """
        Sample new data from the TimeGAN.

        Args:
            n_samples: Number of samples to be generated.

        Returns:
            List of DataFrame objects.
        """
        Z_ = next(self.get_batch_noise(size=n_samples))
        records = self.generator(Z_)
        data = []
        for i in range(records.shape[0]):
            data.append(DataFrame(records[i],
                                 columns=self.num_cols))
        return data

    def define_gan(self):
        """
        Define the GAN architecture.
        """
        self.generator_aux = Generator(self.hidden_dim).build()
        self.supervisor = Supervisor(self.hidden_dim).build()
        self.discriminator = Discriminator(self.hidden_dim).build()
        self.recovery = Recovery(self.hidden_dim,
                                 self.n_seq).build()
        self.embedder = Embedder(self.hidden_dim).build()

        X = Input(shape=[self.seq_len, self.n_seq],
                  batch_size=self.batch_size,
                  name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq],
                  batch_size=self.batch_size,
                  name='RandomNoise')

        #--------------------------------
        # Building the AutoEncoder
        #--------------------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X,
                                 outputs=X_tilde,
                                 name='Autoencoder')

        #---------------------------------
        # Adversarial Supervise Architecture
        #---------------------------------
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = Model(inputs=Z,
                                            outputs=Y_fake,
                                            name='AdversarialSupervised')

        #---------------------------------
        # Adversarial architecture in latent space
        #---------------------------------
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = Model(inputs=Z,
                                          outputs=Y_fake_e,
                                          name='AdversarialEmbedded')
        # ---------------------------------
        # Synthetic data generation
        # ---------------------------------
        X_hat = self.recovery(H_hat)
        self.generator = Model(inputs=Z,
                               outputs=X_hat,
                               name='Generator')

        # --------------------------------
        # Final discriminator model
        # --------------------------------
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name="Discriminator")

        # ----------------------------
        # Define the loss functions
        # ----------------------------
        self._mse = MeanSquaredError()
        self._bce = BinaryCrossentropy()

    @function
    def train_autoencoder(self,
                           x: tf.Tensor,
                           opt: Adam) -> tf.Tensor:
        """
        Train the autoencoder.

        Args:
            x: Input tensor.
            opt: Adam optimizer.

        Returns:
            Tensor of the embedding loss.
        """
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x,
                                          x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0,
                                  var_list)
        opt.apply_gradients(zip(gradients,
                                var_list))
        return tf.sqrt(embedding_loss_t0)

    @function

