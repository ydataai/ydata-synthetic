"""Tensorflow2 implementation of DoppelGANger.

Link to the original article: https://arxiv.org/pdf/1909.13403.pdf

Link to the original Python package: https://github.com/fjxmlzn/DoppelGANger"""

from typing import Optional, NamedTuple

from tensorflow import repeat, expand_dims, GradientTape, sqrt, reduce_mean, reduce_sum
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input, LSTM, Dense
from tensorflow.random import uniform
from tensorflow.dtypes import float32

from ydata_synthetic.synthesizers.gan import BaseModel
from ydata_synthetic.utils.gumbel_softmax import GumbelSoftmaxActivation

class DoppelGANger(BaseModel):
    """DoppelGANger implementation"""
    __MODEL__ = 'DoppelGANger'

    def __init__(self, model_parameters, ):
        super().__init__(model_parameters)

    def define_gan(self):
        #TODO: compile generator with metadata and TS generators
        #TODO: compile critic with auxiliary and TS critic
        raise NotImplementedError

    def gradient_penalty(self, real, fake):
        epsilon = uniform([real.shape[0], 1], 0.0, 1.0, dtype=float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = sqrt(reduce_sum(gradients ** 2))
        d_regularizer = reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def train(self, data, train_args):
        super().train(data=data, num_cols=None, cat_cols=None)
        self.define_gan()

class RealMGenerator:
    """'Real' metadata generator.
    Produces the metadata associated with each sequence."""
    def __init__(self, batch_size: int=64):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time"""
        self.batch_size = batch_size

    def build_model(self, meta_dim: int, noise_dim: int, dim: int=100, activation_info: Optional[NamedTuple]=None,
                    tau: Optional[float] = None) -> Model:
        noise = Input(shape=noise_dim, batch_size=self.batch_size)

        meta = Dense(dim)(noise)
        meta = Dense(dim)(meta)
        meta = Dense(meta_dim)(meta)
        if activation_info:
            meta = GumbelSoftmaxActivation(activation_info, tau=tau)(meta)
        return Model(inputs=noise, outputs=meta)

class FakeMGenerator:
    """'Fake' metadata generator.
    Produces min and max of each timeseries."""
    def __init__(self, batch_size: int=64):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time"""
        self.batch_size = batch_size

    def build_model(self, ts_dim: int, meta_dim: int, noise_dim: int, dim: int=100) -> Model:
        #TODO: fix input_shape; metadata depends on dataset; noise depends on user parameterization
        #TODO: Remove debug code (prints/asserts)
        _, n_ts = ts_dim
        metadata = Input(shape=meta_dim, batch_size=self.batch_size)
        noise = Input(shape=noise_dim, batch_size=self.batch_size)

        input = Concatenate(axis=1)([metadata, noise])

        minmax = Dense(dim)(input)
        minmax = Dense(dim)(minmax)
        minmax = Dense(n_ts*2)(minmax)
        return Model(inputs=[metadata, noise], outputs=minmax)

class TSGenerator:
    """Recurrent model for timeseries generation."""
    def __init__(self, batch_size: int=64, s_len: int=5):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time.
            s_len(int): Length of samples produced by the one-to-many generator when passing one element of a sequence.
                Also called 'batch generation' in the article."""
        self.batch_size = batch_size
        self.s_len = s_len

    def build_model(self, ts_dim: int, meta_dim: int, noise_dim: int, dim: int=100) -> Model:
        seq_len, n_ts = ts_dim
        assert seq_len%self.s_len==0, "The length of the sequences should be a multiple of s_len."
        n_reps = int(seq_len/self.s_len)
        min_max = Input(shape=2*n_ts, batch_size=self.batch_size)
        metadata = Input(shape=meta_dim, batch_size=self.batch_size)
        noise = Input(shape=noise_dim, batch_size=self.batch_size)
        sequence = Input(shape=ts_dim, batch_size=self.batch_size)

        mm_rep = repeat(expand_dims(min_max, axis=1), n_reps, 1)
        meta_rep = repeat(expand_dims(metadata, axis=1), n_reps, 1)
        noise_rep = repeat(expand_dims(noise, axis=1), n_reps, 1)
        sequence_skips = sequence[:, ::self.s_len, :]

        input = Concatenate(axis=2)([mm_rep, meta_rep, noise_rep, sequence_skips])
        rnn = repeat(input, repeats=self.s_len, axis=1)
        rnn = LSTM(dim, return_sequences=True)(rnn)
        rnn = Dense(n_ts, activation='tanh')(rnn)
        return Model(inputs=[min_max, metadata, noise, sequence], outputs=rnn)

class AuxiliarCritic:
    """Auxiliar Critic network, produces a score attributed to the 'realness' of metadata of a sequence.
    This network is called a 'discriminator' in the original article.
    We keep the name critic for consistency with other models in the package using Wasserstein loss."""
    def __init__(self, batch_size: int=64):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time."""
        self.batch_size = batch_size

    def build_model(self, ts_dim: int, meta_dim: int, dim: int=100) -> Model:
        _, n_ts = ts_dim
        min_max = Input(shape=2*n_ts, batch_size=self.batch_size)
        metadata = Input(shape=meta_dim, batch_size=self.batch_size)
        sequence = Input(shape=ts_dim, batch_size=self.batch_size)

        input = Concatenate(axis=1)([min_max, metadata])
        score = Dense(dim)(input)
        score = Dense(dim)(score)
        score = Dense(1)(score)
        return Model(inputs=[min_max, metadata, sequence], outputs=score)

class Critic:
    """Critic network, produces a score attributed to the 'realness' of a full synthetic sequence (with metadata).
    This network is called a 'discriminator' in the original article.
    We keep the name critic for consistency with other models in the package using Wasserstein loss."""
    def __init__(self, batch_size: int=64):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time."""
        self.batch_size = batch_size

    def build_model(self, ts_dim: int, meta_dim: int, dim: int=100) -> Model:
        seq_len, n_ts = ts_dim
        min_max = Input(shape=2*n_ts, batch_size=self.batch_size)
        metadata = Input(shape=meta_dim, batch_size=self.batch_size)
        sequence = Input(shape=ts_dim, batch_size=self.batch_size)

        mm_rep = repeat(expand_dims(min_max, axis=1), seq_len, 1)
        meta_rep = repeat(expand_dims(metadata, axis=1), seq_len, 1)

        input = Concatenate(axis=2)([mm_rep, meta_rep, sequence])
        score = Dense(dim)(input)
        score = Dense(dim)(score)
        score = Dense(1)(score)
        return Model(inputs=[min_max, metadata, sequence], outputs=score)


if __name__ == '__main__':
    from ydata_synthetic.synthesizers import ModelParameters
    from numpy import load

    gan_args = ModelParameters()
    synth = DoppelGANger(gan_args)

    # Dataset - We are using the FCC MBA dataset taken from the original implementation data repo: https://drive.google.com/drive/folders/19hnyG8lN9_WWIac998rT6RtBB9Zit70X
    dataset = load('/home/fsantos/GitRepos/ydata-synthetic/data/data_train.npz')
    for array in dataset.files:
        if array == 'data_attribute':
            pass
            #print(array, '\n', dataset[array][:,:15].sum(axis=1))
        else:
            pass
            #print(array, '\n', dataset[array].shape)

    meta_gen = RealMGenerator()
    meta_gen.build_model(meta_dim=5, noise_dim=15)

    minmax_gen = FakeMGenerator()
    minmax_gen.build_model(ts_dim=(15,3), meta_dim=5, noise_dim=15)

    gen = TSGenerator()
    gen.build_model(ts_dim=(15,3), meta_dim=5, noise_dim=15)

    aux_cri = AuxiliarCritic()
    aux_cri.build_model(ts_dim=(15,3), meta_dim=5)

    cri = Critic()
    cri.build_model(ts_dim=(15,3), meta_dim=5)


