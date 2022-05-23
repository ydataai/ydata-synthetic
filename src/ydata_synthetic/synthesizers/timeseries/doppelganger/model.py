"""Tensorflow2 implementation of DoppelGANger.

Link to the original article: https://arxiv.org/pdf/1909.13403.pdf

Link to the original Python package: https://github.com/fjxmlzn/DoppelGANger"""

from typing import Optional, NamedTuple, Tuple, List

from tensorflow import repeat, expand_dims, GradientTape, sqrt, reduce_mean, reduce_sum, Tensor
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Concatenate, Input, LSTM, Dense
from tensorflow.random import uniform, normal
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

def get_noise_sample(batch_size: int, noise_dim: int):
    return normal([batch_size, noise_dim])

class RealMGenerator(Model):
    """'Real' metadata generator.
    Produces the metadata associated with each sequence."""
    def __init__(self, meta_dim: int, noise_dim: int, dim: int=100, activation_info: Optional[NamedTuple]=None,
                    tau: Optional[float] = None, batch_size: int=64):
        """Arguments:
            meta_dim(int): The cardinality of the produced metadata
            noise_dim(int): The cardinality of the noise array internally consumed by the model
            dim(int): The number of units used in the model layers
            activation_info(Optional[NamedTuple]): Specifies the type of activations to be used for the output
            tau(Optional[float]): Tau parameter of the Gumbel-Softmax (if categorical columns were specified)
            batch_size(int): Number of sequences passed to the model in train time"""
        super().__init__()

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.meta_dim = meta_dim
        self.dim = dim
        self.activation_info = activation_info
        self.tau = tau

    def build(self, _):
        self.model = Sequential(name='Real Metadata Generator')
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.meta_dim))
        if self.activation_info:
            self.model.add(GumbelSoftmaxActivation(self.activation_info, tau=self.tau))

    def call(self, _, training=False) -> Tensor:
        noise = get_noise_sample(self.batch_size, self.noise_dim)
        return self.model(noise)

class FakeMGenerator(Model):
    """'Fake' metadata generator.
    Produces min and max of each timeseries."""
    def __init__(self, noise_dim: int, ts_dim: Tuple[int], dim: int=100, batch_size: int=64):
        """Arguments:
            noise_dim(int): The cardinality of the noise array internally consumed by the model
            dim(int): The number of units used in the model layers
            ts_dim(Tuple[int]): Tuple with dimensions of the synthesized time series as: (seq_length, n_series)
            batch_size(int): Number of sequences passed to the model in train time"""
        super().__init__()

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dim = dim
        _, self.n_ts = ts_dim

    def build(self, _):
        self.model = Sequential(name='Fake Metadata Generator')
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.n_ts*2))

    def call(self, metadata: Tensor, training=False) -> Tensor:
        noise = get_noise_sample(self.batch_size, self.noise_dim)
        input = Concatenate(axis=1)([metadata, noise])
        return self.model(input)

class TSGenerator(Model):
    """Recurrent model for timeseries generation."""
    def __init__(self, ts_dim: Tuple[int], noise_dim: int, meta_dim: int, dim: int=100, s_len: int=5, batch_size: int=64):
        """Arguments:
            ts_dim(Tuple[int]): Tuple with dimensions of the synthesized time series as: (seq_length, n_series)
            noise_dim(int): The cardinality of the noise array internally consumed by the model
            dim(int): The number of units used in the model layers
            meta_dim(int): The cardinality of the produced metadata
            s_len(int): Length of samples produced by the one-to-many generator when passing one element of a sequence.
                Also called 'batch generation' in the article.
            batch_size(int): Number of sequences passed to the model in train time."""
        super().__init__()

        self.batch_size = batch_size
        self.seq_len, self.n_ts = ts_dim
        self.s_len = s_len
        self.noise_dim = noise_dim
        self.dim = dim
        self.meta_dim = meta_dim

        assert self.seq_len%self.s_len==0, "The length of the sequences should be a multiple of s_len."

    def build(self, _) -> Model:
        self.model = Sequential(name='Time Series Generator')
        self.model.add(LSTM(self.dim, return_sequences=True))
        self.model.add(Dense(self.n_ts, activation='tanh'))

    def call(self, inputs: List[Tensor], training=False) -> Tensor:
        meta, mm = inputs
        noise = get_noise_sample(self.batch_size, self.noise_dim)

        mm_rep = repeat(expand_dims(mm, axis=1), self.s_len, 1)
        meta_rep = repeat(expand_dims(meta, axis=1), self.s_len, 1)
        noise_rep = repeat(expand_dims(noise, axis=1), self.s_len, 1)
        input = Concatenate(axis=2)([meta_rep, mm_rep, noise_rep])

        outputs = []
        n_batches = int(self.seq_len/self.s_len)
        for i in range(n_batches):
            outputs.append(self.model(input))
        return Concatenate(axis=1)(outputs)


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

    meta_gen = RealMGenerator(meta_dim=5,noise_dim=16)
    meta_gen([])

    minmax_gen = FakeMGenerator(noise_dim=16,ts_dim=(15,5))
    minmax_gen(meta_gen([]))

    gen = TSGenerator(ts_dim=(15,3), noise_dim=16, meta_dim=5)
    gen([meta_gen([]), minmax_gen(meta_gen([]))])

    aux_cri = AuxiliarCritic()
    aux_cri.build_model(ts_dim=(15,3), meta_dim=5)

    cri = Critic()
    cri.build_model(ts_dim=(15,3), meta_dim=5)


