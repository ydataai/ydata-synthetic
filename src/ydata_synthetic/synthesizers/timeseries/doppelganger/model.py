"""Tensorflow2 implementation of DoppelGANger.

Link to the original article: https://arxiv.org/pdf/1909.13403.pdf

Link to the original Python package: https://github.com/fjxmlzn/DoppelGANger"""

from typing import Tuple

from numpy.random import choice
from tensorflow import convert_to_tensor, gather, repeat, expand_dims, GradientTape, square, sqrt, reduce_mean, reduce_sum, squeeze, Tensor, zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Concatenate, LSTM, Dense
from tensorflow.random import uniform, normal
from tensorflow.dtypes import float32, int32
from tqdm import trange

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.gan import BaseModel

class DoppelGANger(BaseModel):
    """DoppelGANger implementation"""
    __MODEL__ = 'DoppelGANger'

    def __init__(self, model_parameters: ModelParameters, alpha: float=1, gp_weight: float=10):
        super().__init__(model_parameters)

        self.alpha = alpha
        self.critic_iter = model_parameters.n_critic
        self.seq_len = model_parameters.seq_len
        self.gp_weight = gp_weight

    def define_gan(self, meta_dim: int, ts_dim: Tuple[int], s_len: int):
        r_meta_gen = RealMGenerator(meta_dim=meta_dim, noise_dim=self.noise_dim, dim=self.layers_dim, batch_size=self.batch_size)
        f_meta_gen = FakeMGenerator(noise_dim=self.noise_dim, ts_dim=ts_dim, dim=self.layers_dim, batch_size=self.batch_size)
        self.meta_gen = Sequential([r_meta_gen, f_meta_gen], name='Metadata Generator')

        self.generator = Generator(ts_dim=ts_dim, noise_dim=self.noise_dim, meta_dim=meta_dim, dim=self.layers_dim,
                                   s_len=s_len, batch_size=self.batch_size)

        self.aux_crit = AuxiliarCritic(batch_size=self.batch_size, dim=2*self.layers_dim)
        self.critic = Critic(batch_size=self.batch_size, dim=2*self.layers_dim)

        #g_meta_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        c_aux_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        c_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        #self.meta_gen.compile(optimizer=g_meta_optimizer, loss=self.wasserstein_losses)
        self.generator.compile(optimizer=g_optimizer, loss=self.wasserstein_losses('generator'))
        self.aux_crit.compile(optimizer=c_aux_optimizer, loss=self.wasserstein_losses('aux_critic'))
        self.critic.compile(optimizer=c_optimizer, loss=self.wasserstein_losses('critic'))

    def wasserstein_losses(self, model_name: str) -> callable:
        def aux_critic_loss(real: Tensor, fake: Tensor) -> Tensor:
            with GradientTape() as t:
                t.watch([real, fake])
                d_real = self.aux_crit(real)
                d_fake = self.aux_crit(fake)
                d_reg = self.gradient_penalty(real, fake, self.aux_crit)
                loss = reduce_mean(d_fake - d_real + self.gp_weight*d_reg)
            return loss, t.gradient(loss, self.aux_crit.variables)

        def critic_loss(real: Tensor, fake: Tensor) -> Tensor:
            with GradientTape() as t:
                t.watch([real, fake])
                d_real = self.critic(real)
                d_fake = self.critic(fake)
                d_reg = self.gradient_penalty(real, fake, self.critic)
                loss = reduce_mean(d_fake - d_real + self.gp_weight*d_reg)
            return loss, t.gradient(loss, self.critic.variables)

        def generator_loss() -> Tensor:
            with GradientTape() as t:
                fake_meta = self.meta_gen(zeros([]))
                d_fake_aux = self.aux_crit(fake_meta)
                fake_record = self.generator(fake_meta)
                d_fake = self.critic(fake_record)
                loss = reduce_mean(-d_fake - d_fake_aux)
            return loss, t.gradient(loss, self.generator.variables)

        loss_mapper = {'aux_critic': aux_critic_loss,
                       'critic': critic_loss,
                       'generator': generator_loss}
        return loss_mapper[model_name]

    def gradient_penalty(self, real, fake, critic):
        shape = [self.batch_size] + [1]*(len(real.shape)-1)
        epsilon = uniform(shape, 0.0, 1.0, dtype=float32)
        x_hat = epsilon * real + (1 - epsilon) * fake
        with GradientTape() as t:
            t.watch(x_hat)
            d_hat = critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = sqrt(reduce_sum(gradients ** 2, axis=range(len(shape))[1:]))
        d_regularizer = square(ddx - 1.0)
        return d_regularizer

    def get_batch(self, metadata: Tensor, minmax: Tensor, measures: Tensor) -> Tensor:
        batch_idxs = convert_to_tensor(choice(meta.shape[0], self.batch_size), dtype=int32)
        meta_, measures_ = gather(metadata, batch_idxs), gather(measures, batch_idxs)
        minmax_ = repeat(expand_dims(minmax, axis=0), self.batch_size, axis=0)
        meta_ = Concatenate(axis=1)([meta_, minmax_])
        measures_ = Concatenate(axis=2)([repeat(expand_dims(meta_, axis=1), self.seq_len, axis=1), measures_])
        return meta_, measures_

    def train(self, data: Tuple, train_args: TrainParameters, meta_dim: int, ts_dim: Tuple[int], s_len: int=5):
        super().train(data=data, num_cols=None, cat_cols=None)

        self.define_gan(meta_dim, ts_dim, s_len)

        meta, mm, measures = [convert_to_tensor(array, dtype=float32) for array in data]

        iterations = int(abs(meta.shape[0]/self.batch_size)+1)

        for epoch in trange(train_args.epochs):
            for _ in range(iterations):
                for t in range(self.critic_iter):
                    r_meta, r_record = self.get_batch(meta, mm, measures)
                    f_meta = self.meta_gen(zeros([]))
                    f_record = self.generator(f_meta)
                    aux_crit_loss, aux_crit_grad = self.aux_crit.loss(r_meta, f_meta)
                    crit_loss, crit_grad = self.critic.loss(r_record, f_record)
                    self.aux_crit.optimizer.apply_gradients(zip(aux_crit_grad, self.aux_crit.trainable_variables))
                    self.critic.optimizer.apply_gradients(zip(crit_grad, self.critic.trainable_variables))
                gen_loss, gen_grad = self.generator.loss()
                self.generator.optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

            print(f"Epoch: {epoch} | critic loss: {crit_loss} | auxiliary critic loss: {aux_crit_loss}\
| generator loss: {gen_loss}")

def get_noise_sample(batch_size: int, noise_dim: int) -> Tensor:
    return normal([batch_size, noise_dim])


class RealMGenerator(Model):
    """'Real' metadata generator.
    Generates the metadata associated with each sequence."""
    def __init__(self, meta_dim: int, noise_dim: int, dim: int=100, batch_size: int=64):
        """Arguments:
            meta_dim(int): The cardinality of the produced metadata
            noise_dim(int): The cardinality of the noise array internally consumed by the model
            dim(int): The number of units used in the model layers
            batch_size(int): Number of sequences passed to the model in train time"""
        super().__init__()

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.meta_dim = meta_dim
        self.dim = dim

    def build(self, _):
        self.model = Sequential(name='Real Metadata Generator')
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.meta_dim))

    def call(self, _, training=False) -> Tensor:
        noise = get_noise_sample(self.batch_size, self.noise_dim)
        return self.model(noise)


class FakeMGenerator(Model):
    """'Fake' metadata generator.
    Generates min and max of each timeseries conditioned on the provided metadata.
    Returns the full metadata (real metadata input is concatenated)."""
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
        return Concatenate(axis=1)([metadata, self.model(input)])


class Generator(Model):
    """Recurrent model for timeseries generation.
    Generates TimeSeries sequences conditioned on the full metadata."""
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

    def call(self, metadata: Tensor, training=False) -> Tensor:
        noise = get_noise_sample(self.batch_size, self.noise_dim)
        meta_rep = repeat(expand_dims(metadata, axis=1), self.s_len, 1)
        noise_rep = repeat(expand_dims(noise, axis=1), self.s_len, 1)
        input = Concatenate(axis=2)([meta_rep, noise_rep])

        outputs = []
        temporal_batches = int(self.seq_len/self.s_len)
        for i in range(temporal_batches):
            outputs.append(Concatenate(axis=2)([meta_rep, self.model(input)]))
        return Concatenate(axis=1)(outputs)


class AuxiliarCritic(Model):
    """Auxiliar Critic network, produces a score attributed to the 'realness' of metadata of a sequence.
    This network is called a 'discriminator' in the original article.
    We keep the name critic for consistency with other models in the package using Wasserstein loss."""
    def __init__(self, batch_size: int=64, dim: int=100):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time."""
        super().__init__()

        self.batch_size = batch_size
        self.dim = dim

    def build(self, _) -> Model:
        self.model = Sequential(name='Auxiliar Critic')
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(1))

    def call(self, metadata: Tensor, training=False) -> Tensor:
        return squeeze(self.model(metadata))


class Critic(Model):
    """Critic network, produces a score attributed to the 'realness' of a synthetic sequence and its metadata.
    This network is called a 'discriminator' in the original article.
    We keep the name critic for consistency with other models in the package using Wasserstein loss."""
    def __init__(self, batch_size: int=64, dim: int=100):
        """Arguments:
            batch_size(int): Number of sequences passed to the model in train time."""
        super().__init__()

        self.batch_size = batch_size
        self.dim = dim

    def build(self, _) -> Model:
        self.model = Sequential(name='Critic')
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(self.dim))
        self.model.add(Dense(1))

    def call(self, sequence: Tensor, training=False) -> Tensor:
        return squeeze(reduce_mean(self.model(sequence), 1))


if __name__ == '__main__':
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from numpy import load, empty

    # Dataset preparation - We are using the FCC MBA dataset taken from the original implementation data repo: https://drive.google.com/drive/folders/19hnyG8lN9_WWIac998rT6RtBB9Zit70X
    dataset = load('/home/fsantos/GitRepos/ydata-synthetic/data/data_train.npz')

    isps = ['ISP_' + i for i in ['Windstream', 'AT&T', 'CenturyLink', 'Verizon ', 'Comcast', 'Cincinnati Bell ', 'Optimum', 'Hughes', 'Cox', 'Mediacom', 'Hawaiian Telcom', 'Wildblue/ViaSat', 'Charter', 'Frontier ', 'Verizon']]
    techs = ['Tech_' + i for i in ['Fiber', 'DSL', 'IPBB', 'Satellite', 'Cable']]
    states = ['State_' + i for i in ['NV', 'LA', 'FL', 'RI', 'NM', 'GA', 'NE', 'WI', 'SD', 'OH', 'NH', 'CO', 'NJ', 'IN', 'AZ', 'PA', 'KY', 'OR', 'MN', 'IL', 'MD', 'MT', 'MS', 'OK', 'WV', 'ME', 'NY', 'MA', 'VT', 'MI', 'Ia', 'nv', 'AR', 'MO', 'SC', 'DE', 'DC', 'IA', 'TN', 'ID', 'Al', 'CA', 'VA', 'AL', 'CT', 'TX', 'WY', 'WA', 'KS', 'sc', 'NC', 'UT', 'HI']]
    #meta = DataFrame(dataset['data_attribute'], columns = isps+techs+states)
    meta = dataset['data_attribute']

    measures = dataset['data_feature']

    mm = empty(measures.shape[2]*2)
    mm[::2] = dataset['data_feature_min']
    mm[1::2] = dataset['data_feature_max']

    gan_args = ModelParameters(lr=1e-3, betas=(0.9, 0.999), batch_size=100, layers_dim=100, seq_len=56, n_critic=5)
    synth = DoppelGANger(gan_args, alpha=1, gp_weight=10)

    train_args = TrainParameters()

    synth.train((meta, mm, measures), train_args, meta_dim=meta.shape[1], ts_dim=measures.shape[1:], s_len=4)
