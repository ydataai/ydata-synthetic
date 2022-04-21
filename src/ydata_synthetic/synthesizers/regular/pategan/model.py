"PATEGAN implementation supporting Differential Privacy budget specification."
# pylint: disable = W0622, E0401
from math import log
from typing import List, NamedTuple, Optional

from tensorflow import (GradientTape, clip_by_value, constant, expand_dims, ones_like, tensor_scatter_nd_update,
                        transpose, zeros, zeros_like)
from tensorflow.data import Dataset
from tensorflow.dtypes import cast, float64, int64
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.math import abs, exp, pow, reduce_sum, square
from tensorflow.random import uniform
from tensorflow_probability import distributions

from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.gan import BaseModel
from ydata_synthetic.utils.gumbel_softmax import GumbelSoftmaxActivation


# pylint: disable=R0902
class PATEGAN(BaseModel):
    "A basic PATEGAN synthesizer implementation with configurable differential privacy budget."

    __MODEL__='PATEGAN'

    def __init__(self, model_parameters, n_teachers: int, target_delta: float = 1e-5, target_epsilon: float = 5e-2):
        super().__init__(model_parameters)
        self.n_teachers = n_teachers
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    # pylint: disable=W0201
    def define_gan(self, activation_info: Optional[NamedTuple] = None):
        def discriminator():
            return Discriminator(self.batch_size).build_model((self.data_dim,), self.layers_dim)

        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,
                        activation_info=activation_info)
        self.s_discriminator = discriminator()
        self.t_discriminators = [discriminator() for i in range(self.n_teachers)]

        generator_optimizer = Adam(learning_rate=self.g_lr)
        discriminator_optimizer = Adam(learning_rate=self.d_lr)

        loss_fn = BinaryCrossentropy(from_logits=True)
        self.generator.compile(loss=loss_fn, optimizer=generator_optimizer)
        self.s_discriminator.compile(loss=loss_fn, optimizer=discriminator_optimizer)
        for teacher in self.t_discriminators:
            teacher.compile(loss=loss_fn, optimizer=discriminator_optimizer)

    # pylint: disable = C0103
    @staticmethod
    def _moments_acc(n_teachers, votes, lap_scale, l_list):
        q = (2 + lap_scale * abs(2 * votes - n_teachers))/(4 * exp(lap_scale * abs(2 * votes - n_teachers)))

        update = []
        for l in l_list:
            clip = 2 * square(lap_scale) * l * (l + 1)
            t = (1 - q) * pow((1 - q) / (1 - exp(2 * lap_scale) * q), l) + q * exp(2 * lap_scale * l)
            update.append(reduce_sum(clip_by_value(t, clip_value_min=-clip, clip_value_max=clip)))
        return cast(update, dtype=float64)

    def get_data_loader(self, data) -> List[Dataset]:
        "Obtain a List of TF Datasets corresponding to partitions for each teacher in n_teachers."
        loader = []
        SHUFFLE_BUFFER_SIZE = 100

        for teacher_id in range(self.n_teachers):
            start_id = int(teacher_id * len(data) / self.n_teachers)
            end_id = int((teacher_id + 1) * len(data) / self.n_teachers if \
                teacher_id != (self.n_teachers - 1) else len(data))
            loader.append(Dataset.from_tensor_slices(data[start_id:end_id:])\
                .batch(self.batch_size).shuffle(SHUFFLE_BUFFER_SIZE).repeat().as_numpy_iterator())
        return loader

    # pylint:disable=R0913
    def train(self, data, num_cols: List[str], cat_cols: List[str], n_teacher_iters: int = 5, n_student_iters: int = 5,
              n_moments: int = 100, lap_scale: float = 1e-4):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
            n_teacher_iters: Number of train steps of each teacher discriminator per global step
            n_student_iters: Number of train steps of the student discriminator per global step
            n_moments: Number of moments accounted in the privacy budget computations
            lap_scale: Inverse laplace noise scale multiplier
        """
        super().train(data, num_cols, cat_cols)

        data = self.processor.transform(data)
        self.data_dim = data.shape[1]
        self.define_gan(self.processor.col_transform_info)

        alpha = cast([0.0 for _ in range(n_moments)], float64)
        l_list = 1 + cast(range(n_moments), float64)
        lap_scale = cast(lap_scale, float64)

        train_loaders = self.get_data_loader(data)

        steps = 0
        epsilon = 0

        while epsilon < self.target_epsilon:
            # train the teacher descriminator
            for t_2 in range(n_teacher_iters):
                for train_loader, t_discriminator in zip(train_loaders, self.t_discriminators):
                    z = uniform([self.batch_size, self.noise_dim], dtype=float64)

                    with GradientTape() as disc_tape:
                        # loss on real data
                        real_batch=train_loader.next()
                        real_output = t_discriminator(real_batch, training=True)
                        real_loss_disc = t_discriminator.loss(ones_like(real_output), real_output)

                        # loss on fake data
                        fake = self.generator(z)
                        fake_output = t_discriminator(fake, training=True)
                        fake_loss_disc = t_discriminator.loss(zeros_like(fake_output), fake_output)

                        # compute and apply gradients
                        disc_loss = real_loss_disc + fake_loss_disc
                        disc_grad = disc_tape.gradient(disc_loss, t_discriminator.trainable_variables)
                        t_discriminator.optimizer.apply_gradients(zip(disc_grad, t_discriminator.trainable_variables))

            # train the student discriminator
            for t_3 in range(n_student_iters):
                z = uniform([self.batch_size, self.noise_dim], dtype=float64)

                with GradientTape() as stu_tape:
                    # student discriminator loss
                    fake = self.generator(z)
                    predictions, clean_votes = self._pate_voting(fake, self.t_discriminators, lap_scale)
                    outputs = self.s_discriminator(fake)
                    stu_loss = self.s_discriminator.loss(predictions, outputs)

                    # compute and apply gradients
                    gradients_of_stu = stu_tape.gradient(stu_loss, self.s_discriminator.trainable_variables)
                    self.s_discriminator.optimizer.apply_gradients(zip(gradients_of_stu, self.s_discriminator.trainable_variables))

                    # update the moments
                    alpha = alpha + self._moments_acc(self.n_teachers, clean_votes, lap_scale, l_list)

            # train the generator
            z = uniform([self.batch_size, self.noise_dim], dtype=float64)
            with GradientTape() as gen_tape:
                fake = self.generator(z)
                output = self.s_discriminator(fake)
                loss_gen = self.generator.loss(ones_like(output), output)

                # compute and apply gradients
                gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
                self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            # Calculate the current privacy cost
            epsilon = min((alpha - log(self.target_delta)) / l_list).numpy()
            print(f"Step : {steps} Loss SD : {stu_loss:.2e} Loss G : {loss_gen:.2e} Epsilon : {epsilon:.2e}")

            steps += 1

    def _pate_voting(self, data, netTD, lap_scale):
        results = zeros([len(netTD), self.batch_size], dtype=int64)
        for i in enumerate(netTD):
            output = netTD[i](data, training=True)
            pred = transpose(cast((output > 0.5), int64))
            results = tensor_scatter_nd_update(results, constant([[i]]), pred)

        clean_votes = expand_dims(cast(reduce_sum(results, 0), dtype=float64), 1)
        noise_sample = distributions.Laplace(loc=0, scale=1/lap_scale).sample(clean_votes.shape)
        noisy_results = clean_votes + cast(noise_sample, float64)
        noisy_labels = cast((noisy_results > len(netTD)/2), float64)

        return noisy_labels, clean_votes


class Discriminator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4)(input)
        x = ReLU()(x)
        x = Dense(dim * 2)(x)
        x = Dense(1)(x)
        return Model(inputs=input, outputs=x)


class Generator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim, activation_info: Optional[NamedTuple] = None, tau: Optional[float] = None):
        input = Input(shape=input_shape, batch_size = self.batch_size)
        x = Dense(dim)(input)
        x = ReLU()(x)
        x = Dense(dim * 2)(x)
        x = Dense(data_dim)(x)
        if activation_info:
            x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
        return Model(inputs=input, outputs=x)
