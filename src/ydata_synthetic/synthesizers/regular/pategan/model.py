"PATEGAN implementation supporting Differential Privacy budget specification."
# pylint: disable = W0622, E0401
from math import log
from typing import List, NamedTuple, Optional

import tqdm
from tensorflow import (GradientTape, clip_by_value, concat, constant,
                        expand_dims, ones_like, tensor_scatter_nd_update,
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
from ydata_synthetic.utils.gumbel_softmax import ActivationInterface


# pylint: disable=R0902
class PATEGAN(BaseModel):
    "A basic PATEGAN synthesizer implementation with configurable differential privacy budget."

    __MODEL__='PATEGAN'

    def __init__(self, model_parameters, n_teachers: int, target_delta: float, target_epsilon: float):
        super().__init__(model_parameters)
        self.n_teachers = n_teachers
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    # pylint: disable=W0201
    def define_gan(self, processor_info: Optional[NamedTuple] = None):
        def discriminator():
            return Discriminator(self.batch_size).build_model((self.data_dim,), self.layers_dim)

        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,
                        processor_info=processor_info)
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
                .batch(self.batch_size).shuffle(SHUFFLE_BUFFER_SIZE))
        return loader

    # pylint:disable=R0913
    def train(self, data, class_ratios, train_arguments: TrainParameters, num_cols: List[str], cat_cols: List[str]):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            class_ratios:
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        super().train(data, num_cols, cat_cols)

        data = self.processor.transform(data)
        self.data_dim = data.shape[1]
        self.define_gan(self.processor.col_transform_info)

        self.class_ratios = class_ratios

        alpha = cast([0.0 for _ in range(train_arguments.num_moments)], float64)
        l_list = 1 + cast(range(train_arguments.num_moments), float64)

        # print("initial alpha", l_list.shape)

        cross_entropy = BinaryCrossentropy(from_logits=True)

        generator_optimizer = Adam(learning_rate=train_arguments.lr)
        disc_opt_stu = Adam(learning_rate=train_arguments.lr)
        disc_opt_t = [Adam(learning_rate=train_arguments.lr) for i in range(self.n_teachers)]

        train_loader = self.get_data_loader(data, self.batch_size)

        steps = 0
        epsilon = 0

        category_samples = distributions.Categorical(probs=self.class_ratios, dtype=float64)

        while epsilon < self.target_epsilon:
            # train the teacher descriminator
            for t_2 in range(train_arguments.num_teacher_iters):
                for i in range(self.n_teachers):
                    inputs, categories = None, None
                    for b, data_ in enumerate(train_loader[i]):
                        inputs, categories = data_, b  # categories = 0, data_ holds the first batch, why do we do this?
                        #categories will give zero value in each loop as the loop break after running the first time
                        #inputs will have only the first batch of data
                        break

                    with GradientTape() as disc_tape:
                        # train with real
                        dis_data = concat([inputs, zeros((self.batch_size, 1), dtype=float64)], 1)  # Why do we append a column of zeros instead of categories?
                        # print("1st batch data", dis_data.shape)
                        real_output = self.t_discriminators[i](dis_data, training=True)
                        # print(real_output.shape, tf.ones.shape)

                        # train with fake
                        z = uniform([self.batch_size, self.noise_dim], dtype=float64)
                        # print("uniformly distributed noise", z.shape)

                        sample = expand_dims(category_samples.sample(self.batch_size), axis=1)
                        # print("category", sample.shape)

                        fake = self.generator(concat([z, sample], 1))
                        # print('fake', fake.shape)

                        fake_output = self.t_discriminators[i](concat([fake, sample], 1), training=True)
                        # print('fake_output_dis', fake_output.shape)

                        # print("watch", disc_tape.watch(self.teacher_disc[i].trainable_variables)
                        real_loss_disc = cross_entropy(ones_like(real_output), real_output)
                        fake_loss_disc = cross_entropy(zeros_like(fake_output), fake_output)

                        disc_loss = real_loss_disc + fake_loss_disc
                        # print(disc_loss, real_loss_disc, fake_loss_disc)

                        disc_grad = disc_tape.gradient(disc_loss, self.t_discriminators[i].trainable_variables)
                        # print(gradients_of_discriminator)

                        disc_opt_t[i].apply_gradients(zip(disc_grad, self.t_discriminators[i].trainable_variables))

            # train the student discriminator
            for t_3 in range(train_arguments.num_student_iters):
                z = uniform([self.batch_size, self.noise_dim], dtype=float64)

                sample = expand_dims(category_samples.sample(self.batch_size), axis=1)
                # print("category_stu", sample.shape)

                with GradientTape() as stu_tape:
                    fake = self.generator(concat([z, sample], 1))
                    # print('fake_stu', fake.shape)

                    predictions, clean_votes = self._pate_voting(
                        concat([fake, sample], 1), self.t_discriminators, train_arguments.lap_scale)
                    # print("noisy_labels", predictions.shape, "clean_votes", clean_votes.shape)
                    outputs = self.s_discriminator(concat([fake, sample], 1))

                    # update the moments
                    alpha = alpha + self._moments_acc(self.n_teachers, clean_votes, train_arguments.lap_scale, l_list)
                    # print("final_alpha", alpha)

                    stu_loss = cross_entropy(predictions, outputs)
                    gradients_of_stu = stu_tape.gradient(stu_loss, self.s_discriminator.trainable_variables)
                    # print(gradients_of_stu)

                    disc_opt_stu.apply_gradients(zip(gradients_of_stu, self.s_discriminator.trainable_variables))

            # train the generator
            z = uniform([self.batch_size, self.noise_dim], dtype=float64)

            sample_g = expand_dims(category_samples.sample(self.batch_size), axis=1)

            with GradientTape() as gen_tape:
                fake = self.generator(concat([z, sample_g], 1))
                output = self.s_discriminator(concat([fake, sample_g], 1))

                loss_gen = cross_entropy(ones_like(output), output)
                gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            # Calculate the current privacy cost
            epsilon = min((alpha - log(train_arguments.delta)) / l_list)
            if steps % train_arguments.sample_interval == 0:
                print("Step : ", steps, "Loss SD : ", stu_loss, "Loss G : ", loss_gen, "Epsilon : ", epsilon)

            steps += 1
        # self.generator.summary()

    def _pate_voting(self, data, netTD, lap_scale):
        # TODO: Validate the logic against original article
        ## Faz os votos dos teachers (1/0) netTD para cada record em data e guarda em results
        results = zeros([len(netTD), self.batch_size], dtype=int64)
        # print(results)
        for i in range(len(netTD)):
            output = netTD[i](data, training=True)
            pred = transpose(cast((output > 0.5), int64))
            # print(pred)
            results = tensor_scatter_nd_update(results, constant([[i]]), pred)
            # print(results)

        #guarda o somatorio das probabilidades atribuidas por cada disc a cada record (valores entre 0 e len(netTD))
        clean_votes = expand_dims(cast(reduce_sum(results, 0), dtype=float64), 1)
        # print("clean_votes",clean_votes)
        noise_sample = distributions.Laplace(loc=0, scale=1/lap_scale).sample(clean_votes.shape)
        # print("noise_sample", noise_sample)
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

    def build_model(self, input_shape, dim, data_dim, processor_info: Optional[NamedTuple] = None):
        input = Input(shape=input_shape, batch_size = self.batch_size)
        x = Dense(dim)(input)
        x = ReLU()(x)
        x = Dense(dim * 2)(x)
        x = Dense(data_dim)(x)
        if processor_info:
            x = ActivationInterface(processor_info, 'ActivationInterface')(x)
        return Model(inputs=input, outputs=x)
