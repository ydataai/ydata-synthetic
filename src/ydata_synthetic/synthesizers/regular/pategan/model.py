from math import log
from typing import List

from tensorflow import GradientTape, concat, expand_dims, ones_like, zeros, zeros_like
from tensorflow.dtypes import cast, float64
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.random import uniform
from tensorflow_probability import distributions

from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.gan import BaseModel


class PATEGAN(BaseModel):

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
