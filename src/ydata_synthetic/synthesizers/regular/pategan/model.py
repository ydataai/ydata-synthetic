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
    def train(self, data, train_arguments: TrainParameters, num_cols: List[str], cat_cols: List[str]):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """

        super().train(data, num_cols, cat_cols)

        data = self.processor.transform(data)
        self.data_dim = data.shape[1]
        self.define_gan(self.processor.col_transform_info)

        alpha = cast([0.0 for _ in range(train_arguments.num_moments)], float64)
        l_list = 1 + cast(range(train_arguments.num_moments), float64)

        cross_entropy = BinaryCrossentropy(from_logits=True)

        gen_opt = Adam(learning_rate=train_arguments.lr)
        disc_opt_stu = Adam(learning_rate=train_arguments.lr)
        disc_opt_t = [Adam(learning_rate=train_arguments.lr) for i in range(self.n_teachers)]

        train_loader = self.get_data_loader(data, self.batch_size)

        steps = 0
        epsilon = 0

        while epsilon < self.target_epsilon:
            # train the teacher descriminator
            for t_2 in range(train_arguments.num_teacher_iters):
                for i, teacher_loader in zip(range(self.n_teachers), train_loader):
                    with GradientTape() as disc_tape:
                        # train with real
                        real_output = self.t_discriminators[i](next(teacher_loader), training=True)

                        # train with fake
                        z = uniform([self.batch_size, self.noise_dim], dtype=float64)

                        fake = self.generator(z)

                        fake_output = self.t_discriminators[i](fake, training=True)

                        real_loss_disc = cross_entropy(ones_like(real_output), real_output)
                        fake_loss_disc = cross_entropy(zeros_like(fake_output), fake_output)

                        disc_loss = real_loss_disc + fake_loss_disc

                        disc_grad = disc_tape.gradient(disc_loss, self.t_discriminators[i].trainable_variables)

                        disc_opt_t[i].apply_gradients(zip(disc_grad, self.t_discriminators[i].trainable_variables))

            # train the student discriminator
            for t_3 in range(train_arguments.num_student_iters):
                z = uniform([self.batch_size, self.noise_dim], dtype=float64)

                with GradientTape() as stu_tape:
                    fake = self.generator(z)

                    predictions, clean_votes = self._pate_voting(fake, self.t_discriminators, train_arguments.lap_scale)
                    outputs = self.s_discriminator(fake)

                    # update the moments
                    alpha = alpha + self._moments_acc(self.n_teachers, clean_votes, train_arguments.lap_scale, l_list)

                    stu_loss = cross_entropy(predictions, outputs)
                    stu_grad = stu_tape.gradient(stu_loss, self.s_discriminator.trainable_variables)

                    disc_opt_stu.apply_gradients(zip(stu_grad, self.s_discriminator.trainable_variables))

            # train the generator
            z = uniform([self.batch_size, self.noise_dim], dtype=float64)

            with GradientTape() as gen_tape:
                fake = self.generator(z)
                output = self.s_discriminator(fake)

                loss_gen = cross_entropy(ones_like(output), output)  # ones_like to mimic real outputs
                gen_grad = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
                gen_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

            # Calculate the current privacy cost
            epsilon = min((alpha - log(train_arguments.delta)) / l_list)
            if steps % train_arguments.sample_interval == 0:
                print("Step : ", steps, "Loss SD : ", stu_loss, "Loss G : ", loss_gen, "Epsilon : ", epsilon)

            steps += 1
