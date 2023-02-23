"""
    DRAGAN model architecture implementation
"""
import os
from os import path

from typing import Optional, NamedTuple
import tensorflow as tf
import tqdm
from keras import Model, initializers
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

#Import ydata synthetic classes
from ....synthesizers.gan import BaseModel
from ....synthesizers.loss import Mode, gradient_penalty

class DRAGAN(BaseModel):

    __MODEL__='DRAGAN'

    def __init__(self, model_parameters, n_discriminator, gradient_penalty_weight=10):
        # As recommended in DRAGAN paper - https://arxiv.org/abs/1705.07215
        self.n_discriminator = n_discriminator
        self.gradient_penalty_weight = gradient_penalty_weight
        super().__init__(model_parameters)

    def define_gan(self, col_transform_info: Optional[NamedTuple] = None):
        # define generator/discriminator
        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim,
                        activation_info=col_transform_info, tau = self.tau)

        self.discriminator = Discriminator(self.batch_size). \
            build_model(input_shape=(self.data_dim,), dim=self.layers_dim)

        g_optimizer = Adam(self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)
        d_optimizer = Adam(self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2, clipvalue=0.001)
        return g_optimizer, d_optimizer

    def gradient_penalty(self, real, fake):
        gp = gradient_penalty(self.discriminator, real, fake, mode= Mode.DRAGAN)
        return gp

    def update_gradients(self, x, g_optimizer, d_optimizer):
        """
        Compute the gradients for both the Generator and the Discriminator
            x (tf.tensor): real data event
            *_optimizer (tf.OptimizerV2): Optimizer for the * model
        :return: generator gradients, discriminator gradients
        """
        # Update the gradients of critic for n_critic times (Training the critic)
        for _ in range(self.n_discriminator):
            with tf.GradientTape() as d_tape:
                d_loss = self.d_lossfn(x)
            # Get the gradients of the critic
            d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the critic using the optimizer
            d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Update the generator
        with tf.GradientTape() as g_tape:
            gen_loss = self.g_lossfn(x)

        # Get the gradients of the generator
        gen_gradients = g_tape.gradient(gen_loss, self.generator.trainable_variables)

        # Update the weights of the generator
        g_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return d_loss, gen_loss

    def d_lossfn(self, real):
        """
        Calculates the critic losses
        """
        noise = tf.random.normal((self.batch_size, self.noise_dim), dtype=tf.dtypes.float64)
        # run noise through generator
        fake = self.generator(noise)
        # discriminate x and x_gen
        logits_real = self.discriminator(real, training=True)
        logits_fake = self.discriminator(fake, training=True)

        # gradient penalty
        gp = self.gradient_penalty(real, fake)

        # getting the loss of the discriminator.
        d_loss = (tf.reduce_mean(logits_fake)
                  - tf.reduce_mean(logits_real)
                  + gp * self.gradient_penalty_weight)
        return d_loss

    # generator loss
    def g_lossfn(self, real):
        """
        Calculates the Generator losses
        :param real: Data batch we are analyzing
        :return: Loss of the generator
        """
        # generating noise from a uniform distribution
        noise = tf.random.normal((real.shape[0], self.noise_dim), dtype=tf.float64)

        fake = self.generator(noise, training=True)
        logits_fake = self.discriminator(fake, training=True)
        g_loss = -tf.reduce_mean(logits_fake)
        return g_loss

    def get_data_batch(self, train, batch_size):
        buffer_size = len(train)
        #tensor_data = pd.concat([x_train, y_train], axis=1)
        train_loader = tf.data.Dataset.from_tensor_slices(train) \
            .batch(batch_size).shuffle(buffer_size)
        return train_loader

    def train_step(self, train_data, optimizers):
        d_loss, g_loss = self.update_gradients(train_data, *optimizers)
        return d_loss, g_loss

    def fit(self, data, train_arguments, num_cols, cat_cols):
        """
        Args:
            data: A pandas DataFrame or a Numpy array with the data to be synthesized
            train_arguments: GAN training arguments.
            num_cols: List of columns of the data object to be handled as numerical
            cat_cols: List of columns of the data object to be handled as categorical
        """
        super().fit(data, num_cols, cat_cols)

        processed_data = self.processor.transform(data)
        self.data_dim = processed_data.shape[1]
        optimizers = self.define_gan(self.processor.col_transform_info)

        train_loader = self.get_data_batch(processed_data, self.batch_size)

        # Create a summary file
        train_summary_writer = tf.summary.create_file_writer(path.join('..\dragan_test', 'summaries', 'train'))

        with train_summary_writer.as_default():
            for epoch in tqdm.trange(train_arguments.epochs):
                for batch_data in train_loader:
                    batch_data = tf.cast(batch_data, dtype=tf.float32)
                    d_loss, g_loss = self.train_step(batch_data, optimizers)

                print(
                    "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                        epoch, d_loss, g_loss
                    ))

                if epoch % train_arguments.sample_interval == 0:
                    # Test here data generation step
                    # save model checkpoints
                    if path.exists('./cache') is False:
                        os.mkdir('./cache')
                    model_checkpoint_base_name = './cache/' + train_arguments.cache_prefix + '_{}_model_weights_step_{}.h5'
                    self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                    self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))


class Discriminator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.5), activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)

class Generator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim, activation_info: NamedTuple = None, tau: Optional[float] = None):
        input = Input(shape=input_shape, batch_size = self.batch_size)
        x = Dense(dim, kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.5), activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        #if activation_info:
        #    x = GumbelSoftmaxActivation(activation_info, tau=tau)(x)
        return Model(inputs=input, outputs=x)
