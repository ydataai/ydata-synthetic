from functools import partial
from joblib import dump
import numpy as np
from pandas import DataFrame
import tensorflow as tf
from keras.layers import \
    (Input, Dense, LeakyReLU, Dropout, BatchNormalization, ReLU, Concatenate)
from keras import Model

import tensorflow_probability as tfp
from ydata_synthetic.synthesizers.regular.ctgan.utils \
    import ConditionalLoss, RealDataSampler, ConditionalSampler
 
from ydata_synthetic.synthesizers.loss import gradient_penalty, Mode as ModeGP
from ydata_synthetic.synthesizers.gan import BaseModel, ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.regular.ctgan_processor import CTGANDataProcessor

class CTGAN(BaseModel):
    """
    Conditional Tabular GAN model.
    Based on the paper https://arxiv.org/abs/1907.00503.

    Args:
        model_parameters: Parameters used to create the CTGAN model.
    """
    __MODEL__ = 'CTGAN'

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)
        if self.batch_size % 2 != 0 or self.batch_size % self.pac != 0:
            raise ValueError("The batch size needs to be an even value divisible by the PAC.")
        self._model_parameters = model_parameters
        self._real_data_sampler = None
        self._conditional_sampler = None
        self._generator_model = None
        self._critic_model = None

    @staticmethod
    def _create_generator_model(input_dim, generator_dims, data_dim, metadata, tau):
        """
        Creates the generator model.

        Args:
            input_dim: Input dimensionality.
            generator_dims: Dimensions of each hidden layer.
            data_dim: Output dimensionality.
            metadata: Dataset columns metadata.
            tau: Gumbel-Softmax non-negative temperature.
        """
        input = Input(shape=(input_dim, ))
        x = input
        dim = input_dim
        for layer_dim in generator_dims:
            layer_input = x
            x = Dense(layer_dim,
                      kernel_initializer="random_uniform",
                      bias_initializer="random_uniform")(x)
            x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
            x = ReLU()(x)
            x = Concatenate(axis=1)([x, layer_input])
            dim += layer_dim

        def _gumbel_softmax(logits, tau=1.0):
            """Applies the Gumbel-Softmax function to the given logits."""
            gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
            gumbels = gumbel_dist.sample(tf.shape(logits))
            gumbels = (logits + gumbels) / tau
            return tf.nn.softmax(gumbels, -1)

        def _generator_activation(data):
            """Custom activation function for the generator model."""
            data_transformed = []
            for col_md in metadata:
                if col_md.discrete:
                    logits = data[:, col_md.start_idx:col_md.end_idx]
                    data_transformed.append(_gumbel_softmax(logits, tau=tau))
                else:
                    data_transformed.append(tf.math.tanh(data[:, col_md.start_idx:col_md.start_idx+1]))
                    logits = data[:, col_md.start_idx+1:col_md.end_idx]
                    data_transformed.append(_gumbel_softmax(logits, tau=tau))
            return data, tf.concat(data_transformed, axis=1)
            
        x = Dense(data_dim, kernel_initializer="random_uniform",
                  bias_initializer="random_uniform", 
                  activation=_generator_activation)(x)
        return Model(inputs=input, outputs=x)
    
    @staticmethod
    def _create_critic_model(input_dim, critic_dims, pac):
        """
        Creates the critic model.

        Args:
            input_dim: Input dimensionality.
            critic_dims: Dimensions of each hidden layer.
            pac: PAC size.
        """
        input = Input(shape=(input_dim,))
        x = tf.reshape(input, [-1, input_dim * pac])
        for dim in critic_dims:
            x = Dense(dim,
                      kernel_initializer="random_uniform",
                      bias_initializer="random_uniform")(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.5)(x)
        x = Dense(1, kernel_initializer="random_uniform",
                  bias_initializer="random_uniform")(x)
        return Model(inputs=input, outputs=x)

    def fit(self, data: DataFrame, train_arguments: TrainParameters, num_cols: list[str], cat_cols: list[str]):
        """
        Fits the CTGAN model.

        Args:
            data: A pandas DataFrame with the data to be synthesized.
            train_arguments: CTGAN training arguments.
            num_cols: List of columns to be handled as numerical
            cat_cols: List of columns to be handled as categorical
        """
        super().fit(data=data, num_cols=num_cols, cat_cols=cat_cols, train_arguments=train_arguments)

        self._generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self._critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        train_data = self.processor.transform(data)
        metadata = self.processor.metadata
        data_dim = self.processor.output_dimensions

        self._real_data_sampler = RealDataSampler(train_data, metadata)
        self._conditional_sampler = ConditionalSampler(train_data, metadata, train_arguments.log_frequency)
        
        gen_input_dim = self.latent_dim + self._conditional_sampler.output_dimensions
        self._generator_model = self._create_generator_model(
            gen_input_dim, self.generator_dims, data_dim, metadata, self.tau)
        
        crt_input_dim = data_dim + self._conditional_sampler.output_dimensions
        self._critic_model = self._create_critic_model(crt_input_dim, self.critic_dims, self.pac)

        self._generator_model.build((self.batch_size, gen_input_dim))
        self._critic_model.build((self.batch_size, crt_input_dim))

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for epoch in range(train_arguments.epochs):
            for _ in range(steps_per_epoch):
                fake_z = tf.random.normal([self.batch_size, self.latent_dim])
                cond_vector = self._conditional_sampler.sample(self.batch_size)
                if cond_vector is None:
                    real = self._real_data_sampler.sample(self.batch_size)
                else:
                    cond, _, col_idx, opt_idx = cond_vector
                    cond = tf.convert_to_tensor(cond)
                    fake_z = tf.concat([fake_z, cond], 1)
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = self._real_data_sampler.sample_col(col_idx[perm], opt_idx[perm])
                    cond_perm = tf.gather(cond, perm)

                fake, fake_act = self._generator_model(fake_z, training=True)
                real = tf.convert_to_tensor(real.astype('float32'))
                real_cat = real if cond_vector is None else tf.concat([real, cond_perm], 1)
                fake_cat = fake if cond_vector is None else tf.concat([fake_act, cond], 1)
                critic_loss = self._train_critic_step(real_cat, fake_cat)

                fake_z = tf.random.normal([self.batch_size, self.latent_dim])
                cond_vector = self._conditional_sampler.sample(self.batch_size)
                if cond_vector is None:
                    generator_loss = self._train_generator_step(fake_z)
                else:
                    cond, mask, _, _ = cond_vector
                    cond = tf.convert_to_tensor(cond)
                    mask = tf.convert_to_tensor(mask)
                    fake_z = tf.concat([fake_z, cond], axis=1)
                    generator_loss = self._train_generator_step(fake_z, cond, mask, metadata)
            
            print(f"Epoch: {epoch} | critic_loss: {critic_loss} | generator_loss: {generator_loss}")

    def _train_critic_step(self, real, fake):
        """
        Single training iteration of the critic model.

        Args:
            real: Real data.
            fake: Fake data.
        """
        with tf.GradientTape() as tape:
            y_real = self._critic_model(real, training=True)
            y_fake = self._critic_model(fake, training=True)
            gp = gradient_penalty(
                partial(self._critic_model, training=True), real, fake, ModeGP.CTGAN, self.pac)
            rec_loss = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake))
            critic_loss = rec_loss + gp * self.gp_lambda
        gradient = tape.gradient(critic_loss, self._critic_model.trainable_variables)
        self._apply_critic_gradients(gradient, self._critic_model.trainable_variables)
        return critic_loss

    @tf.function
    def _apply_critic_gradients(self, gradient, trainable_variables):
        """
        Updates gradients of the critic model.
        This logic is isolated in order to be optimized as a TF function.

        Args:
            gradient: Gradient.
            trainable_variables: Variables to be updated.
        """
        self._critic_optimizer.apply_gradients(zip(gradient, trainable_variables))

    def _train_generator_step(self, fake_z, cond_vector=None, mask=None, metadata=None):
        """
        Single training iteration of the generator model.

        Args:
            real: Real data.
            fake: Fake data.
            cond_vector: Conditional vector.
            mask: Mask vector.
            metadata: Dataset columns metadata.
        """
        with tf.GradientTape() as tape:
            fake, fake_act = self._generator_model(fake_z, training=True)
            if cond_vector is not None:
                y_fake = self._critic_model(
                    tf.concat([fake_act, cond_vector], 1), training=True)
                cond_loss = ConditionalLoss.compute(fake, cond_vector, mask, metadata)
                generator_loss = -tf.reduce_mean(y_fake) + cond_loss
            else:
                y_fake = self._critic_model(fake_act, training=True)
                generator_loss = -tf.reduce_mean(y_fake)
        gradient = tape.gradient(generator_loss, self._generator_model.trainable_variables)
        gradient = [gradient[i] + self.l2_scale * self._generator_model.trainable_variables[i] for i in range(len(gradient))]
        self._apply_generator_gradients(gradient, self._generator_model.trainable_variables)
        return generator_loss

    @tf.function
    def _apply_generator_gradients(self, gradient, trainable_variables):
        """
        Updates gradients of the generator model.
        This logic is isolated in order to be optimized as a TF function.

        Args:
            gradient: Gradient.
            trainable_variables: Variables to be updated.
        """
        self._generator_optimizer.apply_gradients(zip(gradient, trainable_variables))

    def sample(self, n_samples: int):
        """
        Samples new data from the CTGAN.

        Args:
            n_samples: Number of samples to be generated.
        """
        if n_samples <= 0:
            raise ValueError("Invalid number of samples.")

        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tf.range(steps):
            fake_z = tf.random.normal([self.batch_size, self.latent_dim])
            cond_vec = self._conditional_sampler.sample(self.batch_size, from_active_bits=True)
            if cond_vec is not None:
                cond = tf.constant(cond_vec)
                fake_z = tf.concat([fake_z, cond], 1)

            fake = self._generator_model(fake_z)[1]
            data.append(fake.numpy())

        data = np.concatenate(data, 0)
        data = data[:n_samples]
        return self.processor.inverse_transform(data)
    
    def save(self, path):
        """
        Save the CTGAN model in a pickle file.
        Only the required components to sample new data are saved.

        Args:
            path: Path of the pickle file.
        """
        dump({
            "model_parameters": self._model_parameters,
            "data_dim": self.processor.output_dimensions,
            "gen_input_dim": self.latent_dim + self._conditional_sampler.output_dimensions,
            "generator_dims": self.generator_dims,
            "tau": self.tau,
            "metadata": self.processor.metadata,
            "batch_size": self.batch_size,
            "latent_dim": self.latent_dim,
            "conditional_sampler": self._conditional_sampler.__dict__,
            "generator_model_weights": self._generator_model.get_weights(),
            "processor": self.processor.__dict__
        }, path)

    @staticmethod
    def load(class_dict):
        """
        Load the CTGAN model from a pickle file.
        Only the required components to sample new data are loaded.

        Args:
            class_dict: Class dict loaded from the pickle file.
        """
        new_instance = CTGAN(class_dict["model_parameters"])
        setattr(new_instance, "generator_dims", class_dict["generator_dims"])
        setattr(new_instance, "tau", class_dict["tau"])
        setattr(new_instance, "batch_size", class_dict["batch_size"])
        setattr(new_instance, "latent_dim", class_dict["latent_dim"])

        new_instance._conditional_sampler = ConditionalSampler()
        new_instance._conditional_sampler.__dict__ = class_dict["conditional_sampler"]
        new_instance.processor = CTGANDataProcessor()
        new_instance.processor.__dict__ = class_dict["processor"]

        new_instance._generator_model = new_instance._create_generator_model(
            class_dict["gen_input_dim"], class_dict["generator_dims"], 
            class_dict["data_dim"], class_dict["metadata"], class_dict["tau"])
        
        new_instance._generator_model.build((class_dict["batch_size"], class_dict["gen_input_dim"]))
        new_instance._generator_model.set_weights(class_dict['generator_model_weights'])
        return new_instance