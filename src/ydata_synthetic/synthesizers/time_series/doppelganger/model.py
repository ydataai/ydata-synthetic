import functools
import os
import numpy as np
from tqdm import trange

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from metadata import Metadata, Variable, OutputType, Activation

from ydata_synthetic.synthesizers.time_series.doppelganger import Generator
from ydata_synthetic.synthesizers.time_series.doppelganger import Discriminator, AttrDiscriminator
from ydata_synthetic.synthesizers.gan import Model
from src.synthesizers.utils import checkpoint, loss


class Doppelganger(Model):

    def __init__(self, train_metadata, sample_len,
                 batch_size=100, epochs=100, noise_dim=5,
                 pack=10, d_rounds=1, g_rounds=1, checkpoint_dir='./',
                 checkpoint_epoch=10, output_dir="./"):
        self.metadata = train_metadata
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.pack = pack
        self.noise_dim = noise_dim
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.checkpoint_epoch = checkpoint_epoch
        self.output_dir = output_dir
        self.timeseries = True
        self.real_features = None
        self.real_attributes = None
        self.period = None
        self.time = None
        self.generator = None
        self.discriminator = None
        self.attr_discriminator = None
        self.attributes = False
        self.eps = 1e-8
        self.output_dir = output_dir

    def _traindata_validation(self, data):
        if isinstance(data, (np.ndarray, np.generic)):
            assert len(data.shape) == 3,\
                "Features training array must be 3-dimensional."
            assert self.batch_size < data.shape[0],\
                "Batch size must be smaller than the number of events provided."
            assert data.shape[1] % self.sample_len != 0,\
                "Provided period must be a multiple of the provided sample len."
            self.period = data.shape[1]
        elif isinstance(data, list):
            self.attributes = True
            assert len(data) == 2,\
                "Training data must have Features and Attributes provided."
            assert len(data[0].shape) == 3,\
                "Features must be provided as a 3-dimensional array."
            assert len(data[1].shape) == 2,\
                "Attributes must be provided as a 2-dimensional array."
            assert data[0].shape[0] == data[1].shape[0],\
                "The same number of events must be provided in Features and Attributes."
            assert data[0].shape[0],\
                "Batch size must be smaller than the number of events provided."
            assert data[0].shape[1] % self.sample_len == 0,\
                "Provided sample lenght must be a multiple of train features period."
            self.period = data[0].shape[1]
        else:
            raise ValueError('The provided train data does not have a valid data type.'
                             ' Please provide a list of np.ndarray or a np.array.')

        self.time = int(self.period/self.sample_len)

    @tf.function
    def generator_train(self, g_loss_fn, g_optim):
        """
        Parameters
        ----------
        g_loss_fn Generator loss function
        g_optim Generator optimizer funtion
        Returns A dictionary with the generator calc loss
        -------
        """
        with tf.GradientTape() as t:
            features_z = tf.random.normal(shape=(self.batch_size, self.time, self.noise_dim))
            attr_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
            z = [features_z, attr_z]
            if self.attributes:
                attr_add_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
                z.append(attr_add_z)

            features_fake, attributes_fake = self.generator(z, training=True)

            d_fake = self.discriminator([features_fake, attributes_fake], training=True)
            g_loss = g_loss_fn(d_fake)

            if self.attributes:
                d_attr_fake = self.attr_discriminator(attributes_fake, training=True)
                g_add_attr_loss = g_loss_fn(d_attr_fake)
                g_loss += g_add_attr_loss

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        g_optim.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return {'g_loss': g_loss}

    @tf.function
    def discriminator_train(self, features_real, attributes_real, d_loss_fn, d_optim, d_coe):
        with tf.GradientTape() as t:
            features_z = tf.random.normal(shape=(self.batch_size, self.time, self.noise_dim))
            z = [features_z]
            if self.attributes:
                attr_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
                attr_add_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
                z.append([attr_z, attr_add_z])

            features_fake, attributes_fake = self.generator(z, training=True)

            output_real_d_logit = \
                self.discriminator([features_real, attributes_real], training=True)
            output_fake_d_logit = \
                self.discriminator([features_fake, attributes_fake], training=True)

            x_real_d_loss, x_fake_d_loss = d_loss_fn(output_real_d_logit, output_fake_d_logit)

            gp = loss.gradient_penalty(functools.partial(self.discriminator, training=True),
                                       [features_real, attributes_real],
                                       [features_fake, attributes_fake],
                                       mode='wgan-TS')

            d_loss = (x_real_d_loss + x_fake_d_loss) + gp * d_coe

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        d_optim.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    @tf.function
    def discriminator_aux_train(self, attributes_real, d_aux_loss_fn, d_optim, d_coe):
        with tf.GradientTape() as t:
            features_z = tf.random.normal(shape=(self.batch_size, self.time, self.noise_dim))
            attr_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
            z = [features_z, attr_z]
            if self.attributes:
                attr_add_z = tf.random.normal(shape=(self.batch_size, self.noise_dim))
                z.append(attr_add_z)

            features_fake, attributes_fake = self.generator(z, training=True)

            output_real_d_logit = self.attr_discriminator(attributes_real, training=True)
            output_fake_d_logit = self.attr_discriminator(attributes_fake, training=True)

            x_real_d_loss, x_fake_d_loss = d_aux_loss_fn(output_real_d_logit, output_fake_d_logit)

            gp = loss.gradient_penalty(functools.partial(self.attr_discriminator, training=True),
                                       attributes_real, attributes_fake,
                                       mode='wgan-gp')

            d_loss = (x_real_d_loss + x_fake_d_loss) + gp * d_coe

        d_grad = t.gradient(d_loss, self.attr_discriminator.trainable_variables)
        d_optim.apply_gradients(zip(d_grad, self.attr_discriminator.trainable_variables))

        return {'d_aux_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    def _build_losses(self):
        self.generator = Generator(train_metadata=self.metadata,
                                   noise_dim=self.noise_dim,
                                   sample_len=self.sample_len,
                                   period=self.period).build_model()

        self.discriminator = Discriminator().build_model(feature_shape=(self.period,
                                                                        self.metadata.getfeaturesoutdim()),
                                                         attr_shape=(self.metadata.getallattrdim()),
                                                         batch_size=self.batch_size)

        #Generator inputs
        input_attr = Input(shape=(self.noise_dim), batch_size=self.batch_size)
        input_features = Input(shape=(self.time, self.noise_dim),
                               batch_size=self.batch_size)

        g_input = [input_features, input_attr]

        losses_fn = None
        if self.attributes:
            self.attr_discriminator = AttrDiscriminator().build_model(
                input_shape=(self.metadata.getallattrdim()), batch_size=self.batch_size)
            input_add_attr = Input(shape=(self.noise_dim), batch_size=self.batch_size)
            g_input.append(input_add_attr)
            losses_fn = loss.get_adversarial_losses_fn('wgan', self.attributes)
        else:
            losses_fn = loss.get_adversarial_losses_fn('wgan')

        return losses_fn

    def fit(self, train_data, real_attribute_mask, g_lr=0.0002, d_lr=0.004,
            attr_d_lr=0.001, d_beta1=0.5, attr_d_beta1=0.5, g_beta1=0.5,
            d_rounds=1, g_rounds=1, d_gp_coe=0.2, attr_d_gp_coe=1.0):

        self._traindata_validation(train_data)

        self.real_features = train_data[0]
        self.real_attributes = train_data[1]

        #Init the optimizers
        g_optimizer = Adam(learning_rate=g_lr, beta_1=g_beta1)
        d_optimizer = Adam(learning_rate=d_lr, beta_1=d_beta1)
        if self.attributes:
            d_attr_optimizer = Adam(learning_rate=attr_d_lr, beta_1=attr_d_beta1)
            g_loss_fn, d_loss_fn, d_aux_loss_fn = self._build_losses()
        else:
            g_loss_fn, d_loss_fn = self._build_losses()

        #Define here the GAN training
        train_summary_writer = tf.summary.create_file_writer(os.path.join(self.output_dir, 'summaries', 'train'))

        dataset_len = self.real_features.shape[0]
        epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        batch_num = dataset_len // self.batch_size

        #Init the checkpoints object
        check = checkpoint.Checkpoint(dict(G=self.generator,
                                           D=self.discriminator,
                                           G_optimizer=g_optimizer,
                                           D_optimizer=d_optimizer,
                                           ep_cnt=epoch_counter),
                                      os.path.join(self.output_dir, 'checkpoints'),
                                      max_to_keep=5)
        try:  # restore checkpoint including the epoch counter
            check.restore().assert_existing_objects_matched()
        except Exception as e:
            print(e)

        with train_summary_writer.as_default():
            for epoch in trange(self.epochs, desc='Epochs Loop'):
                if epoch < epoch_counter:
                    continue
                # Update epoch counter
                epoch_counter.assign_add(1)

                #get the data id's
                data_id = np.random.choice(self.real_features.shape[0],
                                           size=(self.real_features.shape[0],
                                                 self.pack))
                # Iterate over the real dataset

                for batch in range(batch_num):
                    #Implement packing here
                    for i in range(self.pack):
                        batch_data_id = data_id[batch * self.batch_size:
                                                (batch + 1) * self.batch_size,i]

                    batch_data_feature = self.real_features[batch_data_id]
                    batch_data_attribute = self.real_attributes[batch_data_id]

                if g_optimizer.iterations.numpy() % self.g_rounds == 0:
                    d_loss_dict = self.discriminator_train(features_real=batch_data_feature,
                                                           attributes_real=batch_data_attribute,
                                                           d_loss_fn=d_loss_fn,
                                                           d_optim=d_optimizer,
                                                           d_coe=d_gp_coe)
                    self.summary(d_loss_dict, step=d_optimizer.iterations, name='d_losses')

                    if self.attributes:
                        d_attr_loss_dict = self.discriminator_aux_train(attributes_real=batch_data_attribute,
                                                                        d_aux_loss_fn=d_aux_loss_fn,
                                                                        d_optim=d_attr_optimizer,
                                                                        d_coe=attr_d_gp_coe)
                        self.summary(d_attr_loss_dict, step=d_attr_optimizer.iterations, name='d_attr_losses')

                if d_optimizer.iterations.numpy() % self.d_rounds == 0:
                    g_loss_dict = self.generator_train(g_loss_fn, g_optimizer)
                    self.summary(g_loss_dict, step=g_optimizer.iterations, name='g_losses')

                if epoch % self.checkpoint_epoch == 0:
                    #Model checkpoint
                    check.save(epoch)


if __name__ == '__main__':

    print('Start testing.')

    # Create here a metadata example to test with the EDP dataset
    def load_data(path):
        data_npz = np.load(path)
        return data_npz["data_feature"], data_npz["data_attribute"], data_npz["real_attribute_mask"]

    def create_test_metadata():
        num_features = 4
        num_attr = [2, 12]
        num_add_attr = [1, 1, 1, 1, 1, 1]
        features = []
        for i in range(num_features):
            is_gen_flag = False
            size = 1
            if i >= num_features - 1:
                is_gen_flag = True
                size = 2

            var = Variable(name='feat_{}'.format(i), var_type=OutputType.CONTINUOUS,
                           size=size, activation=Activation.TANH, is_gen_flag=is_gen_flag)
            features.append(var)

        attributes = []
        for i in range(len(num_attr)):
            var = Variable(name='attr_{}'.format(i),
                           var_type=OutputType.CATEGORICAL, size=num_attr[i],
                           activation=Activation.SOFTMAX, is_gen_flag=False)
            attributes.append(var)

        add_attributes = []
        for j in range(len(num_add_attr)):
            var = Variable(name='add_attr_{}'.format(j), var_type=OutputType.CATEGORICAL, size=num_add_attr[j],
                           activation=Activation.SIGMOID, is_gen_flag=False)
            add_attributes.append(var)
        return Metadata(features, attributes=attributes, add_attributes=add_attributes)

    data_feature, data_attribute, mask_data = load_data("/home/fabiana/Documents/Demos/YData/data/data_train.npz")
    metadata = create_test_metadata()

    metadata.addnewvar(Variable(name='add_attr_10}', var_type=OutputType.CATEGORICAL, size=20,
                                activation=Activation.SIGMOID, is_gen_flag=False))

    train_data = [data_feature, data_attribute]

    print('Data loaded and metadata created.')

    print('Start the generator definition')

    train_data = [data_feature, data_attribute]

    synthesizer = Timesynthesizer(sample_len=24, train_metadata=metadata)
    synthesizer.fit(train_data, real_attribute_mask=mask_data)

    print('Synthetic data generation')

    synth_feature, synth_attr = synthesizer.sample(400)

    with open('results/data_gen.npz', 'wb') as f:
        np.savez(f, features=synth_feature, attributes=synth_attr)

    print('generated_data')
    # Create here the metadata for the EDP use case to be tested with the new Doppelganger

