import tensorflow as tf
import numpy as np
from tqdm import tqdm
import math
from joblib import dump


class DoppelGANgerNetwork(object):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/doppelganger.py.
    """
    def __init__(self,
                 sess,
                 epoch,
                 batch_size,
                 data_feature,
                 data_attribute,
                 attribute_cols_metadata,
                 sample_len,
                 generator,
                 discriminator,
                 rounds,
                 d_gp_coe,
                 num_packing,
                 attr_discriminator=None,
                 attr_d_gp_coe=None,
                 g_attr_d_coe=None,
                 attribute_latent_dim=5,
                 feature_latent_dim=5,
                 fix_feature_network=False,
                 g_lr=0.001,
                 g_beta1=0.5,
                 d_lr=0.001,
                 d_beta1=0.5,
                 attr_d_lr=0.001,
                 attr_d_beta1=0.5):
        """Constructor of DoppelGANger
        Args:
            sess: A tensorflow session
            epoch: Number of training epochs
            batch_size: Training batch size
            data_feature: Training features, in numpy float32 array format.
                The size is [(number of training samples) x (maximum length) x
                (total dimension of features)].
            data_attribute: Training attributes, in numpy float32 array format.
                The size is [(number of training samples) x (total dimension 
                of attributes)]
            sample_len: The time series batch size
            generator: An instance of network.DoppelGANgerGenerator
            discriminator: An instance of network.Discriminator
            rounds: Number of steps per batch
            d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
            num_packing: Packing degree in PacGAN (a method for solving mode
                collapse in NeurIPS 2018, see https://arxiv.org/abs/1712.04086)
            attr_discriminator: An instance of network.AttrDiscriminator. None
                if you do not want to use this auxiliary discriminator
            attr_d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
                for the auxiliary discriminator
            g_attr_d_coe: Weight of the auxiliary discriminator in the
                generator's loss
            attribute_latent_dim: The dimension of noise for generating 
                attributes
            feature_latent_dim: The dimension of noise for generating 
                features
            fix_feature_network: Whether to fix the feature network during 
                training
            g_lr: The learning rate in Adam for training the generator
            g_beta1: The beta1 in Adam for training the generator 
            d_lr: The learning rate in Adam for training the discriminator
            d_beta1: The beta1 in Adam for training the discriminator 
            attr_d_lr: The learning rate in Adam for training the auxiliary
                discriminator
            attr_d_beta1: The beta1 in Adam for training the auxiliary
                discriminator
        """
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.attribute_cols_metadata = attribute_cols_metadata
        self.sample_len = sample_len
        self.generator = generator
        self.discriminator = discriminator
        self.rounds = rounds
        self.attr_discriminator = attr_discriminator
        self.d_gp_coe = d_gp_coe
        self.attr_d_gp_coe = attr_d_gp_coe
        self.g_attr_d_coe = g_attr_d_coe
        self.num_packing = num_packing
        self.attribute_latent_dim = attribute_latent_dim
        self.feature_latent_dim = feature_latent_dim
        self.fix_feature_network = fix_feature_network
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.attr_d_lr = attr_d_lr
        self.attr_d_beta1 = attr_d_beta1

        if self.data_feature is not None:
            if self.data_feature.shape[1] % self.sample_len != 0:
                raise Exception("Length must be a multiple of sample_len.")
            self.sample_time = int(self.data_feature.shape[1] / self.sample_len)
            self.sample_feature_dim = self.data_feature.shape[2]
        if self.data_attribute is not None:
            self.sample_attribute_dim = self.data_attribute.shape[1]
            self.sample_real_attribute_dim = sum([c.output_dim for c in self.attribute_cols_metadata if c.real])

        self.EPS = 1e-8

    def build(self):
        self.build_connection()
        self.build_loss()

    def build_connection(self):
        # build connections for train-fake
        self.g_feature_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_feature_input_noise_train_pl_l.append(
                tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.sample_time, self.feature_latent_dim],
                    name="g_feature_input_noise_train_{}".format(i)))
        self.g_real_attribute_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_real_attribute_input_noise_train_pl_l.append(
                tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.attribute_latent_dim],
                    name="g_real_attribute_input_noise_train_{}".format(i)))
        self.g_addi_attribute_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_addi_attribute_input_noise_train_pl_l.append(
                tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.attribute_latent_dim],
                    name=("g_addi_attribute_input_noise_train_{}".format(i))))
        self.g_feature_input_data_train_pl_l = []
        for i in range(self.num_packing):
            self.g_feature_input_data_train_pl_l.append(
                tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.sample_len * self.sample_feature_dim],
                    name="g_feature_input_data_train_{}".format(i)))

        batch_size = tf.shape(input=self.g_feature_input_noise_train_pl_l[0])[0]
        self.real_attribute_mask_tensor = []
        for col_meta in self.attribute_cols_metadata:
            if col_meta.real:
                sub_mask_tensor = tf.ones((batch_size, col_meta.output_dim))
            else:
                sub_mask_tensor = tf.zeros((batch_size, col_meta.output_dim))
            self.real_attribute_mask_tensor.append(sub_mask_tensor)
        self.real_attribute_mask_tensor = tf.concat(self.real_attribute_mask_tensor,axis=1)

        self.g_output_feature_train_tf_l = []
        self.g_output_attribute_train_tf_l = []
        self.g_output_gen_flag_train_tf_l = []
        self.g_output_length_train_tf_l = []
        self.g_output_argmax_train_tf_l = []
        for i in range(self.num_packing):
            (g_output_feature_train_tf, g_output_attribute_train_tf,
             g_output_gen_flag_train_tf, g_output_length_train_tf,
             g_output_argmax_train_tf) = \
                self.generator.build(
                    self.g_real_attribute_input_noise_train_pl_l[i],
                    self.g_addi_attribute_input_noise_train_pl_l[i],
                    self.g_feature_input_noise_train_pl_l[i],
                    self.g_feature_input_data_train_pl_l[i],
                    train=True)

            if self.fix_feature_network:
                g_output_feature_train_tf = tf.zeros_like(
                    g_output_feature_train_tf)
                g_output_gen_flag_train_tf = tf.zeros_like(
                    g_output_gen_flag_train_tf)
                g_output_attribute_train_tf *= self.real_attribute_mask_tensor

            self.g_output_feature_train_tf_l.append(
                g_output_feature_train_tf)
            self.g_output_attribute_train_tf_l.append(
                g_output_attribute_train_tf)
            self.g_output_gen_flag_train_tf_l.append(
                g_output_gen_flag_train_tf)
            self.g_output_length_train_tf_l.append(
                g_output_length_train_tf)
            self.g_output_argmax_train_tf_l.append(
                g_output_argmax_train_tf)
        self.g_output_feature_train_tf = tf.concat(
            self.g_output_feature_train_tf_l,
            axis=1)
        self.g_output_attribute_train_tf = tf.concat(
            self.g_output_attribute_train_tf_l,
            axis=1)

        self.d_fake_train_tf = self.discriminator.build(
            self.g_output_feature_train_tf,
            self.g_output_attribute_train_tf)

        if self.attr_discriminator is not None:
            self.attr_d_fake_train_tf = self.attr_discriminator.build(
                self.g_output_attribute_train_tf)

        self.real_feature_pl_l = []
        for i in range(self.num_packing):
            real_feature_pl = tf.compat.v1.placeholder(
                tf.float32,
                [None,
                 self.sample_time * self.sample_len,
                 self.sample_feature_dim],
                name="real_feature_{}".format(i))
            if self.fix_feature_network:
                real_feature_pl = tf.zeros_like(
                    real_feature_pl)
            self.real_feature_pl_l.append(real_feature_pl)
        self.real_attribute_pl_l = []
        for i in range(self.num_packing):
            real_attribute_pl = tf.compat.v1.placeholder(
                tf.float32,
                [None, self.sample_attribute_dim],
                name="real_attribute_{}".format(i))
            if self.fix_feature_network:
                real_attribute_pl *= self.real_attribute_mask_tensor
            self.real_attribute_pl_l.append(real_attribute_pl)
        self.real_feature_pl = tf.concat(
            self.real_feature_pl_l,
            axis=1)
        self.real_attribute_pl = tf.concat(
            self.real_attribute_pl_l,
            axis=1)

        self.d_real_train_tf = self.discriminator.build(
            self.real_feature_pl,
            self.real_attribute_pl)
        self.d_real_test_tf = self.discriminator.build(
            self.real_feature_pl,
            self.real_attribute_pl)

        if self.attr_discriminator is not None:
            self.attr_d_real_train_tf = self.attr_discriminator.build(
                self.real_attribute_pl)

        self.g_real_attribute_input_noise_test_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.attribute_latent_dim],
            name="g_real_attribute_input_noise_test")
        self.g_addi_attribute_input_noise_test_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.attribute_latent_dim],
            name="g_addi_attribute_input_noise_test")
        self.g_feature_input_noise_test_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, None, self.feature_latent_dim],
            name="g_feature_input_noise_test")

        self.g_feature_input_data_test_teacher_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, None, self.sample_len * self.sample_feature_dim],
            name="g_feature_input_data_test_teacher")
        (self.g_output_feature_test_teacher_tf,
         self.g_output_attribute_test_teacher_tf,
         self.g_output_gen_flag_test_teacher_tf,
         self.g_output_length_test_teacher_tf, _) = \
            self.generator.build(
                self.g_real_attribute_input_noise_test_pl,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_teacher_pl,
                train=False)

        self.g_feature_input_data_test_free_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.sample_len * self.sample_feature_dim],
            name="g_feature_input_data_test_free")
        (self.g_output_feature_test_free_tf,
         self.g_output_attribute_test_free_tf,
         self.g_output_gen_flag_test_free_tf,
         self.g_output_length_test_free_tf, _) = \
            self.generator.build(
                self.g_real_attribute_input_noise_test_pl,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_free_pl,
                train=False)

        self.given_attribute_attribute_pl = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.sample_real_attribute_dim],
            name="given_attribute")
        (self.g_output_feature_given_attribute_test_free_tf,
         self.g_output_attribute_given_attribute_test_free_tf,
         self.g_output_gen_flag_given_attribute_test_free_tf,
         self.g_output_length_given_attribute_test_free_tf, _) = \
            self.generator.build(
                None,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_free_pl,
                train=False,
                attribute=self.given_attribute_attribute_pl)

    def build_loss(self):
        batch_size = tf.shape(input=self.g_feature_input_noise_train_pl_l[0])[0]

        self.g_loss_d = -tf.reduce_mean(input_tensor=self.d_fake_train_tf)
        if self.attr_discriminator is not None:
            self.g_loss_attr_d = -tf.reduce_mean(input_tensor=self.attr_d_fake_train_tf)
            self.g_loss = (self.g_loss_d +
                           self.g_attr_d_coe * self.g_loss_attr_d)
        else:
            self.g_loss = self.g_loss_d

        self.d_loss_fake = tf.reduce_mean(input_tensor=self.d_fake_train_tf)
        self.d_loss_fake_unflattened = self.d_fake_train_tf
        self.d_loss_real = -tf.reduce_mean(input_tensor=self.d_real_train_tf)
        self.d_loss_real_unflattened = -self.d_real_train_tf
        alpha_dim2 = tf.random.uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.)
        alpha_dim3 = tf.expand_dims(alpha_dim2, 2)
        differences_input_feature = (self.g_output_feature_train_tf -
                                     self.real_feature_pl)
        interpolates_input_feature = (self.real_feature_pl +
                                      alpha_dim3 * differences_input_feature)
        differences_input_attribute = (self.g_output_attribute_train_tf -
                                       self.real_attribute_pl)
        interpolates_input_attribute = (self.real_attribute_pl +
                                        (alpha_dim2 *
                                         differences_input_attribute))
        gradients = tf.gradients(
            ys=self.discriminator.build(
                interpolates_input_feature,
                interpolates_input_attribute),
            xs=[interpolates_input_feature, interpolates_input_attribute])
        slopes1 = tf.reduce_sum(input_tensor=tf.square(gradients[0]),
                                axis=[1, 2])
        slopes2 = tf.reduce_sum(input_tensor=tf.square(gradients[1]),
                                axis=[1])
        slopes = tf.sqrt(slopes1 + slopes2 + self.EPS)
        self.d_loss_gp = tf.reduce_mean(input_tensor=(slopes - 1.)**2)
        self.d_loss_gp_unflattened = (slopes - 1.)**2

        self.d_loss = (self.d_loss_fake +
                       self.d_loss_real +
                       self.d_gp_coe * self.d_loss_gp)

        self.d_loss_unflattened = (self.d_loss_fake_unflattened +
                                   self.d_loss_real_unflattened +
                                   self.d_gp_coe * self.d_loss_gp_unflattened)

        if self.attr_discriminator is not None:
            self.attr_d_loss_fake = tf.reduce_mean(input_tensor=self.attr_d_fake_train_tf)
            self.attr_d_loss_fake_unflattened = self.attr_d_fake_train_tf
            self.attr_d_loss_real = -tf.reduce_mean(input_tensor=self.attr_d_real_train_tf)
            self.attr_d_loss_real_unflattened = -self.attr_d_real_train_tf
            alpha_dim2 = tf.random.uniform(
                shape=[batch_size, 1],
                minval=0.,
                maxval=1.)
            differences_input_attribute = (self.g_output_attribute_train_tf -
                                           self.real_attribute_pl)
            interpolates_input_attribute = (self.real_attribute_pl +
                                            (alpha_dim2 *
                                             differences_input_attribute))
            gradients = tf.gradients(
                ys=self.attr_discriminator.build(
                    interpolates_input_attribute),
                xs=[interpolates_input_attribute])
            slopes1 = tf.reduce_sum(input_tensor=tf.square(gradients[0]),
                                    axis=[1])
            slopes = tf.sqrt(slopes1 + self.EPS)
            self.attr_d_loss_gp = tf.reduce_mean(input_tensor=(slopes - 1.)**2)
            self.attr_d_loss_gp_unflattened = (slopes - 1.)**2

            self.attr_d_loss = (self.attr_d_loss_fake +
                                self.attr_d_loss_real +
                                self.attr_d_gp_coe * self.attr_d_loss_gp)

            self.attr_d_loss_unflattened = \
                (self.attr_d_loss_fake_unflattened +
                 self.attr_d_loss_real_unflattened +
                 self.attr_d_gp_coe * self.attr_d_loss_gp_unflattened)

        self.g_op = \
            tf.compat.v1.train.AdamOptimizer(self.g_lr, self.g_beta1)\
            .minimize(
                self.g_loss,
                var_list=self.generator.trainable_vars)

        self.d_op = \
            tf.compat.v1.train.AdamOptimizer(self.d_lr, self.d_beta1)\
            .minimize(
                self.d_loss,
                var_list=self.discriminator.trainable_vars)

        if self.attr_discriminator is not None:
            self.attr_d_op = \
                tf.compat.v1.train.AdamOptimizer(self.attr_d_lr, self.attr_d_beta1)\
                .minimize(
                    self.attr_d_loss,
                    var_list=self.attr_discriminator.trainable_vars)

    def sample_from(self, real_attribute_input_noise,
                    addi_attribute_input_noise, feature_input_noise,
                    feature_input_data, given_attribute=None,
                    return_gen_flag_feature=False):
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(
            math.ceil(float(feature_input_noise.shape[0]) / self.batch_size))
        for i in range(round_):
            if given_attribute is None:
                if feature_input_data.ndim == 2:
                    (sub_features, sub_attributes, sub_gen_flags,
                     sub_lengths) = self.sess.run(
                        [self.g_output_feature_test_free_tf,
                         self.g_output_attribute_test_free_tf,
                         self.g_output_gen_flag_test_free_tf,
                         self.g_output_length_test_free_tf],
                        feed_dict={
                            self.g_real_attribute_input_noise_test_pl:
                                real_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_addi_attribute_input_noise_test_pl:
                                addi_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_noise_test_pl:
                                feature_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_data_test_free_pl:
                                feature_input_data[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size]})
                else:
                    (sub_features, sub_attributes, sub_gen_flags,
                     sub_lengths) = self.sess.run(
                        [self.g_output_feature_test_teacher_tf,
                         self.g_output_attribute_test_teacher_tf,
                         self.g_output_gen_flag_test_teacher_tf,
                         self.g_output_length_test_teacher_tf],
                        feed_dict={
                            self.g_real_attribute_input_noise_test_pl:
                                real_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_addi_attribute_input_noise_test_pl:
                                addi_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_noise_test_pl:
                                feature_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_data_test_teacher_pl:
                                feature_input_data[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size]})
            else:
                (sub_features, sub_attributes, sub_gen_flags,
                 sub_lengths) = self.sess.run(
                    [self.g_output_feature_given_attribute_test_free_tf,
                     self.g_output_attribute_given_attribute_test_free_tf,
                     self.g_output_gen_flag_given_attribute_test_free_tf,
                     self.g_output_length_given_attribute_test_free_tf],
                    feed_dict={
                        self.g_addi_attribute_input_noise_test_pl:
                            addi_attribute_input_noise[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.g_feature_input_noise_test_pl:
                            feature_input_noise[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.g_feature_input_data_test_free_pl:
                            feature_input_data[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.given_attribute_attribute_pl:
                            given_attribute[
                                i * self.batch_size:
                                (i + 1) * self.batch_size]})
            features.append(sub_features)
            attributes.append(sub_attributes)
            gen_flags.append(sub_gen_flags)
            lengths.append(sub_lengths)

        features = np.concatenate(features, axis=0)
        attributes = np.concatenate(attributes, axis=0)
        gen_flags = np.concatenate(gen_flags, axis=0)
        lengths = np.concatenate(lengths, axis=0)

        if not return_gen_flag_feature:
            features = np.delete(features, [features.shape[2] - 2, features.shape[2] - 1], axis=2)

        assert len(gen_flags.shape) == 3
        assert gen_flags.shape[2] == 1
        gen_flags = gen_flags[:, :, 0]

        return features, attributes, gen_flags, lengths

    def gen_attribute_input_noise(self, num_sample):
        return np.random.normal(
            size=[num_sample, self.attribute_latent_dim])

    def gen_feature_input_noise(self, num_sample, length=1):
        return np.random.normal(
            size=[num_sample, length, self.feature_latent_dim])

    def gen_feature_input_data_free(self, num_sample):
        return np.zeros(
            [num_sample, self.sample_len * self.sample_feature_dim],
            dtype=np.float32)

    def train(self):
        tf.compat.v1.global_variables_initializer().run()

        batch_num = self.data_feature.shape[0] // self.batch_size

        for _ in tqdm(range(self.epoch)):
            data_id = np.random.choice(
                self.data_feature.shape[0],
                size=(self.data_feature.shape[0], self.num_packing))

            for batch_id in range(batch_num):
                feed_dict = {}
                for i in range(self.num_packing):
                    batch_data_id = data_id[batch_id * self.batch_size:
                                            (batch_id + 1) * self.batch_size,
                                            i]
                    batch_data_feature = self.data_feature[batch_data_id]
                    batch_data_attribute = self.data_attribute[batch_data_id]

                    batch_real_attribute_input_noise = \
                        self.gen_attribute_input_noise(self.batch_size)
                    batch_addi_attribute_input_noise = \
                        self.gen_attribute_input_noise(self.batch_size)
                    batch_feature_input_noise = \
                        self.gen_feature_input_noise(
                            self.batch_size, self.sample_time)
                    batch_feature_input_data = \
                        self.gen_feature_input_data_free(self.batch_size)

                    feed_dict[self.real_feature_pl_l[i]] = \
                        batch_data_feature
                    feed_dict[self.real_attribute_pl_l[i]] = \
                        batch_data_attribute
                    feed_dict[self.
                              g_real_attribute_input_noise_train_pl_l[i]] = \
                        batch_real_attribute_input_noise
                    feed_dict[self.
                              g_addi_attribute_input_noise_train_pl_l[i]] = \
                        batch_addi_attribute_input_noise
                    feed_dict[self.g_feature_input_noise_train_pl_l[i]] = \
                        batch_feature_input_noise
                    feed_dict[self.g_feature_input_data_train_pl_l[i]] = \
                        batch_feature_input_data

                for _ in range(self.rounds):
                    self.sess.run(self.d_op, feed_dict=feed_dict)
                    if self.attr_discriminator is not None:
                        self.sess.run(self.attr_d_op, feed_dict=feed_dict)
                    self.sess.run(self.g_op, feed_dict=feed_dict)

    def save(self, path):
        dump({
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "sample_len": self.sample_len,
            "rounds": self.rounds,
            "d_gp_coe": self.d_gp_coe,
            "attr_d_gp_coe": self.attr_d_gp_coe,
            "g_attr_d_coe": self.g_attr_d_coe,
            "num_packing": self.num_packing,
            "attribute_latent_dim": self.attribute_latent_dim,
            "feature_latent_dim": self.feature_latent_dim,
            "fix_feature_network": self.fix_feature_network,
            "g_lr": self.g_lr,
            "g_beta1": self.g_beta1,
            "d_lr": self.d_lr,
            "d_beta1": self.d_beta1,
            "attr_d_lr": self.attr_d_lr,
            "attr_d_beta1": self.attr_d_beta1, 
            "sample_time": self.sample_time,
            "sample_feature_dim": self.sample_feature_dim,
            "sample_attribute_dim": self.sample_attribute_dim,
            "sample_real_attribute_dim": self.sample_real_attribute_dim        
        }, path)
