import tensorflow as tf
import numpy as np


def linear(input_, output_size, scope_name="linear"):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/op.py.
    """
    with tf.compat.v1.variable_scope(scope_name):
        input_ = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        output = tf.compat.v1.layers.dense(
            input_,
            output_size)
        return output


def flatten(input_, scope_name="flatten"):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/op.py.
    """
    with tf.compat.v1.variable_scope(scope_name):
        output = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        return output


class batch_norm(object):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/op.py.
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                  epsilon=self.epsilon,
                                                  scale=True,
                                                  trainable=train,
                                                  name=self.name)(x)


class Network(object):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/network.py.
    """
    def __init__(self, scope_name):
        self.scope_name = scope_name

    @property
    def trainable_vars(self):
        return tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope_name)


class Discriminator(Network):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/network.py.
    """
    def __init__(self,
                 num_layers=5, num_units=200,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.num_layers = num_layers
        self.num_units = num_units

    def build(self, input_feature, input_attribute):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            input_feature = flatten(input_feature)
            input_attribute = flatten(input_attribute)
            input_ = tf.concat([input_feature, input_attribute], 1)
            layers = [input_feature, input_attribute, input_]
            for i in range(self.num_layers - 1):
                with tf.compat.v1.variable_scope("layer{}".format(i)):
                    layers.append(linear(layers[-1], self.num_units))
                    layers.append(tf.nn.relu(layers[-1]))
            with tf.compat.v1.variable_scope("layer{}".format(self.num_layers - 1)):
                layers.append(linear(layers[-1], 1))
                layers.append(tf.squeeze(layers[-1], 1))
            return layers[-1]


class AttrDiscriminator(Network):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/network.py.
    """
    def __init__(self,
                 num_layers=5, num_units=200,
                 scope_name="attrDiscriminator", *args, **kwargs):
        super(AttrDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.num_layers = num_layers
        self.num_units = num_units

    def build(self, input_attribute):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            input_attribute = flatten(input_attribute)
            layers = [input_attribute]
            for i in range(self.num_layers - 1):
                with tf.compat.v1.variable_scope("layer{}".format(i)):
                    layers.append(linear(layers[-1], self.num_units))
                    layers.append(tf.nn.relu(layers[-1]))
            with tf.compat.v1.variable_scope("layer{}".format(self.num_layers - 1)):
                layers.append(linear(layers[-1], 1))
                layers.append(tf.squeeze(layers[-1], 1))
            return layers[-1]


class DoppelGANgerGenerator(Network):
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/network.py.
    """
    def __init__(self, feed_back, noise,
                 measurement_cols_metadata, attribute_cols_metadata, sample_len,
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=1, use_tanh=False,
                 scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.feed_back = feed_back
        self.noise = noise
        self.attribute_num_units = attribute_num_units
        self.attribute_num_layers = attribute_num_layers
        self.feature_num_units = feature_num_units
        self.measurement_cols_metadata = measurement_cols_metadata
        self.attribute_cols_metadata = attribute_cols_metadata
        self.feature_num_layers = feature_num_layers
        self.use_tanh = use_tanh
        self.sample_len = sample_len
        self.feature_out_dim = (np.sum([t.output_dim for t in measurement_cols_metadata]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.output_dim for t in attribute_cols_metadata])
        if not self.noise and not self.feed_back:
            raise Exception("noise and feed_back should have at least one True")

        self.real_attribute_outputs = [c for c in self.attribute_cols_metadata if c.real]
        self.addi_attribute_outputs = [c for c in self.attribute_cols_metadata if not c.real]
        self.real_attribute_out_dim = sum([c.output_dim for c in self.attribute_cols_metadata if c.real])
        self.addi_attribute_out_dim = sum([c.output_dim for c in self.attribute_cols_metadata if not c.real])

        self.gen_flag_id = len(self.measurement_cols_metadata) - 1
        self.STR_REAL = "real"
        self.STR_ADDI = "addi"

    # noqa: MC0001
    def build(self, attribute_input_noise, addi_attribute_input_noise,
              feature_input_noise, feature_input_data, train, attribute=None):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            batch_size = tf.shape(input=feature_input_noise)[0]

            if attribute is None:
                all_attribute = []
                all_discrete_attribute = []
                if len(self.addi_attribute_outputs) > 0 and len(self.real_attribute_outputs) > 0:
                    all_attribute_input_noise = \
                        [attribute_input_noise,
                         addi_attribute_input_noise]
                    all_attribute_outputs = \
                        [self.real_attribute_outputs,
                         self.addi_attribute_outputs]
                    all_attribute_part_name = \
                        [self.STR_REAL, self.STR_ADDI]
                    all_attribute_out_dim = \
                        [self.real_attribute_out_dim,
                         self.addi_attribute_out_dim]
                elif len(self.addi_attribute_outputs) > 0:
                    all_attribute_input_noise = [addi_attribute_input_noise]
                    all_attribute_outputs = [self.addi_attribute_outputs]
                    all_attribute_part_name = [self.STR_ADDI]
                    all_attribute_out_dim = [self.addi_attribute_out_dim]
                else:
                    all_attribute_input_noise = [attribute_input_noise]
                    all_attribute_outputs = [self.real_attribute_outputs]
                    all_attribute_part_name = [self.STR_REAL]
                    all_attribute_out_dim = [self.real_attribute_out_dim]
            else:
                all_attribute = [attribute]
                all_discrete_attribute = [attribute]
                if len(self.addi_attribute_outputs) > 0:
                    all_attribute_input_noise = \
                        [addi_attribute_input_noise]
                    all_attribute_outputs = \
                        [self.addi_attribute_outputs]
                    all_attribute_part_name = \
                        [self.STR_ADDI]
                    all_attribute_out_dim = [self.addi_attribute_out_dim]
                else:
                    all_attribute_input_noise = []
                    all_attribute_outputs = []
                    all_attribute_part_name = []
                    all_attribute_out_dim = []

            for part_i, _ in enumerate(all_attribute_input_noise):
                with tf.compat.v1.variable_scope(
                        "attribute_{}".format(all_attribute_part_name[part_i]),
                        reuse=tf.compat.v1.AUTO_REUSE):

                    if len(all_discrete_attribute) > 0:
                        layers = [tf.concat(
                            [all_attribute_input_noise[part_i]] +
                            all_discrete_attribute,
                            axis=1)]
                    else:
                        layers = [all_attribute_input_noise[part_i]]

                    for i in range(self.attribute_num_layers - 1):
                        with tf.compat.v1.variable_scope("layer{}".format(i)):
                            layers.append(linear(layers[-1], self.attribute_num_units))
                            layers.append(tf.nn.relu(layers[-1]))
                            layers.append(batch_norm()(layers[-1], train=train))
                    with tf.compat.v1.variable_scope(
                            "layer{}".format(self.attribute_num_layers - 1),
                            reuse=tf.compat.v1.AUTO_REUSE):
                        part_attribute = []
                        part_discrete_attribute = []
                        for i in range(len(all_attribute_outputs[part_i])):
                            with tf.compat.v1.variable_scope("output{}".format(i),
                                                   reuse=tf.compat.v1.AUTO_REUSE):
                                output = all_attribute_outputs[part_i][i]

                                sub_output_ori = linear(layers[-1], output.output_dim)
                                if output.discrete:
                                    sub_output = tf.nn.softmax(sub_output_ori)
                                    sub_output_discrete = tf.one_hot(
                                        tf.argmax(input=sub_output, axis=1),
                                        output.output_dim)
                                else:
                                    if self.use_tanh:
                                        sub_output = tf.nn.tanh(sub_output_ori)
                                    else:
                                        sub_output = tf.nn.sigmoid(sub_output_ori)
                                    sub_output_discrete = sub_output
                                part_attribute.append(sub_output)
                                part_discrete_attribute.append(
                                    sub_output_discrete)
                        part_attribute = tf.concat(part_attribute, axis=1)
                        part_discrete_attribute = tf.concat(
                            part_discrete_attribute, axis=1)
                        part_attribute = tf.reshape(
                            part_attribute,
                            [batch_size, all_attribute_out_dim[part_i]])
                        part_discrete_attribute = tf.reshape(
                            part_discrete_attribute,
                            [batch_size, all_attribute_out_dim[part_i]])
                        # batch_size * dim

                    part_discrete_attribute = tf.stop_gradient(
                        part_discrete_attribute)

                    all_attribute.append(part_attribute)
                    all_discrete_attribute.append(part_discrete_attribute)

            all_attribute = tf.concat(all_attribute, axis=1)
            all_discrete_attribute = tf.concat(all_discrete_attribute, axis=1)
            all_attribute = tf.reshape(
                all_attribute,
                [batch_size, self.attribute_out_dim])
            all_discrete_attribute = tf.reshape(
                all_discrete_attribute,
                [batch_size, self.attribute_out_dim])

            with tf.compat.v1.variable_scope("feature", reuse=tf.compat.v1.AUTO_REUSE):
                all_cell = []
                for i in range(self.feature_num_layers):
                    with tf.compat.v1.variable_scope("unit{}".format(i),
                                           reuse=tf.compat.v1.AUTO_REUSE):
                        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                            num_units=self.feature_num_units,
                            state_is_tuple=True)
                        all_cell.append(cell)
                rnn_network = tf.compat.v1.nn.rnn_cell.MultiRNNCell(all_cell)

                feature_input_data_dim = \
                    len(feature_input_data.get_shape().as_list())
                if feature_input_data_dim == 3:
                    feature_input_data_reshape = tf.transpose(
                        a=feature_input_data, perm=[1, 0, 2])
                feature_input_noise_reshape = tf.transpose(
                    a=feature_input_noise, perm=[1, 0, 2])

                initial_state = tf.random.normal(
                    shape=(self.feature_num_layers,
                            2,
                            batch_size,
                            self.feature_num_units),
                    mean=0.0, stddev=1.0)
                initial_state = tf.unstack(initial_state, axis=0)
                initial_state = tuple(
                    [tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                        initial_state[idx][0], initial_state[idx][1])
                        for idx in range(self.feature_num_layers)])

                time = feature_input_noise.get_shape().as_list()[1]
                if time is None:
                    time = tf.shape(input=feature_input_noise)[1]

                def compute(i, state, last_output, all_output,
                            gen_flag, all_gen_flag, all_cur_argmax,
                            last_cell_output):
                    input_all = [all_discrete_attribute]
                    if self.noise:
                        input_all.append(feature_input_noise_reshape[i])
                    if self.feed_back:
                        if feature_input_data_dim == 3:
                            input_all.append(feature_input_data_reshape[i])
                        else:
                            input_all.append(last_output)
                    input_all = tf.concat(input_all, axis=1)

                    cell_new_output, new_state = rnn_network(input_all, state)
                    new_output_all = []
                    id_ = 0
                    for j in range(self.sample_len):
                        for k, _ in enumerate(self.measurement_cols_metadata):
                            with tf.compat.v1.variable_scope("output{}".format(id_),
                                                   reuse=tf.compat.v1.AUTO_REUSE):
                                output = self.measurement_cols_metadata[k]
                                sub_output = linear(cell_new_output, output.output_dim)
                                if output.discrete:
                                    sub_output = tf.nn.softmax(sub_output)
                                else:
                                    if self.use_tanh:
                                        sub_output = tf.nn.tanh(sub_output)
                                    else:
                                        sub_output = tf.nn.sigmoid(sub_output)
                                new_output_all.append(sub_output)
                                id_ += 1
                    new_output = tf.concat(new_output_all, axis=1)

                    for j in range(self.sample_len):
                        all_gen_flag = all_gen_flag.write(
                            i * self.sample_len + j, gen_flag)
                        cur_gen_flag = tf.cast(tf.equal(tf.argmax(
                            input=new_output_all[(j * len(self.measurement_cols_metadata) +
                                            self.gen_flag_id)],
                            axis=1), 0), dtype=tf.float32)
                        cur_gen_flag = tf.reshape(cur_gen_flag, [-1, 1])
                        all_cur_argmax = all_cur_argmax.write(
                            i * self.sample_len + j,
                            tf.argmax(
                                input=new_output_all[(j * len(self.measurement_cols_metadata) +
                                                self.gen_flag_id)],
                                axis=1))
                        gen_flag = gen_flag * cur_gen_flag

                    return (i + 1,
                            new_state,
                            new_output,
                            all_output.write(i, new_output),
                            gen_flag,
                            all_gen_flag,
                            all_cur_argmax,
                            cell_new_output)

                (i, _, _, feature, _, gen_flag, cur_argmax, _) = \
                    tf.while_loop(
                        cond=lambda a, b, c, d, e, f, g, h:
                        tf.logical_and(a < time,
                                       tf.equal(tf.reduce_max(input_tensor=e), 1)),
                        body=compute,
                        loop_vars=(0,
                         initial_state,
                         feature_input_data if feature_input_data_dim == 2
                            else feature_input_data_reshape[0],
                         tf.TensorArray(tf.float32, time),
                         tf.ones((batch_size, 1)),
                         tf.TensorArray(tf.float32, time * self.sample_len),
                         tf.TensorArray(tf.int64, time * self.sample_len),
                         tf.zeros((batch_size, self.feature_num_units))))

                def fill_rest(i, all_output, all_gen_flag, all_cur_argmax):
                    all_output = all_output.write(
                        i, tf.zeros((batch_size, self.feature_out_dim)))

                    for j in range(self.sample_len):
                        all_gen_flag = all_gen_flag.write(
                            i * self.sample_len + j,
                            tf.zeros((batch_size, 1)))
                        all_cur_argmax = all_cur_argmax.write(
                            i * self.sample_len + j,
                            tf.zeros((batch_size,), dtype=tf.int64))
                    return (i + 1,
                            all_output,
                            all_gen_flag,
                            all_cur_argmax)

                _, feature, gen_flag, cur_argmax = tf.while_loop(
                    cond=lambda a, b, c, d: a < time,
                    body=fill_rest,
                    loop_vars=(i, feature, gen_flag, cur_argmax))

                feature = feature.stack()
                # time * batch_size * (dim * sample_len)
                gen_flag = gen_flag.stack()
                # (time * sample_len) * batch_size * 1
                cur_argmax = cur_argmax.stack()

                gen_flag = tf.transpose(a=gen_flag, perm=[1, 0, 2])
                # batch_size * (time * sample_len) * 1
                cur_argmax = tf.transpose(a=cur_argmax, perm=[1, 0])
                # batch_size * (time * sample_len)
                length = tf.reduce_sum(input_tensor=gen_flag, axis=[1, 2])
                # batch_size

                feature = tf.transpose(a=feature, perm=[1, 0, 2])
                # batch_size * time * (dim * sample_len)
                gen_flag_t = tf.reshape(
                    gen_flag,
                    [batch_size, time, self.sample_len])
                # batch_size * time * sample_len
                gen_flag_t = tf.reduce_sum(input_tensor=gen_flag_t, axis=[2])
                # batch_size * time
                gen_flag_t = tf.cast(gen_flag_t > 0.5, dtype=tf.float32)
                gen_flag_t = tf.expand_dims(gen_flag_t, 2)
                # batch_size * time * 1
                gen_flag_t = tf.tile(
                    gen_flag_t,
                    [1, 1, self.feature_out_dim])
                # batch_size * time * (dim * sample_len)
                # zero out the parts after sequence ends
                feature = feature * gen_flag_t
                feature = tf.reshape(
                    feature,
                    [batch_size,
                     time * self.sample_len,
                     self.feature_out_dim / self.sample_len])
                # batch_size * (time * sample_len) * dim

            return feature, all_attribute, gen_flag, length, cur_argmax
