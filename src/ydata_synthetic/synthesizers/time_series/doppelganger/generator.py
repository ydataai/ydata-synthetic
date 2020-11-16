import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, RNN, Dense, ReLU, BatchNormalization, LSTMCell
from tensorflow.keras.backend import one_hot, argmax, reshape, stop_gradient, concatenate, cast, expand_dims, tile

from metadata import Variable, Metadata, OutputType, Activation
from src.synthesizers.conditionalTimeSeries.network import Network

class Generator(Network):
    def __init__(self, train_metadata, noise_dim, period, sample_len,
                 features_dim = (100,), attributes_dim = (100, 100, 100),
                 batch_size = 100, initial_stddev = 0.02,
                 pack=1, scope_name='Generator', *args, **kwargs):

        super(Generator, self).__init__(name=scope_name, *args, **kwargs)

        self.metadata = train_metadata
        self.pack = pack
        self.batch_size = batch_size
        self.sample_len = sample_len
        self.period = period
        self.time = int(period/sample_len)

        self.features_dim = features_dim
        self.attributes_dim = attributes_dim
        self.all_attr_dim = train_metadata.getallattrdim_list()
        self.features_out_dim = train_metadata.getfeaturesoutdim()*self.sample_len

        self.attributes_out_dim = train_metadata.getallattrdim()
        self.noise_dim = noise_dim

        self.initial_stddev = initial_stddev

        self.gen_flag_id = None
        for i, (_,val) in enumerate(self.metadata.details.items()):
            if val.feature and val.is_gen_flag:
                self.gen_flag_id = i
                break

    def build_model(self):
        features_input = Input(shape=(self.time, self.noise_dim),
                               batch_size=self.batch_size)

        time = features_input.get_shape().as_list()[1]

        attributes_input = Input(shape=(self.noise_dim), batch_size=self.batch_size)
        addi_attr_input = Input(shape=(self.noise_dim), batch_size=self.batch_size)

        #calculate the dimensions
        all_attr_noise = [attributes_input, addi_attr_input]
        all_attr_ls = [self.metadata.getallattr(), self.metadata.getalladdattr()]
        #Build the generators for the attributes
        #Layers... features_input needs to concat with the result from this step

        all_attr = []
        all_attr_discrete = []
        for id, attributes in enumerate(all_attr_ls):
            for item in self.attributes_dim:
                x = Dense(item)(all_attr_noise[id])
                x = ReLU()(x)
                x = BatchNormalization()(x)
            outputs = []
            discrete_outputs = []

            #check here out to do it - Given the current structure of the metadata
            for output in attributes:
                sub_output = Dense(output.size, activation=output.activation.value)(x)

                if output.type == OutputType.CATEGORICAL:
                    discrete_suboutput = one_hot(argmax(sub_output, axis=1), output.size)
                elif output.type == OutputType.CONTINUOUS:
                    discrete_suboutput = sub_output
                else:
                    raise('Unknown data type')

                outputs.append(sub_output)
                discrete_outputs.append(discrete_suboutput)

            outputs = concatenate(outputs, axis=1)
            discrete_outputs = concatenate(discrete_outputs, axis=1)
            outputs = reshape(outputs, [self.batch_size, self.all_attr_dim[id]]) #
            discrete_outputs = reshape(discrete_outputs, [self.batch_size, self.all_attr_dim[id]])
            all_attr.append(outputs)
            all_attr_discrete.append(discrete_outputs)

        discrete_outputs = stop_gradient(discrete_outputs)

        all_attr = concatenate(all_attr, axis=1)
        all_attr_discrete = concatenate(all_attr_discrete, axis=1)
        # batch_size * attribute_noise_input_dim
        all_attr = reshape(all_attr, [self.batch_size, self.attributes_out_dim]) #update here with the calc attributes_dim
        all_attr = reshape(all_attr, [self.batch_size, self.attributes_out_dim]) #update here with the calc attributes_dim
        # batch_size * attribute_noise_input_dim

        all_attr_discrete = reshape(all_attr, [self.batch_size, self.attributes_out_dim])
        # batch_size * attrib_input_noise

        cells = []
        for item in self.features_dim:
            #Make the initial_state a trainable feature
            #https://pypi.org/project/trainable-initial-state-rnn/
            #Create a function to init the layers states... or have this as a trainable feature
            cells.append(LSTMCell(units=item))
        lstm = RNN(cells)

        def compute(LSTM, attr_discrete, features_input, features, time, metadata, gen_flag, all_gen_flag, all_cur_argmax):
            for i in range(time):
                inputs = concatenate([attr_discrete, tf.transpose(features_input, [1,0,2])[i]], axis=1)
                shape = inputs.shape
                inputs = reshape(inputs, shape=(shape[0] , 1 , shape[1]))

                lstm_outputs = LSTM(inputs)
                meta_features = metadata.getallfeatures()
                new_output_ls=[]
                for j in range(self.sample_len):
                    for feature in meta_features:
                        sub_output = Dense(feature.size, activation=feature.activation.value)(lstm_outputs)
                        new_output_ls.append(sub_output)
                new_output = concatenate(new_output_ls, axis=1)
                features.write(i, new_output)

                for j in range(self.sample_len):
                    all_gen_flag = all_gen_flag.write(i * self.sample_len + j, gen_flag)

                    #Revise here the current generation flag <- This is the logic that is needed
                    cur_gen_flag = cast(tf.equal(argmax(new_output_ls[(j * self.metadata.num_features + self.gen_flag_id)], axis=1), 0), tf.float32)
                    cur_gen_flag = reshape(cur_gen_flag, [-1,1])
                    all_cur_argmax = all_cur_argmax.write(i * self.sample_len + j,
                                                          argmax(new_output_ls[(j * self.metadata.num_features + self.gen_flag_id)], axis=1))
                    gen_flag = gen_flag * cur_gen_flag

            return features.stack(), all_gen_flag.stack(), gen_flag, all_cur_argmax.stack()

        features, \
        all_gen_flag,\
        gen_flag, \
        all_cur_argmax = compute(lstm,
                           all_attr_discrete,
                           features_input,
                           tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
                           time,
                           self.metadata,
                           tf.ones((self.batch_size, 1)),
                           tf.TensorArray(tf.float32, time * self.sample_len),
                           tf.TensorArray(tf.int64, time * self.sample_len))

        #features, all_gen_flag, all_cur_argmax= fill_rest(time, features, all_gen_flag, all_cur_argmax)

        #Tensor output shape:
        #features: time * batch_size * (dim * sample_len)
        #all_gen_flag: (time * sample_len) * batch_size * 1

        gen_flag = tf.transpose(all_gen_flag, [1, 0, 2])
        # gen_flag: batch_size * (time * sample_len) * 1
        cur_argmax = tf.transpose(all_cur_argmax, [1, 0])
        # batch_size * (time * sample_len)
        length = tf.reduce_sum(gen_flag, [1, 2])
        # batch_size

        features = tf.transpose(features, [1, 0, 2])

        gen_flag_t = reshape(gen_flag,
                            [self.batch_size, time, self.sample_len])
        # gena_flag transpose: batch_size * time * sample_len

        gen_flag_t = tf.reduce_sum(gen_flag_t, [2])
        # batch_size * time
        gen_flag_t = cast(gen_flag_t > 0.5, dtype=tf.float32)
        gen_flag_t = expand_dims(gen_flag_t, 2)
        # batch_size * time * 1

        gen_flag_t = tile(gen_flag_t,
                          [1, 1, self.features_out_dim])

        features = features * gen_flag_t
        # batch_size * time * (features_output)
        features = reshape(features,
                            [self.batch_size, time * self.sample_len, int(self.features_out_dim/self.sample_len)])
        # batch_size * (time * sample_len) * dim

        return Model(inputs = [features_input, attributes_input, addi_attr_input], outputs = [features, all_attr_discrete])

        #return gen_flag,  length, cur_argmax

if __name__ == '__main__':
    print('Start testing.')

    #Create here a metadata example to test with the EDP dataset
    def load_data(path):
        data_npz = np.load(path)
        return data_npz["data_feature"], data_npz["data_attribute"], data_npz["real_attribute_mask"]

    def create_test_metadata():
        num_features = 4
        num_attr = [2,12]
        num_add_attr = [1,1,1,1,1,1]

        features = []
        for i in range(num_features):
            is_gen_flag = False
            size = 1
            if i >= num_features-1:
                is_gen_flag=True
                size = 2

            var = Variable(name='feat_{}'.format(i), var_type=OutputType.CONTINUOUS,
                           size=size, activation=Activation.TANH, is_gen_flag=is_gen_flag)
            features.append(var)

        attributes = []
        for i in range(len(num_attr)):
            var = Variable(name='attr_{}'.format(i), var_type=OutputType.CATEGORICAL, size=num_attr[i], activation=Activation.SOFTMAX, is_gen_flag=False)
            attributes.append(var)

        add_attributes = []
        for j in range(len(num_add_attr)):
            var = Variable(name='add_attr_{}'.format(j), var_type=OutputType.CATEGORICAL, size=num_add_attr[j],
                           activation=Activation.SIGMOID, is_gen_flag=False)
            add_attributes.append(var)
        return  Metadata(features, attributes=attributes, add_attributes=add_attributes)

    data_feature, data_attribute, data_gen_flag = load_data("/home/fabiana/Documents/Demos/YData/data/data_train.npz")
    metadata = create_test_metadata()

    print('Data loaded')
    #Create here the metadata for the EDP use case to be tested with the new Doppelganger

    generator = Generator(metadata=metadata)
    generator.build_model()
    #generator.build_model((28000, 96, 3), (28000, 20))




