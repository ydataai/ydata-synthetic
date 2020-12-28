import os
from os import path
import numpy as np

from ydata_synthetic.synthesizers import gan

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

class CGAN():

    def __init__(self, model_parameters):
        [self.batch_size, lr,self.beta_1, self.beta_2, self.noise_dim,
         self.data_dim, num_classes, self.classes, layers_dim] = model_parameters

        self.generator = Generator(self.batch_size, num_classes). \
            build_model(input_shape=(self.noise_dim,), dim=layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size, num_classes). \
            build_model(input_shape=(self.data_dim,), dim=layers_dim)

        optimizer = Adam(lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,), batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size)
        record = self.generator([z, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator([record, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, label], validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  # wasteful to shuffle every time
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        x = train.loc[train_ix[start_i: stop_i]].values
        return np.reshape(x, (batch_size, -1))

    def train(self, data, train_arguments):
        [cache_prefix, label_dim, epochs, sample_interval, data_dir] = train_arguments

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_x = self.get_data_batch(data, self.batch_size)
            label = batch_x[:, label_dim]
            noise = tf.random.normal((self.batch_size, self.noise_dim))

            # Generate a batch of new records
            gen_records = self.generator([noise, label], training=True)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([batch_x, label], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_records, label], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([noise, label], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Test here data generation step
                # save model checkpoints
                if path.exists('./cache') is False:
                    os.mkdir('./cache')
                model_checkpoint_base_name = './cache/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

                #Here is generating synthetic data
                z = tf.random.normal((432, self.noise_dim))
                label_z = tf.random.uniform((432,), minval=min(self.classes), maxval=max(self.classes)+1, dtype=tf.dtypes.int32)
                gen_data = self.generator([z, label_z])

class Generator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim, data_dim):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        input = multiply([noise, label_embedding])

        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=[noise, label], outputs=x)

class Discriminator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build_model(self, input_shape, dim):
        events = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        events_flat = Flatten()(events)
        input = multiply([events_flat, label_embedding])

        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[events, label], outputs=x)


if __name__ == '__main__':
    import pandas as pd
    from src.ydata_synthetic.preprocessing import transformations
    import sklearn.cluster as cluster

    data = pd.read_csv('/home/fabiana/PycharmProjects/YData/gan-playground/examples/data/creditcard.csv')
    
    data_cols = list(data.columns[data.columns != 'Class'])
    label_cols = ['Class']
    
    print('Dataset columns: {}'.format(data_cols))
    sorted_cols = ['V14', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3',
                   'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9',
                   'V23',
                   'Class']
    processed_data = data[sorted_cols].copy()
    
    data = transformations(data)
    
    # For the purpose of this example we will only synthesize the minority class
    train_data = data.loc[data['Class'] == 1].copy()
    
    print(
        "Dataset info: Number of records - {} Number of varibles - {}".format(train_data.shape[0],
                                                                              train_data.shape[1]))
    algorithm = cluster.KMeans
    args, kwds = (), {'n_clusters': 2, 'random_state': 0}
    labels = algorithm(*args, **kwds).fit_predict(train_data[data_cols])
    
    print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'],
                       index=np.unique(labels)))
    
    fraud_w_classes = train_data.copy()
    fraud_w_classes['Class'] = labels
    
    noise_dim = 32
    dim = 128
    batch_size = 128
    
    log_step = 100
    epochs = 500 + 1
    learning_rate = 5e-4
    models_dir = './cache'
    
    train_sample = data.copy().reset_index(drop=True)
    train_sample = pd.get_dummies(train_sample, columns=['Class'], prefix='Class', drop_first=True)
    label_cols = [i for i in train_sample.columns if 'Class' in i]
    data_cols = [i for i in train_sample.columns if i not in label_cols]
    train_sample[data_cols] = train_sample[data_cols] / 10  # scale to random noise size, one less thing to learn
    train_no_label = train_sample[data_cols]
    
    gan_args = [batch_size, learning_rate, noise_dim, train_sample.shape[1], 2, (0, 1), dim]
    train_args = ['', -1, epochs, log_step, '']

    synthesizer = CGAN(gan_args)
    synthesizer.train(train_sample, train_args)





