import tqdm
from joblib import dump, load
import pandas as pd
import tensorflow as tf
from tensorflow import config as tfconfig

from ydata_synthetic.synthesizers.saving_keras import make_keras_picklable

class Model():
    def __init__(
            self,
            model_parameters
    ):
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        self._model_parameters = model_parameters
        [self.batch_size, self.lr, self.beta_1, self.beta_2, self.noise_dim,
         self.data_dim, self.layers_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    def define_gan(self):
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        return network.trainable_variables

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def model_name(self):
        return self.__class__.__name__

    def train(self, data, train_arguments):
        raise NotImplementedError

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(pd.DataFrame(records))
        return pd.concat(data)

    def save(self, path):
        make_keras_picklable()
        try:
            dump(self, path)
        except:
            raise Exception('Please provide a valid path to save the model.')

    @classmethod
    def load(cls, path):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth
