"Implements a GAN BaseModel synthesizer, not meant to be directly instantiated."
from collections import namedtuple
from typing import List, Optional, Union

import tensorflow as tf
import tqdm
from joblib import dump, load
from numpy import array, vstack
from pandas import DataFrame
from tensorflow import config as tfconfig
from typeguard import typechecked

from ydata_synthetic.preprocessing.regular.processor import (
    RegularDataProcessor, RegularModels)
from ydata_synthetic.preprocessing.timeseries.timeseries_processor import (
    TimeSeriesDataProcessor, TimeSeriesModels)
from ydata_synthetic.synthesizers.saving_keras import make_keras_picklable

_model_parameters = ['batch_size', 'lr', 'betas', 'layers_dim', 'noise_dim',
                     'n_cols', 'seq_len', 'condition', 'n_critic', 'n_features', 'tau_gs']
_model_parameters_df = [128, 1e-4, (None, None), 128, 264,
                        None, None, None, 1, None, 0.2]

_train_parameters = ['cache_prefix', 'label_dim', 'epochs', 'sample_interval', 'labels']

ModelParameters = namedtuple('ModelParameters', _model_parameters, defaults=_model_parameters_df)
TrainParameters = namedtuple('TrainParameters', _train_parameters, defaults=('', None, 300, 50, None))


# pylint: disable=R0902
@typechecked
class BaseModel():
    """
    Base class of GAN synthesizer models.
    The main methods are train (for fitting the synthesizer), save/load and sample (obtain synthetic records).
    Args:
        model_parameters (ModelParameters):
            Set of architectural parameters for model definition.
    """
    __MODEL__ = None

    def __init__(
            self,
            model_parameters: ModelParameters
    ):
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except (ValueError, RuntimeError):
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        #Validate the provided model parameters
        if model_parameters.betas is not None:
            assert len(model_parameters.betas) == 2, "Please provide the betas information as a tuple."

        self.batch_size = model_parameters.batch_size
        self._set_lr(model_parameters.lr)
        self.beta_1 = model_parameters.betas[0]
        self.beta_2 = model_parameters.betas[1]
        self.noise_dim = model_parameters.noise_dim
        self.data_dim = None
        self.layers_dim = model_parameters.layers_dim
        self.processor = None
        if self.__MODEL__ in RegularModels.__members__:
            self.tau = model_parameters.tau_gs

    # pylint: disable=E1101
    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    # pylint: disable=C0103
    def _set_lr(self, lr):
        if isinstance(lr, float):
            self.g_lr=lr
            self.d_lr=lr
        elif isinstance(lr,(list, tuple)):
            assert len(lr)==2, "Please provide a tow values array for the learning rates or a float."
            self.g_lr=lr[0]
            self.d_lr=lr[1]

    def define_gan(self):
        """Define the trainable model components.
        Optionally validate model structure with mock inputs and initialize optimizers."""
        raise NotImplementedError

    @property
    def model_parameters(self):
        "Returns the parameters of the model."
        return self._model_parameters

    @property
    def model_name(self):
        "Returns the model (class) name."
        return self.__class__.__name__

    def train(self,
              data: Union[DataFrame, array],
              num_cols: Optional[List[str]] = None,
              cat_cols: Optional[List[str]] = None) -> Union[DataFrame, array]:
        """Sets up the train session by instantiating an appropriate processor, fitting and storing it as an attribute.
        Args:
            data (Union[DataFrame, array]): Raw data object.
            num_cols (Optional[List[str]]): List of names of numerical columns.
            cat_cols (Optional[List[str]]): List of names of categorical columns.
        """
        if self.__MODEL__ in RegularModels.__members__:
            self.processor = RegularDataProcessor
        elif self.__MODEL__ in TimeSeriesModels.__members__:
            self.processor = TimeSeriesDataProcessor
        else:
            print(f'A DataProcessor is not available for the {self.__MODEL__}.')
        self.processor = self.processor(num_cols = num_cols, cat_cols = cat_cols).fit(data)

    def sample(self, n_samples: int):
        """Generate n_samples synthetic records from the synthesizer.
        The records returned are always a multiple of batch_size (can return excess of up to batch_size - 1 records).
        The samples are returned in the original data format, with any internal preprocessing inverted.

        Args:
            n_samples (int): Intended size of the synthetic sample.
        """
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = tf.random.uniform([self.batch_size, self.noise_dim], dtype=tf.dtypes.float32)
            records = self.generator(z, training=False).numpy()
            data.append(records)
        return self.processor.inverse_transform(array(vstack(data)))

    def save(self, path):
        "Saves the pickled synthesizer instance in the given path."
        #Save only the generator?
        if self.__MODEL__=='WGAN' or self.__MODEL__=='WGAN_GP' or self.__MODEL__=='CWGAN_GP':
            del self.critic
        make_keras_picklable()
        dump(self, path)

    @staticmethod
    def load(path):
        "Loads a pickled synthesizer from the given path."
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except (ValueError, RuntimeError):
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth
