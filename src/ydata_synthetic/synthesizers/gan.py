from collections import namedtuple
from typing import Union, Optional

from pandas import DataFrame, concat
from numpy import array

import tqdm
from joblib import dump, load
import tensorflow as tf
from tensorflow import config as tfconfig
from typeguard import typechecked

from ydata_synthetic.synthesizers.saving_keras import make_keras_picklable
from ydata_synthetic.preprocessing.regular.processor import RegularDataProcessor, RegularModels, RegProcessorArguments
from ydata_synthetic.preprocessing.timeseries.timeseries_processor import TimeSeriesDataProcessor, TimeSeriesModels, TSProcessorArguments

_model_parameters = ['batch_size', 'lr', 'betas', 'layers_dim', 'noise_dim',
                     'n_cols', 'seq_len', 'condition', 'n_critic', 'n_features']
_model_parameters_df = [128, 1e-4, (None, None), 128, 264,
                        None, None, None, 1, None]

_train_parameters = ['cache_prefix', 'label_dim', 'epochs', 'sample_interval', 'labels']

ModelParameters = namedtuple('ModelParameters', _model_parameters, defaults=_model_parameters_df)
TrainParameters = namedtuple('TrainParameters', _train_parameters, defaults=('', None, 300, 50, None))

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
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        #Validate the provided model parameters
        if model_parameters.betas!=None:
            assert len(model_parameters.betas) == 2, "Please provide the betas information as a tuple."

        self.batch_size = model_parameters.batch_size
        self._set_lr(model_parameters.lr)
        self.beta_1 = model_parameters.betas[0]
        self.beta_2 = model_parameters.betas[1]
        self.noise_dim = model_parameters.noise_dim
        self.data_dim = None
        self.layers_dim = model_parameters.layers_dim
        self.processor = None

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    def _set_lr(self, lr):
        if isinstance(lr, float):
            self.g_lr=lr
            self.d_lr=lr
        elif isinstance(lr,list) or isinstance(lr, tuple):
            assert len(lr)==2, "Please provide a tow values array for the learning rates or a float."
            self.g_lr=lr[0]
            self.d_lr=lr[1]

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

    def train(self,
              data: Union[DataFrame, array],
              processor_arguments: Optional[Union[RegProcessorArguments, TSProcessorArguments]] = None,
              preprocess: bool = True) -> Union[DataFrame, array]:
        """Sets up the train session by instantiating an appropriate processor and transforming the data.
        Returns the data object (preprocessed when applicable) for the child class specific train method to resume.
        Args:
            data (Union[DataFrame, array]): Raw data object.
            processor_arguments (Union[RegProcessorArguments, TSProcessorArguments]): Defines the data processor.
            preprocess (bool): Determines if the preprocessor is to be run on the data or not (p.e. preprocessed data).
        Returns:
            processed_data (Union[DataFrame, array]): Processed data object (if applicable).
        """
        processed_data = data
        if preprocess:
            if self.__MODEL__ in RegularModels.__members__:
                self.processor = RegularDataProcessor
            elif self.__MODEL__ in TimeSeriesModels.__members__:
                self.processor = TimeSeriesDataProcessor
            else:
                print(f'A DataProcessor is not available for the {self.__MODEL__}.')
        if self.processor and preprocess and processor_arguments:
            processed_data = self.processor(processor_arguments).fit_transform(data)

        self.data_dim = processed_data.shape[1]
        self.define_gan()

        return processed_data

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(DataFrame(records))
        return concat(data)

    def save(self, path):
        #Save only the generator?
        if self.__MODEL__=='WGAN' or self.__MODEL__=='WGAN_GP':
            self.critic=None
        make_keras_picklable()
        dump(self, path)

    @staticmethod
    def load(path):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth
