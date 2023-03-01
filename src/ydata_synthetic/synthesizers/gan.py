"Implements a GAN BaseModel synthesizer, not meant to be directly instantiated."
from collections import namedtuple
from typing import List, Optional, Union

import tqdm

from numpy import array, vstack, ndarray
from numpy.random import normal
from pandas.api.types import is_float_dtype, is_integer_dtype
from pandas import DataFrame
from pandas import concat

from joblib import dump, load

import tensorflow as tf

from tensorflow import config as tfconfig
from tensorflow import data as tfdata
from tensorflow import random
from typeguard import typechecked

from ydata_synthetic.preprocessing.regular.processor import (
    RegularDataProcessor, RegularModels)
from ydata_synthetic.preprocessing.timeseries.timeseries_processor import (
    TimeSeriesDataProcessor, TimeSeriesModels)
from ydata_synthetic.preprocessing.regular.ctgan_processor import CTGANDataProcessor
from ydata_synthetic.synthesizers.saving_keras import make_keras_picklable

_model_parameters = ['batch_size', 'lr', 'betas', 'layers_dim', 'noise_dim',
                     'n_cols', 'seq_len', 'condition', 'n_critic', 'n_features', 
                     'tau_gs', 'generator_dims', 'critic_dims', 'l2_scale', 
                     'latent_dim', 'gp_lambda', 'pac']
_model_parameters_df = [128, 1e-4, (None, None), 128, 264,
                        None, None, None, 1, None, 0.2, [256, 256], 
                        [256, 256], 1e-6, 128, 10.0, 10]

_train_parameters = ['cache_prefix', 'label_dim', 'epochs', 'sample_interval', 
                     'labels', 'n_clusters', 'epsilon', 'log_frequency']

ModelParameters = namedtuple('ModelParameters', _model_parameters, defaults=_model_parameters_df)
TrainParameters = namedtuple('TrainParameters', _train_parameters, defaults=('', None, 300, 50, None, 10, 0.005, True))


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

        # Additional parameters for the CTGAN
        self.generator_dims = model_parameters.generator_dims
        self.critic_dims = model_parameters.critic_dims
        self.l2_scale = model_parameters.l2_scale
        self.latent_dim = model_parameters.latent_dim
        self.gp_lambda = model_parameters.gp_lambda
        self.pac = model_parameters.pac

        self.processor = None
        if self.__MODEL__ in RegularModels.__members__ or \
            self.__MODEL__ == CTGANDataProcessor.SUPPORTED_MODEL:
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
            assert len(lr)==2, "Please provide a two values array for the learning rates or a float."
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

    def fit(self,
              data: Union[DataFrame, array],
              num_cols: Optional[List[str]] = None,
              cat_cols: Optional[List[str]] = None,
              train_arguments: Optional[TrainParameters] = None) -> Union[DataFrame, array]:
        """
        ### Description:
        Trains and fit a synthesizer model to a given input dataset.

        ### Args:
        `data` (Union[DataFrame, array]): Training data
        `num_cols` (Optional[List[str]]) : List with the names of the categorical columns
        `cat_cols` (Optional[List[str]]): List of names of categorical columns
        `train_arguments` (Optional[TrainParameters]): Training parameters

        ### Returns:
        **self:** *object*
            Fitted synthesizer
        """
        if self.__MODEL__ in RegularModels.__members__:
            self.processor = RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols).fit(data)
        elif self.__MODEL__ in TimeSeriesModels.__members__:
            self.processor = TimeSeriesDataProcessor(num_cols=num_cols, cat_cols=cat_cols).fit(data)
        elif self.__MODEL__ == CTGANDataProcessor.SUPPORTED_MODEL:
            n_clusters = train_arguments.n_clusters
            epsilon = train_arguments.epsilon
            self.processor = CTGANDataProcessor(n_clusters=n_clusters, epsilon=epsilon, 
                                                num_cols=num_cols, cat_cols=cat_cols).fit(data)
        else:
            print(f'A DataProcessor is not available for the {self.__MODEL__}.')

    def sample(self, n_samples: int):
        """
        ### Description:
        Generates samples from the trained synthesizer.

        ### Args:
        `n_samples` (int): Number of rows to generated.

        ### Returns:
        **synth_sample:** pandas.DataFrame, shape (n_samples, n_features)
            Returns the generated synthetic samples.
        """
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = random.uniform([self.batch_size, self.noise_dim], dtype=tf.dtypes.float32)
            records = self.generator(z, training=False).numpy()
            data.append(records)
        return self.processor.inverse_transform(array(vstack(data)))

    def save(self, path):
        """
        ### Description:
        Saves a synthesizer as a pickle.

        ### Args:
        `path` (str): Path to write the synthesizer as a pickle object.
        """
        #Save only the generator?
        if self.__MODEL__=='WGAN' or self.__MODEL__=='WGAN_GP' or self.__MODEL__=='CWGAN_GP':
            del self.critic
        make_keras_picklable()
        dump(self, path)

    @staticmethod
    def load(path):
        """
        ### Description:
        Loads a saved synthesizer from a pickle.

        ### Args:
        `path` (str): Path to read the synthesizer pickle from.
        """
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except (ValueError, RuntimeError):
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth


class ConditionalModel(BaseModel):

    @staticmethod
    def _validate_label_col(data: DataFrame, label_cols: List[str]):
        "Validates the label_col format, raises ValueError if invalid."
        assert all(item in list(data.columns) for item in label_cols), \
            f"The column {label_cols} could not be found on the provided dataset and cannot be used as condition."
        assert all(data[label_cols].isna().sum() == 0), \
            f"The provided {label_cols} contains NaN values, please impute or drop the respective records before proceeding."
        assert all([(is_float_dtype(data[col]) or is_integer_dtype(data[col])) for col in label_cols]), \
            f"The provided {label_cols} are expected to be integers or floats."
        unique_frac = data[label_cols].nunique() / len(data.index)
        assert all(unique_frac < 0.3), \
            f"The provided columns {label_cols} are not valid conditional columns due to high cardinality. Please revise your input."

    def _prep_fit(self, data: DataFrame, label_cols: List[str], num_cols: List[str], cat_cols: List[str]):
        """
            Validate and prepare the data for the training of a conditionalGAN architecture
        Args:
            data:
            label_cols:
            num_cols:
            cat_cols:
        Returns:
        """
        # Validating the label columns
        self._validate_label_col(data, label_cols)
        self._col_order = data.columns
        self.label_col = label_cols

        # Separating labels from the rest of the data to fit the data processor
        data, label = data[data.columns[~data.columns.isin(label_cols)]], data[label_cols].values

        BaseModel.fit(self, data, num_cols, cat_cols)
        return data, label

    def _generate_noise(self):
        "Gaussian noise for the generator input."
        while True:
            yield normal(size=self.noise_dim)

    def get_batch_noise(self):
        "Create a batch iterator for the generator gaussian noise input."
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=tf.dtypes.float32)
                                                .batch(self.batch_size)
                                                .repeat())

    def sample(self, condition: DataFrame) -> ndarray:
        """
            Method to generate synthetic samples from a conditional synth previsously trained.
        Args:
            condition (pandas.DataFrame): A dataframe with the shape (n_cols, nrows) where n_cols=number of columns used to condition the training
            n_samples (int): Number of synthetic samples to be generated

        Returns:
            sample (pandas.DataFrame): A dataframe with the generated synthetic records.
        """
        ##Validate here if the cond_vector=label_dim
        condition = condition.reset_index(drop=True)
        n_samples = len(condition)
        z_dist = random.uniform(shape=(n_samples, self.noise_dim))
        records = self.generator([z_dist, condition], training=False)
        data = self.processor.inverse_transform(array(records))
        data = concat([condition, data], axis=1)
        return data[self._col_order]