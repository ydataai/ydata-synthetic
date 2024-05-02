"Implements a GAN BaseModel synthesizer, not meant to be directly instantiated."
import abc
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, List, NamedTuple, Optional, Union

from joblib import dump, load
from numpy.random import normal
from pandas.api.types import is_float_dtype, is_integer_dtype
from pandas import DataFrame
from pandas import concat
from tensorflow import config as tfconfig
from tensorflow import data as tfdata
from tensorflow import random
from typeguard import typechecked

class ModelParameters(NamedTuple):
    batch_size: int
    lr: Union[float, List[float]]
    betas: Optional[List[float]]
    layers_dim: List[int]
    noise_dim: int
    n_cols: Optional[List[str]]
    seq_len: Optional[int]
    condition: Optional[Any]
    n_critic: int
    n_features: Optional[int]
    tau_gs: float
    generator_dims: List[int]
    critic_dims: List[int]
    l2_scale: float
    latent_dim: int
    gp_lambda: float
    pac: bool
    gamma: float
    tanh: bool

class TrainParameters:
    cache_prefix: str
    label_dim: Optional[int]
    epochs: int
    sample_interval: int
    labels: Optional[List[str]]
    n_clusters: int
    epsilon: float
    log_frequency: int
    measurement_cols: Optional[List[str]]
    sequence_length: Optional[int]
    number_sequences: int
    sample_length: int
    rounds: int

class BaseModel(abc.ABC):
    """
    Abstract class for synthetic data generation models.

    The main methods are train (for fitting the synthesizer), save/load and sample (generating synthetic records).

    """
    __MODEL__ = None

    @abc.abstractmethod
    def fit(self, data: Union[DataFrame, np.ndarray],
                  num_cols: Optional[List[str]] = None,
                  cat_cols: Optional[List[str]] = None):
        """
        Trains and fit a synthesizer model to a given input dataset.

        Args:
            data (Union[DataFrame, np.ndarray]): Training data
            num_cols (Optional[List[str]]) : List with the names of the categorical columns
            cat_cols (Optional[List[str]]): List of names of categorical columns

        Returns:
            self: Fitted synthesizer
        """
        ...

    @abc.abstractmethod
    def sample(self, n_samples: int) -> pd.DataFrame:
        ...

    @classmethod
    def load(cls, path: str):
        ...

    @abc.abstractmethod
    def save(self, path: str):
        ...

class BaseGANModel(BaseModel):
    """
    Base class of GAN synthesizer models.
    The main methods are train (for fitting the synthesizer), save/load and sample (obtain synthetic records).

    Args:
        model_parameters (ModelParameters):
            Set of architectural parameters for model definition.
    """
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
        self.model_parameters = model_parameters
        self.batch_size = self.model_parameters.batch_size
        self._set_lr(self.model_parameters.lr)
        self.beta_1 = self.model_parameters.betas[0]
        self.beta_2 = self.model_parameters.betas[1]
        self.noise_dim = self.model_parameters.noise_dim
        self.data_dim = None
        self.layers_dim = self.model_parameters.layers_dim

        # Additional parameters for the CTGAN
        self.generator_dims = self.model_parameters.generator_dims
        self.critic_dims = self.model_parameters.critic_dims
        self.l2_scale = self.model_parameters.l2_scale
        self.latent_dim = self.model_parameters.latent_dim
        self.gp_lambda = self.model_parameters.gp_lambda
        self.pac = self.model_parameters.pac

        self.use_tanh = self.model_parameters.tanh
        self.processor = None

    def _set_lr(self, lr):
        if isinstance(lr, float):
            self.g_lr = lr
            self.d_lr = lr
        elif isinstance(lr, (list, tuple)):
            assert len(lr) == 2, "Please provide the betas information as a tuple."
            self.g_lr = lr[0]
            self.d_lr = lr[1]

    def define_gan(self):
        """Define the trainable model components.

        Optionally validate model structure with mock inputs and initialize optimizers."""
        raise NotImplementedError

    def fit(self,
              data: Union[DataFrame, np.ndarray],
              num_cols: Optional[List[str]] = None,
              cat_cols: Optional[List[str]] = None,
              train_arguments: Optional[TrainParameters] = None) -> Union[DataFrame, np.ndarray]:
        if num_cols is None:
            num_cols = []
        if cat_cols is None:
            cat_cols = []
        if self.__MODEL__ in ['RegularGAN', 'CTGAN']:
            self.processor = RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols).fit(data)
        elif self.__MODEL__ == 'TimeSeriesGAN':
            self.processor = TimeSeriesDataProcessor(num_cols=num_cols, cat_cols=cat_cols).fit(data)
        elif self.__MODEL__ == 'CTGAN':
            n_clusters = train_arguments.n_clusters
            epsilon = train_arguments.epsilon
            self.processor = CTGANDataProcessor(n_clusters=n_clusters, epsilon=epsilon,
                                                num_cols=num_cols, cat_cols=cat_cols).fit(data)
        elif self.__MODEL__ == 'DoppelGANger':
            measurement_cols = train_arguments.measurement_cols
            sequence_length = train_arguments.sequence_length
            sample_length = train_arguments.sample_length
            self.processor = DoppelGANgerProcessor(num_cols=num_cols, cat_cols=cat_cols,
                                                   measurement_cols=measurement_cols,
                                                   sequence_length=sequence_length,
                                                   sample_length=sample_length,
                                                   normalize_tanh=self.use_tanh).fit(data)
        else:
            raise ValueError(f'A DataProcessor is not available for the {self.__MODEL__}.')

    def sample(self, n_samples: int) -> pd.DataFrame:
        assert n_samples > 0, "Please insert a value bigger than 0 for n_samples parameter."
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            z = random.uniform([self.batch_size, self.noise_dim], dtype=tf.dtypes.float32)
            records = self.generator(z, training=False).numpy()
            data.append(records)
        return self.processor.inverse_transform(np.vstack(data))

    def save(self, path: str):
        """
        Saves a synthesizer as a pickle.

        Args:
            path (str): Path to write the synthesizer as a pickle object.
        """
        # Save only the generator?
        if self.__MODEL__ == 'WGAN' or self.__MODEL__ == 'WGAN_GP' or self.__MODEL__ == 'CWGAN_GP':
            del self.critic
        make_keras_picklable()
        dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Loads a saved synthesizer from a pickle.

        Args:
            path (str): Path to read the synthesizer pickle from.
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
            data: training data
            label_cols: label columns
            num_cols: numerical columns
            cat_cols: categorical columns
        Returns:
            data, label: preprocessed data and labels
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

    def sample(self, condition: Union[DataFrame, np.ndarray]) -> np.ndarray:
        """
            Method to generate synthetic samples from a conditional synth previsously trained.
        Args:
            condition (Union[DataFrame, np.ndarray]): A dataframe or numpy array with the shape (n_cols, nrows) where n_cols=number of columns used to condition the training
            n_samples (int): Number of synthetic samples to be generated

        Returns:
            sample (np.ndarray): A numpy array with the generated synthetic records.
        """
        if not isinstance(condition, DataFrame) and not isinstance(condition, np.ndarray):
            raise ValueError("The condition argument should be a pandas DataFrame or a numpy array.")
        if condition.shape[0] != self.batch_size:
            raise ValueError("The number of rows in the condition argument should match the batch size.")
        if not isinstance(condition.values, np.ndarray):
            raise ValueError("The condition argument should be a pandas DataFrame or a numpy array.")
        condition = condition.reset_index(drop=True)
        n_samples = len(condition)
        z_dist = random.uniform(shape=(n_samples, self.noise_dim))
        records = self.generator([z_dist, condition], training=False)
        data = self.processor.inverse_transform(records)
        data = np.hstack((condition.values, data))
        return data
